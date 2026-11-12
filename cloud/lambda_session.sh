#!/bin/bash
# ======================================================================
#  lambda_session.sh -- the production session on Lambda 8x A100-SXM4-40GB
# ======================================================================
#  THE IDEA
#    8 GPUs = 4 prefill/decode PAIRS, each a full copy of the SPORC setup:
#        pair k:  prefill vLLM on GPU 2k, decode vLLM on GPU 2k+1,
#                 vLLM toy proxy, Jade admission proxy, one power sampler
#    The pairs run at the same time, so one session-hour buys four pair-hours.
#    Kitchen version: four identical two-cook kitchens in one building. We first
#    check that they do not slow each other down (shared CPU, memory, cooling),
#    then give each kitchen a different experiment.
#
#  STEPS -- run one at a time, look at the output, then run the next:
#      bash lambda_session.sh check          # 2 min: hardware, sudo, caps/clocks accepted on all 8 GPUs
#      bash lambda_session.sh up             # ~10 min: start all 4 pairs
#      bash lambda_session.sh interference   # ~5 min: pair 0 alone vs all 4 at once, same load
#      bash lambda_session.sh e1             # ~20 min: rate sweep on all 4 pairs (4 replicates)
#      bash lambda_session.sh levers  8 16   # root levers at the given rates (pick from E1)
#      bash lambda_session.sh admission 8 16 # shed / defer / ramped defer / drain, 4 replicates
#      bash lambda_session.sh rows           # datasheet rows + power tables for everything so far
#      bash lambda_session.sh down           # stop all, reset GPUs, tar the results
#    The servers keep running between steps (they are started in the
#    background), so run the steps inside tmux.
#
#  LEVERS STEP (one lever per pair, all at once):
#      pair 0: power cap on PREFILL GPU   ($CAPS, default 300 225 150 W)
#      pair 1: power cap on DECODE GPU    ($CAPS)
#      pair 2: clock lock on PREFILL GPU  ($CLOCKS, default 1100 810 MHz)
#      pair 3: clock lock on DECODE GPU   ($CLOCKS)
#
#  RESULTS: ~/jade_runs/session_<date>/pair<k>/<step>/   (~/jade_runs/current points to it)
#  NEEDS: lambda_dev_setup.sh run first on this VM.
# ======================================================================
set -uo pipefail
# Each held or in-flight request is an open socket; the default limit (1024) broke the
# 24 req/s defer events on 11 Nov. Raise it for everything started from this shell.
ulimit -n "$(ulimit -Hn)" 2>/dev/null || ulimit -n 65536 2>/dev/null || true

REPO=~/jade; WORK="$REPO/disagg"; DATA=~/jade_data
source ~/jade-venv/bin/activate
MODEL="${JADE_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
TRACE="$DATA/AzureLLMInferenceTrace_conv.csv"
NPAIRS="${NPAIRS:-4}"
CAPS="${CAPS:-300 225 150}"
CLOCKS="${CLOCKS:-1100 810}"
E1_RATES="${E1_RATES:-0.5 1 2 4 8 16 24 32}"
INTERF_RATE="${INTERF_RATE:-8}"

export CUDA_DEVICE_ORDER=PCI_BUS_ID      # CUDA index = NVML index = nvidia-smi index
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
# Same transport list as every SPORC run and the A6000 diagnosis (variant A).
# On this box the GPUs are joined by NVLink, so cuda_ipc rides NVLink, not PCIe.
export UCX_TLS="${UCX_TLS:-tcp,sm,cuda_copy,cuda_ipc}" UCX_NET_DEVICES=all UCX_PROTO_INFO=y
P_CFG='{"kv_connector":"NixlConnector","kv_role":"kv_producer","kv_load_failure_policy":"fail"}'
D_CFG='{"kv_connector":"NixlConnector","kv_role":"kv_consumer","kv_load_failure_policy":"fail"}'

# Ports for pair k: prefill 8100+10k, decode 8200+10k, admission 8300+10k,
# toy proxy 8400+10k, NIXL handshake 5600+10k (prefill) and 5601+10k (decode).
pport() { echo $((8100 + 10 * $1)); }; dport() { echo $((8200 + 10 * $1)); }
aport() { echo $((8300 + 10 * $1)); }; xport() { echo $((8400 + 10 * $1)); }

STEP="${1:-}"; shift || true
if [ "$STEP" = "up" ] || [ "$STEP" = "check" ]; then
  S=~/jade_runs/session_$(date +%Y%m%d_%H%M)
  [ "$STEP" = "up" ] && { mkdir -p "$S"; ln -sfn "$S" ~/jade_runs/current; }
else
  S=$(readlink -f ~/jade_runs/current) || { echo "!!! no session: run 'up' first"; exit 1; }
fi

wait_up() {    # wait_up <port> <pid> <log>: poll /health up to 15 min; stop early if the server died
  for i in $(seq 1 180); do
    curl -sf --max-time 5 "localhost:$1/health" >/dev/null && return 0
    kill -0 "$2" 2>/dev/null || { echo "!!! server on :$1 died -- first error in $3:"
      grep -n -m1 -A12 -E "Traceback|Error" "$3" | cut -c1-200; return 1; }
    sleep 5; done; echo "!!! :$1 not healthy after 15 min"; return 1; }

start_server() {   # start_server <pair> <prefill|decode>: one vLLM instance, detached
  local k=$1 role=$2 L="$S/pair$1"
  local gpu port side cfg
  if [ "$role" = prefill ]; then gpu=$((2*k)); port=$(pport $k); side=$((5600+10*k)); cfg=$P_CFG
  else gpu=$((2*k+1)); port=$(dport $k); side=$((5601+10*k)); cfg=$D_CFG; fi
  # setsid + nohup: the server keeps running after this script exits, so the
  # next step (a new invocation of this script) can use it.
  CUDA_VISIBLE_DEVICES=$gpu VLLM_NIXL_SIDE_CHANNEL_PORT=$side setsid nohup \
    vllm serve "$MODEL" --port $port --gpu-memory-utilization 0.6 --kv-transfer-config "$cfg" \
    > "$L/$role.log" 2>&1 < /dev/null &
  echo $! > "$L/$role.pid"
}

run_pairs() {      # run_pairs <cmd-template>: run it for every pair at once, wait for all
  # The template may use $k, $L (pair dir), $PP/$DP/$AP/$XP (ports), $PG/$DG (GPU indices).
  local pids=() k
  for k in $(seq 0 $((NPAIRS-1))); do
    ( L="$S/pair$k"; PP=$(pport $k); DP=$(dport $k); AP=$(aport $k); XP=$(xport $k)
      PG=$((2*k)); DG=$((2*k+1)); eval "$1" ) &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait $p; done
}

reset_gpus() {     # default power limit + unlocked clocks on every GPU
  for g in $(nvidia-smi --query-gpu=index --format=csv,noheader); do
    local d; d=$(nvidia-smi -i $g --query-gpu=power.default_limit --format=csv,noheader,nounits)
    sudo -n nvidia-smi -i $g -pl "${d%.*}" >/dev/null; sudo -n nvidia-smi -i $g -rgc >/dev/null
  done
}

case "$STEP" in
# ----------------------------------------------------------------------
check)
  echo ">>> $(hostname) $(date)"
  nvidia-smi --query-gpu=index,name,pci.bus_id,driver_version,power.limit,power.default_limit,power.min_limit,power.max_limit,clocks.max.sm --format=csv
  nvidia-smi topo -m          # NV12 etc. = NVLink between GPU pairs
  sudo -n true || { echo "!!! no passwordless sudo"; exit 1; }
  sudo -n nvidia-smi -pm 1 >/dev/null
  # Does every GPU ACCEPT a cap and a clock lock? (Whether they BIND under load
  # was proven on this GPU model on 21 Oct; the lever events re-measure it.)
  ok=1
  for g in $(nvidia-smi --query-gpu=index --format=csv,noheader); do
    sudo -n nvidia-smi -i $g -pl 200 >/dev/null && sudo -n nvidia-smi -i $g -lgc 810,810 >/dev/null || ok=0
    lim=$(nvidia-smi -i $g --query-gpu=power.limit --format=csv,noheader,nounits)
    echo "gpu $g: limit now ${lim} W (expect 200)"; [ "${lim%.*}" = 200 ] || ok=0
  done
  reset_gpus
  nvidia-smi --query-gpu=index,power.limit --format=csv,noheader | tr '\n' ' '; echo "<- back to default"
  [ $ok = 1 ] && echo "CHECK: PASS" || echo "CHECK: FAIL -- caps or clock locks not accepted"
  ;;
# ----------------------------------------------------------------------
up)
  echo ">>> session dir $S"
  nvidia-smi --query-gpu=index,name,power.limit --format=csv > "$S/gpus.csv"; nvidia-smi topo -m > "$S/topo.txt"
  python -c "import vllm; print('vllm', vllm.__version__)" | tee "$S/versions.txt"
  sudo -n nvidia-smi -pm 1 >/dev/null; reset_gpus
  for k in $(seq 0 $((NPAIRS-1))); do
    L="$S/pair$k"; mkdir -p "$L/admission"
    # One sampler per pair for the WHOLE session (--max-s 1 day). Every step
    # cuts its own time window out of this file.
    setsid nohup python "$WORK/jade_power_sampler.py" --gpus $((2*k)) $((2*k+1)) --roles prefill decode \
      --hz 20 --max-s 86400 --out "$L/power.csv" > "$L/sampler.log" 2>&1 < /dev/null &
    echo $! > "$L/sampler.pid"
  done
  # Start ONE server first, alone: its first start compiles the model (torch.compile,
  # CUDA graphs) and fills the shared caches. Seven servers compiling at once into
  # the same cache folders could trip over each other; after this they only read.
  start_server 0 prefill
  wait_up $(pport 0) $(cat "$S/pair0/prefill.pid") "$S/pair0/prefill.log" || exit 1
  echo ">>> pair0 prefill up (caches warm); starting the other 7"
  start_server 0 decode
  for k in $(seq 1 $((NPAIRS-1))); do start_server $k prefill; start_server $k decode; done
  for k in $(seq 0 $((NPAIRS-1))); do
    L="$S/pair$k"
    wait_up $(pport $k) $(cat "$L/prefill.pid") "$L/prefill.log" || exit 1
    wait_up $(dport $k) $(cat "$L/decode.pid") "$L/decode.log" || exit 1
    setsid nohup python "$DATA/toy_proxy_server.py" --port $(xport $k) \
      --prefiller-hosts localhost --prefiller-ports $(pport $k) \
      --decoder-hosts localhost --decoder-ports $(dport $k) > "$L/proxy.log" 2>&1 < /dev/null &
    echo $! > "$L/proxy.pid"
    setsid nohup python "$WORK/jade_admission_proxy.py" --port $(aport $k) --upstream localhost:$(xport $k) \
      --log-dir "$L/admission" > "$L/admission.log" 2>&1 < /dev/null &
    echo $! > "$L/admission.pid"
  done
  sleep 5
  # One real request through each pair's full path (admission -> toy proxy -> prefill -> KV -> decode).
  for k in $(seq 0 $((NPAIRS-1))); do
    code=$(curl -s -o /dev/null -w '%{http_code}' -H 'Content-Type: application/json' \
      -d "{\"model\":\"$MODEL\",\"prompt\":\"Explain why decode is memory bound.\",\"max_tokens\":16}" \
      localhost:$(aport $k)/v1/completions)
    echo "pair $k (GPUs $((2*k)),$((2*k+1))): HTTP $code"
  done
  echo ">>> up. Next: bash $0 interference"
  ;;
# ----------------------------------------------------------------------
interference)
  # Same load on pair 0, first alone, then with all 4 pairs loaded. If pair 0's
  # latency and power barely change, the pairs are independent and every later
  # step can run them in parallel. If not, later steps must run one pair at a time.
  R=$INTERF_RATE
  LG="python $WORK/jade_loadgen.py --model $MODEL --trace $TRACE --rates $R --dur-s 60 --tail-s 30 --gap-s 5 --idle-s 10"
  echo ">>> pair 0 alone at $R req/s"
  mkdir -p "$S/pair0/interf_alone"
  $LG --url localhost:$(xport 0) --out-dir "$S/pair0/interf_alone" > "$S/pair0/interf_alone/summary.txt"
  echo ">>> all $NPAIRS pairs at $R req/s"
  run_pairs 'mkdir -p $L/interf_all; '"$LG"' --url localhost:$XP --seed $((650+k)) --out-dir $L/interf_all > $L/interf_all/summary.txt'
  for d in "$S/pair0/interf_alone" "$S"/pair*/interf_all; do
    ln -sf ../power.csv "$d/e1_power.csv"
    echo "--- ${d#$S/}"; grep -E "^ +[0-9.]+ all" "$d/summary.txt"
    python "$WORK/jade_smoke_plot.py" --dir "$d" --prefix e1_ 2>/dev/null | grep -E "^rate_" || true
  done
  echo ">>> compare pair0/interf_alone with pair0/interf_all: TTFT p50, TPOT p50 and mean W should be within ~5-10 %."
  ;;
# ----------------------------------------------------------------------
e1)
  # E1 on the new hardware: power and latency vs load. Four pairs = four
  # replicates (different request draws: seed 650+k). Sets the rates for the
  # lever steps and decides which SPORC findings stand.
  run_pairs 'mkdir -p $L/e1; python '"$WORK"'/jade_loadgen.py --url localhost:$XP --model '"$MODEL"' --trace '"$TRACE"' --rates '"$E1_RATES"' --seed $((650+k)) --out-dir $L/e1 > $L/e1/summary.txt 2>&1'
  for k in $(seq 0 $((NPAIRS-1))); do
    d="$S/pair$k/e1"; ln -sf ../power.csv "$d/e1_power.csv"
    echo "=== pair $k"; grep -E "^ +[0-9.]+ all" "$d/summary.txt"
    python "$WORK/jade_smoke_plot.py" --dir "$d" --prefix e1_ > "$d/power_summary.txt" 2>&1
    grep -E "^rate_" "$d/power_summary.txt" || true
  done
  echo ">>> pick the lever rates: one where the pair is comfortable, one near the knee (where TTFT p99 starts climbing)."
  ;;
# ----------------------------------------------------------------------
levers)
  [ $# -ge 1 ] || { echo "usage: $0 levers <rate> [rate ...]"; exit 1; }
  RATES="$*"
  ev_for() {   # events for pair k
    case $1 in
      0) for c in $CAPS; do printf "cap:prefill:%s " $c; done ;;
      1) for c in $CAPS; do printf "cap:decode:%s " $c; done ;;
      2) for c in $CLOCKS; do printf "clock:prefill:%s " $c; done ;;
      3) for c in $CLOCKS; do printf "clock:decode:%s " $c; done ;;
    esac; }
  for k in $(seq 0 $((NPAIRS-1))); do echo "pair $k: $(ev_for $k)"; done
  # Different seeds per pair; jade_e3.py resets its GPUs to default in a finally block.
  run_pairs 'mkdir -p $L/levers; python '"$WORK"'/jade_e3.py --proxy localhost:$AP --model '"$MODEL"' --trace '"$TRACE"' --rates '"$RATES"' --events $(ev_for $k) --prefill-gpu $PG --decode-gpu $DG --seed $((650+k)) --out-dir $L/levers > $L/levers/console.txt 2>&1'
  reset_gpus
  for k in $(seq 0 $((NPAIRS-1))); do echo "=== pair $k"; grep -E "^  ev|!!!" "$S/pair$k/levers/console.txt"; done
  echo ">>> Next: bash $0 rows   (or admission)"
  ;;
# ----------------------------------------------------------------------
admission)
  [ $# -ge 1 ] || { echo "usage: $0 admission <rate> [rate ...]"; exit 1; }
  RATES="$*"
  # Same events and rates on all four pairs = four replicates per row. This is
  # the answer to the SPORC caveat that p99 rested on 13-83 requests per window.
  run_pairs 'mkdir -p $L/admission_ev; python '"$WORK"'/jade_e3.py --proxy localhost:$AP --model '"$MODEL"' --trace '"$TRACE"' --rates '"$RATES"' --events shed_reject shed_defer shed_defer_ramp drain --seed $((650+k)) --out-dir $L/admission_ev > $L/admission_ev/console.txt 2>&1'
  for k in $(seq 0 $((NPAIRS-1))); do echo "=== pair $k"; grep -E "^  ev|!!!" "$S/pair$k/admission_ev/console.txt"; done
  ;;
# ----------------------------------------------------------------------
rows)
  NOM=$(nvidia-smi -i 0 --query-gpu=power.default_limit --format=csv,noheader,nounits); NOM=${NOM%.*}
  for d in "$S"/pair*/levers "$S"/pair*/admission_ev; do
    [ -f "$d/e3_events.csv" ] || continue
    ln -sf ../power.csv "$d/e3_power.csv"
    k=$(basename "$(dirname "$d")")
    echo "=== ${d#$S/}"
    python "$WORK/jade_e3_rows.py" --dir "$d" --nominal-w "$NOM" --hardware-ref "lambda_a100_sxm4_8x_$k" | tee "$d/rows.txt"
  done
  ;;
# ----------------------------------------------------------------------
down)
  for f in "$S"/pair*/*.pid; do
    [ -f "$f" ] || continue
    pid=$(cat "$f"); kill -TERM -- -"$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null   # whole process group
  done
  sleep 10
  pkill -9 -u "$USER" -f "vllm serve" 2>/dev/null
  pkill -u "$USER" -f "toy_proxy_server.py|jade_admission_proxy.py|jade_power_sampler.py" 2>/dev/null
  reset_gpus
  nvidia-smi --query-gpu=index,power.limit,memory.used --format=csv,noheader
  cd ~/jade_runs && tar czf "$(basename "$S").tgz" --exclude='*.log' "$(basename "$S")" \
    && tar czf "$(basename "$S")_logs.tgz" "$(basename "$S")"/pair*/*.log
  ls -lh ~/jade_runs/*.tgz
  echo ">>> copy both .tgz files home, then TERMINATE the instance."
  ;;
*)
  sed -n '2,35p' "$0"; exit 1 ;;
esac
