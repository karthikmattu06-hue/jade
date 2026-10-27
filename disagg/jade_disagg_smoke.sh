#!/bin/bash -l
# ======================================================================
#  Jade -- disaggregated prefill/decode SMOKE TEST + E1 (v7) on 2 x A100
#          (go/no-go gate, two-pool power trace, then the E1 rate sweep)
# ======================================================================
#
#  WHAT THIS PROVES
#    That vLLM can run as TWO pools on TWO GPUs of one SPORC node -- a
#    "prefill" instance on GPU 0 and a "decode" instance on GPU 1 -- and hand
#    the KV cache of a request from one to the other. If one request goes
#    client -> proxy -> prefill -> (KV transfer) -> decode -> client and comes
#    back with text, the distributed claim in the DSCI-650 proposal is real.
#    If it cannot be made to work, Jade is a single-GPU study and NOT a good
#    fit for this course. That is why this runs before anything else.
#
#  WHAT v7 ADDS (19 Oct 2026) -- v6 kept as jade_disagg_smoke_v6.sh
#    E1 folded into the same allocation (one queue wait). If the smoke test
#    passes and the Azure trace is staged, the job restarts both servers
#    WITHOUT --enforce-eager (CUDA graphs on: eager mode inflates each decode
#    step ~4x -- 19 ms here vs ~4.3 ms in E0.1 -- which would distort the power
#    curve), then runs jade_loadgen.py: open-loop Poisson load at 0.5-16 req/s
#    through the proxy, Azure-trace request lengths, three equal tiers.
#    Output: e1_power.csv, e1_phases.csv, e1_requests.csv, e1_power_trace.png.
#    Node pin removed: pinned to skl-a-54 the job waited two weeks; unpinned it
#    starts in a day. Topology is recorded from `nvidia-smi topo -m` instead.
#    Set RUN_E1=0 at submit time to run the smoke test alone.
#
#  WHAT v6 ADDED -- v5 is kept as jade_disagg_smoke_v5.sh
#    1. Power on BOTH GPUs. jade_power_sampler.py logs watts, SM clock and the
#       power-cap throttle flag at 20 Hz to power.csv, from before the servers
#       start until the test ends. This is the first piece of the E1 harness.
#    2. A phased client (disagg_smoke_client.py v2): 30 s parked -> 20 requests
#       via the proxy -> 10 s gap -> 20 requests straight to the decode GPU ->
#       30 s parked. Request and phase times go to CSV on the same clock.
#    3. Pinned to skl-a-54 (NODE topology, the 21 Sep node) so the TTFT number
#       is comparable with 75.1 ms. CUDA_DEVICE_ORDER=PCI_BUS_ID so "GPU 0" means
#       the same card to CUDA (vLLM) and to NVML (the sampler).
#    4. jade_smoke_plot.py prints a phase x GPU power table into the .out file
#       and, if matplotlib exists in the env, saves power_trace.png.
#    Done when: TTFT within noise of 21 Sep AND the plot shows parked -> busy ->
#    parked on both pools, with only GPU 1 busy during the mono phase.
#
#  HOW THE PIECES FIT (three processes, one node)
#
#      client --HTTP--> toy_proxy (:8192)
#                          |  1) POST prompt, max_tokens=1  --> prefill vLLM (:8100, GPU 0)
#                          |        computes KV cache, keeps it in GPU memory
#                          |  2) POST same prompt + KV handle --> decode vLLM (:8200, GPU 1)
#                          |        pulls KV blocks from GPU 0 over NIXL, generates tokens
#                          <-- streamed tokens
#
#    "NIXL" is NVIDIA's transfer library; underneath it uses UCX, which picks a
#    transport at runtime: RDMA if there is an InfiniBand NIC, otherwise
#    CUDA-IPC / shared memory over PCIe on a single node. Our A100-PCIe pair has
#    no NVLink, so the transfer rides PCIe. Measuring what that costs is
#    literally experiment E2 -- but today we only need it to work.
#    (Older vLLM docs describe a NCCL-based connector; in 0.28 the supported
#    path is NixlConnector, so the proposal should say "over NIXL/UCX".)
#
#  BEFORE SUBMITTING (once, on the LOGIN node -- compute nodes have no internet)
#      source ~/jade/env.sh
#      pip install nixl                       # UCX + NIXL wheels
#      python -c "import nixl; print('nixl ok')"
#      mkdir -p ~/jade/disagg && cd ~/jade/disagg
#      curl -LO https://raw.githubusercontent.com/vllm-project/vllm/v0.28.0/tests/v1/kv_connector/nixl_integration/toy_proxy_server.py
#      ls -l toy_proxy_server.py          # REQUIRED: not in the pip wheel, and the
#                                         # compute node cannot download it itself.
#                                         # Pin the tag to your vLLM version.
#      # v6+ also needs jade_power_sampler.py and jade_smoke_plot.py in ~/jade/disagg
#      # v7 (E1) also needs jade_loadgen.py, and the Azure trace staged ONCE:
#      mkdir -p ~/jade/traces && cd ~/jade/traces && curl -LO \
#        https://raw.githubusercontent.com/Azure/AzurePublicDataset/master/data/AzureLLMInferenceTrace_conv.csv
#      sbatch ~/jade/disagg/jade_disagg_smoke.sh    # script, client and proxy all live in ~/jade/disagg
#
#  IF "pip install nixl" FAILS on RHEL9 (glibc / UCX wheel mismatch), do not
#  sink a day into it: run this script with CONNECTOR=example. That connector
#  writes the KV cache to node-local /tmp and the decoder reads it back. Slow,
#  but it proves the two-pool topology end to end, and it is a legitimate
#  fallback to name in the proposal's risk section.
#
# ======================================================================

#SBATCH --job-name=jade_disagg
#SBATCH --account=career
#SBATCH --partition=tier3       # tier3 has been scheduling faster than debug for this account (Sep 2026)
#SBATCH --time=0-00:45:00       # smoke ~12 min + restart ~5 min + E1 <=20 min
#SBATCH --cpus-per-task=12
#SBATCH --mem=64g
#SBATCH --gres=gpu:a100:2
##SBATCH --nodelist=skl-a-54   # disabled: pinning cost two weeks of queue (19 Oct). Topology logged below.
#SBATCH --output=/home/km5503/jade/jadedisagg_%j.out
#SBATCH --error=/home/km5503/jade/jadedisagg_%j.err

set -uo pipefail          # no -e: we want to reach the diagnostics even on failure

JADE_DIR=/home/km5503/jade
WORK="$JADE_DIR/disagg"
cd "$JADE_DIR"
source "$JADE_DIR/env.sh"

CONNECTOR="${CONNECTOR:-nixl}"                 # nixl (real transfer) | example (disk fallback)
MODEL="${JADE_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
PREFILL_PORT=8100; DECODE_PORT=8200; PROXY_PORT=8192
LOG="$WORK/logs_${SLURM_JOB_ID}"; mkdir -p "$LOG"
# CUDA numbers GPUs "fastest first"; NVML numbers them by PCI bus. On two identical
# A100s these usually agree, but "usually" is not good enough when the whole point
# is attributing watts to a pool. This makes CUDA use PCI order too, so
# CUDA_VISIBLE_DEVICES=0 (prefill) is the card the sampler calls GPU 0.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
RUN_E1="${RUN_E1:-1}"
TRACE="$JADE_DIR/traces/AzureLLMInferenceTrace_conv.csv"
if [ "$RUN_E1" = "1" ] && [ ! -f "$TRACE" ]; then
  echo "!!! E1 skipped: $TRACE not staged (see header). Smoke test still runs."; RUN_E1=0
fi
for f in jade_power_sampler.py disagg_smoke_client.py jade_smoke_plot.py jade_loadgen.py; do
  [ -f "$WORK/$f" ] || { echo "!!! missing $WORK/$f -- copy it from your Mac (~/Jade/disagg)"; exit 2; }
done

echo ">>> node: $(hostname)  job: $SLURM_JOB_ID  connector: $CONNECTOR  started: $(date)"
echo ">>> vllm: $(python -c 'import vllm,sys; print(vllm.__version__)' 2>&1)"
# Fail FAST if the connector's library is missing: vLLM would otherwise spend
# minutes loading weights and then die inside the connector with a misleading
# "Engine core initialization failed". Install on the login node first:
#   source ~/jade/env.sh && pip install nixl
if [ "$CONNECTOR" = "nixl" ]; then
  if python -c "import nixl" 2>/tmp/nixl_err_$SLURM_JOB_ID; then echo ">>> nixl: import ok"
  else
    echo "!!! nixl is not importable in this env -- NixlConnector cannot work."
    echo "!!! $(head -c 300 /tmp/nixl_err_$SLURM_JOB_ID)"
    echo "!!! Fix: on the login node, 'source ~/jade/env.sh && pip install nixl'."
    echo "!!! Or rerun with CONNECTOR=example (file-backed KV handoff, proves the topology)."
    exit 2
  fi
else
  echo ">>> connector=example: nixl not required"
fi
# The proxy is a vLLM *test helper* -- it is not part of the pip wheel, so it has
# to be downloaded once on the login node (compute nodes have no internet).
# Checked here rather than after startup: without it the two instances load
# weights for six minutes and then the request has nothing to talk to.
if [ ! -f "$WORK/toy_proxy_server.py" ]; then
  echo "!!! missing $WORK/toy_proxy_server.py -- the router between the two pools."
  echo "!!! Fix, on the LOGIN node (has internet; compute nodes do not):"
  echo "!!!   mkdir -p $WORK && cd $WORK && curl -LO https://raw.githubusercontent.com/vllm-project/vllm/v0.28.0/tests/v1/kv_connector/nixl_integration/toy_proxy_server.py"
  exit 2
fi
echo ">>> proxy script: present ($(wc -l < "$WORK/toy_proxy_server.py") lines)"
nvidia-smi --query-gpu=index,name,pci.bus_id,power.limit --format=csv
# How are the two cards connected? "PIX"/"PXB"/"PHB" = same PCIe switch/host
# bridge; "NODE" = same NUMA node; "SYS" = the two GPUs sit on DIFFERENT CPU
# sockets and the transfer crosses the inter-socket link (slowest).
# Observed so far: skl-a-52 = NODE, skl-a-54 = NODE, skl-a-51 = SYS, skl-a-37 = SYS.
# The nodes are NOT equivalent, so a KV-transfer number is only comparable to
# another measured on the same topology -- record the hostname with every run.
nvidia-smi topo -m 2>/dev/null | head -6

# ----------------------------------------------------------------------
#  Model staging (same reason as E0: CephFS is slow and noisy; use node /tmp)
# ----------------------------------------------------------------------
MODEL_DIR="models--$(echo "$MODEL" | sed 's|/|--|g')"
SRC="${HF_HOME:-$HOME/.cache/huggingface}/hub/$MODEL_DIR"
export HF_HOME="/tmp/jade_hf_${SLURM_JOB_ID}"; mkdir -p "$HF_HOME/hub"
cp -r "$SRC" "$HF_HOME/hub/"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export VLLM_CACHE_ROOT="$JADE_DIR/.vllm"

# ----------------------------------------------------------------------
#  Connector config. The JSON is passed verbatim to both instances; only the
#  role differs. kv_load_failure_policy=fail makes a broken transfer an error
#  instead of a silent recompute -- for a smoke test we WANT it to fail loudly.
# ----------------------------------------------------------------------
if [ "$CONNECTOR" = "nixl" ]; then
  P_CFG='{"kv_connector":"NixlConnector","kv_role":"kv_producer","kv_load_failure_policy":"fail"}'
  D_CFG='{"kv_connector":"NixlConnector","kv_role":"kv_consumer","kv_load_failure_policy":"fail"}'
  # Single node, no IB: tell UCX which transports exist here. cuda_ipc is the
  # GPU-to-GPU path over PCIe; sm/tcp are the CPU-side fallbacks.
  export UCX_TLS="${UCX_TLS:-tcp,sm,cuda_copy,cuda_ipc}"
  export UCX_NET_DEVICES=all
else
  KVDIR="/tmp/jade_kv_${SLURM_JOB_ID}"; mkdir -p "$KVDIR"
  P_CFG="{\"kv_connector\":\"ExampleConnector\",\"kv_role\":\"kv_producer\",\"kv_connector_extra_config\":{\"shared_storage_path\":\"$KVDIR\"}}"
  D_CFG="{\"kv_connector\":\"ExampleConnector\",\"kv_role\":\"kv_consumer\",\"kv_connector_extra_config\":{\"shared_storage_path\":\"$KVDIR\"}}"
fi

# ----------------------------------------------------------------------
#  Launch. Each vllm serve is pinned to one GPU with CUDA_VISIBLE_DEVICES.
#  --enforce-eager skips CUDA-graph capture: faster startup, fewer moving
#  parts. Remove it for the real E2 runs (graphs change power and latency).
#  --gpu-memory-utilization 0.6 leaves headroom; the 1.5B model needs ~3 GB.
#  The side-channel port is NIXL's handshake channel; must differ per instance.
# ----------------------------------------------------------------------
PORTS="$PREFILL_PORT $DECODE_PORT"
cleanup() { echo ">>> cleanup"
  # Ask nicely, wait <=30 s, then force. A half-started vLLM ignores SIGTERM, and a bare
  # `wait` then blocked until Slurm's time limit (both jobs on 23 Oct sat idle ~30 min).
  local p; p=$(jobs -p); [ -n "$p" ] && kill $p 2>/dev/null
  for i in $(seq 1 30); do [ -z "$(jobs -pr)" ] && break; sleep 1; done
  p=$(jobs -pr); [ -n "$p" ] && { echo ">>> force-killing $p"; kill -9 $p 2>/dev/null; }
  # vLLM's EngineCore runs as a grandchild, not a shell job: kill it by its port.
  for port in $PORTS; do pkill -9 -u "$USER" -f "vllm serve .*--port $port" 2>/dev/null; done
  rm -rf "$HF_HOME" "/tmp/jade_kv_${SLURM_JOB_ID}"; }
trap cleanup EXIT

# Power sampler FIRST, so the log also catches the parked->loading transition.
# It is a separate background process (NVML only, no CUDA context, ~1 % of a core).
# cleanup()'s kill sends it SIGTERM, which it treats as "flush and exit".
python "$WORK/jade_power_sampler.py" --gpus 0 1 --roles prefill decode --hz 20 \
  --out "$LOG/power.csv" > "$LOG/sampler.log" 2>&1 &
SAMPLER_PID=$!
sleep 2
if kill -0 $SAMPLER_PID 2>/dev/null; then cat "$LOG/sampler.log"
else echo "!!! power sampler died at startup:"; cat "$LOG/sampler.log"; exit 1; fi

# launch_pools <extra vllm flags> <log suffix>: starts both pools, remembers PIDs
# so E1 can stop exactly these two and restart them with different flags.
# $1 is deliberately unquoted: an empty string must expand to nothing.
launch_pools() {
  CUDA_VISIBLE_DEVICES=0 VLLM_NIXL_SIDE_CHANNEL_PORT=5600 \
    vllm serve "$MODEL" --port $PREFILL_PORT $1 --gpu-memory-utilization 0.6 \
    --kv-transfer-config "$P_CFG" > "$LOG/prefill$2.log" 2>&1 &
  P_PID=$!
  CUDA_VISIBLE_DEVICES=1 VLLM_NIXL_SIDE_CHANNEL_PORT=5601 \
    vllm serve "$MODEL" --port $DECODE_PORT $1 --gpu-memory-utilization 0.6 \
    --kv-transfer-config "$D_CFG" > "$LOG/decode$2.log" 2>&1 &
  D_PID=$!
}
launch_pools "--enforce-eager" ""

# Wait for both /health endpoints (up to 6 min; first start compiles nothing
# thanks to --enforce-eager, so this is mostly weight loading).
wait_up() { for i in $(seq 1 180); do curl -sf --max-time 5 "localhost:$1/health" >/dev/null && return 0; sleep 5; done; return 1; }   # 15 min (fresh nodes start slowly)
# On failure print the FIRST error in the log: vLLM's last lines are always the
# generic "Engine core initialization failed"; the real cause is earlier.
root_cause() { echo "--- first error in $1 ---"; grep -n -m1 -B3 -A15 -E "Traceback|Error|error:" "$1" | head -40; echo "--- last 8 lines ---"; tail -8 "$1"; }
wait_up $PREFILL_PORT && echo ">>> prefill instance up (GPU 0)" || { echo "!!! prefill instance failed -- $LOG/prefill.log"; root_cause "$LOG/prefill.log"; exit 1; }
wait_up $DECODE_PORT  && echo ">>> decode instance up (GPU 1)"  || { echo "!!! decode instance failed -- $LOG/decode.log";  root_cause "$LOG/decode.log";  exit 1; }

python "$WORK/toy_proxy_server.py" --port $PROXY_PORT \
  --prefiller-hosts localhost --prefiller-ports $PREFILL_PORT \
  --decoder-hosts localhost --decoder-ports $DECODE_PORT > "$LOG/proxy.log" 2>&1 &
# The proxy's own health endpoint is /healthcheck (the vLLM instances use
# /health). Poll it rather than sleeping: a dead proxy makes the test below
# return HTTP 000, which looks like a KV-transfer failure but is not one.
for i in $(seq 1 20); do curl -sf "localhost:$PROXY_PORT/healthcheck" >/dev/null && break; sleep 1; done
curl -sf "localhost:$PROXY_PORT/healthcheck" >/dev/null \
  && echo ">>> proxy up on :$PROXY_PORT" \
  || { echo "!!! proxy did not start -- $LOG/proxy.log"; tail -15 "$LOG/proxy.log"; exit 1; }

# ----------------------------------------------------------------------
#  THE TEST. One request through the proxy. A ~300-token prompt so prefill is
#  not trivial. Pass = HTTP 200 with non-empty text AND the prefill log shows it
#  served the request with max_tokens=1 (that is the proxy's prefill-only call).
# ----------------------------------------------------------------------
PROMPT=$(python - <<'EOF'
print(("Explain, in careful detail, why a GPU running the decode phase of a "
       "language model is limited by memory bandwidth rather than arithmetic. ")*6)
EOF
)
BODY=$(python -c "import json,sys; print(json.dumps({'model':'$MODEL','prompt':sys.argv[1],'max_tokens':64,'temperature':0}))" "$PROMPT")

echo ">>> sending 1 request through proxy :$PROXY_PORT"
T0=$(date +%s.%N)
RESP=$(curl -s -w '\n%{http_code}' -H 'Content-Type: application/json' -d "$BODY" "localhost:$PROXY_PORT/v1/completions")
T1=$(date +%s.%N)
CODE=$(echo "$RESP" | tail -1); JSON=$(echo "$RESP" | sed '$d')
TEXT=$(echo "$JSON" | python -c "import json,sys; d=json.load(sys.stdin); print(d['choices'][0]['text'][:200].replace('\n',' '))" 2>/dev/null || echo "")
# Evidence that BOTH pools served something. These greps are INFORMATIONAL only:
# vLLM's per-request log wording changes between releases, so a miss here means
# "the log phrasing differs", not "the handoff failed". (grep -c prints 0 and
# exits 1 on no match, hence the `|| true` rather than an extra echo.)
PREFILL_HITS=$( { grep -ciE "Added request|Received request|POST /v1/completions" "$LOG/prefill.log" || true; } 2>/dev/null | head -1)
DECODE_HITS=$(  { grep -ciE "Added request|Received request|POST /v1/completions" "$LOG/decode.log"  || true; } 2>/dev/null | head -1)
# Did the transfer machinery report trouble? With kv_load_failure_policy=fail a
# broken KV pull is an ERROR, not a silent recompute -- so an empty result here
# is what makes a 200 trustworthy as proof of a real handoff.
# Count only real error lines: ERROR level / Exception / Traceback that mention
# the transfer path. Two exclusions learned the hard way (job 21752531): vLLM
# echoes its config at startup, and that line contains
# "kv_load_failure_policy='fail'" -- a naive "kv...fail" match counted it as an
# error and reported FAIL on a working run. The import_utils WARNING traceback
# is a benign optional-import probe that appears on every start.
KV_ERRS=$( grep -E "ERROR|Exception|Traceback" "$LOG/decode.log" 2>/dev/null \
           | grep -iE "nixl|kv|transfer|ucx" \
           | grep -vE "non-default args|KVTransferConfig|import_utils" | wc -l )
# Positive evidence (NIXL only): the two agents exchanged metadata. Logged when
# the decoder first handshakes with the prefiller, i.e. on the first request.
HANDSHAKE=$( { grep -c "NIXL compatibility check passed" "$LOG/decode.log" || true; } 2>/dev/null | head -1)

echo "------------------------------------------------------------------"
echo "HTTP $CODE   wall $(python -c "print(f'{$T1-$T0:.2f}')") s"
echo "text: $TEXT"
echo "prefill log shows request activity : $PREFILL_HITS (informational)"
echo "decode  log shows request activity : $DECODE_HITS (informational)"
echo "NIXL handshake logged on decode    : $HANDSHAKE (expect 1 with connector=nixl)"
echo "KV transfer errors in decode log   : $KV_ERRS (must be 0)"
# PASS criterion. The proxy ALWAYS routes prefill-then-decode by construction,
# and kv_load_failure_policy=fail turns a broken KV pull into an error instead
# of a silent single-GPU recompute. So "200 with real text, no KV errors" is
# sufficient proof that the cache crossed between the two GPUs.
if [ "$CODE" = "200" ] && [ -n "$TEXT" ] && [ "$KV_ERRS" -eq 0 ]; then
  echo "RESULT: PASS -- two pools on two GPUs with KV handoff ($CONNECTOR). Jade is a go for DSCI-650."
else
  echo "RESULT: FAIL -- read $LOG/{prefill,decode,proxy}.log."
  [ "$CODE" = "000" ] && echo "       (HTTP 000 = nothing answered on :$PROXY_PORT -- a proxy problem, not a KV problem.)"
  [ "$KV_ERRS" -gt 0 ] && echo "       (KV errors present -- this IS a transfer problem. Try CONNECTOR=example, then fallback #2 Ray Serve.)"
  echo "--- proxy.log tail ---";   tail -20 "$LOG/proxy.log"
  echo "--- decode.log errors ---"; grep -iE "error|traceback|nixl|ucx" "$LOG/decode.log" | tail -20
fi
echo "------------------------------------------------------------------"

# ----------------------------------------------------------------------
#  PHASED RUN (only if the single request passed): the first E2 number AND
#  the first two-pool power trace in one go.
#    idle_pre 30 s -> disagg n=20 via proxy -> gap 10 s -> mono n=20 straight
#    to the decode GPU -> idle_post 30 s     (~2.5 min)
#  TTFT disagg minus mono = the handoff tax (21 Sep: +33.8 ms). The power CSV
#  over the same window shows what each pool draws parked vs busy.
# ----------------------------------------------------------------------
if [ "$CODE" = "200" ] && [ -n "$TEXT" ]; then
  python "$WORK/disagg_smoke_client.py" --model "$MODEL" --n 20 \
      --proxy "localhost:$PROXY_PORT" --mono "localhost:$DECODE_PORT" \
      --idle-s 30 --gap-s 10 --out-dir "$LOG" | tee "$LOG/ttft_compare.txt"
  # Stop the sampler explicitly (SIGTERM -> it flushes and closes power.csv)
  # BEFORE summarising, so the table reads a complete file.
  kill -TERM $SAMPLER_PID 2>/dev/null; wait $SAMPLER_PID 2>/dev/null
  cat "$LOG/sampler.log" | tail -1
  echo "------------------------------------------------------------------"
  python "$WORK/jade_smoke_plot.py" --dir "$LOG" | tee "$LOG/power_summary.txt"
  SMOKE_OK=1
fi

# ----------------------------------------------------------------------
#  E1 -- power and latency vs offered load (only if the smoke test passed)
# ----------------------------------------------------------------------
if [ "${SMOKE_OK:-0}" = "1" ] && [ "$RUN_E1" = "1" ]; then
  echo "=================================================================="
  echo ">>> E1: restarting both pools with CUDA graphs (no --enforce-eager)"
  # New sampler first, so the restart and graph capture are on the record too.
  python "$WORK/jade_power_sampler.py" --gpus 0 1 --roles prefill decode --hz 20 \
    --out "$LOG/e1_power.csv" > "$LOG/e1_sampler.log" 2>&1 &
  E1_SAMPLER_PID=$!
  kill -TERM $P_PID $D_PID 2>/dev/null; wait $P_PID $D_PID 2>/dev/null
  # vLLM's engine runs in a child process; wait until both GPUs have actually
  # released their memory, or the new servers fail with out-of-memory.
  for i in $(seq 1 30); do
    USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
    [ "$USED" -lt 1000 ] && break; sleep 2
  done
  echo ">>> GPU memory after stop: ${USED} MiB (max of the two)"
  launch_pools "" "_e1"
  wait_up $PREFILL_PORT && echo ">>> prefill up (graphs)" || { echo "!!! prefill restart failed"; root_cause "$LOG/prefill_e1.log"; exit 1; }
  wait_up $DECODE_PORT  && echo ">>> decode up (graphs)"  || { echo "!!! decode restart failed";  root_cause "$LOG/decode_e1.log";  exit 1; }
  # The proxy kept running: it only knows ports, and the ports did not change.
  python "$WORK/jade_loadgen.py" --url "localhost:$PROXY_PORT" --model "$MODEL" \
      --trace "$TRACE" --out-dir "$LOG" | tee "$LOG/e1_summary.txt"
  kill -TERM $E1_SAMPLER_PID 2>/dev/null; wait $E1_SAMPLER_PID 2>/dev/null
  tail -1 "$LOG/e1_sampler.log"
  python "$WORK/jade_smoke_plot.py" --dir "$LOG" --prefix e1_ | tee "$LOG/e1_power_summary.txt"
fi
echo ">>> logs: $LOG  (power.csv, phases.csv, requests.csv, power_trace.png if plotted)"
echo ">>> finished: $(date)"
