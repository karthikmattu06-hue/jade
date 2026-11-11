#!/bin/bash
# ======================================================================
#  lambda_kv_diag.sh -- WHY is the KV hand-off slow? (one prefill/decode pair)
# ======================================================================
#  THE QUESTION
#    On SPORC the hand-off of the KV cache from the prefill GPU to the decode
#    GPU moves ~0.28 GB/s, about 40x below what PCIe can carry. Something on
#    the path is slow. This script finds out WHICH PATH the bytes take.
#
#  BACKGROUND (kitchen version)
#    Prefill cook finishes the prep (the KV cache) and must pass it to the
#    decode cook. UCX is the runner who carries it, and UCX picks its own route:
#      cuda_ipc   hand it straight across the counter   (GPU memory -> GPU memory)
#      cuda_copy  put it down on a side table first     (GPU -> host RAM, or back)
#      sm         pass it through a hatch               (shared host memory between processes)
#      tcp        mail it through the post office       (network stack, even on one machine)
#    UCX_TLS is the list of routes the runner is ALLOWED to use. Our SPORC jobs
#    allowed all four, tcp included. The suspicion: UCX is mailing the cache.
#
#  WHAT IT RUNS -- the same test under two route lists ("variants"):
#      A  tcp,sm,cuda_copy,cuda_ipc   (what every SPORC run used = the control)
#      B  sm,cuda_copy,cuda_ipc       (tcp forbidden: forces a memory path)
#    For each variant:
#      1. start the pair (CUDA graphs on, as in E1/E3)
#      2. 20 requests via the proxy vs 20 straight to decode -> hand-off tax (ms)
#      3. a short load sweep at 4, 8, 16 req/s -> did the 4-8 req/s knee move?
#      4. save which transports UCX chose (UCX_PROTO_INFO) and vLLM's own
#         KV-transfer metrics
#
#  HOW TO READ THE RESULT (summary.txt at the end)
#    B's tax much lower than A's      -> tcp was the slow path. Fix = one env var.
#    B same as A                      -> not tcp; look at the transport lines for
#                                        cuda_copy (host-RAM detour) and at topo -p2p.
#    B fails with KV errors           -> cuda_ipc cannot connect the two processes
#                                        here; the transport lines say why.
#
#  RUN (after lambda_dev_setup.sh):   bash ~/jade/cloud/lambda_kv_diag.sh
#  Takes ~25-30 min (two server starts dominate). Pick GPUs with PREFILL_GPU / DECODE_GPU.
# ======================================================================
set -uo pipefail       # no -e: on a failure we still want the diagnostics below

REPO=~/jade; WORK="$REPO/disagg"; DATA=~/jade_data
source ~/jade-venv/bin/activate
MODEL="${JADE_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
PREFILL_GPU="${PREFILL_GPU:-0}"; DECODE_GPU="${DECODE_GPU:-1}"
PREFILL_PORT=8100; DECODE_PORT=8200; PROXY_PORT=8192
RATES="${DIAG_RATES:-4 8 16}"
OUT=~/jade_runs/kvdiag_$(date +%Y%m%d_%H%M); mkdir -p "$OUT"
exec > >(tee "$OUT/kvdiag.log") 2>&1                 # everything printed is also saved

# Same GPU numbering for CUDA (vLLM) and NVML (nvidia-smi, the sampler): PCI bus order.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false  # weights are already on disk
P_CFG='{"kv_connector":"NixlConnector","kv_role":"kv_producer","kv_load_failure_policy":"fail"}'
D_CFG='{"kv_connector":"NixlConnector","kv_role":"kv_consumer","kv_load_failure_policy":"fail"}'
# kv_load_failure_policy=fail: a broken transfer is an ERROR, not a silent
# recompute on the decode GPU. Without it a "working" run could be hiding
# the fact that no cache was transferred at all.

echo ">>> host $(hostname)  $(date)  prefill=GPU$PREFILL_GPU decode=GPU$DECODE_GPU  out=$OUT"
nvidia-smi --query-gpu=index,name,pci.bus_id,power.limit --format=csv
nvidia-smi topo -m | tee "$OUT/topo_m.txt"
nvidia-smi topo -p2p r | tee "$OUT/topo_p2p.txt" || true

stop_all() {            # stop servers + proxy, then wait until the GPUs are really empty
  local p; p=$(jobs -pr); [ -n "$p" ] && kill $p 2>/dev/null
  sleep 5
  # vLLM's engine is a grandchild process, not a shell job: find it by its port.
  for port in $PREFILL_PORT $DECODE_PORT; do pkill -9 -u "$USER" -f "vllm serve .*--port $port" 2>/dev/null; done
  pkill -u "$USER" -f toy_proxy_server.py 2>/dev/null
  for i in $(seq 1 30); do
    USED=$(nvidia-smi -i "$PREFILL_GPU,$DECODE_GPU" --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
    [ "$USED" -lt 1000 ] && break; sleep 2
  done
}
trap stop_all EXIT
# wait_up <port> <pid>: poll the server's /health page; give up if the process died.
wait_up() { for i in $(seq 1 180); do
    curl -sf --max-time 5 "localhost:$1/health" >/dev/null && return 0
    kill -0 "$2" 2>/dev/null || { echo "!!! server on :$1 exited during startup"; return 1; }
    sleep 5; done; return 1; }

run_variant() {         # $1 = label (A/B), $2 = UCX_TLS list
  local V="$1" TLS="$2" L="$OUT/$1"; mkdir -p "$L"
  echo "=================================================================="
  echo ">>> variant $V: UCX_TLS=$TLS"
  # These four variables are read by UCX inside each vLLM process:
  #   UCX_TLS           allowed transports (the route list above)
  #   UCX_PROTO_INFO=y  print a table of which transport is chosen per message size
  #   UCX_LOG_LEVEL     'info' also prints the devices UCX found
  export UCX_TLS="$TLS" UCX_NET_DEVICES=all UCX_PROTO_INFO=y UCX_LOG_LEVEL=info

  CUDA_VISIBLE_DEVICES=$PREFILL_GPU VLLM_NIXL_SIDE_CHANNEL_PORT=5600 vllm serve "$MODEL" --port $PREFILL_PORT \
    --gpu-memory-utilization 0.6 --kv-transfer-config "$P_CFG" > "$L/prefill.log" 2>&1 &
  local P_PID=$!
  CUDA_VISIBLE_DEVICES=$DECODE_GPU VLLM_NIXL_SIDE_CHANNEL_PORT=5601 vllm serve "$MODEL" --port $DECODE_PORT \
    --gpu-memory-utilization 0.6 --kv-transfer-config "$D_CFG" > "$L/decode.log" 2>&1 &
  local D_PID=$!
  wait_up $PREFILL_PORT $P_PID && wait_up $DECODE_PORT $D_PID \
    || { echo "!!! variant $V: servers did not start"; tail -25 "$L/prefill.log" "$L/decode.log"; stop_all; return 1; }
  python "$DATA/toy_proxy_server.py" --port $PROXY_PORT \
    --prefiller-hosts localhost --prefiller-ports $PREFILL_PORT \
    --decoder-hosts localhost --decoder-ports $DECODE_PORT > "$L/proxy.log" 2>&1 &
  for i in $(seq 1 20); do curl -sf --max-time 5 "localhost:$PROXY_PORT/healthcheck" >/dev/null && break; sleep 1; done

  # (2) hand-off tax: same prompt via the proxy (prefill -> transfer -> decode)
  #     and straight to the decode GPU (no transfer). The TTFT difference is the tax.
  python "$WORK/disagg_smoke_client.py" --model "$MODEL" --n 20 \
    --proxy "localhost:$PROXY_PORT" --mono "localhost:$DECODE_PORT" \
    --idle-s 5 --gap-s 5 --out-dir "$L" | tee "$L/ttft_compare.txt"

  # (3) short sweep: on SPORC the pair saturated between 4 and 8 req/s.
  python "$WORK/jade_loadgen.py" --url "localhost:$PROXY_PORT" --model "$MODEL" \
    --trace "$DATA/AzureLLMInferenceTrace_conv.csv" --rates $RATES \
    --dur-s 30 --tail-s 45 --gap-s 10 --idle-s 5 --out-dir "$L" | tee "$L/sweep.txt"

  # (4) evidence. /metrics is vLLM's built-in counters page (Prometheus format).
  curl -s "localhost:$DECODE_PORT/metrics" | grep -iE "nixl|kv_" | grep -v '^#' > "$L/kv_metrics.txt"
  # Which transports did UCX name? Count the mentions in both logs.
  grep -ohiE "cuda_ipc|cuda_copy|tcp|sysv|posix|cma|knem|xpmem" "$L/prefill.log" "$L/decode.log" \
    | tr 'A-Z' 'a-z' | sort | uniq -c > "$L/transports_count.txt"
  # The protocol table itself (the lines UCX_PROTO_INFO adds).
  grep -hE "rendezvous|zero-copy|copy-in|copy-out|multi-frag|\| *(tagged|stream|active|remote)" \
    "$L/decode.log" "$L/prefill.log" | sort -u | head -60 > "$L/ucx_proto.txt"
  local ERRS; ERRS=$(grep -E "ERROR|Exception|Traceback" "$L/decode.log" | grep -iE "nixl|kv|transfer|ucx" \
    | grep -vE "non-default args|KVTransferConfig|import_utils" | wc -l)

  { echo "---- variant $V  UCX_TLS=$TLS ----"
    grep -E "TTFT median|KV-handoff tax" "$L/ttft_compare.txt"
    echo "KV transfer errors in decode log: $ERRS (must be 0 for the numbers to count)"
    echo "transport mentions:"; cat "$L/transports_count.txt"
    echo "kv metrics (first lines):"; head -8 "$L/kv_metrics.txt"; echo; } >> "$OUT/summary.txt"
  stop_all
}

run_variant A "tcp,sm,cuda_copy,cuda_ipc"
run_variant B "sm,cuda_copy,cuda_ipc"

echo "=================================================================="
cat "$OUT/summary.txt"
echo ">>> sweep results per variant: $OUT/{A,B}/sweep.txt   UCX tables: $OUT/{A,B}/ucx_proto.txt"
cd ~/jade_runs && tar czf "$(basename "$OUT").tgz" "$(basename "$OUT")" && echo ">>> to copy home: ~/jade_runs/$(basename "$OUT").tgz"
echo ">>> finished $(date).  TERMINATE the instance in the Lambda dashboard when done."
