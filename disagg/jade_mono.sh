#!/bin/bash -l
# ======================================================================
#  Jade -- monolithic baseline (B1, single GPU): E1's exact sweep, no P/D split
# ======================================================================
#  WHY
#    E1 (job 21928556, disaggregated, 2 GPUs) hit a throughput knee between 4 and
#    8 req/s while both GPUs had headroom. Running the SAME load against ONE vLLM
#    server on ONE GPU (prefill and decode share the card -- "monolithic") tells us:
#      1. If one GPU sustains 8 req/s fine, the knee came from the disaggregated
#         handoff (NIXL/UCX across sockets, or the toy proxy), not the model.
#      2. Baseline B1 for H2: per-GPU power vs rate without the split, so E3's
#         "how much can be shed" can be compared with and without disaggregation.
#    The proposal's B1 is tensor-parallel on 2 GPUs; this 1-GPU version queues
#    faster (1 GPU) and answers both questions above. TP=2 can follow.
#  WHAT IS IDENTICAL TO E1: model, vLLM version, CUDA graphs on, gpu-memory 0.6,
#    jade_loadgen.py with the same defaults and seed (same arrival schedule and
#    request lengths), 20 Hz sampler, plot/summary script.
#  WHAT DIFFERS: one server, no kv-transfer-config, no proxy (load goes straight in).
#  RUNTIME ~4 min startup + ~10-17 min sweep. Logs: ~/jade/disagg/logs_mono_<job>/
#
#  Needs in ~/jade/disagg: jade_mono.sh jade_loadgen.py jade_power_sampler.py jade_smoke_plot.py
#  and the Azure trace in ~/jade/traces.
#    sbatch --test-only ~/jade/disagg/jade_mono.sh ; sbatch ~/jade/disagg/jade_mono.sh
# ======================================================================
#SBATCH --job-name=jade_mono
#SBATCH --account=career
#SBATCH --partition=tier3
#SBATCH --time=0-00:45:00       # startup can take ~8 min on a fresh node
#SBATCH --cpus-per-task=8
#SBATCH --mem=32g
#SBATCH --gres=gpu:a100:1
#SBATCH --output=/home/km5503/jade/jade_mono_%j.out
#SBATCH --error=/home/km5503/jade/jade_mono_%j.err

set -uo pipefail
JADE_DIR=/home/km5503/jade; WORK="$JADE_DIR/disagg"
cd "$JADE_DIR"; source "$JADE_DIR/env.sh"
MODEL="${JADE_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
TRACE="$JADE_DIR/traces/AzureLLMInferenceTrace_conv.csv"
PORT=8200
LOG="$WORK/logs_mono_${SLURM_JOB_ID}"; mkdir -p "$LOG"
export CUDA_DEVICE_ORDER=PCI_BUS_ID

for f in jade_loadgen.py jade_power_sampler.py jade_smoke_plot.py; do
  [ -f "$WORK/$f" ] || { echo "!!! missing $WORK/$f"; exit 2; }
done
[ -f "$TRACE" ] || { echo "!!! missing $TRACE"; exit 2; }
echo ">>> node: $(hostname)  job: $SLURM_JOB_ID  started: $(date)"
nvidia-smi --query-gpu=index,name,pci.bus_id,power.limit --format=csv

MODEL_DIR="models--$(echo "$MODEL" | sed 's|/|--|g')"
SRC="${HF_HOME:-$HOME/.cache/huggingface}/hub/$MODEL_DIR"
export HF_HOME="/tmp/jade_hf_${SLURM_JOB_ID}"; mkdir -p "$HF_HOME/hub"; cp -r "$SRC" "$HF_HOME/hub/"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false VLLM_CACHE_ROOT="$JADE_DIR/.vllm"

PORTS="$PORT"
cleanup() { echo ">>> cleanup"
  # Ask nicely, wait <=30 s, then force. A half-started vLLM ignores SIGTERM, and a bare
  # `wait` then blocked until Slurm's time limit (both jobs on 23 Oct sat idle ~30 min).
  local p; p=$(jobs -p); [ -n "$p" ] && kill $p 2>/dev/null
  for i in $(seq 1 30); do [ -z "$(jobs -pr)" ] && break; sleep 1; done
  p=$(jobs -pr); [ -n "$p" ] && { echo ">>> force-killing $p"; kill -9 $p 2>/dev/null; }
  # vLLM's EngineCore runs as a grandchild, not a shell job: kill it by its port.
  for port in $PORTS; do pkill -9 -u "$USER" -f "vllm serve .*--port $port" 2>/dev/null; done
  rm -rf "$HF_HOME"; }
# wait_up <port> <server pid>: poll /health for up to 15 min (was 6 min -- on fresh nodes
# skl-a-53/55 just importing vLLM took ~4 min and startup ~7.5 min, so the old limit
# gave up on a healthy server, 23 Oct). Stops early if the server process has died.
# curl --max-time stops a half-started server from hanging the poll itself.
wait_up() { for i in $(seq 1 180); do
    curl -sf --max-time 5 "localhost:$1/health" >/dev/null && return 0
    kill -0 "$2" 2>/dev/null || { echo "!!! server on :$1 exited during startup"; return 1; }
    sleep 5; done; echo "!!! server on :$1 not healthy after 15 min"; return 1; }
trap cleanup EXIT

python "$WORK/jade_power_sampler.py" --gpus 0 --roles mono --hz 20 \
  --out "$LOG/e1_power.csv" > "$LOG/sampler.log" 2>&1 &
SAMPLER_PID=$!; sleep 2; cat "$LOG/sampler.log"

# One server doing both phases on GPU 0. Same flags as E1's pools minus the connector.
CUDA_VISIBLE_DEVICES=0 vllm serve "$MODEL" --port $PORT --gpu-memory-utilization 0.6 > "$LOG/mono.log" 2>&1 &
SRV_PID=$!; T_LAUNCH=$(date +%s)
wait_up $PORT $SRV_PID && echo ">>> mono server up after $(( $(date +%s) - T_LAUNCH )) s" \
  || { echo "!!! server failed"; tail -30 "$LOG/mono.log"; exit 1; }

# Same generator, defaults and seed as E1 -> same arrivals and request lengths.
python "$WORK/jade_loadgen.py" --url "localhost:$PORT" --model "$MODEL" --trace "$TRACE" \
  --out-dir "$LOG" | tee "$LOG/e1_summary.txt"

kill -TERM $SAMPLER_PID 2>/dev/null; wait $SAMPLER_PID 2>/dev/null; tail -1 "$LOG/sampler.log"
python "$WORK/jade_smoke_plot.py" --dir "$LOG" --prefix e1_ | tee "$LOG/e1_power_summary.txt"
echo ">>> vLLM queue at load (Running/Waiting):"
grep -oE "Running: [0-9]+ reqs, Waiting: [0-9]+ reqs" "$LOG/mono.log" | tail -25 | tr '\n' ';'; echo
echo ">>> logs: $LOG   finished: $(date)"
