#!/bin/bash -l
# ======================================================================
#  Jade -- E3-lite: grid events on disaggregated serving, 2 x A100 (SPORC)
# ======================================================================
#  WHAT IT RUNS
#    prefill vLLM (GPU 0) + decode vLLM (GPU 1), CUDA graphs on, NixlConnector
#    -> vLLM toy proxy (:8192) -> jade_admission_proxy.py (:8300, the lever)
#    -> jade_e3.py: at 2, 3, 4 req/s, three events each (shed flex reject,
#       shed flex defer, full drain); each event = 60 s baseline, 120 s hold,
#       60 s after. Power on both GPUs at 20 Hz throughout (jade_power_sampler.py).
#  ALSO: UCX transport logging (UCX_LOG_LEVEL=info) -- E1 hit a throughput knee at
#    4-8 req/s on a cross-socket node; the UCX lines in prefill.log/decode.log say
#    which transport the KV transfer actually used (cuda_ipc = GPU-to-GPU, or a
#    host-memory path). Topology is printed below; record it with the results.
#  RUNTIME ~45 min of events + ~5 min startup. Logs: ~/jade/disagg/logs_e3_<job>/
#
#  BEFORE SUBMITTING (login node), files needed in ~/jade/disagg:
#    jade_e3.sh jade_e3.py jade_loadgen.py jade_admission_proxy.py
#    jade_power_sampler.py jade_smoke_plot.py jade_e3_rows.py toy_proxy_server.py
#  and ~/jade/traces/AzureLLMInferenceTrace_conv.csv (staged for E1).
#    sbatch --test-only ~/jade/disagg/jade_e3.sh     # check the start estimate first
#    sbatch ~/jade/disagg/jade_e3.sh
# ======================================================================
#SBATCH --job-name=jade_e3
#SBATCH --account=career
#SBATCH --partition=tier3
#SBATCH --time=0-01:15:00       # ~8 min startup on a fresh node + ~45 min events + margin
#SBATCH --cpus-per-task=12
#SBATCH --mem=64g
#SBATCH --gres=gpu:a100:2
#SBATCH --output=/home/km5503/jade/jade_e3_%j.out
#SBATCH --error=/home/km5503/jade/jade_e3_%j.err

set -uo pipefail
JADE_DIR=/home/km5503/jade; WORK="$JADE_DIR/disagg"
cd "$JADE_DIR"; source "$JADE_DIR/env.sh"
MODEL="${JADE_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
TRACE="$JADE_DIR/traces/AzureLLMInferenceTrace_conv.csv"
RATES="${E3_RATES:-2 3 4}"
EVENTS="${E3_EVENTS:-shed_reject shed_defer drain}"   # order matters: a defer releases a backlog
GAP_S="${E3_GAP_S:-30}"                                # idle between events so that backlog clears
PREFILL_PORT=8100; DECODE_PORT=8200; PROXY_PORT=8192; ADM_PORT=8300
LOG="$WORK/logs_e3_${SLURM_JOB_ID}"; mkdir -p "$LOG"
export CUDA_DEVICE_ORDER=PCI_BUS_ID          # CUDA and NVML number the GPUs the same way

for f in jade_e3.py jade_e3_rows.py jade_loadgen.py jade_admission_proxy.py jade_power_sampler.py jade_smoke_plot.py toy_proxy_server.py; do
  [ -f "$WORK/$f" ] || { echo "!!! missing $WORK/$f"; exit 2; }
done
[ -f "$TRACE" ] || { echo "!!! missing $TRACE"; exit 2; }
python -c "import nixl" || { echo "!!! nixl not importable in this env"; exit 2; }

echo ">>> node: $(hostname)  job: $SLURM_JOB_ID  started: $(date)  rates: $RATES  events: $EVENTS  gap: ${GAP_S}s"
nvidia-smi --query-gpu=index,name,pci.bus_id,power.limit --format=csv
nvidia-smi topo -m 2>/dev/null | head -4      # GPU0-GPU1 cell: NODE = same socket, SYS = cross-socket

# Model staged to node-local /tmp (CephFS is slow and noisy), as in E0/E1.
MODEL_DIR="models--$(echo "$MODEL" | sed 's|/|--|g')"
SRC="${HF_HOME:-$HOME/.cache/huggingface}/hub/$MODEL_DIR"
export HF_HOME="/tmp/jade_hf_${SLURM_JOB_ID}"; mkdir -p "$HF_HOME/hub"; cp -r "$SRC" "$HF_HOME/hub/"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false VLLM_CACHE_ROOT="$JADE_DIR/.vllm"

P_CFG='{"kv_connector":"NixlConnector","kv_role":"kv_producer","kv_load_failure_policy":"fail"}'
D_CFG='{"kv_connector":"NixlConnector","kv_role":"kv_consumer","kv_load_failure_policy":"fail"}'
export UCX_TLS="${UCX_TLS:-tcp,sm,cuda_copy,cuda_ipc}" UCX_NET_DEVICES=all
export UCX_LOG_LEVEL=info                      # prints the transports UCX selects (new vs E1)

PORTS="$PREFILL_PORT $DECODE_PORT"
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

# Sampler first: records startup too. One CSV for the whole job.
python "$WORK/jade_power_sampler.py" --gpus 0 1 --roles prefill decode --hz 20 \
  --out "$LOG/e3_power.csv" > "$LOG/sampler.log" 2>&1 &
SAMPLER_PID=$!; sleep 2; cat "$LOG/sampler.log"

# Both pools, CUDA graphs ON (no --enforce-eager), as in E1.
CUDA_VISIBLE_DEVICES=0 VLLM_NIXL_SIDE_CHANNEL_PORT=5600 vllm serve "$MODEL" --port $PREFILL_PORT \
  --gpu-memory-utilization 0.6 --kv-transfer-config "$P_CFG" > "$LOG/prefill.log" 2>&1 &
P_PID=$!
CUDA_VISIBLE_DEVICES=1 VLLM_NIXL_SIDE_CHANNEL_PORT=5601 vllm serve "$MODEL" --port $DECODE_PORT \
  --gpu-memory-utilization 0.6 --kv-transfer-config "$D_CFG" > "$LOG/decode.log" 2>&1 &
D_PID=$!; T_LAUNCH=$(date +%s)
wait_up $PREFILL_PORT $P_PID && echo ">>> prefill up after $(( $(date +%s) - T_LAUNCH )) s" || { echo "!!! prefill failed"; tail -30 "$LOG/prefill.log"; exit 1; }
wait_up $DECODE_PORT  $D_PID && echo ">>> decode up after $(( $(date +%s) - T_LAUNCH )) s"  || { echo "!!! decode failed";  tail -30 "$LOG/decode.log";  exit 1; }

python "$WORK/toy_proxy_server.py" --port $PROXY_PORT \
  --prefiller-hosts localhost --prefiller-ports $PREFILL_PORT \
  --decoder-hosts localhost --decoder-ports $DECODE_PORT > "$LOG/proxy.log" 2>&1 &
for i in $(seq 1 20); do curl -sf --max-time 5 "localhost:$PROXY_PORT/healthcheck" >/dev/null && break; sleep 1; done

# The lever. Logs commands.csv and admissions.csv into $LOG.
python "$WORK/jade_admission_proxy.py" --port $ADM_PORT --upstream "localhost:$PROXY_PORT" \
  --log-dir "$LOG" > "$LOG/admission.log" 2>&1 &
for i in $(seq 1 20); do curl -sf --max-time 5 "localhost:$ADM_PORT/state" >/dev/null && break; sleep 1; done
curl -sf --max-time 5 "localhost:$ADM_PORT/state" >/dev/null && echo ">>> admission proxy up" \
  || { echo "!!! admission proxy failed"; cat "$LOG/admission.log"; exit 1; }

echo ">>> UCX transports chosen (first lines):"
grep -hiE "ucp_context|transport|tl_name|cuda_ipc" "$LOG/prefill.log" "$LOG/decode.log" | head -8

python "$WORK/jade_e3.py" --proxy "localhost:$ADM_PORT" --model "$MODEL" --trace "$TRACE" \
  --rates $RATES --events $EVENTS --gap-s $GAP_S --out-dir "$LOG" | tee "$LOG/e3_console.txt"

kill -TERM $SAMPLER_PID 2>/dev/null; wait $SAMPLER_PID 2>/dev/null; tail -1 "$LOG/sampler.log"
python "$WORK/jade_smoke_plot.py" --dir "$LOG" --prefix e3_ > "$LOG/e3_power_summary.txt" 2>&1
grep -E "_(base|hold|after) " "$LOG/e3_power_summary.txt" | head -60
echo ">>> datasheet rows (schema v0.3 definitions):"
python "$WORK/jade_e3_rows.py" --dir "$LOG" | tee "$LOG/e3_rows.txt"
echo ">>> UCX transport lines after load (KV path):"
grep -hiE "cuda_ipc|cuda_copy|tcp|transport" "$LOG/decode.log" | sort | uniq -c | sort -rn | head -8
echo ">>> logs: $LOG   finished: $(date)"
