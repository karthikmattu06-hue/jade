#!/bin/bash
# ======================================================================
#  lambda_preflight.sh -- the paid TEST HOUR on a Lambda A100 VM (~15 min of it)
# ======================================================================
#  Answers, before any real spending:
#    1. What hardware did we get?  (model, power-limit range, driver, NVLink/topology)
#    2. Does sudo work without a password?
#    3. Does a power cap ACTUALLY hold under load, and how fast does it bite?
#       -- on a compute-bound load (like prefill) and a memory-bound one (like decode)
#    4. Does an SM clock lock actually hold?
#    5. Does the second GPU behave the same? (quick compute probe only)
#  Everything goes into ~/preflight_<timestamp>/, tarred at the end for scp.
#
#  Run on the VM:   bash lambda_preflight.sh
#  Then TERMINATE the instance in the Lambda dashboard (see the checklist).
# ======================================================================
set -uo pipefail
OUT=~/preflight_$(date +%Y%m%d_%H%M); mkdir -p "$OUT"; cd "$OUT"
exec > >(tee "$OUT/preflight.log") 2>&1          # everything below is also logged

echo ">>> $(date)  host $(hostname)"
nvidia-smi --query-gpu=index,name,pci.bus_id,driver_version,power.limit,power.default_limit,power.min_limit,power.max_limit,clocks.max.sm --format=csv
nvidia-smi topo -m                                   # NV# = NVLink between the GPUs; SYS/NODE/PIX = PCIe paths
nvidia-smi nvlink --status -i 0 2>/dev/null | head -8
echo ">>> sudo without password:"; sudo -n true && echo yes || { echo "NO -- stop here, the probe cannot run"; exit 1; }
echo ">>> persistence mode on (keeps the driver loaded, so settings are not dropped between processes)"
sudo -n nvidia-smi -pm 1

python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'gpus', torch.cuda.device_count())" \
  || { echo "!!! torch missing -- install: pip install torch"; exit 1; }
python3 -c "import pynvml" 2>/dev/null || pip install -q nvidia-ml-py

P=~/lambda_cap_probe.py
python3 "$P" --gpu 0 --load compute --out "$OUT/gpu0_compute"
python3 "$P" --gpu 0 --load memory  --out "$OUT/gpu0_memory"
if [ "$(nvidia-smi -L | wc -l)" -ge 2 ]; then
  python3 "$P" --gpu 1 --load compute --caps 0.6 --out "$OUT/gpu1_compute"
fi

echo ">>> final state (must show default limits, no clock lock):"
nvidia-smi --query-gpu=index,power.limit,clocks.sm --format=csv
cd ~ && tar czf "$(basename "$OUT").tgz" "$(basename "$OUT")" && echo ">>> results: ~/$(basename "$OUT").tgz"
