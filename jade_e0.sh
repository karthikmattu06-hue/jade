#!/bin/bash -l
# ======================================================================
#  Jade E0.1 -- energy probe launcher   (45 windows: 27 grid + 18 batch sweep)
# ======================================================================
#
#  TWO MODES. Edit the four lines marked "MODE:" and nothing else.
#
#    dev   : --partition=debug   --time=0-01:00:00   1.5B model, no --exclusive
#            Short queue wait. Use for every iteration.
#
#    final : --partition=tier3   --time=0-02:00:00   7B model, --exclusive on
#            Exclusive means nobody else's job shares the node, so CPU
#            contention cannot jitter your timings. It also means Slurm must
#            reserve all 36 cores before you start, which is what caused the
#            33-hour queue wait. Only pay that for the runs you will cite.
#
# ======================================================================

#SBATCH --job-name=jade_e0
#SBATCH --account=career
#SBATCH --partition=debug          # MODE: debug (dev) | tier3 (final)
#SBATCH --time=0-01:00:00          # MODE: 1h (1.5B dev) | 2h (7B final)
#SBATCH --cpus-per-task=6
#SBATCH --mem=36g
#SBATCH --gres=gpu:a100:1
##SBATCH --exclusive               # MODE: uncomment for final runs only
#SBATCH --output=/home/km5503/jade/jadee0_%j.out
#SBATCH --error=/home/km5503/jade/jadee0_%j.err

set -euo pipefail

JADE_DIR=/home/km5503/jade
cd "$JADE_DIR"

# ----------------------------------------------------------------------
#  Environment
# ----------------------------------------------------------------------
# env.sh activates the conda env and then, in this order, puts the conda lib
# directory ahead of the system one (so the loader finds conda's newer
# libstdc++ rather than the system's older one), points CC/CXX at the spack
# gcc by absolute path (RIT's "blindfold" wrapper hides the system compilers),
# and disables the FlashInfer sampler (it JIT-compiles CUDA kernels at runtime,
# which needs nvcc and would put a compiler inside our measurement window).
source "$JADE_DIR/env.sh"

# ----------------------------------------------------------------------
#  Version banner -- read this first in the .out file
# ----------------------------------------------------------------------
# Printed BEFORE python runs, so even a crash tells you which copy was on the
# cluster. If this does not say E0.1 (v3), the scp did not land and nothing
# below is worth reading.
echo ">>> probe file : $(ls -l "$JADE_DIR/energy_probe.py" | awk '{print $5" bytes, "$6" "$7" "$8}')"
echo ">>> probe says : $(grep -m1 '^PROBE_VERSION' "$JADE_DIR/energy_probe.py")"
echo ">>> node: $(hostname)   job: ${SLURM_JOB_ID}   cores: ${SLURM_CPUS_PER_TASK}   started: $(date)"

# ----------------------------------------------------------------------
#  Model staging
# ----------------------------------------------------------------------
# Your home directory is CephFS -- a network filesystem shared across the
# cluster. Loading 3 GB of weights over it is slow and, worse, its speed
# depends on what everyone else is doing, which is not a variable you want
# inside an energy experiment. So we copy the ONE model we need to the compute
# node's local /tmp first, then run fully offline. Copying only the one model
# (not the whole HF cache) is what took staging from 18 GB down to 2.9 GB.
MODEL="${JADE_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
MODEL_DIR="models--$(echo "$MODEL" | sed 's|/|--|g')"

SRC="${HF_HOME:-$HOME/.cache/huggingface}/hub/$MODEL_DIR"
DST="/tmp/jade_hf_${SLURM_JOB_ID}/hub"
mkdir -p "$DST"
echo ">>> staging $MODEL_DIR -> $DST  ($(du -sh "$SRC" | cut -f1))"
cp -r "$SRC" "$DST/"

export HF_HOME="/tmp/jade_hf_${SLURM_JOB_ID}"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

# torch.compile cache stays on CephFS so it survives between jobs -- that is
# what turns a 42-second compile into a cache hit on the second run.
export VLLM_CACHE_ROOT="$JADE_DIR/.vllm"

# ----------------------------------------------------------------------
#  Card identity and its power limits
# ----------------------------------------------------------------------
# power.limit is the cap the driver is enforcing; min_power_limit is the
# floor you could set it to with root. The gap between them is the entire
# budget available to a "cap the GPU" demand-response lever.
nvidia-smi --query-gpu=index,name,power.limit,power.min_limit,power.max_limit,persistence_mode,temperature.gpu,clocks.current.sm --format=csv

# ----------------------------------------------------------------------
#  Run
# ----------------------------------------------------------------------
OUT_PREFIX="$JADE_DIR/jade_e0_${SLURM_JOB_ID}"

python "$JADE_DIR/energy_probe.py" \
  --model "$MODEL" \
  --prompt-lens 128 512 2048 \
  --output-lens 1 64 256 \
  --batch 8 \
  --batch-sweep 1 2 4 8 16 32 \
  --sweep-prompt 128 \
  --sweep-output 256 \
  --reps 3 \
  --min-seconds 3.0 \
  --sample-hz 20 \
  --cool-to 45 \
  --cool-max-s 30 \
  --park-seconds 8 \
  --park-tolerance-w 20 \
  --out-prefix "$OUT_PREFIX"

echo ">>> finished: $(date)"
rm -rf "/tmp/jade_hf_${SLURM_JOB_ID}"
