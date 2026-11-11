#!/bin/bash
# ======================================================================
#  lambda_dev_setup.sh -- one-time setup of a Lambda multi-GPU VM for Jade
# ======================================================================
#  WHAT IT DOES (about 10-15 min, most of it pip and the model download)
#    1. Prints the hardware facts we must record with every result.
#    2. Checks sudo (needed later for power caps and clock locks).
#    3. Clones the Jade repo and builds a Python environment with vLLM + NIXL.
#    4. Downloads the three things SPORC had staged by hand: the vLLM toy
#       proxy, the Azure request trace, and the model weights.
#
#  WHY A SCRIPT: the VM is deleted when you terminate it, so every session
#  starts from a blank disk. One command must rebuild everything, or you pay
#  $4-16/h to retype setup steps.
#
#  RUN (on the VM, after ssh):
#      curl -LO https://raw.githubusercontent.com/karthikmattu06-hue/jade/main/cloud/lambda_dev_setup.sh
#      bash lambda_dev_setup.sh 2>&1 | tee setup.log
#
#  SAME SCRIPT for both boxes (4x A6000 dev box, 8x A100 production box).
# ======================================================================
set -euo pipefail      # stop at the first failing step: a half-built env wastes paid time

# vLLM version. MUST match SPORC, or cloud and SPORC numbers are not comparable.
# Check on SPORC first:  source ~/jade/env.sh && pip show vllm | grep Version
VLLM_VERSION="${VLLM_VERSION:-0.28.0}"
MODEL="${JADE_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
REPO=~/jade            # the git clone (code)
VENV=~/jade-venv       # the Python environment
DATA=~/jade_data       # trace + proxy script (downloaded, not in git)

echo "=== 1. hardware ($(date), host $(hostname)) ==="
# power.min_limit / max_limit = the range a power cap can be set to on this card.
nvidia-smi --query-gpu=index,name,pci.bus_id,driver_version,power.limit,power.min_limit,power.max_limit --format=csv
# topo -m: how each GPU pair is connected.
#   NV# = NVLink (a direct GPU-to-GPU cable, fastest), PIX/PXB = same PCIe switch,
#   NODE = same CPU socket, SYS = different sockets (slowest path).
nvidia-smi topo -m
# topo -p2p r: can GPU i READ GPU j's memory directly (peer-to-peer)?
#   "OK" = yes. Anything else means a GPU-to-GPU copy must detour through
#   host RAM -- one candidate cause of the slow KV hand-off seen on SPORC.
nvidia-smi topo -p2p r || echo "(topo -p2p not supported by this driver)"

echo "=== 2. sudo ==="
sudo -n true && echo "passwordless sudo: yes" || { echo "!!! no passwordless sudo -- caps and clock locks will not work"; exit 1; }
# Persistence mode keeps the driver loaded between processes, so a power cap
# set by one command is still there when the next process starts.
sudo -n nvidia-smi -pm 1 >/dev/null

echo "=== 3. code + environment ==="
[ -d "$REPO/.git" ] && git -C "$REPO" pull --ff-only || git clone https://github.com/karthikmattu06-hue/jade.git "$REPO"
# A venv (virtual environment) is a private folder of Python packages, so vLLM
# and the torch it was built for do not clash with the image's own torch.
# PYTHON 3.12 IS REQUIRED: vLLM 0.28's flashinfer dependency uses syntax that
# crashes on 3.10 (found 10 Nov: "TypeError: 'type' object is not subscriptable").
# SPORC's Jade env is 3.12 too. Ubuntu 22.04 ships 3.10, so we use uv, an
# installer that downloads its own Python and is much faster than pip.
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
[ -x "$VENV/bin/python" ] && "$VENV/bin/python" -c 'import sys; sys.exit(sys.version_info[:2] != (3, 12))' \
  || { rm -rf "$VENV"; uv venv --python 3.12 "$VENV"; }
source "$VENV/bin/activate"
uv pip install -q "vllm==$VLLM_VERSION" nixl matplotlib   # nixl = the KV-transfer library (wraps UCX)
# flashinfer.comm is the import that failed on 3.10 -- test it here, not after a 5-minute server start.
python -c "import flashinfer.comm"
python -c "import vllm, nixl, pynvml, aiohttp; print('vllm', vllm.__version__, '| nixl, pynvml, aiohttp import ok')"

echo "=== 4. proxy script, trace, model ==="
mkdir -p "$DATA"
# The toy proxy is a vLLM TEST helper (routes prefill -> decode). It is not in
# the pip package, so it is fetched from the vLLM repo at the SAME version tag.
[ -f "$DATA/toy_proxy_server.py" ] || curl -fsSL -o "$DATA/toy_proxy_server.py" \
  "https://raw.githubusercontent.com/vllm-project/vllm/v${VLLM_VERSION}/tests/v1/kv_connector/nixl_integration/toy_proxy_server.py"
[ -f "$DATA/AzureLLMInferenceTrace_conv.csv" ] || curl -fsSL -o "$DATA/AzureLLMInferenceTrace_conv.csv" \
  "https://raw.githubusercontent.com/Azure/AzurePublicDataset/master/data/AzureLLMInferenceTrace_conv.csv"
# Download the weights now, so the first vLLM start is not also a download.
python -c "from huggingface_hub import snapshot_download; print('model at', snapshot_download('$MODEL'))"
ls -l "$DATA"

echo "=== done. Next:  bash $REPO/cloud/lambda_kv_diag.sh ==="
