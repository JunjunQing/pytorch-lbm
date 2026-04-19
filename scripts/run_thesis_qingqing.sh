#!/bin/bash
# run_thesis_qingqing.sh
# Machine: qingqing (Intel Arc A750 8GB)
# Studies 4a/4b/6: contact angle sweep + phase diagram (36 cases, ~6.6h GPU)
#
# Intel Arc XPU setup:
#   pip install torch torchvision --index-url https://download.pytorch.org/whl/xpu
#
# Usage: bash scripts/run_thesis_qingqing.sh

cd "$(dirname "$0")/.."

python3 -c "import torch; print('XPU:', torch.xpu.is_available(), torch.xpu.get_device_name(0) if torch.xpu.is_available() else 'N/A')" 2>&1

python3 -u scripts/run_thesis_qingqing.py 2>&1 | tee results/qingqing_run.log
