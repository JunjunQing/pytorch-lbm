#!/bin/bash
# run_thesis_multi_droplet.sh
# Study 10: Multi-droplet sequential impact on curved surfaces
# Machine: qingqing (Intel Arc A750 8GB)
# 11 cases, ~2.8h

cd "$(dirname "$0")/.."

python3 -c "import torch; print('XPU:', torch.xpu.is_available(), torch.xpu.get_device_name(0) if torch.xpu.is_available() else 'N/A')" 2>&1

python3 -u scripts/run_thesis_multi_droplet.py 2>&1 | tee results/multi_droplet_run.log
