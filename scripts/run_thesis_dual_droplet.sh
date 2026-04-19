#!/bin/bash
# run_thesis_dual_droplet.sh
# Study 9: Dual-droplet coalescence on curved surfaces
# Machine: bigblack (NVIDIA GTX 1080 8GB)
# 12 cases, ~3h

cd "$(dirname "$0")/.."

python3 -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0))" 2>&1

python3 -u scripts/run_thesis_dual_droplet.py 2>&1 | tee results/dual_droplet_run.log
