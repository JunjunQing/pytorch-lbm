#!/bin/bash
# run_thesis_bigblack.sh
# Machine: bigblack (AMD R9 5900x + NVIDIA GTX 1080 8GB)
# Studies 3a/3b/5: parameter sweeps (41 cases, ~7.5h GPU)
#
# Standard CUDA setup for GTX 1080:
#   1. Install PyTorch with CUDA: pip install torch
#
# Usage: bash scripts/run_thesis_bigblack.sh

cd "$(dirname "$0")/.."

python3 -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')" 2>&1

python3 -u scripts/run_thesis_bigblack.py 2>&1 | tee results/bigblack_run.log
