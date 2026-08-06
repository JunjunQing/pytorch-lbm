#!/bin/bash
# Wrapper script to run sweep in background
cd /mnt/e/junjun/pytorch_lbm-main
source venv/bin/activate
python3 scripts/run_pending_sweep.py > results/sweep_background.log 2>&1
echo "Sweep completed at $(date)" >> results/sweep_background.log
