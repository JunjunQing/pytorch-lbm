#!/bin/bash
# run_local_77.sh — 本机 CMP 30HX 跑 Studies 3a/3b/5 + 4a/4b/6 (77 cases)
# Studies 9+10 由另一台机器运行

cd "$(dirname "$0")/.."

echo "============================================"
echo "  LOCAL: Studies 3a/3b/5 + 4a/4b/6 (77 cases)"
echo "  Started: $(date)"
echo "============================================"

python3 -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB')"

echo ""
echo "[1/3] Studies 3a/3b/5 (41 cases, ~7.5h)"
echo "============================================"
python3 -u scripts/run_thesis_bigblack.py 2>&1
if [ $? -ne 0 ]; then echo "ERROR in bigblack script"; exit 1; fi

echo ""
echo "[2/3] Studies 4a/4b + 6 (36 cases, ~6.6h)"
echo "============================================"
python3 -u scripts/run_thesis_qingqing_cuda.py 2>&1
if [ $? -ne 0 ]; then echo "ERROR in qingqing script"; exit 1; fi

echo ""
echo "============================================"
echo "  LOCAL DONE: $(date)"
echo "  Studies 9+10 请在另一台机器运行 run_thesis_9_10.py"
echo "============================================"
