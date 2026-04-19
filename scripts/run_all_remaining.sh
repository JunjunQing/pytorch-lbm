#!/bin/bash
# run_all_remaining.sh
# Run all remaining 100 cases on local CMP 30HX
# Order: Studies 3a/3b → 5 → 4a/4b → 6 → 9 → 10

cd "$(dirname "$0")/.."

echo "============================================"
echo "  MASTER: All remaining cases on CMP 30HX"
echo "  Started: $(date)"
echo "============================================"

python3 -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB')"

echo ""
echo "[1/6] Studies 3a/3b/5: We sweep + R* sweep (41 cases, ~7.5h)"
echo "============================================"
python3 -u scripts/run_thesis_bigblack.py 2>&1
if [ $? -ne 0 ]; then echo "ERROR in bigblack script"; exit 1; fi

echo ""
echo "[2/6] Studies 4a/4b: Contact angle sweep (24 cases, ~4.3h)"
echo "============================================"
python3 -u scripts/run_thesis_qingqing_cuda.py 2>&1
if [ $? -ne 0 ]; then echo "ERROR in qingqing script"; exit 1; fi

echo ""
echo "[3/6] Study 6: We-theta phase diagram (12 cases, ~2.1h)"
echo "  (included in qingqing_cuda script)"
echo ""

echo "[4/6] Study 9: Dual-droplet coalescence (12 cases, ~3h)"
echo "============================================"
python3 -u scripts/run_thesis_dual_droplet.py 2>&1
if [ $? -ne 0 ]; then echo "ERROR in dual_droplet script (possible OOM)"; fi

echo ""
echo "[5/6] Study 10: Multi-droplet sequential (11 cases, ~2h)"
echo "============================================"
python3 -u scripts/run_thesis_multi_droplet_cuda.py 2>&1
if [ $? -ne 0 ]; then echo "ERROR in multi_droplet script"; fi

echo ""
echo "============================================"
echo "  ALL DONE: $(date)"
echo "============================================"
