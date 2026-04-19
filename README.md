# PyTorch-LBM: Allen-Cahn Phase-Field Lattice Boltzmann Method

GPU-accelerated lattice Boltzmann simulation of droplet impact on curved surfaces.

## Quick Start

```bash
pip install torch numpy matplotlib

# GPU 1 — Studies 0/1/2/7 (NVIDIA CMP 30HX 6GB, ~6h)
CUDA_VISIBLE_DEVICES=0 python3 scripts/run_thesis_sweep.py

# GPU 2 — Studies 3a/3b/5 (AMD RX 6750 GRE 10GB, ~7.5h)
bash scripts/run_thesis_bigblack.sh

# GPU 3 — Studies 4a/4b/6 (Intel Arc A750 8GB, ~6.6h)
bash scripts/run_thesis_qingqing.sh
```

## Current Parameters (v5, triple GPU)

| Parameter | Value | Description |
|-----------|-------|-------------|
| D0 | 45 | Droplet diameter (lattice units) |
| tau | 0.53 | Relaxation time |
| xi | 4.0 | Interface thickness |
| U0 | -0.05 | Impact velocity |
| Re | 225 | Reynolds number |
| Ma | 0.087 | Mach number |
| VP | ON | Volume Penalization sub-grid boundary |
| amp | 1.8 | Geometric amplification (ridge) |

## Project Structure

```
pytorch_lbm/
├── lbm/                    # Core LBM module
│   ├── fe_ac_solver.py     # Main AC-LBM solver (VP + geometric wetting)
│   ├── fe_config.py        # Configuration
│   ├── fe_droplet.py       # Droplet initialization
│   ├── fe_wetting.py       # Wetting boundary conditions
│   ├── fe_chemical_potential.py
│   ├── fe_measure.py
│   ├── boundary.py / lattice.py
├── geometry/               # Substrate geometry (flat/ridge/convex/concave)
│   └── substrate.py        # Boolean mask + VP fraction field
├── scripts/                # Simulation & plotting
│   ├── run_thesis_sweep.py       # GPU 1: Studies 0/1/2/7 (31 cases)
│   ├── run_thesis_bigblack.py    # GPU 2: Studies 3a/3b/5 (41 cases)
│   ├── run_thesis_bigblack.sh    # Shell wrapper (ROCm env vars)
│   ├── run_thesis_qingqing.py    # GPU 3: Studies 4a/4b/6 (36 cases)
│   ├── run_thesis_qingqing.sh    # Shell wrapper (XPU verification)
│   ├── plot_thesis_results.py    # Data analysis figures
│   └── plot_thesis_droplets.py   # Droplet morphology figures
├── results/                # Output data
│   ├── thesis_sweep_results.json
│   ├── thesis_k_history.json
│   └── thesis_figures/     # Generated figures (PNG)
└── io_utils/               # VTK writer, monitor
```

## Distributed Task Assignment

See [TASK_ASSIGNMENT.md](TASK_ASSIGNMENT.md) for the triple-GPU execution plan (108 cases, ~7.6h).

## GPU Setup

### NVIDIA CUDA (本机 30HX)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129
python3 -c "import torch; print(torch.cuda.is_available())"
```

### AMD DirectML (bigblack 6750 GRE)

```bash
# 1. Install PyTorch
pip install torch

# 2. Install DirectML
pip install torch-directml

# 3. Verify
python3 -c "import torch_directml; print(torch_directml.device())"
```

### Intel XPU (qingqing Arc A750)

```bash
# 1. Install Intel GPU drivers + oneAPI
# See: https://www.intel.com/content/www/us/en/docs/oneapi/installation-guide-linux

# 2. Install PyTorch XPU
pip install torch torchvision --index-url https://download.pytorch.org/whl/xpu

# 3. Verify
python3 -c "import torch; print(torch.xpu.is_available(), torch.xpu.get_device_name(0))"
```

## References

1. Liu et al. (2015) Phys. Fluids 27(12)
2. Fakhari & Bolster (2017) J. Comput. Phys. 334
3. Zhang et al. (2023) Phys. Rev. E — geometric wetting BC
4. Lee & Liu (2010) J. Comput. Phys. 229(20) — free-energy phase-field
