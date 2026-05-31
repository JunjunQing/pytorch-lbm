# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Summary

GPU-accelerated Allen-Cahn phase-field Lattice Boltzmann Method (LBM) simulation of droplet impact on curved surfaces. Implements Fakhari & Bolster (2017) pressure-velocity formulation for high density ratio (828:1) two-phase flows. The primary goal is reproducing Liu et al. (2015) experimental results on asymmetric droplet spreading over cylindrical ridges.

## Running Simulations

### Dependencies

```bash
# Use project virtual environment (avoids system package conflicts)
source .venv/bin/activate
pip install torch numpy matplotlib
# CPU-only PyTorch: pip install torch --extra-index-url https://download.pytorch.org/whl/cpu
```

No requirements.txt or setup.py — direct imports via `sys.path.insert`.

### GPU Options

#### Option 1: Remote Server (TencentCloud) — **Recommended for large simulations**

SSH alias: `2698BV3TencentCloud` (configured in `~/.ssh/config`)

```bash
# [Once per WSL session] Start persistent SSH master connection (reuses socket)
# This avoids re-authentication on every command. The master connection
# stays alive for 4 hours after the last session closes (ControlPersist).
ssh -Nf -o ServerAliveInterval=30 -o ServerAliveCountMax=3 2698BV3TencentCloud

# Check socket status
ls -la ~/.ssh/cm_root@1.13.160.139:60022

# All subsequent SSH commands reuse the same TCP connection (near-instant)
ssh 2698BV3TencentCloud "nvidia-smi"
ssh 2698BV3TencentCloud "cd /mnt/qmingjun/pytorch_lbm && python3 scripts/run_thesis_sweep.py"
```

**Remote Server Details:**
- **Host**: `1.13.160.139` (Tencent Cloud)
- **Port**: `60022`
- **User**: `root`
- **GPU**: NVIDIA CMP 30HX (6GB VRAM) — same as original thesis setup
- **Code dir**: `/mnt/qmingjun/pytorch_lbm/`
- **SSH Key**: `~/.ssh/Debian-Server`

#### Option 2: Local GPU (if available)

```bash
# GPU 1 — NVIDIA (Studies 0/1/2/7)
CUDA_VISIBLE_DEVICES=0 python3 scripts/run_thesis_sweep.py

# GPU 2 — NVIDIA GTX 1080 (Studies 3a/3b/5)
bash scripts/run_thesis_bigblack.sh

# GPU 3 — Intel Arc A750 XPU (Studies 4a/4b/6)
bash scripts/run_thesis_qingqing.sh
```

### Main sweep scripts (thesis parameter studies)

```bash
# Run on remote server
ssh 2698BV3TencentCloud "cd /mnt/qmingjun/pytorch_lbm && python3 scripts/run_thesis_sweep.py"

# Run locally (if GPU available)
CUDA_VISIBLE_DEVICES=0 python3 scripts/run_thesis_sweep.py
```

### Plotting

```bash
source .venv/bin/activate
python3 scripts/plot_thesis_results.py    # Data analysis figures
python3 scripts/plot_thesis_droplets.py   # Droplet morphology snapshots
```

**Note:** Plot scripts read result files at import time and expect `results/thesis_sweep_results.json` and `results/thesis_k_history.json` locally. Download from remote first:
```bash
scp 2698BV3TencentCloud:/mnt/qmingjun/pytorch_lbm/results/thesis_sweep_results.json results/
scp 2698BV3TencentCloud:/mnt/qmingjun/pytorch_lbm/results/thesis_k_history.json results/
```

### Key environment variables

- `CUDA_VISIBLE_DEVICES=0` for NVIDIA GPU selection
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to reduce fragmentation on 6GB VRAM
- `PYTHONUNBUFFERED=1` for real-time log output in nohup

## Architecture

### Core library (`lbm/`)

| Module | Role |
|--------|------|
| `fe_ac_solver.py` | **Main solver.** `AllenCahnSolver` — implements the full AC-LBM algorithm: collision, streaming, bounce-back, wetting BC, mass conservation. Single `step()` method advances one timestep. |
| `fe_config.py` | `FEConfig` dataclass — all simulation parameters (grid, physics, wetting). Auto-computes derived quantities (beta, kappa, M from sigma/xi). |
| `fe_droplet.py` | `create_fe_droplet_with_impact()` — initializes phi field with tanh interface profile and impact velocity weighted by phi. |
| `fe_wetting.py` | `ConningtonLeeWetting` — contact angle boundary condition with geometric gradient correction and curvature-dependent amplification (`geo_amplification`). |
| `fe_chemical_potential.py` | Chemical potential, density, and pressure computations. |
| `fe_measure.py` | Diagnostic measurements: film thickness, contact angles, spreading diameters (Dx via `measure_spread_factor`). |
| `lattice.py` | D2Q9 / D3Q19 lattice definitions (weights, velocities, opposite indices). |
| `boundary.py` | `BounceBack` — standard and volume-penalized bounce-back boundary conditions. |

### Key API nuances (verified)

| Intended API | Actual API |
|---|---|
| `VTKWriter` | `write_vtk` (function, not class) |
| `measure_spreading_diameter` | `measure_spread_factor` |
| `set_initial_condition(C, u, rho)` | `init_fields(phi_init, u_init)` (no rho) |
| `create_fe_droplet_with_impact(D0=..., U0=...)` | `create_fe_droplet_with_impact(radius=..., u_impact=(ux, uy))` |

### Solver timestep order (critical for correctness)

1. Recover phi from f-distribution, compute density and pressure
2. Compute gradients and chemical potential
3. Apply geometric wetting correction (if enabled)
4. Compute forces: surface tension, pressure correction, viscous correction
5. Recover velocity
6. Allen-Cahn source term (sharpening + convection)
7. Collision (f and g distributions)
8. Streaming (torch.roll)
9. Bounce-back (solid nodes)
10. Mass conservation scaling
11. Wall phi correction

### Geometry (`geometry/`)

- `substrate.py` — `create_substrate()` returns a boolean mask; `create_substrate_with_fraction()` returns a continuous solid fraction field for volume penalization. Supports: flat, ridge (half-cylinder), convex (hemisphere), concave (hemispherical cavity).

### I/O (`io_utils/`)

- `monitor.py` — `Monitor` class tracks film thickness, contact angles, mass, max velocity per step.
- `vtk_writer.py` — VTK output for ParaView visualization.

### Scripts (`scripts/`)

Scripts follow a pattern: define `run_case()` with all physical parameters, then loop over parameter combinations calling it. Results are saved as JSON files in `results/`. No test suite exists — validation is done by comparing simulation output against published experimental data.

## Key Technical Details

### GPU backends

- **NVIDIA CUDA**: `device='cuda'`, standard PyTorch CUDA
- **Intel XPU**: `device='xpu'`, requires `pip install torch --index-url https://download.pytorch.org/whl/xpu`
- **AMD DirectML**: legacy support, requires `torch-directml`

The solver handles non-standard devices in `AllenCahnSolver.__init__()` by falling back to raw device objects.

### Volume Penalization (VP)

Curved boundaries on Cartesian grids use a continuous solid fraction field (1 lattice unit smoothing) instead of staircase masks. Partial-solid nodes mix streamed and pre-stream distributions: `f_new = (1-eps)*f_streamed + eps*f_bounced`. Smooth normals are computed from the fraction gradient.

### Geometric amplification (`geo_amplification`)

The AC sharpening term resists contact angle corrections near curved walls. The `amp` parameter amplifies the geometric gradient to compensate. It requires calibration per resolution (amp ∝ N^0.84). Typical values: 1.5–3.4 for D/D₀ ≤ 1.2 at resolutions 80–150.

### Memory constraints

Target GPU has 6GB VRAM. 150×150×200 grid peaks at ~3.6GB with optimizations:
- Only clone solid-node distributions for bounce-back (not full field)
- Delete intermediate tensors immediately
- Pre-compute lattice weight views

Use `torch.float32` (not float64) for production runs.

## Parameter System

All physical parameters flow through `FEConfig`. Key relationships:

```
sigma = rho_l * U0^2 * D0 / We
beta  = 12 * sigma / xi
kappa = beta * xi^2 / 8
M     = 0.02 / beta
```

Standard v5 parameters: D0=45, tau=0.53, xi=5.0, U0=-0.05, We=7.9, theta=162°, rho_l/rho_g=828.

## Documentation

- `README.md` — Quick start and project overview
- `DEVELOPMENT.md` — Comprehensive development docs (math model, algorithm, validation results, 726 lines)
- `TASK_ASSIGNMENT.md` — Multi-GPU task distribution plan (131 cases across 3 machines)

## Known Issues

1. **`plot_thesis_results.py`** — `sorted()` failure on line 80 (None vs float comparison when data has missing `k_ridge` keys). First figure saved, rest fail.
2. **`plot_thesis_droplets.py`** — Syntax error at line 410 (`except ValueError:` without valid try block). Will not run.
