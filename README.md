# PyTorch-LBM: Allen-Cahn Phase-Field Lattice Boltzmann Method

GPU-accelerated lattice Boltzmann simulation of droplet impact on curved surfaces with geometric amplification for wetting boundary conditions.

## Features

- Allen-Cahn phase-field LBM at 828:1 density ratio
- Geometric amplification for accurate wetting on curved surfaces
- Volume penalization for curved solid boundaries
- Support for ridge, convex hemisphere, and concave cavity geometries
- CUDA-accelerated via PyTorch

## Installation

```bash
# Clone repository
git clone https://github.com/JunjunQing/pytorch-lbm.git
cd pytorch-lbm

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install numpy matplotlib

# Verify GPU
python3 -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Quick Start

```bash
# Run parameter sweep
python3 scripts/run_thesis_sweep.py

# Generate paper figures
python3 scripts/paper_figures.py

# Run comprehensive validation
python3 scripts/run_validation.py
```

## Usage Example

```python
import sys
sys.path.insert(0, '.')
from lbm.fe_ac_solver import AllenCahnSolver
from lbm.fe_config import FEConfig
from lbm.fe_droplet import create_fe_droplet_with_impact
from geometry.substrate import create_substrate_with_fraction

# Configure simulation
config = FEConfig(
    nx=150, ny=150, nz=107,
    rho_l=1.0, rho_g=1.0/828.0,
    sigma=0.01424, xi=4.0,
    tau_l=0.53, tau_g=0.53,
    theta_eq=162.0, device='cuda',
    max_steps=2000,
)

# Create solver with geometric amplification
solver = AllenCahnSolver(
    config, geo_amplification=1.5,
    geometric_wetting=True,
)

# Set up geometry
solid, fraction = create_substrate_with_fraction(
    150, 150, 107, substrate_type='ridge',
    R_star=1.0, R_d=22.5)
solver.set_solid(solid, solid_fraction=fraction)

# Initialize and run
phi_init, u_init = create_fe_droplet_with_impact(
    150, 150, 107, center=(75, 75, 80),
    radius=22.5, xi=4.0, u_impact=(0, 0, -0.05))
solver.init_fields(phi_init, u_init)

for step in range(2000):
    solver.step()
```

## Project Structure

```
pytorch-lbm/
├── lbm/                    # Core LBM module
│   ├── fe_ac_solver.py     # Allen-Cahn solver
│   ├── fe_config.py        # Configuration dataclass
│   ├── fe_droplet.py       # Droplet initialization
│   ├── fe_wetting.py       # Wetting BC with geo_amplification
│   └── fe_measure.py       # Diagnostics
├── geometry/               # Substrate geometry
│   └── substrate.py        # ridge/convex/concave/flat
├── scripts/                # Simulation scripts
│   ├── paper_figures.py    # Generate paper figures
│   ├── run_validation.py   # Comprehensive validation
│   └── run_overnight_sweep.py  # Full parameter sweep
├── paper/                  # LaTeX source
│   ├── main.tex
│   └── sections/
└── results/                # Simulation data (JSON)
```

## Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| D₀ | 45 lu | Droplet diameter |
| ξ | 4 lu | Interface thickness |
| ρ_l/ρ_g | 828:1 | Density ratio |
| τ_g | 0.53 | Hydrodynamic relaxation |
| θ_eq | 90°--162° | Contact angle |
| α_geo | 1.5 | Geometric amplification |

## Citation

If you use this code, please cite:

```bibtex
@article{qing2026geometric,
  title={Geometric Amplification for Phase-Field LBM on Curved Surfaces},
  author={Qing, Mingjun},
  year={2026}
}
```

## License

MIT License
