#!/usr/bin/env python3
"""Test LBM film drainage with enlarged gap and sharper interface."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from lbm.fe_config import FEConfig
from lbm.fe_ac_solver import AllenCahnSolver
from lbm.fe_droplet import create_fe_droplet_with_impact
from geometry.substrate import create_substrate_with_fraction
from io_utils.monitor import Monitor
import time

torch.cuda.empty_cache()

# === Modified parameters for film capture ===
D0 = 45.0
R_drop = D0 / 2.0
rho_l = 1.0
rho_g = 1.0 / 828.0
xi = 3.0           # sharper interface (was 4.0)
tau = 0.53
U0 = -0.05
We = 7.9
theta_eq = 162.0
amp = 1.8
gap_val = 20.0     # enlarged gap (was 2.0)
n_base = 150

sigma = rho_l * U0**2 * D0 / We
beta = 12.0 * sigma / xi
kappa = beta * xi**2 / 8.0
M = 0.02 / beta

R_star = 1.0
R_g = abs(R_star) * R_drop
nx, ny = n_base, n_base
nz = int(R_g + gap_val + R_drop + 2*R_drop + 20)
nz = min(nz, 300)

print(f'Grid: {nx}x{ny}x{nz}, xi={xi}, gap={gap_val}')
print(f'Interface span: ~{2.5*xi:.0f} l.u., Approx air: {gap_val - 2*2.5*xi:.0f} l.u.')

config = FEConfig(
    nx=nx, ny=ny, nz=nz,
    rho_l=rho_l, rho_g=rho_g,
    sigma=sigma, xi=xi, beta=beta, kappa=kappa,
    M=M, tau_l=tau, tau_g=tau, tau_h=tau+0.04,
    theta_eq=theta_eq, device='cuda',
    max_steps=5000, output_interval=5001,
    g_force=(0.0, 0.0, 0.0), dx=1e-6,
)

solver = AllenCahnSolver(config, dtype=torch.float32,
    stab_mode='fakhari', boundary_relax=0.0,
    geometric_wetting=True, geo_amplification=amp)

solid, fraction = create_substrate_with_fraction(
    nx, ny, nz, substrate_type='concave', R_star=R_star, R_d=R_drop)
solver.set_solid(solid, solid_fraction=fraction)

surface_height = np.zeros((nx, ny), dtype=int)
for i in range(nx):
    for j in range(ny):
        for k in range(nz-1, -1, -1):
            if solid[i, j, k]:
                surface_height[i, j] = k
                break

cx, cy = nx/2.0, ny/2.0
ridge_top = int(surface_height.max())
cz = ridge_top + gap_val + R_drop
print(f'Substrate top: {ridge_top}, Drop center: {cz}')

C_init, _, u_init = create_fe_droplet_with_impact(
    nx, ny, nz, center=(cx, cy, cz), radius=R_drop, xi=xi,
    rho_l=rho_l, rho_g=rho_g, u_impact=(0.0, 0.0, U0))
solver.init_fields(C_init, u_init)

monitor = Monitor(config)
t0 = time.time()
print('Running 3000 steps...')
for step in range(1, 3001):
    solver.step()
    if step % 30 == 0:
        monitor.record(solver, step=step)

elapsed = time.time() - t0
films = [f for f in monitor.history['film_thickness'] if f != float('inf')]
print(f'\nDone in {elapsed:.0f}s, Records: {len(films)}')
if films:
    print(f'Film range: {min(films)*1e6:.1f} - {max(films)*1e6:.1f} um')
    print(f'First 5: ["{"\", \"".join(f"{f*1e6:.1f}" for f in films[:5])}"]')
    print(f'Last 5:  ["{"\", \"".join(f"{f*1e6:.1f}" for f in films[-5:])}"]')

del solver
torch.cuda.empty_cache()
