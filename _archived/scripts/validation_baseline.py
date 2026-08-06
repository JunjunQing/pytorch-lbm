#!/usr/bin/env python3
"""Validation baseline test: verify k=3.07 at R*=1.0, amp=1.8, xi=4.0"""
import sys, os
sys.path.insert(0, '/mnt/qmingjun/pytorch_lbm')

import torch
import numpy as np
from lbm.fe_config import FEConfig
from lbm.fe_ac_solver import AllenCahnSolver
from lbm.fe_droplet import create_fe_droplet_with_impact
from geometry.substrate import create_substrate_with_fraction

torch.cuda.empty_cache()

# --- Params from run_thesis_sweep.py lines 36-42 ---
D0, R_drop = 45.0, 22.5
rho_l, rho_g = 1.0, 1.0/828.0
xi, tau, U0 = 4.0, 0.53, -0.05
We = 7.9
sigma = rho_l * U0**2 * D0 / We
beta = 12.0 * sigma / xi
kappa = beta * xi**2 / 8.0
M = 0.02 / beta

nx = ny = 150
R_star = 1.0
R_g = abs(R_star) * R_drop
nz = min(int(R_g + 2 + R_drop + 2*R_drop + 15), 300)
print(f"Grid: {nx}x{ny}x{nz}")
print(f"sigma={sigma:.6f}, beta={beta:.6f}, kappa={kappa:.6f}")
print(f"M={M:.6e}, tau={tau}")

# --- Config ---
config = FEConfig(
    nx=nx, ny=ny, nz=nz,
    rho_l=rho_l, rho_g=rho_g,
    sigma=sigma, xi=xi, beta=beta, kappa=kappa, M=M,
    tau_l=tau, tau_g=tau, tau_h=tau+0.04,
    theta_eq=162.0, device='cuda',
    max_steps=3000, output_interval=3001,
    g_force=(0.0, 0.0, 0.0)
)

# --- Solver ---
solver = AllenCahnSolver(
    config, dtype=torch.float32,
    stab_mode='fakhari',
    boundary_relax=0.0,
    geometric_wetting=True,
    geo_amplification=1.8
)

# --- Substrate ---
solid, fraction = create_substrate_with_fraction(
    nx, ny, nz, substrate_type='ridge',
    R_star=R_star, R_d=R_drop)
solver.set_solid(solid, solid_fraction=fraction)

# --- Droplet placement ---
cx, cy = nx/2.0, ny/2.0
solid_np = solver.solid.cpu().numpy()
h = np.zeros((nx, ny), dtype=int)
for i in range(nx):
    for j in range(ny):
        for k in range(nz-1, -1, -1):
            if solid_np[i, j, k]:
                h[i, j] = k
                break
ridge_top = h[nx//2, ny//2]
gap = 2.0
cz = ridge_top + gap + R_drop
cz = min(cz, nz - R_drop - 2)
print(f"Ridge top at z={ridge_top}, droplet cz={cz:.1f}")

# --- Init ---
C_init, _, u_init = create_fe_droplet_with_impact(
    nx, ny, nz, center=(cx, cy, cz), radius=R_drop, xi=xi,
    rho_l=rho_l, rho_g=rho_g, u_impact=(0.0, 0.0, U0))
solver.init_fields(C_init, u_init)

# --- Run ---
max_k = 0.0
max_k_step = 0
Dx_peak = 0.0
Dy_peak = 0.0

for step in range(1, 3001):
    solver.step()
    if step % 50 == 0:
        phi_np = solver.phi.detach().cpu().numpy()
        solid_np = solver.solid.cpu().numpy()
        interface = (phi_np > 0.5) & ~solid_np
        if interface.any():
            coords = np.argwhere(interface)
            Dx = float(coords[:, 0].max() - coords[:, 0].min() + 1)
            Dy = float(coords[:, 1].max() - coords[:, 1].min() + 1)
            k_val = Dx / Dy if Dy > 0 else 0
            if k_val > max_k:
                max_k = k_val
                max_k_step = step
                Dx_peak = Dx
                Dy_peak = Dy
        if step % 500 == 0:
            print(f"  step={step}, max_k={max_k:.4f} (at step {max_k_step})")

print(f"\n=== BASELINE RESULT ===")
print(f"max_k = {max_k:.4f}")
print(f"Dx = {Dx_peak:.0f}, Dy = {Dy_peak:.0f}")
print(f"max_k_step = {max_k_step}")
print(f"Expected: k ≈ 3.07 (from thesis_sweep_results.json)")
print(f"Target:  k = 2.6 (Liu 2015 experiment)")
print(f"Current error: {abs(max_k - 2.6)/2.6*100:.1f}%")