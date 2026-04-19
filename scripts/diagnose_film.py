#!/usr/bin/env python3
"""Diagnose why film thickness measures 2μm instead of expected ~20μm."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from lbm.fe_config import FEConfig
from lbm.fe_ac_solver import AllenCahnSolver
from lbm.fe_droplet import create_fe_droplet_with_impact
from geometry.substrate import create_substrate, create_substrate_with_fraction

torch.cuda.empty_cache()

# Same params as sweep
D0 = 45.0
R_drop = D0 / 2.0
rho_l = 1.0
rho_g = 1.0 / 828.0
xi = 3.0
tau = 0.53
U0 = -0.05
We = 7.9
gap = 20.0
n_base = 150

sigma = rho_l * U0**2 * D0 / We
beta = 12.0 * sigma / xi
kappa = beta * xi**2 / 8.0
M = 0.02 / beta

# Test with R*=1.0 concave (the problematic case)
R_star = 1.0
substrate_type = 'concave'

R_g = abs(R_star) * R_drop
nx, ny = n_base, n_base
nz = int(R_g + gap + R_drop + 2*R_drop + 20)
nz = min(nz, 350)

print(f"=== Diagnostic: {substrate_type} R*={R_star} ===")
print(f"Grid: {nx}x{ny}x{nz}, R_g={R_g:.1f}, gap={gap}")

# Create substrate
solid, fraction = create_substrate_with_fraction(
    nx, ny, nz, substrate_type=substrate_type,
    R_star=R_star, R_d=R_drop)

# Also create boolean substrate (without fraction) for comparison
solid_bool = create_substrate(nx, ny, nz, substrate_type=substrate_type,
                              R_star=R_star, R_d=R_drop)

# Surface heights
def get_surface_height(solid, nx, ny, nz):
    height = np.zeros((nx, ny), dtype=int)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz - 1, -1, -1):
                if solid[i, j, k]:
                    height[i, j] = k
                    break
    return height

surf_frac = get_surface_height(solid, nx, ny, nz)
surf_bool = get_surface_height(solid_bool, nx, ny, nz)

mid = nx // 2
print(f"\nSurface height comparison at y={mid}:")
print(f"  Fraction-based solid: center={surf_frac[mid,mid]}, max={surf_frac.max()}")
print(f"  Boolean solid:        center={surf_bool[mid,mid]}, max={surf_bool.max()}")

# Show profile along x at y=mid
print(f"\n  x-index | r_from_center | surf_frac | surf_bool")
for di in range(-30, 31, 5):
    i = mid + di
    if 0 <= i < nx:
        r = abs(i - mid)
        print(f"  {i:5d}   | {r:5d}          | {surf_frac[i,mid]:5d}      | {surf_bool[i,mid]:5d}")

# Droplet placement
ridge_top = int(surf_frac.max()) if substrate_type == 'concave' else surf_frac[mid, mid]
cz = ridge_top + gap + R_drop
max_cz = nz - R_drop - 2
if cz > max_cz:
    cz = max_cz

print(f"\nDroplet placement:")
print(f"  ridge_top={ridge_top}, cz={cz:.1f}, max_cz={max_cz:.1f}")
print(f"  Droplet bottom at center: cz - R_drop = {cz - R_drop:.1f}")
print(f"  Expected gap at center: {(cz - R_drop) - surf_frac[mid,mid]:.1f} lattice units")

# Create solver and initialize
config = FEConfig(
    nx=nx, ny=ny, nz=nz,
    rho_l=rho_l, rho_g=rho_g,
    sigma=sigma, xi=xi, beta=beta, kappa=kappa,
    M=M, tau_l=tau, tau_g=tau, tau_h=tau+0.04,
    theta_eq=162.0, device='cuda',
    max_steps=100, output_interval=101,
    g_force=(0.0, 0.0, 0.0), dx=1e-6,
)

solver = AllenCahnSolver(config, dtype=torch.float32,
    stab_mode='fakhari', boundary_relax=0.0,
    geometric_wetting=True, geo_amplification=1.8)

solver.set_solid(solid, solid_fraction=fraction)

C_init, _, u_init = create_fe_droplet_with_impact(
    nx, ny, nz, center=(mid, mid, cz), radius=R_drop, xi=xi,
    rho_l=rho_l, rho_g=rho_g, u_impact=(0.0, 0.0, U0))

solver.init_fields(C_init, u_init)

# Check density profile at center column (x=mid, y=mid)
rho_init = solver.rho.cpu().numpy()
phi_init = solver.phi.cpu().numpy()
print(f"\nDensity/phi profile at center column (x={mid}, y={mid}):")
print(f"  k  | rho      | phi      | solid_frac")
for k in range(max(0, ridge_top - 5), min(nz, int(cz + R_drop + 10))):
    rho_val = rho_init[mid, mid, k]
    phi_val = phi_init[mid, mid, k]
    frac_val = fraction[mid, mid, k]
    marker = " <-- SUBLIMATE" if frac_val > 0.5 else ""
    marker += " <-- LIQUID" if rho_val > 0.5 else ""
    print(f"  {k:3d} | {rho_val:.6f} | {phi_val:.6f} | {frac_val:.4f}{marker}")

# Check film thickness manually
rho_threshold = (rho_l + rho_g) / 2
print(f"\nrho_threshold = {rho_threshold:.6f}")

# Find first liquid above substrate at center column
solid_center = solid[mid, mid, :]
sub_top_center = np.where(solid_center)[0].max() if solid_center.any() else 0
for k in range(sub_top_center + 1, nz):
    if rho_init[mid, mid, k] > rho_threshold:
        gap_lu = k - sub_top_center
        print(f"First liquid at center: k={k}, sub_top={sub_top_center}, gap={gap_lu} l.u. = {gap_lu*1e6:.1f}μm")
        break

# Check a few columns near the rim
print(f"\nFilm thickness at various r from center (y={mid}):")
for di in [0, 5, 10, 15, 20, 22, 23, 24, 25, 30]:
    i = mid + di
    if 0 <= i < nx:
        r = abs(di)
        solid_col = solid[i, mid, :]
        if solid_col.any():
            st = np.where(solid_col)[0].max()
        else:
            st = 0
        found = False
        for k in range(st + 1, nz):
            if rho_init[i, mid, k] > rho_threshold:
                gap_lu = k - st
                print(f"  r={r:2d}: sub_top={st:3d}, first_liquid_k={k:3d}, gap={gap_lu:2d} l.u. = {gap_lu*1e-6*1e6:.1f}μm")
                found = True
                break
        if not found:
            print(f"  r={r:2d}: sub_top={st:3d}, NO LIQUID found above substrate")

del solver
torch.cuda.empty_cache()
print("\nDone.")
