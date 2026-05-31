#!/usr/bin/env python3
"""Diagnostic script for R*=2.76 k invariance problem.

Tests three hypotheses:
1. Wetting BC is active (boundary nodes, near_wall nodes)
2. Lower theta alone (no amp) changes k
3. Lower theta with amp changes k further

Also measures: boundary node count, near_wall count, ridge geometry.
"""
import sys, os, json, time
sys.path.insert(0, '/mnt/qmingjun/pytorch_lbm')

import torch
import numpy as np
from lbm.fe_config import FEConfig
from lbm.fe_ac_solver import AllenCahnSolver
from lbm.fe_droplet import create_fe_droplet_with_impact
from geometry.substrate import create_substrate_with_fraction

torch.cuda.empty_cache()

D0, R_drop = 45.0, 22.5
rho_l, rho_g = 1.0, 1.0/828.0
xi, tau, U0 = 4.0, 0.53, -0.05
We = 7.9
sigma = rho_l * U0**2 * D0 / We
beta = 12.0 * sigma / xi
kappa = beta * xi**2 / 8.0
M = 0.02 / beta
nx = ny = 150
N_steps = 3000

def get_surface_height(solid, nx, ny, nz):
    h = np.zeros((nx, ny), dtype=int)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz-1, -1, -1):
                if solid[i, j, k]:
                    h[i, j] = k
                    break
    return h

def run_case(R_star, theta_eq, amp, n_base=150, label=None):
    torch.cuda.empty_cache()
    t0 = time.time()

    R_g = abs(R_star) * R_drop
    nz = min(int(R_g + 2 + R_drop + 2*R_drop + 15), 300)

    config = FEConfig(
        nx=nx, ny=ny, nz=nz,
        rho_l=rho_l, rho_g=rho_g,
        sigma=sigma, xi=xi, beta=beta, kappa=kappa, M=M,
        tau_l=tau, tau_g=tau, tau_h=tau+0.04,
        theta_eq=theta_eq, device='cuda',
        max_steps=N_steps, output_interval=N_steps+1,
        g_force=(0.0, 0.0, 0.0)
    )

    solver = AllenCahnSolver(
        config, dtype=torch.float32,
        stab_mode='fakhari',
        boundary_relax=0.0,
        geometric_wetting=(amp > 0),
        geo_amplification=amp
    )

    solid, fraction = create_substrate_with_fraction(
        nx, ny, nz, substrate_type='ridge',
        R_star=R_star, R_d=R_drop)
    solver.set_solid(solid, solid_fraction=fraction)

    # Diagnostic: count boundary nodes and near_wall nodes
    n_boundary = solver.wetting.is_boundary.sum().item() if solver.wetting else 0
    n_near_wall = solver.wetting.near_wall_mask.sum().item() if solver.wetting else 0
    n_solid = solver.solid.sum().item()

    cx, cy = nx/2.0, ny/2.0
    solid_np = solver.solid.cpu().numpy()
    h = get_surface_height(solid_np, nx, ny, nz)
    ridge_top = h[nx//2, ny//2]
    cz = min(ridge_top + 2 + R_drop, nz - R_drop - 2)

    C_init, _, u_init = create_fe_droplet_with_impact(
        nx, ny, nz, center=(cx, cy, cz), radius=R_drop, xi=xi,
        rho_l=rho_l, rho_g=rho_g, u_impact=(0.0, 0.0, U0))
    solver.init_fields(C_init, u_init)

    max_k = 0.0
    Dx_peak, Dy_peak = 0.0, 0.0
    stable = True

    for step in range(1, N_steps+1):
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
                    Dx_peak = Dx
                    Dy_peak = Dy
            if np.isnan(phi_np).any():
                stable = False
                break

    elapsed = time.time() - t0
    del solver

    return {
        'label': label, 'R_star': R_star, 'theta_eq': theta_eq,
        'amp': amp, 'nz': nz, 'ridge_top': ridge_top,
        'n_boundary': n_boundary, 'n_near_wall': n_near_wall, 'n_solid': n_solid,
        'max_k': max_k, 'Dx': Dx_peak, 'Dy': Dy_peak,
        'stable': stable, 'elapsed_s': round(elapsed, 1)
    }

# ===== EXPERIMENTS =====

print("=" * 70)
print("DIAGNOSTIC: R*=2.76 k invariance investigation")
print("=" * 70)
print(f"Grid: {nx}x{ny}x{{nz}} (variable)")
print(f"Target R*=2.76: k≈1.33, R*=1.0: k≈2.6")
print()

results = []

# H1: Compare boundary geometry R*=1 vs R*=2.76
print("=== H1: Boundary geometry comparison ===")
for R_star in [1.0, 2.76]:
    R_g = abs(R_star) * R_drop
    nz = min(int(R_g + 2 + R_drop + 2*R_drop + 15), 300)
    solid, fraction = create_substrate_with_fraction(
        nx, ny, nz, substrate_type='ridge',
        R_star=R_star, R_d=R_drop)
    n_boundary = np.sum(fraction > 0)  # approximate

    config = FEConfig(
        nx=nx, ny=ny, nz=nz,
        rho_l=rho_l, rho_g=rho_g,
        sigma=sigma, xi=xi, beta=beta, kappa=kappa, M=M,
        tau_l=tau, tau_g=tau, tau_h=tau+0.04,
        theta_eq=162.0, device='cuda',
        max_steps=10, output_interval=11,
        g_force=(0.0, 0.0, 0.0)
    )
    solver = AllenCahnSolver(
        config, dtype=torch.float32,
        stab_mode='fakhari',
        boundary_relax=0.0,
        geometric_wetting=True,
        geo_amplification=1.0
    )
    solver.set_solid(solid, solid_fraction=fraction)

    n_bnd = solver.wetting.is_boundary.sum().item() if solver.wetting else 0
    n_near = solver.wetting.near_wall_mask.sum().item() if solver.wetting else 0

    print(f"  R*={R_star}: ridge_top≈{R_g:.1f}, nz={nz}, boundary_nodes={n_bnd}, "
          f"near_wall={n_near}, solid={solver.solid.sum().item()}")
    del solver

print()

# H2: amp=0 with different theta_eq at R*=2.76
print("=== H2: Theta effect WITHOUT amp (amp=0) at R*=2.76 ===")
for theta in [162.0, 150.0, 140.0]:
    r = run_case(2.76, theta, amp=0.0, label=f"R2.76_t{int(theta)}_a0")
    results.append(r)
    exp_k = 1.33
    print(f"  theta={theta:.0f}° amp=0 | k={r['max_k']:.4f} Dx={r['Dx']:.0f} Dy={r['Dy']:.0f} "
          f"err={(r['max_k']-exp_k)/exp_k*100:+.1f}% bnd={r['n_boundary']} near={r['n_near_wall']}")

print()

# H3: amp=1.0 with different theta_eq at R*=2.76
print("=== H3: Theta effect WITH amp=1.0 at R*=2.76 ===")
for theta in [162.0, 140.0]:
    r = run_case(2.76, theta, amp=1.0, label=f"R2.76_t{int(theta)}_a1")
    results.append(r)
    exp_k = 1.33
    print(f"  theta={theta:.0f}° amp=1.0 | k={r['max_k']:.4f} Dx={r['Dx']:.0f} Dy={r['Dy']:.0f} "
          f"err={(r['max_k']-exp_k)/exp_k*100:+.1f}%")

print()

# H4: R*=1 reference with amp=0
print("=== H4: R*=1.0 reference ===")
for theta in [162.0, 155.0]:
    r = run_case(1.0, theta, amp=0.0, label=f"R1_t{int(theta)}_a0")
    results.append(r)
    exp_k = 2.6
    print(f"  theta={theta:.0f}° amp=0 | k={r['max_k']:.4f} Dx={r['Dx']:.0f} Dy={r['Dy']:.0f} "
          f"err={(r['max_k']-exp_k)/exp_k*100:+.1f}% bnd={r['n_boundary']} near={r['n_near_wall']}")

print()

# H5: R*=1 with amp=1.3 (best from M2)
r = run_case(1.0, 158.0, amp=1.3, label="R1_t158_a1.3")
results.append(r)
print(f"  R*=1.0: theta=158° amp=1.3 | k={r['max_k']:.4f} (BEST from M2)")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"{'Label':>20} {'R*':>4} {'theta':>6} {'amp':>5} {'k':>7} {'Dx':>5} {'Dy':>5} {'bnd':>7} {'near':>7}")
print("-" * 75)
for r in results:
    print(f"{r['label']:>20} {r['R_star']:>4.2f} {r['theta_eq']:>6.0f}° "
          f"{r['amp']:>5.1f} {r['max_k']:>7.4f} {r['Dx']:>5.0f} {r['Dy']:>5.0f} "
          f"{r['n_boundary']:>7} {r['n_near_wall']:>7}")

save_path = '/mnt/qmingjun/pytorch_lbm/results/diag_r276_results.json'
with open(save_path, 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nSaved to {save_path}")

# Key analysis
print("\n" + "=" * 70)
print("KEY FINDINGS")
print("=" * 70)
r276_a0_162 = next(r for r in results if r['label'] == 'R2.76_t162_a0')
r276_a0_140 = next(r for r in results if r['label'] == 'R2.76_t140_a0')
r276_a1_162 = next(r for r in results if r['label'] == 'R2.76_t162_a1')
r1_a0_162 = next(r for r in results if r['label'] == 'R1_t162_a0')
r1_a0_155 = next(r for r in results if r['label'] == 'R1_t155_a0')

print(f"R*=2.76 amp=0 theta=162: k={r276_a0_162['max_k']:.4f} (baseline)")
print(f"R*=2.76 amp=0 theta=140: k={r276_a0_140['max_k']:.4f} (delta={r276_a0_140['max_k']-r276_a0_162['max_k']:+.4f})")
print(f"R*=2.76 amp=1  theta=162: k={r276_a1_162['max_k']:.4f} (amp effect)")
print(f"R*=1.0 amp=0 theta=162:  k={r1_a0_162['max_k']:.4f} (reference)")
print(f"R*=1.0 amp=0 theta=155:  k={r1_a0_155['max_k']:.4f} (theta sensitivity)")

# Is theta alone changing k for R*=2.76?
theta_effect_r276 = r276_a0_140['max_k'] - r276_a0_162['max_k']
amp_effect_r276 = r276_a1_162['max_k'] - r276_a0_162['max_k']
theta_effect_r1 = r1_a0_155['max_k'] - r1_a0_162['max_k']

print(f"\nTheta effect at R*=2.76: {theta_effect_r276:+.4f}")
print(f"Amp effect at R*=2.76: {amp_effect_r276:+.4f}")
print(f"Theta effect at R*=1.0: {theta_effect_r1:+.4f}")

if abs(theta_effect_r276) < 0.01:
    print("\n⚠️ CRITICAL: Theta has NO effect at R*=2.76 (amp-insensitive confirmed)")
else:
    print(f"\n✓ Theta DOES affect k at R*=2.76: {theta_effect_r276:+.4f} range")