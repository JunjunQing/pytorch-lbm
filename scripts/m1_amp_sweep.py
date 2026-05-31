#!/usr/bin/env python3
"""M1: amp sensitivity sweep for R*=1.0 validation.
Scans amp ∈ {0.0, 0.5, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0}
Target: k = 2.6 ± 0.2 (Liu 2015 experiment)
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

# --- Fixed params (from run_thesis_sweep.py) ---
D0, R_drop = 45.0, 22.5
rho_l, rho_g = 1.0, 1.0/828.0
xi, tau, U0 = 4.0, 0.53, -0.05
We = 7.9
sigma = rho_l * U0**2 * D0 / We
beta = 12.0 * sigma / xi
kappa = beta * xi**2 / 8.0
M = 0.02 / beta
theta_eq = 162.0
nx = ny = 150
R_star = 1.0
R_g = abs(R_star) * R_drop
nz = min(int(R_g + 2 + R_drop + 2*R_drop + 15), 300)
N_steps = 3000

print(f"Grid: {nx}x{ny}x{nz}")
print(f"sigma={sigma:.6f}, beta={beta:.6f}, kappa={kappa:.6f}")
print(f"M={M:.6e}, tau={tau}, xi={xi}")
print(f"R*={R_star}, theta_eq={theta_eq}°, We={We}")
print(f"Target k = 2.6 (Liu 2015 exp)")
print("=" * 60)

# --- Surface height helper ---
def get_surface_height(solid, nx, ny, nz):
    h = np.zeros((nx, ny), dtype=int)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz-1, -1, -1):
                if solid[i, j, k]:
                    h[i, j] = k
                    break
    return h

# --- Amp values to scan ---
amp_values = [0.0, 0.5, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0]
results = []

for amp in amp_values:
    torch.cuda.empty_cache()
    t0 = time.time()

    # Config
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

    # Substrate
    solid, fraction = create_substrate_with_fraction(
        nx, ny, nz, substrate_type='ridge',
        R_star=R_star, R_d=R_drop)
    solver.set_solid(solid, solid_fraction=fraction)

    # Droplet placement
    cx, cy = nx/2.0, ny/2.0
    solid_np = solver.solid.cpu().numpy()
    h = get_surface_height(solid_np, nx, ny, nz)
    ridge_top = h[nx//2, ny//2]
    cz = min(ridge_top + 2 + R_drop, nz - R_drop - 2)

    C_init, _, u_init = create_fe_droplet_with_impact(
        nx, ny, nz, center=(cx, cy, cz), radius=R_drop, xi=xi,
        rho_l=rho_l, rho_g=rho_g, u_impact=(0.0, 0.0, U0))
    solver.init_fields(C_init, u_init)

    # Run
    max_k = 0.0
    max_k_step = 0
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
                    max_k_step = step
                    Dx_peak = Dx
                    Dy_peak = Dy
            if np.isnan(phi_np).any():
                stable = False
                break

    elapsed = time.time() - t0
    tag = "OK" if stable else "FAIL"
    exp_k = 2.6
    err_pct = (max_k - exp_k) / exp_k * 100

    print(f"amp={amp:.1f} | k={max_k:.4f} Dx={Dx_peak:.0f} Dy={Dy_peak:.0f} "
          f"step={max_k_step} err={err_pct:+.1f}% [{tag}] ({elapsed:.0f}s)")

    results.append({
        'amp': amp, 'max_k': max_k, 'max_k_step': max_k_step,
        'Dx': Dx_peak, 'Dy': Dy_peak, 'stable': stable,
        'elapsed_s': round(elapsed, 1), 'error_pct': round(err_pct, 1)
    })

    del solver

# --- Summary ---
print("\n" + "=" * 60)
print("M1 AMP SENSITIVITY RESULTS")
print("=" * 60)
print(f"{'amp':>5} {'k':>7} {'Dx':>5} {'Dy':>5} {'step':>6} {'err%':>7} {'status':>6}")
print("-" * 45)
for r in results:
    status = "OK" if r['stable'] else "FAIL"
    print(f"{r['amp']:5.1f} {r['max_k']:7.4f} {r['Dx']:5.0f} {r['Dy']:5.0f} "
          f"{r['max_k_step']:6d} {r['error_pct']:7.1f}% {status:>6}")

# Save results
save_path = '/mnt/qmingjun/pytorch_lbm/results/m1_amp_sweep_results.json'
with open(save_path, 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nSaved to {save_path}")

# Best amp
valid = [r for r in results if r['stable']]
if valid:
    best = min(valid, key=lambda r: abs(r['max_k'] - 2.6))
    print(f"\nBest amp: {best['amp']} -> k={best['max_k']:.4f} (err={best['error_pct']:.1f}%)")
    print(f"Target k=2.6 -> acceptable range 2.4-2.8")
    for r in valid:
        in_range = 2.4 <= r['max_k'] <= 2.8
        print(f"  amp={r['amp']:.1f}: k={r['max_k']:.4f} {'✓' if in_range else '✗'}")