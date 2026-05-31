#!/usr/bin/env python3
"""M2: amp × theta_eq 2D sweep for R*=1.0
Fine-tune around best amp from M1 (amp=1.5 -> k=2.6552)
Also test lower theta_eq to reduce k further if needed.

Target: k = 2.6 ± 0.15 (Liu 2015 experiment)
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
R_star = 1.0
R_g = abs(R_star) * R_drop
nz = min(int(R_g + 2 + R_drop + 2*R_drop + 15), 300)
N_steps = 3000

print(f"Grid: {nx}x{ny}x{nz}")
print(f"M2: amp × theta_eq 2D sweep for R*=1.0")
print(f"Target k = 2.6 ± 0.15")
print("=" * 60)

def get_surface_height(solid, nx, ny, nz):
    h = np.zeros((nx, ny), dtype=int)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz-1, -1, -1):
                if solid[i, j, k]:
                    h[i, j] = k
                    break
    return h

amp_values = [1.3, 1.4, 1.5, 1.6]
theta_values = [155.0, 158.0, 160.0, 162.0]
results = []

total = len(amp_values) * len(theta_values)
count = 0

for amp in amp_values:
    for theta in theta_values:
        count += 1
        torch.cuda.empty_cache()
        t0 = time.time()

        config = FEConfig(
            nx=nx, ny=ny, nz=nz,
            rho_l=rho_l, rho_g=rho_g,
            sigma=sigma, xi=xi, beta=beta, kappa=kappa, M=M,
            tau_l=tau, tau_g=tau, tau_h=tau+0.04,
            theta_eq=theta, device='cuda',
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
        exp_k = 2.6
        err_pct = (max_k - exp_k) / exp_k * 100
        tag = "OK" if stable else "FAIL"
        in_range = abs(max_k - 2.6) <= 0.15 if stable else False

        print(f"[{count}/{total}] amp={amp:.1f} theta={theta:.0f}° | "
              f"k={max_k:.4f} Dx={Dx_peak:.0f} Dy={Dy_peak:.0f} "
              f"err={err_pct:+.1f}% {'✓' if in_range else '✗'} [{tag}] ({elapsed:.0f}s)")

        results.append({
            'amp': amp, 'theta_eq': theta, 'max_k': max_k,
            'max_k_step': max_k_step, 'Dx': Dx_peak, 'Dy': Dy_peak,
            'stable': stable, 'elapsed_s': round(elapsed, 1),
            'error_pct': round(err_pct, 1), 'in_range': in_range
        })

        del solver

# Summary
print("\n" + "=" * 60)
print("M2 AMP × THETA_EQ RESULTS")
print("=" * 60)
print(f"{'amp':>5} {'theta':>6} {'k':>7} {'Dx':>5} {'Dy':>5} {'err%':>7} {'in_range':>8}")
print("-" * 43)
for r in results:
    mark = "✓" if r['in_range'] else " "
    print(f"{r['amp']:5.1f} {r['theta_eq']:6.0f}° {r['max_k']:7.4f} "
          f"{r['Dx']:5.0f} {r['Dy']:5.0f} {r['error_pct']:7.1f}% {mark:>8}")

save_path = '/mnt/qmingjun/pytorch_lbm/results/m2_amp_theta_sweep_results.json'
with open(save_path, 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nSaved to {save_path}")

valid = [r for r in results if r['stable'] and r['in_range']]
if valid:
    best = min(valid, key=lambda r: abs(r['max_k'] - 2.6))
    print(f"\nBest config: amp={best['amp']}, theta={best['theta_eq']}° -> k={best['max_k']:.4f}")
    print(f"Error vs experiment: {best['error_pct']:.1f}%")
else:
    print("\nNo configs in ±0.15 range. Closest to target:")
    all_valid = [r for r in results if r['stable']]
    if all_valid:
        closest = min(all_valid, key=lambda r: abs(r['max_k'] - 2.6))
        print(f"  amp={closest['amp']}, theta={closest['theta_eq']}° -> k={closest['max_k']:.4f} (err={closest['error_pct']:.1f}%)")