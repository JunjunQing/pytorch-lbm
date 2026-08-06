#!/usr/bin/env python3
"""Resolve the archived R*=2.76 mystery (M4 amp-insensitivity + R036).

Background
----------
M4 sweep (R027-R035, June): R*=2.76, theta=162, amp in {0.2..1.5} on grid
150x150x146 -> max_k = 1.6857 bit-identical across ALL amp values (26.7% error
vs Liu 1.33). Conclusion at the time: "wetting BC has NO EFFECT at R*=2.76".

Later comparison run (phase4a): R*=2.76, amp=0.0 on grid 150x150x107 ->
final k = 1.308 (1.7% error). Two variables differ: amp AND grid height.

Code analysis (fe_wetting.py correct_gradient):
  * geometric wetting correction is ONLY applied when geometric_wetting=True
    (solver flag `geometric_wetting=(amp > 0)`) -> amp=0 disables the BC.
  * amplification only active when `_geo_amplification > 1.0`; amp in (0,1]
    executes the IDENTICAL un-amplified code path -> invariance for amp<=1.0
    is trivially expected, not mysterious.

This script runs the deconfounding cases on one grid (150x150x146):
  A: amp=0.0  theta=162  (wetting OFF on the M4 grid)
  E: amp=0.1  theta=162  (is any amp>0 == "wetting ON"?)
  B: amp=0.5  theta=140  (R036 core: does lower theta restore amp effect?)
  D: amp=0.0  theta=140  (low-theta control)
  C: amp=0.5  theta=150  (R036 middle point)

Metrics mirror M4 (max_k over N=3000, Dx/Dy at peak, every 50 steps) plus
final k at steps 2000/3000 for comparison with the phase4a run.

Usage: venv/bin/python3 scripts/resolve_r276.py
"""
import sys
import os
import json
import time

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)

import numpy as np
import torch

from lbm.fe_config import FEConfig
from lbm.fe_ac_solver import AllenCahnSolver
from lbm.fe_droplet import create_fe_droplet_with_impact
from geometry.substrate import create_substrate_with_fraction

torch.cuda.empty_cache()

D0, R_drop = 45.0, 22.5
rho_l, rho_g = 1.0, 1.0 / 828.0
xi, tau, U0 = 4.0, 0.53, -0.05
We = 7.9
sigma = rho_l * U0 ** 2 * D0 / We
beta = 12.0 * sigma / xi
kappa = beta * xi ** 2 / 8.0
M = 0.02 / beta

nx = ny = 150
R_star = 2.76
R_g = abs(R_star) * R_drop
nz = min(int(R_g + 2 + R_drop + 2 * R_drop + 15), 300)
N_steps = 3000
TARGET = 1.33  # Liu et al. 2015 experimental k at D/D0 = 2.76


def get_surface_height(solid):
    """Topmost solid index per (i,j) column (vectorized, keep-first semantics).

    Equivalent to the per-column loop with `break` at the first solid cell
    from the top; np.where must not overwrite already-assigned columns.
    """
    h = np.zeros((solid.shape[0], solid.shape[1]), dtype=int)
    unset = np.ones_like(h, dtype=bool)
    for k in range(solid.shape[2] - 1, -1, -1):
        layer = solid[:, :, k] & unset
        if layer.any():
            h = np.where(layer, k, h)
            unset &= ~layer
    return h


def measure(phi, solid):
    """Return (Dx, Dy, k) for the current interface."""
    interface = (phi > 0.5) & ~solid
    if not interface.any():
        return 0.0, 0.0, 0.0
    coords = np.argwhere(interface)
    Dx = float(coords[:, 0].max() - coords[:, 0].min() + 1)
    Dy = float(coords[:, 1].max() - coords[:, 1].min() + 1)
    return Dx, Dy, (Dx / Dy if Dy > 0 else 0.0)


def run_case(amp, theta_eq, label):
    torch.cuda.empty_cache()
    t0 = time.time()

    config = FEConfig(
        nx=nx, ny=ny, nz=nz,
        rho_l=rho_l, rho_g=rho_g,
        sigma=sigma, xi=xi, beta=beta, kappa=kappa, M=M,
        tau_l=tau, tau_g=tau, tau_h=tau + 0.04,
        theta_eq=theta_eq, device='cuda',
        max_steps=N_steps, output_interval=N_steps + 1,
        g_force=(0.0, 0.0, 0.0),
    )

    solver = AllenCahnSolver(
        config, dtype=torch.float32,
        stab_mode='fakhari',
        boundary_relax=0.0,
        geometric_wetting=(amp > 0),
        geo_amplification=amp,
    )

    solid, fraction = create_substrate_with_fraction(
        nx, ny, nz, substrate_type='ridge',
        R_star=R_star, R_d=R_drop)
    solver.set_solid(solid, solid_fraction=fraction)

    cx, cy = nx / 2.0, ny / 2.0
    h = get_surface_height(solver.solid.cpu().numpy())
    ridge_top = float(h[nx // 2, ny // 2])
    cz = min(ridge_top + 2.0 + R_drop, nz - R_drop - 2)

    C_init, _, u_init = create_fe_droplet_with_impact(
        nx, ny, nz, center=(cx, cy, cz), radius=R_drop, xi=xi,
        rho_l=rho_l, rho_g=rho_g, u_impact=(0.0, 0.0, U0))
    solver.init_fields(C_init, u_init)

    max_k, max_k_step = 0.0, 0
    Dx_peak, Dy_peak = 0.0, 0.0
    k_final_2000 = k_final_3000 = None
    stable = True

    for step in range(1, N_steps + 1):
        solver.step()
        if step % 50 == 0:
            phi_np = solver.phi.detach().cpu().numpy()
            solid_np = solver.solid.cpu().numpy()
            Dx, Dy, k_val = measure(phi_np, solid_np)
            if k_val > max_k:
                max_k, max_k_step = k_val, step
                Dx_peak, Dy_peak = Dx, Dy
            if step == 2000:
                k_final_2000 = k_val
            if step == N_steps:
                k_final_3000 = k_val
            if np.isnan(phi_np).any():
                stable = False
                break

    elapsed = time.time() - t0
    err_pct = (max_k - TARGET) / TARGET * 100

    result = {
        'label': label, 'amp': amp, 'theta_eq': theta_eq,
        'R_star': R_star, 'We': We, 'grid': f'{nx}x{ny}x{nz}',
        'max_k': round(max_k, 4), 'max_k_step': max_k_step,
        'Dx': round(Dx_peak, 1), 'Dy': round(Dy_peak, 1),
        'k_final_2000': round(k_final_2000, 4) if k_final_2000 else None,
        'k_final_3000': round(k_final_3000, 4) if k_final_3000 else None,
        'stable': stable, 'elapsed_s': round(elapsed, 1),
        'error_pct': round(err_pct, 1),
        'in_range': abs(max_k - TARGET) <= 0.15 if stable else False,
    }
    print(f"[{label}] amp={amp:.1f} theta={theta_eq:.0f} | "
          f"max_k={max_k:.4f}@{max_k_step} Dx={Dx_peak:.0f} Dy={Dy_peak:.0f} "
          f"k(2000)={k_final_2000} err={err_pct:+.1f}% "
          f"{'✓' if result['in_range'] else '✗'} ({elapsed:.0f}s)", flush=True)
    del solver
    return result


def main():
    cases = [
        (0.0, 162.0, 'A_amp0_th162'),   # wetting OFF on M4 grid
        (0.1, 162.0, 'E_amp0.1_th162'),  # any amp>0 -> wetting ON?
        (0.5, 140.0, 'B_amp0.5_th140'),  # R036 core
        (0.0, 140.0, 'D_amp0_th140'),    # low-theta control
        (0.5, 150.0, 'C_amp0.5_th150'),  # R036 middle
    ]
    print(f"Grid: {nx}x{ny}x{nz} | R*=2.76 | We=7.9 | N={N_steps} | "
          f"target k={TARGET}±0.15 (Liu 2015)", flush=True)
    results = []
    for amp, theta, label in cases:
        results.append(run_case(amp, theta, label))

    print("\n" + "=" * 66)
    print("R*=2.76 RESOLUTION SWEEP RESULTS")
    print("=" * 66)
    print(f"{'label':<18} {'amp':>4} {'th':>4} {'max_k':>7} {'@step':>6} "
          f"{'Dx':>5} {'Dy':>5} {'k2000':>7} {'err%':>7}")
    print("-" * 66)
    for r in results:
        mark = "✓" if r['in_range'] else " "
        print(f"{r['label']:<18} {r['amp']:>4.1f} {r['theta_eq']:>4.0f} "
              f"{r['max_k']:>7.4f} {r['max_k_step']:>6d} {r['Dx']:>5.0f} "
              f"{r['Dy']:>5.0f} {str(r['k_final_2000']):>7} "
              f"{r['error_pct']:>+6.1f}%{mark}")

    save_path = os.path.join(_project_root, 'results', 'r276_resolution.json')
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {save_path}")


if __name__ == '__main__':
    main()
