#!/usr/bin/env python3
"""Run supplementary simulations to fill data gaps in figures.

fig1: We=25, 30 at N=150, α=1.5 (ridge, R*=1.0)
fig3: θ=140° at N=150, α=1.5 (ridge, R*=1.0, We=7.9)
"""
import sys
import os

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)

import numpy as np
import torch
import time
import json

from lbm.fe_config import FEConfig
from lbm.fe_ac_solver import AllenCahnSolver
from lbm.fe_droplet import create_fe_droplet_with_impact
from geometry.substrate import create_substrate, create_substrate_with_fraction

torch.cuda.empty_cache()

# --- Physical parameters (same as run_thesis_sweep.py) ---
D0 = 45.0
R_drop = D0 / 2.0
rho_l = 1.0
rho_g = 1.0 / 828.0
xi = 4.0
tau = 0.53
U0 = -0.05

# --- Configuration ---
OUTPUT_DIR = os.path.join(_project_root, 'results')
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_BASE = 150
R_STAR = 1.0
ALPHA_GEO = 1.5
N_STEPS = 2000


def get_surface_height(solid, nx, ny, nz):
    """Get topmost solid index per (i,j) column."""
    height = np.zeros((nx, ny), dtype=int)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz - 1, -1, -1):
                if solid[i, j, k]:
                    height[i, j] = k
                    break
    return height


def run_one(label, We=7.9, theta_eq=162.0, amp=ALPHA_GEO, n_steps=N_STEPS):
    """Run a single simulation and return result dict."""
    sigma = rho_l * U0 ** 2 * D0 / We
    beta = 12.0 * sigma / xi
    kappa = beta * xi ** 2 / 8.0
    M = 0.02 / beta

    R_g = abs(R_STAR) * R_drop
    nx = ny = N_BASE
    nz = int(R_g + 2 + R_drop + 2 * R_drop + 15)
    nz = min(nz, 300)

    config = FEConfig(
        nx=nx, ny=ny, nz=nz,
        rho_l=rho_l, rho_g=rho_g,
        sigma=sigma, xi=xi, beta=beta, kappa=kappa,
        M=M, tau_l=tau, tau_g=tau, tau_h=tau + 0.04,
        theta_eq=theta_eq, device='cuda',
        max_steps=n_steps, output_interval=n_steps + 1,
        g_force=(0.0, 0.0, 0.0),
    )

    solver = AllenCahnSolver(
        config, dtype=torch.float32,
        stab_mode='fakhari',
        boundary_relax=0.0,
        geometric_wetting=(amp > 0),
        geo_amplification=amp,
    )

    # Substrate
    solid, fraction = create_substrate_with_fraction(
        nx, ny, nz, substrate_type='ridge',
        R_star=R_STAR, R_d=R_drop)
    solver.set_solid(solid, solid_fraction=fraction)

    # Droplet placement
    surface_height = get_surface_height(solid, nx, ny, nz)
    cx, cy = nx / 2.0, ny / 2.0
    ridge_top = surface_height[nx // 2, ny // 2]
    gap = 2.0
    cz = ridge_top + gap + R_drop
    max_cz = nz - R_drop - 2
    if cz > max_cz:
        cz = max_cz

    C_init, _, u_init = create_fe_droplet_with_impact(
        nx, ny, nz, center=(cx, cy, cz), radius=R_drop, xi=xi,
        rho_l=rho_l, rho_g=rho_g, u_impact=(0.0, 0.0, U0))
    solver.init_fields(C_init, u_init)

    # Run
    t0 = time.time()
    max_k = 0.0
    max_k_step = 0
    Dx_peak = Dy_peak = 0.0
    k_history = []
    stable = True

    for step in range(1, n_steps + 1):
        solver.step()

        if step % 50 == 0:
            phi_np = solver.phi.detach().cpu().numpy()
            solid_np = solver.solid.cpu().numpy()
            interface = (phi_np > 0.5) & ~solid_np

            if interface.any():
                coords = np.argwhere(interface)
                Dx = float(coords[:, 0].max() - coords[:, 0].min() + 1)
                Dy = float(coords[:, 1].max() - coords[:, 1].min() + 1)
                Dz = float(coords[:, 2].max() - coords[:, 2].min() + 1)
                k_val = Dx / Dy if Dy > 0 else 0
            else:
                Dx = Dy = Dz = k_val = 0

            k_history.append({
                'step': step, 'Dx': Dx, 'Dy': Dy, 'Dz': Dz, 'k': k_val
            })

            if k_val > max_k:
                max_k = k_val
                max_k_step = step
                Dx_peak = Dx
                Dy_peak = Dy

            if np.isnan(phi_np).any() or phi_np.max() < 0.01:
                stable = False
                break

            if step % 200 == 0:
                print(f"    step={step}: k={k_val:.4f}, Dx={Dx:.0f}, Dy={Dy:.0f}")

    elapsed = time.time() - t0
    mem_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

    result = {
        'label': label,
        'substrate_type': 'ridge',
        'R_star': R_STAR,
        'We': We,
        'theta_eq': theta_eq,
        'amp': amp,
        'max_k': max_k,
        'max_k_step': max_k_step,
        'Dx': Dx_peak, 'Dy': Dy_peak,
        'stable': stable,
        'grid': f'{nx}x{ny}x{nz}',
        'elapsed_s': elapsed,
        'mem_mb': mem_mb,
        'machine': 'local_gtx1080',
        'history': k_history,
    }

    print(f"  → k_max={max_k:.4f} at step {max_k_step}, elapsed={elapsed:.1f}s")
    return result


def main():
    results = []

    print("="*60)
    print("SUPPLEMENTARY SIMULATIONS — fig1 & fig3")
    print("="*60)

    # fig1: We=25, 30 at N=150, α=1.5
    for We_val in [25.0, 30.0]:
        r = run_one(f'fig1_we{int(We_val)}', We=We_val, theta_eq=162.0, n_steps=2000)
        results.append(r)

    # fig3: θ=140° at N=150, α=1.5, We=7.9
    r = run_one('fig3_theta140', We=7.9, theta_eq=140.0, n_steps=2000)
    results.append(r)

    # Save
    out_path = os.path.join(OUTPUT_DIR, 'supplement_fig1_fig3.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Results saved to {out_path}")
    print(f"\nSummary:")
    for r in results:
        print(f"  {r['label']:20s}: k_max={r['max_k']:.4f} (We={r['We']}, θ={r['theta_eq']}°)")


if __name__ == '__main__':
    main()
