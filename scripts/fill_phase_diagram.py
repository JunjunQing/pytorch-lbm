#!/usr/bin/env python3
"""Fill phase diagram gaps and add method comparisons."""
import sys
import os
import json
import numpy as np
import torch
import time

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)

from lbm.fe_config import FEConfig
from lbm.fe_ac_solver import AllenCahnSolver
from lbm.fe_droplet import create_fe_droplet_with_impact
from geometry.substrate import create_substrate, create_substrate_with_fraction

torch.cuda.empty_cache()

D0 = 45.0
R_drop = D0 / 2.0
rho_l = 1.0
rho_g = 1.0 / 828.0
xi = 4.0
tau = 0.53
U0 = -0.05


def get_surface_height(solid, nx, ny, nz):
    height = np.zeros((nx, ny), dtype=int)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz - 1, -1, -1):
                if solid[i, j, k]:
                    height[i, j] = k
                    break
    return height


def run_case(label, substrate_type, R_star, We, theta_eq, amp,
             n_base=80, n_steps=1500):
    """Run a single simulation case."""
    sigma = rho_l * U0 ** 2 * D0 / We
    beta = 12.0 * sigma / xi
    kappa = beta * xi ** 2 / 8.0
    M = 0.02 / beta

    is_flat = substrate_type == 'flat' or R_star is None
    if is_flat:
        nx, ny, nz = n_base, n_base, n_base
    else:
        R_g = abs(R_star) * R_drop
        nx, ny = n_base, n_base
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

    if is_flat:
        solid, fraction = create_substrate_with_fraction(nx, ny, nz, 'flat')
    else:
        solid, fraction = create_substrate_with_fraction(
            nx, ny, nz, substrate_type, R_star=R_star, R_d=R_drop)
    solver.set_solid(solid, solid_fraction=fraction)

    sh = get_surface_height(solid, nx, ny, nz)
    ridge_top = sh[nx // 2, ny // 2] if not is_flat else 0
    gap = 2.0
    cz = min(ridge_top + gap + R_drop, nz - R_drop - 2) if not is_flat else nz // 2

    C_init, _, u_init = create_fe_droplet_with_impact(
        nx, ny, nz, center=(nx // 2, ny // 2, cz),
        radius=R_drop * n_base / 150, xi=xi, rho_l=rho_l, rho_g=rho_g,
        u_impact=(0.0, 0.0, U0))
    solver.init_fields(C_init, u_init)

    max_k = 0.0
    for step in range(n_steps):
        solver.step()
        if step % 100 == 0:
            phi = solver.phi.detach().cpu().numpy()
            interface = (phi > 0.5) & ~solver.solid.cpu().numpy()
            if interface.any():
                coords = np.argwhere(interface)
                Dx = float(coords[:, 0].max() - coords[:, 0].min() + 1)
                Dy = float(coords[:, 1].max() - coords[:, 1].min() + 1)
                k = Dx / Dy if Dy > 0 else 0
                if k > max_k:
                    max_k = k

    return {
        'label': label, 'substrate_type': substrate_type, 'R_star': R_star,
        'We': We, 'theta_eq': theta_eq, 'amp': amp,
        'max_k': float(max_k), 'grid': f'{nx}×{ny}×{nz}', 'stable': True,
    }


def main():
    print("=" * 60)
    print("  PHASE DIAGRAM GAP FILLING + METHOD COMPARISON")
    print("=" * 60)

    results = []

    # Phase diagram gaps: We=5 at different R*
    print("\n--- Phase Diagram: We=5 sweep ---")
    for R_star in [0.5, 0.7, 1.5, 2.0, 3.0]:
        r = run_case(f'phase_R{R_star}_We5', 'ridge', R_star, 5.0, 162.0, 1.5)
        results.append(r)

    # Phase diagram gaps: We=10 at different R*
    print("\n--- Phase Diagram: We=10 sweep ---")
    for R_star in [0.5, 0.7, 1.5, 2.0, 3.0]:
        r = run_case(f'phase_R{R_star}_We10', 'ridge', R_star, 10.0, 162.0, 1.5)
        results.append(r)

    # Phase diagram gaps: We=15 at different R*
    print("\n--- Phase Diagram: We=15 sweep ---")
    for R_star in [0.5, 0.7, 1.5, 2.0]:
        r = run_case(f'phase_R{R_star}_We15', 'ridge', R_star, 15.0, 162.0, 1.5)
        results.append(r)

    # Method comparison at different We
    print("\n--- Method Comparison at We=15 ---")
    r1 = run_case('method_std_We15', 'ridge', 1.0, 15.0, 162.0, 0.0)
    r2 = run_case('method_zhang_We15', 'ridge', 1.0, 15.0, 162.0, 1.0)
    r3 = run_case('method_amp_We15', 'ridge', 1.0, 15.0, 162.0, 1.5)
    results.extend([r1, r2, r3])

    # Save results
    def convert_numpy(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(i) for i in obj]
        return obj

    out_path = os.path.join(_project_root, 'results', 'phase_diagram_fill.json')
    with open(out_path, 'w') as f:
        json.dump(convert_numpy(results), f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Summary
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    for r in results:
        print(f"  {r['label']:25s} We={r['We']:5.1f} R*={str(r['R_star']):>5s} k={r['max_k']:.3f}")


if __name__ == '__main__':
    main()
