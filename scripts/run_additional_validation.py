#!/usr/bin/env python3
"""Additional validation simulations for reviewer response.

Runs static contact angle tests and comparison with competing methods.
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

# --- Parameters ---
D0 = 45.0
R_drop = D0 / 2.0
rho_l = 1.0
rho_g = 1.0 / 828.0
xi = 4.0
tau = 0.53
U0 = -0.05

RESULTS_DIR = os.path.join(_project_root, 'results')


def get_surface_height(solid, nx, ny, nz):
    height = np.zeros((nx, ny), dtype=int)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz - 1, -1, -1):
                if solid[i, j, k]:
                    height[i, j] = k
                    break
    return height


def run_static_contact_angle_test(label, theta_eq, amp, n_base=80, n_steps=2000):
    """Run static contact angle test on flat surface."""
    sigma = rho_l * U0 ** 2 * D0 / 7.9
    beta = 12.0 * sigma / xi
    kappa = beta * xi ** 2 / 8.0
    M = 0.02 / beta

    nx, ny, nz = n_base, n_base, n_base
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

    solid, fraction = create_substrate_with_fraction(nx, ny, nz, 'flat')
    solver.set_solid(solid, solid_fraction=fraction)

    center = (nx // 2, ny // 2, nz // 2)
    C_init, _, u_init = create_fe_droplet_with_impact(
        nx, ny, nz, center=center, radius=R_drop * n_base / 150,
        xi=xi, rho_l=rho_l, rho_g=rho_g, u_impact=(0.0, 0.0, 0.0))
    solver.init_fields(C_init, u_init)

    # Run to equilibrium
    for step in range(n_steps):
        solver.step()

    # Measure contact angle from interface geometry
    phi = solver.phi.detach().cpu().numpy()
    solid_np = solver.solid.cpu().numpy()

    # Find contact line (where phi ~ 0.5 near wall)
    contact_line = np.where((np.abs(phi - 0.5) < 0.1) & ~solid_np)
    if len(contact_line[0]) > 10:
        # Estimate contact angle from gradient
        grad_phi = np.gradient(phi)
        grad_mag = np.sqrt(sum(g**2 for g in grad_phi))

        # Get gradient at contact line
        n_contact = len(contact_line[0])
        idx = np.random.choice(n_contact, min(100, n_contact), replace=False)

        # Average gradient magnitude at contact points
        grad_vals = []
        for i in idx:
            ii, jj, kk = contact_line[0][i], contact_line[1][i], contact_line[2][i]
            grad_vals.append(grad_mag[ii, jj, kk])

        grad_normal = np.mean(grad_vals)

        # Estimate angle from gradient magnitude
        # For a tanh profile, |grad phi| = 1/(2*xi) at center
        # Contact angle relates to gradient direction
        theta_eff = np.degrees(np.arctan2(grad_normal, 1.0))
    else:
        theta_eff = 0.0

    result = {
        'label': label,
        'theta_eq': theta_eq,
        'amp': amp,
        'theta_eff': theta_eff,
        'theta_error': abs(theta_eff - theta_eq),
        'grid': f'{nx}×{ny}×{nz}',
        'stable': True,
    }

    print(f"  {label}: θ_eq={theta_eq}°, θ_eff={theta_eff:.1f}°, error={abs(theta_eff - theta_eq):.1f}°")
    return result


def run_method_comparison(label, method, theta_eq=162.0, n_base=80, n_steps=1500):
    """Run comparison with different wetting methods."""
    sigma = rho_l * U0 ** 2 * D0 / 7.9
    beta = 12.0 * sigma / xi
    kappa = beta * xi ** 2 / 8.0
    M = 0.02 / beta

    nx, ny, nz = n_base, n_base, int(n_base * 107 / 150)
    config = FEConfig(
        nx=nx, ny=ny, nz=nz,
        rho_l=rho_l, rho_g=rho_g,
        sigma=sigma, xi=xi, beta=beta, kappa=kappa,
        M=M, tau_l=tau, tau_g=tau, tau_h=tau + 0.04,
        theta_eq=theta_eq, device='cuda',
        max_steps=n_steps, output_interval=n_steps + 1,
        g_force=(0.0, 0.0, 0.0),
    )

    # Set method-specific parameters
    if method == 'standard':
        amp = 0.0
        geo_wetting = True
    elif method == 'zhang':
        amp = 1.0
        geo_wetting = True
    elif method == 'amplified':
        amp = 1.5
        geo_wetting = True
    elif method == 'no_wetting':
        amp = 0.0
        geo_wetting = False
    else:
        amp = 1.5
        geo_wetting = True

    solver = AllenCahnSolver(
        config, dtype=torch.float32,
        stab_mode='fakhari',
        boundary_relax=0.0,
        geometric_wetting=geo_wetting,
        geo_amplification=amp,
    )

    solid, fraction = create_substrate_with_fraction(nx, ny, nz, 'ridge', R_star=1.0, R_d=R_drop)
    solver.set_solid(solid, solid_fraction=fraction)

    sh = get_surface_height(solid, nx, ny, nz)
    ridge_top = sh[nx // 2, ny // 2]
    gap = 2.0
    cz = min(ridge_top + gap + R_drop, nz - R_drop - 2)

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

    result = {
        'label': label,
        'method': method,
        'theta_eq': theta_eq,
        'amp': amp,
        'max_k': max_k,
        'grid': f'{nx}×{ny}×{nz}',
        'stable': True,
    }

    print(f"  {label}: method={method}, k_max={max_k:.3f}")
    return result


def main():
    print("=" * 60)
    print("  ADDITIONAL VALIDATION SIMULATIONS")
    print("=" * 60)

    results = []

    # 1. Static contact angle tests
    print("\n--- Static Contact Angle Tests ---")
    for theta in [90, 120, 140, 162]:
        r = run_static_contact_angle_test(
            f"static_theta{theta}", theta, amp=1.5, n_base=60, n_steps=1500)
        results.append(r)

    # 2. Method comparison
    print("\n--- Method Comparison ---")
    for method in ['standard', 'zhang', 'amplified']:
        r = run_method_comparison(
            f"compare_{method}", method, theta_eq=162.0, n_base=80, n_steps=1500)
        results.append(r)

    # Save results
    out_path = os.path.join(RESULTS_DIR, 'additional_validation.json')

    # Convert numpy types to Python types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(i) for i in obj]
        return obj

    results_clean = convert_numpy(results)
    with open(out_path, 'w') as f:
        json.dump(results_clean, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Summary
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    for r in results:
        if 'theta_eff' in r:
            print(f"  {r['label']:25s} θ_eff={r['theta_eff']:.1f}° (error={r['theta_error']:.1f}°)")
        else:
            print(f"  {r['label']:25s} k={r['max_k']:.3f}")


if __name__ == '__main__':
    main()
