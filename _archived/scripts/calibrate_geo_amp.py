#!/usr/bin/env python3
"""Systematic calibration of geo_amplification factor.

Studies:
  1. Resolution dependence: N = 60, 80, 100, 120, 150
  2. Curvature dependence: R* = 0.5, 0.7, 1.0, 1.5, 2.0, 3.0
  3. Contact angle dependence: θ = 90°, 120°, 140°, 162°
  4. amp sweep: 0.5, 1.0, 1.5, 2.0, 2.5, 3.0

For each case, measures:
  - Equilibrium contact angle (target vs actual)
  - Contact angle error
  - Stability (NaN check)
  - Spreading diameter at equilibrium
"""
import sys, os, json, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lbm.fe_config import FEConfig
from lbm.fe_ac_solver import AllenCahnSolver
from lbm.fe_droplet import create_fe_droplet_with_impact
from geometry.substrate import create_substrate_with_fraction
from lbm.fe_measure import measure_contact_angle

torch.cuda.empty_cache()

# Physical params
D0_base = 45.0; rho_l = 1.0; rho_g = 1.0/828.0; xi = 4.0
U0 = -0.05; We = 7.9; tau = 0.53


def get_surface_height(solid, nx, ny, nz):
    h = np.zeros((nx, ny), dtype=int)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz-1, -1, -1):
                if solid[i, j, k]:
                    h[i, j] = k; break
    return h


def run_case(n_base, R_star, theta_eq, amp, N_equil=2000):
    """Run a static droplet equilibration case.

    Drops a droplet on the ridge and lets it equilibrate.
    Measures the final contact angle vs target.
    """
    # Scale D0 with resolution
    D0 = D0_base * n_base / 150.0
    R_drop = D0 / 2.0
    sigma = rho_l * U0**2 * D0 / We
    beta = 12.0 * sigma / xi
    kappa = beta * xi**2 / 8.0
    M = 0.02 / beta

    # Grid sizing
    R_g = abs(R_star) * R_drop
    nz = int(R_g + 2 + R_drop + 2*R_drop + 15)
    nz = min(nz, 300)
    nx, ny = n_base, n_base

    config = FEConfig(
        nx=nx, ny=ny, nz=nz,
        rho_l=rho_l, rho_g=rho_g,
        sigma=sigma, xi=xi, beta=beta, kappa=kappa, M=M,
        tau_l=tau, tau_g=tau, tau_h=tau+0.04,
        theta_eq=theta_eq, device='cuda',
        max_steps=N_equil, output_interval=N_equil+1,
        g_force=(0.0, 0.0, 0.0),
    )

    solver = AllenCahnSolver(
        config, dtype=torch.float32,
        stab_mode='fakhari', boundary_relax=0.0,
        geometric_wetting=(amp > 0), geo_amplification=amp,
    )

    solid, fraction = create_substrate_with_fraction(
        nx, ny, nz, substrate_type='ridge',
        R_star=R_star, R_d=R_drop)
    solver.set_solid(solid, solid_fraction=fraction)

    # Place droplet on ridge surface
    height = get_surface_height(solid, nx, ny, nz)
    ridge_top = height[nx//2, ny//2]
    cx, cy = nx/2.0, ny/2.0
    # Place droplet resting on surface (no impact velocity)
    cz = ridge_top + 2.0 + R_drop

    C_init, _, u_init = create_fe_droplet_with_impact(
        nx, ny, nz, center=(cx, cy, cz), radius=R_drop, xi=xi,
        rho_l=rho_l, rho_g=rho_g, u_impact=(0.0, 0.0, 0.0))  # NO impact
    solver.init_fields(C_init, u_init)

    # Run equilibrium
    for step in range(1, N_equil + 1):
        solver.step()

    # Measure contact angle at end
    phi_np = solver.phi.detach().cpu().numpy()
    solid_np = solver.solid.cpu().numpy()
    theta_actual = measure_contact_angle(phi_np, solid_np, axis='z')

    # Check stability
    stable = not torch.isnan(solver.phi).any() and solver.phi.max() > 0.01

    # Measure spreading diameter
    interface = (phi_np > 0.5) & ~solid_np
    if interface.any():
        coords = np.argwhere(interface)
        Dx = float(coords[:, 0].max() - coords[:, 0].min() + 1)
        Dy = float(coords[:, 1].max() - coords[:, 1].min() + 1)
    else:
        Dx = Dy = 0

    del solver
    torch.cuda.empty_cache()

    return {
        'theta_target': theta_eq,
        'theta_actual': theta_actual,
        'theta_error': abs(theta_actual - theta_eq),
        'Dx': Dx, 'Dy': Dy,
        'stable': stable,
        'grid': f'{nx}x{ny}x{nz}',
    }


def main():
    results = []
    t_start = time.time()

    # =================================================================
    # Study 1: Resolution dependence (R*=1.0, θ=162°, amp=1.5)
    # =================================================================
    print("\n" + "="*60)
    print("  Study 1: Resolution dependence")
    print("  R*=1.0, θ=162°, amp=1.5")
    print("="*60)

    for n in [60, 80, 100, 120, 150]:
        r = run_case(n, R_star=1.0, theta_eq=162.0, amp=1.5, N_equil=2000)
        r['study'] = 'resolution'
        r['n_base'] = n
        results.append(r)
        err = r['theta_error']
        print(f"  N={n:3d}: θ_actual={r['theta_actual']:.1f}° "
              f"(err={err:.1f}°) stable={r['stable']}")

    # =================================================================
    # Study 2: Amp sweep at different resolutions
    # =================================================================
    print("\n" + "="*60)
    print("  Study 2: Amp sweep (R*=1.0, θ=162°)")
    print("="*60)

    for n in [80, 120]:
        print(f"\n  --- N={n} ---")
        for amp in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
            r = run_case(n, R_star=1.0, theta_eq=162.0, amp=amp, N_equil=2000)
            r['study'] = 'amp_sweep'
            r['n_base'] = n
            r['amp'] = amp
            results.append(r)
            err = r['theta_error']
            print(f"    amp={amp:.1f}: θ={r['theta_actual']:.1f}° "
                  f"(err={err:.1f}°) stable={r['stable']}")

    # =================================================================
    # Study 3: Curvature dependence (N=100, θ=162°)
    # =================================================================
    print("\n" + "="*60)
    print("  Study 3: Curvature dependence")
    print("  N=100, θ=162°, amp=1.5")
    print("="*60)

    for R_star in [0.5, 1.0, 2.0, 3.0]:
        r = run_case(100, R_star=R_star, theta_eq=162.0, amp=1.5, N_equil=2000)
        r['study'] = 'curvature'
        r['R_star'] = R_star
        results.append(r)
        err = r['theta_error']
        print(f"  R*={R_star:.1f}: θ={r['theta_actual']:.1f}° "
              f"(err={err:.1f}°) stable={r['stable']}")

    # =================================================================
    # Study 4: Contact angle dependence (N=100, R*=1.0)
    # =================================================================
    print("\n" + "="*60)
    print("  Study 4: Contact angle dependence")
    print("  N=100, R*=1.0, amp=1.5")
    print("="*60)

    for theta in [90.0, 120.0, 140.0, 162.0]:
        r = run_case(100, R_star=1.0, theta_eq=theta, amp=1.5, N_equil=2000)
        r['study'] = 'contact_angle'
        r['theta_eq'] = theta
        results.append(r)
        err = r['theta_error']
        print(f"  θ={theta:.0f}°: θ_actual={r['theta_actual']:.1f}° "
              f"(err={err:.1f}°) stable={r['stable']}")

    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed:.0f}s")

    # Save
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             '..', 'results', 'geo_amp_calibration.json')
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Results saved to {save_path}")

    # Summary table
    print("\n" + "="*60)
    print("  SUMMARY")
    print("="*60)
    for study in ['resolution', 'amp_sweep', 'curvature', 'contact_angle']:
        subset = [r for r in results if r['study'] == study]
        if subset:
            errors = [r['theta_error'] for r in subset if r['stable']]
            if errors:
                print(f"  {study:20s}: avg_err={np.mean(errors):.1f}° "
                      f"max_err={max(errors):.1f}° "
                      f"({sum(1 for r in subset if r['stable'])}/{len(subset)} stable)")


if __name__ == '__main__':
    main()
