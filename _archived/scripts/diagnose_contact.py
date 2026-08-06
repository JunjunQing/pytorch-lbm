#!/usr/bin/env python3
"""Diagnostic: test contact detection and measure actual contact time.

Runs a single case with dense output to understand the droplet dynamics.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from lbm.fe_config import FEConfig
from lbm.fe_ac_solver import AllenCahnSolver
from lbm.fe_droplet import create_fe_droplet_with_impact
from geometry.substrate import create_substrate_with_fraction

D0 = 45.0; R_drop = D0 / 2.0; rho_l = 1.0; rho_g = 1.0 / 828.0
xi = 4.0; tau = 0.53; U0 = -0.05
theta_eq = 162.0

def get_surface_height(solid, nx, ny, nz):
    height = np.zeros((nx, ny), dtype=int)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz - 1, -1, -1):
                if solid[i, j, k]:
                    height[i, j] = k
                    break
    return height

def run_diagnostic(R_star=1.0, We=7.9, N=4000, n_base=100):
    sigma = rho_l * U0 ** 2 * D0 / We
    beta = 12.0 * sigma / xi
    kappa = beta * xi ** 2 / 8.0
    M = 0.02 / beta

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
        max_steps=N, output_interval=N + 1,
        g_force=(0.0, 0.0, 0.0),
    )

    solver = AllenCahnSolver(
        config, dtype=torch.float32,
        stab_mode='fakhari', boundary_relax=0.0,
        geometric_wetting=True, geo_amplification=1.5,
    )

    solid, fraction = create_substrate_with_fraction(
        nx, ny, nz, substrate_type='ridge', R_star=R_star, R_d=R_drop)
    solver.set_solid(solid, solid_fraction=fraction)

    surface_height = get_surface_height(solid, nx, ny, nz)
    ridge_top = surface_height[nx // 2, ny // 2]
    cx, cy = nx / 2.0, ny / 2.0
    gap = 2.0
    cz = ridge_top + gap + R_drop

    print(f"  ridge_top={ridge_top}, cz={cz:.1f}, droplet_bottom={cz - R_drop:.1f}")
    print(f"  gap_to_ridge = {cz - R_drop - ridge_top:.1f} lu")

    C_init, _, u_init = create_fe_droplet_with_impact(
        nx, ny, nz, center=(cx, cy, cz), radius=R_drop, xi=xi,
        rho_l=rho_l, rho_g=rho_g, u_impact=(0.0, 0.0, U0))
    solver.init_fields(C_init, u_init)

    # Track z_min (lowest interface point) — better contact indicator than z_com
    first_contact_step = 0
    contact_time = 0
    z_min_threshold = ridge_top + 3  # within 3 lu of ridge surface

    print(f"  z_min_threshold = {z_min_threshold}")
    print(f"\n  {'step':>5} {'z_com':>7} {'z_min':>7} {'z_max':>7} {'Dx':>4} {'Dy':>4} {'k':>6} {'contact':>8}")

    for step in range(1, N + 1):
        solver.step()

        if step % 50 == 0:
            phi_np = solver.phi.detach().cpu().numpy()
            solid_np = solver.solid.cpu().numpy()
            interface = (phi_np > 0.5) & ~solid_np

            if interface.any():
                coords = np.argwhere(interface)
                z_com = float(coords[:, 2].mean())
                z_min = float(coords[:, 2].min())
                z_max = float(coords[:, 2].max())
                Dx = float(coords[:, 0].max() - coords[:, 0].min() + 1)
                Dy = float(coords[:, 1].max() - coords[:, 1].min() + 1)
                k = Dx / Dy if Dy > 0 else 0
            else:
                z_com = z_min = z_max = 0
                Dx = Dy = k = 0

            # Better contact detection: z_min approaches ridge_top
            if first_contact_step == 0 and z_min <= z_min_threshold:
                first_contact_step = step
                print(f"  >>> FIRST CONTACT at step {step}, z_min={z_min:.1f}")

            if first_contact_step > 0 and contact_time == 0:
                if z_min > z_min_threshold + 10:
                    contact_time = step - first_contact_step
                    print(f"  >>> LIFT-OFF at step {step}, z_min={z_min:.1f}, ct={contact_time}")

            status = ""
            if first_contact_step > 0 and contact_time == 0:
                status = " *** CONTACTED ***"

            print(f"  {step:5d} {z_com:7.1f} {z_min:7.1f} {z_max:7.1f} "
                  f"{Dx:4.0f} {Dy:4.0f} {k:6.4f} {status}")

    print(f"\n  Result: first_contact={first_contact_step}, contact_time={contact_time}")
    del solver
    torch.cuda.empty_cache()
    return contact_time

if __name__ == '__main__':
    print("=" * 60)
    print("  Contact Detection Diagnostic")
    print("=" * 60)
    ct = run_diagnostic(R_star=1.0, We=7.9, N=4000, n_base=100)
    print(f"\n  Final contact_time = {ct}")
