#!/usr/bin/env python3
"""AC-LBM Film Thickness Sweep for Paper v12 Figures.

Runs concave/convex substrate simulations with Monitor class to record
air film thickness h_min(t), drainage time tau, and initial thickness h0.

Output: CSV files compatible with render_v12_figures.py load_all_curvature_data().

Hardware: NVIDIA CMP 30HX (6GB), ~4min/case at 150^3.
"""
import sys
import os
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)

import numpy as np
import torch
import time
import csv

from lbm.fe_config import FEConfig
from lbm.fe_ac_solver import AllenCahnSolver
from lbm.fe_droplet import create_fe_droplet_with_impact
from geometry.substrate import create_substrate, create_substrate_with_fraction
from io_utils.monitor import Monitor

torch.cuda.empty_cache()

# --- Physical parameters (same as thesis sweep, validated config) ---
D0 = 45.0
R_drop = D0 / 2.0
rho_l = 1.0
rho_g = 1.0 / 828.0
xi = 3.0           # sharper interface for thin film resolution
tau = 0.53
U0 = -0.05
GAP = 20.0         # enlarged gap for air film capture (was 2.0)
N_STEPS = 4000     # more steps for full drainage
RECORD_EVERY = 20  # record every 20 steps for high time resolution

# Output directory for CSV data
OUT_DIR = os.path.join(_project_root, 'results', 'film_thickness')
os.makedirs(OUT_DIR, exist_ok=True)


def get_surface_height(solid, nx, ny, nz):
    height = np.zeros((nx, ny), dtype=int)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz - 1, -1, -1):
                if solid[i, j, k]:
                    height[i, j] = k
                    break
    return height


def run_film_case(substrate_type, R_star, We=7.9, theta_eq=162.0, amp=1.8,
                  N=N_STEPS, n_base=150, record_interval=RECORD_EVERY,
                  gap=GAP):
    """Run a single case and record film thickness time series.

    Returns dict with: t_ms, h_min_um, h_cen_um, tau, h0, label, k_history
    """
    sigma = rho_l * U0 ** 2 * D0 / We
    beta = 12.0 * sigma / xi
    kappa = beta * xi ** 2 / 8.0
    M = 0.02 / beta

    is_flat = substrate_type == 'flat' or R_star is None

    # Grid sizing — use enlarged gap for film capture
    if is_flat:
        nz_min = int(R_drop + gap + 2 * R_drop + 20)
        nx, ny, nz = n_base, n_base, nz_min
    else:
        R_g = abs(R_star) * R_drop
        nx, ny = n_base, n_base
        nz = int(R_g + gap + R_drop + 2 * R_drop + 20)
        nz = min(nz, 350)

    config = FEConfig(
        nx=nx, ny=ny, nz=nz,
        rho_l=rho_l, rho_g=rho_g,
        sigma=sigma, xi=xi, beta=beta, kappa=kappa,
        M=M, tau_l=tau, tau_g=tau, tau_h=tau + 0.04,
        theta_eq=theta_eq, device='cuda',
        max_steps=N, output_interval=N + 1,
        g_force=(0.0, 0.0, 0.0),
        dx=1e-6,  # 1 micron per lattice unit
    )

    solver = AllenCahnSolver(
        config, dtype=torch.float32,
        stab_mode='fakhari',
        boundary_relax=0.0,
        geometric_wetting=(amp > 0),
        geo_amplification=amp,
    )

    # Create substrate
    solid, fraction = create_substrate_with_fraction(
        nx, ny, nz, substrate_type=substrate_type,
        R_star=R_star, R_d=R_drop)
    solver.set_solid(solid, solid_fraction=fraction)

    # Place droplet
    surface_height = get_surface_height(solid, nx, ny, nz)
    cx, cy = nx / 2.0, ny / 2.0
    if substrate_type == 'concave':
        ridge_top = int(surface_height.max())
    else:
        ridge_top = surface_height[nx // 2, ny // 2]
    cz = ridge_top + gap + R_drop
    max_cz = nz - R_drop - 2
    if cz > max_cz:
        cz = max_cz

    C_init, _, u_init = create_fe_droplet_with_impact(
        nx, ny, nz, center=(cx, cy, cz), radius=R_drop, xi=xi,
        rho_l=rho_l, rho_g=rho_g, u_impact=(0.0, 0.0, U0)
    )
    solver.init_fields(C_init, u_init)

    # Monitor for film thickness
    monitor = Monitor(config)

    # Time series storage
    t_list = []
    h_min_list = []
    h_cen_list = []
    k_list = []
    z_com_list = []

    stable = True
    t0 = time.time()

    for step in range(1, N + 1):
        solver.step()

        if step % record_interval == 0:
            # Record monitor data (film thickness, contact angle, etc.)
            monitor.record(solver, step=step)

            # Get phi for geometric measurements
            phi_np = solver.phi.detach().cpu().numpy()
            solid_np = solver.solid.cpu().numpy()
            interface = (phi_np > 0.5) & ~solid_np

            # Film thickness from monitor (in physical units)
            film = monitor.history['film_thickness'][-1]
            t_phys = solver.t  # dimensionless time

            # Convert to physical: t[ms] = t_phys * dx / |U0| * 1000
            # Actually solver.t is in lattice units, t_phys = step * dt
            # dt in LBM = 1 (lattice time), physical dt = dx / c_s
            # For comparison with Basilisk, use lattice time directly
            # and convert film thickness to micrometers
            t_ms = step * config.dx / abs(U0) * 1e3  # rough conversion

            # Also compute center film thickness
            mid_i, mid_j = nx // 2, ny // 2
            rho_center = solver.rho.cpu().numpy()

            # k measurement
            if interface.any():
                coords = np.argwhere(interface)
                Dx = float(coords[:, 0].max() - coords[:, 0].min() + 1)
                Dy = float(coords[:, 1].max() - coords[:, 1].min() + 1)
                k_val = Dx / Dy if Dy > 0 else 0
                z_com = float(coords[:, 2].mean())
            else:
                Dx = Dy = k_val = 0
                z_com = 0.0

            h_um = film * 1e6 if film != float('inf') else float('nan')
            t_list.append(t_ms)
            h_min_list.append(h_um)
            h_cen_list.append(h_um)  # simplified
            k_list.append(k_val)
            z_com_list.append(z_com)

            if np.isnan(phi_np).any() or phi_np.max() < 0.01:
                stable = False
                print(f"    UNSTABLE at step {step}")
                break

    elapsed = time.time() - t0
    mem_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    torch.cuda.reset_peak_memory_stats()

    label = f"{substrate_type}_R{R_star}" if R_star else "flat"
    tag = "OK" if stable else "FAIL"

    # Summary
    h_arr = np.array(h_min_list)
    h_valid = h_arr[~np.isnan(h_arr)]
    tau_val = float('nan')
    if len(h_valid) > 0:
        # Drainage time: time when h_min first < some threshold
        # Use 10% of initial thickness as rough threshold
        h0 = h_valid[0]
        threshold = h0 * 0.1
        below = np.where(h_valid < threshold)[0]
        if len(below) > 0:
            tau_val = t_list[below[0]]
        else:
            tau_val = t_list[-1]

    h0_val = h_valid[0] if len(h_valid) > 0 else float('nan')

    print(f"  {label:>25} We={We:5.1f} R*={R_star} | "
          f"h0={h0_val:.1f}μm tau={tau_val:.3f}ms "
          f"steps={len(t_list)} [{tag}] "
          f"[{elapsed:.0f}s, {mem_mb:.0f}MB] "
          f"grid={nx}x{ny}x{nz}")

    # Save CSV
    csv_path = os.path.join(OUT_DIR, f"{label}_film.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['t_ms', 'h_min_um', 'h_cen_um', 'k', 'z_com'])
        for i in range(len(t_list)):
            writer.writerow([t_list[i], h_min_list[i], h_cen_list[i],
                             k_list[i], z_com_list[i]])

    del solver
    torch.cuda.empty_cache()

    return {
        'label': label,
        'substrate_type': substrate_type,
        'R_star': R_star,
        'We': We,
        'theta_eq': theta_eq,
        'amp': amp,
        'h0_um': h0_val,
        'tau_ms': tau_val,
        'stable': stable,
        'grid': f"{nx}x{ny}x{nz}",
        'elapsed_s': round(elapsed, 1),
        'csv_path': csv_path,
    }


def main():
    results = []

    # === Curved substrate sweep for fig8/fig9/fig10 ===
    cases = [
        # (substrate_type, R_star) — matching Basilisk R* values
        ('concave', 0.30),
        ('concave', 0.50),
        ('concave', 0.70),
        ('concave', 0.90),
        ('concave', 1.00),
        ('concave', 2.00),
        ('convex', 0.30),
        ('convex', 0.50),
        ('convex', 0.70),
        ('convex', 0.90),
        ('convex', 1.00),
        ('convex', 2.00),
    ]

    print("=" * 70)
    print("  LBM Film Thickness Sweep for Paper v12")
    print(f"  {len(cases)} cases, gap={GAP}, xi={xi}, ~12min/case")
    print("=" * 70)

    for substrate_type, R_star in cases:
        print(f"\n>>> {substrate_type} R*={R_star}")
        r = run_film_case(substrate_type, R_star, N=N_STEPS, n_base=150,
                          record_interval=RECORD_EVERY, gap=GAP)
        results.append(r)

    # Save summary
    summary_path = os.path.join(OUT_DIR, 'sweep_summary.csv')
    with open(summary_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'label', 'substrate_type', 'R_star', 'We', 'theta_eq', 'amp',
            'h0_um', 'tau_ms', 'stable', 'grid', 'elapsed_s', 'csv_path'])
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    print(f"\n{'=' * 70}")
    print(f"  Done. Summary: {summary_path}")
    print(f"  CSV data: {OUT_DIR}/")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
