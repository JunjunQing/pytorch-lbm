#!/usr/bin/env python3
"""Study 9: Dual-droplet coalescence on curved surfaces.

Machine: bigblack (NVIDIA GTX 1080 8GB) — runs after Studies 3a/3b/5.

Physical motivation: In conformal printing, adjacent nozzles deposit droplets
simultaneously. Understanding coalescence dynamics on curved surfaces is
critical for print quality and line width control.

Studies:
  9a: Substrate type comparison (d/D=1.2, We=7.9) — 4 cases
  9b: Spacing sweep on ridge R*=1.0 — 4 cases
  9c: We sweep on ridge R*=1.0 — 4 cases

Total: 12 cases × ~15min/case ≈ 3h

Hardware: NVIDIA GTX 1080 8GB (wider grid needed: ~250×150×nz)
Usage: bash scripts/run_thesis_dual_droplet.sh
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
from lbm.fe_droplet import create_multi_droplet_with_impact
from geometry.substrate import create_substrate, create_substrate_with_fraction

DEVICE = 'cuda'
torch.cuda.empty_cache()
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# --- Physical parameters (same as single-droplet) ---
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


def run_dual_case(substrate_type='ridge', R_star=None, We=7.9,
                  theta_eq=162.0, amp=1.8, spacing_ratio=1.2,
                  N=4000, n_base=150, dtype=torch.float32, label=None):
    """Run dual-droplet impact simulation.

    Two droplets placed symmetrically along x-axis with center-to-center
    distance = spacing_ratio * D0.

    Parameters
    ----------
    spacing_ratio : float
        Center-to-center distance / D0. 1.0 = touching, 1.2 = slight gap.
    """
    sigma = rho_l * U0 ** 2 * D0 / We
    beta = 12.0 * sigma / xi
    kappa = beta * xi ** 2 / 8.0
    M = 0.02 / beta

    tau_l = tau
    tau_g = tau

    is_flat = substrate_type == 'flat' or R_star is None

    if is_flat:
        nz_min = int(R_drop + 2 + 2 * R_drop + 15)
        # Wider x to fit two droplets: spacing*D0 + 2*R_drop margin
        nx = int(spacing_ratio * D0 + D0 + 20)
        ny = n_base
        nz = nz_min
    else:
        R_g = abs(R_star) * R_drop
        nx = int(spacing_ratio * D0 + D0 + 20)
        ny = n_base
        nz = int(R_g + 2 + R_drop + 2 * R_drop + 15)
        nz = min(nz, 300)

    config = FEConfig(
        nx=nx, ny=ny, nz=nz,
        rho_l=rho_l, rho_g=rho_g,
        sigma=sigma, xi=xi, beta=beta, kappa=kappa,
        M=M, tau_l=tau_l, tau_g=tau_g, tau_h=tau_l + 0.04,
        theta_eq=theta_eq, device=DEVICE,
        max_steps=N, output_interval=N + 1,
        g_force=(0.0, 0.0, 0.0),
    )

    solver = AllenCahnSolver(
        config, dtype=dtype,
        stab_mode='fakhari',
        boundary_relax=0.0,
        geometric_wetting=(amp > 0),
        geo_amplification=amp,
    )

    # Create substrate
    if is_flat:
        solid = create_substrate(nx, ny, nz, substrate_type='flat',
                                 R_star=None, R_d=R_drop)
        solver.set_solid(solid)
    else:
        solid, fraction = create_substrate_with_fraction(
            nx, ny, nz, substrate_type=substrate_type,
            R_star=R_star, R_d=R_drop)
        solver.set_solid(solid, solid_fraction=fraction)

    # Determine droplet placement
    surface_height = get_surface_height(solid, nx, ny, nz)
    cx_total = nx / 2.0
    cy = ny / 2.0

    if substrate_type == 'concave':
        ridge_top = int(surface_height.max())
    else:
        ridge_top = surface_height[nx // 2, ny // 2]
    gap = 2.0
    cz = ridge_top + gap + R_drop
    max_cz = nz - R_drop - 2
    if cz > max_cz:
        cz = max_cz

    # Two droplets symmetric along x-axis
    d_center = spacing_ratio * D0  # center-to-center distance
    cx1 = cx_total - d_center / 2.0
    cx2 = cx_total + d_center / 2.0

    # Ensure droplets are within grid
    cx1 = max(R_drop + 1, cx1)
    cx2 = min(nx - R_drop - 1, cx2)

    centers = [(cx1, cy, cz), (cx2, cy, cz)]
    radii = [R_drop, R_drop]
    u_impacts = [(0.0, 0.0, U0), (0.0, 0.0, U0)]

    C_init, _, u_init = create_multi_droplet_with_impact(
        nx, ny, nz, centers=centers, radii=radii,
        xi=xi, rho_l=rho_l, rho_g=rho_g, u_impacts=u_impacts,
    )
    solver.init_fields(C_init, u_init)

    # Run simulation — track Dx, Dy, coalescence
    max_k = 0.0
    max_k_step = 0
    Dx_peak = 0.0
    Dy_peak = 0.0
    k_history = []
    coalescence_step = None
    stable = True
    t0 = time.time()

    for step in range(1, N + 1):
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
                z_com = float(coords[:, 2].mean())

                # Detect coalescence: check if interface is connected
                # along x-axis (single connected blob vs two separate)
                x_coords = coords[:, 0]
                x_range = x_coords.max() - x_coords.min()
                # If x_range < 2*R_drop + spacing, they've merged
                merged = x_range < (d_center + R_drop * 0.5)
            else:
                Dx = Dy = Dz = k_val = 0
                z_com = 0.0
                merged = False

            if coalescence_step is None and merged:
                coalescence_step = step

            k_history.append({
                'step': step, 'Dx': Dx, 'Dy': Dy, 'Dz': Dz,
                'k': k_val, 'z_com': z_com, 'merged': merged,
            })
            if k_val > max_k:
                max_k = k_val
                max_k_step = step
                Dx_peak = Dx
                Dy_peak = Dy
            if np.isnan(phi_np).any() or phi_np.max() < 0.01:
                stable = False
                break

    elapsed = time.time() - t0
    mem_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    torch.cuda.reset_peak_memory_stats()

    if label is None:
        if is_flat:
            label = f"dual_flat_d{spacing_ratio:.1f}"
        else:
            label = f"dual_{substrate_type}_R{R_star:.1f}_d{spacing_ratio:.1f}"

    coal_str = f"merge@{coalescence_step}" if coalescence_step else "separate"
    tag = "OK" if stable else "FAIL"
    print(f"  {label:>35} We={We:5.1f} amp={amp:.1f} d/D={spacing_ratio:.1f} | "
          f"k={max_k:.4f} Dx={Dx_peak:.0f} Dy={Dy_peak:.0f} "
          f"{coal_str} [{tag}] "
          f"[{elapsed:.0f}s, {mem_mb:.0f}MB] "
          f"grid={nx}x{ny}x{nz}")

    del solver
    torch.cuda.empty_cache()

    return {
        'label': label, 'substrate_type': substrate_type,
        'R_star': R_star, 'We': We, 'theta_eq': theta_eq, 'amp': amp,
        'spacing_ratio': spacing_ratio,
        'max_k': max_k, 'max_k_step': max_k_step,
        'Dx': Dx_peak, 'Dy': Dy_peak,
        'coalescence_step': coalescence_step,
        'stable': stable, 'grid': f"{nx}x{ny}x{nz}",
        'elapsed_s': round(elapsed, 1), 'mem_mb': round(mem_mb, 0),
        'machine': 'bigblack_dual', 'k_history': k_history,
    }


def main():
    results = []

    # =================================================================
    # STUDY 9a: Substrate type comparison (d/D=1.2, We=7.9)
    # =================================================================
    print("\n" + "=" * 70)
    print("  STUDY 9a: Substrate type comparison (d/D=1.2, We=7.9)")
    print("=" * 70)

    for sub_type, R_star, amp_val in [
        ('flat', None, 0.0),
        ('ridge', 1.0, 1.8),
        ('convex', 1.0, 1.8),
        ('concave', 1.0, 1.8),
    ]:
        r = run_dual_case(
            substrate_type=sub_type, R_star=R_star,
            We=7.9, theta_eq=162.0, amp=amp_val,
            spacing_ratio=1.2, N=4000, n_base=150,
            label=f"s9a_dual_{sub_type}_R{R_star or 'flat'}",
        )
        results.append(r)

    # =================================================================
    # STUDY 9b: Spacing sweep on ridge R*=1.0 (We=7.9)
    # =================================================================
    print("\n" + "=" * 70)
    print("  STUDY 9b: Spacing sweep on ridge R*=1.0")
    print("=" * 70)

    for d_ratio in [1.0, 1.2, 1.5, 2.0]:
        r = run_dual_case(
            substrate_type='ridge', R_star=1.0,
            We=7.9, theta_eq=162.0, amp=1.8,
            spacing_ratio=d_ratio, N=4000, n_base=150,
            label=f"s9b_ridge_R1.0_d{d_ratio:.1f}",
        )
        results.append(r)

    # =================================================================
    # STUDY 9c: We sweep on ridge R*=1.0 (d/D=1.2)
    # =================================================================
    print("\n" + "=" * 70)
    print("  STUDY 9c: We sweep on ridge R*=1.0 (d/D=1.2)")
    print("=" * 70)

    for we in [5.0, 10.0, 15.0, 20.0]:
        r = run_dual_case(
            substrate_type='ridge', R_star=1.0,
            We=we, theta_eq=162.0, amp=1.8,
            spacing_ratio=1.2, N=4000, n_base=150,
            label=f"s9c_ridge_R1.0_We{we}",
        )
        results.append(r)

    # =================================================================
    # Summary
    # =================================================================
    print("\n" + "=" * 70)
    print("  STUDY 9: DUAL-DROPLET RESULTS SUMMARY")
    print("=" * 70)

    print(f"\n  {'Label':>35} {'Type':>8} {'R*':>5} {'We':>6} "
          f"{'d/D':>5} {'k_max':>8} {'Dx':>4} {'Dy':>4} "
          f"{'Merge':>8} {'Status':>6}")
    print("  " + "-" * 100)

    for r in results:
        status = "OK" if r['stable'] else "FAIL"
        R_str = f"{r['R_star']:.1f}" if r['R_star'] is not None else "---"
        coal = f"@{r['coalescence_step']}" if r['coalescence_step'] else "separate"
        print(f"  {r['label']:>35} {r['substrate_type']:>8} {R_str:>5} "
              f"{r['We']:6.1f} {r['spacing_ratio']:5.1f} "
              f"{r['max_k']:8.4f} {r['Dx']:4.0f} {r['Dy']:4.0f} "
              f"{coal:>8} {status:>6}")

    # Save results
    save_dir = os.path.join(_project_root, 'results')
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, 'thesis_dual_droplet_results.json')
    save_data = []
    for r in results:
        r_copy = {k: v for k, v in r.items() if k != 'k_history'}
        save_data.append(r_copy)
    with open(save_path, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\n  Results saved to {save_path}")

    history_path = os.path.join(save_dir, 'thesis_dual_droplet_history.json')
    history_data = {r['label']: r.get('k_history', []) for r in results}
    with open(history_path, 'w') as f:
        json.dump(history_data, f, indent=2)
    print(f"  Time histories saved to {history_path}")


if __name__ == "__main__":
    main()
