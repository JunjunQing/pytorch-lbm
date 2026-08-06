#!/usr/bin/env python3
"""Phase Diagram: Weber Number × Curvature Ratio for Asymmetric Droplet Spreading.

Idea 1 from research pipeline: Map the parameter space (We × D/D₀) to construct
a phase diagram of spreading regimes (symmetric → asymmetric → splashing → rebound).

Target: Validate against Liu et al. (2015) Nature Communications data.
Hardware: GTX 1080 (8GB), ~4min/case at 150^3.
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
from geometry.substrate import create_substrate_with_fraction

torch.cuda.empty_cache()

# --- Physical parameters (matching Liu et al. 2015) ---
D0 = 45.0
R_drop = D0 / 2.0
rho_l = 1.0
rho_g = 1.0 / 828.0
xi = 4.0
tau = 0.53
U0_base = -0.05  # Reference velocity for We=7.9

# Contact angle: Liu 2015 used superhydrophobic surfaces
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


def compute_we(U0, D0_val, sigma):
    """Compute Weber number from impact velocity."""
    return rho_l * U0 ** 2 * D0_val / sigma


def run_single_case(R_star, We, amp=1.5, N=3000, n_base=150,
                    dtype=torch.float32, label=None):
    """Run one case: ridge substrate at given R_star and We."""
    # Compute U0 from We
    sigma = rho_l * U0_base ** 2 * D0 / 7.9  # Reference sigma at We=7.9
    U0 = -np.sqrt(We * sigma / (rho_l * D0))

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
        config, dtype=dtype,
        stab_mode='fakhari',
        boundary_relax=0.0,
        geometric_wetting=(amp > 0),
        geo_amplification=amp,
    )

    solid, fraction = create_substrate_with_fraction(
        nx, ny, nz, substrate_type='ridge',
        R_star=R_star, R_d=R_drop)
    solver.set_solid(solid, solid_fraction=fraction)

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

    # Track metrics
    max_k = 0.0
    max_k_step = 0
    Dx_peak = Dy_peak = 0.0
    contact_time = 0
    first_contact_step = 0
    z_com_min = 1e10
    z_com_min_step = 0
    z_com_history = []
    k_history = []
    stable = True
    t0 = time.time()

    # Contact detection threshold: z_min of interface near ridge
    z_contact_thresh = ridge_top + 3

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
                z_min = float(coords[:, 2].min())
            else:
                Dx = Dy = Dz = k_val = 0
                z_com = 0.0
                z_min = 0.0

            z_com_history.append({'step': step, 'z_com': z_com, 'z_min': z_min})
            k_history.append({'step': step, 'k': k_val, 'Dx': Dx, 'Dy': Dy})

            if k_val > max_k:
                max_k = k_val
                max_k_step = step
                Dx_peak = Dx
                Dy_peak = Dy

            # Contact detection: interface z_min reaches near ridge surface
            if first_contact_step == 0 and z_min <= z_contact_thresh:
                first_contact_step = step

            # Track z_com minimum (maximum compression point)
            if first_contact_step > 0 and z_com < z_com_min:
                z_com_min = z_com
                z_com_min_step = step

            # Contact time: from first contact to z_com minimum
            # (most robust metric — doesn't depend on lift-off detection)
            if first_contact_step > 0 and z_com_min_step > first_contact_step:
                # Wait until z_com clearly rises after minimum
                if z_com > z_com_min + 1.0:
                    contact_time = z_com_min_step - first_contact_step

            if np.isnan(phi_np).any() or phi_np.max() < 0.01:
                stable = False
                break

    elapsed = time.time() - t0
    mem_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    torch.cuda.reset_peak_memory_stats()

    if label is None:
        label = f"ridge_R{R_star:.1f}_We{We:.1f}"

    tag = "OK" if stable else "FAIL"
    print(f"  {label:>30} R*={R_star:5.2f} We={We:5.1f} | "
          f"k={max_k:.4f} Dx={Dx_peak:.0f} Dy={Dy_peak:.0f} "
          f"ct={contact_time:5d} [{tag}] "
          f"[{elapsed:.0f}s, {mem_mb:.0f}MB]")

    del solver
    torch.cuda.empty_cache()

    return {
        'label': label,
        'R_star': R_star,
        'We': We,
        'theta_eq': theta_eq,
        'amp': amp,
        'max_k': max_k,
        'max_k_step': max_k_step,
        'Dx': Dx_peak,
        'Dy': Dy_peak,
        'contact_time': contact_time,
        'first_contact_step': first_contact_step,
        'stable': stable,
        'elapsed_s': round(elapsed, 1),
        'mem_mb': round(mem_mb, 0),
        'grid': f"{nx}x{ny}x{nz}",
        'U0': round(abs(U0), 6),
        'k_history': k_history,
    }


def main():
    results = []

    # =================================================================
    # Phase Diagram: We × R* (D/D₀)
    #
    # Liu et al. (2015) key data points:
    #   We ≈ 7.9, D/D₀ = 0.5-2.3, superhydrophobic (θ≈162°)
    #   Contact time minimized at D/D₀ ≈ 1.0
    #   Critical We for breakup: We_c depends on D/D₀
    #
    # Our sweep:
    #   We: [3, 5, 7.9, 10, 15, 20, 30]  (7 values)
    #   R*: [0.5, 0.7, 1.0, 1.5, 2.0, 3.0]  (6 values)
    #   Total: 42 cases × ~4min = ~2.8 hours on GTX 1080
    # =================================================================
    We_values = [3.0, 5.0, 7.9, 10.0, 15.0, 20.0, 30.0]
    R_star_values = [0.5, 0.7, 1.0, 1.5, 2.0, 3.0]

    print("=" * 70)
    print("  PHASE DIAGRAM: We × R* (D/D₀) for Asymmetric Spreading")
    print("  Target: Liu et al. (2015) Nature Communications validation")
    print(f"  Grid: {len(We_values)} We values × {len(R_star_values)} R* values "
          f"= {len(We_values) * len(R_star_values)} cases")
    print("  Estimated time: ~2.8 hours on GTX 1080")
    print("=" * 70)

    # Amp calibration: use amp=1.5 for all cases (conservative default)
    # In production, calibrate per R* like in run_thesis_sweep.py
    amp = 1.5

    for We in We_values:
        for R_star in R_star_values:
            label = f"pd_We{We:.1f}_R{R_star:.1f}"
            r = run_single_case(
                R_star=R_star, We=We, amp=amp,
                N=3000, n_base=150, dtype=torch.float32,
                label=label,
            )
            results.append(r)

    # =================================================================
    # Summary
    # =================================================================
    print("\n" + "=" * 70)
    print("  PHASE DIAGRAM RESULTS")
    print("=" * 70)

    print(f"\n  {'Label':>30} {'R*':>5} {'We':>6} {'k_max':>8} "
          f"{'Dx':>4} {'Dy':>4} {'ct':>5} {'Status':>6}")
    print("  " + "-" * 75)

    for r in results:
        status = "OK" if r['stable'] else "FAIL"
        print(f"  {r['label']:>30} {r['R_star']:5.2f} {r['We']:6.1f} "
              f"{r['max_k']:8.4f} {r['Dx']:4.0f} {r['Dy']:4.0f} "
              f"{r['contact_time']:5d} {status:>6}")

    # Build 2D grid for phase diagram
    print("\n  Phase Diagram (k_max = Dx/Dy, >1 means asymmetric):")
    print(f"  {'R*':>6}", end="")
    for We in We_values:
        print(f"  We={We:5.1f}", end="")
    print()
    print("  " + "-" * (7 + 10 * len(We_values)))

    for R_star in R_star_values:
        print(f"  {R_star:6.2f}", end="")
        for We in We_values:
            # Find matching result
            match = [r for r in results
                     if abs(r['R_star'] - R_star) < 0.01
                     and abs(r['We'] - We) < 0.01]
            if match:
                k = match[0]['max_k']
                print(f"  {k:8.4f}", end="")
            else:
                print(f"  {'---':>8}", end="")
        print()

    # Save
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            '..', 'results')
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, 'phase_diagram_results.json')
    save_data = [{k: v for k, v in r.items() if k != 'k_history'}
                 for r in results]
    with open(save_path, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\n  Results saved to {save_path}")

    history_path = os.path.join(save_dir, 'phase_diagram_k_history.json')
    history_data = {r['label']: r.get('k_history', []) for r in results}
    with open(history_path, 'w') as f:
        json.dump(history_data, f, indent=2)
    print(f"  Time histories saved to {history_path}")


if __name__ == '__main__':
    main()
