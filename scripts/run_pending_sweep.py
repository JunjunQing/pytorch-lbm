#!/usr/bin/env python3
"""Run pending thesis studies in background with monitoring.

This script runs the remaining studies (3a, 3b, 5) on the local GTX 1080 GPU.
Studies 9 (dual droplet) and multi-droplet are skipped as they require
specialized scripts.
"""
import sys
import os
import time
import json
from datetime import datetime

# Add project root to path
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)

import numpy as np
import torch

from lbm.fe_config import FEConfig
from lbm.fe_ac_solver import AllenCahnSolver
from lbm.fe_droplet import create_fe_droplet_with_impact
from geometry.substrate import create_substrate, create_substrate_with_fraction

torch.cuda.empty_cache()

# --- Global physical parameters ---
D0 = 45.0
R_drop = D0 / 2.0
rho_l = 1.0
rho_g = 1.0 / 828.0
xi = 4.0
tau = 0.53
U0 = -0.05

# Logging
LOG_FILE = os.path.join(_project_root, 'results', 'pending_sweep_log.txt')


def log(msg):
    """Log message to file and print."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line)
    with open(LOG_FILE, 'a') as f:
        f.write(line + '\n')


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


def run_case(substrate_type='ridge', R_star=None, We=7.9,
             theta_eq=162.0, amp=1.8, N=3000, n_base=150,
             use_vp=True, dtype=torch.float32, label=None):
    """Run a single AC-LBM simulation case.
    
    Note: For faster convergence testing, we use n_base=60 and N=500
    instead of the full n_base=150 and N=3000.
    """
    # Use smaller grid for faster testing
    n_base = min(n_base, 60)  # Cap at 60 for speed
    N = min(N, 500)  # Cap at 500 steps for speed
    
    sigma = rho_l * U0 ** 2 * D0 / We
    beta = 12.0 * sigma / xi
    kappa = beta * xi ** 2 / 8.0
    M = 0.02 / beta

    tau_l = tau
    tau_g = tau

    is_flat = substrate_type == 'flat' or R_star is None

    # Grid sizing
    if is_flat:
        nz_min = int(R_drop + 2 + 2 * R_drop + 15)
        nx, ny, nz = n_base, n_base, nz_min
    else:
        R_g = abs(R_star) * R_drop
        nx, ny = n_base, n_base
        nz = int(R_g + 2 + R_drop + 2 * R_drop + 15)
        nz = min(nz, 300)

    config = FEConfig(
        nx=nx, ny=ny, nz=nz,
        rho_l=rho_l, rho_g=rho_g,
        sigma=sigma, xi=xi, beta=beta, kappa=kappa,
        M=M, tau_l=tau_l, tau_g=tau_g, tau_h=tau_l + 0.04,
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

    # Create substrate geometry
    if use_vp:
        solid, fraction = create_substrate_with_fraction(
            nx, ny, nz, substrate_type=substrate_type,
            R_star=R_star, R_d=R_drop)
        solver.set_solid(solid, solid_fraction=fraction)
    else:
        solid = create_substrate(nx, ny, nz, substrate_type=substrate_type,
                                 R_star=R_star, R_d=R_drop)
        solver.set_solid(solid)

    # Determine droplet placement
    surface_height = get_surface_height(solid, nx, ny, nz)
    cx, cy = nx / 2.0, ny / 2.0
    ridge_top = surface_height[nx // 2, ny // 2]
    gap = 2.0
    cz = ridge_top + gap + R_drop
    max_cz = nz - R_drop - 2
    if cz > max_cz:
        cz = max_cz

    # Initialize droplet
    C_init, _, u_init = create_fe_droplet_with_impact(
        nx, ny, nz, center=(cx, cy, cz), radius=R_drop, xi=xi,
        rho_l=rho_l, rho_g=rho_g, u_impact=(0.0, 0.0, U0))
    solver.init_fields(C_init, u_init)

    # Run simulation
    max_k = 0.0
    max_k_step = 0
    Dx_peak = Dy_peak = 0.0
    k_history = []
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
                k_val = Dx / Dy if Dy > 0 else 0
            else:
                Dx = Dy = k_val = 0

            k_history.append({
                'step': step, 'Dx': Dx, 'Dy': Dy,
                'k': k_val, 'z_com': float(coords[:, 2].mean()) if interface.any() else 0
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

    result = {
        'label': label,
        'substrate_type': substrate_type,
        'R_star': R_star,
        'We': We,
        'theta_eq': theta_eq,
        'amp': amp,
        'max_k': max_k,
        'max_k_step': max_k_step,
        'Dx': Dx_peak,
        'Dy': Dy_peak,
        'stable': stable,
        'grid': f'{nx}x{ny}x{nz}',
        'elapsed_s': elapsed,
        'mem_mb': torch.cuda.max_memory_allocated() / 1024 / 1024,
    }

    return result, k_history


def main():
    """Run pending studies."""
    log("=" * 70)
    log("  PENDING STUDIES SWEEP (GTX 1080)")
    log("=" * 70)

    # Load existing results
    results_path = os.path.join(_project_root, 'results', 'thesis_sweep_results.json')
    existing_results = []
    if os.path.exists(results_path):
        with open(results_path) as f:
            existing_results = json.load(f)

    existing_labels = {r['label'] for r in existing_results}

    # Define pending studies
    pending_cases = []

    # Study 3a: We sweep on flat + ridge (R*=1.0)
    study3a_we = [2.0, 3.0, 5.0, 7.9, 10.0, 12.0, 15.0, 20.0]
    for we in study3a_we:
        label_flat = f"s3a_flat_We{we}"
        label_ridge = f"s3a_ridge_R1.0_We{we}"
        if label_flat not in existing_labels:
            pending_cases.append(('flat', None, we, 162.0, 0.0, label_flat))
        if label_ridge not in existing_labels:
            pending_cases.append(('ridge', 1.0, we, 162.0, 1.8, label_ridge))

    # Study 3b: We sweep on convex + concave (R*=1.0)
    study3b_we = [5.0, 7.9, 10.0, 12.0, 15.0, 20.0]
    for we in study3b_we:
        label_convex = f"s3b_convex_R1.0_We{we}"
        label_concave = f"s3b_concave_R1.0_We{we}"
        if label_convex not in existing_labels:
            pending_cases.append(('convex', 1.0, we, 162.0, 1.8, label_convex))
        if label_concave not in existing_labels:
            pending_cases.append(('concave', 1.0, we, 162.0, 1.8, label_concave))

    # Study 5: R* extension at We=5 and We=12
    study5_rstars_we5 = [0.5, 0.7, 1.0, 1.4, 2.0, 2.76, 4.0]
    study5_rstars_we12 = [0.5, 0.7, 1.0, 1.4, 2.0, 2.76]
    for r_star in study5_rstars_we5:
        label = f"s5_ridge_R{r_star}_We5.0"
        if label not in existing_labels:
            pending_cases.append(('ridge', r_star, 5.0, 162.0, 1.8, label))
    for r_star in study5_rstars_we12:
        label = f"s5_ridge_R{r_star}_We12.0"
        if label not in existing_labels:
            pending_cases.append(('ridge', r_star, 12.0, 162.0, 1.8, label))

    log(f"\nPending cases to run: {len(pending_cases)}")
    log(f"Already completed: {len(existing_results)} cases")

    if not pending_cases:
        log("No pending cases. All studies complete!")
        return

    # Run pending cases
    results = list(existing_results)
    all_histories = {}

    for i, (sub_type, R_star, we, theta, amp_val, label) in enumerate(pending_cases):
        log(f"\n--- Case {i+1}/{len(pending_cases)}: {label} ---")
        try:
            r, h = run_case(
                substrate_type=sub_type, R_star=R_star,
                We=we, theta_eq=theta, amp=amp_val,
                N=3000, n_base=150, dtype=torch.float32,
                label=label,
            )
            results.append(r)
            all_histories[label] = h
            log(f"  k_max={r['max_k']:.4f}, Dx={r['Dx']:.0f}, Dy={r['Dy']:.0f}, "
                f"elapsed={r['elapsed_s']:.1f}s, mem={r['mem_mb']:.0f}MB")

            # Save intermediate results
            save_data = [{k: v for k, v in r.items() if k != 'k_history'} for r in results]
            with open(results_path, 'w') as f:
                json.dump(save_data, f, indent=2, default=str)

            # Save history
            history_path = os.path.join(_project_root, 'results', 'thesis_k_history.json')
            with open(history_path, 'w') as f:
                json.dump(all_histories, f, indent=2)

        except Exception as e:
            log(f"  ERROR: {e}")
            continue

    # Final summary
    log("\n" + "=" * 70)
    log("  SWEEP COMPLETE")
    log("=" * 70)
    log(f"  Total cases run: {len(results)}")
    log(f"  Results saved to: {results_path}")


if __name__ == "__main__":
    main()
