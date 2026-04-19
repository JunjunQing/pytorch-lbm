#!/usr/bin/env python3
"""Supplementary theta sweep + phase diagram fill + morphology snapshots (15 cases).

Machine: Local CMP 30HX 6GB (CUDA)
Purpose: 1) Fill gaps in theta sweep (ridge R*=1.0, We=7.9)
         2) Fill gaps in We-theta phase diagram
         3) Capture morphology snapshots for thesis figures

Part 1 — Theta sweep densification (8 cases):
  theta = 45, 60, 75, 90, 105, 120, 130, 150 at We=7.9
  (60, 90, 120 are re-runs for snapshot capture)

Part 2 — Phase diagram fill (7 cases):
  (60,10), (90,10), (120,10), (140,5), (140,10), (140,15), (140,20)

All cases: ridge R*=1.0, amp=1.8, 150²×nz, 3000 steps, save snapshots.
Estimated: 15 × ~14min = 3.5h
Usage: python3 scripts/run_supplementary_theta_phase.py
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

DEVICE = 'cuda'
if not torch.cuda.is_available():
    raise RuntimeError('CUDA not available')
torch.cuda.empty_cache()
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# --- Global physical parameters ---
D0 = 45.0
R_drop = D0 / 2.0
rho_l = 1.0
rho_g = 1.0 / 828.0
xi = 4.0
tau = 0.53
U0 = -0.05

RESULTS_DIR = os.path.join(_project_root, 'results')
SNAP_DIR = os.path.join(RESULTS_DIR, 'thesis_snapshots')


def get_surface_height(solid, nx, ny, nz):
    height = np.zeros((nx, ny), dtype=int)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz - 1, -1, -1):
                if solid[i, j, k]:
                    height[i, j] = k
                    break
    return height


def run_case(We=7.9, theta_eq=162.0, amp=1.8, N=3000, n_base=150,
             dtype=torch.float32, label=None,
             save_snapshots=True, snapshot_dir=SNAP_DIR):
    """Run single ridge R*=1.0 case with snapshot saving."""
    substrate_type = 'ridge'
    R_star = 1.0

    sigma = rho_l * U0 ** 2 * D0 / We
    beta = 12.0 * sigma / xi
    kappa = beta * xi ** 2 / 8.0
    M = 0.02 / beta
    tau_l = tau
    tau_g = tau

    R_g = abs(R_star) * R_drop
    nx, ny = n_base, n_base
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

    solid, fraction = create_substrate_with_fraction(
        nx, ny, nz, substrate_type=substrate_type,
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

    max_k = 0.0
    max_k_step = 0
    Dx_peak = 0.0
    Dy_peak = 0.0
    k_history = []
    stable = True
    t0 = time.time()

    snap_steps = set()
    if save_snapshots and snapshot_dir:
        for frac in [0.03, 0.07, 0.13, 0.20, 0.33, 0.47, 0.60, 0.73, 0.87, 1.0]:
            snap_steps.add(int(frac * N))

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
            else:
                Dx = Dy = Dz = k_val = 0
                z_com = 0.0
            k_history.append({
                'step': step, 'Dx': Dx, 'Dy': Dy, 'Dz': Dz,
                'k': k_val, 'z_com': z_com,
            })
            if k_val > max_k:
                max_k = k_val
                max_k_step = step
                Dx_peak = Dx
                Dy_peak = Dy
            if np.isnan(phi_np).any() or phi_np.max() < 0.01:
                stable = False
                break

        if save_snapshots and step in snap_steps and snapshot_dir:
            os.makedirs(snapshot_dir, exist_ok=True)
            phi_np = solver.phi.detach().cpu().numpy()
            solid_np = solver.solid.cpu().numpy()
            u_np = solver.u.detach().cpu().numpy()
            mid_j = ny // 2
            mid_i = nx // 2
            snap_path = os.path.join(snapshot_dir, f"{label}_step{step:05d}.npz")
            np.savez_compressed(snap_path,
                                phi_3d=phi_np, u_3d=u_np,
                                solid_3d=solid_np,
                                xz_slice=phi_np[:, mid_j, :],
                                yz_slice=phi_np[mid_i, :, :],
                                xz_solid=solid_np[:, mid_j, :],
                                step=step, nx=nx, ny=ny, nz=nz)

    elapsed = time.time() - t0
    mem_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    torch.cuda.reset_peak_memory_stats()

    if label is None:
        label = f"sup_theta{theta_eq:.0f}_We{We:.1f}"

    tag = "OK" if stable else "FAIL"
    print(f"  {label:>40} We={We:5.1f} theta={theta_eq:5.1f} | "
          f"k={max_k:.4f} Dx={Dx_peak:.0f} Dy={Dy_peak:.0f} "
          f"step={max_k_step} [{tag}] "
          f"[{elapsed:.0f}s, {mem_mb:.0f}MB] "
          f"grid={nx}x{ny}x{nz}")

    del solver
    torch.cuda.empty_cache()

    return {
        'label': label, 'substrate_type': substrate_type,
        'R_star': R_star, 'We': We, 'theta_eq': theta_eq, 'amp': amp,
        'max_k': max_k, 'max_k_step': max_k_step,
        'Dx': Dx_peak, 'Dy': Dy_peak,
        'stable': stable, 'grid': f"{nx}x{ny}x{nz}",
        'elapsed_s': round(elapsed, 1), 'mem_mb': round(mem_mb, 0),
        'machine': 'local_30hx_cuda', 'k_history': k_history,
    }


def main():
    results = []

    # =================================================================
    # Part 1: Theta sweep densification (8 cases at We=7.9)
    # =================================================================
    print("\n" + "=" * 70)
    print("  Part 1: Theta sweep densification (We=7.9)")
    print("=" * 70)

    for theta in [45, 60, 75, 90, 105, 120, 130, 150]:
        is_rerun = theta in [60, 90, 120]
        label = f"morph_theta{theta:.0f}" if is_rerun else f"sup_theta{theta:.0f}"
        r = run_case(We=7.9, theta_eq=theta, amp=1.8, N=3000, n_base=150,
                     label=label)
        results.append(r)

    # =================================================================
    # Part 2: Phase diagram fill (7 cases)
    # =================================================================
    print("\n" + "=" * 70)
    print("  Part 2: Phase diagram fill")
    print("=" * 70)

    phase_cases = [
        (60, 10.0), (90, 10.0), (120, 10.0),
        (140, 5.0), (140, 10.0), (140, 15.0), (140, 20.0),
    ]
    for theta, we in phase_cases:
        label = f"phase_theta{theta:.0f}_We{we:.1f}"
        r = run_case(We=we, theta_eq=theta, amp=1.8, N=3000, n_base=150,
                     label=label)
        results.append(r)

    # =================================================================
    # Summary
    # =================================================================
    print("\n" + "=" * 70)
    print("  SUPPLEMENTARY THETA + PHASE RESULTS")
    print("=" * 70)
    ok = sum(1 for r in results if r['stable'])
    print(f"\n  Total: {len(results)} cases, {ok} OK, {len(results)-ok} FAIL")
    for r in results:
        tag = "OK" if r['stable'] else "FAIL"
        print(f"  {r['label']:>40} We={r['We']:5.1f} theta={r['theta_eq']:5.0f} "
              f"k={r['max_k']:.4f} [{tag}]")

    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    save_path = os.path.join(RESULTS_DIR, 'supplementary_theta_phase_results.json')
    save_data = [{k: v for k, v in r.items() if k != 'k_history'} for r in results]
    with open(save_path, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\n  Results saved to {save_path}")

    # Save history
    history_path = os.path.join(RESULTS_DIR, 'supplementary_theta_phase_history.json')
    history_data = {r['label']: r.get('k_history', []) for r in results}
    with open(history_path, 'w') as f:
        json.dump(history_data, f, indent=2)
    print(f"  History saved to {history_path}")


if __name__ == "__main__":
    main()
