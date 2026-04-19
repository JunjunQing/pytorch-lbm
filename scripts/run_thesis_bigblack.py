#!/usr/bin/env python3
"""Machine: bigblack (NVIDIA GTX 1080 8GB) — Studies 3a/3b/5.

Studies 3a+3b: We sweep on all substrates (28 cases)
Study 5: R* extended sweep at We=5, 12 (13 cases)
Total: 41 cases × ~11min/case ≈ 7.5h

Hardware: AMD Ryzen 9 5900x + NVIDIA GTX 1080 8GB
Setup: pip install torch (standard CUDA backend)
Usage: bash scripts/run_thesis_bigblack.sh
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
torch.cuda.empty_cache()
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# --- Global physical parameters (v2, 150² + VP) ---
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


def run_case(substrate_type='ridge', R_star=None, We=7.9,
             theta_eq=162.0, amp=1.8, N=3000, n_base=150,
             dtype=torch.float32, label=None,
             save_snapshots=False, snapshot_dir=None):
    sigma = rho_l * U0 ** 2 * D0 / We
    beta = 12.0 * sigma / xi
    kappa = beta * xi ** 2 / 8.0
    M = 0.02 / beta

    tau_l = tau
    tau_g = tau

    is_flat = substrate_type == 'flat' or R_star is None

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

    if is_flat:
        solid = create_substrate(nx, ny, nz, substrate_type='flat',
                                 R_star=None, R_d=R_drop)
        solver.set_solid(solid)
    else:
        solid, fraction = create_substrate_with_fraction(
            nx, ny, nz, substrate_type=substrate_type,
            R_star=R_star, R_d=R_drop)
        solver.set_solid(solid, solid_fraction=fraction)

    surface_height = get_surface_height(solid, nx, ny, nz)
    cx, cy = nx / 2.0, ny / 2.0
    if substrate_type == 'concave':
        ridge_top = int(surface_height.max())
    else:
        ridge_top = surface_height[nx // 2, ny // 2]
    gap = 2.0
    cz = ridge_top + gap + R_drop
    max_cz = nz - R_drop - 2
    if cz > max_cz:
        cz = max_cz

    C_init, _, u_init = create_fe_droplet_with_impact(
        nx, ny, nz, center=(cx, cy, cz), radius=R_drop, xi=xi,
        rho_l=rho_l, rho_g=rho_g, u_impact=(0.0, 0.0, U0)
    )
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
        for frac in [0.03, 0.07, 0.13, 0.20, 0.33, 0.47,
                     0.60, 0.73, 0.87, 1.0]:
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
            snap_path = os.path.join(snapshot_dir,
                                     f"{label}_step{step:05d}.npz")
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
        if is_flat:
            label = "flat"
        else:
            label = f"{substrate_type}_R{R_star:.1f}"

    tag = "OK" if stable else "FAIL"
    print(f"  {label:>30} We={We:5.1f} amp={amp:.1f} theta={theta_eq:5.1f} | "
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
        'machine': 'bigblack_gpu', 'k_history': k_history,
    }


def main():
    results = []
    snapshot_dir = os.path.join(_project_root, 'results', 'thesis_snapshots')

    # =================================================================
    # STUDY 3a: We sweep on flat (8 cases)
    # =================================================================
    print("\n" + "=" * 70)
    print("  STUDY 3a: We sweep on flat (GPU)")
    print("=" * 70)

    for we in [2.0, 3.0, 5.0, 7.9, 10.0, 12.0, 15.0, 20.0]:
        r = run_case(
            substrate_type='flat', R_star=None,
            We=we, theta_eq=162.0, amp=0.0,
            N=3000, n_base=150, label=f"s3a_flat_We{we}",
        )
        results.append(r)

    # =================================================================
    # STUDY 3a: We sweep on ridge R*=1.0 (8 cases)
    # =================================================================
    print("\n" + "=" * 70)
    print("  STUDY 3a: We sweep on ridge R*=1.0 (GPU)")
    print("=" * 70)

    for we in [2.0, 3.0, 5.0, 7.9, 10.0, 12.0, 15.0, 20.0]:
        r = run_case(
            substrate_type='ridge', R_star=1.0,
            We=we, theta_eq=162.0, amp=1.8,
            N=3000, n_base=150, label=f"s3a_ridge_R1.0_We{we}",
        )
        results.append(r)

    # =================================================================
    # STUDY 3b: We sweep on convex R*=1.0 (6 cases)
    # =================================================================
    print("\n" + "=" * 70)
    print("  STUDY 3b: We sweep on convex R*=1.0 (GPU)")
    print("=" * 70)

    for we in [5.0, 7.9, 10.0, 12.0, 15.0, 20.0]:
        r = run_case(
            substrate_type='convex', R_star=1.0,
            We=we, theta_eq=162.0, amp=1.8,
            N=3000, n_base=150, label=f"s3b_convex_R1.0_We{we}",
        )
        results.append(r)

    # =================================================================
    # STUDY 3b: We sweep on concave R*=1.0 (6 cases)
    # =================================================================
    print("\n" + "=" * 70)
    print("  STUDY 3b: We sweep on concave R*=1.0 (GPU)")
    print("=" * 70)

    for we in [5.0, 7.9, 10.0, 12.0, 15.0, 20.0]:
        r = run_case(
            substrate_type='concave', R_star=1.0,
            We=we, theta_eq=162.0, amp=1.8,
            N=3000, n_base=150, label=f"s3b_concave_R1.0_We{we}",
        )
        results.append(r)

    # =================================================================
    # STUDY 5: R* sweep at We=5.0 (7 cases)
    # =================================================================
    print("\n" + "=" * 70)
    print("  STUDY 5: R* sweep at We=5.0 (GPU)")
    print("=" * 70)

    r_star_amp_map = {
        0.5: 2.0, 0.7: 1.8, 1.0: 1.8, 1.4: 1.2,
        2.0: 0.8, 2.76: 0.5, 4.0: 0.3,
    }
    for R_star, amp_val in r_star_amp_map.items():
        r = run_case(
            substrate_type='ridge', R_star=R_star,
            We=5.0, theta_eq=162.0, amp=amp_val,
            N=3000, n_base=150, label=f"s5_ridge_R{R_star}_We5.0",
        )
        results.append(r)

    # =================================================================
    # STUDY 5: R* sweep at We=12.0 (6 cases)
    # =================================================================
    print("\n" + "=" * 70)
    print("  STUDY 5: R* sweep at We=12.0 (GPU)")
    print("=" * 70)

    r_star_amp_we12 = {
        0.5: 2.0, 0.7: 1.8, 1.0: 1.8, 1.4: 1.2,
        2.0: 0.8, 2.76: 0.5,
    }
    for R_star, amp_val in r_star_amp_we12.items():
        r = run_case(
            substrate_type='ridge', R_star=R_star,
            We=12.0, theta_eq=162.0, amp=amp_val,
            N=3000, n_base=150, label=f"s5_ridge_R{R_star}_We12.0",
        )
        results.append(r)

    # =================================================================
    # Summary
    # =================================================================
    print("\n" + "=" * 70)
    print("  BIGBLACK GPU RESULTS SUMMARY")
    print("=" * 70)

    print(f"\n  {'Label':>30} {'Type':>8} {'R*':>5} {'We':>6} "
          f"{'theta':>6} {'amp':>5} {'k_max':>8} {'Dx':>4} {'Dy':>4} "
          f"{'step':>5} {'Status':>6}")
    print("  " + "-" * 95)

    for r in results:
        status = "OK" if r['stable'] else "FAIL"
        R_str = f"{r['R_star']:.1f}" if r['R_star'] is not None else "---"
        print(f"  {r['label']:>30} {r['substrate_type']:>8} {R_str:>5} "
              f"{r['We']:6.1f} {r['theta_eq']:6.1f} {r['amp']:5.1f} "
              f"{r['max_k']:8.4f} {r['Dx']:4.0f} {r['Dy']:4.0f} "
              f"{r['max_k_step']:5d} {status:>6}")

    # Save results
    save_dir = os.path.join(_project_root, 'results')
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, 'thesis_bigblack_results.json')
    save_data = []
    for r in results:
        r_copy = {k: v for k, v in r.items() if k != 'k_history'}
        save_data.append(r_copy)
    with open(save_path, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\n  Results saved to {save_path}")

    history_path = os.path.join(save_dir, 'thesis_bigblack_history.json')
    history_data = {r['label']: r.get('k_history', []) for r in results}
    with open(history_path, 'w') as f:
        json.dump(history_data, f, indent=2)
    print(f"  Time histories saved to {history_path}")


if __name__ == "__main__":
    main()
