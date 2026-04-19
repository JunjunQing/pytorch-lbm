#!/usr/bin/env python3
"""Machine: qingqing (Intel Arc A750 8GB) — Studies 4a/4b/6.

Studies 4a: Contact angle sweep on flat + ridge (12 cases)
Studies 4b: Contact angle sweep on convex/concave + supplementary (12 cases)
Study 6: We-θ phase diagram on ridge R*=1.0 (12 new cases)
Total: 36 cases × ~11min/case ≈ 6.6h

Hardware: Intel Arc A750 8GB (XPU backend)
Setup: pip install torch
Usage: bash scripts/run_thesis_qingqing.sh
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

# Intel Arc A750 uses XPU backend
DEVICE = 'cuda'
if not torch.cuda.is_available():
    raise RuntimeError('CUDA not available')
torch.cuda.empty_cache()
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# --- Global physical parameters (v4, 150² + VP) ---
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
             dtype=torch.float32, label=None):
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
        'machine': 'local_30hx_cuda', 'k_history': k_history,
    }


def main():
    results = []

    # =================================================================
    # STUDY 4a: Contact angle sweep on flat (6 cases)
    # =================================================================
    print("\n" + "=" * 70)
    print("  STUDY 4a: Contact angle sweep on flat (CUDA)")
    print("=" * 70)

    for theta in [30.0, 60.0, 90.0, 120.0, 140.0, 162.0]:
        r = run_case(
            substrate_type='flat', R_star=None,
            We=7.9, theta_eq=theta, amp=0.0,
            N=3000, n_base=150, label=f"s4a_flat_theta{theta:.0f}",
        )
        results.append(r)

    # =================================================================
    # STUDY 4a: Contact angle sweep on ridge R*=1.0 (6 cases)
    # =================================================================
    print("\n" + "=" * 70)
    print("  STUDY 4a: Contact angle sweep on ridge R*=1.0 (CUDA)")
    print("=" * 70)

    for theta in [30.0, 60.0, 90.0, 120.0, 140.0, 162.0]:
        r = run_case(
            substrate_type='ridge', R_star=1.0,
            We=7.9, theta_eq=theta, amp=1.8,
            N=3000, n_base=150, label=f"s4a_ridge_R1.0_theta{theta:.0f}",
        )
        results.append(r)

    # =================================================================
    # STUDY 4b: Contact angle sweep on convex R*=1.0 (5 cases)
    # =================================================================
    print("\n" + "=" * 70)
    print("  STUDY 4b: Contact angle sweep on convex R*=1.0 (CUDA)")
    print("=" * 70)

    for theta in [60.0, 90.0, 120.0, 140.0, 162.0]:
        r = run_case(
            substrate_type='convex', R_star=1.0,
            We=7.9, theta_eq=theta, amp=1.8,
            N=3000, n_base=150, label=f"s4b_convex_R1.0_theta{theta:.0f}",
        )
        results.append(r)

    # =================================================================
    # STUDY 4b: Contact angle sweep on concave R*=1.0 (5 cases)
    # =================================================================
    print("\n" + "=" * 70)
    print("  STUDY 4b: Contact angle sweep on concave R*=1.0 (CUDA)")
    print("=" * 70)

    for theta in [60.0, 90.0, 120.0, 140.0, 162.0]:
        r = run_case(
            substrate_type='concave', R_star=1.0,
            We=7.9, theta_eq=theta, amp=1.8,
            N=3000, n_base=150, label=f"s4b_concave_R1.0_theta{theta:.0f}",
        )
        results.append(r)

    # =================================================================
    # STUDY 4b: Supplementary ridge θ=60 at We=5, 15 (2 cases)
    # =================================================================
    print("\n" + "=" * 70)
    print("  STUDY 4b: Supplementary ridge θ=60 (CUDA)")
    print("=" * 70)

    for we in [5.0, 15.0]:
        r = run_case(
            substrate_type='ridge', R_star=1.0,
            We=we, theta_eq=60.0, amp=1.8,
            N=3000, n_base=150, label=f"s4b_ridge_R1.0_theta60_We{we}",
        )
        results.append(r)

    # =================================================================
    # STUDY 6: We-θ phase diagram on ridge R*=1.0 (12 new cases)
    # θ={60,90,120} × We={5,12,15,20}
    # =================================================================
    print("\n" + "=" * 70)
    print("  STUDY 6: We-θ phase diagram (12 new cases)")
    print("=" * 70)

    for theta in [60.0, 90.0, 120.0]:
        for we in [5.0, 12.0, 15.0, 20.0]:
            r = run_case(
                substrate_type='ridge', R_star=1.0,
                We=we, theta_eq=theta, amp=1.8,
                N=3000, n_base=150,
                label=f"s6_ridge_R1.0_theta{theta:.0f}_We{we}",
            )
            results.append(r)

    # =================================================================
    # Summary
    # =================================================================
    print("\n" + "=" * 70)
    print("  QINGQING RESULTS SUMMARY (CUDA)")
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

    save_path = os.path.join(save_dir, 'thesis_qingqing_results.json')
    save_data = []
    for r in results:
        r_copy = {k: v for k, v in r.items() if k != 'k_history'}
        save_data.append(r_copy)
    with open(save_path, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\n  Results saved to {save_path}")

    history_path = os.path.join(save_dir, 'thesis_qingqing_history.json')
    history_data = {r['label']: r.get('k_history', []) for r in results}
    with open(history_path, 'w') as f:
        json.dump(history_data, f, indent=2)
    print(f"  Time histories saved to {history_path}")


if __name__ == "__main__":
    main()
