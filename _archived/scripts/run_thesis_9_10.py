#!/usr/bin/env python3
"""Studies 9 + 10: Dual-droplet + Multi-droplet simulations (23 cases).

Machine: 另一台 GPU 机器 (CUDA, 建议 ≥6GB VRAM)
预估: ~7.4h (12 dual-droplet + 11 multi-droplet)

Usage: python3 scripts/run_thesis_9_10.py
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
from lbm.fe_droplet import create_fe_droplet_with_impact, create_multi_droplet_with_impact
from geometry.substrate import create_substrate, create_substrate_with_fraction

DEVICE = 'cuda'
torch.cuda.empty_cache()
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

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


# ======================================================================
# Study 9: Dual-droplet coalescence
# ======================================================================

def run_dual_case(substrate_type='ridge', R_star=None, We=7.9,
                  theta_eq=162.0, amp=1.8, spacing_ratio=1.2,
                  N=4000, n_base=150, dtype=torch.float32, label=None):
    sigma = rho_l * U0 ** 2 * D0 / We
    beta = 12.0 * sigma / xi
    kappa = beta * xi ** 2 / 8.0
    M = 0.02 / beta
    tau_l, tau_g = tau, tau

    is_flat = substrate_type == 'flat' or R_star is None
    if is_flat:
        nz_min = int(R_drop + 2 + 2 * R_drop + 15)
        nx = int(spacing_ratio * D0 + D0 + 20)
        ny, nz = n_base, nz_min
    else:
        R_g = abs(R_star) * R_drop
        nx = int(spacing_ratio * D0 + D0 + 20)
        ny = n_base
        nz = min(int(R_g + 2 + R_drop + 2 * R_drop + 15), 300)

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
        config, dtype=dtype, stab_mode='fakhari',
        boundary_relax=0.0, geometric_wetting=(amp > 0),
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

    d_center = spacing_ratio * D0
    cx1 = max(R_drop + 1, cx_total - d_center / 2.0)
    cx2 = min(nx - R_drop - 1, cx_total + d_center / 2.0)
    centers = [(cx1, cy, cz), (cx2, cy, cz)]
    radii = [R_drop, R_drop]
    u_impacts = [(0.0, 0.0, U0), (0.0, 0.0, U0)]

    C_init, _, u_init = create_multi_droplet_with_impact(
        nx, ny, nz, centers=centers, radii=radii,
        xi=xi, rho_l=rho_l, rho_g=rho_g, u_impacts=u_impacts,
    )
    solver.init_fields(C_init, u_init)

    max_k, max_k_step, Dx_peak, Dy_peak = 0, 0, 0, 0
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
                k_val = Dx / Dy if Dy > 0 else 0
                x_range = coords[:, 0].max() - coords[:, 0].min()
                merged = x_range < (d_center + R_drop * 0.5)
            else:
                Dx = Dy = k_val = 0
                merged = False
            if coalescence_step is None and merged:
                coalescence_step = step
            k_history.append({'step': step, 'Dx': Dx, 'Dy': Dy, 'k': k_val, 'merged': merged})
            if k_val > max_k:
                max_k, max_k_step, Dx_peak, Dy_peak = k_val, step, Dx, Dy
            if np.isnan(phi_np).any() or phi_np.max() < 0.01:
                stable = False
                break

    elapsed = time.time() - t0
    mem_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    torch.cuda.reset_peak_memory_stats()
    if label is None:
        label = f"dual_{substrate_type}_R{R_star or 'flat'}_d{spacing_ratio:.1f}"
    coal_str = f"merge@{coalescence_step}" if coalescence_step else "separate"
    tag = "OK" if stable else "FAIL"
    print(f"  {label:>35} We={We:5.1f} d/D={spacing_ratio:.1f} | k={max_k:.4f} {coal_str} [{tag}] [{elapsed:.0f}s, {mem_mb:.0f}MB] {nx}x{ny}x{nz}")
    del solver
    torch.cuda.empty_cache()
    return {
        'label': label, 'substrate_type': substrate_type, 'R_star': R_star,
        'We': We, 'theta_eq': theta_eq, 'amp': amp, 'spacing_ratio': spacing_ratio,
        'max_k': max_k, 'max_k_step': max_k_step, 'Dx': Dx_peak, 'Dy': Dy_peak,
        'coalescence_step': coalescence_step, 'stable': stable,
        'grid': f"{nx}x{ny}x{nz}", 'elapsed_s': round(elapsed, 1),
        'mem_mb': round(mem_mb, 0), 'machine': 'remote_9_10', 'k_history': k_history,
    }


# ======================================================================
# Study 10: Multi-droplet sequential
# ======================================================================

def run_sequential_case(substrate_type='ridge', R_star=None, We=7.9,
                        theta_eq=162.0, amp=1.8,
                        n_drops=2, dt_star=1.0,
                        N=6000, n_base=150, dtype=torch.float32, label=None):
    sigma = rho_l * U0 ** 2 * D0 / We
    beta = 12.0 * sigma / xi
    kappa = beta * xi ** 2 / 8.0
    M = 0.02 / beta
    tau_l, tau_g = tau, tau

    is_flat = substrate_type == 'flat' or R_star is None
    if is_flat:
        nz = min(int(R_drop + 2 + (1 + n_drops) * R_drop + 20), 300)
        nx, ny = n_base, n_base
    else:
        R_g = abs(R_star) * R_drop
        nx, ny = n_base, n_base
        nz = min(int(R_g + 2 + R_drop + (1 + n_drops) * R_drop + 20), 300)

    dt_inject = int(dt_star * D0 / abs(U0))
    inject_steps = [k * dt_inject for k in range(1, n_drops)]

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
        config, dtype=dtype, stab_mode='fakhari',
        boundary_relax=0.0, geometric_wetting=(amp > 0),
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
        rho_l=rho_l, rho_g=rho_g, u_impact=(0.0, 0.0, U0))
    solver.init_fields(C_init, u_init)

    max_Dx, max_Dy, max_Dx_step = 0, 0, 0
    k_history = []
    stable = True
    inject_idx = 0
    t0 = time.time()

    for step in range(1, N + 1):
        if inject_idx < len(inject_steps) and step == inject_steps[inject_idx]:
            solver.inject_droplet(center=(cx, cy, cz), radius=R_drop, xi=xi,
                                  u_impact=(0.0, 0.0, U0))
            inject_idx += 1
        solver.step()
        if step % 50 == 0:
            phi_np = solver.phi.detach().cpu().numpy()
            solid_np = solver.solid.cpu().numpy()
            interface = (phi_np > 0.5) & ~solid_np
            if interface.any():
                coords = np.argwhere(interface)
                Dx = float(coords[:, 0].max() - coords[:, 0].min() + 1)
                Dy = float(coords[:, 1].max() - coords[:, 1].min() + 1)
            else:
                Dx = Dy = 0
            k_history.append({'step': step, 'Dx': Dx, 'Dy': Dy, 'n_deposited': 1 + inject_idx})
            if Dx > max_Dx:
                max_Dx, max_Dy, max_Dx_step = Dx, Dy, step
            if np.isnan(phi_np).any() or phi_np.max() < 0.01:
                stable = False
                break

    elapsed = time.time() - t0
    mem_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    torch.cuda.reset_peak_memory_stats()
    if label is None:
        label = f"seq_{substrate_type}_R{R_star or 'flat'}_n{n_drops}_dt{dt_star:.1f}"
    tag = "OK" if stable else "FAIL"
    print(f"  {label:>40} We={We:5.1f} n={n_drops} dt*={dt_star:.1f} | Dx={max_Dx:.0f} Dy={max_Dy:.0f} [{tag}] [{elapsed:.0f}s, {mem_mb:.0f}MB] {nx}x{ny}x{nz}")
    del solver
    torch.cuda.empty_cache()
    return {
        'label': label, 'substrate_type': substrate_type, 'R_star': R_star,
        'We': We, 'theta_eq': theta_eq, 'amp': amp,
        'n_drops': n_drops, 'dt_star': dt_star,
        'max_Dx': max_Dx, 'max_Dy': max_Dy, 'max_Dx_step': max_Dx_step,
        'stable': stable, 'grid': f"{nx}x{ny}x{nz}",
        'elapsed_s': round(elapsed, 1), 'mem_mb': round(mem_mb, 0),
        'machine': 'remote_9_10', 'k_history': k_history,
    }


def main():
    results = []

    # =================================================================
    # STUDY 9a: Substrate comparison (d/D=1.2, We=7.9)
    # =================================================================
    print("\n" + "=" * 70)
    print("  STUDY 9a: Substrate comparison (d/D=1.2)")
    print("=" * 70)
    for sub, R, amp in [('flat', None, 0.0), ('ridge', 1.0, 1.8),
                         ('convex', 1.0, 1.8), ('concave', 1.0, 1.8)]:
        results.append(run_dual_case(
            substrate_type=sub, R_star=R, We=7.9, theta_eq=162.0, amp=amp,
            spacing_ratio=1.2, N=4000, n_base=150,
            label=f"s9a_dual_{sub}_R{R or 'flat'}"))

    # Study 9b: Spacing sweep
    print("\n" + "=" * 70)
    print("  STUDY 9b: Spacing sweep on ridge R*=1.0")
    print("=" * 70)
    for d in [1.0, 1.2, 1.5, 2.0]:
        results.append(run_dual_case(
            substrate_type='ridge', R_star=1.0, We=7.9, theta_eq=162.0, amp=1.8,
            spacing_ratio=d, N=4000, n_base=150,
            label=f"s9b_ridge_R1.0_d{d:.1f}"))

    # Study 9c: We sweep
    print("\n" + "=" * 70)
    print("  STUDY 9c: We sweep on ridge R*=1.0 (d/D=1.2)")
    print("=" * 70)
    for we in [5.0, 10.0, 15.0, 20.0]:
        results.append(run_dual_case(
            substrate_type='ridge', R_star=1.0, We=we, theta_eq=162.0, amp=1.8,
            spacing_ratio=1.2, N=4000, n_base=150,
            label=f"s9c_ridge_R1.0_We{we}"))

    # =================================================================
    # STUDY 10a: Substrate comparison (2 drops)
    # =================================================================
    print("\n" + "=" * 70)
    print("  STUDY 10a: Substrate comparison (2 drops, dt*=1.0)")
    print("=" * 70)
    for sub, R, amp in [('flat', None, 0.0), ('ridge', 1.0, 1.8),
                         ('convex', 1.0, 1.8), ('concave', 1.0, 1.8)]:
        results.append(run_sequential_case(
            substrate_type=sub, R_star=R, We=7.9, theta_eq=162.0, amp=amp,
            n_drops=2, dt_star=1.0, N=6000, n_base=150,
            label=f"s10a_seq_{sub}_R{R or 'flat'}"))

    # Study 10b: Interval sweep
    print("\n" + "=" * 70)
    print("  STUDY 10b: Interval sweep on ridge R*=1.0")
    print("=" * 70)
    for dt in [0.5, 1.0, 2.0]:
        results.append(run_sequential_case(
            substrate_type='ridge', R_star=1.0, We=7.9, theta_eq=162.0, amp=1.8,
            n_drops=2, dt_star=dt, N=6000, n_base=150,
            label=f"s10b_ridge_R1.0_dt{dt:.1f}"))

    # Study 10c: Droplet count
    print("\n" + "=" * 70)
    print("  STUDY 10c: Droplet count on ridge R*=1.0")
    print("=" * 70)
    for n in [3, 5]:
        results.append(run_sequential_case(
            substrate_type='ridge', R_star=1.0, We=7.9, theta_eq=162.0, amp=1.8,
            n_drops=n, dt_star=1.0, N=n * 3000, n_base=150,
            label=f"s10c_ridge_R1.0_n{n}"))

    # Study 10d: We comparison
    print("\n" + "=" * 70)
    print("  STUDY 10d: We comparison on ridge (2 drops)")
    print("=" * 70)
    for we in [5.0, 15.0]:
        results.append(run_sequential_case(
            substrate_type='ridge', R_star=1.0, We=we, theta_eq=162.0, amp=1.8,
            n_drops=2, dt_star=1.0, N=6000, n_base=150,
            label=f"s10d_ridge_R1.0_We{we}"))

    # =================================================================
    # Summary
    # =================================================================
    print("\n" + "=" * 70)
    print("  STUDIES 9 + 10 RESULTS SUMMARY")
    print("=" * 70)
    ok = sum(1 for r in results if r['stable'])
    print(f"\n  Total: {len(results)} cases, {ok} OK, {len(results)-ok} FAIL")
    for r in results:
        tag = "OK" if r['stable'] else "FAIL"
        if 'spacing_ratio' in r:
            print(f"  {r['label']:>35} d/D={r['spacing_ratio']:.1f} k={r.get('max_k',r.get('max_Dx',0)):.4f} [{tag}]")
        else:
            print(f"  {r['label']:>40} n={r.get('n_drops','-')} Dx={r.get('max_Dx',0):.0f} [{tag}]")

    save_dir = os.path.join(_project_root, 'results')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'thesis_studies_9_10_results.json')
    save_data = [{k: v for k, v in r.items() if k != 'k_history'} for r in results]
    with open(save_path, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\n  Results saved to {save_path}")


if __name__ == "__main__":
    main()
