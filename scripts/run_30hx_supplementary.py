#!/usr/bin/env python3
"""CMP 30HX Supplementary Simulations (Studies 12a-12f, 23 cases).

Grid convergence verification + concave bug fix re-runs + film thickness.

Sub-studies:
  12a: Dual-droplet grid convergence      (4 cases)  — thesis essential
  12b: Multi-droplet grid convergence      (3 cases)  — thesis essential
  12c: Concave re-runs with fix            (5 cases)  — bug fix validation
  12d: Film thickness We scan              (5 cases)  — extends film data
  12e: Long time evolution snapshots        (3 cases)  — time-resolved data
  12f: Line formation grid convergence     (3 cases)  — Chapter 4 validation

Machine: NVIDIA CMP 30HX (6GB VRAM)
Estimated runtime: ~6-7 hours

Usage: python3 scripts/run_30hx_supplementary.py [--skip-existing]
"""
import sys
import os
import argparse
import time
import json

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)

import numpy as np
import torch

DEVICE = 'cuda'
torch.cuda.empty_cache()
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# Physical constants
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


# ============================================================================
# 1. Single droplet (reused for concave re-runs)
# ============================================================================

def run_single(substrate_type='ridge', R_star=1.0, We=7.9,
               theta_eq=162.0, amp=1.8, N=4000,
               n_base=150, dtype=torch.float32, label=None,
               save_snapshots=False):
    """Run single droplet impact case."""
    from lbm.fe_config import FEConfig
    from lbm.fe_ac_solver import AllenCahnSolver
    from lbm.fe_droplet import create_fe_droplet_with_impact
    from geometry.substrate import create_substrate, create_substrate_with_fraction

    sigma = rho_l * U0**2 * D0 / We
    beta = 12.0 * sigma / xi
    kappa = beta * xi**2 / 8.0
    M = 0.02 / beta

    is_flat = substrate_type == 'flat' or R_star is None
    if is_flat:
        nx, ny, nz = n_base, n_base, int(R_drop + 2 + R_drop + 15)
    else:
        R_g = abs(R_star) * R_drop
        nx, ny = n_base, n_base
        nz = int(R_g + 2 + R_drop + R_drop + 15)

    config = FEConfig(
        nx=nx, ny=ny, nz=nz,
        rho_l=rho_l, rho_g=rho_g,
        sigma=sigma, xi=xi, beta=beta, kappa=kappa,
        M=M, tau_l=tau, tau_g=tau, tau_h=tau + 0.04,
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

    solid_np = solid if isinstance(solid, np.ndarray) else np.array(solid)
    surface_height = get_surface_height(solid_np, nx, ny, nz)
    cx, cy = nx / 2.0, ny / 2.0
    if substrate_type == 'concave':
        ridge_top = int(surface_height.max())
    else:
        ridge_top = surface_height[nx // 2, ny // 2]
    cz = ridge_top + 2.0 + R_drop

    C_init, _, u_init = create_fe_droplet_with_impact(
        nx, ny, nz, center=(cx, cy, cz), radius=R_drop, xi=xi,
        rho_l=rho_l, rho_g=rho_g, u_impact=(0.0, 0.0, U0))
    solver.init_fields(C_init, u_init)

    snap_dir = os.path.join(_project_root, 'results', 'chapter4_snapshots')
    os.makedirs(snap_dir, exist_ok=True)
    snap_prefix = label or f"single_{substrate_type}_R{R_star or 'flat'}"
    snap_steps = set()
    if save_snapshots:
        for s in range(500, N + 1, 500):
            snap_steps.add(s)

    max_k, max_k_step, Dx_peak, Dy_peak = 0, 0, 0, 0
    k_history = []
    stable = True
    t0 = time.time()

    for step in range(1, N + 1):
        solver.step()
        if step % 50 == 0:
            phi_np = solver.phi.detach().cpu().numpy()
            interface = (phi_np > 0.5) & ~solid_np
            if interface.any():
                coords = np.argwhere(interface)
                Dx = float(coords[:, 0].max() - coords[:, 0].min() + 1)
                Dy = float(coords[:, 1].max() - coords[:, 1].min() + 1)
                k_val = Dx / Dy if Dy > 0 else 0
            else:
                Dx = Dy = k_val = 0
            k_history.append({'step': step, 'Dx': Dx, 'Dy': Dy, 'k': k_val})
            if k_val > max_k:
                max_k, max_k_step, Dx_peak, Dy_peak = k_val, step, Dx, Dy
            if np.isnan(phi_np).any() or phi_np.max() < 0.01:
                stable = False
                break

        if save_snapshots and step in snap_steps:
            phi_np = solver.phi.detach().cpu().numpy()
            mid_j = ny // 2
            xz_slice = phi_np[:, mid_j, :]
            xz_solid = solid_np[:, mid_j, :]
            np.savez_compressed(
                os.path.join(snap_dir, f'{snap_prefix}_step{step}.npz'),
                xz_slice=xz_slice, xz_solid=xz_solid,
                step=step, nx=nx, ny=ny, nz=nz)

    elapsed = time.time() - t0
    mem_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    torch.cuda.reset_peak_memory_stats()

    if label is None:
        label = f"single_{substrate_type}_R{R_star or 'flat'}"
    tag = "OK" if stable else "FAIL"
    print(f"  {label:>50} We={We:5.1f} θ={theta_eq:3.0f} "
          f"| k={max_k:.4f} Dx={Dx_peak:.0f} Dy={Dy_peak:.0f} "
          f"[{tag}] [{elapsed:.0f}s, {mem_mb:.0f}MB] {nx}x{ny}x{nz}")

    del solver
    torch.cuda.empty_cache()
    return {
        'label': label, 'substrate_type': substrate_type, 'R_star': R_star,
        'We': We, 'theta_eq': theta_eq, 'amp': amp,
        'max_k': max_k, 'max_k_step': max_k_step,
        'Dx': Dx_peak, 'Dy': Dy_peak,
        'stable': stable, 'grid': f"{nx}x{ny}x{nz}",
        'elapsed_s': round(elapsed, 1), 'mem_mb': round(mem_mb, 0),
        'machine': '30hx_supplementary', 'k_history': k_history,
    }


# ============================================================================
# 2. Dual droplet (for grid convergence)
# ============================================================================

def run_dual(substrate_type='ridge', R_star=1.0, We=7.9,
             theta_eq=162.0, amp=1.8, spacing_ratio=1.2,
             N=4000, n_base=150, dtype=torch.float32, label=None,
             save_snapshots=False):
    """Run dual droplet case."""
    from lbm.fe_config import FEConfig
    from lbm.fe_ac_solver import AllenCahnSolver
    from lbm.fe_droplet import create_multi_droplet_with_impact
    from geometry.substrate import create_substrate, create_substrate_with_fraction

    sigma = rho_l * U0**2 * D0 / We
    beta = 12.0 * sigma / xi
    kappa = beta * xi**2 / 8.0
    M = 0.02 / beta

    is_flat = substrate_type == 'flat' or R_star is None
    if is_flat:
        nz = int(R_drop + 2 + 2 * R_drop + 15)
        nx = int(spacing_ratio * D0 + D0 + 20)
        ny = n_base
    else:
        R_g = abs(R_star) * R_drop
        nx = int(spacing_ratio * D0 + D0 + 20)
        ny = n_base
        nz = int(R_g + 2 + R_drop + 2 * R_drop + 15)

    config = FEConfig(
        nx=nx, ny=ny, nz=nz,
        rho_l=rho_l, rho_g=rho_g,
        sigma=sigma, xi=xi, beta=beta, kappa=kappa,
        M=M, tau_l=tau, tau_g=tau, tau_h=tau + 0.04,
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

    solid_np = solid if isinstance(solid, np.ndarray) else np.array(solid)
    surface_height = get_surface_height(solid_np, nx, ny, nz)
    cx_total, cy = nx / 2.0, ny / 2.0
    if substrate_type == 'concave':
        ridge_top = int(surface_height.max())
    else:
        ridge_top = surface_height[nx // 2, ny // 2]
    cz = ridge_top + 2.0 + R_drop

    d_center = spacing_ratio * D0
    cx1 = max(R_drop + 1, cx_total - d_center / 2.0)
    cx2 = min(nx - R_drop - 1, cx_total + d_center / 2.0)

    C_init, _, u_init = create_multi_droplet_with_impact(
        nx, ny, nz,
        centers=[(cx1, cy, cz), (cx2, cy, cz)],
        radii=[R_drop, R_drop], xi=xi,
        rho_l=rho_l, rho_g=rho_g,
        u_impacts=[(0.0, 0.0, U0), (0.0, 0.0, U0)])
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
            k_history.append({'step': step, 'Dx': Dx, 'Dy': Dy,
                              'k': k_val, 'merged': merged})
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
    tag = "OK" if stable else "FAIL"
    coal_str = f"merge@{coalescence_step}" if coalescence_step else "separate"
    print(f"  {label:>50} d/D={spacing_ratio:.1f} We={We:5.1f} "
          f"| k={max_k:.4f} {coal_str:15s} [{tag}] [{elapsed:.0f}s, {mem_mb:.0f}MB]")

    del solver
    torch.cuda.empty_cache()
    return {
        'label': label, 'substrate_type': substrate_type, 'R_star': R_star,
        'We': We, 'theta_eq': theta_eq, 'amp': amp,
        'spacing_ratio': spacing_ratio,
        'max_k': max_k, 'max_k_step': max_k_step,
        'Dx': Dx_peak, 'Dy': Dy_peak,
        'coalescence_step': coalescence_step,
        'stable': stable, 'grid': f"{nx}x{ny}x{nz}",
        'elapsed_s': round(elapsed, 1), 'mem_mb': round(mem_mb, 0),
        'machine': '30hx_supplementary', 'k_history': k_history,
    }


# ============================================================================
# 3. Sequential multi-droplet (for grid convergence)
# ============================================================================

def run_sequential(substrate_type='ridge', R_star=1.0, We=7.9,
                   theta_eq=162.0, amp=1.8,
                   n_drops=3, dt_star=1.0,
                   N=6000, n_base=150, dtype=torch.float32, label=None):
    """Run sequential multi-droplet case."""
    from lbm.fe_config import FEConfig
    from lbm.fe_ac_solver import AllenCahnSolver
    from lbm.fe_droplet import create_fe_droplet_with_impact
    from geometry.substrate import create_substrate, create_substrate_with_fraction

    sigma = rho_l * U0**2 * D0 / We
    beta = 12.0 * sigma / xi
    kappa = beta * xi**2 / 8.0
    M = 0.02 / beta

    is_flat = substrate_type == 'flat' or R_star is None
    if is_flat:
        nx, ny = n_base, n_base
        nz = int(R_drop + 2 + (1 + n_drops) * R_drop + 20)
    else:
        R_g = abs(R_star) * R_drop
        nx, ny = n_base, n_base
        nz = int(R_g + 2 + R_drop + (1 + n_drops) * R_drop + 20)
    nz = min(nz, 300)

    dt_inject = int(dt_star * D0 / abs(U0))
    inject_steps = [k * dt_inject for k in range(1, n_drops)]

    config = FEConfig(
        nx=nx, ny=ny, nz=nz,
        rho_l=rho_l, rho_g=rho_g,
        sigma=sigma, xi=xi, beta=beta, kappa=kappa,
        M=M, tau_l=tau, tau_g=tau, tau_h=tau + 0.04,
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

    solid_np = solid if isinstance(solid, np.ndarray) else np.array(solid)
    surface_height = get_surface_height(solid_np, nx, ny, nz)
    cx, cy = nx / 2.0, ny / 2.0
    if substrate_type == 'concave':
        ridge_top = int(surface_height.max())
    else:
        ridge_top = surface_height[nx // 2, ny // 2]
    cz = ridge_top + 2.0 + R_drop

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
            interface = (phi_np > 0.5) & ~solid_np
            if interface.any():
                coords = np.argwhere(interface)
                Dx = float(coords[:, 0].max() - coords[:, 0].min() + 1)
                Dy = float(coords[:, 1].max() - coords[:, 1].min() + 1)
            else:
                Dx = Dy = 0
            k_history.append({'step': step, 'Dx': Dx, 'Dy': Dy,
                              'n_deposited': 1 + inject_idx})
            if Dx > max_Dx:
                max_Dx, max_Dy, max_Dx_step = Dx, Dy, step
            if np.isnan(phi_np).any() or phi_np.max() < 0.01:
                stable = False
                break

    elapsed = time.time() - t0
    mem_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    torch.cuda.reset_peak_memory_stats()

    if label is None:
        label = f"seq_{substrate_type}_R{R_star or 'flat'}_n{n_drops}"
    tag = "OK" if stable else "FAIL"
    print(f"  {label:>50} n={n_drops} We={We:5.1f} "
          f"| Dx={max_Dx:.0f} Dy={max_Dy:.0f} [{tag}] [{elapsed:.0f}s, {mem_mb:.0f}MB]")

    del solver
    torch.cuda.empty_cache()
    return {
        'label': label, 'substrate_type': substrate_type, 'R_star': R_star,
        'We': We, 'theta_eq': theta_eq, 'amp': amp,
        'n_drops': n_drops, 'dt_star': dt_star,
        'max_Dx': max_Dx, 'max_Dy': max_Dy, 'max_Dx_step': max_Dx_step,
        'stable': stable, 'grid': f"{nx}x{ny}x{nz}",
        'elapsed_s': round(elapsed, 1), 'mem_mb': round(mem_mb, 0),
        'machine': '30hx_supplementary', 'k_history': k_history,
    }


# ============================================================================
# 4. Film thickness with Monitor
# ============================================================================

def run_film(substrate_type='ridge', R_star=1.0, We=7.9,
             N=4000, n_base=150, dtype=torch.float32, label=None):
    """Run with air film thickness monitoring."""
    from lbm.fe_config import FEConfig
    from lbm.fe_ac_solver import AllenCahnSolver
    from lbm.fe_droplet import create_fe_droplet_with_impact
    from geometry.substrate import create_substrate, create_substrate_with_fraction
    from io_utils.monitor import Monitor

    sigma = rho_l * U0**2 * D0 / We
    beta = 12.0 * sigma / xi
    kappa = beta * xi**2 / 8.0
    M = 0.02 / beta

    is_flat = substrate_type == 'flat' or R_star is None
    if is_flat:
        nx, ny, nz = n_base, n_base, int(R_drop + 2 + R_drop + 20)
    else:
        R_g = abs(R_star) * R_drop
        nx, ny = n_base, n_base
        nz = int(R_g + 2 + R_drop + R_drop + 20)

    config = FEConfig(
        nx=nx, ny=ny, nz=nz,
        rho_l=rho_l, rho_g=rho_g,
        sigma=sigma, xi=xi, beta=beta, kappa=kappa,
        M=M, tau_l=tau, tau_g=tau, tau_h=tau + 0.04,
        theta_eq=162.0, device=DEVICE,
        max_steps=N, output_interval=N + 1,
        g_force=(0.0, 0.0, 0.0),
    )
    solver = AllenCahnSolver(
        config, dtype=dtype, stab_mode='fakhari',
        boundary_relax=0.0, geometric_wetting=True,
        geo_amplification=1.8,
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

    monitor = Monitor(config)

    surface_height = get_surface_height(
        solid if isinstance(solid, np.ndarray) else np.array(solid),
        nx, ny, nz)
    cx, cy = nx / 2.0, ny / 2.0
    ridge_top = surface_height[nx // 2, ny // 2]
    cz = ridge_top + 20.0 + R_drop  # large gap for air film capture

    C_init, _, u_init = create_fe_droplet_with_impact(
        nx, ny, nz, center=(cx, cy, cz), radius=R_drop, xi=xi,
        rho_l=rho_l, rho_g=rho_g, u_impact=(0.0, 0.0, U0))
    solver.init_fields(C_init, u_init)

    t0 = time.time()
    for step in range(1, N + 1):
        solver.step()
        if step % 20 == 0:
            monitor.record(step, solver)

    elapsed = time.time() - t0
    mem_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    torch.cuda.reset_peak_memory_stats()

    # Extract film thickness data
    film_history = []
    for i, step in enumerate(monitor.history['step']):
        film_history.append({
            'step': step,
            'film_thickness': monitor.history['film_thickness'][i],
            'u_max': monitor.history['u_max'][i],
        })

    h_min = min((r['film_thickness'] for r in film_history
                 if r['film_thickness'] is not None), default=0)

    if label is None:
        label = f"film_{substrate_type}_R{R_star or 'flat'}_We{We:.1f}"
    print(f"  {label:>50} We={We:5.1f} | h_min={h_min:.3f} [{elapsed:.0f}s, {mem_mb:.0f}MB]")

    del solver
    torch.cuda.empty_cache()
    return {
        'label': label, 'substrate_type': substrate_type, 'R_star': R_star,
        'We': We, 'h_min': h_min,
        'film_history': film_history,
        'stable': True, 'grid': f"{nx}x{ny}x{nz}",
        'elapsed_s': round(elapsed, 1), 'mem_mb': round(mem_mb, 0),
        'machine': '30hx_supplementary',
    }


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-existing', action='store_true')
    args = parser.parse_args()

    save_dir = os.path.join(_project_root, 'results')
    os.makedirs(save_dir, exist_ok=True)
    main_path = os.path.join(save_dir, 'supplementary_30hx_results.json')
    existing_labels = set()
    if os.path.exists(main_path):
        with open(main_path) as f:
            existing_labels = {r['label'] for r in json.load(f)}

    results = []

    def should_run(lbl):
        if args.skip_existing and lbl in existing_labels:
            print(f"  SKIP: {lbl}")
            return False
        return True

    # =================================================================
    # STUDY 12a: Dual-droplet grid convergence (4 cases)
    # Ridge R*=1.0, d/D=1.2, We=7.9, θ=162°
    # =================================================================
    print("\n" + "=" * 70)
    print("  STUDY 12a: Dual-droplet grid convergence")
    print("=" * 70)
    for n_base in [80, 100, 120, 150]:
        lbl = f"s12a_dual_ridge_R1.0_d1.2_grid{n_base}"
        if should_run(lbl):
            results.append(run_dual(
                substrate_type='ridge', R_star=1.0, We=7.9,
                spacing_ratio=1.2, n_base=n_base, label=lbl))

    # =================================================================
    # STUDY 12b: Multi-droplet grid convergence (3 cases)
    # Ridge R*=1.0, n_drops=3, dt*=1.0
    # =================================================================
    print("\n" + "=" * 70)
    print("  STUDY 12b: Multi-droplet grid convergence (n=3)")
    print("=" * 70)
    for n_base in [80, 100, 120]:
        lbl = f"s12b_seq_ridge_R1.0_n3_grid{n_base}"
        if should_run(lbl):
            results.append(run_sequential(
                substrate_type='ridge', R_star=1.0, We=7.9,
                n_drops=3, n_base=n_base, label=lbl))

    # =================================================================
    # STUDY 12c: Concave re-runs with fix (5 cases)
    # Verify corrected concave geometry gives different results
    # =================================================================
    print("\n" + "=" * 70)
    print("  STUDY 12c: Concave re-runs (fraction bug fix)")
    print("=" * 70)
    # Single droplet: key We values
    for We in [5.0, 7.9, 15.0]:
        lbl = f"s12c_single_concave_R1.0_We{We:.1f}_fixed"
        if should_run(lbl):
            results.append(run_single(
                substrate_type='concave', R_star=1.0, We=We,
                n_base=150, label=lbl,
                save_snapshots=(We == 7.9)))
    # Single droplet: theta variation
    lbl = "s12c_single_concave_R1.0_theta90_fixed"
    if should_run(lbl):
        results.append(run_single(
            substrate_type='concave', R_star=1.0, We=7.9,
            theta_eq=90.0, amp=1.8, n_base=150, label=lbl))
    # Dual droplet
    lbl = "s12c_dual_concave_R1.0_d1.2_fixed"
    if should_run(lbl):
        results.append(run_dual(
            substrate_type='concave', R_star=1.0, We=7.9,
            spacing_ratio=1.2, n_base=150, label=lbl,
            save_snapshots=True))

    # =================================================================
    # STUDY 12d: Film thickness We scan (5 cases)
    # Ridge R*=1.0, We dependence of air film
    # =================================================================
    print("\n" + "=" * 70)
    print("  STUDY 12d: Film thickness We scan (ridge R*=1.0)")
    print("=" * 70)
    for We in [3.0, 5.0, 10.0, 15.0, 20.0]:
        lbl = f"s12d_film_ridge_R1.0_We{We:.1f}"
        if should_run(lbl):
            results.append(run_film(
                substrate_type='ridge', R_star=1.0, We=We,
                n_base=120, label=lbl))

    # =================================================================
    # STUDY 12e: Long time evolution (3 cases)
    # N=10000 with fine snapshot interval for time-resolved analysis
    # =================================================================
    print("\n" + "=" * 70)
    print("  STUDY 12e: Long time evolution (N=10000)")
    print("=" * 70)
    for sub in ['ridge', 'convex', 'concave']:
        lbl = f"s12e_single_{sub}_R1.0_long"
        if should_run(lbl):
            results.append(run_single(
                substrate_type=sub, R_star=1.0, We=7.9,
                N=10000, n_base=150, label=lbl,
                save_snapshots=True))

    # =================================================================
    # STUDY 12f: Line formation grid convergence (3 cases)
    # Flat p/D0=0.8, n_drops=5 at different resolutions
    # Validates Chapter 4 results
    # =================================================================
    print("\n" + "=" * 70)
    print("  STUDY 12f: Line formation grid convergence (flat p/D0=0.8)")
    print("=" * 70)
    # Import line formation function from Chapter 4 script
    sys.path.insert(0, os.path.join(_project_root, 'scripts'))
    from run_chapter4_line_formation import run_line_case

    for n_base in [80, 100, 120]:
        lbl = f"s12f_line_flat_p0.8_grid{n_base}"
        if should_run(lbl):
            results.append(run_line_case(
                substrate_type='flat', R_star=None,
                p_ratio=0.8, We=7.9, theta_eq=162.0, amp=0.0,
                n_drops=5, n_base=n_base, label=lbl,
                save_snapshots=(n_base == 120)))

    # =================================================================
    # Summary and save
    # =================================================================
    print("\n" + "=" * 70)
    print("  30HX Supplementary Results Summary")
    print("=" * 70)
    ok = sum(1 for r in results if r['stable'])
    print(f"\n  Total: {len(results)} cases, {ok} OK, {len(results) - ok} FAIL")

    for study in ['s12a', 's12b', 's12c', 's12d', 's12e', 's12f']:
        study_cases = [r for r in results if r['label'].startswith(study)]
        if study_cases:
            print(f"\n  --- {study} ---")
            for r in study_cases:
                tag = "OK" if r['stable'] else "FAIL"
                k = r.get('max_k', r.get('h_min', r.get('max_Dx', 0)))
                print(f"    {r['label']:>50} grid={r.get('grid','')} [{tag}]")

    # Save
    if os.path.exists(main_path):
        with open(main_path) as f:
            all_data = json.load(f)
    else:
        all_data = []

    existing_set = {r['label'] for r in all_data}
    new_count = 0
    for r in results:
        if r['label'] not in existing_set:
            all_data.append({k: v for k, v in r.items()
                             if k not in ('k_history', 'film_history')})
            existing_set.add(r['label'])
            new_count += 1

    with open(main_path, 'w') as f:
        json.dump(all_data, f, indent=2, default=str)
    print(f"\n  Saved {new_count} new results to {main_path}")

    # Save histories separately
    hist_path = os.path.join(save_dir, 'supplementary_30hx_history.json')
    hist_data = {}
    for r in results:
        hist = r.get('k_history') or r.get('film_history')
        if hist:
            hist_data[r['label']] = hist
    if os.path.exists(hist_path):
        with open(hist_path) as f:
            existing_hist = json.load(f)
        existing_hist.update(hist_data)
        hist_data = existing_hist
    with open(hist_path, 'w') as f:
        json.dump(hist_data, f)

    elapsed_total = sum(r['elapsed_s'] for r in results)
    print(f"\n  Total: {elapsed_total:.0f}s ({elapsed_total / 3600:.1f}h)")


if __name__ == "__main__":
    main()
