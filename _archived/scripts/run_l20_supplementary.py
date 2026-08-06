#!/usr/bin/env python3
"""L20 Supplementary Simulations (Studies 13a-13h, ~36 cases).

Covers all remaining gaps for Chapters 3-5:
  13a: Concave single-droplet re-runs     (14 cases) — bug fix (fraction inversion)
  13b: Concave multi-droplet re-runs       ( 3 cases) — bug fix for s10 data
  13c: Line formation — uniform/scalloped  ( 8 cases) — transition morphology
  13d: Line formation — θ on non-ridge     ( 6 cases) — complete substrate × θ matrix
  13e: Dual droplet spacing supplement     ( 3 cases) — fill s9b gaps
  13f: Line formation — large R*           ( 3 cases) — R*=2.76,4.0 matching Liu 2015
  13g: Concave R*=2.0 We sweep            ( 4 cases) — Chapter 3 concave extension
  13h: Line formation grid convergence     ( 3 cases) — ridge p/D=0.8 resolution check

Machine: NVIDIA L20 (48GB VRAM)
Estimated runtime: ~6-8 hours

Usage: python3 scripts/run_l20_supplementary.py [--skip-existing]
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

# ── Device detection ──
if torch.cuda.is_available():
    DEVICE = 'cuda'
    torch.cuda.empty_cache()
    print(f"GPU (CUDA): {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB"
          if hasattr(torch.cuda.get_device_properties(0), 'total_mem')
          else f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
elif hasattr(torch, 'xpu') and torch.xpu.is_available():
    DEVICE = 'xpu'
    torch.xpu.empty_cache()
    print("GPU (XPU): Intel Arc")
else:
    DEVICE = 'cpu'
    print("WARNING: No GPU, using CPU")

# ── Physical constants ──
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


def _get_mem_mb():
    if DEVICE == 'cuda':
        mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        torch.cuda.reset_peak_memory_stats()
        return mb
    elif DEVICE == 'xpu':
        try:
            mb = torch.xpu.max_memory_allocated() / 1024 / 1024
            torch.xpu.reset_peak_memory_stats()
            return mb
        except Exception:
            return 0
    return 0


def _clear_cache():
    if DEVICE == 'cuda':
        torch.cuda.empty_cache()
    elif DEVICE == 'xpu':
        try: torch.xpu.empty_cache()
        except: pass


# ============================================================================
# 1. Single droplet impact
# ============================================================================

def run_single(substrate_type='ridge', R_star=1.0, We=7.9,
               theta_eq=162.0, amp=1.8, N=4000,
               n_base=150, dtype=torch.float32, label=None,
               save_snapshots=False):
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

    snap_dir = os.path.join(_project_root, 'results', 'thesis_snapshots')
    os.makedirs(snap_dir, exist_ok=True)
    snap_prefix = label or f"single_{substrate_type}_R{R_star or 'flat'}"
    snap_steps = set()
    if save_snapshots:
        for s in range(200, N + 1, 200):
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
            np.savez_compressed(
                os.path.join(snap_dir, f'{snap_prefix}_step{step}.npz'),
                xz_slice=phi_np[:, mid_j, :],
                xz_solid=solid_np[:, mid_j, :],
                step=step, nx=nx, ny=ny, nz=nz)

    elapsed = time.time() - t0
    mem_mb = _get_mem_mb()
    if label is None:
        label = f"single_{substrate_type}_R{R_star or 'flat'}"
    tag = "OK" if stable else "FAIL"
    print(f"  {label:>55} We={We:5.1f} θ={theta_eq:3.0f} "
          f"| k={max_k:.4f} Dx={Dx_peak:.0f} Dy={Dy_peak:.0f} "
          f"[{tag}] [{elapsed:.0f}s, {mem_mb:.0f}MB] {nx}x{ny}x{nz}")

    del solver
    _clear_cache()
    return {
        'label': label, 'substrate_type': substrate_type, 'R_star': R_star,
        'We': We, 'theta_eq': theta_eq, 'amp': amp,
        'max_k': max_k, 'max_k_step': max_k_step,
        'Dx': Dx_peak, 'Dy': Dy_peak,
        'stable': stable, 'grid': f"{nx}x{ny}x{nz}",
        'elapsed_s': round(elapsed, 1), 'mem_mb': round(mem_mb, 0),
        'machine': 'L20_supplementary', 'k_history': k_history,
    }


# ============================================================================
# 2. Dual droplet
# ============================================================================

def run_dual(substrate_type='ridge', R_star=1.0, We=7.9,
             theta_eq=162.0, amp=1.8, spacing_ratio=1.2,
             N=4000, n_base=150, dtype=torch.float32, label=None):
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
    mem_mb = _get_mem_mb()
    if label is None:
        label = f"dual_{substrate_type}_R{R_star or 'flat'}_d{spacing_ratio:.1f}"
    tag = "OK" if stable else "FAIL"
    coal_str = f"merge@{coalescence_step}" if coalescence_step else "separate"
    print(f"  {label:>55} d/D={spacing_ratio:.1f} We={We:5.1f} "
          f"| k={max_k:.4f} {coal_str:15s} [{tag}] [{elapsed:.0f}s, {mem_mb:.0f}MB]")

    del solver
    _clear_cache()
    return {
        'label': label, 'substrate_type': substrate_type, 'R_star': R_star,
        'We': We, 'theta_eq': theta_eq, 'amp': amp,
        'spacing_ratio': spacing_ratio,
        'max_k': max_k, 'max_k_step': max_k_step,
        'Dx': Dx_peak, 'Dy': Dy_peak,
        'coalescence_step': coalescence_step,
        'stable': stable, 'grid': f"{nx}x{ny}x{nz}",
        'elapsed_s': round(elapsed, 1), 'mem_mb': round(mem_mb, 0),
        'machine': 'L20_supplementary', 'k_history': k_history,
    }


# ============================================================================
# 3. Sequential multi-droplet
# ============================================================================

def run_sequential(substrate_type='ridge', R_star=1.0, We=7.9,
                   theta_eq=162.0, amp=1.8,
                   n_drops=3, dt_star=1.0,
                   N=6000, n_base=150, dtype=torch.float32, label=None):
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
    mem_mb = _get_mem_mb()
    if label is None:
        label = f"seq_{substrate_type}_R{R_star or 'flat'}_n{n_drops}"
    tag = "OK" if stable else "FAIL"
    print(f"  {label:>55} n={n_drops} We={We:5.1f} "
          f"| Dx={max_Dx:.0f} Dy={max_Dy:.0f} [{tag}] [{elapsed:.0f}s, {mem_mb:.0f}MB]")

    del solver
    _clear_cache()
    return {
        'label': label, 'substrate_type': substrate_type, 'R_star': R_star,
        'We': We, 'theta_eq': theta_eq, 'amp': amp,
        'n_drops': n_drops, 'dt_star': dt_star,
        'max_Dx': max_Dx, 'max_Dy': max_Dy, 'max_Dx_step': max_Dx_step,
        'stable': stable, 'grid': f"{nx}x{ny}x{nz}",
        'elapsed_s': round(elapsed, 1), 'mem_mb': round(mem_mb, 0),
        'machine': 'L20_supplementary', 'k_history': k_history,
    }


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='L20 supplementary simulations')
    parser.add_argument('--skip-existing', action='store_true')
    args = parser.parse_args()

    save_dir = os.path.join(_project_root, 'results')
    os.makedirs(save_dir, exist_ok=True)
    main_path = os.path.join(save_dir, 'supplementary_l20_results.json')
    existing_labels = set()
    if os.path.exists(main_path):
        with open(main_path) as f:
            existing_labels = {r['label'] for r in json.load(f)}
        print(f"Loaded {len(existing_labels)} existing results")

    # Also load existing results from original data files for skip logic
    for fname in ['thesis_sweep_results.json', 'thesis_bigblack_results.json',
                  'thesis_qingqing_results.json', 'studies_9_10.json',
                  'chapter4_line_results.json']:
        fpath = os.path.join(save_dir, fname)
        if os.path.exists(fpath):
            try:
                with open(fpath) as f:
                    for r in json.load(f):
                        existing_labels.add(r['label'])
            except Exception:
                pass

    results = []

    def should_run(lbl):
        if args.skip_existing and lbl in existing_labels:
            print(f"  SKIP: {lbl}")
            return False
        return True

    # =================================================================
    # STUDY 13a: Concave single-droplet re-runs (14 cases)
    # All Chapter 3 concave cases had k≈1.0 due to fraction field bug
    # Re-run with identical parameters on fixed codebase
    # =================================================================
    print("\n" + "=" * 70)
    print("  STUDY 13a: Concave single-droplet re-runs (fraction bug fix)")
    print("=" * 70)

    # --- s2: Substrate type comparison (3 cases) ---
    for R_star, theta in [(1.0, 162.0), (2.0, 162.0)]:
        lbl = f"s2_concave_R{R_star:.1f}"
        if should_run(lbl):
            results.append(run_single(
                substrate_type='concave', R_star=R_star, We=7.9,
                theta_eq=theta, amp=1.8, N=4000, n_base=150,
                label=lbl, save_snapshots=(R_star == 1.0)))
    # s7 time dynamics
    lbl = "s7_concave_R1.0_We7.9"
    if should_run(lbl):
        results.append(run_single(
            substrate_type='concave', R_star=1.0, We=7.9,
            theta_eq=162.0, amp=1.8, N=4000, n_base=150,
            label=lbl, save_snapshots=True))

    # --- s3b: Concave We sweep (6 cases) ---
    for We in [5.0, 7.9, 10.0, 12.0, 15.0, 20.0]:
        lbl = f"s3b_concave_R1.0_We{We}"
        if should_run(lbl):
            results.append(run_single(
                substrate_type='concave', R_star=1.0, We=We,
                theta_eq=162.0, amp=1.8, N=4000, n_base=150,
                label=lbl,
                save_snapshots=(We in [7.9, 15.0])))

    # --- s4b: Concave θ sweep (5 cases) ---
    for theta in [60, 90, 120, 140, 162]:
        lbl = f"s4b_concave_R1.0_theta{theta}"
        if should_run(lbl):
            results.append(run_single(
                substrate_type='concave', R_star=1.0, We=7.9,
                theta_eq=float(theta), amp=1.8, N=4000, n_base=150,
                label=lbl,
                save_snapshots=(theta in [90, 162])))

    # =================================================================
    # STUDY 13b: Concave multi-droplet re-runs (3 cases)
    # s10 concave cases had k=0.000 due to same bug
    # =================================================================
    print("\n" + "=" * 70)
    print("  STUDY 13b: Concave multi-droplet re-runs")
    print("=" * 70)

    # s10a: single substrate comparison n=2
    lbl = "s10a_seq_concave_R1.0"
    if should_run(lbl):
        results.append(run_sequential(
            substrate_type='concave', R_star=1.0, We=7.9,
            n_drops=2, dt_star=1.0, N=6000, n_base=150,
            label=lbl))

    # s10e: spacing on concave
    lbl = "s10e_concave_R1.0_n2"
    if should_run(lbl):
        results.append(run_sequential(
            substrate_type='concave', R_star=1.0, We=7.9,
            n_drops=2, dt_star=1.0, N=6000, n_base=150,
            label=lbl))

    # s10h: concave We=15
    lbl = "s10h_concave_R1.0_We15"
    if should_run(lbl):
        results.append(run_sequential(
            substrate_type='concave', R_star=1.0, We=15.0,
            n_drops=2, dt_star=1.0, N=6000, n_base=150,
            label=lbl))

    # =================================================================
    # STUDY 13c: Line formation — uniform/scalloped transition (8 cases)
    # Current data only shows bulging/isolated. Explore transition:
    #   - Very dense spacing (p/D=0.2) for uniform
    #   - Low We (1.0, 2.0) for gentler impact → smoother lines
    #   - More droplets (n=15) for longer equilibration
    #   - Flat surface at various p/D for baseline
    # =================================================================
    print("\n" + "=" * 70)
    print("  STUDY 13c: Line formation — uniform/scalloped transition")
    print("=" * 70)

    sys.path.insert(0, os.path.join(_project_root, 'scripts'))
    from run_chapter4_line_formation import run_line_case

    # Dense spacing on flat (should promote uniform)
    for p in [0.15, 0.2]:
        lbl = f"s13c_flat_p{p:.2f}_We7.9"
        if should_run(lbl):
            results.append(run_line_case(
                substrate_type='flat', R_star=None,
                p_ratio=p, We=7.9, theta_eq=162.0, amp=0.0,
                n_drops=8, n_base=150, label=lbl,
                save_snapshots=(p == 0.2)))

    # Low We on ridge (gentler impact, smoother lines)
    for We in [1.0, 2.0]:
        lbl = f"s13c_ridge_p0.8_We{We:.1f}"
        if should_run(lbl):
            results.append(run_line_case(
                substrate_type='ridge', R_star=1.0,
                p_ratio=0.8, We=We, theta_eq=162.0, amp=1.8,
                n_drops=5, n_base=150, label=lbl,
                save_snapshots=(We == 2.0)))

    # More droplets at moderate spacing on ridge
    lbl = "s13c_ridge_p0.5_We7.9_n15"
    if should_run(lbl):
        results.append(run_line_case(
            substrate_type='ridge', R_star=1.0,
            p_ratio=0.5, We=7.9, theta_eq=162.0, amp=1.8,
            n_drops=15, n_base=150, label=lbl,
            save_snapshots=True))

    # Dense spacing on ridge
    lbl = "s13c_ridge_p0.2_We7.9"
    if should_run(lbl):
        results.append(run_line_case(
            substrate_type='ridge', R_star=1.0,
            p_ratio=0.2, We=7.9, theta_eq=162.0, amp=1.8,
            n_drops=8, n_base=150, label=lbl,
            save_snapshots=True))

    # High theta (more wetting → wider, more uniform)
    lbl = "s13c_ridge_p0.8_We7.9_theta170"
    if should_run(lbl):
        results.append(run_line_case(
            substrate_type='ridge', R_star=1.0,
            p_ratio=0.8, We=7.9, theta_eq=170.0, amp=1.8,
            n_drops=5, n_base=150, label=lbl,
            save_snapshots=True))

    # Flat with more droplets
    lbl = "s13c_flat_p0.5_We7.9_n10"
    if should_run(lbl):
        results.append(run_line_case(
            substrate_type='flat', R_star=None,
            p_ratio=0.5, We=7.9, theta_eq=162.0, amp=0.0,
            n_drops=10, n_base=150, label=lbl))

    # =================================================================
    # STUDY 13d: Line formation — θ sweep on non-ridge substrates (6 cases)
    # Complete substrate × θ matrix for line formation
    # =================================================================
    print("\n" + "=" * 70)
    print("  STUDY 13d: Line formation — θ on flat/convex/concave")
    print("=" * 70)

    for sub, R_star, amp in [('flat', None, 0.0),
                              ('convex', 1.0, 1.8),
                              ('concave', 1.0, 1.8)]:
        for theta in [90, 150]:
            R_tag = 'flat' if R_star is None else f'R{R_star:.1f}'
            lbl = f"s13d_{sub}_{R_tag}_p0.8_theta{theta}"
            if should_run(lbl):
                results.append(run_line_case(
                    substrate_type=sub, R_star=R_star,
                    p_ratio=0.8, We=7.9, theta_eq=float(theta), amp=amp,
                    n_drops=5, n_base=150, label=lbl,
                    save_snapshots=(theta == 150)))

    # =================================================================
    # STUDY 13e: Dual droplet spacing supplement (3 cases)
    # Fill gaps in s9b spacing scan
    # =================================================================
    print("\n" + "=" * 70)
    print("  STUDY 13e: Dual droplet spacing supplement (ridge)")
    print("=" * 70)

    for d in [0.8, 1.8, 2.5]:
        lbl = f"s13e_dual_ridge_R1.0_d{d:.1f}"
        if should_run(lbl):
            results.append(run_dual(
                substrate_type='ridge', R_star=1.0, We=7.9,
                spacing_ratio=d, n_base=150, label=lbl))

    # =================================================================
    # STUDY 13f: Line formation — large R* (3 cases)
    # Match Liu 2015 experimental R* values: R*=2.76 and R*=4.0
    # =================================================================
    print("\n" + "=" * 70)
    print("  STUDY 13f: Line formation — large R* on ridge")
    print("=" * 70)

    for R_star in [2.76, 4.0]:
        lbl = f"s13f_ridge_R{R_star:.2f}_p0.8"
        if should_run(lbl):
            results.append(run_line_case(
                substrate_type='ridge', R_star=R_star,
                p_ratio=0.8, We=7.9, theta_eq=162.0, amp=1.8,
                n_drops=5, n_base=150, label=lbl,
                save_snapshots=True))

    # Also test flat R*=4.0 equivalent
    lbl = "s13f_flat_p0.8_We7.9_n5"
    if should_run(lbl):
        results.append(run_line_case(
            substrate_type='flat', R_star=None,
            p_ratio=0.8, We=7.9, theta_eq=162.0, amp=0.0,
            n_drops=5, n_base=150, label=lbl))

    # =================================================================
    # STUDY 13g: Concave R*=2.0 We sweep (4 cases)
    # Extend Chapter 3 concave data beyond R*=1.0
    # =================================================================
    print("\n" + "=" * 70)
    print("  STUDY 13g: Concave R*=2.0 We sweep")
    print("=" * 70)

    for We in [5.0, 7.9, 15.0, 25.0]:
        lbl = f"s13g_concave_R2.0_We{We:.1f}"
        if should_run(lbl):
            results.append(run_single(
                substrate_type='concave', R_star=2.0, We=We,
                theta_eq=162.0, amp=1.0, N=4000, n_base=150,
                label=lbl,
                save_snapshots=(We == 7.9)))

    # =================================================================
    # STUDY 13h: Line formation grid convergence (3 cases)
    # Ridge p/D=0.8 at different resolutions
    # =================================================================
    print("\n" + "=" * 70)
    print("  STUDY 13h: Line formation grid convergence (ridge p/D=0.8)")
    print("=" * 70)

    for n_base in [80, 100, 120]:
        lbl = f"s13h_line_ridge_R1.0_p0.8_grid{n_base}"
        if should_run(lbl):
            results.append(run_line_case(
                substrate_type='ridge', R_star=1.0,
                p_ratio=0.8, We=7.9, theta_eq=162.0, amp=1.8,
                n_drops=5, n_base=n_base, label=lbl,
                save_snapshots=(n_base == 120)))

    # =================================================================
    # Summary and save
    # =================================================================
    print("\n" + "=" * 70)
    print("  L20 Supplementary Results Summary")
    print("=" * 70)
    ok = sum(1 for r in results if r.get('stable', True))
    fail = len(results) - ok
    print(f"\n  Total: {len(results)} cases, {ok} OK, {fail} FAIL")

    for study in ['s13a', 's13b', 's13c', 's13d', 's13e', 's13f', 's13g', 's13h']:
        study_cases = [r for r in results if r['label'].startswith(study)]
        if study_cases:
            print(f"\n  --- {study} ({len(study_cases)} cases) ---")
            for r in study_cases:
                tag = "OK" if r.get('stable', True) else "FAIL"
                k = r.get('max_k', r.get('max_Dx', 0))
                morph = r.get('morphology', '')
                print(f"    {r['label']:>55} "
                      f"k/w={'%.4f' % k if 'max_k' in r else '%.0f' % k} "
                      f"{morph:10s} [{tag}]")

    # ── Save main results ──
    if os.path.exists(main_path):
        with open(main_path) as f:
            all_data = json.load(f)
    else:
        all_data = []

    existing_set = {r['label'] for r in all_data}
    new_count = 0
    for r in results:
        if r['label'] not in existing_set:
            # Strip heavy data
            all_data.append({k: v for k, v in r.items()
                             if k not in ('k_history', 'w_profile_final',
                                          'h_profile_final', 'history')})
            existing_set.add(r['label'])
            new_count += 1

    with open(main_path, 'w') as f:
        json.dump(all_data, f, indent=2, default=str)
    print(f"\n  Saved {new_count} new results to {main_path}")

    # ── Update original result files with fixed concave data ──
    print("\n  Updating original result files with fixed concave data...")
    _update_original_results(results, save_dir)

    # ── Save histories ──
    hist_path = os.path.join(save_dir, 'supplementary_l20_history.json')
    hist_data = {}
    for r in results:
        hist = r.get('k_history') or r.get('history')
        if hist:
            hist_data[r['label']] = hist
    if os.path.exists(hist_path):
        with open(hist_path) as f:
            existing_hist = json.load(f)
        existing_hist.update(hist_data)
        hist_data = existing_hist
    with open(hist_path, 'w') as f:
        json.dump(hist_data, f)
    print(f"  History saved ({len(hist_data)} entries)")

    elapsed_total = sum(r['elapsed_s'] for r in results)
    print(f"\n  Total runtime: {elapsed_total:.0f}s ({elapsed_total / 3600:.1f}h)")


def _update_original_results(results, save_dir):
    """Update original result files (thesis_sweep, thesis_bigblack, thesis_qingqing)
    with fixed concave data from re-runs."""
    # Map of label prefix → original file
    file_map = {
        's2_': 'thesis_sweep_results.json',
        's7_': 'thesis_sweep_results.json',
        's3b_': 'thesis_bigblack_results.json',
        's4b_': 'thesis_qingqing_results.json',
    }
    # Map of label prefix → history file
    hist_file_map = {
        's2_': 'thesis_k_history.json',
        's7_': 'thesis_k_history.json',
        's3b_': 'thesis_bigblack_history.json',
        's4b_': 'thesis_qingqing_history.json',
    }

    for prefix, fname in file_map.items():
        fpath = os.path.join(save_dir, fname)
        if not os.path.exists(fpath):
            continue
        with open(fpath) as f:
            data = json.load(f)

        updated = False
        for r_new in results:
            if not r_new['label'].startswith(prefix):
                continue
            if 'max_k' not in r_new:
                continue  # Skip non-single-droplet results
            # Find and replace the bugged entry
            for i, r_old in enumerate(data):
                if r_old['label'] == r_new['label']:
                    # Keep original metadata, update key fields
                    data[i]['max_k'] = r_new['max_k']
                    data[i]['max_k_step'] = r_new.get('max_k_step',
                                                       r_old.get('max_k_step'))
                    data[i]['Dx'] = r_new.get('Dx', r_old.get('Dx'))
                    data[i]['Dy'] = r_new.get('Dy', r_old.get('Dy'))
                    data[i]['stable'] = r_new.get('stable', True)
                    data[i]['grid'] = r_new.get('grid', r_old.get('grid'))
                    data[i]['machine'] = 'L20_fixed'
                    updated = True
                    print(f"    Updated {r_new['label']}: k={r_new['max_k']:.4f}")
                    break

        if updated:
            with open(fpath, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"    → Saved {fname}")

    # Also update studies_9_10.json with fixed s10 concave
    fpath = os.path.join(save_dir, 'studies_9_10.json')
    if os.path.exists(fpath):
        with open(fpath) as f:
            data = json.load(f)
        updated = False
        for r_new in results:
            if not r_new['label'].startswith('s10'):
                continue
            if 'max_Dx' not in r_new:
                continue
            for i, r_old in enumerate(data):
                if r_old['label'] == r_new['label']:
                    data[i]['max_Dx'] = r_new['max_Dx']
                    data[i]['max_Dy'] = r_new['max_Dy']
                    data[i]['max_k'] = r_new['max_Dx'] / r_new['max_Dy'] if r_new['max_Dy'] > 0 else 0
                    data[i]['Dx'] = r_new['max_Dx']
                    data[i]['Dy'] = r_new['max_Dy']
                    data[i]['stable'] = r_new.get('stable', True)
                    data[i]['machine'] = 'L20_fixed'
                    updated = True
                    print(f"    Updated {r_new['label']}: Dx={r_new['max_Dx']:.0f}")
                    break
        if updated:
            with open(fpath, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"    → Saved studies_9_10.json")


if __name__ == "__main__":
    main()
