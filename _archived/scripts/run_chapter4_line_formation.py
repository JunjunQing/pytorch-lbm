#!/usr/bin/env python3
"""Chapter 4: Line Formation on Curved Surfaces (Studies 11a-11i, 39 cases).

Systematic study of inkjet-style line morphology on curved substrates.
Extends Soltman et al. (2008) flat-surface regime map to curved geometries.

Sub-studies:
  11a: Flat baseline spacing scan    (5 cases)  — reproduce Soltman regimes
  11b: Ridge spacing scan            (5 cases)  — curvature effect on regimes
  11c: Convex spacing scan           (3 cases)  — dome substrate
  11d: Concave spacing scan          (3 cases)  — cavity substrate
  11e: Weber number effect           (5 cases)  — inertia dependence
  11f: Contact angle effect          (4 cases)  — wettability dependence
  11g: R* curvature effect           (4 cases)  — curvature ratio scan
  11h: Droplet count effect          (3 cases)  — line length convergence
  11i: Multi-layer stacking          (3 cases)  — aligned/staggered 2-layer

Machine: NVIDIA L20 (48GB VRAM) or similar
Estimated runtime: ~8-9 hours

Usage: python3 scripts/run_chapter4_line_formation.py [--skip-existing]
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
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
elif hasattr(torch, 'xpu') and torch.xpu.is_available():
    DEVICE = 'xpu'
    torch.xpu.empty_cache()
    print("GPU (XPU): Intel Arc")
else:
    DEVICE = 'cpu'
    print("WARNING: No GPU detected, using CPU (very slow)")

# ── Physical constants (same as Chapter 3) ──
D0 = 45.0
R_drop = D0 / 2.0
rho_l = 1.0
rho_g = 1.0 / 828.0
xi = 4.0
tau = 0.53
U0 = -0.05  # downward impact velocity


# ============================================================================
# Measurement functions
# ============================================================================

def measure_line_profile(phi, solid, nx, ny, nz):
    """Measure line width w(y) and height h(y) along the y-axis.

    Returns
    -------
    w : ndarray (ny,) — line width at each y position
    h : ndarray (ny,) — line height at each y position
    """
    w = np.zeros(ny, dtype=np.float64)
    h = np.zeros(ny, dtype=np.float64)
    for j in range(ny):
        slice_2d = phi[:, j, :]
        solid_2d = solid[:, j, :]
        liquid = (slice_2d > 0.5) & ~solid_2d
        if liquid.any():
            coords = np.argwhere(liquid)
            w[j] = coords[:, 0].max() - coords[:, 0].min() + 1
            h[j] = coords[:, 1].max() - coords[:, 1].min() + 1
    return w, h


def measure_line_cross_section(phi, solid, nx, ny, nz, y_positions):
    """Extract cross-sectional profiles at specific y positions.

    Returns dict: {y_idx: {'width_x': float, 'height_z': float}}
    """
    result = {}
    for j in y_positions:
        if j < 0 or j >= ny:
            continue
        slice_2d = phi[:, j, :]
        solid_2d = solid[:, j, :]
        liquid = (slice_2d > 0.5) & ~solid_2d
        if liquid.any():
            coords = np.argwhere(liquid)
            result[j] = {
                'width_x': float(coords[:, 0].max() - coords[:, 0].min() + 1),
                'height_z': float(coords[:, 1].max() - coords[:, 1].min() + 1),
            }
        else:
            result[j] = {'width_x': 0.0, 'height_z': 0.0}
    return result


def classify_morphology(w_profile, p_lattice, D0):
    """Classify line morphology using width coefficient of variation.

    Based on Soltman et al. (2008) regime map logic:
      - isolated:  individual drops, no merging (high CV, gaps present)
      - scalloped:  partial merging with visible individual signatures
      - uniform:   smooth uniform line (low CV)
      - bulging:   over-merged with width variations at junctions
    """
    nonzero = w_profile[w_profile > 1.0]
    if len(nonzero) < 3:
        return 'isolated', 0.0, 0.0

    mean_w = nonzero.mean()
    std_w = nonzero.std()
    cv = std_w / mean_w if mean_w > 0 else 0

    # Check for gaps (isolated segments)
    is_active = w_profile > 1.0
    segments = np.diff(is_active.astype(int))
    n_starts = np.sum(segments == 1)
    n_gaps = np.sum(segments == -1)

    # If multiple separated segments → isolated drops
    if n_starts >= 2:
        gap_fraction = n_gaps / max(n_starts, 1)
        if gap_fraction > 0.3:
            return 'isolated', cv, float(mean_w)

    # Morphology based on CV
    if cv < 0.06:
        return 'uniform', cv, float(mean_w)
    elif cv < 0.15:
        return 'scalloped', cv, float(mean_w)
    else:
        return 'bulging', cv, float(mean_w)


# ============================================================================
# Main simulation function
# ============================================================================

def get_surface_height(solid, nx, ny, nz):
    """Get surface height map from solid mask."""
    height = np.zeros((nx, ny), dtype=int)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz - 1, -1, -1):
                if solid[i, j, k]:
                    height[i, j] = k
                    break
    return height


def run_line_case(
    substrate_type='ridge', R_star=None,
    p_ratio=0.8, We=7.9, theta_eq=162.0, amp=1.8,
    n_drops=5, dt_star=1.0,
    n_layers=1, stagger_offset=0.0,
    N=None, n_base=150, dtype=torch.float32,
    label=None, save_snapshots=False,
):
    """Run a single line formation case.

    Droplets are deposited sequentially along the y-axis at spacing p = p_ratio * D0.
    For multi-layer cases, second layer is deposited after first layer settles.

    Parameters
    ----------
    substrate_type : str — 'flat', 'ridge', 'convex', 'concave'
    R_star : float or None — substrate curvature ratio
    p_ratio : float — p/D0, droplet center-to-center spacing ratio
    We : float — Weber number
    theta_eq : float — equilibrium contact angle (degrees)
    amp : float — geometric wetting amplification
    n_drops : int — droplets per layer
    dt_star : float — dimensionless injection interval
    n_layers : int — number of layers (1 or 2)
    stagger_offset : float — y-offset for second layer as fraction of p_ratio (0=aligned, 0.5=staggered)
    N : int or None — total steps (auto-computed if None)
    n_base : int — base grid resolution in x
    """
    sigma = rho_l * U0 ** 2 * D0 / We
    beta = 12.0 * sigma / xi
    kappa = beta * xi ** 2 / 8.0
    M = 0.02 / beta
    tau_l, tau_g = tau, tau

    p_lattice = p_ratio * D0

    # ── Grid sizing ──
    is_flat = substrate_type == 'flat' or R_star is None
    nx = n_base
    # ny must accommodate all droplets along y
    ny = int((n_drops - 1) * p_lattice + D0 + 30)

    if is_flat:
        # For multi-layer, need extra height
        nz_per_layer = int(R_drop + 2 + R_drop + 15)
        nz = int(nz_per_layer * n_layers + 10)
    else:
        R_g = abs(R_star) * R_drop
        nz_per_layer = int(R_drop + 2 * R_drop + 15)
        nz = int(R_g + 2 + nz_per_layer * n_layers + 10)
    nz = min(nz, 350)

    # Auto-compute total steps
    if N is None:
        # First layer injections + settling + second layer injections + final settling
        layer_steps = (n_drops - 1) * int(dt_star * D0 / abs(U0)) + 2500
        if n_layers == 1:
            N = layer_steps
        else:
            # Extra settling between layers
            N = layer_steps + 1500 + layer_steps

    dt_inject = int(dt_star * D0 / abs(U0))

    # ── Import (deferred to avoid slow startup for --help) ──
    from lbm.fe_config import FEConfig
    from lbm.fe_ac_solver import AllenCahnSolver
    from lbm.fe_droplet import create_fe_droplet_with_impact
    from geometry.substrate import create_substrate, create_substrate_with_fraction

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

    # ── Substrate ──
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

    # ── Droplet positions ──
    cx = nx / 2.0
    # Surface height at center
    if substrate_type == 'concave':
        ridge_top = int(surface_height.max())
    else:
        ridge_top = surface_height[nx // 2, ny // 2]

    gap = 2.0
    cz = ridge_top + gap + R_drop
    max_cz = nz - R_drop - 2
    if cz > max_cz:
        cz = max_cz

    # y positions for each droplet: centered in the domain
    y_total_span = (n_drops - 1) * p_lattice
    y_start = ny / 2.0 - y_total_span / 2.0

    # Injection schedule and positions
    inject_schedule = []  # list of (step, (cx, cy, cz))
    for layer in range(n_layers):
        y_offset = layer * stagger_offset * p_lattice
        for k in range(n_drops):
            if layer == 0 and k == 0:
                continue  # first droplet initialized directly
            y_k = y_start + k * p_lattice + y_offset
            y_k = max(R_drop + 1, min(ny - R_drop - 1, y_k))
            # z position: higher for second layer
            cz_layer = cz + layer * (2 * R_drop + gap)
            cz_layer = min(cz_layer, max_cz)
            step_k = layer * (n_drops * dt_inject + 1500) + k * dt_inject
            if layer > 0:
                step_k += dt_inject  # extra offset for second layer
            inject_schedule.append((step_k, (cx, y_k, cz_layer)))

    # ── Initialize first droplet ──
    y_first = max(R_drop + 1, min(ny - R_drop - 1, y_start))
    C_init, _, u_init = create_fe_droplet_with_impact(
        nx, ny, nz, center=(cx, y_first, cz), radius=R_drop, xi=xi,
        rho_l=rho_l, rho_g=rho_g, u_impact=(0.0, 0.0, U0))
    solver.init_fields(C_init, u_init)

    # ── Snapshot setup ──
    snap_dir = os.path.join(_project_root, 'results', 'chapter4_snapshots')
    os.makedirs(snap_dir, exist_ok=True)
    snap_prefix = label if label else f"line_{substrate_type}_p{p_ratio:.1f}"
    snap_interval = max(500, N // 12)
    snap_steps = set()
    if save_snapshots:
        for s in range(snap_interval, N + 1, snap_interval):
            snap_steps.add(s)
        # Always save final step
        snap_steps.add(N)

    # ── Monitoring history ──
    history = []
    max_width = 0
    max_width_step = 0
    morphology = 'unknown'
    cv_final = 0.0
    mean_w_final = 0.0
    stable = True
    inject_idx = 0
    t0 = time.time()

    # ── Time stepping ──
    for step in range(1, N + 1):
        # Inject droplets on schedule
        while (inject_idx < len(inject_schedule)
               and step == inject_schedule[inject_idx][0]):
            _, pos = inject_schedule[inject_idx]
            solver.inject_droplet(center=pos, radius=R_drop, xi=xi,
                                  u_impact=(0.0, 0.0, U0))
            inject_idx += 1

        solver.step()

        # ── Monitoring (every 100 steps) ──
        if step % 100 == 0:
            phi_np = solver.phi.detach().cpu().numpy()
            w_profile, h_profile = measure_line_profile(
                phi_np, solid_np, nx, ny, nz)

            active_w = w_profile[w_profile > 1.0]
            if len(active_w) > 0:
                w_mean = float(active_w.mean())
                w_max = float(active_w.max())
                w_std = float(active_w.std())
            else:
                w_mean = w_max = w_std = 0.0

            if w_max > max_width:
                max_width = w_max
                max_width_step = step

            history.append({
                'step': step,
                'w_mean': round(w_mean, 2),
                'w_max': round(w_max, 2),
                'w_std': round(w_std, 2),
                'line_coverage': round(np.sum(w_profile > 1.0) / ny, 4),
                'n_injected': 1 + inject_idx,
            })

            # Stability check
            if np.isnan(phi_np).any() or phi_np.max() < 0.01:
                stable = False
                break

        # ── Save snapshots ──
        if save_snapshots and step in snap_steps:
            phi_np = solver.phi.detach().cpu().numpy()
            # Side view: xz slice at y = ny/2
            mid_j = ny // 2
            xz_slice = phi_np[:, mid_j, :]
            xz_solid = solid_np[:, mid_j, :]
            # Top view: xy slice at z = surface height + 5
            z_top = ridge_top + 5
            if z_top < nz:
                xy_slice = phi_np[:, :, z_top]
                xy_solid = solid_np[:, :, z_top]
            else:
                xy_slice = np.zeros((nx, ny), dtype=np.float32)
                xy_solid = np.ones((nx, ny), dtype=bool)
            snap_path = os.path.join(snap_dir, f'{snap_prefix}_step{step}.npz')
            np.savez_compressed(
                snap_path,
                xz_slice=xz_slice, xz_solid=xz_solid,
                xy_slice=xy_slice, xy_solid=xy_solid,
                step=step, nx=nx, ny=ny, nz=nz,
                p_ratio=p_ratio, n_drops=n_drops,
            )

    # ── Final morphology classification ──
    phi_final = solver.phi.detach().cpu().numpy()
    w_final, h_final = measure_line_profile(phi_final, solid_np, nx, ny, nz)
    morphology, cv_final, mean_w_final = classify_morphology(w_final, p_lattice, D0)

    # ── Cross-section profiles ──
    cross_y_positions = [
        int(y_start),                          # first droplet
        int(y_start + p_lattice),              # second droplet
        int(y_start + p_lattice / 2),          # between 1st and 2nd
        int(y_start + (n_drops - 1) * p_lattice),  # last droplet
    ]
    cross_y_positions = [j for j in cross_y_positions if 0 <= j < ny]
    cross_sections = measure_line_cross_section(
        phi_final, solid_np, nx, ny, nz, cross_y_positions)

    elapsed = time.time() - t0
    if DEVICE == 'cuda':
        mem_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        torch.cuda.reset_peak_memory_stats()
    elif DEVICE == 'xpu':
        try:
            mem_mb = torch.xpu.max_memory_allocated() / 1024 / 1024
            torch.xpu.reset_peak_memory_stats()
        except Exception:
            mem_mb = 0
    else:
        mem_mb = 0

    if label is None:
        label = f"line_{substrate_type}_R{R_star or 'flat'}_p{p_ratio:.1f}"
    tag = "OK" if stable else "FAIL"
    print(f"  {label:>50} p/D0={p_ratio:.1f} We={We:5.1f} theta={theta_eq:3.0f} "
          f"| w_mean={mean_w_final:.1f} cv={cv_final:.3f} {morphology:10s} "
          f"[{tag}] [{elapsed:.0f}s, {mem_mb:.0f}MB] {nx}x{ny}x{nz}")

    del solver
    if DEVICE == 'cuda':
        torch.cuda.empty_cache()
    elif DEVICE == 'xpu':
        try:
            torch.xpu.empty_cache()
        except Exception:
            pass

    return {
        'label': label,
        'substrate_type': substrate_type,
        'R_star': R_star,
        'p_ratio': p_ratio,
        'We': We,
        'theta_eq': theta_eq,
        'amp': amp,
        'n_drops': n_drops,
        'n_layers': n_layers,
        'stagger_offset': stagger_offset,
        'morphology': morphology,
        'cv': round(cv_final, 4),
        'w_mean': round(mean_w_final, 2),
        'w_max': round(max_width, 2),
        'w_std': round(cv_final * mean_w_final, 2),
        'max_width_step': max_width_step,
        'cross_sections': {str(k): v for k, v in cross_sections.items()},
        'w_profile_final': w_final.tolist(),
        'h_profile_final': h_final.tolist(),
        'stable': stable,
        'grid': f"{nx}x{ny}x{nz}",
        'elapsed_s': round(elapsed, 1),
        'mem_mb': round(mem_mb, 0),
        'machine': 'L20_chapter4',
        'history': history,
    }


# ============================================================================
# Main execution
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Chapter 4 line formation simulations')
    parser.add_argument('--skip-existing', action='store_true',
                        help='Skip cases whose labels already exist in results')
    args = parser.parse_args()

    # Load existing results for skip logic
    save_dir = os.path.join(_project_root, 'results')
    os.makedirs(save_dir, exist_ok=True)
    main_path = os.path.join(save_dir, 'chapter4_line_results.json')
    existing_labels = set()
    if os.path.exists(main_path):
        with open(main_path) as f:
            existing_data = json.load(f)
        existing_labels = {r['label'] for r in existing_data}
        print(f"Loaded {len(existing_labels)} existing results from {main_path}")

    results = []

    # =================================================================
    # STUDY 11a: Flat baseline spacing scan (5 cases)
    # Reproduce Soltman 2008 regime map on flat surface
    # p/D0 = 0.3 (dense/bulging), 0.5, 0.8, 1.2, 1.6 (sparse/isolated)
    # =================================================================
    print("\n" + "=" * 70)
    print("  STUDY 11a: Flat baseline spacing scan (We=7.9, theta=162)")
    print("=" * 70)
    for p in [0.3, 0.5, 0.8, 1.2, 1.6]:
        lbl = f"s11a_flat_p{p:.1f}"
        if args.skip_existing and lbl in existing_labels:
            print(f"  SKIP: {lbl}")
            continue
        results.append(run_line_case(
            substrate_type='flat', R_star=None,
            p_ratio=p, We=7.9, theta_eq=162.0, amp=0.0,
            n_drops=5, n_base=150,
            label=lbl,
            save_snapshots=(p in [0.5, 0.8, 1.2]),
        ))

    # =================================================================
    # STUDY 11b: Ridge spacing scan (5 cases)
    # How does cylindrical ridge shift regime boundaries?
    # =================================================================
    print("\n" + "=" * 70)
    print("  STUDY 11b: Ridge R*=1.0 spacing scan (We=7.9, theta=162)")
    print("=" * 70)
    for p in [0.3, 0.5, 0.8, 1.2, 1.6]:
        lbl = f"s11b_ridge_R1.0_p{p:.1f}"
        if args.skip_existing and lbl in existing_labels:
            print(f"  SKIP: {lbl}")
            continue
        results.append(run_line_case(
            substrate_type='ridge', R_star=1.0,
            p_ratio=p, We=7.9, theta_eq=162.0, amp=1.8,
            n_drops=5, n_base=150,
            label=lbl,
            save_snapshots=(p in [0.5, 0.8, 1.2]),
        ))

    # =================================================================
    # STUDY 11c: Convex spacing scan (3 cases)
    # Dome substrate: enhanced spreading in x-direction
    # =================================================================
    print("\n" + "=" * 70)
    print("  STUDY 11c: Convex R*=1.0 spacing scan (We=7.9, theta=162)")
    print("=" * 70)
    for p in [0.5, 0.8, 1.2]:
        lbl = f"s11c_convex_R1.0_p{p:.1f}"
        if args.skip_existing and lbl in existing_labels:
            print(f"  SKIP: {lbl}")
            continue
        results.append(run_line_case(
            substrate_type='convex', R_star=1.0,
            p_ratio=p, We=7.9, theta_eq=162.0, amp=1.8,
            n_drops=5, n_base=150,
            label=lbl,
            save_snapshots=(p == 0.8),
        ))

    # =================================================================
    # STUDY 11d: Concave spacing scan (3 cases)
    # Cavity substrate: confined spreading
    # =================================================================
    print("\n" + "=" * 70)
    print("  STUDY 11d: Concave R*=1.0 spacing scan (We=7.9, theta=162)")
    print("=" * 70)
    for p in [0.5, 0.8, 1.2]:
        lbl = f"s11d_concave_R1.0_p{p:.1f}"
        if args.skip_existing and lbl in existing_labels:
            print(f"  SKIP: {lbl}")
            continue
        results.append(run_line_case(
            substrate_type='concave', R_star=1.0,
            p_ratio=p, We=7.9, theta_eq=162.0, amp=1.8,
            n_drops=5, n_base=150,
            label=lbl,
            save_snapshots=(p == 0.8),
        ))

    # =================================================================
    # STUDY 11e: Weber number effect (5 cases)
    # How does impact inertia affect line morphology on ridge?
    # =================================================================
    print("\n" + "=" * 70)
    print("  STUDY 11e: We effect (ridge R*=1.0, p/D0=0.8, theta=162)")
    print("=" * 70)
    for we in [3.0, 5.0, 10.0, 15.0, 25.0]:
        lbl = f"s11e_ridge_R1.0_p0.8_We{we:.1f}"
        if args.skip_existing and lbl in existing_labels:
            print(f"  SKIP: {lbl}")
            continue
        results.append(run_line_case(
            substrate_type='ridge', R_star=1.0,
            p_ratio=0.8, We=we, theta_eq=162.0, amp=1.8,
            n_drops=5, n_base=150,
            label=lbl,
            save_snapshots=(we in [5.0, 15.0]),
        ))

    # =================================================================
    # STUDY 11f: Contact angle effect (4 cases)
    # Wettability dependence of line morphology on ridge
    # =================================================================
    print("\n" + "=" * 70)
    print("  STUDY 11f: theta effect (ridge R*=1.0, p/D0=0.8, We=7.9)")
    print("=" * 70)
    for theta in [60, 90, 120, 150]:
        lbl = f"s11f_ridge_R1.0_p0.8_theta{theta}"
        if args.skip_existing and lbl in existing_labels:
            print(f"  SKIP: {lbl}")
            continue
        results.append(run_line_case(
            substrate_type='ridge', R_star=1.0,
            p_ratio=0.8, We=7.9, theta_eq=float(theta), amp=1.8,
            n_drops=5, n_base=150,
            label=lbl,
            save_snapshots=(theta in [90, 150]),
        ))

    # =================================================================
    # STUDY 11g: R* curvature effect (4 cases)
    # How does ridge curvature ratio affect line morphology?
    # =================================================================
    print("\n" + "=" * 70)
    print("  STUDY 11g: R* effect (ridge, p/D0=0.8, We=7.9, theta=162)")
    print("=" * 70)
    for rstar in [0.5, 0.7, 1.4, 2.0]:
        lbl = f"s11g_ridge_R{rstar:.1f}_p0.8"
        if args.skip_existing and lbl in existing_labels:
            print(f"  SKIP: {lbl}")
            continue
        results.append(run_line_case(
            substrate_type='ridge', R_star=rstar,
            p_ratio=0.8, We=7.9, theta_eq=162.0, amp=1.8,
            n_drops=5, n_base=150,
            label=lbl,
            save_snapshots=(rstar in [0.7, 2.0]),
        ))

    # =================================================================
    # STUDY 11h: Droplet count effect (3 cases)
    # Does line morphology converge with more droplets?
    # =================================================================
    print("\n" + "=" * 70)
    print("  STUDY 11h: Droplet count (ridge R*=1.0, p/D0=0.8)")
    print("=" * 70)
    for n in [3, 7, 10]:
        lbl = f"s11h_ridge_R1.0_p0.8_n{n}"
        if args.skip_existing and lbl in existing_labels:
            print(f"  SKIP: {lbl}")
            continue
        results.append(run_line_case(
            substrate_type='ridge', R_star=1.0,
            p_ratio=0.8, We=7.9, theta_eq=162.0, amp=1.8,
            n_drops=n, n_base=150,
            label=lbl,
            save_snapshots=(n in [7, 10]),
        ))

    # =================================================================
    # STUDY 11i: Multi-layer stacking (3 cases)
    # Two-layer printing: aligned and staggered deposition
    # =================================================================
    print("\n" + "=" * 70)
    print("  STUDY 11i: Multi-layer stacking (p/D0=0.8, We=7.9, theta=162)")
    print("=" * 70)

    # Flat 2-layer aligned
    lbl = "s11i_flat_2layer_aligned"
    if not (args.skip_existing and lbl in existing_labels):
        results.append(run_line_case(
            substrate_type='flat', R_star=None,
            p_ratio=0.8, We=7.9, theta_eq=162.0, amp=0.0,
            n_drops=5, n_layers=2, stagger_offset=0.0,
            n_base=150,
            label=lbl,
            save_snapshots=True,
        ))
    else:
        print(f"  SKIP: {lbl}")

    # Ridge 2-layer aligned
    lbl = "s11i_ridge_2layer_aligned"
    if not (args.skip_existing and lbl in existing_labels):
        results.append(run_line_case(
            substrate_type='ridge', R_star=1.0,
            p_ratio=0.8, We=7.9, theta_eq=162.0, amp=1.8,
            n_drops=5, n_layers=2, stagger_offset=0.0,
            n_base=150,
            label=lbl,
            save_snapshots=True,
        ))
    else:
        print(f"  SKIP: {lbl}")

    # Ridge 2-layer staggered (offset = 0.5 * p_ratio)
    lbl = "s11i_ridge_2layer_staggered"
    if not (args.skip_existing and lbl in existing_labels):
        results.append(run_line_case(
            substrate_type='ridge', R_star=1.0,
            p_ratio=0.8, We=7.9, theta_eq=162.0, amp=1.8,
            n_drops=5, n_layers=2, stagger_offset=0.5,
            n_base=150,
            label=lbl,
            save_snapshots=True,
        ))
    else:
        print(f"  SKIP: {lbl}")

    # =================================================================
    # Summary and save
    # =================================================================
    print("\n" + "=" * 70)
    print("  Chapter 4 Line Formation — Results Summary")
    print("=" * 70)
    ok = sum(1 for r in results if r['stable'])
    print(f"\n  Total: {len(results)} cases, {ok} OK, {len(results) - ok} FAIL")

    for study in ['s11a', 's11b', 's11c', 's11d', 's11e', 's11f', 's11g', 's11h', 's11i']:
        study_cases = [r for r in results if r['label'].startswith(study)]
        if study_cases:
            print(f"\n  --- {study} ---")
            for r in study_cases:
                tag = "OK" if r['stable'] else "FAIL"
                p = r.get('p_ratio', '-')
                morph = r.get('morphology', '?')
                cv = r.get('cv', 0)
                w = r.get('w_mean', 0)
                print(f"    {r['label']:>50} p/D0={p} w={w:6.1f} "
                      f"cv={cv:.3f} {morph:10s} [{tag}]")

    # ── Save to JSON ──
    if os.path.exists(main_path) and not args.skip_existing:
        with open(main_path) as f:
            all_data = json.load(f)
    elif os.path.exists(main_path):
        with open(main_path) as f:
            all_data = json.load(f)
    else:
        all_data = []

    existing_set = {r['label'] for r in all_data}
    new_count = 0
    for r in results:
        if r['label'] not in existing_set:
            # Save without heavy data in main file
            all_data.append({k: v for k, v in r.items()
                             if k not in ('history', 'w_profile_final', 'h_profile_final')})
            existing_set.add(r['label'])
            new_count += 1

    with open(main_path, 'w') as f:
        json.dump(all_data, f, indent=2, default=str)
    print(f"\n  Saved {new_count} new results to {main_path}")

    # ── Save history and profiles separately ──
    hist_path = os.path.join(save_dir, 'chapter4_line_history.json')
    hist_data = {}
    for r in results:
        entry = {}
        if r.get('history'):
            entry['history'] = r['history']
        if r.get('w_profile_final'):
            entry['w_profile'] = r['w_profile_final']
        if r.get('h_profile_final'):
            entry['h_profile'] = r['h_profile_final']
        if entry:
            hist_data[r['label']] = entry

    if os.path.exists(hist_path):
        with open(hist_path) as f:
            existing_hist = json.load(f)
        existing_hist.update(hist_data)
        hist_data = existing_hist
    with open(hist_path, 'w') as f:
        json.dump(hist_data, f)
    print(f"  History/profiles saved to {hist_path} ({len(hist_data)} entries)")

    elapsed_total = sum(r['elapsed_s'] for r in results)
    print(f"\n  Total runtime: {elapsed_total:.0f}s ({elapsed_total / 3600:.1f}h)")


if __name__ == "__main__":
    main()
