#!/usr/bin/env python3
"""AC-LBM Engineering Simulation Campaign for Master's Thesis.

Systematic parameter sweeps across substrate types, curvature ratios,
Weber numbers, and contact angles to support thesis Chapter 3.

Studies:
  1. Substrate type comparison (flat, ridge, convex, concave)
  2. Curvature ratio R* sweep (fill gaps to match Basilisk R values)
  3. Weber number sweep on different substrates
  4. Contact angle effect
  5. Time-resolved dynamics for key cases

Hardware: NVIDIA CMP 30HX (6GB), ~4min/case at 100^3.
"""
import sys
import os

# Add project root to path (works from any CWD)
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

torch.cuda.empty_cache()

# --- Global physical parameters ---
D0 = 45.0
R_drop = D0 / 2.0
rho_l = 1.0
rho_g = 1.0 / 828.0
xi = 4.0
tau = 0.53
U0 = -0.05


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
             use_vp=True, dtype=torch.float32, label=None,
             save_snapshots=False, snapshot_dir=None):
    """Run a single AC-LBM simulation case.

    Parameters
    ----------
    substrate_type : str
        'flat', 'ridge', 'convex', or 'concave'
    R_star : float or None
        R_substrate / R_droplet. None for flat.
    We : float
        Weber number.
    theta_eq : float
        Equilibrium contact angle in degrees.
    amp : float
        Geometric amplification factor.
    N : int
        Maximum number of time steps.
    n_base : int
        Base grid resolution.
    use_vp : bool
        Whether to use volume penalization.
    label : str or None
        Label for this case in output.
    save_snapshots : bool
        Whether to save xz cross-section snapshots.
    snapshot_dir : str or None
        Directory for snapshots.

    Returns
    -------
    dict with simulation results.
    """
    sigma = rho_l * U0 ** 2 * D0 / We
    beta = 12.0 * sigma / xi
    kappa = beta * xi ** 2 / 8.0
    M = 0.02 / beta

    # Use same tau for both phases (validated config from run_paper_sweep.py)
    # Density-weighted tau interpolation is handled inside the solver
    tau_l = tau
    tau_g = tau

    is_flat = substrate_type == 'flat' or R_star is None

    # Grid sizing: adaptive z based on actual geometry
    if is_flat:
        # Flat: droplet center at ~R_drop+2, rebound ~2*R_drop above
        nz_min = int(R_drop + 2 + 2 * R_drop + 15)  # ~82 for D0=45
        nx, ny, nz = n_base, n_base, nz_min
    else:
        R_g = abs(R_star) * R_drop
        nx, ny = n_base, n_base
        # Substrate top + gap + droplet + rebound headroom
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
    # For concave: use rim height (max) instead of center (cavity floor)
    if substrate_type == 'concave':
        ridge_top = int(surface_height.max())
    else:
        ridge_top = surface_height[nx // 2, ny // 2]
    gap = 2.0
    cz = ridge_top + gap + R_drop
    max_cz = nz - R_drop - 2
    if cz > max_cz:
        cz = max_cz

    # Initialize droplet
    C_init, _, u_init = create_fe_droplet_with_impact(
        nx, ny, nz, center=(cx, cy, cz), radius=R_drop, xi=xi,
        rho_l=rho_l, rho_g=rho_g, u_impact=(0.0, 0.0, U0)
    )
    solver.init_fields(C_init, u_init)

    # Run simulation
    max_k = 0.0
    max_k_step = 0
    Dx_peak = 0.0
    Dy_peak = 0.0
    k_history = []
    stable = True
    t0 = time.time()

    # Snapshot timesteps: 10 moments covering approach→impact→spread→retract
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

                # Measure z_com for contact detection
                z_com = float(coords[:, 2].mean())
            else:
                Dx = Dy = Dz = k_val = 0
                z_com = 0.0

            k_history.append({
                'step': step, 'Dx': Dx, 'Dy': Dy, 'Dz': Dz,
                'k': k_val, 'z_com': z_com
            })

            if k_val > max_k:
                max_k = k_val
                max_k_step = step
                Dx_peak = Dx
                Dy_peak = Dy

            if np.isnan(phi_np).any() or phi_np.max() < 0.01:
                stable = False
                break

        # Save full 3D snapshots for post-processing + 2D slices for quick view
        if save_snapshots and step in snap_steps and snapshot_dir:
            os.makedirs(snapshot_dir, exist_ok=True)
            phi_np = solver.phi.detach().cpu().numpy()
            solid_np = solver.solid.cpu().numpy()
            u_np = solver.u.detach().cpu().numpy()  # (3, nx, ny, nz)

            # 2D slices for quick-look plotting
            mid_j = ny // 2
            mid_i = nx // 2
            xz_slice = phi_np[:, mid_j, :]   # (nx, nz) — side view
            yz_slice = phi_np[mid_i, :, :]   # (ny, nz) — front view
            xz_solid = solid_np[:, mid_j, :]

            # Full 3D data for post-processing
            snap_path = os.path.join(snapshot_dir,
                                     f"{label}_step{step:05d}.npz")
            np.savez_compressed(snap_path,
                                phi_3d=phi_np,            # (nx, ny, nz)
                                u_3d=u_np,                # (3, nx, ny, nz)
                                solid_3d=solid_np,        # (nx, ny, nz)
                                xz_slice=xz_slice, yz_slice=yz_slice,
                                xz_solid=xz_solid,
                                step=step, nx=nx, ny=ny, nz=nz)

            if step % 500 == 0 or step == list(snap_steps)[-1]:
                sz_mb = phi_np.nbytes / 1e6 + u_np.nbytes / 1e6
                print(f"    [snapshot] {label} step={step} "
                      f"saved ({sz_mb:.1f}MB compressed)")

    elapsed = time.time() - t0
    mem_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    torch.cuda.reset_peak_memory_stats()

    if label is None:
        if is_flat:
            label = "flat"
        else:
            label = f"{substrate_type}_R{R_star:.1f}"

    tag = "OK" if stable else "FAIL"
    print(f"  {label:>20} We={We:5.1f} amp={amp:.1f} theta={theta_eq:5.1f} | "
          f"k={max_k:.4f} Dx={Dx_peak:.0f} Dy={Dy_peak:.0f} "
          f"step={max_k_step} [{tag}] "
          f"[{elapsed:.0f}s, {mem_mb:.0f}MB] "
          f"grid={nx}x{ny}x{nz}")

    del solver
    torch.cuda.empty_cache()

    return {
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
        'grid': f"{nx}x{ny}x{nz}",
        'elapsed_s': round(elapsed, 1),
        'mem_mb': round(mem_mb, 0),
        'k_history': k_history,
    }


def main():
    results = []
    snapshot_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'results', 'thesis_snapshots')

    # =================================================================
    # STUDY 0: Grid convergence (4 resolutions, ridge R*=1.0, We=7.9)
    # =================================================================
    print("\n" + "=" * 70)
    print("  STUDY 0: Grid convergence (ridge R*=1.0, We=7.9)")
    print("=" * 70)

    # D0 scales with grid, tau adjusts to keep Re=225
    grid_configs = [
        # (n_base, D0, tau, label)
        (80,  24.0, 0.555, '80'),
        (100, 30.0, 0.540, '100'),
        (120, 36.0, 0.533, '120'),
        (150, 45.0, 0.530, '150'),
    ]
    for n_base, d0, tau_val, tag in grid_configs:
        # Override globals temporarily
        global D0, R_drop, tau
        D0_saved, tau_saved = D0, tau
        D0 = d0
        R_drop = D0 / 2.0
        tau = tau_val
        r = run_case(
            substrate_type='ridge', R_star=1.0,
            We=7.9, theta_eq=162.0, amp=1.8,
            N=3000, n_base=n_base, dtype=torch.float32,
            label=f"s0_grid{tag}",
        )
        r['grid_level'] = n_base
        r['D0'] = d0
        results.append(r)
        D0, R_drop, tau = D0_saved, D0_saved / 2.0, tau_saved

    # =================================================================
    # STUDY 1: Liu 2015 benchmark validation (13 D/D0 points)
    # =================================================================
    print("\n" + "=" * 70)
    print("  STUDY 1: Liu 2015 benchmark (13 D/D0 points)")
    print("=" * 70)

    # R* sweep on ridge, We=7.9, theta=162
    liu_cases = [
        # (R_star, amp)
        (0.5,  2.0),
        (0.6,  1.8),
        (0.7,  1.8),
        (0.8,  1.8),
        (1.0,  1.8),
        (1.2,  1.5),
        (1.4,  1.2),
        (1.5,  1.0),
        (2.0,  0.8),
        (2.5,  0.5),
        (2.76, 0.5),
        (3.5,  0.3),
    ]
    for R_star, amp_val in liu_cases:
        r = run_case(
            substrate_type='ridge', R_star=R_star,
            We=7.9, theta_eq=162.0, amp=amp_val,
            N=3000, n_base=150, dtype=torch.float32,
            label=f"s1_ridge_R{R_star}",
            save_snapshots=(R_star in [1.0, 2.0]),
            snapshot_dir=snapshot_dir,
        )
        results.append(r)

    # Flat reference
    r = run_case(
        substrate_type='flat', R_star=None,
        We=7.9, theta_eq=162.0, amp=0.0,
        N=3000, n_base=150, dtype=torch.float32,
        label="s1_flat",
        save_snapshots=True, snapshot_dir=snapshot_dir,
    )
    results.append(r)

    # =================================================================
    # STUDY 2: Substrate type comparison + morphology snapshots
    # =================================================================
    print("\n" + "=" * 70)
    print("  STUDY 2: Substrate comparison + snapshots (We=7.9)")
    print("=" * 70)

    study2_cases = [
        # (substrate_type, R_star, amp, theta_eq)
        ('flat',    None, 0.0, 162.0),
        ('flat',    None, 0.0, 90.0),    # 2h: θ=90° control
        ('ridge',   1.0, 1.8, 162.0),
        ('ridge',   2.0, 0.8, 162.0),
        ('convex',  1.0, 1.8, 162.0),
        ('convex',  2.0, 1.0, 162.0),
        ('concave', 1.0, 1.8, 162.0),
        ('concave', 2.0, 1.0, 162.0),
    ]
    for sub_type, R_star, amp_val, theta_val in study2_cases:
        theta_tag = f"_theta{theta_val:.0f}" if theta_val != 162.0 else ""
        r = run_case(
            substrate_type=sub_type, R_star=R_star,
            We=7.9, theta_eq=theta_val, amp=amp_val,
            N=3000, n_base=150, dtype=torch.float32,
            label=f"s2_{sub_type}_R{R_star or 'flat'}{theta_tag}",
            save_snapshots=True, snapshot_dir=snapshot_dir,
        )
        results.append(r)

    # =================================================================
    # STUDY 7: Time-resolved dynamics (dense output, every 20 steps)
    # =================================================================
    print("\n" + "=" * 70)
    print("  STUDY 7: Time-resolved dynamics (6 key cases)")
    print("=" * 70)

    study7_cases = [
        ('flat',    None, 7.9,  162.0, 0.0),
        ('ridge',   1.0, 7.9,  162.0, 1.8),
        ('ridge',   1.0, 5.0,  162.0, 1.8),
        ('ridge',   1.0, 15.0, 162.0, 1.8),
        ('convex',  1.0, 7.9,  162.0, 1.8),
        ('concave', 1.0, 7.9,  162.0, 1.8),
    ]
    for sub_type, R_star, we, theta, amp_val in study7_cases:
        r = run_case(
            substrate_type=sub_type, R_star=R_star,
            We=we, theta_eq=theta, amp=amp_val,
            N=3000, n_base=150, dtype=torch.float32,
            label=f"s7_{sub_type}_R{R_star or 'flat'}_We{we}",
            save_snapshots=False, snapshot_dir=None,
        )
        results.append(r)

    # =================================================================
    # STUDY 8: Energy budget analysis (4 cases)
    # TODO: Requires energy calculation module (E_k, E_s, E_diss)
    # Placeholder — uncomment when energy module is implemented.
    # =================================================================
    # print("\n" + "=" * 70)
    # print("  STUDY 8: Energy budget analysis (4 cases)")
    # print("=" * 70)
    # study8_cases = [
    #     ('ridge',  1.0, 7.9,  162.0, 1.8),
    #     ('flat',   None, 7.9,  162.0, 0.0),
    #     ('ridge',  1.0, 15.0, 162.0, 1.8),
    #     ('ridge',  2.0, 7.9,  162.0, 0.8),
    # ]
    # for sub_type, R_star, we, theta, amp_val in study8_cases:
    #     r = run_case(...)
    #     results.append(r)

    # =================================================================
    # Summary
    # =================================================================
    print("\n" + "=" * 70)
    print("  THESIS SWEEP RESULTS SUMMARY")
    print("=" * 70)

    print(f"\n  {'Label':>20} {'Type':>8} {'R*':>5} {'We':>6} "
          f"{'theta':>6} {'amp':>5} {'k_max':>8} {'Dx':>4} {'Dy':>4} "
          f"{'step':>5} {'Status':>6}")
    print("  " + "-" * 85)

    for r in results:
        status = "OK" if r['stable'] else "FAIL"
        R_str = f"{r['R_star']:.1f}" if r['R_star'] is not None else "---"
        print(f"  {r['label']:>20} {r['substrate_type']:>8} {R_str:>5} "
              f"{r['We']:6.1f} {r['theta_eq']:6.1f} {r['amp']:5.1f} "
              f"{r['max_k']:8.4f} {r['Dx']:4.0f} {r['Dy']:4.0f} "
              f"{r['max_k_step']:5d} {status:>6}")

    # Save results (strip k_history to keep JSON manageable)
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             '..', 'results', 'thesis_sweep_results.json')
    save_data = []
    for r in results:
        r_copy = {k: v for k, v in r.items() if k != 'k_history'}
        save_data.append(r_copy)

    with open(save_path, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\n  Results saved to {save_path}")

    # Save k_history separately for time-resolved analysis
    history_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'results', 'thesis_k_history.json')
    history_data = {r['label']: r.get('k_history', []) for r in results}
    with open(history_path, 'w') as f:
        json.dump(history_data, f, indent=2)
    print(f"  Time histories saved to {history_path}")


if __name__ == "__main__":
    main()
