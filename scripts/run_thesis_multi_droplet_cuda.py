#!/usr/bin/env python3
"""Study 10: Multi-droplet sequential impact on curved surfaces.

Machine: qingqing (Intel Arc A750 8GB) — runs after Studies 4a/4b/6.

Physical motivation: Actual conformal printing deposits droplets sequentially
along a path. Each new droplet interacts with already-deposited liquid,
affecting line width, thickness uniformity, and print quality.

Uses solver.inject_droplet() to add new droplets at specified timesteps.

Studies:
  10a: Substrate comparison (2 drops, dt*=1.0) — 4 cases
  10b: Time interval sweep on ridge (2 drops) — 3 cases
  10c: Droplet count sweep on ridge (2/3 drops) — 2 cases
  10d: We comparison on ridge — 2 cases

Total: 11 cases × ~15min/case ≈ 2.8h

Hardware: Intel Arc A750 8GB (CUDA backend)
Usage: bash scripts/run_thesis_multi_droplet.sh
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

# --- Physical parameters ---
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


def run_sequential_case(substrate_type='ridge', R_star=None, We=7.9,
                        theta_eq=162.0, amp=1.8,
                        n_drops=2, dt_star=1.0,
                        N=6000, n_base=150, dtype=torch.float32, label=None):
    """Run sequential multi-droplet impact simulation.

    First droplet impacts at t=0, subsequent droplets injected at intervals.
    All droplets impact at the same (x,y) location on the substrate.

    Parameters
    ----------
    n_drops : int
        Number of droplets to deposit (1-5).
    dt_star : float
        Dimensionless interval between drops: dt* = dt * |U0| / D0.
        dt_star=1.0 means interval = D0/|U0| = 900 steps.
    """
    sigma = rho_l * U0 ** 2 * D0 / We
    beta = 12.0 * sigma / xi
    kappa = beta * xi ** 2 / 8.0
    M = 0.02 / beta

    tau_l = tau
    tau_g = tau

    is_flat = substrate_type == 'flat' or R_star is None

    if is_flat:
        nz_min = int(R_drop + 2 + (1 + n_drops) * R_drop + 20)
        nx, ny, nz = n_base, n_base, min(nz_min, 300)
    else:
        R_g = abs(R_star) * R_drop
        nx, ny = n_base, n_base
        nz = int(R_g + 2 + R_drop + (1 + n_drops) * R_drop + 20)
        nz = min(nz, 300)

    # Injection interval in lattice timesteps
    dt_inject = int(dt_star * D0 / abs(U0))
    # Schedule: first drop at t=0 (init), subsequent at dt_inject, 2*dt_inject, ...
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
        config, dtype=dtype,
        stab_mode='fakhari',
        boundary_relax=0.0,
        geometric_wetting=(amp > 0),
        geo_amplification=amp,
    )

    # Create substrate
    if is_flat:
        solid = create_substrate(nx, ny, nz, substrate_type='flat',
                                 R_star=None, R_d=R_drop)
        solver.set_solid(solid)
    else:
        solid, fraction = create_substrate_with_fraction(
            nx, ny, nz, substrate_type=substrate_type,
            R_star=R_star, R_d=R_drop)
        solver.set_solid(solid, solid_fraction=fraction)

    # Determine droplet placement
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

    # Initialize first droplet
    C_init, _, u_init = create_fe_droplet_with_impact(
        nx, ny, nz, center=(cx, cy, cz), radius=R_drop, xi=xi,
        rho_l=rho_l, rho_g=rho_g, u_impact=(0.0, 0.0, U0)
    )
    solver.init_fields(C_init, u_init)

    # Run simulation with sequential injection
    max_Dx = 0.0
    max_Dy = 0.0
    max_Dx_step = 0
    k_history = []
    stable = True
    inject_idx = 0
    t0 = time.time()

    for step in range(1, N + 1):
        # Inject subsequent droplets at scheduled times
        if inject_idx < len(inject_steps) and step == inject_steps[inject_idx]:
            solver.inject_droplet(
                center=(cx, cy, cz), radius=R_drop, xi=xi,
                u_impact=(0.0, 0.0, U0),
            )
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
                Dz = float(coords[:, 2].max() - coords[:, 2].min() + 1)
                k_val = Dx / Dy if Dy > 0 else 0
                z_com = float(coords[:, 2].mean())
                phi_mass = float(phi_np[~solid_np].sum())
            else:
                Dx = Dy = Dz = k_val = 0
                z_com = 0.0
                phi_mass = 0.0

            k_history.append({
                'step': step, 'Dx': Dx, 'Dy': Dy, 'Dz': Dz,
                'k': k_val, 'z_com': z_com, 'phi_mass': phi_mass,
                'n_drops_deposited': 1 + inject_idx,
            })
            if Dx > max_Dx:
                max_Dx = Dx
                max_Dx_step = step
                max_Dy = Dy
            if np.isnan(phi_np).any() or phi_np.max() < 0.01:
                stable = False
                break

    elapsed = time.time() - t0
    mem_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    torch.cuda.reset_peak_memory_stats()

    if label is None:
        if is_flat:
            label = f"seq_flat_n{n_drops}_dt{dt_star:.1f}"
        else:
            label = f"seq_{substrate_type}_R{R_star:.1f}_n{n_drops}_dt{dt_star:.1f}"

    tag = "OK" if stable else "FAIL"
    print(f"  {label:>40} We={We:5.1f} n={n_drops} dt*={dt_star:.1f} | "
          f"Dx_max={max_Dx:.0f} Dy={max_Dy:.0f} "
          f"step={max_Dx_step} [{tag}] "
          f"[{elapsed:.0f}s, {mem_mb:.0f}MB] "
          f"grid={nx}x{ny}x{nz}")

    del solver
    torch.cuda.empty_cache()

    return {
        'label': label, 'substrate_type': substrate_type,
        'R_star': R_star, 'We': We, 'theta_eq': theta_eq, 'amp': amp,
        'n_drops': n_drops, 'dt_star': dt_star,
        'max_Dx': max_Dx, 'max_Dy': max_Dy, 'max_Dx_step': max_Dx_step,
        'stable': stable, 'grid': f"{nx}x{ny}x{nz}",
        'elapsed_s': round(elapsed, 1), 'mem_mb': round(mem_mb, 0),
        'machine': 'local_30hx_cuda', 'k_history': k_history,
    }


def main():
    results = []

    # =================================================================
    # STUDY 10a: Substrate comparison (2 drops, dt*=1.0, We=7.9)
    # =================================================================
    print("\n" + "=" * 70)
    print("  STUDY 10a: Substrate comparison (2 drops, dt*=1.0)")
    print("=" * 70)

    for sub_type, R_star, amp_val in [
        ('flat', None, 0.0),
        ('ridge', 1.0, 1.8),
        ('convex', 1.0, 1.8),
        ('concave', 1.0, 1.8),
    ]:
        r = run_sequential_case(
            substrate_type=sub_type, R_star=R_star,
            We=7.9, theta_eq=162.0, amp=amp_val,
            n_drops=2, dt_star=1.0,
            N=6000, n_base=150,
            label=f"s10a_seq_{sub_type}_R{R_star or 'flat'}",
        )
        results.append(r)

    # =================================================================
    # STUDY 10b: Time interval sweep on ridge R*=1.0 (2 drops)
    # =================================================================
    print("\n" + "=" * 70)
    print("  STUDY 10b: Interval sweep on ridge R*=1.0 (2 drops)")
    print("=" * 70)

    for dt in [0.5, 1.0, 2.0]:
        r = run_sequential_case(
            substrate_type='ridge', R_star=1.0,
            We=7.9, theta_eq=162.0, amp=1.8,
            n_drops=2, dt_star=dt,
            N=6000, n_base=150,
            label=f"s10b_ridge_R1.0_dt{dt:.1f}",
        )
        results.append(r)

    # =================================================================
    # STUDY 10c: Droplet count on ridge R*=1.0 (dt*=1.0)
    # =================================================================
    print("\n" + "=" * 70)
    print("  STUDY 10c: Droplet count on ridge R*=1.0")
    print("=" * 70)

    for n in [3, 5]:
        r = run_sequential_case(
            substrate_type='ridge', R_star=1.0,
            We=7.9, theta_eq=162.0, amp=1.8,
            n_drops=n, dt_star=1.0,
            N=n * 3000, n_base=150,
            label=f"s10c_ridge_R1.0_n{n}",
        )
        results.append(r)

    # =================================================================
    # STUDY 10d: We comparison on ridge (2 drops, dt*=1.0)
    # =================================================================
    print("\n" + "=" * 70)
    print("  STUDY 10d: We comparison on ridge (2 drops)")
    print("=" * 70)

    for we in [5.0, 15.0]:
        r = run_sequential_case(
            substrate_type='ridge', R_star=1.0,
            We=we, theta_eq=162.0, amp=1.8,
            n_drops=2, dt_star=1.0,
            N=6000, n_base=150,
            label=f"s10d_ridge_R1.0_We{we}",
        )
        results.append(r)

    # =================================================================
    # Summary
    # =================================================================
    print("\n" + "=" * 70)
    print("  STUDY 10: MULTI-DROPLET SEQUENTIAL RESULTS SUMMARY")
    print("=" * 70)

    print(f"\n  {'Label':>40} {'Type':>8} {'We':>6} "
          f"{'n':>3} {'dt*':>5} {'Dx_max':>7} {'Dy':>4} "
          f"{'Status':>6}")
    print("  " + "-" * 90)

    for r in results:
        status = "OK" if r['stable'] else "FAIL"
        print(f"  {r['label']:>40} {r['substrate_type']:>8} "
              f"{r['We']:6.1f} {r['n_drops']:3d} {r['dt_star']:5.1f} "
              f"{r['max_Dx']:7.0f} {r['max_Dy']:4.0f} {status:>6}")

    # Save results
    save_dir = os.path.join(_project_root, 'results')
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, 'thesis_multi_droplet_results.json')
    save_data = []
    for r in results:
        r_copy = {k: v for k, v in r.items() if k != 'k_history'}
        save_data.append(r_copy)
    with open(save_path, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\n  Results saved to {save_path}")

    history_path = os.path.join(save_dir, 'thesis_multi_droplet_history.json')
    history_data = {r['label']: r.get('k_history', []) for r in results}
    with open(history_path, 'w') as f:
        json.dump(history_data, f, indent=2)
    print(f"  Time histories saved to {history_path}")


if __name__ == "__main__":
    main()
