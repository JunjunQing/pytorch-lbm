#!/usr/bin/env python3
"""Comprehensive validation study for the geo_amplification paper.

Part 1: R* sweep at N=150, α=1.5, We=7.9 (complete the parameter space)
Part 2: Contact time analysis from existing time histories
Part 3: Mass conservation verification
"""
import sys, os, json, time
import numpy as np

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)

import torch
from lbm.fe_config import FEConfig
from lbm.fe_ac_solver import AllenCahnSolver
from lbm.fe_droplet import create_fe_droplet_with_impact
from geometry.substrate import create_substrate, create_substrate_with_fraction

torch.cuda.empty_cache()

# === Global parameters ===
D0 = 45.0
R_drop = D0 / 2.0
rho_l = 1.0
rho_g = 1.0 / 828.0
xi = 4.0
tau = 0.53
U0 = -0.05
N_BASE = 150
WE = 7.9
THETA = 162.0
ALPHA = 1.5
N_STEPS = 2000

OUTPUT_DIR = os.path.join(_project_root, 'results')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_surface_height(solid, nx, ny, nz):
    height = np.zeros((nx, ny), dtype=int)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz - 1, -1, -1):
                if solid[i, j, k]:
                    height[i, j] = k
                    break
    return height


# ================================================================
# Part 1: R* sweep at N=150
# ================================================================
def run_R_star_sweep():
    print("="*60)
    print("PART 1: R* sweep at N=150, α=1.5, We=7.9")
    print("="*60)

    R_stars = [0.5, 0.7, 1.5, 2.0, 3.0]  # R*=1.0 already done
    results = []

    for R_star in R_stars:
        sigma = rho_l * U0**2 * D0 / WE
        beta = 12.0 * sigma / xi
        kappa = beta * xi**2 / 8.0
        M = 0.02 / beta

        R_g = abs(R_star) * R_drop
        nx = ny = N_BASE
        nz = int(R_g + 2 + R_drop + 2 * R_drop + 15)
        nz = min(nz, 300)

        print(f"\n--- R*={R_star}, grid={nx}x{ny}x{nz} ---")

        config = FEConfig(
            nx=nx, ny=ny, nz=nz,
            rho_l=rho_l, rho_g=rho_g,
            sigma=sigma, xi=xi, beta=beta, kappa=kappa,
            M=M, tau_l=tau, tau_g=tau, tau_h=tau + 0.04,
            theta_eq=THETA, device='cuda',
            max_steps=N_STEPS, output_interval=N_STEPS + 1,
            g_force=(0.0, 0.0, 0.0),
        )

        solver = AllenCahnSolver(
            config, dtype=torch.float32,
            stab_mode='fakhari',
            boundary_relax=0.0,
            geometric_wetting=(ALPHA > 0),
            geo_amplification=ALPHA,
        )

        solid, fraction = create_substrate_with_fraction(
            nx, ny, nz, substrate_type='ridge',
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

        t0 = time.time()
        max_k = 0.0
        max_k_step = 0
        Dx_peak = Dy_peak = 0.0
        k_history = []
        volume_history = []
        z_min_history = []

        for step in range(1, N_STEPS + 1):
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
                    z_min = float(coords[:, 2].min())
                    vol = float(interface.sum())
                else:
                    Dx = Dy = Dz = k_val = 0
                    z_min = 0.0
                    vol = 0.0

                k_history.append({'step': step, 'k': k_val, 'Dx': Dx, 'Dy': Dy, 'Dz': Dz})
                volume_history.append({'step': step, 'vol': vol})
                z_min_history.append({'step': step, 'z_min': z_min})

                if k_val > max_k:
                    max_k = k_val
                    max_k_step = step
                    Dx_peak = Dx
                    Dy_peak = Dy

                if step % 500 == 0:
                    print(f"  step={step}: k={k_val:.4f}, Dx={Dx:.0f}, Dy={Dy:.0f}, vol={vol:.0f}")

        elapsed = time.time() - t0
        mem_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

        # Contact time estimate: when z_min starts increasing after first minimum
        z_vals = [z['z_min'] for z in z_min_history]
        steps_vals = [z['step'] for z in z_min_history]
        # Find first local minimum of z_min after impact
        contact_step = 0
        if len(z_vals) > 10:
            # Find the step where z_min is minimum (rebound start)
            min_idx = np.argmin(z_vals)
            contact_step = steps_vals[min_idx]

        result = {
            'label': f'R_star_{R_star}',
            'substrate_type': 'ridge',
            'R_star': R_star,
            'We': WE,
            'theta_eq': THETA,
            'amp': ALPHA,
            'max_k': max_k,
            'max_k_step': max_k_step,
            'Dx': Dx_peak, 'Dy': Dy_peak,
            'stable': True,
            'grid': f'{nx}x{ny}x{nz}',
            'elapsed_s': elapsed,
            'mem_mb': mem_mb,
            'machine': 'local_gtx1080',
            'contact_step': contact_step,
            'history': k_history,
            'volume_history': volume_history,
            'z_min_history': z_min_history,
        }
        results.append(result)
        print(f"  → k_max={max_k:.4f}, contact_step={contact_step}, elapsed={elapsed:.1f}s")

    # Save
    out_path = os.path.join(OUTPUT_DIR, 'validation_R_star_sweep.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")
    return results


# ================================================================
# Part 2: Contact time analysis from existing data
# ================================================================
def analyze_contact_time():
    print("\n" + "="*60)
    print("PART 2: Contact time analysis")
    print("="*60)

    results = {}

    # Load R* sweep results (just computed)
    rs_path = os.path.join(OUTPUT_DIR, 'validation_R_star_sweep.json')
    if os.path.exists(rs_path):
        with open(rs_path) as f:
            rs_data = json.load(f)
        for item in rs_data:
            R_star = item['R_star']
            contact_step = item['contact_step']
            # Physical time: t = step × (dx / c_s) but in LU, 1 step = 1 LU/ts
            # Contact time in LU: contact_step
            # Convert to physical: t_phys = contact_step × (dx / c_s) = contact_step × (D0/45) × (1/c_s)
            # For comparison, use dimensionless: t* = t × U0 / D0
            t_star = contact_step * abs(U0) / D0
            results[R_star] = {
                'contact_steps': contact_step,
                't_star': t_star,
                'max_k': item['max_k'],
            }
            print(f"  R*={R_star}: contact_steps={contact_step}, t*={t_star:.4f}, k_max={item['max_k']:.4f}")

    # Also load existing R*=1.0 data
    r1_path = os.path.join(OUTPUT_DIR, 'phase4_ridge_n150_comparison.json')
    if os.path.exists(r1_path):
        with open(r1_path) as f:
            r1_data = json.load(f)
        for item in r1_data:
            if abs(item['amp'] - ALPHA) < 0.01:
                h = item.get('history', [])
                if h:
                    steps = [p['step'] for p in h]
                    # Contact time from z_min
                    z_mins = []
                    for p in h:
                        if 'z_min' in p:
                            z_mins.append(p['z_min'])
                    if z_mins:
                        min_idx = np.argmin(z_mins)
                        contact_step = steps[min_idx]
                    else:
                        contact_step = 0
                    t_star = contact_step * abs(U0) / D0
                    results[1.0] = {
                        'contact_steps': contact_step,
                        't_star': t_star,
                        'max_k': item['max_k'],
                    }
                    print(f"  R*=1.0 (from existing): contact_steps={contact_step}, t*={t_star:.4f}, k_max={item['max_k']:.4f}")

    # Save contact time results
    out_path = os.path.join(OUTPUT_DIR, 'validation_contact_time.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")
    return results


# ================================================================
# Part 3: Mass conservation check
# ================================================================
def check_mass_conservation():
    print("\n" + "="*60)
    print("PART 3: Mass conservation check")
    print("="*60)

    # Check from R* sweep data
    rs_path = os.path.join(OUTPUT_DIR, 'validation_R_star_sweep.json')
    if not os.path.exists(rs_path):
        print("  No R* sweep data found, skipping.")
        return

    with open(rs_path) as f:
        rs_data = json.load(f)

    results = []
    for item in rs_data:
        R_star = item['R_star']
        vol_hist = item.get('volume_history', [])
        if len(vol_hist) < 2:
            continue

        v0 = vol_hist[0]['vol']
        v_final = vol_hist[-1]['vol']
        v_max = max(v['vol'] for v in vol_hist)
        v_min = min(v['vol'] for v in vol_hist)

        if v0 > 0:
            drift = (v_final - v0) / v0 * 100
            fluctuation = (v_max - v_min) / v0 * 100
        else:
            drift = fluctuation = 0

        results.append({
            'R_star': R_star,
            'vol_initial': v0,
            'vol_final': v_final,
            'drift_pct': drift,
            'fluctuation_pct': fluctuation,
        })
        print(f"  R*={R_star}: vol_init={v0:.0f}, vol_final={v_final:.0f}, "
              f"drift={drift:+.2f}%, fluctuation={fluctuation:.2f}%")

    out_path = os.path.join(OUTPUT_DIR, 'validation_mass_conservation.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


# ================================================================
# Main
# ================================================================
if __name__ == '__main__':
    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    # Part 1: R* sweep
    run_R_star_sweep()

    # Part 2: Contact time
    analyze_contact_time()

    # Part 3: Mass conservation
    check_mass_conservation()

    print("\n" + "="*60)
    print("VALIDATION COMPLETE")
    print("="*60)
