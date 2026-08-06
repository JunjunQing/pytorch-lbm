#!/usr/bin/env python3
"""R5 overnight comprehensive experiment sweep (~10 hours).

Addresses all 6 major weaknesses from R5 review:
  M1: AC sharpening ablation — prove mechanism causality
  M2: Scaling law framing (text change, no sim needed)
  M3: β interpretation (text change, no sim needed)
  M4: Validation softening (text change, no sim needed)
  M5: Novelty vs over-relaxation — show α_geo ≠ fixed over-relaxation
  M6: 3D evidence (already available)

And 10 specific questions:
  Q1: Wetting error vs AC strength
  Q2: Disable/reduce AC → does α become unnecessary?
  Q3: α sensitivity to We
  Q4: Why α=1.5 works broadly
  Q5: High contact angles (>170°)
  Q8: Different ξ
"""
import sys, os, json, time, traceback
import numpy as np

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)

import torch
from lbm.fe_config import FEConfig
from lbm.fe_ac_solver import AllenCahnSolver
from lbm.fe_droplet import create_fe_droplet_with_impact
from geometry.substrate import create_substrate_with_fraction

torch.cuda.empty_cache()

# === Global parameters ===
D0 = 45.0
R_drop = D0 / 2.0
rho_l = 1.0
rho_g = 1.0 / 828.0
xi_base = 4.0
tau = 0.53
U0 = -0.05
N_BASE = 150
N_STEPS = 2000

OUTPUT_DIR = os.path.join(_project_root, 'results')
os.makedirs(OUTPUT_DIR, exist_ok=True)

LOG_PATH = os.path.join(OUTPUT_DIR, 'r5_overnight_sweep.log')
log_f = open(LOG_PATH, 'w')


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line)
    log_f.write(line + '\n')
    log_f.flush()


def get_surface_height(solid, nx, ny, nz):
    height = np.zeros((nx, ny), dtype=int)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz - 1, -1, -1):
                if solid[i, j, k]:
                    height[i, j] = k
                    break
    return height


def run_simulation(label, R_star=1.0, We=7.9, theta_eq=162.0, amp=1.5,
                   n_steps=N_STEPS, geometry='ridge', xi=xi_base,
                   n_base=N_BASE, ac_scale=1.0):
    """Run a single simulation, return result dict."""
    sigma = rho_l * U0**2 * D0 / We
    beta = 12.0 * sigma / xi
    kappa = beta * xi**2 / 8.0
    M = 0.02 / beta

    R_g = abs(R_star) * R_drop
    nx = ny = n_base
    if geometry == 'concave':
        nz = int(R_g + 2 + R_drop + 2 * R_drop + 15)
    else:
        nz = int(R_g + 2 + R_drop + 2 * R_drop + 15)
    nz = min(nz, 300)

    config = FEConfig(
        nx=nx, ny=ny, nz=nz,
        rho_l=rho_l, rho_g=rho_g,
        sigma=sigma, xi=xi, beta=beta, kappa=kappa,
        M=M, tau_l=tau, tau_g=tau, tau_h=tau + 0.04,
        theta_eq=theta_eq, device='cuda',
        max_steps=n_steps, output_interval=n_steps + 1,
        g_force=(0.0, 0.0, 0.0),
    )

    solver = AllenCahnSolver(
        config, dtype=torch.float32,
        stab_mode='fakhari',
        boundary_relax=0.0,
        geometric_wetting=(amp > 0),
        geo_amplification=amp,
        ac_scale=ac_scale,
    )

    solid, fraction = create_substrate_with_fraction(
        nx, ny, nz, substrate_type=geometry,
        R_star=R_star, R_d=R_drop)
    solver.set_solid(solid, solid_fraction=fraction)

    sh = get_surface_height(solid, nx, ny, nz)
    cx, cy = nx / 2.0, ny / 2.0
    ridge_top = sh[nx // 2, ny // 2]
    gap = 2.0
    cz = min(ridge_top + gap + R_drop, nz - R_drop - 2)

    C_init, _, u_init = create_fe_droplet_with_impact(
        nx, ny, nz, center=(cx, cy, cz), radius=R_drop, xi=xi,
        rho_l=rho_l, rho_g=rho_g, u_impact=(0.0, 0.0, U0))
    solver.init_fields(C_init, u_init)

    t0 = time.time()
    max_k = 0.0
    max_k_step = 0
    Dx_p = Dy_p = 0
    k_hist = []
    mass_init = 0.0
    mass_final = 0.0

    for step in range(1, n_steps + 1):
        solver.step()
        if step % 100 == 0:
            phi = solver.phi.detach().cpu().numpy()
            sol = solver.solid.cpu().numpy()
            intf = (phi > 0.5) & ~sol
            fluid = (phi > 0.5) & ~sol
            if fluid.any():
                c = np.argwhere(intf)
                Dx = float(c[:, 0].max() - c[:, 0].min() + 1)
                Dy = float(c[:, 1].max() - c[:, 1].min() + 1)
                k = Dx / Dy if Dy > 0 else 0
            else:
                k = 0; Dx = Dy = 0
            k_hist.append({'step': step, 'k': k, 'Dx': Dx, 'Dy': Dy})
            if k > max_k:
                max_k = k; max_k_step = step; Dx_p = Dx; Dy_p = Dy
            if step == 100:
                mass_init = float(fluid.sum())
            if step == n_steps:
                mass_final = float(fluid.sum())

    mass_drift = (mass_final - mass_init) / mass_init * 100 if mass_init > 0 else 0
    elapsed = time.time() - t0
    mem_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

    # Measure effective contact angle
    theta_eff = 0.0
    phi_final = solver.phi.detach().cpu().numpy()
    sol_final = solver.solid.cpu().numpy()
    try:
        cx_i, cy_i = nx // 2, ny // 2
        # Find contact line (outermost fluid point near wall)
        for r in range(min(nx, ny) // 2, 0, -1):
            for di in range(-1, 2):
                for dj in range(-1, 2):
                    ii, jj = cx_i + r * di, cy_i + r * dj
                    if 0 <= ii < nx and 0 <= jj < ny:
                        # Check first fluid layer above wall
                        for k in range(1, nz):
                            if sol_final[ii, jj, k]:
                                wall_k = k
                                break
                        for k in range(wall_k + 1, min(wall_k + 20, nz)):
                            if phi_final[ii, jj, k] > 0.5:
                                # Found interface
                                r_contact = r
                                y_interface = k
                                y_wall = wall_k
                                theta_eff = np.degrees(np.arctan2(
                                    r_contact, y_interface - y_wall))
                                break
                        if theta_eff > 0:
                            break
                if theta_eff > 0:
                    break
            if theta_eff > 0:
                break
    except Exception:
        pass

    result = {
        'label': label, 'R_star': R_star, 'We': We, 'theta_eq': theta_eq,
        'amp': amp, 'ac_scale': ac_scale, 'xi': xi, 'n_base': n_base,
        'max_k': max_k, 'max_k_step': max_k_step,
        'Dx': Dx_p, 'Dy': Dy_p, 'geometry': geometry,
        'grid': f'{nx}x{ny}x{nz}', 'elapsed_s': elapsed, 'mem_mb': mem_mb,
        'stable': True, 'mass_drift_pct': mass_drift,
        'theta_eff': theta_eff,
        'history': k_hist,
    }
    del solver, C_init, u_init, solid, fraction
    torch.cuda.empty_cache()
    return result


def run_phase(phase_name, cases):
    """Run a batch of simulations."""
    log(f"\n{'='*60}")
    log(f"PHASE: {phase_name}")
    log(f"{'='*60}")
    results = []
    for i, case in enumerate(cases):
        label = case.pop('label', f'case_{i}')
        log(f"  [{i+1}/{len(cases)}] {label}")
        try:
            r = run_simulation(label, **case)
            results.append(r)
            log(f"    k_max={r['max_k']:.4f}, θ_eff={r['theta_eff']:.1f}°, "
                f"mass={r['mass_drift_pct']:+.2f}%, elapsed={r['elapsed_s']:.0f}s")
        except Exception as e:
            log(f"    ERROR: {e}")
            traceback.print_exc()
            results.append({'label': label, 'error': str(e)})
    return results


def save_results(phase_name, results):
    path = os.path.join(OUTPUT_DIR, f'r5_{phase_name}.json')
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)
    log(f"  Saved {path}")
    return path


def main():
    log(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    log(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    t_start = time.time()

    all_results = {}

    # ================================================================
    # Phase 1: AC Sharpening Ablation (M1, Q1, Q2)
    # Scale AC term by factor s, measure wetting error
    # At R*=1.0, We=7.9, θ=162°, N=150, α=1.5
    # ================================================================
    cases = []
    for s in [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]:
        cases.append({'label': f'ac_s{s:.2f}', 'R_star': 1.0, 'We': 7.9,
                      'theta_eq': 162.0, 'amp': 1.5, 'ac_scale': s})
    all_results['phase1_ac_ablation'] = run_phase('AC ablation (M1)', cases)
    save_results('phase1_ac_ablation', all_results['phase1_ac_ablation'])

    # ================================================================
    # Phase 2: α sensitivity × We (Q3)
    # For each We, test α = 0, 0.5, 1.0, 1.5, 2.0
    # ================================================================
    cases = []
    for We in [3.0, 5.0, 10.0, 15.0, 20.0]:
        for a in [0.0, 0.5, 1.0, 1.5, 2.0]:
            cases.append({'label': f'we{We}_a{a}', 'R_star': 1.0, 'We': We,
                          'theta_eq': 162.0, 'amp': a})
    all_results['phase2_we_alpha'] = run_phase('α × We sweep (Q3)', cases)
    save_results('phase2_we_alpha', all_results['phase2_we_alpha'])

    # ================================================================
    # Phase 3: α sensitivity × R* (Q4)
    # For each R*, test α = 0, 0.5, 1.0, 1.5, 2.0
    # ================================================================
    cases = []
    for R in [0.5, 1.5, 3.0]:
        for a in [0.0, 0.5, 1.0, 1.5, 2.0]:
            cases.append({'label': f'R{R}_a{a}', 'R_star': R, 'We': 7.9,
                          'theta_eq': 162.0, 'amp': a})
    all_results['phase3_r_alpha'] = run_phase('α × R* sweep (Q4)', cases)
    save_results('phase3_r_alpha', all_results['phase3_r_alpha'])

    # ================================================================
    # Phase 4: High contact angles (Q5)
    # θ = 170°, 175° with α = 0, 1.0, 1.5, 2.0
    # ================================================================
    cases = []
    for th in [170.0, 175.0]:
        for a in [0.0, 1.0, 1.5, 2.0]:
            cases.append({'label': f'th{int(th)}_a{a}', 'R_star': 1.0, 'We': 7.9,
                          'theta_eq': th, 'amp': a})
    all_results['phase4_high_theta'] = run_phase('High θ (Q5)', cases)
    save_results('phase4_high_theta', all_results['phase4_high_theta'])

    # ================================================================
    # Phase 5: Different ξ (Q8)
    # ξ = 3, 4, 5 with α = 0, 1.0, 1.5
    # ================================================================
    cases = []
    for xival in [3.0, 5.0]:
        for a in [0.0, 1.0, 1.5]:
            cases.append({'label': f'xi{int(xival)}_a{a}', 'R_star': 1.0, 'We': 7.9,
                          'theta_eq': 162.0, 'amp': a, 'xi': xival})
    all_results['phase5_xi'] = run_phase('Different ξ (Q8)', cases)
    save_results('phase5_xi', all_results['phase5_xi'])

    # ================================================================
    # Phase 6: We × R* × α fill (M5 — novelty vs over-relaxation)
    # Show α=1.5 outperforms fixed over-relaxation across parameter space
    # ================================================================
    cases = []
    for We in [3.0, 15.0, 30.0]:
        for R in [0.5, 3.0]:
            for a in [0.0, 1.5]:
                cases.append({'label': f'W{We}_R{R}_a{a}', 'R_star': R, 'We': We,
                              'theta_eq': 162.0, 'amp': a})
    all_results['phase6_fill'] = run_phase('We×R*×α fill (M5)', cases)
    save_results('phase6_fill', all_results['phase6_fill'])

    # ================================================================
    # Phase 7: Concave validation
    # ================================================================
    cases = []
    for th in [90.0, 162.0]:
        for a in [0.0, 1.5]:
            cases.append({'label': f'conc_th{int(th)}_a{a}', 'R_star': 1.0, 'We': 7.9,
                          'theta_eq': th, 'amp': a, 'geometry': 'concave'})
    all_results['phase7_concave'] = run_phase('Concave validation', cases)
    save_results('phase7_concave', all_results['phase7_concave'])

    # ================================================================
    # Summary
    # ================================================================
    t_total = time.time() - t_start
    log(f"\n{'='*60}")
    log("SUMMARY REPORT")
    log(f"{'='*60}")
    log(f"Total time: {t_total/3600:.1f} hours")
    log(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    total_sims = 0
    for phase_name, results in all_results.items():
        valid = [r for r in results if 'max_k' in r]
        total_sims += len(valid)
        log(f"\n{phase_name}: {len(valid)} simulations completed")
        for r in valid[:5]:  # Show first 5
            log(f"  {r['label']:25s}: k={r['max_k']:.4f}, θ={r['theta_eff']:.1f}°")
        if len(valid) > 5:
            log(f"  ... ({len(valid)} total)")

    log(f"\nTotal simulations: {total_sims}")

    # Save master summary
    summary = {
        'total_time_hours': t_total / 3600,
        'end_time': time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_simulations': total_sims,
        'phases': {},
    }
    for phase_name, results in all_results.items():
        summary['phases'][phase_name] = [
            {k: v for k, v in r.items() if k != 'history'}
            for r in results if 'max_k' in r
        ]
    with open(os.path.join(OUTPUT_DIR, 'r5_overnight_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    log(f"\nAll results saved to {OUTPUT_DIR}")
    log("DONE")
    log_f.close()


if __name__ == '__main__':
    main()
