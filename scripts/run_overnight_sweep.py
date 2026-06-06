#!/usr/bin/env python3
"""Overnight comprehensive parameter sweep (~10 hours).

Phase 1: We sweep at N=150, α=1.5 (6 cases, ~45 min)
Phase 2: R* sweep with α=0 baseline (6 cases, ~45 min)
Phase 3: θ sweep at R*=1.0 (4 cases, ~30 min)
Phase 4: We sweep at R*=0.5 and R*=3.0 (16 cases, ~120 min)
Phase 5: Concave geometry (5 cases, ~40 min)
Phase 6: Sensitivity analysis (5 cases, ~40 min)
Phase 7: Figure generation + paper compilation (~10 min)
Phase 8: Summary report

Total estimated: ~500 min ≈ 8.3 hours
"""
import sys, os, json, time, traceback
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
N_STEPS = 2000

OUTPUT_DIR = os.path.join(_project_root, 'results')
FIG_DIR = os.path.join(_project_root, 'paper', 'figures')
PAPER_DIR = os.path.join(_project_root, 'paper')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Log file
LOG_PATH = os.path.join(OUTPUT_DIR, 'overnight_sweep.log')
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
                   n_steps=N_STEPS, geometry='ridge'):
    """Run a single simulation, return result dict."""
    sigma = rho_l * U0**2 * D0 / We
    beta = 12.0 * sigma / xi
    kappa = beta * xi**2 / 8.0
    M = 0.02 / beta

    R_g = abs(R_star) * R_drop
    nx = ny = N_BASE
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

    for step in range(1, n_steps + 1):
        solver.step()
        if step % 100 == 0:
            phi = solver.phi.detach().cpu().numpy()
            sol = solver.solid.cpu().numpy()
            intf = (phi > 0.5) & ~sol
            if intf.any():
                c = np.argwhere(intf)
                Dx = float(c[:, 0].max() - c[:, 0].min() + 1)
                Dy = float(c[:, 1].max() - c[:, 1].min() + 1)
                k = Dx / Dy if Dy > 0 else 0
            else:
                k = 0; Dx = Dy = 0
            k_hist.append({'step': step, 'k': k, 'Dx': Dx, 'Dy': Dy})
            if k > max_k:
                max_k = k; max_k_step = step; Dx_p = Dx; Dy_p = Dy

    elapsed = time.time() - t0
    mem_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

    result = {
        'label': label, 'R_star': R_star, 'We': We, 'theta_eq': theta_eq,
        'amp': amp, 'max_k': max_k, 'max_k_step': max_k_step,
        'Dx': Dx_p, 'Dy': Dy_p, 'geometry': geometry,
        'grid': f'{nx}x{ny}x{nz}', 'elapsed_s': elapsed, 'mem_mb': mem_mb,
        'stable': True, 'history': k_hist,
    }
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
            log(f"    k_max={r['max_k']:.4f}, elapsed={r['elapsed_s']:.0f}s")
        except Exception as e:
            log(f"    ERROR: {e}")
            traceback.print_exc()
            results.append({'label': label, 'error': str(e)})
    return results


def save_results(phase_name, results):
    path = os.path.join(OUTPUT_DIR, f'overnight_{phase_name}.json')
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
    # Phase 1: We sweep at N=150, α=1.5, R*=1.0
    # ================================================================
    cases = []
    for We in [3.0, 5.0, 7.9, 10.0, 15.0, 20.0]:
        cases.append({'label': f'fig1_we{We}', 'R_star': 1.0, 'We': We,
                      'theta_eq': 162.0, 'amp': 1.5})
    all_results['phase1_we_sweep'] = run_phase('We sweep N=150', cases)
    save_results('phase1_we_sweep', all_results['phase1_we_sweep'])

    # ================================================================
    # Phase 2: R* sweep with α=0 baseline
    # ================================================================
    cases = []
    for R_star in [0.5, 0.7, 1.0, 1.5, 2.0, 3.0]:
        cases.append({'label': f'R{R_star}_amp0', 'R_star': R_star, 'We': 7.9,
                      'theta_eq': 162.0, 'amp': 0.0})
    all_results['phase2_r_amp0'] = run_phase('R* sweep α=0', cases)
    save_results('phase2_r_amp0', all_results['phase2_r_amp0'])

    # ================================================================
    # Phase 3: θ sweep at R*=1.0
    # ================================================================
    cases = []
    for theta in [90.0, 120.0, 140.0]:
        cases.append({'label': f'theta{int(theta)}', 'R_star': 1.0, 'We': 7.9,
                      'theta_eq': theta, 'amp': 1.5})
    all_results['phase3_theta'] = run_phase('θ sweep R*=1.0', cases)
    save_results('phase3_theta', all_results['phase3_theta'])

    # ================================================================
    # Phase 4: We sweep at R*=0.5 and R*=3.0 (α=1.5)
    # ================================================================
    cases = []
    for R_star in [0.5, 3.0]:
        for We in [3.0, 7.9, 15.0, 30.0]:
            cases.append({'label': f'R{R_star}_we{We}', 'R_star': R_star, 'We': We,
                          'theta_eq': 162.0, 'amp': 1.5})
    all_results['phase4_we_Rextreme'] = run_phase('We sweep at extreme R*', cases)
    save_results('phase4_we_Rextreme', all_results['phase4_we_Rextreme'])

    # ================================================================
    # Phase 5: Concave geometry
    # ================================================================
    cases = []
    for We in [7.9, 15.0]:
        cases.append({'label': f'concave_we{We}', 'R_star': 1.0, 'We': We,
                      'theta_eq': 162.0, 'amp': 1.5, 'geometry': 'concave'})
    all_results['phase5_concave'] = run_phase('Concave geometry', cases)
    save_results('phase5_concave', all_results['phase5_concave'])

    # ================================================================
    # Phase 6: Sensitivity analysis (α=0.5, 1.0, 2.0 at R*=1.0)
    # ================================================================
    cases = []
    for amp in [0.5, 1.0, 2.0]:
        cases.append({'label': f'alpha{amp}', 'R_star': 1.0, 'We': 7.9,
                      'theta_eq': 162.0, 'amp': amp})
    all_results['phase6_sensitivity'] = run_phase('α sensitivity', cases)
    save_results('phase6_sensitivity', all_results['phase6_sensitivity'])

    # ================================================================
    # Phase 7: Generate figures + compile paper
    # ================================================================
    log(f"\n{'='*60}")
    log("PHASE 7: Figure generation + paper compilation")
    log(f"{'='*60}")

    import subprocess
    try:
        # Regenerate figures
        log("  Running paper_figures.py...")
        r = subprocess.run(['python3', 'scripts/paper_figures.py'],
                          cwd=_project_root, capture_output=True, text=True, timeout=120)
        log(f"  paper_figures.py: {'OK' if r.returncode == 0 else 'ERROR'}")
        if r.returncode != 0:
            log(f"  stderr: {r.stderr[-500:]}")

        log("  Running phase23_final_figure.py...")
        r = subprocess.run(['python3', 'scripts/phase23_final_figure.py'],
                          cwd=_project_root, capture_output=True, text=True, timeout=120)
        log(f"  phase23_final_figure.py: {'OK' if r.returncode == 0 else 'ERROR'}")

        # Compile paper
        log("  Compiling paper...")
        for _ in range(3):
            r = subprocess.run(['pdflatex', '-interaction=nonstopmode', 'main.tex'],
                              cwd=PAPER_DIR, capture_output=True, text=True, timeout=120)
        log(f"  Paper compilation: {'OK' if r.returncode == 0 else 'ERROR'}")

    except Exception as e:
        log(f"  Figure/compile error: {e}")

    # ================================================================
    # Phase 8: Summary report
    # ================================================================
    t_total = time.time() - t_start
    log(f"\n{'='*60}")
    log("SUMMARY REPORT")
    log(f"{'='*60}")
    log(f"Total time: {t_total/3600:.1f} hours")
    log(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    for phase_name, results in all_results.items():
        valid = [r for r in results if 'max_k' in r]
        log(f"\n{phase_name}: {len(valid)} simulations completed")
        for r in valid:
            log(f"  {r['label']:25s}: k={r['max_k']:.4f}  (We={r['We']}, θ={r['theta_eq']}, α={r['amp']}, R*={r['R_star']})")

    # Save master summary
    summary = {
        'total_time_hours': t_total / 3600,
        'end_time': time.strftime('%Y-%m-%d %H:%M:%S'),
        'phases': {},
    }
    for phase_name, results in all_results.items():
        summary['phases'][phase_name] = [
            {k: v for k, v in r.items() if k != 'history'}
            for r in results if 'max_k' in r
        ]
    with open(os.path.join(OUTPUT_DIR, 'overnight_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    log(f"\nAll results saved to {OUTPUT_DIR}")
    log("DONE")
    log_f.close()


if __name__ == '__main__':
    main()
