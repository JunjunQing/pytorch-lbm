#!/usr/bin/env python3
"""12-HOUR COMPREHENSIVE VALIDATION SCRIPT
==========================================
5 phases, ~12 hours, ~90 simulations at N=150.

Phase 1: PROSPECTIVE α VALIDATION (1.1h, 9 sims)
  Test α_min formula predictions on unseen θ/R* combos.

Phase 2: STATIC SESSILE DROPLET (3.3h, 8 sims)
  Equilibrium contact angle measurement on curved ridge.

Phase 3: PARAMETER SENSITIVITY (1.5h, 12 sims)
  Interface thickness ξ, density ratio, mobility sensitivity.

Phase 4: LIU PARAMETER-MATCHED (1h, 9 sims)
  Exact We=10.6, R*=1.2, θ=160° validation.

Phase 5: COMPREHENSIVE θ×R* MAP (4h, 32 sims)
  Fill gaps: 4 contact angles × 4 curvature ratios × 2 α values.
"""

import sys, os, json, time, math
import numpy as np

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)

import torch
from lbm.fe_config import FEConfig
from lbm.fe_ac_solver import AllenCahnSolver
from lbm.fe_droplet import create_fe_droplet_with_impact
from geometry.substrate import create_substrate_with_fraction

torch.cuda.empty_cache()

OUTPUT_DIR = os.path.join(_project_root, 'results')
LOG_FILE = os.path.join(OUTPUT_DIR, 'validation_12h.log')

def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line)
    with open(LOG_FILE, 'a') as f:
        f.write(line + '\n')

def get_surface_height(solid, nx, ny, nz):
    height = np.zeros((nx, ny), dtype=int)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz-1, -1, -1):
                if solid[i, j, k]:
                    height[i, j] = k
                    break
    return height

def compute_k(solver, solid_np):
    phi = solver.phi.detach().cpu().numpy()
    intf = (phi > 0.5) & ~solid_np
    if not intf.any():
        return 1.0, 0, 0
    c = np.argwhere(intf)
    Dx = float(c[:, 0].max() - c[:, 0].min() + 1)
    Dy = float(c[:, 1].max() - c[:, 1].min() + 1)
    return Dx / Dy if Dy > 0 else 1.0, Dx, Dy

def measure_contact_angle(phi, solid, nx, ny, nz, sh):
    """Contact angle at contact line on ridge."""
    cx = nx / 2.0
    for ix in range(int(cx), nx):
        wall_h = sh[ix, ny//2]
        if wall_h <= 0:
            continue
        fluid_z = int(wall_h) + 1
        if fluid_z >= nz:
            continue
        val = phi[ix, ny//2, fluid_z]
        val_above = phi[ix, ny//2, min(fluid_z+1, nz-1)]
        if (val > 0.5 and val_above < 0.5) or (val < 0.5 and val_above > 0.5):
            gx = (phi[min(ix+1, nx-1), ny//2, fluid_z] - phi[max(ix-1, 0), ny//2, fluid_z]) / 2.0
            gz = (phi[ix, ny//2, min(fluid_z+1, nz-1)] - phi[ix, ny//2, max(fluid_z-1, 0)]) / 2.0
            return math.degrees(abs(math.atan2(gx, -gz)))
    return None

def compute_alpha_pred(R_star, theta_deg, xi=4.0, D0=45.0, C=0.914):
    R_eff = D0 / (2 * R_star)
    cot = abs(1.0 / math.tan(math.radians(theta_deg)))
    return round(1.0 + C * (xi / R_eff) * cot, 2)

# ===========================================================================
# Shared simulation runner
# ===========================================================================

def run_impact(We, R_star, theta_eq, amp, N=150, N_STEPS=2000,
               xi_val=4.0, tau_val=0.53, rho_ratio=828):
    """Standard droplet impact simulation. Returns dict with results."""
    torch.cuda.empty_cache()
    D0 = 45.0; R_drop = D0 / 2.0
    rho_l = 1.0; rho_g = rho_l / rho_ratio; U0 = -0.05

    sigma = rho_l * U0**2 * D0 / We
    beta = 12.0 * sigma / xi_val
    kappa = beta * xi_val**2 / 8.0
    M = 0.02 / beta

    R_g = abs(R_star) * R_drop
    nz = min(int(R_g + 2 + R_drop + 2*R_drop + 15), 300)

    config = FEConfig(nx=N, ny=N, nz=nz, rho_l=rho_l, rho_g=rho_g,
        sigma=sigma, xi=xi_val, beta=beta, kappa=kappa, M=M,
        tau_l=tau_val, tau_g=tau_val, tau_h=tau_val+0.04, theta_eq=theta_eq,
        device='cuda', max_steps=N_STEPS, output_interval=N_STEPS+1,
        g_force=(0.0, 0.0, 0.0))

    solver = AllenCahnSolver(config, dtype=torch.float32, stab_mode='fakhari',
        boundary_relax=0.0, geometric_wetting=(amp>0), geo_amplification=amp)

    solid, fraction = create_substrate_with_fraction(N, N, nz,
        substrate_type='ridge', R_star=R_star, R_d=R_drop)
    solver.set_solid(solid, solid_fraction=fraction)
    solid_np = solid.cpu().numpy() if torch.is_tensor(solid) else solid

    sh = get_surface_height(solid_np, N, N, nz)
    cx, cy = N/2.0, N/2.0
    ridge_top = sh[N//2, N//2]
    cz = min(ridge_top + 2.0 + R_drop, nz - R_drop - 2)

    C_init, _, u_init = create_fe_droplet_with_impact(N, N, nz,
        center=(cx, cy, cz), radius=R_drop, xi=xi_val,
        rho_l=rho_l, rho_g=rho_g, u_impact=(0.0, 0.0, U0))
    solver.init_fields(C_init, u_init)

    t0 = time.time()
    max_k = 1.0; k_at_peak_step = 0

    for step in range(1, N_STEPS+1):
        solver.step()
        if step % 100 == 0:
            k, _, _ = compute_k(solver, solid_np)
            if k > max_k:
                max_k = k
                k_at_peak_step = step

    elapsed = time.time() - t0
    mass0 = float(solver.phi_mass_init)
    phi_f = solver.phi.detach().cpu().numpy()
    mass_f = float(((phi_f > 0.5) & ~solid_np).sum())
    drift = (mass_f - mass0) / mass0 * 100 if mass0 > 0 else 0

    del solver; torch.cuda.empty_cache()
    return {'k_max': max_k, 'mass_drift_pct': drift, 'elapsed_s': elapsed,
            'step_peak': k_at_peak_step}

def run_static(theta_eq, amp, R_star=1.0, N=150, N_STEPS=8000):
    """Static sessile droplet on ridge. Returns dict with θ_eff."""
    torch.cuda.empty_cache()
    D0 = 45.0; R_drop = D0/2.0
    rho_l = 1.0; rho_g = rho_l/828.0; xi = 4.0; tau = 0.53
    We = 100.0; U0 = -0.005

    sigma = rho_l * U0**2 * D0 / We
    beta = 12.0 * sigma / xi; kappa = beta * xi**2 / 8.0; M = 0.02 / beta
    R_g = abs(R_star) * R_drop
    nz = min(int(R_g + 2 + R_drop + 2*R_drop + 15), 300)

    config = FEConfig(nx=N, ny=N, nz=nz, rho_l=rho_l, rho_g=rho_g,
        sigma=sigma, xi=xi, beta=beta, kappa=kappa, M=M,
        tau_l=tau, tau_g=tau, tau_h=tau+0.04, theta_eq=theta_eq,
        device='cuda', max_steps=N_STEPS, output_interval=N_STEPS+1,
        g_force=(0.0, 0.0, 0.0))

    solver = AllenCahnSolver(config, dtype=torch.float32, stab_mode='fakhari',
        boundary_relax=0.0, geometric_wetting=(amp>0), geo_amplification=amp)

    solid, fraction = create_substrate_with_fraction(N, N, nz,
        substrate_type='ridge', R_star=R_star, R_d=R_drop)
    solver.set_solid(solid, solid_fraction=fraction)
    solid_np = solid.cpu().numpy() if torch.is_tensor(solid) else solid

    sh = get_surface_height(solid_np, N, N, nz)
    cx, cy = N/2.0, N/2.0
    ridge_top = sh[N//2, N//2]
    cz = ridge_top + R_drop + 2

    C_init, _, u_init = create_fe_droplet_with_impact(N, N, nz,
        center=(cx, cy, cz), radius=R_drop, xi=xi,
        rho_l=rho_l, rho_g=rho_g, u_impact=(0.0, 0.0, U0))
    solver.init_fields(C_init, u_init)

    t0 = time.time()
    theta_eff_prev = None; stable = 0

    for step in range(1, N_STEPS+1):
        solver.step()
        if step % 500 == 0 and step >= 2000:
            phi = solver.phi.detach().cpu().numpy()
            te = measure_contact_angle(phi, solid_np, N, N, nz, sh)
            if te is not None and theta_eff_prev is not None:
                stable = stable + 1 if abs(te - theta_eff_prev) < 0.5 else 0
            theta_eff_prev = te
            if stable >= 4:
                break

    elapsed = time.time() - t0
    phi_f = solver.phi.detach().cpu().numpy()
    theta_final = measure_contact_angle(phi_f, solid_np, N, N, nz, sh)
    error = abs(theta_final - theta_eq) if theta_final else None

    mass0 = float(solver.phi_mass_init)
    mass_f = float(((phi_f > 0.5) & ~solid_np).sum())
    drift = (mass_f - mass0) / mass0 * 100 if mass0 > 0 else 0

    del solver; torch.cuda.empty_cache()
    return {'theta_eff': theta_final, 'error_deg': error,
            'mass_drift_pct': drift, 'elapsed_s': elapsed, 'final_step': step}

# ===========================================================================
# Phase definitions
# ===========================================================================

PHASES = {
    'phase1_prospective': {
        'desc': 'PROSPECTIVE α VALIDATION: test formula on unseen θ/R*',
        'cases': [
            # (We, R_star, theta_eq)
            (7.9, 0.7, 150.0),   # unseen θ
            (7.9, 1.5, 150.0),   # unseen θ
            (7.9, 1.0, 130.0),   # unseen θ
            (15.0, 0.7, 130.0),  # unseen θ + We
            (15.0, 1.5, 150.0),  # unseen θ + We
            (5.0, 1.0, 140.0),   # unseen θ + We
        ],
        'alpha_values': [0.0, None, 1.5],  # None = use computed α_pred
        'N_STEPS': 2000,
    },
    'phase2_static': {
        'desc': 'STATIC SESSILE DROPLET: equilibrium contact angle',
        'cases': [
            # (theta_eq, R_star)
            (90.0, 1.0),
            (120.0, 1.0),
            (140.0, 1.0),
            (162.0, 1.0),
            (150.0, 1.0),  # extra validation point
        ],
        'alpha_values': [0.0, 1.5],
        'N_STEPS': 8000,
        'is_static': True,
    },
    'phase3_sensitivity': {
        'desc': 'PARAMETER SENSITIVITY: xi, density ratio, mobility',
        'cases': [
            # (We, R_star, theta_eq, xi, rho_ratio, tau)
            (7.9, 1.0, 162.0, 3.0, 828, 0.53),   # thinner interface
            (7.9, 1.0, 162.0, 5.0, 828, 0.53),   # thicker interface
            (7.9, 1.0, 162.0, 4.0, 100, 0.53),   # lower density ratio
            (7.9, 1.0, 162.0, 4.0, 500, 0.53),   # medium density ratio
            (7.9, 1.0, 162.0, 4.0, 1000, 0.53),  # higher density ratio
            (7.9, 1.0, 162.0, 4.0, 828, 0.70),   # higher relaxation
        ],
        'alpha_values': [0.0, 1.5],
        'N_STEPS': 2000,
    },
    'phase4_liu_matched': {
        'desc': 'LIU PARAMETER-MATCHED: We=10.6, R*=1.2, θ=160°',
        'cases': [
            (10.6, 1.2, 160.0),
            (10.6, 1.0, 162.0),
        ],
        'alpha_values': [0.0, None, 1.5],
        'N_STEPS': 2000,
    },
    'phase5_theta_r_map': {
        'desc': 'COMPREHENSIVE θ×R* MAP: fill phase diagram gaps',
        'cases': [
            # θ × R* at We=7.9 (all combos not in existing data)
            (7.9, 0.5, 120.0),
            (7.9, 0.5, 140.0),
            (7.9, 0.7, 120.0),
            (7.9, 0.7, 140.0),
            (7.9, 1.0, 90.0),
            (7.9, 1.0, 120.0),
            (7.9, 1.0, 140.0),
            (7.9, 1.5, 120.0),
            (7.9, 1.5, 140.0),
            (7.9, 2.0, 120.0),
            (7.9, 2.0, 140.0),
            (7.9, 3.0, 90.0),
            (7.9, 3.0, 120.0),
            (7.9, 3.0, 140.0),
            (15.0, 0.5, 120.0),
            (15.0, 1.0, 120.0),
            (15.0, 1.0, 140.0),
            (15.0, 1.5, 120.0),
            (15.0, 1.5, 140.0),
        ],
        'alpha_values': [1.5],  # only amplified (sufficient data points)
        'N_STEPS': 2000,
    },
}

# ===========================================================================
# Main
# ===========================================================================

def main():
    N = 150
    all_results = []
    total_sims = 0
    start_time = time.time()

    log("=" * 80)
    log("12-HOUR COMPREHENSIVE VALIDATION")
    log(f"Device: {torch.cuda.get_device_name(0)} | Grid: {N}x{N}")
    log("=" * 80)

    for phase_name, phase in PHASES.items():
        log(f"\n{'='*60}")
        log(f"PHASE: {phase_name} — {phase['desc']}")
        log(f"{'='*60}")

        is_static = phase.get('is_static', False)

        for case in phase['cases']:
            if is_static:
                theta_eq, R_star = case
                We = None
            elif len(case) == 3:
                We, R_star, theta_eq = case
            else:
                We, R_star, theta_eq, xi_v, rho_r, tau_v = case

            for amp_raw in phase['alpha_values']:
                if amp_raw is None:
                    if len(case) >= 3:
                        amp = compute_alpha_pred(R_star, theta_eq)
                    else:
                        amp = compute_alpha_pred(R_star, theta_eq)
                    amp_label = 'predicted'
                else:
                    amp = amp_raw
                    amp_label = f'α={amp:.1f}'

                total_sims += 1

                # Run
                if is_static:
                    r = run_static(theta_eq, amp, R_star=R_star, N=N,
                                  N_STEPS=phase['N_STEPS'])
                    r['phase'] = phase_name
                    r['We'] = None; r['R_star'] = R_star; r['theta_eq'] = theta_eq
                    r['amp'] = amp; r['amp_label'] = amp_label
                    te = r['theta_eff']
                    er = r['error_deg']
                    tstr = f"θ_eff={te:.1f}° err={er:.1f}°" if te else "θ_eff=N/A"
                    log(f"  [{total_sims}] {phase_name} ({amp_label}): "
                        f"θ={theta_eq}° R*={R_star} → "
                        f"{tstr} ms={r['mass_drift_pct']:.1f}% t={r['elapsed_s']:.0f}s")
                else:
                    if len(case) >= 6:
                        r = run_impact(We, R_star, theta_eq, amp, N=N,
                                      N_STEPS=phase['N_STEPS'],
                                      xi_val=xi_v, rho_ratio=rho_r, tau_val=tau_v)
                        r['xi'] = xi_v; r['rho_ratio'] = rho_r; r['tau'] = tau_v
                    else:
                        r = run_impact(We, R_star, theta_eq, amp, N=N,
                                      N_STEPS=phase['N_STEPS'])
                    r['phase'] = phase_name
                    r['We'] = We; r['R_star'] = R_star; r['theta_eq'] = theta_eq
                    r['amp'] = amp; r['amp_label'] = amp_label
                    log(f"  [{total_sims}] {phase_name} ({amp_label}): "
                        f"We={We} R*={R_star} θ={theta_eq}° → "
                        f"k={r['k_max']:.4f} ms={r['mass_drift_pct']:.1f}% "
                        f"t={r['elapsed_s']:.0f}s")

                all_results.append(r)

    elapsed_total = time.time() - start_time

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------
    outpath = os.path.join(OUTPUT_DIR, 'validation_12h.json')
    with open(outpath, 'w') as f:
        json.dump(all_results, f, indent=2)

    # -----------------------------------------------------------------------
    # Summary reports
    # -----------------------------------------------------------------------
    log(f"\n{'='*80}")
    log(f"ALL DONE — {total_sims} sims in {elapsed_total/3600:.1f} hours")
    log(f"{'='*80}")

    # Phase 1 summary
    log(f"\n--- PHASE 1: PROSPECTIVE α VALIDATION ---")
    p1 = [r for r in all_results if r['phase'] == 'phase1_prospective']
    for (we, rs, th) in sorted(set((r['We'], r['R_star'], r['theta_eq']) for r in p1)):
        rows = [r for r in p1 if abs(r['We']-we)<0.1 and abs(r['R_star']-rs)<0.01 and abs(r['theta_eq']-th)<0.1]
        k0 = next((r['k_max'] for r in rows if r['amp']==0.0), None)
        kp = next((r['k_max'] for r in rows if r['amp_label']=='predicted'), None)
        k15 = next((r['k_max'] for r in rows if r['amp']==1.5), None)
        ap = compute_alpha_pred(rs, th)
        log(f"  We={we:.0f} R*={rs:.1f} θ={th:.0f}° α_pred={ap:.2f}: "
            f"k0={k0:.3f if k0 else 'N/A'} kp={kp:.3f if kp else 'N/A'} k15={k15:.3f if k15 else 'N/A'}")

    # Phase 2 summary
    log(f"\n--- PHASE 2: STATIC SESSILE DROPLET ---")
    p2 = [r for r in all_results if r['phase'] == 'phase2_static']
    for (th, rs) in sorted(set((r['theta_eq'], r['R_star']) for r in p2)):
        rows = [r for r in p2 if abs(r['theta_eq']-th)<0.1 and abs(r['R_star']-rs)<0.01]
        te0 = next((r.get('theta_eff') for r in rows if r['amp']==0.0), None)
        te15 = next((r.get('theta_eff') for r in rows if r['amp']>0), None)
        err0 = next((r.get('error_deg') for r in rows if r['amp']==0.0), None)
        err15 = next((r.get('error_deg') for r in rows if r['amp']>0), None)
        log(f"  θ_eq={th:.0f}°: α=0→{te0:.1f}°(Δ{err0:.1f}°) "
            f"α=1.5→{te15:.1f}°(Δ{err15:.1f}°)" 
            if te0 and te15 and err0 and err15 else
            f"  θ_eq={th:.0f}°: data incomplete")

    log(f"\nSaved: {outpath}")
    log("DONE")

if __name__ == '__main__':
    main()
