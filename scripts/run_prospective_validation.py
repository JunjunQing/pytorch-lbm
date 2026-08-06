#!/usr/bin/env python3
"""Prospective validation: test α_min scaling law on parameters NOT in calibration set.

Hypothesis: α_min = 1 + C * (ξ/R_eff) * cot(θ_eq), where C is calibrated from
R*=1.0 data. This formula should predict the needed α for unseen θ, R* combos.

Test: for each case, run with α=0 (baseline), α=α_pred (formula), α=1.5 (default).
If formula works: k(α_pred) ≥ k(α=0) and k(α_pred) approximately correct.
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
LOG_FILE = os.path.join(OUTPUT_DIR, 'prospective_validation.log')

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
        return 1.0
    c = np.argwhere(intf)
    Dx = float(c[:, 0].max() - c[:, 0].min() + 1)
    Dy = float(c[:, 1].max() - c[:, 1].min() + 1)
    return Dx / Dy if Dy > 0 else 1.0

def run_case(We, R_star, theta_eq, amp, N=150, N_STEPS=2000):
    torch.cuda.empty_cache()
    D0 = 45.0; R_drop = D0/2.0
    rho_l = 1.0; rho_g = 1.0/828.0
    xi = 4.0; tau = 0.53; U0 = -0.05

    sigma = rho_l * U0**2 * D0 / We
    beta = 12.0 * sigma / xi
    kappa = beta * xi**2 / 8.0
    M = 0.02 / beta
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
    solid_np = solid.cpu().numpy()

    sh = get_surface_height(solid_np, N, N, nz)
    cx, cy = N/2.0, N/2.0
    ridge_top = sh[N//2, N//2]
    cz = min(ridge_top + 2.0 + R_drop, nz - R_drop - 2)

    C_init, _, u_init = create_fe_droplet_with_impact(N, N, nz,
        center=(cx, cy, cz), radius=R_drop, xi=xi,
        rho_l=rho_l, rho_g=rho_g, u_impact=(0.0, 0.0, U0))
    solver.init_fields(C_init, u_init)

    t0 = time.time()
    max_k = 1.0

    for step in range(1, N_STEPS+1):
        solver.step()
        if step % 100 == 0:
            k = compute_k(solver, solid_np)
            if k > max_k:
                max_k = k

    elapsed = time.time() - t0
    mass0 = float(solver.phi_mass_init)
    phi_f = solver.phi.detach().cpu().numpy()
    mass_f = float(((phi_f > 0.5) & ~solid_np).sum())
    drift = (mass_f - mass0) / mass0 * 100 if mass0 > 0 else 0

    del solver; torch.cuda.empty_cache()
    return {'k_max': max_k, 'mass_drift_pct': drift, 'elapsed_s': elapsed}

# ---------------------------------------------------------------------------
def compute_alpha_pred(R_star, theta_deg, xi=4.0, D0=45.0, C=0.914):
    """Predict α from scaling law: α = 1 + C*(ξ/R_eff)*cot(θ)"""
    R_eff = D0 / (2 * R_star)  # ridge effective radius
    cot_theta = abs(1.0 / math.tan(math.radians(theta_deg)))
    return round(1.0 + C * (xi / R_eff) * cot_theta, 2)

# Prospective validation cases (NOT in calibration/training set)
cases = [
    # (label, We, R_star, theta_eq)
    ('PV_R07_T150', 7.9, 0.7, 150.0),
    ('PV_R15_T150', 7.9, 1.5, 150.0),
    ('PV_R10_T130', 7.9, 1.0, 130.0),
]

N = 150; N_STEPS = 2000

log("=" * 70)
log("PROSPECTIVE VALIDATION: α_min SCALING LAW")
log("=" * 70)
log(f"Device: {torch.cuda.get_device_name(0)}")
log(f"Grid: {N}x{N}, Steps: {N_STEPS}, Total cases: {len(cases)*3}")
log("")

all_results = []

for label, We, R_star, theta_eq in cases:
    alpha_pred = compute_alpha_pred(R_star, theta_eq)
    log(f"[{label}] We={We}, R*={R_star}, θ={theta_eq}° → α_pred={alpha_pred}")

    for amp, amp_label in [(0.0, 'baseline'), (alpha_pred, 'predicted'), (1.5, 'default')]:
        r = run_case(We, R_star, theta_eq, amp, N=N, N_STEPS=N_STEPS)
        r['label'] = label
        r['We'] = We; r['R_star'] = R_star; r['theta_eq'] = theta_eq
        r['amp'] = amp; r['amp_label'] = amp_label
        r['alpha_pred'] = alpha_pred
        all_results.append(r)
        log(f"  α={amp:.2f} ({amp_label:>9s}): k={r['k_max']:.4f}, "
            f"mass={r['mass_drift_pct']:.1f}%, t={r['elapsed_s']:.0f}s")

    log("")

# ---------------------------------------------------------------------------
log("=" * 70)
log("SUMMARY")
log("=" * 70)
log(f"{'Case':20s} {'α_pred':>6}  {'k(α=0)':>8}  {'k(α_pred)':>8}  {'k(α=1.5)':>8}  {'Δk_pred':>8}")
log("-" * 70)
for label in [c[0] for c in cases]:
    rows = [r for r in all_results if r['label'] == label]
    k0 = [r['k_max'] for r in rows if r['amp'] == 0.0][0]
    kp = [r['k_max'] for r in rows if r['amp_label'] == 'predicted'][0]
    k15 = [r['k_max'] for r in rows if r['amp'] == 1.5][0]
    ap = rows[0]['alpha_pred']
    log(f"{label:20s} {ap:6.2f}  {k0:8.4f}  {kp:8.4f}  {k15:8.4f}  {ap:8.2f}")

outpath = os.path.join(OUTPUT_DIR, 'prospective_validation.json')
with open(outpath, 'w') as f:
    json.dump(all_results, f, indent=2)
log(f"\nSaved: {outpath}")
log("DONE")
