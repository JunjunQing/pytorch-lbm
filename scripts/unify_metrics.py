#!/usr/bin/env python3
"""Unified metric comparison: spreading ratio vs momentum ratio, tracked over time.

Strategy:
  - Track k_spread and k_momentum every 25 steps
  - Report both at peak k_spread moment
  - Also report the time series for cross-validation
"""

import sys, os, json, time
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
LOG = os.path.join(OUTPUT_DIR, 'momentum_unified.log')
log_f = open(LOG, 'w')

def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line); log_f.write(line + '\n'); log_f.flush()

def get_surface_height(solid, nx, ny, nz):
    height = np.zeros((nx, ny), dtype=int)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz - 1, -1, -1):
                if solid[i, j, k]:
                    height[i, j] = k
                    break
    return height

def compute_metrics_at_step(solver, solid_np):
    """Computes both k_spread and k_momentum at current step.
    
    k_spread = Dx / Dy (spreading diameter ratio)
    k_momentum = |py| / |px| (signed momentum ratio, per Liu et al.)
    
    Both computed over interface region (phi > 0.5, not solid).
    """
    phi = solver.phi.detach().cpu().numpy()
    rho = solver.rho.detach().cpu().numpy()
    u = solver.u.detach().cpu().numpy()

    intf = (phi > 0.5) & ~solid_np
    if not intf.any():
        return {'k_spread': 0, 'k_momentum': 0, 'Dx': 0, 'Dy': 0,
                'px': 0, 'py': 0, 'mass': 0}

    c = np.argwhere(intf)
    Dx = float(c[:, 0].max() - c[:, 0].min() + 1)
    Dy = float(c[:, 1].max() - c[:, 1].min() + 1)
    k_spread = Dx / Dy if Dy > 0 else 0

    # Momentum ratio (Liu et al. 2015):
    # p = sum(rho * u * phi) over interface, signed by distance from center
    # By convention, x = along-ridge (spreading direction), y = across-ridge
    cx_i = phi.shape[0] / 2.0
    cy_i = phi.shape[1] / 2.0

    ii, jj, kk = np.where(intf)
    dx = ii - cx_i
    dy = jj - cy_i

    px = np.sum(rho[intf] * u[0][intf] * phi[intf] * np.sign(dx))
    py = np.sum(rho[intf] * u[1][intf] * phi[intf] * np.sign(dy))

    k_momentum = abs(py) / abs(px) if abs(px) > 1e-10 else 0
    mass = float(intf.sum())

    return {'k_spread': k_spread, 'k_momentum': k_momentum,
            'Dx': Dx, 'Dy': Dy, 'px': float(px), 'py': float(py), 'mass': mass}

def run_one(label, We, R_star, theta_eq, amp, N=150, N_STEPS=2000):
    torch.cuda.empty_cache()

    D0 = 45.0
    R_drop = D0 / 2.0
    rho_l = 1.0
    rho_g = 1.0 / 828.0
    xi = 4.0
    tau = 0.53
    U0 = -0.05

    sigma = rho_l * U0**2 * D0 / We
    beta = 12.0 * sigma / xi
    kappa = beta * xi**2 / 8.0
    M = 0.02 / beta
    R_g = abs(R_star) * R_drop
    nz = min(int(R_g + 2 + R_drop + 2 * R_drop + 15), 300)

    config = FEConfig(nx=N, ny=N, nz=nz, rho_l=rho_l, rho_g=rho_g,
        sigma=sigma, xi=xi, beta=beta, kappa=kappa, M=M,
        tau_l=tau, tau_g=tau, tau_h=tau + 0.04, theta_eq=theta_eq,
        device='cuda', max_steps=N_STEPS, output_interval=N_STEPS + 1,
        g_force=(0.0, 0.0, 0.0))

    solver = AllenCahnSolver(config, dtype=torch.float32, stab_mode='fakhari',
        boundary_relax=0.0, geometric_wetting=(amp > 0), geo_amplification=amp)

    solid, fraction = create_substrate_with_fraction(N, N, nz,
        substrate_type='ridge', R_star=R_star, R_d=R_drop)
    solver.set_solid(solid, solid_fraction=fraction)
    solid_np = solid.cpu().numpy() if torch.is_tensor(solid) else solid

    sh = get_surface_height(solid_np, N, N, nz)
    cx, cy = N / 2.0, N / 2.0
    ridge_top = sh[N // 2, N // 2]
    cz = min(ridge_top + 2.0 + R_drop, nz - R_drop - 2)

    C_init, _, u_init = create_fe_droplet_with_impact(N, N, nz,
        center=(cx, cy, cz), radius=R_drop, xi=xi,
        rho_l=rho_l, rho_g=rho_g, u_impact=(0.0, 0.0, U0))
    solver.init_fields(C_init, u_init)

    t0 = time.time()
    ts_history = []           # (step, k_spread, k_momentum, mass)
    max_k_spread = 0
    best_m_at_peak = {}       # momentum metrics at max k_spread
    best_spread_at_mom = {}   # spread metrics at max k_momentum
    max_k_momentum = 0

    TRACK_EVERY = 25

    for step in range(1, N_STEPS + 1):
        solver.step()
        if step % TRACK_EVERY == 0:
            m = compute_metrics_at_step(solver, solid_np)
            ts_history.append({
                'step': step, 'k_spread': m['k_spread'],
                'k_momentum': m['k_momentum'],
                'Dx': m['Dx'], 'Dy': m['Dy'],
                'px': m['px'], 'py': m['py'], 'mass': m['mass'],
            })
            if m['k_spread'] > max_k_spread:
                max_k_spread = m['k_spread']
                best_m_at_peak = m.copy()
                best_m_at_peak['step'] = step
            if m['k_momentum'] > max_k_momentum:
                max_k_momentum = m['k_momentum']
                best_spread_at_mom = m.copy()
                best_spread_at_mom['step'] = step

    elapsed = time.time() - t0

    mass0 = solver.phi_mass_init
    phi_f = solver.phi.detach().cpu().numpy()
    mass_f = float(((phi_f > 0.5) & ~solid_np).sum())
    drift = (mass_f - mass0) / mass0 * 100 if mass0 > 0 else 0

    result = {
        'label': label, 'We': We, 'R_star': R_star, 'theta_eq': theta_eq,
        'amp': amp, 'N': N,
        'k_spread_peak': best_m_at_peak.get('k_spread', 0),
        'k_momentum_at_kspread_peak': best_m_at_peak.get('k_momentum', 0),
        'k_momentum_peak': best_spread_at_mom.get('k_momentum', 0),
        'k_spread_at_kmomentum_peak': best_spread_at_mom.get('k_spread', 0),
        'step_kspread_peak': best_m_at_peak.get('step', 0),
        'step_kmomentum_peak': best_spread_at_mom.get('step', 0),
        'mass_drift': drift,
        'elapsed': elapsed,
        'ts_history': ts_history,  # full time series
    }

    del solver; torch.cuda.empty_cache()
    return result

# ---------------------------------------------------------------------------
# Case definitions
# ---------------------------------------------------------------------------
cases = [
    # Baseline (no amplification)
    ('R1.0_W7.9_a0',     7.9, 1.0, 162.0, 0.0),
    ('R1.0_W10.6_a0',    10.6, 1.0, 162.0, 0.0),
    ('R1.2_W10.6_a0',    10.6, 1.2, 160.0, 0.0),  # Liu matched
    # Amplified
    ('R1.0_W7.9_a15',    7.9, 1.0, 162.0, 1.5),
    ('R1.0_W10.6_a15',   10.6, 1.0, 162.0, 1.5),
    ('R1.2_W10.6_a15',   10.6, 1.2, 160.0, 1.5),  # Liu matched
    # Additional R* values
    ('R0.5_W7.9_a0',     7.9, 0.5, 162.0, 0.0),
    ('R0.5_W7.9_a15',    7.9, 0.5, 162.0, 1.5),
    ('R1.5_W7.9_a0',     7.9, 1.5, 162.0, 0.0),
    ('R1.5_W7.9_a15',    7.9, 1.5, 162.0, 1.5),
    ('R2.0_W7.9_a0',     7.9, 2.0, 162.0, 0.0),
    ('R2.0_W7.9_a15',    7.9, 2.0, 162.0, 1.5),
    # Varying We
    ('R1.0_W3.0_a0',     3.0, 1.0, 162.0, 0.0),
    ('R1.0_W3.0_a15',    3.0, 1.0, 162.0, 1.5),
    ('R1.0_W15.0_a0',    15.0, 1.0, 162.0, 0.0),
    ('R1.0_W15.0_a15',   15.0, 1.0, 162.0, 1.5),
]

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
N = 150
N_STEPS = 2000

log("=" * 70)
log("MOMENTUM UNIFIED METRIC COMPARISON")
log("=" * 70)
log(f"Device: {torch.cuda.get_device_name(0)}")
log(f"Grid: {N}x{N}, Steps: {N_STEPS}")
log(f"Cases: {len(cases)}")
log(f"")

all_results = []
for i, (label, We, R, th, a) in enumerate(cases, 1):
    log(f"[{i}/{len(cases)}] {label}: We={We}, R*={R}, θ={th}°, α={a}")
    r = run_one(label, We, R, th, a, N=N, N_STEPS=N_STEPS)
    all_results.append(r)

    ks = r['k_spread_peak']
    km_s = r['k_momentum_at_kspread_peak']
    km = r['k_momentum_peak']
    ks_m = r['k_spread_at_kmomentum_peak']
    tks = r['step_kspread_peak']
    tkm = r['step_kmomentum_peak']

    log(f"  k_spread(max)={ks:.4f} @ step {tks};  "
        f"k_mom(at that step)={km_s:.4f}")
    log(f"  k_momentum(max)={km:.4f} @ step {tkm};  "
        f"k_spread(at that step)={ks_m:.4f}")
    log(f"  mass_drift={r['mass_drift']:.1f}%,  t={r['elapsed']:.0f}s")
    log(f"")

# ---------------------------------------------------------------------------
# Summary Table
# ---------------------------------------------------------------------------
log("=" * 70)
log("SUMMARY: k_spread vs k_momentum")
log("=" * 70)
hdr = f"{'Case':25s} {'k_spr_peak':>10} {'k_mom@it':>10} {'k_mom_peak':>10} {'k_spr@it':>10} {'Δstep':>8}"
log(hdr)
log("-" * len(hdr))
for r in all_results:
    dstep = abs(r['step_kspread_peak'] - r['step_kmomentum_peak'])
    log(f"{r['label']:25s} {r['k_spread_peak']:10.4f} {r['k_momentum_at_kspread_peak']:10.4f} "
        f"{r['k_momentum_peak']:10.4f} {r['k_spread_at_kmomentum_peak']:10.4f} {dstep:8}")

# Save results
outpath = os.path.join(OUTPUT_DIR, 'momentum_unified.json')
with open(outpath, 'w') as f:
    json.dump(all_results, f, indent=2)

log(f"\nResults saved to: {outpath}")
log("DONE")
log_f.close()
