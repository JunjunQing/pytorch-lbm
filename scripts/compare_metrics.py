#!/usr/bin/env python3
"""Metric comparison: compute momentum ratio at time of maximum spreading."""
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

D0 = 45.0; R_drop = D0 / 2.0; rho_l = 1.0; rho_g = 1.0 / 828.0
xi = 4.0; tau = 0.53; U0 = -0.05; N = 150; N_STEPS = 2000

OUTPUT_DIR = os.path.join(_project_root, 'results')
LOG = os.path.join(OUTPUT_DIR, 'metric_comparison.log')
log_f = open(LOG, 'w')

def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line); log_f.write(line + '\n'); log_f.flush()

def get_surface_height(solid, nx, ny, nz):
    height = np.zeros((nx, ny), dtype=int)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz - 1, -1, -1):
                if solid[i, j, k]: height[i, j] = k; break
    return height

def compute_metrics_at_step(solver):
    """Compute metrics at current step."""
    phi = solver.phi.detach().cpu().numpy()
    sol = solver.solid.cpu().numpy()
    rho = solver.rho.detach().cpu().numpy()
    u = solver.u.detach().cpu().numpy()

    intf = (phi > 0.5) & ~sol
    if not intf.any():
        return {'k_spread': 0, 'k_momentum': 0, 'Dx': 0, 'Dy': 0}

    c = np.argwhere(intf)
    Dx = float(c[:, 0].max() - c[:, 0].min() + 1)
    Dy = float(c[:, 1].max() - c[:, 1].min() + 1)
    k_spread = Dx / Dy if Dy > 0 else 0

    # Momentum in interface region
    # Use outward velocity from droplet center to capture spreading asymmetry
    cx_i, cy_i = phi.shape[0] / 2.0, phi.shape[1] / 2.0
    X = np.arange(phi.shape[0])[:, None, None]
    Y = np.arange(phi.shape[1])[None, :, None]
    # Radial distance from center (approximate)
    dx_from_center = X - cx_i
    dy_from_center = Y - cy_i

    # Momentum weighted by distance from center
    mask = intf
    p_x = np.sum(rho[mask] * u[0][mask] * phi[mask] * np.sign(dx_from_center[mask]))
    p_y = np.sum(rho[mask] * u[1][mask] * phi[mask] * np.sign(dy_from_center[mask]))

    k_momentum = abs(p_y) / abs(p_x) if abs(p_x) > 1e-10 else 0

    return {
        'k_spread': k_spread, 'k_momentum': k_momentum,
        'Dx': Dx, 'Dy': Dy, 'p_x': float(p_x), 'p_y': float(p_y),
    }

def run_one(label, We, R_star, theta_eq, amp):
    torch.cuda.empty_cache()
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

    sh = get_surface_height(solid, N, N, nz)
    cx, cy = N / 2.0, N / 2.0
    ridge_top = sh[N // 2, N // 2]
    cz = min(ridge_top + 2.0 + R_drop, nz - R_drop - 2)

    C_init, _, u_init = create_fe_droplet_with_impact(N, N, nz,
        center=(cx, cy, cz), radius=R_drop, xi=xi,
        rho_l=rho_l, rho_g=rho_g, u_impact=(0.0, 0.0, U0))
    solver.init_fields(C_init, u_init)

    t0 = time.time()
    max_k_spread = 0; best_metrics = {}

    for step in range(1, N_STEPS + 1):
        solver.step()
        if step % 100 == 0:
            m = compute_metrics_at_step(solver)
            if m['k_spread'] > max_k_spread:
                max_k_spread = m['k_spread']
                best_metrics = m.copy()
                best_metrics['step'] = step

    elapsed = time.time() - t0
    mass0 = solver.phi_mass_init
    phi_f = solver.phi.detach().cpu().numpy()
    sol_f = solver.solid.cpu().numpy()
    mass_f = float(((phi_f > 0.5) & ~sol_f).sum())
    drift = (mass_f - mass0) / mass0 * 100 if mass0 > 0 else 0

    result = {
        'label': label, 'We': We, 'R_star': R_star, 'theta_eq': theta_eq,
        'amp': amp, 'k_spread': best_metrics.get('k_spread', 0),
        'k_momentum': best_metrics.get('k_momentum', 0),
        'Dx': best_metrics.get('Dx', 0), 'Dy': best_metrics.get('Dy', 0),
        'mass_drift': drift, 'elapsed': elapsed,
    }
    del solver; torch.cuda.empty_cache()
    return result

# Cases
cases = [
    ('Liu_a0',  10.6, 1.2, 160.0, 0.0),
    ('Liu_a15', 10.6, 1.2, 160.0, 1.5),
    ('R1.0_W7.9_a0',  7.9, 1.0, 162.0, 0.0),
    ('R1.0_W7.9_a15', 7.9, 1.0, 162.0, 1.5),
    ('R1.0_W10.6_a0',  10.6, 1.0, 162.0, 0.0),
    ('R1.0_W10.6_a15', 10.6, 1.0, 162.0, 1.5),
]

log("METRIC COMPARISON: Spreading Ratio vs Momentum Ratio")
log(f"Device: {torch.cuda.get_device_name(0)}")

all_results = []
for label, We, R, th, a in cases:
    log(f"  {label}: We={We}, R*={R}, α={a}")
    r = run_one(label, We, R, th, a)
    all_results.append(r)
    log(f"    k_spread={r['k_spread']:.4f}, k_mom={r['k_momentum']:.4f}, "
        f"Dx/Dy={r['Dx']}/{r['Dy']}, {r['elapsed']:.0f}s")

with open(os.path.join(OUTPUT_DIR, 'metric_comparison.json'), 'w') as f:
    json.dump(all_results, f, indent=2)

log(f"\n{'='*70}")
log("SUMMARY")
log(f"{'='*70}")
log(f"{'Case':25s} {'k_spread':>10} {'k_momentum':>12}")
log("-" * 50)
for r in all_results:
    log(f"{r['label']:25s} {r['k_spread']:10.4f} {r['k_momentum']:12.4f}")
log(f"\nLiu et al.: k_momentum = 1.5-2.0")
log("DONE")
log_f.close()
