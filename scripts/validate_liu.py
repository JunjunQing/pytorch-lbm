#!/usr/bin/env python3
"""Validate against Liu et al. (2015) — run at their exact parameters."""
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

D0 = 45.0
R_drop = D0 / 2.0
rho_l = 1.0
rho_g = 1.0 / 828.0
xi = 4.0
tau = 0.53
U0 = -0.05
N = 150
N_STEPS = 2000

OUTPUT_DIR = os.path.join(_project_root, 'results')
os.makedirs(OUTPUT_DIR, exist_ok=True)
LOG = os.path.join(OUTPUT_DIR, 'validate_liu.log')
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
                    height[i, j] = k; break
    return height

def run_one(label, We, R_star, theta_eq, amp, n_steps=N_STEPS, nz_extra=0):
    sigma = rho_l * U0**2 * D0 / We
    beta = 12.0 * sigma / xi
    kappa = beta * xi**2 / 8.0
    M = 0.02 / beta
    R_g = abs(R_star) * R_drop
    nx = ny = N
    nz = int(R_g + 2 + R_drop + 2 * R_drop + 15) + nz_extra
    nz = min(nz, 300)

    config = FEConfig(nx=nx, ny=ny, nz=nz, rho_l=rho_l, rho_g=rho_g,
        sigma=sigma, xi=xi, beta=beta, kappa=kappa, M=M,
        tau_l=tau, tau_g=tau, tau_h=tau+0.04, theta_eq=theta_eq,
        device='cuda', max_steps=n_steps, output_interval=n_steps+1,
        g_force=(0.0, 0.0, 0.0))

    solver = AllenCahnSolver(config, dtype=torch.float32, stab_mode='fakhari',
        boundary_relax=0.0, geometric_wetting=(amp > 0), geo_amplification=amp)

    solid, fraction = create_substrate_with_fraction(nx, ny, nz,
        substrate_type='ridge', R_star=R_star, R_d=R_drop)
    solver.set_solid(solid, solid_fraction=fraction)

    sh = get_surface_height(solid, nx, ny, nz)
    cx, cy = nx/2.0, ny/2.0
    ridge_top = sh[nx//2, ny//2]
    cz = min(ridge_top + 2.0 + R_drop, nz - R_drop - 2)

    C_init, _, u_init = create_fe_droplet_with_impact(nx, ny, nz,
        center=(cx, cy, cz), radius=R_drop, xi=xi,
        rho_l=rho_l, rho_g=rho_g, u_impact=(0.0, 0.0, U0))
    solver.init_fields(C_init, u_init)

    t0 = time.time()
    max_k = 0; max_k_step = 0; Dx_p = Dy_p = 0; k_hist = []

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
    mass0 = solver.phi_mass_init
    phi_final = solver.phi.detach().cpu().numpy()
    sol_final = solver.solid.cpu().numpy()
    mass_final = float(((phi_final > 0.5) & ~sol_final).sum())
    mass_drift = (mass_final - mass0) / mass0 * 100 if mass0 > 0 else 0

    result = {
        'label': label, 'R_star': R_star, 'We': We, 'theta_eq': theta_eq,
        'amp': amp, 'max_k': max_k, 'max_k_step': max_k_step,
        'Dx': Dx_p, 'Dy': Dy_p, 'geometry': 'ridge',
        'grid': f'{nx}x{ny}x{nz}', 'elapsed_s': elapsed,
        'mass_drift': mass_drift, 'history': k_hist,
    }
    del solver, C_init, u_init, solid, fraction
    torch.cuda.empty_cache()
    return result

# ============================================================
# Cases
# ============================================================
cases = [
    # Liu's exact parameters
    ('Liu_params_a0',  10.6, 1.2, 160.0, 0.0),
    ('Liu_params_a15', 10.6, 1.2, 160.0, 1.5),
    # Our baseline at Liu's We
    ('We10.6_R1.0_a0',  10.6, 1.0, 162.0, 0.0),
    ('We10.6_R1.0_a15', 10.6, 1.0, 162.0, 1.5),
    # Liu's We at different R*
    ('We10.6_R1.5_a0',  10.6, 1.5, 162.0, 0.0),
    ('We10.6_R1.5_a15', 10.6, 1.5, 162.0, 1.5),
]

log("LIU ET AL. (2015) VALIDATION")
log(f"Device: {torch.cuda.get_device_name(0)}")
log(f"Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")

results = []
for label, We, R, th, a in cases:
    log(f"  {label}: We={We}, R*={R}, θ={th}°, α={a}")
    r = run_one(label, We, R, th, a)
    results.append(r)
    log(f"    k_max={r['max_k']:.4f}, Dx/Dy={r['Dx']:.0f}/{r['Dy']:.0f}, "
        f"mass={r['mass_drift']:+.2f}%, {r['elapsed_s']:.0f}s")

# Save
out = os.path.join(OUTPUT_DIR, 'validate_liu_results.json')
with open(out, 'w') as f:
    json.dump(results, f, indent=2)

log(f"\nResults saved to {out}")
log("SUMMARY:")
log(f"  Liu et al. range: k = 1.5-2.0 (momentum ratio)")
for r in results:
    log(f"  {r['label']:20s}: k={r['max_k']:.4f} (Dx={r['Dx']:.0f}, Dy={r['Dy']:.0f})")
log("DONE")
log_f.close()
