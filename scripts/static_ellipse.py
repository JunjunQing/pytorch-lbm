#!/usr/bin/env python3
"""Static sessile droplet on the elliptical ridge — mechanism attribution (B3).

Attribution question: the dynamic contact-line diagnostic measured
theta_geo ~ 135 deg (identical across off/const/adaptive at mid-time) vs the
162 deg target. Is 135 deg the physically correct apparent angle on the
convex elliptical ridge, or residual AC-sharpening resistance?

Static experiment (no impact, Bo=1 seating, drop embedded 2 lu, 10000 steps):
  * theta_geo ~ 135 deg -> apparent-curvature effect (physical; dynamic
    result is then correct and adaptive needs no change)
  * theta_geo ~ 162 deg (like the flat-plate 165.8 deg) -> dynamic AC
    resistance residual; adaptive field needs a dynamic factor g(We)

Modes: off (alpha=0) / const (alpha=1.5) / adaptive (sigma=1 field).
Measurement: geometric contour method (reused from contact_line_diag),
flat-plate reference already available: 165.8 deg (alpha=0).

Usage: venv/bin/python scripts/static_ellipse.py [--mode off|const|adaptive]
Saves results/static_ellipse_<mode>.json
"""
import sys, os, json, time, argparse
import numpy as np
import torch

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)

from scripts.adaptive_alpha_study import (ellipse_ridge, surface_curvature,
    alpha_field, get_surface_height)
from scripts.contact_line_diag import geometric_theta

from lbm.fe_config import FEConfig
from lbm.fe_ac_solver import AllenCahnSolver
from lbm.fe_droplet import create_fe_droplet_with_impact

D0, R_drop = 45.0, 22.5
rho_l, rho_g = 1.0, 1.0 / 828.0
xi, tau = 4.0, 0.53
U0_REF = 0.05
WE = 7.9
A, B = 35.0, 54.4
THETA = 162.0
N_BASE = 150
N_STEPS = 10000
EMBED = 2.0
BO = 1.0
SIGMA = 1.0
ALPHA_MAX = 2.0
OUT_EVERY = 1000
LATE_WINDOW = 3000


def run_static(mode):
    torch.cuda.empty_cache()
    nx = ny = N_BASE
    nz = int(B + 2 * R_drop + 20)

    sigma_p = rho_l * U0_REF ** 2 * D0 / WE
    beta = 12.0 * sigma_p / xi
    kappa_p = beta * xi ** 2 / 8.0
    M = 0.02 / beta
    g_val = BO * sigma_p / (rho_l * D0 ** 2)

    solid, fraction = ellipse_ridge(nx, ny, nz, A, B)
    kappa = surface_curvature(fraction, sigma=SIGMA)
    alpha_np = alpha_field(kappa, THETA, ALPHA_MAX)

    if mode == 'off':
        amp_arg, gw = 0.0, False
    elif mode == 'const':
        amp_arg, gw = 1.5, True
    else:
        amp_arg, gw = torch.from_numpy(alpha_np).to('cuda'), True

    cfg = FEConfig(nx=nx, ny=ny, nz=nz, rho_l=rho_l, rho_g=rho_g,
        sigma=sigma_p, xi=xi, beta=beta, kappa=kappa_p, M=M,
        tau_l=tau, tau_g=tau, tau_h=tau + 0.04, theta_eq=THETA,
        device='cuda', max_steps=N_STEPS, output_interval=N_STEPS + 1,
        g_force=(0.0, 0.0, -g_val))
    solver = AllenCahnSolver(cfg, dtype=torch.float32, stab_mode='fakhari',
        boundary_relax=0.0, geometric_wetting=gw, geo_amplification=amp_arg)
    solver.set_solid(torch.from_numpy(solid),
                     solid_fraction=torch.from_numpy(fraction))

    h = get_surface_height(solid)
    apex = int(h[nx // 2, ny // 2])
    cz = min(apex + 1.0 + R_drop - EMBED, nz - R_drop - 2)
    C_init, _, u_init = create_fe_droplet_with_impact(nx, ny, nz,
        center=(nx / 2.0, ny / 2.0, cz), radius=R_drop, xi=xi,
        rho_l=rho_l, rho_g=rho_g, u_impact=(0.0, 0.0, 0.0))
    solver.init_fields(C_init, u_init)

    t0 = time.time()
    history = []
    stable = True
    for step in range(1, N_STEPS + 1):
        solver.step()
        if step % OUT_EVERY == 0:
            phi = solver.phi.detach().cpu().numpy()
            if np.isnan(phi).any():
                stable = False
                break
            gt = geometric_theta(phi, fraction)
            mass = float((phi > 0.5).sum())
            entry = {'step': step, 'mass': round(mass, 1)}
            if gt:
                entry['theta_left'] = gt[0]['left']
                entry['theta_right'] = gt[0]['right']
                entry['n_contact_cols'] = gt[1]
                print(f"[{step:5d}] theta_geo L/R = {gt[0]['left']}/{gt[0]['right']} "
                      f"mass={mass:.0f}", flush=True)
            else:
                print(f"[{step:5d}] no contact-line detected, mass={mass:.0f}",
                      flush=True)
            history.append(entry)

    elapsed = time.time() - t0
    late = [e for e in history if e.get('theta_left') is not None
            and e['step'] > N_STEPS - LATE_WINDOW]
    theta_late = (round(float(np.mean([e['theta_left'] for e in late])), 1)
                  if late else None)
    mass0 = float((C_init > 0.5).sum())
    mass_final = history[-1]['mass'] if history else mass0
    drift = (mass_final - mass0) / mass0 * 100

    result = {
        'mode': mode, 'theta_eq': THETA, 'grid': f'{nx}x{ny}x{nz}',
        'theta_geo_late_avg': theta_late,
        'theta_geo_final': late[-1]['theta_left'] if late else None,
        'mass_drift_pct': round(drift, 2), 'stable': stable,
        'elapsed_s': round(elapsed, 1),
    }
    print(json.dumps(result), flush=True)
    out = os.path.join(_project_root, 'results', f'static_ellipse_{mode}.json')
    with open(out, 'w') as f:
        json.dump({'result': result, 'history': history}, f, indent=1)
    print(f"saved {out}", flush=True)
    del solver


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['off', 'const', 'adaptive'], default='adaptive')
    args = ap.parse_args()
    print(f"Static ellipse {args.mode} | N={N_BASE} | steps={N_STEPS} | "
          f"theta={THETA} | Bo={BO}", flush=True)
    run_static(args.mode)


if __name__ == '__main__':
    main()
