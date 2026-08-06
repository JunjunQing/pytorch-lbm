#!/usr/bin/env python3
"""Production-resolution static contact angle measurement (reviewer suggestion).

Paper currently states direct contact-angle measurement is not reported.
Final review (Round 5.3) recommended: add ONE direct static contact-angle
measurement at theta_eq=162 deg to support the mechanism claim.

Approach (from run_static_validation.py, which produced one clean result:
N=64, amp=0, BO=1, 8000 steps -> theta_eff=162.3 deg, err 0.3 deg):
  * flat substrate, sessile drop embedded 2 lu into the wall (anchors the
    contact line), gentle gravity Bo=1 seats the cap
  * spherical-cap fit: theta from conserved volume V, contact radius r_c,
    cap height h (Newton solve), measured on the phi=0.5 isosurface
  * production resolution N=150 (same as the paper's simulations)

Pair: amp=0.0 (wetting BC off -> expected theta_eff ~ 162) vs amp=1.5
(geometric wetting on -> over-amplified correction, expected theta_eff < 162).

Usage: venv/bin/python3 scripts/run_static_contact_production.py
Saves results/static_contact_production.json
"""
import sys
import os
import json
import time
import math

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)

import numpy as np
import torch

from lbm.fe_config import FEConfig
from lbm.fe_ac_solver import AllenCahnSolver
from lbm.fe_droplet import create_fe_droplet_with_impact
from geometry.substrate import create_substrate_with_fraction

D0 = 45.0
R_drop = D0 / 2.0
RHO_L = 1.0
RHO_G = 1.0 / 828.0
XI = 4.0
TAU = 0.53
U0_REF = 0.05
WE = 7.9
N = 150              # production resolution
N_STEPS = 10000
EMBED = 2.0          # initial droplet embed depth into the wall (anchoring)
BO = 1.0             # seating gravity (working value from bo1_a0)
OUT_EVERY = 1000
LATE_WINDOW = 3000   # average theta over the last LATE_WINDOW steps


def spherical_cap_angle(volume, r_c, h_max):
    """Contact angle from spherical-cap geometry (axisymmetric, no gravity)."""
    if r_c <= 1.0 or volume <= 0:
        return None
    target = 6.0 * volume / math.pi
    h = h_max if h_max > 0 else r_c
    for _ in range(60):
        f = h ** 3 + 3.0 * r_c ** 2 * h - target
        fp = 3.0 * h ** 2 + 3.0 * r_c ** 2
        h_new = h - f / fp
        if abs(h_new - h) < 1e-8:
            h = h_new
            break
        h = h_new
    if h <= 0:
        return None
    denom = r_c ** 2 - h ** 2
    if abs(denom) < 1e-8:
        return 90.0
    return math.degrees(math.atan2(2.0 * h * r_c, denom))


def measure_flat_contact(phi, solid_np):
    """Contact radius, height, volume on flat substrate; theta via cap fit."""
    mask = phi > 0.5
    vol = float(mask.sum())
    if vol == 0:
        return None
    coords = np.argwhere(mask)
    z_min = int(coords[:, 2].min())
    z_max = int(coords[:, 2].max())
    h_max = z_max - z_min + 1.0
    layer = mask[:, :, z_min]
    if not layer.any():
        return None
    ys, xs = np.argwhere(layer).T
    cy, cx = np.mean(ys), np.mean(xs)
    rs = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    r_c = float(rs.max()) + 0.5
    theta = spherical_cap_angle(vol, r_c, h_max)
    return {'volume': vol, 'r_c': r_c, 'h_max': h_max,
            'theta': theta, 'z_min': z_min, 'z_max': z_max}


def run_case(amp, tag):
    torch.cuda.empty_cache()
    sigma = RHO_L * U0_REF ** 2 * D0 / WE
    beta = 12.0 * sigma / XI
    kappa = beta * XI ** 2 / 8.0
    M = 0.02 / beta
    g_val = BO * sigma / (RHO_L * D0 ** 2)

    nx = ny = N
    nz = int(1 + 2.5 * R_drop + EMBED)
    print(f"[{tag}] theta=162 amp={amp} grid={nx}x{ny}x{nz} "
          f"Bo={BO} N_STEPS={N_STEPS}", flush=True)

    config = FEConfig(nx=nx, ny=ny, nz=nz, rho_l=RHO_L, rho_g=RHO_G,
        sigma=sigma, xi=XI, beta=beta, kappa=kappa, M=M,
        tau_l=TAU, tau_g=TAU, tau_h=TAU + 0.04, theta_eq=162.0,
        device='cuda', max_steps=N_STEPS, output_interval=N_STEPS + 1,
        g_force=(0.0, 0.0, -g_val))

    solver = AllenCahnSolver(config, dtype=torch.float32, stab_mode='fakhari',
        boundary_relax=0.0, geometric_wetting=(amp > 0), geo_amplification=amp)

    solid, fraction = create_substrate_with_fraction(nx, ny, nz,
        substrate_type='flat', R_star=1.0, R_d=R_drop)
    solver.set_solid(solid, solid_fraction=fraction)
    solid_np = np.asarray(solid)

    cz = 1.0 + R_drop - EMBED
    C_init, _, u_init = create_fe_droplet_with_impact(nx, ny, nz,
        center=(nx / 2.0, ny / 2.0, cz), radius=R_drop, xi=XI,
        rho_l=RHO_L, rho_g=RHO_G, u_impact=(0.0, 0.0, 0.0))
    solver.init_fields(C_init, u_init)

    t0 = time.time()
    history = []
    for step in range(1, N_STEPS + 1):
        solver.step()
        if step % OUT_EVERY == 0:
            phi = solver.phi.detach().cpu().numpy()
            m = measure_flat_contact(phi, solid_np)
            if m and m['theta'] is not None:
                history.append({'step': step, 'theta': round(m['theta'], 2),
                                'r_c': round(m['r_c'], 2),
                                'h_max': round(m['h_max'], 2),
                                'volume': round(m['volume'], 1)})
                print(f"  [{step:5d}] theta_eff={m['theta']:.1f}deg "
                      f"r_c={m['r_c']:.1f} h={m['h_max']:.1f} "
                      f"V={m['volume']:.0f}", flush=True)
            else:
                print(f"  [{step:5d}] no contact (r_c unavailable)", flush=True)

    elapsed = time.time() - t0
    late = [h['theta'] for h in history if h['step'] > N_STEPS - LATE_WINDOW]
    theta_late = float(np.mean(late)) if late else None
    theta_last = history[-1]['theta'] if history else None
    mass0 = None
    # final mass drift (volume vs initial liquid volume)
    phi = solver.phi.detach().cpu().numpy()
    vol_final = float((phi > 0.5).sum())
    # initial liquid volume: droplet sphere volume (approx)
    vol_init = 4.0 / 3.0 * math.pi * R_drop ** 3

    result = {
        'tag': tag, 'theta_eq': 162.0, 'amp': amp, 'N': N, 'Bo': BO,
        'theta_late_avg': round(theta_late, 2) if theta_late else None,
        'theta_final': round(theta_last, 2) if theta_last else None,
        'error_deg': (round(theta_late - 162.0, 2)
                      if theta_late is not None else None),
        'mass_drift_pct': round((vol_final - vol_init) / vol_init * 100, 2),
        'elapsed_s': round(elapsed, 1),
        'history': history,
    }
    print(f"[{tag}] theta_late={theta_late:.2f} err="
          f"{result['error_deg']}deg mass_drift={result['mass_drift_pct']}% "
          f"({elapsed:.0f}s)", flush=True)
    del solver
    return result


def main():
    cases = [
        (0.0, 'prod_n150_amp0'),
        (1.5, 'prod_n150_amp15'),
    ]
    results = []
    for amp, tag in cases:
        results.append(run_case(amp, tag))

    print("\n" + "=" * 60)
    print("STATIC CONTACT ANGLE (production N=150, theta_eq=162)")
    print("=" * 60)
    for r in results:
        print(f"  amp={r['amp']}: theta_eff={r['theta_late_avg']}deg "
              f"(err {r['error_deg']}deg) | mass drift {r['mass_drift_pct']}%")

    save_path = os.path.join(_project_root, 'results',
                             'static_contact_production.json')
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {save_path}")


if __name__ == '__main__':
    main()
