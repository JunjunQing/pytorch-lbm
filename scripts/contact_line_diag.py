#!/usr/bin/env python3
"""Contact-line diagnostic for the adaptive alpha field.

At the contact line (near-wall interface band: phi>0.5 adjacent to the
solid-fraction transition), measure per-node:
  * kappa      — local surface curvature (from the precomputed field)
  * alpha_act  — the alpha value actually applied there (alpha field)
  * alpha_th   — theory: 1 + xi*kappa*|cot(theta_eq)|
  * theta_eff  — actual contact angle from the interface vs surface normal

Verifies the adaptive-alpha hypothesis directly: alpha_act should track
alpha_th along the contact line, and theta_eff should stay near theta_eq
where alpha > 1 is applied.

Usage: venv/bin/python scripts/contact_line_diag.py [--mode adaptive|const|off]
Saves results/contact_line_diag_<mode>.json and a diagnostic figure.
"""
import sys, os, json, time, argparse
import numpy as np
import torch

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)

from scripts.adaptive_alpha_study import (ellipse_ridge, surface_curvature,
    alpha_field, get_surface_height)

from lbm.fe_config import FEConfig
from lbm.fe_ac_solver import AllenCahnSolver
from lbm.fe_droplet import create_fe_droplet_with_impact

D0, R_drop = 45.0, 22.5
rho_l, rho_g = 1.0, 1.0 / 828.0
xi, tau, U0 = 4.0, 0.53, -0.05
A, B = 35.0, 54.4
THETA = 162.0
N_BASE, STEPS = 150, 2000
SIGMA = 1.0          # best curvature estimator (10.8% apex error)
ALPHA_MAX = 2.0
DIAG_EVERY = 200


def geometric_theta(phi_np, fraction, y_slice=None):
    """Contact angle from 2D contour tangents (robust to diffuse interface).

    On the y=y_slice plane: surface profile z_s(x) = first fraction=0.5
    crossing from below; interface profile z_i(x) = first phi=0.5 crossing
    above the surface. Contact-line columns are where z_i ~ z_s. The angle
    between the interface tangent and the surface tangent is the contact
    angle (liquid side, measured through the liquid).

    Returns (theta_left, theta_right, n_cols) or None.
    """
    nz = phi_np.shape[2]
    y = phi_np.shape[1] // 2 if y_slice is None else y_slice
    nx = phi_np.shape[0]
    frac = fraction[:, y, :]
    phi = phi_np[:, y, :]

    z_s = np.full(nx, np.nan)
    z_i = np.full(nx, np.nan)
    for x in range(nx):
        col = frac[x]
        # surface: first z where fraction drops below 0.5 (from the top of
        # the solid side we scan upward: solid fraction=1 at z=0)
        above = np.argwhere(col > 0.5)
        if above.size == 0:
            continue
        z_top_solid = above[-1][0]
        # crossing between z_top_solid and z_top_solid+1
        z_s[x] = z_top_solid + 0.5
        # interface: first phi crossing 0.5 above the surface
        pcol = phi[x]
        for zz in range(z_top_solid + 1, nz):
            if pcol[zz] >= 0.5:
                z_i[x] = zz - 0.5
                break

    ok = ~np.isnan(z_s) & ~np.isnan(z_i)
    xs = np.arange(nx)[ok]
    zs = z_s[ok]
    zi = z_i[ok]
    if len(xs) < 6:
        return None
    # contact-line columns: interface close to surface
    contact = np.abs(zi - zs) < 2.5 * xi
    if contact.sum() < 4:
        return None
    xc = xs[contact]
    zic = zi[contact]
    zsc = zs[contact]
    # tangent slopes via local linear fit over +/-2 columns
    def slope(xv, zv, xq):
        m = np.argmin(np.abs(xv - xq))
        lo, hi = max(0, m - 2), min(len(xv), m + 3)
        return np.polyfit(xv[lo:hi], zv[lo:hi], 1)[0]
    # left contact line: min x; right: max x
    results = {}
    for side, xq in [('left', xc.min()), ('right', xc.max())]:
        m_if = slope(xc, zic, xq)
        m_sf = slope(xc, zsc, xq)
        ang_if = np.degrees(np.arctan(m_if))
        ang_sf = np.degrees(np.arctan(m_sf))
        # contact angle through the liquid
        theta = 180.0 - abs(ang_if - ang_sf)
        results[side] = round(float(theta), 1)
    return results, int(contact.sum())


def contact_band_stats(phi_np, fraction, kappa, alpha_np, theta_eq):
    """Stats over the near-wall interface band (the contact line region).

    Band: phi>0.5 (liquid) AND within the solid-fraction transition
    (0.02 < fraction < 0.98) AND gradient of phi significant.
    """
    band = (phi_np > 0.5) & (fraction > 0.02) & (fraction < 0.98)
    if not band.any():
        return None
    gy, gx, gz = np.gradient(phi_np, axis=(0, 1, 2))
    gmag = np.sqrt(gx ** 2 + gy ** 2 + gz ** 2)
    band &= gmag > 0.02  # interface nodes only
    if not band.any():
        return None

    idx = np.argwhere(band)
    k_vals = np.abs(kappa[band])
    a_vals = alpha_np[band]
    cot = abs(1.0 / np.tan(np.radians(theta_eq)))
    a_th = np.clip(1.0 + xi * k_vals * cot, 1.0, ALPHA_MAX)

    # surface normal from fraction gradient (points into solid)
    fy, fx, fz = np.gradient(fraction, axis=(0, 1, 2))
    fmag = np.sqrt(fx ** 2 + fy ** 2 + fz ** 2) + 1e-10
    # interface normal from phi gradient (points into liquid)
    pmag = np.sqrt(gx ** 2 + gy ** 2 + gz ** 2) + 1e-10
    # angle between interface normal and surface normal (into-solid)
    cosang = (gx * fx + gy * fy + gz * fz) / (pmag * fmag)
    cosang = np.clip(cosang, -1.0, 1.0)
    theta_eff = np.degrees(np.arccos(cosang[band]))   # normal-to-normal angle
    theta_eff = 180.0 - theta_eff                     # -> contact angle

    # contact-line x extent (spread in x of the band)
    x_lo, x_hi = idx[:, 0].min(), idx[:, 0].max()

    return {
        'n_nodes': int(band.sum()),
        'x_lo': int(x_lo), 'x_hi': int(x_hi),
        'kappa_mean': round(float(k_vals.mean()), 5),
        'kappa_max': round(float(k_vals.max()), 5),
        'alpha_act_mean': round(float(a_vals.mean()), 4),
        'alpha_th_mean': round(float(a_th.mean()), 4),
        'alpha_dev_mean': round(float(np.abs(a_vals - a_th).mean()), 4),
        'theta_eff_mean': round(float(np.nanmean(theta_eff)), 2),
        'theta_eff_std': round(float(np.nanstd(theta_eff)), 2),
    }


def main():
    global A, B
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', default='adaptive', choices=['adaptive', 'const', 'off'])
    ap.add_argument('--n-base', type=int, default=N_BASE)
    ap.add_argument('--steps', type=int, default=STEPS)
    ap.add_argument('--a', type=float, default=A)
    ap.add_argument('--b', type=float, default=B)
    args = ap.parse_args()
    mode = args.mode
    A, B = args.a, args.b
    nx = ny = args.n_base
    nz = int(B + 2 * R_drop + 20)

    sigma_p = rho_l * U0 ** 2 * D0 / 7.9
    beta = 12.0 * sigma_p / xi
    kappa_p = beta * xi ** 2 / 8.0
    M = 0.02 / beta

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
        device='cuda', max_steps=args.steps, output_interval=args.steps + 1,
        g_force=(0.0, 0.0, 0.0))
    solver = AllenCahnSolver(cfg, dtype=torch.float32, stab_mode='fakhari',
        boundary_relax=0.0, geometric_wetting=gw, geo_amplification=amp_arg)
    solver.set_solid(torch.from_numpy(solid),
                     solid_fraction=torch.from_numpy(fraction))

    h = get_surface_height(solid)
    cz = min(int(h[nx // 2, ny // 2]) + 2.0 + R_drop, nz - R_drop - 2)
    C_init, _, u_init = create_fe_droplet_with_impact(nx, ny, nz,
        center=(nx / 2.0, ny / 2.0, cz), radius=R_drop, xi=xi,
        rho_l=rho_l, rho_g=rho_g, u_impact=(0.0, 0.0, U0))
    solver.init_fields(C_init, u_init)

    t0 = time.time()
    diag = []
    for step in range(1, args.steps + 1):
        solver.step()
        if step % DIAG_EVERY == 0:
            phi = solver.phi.detach().cpu().numpy()
            st = contact_band_stats(phi, fraction, kappa, alpha_np, THETA)
            if st:
                st['step'] = step
                gt = geometric_theta(phi, fraction)
                if gt:
                    st['theta_geo'], st['n_contact_cols'] = gt
                diag.append(st)
                print(f"[{step}] n={st['n_nodes']} x=[{st['x_lo']},{st['x_hi']}] "
                      f"kappa={st['kappa_mean']:.4f} "
                      f"alpha_act={st['alpha_act_mean']:.3f} "
                      f"alpha_th={st['alpha_th_mean']:.3f} "
                      f"dev={st['alpha_dev_mean']:.3f} "
                      f"theta={st['theta_eff_mean']:.1f}±{st['theta_eff_std']:.1f}",
                      flush=True)
    elapsed = time.time() - t0

    out = os.path.join(_project_root, 'results', f'contact_line_diag_{mode}.json')
    with open(out, 'w') as f:
        json.dump({'mode': mode, 'theta_eq': THETA, 'sigma': SIGMA,
                   'elapsed_s': round(elapsed, 1), 'diag': diag}, f, indent=1)

    # summary: late-time averages
    late = diag[-4:] if len(diag) >= 4 else diag
    if late:
        avg = {k: round(float(np.mean([d[k] for d in late])), 4)
               for k in ['alpha_act_mean', 'alpha_th_mean', 'alpha_dev_mean',
                         'theta_eff_mean', 'kappa_mean']}
        print(f"SUMMARY[{mode}] late-time: {avg}", flush=True)
    print(f"saved {out} ({elapsed:.0f}s)", flush=True)


if __name__ == '__main__':
    main()
