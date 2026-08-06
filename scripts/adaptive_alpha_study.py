#!/usr/bin/env python3
"""Adaptive alpha_geo feasibility study — elliptical ridge.

Core question: constant alpha=1.5 is calibrated for a FIXED curvature R*.
On a geometry with SPATIALLY VARYING curvature (elliptical ridge: strong
curvature at the apex -> nearly flat at the flanks) a constant alpha must
either under-amplify (apex) or over-amplify (flanks). This study tests a
per-node adaptive alpha field:

    alpha(x) = clamp(1 + xi * kappa(x) * |cot(theta_eq)|, 1, alpha_max)

where kappa(x) is the local surface curvature estimated from the solid
fraction field (divergence of the normalized gradient, Gaussian-smoothed).

Modes:
  --mode=off      alpha=0            (wetting BC disabled)
  --mode=const    alpha=1.5          (current recommended default)
  --mode=adaptive alpha=field        (this study)
  --sigma N       smoothing radius for curvature estimation (adaptive)
  --alpha-max X   clamp upper bound  (adaptive)
  --n-base N      grid resolution    (default 150)
  --steps N       simulation steps   (default 2000)

Outputs JSON to results/adaptive_alpha_<tag>.json and prints the curvature
estimation diagnostic (estimated vs analytic apex curvature).

Usage: venv/bin/python scripts/adaptive_alpha_study.py --mode=adaptive --sigma=2
"""
import sys, os, json, time, argparse
import numpy as np
import torch

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)

from lbm.fe_config import FEConfig
from lbm.fe_ac_solver import AllenCahnSolver
from lbm.fe_droplet import create_fe_droplet_with_impact

D0, R_drop = 45.0, 22.5
rho_l, rho_g = 1.0, 1.0 / 828.0
xi, tau, U0 = 4.0, 0.53, -0.05
We = 7.9


def ellipse_ridge(nx, ny, nz, a, b, cx=None, xi_wall=2.0):
    """Elliptical ridge (semi-ellipse in x-z plane, extruded along y).

    Returns (solid, fraction). Apex at (cx, b) with curvature kappa_top=b/a^2.
    """
    cx = nx / 2.0 if cx is None else cx
    i, j, k = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz),
                          indexing='ij')
    # signed distance approx: r = sqrt((x/a)^2 + (z/b)^2); sdf ~ (r-1)*min(a,b)
    r = np.sqrt(((i - cx) / a) ** 2 + (k / b) ** 2)
    sdf = (r - 1.0) * min(a, b)
    solid = (sdf < 0.0) & (k > 0)
    solid[:, :, 0] = True  # bottom wall
    fraction = np.clip(0.5 * (1.0 - np.tanh(sdf / xi_wall)), 0.0, 1.0)
    fraction[solid & (k > 0)] = 1.0
    fraction[:, :, 0] = 1.0
    return solid, fraction.astype(np.float32)


def surface_curvature(fraction, sigma=2.0):
    """Estimate local surface curvature kappa = div(n_hat) from fraction field.

    n_hat = grad(f)/|grad(f)|; kappa = div(n_hat). Gaussian-smoothed with
    radius sigma (lattice units). Returns kappa field (nz x ny x nx order
    matching the solver's tensor layout: (nx, ny, nz) numpy -> keep same).
    """
    f = fraction.astype(np.float64)
    # central differences
    gy, gx, gz = np.gradient(f, axis=(0, 1, 2))
    gmag = np.sqrt(gx ** 2 + gy ** 2 + gz ** 2) + 1e-10
    nx_, ny_, nz_ = gx / gmag, gy / gmag, gz / gmag
    # divergence of normalized gradient (second derivatives of n)
    dnx_dx = np.gradient(nx_, axis=1)
    dny_dy = np.gradient(ny_, axis=0)
    dnz_dz = np.gradient(nz_, axis=2)
    kappa = dnx_dx + dny_dy + dnz_dz
    # zero out far-from-surface values (keep only where fraction is not 0/1)
    mask = (fraction > 0.02) & (fraction < 0.98)
    kappa = np.where(mask, kappa, 0.0)
    # Gaussian smoothing
    if sigma > 0:
        kernel = _gaussian_kernel(sigma)
        kappa = _convolve3d(kappa, kernel)
        kappa = np.where(mask, kappa, 0.0)
    return kappa


def _gaussian_kernel(sigma):
    r = int(3 * sigma)
    ax = np.arange(-r, r + 1)
    g = np.exp(-(ax ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    return g


def _convolve3d(arr, kernel1d):
    """Separable 3D convolution with a 1D kernel (no scipy dependency)."""
    for axis in range(3):
        pad = len(kernel1d) // 2
        padded = np.pad(arr, [(pad, pad)] * 3, mode='edge')
        out = np.zeros_like(arr)
        for off in range(-pad, pad + 1):
            sl = [slice(None)] * 3
            for d in range(3):
                start = pad + (off if d == axis else 0)
                sl[d] = slice(start, start + arr.shape[d])
            out += kernel1d[off + pad] * padded[tuple(sl)]
        arr = out
    return arr


def alpha_field(kappa, theta_eq, alpha_max=2.0):
    """alpha(x) = 1 + xi*kappa(x)*|cot(theta_eq)|, clamped to [1, alpha_max]."""
    cot = abs(1.0 / np.tan(np.radians(theta_eq)))
    alpha = 1.0 + xi * np.abs(kappa) * cot
    return np.clip(alpha, 1.0, alpha_max).astype(np.float32)


def get_surface_height(solid):
    h = np.zeros((solid.shape[0], solid.shape[1]), dtype=int)
    unset = np.ones_like(h, dtype=bool)
    for k in range(solid.shape[2] - 1, -1, -1):
        layer = solid[:, :, k] & unset
        if layer.any():
            h = np.where(layer, k, h)
            unset &= ~layer
    return h


def run_case(mode, a, b, theta_eq, n_base=150, n_steps=2000,
             sigma=2.0, alpha_max=2.0, tag='', we=7.9):
    """Run one elliptical-ridge impact case. mode: off | const | adaptive."""
    torch.cuda.empty_cache()
    nx = ny = n_base
    nz = int(b + 2 * R_drop + 20)  # apex height + droplet + margin
    nz = min(nz, 240)

    sigma_p = rho_l * U0 ** 2 * D0 / we
    beta = 12.0 * sigma_p / xi
    kappa_p = beta * xi ** 2 / 8.0
    M = 0.02 / beta

    solid, fraction = ellipse_ridge(nx, ny, nz, a, b)
    kappa = surface_curvature(fraction, sigma=sigma)

    if mode == 'off':
        amp_arg = 0.0
        geo_wet = False
    elif mode == 'const':
        amp_arg = 1.5
        geo_wet = True
    else:  # adaptive
        amp_field = alpha_field(kappa, theta_eq, alpha_max)
        # restrict to a band near the surface for numerical hygiene
        amp_arg = torch.from_numpy(amp_field).to('cuda')
        geo_wet = True

    config = FEConfig(nx=nx, ny=ny, nz=nz, rho_l=rho_l, rho_g=rho_g,
        sigma=sigma_p, xi=xi, beta=beta, kappa=kappa_p, M=M,
        tau_l=tau, tau_g=tau, tau_h=tau + 0.04, theta_eq=theta_eq,
        device='cuda', max_steps=n_steps, output_interval=n_steps + 1,
        g_force=(0.0, 0.0, 0.0))

    solver = AllenCahnSolver(config, dtype=torch.float32, stab_mode='fakhari',
        boundary_relax=0.0, geometric_wetting=geo_wet, geo_amplification=amp_arg)
    solver.set_solid(torch.from_numpy(solid), solid_fraction=torch.from_numpy(fraction))

    h = get_surface_height(solid)
    apex_top = int(h[nx // 2, ny // 2])
    cz = min(apex_top + 2.0 + R_drop, nz - R_drop - 2)

    from lbm.fe_droplet import create_fe_droplet_with_impact
    C_init, _, u_init = create_fe_droplet_with_impact(nx, ny, nz,
        center=(nx / 2.0, ny / 2.0, cz), radius=R_drop, xi=xi,
        rho_l=rho_l, rho_g=rho_g, u_impact=(0.0, 0.0, U0))
    solver.init_fields(C_init, u_init)

    solid_np = solid
    t0 = time.time()
    max_k, max_k_step = 1.0, 0
    Dx_peak, Dy_peak = 0.0, 0.0
    stable = True
    k_history = []
    for step in range(1, n_steps + 1):
        solver.step()
        if step % 100 == 0:
            phi = solver.phi.detach().cpu().numpy()
            if np.isnan(phi).any():
                stable = False
                break
            interface = (phi > 0.5) & ~solid_np
            if interface.any():
                coords = np.argwhere(interface)
                Dx = float(coords[:, 0].max() - coords[:, 0].min() + 1)
                Dy = float(coords[:, 1].max() - coords[:, 1].min() + 1)
                k = Dx / Dy if Dy > 0 else 1.0
                k_history.append({'step': step, 'k': round(k, 4),
                                  'Dx': round(Dx, 1), 'Dy': round(Dy, 1)})
                if k > max_k:
                    max_k, max_k_step = k, step
                    Dx_peak, Dy_peak = Dx, Dy

    elapsed = time.time() - t0
    # curvature estimator diagnostic
    mask = (fraction > 0.02) & (fraction < 0.98)
    kappa_apex = float(np.abs(kappa[nx // 2, ny // 2, :]).max()) if kappa.any() else 0.0
    kappa_analytic = b / a ** 2
    result = {
        'tag': tag, 'mode': mode, 'sigma': sigma, 'alpha_max': alpha_max,
        'a': a, 'b': b, 'theta_eq': theta_eq, 'grid': f'{nx}x{ny}x{nz}',
        'kappa_apex_estimated': round(kappa_apex, 5),
        'kappa_apex_analytic': round(kappa_analytic, 5),
        'kappa_rel_err_pct': round(abs(kappa_apex - kappa_analytic) / kappa_analytic * 100, 1),
        'max_k': round(max_k, 4), 'max_k_step': max_k_step,
        'Dx': round(Dx_peak, 1), 'Dy': round(Dy_peak, 1),
        'stable': stable, 'elapsed_s': round(elapsed, 1),
        'n_alpha_gt1': int((alpha_field(surface_curvature(fraction, sigma),
                                        theta_eq, alpha_max) > 1.0).sum()) if mode == 'adaptive' else 0,
    }
    print(json.dumps(result), flush=True)
    del solver
    return result, k_history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['off', 'const', 'adaptive'], required=True)
    ap.add_argument('--sigma', type=float, default=2.0)
    ap.add_argument('--alpha-max', type=float, default=2.0)
    ap.add_argument('--n-base', type=int, default=150)
    ap.add_argument('--steps', type=int, default=2000)
    ap.add_argument('--a', type=float, default=35.0)
    ap.add_argument('--b', type=float, default=54.4)
    ap.add_argument('--theta', type=float, default=162.0)
    ap.add_argument('--we', type=float, default=7.9)
    ap.add_argument('--tag', default='')
    args = ap.parse_args()

    tag = args.tag or f"{args.mode}_s{args.sigma}"
    r, hist = run_case(args.mode, args.a, args.b, args.theta,
                       n_base=args.n_base, n_steps=args.steps,
                       sigma=args.sigma, alpha_max=args.alpha_max, tag=tag,
                       we=args.we)
    # save per-run history
    out = os.path.join(_project_root, 'results', f'adaptive_alpha_{tag}.json')
    with open(out, 'w') as f:
        json.dump({'result': r, 'history': hist}, f, indent=1)
    print(f"saved {out}", flush=True)


if __name__ == '__main__':
    main()
