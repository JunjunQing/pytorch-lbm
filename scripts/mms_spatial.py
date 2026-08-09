#!/usr/bin/env python3
"""Spatial MMS verification for the discrete gradient operator (W7 fix).

Paper claims "Manufactured solutions (spatial): L2 error converged at
second order" — this script produces the supporting data.

Manufactured field: phi(x,y,z) = sin(2πx/Lx) * cos(2πy/Ly) * exp(z/Lz)
Analytic gradient is known; numeric gradient via the solver's
_lattice_gradient_and_laplacian; L2 error vs resolution N -> order.

CPU-only (no CUDA needed). Saves results/mms_spatial.json.

Usage: venv/bin/python scripts/mms_spatial.py
"""
import sys, os, json
import numpy as np
import torch

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)

from lbm.fe_config import FEConfig
from lbm.fe_ac_solver import AllenCahnSolver


def run(N):
    # Manufactured field in LATTICE coordinates (the discrete gradient
    # operator is defined on a unit lattice; wavelengths equal to the
    # domain size so the field is fully periodic).
    nx = ny = nz = N
    i = np.arange(nx)[:, None, None]
    j = np.arange(ny)[None, :, None]
    k = np.arange(nz)[None, None, :]
    phi = np.sin(2 * np.pi * i / nx) * np.cos(2 * np.pi * j / ny) * np.cos(2 * np.pi * k / nz)
    gx = (2 * np.pi / nx) * np.cos(2 * np.pi * i / nx) * np.cos(2 * np.pi * j / ny) * np.cos(2 * np.pi * k / nz)
    gy = -(2 * np.pi / ny) * np.sin(2 * np.pi * i / nx) * np.sin(2 * np.pi * j / ny) * np.cos(2 * np.pi * k / nz)
    gz = -(2 * np.pi / nz) * np.sin(2 * np.pi * i / nx) * np.cos(2 * np.pi * j / ny) * np.sin(2 * np.pi * k / nz)

    # use the solver's gradient operator (CPU)
    cfg = FEConfig(nx=nx, ny=ny, nz=nz, rho_l=1.0, rho_g=1.0 / 828.0,
        sigma=0.01, xi=4.0, beta=1.0, kappa=1.0, M=1.0,
        tau_l=0.53, tau_g=0.53, tau_h=0.57, theta_eq=162.0,
        device='cpu', max_steps=1, output_interval=2, g_force=(0, 0, 0))
    sol = AllenCahnSolver(cfg, dtype=torch.float64, stab_mode='fakhari',
        boundary_relax=0.0, geometric_wetting=False)
    sol.phi = torch.from_numpy(phi).double()
    grad, lap = sol._lattice_gradient_and_laplacian(sol.phi)
    grad = grad.numpy()  # (3, nx, ny, nz)

    h = 1.0 / nx
    err_x = np.sqrt(((grad[0] - gx) ** 2).mean())
    err_y = np.sqrt(((grad[1] - gy) ** 2).mean())
    err_z = np.sqrt(((grad[2] - gz) ** 2).mean())
    err_total = np.sqrt((err_x ** 2 + err_y ** 2 + err_z ** 2) / 3)
    return h, err_total, err_x, err_y, err_z


def main():
    print("Spatial MMS for the discrete gradient operator", flush=True)
    results = []
    for N in [32, 64, 128]:
        h, err, ex, ey, ez = run(N)
        results.append({'N': N, 'h': h, 'L2_total': err,
                        'L2_x': ex, 'L2_y': ey, 'L2_z': ez})
        print(f"N={N}: L2={err:.6e} (x={ex:.2e} y={ey:.2e} z={ez:.2e})", flush=True)
    # convergence order between successive resolutions
    orders = []
    for i in range(1, len(results)):
        o = (np.log(results[i - 1]['L2_total'] / results[i]['L2_total'])
             / np.log(2))
        orders.append(round(float(o), 2))
        print(f"order N={results[i-1]['N']}->{results[i]['N']}: {o:.2f}",
              flush=True)
    out = {'tests': results, 'orders': orders,
           'verdict': f'converged, measured order ~{orders[-1]:.1f}' if all(o >= 2.0 for o in orders) else 'NOT converging'}
    with open(os.path.join(_project_root, 'results', 'mms_spatial.json'), 'w') as f:
        json.dump(out, f, indent=1)
    print(f"verdict: {out['verdict']} -> results/mms_spatial.json", flush=True)


if __name__ == '__main__':
    main()
