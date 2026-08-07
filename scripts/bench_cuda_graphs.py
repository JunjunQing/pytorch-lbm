#!/usr/bin/env python3
"""B0: CUDA Graphs benchmark for AllenCahnSolver.step().

Question: how much of the per-step cost is kernel-launch overhead (vs
memory bandwidth)? torch.cuda.graphs captures the whole step() as one
graph and replays it — if the speedup is large, a fused CUDA kernel
(B1) is clearly worthwhile; if small, the solver is bandwidth-bound
and B1's gains will be modest.

Measures: regular step time vs graph-replayed step time, at several
grid sizes (launch overhead dominates more at small grids).

Usage: venv/bin/python scripts/bench_cuda_graphs.py [--n 64]
"""
import sys, os, time, argparse
import numpy as np
import torch

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)

from lbm.fe_config import FEConfig
from lbm.fe_ac_solver import AllenCahnSolver
from lbm.fe_droplet import create_fe_droplet_with_impact
from geometry.substrate import create_substrate_with_fraction
from scripts.adaptive_alpha_study import get_surface_height


def build_solver(n_base):
    D0, R_drop = 45.0, 22.5
    rho_l, rho_g = 1.0, 1.0 / 828.0
    xi, tau, U0 = 4.0, 0.53, -0.05
    sigma = rho_l * U0 ** 2 * D0 / 7.9
    beta = 12.0 * sigma / xi
    kappa = beta * xi ** 2 / 8.0
    M = 0.02 / beta
    nz = 60
    cfg = FEConfig(nx=n_base, ny=n_base, nz=nz, rho_l=rho_l, rho_g=rho_g,
        sigma=sigma, xi=xi, beta=beta, kappa=kappa, M=M,
        tau_l=tau, tau_g=tau, tau_h=tau + 0.04, theta_eq=162.0,
        device='cuda', max_steps=10**6, output_interval=10**6 + 1,
        g_force=(0.0, 0.0, 0.0))
    solver = AllenCahnSolver(cfg, dtype=torch.float32, stab_mode='fakhari',
        boundary_relax=0.0, geometric_wetting=True, geo_amplification=1.5)
    solid, fraction = create_substrate_with_fraction(n_base, n_base, nz,
        substrate_type='ridge', R_star=1.0, R_d=R_drop)
    solver.set_solid(torch.from_numpy(solid),
                     solid_fraction=torch.from_numpy(fraction))
    h = get_surface_height(solid)
    cz = min(int(h[n_base // 2, n_base // 2]) + 2 + R_drop, nz - R_drop - 2)
    C, _, u = create_fe_droplet_with_impact(n_base, n_base, nz,
        center=(n_base / 2, n_base / 2, cz), radius=R_drop, xi=xi,
        rho_l=rho_l, rho_g=rho_g, u_impact=(0.0, 0.0, U0))
    solver.init_fields(C, u)
    return solver


def bench(n_base, iters=200, warmup=30):
    solver = build_solver(n_base)
    # warmup (also allocates memory pools)
    for _ in range(warmup):
        solver.step()
    torch.cuda.synchronize()

    # regular path
    t0 = time.perf_counter()
    for _ in range(iters):
        solver.step()
    torch.cuda.synchronize()
    t_reg = (time.perf_counter() - t0) / iters

    # capture graph
    g = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    # capture on a side stream
    with torch.cuda.graph(g):
        solver.step()
    torch.cuda.synchronize()
    # replay
    t0 = time.perf_counter()
    for _ in range(iters):
        g.replay()
    torch.cuda.synchronize()
    t_graph = (time.perf_counter() - t0) / iters

    print(f"N={n_base}  regular={t_reg*1e3:.3f} ms/step  "
          f"graph={t_graph*1e3:.3f} ms/step  speedup={t_reg/t_graph:.2f}x",
          flush=True)
    return {'N': n_base, 'regular_ms': round(t_reg * 1e3, 3),
            'graph_ms': round(t_graph * 1e3, 3),
            'speedup': round(t_reg / t_graph, 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=64)
    ap.add_argument('--iters', type=int, default=200)
    args = ap.parse_args()
    print(f"CUDA Graphs benchmark | device: "
          f"{torch.cuda.get_device_name(0)} | iters={args.iters}", flush=True)
    results = []
    for n in [args.n]:
        results.append(bench(n, iters=args.iters))
    import json
    with open(os.path.join(_project_root, 'results', 'cuda_graphs_bench.json'), 'w') as f:
        json.dump(results, f, indent=1)
    print("saved results/cuda_graphs_bench.json", flush=True)


if __name__ == '__main__':
    main()
