#!/usr/bin/env python3
"""Phase 3: Full amp sweep on convex hemisphere at N=150.

amp ∈ {0.0, 1.0, 1.5, 2.0, 2.5} — 5 cases × ~7.5 min ≈ 38 min
Key question: does optimal amp match ridge (1.5) or differ?
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from lbm.fe_config import FEConfig
from lbm.fe_ac_solver import AllenCahnSolver
from lbm.fe_droplet import create_fe_droplet_with_impact
from geometry.substrate import create_substrate_with_fraction

torch.cuda.empty_cache()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")
if device == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

D0 = 45.0; R_drop = D0 / 2.0
rho_l = 1.0; rho_g = 1.0/828.0
xi = 4.0; tau = 0.53; U0 = -0.05
We = 7.9; theta_eq = 162.0; R_star = 1.0
N_steps = 2000; n_base = 150

sigma = rho_l * U0**2 * D0 / We
beta = 12.0 * sigma / xi
kappa = beta * xi**2 / 8.0
M = 0.02 / beta
R_g = abs(R_star) * R_drop
nx, ny = n_base, n_base
nz = int(R_g + 2 + R_drop + 2*R_drop + 15)
nz = min(nz, 250)

def get_surface_height(solid, nx, ny, nz):
    h = np.zeros((nx, ny), dtype=int)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz-1, -1, -1):
                if solid[i, j, k]:
                    h[i, j] = k; break
    return h

def measure_droplet(phi_np, solid_np, cx, cy):
    mask = (phi_np > 0.5) & ~solid_np
    if not mask.any():
        return {}
    coords = np.argwhere(mask)
    Dx = float(coords[:, 0].max() - coords[:, 0].min() + 1)
    Dy = float(coords[:, 1].max() - coords[:, 1].min() + 1)
    Dz = float(coords[:, 2].max() - coords[:, 2].min() + 1)
    volume = float(phi_np[~solid_np].sum())
    aspect = Dz / max(Dx, Dy, 1.0)
    k = Dx / Dy if Dy > 0 else 1.0
    dr = np.sqrt((coords[:, 0] - cx)**2 + (coords[:, 1] - cy)**2)
    spread_radius = float(dr.max())
    z_max = float(coords[:, 2].max())
    com_xy = (float(coords[:, 0].mean()), float(coords[:, 1].mean()))
    com_z = float(coords[:, 2].mean())
    return {
        'Dx': Dx, 'Dy': Dy, 'Dz': Dz, 'aspect_ratio': aspect,
        'k': k, 'spread_radius': spread_radius,
        'z_max': z_max, 'com_xy': list(com_xy), 'com_z': com_z,
        'volume': volume,
    }

def run_convex(amp):
    torch.cuda.empty_cache()
    config = FEConfig(
        nx=nx, ny=ny, nz=nz,
        rho_l=rho_l, rho_g=rho_g,
        sigma=sigma, xi=xi, beta=beta, kappa=kappa, M=M,
        tau_l=tau, tau_g=tau, tau_h=tau+0.04,
        theta_eq=theta_eq, device=device,
        max_steps=N_steps, output_interval=N_steps+1,
        g_force=(0.0, 0.0, 0.0),
    )
    solver = AllenCahnSolver(
        config, dtype=torch.float32,
        stab_mode='fakhari', boundary_relax=0.0,
        geometric_wetting=(amp > 0), geo_amplification=amp,
    )
    solid, fraction = create_substrate_with_fraction(
        nx, ny, nz, substrate_type='convex',
        R_star=R_star, R_d=R_drop)
    solver.set_solid(solid, solid_fraction=fraction)

    solid_np = solver.solid.cpu().numpy()
    height = get_surface_height(solid_np, nx, ny, nz)
    cx, cy = nx/2.0, ny/2.0
    ridge_top = height[nx//2, ny//2]
    cz = ridge_top + 2.0 + R_drop
    cz = min(cz, nz - R_drop - 2)

    C_init, _, u_init = create_fe_droplet_with_impact(
        nx, ny, nz, center=(cx, cy, cz), radius=R_drop, xi=xi,
        rho_l=rho_l, rho_g=rho_g, u_impact=(0.0, 0.0, U0))
    solver.init_fields(C_init, u_init)

    history = []
    max_k = 0.0; max_spread = 0.0; min_aspect = float('inf')
    t0 = time.time()
    for step in range(1, N_steps+1):
        solver.step()
        if step % 50 == 0:
            phi_np = solver.phi.detach().cpu().numpy()
            solid_np = solver.solid.cpu().numpy()
            m = measure_droplet(phi_np, solid_np, cx, cy)
            if m:
                m['step'] = step
                history.append(m)
                max_k = max(max_k, m['k'])
                max_spread = max(max_spread, m['spread_radius'])
                min_aspect = min(min_aspect, m['aspect_ratio'])
            if np.isnan(phi_np).any() or phi_np.max() < 0.01:
                break

    elapsed = time.time() - t0
    mem_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    torch.cuda.reset_peak_memory_stats()
    stable = not torch.isnan(solver.phi).any() and solver.phi.max() > 0.01
    del solver
    torch.cuda.empty_cache()

    return {
        'amp': amp, 'geometry': 'convex', 'grid': f'{nx}x{ny}x{nz}',
        'final': history[-1] if history else {},
        'max_k': max_k, 'max_spread': max_spread, 'min_aspect': min_aspect,
        'history': history, 'stable': stable,
        'elapsed_s': round(elapsed, 1), 'mem_mb': round(mem_mb, 0),
    }

amp_values = [0.0, 1.0, 1.5, 2.0, 2.5]
results = []

print(f"Grid: {nx}x{ny}x{nz}")
print(f"Amps: {amp_values}")
print(f"Cases: {len(amp_values)}, est ~{len(amp_values)*7.5:.0f} min\n")

for amp in amp_values:
    print(f"--- convex amp={amp} (N=150) ---")
    r = run_convex(amp)
    results.append(r)
    f = r['final']
    tag = "OK" if r['stable'] else "FAIL"
    print(f"  Dx={f.get('Dx',0):.0f} Dy={f.get('Dy',0):.0f} Dz={f.get('Dz',0):.0f}")
    print(f"  spread_R={f.get('spread_radius',0):.1f} z_max={f.get('z_max',0):.1f}")
    print(f"  aspect={f.get('aspect_ratio',0):.4f} min_aspect={r['min_aspect']:.4f}")
    print(f"  max_spread={r['max_spread']:.1f}")
    print(f"  volume={f.get('volume',0):.1f} [{tag}] ({r['elapsed_s']}s, {r['mem_mb']:.0f}MB)\n")

print("="*70)
print("PHASE 3 CONVEX SWEEP N=150 — SUMMARY")
print("="*70)
print(f"{'amp':>5} {'spread_R':>9} {'z_max':>6} {'aspect':>8} "
      f"{'min_aspect':>11} {'max_spread':>11} {'Dx':>5} {'Dy':>5} {'Dz':>5} "
      f"{'vol':>8} {'stab':>5}")
print("-"*80)
for r in sorted(results, key=lambda x: x['amp']):
    f = r['final']
    print(f"{r['amp']:5.1f} {f.get('spread_radius',0):9.1f} "
          f"{f.get('z_max',0):6.0f} {f.get('aspect_ratio',0):8.4f} "
          f"{r['min_aspect']:11.4f} {r['max_spread']:11.1f} "
          f"{f.get('Dx',0):5.0f} {f.get('Dy',0):5.0f} {f.get('Dz',0):5.0f} "
          f"{f.get('volume',0):8.1f} {'OK' if r['stable'] else 'FAIL'}")

# Ridge reference (from N=80 sweep)
print("\n" + "-"*70)
print("Ridge reference (N=80 sweep):")
print(f"{'amp':>5} {'k_max':>7} {'spread_R':>9} {'aspect':>8} "
      f"{'Dx':>5} {'Dy':>5} {'Dz':>5}")
sw_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       '..', 'results', 'phase3_geometry_sweep.json')
if os.path.exists(sw_path):
    with open(sw_path) as f:
        sweep = json.load(f)
    for r in sorted(sweep, key=lambda x: x['amp']):
        if r['geometry'] != 'ridge' or r['amp'] not in amp_values:
            continue
        if not r.get('stable', True):
            continue
        f = r['final']
        print(f"{r['amp']:5.1f} {r.get('max_k',0):7.4f} "
              f"{f.get('spread_radius',0):9.1f} {f.get('aspect_ratio',0):8.4f} "
              f"{f.get('Dx',0):5.0f} {f.get('Dy',0):5.0f} {f.get('Dz',0):5.0f}")

# Save
save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         '..', 'results', 'phase3_convex_sweep_n150.json')
with open(save_path, 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nSaved to {save_path}")
