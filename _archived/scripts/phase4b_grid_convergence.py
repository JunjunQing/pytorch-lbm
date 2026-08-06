#!/usr/bin/env python3
"""Phase 4b: Grid convergence study.

Ridge amp=1.5 at N=100, 120, 180.
Combine with existing data: N=80 (Phase 3), N=150 (just run).
Show k_max converges to Liu 2015 target (2.6).
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

# Fixed physical params (D0=45 is the physical size)
D0 = 45.0
rho_l = 1.0; rho_g = 1.0/828.0
xi = 4.0; tau = 0.53; U0 = -0.05
We = 7.9; theta_eq = 162.0; R_star = 1.0
N_steps = 2000
amp = 1.5

def get_surface_height(solid, nx, ny, nz):
    h = np.zeros((nx, ny), dtype=int)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz-1, -1, -1):
                if solid[i, j, k]:
                    h[i, j] = k; break
    return h

def run_case(n_base):
    R_drop = D0 / 2.0
    sigma = rho_l * U0**2 * D0 / We
    beta = 12.0 * sigma / xi
    kappa = beta * xi**2 / 8.0
    M = 0.02 / beta
    R_g = abs(R_star) * R_drop
    nx, ny = n_base, n_base
    nz = int(R_g + 2 + R_drop + 2*R_drop + 15)
    nz = min(nz, 250)

    # Scale D0 with resolution to maintain physical droplet size in LU
    # Actually D0=45 is fixed, so at different n_base the droplet occupies
    # the same number of LU. This is NOT the standard scaling approach.

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
        geometric_wetting=True, geo_amplification=amp,
    )
    solid, fraction = create_substrate_with_fraction(
        nx, ny, nz, substrate_type='ridge',
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
    max_k = 0.0
    t0 = time.time()
    for step in range(1, N_steps+1):
        solver.step()
        if step % 50 == 0:
            phi_np = solver.phi.detach().cpu().numpy()
            solid_np = solver.solid.cpu().numpy()
            mask = (phi_np > 0.5) & ~solid_np
            if not mask.any():
                continue
            coords = np.argwhere(mask)
            Dx = float(coords[:, 0].max() - coords[:, 0].min() + 1)
            Dy = float(coords[:, 1].max() - coords[:, 1].min() + 1)
            k = Dx / Dy if Dy > 0 else 0
            history.append({'step': step, 'Dx': Dx, 'Dy': Dy, 'k': k})
            max_k = max(max_k, k)
            if np.isnan(phi_np).any() or phi_np.max() < 0.01:
                break

    elapsed = time.time() - t0
    mem_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    torch.cuda.reset_peak_memory_stats()
    stable = not torch.isnan(solver.phi).any() and solver.phi.max() > 0.01
    del solver
    torch.cuda.empty_cache()

    return {
        'n_base': n_base, 'grid': f'{nx}x{ny}x{nz}',
        'max_k': max_k, 'history': history, 'stable': stable,
        'elapsed_s': round(elapsed, 1), 'mem_mb': round(mem_mb, 0),
    }

resolutions = [100, 120, 180]
results = []

for n in resolutions:
    print(f"\n--- Ridge N={n} (amp={amp}) ---")
    r = run_case(n)
    results.append(r)
    print(f"  max_k={r['max_k']:.4f}, grid={r['grid']}")
    print(f"  [{r['elapsed_s']}s, {r['mem_mb']:.0f}MB] stable={r['stable']}")

# Combine with existing data
existing = [
    {'n_base': 80, 'grid': '80x80x107', 'max_k': 2.3548, 'source': 'Phase3 sweep'},
    {'n_base': 150, 'grid': '150x150x107', 'max_k': 2.6552, 'source': 'Phase4 ridge N150'},
]

all_data = existing + results
all_data.sort(key=lambda x: x['n_base'])

print("\n" + "="*70)
print("GRID CONVERGENCE — Ridge amp=1.5")
print("="*70)
print(f"{'N':>5} {'grid':>16} {'k_max':>7} {'error':>8} {'stable':>7}")
print("-" * 50)
target = 2.6
for d in all_data:
    err = (d['max_k'] - target) / target * 100
    print(f"{d['n_base']:5d} {d['grid']:>16} {d['max_k']:7.4f} "
          f"{err:>+7.1f}% {'OK' if d.get('stable', True) else 'FAIL'}")

# Save
save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         '..', 'results', 'phase4_grid_convergence.json')
with open(save_path, 'w') as f:
    json.dump(all_data, f, indent=2, default=str)
print(f"\nSaved to {save_path}")
