#!/usr/bin/env python3
"""Phase 4c: Spurious currents check.

Run static droplet (no impact) on ridge at N=100.
Compare amp=0 vs amp=1.5: max |u| in gas phase.
Geo_amplification should NOT increase spurious currents.
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

D0 = 45.0; R_drop = D0 / 2.0
rho_l = 1.0; rho_g = 1.0/828.0
xi = 4.0; tau = 0.53
We = 7.9; theta_eq = 162.0; R_star = 1.0
N_steps = 1000; n_base = 100

nx = ny = n_base
R_g = abs(R_star) * R_drop
nz = int(R_g + 2 + R_drop + 2*R_drop + 15)
nz = min(nz, 200)

def get_surface_height(solid):
    h = np.zeros((nx, ny), dtype=int)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz-1, -1, -1):
                if solid[i, j, k]:
                    h[i, j] = k; break
    return h

def run_static(amp):
    torch.cuda.empty_cache()
    sigma = rho_l * 0.05**2 * D0 / We
    beta = 12.0 * sigma / xi
    kappa = beta * xi**2 / 8.0
    M = 0.02 / beta

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
        nx, ny, nz, substrate_type='ridge',
        R_star=R_star, R_d=R_drop)
    solver.set_solid(solid, solid_fraction=fraction)

    solid_np = solver.solid.cpu().numpy()
    height = get_surface_height(solid_np)
    cx, cy = nx/2.0, ny/2.0
    ridge_top = height[nx//2, ny//2]
    cz = ridge_top + 2.0 + R_drop
    cz = min(cz, nz - R_drop - 2)

    C_init, _, u_init = create_fe_droplet_with_impact(
        nx, ny, nz, center=(cx, cy, cz), radius=R_drop, xi=xi,
        rho_l=rho_l, rho_g=rho_g, u_impact=(0.0, 0.0, 0.0))
    solver.init_fields(C_init, u_init)

    max_u_hist = []
    max_phi_hist = []
    t0 = time.time()
    for step in range(1, N_steps+1):
        solver.step()
        if step % 50 == 0:
            u = solver.u.detach().cpu().numpy()
            phi = solver.phi.detach().cpu().numpy()
            solid_np = solver.solid.cpu().numpy()
            vel_mag = np.sqrt(u[0]**2 + u[1]**2 + u[2]**2)
            max_u = float(vel_mag.max())
            max_phi = float(phi.max())

            # Gas phase spurious currents: exclude solid nodes
            gas_mask = (phi < 0.01) & ~solid_np
            max_u_gas = float(vel_mag[gas_mask].max()) if gas_mask.sum() > 0 else 0

            max_u_hist.append({'step': step, 'max_u_all': max_u,
                              'max_u_gas': max_u_gas, 'max_phi': max_phi})
            max_phi_hist.append(step)
            if np.isnan(phi).any():
                break

    elapsed = time.time() - t0
    mem_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    torch.cuda.reset_peak_memory_stats()
    del solver
    torch.cuda.empty_cache()

    return {'amp': amp, 'u_history': max_u_hist, 'elapsed_s': round(elapsed, 1),
            'mem_mb': round(mem_mb, 0), 'stable': not any(np.isnan(h['max_u_all']) for h in max_u_hist)}

results = []
for amp in [0.0, 1.0, 1.5]:
    print(f"\n--- Static ridge amp={amp} (N={n_base}) ---")
    r = run_static(amp)
    results.append(r)
    u_hist = r['u_history']
    if u_hist:
        final = u_hist[-1]
        peak = max(u_hist, key=lambda x: x['max_u_all'])
        print(f"  Final: max_u_all={final['max_u_all']:.6e}, max_u_gas={final['max_u_gas']:.6e}")
        print(f"  Peak:  max_u_all={peak['max_u_all']:.6e}, max_u_gas={peak['max_u_gas']:.6e}")
    print(f"  [{r['elapsed_s']}s, {r['mem_mb']:.0f}MB]")

print("\n" + "="*60)
print("SPURIOUS CURRENTS COMPARISON")
print("="*60)
print(f"{'amp':>5} {'final_max_u':>14} {'final_u_gas':>14} {'peak_max_u':>14} {'peak_u_gas':>14}")
print("-" * 65)
for r in results:
    u_hist = r['u_history']
    if u_hist:
        final = u_hist[-1]
        peak = max(u_hist, key=lambda x: x['max_u_all'])
        print(f"{r['amp']:5.1f} {final['max_u_all']:14.6e} {final['max_u_gas']:14.6e} "
              f"{peak['max_u_all']:14.6e} {peak['max_u_gas']:14.6e}")

save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         '..', 'results', 'phase4_spurious_currents.json')
with open(save_path, 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nSaved to {save_path}")
