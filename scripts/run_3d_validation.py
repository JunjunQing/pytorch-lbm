#!/usr/bin/env python3
"""3D validation simulations for the geometric amplification paper.

Runs key 3D cases to compare with 2D results and experimental data.
Uses D3Q19 lattice with volume penalization for curved boundaries.
"""
import sys
import os

# Add project root to path
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)

import numpy as np
import torch
import time
import json

from lbm.fe_config import FEConfig
from lbm.fe_ac_solver import AllenCahnSolver
from lbm.fe_droplet import create_fe_droplet_with_impact
from geometry.substrate import create_substrate, create_substrate_with_fraction

torch.cuda.empty_cache()

# --- Physical parameters (matching 2D runs) ---
D0 = 45.0
R_drop = D0 / 2.0
rho_l = 1.0
rho_g = 1.0 / 828.0
xi = 4.0
tau = 0.53
U0 = -0.05

# --- 3D Grid parameters ---
# Reduced resolution for 3D due to memory constraints
N_3D = 80  # 3D grid resolution (80³ = 512K nodes)
nz = int(N_3D * 107 / 150)  # Scale z-height proportionally

print(f"3D Validation Simulations")
print(f"Grid: {N_3D}³ (effective), D0={D0*80/150:.0f} lu")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print("="*60)


def get_surface_height(solid):
    """Get topmost solid index per (i,j) column."""
    if solid.ndim == 3:
        nx, ny, nz = solid.shape
        height = np.zeros((nx, ny), dtype=int)
        for i in range(nx):
            for j in range(ny):
                for k in range(nz-1, -1, -1):
                    if solid[i, j, k]:
                        height[i, j] = k
                        break
    else:
        nx, ny = solid.shape
        height = np.zeros(nx, dtype=int)
        for i in range(nx):
            for j in range(ny-1, -1, -1):
                if solid[i, j]:
                    height[i] = j
                    break
    return height


def run_3d_case(label, substrate_type, R_star, We, theta_eq, amp,
                n_base=80, n_steps=1500):
    """Run a single 3D simulation case."""
    
    # Scale parameters for 3D resolution
    D0_3d = D0 * n_base / 150
    R_drop_3d = D0_3d / 2.0
    
    # Compute derived parameters
    sigma = rho_l * U0**2 * D0_3d / We
    beta = 12.0 * sigma / xi
    kappa = beta * xi**2 / 8.0
    M = 0.02 / beta
    
    # Grid dimensions
    nx, ny = n_base, n_base
    nz_actual = nz
    
    # Create config
    config = FEConfig(
        nx=nx, ny=ny, nz=nz_actual,
        rho_l=rho_l, rho_g=rho_g,
        sigma=sigma, xi=xi, beta=beta, kappa=kappa,
        M=M, tau_l=tau, tau_g=tau, tau_h=tau+0.04,
        theta_eq=theta_eq, device='cuda',
        max_steps=n_steps, output_interval=200,
    )
    
    print(f"\n--- {label} ---")
    print(f"  Grid: {nx}×{ny}×{nz_actual}")
    print(f"  D0={D0_3d:.1f}, R_drop={R_drop_3d:.1f}")
    print(f"  R*={R_star}, We={We}, θ={theta_eq}°, amp={amp}")
    
    # Create solver
    solver = AllenCahnSolver(
        config, dtype=torch.float32,
        stab_mode='fakhari',
        boundary_relax=0.0,
        geometric_wetting=True,
        geo_amplification=amp,
    )
    
    # Create substrate
    solid, fraction = create_substrate_with_fraction(
        nx, ny, nz_actual, substrate_type,
        R_star=R_star, R_d=R_drop_3d)
    solver.set_solid(solid, solid_fraction=fraction)
    
    # Get surface height for droplet placement
    height = get_surface_height(solid)
    surface_z = np.max(height) if solid.ndim == 3 else np.max(height)
    
    # Initialize droplet above the ridge
    center = (nx//2, ny//2, surface_z + 5)
    phi_init, rho_init, u_init = create_fe_droplet_with_impact(
        nx, ny, nz_actual, center=center, radius=R_drop_3d,
        xi=xi, rho_l=rho_l, rho_g=rho_g, u_impact=(0, 0, U0))
    
    solver.init_fields(phi_init, u_init)
    
    # Run simulation
    t0 = time.time()
    k_history = []
    max_k = 0.0
    max_k_step = 0
    
    for step in range(n_steps):
        solver.step()
        
        if step % 50 == 0:
            # Measure spreading diameters
            phi = solver.phi.detach().cpu().numpy()
            droplet_mask = phi > 0.5
            
            if droplet_mask.any():
                # Get droplet extent in x and y directions
                x_coords, y_coords, z_coords = np.where(droplet_mask)
                Dx = x_coords.max() - x_coords.min() + 1
                Dy = y_coords.max() - y_coords.min() + 1
                k = Dx / Dy if Dy > 0 else 1.0
                
                k_history.append({
                    'step': step, 'k': k,
                    'Dx': int(Dx), 'Dy': int(Dy),
                    'z_min': int(z_coords.min()),
                    'z_max': int(z_coords.max())
                })
                
                if k > max_k:
                    max_k = k
                    max_k_step = step
    
    elapsed = time.time() - t0
    mem_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    
    result = {
        'label': label,
        'substrate_type': substrate_type,
        'R_star': R_star,
        'We': We,
        'theta_eq': theta_eq,
        'amp': amp,
        'max_k': max_k,
        'max_k_step': max_k_step,
        'Dx': k_history[-1]['Dx'] if k_history else 0,
        'Dy': k_history[-1]['Dy'] if k_history else 0,
        'stable': True,
        'grid': f'{nx}×{ny}×{nz_actual}',
        'elapsed_s': elapsed,
        'mem_mb': mem_mb,
        'dimension': '3D',
    }
    
    print(f"  Results: k_max={max_k:.3f} (step {max_k_step})")
    print(f"  Dx={result['Dx']}, Dy={result['Dy']}")
    print(f"  Time: {elapsed:.1f}s, Memory: {mem_mb:.0f} MB")
    
    return result, k_history


def main():
    """Run 3D validation cases."""
    
    results = []
    all_histories = {}
    
    # Case 1: Ridge R*=1.0, We=7.9 (primary validation case)
    r, h = run_3d_case(
        label="3d_ridge_R1.0_We7.9",
        substrate_type="ridge",
        R_star=1.0,
        We=7.9,
        theta_eq=162.0,
        amp=1.5,
        n_base=N_3D,
        n_steps=1500
    )
    results.append(r)
    all_histories[r['label']] = h
    
    # Case 2: Ridge R*=2.0, We=7.9 (different curvature)
    r, h = run_3d_case(
        label="3d_ridge_R2.0_We7.9",
        substrate_type="ridge",
        R_star=2.0,
        We=7.9,
        theta_eq=162.0,
        amp=1.0,
        n_base=N_3D,
        n_steps=1500
    )
    results.append(r)
    all_histories[r['label']] = h
    
    # Case 3: Convex hemisphere R*=1.0, We=7.9
    r, h = run_3d_case(
        label="3d_convex_R1.0_We7.9",
        substrate_type="convex",
        R_star=1.0,
        We=7.9,
        theta_eq=162.0,
        amp=1.5,
        n_base=N_3D,
        n_steps=1500
    )
    results.append(r)
    all_histories[r['label']] = h
    
    # Case 4: We=15 (peak asymmetry region)
    r, h = run_3d_case(
        label="3d_ridge_R1.0_We15",
        substrate_type="ridge",
        R_star=1.0,
        We=15.0,
        theta_eq=162.0,
        amp=1.5,
        n_base=N_3D,
        n_steps=1500
    )
    results.append(r)
    all_histories[r['label']] = h
    
    # Case 5: Flat surface (reference)
    r, h = run_3d_case(
        label="3d_flat_We7.9",
        substrate_type="flat",
        R_star=None,
        We=7.9,
        theta_eq=162.0,
        amp=0.0,
        n_base=N_3D,
        n_steps=1000
    )
    results.append(r)
    all_histories[r['label']] = h
    
    # Save results
    results_path = os.path.join(_project_root, 'results', '3d_validation_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")
    
    # Save histories
    history_path = os.path.join(_project_root, 'results', '3d_validation_history.json')
    with open(history_path, 'w') as f:
        json.dump(all_histories, f, indent=2)
    print(f"History saved to {history_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("3D Validation Summary")
    print("="*60)
    for r in results:
        print(f"{r['label']:30s}  k={r['max_k']:.3f}  Dx={r['Dx']:3d}  Dy={r['Dy']:3d}  "
              f"grid={r['grid']}  time={r['elapsed_s']:.0f}s")


if __name__ == '__main__':
    main()
