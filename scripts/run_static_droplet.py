#!/usr/bin/env python3
"""Static sessile droplet benchmark: measure equilibrium contact angle on curved ridge.

Tests:
1. Place a droplet on ridge with zero impact velocity
2. Run until equilibrium (8000 steps)
3. Measure effective contact angle at contact line
4. Compare for α=0 vs α=1.5 vs prescribed θ

Cases: θ = 90°, 120°, 140°, 162° at R*=1.0, N=150
"""

import sys, os, json, time, math
import numpy as np

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)

import torch
from lbm.fe_config import FEConfig
from lbm.fe_ac_solver import AllenCahnSolver
from lbm.fe_droplet import create_fe_droplet_with_impact
from geometry.substrate import create_substrate_with_fraction

torch.cuda.empty_cache()

OUTPUT_DIR = os.path.join(_project_root, 'results')
LOG_FILE = os.path.join(OUTPUT_DIR, 'static_droplet.log')

def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line)
    with open(LOG_FILE, 'a') as f:
        f.write(line + '\n')

def get_surface_height(solid, nx, ny, nz):
    height = np.zeros((nx, ny), dtype=int)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz-1, -1, -1):
                if solid[i, j, k]:
                    height[i, j] = k
                    break
    return height

def measure_contact_angle(phi, solid, nx, ny, nz, sh):
    """Measure effective contact angle at the contact line on a ridge.

    Method: find the outermost fluid node at the first fluid layer above the wall,
    then compute angle between C=0.5 contour and wall tangent.
    """
    # Contact line: outermost grid point on ridge with C > 0.5
    solid_slice = solid[:, :, 0].copy()  # top view
    # For ridge geometry, wall is curved. Find the first fluid layer above wall.

    # Find droplet center (center of mass of phi > 0.5)
    mask = phi > 0.5
    cz = np.argwhere(mask.any(axis=(0,1))).min()  # bottom-most z with fluid

    # Find contact points: x-direction (along ridge)
    # For each x at the fluid layer above wall, find where phi crosses 0.5
    contact_angles = []
    cx = nx / 2.0

    # Scan in +x direction
    for ix in range(int(cx), nx):
        # Wall height at this x
        wall_h = sh[ix, ny//2]
        if wall_h <= 0:
            continue
        # First fluid layer
        fluid_z = int(wall_h) + 1
        if fluid_z >= nz:
            continue
        # Check phi at this point
        val = phi[ix, ny//2, fluid_z]
        val_above = phi[ix, ny//2, min(fluid_z+1, nz-1)]
        if val > 0.5 > val_above or (val < 0.5 < val_above):
            # Interpolate
            z_int = fluid_z + (0.5 - val) / (val_above - val) if val_above != val else fluid_z
            # Wall tangent: normal to wall. On ridge, wall normal = radial from ridge center.
            # For cylindrical ridge at center (cx_w, cy_w, z_w), the tangent is along x.
            # The angle = atan(dz/dx) at contact line.
            # Simplified: angle between (0,1) and interface normal at contact point.
            # Use phi gradient at contact point
            gx = (phi[min(ix+1, nx-1), ny//2, fluid_z] - phi[max(ix-1, 0), ny//2, fluid_z]) / 2.0
            gz = (phi[ix, ny//2, min(fluid_z+1, nz-1)] - phi[ix, ny//2, max(fluid_z-1, 0)]) / 2.0
            angle = math.degrees(abs(math.atan2(gx, -gz)))  # angle from horizontal
            contact_angles.append((ix, angle))
            break

    if contact_angles:
        return contact_angles[-1][1]  # outermost contact angle
    return None

def run_static_case(theta_eq, amp, N=150, N_STEPS=8000):
    torch.cuda.empty_cache()
    D0 = 45.0; R_drop = D0/2.0; R_star = 1.0
    rho_l = 1.0; rho_g = 1.0/828.0
    xi = 4.0; tau = 0.53
    # Static: no impact velocity
    We = 100.0  # high We means low sigma → more spreading
    U0 = -0.005  # very slow approach

    sigma = rho_l * U0**2 * D0 / We
    beta = 12.0 * sigma / xi
    kappa = beta * xi**2 / 8.0
    M = 0.02 / beta
    R_g = R_star * R_drop
    nz = min(int(R_g + 2 + R_drop + 2*R_drop + 15), 300)

    config = FEConfig(nx=N, ny=N, nz=nz, rho_l=rho_l, rho_g=rho_g,
        sigma=sigma, xi=xi, beta=beta, kappa=kappa, M=M,
        tau_l=tau, tau_g=tau, tau_h=tau+0.04, theta_eq=theta_eq,
        device='cuda', max_steps=N_STEPS, output_interval=N_STEPS+1,
        g_force=(0.0, 0.0, 0.0))

    solver = AllenCahnSolver(config, dtype=torch.float32, stab_mode='fakhari',
        boundary_relax=0.0, geometric_wetting=(amp>0), geo_amplification=amp)

    solid, fraction = create_substrate_with_fraction(N, N, nz,
        substrate_type='ridge', R_star=R_star, R_d=R_drop)
    solver.set_solid(solid, solid_fraction=fraction)
    solid_np = solid.cpu().numpy()

    sh = get_surface_height(solid_np, N, N, nz)
    cx, cy = N/2.0, N/2.0
    ridge_top = sh[N//2, N//2]
    # Place droplet close to surface (minimal gap)
    cz = ridge_top + R_drop + 2

    C_init, _, u_init = create_fe_droplet_with_impact(N, N, nz,
        center=(cx, cy, cz), radius=R_drop, xi=xi,
        rho_l=rho_l, rho_g=rho_g, u_impact=(0.0, 0.0, U0))
    solver.init_fields(C_init, u_init)

    t0 = time.time()
    theta_eff_prev = None
    stable_count = 0

    for step in range(1, N_STEPS+1):
        solver.step()
        if step % 500 == 0 and step >= 2000:
            phi = solver.phi.detach().cpu().numpy()
            theta_eff = measure_contact_angle(phi, solid_np, N, N, nz, sh)
            if theta_eff is not None and theta_eff_prev is not None:
                if abs(theta_eff - theta_eff_prev) < 0.5:
                    stable_count += 1
                else:
                    stable_count = 0
            theta_eff_prev = theta_eff
            if stable_count >= 4:  # stable for 2000 steps
                break

    elapsed = time.time() - t0
    phi_f = solver.phi.detach().cpu().numpy()
    theta_final = measure_contact_angle(phi_f, solid_np, N, N, nz, sh)
    error = abs(theta_final - theta_eq) if theta_final else None

    mass0 = float(solver.phi_mass_init)
    mass_f = float(((phi_f > 0.5) & ~solid_np).sum())
    drift = (mass_f - mass0) / mass0 * 100 if mass0 > 0 else 0

    del solver; torch.cuda.empty_cache()
    return {
        'theta_eq': theta_eq, 'amp': amp,
        'theta_eff': theta_final, 'error_deg': error,
        'mass_drift_pct': drift, 'elapsed_s': elapsed,
        'final_step': step,
    }

# ---------------------------------------------------------------------------
N = 150; N_STEPS = 8000
theta_values = [90.0, 120.0, 140.0, 162.0]

log("=" * 70)
log("STATIC SESSILE DROPLET BENCHMARK")
log("=" * 70)
log(f"Device: {torch.cuda.get_device_name(0)}")
log(f"Ridge R*=1.0, N={N}, Steps up to {N_STEPS}")
log(f"Theta values: {theta_values}")
log("")

all_results = []

for theta_eq in theta_values:
    log(f"≡ θ_eq = {theta_eq}° ≡")
    for amp, label in [(0.0, 'α=0'), (1.5, 'α=1.5')]:
        r = run_static_case(theta_eq, amp, N=N, N_STEPS=N_STEPS)
        r['label'] = label
        log(f"  {label}: θ_eff={r['theta_eff']:.1f}°, "
            f"error={r['error_deg']:.1f}°" if r['error_deg'] is not None else f"  {label}: θ_eff=N/A",
            f", mass={r['mass_drift_pct']:.2f}%, "
            f"t={r['elapsed_s']:.0f}s, step={r['final_step']}")
        all_results.append(r)
    log("")

# ---------------------------------------------------------------------------
log("=" * 70)
log("SUMMARY: Contact Angle Error on Curved Surface")
log("=" * 70)
log(f"{'θ_eq':>6}  {'θ_eff(α=0)':>12}  {'θ_eff(α=1.5)':>12}  {'err_α0':>8}  {'err_α15':>8}")
log("-" * 60)
for theta_eq in theta_values:
    rows = [r for r in all_results if abs(r['theta_eq'] - theta_eq) < 0.1]
    r0 = [r for r in rows if r['amp'] == 0.0][0]
    r15 = [r for r in rows if r['amp'] > 0][0]
    log(f"{theta_eq:6.0f}° {r0['theta_eff'] or 'N/A':>10.1s}  "
        f"{r15['theta_eff'] or 'N/A':>10.1s}  "
        f"{r0['error_deg'] or float('nan'):8.1f}° {r15['error_deg'] or float('nan'):8.1f}°")

outpath = os.path.join(OUTPUT_DIR, 'static_droplet.json')
with open(outpath, 'w') as f:
    json.dump(all_results, f, indent=2)
log(f"\nSaved: {outpath}")
log("DONE")
