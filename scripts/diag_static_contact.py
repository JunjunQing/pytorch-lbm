"""Diagnose why measure_contact_angle returns None in static droplet tests.

Quick run at reduced resolution to inspect:
  1. Whether the droplet actually contacts the wall
  2. phi values in the first fluid layer above the wall
  3. Where the contact line (phi=0.5 crossing) is, if anywhere
"""
import sys, os, time, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

from lbm.fe_config import FEConfig
from lbm.fe_ac_solver import AllenCahnSolver
from lbm.fe_droplet import create_fe_droplet_with_impact
from geometry.substrate import create_substrate_with_fraction


def get_surface_height(solid, nx, ny, nz):
    height = np.zeros((nx, ny), dtype=int)
    solid = solid.cpu().numpy() if torch.is_tensor(solid) else solid
    for i in range(nx):
        for j in range(ny):
            for k in range(nz - 1, -1, -1):
                if solid[i, j, k]:
                    height[i, j] = k
                    break
    return height


def measure_contact_angle(phi, solid, nx, ny, nz, sh, verbose=False):
    """Copy of run_static_droplet.measure_contact_angle + diagnostics."""
    mask = phi > 0.5
    n_liquid = mask.sum()
    if n_liquid == 0:
        if verbose: print("  [diag] no liquid (phi>0.5) at all!")
        return None, {}
    # liquid extent
    coords = np.argwhere(mask)
    zmin_l, zmax_l = coords[:, 2].min(), coords[:, 2].max()
    xmin_l, xmax_l = coords[:, 0].min(), coords[:, 0].max()

    cx = nx / 2.0
    contact_angles = []
    diag = {'n_liquid': int(n_liquid), 'zmin_l': int(zmin_l), 'zmax_l': int(zmax_l),
            'xmin_l': int(xmin_l), 'xmax_l': int(xmax_l)}

    # wall height at ridge top
    wall_top = sh[nx // 2, ny // 2]
    diag['wall_top'] = int(wall_top)

    # phi in first fluid layer at x-center
    fz = int(wall_top) + 1
    diag['first_layer_z'] = fz
    diag['phi_first_layer_center'] = float(phi[nx // 2, ny // 2, fz]) if fz < nz else None
    diag['phi_first_layer_max'] = float(phi[:, ny // 2, fz].max()) if fz < nz else None

    # scan +x for crossing
    n_cross = 0
    for ix in range(int(cx), nx):
        wall_h = sh[ix, ny // 2]
        if wall_h <= 0:
            continue
        fluid_z = int(wall_h) + 1
        if fluid_z >= nz:
            continue
        val = phi[ix, ny // 2, fluid_z]
        val_above = phi[ix, ny // 2, min(fluid_z + 1, nz - 1)]
        if val > 0.5 > val_above or (val < 0.5 < val_above):
            n_cross += 1
            gx = (phi[min(ix + 1, nx - 1), ny // 2, fluid_z] - phi[max(ix - 1, 0), ny // 2, fluid_z]) / 2.0
            gz = (phi[ix, ny // 2, min(fluid_z + 1, nz - 1)] - phi[ix, ny // 2, max(fluid_z - 1, 0)]) / 2.0
            angle = math.degrees(abs(math.atan2(gx, -gz)))
            contact_angles.append((ix, angle))
            break
    diag['n_crossings'] = n_cross
    if contact_angles:
        diag['contact_ix'] = contact_angles[-1][0]
        diag['contact_angle'] = contact_angles[-1][1]
        if verbose:
            print(f"  [diag] contact line at ix={contact_angles[-1][0]}, angle={contact_angles[-1][1]:.1f}deg")
        return contact_angles[-1][1], diag
    if verbose:
        print("  [diag] NO contact line found in first fluid layer scan")
    return None, diag


def run_diag(theta_eq=162.0, amp=1.5, N=80, N_STEPS=2000, U0=-0.005, We=100.0):
    torch.cuda.empty_cache()
    D0 = 45.0; R_drop = D0 / 2.0; R_star = 1.0
    rho_l = 1.0; rho_g = 1.0 / 828.0
    xi = 4.0; tau = 0.53

    sigma = rho_l * U0**2 * D0 / We
    beta = 12.0 * sigma / xi
    kappa = beta * xi**2 / 8.0
    M = 0.02 / beta
    R_g = R_star * R_drop
    nz = min(int(R_g + 2 + R_drop + 2 * R_drop + 15), 300)

    print(f"sigma={sigma:.3e}, beta={beta:.3e}, kappa={kappa:.3e}, M={M:.1f}")
    print(f"grid: {N}x{N}x{nz}")

    config = FEConfig(nx=N, ny=N, nz=nz, rho_l=rho_l, rho_g=rho_g,
        sigma=sigma, xi=xi, beta=beta, kappa=kappa, M=M,
        tau_l=tau, tau_g=tau, tau_h=tau + 0.04, theta_eq=theta_eq,
        device='cuda', max_steps=N_STEPS, output_interval=N_STEPS + 1,
        g_force=(0.0, 0.0, 0.0))

    solver = AllenCahnSolver(config, dtype=torch.float32, stab_mode='fakhari',
        boundary_relax=0.0, geometric_wetting=(amp > 0), geo_amplification=amp)

    solid, fraction = create_substrate_with_fraction(N, N, nz,
        substrate_type='ridge', R_star=R_star, R_d=R_drop)
    solver.set_solid(solid, solid_fraction=fraction)
    solid_np = np.asarray(solid)

    sh = get_surface_height(solid_np, N, N, nz)
    cx, cy = N / 2.0, N / 2.0
    ridge_top = sh[N // 2, N // 2]
    cz = ridge_top + R_drop + 2

    C_init, _, u_init = create_fe_droplet_with_impact(N, N, nz,
        center=(cx, cy, cz), radius=R_drop, xi=xi,
        rho_l=rho_l, rho_g=rho_g, u_impact=(0.0, 0.0, U0))
    solver.init_fields(C_init, u_init)
    print(f"droplet center z={cz}, radius={R_drop}, bottom at {cz - R_drop}, wall_top={ridge_top}")

    t0 = time.time()
    for step in range(1, N_STEPS + 1):
        solver.step()
        if step % 500 == 0:
            phi = solver.phi.detach().cpu().numpy()
            theta_eff, diag = measure_contact_angle(phi, solid_np, N, N, nz, sh)
            print(f"[{step:5d}] theta_eff={theta_eff}, n_liquid={diag['n_liquid']}, "
                  f"z_liquid=[{diag['zmin_l']},{diag['zmax_l']}], wall_top={diag['wall_top']}, "
                  f"first_layer_phi_center={diag['phi_first_layer_center']:.3f}, "
                  f"first_layer_phi_max={diag['phi_first_layer_max']:.3f}, crossings={diag['n_crossings']}")
            print(f"         liquid x=[{diag['xmin_l']},{diag['xmax_l']}]")

    phi_f = solver.phi.detach().cpu().numpy()
    theta_final, diag_f = measure_contact_angle(phi_f, solid_np, N, N, nz, sh, verbose=True)
    print(f"FINAL theta_eff={theta_final}, elapsed={time.time()-t0:.1f}s")
    print(json_diag(diag_f))
    return theta_final


def json_diag(d):
    return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in d.items()}


if __name__ == '__main__':
    theta = float(sys.argv[1]) if len(sys.argv) > 1 else 162.0
    amp = float(sys.argv[2]) if len(sys.argv) > 2 else 1.5
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 80
    steps = int(sys.argv[4]) if len(sys.argv) > 4 else 2000
    run_diag(theta_eq=theta, amp=amp, N=n, N_STEPS=steps)
