"""Static droplet contact angle validation on a flat substrate.

Replaces the failed measure_contact_angle approach in run_static_droplet.py
(which used We=100 -> sigma 12.7x too small -> no wetting force -> no contact line).

Fixes:
  1. Normal sigma (We=7.9, same as dynamic experiments) so wetting force is physical
  2. Flat substrate (standard for static contact angle validation)
  3. Spherical-cap measurement: theta from conserved volume V and contact radius r_c

For an axisymmetric sessile drop (no gravity):
    V = pi*h*(3*r_c^2 + h^2)/6        (spherical cap volume)
    R = (r_c^2 + h^2)/(2*h)           (cap radius)
    theta = arctan(2*h*r_c/(r_c^2 - h^2))   (contact angle from horizontal)

Usage: python scripts/run_static_validation.py [theta] [amp] [N] [N_STEPS] [tag]
"""
import sys, os, time, json, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

from lbm.fe_config import FEConfig
from lbm.fe_ac_solver import AllenCahnSolver
from lbm.fe_droplet import create_fe_droplet_with_impact
from geometry.substrate import create_substrate_with_fraction

D0 = 45.0; R_drop = D0 / 2.0
RHO_L = 1.0; RHO_G = 1.0 / 828.0
XI = 4.0; TAU = 0.53
U0_REF = 0.05       # reference velocity for sigma (normal dynamic sigma)
U0 = -0.005         # gentle approach; wetting force does the equilibration
WE = 7.9            # normal Weber -> physical sigma (decoupled from approach speed)


def spherical_cap_angle(volume, r_c, h_max, wall_z):
    """Contact angle from spherical-cap geometry.

    volume : conserved liquid volume (sum of phi)
    r_c    : contact radius (max extent of phi>0.5 in wall-adjacent fluid layer)
    h_max  : max height of phi>0.5 above the wall
    """
    if r_c <= 1.0 or volume <= 0:
        return None
    # Solve h^3 + 3*r_c^2*h - 6*V/pi = 0 via Newton from h_max seed
    target = 6.0 * volume / math.pi
    h = h_max if h_max > 0 else r_c
    for _ in range(60):
        f = h**3 + 3.0 * r_c**2 * h - target
        fp = 3.0 * h**2 + 3.0 * r_c**2
        h_new = h - f / fp
        if abs(h_new - h) < 1e-8:
            h = h_new
            break
        h = h_new
    if h <= 0:
        return None
    denom = r_c**2 - h**2
    if abs(denom) < 1e-8:
        return 90.0
    # atan2 keeps the angle in the correct quadrant (theta>90 when denom<0)
    theta = math.degrees(math.atan2(2.0 * h * r_c, denom))
    return theta


def measure_flat_contact(phi, solid_np, wall_z=1):
    """Measure contact radius, height, volume on a flat substrate (wall at k=0).

    The phi=0.5 isosurface sits ~xi/2=2 lu above the wall (wall phi=0 flattens
    the interface profile), so the contact patch is measured at zmin, the lowest
    layer containing phi>0.5 -- NOT at the fixed first fluid layer.
    """
    mask = phi > 0.5
    vol = float(mask.sum())
    if vol == 0:
        return None
    coords = np.argwhere(mask)
    z_min = int(coords[:, 2].min())
    z_max = int(coords[:, 2].max())
    h_max = z_max - z_min + 1.0
    # contact patch at the lowest liquid layer
    layer = mask[:, :, z_min]
    if not layer.any():
        return None
    ys, xs = np.argwhere(layer).T
    cy, cx = np.mean(ys), np.mean(xs)
    rs = np.sqrt((xs - cx)**2 + (ys - cy)**2)
    r_c = float(rs.max()) + 0.5
    theta = spherical_cap_angle(vol, r_c, h_max, z_min)
    return {'volume': vol, 'r_c': r_c, 'h_max': h_max, 'z_min': z_min,
            'theta': theta, 'z_max': z_max}


def run_case(theta_eq, amp, N=150, N_STEPS=10000, tag='', EMBED=2.0, BO=3.0, GHOST=None):
    torch.cuda.empty_cache()
    sigma = RHO_L * U0_REF**2 * D0 / WE
    beta = 12.0 * sigma / XI
    kappa = beta * XI**2 / 8.0
    M = 0.02 / beta
    # Gravity to seat the sessile drop: Bo = rho g D0^2 / sigma -> g = Bo*sigma/(rho*D0^2)
    g_val = BO * sigma / (RHO_L * D0**2)
    print(f"  Bo={BO} -> g={g_val:.3e} (capillary length {2*sigma/(RHO_L*g_val):.0f} lu)")

    nx = ny = N
    nz = int(1 + 2.5 * R_drop + EMBED)  # room for tall hydrophobic cap + embed
    print(f"[{tag}] theta={theta_eq} amp={amp} grid={nx}x{ny}x{nz} "
          f"sigma={sigma:.5f} N_STEPS={N_STEPS}", flush=True)

    config = FEConfig(nx=nx, ny=ny, nz=nz, rho_l=RHO_L, rho_g=RHO_G,
        sigma=sigma, xi=XI, beta=beta, kappa=kappa, M=M,
        tau_l=TAU, tau_g=TAU, tau_h=TAU + 0.04, theta_eq=theta_eq,
        device='cuda', max_steps=N_STEPS, output_interval=N_STEPS + 1,
        g_force=(0.0, 0.0, -g_val))

    solver = AllenCahnSolver(config, dtype=torch.float32, stab_mode='fakhari',
        boundary_relax=0.0, geometric_wetting=(amp > 0), geo_amplification=amp,
        ghost_scale=GHOST)

    solid, fraction = create_substrate_with_fraction(nx, ny, nz,
        substrate_type='flat', R_star=1.0, R_d=R_drop)
    solver.set_solid(solid, solid_fraction=fraction)
    solid_np = np.asarray(solid)

    # droplet initially SITTING on the wall (sphere center at z=R_drop, bottom
    # at the wall). No impact velocity: wetting force drives equilibration.
    # (A slow free-fall approach fails: u=0.005 is swamped by spurious currents
    # of order sigma/mu ~ 1.4, so the drop never reaches the wall.)
    # droplet initially EMBEDDED 2 lu into the wall so the contact line is
    # anchored. With zero gravity a non-embedded hydrophobic drop gets pushed
    # off the wall by the wetting BC (observed: zmin=3, floating) and never
    # equilibrates into a sessile cap.
    cz = 1.0 + R_drop - EMBED
    C_init, _, u_init = create_fe_droplet_with_impact(nx, ny, nz,
        center=(nx / 2.0, ny / 2.0, cz), radius=R_drop, xi=XI,
        rho_l=RHO_L, rho_g=RHO_G, u_impact=(0.0, 0.0, 0.0))
    solver.init_fields(C_init, u_init)

    t0 = time.time()
    history = []
    for step in range(1, N_STEPS + 1):
        solver.step()
        if step % 1000 == 0:
            phi = solver.phi.detach().cpu().numpy()
            m = measure_flat_contact(phi, solid_np)
            mask = phi > 0.5
            zs = np.argwhere(mask)
            zmin = zs[:, 2].min() if len(zs) else -1
            lz = phi[:, :, max(zmin - 1, 0)]
            print(f"  [diag {step}] zmin={zmin}, layer_below_max={lz.max():.3f} "
                  f"r_c={m['r_c'] if m else None:.1f} "
                  f"theta={m['theta'] if m and m['theta'] else 0:.1f}",
                  flush=True)
            if m and m['theta'] is not None:
                history.append({'step': step, 'theta': round(m['theta'], 2),
                                'r_c': round(m['r_c'], 2), 'h_max': round(m['h_max'], 2)})
                print(f"  [{step:5d}] theta_eff={m['theta']:.1f}deg r_c={m['r_c']:.1f} "
                      f"h_max={m['h_max']:.1f} V={m['volume']:.0f}", flush=True)
            else:
                print(f"  [{step:5d}] no contact yet (r_c unavailable)", flush=True)

    phi_f = solver.phi.detach().cpu().numpy()
    m_f = measure_flat_contact(phi_f, solid_np)
    theta_final = m_f['theta'] if m_f else None
    mass0 = float(solver.phi_mass_init)
    mass_f = float(((phi_f > 0.5) & ~solid_np).sum())
    drift = (mass_f - mass0) / mass0 * 100.0 if mass0 > 0 else 0.0

    result = {
        'tag': tag, 'theta_eq': theta_eq, 'amp': amp, 'N': N,
        'theta_eff': round(theta_final, 2) if theta_final else None,
        'error_deg': round(abs(theta_final - theta_eq), 2) if theta_final else None,
        'mass_drift_pct': round(drift, 3),
        'elapsed_s': round(time.time() - t0, 1),
        'final_step': N_STEPS, 'history': history[-6:],
    }
    print(f"[{tag}] FINAL theta_eff={result['theta_eff']} "
          f"error={result['error_deg']} mass={result['mass_drift_pct']}% "
          f"({result['elapsed_s']}s)", flush=True)
    return result


if __name__ == '__main__':
    theta = float(sys.argv[1]) if len(sys.argv) > 1 else 162.0
    amp = float(sys.argv[2]) if len(sys.argv) > 2 else 1.5
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 150
    steps = int(sys.argv[4]) if len(sys.argv) > 4 else 10000
    tag = sys.argv[5] if len(sys.argv) > 5 else f"theta{theta:.0f}_amp{amp}"
    embed = float(sys.argv[6]) if len(sys.argv) > 6 else 2.0
    bo = float(sys.argv[7]) if len(sys.argv) > 7 else 3.0
    ghost = float(sys.argv[8]) if len(sys.argv) > 8 else None
    res = run_case(theta, amp, N=n, N_STEPS=steps, tag=tag, EMBED=embed, BO=bo, GHOST=ghost)
    out = 'results/static_contact_validation.json'
    data = []
    if os.path.exists(out):
        data = json.load(open(out))
    data.append(res)
    json.dump(data, open(out, 'w'), indent=1)
    print(f"saved to {out} ({len(data)} cases)")
