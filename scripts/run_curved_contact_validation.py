"""Curved-surface (R*=1.0 ridge) contact angle validation.

Two modes:
  static   -- sessile drop on ridge with gravity (Bo) to seat it,
              measure equilibrium contact angle after relaxation
  dynamic  -- droplet impact (We=7.9) reproducing the paper's tab:ca_error
              geometric-construction measurement (13 deg -> 3 deg claim)

Measurement = paper's method (run_static_droplet.measure_contact_angle):
  - contact line: column (at y=ny/2) where phi crosses 0.5 in the first
    fluid layer above the wall, scanning +x from ridge top
  - angle = atan2(gx, -gz) with g = grad(phi) at the crossing (deg, 0-180)

Usage: python scripts/run_curved_contact_validation.py static|dynamic theta amp N steps [BO]
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
U0_REF = 0.05


def get_surface_height(solid, nx, ny, nz):
    solid = np.asarray(solid)
    height = np.zeros((nx, ny), dtype=int)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz - 1, -1, -1):
                if solid[i, j, k]:
                    height[i, j] = k
                    break
    return height


def measure_curved_angle(phi, sh, N, nz, cx, verbose=False):
    """Contact angle on the ridge (scan +x from ridge top).

    Contact line = OUTERMOST phi>0.5 column in the lowest liquid layer (zmin),
    NOT the first column with a vertical 0.5 crossing (that catches the drop
    bottom interface at the ridge top). Angle = atan2(gx, -gz) at the phi=0.5
    crossing of the contact-line column, scanned upward from zmin.
    """
    mask = phi > 0.5
    zs = np.argwhere(mask)
    if len(zs) == 0:
        return None
    zmin = int(zs[:, 2].min())
    layer = mask[:, N // 2, zmin]
    # rightmost liquid column in the zmin layer
    liquid_cols = np.argwhere(layer).flatten()
    if len(liquid_cols) == 0:
        return None
    ix_cl = int(liquid_cols.max())
    # walk up column ix_cl to find the phi=0.5 crossing (liquid surface)
    col = phi[ix_cl, N // 2, :]
    for fz in range(zmin, nz - 1):
        v = col[fz]
        va = col[fz + 1]
        if v > 0.5 >= va:
            gx = (phi[min(ix_cl + 1, N - 1), N // 2, fz]
                  - phi[max(ix_cl - 1, 0), N // 2, fz]) / 2.0
            gz = (col[fz + 1] - col[max(fz - 1, 0)]) / 2.0
            angle = math.degrees(abs(math.atan2(gx, -gz)))
            return ix_cl, angle
    return None


def setup(theta_eq, amp, N, BO=0.0, N_STEPS=4000, u_impact=(0.0, 0.0, -0.05),
          gap=2.0, embed=0.0):
    """Shared setup. Returns (solver, solid_np, sh, cx, nz)."""
    torch.cuda.empty_cache()
    sigma = RHO_L * U0_REF**2 * D0 / 7.9
    beta = 12.0 * sigma / XI
    kappa = beta * XI**2 / 8.0
    M = 0.02 / beta
    g_val = BO * sigma / (RHO_L * D0**2) if BO > 0 else 0.0

    R_g = 1.0 * R_drop
    nz = min(int(R_g + 2 + R_drop + 2 * R_drop + 15), 300)

    cfg = FEConfig(nx=N, ny=N, nz=nz, rho_l=RHO_L, rho_g=RHO_G,
        sigma=sigma, xi=XI, beta=beta, kappa=kappa, M=M,
        tau_l=TAU, tau_g=TAU, tau_h=TAU + 0.04, theta_eq=theta_eq,
        device='cuda', max_steps=N_STEPS, output_interval=N_STEPS + 1,
        g_force=(0.0, 0.0, -g_val))
    solver = AllenCahnSolver(cfg, dtype=torch.float32, stab_mode='fakhari',
        boundary_relax=0.0, geometric_wetting=(amp > 0), geo_amplification=amp)

    solid, frac = create_substrate_with_fraction(N, N, nz,
        substrate_type='ridge', R_star=1.0, R_d=R_drop)
    solver.set_solid(solid, solid_fraction=frac)
    solid_np = np.asarray(solid)
    sh = get_surface_height(solid_np, N, N, nz)
    cx = N / 2.0
    ridge_top = sh[N // 2, N // 2]
    cz = ridge_top + gap + R_drop - embed

    C, _, u = create_fe_droplet_with_impact(N, N, nz,
        center=(cx, N / 2.0, cz), radius=R_drop, xi=XI,
        rho_l=RHO_L, rho_g=RHO_G, u_impact=u_impact)
    solver.init_fields(C, u)
    return solver, solid_np, sh, cx, nz


def run_static(theta_eq, amp, N=150, N_STEPS=12000, BO=3.0, CONV=1):
    """Sessile drop on ridge with gravity. Measure angle during relaxation.

    CONV=1: early stop when angle and contact line are stable for 3 samples
    (relaxation time ~ rho*R^2/mu ~ 5e4 steps at production params).
    """
    solver, solid_np, sh, cx, nz = setup(theta_eq, amp, N, BO=BO,
        N_STEPS=N_STEPS, u_impact=(0.0, 0.0, 0.0), gap=2.0, embed=2.0)
    t0 = time.time()
    meas = []
    stable = 0
    final_step = N_STEPS
    for step in range(1, N_STEPS + 1):
        solver.step()
        if step % 2000 == 0:
            phi = solver.phi.detach().cpu().numpy()
            res = measure_curved_angle(phi, sh, N, nz, cx)
            if res:
                ix, ang = res
                # wall tangent angle at contact line (curved ridge surface)
                if 0 < ix < N - 1:
                    wt = math.degrees(math.atan(
                        (sh[ix + 1, N // 2] - sh[ix - 1, N // 2]) / 2.0))
                else:
                    wt = 0.0
                meas.append({'step': step, 'ix': ix, 'angle': round(ang, 1),
                             'wall_tilt': round(wt, 1)})
                print(f"  [{step:5d}] contact ix={ix} angle={ang:.1f}deg "
                      f"wall_tilt={wt:.1f}deg", flush=True)
                if CONV and len(meas) >= 3:
                    a = [m['angle'] for m in meas[-3:]]
                    i = [m['ix'] for m in meas[-3:]]
                    if max(a) - min(a) < 0.5 and max(i) - min(i) <= 2:
                        stable += 1
                        if stable >= 3:
                            final_step = step
                            print(f"  [conv] stable at step {step}", flush=True)
                            break
                    else:
                        stable = 0
            else:
                print(f"  [{step:5d}] no contact line", flush=True)
    phi_f = solver.phi.detach().cpu().numpy()
    res_f = measure_curved_angle(phi_f, sh, N, nz, cx)
    # late-time average (last half of measurements)
    late = [m['angle'] for m in meas[len(meas) // 2:]] if meas else []
    angle_final = res_f[1] if res_f else None
    angle_avg = round(float(np.mean(late)), 1) if late else None
    mass0 = float(solver.phi_mass_init)
    mass_f = float(((phi_f > 0.5) & ~np.asarray(solid_np)).sum())
    drift = (mass_f - mass0) / mass0 * 100.0 if mass0 > 0 else 0.0
    result = {'mode': 'static', 'theta_eq': theta_eq, 'amp': amp, 'N': N,
              'BO': BO, 'angle_final': angle_final,
              'angle_late_avg': angle_avg,
              'error_deg': round(abs(angle_final - theta_eq), 1) if angle_final else None,
              'mass_drift_pct': round(drift, 2),
              'elapsed_s': round(time.time() - t0, 1),
              'final_step': final_step,
              'measurements': meas[-10:]}
    print(f"[static] FINAL angle={angle_final} late_avg={angle_avg} "
          f"error={result['error_deg']} mass={drift:.2f}% "
          f"steps={final_step} t={result['elapsed_s']}s", flush=True)
    return result


def run_dynamic(theta_eq, amp, N=150, N_STEPS=2500):
    """Droplet impact on ridge; measure contact angle during spreading."""
    solver, solid_np, sh, cx, nz = setup(theta_eq, amp, N, BO=0.0,
        N_STEPS=N_STEPS, u_impact=(0.0, 0.0, -0.05), gap=2.0)
    t0 = time.time()
    meas = []
    for step in range(1, N_STEPS + 1):
        solver.step()
        if step % 250 == 0:
            phi = solver.phi.detach().cpu().numpy()
            res = measure_curved_angle(phi, sh, N, nz, cx)
            if res:
                ix, ang = res
                meas.append({'step': step, 'ix': ix, 'angle': round(ang, 1)})
                print(f"  [{step:5d}] contact ix={ix} angle={ang:.1f}deg", flush=True)
    phi_f = solver.phi.detach().cpu().numpy()
    res_f = measure_curved_angle(phi_f, sh, N, nz, cx)
    # peak spreading window: angle when contact line is furthest from center
    if meas:
        ix_max = max(m['ix'] for m in meas)
        peak = [m for m in meas if m['ix'] == ix_max]
        angle_peak = round(float(np.mean([m['angle'] for m in peak])), 1)
    else:
        angle_peak = None
    mass0 = float(solver.phi_mass_init)
    mass_f = float(((phi_f > 0.5) & ~np.asarray(solid_np)).sum())
    drift = (mass_f - mass0) / mass0 * 100.0 if mass0 > 0 else 0.0
    result = {'mode': 'dynamic', 'theta_eq': theta_eq, 'amp': amp, 'N': N,
              'BO': 0.0, 'angle_final': res_f[1] if res_f else None,
              'angle_peak_spread': angle_peak,
              'error_deg': round(abs((res_f[1] if res_f else 0) - theta_eq), 1) if res_f else None,
              'mass_drift_pct': round(drift, 2),
              'elapsed_s': round(time.time() - t0, 1),
              'measurements': meas[-10:]}
    print(f"[dynamic] FINAL angle={result['angle_final']} peak_spread={angle_peak} "
          f"mass={drift:.2f}%", flush=True)
    return result


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'static'
    theta = float(sys.argv[2]) if len(sys.argv) > 2 else 162.0
    amp = float(sys.argv[3]) if len(sys.argv) > 3 else 1.5
    n = int(sys.argv[4]) if len(sys.argv) > 4 else 150
    steps = int(sys.argv[5]) if len(sys.argv) > 5 else 12000
    bo = float(sys.argv[6]) if len(sys.argv) > 6 else 3.0
    conv = int(sys.argv[7]) if len(sys.argv) > 7 else 1
    if mode == 'dynamic':
        res = run_dynamic(theta, amp, N=n, N_STEPS=steps)
    else:
        res = run_static(theta, amp, N=n, N_STEPS=steps, BO=bo, CONV=conv)
    out = 'results/curved_contact_validation.json'
    data = []
    if os.path.exists(out):
        data = json.load(open(out))
    data.append(res)
    json.dump(data, open(out, 'w'), indent=1)
    print(f"saved to {out} ({len(data)} cases)")
