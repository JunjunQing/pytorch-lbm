#!/usr/bin/env python3
"""Validation benchmark for AC-LBM solver optimizations.

Runs a quick reference case and checks key metrics against baseline.
Use after each optimization to verify correctness is preserved.

Reference case: R*=1.0, We=7.9, N=80^3, 500 steps
Baseline values measured from unoptimized solver.
"""
import sys, os, numpy as np, torch, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lbm.fe_config import FEConfig
from lbm.fe_ac_solver import AllenCahnSolver
from lbm.fe_droplet import create_fe_droplet_with_impact
from geometry.substrate import create_substrate_with_fraction

# ── Reference parameters ──
D0 = 24.0; R_drop = D0 / 2.0; rho_l = 1.0; rho_g = 1.0 / 828.0
xi = 4.0; U0 = -0.05; We = 7.9; theta_eq = 162.0; tau = 0.555
R_star = 1.0; n = 80; N = 500

# ── Baseline values (from unoptimized solver) ──
BASELINE = {
    'k_at_500': 1.2963,      # k at step 500
    'z_com_at_500': 17.6,    # z_com at step 500
    'Dx_at_500': 35.0,       # Dx at step 500
    'Dy_at_500': 27.0,       # Dy at step 500
    'mass_rel_change': 1e-4,  # max acceptable relative mass change
    'phi_range': (0.0, 1.0),  # phi must stay in [0,1]
}

TOLERANCE = {
    'k': 0.05,       # ±0.05
    'z_com': 1.0,    # ±1.0 lu
    'Dx': 2.0,       # ±2 lu
    'Dy': 2.0,       # ±2 lu
}


def build_config():
    sigma = rho_l * U0 ** 2 * D0 / We
    beta = 12.0 * sigma / xi
    kappa = beta * xi ** 2 / 8.0
    M = 0.02 / beta
    nz = int(R_drop + 2 + R_drop + 2 * R_drop + 15)
    nz = min(nz, 300)
    return FEConfig(
        nx=n, ny=n, nz=nz,
        rho_l=rho_l, rho_g=rho_g,
        sigma=sigma, xi=xi, beta=beta, kappa=kappa, M=M,
        tau_l=tau, tau_g=tau, tau_h=tau + 0.04,
        theta_eq=theta_eq, device='cuda',
        max_steps=N, output_interval=N + 1,
        g_force=(0.0, 0.0, 0.0),
    )


def setup_solver(config):
    solver = AllenCahnSolver(
        config, dtype=torch.float32,
        stab_mode='fakhari', boundary_relax=0.0,
        geometric_wetting=True, geo_amplification=1.8,
    )
    solid, fraction = create_substrate_with_fraction(
        n, n, config.nz, substrate_type='ridge',
        R_star=R_star, R_d=R_drop)
    solver.set_solid(solid, solid_fraction=fraction)

    height = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            for k in range(config.nz - 1, -1, -1):
                if solid[i, j, k]:
                    height[i, j] = k
                    break
    cz = height[n // 2, n // 2] + 2.0 + R_drop
    cx, cy = n / 2.0, n / 2.0

    C_init, _, u_init = create_fe_droplet_with_impact(
        n, n, config.nz, center=(cx, cy, cz),
        radius=R_drop, xi=xi,
        rho_l=rho_l, rho_g=rho_g, u_impact=(0.0, 0.0, U0))
    solver.init_fields(C_init, u_init)
    return solver


def validate():
    config = build_config()
    solver = setup_solver(config)

    # Run and track
    mass_init = solver.phi.sum().item()
    results = {}

    torch.cuda.synchronize()
    t0 = time.time()

    for step in range(1, N + 1):
        solver.step()

        if step % 50 == 0:
            phi_np = solver.phi.detach().cpu().numpy()
            solid_np = solver.solid.cpu().numpy()
            interface = (phi_np > 0.5) & ~solid_np

            if interface.any():
                coords = np.argwhere(interface)
                Dx = float(coords[:, 0].max() - coords[:, 0].min() + 1)
                Dy = float(coords[:, 1].max() - coords[:, 1].min() + 1)
                k = Dx / Dy if Dy > 0 else 0
                z_com = float(coords[:, 2].mean())
            else:
                Dx = Dy = k = z_com = 0

            mass_now = solver.phi.sum().item()
            mass_change = abs(mass_now - mass_init) / max(mass_init, 1e-10)

            results[step] = {
                'k': k, 'Dx': Dx, 'Dy': Dy,
                'z_com': z_com, 'mass_change': mass_change,
            }

    torch.cuda.synchronize()
    elapsed = time.time() - t0
    ms_per_step = elapsed / N * 1000
    mem_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

    # ── Checks ──
    r = results.get(N, {})
    errors = []

    # 1. k check
    k_got = r.get('k', 0)
    k_ref = BASELINE['k_at_500']
    if abs(k_got - k_ref) > TOLERANCE['k']:
        errors.append(f"k: got {k_got:.4f}, expected {k_ref:.4f} ± {TOLERANCE['k']}")

    # 2. z_com check
    z_got = r.get('z_com', 0)
    z_ref = BASELINE['z_com_at_500']
    if abs(z_got - z_ref) > TOLERANCE['z_com']:
        errors.append(f"z_com: got {z_got:.1f}, expected {z_ref:.1f} ± {TOLERANCE['z_com']}")

    # 3. Dx check
    dx_got = r.get('Dx', 0)
    dx_ref = BASELINE['Dx_at_500']
    if abs(dx_got - dx_ref) > TOLERANCE['Dx']:
        errors.append(f"Dx: got {dx_got:.0f}, expected {dx_ref:.0f} ± {TOLERANCE['Dx']}")

    # 4. Dy check
    dy_got = r.get('Dy', 0)
    dy_ref = BASELINE['Dy_at_500']
    if abs(dy_got - dy_ref) > TOLERANCE['Dy']:
        errors.append(f"Dy: got {dy_got:.0f}, expected {dy_ref:.0f} ± {TOLERANCE['Dy']}")

    # 5. Mass conservation
    mc = r.get('mass_change', 1)
    if mc > BASELINE['mass_rel_change']:
        errors.append(f"mass change: {mc:.2e} > {BASELINE['mass_rel_change']:.2e}")

    # 6. Phi range
    phi_min = float(solver.phi.min())
    phi_max = float(solver.phi.max())
    if phi_min < -0.01 or phi_max > 1.01:
        errors.append(f"phi out of range: [{phi_min:.4f}, {phi_max:.4f}]")

    # 7. No NaN
    if torch.isnan(solver.phi).any():
        errors.append("NaN detected in phi")

    # ── Report ──
    passed = len(errors) == 0
    status = "PASS" if passed else "FAIL"

    print(f"\n{'='*50}")
    print(f"  Validation: {status}")
    print(f"{'='*50}")
    print(f"  k={k_got:.4f} (ref={k_ref:.4f})")
    print(f"  z_com={z_got:.1f} (ref={z_ref:.1f})")
    print(f"  Dx={dx_got:.0f} (ref={dx_ref:.0f}), Dy={dy_got:.0f} (ref={dy_ref:.0f})")
    print(f"  mass_change={mc:.2e}")
    print(f"  phi=[{phi_min:.4f}, {phi_max:.4f}]")
    print(f"  Performance: {ms_per_step:.1f} ms/step, {mem_mb:.0f} MB")
    print(f"  Wall time: {elapsed:.1f}s for {N} steps")

    if errors:
        print(f"\n  ERRORS:")
        for e in errors:
            print(f"    - {e}")
    print()

    del solver
    torch.cuda.empty_cache()
    return passed, ms_per_step, mem_mb


if __name__ == '__main__':
    passed, ms, mem = validate()
    sys.exit(0 if passed else 1)
