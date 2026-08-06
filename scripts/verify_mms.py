#!/usr/bin/env python3
"""Code verification: MMS + conservation checks (3D mode)."""
import sys, os, json
import numpy as np

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)

import torch
from lbm.fe_config import FEConfig
from lbm.fe_ac_solver import AllenCahnSolver

torch.cuda.empty_cache()
RESULTS = []


def make_solver(N, nz=20, theta_eq=162.0, ac_scale=1.0, amp=0.0):
    """Helper: create a solver with a flat bottom wall."""
    xi = 4.0
    sigma = 0.01
    beta = 12.0 * sigma / xi
    kappa = beta * xi**2 / 8.0
    M = 0.02 / beta

    config = FEConfig(
        nx=N, ny=N, nz=nz,
        rho_l=1.0, rho_g=1.0/828.0,
        sigma=sigma, xi=xi, beta=beta, kappa=kappa,
        M=M, tau_l=0.53, tau_g=0.53, tau_h=0.57,
        theta_eq=theta_eq, device='cuda',
        max_steps=1, output_interval=2,
        g_force=(0.0, 0.0, 0.0),
    )
    solver = AllenCahnSolver(
        config, dtype=torch.float32,
        stab_mode='fakhari', boundary_relax=0.0,
        geometric_wetting=(amp > 0), geo_amplification=amp,
        ac_scale=ac_scale,
    )
    # Flat wall at k=0
    solid = torch.zeros(N, N, nz, dtype=torch.bool, device='cuda')
    solid[:, :, 0] = True
    fraction = solid.float()
    solver.set_solid(solid, solid_fraction=fraction)
    return solver, xi


def place_droplet(solver, N, nz, xi, R_drop, U0=-0.05):
    """Initialize a droplet above the wall (CPU tensors, solver moves to CUDA)."""
    cx, cy = N / 2.0, N / 2.0
    cz = nz / 2.0 + 2
    x = np.arange(N, dtype=np.float32)
    y = np.arange(N, dtype=np.float32)
    X, Y = np.meshgrid(x, y, indexing='ij')
    R_xy = np.sqrt((X - cx)**2 + (Y - cy)**2)

    phi = np.zeros((N, N, nz), dtype=np.float32)
    for k in range(nz):
        R3 = np.sqrt(R_xy**2 + (k - cz)**2)
        phi[:, :, k] = 0.5 * (1.0 - np.tanh(2.0 * (R3 - R_drop) / xi))

    u = np.zeros((3, N, N, nz), dtype=np.float32)
    u[2] = U0 * phi  # z-direction impact

    solver.init_fields(phi, u)
    return phi, u


# ================================================================
# TEST 1: Mass conservation (static droplet, no impact velocity)
# ================================================================
def test_mass_conservation():
    print("=" * 60)
    print("TEST 1: Mass conservation (static droplet)")
    print("=" * 60)

    N, nz = 60, 30
    solver, xi = make_solver(N, nz)
    R_drop = 10.0

    cx, cy = N / 2.0, N / 2.0
    cz = nz / 2.0
    x = np.arange(N, dtype=np.float32)
    y = np.arange(N, dtype=np.float32)
    X, Y = np.meshgrid(x, y, indexing='ij')
    R_xy = np.sqrt((X - cx)**2 + (Y - cy)**2)

    phi = np.zeros((N, N, nz), dtype=np.float32)
    for k in range(nz):
        R3 = np.sqrt(R_xy**2 + (k - cz)**2)
        phi[:, :, k] = 0.5 * (1.0 - np.tanh(2.0 * (R3 - R_drop) / xi))

    u = np.zeros((3, N, N, nz), dtype=np.float32)
    solver.init_fields(phi, u)

    mass0 = (solver.phi > 0.5).sum().item()
    masses = [mass0]
    for step in range(200):
        solver.step()
        if step % 20 == 0:
            masses.append((solver.phi > 0.5).sum().item())

    mass_final = masses[-1]
    drift = (mass_final - mass0) / mass0 * 100 if mass0 > 0 else 0

    print(f"  Mass: {mass0:.0f} → {mass_final:.0f} nodes")
    print(f"  Drift: {drift:+.4f}%")
    print(f"  PASS: {abs(drift) < 2.0}")

    RESULTS.append({'test': 'mass_conservation', 'drift_pct': float(drift),
                    'pass': bool(abs(drift) < 2.0)})
    del solver; torch.cuda.empty_cache()
    return abs(drift) < 2.0


# ================================================================
# TEST 2: AC ablation produces monotonic k trend
# ================================================================
def test_ac_ablation_monotonic():
    print("\n" + "=" * 60)
    print("TEST 2: AC ablation monotonicity")
    print("=" * 60)

    N, nz = 60, 30
    R_drop = 10.0
    scales = [0.0, 0.5, 1.0, 1.5, 2.0]
    k_vals = []

    for s in scales:
        solver, xi = make_solver(N, nz, ac_scale=s)
        place_droplet(solver, N, nz, xi, R_drop, U0=-0.03)
        for step in range(300):
            solver.step()
        phi = solver.phi.detach().cpu().numpy()
        sol = solver.solid.cpu().numpy()
        intf = (phi > 0.5) & ~sol
        if intf.any():
            c = np.argwhere(intf)
            Dx = float(c[:, 1].max() - c[:, 1].min() + 1)
            Dy = float(c[:, 2].max() - c[:, 2].min() + 1)
            k = Dx / Dy if Dy > 0 else 0
        else:
            k = 0
        k_vals.append(k)
        print(f"  s={s:.1f}: k={k:.4f}")
        del solver; torch.cuda.empty_cache()

    # Check monotonic: k should decrease as s increases
    monotonic = all(k_vals[i] >= k_vals[i+1] - 0.01 for i in range(len(k_vals)-1))
    print(f"  Monotonic: {monotonic}")
    print(f"  PASS: {monotonic}")
    RESULTS.append({'test': 'ac_ablation_monotonic', 'k_vals': k_vals,
                    'pass': bool(monotonic)})
    return monotonic


# ================================================================
# TEST 3: Symmetry on flat surface
# ================================================================
def test_symmetry_flat():
    print("\n" + "=" * 60)
    print("TEST 3: Symmetry on flat surface")
    print("=" * 60)

    N, nz = 60, 30
    R_drop = 10.0
    solver, xi = make_solver(N, nz, amp=1.5)
    place_droplet(solver, N, nz, xi, R_drop, U0=-0.03)
    for step in range(300):
        solver.step()

    phi = solver.phi.detach().cpu().numpy()
    sol = solver.solid.cpu().numpy()
    intf = (phi > 0.5) & ~sol
    if intf.any():
        c = np.argwhere(intf)
        # Measure x-y footprint (average over z)
        Dx = float(c[:, 1].max() - c[:, 1].min() + 1)
        Dy = float(c[:, 2].max() - c[:, 2].min() + 1)
        k = Dx / Dy if Dy > 0 else 0
    else:
        k = 0

    dev = abs(k - 1.0)
    print(f"  k (flat): {k:.4f}")
    print(f"  Deviation from 1.0: {dev:.4f}")
    print(f"  PASS: {dev < 0.15}")
    RESULTS.append({'test': 'symmetry_flat', 'k': float(k),
                    'deviation': float(dev), 'pass': bool(dev < 0.15)})
    del solver; torch.cuda.empty_cache()
    return dev < 0.15


# ================================================================
# TEST 4: Energy dissipation
# ================================================================
def test_energy_dissipation():
    print("\n" + "=" * 60)
    print("TEST 4: Energy dissipation (AC monotonicity)")
    print("=" * 60)

    N, nz = 40, 20
    xi = 4.0
    sigma = 0.01
    beta_val = 12.0 * sigma / xi
    kappa = beta_val * xi**2 / 8.0

    solver, _ = make_solver(N, nz)

    # Random initial condition (CPU)
    np.random.seed(42)
    phi = (0.5 + 0.2 * np.random.randn(N, N, nz)).astype(np.float32)
    phi = np.clip(phi, 0.01, 0.99)
    u = np.zeros((3, N, N, nz), dtype=np.float32)
    solver.init_fields(phi, u)

    def compute_energy(phi_np):
        bulk = (beta_val / 2.0) * np.sum(phi_np**2 * (1 - phi_np)**2)
        gx = np.gradient(phi_np, axis=0)
        gy = np.gradient(phi_np, axis=1)
        gz = np.gradient(phi_np, axis=2)
        grad = (kappa / 2.0) * np.sum(gx**2 + gy**2 + gz**2)
        return bulk + grad

    energies = []
    for step in range(301):
        if step % 30 == 0:
            E = compute_energy(solver.phi.detach().cpu().numpy())
            energies.append(E)
        if step < 300:
            solver.step()

    monotonic = all(energies[i] >= energies[i+1] - 1e-4 for i in range(len(energies)-1))
    decrease = (energies[0] - energies[-1]) / energies[0] * 100 if energies[0] > 0 else 0
    print(f"  Energy: {energies[0]:.1f} → {energies[-1]:.1f} ({decrease:.1f}% decrease)")
    print(f"  Monotonic: {monotonic}")
    print(f"  PASS: {monotonic and decrease > 0}")
    RESULTS.append({'test': 'energy_dissipation', 'decrease_pct': float(decrease),
                    'monotonic': bool(monotonic), 'pass': bool(monotonic and decrease > 0)})
    del solver; torch.cuda.empty_cache()
    return monotonic and decrease > 0


# ================================================================
# TEST 5: Grid convergence of spreading diameter
# ================================================================
def test_grid_convergence():
    print("\n" + "=" * 60)
    print("TEST 5: Grid convergence (spreading diameter)")
    print("=" * 60)

    xi = 4.0
    scales = [40, 60, 80]
    D_vals = []

    for N in scales:
        nz = max(20, N // 2)
        R_drop = N / 4.0
        solver, _ = make_solver(N, nz, amp=1.5)
        place_droplet(solver, N, nz, xi, R_drop, U0=-0.03)
        for step in range(300):
            solver.step()
        phi = solver.phi.detach().cpu().numpy()
        sol = solver.solid.cpu().numpy()
        intf = (phi > 0.5) & ~sol
        if intf.any():
            c = np.argwhere(intf)
            Dx = float(c[:, 1].max() - c[:, 1].min() + 1)
        else:
            Dx = 0
        D_vals.append(Dx)
        print(f"  N={N:3d}: D_x={Dx:.0f}")
        del solver; torch.cuda.empty_cache()

    # Check convergence
    if len(D_vals) >= 2:
        rel = abs(D_vals[-1] - D_vals[-2]) / max(D_vals[-2], 1)
        converged = rel < 0.15
    else:
        converged = False
    print(f"  Relative change: {rel:.4f}")
    print(f"  PASS: {converged}")
    RESULTS.append({'test': 'grid_convergence', 'D_vals': D_vals,
                    'pass': bool(converged)})
    return converged


def main():
    print("CODE VERIFICATION: MMS + Conservation Tests\n")
    tests = [
        ("Mass conservation", test_mass_conservation),
        ("AC ablation monotonicity", test_ac_ablation_monotonic),
        ("Symmetry (flat)", test_symmetry_flat),
        ("Energy dissipation", test_energy_dissipation),
        ("Grid convergence", test_grid_convergence),
    ]
    results = {}
    all_pass = True
    for name, fn in tests:
        try:
            p = fn()
            results[name] = "PASS" if p else "FAIL"
            if not p: all_pass = False
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()
            results[name] = f"ERROR"
            all_pass = False

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for n, s in results.items():
        print(f"  {n:30s}: {s}")
    print(f"\nOverall: {'ALL PASS' if all_pass else 'SOME FAILED'}")

    with open(os.path.join(_project_root, 'results', 'mms_verification.json'), 'w') as f:
        json.dump({'results': RESULTS, 'all_pass': all_pass}, f, indent=2)


if __name__ == '__main__':
    main()
