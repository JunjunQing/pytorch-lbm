"""Conservative Allen-Cahn LBM for high density ratio two-phase flows.

Implements Fakhari et al. (2017) pressure-velocity formulation with
Liang et al. (2018) Allen-Cahn interface tracking:

  - f-distribution: conservative Allen-Cahn equation (interface tracking)
  - g-distribution: pressure-velocity NS (Fakhari 2017 form WITHOUT rho)

The g-equilibrium does NOT multiply velocity terms by rho:
    g_eq_i = P * w_i + s_i(u)           [Fakhari]
This means sum(c_i * g_eq_i) = u (not rho*u).

Velocity recovery:  u = sum(c_i*g_i) + 0.5 * F / rho_eff
  where rho_eff = max(rho, rho_floor) prevents 1/rho_gas blowup.

Forces (Fakhari 2017):
  F_s  = mu_phi * grad(phi)              [surface tension]
  Fp   = -P * delta_rho * cs^2 * grad(phi) [pressure-density correction]
  Fm   = (0.5-tau)/tau * sigma_neq . grad(phi) * delta_rho [viscous correction]

Algorithm follows Fakhari 2017 Fortran reference code ordering:
  1. phi = sum(f)
  2. Compute density, chemical potential, gradients
  3. P = sum(g) [NEW pressure, post-streaming]
  4. Compute g_neq, stress tensor with NEW P, OLD u
  5. Compute total force F = F_s + Fp + Fm + F_body
  6. Recover velocity: u = sum(c*g) + 0.5*F/rho_eff
  7. Allen-Cahn source term (sharpening + conservative d(phi*u)/dt)
  8. Collision f and g (Guo forcing)
  9. Streaming
  10. Bounce-back
  11. Mass conservation
  12. Boundary phi correction for contact angle

References:
  Fakhari, Mitchell, Leonardi, Bolster (2017), PRE 96, 053301
  Liang et al. (2018), PRE 97, 033309
  Liang, Liu, Chai, Shi (2019), PRE 99, 063306 (wetting extension)
"""
import torch
import numpy as np
import math

from .lattice import D2Q9, D3Q19
from .fe_config import FEConfig
from .fe_chemical_potential import compute_density
from .boundary import BounceBack
from .fe_wetting import ConningtonLeeWetting


class AllenCahnSolver:
    """Conservative Allen-Cahn + pressure-velocity NS LBM."""

    def __init__(self, config: FEConfig, dtype=torch.float64, stab_mode='auto',
                 mu_drive=0.0, boundary_relax=0.3, interface_mode='allen-cahn',
                 ghost_scale=None, sharpen_reduce=0.8, mu_wall_scale=1.0,
                 geometric_wetting=False, direct_wetting=False,
                 ch_wall_blend=0.0, geo_amplification=1.0, ac_scale=1.0):
        self.config = config
        # Support non-standard device objects (e.g. XPU string, custom devices)
        # which may not be torch.device instances.
        if isinstance(config.device, torch.device):
            self.device = config.device
        else:
            try:
                self.device = torch.device(config.device)
            except (TypeError, RuntimeError):
                # DirectML or other non-standard device — use as-is
                self.device = config.device
        self.dtype = dtype
        self.step_count = 0
        self.t = 0.0
        self.stab_mode = stab_mode
        self.mu_drive = mu_drive
        self.boundary_relax = boundary_relax
        self.interface_mode = interface_mode  # 'allen-cahn' or 'cahn-hilliard'
        self._ghost_scale = ghost_scale
        self._sharpen_reduce = sharpen_reduce
        self._mu_wall_scale = mu_wall_scale
        self.geometric_wetting = geometric_wetting
        self.direct_wetting = direct_wetting
        self.ch_wall_blend = ch_wall_blend  # AC/CH blending near wall
        self._geo_amplification = geo_amplification
        self._ac_scale = ac_scale

        cfg = config
        shape = cfg.grid_shape
        self.shape = shape
        self.ndim = cfg.dim

        self.lattice = D2Q9() if cfg.dim == 2 else D3Q19()
        self.cs2 = self.lattice.cs2
        Q = self.lattice.q

        self._rho_floor = 0.5 * cfg.rho_l

        self.f = torch.zeros((Q,) + shape, dtype=dtype, device=self.device)
        self.g = torch.zeros((Q,) + shape, dtype=dtype, device=self.device)

        self.phi = torch.zeros(shape, dtype=dtype, device=self.device)
        self.rho = torch.zeros(shape, dtype=dtype, device=self.device)
        self.u = torch.zeros((cfg.dim,) + shape, dtype=dtype, device=self.device)
        self.P = torch.zeros(shape, dtype=dtype, device=self.device)
        self.p = torch.zeros(shape, dtype=dtype, device=self.device)
        self.mu_phi = torch.zeros(shape, dtype=dtype, device=self.device)

        self.solid = torch.zeros(shape, dtype=torch.bool, device=self.device)
        self.phi_mass_init = None
        self.bounce_back = None
        self.wetting = None

        # Volume penalization fields
        self.solid_fraction = None  # float tensor (nx, ny, nz)
        self._partial_mask = None   # bool: 0 < solid_fraction < 1 AND NOT solid
        self._partial_eps = None    # float: solid_fraction at partial nodes
        self._has_partial = False

        self._e = self.lattice.e.to(device=self.device, dtype=dtype)
        self._w = self.lattice.w.to(device=self.device, dtype=dtype)
        self._opp = self.lattice.opp.to(self.device)
        self._opp_list = self._opp.tolist()
        self._e_int = self.lattice.e.long().to(self.device)
        self._w_view = self._w.view(-1, *(1 for _ in shape))

        self._e_outer = self._e.unsqueeze(2) * self._e.unsqueeze(1)
        self._e_sq = (self._e * self._e).sum(dim=1)

        # Pre-compute streaming shift tuples: list of (shift_per_dim,) for each direction
        # Avoids per-step Python loop over dimensions inside streaming
        self._stream_shifts = []
        for i in range(self.lattice.q):
            e_i = self._e_int[i]
            shifts = tuple(int(e_i[d].item()) for d in range(self.ndim))
            self._stream_shifts.append(shifts)

        # Pre-compute grouped streaming: directions sharing the same shift pattern
        # are grouped so torch.roll is applied once per group instead of per-direction.
        # Key: tuple of (dim, shift) pairs. Value: list of direction indices.
        from collections import defaultdict
        shift_groups = defaultdict(list)
        for i in range(self.lattice.q):
            # Encode non-zero shifts as a hashable key
            key = tuple((d, s) for d, s in enumerate(self._stream_shifts[i]) if s != 0)
            shift_groups[key].append(i)
        # Convert to list of (key, [indices]) for fast iteration
        self._stream_groups = [(k, v) for k, v in shift_groups.items()
                               if len(v) > 1]
        # Map: direction -> True if it belongs to a group (for skip check)
        self._stream_group_map = [False] * self.lattice.q
        for _, dirs in self._stream_groups:
            for i in dirs:
                self._stream_group_map[i] = True

        self._tau_f = cfg.tau_h if cfg.tau_h > 0 else cfg.tau_l
        self._tau_g = cfg.tau_g
        self._mobility = self.cs2 * (self._tau_f - 0.5)

        if stab_mode == 'auto':
            self.stab_mode = 'fakhari'

        # Precompute boundary phi target from contact angle
        # Geometric approximation: phi_wall ~ 0.5*(1 - cos(pi - theta))
        # This is a empirical relation that works for extreme angles where
        # the free-energy gradient condition has no solution.
        self._phi_wall_target = None
        if cfg.theta_eq != 90.0:
            # First try the analytical tanh profile solution
            cos_theta = math.cos(math.radians(cfg.theta_eq))
            rhs = -cos_theta / 3.0
            disc = 1.0 - 4.0 * rhs
            if disc >= 0:
                self._phi_wall_target = 0.5 * (1.0 - math.sqrt(disc))
            else:
                # Fallback: geometric approximation for extreme angles
                # phi_wall = 0.5*(1 - cos(pi - theta))
                complement_angle = math.pi - math.radians(cfg.theta_eq)
                self._phi_wall_target = 0.5 * (1.0 - math.cos(complement_angle))

        print(f"AllenCahn Solver: grid={shape}, {self.device}, {dtype}")
        print(f"  rho_l={cfg.rho_l}, rho_g={cfg.rho_g}, ratio={cfg.rho_l/cfg.rho_g:.0f}:1")
        print(f"  tau_f={self._tau_f:.4f} (M={self._mobility:.4f}), tau_g={self._tau_g:.4f}")
        print(f"  beta={cfg.beta:.6f}, kappa={cfg.kappa:.6f}, sigma={cfg.sigma:.6f}")
        print(f"  stab_mode={self.stab_mode}, interface={self.interface_mode}, mu_drive={self.mu_drive}")
        if self.geometric_wetting:
            print(f"  geometric_wetting=True (Zhang 2023 gradient correction)")
        if self.direct_wetting:
            print(f"  direct_wetting=True (Jiang 2024 cubic wall energy)")
        if self.ch_wall_blend > 0:
            print(f"  ch_wall_blend={self.ch_wall_blend:.2f} (AC→CH near wall)")
        if self._phi_wall_target is not None:
            print(f"  boundary_relax={self.boundary_relax}, phi_wall_target={self._phi_wall_target:.4f}")

    def set_solid(self, solid_mask: np.ndarray, solid_fraction: np.ndarray = None):
        self.solid = torch.tensor(solid_mask, dtype=torch.bool, device=self.device)
        self.bounce_back = BounceBack(self.lattice, self.solid, device=self.device)

        # Cached CPU-side flags/indices: solid.any() and solid.nonzero() are
        # device->host syncs that would fire every step; hoist them here so
        # step() stays synchronization-free (and CUDA-graph capturable).
        self._solid_any = bool(self.solid.any().item())
        self._solid_idx = (self.solid.nonzero(as_tuple=False)
                           if self._solid_any else None)

        # Volume penalization: compute partial-solid nodes from fraction field
        if solid_fraction is not None:
            self.solid_fraction = torch.tensor(
                solid_fraction, dtype=self.dtype, device=self.device)
            # Partial solid: fraction > 0 but NOT in solid mask
            partial_np = (solid_fraction > 0.01) & (solid_fraction < 0.99) & ~solid_mask
            self._partial_mask = torch.tensor(
                partial_np, dtype=torch.bool, device=self.device)
            self._partial_eps = self.solid_fraction[self._partial_mask].to(self.dtype)
            self._has_partial = self._partial_mask.any().item()
            if self._has_partial:
                n_partial = self._partial_mask.sum().item()
                print(f"  Volume penalization: {n_partial} partial-solid nodes "
                      f"(eps range [{self._partial_eps.min():.3f}, "
                      f"{self._partial_eps.max():.3f}])")
        else:
            self._has_partial = False

        cfg = self.config
        if cfg.theta_eq != 90.0 and cfg.kappa > 0:
            mode = 'linear' if cfg.theta_eq > 130.0 or cfg.theta_eq < 50.0 else 'quadratic'
            # Convert solid_fraction to numpy for wetting BC precomputation
            sf_np = solid_fraction.cpu().numpy() if torch.is_tensor(solid_fraction) else solid_fraction
            self.wetting = ConningtonLeeWetting(
                self.solid, self.lattice, cfg.phi_c, cfg.kappa,
                sigma=cfg.sigma, theta_deg=cfg.theta_eq,
                device=self.device, mode=mode,
                ghost_scale=self._ghost_scale,
                geo_amplification=self._geo_amplification,
                solid_fraction=sf_np
            )
            print(f"  Wetting BC: theta_eq={cfg.theta_eq:.1f}deg, mode={mode}, "
                  f"geo_amp={self._geo_amplification}")
        print(f"  Solid nodes: {self.solid.sum().item()}")

    def init_fields(self, phi_init: np.ndarray, u_init: np.ndarray = None):
        cfg = self.config
        dt = self.dtype
        phi_t = torch.tensor(phi_init, dtype=dt, device=self.device)

        if u_init is not None:
            u_t = torch.tensor(u_init, dtype=dt, device=self.device)
        else:
            u_t = torch.zeros((cfg.dim,) + cfg.grid_shape, dtype=dt, device=self.device)

        self.phi = torch.clamp(phi_t, 0.0, 1.0)
        self.rho = compute_density(self.phi, cfg.rho_l, cfg.rho_g)
        self.u = u_t
        self.P = torch.zeros_like(self.phi)
        self.p = self.rho * self.cs2
        self.phi_mass_init = self.phi.sum().item()

        self._compute_mu()
        self.f = self._f_equilibrium(self.phi, self.u)
        self.g = self._g_equilibrium(self.P, self.u)

        print(f"  Init: phi=[{self.phi.min():.3f}, {self.phi.max():.3f}], "
              f"rho=[{self.rho.min():.3f}, {self.rho.max():.3f}]")

    # ------------------------------------------------------------------
    # Gradient operators
    # ------------------------------------------------------------------

    def _lattice_gradient(self, phi):
        Q = self.lattice.q
        grad = torch.zeros((self.ndim,) + self.shape, dtype=self.dtype, device=self.device)
        for i in range(1, Q):
            phi_shifted = phi
            shifts = self._stream_shifts[i]
            for d in range(self.ndim):
                if shifts[d] != 0:
                    phi_shifted = torch.roll(phi_shifted, shifts=-shifts[d], dims=d)
            if self.wetting is not None:
                phi_shifted = self.wetting.correct_shifted(phi, phi_shifted, i)
            for d in range(self.ndim):
                grad[d] += self._w[i] * self._e[i, d] * phi_shifted / self.cs2
        return grad

    def _lattice_laplacian(self, phi):
        Q = self.lattice.q
        result = torch.zeros_like(phi)
        for i in range(1, Q):
            phi_shifted = phi
            shifts = self._stream_shifts[i]
            for d in range(self.ndim):
                if shifts[d] != 0:
                    phi_shifted = torch.roll(phi_shifted, shifts=-shifts[d], dims=d)
            if self.wetting is not None:
                phi_shifted = self.wetting.correct_shifted(phi, phi_shifted, i)
            result += 2.0 * self._w[i] * (phi_shifted - phi) / self.cs2
        return result

    def _lattice_gradient_and_laplacian(self, phi):
        """Fused gradient + laplacian in a single pass over lattice directions.

        Saves ~40% vs calling _lattice_gradient and _lattice_laplacian separately,
        because the expensive torch.roll and wetting correction are done once per
        direction instead of twice.
        """
        Q = self.lattice.q
        grad = torch.zeros((self.ndim,) + self.shape, dtype=self.dtype, device=self.device)
        lap = torch.zeros_like(phi)
        w_over_cs2 = self._w / self.cs2
        for i in range(1, Q):
            phi_shifted = phi
            shifts = self._stream_shifts[i]
            for d in range(self.ndim):
                if shifts[d] != 0:
                    phi_shifted = torch.roll(phi_shifted, shifts=-shifts[d], dims=d)
            if self.wetting is not None:
                phi_shifted = self.wetting.correct_shifted(phi, phi_shifted, i)
            diff = phi_shifted - phi
            wi = w_over_cs2[i]
            for d in range(self.ndim):
                grad[d] += wi * self._e[i, d] * phi_shifted
            lap += 2.0 * wi * diff
        return grad, lap

    # ------------------------------------------------------------------
    # Chemical potential
    # ------------------------------------------------------------------

    def _compute_mu(self, lap_phi=None):
        cfg = self.config
        phi = self.phi
        if lap_phi is None:
            lap_phi = self._lattice_laplacian(phi)
        self.mu_phi = (4.0 * cfg.beta * phi * (phi - 1.0) * (phi - 0.5)
                       - cfg.kappa * lap_phi)

        if self.wetting is not None:
            mask = self.wetting.is_boundary
            if self.wetting.mode == 'linear':
                self.mu_phi[mask] += self.wetting._mu_wall * self._mu_wall_scale
            else:
                C_bnd = phi[mask]
                self.mu_phi[mask] += cfg.phi_c * C_bnd * (1.0 - C_bnd)

    # ------------------------------------------------------------------
    # Equilibrium distributions
    # ------------------------------------------------------------------

    def _f_equilibrium(self, phi, u, grad_phi=None):
        eu = torch.einsum('qd,d...->q...', self._e, u)
        u_sq = (u * u).sum(dim=0)
        w_view = self._w_view
        eu_cs2 = eu / self.cs2
        eu_cs2_sq = eu_cs2 * eu_cs2
        del eu
        Gamma = w_view * (1.0 + eu_cs2 + eu_cs2_sq / 2.0
                          - u_sq.unsqueeze(0) / (2.0 * self.cs2))
        del eu_cs2, eu_cs2_sq, u_sq
        return phi.unsqueeze(0) * Gamma

    def _g_equilibrium(self, P, u):
        eu = torch.einsum('qd,d...->q...', self._e, u)
        u_sq = (u * u).sum(dim=0)
        w_view = self._w_view
        eu_cs2 = eu / self.cs2
        del eu
        s = w_view * (eu_cs2 + eu_cs2 * eu_cs2 / 2.0
                      - u_sq.unsqueeze(0) / (2.0 * self.cs2))
        del eu_cs2, u_sq
        return P.unsqueeze(0) * w_view + s

    # ------------------------------------------------------------------
    # Stress tensor
    # ------------------------------------------------------------------

    def _compute_neq_stress(self, g_neq):
        return torch.einsum('qab,q...->ab...', self._e_outer, g_neq)

    # ------------------------------------------------------------------
    # Main time step
    # ------------------------------------------------------------------

    def step(self):
        cfg = self.config
        cs2 = self.cs2
        Q = self.lattice.q
        w_view = self._w_view

        # 1. Recover phi
        self.phi = self.f.sum(dim=0)
        self.phi = torch.clamp(self.phi, 0.0, 1.0)
        if self._solid_any:
            self.phi = torch.where(self.solid, torch.zeros_like(self.phi), self.phi)

        # 2. Density, chemical potential, gradients (fused gradient+laplacian)
        phi_safe = torch.clamp(self.phi, 0.0, 1.0)
        self.rho = compute_density(phi_safe, cfg.rho_l, cfg.rho_g)
        rho_eff = torch.clamp(self.rho, min=self._rho_floor)

        grad_phi, lap_phi = self._lattice_gradient_and_laplacian(self.phi)
        self._compute_mu(lap_phi=lap_phi)
        del lap_phi
        if self.wetting is not None and self.geometric_wetting:
            grad_phi = self.wetting.correct_gradient(grad_phi)
        grad_mu = self._lattice_gradient(self.mu_phi)

        # 3. Recover pressure P = sum(g)
        self.P = self.g.sum(dim=0)
        self.p = self.P * cs2

        # 4. g_neq and stress tensor (NEW P, OLD u) — free early
        g_eq_stress = self._g_equilibrium(self.P, self.u)
        g_neq = self.g - g_eq_stress
        del g_eq_stress  # free ~342 MB
        sigma_neq = self._compute_neq_stress(g_neq)
        del g_neq  # free ~342 MB

        # 5. Total force (Fakhari 2017)
        drho = cfg.rho_l - cfg.rho_g
        F_s = self.mu_phi.unsqueeze(0) * grad_phi
        Fp = -self.P.unsqueeze(0) * drho * cs2 * grad_phi

        tau_visc = self._tau_g
        visc_coeff = (0.5 - tau_visc) / tau_visc
        Fm = visc_coeff * torch.einsum('ab...,b...->a...', sigma_neq, grad_phi) * drho
        del sigma_neq  # free ~486 MB

        g_force = cfg.g_force[:cfg.dim]
        if any(g != 0 for g in g_force):
            F_body = torch.tensor(g_force, dtype=self.dtype, device=self.device)
            grav_shape = (cfg.dim,) + (1,) * len(self.shape)
            F_body = F_body.reshape(grav_shape) * self.rho.unsqueeze(0)
        else:
            F_body = torch.zeros_like(F_s)

        if self.stab_mode == 'fakhari':
            F_total = F_s + Fp + Fm + F_body
        elif self.stab_mode == 'liang':
            F_total = F_s + Fp + F_body
        else:
            F_total = F_s + F_body
        del F_s, Fp, Fm, F_body

        # 6. Recover velocity
        u_from_g = torch.einsum('qd,q...->d...', self._e, self.g)
        self.u = u_from_g + 0.5 * F_total / rho_eff.unsqueeze(0)
        del u_from_g
        if self._solid_any:
            self.u = torch.where(self.solid.unsqueeze(0),
                                 torch.zeros_like(self.u), self.u)

        # 7. Interface tracking source term
        phi_u_now = phi_safe.unsqueeze(0) * self.u
        if not hasattr(self, '_phi_u_prev') or self._phi_u_prev is None:
            self._phi_u_prev = phi_u_now.clone()
        d_phi_u = phi_u_now - self._phi_u_prev
        self._phi_u_prev = phi_u_now.clone()
        del phi_u_now

        if self.interface_mode == 'cahn-hilliard':
            mobility = self._mobility
            tau_f_ch = self._tau_f
            guo_factor = tau_f_ch / (tau_f_ch - 0.5)
            ac_source = d_phi_u + mobility * (grad_mu - grad_phi) * guo_factor
        else:
            # Allen-Cahn (Liang 2018): sharpening + advection correction
            grad_phi_mag = (grad_phi * grad_phi).sum(dim=0).sqrt()
            grad_phi_mag_safe = torch.clamp(grad_phi_mag, min=1e-10)
            n = grad_phi / grad_phi_mag_safe.unsqueeze(0)
            del grad_phi_mag, grad_phi_mag_safe

            W = cfg.xi
            lam = 4.0 * phi_safe * (1.0 - phi_safe) / W
            sharpening = cs2 * lam.unsqueeze(0) * n * self._ac_scale
            del n, lam

            if self.wetting is not None and hasattr(self.wetting, 'wall_weight'):
                wall_w = self.wetting.wall_weight.to(self.dtype)
                reduce = self._sharpen_reduce
                sharpening_factor = (1.0 - reduce * wall_w).unsqueeze(0)
                sharpening = sharpening * sharpening_factor
                del wall_w, sharpening_factor

            ac_source = d_phi_u + sharpening
            del d_phi_u, sharpening

            if self.mu_drive > 0.0:
                mu_drive_alpha = self.mu_drive * self._mobility
                ac_source = ac_source + mu_drive_alpha * grad_mu

        del grad_phi, grad_mu  # free ~324 MB

        tau_f = self._tau_f
        ac_coeff = (1.0 - 0.5 / tau_f)
        eu_ac = torch.einsum('qd,d...->q...', self._e, ac_source)
        F_source = ac_coeff * w_view * eu_ac / cs2
        del eu_ac, ac_source

        # 8. Collision
        # f-collision (Allen-Cahn)
        f_eq = self._f_equilibrium(phi_safe, self.u)
        omega_f = 1.0 / tau_f
        self.f = self.f - omega_f * (self.f - f_eq) + F_source
        del f_eq, F_source

        # g-collision: composition-dependent tau
        # Density-weighted tau interpolation:
        # Interpolate dynamic viscosity eta = rho*nu = rho*cs2*(tau-0.5) linearly,
        # then back-calculate tau from local density rho_eff.
        # This eliminates the spurious viscous barrier (eta peak) at the interface
        # that inverse-tau or linear-tau interpolation creates at high density ratios.
        tau_l = cfg.tau_l
        tau_g = self._tau_g
        if tau_g != tau_l:
            rho_l_val = cfg.rho_l
            rho_g_val = cfg.rho_g
            # eta = rho * (tau - 0.5)   [cs2 cancels out]
            eta_num = phi_safe * rho_l_val * (tau_l - 0.5) + \
                      (1.0 - phi_safe) * rho_g_val * (tau_g - 0.5)
            rho_denom = phi_safe * rho_l_val + (1.0 - phi_safe) * rho_g_val
            tau_eff = 0.5 + eta_num / rho_denom
        else:
            tau_eff = torch.full_like(phi_safe, tau_l)

        eu_F = torch.einsum('qd,d...->q...', self._e, F_total)
        ns_coeff = (1.0 - 0.5 / tau_eff)
        G_g = ns_coeff.unsqueeze(0) * w_view * eu_F / (rho_eff.unsqueeze(0) * cs2)
        del eu_F, F_total

        g_eq = self._g_equilibrium(self.P, self.u)
        omega_g = 1.0 / tau_eff
        self.g = self.g - omega_g.unsqueeze(0) * (self.g - g_eq) + G_g
        del g_eq, G_g

        # 9. Streaming (save only solid-node values for bounce-back)
        solid = self.solid
        has_solid = self._solid_any
        has_partial = self._has_partial

        if has_solid:
            # Save pre-streaming values at solid nodes only (tiny memory)
            f_pre_solid = self.f[:, solid].clone()  # (Q, n_solid)
            g_pre_solid = self.g[:, solid].clone()

        if has_partial:
            # Save pre-streaming values at partial-solid nodes for blending
            f_pre_partial = self.f[:, self._partial_mask].clone()
            g_pre_partial = self.g[:, self._partial_mask].clone()

        # Streaming
        for i in range(Q):
            shifts = self._stream_shifts[i]
            for d in range(self.ndim):
                if shifts[d] != 0:
                    self.f[i] = torch.roll(self.f[i], shifts=shifts[d], dims=d)
                    self.g[i] = torch.roll(self.g[i], shifts=shifts[d], dims=d)

        # 10. Bounce-back (vectorized: no Python loop over Q)
        if has_solid:
            opp = self._opp  # (Q,)
            solid_idx = self._solid_idx  # cached at set_solid
            n_solid = solid_idx.shape[0]
            # f_pre_solid: (Q, n_solid) — pre-streaming values at solid nodes
            # After bounce-back: f[i, solid] = f_pre[opp[i], solid]
            # Vectorized: gather opp-direction values for all directions at once
            opp_expanded = opp.unsqueeze(1).expand(-1, n_solid)  # (Q, n_solid)
            self.f[:, solid] = torch.gather(f_pre_solid, 0, opp_expanded)
            self.g[:, solid] = torch.gather(g_pre_solid, 0, opp_expanded)
            del f_pre_solid, g_pre_solid

        # 10b. Volume penalization for partial-solid nodes
        if has_partial:
            eps = self._partial_eps  # (n_partial,)
            partial = self._partial_mask
            for i in range(Q):
                j = self._opp_list[i]
                f_str_partial = self.f[i][partial].clone()
                g_str_partial = self.g[i][partial].clone()
                self.f[i][partial] = (1.0 - eps) * f_str_partial + eps * f_pre_partial[j]
                self.g[i][partial] = (1.0 - eps) * g_str_partial + eps * g_pre_partial[j]
            del f_pre_partial, g_pre_partial

        # 11. Mass conservation (all-GPU; no host sync so step() is
        # CUDA-graph capturable; behavior identical to the scalar path)
        if self.phi_mass_init is not None:
            mass_now = self.phi.sum()
            safe = mass_now.clamp(min=1e-6)
            correction = torch.where(
                mass_now > 1e-6,
                self.phi_mass_init / safe,
                torch.ones_like(mass_now),
            )
            self.phi = torch.clamp(self.phi * correction, 0.0, 1.0)
            self.f = self.f * correction

        # 12. Boundary phi correction for contact angle enforcement
        if self.wetting is not None and self.boundary_relax > 0:
            mask = self.wetting.is_boundary
            phi_bnd = self.phi[mask].clone()

            # Filter: only apply at boundary nodes where:
            # 1. Wall normal is primarily vertical (ridge top, not sides)
            # 2. Interface actually passes through (0.15 < phi < 0.85)
            # This prevents Dy blowup from thin film fluxes along ridge.
            n_w = self.wetting.wall_normal.to(self.dtype)
            n_w_z = n_w[self.ndim - 1].abs()
            n_w_mag = (n_w * n_w).sum(dim=0).sqrt().clamp(min=1e-10)
            n_w_z_frac = n_w_z[mask] / n_w_mag[mask]
            is_top_surface = n_w_z_frac > 0.5
            in_interface = (phi_bnd > 0.15) & (phi_bnd < 0.85)

            if self.direct_wetting:
                phi_cubic = self.wetting.compute_cubic_boundary_phi(self.phi)
                phi_target = phi_cubic[mask]
                active = in_interface & is_top_surface
            elif (self._phi_wall_target is not None
                  and abs(self._phi_wall_target - 0.5) > 0.05):
                phi_target = torch.full_like(phi_bnd, self._phi_wall_target)
                active = in_interface & is_top_surface
            else:
                active = torch.zeros_like(phi_bnd, dtype=torch.bool)

            if active.any():
                phi_new = phi_bnd[active] + self.boundary_relax * (
                    phi_target[active] - phi_bnd[active])
                phi_new = torch.clamp(phi_new, 0.0, 1.0)
                phi_bnd[active] = phi_new
                self.phi[mask] = phi_bnd
                phi_old_at_active = self.f.sum(dim=0)[mask][active]
                phi_old_safe = torch.clamp(phi_old_at_active, min=1e-10)
                scale = phi_new / phi_old_safe
                scale = torch.clamp(scale, 0.5, 2.0)
                full_mask = mask.clone()
                full_mask_flat = full_mask.reshape(-1)
                active_flat = active.reshape(-1)
                active_indices = torch.where(full_mask_flat)[0][active_flat]
                for i in range(Q):
                    f_flat = self.f[i].reshape(-1)
                    f_flat[active_indices] *= scale

        self.step_count += 1
        self.t += 1.0

    # ------------------------------------------------------------------
    # Mid-simulation droplet injection
    # ------------------------------------------------------------------

    def inject_droplet(self, center, radius, xi, u_impact=None):
        """Inject a new droplet into the simulation at the current timestep.

        Modifies phi, f, g distributions to add the new droplet while
        preserving existing liquid. Uses equilibrium re-initialization
        in the droplet region to minimize transients.

        Parameters
        ----------
        center : tuple of float
            Droplet center in lattice units. 2D: (cx, cy), 3D: (cx, cy, cz).
        radius : float
            Droplet radius in lattice units.
        xi : float
            Interface thickness (should match solver's xi).
        u_impact : tuple of float or None
            Impact velocity. 2D: (ux, uy), 3D: (ux, uy, uz).
            Defaults to zero velocity if not provided.
        """
        cfg = self.config
        ndim = self.ndim
        shape = self.shape

        if u_impact is None:
            u_impact = (0.0,) * ndim

        # Build droplet C field on CPU then move to device
        grids = [np.arange(s) for s in shape]
        mesh = np.meshgrid(*grids, indexing='ij')
        dist_sq = sum((M - center[d]) ** 2 for d, M in enumerate(mesh))
        dist = np.sqrt(dist_sq)
        C_new = 0.5 + 0.5 * np.tanh(2.0 * (radius - dist) / xi)
        C_new = np.clip(C_new, 0.0, 1.0).astype(np.float32)
        C_new_t = torch.tensor(C_new, dtype=self.dtype, device=self.device)

        # Impact velocity field
        u_new = torch.zeros((cfg.dim,) + shape, dtype=self.dtype, device=self.device)
        for d in range(cfg.dim):
            u_new[d] = u_impact[d]

        # Identify droplet core (where new droplet is significantly present)
        core_mask = C_new_t > 0.1

        # Update phi: take max with new droplet
        phi_old = self.phi.clone()
        self.phi = torch.where(core_mask,
                               torch.maximum(self.phi, C_new_t),
                               self.phi)
        self.phi = torch.clamp(self.phi, 0.0, 1.0)

        # Zero out phi in solid
        if self._solid_any:
            self.phi = torch.where(self.solid, torch.zeros_like(self.phi), self.phi)

        # Update density
        self.rho = compute_density(self.phi, cfg.rho_l, cfg.rho_g)

        # Re-equilibrate f and g in the droplet core region
        # Blend velocity: use impact velocity where new droplet dominates,
        # keep existing velocity elsewhere
        C_weight = C_new_t.unsqueeze(0)  # (1, *grid)
        self.u = (1.0 - C_weight) * self.u + C_weight * u_new

        # Zero velocity in solid
        if self._solid_any:
            self.u = torch.where(self.solid.unsqueeze(0),
                                 torch.zeros_like(self.u), self.u)

        # Recompute equilibrium at core nodes
        f_eq_new = self._f_equilibrium(self.phi, self.u)
        g_eq_new = self._g_equilibrium(self.P, self.u)

        # Replace distributions only at core nodes
        core_expand = core_mask.unsqueeze(0)  # (1, *grid)
        self.f = torch.where(core_expand, f_eq_new, self.f)
        self.g = torch.where(core_expand, g_eq_new, self.g)

        # Update mass reference for conservation
        self.phi_mass_init = self.phi.sum().item()

        # Clear phi_u_prev to avoid spurious source from injection
        self._phi_u_prev = None

        center_str = ",".join(f"{c:.0f}" for c in center)
        vel_str = ",".join(f"{u:.3f}" for u in u_impact)
        print(f"  [inject] Droplet at ({center_str}) "
              f"R={radius:.1f} u=({vel_str}) "
              f"core_nodes={core_mask.sum().item()}")
