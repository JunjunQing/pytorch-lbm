"""Geometric wetting boundary conditions for free-energy phase-field LBM.

Supports two modes:
  'quadratic' — Connington & Lee (2013) ghost-fluid with phi*(1-phi) switching.
    Limited to ~130° contact angle because phi*(1-phi) → 0 at contact line.
  'linear' — Linear wall energy g_w(phi) = -h*phi with constant derivative.
    Supports full 0°-180° range. Based on Jacqmin (2000) formulation.

Key references:
  Connington & Lee, JCP 250, 601-615 (2013)
  Jacqmin, J. Fluid Mech. 402, 57-88 (2000)
  Ju, Guo, Yan, Sun, Phys. Rev. E 109, 045307 (2024) — chemical potential approach
  Zhang, Wu, Wang, Nestler, J. Chem. Phys. 159, 164701 (2023) — volume preservation

For the linear wall energy:
  g_w(phi) = -h * phi
  dg_w/dphi = -h  (constant, never vanishes)
  h = sigma * cos(theta)
  Young's equation: cos(theta) = [g_w(0) - g_w(1)] / sigma = h / sigma

  Ghost-node: phi_ghost = phi + (e·n̂) * h/kappa  (constant correction)
  Chemical potential: delta_mu = -h  (constant at boundary nodes)

Relation to phi_c: h = -phi_c / 6  (from sigma = sqrt(2*kappa*beta)/6)
"""
import torch
import numpy as np
import math


class ConningtonLeeWetting:
    """Geometric wetting boundary condition for free-energy phase-field LBM.

    Uses ghost-fluid approach: replaces phi at solid boundary nodes with
    extrapolated values so that gradient and Laplacian computations
    naturally include the wetting condition.
    """

    def __init__(self, solid, lattice, phi_c, kappa, sigma=0.0,
                 theta_deg=90.0, device='cuda', mode='linear',
                 ghost_scale=None, geo_amplification=1.0,
                 solid_fraction=None):
        """
        Parameters
        ----------
        solid : Tensor, shape (*grid), bool
            Solid mask (True at solid nodes).
        lattice : D3Q19
            Lattice object with e, w, opp.
        phi_c : float
            Wetting potential. cos(theta) = -phi_c / sqrt(2*kappa*beta).
        kappa : float
            Interface stiffness parameter.
        sigma : float
            Surface tension (needed for linear mode).
        theta_deg : float
            Target contact angle in degrees (needed for linear mode).
        device : str
            'cuda' or 'cpu'.
        mode : str
            'linear' for constant correction (full 0-180° range), or
            'quadratic' for phi*(1-phi) correction (limited range).
        ghost_scale : float or None
            Scaling factor for the ghost-node correction. If None, auto-selected.
            Higher values = stronger wetting correction. Default auto: 2 for
            moderate angles, 3 for extreme angles.
        solid_fraction : ndarray or None
            Float array of solid fraction (0=fluid, 1=solid). If provided,
            wall normals are computed from its gradient, giving much smoother
            normals on curved surfaces than the discrete neighbor-counting method.
        """
        self.solid = solid
        self.lattice = lattice
        self.phi_c = phi_c
        self.kappa = kappa
        self.sigma = sigma
        self.theta_deg = theta_deg
        self.device = device
        self.mode = mode
        self.ndim = solid.dim()
        self._xi = 5.0  # Interface thickness for near-wall weighting
        self._solid_fraction_np = solid_fraction  # numpy array or None

        # Compute wetting parameters based on mode
        if mode == 'linear':
            # Linear wall energy: g_w(phi) = -h*phi
            # h = sigma * cos(theta)
            # Relation to phi_c: h = -phi_c/6 (from sigma = sqrt(2*kappa*beta)/6)
            cos_theta = math.cos(math.radians(theta_deg))
            self._h = sigma * cos_theta  # Wetting parameter
            # Ghost-node scaling: the lattice gradient sums contributions from
            # multiple directions, diluting the correction from any single ghost
            # value. For D3Q19 flat wall: effective correction ~ 2x weaker.
            if ghost_scale is None:
                # Ghost-node scaling: the lattice gradient/Laplacian sums contributions
                # from multiple directions, diluting the correction from any single
                # ghost value. For D3Q19 flat wall: effective correction ~ 2x weaker.
                # Higher ghost_scale compensates for this dilution.
                if abs(theta_deg - 90.0) > 60.0:
                    ghost_scale = 3.0
                else:
                    ghost_scale = 2.0
            self._h_over_kappa = self._h / kappa * ghost_scale
            # Chemical potential: -h scaled by 1/delta_n (~2 for half-lattice wall)
            self._mu_wall = -self._h * 2.0
            self._ghost_scale = ghost_scale
            print(f"  Wetting (linear): h={self._h:.6f}, "
                  f"h/kappa_eff={self._h_over_kappa:.4f} (scale={ghost_scale}), "
                  f"mu_wall={self._mu_wall:.6f}")
        else:
            # Quadratic (Connington & Lee) mode
            self._phi_c_over_kappa = phi_c / kappa
            self._h = 0.0
            self._h_over_kappa = 0.0
            self._mu_wall = 0.0

        self._geo_amplification = geo_amplification

        # Precompute boundary information
        self._precompute_boundary()

    def _precompute_boundary(self):
        """Identify boundary fluid nodes, wall normals, and solid-direction masks."""
        solid = self.solid
        e = self.lattice.e.numpy().astype(int)
        Q = self.lattice.q
        ndim = self.ndim

        # Boundary fluid nodes: fluid nodes with at least one solid neighbor
        is_boundary = torch.zeros_like(solid)
        for i in range(1, Q):
            shifted_solid = solid.clone()
            for d in range(ndim):
                s = int(e[i, d])
                if s != 0:
                    shifted_solid = torch.roll(shifted_solid, shifts=s, dims=d)
            # This node is fluid, and its neighbor in direction i is solid
            is_boundary |= (shifted_solid & ~solid)

        self.is_boundary = is_boundary  # (*grid), bool

        # Compute wall normals (outward: solid -> fluid)
        solid_np = solid.cpu().numpy()
        boundary_np = is_boundary.cpu().numpy()
        shape = solid_np.shape

        self.wall_normal = torch.zeros((ndim,) + shape, dtype=torch.float32)

        if self._solid_fraction_np is not None:
            # Use gradient of solid fraction for smooth normals on curved surfaces
            self._compute_smooth_normals(solid_np, boundary_np, shape)
        elif ndim == 3:
            # Discrete neighbor-counting (original approach)
            nx, ny, nz = shape
            boundary_coords = np.argwhere(boundary_np)

            for coord in boundary_coords:
                ix, iy, iz = coord
                normal = np.zeros(3)
                count = 0
                for i in range(1, Q):
                    nix = ix + e[i, 0]
                    niy = iy + e[i, 1]
                    niz = iz + e[i, 2]
                    if 0 <= nix < nx and 0 <= niy < ny and 0 <= niz < nz:
                        if solid_np[nix, niy, niz]:
                            normal[0] -= e[i, 0]
                            normal[1] -= e[i, 1]
                            normal[2] -= e[i, 2]
                            count += 1
                if count > 0:
                    normal /= count
                    norm = np.sqrt(normal[0]**2 + normal[1]**2 + normal[2]**2)
                    if norm > 1e-10:
                        normal /= norm
                self.wall_normal[0, ix, iy, iz] = normal[0]
                self.wall_normal[1, ix, iy, iz] = normal[1]
                self.wall_normal[2, ix, iy, iz] = normal[2]
        elif ndim == 2:
            # Discrete neighbor-counting for 2D (D2Q9)
            nx, ny = shape
            boundary_coords = np.argwhere(boundary_np)

            for coord in boundary_coords:
                ix, iy = coord
                normal = np.zeros(2)
                count = 0
                for i in range(1, Q):
                    nix = ix + e[i, 0]
                    niy = iy + e[i, 1]
                    if 0 <= nix < nx and 0 <= niy < ny:
                        if solid_np[nix, niy]:
                            normal[0] -= e[i, 0]
                            normal[1] -= e[i, 1]
                            count += 1
                if count > 0:
                    normal /= count
                    norm = np.sqrt(normal[0]**2 + normal[1]**2)
                    if norm > 1e-10:
                        normal /= norm
                self.wall_normal[0, ix, iy] = normal[0]
                self.wall_normal[1, ix, iy] = normal[1]

        self.wall_normal = self.wall_normal.to(self.device)

        # Precompute near-wall mask and weight for wetting force
        # near_wall_mask: True for nodes within ~xi lattice units of the wall
        # wall_weight: exponential decay from wall (1 at wall, ~0 at xi distance)
        self._precompute_near_wall(solid_np, boundary_np, shape)
        self._precompute_nsw_indices(solid_np, shape)

        # Precompute for each direction i:
        # 1. solid_neighbor_mask[i]: True at boundary fluid nodes whose
        #    neighbor in direction i is solid (needs wetting correction)
        # 2. n_dot_e[i]: n . e_i at each node (for extrapolation)
        # Only store directions that actually have corrections (skip empty ones)
        self._solid_neighbor_mask = [None] * Q
        self._n_dot_e = [None] * Q
        self._has_wetting_dir = [False] * Q

        for i in range(1, Q):
            # Shift solid mask by -e_i to check if neighbor in direction i is solid
            neighbor_is_solid = solid.clone()
            for d in range(ndim):
                s = int(e[i, d])
                if s != 0:
                    neighbor_is_solid = torch.roll(neighbor_is_solid,
                                                    shifts=-s, dims=d)

            mask = neighbor_is_solid & is_boundary
            has_any = mask.any().item()
            self._has_wetting_dir[i] = has_any

            if has_any:
                self._solid_neighbor_mask[i] = mask
                # n . e_i = sum_d n_d * e_id
                n_dot_e = torch.zeros(shape, dtype=torch.float32, device=self.device)
                for d in range(ndim):
                    n_dot_e += self.wall_normal[d] * float(e[i, d])
                self._n_dot_e[i] = n_dot_e

    def _compute_smooth_normals(self, solid_np, boundary_np, shape):
        """Compute wall normals from gradient of solid fraction field.

        The gradient of solid_fraction points from fluid (0) toward solid (1).
        The wall normal points outward (solid -> fluid), so we negate the gradient.

        This gives much smoother normals on curved surfaces than counting
        discrete solid neighbors, eliminating the staircase artifact in
        wetting boundary conditions.
        """
        ndim = self.ndim
        frac = self._solid_fraction_np

        # Compute gradient using central differences
        # np.gradient returns a list of arrays, one per dimension
        grad = np.gradient(frac)  # tuple of (nx, ny, nz) arrays

        # Normal = -gradient (pointing from solid toward fluid)
        mag = np.sqrt(sum(g ** 2 for g in grad))
        mag[mag < 1e-10] = 1e-10

        for d in range(ndim):
            normal_d = -grad[d] / mag
            # Only set at boundary nodes
            self.wall_normal[d] = torch.tensor(
                normal_d * boundary_np, dtype=torch.float32)

        print(f"  Smooth normals: computed from solid_fraction gradient "
              f"({boundary_np.sum()} boundary nodes)")

    def _precompute_near_wall(self, solid_np, boundary_np, shape):
        """Compute near-wall mask and weight for wetting force application.

        Uses efficient torch-based distance propagation from boundary nodes.
        """
        xi = self._xi
        ndim = self.ndim

        # Start with boundary nodes as distance=0, others as large
        dist = torch.full(shape, 100.0, dtype=torch.float32)
        dist[torch.from_numpy(boundary_np)] = 0.0

        # Propagate distance using min-pooling with face neighbors
        n_passes = int(2 * xi) + 1
        for _ in range(n_passes):
            new_dist = dist.clone()
            for d in range(ndim):
                for shift in [-1, 1]:
                    shifted = torch.roll(dist, shifts=shift, dims=d)
                    new_dist = torch.minimum(new_dist, shifted + 1.0)
            dist = new_dist

        # Zero out solid nodes
        solid_t = torch.from_numpy(solid_np)
        dist[solid_t] = 100.0

        # Convert to weight: exponential decay from wall
        wall_weight = torch.exp(-dist / (xi / 2.0))
        wall_weight[solid_t] = 0.0

        # Near-wall mask: within 2*xi of the wall
        near_wall = (dist < 2.0 * xi) & ~solid_t

        self.wall_weight = wall_weight.to(self.device)
        self.near_wall_mask = near_wall.to(self.device)

    def _precompute_nsw_indices(self, solid_np, shape):
        """Precompute flat indices for boundary nodes and next-interior-wall nodes.

        For each boundary fluid node, the 'next interior' node is one lattice
        step in the wall-normal direction (away from the wall, into the fluid).
        This is used by the cubic wall energy phi-setting approach.
        """
        ndim = self.ndim
        bnd_coords = torch.argwhere(self.is_boundary).cpu()  # (n_bnd, ndim) — force CPU
        n_bnd = bnd_coords.shape[0]

        if n_bnd == 0:
            self._bnd_flat_indices = torch.tensor([], dtype=torch.long, device=self.device)
            self._nsw_flat_indices = torch.tensor([], dtype=torch.long, device=self.device)
            self._nsw_is_fluid = torch.tensor([], dtype=torch.bool, device=self.device)
            self._cubic_a = 0.0
            return

        # Wall normals at boundary nodes (everything on CPU for indexing)
        n_w = self.wall_normal.cpu()  # (ndim, *grid)
        n_w_bnd = torch.zeros((n_bnd, ndim), dtype=torch.float32)
        for d in range(ndim):
            idx = tuple(bnd_coords.T)
            n_w_bnd[:, d] = n_w[d][idx]

        # Round to nearest lattice direction for shift
        shift = torch.round(n_w_bnd).long()

        # Next interior node coordinates
        nsw_coords = bnd_coords + shift
        for d in range(ndim):
            nsw_coords[:, d] = nsw_coords[:, d].clamp(0, shape[d] - 1)

        # Compute flat indices
        strides = torch.ones(ndim, dtype=torch.long)
        for d in range(ndim - 2, -1, -1):
            strides[d] = strides[d + 1] * shape[d + 1]

        bnd_flat = (bnd_coords * strides.unsqueeze(0)).sum(dim=1)
        nsw_flat = (nsw_coords * strides.unsqueeze(0)).sum(dim=1)

        self._bnd_flat_indices = bnd_flat.to(self.device)
        self._nsw_flat_indices = nsw_flat.to(self.device)

        # Check which next-interior nodes are actually fluid
        solid_flat = torch.from_numpy(solid_np).reshape(-1)
        self._nsw_is_fluid = ~solid_flat[nsw_flat].to(self.device)

        # Precompute cubic wall energy coefficient
        # a = 6*sigma*cos(theta)/kappa
        cos_theta = math.cos(math.radians(self.theta_deg))
        self._cubic_a = 6.0 * self.sigma * cos_theta / self.kappa

        print(f"  Cubic wetting: a={self._cubic_a:.4f}, n_bnd={n_bnd}, "
              f"nsw_fluid={self._nsw_is_fluid.sum().item()}")

    def compute_cubic_boundary_phi(self, phi):
        """Compute boundary phi using cubic wall energy (Jiang 2024).

        The cubic wall energy g_w(phi) = -(b1/6)*phi^2*(3-2*phi) gives:
          kappa * (n_w · grad_phi) = -b1 * phi_b * (1-phi_b)

        With one-sided difference (phi_nsw - phi_b) = delta_n:
          a*phi_b^2 + (1-a)*phi_b - phi_nsw = 0

        where a = b1/kappa = 6*sigma*cos(theta)/kappa

        Solution: phi_b = [-(1-a) + sqrt((1-a)^2 + 4*a*phi_nsw)] / (2*a)

        The phi*(1-phi) switching function ensures the correction vanishes
        as phi -> 0 or phi -> 1, preventing unphysical negative values.
        """
        if self._bnd_flat_indices.numel() == 0:
            return phi

        a = self._cubic_a
        if abs(a) < 1e-10:
            return phi  # theta ≈ 90°, no correction

        # Gather phi at boundary and next-interior nodes
        phi_flat = phi.reshape(-1)
        bnd_idx = self._bnd_flat_indices
        nsw_idx = self._nsw_flat_indices

        phi_nsw = phi_flat[nsw_idx]
        phi_bnd_old = phi_flat[bnd_idx].clone()

        # Solve quadratic: a*phi_b^2 + (1-a)*phi_b - phi_nsw = 0
        one_minus_a = 1.0 - a
        disc = one_minus_a**2 + 4.0 * a * phi_nsw
        disc = torch.clamp(disc, min=0.0)

        phi_b = (-one_minus_a + torch.sqrt(disc)) / (2.0 * a)
        phi_b = torch.clamp(phi_b, 0.0, 1.0)

        # Only apply where next-interior node is fluid and phi is near interface
        active = self._nsw_is_fluid & (phi_bnd_old > 0.01) & (phi_bnd_old < 0.99)

        # Create corrected phi
        phi_new = phi.clone()
        phi_flat_new = phi_new.reshape(-1)
        phi_flat_new[bnd_idx[active]] = phi_b[active]

        return phi_new

    def correct_shifted(self, phi, phi_shifted, direction_idx):
        """Replace phi at solid neighbor positions with wetting extrapolation.

        Linear mode (full 0-180° range):
          kappa * dphi/dn = h = sigma * cos(theta)
          phi(x+e_i) = phi(x) + (e·n̂) * h/kappa

        Quadratic mode (Connington & Lee, limited range):
          phi(x+e_i) = phi(x) - (n·e_i) * (phi_c/kappa) * phi*(1-phi)
        """
        if direction_idx == 0 or not self._has_wetting_dir[direction_idx]:
            return phi_shifted

        mask = self._solid_neighbor_mask[direction_idx]
        n_dot_e = self._n_dot_e[direction_idx]

        if self.mode == 'linear':
            # Linear wall energy: constant correction
            # h = sigma*cos(theta). For theta>90°: h<0, so ghost value < phi (hydrophobic)
            wetting_val = phi - n_dot_e * self._h_over_kappa
        else:
            # Quadratic (Connington & Lee): phi*(1-phi) switching
            # Sign convention: phi_c > 0 for theta > 90° (hydrophobic)
            # For hydrophobic: need ghost value < phi to create positive grad change
            # Formula: phi_ghost = phi + (e·n̂) * (phi_c/kappa) * phi*(1-phi)
            # For e·n̂ < 0 (into wall): phi_ghost = phi - (phi_c/kappa)*phi*(1-phi) < phi ✓
            wetting_val = phi + n_dot_e * self._phi_c_over_kappa * phi * (1.0 - phi)

        return torch.where(mask, wetting_val, phi_shifted)

    def correct_gradient(self, grad_phi, near_wall_only=True):
        """Correct gradient using geometric contact angle condition.

        Uses the geometric relation (Zhang 2023, J. Chem. Phys. 159, 164701):
          tan(π/2 - θ) = -(n_w · ∇φ) / |P(∇φ)|

        where P(∇φ) = ∇φ - (n_w · ∇φ) * n_w is the tangential projection.

        When near_wall_only=True (default): applies correction at all near-wall
        nodes within the interface region (2*xi from wall). This extends the
        correction beyond the first boundary layer, allowing the AC sharpening
        to follow the corrected direction across the entire interface thickness.

        When near_wall_only=False: applies only at boundary nodes (legacy).
        """
        if near_wall_only:
            mask = self.near_wall_mask
        else:
            mask = self.is_boundary
        if not mask.any():
            return grad_phi

        dtype = grad_phi.dtype
        device = grad_phi.device

        n_w = self.wall_normal.to(dtype=dtype, device=device)

        # Current normal component: g_n = n_w · ∇φ
        g_n = (n_w * grad_phi).sum(dim=0)

        # Tangential gradient: ∇φ_t = ∇φ - (n_w · ∇φ) * n_w
        g_t = grad_phi - g_n.unsqueeze(0) * n_w

        # Tangential magnitude
        g_t_mag = (g_t * g_t).sum(dim=0).sqrt().clamp(min=1e-10)

        # Desired normal gradient from geometric contact angle condition:
        #   (n_w · ∇φ) = -|∇φ_t| * cos(θ) / sin(θ)
        cos_theta = math.cos(math.radians(self.theta_deg))
        sin_theta = math.sin(math.radians(self.theta_deg))
        if abs(sin_theta) < 1e-10:
            return grad_phi  # θ ≈ 0° or 180°, degenerate

        g_n_desired = -g_t_mag * cos_theta / sin_theta

        # Apply amplification factor to overcome AC sharpening resistance.
        # The AC sharpening term pushes phi along the interface normal,
        # overriding the geometric correction. Amplification compensates.
        # amp may be a scalar (>1.0 activates) or a per-node field (tensor
        # matching the grid shape) for spatially adaptive amplification;
        # nodes with amp <= 1.0 keep the un-amplified geometric correction.
        amp = getattr(self, '_geo_amplification', None)
        if amp is not None:
            if torch.is_tensor(amp):
                # Adaptive field: per-node amplification, broadcast over dims
                amp_eff = amp.to(dtype=dtype, device=device)
                g_n_current = g_n
                delta = g_n_desired - g_n_current
                g_n_desired = torch.where(
                    amp_eff > 1.0,
                    g_n_current + delta * amp_eff,
                    g_n_desired,
                )
            elif amp > 1.0:
                # Blend between current g_n and desired g_n with amplification
                g_n_current = g_n
                delta = g_n_desired - g_n_current
                g_n_desired = g_n_current + delta * amp

        # Reconstruct corrected gradient: ∇φ_corrected = ∇φ_t + g_n_desired * n_w
        grad_corrected = g_t + g_n_desired.unsqueeze(0) * n_w

        # Apply at near-wall nodes where gradient is significant
        # (where the interface actually exists)
        grad_mag = (grad_phi * grad_phi).sum(dim=0).sqrt()
        near_interface = grad_mag > 1e-6
        apply_mask = mask & near_interface

        # For near-wall correction, blend smoothly using wall_weight
        # to avoid discontinuity at the near_wall boundary
        if near_wall_only and hasattr(self, 'wall_weight'):
            # Smooth blending: full correction at wall, decaying with distance
            blend = self.wall_weight.to(dtype=dtype)
            mask_3d = apply_mask.unsqueeze(0).expand_as(grad_phi)
            blend_3d = blend.unsqueeze(0).expand_as(grad_phi)
            corrected = grad_phi + blend_3d * (grad_corrected - grad_phi)
            return torch.where(mask_3d, corrected, grad_phi)
        else:
            mask_3d = apply_mask.unsqueeze(0).expand_as(grad_phi)
            return torch.where(mask_3d, grad_corrected, grad_phi)

    def apply_wetting_to_grad_cd(self, C, grad_cd_C):
        """Modify CD gradient of C at boundary nodes for wetting BC.

        Kept for backward compatibility. Prefer using correct_shifted()
        in the gradient/Laplacian stencil loops.
        """
        phi_c_over_kappa = self.phi_c / self.kappa
        C_s = C * (1.0 - C)
        wetting_dCdn = phi_c_over_kappa * C_s

        n = self.wall_normal
        n_mag_sq = (n * n).sum(dim=0, keepdim=True).clamp(min=1e-10)
        n_hat = n / n_mag_sq.sqrt()

        grad_cd_normal = (grad_cd_C * n_hat).sum(dim=0, keepdim=True)
        correction = (wetting_dCdn.unsqueeze(0) - grad_cd_normal) * n_hat

        mask = self.is_boundary.unsqueeze(0)
        grad_cd_C = grad_cd_C + torch.where(mask, correction,
                                              torch.zeros_like(correction))
        return grad_cd_C

    def apply_wetting_to_laplacian(self, C, laplacian_C):
        """Modify Laplacian of C at boundary nodes for wetting BC.

        Kept for backward compatibility. Prefer using correct_shifted().
        """
        phi_c_over_kappa = self.phi_c / self.kappa
        C_s = C * (1.0 - C)
        correction = phi_c_over_kappa * C_s
        mask = self.is_boundary
        laplacian_C = torch.where(mask, laplacian_C + correction, laplacian_C)
        return laplacian_C

    def update_wetting_potential(self, phi_c):
        """Update the wetting potential (e.g., for dynamic contact angle)."""
        self.phi_c = phi_c

    @staticmethod
    def phi_c_from_contact_angle(theta_deg, kappa, beta):
        """Compute wetting potential from desired contact angle.

        cos(theta) = -phi_c / sqrt(2 * kappa * beta)
        phi_c = -cos(theta) * sqrt(2 * kappa * beta)
        """
        import math
        cos_theta = math.cos(math.radians(theta_deg))
        return -cos_theta * math.sqrt(2.0 * kappa * beta)
