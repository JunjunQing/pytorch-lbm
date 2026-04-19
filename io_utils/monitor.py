"""Monitoring utilities for LBM simulations.

Tracks:
- Air film thickness (minimum gap between droplet and substrate)
- Air film thickness map (per-column gap for asymmetry analysis)
- Contact angle (2D)
- Mass conservation
- Maximum velocity
"""
import numpy as np
import torch


class Monitor:
    """Monitor simulation progress and key quantities."""

    def __init__(self, config, log_file=None):
        self.config = config
        self.log_file = log_file
        self.history = {
            "step": [],
            "t": [],
            "rho_min": [],
            "rho_max": [],
            "u_max": [],
            "mass": [],
            "film_thickness": [],
            "contact_angle_left": [],
            "contact_angle_right": [],
        }
        self.initial_mass = None

    def record(self, solver, step=None):
        """Record current state.

        Parameters
        ----------
        solver : LBMSolver
        step : int, optional
        """
        rho_cpu = solver.rho.cpu().numpy()
        u_cpu = solver.u.cpu().numpy()

        u_mag = np.sqrt((u_cpu ** 2).sum(axis=0))
        mass = float(rho_cpu.sum())

        if self.initial_mass is None:
            self.initial_mass = mass

        film = self.compute_film_thickness(rho_cpu, solver.solid.cpu().numpy())
        ca_left, ca_right = self.compute_contact_angle(rho_cpu, solver.solid.cpu().numpy())

        s = step or solver.step_count
        self.history["step"].append(s)
        self.history["t"].append(solver.t)
        self.history["rho_min"].append(float(rho_cpu.min()))
        self.history["rho_max"].append(float(rho_cpu.max()))
        self.history["u_max"].append(float(u_mag.max()))
        self.history["mass"].append(mass)
        self.history["film_thickness"].append(film)
        self.history["contact_angle_left"].append(ca_left)
        self.history["contact_angle_right"].append(ca_right)

    def compute_film_thickness(self, rho, solid):
        """Compute minimum air film thickness (vectorized).

        For each column, finds the gap between the topmost solid node
        and the bottommost liquid node above the substrate.

        Parameters
        ----------
        rho : ndarray, shape (nx, ny) or (nx, ny, nz)
        solid : ndarray, same shape as rho, bool

        Returns
        -------
        thickness : float
            Minimum gap in physical units, or float('inf') if no droplet found.
        """
        rho_threshold = (self.config.rho_l + self.config.rho_g) / 2

        if rho.ndim == 2:
            return self._film_thickness_2d(rho, solid, rho_threshold)
        else:
            return self._film_thickness_3d(rho, solid, rho_threshold)

    def compute_film_thickness_map(self, rho, solid):
        """Compute per-column air film thickness (vectorized).

        Returns
        -------
        gap_map : ndarray
            2D: shape (nx,), gap in lattice units
            3D: shape (nx, ny), gap in lattice units
        """
        rho_threshold = (self.config.rho_l + self.config.rho_g) / 2

        if rho.ndim == 2:
            nx, ny = rho.shape
            j_grid = np.arange(ny)
            sub_top = np.where(solid, j_grid, -1).max(axis=1)  # (nx,)

            # Mask below substrate
            above_sub = rho.copy()
            mask_below = j_grid[np.newaxis, :] <= sub_top[:, np.newaxis]
            above_sub[mask_below] = rho_threshold

            is_liquid = above_sub > rho_threshold
            drop_bottom = np.where(is_liquid, j_grid[np.newaxis, :], ny).argmin(axis=1)

            has_drop = is_liquid.any(axis=1)
            gap = np.where(has_drop, drop_bottom - sub_top, 0).astype(float)
            gap[~has_drop] = np.nan
            return gap
        else:
            nx, ny, nz = rho.shape
            k_grid = np.arange(nz)
            sub_top = np.where(solid, k_grid[np.newaxis, np.newaxis, :], -1).max(axis=2)

            mask_below = k_grid[np.newaxis, np.newaxis, :] <= sub_top[:, :, np.newaxis]
            above_sub = np.where(mask_below, rho_threshold, rho)

            is_liquid = above_sub > rho_threshold
            drop_bottom = np.where(is_liquid, k_grid[np.newaxis, np.newaxis, :], nz).argmin(axis=2)

            has_drop = is_liquid.any(axis=2)
            gap = np.where(has_drop, drop_bottom - sub_top, 0).astype(float)
            gap[~has_drop] = np.nan
            return gap

    def _film_thickness_2d(self, rho, solid, rho_threshold):
        """Vectorized 2D film thickness."""
        nx, ny = rho.shape
        j_grid = np.arange(ny)

        # Substrate top per column
        sub_top = np.where(solid, j_grid, -1).max(axis=1)  # (nx,)

        # Mask below substrate
        above_sub = rho.copy()
        mask_below = j_grid[np.newaxis, :] <= sub_top[:, np.newaxis]
        above_sub[mask_below] = rho_threshold

        # Find first liquid node per column
        is_liquid = above_sub > rho_threshold
        # argmin on where(is_liquid, j, ny) gives first j where liquid
        drop_bottom = np.where(is_liquid, j_grid[np.newaxis, :], ny).argmin(axis=1)

        has_drop = is_liquid.any(axis=1)
        if not has_drop.any():
            return float('inf')

        gap = np.where(has_drop, drop_bottom - sub_top, ny)
        min_gap = gap[has_drop].min()
        return min_gap * self.config.dx if min_gap < ny else float('inf')

    def _film_thickness_3d(self, rho, solid, rho_threshold):
        """Vectorized 3D film thickness."""
        nx, ny, nz = rho.shape
        k_grid = np.arange(nz)

        # Substrate top per column
        sub_top = np.where(solid, k_grid[np.newaxis, np.newaxis, :], -1).max(axis=2)

        # Mask below substrate
        mask_below = k_grid[np.newaxis, np.newaxis, :] <= sub_top[:, :, np.newaxis]
        above_sub = np.where(mask_below, rho_threshold, rho)

        # Find first liquid node per column
        is_liquid = above_sub > rho_threshold
        drop_bottom = np.where(is_liquid, k_grid[np.newaxis, np.newaxis, :], nz).argmin(axis=2)

        has_drop = is_liquid.any(axis=2)
        if not has_drop.any():
            return float('inf')

        gap = np.where(has_drop, drop_bottom - sub_top, nz)
        min_gap = gap[has_drop].min()
        return min_gap * self.config.dx if min_gap < nz else float('inf')

    def compute_contact_angle(self, rho, solid):
        """Measure contact angle of droplet on substrate (2D only).

        Method:
        1. Find substrate surface liquid coverage extent
        2. From contact points, trace the interface (rho = threshold contour)
        3. Fit line to interface near contact point
        4. Angle between fitted line and substrate tangent = contact angle

        Parameters
        ----------
        rho : ndarray, shape (nx, ny) or (nx, ny, nz)
        solid : ndarray, same shape, bool

        Returns
        -------
        ca_left, ca_right : float
            Contact angles at left and right contact points (degrees).
            Returns (nan, nan) for 3D or when no contact detected.
        """
        if rho.ndim != 2:
            return float('nan'), float('nan')

        return self._contact_angle_2d(rho, solid)

    def _contact_angle_2d(self, rho, solid):
        """2D contact angle measurement using density gradient method.

        At the contact line, computes the density gradient to find the
        interface normal direction. The contact angle is derived from the
        angle between the interface normal and the substrate normal.
        """
        nx, ny = rho.shape
        rho_threshold = (self.config.rho_l + self.config.rho_g) / 2

        j_grid = np.arange(ny)
        sub_top = np.where(solid, j_grid, -1).max(axis=1)

        # Find contact columns (liquid just above substrate)
        liquid_at_surface = np.zeros(nx, dtype=bool)
        for i in range(nx):
            j_sub = int(sub_top[i])
            if j_sub + 1 < ny and rho[i, j_sub + 1] > rho_threshold:
                liquid_at_surface[i] = True

        if not liquid_at_surface.any():
            return float('nan'), float('nan')

        contact_cols = np.where(liquid_at_surface)[0]
        i_left = contact_cols[0]
        i_right = contact_cols[-1]

        # Compute density gradient (central differences)
        drho_dx = np.zeros_like(rho)
        drho_dy = np.zeros_like(rho)
        drho_dx[1:-1, :] = (rho[2:, :] - rho[:-2, :]) / 2.0
        drho_dy[:, 1:-1] = (rho[:, 2:] - rho[:, :-2]) / 2.0

        # At contact points, average gradient over a few cells near the substrate
        ca_left = self._angle_from_gradient(drho_dx, drho_dy, sub_top,
                                             i_left, side="left")
        ca_right = self._angle_from_gradient(drho_dx, drho_dy, sub_top,
                                              i_right, side="right")
        return ca_left, ca_right

    def _angle_from_gradient(self, drho_dx, drho_dy, sub_top, i_contact, side="left"):
        """Compute contact angle from density gradient at contact point.

        The contact angle is theta = arccos(|gy|/|g|) where g is the
        density gradient at the contact point. This follows from the
        geometric relationship between the gradient (perpendicular to
        iso-density contours) and the substrate normal.
        """
        nx, ny = drho_dx.shape
        j_sub = int(sub_top[i_contact])
        j = j_sub + 1  # first fluid node above substrate

        if j >= ny:
            return float('nan')

        # Average gradient over a few columns near the contact point
        gx_sum, gy_sum = 0.0, 0.0
        count = 0
        n_avg = 3

        for di in range(n_avg):
            if side == "left":
                i = i_contact + di  # move right into the droplet
            else:
                i = i_contact - di  # move left into the droplet

            if i < 0 or i >= nx:
                continue

            gx = drho_dx[i, j]
            gy = drho_dy[i, j]
            gmag = np.sqrt(gx**2 + gy**2)
            if gmag > 1e-10:
                gx_sum += gx
                gy_sum += gy
                count += 1

        if count == 0:
            return float('nan')

        # Average gradient
        gx = gx_sum / count
        gy = gy_sum / count
        gmag = np.sqrt(gx**2 + gy**2)

        if gmag < 1e-10:
            return float('nan')

        # Contact angle: theta = arccos(|gy| / |g|)
        cos_theta = np.clip(abs(gy) / gmag, 0, 1)
        theta = np.degrees(np.arccos(cos_theta))
        return theta

    def summary(self):
        """Print summary."""
        if not self.history["step"]:
            print("No data recorded.")
            return

        print(f"\nMonitor Summary ({len(self.history['step'])} records)")
        print(f"  rho: [{min(self.history['rho_min']):.4f}, "
              f"{max(self.history['rho_max']):.4f}]")
        print(f"  |u|_max: {max(self.history['u_max']):.4e}")
        if self.initial_mass and self.initial_mass > 0:
            mass_change = (self.history["mass"][-1] - self.initial_mass) / self.initial_mass
            print(f"  Mass conservation: {mass_change:+.4e} (relative change)")

        films = [f for f in self.history["film_thickness"] if f != float('inf')]
        if films:
            print(f"  Min film thickness: {min(films):.6e}")

        ca_l = [a for a in self.history["contact_angle_left"] if not np.isnan(a)]
        ca_r = [a for a in self.history["contact_angle_right"] if not np.isnan(a)]
        if ca_l:
            print(f"  Contact angle (L): {ca_l[-1]:.1f}°")
        if ca_r:
            print(f"  Contact angle (R): {ca_r[-1]:.1f}°")
