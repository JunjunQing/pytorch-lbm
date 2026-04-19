"""Substrate geometry generators for LBM.

Creates boolean masks for solid nodes using vectorized numpy operations.
Supports:
- Flat substrate
- Convex substrate (hemisphere bulging up from bottom)
- Concave substrate (hemispherical cavity in bottom)
- Cylindrical ridge (2D: semicircle, 3D: cylinder along y-axis)
"""
import numpy as np


def create_substrate(nx, ny, nz=1, substrate_type="flat",
                     R_star=None, R_d=None, dx=1.0):
    """Create substrate solid mask.

    The substrate sits at the bottom of the domain. All types include
    a flat bottom wall at j=0 (2D) or k=0 (3D).

    Parameters
    ----------
    nx, ny, nz : int
        Grid dimensions.
    substrate_type : str
        "flat", "convex", "concave", or "ridge".
    R_star : float or None
        R* = R_substrate / R_droplet. Determines substrate curvature.
    R_d : float or None
        Droplet radius (physical units).
    dx : float
        Grid spacing.

    Returns
    -------
    solid : ndarray, shape (nx, ny[, nz]), bool
        True at solid nodes.
    """
    if nz == 1:
        return _create_substrate_2d(nx, ny, substrate_type, R_star, R_d, dx)
    else:
        return _create_substrate_3d(nx, ny, nz, substrate_type, R_star, R_d, dx)


def substrate_surface_height(nx, ny, nz=1, substrate_type="flat",
                              R_star=None, R_d=None, dx=1.0):
    """Compute substrate surface height at each (x[, y]) position.

    Returns the z-index (or y-index for 2D) of the topmost solid node
    in each column.

    Returns
    -------
    height : ndarray
        2D: shape (nx,), surface y-index per column
        3D: shape (nx, ny), surface z-index per column
    """
    solid = create_substrate(nx, ny, nz, substrate_type, R_star, R_d, dx)
    return _compute_surface_height(solid)


def _compute_surface_height(solid):
    """Get topmost solid index per column (vectorized)."""
    if solid.ndim == 2:
        nx, ny = solid.shape
        j_grid = np.arange(ny)
        masked = np.where(solid, j_grid, -1)
        height = masked.max(axis=1)  # (nx,)
        return height
    else:
        nx, ny, nz = solid.shape
        k_grid = np.arange(nz)
        masked = np.where(solid, k_grid[np.newaxis, np.newaxis, :], -1)
        height = masked.max(axis=2)  # (nx, ny)
        return height


def _substrate_radius(R_star, R_d, dx):
    """Compute substrate radius in lattice units."""
    if R_star is None or R_d is None:
        return None
    return abs(R_star) * R_d / dx


# ---- 2D implementations (vectorized) ----

def _create_substrate_2d(nx, ny, substrate_type, R_star, R_d, dx):
    """2D substrate mask — fully vectorized."""
    solid = np.zeros((nx, ny), dtype=bool)

    # Bottom wall (y=0)
    solid[:, 0] = True

    if substrate_type == "flat" or R_star is None or R_d is None:
        return solid

    R_g = _substrate_radius(R_star, R_d, dx)
    cx = nx / 2.0

    i = np.arange(nx)
    j = np.arange(ny)
    I, J = np.meshgrid(i, j, indexing='ij')

    if substrate_type == "convex":
        # Hemisphere center at (cx, R_g): creates a bump of height R_g
        # Solid = inside sphere
        dist_sq = (I - cx)**2 + (J - R_g)**2
        solid |= (dist_sq < R_g**2)

    elif substrate_type == "concave":
        # Hemisphere center at (cx, R_g): creates a cavity of depth R_g
        # Solid = outside sphere AND below sphere equator
        dist_sq = (I - cx)**2 + (J - R_g)**2
        solid |= (dist_sq > R_g**2) & (J < R_g)

    elif substrate_type == "ridge":
        # Semicircular ridge: half-circle on flat wall at j=0
        # Center at (cx, 0), extending upward
        dist_sq = (I - cx)**2 + J**2
        solid |= (dist_sq < R_g**2) & (J > 0)

    return solid


# ---- 3D implementations (vectorized) ----

def _create_substrate_3d(nx, ny, nz, substrate_type, R_star, R_d, dx):
    """3D substrate mask — fully vectorized."""
    solid = np.zeros((nx, ny, nz), dtype=bool)

    # Bottom wall (z=0)
    solid[:, :, 0] = True

    if substrate_type == "flat" or R_star is None or R_d is None:
        return solid

    R_g = _substrate_radius(R_star, R_d, dx)
    cx, cy = nx / 2.0, ny / 2.0

    i, j, k = np.ogrid[:nx, :ny, :nz]

    if substrate_type == "convex":
        # Hemisphere center at (cx, cy, R_g): bump of height R_g
        dist_sq = (i - cx)**2 + (j - cy)**2 + (k - R_g)**2
        solid |= (dist_sq < R_g**2)

    elif substrate_type == "concave":
        # Hemisphere center at (cx, cy, R_g): cavity of depth R_g
        dist_sq = (i - cx)**2 + (j - cy)**2 + (k - R_g)**2
        solid |= (dist_sq > R_g**2) & (k < R_g)

    elif substrate_type == "ridge":
        # Cylindrical ridge along y-axis (axial direction, flat).
        # Semicircular cross-section in x-z plane (azimuthal/curved direction).
        # Half-cylinder sitting on flat plate at z=0.
        # Center of semicircle at (cx, y, 0), extending upward.
        dist_sq = (i - cx)**2 + k**2
        solid |= (dist_sq < R_g**2) & (k > 0)

    return solid


def create_substrate_with_fraction(nx, ny, nz=1, substrate_type="flat",
                                    R_star=None, R_d=None, dx=1.0):
    """Create substrate with solid fraction for volume penalization.

    Returns (solid, fraction) where:
      solid: bool array -- True for fully solid nodes (fraction >= 0.5)
      fraction: float32 array -- 0=fluid, 1=solid, smooth transition at boundaries

    The smooth transition is computed analytically from the signed distance
    to the surface, giving sub-cell geometric resolution without increasing
    the grid size.
    """
    if nz == 1:
        return _create_fraction_2d(nx, ny, substrate_type, R_star, R_d, dx)
    else:
        return _create_fraction_3d(nx, ny, nz, substrate_type, R_star, R_d, dx)


def _create_fraction_3d(nx, ny, nz, substrate_type, R_star, R_d, dx):
    """3D solid fraction field."""
    fraction = np.zeros((nx, ny, nz), dtype=np.float32)

    # Bottom wall at k=0: fully solid
    fraction[:, :, 0] = 1.0

    if substrate_type == "flat" or R_star is None or R_d is None:
        solid = fraction >= 0.5
        return solid, fraction

    R_g = abs(R_star) * R_d / dx
    cx, cy = nx / 2.0, ny / 2.0
    i, j, k = np.ogrid[:nx, :ny, :nz]

    if substrate_type == "ridge":
        # Cylindrical ridge along y-axis
        # Signed distance to cylinder surface: positive inside
        dist = np.sqrt((i - cx) ** 2.0 + k.astype(float) ** 2.0)
        signed_dist = R_g - dist
        frac_ridge = np.clip(signed_dist + 0.5, 0.0, 1.0)
        # Only above bottom wall (k > 0)
        above_floor = (k > 0)
        fraction = np.where(above_floor, np.maximum(fraction, frac_ridge),
                            fraction)

    elif substrate_type == "convex":
        dist = np.sqrt((i - cx) ** 2.0 + (j - cy) ** 2.0
                       + (k - R_g) ** 2.0)
        signed_dist = R_g - dist
        frac_convex = np.clip(signed_dist + 0.5, 0.0, 1.0)
        fraction = np.maximum(fraction, frac_convex)

    elif substrate_type == "concave":
        dist = np.sqrt((i - cx) ** 2.0 + (j - cy) ** 2.0
                       + (k - R_g) ** 2.0)
        signed_dist = dist - R_g  # positive outside sphere (in solid walls)
        frac_concave = np.clip(signed_dist + 0.5, 0.0, 1.0)
        below_equator = (k < R_g)
        fraction = np.where(below_equator,
                            np.maximum(fraction, frac_concave), fraction)

    solid = fraction >= 0.5
    return solid, fraction


def _create_fraction_2d(nx, ny, substrate_type, R_star, R_d, dx):
    """2D solid fraction field."""
    fraction = np.zeros((nx, ny), dtype=np.float32)

    # Bottom wall at j=0
    fraction[:, 0] = 1.0

    if substrate_type == "flat" or R_star is None or R_d is None:
        solid = fraction >= 0.5
        return solid, fraction

    R_g = abs(R_star) * R_d / dx
    cx = nx / 2.0

    i = np.arange(nx)
    j = np.arange(ny)
    I, J = np.meshgrid(i, j, indexing='ij')

    if substrate_type == "convex":
        dist = np.sqrt((I - cx) ** 2.0 + (J - R_g) ** 2.0)
        signed_dist = R_g - dist
        frac_convex = np.clip(signed_dist + 0.5, 0.0, 1.0)
        fraction = np.maximum(fraction, frac_convex)

    elif substrate_type == "concave":
        dist = np.sqrt((I - cx) ** 2.0 + (J - R_g) ** 2.0)
        signed_dist = dist - R_g  # positive outside sphere (in solid walls)
        frac_concave = np.clip(signed_dist + 0.5, 0.0, 1.0)
        below_equator = (J < R_g)
        fraction = np.where(below_equator,
                            np.maximum(fraction, frac_concave), fraction)

    elif substrate_type == "ridge":
        dist = np.sqrt((I - cx) ** 2.0 + J.astype(float) ** 2.0)
        signed_dist = R_g - dist
        frac_ridge = np.clip(signed_dist + 0.5, 0.0, 1.0)
        above_floor = (J > 0)
        fraction = np.where(above_floor,
                            np.maximum(fraction, frac_ridge), fraction)

    solid = fraction >= 0.5
    return solid, fraction
