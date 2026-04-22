"""Measurement utilities for free-energy phase-field droplet simulations.

Provides functions to extract quantitative metrics from the composition
field C: spread factors (Dx, Dy, Dz, ridge ratio k), volume, and
contact angle estimation at the solid wall.
"""
import torch
import numpy as np


def measure_spread_factor(C: torch.Tensor,
                          solid_mask: torch.Tensor = None) -> dict:
    """Compute spread factors Dx, Dy, [Dz] from the composition field.

    The interface is identified by the C > 0.5 isosurface.  Spread factors
    measure the extent of the liquid phase along each axis.

    Works for both 2D (nx, ny) and 3D (nx, ny, nz) arrays.

    Parameters
    ----------
    C : Tensor, shape (nx, ny) or (nx, ny, nz)
        Composition field (1 = liquid, 0 = gas).
    solid_mask : Tensor, optional
        Boolean mask of solid nodes.  Solid nodes are excluded from the
        interface search.

    Returns
    -------
    dict with keys:
        Dx     : float  -- spread in x direction (lattice units)
        Dy     : float  -- spread in y direction
        Dz     : float  -- spread in z direction (0.0 for 2D)
        k      : float  -- ridge ratio Dx / Dy (1.0 for axisymmetric)
        volume : float  -- total liquid volume (sum of C)
        C_max  : float  -- maximum C value
    """
    if isinstance(C, torch.Tensor):
        C_np = C.detach().cpu().numpy()
    else:
        C_np = np.asarray(C)

    if solid_mask is not None:
        if isinstance(solid_mask, torch.Tensor):
            mask_np = solid_mask.detach().cpu().numpy()
        else:
            mask_np = np.asarray(solid_mask)
        # Mask out solid nodes
        C_clean = C_np.copy()
        C_clean[mask_np] = 0.0
    else:
        C_clean = C_np

    ndim = C_clean.ndim

    # Interface nodes: C > 0.5
    interface = C_clean > 0.5

    volume = float(C_clean.sum())
    C_max = float(C_clean.max())

    if not interface.any():
        return {
            "Dx": 0.0, "Dy": 0.0, "Dz": 0.0,
            "k": 0.0, "volume": volume, "C_max": C_max,
        }

    # Find extent along each axis
    coords = np.argwhere(interface)  # (N, ndim) with columns x, y[, z]
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)

    Dx = float(maxs[0] - mins[0] + 1)
    Dy = float(maxs[1] - mins[1] + 1)
    Dz = float(maxs[2] - mins[2] + 1) if ndim == 3 else 0.0

    k = Dx / Dy if Dy > 0 else 0.0

    return {
        "Dx": Dx, "Dy": Dy, "Dz": Dz,
        "k": k, "volume": volume, "C_max": C_max,
    }


def measure_contact_angle(C: torch.Tensor,
                          solid_mask: torch.Tensor,
                          axis: str = "z") -> float:
    """Estimate contact angle from the C field at the solid wall.

    Finds the contact line where C ~ 0.5 at the first fluid layer above
    the wall, then computes the angle of the C = 0.5 contour relative to
    the wall normal.

    For 3D (axis='z'): wall at last dimension.
    For 2D (axis='y'): wall at last dimension (y).

    Parameters
    ----------
    C : Tensor, shape (nx, ny) or (nx, ny, nz)
        Composition field.
    solid_mask : Tensor, same shape as C
        Boolean solid mask (True = solid).
    axis : str
        Wall normal axis ('z' for 3D wall, 'y' for 2D wall).

    Returns
    -------
    float
        Estimated contact angle in degrees.  Returns 180.0 if no
        contact line is found (no wetting).
    """
    if isinstance(C, torch.Tensor):
        C_np = C.detach().cpu().numpy()
    else:
        C_np = np.asarray(C)

    if isinstance(solid_mask, torch.Tensor):
        mask_np = solid_mask.detach().cpu().numpy()
    else:
        mask_np = np.asarray(solid_mask)

    ndim = C_np.ndim
    shape = C_np.shape

    if ndim == 2:
        return _measure_contact_angle_2d(C_np, mask_np)
    else:
        return _measure_contact_angle_3d(C_np, mask_np)


def _measure_contact_angle_2d(C_np, mask_np):
    """Contact angle estimation for 2D simulations.

    Wall is at y=0 (first dimension is x, second is y).
    """
    nx, ny = C_np.shape

    # Find the first fluid row above the solid wall
    y_wall_top = 0
    for y in range(ny):
        if not mask_np[:, y].all():
            y_wall_top = y
            break

    y_fluid = y_wall_top + 1
    if y_fluid >= ny:
        return 180.0

    # Get the C field at the first fluid layer: shape (nx,)
    C_wall = C_np[:, y_fluid]

    # Find contact region: columns where C > 0.5
    contact_mask = C_wall > 0.5
    if not contact_mask.any():
        return 180.0

    cx = nx / 2.0
    contact_x = np.argwhere(contact_mask).flatten()
    radii = np.abs(contact_x - cx)
    idx_outer = np.argmax(radii)
    r_contact = radii[idx_outer]

    if r_contact < 1.0:
        return 180.0

    x_c = contact_x[idx_outer]
    C_column = C_np[x_c, :]  # (ny,)

    # Find y where C crosses 0.5
    y_interface = None
    for y in range(y_fluid, ny - 1):
        if C_column[y] >= 0.5 and C_column[y + 1] < 0.5:
            dy = (C_column[y] - 0.5) / (C_column[y] - C_column[y + 1] + 1e-10)
            y_interface = y + dy
            break

    if y_interface is None:
        return 180.0

    dy = y_interface - y_wall_top
    if dy < 0.5:
        dy = 0.5

    theta_rad = np.arctan2(r_contact, dy)
    return np.degrees(theta_rad)


def _measure_contact_angle_3d(C_np, mask_np):
    """Contact angle estimation for 3D simulations.

    Wall is at z=0. Original implementation preserved.
    """
    nx, ny, nz = C_np.shape

    # Find the first fluid layer above the solid wall
    z_wall_top = 0
    for z in range(nz):
        if not mask_np[:, :, z].all():
            z_wall_top = z
            break

    z_fluid = z_wall_top + 1
    if z_fluid >= nz:
        return 180.0

    C_wall = C_np[:, :, z_fluid]

    contact_mask = C_wall > 0.5
    if not contact_mask.any():
        return 180.0

    cx, cy = nx / 2.0, ny / 2.0

    contact_coords = np.argwhere(contact_mask)
    radii = np.sqrt((contact_coords[:, 0] - cx) ** 2 +
                    (contact_coords[:, 1] - cy) ** 2)
    idx_outer = np.argmax(radii)
    r_contact = radii[idx_outer]

    if r_contact < 1.0:
        return 180.0

    x_c, y_c = contact_coords[idx_outer]
    C_column = C_np[x_c, y_c, :]

    z_interface = None
    for z in range(z_fluid, nz - 1):
        if C_column[z] >= 0.5 and C_column[z + 1] < 0.5:
            dz = (C_column[z] - 0.5) / (C_column[z] - C_column[z + 1] + 1e-10)
            z_interface = z + dz
            break

    if z_interface is None:
        return 180.0

    dz = z_interface - z_wall_top
    if dz < 0.5:
        dz = 0.5

    theta_rad = np.arctan2(r_contact, dz)
    return np.degrees(theta_rad)
