"""Measurement utilities for free-energy phase-field droplet simulations.

Provides functions to extract quantitative metrics from the composition
field C: spread factors (Dx, Dy, Dz, ridge ratio k), volume, and
contact angle estimation at the solid wall.
"""
import torch
import numpy as np


def measure_spread_factor(C: torch.Tensor,
                          solid_mask: torch.Tensor = None) -> dict:
    """Compute spread factors Dx, Dy, Dz from the composition field.

    The interface is identified by the C > 0.5 isosurface.  Spread factors
    measure the extent of the liquid phase along each axis.

    Parameters
    ----------
    C : Tensor, shape (nx, ny, nz)
        Composition field (1 = liquid, 0 = gas).
    solid_mask : Tensor, shape (nx, ny, nz), optional
        Boolean mask of solid nodes.  Solid nodes are excluded from the
        interface search.

    Returns
    -------
    dict with keys:
        Dx     : float  -- spread in x direction (lattice units)
        Dy     : float  -- spread in y direction
        Dz     : float  -- spread in z direction
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
    coords = np.argwhere(interface)  # (N, 3) with columns x, y, z
    x_min, y_min, z_min = coords.min(axis=0)
    x_max, y_max, z_max = coords.max(axis=0)

    Dx = float(x_max - x_min + 1)
    Dy = float(y_max - y_min + 1)
    Dz = float(z_max - z_min + 1)

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

    For a 3D domain with a flat wall at z = 0 (solid at z = 0), the
    contact angle is estimated by:
      1. Locate the first fluid row (z = 1 typically)
      2. Find columns where C > 0.5 at that row (contact region)
      3. Measure the gradient of C in z vs. radial direction
      4. theta = arctan(|dr/dz|) at the contact line

    Parameters
    ----------
    C : Tensor, shape (nx, ny, nz)
        Composition field.
    solid_mask : Tensor, shape (nx, ny, nz)
        Boolean solid mask (True = solid).
    axis : str
        Wall normal axis ('z' for wall at z = 0).

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

    nx, ny, nz = C_np.shape

    # Find the first fluid layer above the solid wall
    # Assume wall at z = 0 (solid_mask[:, :, 0] = True)
    # Find z_wall_top: first z where solid_mask is False
    z_wall_top = 0
    for z in range(nz):
        if not mask_np[:, :, z].all():
            z_wall_top = z
            break

    # The first fluid layer is one above the wall top
    z_fluid = z_wall_top + 1
    if z_fluid >= nz:
        return 180.0

    # Get the C field at the first fluid layer: shape (nx, ny)
    C_wall = C_np[:, :, z_fluid]

    # Find contact line region: columns where C > 0.5
    contact_mask = C_wall > 0.5
    if not contact_mask.any():
        return 180.0

    # Center of the domain
    cx, cy = nx / 2.0, ny / 2.0

    # Find the outermost contact point (farthest from center where C > 0.5)
    contact_coords = np.argwhere(contact_mask)  # (N, 2) columns x, y
    radii = np.sqrt((contact_coords[:, 0] - cx) ** 2 +
                    (contact_coords[:, 1] - cy) ** 2)
    idx_outer = np.argmax(radii)
    r_contact = radii[idx_outer]

    if r_contact < 1.0:
        return 180.0

    # Estimate the slope of the interface at the contact line.
    # Sample C along z at the outermost contact column.
    x_c, y_c = contact_coords[idx_outer]
    C_column = C_np[x_c, y_c, :]  # (nz,)

    # Find z where C crosses 0.5 (the interface height at the contact point)
    z_interface = None
    for z in range(z_fluid, nz - 1):
        if C_column[z] >= 0.5 and C_column[z + 1] < 0.5:
            # Linear interpolation
            dz = (C_column[z] - 0.5) / (C_column[z] - C_column[z + 1] + 1e-10)
            z_interface = z + dz
            break

    if z_interface is None:
        return 180.0

    # The contact angle is the angle between the interface and the wall.
    # Using the geometry: tan(theta) = r_contact / (z_interface - z_wall_top)
    # theta is measured from the wall surface (0 = complete wetting, 180 = no wetting)
    dz = z_interface - z_wall_top
    if dz < 0.5:
        dz = 0.5

    # For a spherical cap, the angle from the wall is:
    # theta = arctan(r_contact / dz) gives the half-angle from vertical.
    # Contact angle (from wall) = 90 + arctan(dz/r_contact) - 90 = arctan(r_contact/dz)
    # More precisely, for a spherical cap geometry:
    #   tan(theta) = r / h where theta is measured from the substrate
    theta_rad = np.arctan2(r_contact, dz)
    theta_deg = np.degrees(theta_rad)

    return theta_deg
