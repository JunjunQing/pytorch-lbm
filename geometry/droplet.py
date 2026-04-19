"""Droplet initialization for LBM.

Creates a smooth density field for a droplet (high density region)
surrounded by gas (low density). Supports curved substrate-aware placement.
"""
import numpy as np


def create_droplet(nx, ny, nz=1, center=None, radius=None,
                   rho_l=1.0, rho_g=0.1, dx=1.0, width=3.0,
                   substrate_height=None, gap=None, u_impact=None):
    """Create initial density field with a droplet.

    Uses tanh profile for smooth interface:
        rho(x) = rho_g + (rho_l - rho_g) * 0.5 * (1 + tanh((R - |x - c|) / (w*sqrt(2))))

    Parameters
    ----------
    nx, ny, nz : int
    center : tuple, optional
        Droplet center in physical units. Default: domain center.
        For 2D: (cx, cy), for 3D: (cx, cy, cz).
    radius : float, optional
        Droplet radius in physical units.
    rho_l, rho_g : float
        Liquid and gas densities.
    dx : float
        Grid spacing.
    width : float
        Interface width in lattice units (number of cells).
    substrate_height : ndarray, optional
        Surface height from substrate_surface_height(). If given, droplet
        is placed above the substrate with `gap` lattice units of clearance.
        Overrides the y (2D) or z (3D) component of `center`.
    gap : float, optional
        Gap between substrate surface and droplet bottom in lattice units.
        Default: 5.
    u_impact : tuple or None, optional
        Impact velocity in lattice units for the droplet interior.
        E.g. (0, -0.01) for 2D, (0, 0, -0.01) for 3D.
        Only applied inside the droplet (where rho > (rho_l+rho_g)/2).
        When provided, returns (rho, u) tuple instead of just rho.

    Returns
    -------
    rho : ndarray, shape (nx, ny[, nz])
        Initial density field.
    u : ndarray, shape (dim, nx, ny[, nz]), only when u_impact is given
        Initial velocity field.
    """
    if nz == 1:
        rho = _create_droplet_2d(nx, ny, center, radius, rho_l, rho_g,
                                  dx, width, substrate_height, gap)
        dim = 2
    else:
        rho = _create_droplet_3d(nx, ny, nz, center, radius, rho_l, rho_g,
                                  dx, width, substrate_height, gap)
        dim = 3

    if u_impact is not None:
        u = _create_impact_velocity(rho, u_impact, rho_l, rho_g, dim)
        return rho, u
    return rho


def _create_droplet_2d(nx, ny, center, radius, rho_l, rho_g, dx, width,
                       substrate_height, gap):
    """2D droplet with optional curved substrate awareness."""
    x = np.arange(nx) * dx
    y = np.arange(ny) * dx

    if radius is None:
        radius = min(nx, ny) * dx * 0.2

    if substrate_height is not None:
        # Find substrate surface at center column, place droplet above it
        mid = nx // 2
        sub_top_y = substrate_height[mid]
        if gap is None:
            gap = 5.0
        drop_bottom = (sub_top_y + gap) * dx
        cy = drop_bottom + radius
        cx = nx * dx / 2 if center is None else center[0]
        center = (cx, cy)
    elif center is None:
        center = (nx * dx / 2, ny * dx * 0.6)

    X, Y = np.meshgrid(x, y, indexing='ij')
    dist = np.sqrt((X - center[0])**2 + (Y - center[1])**2)

    phi = 0.5 * (1.0 + np.tanh((radius - dist) / (width * dx * np.sqrt(2.0))))
    rho = rho_g + (rho_l - rho_g) * phi
    return rho.astype(np.float32)


def _create_droplet_3d(nx, ny, nz, center, radius, rho_l, rho_g, dx, width,
                       substrate_height, gap):
    """3D droplet with optional curved substrate awareness."""
    x = np.arange(nx) * dx
    y = np.arange(ny) * dx
    z = np.arange(nz) * dx

    if radius is None:
        radius = min(nx, ny, nz) * dx * 0.2

    if substrate_height is not None:
        mid_x, mid_y = nx // 2, ny // 2
        sub_top_z = substrate_height[mid_x, mid_y]
        if gap is None:
            gap = 5.0
        drop_bottom = (sub_top_z + gap) * dx
        cz = drop_bottom + radius
        cx = nx * dx / 2 if center is None else center[0]
        cy = ny * dx / 2 if center is None else center[1]
        center = (cx, cy, cz)
    elif center is None:
        center = (nx * dx / 2, ny * dx / 2, nz * dx * 0.6)

    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    dist = np.sqrt((X - center[0])**2 + (Y - center[1])**2 + (Z - center[2])**2)

    phi = 0.5 * (1.0 + np.tanh((radius - dist) / (width * dx * np.sqrt(2.0))))
    rho = rho_g + (rho_l - rho_g) * phi
    return rho.astype(np.float32)


def create_mc_droplet(nx, ny, nz=1, center=None, radius=None,
                      rho1_l=1.0, rho1_g=0.02,
                      rho2_l=0.02, rho2_g=1.0,
                      dx=1.0, width=3.0,
                      substrate_height=None, gap=None):
    """Create initial density fields for two-component MC model.

    Component 1 (heavy): high density inside droplet, low outside.
    Component 2 (light): low density inside droplet, high outside.

    Uses the same tanh profile as create_droplet.

    Returns
    -------
    rho1, rho2 : ndarray, shape (nx, ny[, nz])
    """
    # Compute distance field using the same geometry as single-component
    if nz == 1:
        rho1 = _create_droplet_2d(nx, ny, center, radius, rho1_l, rho1_g,
                                   dx, width, substrate_height, gap)
    else:
        rho1 = _create_droplet_3d(nx, ny, nz, center, radius, rho1_l, rho1_g,
                                   dx, width, substrate_height, gap)

    # Compute phi (0-1 mask of inside droplet) from rho1
    phi = np.clip((rho1 - rho1_g) / (rho1_l - rho1_g + 1e-10), 0, 1)

    # Component 2 is inverted: high outside, low inside
    # phi=1 inside => rho2=rho2_l (low), phi=0 outside => rho2=rho2_g (high)
    rho2 = rho2_l + (rho2_g - rho2_l) * (1.0 - phi)

    return rho1.astype(np.float32), rho2.astype(np.float32)


def _create_impact_velocity(rho, u_impact, rho_l, rho_g, dim):
    """Create velocity field with impact velocity inside the droplet.

    Parameters
    ----------
    rho : ndarray
        Density field from droplet creation.
    u_impact : tuple
        Velocity components in lattice units, e.g. (0, -0.01) for 2D.
    rho_l, rho_g : float
        Liquid and gas densities.
    dim : int
        Dimensionality (2 or 3).

    Returns
    -------
    u : ndarray, shape (dim, *rho.shape)
    """
    assert len(u_impact) == dim, f"u_impact has {len(u_impact)} components, expected {dim}"

    # Smooth mask: same tanh shape as density transition
    alpha = np.clip((rho - rho_g) / (rho_l - rho_g + 1e-10), 0, 1)

    shape = (dim,) + rho.shape
    u = np.zeros(shape, dtype=np.float32)
    for d in range(dim):
        u[d] = u_impact[d] * alpha

    return u
