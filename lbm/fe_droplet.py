"""Order parameter (composition) initialization for free-energy LBM.

Creates a smooth C-field (order parameter) for a droplet with tanh profile:
  C(r) = 0.5 + 0.5 * tanh(2 * (R_drop - |r - center|) / xi)

Inside droplet: C = 1 (liquid)
Outside droplet: C = 0 (gas)
Interface: smooth transition over ~2*xi lattice units

Also creates density and velocity fields from C.

Supports both 2D (nz=1 → grid_shape (nx,ny), velocity (2,nx,ny))
and 3D (nz>1 → grid_shape (nx,ny,nz), velocity (3,nx,ny,nz)).
"""
import numpy as np


def _is_2d(nz):
    return nz == 1


def create_fe_droplet(nx, ny, nz, center=None, radius=None,
                      xi=5.0, rho_l=828.0, rho_g=1.0):
    """Create initial composition, density, and velocity fields.

    Parameters
    ----------
    nx, ny, nz : int
        Grid dimensions.  nz=1 triggers 2D mode.
    center : tuple, optional
        Droplet center in lattice units.
        2D: (cx, cy), 3D: (cx, cy, cz).
    radius : float, optional
        Droplet radius in lattice units.
    xi : float
        Interface thickness in lattice units.
    rho_l, rho_g : float
        Liquid and gas densities.

    Returns
    -------
    C : ndarray, float32
        Composition field (1=liquid, 0=gas).
        Shape: (nx, ny) in 2D, (nx, ny, nz) in 3D.
    rho : ndarray, float32
        Density field (same shape as C).
    u : ndarray, float32
        Velocity field (zero initially).
        Shape: (2, nx, ny) in 2D, (3, nx, ny, nz) in 3D.
    """
    is2d = _is_2d(nz)
    ndim = 2 if is2d else 3

    if radius is None:
        radius = min(nx, ny) * 0.15 if is2d else min(nx, ny, nz) * 0.15
    if center is None:
        if is2d:
            center = (nx / 2.0, ny * 0.6)
        else:
            center = (nx / 2.0, ny / 2.0, nz * 0.6)

    # Distance field
    grids = [np.arange(nx), np.arange(ny)]
    if not is2d:
        grids.append(np.arange(nz))
    mesh = np.meshgrid(*grids, indexing='ij')

    dist_sq = sum((M - center[d]) ** 2 for d, M in enumerate(mesh))
    dist = np.sqrt(dist_sq)

    # Interface profile matching the Allen-Cahn equilibrium with
    # beta = 12*sigma/xi, kappa = 3*sigma*xi/2:
    #   mu_phi = 0  =>  phi = 0.5*(1 - tanh(2*r/xi))
    # For a droplet: C = 0.5 + 0.5*tanh(2*(R-r)/xi)
    # Interface thickness ~xi (from C=0.01 to C=0.99 over ~2.5*xi)
    C = 0.5 + 0.5 * np.tanh(2.0 * (radius - dist) / xi)
    C = np.clip(C, 0.0, 1.0).astype(np.float32)

    # Density from composition
    rho = (C * rho_l + (1.0 - C) * rho_g).astype(np.float32)

    # Zero velocity
    u = np.zeros((ndim,) + C.shape, dtype=np.float32)

    return C, rho, u


def create_fe_droplet_with_impact(nx, ny, nz, center=None, radius=None,
                                   xi=5.0, rho_l=828.0, rho_g=1.0,
                                   u_impact=None):
    """Create initial fields with impact velocity.

    The impact velocity is applied smoothly inside the droplet using the
    composition field as a mask.

    Parameters
    ----------
    u_impact : tuple or None
        2D: (ux, uy), 3D: (ux, uy, uz). Defaults to zero.
    """
    is2d = _is_2d(nz)
    ndim = 2 if is2d else 3
    if u_impact is None:
        u_impact = (0.0,) * ndim

    C, rho, u = create_fe_droplet(nx, ny, nz, center, radius, xi, rho_l, rho_g)

    # Apply impact velocity smoothly using C as weight
    for d in range(ndim):
        u[d] = u_impact[d] * C

    return C, rho, u


def place_droplet_above_substrate(nx, ny, nz, substrate_type="flat",
                                   R_star=None, R_drop=None,
                                   xi=5.0, rho_l=828.0, rho_g=1.0,
                                   gap=5.0):
    """Place droplet above substrate with proper gap.

    Parameters
    ----------
    substrate_type : str
        "flat", "ridge", or "convex"
    R_star : float or None
        R/R_drop ratio for curved substrate.
    R_drop : float
        Droplet radius in lattice units.
    gap : float
        Gap between substrate top and droplet bottom in lattice units.

    Returns
    -------
    C, rho, u : arrays
    """
    is2d = _is_2d(nz)
    cx = nx / 2.0
    cy = ny / 2.0

    if is2d:
        if substrate_type == "flat" or R_star is None:
            cy_drop = 1.0 + gap + R_drop
        else:
            R_g = R_star * R_drop
            y_surface = 2.0 * R_g - 0.5
            cy_drop = y_surface + gap + R_drop
        center = (cx, cy_drop)
        u_impact = (0.0, -0.05)
    else:
        cz = 1.0 + gap + R_drop if (substrate_type == "flat" or R_star is None) else \
             (2.0 * R_star * R_drop - 0.5 + gap + R_drop)
        center = (cx, cy, cz)
        u_impact = (0.0, 0.0, -0.05)

    return create_fe_droplet_with_impact(
        nx, ny, nz, center=center, radius=R_drop,
        xi=xi, rho_l=rho_l, rho_g=rho_g, u_impact=u_impact
    )


def create_multi_droplet_with_impact(nx, ny, nz, centers, radii,
                                      xi=5.0, rho_l=828.0, rho_g=1.0,
                                      u_impacts=None):
    """Create initial fields with multiple droplets.

    Parameters
    ----------
    nx, ny, nz : int
        Grid dimensions. nz=1 triggers 2D mode.
    centers : list of tuple
        List of center coordinates. 2D: (cx, cy), 3D: (cx, cy, cz).
    radii : list of float
        Radius of each droplet.
    xi : float
        Interface thickness.
    rho_l, rho_g : float
        Liquid and gas densities.
    u_impacts : list of tuple or None
        List of velocity tuples matching ndim. If None, zero velocity.

    Returns
    -------
    C : ndarray, float32
    rho : ndarray, float32
    u : ndarray, float32
        Shape: (2, nx, ny) in 2D, (3, nx, ny, nz) in 3D.
    """
    is2d = _is_2d(nz)
    ndim = 2 if is2d else 3
    n_drops = len(centers)
    if u_impacts is None:
        u_impacts = [(0.0,) * ndim] * n_drops

    # Grid
    grids = [np.arange(nx), np.arange(ny)]
    if not is2d:
        grids.append(np.arange(nz))
    mesh = np.meshgrid(*grids, indexing='ij')

    # Build combined C field by taking element-wise max (handles overlap)
    C = np.zeros([nx, ny] if is2d else [nx, ny, nz], dtype=np.float32)
    u_masks = []  # (C_mask, u_impact) pairs

    for k, (center, radius) in enumerate(zip(centers, radii)):
        dist_sq = sum((M - center[d]) ** 2 for d, M in enumerate(mesh))
        dist = np.sqrt(dist_sq)
        C_k = 0.5 + 0.5 * np.tanh(2.0 * (radius - dist) / xi)
        C_k = np.clip(C_k, 0.0, 1.0).astype(np.float32)
        # Use max to handle overlapping interfaces correctly
        C = np.maximum(C, C_k)
        u_masks.append((C_k, u_impacts[k]))

    C = np.clip(C, 0.0, 1.0)
    rho = (C * rho_l + (1.0 - C) * rho_g).astype(np.float32)

    # Velocity: each droplet contributes its impact velocity weighted by its own C
    u = np.zeros((ndim,) + C.shape, dtype=np.float32)
    for C_k, u_imp in u_masks:
        for d in range(ndim):
            u[d] += u_imp[d] * C_k

    return C, rho, u
