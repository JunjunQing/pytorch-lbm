"""Order parameter (composition) initialization for free-energy LBM.

Creates a smooth C-field (order parameter) for a droplet with tanh profile:
  C(r) = 0.5 + 0.5 * tanh(2 * (R_drop - |r - center|) / xi)

Inside droplet: C = 1 (liquid)
Outside droplet: C = 0 (gas)
Interface: smooth transition over ~2*xi lattice units

Also creates density and velocity fields from C.
"""
import numpy as np


def create_fe_droplet(nx, ny, nz, center=None, radius=None,
                      xi=5.0, rho_l=828.0, rho_g=1.0):
    """Create initial composition, density, and velocity fields.

    Parameters
    ----------
    nx, ny, nz : int
        Grid dimensions.
    center : tuple, optional
        Droplet center (cx, cy, cz) in lattice units.
    radius : float, optional
        Droplet radius in lattice units.
    xi : float
        Interface thickness in lattice units.
    rho_l, rho_g : float
        Liquid and gas densities.

    Returns
    -------
    C : ndarray (nx, ny, nz), float32
        Composition field (1=liquid, 0=gas).
    rho : ndarray (nx, ny, nz), float32
        Density field.
    u : ndarray (3, nx, ny, nz), float32
        Velocity field (zero initially).
    """
    if radius is None:
        radius = min(nx, ny, nz) * 0.15
    if center is None:
        center = (nx / 2.0, ny / 2.0, nz * 0.6)

    # 3D distance field
    x = np.arange(nx)
    y = np.arange(ny)
    z = np.arange(nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    dist = np.sqrt((X - center[0])**2 + (Y - center[1])**2 + (Z - center[2])**2)

    # Interface profile matching the Allen-Cahn equilibrium with
    # beta = 12*sigma/xi, kappa = 3*sigma*xi/2:
    #   mu_phi = 0  =>  phi = 0.5*(1 - tanh(2*r/ξ))
    # For a droplet: C = 0.5 + 0.5*tanh(2*(R-r)/xi)
    # Interface thickness ~xi (from C=0.01 to C=0.99 over ~2.5*xi)
    C = 0.5 + 0.5 * np.tanh(2.0 * (radius - dist) / xi)
    C = np.clip(C, 0.0, 1.0).astype(np.float32)

    # Density from composition
    rho = (C * rho_l + (1.0 - C) * rho_g).astype(np.float32)

    # Zero velocity
    u = np.zeros((3, nx, ny, nz), dtype=np.float32)

    return C, rho, u


def create_fe_droplet_with_impact(nx, ny, nz, center=None, radius=None,
                                   xi=5.0, rho_l=828.0, rho_g=1.0,
                                   u_impact=(0.0, 0.0, -0.05)):
    """Create initial fields with impact velocity.

    The impact velocity is applied smoothly inside the droplet using the
    composition field as a mask.
    """
    C, rho, u = create_fe_droplet(nx, ny, nz, center, radius, xi, rho_l, rho_g)

    # Apply impact velocity smoothly using C as weight
    for d in range(3):
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
    cx = nx / 2.0
    cy = ny / 2.0

    if substrate_type == "flat" or R_star is None:
        cz = 1.0 + gap + R_drop
    else:
        R_g = R_star * R_drop
        # Top of ridge/convex at center
        z_surface = 2.0 * R_g - 0.5
        cz = z_surface + gap + R_drop

    return create_fe_droplet_with_impact(
        nx, ny, nz, center=(cx, cy, cz), radius=R_drop,
        xi=xi, rho_l=rho_l, rho_g=rho_g
    )


def create_multi_droplet_with_impact(nx, ny, nz, centers, radii,
                                      xi=5.0, rho_l=828.0, rho_g=1.0,
                                      u_impacts=None):
    """Create initial fields with multiple droplets.

    Parameters
    ----------
    nx, ny, nz : int
        Grid dimensions.
    centers : list of tuple
        List of (cx, cy, cz) for each droplet.
    radii : list of float
        Radius of each droplet.
    xi : float
        Interface thickness.
    rho_l, rho_g : float
        Liquid and gas densities.
    u_impacts : list of tuple or None
        List of (ux, uy, uz) for each droplet. If None, zero velocity.

    Returns
    -------
    C : ndarray (nx, ny, nz), float32
    rho : ndarray (nx, ny, nz), float32
    u : ndarray (3, nx, ny, nz), float32
    """
    n_drops = len(centers)
    if u_impacts is None:
        u_impacts = [(0.0, 0.0, 0.0)] * n_drops

    # Build combined C field by taking element-wise max (handles overlap)
    C = np.zeros((nx, ny, nz), dtype=np.float32)
    u_masks = []  # (C_mask, u_impact) pairs

    x = np.arange(nx)
    y = np.arange(ny)
    z = np.arange(nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    for k, (center, radius) in enumerate(zip(centers, radii)):
        dist = np.sqrt((X - center[0])**2 + (Y - center[1])**2 + (Z - center[2])**2)
        C_k = 0.5 + 0.5 * np.tanh(2.0 * (radius - dist) / xi)
        C_k = np.clip(C_k, 0.0, 1.0).astype(np.float32)
        # Use max to handle overlapping interfaces correctly
        C = np.maximum(C, C_k)
        u_masks.append((C_k, u_impacts[k]))

    C = np.clip(C, 0.0, 1.0)
    rho = (C * rho_l + (1.0 - C) * rho_g).astype(np.float32)

    # Velocity: each droplet contributes its impact velocity weighted by its own C
    u = np.zeros((3, nx, ny, nz), dtype=np.float32)
    for C_k, u_imp in u_masks:
        for d in range(3):
            u[d] += u_imp[d] * C_k

    return C, rho, u
