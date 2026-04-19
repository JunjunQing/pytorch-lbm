"""Chemical potential and macroscopic quantity computation for free-energy LBM.

Implements the core thermodynamic functions for the Lee & Liu (2010) free-energy
phase-field model:

  Bulk free energy:       E_0(C) = beta * C^2 * (C - 1)^2
  Chemical potential:     mu_0 = dE_0/dC = 4 * beta * C * (C - 1) * (C - 0.5)
  Full chemical potential: mu = mu_0 - kappa * laplacian(C)
  Artificial free energy: E_A(C) = beta_A * C^2  (active when C < 0)
  Modified chemical pot.:  mu_hat = mu_0 + dE_A/dC - kappa * laplacian(C)

Macroscopic quantities:
  Density:       rho = C * rho_l + (1 - C) * rho_g
  Relaxation:    1/tau = C / tau_l + (1 - C) / tau_g
  Viscosity:     eta = rho * tau * cs^2

References
----------
Lee, T. & Liu, L. (2010). Lattice Boltzmann simulations of micron-scale
drop impact on dry surfaces. J. Comput. Phys., 229, 8045--8063.
"""
import torch


def compute_bulk_chemical_potential(C: torch.Tensor, beta: float) -> torch.Tensor:
    """Compute bulk chemical potential mu_0 = dE_0/dC.

    E_0 = beta * C^2 * (C - 1)^2
    mu_0 = 4 * beta * C * (C - 1) * (C - 0.5)

    Parameters
    ----------
    C : Tensor, shape (*grid_shape)
        Composition field (order parameter). C=1 liquid, C=0 gas.
    beta : float
        Free energy parameter.

    Returns
    -------
    mu_0 : Tensor, shape (*grid_shape)
        Bulk chemical potential.
    """
    return 4.0 * beta * C * (C - 1.0) * (C - 0.5)


def compute_artificial_correction(C: torch.Tensor,
                                  beta_A: float) -> torch.Tensor:
    """Compute the artificial free-energy correction dE_A/dC.

    E_A = beta_A * C^2 is only active when C < 0, preventing unphysical
    negative composition values that can arise from numerical errors.

        dE_A/dC = 2 * beta_A * C   (when C < 0, else 0)

    Parameters
    ----------
    C : Tensor, shape (*grid_shape)
        Composition field.
    beta_A : float
        Artificial free-energy coefficient.

    Returns
    -------
    dE_A : Tensor, shape (*grid_shape)
        Artificial free-energy correction (zero where C >= 0).
    """
    return torch.where(C < 0.0, 2.0 * beta_A * C, torch.zeros_like(C))


def compute_chemical_potential(C: torch.Tensor,
                               laplacian_C: torch.Tensor,
                               kappa: float,
                               beta: float,
                               beta_A: float = 0.25) -> torch.Tensor:
    """Compute modified chemical potential mu_hat.

    mu_hat = mu_0 + dE_A/dC - kappa * laplacian(C)

    The artificial free energy E_A = beta_A * C^2 prevents negative C
    and is only active when C < 0.

    Parameters
    ----------
    C : Tensor, shape (*grid_shape)
        Composition field.
    laplacian_C : Tensor, shape (*grid_shape)
        Laplacian of C (from IsotropicGradient.laplacian).
    kappa : float
        Interface stiffness.
    beta : float
        Free energy parameter.
    beta_A : float
        Artificial free energy coefficient (default 0.25).

    Returns
    -------
    mu_hat : Tensor, shape (*grid_shape)
        Modified chemical potential.
    """
    mu_0 = compute_bulk_chemical_potential(C, beta)
    dE_A = compute_artificial_correction(C, beta_A)
    mu_hat = mu_0 + dE_A - kappa * laplacian_C
    return mu_hat


def compute_density(C: torch.Tensor,
                    rho_l: float,
                    rho_g: float) -> torch.Tensor:
    """Compute mixture density from composition.

    rho = C * rho_l + (1 - C) * rho_g

    Parameters
    ----------
    C : Tensor, shape (*grid_shape)
        Composition field.
    rho_l, rho_g : float
        Liquid and gas densities.

    Returns
    -------
    rho : Tensor, shape (*grid_shape)
        Mixture density.
    """
    return C * rho_l + (1.0 - C) * rho_g


def compute_tau(C: torch.Tensor,
                tau_l: float,
                tau_g: float) -> torch.Tensor:
    """Compute composition-dependent relaxation time (LINEAR interpolation).

    tau = C * tau_l + (1 - C) * tau_g

    This matches the Palabos implementation and Lee & Liu 2010.
    The collision then uses omega = 1/(tau + 0.5) for stability.

    Parameters
    ----------
    C : Tensor, shape (*grid_shape)
        Composition field.
    tau_l, tau_g : float
        Liquid and gas relaxation times.

    Returns
    -------
    tau : Tensor, shape (*grid_shape)
        Local relaxation time.
    """
    C_safe = torch.clamp(C, 0.0, 1.0)
    return C_safe * tau_l + (1.0 - C_safe) * tau_g


def compute_tau_factor(C: torch.Tensor,
                       tau_l: float,
                       tau_g: float) -> torch.Tensor:
    """Compute collision omega = 1/(tau + 0.5) with linear tau interpolation.

    tau = C * tau_l + (1 - C) * tau_g
    omega = 1 / (tau + 0.5)

    This matches the Palabos HeLeeProcessor3D implementation exactly.

    Parameters
    ----------
    C : Tensor, shape (*grid_shape)
        Composition field.
    tau_l, tau_g : float
        Liquid and gas relaxation times.

    Returns
    -------
    omega : Tensor, shape (*grid_shape)
        Collision relaxation factor 1/(tau + 0.5).
    """
    tau = compute_tau(C, tau_l, tau_g)
    return 1.0 / (tau + 0.5)


def compute_tau_inv(C: torch.Tensor,
                    tau_l: float,
                    tau_g: float) -> torch.Tensor:
    """Compute 1/tau = C/tau_l + (1-C)/tau_g directly (more stable for BGK).

    Parameters
    ----------
    C : Tensor, shape (*grid_shape)
        Composition field.
    tau_l, tau_g : float
        Liquid and gas relaxation times.

    Returns
    -------
    tau_inv : Tensor, shape (*grid_shape)
        Inverse relaxation time.
    """
    C_safe = torch.clamp(C, 0.0, 1.0)
    return C_safe / max(tau_l, 0.501) + (1.0 - C_safe) / max(tau_g, 0.501)


def compute_viscosity(rho: torch.Tensor,
                      tau: torch.Tensor,
                      cs2: float = 1.0 / 3.0) -> torch.Tensor:
    """Compute dynamic viscosity eta = rho * tau * cs^2.

    Parameters
    ----------
    rho : Tensor, shape (*grid_shape)
        Density field.
    tau : Tensor, shape (*grid_shape)
        Relaxation time field.
    cs2 : float
        Speed of sound squared (default 1/3).

    Returns
    -------
    eta : Tensor, shape (*grid_shape)
        Dynamic viscosity.
    """
    return rho * tau * cs2
