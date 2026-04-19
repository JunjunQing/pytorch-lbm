"""Configuration for free-energy phase-field LBM simulations (Lee & Liu 2010).

Implements the two-distribution approach:
  - g-distribution: Navier-Stokes (momentum equation)
  - h-distribution: Cahn-Hilliard (phase-field / order parameter)

The free-energy functional is:
  Psi = integral [ beta * psi(phi) + (kappa/2) * |grad(phi)|^2 ] dV

where psi(phi) is a double-well potential with bulk free-energy parameter beta,
and kappa controls the interface stiffness (gradient energy).

Key parameter relationships:
  sigma = sqrt(2 * kappa * beta) / 6   (surface tension)
  kappa  = beta * xi^2 / 8              (interface stiffness from thickness)
  M      = mobility for Cahn-Hilliard   (controls interface relaxation)
  cos(theta_eq) = -phi_c / sqrt(2 * kappa * beta)  (contact angle)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class FEConfig:
    """Free-energy phase-field LBM simulation parameters (Lee & Liu 2010).

    Parameter derivation strategy:
      - Specify sigma and xi -> compute beta and kappa
      - Or specify beta and xi -> compute sigma and kappa

    Use the class method ``from_paper_params`` to build a config that matches
    published (We, Oh, D0) dimensional parameters.

    All quantities are in lattice units unless noted.
    """

    # ------------------------------------------------------------------ #
    # Grid dimensions
    # ------------------------------------------------------------------ #
    nx: int = 200
    ny: int = 200
    nz: int = 1              # nz == 1 means 2D

    # Lattice spacing and time step (physical units, for post-processing)
    dx: float = 1e-6         # [m]
    dt: float = 1e-6         # [s]

    # ------------------------------------------------------------------ #
    # Phase-field / free-energy parameters
    # ------------------------------------------------------------------ #
    rho_l: float = 828.0     # Liquid density  (lattice units)
    rho_g: float = 1.0       # Gas density     (lattice units)
    sigma: float = 0.01      # Surface tension  (lattice units)
    xi: float = 5.0          # Interface thickness (lattice units)

    # beta controls the double-well depth.  When 0, it is auto-computed
    # from sigma and xi at __post_init__ time.
    beta: float = 0.0

    # kappa controls gradient energy.  When 0, auto-computed from beta, xi.
    kappa: float = 0.0

    # Mobility for the Cahn-Hilliard (h-distribution) equation.
    # When 0, defaults to 0.02 / beta.
    M: float = 0.0

    # Artificial free-energy coefficient (Lee 2009 stabilisation term).
    beta_A: float = 0.25

    # ------------------------------------------------------------------ #
    # Relaxation times
    # ------------------------------------------------------------------ #
    # tau_l: liquid relaxation time.  When 0, set from viscosity or Oh.
    tau_l: float = 0.0
    # tau_g: gas relaxation time (usually larger, lower viscosity).
    tau_g: float = 0.6
    # tau_h: relaxation for h-distribution (Allen-Cahn). tau_h=1.0 critical for stability.
    tau_h: float = 1.0

    # ------------------------------------------------------------------ #
    # Wetting / contact angle
    # ------------------------------------------------------------------ #
    theta_eq: float = 162.0  # Equilibrium contact angle [degrees]
    # phi_c: wetting potential.  When 0, computed from theta_eq.
    phi_c: float = 0.0

    # ------------------------------------------------------------------ #
    # Body forces (lattice units)
    # ------------------------------------------------------------------ #
    g_force: Tuple[float, ...] = (0.0, 0.0, 0.0)

    # ------------------------------------------------------------------ #
    # Simulation control
    # ------------------------------------------------------------------ #
    max_steps: int = 10000
    output_interval: int = 100

    # ------------------------------------------------------------------ #
    # Device / precision
    # ------------------------------------------------------------------ #
    device: str = "cuda"
    dtype: str = "float32"
    # Force FFT pressure (constant-density Poisson) instead of variable-density CG.
    # Set True for quick tests; False (default) uses variable-density CG when
    # density ratio > 2, which is needed for ridge impact asymmetry.
    use_fft_pressure: bool = False

    # ------------------------------------------------------------------ #
    # Post-init: derive dependent parameters
    # ------------------------------------------------------------------ #
    def __post_init__(self):
        # --- Derive sigma, beta, kappa from whichever pair is given ---
        #
        # Governing relations:
        #   (1) sigma = sqrt(2 * kappa * beta) / 6
        #   (2) kappa = beta * xi^2 / 8
        #
        # Strategy:
        #   A) sigma and xi given, beta=0, kappa=0:
        #        beta  = 12 * sigma / xi          [from (1)+(2)]
        #        kappa = beta * xi^2 / 8           [from (2)]
        #
        #   B) beta and xi given, sigma=0, kappa=0:
        #        kappa = beta * xi^2 / 8           [from (2)]
        #        sigma = beta * xi / 12            [from (1)+(2)]
        #
        #   C) All three given explicitly: use as-is.
        #
        if self.sigma > 0.0 and self.beta == 0.0:
            # Case A: sigma + xi -> beta, kappa
            self.beta = 12.0 * self.sigma / self.xi
        elif self.sigma == 0.0 and self.beta > 0.0:
            # Case B: beta + xi -> sigma, kappa
            self.sigma = self.beta * self.xi / 12.0

        if self.kappa == 0.0:
            self.kappa = self.beta * self.xi ** 2 / 8.0

        # --- Mobility ---
        if self.M == 0.0:
            self.M = 0.02 / self.beta

        # --- Wetting potential from contact angle ---
        # cos(theta) = -phi_c / sqrt(2 * kappa * beta)
        if self.phi_c == 0.0 and self.theta_eq != 90.0:
            cos_theta = math.cos(math.radians(self.theta_eq))
            self.phi_c = -cos_theta * math.sqrt(2.0 * self.kappa * self.beta)

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #
    @property
    def dim(self) -> int:
        return 2 if self.nz == 1 else 3

    @property
    def grid_shape(self) -> Tuple[int, ...]:
        if self.nz == 1:
            return (self.nx, self.ny)
        return (self.nx, self.ny, self.nz)

    @property
    def density_ratio(self) -> float:
        return self.rho_l / self.rho_g

    # ------------------------------------------------------------------ #
    # Derived quantities
    # ------------------------------------------------------------------ #
    @property
    def nu_l(self) -> float:
        """Kinematic viscosity of the liquid phase (lattice units)."""
        return (self.tau_l - 0.5) / 3.0

    @property
    def nu_g(self) -> float:
        """Kinematic viscosity of the gas phase (lattice units)."""
        return (self.tau_g - 0.5) / 3.0

    @property
    def mu_l(self) -> float:
        """Dynamic viscosity of the liquid (lattice units)."""
        return self.rho_l * self.nu_l

    @property
    def mu_g(self) -> float:
        """Dynamic viscosity of the gas (lattice units)."""
        return self.rho_g * self.nu_g

    # ------------------------------------------------------------------ #
    # Helper: compute tau from Ohnesorge number
    # ------------------------------------------------------------------ #
    @staticmethod
    def compute_tau_from_oh(Oh: float, D0: float, rho_l: float = 828.0,
                            sigma: float = 0.01) -> float:
        """Compute liquid relaxation time from Ohnesorge number.

        Oh = mu_l / sqrt(rho_l * sigma * D0)
        => mu_l = Oh * sqrt(rho_l * sigma * D0)
        => nu_l = mu_l / rho_l
        => tau_l = 3 * nu_l + 0.5

        Parameters
        ----------
        Oh : float
            Ohnesorge number (dimensionless).
        D0 : float
            Droplet diameter in lattice units.
        rho_l : float
            Liquid density in lattice units.
        sigma : float
            Surface tension in lattice units.

        Returns
        -------
        float
            Relaxation time tau_l.
        """
        mu_l = Oh * math.sqrt(rho_l * sigma * D0)
        nu_l = mu_l / rho_l
        tau_l = 3.0 * nu_l + 0.5
        return tau_l

    # ------------------------------------------------------------------ #
    # Class method: create config from paper (dimensionless) parameters
    # ------------------------------------------------------------------ #
    @classmethod
    def from_paper_params(
        cls,
        We: float,
        Oh: float,
        D0: float,
        theta_eq: float = 162.0,
        rho_l: float = 828.0,
        rho_g: float = 1.0,
        xi: float = 5.0,
        beta_A: float = 0.25,
        U0: float = 0.05,
        nx: int = 200,
        ny: int = 200,
        nz: int = 1,
        tau_g: float = 0.6,
        device: str = "cuda",
        max_steps: int = 10000,
        output_interval: int = 100,
    ) -> "FEConfig":
        """Create FEConfig matching Liu 2015 (or similar) paper parameters.

        Given dimensionless numbers We and Oh, along with D0 in lattice
        units, this computes the required sigma, tau_l, etc.

        The impact velocity U0 is chosen first (default 0.05 in lattice
        units, well within the low-Mach stability regime).  Then sigma
        and tau_l are derived to satisfy We and Oh.

        Parameters
        ----------
        We : float
            Weber number = rho_l * U0^2 * D0 / sigma.
        Oh : float
            Ohnesorge number = mu_l / sqrt(rho_l * sigma * D0).
        D0 : float
            Droplet diameter in lattice units.
        theta_eq : float
            Equilibrium contact angle [degrees].
        rho_l, rho_g : float
            Densities in lattice units.
        xi : float
            Interface thickness in lattice units.
        beta_A : float
            Artificial free-energy coefficient.
        U0 : float
            Impact velocity in lattice units (default 0.05).
        nx, ny, nz : int
            Grid size (nz=1 for 2D).
        tau_g : float
            Gas relaxation time.
        device : str
            ``"cuda"`` or ``"cpu"``.
        max_steps : int
            Maximum simulation steps.
        output_interval : int
            Steps between output frames.

        Returns
        -------
        FEConfig
            Fully resolved configuration.
        """
        # Derive sigma from We:  We = rho_l * U0^2 * D0 / sigma
        sigma = rho_l * U0 ** 2 * D0 / We

        tau_l = cls.compute_tau_from_oh(Oh, D0, rho_l, sigma)

        # Pre-compute all derived quantities so __post_init__ does not
        # override them.
        beta = 12.0 * sigma / xi
        kappa = beta * xi ** 2 / 8.0
        cos_theta = math.cos(math.radians(theta_eq))
        phi_c = -cos_theta * math.sqrt(2.0 * kappa * beta)

        return cls(
            nx=nx, ny=ny, nz=nz,
            rho_l=rho_l, rho_g=rho_g,
            sigma=sigma, xi=xi,
            beta=beta, kappa=kappa,
            M=0.02 / beta,
            beta_A=beta_A,
            tau_l=tau_l, tau_g=tau_g, tau_h=0.5,
            theta_eq=theta_eq,
            phi_c=phi_c,
            max_steps=max_steps,
            output_interval=output_interval,
            device=device,
        )

    # ------------------------------------------------------------------ #
    # Print summary
    # ------------------------------------------------------------------ #
    def print_info(self):
        """Print a summary of all configuration parameters."""
        print(f"Free-Energy Phase-Field LBM Config (Lee & Liu 2010)")
        print(f"  Grid: {self.grid_shape}, dim={self.dim}, device={self.device}")
        print(f"  --- Densities ---")
        print(f"  rho_l={self.rho_l}, rho_g={self.rho_g}, "
              f"ratio={self.density_ratio:.1f}")
        print(f"  --- Free Energy ---")
        print(f"  sigma={self.sigma:.6f}, xi={self.xi:.2f}")
        print(f"  beta={self.beta:.6f}, kappa={self.kappa:.6f}")
        print(f"  M={self.M:.6e}, beta_A={self.beta_A}")
        print(f"  --- Relaxation ---")
        print(f"  tau_l={self.tau_l:.4f}, tau_g={self.tau_g:.4f}, tau_h={self.tau_h}")
        print(f"  nu_l={self.nu_l:.6f}, nu_g={self.nu_g:.6f}")
        print(f"  mu_l={self.mu_l:.6f}, mu_g={self.mu_g:.6f}")
        print(f"  --- Wetting ---")
        print(f"  theta_eq={self.theta_eq:.1f} deg, phi_c={self.phi_c:.6f}")
        if self.kappa > 0 and self.beta > 0:
            cos_check = -self.phi_c / math.sqrt(2.0 * self.kappa * self.beta)
            theta_check = math.degrees(math.acos(
                max(-1.0, min(1.0, cos_check))))
            print(f"  (check: cos(theta)={cos_check:.4f}, "
                  f"theta={theta_check:.1f} deg)")
        print(f"  --- Simulation ---")
        print(f"  max_steps={self.max_steps}, output_interval={self.output_interval}")
