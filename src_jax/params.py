"""
Parameters for 1D Heat Diffusion Simulation.

Physical problem: du/dt = alpha * d²u/dx²
Domain: x in [0, L]
Boundary conditions: u(0,t) = u(L,t) = 0 (Dirichlet)
Initial condition: u(x,0) = sin(pi*x/L)

Analytical solution: u(x,t) = sin(pi*x/L) * exp(-alpha*(pi/L)²*t)

Translated from: params_mod.f90
"""

# CRITICAL: Enable float64 BEFORE other JAX imports
from jax import config
config.update("jax_enable_x64", True)

from typing import NamedTuple
import jax.numpy as jnp

# Mathematical constant
PI = jnp.float64(3.14159265358979323846)


class HeatParams(NamedTuple):
    """Immutable simulation parameters.

    Attributes:
        nx: Number of spatial grid points
        L: Domain length [m]
        alpha: Thermal diffusivity [m²/s]
        cfl: CFL number (stability requires <= 0.5)
        t_end: End time [s]
        output_freq: Output every N steps
    """
    nx: int = 101
    L: float = 1.0
    alpha: float = 0.01
    cfl: float = 0.4
    t_end: float = 1.0
    output_freq: int = 100

    @property
    def dx(self) -> float:
        """Spatial step size."""
        return self.L / (self.nx - 1)

    @property
    def dt(self) -> float:
        """Time step (derived from CFL condition)."""
        return self.cfl * self.dx * self.dx / self.alpha

    @property
    def n_steps(self) -> int:
        """Total number of time steps."""
        return int(self.t_end / self.dt)

    @property
    def r(self) -> float:
        """FTCS stability parameter: r = alpha * dt / dx²."""
        return self.alpha * self.dt / (self.dx * self.dx)


def print_params(params: HeatParams) -> None:
    """Print simulation parameters (mirrors Fortran print_params)."""
    print("=" * 44)
    print("1D Heat Diffusion Simulation Parameters")
    print("=" * 44)
    print(f"  Grid points (nx):     {params.nx:6d}")
    print(f"  Domain length (L):    {params.L:10.6f}")
    print(f"  Grid spacing (dx):    {params.dx:12.4e}")
    print(f"  Diffusivity (alpha):  {params.alpha:12.4e}")
    print(f"  End time (t_end):     {params.t_end:10.6f}")
    print(f"  Time step (dt):       {params.dt:12.4e}")
    print(f"  CFL number:           {params.cfl:10.6f}")
    print(f"  Total time steps:     {params.n_steps:6d}")
    print("=" * 44)
