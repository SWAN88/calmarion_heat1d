"""
Initial conditions and grid setup for 1D Heat Diffusion.

Translated from: init_mod.f90
"""

# CRITICAL: Enable float64 BEFORE other JAX imports
from jax import config
config.update("jax_enable_x64", True)

import jax.numpy as jnp
from src_jax.params import HeatParams, PI


def init_grid(params: HeatParams) -> jnp.ndarray:
    """Initialize spatial grid.

    Translated from: init_mod.f90, init_grid subroutine (lines 10-21)

    Args:
        params: Simulation parameters

    Returns:
        x: Spatial grid array of shape (nx,)
    """
    return jnp.linspace(0.0, params.L, params.nx)


def init_temperature(x: jnp.ndarray, params: HeatParams) -> jnp.ndarray:
    """Set initial temperature distribution.

    Initial condition: u(x,0) = sin(pi*x/L)
    Boundary conditions: u(0) = u(L) = 0

    Translated from: init_mod.f90, init_temperature subroutine (lines 23-40)

    Args:
        x: Spatial grid
        params: Simulation parameters

    Returns:
        u: Initial temperature field
    """
    u = jnp.sin(PI * x / params.L)
    # Enforce boundary conditions (already satisfied by sin, but explicit)
    u = u.at[0].set(0.0)
    u = u.at[-1].set(0.0)
    return u


def analytical_solution(x: jnp.ndarray, t: float, params: HeatParams) -> jnp.ndarray:
    """Compute analytical solution at time t.

    u(x,t) = sin(pi*x/L) * exp(-alpha*(pi/L)²*t)

    Translated from: init_mod.f90, analytical_solution function (lines 42-59)

    Args:
        x: Spatial grid
        t: Current time
        params: Simulation parameters

    Returns:
        u_exact: Analytical solution at time t
    """
    decay_rate = params.alpha * (PI / params.L) ** 2
    return jnp.sin(PI * x / params.L) * jnp.exp(-decay_rate * t)
