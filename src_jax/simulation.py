"""
Core simulation routines for 1D Heat Diffusion.

Method: Forward Time Central Space (FTCS) explicit scheme

u(i,n+1) = u(i,n) + r * [u(i-1,n) - 2*u(i,n) + u(i+1,n)]
where r = alpha * dt / dx²

Stability condition: r <= 0.5 (CFL condition)

Translated from: simulation_mod.f90
"""

# CRITICAL: Enable float64 BEFORE other JAX imports
from jax import config
config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
from jax.lax import scan
from typing import NamedTuple

from src_jax.params import HeatParams


class SimulationState(NamedTuple):
    """Immutable simulation state.

    Attributes:
        u: Temperature field at current time
        t: Current simulation time
        step: Time step counter
    """
    u: jnp.ndarray
    t: float
    step: int


@jax.jit
def timestep_ftcs(u: jnp.ndarray, params: HeatParams) -> jnp.ndarray:
    """Perform one FTCS time step.

    Translated from: simulation_mod.f90, timestep_ftcs subroutine (lines 17-39)

    Uses vectorized slicing for the stencil operation:
    u_new[i] = u[i] + r * (u[i-1] - 2*u[i] + u[i+1])

    Args:
        u: Temperature at current time
        params: Simulation parameters

    Returns:
        u_new: Temperature at next time step
    """
    r = params.r

    # Vectorized FTCS stencil for interior points
    # u[:-2] = u[i-1], u[1:-1] = u[i], u[2:] = u[i+1]
    u_interior = u[1:-1] + r * (u[:-2] - 2.0 * u[1:-1] + u[2:])

    # Build new array with boundary conditions
    u_new = jnp.zeros_like(u)
    u_new = u_new.at[1:-1].set(u_interior)
    # Dirichlet BCs: u_new[0] = 0, u_new[-1] = 0 (already zero)

    return u_new


@jax.jit
def compute_l2_error(u: jnp.ndarray, u_exact: jnp.ndarray) -> float:
    """Compute L2 norm of error between numerical and analytical solution.

    Translated from: simulation_mod.f90, compute_error function (lines 41-57)

    Args:
        u: Numerical solution
        u_exact: Analytical solution

    Returns:
        l2_error: sqrt(mean((u - u_exact)²))
    """
    return jnp.sqrt(jnp.mean((u - u_exact) ** 2))


@jax.jit
def compute_max_error(u: jnp.ndarray, u_exact: jnp.ndarray) -> float:
    """Compute maximum (L-infinity) error.

    Translated from: simulation_mod.f90, compute_max_error function (lines 59-73)

    Args:
        u: Numerical solution
        u_exact: Analytical solution

    Returns:
        max_error: max(|u - u_exact|)
    """
    return jnp.max(jnp.abs(u - u_exact))


@jax.jit
def compute_total_heat(u: jnp.ndarray, dx: float) -> float:
    """Compute total heat content (integral of u over domain).

    Uses trapezoidal rule.

    Translated from: simulation_mod.f90, compute_total_heat function (lines 75-90)

    Args:
        u: Temperature field
        dx: Grid spacing

    Returns:
        total: Integral of u over domain
    """
    # Trapezoidal rule: (u[0]/2 + sum(u[1:-1]) + u[-1]/2) * dx
    return (0.5 * (u[0] + u[-1]) + jnp.sum(u[1:-1])) * dx


def run_simulation(
    u_init: jnp.ndarray,
    params: HeatParams,
    n_steps: int
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Run simulation for n_steps using lax.scan.

    JIT-compiled time stepping loop for efficiency.

    Args:
        u_init: Initial temperature field
        params: Simulation parameters
        n_steps: Number of time steps to run

    Returns:
        u_final: Final temperature field
        u_history: Array of shape (n_steps+1, nx) with full history
    """

    def step_fn(u: jnp.ndarray, _) -> tuple[jnp.ndarray, jnp.ndarray]:
        u_new = timestep_ftcs(u, params)
        return u_new, u_new

    # Run simulation with scan
    u_final, u_history = scan(step_fn, u_init, None, length=n_steps)

    # Prepend initial condition to history
    u_history = jnp.vstack([u_init[None, :], u_history])

    return u_final, u_history


def run_simulation_with_output(
    u_init: jnp.ndarray,
    x: jnp.ndarray,
    params: HeatParams,
    output_callback=None
) -> SimulationState:
    """Run simulation with periodic output (for CLI usage).

    This version supports callbacks for output but is not fully JIT-compiled.

    Args:
        u_init: Initial temperature field
        x: Spatial grid
        params: Simulation parameters
        output_callback: Optional callback(step, t, u, u_exact, l2_err, max_err)

    Returns:
        final_state: Final simulation state
    """
    from src_jax.init import analytical_solution

    u = u_init
    t = 0.0

    for step in range(params.n_steps + 1):
        u_exact = analytical_solution(x, t, params)
        l2_err = compute_l2_error(u, u_exact)
        max_err = compute_max_error(u, u_exact)

        if output_callback and step % params.output_freq == 0:
            output_callback(step, t, u, u_exact, l2_err, max_err)

        if step < params.n_steps:
            u = timestep_ftcs(u, params)
            t = t + params.dt

    return SimulationState(u=u, t=t, step=params.n_steps)
