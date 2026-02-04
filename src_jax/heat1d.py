#!/usr/bin/env python3
"""
1D Heat Diffusion Simulation

Solves: du/dt = alpha * d²u/dx²
Method: Forward Time Central Space (FTCS) explicit scheme

Domain: x in [0, L]
Boundary conditions: u(0,t) = u(L,t) = 0 (Dirichlet)
Initial condition: u(x,0) = sin(pi*x/L)
Analytical solution: u(x,t) = sin(pi*x/L) * exp(-alpha*(pi/L)²*t)

Translated from: heat1d.f90

Usage:
    python -m src_jax.heat1d
"""

# CRITICAL: Enable float64 BEFORE other JAX imports
from jax import config
config.update("jax_enable_x64", True)

import jax.numpy as jnp
from pathlib import Path

from src_jax.params import HeatParams, print_params
from src_jax.init import init_grid, init_temperature, analytical_solution
from src_jax.simulation import (
    timestep_ftcs,
    compute_l2_error,
    compute_max_error,
    compute_total_heat,
)


def main():
    """Main simulation driver."""
    # Initialize parameters (default matches Fortran)
    params = HeatParams()

    print_params(params)

    # Initialize grid and temperature field
    x = init_grid(params)
    u = init_temperature(x, params)

    # Initial state
    t = 0.0
    step = 0

    # Compute initial errors
    u_exact = analytical_solution(x, t, params)
    l2_error = compute_l2_error(u, u_exact)
    max_error = compute_max_error(u, u_exact)
    total_heat = compute_total_heat(u, params.dx)

    print()
    print("Starting time integration...")
    print("Step      Time          L2 Error      Max Error")
    print("-" * 48)
    print(f"{step:6d}{t:12.6f}{float(l2_error):14.6e}{float(max_error):14.6e}")

    # Main time loop
    while t < params.t_end:
        # Advance one time step
        u = timestep_ftcs(u, params)
        t = t + params.dt
        step = step + 1

        # Compute errors
        u_exact = analytical_solution(x, t, params)
        l2_error = compute_l2_error(u, u_exact)
        max_error = compute_max_error(u, u_exact)
        total_heat = compute_total_heat(u, params.dx)

        # Output
        if step % params.output_freq == 0:
            print(f"{step:6d}{t:12.6f}{float(l2_error):14.6e}{float(max_error):14.6e}")

    # Final output
    print("-" * 48)
    print()
    print("Final Results:")
    print(f"  Total steps:  {step:8d}")
    print(f"  Final time:   {t:12.8f}")
    print(f"  L2 error:     {float(l2_error):14.6e}")
    print(f"  Max error:    {float(max_error):14.6e}")
    print()
    print("Simulation completed successfully.")


if __name__ == "__main__":
    main()
