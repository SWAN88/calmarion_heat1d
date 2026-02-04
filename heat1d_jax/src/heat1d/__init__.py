"""1D Heat Diffusion Solver using JAX."""

import jax
jax.config.update("jax_enable_x64", True)

from heat1d.config import SimulationConfig
from heat1d.grid import create_grid, initial_condition, analytical_solution
from heat1d.solver import ftcs_step
from heat1d.diagnostics import compute_l2_error, compute_max_error, compute_total_heat
from heat1d.io import OutputHandler
from heat1d.runner import run_simulation

__all__ = [
    "SimulationConfig",
    "create_grid",
    "initial_condition",
    "analytical_solution",
    "ftcs_step",
    "compute_l2_error",
    "compute_max_error",
    "compute_total_heat",
    "OutputHandler",
    "run_simulation",
]
