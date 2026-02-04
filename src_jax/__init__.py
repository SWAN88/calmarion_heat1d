"""
1D Heat Diffusion Solver in JAX

Migrated from Fortran implementation.
Solves: du/dt = alpha * d²u/dx² using FTCS scheme.
"""

# CRITICAL: Enable float64 BEFORE other JAX imports
from jax import config
config.update("jax_enable_x64", True)

from src_jax.params import HeatParams
from src_jax.simulation import (
    SimulationState,
    timestep_ftcs,
    compute_l2_error,
    compute_max_error,
    compute_total_heat,
    run_simulation,
)
from src_jax.init import init_grid, init_temperature, analytical_solution

__all__ = [
    'HeatParams',
    'SimulationState',
    'init_grid',
    'init_temperature',
    'analytical_solution',
    'timestep_ftcs',
    'compute_l2_error',
    'compute_max_error',
    'compute_total_heat',
    'run_simulation',
]
