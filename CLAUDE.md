# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

1D heat diffusion solver using FTCS (Forward Time Central Space) explicit finite difference scheme. The repository contains both the original Fortran implementation and a JAX/Python port for GPU acceleration and automatic differentiation.

## Build & Run Commands

### Python/JAX (Primary)

```bash
pip install -e ".[dev]"       # Install with dev dependencies
python -m src_jax.heat1d      # Run simulation
pytest                        # Run all tests
pytest --cov=src_jax          # Run tests with coverage
```

### Fortran (Reference)

```bash
cd src
make           # Compile with gfortran
make run       # Compile and run simulation (creates output/ directory)
make clean     # Remove executables, object files, and .mod files
make cleanall  # Also remove output/ directory
```

The Fortran simulation produces output files in `output/`:
- `solution_NNNNNN.csv` - Solution snapshots (x, u_numerical, u_analytical, error)
- `history.csv` - Time series of L2/max errors and total heat
- `report.txt` - Final summary

## Project Structure

```
├── src_jax/              # JAX/Python implementation
│   ├── params.py         # HeatParams NamedTuple with derived properties
│   ├── init.py           # Grid/temperature initialization, analytical solution
│   ├── simulation.py     # FTCS kernel, error computation, run_simulation
│   └── heat1d.py         # Main simulation driver
├── src/                  # Fortran reference implementation
│   ├── params_mod.f90    # Compile-time parameters
│   ├── init_mod.f90      # Initialization routines
│   ├── simulation_mod.f90# FTCS kernel and error computation
│   ├── output_mod.f90    # CSV output routines
│   └── heat1d.f90        # Main program
├── tests/                # Validation tests
│   ├── conftest.py       # Pytest fixtures with Fortran reference data
│   └── test_heat1d.py    # JAX vs Fortran comparison tests
└── scripts/              # Utility scripts
    └── extract_reference_data.py
```

## Architecture

**Fortran module dependency chain:**
```
params_mod → init_mod
           → simulation_mod  → heat1d (main)
           → output_mod
```

**JAX module structure mirrors Fortran:**
- `params.py`: Immutable `HeatParams` NamedTuple with derived properties (dx, dt, r)
- `init.py`: `init_grid()`, `init_temperature()`, `analytical_solution()`
- `simulation.py`: `timestep_ftcs()`, `compute_l2_error()`, `compute_max_error()`, `compute_total_heat()`, `run_simulation()` (lax.scan-based)
- `heat1d.py`: Main driver with time loop

## Key Constants

Default parameters (both implementations):
- `nx = 101` grid points
- `L = 1.0` domain length
- `alpha = 0.01` thermal diffusivity
- `cfl = 0.4` (stability requires ≤ 0.5)
- `t_end = 1.0` end time
- `output_freq = 100` steps between outputs

## Validation

Analytical solution: `u(x,t) = sin(πx/L) * exp(-α(π/L)²t)`

Expected errors with default parameters:
- L2 error: ~7.25×10⁻⁶
- Max error: ~1.03×10⁻⁵

Tests validate JAX results against Fortran reference data within:
- Relative tolerance: 1e-10
- Absolute tolerance: 1e-12

## Dependencies

- Python ≥3.10
- JAX ≥0.4.0
- NumPy ≥1.24.0
- pytest ≥7.0.0 (dev)
