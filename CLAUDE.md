# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Run Commands

```bash
cd src
make           # Compile with gfortran
make run       # Compile and run simulation (creates output/ directory)
make clean     # Remove executables, object files, and .mod files
make cleanall  # Also remove output/ directory
```

The simulation produces output files in `output/`:
- `solution_NNNNNN.csv` - Solution snapshots (x, u_numerical, u_analytical, error)
- `history.csv` - Time series of L2/max errors and total heat
- `report.txt` - Final summary

## Architecture

This is a 1D heat diffusion solver using the Forward Time Central Space (FTCS) explicit finite difference scheme. The code is structured as a testbed for Fortran → Python/JAX migration.

**Module dependency chain:**
```
params_mod → init_mod
           → simulation_mod  → heat1d (main)
           → output_mod
```

- **params_mod.f90**: Compile-time parameters (grid size `nx`, domain length `L`, diffusivity `alpha`, CFL number). All derived quantities (dx, dt) are computed here.
- **init_mod.f90**: Grid initialization, initial condition (sin wave), and analytical solution function for validation.
- **simulation_mod.f90**: FTCS time-stepping kernel and error computation (L2, max, total heat).
- **output_mod.f90**: CSV output routines with Fortran unit I/O.
- **heat1d.f90**: Main program with time loop.

## Key Constants

Parameters are defined as Fortran `parameter` constants in `params_mod.f90`:
- `nx = 101` grid points
- `alpha = 0.01` thermal diffusivity
- `cfl = 0.4` (stability requires ≤ 0.5)
- `output_freq = 100` steps between outputs

## Validation

The code includes an analytical solution: `u(x,t) = sin(πx/L) * exp(-α(π/L)²t)`

Expected errors with default parameters: L2 ~ 10⁻⁶, Max ~ 10⁻⁵
