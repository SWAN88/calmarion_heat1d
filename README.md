# heat1d - 1D Heat Diffusion Simulation

**Level 0 Testbed for Fortran → Python/JAX Migration**

## Overview

This is a minimal, well-structured Fortran 90 implementation of the 1D heat diffusion equation, designed as a simple testbed for experimenting with Claude Code configurations and migration workflows.

## Physical Problem

Solves the 1D heat equation:
```
∂u/∂t = α · ∂²u/∂x²
```

With:
- **Domain**: x ∈ [0, L]
- **Boundary conditions**: u(0,t) = u(L,t) = 0 (Dirichlet)
- **Initial condition**: u(x,0) = sin(πx/L)
- **Analytical solution**: u(x,t) = sin(πx/L) · exp(-α(π/L)²t)

## Numerical Method

Forward Time Central Space (FTCS) explicit scheme:
```
u(i,n+1) = u(i,n) + r·[u(i-1,n) - 2·u(i,n) + u(i+1,n)]
```
where r = α·dt/dx² (stability requires r ≤ 0.5)

## Code Structure

```
heat1d/
├── src/
│   ├── params_mod.f90      # Parameters and constants
│   ├── init_mod.f90        # Grid and initial conditions
│   ├── simulation_mod.f90  # FTCS time stepping
│   ├── output_mod.f90      # Data output routines
│   ├── heat1d.f90          # Main program
│   └── Makefile
└── output/                  # Generated output files
```

| File | Lines | Purpose |
|------|-------|---------|
| params_mod.f90 | ~50 | Grid/physics parameters |
| init_mod.f90 | ~50 | Initial conditions + analytical solution |
| simulation_mod.f90 | ~70 | FTCS algorithm + error computation |
| output_mod.f90 | ~100 | CSV output routines |
| heat1d.f90 | ~90 | Main time loop |
| **Total** | **~360** | Simple, readable codebase |

## Build & Run

```bash
cd src
make        # Compile
make run    # Run simulation
make clean  # Clean build files
```

## Output Files

- `solution_NNNNNN.csv` - Solution at each output step (x, u_numerical, u_analytical, error)
- `history.csv` - Time series of errors
- `report.txt` - Final summary

## Validation

The code includes an analytical solution for direct validation:
- L2 error: ~10⁻⁶
- Max error: ~10⁻⁵

These errors are consistent with the O(dt) + O(dx²) truncation error of the FTCS scheme.

## Why Level 0?

This simulation is ideal for testing Claude Code configurations because:

1. **Simple physics** - Easy to understand and validate
2. **Modular structure** - Similar to larger codes (like Navier-Stokes)
3. **Analytical solution** - Definitive validation metric
4. **Fast execution** - Quick iteration cycles
5. **Clear migration path** - Each module has an obvious Python/JAX equivalent

## Migration Targets

| Fortran Module | Python Equivalent | JAX Optimization |
|----------------|-------------------|------------------|
| params_mod | dataclass / namedtuple | Static config |
| init_mod | NumPy functions | jnp.sin |
| simulation_mod | NumPy stencil | @jit, scan |
| output_mod | pandas / csv | Same |

## Next Steps (Level 1)

After validating Claude Code settings with this Level 0 code, proceed to the more complex Navier-Stokes simulation (navier/).
