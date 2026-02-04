# Heat1D Migration Report

## Summary

Successfully migrated 1D Heat Diffusion solver from Fortran to Python/JAX with numerical validation within machine precision.

| Metric | Status |
|--------|--------|
| Tests | 20/20 passed |
| Numerical accuracy | rtol < 1e-10 |
| Vectorization | Fully optimized |
| JIT compilation | Applied to all kernels |

## Migration Details

### Source (Fortran)
```
src/
├── params_mod.f90      # Parameters
├── init_mod.f90        # Grid/initial conditions
├── simulation_mod.f90  # FTCS kernel
├── output_mod.f90      # CSV output
└── heat1d.f90          # Main program
```

### Target (JAX)
```
src_jax/
├── __init__.py         # Package exports
├── params.py           # HeatParams NamedTuple
├── init.py             # Vectorized initialization
├── simulation.py       # JIT-compiled FTCS kernel
└── heat1d.py           # CLI driver
```

## Numerical Validation

### Final Results Comparison

| Metric | Fortran | JAX | Match |
|--------|---------|-----|-------|
| L2 Error | 7.245839E-06 | 7.245839e-06 | Exact |
| Max Error | 1.029827E-05 | 1.029827e-05 | Exact |
| Total Heat (t=0) | 0.63656741 | 0.63656741 | Exact |
| Total Heat (t=1) | 0.57673501 | 0.57673501 | Exact |

### Test Categories

1. **Parameter Validation** (2 tests)
   - Parameters match Fortran reference
   - CFL stability condition verified

2. **Initialization** (3 tests)
   - Spatial grid matches reference
   - Initial temperature matches reference
   - Boundary conditions preserved

3. **Analytical Solution** (2 tests)
   - Solution at t=1.0 matches reference
   - Solution at t=0 matches initial condition

4. **FTCS Kernel** (4 tests)
   - Single timestep preserves BCs
   - Array shape preserved
   - 100-step solution matches reference
   - Full simulation matches reference

5. **Error Computation** (2 tests)
   - L2 error matches Fortran
   - Max error matches Fortran

6. **Total Heat** (2 tests)
   - Initial heat integral matches
   - Final heat integral matches

7. **Simulation Runner** (3 tests)
   - Final state matches reference
   - History shape correct
   - Initial condition in history

8. **Convergence** (2 tests)
   - Error decreases with grid refinement
   - Spatial convergence rate ~2nd order

## Key JAX Patterns Used

### 1. Float64 Precision
```python
from jax import config
config.update("jax_enable_x64", True)
```

### 2. Immutable State
```python
class SimulationState(NamedTuple):
    u: jnp.ndarray
    t: float
    step: int
```

### 3. Vectorized FTCS Stencil
```python
@jax.jit
def timestep_ftcs(u, params):
    u_interior = u[1:-1] + r * (u[:-2] - 2.0 * u[1:-1] + u[2:])
    u_new = jnp.zeros_like(u)
    u_new = u_new.at[1:-1].set(u_interior)
    return u_new
```

### 4. Time Stepping with lax.scan
```python
u_final, u_history = scan(step_fn, u_init, None, length=n_steps)
```

### 5. Immutable Array Updates
```python
u = u.at[0].set(0.0)
u = u.at[-1].set(0.0)
```

## Usage

### Run JAX Simulation
```bash
source .venv/bin/activate
python -m src_jax.heat1d
```

### Run Tests
```bash
source .venv/bin/activate
pytest tests/test_heat1d.py -v
```

### Run Fortran (for comparison)
```bash
cd src && make run
```

## Files Created

| File | Purpose |
|------|---------|
| `src_jax/params.py` | Simulation parameters |
| `src_jax/init.py` | Grid and initial conditions |
| `src_jax/simulation.py` | FTCS kernel and error computation |
| `src_jax/heat1d.py` | Main CLI program |
| `tests/test_heat1d.py` | 20 validation tests |
| `tests/conftest.py` | Pytest fixtures |
| `tests/reference_data/*.npy` | Fortran reference data |
| `pyproject.toml` | Package configuration |
| `scripts/extract_reference_data.py` | Reference data extraction |

## Conclusion

The migration successfully preserves numerical accuracy within machine precision (rtol < 1e-10). The JAX implementation uses:

- Full vectorization (no Python loops on arrays)
- JIT compilation for performance
- Immutable patterns for JAX compatibility
- lax.scan for efficient time stepping

The code is ready for:
- GPU acceleration (via JAX)
- Automatic differentiation for inverse problems
- Extension to higher dimensions
