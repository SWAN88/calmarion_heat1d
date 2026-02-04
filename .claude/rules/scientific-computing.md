# Scientific Computing Rules

These rules apply to all scientific computing and physics simulation code.

## Numerical Precision

### ALWAYS enable float64 in JAX

```python
# REQUIRED at TOP of every script, BEFORE other JAX imports
from jax import config
config.update("jax_enable_x64", True)
```

### Validation tolerances

- Relative tolerance: `rtol = 1e-10`
- Absolute tolerance: `atol = 1e-12`

```python
assert jnp.allclose(result, reference, rtol=1e-10, atol=1e-12)
```

## Vectorization

### NEVER use Python for-loops on arrays

```python
# ❌ FORBIDDEN
for i in range(n):
    result[i] = arr[i] * 2

# ✅ REQUIRED
result = arr * 2
```

### Use JAX patterns for loops

| Need | Use |
|------|-----|
| Element-wise | Direct array ops |
| Batched | `jax.vmap` |
| Time stepping | `jax.lax.scan` |
| Conditional | `jnp.where` |
| Reduction | `jnp.sum`, `jnp.mean`, etc. |

## JIT Compilation

### Apply @jax.jit to compute-intensive functions

```python
@jax.jit
def compute_heavy(x, params):
    # ... expensive computation
    return result
```

### Avoid JIT-incompatible patterns

- No Python control flow on traced values
- No dynamic array shapes
- No side effects (print, global modification)

## Testing

### Every numerical function needs validation tests

```python
def test_laplacian():
    """Validate against analytical solution."""
    result = laplacian(test_input, dx)
    expected = analytical_laplacian(test_input)
    assert jnp.allclose(result, expected, rtol=1e-10, atol=1e-12)
```

### Test categories

1. **Unit tests**: Individual function validation
2. **Validation tests**: Against reference (MATLAB/Fortran/analytical)
3. **Convergence tests**: Spatial/temporal convergence rates
4. **Conservation tests**: Energy/mass/momentum conservation

## Code Style

### Immutable updates

```python
# ❌ FORBIDDEN
arr[0] = value

# ✅ REQUIRED
arr = arr.at[0].set(value)
```

### Pure functions

```python
@jax.jit
def update(state, params):
    """Pure function - no side effects."""
    # Compute new state
    return new_state  # Return, don't modify in place
```

### Document source location

When translating from Fortran/MATLAB:

```python
def compute_flux(u, v, dx):
    """
    Compute advective flux.
    
    Translated from: solver.f90, lines 45-67
    Original name: COMPUTE_FLUX
    """
```
