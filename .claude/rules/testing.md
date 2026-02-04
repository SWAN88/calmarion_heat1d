# Testing Requirements for Scientific Computing

## Validation Standard

All numerical results must match reference solutions within:
- **Relative error**: < 1e-10 (machine precision)
- **Absolute error**: < 1e-12 for near-zero values

## Test Types (required)

1. **Unit Tests** - Individual functions, matrix operations
2. **Validation Tests** - Compare against MATLAB/CalculiX reference
3. **Convergence Tests** - Verify numerical stability

## Test-Driven Development

MANDATORY workflow:
1. Generate reference data from MATLAB/CalculiX first
2. Write test with expected values (RED)
3. Run test - it should FAIL
4. Implement JAX/Python code (GREEN)
5. Run test - it should PASS within tolerance
6. Profile performance (IMPROVE)

## Comparison Pattern
```python
import jax.numpy as jnp

def assert_numerical_equal(result, reference, rtol=1e-10, atol=1e-12):
    """Validate against reference solution."""
    assert jnp.allclose(result, reference, rtol=rtol, atol=atol), \
        f"Max error: {jnp.max(jnp.abs(result - reference))}"
```

## Reference Data Management

- Store in `tests/reference_data/`
- Use `.npy` format for arrays
- Document source (MATLAB script, CalculiX version)

## Agent Support

- **tdd-guide** - Use for structuring validation tests
- **code-reviewer** - Verify numerical accuracy patterns
