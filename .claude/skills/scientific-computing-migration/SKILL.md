---
name: scientific-computing-migration
description: |
  Migrate Fortran/MATLAB/C++ physics simulation code to Python/JAX with 
  numerical validation within machine precision. Covers heat transfer, CFD, 
  finite element analysis, CalculiX integration, BEM methods, and inverse 
  problems with automatic differentiation.
  
  Triggers: "Fortran to JAX", "MATLAB to Python", "migrate simulation", 
  "convert physics code", "CalculiX", "inverse problem", "heat transfer", 
  "finite element", "BEM", "CFD", ".f90 files", ".m files"
metadata:
  author: katsuya
  version: 2.0.0
  domain: scientific-computing
---

# Scientific Computing Migration Skill

A systematic workflow for migrating legacy physics simulation code to modern 
Python/JAX while maintaining numerical accuracy within machine precision.

## Quick Reference Card

| Standard | Value | Notes |
|----------|-------|-------|
| Relative tolerance | `rtol = 1e-10` | For values > 1e-10 |
| Absolute tolerance | `atol = 1e-12` | For near-zero values |
| Float precision | `jnp.float64` | ALWAYS enable x64 |
| Vectorization | MANDATORY | NEVER use Python for-loops on arrays |

---

## Core Principles

### 1. STRICT NUMERICAL ACCURACY

All results MUST match reference solutions within machine precision:

```python
# Validation standard - use this everywhere
rtol = 1e-10  # Relative tolerance
atol = 1e-12  # Absolute tolerance

assert jnp.allclose(result, reference, rtol=rtol, atol=atol), \
    f"Validation FAILED: max_error = {jnp.max(jnp.abs(result - reference)):.2e}"
```

### 2. ALWAYS ENABLE FLOAT64

```python
# CRITICAL: Put this at the TOP of every script BEFORE other JAX imports
from jax import config
config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
```

### 3. NEVER USE PYTHON FOR-LOOPS FOR ARRAY OPERATIONS

```python
# ❌ FORBIDDEN - #1 cause of slow code
for i in range(nx):
    for j in range(ny):
        laplacian[i,j] = (u[i+1,j] - 2*u[i,j] + u[i-1,j]) / dx**2

# ✅ REQUIRED - Vectorized with roll (periodic BC)
laplacian = (jnp.roll(u, -1, axis=0) - 2*u + jnp.roll(u, 1, axis=0)) / dx**2

# ✅ REQUIRED - Vectorized with slicing (non-periodic BC)
laplacian = (u[2:,1:-1] - 2*u[1:-1,1:-1] + u[:-2,1:-1]) / dx**2
```

### 4. PURE FUNCTIONS FOR JIT COMPILATION

```python
@jax.jit
def update_temperature(T, params):
    """Pure function - no side effects, returns new array."""
    T_new = T + params.dt * compute_laplacian(T, params.dx)
    T_new = T_new.at[0].set(params.T_left)    # Dirichlet left
    T_new = T_new.at[-1].set(params.T_right)  # Dirichlet right
    return T_new
```

### 5. IMMUTABLE UPDATES WITH .at[].set()

```python
# ❌ FORBIDDEN - JAX arrays are immutable
T[0] = T_left

# ✅ REQUIRED - Functional update
T = T.at[0].set(T_left)
T = T.at[-1].set(T_right)
T = T.at[1:-1].set(T_interior)
```

---

## Migration Workflow

### Phase 1: Source Code Analysis

**Objective**: Fully understand the original codebase before conversion.

**Steps**:
1. List all source files and dependencies
2. Map subroutine/function call graph
3. Document array dimensions and memory layout
4. Identify global variables, COMMON blocks, module variables
5. Note I/O operations and file formats

**Commands**:
```bash
# Find all source files
find . -name "*.f90" -o -name "*.f" -o -name "*.F90" -o -name "*.m"

# Extract subroutine signatures
grep -n "subroutine\|function\|module" *.f90
```

**Output**: `analysis/structure.md`

**Agent**: Use `fortran-jax-translator` for analysis

### Phase 2: NumPy Translation (Direct)

**Objective**: Create direct translation preserving original logic.

**Rules**:
- Keep explicit loops initially (for validation)
- Adjust indices (Fortran 1-based → Python 0-based)
- Handle column-major vs row-major differences
- Document every function with original source location

**Output**: Working NumPy code that matches original exactly

### Phase 3: Numerical Validation (NumPy)

**Objective**: Verify NumPy implementation matches original.

**Steps**:
1. Generate test cases from original code
2. Run `scripts/numerical_validator.py`
3. Verify rtol < 1e-10 for all test cases
4. Check conservation laws (energy, mass, momentum)

**Acceptance Criteria**:
| Metric | Threshold |
|--------|-----------|
| Relative error | < 1e-10 |
| Absolute error | < 1e-12 |
| Energy drift | < 1e-12 per step |

### Phase 4: JAX Vectorization

**Objective**: Convert NumPy to JAX with full vectorization.

**Steps**:
1. Replace `import numpy as np` → `import jax.numpy as jnp`
2. Remove ALL for-loops over arrays
3. Apply `@jax.jit` to compute-intensive functions
4. Use `jax.vmap` for batched operations
5. Use `jax.lax.scan` for time stepping

**Agent**: Use `vectorization-reviewer` to find remaining anti-patterns

### Phase 5: Final Validation and Benchmarking

**Objective**: Verify JAX version and measure performance.

**Steps**:
1. Re-run numerical validation
2. Run `scripts/benchmark.py`
3. Test GPU acceleration (if available)
4. Generate migration report

**Output**: `reports/migration_report.md`

---

## Key Patterns

### State Management with NamedTuple

```python
from typing import NamedTuple
import jax.numpy as jnp

class SimulationState(NamedTuple):
    """Immutable simulation state."""
    T: jnp.ndarray      # Temperature field
    t: float            # Current time
    step: int           # Step counter
    energy: float       # Total energy (conservation check)
```

### Time Stepping with lax.scan

```python
from jax.lax import scan

def run_simulation(state_init, params, n_steps):
    """JIT-compiled time stepping."""
    
    def step_fn(state, _):
        T_new = update_temperature(state.T, params)
        new_state = state._replace(
            T=T_new,
            t=state.t + params.dt,
            step=state.step + 1
        )
        return new_state, state.T
    
    final_state, T_history = scan(step_fn, state_init, None, length=n_steps)
    return final_state, T_history
```

### Parameter Estimation with Auto-Diff

```python
from jax import value_and_grad
import optax

def estimate_parameter(T_observed, forward_fn, param_init, n_iter=1000):
    """Gradient-based parameter estimation."""
    
    def loss_fn(param):
        T_pred = forward_fn(param)
        return jnp.mean((T_pred - T_observed)**2)
    
    optimizer = optax.adam(0.01)
    param = param_init
    opt_state = optimizer.init(param)
    
    for _ in range(n_iter):
        loss, grads = value_and_grad(loss_fn)(param)
        updates, opt_state = optimizer.update(grads, opt_state)
        param = optax.apply_updates(param, updates)
    
    return param
```

---

## Quick Reference Tables

### Fortran → JAX Type Mapping

| Fortran | JAX | Notes |
|---------|-----|-------|
| `REAL*8`, `DOUBLE PRECISION` | `jnp.float64` | Always use float64 |
| `INTEGER` | `jnp.int32` | |
| `INTEGER*8` | `jnp.int64` | |
| `COMPLEX*16` | `jnp.complex128` | |
| `Array(N,M)` | `jnp.ndarray` | Watch memory layout |

### Loop → Vectorization Patterns

| Original Pattern | JAX Replacement |
|------------------|-----------------|
| `for i in range(n): arr[i]` | Direct array ops or `vmap` |
| Nested loops (stencil) | `jnp.roll` or slicing |
| `if arr[i] > 0:` | `jnp.where(arr > 0, ...)` |
| `sum += arr[i]` | `jnp.sum(arr)` |
| Time stepping loop | `jax.lax.scan` |

### CFL Stability Conditions

| Equation | Condition | Safe Value |
|----------|-----------|------------|
| Heat (explicit) | α·dt/dx² ≤ 0.5 | 0.4 |
| Advection | u·dt/dx ≤ 1.0 | 0.8 |
| Wave | c·dt/dx ≤ 1.0 | 0.8 |

---

## Directory Structure

```
scientific-computing-migration/
├── SKILL.md                          # This file
├── references/
│   ├── fortran-to-python.md          # Type/syntax conversion
│   ├── jax-patterns.md               # JAX idioms
│   ├── stencil-patterns.md           # Finite difference stencils
│   ├── calculix-integration.md       # CalculiX I/O
│   └── inverse-problems.md           # Auto-diff optimization
├── scripts/
│   ├── numerical_validator.py        # Validation script
│   ├── benchmark.py                  # Performance comparison
│   └── vectorization_checker.py      # Anti-pattern detector
└── templates/
    ├── heat1d.py                     # 1D heat equation starter
    └── simulation.py                 # Generic simulation starter
```

---

## Related Agents

| Agent | Purpose |
|-------|---------|
| `fortran-jax-translator` | Converting Fortran/MATLAB code |
| `vectorization-reviewer` | Reviewing for loop anti-patterns |

---

## Troubleshooting

### Numerical Errors Won't Converge

1. Verify float64 is enabled (`jax_enable_x64`)
2. Check operation order (floating-point not associative)
3. Compare intermediate results at finer granularity
4. Look for uninitialized values in original

### JAX JIT Compilation Errors

1. Remove Python control flow → use `jax.lax.cond`
2. Ensure no dynamic array shapes
3. Check for side effects (print, globals)
4. Inspect with `jax.make_jaxpr(fn)(args)`

### Performance Not Improved

1. Verify JIT is compiling (watch for recompilation)
2. Check for unnecessary CPU↔GPU transfers
3. Profile with `jax.profiler`
4. Ensure vmap replaces loops, not adds overhead
