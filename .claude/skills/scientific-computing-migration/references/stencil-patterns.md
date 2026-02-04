# Finite Difference Stencil Patterns for JAX

Complete reference for implementing finite difference methods in JAX.

## Setup (CRITICAL)

```python
# At TOP of every script - BEFORE other JAX imports
from jax import config
config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
from jax import jit
```

---

## 1D Stencils

### First Derivative - Central O(Δx²)

```python
@jit
def d_dx_central(f, dx):
    """df/dx using central difference."""
    return (jnp.roll(f, -1) - jnp.roll(f, 1)) / (2 * dx)
```

### First Derivative - Forward/Backward O(Δx)

```python
@jit
def d_dx_forward(f, dx):
    return (jnp.roll(f, -1) - f) / dx

@jit
def d_dx_backward(f, dx):
    return (f - jnp.roll(f, 1)) / dx
```

### Second Derivative O(Δx²)

```python
@jit
def d2_dx2(f, dx):
    """d²f/dx² central difference."""
    return (jnp.roll(f, -1) - 2*f + jnp.roll(f, 1)) / dx**2
```

### Fourth Derivative O(Δx²)

```python
@jit
def d4_dx4(f, dx):
    """d⁴f/dx⁴ for biharmonic equations."""
    return (jnp.roll(f,-2) - 4*jnp.roll(f,-1) + 6*f 
            - 4*jnp.roll(f,1) + jnp.roll(f,2)) / dx**4
```

### Upwind Scheme (Advection-Stable)

```python
@jit
def d_dx_upwind(u, f, dx):
    """df/dx with upwind based on velocity u."""
    f_left = jnp.roll(f, 1)
    f_right = jnp.roll(f, -1)
    return jnp.where(u >= 0,
                     (f - f_left) / dx,
                     (f_right - f) / dx)
```

---

## 2D Stencils

### Laplacian - 5-Point O(Δx²)

```python
@jit
def laplacian_2d(f, dx, dy):
    """∇²f = ∂²f/∂x² + ∂²f/∂y² (periodic BC)"""
    d2x = (jnp.roll(f,-1,axis=0) - 2*f + jnp.roll(f,1,axis=0)) / dx**2
    d2y = (jnp.roll(f,-1,axis=1) - 2*f + jnp.roll(f,1,axis=1)) / dy**2
    return d2x + d2y
```

### Laplacian - Interior Only (Non-Periodic)

```python
@jit
def laplacian_interior(f, dx, dy):
    """Laplacian on interior points. Returns (nx-2, ny-2) array."""
    d2x = (f[2:,1:-1] - 2*f[1:-1,1:-1] + f[:-2,1:-1]) / dx**2
    d2y = (f[1:-1,2:] - 2*f[1:-1,1:-1] + f[1:-1,:-2]) / dy**2
    return d2x + d2y
```

### Gradient

```python
@jit
def gradient_2d(f, dx, dy):
    """∇f = (∂f/∂x, ∂f/∂y)"""
    df_dx = (jnp.roll(f,-1,axis=0) - jnp.roll(f,1,axis=0)) / (2*dx)
    df_dy = (jnp.roll(f,-1,axis=1) - jnp.roll(f,1,axis=1)) / (2*dy)
    return df_dx, df_dy
```

### Divergence

```python
@jit
def divergence_2d(u, v, dx, dy):
    """∇·(u,v) = ∂u/∂x + ∂v/∂y"""
    du_dx = (jnp.roll(u,-1,axis=0) - jnp.roll(u,1,axis=0)) / (2*dx)
    dv_dy = (jnp.roll(v,-1,axis=1) - jnp.roll(v,1,axis=1)) / (2*dy)
    return du_dx + dv_dy
```

### Curl / Vorticity (2D)

```python
@jit
def curl_2d(u, v, dx, dy):
    """ω = ∂v/∂x - ∂u/∂y"""
    dv_dx = (jnp.roll(v,-1,axis=0) - jnp.roll(v,1,axis=0)) / (2*dx)
    du_dy = (jnp.roll(u,-1,axis=1) - jnp.roll(u,1,axis=1)) / (2*dy)
    return dv_dx - du_dy
```

### Advection with Upwind (2D)

```python
@jit
def advection_upwind_2d(u, v, phi, dx, dy):
    """(u·∇)φ with upwind scheme."""
    phi_xm, phi_xp = jnp.roll(phi,1,axis=0), jnp.roll(phi,-1,axis=0)
    phi_ym, phi_yp = jnp.roll(phi,1,axis=1), jnp.roll(phi,-1,axis=1)
    
    dphi_dx = jnp.where(u >= 0, (phi-phi_xm)/dx, (phi_xp-phi)/dx)
    dphi_dy = jnp.where(v >= 0, (phi-phi_ym)/dy, (phi_yp-phi)/dy)
    
    return u * dphi_dx + v * dphi_dy
```

---

## 3D Stencils

### Laplacian 3D

```python
@jit
def laplacian_3d(f, dx, dy, dz):
    """∇²f in 3D."""
    d2x = (jnp.roll(f,-1,axis=0) - 2*f + jnp.roll(f,1,axis=0)) / dx**2
    d2y = (jnp.roll(f,-1,axis=1) - 2*f + jnp.roll(f,1,axis=1)) / dy**2
    d2z = (jnp.roll(f,-1,axis=2) - 2*f + jnp.roll(f,1,axis=2)) / dz**2
    return d2x + d2y + d2z
```

---

## Boundary Conditions

### Dirichlet (Fixed Value)

```python
@jit
def apply_dirichlet_1d(f, left, right):
    return f.at[0].set(left).at[-1].set(right)

@jit
def apply_dirichlet_2d(f, left, right, bottom, top):
    f = f.at[0,:].set(left)
    f = f.at[-1,:].set(right)
    f = f.at[:,0].set(bottom)
    f = f.at[:,-1].set(top)
    return f
```

### Neumann (Zero Gradient)

```python
@jit
def apply_neumann_1d(f):
    """Zero-gradient (insulated) BC."""
    return f.at[0].set(f[1]).at[-1].set(f[-2])

@jit
def apply_neumann_flux_1d(f, q_left, q_right, dx, k):
    """Fixed heat flux: q = -k dT/dx"""
    f = f.at[0].set(f[1] + q_left * dx / k)
    f = f.at[-1].set(f[-2] - q_right * dx / k)
    return f
```

### Ghost Cell Padding

```python
@jit
def add_ghost_cells(f, bc_type='periodic'):
    if bc_type == 'periodic':
        return jnp.pad(f, 1, mode='wrap')
    elif bc_type == 'neumann':
        return jnp.pad(f, 1, mode='edge')
    elif bc_type == 'dirichlet':
        return jnp.pad(f, 1, mode='constant', constant_values=0)
```

---

## Time Integration

### Forward Euler

```python
@jit
def euler_step(f, dfdt_fn, dt, *args):
    return f + dt * dfdt_fn(f, *args)
```

### Runge-Kutta 4

```python
@jit
def rk4_step(f, dfdt_fn, dt, *args):
    k1 = dfdt_fn(f, *args)
    k2 = dfdt_fn(f + 0.5*dt*k1, *args)
    k3 = dfdt_fn(f + 0.5*dt*k2, *args)
    k4 = dfdt_fn(f + dt*k3, *args)
    return f + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
```

### Time Loop with lax.scan

```python
from jax.lax import scan

def run_simulation(f_init, dfdt_fn, dt, n_steps, *args):
    def step(f, _):
        f_new = euler_step(f, dfdt_fn, dt, *args)
        return f_new, f_new
    
    f_final, f_history = scan(step, f_init, None, length=n_steps)
    return f_final, f_history
```

---

## Complete Example: 1D Heat Equation

```python
from jax import config
config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
from jax.lax import scan

@jax.jit
def solve_heat_1d(T_init, alpha, dx, dt, n_steps, T_left, T_right):
    """Solve 1D heat equation with Dirichlet BCs."""
    
    # Check CFL
    cfl = alpha * dt / dx**2
    # assert cfl <= 0.5  # Uncomment for safety
    
    def step(T, _):
        dTdt = alpha * (jnp.roll(T,-1) - 2*T + jnp.roll(T,1)) / dx**2
        T_new = T + dt * dTdt
        T_new = T_new.at[0].set(T_left)
        T_new = T_new.at[-1].set(T_right)
        return T_new, T_new
    
    return scan(step, T_init, None, length=n_steps)

# Usage
nx = 101
L = 1.0
dx = L / (nx - 1)
alpha = 1e-4
dt = 0.4 * dx**2 / alpha  # CFL = 0.4

x = jnp.linspace(0, L, nx)
T_init = jnp.sin(jnp.pi * x / L) * 100

T_final, T_history = solve_heat_1d(T_init, alpha, dx, dt, 1000, 0.0, 0.0)
```

---

## Stability Conditions

| Equation | CFL Condition | Safe Value |
|----------|---------------|------------|
| Heat (explicit) | α·dt/dx² ≤ 0.5 | 0.4 |
| Advection | u·dt/dx ≤ 1.0 | 0.8 |
| Wave | c·dt/dx ≤ 1.0 | 0.8 |

```python
def check_heat_cfl(alpha, dx, dt):
    cfl = alpha * dt / dx**2
    if cfl > 0.5:
        max_dt = 0.5 * dx**2 / alpha
        raise ValueError(f"CFL={cfl:.3f}>0.5. Use dt<={max_dt:.2e}")
    return cfl
```
