# Inverse Problems with JAX Auto-Differentiation

Leveraging JAX's automatic differentiation for parameter estimation and optimization.

## Overview

JAX enables efficient gradient-based optimization for inverse problems:
- Parameter estimation (thermal conductivity, diffusivity)
- Source term identification
- Boundary condition reconstruction
- Nanoparticle optical property fitting

---

## Basic Pattern

```python
from jax import config
config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
from jax import grad, value_and_grad, jit
import optax

# Generic inverse problem structure:
# 1. Forward model: params → prediction
# 2. Loss function: (prediction, observation) → scalar
# 3. Gradient: d(loss)/d(params) via auto-diff
# 4. Optimization: update params to minimize loss
```

---

## Heat Transfer Inverse Problems

### Estimate Thermal Conductivity

```python
@jit
def forward_heat_1d(k, T_left, T_right, L, nx):
    """
    Solve steady-state 1D heat equation.
    For uniform k, analytical solution is linear.
    """
    x = jnp.linspace(0, L, nx)
    T = T_left + (T_right - T_left) * x / L
    return T


@jit
def forward_heat_1d_transient(k, rho, cp, T_init, T_left, T_right, dx, dt, n_steps):
    """Solve transient 1D heat equation with FTCS scheme."""
    alpha = k / (rho * cp)
    r = alpha * dt / dx**2
    
    def step(T, _):
        T_new = T.at[1:-1].set(
            T[1:-1] + r * (T[2:] - 2*T[1:-1] + T[:-2])
        )
        T_new = T_new.at[0].set(T_left)
        T_new = T_new.at[-1].set(T_right)
        return T_new, T_new
    
    T_final, _ = jax.lax.scan(step, T_init, None, length=n_steps)
    return T_final


def estimate_conductivity(T_observed, T_left, T_right, L, nx, k_init=1.0, n_iter=100):
    """Estimate thermal conductivity from observed temperature."""
    
    def loss_fn(k):
        T_pred = forward_heat_1d(k, T_left, T_right, L, nx)
        return jnp.mean((T_pred - T_observed)**2)
    
    optimizer = optax.adam(0.1)
    k = k_init
    opt_state = optimizer.init(k)
    
    for i in range(n_iter):
        loss, grads = value_and_grad(loss_fn)(k)
        updates, opt_state = optimizer.update(grads, opt_state)
        k = optax.apply_updates(k, updates)
        
        if i % 20 == 0:
            print(f"Iter {i}: loss = {loss:.6e}, k = {k:.6f}")
    
    return k
```

### Estimate Source Term Distribution

```python
@jit
def forward_with_source(T_init, source, k, rho, cp, dx, dt, n_steps):
    """
    Heat equation with spatially varying source:
    dT/dt = α d²T/dx² + Q/(ρ cp)
    """
    alpha = k / (rho * cp)
    r = alpha * dt / dx**2
    source_term = source / (rho * cp) * dt
    
    def step(T, _):
        diffusion = r * (jnp.roll(T, -1) - 2*T + jnp.roll(T, 1))
        T_new = T + diffusion + source_term
        T_new = T_new.at[0].set(T[0])  # Keep BCs
        T_new = T_new.at[-1].set(T[-1])
        return T_new, None
    
    T_final, _ = jax.lax.scan(step, T_init, None, length=n_steps)
    return T_final


def estimate_source(T_observed, T_init, k, rho, cp, dx, dt, n_steps, n_iter=500):
    """Estimate source distribution from observed temperature."""
    nx = len(T_init)
    source_init = jnp.zeros(nx)
    
    def loss_fn(source):
        T_pred = forward_with_source(T_init, source, k, rho, cp, dx, dt, n_steps)
        data_loss = jnp.mean((T_pred - T_observed)**2)
        reg_loss = 0.01 * jnp.mean(jnp.diff(source)**2)  # Smoothness
        return data_loss + reg_loss
    
    optimizer = optax.adam(0.01)
    source = source_init
    opt_state = optimizer.init(source)
    
    for _ in range(n_iter):
        loss, grads = value_and_grad(loss_fn)(source)
        updates, opt_state = optimizer.update(grads, opt_state)
        source = optax.apply_updates(source, updates)
    
    return source
```

---

## Boundary Condition Estimation

### Estimate Unknown Heat Flux

```python
@jit
def forward_neumann_bc(T_init, q_left, T_right, k, rho, cp, dx, dt, n_steps):
    """
    Solve with Neumann BC (heat flux) at left boundary.
    q = -k dT/dx → T[0] = T[1] + q*dx/k
    """
    alpha = k / (rho * cp)
    r = alpha * dt / dx**2
    
    def step(T, _):
        T_new = T.at[1:-1].set(
            T[1:-1] + r * (T[2:] - 2*T[1:-1] + T[:-2])
        )
        T_new = T_new.at[0].set(T[1] + q_left * dx / k)  # Neumann
        T_new = T_new.at[-1].set(T_right)  # Dirichlet
        return T_new, None
    
    T_final, _ = jax.lax.scan(step, T_init, None, length=n_steps)
    return T_final


def estimate_heat_flux(T_observed, T_init, T_right, k, rho, cp, 
                       dx, dt, n_steps, sensor_indices):
    """
    Estimate boundary heat flux from interior temperature sensors.
    """
    def loss_fn(q_left):
        T_pred = forward_neumann_bc(T_init, q_left, T_right, k, rho, cp, dx, dt, n_steps)
        return jnp.mean((T_pred[sensor_indices] - T_observed[sensor_indices])**2)
    
    optimizer = optax.adam(1.0)
    q = 0.0
    opt_state = optimizer.init(q)
    
    for _ in range(200):
        loss, grads = value_and_grad(loss_fn)(q)
        updates, opt_state = optimizer.update(grads, opt_state)
        q = optax.apply_updates(q, updates)
    
    return q
```

---

## Nanoparticle Parameter Fitting

```python
from typing import NamedTuple

class NanoparticleParams(NamedTuple):
    """Drude model parameters for nanoparticle optical response."""
    radius: float       # Particle radius [nm]
    eps_inf: float      # High-frequency permittivity
    omega_p: float      # Plasma frequency [eV]
    gamma: float        # Damping rate [eV]


@jit
def drude_permittivity(omega, params):
    """Drude model for metal permittivity."""
    return params.eps_inf - params.omega_p**2 / (omega**2 + 1j * params.gamma * omega)


@jit
def mie_extinction_quasistatic(omega, params, eps_medium=1.0):
    """Quasi-static Mie extinction for small particles."""
    eps = drude_permittivity(omega, params)
    alpha = 4 * jnp.pi * params.radius**3 * (eps - eps_medium) / (eps + 2*eps_medium)
    k = omega / 197.3  # Convert eV to 1/nm
    return k * jnp.imag(alpha)


def fit_nanoparticle_spectrum(omega_exp, sigma_exp, params_init, n_iter=1000):
    """Fit nanoparticle parameters to experimental spectrum."""
    
    def params_to_array(p):
        return jnp.array([p.radius, p.eps_inf, p.omega_p, p.gamma])
    
    def array_to_params(arr):
        return NanoparticleParams(arr[0], arr[1], arr[2], arr[3])
    
    def loss_fn(params_arr):
        params = array_to_params(params_arr)
        sigma_pred = mie_extinction_quasistatic(omega_exp, params)
        return jnp.mean((sigma_pred - sigma_exp)**2)
    
    params_arr = params_to_array(params_init)
    optimizer = optax.adam(0.001)
    opt_state = optimizer.init(params_arr)
    
    for _ in range(n_iter):
        loss, grads = value_and_grad(loss_fn)(params_arr)
        updates, opt_state = optimizer.update(grads, opt_state)
        params_arr = optax.apply_updates(params_arr, updates)
        params_arr = jnp.maximum(params_arr, 1e-6)  # Enforce positivity
    
    return array_to_params(params_arr)
```

---

## Sensitivity Analysis

```python
from jax import jacobian, hessian

def sensitivity_analysis(forward_fn, params):
    """
    Compute sensitivity of output to each parameter.
    Returns Jacobian: d(output_i) / d(param_j)
    """
    J = jacobian(forward_fn)(params)
    return J


def uncertainty_propagation(forward_fn, params, param_covariance):
    """
    Propagate parameter uncertainty to output.
    Linear approximation: Σ_output ≈ J Σ_params J^T
    """
    J = jacobian(forward_fn)(params)
    return J @ param_covariance @ J.T
```

---

## Best Practices

### 1. Parameter Scaling

```python
# Normalize parameters to similar magnitudes
scale_factors = jnp.array([1e3, 1.0, 1.0, 1e-2])  # Example
params_normalized = params / scale_factors

def loss_scaled(params_norm):
    params = params_norm * scale_factors
    return loss_fn(params)
```

### 2. Regularization

```python
def loss_with_regularization(params):
    data_loss = jnp.mean((forward(params) - observation)**2)
    l2_reg = 0.01 * jnp.sum(params**2)  # Tikhonov
    tv_reg = 0.001 * jnp.sum(jnp.abs(jnp.diff(params)))  # Total variation
    return data_loss + l2_reg + tv_reg
```

### 3. Multi-Start Optimization

```python
def multi_start_optimize(loss_fn, bounds, n_starts=10):
    """Run from multiple initial points."""
    best_loss, best_params = jnp.inf, None
    key = jax.random.PRNGKey(0)
    
    for _ in range(n_starts):
        key, subkey = jax.random.split(key)
        params_init = jax.random.uniform(subkey, minval=bounds[0], maxval=bounds[1])
        params_opt = optimize(loss_fn, params_init)
        loss = loss_fn(params_opt)
        if loss < best_loss:
            best_loss, best_params = loss, params_opt
    
    return best_params
```

### 4. Gradient Verification

```python
def check_gradients(loss_fn, params, eps=1e-5):
    """Verify auto-diff against finite differences."""
    grad_auto = grad(loss_fn)(params)
    grad_fd = jnp.zeros_like(params)
    
    for i in range(len(params)):
        params_plus = params.at[i].add(eps)
        params_minus = params.at[i].add(-eps)
        grad_fd = grad_fd.at[i].set(
            (loss_fn(params_plus) - loss_fn(params_minus)) / (2 * eps)
        )
    
    rel_error = jnp.abs(grad_auto - grad_fd) / (jnp.abs(grad_auto) + 1e-10)
    print(f"Max gradient error: {jnp.max(rel_error):.2e}")
    return jnp.allclose(grad_auto, grad_fd, rtol=1e-4)
```
