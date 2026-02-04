#!/usr/bin/env python3
"""
1D Heat Equation Solver Template

Solves: ∂T/∂t = α ∂²T/∂x²

Usage:
    python heat1d.py

This is a template for scientific computing migration projects.
Modify parameters and boundary conditions as needed.
"""

# CRITICAL: Enable float64 BEFORE other JAX imports
from jax import config
config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
from jax.lax import scan
from typing import NamedTuple
import matplotlib.pyplot as plt


class HeatParams(NamedTuple):
    """Simulation parameters."""
    nx: int           # Number of grid points
    L: float          # Domain length
    alpha: float      # Thermal diffusivity
    T_left: float     # Left boundary temperature
    T_right: float    # Right boundary temperature
    dt: float         # Time step
    n_steps: int      # Number of time steps
    
    @property
    def dx(self) -> float:
        return self.L / (self.nx - 1)
    
    @property
    def cfl(self) -> float:
        return self.alpha * self.dt / self.dx**2


class SimState(NamedTuple):
    """Simulation state."""
    T: jnp.ndarray    # Temperature field
    t: float          # Current time
    step: int         # Step counter


@jax.jit
def compute_laplacian(T: jnp.ndarray, dx: float) -> jnp.ndarray:
    """Compute d²T/dx² using central difference."""
    return (jnp.roll(T, -1) - 2*T + jnp.roll(T, 1)) / dx**2


@jax.jit
def apply_boundary_conditions(T: jnp.ndarray, T_left: float, T_right: float) -> jnp.ndarray:
    """Apply Dirichlet boundary conditions."""
    return T.at[0].set(T_left).at[-1].set(T_right)


@jax.jit
def heat_step(state: SimState, params: HeatParams) -> SimState:
    """Single time step using Forward Euler."""
    # Compute diffusion term
    laplacian = compute_laplacian(state.T, params.dx)
    
    # Update temperature (Euler step)
    T_new = state.T + params.dt * params.alpha * laplacian
    
    # Apply boundary conditions
    T_new = apply_boundary_conditions(T_new, params.T_left, params.T_right)
    
    return SimState(
        T=T_new,
        t=state.t + params.dt,
        step=state.step + 1
    )


def run_simulation(initial_state: SimState, params: HeatParams) -> tuple[SimState, jnp.ndarray]:
    """Run simulation using lax.scan for JIT efficiency."""
    
    def step_fn(state, _):
        new_state = heat_step(state, params)
        return new_state, new_state.T
    
    final_state, T_history = scan(step_fn, initial_state, None, length=params.n_steps)
    return final_state, T_history


def analytical_solution(x: jnp.ndarray, t: float, alpha: float, L: float) -> jnp.ndarray:
    """Analytical solution for T(x,0) = sin(πx/L), T(0,t) = T(L,t) = 0."""
    return jnp.sin(jnp.pi * x / L) * jnp.exp(-alpha * (jnp.pi / L)**2 * t)


def main():
    # Parameters
    params = HeatParams(
        nx=101,
        L=1.0,
        alpha=1e-4,
        T_left=0.0,
        T_right=0.0,
        dt=0.0,  # Will be computed for CFL
        n_steps=1000
    )
    
    # Compute stable time step (CFL = 0.4 < 0.5)
    dx = params.dx
    dt = 0.4 * dx**2 / params.alpha
    params = params._replace(dt=dt)
    
    print(f"Grid: nx={params.nx}, dx={params.dx:.4e}")
    print(f"Time step: dt={params.dt:.4e}")
    print(f"CFL number: {params.cfl:.3f}")
    
    # Check stability
    assert params.cfl <= 0.5, f"CFL = {params.cfl:.3f} > 0.5, unstable!"
    
    # Initial condition: sin(πx/L)
    x = jnp.linspace(0, params.L, params.nx)
    T_init = jnp.sin(jnp.pi * x / params.L) * 100  # Amplitude 100
    
    initial_state = SimState(T=T_init, t=0.0, step=0)
    
    # Run simulation
    print(f"\nRunning {params.n_steps} steps...")
    final_state, T_history = run_simulation(initial_state, params)
    print(f"Final time: t = {final_state.t:.4f}")
    
    # Compare with analytical solution
    T_analytical = analytical_solution(x, final_state.t, params.alpha, params.L) * 100
    
    # Validation
    max_error = jnp.max(jnp.abs(final_state.T - T_analytical))
    rel_error = max_error / jnp.max(jnp.abs(T_analytical))
    
    print(f"\nValidation against analytical solution:")
    print(f"  Max absolute error: {max_error:.2e}")
    print(f"  Max relative error: {rel_error:.2e}")
    
    if jnp.allclose(final_state.T, T_analytical, rtol=1e-2):
        print("  ✓ PASSED (within 1% tolerance)")
    else:
        print("  ✗ FAILED")
    
    # Plot results
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(x, T_init, 'b--', label='Initial')
    plt.plot(x, final_state.T, 'r-', label='Final (JAX)')
    plt.plot(x, T_analytical, 'g:', label='Analytical', linewidth=2)
    plt.xlabel('x')
    plt.ylabel('T')
    plt.legend()
    plt.title('Temperature Distribution')
    
    plt.subplot(1, 2, 2)
    plt.imshow(T_history[::10, :].T, aspect='auto', origin='lower',
               extent=[0, final_state.t, 0, params.L])
    plt.colorbar(label='T')
    plt.xlabel('Time')
    plt.ylabel('x')
    plt.title('Temperature Evolution')
    
    plt.tight_layout()
    plt.savefig('heat1d_result.png', dpi=150)
    plt.show()
    print("\nPlot saved to heat1d_result.png")


if __name__ == "__main__":
    main()
