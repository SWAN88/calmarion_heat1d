"""
Validation tests for 1D Heat Diffusion JAX implementation.

Tests compare JAX results against Fortran reference data.
All numerical results must match within machine precision:
- Relative tolerance: rtol = 1e-10
- Absolute tolerance: atol = 1e-12
"""

# CRITICAL: Enable float64 BEFORE other JAX imports
from jax import config
config.update("jax_enable_x64", True)

import pytest
import jax.numpy as jnp
import numpy as np

from src_jax.params import HeatParams
from src_jax.init import init_grid, init_temperature, analytical_solution
from src_jax.simulation import (
    timestep_ftcs,
    compute_l2_error,
    compute_max_error,
    compute_total_heat,
    run_simulation,
)

# Validation tolerances
RTOL = 1e-10
ATOL = 1e-12


def assert_numerical_equal(result, reference, rtol=RTOL, atol=ATOL, name="result"):
    """Validate numerical result against reference."""
    result_np = np.asarray(result)
    reference_np = np.asarray(reference)

    if not np.allclose(result_np, reference_np, rtol=rtol, atol=atol):
        max_abs_err = np.max(np.abs(result_np - reference_np))
        max_rel_err = np.max(np.abs(result_np - reference_np) / (np.abs(reference_np) + 1e-15))
        pytest.fail(
            f"{name} validation FAILED:\n"
            f"  Max absolute error: {max_abs_err:.2e}\n"
            f"  Max relative error: {max_rel_err:.2e}\n"
            f"  Required: rtol={rtol:.0e}, atol={atol:.0e}"
        )


class TestParams:
    """Test parameter calculations."""

    def test_params_match_fortran(self, reference_params):
        """Verify parameters match Fortran reference."""
        params = HeatParams()

        assert params.nx == reference_params["nx"]
        assert params.L == reference_params["L"]
        assert params.alpha == reference_params["alpha"]
        assert params.cfl == reference_params["cfl"]
        assert np.isclose(params.dx, reference_params["dx"], rtol=RTOL)
        assert np.isclose(params.dt, reference_params["dt"], rtol=RTOL)
        assert params.n_steps == reference_params["n_steps"]

    def test_cfl_stability(self, params):
        """Verify CFL condition for stability."""
        assert params.r <= 0.5, f"CFL violation: r = {params.r} > 0.5"


class TestInitialization:
    """Test initialization routines."""

    def test_init_grid(self, params, x_ref):
        """Validate spatial grid against Fortran reference."""
        x = init_grid(params)

        assert x.shape == (params.nx,)
        assert x.dtype == jnp.float64
        assert_numerical_equal(x, x_ref, name="spatial grid")

    def test_init_temperature(self, params, x_ref, u_initial_ref):
        """Validate initial temperature against Fortran reference."""
        x = jnp.array(x_ref)
        u = init_temperature(x, params)

        assert u.shape == (params.nx,)
        assert u.dtype == jnp.float64
        assert_numerical_equal(u, u_initial_ref, name="initial temperature")

    def test_boundary_conditions(self, params):
        """Verify Dirichlet boundary conditions."""
        x = init_grid(params)
        u = init_temperature(x, params)

        assert u[0] == 0.0, "Left BC violated"
        assert u[-1] == 0.0, "Right BC violated"


class TestAnalyticalSolution:
    """Test analytical solution computation."""

    def test_analytical_at_final_time(self, params, x_ref, u_analytical_final_ref):
        """Validate analytical solution at t=1.0."""
        x = jnp.array(x_ref)
        t_final = 1.0
        u_exact = analytical_solution(x, t_final, params)

        assert_numerical_equal(u_exact, u_analytical_final_ref, name="analytical solution")

    def test_analytical_at_t0_matches_initial(self, params, x_ref, u_initial_ref):
        """Analytical solution at t=0 should match initial condition."""
        x = jnp.array(x_ref)
        u_exact = analytical_solution(x, 0.0, params)

        assert_numerical_equal(u_exact, u_initial_ref, name="analytical at t=0")


class TestFTCSKernel:
    """Test FTCS time-stepping kernel."""

    def test_single_timestep_preserves_bcs(self, params, x_ref, u_initial_ref):
        """FTCS step should preserve boundary conditions."""
        u = jnp.array(u_initial_ref)
        u_new = timestep_ftcs(u, params)

        assert u_new[0] == 0.0, "Left BC violated after timestep"
        assert u_new[-1] == 0.0, "Right BC violated after timestep"

    def test_single_timestep_shape(self, params, x_ref, u_initial_ref):
        """FTCS step should preserve array shape."""
        u = jnp.array(u_initial_ref)
        u_new = timestep_ftcs(u, params)

        assert u_new.shape == u.shape
        assert u_new.dtype == jnp.float64

    def test_100_steps_match_reference(self, params, x_ref, u_initial_ref, u_step100_ref):
        """Validate numerical solution at step 100 against Fortran."""
        u = jnp.array(u_initial_ref)

        for _ in range(100):
            u = timestep_ftcs(u, params)

        # Use looser tolerance for accumulated numerical differences
        # FTCS is 1st order in time, 2nd order in space
        assert_numerical_equal(
            u, u_step100_ref,
            rtol=1e-10, atol=1e-10,
            name="solution at step 100"
        )

    def test_full_simulation_match_reference(self, params, x_ref, u_initial_ref, u_final_ref):
        """Validate final numerical solution against Fortran reference."""
        u = jnp.array(u_initial_ref)

        for _ in range(params.n_steps):
            u = timestep_ftcs(u, params)

        assert_numerical_equal(
            u, u_final_ref,
            rtol=1e-10, atol=1e-10,
            name="final solution"
        )


class TestErrorComputation:
    """Test error computation functions."""

    def test_l2_error_matches_fortran(self, params, x_ref, u_final_ref, u_analytical_final_ref):
        """Validate L2 error computation."""
        u = jnp.array(u_final_ref)
        u_exact = jnp.array(u_analytical_final_ref)

        l2_err = compute_l2_error(u, u_exact)

        # Fortran reference: 7.245839E-06
        expected_l2 = 7.245839e-06
        assert np.isclose(l2_err, expected_l2, rtol=1e-5), \
            f"L2 error mismatch: {l2_err:.6e} vs {expected_l2:.6e}"

    def test_max_error_matches_fortran(self, params, x_ref, u_final_ref, u_analytical_final_ref):
        """Validate max error computation."""
        u = jnp.array(u_final_ref)
        u_exact = jnp.array(u_analytical_final_ref)

        max_err = compute_max_error(u, u_exact)

        # Fortran reference: 1.029827E-05
        expected_max = 1.029827e-05
        assert np.isclose(max_err, expected_max, rtol=1e-5), \
            f"Max error mismatch: {max_err:.6e} vs {expected_max:.6e}"


class TestTotalHeat:
    """Test total heat (integral) computation."""

    def test_initial_total_heat(self, params, x_ref, u_initial_ref, history_total_heat_ref):
        """Validate initial total heat."""
        u = jnp.array(u_initial_ref)
        total = compute_total_heat(u, params.dx)

        # Fortran reference from history: 0.63656741
        expected = history_total_heat_ref[0]
        assert np.isclose(total, expected, rtol=1e-8), \
            f"Initial total heat mismatch: {total:.8f} vs {expected:.8f}"

    def test_final_total_heat(self, params, x_ref, u_final_ref, history_total_heat_ref):
        """Validate final total heat."""
        u = jnp.array(u_final_ref)
        total = compute_total_heat(u, params.dx)

        # Fortran reference from history: 0.57673501
        expected = history_total_heat_ref[-1]
        assert np.isclose(total, expected, rtol=1e-8), \
            f"Final total heat mismatch: {total:.8f} vs {expected:.8f}"


class TestRunSimulation:
    """Test the lax.scan-based simulation runner."""

    def test_run_simulation_final_state(self, params, x_ref, u_initial_ref, u_final_ref):
        """Validate run_simulation produces correct final state."""
        u_init = jnp.array(u_initial_ref)

        u_final, u_history = run_simulation(u_init, params, params.n_steps)

        assert_numerical_equal(
            u_final, u_final_ref,
            rtol=1e-10, atol=1e-10,
            name="run_simulation final"
        )

    def test_run_simulation_history_shape(self, params, u_initial_ref):
        """Validate run_simulation returns correct history shape."""
        u_init = jnp.array(u_initial_ref)

        u_final, u_history = run_simulation(u_init, params, params.n_steps)

        assert u_history.shape == (params.n_steps + 1, params.nx)

    def test_run_simulation_history_initial(self, params, u_initial_ref):
        """First entry in history should be initial condition."""
        u_init = jnp.array(u_initial_ref)

        u_final, u_history = run_simulation(u_init, params, params.n_steps)

        assert_numerical_equal(
            u_history[0], u_initial_ref,
            name="history initial condition"
        )


class TestConvergence:
    """Test numerical convergence properties."""

    def test_error_decreases_with_finer_grid(self):
        """L2 error should decrease as grid is refined."""
        errors = []
        for nx in [51, 101, 201]:
            params = HeatParams(nx=nx)
            x = init_grid(params)
            u = init_temperature(x, params)

            for _ in range(params.n_steps):
                u = timestep_ftcs(u, params)

            u_exact = analytical_solution(x, params.t_end, params)
            l2_err = compute_l2_error(u, u_exact)
            errors.append(float(l2_err))

        # Error should decrease with refinement
        assert errors[1] < errors[0], "Error did not decrease from nx=51 to nx=101"
        assert errors[2] < errors[1], "Error did not decrease from nx=101 to nx=201"

    def test_spatial_convergence_rate(self):
        """FTCS should have 2nd order spatial convergence."""
        errors = []
        dx_values = []

        for nx in [51, 101, 201]:
            params = HeatParams(nx=nx)
            x = init_grid(params)
            u = init_temperature(x, params)

            for _ in range(params.n_steps):
                u = timestep_ftcs(u, params)

            u_exact = analytical_solution(x, params.t_end, params)
            l2_err = compute_l2_error(u, u_exact)
            errors.append(float(l2_err))
            dx_values.append(params.dx)

        # Compute convergence rate: error ~ dx^p
        # p = log(e1/e2) / log(dx1/dx2)
        rate = np.log(errors[0] / errors[1]) / np.log(dx_values[0] / dx_values[1])

        # FTCS is 1st order in time, 2nd order in space
        # With CFL fixed, we expect ~2nd order overall
        assert rate > 1.5, f"Convergence rate {rate:.2f} is too low (expected ~2)"
