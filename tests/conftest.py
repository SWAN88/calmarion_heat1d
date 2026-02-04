"""Pytest configuration and fixtures for heat1d tests."""

# CRITICAL: Enable float64 BEFORE other JAX imports
from jax import config
config.update("jax_enable_x64", True)

import pytest
import json
import numpy as np
from pathlib import Path

from src_jax.params import HeatParams


@pytest.fixture
def reference_dir() -> Path:
    """Path to reference data directory."""
    return Path(__file__).parent / "reference_data"


@pytest.fixture
def params() -> HeatParams:
    """Default simulation parameters matching Fortran."""
    return HeatParams()


@pytest.fixture
def reference_params(reference_dir: Path) -> dict:
    """Load reference parameters from JSON."""
    with open(reference_dir / "params.json") as f:
        return json.load(f)


@pytest.fixture
def x_ref(reference_dir: Path) -> np.ndarray:
    """Reference spatial grid."""
    return np.load(reference_dir / "x.npy")


@pytest.fixture
def u_initial_ref(reference_dir: Path) -> np.ndarray:
    """Reference initial temperature field."""
    return np.load(reference_dir / "u_initial.npy")


@pytest.fixture
def u_final_ref(reference_dir: Path) -> np.ndarray:
    """Reference final temperature field (step 250)."""
    return np.load(reference_dir / "u_final.npy")


@pytest.fixture
def u_step100_ref(reference_dir: Path) -> np.ndarray:
    """Reference temperature field at step 100."""
    return np.load(reference_dir / "u_step100.npy")


@pytest.fixture
def u_analytical_final_ref(reference_dir: Path) -> np.ndarray:
    """Reference analytical solution at final time."""
    return np.load(reference_dir / "u_analytical_final.npy")


@pytest.fixture
def history_total_heat_ref(reference_dir: Path) -> np.ndarray:
    """Reference total heat values."""
    return np.load(reference_dir / "history_total_heat.npy")
