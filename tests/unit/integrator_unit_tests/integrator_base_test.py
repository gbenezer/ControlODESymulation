# Copyright (C) 2025 Gil Benezer
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
Unit Tests for IntegratorBase
=============================

Tests the abstract base class for ODE integrators, including:
1. StepMode enum definitions
2. IntegrationResult TypedDict
3. Abstract interface enforcement
4. Initialization and validation
5. Statistics tracking
6. String representations
7. Equilibrium-based integration
8. Autonomous system integration
9. Backend consistency
10. Control function handling
11. Array dimension validation
12. Time span validation
13. Integration termination conditions
14. Dense output handling
15. Step size handling
"""

import warnings

import numpy as np
import pytest

from src.systems.base.numerical_integration.integrator_base import (
    IntegratorBase,
    StepMode,
)
from src.types.trajectories import IntegrationResult

# ============================================================================
# Helper Functions
# ============================================================================


def _torch_available():
    """Check if PyTorch is available."""
    try:
        import torch

        return True
    except ImportError:
        return False


def _jax_available():
    """Check if JAX is available."""
    try:
        import jax.numpy as jnp

        return True
    except ImportError:
        return False


# ============================================================================
# Mock Systems
# ============================================================================


class MockSystem:
    """Basic mock system for testing."""

    def __init__(self, nx=1, nu=1):
        self.nx = nx
        self.nu = nu
        self._initialized = True
        self._default_backend = "numpy"

    def __call__(self, x, u, backend="numpy"):
        """Simple linear dynamics: dx = -x + u"""
        return -x + u


class MockAutonomousSystem:
    """Mock autonomous system (nu=0) for testing."""

    def __init__(self, nx=2):
        self.nx = nx
        self.nu = 0
        self._initialized = True
        self._default_backend = "numpy"

        self.equilibria = MockEquilibriumHandler(nx=nx, nu=0)
        self.equilibria.add_equilibrium("zero", np.zeros(nx), np.array([]))

    def __call__(self, x, u=None, backend="numpy"):
        """Autonomous dynamics: dx = -x"""
        return -x

    def get_equilibrium(self, name=None, backend=None):
        """Get equilibrium in specified backend."""
        backend = backend or self._default_backend
        return self.equilibria.get_both(name, backend)


class MockSystemWithEquilibria:
    """Mock system with equilibrium handler for testing."""

    def __init__(self, nx=2, nu=1):
        self.nx = nx
        self.nu = nu
        self._initialized = True
        self._default_backend = "numpy"

        self.equilibria = MockEquilibriumHandler(nx=nx, nu=nu)
        self.equilibria.add_equilibrium("zero", np.zeros(nx), np.zeros(nu))
        self.equilibria.add_equilibrium("custom", np.array([1.0, 0.5]), np.array([0.2]))

    def __call__(self, x, u, backend="numpy"):
        """Simple linear dynamics: dx = -x + u"""
        return -x + u

    def get_equilibrium(self, name=None, backend=None):
        """Get equilibrium in specified backend."""
        backend = backend or self._default_backend
        return self.equilibria.get_both(name, backend)

    def list_equilibria(self):
        """List equilibrium names."""
        return self.equilibria.list_names()


class MockEquilibriumHandler:
    """Minimal mock of EquilibriumHandler for testing."""

    def __init__(self, nx, nu):
        self._nx = nx
        self._nu = nu
        self._equilibria = {}
        self._default = "origin"

        # Origin equilibrium
        self._equilibria["origin"] = {
            "x": np.zeros(nx),
            "u": np.zeros(nu) if nu > 0 else np.array([]),
            "metadata": {},
        }

    def add_equilibrium(self, name, x_eq, u_eq):
        """Add equilibrium."""
        self._equilibria[name] = {"x": np.asarray(x_eq), "u": np.asarray(u_eq), "metadata": {}}

    def get_x(self, name=None, backend="numpy"):
        """Get equilibrium state."""
        name = name or self._default
        x = self._equilibria[name]["x"]
        return self._convert_to_backend(x, backend)

    def get_u(self, name=None, backend="numpy"):
        """Get equilibrium control."""
        name = name or self._default
        u = self._equilibria[name]["u"]
        return self._convert_to_backend(u, backend)

    def get_both(self, name=None, backend="numpy"):
        """Get both state and control."""
        return self.get_x(name, backend), self.get_u(name, backend)

    def list_names(self):
        """List equilibrium names."""
        return list(self._equilibria.keys())

    def _convert_to_backend(self, arr, backend):
        """Convert to backend."""
        if backend == "numpy":
            return arr
        elif backend == "torch" and _torch_available():
            import torch

            return torch.tensor(arr, dtype=torch.float64)
        elif backend == "jax" and _jax_available():
            import jax.numpy as jnp

            return jnp.array(arr)
        return arr


# ============================================================================
# Concrete Test Integrator
# ============================================================================


class ConcreteTestIntegrator(IntegratorBase):
    """
    Minimal concrete integrator for testing base class.

    Note: Named 'ConcreteTestIntegrator' instead of 'TestIntegrator'
    to avoid pytest collection warning.
    """

    def step(self, x, u, dt=None):
        """Simple Euler step."""
        dt = dt or self.dt
        # Use _evaluate_dynamics for proper statistics tracking
        f = self._evaluate_dynamics(x, u)
        self._stats["total_steps"] += 1
        return x + dt * f

    def integrate(self, x0, u_func=None, t_span=(0, 1), t_eval=None, dense_output=False):
        """Basic integration loop."""
        t0, tf = t_span
        t = t0
        x = np.asarray(x0)

        t_history = [t]
        x_history = [x.copy()]

        steps = 0
        while t < tf and steps < self.max_steps:
            if u_func is not None:
                u = u_func(t)
            else:
                u = np.array([]) if self.system.nu == 0 else np.zeros(self.system.nu)

            x = self.step(x, u)
            t += self.dt

            t_history.append(t)
            x_history.append(x.copy())

            steps += 1
            # Note: total_steps is incremented in step()

        # Return TypedDict result
        result: IntegrationResult = {
            "t": np.array(t_history),
            "x": np.array(x_history),
            "success": True,
            "nsteps": steps,
            "nfev": self._stats["total_fev"],
            "solver": self.name,
        }
        return result

    @property
    def name(self):
        return "ConcreteTestIntegrator"


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def mock_system():
    """Fixture for basic mock system."""
    return MockSystem(nx=1, nu=1)


@pytest.fixture
def mock_autonomous_system():
    """Fixture for autonomous mock system."""
    return MockAutonomousSystem(nx=2)


@pytest.fixture
def mock_system_with_equilibria():
    """Fixture for mock system with equilibria."""
    return MockSystemWithEquilibria(nx=2, nu=1)


@pytest.fixture
def integrator(mock_system):
    """Fixture for basic integrator."""
    return ConcreteTestIntegrator(mock_system, dt=0.01, backend="numpy")


@pytest.fixture
def integrator_autonomous(mock_autonomous_system):
    """Fixture for autonomous integrator."""
    return ConcreteTestIntegrator(mock_autonomous_system, dt=0.01, backend="numpy")


@pytest.fixture
def integrator_with_equilibria(mock_system_with_equilibria):
    """Fixture for integrator with equilibria."""
    return ConcreteTestIntegrator(mock_system_with_equilibria, dt=0.01, backend="numpy")


# ============================================================================
# Test Class: StepMode Enum
# ============================================================================


class TestStepModeEnum:
    """Test StepMode enumeration."""

    def test_fixed_mode_exists(self):
        """Test that FIXED mode is defined."""
        assert hasattr(StepMode, "FIXED")

    def test_adaptive_mode_exists(self):
        """Test that ADAPTIVE mode is defined."""
        assert hasattr(StepMode, "ADAPTIVE")

    def test_fixed_mode_value(self):
        """Test FIXED mode string value."""
        assert StepMode.FIXED.value == "fixed"

    def test_adaptive_mode_value(self):
        """Test ADAPTIVE mode string value."""
        assert StepMode.ADAPTIVE.value == "adaptive"


# ============================================================================
# Test Class: IntegrationResult TypedDict
# ============================================================================


class TestIntegrationResult:
    """Test IntegrationResult TypedDict structure."""

    def test_result_is_dict(self, integrator):
        """Test that result is a dict."""
        result = integrator.integrate(x0=np.array([1.0]), u_func=lambda t: np.array([0.0]))
        assert isinstance(result, dict)

    def test_result_has_required_fields(self, integrator):
        """Test that result has all required fields."""
        result = integrator.integrate(x0=np.array([1.0]), u_func=lambda t: np.array([0.0]))

        # Required fields
        assert "t" in result
        assert "x" in result
        assert "success" in result
        assert "nsteps" in result
        assert "solver" in result

    def test_result_field_types(self, integrator):
        """Test result field types."""
        result = integrator.integrate(
            x0=np.array([1.0]), u_func=lambda t: np.array([0.0]), t_span=(0, 0.5)
        )

        assert isinstance(result["t"], np.ndarray)
        assert isinstance(result["x"], np.ndarray)
        assert isinstance(result["success"], (bool, np.bool_))
        assert isinstance(result["nsteps"], (int, np.integer))
        assert isinstance(result["solver"], str)

    def test_result_dict_access(self, integrator):
        """Test dict-style access to result fields."""
        result = integrator.integrate(x0=np.array([1.0]), u_func=lambda t: np.array([0.0]))

        # Should work with dict access
        t = result["t"]
        x = result["x"]
        success = result["success"]

        assert t is not None
        assert x is not None
        assert success is not None


# ============================================================================
# Test Class: Abstract Interface
# ============================================================================


class TestAbstractInterface:
    """Test abstract interface enforcement."""

    def test_cannot_instantiate_base_class(self, mock_system):
        """Test that IntegratorBase cannot be instantiated directly."""
        with pytest.raises(TypeError):
            IntegratorBase(mock_system, dt=0.01)

    def test_concrete_class_implements_step(self):
        """Test that concrete class implements step()."""
        integrator = ConcreteTestIntegrator(MockSystem(), dt=0.01)
        assert hasattr(integrator, "step")
        assert callable(integrator.step)

    def test_concrete_class_implements_integrate(self):
        """Test that concrete class implements integrate()."""
        integrator = ConcreteTestIntegrator(MockSystem(), dt=0.01)
        assert hasattr(integrator, "integrate")
        assert callable(integrator.integrate)

    def test_concrete_class_implements_name(self):
        """Test that concrete class implements name property."""
        integrator = ConcreteTestIntegrator(MockSystem(), dt=0.01)
        assert hasattr(integrator, "name")
        assert isinstance(integrator.name, str)


# ============================================================================
# Test Class: Initialization and Validation
# ============================================================================


class TestInitializationAndValidation:
    """Test integrator initialization and parameter validation."""

    def test_initialization_with_required_params(self, mock_system):
        """Test initialization with required parameters."""
        integrator = ConcreteTestIntegrator(mock_system, dt=0.01)

        assert integrator.system is mock_system
        assert integrator.dt == 0.01
        assert integrator.backend == "numpy"

    def test_initialization_with_optional_params(self, mock_system):
        """Test initialization with optional parameters."""
        integrator = ConcreteTestIntegrator(
            mock_system, dt=0.01, backend="numpy", rtol=1e-8, atol=1e-10
        )

        assert integrator.rtol == 1e-8
        assert integrator.atol == 1e-10

    def test_fixed_mode_requires_dt(self, mock_system):
        """Test that FIXED mode requires dt."""
        with pytest.raises(ValueError, match="Time step dt is required"):
            ConcreteTestIntegrator(mock_system, dt=None, step_mode=StepMode.FIXED)

    def test_adaptive_mode_default_dt(self, mock_system):
        """Test that ADAPTIVE mode uses default dt if not provided."""
        integrator = ConcreteTestIntegrator(mock_system, dt=None, step_mode=StepMode.ADAPTIVE)

        assert integrator.dt == 0.01  # Default value

    def test_invalid_backend_raises_error(self, mock_system):
        """Test that invalid backend raises ValueError."""
        with pytest.raises(ValueError, match="Invalid backend"):
            ConcreteTestIntegrator(mock_system, dt=0.01, backend="invalid")

    def test_valid_backends(self, mock_system):
        """Test that valid backends are accepted."""
        for backend in ["numpy", "torch", "jax"]:
            integrator = ConcreteTestIntegrator(mock_system, dt=0.01, backend=backend)
            assert integrator.backend == backend


# ============================================================================
# Test Class: Statistics Tracking
# ============================================================================


class TestStatisticsTracking:
    """Test integration statistics tracking."""

    def test_stats_initialization(self, integrator):
        """Test that statistics are initialized to zero."""
        stats = integrator.get_stats()

        assert stats["total_steps"] == 0
        assert stats["total_fev"] == 0
        assert stats["total_time"] == 0.0

    def test_stats_updated_after_integration(self, integrator):
        """Test that statistics are updated after integration."""
        integrator.integrate(x0=np.array([1.0]), u_func=lambda t: np.array([0.0]), t_span=(0, 1))

        stats = integrator.get_stats()

        assert stats["total_steps"] > 0
        assert stats["total_fev"] > 0

    def test_reset_stats(self, integrator):
        """Test resetting statistics."""
        # Run integration
        integrator.integrate(x0=np.array([1.0]), u_func=lambda t: np.array([0.0]), t_span=(0, 1))

        # Reset
        integrator.reset_stats()

        stats = integrator.get_stats()
        assert stats["total_steps"] == 0
        assert stats["total_fev"] == 0
        assert stats["total_time"] == 0.0

    def test_avg_fev_per_step_calculation(self, integrator):
        """Test average function evaluations per step calculation."""
        integrator.integrate(x0=np.array([1.0]), u_func=lambda t: np.array([0.0]), t_span=(0, 1))

        stats = integrator.get_stats()

        assert "avg_fev_per_step" in stats
        assert stats["avg_fev_per_step"] > 0


# ============================================================================
# Test Class: String Representations
# ============================================================================


class TestStringRepresentations:
    """Test string representation methods."""

    def test_repr(self, integrator):
        """Test __repr__ method."""
        repr_str = repr(integrator)

        assert "ConcreteTestIntegrator" in repr_str
        assert "dt=" in repr_str
        assert "numpy" in repr_str

    def test_str(self, integrator):
        """Test __str__ method."""
        str_str = str(integrator)

        assert "ConcreteTestIntegrator" in str_str
        assert "0.01" in str_str  # dt value
        assert "numpy" in str_str

    def test_name_property(self, integrator):
        """Test name property."""
        assert integrator.name == "ConcreteTestIntegrator"


# ============================================================================
# Test Class: Equilibrium-Based Integration
# ============================================================================


class TestEquilibriumBasedIntegration:
    """Test integration from equilibrium points."""

    def test_integration_from_origin(self, integrator_with_equilibria):
        """Test integration from origin equilibrium."""
        x_eq, u_eq = integrator_with_equilibria.system.get_equilibrium("zero")

        result = integrator_with_equilibria.integrate(
            x0=x_eq, u_func=lambda t: u_eq, t_span=(0, 1)
        )

        assert result["success"]
        assert len(result["t"]) > 1

    def test_integration_from_custom_equilibrium(self, integrator_with_equilibria):
        """Test integration from custom equilibrium."""
        x_eq, u_eq = integrator_with_equilibria.system.get_equilibrium("custom")

        result = integrator_with_equilibria.integrate(
            x0=x_eq, u_func=lambda t: u_eq, t_span=(0, 1)
        )

        assert result["success"]

    def test_equilibrium_list(self, integrator_with_equilibria):
        """Test listing available equilibria."""
        eq_names = integrator_with_equilibria.system.list_equilibria()

        assert "origin" in eq_names
        assert "zero" in eq_names
        assert "custom" in eq_names


# ============================================================================
# Test Class: Autonomous System Integration
# ============================================================================


class TestAutonomousSystemIntegration:
    """Test integration of autonomous systems (nu=0)."""

    def test_autonomous_system_initialization(self, integrator_autonomous):
        """Test that autonomous system has nu=0."""
        assert integrator_autonomous.system.nu == 0

    def test_autonomous_integration_no_control(self, integrator_autonomous):
        """Test integration without control input."""
        x0 = np.array([1.0, 0.5])

        result = integrator_autonomous.integrate(x0=x0, u_func=None, t_span=(0, 1))

        assert result["success"]
        assert len(result["t"]) > 1
        assert result["x"].shape[1] == 2  # nx=2

    def test_autonomous_integration_from_equilibrium(self, integrator_autonomous):
        """Test autonomous integration from equilibrium."""
        x_eq, _ = integrator_autonomous.system.get_equilibrium("zero")

        result = integrator_autonomous.integrate(x0=x_eq, u_func=None, t_span=(0, 1))

        assert result["success"]


# ============================================================================
# Test Class: Backend Consistency
# ============================================================================


class TestBackendConsistency:
    """Test consistency across different backends."""

    def test_numpy_backend(self, mock_system):
        """Test NumPy backend."""
        integrator = ConcreteTestIntegrator(mock_system, dt=0.01, backend="numpy")

        result = integrator.integrate(
            x0=np.array([1.0]), u_func=lambda t: np.array([0.0]), t_span=(0, 0.5)
        )

        assert result["success"]
        assert isinstance(result["x"], np.ndarray)

    @pytest.mark.skipif(not _torch_available(), reason="PyTorch not available")
    def test_torch_backend(self, mock_system):
        """Test PyTorch backend."""
        import torch

        integrator = ConcreteTestIntegrator(mock_system, dt=0.01, backend="torch")

        # Note: Our mock integrator still uses NumPy internally
        result = integrator.integrate(
            x0=np.array([1.0]), u_func=lambda t: np.array([0.0]), t_span=(0, 0.5)
        )

        assert result["success"]

    @pytest.mark.skipif(not _jax_available(), reason="JAX not available")
    def test_jax_backend(self, mock_system):
        """Test JAX backend."""
        import jax.numpy as jnp

        integrator = ConcreteTestIntegrator(mock_system, dt=0.01, backend="jax")

        # Note: Our mock integrator still uses NumPy internally
        result = integrator.integrate(
            x0=np.array([1.0]), u_func=lambda t: np.array([0.0]), t_span=(0, 0.5)
        )

        assert result["success"]


# ============================================================================
# Test Class: Array Dimension Validation
# ============================================================================


class TestArrayDimensionValidation:
    """Test validation of array dimensions."""

    def test_1d_state_integration(self, integrator):
        """Test integration with 1D state."""
        x0 = np.array([1.0])

        result = integrator.integrate(x0=x0, u_func=lambda t: np.array([0.0]), t_span=(0, 1))

        assert result["success"]
        assert result["x"].shape[1] == 1

    def test_2d_state_integration(self, integrator_with_equilibria):
        """Test integration with 2D state."""
        x0 = np.array([1.0, 0.5])

        result = integrator_with_equilibria.integrate(
            x0=x0, u_func=lambda t: np.array([0.0]), t_span=(0, 1)
        )

        assert result["success"]
        assert result["x"].shape[1] == 2

    def test_output_shape_consistency(self, integrator_with_equilibria):
        """Test that output shape matches input."""
        x0 = np.array([1.0, 0.5])

        result = integrator_with_equilibria.integrate(
            x0=x0, u_func=lambda t: np.array([0.0]), t_span=(0, 1)
        )

        assert result["x"].shape[0] == len(result["t"])
        assert result["x"].shape[1] == len(x0)


# ============================================================================
# Test Class: Control Function Handling
# ============================================================================


class TestControlFunctionHandling:
    """Test various control function patterns."""

    def test_constant_control_function(self, integrator_with_equilibria):
        """Test integration with constant control."""
        u_const = np.array([0.5])

        result = integrator_with_equilibria.integrate(
            x0=np.array([0.0, 0.0]), u_func=lambda t: u_const, t_span=(0, 0.5)
        )

        assert result["success"]

    def test_time_varying_control_function(self, integrator_with_equilibria):
        """Test integration with time-varying control."""

        def u_func(t):
            return np.array([np.sin(t)])

        result = integrator_with_equilibria.integrate(
            x0=np.array([0.0, 0.0]), u_func=u_func, t_span=(0, 1)
        )

        assert result["success"]

    def test_none_control_autonomous(self, integrator_autonomous):
        """Test that None control works for autonomous systems."""
        result = integrator_autonomous.integrate(
            x0=np.array([1.0, 0.5]), u_func=None, t_span=(0, 1)
        )

        assert result["success"]

    def test_control_function_called_at_each_step(self, integrator_with_equilibria):
        """Test that control function is evaluated at each time step."""
        call_times = []

        def u_func(t):
            call_times.append(t)
            return np.array([0.0])

        result = integrator_with_equilibria.integrate(
            x0=np.array([0.0, 0.0]), u_func=u_func, t_span=(0, 1)
        )

        assert result["success"]
        assert len(call_times) > 5


# ============================================================================
# Test Class: Time Span Validation
# ============================================================================


class TestTimeSpanValidation:
    """Test validation of time span parameters."""

    def test_negative_time_span(self, integrator_with_equilibria):
        """Test integration with negative time direction."""
        x0 = np.array([1.0, 0.5])

        result = integrator_with_equilibria.integrate(
            x0=x0, u_func=lambda t: np.array([0.0]), t_span=(1, 0)
        )

        assert result is not None

    def test_zero_duration_integration(self, integrator_with_equilibria):
        """Test integration with zero duration."""
        x0 = np.array([1.0, 0.5])

        result = integrator_with_equilibria.integrate(
            x0=x0, u_func=lambda t: np.array([0.0]), t_span=(0, 0)
        )

        assert result["success"]
        assert len(result["t"]) >= 1
        np.testing.assert_array_equal(result["x"][0], x0)

    def test_very_short_time_span(self, integrator_with_equilibria):
        """Test integration over very short time."""
        x0 = np.array([1.0, 0.5])

        result = integrator_with_equilibria.integrate(
            x0=x0, u_func=lambda t: np.array([0.0]), t_span=(0, 0.001)
        )

        assert result["success"]
        assert len(result["t"]) >= 1


# ============================================================================
# Test Class: Integration Termination
# ============================================================================


class TestIntegrationTermination:
    """Test integration termination conditions."""

    def test_max_steps_termination(self, mock_system_with_equilibria):
        """Test that integration stops at max_steps."""
        integrator = ConcreteTestIntegrator(mock_system_with_equilibria, dt=0.01, max_steps=10)

        x0 = np.array([1.0, 0.5])

        result = integrator.integrate(x0=x0, u_func=lambda t: np.array([0.0]), t_span=(0, 100))

        assert result["nsteps"] <= 11

    def test_normal_termination(self, integrator_with_equilibria):
        """Test normal termination at end time."""
        x0 = np.array([1.0, 0.5])

        result = integrator_with_equilibria.integrate(
            x0=x0, u_func=lambda t: np.array([0.0]), t_span=(0, 1)
        )

        assert result["success"]
        assert result["t"][-1] >= 0.99


# ============================================================================
# Test Class: Dense Output
# ============================================================================


class TestDenseOutput:
    """Test dense output functionality."""

    def test_dense_output_flag(self, integrator_with_equilibria):
        """Test that dense_output flag is passed correctly."""
        x0 = np.array([1.0, 0.5])

        result = integrator_with_equilibria.integrate(
            x0=x0, u_func=lambda t: np.array([0.0]), t_span=(0, 1), dense_output=True
        )

        assert result["success"]

    def test_t_eval_points(self, integrator_with_equilibria):
        """Test evaluation at specific time points."""
        x0 = np.array([1.0, 0.5])
        t_eval = np.array([0, 0.5, 1.0])

        result = integrator_with_equilibria.integrate(
            x0=x0, u_func=lambda t: np.array([0.0]), t_span=(0, 1), t_eval=t_eval
        )

        assert result["success"]


# ============================================================================
# Test Class: Step Size Handling
# ============================================================================


class TestStepSizeHandling:
    """Test step size parameter handling."""

    def test_small_step_size(self, mock_system_with_equilibria):
        """Test integration with very small step size."""
        integrator = ConcreteTestIntegrator(mock_system_with_equilibria, dt=0.0001)

        x0 = np.array([1.0, 0.5])

        result = integrator.integrate(x0=x0, u_func=lambda t: np.array([0.0]), t_span=(0, 0.1))

        assert result["success"]
        assert result["nsteps"] > 100

    def test_large_step_size(self, mock_system_with_equilibria):
        """Test integration with large step size."""
        integrator = ConcreteTestIntegrator(mock_system_with_equilibria, dt=0.5)

        x0 = np.array([1.0, 0.5])

        result = integrator.integrate(x0=x0, u_func=lambda t: np.array([0.0]), t_span=(0, 1))

        assert result["success"]
        assert result["nsteps"] < 10

    def test_step_size_consistency(self, mock_system_with_equilibria):
        """Test that step size is consistent."""
        dt = 0.1
        integrator = ConcreteTestIntegrator(mock_system_with_equilibria, dt=dt)

        x0 = np.array([1.0, 0.5])

        result = integrator.integrate(x0=x0, u_func=lambda t: np.array([0.0]), t_span=(0, 1))

        assert result["success"]

        if len(result["t"]) > 1:
            dts = np.diff(result["t"])
            np.testing.assert_allclose(dts, dt, rtol=0.1)


# ============================================================================
# Test Class: Error Handling
# ============================================================================


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_nan_in_initial_condition(self, integrator_with_equilibria):
        """Test handling of NaN in initial condition."""
        x0_nan = np.array([np.nan, 0.0])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)

            try:
                result = integrator_with_equilibria.integrate(
                    x0=x0_nan, u_func=lambda t: np.array([0.0]), t_span=(0, 0.1)
                )
                assert np.any(np.isnan(result["x"]))
            except (ValueError, RuntimeError):
                pass

    def test_inf_in_initial_condition(self, integrator_with_equilibria):
        """Test handling of Inf in initial condition."""
        x0_inf = np.array([np.inf, 0.0])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)

            try:
                result = integrator_with_equilibria.integrate(
                    x0=x0_inf, u_func=lambda t: np.array([0.0]), t_span=(0, 0.1)
                )
                assert np.any(np.isinf(result["x"]))
            except (ValueError, RuntimeError):
                pass

    def test_none_initial_condition(self, integrator_with_equilibria):
        """Test handling of None initial condition."""
        with pytest.raises((ValueError, TypeError, AttributeError)):
            integrator_with_equilibria.integrate(
                x0=None, u_func=lambda t: np.array([0.0]), t_span=(0, 0.1)
            )


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])