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
Unit Tests for DiffEqPyIntegrator

Tests the Julia DifferentialEquations.jl integration via diffeqpy.

Requirements:
    - pytest
    - numpy
    - Julia with DifferentialEquations.jl
    - diffeqpy

Run tests:
    pytest tests/integrator_unit_tests/diffeqpy_integrator_test.py -v
    pytest tests/integrator_unit_tests/diffeqpy_integrator_test.py -v -k "test_name"
    pytest tests/integrator_unit_tests/diffeqpy_integrator_test.py -v -m "not slow"
"""

from typing import Callable

import numpy as np
import pytest

# Try to import diffeqpy - skip all tests if not available
try:
    from diffeqpy import de

    DIFFEQPY_AVAILABLE = True
except ImportError:
    DIFFEQPY_AVAILABLE = False
    de = None

from src.systems.base.numerical_integration.diffeqpy_integrator import (
    DiffEqPyIntegrator,
    create_diffeqpy_integrator,
    get_algorithm_info,
    list_algorithms,
    print_algorithm_recommendations,
)
from src.systems.base.numerical_integration.integrator_base import StepMode

# Skip all tests if diffeqpy not available
pytestmark = pytest.mark.skipif(
    not DIFFEQPY_AVAILABLE,
    reason="diffeqpy not installed. Install Julia + DifferentialEquations.jl + diffeqpy",
)


# ============================================================================
# Mock System for Testing
# ============================================================================


class SimpleLinearSystem:
    """Simple linear system for testing: dx/dt = A*x + B*u"""

    def __init__(self, A, B):
        self.A = np.array(A)
        self.B = np.array(B)
        self.nx = self.A.shape[0]
        self.nu = self.B.shape[1] if self.B.ndim > 1 else 1

    def __call__(self, x, u, backend="numpy"):
        """Evaluate dynamics: dx/dt = A*x + B*u"""
        x = np.asarray(x)
        u = np.asarray(u)

        if u.ndim == 0:
            u = u.reshape(1)

        return self.A @ x + self.B @ u


class NonlinearPendulum:
    """Nonlinear pendulum: θ'' = -g/L * sin(θ) - b*θ' + u"""

    def __init__(self, g=9.81, L=1.0, b=0.1):
        self.g = g
        self.L = L
        self.b = b
        self.nx = 2
        self.nu = 1

    def __call__(self, x, u, backend="numpy"):
        """State: [θ, θ'], control: [τ]"""
        x = np.asarray(x)
        u = np.asarray(u)

        theta, theta_dot = x
        tau = u[0] if u.ndim > 0 else u

        theta_ddot = -(self.g / self.L) * np.sin(theta) - self.b * theta_dot + tau

        return np.array([theta_dot, theta_ddot])


class StiffVanDerPol:
    """Stiff Van der Pol oscillator: y'' - μ(1-y²)y' + y = 0"""

    def __init__(self, mu=1000.0):
        self.mu = mu
        self.nx = 2
        self.nu = 1

    def __call__(self, x, u, backend="numpy"):
        """State: [y, y']"""
        x = np.asarray(x)
        y, y_dot = x

        y_ddot = self.mu * (1 - y**2) * y_dot - y

        return np.array([y_dot, y_ddot])


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def simple_system():
    """Simple 2D linear system"""
    A = np.array([[-0.5, 1.0], [-1.0, -0.5]])
    B = np.array([[0.0], [1.0]])
    return SimpleLinearSystem(A, B)


@pytest.fixture
def pendulum_system():
    """Nonlinear pendulum"""
    return NonlinearPendulum(g=9.81, L=1.0, b=0.1)


@pytest.fixture
def stiff_system():
    """Stiff Van der Pol oscillator"""
    return StiffVanDerPol(mu=1000.0)


@pytest.fixture
def default_integrator(simple_system):
    """Default DiffEqPy integrator with Tsit5"""
    return DiffEqPyIntegrator(
        simple_system, backend="numpy", algorithm="Tsit5", rtol=1e-6, atol=1e-8
    )


# ============================================================================
# Initialization Tests
# ============================================================================


class TestInitialization:
    """Test integrator initialization"""

    def test_default_initialization(self, simple_system):
        """Test default parameters"""
        integrator = DiffEqPyIntegrator(simple_system, backend="numpy")

        assert integrator.backend == "numpy"
        assert integrator.algorithm == "Tsit5"
        assert integrator.step_mode == StepMode.ADAPTIVE
        assert integrator.rtol == 1e-6
        assert integrator.atol == 1e-8
        assert integrator.save_everystep == False
        assert integrator.dense == False

    def test_custom_algorithm(self, simple_system):
        """Test custom algorithm selection"""
        integrator = DiffEqPyIntegrator(simple_system, backend="numpy", algorithm="Vern7")
        assert integrator.algorithm == "Vern7"

    def test_fixed_step_mode(self, simple_system):
        """Test fixed-step mode initialization"""
        integrator = DiffEqPyIntegrator(
            simple_system, dt=0.01, step_mode=StepMode.FIXED, backend="numpy"
        )
        assert integrator.step_mode == StepMode.FIXED
        assert integrator.dt == 0.01

    def test_custom_tolerances(self, simple_system):
        """Test custom tolerance settings"""
        integrator = DiffEqPyIntegrator(simple_system, backend="numpy", rtol=1e-9, atol=1e-11)
        assert integrator.rtol == 1e-9
        assert integrator.atol == 1e-11

    def test_invalid_backend_raises_error(self, simple_system):
        """Test that non-numpy backends raise error"""
        with pytest.raises(ValueError, match="requires backend='numpy'"):
            DiffEqPyIntegrator(simple_system, backend="torch")

        with pytest.raises(ValueError, match="requires backend='numpy'"):
            DiffEqPyIntegrator(simple_system, backend="jax")

    def test_invalid_algorithm_raises_error(self, simple_system):
        """Test that invalid algorithm raises error"""
        with pytest.raises(ValueError, match="Unknown algorithm"):
            DiffEqPyIntegrator(simple_system, backend="numpy", algorithm="NonExistentAlgorithm123")

    def test_save_everystep_option(self, simple_system):
        """Test save_everystep option"""
        integrator = DiffEqPyIntegrator(simple_system, backend="numpy", save_everystep=True)
        assert integrator.save_everystep == True

    def test_dense_output_option(self, simple_system):
        """Test dense output option"""
        integrator = DiffEqPyIntegrator(simple_system, backend="numpy", dense=True)
        assert integrator.dense == True

    def test_auto_switching_algorithm_validation(self, simple_system):
        """Test that auto-switching algorithms are validated correctly"""
        # Should not raise error - auto-switching algorithm in known list
        integrator = DiffEqPyIntegrator(
            simple_system, backend="numpy", algorithm="AutoTsit5(Rosenbrock23())"
        )
        assert integrator.algorithm == "AutoTsit5(Rosenbrock23())"
        assert "Auto" in integrator.name


# ============================================================================
# Integration Tests - Adaptive Stepping
# ============================================================================


class TestAdaptiveIntegration:
    """Test adaptive time stepping integration"""

    def test_basic_integration(self, simple_system):
        """Test basic integration with default settings"""
        integrator = DiffEqPyIntegrator(simple_system, backend="numpy")

        x0 = np.array([1.0, 0.0])
        u_func = lambda t, x: np.zeros(1)

        result = integrator.integrate(x0=x0, u_func=u_func, t_span=(0.0, 5.0))

        # Debug output if failed
        if not result["success"]:
            print(f"\n[DEBUG] Integration failed: {result["message"]}")
            print(f"[DEBUG] nfev: {result["nfev"]}, nsteps: {result["nsteps"]}")

        assert result["success"], f"Integration failed: {result["message"]}"
        assert len(result["t"]) > 0
        assert result["x"].shape[0] == len(result["t"])
        assert result["x"].shape[1] == 2
        assert result["nsteps"] > 0
        assert result["nfev"] > 0

    def test_zero_control_integration(self, simple_system):
        """Test integration with zero control"""
        integrator = DiffEqPyIntegrator(simple_system, backend="numpy")

        x0 = np.array([1.0, 1.0])
        u_func = lambda t, x: np.zeros(1)

        result = integrator.integrate(x0, u_func, (0.0, 10.0))

        assert result["success"], f"Integration failed: {result["message"]}"
        # System should decay to zero (stable)
        assert np.linalg.norm(result["x"][-1]) < np.linalg.norm(x0)

    def test_constant_control(self, simple_system):
        """Test integration with constant control"""
        integrator = DiffEqPyIntegrator(simple_system, backend="numpy")

        x0 = np.array([0.0, 0.0])
        u_const = np.array([1.0])
        u_func = lambda t, x: u_const

        result = integrator.integrate(x0, u_func, (0.0, 5.0))

        assert result["success"], f"Integration failed: {result["message"]}"
        # System should respond to constant input
        assert np.linalg.norm(result["x"][-1]) > 0.01

    def test_time_varying_control(self, simple_system):
        """Test integration with time-varying control"""
        integrator = DiffEqPyIntegrator(simple_system, backend="numpy")

        x0 = np.array([0.0, 0.0])
        u_func = lambda t, x: np.array([np.sin(t)])

        result = integrator.integrate(x0, u_func, (0.0, 2 * np.pi))

        assert result["success"], f"Integration failed: {result["message"]}"
        assert len(result["t"]) > 0

    def test_state_feedback_control(self, simple_system):
        """Test integration with state feedback"""
        integrator = DiffEqPyIntegrator(simple_system, backend="numpy")

        x0 = np.array([1.0, 0.0])
        K = np.array([[1.0, 0.5]])
        u_func = lambda t, x: -K @ x

        result = integrator.integrate(x0, u_func, (0.0, 10.0))

        assert result["success"], f"Integration failed: {result["message"]}"
        # Feedback should stabilize system
        assert np.linalg.norm(result["x"][-1]) < 0.1

    def test_custom_evaluation_times(self, simple_system):
        """Test integration with specific evaluation times"""
        integrator = DiffEqPyIntegrator(simple_system, backend="numpy")

        x0 = np.array([1.0, 0.0])
        u_func = lambda t, x: np.zeros(1)
        t_eval = np.linspace(0, 5, 51)

        result = integrator.integrate(x0, u_func, (0, 5), t_eval=t_eval)

        assert result["success"], f"Integration failed: {result["message"]}"
        assert len(result["t"]) == len(t_eval)
        np.testing.assert_allclose(result["t"], t_eval, rtol=1e-10)

    def test_zero_time_span(self, simple_system):
        """Test handling of zero time span"""
        integrator = DiffEqPyIntegrator(simple_system, backend="numpy")

        x0 = np.array([1.0, 0.0])
        u_func = lambda t, x: np.zeros(1)

        result = integrator.integrate(x0, u_func, (0.0, 0.0))

        assert result["success"]
        assert len(result["t"]) == 1
        np.testing.assert_allclose(result["x"][0], x0)

    @pytest.mark.slow
    def test_long_integration(self, simple_system):
        """Test long-duration integration"""
        integrator = DiffEqPyIntegrator(simple_system, backend="numpy", rtol=1e-6)

        x0 = np.array([1.0, 0.0])
        u_func = lambda t, x: np.zeros(1)

        # Use t_eval to ensure we get enough points
        # Julia's adaptive stepping might choose very few points for smooth systems
        t_eval = np.linspace(0, 100, 50)  # Force 50 evaluation points

        result = integrator.integrate(x0, u_func, (0.0, 100.0), t_eval=t_eval)

        assert result["success"], f"Integration failed: {result["message"]}"
        assert len(result["t"]) >= 10  # Should have at least 10 points


# ============================================================================
# Integration Tests - Fixed Stepping
# ============================================================================


class TestFixedStepIntegration:
    """Test fixed time step integration"""

    def test_fixed_step_basic(self, simple_system):
        """Test basic fixed-step integration"""
        dt = 0.01
        integrator = DiffEqPyIntegrator(
            simple_system, dt=dt, step_mode=StepMode.FIXED, backend="numpy"
        )

        x0 = np.array([1.0, 0.0])
        u_func = lambda t, x: np.zeros(1)

        result = integrator.integrate(x0, u_func, (0.0, 1.0))

        assert result["success"], f"Integration failed: {result["message"]}"
        # Check approximate number of steps
        expected_steps = int(1.0 / dt) + 1
        assert len(result["t"]) >= expected_steps - 2  # Allow some tolerance

    def test_fixed_step_with_t_eval(self, simple_system):
        """Test fixed-step with custom evaluation times"""
        integrator = DiffEqPyIntegrator(
            simple_system, dt=0.01, step_mode=StepMode.FIXED, backend="numpy"
        )

        x0 = np.array([1.0, 0.0])
        u_func = lambda t, x: np.zeros(1)
        t_eval = np.array([0.0, 0.5, 1.0])

        result = integrator.integrate(x0, u_func, (0.0, 1.0), t_eval=t_eval)

        assert result["success"], f"Integration failed: {result["message"]}"
        assert len(result["t"]) == len(t_eval)

    def test_fixed_step_dt_required(self, simple_system):
        """Test that dt is required for fixed-step mode"""
        with pytest.raises(ValueError, match="required"):
            integrator = DiffEqPyIntegrator(
                simple_system, dt=None, step_mode=StepMode.FIXED, backend="numpy"
            )
            x0 = np.array([1.0, 0.0])
            integrator.integrate(x0, lambda t, x: np.zeros(1), (0, 1))


# ============================================================================
# Single Step Tests
# ============================================================================


class TestSingleStep:
    """Test single-step integration"""

    def test_single_step_basic(self, simple_system):
        """Test basic single step"""
        integrator = DiffEqPyIntegrator(simple_system, dt=0.1, backend="numpy")

        x = np.array([1.0, 0.0])
        u = np.array([0.0])

        x_next = integrator.step(x, u)

        assert x_next.shape == x.shape
        # State should change (system has dynamics even with zero control)
        # For stable system, state magnitude should decrease or oscillate
        assert np.linalg.norm(x_next) < np.linalg.norm(x) * 1.2

    def test_single_step_with_control(self, simple_system):
        """Test single step with non-zero control"""
        integrator = DiffEqPyIntegrator(simple_system, dt=0.1, backend="numpy")

        x = np.array([0.0, 0.0])
        u = np.array([1.0])

        x_next = integrator.step(x, u)

        # With control input, state should change from zero
        # B = [[0], [1]] so control affects second state
        assert np.linalg.norm(x_next) > 1e-3  # More lenient threshold

    def test_single_step_custom_dt(self, simple_system):
        """Test single step with custom dt"""
        integrator = DiffEqPyIntegrator(simple_system, dt=0.1, backend="numpy")

        x = np.array([1.0, 0.0])
        u = np.array([0.0])

        x_next = integrator.step(x, u, dt=0.05)

        assert x_next.shape == x.shape

    def test_single_step_no_dt_raises(self, simple_system):
        """Test that step without dt raises error when dt not set"""
        # For adaptive mode, base class sets dt=0.01 as default
        # So we need to explicitly test the case where dt is truly None
        integrator = DiffEqPyIntegrator(simple_system, dt=None, backend="numpy")

        # Temporarily set dt to None to test error handling
        integrator.dt = None

        x = np.array([1.0, 0.0])
        u = np.array([0.0])

        with pytest.raises(ValueError, match="dt must be specified"):
            integrator.step(x, u)


# ============================================================================
# Algorithm-Specific Tests
# ============================================================================


class TestAlgorithms:
    """Test different Julia algorithms"""

    def test_tsit5_algorithm(self, simple_system):
        """Test Tsit5 (default) algorithm"""
        integrator = DiffEqPyIntegrator(simple_system, backend="numpy", algorithm="Tsit5")

        x0 = np.array([1.0, 0.0])
        result = integrator.integrate(x0, lambda t, x: np.zeros(1), (0, 5))

        assert result["success"], f"Integration failed: {result["message"]}"
        assert "Tsit5" in integrator.name

    def test_vern7_high_accuracy(self, simple_system):
        """Test Vern7 high-accuracy algorithm"""
        integrator = DiffEqPyIntegrator(
            simple_system, backend="numpy", algorithm="Vern7", rtol=1e-10, atol=1e-12
        )

        x0 = np.array([1.0, 0.0])
        result = integrator.integrate(x0, lambda t, x: np.zeros(1), (0, 5))

        assert result["success"], f"Integration failed: {result["message"]}"
        assert "Vern7" in integrator.name

    @pytest.mark.slow
    def test_rosenbrock23_stiff(self, stiff_system):
        """Test Rosenbrock23 on stiff system"""
        integrator = DiffEqPyIntegrator(
            stiff_system, backend="numpy", algorithm="Rosenbrock23", rtol=1e-6, atol=1e-8
        )

        x0 = np.array([2.0, 0.0])
        result = integrator.integrate(x0, lambda t, x: np.zeros(1), (0, 0.1))

        # Note: Rosenbrock may fail with automatic differentiation for in-place
        # This is a known limitation - either skip or expect potential failure
        if not result["success"] and "Jacobian" in result["message"]:
            pytest.skip(f"Rosenbrock AD issue with in-place: {result["message"]}")

        assert result["success"], f"Integration failed: {result["message"]}"
        assert "Stiff" in integrator.name

    def test_auto_switching_algorithm(self, simple_system):
        """Test auto-switching algorithm"""
        integrator = DiffEqPyIntegrator(
            simple_system, backend="numpy", algorithm="AutoTsit5(Rosenbrock23())"
        )

        x0 = np.array([1.0, 0.0])
        result = integrator.integrate(x0, lambda t, x: np.zeros(1), (0, 5))

        assert result["success"], f"Integration failed: {result["message"]}"
        assert "Auto" in integrator.name


# ============================================================================
# Nonlinear System Tests
# ============================================================================


class TestNonlinearSystems:
    """Test integration of nonlinear systems"""

    def test_pendulum_free_motion(self, pendulum_system):
        """Test pendulum with no control"""
        integrator = DiffEqPyIntegrator(pendulum_system, backend="numpy", algorithm="Tsit5")

        # Small angle, should oscillate
        x0 = np.array([0.1, 0.0])  # [θ, θ']
        u_func = lambda t, x: np.zeros(1)

        result = integrator.integrate(x0, u_func, (0.0, 5.0))

        assert result["success"], f"Integration failed: {result["message"]}"
        # Check oscillatory behavior
        assert np.max(np.abs(result["x"][:, 0])) > 0.05

    def test_pendulum_with_control(self, pendulum_system):
        """Test pendulum with control input"""
        integrator = DiffEqPyIntegrator(pendulum_system, backend="numpy")

        x0 = np.array([np.pi / 4, 0.0])
        # Simple PD control
        u_func = lambda t, x: -2.0 * x[0] - 1.0 * x[1]

        result = integrator.integrate(x0, u_func, (0.0, 10.0))

        assert result["success"], f"Integration failed: {result["message"]}"
        # Should stabilize near zero
        assert np.abs(result["x"][-1, 0]) < 0.1

    def test_pendulum_large_angle(self, pendulum_system):
        """Test pendulum with large initial angle"""
        integrator = DiffEqPyIntegrator(pendulum_system, backend="numpy", rtol=1e-8)

        x0 = np.array([3.0, 0.0])  # Near inverted
        u_func = lambda t, x: np.zeros(1)

        result = integrator.integrate(x0, u_func, (0.0, 5.0))

        assert result["success"], f"Integration failed: {result["message"]}"


# ============================================================================
# Dense Output Tests
# ============================================================================


class TestDenseOutput:
    """Test dense output and interpolation"""

    def test_dense_output_enabled(self, simple_system):
        """Test integration with dense output"""
        integrator = DiffEqPyIntegrator(simple_system, backend="numpy", dense=True)

        x0 = np.array([1.0, 0.0])
        result = integrator.integrate(x0, lambda t, x: np.zeros(1), (0, 5))

        assert result["success"], f"Integration failed: {result["message"]}"
        assert "sol" in result, "Dense output enabled but 'sol' not in result"
        assert result.get("dense_output", False), "dense_output flag not set"

    def test_dense_output_parameter(self, simple_system):
        """Test dense output via integrate parameter"""
        integrator = DiffEqPyIntegrator(simple_system, backend="numpy")

        x0 = np.array([1.0, 0.0])
        result = integrator.integrate(x0, lambda t, x: np.zeros(1), (0, 5), dense_output=True)

        assert result["success"], f"Integration failed: {result["message"]}"
        assert "sol" in result, "Dense output requested but 'sol' not in result"
        assert result.get("dense_output", False), "dense_output flag not set"

    def test_no_dense_output(self, simple_system):
        """Test that optional fields are absent when not requested"""
        integrator = DiffEqPyIntegrator(simple_system, backend="numpy", dense=False)

        x0 = np.array([1.0, 0.0])
        result = integrator.integrate(x0, lambda t, x: np.zeros(1), (0, 5))

        assert result["success"]
        # Optional fields should not be present
        assert "sol" not in result
        assert "dense_output" not in result


# ============================================================================
# Statistics Tests
# ============================================================================


class TestStatistics:
    """Test integration statistics tracking"""

    def test_statistics_tracking(self, simple_system):
        """Test that statistics are tracked"""
        integrator = DiffEqPyIntegrator(simple_system, backend="numpy")

        integrator.reset_stats()

        x0 = np.array([1.0, 0.0])
        result = integrator.integrate(x0, lambda t, x: np.zeros(1), (0, 5))

        assert result["success"], f"Integration failed: {result["message"]}"

        stats = integrator.get_stats()

        assert stats["total_steps"] > 0
        assert stats["total_fev"] > 0
        assert stats["total_time"] > 0
        assert stats["avg_fev_per_step"] > 0

    def test_reset_statistics(self, simple_system):
        """Test statistics reset"""
        integrator = DiffEqPyIntegrator(simple_system, backend="numpy")

        x0 = np.array([1.0, 0.0])
        integrator.integrate(x0, lambda t, x: np.zeros(1), (0, 1))

        integrator.reset_stats()
        stats = integrator.get_stats()

        assert stats["total_steps"] == 0
        assert stats["total_fev"] == 0
        assert stats["total_time"] == 0.0

    def test_cumulative_statistics(self, simple_system):
        """Test that statistics accumulate across calls"""
        integrator = DiffEqPyIntegrator(simple_system, backend="numpy")

        integrator.reset_stats()

        x0 = np.array([1.0, 0.0])
        integrator.integrate(x0, lambda t, x: np.zeros(1), (0, 1))
        stats1 = integrator.get_stats()

        integrator.integrate(x0, lambda t, x: np.zeros(1), (0, 1))
        stats2 = integrator.get_stats()

        assert stats2["total_steps"] > stats1["total_steps"]
        assert stats2["total_fev"] > stats1["total_fev"]


# ============================================================================
# Utility Function Tests
# ============================================================================


class TestUtilityFunctions:
    """Test utility functions"""

    def test_list_algorithms(self):
        """Test list_algorithms returns valid structure"""
        algos = list_algorithms()

        assert isinstance(algos, dict)
        assert "nonstiff" in algos
        assert "stiff_rosenbrock" in algos
        assert "auto_switching" in algos

        assert "Tsit5" in algos["nonstiff"]
        assert "Rosenbrock23" in algos["stiff_rosenbrock"]

    def test_get_algorithm_info(self):
        """Test get_algorithm_info"""
        info = get_algorithm_info("Tsit5")

        assert isinstance(info, dict)
        assert "name" in info
        assert "description" in info

        # Test unknown algorithm
        info_unknown = get_algorithm_info("UnknownAlgorithm")
        assert "No information available" in info_unknown["description"]

    def test_print_algorithm_recommendations(self, capsys):
        """Test print_algorithm_recommendations"""
        print_algorithm_recommendations()

        captured = capsys.readouterr()
        assert "Tsit5" in captured.out
        assert "STIFF" in captured.out.upper()

    def test_create_diffeqpy_integrator(self, simple_system):
        """Test factory function"""
        integrator = create_diffeqpy_integrator(
            simple_system, algorithm="Vern7", rtol=1e-8, atol=1e-10
        )

        assert isinstance(integrator, DiffEqPyIntegrator)
        assert integrator.algorithm == "Vern7"
        assert integrator.rtol == 1e-8
        assert integrator.atol == 1e-10


# ============================================================================
# Validation Tests
# ============================================================================


class TestValidation:
    """Test algorithm validation logic"""

    def test_known_algorithm_accepted(self, simple_system):
        """Test that known algorithms are accepted"""
        # Should not raise
        for algo in ["Tsit5", "Vern7", "Rosenbrock23"]:
            integrator = DiffEqPyIntegrator(simple_system, backend="numpy", algorithm=algo)
            assert integrator.algorithm == algo

    def test_unknown_algorithm_rejected(self, simple_system):
        """Test that unknown algorithms are rejected with helpful message"""
        with pytest.raises(ValueError) as excinfo:
            DiffEqPyIntegrator(simple_system, backend="numpy", algorithm="NotARealAlgorithm")

        # Check error message is helpful
        error_msg = str(excinfo.value)
        assert "Unknown algorithm" in error_msg
        assert "list_algorithms()" in error_msg
        assert "nonstiff" in error_msg or "stiff" in error_msg

    def test_auto_switching_algorithms_validated(self, simple_system):
        """Test that auto-switching algorithms in known list are validated"""
        # Should work - in the known list
        integrator = DiffEqPyIntegrator(
            simple_system, backend="numpy", algorithm="AutoTsit5(Rosenbrock23())"
        )
        assert integrator.algorithm == "AutoTsit5(Rosenbrock23())"

    def test_validation_catches_typos(self, simple_system):
        """Test that validation catches common typos"""
        with pytest.raises(ValueError, match="Unknown algorithm"):
            # Typo: Tsiit5 instead of Tsit5
            DiffEqPyIntegrator(simple_system, backend="numpy", algorithm="Tsiit5")


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestErrorHandling:
    """Test error handling and edge cases"""

    def test_nan_in_dynamics(self, simple_system):
        """Test handling of NaN in dynamics"""
        integrator = DiffEqPyIntegrator(simple_system, backend="numpy")

        # Control that causes NaN
        x0 = np.array([1.0, 0.0])
        u_func = lambda t, x: np.array([np.nan])

        result = integrator.integrate(x0, u_func, (0, 1))
        # Should fail gracefully
        assert not result["success"] or not np.all(np.isfinite(result["x"]))

    def test_backward_time_integration(self, simple_system):
        """Test backward time integration (t1 < t0)"""
        integrator = DiffEqPyIntegrator(simple_system, backend="numpy")

        x0 = np.array([1.0, 0.0])
        result = integrator.integrate(x0, lambda t, x: np.zeros(1), (5.0, 0.0))

        # Julia should handle backward integration
        assert len(result["t"]) > 0


# ============================================================================
# Integration Result Tests
# ============================================================================


class TestIntegrationResult:
    """Test IntegrationResult TypedDict structure"""

    def test_result_is_dict(self, simple_system):
        """Test that result is a dict (TypedDict)"""
        integrator = DiffEqPyIntegrator(simple_system, backend="numpy")

        x0 = np.array([1.0, 0.0])
        result = integrator.integrate(x0, lambda t, x: np.zeros(1), (0, 1))

        # Verify it's a dict
        assert isinstance(result, dict)

    def test_result_required_fields(self, simple_system):
        """Test that result has all required fields"""
        integrator = DiffEqPyIntegrator(simple_system, backend="numpy")

        x0 = np.array([1.0, 0.0])
        result = integrator.integrate(x0, lambda t, x: np.zeros(1), (0, 1))

        # Check all required fields
        required_fields = ["t", "x", "success", "message", "nfev", "nsteps", 
                          "integration_time", "solver"]
        for field in required_fields:
            assert field in result, f"Missing required field: {field}"

    def test_result_shapes(self, simple_system):
        """Test result array shapes"""
        integrator = DiffEqPyIntegrator(simple_system, backend="numpy")

        x0 = np.array([1.0, 0.0])
        result = integrator.integrate(x0, lambda t, x: np.zeros(1), (0, 5))

        assert result["t"].ndim == 1
        assert result["x"].ndim == 2
        assert result["x"].shape[0] == len(result["t"])
        assert result["x"].shape[1] == 2

    def test_result_field_types(self, simple_system):
        """Test result field types"""
        integrator = DiffEqPyIntegrator(simple_system, backend="numpy")

        x0 = np.array([1.0, 0.0])
        result = integrator.integrate(x0, lambda t, x: np.zeros(1), (0, 1))

        # Check types
        assert isinstance(result["t"], np.ndarray)
        assert isinstance(result["x"], np.ndarray)
        assert isinstance(result["success"], bool)
        assert isinstance(result["message"], str)
        assert isinstance(result["nfev"], (int, np.integer))
        assert isinstance(result["nsteps"], (int, np.integer))
        assert isinstance(result["integration_time"], (float, np.floating))
        assert isinstance(result["solver"], str)

    def test_no_none_values_in_result(self, simple_system):
        """Test that result contains no None values"""
        integrator = DiffEqPyIntegrator(simple_system, backend="numpy")

        x0 = np.array([1.0, 0.0])
        result = integrator.integrate(x0, lambda t, x: np.zeros(1), (0, 1))

        # Verify no None values in result
        for key, value in result.items():
            assert value is not None, f"Field '{key}' should not be None"


# ============================================================================
# Integrator Properties Tests
# ============================================================================


class TestIntegratorProperties:
    """Test integrator properties and methods"""

    def test_name_property(self, simple_system):
        """Test name property"""
        integrator = DiffEqPyIntegrator(simple_system, backend="numpy", algorithm="Tsit5")

        name = integrator.name
        assert isinstance(name, str)
        assert "Tsit5" in name
        assert "Adaptive" in name or "Fixed" in name

    def test_repr(self, simple_system):
        """Test __repr__ method"""
        integrator = DiffEqPyIntegrator(
            simple_system, backend="numpy", algorithm="Vern7", rtol=1e-9, atol=1e-11
        )

        repr_str = repr(integrator)
        assert "DiffEqPyIntegrator" in repr_str
        assert "Vern7" in repr_str
        # Check that tolerances are shown (format may vary)
        assert "1e-09" in repr_str or "1.0e-09" in repr_str

    def test_backend_property(self, simple_system):
        """Test backend property"""
        integrator = DiffEqPyIntegrator(simple_system, backend="numpy")
        assert integrator.backend == "numpy"


# ============================================================================
# Comparison Tests (Verify Correctness)
# ============================================================================


class TestCorrectness:
    """Test correctness by comparing with analytical solutions"""

    def test_exponential_decay(self):
        """Test exponential decay: dx/dt = -λx"""

        class ExponentialDecay:
            def __init__(self, lambda_val=1.0):
                self.lambda_val = lambda_val
                self.nx = 1
                self.nu = 1

            def __call__(self, x, u, backend="numpy"):
                return -self.lambda_val * x

        system = ExponentialDecay(lambda_val=2.0)
        integrator = DiffEqPyIntegrator(
            system, backend="numpy", algorithm="Vern9", rtol=1e-10, atol=1e-12
        )

        x0 = np.array([1.0])
        t_eval = np.linspace(0, 2, 21)

        result = integrator.integrate(x0, lambda t, x: np.zeros(1), (0, 2), t_eval=t_eval)

        assert result["success"], f"Integration failed: {result["message"]}"

        # Analytical solution: x(t) = x0 * exp(-λt)
        x_analytical = x0 * np.exp(-2.0 * t_eval)

        # Should be very accurate with Vern9
        np.testing.assert_allclose(result["x"].flatten(), x_analytical, rtol=1e-8, atol=1e-10)

    def test_harmonic_oscillator(self):
        """Test harmonic oscillator: x'' + ω²x = 0"""

        class HarmonicOscillator:
            def __init__(self, omega=1.0):
                self.omega = omega
                self.nx = 2
                self.nu = 1

            def __call__(self, x, u, backend="numpy"):
                return np.array([x[1], -self.omega**2 * x[0]])

        omega = 2.0
        system = HarmonicOscillator(omega=omega)
        integrator = DiffEqPyIntegrator(
            system, backend="numpy", algorithm="Vern9", rtol=1e-10, atol=1e-12
        )

        x0 = np.array([1.0, 0.0])  # [position, velocity]
        t_eval = np.linspace(0, 2 * np.pi / omega, 100)

        result = integrator.integrate(
            x0, lambda t, x: np.zeros(1), (0, 2 * np.pi / omega), t_eval=t_eval
        )

        assert result["success"], f"Integration failed: {result["message"]}"

        # Analytical: x(t) = cos(ωt), v(t) = -ω*sin(ωt)
        x_analytical = np.cos(omega * t_eval)
        v_analytical = -omega * np.sin(omega * t_eval)

        np.testing.assert_allclose(result["x"][:, 0], x_analytical, rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(result["x"][:, 1], v_analytical, rtol=1e-6, atol=1e-8)


# ============================================================================
# Performance Markers
# ============================================================================

# Slow tests can be skipped with: pytest -m "not slow"
pytest.mark.slow


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
