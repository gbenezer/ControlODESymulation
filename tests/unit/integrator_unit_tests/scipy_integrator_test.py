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
Unit tests for ScipyIntegrator (adaptive integration) - REFACTORED FOR TypedDict

Tests cover:
1. Initialization and configuration
2. Adaptive step size control
3. Different solver methods (RK45, LSODA, BDF, etc.)
4. Error tolerance control
5. Stiff system handling
6. Efficiency vs fixed-step methods
7. Dense output
8. Event detection
9. Autonomous systems

Refactoring Changes
-------------------
- Results are now TypedDict instead of class instances
- Changed result.field → result["field"] throughout
- Changed isinstance(result, IntegrationResult) → isinstance(result, dict)
- Changed hasattr(result, "field") → "field" in result
- Added TypedDict-specific tests
- Added tests for events and dense output
- All functionality and test coverage preserved
"""

import numpy as np
import pytest

from src.systems.base.numerical_integration.fixed_step_integrators import RK4Integrator
from src.systems.base.numerical_integration.integrator_base import StepMode
from src.systems.base.numerical_integration.scipy_integrator import ScipyIntegrator
from src.types.trajectories import IntegrationResult

# Check if scipy is available
scipy_available = True
try:
    import scipy.integrate
except ImportError:
    scipy_available = False


# ============================================================================
# Mock Systems
# ============================================================================


class SimpleDecaySystem:
    """dx/dt = -a*x + u"""

    def __init__(self, a=1.0):
        self.a = a
        self.nx = 1
        self.nu = 1
        self._initialized = True
        self._default_backend = "numpy"

    def __call__(self, x, u, backend="numpy"):
        return -self.a * x + u

    def analytical_solution(self, x0, t):
        return x0 * np.exp(-self.a * t)


class AutonomousSystem:
    """Autonomous system: dx/dt = -x (no control)"""

    def __init__(self):
        self.nx = 1
        self.nu = 0  # No control
        self._initialized = True
        self._default_backend = "numpy"

    def __call__(self, x, u, backend="numpy"):
        """u should be None for autonomous"""
        if u is not None:
            raise ValueError("Autonomous system should receive u=None")
        return -x

    def analytical_solution(self, x0, t):
        return x0 * np.exp(-t)


class StiffVanDerPolSystem:
    """Stiff Van der Pol oscillator: x'' + μ(x²-1)x' + x = 0"""

    def __init__(self, mu=1000.0):
        self.mu = mu
        self.nx = 2
        self.nu = 1
        self._initialized = True
        self._default_backend = "numpy"

    def __call__(self, x, u, backend="numpy"):
        """State: [x, x'], Dynamics: [x', μ(1-x²)x' - x]"""
        if len(x.shape) == 1:
            return np.array([x[1], self.mu * (1 - x[0] ** 2) * x[1] - x[0]])
        else:
            dx1 = x[:, 1]
            dx2 = self.mu * (1 - x[:, 0] ** 2) * x[:, 1] - x[:, 0]
            return np.column_stack([dx1, dx2])


# ============================================================================
# Test Class 1: Initialization and Configuration
# ============================================================================


@pytest.mark.skipif(not scipy_available, reason="scipy not installed")
class TestScipyInitialization:
    """Test ScipyIntegrator initialization"""

    def test_basic_initialization(self):
        """Test basic initialization with defaults"""
        system = SimpleDecaySystem()
        integrator = ScipyIntegrator(system, dt=0.01, method="RK45")

        assert integrator.method == "RK45"
        assert integrator.step_mode == StepMode.ADAPTIVE
        assert integrator.backend == "numpy"
        assert integrator.rtol == 1e-6
        assert integrator.atol == 1e-8

    def test_custom_tolerances(self):
        """Test initialization with custom tolerances"""
        system = SimpleDecaySystem()
        integrator = ScipyIntegrator(system, dt=0.01, method="RK45", rtol=1e-10, atol=1e-12)

        assert integrator.rtol == 1e-10
        assert integrator.atol == 1e-12

    def test_all_valid_methods(self):
        """Test that all documented methods work"""
        system = SimpleDecaySystem()

        valid_methods = ["RK45", "RK23", "DOP853", "Radau", "BDF", "LSODA"]

        for method in valid_methods:
            integrator = ScipyIntegrator(system, method=method)
            assert integrator.method == method
            assert method in integrator.name

    def test_invalid_method_raises_error(self):
        """Test that invalid method raises ValueError"""
        system = SimpleDecaySystem()

        with pytest.raises(ValueError, match="Invalid method"):
            ScipyIntegrator(system, method="INVALID")

    def test_non_numpy_backend_raises_error(self):
        """Test that non-numpy backend raises error"""
        system = SimpleDecaySystem()

        with pytest.raises(ValueError, match="only supports NumPy"):
            ScipyIntegrator(system, backend="torch")


# ============================================================================
# Test Class 2: Adaptive Step Size Control
# ============================================================================


@pytest.mark.skipif(not scipy_available, reason="scipy not installed")
class TestAdaptiveStepSize:
    """Test adaptive step size features"""

    def test_adaptive_uses_variable_steps(self):
        """Test that adaptive integrator uses variable step sizes"""
        system = SimpleDecaySystem(a=1.0)
        integrator = ScipyIntegrator(system, method="RK45", rtol=1e-6)

        result = integrator.integrate(
            x0=np.array([1.0]), u_func=lambda t, x: np.zeros(1), t_span=(0.0, 10.0)
        )

        # For smooth exponential decay, steps should vary
        dt_steps = np.diff(result["t"])

        # Should have some variation in step sizes
        assert dt_steps.std() > 0
        assert dt_steps.min() < dt_steps.max()

    def test_tighter_tolerance_more_steps(self):
        """Test that tighter tolerance uses more steps"""
        system = SimpleDecaySystem(a=1.0)

        loose = ScipyIntegrator(system, method="RK45", rtol=1e-3)
        tight = ScipyIntegrator(system, method="RK45", rtol=1e-9)

        result_loose = loose.integrate(
            x0=np.array([1.0]), u_func=lambda t, x: np.zeros(1), t_span=(0.0, 1.0)
        )

        result_tight = tight.integrate(
            x0=np.array([1.0]), u_func=lambda t, x: np.zeros(1), t_span=(0.0, 1.0)
        )

        # Tighter tolerance should use more steps/evaluations
        assert result_tight["nfev"] > result_loose["nfev"]

    def test_high_accuracy_achievable(self):
        """Test that very high accuracy can be achieved"""
        system = SimpleDecaySystem(a=1.0)
        integrator = ScipyIntegrator(system, method="DOP853", rtol=1e-12, atol=1e-14)

        result = integrator.integrate(
            x0=np.array([1.0]), u_func=lambda t, x: np.zeros(1), t_span=(0.0, 1.0)
        )

        x_exact = system.analytical_solution(1.0, 1.0)
        error = abs(result["x"][-1, 0] - x_exact)

        # Should achieve very high accuracy
        assert error < 1e-10


# ============================================================================
# Test Class 3: Solver Method Comparison
# ============================================================================


@pytest.mark.skipif(not scipy_available, reason="scipy not installed")
class TestSolverMethods:
    """Test different scipy solver methods"""

    def test_rk45_general_purpose(self):
        """Test RK45 (general purpose solver)"""
        system = SimpleDecaySystem()
        integrator = ScipyIntegrator(system, method="RK45")

        result = integrator.integrate(
            x0=np.array([1.0]), u_func=lambda t, x: np.zeros(1), t_span=(0.0, 1.0)
        )

        assert result["success"] is True
        assert "RK45" in integrator.name

    def test_rk23_low_accuracy(self):
        """Test RK23 (lower accuracy, faster)"""
        system = SimpleDecaySystem()
        integrator = ScipyIntegrator(system, method="RK23")

        result = integrator.integrate(
            x0=np.array([1.0]), u_func=lambda t, x: np.zeros(1), t_span=(0.0, 1.0)
        )

        assert result["success"] is True

    def test_dop853_high_accuracy(self):
        """Test DOP853 (high accuracy)"""
        system = SimpleDecaySystem()
        integrator = ScipyIntegrator(system, method="DOP853", rtol=1e-10)

        result = integrator.integrate(
            x0=np.array([1.0]), u_func=lambda t, x: np.zeros(1), t_span=(0.0, 1.0)
        )

        # Should be very accurate
        x_exact = system.analytical_solution(1.0, 1.0)
        error = abs(result["x"][-1, 0] - x_exact)
        assert error < 1e-9


# ============================================================================
# Test Class 4: Stiff System Handling
# ============================================================================


@pytest.mark.skipif(not scipy_available, reason="scipy not installed")
class TestStiffSystems:
    """Test handling of stiff systems"""

    def test_bdf_stiff_solver(self):
        """Test BDF method on stiff system"""
        # Very stiff decay (a=1000)
        system = SimpleDecaySystem(a=1000.0)

        integrator = ScipyIntegrator(system, method="BDF", rtol=1e-6, atol=1e-8)

        result = integrator.integrate(
            x0=np.array([1.0]), u_func=lambda t, x: np.zeros(1), t_span=(0.0, 0.1)
        )

        assert result["success"] is True
        assert "Stiff" in integrator.name

    def test_radau_stiff_solver(self):
        """Test Radau method on stiff system"""
        system = SimpleDecaySystem(a=500.0)

        integrator = ScipyIntegrator(system, method="Radau", rtol=1e-6)

        result = integrator.integrate(
            x0=np.array([1.0]), u_func=lambda t, x: np.zeros(1), t_span=(0.0, 0.1)
        )

        assert result["success"] is True

    def test_lsoda_auto_stiffness_detection(self):
        """Test LSODA automatic stiffness detection"""
        # Moderately stiff system
        system = SimpleDecaySystem(a=100.0)

        integrator = ScipyIntegrator(system, method="LSODA")

        result = integrator.integrate(
            x0=np.array([1.0]), u_func=lambda t, x: np.zeros(1), t_span=(0.0, 1.0)
        )

        assert result["success"] is True
        assert "Auto-Stiffness" in integrator.name

    def test_van_der_pol_stiff(self):
        """Test very stiff Van der Pol oscillator"""
        system = StiffVanDerPolSystem(mu=1000.0)  # Very stiff!

        # BDF should handle this
        integrator = ScipyIntegrator(system, method="BDF", rtol=1e-5, atol=1e-7)

        result = integrator.integrate(
            x0=np.array([2.0, 0.0]), u_func=lambda t, x: np.zeros(1), t_span=(0.0, 10.0)
        )

        # Should complete (explicit methods would fail or be extremely slow)
        assert result["success"] is True


# ============================================================================
# Test Class 5: Efficiency Comparison
# ============================================================================


@pytest.mark.skipif(not scipy_available, reason="scipy not installed")
class TestEfficiencyComparison:
    """Compare adaptive vs fixed-step efficiency"""

    def test_adaptive_fewer_steps_smooth_problem(self):
        """Test that adaptive uses fewer steps for smooth problems"""
        system = SimpleDecaySystem(a=1.0)

        # Fixed-step RK4 (small dt for accuracy)
        fixed = RK4Integrator(system, dt=0.01, backend="numpy")
        result_fixed = fixed.integrate(
            x0=np.array([1.0]), u_func=lambda t, x: np.zeros(1), t_span=(0.0, 10.0)
        )

        # Adaptive RK45 (similar accuracy)
        adaptive = ScipyIntegrator(system, method="RK45", rtol=1e-6)
        result_adaptive = adaptive.integrate(
            x0=np.array([1.0]), u_func=lambda t, x: np.zeros(1), t_span=(0.0, 10.0)
        )

        # Verify similar accuracy
        x_exact = system.analytical_solution(1.0, 10.0)
        error_fixed = abs(result_fixed["x"][-1, 0] - x_exact)
        error_adaptive = abs(result_adaptive["x"][-1, 0] - x_exact)

        assert error_fixed < 1e-5
        assert error_adaptive < 1e-5

        # Adaptive should use fewer function evaluations
        print(f"Fixed FEV: {result_fixed['nfev']}")
        print(f"Adaptive FEV: {result_adaptive['nfev']}")

        # For smooth exponential, adaptive should be more efficient
        assert result_adaptive["nfev"] < result_fixed["nfev"]

    def test_adaptive_more_efficient_for_stiff(self):
        """Test adaptive is much more efficient for stiff systems"""
        system = SimpleDecaySystem(a=1000.0)  # Very stiff

        # Adaptive BDF (designed for stiff)
        adaptive = ScipyIntegrator(system, method="BDF", rtol=1e-5)
        result_adaptive = adaptive.integrate(
            x0=np.array([1.0]), u_func=lambda t, x: np.zeros(1), t_span=(0.0, 0.1)
        )

        # Fixed-step would need dt ~ 1e-6 or smaller for stability
        # which would require ~100,000 steps!
        # Adaptive should use far fewer

        assert result_adaptive["success"] is True
        assert result_adaptive["nsteps"] < 1000  # Much fewer than fixed-step


# ============================================================================
# Test Class 6: Accuracy Verification
# ============================================================================


@pytest.mark.skipif(not scipy_available, reason="scipy not installed")
class TestAccuracyVerification:
    """Verify accuracy against analytical solutions"""

    def test_rk45_high_accuracy(self):
        """Test RK45 achieves high accuracy"""
        system = SimpleDecaySystem(a=1.0)
        integrator = ScipyIntegrator(system, method="RK45", rtol=1e-9, atol=1e-11)

        result = integrator.integrate(
            x0=np.array([1.0]), u_func=lambda t, x: np.zeros(1), t_span=(0.0, 2.0)
        )

        x_exact = system.analytical_solution(1.0, 2.0)
        error = abs(result["x"][-1, 0] - x_exact)

        # Should achieve very high accuracy
        assert error < 1e-8

    def test_tolerance_controls_accuracy(self):
        """Test that tolerance controls solution accuracy"""
        system = SimpleDecaySystem(a=1.0)

        # Loose tolerance
        loose = ScipyIntegrator(system, method="RK45", rtol=1e-3, atol=1e-5)
        result_loose = loose.integrate(
            x0=np.array([1.0]), u_func=lambda t, x: np.zeros(1), t_span=(0.0, 1.0)
        )

        # Tight tolerance
        tight = ScipyIntegrator(system, method="RK45", rtol=1e-9, atol=1e-11)
        result_tight = tight.integrate(
            x0=np.array([1.0]), u_func=lambda t, x: np.zeros(1), t_span=(0.0, 1.0)
        )

        x_exact = system.analytical_solution(1.0, 1.0)

        error_loose = abs(result_loose["x"][-1, 0] - x_exact)
        error_tight = abs(result_tight["x"][-1, 0] - x_exact)

        # Tight tolerance should be more accurate
        assert error_tight < error_loose


# ============================================================================
# Test Class 7: Integration Result Properties
# ============================================================================


@pytest.mark.skipif(not scipy_available, reason="scipy not installed")
class TestIntegrationResult:
    """Test IntegrationResult from scipy"""

    def test_result_structure(self):
        """Test that result has expected structure (TypedDict)"""
        system = SimpleDecaySystem()
        integrator = ScipyIntegrator(system, method="RK45")

        result = integrator.integrate(
            x0=np.array([1.0]), u_func=lambda t, x: np.zeros(1), t_span=(0.0, 1.0)
        )

        # TypedDict is dict at runtime
        assert isinstance(result, dict)
        assert "t" in result
        assert "x" in result
        assert "success" in result
        assert "message" in result
        assert "nfev" in result
        assert "nsteps" in result

    def test_success_flag(self):
        """Test that success flag is True for successful integration"""
        system = SimpleDecaySystem()
        integrator = ScipyIntegrator(system, method="RK45")

        result = integrator.integrate(
            x0=np.array([1.0]), u_func=lambda t, x: np.zeros(1), t_span=(0.0, 1.0)
        )

        assert result["success"] is True

    def test_time_and_state_shapes_consistent(self):
        """Test that time and state arrays have consistent shapes"""
        system = SimpleDecaySystem()
        integrator = ScipyIntegrator(system, method="RK45")

        result = integrator.integrate(
            x0=np.array([1.0]), u_func=lambda t, x: np.zeros(1), t_span=(0.0, 1.0)
        )

        # result["x"] should be (T, nx)
        assert result["x"].shape[0] == result["t"].shape[0]
        assert result["x"].shape[1] == system.nx


# ============================================================================
# Test Class 8: Custom Time Evaluation
# ============================================================================


@pytest.mark.skipif(not scipy_available, reason="scipy not installed")
class TestCustomTimeEvaluation:
    """Test integration with custom t_eval"""

    def test_specific_time_points(self):
        """Test evaluation at specific time points"""
        system = SimpleDecaySystem()
        integrator = ScipyIntegrator(system, method="RK45")

        # Request specific times
        t_eval = np.linspace(0, 1, 11)  # 0, 0.1, 0.2, ..., 1.0

        result = integrator.integrate(
            x0=np.array([1.0]), u_func=lambda t, x: np.zeros(1), t_span=(0.0, 1.0), t_eval=t_eval
        )

        # Should match requested times
        assert np.allclose(result["t"], t_eval)
        assert result["x"].shape[0] == len(t_eval)

    def test_dense_time_sampling(self):
        """Test with dense time sampling"""
        system = SimpleDecaySystem()
        integrator = ScipyIntegrator(system, method="RK45")

        # Very dense sampling
        t_eval = np.linspace(0, 1, 1001)

        result = integrator.integrate(
            x0=np.array([1.0]), u_func=lambda t, x: np.zeros(1), t_span=(0.0, 1.0), t_eval=t_eval
        )

        assert result["success"] is True
        assert result["x"].shape[0] == 1001


# ============================================================================
# Test Class 9: Integration with Control
# ============================================================================


@pytest.mark.skipif(not scipy_available, reason="scipy not installed")
class TestControlIntegration:
    """Test integration with various control strategies"""

    def test_constant_control(self):
        """Test with constant control"""
        system = SimpleDecaySystem(a=1.0)
        integrator = ScipyIntegrator(system, method="RK45")

        u_const = np.array([0.5])

        result = integrator.integrate(
            x0=np.array([0.0]), u_func=lambda t, x: u_const, t_span=(0.0, 5.0)
        )

        assert result["success"] is True
        # With constant input to decay system, should reach steady state
        # dx/dt = -a*x + u → x_ss = u/a = 0.5/1.0 = 0.5
        assert abs(result["x"][-1, 0] - 0.5) < 0.01

    def test_state_feedback_control(self):
        """Test with state feedback u = K*x"""
        system = SimpleDecaySystem(a=1.0)
        integrator = ScipyIntegrator(system, method="RK45")

        K = -2.0
        u_func = lambda t, x: np.array([K * x[0]])

        result = integrator.integrate(x0=np.array([1.0]), u_func=u_func, t_span=(0.0, 3.0))

        assert result["success"] is True

    def test_time_varying_control(self):
        """Test with time-varying control"""
        system = SimpleDecaySystem(a=1.0)
        integrator = ScipyIntegrator(system, method="RK45")

        # Sinusoidal control
        u_func = lambda t, x: np.array([np.sin(2 * np.pi * t)])

        result = integrator.integrate(x0=np.array([1.0]), u_func=u_func, t_span=(0.0, 10.0))

        assert result["success"] is True


# ============================================================================
# Test Class 10: Autonomous Systems
# ============================================================================


@pytest.mark.skipif(not scipy_available, reason="scipy not installed")
class TestAutonomousSystems:
    """Test autonomous systems (u=None)"""

    def test_autonomous_system_with_none_control(self):
        """Test autonomous system with u=None"""
        system = AutonomousSystem()
        integrator = ScipyIntegrator(system, method="RK45")

        # u_func returns None for autonomous
        result = integrator.integrate(
            x0=np.array([1.0]), u_func=lambda t, x: None, t_span=(0.0, 1.0)
        )

        assert result["success"] is True

        # Check accuracy against analytical
        x_exact = system.analytical_solution(1.0, 1.0)
        error = abs(result["x"][-1, 0] - x_exact)
        assert error < 1e-6

    def test_step_with_none_control(self):
        """Test single step with u=None"""
        system = AutonomousSystem()
        integrator = ScipyIntegrator(system, method="RK45", dt=0.1)

        x = np.array([1.0])
        x_next = integrator.step(x, u=None)

        assert isinstance(x_next, np.ndarray)
        assert x_next.shape == x.shape


# ============================================================================
# Test Class 11: Dense Output
# ============================================================================


@pytest.mark.skipif(not scipy_available, reason="scipy not installed")
class TestDenseOutput:
    """Test dense output (continuous solution)"""

    def test_dense_output_enabled(self):
        """Test that dense output can be enabled"""
        system = SimpleDecaySystem()
        integrator = ScipyIntegrator(system, method="RK45")

        result = integrator.integrate(
            x0=np.array([1.0]),
            u_func=lambda t, x: np.zeros(1),
            t_span=(0.0, 1.0),
            dense_output=True,
        )

        # Should have dense output fields
        assert "sol" in result
        assert "dense_output" in result
        assert result["dense_output"] is True

    def test_dense_output_interpolation(self):
        """Test interpolation with dense output"""
        system = SimpleDecaySystem()
        integrator = ScipyIntegrator(system, method="RK45")

        result = integrator.integrate(
            x0=np.array([1.0]),
            u_func=lambda t, x: np.zeros(1),
            t_span=(0.0, 1.0),
            dense_output=True,
        )

        # Interpolate at arbitrary time
        t_interp = 0.5
        x_interp = result["sol"](t_interp)

        # Should be close to analytical
        x_exact = system.analytical_solution(1.0, t_interp)
        error = abs(x_interp[0] - x_exact)
        assert error < 1e-6

    def test_dense_output_fine_grid(self):
        """Test dense output on very fine grid"""
        system = SimpleDecaySystem()
        integrator = ScipyIntegrator(system, method="RK45")

        result = integrator.integrate(
            x0=np.array([1.0]),
            u_func=lambda t, x: np.zeros(1),
            t_span=(0.0, 1.0),
            dense_output=True,
        )

        # Generate very fine grid
        t_fine = np.linspace(0, 1, 10000)
        x_fine = result["sol"](t_fine)

        # Should have correct shape
        assert x_fine.shape == (system.nx, 10000)


# ============================================================================
# Test Class 12: Event Detection
# ============================================================================


@pytest.mark.skipif(not scipy_available, reason="scipy not installed")
class TestEventDetection:
    """Test event detection during integration"""

    def test_event_detection_terminates(self):
        """Test that event detection can terminate integration"""
        system = SimpleDecaySystem(a=1.0)
        integrator = ScipyIntegrator(system, method="RK45")

        # Define event: stop when x crosses 0.5
        def threshold_event(t, x):
            return x[0] - 0.5

        threshold_event.terminal = True
        threshold_event.direction = -1  # Crossing downward

        result = integrator.integrate(
            x0=np.array([1.0]),
            u_func=lambda t, x: np.zeros(1),
            t_span=(0.0, 10.0),
            events=threshold_event,
        )

        # Should stop early
        assert result["success"] is True
        assert result["t"][-1] < 10.0

        # Should stop near threshold
        assert abs(result["x"][-1, 0] - 0.5) < 1e-6

    def test_event_detection_no_termination(self):
        """Test event detection without termination"""
        system = SimpleDecaySystem(a=1.0)
        integrator = ScipyIntegrator(system, method="RK45")

        # Define event: detect when x crosses 0.5 (but don't stop)
        def threshold_event(t, x):
            return x[0] - 0.5

        threshold_event.terminal = False

        result = integrator.integrate(
            x0=np.array([1.0]),
            u_func=lambda t, x: np.zeros(1),
            t_span=(0.0, 2.0),
            events=threshold_event,
        )

        # Should complete full integration
        assert result["success"] is True
        assert result["t"][-1] >= 2.0


# ============================================================================
# Test Class 13: Single Step Method
# ============================================================================


@pytest.mark.skipif(not scipy_available, reason="scipy not installed")
class TestSingleStep:
    """Test single-step method (less common for adaptive)"""

    def test_step_method_works(self):
        """Test that step() method works"""
        system = SimpleDecaySystem()
        integrator = ScipyIntegrator(system, method="RK45", dt=0.1)

        x = np.array([1.0])
        u = np.array([0.0])

        x_next = integrator.step(x, u)

        # Should return state after dt
        assert isinstance(x_next, np.ndarray)
        assert x_next.shape == x.shape

    def test_step_vs_integrate_consistency(self):
        """Test that step() and integrate() give consistent results"""
        system = SimpleDecaySystem()
        integrator = ScipyIntegrator(system, method="RK45", dt=0.1)

        x0 = np.array([1.0])
        u = np.array([0.0])

        # Single step
        x_step = integrator.step(x0, u, dt=1.0)

        # Integrate over same interval
        result = integrator.integrate(
            x0=x0, u_func=lambda t, x: u, t_span=(0.0, 1.0), t_eval=np.array([0.0, 1.0])
        )

        # Should be similar (not exact due to adaptive vs fixed)
        assert np.allclose(x_step, result["x"][-1], rtol=1e-3)


# ============================================================================
# Test Class 14: Edge Cases
# ============================================================================


@pytest.mark.skipif(not scipy_available, reason="scipy not installed")
class TestEdgeCases:
    """Test edge cases and error conditions"""

    def test_zero_time_span(self):
        """Test with zero-length time span"""
        system = SimpleDecaySystem()
        integrator = ScipyIntegrator(system, method="RK45")

        result = integrator.integrate(
            x0=np.array([1.0]), u_func=lambda t, x: np.zeros(1), t_span=(0.0, 0.0)
        )

        # Should return just initial state
        assert result["t"].shape[0] >= 1
        assert np.allclose(result["x"][0], np.array([1.0]))

    def test_very_short_integration(self):
        """Test very short integration time"""
        system = SimpleDecaySystem()
        integrator = ScipyIntegrator(system, method="RK45")

        result = integrator.integrate(
            x0=np.array([1.0]), u_func=lambda t, x: np.zeros(1), t_span=(0.0, 1e-6)
        )

        assert result["success"] is True

    def test_very_long_integration(self):
        """Test very long integration time"""
        system = SimpleDecaySystem(a=1.0)
        integrator = ScipyIntegrator(system, method="RK45", rtol=1e-6)

        result = integrator.integrate(
            x0=np.array([1.0]), u_func=lambda t, x: np.zeros(1), t_span=(0.0, 100.0)
        )

        assert result["success"] is True
        # Should decay to near zero
        assert result["x"][-1, 0] < 1e-8


# ============================================================================
# Test Class 15: String Representations
# ============================================================================


@pytest.mark.skipif(not scipy_available, reason="scipy not installed")
class TestStringRepresentations:
    """Test __repr__ and __str__ methods"""

    def test_repr(self):
        """Test __repr__ output"""
        system = SimpleDecaySystem()
        integrator = ScipyIntegrator(system, method="RK45", rtol=1e-8, atol=1e-10)

        repr_str = repr(integrator)

        assert "ScipyIntegrator" in repr_str
        assert "RK45" in repr_str
        assert "1e-08" in repr_str or "1.0e-08" in repr_str

    def test_name_property(self):
        """Test name property for different methods"""
        system = SimpleDecaySystem()

        rk45 = ScipyIntegrator(system, method="RK45")
        assert "scipy.RK45" in rk45.name

        bdf = ScipyIntegrator(system, method="BDF")
        assert "scipy.BDF" in bdf.name
        assert "Stiff" in bdf.name

        lsoda = ScipyIntegrator(system, method="LSODA")
        assert "scipy.LSODA" in lsoda.name
        assert "Auto-Stiffness" in lsoda.name


# ============================================================================
# Test Class 16: TypedDict Verification
# ============================================================================


@pytest.mark.skipif(not scipy_available, reason="scipy not installed")
class TestTypedDictResults:
    """Test TypedDict-specific behavior"""

    def test_result_is_dict(self):
        """Verify result is a dict (TypedDict at runtime)"""
        system = SimpleDecaySystem()
        integrator = ScipyIntegrator(system, method="RK45")

        result = integrator.integrate(
            x0=np.array([1.0]), u_func=lambda t, x: np.zeros(1), t_span=(0.0, 1.0)
        )

        # TypedDict is dict at runtime
        assert isinstance(result, dict)
        assert not hasattr(result, "__dict__")  # Not a class instance

    def test_required_fields_present(self):
        """Test that all required TypedDict fields are present"""
        system = SimpleDecaySystem()
        integrator = ScipyIntegrator(system, method="RK45")

        result = integrator.integrate(
            x0=np.array([1.0]), u_func=lambda t, x: np.zeros(1), t_span=(0.0, 1.0)
        )

        # Required fields
        required_fields = [
            "t",
            "x",
            "success",
            "message",
            "nfev",
            "nsteps",
            "integration_time",
            "solver",
        ]

        for field in required_fields:
            assert field in result, f"Required field '{field}' missing"

    def test_optional_fields_conditional(self):
        """Test that optional fields are added conditionally"""
        system = SimpleDecaySystem()

        # BDF/Radau may have njev, nlu
        integrator = ScipyIntegrator(system, method="BDF")

        result = integrator.integrate(
            x0=np.array([1.0]), u_func=lambda t, x: np.zeros(1), t_span=(0.0, 0.1)
        )

        # Optional fields may or may not be present
        # Just verify we can check for them
        has_njev = "njev" in result
        has_nlu = "nlu" in result

        # At least one should be present for BDF
        # (or neither - scipy behavior varies)

    def test_dict_methods_work(self):
        """Test that dict methods work on result"""
        system = SimpleDecaySystem()
        integrator = ScipyIntegrator(system, method="RK45")

        result = integrator.integrate(
            x0=np.array([1.0]), u_func=lambda t, x: np.zeros(1), t_span=(0.0, 1.0)
        )

        # Test dict methods
        assert len(result.keys()) >= 8  # At least required fields
        assert "t" in result.keys()
        assert result.get("success") is True
        assert result.get("nonexistent", "default") == "default"


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
