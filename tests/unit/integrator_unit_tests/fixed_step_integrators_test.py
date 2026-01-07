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
Unit tests for fixed-step integrators - REFACTORED FOR TypedDict

Tests cover:
1. Explicit Euler integrator
2. Midpoint (RK2) integrator
3. RK4 integrator
4. Single-step vs multi-step integration
5. Accuracy verification against analytical solutions
6. Convergence order verification
7. Factory function
8. Backend support

Refactoring Changes
-------------------
- Results are now TypedDict instead of class instances
- Changed result.field → result["field"] throughout
- Changed isinstance(result, IntegrationResult) → isinstance(result, dict)
- Changed hasattr(result, "field") → "field" in result
- All functionality and test coverage preserved
"""

import numpy as np
import pytest

from cdesym.systems.base.numerical_integration.fixed_step_integrators import (
    ExplicitEulerIntegrator,
    HeunIntegrator,
    MidpointIntegrator,
    RK4Integrator,
    create_fixed_step_integrator,
)
from cdesym.systems.base.numerical_integration.integrator_base import StepMode

# Conditional imports
torch_available = True
try:
    import torch
except ImportError:
    torch_available = False

jax_available = True
try:
    import jax
    import jax.numpy as jnp
except ImportError:
    jax_available = False


# ============================================================================
# Mock System with Analytical Solution
# ============================================================================


class ExponentialDecaySystem:
    """Simple system: dx/dt = -a*x with analytical solution"""

    def __init__(self, a=1.0):
        self.a = a
        self.nx = 1
        self.nu = 1
        self.order = 1
        self._initialized = True
        self._default_backend = "numpy"

    def __call__(self, x, u, backend="numpy"):
        """Evaluate dynamics: dx/dt = -a*x (control ignored)"""
        return -self.a * x

    def analytical_solution(self, x0, t):
        """Analytical solution: x(t) = x0 * exp(-a*t)"""
        return x0 * np.exp(-self.a * t)


class HarmonicOscillator:
    """Harmonic oscillator: d²x/dt² = -k*x with analytical solution"""

    def __init__(self, k=1.0):
        self.k = k
        self.omega = np.sqrt(k)  # Natural frequency
        self.nx = 2  # [x, v]
        self.nu = 1
        self.order = 1  # First-order state-space form
        self._initialized = True
        self._default_backend = "numpy"

    def __call__(self, x, u, backend="numpy"):
        """Dynamics: [v, -k*x]"""
        if len(x.shape) == 1:
            return np.array([x[1], -self.k * x[0]])
        # Batched
        return np.column_stack([x[:, 1], -self.k * x[:, 0]])

    def analytical_solution(self, x0, v0, t):
        """Analytical solution: x(t) = x0*cos(ωt) + (v0/ω)*sin(ωt)"""
        x = x0 * np.cos(self.omega * t) + (v0 / self.omega) * np.sin(self.omega * t)
        v = -x0 * self.omega * np.sin(self.omega * t) + v0 * np.cos(self.omega * t)
        return np.array([x, v])


# ============================================================================
# Test Class 1: Explicit Euler Integrator
# ============================================================================


class TestExplicitEuler:
    """Test Explicit Euler integrator"""

    def test_initialization(self):
        """Test Euler integrator initialization"""
        system = ExponentialDecaySystem()
        integrator = ExplicitEulerIntegrator(system, dt=0.01, backend="numpy")

        assert integrator.dt == 0.01
        assert integrator.step_mode == StepMode.FIXED
        assert integrator.backend == "numpy"
        assert "Euler" in integrator.name

    def test_single_step_decay(self):
        """Test single Euler step on exponential decay"""
        system = ExponentialDecaySystem(a=2.0)
        integrator = ExplicitEulerIntegrator(system, dt=0.1, backend="numpy")

        x = np.array([1.0])
        u = np.array([0.0])

        x_next = integrator.step(x, u)

        # x_next = x + dt * f(x) = 1 + 0.1 * (-2*1) = 0.8
        assert np.allclose(x_next, np.array([0.8]))

    def test_multi_step_integration(self):
        """Test multi-step integration returns TypedDict"""
        system = ExponentialDecaySystem(a=1.0)
        integrator = ExplicitEulerIntegrator(system, dt=0.1, backend="numpy")

        x0 = np.array([1.0])
        u_func = lambda t, x: np.zeros(1)

        result = integrator.integrate(x0=x0, u_func=u_func, t_span=(0.0, 1.0))

        # Verify TypedDict (dict) instead of class instance
        assert isinstance(result, dict)
        assert result["success"] is True
        assert result["t"][0] == 0.0
        assert result["t"][-1] == 1.0
        assert result["x"].shape[1] == 1  # nx=1

    def test_accuracy_exponential_decay(self):
        """Test accuracy against analytical solution"""
        system = ExponentialDecaySystem(a=1.0)
        integrator = ExplicitEulerIntegrator(system, dt=0.01, backend="numpy")

        x0 = np.array([1.0])
        result = integrator.integrate(x0=x0, u_func=lambda t, x: np.zeros(1), t_span=(0.0, 1.0))

        # Compare to analytical
        x_analytical = system.analytical_solution(x0[0], 1.0)
        x_numerical = result["x"][-1, 0]

        # Euler with dt=0.01 should have reasonable accuracy
        error = abs(x_numerical - x_analytical)
        assert error < 0.01  # 1% error acceptable for Euler

    def test_integration_result_fields(self):
        """Test that IntegrationResult has expected fields (TypedDict)"""
        system = ExponentialDecaySystem()
        integrator = ExplicitEulerIntegrator(system, dt=0.01, backend="numpy")

        result = integrator.integrate(
            x0=np.array([1.0]),
            u_func=lambda t, x: np.zeros(1),
            t_span=(0.0, 0.5),
        )

        # Verify required fields exist (dict access)
        assert "t" in result
        assert "x" in result
        assert "success" in result
        assert "nfev" in result
        assert "nsteps" in result
        assert "message" in result
        assert "solver" in result
        assert "integration_time" in result

        # Verify values
        assert result["nsteps"] > 0
        assert result["nfev"] > 0


# ============================================================================
# Test Class 2: Midpoint Integrator
# ============================================================================


class TestMidpoint:
    """Test Midpoint (RK2) integrator"""

    def test_initialization(self):
        """Test Midpoint integrator initialization"""
        system = ExponentialDecaySystem()
        integrator = MidpointIntegrator(system, dt=0.01, backend="numpy")

        assert integrator.dt == 0.01
        assert "Midpoint" in integrator.name or "RK2" in integrator.name

    def test_single_step(self):
        """Test single midpoint step"""
        system = ExponentialDecaySystem(a=2.0)
        integrator = MidpointIntegrator(system, dt=0.1, backend="numpy")

        x = np.array([1.0])
        u = np.array([0.0])

        x_next = integrator.step(x, u)

        # k1 = -2*1 = -2
        # x_mid = 1 + 0.05*(-2) = 0.9
        # k2 = -2*0.9 = -1.8
        # x_next = 1 + 0.1*(-1.8) = 0.82
        assert np.allclose(x_next, np.array([0.82]))

    def test_better_than_euler(self):
        """Test that Midpoint is more accurate than Euler"""
        system = ExponentialDecaySystem(a=1.0)
        dt = 0.1

        euler = ExplicitEulerIntegrator(system, dt=dt, backend="numpy")
        midpoint = MidpointIntegrator(system, dt=dt, backend="numpy")

        x0 = np.array([1.0])
        u_func = lambda t, x: np.zeros(1)

        result_euler = euler.integrate(x0, u_func, (0.0, 1.0))
        result_midpoint = midpoint.integrate(x0, u_func, (0.0, 1.0))

        x_exact = system.analytical_solution(x0[0], 1.0)

        error_euler = abs(result_euler["x"][-1, 0] - x_exact)
        error_midpoint = abs(result_midpoint["x"][-1, 0] - x_exact)

        # Midpoint should be significantly better
        assert error_midpoint < error_euler
        assert error_midpoint < 0.001  # Much better accuracy

    def test_convergence_order_2(self):
        """Verify that Midpoint has order 2 convergence"""
        system = ExponentialDecaySystem(a=1.0)
        x0 = np.array([1.0])
        t_final = 1.0
        x_exact = system.analytical_solution(x0[0], t_final)

        # Test with decreasing dt
        dt_values = [0.1, 0.05, 0.025]
        errors = []

        for dt in dt_values:
            integrator = MidpointIntegrator(system, dt=dt, backend="numpy")
            result = integrator.integrate(x0, lambda t, x: np.zeros(1), (0.0, t_final))
            error = abs(result["x"][-1, 0] - x_exact)
            errors.append(error)

        # Check order 2: error(dt) ≈ C * dt²
        # So error(dt/2) / error(dt) ≈ 1/4
        ratio1 = errors[0] / errors[1]  # Should be ~4
        ratio2 = errors[1] / errors[2]  # Should be ~4

        assert 3.0 < ratio1 < 5.0  # Allow some tolerance
        assert 3.0 < ratio2 < 5.0

# ============================================================================
# Test Class 3: HeunIntegrator
# ============================================================================


class TestHeunIntegrator:
    """Test Heun's method (Improved Euler / Explicit Trapezoid) integrator."""

    def test_single_step_accuracy(self):
        """Test Heun's method single step against analytical solution."""
        system = ExponentialDecaySystem(a=1.0)
        integrator = HeunIntegrator(system, dt=0.1, backend="numpy")

        x0 = np.array([1.0])
        u = np.array([0.0])

        # Take one Heun step
        x1 = integrator.step(x0, u)

        # Analytical solution
        x1_exact = system.analytical_solution(x0[0], 0.1)

        # Heun (2nd order) should be more accurate than Euler
        error = abs(x1[0] - x1_exact)
        assert error < 0.001, f"Heun error {error} too large"

    def test_predictor_corrector_structure(self):
        """Test that Heun uses predictor-corrector structure (2 evals per step)."""
        system = ExponentialDecaySystem(a=1.0)
        integrator = HeunIntegrator(system, dt=0.1, backend="numpy")

        x0 = np.array([1.0])
        
        # Take a step
        integrator.step(x0, np.array([0.0]))
        
        # Heun should have 1 step, but we can't directly check nfev in step()
        # This is tested more thoroughly in test_function_evaluations

    def test_convergence_order(self):
        """Verify Heun's method exhibits 2nd-order convergence."""
        system = ExponentialDecaySystem(a=1.0)
        x0 = np.array([1.0])
        t_final = 1.0

        # Test with different dt values
        dt_values = [0.1, 0.05, 0.025]
        errors = []

        for dt in dt_values:
            integrator = HeunIntegrator(system, dt=dt, backend="numpy")
            
            # Integrate
            result = integrator.integrate(
                x0=x0,
                u_func=lambda t, x: np.array([0.0]),
                t_span=(0.0, t_final),
            )

            # Compare to analytical
            x_final_exact = system.analytical_solution(x0[0], t_final)
            error = abs(result["x"][-1, 0] - x_final_exact)
            errors.append(error)

        # Check convergence order (error should decrease by ~4x when dt halves)
        ratio1 = errors[0] / errors[1]
        ratio2 = errors[1] / errors[2]

        # For 2nd order: ratio should be close to 4.0
        assert 3.0 < ratio1 < 5.0, f"First ratio {ratio1} not ~4 (2nd order)"
        assert 3.0 < ratio2 < 5.0, f"Second ratio {ratio2} not ~4 (2nd order)"

    def test_heun_vs_euler_accuracy(self):
        """Verify Heun is significantly more accurate than Euler."""
        system = ExponentialDecaySystem(a=1.0)
        x0 = np.array([1.0])
        dt = 0.1
        t_final = 1.0

        # Heun integration
        heun = HeunIntegrator(system, dt=dt, backend="numpy")
        result_heun = heun.integrate(
            x0=x0,
            u_func=lambda t, x: np.array([0.0]),
            t_span=(0.0, t_final),
        )

        # Euler integration
        euler = ExplicitEulerIntegrator(system, dt=dt, backend="numpy")
        result_euler = euler.integrate(
            x0=x0,
            u_func=lambda t, x: np.array([0.0]),
            t_span=(0.0, t_final),
        )

        # Exact solution
        x_exact = system.analytical_solution(x0[0], t_final)

        # Calculate errors
        error_heun = abs(result_heun["x"][-1, 0] - x_exact)
        error_euler = abs(result_euler["x"][-1, 0] - x_exact)

        # Heun should be at least 10x more accurate
        assert error_heun < error_euler / 10, (
            f"Heun ({error_heun}) not significantly better than Euler ({error_euler})"
        )

    def test_heun_vs_midpoint_similar_accuracy(self):
        """Verify Heun and Midpoint have similar accuracy (both 2nd order)."""
        system = ExponentialDecaySystem(a=1.0)
        x0 = np.array([1.0])
        dt = 0.05
        t_final = 1.0

        # Heun integration
        heun = HeunIntegrator(system, dt=dt, backend="numpy")
        result_heun = heun.integrate(
            x0=x0,
            u_func=lambda t, x: np.array([0.0]),
            t_span=(0.0, t_final),
        )

        # Midpoint integration
        midpoint = MidpointIntegrator(system, dt=dt, backend="numpy")
        result_midpoint = midpoint.integrate(
            x0=x0,
            u_func=lambda t, x: np.array([0.0]),
            t_span=(0.0, t_final),
        )

        # Exact solution
        x_exact = system.analytical_solution(x0[0], t_final)

        # Calculate errors
        error_heun = abs(result_heun["x"][-1, 0] - x_exact)
        error_midpoint = abs(result_midpoint["x"][-1, 0] - x_exact)

        # Both are 2nd order, should be within factor of 2 of each other
        ratio = max(error_heun, error_midpoint) / min(error_heun, error_midpoint)
        assert ratio < 2.0, (
            f"Heun ({error_heun}) and Midpoint ({error_midpoint}) accuracy too different"
        )

    def test_function_evaluations(self):
        """Test Heun uses exactly 2 function evaluations per step."""
        system = ExponentialDecaySystem()
        integrator = HeunIntegrator(system, dt=0.1, backend="numpy")

        result = integrator.integrate(
            x0=np.array([1.0]),
            u_func=lambda t, x: np.array([0.0]),
            t_span=(0.0, 1.0),
        )

        # Heun method: predictor + corrector = 2 evaluations per step
        assert result["nfev"] == 2 * result["nsteps"], (
            f"Heun should use 2 evals/step: {result['nfev']} != 2 * {result['nsteps']}"
        )

    def test_integrator_name(self):
        """Test integrator name property."""
        system = ExponentialDecaySystem()
        integrator = HeunIntegrator(system, dt=0.1, backend="numpy")

        name = integrator.name.lower()
        assert "heun" in name or "improved euler" in name, (
            f"Name '{integrator.name}' should contain 'heun' or 'improved euler'"
        )

    def test_step_mode(self):
        """Test that step_mode is FIXED."""
        system = ExponentialDecaySystem()
        integrator = HeunIntegrator(system, dt=0.1, backend="numpy")
        
        assert integrator.step_mode == StepMode.FIXED

    def test_harmonic_oscillator(self):
        """Test Heun on 2D harmonic oscillator system."""
        system = HarmonicOscillator(k=1.0)
        integrator = HeunIntegrator(system, dt=0.05, backend="numpy")

        x0 = np.array([1.0, 0.0])  # [position, velocity]
        t_final = 2 * np.pi  # One full period

        result = integrator.integrate(
            x0=x0,
            u_func=lambda t, x: np.array([0.0]),
            t_span=(0.0, t_final),
        )

        # After one period, should be back near start
        x_final = result["x"][-1]
        error = np.linalg.norm(x_final - x0)
        
        # Heun should maintain reasonable accuracy over one period
        assert error < 0.1, f"Harmonic oscillator error {error} too large"

    @pytest.mark.skipif(not torch_available, reason="PyTorch not installed")
    def test_torch_backend(self):
        """Test Heun with PyTorch backend."""
        system = ExponentialDecaySystem()
        integrator = HeunIntegrator(system, dt=0.1, backend="torch")

        x0 = torch.tensor([1.0])
        u = torch.tensor([0.0])

        x1 = integrator.step(x0, u)
        
        assert isinstance(x1, torch.Tensor), "Result should be torch.Tensor"
        assert integrator.backend == "torch"

    @pytest.mark.skipif(not jax_available, reason="JAX not installed")
    def test_jax_backend(self):
        """Test Heun with JAX backend."""
        system = ExponentialDecaySystem()
        integrator = HeunIntegrator(system, dt=0.1, backend="jax")

        x0 = jnp.array([1.0])
        u = jnp.array([0.0])

        x1 = integrator.step(x0, u)
        
        assert isinstance(x1, jnp.ndarray), "Result should be jax.numpy.ndarray"
        assert integrator.backend == "jax"

    def test_result_is_typed_dict(self):
        """Test that integrate returns a TypedDict (dict)."""
        system = ExponentialDecaySystem()
        integrator = HeunIntegrator(system, dt=0.1, backend="numpy")

        result = integrator.integrate(
            x0=np.array([1.0]),
            u_func=lambda t, x: np.array([0.0]),
            t_span=(0.0, 1.0),
        )

        # Should be a dict (TypedDict)
        assert isinstance(result, dict)
        
        # Should have required keys
        assert "t" in result
        assert "x" in result
        assert "success" in result
        assert "nfev" in result
        assert "nsteps" in result
        assert "solver" in result

    def test_statistics_tracking(self):
        """Test that statistics are properly tracked."""
        system = ExponentialDecaySystem()
        integrator = HeunIntegrator(system, dt=0.1, backend="numpy")

        result = integrator.integrate(
            x0=np.array([1.0]),
            u_func=lambda t, x: np.array([0.0]),
            t_span=(0.0, 1.0),
        )

        stats = integrator.get_stats()
        
        assert stats["total_steps"] == result["nsteps"]
        assert stats["total_time"] > 0
        assert stats["total_fev"] == result["nfev"]

# ============================================================================
# Test Class 4: RK4 Integrator
# ============================================================================


class TestRK4:
    """Test RK4 (4th-order) integrator"""

    def test_initialization(self):
        """Test RK4 integrator initialization"""
        system = ExponentialDecaySystem()
        integrator = RK4Integrator(system, dt=0.01, backend="numpy")

        assert integrator.dt == 0.01
        assert "RK4" in integrator.name

    def test_single_step(self):
        """Test single RK4 step"""
        system = ExponentialDecaySystem(a=1.0)
        integrator = RK4Integrator(system, dt=0.1, backend="numpy")

        x = np.array([1.0])
        u = np.array([0.0])

        x_next = integrator.step(x, u)

        # Should be close to analytical solution
        x_exact = system.analytical_solution(1.0, 0.1)
        assert np.allclose(x_next[0], x_exact, rtol=1e-6)

    def test_high_accuracy(self):
        """Test RK4 high accuracy"""
        system = ExponentialDecaySystem(a=1.0)
        integrator = RK4Integrator(system, dt=0.1, backend="numpy")

        x0 = np.array([1.0])
        result = integrator.integrate(x0, lambda t, x: np.zeros(1), (0.0, 1.0))

        x_exact = system.analytical_solution(x0[0], 1.0)
        error = abs(result["x"][-1, 0] - x_exact)

        # RK4 should have very high accuracy
        assert error < 1e-6

    def test_four_function_evaluations_per_step(self):
        """Test that RK4 uses 4 function evaluations per step"""
        system = ExponentialDecaySystem()
        integrator = RK4Integrator(system, dt=0.1, backend="numpy")

        x0 = np.array([1.0])
        integrator.reset_stats()

        result = integrator.integrate(x0, lambda t, x: np.zeros(1), (0.0, 1.0))

        # Should be 4 evaluations per step
        expected_fev = result["nsteps"] * 4
        assert result["nfev"] == expected_fev

    def test_convergence_order_4(self):
        """Verify that RK4 has order 4 convergence"""
        system = ExponentialDecaySystem(a=1.0)
        x0 = np.array([1.0])
        t_final = 1.0
        x_exact = system.analytical_solution(x0[0], t_final)

        # Test with decreasing dt
        dt_values = [0.1, 0.05, 0.025]
        errors = []

        for dt in dt_values:
            integrator = RK4Integrator(system, dt=dt, backend="numpy")
            result = integrator.integrate(x0, lambda t, x: np.zeros(1), (0.0, t_final))
            error = abs(result["x"][-1, 0] - x_exact)
            errors.append(error)

        # Check order 4: error(dt) ≈ C * dt⁴
        # So error(dt/2) / error(dt) ≈ 1/16
        ratio1 = errors[0] / errors[1]
        ratio2 = errors[1] / errors[2]

        # Should be close to 16
        assert 10.0 < ratio1 < 20.0
        assert 10.0 < ratio2 < 20.0


# ============================================================================
# Test Class 5: Accuracy Comparison
# ============================================================================


class TestAccuracyComparison:
    """Compare accuracy of different integrators"""

    def test_euler_vs_rk4_accuracy(self):
        """Compare Euler vs RK4 accuracy on same problem"""
        system = ExponentialDecaySystem(a=1.0)
        dt = 0.1

        euler = ExplicitEulerIntegrator(system, dt=dt, backend="numpy")
        rk4 = RK4Integrator(system, dt=dt, backend="numpy")

        x0 = np.array([1.0])
        u_func = lambda t, x: np.zeros(1)

        result_euler = euler.integrate(x0, u_func, (0.0, 1.0))
        result_rk4 = rk4.integrate(x0, u_func, (0.0, 1.0))

        x_exact = system.analytical_solution(x0[0], 1.0)
        error_euler = abs(result_euler["x"][-1, 0] - x_exact)
        error_rk4 = abs(result_rk4["x"][-1, 0] - x_exact)

        # Error reduction should be > 100x for same dt
        assert error_euler / error_rk4 > 100


# ============================================================================
# Test Class 6: Integration with Control
# ============================================================================


class TestIntegrationWithControl:
    """Test integration with various control strategies"""

    def test_zero_control(self):
        """Test integration with zero control"""
        system = ExponentialDecaySystem(a=1.0)
        integrator = RK4Integrator(system, dt=0.01, backend="numpy")

        result = integrator.integrate(
            x0=np.array([1.0]),
            u_func=lambda t, x: np.zeros(1),
            t_span=(0.0, 1.0),
        )

        # Should match uncontrolled dynamics
        x_exact = system.analytical_solution(1.0, 1.0)
        assert np.allclose(result["x"][-1, 0], x_exact, rtol=1e-5)

    def test_state_feedback_control(self):
        """Test with state feedback u(x)"""
        system = ExponentialDecaySystem(a=1.0)
        integrator = RK4Integrator(system, dt=0.01, backend="numpy")

        # Proportional control
        K = -0.5
        u_func = lambda t, x: np.array([K * x[0]])

        result = integrator.integrate(x0=np.array([1.0]), u_func=u_func, t_span=(0.0, 2.0))

        assert result["success"] is True

    def test_time_varying_control(self):
        """Test with time-varying control u(t)"""
        system = ExponentialDecaySystem(a=1.0)
        integrator = RK4Integrator(system, dt=0.01, backend="numpy")

        # Sinusoidal input
        u_func = lambda t, x: np.array([0.5 * np.sin(2 * np.pi * t)])

        result = integrator.integrate(x0=np.array([0.0]), u_func=u_func, t_span=(0.0, 5.0))

        assert result["success"] is True


# ============================================================================
# Test Class 7: Custom Time Evaluation Points
# ============================================================================


class TestCustomTimeEvaluation:
    """Test integration with custom t_eval"""

    def test_uniform_time_grid(self):
        """Test with uniform time grid"""
        system = ExponentialDecaySystem()
        integrator = RK4Integrator(system, dt=0.01, backend="numpy")

        t_eval = np.linspace(0, 1, 101)  # Uniform grid

        result = integrator.integrate(
            x0=np.array([1.0]),
            u_func=lambda t, x: np.zeros(1),
            t_span=(0.0, 1.0),
            t_eval=t_eval,
        )

        # Should match requested points
        assert np.allclose(result["t"], t_eval)
        assert result["x"].shape[0] == len(t_eval)

    def test_nonuniform_time_grid(self):
        """Test with non-uniform time grid"""
        system = ExponentialDecaySystem()
        integrator = RK4Integrator(system, dt=0.01, backend="numpy")

        # Non-uniform: dense at start, sparse at end
        t_eval = np.concatenate([np.linspace(0, 0.1, 50), np.linspace(0.1, 1.0, 20)])

        result = integrator.integrate(
            x0=np.array([1.0]),
            u_func=lambda t, x: np.zeros(1),
            t_span=(0.0, 1.0),
            t_eval=t_eval,
        )

        assert result["success"] is True
        assert result["x"].shape[0] == len(t_eval)


# ============================================================================
# Test Class 8: Backend Support
# ============================================================================


class TestBackendSupport:
    """Test integration across different backends"""

    def test_numpy_backend(self):
        """Test NumPy backend"""
        system = ExponentialDecaySystem()
        integrator = RK4Integrator(system, dt=0.01, backend="numpy")

        x = np.array([1.0])
        u = np.array([0.0])

        x_next = integrator.step(x, u)

        assert isinstance(x_next, np.ndarray)

    @pytest.mark.skipif(not torch_available, reason="PyTorch not installed")
    def test_torch_backend(self):
        """Test PyTorch backend"""
        system = ExponentialDecaySystem()
        integrator = RK4Integrator(system, dt=0.01, backend="torch")

        x = torch.tensor([1.0])
        u = torch.tensor([0.0])

        x_next = integrator.step(x, u)

        assert isinstance(x_next, torch.Tensor)

    @pytest.mark.skipif(not jax_available, reason="JAX not installed")
    def test_jax_backend(self):
        """Test JAX backend"""
        system = ExponentialDecaySystem()
        integrator = RK4Integrator(system, dt=0.01, backend="jax")

        x = jnp.array([1.0])
        u = jnp.array([0.0])

        x_next = integrator.step(x, u)

        assert isinstance(x_next, jnp.ndarray)


# ============================================================================
# Test Class 9: Factory Function
# ============================================================================


class TestFactoryFunction:
    """Test create_fixed_step_integrator factory"""

    def test_create_euler(self):
        """Test creating Euler integrator via factory"""
        system = ExponentialDecaySystem()
        integrator = create_fixed_step_integrator("euler", system, dt=0.01)

        assert isinstance(integrator, ExplicitEulerIntegrator)
        assert integrator.dt == 0.01

    def test_create_midpoint(self):
        """Test creating Midpoint integrator via factory"""
        system = ExponentialDecaySystem()
        integrator = create_fixed_step_integrator("midpoint", system, dt=0.01)

        assert isinstance(integrator, MidpointIntegrator)

    def test_create_rk4(self):
        """Test creating RK4 integrator via factory"""
        system = ExponentialDecaySystem()
        integrator = create_fixed_step_integrator("rk4", system, dt=0.01)

        assert isinstance(integrator, RK4Integrator)

    def test_create_with_backend(self):
        """Test factory respects backend parameter"""
        system = ExponentialDecaySystem()
        integrator = create_fixed_step_integrator("rk4", system, dt=0.01, backend="numpy")

        assert integrator.backend == "numpy"

    def test_invalid_method_name(self):
        """Test factory raises error for invalid method"""
        system = ExponentialDecaySystem()

        with pytest.raises(ValueError, match="Unknown method"):
            create_fixed_step_integrator("invalid", system, dt=0.01)
            
    def test_create_heun(self):
        """Test creating Heun integrator via factory"""
        system = ExponentialDecaySystem()
        integrator = create_fixed_step_integrator("heun", system, dt=0.01)

        assert isinstance(integrator, HeunIntegrator)
        assert integrator.dt == 0.01
        assert integrator.backend == "numpy"


# ============================================================================
# Test Class 10: Performance Statistics
# ============================================================================


class TestPerformanceStatistics:
    """Test performance tracking across integrators"""

    def test_stats_accumulate_across_steps(self):
        """Test that statistics accumulate"""
        system = ExponentialDecaySystem()
        integrator = RK4Integrator(system, dt=0.1, backend="numpy")

        x = np.array([1.0])
        u = np.array([0.0])

        integrator.reset_stats()

        # Take multiple steps
        for _ in range(10):
            x = integrator.step(x, u)

        stats = integrator.get_stats()

        assert stats["total_steps"] == 10
        assert stats["total_fev"] == 40  # 4 per step for RK4
        assert stats["avg_fev_per_step"] == 4.0

    def test_integration_tracks_time(self):
        """Test that integration time is tracked in result"""
        system = ExponentialDecaySystem()
        integrator = RK4Integrator(system, dt=0.01, backend="numpy")

        result = integrator.integrate(
            x0=np.array([1.0]),
            u_func=lambda t, x: np.zeros(1),
            t_span=(0.0, 1.0),
        )

        # Verify result has required fields
        assert result["success"] is True
        assert result["nsteps"] > 0
        assert "integration_time" in result
        assert result["integration_time"] >= 0.0  # Should be non-negative


# ============================================================================
# Test Class 11: TypedDict Verification
# ============================================================================


class TestTypedDictResults:
    """Test TypedDict-specific behavior"""

    def test_result_is_dict(self):
        """Verify result is a dict (TypedDict at runtime)"""
        system = ExponentialDecaySystem()
        integrator = RK4Integrator(system, dt=0.01, backend="numpy")

        result = integrator.integrate(
            x0=np.array([1.0]),
            u_func=lambda t, x: np.zeros(1),
            t_span=(0.0, 1.0),
        )

        # TypedDict is dict at runtime
        assert isinstance(result, dict)
        assert not hasattr(result, "__dict__")  # Not a class instance

    def test_all_required_fields_present(self):
        """Test that all required TypedDict fields are present"""
        system = ExponentialDecaySystem()
        integrator = RK4Integrator(system, dt=0.01, backend="numpy")

        result = integrator.integrate(
            x0=np.array([1.0]),
            u_func=lambda t, x: np.zeros(1),
            t_span=(0.0, 1.0),
        )

        # Required fields from IntegrationResult TypedDict
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
            assert field in result, f"Required field '{field}' missing from result"

    def test_dict_access_methods_work(self):
        """Test that dict methods work on result"""
        system = ExponentialDecaySystem()
        integrator = RK4Integrator(system, dt=0.01, backend="numpy")

        result = integrator.integrate(
            x0=np.array([1.0]),
            u_func=lambda t, x: np.zeros(1),
            t_span=(0.0, 1.0),
        )

        # Test dict methods
        assert len(result.keys()) >= 8  # At least required fields
        assert "t" in result.keys()
        assert result.get("success") is True
        assert result.get("nonexistent_field", "default") == "default"

    def test_result_can_be_unpacked(self):
        """Test that result can be unpacked like a dict"""
        system = ExponentialDecaySystem()
        integrator = RK4Integrator(system, dt=0.01, backend="numpy")

        result = integrator.integrate(
            x0=np.array([1.0]),
            u_func=lambda t, x: np.zeros(1),
            t_span=(0.0, 1.0),
        )

        # Unpack specific fields
        t = result["t"]
        x = result["x"]
        success = result["success"]

        assert len(t) > 0
        assert x.shape[0] == len(t)
        assert success is True


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
