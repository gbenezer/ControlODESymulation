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

import unittest
from typing import Optional
from unittest.mock import Mock, MagicMock

import numpy as np

from src.types.core import ControlVector, StateVector
from src.types.linearization import ContinuousLinearization
from src.types.trajectories import IntegrationResult, SimulationResult
from src.systems.base.core.continuous_system_base import ContinuousSystemBase

# Conditional imports for backends
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


# =============================================================================
# Test System Implementations
# =============================================================================

class SimpleContinuousSystem(ContinuousSystemBase):
    """Concrete implementation for testing: dx/dt = -x + u"""

    def __init__(self, nx=2, nu=1):
        self.nx = nx
        self.nu = nu
        self.ny = nx
        # Store linearization matrices
        self.A = -np.eye(nx)
        self.B = np.ones((nx, nu))
    
    def __call__(self, x, u=None, t=0.0):
        if u is None:
            u = np.zeros(self.nu)
        
        if x.ndim == 2:
            u = u if u.ndim == 2 else u.reshape(-1, 1)
        
        return self.A @ x + self.B @ u  # Explicit and clear

    def integrate(
        self, x0: StateVector, u=None, t_span=(0.0, 1.0), method="RK45", **kwargs
    ) -> IntegrationResult:
        """Simple Euler integration returning IntegrationResult."""
        t_start, t_end = t_span
        dt_integrator = kwargs.get("max_step", 0.01)
        t_eval = kwargs.get("t_eval", None)  # Support requested time points
        
        # Handle backward integration
        if t_end < t_start:
            dt_integrator = -dt_integrator

        t_points = []
        states_list = []
        
        t = t_start
        x = x0.copy()
        nfev = 0
        
        t_points.append(t)
        states_list.append(x.copy())

        # Integration loop
        if t_end >= t_start:
            while t < t_end:
                if u is None:
                    u_val = None
                elif callable(u):
                    u_val = u(t, x)  # Note: signature is (t, x) not (x, t)
                else:
                    u_val = u

                dxdt = self(x, u_val, t)
                nfev += 1
                
                x = x + dt_integrator * dxdt
                t = min(t + dt_integrator, t_end)
                
                t_points.append(t)
                states_list.append(x.copy())
        else:
            # Backward integration
            while t > t_end:
                if u is None:
                    u_val = None
                elif callable(u):
                    u_val = u(t, x)
                else:
                    u_val = u

                dxdt = self(x, u_val, t)
                nfev += 1
                
                x = x + dt_integrator * dxdt
                t = max(t + dt_integrator, t_end)
                
                t_points.append(t)
                states_list.append(x.copy())

        t_array = np.array(t_points)
        states_array = np.array(states_list)  # Shape: (T, nx) - time-major
        
        # If specific times requested, interpolate
        if t_eval is not None:
            states_interp = np.zeros((len(t_eval), self.nx))
            for i in range(self.nx):
                states_interp[:, i] = np.interp(t_eval, t_array, states_array[:, i])
            t_array = np.array(t_eval)
            states_array = states_interp

        return {
            "t": t_array,
            "x": states_array,  # Changed from "y", now (T, nx)
            "success": True,
            "message": "Integration successful",
            "nfev": nfev,
            "njev": 0,
            "nlu": 0,
            "status": 0,
            "nsteps": len(t_array) - 1,
            "integration_time": 0.0,
            "solver": method
        }

    def linearize(self, x_eq, u_eq=None):
        return (self.A, self.B)  # Just return stored matrices


class TimeVaryingSystem(ContinuousSystemBase):
    """Time-varying system: dx/dt = -t*x + u"""
    
    def __init__(self, nx=2, nu=1):
        self.nx = nx
        self.nu = nu
        self.ny = nx
    
    def __call__(self, x, u=None, t=0.0):
        u = u if u is not None else np.zeros(self.nu)
        return -t * x + u
    
    def integrate(self, x0, u=None, t_span=(0.0, 1.0), method="RK45", **kwargs):
        t_eval = kwargs.get("t_eval", None)
        
        if t_eval is not None:
            t_points = np.array(t_eval)
        else:
            t_points = np.linspace(t_span[0], t_span[1], 50)
        
        # Simple analytical solution: x(t) = x0 * exp(-t²/2)
        states = np.outer(np.exp(-0.5 * t_points**2), x0)  # (T, nx)
        
        return {
            "t": t_points,
            "x": states,  # Changed from "y", now (T, nx)
            "success": True,
            "message": "Success",
            "nfev": 50,
            "njev": 0,
            "nlu": 0,
            "status": 0,
            "nsteps": len(t_points) - 1,
            "integration_time": 0.0,
            "solver": method
        }
    
    def linearize(self, x_eq, u_eq=None):
        # At time t=0
        A = np.zeros((self.nx, self.nx))  # -t*I at t=0
        B = np.ones((self.nx, self.nu))
        return (A, B)
    
    @property
    def is_time_varying(self):
        return True


class StochasticSystem(ContinuousSystemBase):
    """Stochastic system for testing: dx = (-x + u)dt + G*dW"""
    
    def __init__(self, nx=2, nu=1, nw=2):
        self.nx = nx
        self.nu = nu
        self.nw = nw
        self.ny = nx
    
    def __call__(self, x, u=None, t=0.0):
        u = u if u is not None else np.zeros(self.nu)
        return -x + u
    
    def integrate(self, x0, u=None, t_span=(0.0, 1.0), method="EM", **kwargs):
        t_eval = kwargs.get("t_eval", None)
        
        if t_eval is not None:
            t_points = np.array(t_eval)
        else:
            t_points = np.linspace(t_span[0], t_span[1], 100)
        
        # Mock SDE solution
        states = np.outer(np.exp(-t_points), x0)  # (T, nx)
        
        return {
            "t": t_points,
            "x": states,  # Changed from "y", now (T, nx)
            "success": True,
            "message": "Success",
            "nfev": 100,
            "njev": 0,
            "nlu": 0,
            "status": 0,
            "nsteps": len(t_points) - 1,
            "integration_time": 0.0,
            "solver": method
        }
    
    def linearize(self, x_eq, u_eq=None):
        """Returns (A, B, G) for stochastic"""
        A = -np.eye(self.nx)
        B = np.ones((self.nx, self.nu))
        G = 0.1 * np.eye(self.nx, self.nw)
        return (A, B, G)
    
    @property
    def is_stochastic(self):
        return True


class NonlinearPendulum(ContinuousSystemBase):
    """Nonlinear pendulum for testing: dx/dt = [v, -sin(theta)/L + u]"""
    
    def __init__(self, L=1.0, g=9.81):
        self.nx = 2  # [theta, theta_dot]
        self.nu = 1  # torque
        self.ny = 2
        self.L = L
        self.g = g
    
    def __call__(self, x, u=None, t=0.0):
        theta, theta_dot = x
        u_val = u[0] if u is not None else 0.0
        return np.array([theta_dot, -self.g / self.L * np.sin(theta) + u_val])
    
    def integrate(self, x0, u=None, t_span=(0.0, 1.0), method="RK45", **kwargs):
        t_eval = kwargs.get("t_eval", None)
        
        if t_eval is not None:
            t_points = np.array(t_eval)
        else:
            t_points = np.linspace(t_span[0], t_span[1], 100)
        
        dt = t_points[1] - t_points[0]
        states = np.zeros((len(t_points), self.nx))  # (T, nx)
        states[0, :] = x0
        
        for i in range(1, len(t_points)):
            u_val = u(t_points[i-1], states[i-1, :]) if callable(u) else u
            dxdt = self(states[i-1, :], u_val, t_points[i-1])
            states[i, :] = states[i-1, :] + dt * dxdt
        
        return {
            "t": t_points,
            "x": states,  # Changed from "y", now (T, nx)
            "success": True,
            "message": "Success",
            "nfev": len(t_points),
            "njev": 0,
            "nlu": 0,
            "status": 0,
            "nsteps": len(t_points) - 1,
            "integration_time": 0.0,
            "solver": method
        }
    
    def linearize(self, x_eq, u_eq=None):
        """Linearize around equilibrium"""
        theta_eq = x_eq[0]
        A = np.array([[0, 1],
                      [-self.g / self.L * np.cos(theta_eq), 0]])
        B = np.array([[0], [1]])
        return (A, B)


# =============================================================================
# Test Suite
# =============================================================================

class TestContinuousSystemBase(unittest.TestCase):
    """Comprehensive test suite for ContinuousSystemBase abstract class."""

    def setUp(self):
        """Create test systems."""
        self.system = SimpleContinuousSystem(nx=2, nu=1)
        self.time_varying = TimeVaryingSystem(nx=2, nu=1)
        self.stochastic = StochasticSystem(nx=2, nu=1, nw=2)
        self.pendulum = NonlinearPendulum()

    # =========================================================================
    # Section 1: Abstract Class Properties and Inheritance
    # =========================================================================

    def test_cannot_instantiate_abstract_class(self):
        """ContinuousSystemBase cannot be instantiated directly."""
        with self.assertRaises(TypeError):
            system = ContinuousSystemBase()

    def test_concrete_class_instantiation(self):
        """Concrete implementation can be instantiated."""
        system = SimpleContinuousSystem()
        self.assertIsInstance(system, ContinuousSystemBase)

    def test_missing_call_method_raises_error(self):
        """Class missing __call__ cannot be instantiated."""
        class IncompleteSystem(ContinuousSystemBase):
            def integrate(self, x0, u, t_span, method="RK45", **kwargs):
                return {}
            def linearize(self, x_eq, u_eq):
                return (np.eye(2), np.eye(2, 1))
        
        with self.assertRaises(TypeError):
            system = IncompleteSystem()

    def test_missing_integrate_method_raises_error(self):
        """Class missing integrate() cannot be instantiated."""
        class IncompleteSystem(ContinuousSystemBase):
            def __call__(self, x, u=None, t=0.0):
                return x
            def linearize(self, x_eq, u_eq):
                return (np.eye(2), np.eye(2, 1))
        
        with self.assertRaises(TypeError):
            system = IncompleteSystem()

    def test_missing_linearize_method_raises_error(self):
        """Class missing linearize() cannot be instantiated."""
        class IncompleteSystem(ContinuousSystemBase):
            def __call__(self, x, u=None, t=0.0):
                return x
            def integrate(self, x0, u, t_span, method="RK45", **kwargs):
                return {}
        
        with self.assertRaises(TypeError):
            system = IncompleteSystem()

    def test_subclass_type_check(self):
        """Concrete implementation is instance and subclass of base."""
        self.assertIsInstance(self.system, ContinuousSystemBase)
        self.assertTrue(issubclass(SimpleContinuousSystem, ContinuousSystemBase))

    def test_multiple_concrete_implementations(self):
        """Multiple concrete implementations can coexist."""
        systems = [
            SimpleContinuousSystem(),
            TimeVaryingSystem(),
            StochasticSystem(),
            NonlinearPendulum()
        ]
        
        for sys in systems:
            self.assertIsInstance(sys, ContinuousSystemBase)

    # =========================================================================
    # Section 2: __call__ Method (Dynamics Evaluation) - Edge Cases
    # =========================================================================

    def test_call_with_control(self):
        """Evaluate dynamics with control input."""
        x = np.array([1.0, 2.0])
        u = np.array([0.5])

        dxdt = self.system(x, u)

        self.assertEqual(dxdt.shape, (2,))
        expected = np.array([-0.5, -1.5])
        np.testing.assert_array_almost_equal(dxdt, expected)

    def test_call_without_control(self):
        """Evaluate autonomous dynamics (u=None)."""
        x = np.array([1.0, 2.0])

        dxdt = self.system(x, u=None)

        expected = np.array([-1.0, -2.0])
        np.testing.assert_array_almost_equal(dxdt, expected)

    def test_call_with_zero_state(self):
        """Evaluate dynamics at origin."""
        x = np.zeros(2)
        u = np.zeros(1)
        
        dxdt = self.system(x, u)
        
        np.testing.assert_array_almost_equal(dxdt, np.zeros(2))

    def test_call_with_zero_control(self):
        """Evaluate dynamics with zero control."""
        x = np.array([1.0, 2.0])
        u = np.zeros(1)
        
        dxdt = self.system(x, u)
        
        # dx/dt = -x + 0 = -x
        expected = -x
        np.testing.assert_array_almost_equal(dxdt, expected)

    def test_call_returns_correct_shape(self):
        """Dynamics returns vector of same shape as state."""
        for nx in [1, 2, 5, 10]:
            sys = SimpleContinuousSystem(nx=nx, nu=1)
            x = np.random.randn(nx)
            u = np.random.randn(1)
            
            dxdt = sys(x, u)
            
            self.assertEqual(dxdt.shape, x.shape)

    def test_call_with_large_state(self):
        """Handle high-dimensional states."""
        sys = SimpleContinuousSystem(nx=100, nu=10)
        x = np.random.randn(100)
        u = np.random.randn(10)
        
        dxdt = sys(x, u)
        
        self.assertEqual(dxdt.shape, (100,))

    def test_call_time_varying_different_times(self):
        """Time-varying system produces different outputs at different times."""
        x = np.array([1.0, 1.0])
        u = np.array([0.0])
        
        dxdt_t0 = self.time_varying(x, u, t=0.0)
        dxdt_t1 = self.time_varying(x, u, t=1.0)
        dxdt_t5 = self.time_varying(x, u, t=5.0)
        
        # dx/dt = -t*x, so should vary with t
        self.assertAlmostEqual(dxdt_t0[0], 0.0)  # -0*1 + 0
        self.assertAlmostEqual(dxdt_t1[0], -1.0)  # -1*1 + 0
        self.assertAlmostEqual(dxdt_t5[0], -5.0)  # -5*1 + 0

    def test_call_batch_evaluation_2d(self):
        """Evaluate dynamics for multiple states (batched)."""
        x_batch = np.random.randn(2, 10)  # 10 states
        u_batch = np.random.randn(1, 10)  # 10 controls
        
        dxdt_batch = self.system(x_batch, u_batch)
        
        self.assertEqual(dxdt_batch.shape, (2, 10))

    def test_call_nonlinear_system(self):
        """Nonlinear dynamics evaluation."""
        x = np.array([np.pi/4, 0.0])  # 45 degrees, no velocity
        u = np.array([0.0])
        
        dxdt = self.pendulum(x, u)
        
        # dx/dt = [v, -g/L * sin(theta)]
        # = [0, -9.81 * sin(pi/4)] = [0, -6.93]
        self.assertAlmostEqual(dxdt[0], 0.0)
        self.assertAlmostEqual(dxdt[1], -9.81 * np.sin(np.pi/4), places=5)

    def test_call_with_extreme_values(self):
        """Handle extreme but valid input values."""
        x = np.array([1e6, -1e6])
        u = np.array([1e3])
        
        dxdt = self.system(x, u)
        
        # Should not overflow or raise errors
        self.assertTrue(np.all(np.isfinite(dxdt)))

    # =========================================================================
    # Section 3: integrate() Method - Comprehensive Testing
    # =========================================================================

    def test_integrate_returns_integration_result(self):
        """integrate() returns IntegrationResult with all required fields."""
        x0 = np.array([1.0, 1.0])
        result = self.system.integrate(x0, u=None, t_span=(0.0, 1.0))

        # Check type
        self.assertIsInstance(result, dict)
        
        # Check all required fields
        required_fields = ["t", "x", "success", "message", "nfev"]
        for field in required_fields:
            self.assertIn(field, result)

    def test_integrate_time_span_respected(self):
        """Integration respects specified time span."""
        x0 = np.array([1.0, 1.0])
        t_span = (0.5, 3.7)
        
        result = self.system.integrate(x0, t_span=t_span)
        
        self.assertAlmostEqual(result["t"][0], t_span[0])
        self.assertAlmostEqual(result["t"][-1], t_span[1], places=4)

    def test_integrate_backward_time(self):
        """Integration can run backward in time."""
        x0 = np.array([1.0, 1.0])
        result = self.system.integrate(x0, t_span=(1.0, 0.0))
        
        self.assertGreater(result["t"][0], result["t"][-1])

    def test_integrate_with_constant_control(self):
        """Integrate with constant control vector."""
        x0 = np.array([1.0, 1.0])
        u = np.array([0.5])
        
        result = self.system.integrate(x0, u, t_span=(0.0, 1.0))
        
        self.assertTrue(result["success"])
        self.assertEqual(result["x"].shape[1], self.system.nx)

    def test_integrate_with_time_varying_control_function(self):
        """Integrate with callable control function."""
        x0 = np.array([1.0, 1.0])
        u_func = lambda t, x: np.array([np.sin(2*np.pi*t)])
        
        result = self.system.integrate(x0, u_func, t_span=(0.0, 2.0))
        
        self.assertTrue(result["success"])

    def test_integrate_with_piecewise_control(self):
        """Integrate with piecewise constant control."""
        x0 = np.array([1.0, 1.0])
        
        def u_piecewise(t, x):
            if t < 0.5:
                return np.array([1.0])
            else:
                return np.array([-1.0])
        
        result = self.system.integrate(x0, u_piecewise, t_span=(0.0, 1.0))
        
        self.assertTrue(result["success"])

    def test_integrate_different_methods(self):
        """Integration with different method specifications."""
        x0 = np.array([1.0, 1.0])
        
        for method in ["RK45", "RK23", "Euler"]:
            result = self.system.integrate(x0, method=method)
            self.assertTrue(result["success"], f"Method {method} failed")

    def test_integrate_custom_max_step(self):
        """Integration respects max_step parameter."""
        x0 = np.array([1.0, 1.0])
        
        result = self.system.integrate(x0, max_step=0.001)
        
        # Should have more points with smaller step
        self.assertGreater(len(result["t"]), 100)

    def test_integrate_solver_diagnostics_populated(self):
        """Solver diagnostics are properly populated."""
        x0 = np.array([1.0, 1.0])
        result = self.system.integrate(x0)
        
        self.assertIsInstance(result["success"], bool)
        self.assertIsInstance(result["nfev"], int)
        self.assertGreater(result["nfev"], 0)
        self.assertIsInstance(result["message"], str)

    def test_integrate_states_shape_correct(self):
        """Integrated states have correct shape."""
        x0 = np.array([1.0, 1.0])
        result = self.system.integrate(x0)
        
        # x should be (T, nx) - time-major ordering
        self.assertEqual(result["x"].shape[0], len(result["t"]))
        self.assertEqual(result["x"].shape[1], self.system.nx)

    def test_integrate_initial_condition_preserved(self):
        """Initial condition appears in integrated trajectory."""
        x0 = np.array([2.5, -1.3])
        result = self.system.integrate(x0)
        
        # First row is initial condition (time-major: (T, nx))
        np.testing.assert_array_almost_equal(result["x"][0, :], x0)

    def test_integrate_autonomous_decay(self):
        """Autonomous stable system decays over time."""
        x0 = np.array([1.0, 1.0])
        result = self.system.integrate(x0, u=None, t_span=(0.0, 5.0))
        
        # System is dx/dt = -x, should decay exponentially
        final_state = result["x"][-1, :]  # Last row (time-major)
        self.assertLess(np.linalg.norm(final_state), 0.1 * np.linalg.norm(x0))

    def test_integrate_different_initial_conditions(self):
        """Integration works for various initial conditions."""
        initial_conditions = [
            np.array([0.0, 0.0]),
            np.array([1.0, 1.0]),
            np.array([-5.0, 3.0]),
            np.array([1e-6, 1e-6])
        ]
        
        for x0 in initial_conditions:
            result = self.system.integrate(x0)
            self.assertTrue(result["success"])

    # =========================================================================
    # Section 4: simulate() Method - Comprehensive Testing
    # =========================================================================

    def test_simulate_returns_simulation_result(self):
        """simulate() returns SimulationResult with required fields."""
        x0 = np.array([1.0, 1.0])
        result = self.system.simulate(x0, t_span=(0.0, 1.0), dt=0.1)

        self.assertIsInstance(result, dict)
        self.assertIn("time", result)
        self.assertIn("states", result)
        self.assertIn("metadata", result)

    def test_simulate_regular_time_grid(self):
        """simulate() produces regular time grid."""
        x0 = np.array([1.0, 1.0])
        dt = 0.1
        result = self.system.simulate(x0, t_span=(0.0, 1.0), dt=dt)

        time = result["time"]
        time_diff = np.diff(time)
        np.testing.assert_array_almost_equal(time_diff, dt * np.ones(len(time_diff)))

    def test_simulate_custom_dt(self):
        """simulate() respects custom dt parameter."""
        x0 = np.array([1.0, 1.0])
        
        for dt in [0.01, 0.05, 0.1, 0.2]:
            result = self.system.simulate(x0, t_span=(0.0, 1.0), dt=dt)
            expected_points = int(1.0 / dt) + 1
            self.assertEqual(len(result["time"]), expected_points)

    def test_simulate_without_controller(self):
        """simulate() works with controller=None (open-loop)."""
        x0 = np.array([1.0, 1.0])
        result = self.system.simulate(x0, controller=None, t_span=(0.0, 1.0), dt=0.1)
        
        self.assertIn("states", result)

    def test_simulate_no_solver_diagnostics_in_top_level(self):
        """simulate() hides solver internals from top level."""
        x0 = np.array([1.0, 1.0])
        result = self.system.simulate(x0, t_span=(0.0, 1.0), dt=0.1)

        # Solver details hidden
        self.assertNotIn("nfev", result)
        self.assertNotIn("njev", result)
        self.assertNotIn("nlu", result)

    def test_simulate_metadata_contains_method(self):
        """simulate() metadata includes integration method."""
        x0 = np.array([1.0, 1.0])
        result = self.system.simulate(x0, t_span=(0.0, 1.0), dt=0.1, method="RK23")
        
        self.assertIn("method", result["metadata"])

    def test_simulate_metadata_contains_dt(self):
        """simulate() metadata includes time step."""
        x0 = np.array([1.0, 1.0])
        dt = 0.05
        result = self.system.simulate(x0, t_span=(0.0, 1.0), dt=dt)
        
        self.assertIn("dt", result["metadata"])
        self.assertEqual(result["metadata"]["dt"], dt)

    def test_simulate_time_span_different_start(self):
        """simulate() works with non-zero start time."""
        x0 = np.array([1.0, 1.0])
        result = self.system.simulate(x0, t_span=(2.0, 5.0), dt=0.1)
        
        self.assertAlmostEqual(result["time"][0], 2.0)
        self.assertAlmostEqual(result["time"][-1], 5.0)

    def test_simulate_states_shape(self):
        """simulate() states have correct shape."""
        x0 = np.array([1.0, 1.0])
        result = self.system.simulate(x0, t_span=(0.0, 1.0), dt=0.1)
        
        n_points = len(result["time"])
        # FIXED: Time-major convention (T, nx) instead of (nx, T)
        self.assertEqual(result["states"].shape, (n_points, self.system.nx))

    # =========================================================================
    # Section 5: linearize() Method - Comprehensive Testing
    # =========================================================================

    def test_linearize_at_origin(self):
        """Linearize at origin equilibrium."""
        x_eq = np.zeros(2)
        u_eq = np.zeros(1)

        lin = self.system.linearize(x_eq, u_eq)

        self.assertIsInstance(lin, tuple)
        self.assertEqual(len(lin), 2)
        
        A, B = lin
        self.assertEqual(A.shape, (2, 2))
        self.assertEqual(B.shape, (2, 1))

    def test_linearize_known_system_matrices(self):
        """Linearization matrices match analytical result."""
        x_eq = np.zeros(2)
        u_eq = np.zeros(1)

        A, B = self.system.linearize(x_eq, u_eq)

        # For dx/dt = -x + u: A = -I, B = I
        expected_A = -np.eye(2)
        expected_B = np.ones((2, 1))

        np.testing.assert_array_almost_equal(A, expected_A)
        np.testing.assert_array_almost_equal(B, expected_B)

    def test_linearize_at_nonzero_equilibrium(self):
        """Linearize at non-origin equilibrium."""
        x_eq = np.array([1.0, 2.0])
        u_eq = np.array([0.5])
        
        lin = self.system.linearize(x_eq, u_eq)
        
        self.assertIsInstance(lin, tuple)
        A, B = lin
        self.assertEqual(A.shape, (self.system.nx, self.system.nx))

    def test_linearize_without_control_equilibrium(self):
        """Linearize with u_eq=None."""
        x_eq = np.zeros(2)
        
        lin = self.system.linearize(x_eq, u_eq=None)
        
        A, B = lin
        self.assertIsNotNone(A)
        self.assertIsNotNone(B)

    def test_linearize_stochastic_system_returns_three_matrices(self):
        """Stochastic system linearization returns (A, B, G)."""
        x_eq = np.zeros(2)
        u_eq = np.zeros(1)
        
        lin = self.stochastic.linearize(x_eq, u_eq)
        
        # Should return 3-tuple
        self.assertIsInstance(lin, tuple)
        self.assertEqual(len(lin), 3)
        
        A, B, G = lin
        self.assertEqual(A.shape, (2, 2))
        self.assertEqual(B.shape, (2, 1))
        self.assertEqual(G.shape, (2, self.stochastic.nw))

    def test_linearize_nonlinear_pendulum_at_bottom(self):
        """Linearize pendulum at bottom equilibrium."""
        x_eq = np.array([0.0, 0.0])  # Bottom, no velocity
        u_eq = np.array([0.0])
        
        A, B = self.pendulum.linearize(x_eq, u_eq)
        
        # At theta=0: A = [[0, 1], [-g/L, 0]]
        expected_A = np.array([[0, 1],
                              [-self.pendulum.g / self.pendulum.L, 0]])
        expected_B = np.array([[0], [1]])
        
        np.testing.assert_array_almost_equal(A, expected_A)
        np.testing.assert_array_almost_equal(B, expected_B)

    def test_linearize_nonlinear_pendulum_at_angle(self):
        """Linearize pendulum at non-zero angle."""
        x_eq = np.array([np.pi/4, 0.0])
        u_eq = np.array([0.0])
        
        A, B = self.pendulum.linearize(x_eq, u_eq)
        
        # Should linearize around pi/4
        expected_A22 = -self.pendulum.g / self.pendulum.L * np.cos(np.pi/4)
        self.assertAlmostEqual(A[1, 0], expected_A22, places=5)

    def test_linearize_dimensions_match_system_dimensions(self):
        """Linearization dimensions match system dimensions."""
        for nx, nu in [(1, 1), (2, 1), (3, 2), (5, 3)]:
            sys = SimpleContinuousSystem(nx=nx, nu=nu)
            x_eq = np.zeros(nx)
            u_eq = np.zeros(nu)
            
            A, B = sys.linearize(x_eq, u_eq)
            
            self.assertEqual(A.shape, (nx, nx))
            self.assertEqual(B.shape, (nx, nu))

    def test_linearize_stability_check(self):
        """Use linearization to check stability."""
        x_eq = np.zeros(2)
        u_eq = np.zeros(1)
        
        A, B = self.system.linearize(x_eq, u_eq)
        
        eigenvalues = np.linalg.eigvals(A)
        is_stable = np.all(np.real(eigenvalues) < 0)
        
        # System dx/dt = -x is stable
        self.assertTrue(is_stable)

    # =========================================================================
    # Section 6: Properties - Comprehensive Testing
    # =========================================================================

    def test_is_continuous_returns_true(self):
        """is_continuous property returns True for continuous systems."""
        self.assertTrue(self.system.is_continuous)
        self.assertTrue(self.time_varying.is_continuous)
        self.assertTrue(self.stochastic.is_continuous)

    def test_is_discrete_returns_false(self):
        """is_discrete property returns False for continuous systems."""
        self.assertFalse(self.system.is_discrete)
        self.assertFalse(self.time_varying.is_discrete)
        self.assertFalse(self.stochastic.is_discrete)

    def test_is_stochastic_default_false(self):
        """is_stochastic returns False by default."""
        self.assertFalse(self.system.is_stochastic)
        self.assertFalse(self.time_varying.is_stochastic)

    def test_is_stochastic_true_when_overridden(self):
        """is_stochastic returns True when overridden."""
        self.assertTrue(self.stochastic.is_stochastic)

    def test_is_time_varying_default_false(self):
        """is_time_varying returns False by default."""
        self.assertFalse(self.system.is_time_varying)
        self.assertFalse(self.stochastic.is_time_varying)

    def test_is_time_varying_true_when_overridden(self):
        """is_time_varying returns True when overridden."""
        self.assertTrue(self.time_varying.is_time_varying)

    def test_properties_are_read_only(self):
        """System properties should be read-only (if possible)."""
        # This tests the property mechanism works
        self.assertIsInstance(self.system.is_continuous, bool)
        self.assertIsInstance(self.system.is_discrete, bool)
        self.assertIsInstance(self.system.is_stochastic, bool)
        self.assertIsInstance(self.system.is_time_varying, bool)

    # =========================================================================
    # Section 7: __repr__ and String Representation
    # =========================================================================

    def test_repr_contains_class_name(self):
        """String representation contains class name."""
        repr_str = repr(self.system)
        self.assertIn("SimpleContinuousSystem", repr_str)

    def test_repr_contains_dimensions(self):
        """String representation contains system dimensions."""
        repr_str = repr(self.system)
        self.assertIn("nx=2", repr_str)
        self.assertIn("nu=1", repr_str)

    def test_repr_different_systems_different_names(self):
        """Different systems have different string representations."""
        repr1 = repr(self.system)
        repr2 = repr(self.time_varying)
        repr3 = repr(self.stochastic)
        
        self.assertNotEqual(repr1, repr2)
        self.assertNotEqual(repr2, repr3)

    def test_repr_is_informative(self):
        """String representation is informative."""
        repr_str = repr(self.pendulum)
        
        # Should contain useful info
        self.assertIsInstance(repr_str, str)
        self.assertGreater(len(repr_str), 10)

    # =========================================================================
    # Section 8: Polymorphism and Generic Programming
    # =========================================================================

    def test_polymorphic_function_call(self):
        """Generic function works with any ContinuousSystemBase."""
        def evaluate_at_origin(sys: ContinuousSystemBase):
            x = np.zeros(sys.nx)
            u = np.zeros(sys.nu)
            return sys(x, u, t=0.0)
        
        for system in [self.system, self.time_varying, self.stochastic]:
            dxdt = evaluate_at_origin(system)
            self.assertEqual(dxdt.shape, (system.nx,))

    def test_polymorphic_linearization(self):
        """Generic linearization works for all systems."""
        def check_stability(sys: ContinuousSystemBase) -> bool:
            x_eq = np.zeros(sys.nx)
            u_eq = np.zeros(sys.nu)
            lin = sys.linearize(x_eq, u_eq)
            A = lin[0]  # First element is always A
            eigenvalues = np.linalg.eigvals(A)
            return np.all(np.real(eigenvalues) < 0)
        
        # Simple and stochastic systems should be stable at origin
        self.assertTrue(check_stability(self.system))
        self.assertTrue(check_stability(self.stochastic))
        # Time-varying system at t=0 has A=0, so eigenvalue=0 (not stable, not unstable)
        # Skip time-varying for this test

    def test_polymorphic_integration(self):
        """Generic integration works for all systems."""
        def simulate_from_origin(sys: ContinuousSystemBase, T=1.0):
            x0 = np.zeros(sys.nx)
            return sys.integrate(x0, t_span=(0.0, T))
        
        for system in [self.system, self.time_varying, self.stochastic]:
            result = simulate_from_origin(system)
            self.assertTrue(result["success"])

    def test_list_of_systems(self):
        """List of different systems can be processed uniformly."""
        systems = [self.system, self.time_varying, self.stochastic, self.pendulum]
        
        # All should be instances of base
        for sys in systems:
            self.assertIsInstance(sys, ContinuousSystemBase)
        
        # All should be callable
        for sys in systems:
            x = np.zeros(sys.nx)
            u = np.zeros(sys.nu)
            dxdt = sys(x, u)
            self.assertEqual(dxdt.shape, (sys.nx,))

    # =========================================================================
    # Section 9: Error Handling and Edge Cases
    # =========================================================================

    def test_call_with_nan_state(self):
        """Calling with NaN state should not crash (may return NaN)."""
        x = np.array([np.nan, 1.0])
        u = np.array([0.5])
        
        dxdt = self.system(x, u)
        
        # Should return something (even if NaN)
        self.assertEqual(dxdt.shape, (2,))

    def test_call_with_inf_state(self):
        """Calling with infinite state should not crash."""
        x = np.array([np.inf, 1.0])
        u = np.array([0.5])
        
        dxdt = self.system(x, u)
        
        self.assertEqual(dxdt.shape, (2,))

    def test_integrate_with_negative_time_span(self):
        """Integration backwards in time should work."""
        x0 = np.array([1.0, 1.0])
        result = self.system.integrate(x0, t_span=(1.0, 0.0))
        
        self.assertTrue(result["success"])
        self.assertGreater(result["t"][0], result["t"][-1])

    def test_integrate_zero_duration(self):
        """Integration with t_start = t_end should handle gracefully."""
        x0 = np.array([1.0, 1.0])
        result = self.system.integrate(x0, t_span=(1.0, 1.0))
        
        # Should return initial state only
        self.assertEqual(len(result["t"]), 1)

    def test_simulate_very_small_dt(self):
        """simulate() with very small dt should work."""
        x0 = np.array([1.0, 1.0])
        result = self.system.simulate(x0, t_span=(0.0, 0.1), dt=0.001)
        
        self.assertEqual(len(result["time"]), 101)

    def test_linearize_at_different_dimensions(self):
        """Linearization works for systems of different dimensions."""
        for nx, nu in [(1, 1), (10, 5), (20, 10)]:
            sys = SimpleContinuousSystem(nx=nx, nu=nu)
            x_eq = np.zeros(nx)
            u_eq = np.zeros(nu)
            
            A, B = sys.linearize(x_eq, u_eq)
            
            self.assertEqual(A.shape, (nx, nx))
            self.assertEqual(B.shape, (nx, nu))


if __name__ == "__main__":
    unittest.main()