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
from collections.abc import Sequence
from typing import Callable, Optional, Union

import numpy as np

from src.systems.base.core.discrete_system_base import DiscreteSystemBase
from src.types.core import ControlVector, StateVector
from src.types.linearization import DiscreteLinearization

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


class SimpleDiscreteSystem(DiscreteSystemBase):
    """Concrete implementation: x[k+1] = 0.9*x[k] + 0.1*u[k]"""

    def __init__(self, nx=2, nu=1, dt_val=0.1):
        self.nx = nx
        self.nu = nu
        self.ny = nx
        self._dt = dt_val
        self.Ad = 0.9 * np.eye(nx)
        self.Bd = 0.1 * np.ones((nx, nu))

    @property
    def dt(self) -> float:
        return self._dt

    def step(self, x: StateVector, u: Optional[ControlVector] = None, k: int = 0) -> StateVector:
        """x[k+1] = Ad*x[k] + Bd*u[k]"""
        if u is None:
            u = np.zeros(self.nu)

        if x.ndim == 2:
            u = u if u.ndim == 2 else u.reshape(-1, 1)
            return self.Ad @ x + self.Bd @ u

        return self.Ad @ x + self.Bd @ u.flatten()

    def simulate(
        self,
        x0: StateVector,
        u_sequence: Optional[
            Union[ControlVector, Sequence[ControlVector], Callable[[int], ControlVector]]
        ] = None,
        n_steps: int = 100,
        **kwargs,
    ) -> dict:
        """Simulate for n_steps - returns plain dict with TIME-MAJOR order."""
        states = np.zeros((n_steps + 1, self.nx))  # TIME-MAJOR: (n_steps+1, nx)
        states[0, :] = x0

        controls = [] if u_sequence is not None else None

        for k in range(n_steps):
            if u_sequence is None:
                u = None
            elif callable(u_sequence):
                u = u_sequence(k)
            elif isinstance(u_sequence, np.ndarray) and u_sequence.ndim == 1:
                u = u_sequence
            else:
                u = u_sequence[k]

            if controls is not None and u is not None:
                controls.append(u)

            states[k + 1, :] = self.step(states[k, :], u, k)

        controls_array = np.array(controls) if controls else None  # (n_steps, nu)

        return {
            "states": states,  # Shape: (n_steps+1, nx)
            "controls": controls_array,  # Shape: (n_steps, nu) or None
            "time_steps": np.arange(n_steps + 1),
            "dt": self.dt,
            "metadata": kwargs,
        }

    def linearize(
        self, x_eq: StateVector, u_eq: Optional[ControlVector] = None,
    ) -> DiscreteLinearization:
        """Already linear, return (Ad, Bd) tuple."""
        return (self.Ad, self.Bd)


class TimeVaryingDiscreteSystem(DiscreteSystemBase):
    """Time-varying: x[k+1] = (0.9 - 0.01*k)*x[k] + u[k]"""

    def __init__(self, nx=2, nu=1, dt_val=0.1):
        self.nx = nx
        self.nu = nu
        self.ny = nx
        self._dt = dt_val

    @property
    def dt(self) -> float:
        return self._dt

    def step(self, x, u=None, k=0):
        u = u if u is not None else np.zeros(self.nu)
        alpha = 0.9 - 0.01 * k
        return alpha * x + u

    def simulate(self, x0, u_sequence=None, n_steps=100, **kwargs):
        states = np.zeros((n_steps + 1, self.nx))  # TIME-MAJOR
        states[0, :] = x0
        controls = []

        for k in range(n_steps):
            u = self._get_control(u_sequence, k)
            if u is not None:
                controls.append(u)
            states[k + 1, :] = self.step(states[k, :], u, k)

        return {
            "states": states,  # (n_steps+1, nx)
            "controls": np.array(controls) if controls else None,  # (n_steps, nu)
            "time_steps": np.arange(n_steps + 1),
            "dt": self.dt,
            "metadata": kwargs,
        }

    def linearize(self, x_eq, u_eq=None):
        # Linearization at k=0
        Ad = 0.9 * np.eye(self.nx)
        Bd = np.ones((self.nx, self.nu))
        return (Ad, Bd)

    @property
    def is_time_varying(self):
        return True

    def _get_control(self, u_sequence, k):
        if u_sequence is None:
            return None
        if callable(u_sequence):
            return u_sequence(k)
        if isinstance(u_sequence, np.ndarray) and u_sequence.ndim == 1:
            return u_sequence
        return u_sequence[k]


class UnstableDiscreteSystem(DiscreteSystemBase):
    """Unstable system: x[k+1] = 1.1*x[k] + u[k]"""

    def __init__(self, nx=2, nu=1, dt_val=0.1):
        self.nx = nx
        self.nu = nu
        self.ny = nx
        self._dt = dt_val
        self.Ad = 1.1 * np.eye(nx)
        self.Bd = np.ones((nx, nu))

    @property
    def dt(self) -> float:
        return self._dt

    def step(self, x, u=None, k=0):
        u = u if u is not None else np.zeros(self.nu)
        return self.Ad @ x + self.Bd @ u.flatten()

    def simulate(self, x0, u_sequence=None, n_steps=100, **kwargs):
        states = np.zeros((n_steps + 1, self.nx))  # TIME-MAJOR
        states[0, :] = x0
        controls = []

        for k in range(n_steps):
            u = self._get_control(u_sequence, k)
            if u is not None:
                controls.append(u)
            states[k + 1, :] = self.step(states[k, :], u, k)

        return {
            "states": states,
            "controls": np.array(controls) if controls else None,
            "time_steps": np.arange(n_steps + 1),
            "dt": self.dt,
            "metadata": kwargs,
        }

    def linearize(self, x_eq, u_eq=None):
        return (self.Ad, self.Bd)

    def _get_control(self, u_sequence, k):
        if u_sequence is None:
            return None
        if callable(u_sequence):
            return u_sequence(k)
        if isinstance(u_sequence, np.ndarray) and u_sequence.ndim == 1:
            return u_sequence
        return u_sequence[k]


class NonlinearDiscreteSystem(DiscreteSystemBase):
    """Nonlinear: x[k+1] = tanh(x[k]) + u[k]"""

    def __init__(self, nx=2, nu=1, dt_val=0.1):
        self.nx = nx
        self.nu = nu
        self.ny = nx
        self._dt = dt_val

    @property
    def dt(self) -> float:
        return self._dt

    def step(self, x, u=None, k=0):
        u = u if u is not None else np.zeros(self.nu)
        return np.tanh(x) + u

    def simulate(self, x0, u_sequence=None, n_steps=100, **kwargs):
        states = np.zeros((n_steps + 1, self.nx))  # TIME-MAJOR
        states[0, :] = x0
        controls = []

        for k in range(n_steps):
            u = self._get_control(u_sequence, k)
            if u is not None:
                controls.append(u)
            states[k + 1, :] = self.step(states[k, :], u, k)

        return {
            "states": states,
            "controls": np.array(controls) if controls else None,
            "time_steps": np.arange(n_steps + 1),
            "dt": self.dt,
            "metadata": kwargs,
        }

    def linearize(self, x_eq, u_eq=None):
        # Ad = diag(sech^2(x_eq))
        Ad = np.diag(1.0 / np.cosh(x_eq) ** 2)
        Bd = np.ones((self.nx, self.nu))
        return (Ad, Bd)

    def _get_control(self, u_sequence, k):
        if u_sequence is None:
            return None
        if callable(u_sequence):
            return u_sequence(k)
        if isinstance(u_sequence, np.ndarray) and u_sequence.ndim == 1:
            return u_sequence
        return u_sequence[k]


# =============================================================================
# Test Suite
# =============================================================================


class TestDiscreteSystemBase(unittest.TestCase):
    """Comprehensive test suite for DiscreteSystemBase abstract class."""

    def setUp(self):
        """Create test systems."""
        self.system = SimpleDiscreteSystem(nx=2, nu=1, dt_val=0.1)
        self.time_varying = TimeVaryingDiscreteSystem(nx=2, nu=1, dt_val=0.1)
        self.unstable = UnstableDiscreteSystem(nx=2, nu=1, dt_val=0.1)
        self.nonlinear = NonlinearDiscreteSystem(nx=2, nu=1, dt_val=0.1)

    # =========================================================================
    # Section 1: Abstract Class Properties and Inheritance
    # =========================================================================

    def test_cannot_instantiate_abstract_class(self):
        """DiscreteSystemBase cannot be instantiated directly."""
        with self.assertRaises(TypeError):
            system = DiscreteSystemBase()

    def test_concrete_class_instantiation(self):
        """Concrete implementation can be instantiated."""
        system = SimpleDiscreteSystem()
        self.assertIsInstance(system, DiscreteSystemBase)

    def test_missing_dt_property_raises_error(self):
        """Class missing dt property cannot be instantiated."""

        class IncompleteSystem(DiscreteSystemBase):
            def step(self, x, u=None, k=0):
                return x

            def simulate(self, x0, u_sequence, n_steps):
                return {}

            def linearize(self, x_eq, u_eq):
                return (np.eye(2), np.eye(2, 1))

        with self.assertRaises(TypeError):
            system = IncompleteSystem()

    def test_missing_step_method_raises_error(self):
        """Class missing step() cannot be instantiated."""

        class IncompleteSystem(DiscreteSystemBase):
            @property
            def dt(self):
                return 0.1

            def simulate(self, x0, u_sequence, n_steps):
                return {}

            def linearize(self, x_eq, u_eq):
                return (np.eye(2), np.eye(2, 1))

        with self.assertRaises(TypeError):
            system = IncompleteSystem()

    def test_missing_simulate_method_raises_error(self):
        """Class missing simulate() cannot be instantiated."""

        class IncompleteSystem(DiscreteSystemBase):
            @property
            def dt(self):
                return 0.1

            def step(self, x, u=None, k=0):
                return x

            def linearize(self, x_eq, u_eq):
                return (np.eye(2), np.eye(2, 1))

        with self.assertRaises(TypeError):
            system = IncompleteSystem()

    def test_missing_linearize_method_raises_error(self):
        """Class missing linearize() cannot be instantiated."""

        class IncompleteSystem(DiscreteSystemBase):
            @property
            def dt(self):
                return 0.1

            def step(self, x, u=None, k=0):
                return x

            def simulate(self, x0, u_sequence, n_steps):
                return {}

        with self.assertRaises(TypeError):
            system = IncompleteSystem()

    def test_subclass_type_check(self):
        """Concrete implementation is instance and subclass of base."""
        self.assertIsInstance(self.system, DiscreteSystemBase)
        self.assertTrue(issubclass(SimpleDiscreteSystem, DiscreteSystemBase))

    def test_multiple_concrete_implementations(self):
        """Multiple concrete implementations can coexist."""
        systems = [
            SimpleDiscreteSystem(),
            TimeVaryingDiscreteSystem(),
            UnstableDiscreteSystem(),
            NonlinearDiscreteSystem(),
        ]

        for sys in systems:
            self.assertIsInstance(sys, DiscreteSystemBase)

    # =========================================================================
    # Section 2: dt Property - Comprehensive Testing
    # =========================================================================

    def test_dt_property_exists(self):
        """dt property is accessible."""
        self.assertEqual(self.system.dt, 0.1)

    def test_dt_property_positive(self):
        """dt property should be positive."""
        self.assertGreater(self.system.dt, 0)

    def test_dt_property_different_values(self):
        """dt property can have different values."""
        for dt_val in [0.01, 0.05, 0.1, 0.5, 1.0]:
            sys = SimpleDiscreteSystem(dt_val=dt_val)
            self.assertEqual(sys.dt, dt_val)

    def test_dt_property_is_float(self):
        """dt property returns a float."""
        self.assertIsInstance(self.system.dt, float)

    def test_sampling_frequency_calculation(self):
        """Sampling frequency is correctly computed from dt."""
        for dt_val in [0.01, 0.1, 1.0]:
            sys = SimpleDiscreteSystem(dt_val=dt_val)
            expected_freq = 1.0 / dt_val
            self.assertAlmostEqual(sys.sampling_frequency, expected_freq)

    def test_sampling_frequency_property_exists(self):
        """sampling_frequency property is accessible."""
        freq = self.system.sampling_frequency
        self.assertIsInstance(freq, float)
        self.assertGreater(freq, 0)

    # =========================================================================
    # Section 3: step() Method - Comprehensive Testing
    # =========================================================================

    def test_step_with_control(self):
        """Step forward with control input."""
        x = np.array([1.0, 2.0])
        u = np.array([0.5])

        x_next = self.system.step(x, u)

        self.assertEqual(x_next.shape, (2,))
        expected = np.array([0.95, 1.85])
        np.testing.assert_array_almost_equal(x_next, expected)

    def test_step_without_control(self):
        """Step without control (u=None)."""
        x = np.array([1.0, 2.0])

        x_next = self.system.step(x, u=None)

        expected = 0.9 * x  # No control, just dynamics
        np.testing.assert_array_almost_equal(x_next, expected)

    def test_step_with_zero_control(self):
        """Step with explicit zero control."""
        x = np.array([1.0, 2.0])
        u = np.zeros(1)

        x_next = self.system.step(x, u)

        expected = 0.9 * x
        np.testing.assert_array_almost_equal(x_next, expected)

    def test_step_from_zero_state(self):
        """Step from zero state."""
        x = np.zeros(2)
        u = np.array([0.5])

        x_next = self.system.step(x, u)

        # x[k+1] = 0.9*0 + 0.1*0.5 = 0.05
        expected = np.array([0.05, 0.05])
        np.testing.assert_array_almost_equal(x_next, expected)

    def test_step_returns_correct_shape(self):
        """Step returns state with correct shape."""
        x = np.array([1.0, 2.0])
        u = np.array([0.5])

        x_next = self.system.step(x, u)

        self.assertEqual(x_next.shape, x.shape)

    def test_step_with_different_dimensions(self):
        """Step works for systems of different dimensions."""
        for nx, nu in [(1, 1), (3, 1), (5, 2)]:
            sys = SimpleDiscreteSystem(nx=nx, nu=nu)
            x = np.random.randn(nx)
            u = np.random.randn(nu)

            x_next = sys.step(x, u)

            self.assertEqual(x_next.shape, (nx,))

    def test_step_with_time_index(self):
        """Step accepts time index parameter."""
        x = np.array([1.0, 2.0])
        u = np.array([0.5])

        # For time-invariant system, k shouldn't matter
        x_next_k0 = self.system.step(x, u, k=0)
        x_next_k5 = self.system.step(x, u, k=5)

        np.testing.assert_array_almost_equal(x_next_k0, x_next_k5)

    def test_step_time_varying_depends_on_k(self):
        """Time-varying system step depends on k."""
        x = np.array([1.0, 1.0])
        u = np.zeros(1)

        x_next_k0 = self.time_varying.step(x, u, k=0)
        x_next_k10 = self.time_varying.step(x, u, k=10)

        # Should be different due to time-varying dynamics
        self.assertFalse(np.allclose(x_next_k0, x_next_k10))

    def test_step_nonlinear_system(self):
        """Nonlinear system step evaluation."""
        x = np.array([0.5, -0.3])
        u = np.array([0.1])

        x_next = self.nonlinear.step(x, u)

        # x[k+1] = tanh(x) + u
        expected = np.tanh(x) + u
        np.testing.assert_array_almost_equal(x_next, expected)

    def test_step_multiple_times(self):
        """Multiple step calls propagate state correctly."""
        x = np.array([1.0, 1.0])
        u = np.array([0.0])

        # Step 3 times
        x1 = self.system.step(x, u)
        x2 = self.system.step(x1, u)
        x3 = self.system.step(x2, u)

        # Should decay exponentially
        self.assertLess(np.linalg.norm(x3), np.linalg.norm(x))

    # =========================================================================
    # Section 4: simulate() Method - Comprehensive Testing
    # =========================================================================

    def test_simulate_basic(self):
        """Basic simulation test."""
        x0 = np.array([1.0, 1.0])
        result = self.system.simulate(x0, u_sequence=None, n_steps=10)

        self.assertIsInstance(result, dict)
        self.assertIn("states", result)
        self.assertIn("time_steps", result)

    def test_simulate_result_structure(self):
        """Simulation result has all required fields."""
        x0 = np.array([1.0, 1.0])
        result = self.system.simulate(x0, n_steps=10)

        required_fields = ["states", "controls", "time_steps", "dt", "metadata"]
        for field in required_fields:
            self.assertIn(field, result)

    def test_simulate_states_shape(self):
        """Simulation states have correct shape (TIME-MAJOR)."""
        x0 = np.array([1.0, 1.0])
        n_steps = 10
        result = self.system.simulate(x0, n_steps=n_steps)

        # States should be (n_steps+1, nx) - TIME-MAJOR includes initial state
        self.assertEqual(result["states"].shape, (n_steps + 1, self.system.nx))

    def test_simulate_includes_initial_state(self):
        """Simulation result includes initial state."""
        x0 = np.array([2.5, -1.3])
        result = self.system.simulate(x0, n_steps=5)

        np.testing.assert_array_equal(result["states"][0, :], x0)

    def test_simulate_time_steps_correct(self):
        """Time steps array is correct."""
        x0 = np.array([1.0, 1.0])
        n_steps = 20
        result = self.system.simulate(x0, n_steps=n_steps)

        expected_steps = np.arange(n_steps + 1)
        np.testing.assert_array_equal(result["time_steps"], expected_steps)

    def test_simulate_dt_stored(self):
        """Simulation result stores dt."""
        x0 = np.array([1.0, 1.0])
        result = self.system.simulate(x0, n_steps=10)

        self.assertEqual(result["dt"], self.system.dt)

    def test_simulate_with_constant_control(self):
        """Simulate with constant control."""
        x0 = np.array([1.0, 1.0])
        u = np.array([0.5])

        result = self.system.simulate(x0, u, n_steps=10)

        self.assertIsNotNone(result["controls"])
        # Controls should be (n_steps, nu)
        self.assertEqual(result["controls"].shape, (10, self.system.nu))

    def test_simulate_with_control_sequence(self):
        """Simulate with pre-computed control sequence."""
        x0 = np.array([1.0, 1.0])
        u_seq = [np.array([0.1 * k]) for k in range(10)]

        result = self.system.simulate(x0, u_seq, n_steps=10)

        self.assertIsNotNone(result["controls"])
        self.assertEqual(result["controls"].shape, (10, self.system.nu))

    def test_simulate_with_control_function(self):
        """Simulate with control as function of time step."""
        x0 = np.array([1.0, 1.0])
        u_func = lambda k: np.array([0.5 * np.sin(k * 0.1)])

        result = self.system.simulate(x0, u_func, n_steps=20)

        self.assertTrue("controls" in result)
        self.assertEqual(result["controls"].shape, (20, self.system.nu))

    def test_simulate_autonomous(self):
        """Simulate autonomous system (no control)."""
        x0 = np.array([1.0, 1.0])
        result = self.system.simulate(x0, u_sequence=None, n_steps=20)

        # Stable system should decay
        final_state = result["states"][-1, :]
        self.assertLess(np.linalg.norm(final_state), np.linalg.norm(x0))

    def test_simulate_zero_steps(self):
        """Simulate with n_steps=0 returns only initial state."""
        x0 = np.array([1.0, 1.0])
        result = self.system.simulate(x0, n_steps=0)

        self.assertEqual(result["states"].shape, (1, self.system.nx))
        np.testing.assert_array_equal(result["states"][0, :], x0)

    def test_simulate_one_step(self):
        """Simulate with n_steps=1 works correctly."""
        x0 = np.array([1.0, 1.0])
        u = np.array([0.5])

        result = self.system.simulate(x0, u, n_steps=1)

        # Should have 2 states: x[0] and x[1]
        self.assertEqual(result["states"].shape, (2, self.system.nx))

        # x[1] should match manual step
        x1_manual = self.system.step(x0, u, k=0)
        np.testing.assert_array_almost_equal(result["states"][1, :], x1_manual)

    def test_simulate_many_steps(self):
        """Simulate with many steps."""
        x0 = np.array([1.0, 1.0])
        n_steps = 1000

        result = self.system.simulate(x0, n_steps=n_steps)

        self.assertEqual(result["states"].shape, (n_steps + 1, self.system.nx))

    def test_simulate_metadata_stores_kwargs(self):
        """Simulation metadata stores additional kwargs."""
        x0 = np.array([1.0, 1.0])
        result = self.system.simulate(x0, n_steps=10, custom_param="test")

        self.assertIn("custom_param", result["metadata"])
        self.assertEqual(result["metadata"]["custom_param"], "test")

    def test_simulate_different_dimensions(self):
        """Simulate works for different system dimensions."""
        for nx, nu in [(1, 1), (5, 2), (10, 5)]:
            sys = SimpleDiscreteSystem(nx=nx, nu=nu)
            x0 = np.random.randn(nx)
            u = np.random.randn(nu)

            result = sys.simulate(x0, u, n_steps=10)

            self.assertEqual(result["states"].shape, (11, nx))
            self.assertEqual(result["controls"].shape, (10, nu))

    def test_simulate_unstable_system_grows(self):
        """Unstable system simulation shows growth."""
        x0 = np.array([1.0, 1.0])
        result = self.unstable.simulate(x0, u_sequence=None, n_steps=20)

        # Unstable system should grow
        final_state = result["states"][-1, :]
        self.assertGreater(np.linalg.norm(final_state), np.linalg.norm(x0))

    # =========================================================================
    # Section 5: rollout() Method - Comprehensive Testing
    # =========================================================================

    def test_rollout_without_policy(self):
        """rollout() without policy is open-loop."""
        x0 = np.array([1.0, 1.0])
        result = self.system.rollout(x0, policy=None, n_steps=10)

        self.assertIsInstance(result, dict)
        self.assertIn("states", result)
        # Should have time-major shape
        self.assertEqual(result["states"].shape, (11, self.system.nx))

    def test_rollout_with_state_feedback(self):
        """rollout() with state feedback policy."""
        x0 = np.array([1.0, 1.0])
        K = np.array([[-0.5, -0.5]])

        def policy(x, k):
            return -K @ x

        result = self.system.rollout(x0, policy, n_steps=20)

        self.assertTrue("controls" in result)
        self.assertIsNotNone(result["controls"])
        # Controls should be time-major too
        self.assertEqual(result["controls"].shape, (20, self.system.nu))

    def test_rollout_closed_loop_flag(self):
        """rollout() sets closed_loop flag in metadata."""
        x0 = np.array([1.0, 1.0])

        def policy(x, k):
            return np.array([0.0])

        result = self.system.rollout(x0, policy, n_steps=10)

        self.assertIn("closed_loop", result["metadata"])
        self.assertTrue(result["metadata"]["closed_loop"])

    def test_rollout_time_varying_policy(self):
        """rollout() with time-varying policy."""
        x0 = np.array([1.0, 1.0])

        def policy(x, k):
            return np.array([0.1 * k])  # Increases with time

        result = self.system.rollout(x0, policy, n_steps=10)

        # Controls should vary
        controls = result["controls"]
        self.assertNotEqual(controls[0, 0], controls[-1, 0])

    def test_rollout_stabilizing_policy(self):
        """rollout() with stabilizing policy improves stability."""
        x0 = np.array([1.0, 1.0])

        # LQR-like gain
        K = np.array([[1.0, 1.0]])

        def policy(x, k):
            return -K @ x

        result = self.system.rollout(x0, policy, n_steps=50)

        # Should converge to origin
        final_state = result["states"][-1, :]
        self.assertLess(np.linalg.norm(final_state), 0.1)

    # =========================================================================
    # Section 6: linearize() Method - Comprehensive Testing
    # =========================================================================

    def test_linearize_at_origin(self):
        """Linearize at origin."""
        x_eq = np.zeros(2)
        u_eq = np.zeros(1)

        lin = self.system.linearize(x_eq, u_eq)

        self.assertIsInstance(lin, tuple)
        self.assertEqual(len(lin), 2)

        Ad, Bd = lin
        self.assertEqual(Ad.shape, (2, 2))
        self.assertEqual(Bd.shape, (2, 1))

    def test_linearize_known_system(self):
        """Linearization matches analytical result."""
        x_eq = np.zeros(2)
        u_eq = np.zeros(1)

        Ad, Bd = self.system.linearize(x_eq, u_eq)

        # For x[k+1] = 0.9*x[k] + 0.1*u[k]
        expected_Ad = 0.9 * np.eye(2)
        expected_Bd = 0.1 * np.ones((2, 1))

        np.testing.assert_array_almost_equal(Ad, expected_Ad)
        np.testing.assert_array_almost_equal(Bd, expected_Bd)

    def test_linearize_without_control(self):
        """Linearize with u_eq=None."""
        x_eq = np.zeros(2)

        lin = self.system.linearize(x_eq, u_eq=None)

        Ad, Bd = lin
        self.assertIsNotNone(Ad)
        self.assertIsNotNone(Bd)

    def test_linearize_nonzero_equilibrium(self):
        """Linearize at non-zero equilibrium."""
        x_eq = np.array([1.0, 2.0])
        u_eq = np.array([0.5])

        lin = self.system.linearize(x_eq, u_eq)

        Ad, Bd = lin
        self.assertEqual(Ad.shape, (self.system.nx, self.system.nx))

    def test_linearize_correct_dimensions(self):
        """Linearization has correct dimensions."""
        x_eq = np.zeros(2)
        u_eq = np.zeros(1)

        Ad, Bd = self.system.linearize(x_eq, u_eq)

        self.assertEqual(Ad.shape, (self.system.nx, self.system.nx))
        self.assertEqual(Bd.shape, (self.system.nx, self.system.nu))

    def test_linearize_nonlinear_system_at_origin(self):
        """Linearize nonlinear system at origin."""
        x_eq = np.zeros(2)
        u_eq = np.zeros(1)

        Ad, Bd = self.nonlinear.linearize(x_eq, u_eq)

        # At x=0: tanh'(0) = 1, so Ad = I
        expected_Ad = np.eye(2)
        np.testing.assert_array_almost_equal(Ad, expected_Ad)

    def test_linearize_nonlinear_system_away_from_origin(self):
        """Linearize nonlinear system at non-zero point."""
        x_eq = np.array([1.0, -0.5])
        u_eq = np.zeros(1)

        Ad, Bd = self.nonlinear.linearize(x_eq, u_eq)

        # Ad should be diagonal with sech^2 values
        self.assertTrue(np.allclose(Ad, np.diag(np.diag(Ad))))

    def test_linearize_different_dimensions(self):
        """Linearization works for different dimensions."""
        for nx, nu in [(1, 1), (5, 2), (10, 5)]:
            sys = SimpleDiscreteSystem(nx=nx, nu=nu)
            x_eq = np.zeros(nx)
            u_eq = np.zeros(nu)

            Ad, Bd = sys.linearize(x_eq, u_eq)

            self.assertEqual(Ad.shape, (nx, nx))
            self.assertEqual(Bd.shape, (nx, nu))

    def test_linearize_stability_check(self):
        """Use linearization to check stability."""
        x_eq = np.zeros(2)
        u_eq = np.zeros(1)

        Ad, Bd = self.system.linearize(x_eq, u_eq)

        eigenvalues = np.linalg.eigvals(Ad)
        is_stable = np.all(np.abs(eigenvalues) < 1)

        # System with 0.9 eigenvalue is stable
        self.assertTrue(is_stable)

    def test_linearize_unstable_system(self):
        """Linearization detects unstable system."""
        x_eq = np.zeros(2)
        u_eq = np.zeros(1)

        Ad, Bd = self.unstable.linearize(x_eq, u_eq)

        eigenvalues = np.linalg.eigvals(Ad)
        is_unstable = np.any(np.abs(eigenvalues) > 1)

        # System with 1.1 eigenvalue is unstable
        self.assertTrue(is_unstable)

    # =========================================================================
    # Section 7: Properties - Comprehensive Testing
    # =========================================================================

    def test_is_continuous_returns_false(self):
        """is_continuous returns False for discrete systems."""
        self.assertFalse(self.system.is_continuous)

    def test_is_discrete_returns_true(self):
        """is_discrete returns True for discrete systems."""
        self.assertTrue(self.system.is_discrete)

    def test_is_stochastic_default_false(self):
        """is_stochastic returns False by default."""
        self.assertFalse(self.system.is_stochastic)

    def test_is_time_varying_default_false(self):
        """is_time_varying returns False by default."""
        self.assertFalse(self.system.is_time_varying)

    def test_is_time_varying_true_when_overridden(self):
        """is_time_varying returns True when overridden."""
        self.assertTrue(self.time_varying.is_time_varying)

    def test_sampling_frequency_computed_correctly(self):
        """Sampling frequency is 1/dt."""
        expected_freq = 1.0 / self.system.dt
        self.assertAlmostEqual(self.system.sampling_frequency, expected_freq)

    def test_properties_are_boolean(self):
        """System properties return boolean values."""
        self.assertIsInstance(self.system.is_continuous, bool)
        self.assertIsInstance(self.system.is_discrete, bool)
        self.assertIsInstance(self.system.is_stochastic, bool)
        self.assertIsInstance(self.system.is_time_varying, bool)

    # =========================================================================
    # Section 8: __repr__ and String Representation
    # =========================================================================

    def test_repr_contains_class_name(self):
        """String representation contains class name."""
        repr_str = repr(self.system)
        self.assertIn("SimpleDiscreteSystem", repr_str)

    def test_repr_contains_dimensions(self):
        """String representation contains dimensions."""
        repr_str = repr(self.system)
        self.assertIn("nx=2", repr_str)
        self.assertIn("nu=1", repr_str)
        self.assertIn("dt=0.1", repr_str)

    def test_repr_different_systems(self):
        """Different systems have different representations."""
        repr1 = repr(self.system)
        repr2 = repr(self.time_varying)

        self.assertNotEqual(repr1, repr2)

    # =========================================================================
    # Section 9: Polymorphism and Generic Programming
    # =========================================================================

    def test_polymorphic_usage(self):
        """System can be used polymorphically."""

        def check_stability(sys: DiscreteSystemBase):
            x_eq = np.zeros(sys.nx)
            u_eq = np.zeros(sys.nu)
            lin = sys.linearize(x_eq, u_eq)
            Ad, Bd = lin
            eigenvalues = np.linalg.eigvals(Ad)
            return np.all(np.abs(eigenvalues) < 1)

        self.assertTrue(check_stability(self.system))
        self.assertFalse(check_stability(self.unstable))

    def test_polymorphic_simulation(self):
        """Generic simulation works for all systems."""

        def simulate_from_random(sys: DiscreteSystemBase, n_steps=10):
            x0 = np.random.randn(sys.nx)
            return sys.simulate(x0, n_steps=n_steps)

        for system in [self.system, self.time_varying, self.nonlinear]:
            result = simulate_from_random(system)
            self.assertIn("states", result)

    def test_list_of_systems(self):
        """List of systems can be processed uniformly."""
        systems = [self.system, self.time_varying, self.unstable, self.nonlinear]

        for sys in systems:
            self.assertIsInstance(sys, DiscreteSystemBase)
            self.assertGreater(sys.dt, 0)

    # =========================================================================
    # Section 10: Error Handling and Edge Cases
    # =========================================================================

    def test_step_with_nan_state(self):
        """Step with NaN state doesn't crash."""
        x = np.array([np.nan, 1.0])
        u = np.array([0.5])

        x_next = self.system.step(x, u)

        # Should return something (even if NaN)
        self.assertEqual(x_next.shape, (2,))

    def test_step_with_inf_state(self):
        """Step with infinite state doesn't crash."""
        x = np.array([np.inf, 1.0])
        u = np.array([0.5])

        x_next = self.system.step(x, u)

        self.assertEqual(x_next.shape, (2,))

    def test_simulate_with_extreme_n_steps(self):
        """Simulate handles very large n_steps."""
        x0 = np.array([0.1, 0.1])  # Small initial state

        # This should work but may take time
        result = self.system.simulate(x0, n_steps=10000)

        self.assertEqual(result["states"].shape, (10001, self.system.nx))

    def test_simulate_with_zero_initial_state(self):
        """Simulate from zero initial state."""
        x0 = np.zeros(2)
        u = np.array([0.5])

        result = self.system.simulate(x0, u, n_steps=10)

        # Should propagate due to control
        final_state = result["states"][-1, :]
        self.assertGreater(np.linalg.norm(final_state), 0)


if __name__ == "__main__":
    unittest.main()
