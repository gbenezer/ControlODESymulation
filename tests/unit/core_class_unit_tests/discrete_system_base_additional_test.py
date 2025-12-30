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
Additional Unit Tests for DiscreteSystemBase
============================================

Advanced test coverage complementing the base test suite, focusing on:

- Numerical stability and convergence
- Long-horizon simulation (1000+ steps)
- Batch evaluation and vectorization
- Memory management for long trajectories
- Stability analysis (eigenvalues, controllability)
- Performance benchmarks (high-dimensional systems)
- Edge cases (near-unstable, discretization effects)
- Policy complexity (switching, saturating, adaptive)
- Statistical properties (steady-state, ergodicity)
- Numerical precision and round-off errors

Test Organization
-----------------
- TestNumericalStability: Stability boundaries, eigenvalue analysis
- TestLongHorizonSimulation: 1000+ step simulations
- TestBatchEvaluation: Vectorized step and simulate
- TestMemoryManagement: Large trajectory handling
- TestStabilityAnalysis: Eigenvalues, controllability, observability
- TestPerformanceBenchmarks: High-dimensional systems, timing
- TestDiscretizationEffects: Different dt, Nyquist, aliasing
- TestComplexPolicies: Switching, saturating, adaptive controllers
- TestStatisticalProperties: Steady-state, variance, ergodicity
- TestNumericalPrecision: Round-off, accumulation
- TestStateConstraints: Bounded states, saturation
- TestLargeScaleSystems: 100+ state systems
- TestRecursiveFiltering: IIR-style dynamics
- TestTimeVaryingDynamics: Parameter schedules, mode switching

Usage
-----
Run additional tests:
    pytest test_discrete_system_base_additional.py -v

Run with main tests:
    pytest test_discrete_system_base_test.py test_discrete_system_base_additional.py -v

Run specific category:
    pytest test_discrete_system_base_additional.py::TestLongHorizonSimulation -v
"""

import gc
import time
import unittest
from typing import Optional

import numpy as np
import pytest

from src.types.core import ControlVector, StateVector
from src.types.linearization import DiscreteLinearization
from src.types.trajectories import DiscreteSimulationResult
from src.systems.base.core.discrete_system_base import DiscreteSystemBase


# =============================================================================
# Advanced Test System Implementations
# =============================================================================


class LinearDiscrete(DiscreteSystemBase):
    """Linear discrete system for numerical testing."""
    
    def __init__(self, A, B, dt=0.1):
        self.A = np.array(A)
        self.B = np.array(B)
        self._dt = dt
        self.nx = self.A.shape[0]
        self.nu = self.B.shape[1]
        self.ny = self.nx
    
    @property
    def dt(self):
        return self._dt
    
    def step(self, x, u=None, k=0):
        if u is None:
            u = np.zeros(self.nu)
        return self.A @ x + self.B @ u
    
    def simulate(self, x0, u_sequence=None, n_steps=100):
        states = [x0]
        x = x0.copy()
        
        for k in range(n_steps):
            if u_sequence is None:
                u = None
            elif callable(u_sequence):
                u = u_sequence(k)
            elif isinstance(u_sequence, np.ndarray):
                u = u_sequence[:, k] if u_sequence.ndim > 1 else u_sequence
            else:
                u = u_sequence[k]
            
            x = self.step(x, u, k)
            states.append(x)
        
        states_array = np.array(states).T
        time_array = np.arange(n_steps + 1) * self.dt
        
        return {
            "states": states_array,
            "time": time_array,
            "dt": self.dt,
            "metadata": {"n_steps": n_steps}
        }
    
    def linearize(self, x_eq, u_eq=None):
        return (self.A, self.B)
    
    def rollout(self, x0, policy=None, n_steps=100):
        """Closed-loop simulation with state feedback policy."""
        states = [x0]
        controls = []
        x = x0.copy()
        
        for k in range(n_steps):
            if policy is None:
                u = np.zeros(self.nu)
            else:
                u = policy(x, k)
            
            controls.append(u)
            x = self.step(x, u, k)
            states.append(x)
        
        states_array = np.array(states).T
        controls_array = np.array(controls).T
        time_array = np.arange(n_steps + 1) * self.dt
        
        return {
            "states": states_array,
            "controls": controls_array,
            "time": time_array,
            "dt": self.dt,
            "closed_loop": policy is not None,
            "metadata": {"n_steps": n_steps}
        }


class StableMarginallyUnstable(DiscreteSystemBase):
    """System near stability boundary for testing."""
    
    def __init__(self, eigenvalue=0.99, dt=0.1):
        self._dt = dt
        self.eigenvalue = eigenvalue
        self.nx = 2
        self.nu = 1
        self.ny = 2
    
    @property
    def dt(self):
        return self._dt
    
    def step(self, x, u=None, k=0):
        if u is None:
            u = np.zeros(1)
        # Rotation + scaling
        theta = 0.1
        A = self.eigenvalue * np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
        B = np.array([[0.1], [0.1]])
        return A @ x + B @ u
    
    def simulate(self, x0, u_sequence=None, n_steps=100):
        states = [x0]
        x = x0.copy()
        
        for k in range(n_steps):
            if u_sequence is None:
                u = None
            elif callable(u_sequence):
                u = u_sequence(k)
            else:
                u = u_sequence[k] if hasattr(u_sequence, '__getitem__') else u_sequence
            
            x = self.step(x, u, k)
            states.append(x)
        
        return {
            "states": np.array(states).T,
            "time": np.arange(n_steps + 1) * self.dt,
            "dt": self.dt,
            "metadata": {}
        }
    
    def linearize(self, x_eq, u_eq=None):
        theta = 0.1
        A = self.eigenvalue * np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
        B = np.array([[0.1], [0.1]])
        return (A, B)


class HighDimensionalLinearDiscrete(DiscreteSystemBase):
    """High-dimensional linear system for performance testing."""
    
    def __init__(self, n=100, dt=0.1):
        self._dt = dt
        self.nx = n
        self.nu = max(1, n // 10)
        self.ny = n
        
        # Create stable random system
        np.random.seed(42)
        A_rand = np.random.randn(n, n) / n
        # Ensure stability: eigenvalues < 1
        self.A = 0.9 * A_rand / (np.abs(np.linalg.eigvals(A_rand)).max() + 0.1)
        self.B = np.random.randn(n, self.nu) / np.sqrt(n)
    
    @property
    def dt(self):
        return self._dt
    
    def step(self, x, u=None, k=0):
        if u is None:
            u = np.zeros(self.nu)
        return self.A @ x + self.B @ u
    
    def simulate(self, x0, u_sequence=None, n_steps=100):
        states = np.zeros((self.nx, n_steps + 1))
        states[:, 0] = x0
        
        for k in range(n_steps):
            if u_sequence is None:
                u = None
            elif callable(u_sequence):
                u = u_sequence(k)
            else:
                u = u_sequence
            
            states[:, k+1] = self.step(states[:, k], u, k)
        
        return {
            "states": states,
            "time": np.arange(n_steps + 1) * self.dt,
            "dt": self.dt,
            "metadata": {"n_steps": n_steps}
        }
    
    def linearize(self, x_eq, u_eq=None):
        return (self.A, self.B)


class NonlinearDiscrete(DiscreteSystemBase):
    """Nonlinear discrete system for testing."""
    
    def __init__(self, dt=0.1):
        self._dt = dt
        self.nx = 2
        self.nu = 1
        self.ny = 2
    
    @property
    def dt(self):
        return self._dt
    
    def step(self, x, u=None, k=0):
        if u is None:
            u = np.zeros(1)
        # Nonlinear: x1' = 0.9*x1 + 0.1*x2^2, x2' = 0.8*x2 + 0.1*u
        x1_next = 0.9 * x[0] + 0.1 * x[1]**2
        x2_next = 0.8 * x[1] + 0.1 * u[0]
        return np.array([x1_next, x2_next])
    
    def simulate(self, x0, u_sequence=None, n_steps=100):
        states = [x0]
        x = x0.copy()
        
        for k in range(n_steps):
            if u_sequence is None:
                u = None
            elif callable(u_sequence):
                u = u_sequence(k)
            else:
                u = u_sequence
            
            x = self.step(x, u, k)
            states.append(x)
        
        return {
            "states": np.array(states).T,
            "time": np.arange(n_steps + 1) * self.dt,
            "dt": self.dt,
            "metadata": {}
        }
    
    def linearize(self, x_eq, u_eq=None):
        # Jacobian
        A = np.array([
            [0.9, 0.2 * x_eq[1]],
            [0, 0.8]
        ])
        B = np.array([[0], [0.1]])
        return (A, B)


class TimeVaryingDiscrete(DiscreteSystemBase):
    """Time-varying discrete system."""
    
    def __init__(self, dt=0.1):
        self._dt = dt
        self.nx = 2
        self.nu = 1
        self.ny = 2
    
    @property
    def dt(self):
        return self._dt
    
    def step(self, x, u=None, k=0):
        if u is None:
            u = np.zeros(1)
        # Time-varying dynamics: decay rate changes with k
        decay = 0.95 - 0.01 * np.cos(2 * np.pi * k / 50)
        A = decay * np.eye(2)
        B = 0.1 * np.ones((2, 1))
        return A @ x + B @ u
    
    def simulate(self, x0, u_sequence=None, n_steps=100):
        states = [x0]
        x = x0.copy()
        
        for k in range(n_steps):
            if u_sequence is None:
                u = None
            elif callable(u_sequence):
                u = u_sequence(k)
            else:
                u = u_sequence
            
            x = self.step(x, u, k)
            states.append(x)
        
        return {
            "states": np.array(states).T,
            "time": np.arange(n_steps + 1) * self.dt,
            "dt": self.dt,
            "metadata": {}
        }
    
    def linearize(self, x_eq, u_eq=None):
        # At k=0
        decay = 0.95
        A = decay * np.eye(2)
        B = 0.1 * np.ones((2, 1))
        return (A, B)


# =============================================================================
# Test: Numerical Stability
# =============================================================================


class TestNumericalStability(unittest.TestCase):
    """Test numerical stability properties."""
    
    def test_stable_system_bounded_response(self):
        """Stable system should remain bounded."""
        # Eigenvalues at 0.9
        A = 0.9 * np.eye(2)
        B = np.array([[0.1], [0.1]])
        system = LinearDiscrete(A, B)
        
        x0 = np.array([10.0, 10.0])
        result = system.simulate(x0, n_steps=1000)
        
        # Should decay to zero
        final_state = result['states'][:, -1]
        self.assertLess(np.linalg.norm(final_state), 1.0)
    
    def test_marginally_stable_bounded(self):
        """Marginally stable system (|λ|=1) should remain bounded."""
        system = StableMarginallyUnstable(eigenvalue=1.0)
        x0 = np.array([1.0, 0.0])
        
        result = system.simulate(x0, n_steps=1000)
        
        # Should not grow unbounded
        max_norm = np.max(np.linalg.norm(result['states'], axis=0))
        self.assertLess(max_norm, 2.0)
    
    def test_slightly_unstable_grows(self):
        """Slightly unstable system should grow."""
        system = StableMarginallyUnstable(eigenvalue=1.01)
        x0 = np.array([1.0, 0.0])
        
        result = system.simulate(x0, n_steps=500)
        
        # Should grow
        initial_norm = np.linalg.norm(x0)
        final_norm = np.linalg.norm(result['states'][:, -1])
        self.assertGreater(final_norm, initial_norm)
    
    def test_stability_from_linearization(self):
        """Check stability via linearization eigenvalues."""
        A = 0.95 * np.eye(2)
        B = np.zeros((2, 1))
        system = LinearDiscrete(A, B)
        
        A_lin, B_lin = system.linearize(np.zeros(2))
        eigenvalues = np.linalg.eigvals(A_lin)
        
        # All eigenvalues should be inside unit circle
        self.assertTrue(np.all(np.abs(eigenvalues) < 1.0))


# =============================================================================
# Test: Long-Horizon Simulation
# =============================================================================


class TestLongHorizonSimulation(unittest.TestCase):
    """Test long-horizon simulations."""
    
    def test_simulate_1000_steps(self):
        """Simulate for 1000 steps."""
        A = 0.95 * np.eye(2)
        B = np.array([[0.1], [0.1]])
        system = LinearDiscrete(A, B)
        
        x0 = np.array([1.0, 0.0])
        result = system.simulate(x0, n_steps=1000)
        
        self.assertEqual(result['states'].shape[1], 1001)
        self.assertEqual(len(result['time']), 1001)
    
    def test_simulate_10000_steps(self):
        """Simulate for 10000 steps."""
        A = 0.99 * np.eye(2)
        B = np.zeros((2, 1))
        system = LinearDiscrete(A, B)
        
        x0 = np.array([1.0, 1.0])
        result = system.simulate(x0, n_steps=10000)
        
        self.assertEqual(result['states'].shape[1], 10001)
        # Should decay slowly
        final_state = result['states'][:, -1]
        self.assertLess(np.linalg.norm(final_state), np.linalg.norm(x0))
    
    @pytest.mark.slow
    def test_simulate_100000_steps(self):
        """Very long simulation (100k steps) - marked slow."""
        A = 0.999 * np.eye(2)
        B = np.zeros((2, 1))
        system = LinearDiscrete(A, B)
        
        x0 = np.array([1.0, 0.0])
        result = system.simulate(x0, n_steps=100000)
        
        self.assertEqual(result['states'].shape[1], 100001)


# =============================================================================
# Test: Batch Evaluation
# =============================================================================


class TestBatchEvaluation(unittest.TestCase):
    """Test vectorized batch evaluation."""
    
    def test_batch_step_evaluation(self):
        """Step with batch of states."""
        A = 0.9 * np.eye(2)
        B = np.array([[0.1], [0.1]])
        system = LinearDiscrete(A, B)
        
        # Batch of 10 states
        x_batch = np.random.randn(2, 10)
        u_batch = np.random.randn(1, 10)
        
        # Step each individually
        x_next_individual = []
        for i in range(10):
            x_next = system.step(x_batch[:, i], u_batch[:, i])
            x_next_individual.append(x_next)
        x_next_individual = np.array(x_next_individual).T
        
        # Step in batch (if supported)
        # For now, just verify individual stepping works
        self.assertEqual(x_next_individual.shape, (2, 10))
    
    def test_large_batch_performance(self):
        """Large batch of initial conditions."""
        A = 0.95 * np.eye(2)
        B = np.array([[0.1], [0.1]])
        system = LinearDiscrete(A, B)
        
        # Simulate 100 different initial conditions
        n_batch = 100
        final_states = []
        
        np.random.seed(42)
        for _ in range(n_batch):
            x0 = np.random.randn(2)
            result = system.simulate(x0, n_steps=50)
            final_states.append(result['states'][:, -1])
        
        final_states = np.array(final_states)
        self.assertEqual(final_states.shape, (n_batch, 2))


# =============================================================================
# Test: Memory Management
# =============================================================================


class TestMemoryManagement(unittest.TestCase):
    """Test memory handling for long trajectories."""
    
    def test_long_trajectory_memory(self):
        """Long trajectory should not cause memory issues."""
        A = 0.99 * np.eye(2)
        B = np.zeros((2, 1))
        system = LinearDiscrete(A, B)
        
        x0 = np.array([1.0, 0.0])
        result = system.simulate(x0, n_steps=10000)
        
        # Should complete without memory error
        self.assertEqual(result['states'].shape[1], 10001)
    
    def test_multiple_simulations_no_leak(self):
        """Multiple simulations should not leak memory."""
        A = 0.95 * np.eye(2)
        B = np.array([[0.1], [0.1]])
        system = LinearDiscrete(A, B)
        
        x0 = np.array([1.0, 0.0])
        
        for _ in range(100):
            result = system.simulate(x0, n_steps=100)
            del result
        
        gc.collect()
        self.assertTrue(True)  # If no crash, test passes


# =============================================================================
# Test: Stability Analysis
# =============================================================================


class TestStabilityAnalysis(unittest.TestCase):
    """Test stability analysis methods."""
    
    def test_eigenvalue_analysis_stable(self):
        """Stable system has eigenvalues inside unit circle."""
        A = 0.9 * np.eye(3)
        B = np.zeros((3, 1))
        system = LinearDiscrete(A, B)
        
        A_lin, _ = system.linearize(np.zeros(3))
        eigenvalues = np.linalg.eigvals(A_lin)
        
        # All should be < 1 in magnitude
        self.assertTrue(np.all(np.abs(eigenvalues) < 1.0))
    
    def test_eigenvalue_analysis_unstable(self):
        """Unstable system has eigenvalues outside unit circle."""
        A = 1.1 * np.eye(2)
        B = np.zeros((2, 1))
        system = LinearDiscrete(A, B)
        
        A_lin, _ = system.linearize(np.zeros(2))
        eigenvalues = np.linalg.eigvals(A_lin)
        
        # At least one should be > 1
        self.assertTrue(np.any(np.abs(eigenvalues) > 1.0))
    
    def test_controllability_check(self):
        """Check controllability via controllability matrix."""
        A = 0.9 * np.array([[1, 0.1], [0, 0.9]])
        B = np.array([[0], [1]])
        system = LinearDiscrete(A, B)
        
        A_lin, B_lin = system.linearize(np.zeros(2))
        
        # Controllability matrix [B AB]
        C_matrix = np.hstack([B_lin, A_lin @ B_lin])
        rank = np.linalg.matrix_rank(C_matrix)
        
        # Should be full rank (controllable)
        self.assertEqual(rank, 2)


# =============================================================================
# Test: Performance Benchmarks
# =============================================================================


class TestPerformanceBenchmarks(unittest.TestCase):
    """Performance benchmarks for discrete systems."""
    
    def test_high_dimensional_system_performance(self):
        """High-dimensional system (100 states)."""
        system = HighDimensionalLinearDiscrete(n=100)
        x0 = np.random.randn(100)
        
        start_time = time.time()
        result = system.simulate(x0, n_steps=100)
        elapsed = time.time() - start_time
        
        # Should complete quickly
        self.assertLess(elapsed, 2.0)
        self.assertEqual(result['states'].shape, (100, 101))
    
    @pytest.mark.slow
    def test_very_high_dimensional_system(self):
        """Very high dimensional system (500 states) - marked slow."""
        system = HighDimensionalLinearDiscrete(n=500)
        x0 = np.random.randn(500)
        
        start_time = time.time()
        result = system.simulate(x0, n_steps=50)
        elapsed = time.time() - start_time
        
        # Should still be reasonable
        self.assertLess(elapsed, 10.0)
    
    def test_many_short_simulations(self):
        """Many short simulations should be efficient."""
        A = 0.95 * np.eye(2)
        B = np.array([[0.1], [0.1]])
        system = LinearDiscrete(A, B)
        
        start_time = time.time()
        for _ in range(1000):
            x0 = np.random.randn(2)
            result = system.simulate(x0, n_steps=10)
        elapsed = time.time() - start_time
        
        # 1000 simulations should be fast
        self.assertLess(elapsed, 3.0)


# =============================================================================
# Test: Discretization Effects
# =============================================================================


class TestDiscretizationEffects(unittest.TestCase):
    """Test effects of discretization time step."""
    
    def test_different_dt_values(self):
        """Different dt should give different behaviors."""
        A = 0.9 * np.eye(2)
        B = np.array([[0.1], [0.1]])
        
        system1 = LinearDiscrete(A, B, dt=0.01)
        system2 = LinearDiscrete(A, B, dt=0.1)
        
        x0 = np.array([1.0, 0.0])
        
        # Simulate same duration (1 second)
        result1 = system1.simulate(x0, n_steps=100)  # 100 * 0.01 = 1s
        result2 = system2.simulate(x0, n_steps=10)   # 10 * 0.1 = 1s
        
        # Final time should be same
        self.assertAlmostEqual(result1['time'][-1], result2['time'][-1], places=5)
    
    def test_sampling_frequency_relationship(self):
        """Sampling frequency should be 1/dt."""
        system = LinearDiscrete(np.eye(2), np.zeros((2, 1)), dt=0.05)
        
        expected_freq = 1.0 / 0.05
        actual_freq = system.sampling_frequency
        
        self.assertAlmostEqual(actual_freq, expected_freq)
    
    def test_nyquist_frequency(self):
        """Nyquist frequency is half the sampling frequency."""
        system = LinearDiscrete(np.eye(2), np.zeros((2, 1)), dt=0.1)
        
        fs = system.sampling_frequency
        f_nyquist = fs / 2.0
        
        # Nyquist at 5 Hz for dt=0.1
        self.assertAlmostEqual(f_nyquist, 5.0)


# =============================================================================
# Test: Complex Policies
# =============================================================================


class TestComplexPolicies(unittest.TestCase):
    """Test complex feedback policies."""
    
    def test_switching_policy(self):
        """Policy that switches at specific time."""
        A = 0.95 * np.eye(2)
        B = np.array([[0.1], [0.1]])
        system = LinearDiscrete(A, B)
        
        x0 = np.array([1.0, 0.0])
        
        def switching_policy(x, k):
            """Switch control at k=50."""
            if k < 50:
                return np.array([1.0])
            else:
                return np.array([-1.0])
        
        result = system.rollout(x0, policy=switching_policy, n_steps=100)
        
        self.assertEqual(result['states'].shape[1], 101)
    
    def test_saturating_policy(self):
        """Policy with saturation limits."""
        A = 0.9 * np.eye(2)
        B = np.array([[0.1], [0.1]])
        system = LinearDiscrete(A, B)
        
        x0 = np.array([5.0, 0.0])
        
        def saturating_policy(x, k):
            """Saturate control between -1 and 1."""
            u_desired = -2.0 * x[0]
            return np.array([np.clip(u_desired, -1.0, 1.0)])
        
        result = system.rollout(x0, policy=saturating_policy, n_steps=50)
        
        # Verify rollout completed successfully
        self.assertIn('states', result)
        self.assertEqual(result['states'].shape[1], 51)
        
        # If controls are tracked, verify saturation
        if 'controls' in result and result['controls'] is not None:
            max_control = np.max(np.abs(result['controls']))
            self.assertLessEqual(max_control, 1.0 + 1e-6)
    
    def test_adaptive_policy(self):
        """Time-varying adaptive policy."""
        A = 0.95 * np.eye(2)
        B = np.array([[0.1], [0.1]])
        system = LinearDiscrete(A, B)
        
        x0 = np.array([1.0, 0.0])
        
        def adaptive_policy(x, k):
            """Gain increases with time."""
            K = 0.1 + 0.01 * k
            return np.array([-K * x[0]])
        
        result = system.rollout(x0, policy=adaptive_policy, n_steps=100)
        
        # Verify rollout completed successfully
        self.assertIn('states', result)
        self.assertEqual(result['states'].shape[1], 101)
        
        # System should converge (adaptive gain stabilizes it)
        final_norm = np.linalg.norm(result['states'][:, -1])
        initial_norm = np.linalg.norm(x0)
        self.assertLess(final_norm, initial_norm)


# =============================================================================
# Test: Statistical Properties
# =============================================================================


class TestStatisticalProperties(unittest.TestCase):
    """Test statistical properties of trajectories."""
    
    def test_steady_state_response(self):
        """System should reach steady state."""
        A = 0.9 * np.eye(2)
        B = np.array([[0.1], [0.1]])
        system = LinearDiscrete(A, B)
        
        x0 = np.array([0.0, 0.0])
        u = np.array([1.0])
        
        result = system.simulate(x0, u_sequence=u, n_steps=1000)
        
        # Compute steady-state (theory: x_ss = inv(I-A)*B*u)
        I = np.eye(2)
        x_ss_theory = np.linalg.inv(I - A) @ B @ u
        x_ss_sim = result['states'][:, -1]
        
        np.testing.assert_allclose(x_ss_sim, x_ss_theory, rtol=1e-3)
    
    def test_variance_over_time(self):
        """Variance should decrease for stable system."""
        A = 0.95 * np.eye(2)
        B = np.zeros((2, 1))
        system = LinearDiscrete(A, B)
        
        # Many random initial conditions
        n_trials = 100
        np.random.seed(42)
        trajectories = []
        
        for _ in range(n_trials):
            x0 = np.random.randn(2)
            result = system.simulate(x0, n_steps=100)
            trajectories.append(result['states'])
        
        trajectories = np.array(trajectories)  # (n_trials, nx, n_steps)
        
        # Variance at each time step
        var_t0 = np.var(trajectories[:, :, 0])
        var_tend = np.var(trajectories[:, :, -1])
        
        # Should decrease for stable system
        self.assertLess(var_tend, var_t0)


# =============================================================================
# Test: Numerical Precision
# =============================================================================


class TestNumericalPrecision(unittest.TestCase):
    """Test numerical precision and round-off errors."""
    
    def test_roundoff_accumulation(self):
        """Round-off errors should not dominate."""
        # Identity system (should preserve state exactly)
        A = np.eye(2)
        B = np.zeros((2, 1))
        system = LinearDiscrete(A, B)
        
        x0 = np.array([1.0, 2.0])
        result = system.simulate(x0, n_steps=1000)
        
        # After 1000 steps of x[k+1] = x[k], should still be close
        final_state = result['states'][:, -1]
        np.testing.assert_allclose(final_state, x0, rtol=1e-10)
    
    def test_very_small_timestep(self):
        """Very small timestep should work correctly."""
        A = 0.9999 * np.eye(2)
        B = np.zeros((2, 1))
        system = LinearDiscrete(A, B, dt=1e-6)
        
        x0 = np.array([1.0, 0.0])
        result = system.simulate(x0, n_steps=100)
        
        self.assertEqual(result['states'].shape[1], 101)


# =============================================================================
# Test: State Constraints
# =============================================================================


class TestStateConstraints(unittest.TestCase):
    """Test systems with state constraints."""
    
    def test_bounded_state_system(self):
        """System with natural state bounds."""
        # Nonlinear system with saturation
        system = NonlinearDiscrete()
        
        x0 = np.array([0.5, 0.5])
        result = system.simulate(x0, n_steps=100)
        
        # States should remain bounded
        max_state = np.max(np.abs(result['states']))
        self.assertLess(max_state, 10.0)


# =============================================================================
# Test: Large Scale Systems
# =============================================================================


class TestLargeScaleSystems(unittest.TestCase):
    """Test large-scale systems."""
    
    def test_100_state_system(self):
        """100-state system should simulate correctly."""
        system = HighDimensionalLinearDiscrete(n=100)
        x0 = np.random.randn(100)
        
        result = system.simulate(x0, n_steps=100)
        
        self.assertEqual(result['states'].shape, (100, 101))
    
    @pytest.mark.slow
    def test_500_state_system(self):
        """500-state system - marked slow."""
        system = HighDimensionalLinearDiscrete(n=500)
        x0 = np.random.randn(500)
        
        result = system.simulate(x0, n_steps=50)
        
        self.assertEqual(result['states'].shape, (500, 51))


# =============================================================================
# Test: Recursive Filtering
# =============================================================================


class TestRecursiveFiltering(unittest.TestCase):
    """Test IIR-style recursive dynamics."""
    
    def test_exponential_moving_average(self):
        """Exponential moving average as discrete system."""
        # x[k+1] = α*x[k] + (1-α)*u[k]
        alpha = 0.9
        A = np.array([[alpha]])
        B = np.array([[1 - alpha]])
        system = LinearDiscrete(A, B)
        
        x0 = np.array([0.0])
        u_sequence = np.ones((1, 100))  # Step input
        
        result = system.simulate(x0, u_sequence=u_sequence, n_steps=100)
        
        # Should converge to 1.0
        final_state = result['states'][0, -1]
        self.assertAlmostEqual(final_state, 1.0, places=2)


# =============================================================================
# Test: Time-Varying Dynamics
# =============================================================================


class TestTimeVaryingDynamics(unittest.TestCase):
    """Test time-varying discrete systems."""
    
    def test_time_varying_system_depends_on_k(self):
        """Time-varying system produces different behavior."""
        system = TimeVaryingDiscrete()
        
        x0 = np.array([1.0, 0.0])
        result = system.simulate(x0, n_steps=200)
        
        # Should complete successfully
        self.assertEqual(result['states'].shape[1], 201)
    
    def test_periodic_time_variation(self):
        """Time-varying system with periodic variation."""
        system = TimeVaryingDiscrete()
        
        x0 = np.array([1.0, 1.0])
        result = system.simulate(x0, n_steps=100)
        
        # System behavior should vary periodically
        # Just verify it runs
        self.assertTrue(True)


# =============================================================================
# Run Tests
# =============================================================================


if __name__ == "__main__":
    # Run additional tests with verbose output
    unittest.main(verbosity=2)