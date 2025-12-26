# Copyright (C) 2025 Gil Benezer
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""
Unit Tests for DiscreteSystemBase
=================================

Tests the abstract base class interface for discrete-time systems.
"""

import unittest
from typing import Optional, Sequence, Union, Callable

import numpy as np

from src.types.core import ControlVector, StateVector
from src.types.linearization import DiscreteLinearization
from src.types.trajectories import DiscreteSimulationResult

# Import after src.types
import sys
sys.path.insert(0, '/home/claude')
from discrete_system_base import DiscreteSystemBase


class SimpleDiscreteSystem(DiscreteSystemBase):
    """
    Concrete implementation for testing.
    
    Implements: x[k+1] = 0.9*x[k] + 0.1*u[k] (stable discrete linear system)
    """
    
    def __init__(self, nx=2, nu=1, dt=0.1):
        self.nx = nx
        self.nu = nu
        self.ny = nx
        self._dt = dt
        self.Ad = 0.9 * np.eye(nx)
        self.Bd = 0.1 * np.ones((nx, nu))
    
    @property
    def dt(self) -> float:
        return self._dt
    
    def step(
        self,
        x: StateVector,
        u: Optional[ControlVector] = None,
        k: int = 0
    ) -> StateVector:
        """x[k+1] = Ad*x[k] + Bd*u[k]"""
        if u is None:
            u = np.zeros(self.nu)
        
        # Handle batched inputs
        if x.ndim == 2:
            u = u if u.ndim == 2 else u.reshape(-1, 1)
            return self.Ad @ x + self.Bd @ u
        
        return self.Ad @ x + self.Bd @ u.flatten()
    
    def simulate(
        self,
        x0: StateVector,
        u_sequence: Optional[Union[ControlVector, Sequence[ControlVector], 
                                   Callable[[int], ControlVector]]] = None,
        n_steps: int = 100,
        **kwargs
    ) -> DiscreteSimulationResult:
        """Simulate for n_steps."""
        states = np.zeros((self.nx, n_steps + 1))
        states[:, 0] = x0
        
        controls = [] if u_sequence is not None else None
        
        for k in range(n_steps):
            # Evaluate control
            if u_sequence is None:
                u = None
            elif callable(u_sequence):
                u = u_sequence(k)
            elif isinstance(u_sequence, np.ndarray) and u_sequence.ndim == 1:
                # Constant control
                u = u_sequence
            else:
                # Sequence
                u = u_sequence[k]
            
            if controls is not None and u is not None:
                controls.append(u)
            
            # Step forward
            states[:, k+1] = self.step(states[:, k], u, k)
        
        controls_array = np.array(controls).T if controls else None
        
        return DiscreteSimulationResult(
            states=states,
            controls=controls_array,
            time_steps=np.arange(n_steps + 1),
            dt=self.dt,
            metadata=kwargs
        )
    
    def linearize(
        self,
        x_eq: StateVector,
        u_eq: Optional[ControlVector] = None
    ) -> DiscreteLinearization:
        """Already linear, return Ad and Bd."""
        return DiscreteLinearization(
            Ad=self.Ad,
            Bd=self.Bd,
            x_eq=x_eq,
            u_eq=u_eq if u_eq is not None else np.zeros(self.nu),
            dt=self.dt
        )


class TimeVaryingDiscreteSystem(DiscreteSystemBase):
    """Time-varying discrete system for testing."""
    
    def __init__(self, dt=0.1):
        self.nx = 1
        self.nu = 1
        self._dt = dt
    
    @property
    def dt(self):
        return self._dt
    
    def step(self, x, u=None, k=0):
        """x[k+1] = (0.9 - 0.01*k)*x[k] + u[k]"""
        u = u if u is not None else np.array([0.0])
        decay = 0.9 - 0.01 * k
        return decay * x + u
    
    def simulate(self, x0, u_sequence=None, n_steps=100, **kwargs):
        states = np.zeros((self.nx, n_steps + 1))
        states[:, 0] = x0
        
        for k in range(n_steps):
            u = u_sequence(k) if callable(u_sequence) else None
            states[:, k+1] = self.step(states[:, k], u, k)
        
        return DiscreteSimulationResult(
            states=states, controls=None, time_steps=np.arange(n_steps + 1),
            dt=self.dt, metadata={}
        )
    
    def linearize(self, x_eq, u_eq=None):
        # Time-varying, linearization depends on k
        raise NotImplementedError("Use linearize_at_step(k)")
    
    @property
    def is_time_varying(self):
        return True


class TestDiscreteSystemBase(unittest.TestCase):
    """Test suite for DiscreteSystemBase abstract class."""
    
    def setUp(self):
        """Create test systems."""
        self.system = SimpleDiscreteSystem(nx=2, nu=1, dt=0.1)
        self.time_varying = TimeVaryingDiscreteSystem(dt=0.05)
    
    # =========================================================================
    # Test Abstract Class Properties
    # =========================================================================
    
    def test_cannot_instantiate_abstract_class(self):
        """DiscreteSystemBase cannot be instantiated directly."""
        with self.assertRaises(TypeError):
            system = DiscreteSystemBase()
    
    def test_concrete_class_instantiation(self):
        """Concrete implementation can be instantiated."""
        system = SimpleDiscreteSystem()
        self.assertIsInstance(system, DiscreteSystemBase)
    
    def test_missing_abstract_method_raises_error(self):
        """Class missing abstract methods cannot be instantiated."""
        
        class IncompleteSystem(DiscreteSystemBase):
            def step(self, x, u=None, k=0):
                return x
            # Missing simulate(), linearize(), and dt property
        
        with self.assertRaises(TypeError):
            system = IncompleteSystem()
    
    def test_missing_dt_property_raises_error(self):
        """Class without dt property cannot be instantiated."""
        
        class NoDtSystem(DiscreteSystemBase):
            def step(self, x, u=None, k=0):
                return x
            def simulate(self, x0, u_sequence=None, n_steps=100, **kwargs):
                return None
            def linearize(self, x_eq, u_eq=None):
                return None
            # Missing dt property
        
        with self.assertRaises(TypeError):
            system = NoDtSystem()
    
    # =========================================================================
    # Test dt Property
    # =========================================================================
    
    def test_dt_property_exists(self):
        """dt property is accessible."""
        self.assertEqual(self.system.dt, 0.1)
    
    def test_dt_property_positive(self):
        """dt should be positive."""
        self.assertGreater(self.system.dt, 0.0)
    
    def test_sampling_frequency(self):
        """Sampling frequency is 1/dt."""
        expected_freq = 1.0 / self.system.dt
        self.assertAlmostEqual(self.system.sampling_frequency, expected_freq)
    
    # =========================================================================
    # Test step() Method
    # =========================================================================
    
    def test_step_with_control(self):
        """Step forward with control input."""
        x = np.array([1.0, 2.0])
        u = np.array([0.5])
        
        x_next = self.system.step(x, u)
        
        self.assertEqual(x_next.shape, (2,))
        # Expected: 0.9*x + 0.1*u = [0.9, 1.8] + [0.05, 0.05] = [0.95, 1.85]
        expected = np.array([0.95, 1.85])
        np.testing.assert_array_almost_equal(x_next, expected)
    
    def test_step_without_control(self):
        """Step forward without control (u=None)."""
        x = np.array([1.0, 2.0])
        
        x_next = self.system.step(x, u=None)
        
        # Expected: 0.9*x = [0.9, 1.8]
        expected = np.array([0.9, 1.8])
        np.testing.assert_array_almost_equal(x_next, expected)
    
    def test_step_with_time_index(self):
        """Time index k is accepted (may be ignored for time-invariant)."""
        x = np.array([1.0, 2.0])
        u = np.array([0.5])
        
        x_next = self.system.step(x, u, k=10)
        
        # Time-invariant system ignores k
        expected = np.array([0.95, 1.85])
        np.testing.assert_array_almost_equal(x_next, expected)
    
    def test_step_batch_evaluation(self):
        """Step multiple states simultaneously."""
        x_batch = np.array([[1.0, 2.0, 3.0],
                           [0.5, 1.5, 2.5]])  # 3 states
        u_batch = np.array([[0.5, 0.3, 0.1]])  # 3 controls
        
        x_next_batch = self.system.step(x_batch, u_batch)
        
        self.assertEqual(x_next_batch.shape, (2, 3))
    
    def test_time_varying_system_step(self):
        """Time-varying system uses time index k."""
        x = np.array([1.0])
        u = np.array([0.0])
        
        x_next_k0 = self.time_varying.step(x, u, k=0)
        x_next_k10 = self.time_varying.step(x, u, k=10)
        
        # x[k+1] = (0.9 - 0.01*k)*x[k], so should be different
        self.assertNotEqual(x_next_k0[0], x_next_k10[0])
    
    def test_step_preserves_shape(self):
        """Output has same shape as input state."""
        x = np.array([1.0, 2.0])
        x_next = self.system.step(x)
        
        self.assertEqual(x.shape, x_next.shape)
    
    # =========================================================================
    # Test simulate() Method
    # =========================================================================
    
    def test_simulate_basic(self):
        """Basic simulation test."""
        x0 = np.array([1.0, 1.0])
        result = self.system.simulate(x0, u_sequence=None, n_steps=10)
        
        self.assertIsInstance(result, DiscreteSimulationResult)
        self.assertIsNotNone(result.states)
        self.assertIsNotNone(result.time_steps)
    
    def test_simulate_constant_control(self):
        """Simulate with constant control."""
        x0 = np.array([1.0, 1.0])
        u = np.array([0.5])
        
        result = self.system.simulate(x0, u, n_steps=10)
        
        self.assertEqual(result.states.shape[0], 2)  # nx
        self.assertEqual(result.states.shape[1], 11)  # n_steps + 1
    
    def test_simulate_control_sequence(self):
        """Simulate with pre-computed control sequence."""
        x0 = np.array([1.0, 1.0])
        u_seq = [np.array([0.1 * k]) for k in range(10)]
        
        result = self.system.simulate(x0, u_seq, n_steps=10)
        
        self.assertIsInstance(result, DiscreteSimulationResult)
    
    def test_simulate_control_function(self):
        """Simulate with time-varying control function."""
        x0 = np.array([1.0, 1.0])
        u_func = lambda k: np.array([0.5 * np.sin(k * 0.1)])
        
        result = self.system.simulate(x0, u_func, n_steps=10)
        
        self.assertIsInstance(result, DiscreteSimulationResult)
    
    def test_simulate_result_structure(self):
        """Simulation result has correct structure."""
        x0 = np.array([1.0, 1.0])
        result = self.system.simulate(x0, n_steps=10)
        
        # Check DiscreteSimulationResult attributes
        self.assertTrue(hasattr(result, 'states'))
        self.assertTrue(hasattr(result, 'controls'))
        self.assertTrue(hasattr(result, 'time_steps'))
        self.assertTrue(hasattr(result, 'dt'))
        self.assertTrue(hasattr(result, 'metadata'))
    
    def test_simulate_time_steps(self):
        """Time steps are correct."""
        x0 = np.array([1.0, 1.0])
        n_steps = 10
        result = self.system.simulate(x0, n_steps=n_steps)
        
        expected_steps = np.arange(n_steps + 1)
        np.testing.assert_array_equal(result.time_steps, expected_steps)
    
    def test_simulate_includes_initial_state(self):
        """Simulation includes initial state x0."""
        x0 = np.array([1.0, 2.0])
        result = self.system.simulate(x0, n_steps=5)
        
        np.testing.assert_array_equal(result.states[:, 0], x0)
    
    def test_simulate_autonomous(self):
        """Simulate autonomous system (no control)."""
        x0 = np.array([1.0, 1.0])
        result = self.system.simulate(x0, u_sequence=None, n_steps=10)
        
        # Should decay to zero for stable system
        self.assertLess(np.linalg.norm(result.states[:, -1]), 
                       np.linalg.norm(x0))
    
    # =========================================================================
    # Test linearize() Method
    # =========================================================================
    
    def test_linearize_at_origin(self):
        """Linearize at origin."""
        x_eq = np.zeros(2)
        u_eq = np.zeros(1)
        
        lin = self.system.linearize(x_eq, u_eq)
        
        self.assertIsInstance(lin, DiscreteLinearization)
        self.assertEqual(lin.Ad.shape, (2, 2))
        self.assertEqual(lin.Bd.shape, (2, 1))
    
    def test_linearize_without_control(self):
        """Linearize with u_eq=None."""
        x_eq = np.zeros(2)
        
        lin = self.system.linearize(x_eq, u_eq=None)
        
        self.assertIsNotNone(lin.Ad)
        self.assertIsNotNone(lin.Bd)
    
    def test_linearize_equilibrium_stored(self):
        """Linearization stores equilibrium point."""
        x_eq = np.array([1.0, 2.0])
        u_eq = np.array([0.5])
        
        lin = self.system.linearize(x_eq, u_eq)
        
        np.testing.assert_array_equal(lin.x_eq, x_eq)
        np.testing.assert_array_equal(lin.u_eq, u_eq)
    
    def test_linearize_dt_stored(self):
        """Linearization stores time step."""
        x_eq = np.zeros(2)
        u_eq = np.zeros(1)
        
        lin = self.system.linearize(x_eq, u_eq)
        
        self.assertEqual(lin.dt, self.system.dt)
    
    def test_linearize_correct_dimensions(self):
        """Linearization matrices have correct dimensions."""
        x_eq = np.zeros(2)
        u_eq = np.zeros(1)
        
        lin = self.system.linearize(x_eq, u_eq)
        
        # Ad should be (nx, nx), Bd should be (nx, nu)
        self.assertEqual(lin.Ad.shape, (self.system.nx, self.system.nx))
        self.assertEqual(lin.Bd.shape, (self.system.nx, self.system.nu))
    
    def test_linearize_known_system(self):
        """Linearization of known system matches analytical result."""
        x_eq = np.zeros(2)
        u_eq = np.zeros(1)
        
        lin = self.system.linearize(x_eq, u_eq)
        
        # For x[k+1] = 0.9*x[k] + 0.1*u[k]: Ad = 0.9*I, Bd = 0.1*I
        expected_Ad = 0.9 * np.eye(2)
        expected_Bd = 0.1 * np.ones((2, 1))
        
        np.testing.assert_array_almost_equal(lin.Ad, expected_Ad)
        np.testing.assert_array_almost_equal(lin.Bd, expected_Bd)
    
    # =========================================================================
    # Test rollout() Method
    # =========================================================================
    
    def test_rollout_without_policy(self):
        """Rollout without feedback policy (open-loop)."""
        x0 = np.array([1.0, 1.0])
        
        result = self.system.rollout(x0, policy=None, n_steps=10)
        
        self.assertIsInstance(result, DiscreteSimulationResult)
    
    def test_rollout_with_state_feedback(self):
        """Rollout with state feedback policy."""
        x0 = np.array([1.0, 1.0])
        K = np.array([[-0.5, -0.5]])  # Stabilizing gain
        
        def policy(x, k):
            return -K @ x
        
        result = self.system.rollout(x0, policy, n_steps=20)
        
        # Should stabilize to zero
        self.assertLess(np.linalg.norm(result.states[:, -1]), 0.1)
    
    def test_rollout_time_varying_policy(self):
        """Rollout with time-varying policy."""
        x0 = np.array([1.0, 1.0])
        
        def policy(x, k):
            return np.array([0.1 * np.sin(k * 0.1)])
        
        result = self.system.rollout(x0, policy, n_steps=10)
        
        self.assertEqual(result.states.shape[1], 11)  # n_steps + 1
    
    def test_rollout_closed_loop_flag(self):
        """Rollout sets closed_loop flag in metadata."""
        x0 = np.array([1.0, 1.0])
        
        def policy(x, k):
            return np.zeros(1)
        
        result = self.system.rollout(x0, policy, n_steps=5)
        
        self.assertTrue(result.metadata.get('closed_loop', False))
    
    # =========================================================================
    # Test Properties
    # =========================================================================
    
    def test_is_continuous_property(self):
        """is_continuous returns False."""
        self.assertFalse(self.system.is_continuous)
    
    def test_is_discrete_property(self):
        """is_discrete returns True."""
        self.assertTrue(self.system.is_discrete)
    
    def test_is_stochastic_property(self):
        """is_stochastic returns False by default."""
        self.assertFalse(self.system.is_stochastic)
    
    def test_is_time_varying_property_default(self):
        """is_time_varying returns False by default."""
        self.assertFalse(self.system.is_time_varying)
    
    def test_is_time_varying_property_override(self):
        """is_time_varying can be overridden."""
        self.assertTrue(self.time_varying.is_time_varying)
    
    # =========================================================================
    # Test __repr__ Method
    # =========================================================================
    
    def test_repr(self):
        """String representation contains key info."""
        repr_str = repr(self.system)
        
        self.assertIn('SimpleDiscreteSystem', repr_str)
        self.assertIn('nx=2', repr_str)
        self.assertIn('nu=1', repr_str)
        self.assertIn('dt=0.1', repr_str)
    
    # =========================================================================
    # Test Type Safety and Interface Contracts
    # =========================================================================
    
    def test_polymorphic_usage(self):
        """System can be used polymorphically via base class."""
        def check_stability(sys: DiscreteSystemBase):
            """Generic function that works with any discrete system."""
            x_eq = np.zeros(sys.nx)
            u_eq = np.zeros(sys.nu)
            lin = sys.linearize(x_eq, u_eq)
            eigenvalues = np.linalg.eigvals(lin.Ad)
            return np.all(np.abs(eigenvalues) < 1)
        
        is_stable = check_stability(self.system)
        self.assertTrue(is_stable)
    
    def test_subclass_type_check(self):
        """Concrete implementation is instance of base class."""
        self.assertIsInstance(self.system, DiscreteSystemBase)
        self.assertTrue(issubclass(SimpleDiscreteSystem, DiscreteSystemBase))
    
    # =========================================================================
    # Edge Cases and Error Handling
    # =========================================================================
    
    def test_zero_control_equivalent_to_none(self):
        """u=None and u=zeros should give same result."""
        x = np.array([1.0, 2.0])
        u_none = None
        u_zero = np.zeros(self.system.nu)
        
        x_next_none = self.system.step(x, u_none)
        x_next_zero = self.system.step(x, u_zero)
        
        np.testing.assert_array_almost_equal(x_next_none, x_next_zero)
    
    def test_simulate_zero_steps(self):
        """Simulate with n_steps=0 returns only initial state."""
        x0 = np.array([1.0, 1.0])
        result = self.system.simulate(x0, n_steps=0)
        
        self.assertEqual(result.states.shape[1], 1)
        np.testing.assert_array_equal(result.states[:, 0], x0)
    
    def test_stability_check(self):
        """System with |eigenvalues| < 1 is stable."""
        x_eq = np.zeros(2)
        u_eq = np.zeros(1)
        lin = self.system.linearize(x_eq, u_eq)
        
        eigenvalues = np.linalg.eigvals(lin.Ad)
        is_stable = np.all(np.abs(eigenvalues) < 1)
        
        # 0.9 < 1, so system is stable
        self.assertTrue(is_stable)


if __name__ == '__main__':
    unittest.main()
