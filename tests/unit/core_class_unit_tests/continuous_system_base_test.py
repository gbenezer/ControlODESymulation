# Copyright (C) 2025 Gil Benezer
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""
Unit Tests for ContinuousSystemBase
===================================

Tests the abstract base class interface for continuous-time systems.
"""

import unittest
from abc import ABC
from typing import Optional

import numpy as np

from src.types.core import ControlVector, StateVector
from src.types.linearization import ContinuousLinearization
from src.types.trajectories import SimulationResult

# Import after src.types
import sys
sys.path.insert(0, '/home/claude')
from continuous_system_base import ContinuousSystemBase


class SimpleContinuousSystem(ContinuousSystemBase):
    """
    Concrete implementation for testing.
    
    Implements: dx/dt = -x + u (stable linear system)
    """
    
    def __init__(self, nx=2, nu=1):
        self.nx = nx
        self.nu = nu
        self.ny = nx
        
    def __call__(
        self,
        x: StateVector,
        u: Optional[ControlVector] = None,
        t: float = 0.0
    ) -> StateVector:
        """dx/dt = -x + u"""
        if u is None:
            u = np.zeros(self.nu)
        
        # Handle batched inputs
        if x.ndim == 2:
            u = u if u.ndim == 2 else u.reshape(-1, 1)
        
        return -x + u if u is not None else -x
    
    def integrate(
        self,
        x0: StateVector,
        u=None,
        t_span=(0.0, 1.0),
        dt=None,
        method="RK45",
        **kwargs
    ) -> SimulationResult:
        """Simple Euler integration for testing."""
        t_start, t_end = t_span
        dt = dt if dt is not None else 0.01
        
        t_points = np.arange(t_start, t_end + dt, dt)
        n_steps = len(t_points)
        
        states = np.zeros((self.nx, n_steps))
        states[:, 0] = x0
        
        for i in range(1, n_steps):
            t = t_points[i-1]
            
            # Evaluate control
            if u is None:
                u_val = None
            elif callable(u):
                u_val = u(t)
            else:
                u_val = u
            
            # Euler step
            dxdt = self(states[:, i-1], u_val, t)
            states[:, i] = states[:, i-1] + dt * dxdt
        
        return SimulationResult(
            time=t_points,
            states=states,
            controls=None,
            metadata={'method': method, 'dt': dt}
        )
    
    def linearize(
        self,
        x_eq: StateVector,
        u_eq: Optional[ControlVector] = None
    ) -> ContinuousLinearization:
        """Linearization: A = -I, B = I"""
        A = -np.eye(self.nx)
        B = np.ones((self.nx, self.nu))
        
        return ContinuousLinearization(
            A=A,
            B=B,
            x_eq=x_eq,
            u_eq=u_eq if u_eq is not None else np.zeros(self.nu)
        )


class TimeVaryingSystem(ContinuousSystemBase):
    """Time-varying system for testing."""
    
    def __init__(self):
        self.nx = 1
        self.nu = 1
        
    def __call__(self, x, u=None, t=0.0):
        """dx/dt = -t*x + u"""
        u = u if u is not None else np.array([0.0])
        return -t * x + u
    
    def integrate(self, x0, u=None, t_span=(0.0, 1.0), dt=None, **kwargs):
        # Simplified - just return mock result
        t = np.linspace(t_span[0], t_span[1], 100)
        states = np.outer(x0, np.exp(-t))
        return SimulationResult(time=t, states=states, controls=None, metadata={})
    
    def linearize(self, x_eq, u_eq=None):
        # Time-varying, linearization not well-defined
        raise NotImplementedError("Linearization not defined for time-varying")
    
    @property
    def is_time_varying(self):
        return True


class TestContinuousSystemBase(unittest.TestCase):
    """Test suite for ContinuousSystemBase abstract class."""
    
    def setUp(self):
        """Create test systems."""
        self.system = SimpleContinuousSystem(nx=2, nu=1)
        self.time_varying = TimeVaryingSystem()
    
    # =========================================================================
    # Test Abstract Class Properties
    # =========================================================================
    
    def test_cannot_instantiate_abstract_class(self):
        """ContinuousSystemBase cannot be instantiated directly."""
        with self.assertRaises(TypeError):
            system = ContinuousSystemBase()
    
    def test_concrete_class_instantiation(self):
        """Concrete implementation can be instantiated."""
        system = SimpleContinuousSystem()
        self.assertIsInstance(system, ContinuousSystemBase)
    
    def test_missing_abstract_method_raises_error(self):
        """Class missing abstract methods cannot be instantiated."""
        
        class IncompleteSystem(ContinuousSystemBase):
            def __call__(self, x, u=None, t=0.0):
                return x
            # Missing integrate() and linearize()
        
        with self.assertRaises(TypeError):
            system = IncompleteSystem()
    
    # =========================================================================
    # Test __call__ Method (Dynamics Evaluation)
    # =========================================================================
    
    def test_call_with_control(self):
        """Evaluate dynamics with control input."""
        x = np.array([1.0, 2.0])
        u = np.array([0.5])
        
        dxdt = self.system(x, u)
        
        self.assertEqual(dxdt.shape, (2,))
        # Expected: -x + u = [-1, -2] + [0.5, 0.5] = [-0.5, -1.5]
        expected = np.array([-0.5, -1.5])
        np.testing.assert_array_almost_equal(dxdt, expected)
    
    def test_call_without_control(self):
        """Evaluate autonomous dynamics (u=None)."""
        x = np.array([1.0, 2.0])
        
        dxdt = self.system(x, u=None)
        
        # Expected: -x = [-1, -2]
        expected = np.array([-1.0, -2.0])
        np.testing.assert_array_almost_equal(dxdt, expected)
    
    def test_call_with_time(self):
        """Time parameter is accepted (may be ignored for time-invariant)."""
        x = np.array([1.0, 2.0])
        u = np.array([0.5])
        
        dxdt = self.system(x, u, t=5.0)
        
        # Time-invariant system ignores t
        expected = np.array([-0.5, -1.5])
        np.testing.assert_array_almost_equal(dxdt, expected)
    
    def test_call_batch_evaluation(self):
        """Evaluate dynamics for multiple states simultaneously."""
        x_batch = np.array([[1.0, 2.0, 3.0],
                           [0.5, 1.5, 2.5]])  # 3 states
        u_batch = np.array([[0.5, 0.3, 0.1]])  # 3 controls
        
        dxdt_batch = self.system(x_batch, u_batch)
        
        self.assertEqual(dxdt_batch.shape, (2, 3))
    
    def test_time_varying_system(self):
        """Time-varying system uses time parameter."""
        x = np.array([1.0])
        u = np.array([0.0])
        
        dxdt_t0 = self.time_varying(x, u, t=0.0)
        dxdt_t1 = self.time_varying(x, u, t=1.0)
        
        # dx/dt = -t*x, so should be different
        self.assertNotEqual(dxdt_t0[0], dxdt_t1[0])
    
    # =========================================================================
    # Test integrate() Method
    # =========================================================================
    
    def test_integrate_basic(self):
        """Basic integration test."""
        x0 = np.array([1.0, 1.0])
        result = self.system.integrate(x0, u=None, t_span=(0.0, 1.0), dt=0.01)
        
        self.assertIsInstance(result, SimulationResult)
        self.assertIsNotNone(result.time)
        self.assertIsNotNone(result.states)
    
    def test_integrate_constant_control(self):
        """Integrate with constant control."""
        x0 = np.array([1.0, 1.0])
        u = np.array([0.5])
        
        result = self.system.integrate(x0, u, t_span=(0.0, 1.0), dt=0.01)
        
        self.assertEqual(result.states.shape[0], 2)  # nx
        self.assertGreater(result.states.shape[1], 1)  # Multiple time steps
    
    def test_integrate_time_varying_control(self):
        """Integrate with time-varying control function."""
        x0 = np.array([1.0, 1.0])
        u_func = lambda t: np.array([np.sin(t)])
        
        result = self.system.integrate(x0, u_func, t_span=(0.0, 1.0), dt=0.01)
        
        self.assertIsInstance(result, SimulationResult)
    
    def test_integrate_custom_time_span(self):
        """Integrate over custom time interval."""
        x0 = np.array([1.0, 1.0])
        
        result = self.system.integrate(x0, t_span=(2.0, 5.0), dt=0.1)
        
        self.assertAlmostEqual(result.time[0], 2.0)
        self.assertAlmostEqual(result.time[-1], 5.0, places=5)
    
    def test_integrate_result_structure(self):
        """Integration result has correct structure."""
        x0 = np.array([1.0, 1.0])
        result = self.system.integrate(x0, t_span=(0.0, 1.0), dt=0.1)
        
        # Check SimulationResult attributes
        self.assertTrue(hasattr(result, 'time'))
        self.assertTrue(hasattr(result, 'states'))
        self.assertTrue(hasattr(result, 'controls'))
        self.assertTrue(hasattr(result, 'metadata'))
    
    # =========================================================================
    # Test linearize() Method
    # =========================================================================
    
    def test_linearize_at_origin(self):
        """Linearize at origin."""
        x_eq = np.zeros(2)
        u_eq = np.zeros(1)
        
        lin = self.system.linearize(x_eq, u_eq)
        
        self.assertIsInstance(lin, ContinuousLinearization)
        self.assertEqual(lin.A.shape, (2, 2))
        self.assertEqual(lin.B.shape, (2, 1))
    
    def test_linearize_without_control(self):
        """Linearize with u_eq=None (autonomous)."""
        x_eq = np.zeros(2)
        
        lin = self.system.linearize(x_eq, u_eq=None)
        
        self.assertIsNotNone(lin.A)
        self.assertIsNotNone(lin.B)
    
    def test_linearize_equilibrium_stored(self):
        """Linearization stores equilibrium point."""
        x_eq = np.array([1.0, 2.0])
        u_eq = np.array([0.5])
        
        lin = self.system.linearize(x_eq, u_eq)
        
        np.testing.assert_array_equal(lin.x_eq, x_eq)
        np.testing.assert_array_equal(lin.u_eq, u_eq)
    
    def test_linearize_correct_dimensions(self):
        """Linearization matrices have correct dimensions."""
        x_eq = np.zeros(2)
        u_eq = np.zeros(1)
        
        lin = self.system.linearize(x_eq, u_eq)
        
        # A should be (nx, nx), B should be (nx, nu)
        self.assertEqual(lin.A.shape, (self.system.nx, self.system.nx))
        self.assertEqual(lin.B.shape, (self.system.nx, self.system.nu))
    
    def test_linearize_known_system(self):
        """Linearization of known system matches analytical result."""
        x_eq = np.zeros(2)
        u_eq = np.zeros(1)
        
        lin = self.system.linearize(x_eq, u_eq)
        
        # For dx/dt = -x + u: A = -I, B = I
        expected_A = -np.eye(2)
        expected_B = np.ones((2, 1))
        
        np.testing.assert_array_almost_equal(lin.A, expected_A)
        np.testing.assert_array_almost_equal(lin.B, expected_B)
    
    # =========================================================================
    # Test simulate() Method (Concrete Implementation)
    # =========================================================================
    
    def test_simulate_without_controller(self):
        """Simulate without feedback controller (open-loop)."""
        x0 = np.array([1.0, 1.0])
        
        result = self.system.simulate(x0, t_span=(0.0, 1.0), dt=0.01)
        
        self.assertIsInstance(result, SimulationResult)
    
    # =========================================================================
    # Test Properties
    # =========================================================================
    
    def test_is_continuous_property(self):
        """is_continuous returns True."""
        self.assertTrue(self.system.is_continuous)
    
    def test_is_discrete_property(self):
        """is_discrete returns False."""
        self.assertFalse(self.system.is_discrete)
    
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
        
        self.assertIn('SimpleContinuousSystem', repr_str)
        self.assertIn('nx=2', repr_str)
        self.assertIn('nu=1', repr_str)
    
    # =========================================================================
    # Test Type Safety and Interface Contracts
    # =========================================================================
    
    def test_polymorphic_usage(self):
        """System can be used polymorphically via base class."""
        def analyze_system(sys: ContinuousSystemBase):
            """Generic function that works with any continuous system."""
            x_eq = np.zeros(sys.nx)
            u_eq = np.zeros(sys.nu)
            lin = sys.linearize(x_eq, u_eq)
            return lin.A
        
        A = analyze_system(self.system)
        self.assertEqual(A.shape, (2, 2))
    
    def test_subclass_type_check(self):
        """Concrete implementation is instance of base class."""
        self.assertIsInstance(self.system, ContinuousSystemBase)
        self.assertTrue(issubclass(SimpleContinuousSystem, ContinuousSystemBase))
    
    # =========================================================================
    # Edge Cases and Error Handling
    # =========================================================================
    
    def test_zero_control_equivalent_to_none(self):
        """u=None and u=zeros should give same result."""
        x = np.array([1.0, 2.0])
        u_none = None
        u_zero = np.zeros(self.system.nu)
        
        dxdt_none = self.system(x, u_none)
        dxdt_zero = self.system(x, u_zero)
        
        # May not be identical due to implementation, but should be close
        np.testing.assert_array_almost_equal(dxdt_none, dxdt_zero)


if __name__ == '__main__':
    unittest.main()
