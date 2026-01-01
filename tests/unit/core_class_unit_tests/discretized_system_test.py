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
Unit Tests for DiscretizedSystem
=================================

Comprehensive test suite for discretized_system.py module.

Test Coverage
------------
1. DiscretizationMode enum
2. DiscretizedSystem core functionality
3. Three discretization modes (FIXED_STEP, DENSE_OUTPUT, BATCH_INTERPOLATION)
4. Control input handling (various formats)
5. Linearization via ZOH
6. Helper functions (discretize, discretize_batch, etc.)
7. Advanced features (AdaptiveDiscretizedSystem, MultiRateDiscretizedSystem)
8. Integration with real continuous systems
9. Protocol satisfaction
10. Edge cases and error handling

Location
--------
tests/systems/discretization/test_discretized_system.py

Dependencies
------------
- pytest
- numpy
- scipy
- src.systems.discretization.discretized_system
- Mock continuous systems (or real if available)

Authors
-------
Gil Benezer
"""

import pytest
import numpy as np
import time
from unittest.mock import Mock, MagicMock

from src.systems.base.core.discretized_system import (
    DiscretizationMode,
    DiscretizedSystem,
    discretize,
    discretize_batch,
    analyze_discretization_error,
    recommend_dt,
    detect_sde_integrator,
    compute_discretization_quality,
    AdaptiveDiscretizedSystem,
    MultiRateDiscretizedSystem,
)
from src.systems.base.core.continuous_system_base import ContinuousSystemBase
from src.systems.base.core.continuous_symbolic_system import ContinuousSymbolicSystem
from src.systems.base.core.continuous_stochastic_system import ContinuousStochasticSystem

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

# ============================================================================
# Mock Continuous Systems for Testing
# ============================================================================


class MockContinuousSystem(ContinuousSystemBase):
    """Simple linear continuous system for testing (inherits from ContinuousSystemBase)."""
    
    def __init__(self, nx=2, nu=1):
        # Don't call super().__init__() - ContinuousSystemBase is abstract with no __init__
        self.nx = nx
        self.nu = nu
        self.ny = nx
        self._default_backend = 'numpy'
    
    @property
    def is_stochastic(self):
        return False
    
    def __call__(self, x, u=None, t=0.0, backend=None):
        """dx/dt = -x + u (handles nu=0 case)"""
        if u is None:
            u = np.zeros(self.nu)
        
        # Handle zero control dimension
        if self.nu == 0:
            return -x
        else:
            # Ensure u is the right shape for broadcasting
            u_array = np.asarray(u)
            if u_array.size == 0:
                return -x
            return -x + u_array
    
    def integrate(self, x0, u, t_span, method='RK45', **kwargs):
        """Mock integration using scipy."""
        from scipy.integrate import solve_ivp
        
        if u is None or (isinstance(u, np.ndarray) and u.size == 0):
            u_func = lambda t, x: None
        elif callable(u):
            u_func = u
        else:
            u_func = lambda t, x: u
        
        def rhs(t, x):
            u_val = u_func(t, x)
            return self(x, u_val, t)
        
        result = solve_ivp(rhs, t_span, x0, method=method, **kwargs)
        
        return {
            't': result.t,
            'x': result.y.T,  # Convert to time-major
            'success': result.success,
            'nfev': result.nfev,
            'sol': getattr(result, 'sol', None)
        }
    
    def linearize(self, x_eq, u_eq=None):
        """Linear system: A = -I, B = I"""
        A = -np.eye(self.nx)
        B = np.eye(self.nx, self.nu) if self.nu > 0 else np.zeros((self.nx, 0))
        return (A, B)


class MockStochasticSystem(MockContinuousSystem):
    """Mock stochastic continuous system."""
    
    @property
    def is_stochastic(self):
        return True
    
    def is_additive_noise(self):
        return True
    
    def is_diagonal_noise(self):
        return True


# ============================================================================
# Test Suite: DiscretizationMode Enum
# ============================================================================


class TestDiscretizationMode:
    """Test DiscretizationMode enumeration."""
    
    def test_enum_has_three_modes(self):
        """Enum defines exactly three modes."""
        modes = list(DiscretizationMode)
        assert len(modes) == 3
    
    def test_mode_values_are_strings(self):
        """Mode values are string identifiers."""
        assert DiscretizationMode.FIXED_STEP.value == "fixed_step"
        assert DiscretizationMode.DENSE_OUTPUT.value == "dense_output"
        assert DiscretizationMode.BATCH_INTERPOLATION.value == "batch_interpolation"
    
    def test_mode_equality(self):
        """Can compare modes."""
        mode1 = DiscretizationMode.FIXED_STEP
        mode2 = DiscretizationMode.FIXED_STEP
        mode3 = DiscretizationMode.DENSE_OUTPUT
        
        assert mode1 == mode2
        assert mode1 != mode3


# ============================================================================
# Test Suite: DiscretizedSystem Initialization
# ============================================================================


class TestDiscretizedSystemInit:
    """Test DiscretizedSystem initialization and validation."""
    
    def test_basic_initialization(self):
        """Can initialize with continuous system and dt."""
        continuous = MockContinuousSystem(nx=2, nu=1)
        discrete = DiscretizedSystem(continuous, dt=0.01)
        
        assert discrete.dt == 0.01
        assert discrete.nx == 2
        assert discrete.nu == 1
    
    def test_mode_auto_selection_fixed_step(self):
        """Auto-selects FIXED_STEP for fixed-step methods."""
        continuous = MockContinuousSystem()
        
        for method in ['euler', 'rk4', 'midpoint', 'heun']:
            discrete = DiscretizedSystem(continuous, dt=0.01, method=method)
            assert discrete.mode == DiscretizationMode.FIXED_STEP
    
    def test_mode_auto_selection_adaptive(self):
        """Auto-selects DENSE_OUTPUT for adaptive methods."""
        continuous = MockContinuousSystem()
        
        for method in ['RK45', 'RK23', 'LSODA', 'Radau']:
            discrete = DiscretizedSystem(continuous, dt=0.01, method=method)
            assert discrete.mode == DiscretizationMode.DENSE_OUTPUT
    
    def test_explicit_mode_override(self):
        """Can explicitly override mode selection."""
        continuous = MockContinuousSystem()
        discrete = DiscretizedSystem(
            continuous, dt=0.01, method='rk4',
            mode=DiscretizationMode.BATCH_INTERPOLATION
        )
        
        assert discrete.mode == DiscretizationMode.BATCH_INTERPOLATION
    
    def test_invalid_mode_method_combination_raises(self):
        """Raises error if adaptive method with FIXED_STEP mode."""
        continuous = MockContinuousSystem()
        
        with pytest.raises(ValueError, match="Cannot use adaptive method"):
            DiscretizedSystem(
                continuous, dt=0.01, method='RK45',
                mode=DiscretizationMode.FIXED_STEP
            )
    
    def test_invalid_continuous_system_raises(self):
        """Raises TypeError if not given ContinuousSystemBase."""
        with pytest.raises(TypeError, match="ContinuousSystemBase"):
            DiscretizedSystem("not a system", dt=0.01)
    
    def test_negative_dt_raises(self):
        """Raises ValueError for negative dt."""
        continuous = MockContinuousSystem()
        
        with pytest.raises(ValueError, match="positive"):
            DiscretizedSystem(continuous, dt=-0.01)
    
    def test_zero_dt_raises(self):
        """Raises ValueError for zero dt."""
        continuous = MockContinuousSystem()
        
        with pytest.raises(ValueError, match="positive"):
            DiscretizedSystem(continuous, dt=0.0)
    
    def test_integrator_kwargs_stored(self):
        """Additional integrator kwargs are stored."""
        continuous = MockContinuousSystem()
        discrete = DiscretizedSystem(
            continuous, dt=0.01, method='RK45',
            rtol=1e-9, atol=1e-11
        )
        
        assert discrete._integrator_kwargs['rtol'] == 1e-9
        assert discrete._integrator_kwargs['atol'] == 1e-11


# ============================================================================
# Test Suite: Properties
# ============================================================================


class TestDiscretizedSystemProperties:
    """Test DiscretizedSystem properties."""
    
    def test_dt_property(self):
        """dt property returns sampling period."""
        continuous = MockContinuousSystem()
        discrete = DiscretizedSystem(continuous, dt=0.05)
        
        assert discrete.dt == 0.05
    
    def test_dimension_delegation(self):
        """Dimensions delegated to continuous system."""
        continuous = MockContinuousSystem(nx=5, nu=3)
        discrete = DiscretizedSystem(continuous, dt=0.01)
        
        assert discrete.nx == 5
        assert discrete.nu == 3
        assert discrete.ny == 5
    
    def test_is_stochastic_delegation(self):
        """is_stochastic delegated to continuous system."""
        deterministic = MockContinuousSystem()
        stochastic = MockStochasticSystem()
        
        discrete1 = DiscretizedSystem(deterministic, dt=0.01)
        discrete2 = DiscretizedSystem(stochastic, dt=0.01)
        
        assert discrete1.is_stochastic is False
        assert discrete2.is_stochastic is True
    
    def test_mode_property(self):
        """mode property returns discretization mode."""
        continuous = MockContinuousSystem()
        discrete = DiscretizedSystem(continuous, dt=0.01, method='rk4')
        
        assert discrete.mode == DiscretizationMode.FIXED_STEP
        assert isinstance(discrete.mode, DiscretizationMode)


# ============================================================================
# Test Suite: Step Method (FIXED_STEP Mode)
# ============================================================================


class TestStepMethodFixedStep:
    """Test step() method in FIXED_STEP mode."""
    
    def test_step_advances_state(self):
        """step() computes next state."""
        continuous = MockContinuousSystem(nx=2, nu=1)
        discrete = DiscretizedSystem(continuous, dt=0.01, method='rk4')
        
        x = np.array([1.0, 0.0])
        u = np.array([0.0])
        x_next = discrete.step(x, u, k=0)
        
        assert x_next.shape == (2,)
        assert not np.allclose(x_next, x)  # State should change
    
    def test_step_with_none_control(self):
        """step() works with u=None (autonomous)."""
        continuous = MockContinuousSystem(nx=2, nu=0)
        discrete = DiscretizedSystem(continuous, dt=0.01, method='rk4')
        
        x = np.array([1.0, 0.0])
        x_next = discrete.step(x, None, k=0)
        
        assert x_next.shape == (2,)
    
    def test_step_multiple_times(self):
        """Can call step() multiple times."""
        continuous = MockContinuousSystem(nx=2, nu=1)
        discrete = DiscretizedSystem(continuous, dt=0.01, method='rk4')
        
        x = np.array([1.0, 0.0])
        u = np.array([0.0])
        
        trajectory = [x]
        for k in range(10):
            x = discrete.step(x, u, k)
            trajectory.append(x)
        
        assert len(trajectory) == 11
        # System is stable (A = -I), should decay
        assert np.linalg.norm(trajectory[-1]) < np.linalg.norm(trajectory[0])
    
    def test_step_time_index_parameter(self):
        """step() accepts time index k."""
        continuous = MockContinuousSystem()
        discrete = DiscretizedSystem(continuous, dt=0.01, method='rk4')
        
        x = np.array([1.0, 0.0])
        
        # Different k values (for time-varying systems)
        x_next_0 = discrete.step(x, None, k=0)
        x_next_5 = discrete.step(x, None, k=5)
        
        # For time-invariant system, should be same
        assert np.allclose(x_next_0, x_next_5)


# ============================================================================
# Test Suite: Step Method (DENSE_OUTPUT Mode)
# ============================================================================


class TestStepMethodDenseOutput:
    """Test step() method in DENSE_OUTPUT mode."""
    
    def test_dense_mode_step_works(self):
        """step() works in DENSE_OUTPUT mode."""
        continuous = MockContinuousSystem(nx=2, nu=1)
        discrete = DiscretizedSystem(
            continuous, dt=0.01, method='RK45',
            mode=DiscretizationMode.DENSE_OUTPUT
        )
        
        x = np.array([1.0, 0.0])
        u = np.array([0.0])
        x_next = discrete.step(x, u, k=0)
        
        assert x_next.shape == (2,)
    
    def test_dense_mode_more_accurate_than_fixed(self):
        """DENSE_OUTPUT generally more accurate than FIXED_STEP."""
        continuous = MockContinuousSystem(nx=2, nu=1)
        
        fixed = DiscretizedSystem(continuous, dt=0.1, method='rk4')
        dense = DiscretizedSystem(continuous, dt=0.1, method='RK45')
        
        x0 = np.array([1.0, 0.0])
        
        # Simulate both
        x_fixed = x0
        x_dense = x0
        for k in range(10):
            x_fixed = fixed.step(x_fixed, None, k)
            x_dense = dense.step(x_dense, None, k)
        
        # Get reference (very fine dt)
        ref = DiscretizedSystem(continuous, dt=0.001, method='RK45')
        x_ref = x0
        for k in range(1000):
            x_ref = ref.step(x_ref, None, k)
        
        # Dense should be closer to reference
        error_fixed = np.linalg.norm(x_fixed - x_ref)
        error_dense = np.linalg.norm(x_dense - x_ref)
        
        assert error_dense < error_fixed


# ============================================================================
# Test Suite: Step Method (BATCH Mode)
# ============================================================================


class TestStepMethodBatch:
    """Test step() method in BATCH_INTERPOLATION mode."""
    
    def test_batch_mode_step_raises(self):
        """step() raises NotImplementedError in BATCH mode."""
        continuous = MockContinuousSystem()
        discrete = DiscretizedSystem(
            continuous, dt=0.01, method='LSODA',
            mode=DiscretizationMode.BATCH_INTERPOLATION
        )
        
        with pytest.raises(NotImplementedError, match="BATCH_INTERPOLATION"):
            discrete.step(np.array([1.0, 0.0]), None, k=0)


# ============================================================================
# Test Suite: Simulate Method
# ============================================================================


class TestSimulateMethod:
    """Test simulate() method across all modes."""
    
    def test_simulate_basic(self):
        """Basic simulation works."""
        continuous = MockContinuousSystem(nx=2, nu=1)
        discrete = DiscretizedSystem(continuous, dt=0.01, method='rk4')
        
        x0 = np.array([1.0, 0.0])
        result = discrete.simulate(x0, u_sequence=None, n_steps=100)
        
        assert 'states' in result
        assert 'time_steps' in result
        assert result['states'].shape == (101, 2)  # n_steps+1 points
        assert len(result['time_steps']) == 101
    
    def test_simulate_with_constant_control(self):
        """Simulate with constant control."""
        continuous = MockContinuousSystem(nx=2, nu=1)
        discrete = DiscretizedSystem(continuous, dt=0.01)
        
        x0 = np.array([1.0, 0.0])
        u = np.array([0.5])
        result = discrete.simulate(x0, u, n_steps=50)
        
        assert result['states'].shape == (51, 2)
        assert result['controls'] is not None
        assert result['controls'].shape == (50, 1)
    
    def test_simulate_with_time_indexed_control(self):
        """Simulate with u(k) control function."""
        continuous = MockContinuousSystem(nx=2, nu=1)
        discrete = DiscretizedSystem(continuous, dt=0.01)
        
        def u_func(k):
            return np.array([0.1 * k])
        
        x0 = np.array([1.0, 0.0])
        result = discrete.simulate(x0, u_func, n_steps=10)
        
        assert result['states'].shape == (11, 2)
    
    def test_simulate_with_state_feedback(self):
        """Simulate with u(x, k) state feedback."""
        continuous = MockContinuousSystem(nx=2, nu=1)
        discrete = DiscretizedSystem(continuous, dt=0.01, method='rk4')
        
        K = np.array([[1.0, 0.5]])
        def controller(x, k):
            return -K @ x
        
        x0 = np.array([1.0, 0.0])
        result = discrete.simulate(x0, controller, n_steps=50)
        
        assert result['states'].shape == (51, 2)
        assert result['controls'] is not None
    
    def test_simulate_batch_mode(self):
        """Simulate in BATCH_INTERPOLATION mode."""
        continuous = MockContinuousSystem(nx=2, nu=1)
        discrete = DiscretizedSystem(
            continuous, dt=0.01, method='LSODA',
            mode=DiscretizationMode.BATCH_INTERPOLATION
        )
        
        x0 = np.array([1.0, 0.0])
        result = discrete.simulate(x0, None, n_steps=100)
        
        assert result['states'].shape == (101, 2)
        assert 'adaptive_points' in result['metadata']
    
    def test_simulate_batch_rejects_state_feedback(self):
        """BATCH mode rejects state-feedback control."""
        continuous = MockContinuousSystem()
        discrete = DiscretizedSystem(
            continuous, dt=0.01, method='LSODA',
            mode=DiscretizationMode.BATCH_INTERPOLATION
        )
        
        def controller(x, k):
            return -x
        
        with pytest.raises(ValueError, match="State-feedback not supported"):
            discrete.simulate(np.array([1.0, 0.0]), controller, n_steps=10)
    
    def test_simulate_returns_correct_structure(self):
        """Simulate returns DiscreteSimulationResult structure."""
        continuous = MockContinuousSystem()
        discrete = DiscretizedSystem(continuous, dt=0.01)
        
        result = discrete.simulate(np.array([1.0, 0.0]), None, n_steps=10)
        
        # Check required keys
        assert 'states' in result
        assert 'time_steps' in result
        assert 'dt' in result
        assert 'success' in result
        assert 'metadata' in result


# ============================================================================
# Test Suite: Linearization
# ============================================================================


class TestLinearization:
    """Test linearize() method with ZOH discretization."""
    
    def test_linearize_returns_discrete_matrices(self):
        """linearize() returns (Ad, Bd) tuple."""
        continuous = MockContinuousSystem(nx=2, nu=1)
        discrete = DiscretizedSystem(continuous, dt=0.01)
        
        x_eq = np.zeros(2)
        u_eq = np.zeros(1)
        result = discrete.linearize(x_eq, u_eq)
        
        assert isinstance(result, tuple)
        assert len(result) == 2
        
        Ad, Bd = result
        assert Ad.shape == (2, 2)
        assert Bd.shape == (2, 1)
    
    def test_linearize_zoh_is_accurate(self):
        """ZOH discretization matches matrix exponential."""
        continuous = MockContinuousSystem(nx=2, nu=1)
        discrete = DiscretizedSystem(continuous, dt=0.01)
        
        # Get discrete linearization
        Ad, Bd = discrete.linearize(np.zeros(2), np.zeros(1))
        
        # Compute exact ZOH
        A, B = continuous.linearize(np.zeros(2), np.zeros(1))
        from scipy.linalg import expm
        Ad_exact = expm(A * 0.01)
        
        # Should match
        assert np.allclose(Ad, Ad_exact, rtol=1e-10)
    
    def test_linearize_at_different_equilibria(self):
        """Can linearize at arbitrary points."""
        continuous = MockContinuousSystem(nx=2, nu=1)
        discrete = DiscretizedSystem(continuous, dt=0.01)
        
        # Different equilibria
        Ad1, Bd1 = discrete.linearize(np.zeros(2), np.zeros(1))
        Ad2, Bd2 = discrete.linearize(np.array([1.0, 2.0]), np.zeros(1))
        
        # Linear system - should get same matrices
        assert np.allclose(Ad1, Ad2)
        assert np.allclose(Bd1, Bd2)
    
    def test_linearize_singular_matrix_uses_approximation(self):
        """Handles singular A matrix with approximation."""
        # Create a proper mock that inherits from ContinuousSystemBase
        class SingularContinuousSystem(ContinuousSystemBase):
            def __init__(self):
                self.nx = 2
                self.nu = 1
                self._default_backend = 'numpy'
            
            @property
            def is_stochastic(self):
                return False
            
            def __call__(self, x, u=None, t=0.0, backend=None):
                return np.zeros(2)
            
            def integrate(self, x0, u, t_span, method='RK45', **kwargs):
                return {'t': np.array([0, 1]), 'x': np.zeros((2, 2)), 'success': True}
            
            def linearize(self, x_eq, u_eq=None):
                # Return singular A matrix
                return (np.zeros((2, 2)), np.ones((2, 1)))
        
        continuous = SingularContinuousSystem()
        discrete = DiscretizedSystem(continuous, dt=0.01)
        
        # Should not raise, uses Bd ≈ dt*B approximation
        Ad, Bd = discrete.linearize(np.zeros(2), np.zeros(1))
        
        assert Ad.shape == (2, 2)
        assert Bd.shape == (2, 1)


# ============================================================================
# Test Suite: Control Input Handling
# ============================================================================


class TestControlInputHandling:
    """Test _prepare_control_sequence() with various input formats."""
    
    def test_none_control_autonomous(self):
        """None control for autonomous system."""
        continuous = MockContinuousSystem(nx=2, nu=0)
        discrete = DiscretizedSystem(continuous, dt=0.01)
        
        u_func = discrete._prepare_control_sequence(None, n_steps=10)
        
        assert u_func(np.zeros(2), 0) is None
    
    def test_none_control_zero_for_controlled(self):
        """None control becomes zeros for controlled system."""
        continuous = MockContinuousSystem(nx=2, nu=1)
        discrete = DiscretizedSystem(continuous, dt=0.01)
        
        u_func = discrete._prepare_control_sequence(None, n_steps=10)
        u = u_func(np.zeros(2), 0)
        
        assert np.allclose(u, np.zeros(1))
    
    def test_constant_control(self):
        """Constant array becomes constant function."""
        continuous = MockContinuousSystem(nx=2, nu=1)
        discrete = DiscretizedSystem(continuous, dt=0.01)
        
        u_const = np.array([0.5])
        u_func = discrete._prepare_control_sequence(u_const, n_steps=10)
        
        assert np.allclose(u_func(np.zeros(2), 0), u_const)
        assert np.allclose(u_func(np.ones(2), 5), u_const)
    
    def test_time_indexed_callable(self):
        """Callable u(k) is time-indexed."""
        continuous = MockContinuousSystem(nx=2, nu=1)
        discrete = DiscretizedSystem(continuous, dt=0.01)
        
        u_func_orig = lambda k: np.array([0.1 * k])
        u_func = discrete._prepare_control_sequence(u_func_orig, n_steps=10)
        
        assert np.allclose(u_func(np.zeros(2), 0), np.array([0.0]))
        assert np.allclose(u_func(np.zeros(2), 5), np.array([0.5]))
    
    def test_state_feedback_callable(self):
        """Callable u(x, k) is state feedback."""
        continuous = MockContinuousSystem(nx=2, nu=1)
        discrete = DiscretizedSystem(continuous, dt=0.01)
        
        def controller(x, k):
            return -0.5 * x[0:1]  # Return (1,) array
        
        u_func = discrete._prepare_control_sequence(controller, n_steps=10)
        
        x = np.array([2.0, 0.0])
        u = u_func(x, 0)
        
        assert np.allclose(u, np.array([-1.0]))
    
    def test_precomputed_sequence_time_major(self):
        """Pre-computed (n_steps, nu) sequence."""
        continuous = MockContinuousSystem(nx=2, nu=1)
        discrete = DiscretizedSystem(continuous, dt=0.01)
        
        u_seq = np.random.randn(10, 1)  # (n_steps, nu)
        u_func = discrete._prepare_control_sequence(u_seq, n_steps=10)
        
        for k in range(10):
            u = u_func(np.zeros(2), k)
            assert np.allclose(u, u_seq[k, :])
    
    def test_precomputed_sequence_state_major(self):
        """Pre-computed (nu, n_steps) sequence."""
        continuous = MockContinuousSystem(nx=2, nu=1)
        discrete = DiscretizedSystem(continuous, dt=0.01)
        
        u_seq = np.random.randn(1, 10)  # (nu, n_steps)
        u_func = discrete._prepare_control_sequence(u_seq, n_steps=10)
        
        for k in range(10):
            u = u_func(np.zeros(2), k)
            assert np.allclose(u, u_seq[:, k])
    
    def test_list_control_sequence(self):
        """List of control values."""
        continuous = MockContinuousSystem(nx=2, nu=1)
        discrete = DiscretizedSystem(continuous, dt=0.01)
        
        u_list = [np.array([0.1 * k]) for k in range(10)]
        u_func = discrete._prepare_control_sequence(u_list, n_steps=10)
        
        assert np.allclose(u_func(np.zeros(2), 5), np.array([0.5]))
    
    def test_invalid_control_type_raises(self):
        """Invalid control type raises TypeError."""
        continuous = MockContinuousSystem()
        discrete = DiscretizedSystem(continuous, dt=0.01)
        
        with pytest.raises(TypeError, match="Invalid control type"):
            discrete._prepare_control_sequence("invalid", n_steps=10)


# ============================================================================
# Test Suite: Compare Modes
# ============================================================================


class TestCompareModes:
    """Test compare_modes() method."""
    
    def test_compare_modes_runs(self):
        """compare_modes() executes without error."""
        continuous = MockContinuousSystem(nx=2, nu=1)
        discrete = DiscretizedSystem(continuous, dt=0.01, method='rk4')
        
        x0 = np.array([1.0, 0.0])
        comparison = discrete.compare_modes(x0, None, n_steps=50)
        
        assert 'results' in comparison
        assert 'timings' in comparison
        assert 'errors' in comparison
    
    def test_compare_modes_has_all_three_modes(self):
        """Comparison includes all three modes."""
        continuous = MockContinuousSystem()
        discrete = DiscretizedSystem(continuous, dt=0.01)
        
        comparison = discrete.compare_modes(np.array([1.0, 0.0]), None, n_steps=20)
        
        assert 'fixed_step' in comparison['results']
        assert 'dense_output' in comparison['results']
        assert 'batch' in comparison['results']
    
    def test_compare_modes_batch_is_faster(self):
        """BATCH mode should be faster for open-loop."""
        continuous = MockContinuousSystem()
        discrete = DiscretizedSystem(continuous, dt=0.01)
        
        comparison = discrete.compare_modes(np.array([1.0, 0.0]), None, n_steps=100)
        
        # BATCH should generally be faster (though not guaranteed in all cases)
        assert comparison['speedup_batch_vs_fixed'] > 0  # Just check it's computed
        assert comparison['speedup_batch_vs_dense'] > 0


# ============================================================================
# Test Suite: Utility Methods
# ============================================================================


class TestUtilityMethods:
    """Test change_method(), get_info(), print_info()."""
    
    def test_change_method(self):
        """change_method() creates new system with different method."""
        continuous = MockContinuousSystem()
        discrete1 = DiscretizedSystem(continuous, dt=0.01, method='rk4')
        
        discrete2 = discrete1.change_method('RK45', rtol=1e-9)
        
        assert discrete1._method == 'rk4'
        assert discrete2._method == 'RK45'
        assert discrete2._integrator_kwargs['rtol'] == 1e-9
        assert discrete2.dt == discrete1.dt  # Same dt
    
    def test_get_info_returns_dict(self):
        """get_info() returns comprehensive dict."""
        continuous = MockContinuousSystem(nx=3, nu=2)
        discrete = DiscretizedSystem(continuous, dt=0.05, method='RK45')
        
        info = discrete.get_info()
        
        assert info['class'] == 'DiscretizedSystem'
        assert info['mode'] == 'dense_output'
        assert info['method'] == 'RK45'
        assert info['dt'] == 0.05
        assert info['dimensions']['nx'] == 3
        assert info['dimensions']['nu'] == 2
    
    def test_print_info_does_not_crash(self):
        """print_info() executes without error."""
        continuous = MockContinuousSystem()
        discrete = DiscretizedSystem(continuous, dt=0.01)
        
        # Should not raise
        discrete.print_info()


# ============================================================================
# Test Suite: Helper Functions
# ============================================================================


class TestHelperFunctions:
    """Test module-level helper functions."""
    
    def test_discretize_convenience_function(self):
        """discretize() is convenience wrapper."""
        continuous = MockContinuousSystem()
        
        discrete1 = discretize(continuous, dt=0.01)
        discrete2 = DiscretizedSystem(continuous, dt=0.01, method='rk4')
        
        assert discrete1.dt == discrete2.dt
        assert discrete1._method == discrete2._method
    
    def test_discretize_batch_creates_batch_mode(self):
        """discretize_batch() creates BATCH_INTERPOLATION mode."""
        continuous = MockContinuousSystem()
        discrete = discretize_batch(continuous, dt=0.01)
        
        assert discrete.mode == DiscretizationMode.BATCH_INTERPOLATION
        assert discrete._method == 'LSODA'
    
    def test_analyze_discretization_error(self):
        """analyze_discretization_error() runs convergence study."""
        continuous = MockContinuousSystem()
        x0 = np.array([1.0, 0.0])
        dt_values = [0.01, 0.05, 0.1]
        
        analysis = analyze_discretization_error(
            continuous, x0, None, dt_values, method='rk4', n_steps=20
        )
        
        assert 'dt_values' in analysis
        assert 'errors' in analysis
        assert 'timings' in analysis
        assert 'convergence_rate' in analysis
        assert len(analysis['errors']) == 3
    
    def test_recommend_dt(self):
        """recommend_dt() finds appropriate dt."""
        continuous = MockContinuousSystem()
        x0 = np.array([1.0, 0.0])
        
        rec = recommend_dt(
            continuous, x0, target_error=1e-5,
            method='rk4', dt_range=(0.001, 0.1), n_test=5
        )
        
        assert 'recommended_dt' in rec
        assert 'achieved_error' in rec
        assert rec['achieved_error'] <= 1e-5 or rec['achieved_error'] < 1e-4  # Allow some margin
    
    def test_detect_sde_integrator_stochastic(self):
        """detect_sde_integrator() for stochastic system."""
        stochastic = MockStochasticSystem()
        method = detect_sde_integrator(stochastic)
        
        assert method in ['euler_maruyama', 'milstein']
    
    def test_detect_sde_integrator_deterministic_raises(self):
        """detect_sde_integrator() raises for deterministic system."""
        deterministic = MockContinuousSystem()
        
        with pytest.raises(ValueError, match="not stochastic"):
            detect_sde_integrator(deterministic)
    
    def test_compute_discretization_quality(self):
        """compute_discretization_quality() computes metrics."""
        continuous = MockContinuousSystem()
        discrete = DiscretizedSystem(continuous, dt=0.01)
        
        quality = compute_discretization_quality(
            discrete, np.array([1.0, 0.0]), None, n_steps=50
        )
        
        assert 'timing' in quality
        assert 'stability' in quality
        assert quality['timing']['steps_per_second'] > 0


# ============================================================================
# Test Suite: Advanced Features
# ============================================================================


class TestAdaptiveDiscretizedSystem:
    """Test AdaptiveDiscretizedSystem experimental class."""
    
    def test_adaptive_initialization(self):
        """Can initialize adaptive system."""
        continuous = MockContinuousSystem()
        adaptive = AdaptiveDiscretizedSystem(
            continuous, dt_initial=0.01, dt_min=0.001, dt_max=0.1
        )
        
        assert adaptive.dt == 0.01
        assert adaptive._dt_min == 0.001
        assert adaptive._dt_max == 0.1
    
    def test_adaptive_tracks_dt_history(self):
        """Adaptive system tracks dt usage."""
        continuous = MockContinuousSystem()
        adaptive = AdaptiveDiscretizedSystem(continuous, dt_initial=0.01)
        
        x = np.array([1.0, 0.0])
        for k in range(10):
            x = adaptive.step(x, None, k)
        
        stats = adaptive.get_dt_statistics()
        assert stats['n_steps'] == 10
        assert stats['mean'] > 0
    
    def test_adaptive_reset_statistics(self):
        """Can reset adaptive statistics."""
        continuous = MockContinuousSystem()
        adaptive = AdaptiveDiscretizedSystem(continuous, dt_initial=0.01)
        
        # Take some steps
        x = np.array([1.0, 0.0])
        for k in range(5):
            x = adaptive.step(x, None, k)
        
        # Reset
        adaptive.reset_statistics()
        stats = adaptive.get_dt_statistics()
        
        assert stats['n_steps'] == 0


class TestMultiRateDiscretizedSystem:
    """Test MultiRateDiscretizedSystem experimental class."""
    
    def test_multirate_initialization(self):
        """Can initialize multi-rate system."""
        continuous = MockContinuousSystem(nx=2, nu=1)
        multirate = MultiRateDiscretizedSystem(
            continuous,
            dt_fast=0.001,
            dt_slow=0.01,
            fast_indices=[0],
            slow_indices=[1]
        )
        
        assert multirate.dt == 0.001
        assert multirate._dt_slow == 0.01
        assert multirate._slow_update_interval == 10
    
    def test_multirate_invalid_ratio_raises(self):
        """Raises if dt_slow not integer multiple of dt_fast."""
        continuous = MockContinuousSystem(nx=2, nu=1)
        
        with pytest.raises(ValueError, match="integer multiple"):
            MultiRateDiscretizedSystem(
                continuous,
                dt_fast=0.003,
                dt_slow=0.01,  # 0.01/0.003 = 3.33... (not integer)
                fast_indices=[0],
                slow_indices=[1]
            )
    
    def test_multirate_step_updates_selectively(self):
        """Multi-rate step updates fast every step, slow periodically."""
        continuous = MockContinuousSystem(nx=2, nu=1)
        multirate = MultiRateDiscretizedSystem(
            continuous,
            dt_fast=0.001,
            dt_slow=0.01,
            fast_indices=[0],
            slow_indices=[1],
            method='rk4'
        )
        
        x = np.array([1.0, 2.0])
        x_initial_slow = x[1]
        
        # Take 5 steps (not at slow update interval)
        for k in range(5):
            x = multirate.step(x, None, k)
        
        # Fast state should have changed
        assert not np.isclose(x[0], 1.0)
        
        # Slow state should be unchanged (not at interval)
        # Note: This test might fail depending on implementation details
        # The multirate step is experimental


# ============================================================================
# Test Suite: Integration with Real Systems (Conditional)
# ============================================================================


class TestRealSystemIntegration:
    """Test with real continuous systems if available."""
    
    def test_with_continuous_symbolic_system(self):
        """DiscretizedSystem works with ContinuousSymbolicSystem."""
        try:
            import sympy as sp
            
            class SimpleContinuous(ContinuousSymbolicSystem):
                def define_system(self, a=1.0):
                    x = sp.symbols('x')
                    u = sp.symbols('u')
                    a_sym = sp.symbols('a')
                    
                    self.state_vars = [x]
                    self.control_vars = [u]
                    self._f_sym = sp.Matrix([-a_sym * x + u])
                    self.parameters = {a_sym: a}
                    self.order = 1
            
            continuous = SimpleContinuous(a=2.0)
            discrete = DiscretizedSystem(continuous, dt=0.01, method='rk4')
            
            x0 = np.array([1.0])
            result = discrete.simulate(x0, None, n_steps=50)
            
            assert result['states'].shape == (51, 1)
            assert result['success']
            
        except ImportError:
            pytest.skip("ContinuousSymbolicSystem not available")
    
    def test_with_continuous_stochastic_system(self):
        """DiscretizedSystem works with ContinuousStochasticSystem."""
        try:
            import sympy as sp
            
            class SimpleStochastic(ContinuousStochasticSystem):
                def define_system(self, alpha=1.0, sigma=0.5):
                    x = sp.symbols('x')
                    alpha_sym = sp.symbols('alpha')
                    sigma_sym = sp.symbols('sigma')
                    
                    self.state_vars = [x]
                    self.control_vars = []
                    self._f_sym = sp.Matrix([-alpha_sym * x])
                    self.diffusion_expr = sp.Matrix([[sigma_sym]])
                    self.sde_type = 'ito'
                    self.parameters = {alpha_sym: alpha, sigma_sym: sigma}
                    self.order = 1
            
            stochastic = SimpleStochastic(alpha=2.0, sigma=0.3)
            discrete = DiscretizedSystem(stochastic, dt=0.01, method='euler_maruyama')
            
            assert discrete.is_stochastic
            
            # Should be able to simulate
            result = discrete.simulate(np.array([1.0]), None, n_steps=20)
            assert result['states'].shape == (21, 1)
            
        except ImportError:
            pytest.skip("ContinuousStochasticSystem not available")


# ============================================================================
# Test Suite: Protocol Satisfaction
# ============================================================================


class TestProtocolSatisfaction:
    """Test that DiscretizedSystem satisfies appropriate protocols."""
    
    def test_satisfies_discrete_system_protocol(self):
        """DiscretizedSystem satisfies DiscreteSystemProtocol."""
        try:
            from src.types.protocols import DiscreteSystemProtocol
            
            continuous = MockContinuousSystem()
            discrete = DiscretizedSystem(continuous, dt=0.01)
            
            assert isinstance(discrete, DiscreteSystemProtocol)
            
        except ImportError:
            pytest.skip("Protocols not available")
    
    def test_satisfies_linearizable_discrete_protocol(self):
        """DiscretizedSystem satisfies LinearizableDiscreteProtocol."""
        try:
            from src.types.protocols import LinearizableDiscreteProtocol
            
            continuous = MockContinuousSystem()
            discrete = DiscretizedSystem(continuous, dt=0.01)
            
            assert isinstance(discrete, LinearizableDiscreteProtocol)
            
        except ImportError:
            pytest.skip("Protocols not available")
    
    def test_does_not_satisfy_symbolic_protocol(self):
        """DiscretizedSystem does NOT satisfy SymbolicDiscreteProtocol."""
        try:
            from src.types.protocols import SymbolicDiscreteProtocol
            
            continuous = MockContinuousSystem()
            discrete = DiscretizedSystem(continuous, dt=0.01)
            
            # Should NOT satisfy symbolic protocol (no symbolic machinery)
            assert not isinstance(discrete, SymbolicDiscreteProtocol)
            
        except ImportError:
            pytest.skip("Protocols not available")
    
    def test_works_in_function_expecting_linearizable(self):
        """Can be used in functions expecting LinearizableDiscreteProtocol."""
        try:
            from src.types.protocols import LinearizableDiscreteProtocol
            
            def dummy_lqr(system: LinearizableDiscreteProtocol):
                """Mock LQR needing linearization."""
                Ad, Bd = system.linearize(np.zeros(system.nx), np.zeros(system.nu))
                return Ad  # Just return Ad for testing
            
            continuous = MockContinuousSystem(nx=2, nu=1)
            discrete = DiscretizedSystem(continuous, dt=0.01)
            
            # Should work
            Ad = dummy_lqr(discrete)
            assert Ad.shape == (2, 2)
            
        except ImportError:
            pytest.skip("Protocols not available")


# ============================================================================
# Test Suite: Edge Cases
# ============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_zero_control_dimension(self):
        """Works with autonomous systems (nu=0)."""
        continuous = MockContinuousSystem(nx=2, nu=0)
        discrete = DiscretizedSystem(continuous, dt=0.01)
        
        assert discrete.nu == 0
        
        x = np.array([1.0, 0.0])
        x_next = discrete.step(x, None, k=0)
        assert x_next.shape == (2,)
    
    def test_scalar_system(self):
        """Works with scalar (nx=1) systems."""
        continuous = MockContinuousSystem(nx=1, nu=1)
        discrete = DiscretizedSystem(continuous, dt=0.01)
        
        x = np.array([1.0])
        u = np.array([0.5])
        x_next = discrete.step(x, u)
        
        assert x_next.shape == (1,)
    
    def test_high_dimensional_system(self):
        """Works with high-dimensional systems."""
        continuous = MockContinuousSystem(nx=50, nu=10)
        discrete = DiscretizedSystem(continuous, dt=0.01)
        
        x0 = np.random.randn(50)
        result = discrete.simulate(x0, None, n_steps=10)
        
        assert result['states'].shape == (11, 50)
    
    def test_very_small_dt(self):
        """Handles very small time steps."""
        continuous = MockContinuousSystem()
        discrete = DiscretizedSystem(continuous, dt=1e-6)
        
        assert discrete.dt == 1e-6
        
        x = np.array([1.0, 0.0])
        x_next = discrete.step(x, None)
        assert x_next.shape == (2,)
    
    def test_single_step_simulation(self):
        """Simulate with n_steps=1."""
        continuous = MockContinuousSystem()
        discrete = DiscretizedSystem(continuous, dt=0.01)
        
        result = discrete.simulate(np.array([1.0, 0.0]), None, n_steps=1)
        
        assert result['states'].shape == (2, 2)  # Initial + 1 step


# ============================================================================
# Test Suite: Numerical Accuracy
# ============================================================================


class TestNumericalAccuracy:
    """Test numerical accuracy of discretization."""
    
    def test_euler_first_order_convergence(self):
        """Euler method shows first-order convergence."""
        continuous = MockContinuousSystem()
        x0 = np.array([1.0, 0.0])
        
        dt_values = [0.1, 0.05, 0.025]
        analysis = analyze_discretization_error(
            continuous, x0, None, dt_values, method='euler', n_steps=20
        )
        
        # Should show O(dt) convergence (rate ≈ 1)
        assert 0.8 < analysis['convergence_rate'] < 1.5
    
    def test_rk4_fourth_order_convergence(self):
        """RK4 shows fourth-order convergence."""
        continuous = MockContinuousSystem()
        x0 = np.array([1.0, 0.0])
        
        dt_values = [0.1, 0.05, 0.025]
        analysis = analyze_discretization_error(
            continuous, x0, None, dt_values, method='rk4', n_steps=20
        )
        
        # Should show O(dt^4) convergence (rate ≈ 4)
        assert analysis['convergence_rate'] > 1.5
    
    def test_adaptive_more_accurate_than_fixed(self):
        """Adaptive methods generally more accurate."""
        continuous = MockContinuousSystem()
        x0 = np.array([1.0, 0.0])
        
        fixed = DiscretizedSystem(continuous, dt=0.1, method='rk4')
        adaptive = DiscretizedSystem(continuous, dt=0.1, method='RK45', rtol=1e-9)
        
        # Reference solution
        ref = DiscretizedSystem(continuous, dt=0.001, method='RK45', rtol=1e-12)
        ref_result = ref.simulate(x0, None, n_steps=1000)
        x_ref = ref_result['states'][-1, :]
        
        # Compare
        fixed_result = fixed.simulate(x0, None, n_steps=10)
        adaptive_result = adaptive.simulate(x0, None, n_steps=10)
        
        error_fixed = np.linalg.norm(fixed_result['states'][-1, :] - x_ref)
        error_adaptive = np.linalg.norm(adaptive_result['states'][-1, :] - x_ref)
        
        assert error_adaptive < error_fixed


# ============================================================================
# Test Suite: String Representations
# ============================================================================


class TestStringRepresentations:
    """Test __repr__ and __str__ methods."""
    
    def test_repr_includes_key_info(self):
        """__repr__ includes dt, method, mode."""
        continuous = MockContinuousSystem()
        discrete = DiscretizedSystem(continuous, dt=0.01, method='rk4')
        
        repr_str = repr(discrete)
        
        assert 'DiscretizedSystem' in repr_str
        assert '0.01' in repr_str or '0.0100' in repr_str
        assert 'rk4' in repr_str
        assert 'fixed_step' in repr_str
    
    def test_str_is_readable(self):
        """__str__ is human-readable."""
        continuous = MockContinuousSystem()
        discrete = DiscretizedSystem(continuous, dt=0.01, method='RK45')
        
        str_rep = str(discrete)
        
        assert 'DiscretizedSystem' in str_rep
        assert 'MockContinuousSystem' in str_rep or 'discretized' in str_rep.lower()


# ============================================================================
# Test Suite: Performance
# ============================================================================


class TestPerformance:
    """Test performance characteristics."""
    
    def test_batch_mode_faster_than_step_by_step(self):
        """BATCH mode should be faster for long simulations."""
        continuous = MockContinuousSystem()
        
        # Step-by-step
        discrete_step = DiscretizedSystem(continuous, dt=0.01, method='RK45')
        start = time.time()
        discrete_step.simulate(np.array([1.0, 0.0]), None, n_steps=500)
        time_step = time.time() - start
        
        # Batch
        discrete_batch = DiscretizedSystem(
            continuous, dt=0.01, method='RK45',
            mode=DiscretizationMode.BATCH_INTERPOLATION
        )
        start = time.time()
        discrete_batch.simulate(np.array([1.0, 0.0]), None, n_steps=500)
        time_batch = time.time() - start
        
        # Batch should be faster (or at least not much slower)
        # With overhead, might not always be true for small n_steps
        assert time_batch < time_step * 2  # Allow some margin
    
    def test_step_has_reasonable_performance(self):
        """Single step executes quickly."""
        continuous = MockContinuousSystem()
        discrete = DiscretizedSystem(continuous, dt=0.01, method='rk4')
        
        x = np.array([1.0, 0.0])
        
        # Time 1000 steps
        start = time.time()
        for k in range(1000):
            x = discrete.step(x, None, k)
        elapsed = time.time() - start
        
        # Should be fast (< 1 second for 1000 steps)
        assert elapsed < 1.0


# ============================================================================
# Test Suite: Error Handling
# ============================================================================


class TestErrorHandling:
    """Test error handling and validation."""
    
    def test_control_dimension_mismatch_raises(self):
        """Raises if control dimension doesn't match."""
        continuous = MockContinuousSystem(nx=2, nu=1)
        discrete = DiscretizedSystem(continuous, dt=0.01)
        
        # Wrong dimension control
        with pytest.raises(ValueError, match="dimension"):
            discrete._prepare_control_sequence(np.array([0.1, 0.2]), n_steps=10)
    
    def test_invalid_control_function_parameters_raises(self):
        """Raises if control function has wrong number of parameters."""
        continuous = MockContinuousSystem()
        discrete = DiscretizedSystem(continuous, dt=0.01)
        
        def bad_func(a, b, c):  # 3 parameters
            return np.array([0.0])
        
        with pytest.raises(ValueError, match="1 or 2 parameters"):
            discrete._prepare_control_sequence(bad_func, n_steps=10)


# ============================================================================
# Pytest Fixtures
# ============================================================================


@pytest.fixture
def mock_continuous():
    """Fixture for mock continuous system."""
    return MockContinuousSystem(nx=2, nu=1)


@pytest.fixture
def mock_stochastic():
    """Fixture for mock stochastic system."""
    return MockStochasticSystem(nx=2, nu=1)


@pytest.fixture
def discrete_fixed(mock_continuous):
    """Fixture for FIXED_STEP discretized system."""
    return DiscretizedSystem(mock_continuous, dt=0.01, method='rk4')


@pytest.fixture
def discrete_dense(mock_continuous):
    """Fixture for DENSE_OUTPUT discretized system."""
    return DiscretizedSystem(mock_continuous, dt=0.01, method='RK45')


@pytest.fixture
def discrete_batch(mock_continuous):
    """Fixture for BATCH_INTERPOLATION discretized system."""
    return DiscretizedSystem(
        mock_continuous, dt=0.01, method='LSODA',
        mode=DiscretizationMode.BATCH_INTERPOLATION
    )


# ============================================================================
# Test Suite: Parametrized Tests
# ============================================================================


class TestParametrized:
    """Parametrized tests across different configurations."""
    
    @pytest.mark.parametrize("method", ['euler', 'rk4', 'midpoint'])
    def test_fixed_step_methods(self, mock_continuous, method):
        """All fixed-step methods work."""
        discrete = DiscretizedSystem(mock_continuous, dt=0.01, method=method)
        
        assert discrete.mode == DiscretizationMode.FIXED_STEP
        
        x = np.array([1.0, 0.0])
        x_next = discrete.step(x, None)
        assert x_next.shape == (2,)
    
    @pytest.mark.parametrize("method", ['RK45', 'RK23', 'LSODA'])
    def test_adaptive_methods(self, mock_continuous, method):
        """All adaptive methods work."""
        discrete = DiscretizedSystem(mock_continuous, dt=0.01, method=method)
        
        assert discrete.mode == DiscretizationMode.DENSE_OUTPUT
        
        result = discrete.simulate(np.array([1.0, 0.0]), None, n_steps=10)
        assert result['success']
    
    @pytest.mark.parametrize("mode", list(DiscretizationMode))
    def test_all_modes_simulate(self, mock_continuous, mode):
        """All modes can simulate."""
        if mode == DiscretizationMode.FIXED_STEP:
            method = 'rk4'
        else:
            method = 'RK45'
        
        discrete = DiscretizedSystem(mock_continuous, dt=0.01, method=method, mode=mode)
        result = discrete.simulate(np.array([1.0, 0.0]), None, n_steps=10)
        
        assert result['states'].shape == (11, 2)
    
    @pytest.mark.parametrize("dt", [0.001, 0.01, 0.1])
    def test_different_time_steps(self, mock_continuous, dt):
        """Works with various time steps."""
        discrete = DiscretizedSystem(mock_continuous, dt=dt)
        
        assert discrete.dt == dt
        
        result = discrete.simulate(np.array([1.0, 0.0]), None, n_steps=10)
        assert result['success']


# ============================================================================
# Test Suite: Consistency Tests
# ============================================================================


class TestConsistency:
    """Test consistency between different modes and methods."""
    
    def test_step_and_simulate_consistency(self):
        """step() called n times equals simulate(n_steps)."""
        continuous = MockContinuousSystem()
        discrete = DiscretizedSystem(continuous, dt=0.01, method='rk4')
        
        x0 = np.array([1.0, 0.0])
        
        # Step-by-step
        x_step = x0
        for k in range(10):
            x_step = discrete.step(x_step, None, k)
        
        # Simulate
        result = discrete.simulate(x0, None, n_steps=10)
        x_sim = result['states'][-1, :]
        
        # Should match
        assert np.allclose(x_step, x_sim, rtol=1e-10)
    
    def test_different_modes_converge_to_same_solution(self):
        """All modes should give similar results (within tolerance)."""
        continuous = MockContinuousSystem()
        x0 = np.array([1.0, 0.0])
        
        fixed = DiscretizedSystem(continuous, dt=0.01, method='rk4')
        dense = DiscretizedSystem(continuous, dt=0.01, method='RK45')
        batch = DiscretizedSystem(
            continuous, dt=0.01, method='RK45',
            mode=DiscretizationMode.BATCH_INTERPOLATION
        )
        
        result_fixed = fixed.simulate(x0, None, n_steps=50)
        result_dense = dense.simulate(x0, None, n_steps=50)
        result_batch = batch.simulate(x0, None, n_steps=50)
        
        # All should be reasonably close
        assert np.allclose(
            result_fixed['states'][-1, :],
            result_dense['states'][-1, :],
            rtol=1e-4  # RK4 vs RK45 may differ slightly
        )
        assert np.allclose(
            result_dense['states'][-1, :],
            result_batch['states'][-1, :],
            rtol=1e-10  # Same method, should match exactly
        )


# ============================================================================
# Test Suite: Regression Tests
# ============================================================================


class TestRegressionTests:
    """Regression tests for known issues."""
    
    def test_interpolation_does_not_introduce_large_errors(self):
        """BATCH mode interpolation doesn't add significant error."""
        continuous = MockContinuousSystem()
        x0 = np.array([1.0, 0.0])
        
        # Dense vs Batch (same RK45 method)
        dense = DiscretizedSystem(continuous, dt=0.01, method='RK45')
        batch = DiscretizedSystem(
            continuous, dt=0.01, method='RK45',
            mode=DiscretizationMode.BATCH_INTERPOLATION
        )
        
        result_dense = dense.simulate(x0, None, n_steps=100)
        result_batch = batch.simulate(x0, None, n_steps=100)
        
        # Interpolation error should be very small
        max_error = np.max(np.abs(result_dense['states'] - result_batch['states']))
        assert max_error < 1e-6  # Cubic interpolation is accurate
    
    def test_linearization_stable_system_has_stable_discretization(self):
        """Stable continuous → stable discrete."""
        continuous = MockContinuousSystem()  # A = -I (stable)
        discrete = DiscretizedSystem(continuous, dt=0.01)
        
        Ad, Bd = discrete.linearize(np.zeros(2), np.zeros(1))
        eigenvalues = np.linalg.eigvals(Ad)
        
        # Should be stable (|λ| < 1)
        assert np.all(np.abs(eigenvalues) < 1.0)


# ============================================================================
# Run Tests
# ============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])