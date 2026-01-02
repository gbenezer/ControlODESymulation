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
Unit Tests for Control Synthesis Wrapper

Tests cover:
- Wrapper initialization and configuration
- Method delegation to classical functions
- Backend consistency and propagation
- Return type validation
- Integration with classical control functions
- Multi-backend support
- Error propagation

Test Structure:
- TestControlSynthesisInit: Initialization and configuration
- TestLQRMethods: LQR continuous and discrete wrappers
- TestKalmanMethod: Kalman filter wrapper
- TestLQGMethod: LQG controller wrapper
- TestBackendConsistency: Backend propagation and consistency
- TestDelegation: Verify proper delegation to classical functions
- TestIntegration: End-to-end usage patterns
- TestErrorPropagation: Error handling from underlying functions

The ControlSynthesis class is a thin wrapper, so tests focus on:
1. Correct delegation to classical functions
2. Backend parameter passing
3. Return type consistency
4. No state mutation or side effects
"""

import unittest
from typing import Dict, Any
from unittest.mock import Mock, patch, call

import numpy as np
from numpy.testing import assert_allclose

# Optional backends for testing
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    HAS_JAX = False

from src.control.control_synthesis import ControlSynthesis
from src.types.control_classical import (
    KalmanFilterResult,
    LQGResult,
    LQRResult,
)


# ============================================================================
# Test Fixtures and Utilities
# ============================================================================

class SynthesisTestCase(unittest.TestCase):
    """Base class with common test utilities."""
    
    def setUp(self):
        """Set up common test systems and matrices."""
        # Standard test system (double integrator)
        self.A = np.array([[0, 1], [0, 0]])
        self.B = np.array([[0], [1]])
        self.C = np.array([[1, 0]])
        
        # Stable system for testing
        self.A_stable = np.array([[0, 1], [-2, -3]])
        self.B_stable = np.array([[0], [1]])
        self.C_stable = np.array([[1, 0]])
        
        # Discrete system
        dt = 0.1
        self.Ad = np.array([[1, dt], [0, 1]])
        self.Bd = np.array([[0.5*dt**2], [dt]])
        
        # Cost matrices
        self.Q = np.diag([10, 1])
        self.R = np.array([[0.1]])
        
        # Noise covariances
        self.Q_process = 0.01 * np.eye(2)
        self.R_meas = 0.1 * np.eye(1)
        
        # Tolerance
        self.rtol = 1e-5
        self.atol = 1e-8
    
    def assert_lqr_result_valid(self, result: LQRResult, nx: int, nu: int):
        """Validate LQR result structure and shapes."""
        self.assertIn('gain', result)
        self.assertIn('cost_to_go', result)
        self.assertIn('closed_loop_eigenvalues', result)
        self.assertIn('stability_margin', result)
        
        self.assertEqual(result['gain'].shape, (nu, nx))
        self.assertEqual(result['cost_to_go'].shape, (nx, nx))
        self.assertEqual(len(result['closed_loop_eigenvalues']), nx)
        self.assertIsInstance(result['stability_margin'], (float, np.floating))
    
    def assert_kalman_result_valid(self, result: KalmanFilterResult, nx: int, ny: int):
        """Validate Kalman filter result structure and shapes."""
        self.assertIn('gain', result)
        self.assertIn('error_covariance', result)
        self.assertIn('innovation_covariance', result)
        self.assertIn('observer_eigenvalues', result)
        
        self.assertEqual(result['gain'].shape, (nx, ny))
        self.assertEqual(result['error_covariance'].shape, (nx, nx))
        self.assertEqual(result['innovation_covariance'].shape, (ny, ny))
        self.assertEqual(len(result['observer_eigenvalues']), nx)
    
    def assert_lqg_result_valid(self, result: LQGResult, nx: int, nu: int, ny: int):
        """Validate LQG result structure and shapes."""
        self.assertIn('controller_gain', result)
        self.assertIn('estimator_gain', result)
        self.assertIn('controller_riccati', result)
        self.assertIn('estimator_covariance', result)
        self.assertIn('closed_loop_eigenvalues', result)
        self.assertIn('observer_eigenvalues', result)
        
        self.assertEqual(result['controller_gain'].shape, (nu, nx))
        self.assertEqual(result['estimator_gain'].shape, (nx, ny))
        self.assertEqual(result['controller_riccati'].shape, (nx, nx))
        self.assertEqual(result['estimator_covariance'].shape, (nx, nx))


# ============================================================================
# Initialization Tests
# ============================================================================

class TestControlSynthesisInit(SynthesisTestCase):
    """Test ControlSynthesis initialization and configuration."""
    
    def test_default_initialization(self):
        """Test default initialization with numpy backend."""
        synthesis = ControlSynthesis()
        
        self.assertEqual(synthesis.backend, 'numpy')
    
    def test_initialization_with_backend(self):
        """Test initialization with specified backend."""
        backends = ['numpy', 'torch', 'jax']
        
        for backend in backends:
            with self.subTest(backend=backend):
                synthesis = ControlSynthesis(backend=backend)
                self.assertEqual(synthesis.backend, backend)
    
    def test_is_stateless(self):
        """Test that ControlSynthesis holds minimal state."""
        synthesis = ControlSynthesis(backend='numpy')
        
        # Should only have backend attribute
        self.assertEqual(len(vars(synthesis)), 1)
        self.assertIn('backend', vars(synthesis))
    
    def test_multiple_instances_independent(self):
        """Test that multiple instances are independent."""
        synthesis1 = ControlSynthesis(backend='numpy')
        synthesis2 = ControlSynthesis(backend='torch')
        
        self.assertEqual(synthesis1.backend, 'numpy')
        self.assertEqual(synthesis2.backend, 'torch')
        
        # Changing one shouldn't affect the other
        synthesis1.backend = 'jax'
        self.assertEqual(synthesis1.backend, 'jax')
        self.assertEqual(synthesis2.backend, 'torch')


# ============================================================================
# LQR Method Tests
# ============================================================================

class TestLQRMethods(SynthesisTestCase):
    """Test LQR wrapper methods."""
    
    def test_design_lqr_continuous_basic(self):
        """Test basic continuous LQR design."""
        synthesis = ControlSynthesis(backend='numpy')
        
        result = synthesis.design_lqr_continuous(
            self.A_stable,
            self.B_stable,
            self.Q,
            self.R
        )
        
        self.assert_lqr_result_valid(result, nx=2, nu=1)
        self.assertGreater(result['stability_margin'], 0)
    
    def test_design_lqr_continuous_with_cross_term(self):
        """Test continuous LQR with cross-coupling term."""
        synthesis = ControlSynthesis(backend='numpy')
        N = np.array([[0.5], [0.1]])
        
        result = synthesis.design_lqr_continuous(
            self.A_stable,
            self.B_stable,
            self.Q,
            self.R,
            N=N
        )
        
        self.assert_lqr_result_valid(result, nx=2, nu=1)
        self.assertGreater(result['stability_margin'], 0)
    
    def test_design_lqr_discrete_basic(self):
        """Test basic discrete LQR design."""
        synthesis = ControlSynthesis(backend='numpy')
        
        result = synthesis.design_lqr_discrete(
            self.Ad,
            self.Bd,
            self.Q,
            self.R
        )
        
        self.assert_lqr_result_valid(result, nx=2, nu=1)
        self.assertGreater(result['stability_margin'], 0)
    
    def test_design_lqr_discrete_with_cross_term(self):
        """Test discrete LQR with cross-coupling term."""
        synthesis = ControlSynthesis(backend='numpy')
        N = np.array([[0.5], [0.1]])
        
        result = synthesis.design_lqr_discrete(
            self.Ad,
            self.Bd,
            self.Q,
            self.R,
            N=N
        )
        
        self.assert_lqr_result_valid(result, nx=2, nu=1)
    
    def test_lqr_results_are_numpy_arrays(self):
        """Test that results are in expected backend format."""
        synthesis = ControlSynthesis(backend='numpy')
        
        result = synthesis.design_lqr_continuous(
            self.A_stable,
            self.B_stable,
            self.Q,
            self.R
        )
        
        self.assertIsInstance(result['gain'], np.ndarray)
        self.assertIsInstance(result['cost_to_go'], np.ndarray)
        self.assertIsInstance(result['closed_loop_eigenvalues'], np.ndarray)


# ============================================================================
# Kalman Filter Method Tests
# ============================================================================

class TestKalmanMethod(SynthesisTestCase):
    """Test Kalman filter wrapper method."""
    
    def test_design_kalman_discrete_basic(self):
        """Test basic discrete Kalman filter design."""
        synthesis = ControlSynthesis(backend='numpy')
        
        result = synthesis.design_kalman(
            self.Ad,
            self.C,
            self.Q_process,
            self.R_meas,
            system_type='discrete'
        )
        
        self.assert_kalman_result_valid(result, nx=2, ny=1)
    
    def test_design_kalman_continuous(self):
        """Test continuous Kalman filter design."""
        synthesis = ControlSynthesis(backend='numpy')
        
        result = synthesis.design_kalman(
            self.A_stable,
            self.C_stable,
            self.Q_process,
            self.R_meas,
            system_type='continuous'
        )
        
        self.assert_kalman_result_valid(result, nx=2, ny=1)
    
    def test_design_kalman_default_discrete(self):
        """Test that default system_type is discrete."""
        synthesis = ControlSynthesis(backend='numpy')
        
        # Should work without explicit system_type
        result = synthesis.design_kalman(
            self.Ad,
            self.C,
            self.Q_process,
            self.R_meas
        )
        
        self.assert_kalman_result_valid(result, nx=2, ny=1)
    
    def test_kalman_observer_stability(self):
        """Test that Kalman filter produces stable observer."""
        synthesis = ControlSynthesis(backend='numpy')
        
        result = synthesis.design_kalman(
            self.Ad,
            self.C,
            self.Q_process,
            self.R_meas,
            system_type='discrete'
        )
        
        # Observer should be stable (|λ| < 1)
        max_mag = np.max(np.abs(result['observer_eigenvalues']))
        self.assertLess(max_mag, 1.0)


# ============================================================================
# LQG Method Tests
# ============================================================================

class TestLQGMethod(SynthesisTestCase):
    """Test LQG wrapper method."""
    
    def test_design_lqg_discrete_basic(self):
        """Test basic discrete LQG design."""
        synthesis = ControlSynthesis(backend='numpy')
        
        result = synthesis.design_lqg(
            self.Ad,
            self.Bd,
            self.C,
            self.Q,
            self.R,
            self.Q_process,
            self.R_meas,
            system_type='discrete'
        )
        
        self.assert_lqg_result_valid(result, nx=2, nu=1, ny=1)
    
    def test_design_lqg_continuous(self):
        """Test continuous LQG design."""
        synthesis = ControlSynthesis(backend='numpy')
        
        result = synthesis.design_lqg(
            self.A_stable,
            self.B_stable,
            self.C_stable,
            self.Q,
            self.R,
            self.Q_process,
            self.R_meas,
            system_type='continuous'
        )
        
        self.assert_lqg_result_valid(result, nx=2, nu=1, ny=1)
    
    def test_design_lqg_default_discrete(self):
        """Test that default system_type is discrete."""
        synthesis = ControlSynthesis(backend='numpy')
        
        result = synthesis.design_lqg(
            self.Ad,
            self.Bd,
            self.C,
            self.Q,
            self.R,
            self.Q_process,
            self.R_meas
        )
        
        self.assert_lqg_result_valid(result, nx=2, nu=1, ny=1)
    
    def test_lqg_both_stable(self):
        """Test that LQG produces stable controller and observer."""
        synthesis = ControlSynthesis(backend='numpy')
        
        result = synthesis.design_lqg(
            self.Ad,
            self.Bd,
            self.C,
            self.Q,
            self.R,
            self.Q_process,
            self.R_meas,
            system_type='discrete'
        )
        
        # Controller should be stable
        ctrl_max = np.max(np.abs(result['closed_loop_eigenvalues']))
        self.assertLess(ctrl_max, 1.0)
        
        # Observer should be stable
        obs_max = np.max(np.abs(result['observer_eigenvalues']))
        self.assertLess(obs_max, 1.0)


# ============================================================================
# Backend Consistency Tests
# ============================================================================

class TestBackendConsistency(SynthesisTestCase):
    """Test backend propagation and consistency."""
    
    def test_backend_passed_to_lqr_continuous(self):
        """Test that backend is passed to underlying LQR function."""
        with patch('src.control.synthesis.design_lqr_continuous') as mock_lqr:
            mock_lqr.return_value = {
                'gain': np.zeros((1, 2)),
                'cost_to_go': np.eye(2),
                'closed_loop_eigenvalues': np.array([-1, -2]),
                'stability_margin': 1.0
            }
            
            synthesis = ControlSynthesis(backend='torch')
            synthesis.design_lqr_continuous(
                self.A_stable,
                self.B_stable,
                self.Q,
                self.R
            )
            
            # Verify backend was passed
            mock_lqr.assert_called_once()
            call_kwargs = mock_lqr.call_args[1]
            self.assertEqual(call_kwargs['backend'], 'torch')
    
    def test_backend_passed_to_lqr_discrete(self):
        """Test that backend is passed to discrete LQR."""
        with patch('src.control.synthesis.design_lqr_discrete') as mock_lqr:
            mock_lqr.return_value = {
                'gain': np.zeros((1, 2)),
                'cost_to_go': np.eye(2),
                'closed_loop_eigenvalues': np.array([0.5, 0.6]),
                'stability_margin': 0.4
            }
            
            synthesis = ControlSynthesis(backend='jax')
            synthesis.design_lqr_discrete(
                self.Ad,
                self.Bd,
                self.Q,
                self.R
            )
            
            call_kwargs = mock_lqr.call_args[1]
            self.assertEqual(call_kwargs['backend'], 'jax')
    
    def test_backend_passed_to_kalman(self):
        """Test that backend is passed to Kalman filter."""
        with patch('src.control.synthesis.design_kalman_filter') as mock_kalman:
            mock_kalman.return_value = {
                'gain': np.zeros((2, 1)),
                'error_covariance': np.eye(2),
                'innovation_covariance': np.array([[0.1]]),
                'observer_eigenvalues': np.array([0.5, 0.6])
            }
            
            synthesis = ControlSynthesis(backend='numpy')
            synthesis.design_kalman(
                self.Ad,
                self.C,
                self.Q_process,
                self.R_meas
            )
            
            call_kwargs = mock_kalman.call_args[1]
            self.assertEqual(call_kwargs['backend'], 'numpy')
    
    def test_backend_passed_to_lqg(self):
        """Test that backend is passed to LQG."""
        with patch('src.control.synthesis.design_lqg') as mock_lqg:
            mock_lqg.return_value = {
                'controller_gain': np.zeros((1, 2)),
                'estimator_gain': np.zeros((2, 1)),
                'controller_riccati': np.eye(2),
                'estimator_covariance': np.eye(2),
                'closed_loop_eigenvalues': np.array([0.5, 0.6]),
                'observer_eigenvalues': np.array([0.4, 0.5])
            }
            
            synthesis = ControlSynthesis(backend='torch')
            synthesis.design_lqg(
                self.Ad,
                self.Bd,
                self.C,
                self.Q,
                self.R,
                self.Q_process,
                self.R_meas
            )
            
            call_kwargs = mock_lqg.call_args[1]
            self.assertEqual(call_kwargs['backend'], 'torch')
    
    @unittest.skipIf(not HAS_TORCH, "PyTorch not available")
    def test_torch_backend_end_to_end(self):
        """Test torch backend with actual computation."""
        synthesis = ControlSynthesis(backend='torch')
        
        A_torch = torch.tensor(self.A_stable, dtype=torch.float64)
        B_torch = torch.tensor(self.B_stable, dtype=torch.float64)
        Q_torch = torch.tensor(self.Q, dtype=torch.float64)
        R_torch = torch.tensor(self.R, dtype=torch.float64)
        
        result = synthesis.design_lqr_continuous(
            A_torch, B_torch, Q_torch, R_torch
        )
        
        # Result should be torch tensors
        self.assertIsInstance(result['gain'], torch.Tensor)
        self.assertIsInstance(result['cost_to_go'], torch.Tensor)
    
    @unittest.skipIf(not HAS_JAX, "JAX not available")
    def test_jax_backend_end_to_end(self):
        """Test jax backend with actual computation."""
        synthesis = ControlSynthesis(backend='jax')
        
        A_jax = jnp.array(self.A_stable)
        B_jax = jnp.array(self.B_stable)
        Q_jax = jnp.array(self.Q)
        R_jax = jnp.array(self.R)
        
        result = synthesis.design_lqr_continuous(
            A_jax, B_jax, Q_jax, R_jax
        )
        
        # Result should be jax arrays
        self.assertIsInstance(result['gain'], jnp.ndarray)
        self.assertIsInstance(result['cost_to_go'], jnp.ndarray)


# ============================================================================
# Delegation Tests
# ============================================================================

class TestDelegation(SynthesisTestCase):
    """Test proper delegation to classical functions."""
    
    def test_lqr_continuous_delegates_correctly(self):
        """Test that design_lqr_continuous delegates with correct arguments."""
        with patch('src.control.synthesis.design_lqr_continuous') as mock_func:
            mock_func.return_value = {
                'gain': np.zeros((1, 2)),
                'cost_to_go': np.eye(2),
                'closed_loop_eigenvalues': np.array([-1, -2]),
                'stability_margin': 1.0
            }
            
            synthesis = ControlSynthesis(backend='numpy')
            N = np.array([[0.5], [0.1]])
            
            synthesis.design_lqr_continuous(
                self.A_stable,
                self.B_stable,
                self.Q,
                self.R,
                N=N
            )
            
            # Verify correct delegation
            mock_func.assert_called_once_with(
                self.A_stable,
                self.B_stable,
                self.Q,
                self.R,
                N,
                backend='numpy'
            )
    
    def test_lqr_discrete_delegates_correctly(self):
        """Test that design_lqr_discrete delegates with correct arguments."""
        with patch('src.control.synthesis.design_lqr_discrete') as mock_func:
            mock_func.return_value = {
                'gain': np.zeros((1, 2)),
                'cost_to_go': np.eye(2),
                'closed_loop_eigenvalues': np.array([0.5, 0.6]),
                'stability_margin': 0.4
            }
            
            synthesis = ControlSynthesis(backend='numpy')
            
            synthesis.design_lqr_discrete(
                self.Ad,
                self.Bd,
                self.Q,
                self.R
            )
            
            mock_func.assert_called_once_with(
                self.Ad,
                self.Bd,
                self.Q,
                self.R,
                None,  # N defaults to None
                backend='numpy'
            )
    
    def test_kalman_delegates_correctly(self):
        """Test that design_kalman delegates with correct arguments."""
        with patch('src.control.synthesis.design_kalman_filter') as mock_func:
            mock_func.return_value = {
                'gain': np.zeros((2, 1)),
                'error_covariance': np.eye(2),
                'innovation_covariance': np.array([[0.1]]),
                'observer_eigenvalues': np.array([0.5, 0.6])
            }
            
            synthesis = ControlSynthesis(backend='numpy')
            
            synthesis.design_kalman(
                self.Ad,
                self.C,
                self.Q_process,
                self.R_meas,
                system_type='discrete'
            )
            
            mock_func.assert_called_once_with(
                self.Ad,
                self.C,
                self.Q_process,
                self.R_meas,
                'discrete',
                backend='numpy'
            )
    
    def test_lqg_delegates_correctly(self):
        """Test that design_lqg delegates with correct arguments."""
        with patch('src.control.synthesis.design_lqg') as mock_func:
            mock_func.return_value = {
                'controller_gain': np.zeros((1, 2)),
                'estimator_gain': np.zeros((2, 1)),
                'controller_riccati': np.eye(2),
                'estimator_covariance': np.eye(2),
                'closed_loop_eigenvalues': np.array([0.5, 0.6]),
                'observer_eigenvalues': np.array([0.4, 0.5])
            }
            
            synthesis = ControlSynthesis(backend='numpy')
            
            synthesis.design_lqg(
                self.Ad,
                self.Bd,
                self.C,
                self.Q,
                self.R,
                self.Q_process,
                self.R_meas,
                system_type='discrete'
            )
            
            mock_func.assert_called_once_with(
                self.Ad,
                self.Bd,
                self.C,
                self.Q,
                self.R,
                self.Q_process,
                self.R_meas,
                'discrete',
                backend='numpy'
            )
    
    def test_no_state_mutation(self):
        """Test that methods don't mutate wrapper state."""
        synthesis = ControlSynthesis(backend='numpy')
        initial_backend = synthesis.backend
        
        # Call various methods
        synthesis.design_lqr_continuous(
            self.A_stable,
            self.B_stable,
            self.Q,
            self.R
        )
        
        synthesis.design_kalman(
            self.Ad,
            self.C,
            self.Q_process,
            self.R_meas
        )
        
        # Backend should remain unchanged
        self.assertEqual(synthesis.backend, initial_backend)
        
        # Should still only have one attribute
        self.assertEqual(len(vars(synthesis)), 1)


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration(SynthesisTestCase):
    """Test integration and typical usage patterns."""
    
    def test_complete_lqg_workflow(self):
        """Test complete LQG design workflow."""
        synthesis = ControlSynthesis(backend='numpy')
        
        # Design LQG controller
        lqg_result = synthesis.design_lqg(
            self.Ad,
            self.Bd,
            self.C,
            self.Q,
            self.R,
            self.Q_process,
            self.R_meas,
            system_type='discrete'
        )
        
        # Extract gains
        K = lqg_result['controller_gain']
        L = lqg_result['estimator_gain']
        
        # Verify shapes
        self.assertEqual(K.shape, (1, 2))
        self.assertEqual(L.shape, (2, 1))
        
        # Verify both are stable
        ctrl_stable = np.all(np.abs(lqg_result['closed_loop_eigenvalues']) < 1)
        obs_stable = np.all(np.abs(lqg_result['observer_eigenvalues']) < 1)
        
        self.assertTrue(ctrl_stable)
        self.assertTrue(obs_stable)
    
    def test_lqr_then_kalman_separately(self):
        """Test designing LQR and Kalman separately (vs combined LQG)."""
        synthesis = ControlSynthesis(backend='numpy')
        
        # Design LQR
        lqr_result = synthesis.design_lqr_discrete(
            self.Ad,
            self.Bd,
            self.Q,
            self.R
        )
        
        # Design Kalman
        kalman_result = synthesis.design_kalman(
            self.Ad,
            self.C,
            self.Q_process,
            self.R_meas,
            system_type='discrete'
        )
        
        # Design LQG (combined)
        lqg_result = synthesis.design_lqg(
            self.Ad,
            self.Bd,
            self.C,
            self.Q,
            self.R,
            self.Q_process,
            self.R_meas,
            system_type='discrete'
        )
        
        # Separation principle: gains should match
        assert_allclose(
            lqg_result['controller_gain'],
            lqr_result['gain'],
            rtol=self.rtol, atol=self.atol
        )
        assert_allclose(
            lqg_result['estimator_gain'],
            kalman_result['gain'],
            rtol=self.rtol, atol=self.atol
        )
    
    def test_continuous_vs_discrete_consistency(self):
        """Test that continuous and discrete designs are consistent."""
        synthesis = ControlSynthesis(backend='numpy')
        
        # Continuous LQR
        result_c = synthesis.design_lqr_continuous(
            self.A_stable,
            self.B_stable,
            self.Q,
            self.R
        )
        
        # Discrete LQR (on discretized system)
        result_d = synthesis.design_lqr_discrete(
            self.Ad,
            self.Bd,
            self.Q,
            self.R
        )
        
        # Both should be stable
        self.assertGreater(result_c['stability_margin'], 0)
        self.assertGreater(result_d['stability_margin'], 0)
        
        # Both should produce valid gains
        self.assertFalse(np.any(np.isnan(result_c['gain'])))
        self.assertFalse(np.any(np.isnan(result_d['gain'])))
    
    def test_multiple_designs_with_same_wrapper(self):
        """Test that wrapper can be reused for multiple designs."""
        synthesis = ControlSynthesis(backend='numpy')
        
        # Multiple LQR designs with different weights
        Q1 = np.diag([10, 1])
        Q2 = np.diag([100, 10])
        Q3 = np.diag([1, 1])
        
        result1 = synthesis.design_lqr_continuous(
            self.A_stable, self.B_stable, Q1, self.R
        )
        result2 = synthesis.design_lqr_continuous(
            self.A_stable, self.B_stable, Q2, self.R
        )
        result3 = synthesis.design_lqr_continuous(
            self.A_stable, self.B_stable, Q3, self.R
        )
        
        # All should be valid and different
        self.assert_lqr_result_valid(result1, nx=2, nu=1)
        self.assert_lqr_result_valid(result2, nx=2, nu=1)
        self.assert_lqr_result_valid(result3, nx=2, nu=1)
        
        # Higher Q should give larger gains
        K1_norm = np.linalg.norm(result1['gain'])
        K2_norm = np.linalg.norm(result2['gain'])
        K3_norm = np.linalg.norm(result3['gain'])
        
        self.assertGreater(K2_norm, K1_norm)
        self.assertGreater(K1_norm, K3_norm)


# ============================================================================
# Error Propagation Tests
# ============================================================================

class TestErrorPropagation(SynthesisTestCase):
    """Test that errors from underlying functions are properly propagated."""
    
    def test_lqr_dimension_error_propagates(self):
        """Test that dimension errors from LQR propagate correctly."""
        synthesis = ControlSynthesis(backend='numpy')
        
        # Mismatched dimensions
        A_wrong = np.eye(3)  # 3x3 instead of 2x2
        
        with self.assertRaises(ValueError):
            synthesis.design_lqr_continuous(
                A_wrong,
                self.B_stable,  # 2x1
                self.Q,
                self.R
            )
    
    def test_kalman_dimension_error_propagates(self):
        """Test that dimension errors from Kalman propagate correctly."""
        synthesis = ControlSynthesis(backend='numpy')
        
        # Mismatched dimensions
        C_wrong = np.array([[1, 0, 0]])  # 1x3 instead of 1x2
        
        with self.assertRaises(ValueError):
            synthesis.design_kalman(
                self.Ad,  # 2x2
                C_wrong,
                self.Q_process,
                self.R_meas
            )
    
    def test_lqg_invalid_system_type_propagates(self):
        """Test that invalid system type error propagates."""
        synthesis = ControlSynthesis(backend='numpy')
        
        with self.assertRaises(ValueError):
            synthesis.design_lqg(
                self.Ad,
                self.Bd,
                self.C,
                self.Q,
                self.R,
                self.Q_process,
                self.R_meas,
                system_type='invalid'
            )
    
    def test_riccati_failure_propagates(self):
        """Test that Riccati equation failures propagate."""
        synthesis = ControlSynthesis(backend='numpy')
        
        # Uncontrollable system
        A = np.array([[1, 0], [0, 2]])
        B = np.array([[0], [0]])  # No control authority
        
        try:
            result = synthesis.design_lqr_continuous(A, B, self.Q, self.R)
            # If it doesn't raise, the result should indicate instability
            # (can't stabilize uncontrollable system)
            A_cl = A - B @ result['gain']
            max_real = np.max(np.real(np.linalg.eigvals(A_cl)))
            self.assertGreater(max_real, 0, "Should be unstable")
        except np.linalg.LinAlgError:
            # Expected for uncontrollable system
            pass


# ============================================================================
# Documentation and Interface Tests
# ============================================================================

class TestInterface(SynthesisTestCase):
    """Test that interface matches documentation."""
    
    def test_all_methods_exist(self):
        """Test that all documented methods exist."""
        synthesis = ControlSynthesis()
        
        required_methods = [
            'design_lqr_continuous',
            'design_lqr_discrete',
            'design_kalman',
            'design_lqg',
        ]
        
        for method in required_methods:
            self.assertTrue(
                hasattr(synthesis, method),
                f"Missing method: {method}"
            )
            self.assertTrue(
                callable(getattr(synthesis, method)),
                f"{method} is not callable"
            )
    
    def test_method_signatures_match_docs(self):
        """Test that method signatures match documentation."""
        import inspect
        
        synthesis = ControlSynthesis()
        
        # Check design_lqr_continuous signature
        sig = inspect.signature(synthesis.design_lqr_continuous)
        params = list(sig.parameters.keys())
        expected = ['A', 'B', 'Q', 'R', 'N']
        self.assertEqual(params, expected)
        
        # Check design_kalman signature
        sig = inspect.signature(synthesis.design_kalman)
        params = list(sig.parameters.keys())
        expected = ['A', 'C', 'Q', 'R', 'system_type']
        self.assertEqual(params, expected)
        
        # Check design_lqg signature
        sig = inspect.signature(synthesis.design_lqg)
        params = list(sig.parameters.keys())
        expected = ['A', 'B', 'C', 'Q_state', 'R_control', 
                    'Q_process', 'R_measurement', 'system_type']
        self.assertEqual(params, expected)
    
    def test_return_types_match_docs(self):
        """Test that return types match documentation."""
        synthesis = ControlSynthesis(backend='numpy')
        
        # LQR should return LQRResult
        lqr_result = synthesis.design_lqr_continuous(
            self.A_stable, self.B_stable, self.Q, self.R
        )
        self.assertIsInstance(lqr_result, dict)
        self.assertIn('gain', lqr_result)
        
        # Kalman should return KalmanFilterResult
        kalman_result = synthesis.design_kalman(
            self.Ad, self.C, self.Q_process, self.R_meas
        )
        self.assertIsInstance(kalman_result, dict)
        self.assertIn('gain', kalman_result)
        
        # LQG should return LQGResult
        lqg_result = synthesis.design_lqg(
            self.Ad, self.Bd, self.C,
            self.Q, self.R,
            self.Q_process, self.R_meas
        )
        self.assertIsInstance(lqg_result, dict)
        self.assertIn('controller_gain', lqg_result)
        self.assertIn('estimator_gain', lqg_result)


# ============================================================================
# Test Suite
# ============================================================================

def suite():
    """Create test suite."""
    test_suite = unittest.TestSuite()
    
    # Add all test classes
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestControlSynthesisInit))
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestLQRMethods))
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestKalmanMethod))
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestLQGMethod))
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestBackendConsistency))
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestDelegation))
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestIntegration))
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestErrorPropagation))
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestInterface))
    
    return test_suite


if __name__ == '__main__':
    # Run tests with verbosity
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())