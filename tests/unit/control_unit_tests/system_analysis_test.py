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
Unit Tests for System Analysis Wrapper

Tests cover:
- Wrapper initialization and configuration
- Method delegation to classical analysis functions
- Backend consistency and propagation
- Return type validation
- Comprehensive analyze_linearization method
- Integration with classical analysis functions
- Error propagation

Test Structure:
- TestSystemAnalysisInit: Initialization and configuration
- TestStabilityMethod: Stability analysis wrapper
- TestControllabilityMethod: Controllability analysis wrapper
- TestObservabilityMethod: Observability analysis wrapper
- TestAnalyzeLinearization: Comprehensive analysis method
- TestBackendConsistency: Backend propagation
- TestDelegation: Verify proper delegation to classical functions
- TestIntegration: End-to-end usage patterns
- TestErrorPropagation: Error handling from underlying functions
- TestInterface: API and documentation consistency

The SystemAnalysis class is a thin wrapper, so tests focus on:
1. Correct delegation to classical functions
2. Backend parameter passing (where applicable)
3. Return type consistency
4. No state mutation or side effects
5. Comprehensive analysis method correctness
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

from src.control.system_analysis import SystemAnalysis
from src.types.control_classical import (
    ControllabilityInfo,
    ObservabilityInfo,
    StabilityInfo,
)


# ============================================================================
# Test Fixtures and Utilities
# ============================================================================

class AnalysisTestCase(unittest.TestCase):
    """Base class with common test utilities."""
    
    def setUp(self):
        """Set up common test systems and matrices."""
        # Stable continuous system
        self.A_stable = np.array([[0, 1], [-2, -3]])
        self.B_stable = np.array([[0], [1]])
        self.C_stable = np.array([[1, 0]])
        
        # Unstable continuous system
        self.A_unstable = np.array([[1, 1], [0, 1]])
        self.B_unstable = np.array([[0], [1]])
        
        # Marginally stable system (pure oscillator)
        self.A_marginal = np.array([[0, 1], [-1, 0]])
        
        # Stable discrete system
        self.Ad_stable = 0.9 * np.eye(2)
        self.Bd_stable = np.array([[0.1], [0.2]])
        
        # Controllable system
        self.A_ctrl = np.array([[0, 1], [0, 0]])  # Double integrator
        self.B_ctrl = np.array([[0], [1]])
        
        # Uncontrollable system
        self.A_unctrl = np.array([[1, 0], [0, 2]])
        self.B_unctrl = np.array([[0], [1]])  # Only second state affected
        
        # Observable system
        self.A_obs = np.array([[0, 1], [-2, -3]])
        self.C_obs = np.array([[1, 0]])  # Measure first state
        
        # Unobservable system
        self.A_unobs = np.array([[1, 0], [0, 2]])
        self.C_unobs = np.array([[1, 0]])  # Only measures first state
        
        # Tolerance
        self.rtol = 1e-5
        self.atol = 1e-8
    
    def assert_stability_info_valid(self, info: StabilityInfo):
        """Validate StabilityInfo structure."""
        required_keys = [
            'eigenvalues', 'magnitudes', 'max_magnitude',
            'spectral_radius', 'is_stable', 'is_marginally_stable',
            'is_unstable'
        ]
        for key in required_keys:
            self.assertIn(key, info, f"Missing key: {key}")
        
        # Check types
        self.assertIsInstance(info['eigenvalues'], np.ndarray)
        self.assertIsInstance(info['magnitudes'], np.ndarray)
        self.assertIsInstance(info['max_magnitude'], (float, np.floating))
        self.assertIsInstance(info['spectral_radius'], (float, np.floating))
        self.assertIsInstance(info['is_stable'], bool)
        self.assertIsInstance(info['is_marginally_stable'], bool)
        self.assertIsInstance(info['is_unstable'], bool)
    
    def assert_controllability_info_valid(self, info: ControllabilityInfo):
        """Validate ControllabilityInfo structure."""
        required_keys = [
            'controllability_matrix', 'rank',
            'is_controllable', 'uncontrollable_modes'
        ]
        for key in required_keys:
            self.assertIn(key, info, f"Missing key: {key}")
        
        # Check types
        self.assertIsInstance(info['controllability_matrix'], np.ndarray)
        self.assertIsInstance(info['rank'], (int, np.integer))
        self.assertIsInstance(info['is_controllable'], bool)
    
    def assert_observability_info_valid(self, info: ObservabilityInfo):
        """Validate ObservabilityInfo structure."""
        required_keys = [
            'observability_matrix', 'rank',
            'is_observable', 'unobservable_modes'
        ]
        for key in required_keys:
            self.assertIn(key, info, f"Missing key: {key}")
        
        # Check types
        self.assertIsInstance(info['observability_matrix'], np.ndarray)
        self.assertIsInstance(info['rank'], (int, np.integer))
        self.assertIsInstance(info['is_observable'], bool)


# ============================================================================
# Initialization Tests
# ============================================================================

class TestSystemAnalysisInit(AnalysisTestCase):
    """Test SystemAnalysis initialization and configuration."""
    
    def test_default_initialization(self):
        """Test default initialization with numpy backend."""
        analyzer = SystemAnalysis()
        
        self.assertEqual(analyzer.backend, 'numpy')
    
    def test_initialization_with_backend(self):
        """Test initialization with specified backend."""
        backends = ['numpy', 'torch', 'jax']
        
        for backend in backends:
            with self.subTest(backend=backend):
                analyzer = SystemAnalysis(backend=backend)
                self.assertEqual(analyzer.backend, backend)
    
    def test_is_stateless(self):
        """Test that SystemAnalysis holds minimal state."""
        analyzer = SystemAnalysis(backend='numpy')
        
        # Should only have backend attribute
        self.assertEqual(len(vars(analyzer)), 1)
        self.assertIn('backend', vars(analyzer))
    
    def test_multiple_instances_independent(self):
        """Test that multiple instances are independent."""
        analyzer1 = SystemAnalysis(backend='numpy')
        analyzer2 = SystemAnalysis(backend='torch')
        
        self.assertEqual(analyzer1.backend, 'numpy')
        self.assertEqual(analyzer2.backend, 'torch')
        
        # Changing one shouldn't affect the other
        analyzer1.backend = 'jax'
        self.assertEqual(analyzer1.backend, 'jax')
        self.assertEqual(analyzer2.backend, 'torch')


# ============================================================================
# Stability Method Tests
# ============================================================================

class TestStabilityMethod(AnalysisTestCase):
    """Test stability analysis wrapper method."""
    
    def test_stability_continuous_stable(self):
        """Test stability analysis for stable continuous system."""
        analyzer = SystemAnalysis(backend='numpy')
        
        info = analyzer.stability(self.A_stable, system_type='continuous')
        
        self.assert_stability_info_valid(info)
        self.assertTrue(info['is_stable'])
        self.assertFalse(info['is_unstable'])
        self.assertFalse(info['is_marginally_stable'])
    
    def test_stability_continuous_unstable(self):
        """Test stability analysis for unstable continuous system."""
        analyzer = SystemAnalysis(backend='numpy')
        
        info = analyzer.stability(self.A_unstable, system_type='continuous')
        
        self.assert_stability_info_valid(info)
        self.assertFalse(info['is_stable'])
        self.assertTrue(info['is_unstable'])
    
    def test_stability_continuous_marginal(self):
        """Test stability analysis for marginally stable system."""
        analyzer = SystemAnalysis(backend='numpy')
        
        info = analyzer.stability(self.A_marginal, system_type='continuous')
        
        self.assert_stability_info_valid(info)
        self.assertTrue(info['is_marginally_stable'])
        self.assertFalse(info['is_stable'])
    
    def test_stability_discrete_stable(self):
        """Test stability analysis for stable discrete system."""
        analyzer = SystemAnalysis(backend='numpy')
        
        info = analyzer.stability(self.Ad_stable, system_type='discrete')
        
        self.assert_stability_info_valid(info)
        self.assertTrue(info['is_stable'])
        self.assertLess(info['spectral_radius'], 1.0)
    
    def test_stability_default_continuous(self):
        """Test that default system_type is continuous."""
        analyzer = SystemAnalysis(backend='numpy')
        
        # Should work without explicit system_type
        info = analyzer.stability(self.A_stable)
        
        self.assert_stability_info_valid(info)
    
    def test_stability_custom_tolerance(self):
        """Test stability with custom tolerance."""
        analyzer = SystemAnalysis(backend='numpy')
        
        info = analyzer.stability(
            self.A_marginal,
            system_type='continuous',
            tolerance=1e-6
        )
        
        self.assert_stability_info_valid(info)
    
    def test_stability_eigenvalue_computation(self):
        """Test that eigenvalues are correctly computed."""
        analyzer = SystemAnalysis(backend='numpy')
        
        info = analyzer.stability(self.A_stable, system_type='continuous')
        
        # Verify eigenvalues match numpy computation
        expected_eigs = np.linalg.eigvals(self.A_stable)
        computed_eigs = np.sort(info['eigenvalues'])
        expected_eigs_sorted = np.sort(expected_eigs)
        
        assert_allclose(computed_eigs, expected_eigs_sorted, rtol=self.rtol)


# ============================================================================
# Controllability Method Tests
# ============================================================================

class TestControllabilityMethod(AnalysisTestCase):
    """Test controllability analysis wrapper method."""
    
    def test_controllability_controllable_system(self):
        """Test controllability of fully controllable system."""
        analyzer = SystemAnalysis(backend='numpy')
        
        info = analyzer.controllability(self.A_ctrl, self.B_ctrl)
        
        self.assert_controllability_info_valid(info)
        self.assertTrue(info['is_controllable'])
        self.assertEqual(info['rank'], self.A_ctrl.shape[0])
    
    def test_controllability_uncontrollable_system(self):
        """Test controllability of uncontrollable system."""
        analyzer = SystemAnalysis(backend='numpy')
        
        info = analyzer.controllability(self.A_unctrl, self.B_unctrl)
        
        self.assert_controllability_info_valid(info)
        self.assertFalse(info['is_controllable'])
        self.assertLess(info['rank'], self.A_unctrl.shape[0])
    
    def test_controllability_matrix_shape(self):
        """Test controllability matrix has correct shape."""
        analyzer = SystemAnalysis(backend='numpy')
        
        info = analyzer.controllability(self.A_ctrl, self.B_ctrl)
        
        nx = self.A_ctrl.shape[0]
        nu = self.B_ctrl.shape[1]
        expected_shape = (nx, nx * nu)
        
        self.assertEqual(info['controllability_matrix'].shape, expected_shape)
    
    def test_controllability_custom_tolerance(self):
        """Test controllability with custom tolerance."""
        analyzer = SystemAnalysis(backend='numpy')
        
        info = analyzer.controllability(
            self.A_ctrl,
            self.B_ctrl,
            tolerance=1e-6
        )
        
        self.assert_controllability_info_valid(info)
    
    def test_controllability_multi_input(self):
        """Test controllability with multiple inputs."""
        analyzer = SystemAnalysis(backend='numpy')
        
        A = np.array([[0, 1, 0], [0, 0, 1], [-1, -2, -3]])
        B = np.array([[0, 0], [1, 0], [0, 1]])
        
        info = analyzer.controllability(A, B)
        
        self.assert_controllability_info_valid(info)
        self.assertTrue(info['is_controllable'])
        self.assertEqual(info['controllability_matrix'].shape, (3, 6))


# ============================================================================
# Observability Method Tests
# ============================================================================

class TestObservabilityMethod(AnalysisTestCase):
    """Test observability analysis wrapper method."""
    
    def test_observability_observable_system(self):
        """Test observability of fully observable system."""
        analyzer = SystemAnalysis(backend='numpy')
        
        info = analyzer.observability(self.A_obs, self.C_obs)
        
        self.assert_observability_info_valid(info)
        self.assertTrue(info['is_observable'])
        self.assertEqual(info['rank'], self.A_obs.shape[0])
    
    def test_observability_unobservable_system(self):
        """Test observability of unobservable system."""
        analyzer = SystemAnalysis(backend='numpy')
        
        info = analyzer.observability(self.A_unobs, self.C_unobs)
        
        self.assert_observability_info_valid(info)
        self.assertFalse(info['is_observable'])
        self.assertLess(info['rank'], self.A_unobs.shape[0])
    
    def test_observability_matrix_shape(self):
        """Test observability matrix has correct shape."""
        analyzer = SystemAnalysis(backend='numpy')
        
        info = analyzer.observability(self.A_obs, self.C_obs)
        
        nx = self.A_obs.shape[0]
        ny = self.C_obs.shape[0]
        expected_shape = (nx * ny, nx)
        
        self.assertEqual(info['observability_matrix'].shape, expected_shape)
    
    def test_observability_custom_tolerance(self):
        """Test observability with custom tolerance."""
        analyzer = SystemAnalysis(backend='numpy')
        
        info = analyzer.observability(
            self.A_obs,
            self.C_obs,
            tolerance=1e-6
        )
        
        self.assert_observability_info_valid(info)
    
    def test_observability_full_state_measurement(self):
        """Test observability with full state measurement."""
        analyzer = SystemAnalysis(backend='numpy')
        
        A = np.array([[0, 1], [-2, -3]])
        C = np.eye(2)  # Measure all states
        
        info = analyzer.observability(A, C)
        
        self.assert_observability_info_valid(info)
        self.assertTrue(info['is_observable'])


# ============================================================================
# Analyze Linearization Method Tests
# ============================================================================

class TestAnalyzeLinearization(AnalysisTestCase):
    """Test comprehensive analyze_linearization method."""
    
    def test_analyze_linearization_basic(self):
        """Test basic comprehensive analysis."""
        analyzer = SystemAnalysis(backend='numpy')
        
        result = analyzer.analyze_linearization(
            self.A_stable,
            self.B_stable,
            self.C_stable,
            system_type='continuous'
        )
        
        # Check structure
        self.assertIn('stability', result)
        self.assertIn('controllability', result)
        self.assertIn('observability', result)
        self.assertIn('summary', result)
        
        # Verify each component is valid
        self.assert_stability_info_valid(result['stability'])
        self.assert_controllability_info_valid(result['controllability'])
        self.assert_observability_info_valid(result['observability'])
    
    def test_analyze_linearization_summary(self):
        """Test summary component of comprehensive analysis."""
        analyzer = SystemAnalysis(backend='numpy')
        
        result = analyzer.analyze_linearization(
            self.A_stable,
            self.B_stable,
            self.C_stable,
            system_type='continuous'
        )
        
        summary = result['summary']
        
        # Check required summary keys
        required_keys = [
            'is_stable', 'is_controllable', 'is_observable',
            'ready_for_lqr', 'ready_for_kalman', 'ready_for_lqg',
            'stabilizable', 'detectable'
        ]
        for key in required_keys:
            self.assertIn(key, summary, f"Missing summary key: {key}")
            self.assertIsInstance(summary[key], bool)
    
    def test_analyze_linearization_ready_for_lqr(self):
        """Test ready_for_lqr flag in summary."""
        analyzer = SystemAnalysis(backend='numpy')
        
        # Controllable system
        result = analyzer.analyze_linearization(
            self.A_ctrl,
            self.B_ctrl,
            self.C_obs,
            system_type='continuous'
        )
        
        self.assertTrue(result['summary']['ready_for_lqr'])
        self.assertTrue(result['controllability']['is_controllable'])
    
    def test_analyze_linearization_ready_for_kalman(self):
        """Test ready_for_kalman flag in summary."""
        analyzer = SystemAnalysis(backend='numpy')
        
        # Observable system
        result = analyzer.analyze_linearization(
            self.A_obs,
            self.B_stable,
            self.C_obs,
            system_type='continuous'
        )
        
        self.assertTrue(result['summary']['ready_for_kalman'])
        self.assertTrue(result['observability']['is_observable'])
    
    def test_analyze_linearization_ready_for_lqg(self):
        """Test ready_for_lqg flag requires both controllable and observable."""
        analyzer = SystemAnalysis(backend='numpy')
        
        # System that is both controllable and observable
        result = analyzer.analyze_linearization(
            self.A_stable,
            self.B_stable,
            self.C_stable,
            system_type='continuous'
        )
        
        is_ctrl = result['controllability']['is_controllable']
        is_obs = result['observability']['is_observable']
        ready_lqg = result['summary']['ready_for_lqg']
        
        # ready_for_lqg should be true iff both controllable and observable
        self.assertEqual(ready_lqg, is_ctrl and is_obs)
    
    def test_analyze_linearization_discrete(self):
        """Test comprehensive analysis for discrete system."""
        analyzer = SystemAnalysis(backend='numpy')
        
        result = analyzer.analyze_linearization(
            self.Ad_stable,
            self.Bd_stable,
            self.C_stable,
            system_type='discrete'
        )
        
        # Should work for discrete systems
        self.assert_stability_info_valid(result['stability'])
        self.assertTrue(result['stability']['is_stable'])
    
    def test_analyze_linearization_default_continuous(self):
        """Test that default system_type is continuous."""
        analyzer = SystemAnalysis(backend='numpy')
        
        # Should work without explicit system_type
        result = analyzer.analyze_linearization(
            self.A_stable,
            self.B_stable,
            self.C_stable
        )
        
        self.assertIn('stability', result)
    
    def test_analyze_linearization_consistency(self):
        """Test that comprehensive analysis is consistent with individual methods."""
        analyzer = SystemAnalysis(backend='numpy')
        
        # Comprehensive analysis
        result = analyzer.analyze_linearization(
            self.A_stable,
            self.B_stable,
            self.C_stable,
            system_type='continuous'
        )
        
        # Individual analyses
        stability_indiv = analyzer.stability(self.A_stable, 'continuous')
        ctrl_indiv = analyzer.controllability(self.A_stable, self.B_stable)
        obs_indiv = analyzer.observability(self.A_stable, self.C_stable)
        
        # Should match
        self.assertEqual(
            result['stability']['is_stable'],
            stability_indiv['is_stable']
        )
        self.assertEqual(
            result['controllability']['is_controllable'],
            ctrl_indiv['is_controllable']
        )
        self.assertEqual(
            result['observability']['is_observable'],
            obs_indiv['is_observable']
        )


# ============================================================================
# Backend Consistency Tests
# ============================================================================

class TestBackendConsistency(AnalysisTestCase):
    """Test backend storage and consistency."""
    
    def test_backend_stored_correctly(self):
        """Test that backend is stored in wrapper."""
        backends = ['numpy', 'torch', 'jax']
        
        for backend in backends:
            with self.subTest(backend=backend):
                analyzer = SystemAnalysis(backend=backend)
                self.assertEqual(analyzer.backend, backend)
    
    def test_backend_not_passed_to_stability(self):
        """Test that stability method doesn't use backend parameter."""
        # Note: analyze_stability doesn't take backend parameter
        with patch('src.control.system_analysis.analyze_stability') as mock_func:
            mock_func.return_value = {
                'eigenvalues': np.array([-1, -2]),
                'magnitudes': np.array([1, 2]),
                'max_magnitude': 2.0,
                'spectral_radius': 2.0,
                'is_stable': True,
                'is_marginally_stable': False,
                'is_unstable': False
            }
            
            analyzer = SystemAnalysis(backend='torch')
            analyzer.stability(self.A_stable, system_type='continuous')
            
            # Verify backend NOT passed (stability analysis doesn't need it)
            call_kwargs = mock_func.call_args[1] if mock_func.call_args[1] else {}
            self.assertNotIn('backend', call_kwargs)
    
    def test_backend_not_passed_to_controllability(self):
        """Test that controllability method doesn't use backend parameter."""
        with patch('src.control.system_analysis.analyze_controllability') as mock_func:
            mock_func.return_value = {
                'controllability_matrix': np.eye(2),
                'rank': 2,
                'is_controllable': True,
                'uncontrollable_modes': None
            }
            
            analyzer = SystemAnalysis(backend='jax')
            analyzer.controllability(self.A_stable, self.B_stable)
            
            # Verify backend NOT passed
            call_kwargs = mock_func.call_args[1] if mock_func.call_args[1] else {}
            self.assertNotIn('backend', call_kwargs)
    
    def test_backend_not_mutated_by_methods(self):
        """Test that calling methods doesn't change backend."""
        analyzer = SystemAnalysis(backend='numpy')
        original_backend = analyzer.backend
        
        # Call various methods
        analyzer.stability(self.A_stable)
        analyzer.controllability(self.A_stable, self.B_stable)
        analyzer.observability(self.A_stable, self.C_stable)
        analyzer.analyze_linearization(
            self.A_stable, self.B_stable, self.C_stable
        )
        
        # Backend should remain unchanged
        self.assertEqual(analyzer.backend, original_backend)


# ============================================================================
# Delegation Tests
# ============================================================================

class TestDelegation(AnalysisTestCase):
    """Test proper delegation to classical functions."""
    
    def test_stability_delegates_correctly(self):
        """Test that stability delegates with correct arguments."""
        with patch('src.control.system_analysis.analyze_stability') as mock_func:
            mock_func.return_value = {
                'eigenvalues': np.array([-1, -2]),
                'magnitudes': np.array([1, 2]),
                'max_magnitude': 2.0,
                'spectral_radius': 2.0,
                'is_stable': True,
                'is_marginally_stable': False,
                'is_unstable': False
            }
            
            analyzer = SystemAnalysis(backend='numpy')
            
            analyzer.stability(
                self.A_stable,
                system_type='continuous',
                tolerance=1e-8
            )
            
            # Verify correct delegation
            mock_func.assert_called_once_with(
                self.A_stable,
                'continuous',
                1e-8
            )
    
    def test_controllability_delegates_correctly(self):
        """Test that controllability delegates with correct arguments."""
        with patch('src.control.system_analysis.analyze_controllability') as mock_func:
            mock_func.return_value = {
                'controllability_matrix': np.eye(2),
                'rank': 2,
                'is_controllable': True,
                'uncontrollable_modes': None
            }
            
            analyzer = SystemAnalysis(backend='numpy')
            
            analyzer.controllability(
                self.A_stable,
                self.B_stable,
                tolerance=1e-9
            )
            
            mock_func.assert_called_once_with(
                self.A_stable,
                self.B_stable,
                1e-9
            )
    
    def test_observability_delegates_correctly(self):
        """Test that observability delegates with correct arguments."""
        with patch('src.control.system_analysis.analyze_observability') as mock_func:
            mock_func.return_value = {
                'observability_matrix': np.eye(2),
                'rank': 2,
                'is_observable': True,
                'unobservable_modes': None
            }
            
            analyzer = SystemAnalysis(backend='numpy')
            
            analyzer.observability(
                self.A_stable,
                self.C_stable,
                tolerance=1e-7
            )
            
            mock_func.assert_called_once_with(
                self.A_stable,
                self.C_stable,
                1e-7
            )
    
    def test_analyze_linearization_calls_all_methods(self):
        """Test that analyze_linearization calls all individual methods."""
        analyzer = SystemAnalysis(backend='numpy')
        
        with patch.object(analyzer, 'stability') as mock_stab, \
             patch.object(analyzer, 'controllability') as mock_ctrl, \
             patch.object(analyzer, 'observability') as mock_obs:
            
            # Setup return values
            mock_stab.return_value = {
                'eigenvalues': np.array([-1, -2]),
                'magnitudes': np.array([1, 2]),
                'max_magnitude': 2.0,
                'spectral_radius': 2.0,
                'is_stable': True,
                'is_marginally_stable': False,
                'is_unstable': False
            }
            mock_ctrl.return_value = {
                'controllability_matrix': np.eye(2),
                'rank': 2,
                'is_controllable': True,
                'uncontrollable_modes': None
            }
            mock_obs.return_value = {
                'observability_matrix': np.eye(2),
                'rank': 2,
                'is_observable': True,
                'unobservable_modes': None
            }
            
            analyzer.analyze_linearization(
                self.A_stable,
                self.B_stable,
                self.C_stable,
                system_type='continuous'
            )
            
            # Verify all methods were called
            mock_stab.assert_called_once_with(self.A_stable, 'continuous')
            mock_ctrl.assert_called_once_with(self.A_stable, self.B_stable)
            mock_obs.assert_called_once_with(self.A_stable, self.C_stable)
    
    def test_no_state_mutation(self):
        """Test that methods don't mutate wrapper state."""
        analyzer = SystemAnalysis(backend='numpy')
        
        # Should only have one attribute initially
        self.assertEqual(len(vars(analyzer)), 1)
        
        # Call methods
        analyzer.stability(self.A_stable)
        analyzer.controllability(self.A_stable, self.B_stable)
        analyzer.observability(self.A_stable, self.C_stable)
        
        # Should still only have one attribute
        self.assertEqual(len(vars(analyzer)), 1)


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration(AnalysisTestCase):
    """Test integration and typical usage patterns."""
    
    def test_complete_system_analysis_workflow(self):
        """Test complete system analysis workflow."""
        analyzer = SystemAnalysis(backend='numpy')
        
        # Perform comprehensive analysis
        result = analyzer.analyze_linearization(
            self.A_stable,
            self.B_stable,
            self.C_stable,
            system_type='continuous'
        )
        
        # Use results for decision making
        if result['summary']['ready_for_lqr']:
            # System is controllable - could design LQR
            self.assertTrue(result['controllability']['is_controllable'])
        
        if result['summary']['ready_for_kalman']:
            # System is observable - could design Kalman filter
            self.assertTrue(result['observability']['is_observable'])
        
        if result['summary']['ready_for_lqg']:
            # Can design full LQG controller
            self.assertTrue(result['summary']['is_controllable'])
            self.assertTrue(result['summary']['is_observable'])
    
    def test_individual_analyses_match_comprehensive(self):
        """Test that individual analyses match comprehensive analysis."""
        analyzer = SystemAnalysis(backend='numpy')
        
        # Individual
        stab = analyzer.stability(self.A_stable, 'continuous')
        ctrl = analyzer.controllability(self.A_stable, self.B_stable)
        obs = analyzer.observability(self.A_stable, self.C_stable)
        
        # Comprehensive
        result = analyzer.analyze_linearization(
            self.A_stable,
            self.B_stable,
            self.C_stable,
            system_type='continuous'
        )
        
        # Should match
        assert_allclose(
            stab['eigenvalues'],
            result['stability']['eigenvalues']
        )
        self.assertEqual(
            ctrl['rank'],
            result['controllability']['rank']
        )
        self.assertEqual(
            obs['rank'],
            result['observability']['rank']
        )
    
    def test_multiple_analyses_with_same_wrapper(self):
        """Test that wrapper can be reused for multiple analyses."""
        analyzer = SystemAnalysis(backend='numpy')
        
        # Analyze multiple systems
        systems = [
            (self.A_stable, self.B_stable, self.C_stable),
            (self.A_unstable, self.B_unstable, self.C_stable),
            (self.A_obs, self.B_stable, self.C_obs),
        ]
        
        results = []
        for A, B, C in systems:
            result = analyzer.analyze_linearization(A, B, C)
            results.append(result)
        
        # All should be valid
        self.assertEqual(len(results), 3)
        for result in results:
            self.assertIn('stability', result)
            self.assertIn('controllability', result)
            self.assertIn('observability', result)
    
    def test_unstable_uncontrollable_system(self):
        """Test analysis of unstable uncontrollable system."""
        analyzer = SystemAnalysis(backend='numpy')
        
        # Unstable and uncontrollable - bad combination
        result = analyzer.analyze_linearization(
            self.A_unstable,
            self.B_unctrl,
            self.C_stable,
            system_type='continuous'
        )
        
        # Should detect problems
        self.assertTrue(result['stability']['is_unstable'])
        self.assertFalse(result['controllability']['is_controllable'])
        self.assertFalse(result['summary']['ready_for_lqr'])


# ============================================================================
# Error Propagation Tests
# ============================================================================

class TestErrorPropagation(AnalysisTestCase):
    """Test that errors from underlying functions are properly propagated."""
    
    def test_stability_invalid_system_type(self):
        """Test that invalid system type error propagates."""
        analyzer = SystemAnalysis(backend='numpy')
        
        with self.assertRaises(ValueError):
            analyzer.stability(self.A_stable, system_type='invalid')
    
    def test_stability_non_square_matrix(self):
        """Test that non-square matrix error propagates."""
        analyzer = SystemAnalysis(backend='numpy')
        
        A_wrong = np.array([[1, 2, 3], [4, 5, 6]])
        
        with self.assertRaises(ValueError):
            analyzer.stability(A_wrong)
    
    def test_controllability_dimension_mismatch(self):
        """Test that dimension mismatch error propagates."""
        analyzer = SystemAnalysis(backend='numpy')
        
        A = np.eye(2)
        B_wrong = np.array([[1], [2], [3]])  # Wrong dimensions
        
        with self.assertRaises(ValueError):
            analyzer.controllability(A, B_wrong)
    
    def test_observability_dimension_mismatch(self):
        """Test that dimension mismatch error propagates."""
        analyzer = SystemAnalysis(backend='numpy')
        
        A = np.eye(2)
        C_wrong = np.array([[1, 0, 0]])  # Wrong dimensions
        
        with self.assertRaises(ValueError):
            analyzer.observability(A, C_wrong)
    
    def test_analyze_linearization_propagates_errors(self):
        """Test that errors in comprehensive analysis propagate."""
        analyzer = SystemAnalysis(backend='numpy')
        
        A_wrong = np.array([[1, 2, 3], [4, 5, 6]])  # Non-square
        
        with self.assertRaises(ValueError):
            analyzer.analyze_linearization(
                A_wrong,
                self.B_stable,
                self.C_stable
            )


# ============================================================================
# Interface and Documentation Tests
# ============================================================================

class TestInterface(AnalysisTestCase):
    """Test that interface matches documentation."""
    
    def test_all_methods_exist(self):
        """Test that all documented methods exist."""
        analyzer = SystemAnalysis()
        
        required_methods = [
            'stability',
            'controllability',
            'observability',
            'analyze_linearization',
        ]
        
        for method in required_methods:
            self.assertTrue(
                hasattr(analyzer, method),
                f"Missing method: {method}"
            )
            self.assertTrue(
                callable(getattr(analyzer, method)),
                f"{method} is not callable"
            )
    
    def test_method_signatures_match_docs(self):
        """Test that method signatures match documentation."""
        import inspect
        
        analyzer = SystemAnalysis()
        
        # Check stability signature
        sig = inspect.signature(analyzer.stability)
        params = list(sig.parameters.keys())
        expected = ['A', 'system_type', 'tolerance']
        self.assertEqual(params, expected)
        
        # Check controllability signature
        sig = inspect.signature(analyzer.controllability)
        params = list(sig.parameters.keys())
        expected = ['A', 'B', 'tolerance']
        self.assertEqual(params, expected)
        
        # Check observability signature
        sig = inspect.signature(analyzer.observability)
        params = list(sig.parameters.keys())
        expected = ['A', 'C', 'tolerance']
        self.assertEqual(params, expected)
        
        # Check analyze_linearization signature
        sig = inspect.signature(analyzer.analyze_linearization)
        params = list(sig.parameters.keys())
        expected = ['A', 'B', 'C', 'system_type']
        self.assertEqual(params, expected)
    
    def test_return_types_match_docs(self):
        """Test that return types match documentation."""
        analyzer = SystemAnalysis(backend='numpy')
        
        # Stability should return StabilityInfo (dict)
        stab = analyzer.stability(self.A_stable)
        self.assertIsInstance(stab, dict)
        self.assertIn('is_stable', stab)
        
        # Controllability should return ControllabilityInfo (dict)
        ctrl = analyzer.controllability(self.A_stable, self.B_stable)
        self.assertIsInstance(ctrl, dict)
        self.assertIn('is_controllable', ctrl)
        
        # Observability should return ObservabilityInfo (dict)
        obs = analyzer.observability(self.A_stable, self.C_stable)
        self.assertIsInstance(obs, dict)
        self.assertIn('is_observable', obs)
        
        # Analyze_linearization should return dict with specific structure
        result = analyzer.analyze_linearization(
            self.A_stable, self.B_stable, self.C_stable
        )
        self.assertIsInstance(result, dict)
        self.assertIn('stability', result)
        self.assertIn('controllability', result)
        self.assertIn('observability', result)
        self.assertIn('summary', result)
    
    def test_default_parameter_values(self):
        """Test that default parameter values work correctly."""
        analyzer = SystemAnalysis()
        
        # stability defaults
        info = analyzer.stability(self.A_stable)
        self.assertIsInstance(info, dict)
        
        # controllability defaults
        info = analyzer.controllability(self.A_stable, self.B_stable)
        self.assertIsInstance(info, dict)
        
        # observability defaults
        info = analyzer.observability(self.A_stable, self.C_stable)
        self.assertIsInstance(info, dict)
        
        # analyze_linearization defaults
        result = analyzer.analyze_linearization(
            self.A_stable, self.B_stable, self.C_stable
        )
        self.assertIsInstance(result, dict)


# ============================================================================
# Test Suite
# ============================================================================

def suite():
    """Create test suite."""
    test_suite = unittest.TestSuite()
    
    # Add all test classes
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestSystemAnalysisInit))
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestStabilityMethod))
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestControllabilityMethod))
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestObservabilityMethod))
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestAnalyzeLinearization))
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