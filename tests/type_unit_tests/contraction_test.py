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
Unit Tests for Contraction Theory Types

Tests TypedDict definitions and usage patterns for contraction analysis
and control types.
"""

import pytest
import numpy as np
from src.types.contraction import (
    ContractionMetric,
    ContractionRate,
    ContractionAnalysisResult,
    CCMResult,
    FunnelingResult,
    IncrementalStabilityResult,
)


class TestTypeAliases:
    """Test type aliases for contraction."""
    
    def test_contraction_metric_constant(self):
        """Test constant contraction metric."""
        # Euclidean metric
        M: ContractionMetric = np.eye(3)
        
        assert M.shape == (3, 3)
        # Should be symmetric positive definite
        assert np.allclose(M, M.T)
        eigenvalues = np.linalg.eigvals(M)
        assert np.all(eigenvalues > 0)
    
    def test_contraction_metric_weighted(self):
        """Test weighted contraction metric."""
        # Diagonal metric with weights
        M: ContractionMetric = np.diag([2.0, 1.0, 0.5])
        
        assert M.shape == (3, 3)
        assert np.allclose(M, M.T)
        assert np.all(np.linalg.eigvals(M) > 0)
    
    def test_contraction_rate(self):
        """Test contraction rate."""
        beta: ContractionRate = 0.5
        
        assert beta > 0
        # Time to 95% convergence
        t_95 = 3.0 / beta
        assert t_95 == 6.0


class TestContractionAnalysisResult:
    """Test ContractionAnalysisResult TypedDict."""
    
    def test_contraction_analysis_result_creation(self):
        """Test creating contraction analysis result."""
        result: ContractionAnalysisResult = {
            'is_contracting': True,
            'contraction_rate': 0.5,
            'metric': np.eye(2),
            'metric_type': 'constant',
            'verification_method': 'LMI',
            'convergence_bound': lambda t, delta0: delta0 * np.exp(-0.5 * t),
            'exponential_convergence': True,
            'incremental_stability': True,
            'condition_number': 1.0,
        }
        
        assert result['is_contracting'] == True
        assert result['contraction_rate'] > 0
        assert result['metric'].shape == (2, 2)
    
    def test_contraction_positive_case(self):
        """Test contracting system."""
        beta = 0.8
        M = np.eye(3)
        
        result: ContractionAnalysisResult = {
            'is_contracting': True,
            'contraction_rate': beta,
            'metric': M,
            'metric_type': 'constant',
            'verification_method': 'analytic',
            'exponential_convergence': True,
            'incremental_stability': True,
        }
        
        # Contracting system
        assert result['is_contracting'] == True
        assert result['exponential_convergence'] == True
    
    def test_contraction_convergence_bound(self):
        """Test convergence bound function."""
        beta = 0.5
        
        # Exponential bound
        bound = lambda t, delta0: delta0 * np.exp(-beta * t)
        
        result: ContractionAnalysisResult = {
            'is_contracting': True,
            'contraction_rate': beta,
            'metric': np.eye(2),
            'metric_type': 'constant',
            'verification_method': 'LMI',
            'convergence_bound': bound,
            'exponential_convergence': True,
            'incremental_stability': True,
        }
        
        # Check bound
        delta0 = 1.0
        t = 2.0
        assert np.isclose(bound(t, delta0), np.exp(-beta * t))
    
    def test_contraction_negative_case(self):
        """Test non-contracting system."""
        result: ContractionAnalysisResult = {
            'is_contracting': False,
            'metric_type': 'state_dependent',
            'verification_method': 'SOS',
            'exponential_convergence': False,
            'incremental_stability': False,
        }
        
        assert result['is_contracting'] == False
        assert 'contraction_rate' not in result


class TestCCMResult:
    """Test CCMResult TypedDict."""
    
    def test_ccm_result_creation(self):
        """Test creating CCM result."""
        K = np.array([[1.5, 0.8]])
        M = 2 * np.eye(2)
        
        result: CCMResult = {
            'feedback_gain': K,
            'metric': M,
            'contraction_rate': 0.5,
            'metric_condition_number': np.array([1.0, 1.2, 1.5]),
            'contraction_verified': True,
            'robustness_margin': 0.3,
        }
        
        assert result['feedback_gain'].shape == (1, 2)
        assert result['metric'].shape == (2, 2)
        assert result['contraction_verified'] == True
    
    def test_ccm_constant_gain(self):
        """Test CCM with constant gain."""
        # Constant feedback gain
        K = np.array([[2.0, 1.0]])
        
        result: CCMResult = {
            'feedback_gain': K,
            'metric': np.eye(2),
            'contraction_rate': 0.6,
            'metric_condition_number': np.ones(1),
            'contraction_verified': True,
            'robustness_margin': 0.5,
        }
        
        # Apply to state
        x = np.array([1.0, 0.5])
        u = K @ x
        assert u.shape == (1,)
    
    def test_ccm_state_dependent_gain(self):
        """Test CCM with state-dependent gain."""
        # State-dependent gain K(x)
        def K(x):
            return np.array([[1.0 + x[0]**2, 0.5]])
        
        result: CCMResult = {
            'feedback_gain': K,
            'metric': np.eye(2),
            'contraction_rate': 0.4,
            'metric_condition_number': np.linspace(1.0, 2.0, 10),
            'contraction_verified': True,
            'robustness_margin': 0.2,
        }
        
        # Apply to state
        x = np.array([1.0, 0.5])
        u = result['feedback_gain'](x) @ x
        assert u.shape == (1,)
    
    def test_ccm_geodesic_distance(self):
        """Test geodesic distance in metric."""
        M = np.eye(2)
        
        # Geodesic distance function
        def d_M(x1, x2):
            dx = x1 - x2
            return np.sqrt(dx @ M @ dx)
        
        result: CCMResult = {
            'feedback_gain': np.array([[1.0, 0.5]]),
            'metric': M,
            'contraction_rate': 0.5,
            'metric_condition_number': np.ones(1),
            'contraction_verified': True,
            'robustness_margin': 0.3,
            'geodesic_distance': d_M,
        }
        
        # Compute distance
        x1 = np.array([1.0, 0.5])
        x2 = np.array([0.5, 0.3])
        dist = result['geodesic_distance'](x1, x2)
        assert dist > 0


class TestFunnelingResult:
    """Test FunnelingResult TypedDict."""
    
    def test_funneling_result_creation(self):
        """Test creating funnel control result."""
        # Funnel: ρ(t) = (ρ₀ - ρ_∞)e^(-λt) + ρ_∞
        rho_0 = 1.0
        rho_inf = 0.1
        lam = 0.5
        funnel = lambda t: (rho_0 - rho_inf) * np.exp(-lam * t) + rho_inf
        
        result: FunnelingResult = {
            'controller': lambda x, t: -0.5 * x,
            'tracking_funnel': funnel,
            'funnel_shape': 'exponential',
            'reference_trajectory': np.zeros((100, 2)),
            'performance_bound': funnel,
            'transient_bound': 1.5,
            'contraction_rate': 0.5,
        }
        
        assert callable(result['controller'])
        assert callable(result['tracking_funnel'])
        assert result['funnel_shape'] == 'exponential'
    
    def test_funneling_exponential_shape(self):
        """Test exponential funnel shape."""
        # Exponential funnel
        funnel = lambda t: 1.0 * np.exp(-0.5 * t) + 0.1
        
        result: FunnelingResult = {
            'controller': lambda x, t: np.zeros(1),
            'tracking_funnel': funnel,
            'funnel_shape': 'exponential',
            'reference_trajectory': np.zeros((50, 2)),
            'performance_bound': funnel,
            'transient_bound': 1.0,
            'contraction_rate': 0.5,
        }
        
        # Funnel should decay
        t_vals = [0, 2, 5, 10]
        rho_vals = [funnel(t) for t in t_vals]
        assert all(rho_vals[i] >= rho_vals[i+1] for i in range(len(rho_vals)-1))
    
    def test_funneling_controller_application(self):
        """Test applying funnel controller."""
        # Simple proportional controller
        controller = lambda x, t: -0.5 * x
        
        result: FunnelingResult = {
            'controller': controller,
            'tracking_funnel': lambda t: np.exp(-t),
            'funnel_shape': 'exponential',
            'reference_trajectory': np.zeros((20, 2)),
            'performance_bound': lambda t: np.exp(-t),
            'transient_bound': 1.0,
            'contraction_rate': 1.0,
        }
        
        # Apply controller
        x = np.array([1.0, 0.5])
        t = 1.0
        u = result['controller'](x, t)
        assert u.shape == (2,)


class TestIncrementalStabilityResult:
    """Test IncrementalStabilityResult TypedDict."""
    
    def test_incremental_stability_result_creation(self):
        """Test creating incremental stability result."""
        result: IncrementalStabilityResult = {
            'incrementally_stable': True,
            'contraction_rate': 0.5,
            'metric': np.eye(2),
            'kl_bound': lambda delta0, t: delta0 * np.exp(-0.5 * t),
            'convergence_type': 'exponential',
        }
        
        assert result['incrementally_stable'] == True
        assert result['convergence_type'] == 'exponential'
    
    def test_incremental_stability_positive(self):
        """Test incrementally stable system."""
        beta = 0.6
        
        result: IncrementalStabilityResult = {
            'incrementally_stable': True,
            'contraction_rate': beta,
            'metric': np.eye(3),
            'convergence_type': 'exponential',
        }
        
        # δ-GAS: all trajectories converge
        assert result['incrementally_stable'] == True
        assert result['contraction_rate'] > 0
    
    def test_incremental_stability_kl_bound(self):
        """Test KL stability bound."""
        # KL bound: β(||δx(0)||, t)
        kl = lambda delta0, t: delta0 * np.exp(-0.5 * t)
        
        result: IncrementalStabilityResult = {
            'incrementally_stable': True,
            'contraction_rate': 0.5,
            'kl_bound': kl,
            'convergence_type': 'exponential',
        }
        
        # KL bound should decay to 0
        delta0 = 1.0
        assert kl(delta0, 0) == delta0
        assert kl(delta0, 10) < 0.01
    
    def test_incremental_stability_convergence_types(self):
        """Test different convergence types."""
        types = ['exponential', 'asymptotic', 'finite_time']
        
        for conv_type in types:
            result: IncrementalStabilityResult = {
                'incrementally_stable': True,
                'convergence_type': conv_type,
            }
            assert result['convergence_type'] == conv_type


class TestPracticalUseCases:
    """Test realistic usage patterns."""
    
    def test_contraction_analysis_workflow(self):
        """Test complete contraction analysis."""
        # Linear system: ẋ = Ax
        A = np.array([[-1, 0.5], [0, -2]])
        
        # Check contraction
        result: ContractionAnalysisResult = {
            'is_contracting': True,
            'contraction_rate': 0.5,
            'metric': np.eye(2),
            'metric_type': 'constant',
            'verification_method': 'analytic',
            'exponential_convergence': True,
            'incremental_stability': True,
            'condition_number': 1.0,
        }
        
        if result['is_contracting']:
            beta = result['contraction_rate']
            # Design based on contraction
            assert beta > 0
    
    def test_ccm_controller_design(self):
        """Test CCM controller design workflow."""
        # Design CCM controller
        result: CCMResult = {
            'feedback_gain': np.array([[2.0, 1.0]]),
            'metric': 1.5 * np.eye(2),
            'contraction_rate': 0.6,
            'metric_condition_number': np.ones(1) * 1.5,
            'contraction_verified': True,
            'robustness_margin': 0.4,
        }
        
        if result['contraction_verified']:
            K = result['feedback_gain']
            # Apply controller
            x = np.array([1.0, 0.5])
            u = K @ x
            assert u.shape == (1,)
    
    def test_funnel_tracking(self):
        """Test funnel control for tracking."""
        # Reference trajectory
        t_ref = np.linspace(0, 10, 100)
        x_ref = np.column_stack([np.sin(t_ref), np.cos(t_ref)])
        
        # Funnel
        funnel = lambda t: 0.5 * np.exp(-0.3 * t) + 0.05
        
        result: FunnelingResult = {
            'controller': lambda x, t: -0.5 * (x - x_ref[int(t*10)]),
            'tracking_funnel': funnel,
            'funnel_shape': 'exponential',
            'reference_trajectory': x_ref,
            'performance_bound': funnel,
            'transient_bound': 0.5,
            'contraction_rate': 0.3,
        }
        
        # Simulate
        controller = result['controller']
        u = controller(np.array([1.0, 0.5]), 1.0)
        assert u.shape == (2,)


class TestNumericalProperties:
    """Test numerical properties of results."""
    
    def test_contraction_rate_positive(self):
        """Test contraction rate is positive."""
        result: ContractionAnalysisResult = {
            'is_contracting': True,
            'contraction_rate': 0.5,
            'metric': np.eye(2),
            'metric_type': 'constant',
            'verification_method': 'LMI',
            'exponential_convergence': True,
            'incremental_stability': True,
        }
        
        assert result['contraction_rate'] > 0
    
    def test_metric_positive_definite(self):
        """Test metric is positive definite."""
        M = np.array([[2, 0.5], [0.5, 1]])
        
        # Check positive definite
        eigenvalues = np.linalg.eigvals(M)
        assert np.all(eigenvalues > 0)
        
        result: ContractionAnalysisResult = {
            'is_contracting': True,
            'contraction_rate': 0.4,
            'metric': M,
            'metric_type': 'constant',
            'verification_method': 'LMI',
            'exponential_convergence': True,
            'incremental_stability': True,
        }
        
        assert np.allclose(result['metric'], result['metric'].T)
    
    def test_funnel_monotonic_decrease(self):
        """Test funnel decreases monotonically."""
        funnel = lambda t: 1.0 * np.exp(-0.5 * t) + 0.1
        
        # Sample times
        t_vals = np.linspace(0, 10, 20)
        rho_vals = [funnel(t) for t in t_vals]
        
        # Should be monotonically decreasing
        assert all(rho_vals[i] >= rho_vals[i+1] for i in range(len(rho_vals)-1))


class TestDocumentationExamples:
    """Test that documentation examples work."""
    
    def test_contraction_analysis_example(self):
        """Test ContractionAnalysisResult example from docstring."""
        result: ContractionAnalysisResult = {
            'is_contracting': True,
            'contraction_rate': 0.5,
            'metric': np.eye(2),
            'metric_type': 'constant',
            'verification_method': 'LMI',
            'exponential_convergence': True,
            'incremental_stability': True,
        }
        
        if result['is_contracting']:
            beta = result['contraction_rate']
            assert beta > 0
    
    def test_ccm_example(self):
        """Test CCMResult example structure."""
        result: CCMResult = {
            'feedback_gain': np.array([[1.5, 0.8]]),
            'metric': np.eye(2),
            'contraction_rate': 0.5,
            'metric_condition_number': np.ones(1),
            'contraction_verified': True,
            'robustness_margin': 0.3,
        }
        
        assert result['contraction_verified'] == True
    
    def test_funneling_example(self):
        """Test FunnelingResult example structure."""
        result: FunnelingResult = {
            'controller': lambda x, t: -0.5 * x,
            'tracking_funnel': lambda t: np.exp(-0.5 * t),
            'funnel_shape': 'exponential',
            'reference_trajectory': np.zeros((50, 2)),
            'performance_bound': lambda t: np.exp(-0.5 * t),
            'transient_bound': 1.0,
            'contraction_rate': 0.5,
        }
        
        assert callable(result['controller'])
        assert callable(result['tracking_funnel'])


class TestFieldPresence:
    """Test that all fields are accessible."""
    
    def test_contraction_analysis_has_required_fields(self):
        """Test ContractionAnalysisResult has core fields."""
        result: ContractionAnalysisResult = {
            'is_contracting': True,
            'contraction_rate': 0.5,
            'metric': np.eye(2),
            'metric_type': 'constant',
            'verification_method': 'LMI',
            'exponential_convergence': True,
            'incremental_stability': True,
        }
        
        assert 'is_contracting' in result
        assert 'contraction_rate' in result
        assert 'metric' in result
    
    def test_ccm_has_required_fields(self):
        """Test CCMResult has core fields."""
        result: CCMResult = {
            'feedback_gain': np.array([[1.0, 0.5]]),
            'metric': np.eye(2),
            'contraction_rate': 0.5,
            'metric_condition_number': np.ones(1),
            'contraction_verified': True,
            'robustness_margin': 0.3,
        }
        
        assert 'feedback_gain' in result
        assert 'metric' in result
        assert 'contraction_verified' in result


if __name__ == '__main__':
    pytest.main([__file__, '-v'])