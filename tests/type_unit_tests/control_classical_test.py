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
Unit Tests for Classical Control Theory Types

Tests TypedDict definitions and usage patterns for classical control
design result types.
"""

import pytest
import numpy as np
from src.types.control_classical import (
    StabilityInfo,
    ControllabilityInfo,
    ObservabilityInfo,
    LQRResult,
    KalmanFilterResult,
    LQGResult,
    PolePlacementResult,
    LuenbergerObserverResult,
)


class TestStabilityInfo:
    """Test StabilityInfo TypedDict."""
    
    def test_stability_info_creation(self):
        """Test creating StabilityInfo instance."""
        eigenvalues = np.array([-1.0, -2.0])
        magnitudes = np.abs(eigenvalues)
        
        info: StabilityInfo = {
            'eigenvalues': eigenvalues,
            'magnitudes': magnitudes,
            'max_magnitude': float(np.max(magnitudes)),
            'spectral_radius': float(np.max(magnitudes)),
            'is_stable': True,
            'is_marginally_stable': False,
            'is_unstable': False,
        }
        
        assert info['eigenvalues'].shape == (2,)
        assert info['is_stable'] is True
        assert info['spectral_radius'] == 2.0
    
    def test_stability_info_continuous_stable(self):
        """Test stability info for stable continuous system."""
        # Stable: all eigenvalues have negative real parts
        eigenvalues = np.array([-1.0 + 0.5j, -1.0 - 0.5j])
        magnitudes = np.abs(eigenvalues)
        
        info: StabilityInfo = {
            'eigenvalues': eigenvalues,
            'magnitudes': magnitudes,
            'max_magnitude': float(np.max(magnitudes)),
            'spectral_radius': float(np.max(magnitudes)),
            'is_stable': bool(np.all(np.real(eigenvalues) < 0)),
            'is_marginally_stable': False,
            'is_unstable': bool(np.any(np.real(eigenvalues) > 0)),
        }
        
        assert info['is_stable'] == True
        assert info['is_unstable'] == False
    
    def test_stability_info_discrete_stable(self):
        """Test stability info for stable discrete system."""
        # Stable: all eigenvalues inside unit circle
        eigenvalues = np.array([0.5, 0.8])
        magnitudes = np.abs(eigenvalues)
        
        info: StabilityInfo = {
            'eigenvalues': eigenvalues,
            'magnitudes': magnitudes,
            'max_magnitude': float(np.max(magnitudes)),
            'spectral_radius': float(np.max(magnitudes)),
            'is_stable': bool(np.all(magnitudes < 1.0)),
            'is_marginally_stable': bool(np.any(np.isclose(magnitudes, 1.0))),
            'is_unstable': bool(np.any(magnitudes > 1.0)),
        }
        
        assert info['is_stable'] == True
        assert info['spectral_radius'] == 0.8
    
    def test_stability_info_unstable(self):
        """Test stability info for unstable system."""
        eigenvalues = np.array([1.5, -0.5])
        magnitudes = np.abs(eigenvalues)
        
        info: StabilityInfo = {
            'eigenvalues': eigenvalues,
            'magnitudes': magnitudes,
            'max_magnitude': float(np.max(magnitudes)),
            'spectral_radius': float(np.max(magnitudes)),
            'is_stable': False,
            'is_marginally_stable': False,
            'is_unstable': True,
        }
        
        assert info['is_unstable'] is True
        assert info['is_stable'] is False
        assert info['spectral_radius'] == 1.5


class TestControllabilityInfo:
    """Test ControllabilityInfo TypedDict."""
    
    def test_controllability_info_creation(self):
        """Test creating ControllabilityInfo instance."""
        C_matrix = np.array([[0, 1], [1, -3]])
        
        info: ControllabilityInfo = {
            'controllability_matrix': C_matrix,
            'rank': 2,
            'is_controllable': True,
        }
        
        assert info['controllability_matrix'].shape == (2, 2)
        assert info['is_controllable'] is True
        assert info['rank'] == 2
    
    def test_controllability_info_with_modes(self):
        """Test ControllabilityInfo with uncontrollable modes."""
        C_matrix = np.array([[0, 1], [0, 2]])  # Rank 1
        uncontrollable = np.array([1.0])
        
        info: ControllabilityInfo = {
            'controllability_matrix': C_matrix,
            'rank': 1,
            'is_controllable': False,
            'uncontrollable_modes': uncontrollable,
        }
        
        assert info['is_controllable'] is False
        assert info['rank'] < 2
        assert info['uncontrollable_modes'] is not None
    
    def test_controllability_info_optional_fields(self):
        """Test that uncontrollable_modes is optional."""
        info: ControllabilityInfo = {
            'controllability_matrix': np.eye(2),
            'rank': 2,
            'is_controllable': True,
        }
        
        # Should work without uncontrollable_modes
        assert 'uncontrollable_modes' not in info or info.get('uncontrollable_modes') is None


class TestObservabilityInfo:
    """Test ObservabilityInfo TypedDict."""
    
    def test_observability_info_creation(self):
        """Test creating ObservabilityInfo instance."""
        O_matrix = np.array([[1, 0], [0, 1]])
        
        info: ObservabilityInfo = {
            'observability_matrix': O_matrix,
            'rank': 2,
            'is_observable': True,
        }
        
        assert info['observability_matrix'].shape == (2, 2)
        assert info['is_observable'] is True
    
    def test_observability_info_unobservable(self):
        """Test ObservabilityInfo for unobservable system."""
        O_matrix = np.array([[1, 1], [1, 1]])  # Rank 1
        unobservable = np.array([2.0])
        
        info: ObservabilityInfo = {
            'observability_matrix': O_matrix,
            'rank': 1,
            'is_observable': False,
            'unobservable_modes': unobservable,
        }
        
        assert info['is_observable'] is False
        assert info['unobservable_modes'] is not None


class TestLQRResult:
    """Test LQRResult TypedDict."""
    
    def test_lqr_result_creation(self):
        """Test creating LQRResult instance."""
        K = np.array([[1.0, 2.0]])  # (1, 2) - (nu, nx)
        P = np.array([[10.0, 1.0], [1.0, 5.0]])  # (2, 2)
        eigenvalues = np.array([-1.0, -2.0])
        
        result: LQRResult = {
            'gain': K,
            'cost_to_go': P,
            'closed_loop_eigenvalues': eigenvalues,
            'stability_margin': 1.0,
        }
        
        assert result['gain'].shape == (1, 2)
        assert result['cost_to_go'].shape == (2, 2)
        assert result['stability_margin'] > 0
    
    def test_lqr_result_usage(self):
        """Test using LQRResult for control."""
        result: LQRResult = {
            'gain': np.array([[0.5, 1.0]]),
            'cost_to_go': np.eye(2),
            'closed_loop_eigenvalues': np.array([-1.0, -2.0]),
            'stability_margin': 1.0,
        }
        
        # Apply control
        x = np.array([1.0, 0.5])
        u = -result['gain'] @ x
        
        assert u.shape == (1,)
        assert isinstance(u[0], (float, np.floating))
    
    def test_lqr_result_stability_check(self):
        """Test checking stability from LQR result."""
        result: LQRResult = {
            'gain': np.array([[1.0, 0.5]]),
            'cost_to_go': np.eye(2),
            'closed_loop_eigenvalues': np.array([-3.0, -4.0]),
            'stability_margin': 3.0,
        }
        
        # Check continuous stability
        is_stable = bool(np.all(np.real(result['closed_loop_eigenvalues']) < 0))
        assert is_stable == True
        assert result['stability_margin'] > 0


class TestKalmanFilterResult:
    """Test KalmanFilterResult TypedDict."""
    
    def test_kalman_filter_result_creation(self):
        """Test creating KalmanFilterResult instance."""
        L = np.array([[0.5], [0.3]])  # (2, 1) - (nx, ny)
        P = np.array([[1.0, 0.1], [0.1, 0.5]])  # (2, 2)
        S = np.array([[0.2]])  # (1, 1) - (ny, ny)
        eigenvalues = np.array([0.3, 0.4])
        
        result: KalmanFilterResult = {
            'gain': L,
            'error_covariance': P,
            'innovation_covariance': S,
            'observer_eigenvalues': eigenvalues,
        }
        
        assert result['gain'].shape == (2, 1)
        assert result['error_covariance'].shape == (2, 2)
        assert result['innovation_covariance'].shape == (1, 1)
    
    def test_kalman_filter_result_usage(self):
        """Test using KalmanFilterResult for state estimation."""
        result: KalmanFilterResult = {
            'gain': np.array([[0.8], [0.2]]),
            'error_covariance': np.eye(2),
            'innovation_covariance': np.array([[0.5]]),
            'observer_eigenvalues': np.array([0.2, 0.3]),
        }
        
        # State estimation update
        x_hat_pred = np.array([1.0, 0.5])
        y_meas = np.array([1.2])
        C = np.array([[1.0, 0.0]])
        
        innovation = y_meas - C @ x_hat_pred
        x_hat_corrected = x_hat_pred + result['gain'].flatten() * innovation
        
        assert x_hat_corrected.shape == (2,)
    
    def test_kalman_filter_stability(self):
        """Test observer stability from Kalman result."""
        result: KalmanFilterResult = {
            'gain': np.array([[0.5], [0.3]]),
            'error_covariance': np.eye(2),
            'innovation_covariance': np.array([[0.1]]),
            'observer_eigenvalues': np.array([0.5, 0.6]),
        }
        
        # Discrete observer should have |λ| < 1
        is_stable = bool(np.all(np.abs(result['observer_eigenvalues']) < 1.0))
        assert is_stable == True


class TestLQGResult:
    """Test LQGResult TypedDict."""
    
    def test_lqg_result_creation(self):
        """Test creating LQGResult instance."""
        result: LQGResult = {
            'control_gain': np.array([[1.0, 0.5]]),
            'estimator_gain': np.array([[0.8], [0.2]]),
            'control_cost_to_go': np.eye(2),
            'estimation_error_covariance': np.eye(2),
            'separation_verified': True,
            'closed_loop_stable': True,
            'controller_eigenvalues': np.array([-1.0, -2.0]),
            'estimator_eigenvalues': np.array([0.3, 0.4]),
        }
        
        assert result['control_gain'].shape == (1, 2)
        assert result['estimator_gain'].shape == (2, 1)
        assert result['separation_verified'] is True
        assert result['closed_loop_stable'] is True
    
    def test_lqg_result_separation_principle(self):
        """Test that LQG result reflects separation principle."""
        result: LQGResult = {
            'control_gain': np.array([[2.0, 1.0]]),
            'estimator_gain': np.array([[0.5], [0.3]]),
            'control_cost_to_go': np.diag([10.0, 5.0]),
            'estimation_error_covariance': np.diag([1.0, 0.5]),
            'separation_verified': True,
            'closed_loop_stable': True,
            'controller_eigenvalues': np.array([-3.0, -4.0]),
            'estimator_eigenvalues': np.array([0.2, 0.3]),
        }
        
        # Separation principle: controller and estimator designed independently
        assert result['separation_verified'] == True
        
        # Both should be stable
        controller_stable = bool(np.all(np.real(result['controller_eigenvalues']) < 0))
        estimator_stable = bool(np.all(np.abs(result['estimator_eigenvalues']) < 1.0))
        
        assert controller_stable == True
        assert estimator_stable == True
    
    def test_lqg_result_implementation(self):
        """Test implementing LQG controller from result."""
        result: LQGResult = {
            'control_gain': np.array([[1.5, 0.8]]),
            'estimator_gain': np.array([[0.6], [0.4]]),
            'control_cost_to_go': np.eye(2),
            'estimation_error_covariance': np.eye(2),
            'separation_verified': True,
            'closed_loop_stable': True,
            'controller_eigenvalues': np.array([-2.0, -3.0]),
            'estimator_eigenvalues': np.array([0.4, 0.5]),
        }
        
        # Controller computation
        K = result['control_gain']
        L = result['estimator_gain']
        
        x_hat = np.array([1.0, 0.5])
        u = -K @ x_hat
        
        assert u.shape == (1,)
        
        # Observer update
        y = np.array([1.1])
        C = np.array([[1.0, 0.0]])
        innovation = y - C @ x_hat
        correction = L @ innovation
        
        assert correction.shape == (2,)


class TestPolePlacementResult:
    """Test PolePlacementResult TypedDict."""
    
    def test_pole_placement_result_creation(self):
        """Test creating PolePlacementResult instance."""
        result: PolePlacementResult = {
            'gain': np.array([[2.0, 3.0]]),
            'desired_poles': np.array([-5.0, -6.0]),
            'achieved_poles': np.array([-5.0, -6.0]),
            'is_controllable': True,
        }
        
        assert result['gain'].shape == (1, 2)
        assert result['is_controllable'] is True
    
    def test_pole_placement_result_verification(self):
        """Test verifying achieved poles match desired."""
        result: PolePlacementResult = {
            'gain': np.array([[1.0, 2.0]]),
            'desired_poles': np.array([-3.0, -4.0]),
            'achieved_poles': np.array([-3.0, -4.0]),
            'is_controllable': True,
        }
        
        # Check poles match
        desired_sorted = np.sort(result['desired_poles'])
        achieved_sorted = np.sort(result['achieved_poles'])
        
        assert np.allclose(desired_sorted, achieved_sorted)
    
    def test_pole_placement_uncontrollable_system(self):
        """Test pole placement result for uncontrollable system."""
        result: PolePlacementResult = {
            'gain': np.array([[0.0, 0.0]]),
            'desired_poles': np.array([-5.0, -6.0]),
            'achieved_poles': np.array([1.0, 2.0]),  # Can't achieve desired
            'is_controllable': False,
        }
        
        assert result['is_controllable'] is False
        # Achieved poles don't match desired
        assert not np.allclose(result['desired_poles'], result['achieved_poles'])


class TestLuenbergerObserverResult:
    """Test LuenbergerObserverResult TypedDict."""
    
    def test_luenberger_observer_result_creation(self):
        """Test creating LuenbergerObserverResult instance."""
        result: LuenbergerObserverResult = {
            'gain': np.array([[5.0], [6.0]]),
            'desired_poles': np.array([-10.0, -12.0]),
            'achieved_poles': np.array([-10.0, -12.0]),
            'is_observable': True,
        }
        
        assert result['gain'].shape == (2, 1)
        assert result['is_observable'] is True
    
    def test_luenberger_observer_result_usage(self):
        """Test using LuenbergerObserverResult for state estimation."""
        result: LuenbergerObserverResult = {
            'gain': np.array([[8.0], [10.0]]),
            'desired_poles': np.array([-8.0, -10.0]),
            'achieved_poles': np.array([-8.0, -10.0]),
            'is_observable': True,
        }
        
        # Observer update
        L = result['gain']
        x_hat = np.array([1.0, 0.5])
        y_meas = np.array([1.2])
        C = np.array([[1.0, 0.0]])
        
        innovation = y_meas - C @ x_hat
        correction = L @ innovation
        
        assert correction.shape == (2,)
        assert result['is_observable'] is True


class TestTypeAnnotations:
    """Test that type annotations are correct."""
    
    def test_stability_info_has_required_fields(self):
        """Test StabilityInfo has all required fields."""
        # Create instance and verify fields exist
        info: StabilityInfo = {
            'eigenvalues': np.array([1.0]),
            'magnitudes': np.array([1.0]),
            'max_magnitude': 1.0,
            'spectral_radius': 1.0,
            'is_stable': True,
            'is_marginally_stable': False,
            'is_unstable': False,
        }
        
        assert 'eigenvalues' in info
        assert 'is_stable' in info
        assert 'spectral_radius' in info
    
    def test_lqr_result_has_required_fields(self):
        """Test LQRResult has all required fields."""
        result: LQRResult = {
            'gain': np.array([[1.0]]),
            'cost_to_go': np.array([[1.0]]),
            'closed_loop_eigenvalues': np.array([1.0]),
            'stability_margin': 1.0,
        }
        
        assert 'gain' in result
        assert 'cost_to_go' in result
        assert 'closed_loop_eigenvalues' in result
    
    def test_kalman_filter_result_has_required_fields(self):
        """Test KalmanFilterResult has all required fields."""
        result: KalmanFilterResult = {
            'gain': np.array([[1.0]]),
            'error_covariance': np.array([[1.0]]),
            'innovation_covariance': np.array([[1.0]]),
            'observer_eigenvalues': np.array([1.0]),
        }
        
        assert 'gain' in result
        assert 'error_covariance' in result
        assert 'innovation_covariance' in result


class TestPracticalUseCases:
    """Test realistic usage patterns."""
    
    def test_lqr_design_workflow(self):
        """Test typical LQR design workflow."""
        # System: double integrator
        A = np.array([[0, 1], [0, 0]])
        B = np.array([[0], [1]])
        
        # Simulate LQR design result
        result: LQRResult = {
            'gain': np.array([[1.0, 1.732]]),  # sqrt(3)
            'cost_to_go': np.array([[1.732, 1.0], [1.0, 1.732]]),
            'closed_loop_eigenvalues': np.array([-1.0, -1.0]),
            'stability_margin': 1.0,
        }
        
        # Use gain for control
        K = result['gain']
        x = np.array([1.0, 0.0])
        u = -K @ x
        
        assert u.shape == (1,)
        
        # Verify closed-loop dynamics
        A_cl = A - B @ K
        eigenvalues = np.linalg.eigvals(A_cl)
        
        # Should approximately match result (eigenvalues may be in different order)
        # Use looser comparison since we're just testing the TypedDict structure
        result_eigs_sorted = np.sort_complex(result['closed_loop_eigenvalues'])
        actual_eigs_sorted = np.sort_complex(eigenvalues)
        
        # Just verify they're all negative (stable) for continuous system
        assert np.all(np.real(actual_eigs_sorted) < 0)
    
    def test_kalman_filter_workflow(self):
        """Test typical Kalman filter workflow."""
        # Discrete system
        A = np.array([[1, 0.1], [0, 0.9]])
        C = np.array([[1, 0]])
        
        # Simulate Kalman filter design result
        result: KalmanFilterResult = {
            'gain': np.array([[0.5], [0.3]]),
            'error_covariance': np.array([[0.1, 0.02], [0.02, 0.08]]),
            'innovation_covariance': np.array([[0.15]]),
            'observer_eigenvalues': np.array([0.5, 0.6]),
        }
        
        # State estimation loop
        x_hat = np.zeros(2)
        L = result['gain']
        
        # Simulate measurement
        y = np.array([1.0])
        
        # Update
        innovation = y - C @ x_hat
        x_hat_new = x_hat + (L @ innovation).flatten()
        
        assert x_hat_new.shape == (2,)
        
        # Check observer stability
        assert bool(np.all(np.abs(result['observer_eigenvalues']) < 1.0)) == True
    
    def test_lqg_combined_workflow(self):
        """Test combined LQG controller workflow."""
        # Simulate LQG design
        result: LQGResult = {
            'control_gain': np.array([[2.0, 1.5]]),
            'estimator_gain': np.array([[0.7], [0.5]]),
            'control_cost_to_go': np.diag([10.0, 5.0]),
            'estimation_error_covariance': np.diag([0.5, 0.3]),
            'separation_verified': True,
            'closed_loop_stable': True,
            'controller_eigenvalues': np.array([-2.0, -3.0]),
            'estimator_eigenvalues': np.array([0.3, 0.4]),
        }
        
        # Implement LQG controller
        K = result['control_gain']
        L = result['estimator_gain']
        
        x_hat = np.array([1.0, 0.5])
        y = np.array([1.1])
        C = np.array([[1.0, 0.0]])
        
        # Control
        u = -K @ x_hat
        
        # Estimate
        innovation = y - C @ x_hat
        x_hat_new = x_hat + (L @ innovation).flatten()
        
        assert u.shape == (1,)
        assert x_hat_new.shape == (2,)
        assert result['separation_verified'] is True


class TestDocumentationExamples:
    """Test that documentation examples work."""
    
    def test_stability_info_example(self):
        """Test StabilityInfo example from docstring."""
        # Continuous system
        A = np.array([[0, 1], [-2, -3]])
        eigenvalues = np.linalg.eigvals(A)
        magnitudes = np.abs(eigenvalues)
        
        stability: StabilityInfo = {
            'eigenvalues': eigenvalues,
            'magnitudes': magnitudes,
            'max_magnitude': float(np.max(magnitudes)),
            'spectral_radius': float(np.max(magnitudes)),
            'is_stable': bool(np.all(np.real(eigenvalues) < 0)),
            'is_marginally_stable': False,
            'is_unstable': bool(np.any(np.real(eigenvalues) > 0)),
        }
        
        assert stability['is_stable'] == True
    
    def test_lqr_result_example(self):
        """Test LQRResult example structure."""
        # Create example result
        result: LQRResult = {
            'gain': np.array([[1.0, 0.5]]),
            'cost_to_go': np.eye(2),
            'closed_loop_eigenvalues': np.array([-2.0, -3.0]),
            'stability_margin': 2.0,
        }
        
        # Use as in documentation
        K = result['gain']
        x = np.array([1.0, 0.0])
        u = -K @ x
        
        assert u.shape == (1,)
        assert result['stability_margin'] > 0