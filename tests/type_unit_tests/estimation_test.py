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
Unit Tests for Nonlinear State Estimation Types

Tests TypedDict definitions and usage patterns for nonlinear estimation
result types (EKF, UKF, Particle Filter).
"""

import pytest
import numpy as np
from src.types.estimation import (
    EKFResult,
    UKFResult,
    ParticleFilterResult,
)


class TestEKFResult:
    """Test EKFResult TypedDict."""
    
    def test_ekf_result_creation(self):
        """Test creating EKFResult instance."""
        result: EKFResult = {
            'state_estimate': np.array([1.0, 0.5]),
            'covariance': np.eye(2) * 0.1,
            'innovation': np.array([0.05]),
            'innovation_covariance': np.array([[0.15]]),
            'kalman_gain': np.array([[0.5], [0.3]]),
            'likelihood': -2.5,
        }
        
        assert result['state_estimate'].shape == (2,)
        assert result['covariance'].shape == (2, 2)
        assert result['innovation'].shape == (1,)
        assert result['kalman_gain'].shape == (2, 1)
    
    def test_ekf_result_minimal(self):
        """Test EKFResult with minimal fields."""
        result: EKFResult = {
            'state_estimate': np.array([0.8, 0.3]),
            'covariance': np.diag([0.05, 0.03]),
        }
        
        assert 'state_estimate' in result
        assert 'covariance' in result
        # Optional fields may be absent
        assert 'innovation' not in result or result.get('innovation') is not None
    
    def test_ekf_result_usage(self):
        """Test using EKF result for state estimation."""
        result: EKFResult = {
            'state_estimate': np.array([1.2, -0.3]),
            'covariance': np.diag([0.1, 0.08]),
            'innovation': np.array([0.1]),
            'innovation_covariance': np.array([[0.2]]),
            'kalman_gain': np.array([[0.6], [0.4]]),
            'likelihood': -1.8,
        }
        
        # Extract estimate
        x_hat = result['state_estimate']
        P = result['covariance']
        
        # Verify covariance is positive definite
        eigenvalues = np.linalg.eigvals(P)
        assert np.all(eigenvalues > 0)
        
        # Verify symmetric
        assert np.allclose(P, P.T)
    
    def test_ekf_result_innovation_check(self):
        """Test checking innovation for outliers."""
        # Normal innovation
        result_normal: EKFResult = {
            'state_estimate': np.array([1.0, 0.0]),
            'covariance': np.eye(2),
            'innovation': np.array([0.1]),
            'innovation_covariance': np.array([[0.5]]),
            'kalman_gain': np.array([[0.5], [0.3]]),
            'likelihood': -0.5,
        }
        
        # Large innovation (possible outlier)
        result_outlier: EKFResult = {
            'state_estimate': np.array([1.0, 0.0]),
            'covariance': np.eye(2),
            'innovation': np.array([5.0]),  # Large!
            'innovation_covariance': np.array([[0.5]]),
            'kalman_gain': np.array([[0.5], [0.3]]),
            'likelihood': -15.0,  # Very negative
        }
        
        # Normalized innovation
        innov_norm = result_normal['innovation'] @ np.linalg.inv(
            result_normal['innovation_covariance']
        ) @ result_normal['innovation']
        innov_outlier = result_outlier['innovation'] @ np.linalg.inv(
            result_outlier['innovation_covariance']
        ) @ result_outlier['innovation']
        
        assert innov_outlier > innov_norm


class TestUKFResult:
    """Test UKFResult TypedDict."""
    
    def test_ukf_result_creation(self):
        """Test creating UKFResult instance."""
        nx = 2
        n_sigma = 2 * nx + 1  # 5 sigma points
        
        result: UKFResult = {
            'state_estimate': np.array([1.0, 0.5]),
            'covariance': np.eye(2) * 0.1,
            'innovation': np.array([0.05]),
            'sigma_points': np.random.randn(n_sigma, nx),
            'weights_mean': np.ones(n_sigma) / n_sigma,
            'weights_covariance': np.ones(n_sigma) / n_sigma,
        }
        
        assert result['state_estimate'].shape == (2,)
        assert result['sigma_points'].shape == (5, 2)
        assert result['weights_mean'].shape == (5,)
        assert result['weights_covariance'].shape == (5,)
    
    def test_ukf_result_weight_properties(self):
        """Test UKF weight properties."""
        n_sigma = 5
        
        result: UKFResult = {
            'state_estimate': np.array([1.0, 0.5]),
            'covariance': np.eye(2),
            'innovation': np.array([0.0]),
            'sigma_points': np.random.randn(n_sigma, 2),
            'weights_mean': np.array([0.5, 0.2, 0.1, 0.1, 0.1]),
            'weights_covariance': np.array([0.5, 0.2, 0.1, 0.1, 0.1]),
        }
        
        # Weights should sum to 1
        w_mean_sum = np.sum(result['weights_mean'])
        w_cov_sum = np.sum(result['weights_covariance'])
        
        assert np.isclose(w_mean_sum, 1.0)
        assert np.isclose(w_cov_sum, 1.0)
    
    def test_ukf_result_sigma_point_recovery(self):
        """Test recovering mean from sigma points."""
        nx = 2
        x_true = np.array([1.5, -0.5])
        
        # Create symmetric sigma points around mean
        sigma_points = np.array([
            x_true,  # Center
            x_true + np.array([1.0, 0.0]),
            x_true + np.array([0.0, 1.0]),
            x_true - np.array([1.0, 0.0]),
            x_true - np.array([0.0, 1.0]),
        ])
        
        # Uniform weights
        weights = np.ones(5) / 5
        
        result: UKFResult = {
            'state_estimate': x_true,
            'covariance': np.eye(2),
            'innovation': np.array([0.0]),
            'sigma_points': sigma_points,
            'weights_mean': weights,
            'weights_covariance': weights,
        }
        
        # Recover mean from sigma points
        x_recovered = np.sum(sigma_points.T * weights, axis=1)
        
        assert np.allclose(x_recovered, x_true)
    
    def test_ukf_result_minimal(self):
        """Test UKFResult with minimal required fields."""
        result: UKFResult = {
            'state_estimate': np.array([0.5, 0.3]),
            'covariance': np.eye(2) * 0.1,
        }
        
        assert 'state_estimate' in result
        assert 'covariance' in result


class TestParticleFilterResult:
    """Test ParticleFilterResult TypedDict."""
    
    def test_particle_filter_result_creation(self):
        """Test creating ParticleFilterResult instance."""
        n_particles = 100
        nx = 2
        
        particles = np.random.randn(n_particles, nx)
        weights = np.ones(n_particles) / n_particles
        
        result: ParticleFilterResult = {
            'state_estimate': np.mean(particles, axis=0),
            'covariance': np.cov(particles.T),
            'particles': particles,
            'weights': weights,
            'effective_sample_size': float(n_particles),
            'resampled': False,
        }
        
        assert result['particles'].shape == (100, 2)
        assert result['weights'].shape == (100,)
        assert result['effective_sample_size'] > 0
    
    def test_particle_filter_ess_computation(self):
        """Test effective sample size computation."""
        n_particles = 100
        
        # Uniform weights (no degeneracy)
        weights_uniform = np.ones(n_particles) / n_particles
        ess_uniform = 1.0 / np.sum(weights_uniform**2)
        
        result_uniform: ParticleFilterResult = {
            'state_estimate': np.zeros(2),
            'covariance': np.eye(2),
            'particles': np.random.randn(n_particles, 2),
            'weights': weights_uniform,
            'effective_sample_size': ess_uniform,
            'resampled': False,
        }
        
        # Degenerate weights (one particle has all weight)
        weights_degenerate = np.zeros(n_particles)
        weights_degenerate[0] = 1.0
        ess_degenerate = 1.0 / np.sum(weights_degenerate**2)
        
        result_degenerate: ParticleFilterResult = {
            'state_estimate': np.zeros(2),
            'covariance': np.eye(2),
            'particles': np.random.randn(n_particles, 2),
            'weights': weights_degenerate,
            'effective_sample_size': ess_degenerate,
            'resampled': True,  # Would trigger resampling
        }
        
        # ESS should be high for uniform, low for degenerate
        assert result_uniform['effective_sample_size'] > 50
        assert result_degenerate['effective_sample_size'] < 10
    
    def test_particle_filter_weighted_mean(self):
        """Test computing weighted mean from particles."""
        # Create particles with known distribution
        particles = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ])
        
        # Weights favor first particle
        weights = np.array([0.7, 0.1, 0.1, 0.1])
        
        # Weighted mean
        x_hat = np.sum(particles.T * weights, axis=1)
        
        result: ParticleFilterResult = {
            'state_estimate': x_hat,
            'covariance': np.eye(2),
            'particles': particles,
            'weights': weights,
            'effective_sample_size': 1.0 / np.sum(weights**2),
            'resampled': False,
        }
        
        # Should be close to first particle
        assert np.allclose(result['state_estimate'], np.array([0.3, 0.3]))
    
    def test_particle_filter_resampling_indicator(self):
        """Test resampling indicator."""
        # Before resampling
        result_before: ParticleFilterResult = {
            'state_estimate': np.zeros(2),
            'covariance': np.eye(2),
            'particles': np.random.randn(100, 2),
            'weights': np.random.dirichlet(np.ones(100)),
            'effective_sample_size': 30.0,  # Low ESS
            'resampled': False,
        }
        
        # After resampling
        result_after: ParticleFilterResult = {
            'state_estimate': np.zeros(2),
            'covariance': np.eye(2),
            'particles': np.random.randn(100, 2),
            'weights': np.ones(100) / 100,  # Uniform after resampling
            'effective_sample_size': 100.0,  # Full ESS
            'resampled': True,
        }
        
        assert result_before['resampled'] == False
        assert result_after['resampled'] == True
        assert result_after['effective_sample_size'] > result_before['effective_sample_size']


class TestFilterComparisons:
    """Test comparisons between different filters."""
    
    def test_ekf_ukf_same_linear_system(self):
        """Test that EKF and UKF give similar results for linear systems."""
        # For linear systems, EKF and UKF should be equivalent
        x_hat = np.array([1.0, 0.5])
        P = np.diag([0.1, 0.08])
        
        ekf_result: EKFResult = {
            'state_estimate': x_hat,
            'covariance': P,
            'innovation': np.array([0.05]),
            'innovation_covariance': np.array([[0.15]]),
            'kalman_gain': np.array([[0.5], [0.3]]),
            'likelihood': -1.5,
        }
        
        ukf_result: UKFResult = {
            'state_estimate': x_hat,
            'covariance': P,
            'innovation': np.array([0.05]),
            'sigma_points': np.random.randn(5, 2),
            'weights_mean': np.ones(5) / 5,
            'weights_covariance': np.ones(5) / 5,
        }
        
        # State estimates should match
        assert np.allclose(ekf_result['state_estimate'], ukf_result['state_estimate'])
        assert np.allclose(ekf_result['covariance'], ukf_result['covariance'])
    
    def test_particle_filter_gaussian_approximation(self):
        """Test that particle filter can approximate Gaussian."""
        # Generate particles from Gaussian
        n_particles = 1000
        x_mean = np.array([1.0, -0.5])
        P = np.diag([0.5, 0.3])
        
        particles = np.random.multivariate_normal(x_mean, P, n_particles)
        weights = np.ones(n_particles) / n_particles
        
        # Compute sample mean and covariance
        x_hat = np.average(particles, axis=0, weights=weights)
        P_hat = np.cov(particles.T, aweights=weights)
        
        result: ParticleFilterResult = {
            'state_estimate': x_hat,
            'covariance': P_hat,
            'particles': particles,
            'weights': weights,
            'effective_sample_size': float(n_particles),
            'resampled': False,
        }
        
        # Sample mean should approximate true mean
        assert np.allclose(result['state_estimate'], x_mean, atol=0.1)
        
        # Sample covariance should approximate true covariance
        assert np.allclose(result['covariance'], P, atol=0.1)


class TestNumericalProperties:
    """Test numerical properties of estimation results."""
    
    def test_ekf_covariance_positive_definite(self):
        """Test that EKF covariance is positive definite."""
        P = np.array([[0.1, 0.02], [0.02, 0.05]])
        
        result: EKFResult = {
            'state_estimate': np.array([1.0, 0.5]),
            'covariance': P,
            'innovation': np.array([0.0]),
            'innovation_covariance': np.array([[0.1]]),
            'kalman_gain': np.array([[0.5], [0.3]]),
            'likelihood': 0.0,
        }
        
        # Check positive definite
        eigenvalues = np.linalg.eigvals(result['covariance'])
        assert np.all(eigenvalues > 0)
        
        # Check symmetric
        assert np.allclose(P, P.T)
    
    def test_ukf_weights_normalize(self):
        """Test that UKF weights are normalized."""
        result: UKFResult = {
            'state_estimate': np.array([1.0, 0.0]),
            'covariance': np.eye(2),
            'innovation': np.array([0.0]),
            'sigma_points': np.random.randn(5, 2),
            'weights_mean': np.array([0.4, 0.15, 0.15, 0.15, 0.15]),
            'weights_covariance': np.array([0.4, 0.15, 0.15, 0.15, 0.15]),
        }
        
        assert np.isclose(np.sum(result['weights_mean']), 1.0)
        assert np.isclose(np.sum(result['weights_covariance']), 1.0)
    
    def test_particle_filter_weights_normalize(self):
        """Test that particle weights are normalized."""
        weights = np.random.rand(100)
        weights = weights / np.sum(weights)  # Normalize
        
        result: ParticleFilterResult = {
            'state_estimate': np.zeros(2),
            'covariance': np.eye(2),
            'particles': np.random.randn(100, 2),
            'weights': weights,
            'effective_sample_size': 50.0,
            'resampled': False,
        }
        
        assert np.isclose(np.sum(result['weights']), 1.0)
        assert np.all(result['weights'] >= 0)
        assert np.all(result['weights'] <= 1)


class TestPracticalUseCases:
    """Test realistic usage patterns."""
    
    def test_ekf_nonlinear_pendulum(self):
        """Test EKF for nonlinear pendulum estimation."""
        # Simulate EKF update for pendulum
        result: EKFResult = {
            'state_estimate': np.array([0.1, 0.05]),  # [theta, omega]
            'covariance': np.diag([0.01, 0.005]),
            'innovation': np.array([0.02]),
            'innovation_covariance': np.array([[0.05]]),
            'kalman_gain': np.array([[0.4], [0.3]]),
            'likelihood': -0.5,
        }
        
        # Use estimate
        theta_hat, omega_hat = result['state_estimate']
        
        assert -np.pi < theta_hat < np.pi
        assert isinstance(float(omega_hat), float)
    
    def test_ukf_bearings_only_tracking(self):
        """Test UKF for bearings-only tracking."""
        # 4D state: [x, y, vx, vy]
        nx = 4
        n_sigma = 2 * nx + 1
        
        result: UKFResult = {
            'state_estimate': np.array([10.0, 5.0, 0.5, -0.3]),
            'covariance': np.diag([1.0, 1.0, 0.1, 0.1]),
            'innovation': np.array([0.01]),
            'sigma_points': np.random.randn(n_sigma, nx),
            'weights_mean': np.ones(n_sigma) / n_sigma,
            'weights_covariance': np.ones(n_sigma) / n_sigma,
        }
        
        # Extract position and velocity
        pos = result['state_estimate'][:2]
        vel = result['state_estimate'][2:]
        
        assert pos.shape == (2,)
        assert vel.shape == (2,)
    
    def test_particle_filter_multimodal(self):
        """Test particle filter for multimodal distribution."""
        # Create bimodal particle distribution
        particles_mode1 = np.random.randn(500, 2) * 0.1 + np.array([1.0, 0.0])
        particles_mode2 = np.random.randn(500, 2) * 0.1 + np.array([-1.0, 0.0])
        particles = np.vstack([particles_mode1, particles_mode2])
        
        weights = np.ones(1000) / 1000
        
        result: ParticleFilterResult = {
            'state_estimate': np.average(particles, axis=0, weights=weights),
            'covariance': np.cov(particles.T, aweights=weights),
            'particles': particles,
            'weights': weights,
            'effective_sample_size': 1000.0,
            'resampled': False,
        }
        
        # Mean should be near origin (between modes)
        assert np.abs(result['state_estimate'][0]) < 0.5
        
        # Variance should be large (bimodal)
        assert result['covariance'][0, 0] > 0.5


class TestDocumentationExamples:
    """Test that documentation examples work."""
    
    def test_ekf_result_example(self):
        """Test EKFResult example from docstring."""
        result: EKFResult = {
            'state_estimate': np.array([0.1, 0.0]),
            'covariance': np.eye(2),
            'innovation': np.array([0.02]),
            'innovation_covariance': np.array([[0.1]]),
            'kalman_gain': np.array([[0.5], [0.3]]),
            'likelihood': -0.5,
        }
        
        x_hat = result['state_estimate']
        innovation = result['innovation']
        
        assert x_hat.shape == (2,)
        assert innovation.shape == (1,)
    
    def test_ukf_result_example(self):
        """Test UKFResult example structure."""
        result: UKFResult = {
            'state_estimate': np.array([1.0, 1.0, 0.1, 0.1]),
            'covariance': np.eye(4),
            'innovation': np.array([0.01]),
            'sigma_points': np.random.randn(9, 4),
            'weights_mean': np.ones(9) / 9,
            'weights_covariance': np.ones(9) / 9,
        }
        
        sigma_points = result['sigma_points']
        assert sigma_points.shape == (9, 4)
    
    def test_particle_filter_example(self):
        """Test ParticleFilterResult example structure."""
        n_particles = 1000
        particles = np.random.randn(n_particles, 2)
        weights = np.ones(n_particles) / n_particles
        
        result: ParticleFilterResult = {
            'state_estimate': np.mean(particles, axis=0),
            'covariance': np.cov(particles.T),
            'particles': particles,
            'weights': weights,
            'effective_sample_size': float(n_particles),
            'resampled': False,
        }
        
        ess = result['effective_sample_size']
        assert ess > 0
        assert ess <= n_particles


class TestFieldPresence:
    """Test that all fields are accessible."""
    
    def test_ekf_result_has_required_fields(self):
        """Test EKFResult has core fields."""
        result: EKFResult = {
            'state_estimate': np.zeros(2),
            'covariance': np.eye(2),
        }
        
        assert 'state_estimate' in result
        assert 'covariance' in result
    
    def test_ukf_result_has_required_fields(self):
        """Test UKFResult has core fields."""
        result: UKFResult = {
            'state_estimate': np.zeros(2),
            'covariance': np.eye(2),
        }
        
        assert 'state_estimate' in result
        assert 'covariance' in result
    
    def test_particle_filter_has_required_fields(self):
        """Test ParticleFilterResult has core fields."""
        result: ParticleFilterResult = {
            'state_estimate': np.zeros(2),
            'covariance': np.eye(2),
            'particles': np.zeros((10, 2)),
            'weights': np.ones(10) / 10,
            'effective_sample_size': 10.0,
            'resampled': False,
        }
        
        assert 'state_estimate' in result
        assert 'particles' in result
        assert 'weights' in result
        assert 'effective_sample_size' in result


if __name__ == '__main__':
    pytest.main([__file__, '-v'])