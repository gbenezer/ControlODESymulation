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
Unit Tests for System Identification Types

Tests TypedDict definitions and usage patterns for system identification
result types.
"""

import pytest
import numpy as np
from src.types.identification import (
    HankelMatrix,
    ToeplitzMatrix,
    TrajectoryMatrix,
    MarkovParameters,
    SystemIDResult,
    SubspaceIDResult,
    ERAResult,
    DMDResult,
    SINDyResult,
    KoopmanResult,
)


class TestTypeAliases:
    """Test type aliases for identification."""
    
    def test_hankel_matrix(self):
        """Test HankelMatrix type alias."""
        # Build simple Hankel matrix
        data = np.array([1, 2, 3, 4, 5])
        H: HankelMatrix = np.array([
            [data[0], data[1], data[2]],
            [data[1], data[2], data[3]],
            [data[2], data[3], data[4]],
        ])
        
        assert H.shape == (3, 3)
        # Check Hankel property (constant diagonals)
        assert H[0, 1] == H[1, 0]  # Same diagonal
    
    def test_trajectory_matrix(self):
        """Test TrajectoryMatrix for DMD."""
        # Snapshot matrices
        X: TrajectoryMatrix = np.random.randn(3, 100)  # 3 states, 100 snapshots
        Y: TrajectoryMatrix = np.random.randn(3, 100)  # Next snapshots
        
        assert X.shape == Y.shape
        assert X.shape[1] == 100  # Time dimension


class TestSystemIDResult:
    """Test SystemIDResult TypedDict."""
    
    def test_system_id_result_creation(self):
        """Test creating SystemIDResult instance."""
        result: SystemIDResult = {
            'A': np.array([[0.9, 0.1], [0, 0.8]]),
            'B': np.array([[0], [1]]),
            'C': np.array([[1, 0]]),
            'D': np.array([[0]]),
            'order': 2,
            'fit_percentage': 95.5,
            'residuals': np.random.randn(100, 1),
            'method': 'n4sid',
        }
        
        assert result['A'].shape == (2, 2)
        assert result['order'] == 2
        assert result['fit_percentage'] > 90
    
    def test_system_id_with_stochastic_info(self):
        """Test SystemIDResult with stochastic components."""
        result: SystemIDResult = {
            'A': np.eye(2) * 0.9,
            'B': np.array([[0], [0.1]]),
            'C': np.array([[1, 0]]),
            'D': np.zeros((1, 1)),
            'G': np.array([[0.1], [0.05]]),  # Noise gain
            'Q': 0.01 * np.eye(2),
            'R': 0.1 * np.eye(1),
            'order': 2,
            'fit_percentage': 92.0,
            'residuals': np.random.randn(50, 1),
            'method': 'n4sid',
        }
        
        assert 'G' in result
        assert 'Q' in result
        assert 'R' in result
    
    def test_system_id_fit_quality(self):
        """Test fit quality metrics."""
        # Good fit
        result_good: SystemIDResult = {
            'A': np.eye(2),
            'B': np.zeros((2, 1)),
            'C': np.array([[1, 0]]),
            'D': np.zeros((1, 1)),
            'order': 2,
            'fit_percentage': 98.5,
            'residuals': np.random.randn(100, 1) * 0.01,  # Small residuals
            'method': 'moesp',
        }
        
        # Poor fit
        result_poor: SystemIDResult = {
            'A': np.eye(2),
            'B': np.zeros((2, 1)),
            'C': np.array([[1, 0]]),
            'D': np.zeros((1, 1)),
            'order': 2,
            'fit_percentage': 65.0,
            'residuals': np.random.randn(100, 1),  # Large residuals
            'method': 'era',
        }
        
        assert result_good['fit_percentage'] > result_poor['fit_percentage']


class TestSubspaceIDResult:
    """Test SubspaceIDResult TypedDict."""
    
    def test_subspace_id_result_creation(self):
        """Test creating SubspaceIDResult instance."""
        nx, nu, ny = 3, 1, 2
        i = 10  # Horizon
        
        result: SubspaceIDResult = {
            'A': np.random.randn(nx, nx),
            'B': np.random.randn(nx, nu),
            'C': np.random.randn(ny, nx),
            'D': np.zeros((ny, nu)),
            'observability_matrix': np.random.randn(i*ny, nx),
            'controllability_matrix': np.random.randn(nx, i*nu),
            'hankel_matrix': np.random.randn(i*ny, 100),
            'projection_matrix': np.random.randn(i*ny, nx),
            'singular_values': np.logspace(0, -3, i),
            'order': nx,
            'fit_quality': 94.2,
        }
        
        assert result['A'].shape == (nx, nx)
        assert result['observability_matrix'].shape == (i*ny, nx)
        assert result['order'] == nx
    
    def test_subspace_id_order_selection(self):
        """Test order selection via singular values."""
        result: SubspaceIDResult = {
            'A': np.eye(4),
            'B': np.zeros((4, 1)),
            'C': np.array([[1, 0, 0, 0]]),
            'D': np.zeros((1, 1)),
            'observability_matrix': np.random.randn(20, 4),
            'controllability_matrix': np.random.randn(4, 10),
            'hankel_matrix': np.random.randn(20, 50),
            'projection_matrix': np.random.randn(20, 4),
            'singular_values': np.array([10.0, 8.0, 1.0, 0.5, 0.05, 0.01]),
            'order': 4,
            'fit_quality': 93.0,
        }
        
        # Significant singular values indicate order
        sv = result['singular_values']
        # Gap after 4th singular value
        assert sv[3] > sv[4] * 10


class TestERAResult:
    """Test ERAResult TypedDict."""
    
    def test_era_result_creation(self):
        """Test creating ERAResult instance."""
        nx, nu, ny = 3, 1, 2
        
        result: ERAResult = {
            'A': np.random.randn(nx, nx),
            'B': np.random.randn(nx, nu),
            'C': np.random.randn(ny, nx),
            'D': np.zeros((ny, nu)),
            'hankel_matrix': np.random.randn(20, 20),
            'singular_values': np.logspace(0, -2, 10),
            'system_modes': np.array([0.9, 0.8+0.1j, 0.8-0.1j]),
            'order': nx,
            'observability_matrix': np.random.randn(20, nx),
            'controllability_matrix': np.random.randn(nx, 10),
        }
        
        assert result['A'].shape == (nx, nx)
        assert len(result['system_modes']) == nx
    
    def test_era_stability_check(self):
        """Test checking stability via system modes."""
        # Stable system (discrete)
        result_stable: ERAResult = {
            'A': np.diag([0.9, 0.85, 0.8]),
            'B': np.zeros((3, 1)),
            'C': np.array([[1, 0, 0]]),
            'D': np.zeros((1, 1)),
            'hankel_matrix': np.random.randn(10, 10),
            'singular_values': np.logspace(0, -2, 5),
            'system_modes': np.array([0.9, 0.85, 0.8]),
            'order': 3,
            'observability_matrix': np.random.randn(10, 3),
            'controllability_matrix': np.random.randn(3, 10),
        }
        
        # Check stability
        is_stable = bool(np.all(np.abs(result_stable['system_modes']) < 1.0))
        assert is_stable == True


class TestDMDResult:
    """Test DMDResult TypedDict."""
    
    def test_dmd_result_creation(self):
        """Test creating DMDResult instance."""
        nx = 5
        rank = 3
        
        result: DMDResult = {
            'dynamics_matrix': np.random.randn(nx, nx),
            'modes': np.random.randn(nx, rank),
            'eigenvalues': np.array([0.95, 0.9+0.1j, 0.9-0.1j]),
            'amplitudes': np.array([2.0, 1.0, 1.0]),
            'frequencies': np.array([0.0, 0.5, -0.5]),
            'growth_rates': np.array([-0.05, -0.1, -0.1]),
            'rank': rank,
            'singular_values': np.logspace(0, -2, rank),
        }
        
        assert result['modes'].shape == (nx, rank)
        assert len(result['eigenvalues']) == rank
    
    def test_dmd_modal_analysis(self):
        """Test DMD modal decomposition."""
        result: DMDResult = {
            'dynamics_matrix': np.eye(4),
            'modes': np.eye(4, 3),
            'eigenvalues': np.array([0.98, 0.95+0.1j, 0.95-0.1j]),
            'amplitudes': np.array([5.0, 2.0, 2.0]),
            'frequencies': np.array([0.0, 0.314, -0.314]),
            'growth_rates': np.array([-0.02, -0.05, -0.05]),
            'rank': 3,
            'singular_values': np.array([10.0, 5.0, 1.0]),
        }
        
        # Find dominant mode
        amplitudes = result['amplitudes']
        dominant_idx = np.argmax(np.abs(amplitudes))
        
        assert dominant_idx == 0
        assert result['frequencies'][dominant_idx] == 0.0  # DC mode


class TestSINDyResult:
    """Test SINDyResult TypedDict."""
    
    def test_sindy_result_creation(self):
        """Test creating SINDyResult instance."""
        result: SINDyResult = {
            'coefficients': np.array([
                [1.0, 0.0, -0.5],  # dx1/dt = x1 - 0.5*x3
                [0.0, 0.8, 0.0],   # dx2/dt = 0.8*x2
            ]),
            'active_terms': ['x1', 'x2', 'x1*x2'],
            'library_functions': [lambda x: x, lambda x: x**2],
            'sparsity_level': 0.33,  # 33% zeros
            'reconstruction_error': 0.05,
            'condition_number': 10.5,
            'selected_features': [0, 1, 2],
        }
        
        assert result['coefficients'].shape == (2, 3)
        assert result['sparsity_level'] < 1.0
    
    def test_sindy_sparse_structure(self):
        """Test sparsity of SINDy result."""
        coeffs = np.array([
            [1.0, 0.0, 0.0, -2.0, 0.0],
            [0.0, 1.5, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.8, 0.0, 0.0],
        ])
        
        # Count zeros
        n_zeros = np.sum(coeffs == 0.0)
        n_total = coeffs.size
        sparsity = n_zeros / n_total
        
        result: SINDyResult = {
            'coefficients': coeffs,
            'active_terms': ['x1', 'x2', 'x3', 'x1^2', 'x2^2'],
            'library_functions': [],
            'sparsity_level': sparsity,
            'reconstruction_error': 0.01,
            'condition_number': 5.0,
            'selected_features': [0, 1, 2, 3],
        }
        
        assert result['sparsity_level'] > 0.5  # More than 50% zeros


class TestKoopmanResult:
    """Test KoopmanResult TypedDict."""
    
    def test_koopman_result_creation(self):
        """Test creating KoopmanResult instance."""
        lifted_dim = 6
        
        result: KoopmanResult = {
            'koopman_operator': np.random.randn(lifted_dim, lifted_dim),
            'lifting_functions': [lambda x: x, lambda x: x**2],
            'lifted_dimension': lifted_dim,
            'eigenvalues': np.random.rand(lifted_dim) + 1j*np.random.rand(lifted_dim),
            'eigenfunctions': np.random.randn(lifted_dim, lifted_dim),
            'reconstruction_error': 0.02,
        }
        
        assert result['koopman_operator'].shape == (lifted_dim, lifted_dim)
        assert result['lifted_dimension'] == lifted_dim
    
    def test_koopman_linearity(self):
        """Test Koopman linear representation."""
        # In lifted space, dynamics should be linear
        lifted_dim = 4
        
        result: KoopmanResult = {
            'koopman_operator': np.diag([0.95, 0.90, 0.85, 0.80]),
            'lifting_functions': [
                lambda x: x[0],
                lambda x: x[1],
                lambda x: x[0]**2,
                lambda x: x[1]**2,
            ],
            'lifted_dimension': lifted_dim,
            'eigenvalues': np.array([0.95, 0.90, 0.85, 0.80]),
            'eigenfunctions': np.eye(lifted_dim),
            'reconstruction_error': 0.001,
        }
        
        # Koopman operator should be linear
        K = result['koopman_operator']
        phi0 = np.array([1.0, 0.5, 1.0, 0.25])
        phi1 = K @ phi0
        
        assert phi1.shape == (lifted_dim,)


class TestPracticalUseCases:
    """Test realistic usage patterns."""
    
    def test_system_id_workflow(self):
        """Test complete system ID workflow."""
        # Simulate data collection
        u_data = np.random.randn(200, 1)
        y_data = np.random.randn(200, 1)
        
        # Identification
        result: SystemIDResult = {
            'A': np.array([[0.95, 0.1], [0, 0.90]]),
            'B': np.array([[0], [1]]),
            'C': np.array([[1, 0]]),
            'D': np.zeros((1, 1)),
            'order': 2,
            'fit_percentage': 94.0,
            'residuals': y_data - np.random.randn(200, 1) * 0.1,
            'method': 'n4sid',
        }
        
        # Validate
        if result['fit_percentage'] > 90:
            A, B, C, D = result['A'], result['B'], result['C'], result['D']
            # Use for control design
            assert A.shape == (2, 2)
    
    def test_dmd_time_series_analysis(self):
        """Test DMD for time series analysis."""
        # Generate synthetic data
        t = np.linspace(0, 10, 100)
        data = np.sin(2*np.pi*t) + 0.5*np.sin(4*np.pi*t)
        
        # DMD
        result: DMDResult = {
            'dynamics_matrix': np.eye(2),
            'modes': np.random.randn(100, 2),
            'eigenvalues': np.array([
                np.exp(1j*2*np.pi*0.1),  # f=1 Hz
                np.exp(1j*2*np.pi*0.2),  # f=2 Hz
            ]),
            'amplitudes': np.array([1.0, 0.5]),
            'frequencies': np.array([2*np.pi, 4*np.pi]),
            'growth_rates': np.array([0.0, 0.0]),
            'rank': 2,
            'singular_values': np.array([10.0, 5.0]),
        }
        
        # Identify frequencies
        freqs = result['frequencies']
        assert len(freqs) == 2


class TestNumericalProperties:
    """Test numerical properties of results."""
    
    def test_singular_values_monotonic(self):
        """Test singular values are monotonically decreasing."""
        result: SubspaceIDResult = {
            'A': np.eye(3),
            'B': np.zeros((3, 1)),
            'C': np.array([[1, 0, 0]]),
            'D': np.zeros((1, 1)),
            'observability_matrix': np.random.randn(10, 3),
            'controllability_matrix': np.random.randn(3, 5),
            'hankel_matrix': np.random.randn(10, 20),
            'projection_matrix': np.random.randn(10, 3),
            'singular_values': np.array([10.0, 5.0, 1.0, 0.1, 0.01]),
            'order': 3,
            'fit_quality': 90.0,
        }
        
        sv = result['singular_values']
        # Should be monotonically decreasing
        assert np.all(sv[:-1] >= sv[1:])
    
    def test_dmd_eigenvalue_magnitude(self):
        """Test DMD eigenvalue properties."""
        result: DMDResult = {
            'dynamics_matrix': np.eye(3),
            'modes': np.eye(3),
            'eigenvalues': np.array([0.95, 0.90+0.1j, 0.90-0.1j]),
            'amplitudes': np.array([1.0, 0.5, 0.5]),
            'frequencies': np.array([0.0, 0.5, -0.5]),
            'growth_rates': np.array([-0.05, -0.1, -0.1]),
            'rank': 3,
            'singular_values': np.array([10.0, 5.0, 1.0]),
        }
        
        # For stable system, eigenvalues should be inside unit circle
        eigenvals = result['eigenvalues']
        magnitudes = np.abs(eigenvals)
        assert np.all(magnitudes <= 1.0)


class TestDocumentationExamples:
    """Test that documentation examples work."""
    
    def test_system_id_example(self):
        """Test SystemIDResult example from docstring."""
        result: SystemIDResult = {
            'A': np.array([[0.9, 0.1], [0, 0.8]]),
            'B': np.array([[0], [0.1]]),
            'C': np.array([[1, 0]]),
            'D': np.zeros((1, 1)),
            'order': 2,
            'fit_percentage': 95.0,
            'residuals': np.random.randn(100, 1) * 0.05,
            'method': 'n4sid',
        }
        
        A, B, C, D = result['A'], result['B'], result['C'], result['D']
        assert A.shape == (2, 2)
        assert result['fit_percentage'] > 90
    
    def test_dmd_example(self):
        """Test DMDResult example structure."""
        result: DMDResult = {
            'dynamics_matrix': np.eye(3),
            'modes': np.random.randn(3, 2),
            'eigenvalues': np.array([0.95, 0.90]),
            'amplitudes': np.array([2.0, 1.0]),
            'frequencies': np.array([0.0, 0.5]),
            'growth_rates': np.array([-0.05, -0.1]),
            'rank': 2,
            'singular_values': np.array([10.0, 5.0]),
        }
        
        modes = result['modes']
        eigenvalues = result['eigenvalues']
        assert modes.shape == (3, 2)
        assert len(eigenvalues) == 2


class TestFieldPresence:
    """Test that all fields are accessible."""
    
    def test_system_id_has_required_fields(self):
        """Test SystemIDResult has core fields."""
        result: SystemIDResult = {
            'A': np.eye(2),
            'B': np.zeros((2, 1)),
            'C': np.array([[1, 0]]),
            'D': np.zeros((1, 1)),
            'order': 2,
            'fit_percentage': 90.0,
            'residuals': np.zeros((10, 1)),
            'method': 'n4sid',
        }
        
        assert 'A' in result
        assert 'order' in result
        assert 'fit_percentage' in result
    
    def test_dmd_has_required_fields(self):
        """Test DMDResult has core fields."""
        result: DMDResult = {
            'dynamics_matrix': np.eye(3),
            'modes': np.eye(3, 2),
            'eigenvalues': np.array([0.9, 0.8]),
            'rank': 2,
        }
        
        assert 'dynamics_matrix' in result
        assert 'modes' in result
        assert 'eigenvalues' in result


if __name__ == '__main__':
    pytest.main([__file__, '-v'])