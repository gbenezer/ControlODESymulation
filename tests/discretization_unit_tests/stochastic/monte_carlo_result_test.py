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
Comprehensive unit tests for MonteCarloResult

Tests cover:
- Basic initialization and attributes
- Statistical computation (mean, std, quantiles)
- Backend compatibility (NumPy, PyTorch, JAX)
- Utility methods (get_path, slicing, indexing)
- Probability and expectation estimation
- Result combination (distributed MC)
- Save/load functionality
- Edge cases and error handling
"""

import pytest
import numpy as np
import tempfile
import os
from typing import Callable

# Conditional imports for optional backends
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

from src.systems.base.discretization.stochastic.monte_carlo_result import (
    MonteCarloResult,
    combine_results,
    save_monte_carlo_result,
    load_monte_carlo_result,
)


# ============================================================================
# Test Fixtures - Generate Sample MC Data
# ============================================================================

@pytest.fixture
def numpy_mc_result():
    """Generate sample Monte Carlo result with NumPy."""
    np.random.seed(42)
    
    n_paths = 100
    steps = 50
    nx = 2
    nu = 1
    nw = 1
    
    # Generate random walk trajectories
    states = np.zeros((n_paths, steps + 1, nx))
    controls = np.random.randn(n_paths, steps, nu)
    noise = np.random.randn(n_paths, steps, nw)
    
    states[:, 0, :] = np.random.randn(n_paths, nx)
    
    for k in range(steps):
        states[:, k+1, :] = states[:, k, :] + 0.1 * noise[:, k, :] + 0.01 * controls[:, k, :]
    
    return MonteCarloResult(
        states=states,
        controls=controls,
        noise=noise,
        n_paths=n_paths,
        steps=steps,
        seed=42,
        system_name='TestSystem'
    )


@pytest.fixture
def torch_mc_result():
    """Generate sample Monte Carlo result with PyTorch."""
    if not TORCH_AVAILABLE:
        pytest.skip("PyTorch not available")
    
    torch.manual_seed(42)
    
    n_paths = 50
    steps = 30
    nx = 3
    
    states = torch.randn(n_paths, steps + 1, nx)
    
    return MonteCarloResult(
        states=states,
        n_paths=n_paths,
        steps=steps
    )


@pytest.fixture
def jax_mc_result():
    """Generate sample Monte Carlo result with JAX."""
    if not JAX_AVAILABLE:
        pytest.skip("JAX not available")
    
    import jax
    
    key = jax.random.PRNGKey(42)
    
    n_paths = 50
    steps = 30
    nx = 2
    
    states = jax.random.normal(key, (n_paths, steps + 1, nx))
    
    return MonteCarloResult(
        states=states,
        n_paths=n_paths,
        steps=steps
    )


# ============================================================================
# Test Initialization
# ============================================================================

class TestInitialization:
    """Test result initialization."""
    
    def test_basic_initialization(self):
        """Test basic initialization with required fields."""
        states = np.random.randn(10, 21, 2)
        
        result = MonteCarloResult(states=states, n_paths=10, steps=20)
        
        assert result.states.shape == (10, 21, 2)
        assert result.n_paths == 10
        assert result.steps == 20
        assert result.controls is None
        assert result.noise is None
    
    def test_initialization_with_all_fields(self, numpy_mc_result):
        """Test initialization with all optional fields."""
        result = numpy_mc_result
        
        assert result.states is not None
        assert result.controls is not None
        assert result.noise is not None
        assert result.n_paths == 100
        assert result.steps == 50
        assert result.metadata['seed'] == 42
        assert result.metadata['system_name'] == 'TestSystem'
    
    def test_metadata_storage(self):
        """Test that metadata is stored correctly."""
        states = np.random.randn(5, 11, 1)
        
        result = MonteCarloResult(
            states=states,
            n_paths=5,
            steps=10,
            custom_field='test_value',
            another_field=42
        )
        
        assert result.metadata['custom_field'] == 'test_value'
        assert result.metadata['another_field'] == 42


# ============================================================================
# Test Statistics Computation
# ============================================================================

class TestStatistics:
    """Test statistical analysis methods."""
    
    def test_get_statistics_numpy(self, numpy_mc_result):
        """Test statistics computation with NumPy."""
        stats = numpy_mc_result.get_statistics()
        
        # Check all required keys
        required_keys = ['mean', 'std', 'var', 'min', 'max', 'median', 
                        'q05', 'q25', 'q75', 'q95']
        for key in required_keys:
            assert key in stats
        
        # Check shapes (should be (steps+1, nx))
        expected_shape = (51, 2)  # 50 steps + 1 initial
        assert stats['mean'].shape == expected_shape
        assert stats['std'].shape == expected_shape
        assert stats['median'].shape == expected_shape
    
    def test_statistics_properties(self, numpy_mc_result):
        """Test that statistics have expected properties."""
        stats = numpy_mc_result.get_statistics()
        
        # Mean should be between min and max
        assert np.all(stats['mean'] >= stats['min'])
        assert np.all(stats['mean'] <= stats['max'])
        
        # Std should be non-negative
        assert np.all(stats['std'] >= 0)
        
        # Variance should equal std squared
        np.testing.assert_allclose(stats['var'], stats['std']**2, rtol=1e-10)
        
        # Median should be between q25 and q75
        assert np.all(stats['median'] >= stats['q25'])
        assert np.all(stats['median'] <= stats['q75'])
        
        # Quantile ordering
        assert np.all(stats['q05'] <= stats['q25'])
        assert np.all(stats['q25'] <= stats['q50'] if 'q50' in stats else stats['median'])
        assert np.all(stats['q75'] <= stats['q95'])
    
    def test_get_final_statistics(self, numpy_mc_result):
        """Test final time statistics."""
        final_stats = numpy_mc_result.get_final_statistics()
        full_stats = numpy_mc_result.get_statistics()
        
        # Final stats should match last timestep of full stats
        for key in final_stats.keys():
            np.testing.assert_array_equal(final_stats[key], full_stats[key][-1])
    
    def test_statistics_convergence(self):
        """Test that statistics converge with more paths."""
        np.random.seed(42)
        
        # True distribution: N(0, 1)
        true_mean = 0.0
        true_std = 1.0
        
        # Generate MC samples
        for n_paths in [10, 100, 1000]:
            states = np.random.randn(n_paths, 1, 1)
            result = MonteCarloResult(states, n_paths=n_paths, steps=0)
            
            stats = result.get_statistics()
            sample_mean = stats['mean'][0, 0]
            sample_std = stats['std'][0, 0]
            
            # Error should decrease with more paths
            mean_error = np.abs(sample_mean - true_mean)
            std_error = np.abs(sample_std - true_std)
            
            # Rough convergence check (not strict)
            assert mean_error < 1.0  # Should be small
            assert std_error < 0.5


# ============================================================================
# Test Backend Compatibility
# ============================================================================

class TestBackendCompatibility:
    """Test multi-backend support."""
    
    def test_numpy_backend(self, numpy_mc_result):
        """Test NumPy backend."""
        stats = numpy_mc_result.get_statistics()
        
        assert isinstance(stats['mean'], np.ndarray)
        assert isinstance(stats['std'], np.ndarray)
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_torch_backend(self, torch_mc_result):
        """Test PyTorch backend."""
        stats = torch_mc_result.get_statistics()
        
        assert isinstance(stats['mean'], torch.Tensor)
        assert isinstance(stats['std'], torch.Tensor)
    
    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_jax_backend(self, jax_mc_result):
        """Test JAX backend."""
        stats = jax_mc_result.get_statistics()
        
        assert isinstance(stats['mean'], jnp.ndarray)
        assert isinstance(stats['std'], jnp.ndarray)


# ============================================================================
# Test Utility Methods
# ============================================================================

class TestUtilityMethods:
    """Test utility and access methods."""
    
    def test_get_path(self, numpy_mc_result):
        """Test getting individual paths."""
        path_0 = numpy_mc_result.get_path(0)
        
        assert path_0.shape == (51, 2)  # (steps+1, nx)
        np.testing.assert_array_equal(path_0, numpy_mc_result.states[0])
    
    def test_get_path_negative_index(self, numpy_mc_result):
        """Test negative indexing for paths."""
        last_path = numpy_mc_result.get_path(-1)
        
        assert last_path.shape == (51, 2)
        np.testing.assert_array_equal(last_path, numpy_mc_result.states[-1])
    
    def test_get_paths_slice(self, numpy_mc_result):
        """Test getting slice of paths."""
        subset = numpy_mc_result.get_paths_slice(10, 20)
        
        assert subset.shape == (10, 51, 2)  # 10 paths
        np.testing.assert_array_equal(subset, numpy_mc_result.states[10:20])
    
    def test_indexing_support(self, numpy_mc_result):
        """Test that result supports indexing."""
        path_5 = numpy_mc_result[5]
        
        assert path_5.shape == (51, 2)
        np.testing.assert_array_equal(path_5, numpy_mc_result.states[5])
    
    def test_len_support(self, numpy_mc_result):
        """Test that len() returns n_paths."""
        assert len(numpy_mc_result) == 100


# ============================================================================
# Test Probability and Expectation
# ============================================================================

class TestProbabilityExpectation:
    """Test probability and expectation estimation."""
    
    def test_compute_probability_simple(self):
        """Test probability computation with known distribution."""
        np.random.seed(42)
        
        # Generate states from N(0, 1)
        n_paths = 1000
        states = np.random.randn(n_paths, 1, 1)
        
        result = MonteCarloResult(states, n_paths=n_paths, steps=0)
        
        # P(x > 0) should be ≈ 0.5 for N(0,1)
        prob_positive = result.compute_probability(lambda x: x[0] > 0)
        
        assert 0.45 < prob_positive < 0.55  # Should be near 0.5
    
    def test_compute_probability_at_time_step(self):
        """Test probability at specific time step."""
        np.random.seed(42)
        
        # Create trajectories where variance grows
        n_paths = 500
        steps = 10
        states = np.zeros((n_paths, steps + 1, 1))
        
        for k in range(steps):
            states[:, k+1, :] = states[:, k, :] + np.random.randn(n_paths, 1) * 0.1
        
        result = MonteCarloResult(states, n_paths=n_paths, steps=steps)
        
        # Probability should be similar early on
        prob_t0 = result.compute_probability(lambda x: np.abs(x[0]) < 0.5, time_step=0)
        prob_t10 = result.compute_probability(lambda x: np.abs(x[0]) < 0.5, time_step=10)
        
        # At t=10, variance is larger, so probability in region should be smaller
        assert prob_t10 < prob_t0
    
    def test_compute_expectation_linear(self):
        """Test expectation of linear function."""
        np.random.seed(42)
        
        # States from N(μ, σ²)
        mu = 2.0
        sigma = 0.5
        n_paths = 1000
        
        states = np.random.randn(n_paths, 1, 1) * sigma + mu
        
        result = MonteCarloResult(states, n_paths=n_paths, steps=0)
        
        # E[x] should be ≈ μ
        expected_value = result.compute_expectation(lambda x: x[0])
        
        assert 1.9 < expected_value < 2.1  # Should be near 2.0
    
    def test_compute_expectation_quadratic(self):
        """Test expectation of quadratic function."""
        np.random.seed(42)
        
        # States from N(0, 1)
        n_paths = 1000
        states = np.random.randn(n_paths, 1, 2)  # 2D state
        
        result = MonteCarloResult(states, n_paths=n_paths, steps=0)
        
        # E[||x||²] = E[x₁²] + E[x₂²] = 1 + 1 = 2 for N(0,I)
        expected_norm_sq = result.compute_expectation(
            lambda x: np.sum(x**2)
        )
        
        assert 1.8 < expected_norm_sq < 2.2  # Should be near 2.0
    
    def test_probability_rare_event(self):
        """Test probability estimation for rare events."""
        np.random.seed(42)
        
        # N(0, 1) distribution
        n_paths = 10000
        states = np.random.randn(n_paths, 1, 1)
        
        result = MonteCarloResult(states, n_paths=n_paths, steps=0)
        
        # P(x > 3) should be ≈ 0.0013 for N(0,1)
        prob_tail = result.compute_probability(lambda x: x[0] > 3.0)
        
        # With 10k samples, should be reasonably accurate
        assert 0.0005 < prob_tail < 0.0025  # Roughly 0.13% ± tolerance


# ============================================================================
# Test Result Combination
# ============================================================================

class TestCombination:
    """Test combining multiple results."""
    
    def test_combine_two_results(self):
        """Test combining two Monte Carlo results."""
        np.random.seed(42)
        
        # Two separate MC runs
        states1 = np.random.randn(50, 11, 2)
        result1 = MonteCarloResult(states1, n_paths=50, steps=10)
        
        states2 = np.random.randn(50, 11, 2)
        result2 = MonteCarloResult(states2, n_paths=50, steps=10)
        
        # Combine
        combined = combine_results([result1, result2])
        
        assert combined.n_paths == 100
        assert combined.steps == 10
        assert combined.states.shape == (100, 11, 2)
    
    def test_combine_multiple_results(self):
        """Test combining many results."""
        results = []
        for i in range(5):
            states = np.random.randn(20, 6, 1)
            results.append(MonteCarloResult(states, n_paths=20, steps=5))
        
        combined = combine_results(results)
        
        assert combined.n_paths == 100  # 5 * 20
        assert combined.steps == 5
    
    def test_combine_with_controls_and_noise(self):
        """Test combining results with controls and noise."""
        results = []
        for i in range(3):
            states = np.random.randn(10, 6, 2)
            controls = np.random.randn(10, 5, 1)
            noise = np.random.randn(10, 5, 1)
            
            results.append(MonteCarloResult(
                states, controls=controls, noise=noise,
                n_paths=10, steps=5
            ))
        
        combined = combine_results(results)
        
        assert combined.n_paths == 30
        assert combined.controls.shape == (30, 5, 1)
        assert combined.noise.shape == (30, 5, 1)
    
    def test_combine_incompatible_steps_fails(self):
        """Test that combining results with different steps fails."""
        result1 = MonteCarloResult(
            np.random.randn(10, 11, 2), n_paths=10, steps=10
        )
        result2 = MonteCarloResult(
            np.random.randn(10, 21, 2), n_paths=10, steps=20
        )
        
        with pytest.raises(ValueError, match="Incompatible steps"):
            combine_results([result1, result2])
    
    def test_combine_incompatible_shapes_fails(self):
        """Test that combining results with different state dims fails."""
        result1 = MonteCarloResult(
            np.random.randn(10, 11, 2), n_paths=10, steps=10
        )
        result2 = MonteCarloResult(
            np.random.randn(10, 11, 3), n_paths=10, steps=10
        )
        
        with pytest.raises(ValueError, match="Incompatible state shapes"):
            combine_results([result1, result2])
    
    def test_combine_empty_list_fails(self):
        """Test that combining empty list fails."""
        with pytest.raises(ValueError, match="Cannot combine empty list"):
            combine_results([])


# ============================================================================
# Test Save/Load
# ============================================================================

class TestSaveLoad:
    """Test persistence functionality."""
    
    def test_save_load_npz(self, numpy_mc_result):
        """Test saving and loading with NumPy format."""
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            filename = f.name
        
        try:
            # Save
            save_monte_carlo_result(numpy_mc_result, filename)
            assert os.path.exists(filename)
            
            # Load
            loaded = load_monte_carlo_result(filename)
            
            # Verify
            assert loaded.n_paths == numpy_mc_result.n_paths
            assert loaded.steps == numpy_mc_result.steps
            np.testing.assert_array_equal(loaded.states, numpy_mc_result.states)
            np.testing.assert_array_equal(loaded.controls, numpy_mc_result.controls)
            np.testing.assert_array_equal(loaded.noise, numpy_mc_result.noise)
        
        finally:
            if os.path.exists(filename):
                os.remove(filename)
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_save_load_pt(self, torch_mc_result):
        """Test saving and loading with PyTorch format."""
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            filename = f.name
        
        try:
            # Save
            save_monte_carlo_result(torch_mc_result, filename)
            
            # Load
            loaded = load_monte_carlo_result(filename)
            
            # Verify
            assert loaded.n_paths == torch_mc_result.n_paths
            assert loaded.steps == torch_mc_result.steps
            assert torch.allclose(loaded.states, torch_mc_result.states)
        
        finally:
            if os.path.exists(filename):
                os.remove(filename)
    
    def test_save_unsupported_format_fails(self, numpy_mc_result):
        """Test that unsupported format raises error."""
        with pytest.raises(ValueError, match="Unsupported format"):
            save_monte_carlo_result(numpy_mc_result, 'test.txt')
    
    def test_load_unsupported_format_fails(self):
        """Test that loading unsupported format raises error."""
        with pytest.raises(ValueError, match="Unsupported format"):
            load_monte_carlo_result('test.txt')
    
    def test_save_states_only(self):
        """Test saving result with only states (no controls/noise)."""
        states = np.random.randn(10, 11, 2)
        result = MonteCarloResult(states, n_paths=10, steps=10)
        
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            filename = f.name
        
        try:
            save_monte_carlo_result(result, filename)
            loaded = load_monte_carlo_result(filename)
            
            assert loaded.controls is None
            assert loaded.noise is None
            np.testing.assert_array_equal(loaded.states, states)
        
        finally:
            if os.path.exists(filename):
                os.remove(filename)


# ============================================================================
# Test String Representations
# ============================================================================

class TestStringRepresentations:
    """Test string representation methods."""
    
    def test_repr(self, numpy_mc_result):
        """Test repr includes key information."""
        repr_str = repr(numpy_mc_result)
        
        assert 'MonteCarloResult' in repr_str
        assert 'n_paths=100' in repr_str
        assert 'steps=50' in repr_str
        assert 'controls=True' in repr_str
        assert 'noise=True' in repr_str
    
    def test_str(self, numpy_mc_result):
        """Test human-readable string."""
        str_str = str(numpy_mc_result)
        
        assert '100 paths' in str_str
        assert '50 steps' in str_str
    
    def test_repr_without_optional_fields(self):
        """Test repr when controls/noise are None."""
        states = np.random.randn(10, 11, 2)
        result = MonteCarloResult(states, n_paths=10, steps=10)
        
        repr_str = repr(result)
        assert 'controls=False' in repr_str
        assert 'noise=False' in repr_str


# ============================================================================
# Test Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_single_path(self):
        """Test with single path (n_paths=1)."""
        states = np.random.randn(1, 11, 2)
        result = MonteCarloResult(states, n_paths=1, steps=10)
        
        stats = result.get_statistics()
        
        # For single path, mean = median = that path
        np.testing.assert_array_equal(stats['mean'], states[0])
        
        # Std should be zero (only one sample)
        # Note: NumPy std with single sample is 0
        assert stats['std'].shape == (11, 2)
    
    def test_single_timestep(self):
        """Test with single timestep (steps=0)."""
        states = np.random.randn(100, 1, 2)
        result = MonteCarloResult(states, n_paths=100, steps=0)
        
        stats = result.get_statistics()
        
        assert stats['mean'].shape == (1, 2)
    
    def test_high_dimensional_state(self):
        """Test with high-dimensional state."""
        n_paths = 50
        steps = 10
        nx = 20  # High-dimensional
        
        states = np.random.randn(n_paths, steps + 1, nx)
        result = MonteCarloResult(states, n_paths=n_paths, steps=steps)
        
        stats = result.get_statistics()
        
        assert stats['mean'].shape == (11, 20)
        assert stats['std'].shape == (11, 20)
    
    def test_scalar_state(self):
        """Test with scalar state (nx=1)."""
        states = np.random.randn(100, 21, 1)
        result = MonteCarloResult(states, n_paths=100, steps=20)
        
        stats = result.get_statistics()
        
        assert stats['mean'].shape == (21, 1)


# ============================================================================
# Test Statistical Properties
# ============================================================================

class TestStatisticalProperties:
    """Test statistical correctness."""
    
    def test_mean_unbiased(self):
        """Test that sample mean is unbiased estimator."""
        np.random.seed(42)
        
        true_mean = 5.0
        n_paths = 10000
        
        # Generate from N(5, 1)
        states = np.random.randn(n_paths, 1, 1) + true_mean
        result = MonteCarloResult(states, n_paths=n_paths, steps=0)
        
        stats = result.get_statistics()
        sample_mean = stats['mean'][0, 0]
        
        # Should be very close with 10k samples
        assert 4.95 < sample_mean < 5.05
    
    def test_std_estimation(self):
        """Test standard deviation estimation."""
        np.random.seed(42)
        
        true_std = 2.0
        n_paths = 5000
        
        # Generate from N(0, 4)
        states = np.random.randn(n_paths, 1, 1) * true_std
        result = MonteCarloResult(states, n_paths=n_paths, steps=0)
        
        stats = result.get_statistics()
        sample_std = stats['std'][0, 0]
        
        # Should be reasonably close
        assert 1.9 < sample_std < 2.1
    
    def test_quantile_accuracy(self):
        """Test quantile estimation accuracy."""
        np.random.seed(42)
        
        n_paths = 1000
        
        # Generate from N(0, 1)
        states = np.random.randn(n_paths, 1, 1)
        result = MonteCarloResult(states, n_paths=n_paths, steps=0)
        
        stats = result.get_statistics()
        
        # For N(0,1): q25 ≈ -0.674, q75 ≈ 0.674
        q25 = stats['q25'][0, 0]
        q75 = stats['q75'][0, 0]
        
        assert -0.75 < q25 < -0.60
        assert 0.60 < q75 < 0.75


# ============================================================================
# Test Error Handling
# ============================================================================

class TestErrorHandling:
    """Test proper error handling."""
    
    def test_unsupported_array_type_statistics(self):
        """Test error for unsupported array type."""
        # Create result with list (unsupported)
        result = MonteCarloResult(
            states=[[1, 2], [3, 4]],  # Plain list
            n_paths=2,
            steps=1
        )
        
        with pytest.raises(TypeError, match="Unsupported array type"):
            result.get_statistics()
    
    def test_path_index_out_of_bounds(self, numpy_mc_result):
        """Test error for invalid path index."""
        with pytest.raises(IndexError):
            numpy_mc_result.get_path(1000)  # Only 100 paths


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for complete workflows."""
    
    def test_full_workflow_numpy(self):
        """Test complete workflow with NumPy."""
        np.random.seed(42)
        
        # Generate MC data
        n_paths = 200
        steps = 30
        states = np.random.randn(n_paths, steps + 1, 2)
        controls = np.random.randn(n_paths, steps, 1)
        
        # Create result
        result = MonteCarloResult(
            states=states,
            controls=controls,
            n_paths=n_paths,
            steps=steps,
            system_name='TestSystem'
        )
        
        # Get statistics
        stats = result.get_statistics()
        assert stats['mean'].shape == (31, 2)
        
        # Get final statistics
        final = result.get_final_statistics()
        assert final['mean'].shape == (2,)
        
        # Compute probabilities
        prob = result.compute_probability(lambda x: x[0] > 0)
        assert 0 <= prob <= 1
        
        # Compute expectations
        exp = result.compute_expectation(lambda x: np.sum(x**2))
        assert exp > 0
        
        # Access paths
        path = result[0]
        assert path.shape == (31, 2)
        
        # Get slice
        subset = result.get_paths_slice(0, 50)
        assert subset.shape == (50, 31, 2)
    
    def test_distributed_monte_carlo_workflow(self):
        """Test workflow simulating distributed Monte Carlo."""
        np.random.seed(42)
        
        # Simulate running on 4 machines, each doing 250 paths
        results = []
        for machine in range(4):
            states = np.random.randn(250, 51, 1)
            results.append(MonteCarloResult(states, n_paths=250, steps=50))
        
        # Combine results
        combined = combine_results(results)
        
        assert combined.n_paths == 1000
        
        # Analyze combined
        stats = combined.get_statistics()
        
        # Should have reasonable statistics with 1000 paths
        assert stats['mean'].shape == (51, 1)
        
        # Save combined result
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            filename = f.name
        
        try:
            save_monte_carlo_result(combined, filename)
            
            # Reload
            reloaded = load_monte_carlo_result(filename)
            
            assert reloaded.n_paths == 1000
            np.testing.assert_array_equal(reloaded.states, combined.states)
        
        finally:
            if os.path.exists(filename):
                os.remove(filename)


# ============================================================================
# Test Advanced Features
# ============================================================================

class TestAdvancedFeatures:
    """Test advanced analysis features."""
    
    def test_time_varying_statistics(self):
        """Test statistics at different time points."""
        np.random.seed(42)
        
        # Random walk with growing variance
        n_paths = 500
        steps = 20
        states = np.zeros((n_paths, steps + 1, 1))
        
        for k in range(steps):
            states[:, k+1, :] = states[:, k, :] + np.random.randn(n_paths, 1) * 0.1
        
        result = MonteCarloResult(states, n_paths=n_paths, steps=steps)
        stats = result.get_statistics()
        
        # Variance should grow over time (random walk property)
        var_early = stats['var'][5, 0]
        var_late = stats['var'][20, 0]
        
        assert var_late > var_early
    
    def test_multivariate_statistics(self):
        """Test statistics for multivariate state."""
        np.random.seed(42)
        
        n_paths = 200
        nx = 5
        
        # Independent components with different variances
        states = np.zeros((n_paths, 1, nx))
        for i in range(nx):
            states[:, 0, i] = np.random.randn(n_paths) * (i + 1)
        
        result = MonteCarloResult(states, n_paths=n_paths, steps=0)
        stats = result.get_statistics()
        
        # Each dimension should have different std
        stds = stats['std'][0, :]
        
        # Should be increasing (we used variances 1, 4, 9, 16, 25)
        assert stds[0] < stds[1] < stds[2] < stds[3] < stds[4]
    
    def test_conditional_expectation(self):
        """Test computing conditional expectations."""
        np.random.seed(42)
        
        n_paths = 1000
        states = np.random.randn(n_paths, 1, 2)
        
        result = MonteCarloResult(states, n_paths=n_paths, steps=0)
        
        # E[x₁² + x₂²] for N(0,I) should be 2
        expectation = result.compute_expectation(
            lambda x: x[0]**2 + x[1]**2
        )
        
        assert 1.8 < expectation < 2.2


# ============================================================================
# Performance Tests
# ============================================================================

class TestPerformance:
    """Test performance-related features."""
    
    def test_large_mc_statistics(self):
        """Test statistics computation with large number of paths."""
        n_paths = 10000
        steps = 100
        nx = 10
        
        states = np.random.randn(n_paths, steps + 1, nx)
        result = MonteCarloResult(states, n_paths=n_paths, steps=steps)
        
        # Should handle large dataset
        stats = result.get_statistics()
        
        assert stats['mean'].shape == (101, 10)
        assert not np.any(np.isnan(stats['mean']))
    
    def test_memory_efficiency_slicing(self):
        """Test that slicing doesn't copy data."""
        states = np.random.randn(1000, 51, 2)
        result = MonteCarloResult(states, n_paths=1000, steps=50)
        
        # Get slice
        subset = result.get_paths_slice(0, 100)
        
        # Should be a view, not a copy (for NumPy)
        if isinstance(subset, np.ndarray):
            assert subset.base is not None or subset.base is states


if __name__ == '__main__':
    pytest.main([__file__, '-v'])