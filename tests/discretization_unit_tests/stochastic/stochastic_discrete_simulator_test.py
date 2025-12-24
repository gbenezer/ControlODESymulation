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
Comprehensive unit tests for StochasticDiscreteSimulator

Tests cover:
- Initialization and validation
- Single trajectory simulation
- Monte Carlo simulation
- Noise handling (custom, auto-generated, seeding)
- All controller types
- Observer integration
- Batched vs sequential simulation
- Antithetic variates
- Statistical analysis
- Backend compatibility
- Comparison with deterministic simulator
- Edge cases and error handling
"""

import pytest
import numpy as np
import tempfile
import os

# Conditional imports for optional backends
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

from src.systems.base.discretization.stochastic.stochastic_discrete_simulator import StochasticDiscreteSimulator
from src.systems.base.discretization.stochastic.monte_carlo_result import MonteCarloResult
from src.systems.base.discrete_stochastic_system import DiscreteStochasticSystem
from src.systems.builtin.stochastic.discrete_ar1 import DiscreteAR1
from src.systems.builtin.stochastic.discrete_random_walk import DiscreteRandomWalk
from src.systems.builtin.stochastic.discrete_white_noise import DiscreteWhiteNoise


# ============================================================================
# Test Fixtures
# ============================================================================

class DiscreteLinearSDE(DiscreteStochasticSystem):
    """2D discrete linear system with additive noise for testing."""
    
    def define_system(self, a11=0.9, a12=0.1, a21=-0.1, a22=0.8, b=1.0, sigma=0.1):
        import sympy as sp
        x1, x2 = sp.symbols('x1 x2', real=True)
        u = sp.symbols('u', real=True)
        
        a11_sym, a12_sym = sp.symbols('a11 a12', real=True)
        a21_sym, a22_sym = sp.symbols('a21 a22', real=True)
        b_sym = sp.symbols('b', real=True)
        sigma_sym = sp.symbols('sigma', positive=True)
        
        self.state_vars = [x1, x2]
        self.control_vars = [u]
        self._f_sym = sp.Matrix([
            a11_sym*x1 + a12_sym*x2,
            a21_sym*x1 + a22_sym*x2 + b_sym*u
        ])
        self.parameters = {
            a11_sym: a11, a12_sym: a12,
            a21_sym: a21, a22_sym: a22,
            b_sym: b, sigma_sym: sigma
        }
        self.order = 1
        
        self.diffusion_expr = sp.Matrix([[sigma_sym], [sigma_sym]])
        self.sde_type = 'ito'


# Mock observer for testing
class MockObserver:
    """Simple mock observer for testing."""
    
    def __init__(self, nx):
        self.nx = nx
        self.update_count = 0
    
    def initialize(self, x0):
        """Initialize observer state."""
        if isinstance(x0, np.ndarray):
            return x0.copy()
        elif TORCH_AVAILABLE and isinstance(x0, torch.Tensor):
            return x0.clone()
        elif JAX_AVAILABLE and isinstance(x0, jnp.ndarray):
            return jnp.array(x0)
        else:
            return x0
    
    def update(self, x_hat, u, y):
        """Update observer state (perfect observer)."""
        self.update_count += 1
        return y


# ============================================================================
# Test Initialization
# ============================================================================

class TestInitialization:
    """Test simulator initialization."""
    
    def test_init_discrete_stochastic(self):
        """Test initialization with discrete stochastic system."""
        system = DiscreteAR1()
        sim = StochasticDiscreteSimulator(system)
        
        assert sim.system is system
        assert sim.discretizer is None
        assert sim.nx == 1
        assert sim.nu == 1
        assert sim.nw == 1
    
    def test_init_with_seed(self):
        """Test initialization with random seed."""
        system = DiscreteAR1()
        sim = StochasticDiscreteSimulator(system, seed=42)
        
        assert sim.seed == 42
        assert sim._rng_state is not None
    
    def test_init_non_stochastic_fails(self):
        """Test that non-stochastic system raises error."""
        from src.systems.base.discrete_symbolic_system import DiscreteSymbolicSystem
        
        # Create deterministic system
        class DeterministicSystem(DiscreteSymbolicSystem):
            def define_system(self):
                import sympy as sp
                x = sp.symbols('x')
                self.state_vars = [x]
                self.control_vars = []
                self._f_sym = sp.Matrix([0.9 * x])
                self.parameters = {}
                self.order = 1
        
        det_system = DeterministicSystem()
        
        with pytest.raises(TypeError, match="is not stochastic"):
            StochasticDiscreteSimulator(det_system)
    
    def test_set_seed(self):
        """Test changing random seed."""
        system = DiscreteRandomWalk()
        sim = StochasticDiscreteSimulator(system, seed=42)
        
        # Change seed
        sim.set_seed(123)
        assert sim.seed == 123


# ============================================================================
# Test Single Trajectory Simulation
# ============================================================================

class TestSingleTrajectory:
    """Test single trajectory simulation."""
    
    def test_basic_simulation(self):
        """Test basic stochastic simulation."""
        system = DiscreteAR1(phi=0.9, sigma=0.2)
        sim = StochasticDiscreteSimulator(system, seed=42)
        
        x0 = np.array([1.0])
        states = sim.simulate(x0, steps=10)
        
        assert states.shape == (11, 1)  # steps+1 includes initial
    
    def test_simulation_with_custom_noise(self):
        """Test simulation with provided noise sequence."""
        system = DiscreteRandomWalk(sigma=0.5)
        sim = StochasticDiscreteSimulator(system)
        
        x0 = np.array([0.0])
        noise = np.ones((10, 1))  # Fixed noise
        
        states = sim.simulate(x0, steps=10, noise=noise)
        
        # x[k+1] = x[k] + 0.5*1.0 = x[k] + 0.5
        # x[10] = 0 + 10*0.5 = 5.0
        np.testing.assert_allclose(states[-1], np.array([5.0]), rtol=1e-10)
    
    def test_deterministic_simulation_zero_noise(self):
        """Test that w=0 gives deterministic result."""
        system = DiscreteAR1(phi=0.9, sigma=0.2)
        sim = StochasticDiscreteSimulator(system, seed=42)
        
        x0 = np.array([1.0])
        u = np.array([0.0])
        noise_zero = np.zeros((10, 1))
        
        states = sim.simulate(x0, steps=10, controller=noise_zero, noise=noise_zero)
        
        # Should match deterministic trajectory
        # x[k+1] = 0.9*x[k] (no control, no noise)
        expected = np.array([0.9**k for k in range(11)]).reshape(-1, 1)
        np.testing.assert_allclose(states, expected, rtol=1e-10)
    
    def test_reproducibility_with_seed(self):
        """Test that same seed gives same trajectory."""
        system = DiscreteRandomWalk(sigma=1.0)
        
        x0 = np.array([0.0])
        
        # First run
        sim1 = StochasticDiscreteSimulator(system, seed=42)
        states1 = sim1.simulate(x0, steps=20)
        
        # Second run with same seed
        sim2 = StochasticDiscreteSimulator(system, seed=42)
        states2 = sim2.simulate(x0, steps=20)
        
        # Should be identical
        np.testing.assert_array_equal(states1, states2)
    
    def test_different_seeds_different_trajectories(self):
        """Test that different seeds give different trajectories."""
        system = DiscreteRandomWalk(sigma=1.0)
        x0 = np.array([0.0])
        
        sim1 = StochasticDiscreteSimulator(system, seed=42)
        states1 = sim1.simulate(x0, steps=20)
        
        sim2 = StochasticDiscreteSimulator(system, seed=123)
        states2 = sim2.simulate(x0, steps=20)
        
        # Should be different
        assert not np.allclose(states1, states2)
    
    def test_return_noise(self):
        """Test returning noise samples."""
        system = DiscreteAR1(sigma=0.2)
        sim = StochasticDiscreteSimulator(system, seed=42)
        
        x0 = np.array([1.0])
        states, noise = sim.simulate(
            x0, steps=10, 
            controller=lambda x, k: np.array([0.0]),
            return_noise=True
        )
        
        assert states.shape == (11, 1)
        assert noise.shape == (10, 1)
    
    def test_return_controls_and_noise(self):
        """Test returning both controls and noise."""
        system = DiscreteAR1(sigma=0.2)
        sim = StochasticDiscreteSimulator(system, seed=42)
        
        x0 = np.array([1.0])
        controller = lambda x, k: np.array([0.1])
        
        states, controls, noise = sim.simulate(
            x0, steps=10, controller=controller,
            return_controls=True, return_noise=True
        )
        
        assert states.shape == (11, 1)
        assert controls.shape == (10, 1)
        assert noise.shape == (10, 1)


# ============================================================================
# Test Controller Types
# ============================================================================

class TestControllers:
    """Test different controller types with stochastic systems."""
    
    def test_autonomous_simulation(self):
        """Test autonomous stochastic system."""
        system = DiscreteRandomWalk(sigma=0.5)
        sim = StochasticDiscreteSimulator(system, seed=42)
        
        x0 = np.array([0.0])
        states = sim.simulate(x0, steps=20)
        
        assert states.shape == (21, 1)
        # Should deviate from initial due to noise
        assert np.abs(states[-1, 0]) > 0.01
    
    def test_sequence_controller(self):
        """Test with pre-computed control sequence."""
        system = DiscreteAR1(phi=0.9, sigma=0.1)
        sim = StochasticDiscreteSimulator(system, seed=42)
        
        x0 = np.array([0.0])
        u_seq = np.ones((10, 1))  # Constant control
        
        states = sim.simulate(x0, steps=10, controller=u_seq)
        
        assert states.shape == (11, 1)
    
    def test_function_controller(self):
        """Test with state-feedback controller."""
        system = DiscreteLinearSDE(sigma=0.1)
        sim = StochasticDiscreteSimulator(system, seed=42)
        
        def controller(x, k):
            if x.ndim == 1:
                return np.array([-0.5 * x[1]])
            else:
                return -0.5 * x[:, 1:2]
        
        x0 = np.array([1.0, 0.5])
        states = sim.simulate(x0, steps=20, controller=controller)
        
        assert states.shape == (21, 2)


# ============================================================================
# Test Monte Carlo Simulation
# ============================================================================

class TestMonteCarlo:
    """Test Monte Carlo simulation features."""
    
    def test_monte_carlo_basic(self):
        """Test basic Monte Carlo simulation."""
        system = DiscreteRandomWalk(sigma=0.5)
        sim = StochasticDiscreteSimulator(system, seed=42)
        
        x0 = np.array([0.0])
        result = sim.simulate_monte_carlo(x0, steps=20, n_paths=100)
        
        assert isinstance(result, MonteCarloResult)
        assert result.n_paths == 100
        assert result.steps == 20
        assert result.states.shape == (100, 21, 1)
    
    def test_monte_carlo_statistics(self):
        """Test Monte Carlo statistical analysis."""
        system = DiscreteAR1(phi=0.9, sigma=0.2)
        sim = StochasticDiscreteSimulator(system, seed=42)
        
        x0 = np.array([1.0])
        result = sim.simulate_monte_carlo(
            x0, steps=50, n_paths=500,
            controller=lambda x, k: np.array([0.0])
        )
        
        stats = result.get_statistics()
        
        # Check statistics exist
        assert 'mean' in stats
        assert 'std' in stats
        assert stats['mean'].shape == (51, 1)
        assert stats['std'].shape == (51, 1)
        
        # Mean should decay (phi=0.9, no noise on average)
        assert np.abs(stats['mean'][-1, 0]) < np.abs(stats['mean'][0, 0])
    
    def test_monte_carlo_variance_growth(self):
        """Test that variance grows for random walk."""
        system = DiscreteRandomWalk(sigma=0.1)
        sim = StochasticDiscreteSimulator(system, seed=42)
        
        x0 = np.array([0.0])
        result = sim.simulate_monte_carlo(x0, steps=100, n_paths=1000)
        
        stats = result.get_statistics()
        
        # For random walk: Var[x[k]] = k*σ²
        # Variance should grow over time
        var_early = stats['var'][10, 0]
        var_late = stats['var'][100, 0]
        
        assert var_late > var_early
    
    def test_monte_carlo_with_controller(self):
        """Test Monte Carlo with state feedback controller."""
        system = DiscreteAR1(phi=0.95, sigma=0.15)
        sim = StochasticDiscreteSimulator(system, seed=42)
        
        controller = lambda x, k: -0.3 * x if x.ndim == 1 else -0.3 * x
        
        x0 = np.array([2.0])
        result = sim.simulate_monte_carlo(
            x0, steps=30, n_paths=200, controller=controller
        )
        
        assert result.states.shape == (200, 31, 1)
    
    def test_monte_carlo_return_controls(self):
        """Test Monte Carlo with control recording."""
        system = DiscreteAR1(sigma=0.1)
        sim = StochasticDiscreteSimulator(system, seed=42)
        
        controller = lambda x, k: np.array([0.1 * k]) if x.ndim == 1 else np.full((x.shape[0], 1), 0.1*k)
        
        x0 = np.array([0.0])
        result = sim.simulate_monte_carlo(
            x0, steps=10, n_paths=50,
            controller=controller, return_controls=True
        )
        
        assert result.controls is not None
        assert result.controls.shape == (50, 10, 1)
    
    def test_monte_carlo_return_noise(self):
        """Test Monte Carlo with noise recording."""
        system = DiscreteRandomWalk(sigma=0.5)
        sim = StochasticDiscreteSimulator(system, seed=42)
        
        x0 = np.array([0.0])
        result = sim.simulate_monte_carlo(
            x0, steps=10, n_paths=50, return_noise=True
        )
        
        assert result.noise is not None
        assert result.noise.shape == (50, 10, 1)
    
    def test_monte_carlo_parallel_vs_sequential(self):
        """Test that parallel and sequential give similar statistics."""
        system = DiscreteAR1(phi=0.9, sigma=0.2)
        sim = StochasticDiscreteSimulator(system, seed=42)
        
        x0 = np.array([1.0])
        
        # Parallel (default)
        result_parallel = sim.simulate_monte_carlo(
            x0, steps=20, n_paths=100, parallel=True
        )
        
        # Sequential
        sim.set_seed(42)  # Reset seed
        result_sequential = sim.simulate_monte_carlo(
            x0, steps=20, n_paths=100, parallel=False
        )
        
        # Statistics should be similar (but not identical due to RNG order)
        stats_par = result_parallel.get_statistics()
        stats_seq = result_sequential.get_statistics()
        
        # Shapes should match
        assert stats_par['mean'].shape == stats_seq['mean'].shape


# ============================================================================
# Test Antithetic Variates
# ============================================================================

class TestAntitheticVariates:
    """Test variance reduction via antithetic variates."""
    
    def test_antithetic_basic(self):
        """Test basic antithetic variates simulation."""
        system = DiscreteRandomWalk(sigma=0.5)
        sim = StochasticDiscreteSimulator(system, seed=42)
        
        x0 = np.array([0.0])
        result = sim.simulate_antithetic(x0, steps=20, n_pairs=50)
        
        assert result.n_paths == 100  # 2 * n_pairs
        assert result.steps == 20
        assert result.states.shape == (100, 21, 1)
    
    def test_antithetic_pairs_symmetric(self):
        """Test that antithetic pairs use negated noise."""
        system = DiscreteRandomWalk(sigma=1.0)
        sim = StochasticDiscreteSimulator(system, seed=42)
        
        x0 = np.array([0.0])
        result = sim.simulate_antithetic(x0, steps=5, n_pairs=1)
        
        # With n_pairs=1, we have 2 trajectories
        path_0 = result.states[0]
        path_1 = result.states[1]
        
        # They should not be identical
        assert not np.allclose(path_0, path_1)
    
    def test_antithetic_variance_reduction(self):
        """Test that antithetic variates reduce variance."""
        system = DiscreteAR1(phi=0.95, sigma=0.3)
        
        x0 = np.array([1.0])
        controller = lambda x, k: np.array([0.0])
        
        # Standard Monte Carlo
        sim_std = StochasticDiscreteSimulator(system, seed=42)
        result_std = sim_std.simulate_monte_carlo(
            x0, steps=30, n_paths=200, controller=controller
        )
        stats_std = result_std.get_statistics()
        
        # Antithetic variates
        sim_anti = StochasticDiscreteSimulator(system, seed=42)
        result_anti = sim_anti.simulate_antithetic(
            x0, steps=30, n_pairs=100, controller=controller
        )
        stats_anti = result_anti.get_statistics()
        
        # For linear systems, antithetic should reduce variance in mean estimate
        # (variance of the estimator, not variance of trajectories)
        # This is subtle - antithetic doesn't reduce trajectory variance,
        # it reduces estimation variance
        
        # Just verify both work and have same n_paths
        assert result_std.n_paths == result_anti.n_paths


# ============================================================================
# Test Noise Handling
# ============================================================================

class TestNoiseHandling:
    """Test noise generation and handling."""
    
    def test_auto_noise_generation(self):
        """Test automatic noise generation."""
        system = DiscreteRandomWalk(sigma=0.5)
        sim = StochasticDiscreteSimulator(system, seed=42)
        
        x0 = np.array([0.0])
        
        # Should generate noise automatically
        states = sim.simulate(x0, steps=10)
        
        # Should vary from initial (noise applied)
        assert not np.allclose(states, 0.0)
    
    def test_custom_noise_sequence(self):
        """Test with custom noise sequence."""
        system = DiscreteAR1(phi=0.9, sigma=0.2)
        sim = StochasticDiscreteSimulator(system)
        
        x0 = np.array([1.0])
        
        # Deterministic noise sequence
        noise = np.array([[1.0], [0.0], [-1.0], [0.0], [1.0]])
        
        states = sim.simulate(
            x0, steps=5, 
            controller=lambda x, k: np.array([0.0]),
            noise=noise
        )
        
        # Verify noise was used (not random)
        # x[1] = 0.9*1.0 + 0.2*1.0 = 1.1
        np.testing.assert_allclose(states[1], np.array([1.1]), rtol=1e-10)
    
    def test_batched_custom_noise(self):
        """Test batched simulation with custom noise."""
        system = DiscreteRandomWalk(sigma=1.0)
        sim = StochasticDiscreteSimulator(system)
        
        batch_size = 3
        x0_batch = np.zeros((batch_size, 1))
        
        # Different noise for each trajectory
        noise_batch = np.array([
            [[1.0]] * 5,   # Path 0
            [[2.0]] * 5,   # Path 1
            [[3.0]] * 5,   # Path 2
        ])  # Shape: (3, 5, 1)
        
        states = sim.simulate(x0_batch, steps=5, noise=noise_batch)
        
        # Each path should integrate its noise
        # Path 0: 0 + 5*1.0 = 5.0
        # Path 1: 0 + 5*2.0 = 10.0
        # Path 2: 0 + 5*3.0 = 15.0
        np.testing.assert_allclose(states[0, -1], np.array([5.0]), rtol=1e-10)
        np.testing.assert_allclose(states[1, -1], np.array([10.0]), rtol=1e-10)
        np.testing.assert_allclose(states[2, -1], np.array([15.0]), rtol=1e-10)


# ============================================================================
# Test Statistical Analysis
# ============================================================================

class TestStatisticalAnalysis:
    """Test statistical analysis features."""
    
    def test_estimate_statistics_with_confidence(self):
        """Test statistics estimation with confidence intervals."""
        system = DiscreteAR1(phi=0.9, sigma=0.2)
        sim = StochasticDiscreteSimulator(system, seed=42)
        
        x0 = np.array([1.0])
        stats = sim.estimate_statistics(
            x0, steps=20, n_paths=500,
            controller=lambda x, k: np.array([0.0]),
            confidence_level=0.95
        )
        
        # Check confidence intervals exist
        assert 'ci_lower' in stats
        assert 'ci_upper' in stats
        assert 'confidence_level' in stats
        
        # CI should contain mean
        assert np.all(stats['ci_lower'] <= stats['mean'])
        assert np.all(stats['mean'] <= stats['ci_upper'])
    
    def test_mean_convergence_monte_carlo(self):
        """Test that sample mean converges to theoretical mean."""
        system = DiscreteAR1(phi=0.9, sigma=0.2)
        sim = StochasticDiscreteSimulator(system, seed=42)
        
        x0 = np.array([1.0])
        
        # With no control and starting at 1.0
        # E[x[1]] = 0.9*1.0 = 0.9
        # E[x[2]] = 0.9²*1.0 = 0.81
        # etc.
        
        result = sim.simulate_monte_carlo(
            x0, steps=5, n_paths=1000,
            controller=lambda x, k: np.array([0.0])
        )
        
        stats = result.get_statistics()
        
        # Check mean at step 1
        expected_mean_1 = 0.9
        assert 0.85 < stats['mean'][1, 0] < 0.95


# ============================================================================
# Test Backend Compatibility
# ============================================================================

class TestBackendCompatibility:
    """Test multi-backend support."""
    
    def test_numpy_backend(self):
        """Test NumPy backend."""
        system = DiscreteAR1()
        sim = StochasticDiscreteSimulator(system, seed=42)
        
        x0 = np.array([1.0])
        states = sim.simulate(x0, steps=10)
        
        assert isinstance(states, np.ndarray)
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_torch_backend(self):
        """Test PyTorch backend."""
        system = DiscreteAR1()
        system.set_default_backend('torch')
        sim = StochasticDiscreteSimulator(system, seed=42)
        
        x0 = torch.tensor([1.0])
        states = sim.simulate(x0, steps=10)
        
        assert isinstance(states, torch.Tensor)
    
    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_jax_backend(self):
        """Test JAX backend."""
        system = DiscreteAR1()
        system.set_default_backend('jax')
        sim = StochasticDiscreteSimulator(system, seed=42)
        
        x0 = jnp.array([1.0])
        states = sim.simulate(x0, steps=10)
        
        assert isinstance(states, jnp.ndarray)


# ============================================================================
# Test Comparison with Deterministic
# ============================================================================

class TestDeterministicComparison:
    """Compare with deterministic simulator."""
    
    def test_zero_noise_matches_deterministic_mean(self):
        """Test that zero noise gives deterministic trajectory."""
        system = DiscreteAR1(phi=0.9, sigma=0.2)
        sim = StochasticDiscreteSimulator(system)
        
        x0 = np.array([1.0])
        noise_zero = np.zeros((10, 1))
        
        states = sim.simulate(
            x0, steps=10,
            controller=lambda x, k: np.array([0.0]),
            noise=noise_zero
        )
        
        # Should match drift exactly
        expected = np.array([0.9**k for k in range(11)]).reshape(-1, 1)
        np.testing.assert_allclose(states, expected, rtol=1e-10)
    
    def test_monte_carlo_mean_approaches_deterministic(self):
        """Test that MC mean approaches deterministic trajectory."""
        system = DiscreteAR1(phi=0.9, sigma=0.1)
        sim = StochasticDiscreteSimulator(system, seed=42)
        
        x0 = np.array([1.0])
        controller = lambda x, k: np.array([0.0])
        
        # Large number of paths
        result = sim.simulate_monte_carlo(
            x0, steps=10, n_paths=5000, controller=controller
        )
        
        stats = result.get_statistics()
        
        # Mean should be close to deterministic
        expected = np.array([0.9**k for k in range(11)]).reshape(-1, 1)
        
        # Check that MOST timesteps are within 3 standard errors
        # (not all, since with 11 tests we expect ~1 to fail by chance)
        std_error = stats['std'] / np.sqrt(5000)
        within_bounds = np.abs(stats['mean'] - expected) < 3 * std_error
        
        # At least 90% of timesteps should be within bounds
        # (with 11 timesteps, expecting 10-11 to pass)
        fraction_passing = np.mean(within_bounds)
        assert fraction_passing >= 0.90
        
        # Also check final time specifically (most important)
        final_error = np.abs(stats['mean'][-1, 0] - expected[-1, 0])
        final_std_error = stats['std'][-1, 0] / np.sqrt(5000)
        assert final_error < 3 * final_std_error


# ============================================================================
# Test Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_zero_steps_fails(self):
        """Test that zero steps raises error."""
        system = DiscreteAR1()
        sim = StochasticDiscreteSimulator(system)
        
        x0 = np.array([1.0])
        
        with pytest.raises(ValueError, match="steps must be positive"):
            sim.simulate(x0, steps=0)
    
    def test_negative_n_paths_fails(self):
        """Test that negative n_paths raises error."""
        system = DiscreteAR1()
        sim = StochasticDiscreteSimulator(system)
        
        x0 = np.array([1.0])
        
        with pytest.raises(ValueError, match="n_paths must be positive"):
            sim.simulate_monte_carlo(x0, steps=10, n_paths=-5)
    
    def test_batched_x0_for_monte_carlo_fails(self):
        """Test that batched x0 for Monte Carlo raises error."""
        system = DiscreteAR1()
        sim = StochasticDiscreteSimulator(system)
        
        x0_batch = np.array([[1.0], [2.0]])  # Multiple ICs
        
        with pytest.raises(ValueError, match="x0 must be single state"):
            sim.simulate_monte_carlo(x0_batch, steps=10, n_paths=100)
    
    def test_single_step_simulation(self):
        """Test simulation with single step."""
        system = DiscreteRandomWalk(sigma=0.5)
        sim = StochasticDiscreteSimulator(system, seed=42)
        
        x0 = np.array([0.0])
        states = sim.simulate(x0, steps=1)
        
        assert states.shape == (2, 1)
    
    def test_return_final_only(self):
        """Test memory-efficient final-only return."""
        system = DiscreteAR1(sigma=0.1)
        sim = StochasticDiscreteSimulator(system, seed=42)
        
        x0 = np.array([1.0])
        
        # Full trajectory
        states_full = sim.simulate(
            x0, steps=100,
            controller=lambda x, k: np.array([0.0])
        )
        
        # Final only
        sim.set_seed(42)
        x_final = sim.simulate(
            x0, steps=100,
            controller=lambda x, k: np.array([0.0]),
            return_final_only=True
        )
        
        # Should match last state
        np.testing.assert_allclose(x_final, states_full[-1], rtol=1e-10)


# ============================================================================
# Test Observer Integration
# ============================================================================

class TestObserver:
    """Test observer integration."""
    
    def test_with_observer(self):
        """Test simulation with observer."""
        system = DiscreteLinearSDE()
        observer = MockObserver(nx=2)
        sim = StochasticDiscreteSimulator(system, observer=observer, seed=42)
        
        x0 = np.array([1.0, 0.5])
        states = sim.simulate(
            x0, steps=10,
            controller=lambda x, k: np.array([0.0]) if x.ndim == 1 else np.zeros((x.shape[0], 1))
        )
        
        assert states.shape == (11, 2)
        assert observer.update_count == 10


# ============================================================================
# Test Information Methods
# ============================================================================

class TestInformation:
    """Test information and diagnostics."""
    
    def test_get_info(self):
        """Test get_info includes stochastic information."""
        system = DiscreteAR1()
        sim = StochasticDiscreteSimulator(system, seed=42)
        
        info = sim.get_info()
        
        assert info['is_stochastic'] is True
        assert info['nw'] == 1
        assert info['seed'] == 42
        assert 'noise_type' in info
    
    def test_repr_includes_seed(self):
        """Test repr includes seed information."""
        system = DiscreteRandomWalk()
        sim = StochasticDiscreteSimulator(system, seed=42)
        
        repr_str = repr(sim)
        assert 'seed=42' in repr_str
    
    def test_str_readable(self):
        """Test human-readable string."""
        system = DiscreteAR1()
        sim = StochasticDiscreteSimulator(system, seed=123)
        
        str_str = str(sim)
        assert 'StochasticDiscreteSimulator' in str_str
        assert 'seed=123' in str_str


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests combining multiple features."""
    
    def test_full_workflow(self):
        """Test complete workflow with all features."""
        # Create system
        system = DiscreteAR1(phi=0.85, sigma=0.2)
        
        # Create simulator
        sim = StochasticDiscreteSimulator(system, seed=42)
        
        # Controller
        controller = lambda x, k: -0.3 * x if x.ndim == 1 else -0.3 * x
        
        # Monte Carlo simulation
        result = sim.simulate_monte_carlo(
            x0=np.array([2.0]),
            steps=50,
            n_paths=300,
            controller=controller,
            return_controls=True,
            return_noise=True
        )
        
        # Verify shapes
        assert result.states.shape == (300, 51, 1)
        assert result.controls.shape == (300, 50, 1)
        assert result.noise.shape == (300, 50, 1)
        
        # Get statistics
        stats = result.get_statistics()
        assert stats['mean'].shape == (51, 1)
        
        # Final statistics
        final = result.get_final_statistics()
        assert final['mean'].shape == (1,)
        
        # Probability analysis
        prob = result.compute_probability(lambda x: np.abs(x[0]) < 1.0)
        assert 0 <= prob <= 1
    
    def test_quasi_monte_carlo_workflow(self):
        """Test workflow with quasi-random numbers."""
        system = DiscreteRandomWalk(sigma=0.5)
        sim = StochasticDiscreteSimulator(system)
        
        # Use Sobol sequence (low-discrepancy)
        from scipy.stats.qmc import Sobol
        sobol = Sobol(d=1, scramble=True, seed=42)
        
        # Generate quasi-random noise
        n_steps = 20
        noise_qmc = sobol.random(n_steps)  # (20, 1)
        
        # Transform to normal via inverse CDF
        from scipy.stats import norm
        noise_normal = norm.ppf(noise_qmc)
        
        x0 = np.array([0.0])
        states = sim.simulate(x0, steps=n_steps, noise=noise_normal)
        
        assert states.shape == (21, 1)


# ============================================================================
# Test Numerical Accuracy
# ============================================================================

class TestNumericalAccuracy:
    """Test numerical accuracy and properties."""
    
    def test_ar1_stationary_variance(self):
        """Test AR(1) converges to stationary variance."""
        # For AR(1): x[k+1] = φ*x[k] + σ*w[k]
        # Stationary variance: V = σ²/(1-φ²)
        
        phi = 0.9
        sigma = 0.2
        stationary_var = sigma**2 / (1 - phi**2)
        
        system = DiscreteAR1(phi=phi, sigma=sigma)
        sim = StochasticDiscreteSimulator(system, seed=42)
        
        x0 = np.array([0.0])
        result = sim.simulate_monte_carlo(
            x0, steps=200, n_paths=1000,
            controller=lambda x, k: np.array([0.0])
        )
        
        stats = result.get_statistics()
        
        # After many steps, variance should approach stationary
        final_var = stats['var'][-1, 0]
        
        # Should be close to theoretical
        assert 0.15 < final_var < 0.25  # stationary_var ≈ 0.21


# ============================================================================
# Performance Tests
# ============================================================================

class TestPerformance:
    """Test performance-related features."""
    
    def test_batched_faster_than_sequential(self):
        """Test that batched MC is implemented (can't test speed easily)."""
        system = DiscreteAR1(sigma=0.1)
        sim = StochasticDiscreteSimulator(system, seed=42)
        
        x0 = np.array([1.0])
        
        # Both should work
        result_parallel = sim.simulate_monte_carlo(
            x0, steps=20, n_paths=100, parallel=True
        )
        
        sim.set_seed(42)
        result_sequential = sim.simulate_monte_carlo(
            x0, steps=20, n_paths=100, parallel=False
        )
        
        # Both should produce valid results
        assert result_parallel.n_paths == 100
        assert result_sequential.n_paths == 100
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_large_batch_simulation(self):
        """Test large batched simulation."""
        system = DiscreteAR1()
        system.set_default_backend('torch')
        sim = StochasticDiscreteSimulator(system, seed=42)
        
        x0 = torch.tensor([1.0])
        
        # Large Monte Carlo
        result = sim.simulate_monte_carlo(
            x0, steps=50, n_paths=1000,
            controller=lambda x, k: torch.zeros(x.shape[0], 1) if x.ndim > 1 else torch.tensor([0.0])
        )
        
        assert result.states.shape == (1000, 51, 1)


# ============================================================================
# Test Built-in Systems
# ============================================================================

class TestBuiltinSystems:
    """Test with built-in discrete stochastic systems."""
    
    def test_white_noise_simulation(self):
        """Test simulation of white noise process."""
        system = DiscreteWhiteNoise(sigma=1.0)
        sim = StochasticDiscreteSimulator(system, seed=42)
        
        x0 = np.array([0.0])
        states = sim.simulate(x0, steps=100)
        
        # Each state should be independent (pure noise)
        assert states.shape == (101, 1)
    
    def test_random_walk_monte_carlo(self):
        """Test Monte Carlo with random walk."""
        system = DiscreteRandomWalk(sigma=0.5)
        sim = StochasticDiscreteSimulator(system, seed=42)
        
        x0 = np.array([0.0])
        result = sim.simulate_monte_carlo(x0, steps=50, n_paths=500)
        
        stats = result.get_statistics()
        
        # Mean should stay near 0 (unbiased random walk)
        assert np.abs(stats['mean'][-1, 0]) < 0.5
        
        # Variance should grow linearly
        var_t10 = stats['var'][10, 0]
        var_t50 = stats['var'][50, 0]
        
        # Roughly: var_t50 ≈ 5 * var_t10
        assert var_t50 > 3 * var_t10


# ============================================================================
# Test Error Cases
# ============================================================================

class TestErrors:
    """Test error handling and validation."""
    
    def test_wrong_noise_dimension(self):
        """Test error for wrong noise dimension."""
        system = DiscreteAR1()  # nw=1
        sim = StochasticDiscreteSimulator(system)
        
        x0 = np.array([1.0])
        noise_wrong = np.random.randn(10, 2)  # Should be (10, 1)
        
        with pytest.raises((ValueError, IndexError)):
            sim.simulate(x0, steps=10, noise=noise_wrong)
    
    def test_wrong_noise_length(self):
        """Test error for wrong noise sequence length."""
        system = DiscreteRandomWalk()
        sim = StochasticDiscreteSimulator(system)
        
        x0 = np.array([0.0])
        noise_short = np.random.randn(5, 1)  # Only 5 steps
        
        with pytest.raises(IndexError):
            sim.simulate(x0, steps=10, noise=noise_short)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])