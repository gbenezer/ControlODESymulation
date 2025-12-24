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
Comprehensive unit tests for StochasticDiscreteLinearization

Tests cover:
- Initialization and validation
- Basic stochastic linearization (drift + diffusion)
- Caching behavior for stochastic systems
- Equilibrium-based linearization
- Process noise covariance computation
- Mean-square stability analysis
- Steady-state covariance computation
- Multiple discretization methods for SDEs
- Batch equilibria linearization
- Gain scheduling support
- Cache management and statistics
- Backend compatibility
- Edge cases and error handling
- Integration with parent DiscreteLinearization
"""

import pytest
import numpy as np
import sympy as sp
import warnings

# Conditional imports
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

from src.systems.base.discretization.stochastic.stochastic_discrete_linearization import StochasticDiscreteLinearization
from src.systems.base.discrete_stochastic_system import DiscreteStochasticSystem
from src.systems.base.stochastic_dynamical_system import StochasticDynamicalSystem
from src.systems.base.discretization.stochastic.stochastic_discretizer import StochasticDiscretizer


# ============================================================================
# Test Fixtures - Stochastic Systems
# ============================================================================

class DiscreteLinearStochasticSystem(DiscreteStochasticSystem):
    """Simple 2D discrete linear stochastic system."""
    
    def define_system(self, a11=0.9, a12=0.1, a21=-0.1, a22=0.8, b1=0.0, b2=1.0, sigma1=0.1, sigma2=0.05):
        x1, x2 = sp.symbols('x1 x2', real=True)
        u = sp.symbols('u', real=True)
        
        a11_sym, a12_sym = sp.symbols('a11 a12', real=True)
        a21_sym, a22_sym = sp.symbols('a21 a22', real=True)
        b1_sym, b2_sym = sp.symbols('b1 b2', real=True)
        sigma1_sym, sigma2_sym = sp.symbols('sigma1 sigma2', positive=True)
        
        self.state_vars = [x1, x2]
        self.control_vars = [u]
        
        # Drift: x[k+1] = A*x[k] + B*u[k] + G*w[k]
        self._f_sym = sp.Matrix([
            a11_sym*x1 + a12_sym*x2 + b1_sym*u,
            a21_sym*x1 + a22_sym*x2 + b2_sym*u
        ])
        
        # Diffusion: diagonal noise
        self.diffusion_expr = sp.Matrix([
            [sigma1_sym, 0],
            [0, sigma2_sym]
        ])
        
        self.parameters = {
            a11_sym: a11, a12_sym: a12,
            a21_sym: a21, a22_sym: a22,
            b1_sym: b1, b2_sym: b2,
            sigma1_sym: sigma1, sigma2_sym: sigma2
        }
        self.order = 1
        self.sde_type = 'ito'


class DiscreteNonlinearStochasticSystem(DiscreteStochasticSystem):
    """Nonlinear discrete stochastic system."""
    
    def define_system(self, k=1.0, sigma=0.1):
        x = sp.symbols('x', real=True)
        u = sp.symbols('u', real=True)
        k_sym = sp.symbols('k', positive=True)
        sigma_sym = sp.symbols('sigma', positive=True)
        
        self.state_vars = [x]
        self.control_vars = [u]
        
        # Drift: x[k+1] = x[k] - k*sin(x[k]) + u[k]
        self._f_sym = sp.Matrix([x - k_sym * sp.sin(x) + u])
        
        # State-dependent diffusion
        self.diffusion_expr = sp.Matrix([[sigma_sym * sp.cos(x)]])
        
        self.parameters = {k_sym: k, sigma_sym: sigma}
        self.order = 1
        self.sde_type = 'ito'


class SimpleOrnsteinUhlenbeck(StochasticDynamicalSystem):
    """Simple 1D Ornstein-Uhlenbeck process (continuous SDE)."""
    
    def define_system(self, theta=1.0, sigma=0.1):
        x = sp.symbols('x', real=True)
        u = sp.symbols('u', real=True)
        theta_sym = sp.symbols('theta', positive=True)
        sigma_sym = sp.symbols('sigma', positive=True)
        
        self.state_vars = [x]
        self.control_vars = [u]
        
        # Drift: dx = -theta*x*dt + u*dt
        self._f_sym = sp.Matrix([-theta_sym * x + u])
        
        # Diffusion: constant
        self.diffusion_expr = sp.Matrix([[sigma_sym]])
        
        self.parameters = {theta_sym: theta, sigma_sym: sigma}
        self.order = 1
        self.sde_type = 'ito'


# ============================================================================
# Test Initialization
# ============================================================================

class TestInitialization:
    """Test stochastic linearization cache initialization."""
    
    def test_init_discrete_stochastic(self):
        """Test initialization with discrete stochastic system."""
        system = DiscreteLinearStochasticSystem()
        lin = StochasticDiscreteLinearization(system)
        
        assert lin.system is system
        assert lin.discretizer is None
        assert lin.backend == 'numpy'
        assert len(lin._cache) == 0
    
    def test_init_discretized_sde(self):
        """Test initialization with discretized continuous SDE."""
        system = SimpleOrnsteinUhlenbeck()
        # Let StochasticDiscretizer auto-select appropriate method for backend
        discretizer = StochasticDiscretizer(system, dt=0.01)
        lin = StochasticDiscreteLinearization(system, discretizer=discretizer)
        
        assert lin.system is system
        assert lin.discretizer is discretizer
    
    def test_init_continuous_sde_without_discretizer_fails(self):
        """Test that continuous SDE without discretizer fails."""
        system = SimpleOrnsteinUhlenbeck()
        
        with pytest.raises(TypeError, match="requires a Discretizer"):
            StochasticDiscreteLinearization(system)
    
    def test_init_deterministic_system_fails(self):
        """Test that deterministic system raises ValueError."""
        from tests.discretization_unit_tests.discrete_linearization_test import DiscreteLinearSystem
        
        system = DiscreteLinearSystem()
        
        with pytest.raises(ValueError, match="requires a stochastic system"):
            StochasticDiscreteLinearization(system)
    
    def test_init_validates_stochastic_discrete(self):
        """Test validation accepts DiscreteStochasticSystem."""
        system = DiscreteLinearStochasticSystem()
        lin = StochasticDiscreteLinearization(system)
        
        assert lin.system is system
    
    def test_init_validates_stochastic_continuous(self):
        """Test validation accepts StochasticDynamicalSystem."""
        system = SimpleOrnsteinUhlenbeck()
        discretizer = StochasticDiscretizer(system, dt=0.01)
        lin = StochasticDiscreteLinearization(system, discretizer=discretizer)
        
        assert lin.system is system
    
    def test_init_rejects_wrong_discretizer_type(self):
        """Test that regular Discretizer is rejected for stochastic systems."""
        from src.systems.base.discretization.discretizer import Discretizer
        
        system = SimpleOrnsteinUhlenbeck()
        # Try to use regular Discretizer (wrong type for SDE)
        discretizer = Discretizer(system, dt=0.01, method='euler')
        
        with pytest.raises(TypeError, match="discretizer must be StochasticDiscretizer"):
            StochasticDiscreteLinearization(system, discretizer=discretizer)


# ============================================================================
# Test Basic Stochastic Linearization
# ============================================================================

class TestBasicStochasticLinearization:
    """Test basic stochastic linearization computation."""
    
    def test_linearize_discrete_stochastic_linear(self):
        """Test linearization of linear discrete stochastic system."""
        system = DiscreteLinearStochasticSystem(
            a11=0.9, a12=0.1, a21=-0.1, a22=0.8,
            b1=0.0, b2=1.0, sigma1=0.1, sigma2=0.05
        )
        lin = StochasticDiscreteLinearization(system)
        
        x_eq = np.array([0.0, 0.0])
        u_eq = np.array([0.0])
        
        Ad, Bd, Gd = lin.compute(x_eq, u_eq)
        
        # For linear system, A, B, G are constant
        expected_A = np.array([[0.9, 0.1], [-0.1, 0.8]])
        expected_B = np.array([[0.0], [1.0]])
        expected_G = np.array([[0.1, 0.0], [0.0, 0.05]])
        
        np.testing.assert_allclose(Ad, expected_A, rtol=1e-10)
        np.testing.assert_allclose(Bd, expected_B, rtol=1e-10)
        np.testing.assert_allclose(Gd, expected_G, rtol=1e-10)
    
    def test_linearize_discrete_stochastic_nonlinear(self):
        """Test linearization of nonlinear discrete stochastic system."""
        system = DiscreteNonlinearStochasticSystem(k=1.0, sigma=0.1)
        lin = StochasticDiscreteLinearization(system)
        
        # At x=0: sin(0)=0, cos(0)=1
        x_eq = np.array([0.0])
        u_eq = np.array([0.0])
        
        Ad, Bd, Gd = lin.compute(x_eq, u_eq)
        
        # A = ∂(x - sin(x) + u)/∂x = 1 - cos(x) = 1 - 1 = 0
        # B = ∂(x - sin(x) + u)/∂u = 1
        # G = sigma*cos(x) = 0.1*1 = 0.1
        np.testing.assert_allclose(Ad, np.array([[0.0]]), atol=1e-10)
        np.testing.assert_allclose(Bd, np.array([[1.0]]), rtol=1e-10)
        np.testing.assert_allclose(Gd, np.array([[0.1]]), rtol=1e-10)
    
    def test_linearize_at_nonzero_point(self):
        """Test linearization at non-equilibrium point."""
        system = DiscreteNonlinearStochasticSystem(k=1.0, sigma=0.1)
        lin = StochasticDiscreteLinearization(system)
        
        # At x=π/2: cos(π/2)=0
        x_eq = np.array([np.pi/2])
        u_eq = np.array([0.0])
        
        Ad, Bd, Gd = lin.compute(x_eq, u_eq)
        
        # A = 1 - cos(π/2) = 1 - 0 = 1
        # G = 0.1*cos(π/2) = 0.1*0 = 0
        np.testing.assert_allclose(Ad, np.array([[1.0]]), rtol=1e-10)
        np.testing.assert_allclose(Gd, np.array([[0.0]]), atol=1e-10)
    
    def test_linearize_discretized_sde(self):
        """Test linearization of discretized continuous SDE."""
        system = SimpleOrnsteinUhlenbeck(theta=1.0, sigma=0.1)
        discretizer = StochasticDiscretizer(system, dt=0.01)
        lin = StochasticDiscreteLinearization(system, discretizer=discretizer)
        
        x_eq = np.array([0.0])
        u_eq = np.array([0.0])
        
        Ad, Bd, Gd = lin.compute(x_eq, u_eq, method='euler')
        
        # Euler: Ad = 1 + dt*(-theta) = 1 - 0.01 = 0.99
        # Bd = dt*1 = 0.01
        # Gd = sqrt(dt)*sigma = sqrt(0.01)*0.1 = 0.01
        expected_A = np.array([[0.99]])
        expected_B = np.array([[0.01]])
        expected_G = np.array([[0.01]])
        
        np.testing.assert_allclose(Ad, expected_A, rtol=1e-10)
        np.testing.assert_allclose(Bd, expected_B, rtol=1e-10)
        np.testing.assert_allclose(Gd, expected_G, rtol=1e-10)
    
    def test_returns_three_tuple(self):
        """Test that compute returns 3-tuple (Ad, Bd, Gd)."""
        system = DiscreteLinearStochasticSystem()
        lin = StochasticDiscreteLinearization(system)
        
        result = lin.compute(np.zeros(2), np.zeros(1))
        
        assert isinstance(result, tuple)
        assert len(result) == 3
        
        Ad, Bd, Gd = result
        assert Ad.shape == (2, 2)
        assert Bd.shape == (2, 1)
        assert Gd.shape == (2, 2)


# ============================================================================
# Test Caching Behavior
# ============================================================================

class TestCaching:
    """Test cache functionality for stochastic systems."""
    
    def test_cache_hit(self):
        """Test that second call uses cache."""
        system = DiscreteLinearStochasticSystem()
        lin = StochasticDiscreteLinearization(system)
        
        x_eq = np.array([0.0, 0.0])
        u_eq = np.array([0.0])
        
        # First call
        Ad1, Bd1, Gd1 = lin.compute(x_eq, u_eq)
        computes_1 = lin._stats['computes']
        
        # Second call
        Ad2, Bd2, Gd2 = lin.compute(x_eq, u_eq)
        computes_2 = lin._stats['computes']
        
        # Should be same objects (cached)
        assert Ad1 is Ad2
        assert Bd1 is Bd2
        assert Gd1 is Gd2
        
        # Stats should show cache hit
        assert computes_2 == computes_1  # No new computation
        assert lin._stats['cache_hits'] == 1
    
    def test_cache_different_points(self):
        """Test that different points are cached separately."""
        system = DiscreteNonlinearStochasticSystem()
        lin = StochasticDiscreteLinearization(system)
        
        x_eq1 = np.array([0.0])
        x_eq2 = np.array([1.0])
        u_eq = np.array([0.0])
        
        Ad1, Bd1, Gd1 = lin.compute(x_eq1, u_eq)
        Ad2, Bd2, Gd2 = lin.compute(x_eq2, u_eq)
        
        # Should be different (nonlinear system, different points)
        assert not np.allclose(Ad1, Ad2, rtol=1e-6)
        assert not np.allclose(Gd1, Gd2, rtol=1e-6)
        
        # Both should be cached (2 different keys)
        assert len(lin._cache) == 2
    
    def test_cache_stores_three_tuple(self):
        """Test that cache stores (Ad, Bd, Gd) tuples."""
        system = DiscreteLinearStochasticSystem()
        lin = StochasticDiscreteLinearization(system)
        
        x_eq = np.array([0.0, 0.0])
        u_eq = np.array([0.0])
        
        # Compute and cache
        Ad, Bd, Gd = lin.compute(x_eq, u_eq)
        
        # Get from cache
        cached = lin.get_cached(x_eq, u_eq)
        
        assert cached is not None
        assert len(cached) == 3
        
        Ad_cached, Bd_cached, Gd_cached = cached
        assert Ad is Ad_cached
        assert Bd is Bd_cached
        assert Gd is Gd_cached
    
    def test_use_cache_false_recomputes(self):
        """Test that use_cache=False forces recomputation."""
        system = DiscreteLinearStochasticSystem()
        lin = StochasticDiscreteLinearization(system)
        
        x_eq = np.array([0.0, 0.0])
        u_eq = np.array([0.0])
        
        # First call
        Ad1, Bd1, Gd1 = lin.compute(x_eq, u_eq)
        computes_1 = lin._stats['computes']
        
        # Force recompute
        Ad2, Bd2, Gd2 = lin.compute(x_eq, u_eq, use_cache=False)
        computes_2 = lin._stats['computes']
        
        # Should have computed again
        assert computes_2 == computes_1 + 1


# ============================================================================
# Test Equilibrium-Based Linearization
# ============================================================================

class TestEquilibriumLinearization:
    """Test linearization using equilibrium names."""
    
    def test_linearize_at_named_equilibrium(self):
        """Test linearization using equilibrium name."""
        system = DiscreteLinearStochasticSystem()
        system.add_equilibrium('origin', np.array([0.0, 0.0]), np.array([0.0]))
        
        lin = StochasticDiscreteLinearization(system)
        
        # Use equilibrium name
        Ad, Bd, Gd = lin.compute('origin')
        
        assert Ad.shape == (2, 2)
        assert Bd.shape == (2, 1)
        assert Gd.shape == (2, 2)
    
    def test_compute_at_equilibria(self):
        """Test batch linearization at all equilibria."""
        system = DiscreteLinearStochasticSystem()
        system.add_equilibrium('eq1', np.array([0.0, 0.0]), np.array([0.0]))
        system.add_equilibrium('eq2', np.array([1.0, 0.0]), np.array([0.0]))
        
        lin = StochasticDiscreteLinearization(system)
        
        # Linearize at all equilibria
        linearizations = lin.compute_at_equilibria()
        
        assert len(linearizations) >= 2
        assert 'eq1' in linearizations
        assert 'eq2' in linearizations
        
        # Each should be (Ad, Bd, Gd) tuple
        for name, (Ad, Bd, Gd) in linearizations.items():
            assert Ad.shape == (2, 2)
            assert Bd.shape == (2, 1)
            assert Gd.shape == (2, 2)
    
    def test_compute_at_specific_equilibria(self):
        """Test linearization at subset of equilibria."""
        system = DiscreteLinearStochasticSystem()
        system.add_equilibrium('eq1', np.array([0.0, 0.0]), np.array([0.0]))
        system.add_equilibrium('eq2', np.array([1.0, 0.0]), np.array([0.0]))
        system.add_equilibrium('eq3', np.array([0.0, 1.0]), np.array([0.0]))
        
        lin = StochasticDiscreteLinearization(system)
        
        # Linearize at specific subset
        linearizations = lin.compute_at_equilibria(['eq1', 'eq2'])
        
        assert len(linearizations) == 2
        assert 'eq1' in linearizations
        assert 'eq2' in linearizations
        assert 'eq3' not in linearizations


# ============================================================================
# Test Process Noise Covariance
# ============================================================================

class TestProcessNoiseCovariance:
    """Test process noise covariance computation."""
    
    def test_compute_process_noise_white_noise(self):
        """Test process noise covariance with white noise."""
        system = DiscreteLinearStochasticSystem(sigma1=0.1, sigma2=0.05)
        lin = StochasticDiscreteLinearization(system)
        
        x_eq = np.array([0.0, 0.0])
        u_eq = np.array([0.0])
        
        Q = lin.compute_process_noise_covariance(x_eq, u_eq)
        
        # Q = Gd @ Gd.T
        # Gd = diag([0.1, 0.05])
        # Q = diag([0.01, 0.0025])
        expected_Q = np.array([[0.01, 0.0], [0.0, 0.0025]])
        
        np.testing.assert_allclose(Q, expected_Q, rtol=1e-10)
    
    def test_compute_process_noise_colored_noise(self):
        """Test process noise covariance with colored noise."""
        system = DiscreteLinearStochasticSystem(sigma1=0.1, sigma2=0.05)
        lin = StochasticDiscreteLinearization(system)
        
        x_eq = np.array([0.0, 0.0])
        u_eq = np.array([0.0])
        
        # Colored noise covariance
        Qw = np.array([[1.0, 0.5], [0.5, 2.0]])
        
        Q = lin.compute_process_noise_covariance(x_eq, u_eq, noise_covariance=Qw)
        
        # Q = Gd @ Qw @ Gd.T
        Gd = np.array([[0.1, 0.0], [0.0, 0.05]])
        expected_Q = Gd @ Qw @ Gd.T
        
        np.testing.assert_allclose(Q, expected_Q, rtol=1e-10)
    
    def test_process_noise_covariance_is_symmetric(self):
        """Test that process noise covariance is symmetric."""
        system = DiscreteLinearStochasticSystem()
        lin = StochasticDiscreteLinearization(system)
        
        Q = lin.compute_process_noise_covariance(np.zeros(2), np.zeros(1))
        
        np.testing.assert_allclose(Q, Q.T, rtol=1e-10)
    
    def test_process_noise_covariance_is_positive_semidefinite(self):
        """Test that process noise covariance is positive semidefinite."""
        system = DiscreteLinearStochasticSystem()
        lin = StochasticDiscreteLinearization(system)
        
        Q = lin.compute_process_noise_covariance(np.zeros(2), np.zeros(1))
        
        # Check eigenvalues are non-negative
        eigenvalues = np.linalg.eigvals(Q)
        assert np.all(eigenvalues >= -1e-10)


# ============================================================================
# Test Mean-Square Stability
# ============================================================================

class TestMeanSquareStability:
    """Test mean-square stability analysis."""
    
    def test_check_ms_stability_stable_system(self):
        """Test stability check for mean-square stable system."""
        system = DiscreteLinearStochasticSystem(a11=0.8, a12=0.0, a21=0.0, a22=0.7)
        lin = StochasticDiscreteLinearization(system)
        
        x_eq = np.array([0.0, 0.0])
        u_eq = np.array([0.0])
        
        stability = lin.check_mean_square_stability(x_eq, u_eq)
        
        assert stability['is_ms_stable'] is True
        assert stability['is_unstable'] is False
        assert stability['max_magnitude'] < 1.0
        assert len(stability['eigenvalues']) == 2
    
    def test_check_ms_stability_unstable_system(self):
        """Test stability check for unstable system."""
        system = DiscreteLinearStochasticSystem(a11=1.1, a12=0.0, a21=0.0, a22=0.9)
        lin = StochasticDiscreteLinearization(system)
        
        x_eq = np.array([0.0, 0.0])
        u_eq = np.array([0.0])
        
        stability = lin.check_mean_square_stability(x_eq, u_eq)
        
        assert stability['is_ms_stable'] is False
        assert stability['is_unstable'] is True
        assert stability['max_magnitude'] > 1.0
    
    def test_steady_state_covariance_computed_when_stable(self):
        """Test steady-state covariance is computed for stable systems."""
        system = DiscreteLinearStochasticSystem(a11=0.8, a12=0.0, a21=0.0, a22=0.7)
        lin = StochasticDiscreteLinearization(system)
        
        stability = lin.check_mean_square_stability(np.zeros(2), np.zeros(1))
        
        assert stability['is_ms_stable'] is True
        assert stability['steady_state_covariance'] is not None
        
        P_ss = stability['steady_state_covariance']
        assert P_ss.shape == (2, 2)
        
        # Should be symmetric
        np.testing.assert_allclose(P_ss, P_ss.T, rtol=1e-10)
        
        # Should be positive semidefinite
        eigenvalues = np.linalg.eigvals(P_ss)
        assert np.all(eigenvalues >= -1e-10)
    
    def test_steady_state_covariance_none_when_unstable(self):
        """Test steady-state covariance is None for unstable systems."""
        system = DiscreteLinearStochasticSystem(a11=1.1, a12=0.0, a21=0.0, a22=0.9)
        lin = StochasticDiscreteLinearization(system)
        
        stability = lin.check_mean_square_stability(np.zeros(2), np.zeros(1))
        
        assert stability['is_ms_stable'] is False
        assert stability['steady_state_covariance'] is None
    
    def test_steady_state_covariance_satisfies_lyapunov(self):
        """Test that steady-state covariance satisfies Lyapunov equation."""
        system = DiscreteLinearStochasticSystem(a11=0.9, a12=0.1, a21=-0.1, a22=0.8)
        lin = StochasticDiscreteLinearization(system)
        
        x_eq = np.zeros(2)
        u_eq = np.zeros(1)
        
        stability = lin.check_mean_square_stability(x_eq, u_eq)
        
        if stability['is_ms_stable']:
            P_ss = stability['steady_state_covariance']
            Ad, Bd, Gd = lin.compute(x_eq, u_eq)
            Q = lin.compute_process_noise_covariance(x_eq, u_eq)
            
            # Check P = Ad*P*Ad' + Q
            lhs = P_ss
            rhs = Ad @ P_ss @ Ad.T + Q
            
            np.testing.assert_allclose(lhs, rhs, rtol=1e-6)


# ============================================================================
# Test Discretization Methods for SDEs
# ============================================================================

class TestSDEDiscretizationMethods:
    """Test different discretization methods for SDEs."""
    
    def test_euler_maruyama_method(self):
        """Test Euler-Maruyama discretization."""
        system = SimpleOrnsteinUhlenbeck(theta=1.0, sigma=0.1)
        discretizer = StochasticDiscretizer(system, dt=0.01)
        lin = StochasticDiscreteLinearization(system, discretizer=discretizer)
        
        x_eq = np.array([0.0])
        u_eq = np.array([0.0])
        
        Ad, Bd, Gd = lin.compute(x_eq, u_eq, method='euler')
        
        # Euler-Maruyama: Ad = 1 + dt*(-theta), Gd = sqrt(dt)*sigma
        assert Ad.shape == (1, 1)
        assert Gd.shape == (1, 1)
        
        # Check Gd has sqrt(dt) scaling
        expected_Gd = np.sqrt(0.01) * 0.1
        np.testing.assert_allclose(Gd, [[expected_Gd]], rtol=1e-10)
    
    def test_different_methods_warn_for_diffusion(self):
        """Test that non-Euler methods work with StochasticDiscretizer."""
        system = SimpleOrnsteinUhlenbeck()
        # StochasticDiscretizer handles diffusion correctly for all methods
        discretizer = StochasticDiscretizer(system, dt=0.01)
        lin = StochasticDiscreteLinearization(system, discretizer=discretizer)
        
        # Request 'exact' method for linearization
        # StochasticDiscretizer handles this without warnings
        Ad, Bd, Gd = lin.compute(np.array([0.0]), np.array([0.0]), method='exact')
        
        # Should work without warnings - StochasticDiscretizer handles it
        assert Ad.shape == (1, 1)
        assert Gd.shape == (1, 1)


# ============================================================================
# Test Cache Management
# ============================================================================

class TestCacheManagement:
    """Test cache management features."""
    
    def test_reset_cache(self):
        """Test clearing entire cache."""
        system = DiscreteLinearStochasticSystem()
        lin = StochasticDiscreteLinearization(system)
        
        # Cache some linearizations
        lin.compute(np.zeros(2), np.zeros(1))
        lin.compute(np.ones(2), np.zeros(1))
        
        assert len(lin._cache) == 2
        
        # Reset
        lin.reset_cache()
        
        assert len(lin._cache) == 0
    
    def test_list_cached(self):
        """Test listing cached items."""
        system = DiscreteLinearStochasticSystem()
        system.add_equilibrium('origin', np.zeros(2), np.zeros(1))
        
        lin = StochasticDiscreteLinearization(system)
        
        # Initially empty
        assert len(lin.list_cached()) == 0
        
        # Cache some
        lin.compute('origin')
        lin.compute(np.ones(2), np.zeros(1))
        
        cached = lin.list_cached()
        assert len(cached) == 2
        assert 'origin' in cached
    
    def test_clear_cache(self):
        """Test clear_cache clears cache and stats."""
        system = DiscreteLinearStochasticSystem()
        lin = StochasticDiscreteLinearization(system)
        
        # Generate some activity
        lin.compute(np.zeros(2), np.zeros(1))
        lin.compute(np.zeros(2), np.zeros(1))  # Cache hit
        
        assert lin._stats['computes'] > 0
        assert lin._stats['cache_hits'] > 0
        
        # Clear
        lin.clear_cache()
        
        assert len(lin._cache) == 0
        assert lin._stats['computes'] == 0
        assert lin._stats['cache_hits'] == 0


# ============================================================================
# Test Statistics
# ============================================================================

class TestStatistics:
    """Test cache statistics."""
    
    def test_get_stats(self):
        """Test statistics collection."""
        system = DiscreteLinearStochasticSystem()
        lin = StochasticDiscreteLinearization(system)
        
        x_eq = np.zeros(2)
        u_eq = np.zeros(1)
        
        # Initial stats
        stats = lin.get_stats()
        assert stats['computes'] == 0
        assert stats['cache_hits'] == 0
        assert stats['hit_rate'] == 0.0
        
        # First call (compute)
        lin.compute(x_eq, u_eq)
        stats = lin.get_stats()
        assert stats['computes'] == 1
        
        # Second call (cache hit)
        lin.compute(x_eq, u_eq)
        stats = lin.get_stats()
        assert stats['cache_hits'] == 1
        assert stats['hit_rate'] == 0.5


# ============================================================================
# Test Backend Compatibility
# ============================================================================

class TestBackendCompatibility:
    """Test multi-backend support."""
    
    def test_numpy_backend(self):
        """Test NumPy backend."""
        system = DiscreteLinearStochasticSystem()
        lin = StochasticDiscreteLinearization(system)
        
        Ad, Bd, Gd = lin.compute(np.zeros(2), np.zeros(1))
        
        assert isinstance(Ad, np.ndarray)
        assert isinstance(Bd, np.ndarray)
        assert isinstance(Gd, np.ndarray)
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_torch_backend(self):
        """Test PyTorch backend."""
        system = DiscreteLinearStochasticSystem()
        system.set_default_backend('torch')
        lin = StochasticDiscreteLinearization(system)
        
        x_eq = torch.zeros(2)
        u_eq = torch.zeros(1)
        
        Ad, Bd, Gd = lin.compute(x_eq, u_eq)
        
        assert isinstance(Ad, torch.Tensor)
        assert isinstance(Bd, torch.Tensor)
        assert isinstance(Gd, torch.Tensor)
    
    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_jax_backend(self):
        """Test JAX backend."""
        system = DiscreteLinearStochasticSystem()
        system.set_default_backend('jax')
        lin = StochasticDiscreteLinearization(system)
        
        x_eq = jnp.zeros(2)
        u_eq = jnp.zeros(1)
        
        Ad, Bd, Gd = lin.compute(x_eq, u_eq)
        
        assert isinstance(Ad, jnp.ndarray)
        assert isinstance(Bd, jnp.ndarray)
        assert isinstance(Gd, jnp.ndarray)


# ============================================================================
# Test Gain Scheduling
# ============================================================================

class TestGainScheduling:
    """Test gain scheduling support."""
    
    def test_precompute_at_grid(self):
        """Test precomputing stochastic linearizations at grid."""
        system = DiscreteNonlinearStochasticSystem()
        lin = StochasticDiscreteLinearization(system)
        
        # Create grid
        x_grid = np.linspace(-1, 1, 5).reshape(-1, 1)
        u_grid = np.zeros((5, 1))
        
        # Precompute
        lin.precompute_at_grid(x_grid, u_grid)
        
        # All should be cached
        assert lin._stats['computes'] == 5
        assert len(lin._cache) >= 5
    
    def test_gain_scheduling_workflow(self):
        """Test complete gain scheduling workflow."""
        system = DiscreteNonlinearStochasticSystem()
        lin = StochasticDiscreteLinearization(system)
        
        # Define operating points
        operating_points = [
            (np.array([0.0]), np.array([0.0])),
            (np.array([0.5]), np.array([0.0])),
            (np.array([1.0]), np.array([0.0])),
        ]
        
        # Precompute all
        for x, u in operating_points:
            lin.compute(x, u)
        
        # Verify all cached
        for x, u in operating_points:
            assert lin.is_cached(x, u)
        
        # Second access is instant
        stats_before = lin.get_stats()
        for x, u in operating_points:
            Ad, Bd, Gd = lin.compute(x, u)
        
        # All should be cache hits
        stats_after = lin.get_stats()
        assert stats_after['cache_hits'] == stats_before['cache_hits'] + 3


# ============================================================================
# Test Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_single_state_system(self):
        """Test 1D stochastic system."""
        system = DiscreteNonlinearStochasticSystem()
        lin = StochasticDiscreteLinearization(system)
        
        Ad, Bd, Gd = lin.compute(np.array([0.0]), np.array([0.0]))
        
        assert Ad.shape == (1, 1)
        assert Bd.shape == (1, 1)
        assert Gd.shape == (1, 1)
    
    def test_autonomous_stochastic_system(self):
        """Test autonomous stochastic system (nu=0)."""
        class AutonomousStochasticSystem(DiscreteStochasticSystem):
            def define_system(self, sigma=0.1):
                x = sp.symbols('x', real=True)
                sigma_sym = sp.symbols('sigma', positive=True)
                
                self.state_vars = [x]
                self.control_vars = []
                self._f_sym = sp.Matrix([0.9*x])
                self.diffusion_expr = sp.Matrix([[sigma_sym]])
                self.parameters = {sigma_sym: sigma}
                self.order = 1
                self.sde_type = 'ito'
        
        system = AutonomousStochasticSystem()
        lin = StochasticDiscreteLinearization(system)
        
        Ad, Bd, Gd = lin.compute(np.array([0.0]))
        
        assert Ad.shape == (1, 1)
        assert Bd.shape == (1, 0)  # No control
        assert Gd.shape == (1, 1)


# ============================================================================
# Test Information Methods
# ============================================================================

class TestInformation:
    """Test information and diagnostic methods."""
    
    def test_get_info(self):
        """Test get_info method."""
        system = DiscreteLinearStochasticSystem()
        lin = StochasticDiscreteLinearization(system)
        
        # Cache something
        lin.compute(np.zeros(2), np.zeros(1))
        
        info = lin.get_info()
        
        assert info['system'] == 'DiscreteLinearStochasticSystem'
        assert info['system_type'] == 'discrete'
        assert info['discretizer'] is None
        assert info['backend'] == 'numpy'
        assert info['cache_size'] == 1
        assert info['linearization_type'] == 'stochastic'
        assert 'statistics' in info
    
    def test_get_info_discretized(self):
        """Test get_info for discretized SDE."""
        system = SimpleOrnsteinUhlenbeck()
        discretizer = StochasticDiscretizer(system, dt=0.01)
        lin = StochasticDiscreteLinearization(system, discretizer=discretizer)
        
        info = lin.get_info()
        
        assert info['system_type'] == 'discretized'
        # Method will be auto-selected based on backend
        assert info['discretizer'] in ['EM', 'euler', 'Euler']
        assert info['linearization_type'] == 'stochastic'
    
    def test_repr_str(self):
        """Test string representations."""
        system = DiscreteLinearStochasticSystem()
        lin = StochasticDiscreteLinearization(system)
        
        # Before caching
        repr_str = repr(lin)
        assert 'StochasticDiscreteLinearization' in repr_str
        assert 'cache_size=0' in repr_str
        
        # After caching
        lin.compute(np.zeros(2), np.zeros(1))
        repr_str = repr(lin)
        assert 'cache_size=1' in repr_str
        
        # String representation
        str_repr = str(lin)
        assert 'discrete-stochastic' in str_repr


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for complete workflows."""
    
    def test_kalman_filter_design_workflow(self):
        """Test typical Kalman filter design workflow."""
        system = DiscreteLinearStochasticSystem()
        system.add_equilibrium('origin', np.zeros(2), np.zeros(1))
        
        lin = StochasticDiscreteLinearization(system)
        
        # Get linearization
        Ad, Bd, Gd = lin.compute('origin')
        
        # Process noise covariance
        Q = lin.compute_process_noise_covariance('origin')
        
        # Observation matrix (not cached, from system directly)
        Cd = system.linearized_observation(np.zeros(2), backend='numpy')
        
        # Verify shapes for Kalman filter
        assert Ad.shape == (2, 2)
        assert Bd.shape == (2, 1)
        assert Gd.shape == (2, 2)
        assert Q.shape == (2, 2)
        assert Cd.shape == (2, 2)  # Assuming identity observation
    
    def test_lqg_design_workflow(self):
        """Test LQG (LQR + Kalman) design workflow."""
        system = DiscreteLinearStochasticSystem(a11=0.9, a12=0.1, a21=-0.1, a22=0.8)
        lin = StochasticDiscreteLinearization(system)
        
        x_eq = np.zeros(2)
        u_eq = np.zeros(1)
        
        # Get system matrices
        Ad, Bd, Gd = lin.compute(x_eq, u_eq)
        Q = lin.compute_process_noise_covariance(x_eq, u_eq)
        
        # Check stability (needed for LQG)
        stability = lin.check_mean_square_stability(x_eq, u_eq)
        assert stability['is_ms_stable']
        
        # Would design LQR and Kalman filter here
        # K_lqr = solve_lqr(Ad, Bd, Q_cost, R_cost)
        # K_kf = design_kalman_filter(Ad, Cd, Q, R_meas)
        
        assert Ad.shape == (2, 2)
        assert Q.shape == (2, 2)
    
    def test_gain_scheduled_kalman_filtering(self):
        """Test gain-scheduled Kalman filtering workflow."""
        system = DiscreteNonlinearStochasticSystem()
        lin = StochasticDiscreteLinearization(system)
        
        # Operating points
        x_points = [-1.0, -0.5, 0.0, 0.5, 1.0]
        
        # Precompute all linearizations and process noise
        filters = {}
        for x_val in x_points:
            x_eq = np.array([x_val])
            u_eq = np.array([0.0])
            
            Ad, Bd, Gd = lin.compute(x_eq, u_eq)
            Q = lin.compute_process_noise_covariance(x_eq, u_eq)
            
            filters[x_val] = {'Ad': Ad, 'Gd': Gd, 'Q': Q}
        
        # All cached
        assert len(filters) == 5
        
        # Future accesses are instant
        for x_val in x_points:
            assert lin.is_cached(np.array([x_val]), np.array([0.0]))


# ============================================================================
# Test Numerical Accuracy
# ============================================================================

class TestNumericalAccuracy:
    """Test numerical accuracy and consistency."""
    
    def test_linear_system_constant_linearization(self):
        """Test that linear system has constant linearization."""
        system = DiscreteLinearStochasticSystem()
        lin = StochasticDiscreteLinearization(system)
        
        # Linearize at different points
        Ad1, Bd1, Gd1 = lin.compute(np.array([0.0, 0.0]), np.array([0.0]))
        Ad2, Bd2, Gd2 = lin.compute(np.array([1.0, 1.0]), np.array([0.0]))
        Ad3, Bd3, Gd3 = lin.compute(np.array([-1.0, 2.0]), np.array([0.0]))
        
        # For linear system, should all be identical
        np.testing.assert_allclose(Ad1, Ad2, rtol=1e-10)
        np.testing.assert_allclose(Ad1, Ad3, rtol=1e-10)
        np.testing.assert_allclose(Bd1, Bd2, rtol=1e-10)
        np.testing.assert_allclose(Gd1, Gd2, rtol=1e-10)
    
    def test_nonlinear_system_varying_linearization(self):
        """Test that nonlinear system has different linearizations."""
        system = DiscreteNonlinearStochasticSystem()
        lin = StochasticDiscreteLinearization(system)
        
        # Linearize at different points
        Ad1, Bd1, Gd1 = lin.compute(np.array([0.0]), np.array([0.0]))
        Ad2, Bd2, Gd2 = lin.compute(np.array([np.pi]), np.array([0.0]))
        
        # Should be different for nonlinear system
        assert not np.allclose(Ad1, Ad2, rtol=1e-6)
        assert not np.allclose(Gd1, Gd2, rtol=1e-6)
    
    def test_process_noise_scales_with_dt(self):
        """Test that process noise scales correctly with timestep."""
        system = SimpleOrnsteinUhlenbeck(sigma=0.1)
        
        dt1 = 0.01
        dt2 = 0.04
        
        discretizer1 = StochasticDiscretizer(system, dt=dt1)
        discretizer2 = StochasticDiscretizer(system, dt=dt2)
        
        lin1 = StochasticDiscreteLinearization(system, discretizer=discretizer1)
        lin2 = StochasticDiscreteLinearization(system, discretizer=discretizer2)
        
        x_eq = np.array([0.0])
        u_eq = np.array([0.0])
        
        Q1 = lin1.compute_process_noise_covariance(x_eq, u_eq, method='euler')
        Q2 = lin2.compute_process_noise_covariance(x_eq, u_eq, method='euler')
        
        # Q should scale linearly with dt for Euler-Maruyama
        # Q = Gd @ Gd.T = (sqrt(dt)*sigma)^2 = dt*sigma^2
        ratio = Q2[0, 0] / Q1[0, 0]
        expected_ratio = dt2 / dt1
        
        np.testing.assert_allclose(ratio, expected_ratio, rtol=1e-6)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])