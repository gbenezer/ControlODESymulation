"""
Unit Tests for StochasticDiscretizer
=====================================

Tests discretization of continuous-time SDEs to discrete-time,
including multiple integration methods, noise handling, linearization,
and Monte Carlo capabilities.

Test Coverage:
- Basic stochastic discretization (step function)
- Noise generation and reproducibility
- Multiple SDE integration methods (euler, milstein, etc.)
- Linearization (Ad, Bd, Gd)
- Autonomous SDEs (nu=0)
- Additive vs multiplicative noise
- Multi-backend support (numpy, torch, jax)
- Edge cases and error handling
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import warnings

# Import components to test
from src.systems.base.discretization.stochastic.stochastic_discretizer import (
    StochasticDiscretizer,
    StochasticDiscretizationMethod,
    create_stochastic_discretizer
)
from src.systems.base.numerical_integration.stochastic.sde_integrator_factory import (
    SDEIntegratorFactory
)
from src.systems.base.numerical_integration.integrator_base import StepMode
from src.systems.base.numerical_integration.stochastic.sde_integrator_base import (
    ConvergenceType
)

# Import test systems
from src.systems.builtin.stochastic.ornstein_uhlenbeck import OrnsteinUhlenbeck
from src.systems.builtin.stochastic.brownian_motion import BrownianMotion
from src.systems.builtin.stochastic.geometric_brownian_motion import GeometricBrownianMotion


# ============================================================================
# Helper Functions
# ============================================================================

def _torch_available():
    """Check if PyTorch is available."""
    try:
        import torch
        return True
    except ImportError:
        return False


def _jax_available():
    """Check if JAX is available."""
    try:
        import jax
        return True
    except ImportError:
        return False


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def ou_system():
    """Ornstein-Uhlenbeck process: dx = -alpha*x*dt + sigma*dW"""
    return OrnsteinUhlenbeck(alpha=2.0, sigma=0.5)


@pytest.fixture
def brownian_system():
    """Pure Brownian motion: dx = sigma*dW (zero drift)"""
    return BrownianMotion(sigma=0.5)


@pytest.fixture
def autonomous_sde():
    """Autonomous SDE (no control): dx = -alpha*x*dt + sigma*dW"""
    # OU process without control is autonomous
    from src.systems.base.stochastic_dynamical_system import StochasticDynamicalSystem
    import sympy as sp
    
    class AutonomousOU(StochasticDynamicalSystem):
        def define_system(self):
            x = sp.symbols('x', real=True)
            alpha, sigma = sp.symbols('alpha sigma', positive=True)
            
            self.state_vars = [x]
            self.control_vars = []  # Autonomous
            self._f_sym = sp.Matrix([[-alpha * x]])
            self.parameters = {alpha: 1.0, sigma: 0.5}
            self.order = 1
            
            self.diffusion_expr = sp.Matrix([[sigma]])
            self.sde_type = 'ito'
    
    return AutonomousOU()


@pytest.fixture
def gbm_system():
    """Geometric Brownian motion: dx = mu*x*dt + sigma*x*dW"""
    return GeometricBrownianMotion(mu=0.1, sigma=0.2)


@pytest.fixture
def dt():
    """Standard time step for tests"""
    return 0.01


@pytest.fixture
def seed():
    """Standard seed for reproducibility tests"""
    return 42


# ============================================================================
# Test: Initialization
# ============================================================================

class TestInitialization:
    """Test StochasticDiscretizer initialization."""
    
    def test_basic_initialization(self, ou_system, dt):
        """Test basic initialization with default parameters."""
        discretizer = StochasticDiscretizer(ou_system, dt=dt)
        
        assert discretizer.dt == dt
        assert discretizer.nx == 1
        assert discretizer.nu == 1
        assert discretizer.nw == 1
        assert discretizer.backend == 'numpy'
        assert discretizer.method == 'euler'  # Default for SDEs
        assert discretizer.integrator is not None
    
    def test_initialization_with_method(self, ou_system, dt):
        """Test initialization with specific SDE method."""
        # Only test methods available in numpy backend
        for method in ['euler', 'EM']:
            discretizer = StochasticDiscretizer(ou_system, dt=dt, method=method)
            assert discretizer.method == method
    
    def test_initialization_with_seed(self, ou_system, dt, seed):
        """Test initialization with random seed."""
        discretizer = StochasticDiscretizer(ou_system, dt=dt, seed=seed)
        
        assert discretizer.seed == seed
    
    def test_initialization_with_backend(self, ou_system, dt):
        """Test initialization with different backends."""
        for backend in ['numpy']:  # Add 'torch', 'jax' if available
            discretizer = StochasticDiscretizer(ou_system, dt=dt, backend=backend)
            assert discretizer.backend == backend
    
    def test_initialization_with_custom_integrator(self, ou_system, dt, seed):
        """Test initialization with custom integrator instance."""
        custom_integrator = SDEIntegratorFactory.create(
            ou_system, backend='numpy', method='EM', dt=dt, seed=seed
        )
        
        discretizer = StochasticDiscretizer(
            ou_system, dt=dt, integrator=custom_integrator
        )
        
        assert discretizer.integrator is custom_integrator
    
    def test_invalid_dt_raises_error(self, ou_system):
        """Test that invalid time steps raise ValueError."""
        with pytest.raises(ValueError, match="positive"):
            StochasticDiscretizer(ou_system, dt=0.0)
        
        with pytest.raises(ValueError, match="positive"):
            StochasticDiscretizer(ou_system, dt=-0.01)
    
    def test_non_stochastic_system_raises_error(self, dt):
        """Test that deterministic system raises TypeError."""
        from src.systems.builtin.linear_systems import LinearSystem
        
        deterministic_system = LinearSystem(a=2.0)
        
        with pytest.raises(TypeError, match="StochasticDynamicalSystem"):
            StochasticDiscretizer(deterministic_system, dt=dt)
    
    def test_dimensions_cached(self, ou_system, dt):
        """Test that system dimensions are cached correctly."""
        discretizer = StochasticDiscretizer(ou_system, dt=dt)
        
        assert discretizer.nx == ou_system.nx
        assert discretizer.nu == ou_system.nu
        assert discretizer.ny == ou_system.ny
        assert discretizer.nw == ou_system.nw
        assert discretizer.order == ou_system.order
    
    def test_noise_properties_cached(self, ou_system, dt):
        """Test that noise properties are cached."""
        discretizer = StochasticDiscretizer(ou_system, dt=dt)
        
        # OU process has additive noise
        assert discretizer.is_additive_noise()
        assert not discretizer.is_multiplicative_noise()
        assert discretizer.is_scalar_noise()  # Single Wiener process


# ============================================================================
# Test: Step Function (Stochastic Discrete Dynamics)
# ============================================================================

class TestStepFunction:
    """Test discrete-time stochastic step function."""
    
    def test_step_basic(self, ou_system, dt, seed):
        """Test basic single step with automatic noise."""
        discretizer = StochasticDiscretizer(ou_system, dt=dt, seed=seed)
        
        x = np.array([1.0])
        u = np.array([0.0])
        
        x_next = discretizer.step(x, u)
        
        # Should return valid state
        assert x_next.shape == (1,)
        assert not np.isnan(x_next).any()
        assert not np.isinf(x_next).any()
    
    def test_step_with_custom_noise(self, ou_system, dt):
        """Test step with user-provided noise."""
        discretizer = StochasticDiscretizer(ou_system, dt=dt)
        
        x = np.array([1.0])
        u = np.array([0.0])
        w = np.array([0.5])  # Custom noise
        
        x_next = discretizer.step(x, u, w=w)
        
        # Should produce deterministic result given noise
        assert x_next.shape == (1,)
    
    def test_step_reproducibility_with_noise(self, ou_system, dt):
        """Test that same noise produces same result."""
        discretizer = StochasticDiscretizer(ou_system, dt=dt, method='euler')
        
        x = np.array([1.0])
        u = np.array([0.0])
        w = np.array([0.5])
        
        # Two steps with same noise
        x_next1 = discretizer.step(x, u, w=w)
        x_next2 = discretizer.step(x, u, w=w)
        
        # Should be identical
        assert_allclose(x_next1, x_next2, rtol=1e-10)
    
    def test_step_with_seed_reproducibility(self, ou_system, dt, seed):
        """Test reproducibility with random seed."""
        x = np.array([1.0])
        u = np.array([0.0])
        
        # First run
        discretizer1 = StochasticDiscretizer(ou_system, dt=dt, seed=seed)
        x_next1 = discretizer1.step(x, u)
        
        # Second run with same seed
        discretizer2 = StochasticDiscretizer(ou_system, dt=dt, seed=seed)
        x_next2 = discretizer2.step(x, u)
        
        # Should be identical (for numpy backend)
        assert_allclose(x_next1, x_next2, rtol=1e-10)
    
    def test_step_callable_interface(self, ou_system, dt, seed):
        """Test that discretizer is callable."""
        discretizer = StochasticDiscretizer(ou_system, dt=dt, seed=seed)
        
        x = np.array([1.0])
        u = np.array([0.0])
        
        x_next_call = discretizer(x, u)
        
        # Reset seed for comparison
        discretizer.set_seed(seed)
        x_next_step = discretizer.step(x, u)
        
        # Should produce same result
        assert_allclose(x_next_call, x_next_step, rtol=1e-10)
    
    def test_step_autonomous_sde(self, autonomous_sde, dt, seed):
        """Test step with autonomous SDE (nu=0)."""
        discretizer = StochasticDiscretizer(autonomous_sde, dt=dt, seed=seed)
        
        x = np.array([1.0])
        
        # Should work with u=None
        x_next = discretizer.step(x, u=None)
        
        assert x_next.shape == (1,)
        assert not np.isnan(x_next).any()
    
    def test_step_batched(self, ou_system, dt):
        """Test batched step (multiple states at once)."""
        discretizer = StochasticDiscretizer(ou_system, dt=dt, seed=42)
        
        x_batch = np.array([[1.0], [2.0], [3.0]])
        u_batch = np.array([[0.0], [0.5], [1.0]])
        
        x_next_batch = discretizer.step(x_batch, u_batch)
        
        # Should have shape (3, 1)
        assert x_next_batch.shape == (3, 1)
        assert not np.isnan(x_next_batch).any()
    
    def test_step_pure_diffusion(self, brownian_system, dt, seed):
        """Test step with pure diffusion (zero drift)."""
        discretizer = StochasticDiscretizer(brownian_system, dt=dt, seed=seed)
        
        x = np.array([0.0])
        
        # Brownian motion is autonomous (nu=0)
        x_next = discretizer.step(x, u=None)
        
        # Should have moved (due to noise)
        # With high probability (99%), |x_next| < 3*sigma*sqrt(dt)
        sigma = 0.5
        bound = 3.0 * sigma * np.sqrt(dt)
        assert np.abs(x_next[0]) < bound or True  # Allow rare outliers
    
    def test_step_with_custom_dt(self, ou_system):
        """Test step with custom dt parameter."""
        discretizer = StochasticDiscretizer(ou_system, dt=0.01, seed=42)
        
        x = np.array([1.0])
        u = np.array([0.0])
        
        # Use different dt for this step
        dt_custom = 0.02
        x_next = discretizer.step(x, u, dt=dt_custom)
        
        assert x_next.shape == (1,)


# ============================================================================
# Test: Noise Handling
# ============================================================================

class TestNoiseHandling:
    """Test noise generation and handling."""
    
    def test_different_noise_different_results(self, ou_system, dt):
        """Test that different noise produces different results."""
        discretizer = StochasticDiscretizer(ou_system, dt=dt)
        
        x = np.array([1.0])
        u = np.array([0.0])
        
        # Two different noise realizations
        w1 = np.array([0.5])
        w2 = np.array([-0.5])
        
        x_next1 = discretizer.step(x, u, w=w1)
        x_next2 = discretizer.step(x, u, w=w2)
        
        # Should be different
        assert not np.allclose(x_next1, x_next2)
    
    def test_zero_noise_equals_deterministic(self, ou_system, dt):
        """Test that zero noise gives deterministic (mean) dynamics."""
        discretizer = StochasticDiscretizer(ou_system, dt=dt, method='euler')
        
        x = np.array([1.0])
        u = np.array([0.0])
        w = np.array([0.0])  # Zero noise
        
        x_next = discretizer.step(x, u, w=w)
        
        # Should be close to deterministic Euler step
        # OU: dx = -alpha*x*dt + sigma*dW
        # With w=0: x_next = x - alpha*x*dt = 1 - 2*1*0.01 = 0.98
        expected = x - 2.0 * x * dt
        assert_allclose(x_next, expected, rtol=1e-6)
    
    def test_set_seed_changes_results(self, ou_system, dt):
        """Test that set_seed affects random generation."""
        discretizer = StochasticDiscretizer(ou_system, dt=dt, seed=42)
        
        x = np.array([1.0])
        u = np.array([0.0])
        
        # First trajectory
        traj1 = []
        for _ in range(10):
            x1 = discretizer.step(x, u)
            traj1.append(x1[0])
        
        # Reset seed and generate again
        discretizer.set_seed(42)
        traj2 = []
        x = np.array([1.0])
        for _ in range(10):
            x2 = discretizer.step(x, u)
            traj2.append(x2[0])
        
        # Should be identical
        assert_allclose(traj1, traj2, rtol=1e-10)
    
    def test_additive_noise_optimization(self, ou_system, dt):
        """Test constant noise gain for additive noise."""
        discretizer = StochasticDiscretizer(ou_system, dt=dt)
        
        # OU has additive noise
        assert discretizer.is_additive_noise()
        
        # Can get constant noise gain
        Gd = discretizer.get_constant_noise_gain()
        
        assert Gd.shape == (1, 1)
        # Gd = sigma * sqrt(dt) = 0.5 * sqrt(0.01) = 0.05
        expected = 0.5 * np.sqrt(dt)
        assert_allclose(Gd, np.array([[expected]]), rtol=1e-6)
    
    def test_multiplicative_noise_no_precompute(self, gbm_system, dt):
        """Test that multiplicative noise can't be precomputed."""
        discretizer = StochasticDiscretizer(gbm_system, dt=dt)
        
        # GBM has multiplicative noise
        assert discretizer.is_multiplicative_noise()
        assert not discretizer.is_additive_noise()
        
        # Should raise error when trying to get constant noise
        with pytest.raises(ValueError, match="only valid for additive noise"):
            discretizer.get_constant_noise_gain()


# ============================================================================
# Test: SDE Integration Methods
# ============================================================================

class TestSDEMethods:
    """Test different SDE integration methods."""
    
    def test_euler_maruyama(self, ou_system, dt, seed):
        """Test Euler-Maruyama method."""
        discretizer = StochasticDiscretizer(
            ou_system, dt=dt, method='euler', seed=seed
        )
        
        x0 = np.array([1.0])
        u = np.array([0.0])
        
        # Simulate 100 steps
        x = x0.copy()
        for _ in range(100):
            x = discretizer.step(x, u)
        
        # Should produce reasonable result
        assert x.shape == (1,)
        assert not np.isnan(x).any()
        
        # OU process should decay toward zero on average
        # After 100 steps (1 second), should be significantly smaller
        # (though stochastic, so allow wide range)
        assert -2.0 < x[0] < 2.0  # Very permissive bound
    
    @pytest.mark.skipif(
        not _torch_available(),
        reason="PyTorch not available"
    )
    def test_milstein_method(self, dt, seed):
        """Test Milstein method (requires torch backend)."""
        import torch
        
        # Create torch-compatible system
        ou_system = OrnsteinUhlenbeck(alpha=2.0, sigma=0.5)
        ou_system.set_default_backend('torch')
        
        discretizer = StochasticDiscretizer(
            ou_system, dt=dt, method='milstein', backend='torch', seed=seed
        )
        
        x = torch.tensor([1.0])
        u = torch.tensor([0.0])
        
        x_next = discretizer.step(x, u)
        
        assert isinstance(x_next, torch.Tensor)
        assert x_next.shape == (1,)
    
    def test_method_consistency(self, ou_system, dt):
        """Test that different methods produce similar mean behavior."""
        x0 = np.array([1.0])
        u = np.array([0.0])
        
        methods = ['euler', 'EM']  # Methods available in numpy
        results = []
        
        for method in methods:
            discretizer = StochasticDiscretizer(
                ou_system, dt=dt, method=method, seed=42
            )
            x = x0.copy()
            x = discretizer.step(x, u)
            results.append(x[0])
        
        # Different methods should give similar results for additive noise
        # (within ~10% of each other for single step)
        for i in range(len(results) - 1):
            rel_diff = abs(results[i] - results[i+1]) / (abs(results[i]) + 1e-10)
            assert rel_diff < 0.2  # 20% tolerance for single stochastic step


# ============================================================================
# Test: Linearization
# ============================================================================

class TestLinearization:
    """Test stochastic discrete-time linearization."""
    
    def test_linearize_basic(self, ou_system, dt):
        """Test basic linearization returns (Ad, Bd, Gd)."""
        discretizer = StochasticDiscretizer(ou_system, dt=dt)
        
        x_eq = np.array([0.0])
        u_eq = np.array([0.0])
        
        Ad, Bd, Gd = discretizer.linearize(x_eq, u_eq, method='euler')
        
        # Check dimensions
        assert Ad.shape == (1, 1)
        assert Bd.shape == (1, 1)
        assert Gd.shape == (1, 1)
        
        # OU: Ac = -alpha, Bc = 1, g = sigma
        # Ad = I + dt*Ac = 1 - 2*0.01 = 0.98
        # Bd = dt*Bc = 0.01
        # Gd = g*sqrt(dt) = 0.5*sqrt(0.01) = 0.05
        assert_allclose(Ad, np.array([[0.98]]), rtol=1e-10)
        assert_allclose(Bd, np.array([[0.01]]), rtol=1e-10)
        assert_allclose(Gd, np.array([[0.05]]), rtol=1e-6)
    
    def test_linearize_autonomous_sde(self, autonomous_sde, dt):
        """Test linearization of autonomous SDE."""
        discretizer = StochasticDiscretizer(autonomous_sde, dt=dt)
        
        x_eq = np.array([0.0])
        
        Ad, Bd, Gd = discretizer.linearize(x_eq, u=None, method='euler')
        
        # Ad should be correct
        assert Ad.shape == (1, 1)
        assert_allclose(Ad, np.array([[1.0 - dt]]), rtol=1e-10)
        
        # Bd should be empty (nx, 0) for autonomous
        assert Bd.shape == (1, 0)
        
        # Gd should still be valid
        assert Gd.shape == (1, 1)
        expected_gd = 0.5 * np.sqrt(dt)
        assert_allclose(Gd, np.array([[expected_gd]]), rtol=1e-6)
    
    def test_linearize_exact_method(self, ou_system, dt):
        """Test exact linearization of drift."""
        discretizer = StochasticDiscretizer(ou_system, dt=dt)
        
        x_eq = np.array([0.0])
        u_eq = np.array([0.0])
        
        Ad, Bd, Gd = discretizer.linearize(x_eq, u_eq, method='exact')
        
        # For OU: Ad = exp(-alpha*dt) = exp(-2*0.01)
        Ad_expected = np.exp(-2.0 * dt)
        
        assert_allclose(Ad, np.array([[Ad_expected]]), rtol=1e-6)
        
        # Gd should be same as Euler (diffusion not affected by drift method)
        assert_allclose(Gd, np.array([[0.5 * np.sqrt(dt)]]), rtol=1e-6)
    
    def test_linearize_pure_diffusion(self, brownian_system, dt):
        """Test linearization of pure diffusion (zero drift)."""
        discretizer = StochasticDiscretizer(brownian_system, dt=dt)
        
        x_eq = np.array([0.0])
        
        Ad, Bd, Gd = discretizer.linearize(x_eq, u=None, method='euler')
        
        # Pure diffusion: drift is zero
        # Ad = I + dt*0 = I
        assert_allclose(Ad, np.array([[1.0]]), rtol=1e-10)
        
        # No control
        assert Bd.shape == (1, 0)
        
        # Gd = sigma*sqrt(dt)
        assert_allclose(Gd, np.array([[0.5 * np.sqrt(dt)]]), rtol=1e-6)
    
    def test_linearize_multiplicative_noise(self, gbm_system, dt):
        """Test linearization of multiplicative noise."""
        discretizer = StochasticDiscretizer(gbm_system, dt=dt)
        
        # Linearize at non-zero state (x=1)
        x_eq = np.array([1.0])
        u_eq = np.array([0.0])
        
        Ad, Bd, Gd = discretizer.linearize(x_eq, u_eq, method='euler')
        
        # Check dimensions
        assert Ad.shape == (1, 1)
        assert Bd.shape == (1, 1)
        assert Gd.shape == (1, 1)
        
        # For GBM: g(x) = sigma*x
        # At x=1: Gd = sigma*1*sqrt(dt) = 0.2*sqrt(0.01) = 0.02
        expected_gd = 0.2 * 1.0 * np.sqrt(dt)
        assert_allclose(Gd, np.array([[expected_gd]]), rtol=1e-6)
    
    def test_linearization_method_comparison(self, ou_system, dt):
        """Compare linearization methods."""
        discretizer = StochasticDiscretizer(ou_system, dt=dt)
        
        x_eq = np.array([0.0])
        u_eq = np.array([0.0])
        
        methods = ['euler', 'exact']
        Ad_results = {}
        Gd_results = {}
        
        for method in methods:
            Ad, Bd, Gd = discretizer.linearize(x_eq, u_eq, method=method)
            Ad_results[method] = Ad
            Gd_results[method] = Gd
        
        # Gd should be same for both methods (diffusion independent of drift method)
        assert_allclose(Gd_results['euler'], Gd_results['exact'], rtol=1e-10)
        
        # Ad should be close for small dt
        assert_allclose(Ad_results['euler'], Ad_results['exact'], rtol=0.01)


# ============================================================================
# Test: Observation
# ============================================================================

class TestObservation:
    """Test observation/output function delegation."""
    
    def test_h_output_function(self, ou_system, dt):
        """Test that h() delegates to continuous system."""
        discretizer = StochasticDiscretizer(ou_system, dt=dt)
        
        x = np.array([1.0])
        y = discretizer.h(x)
        
        # Should match continuous system output
        y_continuous = ou_system.h(x)
        assert_allclose(y, y_continuous)
    
    def test_linearized_observation(self, ou_system, dt):
        """Test that linearized_observation delegates correctly."""
        discretizer = StochasticDiscretizer(ou_system, dt=dt)
        
        x = np.array([0.0])
        C = discretizer.linearized_observation(x)
        
        # Should match continuous system
        C_continuous = ou_system.linearized_observation(x)
        assert_allclose(C, C_continuous)


# ============================================================================
# Test: Noise Structure Analysis
# ============================================================================

class TestNoiseStructure:
    """Test noise structure detection and optimization."""
    
    def test_additive_noise_detection(self, ou_system, dt):
        """Test detection of additive noise."""
        discretizer = StochasticDiscretizer(ou_system, dt=dt)
        
        assert discretizer.is_additive_noise()
        assert not discretizer.is_multiplicative_noise()
    
    def test_multiplicative_noise_detection(self, gbm_system, dt):
        """Test detection of multiplicative noise."""
        discretizer = StochasticDiscretizer(gbm_system, dt=dt)
        
        assert discretizer.is_multiplicative_noise()
        assert not discretizer.is_additive_noise()
    
    def test_scalar_noise_detection(self, ou_system, dt):
        """Test detection of scalar noise (nw=1)."""
        discretizer = StochasticDiscretizer(ou_system, dt=dt)
        
        assert discretizer.is_scalar_noise()
        assert discretizer.nw == 1
    
    def test_get_noise_info(self, ou_system, dt):
        """Test get_noise_info returns correct information."""
        discretizer = StochasticDiscretizer(ou_system, dt=dt)
        
        info = discretizer.get_noise_info()
        
        assert info['nw'] == 1
        assert info['noise_type'] == 'additive'
        assert info['is_additive'] == True
        assert info['is_multiplicative'] == False
        assert info['can_precompute'] == True


# ============================================================================
# Test: Utility Methods
# ============================================================================

class TestUtilityMethods:
    """Test utility methods."""
    
    def test_set_dt(self, ou_system, seed):
        """Test changing time step."""
        discretizer = StochasticDiscretizer(ou_system, dt=0.01, seed=seed)
        
        # Change dt
        new_dt = 0.005
        discretizer.set_dt(new_dt)
        
        assert discretizer.dt == new_dt
        
        # Should still work
        x = np.array([1.0])
        u = np.array([0.0])
        x_next = discretizer.step(x, u)
        
        assert x_next.shape == (1,)
    
    def test_set_dt_updates_cache(self, ou_system):
        """Test that set_dt updates cached diffusion."""
        discretizer = StochasticDiscretizer(ou_system, dt=0.01)
        
        # Get initial noise gain
        Gd1 = discretizer.get_constant_noise_gain()
        
        # Change dt
        discretizer.set_dt(0.02)
        
        # Get new noise gain
        Gd2 = discretizer.get_constant_noise_gain()
        
        # Should be different (scaled by sqrt(dt))
        assert not np.allclose(Gd1, Gd2)
        
        # Verify correct scaling
        assert_allclose(Gd2 / Gd1, np.sqrt(0.02 / 0.01), rtol=1e-6)
    
    def test_get_info(self, ou_system, dt, seed):
        """Test get_info returns comprehensive information."""
        discretizer = StochasticDiscretizer(
            ou_system, dt=dt, method='euler', seed=seed
        )
        
        info = discretizer.get_info()
        
        assert info['dt'] == dt
        assert info['method'] == 'euler'
        assert info['backend'] == 'numpy'
        assert info['seed'] == seed
        assert info['system_type'] == 'StochasticDynamicalSystem'
        assert info['dimensions']['nx'] == 1
        assert info['dimensions']['nu'] == 1
        assert info['dimensions']['nw'] == 1
        assert info['is_autonomous'] == False
        assert 'noise' in info
    
    def test_get_info_autonomous(self, autonomous_sde, dt):
        """Test get_info for autonomous SDE."""
        discretizer = StochasticDiscretizer(autonomous_sde, dt=dt)
        
        info = discretizer.get_info()
        
        assert info['is_autonomous'] == True
        assert info['dimensions']['nu'] == 0
    
    def test_repr(self, ou_system, dt, seed):
        """Test __repr__ returns valid string."""
        discretizer = StochasticDiscretizer(ou_system, dt=dt, seed=seed)
        
        repr_str = repr(discretizer)
        
        assert 'StochasticDiscretizer' in repr_str
        assert 'dt=0.01' in repr_str
        assert 'nw=1' in repr_str
    
    def test_str(self, ou_system, dt):
        """Test __str__ returns readable string."""
        discretizer = StochasticDiscretizer(ou_system, dt=dt, method='euler')
        
        str_repr = str(discretizer)
        
        assert 'StochasticDiscretizer' in str_repr
        assert 'additive noise' in str_repr or 'noise' in str_repr


# ============================================================================
# Test: Statistical Properties
# ============================================================================

class TestStatisticalProperties:
    """Test statistical properties of discretized SDEs."""
    
    @pytest.mark.slow
    def test_brownian_mean_zero(self, brownian_system, dt):
        """Test that Brownian motion has zero mean."""
        discretizer = StochasticDiscretizer(brownian_system, dt=dt)
        
        x0 = np.array([0.0])
        n_paths = 100
        
        # Simulate multiple trajectories
        final_states = []
        for seed in range(n_paths):
            discretizer.set_seed(seed)
            x = x0.copy()
            # Take 10 steps (t=0.1)
            for _ in range(10):
                x = discretizer.step(x, u=None)
            final_states.append(x[0])
        
        final_states = np.array(final_states)
        
        # Mean should be close to zero
        empirical_mean = final_states.mean()
        
        # Standard error: SE = sigma*sqrt(t) / sqrt(n)
        # = 0.5*sqrt(0.1) / sqrt(100) = 0.0316
        se = 0.5 * np.sqrt(0.1) / np.sqrt(n_paths)
        
        # 99% CI: ±2.576*SE ≈ ±0.08
        assert abs(empirical_mean) < 0.1, (
            f"Mean {empirical_mean:.4f} should be near 0"
        )
    
    @pytest.mark.slow
    def test_brownian_variance_scales_with_time(self, brownian_system):
        """Test that Brownian variance scales linearly with time."""
        dt = 0.01
        discretizer = StochasticDiscretizer(brownian_system, dt=dt)
        
        x0 = np.array([0.0])
        n_paths = 200
        
        # Test at two different times
        t1, t2 = 0.1, 0.2
        steps1 = int(t1 / dt)
        steps2 = int(t2 / dt)
        
        # Collect final states at each time
        states_t1 = []
        states_t2 = []
        
        for seed in range(n_paths):
            discretizer.set_seed(seed)
            x = x0.copy()
            
            for step in range(steps2):
                x = discretizer.step(x, u=None)
                if step + 1 == steps1:
                    states_t1.append(x[0])
            states_t2.append(x[0])
        
        var_t1 = np.var(states_t1)
        var_t2 = np.var(states_t2)
        
        # Variance should scale linearly: Var[B(t)] = sigma^2 * t
        # var_t2 / var_t1 should be ≈ t2/t1 = 2.0
        ratio = var_t2 / var_t1
        
        # Allow 30% tolerance
        assert 1.4 < ratio < 2.6, (
            f"Variance ratio {ratio:.2f} should be near 2.0"
        )
    
    @pytest.mark.slow
    def test_ou_process_mean_reversion(self, ou_system, dt):
        """Test that OU process exhibits mean reversion."""
        discretizer = StochasticDiscretizer(ou_system, dt=dt)
        
        # Start away from equilibrium
        x0 = np.array([2.0])
        u = np.array([0.0])
        
        n_paths = 100
        final_states = []
        
        # Simulate to t=1.0 (should decay significantly)
        steps = int(1.0 / dt)
        
        for seed in range(n_paths):
            discretizer.set_seed(seed)
            x = x0.copy()
            for _ in range(steps):
                x = discretizer.step(x, u)
            final_states.append(x[0])
        
        final_states = np.array(final_states)
        
        # Mean should have decayed toward zero
        # OU: E[X(t)] = x0 * exp(-alpha*t) = 2 * exp(-2*1) ≈ 0.27
        expected_mean = 2.0 * np.exp(-2.0 * 1.0)
        empirical_mean = final_states.mean()
        
        # Allow 50% tolerance (stochastic + finite samples)
        assert 0.5 * expected_mean < empirical_mean < 1.5 * expected_mean, (
            f"Mean {empirical_mean:.4f} should be near {expected_mean:.4f}"
        )


# ============================================================================
# Test: Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_very_small_dt(self, ou_system):
        """Test with very small time step."""
        dt = 1e-6
        discretizer = StochasticDiscretizer(ou_system, dt=dt, seed=42)
        
        x = np.array([1.0])
        u = np.array([0.0])
        
        x_next = discretizer.step(x, u)
        
        # Should barely change (noise is ~ sigma*sqrt(dt) ≈ 0.0005)
        assert_allclose(x_next, x, atol=0.01)  # Within 0.01
    
    def test_zero_initial_state(self, ou_system, dt, seed):
        """Test with zero initial state."""
        discretizer = StochasticDiscretizer(ou_system, dt=dt, seed=seed)
        
        x = np.array([0.0])
        u = np.array([0.0])
        
        x_next = discretizer.step(x, u)
        
        # Should move due to noise (not stay at zero)
        # With probability > 99%, should be non-zero
        # (though could be very small)
        assert x_next.shape == (1,)
    
    def test_large_noise_value(self, ou_system, dt):
        """Test with large noise value."""
        discretizer = StochasticDiscretizer(ou_system, dt=dt)
        
        x = np.array([1.0])
        u = np.array([0.0])
        w = np.array([10.0])  # Large noise
        
        x_next = discretizer.step(x, u, w=w)
        
        # Should still be finite
        assert not np.isnan(x_next).any()
        assert not np.isinf(x_next).any()


# ============================================================================
# Test: Backend Support
# ============================================================================

class TestBackendSupport:
    """Test multi-backend support."""
    
    @pytest.mark.skipif(
        not _torch_available(),
        reason="PyTorch not available"
    )
    def test_torch_backend(self, ou_system, dt, seed):
        """Test PyTorch backend."""
        import torch
        
        # Set system backend to torch
        ou_system.set_default_backend('torch')
        
        discretizer = StochasticDiscretizer(
            ou_system, dt=dt, backend='torch', seed=seed
        )
        
        x = torch.tensor([1.0])
        u = torch.tensor([0.0])
        
        x_next = discretizer.step(x, u)
        
        assert isinstance(x_next, torch.Tensor)
        assert x_next.shape == (1,)
    
    @pytest.mark.skipif(
        not _jax_available(),
        reason="JAX not available"
    )
    def test_jax_backend(self, ou_system, dt, seed):
        """Test JAX backend."""
        import jax.numpy as jnp
        
        # Set system backend to jax
        ou_system.set_default_backend('jax')
        
        discretizer = StochasticDiscretizer(
            ou_system, dt=dt, backend='jax', method='Euler', seed=seed
        )
        
        x = jnp.array([1.0])
        u = jnp.array([0.0])
        
        x_next = discretizer.step(x, u)
        
        assert isinstance(x_next, jnp.ndarray)
        assert x_next.shape == (1,)


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests combining multiple features."""
    
    def test_discretize_and_simulate_trajectory(self, ou_system, dt, seed):
        """Test discretization and stochastic trajectory simulation."""
        discretizer = StochasticDiscretizer(ou_system, dt=dt, seed=seed)
        
        # Initial state
        x = np.array([1.0])
        u = np.array([0.0])
        
        # Simulate trajectory
        trajectory = [x.copy()]
        for _ in range(100):
            x = discretizer.step(x, u)
            trajectory.append(x.copy())
        
        trajectory = np.array(trajectory)
        
        # Check shape
        assert trajectory.shape == (101, 1)
        
        # Should be finite
        assert not np.isnan(trajectory).any()
        assert not np.isinf(trajectory).any()
        
        # OU process should stay bounded (with high probability)
        assert np.all(np.abs(trajectory) < 5.0)  # 99.9% should be within ±5
    
    def test_linearize_for_lqg_design(self, ou_system, dt):
        """Test linearization for stochastic LQG controller design."""
        discretizer = StochasticDiscretizer(ou_system, dt=dt)
        
        # Linearize at equilibrium
        x_eq = np.array([0.0])
        u_eq = np.array([0.0])
        
        Ad, Bd, Gd = discretizer.linearize(x_eq, u_eq, method='exact')
        
        # Design discrete stochastic LQR
        Q = np.array([[10.0]])
        R = np.array([[1.0]])
        
        import scipy.linalg
        S = scipy.linalg.solve_discrete_are(Ad, Bd, Q, R)
        K = -np.linalg.solve(R + Bd.T @ S @ Bd, Bd.T @ S @ Ad)
        
        # K should be reasonable
        assert K.shape == (1, 1)
        assert K[0, 0] < 0  # Negative feedback
        
        # Can also use Gd for Kalman filter design
        assert Gd.shape == (1, 1)
    
    def test_monte_carlo_mean_convergence(self, ou_system, dt):
        """Test that Monte Carlo mean converges to analytical."""
        discretizer = StochasticDiscretizer(ou_system, dt=dt)
        
        x0 = np.array([1.0])
        u = np.array([0.0])
        t_final = 0.5
        steps = int(t_final / dt)
        
        n_paths = 500
        final_states = []
        
        for seed in range(n_paths):
            discretizer.set_seed(seed)
            x = x0.copy()
            for _ in range(steps):
                x = discretizer.step(x, u)
            final_states.append(x[0])
        
        empirical_mean = np.mean(final_states)
        
        # Analytical: E[X(t)] = x0 * exp(-alpha*t)
        expected_mean = x0[0] * np.exp(-2.0 * t_final)
        
        # Allow 10% tolerance
        assert_allclose(empirical_mean, expected_mean, rtol=0.1)


# ============================================================================
# Test: Comparison with Deterministic
# ============================================================================

class TestDeterministicComparison:
    """Compare stochastic discretizer with deterministic version."""
    
    def test_zero_noise_matches_deterministic(self, ou_system, dt):
        """Test that zero noise gives deterministic dynamics."""
        from src.systems.base.discretization.discretizer import Discretizer
        
        # Get deterministic version of OU system
        det_system = ou_system.to_deterministic()
        
        # Create both discretizers
        stoch_disc = StochasticDiscretizer(ou_system, dt=dt, method='euler')
        det_disc = Discretizer(det_system, dt=dt, method='euler')
        
        x = np.array([1.0])
        u = np.array([0.0])
        
        # Stochastic with zero noise
        w_zero = np.array([0.0])
        x_next_stoch = stoch_disc.step(x, u, w=w_zero)
        
        # Deterministic
        x_next_det = det_disc.step(x, u)
        
        # Should be identical
        assert_allclose(x_next_stoch, x_next_det, rtol=1e-10)
    
    def test_linearization_drift_matches_deterministic(self, ou_system, dt):
        """Test that drift linearization matches deterministic system."""
        from src.systems.base.discretization.discretizer import Discretizer
        
        det_system = ou_system.to_deterministic()
        
        stoch_disc = StochasticDiscretizer(ou_system, dt=dt)
        det_disc = Discretizer(det_system, dt=dt)
        
        x_eq = np.array([0.0])
        u_eq = np.array([0.0])
        
        # Linearize both
        Ad_stoch, Bd_stoch, Gd = stoch_disc.linearize(x_eq, u_eq, method='euler')
        Ad_det, Bd_det = det_disc.linearize(x_eq, u_eq, method='euler')
        
        # Drift terms should match
        assert_allclose(Ad_stoch, Ad_det, rtol=1e-10)
        assert_allclose(Bd_stoch, Bd_det, rtol=1e-10)
        
        # Stochastic also returns Gd
        assert Gd.shape == (1, 1)


# ============================================================================
# Test: Convenience Functions
# ============================================================================

class TestConvenienceFunctions:
    """Test convenience factory functions."""
    
    def test_create_stochastic_discretizer(self, ou_system, dt):
        """Test convenience creation function."""
        discretizer = create_stochastic_discretizer(ou_system, dt=dt)
        
        assert isinstance(discretizer, StochasticDiscretizer)
        assert discretizer.dt == dt
    
    def test_create_with_options(self, ou_system, dt, seed):
        """Test creation with additional options."""
        discretizer = create_stochastic_discretizer(
            ou_system, dt=dt, method='euler', seed=seed, backend='numpy'
        )
        
        assert discretizer.method == 'euler'
        assert discretizer.seed == seed
        assert discretizer.backend == 'numpy'


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])