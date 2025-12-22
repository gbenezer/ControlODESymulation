"""
Unit Tests for TorchSDEIntegrator

Tests PyTorch-based SDE integration via torchsde, including:
- Initialization and validation
- Method selection and availability
- Integration with autonomous and controlled systems
- Pure diffusion systems (zero drift)
- PyTorch-specific features (gradients, GPU, batching)
- Adjoint method for neural SDEs
- Reproducibility with seeds
- Error handling and edge cases
"""

import pytest
import numpy as np

# Check if PyTorch and torchsde are available
try:
    import torch
    import torchsde
    TORCH_AVAILABLE = True
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    TORCH_AVAILABLE = False
    CUDA_AVAILABLE = False

from src.systems.base.numerical_integration.stochastic.torchsde_integrator import (
    TorchSDEIntegrator,
    create_torchsde_integrator,
    list_torchsde_methods,
)
from src.systems.base.numerical_integration.stochastic.sde_integrator_base import (
    SDEType,
    ConvergenceType,
    StepMode
)
from src.systems.base.stochastic_dynamical_system import StochasticDynamicalSystem


# ============================================================================
# Skip Tests if PyTorch/torchsde Not Available
# ============================================================================

pytestmark = pytest.mark.skipif(
    not TORCH_AVAILABLE,
    reason="PyTorch or torchsde not installed. Install: pip install torch torchsde"
)


# ============================================================================
# Mock SDE Systems for Testing
# ============================================================================

class OrnsteinUhlenbeck(StochasticDynamicalSystem):
    """Ornstein-Uhlenbeck process: dx = -alpha * x * dt + sigma * dW"""
    
    def define_system(self, alpha=1.0, sigma=0.5):
        import sympy as sp
        
        x = sp.symbols('x', real=True)
        alpha_sym = sp.symbols('alpha', positive=True)
        sigma_sym = sp.symbols('sigma', positive=True)
        
        self.state_vars = [x]
        self.control_vars = []  # Autonomous
        self._f_sym = sp.Matrix([[-alpha_sym * x]])
        self.parameters = {alpha_sym: alpha, sigma_sym: sigma}
        self.order = 1
        
        self.diffusion_expr = sp.Matrix([[sigma_sym]])
        self.sde_type = 'ito'


class BrownianMotion(StochasticDynamicalSystem):
    """Pure Brownian motion: dx = sigma * dW (zero drift)"""
    
    def define_system(self, sigma=1.0):
        import sympy as sp
        
        x = sp.symbols('x', real=True)
        sigma_sym = sp.symbols('sigma', positive=True)
        
        self.state_vars = [x]
        self.control_vars = []  # Autonomous
        self._f_sym = sp.Matrix([[0]])  # Zero drift!
        self.parameters = {sigma_sym: sigma}
        self.order = 1
        
        self.diffusion_expr = sp.Matrix([[sigma_sym]])
        self.sde_type = 'ito'


class ControlledOU(StochasticDynamicalSystem):
    """Controlled OU: dx = (-alpha * x + u) * dt + sigma * dW"""
    
    def define_system(self, alpha=1.0, sigma=0.5):
        import sympy as sp
        
        x = sp.symbols('x', real=True)
        u = sp.symbols('u', real=True)
        alpha_sym = sp.symbols('alpha', positive=True)
        sigma_sym = sp.symbols('sigma', positive=True)
        
        self.state_vars = [x]
        self.control_vars = [u]
        self._f_sym = sp.Matrix([[-alpha_sym * x + u]])
        self.parameters = {alpha_sym: alpha, sigma_sym: sigma}
        self.order = 1
        
        self.diffusion_expr = sp.Matrix([[sigma_sym]])
        self.sde_type = 'ito'


class TwoDimensionalOU(StochasticDynamicalSystem):
    """2D OU with diagonal noise (autonomous)"""
    
    def define_system(self, alpha=1.0, sigma1=0.5, sigma2=0.3):
        import sympy as sp
        
        x1, x2 = sp.symbols('x1 x2', real=True)
        alpha_sym = sp.symbols('alpha', positive=True)
        sigma1_sym = sp.symbols('sigma1', positive=True)
        sigma2_sym = sp.symbols('sigma2', positive=True)
        
        self.state_vars = [x1, x2]
        self.control_vars = []  # Autonomous
        self._f_sym = sp.Matrix([
            [-alpha_sym * x1],
            [-alpha_sym * x2]
        ])
        self.parameters = {
            alpha_sym: alpha,
            sigma1_sym: sigma1,
            sigma2_sym: sigma2
        }
        self.order = 1
        
        self.diffusion_expr = sp.Matrix([
            [sigma1_sym, 0],
            [0, sigma2_sym]
        ])
        self.sde_type = 'ito'


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def ou_system():
    """Create Ornstein-Uhlenbeck system."""
    return OrnsteinUhlenbeck(alpha=1.0, sigma=0.5)


@pytest.fixture
def brownian_system():
    """Create pure Brownian motion."""
    return BrownianMotion(sigma=1.0)


@pytest.fixture
def controlled_system():
    """Create controlled OU system."""
    return ControlledOU(alpha=1.0, sigma=0.5)


@pytest.fixture
def ou_2d_system():
    """Create 2D OU system."""
    return TwoDimensionalOU(alpha=1.0, sigma1=0.5, sigma2=0.3)


@pytest.fixture
def integrator_euler(ou_system):
    """Create Euler integrator."""
    return TorchSDEIntegrator(
        ou_system,
        dt=0.01,
        method='euler',
        seed=42
    )


# ============================================================================
# Test Class: Initialization and Validation
# ============================================================================

class TestTorchSDEInitialization:
    """Test initialization and validation."""
    
    def test_basic_initialization(self, ou_system):
        """Test basic integrator initialization."""
        integrator = TorchSDEIntegrator(
            ou_system,
            dt=0.01,
            method='euler'
        )
        
        assert integrator.sde_system is ou_system
        assert integrator.dt == 0.01
        assert integrator.method == 'euler'
        assert integrator.backend == 'torch'
    
    def test_backend_must_be_torch(self, ou_system):
        """Test that non-torch backend raises error."""
        with pytest.raises(ValueError, match="requires backend='torch'"):
            TorchSDEIntegrator(
                ou_system,
                dt=0.01,
                method='euler',
                backend='jax'
            )
    
    def test_invalid_method_raises(self, ou_system):
        """Test that invalid method raises error."""
        with pytest.raises(ValueError, match="Unknown method"):
            TorchSDEIntegrator(
                ou_system,
                dt=0.01,
                method='nonexistent_method'
            )
    
    def test_valid_methods_accepted(self, ou_system):
        """Test that all listed methods are accepted."""
        methods = ['euler', 'milstein', 'srk', 'midpoint']
        
        for method in methods:
            integrator = TorchSDEIntegrator(
                ou_system,
                dt=0.01,
                method=method
            )
            assert integrator.method == method
    
    def test_adjoint_flag(self, ou_system):
        """Test adjoint method flag."""
        integrator = TorchSDEIntegrator(
            ou_system,
            dt=0.01,
            method='euler',
            adjoint=True
        )
        
        assert integrator.use_adjoint is True
    
    def test_noise_type_auto_detection_additive(self, ou_system):
        """Test automatic noise type detection for additive noise."""
        integrator = TorchSDEIntegrator(
            ou_system,
            dt=0.01,
            method='euler'
        )
        
        assert integrator.noise_type == 'additive'
    
    def test_noise_type_auto_detection_diagonal(self, ou_2d_system):
        """Test automatic noise type detection for diagonal noise."""
        integrator = TorchSDEIntegrator(
            ou_2d_system,
            dt=0.01,
            method='euler'
        )
        
        assert integrator.noise_type == 'diagonal'
    
    def test_seed_initialization(self, ou_system):
        """Test random seed initialization."""
        integrator = TorchSDEIntegrator(
            ou_system,
            dt=0.01,
            method='euler',
            seed=42
        )
        
        assert integrator.seed == 42
    
    def test_default_device_cpu(self, integrator_euler):
        """Test that default device is CPU."""
        assert integrator_euler.device == torch.device('cpu')


# ============================================================================
# Test Class: Autonomous Systems
# ============================================================================

class TestAutonomousSystems:
    """Test integration of autonomous SDE systems."""
    
    def test_autonomous_ou_integration(self, ou_system):
        """Test basic integration of autonomous OU process."""
        integrator = TorchSDEIntegrator(
            ou_system,
            dt=0.01,
            method='euler',
            seed=42
        )
        
        x0 = torch.tensor([1.0])
        u_func = lambda t, x: None
        t_span = (0.0, 1.0)
        
        result = integrator.integrate(x0, u_func, t_span)
        
        assert result.success, f"Integration failed: {result.message}"
        assert result.x.shape[0] > 10, "Not enough time points"
        assert result.x.shape[1] == 1, "Wrong state dimension"
        assert result.nsteps > 0, "No steps recorded"
        
        # State should have evolved
        assert not torch.allclose(result.x[-1], x0), "State didn't evolve"
    
    def test_autonomous_2d_integration(self, ou_2d_system):
        """Test integration of 2D autonomous system."""
        integrator = TorchSDEIntegrator(
            ou_2d_system,
            dt=0.01,
            method='euler',
            seed=42
        )
        
        x0 = torch.tensor([1.0, 2.0])
        u_func = lambda t, x: None
        t_span = (0.0, 1.0)
        
        result = integrator.integrate(x0, u_func, t_span)
        
        assert result.success
        assert result.x.shape[1] == 2, "Should be 2D system"
        assert result.x.shape[0] > 10, "Need multiple time points"
        
        # Both dimensions should evolve
        assert not torch.allclose(result.x[-1, 0], x0[0])
        assert not torch.allclose(result.x[-1, 1], x0[1])
    
    def test_autonomous_reproducibility_with_seed(self, ou_system):
        """Test that same seed gives reproducible results."""
        x0 = torch.tensor([1.0])
        u_func = lambda t, x: None
        t_span = (0.0, 1.0)
        
        # First run
        torch.manual_seed(42)
        integrator1 = TorchSDEIntegrator(
            ou_system, dt=0.01, method='euler', seed=42
        )
        result1 = integrator1.integrate(x0, u_func, t_span)
        
        # Second run with same seed
        torch.manual_seed(42)
        integrator2 = TorchSDEIntegrator(
            ou_system, dt=0.01, method='euler', seed=42
        )
        result2 = integrator2.integrate(x0, u_func, t_span)
        
        # Results should be identical (PyTorch has good seed control)
        torch.testing.assert_close(result1.x, result2.x, rtol=1e-5, atol=1e-7)
    
    def test_autonomous_different_seeds_differ(self, ou_system):
        """Test that different seeds give different results."""
        x0 = torch.tensor([1.0])
        u_func = lambda t, x: None
        t_span = (0.0, 1.0)
        
        # Two runs with different seeds
        torch.manual_seed(42)
        integrator1 = TorchSDEIntegrator(
            ou_system, dt=0.01, method='euler', seed=42
        )
        result1 = integrator1.integrate(x0, u_func, t_span)
        
        torch.manual_seed(123)
        integrator2 = TorchSDEIntegrator(
            ou_system, dt=0.01, method='euler', seed=123
        )
        result2 = integrator2.integrate(x0, u_func, t_span)
        
        # Should be different
        assert not torch.allclose(result1.x, result2.x)


# ============================================================================
# Test Class: Pure Diffusion Systems
# ============================================================================

class TestPureDiffusionSystems:
    """Test pure diffusion systems (zero drift)."""
    
    def test_pure_diffusion_properties(self, brownian_system):
        """Test that Brownian motion system has correct properties."""
        assert brownian_system.is_pure_diffusion()
        assert brownian_system.nu == 0  # Autonomous
        assert brownian_system.nx == 1
        assert brownian_system.nw == 1
    
    def test_pure_diffusion_integration(self, brownian_system):
        """Test basic integration of pure Brownian motion."""
        integrator = TorchSDEIntegrator(
            brownian_system,
            dt=0.01,
            method='euler',
            seed=42
        )
        
        x0 = torch.tensor([0.0])
        u_func = lambda t, x: None
        t_span = (0.0, 1.0)
        
        result = integrator.integrate(x0, u_func, t_span)
        
        assert result.success
        assert result.x.shape[0] > 10
        assert result.nsteps > 0
    
    def test_pure_diffusion_zero_drift(self, brownian_system):
        """Test that drift is exactly zero for pure diffusion."""
        integrator = TorchSDEIntegrator(
            brownian_system,
            dt=0.01,
            method='euler',
            seed=42
        )
        
        x = torch.tensor([1.0])
        
        # Evaluate drift (should be zero)
        drift = integrator._evaluate_drift(x, None)
        
        assert torch.allclose(drift, torch.tensor([0.0]))
    
    def test_pure_diffusion_reproducibility(self, brownian_system):
        """Test reproducibility of pure diffusion with seeds."""
        x0 = torch.tensor([0.0])
        u_func = lambda t, x: None
        t_span = (0.0, 1.0)
        
        # Two runs with same seed
        torch.manual_seed(100)
        integrator1 = TorchSDEIntegrator(
            brownian_system, dt=0.01, method='euler', seed=100
        )
        result1 = integrator1.integrate(x0, u_func, t_span)
        
        torch.manual_seed(100)
        integrator2 = TorchSDEIntegrator(
            brownian_system, dt=0.01, method='euler', seed=100
        )
        result2 = integrator2.integrate(x0, u_func, t_span)
        
        # Should be identical
        torch.testing.assert_close(result1.x, result2.x, rtol=1e-6, atol=1e-8)
    
    def test_pure_diffusion_state_evolution(self, brownian_system):
        """Test that pure diffusion actually moves the state."""
        x0 = torch.tensor([0.0])
        u_func = lambda t, x: None
        t_span = (0.0, 2.0)
        
        # Run a few times with different seeds
        final_states = []
        for seed in range(10):
            torch.manual_seed(seed)
            integrator = TorchSDEIntegrator(
                brownian_system, dt=0.01, method='euler', seed=seed
            )
            result = integrator.integrate(x0, u_func, t_span)
            final_states.append(result.x[-1, 0].item())
        
        # States should vary
        unique_values = len(set([round(s, 6) for s in final_states]))
        assert unique_values > 5, "States should vary across runs"
    
    @pytest.mark.slow
    def test_pure_diffusion_variance_growth(self, brownian_system):
        """Test that variance grows linearly: Var(X(t)) = sigma^2 * t."""
        x0 = torch.tensor([0.0])
        u_func = lambda t, x: None
        
        # Test at two different times
        t1 = 0.5
        t2 = 1.5
        n_paths = 100
        
        # Collect samples at t1
        states_t1 = []
        for seed in range(n_paths):
            torch.manual_seed(seed)
            integrator = TorchSDEIntegrator(
                brownian_system, dt=0.01, method='euler', seed=seed
            )
            result = integrator.integrate(x0, u_func, (0.0, t1))
            states_t1.append(result.x[-1, 0].item())
        
        # Collect samples at t2
        states_t2 = []
        for seed in range(n_paths):
            torch.manual_seed(seed)
            integrator = TorchSDEIntegrator(
                brownian_system, dt=0.01, method='euler', seed=seed
            )
            result = integrator.integrate(x0, u_func, (0.0, t2))
            states_t2.append(result.x[-1, 0].item())
        
        var_t1 = np.var(states_t1)
        var_t2 = np.var(states_t2)
        
        # Variance ratio should equal time ratio
        time_ratio = t2 / t1
        var_ratio = var_t2 / var_t1
        
        # Allow 30% tolerance
        assert abs(var_ratio - time_ratio) / time_ratio < 0.3


# ============================================================================
# Test Class: Controlled Systems
# ============================================================================

class TestControlledSystems:
    """Test integration with control inputs."""
    
    def test_controlled_integration(self, controlled_system):
        """Test integration with constant control."""
        integrator = TorchSDEIntegrator(
            controlled_system,
            dt=0.01,
            method='euler',
            seed=42
        )
        
        x0 = torch.tensor([1.0])
        u_func = lambda t, x: torch.tensor([0.5])
        t_span = (0.0, 1.0)
        
        result = integrator.integrate(x0, u_func, t_span)
        
        assert result.success
        assert result.x.shape[0] > 10
        assert result.nsteps > 0
    
    def test_state_feedback_control(self, controlled_system):
        """Test state feedback control."""
        integrator = TorchSDEIntegrator(
            controlled_system,
            dt=0.01,
            method='euler',
            seed=42
        )
        
        x0 = torch.tensor([1.0])
        K = torch.tensor([2.0])
        u_func = lambda t, x: -K * x
        t_span = (0.0, 2.0)
        
        result = integrator.integrate(x0, u_func, t_span)
        
        assert result.success
        assert result.x.shape[0] > 10
    
    def test_time_varying_control(self, controlled_system):
        """Test time-varying control."""
        integrator = TorchSDEIntegrator(
            controlled_system,
            dt=0.01,
            method='euler',
            seed=42
        )
        
        x0 = torch.tensor([0.0])
        u_func = lambda t, x: torch.tensor([np.sin(2*np.pi*float(t))])
        t_span = (0.0, 2.0)
        
        result = integrator.integrate(x0, u_func, t_span)
        
        assert result.success


# ============================================================================
# Test Class: Integration Methods
# ============================================================================

class TestIntegrationMethods:
    """Test integration functionality."""
    
    def test_integrate_returns_result(self, integrator_euler):
        """Test that integrate returns proper result object."""
        x0 = torch.tensor([1.0])
        u_func = lambda t, x: None
        t_span = (0.0, 1.0)
        
        result = integrator_euler.integrate(x0, u_func, t_span)
        
        assert hasattr(result, 't')
        assert hasattr(result, 'x')
        assert hasattr(result, 'success')
        assert hasattr(result, 'nsteps')
        assert result.success
    
    def test_integrate_with_t_eval(self, integrator_euler):
        """Test integration with specific evaluation times."""
        x0 = torch.tensor([1.0])
        u_func = lambda t, x: None
        t_span = (0.0, 1.0)
        t_eval = torch.linspace(0, 1, 51)
        
        result = integrator_euler.integrate(x0, u_func, t_span, t_eval=t_eval)
        
        assert result.success
        assert len(result.t) == len(t_eval)
    
    def test_step_method(self, integrator_euler):
        """Test single step method."""
        x0 = torch.tensor([1.0])
        u = None
        dt = 0.01
        
        x1 = integrator_euler.step(x0, u, dt)
        
        assert x1.shape == x0.shape
        assert torch.all(torch.isfinite(x1))
    
    def test_statistics_tracked(self, integrator_euler):
        """Test that statistics are tracked during integration."""
        x0 = torch.tensor([1.0])
        u_func = lambda t, x: None
        t_span = (0.0, 0.5)
        
        integrator_euler.reset_stats()
        result = integrator_euler.integrate(x0, u_func, t_span)
        
        stats = integrator_euler.get_sde_stats()
        assert stats['total_fev'] > 0
        assert stats['diffusion_evals'] > 0
        assert stats['total_steps'] > 0


# ============================================================================
# Test Class: Method Selection
# ============================================================================

class TestMethodSelection:
    """Test method recommendation and information."""
    
    def test_list_methods(self):
        """Test that list_methods returns categories."""
        methods = TorchSDEIntegrator.list_methods()
        
        assert 'basic' in methods
        assert 'high_accuracy' in methods
        assert isinstance(methods['basic'], list)
        assert 'euler' in methods['basic']
    
    def test_get_method_info(self):
        """Test getting method information."""
        info = TorchSDEIntegrator.get_method_info('euler')
        
        assert 'name' in info
        assert 'description' in info
        assert info['strong_order'] == 0.5
        assert info['weak_order'] == 1.0
    
    def test_recommend_method_neural_sde(self):
        """Test method recommendation for neural SDEs."""
        method = TorchSDEIntegrator.recommend_method('neural_sde')
        
        assert method == 'euler'
    
    def test_recommend_method_high_accuracy(self):
        """Test method recommendation for high accuracy."""
        method = TorchSDEIntegrator.recommend_method('high_accuracy')
        
        assert method == 'srk'


# ============================================================================
# Test Class: PyTorch-Specific Features
# ============================================================================

class TestPyTorchFeatures:
    """Test PyTorch-specific features."""
    
    def test_integrate_with_gradient(self, ou_system):
        """Test gradient computation through integration."""
        integrator = TorchSDEIntegrator(
            ou_system,
            dt=0.01,
            method='euler',
            seed=42
        )
        
        x0 = torch.tensor([1.0], requires_grad=True)
        u_func = lambda t, x: None
        t_span = (0.0, 0.5)
        
        def loss_fn(result):
            return torch.sum(result.x[-1]**2)
        
        loss, grad = integrator.integrate_with_gradient(
            x0, u_func, t_span, loss_fn
        )
        
        assert isinstance(loss, float)
        assert grad.shape == x0.shape
        assert torch.all(torch.isfinite(grad))
    
    def test_enable_disable_adjoint(self, ou_system):
        """Test enabling and disabling adjoint method."""
        integrator = TorchSDEIntegrator(
            ou_system,
            dt=0.01,
            method='euler'
        )
        
        assert not integrator.use_adjoint
        
        integrator.enable_adjoint()
        assert integrator.use_adjoint
        
        integrator.disable_adjoint()
        assert not integrator.use_adjoint
    
    def test_vectorized_step(self, ou_system):
        """Test vectorized step over batch."""
        integrator = TorchSDEIntegrator(
            ou_system,
            dt=0.01,
            method='euler',
            seed=42
        )
        
        x_batch = torch.tensor([[1.0], [2.0], [3.0]])
        u_batch = None  # Autonomous
        dt = 0.01
        
        x_next = integrator.vectorized_step(x_batch, u_batch, dt)
        
        assert x_next.shape == x_batch.shape
        assert torch.all(torch.isfinite(x_next))
    
    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_gpu_acceleration(self, ou_system):
        """Test GPU acceleration."""
        integrator = TorchSDEIntegrator(
            ou_system,
            dt=0.01,
            method='euler',
            seed=42
        )
        
        integrator.to_device('cuda')
        
        x0 = torch.tensor([1.0], device='cuda')
        u_func = lambda t, x: None
        t_span = (0.0, 0.5)
        
        result = integrator.integrate(x0, u_func, t_span)
        
        assert result.success
        assert result.x.device.type == 'cuda'


# ============================================================================
# Test Class: Different Methods
# ============================================================================

class TestDifferentMethods:
    """Test different integration methods."""
    
    def test_euler_method(self, ou_system):
        """Test Euler-Maruyama method."""
        integrator = TorchSDEIntegrator(
            ou_system,
            dt=0.01,
            method='euler',
            seed=42
        )
        
        x0 = torch.tensor([1.0])
        result = integrator.integrate(x0, lambda t, x: None, (0.0, 1.0))
        
        assert result.success
    
    def test_milstein_method(self, ou_system):
        """Test Milstein method."""
        integrator = TorchSDEIntegrator(
            ou_system,
            dt=0.01,
            method='milstein',
            seed=42
        )
        
        x0 = torch.tensor([1.0])
        result = integrator.integrate(x0, lambda t, x: None, (0.0, 1.0))
        
        assert result.success
    
    def test_srk_method(self, ou_system):
        """Test Stochastic Runge-Kutta method."""
        integrator = TorchSDEIntegrator(
            ou_system,
            dt=0.01,
            method='srk',
            seed=42
        )
        
        x0 = torch.tensor([1.0])
        result = integrator.integrate(x0, lambda t, x: None, (0.0, 1.0))
        
        assert result.success
    
    def test_midpoint_method(self, ou_system):
        """Test midpoint method."""
        integrator = TorchSDEIntegrator(
            ou_system,
            dt=0.01,
            method='midpoint',
            seed=42
        )
        
        x0 = torch.tensor([1.0])
        result = integrator.integrate(x0, lambda t, x: None, (0.0, 1.0))
        
        assert result.success


# ============================================================================
# Test Class: Adjoint Method
# ============================================================================

class TestAdjointMethod:
    """Test adjoint method for neural SDEs."""
    
    def test_adjoint_integration(self, ou_system):
        """Test integration with adjoint method."""
        integrator = TorchSDEIntegrator(
            ou_system,
            dt=0.01,
            method='euler',
            adjoint=True,
            seed=42
        )
        
        x0 = torch.tensor([1.0])
        u_func = lambda t, x: None
        t_span = (0.0, 1.0)
        
        result = integrator.integrate(x0, u_func, t_span)
        
        assert result.success
    
    def test_adjoint_gradient_computation(self, ou_system):
        """Test that adjoint method computes gradients."""
        integrator = TorchSDEIntegrator(
            ou_system,
            dt=0.01,
            method='euler',
            adjoint=True,
            seed=42
        )
        
        x0 = torch.tensor([1.0], requires_grad=True)
        u_func = lambda t, x: None
        t_span = (0.0, 0.5)
        
        result = integrator.integrate(x0, u_func, t_span)
        loss = torch.sum(result.x[-1]**2)
        loss.backward()
        
        assert x0.grad is not None
        assert torch.all(torch.isfinite(x0.grad))


# ============================================================================
# Test Class: Device Management
# ============================================================================

class TestDeviceManagement:
    """Test device management (CPU/GPU)."""
    
    def test_default_device_cpu(self, integrator_euler):
        """Test that default device is CPU."""
        assert integrator_euler.device == torch.device('cpu')
    
    def test_to_device_cpu(self, integrator_euler):
        """Test moving to CPU device."""
        integrator_euler.to_device('cpu')
        assert integrator_euler.device == torch.device('cpu')
    
    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_to_device_cuda(self, integrator_euler):
        """Test moving to CUDA device."""
        integrator_euler.to_device('cuda')
        assert integrator_euler.device == torch.device('cuda')
        
        integrator_euler.to_device('cuda:0')
        assert integrator_euler.device == torch.device('cuda:0')


# ============================================================================
# Test Class: Edge Cases and Error Handling
# ============================================================================

class TestEdgeCasesErrorHandling:
    """Test edge cases and error handling."""
    
    def test_invalid_time_span_raises(self, integrator_euler):
        """Test that backward time span raises error."""
        x0 = torch.tensor([1.0])
        u_func = lambda t, x: None
        t_span = (1.0, 0.0)  # Backward
        
        with pytest.raises(ValueError, match="End time must be greater"):
            integrator_euler.integrate(x0, u_func, t_span)
    
    def test_missing_dt_raises(self, ou_system):
        """Test that fixed mode without dt raises error."""
        integrator = TorchSDEIntegrator(
            ou_system,
            dt=None,
            step_mode=StepMode.FIXED,
            method='euler'
        )
        
        with pytest.raises(ValueError, match="dt must be specified"):
            integrator.step(torch.tensor([1.0]), None)
    
    def test_short_time_span(self, integrator_euler):
        """Test integration with very short time span."""
        x0 = torch.tensor([1.0])
        u_func = lambda t, x: None
        t_span = (0.0, 0.01)
        
        result = integrator_euler.integrate(x0, u_func, t_span)
        
        assert result.success or len(result.t) >= 1


# ============================================================================
# Test Class: Utility Functions
# ============================================================================

class TestUtilityFunctions:
    """Test utility and helper functions."""
    
    def test_create_torchsde_integrator(self, ou_system):
        """Test factory function."""
        integrator = create_torchsde_integrator(
            ou_system,
            method='euler',
            dt=0.01
        )
        
        assert isinstance(integrator, TorchSDEIntegrator)
        assert integrator.method == 'euler'
    
    def test_list_torchsde_methods_output(self, capsys):
        """Test that list function prints output."""
        list_torchsde_methods()
        
        captured = capsys.readouterr()
        assert 'TorchSDE Methods' in captured.out
        assert 'euler' in captured.out


# ============================================================================
# Test Class: Noise Type Handling
# ============================================================================

class TestNoiseTypeHandling:
    """Test noise type detection and handling."""
    
    def test_additive_noise_detection(self, ou_system):
        """Test detection of additive noise."""
        integrator = TorchSDEIntegrator(
            ou_system,
            dt=0.01,
            method='euler'
        )
        
        assert integrator.noise_type == 'additive'
    
    def test_diagonal_noise_detection(self, ou_2d_system):
        """Test detection of diagonal noise."""
        integrator = TorchSDEIntegrator(
            ou_2d_system,
            dt=0.01,
            method='euler'
        )
        
        assert integrator.noise_type == 'diagonal'
    
    def test_manual_noise_type_override(self, ou_system):
        """Test manual noise type specification."""
        integrator = TorchSDEIntegrator(
            ou_system,
            dt=0.01,
            method='euler',
            noise_type='general'
        )
        
        assert integrator.noise_type == 'general'


# ============================================================================
# Test Class: Qualitative Behavior
# ============================================================================

class TestQualitativeBehavior:
    """Test qualitative behavior of integration."""
    
    def test_ou_mean_reversion(self, ou_system):
        """Test that OU process shows mean reversion."""
        integrator = TorchSDEIntegrator(
            ou_system,
            dt=0.01,
            method='euler',
            seed=42
        )
        
        x0 = torch.tensor([5.0])  # Start far from equilibrium
        u_func = lambda t, x: None
        t_span = (0.0, 3.0)
        
        # Run a few trajectories
        final_states = []
        for seed in range(10):
            torch.manual_seed(seed)
            integrator_fresh = TorchSDEIntegrator(
                ou_system, dt=0.01, method='euler', seed=seed
            )
            result = integrator_fresh.integrate(x0, u_func, t_span)
            final_states.append(result.x[-1, 0].item())
        
        # Mean should be closer to zero than initial
        mean_final = np.mean(final_states)
        assert abs(mean_final) < abs(x0.item())
    
    def test_diffusion_increases_spread(self, ou_system):
        """Test that diffusion causes trajectories to spread."""
        x0 = torch.tensor([0.0])
        u_func = lambda t, x: None
        t_span = (0.0, 0.5)
        
        # Multiple runs with different seeds
        states = []
        for seed in range(20):
            torch.manual_seed(seed)
            integrator = TorchSDEIntegrator(
                ou_system, dt=0.01, method='euler', seed=seed
            )
            result = integrator.integrate(x0, u_func, t_span)
            states.append(result.x[-1, 0].item())
        
        # States should have non-zero spread
        spread = np.std(states)
        assert spread > 0.05, f"Too little spread: {spread}"


# ============================================================================
# Test Class: String Representations
# ============================================================================

class TestStringRepresentations:
    """Test string representations."""
    
    def test_integrator_name(self, integrator_euler):
        """Test integrator name property."""
        name = integrator_euler.name
        
        assert 'torchsde' in name
        assert 'euler' in name
        assert 'Fixed' in name or 'Adaptive' in name
    
    def test_integrator_name_with_adjoint(self, ou_system):
        """Test integrator name includes adjoint flag."""
        integrator = TorchSDEIntegrator(
            ou_system,
            dt=0.01,
            method='euler',
            adjoint=True
        )
        
        name = integrator.name
        assert 'Adjoint' in name
    
    def test_repr(self, integrator_euler):
        """Test __repr__ method."""
        repr_str = repr(integrator_euler)
        
        assert 'TorchSDEIntegrator' in repr_str


if __name__ == '__main__':
    pytest.main([__file__, '-v'])