"""
Unit tests for SDEIntegratorFactory

Tests factory creation, method routing, use-case helpers, and integration
for controlled systems, autonomous systems, and pure diffusion processes.

Run with:
    pytest test_sde_integrator_factory.py -v
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

# Import factory and base classes
from src.systems.base.numerical_integration.stochastic.sde_integrator_factory import (
    SDEIntegratorFactory,
    SDEIntegratorType,
    create_sde_integrator,
    auto_sde_integrator,
)
from src.systems.base.numerical_integration.integrator_base import StepMode
from src.systems.base.numerical_integration.stochastic.sde_integrator_base import (
    SDEIntegratorBase,
    ConvergenceType,
    SDEType,
    SDEIntegrationResult,
)
from src.systems.base.stochastic_dynamical_system import StochasticDynamicalSystem


# ============================================================================
# Mock SDE Systems for Testing
# ============================================================================

class MockSDESystem(StochasticDynamicalSystem):
    """Mock stochastic dynamical system for testing."""
    
    def __init__(self, nx=2, nu=1, nw=2, noise_type='general'):
        # Don't call super().__init__() to avoid full initialization
        # Just set the required attributes
        self.nx = nx
        self.nu = nu
        self.nw = nw
        self.sde_type = SDEType.ITO
        self._noise_type = noise_type
        
        # Mock the system name
        self._system_name = "MockSDESystem"
    
    def define_system(self):
        """Required abstract method - mock implementation."""
        pass
    
    def drift(self, x, u, backend='numpy'):
        """Mock drift function: f(x, u) = -x + u"""
        if u is None:
            return -x  # Autonomous
        if backend == 'numpy':
            return -x + np.asarray(u)[:self.nx]
        elif backend == 'torch':
            import torch
            return -x + torch.as_tensor(u)[:self.nx]
        elif backend == 'jax':
            import jax.numpy as jnp
            return -x + jnp.asarray(u)[:self.nx]
    
    def diffusion(self, x, u, backend='numpy'):
        """Mock diffusion function: g(x, u) = constant or multiplicative"""
        if self._noise_type == 'additive':
            # Constant diffusion
            if backend == 'numpy':
                return 0.1 * np.ones((self.nx, self.nw))
            elif backend == 'torch':
                import torch
                return 0.1 * torch.ones((self.nx, self.nw))
            elif backend == 'jax':
                import jax.numpy as jnp
                return 0.1 * jnp.ones((self.nx, self.nw))
        else:
            # State-dependent diffusion
            if backend == 'numpy':
                return 0.1 * np.diag(np.abs(x))[:, :self.nw]
            elif backend == 'torch':
                import torch
                return 0.1 * torch.diag(torch.abs(x))[:, :self.nw]
            elif backend == 'jax':
                import jax.numpy as jnp
                return 0.1 * jnp.diag(jnp.abs(x))[:, :self.nw]
    
    def is_additive_noise(self):
        return self._noise_type == 'additive'
    
    def is_diagonal_noise(self):
        return self._noise_type == 'diagonal'
    
    def is_multiplicative_noise(self):
        return self._noise_type != 'additive'
    
    def is_scalar_noise(self):
        return self.nw == 1 and self.nx == 1
    
    def get_noise_type(self):
        from src.systems.base.utils.stochastic.noise_analysis import NoiseType
        if self._noise_type == 'additive':
            return NoiseType.ADDITIVE
        elif self._noise_type == 'diagonal':
            return NoiseType.DIAGONAL
        else:
            return NoiseType.GENERAL
    
    def get_constant_noise(self, backend='numpy'):
        if self._noise_type == 'additive':
            return self.diffusion(np.zeros(self.nx), None, backend)
        return None


class MockAutonomousSDESystem(MockSDESystem):
    """Mock autonomous SDE system (nu=0)."""
    
    def __init__(self, nx=2, nw=2, noise_type='general'):
        super().__init__(nx=nx, nu=0, nw=nw, noise_type=noise_type)
    
    def drift(self, x, u, backend='numpy'):
        """Autonomous drift: f(x) = -x"""
        return -x
    
    def diffusion(self, x, u, backend='numpy'):
        """Autonomous diffusion"""
        return super().diffusion(x, None, backend)


class MockPureDiffusionSystem(MockSDESystem):
    """Pure diffusion process (zero drift)."""
    
    def __init__(self, nx=2, nw=2, noise_type='additive'):
        super().__init__(nx=nx, nu=0, nw=nw, noise_type=noise_type)
    
    def drift(self, x, u, backend='numpy'):
        """Zero drift"""
        if backend == 'numpy':
            return np.zeros_like(x)
        elif backend == 'torch':
            import torch
            return torch.zeros_like(x)
        elif backend == 'jax':
            import jax.numpy as jnp
            return jnp.zeros_like(x)


# ============================================================================
# Mock Integrators
# ============================================================================

class MockSDEIntegrator(SDEIntegratorBase):
    """Mock SDE integrator for testing factory."""
    
    def __init__(self, sde_system, dt=0.01, backend='numpy', 
                 method='mock', **options):
        super().__init__(
            sde_system,
            dt=dt,
            step_mode=StepMode.FIXED,
            backend=backend,
            **options
        )
        self.method = method
        self._name = f"Mock-{method}"
    
    @property
    def name(self):
        return self._name
    
    def step(self, x, u=None, dt=None, dW=None):
        """Mock step - just return x + small change"""
        dt = dt if dt is not None else self.dt
        dx = self._evaluate_drift(x, u)
        return x + dt * dx
    
    def integrate(self, x0, u_func, t_span, t_eval=None, dense_output=False):
        """Mock integration"""
        t0, tf = t_span
        
        # Create simple trajectory
        if t_eval is None:
            n_steps = max(2, int((tf - t0) / self.dt) + 1)
            t_points = np.linspace(t0, tf, n_steps)
        else:
            t_points = np.asarray(t_eval)
        
        trajectory = [x0]
        x = x0
        
        for i in range(len(t_points) - 1):
            t = t_points[i]
            u = u_func(float(t), x)
            x = self.step(x, u)
            trajectory.append(x)
        
        x_traj = np.stack(trajectory)
        
        return SDEIntegrationResult(
            t=t_points,
            x=x_traj,
            success=True,
            nfev=len(t_points) - 1,
            nsteps=len(t_points) - 1,
            diffusion_evals=len(t_points) - 1,
            n_paths=1,
            convergence_type=self.convergence_type,
            solver=self.method,
            sde_type=self.sde_type,
        )


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def controlled_sde_system():
    """Controlled SDE system with nu > 0."""
    return MockSDESystem(nx=2, nu=1, nw=2, noise_type='general')


@pytest.fixture
def autonomous_sde_system():
    """Autonomous SDE system with nu = 0."""
    return MockAutonomousSDESystem(nx=2, nw=2, noise_type='general')


@pytest.fixture
def pure_diffusion_system():
    """Pure diffusion process (zero drift)."""
    return MockPureDiffusionSystem(nx=2, nw=2, noise_type='additive')


@pytest.fixture
def additive_noise_system():
    """System with additive noise."""
    return MockSDESystem(nx=2, nu=1, nw=2, noise_type='additive')


@pytest.fixture
def diagonal_noise_system():
    """System with diagonal noise."""
    return MockSDESystem(nx=2, nu=1, nw=2, noise_type='diagonal')


# ============================================================================
# Test Backend Defaults and Method Routing
# ============================================================================

class TestBackendDefaults:
    """Test default method selection for each backend."""
    
    def test_numpy_default(self, controlled_sde_system):
        """Test NumPy backend defaults to Julia EM."""
        mock_class = Mock(return_value=MockSDEIntegrator(controlled_sde_system))
        
        with patch('src.systems.base.numerical_integration.stochastic.sde_integrator_factory.DiffEqPySDEIntegrator', mock_class):
            integrator = SDEIntegratorFactory.create(
                controlled_sde_system,
                backend='numpy'
            )
            
            # Check that DiffEqPy was called with EM algorithm
            mock_class.assert_called_once()
            call_kwargs = mock_class.call_args[1]
            assert call_kwargs['algorithm'] == 'EM'
            assert call_kwargs['backend'] == 'numpy'
    
    def test_torch_default(self, controlled_sde_system):
        """Test PyTorch backend defaults to euler."""
        mock_class = Mock(return_value=MockSDEIntegrator(controlled_sde_system))
        
        with patch('src.systems.base.numerical_integration.stochastic.sde_integrator_factory.TorchSDEIntegrator', mock_class):
            integrator = SDEIntegratorFactory.create(
                controlled_sde_system,
                backend='torch'
            )
            
            mock_class.assert_called_once()
            call_kwargs = mock_class.call_args[1]
            assert call_kwargs['method'] == 'euler'
            assert call_kwargs['backend'] == 'torch'
    
    def test_jax_default(self, controlled_sde_system):
        """Test JAX backend defaults to Euler."""
        mock_class = Mock(return_value=MockSDEIntegrator(controlled_sde_system))
        
        with patch('src.systems.base.numerical_integration.stochastic.sde_integrator_factory.DiffraxSDEIntegrator', mock_class):
            integrator = SDEIntegratorFactory.create(
                controlled_sde_system,
                backend='jax'
            )
            
            mock_class.assert_called_once()
            call_kwargs = mock_class.call_args[1]
            assert call_kwargs['solver'] == 'Euler'
            assert call_kwargs['backend'] == 'jax'


class TestMethodRouting:
    """Test that methods are routed to correct backends."""
    
    def test_julia_method_routes_to_numpy(self, controlled_sde_system):
        """Julia methods should route to NumPy backend."""
        mock_class = Mock(return_value=MockSDEIntegrator(controlled_sde_system))
        
        with patch('src.systems.base.numerical_integration.stochastic.sde_integrator_factory.DiffEqPySDEIntegrator', mock_class):
            integrator = SDEIntegratorFactory.create(
                controlled_sde_system,
                backend='numpy',
                method='SRIW1'
            )
            
            call_kwargs = mock_class.call_args[1]
            assert call_kwargs['algorithm'] == 'SRIW1'
    
    def test_torchsde_method_routes_to_torch(self, controlled_sde_system):
        """TorchSDE methods should route to PyTorch backend."""
        mock_class = Mock(return_value=MockSDEIntegrator(controlled_sde_system))
        
        with patch('src.systems.base.numerical_integration.stochastic.sde_integrator_factory.TorchSDEIntegrator', mock_class):
            integrator = SDEIntegratorFactory.create(
                controlled_sde_system,
                backend='torch',
                method='milstein'
            )
            
            call_kwargs = mock_class.call_args[1]
            assert call_kwargs['method'] == 'milstein'
    
    def test_diffrax_method_routes_to_jax(self, controlled_sde_system):
        """Diffrax methods should route to JAX backend."""
        mock_class = Mock(return_value=MockSDEIntegrator(controlled_sde_system))
        
        with patch('src.systems.base.numerical_integration.stochastic.sde_integrator_factory.DiffraxSDEIntegrator', mock_class):
            integrator = SDEIntegratorFactory.create(
                controlled_sde_system,
                backend='jax',
                method='SEA'
            )
            
            call_kwargs = mock_class.call_args[1]
            assert call_kwargs['solver'] == 'SEA'
    
    def test_wrong_backend_raises_error(self, controlled_sde_system):
        """Using method with wrong backend should raise error."""
        with pytest.raises(ValueError, match="requires backend"):
            SDEIntegratorFactory.create(
                controlled_sde_system,
                backend='torch',
                method='SRIW1'  # Julia method
            )
    
    def test_invalid_backend_raises_error(self, controlled_sde_system):
        """Invalid backend should raise error."""
        with pytest.raises(ValueError, match="Invalid backend"):
            SDEIntegratorFactory.create(
                controlled_sde_system,
                backend='invalid'
            )


# ============================================================================
# Test Controlled System Integration
# ============================================================================

class TestControlledSystemIntegration:
    """Test integration with controlled systems (nu > 0)."""
    
    def test_controlled_system_creation(self, controlled_sde_system):
        """Test factory creates integrator for controlled system."""
        mock_class = Mock(return_value=MockSDEIntegrator(controlled_sde_system))
        
        with patch('src.systems.base.numerical_integration.stochastic.sde_integrator_factory.DiffEqPySDEIntegrator', mock_class):
            integrator = SDEIntegratorFactory.create(
                controlled_sde_system,
                backend='numpy',
                method='EM',
                dt=0.01
            )
            
            assert isinstance(integrator, SDEIntegratorBase)
            assert controlled_sde_system.nu > 0  # Verify it's controlled
    
    def test_controlled_integration_with_feedback(self, controlled_sde_system):
        """Test integration with state feedback control."""
        integrator = MockSDEIntegrator(controlled_sde_system, backend='numpy')
        
        with patch('src.systems.base.numerical_integration.stochastic.sde_integrator_factory.DiffEqPySDEIntegrator', return_value=integrator):
            integrator = SDEIntegratorFactory.create(
                controlled_sde_system,
                backend='numpy',
                method='EM'
            )
            
            # State feedback controller
            K = np.array([[0.5, 0.5]])
            u_func = lambda t, x: -K @ x
            
            x0 = np.array([1.0, 0.5])
            result = integrator.integrate(
                x0=x0,
                u_func=u_func,
                t_span=(0.0, 1.0)
            )
            
            assert result.success
            assert len(result.t) > 1
            assert result.x.shape[0] == len(result.t)
            assert result.x.shape[1] == controlled_sde_system.nx
    
    def test_controlled_integration_constant_control(self, controlled_sde_system):
        """Test integration with constant control."""
        integrator = MockSDEIntegrator(controlled_sde_system, backend='numpy')
        
        with patch('src.systems.base.numerical_integration.stochastic.sde_integrator_factory.DiffEqPySDEIntegrator', return_value=integrator):
            integrator = SDEIntegratorFactory.create(
                controlled_sde_system,
                backend='numpy'
            )
            
            # Constant control
            u_const = np.array([0.1])
            u_func = lambda t, x: u_const
            
            x0 = np.array([1.0, 0.5])
            result = integrator.integrate(
                x0=x0,
                u_func=u_func,
                t_span=(0.0, 1.0)
            )
            
            assert result.success
            assert result.x.shape[1] == controlled_sde_system.nx


# ============================================================================
# Test Autonomous System Integration
# ============================================================================

class TestAutonomousSystemIntegration:
    """Test integration with autonomous systems (nu = 0)."""
    
    def test_autonomous_system_creation(self, autonomous_sde_system):
        """Test factory creates integrator for autonomous system."""
        mock_class = Mock(return_value=MockSDEIntegrator(autonomous_sde_system))
        
        with patch('src.systems.base.numerical_integration.stochastic.sde_integrator_factory.DiffEqPySDEIntegrator', mock_class):
            integrator = SDEIntegratorFactory.create(
                autonomous_sde_system,
                backend='numpy',
                method='EM'
            )
            
            assert isinstance(integrator, SDEIntegratorBase)
            assert autonomous_sde_system.nu == 0  # Verify it's autonomous
    
    def test_autonomous_integration_with_none_control(self, autonomous_sde_system):
        """Test autonomous integration with u=None."""
        integrator = MockSDEIntegrator(autonomous_sde_system, backend='numpy')
        
        with patch('src.systems.base.numerical_integration.stochastic.sde_integrator_factory.DiffEqPySDEIntegrator', return_value=integrator):
            integrator = SDEIntegratorFactory.create(
                autonomous_sde_system,
                backend='numpy'
            )
            
            # Control function returns None
            u_func = lambda t, x: None
            
            x0 = np.array([1.0, 0.5])
            result = integrator.integrate(
                x0=x0,
                u_func=u_func,
                t_span=(0.0, 1.0)
            )
            
            assert result.success
            assert len(result.t) > 1
            assert result.x.shape[1] == autonomous_sde_system.nx
    
    def test_autonomous_with_torch_backend(self, autonomous_sde_system):
        """Test autonomous system with PyTorch backend."""
        integrator = MockSDEIntegrator(autonomous_sde_system, backend='torch')
        
        with patch('src.systems.base.numerical_integration.stochastic.sde_integrator_factory.TorchSDEIntegrator', return_value=integrator):
            integrator = SDEIntegratorFactory.create(
                autonomous_sde_system,
                backend='torch',
                method='euler'
            )
            
            try:
                import torch
                x0 = torch.tensor([1.0, 0.5])
                u_func = lambda t, x: None
                
                result = integrator.integrate(
                    x0=x0,
                    u_func=u_func,
                    t_span=(0.0, 1.0)
                )
                
                assert result.success
            except ImportError:
                pytest.skip("PyTorch not available")
    
    def test_autonomous_with_jax_backend(self, autonomous_sde_system):
        """Test autonomous system with JAX backend."""
        integrator = MockSDEIntegrator(autonomous_sde_system, backend='jax')
        
        with patch('src.systems.base.numerical_integration.stochastic.sde_integrator_factory.DiffraxSDEIntegrator', return_value=integrator):
            integrator = SDEIntegratorFactory.create(
                autonomous_sde_system,
                backend='jax',
                method='Euler'
            )
            
            try:
                import jax.numpy as jnp
                x0 = jnp.array([1.0, 0.5])
                u_func = lambda t, x: None
                
                result = integrator.integrate(
                    x0=x0,
                    u_func=u_func,
                    t_span=(0.0, 1.0)
                )
                
                assert result.success
            except ImportError:
                pytest.skip("JAX not available")


# ============================================================================
# Test Pure Diffusion Processes
# ============================================================================

class TestPureDiffusionProcesses:
    """Test integration of pure diffusion processes (zero drift)."""
    
    def test_pure_diffusion_creation(self, pure_diffusion_system):
        """Test factory creates integrator for pure diffusion."""
        mock_class = Mock(return_value=MockSDEIntegrator(pure_diffusion_system))
        
        with patch('src.systems.base.numerical_integration.stochastic.sde_integrator_factory.DiffEqPySDEIntegrator', mock_class):
            integrator = SDEIntegratorFactory.create(
                pure_diffusion_system,
                backend='numpy',
                method='EM'
            )
            
            assert isinstance(integrator, SDEIntegratorBase)
    
    def test_pure_diffusion_integration(self, pure_diffusion_system):
        """Test integration of pure diffusion (Brownian motion)."""
        integrator = MockSDEIntegrator(pure_diffusion_system, backend='numpy')
        
        with patch('src.systems.base.numerical_integration.stochastic.sde_integrator_factory.DiffEqPySDEIntegrator', return_value=integrator):
            integrator = SDEIntegratorFactory.create(
                pure_diffusion_system,
                backend='numpy'
            )
            
            x0 = np.array([0.0, 0.0])
            u_func = lambda t, x: None
            
            result = integrator.integrate(
                x0=x0,
                u_func=u_func,
                t_span=(0.0, 1.0)
            )
            
            assert result.success
    
    def test_brownian_motion_with_additive_noise(self, pure_diffusion_system):
        """Test Brownian motion (pure diffusion with additive noise)."""
        # Ensure additive noise
        pure_diffusion_system._noise_type = 'additive'
        
        integrator = MockSDEIntegrator(pure_diffusion_system, backend='numpy')
        
        with patch('src.systems.base.numerical_integration.stochastic.sde_integrator_factory.DiffEqPySDEIntegrator', return_value=integrator):
            integrator = SDEIntegratorFactory.create(
                pure_diffusion_system,
                backend='numpy',
                method='EM'
            )
            
            x0 = np.array([0.0, 0.0])
            u_func = lambda t, x: None
            
            result = integrator.integrate(
                x0=x0,
                u_func=u_func,
                t_span=(0.0, 1.0)
            )
            
            assert result.success
            assert pure_diffusion_system.is_additive_noise()


# ============================================================================
# Test Use-Case Helpers
# ============================================================================

class TestUseCaseHelpers:
    """Test use-case specific factory methods."""
    
    def test_for_production_defaults(self, controlled_sde_system):
        """Test for_production() with default settings."""
        mock_class = Mock(return_value=MockSDEIntegrator(controlled_sde_system))
        
        with patch('src.systems.base.numerical_integration.stochastic.sde_integrator_factory.DiffEqPySDEIntegrator', mock_class):
            integrator = SDEIntegratorFactory.for_production(controlled_sde_system)
            
            mock_class.assert_called_once()
            assert mock_class.call_args[1]['backend'] == 'numpy'
    
    def test_for_production_additive_noise(self, additive_noise_system):
        """Test for_production() optimizes for additive noise."""
        mock_class = Mock(return_value=MockSDEIntegrator(additive_noise_system))
        
        with patch('src.systems.base.numerical_integration.stochastic.sde_integrator_factory.DiffEqPySDEIntegrator', mock_class):
            integrator = SDEIntegratorFactory.for_production(
                additive_noise_system,
                noise_type='additive'
            )
            
            # Should use SRA3 for additive noise
            call_kwargs = mock_class.call_args[1]
            assert call_kwargs['algorithm'] == 'SRA3'
    
    def test_for_production_diagonal_noise(self, diagonal_noise_system):
        """Test for_production() optimizes for diagonal noise."""
        mock_class = Mock(return_value=MockSDEIntegrator(diagonal_noise_system))
        
        with patch('src.systems.base.numerical_integration.stochastic.sde_integrator_factory.DiffEqPySDEIntegrator', mock_class):
            integrator = SDEIntegratorFactory.for_production(
                diagonal_noise_system,
                noise_type='diagonal'
            )
            
            # Should use SRIW1 for diagonal noise
            call_kwargs = mock_class.call_args[1]
            assert call_kwargs['algorithm'] == 'SRIW1'
    
    def test_for_julia(self, controlled_sde_system):
        """Test for_julia() helper."""
        mock_class = Mock(return_value=MockSDEIntegrator(controlled_sde_system))
        
        with patch('src.systems.base.numerical_integration.stochastic.sde_integrator_factory.DiffEqPySDEIntegrator', mock_class):
            integrator = SDEIntegratorFactory.for_julia(
                controlled_sde_system,
                algorithm='SRIW1'
            )
            
            call_kwargs = mock_class.call_args[1]
            assert call_kwargs['algorithm'] == 'SRIW1'
            assert call_kwargs['backend'] == 'numpy'
    
    def test_for_optimization_jax(self, controlled_sde_system):
        """Test for_optimization() with JAX."""
        mock_class = Mock(return_value=MockSDEIntegrator(controlled_sde_system))
        
        with patch('src.systems.base.numerical_integration.stochastic.sde_integrator_factory.DiffraxSDEIntegrator', mock_class):
            with patch('src.systems.base.numerical_integration.stochastic.sde_integrator_factory.jax'):
                integrator = SDEIntegratorFactory.for_optimization(
                    controlled_sde_system,
                    prefer_backend='jax'
                )
                
                call_kwargs = mock_class.call_args[1]
                assert call_kwargs['backend'] == 'jax'
    
    def test_for_neural_sde(self, controlled_sde_system):
        """Test for_neural_sde() helper."""
        mock_class = Mock(return_value=MockSDEIntegrator(controlled_sde_system))
        
        with patch('src.systems.base.numerical_integration.stochastic.sde_integrator_factory.TorchSDEIntegrator', mock_class):
            integrator = SDEIntegratorFactory.for_neural_sde(controlled_sde_system)
            
            call_kwargs = mock_class.call_args[1]
            assert call_kwargs['backend'] == 'torch'
            assert call_kwargs['method'] == 'euler'
            assert call_kwargs['adjoint'] == True  # Memory-efficient
    
    def test_for_monte_carlo(self, controlled_sde_system):
        """Test for_monte_carlo() helper."""
        mock_class = Mock(return_value=MockSDEIntegrator(controlled_sde_system))
        
        with patch('src.systems.base.numerical_integration.stochastic.sde_integrator_factory.DiffEqPySDEIntegrator', mock_class):
            integrator = SDEIntegratorFactory.for_monte_carlo(controlled_sde_system)
            
            call_kwargs = mock_class.call_args[1]
            assert call_kwargs['convergence_type'] == ConvergenceType.WEAK
    
    def test_for_simple_simulation(self, controlled_sde_system):
        """Test for_simple_simulation() helper."""
        mock_class = Mock(return_value=MockSDEIntegrator(controlled_sde_system))
        
        with patch('src.systems.base.numerical_integration.stochastic.sde_integrator_factory.DiffEqPySDEIntegrator', mock_class):
            integrator = SDEIntegratorFactory.for_simple_simulation(
                controlled_sde_system,
                dt=0.01,
                seed=42
            )
            
            call_kwargs = mock_class.call_args[1]
            assert call_kwargs['algorithm'] == 'EM'
            assert call_kwargs['dt'] == 0.01
            assert call_kwargs['seed'] == 42


# ============================================================================
# Test Auto Selection
# ============================================================================

class TestAutoSelection:
    """Test automatic integrator selection."""
    
    def test_auto_prefers_jax(self, controlled_sde_system):
        """Test auto() prefers JAX when available."""
        mock_class = Mock(return_value=MockSDEIntegrator(controlled_sde_system))
        
        with patch('src.systems.base.numerical_integration.stochastic.sde_integrator_factory.DiffraxSDEIntegrator', mock_class):
            with patch('src.systems.base.numerical_integration.stochastic.sde_integrator_factory.jax'):
                integrator = SDEIntegratorFactory.auto(controlled_sde_system)
                
                # Should prefer JAX
                assert mock_class.called
    
    def test_auto_falls_back_to_torch(self, controlled_sde_system):
        """Test auto() falls back to PyTorch."""
        mock_class = Mock(return_value=MockSDEIntegrator(controlled_sde_system))
        
        # Mock JAX as unavailable but torch available
        with patch('src.systems.base.numerical_integration.stochastic.sde_integrator_factory.TorchSDEIntegrator', mock_class):
            # Make jax import fail
            import sys
            jax_module = sys.modules.get('jax')
            if 'jax' in sys.modules:
                del sys.modules['jax']
            
            try:
                with patch('builtins.__import__', side_effect=lambda name, *args, **kwargs: 
                          __import__(name, *args, **kwargs) if name != 'jax' else (_ for _ in ()).throw(ImportError())):
                    with patch('src.systems.base.numerical_integration.stochastic.sde_integrator_factory.torch'):
                        integrator = SDEIntegratorFactory.auto(controlled_sde_system)
                        assert mock_class.called
            finally:
                if jax_module:
                    sys.modules['jax'] = jax_module
    
    def test_auto_falls_back_to_numpy(self, controlled_sde_system):
        """Test auto() falls back to NumPy/Julia."""
        mock_class = Mock(return_value=MockSDEIntegrator(controlled_sde_system))
        
        with patch('src.systems.base.numerical_integration.stochastic.sde_integrator_factory.DiffEqPySDEIntegrator', mock_class):
            # Make both jax and torch unavailable
            import sys
            jax_module = sys.modules.get('jax')
            torch_module = sys.modules.get('torch')
            
            if 'jax' in sys.modules:
                del sys.modules['jax']
            if 'torch' in sys.modules:
                del sys.modules['torch']
            
            try:
                def import_mock(name, *args, **kwargs):
                    if name in ['jax', 'torch']:
                        raise ImportError()
                    return __import__(name, *args, **kwargs)
                
                with patch('builtins.__import__', side_effect=import_mock):
                    integrator = SDEIntegratorFactory.auto(controlled_sde_system)
                    assert mock_class.called
                    assert mock_class.call_args[1]['backend'] == 'numpy'
            finally:
                if jax_module:
                    sys.modules['jax'] = jax_module
                if torch_module:
                    sys.modules['torch'] = torch_module


# ============================================================================
# Test Utility Functions
# ============================================================================

class TestUtilityFunctions:
    """Test utility functions and information methods."""
    
    def test_list_methods_all_backends(self):
        """Test list_methods() returns all backends."""
        methods = SDEIntegratorFactory.list_methods()
        
        assert 'numpy' in methods
        assert 'torch' in methods
        assert 'jax' in methods
        assert isinstance(methods['numpy'], list)
        assert len(methods['numpy']) > 0
    
    def test_list_methods_single_backend(self):
        """Test list_methods() for single backend."""
        methods = SDEIntegratorFactory.list_methods('numpy')
        
        assert 'numpy' in methods
        assert 'torch' not in methods
        assert 'jax' not in methods
    
    def test_recommend_production(self):
        """Test recommend() for production use case."""
        rec = SDEIntegratorFactory.recommend('production')
        
        assert rec['backend'] == 'numpy'
        assert 'method' in rec
        assert rec['convergence_type'] == ConvergenceType.STRONG
    
    def test_recommend_optimization(self):
        """Test recommend() for optimization use case."""
        rec = SDEIntegratorFactory.recommend('optimization', has_jax=True)
        
        assert rec['backend'] == 'jax'
        assert rec['convergence_type'] == ConvergenceType.STRONG
    
    def test_recommend_neural_sde(self):
        """Test recommend() for neural SDE use case."""
        rec = SDEIntegratorFactory.recommend('neural_sde')
        
        assert rec['backend'] == 'torch'
        assert rec['adjoint'] == True
    
    def test_recommend_monte_carlo(self):
        """Test recommend() for Monte Carlo use case."""
        rec = SDEIntegratorFactory.recommend('monte_carlo')
        
        assert rec['convergence_type'] == ConvergenceType.WEAK
    
    def test_recommend_invalid_use_case(self):
        """Test recommend() with invalid use case."""
        with pytest.raises(ValueError, match="Unknown use case"):
            SDEIntegratorFactory.recommend('invalid_use_case')
    
    def test_get_info_julia_method(self):
        """Test get_info() for Julia method."""
        # Mock the import and the method
        with patch('src.systems.base.numerical_integration.stochastic.sde_integrator_factory.DiffEqPySDEIntegrator') as mock_module:
            mock_module.get_algorithm_info = Mock(return_value={
                'name': 'SRIW1',
                'description': 'High accuracy'
            })
            
            info = SDEIntegratorFactory.get_info('numpy', 'SRIW1')
            
            assert 'name' in info
            assert 'description' in info


# ============================================================================
# Test Convenience Functions
# ============================================================================

class TestConvenienceFunctions:
    """Test convenience wrapper functions."""
    
    def test_create_sde_integrator(self, controlled_sde_system):
        """Test create_sde_integrator() convenience function."""
        mock_class = Mock(return_value=MockSDEIntegrator(controlled_sde_system))
        
        with patch('src.systems.base.numerical_integration.stochastic.sde_integrator_factory.DiffEqPySDEIntegrator', mock_class):
            integrator = create_sde_integrator(controlled_sde_system)
            
            assert isinstance(integrator, SDEIntegratorBase)
            mock_class.assert_called_once()
    
    def test_auto_sde_integrator(self, controlled_sde_system):
        """Test auto_sde_integrator() convenience function."""
        mock_class = Mock(return_value=MockSDEIntegrator(controlled_sde_system))
        
        with patch('src.systems.base.numerical_integration.stochastic.sde_integrator_factory.DiffEqPySDEIntegrator', mock_class):
            # Make jax and torch unavailable for predictable behavior
            import sys
            jax_module = sys.modules.get('jax')
            torch_module = sys.modules.get('torch')
            
            if 'jax' in sys.modules:
                del sys.modules['jax']
            if 'torch' in sys.modules:
                del sys.modules['torch']
            
            try:
                def import_mock(name, *args, **kwargs):
                    if name in ['jax', 'torch']:
                        raise ImportError()
                    return __import__(name, *args, **kwargs)
                
                with patch('builtins.__import__', side_effect=import_mock):
                    integrator = auto_sde_integrator(controlled_sde_system, seed=42)
                    assert isinstance(integrator, SDEIntegratorBase)
            finally:
                if jax_module:
                    sys.modules['jax'] = jax_module
                if torch_module:
                    sys.modules['torch'] = torch_module


# ============================================================================
# Test Options Propagation
# ============================================================================

class TestOptionsPropagation:
    """Test that options are correctly propagated to integrators."""
    
    def test_dt_propagation(self, controlled_sde_system):
        """Test dt option is propagated."""
        mock_class = Mock(return_value=MockSDEIntegrator(controlled_sde_system))
        
        with patch('src.systems.base.numerical_integration.stochastic.sde_integrator_factory.DiffEqPySDEIntegrator', mock_class):
            integrator = SDEIntegratorFactory.create(
                controlled_sde_system,
                backend='numpy',
                dt=0.001
            )
            
            call_kwargs = mock_class.call_args[1]
            assert call_kwargs['dt'] == 0.001
    
    def test_seed_propagation(self, controlled_sde_system):
        """Test seed option is propagated."""
        mock_class = Mock(return_value=MockSDEIntegrator(controlled_sde_system))
        
        with patch('src.systems.base.numerical_integration.stochastic.sde_integrator_factory.DiffEqPySDEIntegrator', mock_class):
            integrator = SDEIntegratorFactory.create(
                controlled_sde_system,
                backend='numpy',
                seed=12345
            )
            
            call_kwargs = mock_class.call_args[1]
            assert call_kwargs['seed'] == 12345
    
    def test_convergence_type_propagation(self, controlled_sde_system):
        """Test convergence_type option is propagated."""
        mock_class = Mock(return_value=MockSDEIntegrator(controlled_sde_system))
        
        with patch('src.systems.base.numerical_integration.stochastic.sde_integrator_factory.DiffEqPySDEIntegrator', mock_class):
            integrator = SDEIntegratorFactory.create(
                controlled_sde_system,
                backend='numpy',
                convergence_type=ConvergenceType.WEAK
            )
            
            call_kwargs = mock_class.call_args[1]
            assert call_kwargs['convergence_type'] == ConvergenceType.WEAK
    
    def test_custom_options_propagation(self, controlled_sde_system):
        """Test custom options are propagated."""
        mock_class = Mock(return_value=MockSDEIntegrator(controlled_sde_system))
        
        with patch('src.systems.base.numerical_integration.stochastic.sde_integrator_factory.DiffEqPySDEIntegrator', mock_class):
            integrator = SDEIntegratorFactory.create(
                controlled_sde_system,
                backend='numpy',
                rtol=1e-6,
                atol=1e-8,
                custom_option='value'
            )
            
            call_kwargs = mock_class.call_args[1]
            assert call_kwargs['rtol'] == 1e-6
            assert call_kwargs['atol'] == 1e-8
            assert call_kwargs['custom_option'] == 'value'


# ============================================================================
# Integration Tests (End-to-End)
# ============================================================================

class TestEndToEndIntegration:
    """End-to-end integration tests with real mock integrators."""
    
    def test_controlled_system_full_workflow(self, controlled_sde_system):
        """Test full workflow with controlled system."""
        integrator = MockSDEIntegrator(controlled_sde_system, backend='numpy')
        
        with patch('src.systems.base.numerical_integration.stochastic.sde_integrator_factory.DiffEqPySDEIntegrator', return_value=integrator):
            # Create via factory
            integrator = SDEIntegratorFactory.create(
                controlled_sde_system,
                backend='numpy',
                method='EM',
                dt=0.01,
                seed=42
            )
            
            # Set up control
            K = np.array([[1.0, 0.5]])
            u_func = lambda t, x: -K @ x
            
            # Initial condition
            x0 = np.array([1.0, 0.5])
            
            # Integrate
            result = integrator.integrate(
                x0=x0,
                u_func=u_func,
                t_span=(0.0, 2.0)
            )
            
            # Verify result
            assert result.success
            assert len(result.t) > 10
            assert result.x.shape == (len(result.t), controlled_sde_system.nx)
            assert result.nsteps > 0
            assert result.n_paths == 1
    
    def test_autonomous_system_full_workflow(self, autonomous_sde_system):
        """Test full workflow with autonomous system."""
        integrator = MockSDEIntegrator(autonomous_sde_system, backend='numpy')
        
        with patch('src.systems.base.numerical_integration.stochastic.sde_integrator_factory.DiffEqPySDEIntegrator', return_value=integrator):
            # Create via factory
            integrator = SDEIntegratorFactory.auto(autonomous_sde_system)
            
            # No control
            u_func = lambda t, x: None
            
            # Initial condition
            x0 = np.array([1.0, 0.5])
            
            # Integrate
            result = integrator.integrate(
                x0=x0,
                u_func=u_func,
                t_span=(0.0, 2.0)
            )
            
            # Verify result
            assert result.success
            assert len(result.t) > 10
            assert result.x.shape == (len(result.t), autonomous_sde_system.nx)
    
    def test_pure_diffusion_full_workflow(self, pure_diffusion_system):
        """Test full workflow with pure diffusion process."""
        integrator = MockSDEIntegrator(pure_diffusion_system, backend='numpy')
        
        with patch('src.systems.base.numerical_integration.stochastic.sde_integrator_factory.DiffEqPySDEIntegrator', return_value=integrator):
            # Create via factory
            integrator = SDEIntegratorFactory.for_simple_simulation(
                pure_diffusion_system,
                dt=0.01,
                seed=42
            )
            
            # No control
            u_func = lambda t, x: None
            
            # Initial condition
            x0 = np.array([0.0, 0.0])
            
            # Integrate
            result = integrator.integrate(
                x0=x0,
                u_func=u_func,
                t_span=(0.0, 1.0)
            )
            
            # Verify result
            assert result.success
            assert len(result.t) > 10
            # Pure diffusion should keep state near origin in mock
            # (since drift is zero)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])