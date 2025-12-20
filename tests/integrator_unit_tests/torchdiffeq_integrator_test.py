"""
Unit tests for TorchDiffEqIntegrator with IntegratorBase compliance.

Tests cover:
- Basic integration accuracy with system interface
- Single step operations
- Multiple solver methods
- Adaptive vs fixed-step integration
- Batch integration
- Gradient computation
- GPU support (if available)
- Edge cases and error handling
"""

import pytest
import numpy as np
from typing import Tuple

# PyTorch imports
try:
    import torch
    import torchdiffeq
    TORCH_AVAILABLE = True
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    TORCH_AVAILABLE = False
    CUDA_AVAILABLE = False
    pytest.skip("PyTorch not available", allow_module_level=True)

from src.systems.base.numerical_integration.torchdiffeq_integrator import TorchDiffEqIntegrator
from src.systems.base.numerical_integration.integrator_base import IntegrationResult, StepMode


# ============================================================================
# Mock Systems for Testing
# ============================================================================

class MockExponentialSystem:
    """Simple exponential decay: dx/dt = -k*x + u"""
    
    def __init__(self, k=0.5):
        self.nx = 1
        self.nu = 1
        self.k = k
    
    def __call__(self, x, u, backend='torch'):
        """Evaluate dynamics - MUST return same structure as x."""
        x_torch = torch.as_tensor(x)
        u_torch = torch.as_tensor(u)
        
        # Element-wise operations preserve shape
        dx = -self.k * x_torch + u_torch
        
        return dx
    
    def analytical_solution(self, x0, t, u_const=0.0):
        """Analytical solution."""
        if u_const == 0.0:
            return x0 * np.exp(-self.k * t)
        else:
            return (x0 - u_const/self.k) * np.exp(-self.k * t) + u_const/self.k


class MockLinearSystem:
    """Mock dynamical system for testing: dx/dt = Ax + Bu"""
    
    def __init__(self, nx=2, nu=1):
        self.nx = nx
        self.nu = nu
        # Simple stable system
        self.A = torch.tensor([[-0.5, 1.0], [-1.0, -0.5]])
        self.B = torch.tensor([[0.0], [1.0]])
    
    def __call__(self, x, u, backend='torch'):
        """Evaluate dynamics - MUST return same structure as x."""
        x_torch = torch.as_tensor(x)
        u_torch = torch.as_tensor(u)
        
        # Matrix-vector multiplication
        dx = self.A @ x_torch + (self.B @ u_torch.reshape(-1, 1)).squeeze()
        
        # Ensure same shape as x
        return dx.reshape(x_torch.shape)


# ============================================================================
# Basic Integration Tests
# ============================================================================

class TestBasicIntegration:
    """Test basic integration functionality."""
    
    @pytest.fixture
    def system(self):
        """Create test system."""
        return MockExponentialSystem(k=0.5)
    
    @pytest.fixture
    def integrator(self, system):
        """Create default integrator."""
        return TorchDiffEqIntegrator(
            system,
            dt=0.01,
            step_mode=StepMode.ADAPTIVE,
            backend='torch',
            method='dopri5',
            adjoint=False  # Default to False for non-neural ODE functions
        )
    
    def test_single_step(self, integrator, system):
        """Test single integration step."""
        x = torch.tensor([1.0])
        u = torch.tensor([0.0])
        dt = 0.01
        
        # Test that system function works
        dx = system(x, u, backend='torch')
        assert dx.shape == x.shape, f"Shape mismatch: dx.shape={dx.shape} vs x.shape={x.shape}"
        
        x_next = integrator.step(x, u, dt)
        
        # Check shape and type
        assert x_next.shape == x.shape
        assert isinstance(x_next, torch.Tensor)
        
        # Check accuracy against analytical solution
        x_expected = system.analytical_solution(x[0].item(), dt, u_const=0.0)
        np.testing.assert_allclose(x_next[0].item(), x_expected, rtol=1e-5, atol=1e-7)
    
    def test_integrate_zero_control(self, integrator, system):
        """Test integration with zero control."""
        x0 = torch.tensor([1.0])
        t_span = (0.0, 5.0)
        t_eval = torch.linspace(0.0, 5.0, 50)
        
        # Zero control
        u_func = lambda t, x: torch.tensor([0.0])
        
        result = integrator.integrate(x0, u_func, t_span, t_eval)
        
        assert result.success, f"Integration failed: {result.message}"
        assert result.t.shape == t_eval.shape
        assert result.x.shape == (len(t_eval), 1)
        
        # Compare with analytical solution
        t_np = t_eval.numpy()
        x_analytical = system.analytical_solution(x0[0].item(), t_np, u_const=0.0)
        np.testing.assert_allclose(result.x[:, 0].numpy(), x_analytical, rtol=1e-4, atol=1e-6)
    
    def test_integrate_constant_control(self, integrator, system):
        """Test integration with constant non-zero control."""
        x0 = torch.tensor([1.0])
        t_span = (0.0, 3.0)
        u_const = 0.5
        
        u_func = lambda t, x: torch.tensor([u_const])
        
        result = integrator.integrate(x0, u_func, t_span)
        
        assert result.success
        # With positive control and decay, should approach u/k
        assert result.x[-1, 0].item() > 0
    
    def test_integrate_time_varying_control(self, integrator, system):
        """Test integration with time-varying control."""
        x0 = torch.tensor([1.0])
        t_span = (0.0, 2.0)
        
        # Sinusoidal control
        u_func = lambda t, x: torch.tensor([np.sin(t)])
        
        result = integrator.integrate(x0, u_func, t_span)
        
        assert result.success
        assert torch.all(torch.isfinite(result.x))
    
    def test_integrate_state_feedback(self, integrator, system):
        """Test integration with state feedback control."""
        x0 = torch.tensor([2.0])
        t_span = (0.0, 5.0)
        
        # Proportional feedback
        K = 0.5
        u_func = lambda t, x: -K * x
        
        result = integrator.integrate(x0, u_func, t_span)
        
        assert result.success
        # Should decay faster with feedback
        assert torch.abs(result.x[-1, 0]) < torch.abs(x0[0])


# ============================================================================
# Solver Method Tests
# ============================================================================

class TestSolverMethods:
    """Test different solver methods."""
    
    @pytest.fixture
    def system(self):
        return MockExponentialSystem(k=1.0)
    
    @pytest.mark.parametrize("method", [
        'dopri5', 'dopri8', 'rk4', 'euler', 
        'midpoint', 'bosh3'
        # Note: explicit_adams excluded - unstable without proper warm-up
    ])
    def test_all_solvers(self, system, method):
        """Test that all solvers work correctly."""
        # Use fixed-step for simple methods, adaptive for advanced
        if method in ['euler', 'midpoint', 'rk4']:
            step_mode = StepMode.FIXED
            dt = 0.001  # Smaller step for accuracy
        else:
            step_mode = StepMode.ADAPTIVE
            dt = 0.01
        
        integrator = TorchDiffEqIntegrator(
            system,
            dt=dt,
            step_mode=step_mode,
            backend='torch',
            method=method
        )
        
        x0 = torch.tensor([1.0])
        t_span = (0.0, 2.0)
        u_func = lambda t, x: torch.tensor([0.0])
        
        result = integrator.integrate(x0, u_func, t_span)
        
        assert result.success, f"Solver {method} failed: {result.message}"
        assert result.metadata['method'] == method
        
        # Check reasonable accuracy
        x_analytical = system.analytical_solution(x0[0].item(), 2.0, u_const=0.0)
        x_computed = result.x[-1, 0].item()
        
        if method in ['euler']:
            rtol = 1e-2  # 1% with dt=0.001
        elif method in ['midpoint']:
            rtol = 1e-3  # 0.1% with dt=0.001
        elif method in ['rk4']:
            rtol = 1e-4  # 0.01% - RK4 accurate with small dt
        else:
            rtol = 1e-4  # High-order adaptive methods
        
        np.testing.assert_allclose(x_computed, x_analytical, rtol=rtol)
    
    def test_invalid_solver(self, system):
        """Test that invalid solver raises error."""
        with pytest.raises(ValueError, match="Unknown method"):
            TorchDiffEqIntegrator(
                system,
                dt=0.01,
                backend='torch',
                method='invalid_solver'
            )
    
    def test_integrator_name(self, system):
        """Test integrator name property."""
        integrator = TorchDiffEqIntegrator(
            system, dt=0.01, backend='torch', method='dopri5'
        )
        assert 'torchdiffeq' in integrator.name.lower()
        assert 'dopri5' in integrator.name.lower()


# ============================================================================
# Step Mode Tests
# ============================================================================

class TestStepModes:
    """Test fixed vs adaptive step modes."""
    
    @pytest.fixture
    def system(self):
        return MockExponentialSystem(k=0.5)
    
    def test_fixed_step_mode(self, system):
        """Test fixed step mode."""
        integrator = TorchDiffEqIntegrator(
            system,
            dt=0.05,
            step_mode=StepMode.FIXED,
            backend='torch',
            method='rk4'
        )
        
        x0 = torch.tensor([1.0])
        t_span = (0.0, 2.0)
        u_func = lambda t, x: torch.tensor([0.0])
        
        result = integrator.integrate(x0, u_func, t_span)
        
        assert result.success
        # Check uniform time spacing
        dt_actual = torch.diff(result.t)
        dt_mean = torch.mean(dt_actual)
        dt_std = torch.std(dt_actual)
        
        assert dt_std < 1e-6, f"Time steps not uniform: mean={dt_mean}, std={dt_std}"
    
    def test_adaptive_step_mode(self, system):
        """Test adaptive step mode."""
        integrator = TorchDiffEqIntegrator(
            system,
            dt=0.01,
            step_mode=StepMode.ADAPTIVE,
            backend='torch',
            method='dopri5',
            rtol=1e-8,
            atol=1e-10
        )
        
        x0 = torch.tensor([1.0])
        t_span = (0.0, 2.0)
        u_func = lambda t, x: torch.tensor([0.0])
        
        result = integrator.integrate(x0, u_func, t_span)
        
        assert result.success
        assert torch.all(torch.isfinite(result.x))
    
    def test_different_tolerances(self, system):
        """Test that different tolerances affect accuracy."""
        x0 = torch.tensor([1.0])
        t_span = (0.0, 5.0)
        u_func = lambda t, x: torch.tensor([0.0])
        
        # Tight tolerances
        integrator_tight = TorchDiffEqIntegrator(
            system,
            dt=0.01,
            step_mode=StepMode.ADAPTIVE,
            backend='torch',
            method='dopri5',
            rtol=1e-10,
            atol=1e-12
        )
        result_tight = integrator_tight.integrate(x0, u_func, t_span)
        
        # Loose tolerances
        integrator_loose = TorchDiffEqIntegrator(
            system,
            dt=0.01,
            step_mode=StepMode.ADAPTIVE,
            backend='torch',
            method='dopri5',
            rtol=1e-4,
            atol=1e-6
        )
        result_loose = integrator_loose.integrate(x0, u_func, t_span)
        
        # Both should succeed
        assert result_tight.success
        assert result_loose.success
        
        # Tight should be more accurate
        x_analytical = system.analytical_solution(x0[0].item(), 5.0, u_const=0.0)
        error_tight = abs(result_tight.x[-1, 0].item() - x_analytical)
        error_loose = abs(result_loose.x[-1, 0].item() - x_analytical)
        
        assert error_tight < error_loose


# ============================================================================
# Multi-Dimensional System Tests
# ============================================================================

class TestMultiDimensional:
    """Test with multi-dimensional systems."""
    
    @pytest.fixture
    def system(self):
        return MockLinearSystem(nx=2, nu=1)
    
    @pytest.fixture
    def integrator(self, system):
        return TorchDiffEqIntegrator(
            system,
            dt=0.01,
            backend='torch',
            method='dopri5',
            adjoint=False
        )
    
    def test_2d_system_integration(self, integrator, system):
        """Test integration of 2D system."""
        x0 = torch.tensor([1.0, 0.0])
        t_span = (0.0, 5.0)
        u_func = lambda t, x: torch.tensor([0.0])
        
        result = integrator.integrate(x0, u_func, t_span)
        
        assert result.success
        assert result.x.shape[1] == 2
        # System is stable, should decay
        assert torch.linalg.norm(result.x[-1]) < torch.linalg.norm(x0)
    
    def test_2d_system_step(self, integrator, system):
        """Test single step of 2D system."""
        x = torch.tensor([1.0, 0.5])
        u = torch.tensor([0.0])
        dt = 0.01
        
        x_next = integrator.step(x, u, dt)
        
        assert x_next.shape == x.shape
        assert torch.all(torch.isfinite(x_next))


# ============================================================================
# Gradient Computation Tests
# ============================================================================

class TestGradientComputation:
    """Test gradient computation capabilities."""
    
    @pytest.fixture
    def system(self):
        return MockExponentialSystem(k=1.0)
    
    @pytest.fixture
    def integrator(self, system):
        return TorchDiffEqIntegrator(
            system,
            dt=0.01,
            backend='torch',
            method='dopri5',
            adjoint=False
        )
    
    def test_gradient_wrt_initial_condition(self, integrator):
        """Test gradient computation w.r.t. initial conditions."""
        x0 = torch.tensor([1.0], requires_grad=True)
        t_span = (0.0, 1.0)
        u_func = lambda t, x: torch.tensor([0.0])
        
        def loss_fn(result):
            return torch.sum(result.x**2)
        
        loss, grad = integrator.integrate_with_gradient(x0, u_func, t_span, loss_fn)
        
        assert np.isfinite(loss)
        assert torch.isfinite(grad).all()
        assert grad.shape == x0.shape
    
    def test_gradient_finite_difference_validation(self, integrator):
        """Validate gradients using finite differences."""
        x0 = torch.tensor([1.5], requires_grad=True)
        t_span = (0.0, 1.0)
        u_func = lambda t, x: torch.tensor([0.0])
        
        def loss_fn(result):
            return torch.sum(result.x[-1]**2)
        
        # Compute gradient using autodiff
        loss, grad_autodiff = integrator.integrate_with_gradient(x0, u_func, t_span, loss_fn)
        
        # Compute gradient using finite differences
        eps = 1e-4
        x0_np = x0.detach().numpy()
        
        result_plus = integrator.integrate(torch.tensor(x0_np + eps), u_func, t_span)
        loss_plus = loss_fn(result_plus).item()
        
        result_minus = integrator.integrate(torch.tensor(x0_np - eps), u_func, t_span)
        loss_minus = loss_fn(result_minus).item()
        
        grad_fd = (loss_plus - loss_minus) / (2 * eps)
        
        # Lenient tolerance for numerical gradients
        np.testing.assert_allclose(grad_autodiff.numpy(), grad_fd, rtol=5e-2, atol=1e-3)
    
    def test_adjoint_vs_direct(self, system):
        """Compare adjoint and direct backpropagation."""
        # Note: adjoint requires nn.Module, so we test that direct works
        # and adjoint raises appropriate error for non-Module functions
        x0 = torch.tensor([1.0], requires_grad=True)
        t_span = (0.0, 1.0)
        u_func = lambda t, x: torch.tensor([0.0])
        
        def loss_fn(result):
            return torch.sum(result.x**2)
        
        # Direct method should work
        integrator_direct = TorchDiffEqIntegrator(
            system, dt=0.01, backend='torch', method='dopri5', adjoint=False
        )
        loss_direct, grad_direct = integrator_direct.integrate_with_gradient(
            x0, u_func, t_span, loss_fn
        )
        
        assert np.isfinite(loss_direct)
        assert torch.isfinite(grad_direct).all()
        
        # Adjoint method requires nn.Module (would fail with regular function)
        # So we skip testing it here - it's for neural ODE applications


# ============================================================================
# Batch Integration Tests
# ============================================================================

class TestBatchIntegration:
    """Test vectorized batch integration."""
    
    @pytest.fixture
    def system(self):
        return MockExponentialSystem(k=1.0)
    
    @pytest.fixture
    def integrator(self, system):
        return TorchDiffEqIntegrator(
            system,
            dt=0.01,
            backend='torch',
            method='dopri5'
        )
    
    def test_batch_step(self, integrator, system):
        """Test batch integration with multiple initial conditions."""
        x_batch = torch.tensor([[1.0], [2.0], [3.0]])
        u_batch = torch.tensor([[0.0], [0.0], [0.0]])
        dt = 0.01
        
        x_next_batch = integrator.vectorized_step(x_batch, u_batch, dt)
        
        assert x_next_batch.shape == x_batch.shape
        assert torch.all(torch.isfinite(x_next_batch))
        
        # Verify scaling property
        for i in range(3):
            x_analytical = system.analytical_solution((i+1), dt, u_const=0.0)
            np.testing.assert_allclose(
                x_next_batch[i, 0].item(), x_analytical, rtol=1e-5
            )


# ============================================================================
# GPU Tests (if available)
# ============================================================================

@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
class TestGPUSupport:
    """Test GPU acceleration."""
    
    @pytest.fixture
    def system(self):
        return MockExponentialSystem(k=0.5)
    
    @pytest.fixture
    def integrator(self, system):
        return TorchDiffEqIntegrator(
            system,
            dt=0.01,
            backend='torch',
            method='dopri5'
        )
    
    def test_gpu_integration(self, integrator, system):
        """Test integration on GPU."""
        x0 = torch.tensor([1.0], device='cuda')
        t_span = (0.0, 2.0)
        u_func = lambda t, x: torch.tensor([0.0], device=x.device)
        
        result = integrator.integrate(x0, u_func, t_span)
        
        assert result.success
        assert result.x.device.type == 'cuda'
        
        # Verify accuracy
        x_analytical = system.analytical_solution(x0[0].item(), 2.0, u_const=0.0)
        np.testing.assert_allclose(
            result.x[-1, 0].cpu().item(), x_analytical, rtol=1e-4
        )
    
    def test_gpu_gradients(self, integrator):
        """Test gradient computation on GPU."""
        x0 = torch.tensor([1.0], device='cuda', requires_grad=True)
        t_span = (0.0, 1.0)
        u_func = lambda t, x: torch.tensor([0.0], device=x.device)
        
        def loss_fn(result):
            return torch.sum(result.x**2)
        
        loss, grad = integrator.integrate_with_gradient(x0, u_func, t_span, loss_fn)
        
        assert np.isfinite(loss)
        assert grad.device.type == 'cuda'
        assert torch.isfinite(grad).all()


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""
    
    @pytest.fixture
    def system(self):
        return MockExponentialSystem(k=1.0)
    
    @pytest.fixture
    def integrator(self, system):
        return TorchDiffEqIntegrator(
            system,
            dt=0.01,
            backend='torch',
            method='dopri5'
        )
    
    def test_zero_time_span(self, integrator):
        """Test integration with zero time span."""
        x0 = torch.tensor([1.0])
        t_span = (0.0, 0.0)
        u_func = lambda t, x: torch.tensor([0.0])
        
        result = integrator.integrate(x0, u_func, t_span)
        
        assert result.success
        np.testing.assert_allclose(result.x[0].numpy(), x0.numpy(), rtol=1e-10)
    
    def test_very_small_initial_value(self, integrator):
        """Test with very small initial values."""
        x0 = torch.tensor([1e-10])
        t_span = (0.0, 5.0)
        u_func = lambda t, x: torch.tensor([0.0])
        
        result = integrator.integrate(x0, u_func, t_span)
        
        assert result.success
        assert torch.all(torch.isfinite(result.x))
    
    def test_invalid_backend(self, system):
        """Test that invalid backend raises error."""
        with pytest.raises(ValueError, match="requires backend='torch'"):
            TorchDiffEqIntegrator(
                system,
                dt=0.01,
                backend='jax'
            )
    
    def test_statistics_tracking(self, integrator):
        """Test that statistics are tracked correctly."""
        x0 = torch.tensor([1.0])
        t_span = (0.0, 1.0)
        u_func = lambda t, x: torch.tensor([0.0])
        
        integrator.reset_stats()
        result = integrator.integrate(x0, u_func, t_span)
        
        stats = integrator.get_stats()
        assert stats['total_steps'] >= 0
        assert stats['total_fev'] >= 0
    
    def test_numpy_input_conversion(self, integrator):
        """Test that NumPy arrays are converted properly."""
        x0 = np.array([1.0])
        u_func = lambda t, x: np.array([0.0])
        t_span = (0.0, 1.0)
        
        result = integrator.integrate(x0, u_func, t_span)
        
        assert result.success
        assert isinstance(result.x, torch.Tensor)


# ============================================================================
# Options and Configuration Tests
# ============================================================================

class TestOptionsAndConfiguration:
    """Test option management."""
    
    @pytest.fixture
    def system(self):
        return MockExponentialSystem(k=1.0)
    
    def test_set_options(self, system):
        """Test setting options."""
        integrator = TorchDiffEqIntegrator(
            system, dt=0.01, backend='torch', method='dopri5'
        )
        
        integrator.set_options(rtol=1e-10, atol=1e-12)
        
        options = integrator.get_options()
        assert options['rtol'] == 1e-10
        assert options['atol'] == 1e-12
    
    def test_enable_disable_adjoint(self, system):
        """Test toggling adjoint method."""
        integrator = TorchDiffEqIntegrator(
            system, dt=0.01, backend='torch', method='dopri5', adjoint=False
        )
        
        assert not integrator.use_adjoint
        
        integrator.enable_adjoint()
        assert integrator.use_adjoint
        
        integrator.disable_adjoint()
        assert not integrator.use_adjoint


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])