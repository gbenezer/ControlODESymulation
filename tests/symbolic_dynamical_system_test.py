"""
Unit tests for SymbolicDynamicalSystem
Tests all methods with multiple backends (torch, jax, numpy)
"""

import pytest
import numpy as np
import torch
import sympy as sp
from typing import Tuple

# Try to import JAX, mark tests as skipped if not available
try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jnp = None

from src.systems.base.symbolic_dynamical_system import SymbolicDynamicalSystem


# ============================================================================
# Test Fixtures: Concrete System Implementations
# ============================================================================

class SimplePendulum(SymbolicDynamicalSystem):
    """Simple pendulum for testing: θ̈ = -g/l * sin(θ) - b*θ̇ + τ/(m*l²)"""
    
    def __init__(self, m=1.0, l=1.0, g=9.81, b=0.1):
        super().__init__()
        self.define_system(m, l, g, b)
    
    def define_system(self, m, l, g, b):
        # State: [θ, θ̇]
        theta, theta_dot = sp.symbols('theta theta_dot', real=True)
        tau = sp.symbols('tau', real=True)
        
        self.state_vars = [theta, theta_dot]
        self.control_vars = [tau]
        self.output_vars = []
        
        # Dynamics
        self._f_sym = sp.Matrix([
            theta_dot,
            -g/l * sp.sin(theta) - b * theta_dot + tau / (m * l**2)
        ])
        
        # Parameters
        m_sym, l_sym, g_sym, b_sym = sp.symbols('m l g b', real=True, positive=True)
        self.parameters = {m_sym: m, l_sym: l, g_sym: g, b_sym: b}
        
        self.order = 1


class PartialObservationPendulum(SymbolicDynamicalSystem):
    """Pendulum with only angle measured (partial observation)"""
    
    def __init__(self, m=1.0, l=1.0, g=9.81, b=0.1):
        super().__init__()
        self.define_system(m, l, g, b)
    
    def define_system(self, m, l, g, b):
        theta, theta_dot = sp.symbols('theta theta_dot', real=True)
        tau = sp.symbols('tau', real=True)
        
        self.state_vars = [theta, theta_dot]
        self.control_vars = [tau]
        
        # Dynamics
        self._f_sym = sp.Matrix([
            theta_dot,
            -g/l * sp.sin(theta) - b * theta_dot + tau / (m * l**2)
        ])
        
        # Output: only observe angle
        self._h_sym = sp.Matrix([theta])
        self.output_vars = [sp.symbols('y')]
        
        m_sym, l_sym, g_sym, b_sym = sp.symbols('m l g b', real=True, positive=True)
        self.parameters = {m_sym: m, l_sym: l, g_sym: g, b_sym: b}
        
        self.order = 1


class LinearSystem(SymbolicDynamicalSystem):
    """Simple 2D linear system: ẋ = Ax + Bu"""
    
    def __init__(self):
        super().__init__()
        self.define_system()
    
    def define_system(self):
        x1, x2 = sp.symbols('x1 x2', real=True)
        u1 = sp.symbols('u1', real=True)
        
        self.state_vars = [x1, x2]
        self.control_vars = [u1]
        
        # ẋ₁ = -x₁ + x₂
        # ẋ₂ = -2x₂ + u₁
        self._f_sym = sp.Matrix([
            -x1 + x2,
            -2*x2 + u1
        ])
        
        self.parameters = {}
        self.order = 1


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def pendulum():
    """Create a simple pendulum system"""
    return SimplePendulum(m=0.15, l=0.5, g=9.81, b=0.1)


@pytest.fixture
def partial_obs_pendulum():
    """Create a pendulum with partial observation"""
    return PartialObservationPendulum(m=0.15, l=0.5, g=9.81, b=0.1)


@pytest.fixture
def linear_system():
    """Create a linear system"""
    return LinearSystem()


@pytest.fixture
def sample_state_control():
    """Sample state and control for testing"""
    return {
        'x_torch': torch.tensor([0.5, 0.1]),
        'u_torch': torch.tensor([0.2]),
        'x_np': np.array([0.5, 0.1]),
        'u_np': np.array([0.2]),
        'x_jax': jnp.array([0.5, 0.1]) if JAX_AVAILABLE else None,
        'u_jax': jnp.array([0.2]) if JAX_AVAILABLE else None,
    }


# ============================================================================
# Test System Properties
# ============================================================================

class TestSystemProperties:
    """Test basic system properties and configuration"""
    
    def test_dimensions(self, pendulum):
        """Test that system dimensions are correct"""
        assert pendulum.nx == 2
        assert pendulum.nu == 1
        assert pendulum.ny == 2  # Full state observation by default
        assert pendulum.nq == 2
        assert pendulum.order == 1
    
    def test_partial_observation_dimensions(self, partial_obs_pendulum):
        """Test dimensions with partial observation"""
        assert partial_obs_pendulum.nx == 2
        assert partial_obs_pendulum.nu == 1
        assert partial_obs_pendulum.ny == 1  # Only angle observed
    
    def test_equilibrium_points(self, pendulum):
        """Test default equilibrium points"""
        x_eq = pendulum.x_equilibrium
        u_eq = pendulum.u_equilibrium
        
        assert isinstance(x_eq, torch.Tensor)
        assert isinstance(u_eq, torch.Tensor)
        assert x_eq.shape == (2,)
        assert u_eq.shape == (1,)
        assert torch.allclose(x_eq, torch.zeros(2))
        assert torch.allclose(u_eq, torch.zeros(1))
    
    def test_validate_system(self, pendulum):
        """Test system validation"""
        assert pendulum._validate_system()
    
    def test_substitute_parameters(self, pendulum):
        """Test parameter substitution"""
        theta, theta_dot = sp.symbols('theta theta_dot')
        m, l, g, b = sp.symbols('m l g b', positive=True)
        
        expr = m * l * g * theta + b * theta_dot
        
        # Before substitution, expression contains symbols
        assert m in expr.free_symbols
        
        # After substitution, parameters should be replaced
        substituted = pendulum.substitute_parameters(expr)
        assert m not in substituted.free_symbols
    
    def test_repr_and_str(self, pendulum):
        """Test string representations"""
        repr_str = repr(pendulum)
        str_str = str(pendulum)
        
        assert "SimplePendulum" in repr_str
        assert "nx=2" in repr_str
        assert "nu=1" in repr_str
        assert "SimplePendulum" in str_str


# ============================================================================
# Test Function Generation
# ============================================================================

class TestFunctionGeneration:
    """Test generation of numerical functions from symbolic expressions"""
    
    def test_generate_numpy_function(self, pendulum):
        """Test NumPy function generation"""
        func = pendulum.generate_numpy_function()
        assert callable(func)
        assert pendulum._f_numpy is not None
    
    def test_generate_torch_function_lambdify(self, pendulum):
        """Test PyTorch function generation with lambdify"""
        func = pendulum.generate_torch_function(method='lambdify')
        assert callable(func)
        assert pendulum._f_torch is not None
    
    def test_generate_torch_function_codegen(self, pendulum):
        """Test PyTorch function generation with codegen"""
        func = pendulum.generate_torch_function(method='codegen')
        assert callable(func)
        assert pendulum._f_torch is not None
    
    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_generate_jax_function(self, pendulum):
        """Test JAX function generation"""
        func = pendulum.generate_jax_function(jit=True)
        assert callable(func)
        assert pendulum._f_jax is not None
    
    def test_generate_dynamics_function_torch(self, pendulum):
        """Test unified dynamics function generation for torch"""
        func = pendulum.generate_dynamics_function(backend='torch')
        assert callable(func)
        assert pendulum._f_torch is not None
    
    def test_generate_dynamics_function_numpy(self, pendulum):
        """Test unified dynamics function generation for numpy"""
        func = pendulum.generate_dynamics_function(backend='numpy')
        assert callable(func)
        assert pendulum._f_numpy is not None
    
    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_generate_dynamics_function_jax(self, pendulum):
        """Test unified dynamics function generation for JAX"""
        func = pendulum.generate_dynamics_function(backend='jax')
        assert callable(func)
        assert pendulum._f_jax is not None


# ============================================================================
# Test Forward Dynamics
# ============================================================================

class TestForwardDynamics:
    """Test forward dynamics evaluation across backends"""
    
    def test_forward_torch_single(self, pendulum, sample_state_control):
        """Test PyTorch forward dynamics with single state"""
        x = sample_state_control['x_torch']
        u = sample_state_control['u_torch']
        
        dx = pendulum.forward(x, u)
        
        assert isinstance(dx, torch.Tensor)
        assert dx.shape == (2,)
        assert not torch.isnan(dx).any()
        assert not torch.isinf(dx).any()
    
    def test_forward_torch_batched(self, pendulum):
        """Test PyTorch forward dynamics with batched states"""
        x = torch.randn(10, 2)
        u = torch.randn(10, 1)
        
        dx = pendulum.forward(x, u)
        
        assert isinstance(dx, torch.Tensor)
        assert dx.shape == (10, 2)
        assert not torch.isnan(dx).any()
    
    def test_forward_numpy_single(self, pendulum, sample_state_control):
        """Test NumPy forward dynamics with single state"""
        x = sample_state_control['x_np']
        u = sample_state_control['u_np']
        
        dx = pendulum.forward(x, u)
        
        assert isinstance(dx, np.ndarray)
        assert dx.shape == (2,)
        assert not np.isnan(dx).any()
    
    def test_forward_numpy_batched(self, pendulum):
        """Test NumPy forward dynamics with batched states"""
        x = np.random.randn(10, 2)
        u = np.random.randn(10, 1)
        
        dx = pendulum.forward(x, u)
        
        assert isinstance(dx, np.ndarray)
        assert dx.shape == (10, 2)
    
    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_forward_jax_single(self, pendulum, sample_state_control):
        """Test JAX forward dynamics with single state"""
        x = sample_state_control['x_jax']
        u = sample_state_control['u_jax']
        
        dx = pendulum.forward(x, u)
        
        assert isinstance(dx, jnp.ndarray)
        assert dx.shape == (2,)
        assert not jnp.isnan(dx).any()
    
    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_forward_jax_batched(self, pendulum):
        """Test JAX forward dynamics with batched states"""
        x = jnp.array(np.random.randn(10, 2))
        u = jnp.array(np.random.randn(10, 1))
        
        dx = pendulum.forward(x, u)
        
        assert isinstance(dx, jnp.ndarray)
        assert dx.shape == (10, 2)
    
    def test_forward_cross_backend_consistency(self, pendulum, sample_state_control):
        """Test that all backends give consistent results"""
        x_torch = sample_state_control['x_torch']
        u_torch = sample_state_control['u_torch']
        x_np = sample_state_control['x_np']
        u_np = sample_state_control['u_np']
        
        dx_torch = pendulum.forward(x_torch, u_torch)
        dx_np = pendulum.forward(x_np, u_np)
        
        # Convert to numpy for comparison
        dx_torch_np = dx_torch.detach().numpy()
        
        assert np.allclose(dx_torch_np, dx_np, rtol=1e-5, atol=1e-6)
    
    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_forward_jax_torch_consistency(self, pendulum, sample_state_control):
        """Test that JAX and PyTorch give consistent results"""
        x_torch = sample_state_control['x_torch']
        u_torch = sample_state_control['u_torch']
        x_jax = sample_state_control['x_jax']
        u_jax = sample_state_control['u_jax']
        
        dx_torch = pendulum.forward(x_torch, u_torch)
        dx_jax = pendulum.forward(x_jax, u_jax)
        
        dx_torch_np = dx_torch.detach().numpy()
        dx_jax_np = np.array(dx_jax)
        
        assert np.allclose(dx_torch_np, dx_jax_np, rtol=1e-5, atol=1e-6)
    
    def test_forward_invalid_dimensions(self, pendulum):
        """Test that forward raises error with invalid dimensions"""
        x = torch.tensor([0.5])  # Wrong dimension
        u = torch.tensor([0.2])
        
        with pytest.raises(ValueError, match="Expected state dimension"):
            pendulum.forward(x, u)
    
    def test_forward_invalid_type(self, pendulum):
        """Test that forward raises error with invalid input type"""
        x = [0.5, 0.1]  # List instead of tensor/array
        u = [0.2]
        
        with pytest.raises(TypeError, match="Unsupported input type"):
            pendulum.forward(x, u)


# ============================================================================
# Test Linearization
# ============================================================================

class TestLinearization:
    """Test linearization methods"""
    
    def test_cache_jacobians_torch(self, pendulum):
        """Test Jacobian caching for PyTorch"""
        pendulum._cache_jacobians(backend='torch')
        
        assert pendulum._A_sym_cached is not None
        assert pendulum._B_sym_cached is not None
        assert hasattr(pendulum, '_A_torch_fn')
        assert hasattr(pendulum, '_B_torch_fn')
    
    def test_cache_jacobians_numpy(self, pendulum):
        """Test Jacobian caching for NumPy"""
        pendulum._cache_jacobians(backend='numpy')
        
        assert pendulum._A_sym_cached is not None
        assert pendulum._B_sym_cached is not None
    
    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_cache_jacobians_jax(self, pendulum):
        """Test Jacobian caching for JAX"""
        pendulum._cache_jacobians(backend='jax')
        
        assert pendulum._A_sym_cached is not None
        assert pendulum._B_sym_cached is not None
        assert hasattr(pendulum, '_A_jax_fn')
        assert hasattr(pendulum, '_B_jax_fn')
    
    def test_linearized_dynamics_symbolic(self, pendulum):
        """Test symbolic linearization"""
        x_eq = sp.Matrix([0, 0])
        u_eq = sp.Matrix([0])
        
        A, B = pendulum.linearized_dynamics_symbolic(x_eq, u_eq)
        
        assert isinstance(A, sp.Matrix)
        assert isinstance(B, sp.Matrix)
        assert A.shape == (2, 2)
        assert B.shape == (2, 1)
    
    def test_linearized_dynamics_torch(self, pendulum, sample_state_control):
        """Test numerical linearization with PyTorch"""
        x = sample_state_control['x_torch']
        u = sample_state_control['u_torch']
        
        A, B = pendulum.linearized_dynamics(x, u)
        
        assert isinstance(A, torch.Tensor)
        assert isinstance(B, torch.Tensor)
        assert A.shape == (2, 2)
        assert B.shape == (2, 1)
        assert not torch.isnan(A).any()
        assert not torch.isnan(B).any()
    
    def test_linearized_dynamics_numpy(self, pendulum, sample_state_control):
        """Test numerical linearization with NumPy"""
        x = sample_state_control['x_np']
        u = sample_state_control['u_np']
        
        A, B = pendulum.linearized_dynamics(x, u)
        
        assert isinstance(A, np.ndarray)
        assert isinstance(B, np.ndarray)
        assert A.shape == (2, 2)
        assert B.shape == (2, 1)
    
    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_linearized_dynamics_jax(self, pendulum, sample_state_control):
        """Test numerical linearization with JAX"""
        x = sample_state_control['x_jax']
        u = sample_state_control['u_jax']
        
        A, B = pendulum.linearized_dynamics(x, u)
        
        assert isinstance(A, jnp.ndarray)
        assert isinstance(B, jnp.ndarray)
        assert A.shape == (2, 2)
        assert B.shape == (2, 1)
    
    def test_linearized_dynamics_batched(self, pendulum):
        """Test batched linearization"""
        x = torch.randn(5, 2)
        u = torch.randn(5, 1)
        
        A, B = pendulum.linearized_dynamics(x, u)
        
        assert A.shape == (5, 2, 2)
        assert B.shape == (5, 2, 1)
    
    def test_linearized_dynamics_cross_backend(self, pendulum, sample_state_control):
        """Test that linearization is consistent across backends"""
        x_torch = sample_state_control['x_torch']
        u_torch = sample_state_control['u_torch']
        x_np = sample_state_control['x_np']
        u_np = sample_state_control['u_np']
        
        A_torch, B_torch = pendulum.linearized_dynamics(x_torch, u_torch)
        A_np, B_np = pendulum.linearized_dynamics(x_np, u_np)
        
        A_torch_np = A_torch.detach().numpy()
        B_torch_np = B_torch.detach().numpy()
        
        assert np.allclose(A_torch_np, A_np, rtol=1e-4, atol=1e-5)
        assert np.allclose(B_torch_np, B_np, rtol=1e-4, atol=1e-5)


# ============================================================================
# Test Observation
# ============================================================================

class TestObservation:
    """Test observation and output linearization"""
    
    def test_linearized_observation_symbolic_full_state(self, pendulum):
        """Test symbolic observation linearization with full state observation"""
        x_eq = sp.Matrix([0, 0])
        C = pendulum.linearized_observation_symbolic(x_eq)
        
        # Should be identity for full state observation
        assert C.shape == (2, 2)
        assert C == sp.eye(2)
    
    def test_linearized_observation_symbolic_partial(self, partial_obs_pendulum):
        """Test symbolic observation linearization with partial observation"""
        x_eq = sp.Matrix([0, 0])
        C = partial_obs_pendulum.linearized_observation_symbolic(x_eq)
        
        # Should be [1, 0] for angle-only observation
        assert C.shape == (1, 2)
    
    def test_linearized_observation_torch_full_state(self, pendulum, sample_state_control):
        """Test numerical observation linearization (full state)"""
        x = sample_state_control['x_torch']
        C = pendulum.linearized_observation(x)
        
        assert isinstance(C, torch.Tensor)
        assert C.shape == (2, 2)
        # Should be identity
        assert torch.allclose(C, torch.eye(2), atol=1e-6)
    
    def test_linearized_observation_torch_partial(self, partial_obs_pendulum, sample_state_control):
        """Test numerical observation linearization (partial observation)"""
        x = sample_state_control['x_torch']
        C = partial_obs_pendulum.linearized_observation(x)
        
        assert isinstance(C, torch.Tensor)
        assert C.shape == (1, 2)
        # Should be [1, 0]
        expected = torch.tensor([[1.0, 0.0]])
        assert torch.allclose(C, expected, atol=1e-6)
    
    def test_linearized_observation_numpy(self, partial_obs_pendulum, sample_state_control):
        """Test observation linearization with NumPy"""
        x = sample_state_control['x_np']
        C = partial_obs_pendulum.linearized_observation(x)
        
        assert isinstance(C, np.ndarray)
        assert C.shape == (1, 2)
    
    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_linearized_observation_jax(self, partial_obs_pendulum, sample_state_control):
        """Test observation linearization with JAX"""
        x = sample_state_control['x_jax']
        C = partial_obs_pendulum.linearized_observation(x)
        
        assert isinstance(C, jnp.ndarray)
        assert C.shape == (1, 2)
    
    def test_h_full_state_torch(self, pendulum, sample_state_control):
        """Test output function with full state observation"""
        x = sample_state_control['x_torch']
        y = pendulum.h(x)
        
        assert isinstance(y, torch.Tensor)
        assert y.shape == x.shape
        assert torch.allclose(y, x)  # Full state observation
    
    def test_h_partial_state_torch(self, partial_obs_pendulum, sample_state_control):
        """Test output function with partial observation"""
        x = sample_state_control['x_torch']
        y = partial_obs_pendulum.h(x)
        
        assert isinstance(y, torch.Tensor)
        assert y.shape == (1,)
        assert torch.allclose(y, x[0:1])  # Only first element (angle)
    
    def test_h_numpy(self, partial_obs_pendulum, sample_state_control):
        """Test output function with NumPy"""
        x = sample_state_control['x_np']
        y = partial_obs_pendulum.h(x)
        
        assert isinstance(y, np.ndarray)
        # Accept both (1,) and (1, 1) as valid shapes for single output
        assert y.shape in [(1,), (1, 1)] or y.size == 1
        # Check value is correct
        assert np.allclose(y.flatten(), x[0:1])
    
    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_h_jax(self, partial_obs_pendulum, sample_state_control):
        """Test output function with JAX"""
        x = sample_state_control['x_jax']
        y = partial_obs_pendulum.h(x)
        
        assert isinstance(y, jnp.ndarray)
        assert y.shape == (1,)
    
    def test_h_batched(self, partial_obs_pendulum):
        """Test batched output evaluation"""
        x = torch.randn(5, 2)
        y = partial_obs_pendulum.h(x)
        
        assert y.shape == (5, 1)


# ============================================================================
# Test Control Design Methods
# ============================================================================

class TestControlDesign:
    """Test LQR, Kalman gain, and LQG methods"""
    
    def test_lqr_control(self, linear_system):
        """Test LQR control design"""
        Q = np.eye(2)
        R = np.array([[1.0]])
        
        K, S = linear_system.lqr_control(Q, R)
        
        assert isinstance(K, np.ndarray)
        assert isinstance(S, np.ndarray)
        assert K.shape == (1, 2)  # (nu, nx)
        assert S.shape == (2, 2)  # (nx, nx)
        
        # Check that S is positive definite
        eigenvalues = np.linalg.eigvals(S)
        assert np.all(eigenvalues > 0)
    
    def test_lqr_with_custom_equilibrium(self, pendulum):
        """Test LQR with custom equilibrium point"""
        Q = np.diag([10.0, 1.0])
        R = np.array([[0.1]])
        x_eq = torch.tensor([np.pi, 0.0])  # Inverted pendulum
        u_eq = torch.tensor([0.0])
        
        K, S = pendulum.lqr_control(Q, R, x_eq, u_eq)
        
        assert K.shape == (1, 2)
        assert S.shape == (2, 2)
    
    def test_kalman_gain(self, linear_system):
        """Test Kalman gain computation"""
        Q_process = np.eye(2) * 0.01
        R_measurement = np.eye(2) * 0.1
        
        L = linear_system.kalman_gain(Q_process, R_measurement)
        
        assert isinstance(L, np.ndarray)
        assert L.shape == (2, 2)  # (nx, ny)
    
    def test_kalman_gain_partial_observation(self, partial_obs_pendulum):
        """Test Kalman gain with partial observation"""
        Q_process = np.eye(2) * 0.01
        R_measurement = np.array([[0.1]])  # Only one measurement
        
        L = partial_obs_pendulum.kalman_gain(Q_process, R_measurement)
        
        assert isinstance(L, np.ndarray)
        assert L.shape == (2, 1)  # (nx, ny) where ny=1
    
    def test_lqg_control(self, linear_system):
        """Test LQG controller design"""
        Q_lqr = np.eye(2)
        R_lqr = np.array([[1.0]])
        Q_process = np.eye(2) * 0.01
        R_measurement = np.eye(2) * 0.1
        
        K, L = linear_system.lqg_control(Q_lqr, R_lqr, Q_process, R_measurement)
        
        assert isinstance(K, np.ndarray)
        assert isinstance(L, np.ndarray)
        assert K.shape == (1, 2)
        assert L.shape == (2, 2)
    
    def test_lqg_closed_loop_matrix(self, linear_system):
        """Test closed-loop system matrix computation"""
        Q_lqr = np.eye(2)
        R_lqr = np.array([[1.0]])
        Q_process = np.eye(2) * 0.01
        R_measurement = np.eye(2) * 0.1
        
        K, L = linear_system.lqg_control(Q_lqr, R_lqr, Q_process, R_measurement)
        A_cl = linear_system.lqg_closed_loop_matrix(K, L)
        
        assert isinstance(A_cl, np.ndarray)
        assert A_cl.shape == (4, 4)  # (2*nx, 2*nx)
        
        # Check that closed-loop is stable
        eigenvalues = np.linalg.eigvals(A_cl)
        assert np.all(np.real(eigenvalues) < 0), "Closed-loop should be stable"


# ============================================================================
# Test Analysis Methods
# ============================================================================

class TestAnalysisMethods:
    """Test equilibrium and stability analysis methods"""
    
    def test_check_equilibrium_at_origin(self, pendulum):
        """Test equilibrium check at origin"""
        x_eq = torch.tensor([0.0, 0.0])
        u_eq = torch.tensor([0.0])
        
        is_eq, max_deriv = pendulum.check_equilibrium(x_eq, u_eq)
        
        assert is_eq
        assert max_deriv < 1e-6
    
    def test_check_equilibrium_not_at_origin(self, pendulum):
        """Test equilibrium check at non-equilibrium point"""
        x = torch.tensor([0.5, 0.1])
        u = torch.tensor([0.0])
        
        is_eq, max_deriv = pendulum.check_equilibrium(x, u)
        
        assert not is_eq
        assert max_deriv > 1e-6
    
    def test_eigenvalues_at_equilibrium(self, linear_system):
        """Test eigenvalue computation"""
        eigs = linear_system.eigenvalues_at_equilibrium()
        
        assert isinstance(eigs, np.ndarray)
        assert eigs.shape == (2,)
        # Linear system should be stable (eigenvalues have negative real part)
        assert np.all(np.real(eigs) < 0)
    
    def test_is_stable_equilibrium(self, linear_system):
        """Test stability check"""
        is_stable = linear_system.is_stable_equilibrium(discrete_time=False)
        assert is_stable
    
    def test_verify_jacobians(self, pendulum, sample_state_control):
        """Test Jacobian verification against autograd"""
        x = sample_state_control['x_torch']
        u = sample_state_control['u_torch']
        
        result = pendulum.verify_jacobians(x, u, tol=1e-3)
        
        assert isinstance(result, dict)
        assert 'A_match' in result
        assert 'B_match' in result
        assert 'A_error' in result
        assert 'B_error' in result
        
        assert result['A_match']
        assert result['B_match']
        assert result['A_error'] < 1e-3
        assert result['B_error'] < 1e-3
    
    def test_check_numerical_stability(self, pendulum, sample_state_control):
        """Test numerical stability check"""
        x = sample_state_control['x_torch']
        u = sample_state_control['u_torch']
        
        result = pendulum.check_numerical_stability(x, u)
        
        assert isinstance(result, dict)
        assert 'has_nan' in result
        assert 'has_inf' in result
        assert 'max_derivative' in result
        assert 'is_stable' in result
        
        assert not result['has_nan']
        assert not result['has_inf']
        assert result['is_stable']


# ============================================================================
# Test Utility Methods
# ============================================================================

class TestUtilityMethods:
    """Test utility and helper methods"""
    
    def test_print_equations(self, pendulum, capsys):
        """Test equation printing"""
        pendulum.print_equations(simplify=True)
        
        captured = capsys.readouterr()
        assert "SimplePendulum" in captured.out
        assert "State Variables" in captured.out
        assert "Dynamics" in captured.out
    
    def test_clone(self, pendulum):
        """Test system cloning"""
        cloned = pendulum.clone()
        
        assert cloned is not pendulum
        assert cloned.nx == pendulum.nx
        assert cloned.nu == pendulum.nu
        assert len(cloned.parameters) == len(pendulum.parameters)
    
    def test_to_device_cpu(self, pendulum):
        """Test moving to CPU device"""
        pendulum_cpu = pendulum.to_device('cpu')
        assert pendulum_cpu is pendulum  # Returns self
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_to_device_cuda(self, pendulum):
        """Test moving to CUDA device"""
        pendulum_cuda = pendulum.to_device('cuda')
        assert pendulum_cuda is pendulum
    
    def test_get_config_dict(self, pendulum):
        """Test configuration dictionary generation"""
        config = pendulum.get_config_dict()
        
        assert isinstance(config, dict)
        assert 'class_name' in config
        assert 'parameters' in config
        assert 'nx' in config
        assert 'nu' in config
        assert config['class_name'] == 'SimplePendulum'
        assert config['nx'] == 2
        assert config['nu'] == 1
    
    def test_save_config_json(self, pendulum, tmp_path):
        """Test saving configuration to JSON"""
        filepath = tmp_path / "config.json"
        pendulum.save_config(str(filepath))
        
        assert filepath.exists()
        
        import json
        with open(filepath, 'r') as f:
            config = json.load(f)
        
        assert config['class_name'] == 'SimplePendulum'
        assert config['nx'] == 2
    
    def test_save_config_pt(self, pendulum, tmp_path):
        """Test saving configuration to PyTorch file"""
        filepath = tmp_path / "config.pt"
        pendulum.save_config(str(filepath))
        
        assert filepath.exists()
        
        config = torch.load(filepath)
        assert config['class_name'] == 'SimplePendulum'
    
    def test_save_config_invalid_format(self, pendulum, tmp_path):
        """Test that invalid format raises error"""
        filepath = tmp_path / "config.txt"
        
        with pytest.raises(ValueError, match="Unsupported file format"):
            pendulum.save_config(str(filepath))
    
    def test_get_performance_stats(self, pendulum, sample_state_control):
        """Test performance statistics tracking"""
        # Reset stats
        pendulum.reset_performance_stats()
        
        # Make some forward calls
        x = sample_state_control['x_torch']
        u = sample_state_control['u_torch']
        for _ in range(5):
            pendulum.forward(x, u)
        
        stats = pendulum.get_performance_stats()
        
        assert isinstance(stats, dict)
        assert 'forward_calls' in stats
        assert 'forward_time' in stats
        assert 'avg_forward_time' in stats
        assert stats['forward_calls'] == 5
        assert stats['forward_time'] > 0
    
    def test_reset_performance_stats(self, pendulum, sample_state_control):
        """Test performance statistics reset"""
        # Make some calls
        x = sample_state_control['x_torch']
        u = sample_state_control['u_torch']
        pendulum.forward(x, u)
        
        # Reset
        pendulum.reset_performance_stats()
        
        stats = pendulum.get_performance_stats()
        assert stats['forward_calls'] == 0
        assert stats['forward_time'] == 0.0


# ============================================================================
# Test Edge Cases and Error Handling
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_forward_with_zero_dimensional_input(self, pendulum):
        """Test that 0D inputs raise error"""
        x = torch.tensor(0.5)  # 0D tensor
        u = torch.tensor(0.2)
        
        with pytest.raises(ValueError, match="Input tensors must be at least 1D"):
            pendulum.forward(x, u)
    
    def test_linearization_with_scalar_R(self, linear_system):
        """Test LQR with scalar R matrix"""
        Q = np.eye(2)
        R = 1.0  # Scalar
        
        K, S = linear_system.lqr_control(Q, R)
        
        assert K.shape == (1, 2)
    
    def test_linearization_with_incompatible_dimensions(self, linear_system):
        """Test that incompatible dimensions raise error"""
        Q = np.eye(3)  # Wrong size
        R = np.array([[1.0]])
        
        with pytest.raises(ValueError):
            linear_system.lqr_control(Q, R)
    
    def test_batched_linearization_maintains_gradient(self, pendulum):
        """Test that batched linearization maintains gradients"""
        x = torch.randn(5, 2, requires_grad=True)
        u = torch.randn(5, 1, requires_grad=True)
        
        A, B = pendulum.linearized_dynamics(x, u)
        
        # Check that we can compute gradients through linearization
        loss = A.sum() + B.sum()
        loss.backward()
        
        assert x.grad is not None
        assert u.grad is not None


# ============================================================================
# Test Performance
# ============================================================================

class TestPerformance:
    """Test performance-related functionality"""
    
    def test_cached_jacobians_faster(self, pendulum, sample_state_control):
        """Test that cached Jacobians are faster than symbolic evaluation"""
        import time
        
        x = sample_state_control['x_torch']
        u = sample_state_control['u_torch']
        
        # Without caching
        start = time.time()
        for _ in range(10):
            A1, B1 = pendulum.linearized_dynamics(x, u)
        time_uncached = time.time() - start
        
        # With caching
        pendulum._cache_jacobians(backend='torch')
        start = time.time()
        for _ in range(10):
            A2, B2 = pendulum.linearized_dynamics(x, u)
        time_cached = time.time() - start
        
        # Cached should be faster (or at least not slower)
        assert time_cached <= time_uncached * 1.5  # Allow some margin
        
        # Results should be the same
        assert torch.allclose(A1, A2, atol=1e-6)
        assert torch.allclose(B1, B2, atol=1e-6)
    
    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_jax_jit_compilation(self, pendulum):
        """Test that JAX JIT compilation works"""
        import time
        
        pendulum.generate_jax_function(jit=True)
        
        x = jnp.array([0.5, 0.1])
        u = jnp.array([0.2])
        
        # First call (compilation)
        start = time.time()
        dx1 = pendulum.forward(x, u)
        time_first = time.time() - start
        
        # Second call (should be faster due to JIT)
        start = time.time()
        dx2 = pendulum.forward(x, u)
        time_second = time.time() - start
        
        # Results should be identical
        assert jnp.allclose(dx1, dx2)
        
        # Second call should be faster (with some margin for variance)
        # Note: This might not always be true in test environments
        print(f"First call: {time_first:.6f}s, Second call: {time_second:.6f}s")


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])