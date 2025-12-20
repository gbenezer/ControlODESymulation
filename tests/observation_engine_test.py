"""
Comprehensive unit tests for ObservationEngine

Tests cover:
1. NumPy output evaluation
2. PyTorch output evaluation
3. JAX output evaluation
4. Default output (identity)
5. Custom output functions
6. Linearized observation (C matrix)
7. Symbolic observation Jacobian
"""

import pytest
import numpy as np
import sympy as sp

# Conditional imports
torch_available = True
try:
    import torch
except ImportError:
    torch_available = False

jax_available = True
try:
    import jax
    import jax.numpy as jnp
except ImportError:
    jax_available = False

from src.systems.base.observation_engine import ObservationEngine
from src.systems.base.code_generator import CodeGenerator
from src.systems.base.backend_manager import BackendManager


# ============================================================================
# Mock Systems
# ============================================================================


class MockSystemNoOutput:
    """System without custom output (uses identity)"""
    
    def __init__(self):
        x = sp.symbols('x', real=True)
        u = sp.symbols('u', real=True)
        
        self.state_vars = [x]
        self.control_vars = [u]
        self._f_sym = sp.Matrix([-x + u])
        self._h_sym = None  # No custom output
        self.parameters = {}
        self.nx = 1
        self.nu = 1
        self.ny = 1
        self.order = 1
    
    def substitute_parameters(self, expr):
        return expr


class MockSystemCustomOutput:
    """System with custom output: y = [x1, x1^2 + x2^2]"""
    
    def __init__(self):
        x1, x2 = sp.symbols('x1 x2', real=True)
        u = sp.symbols('u', real=True)
        
        self.state_vars = [x1, x2]
        self.control_vars = [u]
        self._f_sym = sp.Matrix([x2, -x1])
        self._h_sym = sp.Matrix([x1, x1**2 + x2**2])  # Custom output
        self.parameters = {}
        self.nx = 2
        self.nu = 1
        self.ny = 2
        self.order = 1
    
    def substitute_parameters(self, expr):
        return expr


# ============================================================================
# Test Class 1: Default Output (Identity)
# ============================================================================


class TestDefaultOutput:
    """Test systems without custom output functions"""
    
    def test_evaluate_numpy_identity(self):
        """Test default output returns state (identity)"""
        system = MockSystemNoOutput()
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        engine = ObservationEngine(system, code_gen, backend_mgr)
        
        x = np.array([1.0])
        y = engine.evaluate(x, backend='numpy')
        
        assert isinstance(y, np.ndarray)
        assert np.allclose(y, x)
    
    @pytest.mark.skipif(not torch_available, reason="PyTorch not installed")
    def test_evaluate_torch_identity(self):
        """Test default output with PyTorch"""
        system = MockSystemNoOutput()
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        engine = ObservationEngine(system, code_gen, backend_mgr)
        
        x = torch.tensor([1.0])
        y = engine.evaluate(x, backend='torch')
        
        assert isinstance(y, torch.Tensor)
        assert torch.allclose(y, x)
    
    def test_jacobian_numpy_identity(self):
        """Test observation Jacobian is identity for default output"""
        system = MockSystemNoOutput()
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        engine = ObservationEngine(system, code_gen, backend_mgr)
        
        x = np.array([1.0])
        C = engine.compute_jacobian(x, backend='numpy')
        
        assert C.shape == (1, 1)
        assert np.allclose(C, np.eye(1))


# ============================================================================
# Test Class 2: Custom Output Evaluation
# ============================================================================


class TestCustomOutput:
    """Test systems with custom output functions"""
    
    def test_evaluate_numpy_custom(self):
        """Test custom output with NumPy"""
        system = MockSystemCustomOutput()
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        engine = ObservationEngine(system, code_gen, backend_mgr)
        
        x = np.array([1.0, 2.0])
        y = engine.evaluate(x, backend='numpy')
        
        # y = [x1, x1^2 + x2^2] = [1, 1 + 4] = [1, 5]
        assert y.shape == (2,)
        assert np.allclose(y, np.array([1.0, 5.0]))
    
    @pytest.mark.skipif(not torch_available, reason="PyTorch not installed")
    def test_evaluate_torch_custom(self):
        """Test custom output with PyTorch"""
        system = MockSystemCustomOutput()
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        engine = ObservationEngine(system, code_gen, backend_mgr)
        
        x = torch.tensor([1.0, 2.0])
        y = engine.evaluate(x, backend='torch')
        
        assert isinstance(y, torch.Tensor)
        assert torch.allclose(y, torch.tensor([1.0, 5.0]))
    
    @pytest.mark.skipif(not jax_available, reason="JAX not installed")
    def test_evaluate_jax_custom(self):
        """Test custom output with JAX"""
        system = MockSystemCustomOutput()
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        engine = ObservationEngine(system, code_gen, backend_mgr)
        
        x = jnp.array([1.0, 2.0])
        y = engine.evaluate(x, backend='jax')
        
        assert isinstance(y, jnp.ndarray)
        assert jnp.allclose(y, jnp.array([1.0, 5.0]))
    
    def test_evaluate_auto_detect(self):
        """Test auto-detection of backend"""
        system = MockSystemCustomOutput()
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        engine = ObservationEngine(system, code_gen, backend_mgr)
        
        x = np.array([1.0, 2.0])
        y = engine.evaluate(x)  # No backend specified
        
        assert isinstance(y, np.ndarray)


# ============================================================================
# Test Class 3: Observation Jacobian (C matrix)
# ============================================================================


class TestObservationJacobian:
    """Test linearized observation computation"""
    
    def test_compute_jacobian_numpy(self):
        """Test C matrix computation with NumPy"""
        system = MockSystemCustomOutput()
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        engine = ObservationEngine(system, code_gen, backend_mgr)
        
        x = np.array([1.0, 2.0])
        C = engine.compute_jacobian(x, backend='numpy')
        
        # C = ∂h/∂x = [[1, 0], [2*x1, 2*x2]] = [[1, 0], [2, 4]]
        expected_C = np.array([[1.0, 0.0], [2.0, 4.0]])
        
        assert C.shape == (2, 2)
        assert np.allclose(C, expected_C)
    
    @pytest.mark.skipif(not torch_available, reason="PyTorch not installed")
    def test_compute_jacobian_torch(self):
        """Test C matrix with PyTorch"""
        system = MockSystemCustomOutput()
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        engine = ObservationEngine(system, code_gen, backend_mgr)
        
        x = torch.tensor([1.0, 2.0])
        C = engine.compute_jacobian(x, backend='torch')
        
        expected_C = torch.tensor([[1.0, 0.0], [2.0, 4.0]])
        
        assert isinstance(C, torch.Tensor)
        assert torch.allclose(C, expected_C)
    
    @pytest.mark.skipif(not jax_available, reason="JAX not installed")
    def test_compute_jacobian_jax(self):
        """Test C matrix with JAX (uses autodiff)"""
        system = MockSystemCustomOutput()
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        engine = ObservationEngine(system, code_gen, backend_mgr)
        
        x = jnp.array([1.0, 2.0])
        C = engine.compute_jacobian(x, backend='jax')
        
        expected_C = jnp.array([[1.0, 0.0], [2.0, 4.0]])
        
        assert isinstance(C, jnp.ndarray)
        assert jnp.allclose(C, expected_C)
    
    def test_compute_symbolic(self):
        """Test symbolic C matrix computation"""
        system = MockSystemCustomOutput()
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        engine = ObservationEngine(system, code_gen, backend_mgr)
        
        x_eq = sp.Matrix([1, 2])
        C_sym = engine.compute_symbolic(x_eq)
        
        assert isinstance(C_sym, sp.Matrix)
        
        C_np = np.array(C_sym, dtype=float)
        expected_C = np.array([[1.0, 0.0], [2.0, 4.0]])
        
        assert np.allclose(C_np, expected_C)


# ============================================================================
# Test Class 4: Backend Conversion
# ============================================================================


class TestBackendConversion:
    """Test backend conversion and dispatch"""
    
    @pytest.mark.skipif(not torch_available, reason="PyTorch not installed")
    def test_force_backend_conversion(self):
        """Test forcing backend with conversion"""
        system = MockSystemCustomOutput()
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        engine = ObservationEngine(system, code_gen, backend_mgr)
        
        # NumPy input, force torch backend
        x = np.array([1.0, 2.0])
        y = engine.evaluate(x, backend='torch')
        
        # Should return torch tensor
        assert isinstance(y, torch.Tensor)
        assert torch.allclose(y, torch.tensor([1.0, 5.0]))


# ============================================================================
# Run Tests
# ============================================================================


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])