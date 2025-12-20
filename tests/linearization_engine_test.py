"""
Comprehensive unit tests for LinearizationEngine

Tests cover:
1. NumPy linearization
2. PyTorch linearization
3. JAX linearization
4. Symbolic linearization
5. Second-order systems
6. Jacobian verification
7. Performance tracking
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

from src.systems.base.linearization_engine import LinearizationEngine
from src.systems.base.code_generator import CodeGenerator
from src.systems.base.backend_manager import BackendManager


# ============================================================================
# Mock Systems
# ============================================================================


class MockLinearSystem:
    """Simple linear system: dx/dt = -a*x + u"""
    
    def __init__(self, a=2.0):
        x = sp.symbols('x', real=True)
        u = sp.symbols('u', real=True)
        a_sym = sp.symbols('a', real=True, positive=True)
        
        self.state_vars = [x]
        self.control_vars = [u]
        self._f_sym = sp.Matrix([-a_sym * x + u])
        self._h_sym = None
        self.parameters = {a_sym: a}
        self.nx = 1
        self.nu = 1
        self.ny = 1
        self.nq = 1
        self.order = 1
    
    def substitute_parameters(self, expr):
        return expr.subs(self.parameters)


class MockSecondOrderSystem:
    """Harmonic oscillator: q̈ = -k*q - c*q̇ + u"""
    
    def __init__(self, k=10.0, c=0.5):
        q, q_dot = sp.symbols('q q_dot', real=True)
        u = sp.symbols('u', real=True)
        k_sym, c_sym = sp.symbols('k c', real=True, positive=True)
        
        self.state_vars = [q, q_dot]
        self.control_vars = [u]
        self._f_sym = sp.Matrix([-k_sym * q - c_sym * q_dot + u])
        self._h_sym = None
        self.parameters = {k_sym: k, c_sym: c}
        self.nx = 2
        self.nu = 1
        self.ny = 2
        self.nq = 1
        self.order = 2
    
    def substitute_parameters(self, expr):
        return expr.subs(self.parameters)


# ============================================================================
# Test Class 1: NumPy Linearization
# ============================================================================


class TestNumpyLinearization:
    """Test NumPy backend linearization"""
    
    def test_compute_dynamics_numpy(self):
        """Test linearization with NumPy"""
        system = MockLinearSystem(a=2.0)
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        engine = LinearizationEngine(system, code_gen, backend_mgr)
        
        x = np.array([1.0])
        u = np.array([0.0])
        
        A, B = engine.compute_dynamics(x, u, backend='numpy')
        
        assert isinstance(A, np.ndarray)
        assert isinstance(B, np.ndarray)
        assert A.shape == (1, 1)
        assert B.shape == (1, 1)
        
        # ∂f/∂x = -a = -2
        assert np.allclose(A, np.array([[-2.0]]))
        # ∂f/∂u = 1
        assert np.allclose(B, np.array([[1.0]]))
    
    def test_compute_dynamics_numpy_batched(self):
        """Test batched linearization"""
        system = MockLinearSystem(a=2.0)
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        engine = LinearizationEngine(system, code_gen, backend_mgr)
        
        x = np.array([[1.0], [2.0], [3.0]])
        u = np.array([[0.0], [0.0], [0.0]])
        
        A, B = engine.compute_dynamics(x, u, backend='numpy')
        
        assert A.shape == (3, 1, 1)
        assert B.shape == (3, 1, 1)
        
        # All should be same (linear system)
        for i in range(3):
            assert np.allclose(A[i], np.array([[-2.0]]))
            assert np.allclose(B[i], np.array([[1.0]]))
    
    def test_compute_dynamics_second_order(self):
        """Test second-order system linearization"""
        system = MockSecondOrderSystem(k=10.0, c=0.5)
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        engine = LinearizationEngine(system, code_gen, backend_mgr)
        
        x = np.array([0.0, 0.0])
        u = np.array([0.0])
        
        A, B = engine.compute_dynamics(x, u, backend='numpy')
        
        # State-space form: dx/dt = [0 1; -k -c][q; q̇] + [0; 1]u
        expected_A = np.array([[0, 1], [-10, -0.5]])
        expected_B = np.array([[0], [1]])
        
        assert A.shape == (2, 2)
        assert B.shape == (2, 1)
        assert np.allclose(A, expected_A)
        assert np.allclose(B, expected_B)


# ============================================================================
# Test Class 2: PyTorch Linearization
# ============================================================================


class TestTorchLinearization:
    """Test PyTorch backend linearization"""
    
    @pytest.mark.skipif(not torch_available, reason="PyTorch not installed")
    def test_compute_dynamics_torch(self):
        """Test linearization with PyTorch"""
        system = MockLinearSystem(a=2.0)
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        engine = LinearizationEngine(system, code_gen, backend_mgr)
        
        x = torch.tensor([1.0])
        u = torch.tensor([0.0])
        
        A, B = engine.compute_dynamics(x, u, backend='torch')
        
        assert isinstance(A, torch.Tensor)
        assert isinstance(B, torch.Tensor)
        assert A.shape == (1, 1)
        assert B.shape == (1, 1)
        assert torch.allclose(A, torch.tensor([[-2.0]]))
        assert torch.allclose(B, torch.tensor([[1.0]]))
    
    @pytest.mark.skipif(not torch_available, reason="PyTorch not installed")
    def test_compute_dynamics_torch_auto_detect(self):
        """Test auto-detection of PyTorch backend"""
        system = MockLinearSystem(a=2.0)
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        engine = LinearizationEngine(system, code_gen, backend_mgr)
        
        x = torch.tensor([1.0])
        u = torch.tensor([0.0])
        
        # Should auto-detect torch
        A, B = engine.compute_dynamics(x, u)
        
        assert isinstance(A, torch.Tensor)
        assert isinstance(B, torch.Tensor)


# ============================================================================
# Test Class 3: JAX Linearization
# ============================================================================


class TestJaxLinearization:
    """Test JAX backend linearization"""
    
    @pytest.mark.skipif(not jax_available, reason="JAX not installed")
    def test_compute_dynamics_jax(self):
        """Test linearization with JAX (uses autodiff)"""
        system = MockLinearSystem(a=2.0)
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        engine = LinearizationEngine(system, code_gen, backend_mgr)
        
        x = jnp.array([1.0])
        u = jnp.array([0.0])
        
        A, B = engine.compute_dynamics(x, u, backend='jax')
        
        assert isinstance(A, jnp.ndarray)
        assert isinstance(B, jnp.ndarray)
        assert jnp.allclose(A, jnp.array([[-2.0]]))
        assert jnp.allclose(B, jnp.array([[1.0]]))


# ============================================================================
# Test Class 4: Symbolic Linearization
# ============================================================================


class TestSymbolicLinearization:
    """Test symbolic linearization"""
    
    def test_compute_symbolic_first_order(self):
        """Test symbolic linearization for first-order system"""
        system = MockLinearSystem(a=2.0)
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        engine = LinearizationEngine(system, code_gen, backend_mgr)
        
        x_eq = sp.Matrix([0])
        u_eq = sp.Matrix([0])
        
        A_sym, B_sym = engine.compute_symbolic(x_eq, u_eq)
        
        assert isinstance(A_sym, sp.Matrix)
        assert isinstance(B_sym, sp.Matrix)
        
        # Convert to numpy
        A_np = np.array(A_sym, dtype=float)
        B_np = np.array(B_sym, dtype=float)
        
        assert np.allclose(A_np, np.array([[-2.0]]))
        assert np.allclose(B_np, np.array([[1.0]]))
    
    def test_compute_symbolic_second_order(self):
        """Test symbolic linearization for second-order system"""
        system = MockSecondOrderSystem(k=10.0, c=0.5)
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        engine = LinearizationEngine(system, code_gen, backend_mgr)
        
        x_eq = sp.Matrix([0, 0])
        u_eq = sp.Matrix([0])
        
        A_sym, B_sym = engine.compute_symbolic(x_eq, u_eq)
        
        # Should construct state-space form
        A_np = np.array(A_sym, dtype=float)
        B_np = np.array(B_sym, dtype=float)
        
        expected_A = np.array([[0, 1], [-10, -0.5]])
        expected_B = np.array([[0], [1]])
        
        assert np.allclose(A_np, expected_A)
        assert np.allclose(B_np, expected_B)


# ============================================================================
# Test Class 5: Jacobian Verification
# ============================================================================


class TestJacobianVerification:
    """Test Jacobian verification against autodiff"""
    
    @pytest.mark.skipif(not torch_available, reason="PyTorch not installed")
    def test_verify_jacobians_torch(self):
        """Test verification with PyTorch autodiff"""
        system = MockLinearSystem(a=2.0)
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        engine = LinearizationEngine(system, code_gen, backend_mgr)
        
        x = torch.tensor([1.0])
        u = torch.tensor([0.0])
        
        results = engine.verify_jacobians(x, u, backend='torch', tol=1e-4)
        
        assert results['A_match'] is True
        assert results['B_match'] is True
        assert results['A_error'] < 1e-4
        assert results['B_error'] < 1e-4
    
    @pytest.mark.skipif(not jax_available, reason="JAX not installed")
    def test_verify_jacobians_jax(self):
        """Test verification with JAX autodiff"""
        system = MockLinearSystem(a=2.0)
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        engine = LinearizationEngine(system, code_gen, backend_mgr)
        
        x = jnp.array([1.0])
        u = jnp.array([0.0])
        
        results = engine.verify_jacobians(x, u, backend='jax', tol=1e-4)
        
        assert results['A_match'] is True
        assert results['B_match'] is True
    
    def test_verify_jacobians_numpy_fails(self):
        """Test that NumPy backend is rejected"""
        system = MockLinearSystem()
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        engine = LinearizationEngine(system, code_gen, backend_mgr)
        
        x = np.array([1.0])
        u = np.array([0.0])
        
        with pytest.raises(ValueError, match="requires autodiff"):
            engine.verify_jacobians(x, u, backend='numpy')
    
    @pytest.mark.skipif(not torch_available, reason="PyTorch not installed")
    def test_verify_second_order_jacobians(self):
        """Test verification for second-order system"""
        system = MockSecondOrderSystem(k=10.0, c=0.5)
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        engine = LinearizationEngine(system, code_gen, backend_mgr)
        
        x = torch.tensor([0.1, 0.0])
        u = torch.tensor([0.0])
        
        results = engine.verify_jacobians(x, u, backend='torch', tol=1e-3)
        
        assert results['A_match'] is True
        assert results['B_match'] is True


# ============================================================================
# Test Class 6: Performance Tracking
# ============================================================================


class TestPerformanceTracking:
    """Test performance statistics"""
    
    def test_initial_stats(self):
        """Test initial stats are zero"""
        system = MockLinearSystem()
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        engine = LinearizationEngine(system, code_gen, backend_mgr)
        
        stats = engine.get_stats()
        
        assert stats['calls'] == 0
        assert stats['total_time'] == 0.0
        assert stats['avg_time'] == 0.0
    
    def test_stats_increment(self):
        """Test that stats increment"""
        system = MockLinearSystem()
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        engine = LinearizationEngine(system, code_gen, backend_mgr)
        
        x = np.array([1.0])
        u = np.array([0.0])
        
        engine.compute_dynamics(x, u)
        engine.compute_dynamics(x, u)
        
        stats = engine.get_stats()
        
        assert stats['calls'] == 2
        assert stats['total_time'] > 0
    
    def test_reset_stats(self):
        """Test resetting stats"""
        system = MockLinearSystem()
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        engine = LinearizationEngine(system, code_gen, backend_mgr)
        
        x = np.array([1.0])
        u = np.array([0.0])
        
        engine.compute_dynamics(x, u)
        engine.reset_stats()
        
        stats = engine.get_stats()
        assert stats['calls'] == 0


# ============================================================================
# Test Class 7: String Representations
# ============================================================================


class TestStringRepresentations:
    """Test __repr__ and __str__"""
    
    def test_repr(self):
        """Test __repr__ output"""
        system = MockLinearSystem()
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        engine = LinearizationEngine(system, code_gen, backend_mgr)
        
        repr_str = repr(engine)
        
        assert 'LinearizationEngine' in repr_str
        assert 'nx=1' in repr_str
    
    def test_str(self):
        """Test __str__ output"""
        system = MockLinearSystem()
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        engine = LinearizationEngine(system, code_gen, backend_mgr)
        
        str_repr = str(engine)
        
        assert 'LinearizationEngine' in str_repr


# ============================================================================
# Test Class 8: Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests"""
    
    def test_full_workflow(self):
        """Test complete workflow"""
        system = MockLinearSystem(a=2.0)
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        engine = LinearizationEngine(system, code_gen, backend_mgr)
        
        x = np.array([1.0])
        u = np.array([0.0])
        
        # Numerical linearization
        A_num, B_num = engine.compute_dynamics(x, u)
        
        # Symbolic linearization
        A_sym, B_sym = engine.compute_symbolic(sp.Matrix([1.0]), sp.Matrix([0.0]))
        A_sym_np = np.array(A_sym, dtype=float)
        B_sym_np = np.array(B_sym, dtype=float)
        
        # Should match
        assert np.allclose(A_num, A_sym_np)
        assert np.allclose(B_num, B_sym_np)
    
    def test_multi_backend_consistency(self):
        """Test that all backends give same result"""
        system = MockLinearSystem(a=2.0)
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        engine = LinearizationEngine(system, code_gen, backend_mgr)
        
        x_np = np.array([1.0])
        u_np = np.array([0.0])
        
        A_np, B_np = engine.compute_dynamics(x_np, u_np, backend='numpy')
        
        backends_to_test = ['numpy']
        if torch_available:
            backends_to_test.append('torch')
        if jax_available:
            backends_to_test.append('jax')
        
        for backend in backends_to_test:
            A, B = engine.compute_dynamics(x_np, u_np, backend=backend)
            A_val = np.array(A) if not isinstance(A, np.ndarray) else A
            B_val = np.array(B) if not isinstance(B, np.ndarray) else B
            
            assert np.allclose(A_val, A_np), f"{backend} A doesn't match NumPy"
            assert np.allclose(B_val, B_np), f"{backend} B doesn't match NumPy"


# ============================================================================
# Run Tests
# ============================================================================


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])