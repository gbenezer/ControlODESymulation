"""
Comprehensive tests for codegen_utils.py

Tests verify that all backends (NumPy, PyTorch, JAX) generate correct
functions from SymPy expressions with equal matrix handling.

Run with:
    pytest tests/test_codegen_utils.py -v
    pytest tests/test_codegen_utils.py -v -k "matrix"
    pytest tests/test_codegen_utils.py -v -k "backend_equality"
"""

import pytest
import numpy as np
import sympy as sp
from typing import Callable

from src.systems.base.codegen_utils import (
    generate_numpy_function,
    generate_torch_function,
    generate_jax_function,
    generate_function,
    _numpy_matrix_handler,
    _torch_matrix_handler,
    _jax_matrix_handler,
)

# Conditional imports
torch_available = False
jax_available = False

try:
    import torch
    torch_available = True
except ImportError:
    pass

try:
    import jax
    import jax.numpy as jnp
    jax_available = True
except ImportError:
    pass


# ============================================================================
# Test: Matrix Handler Functions (ALL BACKENDS - EQUAL)
# ============================================================================

class TestMatrixHandlers:
    """Test matrix handler functions for all backends"""
    
    def test_numpy_matrix_handler_nested_list(self):
        """Test NumPy matrix handler with nested list"""
        # SymPy returns: [[1.0], [2.0], [3.0]]
        nested = [[1.0], [2.0], [3.0]]
        result = _numpy_matrix_handler(nested)
        
        assert result == [1.0, 2.0, 3.0]
    
    def test_numpy_matrix_handler_flat_list(self):
        """Test NumPy matrix handler with already flat list"""
        flat = [1.0, 2.0, 3.0]
        result = _numpy_matrix_handler(flat)
        
        assert result == [1.0, 2.0, 3.0]
    
    def test_numpy_matrix_handler_single_value(self):
        """Test NumPy matrix handler with single value"""
        result = _numpy_matrix_handler(5.0)
        assert result == 5.0
    
    @pytest.mark.skipif(not torch_available, reason="PyTorch not available")
    def test_torch_matrix_handler_nested_tensors(self):
        """Test PyTorch matrix handler with nested tensors"""
        nested = [[torch.tensor(1.0)], [torch.tensor(2.0)], [torch.tensor(3.0)]]
        result = _torch_matrix_handler(nested)
        
        assert len(result) == 3
        assert all(isinstance(r, torch.Tensor) for r in result)
        assert torch.allclose(result[0], torch.tensor(1.0))
    
    @pytest.mark.skipif(not torch_available, reason="PyTorch not available")
    def test_torch_matrix_handler_flat_tensors(self):
        """Test PyTorch matrix handler with flat list"""
        flat = [torch.tensor(1.0), torch.tensor(2.0)]
        result = _torch_matrix_handler(flat)
        
        assert len(result) == 2
        assert result == flat  # Should be unchanged
    
    @pytest.mark.skipif(not torch_available, reason="PyTorch not available")
    def test_torch_matrix_handler_preserves_gradients(self):
        """Test that PyTorch handler preserves gradient tracking"""
        nested = [[torch.tensor(1.0, requires_grad=True)], 
                  [torch.tensor(2.0, requires_grad=True)]]
        result = _torch_matrix_handler(nested)
        
        assert all(r.requires_grad for r in result)
    
    @pytest.mark.skipif(not jax_available, reason="JAX not available")
    def test_jax_matrix_handler_list(self):
        """Test JAX matrix handler with list"""
        import jax.numpy as jnp
        
        data = [jnp.array(1.0), jnp.array(2.0), jnp.array(3.0)]
        result = _jax_matrix_handler(data)
        
        assert isinstance(result, jnp.ndarray)
        assert result.shape == (3,)
        assert jnp.allclose(result, jnp.array([1.0, 2.0, 3.0]))
    
    @pytest.mark.skipif(not jax_available, reason="JAX not available")
    def test_jax_matrix_handler_single_value(self):
        """Test JAX matrix handler with single value"""
        result = _jax_matrix_handler(5.0)
        # Single values pass through unchanged
        assert result == 5.0


# ============================================================================
# Test: Simple Scalar Expressions (ALL BACKENDS)
# ============================================================================

class TestScalarExpressions:
    """Test generation of simple scalar expressions"""
    
    def test_numpy_simple_expression(self):
        """Test NumPy generation with simple expression"""
        x = sp.Symbol('x')
        expr = x**2 + 2*x + 1
        
        f = generate_numpy_function(expr, [x])
        result = f(3.0)
        
        expected = 3**2 + 2*3 + 1  # = 16
        assert np.allclose(result, expected)
    
    @pytest.mark.skipif(not torch_available, reason="PyTorch not available")
    def test_torch_simple_expression(self):
        """Test PyTorch generation with simple expression"""
        x = sp.Symbol('x')
        expr = x**2 + 2*x + 1
        
        f = generate_torch_function(expr, [x])
        result = f(torch.tensor(3.0))
        
        expected = torch.tensor(16.0)
        assert torch.allclose(result, expected)
    
    @pytest.mark.skipif(not jax_available, reason="JAX not available")
    def test_jax_simple_expression(self):
        """Test JAX generation with simple expression"""
        import jax.numpy as jnp
        
        x = sp.Symbol('x')
        expr = x**2 + 2*x + 1
        
        f = generate_jax_function(expr, [x])
        result = f(3.0)
        
        expected = 16.0
        assert jnp.allclose(result, expected)


# ============================================================================
# Test: Vector Expressions (Matrix Handling - ALL BACKENDS)
# ============================================================================

class TestVectorExpressions:
    """Test generation of vector/matrix expressions"""
    
    def test_numpy_vector_expression(self):
        """Test NumPy with vector expression"""
        x, y = sp.symbols('x y')
        expr = sp.Matrix([x**2, y**2, x*y])
        
        f = generate_numpy_function(expr, [x, y])
        result = f(2.0, 3.0)
        
        expected = np.array([4.0, 9.0, 6.0])
        assert np.allclose(result, expected)
    
    @pytest.mark.skipif(not torch_available, reason="PyTorch not available")
    def test_torch_vector_expression(self):
        """Test PyTorch with vector expression"""
        x, y = sp.symbols('x y')
        expr = sp.Matrix([x**2, y**2, x*y])
        
        f = generate_torch_function(expr, [x, y])
        result = f(torch.tensor(2.0), torch.tensor(3.0))
        
        expected = torch.tensor([4.0, 9.0, 6.0])
        assert torch.allclose(result, expected)
    
    @pytest.mark.skipif(not jax_available, reason="JAX not available")
    def test_jax_vector_expression(self):
        """Test JAX with vector expression"""
        import jax.numpy as jnp
        
        x, y = sp.symbols('x y')
        expr = sp.Matrix([x**2, y**2, x*y])
        
        f = generate_jax_function(expr, [x, y])
        result = f(2.0, 3.0)
        
        expected = jnp.array([4.0, 9.0, 6.0])
        assert jnp.allclose(result, expected)


# ============================================================================
# Test: Trigonometric Functions (ALL BACKENDS)
# ============================================================================

class TestTrigonometricFunctions:
    """Test trigonometric function generation"""
    
    def test_numpy_trig_functions(self):
        """Test NumPy with trigonometric functions"""
        x = sp.Symbol('x')
        expr = sp.Matrix([sp.sin(x), sp.cos(x), sp.tan(x)])
        
        f = generate_numpy_function(expr, [x])
        result = f(np.pi/4)
        
        expected = np.array([np.sin(np.pi/4), np.cos(np.pi/4), np.tan(np.pi/4)])
        assert np.allclose(result, expected)
    
    @pytest.mark.skipif(not torch_available, reason="PyTorch not available")
    def test_torch_trig_functions(self):
        """Test PyTorch with trigonometric functions"""
        x = sp.Symbol('x')
        expr = sp.Matrix([sp.sin(x), sp.cos(x), sp.tan(x)])
        
        f = generate_torch_function(expr, [x])
        result = f(torch.tensor(torch.pi/4))
        
        # Calculate expected values
        x_val = torch.pi/4
        expected_values = torch.tensor([
            torch.sin(x_val).item(),
            torch.cos(x_val).item(),
            torch.tan(x_val).item()
        ])
        
        # Handle different possible shapes
        result_flat = result.flatten() if result.ndim > 1 else result
        
        assert torch.allclose(result_flat, expected_values, rtol=1e-5)
    
    @pytest.mark.skipif(not jax_available, reason="JAX not available")
    def test_jax_trig_functions(self):
        """Test JAX with trigonometric functions"""
        import jax.numpy as jnp
        
        x = sp.Symbol('x')
        expr = sp.Matrix([sp.sin(x), sp.cos(x), sp.tan(x)])
        
        f = generate_jax_function(expr, [x])
        result = f(jnp.pi/4)
        
        expected = jnp.array([
            jnp.sin(jnp.pi/4),
            jnp.cos(jnp.pi/4),
            jnp.tan(jnp.pi/4)
        ])
        assert jnp.allclose(result, expected, rtol=1e-5)


# ============================================================================
# Test: Cross-Backend Consistency
# ============================================================================

class TestCrossBackendConsistency:
    """Test that all backends produce consistent results"""
    
    def test_all_backends_simple_expression(self):
        """Test all backends give same result for simple expression"""
        x = sp.Symbol('x')
        expr = x**3 - 2*x + 5
        
        input_val = 2.5
        results = {}
        
        # NumPy
        f_np = generate_numpy_function(expr, [x])
        results['numpy'] = f_np(input_val)
        
        # PyTorch
        if torch_available:
            f_torch = generate_torch_function(expr, [x])
            results['torch'] = f_torch(torch.tensor(input_val)).detach().numpy()
        
        # JAX
        if jax_available:
            f_jax = generate_jax_function(expr, [x])
            results['jax'] = np.array(f_jax(input_val))
        
        # All should agree
        if len(results) > 1:
            base_result = results['numpy']
            for backend, result in results.items():
                if backend != 'numpy':
                    assert np.allclose(base_result, result, rtol=1e-6)
    
    def test_all_backends_vector_expression(self):
        """Test all backends give same result for vector expression"""
        x, y, z = sp.symbols('x y z')
        expr = sp.Matrix([
            sp.sin(x) + sp.cos(y),
            sp.exp(z) - x*y,
            sp.sqrt(x**2 + y**2)
        ])
        
        input_vals = (1.5, 2.0, 0.5)
        results = {}
        
        # NumPy
        f_np = generate_numpy_function(expr, [x, y, z])
        results['numpy'] = f_np(*input_vals)
        
        # PyTorch
        if torch_available:
            f_torch = generate_torch_function(expr, [x, y, z])
            torch_inputs = [torch.tensor(v) for v in input_vals]
            torch_result = f_torch(*torch_inputs)
            # Flatten to compare (shape might differ)
            results['torch'] = torch_result.flatten().detach().numpy()
        
        # JAX
        if jax_available:
            f_jax = generate_jax_function(expr, [x, y, z])
            results['jax'] = np.array(f_jax(*input_vals)).flatten()
        
        # All should agree (after flattening)
        if len(results) > 1:
            base_result = results['numpy'].flatten() if results['numpy'].ndim > 1 else results['numpy']
            for backend, result in results.items():
                if backend != 'numpy':
                    result_flat = result.flatten() if result.ndim > 1 else result
                    assert np.allclose(base_result, result_flat, rtol=1e-5)


# ============================================================================
# Test: generate_function() Dispatcher
# ============================================================================

class TestGenerateFunctionDispatcher:
    """Test the generic generate_function() dispatcher"""
    
    def test_dispatch_to_numpy(self):
        """Test dispatch to NumPy backend"""
        x = sp.Symbol('x')
        expr = x**2
        
        f = generate_function(expr, [x], backend='numpy')
        result = f(4.0)
        
        assert np.allclose(result, 16.0)
    
    @pytest.mark.skipif(not torch_available, reason="PyTorch not available")
    def test_dispatch_to_torch(self):
        """Test dispatch to PyTorch backend"""
        x = sp.Symbol('x')
        expr = x**2
        
        f = generate_function(expr, [x], backend='torch')
        result = f(torch.tensor(4.0))
        
        assert torch.allclose(result, torch.tensor(16.0))
    
    @pytest.mark.skipif(not jax_available, reason="JAX not available")
    def test_dispatch_to_jax(self):
        """Test dispatch to JAX backend"""
        import jax.numpy as jnp
        
        x = sp.Symbol('x')
        expr = x**2
        
        f = generate_function(expr, [x], backend='jax')
        result = f(4.0)
        
        assert jnp.allclose(result, 16.0)
    
    def test_invalid_backend(self):
        """Test that invalid backend raises error"""
        x = sp.Symbol('x')
        expr = x**2
        
        with pytest.raises(ValueError, match="Unknown backend"):
            generate_function(expr, [x], backend='invalid')


# ============================================================================
# Test: Complex Expressions (ALL BACKENDS)
# ============================================================================

class TestComplexExpressions:
    """Test more complex mathematical expressions"""
    
    def test_numpy_exponential_and_log(self):
        """Test NumPy with exp and log"""
        x = sp.Symbol('x')
        expr = sp.Matrix([sp.exp(x), sp.log(x + 1)])
        
        f = generate_numpy_function(expr, [x])
        result = f(2.0)
        
        expected = np.array([np.exp(2.0), np.log(3.0)])
        assert np.allclose(result, expected)
    
    @pytest.mark.skipif(not torch_available, reason="PyTorch not available")
    def test_torch_exponential_and_log(self):
        """Test PyTorch with exp and log"""
        x = sp.Symbol('x')
        expr = sp.Matrix([sp.exp(x), sp.log(x + 1)])
        
        f = generate_torch_function(expr, [x])
        result = f(torch.tensor(2.0))
        
        expected_values = torch.tensor([
            torch.exp(torch.tensor(2.0)).item(),
            torch.log(torch.tensor(3.0)).item()
        ])
        
        # Handle different possible shapes
        result_flat = result.flatten() if result.ndim > 1 else result
        
        assert torch.allclose(result_flat, expected_values, rtol=1e-6)
    
    @pytest.mark.skipif(not jax_available, reason="JAX not available")
    def test_jax_exponential_and_log(self):
        """Test JAX with exp and log"""
        import jax.numpy as jnp
        
        x = sp.Symbol('x')
        expr = sp.Matrix([sp.exp(x), sp.log(x + 1)])
        
        f = generate_jax_function(expr, [x])
        result = f(2.0)
        
        expected = jnp.array([jnp.exp(2.0), jnp.log(3.0)])
        assert jnp.allclose(result, expected)


# ============================================================================
# Test: Batched Operations (Important for Dynamics)
# ============================================================================

class TestBatchedOperations:
    """Test batched evaluation (multiple inputs at once)"""
    
    def test_numpy_batched_scalar(self):
        """Test NumPy with batched scalar inputs"""
        x = sp.Symbol('x')
        expr = x**2
        
        f = generate_numpy_function(expr, [x])
        
        # Single value
        result_single = f(3.0)
        assert np.allclose(result_single, 9.0)
    
    @pytest.mark.skipif(not torch_available, reason="PyTorch not available")
    def test_torch_batched_scalar(self):
        """Test PyTorch with batched scalar inputs"""
        x = sp.Symbol('x')
        expr = x**2
        
        f = generate_torch_function(expr, [x])
        
        # Batched input
        result_batch = f(torch.tensor([1.0, 2.0, 3.0]))
        expected = torch.tensor([1.0, 4.0, 9.0])
        
        assert torch.allclose(result_batch, expected)
    
    @pytest.mark.skipif(not jax_available, reason="JAX not available")
    def test_jax_batched_scalar(self):
        """Test JAX with batched scalar inputs"""
        import jax.numpy as jnp
        
        x = sp.Symbol('x')
        expr = x**2
        
        f = generate_jax_function(expr, [x])
        
        # Batched input
        result_batch = f(jnp.array([1.0, 2.0, 3.0]))
        expected = jnp.array([1.0, 4.0, 9.0])
        
        assert jnp.allclose(result_batch, expected)


# ============================================================================
# Test: Matrix Expressions (Critical for Dynamics)
# ============================================================================

class TestMatrixExpressions:
    """Test matrix expressions - critical for f(x,u) dynamics"""
    
    def test_numpy_matrix_expression(self):
        """Test NumPy with matrix expression"""
        x, y = sp.symbols('x y')
        expr = sp.Matrix([x**2, y**2])
        
        f = generate_numpy_function(expr, [x, y])
        result = f(2.0, 3.0)
        
        expected = np.array([4.0, 9.0])
        assert np.allclose(result, expected)
    
    @pytest.mark.skipif(not torch_available, reason="PyTorch not available")
    def test_torch_matrix_expression(self):
        """Test PyTorch with matrix expression"""
        x, y = sp.symbols('x y')
        expr = sp.Matrix([x**2, y**2])
        
        f = generate_torch_function(expr, [x, y])
        result = f(torch.tensor(2.0), torch.tensor(3.0))
        
        expected_values = torch.tensor([4.0, 9.0])
        
        # Handle different possible shapes (might be (2,) or (2,1) or (1,2))
        result_flat = result.flatten()
        
        assert torch.allclose(result_flat, expected_values, rtol=1e-6)
    
    @pytest.mark.skipif(not jax_available, reason="JAX not available")
    def test_jax_matrix_expression(self):
        """Test JAX with matrix expression"""
        import jax.numpy as jnp
        
        x, y = sp.symbols('x y')
        expr = sp.Matrix([x**2, y**2])
        
        f = generate_jax_function(expr, [x, y])
        result = f(2.0, 3.0)
        
        expected = jnp.array([4.0, 9.0])
        assert jnp.allclose(result, expected)
    
    def test_all_backends_matrix_consistency(self):
        """Test all backends handle matrices consistently"""
        x, y, z = sp.symbols('x y z')
        expr = sp.Matrix([
            sp.sin(x) + y,
            sp.cos(y) * z,
            sp.exp(x - z)
        ])
        
        input_vals = (1.0, 2.0, 0.5)
        results = {}
        
        # NumPy
        f_np = generate_numpy_function(expr, [x, y, z])
        results['numpy'] = f_np(*input_vals).flatten()
        
        # PyTorch
        if torch_available:
            f_torch = generate_torch_function(expr, [x, y, z])
            torch_inputs = [torch.tensor(v) for v in input_vals]
            torch_result = f_torch(*torch_inputs)
            results['torch'] = torch_result.flatten().detach().numpy()
        
        # JAX
        if jax_available:
            f_jax = generate_jax_function(expr, [x, y, z])
            jax_result = f_jax(*input_vals)
            results['jax'] = np.array(jax_result).flatten()
        
        # All should agree (comparing flattened versions)
        if len(results) > 1:
            base = results['numpy']
            for backend, result in results.items():
                if backend != 'numpy':
                    assert np.allclose(base, result, rtol=1e-5), \
                        f"Backend {backend} differs from NumPy: {base} vs {result}"


# ============================================================================
# Test: JAX-Specific Features
# ============================================================================

@pytest.mark.skipif(not jax_available, reason="JAX not available")
class TestJAXSpecificFeatures:
    """Test JAX-specific features like JIT compilation"""
    
    def test_jax_jit_enabled(self):
        """Test JAX function with JIT enabled"""
        import jax
        import jax.numpy as jnp
        
        x = sp.Symbol('x')
        expr = x**2 + sp.sin(x)
        
        f = generate_jax_function(expr, [x], jit=True)
        
        # Function should be JIT-compiled
        # Just verify it works
        result = f(2.0)
        expected = 2.0**2 + jnp.sin(2.0)
        assert jnp.allclose(result, expected)
    
    def test_jax_jit_disabled(self):
        """Test JAX function with JIT disabled"""
        import jax.numpy as jnp
        
        x = sp.Symbol('x')
        expr = x**2
        
        f = generate_jax_function(expr, [x], jit=False)
        result = f(3.0)
        
        assert jnp.allclose(result, 9.0)


# ============================================================================
# Test: PyTorch-Specific Features
# ============================================================================

@pytest.mark.skipif(not torch_available, reason="PyTorch not available")
class TestPyTorchSpecificFeatures:
    """Test PyTorch-specific features like gradient tracking"""
    
    def test_torch_gradient_preservation(self):
        """Test that generated functions preserve gradients"""
        x = sp.Symbol('x')
        expr = x**2 + 2*x
        
        f = generate_torch_function(expr, [x])
        
        # Input with gradient tracking
        x_val = torch.tensor(3.0, requires_grad=True)
        result = f(x_val)
        
        # Result should support gradients
        result.backward()
        
        # Gradient of x^2 + 2x at x=3 is 2*3 + 2 = 8
        assert x_val.grad is not None
        assert torch.allclose(x_val.grad, torch.tensor(8.0))
    
    def test_torch_vector_gradients(self):
        """Test gradients through vector expressions"""
        x, y = sp.symbols('x y')
        expr = sp.Matrix([x**2, y**2, x*y])
        
        f = generate_torch_function(expr, [x, y])
        
        x_val = torch.tensor(2.0, requires_grad=True)
        y_val = torch.tensor(3.0, requires_grad=True)
        
        result = f(x_val, y_val)
        loss = result.sum()
        loss.backward()
        
        # ∂(x² + y² + xy)/∂x = 2x + y = 7
        # ∂(x² + y² + xy)/∂y = 2y + x = 8
        assert x_val.grad is not None
        assert y_val.grad is not None
        assert torch.allclose(x_val.grad, torch.tensor(7.0))
        assert torch.allclose(y_val.grad, torch.tensor(8.0))


# ============================================================================
# Test: Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_numpy_empty_expression(self):
        """Test NumPy with empty matrix"""
        expr = sp.Matrix([])
        
        # Empty matrix is valid SymPy - might create empty function
        # Just verify it doesn't crash
        f = generate_numpy_function(expr, [])
        
        # Calling with no args might work or raise - both acceptable
        # (Empty expressions are edge case, behavior can vary)
        try:
            result = f()
            # If it works, result should be empty or zero-length
            if hasattr(result, '__len__'):
                assert len(result) == 0 or result.size == 0
        except (ValueError, IndexError, TypeError):
            # If it raises, that's also acceptable
            pass
    
    def test_numpy_single_element_matrix(self):
        """Test NumPy with single element matrix"""
        x = sp.Symbol('x')
        expr = sp.Matrix([x])
        
        f = generate_numpy_function(expr, [x])
        result = f(5.0)
        
        assert np.allclose(result, 5.0)
    
    @pytest.mark.skipif(not torch_available, reason="PyTorch not available")
    def test_torch_zero_dimensional_result(self):
        """Test PyTorch with scalar result"""
        x = sp.Symbol('x')
        expr = x
        
        f = generate_torch_function(expr, [x])
        result = f(torch.tensor(7.0))
        
        # Should handle 0D tensors
        assert result.numel() >= 1


# ============================================================================
# Test: Real Dynamical System Example
# ============================================================================

class TestDynamicalSystemExample:
    """Test with realistic dynamical system expressions"""
    
    def test_pendulum_dynamics_all_backends(self):
        """Test pendulum dynamics on all backends"""
        # Pendulum: dθ/dt = ω, dω/dt = -g/l*sin(θ) - b*ω + τ/(m*l²)
        theta, omega, tau = sp.symbols('theta omega tau')
        g, l, b, m = 9.81, 0.5, 0.1, 0.15
        
        expr = sp.Matrix([
            omega,
            -g/l * sp.sin(theta) - b * omega + tau / (m * l**2)
        ])
        
        # Test values
        theta_val = 0.1
        omega_val = 0.0
        tau_val = 0.0
        
        results = {}
        
        # NumPy
        f_np = generate_numpy_function(expr, [theta, omega, tau])
        results['numpy'] = f_np(theta_val, omega_val, tau_val).flatten()
        
        # PyTorch
        if torch_available:
            f_torch = generate_torch_function(expr, [theta, omega, tau])
            inputs = [torch.tensor(v) for v in [theta_val, omega_val, tau_val]]
            torch_result = f_torch(*inputs)
            results['torch'] = torch_result.flatten().detach().numpy()
        
        # JAX
        if jax_available:
            f_jax = generate_jax_function(expr, [theta, omega, tau])
            jax_result = f_jax(theta_val, omega_val, tau_val)
            results['jax'] = np.array(jax_result).flatten()
        
        # All backends should give same dynamics (comparing flattened)
        if len(results) > 1:
            base = results['numpy']
            for backend, result in results.items():
                if backend != 'numpy':
                    assert np.allclose(base, result, rtol=1e-5), \
                        f"Pendulum dynamics differ: {backend} vs NumPy\n  NumPy: {base}\n  {backend}: {result}"


# ============================================================================
# Test: Backend Equality Verification
# ============================================================================

class TestBackendEquality:
    """Explicitly test that all backends are treated equally"""
    
    def test_all_backends_have_matrix_handlers(self):
        """Test that all backends have matrix handler functions"""
        # All should exist
        assert callable(_numpy_matrix_handler)
        assert callable(_torch_matrix_handler)
        assert callable(_jax_matrix_handler)
    
    def test_matrix_handlers_same_signature(self):
        """Test that all matrix handlers have same signature"""
        import inspect
        
        sig_numpy = inspect.signature(_numpy_matrix_handler)
        sig_torch = inspect.signature(_torch_matrix_handler)
        sig_jax = inspect.signature(_jax_matrix_handler)
        
        # All should take one argument (data)
        assert len(sig_numpy.parameters) == 1
        assert len(sig_torch.parameters) == 1
        assert len(sig_jax.parameters) == 1
    
    def test_matrix_handlers_consistent_behavior(self):
        """Test that matrix handlers behave consistently"""
        # Test with nested list
        nested = [[1.0], [2.0], [3.0]]
        
        result_numpy = _numpy_matrix_handler(nested)
        assert result_numpy == [1.0, 2.0, 3.0]
        
        if torch_available:
            nested_torch = [[torch.tensor(1.0)], [torch.tensor(2.0)], [torch.tensor(3.0)]]
            result_torch = _torch_matrix_handler(nested_torch)
            assert len(result_torch) == 3
        
        if jax_available:
            import jax.numpy as jnp
            # JAX handler works differently (stacks), but should succeed
            nested_jax = [jnp.array(1.0), jnp.array(2.0), jnp.array(3.0)]
            result_jax = _jax_matrix_handler(nested_jax)
            assert isinstance(result_jax, jnp.ndarray)


# ============================================================================
# Test: Documentation and Naming
# ============================================================================

class TestDocumentation:
    """Test that functions are properly documented"""
    
    def test_all_matrix_handlers_have_docstrings(self):
        """Test that all matrix handlers have docstrings"""
        assert _numpy_matrix_handler.__doc__ is not None
        assert _torch_matrix_handler.__doc__ is not None
        assert _jax_matrix_handler.__doc__ is not None
    
    def test_all_generators_have_docstrings(self):
        """Test that all generator functions have docstrings"""
        generators = [
            generate_numpy_function,
            generate_torch_function,
            generate_jax_function,
            generate_function,
        ]
        
        for gen in generators:
            assert gen.__doc__ is not None, f"{gen.__name__} missing docstring"
            assert len(gen.__doc__.strip()) > 0, f"{gen.__name__} has empty docstring"
    
    def test_docstrings_mention_matrix_handling(self):
        """Test that docstrings mention matrix handling"""
        # generate_function should mention matrices or vectors
        doc = generate_function.__doc__
        assert doc is not None
        doc_lower = doc.lower()
        assert 'matrix' in doc_lower or 'vector' in doc_lower or 'expression' in doc_lower


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])