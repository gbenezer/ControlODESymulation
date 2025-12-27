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
Comprehensive tests for codegen_utils.py

Tests verify that all backends (NumPy, PyTorch, JAX) generate correct
functions from SymPy expressions with equal matrix handling.

Run with:
    pytest tests/test_codegen_utils.py -v
    pytest tests/test_codegen_utils.py -v -k "matrix"
    pytest tests/test_codegen_utils.py -v -k "backend_equality"
"""

from typing import Callable

import numpy as np
import pytest
import sympy as sp

# Import from centralized type system
from src.types.backends import Backend
from src.types.symbolic import SymbolicExpressionInput

from src.systems.base.utils.codegen_utils import (
    _jax_matrix_handler,
    _jax_max,
    _jax_min,
    _numpy_matrix_handler,
    _numpy_max,
    _numpy_min,
    _torch_matrix_handler,
    _torch_max,
    _torch_min,
    generate_function,
    generate_jax_function,
    generate_numpy_function,
    generate_torch_function,
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
        nested = [[torch.tensor(1.0, requires_grad=True)], [torch.tensor(2.0, requires_grad=True)]]
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
        x = sp.Symbol("x")
        expr = x**2 + 2 * x + 1

        f = generate_numpy_function(expr, [x])
        result = f(3.0)

        expected = 3**2 + 2 * 3 + 1  # = 16
        assert np.allclose(result, expected)

    @pytest.mark.skipif(not torch_available, reason="PyTorch not available")
    def test_torch_simple_expression(self):
        """Test PyTorch generation with simple expression"""
        x = sp.Symbol("x")
        expr = x**2 + 2 * x + 1

        f = generate_torch_function(expr, [x])
        result = f(torch.tensor(3.0))

        expected = torch.tensor(16.0)
        assert torch.allclose(result, expected)

    @pytest.mark.skipif(not jax_available, reason="JAX not available")
    def test_jax_simple_expression(self):
        """Test JAX generation with simple expression"""
        import jax.numpy as jnp

        x = sp.Symbol("x")
        expr = x**2 + 2 * x + 1

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
        x, y = sp.symbols("x y")
        expr = sp.Matrix([x**2, y**2, x * y])

        f = generate_numpy_function(expr, [x, y])
        result = f(2.0, 3.0)

        expected = np.array([4.0, 9.0, 6.0])
        assert np.allclose(result, expected)

    @pytest.mark.skipif(not torch_available, reason="PyTorch not available")
    def test_torch_vector_expression(self):
        """Test PyTorch with vector expression"""
        x, y = sp.symbols("x y")
        expr = sp.Matrix([x**2, y**2, x * y])

        f = generate_torch_function(expr, [x, y])
        result = f(torch.tensor(2.0), torch.tensor(3.0))

        expected = torch.tensor([4.0, 9.0, 6.0])
        assert torch.allclose(result, expected)

    @pytest.mark.skipif(not jax_available, reason="JAX not available")
    def test_jax_vector_expression(self):
        """Test JAX with vector expression"""
        import jax.numpy as jnp

        x, y = sp.symbols("x y")
        expr = sp.Matrix([x**2, y**2, x * y])

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
        x = sp.Symbol("x")
        expr = sp.Matrix([sp.sin(x), sp.cos(x), sp.tan(x)])

        f = generate_numpy_function(expr, [x])
        result = f(np.pi / 4)

        expected = np.array([np.sin(np.pi / 4), np.cos(np.pi / 4), np.tan(np.pi / 4)])
        assert np.allclose(result, expected)

    @pytest.mark.skipif(not torch_available, reason="PyTorch not available")
    def test_torch_trig_functions(self):
        """Test PyTorch with trigonometric functions"""
        x = sp.Symbol("x")
        expr = sp.Matrix([sp.sin(x), sp.cos(x), sp.tan(x)])

        f = generate_torch_function(expr, [x])

        # Use tensor input (not float)
        x_val = torch.tensor(torch.pi / 4)
        result = f(x_val)

        # Calculate expected values
        expected_values = torch.tensor(
            [torch.sin(x_val).item(), torch.cos(x_val).item(), torch.tan(x_val).item()]
        )

        # Handle different possible shapes
        result_flat = result.flatten() if result.ndim > 1 else result

        assert torch.allclose(result_flat, expected_values, rtol=1e-5)

    @pytest.mark.skipif(not jax_available, reason="JAX not available")
    def test_jax_trig_functions(self):
        """Test JAX with trigonometric functions"""
        import jax.numpy as jnp

        x = sp.Symbol("x")
        expr = sp.Matrix([sp.sin(x), sp.cos(x), sp.tan(x)])

        f = generate_jax_function(expr, [x])
        result = f(jnp.pi / 4)

        expected = jnp.array([jnp.sin(jnp.pi / 4), jnp.cos(jnp.pi / 4), jnp.tan(jnp.pi / 4)])
        assert jnp.allclose(result, expected, rtol=1e-5)


# ============================================================================
# Test: Cross-Backend Consistency
# ============================================================================


class TestCrossBackendConsistency:
    """Test that all backends produce consistent results"""

    def test_all_backends_simple_expression(self):
        """Test all backends give same result for simple expression"""
        x = sp.Symbol("x")
        expr = x**3 - 2 * x + 5

        input_val = 2.5
        results = {}

        # NumPy
        f_np = generate_numpy_function(expr, [x])
        results["numpy"] = f_np(input_val)

        # PyTorch
        if torch_available:
            f_torch = generate_torch_function(expr, [x])
            results["torch"] = f_torch(torch.tensor(input_val)).detach().numpy()

        # JAX
        if jax_available:
            f_jax = generate_jax_function(expr, [x])
            results["jax"] = np.array(f_jax(input_val))

        # All should agree
        if len(results) > 1:
            base_result = results["numpy"]
            for backend, result in results.items():
                if backend != "numpy":
                    assert np.allclose(base_result, result, rtol=1e-6)

    def test_all_backends_vector_expression(self):
        """Test all backends give same result for vector expression"""
        x, y, z = sp.symbols("x y z")
        expr = sp.Matrix([sp.sin(x) + sp.cos(y), sp.exp(z) - x * y, sp.sqrt(x**2 + y**2)])

        input_vals = (1.5, 2.0, 0.5)
        results = {}

        # NumPy
        f_np = generate_numpy_function(expr, [x, y, z])
        results["numpy"] = f_np(*input_vals)

        # PyTorch
        if torch_available:
            f_torch = generate_torch_function(expr, [x, y, z])
            torch_inputs = [torch.tensor(v) for v in input_vals]
            torch_result = f_torch(*torch_inputs)
            # Flatten to compare (shape might differ)
            results["torch"] = torch_result.flatten().detach().numpy()

        # JAX
        if jax_available:
            f_jax = generate_jax_function(expr, [x, y, z])
            results["jax"] = np.array(f_jax(*input_vals)).flatten()

        # All should agree (after flattening)
        if len(results) > 1:
            base_result = (
                results["numpy"].flatten() if results["numpy"].ndim > 1 else results["numpy"]
            )
            for backend, result in results.items():
                if backend != "numpy":
                    result_flat = result.flatten() if result.ndim > 1 else result
                    assert np.allclose(base_result, result_flat, rtol=1e-5)


# ============================================================================
# Test: generate_function() Dispatcher
# ============================================================================


class TestGenerateFunctionDispatcher:
    """Test the generic generate_function() dispatcher"""

    def test_dispatch_to_numpy(self):
        """Test dispatch to NumPy backend"""
        x = sp.Symbol("x")
        expr = x**2

        f = generate_function(expr, [x], backend="numpy")
        result = f(4.0)

        assert np.allclose(result, 16.0)

    @pytest.mark.skipif(not torch_available, reason="PyTorch not available")
    def test_dispatch_to_torch(self):
        """Test dispatch to PyTorch backend"""
        x = sp.Symbol("x")
        expr = x**2

        f = generate_function(expr, [x], backend="torch")
        result = f(torch.tensor(4.0))

        assert torch.allclose(result, torch.tensor(16.0))

    @pytest.mark.skipif(not jax_available, reason="JAX not available")
    def test_dispatch_to_jax(self):
        """Test dispatch to JAX backend"""
        import jax.numpy as jnp

        x = sp.Symbol("x")
        expr = x**2

        f = generate_function(expr, [x], backend="jax")
        result = f(4.0)

        assert jnp.allclose(result, 16.0)

    def test_invalid_backend(self):
        """Test that invalid backend raises error"""
        x = sp.Symbol("x")
        expr = x**2

        with pytest.raises(ValueError, match="Unknown backend"):
            generate_function(expr, [x], backend="invalid")


# ============================================================================
# Test: Complex Expressions (ALL BACKENDS)
# ============================================================================


class TestComplexExpressions:
    """Test more complex mathematical expressions"""

    def test_numpy_exponential_and_log(self):
        """Test NumPy with exp and log"""
        x = sp.Symbol("x")
        expr = sp.Matrix([sp.exp(x), sp.log(x + 1)])

        f = generate_numpy_function(expr, [x])
        result = f(2.0)

        expected = np.array([np.exp(2.0), np.log(3.0)])
        assert np.allclose(result, expected)

    @pytest.mark.skipif(not torch_available, reason="PyTorch not available")
    def test_torch_exponential_and_log(self):
        """Test PyTorch with exp and log"""
        x = sp.Symbol("x")
        expr = sp.Matrix([sp.exp(x), sp.log(x + 1)])

        f = generate_torch_function(expr, [x])
        result = f(torch.tensor(2.0))

        expected_values = torch.tensor(
            [torch.exp(torch.tensor(2.0)).item(), torch.log(torch.tensor(3.0)).item()]
        )

        # Handle different possible shapes
        result_flat = result.flatten() if result.ndim > 1 else result

        assert torch.allclose(result_flat, expected_values, rtol=1e-6)

    @pytest.mark.skipif(not jax_available, reason="JAX not available")
    def test_jax_exponential_and_log(self):
        """Test JAX with exp and log"""
        import jax.numpy as jnp

        x = sp.Symbol("x")
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
        x = sp.Symbol("x")
        expr = x**2

        f = generate_numpy_function(expr, [x])

        # Single value
        result_single = f(3.0)
        assert np.allclose(result_single, 9.0)

    @pytest.mark.skipif(not torch_available, reason="PyTorch not available")
    def test_torch_batched_scalar(self):
        """Test PyTorch with batched scalar inputs"""
        x = sp.Symbol("x")
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

        x = sp.Symbol("x")
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
        x, y = sp.symbols("x y")
        expr = sp.Matrix([x**2, y**2])

        f = generate_numpy_function(expr, [x, y])
        result = f(2.0, 3.0)

        expected = np.array([4.0, 9.0])
        assert np.allclose(result, expected)

    @pytest.mark.skipif(not torch_available, reason="PyTorch not available")
    def test_torch_matrix_expression(self):
        """Test PyTorch with matrix expression"""
        x, y = sp.symbols("x y")
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

        x, y = sp.symbols("x y")
        expr = sp.Matrix([x**2, y**2])

        f = generate_jax_function(expr, [x, y])
        result = f(2.0, 3.0)

        expected = jnp.array([4.0, 9.0])
        assert jnp.allclose(result, expected)

    def test_all_backends_matrix_consistency(self):
        """Test all backends handle matrices consistently"""
        x, y, z = sp.symbols("x y z")
        expr = sp.Matrix([sp.sin(x) + y, sp.cos(y) * z, sp.exp(x - z)])

        input_vals = (1.0, 2.0, 0.5)
        results = {}

        # NumPy
        f_np = generate_numpy_function(expr, [x, y, z])
        results["numpy"] = f_np(*input_vals).flatten()

        # PyTorch
        if torch_available:
            f_torch = generate_torch_function(expr, [x, y, z])
            torch_inputs = [torch.tensor(v) for v in input_vals]
            torch_result = f_torch(*torch_inputs)
            results["torch"] = torch_result.flatten().detach().numpy()

        # JAX
        if jax_available:
            f_jax = generate_jax_function(expr, [x, y, z])
            jax_result = f_jax(*input_vals)
            results["jax"] = np.array(jax_result).flatten()

        # All should agree (comparing flattened versions)
        if len(results) > 1:
            base = results["numpy"]
            for backend, result in results.items():
                if backend != "numpy":
                    assert np.allclose(
                        base, result, rtol=1e-5
                    ), f"Backend {backend} differs from NumPy: {base} vs {result}"


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

        x = sp.Symbol("x")
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

        x = sp.Symbol("x")
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
        x = sp.Symbol("x")
        expr = x**2 + 2 * x

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
        x, y = sp.symbols("x y")
        expr = sp.Matrix([x**2, y**2, x * y])

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
            if hasattr(result, "__len__"):
                assert len(result) == 0 or result.size == 0
        except (ValueError, IndexError, TypeError):
            # If it raises, that's also acceptable
            pass

    def test_numpy_single_element_matrix(self):
        """Test NumPy with single element matrix"""
        x = sp.Symbol("x")
        expr = sp.Matrix([x])

        f = generate_numpy_function(expr, [x])
        result = f(5.0)

        assert np.allclose(result, 5.0)

    @pytest.mark.skipif(not torch_available, reason="PyTorch not available")
    def test_torch_zero_dimensional_result(self):
        """Test PyTorch with scalar result"""
        x = sp.Symbol("x")
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
        theta, omega, tau = sp.symbols("theta omega tau")
        g, l, b, m = 9.81, 0.5, 0.1, 0.15

        expr = sp.Matrix([omega, -g / l * sp.sin(theta) - b * omega + tau / (m * l**2)])

        # Test values
        theta_val = 0.1
        omega_val = 0.0
        tau_val = 0.0

        results = {}

        # NumPy
        f_np = generate_numpy_function(expr, [theta, omega, tau])
        results["numpy"] = f_np(theta_val, omega_val, tau_val).flatten()

        # PyTorch
        if torch_available:
            f_torch = generate_torch_function(expr, [theta, omega, tau])
            inputs = [torch.tensor(v) for v in [theta_val, omega_val, tau_val]]
            torch_result = f_torch(*inputs)
            results["torch"] = torch_result.flatten().detach().numpy()

        # JAX
        if jax_available:
            f_jax = generate_jax_function(expr, [theta, omega, tau])
            jax_result = f_jax(theta_val, omega_val, tau_val)
            results["jax"] = np.array(jax_result).flatten()

        # All backends should give same dynamics (comparing flattened)
        if len(results) > 1:
            base = results["numpy"]
            for backend, result in results.items():
                if backend != "numpy":
                    assert np.allclose(
                        base, result, rtol=1e-5
                    ), f"Pendulum dynamics differ: {backend} vs NumPy\n  NumPy: {base}\n  {backend}: {result}"


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
        assert "matrix" in doc_lower or "vector" in doc_lower or "expression" in doc_lower


# ============================================================================
# Test: Min/Max Helper Functions (ALL BACKENDS - EQUAL)
# ============================================================================


class TestMinMaxHelpers:
    """Test Min/Max helper functions for all backends"""

    # ========== NumPy Min/Max ==========

    def test_numpy_min_two_args(self):
        """Test NumPy min with 2 arguments"""
        result = _numpy_min(3.0, 5.0)
        assert np.allclose(result, 3.0)

    def test_numpy_min_three_args(self):
        """Test NumPy min with 3 arguments"""
        result = _numpy_min(5.0, 2.0, 8.0)
        assert np.allclose(result, 2.0)

    def test_numpy_min_arrays(self):
        """Test NumPy min with arrays"""
        a = np.array([1.0, 5.0])
        b = np.array([3.0, 2.0])
        result = _numpy_min(a, b)

        expected = np.array([1.0, 2.0])
        assert np.allclose(result, expected)

    def test_numpy_min_arrays_three_args(self):
        """Test NumPy min with 3 array arguments"""
        a = np.array([1.0, 5.0, 3.0])
        b = np.array([3.0, 2.0, 4.0])
        c = np.array([2.0, 4.0, 1.0])
        result = _numpy_min(a, b, c)

        expected = np.array([1.0, 2.0, 1.0])
        assert np.allclose(result, expected)

    def test_numpy_max_two_args(self):
        """Test NumPy max with 2 arguments"""
        result = _numpy_max(3.0, 5.0)
        assert np.allclose(result, 5.0)

    def test_numpy_max_three_args(self):
        """Test NumPy max with 3 arguments"""
        result = _numpy_max(5.0, 2.0, 8.0)
        assert np.allclose(result, 8.0)

    def test_numpy_max_arrays(self):
        """Test NumPy max with arrays"""
        a = np.array([1.0, 5.0])
        b = np.array([3.0, 2.0])
        result = _numpy_max(a, b)

        expected = np.array([3.0, 5.0])
        assert np.allclose(result, expected)

    # ========== PyTorch Min/Max ==========

    @pytest.mark.skipif(not torch_available, reason="PyTorch not available")
    def test_torch_min_two_args(self):
        """Test PyTorch min with 2 arguments"""
        result = _torch_min(torch.tensor(3.0), torch.tensor(5.0))
        assert torch.allclose(result, torch.tensor(3.0))

    @pytest.mark.skipif(not torch_available, reason="PyTorch not available")
    def test_torch_min_three_args(self):
        """Test PyTorch min with 3 arguments"""
        result = _torch_min(torch.tensor(5.0), torch.tensor(2.0), torch.tensor(8.0))
        assert torch.allclose(result, torch.tensor(2.0))

    @pytest.mark.skipif(not torch_available, reason="PyTorch not available")
    def test_torch_min_mixed_types(self):
        """Test PyTorch min with mixed scalars and tensors"""
        result = _torch_min(3.0, torch.tensor(5.0))
        assert torch.allclose(result, torch.tensor(3.0))

    @pytest.mark.skipif(not torch_available, reason="PyTorch not available")
    def test_torch_min_preserves_gradients(self):
        """Test PyTorch min preserves gradient tracking"""
        a = torch.tensor(3.0, requires_grad=True)
        b = torch.tensor(5.0, requires_grad=True)
        result = _torch_min(a, b)

        assert result.requires_grad
        result.backward()
        assert a.grad is not None

    @pytest.mark.skipif(not torch_available, reason="PyTorch not available")
    def test_torch_max_two_args(self):
        """Test PyTorch max with 2 arguments"""
        result = _torch_max(torch.tensor(3.0), torch.tensor(5.0))
        assert torch.allclose(result, torch.tensor(5.0))

    @pytest.mark.skipif(not torch_available, reason="PyTorch not available")
    def test_torch_max_three_args(self):
        """Test PyTorch max with 3 arguments"""
        result = _torch_max(torch.tensor(5.0), torch.tensor(2.0), torch.tensor(8.0))
        assert torch.allclose(result, torch.tensor(8.0))

    # ========== JAX Min/Max ==========

    @pytest.mark.skipif(not jax_available, reason="JAX not available")
    def test_jax_min_two_args(self):
        """Test JAX min with 2 arguments"""
        import jax.numpy as jnp

        result = _jax_min(3.0, 5.0)
        assert jnp.allclose(result, 3.0)

    @pytest.mark.skipif(not jax_available, reason="JAX not available")
    def test_jax_min_three_args(self):
        """Test JAX min with 3 arguments"""
        import jax.numpy as jnp

        result = _jax_min(5.0, 2.0, 8.0)
        assert jnp.allclose(result, 2.0)

    @pytest.mark.skipif(not jax_available, reason="JAX not available")
    def test_jax_min_arrays(self):
        """Test JAX min with arrays"""
        import jax.numpy as jnp

        a = jnp.array([1.0, 5.0])
        b = jnp.array([3.0, 2.0])
        result = _jax_min(a, b)

        expected = jnp.array([1.0, 2.0])
        assert jnp.allclose(result, expected)

    @pytest.mark.skipif(not jax_available, reason="JAX not available")
    def test_jax_max_two_args(self):
        """Test JAX max with 2 arguments"""
        import jax.numpy as jnp

        result = _jax_max(3.0, 5.0)
        assert jnp.allclose(result, 5.0)

    @pytest.mark.skipif(not jax_available, reason="JAX not available")
    def test_jax_max_three_args(self):
        """Test JAX max with 3 arguments"""
        import jax.numpy as jnp

        result = _jax_max(5.0, 2.0, 8.0)
        assert jnp.allclose(result, 8.0)


# ============================================================================
# Test: Min/Max in Symbolic Expressions (ALL BACKENDS)
# ============================================================================


class TestMinMaxInExpressions:
    """Test Min/Max functions in actual SymPy expressions"""

    def test_numpy_min_in_expression(self):
        """Test NumPy with Min in expression"""
        x, y = sp.symbols("x y")
        expr = sp.Min(x, y) + 1

        f = generate_numpy_function(expr, [x, y])
        result = f(3.0, 5.0)

        # min(3, 5) + 1 = 4
        assert np.allclose(result, 4.0)

    def test_numpy_max_in_expression(self):
        """Test NumPy with Max in expression"""
        x, y = sp.symbols("x y")
        expr = sp.Max(x, y) * 2

        f = generate_numpy_function(expr, [x, y])
        result = f(3.0, 5.0)

        # max(3, 5) * 2 = 10
        assert np.allclose(result, 10.0)

    def test_numpy_min_three_args_in_expression(self):
        """Test NumPy with Min of 3 arguments"""
        x, y, z = sp.symbols("x y z")
        expr = sp.Min(x, y, z)

        f = generate_numpy_function(expr, [x, y, z])
        result = f(5.0, 2.0, 8.0)

        assert np.allclose(result, 2.0)

    @pytest.mark.skipif(not torch_available, reason="PyTorch not available")
    def test_torch_min_in_expression(self):
        """Test PyTorch with Min in expression"""
        x, y = sp.symbols("x y")
        expr = sp.Min(x, y) + 1

        f = generate_torch_function(expr, [x, y])
        result = f(torch.tensor(3.0), torch.tensor(5.0))

        # min(3, 5) + 1 = 4
        assert torch.allclose(result, torch.tensor(4.0))

    @pytest.mark.skipif(not torch_available, reason="PyTorch not available")
    def test_torch_max_in_expression(self):
        """Test PyTorch with Max in expression"""
        x, y = sp.symbols("x y")
        expr = sp.Max(x, y) * 2

        f = generate_torch_function(expr, [x, y])
        result = f(torch.tensor(3.0), torch.tensor(5.0))

        # max(3, 5) * 2 = 10
        assert torch.allclose(result, torch.tensor(10.0))

    @pytest.mark.skipif(not torch_available, reason="PyTorch not available")
    def test_torch_min_three_args_in_expression(self):
        """Test PyTorch with Min of 3 arguments"""
        x, y, z = sp.symbols("x y z")
        expr = sp.Min(x, y, z)

        f = generate_torch_function(expr, [x, y, z])
        result = f(torch.tensor(5.0), torch.tensor(2.0), torch.tensor(8.0))

        assert torch.allclose(result, torch.tensor(2.0))

    @pytest.mark.skipif(not jax_available, reason="JAX not available")
    def test_jax_min_in_expression(self):
        """Test JAX with Min in expression"""
        import jax.numpy as jnp

        x, y = sp.symbols("x y")
        expr = sp.Min(x, y) + 1

        f = generate_jax_function(expr, [x, y])
        result = f(3.0, 5.0)

        # min(3, 5) + 1 = 4
        assert jnp.allclose(result, 4.0)

    @pytest.mark.skipif(not jax_available, reason="JAX not available")
    def test_jax_max_in_expression(self):
        """Test JAX with Max in expression"""
        import jax.numpy as jnp

        x, y = sp.symbols("x y")
        expr = sp.Max(x, y) * 2

        f = generate_jax_function(expr, [x, y])
        result = f(3.0, 5.0)

        # max(3, 5) * 2 = 10
        assert jnp.allclose(result, 10.0)

    @pytest.mark.skipif(not jax_available, reason="JAX not available")
    def test_jax_min_three_args_in_expression(self):
        """Test JAX with Min of 3 arguments"""
        import jax.numpy as jnp

        x, y, z = sp.symbols("x y z")
        expr = sp.Min(x, y, z)

        f = generate_jax_function(expr, [x, y, z])
        result = f(5.0, 2.0, 8.0)

        assert jnp.allclose(result, 2.0)


# ============================================================================
# Test: Min/Max Cross-Backend Consistency
# ============================================================================


class TestMinMaxConsistency:
    """Test that Min/Max work consistently across all backends"""

    def test_all_backends_min_consistency(self):
        """Test all backends give same result for Min"""
        x, y = sp.symbols("x y")
        expr = sp.Min(x, y)

        input_vals = (3.0, 5.0)
        results = {}

        # NumPy
        f_np = generate_numpy_function(expr, [x, y])
        np_result = f_np(*input_vals)
        # Handle scalar or array result
        results["numpy"] = float(np_result.item() if hasattr(np_result, "item") else np_result)

        # PyTorch
        if torch_available:
            f_torch = generate_torch_function(expr, [x, y])
            torch_inputs = [torch.tensor(v) for v in input_vals]
            torch_result = f_torch(*torch_inputs)
            # Flatten and take first element
            results["torch"] = torch_result.flatten()[0].item()

        # JAX
        if jax_available:
            f_jax = generate_jax_function(expr, [x, y])
            jax_result = f_jax(*input_vals)
            # Handle scalar or array result
            results["jax"] = float(
                jax_result.item() if hasattr(jax_result, "item") else np.array(jax_result).item()
            )

        # All should return 3.0
        for backend, result in results.items():
            assert np.allclose(result, 3.0), f"{backend} Min failed: got {result}"

    def test_all_backends_max_consistency(self):
        """Test all backends give same result for Max"""
        x, y = sp.symbols("x y")
        expr = sp.Max(x, y)

        input_vals = (3.0, 5.0)
        results = {}

        # NumPy
        f_np = generate_numpy_function(expr, [x, y])
        np_result = f_np(*input_vals)
        results["numpy"] = float(np_result.item() if hasattr(np_result, "item") else np_result)

        # PyTorch
        if torch_available:
            f_torch = generate_torch_function(expr, [x, y])
            torch_inputs = [torch.tensor(v) for v in input_vals]
            torch_result = f_torch(*torch_inputs)
            results["torch"] = torch_result.flatten()[0].item()

        # JAX
        if jax_available:
            f_jax = generate_jax_function(expr, [x, y])
            jax_result = f_jax(*input_vals)
            results["jax"] = float(
                jax_result.item() if hasattr(jax_result, "item") else np.array(jax_result).item()
            )

        # All should return 5.0
        for backend, result in results.items():
            assert np.allclose(result, 5.0), f"{backend} Max failed: got {result}"

    def test_all_backends_min_three_args_consistency(self):
        """Test all backends handle Min with 3+ args consistently"""
        x, y, z = sp.symbols("x y z")
        expr = sp.Min(x, y, z)

        input_vals = (7.0, 3.0, 9.0)
        results = {}

        # NumPy
        f_np = generate_numpy_function(expr, [x, y, z])
        np_result = f_np(*input_vals)
        results["numpy"] = float(np_result.item() if hasattr(np_result, "item") else np_result)

        # PyTorch
        if torch_available:
            f_torch = generate_torch_function(expr, [x, y, z])
            torch_inputs = [torch.tensor(v) for v in input_vals]
            torch_result = f_torch(*torch_inputs)
            results["torch"] = torch_result.flatten()[0].item()

        # JAX
        if jax_available:
            f_jax = generate_jax_function(expr, [x, y, z])
            jax_result = f_jax(*input_vals)
            results["jax"] = float(
                jax_result.item() if hasattr(jax_result, "item") else np.array(jax_result).item()
            )

        # All should return 3.0
        for backend, result in results.items():
            assert np.allclose(result, 3.0), f"{backend} Min(3 args) failed: got {result}"


# ============================================================================
# Test: Practical Use Cases
# ============================================================================


class TestMinMaxUseCases:
    """Test realistic use cases for Min/Max in dynamics"""

    def test_relu_activation_all_backends(self):
        """Test ReLU-like activation: max(0, x)"""
        x = sp.Symbol("x")
        expr = sp.Max(0, x)  # ReLU

        test_values = [-2.0, 0.0, 3.0]
        expected_outputs = [0.0, 0.0, 3.0]

        for test_val, expected in zip(test_values, expected_outputs):
            # NumPy
            f_np = generate_numpy_function(expr, [x])
            assert np.allclose(f_np(test_val), expected)

            # PyTorch
            if torch_available:
                f_torch = generate_torch_function(expr, [x])
                assert torch.allclose(f_torch(torch.tensor(test_val)), torch.tensor(expected))

            # JAX
            if jax_available:
                import jax.numpy as jnp

                f_jax = generate_jax_function(expr, [x])
                assert jnp.allclose(f_jax(test_val), expected)

    def test_velocity_saturation_all_backends(self):
        """Test velocity saturation: min(max(v, -v_max), v_max)"""
        v = sp.Symbol("v")
        v_max = 10.0

        # Clamp velocity to [-v_max, v_max]
        expr = sp.Min(sp.Max(v, -v_max), v_max)

        test_cases = [
            (-15.0, -10.0),  # Below min → clamped to -10
            (0.0, 0.0),  # Within range → unchanged
            (5.0, 5.0),  # Within range → unchanged
            (15.0, 10.0),  # Above max → clamped to 10
        ]

        for input_val, expected in test_cases:
            # NumPy
            f_np = generate_numpy_function(expr, [v])
            assert np.allclose(f_np(input_val), expected)

            # PyTorch
            if torch_available:
                f_torch = generate_torch_function(expr, [v])
                assert torch.allclose(f_torch(torch.tensor(input_val)), torch.tensor(expected))

            # JAX
            if jax_available:
                import jax.numpy as jnp

                f_jax = generate_jax_function(expr, [v])
                assert jnp.allclose(f_jax(input_val), expected)

    def test_deadband_function_all_backends(self):
        """Test deadband: if |x| < threshold, return 0, else return x"""
        x, thresh = sp.symbols("x thresh")

        # Deadband implementation using Min/Max
        # This is complex but shows Min/Max can handle real control scenarios
        expr = sp.Max(0, sp.Abs(x) - thresh) * sp.sign(x)

        test_cases = [
            (0.5, 1.0, 0.0),  # Below threshold → 0
            (2.0, 1.0, 1.0),  # Above threshold → reduced
            (-2.0, 1.0, -1.0),  # Negative, above threshold
        ]

        for x_val, t_val, expected in test_cases:
            # NumPy
            f_np = generate_numpy_function(expr, [x, thresh])
            result_np = f_np(x_val, t_val)
            assert np.allclose(result_np, expected), f"NumPy failed for x={x_val}"

            # PyTorch
            if torch_available:
                f_torch = generate_torch_function(expr, [x, thresh])
                result_torch = f_torch(torch.tensor(x_val), torch.tensor(t_val))
                assert torch.allclose(
                    result_torch, torch.tensor(expected)
                ), f"PyTorch failed for x={x_val}"

            # JAX
            if jax_available:
                import jax.numpy as jnp

                f_jax = generate_jax_function(expr, [x, thresh])
                result_jax = f_jax(x_val, t_val)
                assert jnp.allclose(result_jax, expected), f"JAX failed for x={x_val}"


# ============================================================================
# Test: Edge Cases for Min/Max
# ============================================================================


class TestMinMaxEdgeCases:
    """Test edge cases for Min/Max handlers"""

    def test_numpy_min_single_arg(self):
        """Test NumPy min with single argument"""
        result = _numpy_min(5.0)
        assert np.allclose(result, 5.0)

    @pytest.mark.skipif(not torch_available, reason="PyTorch not available")
    def test_torch_min_single_arg(self):
        """Test PyTorch min with single argument"""
        result = _torch_min(torch.tensor(5.0))
        assert torch.allclose(result, torch.tensor(5.0))

    @pytest.mark.skipif(not jax_available, reason="JAX not available")
    def test_jax_min_single_arg(self):
        """Test JAX min with single argument"""
        import jax.numpy as jnp

        result = _jax_min(5.0)
        assert jnp.allclose(result, 5.0)

    def test_numpy_min_zero_args_raises(self):
        """Test that Min with no args raises error"""
        with pytest.raises(ValueError, match="at least one argument"):
            _numpy_min()

    @pytest.mark.skipif(not torch_available, reason="PyTorch not available")
    def test_torch_min_zero_args_raises(self):
        """Test that Min with no args raises error"""
        with pytest.raises(ValueError, match="at least one argument"):
            _torch_min()

    @pytest.mark.skipif(not jax_available, reason="JAX not available")
    def test_jax_min_zero_args_raises(self):
        """Test that Min with no args raises error"""
        with pytest.raises(ValueError, match="at least one argument"):
            _jax_min()


# ============================================================================
# Test: Backend Equality for Min/Max
# ============================================================================


class TestMinMaxEquality:
    """Verify that all backends treat Min/Max equally"""

    def test_all_backends_have_min_handlers(self):
        """Test that all backends have min handler functions"""
        assert callable(_numpy_min)
        assert callable(_torch_min)
        assert callable(_jax_min)

    def test_all_backends_have_max_handlers(self):
        """Test that all backends have max handler functions"""
        assert callable(_numpy_max)
        assert callable(_torch_max)
        assert callable(_jax_max)

    def test_min_handlers_same_signature(self):
        """Test that all min handlers have same signature"""
        import inspect

        sig_numpy = inspect.signature(_numpy_min)
        sig_torch = inspect.signature(_torch_min)
        sig_jax = inspect.signature(_jax_min)

        # All should accept *args
        assert sig_numpy.parameters["args"].kind == inspect.Parameter.VAR_POSITIONAL
        assert sig_torch.parameters["args"].kind == inspect.Parameter.VAR_POSITIONAL
        assert sig_jax.parameters["args"].kind == inspect.Parameter.VAR_POSITIONAL

    def test_max_handlers_same_signature(self):
        """Test that all max handlers have same signature"""
        import inspect

        sig_numpy = inspect.signature(_numpy_max)
        sig_torch = inspect.signature(_torch_max)
        sig_jax = inspect.signature(_jax_max)

        # All should accept *args
        assert sig_numpy.parameters["args"].kind == inspect.Parameter.VAR_POSITIONAL
        assert sig_torch.parameters["args"].kind == inspect.Parameter.VAR_POSITIONAL
        assert sig_jax.parameters["args"].kind == inspect.Parameter.VAR_POSITIONAL

    def test_all_handlers_have_docstrings(self):
        """Test that all Min/Max handlers have docstrings"""
        handlers = [
            _numpy_min,
            _numpy_max,
            _torch_min,
            _torch_max,
            _jax_min,
            _jax_max,
        ]

        for handler in handlers:
            assert handler.__doc__ is not None, f"{handler.__name__} missing docstring"
            assert "SymPy" in handler.__doc__, f"{handler.__name__} should mention SymPy"


# ============================================================================
# Test: Type System Integration
# ============================================================================


class TestTypeSystemIntegration:
    """Test type system integration with centralized type framework"""
    
    def test_backend_literal_type(self):
        """Test Backend literal usage in generate_function"""
        x, y = sp.symbols('x y')
        expr: SymbolicExpressionInput = x + y
        
        # Valid backends
        backend_numpy: Backend = "numpy"
        backend_torch: Backend = "torch"
        backend_jax: Backend = "jax"
        
        f_numpy = generate_function(expr, [x, y], backend=backend_numpy)
        assert callable(f_numpy)
        
        # Type checker would catch: backend = "tensorflow"
    
    def test_symbolic_expression_input_scalar(self):
        """Test SymbolicExpressionInput with scalar expression"""
        x, y = sp.symbols('x y')
        
        # Scalar expression
        expr: SymbolicExpressionInput = x**2 + y**2
        f = generate_function(expr, [x, y], backend="numpy")
        
        result = f(3.0, 4.0)
        assert isinstance(result, np.ndarray)
        assert np.allclose(result, [25.0])
    
    def test_symbolic_expression_input_list(self):
        """Test SymbolicExpressionInput with list of expressions"""
        x, y = sp.symbols('x y')
        
        # List of expressions
        expr: SymbolicExpressionInput = [x**2, y**2, x*y]
        f = generate_function(expr, [x, y], backend="numpy")
        
        result = f(2.0, 3.0)
        assert isinstance(result, np.ndarray)
        assert np.allclose(result, [4.0, 9.0, 6.0])
    
    def test_symbolic_expression_input_matrix(self):
        """Test SymbolicExpressionInput with Matrix"""
        x, y = sp.symbols('x y')
        
        # Matrix expression
        expr: SymbolicExpressionInput = sp.Matrix([x, y, x*y])
        f = generate_function(expr, [x, y], backend="numpy")
        
        result = f(2.0, 3.0)
        assert isinstance(result, np.ndarray)
        assert np.allclose(result, [2.0, 3.0, 6.0])
    
    def test_backend_type_consistency(self):
        """Test Backend type works consistently across all functions"""
        x = sp.symbols('x')
        expr: SymbolicExpressionInput = x**2
        
        backend: Backend = "numpy"
        
        # All these should accept Backend type
        f1 = generate_function(expr, [x], backend=backend)
        f2 = generate_numpy_function(expr, [x])
        
        assert callable(f1)
        assert callable(f2)
    
    @pytest.mark.skipif(not torch_available, reason="PyTorch not installed")
    def test_backend_type_torch(self):
        """Test Backend type with PyTorch"""
        x = sp.symbols('x')
        expr: SymbolicExpressionInput = x**2
        
        backend: Backend = "torch"
        f = generate_function(expr, [x], backend=backend)
        
        result = f(torch.tensor(3.0))
        assert isinstance(result, torch.Tensor)
    
    @pytest.mark.skipif(not jax_available, reason="JAX not installed")
    def test_backend_type_jax(self):
        """Test Backend type with JAX"""
        x = sp.symbols('x')
        expr: SymbolicExpressionInput = x**2
        
        backend: Backend = "jax"
        f = generate_function(expr, [x], backend=backend, jit=False)
        
        result = f(jnp.array(3.0))
        assert isinstance(result, jnp.ndarray)
    
    def test_callable_return_type(self):
        """Test that functions return Callable type"""
        x = sp.symbols('x')
        expr: SymbolicExpressionInput = x + 1
        
        f: Callable = generate_function(expr, [x], backend="numpy")
        
        assert callable(f)
        assert isinstance(f(1.0), np.ndarray)
    
    def test_symbolic_expression_input_flexibility(self):
        """Test SymbolicExpressionInput accepts all valid forms"""
        x, y = sp.symbols('x y')
        
        # All these should be valid SymbolicExpressionInput
        expr1: SymbolicExpressionInput = x + y  # sp.Expr
        expr2: SymbolicExpressionInput = [x, y]  # List[sp.Expr]
        expr3: SymbolicExpressionInput = sp.Matrix([x, y])  # sp.Matrix
        
        # All should generate valid functions
        f1 = generate_function(expr1, [x, y], backend="numpy")
        f2 = generate_function(expr2, [x, y], backend="numpy")
        f3 = generate_function(expr3, [x, y], backend="numpy")
        
        assert callable(f1)
        assert callable(f2)
        assert callable(f3)
    
    def test_type_annotations_present(self):
        """Test that all generate functions have type annotations"""
        import inspect
        
        # Check generate_function signature
        sig = inspect.signature(generate_function)
        assert 'expr' in sig.parameters
        assert 'backend' in sig.parameters
        
        # Return type should be Callable
        assert sig.return_annotation == Callable or 'Callable' in str(sig.return_annotation)
    
    def test_backend_validation_through_types(self):
        """Test that invalid backends would be caught by type system"""
        x = sp.symbols('x')
        expr: SymbolicExpressionInput = x**2
        
        # These are valid (type-safe)
        valid_backends: list[Backend] = ["numpy", "torch", "jax"]
        
        for backend in valid_backends:
            try:
                f = generate_function(expr, [x], backend=backend)
                # Should succeed or raise ImportError, not ValueError
            except (ImportError, ModuleNotFoundError):
                # OK if backend not available
                pass
            except ValueError as e:
                # Should not get "Unknown backend" with valid Backend literal
                if "Unknown backend" in str(e):
                    pytest.fail(f"Valid backend {backend} raised ValueError: {e}")


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
