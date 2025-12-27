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
Comprehensive unit tests for CodeGenerator

Tests cover:
1. Dynamics function generation
2. Output function generation
3. Jacobian generation
4. Caching behavior
5. Compilation
6. Cache management
7. Information retrieval
"""

import numpy as np
import pytest
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

# Import from centralized type system
from src.types.backends import Backend

from src.systems.base.utils.code_generator import CodeGenerator

# ============================================================================
# Mock System for Testing
# ============================================================================


class MockSystem:
    """Mock dynamical system for testing"""

    def __init__(self, with_output=False):
        # Define symbols
        x = sp.symbols("x", real=True)
        u = sp.symbols("u", real=True)
        a = sp.symbols("a", real=True, positive=True)

        # System definition
        self.state_vars = [x]
        self.control_vars = [u]
        self._f_sym = sp.Matrix([-a * x + u])
        self.parameters = {a: 2.0}

        # Output function (optional)
        if with_output:
            self._h_sym = sp.Matrix([x**2])
        else:
            self._h_sym = None

    def substitute_parameters(self, expr):
        """Substitute parameter values"""
        return expr.subs(self.parameters)


class MockSecondOrderSystem:
    """Mock second-order system for testing Jacobians"""

    def __init__(self):
        q, q_dot = sp.symbols("q q_dot", real=True)
        u = sp.symbols("u", real=True)
        k, c = sp.symbols("k c", real=True, positive=True)

        self.state_vars = [q, q_dot]
        self.control_vars = [u]
        self._f_sym = sp.Matrix([-k * q - c * q_dot + u])
        self._h_sym = sp.Matrix([q])  # Output is position
        self.parameters = {k: 10.0, c: 0.5}

    def substitute_parameters(self, expr):
        return expr.subs(self.parameters)


# ============================================================================
# Test Class 1: Dynamics Function Generation
# ============================================================================


class TestDynamicsGeneration:
    """Test dynamics function generation"""

    def test_generate_dynamics_numpy(self):
        """Test generating NumPy dynamics function"""
        system = MockSystem()
        code_gen = CodeGenerator(system)

        f_numpy = code_gen.generate_dynamics("numpy")

        assert callable(f_numpy)
        assert code_gen._f_funcs["numpy"] is not None

    @pytest.mark.skipif(not torch_available, reason="PyTorch not installed")
    def test_generate_dynamics_torch(self):
        """Test generating PyTorch dynamics function"""
        system = MockSystem()
        code_gen = CodeGenerator(system)

        f_torch = code_gen.generate_dynamics("torch")

        assert callable(f_torch)
        assert code_gen._f_funcs["torch"] is not None

    @pytest.mark.skipif(not jax_available, reason="JAX not installed")
    def test_generate_dynamics_jax(self):
        """Test generating JAX dynamics function"""
        system = MockSystem()
        code_gen = CodeGenerator(system)

        f_jax = code_gen.generate_dynamics("jax", jit=True)

        assert callable(f_jax)
        assert code_gen._f_funcs["jax"] is not None

    def test_dynamics_caching(self):
        """Test that generated functions are cached"""
        system = MockSystem()
        code_gen = CodeGenerator(system)

        # Generate twice
        f1 = code_gen.generate_dynamics("numpy")
        f2 = code_gen.generate_dynamics("numpy")

        # Should be same object (cached)
        assert f1 is f2

    def test_get_dynamics_before_generation(self):
        """Test getting dynamics before generation returns None"""
        system = MockSystem()
        code_gen = CodeGenerator(system)

        f = code_gen.get_dynamics("numpy")

        assert f is None

    def test_get_dynamics_after_generation(self):
        """Test getting dynamics after generation"""
        system = MockSystem()
        code_gen = CodeGenerator(system)

        code_gen.generate_dynamics("numpy")
        f = code_gen.get_dynamics("numpy")

        assert f is not None
        assert callable(f)


# ============================================================================
# Test Class 2: Output Function Generation
# ============================================================================


class TestOutputGeneration:
    """Test output function generation"""

    def test_generate_output_numpy(self):
        """Test generating NumPy output function"""
        system = MockSystem(with_output=True)
        code_gen = CodeGenerator(system)

        h_numpy = code_gen.generate_output("numpy")

        assert callable(h_numpy)
        assert code_gen._h_funcs["numpy"] is not None

    def test_generate_output_no_custom_output(self):
        """Test generating output when no custom output defined"""
        system = MockSystem(with_output=False)
        code_gen = CodeGenerator(system)

        h_numpy = code_gen.generate_output("numpy")

        assert h_numpy is None

    @pytest.mark.skipif(not torch_available, reason="PyTorch not installed")
    def test_generate_output_torch(self):
        """Test generating PyTorch output function"""
        system = MockSystem(with_output=True)
        code_gen = CodeGenerator(system)

        h_torch = code_gen.generate_output("torch")

        assert callable(h_torch)

    def test_output_caching(self):
        """Test that output functions are cached"""
        system = MockSystem(with_output=True)
        code_gen = CodeGenerator(system)

        h1 = code_gen.generate_output("numpy")
        h2 = code_gen.generate_output("numpy")

        assert h1 is h2

    def test_get_output_before_generation(self):
        """Test getting output before generation"""
        system = MockSystem(with_output=True)
        code_gen = CodeGenerator(system)

        h = code_gen.get_output("numpy")

        assert h is None

    def test_get_output_after_generation(self):
        """Test getting output after generation"""
        system = MockSystem(with_output=True)
        code_gen = CodeGenerator(system)

        code_gen.generate_output("numpy")
        h = code_gen.get_output("numpy")

        assert h is not None
        assert callable(h)


# ============================================================================
# Test Class 3: Jacobian Generation
# ============================================================================


class TestJacobianGeneration:
    """Test Jacobian function generation"""

    def test_generate_dynamics_jacobians_numpy(self):
        """Test generating A and B matrices"""
        system = MockSecondOrderSystem()
        code_gen = CodeGenerator(system)

        A_func, B_func = code_gen.generate_dynamics_jacobians("numpy")

        assert callable(A_func)
        assert callable(B_func)
        assert code_gen._A_funcs["numpy"] is not None
        assert code_gen._B_funcs["numpy"] is not None

    @pytest.mark.skipif(not torch_available, reason="PyTorch not installed")
    def test_generate_dynamics_jacobians_torch(self):
        """Test generating Jacobians for torch"""
        system = MockSecondOrderSystem()
        code_gen = CodeGenerator(system)

        A_func, B_func = code_gen.generate_dynamics_jacobians("torch")

        assert callable(A_func)
        assert callable(B_func)

    def test_jacobian_caching(self):
        """Test that Jacobians are cached"""
        system = MockSecondOrderSystem()
        code_gen = CodeGenerator(system)

        A1, B1 = code_gen.generate_dynamics_jacobians("numpy")
        A2, B2 = code_gen.generate_dynamics_jacobians("numpy")

        assert A1 is A2
        assert B1 is B2

    def test_symbolic_jacobian_computed_once(self):
        """Test that symbolic Jacobian is only computed once"""
        system = MockSecondOrderSystem()
        code_gen = CodeGenerator(system)

        # Generate for first backend
        code_gen.generate_dynamics_jacobians("numpy")
        A_sym_1 = code_gen._A_sym_cache

        # Generate for second backend
        if torch_available:
            code_gen.generate_dynamics_jacobians("torch")
            A_sym_2 = code_gen._A_sym_cache

            # Should be same object (computed once)
            assert A_sym_1 is A_sym_2

    def test_generate_observation_jacobian(self):
        """Test generating C matrix"""
        system = MockSecondOrderSystem()
        code_gen = CodeGenerator(system)

        C_func = code_gen.generate_observation_jacobian("numpy")

        assert callable(C_func)
        assert code_gen._C_funcs["numpy"] is not None

    def test_observation_jacobian_no_output(self):
        """Test observation Jacobian when no custom output"""
        system = MockSystem(with_output=False)
        code_gen = CodeGenerator(system)

        C_func = code_gen.generate_observation_jacobian("numpy")

        assert C_func is None

    def test_get_jacobians(self):
        """Test getting all Jacobians at once"""
        system = MockSecondOrderSystem()
        code_gen = CodeGenerator(system)

        # Before generation
        A, B, C = code_gen.get_jacobians("numpy")
        assert A is None and B is None and C is None

        # After generation
        code_gen.generate_dynamics_jacobians("numpy")
        code_gen.generate_observation_jacobian("numpy")

        A, B, C = code_gen.get_jacobians("numpy")
        assert A is not None
        assert B is not None
        assert C is not None


# ============================================================================
# Test Class 4: Compilation
# ============================================================================


class TestCompilation:
    """Test compile_all method"""

    def test_compile_all_single_backend(self):
        """Test compiling single backend"""
        system = MockSystem(with_output=True)
        code_gen = CodeGenerator(system)

        timings = code_gen.compile_all(backends=["numpy"], verbose=False)

        assert "numpy" in timings
        assert "f" in timings["numpy"]
        assert "h" in timings["numpy"]
        assert timings["numpy"]["f"] is not None

    def test_compile_all_multiple_backends(self):
        """Test compiling multiple backends"""
        system = MockSystem()
        code_gen = CodeGenerator(system)

        backends = ["numpy"]
        if torch_available:
            backends.append("torch")
        if jax_available:
            backends.append("jax")

        timings = code_gen.compile_all(backends=backends, verbose=False)

        for backend in backends:
            assert backend in timings
            assert timings[backend]["f"] is not None

    def test_compile_with_jacobians(self):
        """Test compiling with Jacobians"""
        system = MockSecondOrderSystem()
        code_gen = CodeGenerator(system)

        timings = code_gen.compile_all(backends=["numpy"], include_jacobians=True, verbose=False)

        assert "A_B" in timings["numpy"]
        assert "C" in timings["numpy"]

    def test_compile_verbose(self, capsys):
        """Test verbose compilation output"""
        system = MockSystem()
        code_gen = CodeGenerator(system)

        code_gen.compile_all(backends=["numpy"], verbose=True)

        captured = capsys.readouterr()
        assert "Compiling numpy" in captured.out
        assert "f:" in captured.out


# ============================================================================
# Test Class 5: Cache Management
# ============================================================================


class TestCacheManagement:
    """Test cache management"""

    def test_reset_cache_single_backend(self):
        """Test resetting cache for single backend"""
        system = MockSystem()
        code_gen = CodeGenerator(system)

        # Generate and verify cached
        code_gen.generate_dynamics("numpy")
        assert code_gen._f_funcs["numpy"] is not None

        # Reset
        code_gen.reset_cache(["numpy"])

        assert code_gen._f_funcs["numpy"] is None

    def test_reset_cache_all_backends(self):
        """Test resetting all caches"""
        system = MockSystem()
        code_gen = CodeGenerator(system)

        # Generate for multiple backends
        code_gen.generate_dynamics("numpy")
        if torch_available:
            code_gen.generate_dynamics("torch")

        # Reset all
        code_gen.reset_cache()

        assert code_gen._f_funcs["numpy"] is None
        assert code_gen._f_funcs["torch"] is None
        assert code_gen._f_funcs["jax"] is None

    def test_reset_cache_preserves_other_backends(self):
        """Test that resetting one backend preserves others"""
        system = MockSystem()
        code_gen = CodeGenerator(system)

        # Generate for two backends
        code_gen.generate_dynamics("numpy")
        if torch_available:
            code_gen.generate_dynamics("torch")

            # Reset only torch
            code_gen.reset_cache(["torch"])

            # NumPy should still be cached
            assert code_gen._f_funcs["numpy"] is not None
            assert code_gen._f_funcs["torch"] is None


# ============================================================================
# Test Class 6: Information Retrieval
# ============================================================================


class TestInformationRetrieval:
    """Test information and status methods"""

    def test_is_compiled_before_generation(self):
        """Test is_compiled before generation"""
        system = MockSystem()
        code_gen = CodeGenerator(system)

        status = code_gen.is_compiled("numpy")

        assert status["f"] is False
        assert status["h"] is False
        assert status["A"] is False

    def test_is_compiled_after_generation(self):
        """Test is_compiled after generation"""
        system = MockSystem(with_output=True)
        code_gen = CodeGenerator(system)

        code_gen.generate_dynamics("numpy")
        code_gen.generate_output("numpy")

        status = code_gen.is_compiled("numpy")

        assert status["f"] is True
        assert status["h"] is True
        assert status["A"] is False  # Not generated yet

    def test_get_info(self):
        """Test get_info method"""
        system = MockSystem()
        code_gen = CodeGenerator(system)

        code_gen.generate_dynamics("numpy")

        info = code_gen.get_info()

        assert "numpy" in info
        assert "torch" in info
        assert "jax" in info
        assert info["numpy"]["f"] is True
        assert info["torch"]["f"] is False


# ============================================================================
# Test Class 7: String Representations
# ============================================================================


class TestStringRepresentations:
    """Test __repr__ and __str__ methods"""

    def test_repr(self):
        """Test __repr__ output"""
        system = MockSystem()
        code_gen = CodeGenerator(system)

        repr_str = repr(code_gen)

        assert "CodeGenerator" in repr_str
        assert "compiled_backends" in repr_str

    def test_repr_after_compilation(self):
        """Test __repr__ shows compiled backends"""
        system = MockSystem()
        code_gen = CodeGenerator(system)

        code_gen.generate_dynamics("numpy")
        repr_str = repr(code_gen)

        assert "numpy" in repr_str

    def test_str(self):
        """Test __str__ output"""
        system = MockSystem()
        code_gen = CodeGenerator(system)

        code_gen.generate_dynamics("numpy")
        str_repr = str(code_gen)

        assert "CodeGenerator" in str_repr
        assert "1/3" in str_repr or "2/3" in str_repr or "3/3" in str_repr


# ============================================================================
# Test Class 8: Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests"""

    def test_full_workflow(self):
        """Test complete workflow"""
        system = MockSecondOrderSystem()
        code_gen = CodeGenerator(system)

        # Compile everything
        timings = code_gen.compile_all(backends=["numpy"], include_jacobians=True, verbose=False)

        # Verify all compiled
        status = code_gen.is_compiled("numpy")
        assert status["f"] is True
        assert status["h"] is True
        assert status["A"] is True
        assert status["B"] is True
        assert status["C"] is True

        # Reset and verify cleared
        code_gen.reset_cache(["numpy"])
        status = code_gen.is_compiled("numpy")
        assert status["f"] is False


# ============================================================================
# Test Class 7: Type System Integration
# ============================================================================


class TestTypeSystemIntegration:
    """Test type system integration with centralized type framework"""
    
    def test_backend_literal_in_generate_dynamics(self):
        """Test Backend literal usage in generate_dynamics"""
        system = MockSystem()
        code_gen = CodeGenerator(system)
        
        # Valid backends
        backend_numpy: Backend = "numpy"
        backend_torch: Backend = "torch"
        backend_jax: Backend = "jax"
        
        f_numpy = code_gen.generate_dynamics(backend_numpy)
        assert callable(f_numpy)
        
        # Type checker would catch: backend = "tensorflow"
    
    def test_backend_literal_in_compile_all(self):
        """Test Backend literal in compile_all backends list"""
        system = MockSystem()
        code_gen = CodeGenerator(system)
        
        # Type-safe backend list
        backends: list[Backend] = ["numpy", "torch"]
        
        timings = code_gen.compile_all(backends=backends, verbose=False)
        
        assert "numpy" in timings
        if torch_available:
            assert "torch" in timings
    
    def test_backend_literal_in_reset_cache(self):
        """Test Backend literal in reset_cache"""
        system = MockSystem()
        code_gen = CodeGenerator(system)
        
        # Generate some functions
        code_gen.generate_dynamics("numpy")
        
        # Type-safe reset
        backends: list[Backend] = ["numpy"]
        code_gen.reset_cache(backends=backends)
        
        # Verify cleared
        f = code_gen.get_dynamics("numpy")
        assert f is None
    
    def test_backend_literal_in_get_methods(self):
        """Test Backend literal in getter methods"""
        system = MockSystem()
        code_gen = CodeGenerator(system)
        
        backend: Backend = "numpy"
        
        # Generate
        code_gen.generate_dynamics(backend)
        
        # Get with Backend type
        f = code_gen.get_dynamics(backend)
        assert callable(f)
    
    def test_backend_literal_in_jacobian_methods(self):
        """Test Backend literal in Jacobian methods"""
        system = MockSecondOrderSystem()
        code_gen = CodeGenerator(system)
        
        backend: Backend = "numpy"
        
        # Generate Jacobians
        A_func, B_func = code_gen.generate_dynamics_jacobians(backend)
        
        assert callable(A_func)
        assert callable(B_func)
    
    def test_backend_consistency_across_methods(self):
        """Test Backend type works consistently across all methods"""
        system = MockSystem(with_output=True)
        code_gen = CodeGenerator(system)
        
        backend: Backend = "numpy"
        
        # All these should accept Backend type
        f = code_gen.generate_dynamics(backend)
        h = code_gen.generate_output(backend)
        f_get = code_gen.get_dynamics(backend)
        h_get = code_gen.get_output(backend)
        status = code_gen.is_compiled(backend)
        
        assert callable(f)
        assert callable(h)
        assert callable(f_get)
        assert callable(h_get)
        assert isinstance(status, dict)
    
    @pytest.mark.skipif(not torch_available, reason="PyTorch not installed")
    def test_backend_type_torch(self):
        """Test Backend type with PyTorch backend"""
        system = MockSystem()
        code_gen = CodeGenerator(system)
        
        backend: Backend = "torch"
        f = code_gen.generate_dynamics(backend)
        
        # Test with torch tensor
        x = torch.tensor([1.0])
        u = torch.tensor([0.5])
        result = f(x, u)
        
        assert isinstance(result, torch.Tensor)
    
    @pytest.mark.skipif(not jax_available, reason="JAX not installed")
    def test_backend_type_jax(self):
        """Test Backend type with JAX backend"""
        system = MockSystem()
        code_gen = CodeGenerator(system)
        
        backend: Backend = "jax"
        f = code_gen.generate_dynamics(backend, jit=False)
        
        # Test with jax array
        x = jnp.array([1.0])
        u = jnp.array([0.5])
        result = f(x, u)
        
        assert isinstance(result, jnp.ndarray)
    
    def test_callable_return_types(self):
        """Test that generated functions return callable types"""
        from typing import Callable
        
        system = MockSystem(with_output=True)
        code_gen = CodeGenerator(system)
        
        f: Callable = code_gen.generate_dynamics("numpy")
        h: Callable = code_gen.generate_output("numpy")
        
        assert callable(f)
        assert callable(h)
    
    def test_backend_list_type_in_compile_all(self):
        """Test Optional[List[Backend]] type in compile_all"""
        system = MockSystem()
        code_gen = CodeGenerator(system)
        
        # None should compile all available
        timings1 = code_gen.compile_all(backends=None, verbose=False)
        assert len(timings1) > 0
        
        # Specific list
        backends: list[Backend] = ["numpy"]
        timings2 = code_gen.compile_all(backends=backends, verbose=False)
        assert "numpy" in timings2
    
    def test_type_safety_prevents_invalid_backends(self):
        """Test that type system would catch invalid backends"""
        system = MockSystem()
        code_gen = CodeGenerator(system)
        
        # These are valid (type-safe)
        valid_backends: list[Backend] = ["numpy", "torch", "jax"]
        
        for backend in valid_backends:
            try:
                f = code_gen.generate_dynamics(backend)
                assert callable(f)
            except (ImportError, ModuleNotFoundError):
                # OK if backend not available
                pass
    
    def test_backend_in_cache_keys(self):
        """Test that backend acts as proper cache key"""
        system = MockSystem()
        code_gen = CodeGenerator(system)
        
        backend1: Backend = "numpy"
        backend2: Backend = "torch"
        
        # Generate for numpy
        f1 = code_gen.generate_dynamics(backend1)
        f1_again = code_gen.generate_dynamics(backend1)
        
        # Should be cached (same object)
        assert f1 is f1_again
        
        # Different backend should generate different function
        if torch_available:
            f2 = code_gen.generate_dynamics(backend2)
            assert f2 is not f1
    
    def test_type_annotations_in_code_generator(self):
        """Test that CodeGenerator methods have proper type annotations"""
        import inspect
        
        # Check generate_dynamics signature
        sig = inspect.signature(CodeGenerator.generate_dynamics)
        assert 'backend' in sig.parameters
        
        # Check compile_all signature
        sig = inspect.signature(CodeGenerator.compile_all)
        assert 'backends' in sig.parameters


# ============================================================================
# Run Tests
# ============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
