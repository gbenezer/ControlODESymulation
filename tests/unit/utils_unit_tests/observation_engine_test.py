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
    
from src.types.core import StateVector, OutputVector, OutputMatrix
from src.types.backends import Backend
from src.systems.base.utils.backend_manager import BackendManager
from src.systems.base.utils.code_generator import CodeGenerator
from src.systems.base.utils.observation_engine import ObservationEngine

# ============================================================================
# Mock Systems
# ============================================================================


class MockSystemNoOutput:
    """System without custom output (uses identity)"""

    def __init__(self):
        x = sp.symbols("x", real=True)
        u = sp.symbols("u", real=True)

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
        x1, x2 = sp.symbols("x1 x2", real=True)
        u = sp.symbols("u", real=True)

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

        x: StateVector = np.array([1.0])
        y: OutputVector = engine.evaluate(x, backend="numpy")

        assert isinstance(y, np.ndarray)
        assert np.allclose(y, x)

    @pytest.mark.skipif(not torch_available, reason="PyTorch not installed")
    def test_evaluate_torch_identity(self):
        """Test default output with PyTorch"""
        system = MockSystemNoOutput()
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        engine = ObservationEngine(system, code_gen, backend_mgr)

        x: StateVector = torch.tensor([1.0])
        y: OutputVector = engine.evaluate(x, backend="torch")

        assert isinstance(y, torch.Tensor)
        assert torch.allclose(y, x)

    def test_jacobian_numpy_identity(self):
        """Test observation Jacobian is identity for default output"""
        system = MockSystemNoOutput()
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        engine = ObservationEngine(system, code_gen, backend_mgr)

        x: StateVector = np.array([1.0])
        C: OutputMatrix = engine.compute_jacobian(x, backend="numpy")

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

        x: StateVector = np.array([1.0, 2.0])
        y: OutputVector = engine.evaluate(x, backend="numpy")
        
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

        x: StateVector = torch.tensor([1.0, 2.0])
        y: OutputVector = engine.evaluate(x, backend="torch")

        assert isinstance(y, torch.Tensor)
        assert torch.allclose(y, torch.tensor([1.0, 5.0]))

    @pytest.mark.skipif(not jax_available, reason="JAX not installed")
    def test_evaluate_jax_custom(self):
        """Test custom output with JAX"""
        system = MockSystemCustomOutput()
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        engine = ObservationEngine(system, code_gen, backend_mgr)

        x: StateVector = jnp.array([1.0, 2.0])
        y: OutputVector = engine.evaluate(x, backend="jax")

        assert isinstance(y, jnp.ndarray)
        assert jnp.allclose(y, jnp.array([1.0, 5.0]))

    def test_evaluate_auto_detect(self):
        """Test auto-detection of backend"""
        system = MockSystemCustomOutput()
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        engine = ObservationEngine(system, code_gen, backend_mgr)

        x: StateVector = np.array([1.0, 2.0])
        y: OutputVector = engine.evaluate(x)  # No backend specified

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

        x: StateVector = np.array([1.0, 2.0])
        C: OutputMatrix = engine.compute_jacobian(x, backend="numpy")

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

        x: StateVector = torch.tensor([1.0, 2.0])
        C: OutputMatrix = engine.compute_jacobian(x, backend="torch")

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

        x: StateVector = jnp.array([1.0, 2.0])
        C: OutputMatrix = engine.compute_jacobian(x, backend="jax")

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
        x: StateVector = np.array([1.0, 2.0])
        y: OutputVector = engine.evaluate(x, backend="torch")

        # Should return torch tensor
        assert isinstance(y, torch.Tensor)
        assert torch.allclose(y, torch.tensor([1.0, 5.0]))


# ============================================================================
# Test Class 5: Type System Integration
# ============================================================================


class TestTypeSystemIntegration:
    """Test type system integration and semantic types"""
    
    def test_state_vector_type_annotation(self):
        """Test StateVector type annotation works"""
        system = MockSystemCustomOutput()
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        engine = ObservationEngine(system, code_gen, backend_mgr)
        
        x: StateVector = np.array([1.0, 2.0])
        
        # StateVector is just ArrayLike, but semantically meaningful
        assert isinstance(x, np.ndarray)
    
    def test_output_vector_type_annotation(self):
        """Test OutputVector type annotation works"""
        system = MockSystemCustomOutput()
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        engine = ObservationEngine(system, code_gen, backend_mgr)
        
        x: StateVector = np.array([1.0, 2.0])
        y: OutputVector = engine.evaluate(x, backend="numpy")
        
        assert isinstance(y, np.ndarray)
        assert y.shape == (2,)
    
    def test_output_matrix_type_annotation(self):
        """Test OutputMatrix type annotation works"""
        system = MockSystemCustomOutput()
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        engine = ObservationEngine(system, code_gen, backend_mgr)
        
        x: StateVector = np.array([1.0, 2.0])
        C: OutputMatrix = engine.compute_jacobian(x, backend="numpy")
        
        assert isinstance(C, np.ndarray)
        assert C.shape == (2, 2)  # (ny, nx)
    
    def test_backend_literal_type(self):
        """Test Backend literal usage"""
        system = MockSystemCustomOutput()
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        engine = ObservationEngine(system, code_gen, backend_mgr)
        
        x: StateVector = np.array([1.0, 2.0])
        backend: Backend = "numpy"
        y: OutputVector = engine.evaluate(x, backend=backend)
        
        # Type checker validates only 'numpy', 'torch', 'jax'
        assert isinstance(y, np.ndarray)
    
    def test_state_to_output_flow(self):
        """Test semantic type flow: StateVector -> OutputVector"""
        system = MockSystemCustomOutput()
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        engine = ObservationEngine(system, code_gen, backend_mgr)
        
        x: StateVector = np.array([1.0, 2.0])
        y: OutputVector = engine.evaluate(x)
        
        # Types show intent: state -> output mapping
        assert isinstance(y, np.ndarray)
        assert y.shape == (2,)
    
    def test_state_to_output_matrix_flow(self):
        """Test semantic type flow: StateVector -> OutputMatrix"""
        system = MockSystemCustomOutput()
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        engine = ObservationEngine(system, code_gen, backend_mgr)
        
        x: StateVector = np.array([1.0, 2.0])
        C: OutputMatrix = engine.compute_jacobian(x)
        
        # Types show intent: state -> Jacobian matrix
        assert isinstance(C, np.ndarray)
        assert C.shape == (2, 2)  # (ny, nx)
    
    @pytest.mark.skipif(not torch_available, reason="PyTorch not installed")
    def test_type_consistency_across_backends(self):
        """Test types work consistently across NumPy/PyTorch"""
        system = MockSystemCustomOutput()
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        engine = ObservationEngine(system, code_gen, backend_mgr)
        
        # NumPy
        x_np: StateVector = np.array([1.0, 2.0])
        y_np: OutputVector = engine.evaluate(x_np, backend="numpy")
        C_np: OutputMatrix = engine.compute_jacobian(x_np, backend="numpy")
        
        # PyTorch
        x_torch: StateVector = torch.tensor([1.0, 2.0])
        y_torch: OutputVector = engine.evaluate(x_torch, backend="torch")
        C_torch: OutputMatrix = engine.compute_jacobian(x_torch, backend="torch")
        
        # Same semantic types, different backends
        assert isinstance(y_np, np.ndarray)
        assert isinstance(y_torch, torch.Tensor)
        assert isinstance(C_np, np.ndarray)
        assert isinstance(C_torch, torch.Tensor)
    
    def test_semantic_clarity_over_generic(self):
        """Test semantic types provide clarity over generic ArrayLike"""
        system = MockSystemCustomOutput()
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        engine = ObservationEngine(system, code_gen, backend_mgr)
        
        # Generic type (old style) - unclear purpose
        x_generic: np.ndarray = np.array([1.0, 2.0])
        
        # Semantic type (new style) - clear purpose
        x_semantic: StateVector = np.array([1.0, 2.0])
        
        # Both work, but semantic is clearer about intent
        y_generic = engine.evaluate(x_generic)
        y_semantic: OutputVector = engine.evaluate(x_semantic)
        
        assert np.allclose(y_generic, y_semantic)
    
    def test_output_shape_validation(self):
        """Test OutputVector and OutputMatrix have correct shapes"""
        system = MockSystemCustomOutput()
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        engine = ObservationEngine(system, code_gen, backend_mgr)
        
        x: StateVector = np.array([1.0, 2.0])  # (nx,) = (2,)
        
        # OutputVector should be (ny,) = (2,)
        y: OutputVector = engine.evaluate(x)
        assert y.shape == (system.ny,)
        
        # OutputMatrix should be (ny, nx) = (2, 2)
        C: OutputMatrix = engine.compute_jacobian(x)
        assert C.shape == (system.ny, system.nx)
    
    def test_identity_observation_types(self):
        """Test types work for identity observation (no custom output)"""
        system = MockSystemNoOutput()
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        engine = ObservationEngine(system, code_gen, backend_mgr)
        
        x: StateVector = np.array([1.0])
        
        # Identity: y = x
        y: OutputVector = engine.evaluate(x)
        assert np.allclose(y, x)
        
        # Identity: C = I
        C: OutputMatrix = engine.compute_jacobian(x)
        assert np.allclose(C, np.eye(1))

# ============================================================================
# Run Tests
# ============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
