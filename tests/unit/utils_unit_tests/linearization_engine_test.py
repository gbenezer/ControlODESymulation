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
Comprehensive unit tests for LinearizationEngine

Tests cover:
1. NumPy linearization (controlled and autonomous)
2. PyTorch linearization (controlled and autonomous)
3. JAX linearization (controlled and autonomous)
4. Symbolic linearization (controlled and autonomous)
5. Second-order systems
6. Jacobian verification (controlled and autonomous)
7. Performance tracking
8. Autonomous system edge cases
9. Type system integration
"""

import numpy as np
import pytest
import sympy as sp
from typing import Optional, Tuple

# Type system imports
from src.types import ArrayLike
from src.types.backends import Backend
from src.types.core import ControlVector, InputMatrix, StateMatrix, StateVector

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

from src.systems.base.utils.backend_manager import BackendManager
from src.systems.base.utils.code_generator import CodeGenerator
from src.systems.base.utils.linearization_engine import LinearizationEngine

# ============================================================================
# Mock Systems
# ============================================================================


class MockLinearSystem:
    """Simple linear system: dx/dt = -a*x + u"""

    def __init__(self, a=2.0):
        x = sp.symbols("x", real=True)
        u = sp.symbols("u", real=True)
        a_sym = sp.symbols("a", real=True, positive=True)

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
        q, q_dot = sp.symbols("q q_dot", real=True)
        u = sp.symbols("u", real=True)
        k_sym, c_sym = sp.symbols("k c", real=True, positive=True)

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


class MockAutonomousSystem:
    """Autonomous system: dx/dt = -a*x (no control)"""

    def __init__(self, a=2.0):
        x = sp.symbols("x", real=True)
        a_sym = sp.symbols("a", real=True, positive=True)

        self.state_vars = [x]
        self.control_vars = []  # AUTONOMOUS
        self._f_sym = sp.Matrix([-a_sym * x])
        self._h_sym = None
        self.parameters = {a_sym: a}
        self.nx = 1
        self.nu = 0  # AUTONOMOUS
        self.ny = 1
        self.nq = 1
        self.order = 1

    def substitute_parameters(self, expr):
        return expr.subs(self.parameters)


class MockAutonomous2D:
    """2D autonomous system: Free oscillator"""

    def __init__(self, k=10.0):
        x1, x2 = sp.symbols("x1 x2", real=True)
        k_sym = sp.symbols("k", real=True, positive=True)

        self.state_vars = [x1, x2]
        self.control_vars = []  # AUTONOMOUS
        self._f_sym = sp.Matrix([x2, -k_sym * x1])
        self._h_sym = None
        self.parameters = {k_sym: k}
        self.nx = 2
        self.nu = 0  # AUTONOMOUS
        self.ny = 2
        self.nq = 2
        self.order = 1

    def substitute_parameters(self, expr):
        return expr.subs(self.parameters)


# ============================================================================
# Test Class 0: Type System Integration
# ============================================================================


class TestTypeSystemIntegration:
    """Test integration with centralized type system"""

    def test_state_matrix_type_annotation(self):
        """Test that StateMatrix type annotation works correctly"""
        system = MockLinearSystem(a=2.0)
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        engine = LinearizationEngine(system, code_gen, backend_mgr)

        # Type annotations
        x: StateVector = np.array([1.0])
        u: ControlVector = np.array([0.0])
        
        A: StateMatrix
        B: InputMatrix
        A, B = engine.compute_dynamics(x, u, backend="numpy")
        
        # Verify types and shapes
        assert isinstance(A, np.ndarray)
        assert isinstance(B, np.ndarray)
        assert A.shape == (1, 1)  # StateMatrix is (nx, nx)
        assert B.shape == (1, 1)  # InputMatrix is (nx, nu)

    def test_input_matrix_type_annotation(self):
        """Test that InputMatrix type annotation works correctly"""
        system = MockLinearSystem(a=2.0)
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        engine = LinearizationEngine(system, code_gen, backend_mgr)

        x: StateVector = np.array([1.0])
        u: ControlVector = np.array([0.0])
        
        # Clear semantic meaning
        A: StateMatrix  # State Jacobian ∂f/∂x
        B: InputMatrix  # Input Jacobian ∂f/∂u
        A, B = engine.compute_dynamics(x, u)
        
        assert isinstance(A, np.ndarray)
        assert isinstance(B, np.ndarray)

    def test_backend_literal_type(self):
        """Test that Backend literal type is used correctly"""
        system = MockLinearSystem(a=2.0)
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        engine = LinearizationEngine(system, code_gen, backend_mgr)

        x: StateVector = np.array([1.0])
        u: ControlVector = np.array([0.0])

        # Valid backends should work
        backend: Backend = "numpy"
        A, B = engine.compute_dynamics(x, u, backend=backend)
        assert isinstance(A, np.ndarray)

    def test_tuple_return_type(self):
        """Test that return type is Tuple[StateMatrix, InputMatrix]"""
        system = MockLinearSystem(a=2.0)
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        engine = LinearizationEngine(system, code_gen, backend_mgr)

        x: StateVector = np.array([1.0])
        u: ControlVector = np.array([0.0])
        
        # Return type is explicit tuple
        result: Tuple[StateMatrix, InputMatrix] = engine.compute_dynamics(x, u)
        A, B = result
        
        # First element is StateMatrix
        assert isinstance(A, np.ndarray)
        assert A.shape == (1, 1)
        
        # Second element is InputMatrix
        assert isinstance(B, np.ndarray)
        assert B.shape == (1, 1)

    def test_optional_control_vector_autonomous(self):
        """Test Optional[ControlVector] for autonomous systems"""
        system = MockAutonomousSystem(a=2.0)
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        engine = LinearizationEngine(system, code_gen, backend_mgr)

        x: StateVector = np.array([1.0])
        u: Optional[ControlVector] = None  # Autonomous system
        
        A: StateMatrix
        B: InputMatrix
        A, B = engine.compute_dynamics(x, u)
        
        assert A.shape == (1, 1)
        assert B.shape == (1, 0)  # Empty InputMatrix for autonomous

    @pytest.mark.skipif(not torch_available, reason="PyTorch not installed")
    def test_type_consistency_across_backends(self):
        """Test that types are consistent across backends"""
        system = MockLinearSystem(a=2.0)
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        engine = LinearizationEngine(system, code_gen, backend_mgr)

        # NumPy arrays are StateVector/ControlVector
        x_np: StateVector = np.array([1.0])
        u_np: ControlVector = np.array([0.0])
        A_np: StateMatrix
        B_np: InputMatrix
        A_np, B_np = engine.compute_dynamics(x_np, u_np, backend="numpy")
        
        # PyTorch tensors are also StateVector/ControlVector
        x_torch: StateVector = torch.tensor([1.0])
        u_torch: ControlVector = torch.tensor([0.0])
        A_torch: StateMatrix
        B_torch: InputMatrix
        A_torch, B_torch = engine.compute_dynamics(x_torch, u_torch, backend="torch")
        
        # Both backends work with semantic types
        assert isinstance(A_np, np.ndarray)
        assert isinstance(B_np, np.ndarray)
        assert isinstance(A_torch, torch.Tensor)
        assert isinstance(B_torch, torch.Tensor)

    def test_semantic_clarity_over_generic(self):
        """Test that semantic types provide clarity over generic ArrayLike"""
        system = MockLinearSystem(a=2.0)
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        engine = LinearizationEngine(system, code_gen, backend_mgr)

        x: StateVector = np.array([1.0])
        u: ControlVector = np.array([0.0])
        
        # Clear semantic meaning - no ambiguity
        A: StateMatrix  # Obviously the state Jacobian
        B: InputMatrix  # Obviously the input Jacobian
        A, B = engine.compute_dynamics(x, u)
        
        # vs generic (ambiguous):
        # result1, result2 = engine.compute_dynamics(x, u)  # Which is which?
        
        # Verify correct values
        assert np.allclose(A, np.array([[-2.0]]))  # ∂f/∂x
        assert np.allclose(B, np.array([[1.0]]))   # ∂f/∂u

    def test_second_order_system_types(self):
        """Test type annotations with second-order systems"""
        system = MockSecondOrderSystem(k=10.0, c=0.5)
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        engine = LinearizationEngine(system, code_gen, backend_mgr)

        x: StateVector = np.array([0.1, 0.0])  # [q, q_dot]
        u: ControlVector = np.array([0.0])
        
        A: StateMatrix  # (2, 2) for state-space form
        B: InputMatrix  # (2, 1) for state-space form
        A, B = engine.compute_dynamics(x, u)
        
        # Second-order system has state-space linearization
        assert A.shape == (2, 2)
        assert B.shape == (2, 1)

    def test_arraylike_compatibility(self):
        """Test that ArrayLike is still compatible (less semantic but valid)"""
        system = MockLinearSystem(a=2.0)
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        engine = LinearizationEngine(system, code_gen, backend_mgr)

        # Can still use ArrayLike (less semantic)
        x: ArrayLike = np.array([1.0])
        u: ArrayLike = np.array([0.0])
        
        A, B = engine.compute_dynamics(x, u)
        assert isinstance(A, np.ndarray)
        assert isinstance(B, np.ndarray)


# ============================================================================
# Test Class 1: NumPy Linearization (Controlled)
# ============================================================================


class TestNumpyLinearizationControlled:
    """Test NumPy backend linearization for controlled systems"""

    def test_compute_dynamics_numpy(self):
        """Test linearization with NumPy"""
        system = MockLinearSystem(a=2.0)
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        engine = LinearizationEngine(system, code_gen, backend_mgr)

        # Type annotations show intent
        x: StateVector = np.array([1.0])
        u: ControlVector = np.array([0.0])

        A: StateMatrix
        B: InputMatrix
        A, B = engine.compute_dynamics(x, u, backend="numpy")

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

        A, B = engine.compute_dynamics(x, u, backend="numpy")

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

        A, B = engine.compute_dynamics(x, u, backend="numpy")

        # State-space form: dx/dt = [0 1; -k -c][q; q̇] + [0; 1]u
        expected_A = np.array([[0, 1], [-10, -0.5]])
        expected_B = np.array([[0], [1]])

        assert A.shape == (2, 2)
        assert B.shape == (2, 1)
        assert np.allclose(A, expected_A)
        assert np.allclose(B, expected_B)


# ============================================================================
# Test Class 2: NumPy Linearization (Autonomous)
# ============================================================================


class TestNumpyLinearizationAutonomous:
    """Test NumPy backend linearization for autonomous systems"""

    def test_compute_dynamics_autonomous_u_none(self):
        """Test linearization with u=None for autonomous system"""
        system = MockAutonomousSystem(a=2.0)
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        engine = LinearizationEngine(system, code_gen, backend_mgr)

        # Type annotations for autonomous system
        x: StateVector = np.array([1.0])
        u: Optional[ControlVector] = None  # Explicit None for autonomous

        A: StateMatrix
        B: InputMatrix
        A, B = engine.compute_dynamics(x, u=u, backend="numpy")

        assert isinstance(A, np.ndarray)
        assert isinstance(B, np.ndarray)
        assert A.shape == (1, 1)
        assert B.shape == (1, 0)  # Empty B matrix

        # ∂f/∂x = -a = -2
        assert np.allclose(A, np.array([[-2.0]]))

    def test_compute_dynamics_autonomous_2d(self):
        """Test 2D autonomous system"""
        system = MockAutonomous2D(k=10.0)
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        engine = LinearizationEngine(system, code_gen, backend_mgr)

        x = np.array([0.1, 0.0])

        A, B = engine.compute_dynamics(x, backend="numpy")

        assert A.shape == (2, 2)
        assert B.shape == (2, 0)  # Empty B matrix

        expected_A = np.array([[0, 1], [-10, 0]])
        np.testing.assert_allclose(A, expected_A)

    def test_autonomous_requires_u_none(self):
        """Test that controlled system requires u"""
        system = MockLinearSystem(a=2.0)
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        engine = LinearizationEngine(system, code_gen, backend_mgr)

        x = np.array([1.0])

        with pytest.raises(ValueError, match="requires control input"):
            engine.compute_dynamics(x, u=None)

    def test_autonomous_batched(self):
        """Test batched linearization for autonomous system"""
        system = MockAutonomousSystem(a=2.0)
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        engine = LinearizationEngine(system, code_gen, backend_mgr)

        x = np.array([[1.0], [2.0]])

        A, B = engine.compute_dynamics(x, backend="numpy")

        assert A.shape == (2, 1, 1)
        assert B.shape == (2, 1, 0)  # Empty B for all batches


# ============================================================================
# Test Class 3: PyTorch Linearization (Controlled)
# ============================================================================


class TestTorchLinearizationControlled:
    """Test PyTorch backend linearization for controlled systems"""

    @pytest.mark.skipif(not torch_available, reason="PyTorch not installed")
    def test_compute_dynamics_torch(self):
        """Test linearization with PyTorch"""
        system = MockLinearSystem(a=2.0)
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        engine = LinearizationEngine(system, code_gen, backend_mgr)

        x = torch.tensor([1.0])
        u = torch.tensor([0.0])

        A, B = engine.compute_dynamics(x, u, backend="torch")

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
# Test Class 4: PyTorch Linearization (Autonomous)
# ============================================================================


class TestTorchLinearizationAutonomous:
    """Test PyTorch backend linearization for autonomous systems"""

    @pytest.mark.skipif(not torch_available, reason="PyTorch not installed")
    def test_compute_dynamics_autonomous_torch(self):
        """Test autonomous system with PyTorch"""
        system = MockAutonomousSystem(a=2.0)
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        engine = LinearizationEngine(system, code_gen, backend_mgr)

        x = torch.tensor([1.0])

        A, B = engine.compute_dynamics(x, u=None, backend="torch")

        assert isinstance(A, torch.Tensor)
        assert isinstance(B, torch.Tensor)
        assert A.shape == (1, 1)
        assert B.shape == (1, 0)  # Empty B
        torch.testing.assert_close(A, torch.tensor([[-2.0]]))

    @pytest.mark.skipif(not torch_available, reason="PyTorch not installed")
    def test_autonomous_2d_torch(self):
        """Test 2D autonomous system with PyTorch"""
        system = MockAutonomous2D(k=10.0)
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        engine = LinearizationEngine(system, code_gen, backend_mgr)

        x = torch.tensor([0.1, 0.0])

        A, B = engine.compute_dynamics(x, backend="torch")

        assert A.shape == (2, 2)
        assert B.shape == (2, 0)


# ============================================================================
# Test Class 5: JAX Linearization (Controlled)
# ============================================================================


class TestJaxLinearizationControlled:
    """Test JAX backend linearization for controlled systems"""

    @pytest.mark.skipif(not jax_available, reason="JAX not installed")
    def test_compute_dynamics_jax(self):
        """Test linearization with JAX (uses autodiff)"""
        system = MockLinearSystem(a=2.0)
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        engine = LinearizationEngine(system, code_gen, backend_mgr)

        x = jnp.array([1.0])
        u = jnp.array([0.0])

        A, B = engine.compute_dynamics(x, u, backend="jax")

        assert isinstance(A, jnp.ndarray)
        assert isinstance(B, jnp.ndarray)
        assert jnp.allclose(A, jnp.array([[-2.0]]))
        assert jnp.allclose(B, jnp.array([[1.0]]))


# ============================================================================
# Test Class 6: JAX Linearization (Autonomous)
# ============================================================================


class TestJaxLinearizationAutonomous:
    """Test JAX backend linearization for autonomous systems"""

    @pytest.mark.skipif(not jax_available, reason="JAX not installed")
    def test_compute_dynamics_autonomous_jax(self):
        """Test autonomous system with JAX"""
        system = MockAutonomousSystem(a=2.0)
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        engine = LinearizationEngine(system, code_gen, backend_mgr)

        x = jnp.array([1.0])

        A, B = engine.compute_dynamics(x, u=None, backend="jax")

        assert isinstance(A, jnp.ndarray)
        assert isinstance(B, jnp.ndarray)
        assert A.shape == (1, 1)
        assert B.shape == (1, 0)  # Empty B
        np.testing.assert_allclose(np.array(A), np.array([[-2.0]]))

    @pytest.mark.skipif(not jax_available, reason="JAX not installed")
    def test_autonomous_2d_jax(self):
        """Test 2D autonomous system with JAX"""
        system = MockAutonomous2D(k=10.0)
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        engine = LinearizationEngine(system, code_gen, backend_mgr)

        x = jnp.array([0.1, 0.0])

        A, B = engine.compute_dynamics(x, backend="jax")

        assert A.shape == (2, 2)
        assert B.shape == (2, 0)


# ============================================================================
# Test Class 7: Symbolic Linearization (Controlled)
# ============================================================================


class TestSymbolicLinearizationControlled:
    """Test symbolic linearization for controlled systems"""

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
        assert A_sym.shape == (1, 1)
        assert B_sym.shape == (1, 1)

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

        assert A_sym.shape == (2, 2)
        assert B_sym.shape == (2, 1)
        assert np.allclose(A_np, expected_A)
        assert np.allclose(B_np, expected_B)


# ============================================================================
# Test Class 8: Symbolic Linearization (Autonomous)
# ============================================================================


class TestSymbolicLinearizationAutonomous:
    """Test symbolic linearization for autonomous systems"""

    def test_compute_symbolic_autonomous(self):
        """Test symbolic linearization for autonomous system"""
        system = MockAutonomousSystem(a=2.0)
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        engine = LinearizationEngine(system, code_gen, backend_mgr)

        x_eq = sp.Matrix([0])

        # u_eq=None for autonomous
        A_sym, B_sym = engine.compute_symbolic(x_eq, u_eq=None)

        assert isinstance(A_sym, sp.Matrix)
        assert isinstance(B_sym, sp.Matrix)
        assert A_sym.shape == (1, 1)
        assert B_sym.shape == (1, 0)  # Empty B

        A_np = np.array(A_sym, dtype=float)
        assert np.allclose(A_np, np.array([[-2.0]]))

    def test_compute_symbolic_autonomous_2d(self):
        """Test symbolic linearization for 2D autonomous system"""
        system = MockAutonomous2D(k=10.0)
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        engine = LinearizationEngine(system, code_gen, backend_mgr)

        x_eq = sp.Matrix([0, 0])

        A_sym, B_sym = engine.compute_symbolic(x_eq)

        assert A_sym.shape == (2, 2)
        assert B_sym.shape == (2, 0)  # Empty B

        A_np = np.array(A_sym, dtype=float)
        expected_A = np.array([[0, 1], [-10, 0]])
        np.testing.assert_allclose(A_np, expected_A)


# ============================================================================
# Test Class 9: Jacobian Verification (Controlled)
# ============================================================================


class TestJacobianVerificationControlled:
    """Test Jacobian verification against autodiff for controlled systems"""

    @pytest.mark.skipif(not torch_available, reason="PyTorch not installed")
    def test_verify_jacobians_torch(self):
        """Test verification with PyTorch autodiff"""
        system = MockLinearSystem(a=2.0)
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        engine = LinearizationEngine(system, code_gen, backend_mgr)

        x = torch.tensor([1.0])
        u = torch.tensor([0.0])

        results = engine.verify_jacobians(x, u, backend="torch", tol=1e-4)

        assert results["A_match"] is True
        assert results["B_match"] is True
        assert results["A_error"] < 1e-4
        assert results["B_error"] < 1e-4

    @pytest.mark.skipif(not jax_available, reason="JAX not installed")
    def test_verify_jacobians_jax(self):
        """Test verification with JAX autodiff"""
        system = MockLinearSystem(a=2.0)
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        engine = LinearizationEngine(system, code_gen, backend_mgr)

        x = jnp.array([1.0])
        u = jnp.array([0.0])

        results = engine.verify_jacobians(x, u, backend="jax", tol=1e-4)

        assert results["A_match"] is True
        assert results["B_match"] is True

    def test_verify_jacobians_numpy_fails(self):
        """Test that NumPy backend is rejected"""
        system = MockLinearSystem()
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        engine = LinearizationEngine(system, code_gen, backend_mgr)

        x = np.array([1.0])
        u = np.array([0.0])

        with pytest.raises(ValueError, match="requires autodiff"):
            engine.verify_jacobians(x, u, backend="numpy")

    @pytest.mark.skipif(not torch_available, reason="PyTorch not installed")
    def test_verify_second_order_jacobians(self):
        """Test verification for second-order system"""
        system = MockSecondOrderSystem(k=10.0, c=0.5)
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        engine = LinearizationEngine(system, code_gen, backend_mgr)

        x = torch.tensor([0.1, 0.0])
        u = torch.tensor([0.0])

        results = engine.verify_jacobians(x, u, backend="torch", tol=1e-3)

        assert results["A_match"] is True
        assert results["B_match"] is True


# ============================================================================
# Test Class 10: Jacobian Verification (Autonomous)
# ============================================================================


class TestJacobianVerificationAutonomous:
    """Test Jacobian verification for autonomous systems"""

    @pytest.mark.skipif(not torch_available, reason="PyTorch not installed")
    def test_verify_jacobians_autonomous_torch(self):
        """Test verification for autonomous system with PyTorch"""
        system = MockAutonomousSystem(a=2.0)
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        engine = LinearizationEngine(system, code_gen, backend_mgr)

        x = torch.tensor([1.0])

        results = engine.verify_jacobians(x, u=None, backend="torch", tol=1e-4)

        assert results["A_match"] is True
        assert results["B_match"] is True  # Trivially true for empty B
        assert results["A_error"] < 1e-4
        assert results["B_error"] == 0.0  # Zero for empty matrix

    @pytest.mark.skipif(not jax_available, reason="JAX not installed")
    def test_verify_jacobians_autonomous_jax(self):
        """Test verification for autonomous system with JAX"""
        system = MockAutonomousSystem(a=2.0)
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        engine = LinearizationEngine(system, code_gen, backend_mgr)

        x = jnp.array([1.0])

        results = engine.verify_jacobians(x, backend="jax", tol=1e-4)

        assert results["A_match"] is True
        assert results["B_match"] is True
        assert results["B_error"] == 0.0


# ============================================================================
# Test Class 11: Performance Tracking
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

        assert stats["calls"] == 0
        assert stats["total_time"] == 0.0
        assert stats["avg_time"] == 0.0

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

        assert stats["calls"] == 2
        assert stats["total_time"] > 0

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
        assert stats["calls"] == 0


# ============================================================================
# Test Class 12: String Representations
# ============================================================================


class TestStringRepresentations:
    """Test __repr__ and __str__"""

    def test_repr_controlled(self):
        """Test __repr__ output for controlled system"""
        system = MockLinearSystem()
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        engine = LinearizationEngine(system, code_gen, backend_mgr)

        repr_str = repr(engine)

        assert "LinearizationEngine" in repr_str
        assert "nx=1" in repr_str
        assert "nu=1" in repr_str

    def test_repr_autonomous(self):
        """Test __repr__ output for autonomous system"""
        system = MockAutonomousSystem()
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        engine = LinearizationEngine(system, code_gen, backend_mgr)

        repr_str = repr(engine)

        assert "LinearizationEngine" in repr_str
        assert "nu=0" in repr_str
        assert "(autonomous)" in repr_str

    def test_str(self):
        """Test __str__ output"""
        system = MockLinearSystem()
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        engine = LinearizationEngine(system, code_gen, backend_mgr)

        str_repr = str(engine)

        assert "LinearizationEngine" in str_repr


# ============================================================================
# Test Class 13: Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests"""

    def test_full_workflow_controlled(self):
        """Test complete workflow for controlled system"""
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

    def test_full_workflow_autonomous(self):
        """Test complete workflow for autonomous system"""
        system = MockAutonomousSystem(a=2.0)
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        engine = LinearizationEngine(system, code_gen, backend_mgr)

        x = np.array([1.0])

        # Numerical linearization
        A_num, B_num = engine.compute_dynamics(x)

        # Symbolic linearization
        A_sym, B_sym = engine.compute_symbolic(sp.Matrix([1.0]))
        A_sym_np = np.array(A_sym, dtype=float)

        # Should match
        assert np.allclose(A_num, A_sym_np)
        assert B_num.shape == (1, 0)
        assert B_sym.shape == (1, 0)

    def test_multi_backend_consistency(self):
        """Test that all backends give same result"""
        system = MockLinearSystem(a=2.0)
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        engine = LinearizationEngine(system, code_gen, backend_mgr)

        x_np = np.array([1.0])
        u_np = np.array([0.0])

        A_np, B_np = engine.compute_dynamics(x_np, u_np, backend="numpy")

        backends_to_test = ["numpy"]
        if torch_available:
            backends_to_test.append("torch")
        if jax_available:
            backends_to_test.append("jax")

        for backend in backends_to_test:
            A, B = engine.compute_dynamics(x_np, u_np, backend=backend)
            A_val = np.array(A) if not isinstance(A, np.ndarray) else A
            B_val = np.array(B) if not isinstance(B, np.ndarray) else B

            assert np.allclose(A_val, A_np), f"{backend} A doesn't match NumPy"
            assert np.allclose(B_val, B_np), f"{backend} B doesn't match NumPy"

    def test_multi_backend_consistency_autonomous(self):
        """Test that all backends give same result for autonomous system"""
        system = MockAutonomous2D(k=10.0)
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        engine = LinearizationEngine(system, code_gen, backend_mgr)

        x_np = np.array([0.1, 0.0])

        A_np, B_np = engine.compute_dynamics(x_np, backend="numpy")

        backends_to_test = ["numpy"]
        if torch_available:
            backends_to_test.append("torch")
        if jax_available:
            backends_to_test.append("jax")

        for backend in backends_to_test:
            A, B = engine.compute_dynamics(x_np, backend=backend)
            A_val = np.array(A) if not isinstance(A, np.ndarray) else A
            B_val = np.array(B) if not isinstance(B, np.ndarray) else B

            assert np.allclose(A_val, A_np), f"{backend} A doesn't match NumPy"
            assert B_val.shape == B_np.shape, f"{backend} B shape mismatch"
            assert B_val.shape == (2, 0), f"{backend} B should be (2, 0)"


# ============================================================================
# Run Tests
# ============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
