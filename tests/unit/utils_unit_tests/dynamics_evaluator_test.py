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
Comprehensive unit tests for DynamicsEvaluator

Tests cover:
1. NumPy backend evaluation (controlled and autonomous)
2. PyTorch backend evaluation (controlled and autonomous)
3. JAX backend evaluation (controlled and autonomous)
4. Input validation
5. Shape handling (batched vs single)
6. Backend dispatch
7. Performance tracking
8. Error handling
9. Autonomous system support
10. Type system integration
"""

import numpy as np
import pytest
import sympy as sp
from typing import Optional

# Type system imports
from src.types import ArrayLike
from src.types.backends import Backend
from src.types.core import ControlVector, StateVector
from src.types.utilities import ExecutionStats

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
from src.systems.base.utils.dynamics_evaluator import DynamicsEvaluator

# ============================================================================
# Mock Systems for Testing
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
        self.parameters = {a_sym: a}
        self.nx = 1
        self.nu = 1
        self.order = 1

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
        self.parameters = {a_sym: a}
        self.nx = 1
        self.nu = 0  # AUTONOMOUS
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
        self.parameters = {k_sym: k}
        self.nx = 2
        self.nu = 0  # AUTONOMOUS
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
        self.parameters = {k_sym: k, c_sym: c}
        self.nx = 2
        self.nu = 1
        self.nq = 1
        self.order = 2

    def substitute_parameters(self, expr):
        return expr.subs(self.parameters)


class MockMultiInputSystem:
    """System with multiple states and controls"""

    def __init__(self):
        x1, x2 = sp.symbols("x1 x2", real=True)
        u1, u2 = sp.symbols("u1 u2", real=True)

        self.state_vars = [x1, x2]
        self.control_vars = [u1, u2]
        self._f_sym = sp.Matrix([x2 + u1, -x1 + u2])
        self.parameters = {}
        self.nx = 2
        self.nu = 2
        self.order = 1

    def substitute_parameters(self, expr):
        return expr


# ============================================================================
# Test Class 0: Type System Integration
# ============================================================================


class TestTypeSystemIntegration:
    """Test integration with centralized type system"""

    def test_state_vector_type_annotation(self):
        """Test that StateVector type annotation works correctly"""
        system = MockLinearSystem(a=2.0)
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        evaluator = DynamicsEvaluator(system, code_gen, backend_mgr)

        # Type annotation should work
        x: StateVector = np.array([1.0])
        u: ControlVector = np.array([0.5])
        
        dx: StateVector = evaluator.evaluate(x, u, backend="numpy")
        
        assert isinstance(dx, np.ndarray)
        assert dx.shape == (1,)

    def test_control_vector_type_annotation(self):
        """Test that ControlVector type annotation works correctly"""
        system = MockLinearSystem(a=2.0)
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        evaluator = DynamicsEvaluator(system, code_gen, backend_mgr)

        # Type annotations
        x: StateVector = np.array([1.0])
        u: ControlVector = np.array([0.5])
        
        # Should accept typed parameters
        dx = evaluator.evaluate(x, u)
        assert isinstance(dx, np.ndarray)

    def test_backend_literal_type(self):
        """Test that Backend literal type is used correctly"""
        system = MockLinearSystem(a=2.0)
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        evaluator = DynamicsEvaluator(system, code_gen, backend_mgr)

        x: StateVector = np.array([1.0])
        u: ControlVector = np.array([0.5])

        # Valid backends should work
        backend: Backend = "numpy"
        dx = evaluator.evaluate(x, u, backend=backend)
        assert isinstance(dx, np.ndarray)

        backend = "torch"  # Type checker allows this
        backend = "jax"    # Type checker allows this

    def test_optional_control_vector(self):
        """Test that Optional[ControlVector] works for autonomous systems"""
        system = MockAutonomousSystem(a=2.0)
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        evaluator = DynamicsEvaluator(system, code_gen, backend_mgr)

        # Type annotation with Optional
        x: StateVector = np.array([1.0])
        u: Optional[ControlVector] = None  # Autonomous system
        
        dx: StateVector = evaluator.evaluate(x, u)
        assert isinstance(dx, np.ndarray)

    def test_return_type_is_state_vector(self):
        """Test that return type is StateVector"""
        system = MockLinearSystem(a=2.0)
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        evaluator = DynamicsEvaluator(system, code_gen, backend_mgr)

        x: StateVector = np.array([1.0])
        u: ControlVector = np.array([0.5])
        
        # Return type should be StateVector
        result: StateVector = evaluator.evaluate(x, u)
        
        # Verify it's actually a state derivative
        assert isinstance(result, np.ndarray)
        assert result.shape == (1,)

    @pytest.mark.skipif(not torch_available, reason="PyTorch not installed")
    def test_type_consistency_across_backends(self):
        """Test that types are consistent across backends"""
        system = MockLinearSystem(a=2.0)
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        evaluator = DynamicsEvaluator(system, code_gen, backend_mgr)

        # NumPy arrays are StateVector/ControlVector
        x_np: StateVector = np.array([1.0])
        u_np: ControlVector = np.array([0.5])
        dx_np: StateVector = evaluator.evaluate(x_np, u_np, backend="numpy")
        
        # PyTorch tensors are also StateVector/ControlVector
        x_torch: StateVector = torch.tensor([1.0])
        u_torch: ControlVector = torch.tensor([0.5])
        dx_torch: StateVector = evaluator.evaluate(x_torch, u_torch, backend="torch")
        
        # Both should work
        assert isinstance(dx_np, np.ndarray)
        assert isinstance(dx_torch, torch.Tensor)

    def test_type_annotations_in_batched_case(self):
        """Test that type annotations work for batched inputs"""
        system = MockLinearSystem(a=2.0)
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        evaluator = DynamicsEvaluator(system, code_gen, backend_mgr)

        # Batched inputs are still StateVector/ControlVector
        x: StateVector = np.array([[1.0], [2.0], [3.0]])
        u: ControlVector = np.array([[0.5], [0.5], [0.5]])
        
        dx: StateVector = evaluator.evaluate(x, u)
        
        assert dx.shape == (3, 1)

    def test_arraylike_is_compatible(self):
        """Test that ArrayLike is compatible with semantic types"""
        system = MockLinearSystem(a=2.0)
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        evaluator = DynamicsEvaluator(system, code_gen, backend_mgr)

        # ArrayLike can be used as well
        x: ArrayLike = np.array([1.0])
        u: ArrayLike = np.array([0.5])
        
        # Should work with ArrayLike (less semantic but still valid)
        dx = evaluator.evaluate(x, u)
        assert isinstance(dx, np.ndarray)
    
    def test_execution_stats_type(self):
        """Test that get_stats returns ExecutionStats TypedDict"""
        system = MockLinearSystem(a=2.0)
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        evaluator = DynamicsEvaluator(system, code_gen, backend_mgr)
        
        x: StateVector = np.array([1.0])
        u: ControlVector = np.array([0.5])
        
        # Make a call
        evaluator.evaluate(x, u)
        
        # get_stats should return ExecutionStats
        stats: ExecutionStats = evaluator.get_stats()
        
        # Verify structure
        assert "calls" in stats
        assert "total_time" in stats
        assert "avg_time" in stats
        
        # Verify types
        assert isinstance(stats["calls"], int)
        assert isinstance(stats["total_time"], float)
        assert isinstance(stats["avg_time"], float)
        
        # Verify values
        assert stats["calls"] == 1
        assert stats["total_time"] > 0
        assert stats["avg_time"] > 0


# ============================================================================
# Test Class 1: NumPy Backend Evaluation (Controlled)
# ============================================================================


class TestNumpyEvaluationControlled:
    """Test NumPy backend evaluation for controlled systems"""

    def test_evaluate_numpy_single(self):
        """Test single point evaluation with NumPy"""
        system = MockLinearSystem(a=2.0)
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        evaluator = DynamicsEvaluator(system, code_gen, backend_mgr)

        # Type annotations show intent
        x: StateVector = np.array([1.0])
        u: ControlVector = np.array([0.5])

        dx: StateVector = evaluator.evaluate(x, u, backend="numpy")

        # dx = -2*x + u = -2*1 + 0.5 = -1.5
        assert isinstance(dx, np.ndarray)
        assert dx.shape == (1,)
        assert np.allclose(dx, np.array([-1.5]))

    def test_evaluate_numpy_batched(self):
        """Test batched evaluation with NumPy"""
        system = MockLinearSystem(a=2.0)
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        evaluator = DynamicsEvaluator(system, code_gen, backend_mgr)

        x = np.array([[1.0], [2.0], [3.0]])
        u = np.array([[0.5], [0.5], [0.5]])

        dx = evaluator.evaluate(x, u, backend="numpy")

        assert dx.shape == (3, 1)
        expected = np.array([[-1.5], [-3.5], [-5.5]])
        assert np.allclose(dx, expected)

    def test_evaluate_numpy_auto_detect(self):
        """Test auto-detection of NumPy backend"""
        system = MockLinearSystem(a=1.0)
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        evaluator = DynamicsEvaluator(system, code_gen, backend_mgr)

        x = np.array([1.0])
        u = np.array([0.0])

        # Should auto-detect NumPy
        dx = evaluator.evaluate(x, u)

        assert isinstance(dx, np.ndarray)

    def test_evaluate_numpy_second_order(self):
        """Test second-order system evaluation"""
        system = MockSecondOrderSystem(k=10.0, c=0.5)
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        evaluator = DynamicsEvaluator(system, code_gen, backend_mgr)

        x = np.array([0.1, 0.0])  # [q, q_dot]
        u = np.array([0.0])

        dx = evaluator.evaluate(x, u, backend="numpy")

        # dx = q_ddot = -10*0.1 - 0.5*0 + 0 = -1.0
        assert dx.shape == (1,)
        assert np.allclose(dx, np.array([-1.0]))

    def test_evaluate_numpy_multi_input(self):
        """Test system with multiple states and controls"""
        system = MockMultiInputSystem()
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        evaluator = DynamicsEvaluator(system, code_gen, backend_mgr)

        x = np.array([1.0, 2.0])
        u = np.array([0.5, -0.5])

        dx = evaluator.evaluate(x, u, backend="numpy")

        # dx = [x2 + u1, -x1 + u2] = [2.0 + 0.5, -1.0 + (-0.5)] = [2.5, -1.5]
        assert dx.shape == (2,)
        assert np.allclose(dx, np.array([2.5, -1.5]))


# ============================================================================
# Test Class 2: NumPy Backend Evaluation (Autonomous)
# ============================================================================


class TestNumpyEvaluationAutonomous:
    """Test NumPy backend evaluation for autonomous systems"""

    def test_evaluate_autonomous_u_none(self):
        """Test autonomous system with u=None"""
        system = MockAutonomousSystem(a=2.0)
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        evaluator = DynamicsEvaluator(system, code_gen, backend_mgr)

        # Type annotations for autonomous system
        x: StateVector = np.array([1.0])
        u: Optional[ControlVector] = None  # Explicit None for autonomous

        dx: StateVector = evaluator.evaluate(x, u=u, backend="numpy")

        # dx = -2*x = -2
        assert isinstance(dx, np.ndarray)
        assert dx.shape == (1,)
        assert np.allclose(dx, np.array([-2.0]))

    def test_evaluate_autonomous_no_u_arg(self):
        """Test autonomous system without passing u"""
        system = MockAutonomousSystem(a=2.0)
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        evaluator = DynamicsEvaluator(system, code_gen, backend_mgr)

        x = np.array([1.0])

        # Don't pass u at all
        dx = evaluator.evaluate(x, backend="numpy")

        assert np.allclose(dx, np.array([-2.0]))

    def test_evaluate_autonomous_2d(self):
        """Test 2D autonomous system"""
        system = MockAutonomous2D(k=10.0)
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        evaluator = DynamicsEvaluator(system, code_gen, backend_mgr)

        x = np.array([0.1, 0.0])

        dx = evaluator.evaluate(x, backend="numpy")

        # dx = [x2, -k*x1] = [0.0, -10*0.1] = [0.0, -1.0]
        assert dx.shape == (2,)
        assert np.allclose(dx, np.array([0.0, -1.0]))

    def test_autonomous_batched(self):
        """Test batched autonomous evaluation"""
        system = MockAutonomousSystem(a=2.0)
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        evaluator = DynamicsEvaluator(system, code_gen, backend_mgr)

        x = np.array([[1.0], [2.0], [3.0]])

        dx = evaluator.evaluate(x, backend="numpy")

        assert dx.shape == (3, 1)
        expected = np.array([[-2.0], [-4.0], [-6.0]])
        assert np.allclose(dx, expected)

    def test_autonomous_rejects_control(self):
        """Test that autonomous system rejects control input"""
        system = MockAutonomousSystem(a=2.0)
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        evaluator = DynamicsEvaluator(system, code_gen, backend_mgr)

        x = np.array([1.0])
        u = np.array([0.5])  # Should be rejected

        with pytest.raises(ValueError, match="does not accept control input"):
            evaluator.evaluate(x, u, backend="numpy")

    def test_controlled_requires_u(self):
        """Test that controlled system requires u"""
        system = MockLinearSystem(a=2.0)
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        evaluator = DynamicsEvaluator(system, code_gen, backend_mgr)

        x = np.array([1.0])

        with pytest.raises(ValueError, match="requires control input"):
            evaluator.evaluate(x, u=None, backend="numpy")


# ============================================================================
# Test Class 3: PyTorch Backend Evaluation (Controlled)
# ============================================================================


class TestTorchEvaluationControlled:
    """Test PyTorch backend evaluation for controlled systems"""

    @pytest.mark.skipif(not torch_available, reason="PyTorch not installed")
    def test_evaluate_torch_single(self):
        """Test single point evaluation with PyTorch"""
        system = MockLinearSystem(a=2.0)
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        evaluator = DynamicsEvaluator(system, code_gen, backend_mgr)

        x = torch.tensor([1.0])
        u = torch.tensor([0.5])

        dx = evaluator.evaluate(x, u, backend="torch")

        assert isinstance(dx, torch.Tensor)
        assert dx.shape == (1,)
        assert torch.allclose(dx, torch.tensor([-1.5]))

    @pytest.mark.skipif(not torch_available, reason="PyTorch not installed")
    def test_evaluate_torch_batched(self):
        """Test batched evaluation with PyTorch"""
        system = MockLinearSystem(a=2.0)
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        evaluator = DynamicsEvaluator(system, code_gen, backend_mgr)

        x = torch.tensor([[1.0], [2.0], [3.0]])
        u = torch.tensor([[0.5], [0.5], [0.5]])

        dx = evaluator.evaluate(x, u, backend="torch")

        assert dx.shape == (3, 1)
        expected = torch.tensor([[-1.5], [-3.5], [-5.5]])
        assert torch.allclose(dx, expected)

    @pytest.mark.skipif(not torch_available, reason="PyTorch not installed")
    def test_evaluate_torch_auto_detect(self):
        """Test auto-detection of PyTorch backend"""
        system = MockLinearSystem(a=1.0)
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        evaluator = DynamicsEvaluator(system, code_gen, backend_mgr)

        x = torch.tensor([1.0])
        u = torch.tensor([0.0])

        # Should auto-detect torch
        dx = evaluator.evaluate(x, u)

        assert isinstance(dx, torch.Tensor)

    @pytest.mark.skipif(not torch_available, reason="PyTorch not installed")
    def test_evaluate_torch_gradients(self):
        """Test that PyTorch gradients work"""
        system = MockLinearSystem(a=2.0)
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        evaluator = DynamicsEvaluator(system, code_gen, backend_mgr)

        x = torch.tensor([1.0], requires_grad=True)
        u = torch.tensor([0.5])

        dx = evaluator.evaluate(x, u, backend="torch")
        dx.backward()

        # df/dx = -a = -2
        assert x.grad is not None
        assert torch.allclose(x.grad, torch.tensor([-2.0]))


# ============================================================================
# Test Class 4: PyTorch Backend Evaluation (Autonomous)
# ============================================================================


class TestTorchEvaluationAutonomous:
    """Test PyTorch backend evaluation for autonomous systems"""

    @pytest.mark.skipif(not torch_available, reason="PyTorch not installed")
    def test_evaluate_autonomous_torch(self):
        """Test autonomous system with PyTorch"""
        system = MockAutonomousSystem(a=2.0)
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        evaluator = DynamicsEvaluator(system, code_gen, backend_mgr)

        x = torch.tensor([1.0])

        dx = evaluator.evaluate(x, u=None, backend="torch")

        assert isinstance(dx, torch.Tensor)
        assert dx.shape == (1,)
        assert torch.allclose(dx, torch.tensor([-2.0]))

    @pytest.mark.skipif(not torch_available, reason="PyTorch not installed")
    def test_autonomous_2d_torch(self):
        """Test 2D autonomous system with PyTorch"""
        system = MockAutonomous2D(k=10.0)
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        evaluator = DynamicsEvaluator(system, code_gen, backend_mgr)

        x = torch.tensor([0.1, 0.0])

        dx = evaluator.evaluate(x, backend="torch")

        assert dx.shape == (2,)
        assert torch.allclose(dx, torch.tensor([0.0, -1.0]))


# ============================================================================
# Test Class 5: JAX Backend Evaluation (Controlled)
# ============================================================================


class TestJaxEvaluationControlled:
    """Test JAX backend evaluation for controlled systems"""

    @pytest.mark.skipif(not jax_available, reason="JAX not installed")
    def test_evaluate_jax_single(self):
        """Test single point evaluation with JAX"""
        system = MockLinearSystem(a=2.0)
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        evaluator = DynamicsEvaluator(system, code_gen, backend_mgr)

        x = jnp.array([1.0])
        u = jnp.array([0.5])

        dx = evaluator.evaluate(x, u, backend="jax")

        assert isinstance(dx, jnp.ndarray)
        assert jnp.allclose(dx, jnp.array([-1.5]))

    @pytest.mark.skipif(not jax_available, reason="JAX not installed")
    def test_evaluate_jax_batched(self):
        """Test batched evaluation with JAX (uses vmap)"""
        system = MockLinearSystem(a=2.0)
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        evaluator = DynamicsEvaluator(system, code_gen, backend_mgr)

        x = jnp.array([[1.0], [2.0], [3.0]])
        u = jnp.array([[0.5], [0.5], [0.5]])

        dx = evaluator.evaluate(x, u, backend="jax")

        assert dx.shape == (3, 1)
        expected = jnp.array([[-1.5], [-3.5], [-5.5]])
        assert jnp.allclose(dx, expected)

    @pytest.mark.skipif(not jax_available, reason="JAX not installed")
    def test_evaluate_jax_auto_detect(self):
        """Test auto-detection of JAX backend"""
        system = MockLinearSystem(a=1.0)
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        evaluator = DynamicsEvaluator(system, code_gen, backend_mgr)

        x = jnp.array([1.0])
        u = jnp.array([0.0])

        # Should auto-detect JAX
        dx = evaluator.evaluate(x, u)

        assert isinstance(dx, jnp.ndarray)


# ============================================================================
# Test Class 6: JAX Backend Evaluation (Autonomous)
# ============================================================================


class TestJaxEvaluationAutonomous:
    """Test JAX backend evaluation for autonomous systems"""

    @pytest.mark.skipif(not jax_available, reason="JAX not installed")
    def test_evaluate_autonomous_jax(self):
        """Test autonomous system with JAX"""
        system = MockAutonomousSystem(a=2.0)
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        evaluator = DynamicsEvaluator(system, code_gen, backend_mgr)

        x = jnp.array([1.0])

        dx = evaluator.evaluate(x, backend="jax")

        assert isinstance(dx, jnp.ndarray)
        assert jnp.allclose(dx, jnp.array([-2.0]))

    @pytest.mark.skipif(not jax_available, reason="JAX not installed")
    def test_autonomous_2d_jax(self):
        """Test 2D autonomous system with JAX"""
        system = MockAutonomous2D(k=10.0)
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        evaluator = DynamicsEvaluator(system, code_gen, backend_mgr)

        x = jnp.array([0.1, 0.0])

        dx = evaluator.evaluate(x, backend="jax")

        assert dx.shape == (2,)
        assert jnp.allclose(dx, jnp.array([0.0, -1.0]))


# ============================================================================
# Test Class 7: Backend Conversion and Dispatch
# ============================================================================


class TestBackendDispatch:
    """Test backend dispatch and conversion"""

    @pytest.mark.skipif(not torch_available, reason="PyTorch not installed")
    def test_force_backend_conversion(self):
        """Test forcing backend with input conversion"""
        system = MockLinearSystem(a=2.0)
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        evaluator = DynamicsEvaluator(system, code_gen, backend_mgr)

        # NumPy input, force torch backend
        x = np.array([1.0])
        u = np.array([0.5])

        dx = evaluator.evaluate(x, u, backend="torch")

        # Should return torch tensor
        assert isinstance(dx, torch.Tensor)
        assert torch.allclose(dx, torch.tensor([-1.5]))

    def test_default_backend(self):
        """Test using configured default backend"""
        system = MockLinearSystem(a=2.0)
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        backend_mgr.set_default("numpy")
        evaluator = DynamicsEvaluator(system, code_gen, backend_mgr)

        x = np.array([1.0])
        u = np.array([0.5])

        dx = evaluator.evaluate(x, u, backend="default")

        assert isinstance(dx, np.ndarray)

    def test_multi_backend_consistency_controlled(self):
        """Test that all backends give same result for controlled systems"""
        system = MockLinearSystem(a=2.0)
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        evaluator = DynamicsEvaluator(system, code_gen, backend_mgr)

        x_np = np.array([1.0])
        u_np = np.array([0.5])

        # Evaluate with NumPy
        dx_np = evaluator.evaluate(x_np, u_np, backend="numpy")

        backends_to_test = ["numpy"]
        if torch_available:
            backends_to_test.append("torch")
        if jax_available:
            backends_to_test.append("jax")

        # All backends should give same result
        for backend in backends_to_test:
            dx = evaluator.evaluate(x_np, u_np, backend=backend)
            dx_val = np.array(dx) if not isinstance(dx, np.ndarray) else dx

            assert np.allclose(dx_val, dx_np), f"{backend} doesn't match NumPy"

    def test_multi_backend_consistency_autonomous(self):
        """Test that all backends give same result for autonomous systems"""
        system = MockAutonomousSystem(a=2.0)
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        evaluator = DynamicsEvaluator(system, code_gen, backend_mgr)

        x_np = np.array([1.0])

        # Evaluate with NumPy
        dx_np = evaluator.evaluate(x_np, backend="numpy")

        backends_to_test = ["numpy"]
        if torch_available:
            backends_to_test.append("torch")
        if jax_available:
            backends_to_test.append("jax")

        # All backends should give same result
        for backend in backends_to_test:
            dx = evaluator.evaluate(x_np, backend=backend)
            dx_val = np.array(dx) if not isinstance(dx, np.ndarray) else dx

            assert np.allclose(dx_val, dx_np), f"{backend} doesn't match NumPy"


# ============================================================================
# Test Class 8: Input Validation
# ============================================================================


class TestInputValidation:
    """Test input validation and error handling"""

    def test_zero_dimensional_input_error(self):
        """Test error on scalar input"""
        system = MockLinearSystem()
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        evaluator = DynamicsEvaluator(system, code_gen, backend_mgr)

        with pytest.raises(ValueError, match="at least 1D"):
            evaluator.evaluate(np.array(1.0), np.array(0.0))

    def test_wrong_state_dimension(self):
        """Test error on wrong state dimension"""
        system = MockLinearSystem()
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        evaluator = DynamicsEvaluator(system, code_gen, backend_mgr)

        x = np.array([1.0, 2.0])  # 2D but system is 1D!
        u = np.array([0.0])

        with pytest.raises(ValueError, match="Expected state dimension"):
            evaluator.evaluate(x, u, backend="numpy")

    def test_wrong_control_dimension(self):
        """Test error on wrong control dimension"""
        system = MockLinearSystem()
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        evaluator = DynamicsEvaluator(system, code_gen, backend_mgr)

        x = np.array([1.0])
        u = np.array([0.0, 0.0])  # 2D but system has 1 control!

        with pytest.raises(ValueError, match="Expected control dimension"):
            evaluator.evaluate(x, u, backend="numpy")

    @pytest.mark.skipif(not torch_available, reason="PyTorch not installed")
    def test_torch_dimension_validation(self):
        """Test dimension validation for PyTorch"""
        system = MockLinearSystem()
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        evaluator = DynamicsEvaluator(system, code_gen, backend_mgr)

        x = torch.tensor([1.0, 2.0])  # Wrong size
        u = torch.tensor([0.0])

        with pytest.raises(ValueError, match="Expected state dimension"):
            evaluator.evaluate(x, u, backend="torch")


# ============================================================================
# Test Class 9: Performance Tracking
# ============================================================================


class TestPerformanceTracking:
    """Test performance statistics tracking"""

    def test_initial_stats(self):
        """Test initial stats are zero"""
        system = MockLinearSystem()
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        evaluator = DynamicsEvaluator(system, code_gen, backend_mgr)

        stats: ExecutionStats = evaluator.get_stats()

        assert stats["calls"] == 0
        assert stats["total_time"] == 0.0
        assert stats["avg_time"] == 0.0

    def test_stats_increment(self):
        """Test that stats increment with calls"""
        system = MockLinearSystem()
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        evaluator = DynamicsEvaluator(system, code_gen, backend_mgr)

        x: StateVector = np.array([1.0])
        u: ControlVector = np.array([0.0])

        # Make several calls
        evaluator.evaluate(x, u)
        evaluator.evaluate(x, u)
        evaluator.evaluate(x, u)

        stats: ExecutionStats = evaluator.get_stats()

        assert stats["calls"] == 3
        assert stats["total_time"] > 0
        assert stats["avg_time"] > 0

    def test_reset_stats(self):
        """Test resetting performance stats"""
        system = MockLinearSystem()
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        evaluator = DynamicsEvaluator(system, code_gen, backend_mgr)

        x: StateVector = np.array([1.0])
        u: ControlVector = np.array([0.0])

        # Make some calls
        evaluator.evaluate(x, u)
        assert evaluator.get_stats()["calls"] > 0

        # Reset
        evaluator.reset_stats()
        stats: ExecutionStats = evaluator.get_stats()

        assert stats["calls"] == 0
        assert stats["total_time"] == 0.0


# ============================================================================
# Test Class 10: Function Caching
# ============================================================================


class TestFunctionCaching:
    """Test that generated functions are properly cached"""

    def test_function_reuse(self):
        """Test that cached functions are reused"""
        system = MockLinearSystem()
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        evaluator = DynamicsEvaluator(system, code_gen, backend_mgr)

        x = np.array([1.0])
        u = np.array([0.0])

        # First call generates function
        evaluator.evaluate(x, u, backend="numpy")
        func1 = code_gen.get_dynamics("numpy")

        # Second call reuses cached function
        evaluator.evaluate(x, u, backend="numpy")
        func2 = code_gen.get_dynamics("numpy")

        assert func1 is func2  # Same function object


# ============================================================================
# Test Class 11: String Representations
# ============================================================================


class TestStringRepresentations:
    """Test __repr__ and __str__ methods"""

    def test_repr_controlled(self):
        """Test __repr__ output for controlled system"""
        system = MockLinearSystem()
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        evaluator = DynamicsEvaluator(system, code_gen, backend_mgr)

        repr_str = repr(evaluator)

        assert "DynamicsEvaluator" in repr_str
        assert "nx=1" in repr_str
        assert "nu=1" in repr_str

    def test_repr_autonomous(self):
        """Test __repr__ output for autonomous system"""
        system = MockAutonomousSystem()
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        evaluator = DynamicsEvaluator(system, code_gen, backend_mgr)

        repr_str = repr(evaluator)

        assert "DynamicsEvaluator" in repr_str
        assert "nu=0" in repr_str
        assert "(autonomous)" in repr_str

    def test_str(self):
        """Test __str__ output"""
        system = MockLinearSystem()
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        evaluator = DynamicsEvaluator(system, code_gen, backend_mgr)

        str_repr = str(evaluator)

        assert "DynamicsEvaluator" in str_repr
        assert "calls=" in str_repr


# ============================================================================
# Test Class 12: Edge Cases
# ============================================================================


class TestEdgeCases:
    """Test edge cases and special scenarios"""

    def test_large_batch(self):
        """Test with large batch size"""
        system = MockLinearSystem()
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        evaluator = DynamicsEvaluator(system, code_gen, backend_mgr)

        batch_size = 100
        x = np.random.randn(batch_size, 1)
        u = np.random.randn(batch_size, 1)

        dx = evaluator.evaluate(x, u, backend="numpy")

        assert dx.shape == (batch_size, 1)

    def test_zero_state(self):
        """Test evaluation at zero state"""
        system = MockLinearSystem(a=2.0)
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        evaluator = DynamicsEvaluator(system, code_gen, backend_mgr)

        x = np.array([0.0])
        u = np.array([1.0])

        dx = evaluator.evaluate(x, u, backend="numpy")

        # dx = -2*0 + 1 = 1
        assert np.allclose(dx, np.array([1.0]))


# ============================================================================
# Test Class 13: Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests combining multiple features"""

    def test_full_workflow_controlled(self):
        """Test complete evaluation workflow for controlled systems"""
        system = MockLinearSystem(a=2.0)
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        evaluator = DynamicsEvaluator(system, code_gen, backend_mgr)

        # Test multiple evaluations
        for _ in range(5):
            x = np.random.randn(1)
            u = np.random.randn(1)
            dx = evaluator.evaluate(x, u)

            # Verify result shape
            assert dx.shape == (1,)

        # Check stats
        stats = evaluator.get_stats()
        assert stats["calls"] == 5

    def test_full_workflow_autonomous(self):
        """Test complete evaluation workflow for autonomous systems"""
        system = MockAutonomousSystem(a=2.0)
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        evaluator = DynamicsEvaluator(system, code_gen, backend_mgr)

        # Test multiple evaluations
        for _ in range(5):
            x = np.random.randn(1)
            dx = evaluator.evaluate(x)  # No u

            # Verify result shape
            assert dx.shape == (1,)

        # Check stats
        stats = evaluator.get_stats()
        assert stats["calls"] == 5

    @pytest.mark.skipif(
        not (torch_available and jax_available), reason="Both PyTorch and JAX required"
    )
    def test_multi_backend_workflow(self):
        """Test switching between backends"""
        system = MockLinearSystem(a=2.0)
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        evaluator = DynamicsEvaluator(system, code_gen, backend_mgr)

        x_np = np.array([1.0])
        u_np = np.array([0.5])

        # Evaluate with different backends
        dx_np = evaluator.evaluate(x_np, u_np, backend="numpy")
        dx_torch = evaluator.evaluate(x_np, u_np, backend="torch")
        dx_jax = evaluator.evaluate(x_np, u_np, backend="jax")

        # All should give same result
        assert np.allclose(dx_np, np.array(dx_torch))
        assert np.allclose(dx_np, np.array(dx_jax))


# ============================================================================
# Run Tests
# ============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
