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
Unit Tests for Core Types Module

Tests cover:
- Type alias definitions
- Type variable usage
- Function signature compliance
- Runtime type validation
- Backend compatibility
- Dimension validation
- Callback signatures
- Generic type preservation
- Documentation completeness
"""

from typing import Optional, get_type_hints

import numpy as np
import pytest

# Conditional imports
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import jax.numpy as jnp

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

from src.types.core import (  # Arrays; Vectors; Matrices; Dimensions; Equilibria; Functions; Callbacks; Type variables
    ArrayLike,
    C,
    Constraint,
    ControllabilityMatrix,
    ControlMatrix,
    ControlPolicy,
    ControlVector,
    CostFunction,
    CostMatrix,
    CovarianceMatrix,
    DiffusionFunction,
    DiffusionMatrix,
    DimensionTuple,
    DynamicsFunction,
    EquilibriumControl,
    EquilibriumIdentifier,
    EquilibriumName,
    EquilibriumPoint,
    EquilibriumState,
    FeedthroughMatrix,
    GainMatrix,
    IntegerLike,
    IntegrationCallback,
    MatrixT,
    NoiseVector,
    NumpyArray,
    ObservabilityMatrix,
    OptimizationCallback,
    OutputFunction,
    OutputMatrix,
    OutputVector,
    ParameterVector,
    ResidualVector,
    S,
    ScalarLike,
    SimulationCallback,
    StateEstimator,
    StateMatrix,
    StateVector,
    SystemDimensions,
    T,
)

# ============================================================================
# Test Fixtures - Example Functions with Type Annotations
# ============================================================================


def example_dynamics(x: StateVector, u: ControlVector) -> StateVector:
    """Example dynamics function for testing."""
    return 0.9 * x + u


def example_output(x: StateVector) -> OutputVector:
    """Example output function for testing."""
    return x[:2]  # First two states


def example_diffusion(x: StateVector, u: ControlVector) -> DiffusionMatrix:
    """Example diffusion function for testing."""
    return 0.1 * np.eye(len(x))


def example_policy(x: StateVector) -> ControlVector:
    """Example control policy for testing."""
    K = np.array([[1.0, 0.5]])
    return -K @ x


def example_estimator(y: OutputVector) -> StateVector:
    """Example state estimator for testing."""
    return np.concatenate([y, np.zeros(1)])  # Add unobserved state


def example_cost(x: StateVector, u: ControlVector) -> float:
    """Example cost function for testing."""
    return float(x.T @ x + 0.1 * u.T @ u)


def example_constraint(x: StateVector, u: ControlVector) -> ArrayLike:
    """Example constraint for testing."""
    return np.abs(u) - 1.0  # |u| ≤ 1


def example_integration_callback(t: float, x: StateVector) -> bool:
    """Example integration callback for testing."""
    return np.linalg.norm(x) > 100  # Stop if diverging


def example_simulation_callback(k: int, x: StateVector, u: ControlVector) -> None:
    """Example simulation callback for testing."""
    if k % 10 == 0:
        pass  # Would log here


def example_optimization_callback(x: ArrayLike) -> None:
    """Example optimization callback for testing."""
    # Would monitor convergence


# ============================================================================
# Test Type Aliases Are Defined
# ============================================================================


class TestTypeAliasesExist:
    """Test that all type aliases are defined and importable."""

    def test_array_types_exist(self):
        """Test basic array types are defined."""
        assert ArrayLike is not None
        assert NumpyArray is not None
        assert ScalarLike is not None
        assert IntegerLike is not None

    def test_vector_types_exist(self):
        """Test vector types are defined."""
        assert StateVector is not None
        assert ControlVector is not None
        assert OutputVector is not None
        assert NoiseVector is not None
        assert ParameterVector is not None
        assert ResidualVector is not None

    def test_matrix_types_exist(self):
        """Test matrix types are defined."""
        assert StateMatrix is not None
        assert ControlMatrix is not None
        assert OutputMatrix is not None
        assert DiffusionMatrix is not None
        assert CovarianceMatrix is not None
        assert GainMatrix is not None

    def test_dimension_types_exist(self):
        """Test dimension types are defined."""
        assert SystemDimensions is not None
        assert DimensionTuple is not None

    def test_equilibrium_types_exist(self):
        """Test equilibrium types are defined."""
        assert EquilibriumState is not None
        assert EquilibriumControl is not None
        assert EquilibriumPoint is not None
        assert EquilibriumName is not None

    def test_function_types_exist(self):
        """Test function types are defined."""
        assert DynamicsFunction is not None
        assert OutputFunction is not None
        assert ControlPolicy is not None
        assert CostFunction is not None

    def test_type_variables_exist(self):
        """Test generic type variables are defined."""
        assert T is not None
        assert S is not None
        assert C is not None
        assert MatrixT is not None


# ============================================================================
# Test Array Type Compatibility
# ============================================================================


class TestArrayTypeCompatibility:
    """Test that array types work with different backends."""

    def test_numpy_arrays_match_arraylike(self):
        """Test NumPy arrays satisfy ArrayLike."""
        x: ArrayLike = np.array([1.0, 2.0, 3.0])
        assert isinstance(x, np.ndarray)

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_torch_tensors_match_arraylike(self):
        """Test PyTorch tensors satisfy ArrayLike."""
        import torch

        x: ArrayLike = torch.tensor([1.0, 2.0, 3.0])
        assert isinstance(x, torch.Tensor)

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_jax_arrays_match_arraylike(self):
        """Test JAX arrays satisfy ArrayLike."""
        import jax.numpy as jnp

        x: ArrayLike = jnp.array([1.0, 2.0, 3.0])
        assert isinstance(x, jnp.ndarray)

    def test_numpy_array_type_specific(self):
        """Test NumpyArray type is np.ndarray."""
        x: NumpyArray = np.array([1, 2, 3])
        assert isinstance(x, np.ndarray)


# ============================================================================
# Test Vector Types
# ============================================================================


class TestVectorTypes:
    """Test semantic vector types."""

    def test_state_vector_single(self):
        """Test single state vector."""
        x: StateVector = np.array([1.0, 0.0, 0.5])
        assert x.shape == (3,)

    def test_state_vector_batched(self):
        """Test batched state vectors."""
        x_batch: StateVector = np.random.randn(100, 3)
        assert x_batch.shape == (100, 3)

    def test_control_vector_single(self):
        """Test single control vector."""
        u: ControlVector = np.array([0.5])
        assert u.shape == (1,)

    def test_control_vector_sequence(self):
        """Test control sequence."""
        u_seq: ControlVector = np.zeros((100, 1))
        assert u_seq.shape == (100, 1)

    def test_output_vector(self):
        """Test output vector."""
        y: OutputVector = np.array([1.0, 0.0])
        assert y.shape == (2,)

    def test_noise_vector(self):
        """Test noise vector."""
        w: NoiseVector = np.random.randn(2)
        assert w.shape == (2,)

    def test_parameter_vector(self):
        """Test parameter vector."""
        theta: ParameterVector = np.array([1.0, 0.5, 2.0])
        assert theta.shape == (3,)

    def test_residual_vector(self):
        """Test residual vector."""
        residual: ResidualVector = np.array([0.1, -0.05])
        assert residual.shape == (2,)


# ============================================================================
# Test Matrix Types
# ============================================================================


class TestMatrixTypes:
    """Test semantic matrix types."""

    def test_state_matrix(self):
        """Test state matrix (nx, nx)."""
        A: StateMatrix = np.array([[0.9, 0.1], [-0.1, 0.8]])
        assert A.shape == (2, 2)

    def test_control_matrix(self):
        """Test control matrix (nx, nu)."""
        B: ControlMatrix = np.array([[0], [1]])
        assert B.shape == (2, 1)

    def test_output_matrix(self):
        """Test output matrix (ny, nx)."""
        C: OutputMatrix = np.eye(3)
        assert C.shape == (3, 3)

    def test_diffusion_matrix(self):
        """Test diffusion matrix (nx, nw)."""
        G: DiffusionMatrix = 0.1 * np.eye(2)
        assert G.shape == (2, 2)

    def test_covariance_matrix(self):
        """Test covariance matrix is symmetric."""
        P: CovarianceMatrix = np.array([[1.0, 0.5], [0.5, 2.0]])
        # Should be symmetric
        np.testing.assert_allclose(P, P.T)

    def test_gain_matrix_lqr(self):
        """Test LQR gain matrix (nu, nx)."""
        K: GainMatrix = np.array([[1.0, 0.5]])
        assert K.shape == (1, 2)

    def test_gain_matrix_kalman(self):
        """Test Kalman gain matrix (nx, ny)."""
        L: GainMatrix = np.array([[0.1], [0.2]])
        assert L.shape == (2, 1)

    def test_controllability_matrix(self):
        """Test controllability matrix (nx, nx*nu)."""
        C: ControllabilityMatrix = np.random.randn(3, 6)  # nx=3, nu=2
        assert C.shape == (3, 6)

    def test_observability_matrix(self):
        """Test observability matrix (nx*ny, nx)."""
        O: ObservabilityMatrix = np.random.randn(6, 3)  # nx=3, ny=2
        assert O.shape == (6, 3)


# ============================================================================
# Test Dimension Types
# ============================================================================


class TestDimensionTypes:
    """Test dimension specification types."""

    def test_system_dimensions_dict(self):
        """Test SystemDimensions TypedDict."""
        dims: SystemDimensions = {
            "nx": 3,
            "nu": 2,
            "ny": 3,
        }

        assert dims["nx"] == 3
        assert dims["nu"] == 2
        assert dims["ny"] == 3

    def test_system_dimensions_with_noise(self):
        """Test SystemDimensions with stochastic fields."""
        dims: SystemDimensions = {
            "nx": 2,
            "nu": 1,
            "ny": 2,
            "nw": 2,
        }

        assert dims["nw"] == 2

    def test_system_dimensions_partial(self):
        """Test partial SystemDimensions (total=False)."""
        dims: SystemDimensions = {"nx": 3}
        assert "nx" in dims
        assert "nu" not in dims  # Partial is OK

    def test_dimension_tuple(self):
        """Test simple dimension tuple."""
        dims: DimensionTuple = (3, 2, 3)
        nx, nu, ny = dims

        assert nx == 3
        assert nu == 2
        assert ny == 3


# ============================================================================
# Test Equilibrium Types
# ============================================================================


class TestEquilibriumTypes:
    """Test equilibrium specification types."""

    def test_equilibrium_state(self):
        """Test equilibrium state is StateVector."""
        x_eq: EquilibriumState = np.zeros(3)
        assert x_eq.shape == (3,)

    def test_equilibrium_control(self):
        """Test equilibrium control is ControlVector."""
        u_eq: EquilibriumControl = np.zeros(1)
        assert u_eq.shape == (1,)

    def test_equilibrium_point_tuple(self):
        """Test equilibrium point as tuple."""
        equilibrium: EquilibriumPoint = (np.zeros(3), np.zeros(1))
        x_eq, u_eq = equilibrium

        assert x_eq.shape == (3,)
        assert u_eq.shape == (1,)

    def test_equilibrium_name_string(self):
        """Test equilibrium name is string."""
        name: EquilibriumName = "origin"
        assert isinstance(name, str)

    def test_equilibrium_identifier_by_name(self):
        """Test identifier can be name."""
        identifier: EquilibriumIdentifier = "upright"
        assert isinstance(identifier, str)

    def test_equilibrium_identifier_by_state(self):
        """Test identifier can be state."""
        identifier: EquilibriumIdentifier = np.zeros(3)
        assert isinstance(identifier, np.ndarray)


# ============================================================================
# Test Function Type Signatures
# ============================================================================


class TestFunctionTypes:
    """Test function type signatures match definitions."""

    def test_dynamics_function_signature(self):
        """Test DynamicsFunction signature."""
        f: DynamicsFunction = example_dynamics

        x = np.array([1.0, 0.0])
        u = np.array([0.5])

        x_next = f(x, u)
        assert isinstance(x_next, np.ndarray)
        assert x_next.shape == x.shape

    def test_output_function_signature(self):
        """Test OutputFunction signature."""
        h: OutputFunction = example_output

        x = np.array([1.0, 0.0, 0.5])
        y = h(x)

        assert isinstance(y, np.ndarray)
        assert y.shape == (2,)

    def test_diffusion_function_signature(self):
        """Test DiffusionFunction signature."""
        g: DiffusionFunction = example_diffusion

        x = np.array([1.0, 0.0])
        u = np.array([0.5])

        G = g(x, u)
        assert G.shape == (2, 2)

    def test_control_policy_signature(self):
        """Test ControlPolicy signature."""
        policy: ControlPolicy = example_policy

        x = np.array([1.0, 0.0])
        u = policy(x)

        assert u.shape == (1,)

    def test_state_estimator_signature(self):
        """Test StateEstimator signature."""
        estimator: StateEstimator = example_estimator

        y = np.array([1.0, 0.0])
        x_hat = estimator(y)

        assert x_hat.shape == (3,)

    def test_cost_function_signature(self):
        """Test CostFunction signature."""
        cost: CostFunction = example_cost

        x = np.array([1.0, 0.0])
        u = np.array([0.5])

        J = cost(x, u)
        assert isinstance(J, float)

    def test_constraint_function_signature(self):
        """Test Constraint signature."""
        constraint: Constraint = example_constraint

        x = np.array([1.0, 0.0])
        u = np.array([0.5])

        c = constraint(x, u)
        assert isinstance(c, np.ndarray)


# ============================================================================
# Test Callback Types
# ============================================================================


class TestCallbackTypes:
    """Test callback function signatures."""

    def test_integration_callback_signature(self):
        """Test IntegrationCallback signature."""
        callback: IntegrationCallback = example_integration_callback

        t = 0.5
        x = np.array([1.0, 0.0])

        stop = callback(t, x)
        # Accept both Python bool and NumPy bool
        assert isinstance(stop, (bool, np.bool_))

    def test_simulation_callback_signature(self):
        """Test SimulationCallback signature."""
        callback: SimulationCallback = example_simulation_callback

        k = 10
        x = np.array([1.0, 0.0])
        u = np.array([0.5])

        # Should not raise
        callback(k, x, u)

    def test_optimization_callback_signature(self):
        """Test OptimizationCallback signature."""
        callback: OptimizationCallback = example_optimization_callback

        x = np.array([1.0, 0.0, 0.5])

        # Should not raise
        callback(x)


# ============================================================================
# Test Type Variable Usage
# ============================================================================


class TestTypeVariables:
    """Test generic type variables."""

    def test_generic_array_type_preservation(self):
        """Test that generic T preserves array type."""

        def scale(x: T, factor: float) -> T:
            return x * factor

        # NumPy
        x_np = np.array([1, 2, 3])
        result_np = scale(x_np, 2.0)
        assert isinstance(result_np, np.ndarray)

        # PyTorch
        if TORCH_AVAILABLE:
            x_torch = torch.tensor([1, 2, 3])
            result_torch = scale(x_torch, 2.0)
            assert isinstance(result_torch, torch.Tensor)

    def test_matrix_type_variable(self):
        """Test MatrixT for generic matrix operations."""

        def check_symmetry(M: MatrixT) -> bool:
            M_np = np.asarray(M)
            return np.allclose(M_np, M_np.T)

        # State matrix
        A: StateMatrix = np.array([[1, 0], [0, 1]])
        assert check_symmetry(A)

        # Covariance matrix
        P: CovarianceMatrix = np.eye(3)
        assert check_symmetry(P)


# ============================================================================
# Test Semantic Type Equivalence
# ============================================================================


class TestSemanticTypeEquivalence:
    """Test that semantic types are equivalent to base types."""

    def test_state_vector_is_arraylike(self):
        """Test StateVector is same as ArrayLike."""
        # Both should accept same values
        x1: StateVector = np.array([1.0, 0.0, 0.5])
        x2: ArrayLike = np.array([1.0, 0.0, 0.5])

        # Types are aliases
        assert StateVector == ArrayLike

    def test_state_matrix_is_arraylike(self):
        """Test StateMatrix is same as ArrayLike."""
        A1: StateMatrix = np.eye(2)
        A2: ArrayLike = np.eye(2)

        assert StateMatrix == ArrayLike

    def test_equilibrium_state_is_state_vector(self):
        """Test EquilibriumState is StateVector."""
        x_eq1: EquilibriumState = np.zeros(3)
        x_eq2: StateVector = np.zeros(3)

        assert EquilibriumState == StateVector


# ============================================================================
# Test Multi-Backend Usage
# ============================================================================


class TestMultiBackendUsage:
    """Test types work across different backends."""

    def test_function_accepts_any_backend_numpy(self):
        """Test function with ArrayLike accepts NumPy."""

        def process(x: ArrayLike) -> ArrayLike:
            return x * 2

        x_np = np.array([1, 2, 3])
        result = process(x_np)
        assert isinstance(result, np.ndarray)

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_function_accepts_any_backend_torch(self):
        """Test function with ArrayLike accepts PyTorch."""

        def process(x: ArrayLike) -> ArrayLike:
            return x * 2

        x_torch = torch.tensor([1.0, 2.0, 3.0])
        result = process(x_torch)
        assert isinstance(result, torch.Tensor)

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_function_accepts_any_backend_jax(self):
        """Test function with ArrayLike accepts JAX."""

        def process(x: ArrayLike) -> ArrayLike:
            return x * 2

        x_jax = jnp.array([1.0, 2.0, 3.0])
        result = process(x_jax)
        assert isinstance(result, jnp.ndarray)


# ============================================================================
# Test Realistic Usage Patterns
# ============================================================================


class TestRealisticUsagePatterns:
    """Test types in realistic control/estimation scenarios."""

    def test_lqr_controller_pattern(self):
        """Test LQR controller type usage."""

        def lqr_control(x: StateVector, K: GainMatrix) -> ControlVector:
            return -K @ x

        x = np.array([1.0, 0.0])
        K = np.array([[1.0, 0.5]])

        u = lqr_control(x, K)

        assert u.shape == (1,)
        assert isinstance(u, np.ndarray)

    def test_kalman_filter_pattern(self):
        """Test Kalman filter type usage."""

        def kalman_update(
            x_pred: StateVector,
            y: OutputVector,
            C: OutputMatrix,
            L: GainMatrix,
        ) -> StateVector:
            innovation = y - C @ x_pred
            return x_pred + L @ innovation

        x_pred = np.array([1.0, 0.0])
        y = np.array([0.9])
        C = np.array([[1.0, 0.0]])
        L = np.array([[0.5], [0.0]])

        x_updated = kalman_update(x_pred, y, C, L)

        assert x_updated.shape == (2,)

    def test_mpc_pattern(self):
        """Test MPC type usage."""

        def mpc_solve(
            x0: StateVector,
            Ad: StateMatrix,
            Bd: ControlMatrix,
            Q: CostMatrix,
            R: CostMatrix,
            horizon: int,
        ) -> ControlVector:
            # Simplified MPC (just return first control)
            u_opt = -np.linalg.pinv(R + Bd.T @ Q @ Bd) @ (Bd.T @ Q @ Ad @ x0)
            return u_opt

        x0 = np.array([1.0, 0.0])
        Ad = np.eye(2)
        Bd = np.array([[0], [1]])
        Q = np.eye(2)
        R = np.eye(1)

        u = mpc_solve(x0, Ad, Bd, Q, R, horizon=10)

        assert u.shape == (1,)

    def test_sde_simulation_pattern(self):
        """Test stochastic dynamics type usage."""

        def sde_step(
            x: StateVector,
            u: ControlVector,
            w: NoiseVector,
            Ac: StateMatrix,
            Bc: ControlMatrix,
            Gc: DiffusionMatrix,
            dt: float,
        ) -> StateVector:
            # Euler-Maruyama step
            drift = Ac @ x + Bc @ u
            diffusion = Gc @ w
            return x + drift * dt + diffusion * np.sqrt(dt)

        x = np.array([1.0, 0.0])
        u = np.array([0.5])
        w = np.random.randn(2)
        Ac = np.array([[0, 1], [-1, 0]])
        Bc = np.array([[0], [1]])
        Gc = 0.1 * np.eye(2)
        dt = 0.01

        x_next = sde_step(x, u, w, Ac, Bc, Gc, dt)

        assert x_next.shape == (2,)


# ============================================================================
# Test Function Composition
# ============================================================================


class TestFunctionComposition:
    """Test that typed functions compose correctly."""

    def test_dynamics_to_control_composition(self):
        """Test composing dynamics and control functions."""

        def dynamics(x: StateVector, u: ControlVector) -> StateVector:
            A = np.array([[0.9, 0.1], [-0.1, 0.8]])
            B = np.array([[0], [1]])
            return A @ x + B @ u

        def controller(x: StateVector) -> ControlVector:
            K = np.array([[1.0, 0.5]])
            return -K @ x

        # Compose: closed-loop dynamics
        def closed_loop(x: StateVector) -> StateVector:
            u = controller(x)
            return dynamics(x, u)

        x = np.array([1.0, 0.0])
        x_next = closed_loop(x)

        assert x_next.shape == (2,)

    def test_estimator_to_controller_composition(self):
        """Test composing estimator and controller (output feedback)."""

        def estimator(y: OutputVector) -> StateVector:
            # Simple estimator (assume y = x for test)
            return y

        def controller(x: StateVector) -> ControlVector:
            K = np.array([[1.0, 0.5]])
            return -K @ x

        # Compose: output feedback controller
        def output_feedback(y: OutputVector) -> ControlVector:
            x_hat = estimator(y)
            return controller(x_hat)

        y = np.array([1.0, 0.0])
        u = output_feedback(y)

        assert u.shape == (1,)


# ============================================================================
# Test Docstring Completeness
# ============================================================================


class TestDocstrings:
    """Test that types have proper documentation."""

    def test_arraylike_exists(self):
        """Test ArrayLike type exists."""
        assert ArrayLike is not None

    def test_state_vector_exists(self):
        """Test StateVector type exists."""
        assert StateVector is not None

    def test_all_vector_types_documented(self):
        """Test all vector types exist (documentation test)."""
        vector_types = [
            StateVector,
            ControlVector,
            OutputVector,
            NoiseVector,
            ParameterVector,
            ResidualVector,
        ]

        for vtype in vector_types:
            assert vtype is not None

    def test_all_matrix_types_documented(self):
        """Test all matrix types exist."""
        matrix_types = [
            StateMatrix,
            ControlMatrix,
            OutputMatrix,
            DiffusionMatrix,
            FeedthroughMatrix,
            CovarianceMatrix,
            GainMatrix,
            ControllabilityMatrix,
            ObservabilityMatrix,
            CostMatrix,
        ]

        for mtype in matrix_types:
            assert mtype is not None


# ============================================================================
# Test Edge Cases
# ============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_control_autonomous(self):
        """Test empty control for autonomous systems."""
        u_autonomous: Optional[ControlVector] = None

        def autonomous_dynamics(x: StateVector, u: Optional[ControlVector] = None) -> StateVector:
            return 0.9 * x

        x = np.array([1.0, 0.0])
        x_next = autonomous_dynamics(x, u_autonomous)

        assert x_next.shape == (2,)

    def test_scalar_state_system(self):
        """Test 1D system (nx=1)."""
        x: StateVector = np.array([1.0])
        A: StateMatrix = np.array([[0.9]])

        assert x.shape == (1,)
        assert A.shape == (1, 1)

    def test_high_dimensional_system(self):
        """Test high-dimensional system."""
        nx = 100
        x: StateVector = np.random.randn(nx)
        A: StateMatrix = np.eye(nx)

        assert x.shape == (nx,)
        assert A.shape == (nx, nx)

    def test_zero_dimensional_control(self):
        """Test autonomous system (nu=0)."""
        # Autonomous: u should be None or empty
        u_none: Optional[ControlVector] = None
        u_empty: ControlVector = np.array([])

        assert u_none is None
        assert u_empty.shape == (0,)


# ============================================================================
# Test Type Checking (Static)
# ============================================================================


class TestStaticTypeChecking:
    """Test that types enable static type checking."""

    def test_type_hints_extractable(self):
        """Test that function type hints can be extracted."""
        hints = get_type_hints(example_dynamics)

        assert "x" in hints
        assert "u" in hints
        assert "return" in hints

    def test_typed_function_has_annotations(self):
        """Test typed functions have __annotations__."""
        assert hasattr(example_dynamics, "__annotations__")
        assert "x" in example_dynamics.__annotations__

    def test_callback_has_correct_signature(self):
        """Test callback signature is correct."""
        sig = get_type_hints(example_integration_callback)

        assert "t" in sig
        assert "x" in sig
        assert "return" in sig


# ============================================================================
# Test Practical Examples from Docstrings
# ============================================================================


class TestDocstringExamples:
    """Test that examples from docstrings actually work."""

    def test_arraylike_example(self):
        """Test ArrayLike example."""

        def process(data: ArrayLike) -> ArrayLike:
            return data * 2

        x = np.array([1, 2, 3])
        result = process(x)
        np.testing.assert_array_equal(result, [2, 4, 6])

    def test_state_vector_example(self):
        """Test StateVector examples."""
        # Single state
        x: StateVector = np.array([1.0, 0.0, 0.5])
        assert x.shape == (3,)

        # Batched
        x_batch: StateVector = np.random.randn(100, 3)
        assert x_batch.shape == (100, 3)

    def test_equilibrium_point_example(self):
        """Test EquilibriumPoint example."""
        equilibrium: EquilibriumPoint = (np.zeros(3), np.zeros(1))
        x_eq, u_eq = equilibrium

        assert x_eq.shape == (3,)
        assert u_eq.shape == (1,)

    def test_lqr_gain_example(self):
        """Test GainMatrix LQR example."""
        K_lqr: GainMatrix = np.array([[1.0, 0.5]])
        x = np.array([1.0, 0.0])
        u = -K_lqr @ x

        expected = np.array([-1.0])
        np.testing.assert_allclose(u, expected)


# ============================================================================
# Test Type Safety
# ============================================================================


class TestTypeSafety:
    """Test that types catch common mistakes (when using type checker)."""

    def test_dimension_mismatch_caught_at_runtime(self):
        """Test dimension mismatch raises error."""
        A = np.array([[1, 0], [0, 1]])
        x = np.array([1])  # Wrong dimension

        # NumPy will raise on matrix multiply
        with pytest.raises((ValueError, IndexError)):
            _ = A @ x  # Shape mismatch

    def test_none_for_autonomous_accepted(self):
        """Test None is valid for autonomous systems."""

        def autonomous_f(x: StateVector, u: Optional[ControlVector] = None) -> StateVector:
            return 0.9 * x

        x = np.array([1.0, 0.0])
        x_next = autonomous_f(x, None)

        assert x_next.shape == (2,)


# ============================================================================
# Test Import Patterns
# ============================================================================


class TestImportPatterns:
    """Test common import patterns work."""

    def test_import_multiple_types(self):
        """Test importing multiple types together."""
        from src.types.core import (
            ControlVector,
            GainMatrix,
            StateMatrix,
            StateVector,
        )

        assert StateVector is not None
        assert ControlVector is not None
        assert StateMatrix is not None
        assert GainMatrix is not None

    def test_import_all_from_core(self):
        """Test importing all from core."""
        from src.types import core

        # Should have all exported types
        assert hasattr(core, "StateVector")
        assert hasattr(core, "ControlVector")
        assert hasattr(core, "StateMatrix")
        assert hasattr(core, "GainMatrix")


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests with complete workflows."""

    def test_complete_lqr_workflow(self):
        """Test complete LQR design with types."""
        # System matrices
        A: StateMatrix = np.array([[0.9, 0.1], [-0.1, 0.8]])
        B: ControlMatrix = np.array([[0], [1]])

        # Cost matrices
        Q: CostMatrix = np.eye(2)
        R: CostMatrix = np.array([[0.1]])

        # Equilibrium
        x_eq: EquilibriumState = np.zeros(2)
        u_eq: EquilibriumControl = np.zeros(1)

        # Would solve DARE here...
        # K: GainMatrix = solve_dare(A, B, Q, R)

        # For test, use arbitrary gain
        K: GainMatrix = np.array([[1.0, 0.5]])

        # Controller
        policy: ControlPolicy = lambda x: -K @ x

        # Simulate
        x = np.array([1.0, 1.0])
        u = policy(x)

        assert u.shape == (1,)

    def test_complete_kalman_workflow(self):
        """Test complete Kalman filter with types."""
        # System matrices
        A: StateMatrix = np.eye(2)
        C: OutputMatrix = np.array([[1, 0]])

        # Noise covariances
        Q: CovarianceMatrix = 0.01 * np.eye(2)
        R: CovarianceMatrix = np.array([[0.1]])

        # Would solve DARE here...
        # L: GainMatrix = solve_dare(A.T, C.T, Q, R).T

        # For test, use arbitrary gain
        L: GainMatrix = np.array([[0.5], [0.1]])

        # Estimator
        x_hat: StateVector = np.array([0.0, 0.0])
        y: OutputVector = np.array([1.0])

        estimator: StateEstimator = lambda y_meas: x_hat + L @ (y_meas - C @ x_hat)

        x_hat_new = estimator(y)

        assert x_hat_new.shape == (2,)

    def test_complete_identification_pattern(self):
        """Test system identification type usage."""
        # Data
        u_data: ControlSequence = np.random.randn(100, 1)
        y_data: OutputSequence = np.random.randn(100, 2)

        # Build Hankel matrix (simplified)
        def build_hankel(data: ArrayLike, rows: int, cols: int) -> ArrayLike:
            n_samples = data.shape[0]
            H = np.zeros((rows * data.shape[1], cols))
            for i in range(rows):
                for j in range(cols):
                    if i + j < n_samples:
                        H[i * data.shape[1] : (i + 1) * data.shape[1], j] = data[i + j]
            return H

        H = build_hankel(y_data, rows=10, cols=20)

        # SVD for order selection
        U, s, Vt = np.linalg.svd(H)

        assert H.shape[0] == 10 * 2  # rows * ny
        assert H.shape[1] == 20


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
