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
Unit Tests for Linearization Types Module

Tests cover:
- Linearization result tuple types
- Deterministic vs stochastic linearization
- Continuous vs discrete linearization
- Observation linearization
- Full linearization with output
- Jacobian-specific types
- Type unpacking patterns
- Polymorphic linearization handling
- Cache key types
- Type aliases and consistency
- Documentation example correctness
"""


import numpy as np
import pytest

from src.types.core import (
    ControlMatrix,
    DiffusionMatrix,
    FeedthroughMatrix,
    OutputMatrix,
    StateMatrix,
)
from src.types.linearization import (  # Main result types; Time-domain specific aliases; Full linearization; Jacobian-specific; Cache
    ContinuousLinearization,
    ContinuousStochasticLinearization,
    ControlJacobian,
    DeterministicLinearization,
    DiffusionJacobian,
    DiscreteLinearization,
    DiscreteStochasticLinearization,
    FullLinearization,
    FullStochasticLinearization,
    LinearizationCacheKey,
    LinearizationResult,
    ObservationLinearization,
    OutputJacobian,
    StateJacobian,
    StochasticLinearization,
)

# ============================================================================
# Test Deterministic Linearization
# ============================================================================


class TestDeterministicLinearization:
    """Test deterministic linearization type."""

    def test_deterministic_linearization_tuple_structure(self):
        """Test DeterministicLinearization is 2-tuple."""
        # Create example matrices
        A: StateMatrix = np.array([[0, 1], [-1, 0]])
        B: ControlMatrix = np.array([[0], [1]])

        result: DeterministicLinearization = (A, B)

        assert len(result) == 2
        assert isinstance(result, tuple)

    def test_deterministic_linearization_unpacking(self):
        """Test unpacking DeterministicLinearization."""
        A = np.random.randn(3, 3)
        B = np.random.randn(3, 2)

        result: DeterministicLinearization = (A, B)

        # Unpack
        A_unpacked, B_unpacked = result

        assert np.array_equal(A_unpacked, A)
        assert np.array_equal(B_unpacked, B)

    def test_deterministic_linearization_indexing(self):
        """Test indexing DeterministicLinearization."""
        A = np.eye(2)
        B = np.ones((2, 1))

        result: DeterministicLinearization = (A, B)

        assert np.array_equal(result[0], A)
        assert np.array_equal(result[1], B)

    def test_deterministic_linearization_continuous(self):
        """Test continuous system linearization."""
        # Continuous: dx/dt = Ac*x + Bc*u
        Ac = np.array([[0, 1], [-2, -3]])
        Bc = np.array([[0], [1]])

        result: DeterministicLinearization = (Ac, Bc)

        # Continuous stability: Re(λ) < 0
        eigenvalues = np.linalg.eigvals(result[0])
        is_stable = np.all(np.real(eigenvalues) < 0)

        assert is_stable

    def test_deterministic_linearization_discrete(self):
        """Test discrete system linearization."""
        # Discrete: x[k+1] = Ad*x[k] + Bd*u[k]
        Ad = np.array([[0.9, 0.1], [-0.1, 0.9]])
        Bd = np.array([[0.01], [0.05]])

        result: DeterministicLinearization = (Ad, Bd)

        # Discrete stability: |λ| < 1
        eigenvalues = np.linalg.eigvals(result[0])
        is_stable = np.all(np.abs(eigenvalues) < 1.0)

        assert is_stable


# ============================================================================
# Test Stochastic Linearization
# ============================================================================


class TestStochasticLinearization:
    """Test stochastic linearization type."""

    def test_stochastic_linearization_tuple_structure(self):
        """Test StochasticLinearization is 3-tuple."""
        A = np.array([[0, 1], [-1, -0.5]])
        B = np.array([[0], [1]])
        G = np.array([[0.1], [0.2]])

        result: StochasticLinearization = (A, B, G)

        assert len(result) == 3
        assert isinstance(result, tuple)

    def test_stochastic_linearization_unpacking(self):
        """Test unpacking StochasticLinearization."""
        A = np.random.randn(2, 2)
        B = np.random.randn(2, 1)
        G = np.random.randn(2, 2)

        result: StochasticLinearization = (A, B, G)

        # Unpack all three
        A_unpacked, B_unpacked, G_unpacked = result

        assert np.array_equal(A_unpacked, A)
        assert np.array_equal(B_unpacked, B)
        assert np.array_equal(G_unpacked, G)

    def test_stochastic_linearization_indexing(self):
        """Test indexing StochasticLinearization."""
        A = np.eye(2)
        B = np.ones((2, 1))
        G = 0.1 * np.eye(2)

        result: StochasticLinearization = (A, B, G)

        assert np.array_equal(result[0], A)
        assert np.array_equal(result[1], B)
        assert np.array_equal(result[2], G)

    def test_stochastic_linearization_continuous_sde(self):
        """Test continuous SDE linearization."""
        # dx = (Ac*x + Bc*u)dt + Gc*dW
        Ac = np.array([[-0.5, 1], [-1, -0.5]])
        Bc = np.array([[0], [1]])
        Gc = np.array([[0.1, 0], [0, 0.2]])

        result: StochasticLinearization = (Ac, Bc, Gc)

        # Process noise covariance
        Q = result[2] @ result[2].T

        assert Q.shape == (2, 2)
        assert np.all(np.linalg.eigvals(Q) >= 0)  # PSD

    def test_stochastic_linearization_discrete(self):
        """Test discrete stochastic linearization."""
        # x[k+1] = Ad*x[k] + Bd*u[k] + Gd*w[k]
        Ad = np.array([[0.95, 0.05], [-0.05, 0.95]])
        Bd = np.array([[0.01], [0.02]])
        Gd = np.array([[0.1], [0.1]])

        result: StochasticLinearization = (Ad, Bd, Gd)

        # Discrete noise covariance
        Q = result[2] @ result[2].T

        assert Q.shape == (2, 2)


# ============================================================================
# Test Polymorphic LinearizationResult
# ============================================================================


class TestLinearizationResult:
    """Test polymorphic LinearizationResult union type."""

    def test_linearization_result_deterministic(self):
        """Test LinearizationResult with deterministic system."""
        A = np.random.randn(3, 3)
        B = np.random.randn(3, 1)

        result: LinearizationResult = (A, B)

        assert len(result) == 2

    def test_linearization_result_stochastic(self):
        """Test LinearizationResult with stochastic system."""
        A = np.random.randn(3, 3)
        B = np.random.randn(3, 1)
        G = np.random.randn(3, 2)

        result: LinearizationResult = (A, B, G)

        assert len(result) == 3

    def test_linearization_result_polymorphic_unpacking(self):
        """Test polymorphic unpacking pattern from docstring."""
        # Deterministic
        result_det: LinearizationResult = (np.eye(2), np.ones((2, 1)))

        A_det = result_det[0]
        B_det = result_det[1]
        assert len(result_det) == 2

        # Stochastic
        result_stoch: LinearizationResult = (np.eye(2), np.ones((2, 1)), 0.1 * np.eye(2))

        A_stoch = result_stoch[0]
        B_stoch = result_stoch[1]
        if len(result_stoch) == 3:
            G_stoch = result_stoch[2]
            has_diffusion = True
        else:
            has_diffusion = False

        assert has_diffusion
        assert G_stoch.shape == (2, 2)

    def test_linearization_result_type_checking(self):
        """Test checking if result is stochastic."""

        def is_stochastic(result: LinearizationResult) -> bool:
            """Check if linearization includes diffusion."""
            return len(result) == 3

        det_result: LinearizationResult = (np.eye(2), np.ones((2, 1)))
        stoch_result: LinearizationResult = (np.eye(2), np.ones((2, 1)), 0.1 * np.eye(2))

        assert not is_stochastic(det_result)
        assert is_stochastic(stoch_result)

    def test_linearization_result_analysis_function(self):
        """Test polymorphic analysis function from docstring."""

        def analyze_linearization(result: LinearizationResult) -> dict:
            """Analyze linearization (works with both types)."""
            A = result[0]
            B = result[1]

            info = {"eigenvalues": np.linalg.eigvals(A), "is_stochastic": len(result) == 3}

            if len(result) == 3:
                G = result[2]
                info["process_noise_cov"] = G @ G.T

            return info

        # Test with deterministic
        det_result: LinearizationResult = (np.eye(2), np.zeros((2, 1)))
        info_det = analyze_linearization(det_result)

        assert "eigenvalues" in info_det
        assert not info_det["is_stochastic"]
        assert "process_noise_cov" not in info_det

        # Test with stochastic
        stoch_result: LinearizationResult = (np.eye(2), np.zeros((2, 1)), 0.1 * np.eye(2))
        info_stoch = analyze_linearization(stoch_result)

        assert info_stoch["is_stochastic"]
        assert "process_noise_cov" in info_stoch


# ============================================================================
# Test Observation Linearization
# ============================================================================


class TestObservationLinearization:
    """Test observation/output linearization type."""

    def test_observation_linearization_structure(self):
        """Test ObservationLinearization is (C, D) tuple."""
        C: OutputMatrix = np.array([[1, 0], [0, 1]])
        D: FeedthroughMatrix = np.zeros((2, 1))

        result: ObservationLinearization = (C, D)

        assert len(result) == 2

    def test_observation_linearization_unpacking(self):
        """Test unpacking observation linearization."""
        C = np.random.randn(2, 3)
        D = np.zeros((2, 1))

        result: ObservationLinearization = (C, D)

        C_unpacked, D_unpacked = result

        assert np.array_equal(C_unpacked, C)
        assert np.array_equal(D_unpacked, D)

    def test_observation_linearization_full_state(self):
        """Test full state observation (common case)."""
        nx = 3
        nu = 2

        # y = x (full state observation)
        C: OutputMatrix = np.eye(nx)
        D: FeedthroughMatrix = np.zeros((nx, nu))

        result: ObservationLinearization = (C, D)

        assert result[0].shape == (nx, nx)
        assert result[1].shape == (nx, nu)
        assert np.array_equal(result[0], np.eye(nx))

    def test_observation_linearization_partial_state(self):
        """Test partial state observation."""
        # Measure only position, not velocity
        C: OutputMatrix = np.array([[1, 0]])  # (ny=1, nx=2)
        D: FeedthroughMatrix = np.zeros((1, 1))  # (ny=1, nu=1)

        result: ObservationLinearization = (C, D)

        assert result[0].shape == (1, 2)
        assert result[1].shape == (1, 1)


# ============================================================================
# Test Time-Domain Specific Aliases
# ============================================================================


class TestTimeDomainAliases:
    """Test continuous/discrete specific type aliases."""

    def test_continuous_linearization_alias(self):
        """Test ContinuousLinearization is alias for deterministic."""
        Ac = np.array([[-1, 0], [0, -2]])
        Bc = np.array([[1], [0]])

        # Should accept same structure
        result_det: DeterministicLinearization = (Ac, Bc)
        result_cont: ContinuousLinearization = (Ac, Bc)

        assert result_det == result_cont

    def test_discrete_linearization_alias(self):
        """Test DiscreteLinearization is alias for deterministic."""
        Ad = np.array([[0.9, 0], [0, 0.8]])
        Bd = np.array([[0.1], [0.2]])

        result_det: DeterministicLinearization = (Ad, Bd)
        result_disc: DiscreteLinearization = (Ad, Bd)

        assert result_det == result_disc

    def test_continuous_stochastic_linearization_alias(self):
        """Test ContinuousStochasticLinearization alias."""
        Ac = np.array([[-1, 0], [0, -2]])
        Bc = np.array([[1], [0]])
        Gc = np.array([[0.1], [0.1]])

        result_stoch: StochasticLinearization = (Ac, Bc, Gc)
        result_cont_stoch: ContinuousStochasticLinearization = (Ac, Bc, Gc)

        assert result_stoch == result_cont_stoch

    def test_discrete_stochastic_linearization_alias(self):
        """Test DiscreteStochasticLinearization alias."""
        Ad = np.array([[0.9, 0], [0, 0.8]])
        Bd = np.array([[0.1], [0.2]])
        Gd = np.array([[0.05], [0.05]])

        result_stoch: StochasticLinearization = (Ad, Bd, Gd)
        result_disc_stoch: DiscreteStochasticLinearization = (Ad, Bd, Gd)

        assert result_stoch == result_disc_stoch


# ============================================================================
# Test Full Linearization Types
# ============================================================================


class TestFullLinearization:
    """Test full linearization including output."""

    def test_full_linearization_structure(self):
        """Test FullLinearization is 4-tuple (A, B, C, D)."""
        A = np.random.randn(2, 2)
        B = np.random.randn(2, 1)
        C = np.eye(2)
        D = np.zeros((2, 1))

        result: FullLinearization = (A, B, C, D)

        assert len(result) == 4

    def test_full_linearization_unpacking(self):
        """Test unpacking full linearization."""
        A = np.eye(3)
        B = np.ones((3, 2))
        C = np.eye(3)
        D = np.zeros((3, 2))

        result: FullLinearization = (A, B, C, D)

        A_out, B_out, C_out, D_out = result

        assert np.array_equal(A_out, A)
        assert np.array_equal(B_out, B)
        assert np.array_equal(C_out, C)
        assert np.array_equal(D_out, D)

    def test_full_linearization_state_space(self):
        """Test full linearization for state space model."""
        # State space: dx/dt = A*x + B*u, y = C*x + D*u
        A = np.array([[0, 1], [-2, -3]])
        B = np.array([[0], [1]])
        C = np.array([[1, 0]])  # Measure position only
        D = np.array([[0]])  # No feedthrough

        result: FullLinearization = (A, B, C, D)

        # Check dimensions
        nx = A.shape[0]
        nu = B.shape[1]
        ny = C.shape[0]

        assert result[0].shape == (nx, nx)
        assert result[1].shape == (nx, nu)
        assert result[2].shape == (ny, nx)
        assert result[3].shape == (ny, nu)

    def test_full_stochastic_linearization_structure(self):
        """Test FullStochasticLinearization is 5-tuple."""
        A = np.random.randn(2, 2)
        B = np.random.randn(2, 1)
        G = np.random.randn(2, 1)
        C = np.eye(2)
        D = np.zeros((2, 1))

        result: FullStochasticLinearization = (A, B, G, C, D)

        assert len(result) == 5

    def test_full_stochastic_linearization_unpacking(self):
        """Test unpacking full stochastic linearization."""
        A = np.eye(2)
        B = np.ones((2, 1))
        G = 0.1 * np.eye(2)
        C = np.eye(2)
        D = np.zeros((2, 1))

        result: FullStochasticLinearization = (A, B, G, C, D)

        A_out, B_out, G_out, C_out, D_out = result

        assert np.array_equal(A_out, A)
        assert np.array_equal(B_out, B)
        assert np.array_equal(G_out, G)
        assert np.array_equal(C_out, C)
        assert np.array_equal(D_out, D)


# ============================================================================
# Test Jacobian-Specific Types
# ============================================================================


class TestJacobianTypes:
    """Test Jacobian-specific type aliases."""

    def test_state_jacobian_is_state_matrix(self):
        """Test StateJacobian is alias for StateMatrix."""
        # ∂f/∂x
        A: StateJacobian = np.array([[0, 1], [-1, -0.5]])

        assert A.shape == (2, 2)

        # Should be accepted as StateMatrix
        A_as_state: StateMatrix = A
        assert np.array_equal(A, A_as_state)

    def test_control_jacobian_is_control_matrix(self):
        """Test ControlJacobian is alias for ControlMatrix."""
        # ∂f/∂u
        B: ControlJacobian = np.array([[0], [1]])

        assert B.shape == (2, 1)

        # Should be accepted as ControlMatrix
        B_as_control: ControlMatrix = B
        assert np.array_equal(B, B_as_control)

    def test_output_jacobian_is_output_matrix(self):
        """Test OutputJacobian is alias for OutputMatrix."""
        # ∂h/∂x
        C: OutputJacobian = np.array([[1, 0]])

        assert C.shape == (1, 2)

        # Should be accepted as OutputMatrix
        C_as_output: OutputMatrix = C
        assert np.array_equal(C, C_as_output)

    def test_diffusion_jacobian_is_diffusion_matrix(self):
        """Test DiffusionJacobian is alias for DiffusionMatrix."""
        # ∂g/∂x or g(x) for additive noise
        G: DiffusionJacobian = np.array([[0.1], [0.2]])

        assert G.shape == (2, 1)

        # Should be accepted as DiffusionMatrix
        G_as_diffusion: DiffusionMatrix = G
        assert np.array_equal(G, G_as_diffusion)

    def test_jacobian_usage_in_linearization(self):
        """Test using Jacobian types in linearization."""
        # Compute Jacobians
        A_jac: StateJacobian = np.array([[0, 1], [-2, -1]])
        B_jac: ControlJacobian = np.array([[0], [1]])

        # Use in linearization result
        result: DeterministicLinearization = (A_jac, B_jac)

        assert len(result) == 2


# ============================================================================
# Test Cache Key Type
# ============================================================================


class TestCacheKeyType:
    """Test linearization cache key type."""

    def test_cache_key_is_string(self):
        """Test LinearizationCacheKey is string."""
        key: LinearizationCacheKey = "x_eq=[0,0]_u_eq=[0]_method=euler"

        assert isinstance(key, str)

    def test_cache_key_construction(self):
        """Test constructing cache keys."""
        x_eq = np.array([0.0, 0.0])
        u_eq = np.array([0.0])
        method = "exact"

        # Construct key (simplified hash)
        key: LinearizationCacheKey = (
            f"x_eq={hash(x_eq.tobytes())}_u_eq={hash(u_eq.tobytes())}_method={method}"
        )

        assert isinstance(key, str)
        assert "method=exact" in key

    def test_cache_key_in_dict(self):
        """Test using cache key in dictionary."""
        cache: dict[LinearizationCacheKey, DeterministicLinearization] = {}

        # Store linearizations
        key1: LinearizationCacheKey = "x_eq=abc123_u_eq=def456"
        result1 = (np.eye(2), np.ones((2, 1)))
        cache[key1] = result1

        key2: LinearizationCacheKey = "x_eq=xyz789_u_eq=uvw012"
        result2 = (np.zeros((2, 2)), np.zeros((2, 1)))
        cache[key2] = result2

        assert len(cache) == 2
        assert np.array_equal(cache[key1][0], np.eye(2))


# ============================================================================
# Test Realistic Usage Patterns
# ============================================================================


class TestRealisticUsage:
    """Test types in realistic scenarios."""

    def test_lqr_controller_design_workflow(self):
        """Test linearization for LQR design."""
        # 1. Linearize system
        Ac = np.array([[0, 1], [-2, -1]])
        Bc = np.array([[0], [1]])
        result: ContinuousLinearization = (Ac, Bc)

        # 2. Design LQR
        Q = np.eye(2)
        R = np.array([[1.0]])

        # Solve CARE (simplified - would use scipy)
        # P = solve_continuous_are(Ac, Bc, Q, R)
        # K = np.linalg.inv(R) @ Bc.T @ P

        assert result[0].shape == (2, 2)
        assert result[1].shape == (2, 1)

    def test_kalman_filter_design_workflow(self):
        """Test linearization for Kalman filter."""
        # 1. Dynamics linearization
        Ad = np.array([[0.9, 0.1], [-0.1, 0.9]])
        Bd = np.array([[0.01], [0.02]])
        Gd = np.array([[0.05], [0.05]])

        dynamics: DiscreteStochasticLinearization = (Ad, Bd, Gd)

        # 2. Observation linearization
        C = np.eye(2)
        D = np.zeros((2, 1))

        observation: ObservationLinearization = (C, D)

        # 3. Design Kalman filter
        Q_noise = dynamics[2] @ dynamics[2].T
        R_noise = 0.1 * np.eye(2)

        assert Q_noise.shape == (2, 2)

    def test_lqg_controller_design_workflow(self):
        """Test full linearization for LQG design."""
        # Full linearization needed for LQG
        A = np.array([[0, 1], [-2, -1]])
        B = np.array([[0], [1]])
        G = np.array([[0.1], [0.2]])
        C = np.eye(2)
        D = np.zeros((2, 1))

        full: FullStochasticLinearization = (A, B, G, C, D)

        # Extract for separate LQR and Kalman design
        A_control, B_control, G_noise, C_obs, _ = full

        # LQR design uses (A, B)
        lqr_linearization: DeterministicLinearization = (A_control, B_control)

        # Kalman design uses (A, G, C)
        kalman_dynamics: StochasticLinearization = (A_control, B_control, G_noise)

        assert len(lqr_linearization) == 2
        assert len(kalman_dynamics) == 3

    def test_adaptive_linearization_workflow(self):
        """Test storing multiple linearizations at different points."""
        # Dictionary of linearizations at different equilibria
        linearizations: dict[str, LinearizationResult] = {}

        # Origin
        A_origin = np.array([[0, 1], [-1, 0]])
        B_origin = np.array([[0], [1]])
        linearizations["origin"] = (A_origin, B_origin)

        # Another point (stochastic)
        A_point = np.array([[-1, 0], [0, -2]])
        B_point = np.array([[1], [0]])
        G_point = np.array([[0.1], [0.1]])
        linearizations["point_1"] = (A_point, B_point, G_point)

        # Retrieve and check
        origin_lin = linearizations["origin"]
        assert len(origin_lin) == 2

        point_lin = linearizations["point_1"]
        assert len(point_lin) == 3


# ============================================================================
# Test Documentation Examples
# ============================================================================


class TestDocumentationExamples:
    """Test examples from docstrings work correctly."""

    def test_deterministic_linearization_example(self):
        """Test DeterministicLinearization docstring example."""
        # Continuous system linearization
        Ac, Bc = (np.array([[0, 1], [-1, 0]]), np.array([[0], [1]]))

        assert Ac.shape == (2, 2)
        assert Bc.shape == (2, 1)

        # Use for LQR design (example from docstring)
        # from scipy.linalg import solve_continuous_are
        # P = solve_continuous_are(Ac, Bc, Q, R)
        # K = np.linalg.inv(R) @ Bc.T @ P

    def test_stochastic_linearization_example(self):
        """Test StochasticLinearization docstring example."""
        # Continuous SDE linearization
        Ac = np.array([[-1, 0], [0, -2]])
        Bc = np.array([[1], [0]])
        Gc = np.array([[0.1], [0.1]])

        result: StochasticLinearization = (Ac, Bc, Gc)

        # Process noise covariance
        Q = result[2] @ result[2].T

        assert Q.shape == (2, 2)

    def test_polymorphic_linearization_example(self):
        """Test LinearizationResult polymorphic example."""

        def analyze_linearization(result: LinearizationResult) -> dict:
            """Polymorphic function from docstring."""
            A = result[0]
            B = result[1]

            info = {"eigenvalues": np.linalg.eigvals(A), "is_stochastic": len(result) == 3}

            if len(result) == 3:
                G = result[2]
                info["process_noise_cov"] = G @ G.T

            return info

        # Works with deterministic
        det_result: LinearizationResult = (np.eye(2), np.zeros((2, 1)))
        info_det = analyze_linearization(det_result)
        assert not info_det["is_stochastic"]

        # Works with stochastic
        stoch_result: LinearizationResult = (np.eye(2), np.zeros((2, 1)), 0.1 * np.eye(2))
        info_stoch = analyze_linearization(stoch_result)
        assert info_stoch["is_stochastic"]

    def test_observation_linearization_example(self):
        """Test ObservationLinearization docstring example."""
        # Full state observation (common case)
        nx = 3
        nu = 1

        C_full = np.eye(nx)  # y = x
        D_full = np.zeros((nx, nu))  # No feedthrough

        obs: ObservationLinearization = (C_full, D_full)

        assert obs[0].shape == (3, 3)
        assert obs[1].shape == (3, 1)


# ============================================================================
# Test Type Consistency
# ============================================================================


class TestTypeConsistency:
    """Test type consistency across module."""

    def test_time_domain_aliases_are_consistent(self):
        """Test time-domain aliases refer to same types."""
        # Continuous and Discrete should be same as Deterministic
        A = np.eye(2)
        B = np.ones((2, 1))

        det: DeterministicLinearization = (A, B)
        cont: ContinuousLinearization = (A, B)
        disc: DiscreteLinearization = (A, B)

        # All should be valid (same type)
        assert det == cont == disc

    def test_stochastic_aliases_are_consistent(self):
        """Test stochastic aliases refer to same types."""
        A = np.eye(2)
        B = np.ones((2, 1))
        G = 0.1 * np.eye(2)

        stoch: StochasticLinearization = (A, B, G)
        cont_stoch: ContinuousStochasticLinearization = (A, B, G)
        disc_stoch: DiscreteStochasticLinearization = (A, B, G)

        assert stoch == cont_stoch == disc_stoch

    def test_jacobian_aliases_are_consistent(self):
        """Test Jacobian aliases refer to base matrix types."""
        A: StateJacobian = np.eye(2)
        A_matrix: StateMatrix = np.eye(2)

        assert np.array_equal(A, A_matrix)

        B: ControlJacobian = np.ones((2, 1))
        B_matrix: ControlMatrix = np.ones((2, 1))

        assert np.array_equal(B, B_matrix)


# ============================================================================
# Test Edge Cases
# ============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_scalar_system_linearization(self):
        """Test linearization of scalar (1D) system."""
        # dx/dt = a*x + b*u
        A: StateMatrix = np.array([[2.0]])
        B: ControlMatrix = np.array([[1.0]])

        result: DeterministicLinearization = (A, B)

        assert result[0].shape == (1, 1)
        assert result[1].shape == (1, 1)

    def test_autonomous_system_linearization(self):
        """Test linearization of autonomous system (nu=0)."""
        # dx/dt = f(x), no control
        A = np.array([[0, 1], [-1, -0.5]])
        B = np.zeros((2, 0))  # No control input

        result: DeterministicLinearization = (A, B)

        assert result[1].shape == (2, 0)

    def test_fully_actuated_system(self):
        """Test linearization where nu = nx (fully actuated)."""
        nx = 3
        A = np.random.randn(nx, nx)
        B = np.eye(nx)  # Fully actuated

        result: DeterministicLinearization = (A, B)

        assert result[1].shape == (nx, nx)

    def test_high_dimensional_system(self):
        """Test linearization of high-dimensional system."""
        nx = 100
        nu = 50

        A = np.random.randn(nx, nx)
        B = np.random.randn(nx, nu)

        result: DeterministicLinearization = (A, B)

        assert result[0].shape == (nx, nx)
        assert result[1].shape == (nx, nu)


# ============================================================================
# Test Matrix Properties
# ============================================================================


class TestMatrixProperties:
    """Test expected properties of linearization matrices."""

    def test_stability_from_eigenvalues(self):
        """Test checking stability from linearization."""
        # Stable continuous system
        Ac_stable = np.array([[-1, 0], [0, -2]])
        Bc = np.array([[1], [0]])

        result: ContinuousLinearization = (Ac_stable, Bc)

        eigenvalues = np.linalg.eigvals(result[0])
        is_stable = np.all(np.real(eigenvalues) < 0)

        assert is_stable

        # Unstable continuous system
        Ac_unstable = np.array([[1, 0], [0, 2]])
        result_unstable: ContinuousLinearization = (Ac_unstable, Bc)

        eigenvalues_unstable = np.linalg.eigvals(result_unstable[0])
        is_unstable = np.any(np.real(eigenvalues_unstable) > 0)

        assert is_unstable

    def test_controllability_from_linearization(self):
        """Test checking controllability from linearization."""
        # Controllable system
        A = np.array([[0, 1], [0, 0]])
        B = np.array([[0], [1]])

        result: DeterministicLinearization = (A, B)

        # Controllability matrix
        C_ctrl = np.hstack([result[1], result[0] @ result[1]])
        rank = np.linalg.matrix_rank(C_ctrl)

        is_controllable = rank == A.shape[0]
        assert is_controllable

    def test_process_noise_covariance(self):
        """Test computing process noise covariance."""
        A = np.eye(2)
        B = np.zeros((2, 1))
        G = np.array([[0.1], [0.2]])

        result: StochasticLinearization = (A, B, G)

        # Q = G * G^T
        Q = result[2] @ result[2].T

        # Should be symmetric PSD
        assert np.allclose(Q, Q.T)
        eigenvalues = np.linalg.eigvals(Q)
        assert np.all(eigenvalues >= -1e-10)  # Non-negative (within tolerance)


# ============================================================================
# Test Type Safety
# ============================================================================


class TestTypeSafety:
    """Test type safety and type checking."""

    def test_deterministic_requires_two_elements(self):
        """Test DeterministicLinearization requires exactly 2 elements."""
        A = np.eye(2)
        B = np.ones((2, 1))

        # Valid
        result: DeterministicLinearization = (A, B)
        assert len(result) == 2

        # Note: Type system doesn't enforce at runtime, but mypy would catch
        # incorrect_result: DeterministicLinearization = (A, B, np.eye(2))  # mypy error

    def test_stochastic_requires_three_elements(self):
        """Test StochasticLinearization requires exactly 3 elements."""
        A = np.eye(2)
        B = np.ones((2, 1))
        G = 0.1 * np.eye(2)

        # Valid
        result: StochasticLinearization = (A, B, G)
        assert len(result) == 3

    def test_observation_requires_two_elements(self):
        """Test ObservationLinearization requires exactly 2 elements."""
        C = np.eye(2)
        D = np.zeros((2, 1))

        # Valid
        result: ObservationLinearization = (C, D)
        assert len(result) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
