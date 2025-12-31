# Copyright (C) 2025 Gil Benezer
# License: AGPL-3.0

"""
Unit Tests for Noise Analysis Module - UPDATED FOR FIXED LOGIC
===============================================================

Comprehensive test suite for the CORRECTED NoiseCharacterizer.

Tests cover:
- All 5 noise types (ADDITIVE, SCALAR, DIAGONAL, MULTIPLICATIVE, GENERAL)
- Dependency detection (state, control, time)
- Diagonal structure detection
- Solver recommendations
- Validation methods
- Optimization hints
- Edge cases
- Integration workflows

Key Changes from Original:
- MULTIPLICATIVE is now properly reachable (bug fixed)
- Added tests specifically for MULTIPLICATIVE classification
- Updated expected classifications based on corrected logic

Classification Priority (CORRECTED):
1. ADDITIVE - constant
2. SCALAR - nw=1
3. DIAGONAL - diagonal matrix
4. MULTIPLICATIVE - state-dependent, non-diagonal, non-scalar ✓ (FIXED)
5. GENERAL - fallback
"""

import pytest
import sympy as sp

from src.systems.base.utils.stochastic.noise_analysis import (
    NoiseCharacteristics,
    NoiseCharacterizer,
    NoiseType,
    SDEType,
    analyze_noise_structure,
)


# ============================================================================
# Fixtures - Test Diffusion Expressions
# ============================================================================


@pytest.fixture
def additive_noise_2d():
    """2D system with additive (constant) noise."""
    x1, x2 = sp.symbols("x1 x2")
    u = sp.symbols("u")

    # Constant diffusion - doesn't depend on anything
    diffusion = sp.Matrix([[0.1], [0.2]])

    return {
        "diffusion": diffusion,
        "state_vars": [x1, x2],
        "control_vars": [u],
        "nx": 2,
        "nw": 1,
    }


@pytest.fixture
def scalar_multiplicative_noise():
    """1D system with state-dependent noise, nw=1 → SCALAR."""
    x = sp.symbols("x")
    u = sp.symbols("u")

    # State-dependent diffusion, single noise source
    diffusion = sp.Matrix([[0.2 * x]])

    return {
        "diffusion": diffusion,
        "state_vars": [x],
        "control_vars": [u],
        "nx": 1,
        "nw": 1,
    }


@pytest.fixture
def diagonal_noise_3d():
    """3D system with diagonal multiplicative noise → DIAGONAL."""
    x1, x2, x3 = sp.symbols("x1 x2 x3")
    u = sp.symbols("u")

    # Diagonal diffusion - each state has independent noise
    diffusion = sp.Matrix([[0.1 * x1, 0, 0], [0, 0.2 * x2, 0], [0, 0, 0.3 * x3]])

    return {
        "diffusion": diffusion,
        "state_vars": [x1, x2, x3],
        "control_vars": [u],
        "nx": 3,
        "nw": 3,
    }


@pytest.fixture
def true_multiplicative_noise():
    """Non-diagonal, non-scalar, state-dependent → MULTIPLICATIVE."""
    x1, x2 = sp.symbols("x1 x2")
    
    # Key: nw=2 (not scalar), non-diagonal, state-dependent
    diffusion = sp.Matrix([[0.1*x1, 0.05*x2], [0.0, 0.2*x2]])

    return {
        "diffusion": diffusion,
        "state_vars": [x1, x2],
        "control_vars": [],
        "nx": 2,
        "nw": 2,
    }


@pytest.fixture
def scalar_noise_2d():
    """2D system with scalar noise (single Wiener process)."""
    x1, x2 = sp.symbols("x1 x2")
    u = sp.symbols("u")

    # Single noise source affects both states
    diffusion = sp.Matrix([[0.1], [0.2]])

    return {
        "diffusion": diffusion,
        "state_vars": [x1, x2],
        "control_vars": [u],
        "nx": 2,
        "nw": 1,
    }


@pytest.fixture
def control_dependent_noise():
    """System where noise depends on control input."""
    x = sp.symbols("x")
    u = sp.symbols("u")

    # Control-modulated noise
    diffusion = sp.Matrix([[0.1 * (1 + u)]])

    return {
        "diffusion": diffusion,
        "state_vars": [x],
        "control_vars": [u],
        "nx": 1,
        "nw": 1,
    }


@pytest.fixture
def time_varying_noise():
    """System with time-varying noise."""
    x = sp.symbols("x")
    u = sp.symbols("u")
    t = sp.symbols("t")

    # Time-varying diffusion
    diffusion = sp.Matrix([[0.1 * sp.sin(t)]])

    return {
        "diffusion": diffusion,
        "state_vars": [x],
        "control_vars": [u],
        "time_var": t,
        "nx": 1,
        "nw": 1,
    }


@pytest.fixture
def general_noise_2d():
    """2D system with general (fully coupled) noise → GENERAL."""
    x1, x2 = sp.symbols("x1 x2")
    u = sp.symbols("u")

    # Full matrix - states couple through noise, all elements non-zero
    diffusion = sp.Matrix([[0.1 * x1, 0.05 * x1], [0.05 * x2, 0.2 * x2]])

    return {
        "diffusion": diffusion,
        "state_vars": [x1, x2],
        "control_vars": [u],
        "nx": 2,
        "nw": 2,
    }


@pytest.fixture
def rectangular_noise():
    """System with rectangular diffusion matrix (nx != nw)."""
    x1, x2, x3 = sp.symbols("x1 x2 x3")
    u = sp.symbols("u")

    # 3 states, 2 noise sources
    diffusion = sp.Matrix([[0.1, 0.0], [0.0, 0.2], [0.1, 0.1]])

    return {
        "diffusion": diffusion,
        "state_vars": [x1, x2, x3],
        "control_vars": [u],
        "nx": 3,
        "nw": 2,
    }


@pytest.fixture
def mixed_dependency_noise():
    """System with noise depending on state, control, and time."""
    x = sp.symbols("x")
    u = sp.symbols("u")
    t = sp.symbols("t")

    # Depends on everything
    diffusion = sp.Matrix([[0.1 * x * (1 + u) * sp.cos(t)]])

    return {
        "diffusion": diffusion,
        "state_vars": [x],
        "control_vars": [u],
        "time_var": t,
        "nx": 1,
        "nw": 1,
    }


# ============================================================================
# Test NoiseCharacterizer Initialization
# ============================================================================


class TestNoiseCharacterizerInit:
    """Test NoiseCharacterizer initialization."""

    def test_basic_initialization(self, additive_noise_2d):
        """Test basic characterizer creation."""
        char = NoiseCharacterizer(
            additive_noise_2d["diffusion"],
            additive_noise_2d["state_vars"],
            additive_noise_2d["control_vars"],
        )

        assert char.nx == 2
        assert char.nw == 1
        assert len(char.state_vars) == 2
        assert len(char.control_vars) == 1
        assert char.time_var is None

    def test_initialization_with_time(self, time_varying_noise):
        """Test initialization with time variable."""
        char = NoiseCharacterizer(
            time_varying_noise["diffusion"],
            time_varying_noise["state_vars"],
            time_varying_noise["control_vars"],
            time_var=time_varying_noise["time_var"],
        )

        assert char.time_var is not None
        assert char.time_var == time_varying_noise["time_var"]

    def test_dimension_extraction(self, diagonal_noise_3d):
        """Test automatic dimension extraction."""
        char = NoiseCharacterizer(
            diagonal_noise_3d["diffusion"],
            diagonal_noise_3d["state_vars"],
            diagonal_noise_3d["control_vars"],
        )

        assert char.nx == 3
        assert char.nw == 3

    def test_rectangular_matrix(self, rectangular_noise):
        """Test with rectangular diffusion matrix."""
        char = NoiseCharacterizer(
            rectangular_noise["diffusion"],
            rectangular_noise["state_vars"],
            rectangular_noise["control_vars"],
        )

        assert char.nx == 3
        assert char.nw == 2


# ============================================================================
# Test Additive Noise Detection
# ============================================================================


class TestAdditiveNoiseDetection:
    """Test detection of additive (constant) noise."""

    def test_detect_additive_noise(self, additive_noise_2d):
        """Test detection of additive noise."""
        char = NoiseCharacterizer(
            additive_noise_2d["diffusion"],
            additive_noise_2d["state_vars"],
            additive_noise_2d["control_vars"],
        )

        result = char.analyze()

        assert result.is_additive
        assert result.noise_type == NoiseType.ADDITIVE
        assert not result.depends_on_state
        assert not result.depends_on_control
        assert not result.depends_on_time

    def test_additive_has_no_dependencies(self, additive_noise_2d):
        """Test that additive noise has no dependencies."""
        char = NoiseCharacterizer(
            additive_noise_2d["diffusion"],
            additive_noise_2d["state_vars"],
            additive_noise_2d["control_vars"],
        )

        result = char.analyze()

        assert len(result.state_dependencies) == 0
        assert len(result.control_dependencies) == 0

    def test_additive_scalar_noise(self, scalar_noise_2d):
        """Test additive noise with scalar noise source."""
        char = NoiseCharacterizer(
            scalar_noise_2d["diffusion"],
            scalar_noise_2d["state_vars"],
            scalar_noise_2d["control_vars"],
        )

        result = char.analyze()

        assert result.is_additive
        assert result.is_scalar
        assert result.noise_type == NoiseType.ADDITIVE  # Additive takes priority


# ============================================================================
# Test Multiplicative Noise Detection
# ============================================================================


class TestMultiplicativeNoiseDetection:
    """Test detection of multiplicative (state-dependent) noise."""

    def test_detect_multiplicative_nature(self, scalar_multiplicative_noise):
        """Test that multiplicative NATURE is detected (even if classified as SCALAR)."""
        char = NoiseCharacterizer(
            scalar_multiplicative_noise["diffusion"],
            scalar_multiplicative_noise["state_vars"],
            scalar_multiplicative_noise["control_vars"],
        )

        result = char.analyze()

        # is_multiplicative flag should be True
        assert result.is_multiplicative
        assert result.depends_on_state
        assert not result.is_additive
        
        # But classified as SCALAR because nw=1
        assert result.noise_type == NoiseType.SCALAR

    def test_multiplicative_state_dependencies(self, scalar_multiplicative_noise):
        """Test tracking of state dependencies."""
        char = NoiseCharacterizer(
            scalar_multiplicative_noise["diffusion"],
            scalar_multiplicative_noise["state_vars"],
            scalar_multiplicative_noise["control_vars"],
        )

        result = char.analyze()

        assert len(result.state_dependencies) == 1
        assert scalar_multiplicative_noise["state_vars"][0] in result.state_dependencies

    def test_true_multiplicative_classification(self, true_multiplicative_noise):
        """Test TRUE MULTIPLICATIVE classification (FIXED!)."""
        char = NoiseCharacterizer(
            true_multiplicative_noise["diffusion"],
            true_multiplicative_noise["state_vars"],
            true_multiplicative_noise["control_vars"],
        )

        result = char.analyze()

        # With the FIX, non-diagonal + non-scalar + state-dependent → MULTIPLICATIVE
        assert result.noise_type == NoiseType.MULTIPLICATIVE
        assert result.is_multiplicative
        assert not result.is_scalar
        assert not result.is_diagonal
        assert result.depends_on_state

    def test_multiplicative_general_matrix(self, general_noise_2d):
        """Test multiplicative noise with general (coupled) matrix."""
        char = NoiseCharacterizer(
            general_noise_2d["diffusion"],
            general_noise_2d["state_vars"],
            general_noise_2d["control_vars"],
        )

        result = char.analyze()

        assert result.is_multiplicative
        assert result.depends_on_state
        assert len(result.state_dependencies) == 2  # Both x1 and x2


# ============================================================================
# Test Diagonal Noise Detection
# ============================================================================


class TestDiagonalNoiseDetection:
    """Test detection of diagonal noise structure."""

    def test_detect_diagonal_noise(self, diagonal_noise_3d):
        """Test detection of diagonal noise matrix."""
        char = NoiseCharacterizer(
            diagonal_noise_3d["diffusion"],
            diagonal_noise_3d["state_vars"],
            diagonal_noise_3d["control_vars"],
        )

        result = char.analyze()

        assert result.is_diagonal
        assert result.noise_type == NoiseType.DIAGONAL

    def test_non_diagonal_fails(self, general_noise_2d):
        """Test that coupled noise is not classified as diagonal."""
        char = NoiseCharacterizer(
            general_noise_2d["diffusion"],
            general_noise_2d["state_vars"],
            general_noise_2d["control_vars"],
        )

        result = char.analyze()

        assert not result.is_diagonal

    def test_rectangular_not_diagonal(self, rectangular_noise):
        """Test that rectangular matrices cannot be diagonal."""
        char = NoiseCharacterizer(
            rectangular_noise["diffusion"],
            rectangular_noise["state_vars"],
            rectangular_noise["control_vars"],
        )

        result = char.analyze()

        assert not result.is_diagonal  # nx != nw

    def test_diagonal_with_zero_elements(self):
        """Test diagonal detection with some zero diagonal elements."""
        x1, x2, x3 = sp.symbols("x1 x2 x3")
        u = sp.symbols("u")

        # Diagonal with zero on diagonal
        diffusion = sp.Matrix([[0.1 * x1, 0, 0], [0, 0, 0], [0, 0, 0.3 * x3]])

        char = NoiseCharacterizer(diffusion, [x1, x2, x3], [u])
        result = char.analyze()

        assert result.is_diagonal  # Still diagonal structure

    def test_diagonal_constant(self):
        """Test constant diagonal matrix."""
        x1, x2 = sp.symbols("x1 x2")
        diffusion = sp.Matrix([[0.1, 0], [0, 0.2]])
        
        char = NoiseCharacterizer(diffusion, [x1, x2], [])
        result = char.analyze()
        
        assert result.is_diagonal
        assert result.is_additive
        # ADDITIVE takes priority
        assert result.noise_type == NoiseType.ADDITIVE


# ============================================================================
# Test Scalar Noise Detection
# ============================================================================


class TestScalarNoiseDetection:
    """Test detection of scalar noise (single Wiener process)."""

    def test_detect_scalar_noise(self, scalar_noise_2d):
        """Test detection of scalar noise."""
        char = NoiseCharacterizer(
            scalar_noise_2d["diffusion"],
            scalar_noise_2d["state_vars"],
            scalar_noise_2d["control_vars"],
        )

        result = char.analyze()

        assert result.is_scalar
        assert result.num_wiener == 1

    def test_multiplicative_scalar(self, scalar_multiplicative_noise):
        """Test scalar multiplicative noise."""
        char = NoiseCharacterizer(
            scalar_multiplicative_noise["diffusion"],
            scalar_multiplicative_noise["state_vars"],
            scalar_multiplicative_noise["control_vars"],
        )

        result = char.analyze()

        assert result.is_scalar
        assert result.is_multiplicative
        assert result.noise_type == NoiseType.SCALAR  # Scalar takes priority

    def test_non_scalar_multiple_wiener(self, diagonal_noise_3d):
        """Test that multiple Wiener processes are not scalar."""
        char = NoiseCharacterizer(
            diagonal_noise_3d["diffusion"],
            diagonal_noise_3d["state_vars"],
            diagonal_noise_3d["control_vars"],
        )

        result = char.analyze()

        assert not result.is_scalar
        assert result.num_wiener == 3


# ============================================================================
# Test Control Dependency Detection
# ============================================================================


class TestControlDependencyDetection:
    """Test detection of control-dependent noise."""

    def test_detect_control_dependency(self, control_dependent_noise):
        """Test detection of control-dependent noise."""
        char = NoiseCharacterizer(
            control_dependent_noise["diffusion"],
            control_dependent_noise["state_vars"],
            control_dependent_noise["control_vars"],
        )

        result = char.analyze()

        assert result.depends_on_control
        assert not result.is_additive
        assert len(result.control_dependencies) == 1

    def test_control_dependency_tracking(self, control_dependent_noise):
        """Test tracking of specific control dependencies."""
        char = NoiseCharacterizer(
            control_dependent_noise["diffusion"],
            control_dependent_noise["state_vars"],
            control_dependent_noise["control_vars"],
        )

        result = char.analyze()

        assert control_dependent_noise["control_vars"][0] in result.control_dependencies

    def test_no_control_dependency_additive(self, additive_noise_2d):
        """Test that additive noise has no control dependency."""
        char = NoiseCharacterizer(
            additive_noise_2d["diffusion"],
            additive_noise_2d["state_vars"],
            additive_noise_2d["control_vars"],
        )

        result = char.analyze()

        assert not result.depends_on_control
        assert len(result.control_dependencies) == 0


# ============================================================================
# Test Time Dependency Detection
# ============================================================================


class TestTimeDependencyDetection:
    """Test detection of time-varying noise."""

    def test_detect_time_dependency(self, time_varying_noise):
        """Test detection of time-varying noise."""
        char = NoiseCharacterizer(
            time_varying_noise["diffusion"],
            time_varying_noise["state_vars"],
            time_varying_noise["control_vars"],
            time_var=time_varying_noise["time_var"],
        )

        result = char.analyze()

        assert result.depends_on_time
        assert not result.is_additive

    def test_no_time_dependency_without_time_var(self, additive_noise_2d):
        """Test that systems without time_var don't have time dependency."""
        char = NoiseCharacterizer(
            additive_noise_2d["diffusion"],
            additive_noise_2d["state_vars"],
            additive_noise_2d["control_vars"],
            time_var=None,
        )

        result = char.analyze()

        assert not result.depends_on_time


# ============================================================================
# Test Mixed Dependencies
# ============================================================================


class TestMixedDependencies:
    """Test noise with multiple dependencies."""

    def test_mixed_dependencies(self, mixed_dependency_noise):
        """Test noise depending on state, control, and time."""
        char = NoiseCharacterizer(
            mixed_dependency_noise["diffusion"],
            mixed_dependency_noise["state_vars"],
            mixed_dependency_noise["control_vars"],
            time_var=mixed_dependency_noise["time_var"],
        )

        result = char.analyze()

        assert result.depends_on_state
        assert result.depends_on_control
        assert result.depends_on_time
        assert not result.is_additive
        assert result.is_multiplicative

    def test_all_dependencies_tracked(self, mixed_dependency_noise):
        """Test that all dependencies are tracked."""
        char = NoiseCharacterizer(
            mixed_dependency_noise["diffusion"],
            mixed_dependency_noise["state_vars"],
            mixed_dependency_noise["control_vars"],
            time_var=mixed_dependency_noise["time_var"],
        )

        result = char.analyze()

        assert len(result.state_dependencies) == 1
        assert len(result.control_dependencies) == 1


# ============================================================================
# Test Noise Type Classification (CORRECTED)
# ============================================================================


class TestNoiseTypeClassificationCORRECTED:
    """Test FIXED classification logic and priority."""

    def test_additive_priority(self, additive_noise_2d):
        """ADDITIVE has highest priority."""
        char = NoiseCharacterizer.from_dict(additive_noise_2d)
        result = char.analyze()
        
        assert result.noise_type == NoiseType.ADDITIVE

    def test_scalar_priority_over_multiplicative(self, scalar_multiplicative_noise):
        """SCALAR priority when nw=1."""
        char = NoiseCharacterizer.from_dict(scalar_multiplicative_noise)
        result = char.analyze()
        
        # Classified as SCALAR even though multiplicative
        assert result.noise_type == NoiseType.SCALAR
        assert result.is_multiplicative  # Still true

    def test_diagonal_priority(self, diagonal_noise_3d):
        """DIAGONAL priority for diagonal matrices."""
        char = NoiseCharacterizer.from_dict(diagonal_noise_3d)
        result = char.analyze()
        
        assert result.noise_type == NoiseType.DIAGONAL

    def test_multiplicative_NOW_REACHABLE(self, true_multiplicative_noise):
        """MULTIPLICATIVE now works (FIXED!)."""
        char = NoiseCharacterizer.from_dict(true_multiplicative_noise)
        result = char.analyze()
        
        # With the FIX, this returns MULTIPLICATIVE!
        assert result.noise_type == NoiseType.MULTIPLICATIVE
        assert result.is_multiplicative
        assert not result.is_scalar
        assert not result.is_diagonal

    def test_general_fallback(self, general_noise_2d):
        """Fully coupled now classified as MULTIPLICATIVE (after fix)."""
        char = NoiseCharacterizer.from_dict(general_noise_2d)
        result = char.analyze()
        
        # After fix: non-diagonal state-dependent → MULTIPLICATIVE
        assert result.noise_type == NoiseType.MULTIPLICATIVE
        assert result.is_multiplicative


# ============================================================================
# Test Validation Methods
# ============================================================================


class TestValidationMethods:
    """Test noise type validation."""

    def test_validate_additive_claim_correct(self, additive_noise_2d):
        """Correct additive claim validates."""
        char = NoiseCharacterizer.from_dict(additive_noise_2d)
        assert char.validate_noise_type_claim("additive")

    def test_validate_additive_claim_incorrect(self, scalar_multiplicative_noise):
        """Incorrect additive claim raises error."""
        char = NoiseCharacterizer.from_dict(scalar_multiplicative_noise)
        
        with pytest.raises(ValueError, match="Claimed noise_type='additive'"):
            char.validate_noise_type_claim("additive")

    def test_validate_diagonal_claim_correct(self, diagonal_noise_3d):
        """Correct diagonal claim validates."""
        char = NoiseCharacterizer.from_dict(diagonal_noise_3d)
        assert char.validate_noise_type_claim("diagonal")

    def test_validate_diagonal_claim_incorrect(self, general_noise_2d):
        """Incorrect diagonal claim raises error."""
        char = NoiseCharacterizer.from_dict(general_noise_2d)
        
        with pytest.raises(ValueError, match="Claimed noise_type='diagonal'"):
            char.validate_noise_type_claim("diagonal")

    def test_validate_scalar_claim_correct(self, scalar_noise_2d):
        """Correct scalar claim validates."""
        char = NoiseCharacterizer.from_dict(scalar_noise_2d)
        assert char.validate_noise_type_claim("scalar")

    def test_validate_scalar_claim_incorrect(self, diagonal_noise_3d):
        """Incorrect scalar claim raises error."""
        char = NoiseCharacterizer.from_dict(diagonal_noise_3d)
        
        with pytest.raises(ValueError, match="Claimed noise_type='scalar'"):
            char.validate_noise_type_claim("scalar")

    def test_validate_multiplicative_claim_correct(self, true_multiplicative_noise):
        """Correct multiplicative claim validates."""
        char = NoiseCharacterizer.from_dict(true_multiplicative_noise)
        assert char.validate_noise_type_claim("multiplicative")

    def test_validate_multiplicative_claim_incorrect(self, additive_noise_2d):
        """Incorrect multiplicative claim raises error."""
        char = NoiseCharacterizer.from_dict(additive_noise_2d)
        
        with pytest.raises(ValueError, match="Claimed noise_type='multiplicative'"):
            char.validate_noise_type_claim("multiplicative")


# ============================================================================
# Test Optimization Hints
# ============================================================================


class TestOptimizationHints:
    """Test optimization hint generation."""

    def test_optimization_hints_additive(self, additive_noise_2d):
        """Additive enables precomputation."""
        char = NoiseCharacterizer.from_dict(additive_noise_2d)
        result = char.analyze()  # Get characteristics first!
        hints = char.get_optimization_hints()

        assert hints["can_precompute_diffusion"]
        assert hints["complexity"] == "O(1) - constant, precomputable"

    def test_optimization_hints_diagonal(self, diagonal_noise_3d):
        """Diagonal enables element-wise operations."""
        char = NoiseCharacterizer.from_dict(diagonal_noise_3d)
        result = char.analyze()  # Get characteristics first!
        hints = char.get_optimization_hints()

        assert hints["can_use_diagonal_solver"]
        assert not hints["can_precompute_diffusion"]
        assert hints["complexity"] == "O(nx) - element-wise"

    def test_optimization_hints_scalar(self, scalar_multiplicative_noise):
        """Scalar enables simple multiplication."""
        char = NoiseCharacterizer.from_dict(scalar_multiplicative_noise)
        result = char.analyze()  # Get characteristics first!
        hints = char.get_optimization_hints()

        assert hints["can_use_scalar_solver"]
        assert hints["complexity"] == "O(nx) - scalar multiplication"

    def test_optimization_hints_multiplicative(self, true_multiplicative_noise):
        """Multiplicative requires re-evaluation."""
        char = NoiseCharacterizer.from_dict(true_multiplicative_noise)
        result = char.analyze()  # Get characteristics first!
        hints = char.get_optimization_hints()

        assert not hints["can_precompute_diffusion"]
        assert hints["requires_reevaluation"]
        assert hints["complexity"] == "O(nx * nw) - full matrix, state-dependent"

    def test_optimization_hints_general(self, general_noise_2d):
        """General has no special optimizations."""
        char = NoiseCharacterizer.from_dict(general_noise_2d)
        result = char.analyze()  # Get characteristics first!
        hints = char.get_optimization_hints()

        assert not hints["can_precompute_diffusion"]
        assert not hints["can_use_diagonal_solver"]

    def test_backend_recommendations_additive(self, additive_noise_2d):
        """JAX recommended first for additive."""
        char = NoiseCharacterizer.from_dict(additive_noise_2d)
        result = char.analyze()  # Get characteristics first!
        hints = char.get_optimization_hints()

        backends = hints["recommended_backends"]
        assert "jax" in backends
        assert backends[0] == "jax"

    def test_backend_recommendations_general(self, general_noise_2d):
        """All backends for general."""
        char = NoiseCharacterizer.from_dict(general_noise_2d)
        result = char.analyze()  # Get characteristics first!
        hints = char.get_optimization_hints()

        backends = hints["recommended_backends"]
        assert "jax" in backends
        assert "torch" in backends
        assert "numpy" in backends


# ============================================================================
# Test Solver Recommendations
# ============================================================================


class TestSolverRecommendations:
    """Test solver recommendations for different noise types."""

    def test_jax_solvers_additive(self, additive_noise_2d):
        """JAX additive solvers."""
        char = NoiseCharacterizer.from_dict(additive_noise_2d)
        result = char.analyze()
        
        solvers = result.recommended_solvers("jax")
        assert "sea" in solvers
        assert "shark" in solvers
        assert "sra1" in solvers

    def test_jax_solvers_scalar(self, scalar_multiplicative_noise):
        """JAX scalar solvers."""
        char = NoiseCharacterizer.from_dict(scalar_multiplicative_noise)
        result = char.analyze()
        
        solvers = result.recommended_solvers("jax")
        assert "euler_heun" in solvers
        assert "heun" in solvers

    def test_jax_solvers_diagonal(self, diagonal_noise_3d):
        """JAX diagonal solvers."""
        char = NoiseCharacterizer.from_dict(diagonal_noise_3d)
        result = char.analyze()
        
        solvers = result.recommended_solvers("jax")
        assert "euler_heun" in solvers
        assert "spark" in solvers

    def test_jax_solvers_multiplicative(self, true_multiplicative_noise):
        """JAX multiplicative solvers."""
        char = NoiseCharacterizer.from_dict(true_multiplicative_noise)
        result = char.analyze()
        
        solvers = result.recommended_solvers("jax")
        assert len(solvers) > 0
        assert "euler_heun" in solvers

    def test_torch_solvers_additive(self, additive_noise_2d):
        """PyTorch additive solvers."""
        char = NoiseCharacterizer.from_dict(additive_noise_2d)
        result = char.analyze()
        
        solvers = result.recommended_solvers("torch")
        assert any(s in ["euler", "milstein", "srk"] for s in solvers)

    def test_torch_solvers_general(self, general_noise_2d):
        """PyTorch general solvers."""
        char = NoiseCharacterizer.from_dict(general_noise_2d)
        result = char.analyze()
        
        solvers = result.recommended_solvers("torch")
        assert len(solvers) > 0

    def test_numpy_solvers_additive(self, additive_noise_2d):
        """NumPy (Julia) additive solvers."""
        char = NoiseCharacterizer.from_dict(additive_noise_2d)
        result = char.analyze()
        
        solvers = result.recommended_solvers("numpy")
        assert any(s.startswith("SRA") for s in solvers)

    def test_numpy_solvers_multiplicative(self, true_multiplicative_noise):
        """NumPy (Julia) multiplicative solvers."""
        char = NoiseCharacterizer.from_dict(true_multiplicative_noise)
        result = char.analyze()
        
        solvers = result.recommended_solvers("numpy")
        assert "SRIW1" in solvers or "SRI" in solvers


# ============================================================================
# Test Characteristics Property
# ============================================================================


class TestCharacteristicsProperty:
    """Test characteristics property and lazy loading."""

    def test_lazy_loading(self, additive_noise_2d):
        """Characteristics computed on first access."""
        char = NoiseCharacterizer.from_dict(additive_noise_2d)
        
        assert char._characteristics is None
        
        result = char.characteristics
        
        assert char._characteristics is not None
        assert result is char._characteristics

    def test_caching(self, additive_noise_2d):
        """Characteristics cached after first access."""
        char = NoiseCharacterizer.from_dict(additive_noise_2d)
        
        result1 = char.characteristics
        result2 = char.characteristics
        
        assert result1 is result2


# ============================================================================
# Test String Representations
# ============================================================================


class TestStringRepresentations:
    """Test string methods."""

    def test_repr_before_analysis(self, additive_noise_2d):
        """Test __repr__ before analysis."""
        char = NoiseCharacterizer.from_dict(additive_noise_2d)
        
        repr_str = repr(char)
        assert "not yet analyzed" in repr_str

    def test_repr_after_analysis(self, additive_noise_2d):
        """Test __repr__ after analysis."""
        char = NoiseCharacterizer.from_dict(additive_noise_2d)
        _ = char.characteristics
        
        repr_str = repr(char)
        assert "NoiseCharacterizer" in repr_str
        assert "additive" in repr_str


# ============================================================================
# Test Edge Cases
# ============================================================================


class TestEdgeCases:
    """Test edge cases."""

    def test_zero_diffusion(self):
        """Zero diffusion → ADDITIVE."""
        x, u = sp.symbols("x u")
        diffusion = sp.Matrix([[0]])
        
        char = NoiseCharacterizer(diffusion, [x], [u])
        result = char.analyze()
        
        assert result.is_additive
        assert result.noise_type == NoiseType.ADDITIVE

    def test_identity_diffusion(self):
        """Identity → ADDITIVE + DIAGONAL."""
        x1, x2 = sp.symbols("x1 x2")
        diffusion = sp.Matrix([[1, 0], [0, 1]])
        
        char = NoiseCharacterizer(diffusion, [x1, x2], [])
        result = char.analyze()
        
        assert result.is_additive
        assert result.is_diagonal
        assert result.noise_type == NoiseType.ADDITIVE

    def test_single_element_matrix(self):
        """1×1 matrix."""
        x, u = sp.symbols("x u")
        diffusion = sp.Matrix([[0.1]])
        
        char = NoiseCharacterizer(diffusion, [x], [u])
        result = char.analyze()
        
        assert result.is_scalar
        assert result.is_additive
        assert result.is_diagonal

    def test_complex_symbolic_expression(self):
        """Complex expression with trig."""
        x1, x2 = sp.symbols("x1 x2")
        diffusion = sp.Matrix([[sp.sin(x1) + sp.cos(x2)]])
        
        char = NoiseCharacterizer(diffusion, [x1, x2], [])
        result = char.analyze()
        
        assert result.is_multiplicative
        assert result.depends_on_state
        assert len(result.state_dependencies) == 2

    def test_large_matrix(self):
        """Large diffusion matrix."""
        n = 10
        state_vars = sp.symbols(f'x0:{n}')
        diffusion = sp.Matrix([[sp.Symbol(f's{i}') if i == j else 0 
                                for j in range(n)] for i in range(n)])
        
        char = NoiseCharacterizer(diffusion, list(state_vars), [])
        result = char.analyze()
        
        assert result.is_diagonal
        assert result.num_wiener == n


# ============================================================================
# Test Integration Workflows
# ============================================================================


class TestIntegrationWorkflows:
    """Integration tests with complete workflows."""

    def test_full_workflow_additive(self, additive_noise_2d):
        """Complete workflow for additive."""
        char = NoiseCharacterizer.from_dict(additive_noise_2d)
        
        result = char.analyze()
        assert result.noise_type == NoiseType.ADDITIVE
        
        solvers = result.recommended_solvers("jax")
        assert len(solvers) > 0
        
        hints = char.get_optimization_hints()
        assert hints["can_precompute_diffusion"]
        
        assert char.validate_noise_type_claim("additive")

    def test_full_workflow_multiplicative(self, true_multiplicative_noise):
        """Complete workflow for multiplicative."""
        char = NoiseCharacterizer.from_dict(true_multiplicative_noise)
        
        result = char.characteristics
        assert result.noise_type == NoiseType.MULTIPLICATIVE
        assert result.is_multiplicative
        assert result.depends_on_state
        
        hints = char.get_optimization_hints()
        assert not hints["can_precompute_diffusion"]
        assert hints["requires_reevaluation"]
        
        assert char.validate_noise_type_claim("multiplicative")

    def test_full_workflow_diagonal(self, diagonal_noise_3d):
        """Complete workflow for diagonal."""
        char = NoiseCharacterizer.from_dict(diagonal_noise_3d)
        
        result = char.characteristics
        assert result.noise_type == NoiseType.DIAGONAL
        assert result.is_diagonal
        assert result.is_multiplicative
        
        hints = char.get_optimization_hints()
        assert hints["can_use_diagonal_solver"]
        
        assert char.validate_noise_type_claim("diagonal")


# ============================================================================
# Test Convenience Function
# ============================================================================


class TestConvenienceFunction:
    """Test analyze_noise_structure convenience function."""

    def test_convenience_function(self, additive_noise_2d):
        """Test convenience function works."""
        result = analyze_noise_structure(
            additive_noise_2d["diffusion"],
            additive_noise_2d["state_vars"],
            additive_noise_2d["control_vars"],
        )
        
        assert isinstance(result, NoiseCharacteristics)
        assert result.noise_type == NoiseType.ADDITIVE


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])