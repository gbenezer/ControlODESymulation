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
Unit Tests for DiffusionHandler

Tests code generation, caching, optimization, and integration with NoiseCharacterizer.
Mirrors test structure from test_code_generator.py for consistency.
"""


import numpy as np
import pytest
import sympy as sp

from src.systems.base.utils.stochastic.diffusion_handler import (
    DiffusionHandler,
    create_diffusion_handler,
)
from src.systems.base.utils.stochastic.noise_analysis import NoiseType

# ============================================================================
# Fixtures - Test Systems
# ============================================================================


@pytest.fixture
def additive_noise_2d():
    """2D system with additive (constant) noise."""
    x1, x2 = sp.symbols("x1 x2")
    u = sp.symbols("u")

    # Constant diffusion matrix
    diffusion = sp.Matrix([[0.1], [0.2]])

    return {
        "diffusion": diffusion,
        "state_vars": [x1, x2],
        "control_vars": [u],
        "nx": 2,
        "nw": 1,
    }


@pytest.fixture
def multiplicative_noise_1d():
    """1D system with state-dependent (multiplicative) noise."""
    x = sp.symbols("x")
    u = sp.symbols("u")
    sigma = sp.Symbol("sigma")

    # State-dependent diffusion
    diffusion = sp.Matrix([[sigma * x]])

    return {
        "diffusion": diffusion,
        "state_vars": [x],
        "control_vars": [u],
        "parameters": {sigma: 0.2},
        "nx": 1,
        "nw": 1,
    }


@pytest.fixture
def diagonal_noise_3d():
    """3D system with diagonal multiplicative noise."""
    x1, x2, x3 = sp.symbols("x1 x2 x3")
    u = sp.symbols("u")

    # Diagonal diffusion (each state has independent noise)
    diffusion = sp.Matrix([[0.1 * x1, 0, 0], [0, 0.2 * x2, 0], [0, 0, 0.3 * x3]])

    return {
        "diffusion": diffusion,
        "state_vars": [x1, x2, x3],
        "control_vars": [u],
        "nx": 3,
        "nw": 3,
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
def scalar_noise_2d():
    """2D system with scalar noise affecting both states."""
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


# ============================================================================
# Test Initialization
# ============================================================================


class TestInitialization:
    """Test DiffusionHandler initialization and property extraction."""

    def test_basic_initialization(self, additive_noise_2d):
        """Test basic handler creation."""
        handler = DiffusionHandler(
            additive_noise_2d["diffusion"],
            additive_noise_2d["state_vars"],
            additive_noise_2d["control_vars"],
        )

        assert handler.nx == 2
        assert handler.nw == 1
        assert len(handler.state_vars) == 2
        assert len(handler.control_vars) == 1
        assert handler.time_var is None
        assert len(handler.parameters) == 0

    def test_initialization_with_parameters(self, multiplicative_noise_1d):
        """Test initialization with parameter substitution."""
        handler = DiffusionHandler(
            multiplicative_noise_1d["diffusion"],
            multiplicative_noise_1d["state_vars"],
            multiplicative_noise_1d["control_vars"],
            parameters=multiplicative_noise_1d["parameters"],
        )

        assert handler.nx == 1
        assert handler.nw == 1
        assert len(handler.parameters) == 1

    def test_initialization_with_time_var(self, time_varying_noise):
        """Test initialization with time variable."""
        handler = DiffusionHandler(
            time_varying_noise["diffusion"],
            time_varying_noise["state_vars"],
            time_varying_noise["control_vars"],
            time_var=time_varying_noise["time_var"],
        )

        assert handler.time_var is not None
        assert handler.characteristics.depends_on_time

    def test_dimension_extraction(self, diagonal_noise_3d):
        """Test automatic dimension extraction."""
        handler = DiffusionHandler(
            diagonal_noise_3d["diffusion"],
            diagonal_noise_3d["state_vars"],
            diagonal_noise_3d["control_vars"],
        )

        assert handler.nx == 3
        assert handler.nw == 3
        assert handler.diffusion_expr.shape == (3, 3)

    def test_noise_characterizer_composition(self, additive_noise_2d):
        """Test automatic NoiseCharacterizer composition."""
        handler = DiffusionHandler(
            additive_noise_2d["diffusion"],
            additive_noise_2d["state_vars"],
            additive_noise_2d["control_vars"],
        )

        # Should have characterizer
        assert hasattr(handler, "characterizer")
        assert handler.characterizer is not None

        # Should have characteristics accessible
        char = handler.characteristics
        assert char is not None
        assert char.noise_type == NoiseType.ADDITIVE


# ============================================================================
# Test Noise Characterization
# ============================================================================


class TestNoiseCharacterization:
    """Test integration with NoiseCharacterizer."""

    def test_additive_noise_detection(self, additive_noise_2d):
        """Test detection of additive noise."""
        handler = DiffusionHandler(
            additive_noise_2d["diffusion"],
            additive_noise_2d["state_vars"],
            additive_noise_2d["control_vars"],
        )

        char = handler.characteristics
        assert char.noise_type == NoiseType.ADDITIVE
        assert char.is_additive
        assert not char.is_multiplicative
        assert not char.depends_on_state
        assert not char.depends_on_control
        assert not char.depends_on_time

    def test_multiplicative_noise_detection(self, multiplicative_noise_1d):
        """Test detection of multiplicative noise."""
        handler = DiffusionHandler(
            multiplicative_noise_1d["diffusion"],
            multiplicative_noise_1d["state_vars"],
            multiplicative_noise_1d["control_vars"],
            parameters=multiplicative_noise_1d["parameters"],
        )

        char = handler.characteristics
        assert char.noise_type == NoiseType.SCALAR  # Changed from MULTIPLICATIVE
        assert not char.is_additive
        assert char.is_multiplicative  # Still multiplicative, just classified as scalar
        assert char.depends_on_state

    def test_diagonal_noise_detection(self, diagonal_noise_3d):
        """Test detection of diagonal noise structure."""
        handler = DiffusionHandler(
            diagonal_noise_3d["diffusion"],
            diagonal_noise_3d["state_vars"],
            diagonal_noise_3d["control_vars"],
        )

        char = handler.characteristics
        assert char.is_diagonal
        assert char.depends_on_state

    def test_scalar_noise_detection(self, scalar_noise_2d):
        """Test detection of scalar noise."""
        handler = DiffusionHandler(
            scalar_noise_2d["diffusion"],
            scalar_noise_2d["state_vars"],
            scalar_noise_2d["control_vars"],
        )

        char = handler.characteristics
        assert char.is_scalar
        assert char.num_wiener == 1
        assert handler.nw == 1

    def test_control_dependency_detection(self, control_dependent_noise):
        """Test detection of control-dependent noise."""
        handler = DiffusionHandler(
            control_dependent_noise["diffusion"],
            control_dependent_noise["state_vars"],
            control_dependent_noise["control_vars"],
        )

        char = handler.characteristics
        assert char.depends_on_control
        assert not char.is_additive

    def test_time_dependency_detection(self, time_varying_noise):
        """Test detection of time-varying noise."""
        handler = DiffusionHandler(
            time_varying_noise["diffusion"],
            time_varying_noise["state_vars"],
            time_varying_noise["control_vars"],
            time_var=time_varying_noise["time_var"],
        )

        char = handler.characteristics
        assert char.depends_on_time
        assert not char.is_additive

    def test_state_dependencies(self, multiplicative_noise_1d):
        """Test tracking of specific state dependencies."""
        handler = DiffusionHandler(
            multiplicative_noise_1d["diffusion"],
            multiplicative_noise_1d["state_vars"],
            multiplicative_noise_1d["control_vars"],
            parameters=multiplicative_noise_1d["parameters"],
        )

        char = handler.characteristics
        assert len(char.state_dependencies) == 1
        assert multiplicative_noise_1d["state_vars"][0] in char.state_dependencies


# ============================================================================
# Test Code Generation - NumPy
# ============================================================================


class TestCodeGenerationNumPy:
    """Test NumPy code generation."""

    def test_generate_additive_noise_numpy(self, additive_noise_2d):
        """Test NumPy generation for additive noise."""
        handler = DiffusionHandler(
            additive_noise_2d["diffusion"],
            additive_noise_2d["state_vars"],
            additive_noise_2d["control_vars"],
        )

        g_func = handler.generate_function("numpy")

        # Test evaluation at arbitrary points (should be constant)
        result1 = g_func(1.0, 2.0, 0.5)
        result2 = g_func(0.0, 0.0, 0.0)

        assert isinstance(result1, np.ndarray)
        assert result1.shape == (2, 1)
        np.testing.assert_array_almost_equal(result1, [[0.1], [0.2]])
        np.testing.assert_array_almost_equal(result1, result2)

    def test_generate_multiplicative_noise_numpy(self, multiplicative_noise_1d):
        """Test NumPy generation for multiplicative noise."""
        handler = DiffusionHandler(
            multiplicative_noise_1d["diffusion"],
            multiplicative_noise_1d["state_vars"],
            multiplicative_noise_1d["control_vars"],
            parameters=multiplicative_noise_1d["parameters"],
        )

        g_func = handler.generate_function("numpy")

        # Test at x=2.0, u=0.0 -> expect [[0.2 * 2.0]] = [[0.4]]
        result = g_func(2.0, 0.0)

        assert isinstance(result, np.ndarray)
        assert result.shape == (1, 1)
        np.testing.assert_array_almost_equal(result, [[0.4]])

    def test_generate_diagonal_noise_numpy(self, diagonal_noise_3d):
        """Test NumPy generation for diagonal noise."""
        handler = DiffusionHandler(
            diagonal_noise_3d["diffusion"],
            diagonal_noise_3d["state_vars"],
            diagonal_noise_3d["control_vars"],
        )

        g_func = handler.generate_function("numpy")

        # Test at x=[1, 2, 3], u=0
        result = g_func(1.0, 2.0, 3.0, 0.0)

        expected = np.array([[0.1, 0, 0], [0, 0.4, 0], [0, 0, 0.9]])

        assert result.shape == (3, 3)
        np.testing.assert_array_almost_equal(result, expected)

    def test_numpy_callable_verification(self, additive_noise_2d):
        """Test that generated NumPy function is callable."""
        handler = DiffusionHandler(
            additive_noise_2d["diffusion"],
            additive_noise_2d["state_vars"],
            additive_noise_2d["control_vars"],
        )

        g_func = handler.generate_function("numpy")
        assert callable(g_func)


# ============================================================================
# Test Code Generation - PyTorch
# ============================================================================


class TestCodeGenerationTorch:
    """Test PyTorch code generation."""

    def test_generate_additive_noise_torch(self, additive_noise_2d):
        """Test PyTorch generation for additive noise."""
        pytest.importorskip("torch")
        import torch

        handler = DiffusionHandler(
            additive_noise_2d["diffusion"],
            additive_noise_2d["state_vars"],
            additive_noise_2d["control_vars"],
        )

        g_func = handler.generate_function("torch")

        # Test with tensors
        x1 = torch.tensor(1.0)
        x2 = torch.tensor(2.0)
        u = torch.tensor(0.5)

        result = g_func(x1, x2, u)

        assert isinstance(result, torch.Tensor)
        assert result.shape == (2, 1)

        expected = torch.tensor([[0.1], [0.2]])
        torch.testing.assert_close(result, expected)

    def test_generate_multiplicative_noise_torch(self, multiplicative_noise_1d):
        """Test PyTorch generation for multiplicative noise."""
        pytest.importorskip("torch")
        import torch

        handler = DiffusionHandler(
            multiplicative_noise_1d["diffusion"],
            multiplicative_noise_1d["state_vars"],
            multiplicative_noise_1d["control_vars"],
            parameters=multiplicative_noise_1d["parameters"],
        )

        g_func = handler.generate_function("torch")

        x = torch.tensor(2.0)
        u = torch.tensor(0.0)
        result = g_func(x, u)

        assert isinstance(result, torch.Tensor)
        expected = torch.tensor([[0.4]])
        torch.testing.assert_close(result, expected)

    def test_torch_gradient_flow(self, multiplicative_noise_1d):
        """Test that gradients flow through PyTorch diffusion function."""
        pytest.importorskip("torch")
        import torch

        handler = DiffusionHandler(
            multiplicative_noise_1d["diffusion"],
            multiplicative_noise_1d["state_vars"],
            multiplicative_noise_1d["control_vars"],
            parameters=multiplicative_noise_1d["parameters"],
        )

        g_func = handler.generate_function("torch")

        x = torch.tensor(2.0, requires_grad=True)
        u = torch.tensor(0.0)

        result = g_func(x, u)
        loss = result.sum()
        loss.backward()

        # Gradient should be 0.2 (sigma)
        assert x.grad is not None
        assert torch.isclose(x.grad, torch.tensor(0.2))


# ============================================================================
# Test Code Generation - JAX
# ============================================================================


class TestCodeGenerationJAX:
    """Test JAX code generation."""

    def test_generate_additive_noise_jax(self, additive_noise_2d):
        """Test JAX generation for additive noise."""
        pytest.importorskip("jax")
        import jax.numpy as jnp

        handler = DiffusionHandler(
            additive_noise_2d["diffusion"],
            additive_noise_2d["state_vars"],
            additive_noise_2d["control_vars"],
        )

        g_func = handler.generate_function("jax")

        result = g_func(jnp.array(1.0), jnp.array(2.0), jnp.array(0.5))

        assert isinstance(result, jnp.ndarray)
        assert result.shape == (2, 1)

        expected = jnp.array([[0.1], [0.2]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_generate_multiplicative_noise_jax(self, multiplicative_noise_1d):
        """Test JAX generation for multiplicative noise."""
        pytest.importorskip("jax")
        import jax.numpy as jnp

        handler = DiffusionHandler(
            multiplicative_noise_1d["diffusion"],
            multiplicative_noise_1d["state_vars"],
            multiplicative_noise_1d["control_vars"],
            parameters=multiplicative_noise_1d["parameters"],
        )

        g_func = handler.generate_function("jax")

        result = g_func(jnp.array(2.0), jnp.array(0.0))

        expected = jnp.array([[0.4]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_jax_jit_compilation(self, additive_noise_2d):
        """Test JAX JIT compilation option."""
        pytest.importorskip("jax")
        import jax.numpy as jnp

        handler = DiffusionHandler(
            additive_noise_2d["diffusion"],
            additive_noise_2d["state_vars"],
            additive_noise_2d["control_vars"],
        )

        g_func = handler.generate_function("jax", jit=True)

        # Should work with JIT
        result = g_func(jnp.array(1.0), jnp.array(2.0), jnp.array(0.5))
        assert isinstance(result, jnp.ndarray)

    def test_jax_gradient_computation(self, multiplicative_noise_1d):
        """Test gradient computation through JAX diffusion function."""
        pytest.importorskip("jax")
        import jax
        import jax.numpy as jnp

        handler = DiffusionHandler(
            multiplicative_noise_1d["diffusion"],
            multiplicative_noise_1d["state_vars"],
            multiplicative_noise_1d["control_vars"],
            parameters=multiplicative_noise_1d["parameters"],
        )

        g_func = handler.generate_function("jax")

        # Create gradient function
        def loss_fn(x):
            result = g_func(x, jnp.array(0.0))
            return jnp.sum(result)

        grad_fn = jax.grad(loss_fn)
        grad = grad_fn(jnp.array(2.0))

        # Gradient should be 0.2 (sigma)
        assert jnp.isclose(grad, 0.2)


# ============================================================================
# Test Caching
# ============================================================================


class TestCaching:
    """Test function caching mechanisms."""

    def test_cache_on_second_call(self, additive_noise_2d):
        """Test that second generation call returns cached function."""
        handler = DiffusionHandler(
            additive_noise_2d["diffusion"],
            additive_noise_2d["state_vars"],
            additive_noise_2d["control_vars"],
        )

        # First call generates
        g1 = handler.generate_function("numpy")

        # Second call should return cached
        g2 = handler.generate_function("numpy")

        assert g1 is g2  # Same object reference

    def test_cache_per_backend(self, additive_noise_2d):
        """Test independent caching per backend."""
        handler = DiffusionHandler(
            additive_noise_2d["diffusion"],
            additive_noise_2d["state_vars"],
            additive_noise_2d["control_vars"],
        )

        g_numpy = handler.generate_function("numpy")
        g_torch = handler.generate_function("torch")
        g_jax = handler.generate_function("jax")

        # Different functions for different backends
        assert g_numpy is not g_torch
        assert g_numpy is not g_jax
        assert g_torch is not g_jax

        # But cached on repeat
        assert handler.generate_function("numpy") is g_numpy

    def test_get_function_cached(self, additive_noise_2d):
        """Test get_function returns cached without generating."""
        handler = DiffusionHandler(
            additive_noise_2d["diffusion"],
            additive_noise_2d["state_vars"],
            additive_noise_2d["control_vars"],
        )

        # Initially None
        assert handler.get_function("numpy") is None

        # Generate
        g_func = handler.generate_function("numpy")

        # Now cached
        assert handler.get_function("numpy") is g_func

    def test_reset_cache_single_backend(self, additive_noise_2d):
        """Test resetting cache for single backend."""
        handler = DiffusionHandler(
            additive_noise_2d["diffusion"],
            additive_noise_2d["state_vars"],
            additive_noise_2d["control_vars"],
        )

        g_numpy = handler.generate_function("numpy")
        g_torch = handler.generate_function("torch")

        # Reset only numpy
        handler.reset_cache(["numpy"])

        assert handler.get_function("numpy") is None
        assert handler.get_function("torch") is g_torch

    def test_reset_cache_all_backends(self, additive_noise_2d):
        """Test resetting cache for all backends."""
        handler = DiffusionHandler(
            additive_noise_2d["diffusion"],
            additive_noise_2d["state_vars"],
            additive_noise_2d["control_vars"],
        )

        handler.generate_function("numpy")
        handler.generate_function("torch")
        handler.generate_function("jax")

        handler.reset_cache()

        assert handler.get_function("numpy") is None
        assert handler.get_function("torch") is None
        assert handler.get_function("jax") is None

    def test_is_compiled(self, additive_noise_2d):
        """Test is_compiled status check."""
        handler = DiffusionHandler(
            additive_noise_2d["diffusion"],
            additive_noise_2d["state_vars"],
            additive_noise_2d["control_vars"],
        )

        assert not handler.is_compiled("numpy")

        handler.generate_function("numpy")
        assert handler.is_compiled("numpy")
        assert not handler.is_compiled("torch")


# ============================================================================
# Test Constant Noise Optimization (Additive)
# ============================================================================


class TestConstantNoiseOptimization:
    """Test optimization for additive (constant) noise."""

    def test_get_constant_noise_numpy(self, additive_noise_2d):
        """Test getting constant noise matrix for NumPy."""
        handler = DiffusionHandler(
            additive_noise_2d["diffusion"],
            additive_noise_2d["state_vars"],
            additive_noise_2d["control_vars"],
        )

        G = handler.get_constant_noise("numpy")

        assert isinstance(G, np.ndarray)
        assert G.shape == (2, 1)
        np.testing.assert_array_almost_equal(G, [[0.1], [0.2]])

    def test_constant_noise_is_cached(self, additive_noise_2d):
        """Test that constant noise is cached after first computation."""
        handler = DiffusionHandler(
            additive_noise_2d["diffusion"],
            additive_noise_2d["state_vars"],
            additive_noise_2d["control_vars"],
        )

        assert not handler.has_constant_noise()

        G1 = handler.get_constant_noise("numpy")
        assert handler.has_constant_noise()

        G2 = handler.get_constant_noise("numpy")
        assert G1 is G2  # Same object reference

    def test_constant_noise_multiple_backends(self, additive_noise_2d):
        """Test constant noise for multiple backends."""
        pytest.importorskip("torch")

        handler = DiffusionHandler(
            additive_noise_2d["diffusion"],
            additive_noise_2d["state_vars"],
            additive_noise_2d["control_vars"],
        )

        G_numpy = handler.get_constant_noise("numpy")
        G_torch = handler.get_constant_noise("torch")
        # Now that handlers preserve backend array type,
        # conversion needs to occur for this test
        G_torch_numpy = G_torch.detach().numpy()

        assert isinstance(G_numpy, np.ndarray)
        assert isinstance(G_torch_numpy, np.ndarray)

        np.testing.assert_array_almost_equal(G_numpy, G_torch_numpy)

    def test_constant_noise_error_for_multiplicative(self, multiplicative_noise_1d):
        """Test error when requesting constant noise for multiplicative noise."""
        handler = DiffusionHandler(
            multiplicative_noise_1d["diffusion"],
            multiplicative_noise_1d["state_vars"],
            multiplicative_noise_1d["control_vars"],
            parameters=multiplicative_noise_1d["parameters"],
        )

        with pytest.raises(ValueError, match="only valid for additive noise"):
            handler.get_constant_noise("numpy")

    def test_can_optimize_for_additive(self, additive_noise_2d, multiplicative_noise_1d):
        """Test detection of additive noise optimization opportunity."""
        handler_additive = DiffusionHandler(
            additive_noise_2d["diffusion"],
            additive_noise_2d["state_vars"],
            additive_noise_2d["control_vars"],
        )

        handler_multiplicative = DiffusionHandler(
            multiplicative_noise_1d["diffusion"],
            multiplicative_noise_1d["state_vars"],
            multiplicative_noise_1d["control_vars"],
            parameters=multiplicative_noise_1d["parameters"],
        )

        assert handler_additive.can_optimize_for_additive()
        assert not handler_multiplicative.can_optimize_for_additive()

    def test_constant_noise_reset_with_cache(self, additive_noise_2d):
        """Test that reset_cache also clears constant noise."""
        handler = DiffusionHandler(
            additive_noise_2d["diffusion"],
            additive_noise_2d["state_vars"],
            additive_noise_2d["control_vars"],
        )

        handler.get_constant_noise("numpy")
        assert handler.has_constant_noise()

        handler.reset_cache(["numpy"])
        assert not handler.has_constant_noise()


# ============================================================================
# Test Optimization Opportunities
# ============================================================================


class TestOptimizationOpportunities:
    """Test identification of optimization opportunities."""

    def test_additive_optimization_flags(self, additive_noise_2d):
        """Test optimization flags for additive noise."""
        handler = DiffusionHandler(
            additive_noise_2d["diffusion"],
            additive_noise_2d["state_vars"],
            additive_noise_2d["control_vars"],
        )

        opts = handler.get_optimization_opportunities()

        assert opts["precompute_diffusion"]
        assert opts["vectorize_easily"]
        assert opts["cache_diffusion"]

    def test_diagonal_optimization_flags(self, diagonal_noise_3d):
        """Test optimization flags for diagonal noise."""
        handler = DiffusionHandler(
            diagonal_noise_3d["diffusion"],
            diagonal_noise_3d["state_vars"],
            diagonal_noise_3d["control_vars"],
        )

        opts = handler.get_optimization_opportunities()

        assert opts["use_diagonal_solver"]
        assert opts["vectorize_easily"]
        assert not opts["precompute_diffusion"]  # State-dependent

    def test_scalar_optimization_flags(self, scalar_noise_2d):
        """Test optimization flags for scalar noise."""
        handler = DiffusionHandler(
            scalar_noise_2d["diffusion"],
            scalar_noise_2d["state_vars"],
            scalar_noise_2d["control_vars"],
        )

        opts = handler.get_optimization_opportunities()

        assert opts["use_scalar_solver"]

    def test_multiplicative_optimization_flags(self, multiplicative_noise_1d):
        """Test optimization flags for multiplicative noise."""
        handler = DiffusionHandler(
            multiplicative_noise_1d["diffusion"],
            multiplicative_noise_1d["state_vars"],
            multiplicative_noise_1d["control_vars"],
            parameters=multiplicative_noise_1d["parameters"],
        )

        opts = handler.get_optimization_opportunities()

        assert not opts["precompute_diffusion"]
        assert not opts["cache_diffusion"]


# ============================================================================
# Test Parameter Substitution
# ============================================================================


class TestParameterSubstitution:
    """Test parameter substitution in diffusion expressions."""

    def test_parameter_substitution(self, multiplicative_noise_1d):
        """Test that parameters are substituted correctly."""
        handler = DiffusionHandler(
            multiplicative_noise_1d["diffusion"],
            multiplicative_noise_1d["state_vars"],
            multiplicative_noise_1d["control_vars"],
            parameters=multiplicative_noise_1d["parameters"],
        )

        # Generate function (triggers substitution)
        g_func = handler.generate_function("numpy")

        # sigma=0.2, x=1.0 -> expect 0.2
        result = g_func(1.0, 0.0)
        np.testing.assert_almost_equal(result[0, 0], 0.2)

    def test_multiple_parameters(self):
        """Test substitution with multiple parameters."""
        x = sp.symbols("x")
        u = sp.symbols("u")
        a, b = sp.symbols("a b")

        diffusion = sp.Matrix([[a * x + b]])

        handler = DiffusionHandler(
            diffusion,
            [x],
            [u],
            parameters={a: 0.1, b: 0.5},
        )

        g_func = handler.generate_function("numpy")

        # a*x + b = 0.1*2.0 + 0.5 = 0.7
        result = g_func(2.0, 0.0)
        np.testing.assert_almost_equal(result[0, 0], 0.7)

    def test_no_parameters(self, additive_noise_2d):
        """Test that no substitution occurs when no parameters."""
        handler = DiffusionHandler(
            additive_noise_2d["diffusion"],
            additive_noise_2d["state_vars"],
            additive_noise_2d["control_vars"],
        )

        # Should work fine without parameters
        g_func = handler.generate_function("numpy")
        result = g_func(1.0, 2.0, 0.0)

        assert result is not None


# ============================================================================
# Test Compilation Utilities
# ============================================================================


class TestCompilationUtilities:
    """Test compilation and warmup utilities."""

    def test_compile_all_default_backends(self, additive_noise_2d):
        """Test compiling all backends."""
        handler = DiffusionHandler(
            additive_noise_2d["diffusion"],
            additive_noise_2d["state_vars"],
            additive_noise_2d["control_vars"],
        )

        timings = handler.compile_all(verbose=False)

        assert "numpy" in timings
        assert "torch" in timings
        assert "jax" in timings

        # All should be compiled now
        assert handler.is_compiled("numpy")
        assert handler.is_compiled("torch")
        assert handler.is_compiled("jax")

    def test_compile_all_selected_backends(self, additive_noise_2d):
        """Test compiling selected backends."""
        handler = DiffusionHandler(
            additive_noise_2d["diffusion"],
            additive_noise_2d["state_vars"],
            additive_noise_2d["control_vars"],
        )

        timings = handler.compile_all(backends=["numpy"], verbose=False)

        assert "numpy" in timings
        assert "torch" not in timings

        assert handler.is_compiled("numpy")
        assert not handler.is_compiled("torch")

    def test_compile_all_timing_values(self, additive_noise_2d):
        """Test that compilation timing is recorded."""
        handler = DiffusionHandler(
            additive_noise_2d["diffusion"],
            additive_noise_2d["state_vars"],
            additive_noise_2d["control_vars"],
        )

        timings = handler.compile_all(backends=["numpy"], verbose=False)

        assert timings["numpy"] is not None
        assert isinstance(timings["numpy"], float)
        assert timings["numpy"] > 0

    def test_warmup_numpy(self, additive_noise_2d):
        """Test warmup for NumPy backend."""
        handler = DiffusionHandler(
            additive_noise_2d["diffusion"],
            additive_noise_2d["state_vars"],
            additive_noise_2d["control_vars"],
        )

        # Should not raise
        handler.warmup("numpy")

        # Should be compiled after warmup
        assert handler.is_compiled("numpy")

    def test_warmup_jax(self, additive_noise_2d):
        """Test warmup for JAX (JIT) backend."""
        pytest.importorskip("jax")

        handler = DiffusionHandler(
            additive_noise_2d["diffusion"],
            additive_noise_2d["state_vars"],
            additive_noise_2d["control_vars"],
        )

        handler.warmup("jax")
        assert handler.is_compiled("jax")


# ============================================================================
# Test Statistics
# ============================================================================


class TestStatistics:
    """Test generation statistics tracking."""

    def test_initial_stats(self, additive_noise_2d):
        """Test initial statistics state."""
        handler = DiffusionHandler(
            additive_noise_2d["diffusion"],
            additive_noise_2d["state_vars"],
            additive_noise_2d["control_vars"],
        )

        stats = handler.get_stats()

        assert stats["generations"] == 0
        assert stats["cache_hits"] == 0
        assert stats["total_calls"] == 0
        assert stats["cache_hit_rate"] == 0.0

    def test_generation_counting(self, additive_noise_2d):
        """Test that generations are counted."""
        handler = DiffusionHandler(
            additive_noise_2d["diffusion"],
            additive_noise_2d["state_vars"],
            additive_noise_2d["control_vars"],
        )

        handler.generate_function("numpy")
        handler.generate_function("torch")

        stats = handler.get_stats()
        assert stats["generations"] == 2

    def test_cache_hit_counting(self, additive_noise_2d):
        """Test that cache hits are counted."""
        handler = DiffusionHandler(
            additive_noise_2d["diffusion"],
            additive_noise_2d["state_vars"],
            additive_noise_2d["control_vars"],
        )

        handler.generate_function("numpy")  # generation
        handler.generate_function("numpy")  # cache hit
        handler.generate_function("numpy")  # cache hit

        stats = handler.get_stats()
        assert stats["generations"] == 1
        assert stats["cache_hits"] == 2
        assert stats["total_calls"] == 3

    def test_cache_hit_rate(self, additive_noise_2d):
        """Test cache hit rate calculation."""
        handler = DiffusionHandler(
            additive_noise_2d["diffusion"],
            additive_noise_2d["state_vars"],
            additive_noise_2d["control_vars"],
        )

        handler.generate_function("numpy")
        handler.generate_function("numpy")
        handler.generate_function("torch")
        handler.generate_function("numpy")

        stats = handler.get_stats()
        # 2 generations, 2 cache hits, rate = 2/4 = 0.5
        assert stats["cache_hit_rate"] == 0.5

    def test_timing_stats(self, additive_noise_2d):
        """Test timing statistics."""
        handler = DiffusionHandler(
            additive_noise_2d["diffusion"],
            additive_noise_2d["state_vars"],
            additive_noise_2d["control_vars"],
        )

        handler.generate_function("numpy")

        stats = handler.get_stats()
        assert stats["total_time"] > 0
        assert stats["avg_generation_time"] > 0

    def test_reset_stats(self, additive_noise_2d):
        """Test statistics reset."""
        handler = DiffusionHandler(
            additive_noise_2d["diffusion"],
            additive_noise_2d["state_vars"],
            additive_noise_2d["control_vars"],
        )

        handler.generate_function("numpy")
        handler.reset_stats()

        stats = handler.get_stats()
        assert stats["generations"] == 0
        assert stats["cache_hits"] == 0


# ============================================================================
# Test Information Retrieval
# ============================================================================


class TestInformationRetrieval:
    """Test get_info() comprehensive information."""

    def test_get_info_structure(self, additive_noise_2d):
        """Test structure of info dictionary."""
        handler = DiffusionHandler(
            additive_noise_2d["diffusion"],
            additive_noise_2d["state_vars"],
            additive_noise_2d["control_vars"],
        )

        info = handler.get_info()

        assert "dimensions" in info
        assert "noise_type" in info
        assert "characteristics" in info
        assert "compiled" in info
        assert "constant_noise_cached" in info
        assert "statistics" in info

    def test_get_info_dimensions(self, diagonal_noise_3d):
        """Test dimension information."""
        handler = DiffusionHandler(
            diagonal_noise_3d["diffusion"],
            diagonal_noise_3d["state_vars"],
            diagonal_noise_3d["control_vars"],
        )

        info = handler.get_info()

        assert info["dimensions"]["nx"] == 3
        assert info["dimensions"]["nw"] == 3
        assert info["dimensions"]["num_parameters"] == 0

    def test_get_info_noise_type(self, additive_noise_2d):
        """Test noise type in info."""
        handler = DiffusionHandler(
            additive_noise_2d["diffusion"],
            additive_noise_2d["state_vars"],
            additive_noise_2d["control_vars"],
        )

        info = handler.get_info()
        assert info["noise_type"] == "additive"

    def test_get_info_characteristics(self, multiplicative_noise_1d):
        """Test characteristics in info."""
        handler = DiffusionHandler(
            multiplicative_noise_1d["diffusion"],
            multiplicative_noise_1d["state_vars"],
            multiplicative_noise_1d["control_vars"],
            parameters=multiplicative_noise_1d["parameters"],
        )

        info = handler.get_info()
        char = info["characteristics"]

        assert not char["is_additive"]
        assert char["is_multiplicative"]
        assert char["depends_on"]["state"]

    def test_get_info_compilation_status(self, additive_noise_2d):
        """Test compilation status in info."""
        handler = DiffusionHandler(
            additive_noise_2d["diffusion"],
            additive_noise_2d["state_vars"],
            additive_noise_2d["control_vars"],
        )

        handler.generate_function("numpy")

        info = handler.get_info()

        assert info["compiled"]["numpy"]
        assert not info["compiled"]["torch"]
        assert not info["compiled"]["jax"]


# ============================================================================
# Test String Representations
# ============================================================================


class TestStringRepresentations:
    """Test __repr__ and __str__ methods."""

    def test_repr(self, additive_noise_2d):
        """Test __repr__ output."""
        handler = DiffusionHandler(
            additive_noise_2d["diffusion"],
            additive_noise_2d["state_vars"],
            additive_noise_2d["control_vars"],
        )

        repr_str = repr(handler)

        assert "DiffusionHandler" in repr_str
        assert "nx=2" in repr_str
        assert "nw=1" in repr_str
        assert "additive" in repr_str

    def test_str(self, diagonal_noise_3d):
        """Test __str__ output."""
        handler = DiffusionHandler(
            diagonal_noise_3d["diffusion"],
            diagonal_noise_3d["state_vars"],
            diagonal_noise_3d["control_vars"],
        )

        str_repr = str(handler)

        assert "DiffusionHandler" in str_repr
        assert "(3, 3)" in str_repr

    def test_repr_with_compiled_backends(self, additive_noise_2d):
        """Test __repr__ shows compiled backends."""
        handler = DiffusionHandler(
            additive_noise_2d["diffusion"],
            additive_noise_2d["state_vars"],
            additive_noise_2d["control_vars"],
        )

        handler.generate_function("numpy")
        handler.generate_function("torch")

        repr_str = repr(handler)
        assert "numpy" in repr_str
        assert "torch" in repr_str


# ============================================================================
# Test Utility Functions
# ============================================================================


class TestUtilityFunctions:
    """Test module-level utility functions."""

    def test_create_diffusion_handler(self, additive_noise_2d):
        """Test convenience creation function."""
        handler = create_diffusion_handler(
            additive_noise_2d["diffusion"],
            additive_noise_2d["state_vars"],
            additive_noise_2d["control_vars"],
        )

        assert isinstance(handler, DiffusionHandler)
        assert handler.nx == 2
        assert handler.nw == 1

    def test_create_diffusion_handler_with_kwargs(self, multiplicative_noise_1d):
        """Test convenience function with kwargs."""
        handler = create_diffusion_handler(
            multiplicative_noise_1d["diffusion"],
            multiplicative_noise_1d["state_vars"],
            multiplicative_noise_1d["control_vars"],
            parameters=multiplicative_noise_1d["parameters"],
        )

        assert isinstance(handler, DiffusionHandler)
        assert len(handler.parameters) == 1


# ============================================================================
# Test Edge Cases
# ============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_single_state_single_noise(self):
        """Test minimal 1D system."""
        x = sp.symbols("x")
        u = sp.symbols("u")
        diffusion = sp.Matrix([[0.1]])

        handler = DiffusionHandler(diffusion, [x], [u])

        assert handler.nx == 1
        assert handler.nw == 1

        g_func = handler.generate_function("numpy")
        result = g_func(1.0, 0.0)

        np.testing.assert_almost_equal(result[0, 0], 0.1)

    def test_no_control_variables(self):
        """Test system with no control inputs."""
        x = sp.symbols("x")
        diffusion = sp.Matrix([[0.1]])

        handler = DiffusionHandler(diffusion, [x], [])

        g_func = handler.generate_function("numpy")
        result = g_func(1.0)  # Only state, no control

        assert result is not None

    def test_zero_diffusion(self):
        """Test zero diffusion matrix."""
        x = sp.symbols("x")
        u = sp.symbols("u")
        diffusion = sp.Matrix([[0.0]])

        handler = DiffusionHandler(diffusion, [x], [u])

        g_func = handler.generate_function("numpy")
        result = g_func(1.0, 0.0)

        np.testing.assert_almost_equal(result[0, 0], 0.0)

    def test_complex_symbolic_expression(self):
        """Test complex symbolic diffusion expression."""
        x1, x2 = sp.symbols("x1 x2")
        u = sp.symbols("u")

        # Complex expression with trig functions
        diffusion = sp.Matrix([[0.1 * sp.sin(x1) + 0.2 * sp.cos(x2)]])

        handler = DiffusionHandler(diffusion, [x1, x2], [u])

        g_func = handler.generate_function("numpy")
        result = g_func(0.0, 0.0, 0.0)

        # sin(0) + cos(0) * 0.2 = 0.2
        np.testing.assert_almost_equal(result[0, 0], 0.2)


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests with complete workflows."""

    def test_full_workflow_additive(self, additive_noise_2d):
        """Test complete workflow for additive noise."""
        handler = DiffusionHandler(
            additive_noise_2d["diffusion"],
            additive_noise_2d["state_vars"],
            additive_noise_2d["control_vars"],
        )

        # Check characteristics
        assert handler.characteristics.is_additive

        # Generate functions
        g_numpy = handler.generate_function("numpy")
        g_torch = handler.generate_function("torch")

        # Get constant noise
        G = handler.get_constant_noise("numpy")

        # Evaluate
        result = g_numpy(1.0, 2.0, 0.5)

        # Should match constant
        np.testing.assert_array_almost_equal(result, G)

        # Check statistics
        stats = handler.get_stats()
        assert stats["generations"] == 2  # numpy, torch

        # Get info
        info = handler.get_info()
        assert info["noise_type"] == "additive"
        assert info["compiled"]["numpy"]
        assert info["compiled"]["torch"]

    def test_full_workflow_multiplicative(self, multiplicative_noise_1d):
        """Test complete workflow for multiplicative noise."""
        handler = DiffusionHandler(
            multiplicative_noise_1d["diffusion"],
            multiplicative_noise_1d["state_vars"],
            multiplicative_noise_1d["control_vars"],
            parameters=multiplicative_noise_1d["parameters"],
        )

        # Check characteristics
        assert handler.characteristics.is_multiplicative
        assert handler.characteristics.depends_on_state

        # Cannot use constant noise optimization
        assert not handler.can_optimize_for_additive()

        with pytest.raises(ValueError):
            handler.get_constant_noise("numpy")

        # Generate and evaluate
        g_func = handler.generate_function("numpy")

        # Test at different states
        result1 = g_func(1.0, 0.0)
        result2 = g_func(2.0, 0.0)

        # Should be different (state-dependent)
        assert not np.allclose(result1, result2)

        # Check optimization opportunities
        opts = handler.get_optimization_opportunities()
        assert not opts["precompute_diffusion"]


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
