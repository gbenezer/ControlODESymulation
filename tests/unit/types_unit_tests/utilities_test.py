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
Unit Tests for Utilities Types Module

Tests cover:
- Protocol definitions (structural subtyping)
- Type guard functions (is_batched, is_numpy, etc.)
- Backend detection (get_backend)
- Type conversions (ensure_numpy, ensure_backend)
- Shape validators (check_state_shape, check_control_shape)
- Dimension extraction
- ArrayConverter class
- Cache and metadata types
- Validation and performance types
- Edge cases and error handling
"""

from typing import Optional

import numpy as np
import pytest

from src.types.core import ControlVector, StateVector
from src.types.utilities import (  # Protocols; Type guards; Converters; Validators; Cache and metadata; Validation and performance
    ArrayConverter,
    CacheKey,
    CacheStatistics,
    LinearizableProtocol,
    Metadata,
    PerformanceMetrics,
    SimulatableProtocol,
    ValidationResult,
    check_control_shape,
    check_state_shape,
    ensure_backend,
    ensure_numpy,
    extract_dimensions,
    get_array_shape,
    get_backend,
    get_batch_size,
    is_batched,
    is_jax,
    is_numpy,
    is_torch,
)

# Optional PyTorch import for tests
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Optional JAX import for tests
try:
    import jax.numpy as jnp

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False


# ============================================================================
# Test Protocol Definitions
# ============================================================================


class TestProtocols:
    """Test protocol definitions for structural subtyping."""

    def test_linearizable_protocol_compliance(self):
        """Test that class with linearize() satisfies protocol."""

        class MockLinearizableSystem:
            def linearize(self, x_eq: StateVector, u_eq: Optional[ControlVector] = None, **kwargs):
                A = np.eye(2)
                B = np.zeros((2, 1))
                return (A, B)

        system = MockLinearizableSystem()

        # Should satisfy protocol (structural subtyping)
        # In practice, mypy would verify this statically
        assert hasattr(system, "linearize")
        assert callable(system.linearize)

        # Should work with protocol-typed function
        def analyze(sys: LinearizableProtocol):
            result = sys.linearize(np.zeros(2), np.zeros(1))
            return result

        result = analyze(system)
        assert len(result) == 2

    def test_simulatable_protocol_compliance(self):
        """Test that class with step() satisfies protocol."""

        class MockSimulatableSystem:
            def step(self, x: StateVector, u: Optional[ControlVector] = None, **kwargs):
                return x + 0.01 * u if u is not None else x

        system = MockSimulatableSystem()

        assert hasattr(system, "step")
        assert callable(system.step)

        # Should work with protocol
        def simulate(sys: SimulatableProtocol, x0, u):
            return sys.step(x0, u)

        x_next = simulate(system, np.zeros(2), np.ones(2))
        assert x_next.shape == (2,)

    def test_stochastic_protocol_compliance(self):
        """Test that stochastic class satisfies protocol."""

        class MockStochasticSystem:
            @property
            def is_stochastic(self) -> bool:
                return True

            def diffusion(self, x: StateVector, u: Optional[ControlVector] = None, **kwargs):
                return 0.1 * np.eye(len(x))

        system = MockStochasticSystem()

        assert hasattr(system, "is_stochastic")
        assert hasattr(system, "diffusion")
        assert system.is_stochastic is True

        G = system.diffusion(np.zeros(3))
        assert G.shape == (3, 3)


# ============================================================================
# Test Type Guard Functions
# ============================================================================


class TestTypeGuards:
    """Test type guard functions."""

    def test_is_batched_single_vector(self):
        """Test is_batched with single vector."""
        x = np.array([1, 2, 3])
        assert not is_batched(x)

    def test_is_batched_batch_vectors(self):
        """Test is_batched with batched vectors."""
        x_batch = np.array([[1, 2, 3], [4, 5, 6]])
        assert is_batched(x_batch)

    def test_is_batched_trajectory(self):
        """Test is_batched with trajectory (time series)."""
        x_traj = np.random.randn(100, 3)
        assert is_batched(x_traj)

    def test_is_batched_scalar(self):
        """Test is_batched with scalar."""
        x_scalar = np.array(5.0)
        assert not is_batched(x_scalar)

    def test_get_batch_size_single(self):
        """Test get_batch_size with single vector."""
        x = np.array([1, 2, 3])
        assert get_batch_size(x) is None

    def test_get_batch_size_batched(self):
        """Test get_batch_size with batched vectors."""
        x_batch = np.random.randn(50, 3)
        assert get_batch_size(x_batch) == 50

    def test_get_batch_size_trajectory(self):
        """Test get_batch_size with trajectory."""
        x_traj = np.random.randn(100, 5)
        assert get_batch_size(x_traj) == 100

    def test_is_numpy_true(self):
        """Test is_numpy with NumPy array."""
        x = np.array([1, 2, 3])
        assert is_numpy(x)

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_is_numpy_false_torch(self):
        """Test is_numpy with PyTorch tensor."""
        x = torch.tensor([1, 2, 3])
        assert not is_numpy(x)

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_is_torch_true(self):
        """Test is_torch with PyTorch tensor."""
        x = torch.tensor([1, 2, 3])
        assert is_torch(x)

    def test_is_torch_false_numpy(self):
        """Test is_torch with NumPy array."""
        x = np.array([1, 2, 3])
        assert not is_torch(x)

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_is_jax_true(self):
        """Test is_jax with JAX array."""
        x = jnp.array([1, 2, 3])
        assert is_jax(x)

    def test_is_jax_false_numpy(self):
        """Test is_jax with NumPy array."""
        x = np.array([1, 2, 3])
        assert not is_jax(x)

    def test_get_backend_numpy(self):
        """Test get_backend with NumPy array."""
        x = np.array([1, 2, 3])
        assert get_backend(x) == "numpy"

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_get_backend_torch(self):
        """Test get_backend with PyTorch tensor."""
        x = torch.tensor([1, 2, 3])
        assert get_backend(x) == "torch"

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_get_backend_jax(self):
        """Test get_backend with JAX array."""
        x = jnp.array([1, 2, 3])
        assert get_backend(x) == "jax"

    def test_get_backend_unknown_raises(self):
        """Test get_backend raises for unknown type."""
        with pytest.raises(TypeError, match="Unknown backend"):
            get_backend([1, 2, 3])  # Plain list


# ============================================================================
# Test Type Conversion Functions
# ============================================================================


class TestTypeConversions:
    """Test type conversion functions."""

    def test_ensure_numpy_from_numpy(self):
        """Test ensure_numpy with NumPy input (no conversion)."""
        x_in = np.array([1.0, 2.0, 3.0])
        x_out = ensure_numpy(x_in)

        assert isinstance(x_out, np.ndarray)
        assert np.array_equal(x_out, x_in)
        assert x_out is x_in  # Should be same object

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_ensure_numpy_from_torch(self):
        """Test ensure_numpy with PyTorch tensor."""
        x_torch = torch.tensor([1.0, 2.0, 3.0])
        x_np = ensure_numpy(x_torch)

        assert isinstance(x_np, np.ndarray)
        assert np.allclose(x_np, [1.0, 2.0, 3.0])

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_ensure_numpy_from_torch_gpu(self):
        """Test ensure_numpy with PyTorch GPU tensor."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        x_cuda = torch.tensor([1.0, 2.0, 3.0], device="cuda")
        x_np = ensure_numpy(x_cuda)

        assert isinstance(x_np, np.ndarray)
        assert np.allclose(x_np, [1.0, 2.0, 3.0])

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_ensure_numpy_from_jax(self):
        """Test ensure_numpy with JAX array."""
        x_jax = jnp.array([1.0, 2.0, 3.0])
        x_np = ensure_numpy(x_jax)

        assert isinstance(x_np, np.ndarray)
        assert np.allclose(x_np, [1.0, 2.0, 3.0])

    def test_ensure_backend_numpy_to_numpy(self):
        """Test ensure_backend NumPy to NumPy (no conversion)."""
        x_in = np.array([1, 2, 3])
        x_out = ensure_backend(x_in, "numpy")

        assert isinstance(x_out, np.ndarray)
        assert x_out is x_in

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_ensure_backend_numpy_to_torch(self):
        """Test ensure_backend NumPy to PyTorch."""
        x_np = np.array([1.0, 2.0, 3.0])
        x_torch = ensure_backend(x_np, "torch")

        assert isinstance(x_torch, torch.Tensor)
        # Convert to same dtype for comparison
        expected = torch.tensor([1.0, 2.0, 3.0], dtype=x_torch.dtype)
        assert torch.allclose(x_torch, expected)

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_ensure_backend_numpy_to_jax(self):
        """Test ensure_backend NumPy to JAX."""
        x_np = np.array([1.0, 2.0, 3.0])
        x_jax = ensure_backend(x_np, "jax")

        assert isinstance(x_jax, jnp.ndarray)
        assert np.allclose(np.array(x_jax), [1.0, 2.0, 3.0])

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_ensure_backend_torch_to_numpy(self):
        """Test ensure_backend PyTorch to NumPy."""
        x_torch = torch.tensor([1.0, 2.0, 3.0])
        x_np = ensure_backend(x_torch, "numpy")

        assert isinstance(x_np, np.ndarray)
        assert np.allclose(x_np, [1.0, 2.0, 3.0])

    def test_ensure_backend_invalid_raises(self):
        """Test ensure_backend with invalid backend raises."""
        x = np.array([1, 2, 3])
        with pytest.raises((ValueError, TypeError)):
            ensure_backend(x, "invalid_backend")


# ============================================================================
# Test ArrayConverter Class
# ============================================================================


class TestArrayConverter:
    """Test ArrayConverter class."""

    def test_array_converter_to_numpy(self):
        """Test ArrayConverter.to_numpy()."""
        x = np.array([1, 2, 3])
        x_np = ArrayConverter.to_numpy(x)

        assert isinstance(x_np, np.ndarray)
        assert np.array_equal(x_np, x)

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_array_converter_to_torch(self):
        """Test ArrayConverter.to_torch()."""
        x_np = np.array([1.0, 2.0, 3.0])
        x_torch = ArrayConverter.to_torch(x_np)

        assert isinstance(x_torch, torch.Tensor)
        # Convert to same dtype for comparison
        expected = torch.tensor([1.0, 2.0, 3.0], dtype=x_torch.dtype)
        assert torch.allclose(x_torch, expected)

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_array_converter_to_torch_with_device(self):
        """Test ArrayConverter.to_torch() with device."""
        x_np = np.array([1.0, 2.0, 3.0])
        x_torch = ArrayConverter.to_torch(x_np, device="cpu")

        assert x_torch.device.type == "cpu"

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_array_converter_to_jax(self):
        """Test ArrayConverter.to_jax()."""
        x_np = np.array([1.0, 2.0, 3.0])
        x_jax = ArrayConverter.to_jax(x_np)

        assert isinstance(x_jax, jnp.ndarray)
        assert np.allclose(np.array(x_jax), [1.0, 2.0, 3.0])

    def test_array_converter_convert_to_numpy(self):
        """Test ArrayConverter.convert() to NumPy."""
        x = np.array([1, 2, 3])
        x_converted = ArrayConverter.convert(x, "numpy")

        assert isinstance(x_converted, np.ndarray)

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_array_converter_convert_to_torch(self):
        """Test ArrayConverter.convert() to PyTorch."""
        x_np = np.array([1.0, 2.0, 3.0])
        x_torch = ArrayConverter.convert(x_np, "torch")

        assert isinstance(x_torch, torch.Tensor)


# ============================================================================
# Test Shape Validators
# ============================================================================


class TestShapeValidators:
    """Test shape validation functions."""

    def test_check_state_shape_valid_single(self):
        """Test check_state_shape with valid single state."""
        x = np.array([1.0, 2.0, 3.0])
        check_state_shape(x, nx=3)  # Should not raise

    def test_check_state_shape_valid_batched(self):
        """Test check_state_shape with valid batched states."""
        x_batch = np.random.randn(10, 3)
        check_state_shape(x_batch, nx=3)  # Should not raise

    def test_check_state_shape_invalid_single(self):
        """Test check_state_shape with invalid dimension."""
        x = np.array([1.0, 2.0])
        with pytest.raises(ValueError, match="incorrect dimension"):
            check_state_shape(x, nx=3)

    def test_check_state_shape_invalid_batched(self):
        """Test check_state_shape with invalid batched dimension."""
        x_batch = np.random.randn(10, 2)
        with pytest.raises(ValueError, match="incorrect dimension"):
            check_state_shape(x_batch, nx=3)

    def test_check_state_shape_custom_name(self):
        """Test check_state_shape with custom parameter name."""
        x = np.array([1.0, 2.0])
        with pytest.raises(ValueError, match="initial_state"):
            check_state_shape(x, nx=3, name="initial_state")

    def test_check_control_shape_valid_single(self):
        """Test check_control_shape with valid single control."""
        u = np.array([0.5])
        check_control_shape(u, nu=1)  # Should not raise

    def test_check_control_shape_valid_batched(self):
        """Test check_control_shape with valid batched controls."""
        u_batch = np.random.randn(10, 2)
        check_control_shape(u_batch, nu=2)  # Should not raise

    def test_check_control_shape_invalid_single(self):
        """Test check_control_shape with invalid dimension."""
        u = np.array([0.5, 0.3])
        with pytest.raises(ValueError, match="incorrect dimension"):
            check_control_shape(u, nu=1)

    def test_check_control_shape_invalid_batched(self):
        """Test check_control_shape with invalid batched dimension."""
        u_batch = np.random.randn(10, 2)
        with pytest.raises(ValueError, match="incorrect dimension"):
            check_control_shape(u_batch, nu=3)

    def test_get_array_shape_1d(self):
        """Test get_array_shape with 1D array."""
        x = np.array([1, 2, 3])
        shape = get_array_shape(x)
        assert shape == (3,)

    def test_get_array_shape_2d(self):
        """Test get_array_shape with 2D array."""
        x = np.random.randn(10, 3)
        shape = get_array_shape(x)
        assert shape == (10, 3)

    def test_get_array_shape_scalar(self):
        """Test get_array_shape with scalar."""
        x = np.array(5.0)
        shape = get_array_shape(x)
        assert shape == ()

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_get_array_shape_torch(self):
        """Test get_array_shape with PyTorch tensor."""
        x = torch.randn(5, 2)
        shape = get_array_shape(x)
        assert shape == (5, 2)


# ============================================================================
# Test Dimension Extraction
# ============================================================================


class TestDimensionExtraction:
    """Test dimension extraction from vectors."""

    def test_extract_dimensions_all_vectors(self):
        """Test extract_dimensions with all vectors provided."""
        x = np.array([1, 2, 3])
        u = np.array([0.5])
        y = np.array([1, 2, 3])

        dims = extract_dimensions(x, u, y)

        assert dims["nx"] == 3
        assert dims["nu"] == 1
        assert dims["ny"] == 3
        assert dims["nw"] == 0

    def test_extract_dimensions_state_only(self):
        """Test extract_dimensions with state only."""
        x = np.array([1, 2, 3, 4])

        dims = extract_dimensions(x=x)

        assert dims["nx"] == 4
        assert dims["nu"] == 0
        assert dims["ny"] == 4  # Defaults to nx

    def test_extract_dimensions_state_and_control(self):
        """Test extract_dimensions with state and control."""
        x = np.array([1, 2, 3])
        u = np.array([0.5, 0.3])

        dims = extract_dimensions(x=x, u=u)

        assert dims["nx"] == 3
        assert dims["nu"] == 2
        assert dims["ny"] == 3

    def test_extract_dimensions_batched_vectors(self):
        """Test extract_dimensions with batched vectors."""
        x_batch = np.random.randn(10, 5)
        u_batch = np.random.randn(10, 2)

        dims = extract_dimensions(x_batch, u_batch)

        assert dims["nx"] == 5
        assert dims["nu"] == 2

    def test_extract_dimensions_empty(self):
        """Test extract_dimensions with no vectors."""
        dims = extract_dimensions()

        assert dims["nx"] == 0
        assert dims["nu"] == 0
        assert dims["ny"] == 0


# ============================================================================
# Test Cache and Metadata Types
# ============================================================================


class TestCacheAndMetadata:
    """Test cache and metadata types."""

    def test_cache_key_is_string(self):
        """Test CacheKey is string type."""
        key: CacheKey = "x_eq=[0,0]_u_eq=[0]"
        assert isinstance(key, str)

    def test_cache_key_in_dict(self):
        """Test using CacheKey in dictionary."""
        cache: dict[CacheKey, tuple] = {}

        key: CacheKey = "linearization_1"
        cache[key] = (np.eye(2), np.zeros((2, 1)))

        assert key in cache
        assert len(cache[key]) == 2

    def test_cache_statistics_structure(self):
        """Test CacheStatistics TypedDict structure."""
        stats: CacheStatistics = {
            "computes": 10,
            "cache_hits": 40,
            "cache_misses": 10,
            "total_requests": 50,
            "cache_size": 10,
            "hit_rate": 0.8,
        }

        assert stats["computes"] == 10
        assert stats["hit_rate"] == 0.8

    def test_cache_statistics_hit_rate_calculation(self):
        """Test hit rate calculation in CacheStatistics."""
        stats: CacheStatistics = {
            "computes": 5,
            "cache_hits": 45,
            "cache_misses": 5,
            "total_requests": 50,
            "cache_size": 5,
            "hit_rate": 45 / 50,
        }

        assert stats["hit_rate"] == 0.9

    def test_metadata_flexible_structure(self):
        """Test Metadata allows arbitrary keys."""
        metadata: Metadata = {
            "timestamp": "2025-01-01T00:00:00",
            "version": "1.0.0",
            "author": "user",
            "custom_field": 42,
            "nested": {"key": "value"},
        }

        assert metadata["version"] == "1.0.0"
        assert metadata["custom_field"] == 42


# ============================================================================
# Test Validation and Performance Types
# ============================================================================


class TestValidationAndPerformance:
    """Test validation and performance metric types."""

    def test_validation_result_valid(self):
        """Test ValidationResult for valid case."""
        result: ValidationResult = {
            "valid": True,
            "errors": [],
            "warnings": ["Minor issue"],
            "checks_passed": 10,
            "checks_total": 10,
        }

        assert result["valid"] is True
        assert len(result["errors"]) == 0
        assert result["checks_passed"] == result["checks_total"]

    def test_validation_result_invalid(self):
        """Test ValidationResult for invalid case."""
        result: ValidationResult = {
            "valid": False,
            "errors": ["Critical error", "Another error"],
            "warnings": [],
            "checks_passed": 8,
            "checks_total": 10,
        }

        assert result["valid"] is False
        assert len(result["errors"]) == 2
        assert result["checks_passed"] < result["checks_total"]

    def test_validation_result_partial(self):
        """Test ValidationResult with partial fields (total=False)."""
        result: ValidationResult = {"valid": True}

        assert "valid" in result
        assert "errors" not in result

    def test_performance_metrics_complete(self):
        """Test PerformanceMetrics with all fields."""
        metrics: PerformanceMetrics = {
            "settling_time": 2.5,
            "rise_time": 0.8,
            "overshoot": 12.3,
            "steady_state_error": 0.05,
            "control_effort": 15.2,
            "trajectory_cost": 100.5,
        }

        assert metrics["settling_time"] == 2.5
        assert metrics["overshoot"] == 12.3

    def test_performance_metrics_partial(self):
        """Test PerformanceMetrics with subset of fields."""
        metrics: PerformanceMetrics = {
            "settling_time": 3.0,
            "overshoot": 8.5,
        }

        assert "settling_time" in metrics
        assert "rise_time" not in metrics


# ============================================================================
# Test Realistic Usage Patterns
# ============================================================================


class TestRealisticUsage:
    """Test types in realistic scenarios."""

    def test_backend_agnostic_function(self):
        """Test function that works with any backend."""

        def process_array(x):
            """Process array regardless of backend."""
            # Detect backend
            backend = get_backend(x)

            # Convert to NumPy for processing
            x_np = ensure_numpy(x)

            # Process
            result = x_np * 2

            # Convert back to original backend
            return ensure_backend(result, backend)

        # Test with NumPy
        x_np = np.array([1, 2, 3])
        result_np = process_array(x_np)
        assert is_numpy(result_np)
        assert np.array_equal(result_np, [2, 4, 6])

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_backend_agnostic_with_torch(self):
        """Test backend-agnostic function with PyTorch."""

        def compute(x):
            backend = get_backend(x)
            x_np = ensure_numpy(x)
            result = x_np + 1
            return ensure_backend(result, backend)

        x_torch = torch.tensor([1.0, 2.0, 3.0])
        result_torch = compute(x_torch)
        assert is_torch(result_torch)

    def test_validate_and_process_state(self):
        """Test validation before processing."""

        def safe_process(x, nx):
            # Validate first
            check_state_shape(x, nx, name="input_state")

            # Process
            return x * 2

        # Valid input
        x = np.array([1, 2, 3])
        result = safe_process(x, nx=3)
        assert np.array_equal(result, [2, 4, 6])

        # Invalid input
        x_wrong = np.array([1, 2])
        with pytest.raises(ValueError):
            safe_process(x_wrong, nx=3)

    def test_protocol_based_controller(self):
        """Test using protocol for polymorphic controller."""

        class SimpleSystem:
            def linearize(self, x_eq, u_eq=None, **kwargs):
                A = np.array([[-1, 0], [0, -2]])
                B = np.array([[1], [0]])
                return (A, B)

        def design_lqr(system: LinearizableProtocol, x_eq, u_eq):
            """Design LQR for any linearizable system."""
            result = system.linearize(x_eq, u_eq)
            A, B = result[0], result[1]

            # Check stability
            eigenvalues = np.linalg.eigvals(A)
            is_stable = np.all(np.real(eigenvalues) < 0)

            return {"A": A, "B": B, "stable": is_stable}

        system = SimpleSystem()
        lqr_result = design_lqr(system, np.zeros(2), np.zeros(1))
        assert lqr_result["stable"] == True  # Use == instead of is for NumPy bool

    def test_caching_workflow(self):
        """Test caching workflow with cache statistics."""
        cache: dict[CacheKey, tuple] = {}
        stats: CacheStatistics = {
            "computes": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "total_requests": 0,
            "cache_size": 0,
            "hit_rate": 0.0,
        }

        def expensive_computation(x, cache_dict, cache_stats):
            """Expensive computation with caching."""
            key: CacheKey = f"x={hash(x.tobytes())}"
            cache_stats["total_requests"] += 1

            if key in cache_dict:
                cache_stats["cache_hits"] += 1
                return cache_dict[key]
            cache_stats["cache_misses"] += 1
            cache_stats["computes"] += 1
            result = x @ x.T  # Expensive operation
            cache_dict[key] = result
            cache_stats["cache_size"] = len(cache_dict)
            return result

        # First call - cache miss
        x = np.random.randn(10, 10)
        result1 = expensive_computation(x, cache, stats)
        assert stats["cache_misses"] == 1
        assert stats["cache_hits"] == 0

        # Second call - cache hit
        result2 = expensive_computation(x, cache, stats)
        assert stats["cache_hits"] == 1
        assert np.array_equal(result1, result2)


# ============================================================================
# Test Edge Cases
# ============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_is_batched_3d_array(self):
        """Test is_batched with 3D array (batched trajectory)."""
        x = np.random.randn(10, 100, 3)  # (batch, time, nx)
        assert is_batched(x)
        assert get_batch_size(x) == 10

    def test_check_shape_empty_array(self):
        """Test shape checking with empty array."""
        x = np.array([])
        with pytest.raises(ValueError):
            check_state_shape(x, nx=3)

    def test_extract_dimensions_scalar_system(self):
        """Test extract_dimensions with scalar system."""
        x = np.array([1.0])
        u = np.array([0.5])

        dims = extract_dimensions(x, u)

        assert dims["nx"] == 1
        assert dims["nu"] == 1

    def test_extract_dimensions_autonomous_system(self):
        """Test extract_dimensions with autonomous system (nu=0)."""
        x = np.array([1, 2, 3])
        u = np.array([])

        dims = extract_dimensions(x, u)

        assert dims["nx"] == 3
        assert dims["nu"] == 0

    def test_ensure_numpy_preserves_dtype(self):
        """Test ensure_numpy preserves data type."""
        x_float32 = np.array([1, 2, 3], dtype=np.float32)
        x_converted = ensure_numpy(x_float32)

        assert x_converted.dtype == np.float32

    def test_validation_result_empty_lists(self):
        """Test ValidationResult with empty error/warning lists."""
        result: ValidationResult = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "checks_passed": 5,
            "checks_total": 5,
        }

        assert len(result["errors"]) == 0
        assert len(result["warnings"]) == 0


# ============================================================================
# Test Documentation Examples
# ============================================================================


class TestDocumentationExamples:
    """Test examples from docstrings work correctly."""

    def test_is_batched_example(self):
        """Test is_batched docstring example."""
        # Single state vector
        x_single = np.array([1, 2, 3])
        assert is_batched(x_single) is False

        # Batched state vectors
        x_batch = np.array([[1, 2, 3], [4, 5, 6]])
        assert is_batched(x_batch) is True

    def test_get_backend_example(self):
        """Test get_backend docstring example."""
        x_np = np.array([1, 2, 3])
        assert get_backend(x_np) == "numpy"

    def test_ensure_numpy_example(self):
        """Test ensure_numpy docstring example."""
        x_np2 = ensure_numpy(np.array([1, 2, 3]))
        assert type(x_np2) == np.ndarray

    def test_check_state_shape_example(self):
        """Test check_state_shape docstring example."""
        x = np.array([1, 2, 3])
        check_state_shape(x, nx=3, name="initial_state")  # Should not raise

    def test_extract_dimensions_example(self):
        """Test extract_dimensions docstring example."""
        x = np.array([1, 2, 3])
        u = np.array([0.5])
        y = np.array([1, 2, 3])

        dims = extract_dimensions(x, u, y)
        assert dims["nx"] == 3
        assert dims["nu"] == 1
        assert dims["ny"] == 3


# ============================================================================
# Test Error Messages
# ============================================================================


class TestErrorMessages:
    """Test that error messages are informative."""

    def test_check_state_shape_error_message(self):
        """Test check_state_shape gives helpful error."""
        x = np.array([1, 2])
        with pytest.raises(ValueError) as exc_info:
            check_state_shape(x, nx=3, name="test_state")

        assert "test_state" in str(exc_info.value)
        assert "incorrect dimension" in str(exc_info.value)

    def test_get_backend_error_message(self):
        """Test get_backend gives helpful error for unknown type."""
        with pytest.raises(TypeError) as exc_info:
            get_backend([1, 2, 3])

        assert "Unknown backend" in str(exc_info.value)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
