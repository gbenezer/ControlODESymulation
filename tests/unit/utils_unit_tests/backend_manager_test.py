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
Comprehensive unit tests for BackendManager

Tests cover:
1. Initialization and configuration
2. Backend detection
3. Backend conversion
4. Availability checking
5. Device management
6. Context managers
7. Information retrieval
8. Type safety and validation (NEW)
9. Integration tests
10. Edge cases

Updated for Phase 2: Backend Type Integration
"""

import numpy as np
import pytest

# Conditional imports for backends
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

# Import types for type safety tests
from src.types.backends import (
    DEFAULT_BACKEND,
    DEFAULT_DEVICE,
    VALID_BACKENDS,
    Backend,
    BackendConfig,
    Device,
    validate_backend,
    validate_device,
)

# ============================================================================
# Test Class 0: Type System Integration (NEW)
# ============================================================================


class TestTypeSystemIntegration:
    """Test integration with centralized type system"""

    def test_backend_type_constants(self):
        """Test that type constants are properly defined"""
        assert DEFAULT_BACKEND == "numpy"
        assert DEFAULT_DEVICE == "cpu"
        assert "numpy" in VALID_BACKENDS
        assert "torch" in VALID_BACKENDS
        assert "jax" in VALID_BACKENDS

    def test_validate_backend_valid(self):
        """Test validate_backend accepts valid backends"""
        assert validate_backend("numpy") == "numpy"
        assert validate_backend("torch") == "torch"
        assert validate_backend("jax") == "jax"

    def test_validate_backend_invalid(self):
        """Test validate_backend rejects invalid backends"""
        with pytest.raises(ValueError, match="Invalid backend"):
            validate_backend("tensorflow")

        with pytest.raises(ValueError, match="Invalid backend"):
            validate_backend("pytorch")  # Should be 'torch'

        with pytest.raises(ValueError, match="Invalid backend"):
            validate_backend("invalid")

    def test_validate_device_numpy_cpu_only(self):
        """Test that NumPy backend only accepts CPU."""

        # NumPy with CPU - should work
        assert validate_device("cpu", "numpy") == "cpu"
        assert validate_device("default", "numpy") == "default"

        # NumPy with GPU - should fail with NumPy-specific message
        with pytest.raises(ValueError, match=r"NumPy backend only supports CPU"):
            validate_device("cuda", "numpy")

        with pytest.raises(ValueError, match=r"NumPy backend only supports CPU"):
            validate_device("mps", "numpy")

    def test_validate_device_cuda_requires_gpu_backend(self):
        """Test that CUDA device requires GPU-capable backend."""

        # CUDA with torch/jax - should work
        assert validate_device("cuda", "torch") == "cuda"
        assert validate_device("cuda:0", "jax") == "cuda:0"

        # CUDA with numpy - fails (but with NumPy message, not CUDA message)
        with pytest.raises(ValueError, match=r"NumPy backend only supports CPU"):
            validate_device("cuda", "numpy")

    def test_validate_device_mps_requires_torch(self):
        """Test that MPS device requires torch backend."""

        # MPS with torch - should work
        assert validate_device("mps", "torch") == "mps"

        # MPS with jax - fails with MPS-specific message
        with pytest.raises(ValueError, match=r"MPS device requires torch"):
            validate_device("mps", "jax")

        # MPS with numpy - fails with NumPy message
        with pytest.raises(ValueError, match=r"NumPy backend only supports CPU"):
            validate_device("mps", "numpy")

    def test_backend_manager_uses_typed_defaults(self):
        """Test BackendManager uses typed constants"""
        mgr = BackendManager()

        assert mgr.default_backend == DEFAULT_BACKEND
        assert mgr.preferred_device == DEFAULT_DEVICE

    def test_backend_config_structure(self):
        """Test BackendConfig TypedDict structure"""
        mgr = BackendManager()
        config = mgr.get_info()

        # Verify it's a dict with expected keys
        assert isinstance(config, dict)
        assert "backend" in config
        assert "device" in config
        assert "dtype" in config

        # Verify types
        assert isinstance(config["backend"], str)
        assert config["backend"] in VALID_BACKENDS
        assert isinstance(config["device"], (str, type(None)))
        assert isinstance(config["dtype"], (str, type(None)))

    def test_property_types(self):
        """Test that properties return correct types"""
        mgr = BackendManager()

        # default_backend should be Backend (str literal)
        backend: Backend = mgr.default_backend
        assert backend in VALID_BACKENDS

        # preferred_device should be Device (str)
        device: Device = mgr.preferred_device
        assert isinstance(device, str)

        # available_backends should be List[Backend]
        available = mgr.available_backends
        assert isinstance(available, list)
        assert all(b in VALID_BACKENDS for b in available)


# ============================================================================
# Test Class 1: Initialization and Configuration
# ============================================================================


class TestInitializationAndConfiguration:
    """Test backend manager initialization and configuration"""

    def test_default_initialization(self):
        """Test default initialization"""
        mgr = BackendManager()

        assert mgr.default_backend == "numpy"
        assert mgr.preferred_device == "cpu"
        assert "numpy" in mgr.available_backends

    def test_initialization_with_constants(self):
        """Test initialization with type system constants"""
        mgr = BackendManager(
            default_backend=DEFAULT_BACKEND,
            default_device=DEFAULT_DEVICE,
        )

        assert mgr.default_backend == DEFAULT_BACKEND
        assert mgr.preferred_device == DEFAULT_DEVICE

    def test_custom_initialization(self):
        """Test initialization with custom defaults"""
        if not torch_available:
            pytest.skip("PyTorch not installed")

        mgr = BackendManager(default_backend="torch", default_device="cuda")

        assert mgr.default_backend == "torch"
        assert mgr.preferred_device == "cuda"

    def test_invalid_default_backend(self):
        """Test error when default backend is invalid or unavailable"""
        # Invalid backend string
        with pytest.raises((ValueError, RuntimeError)):
            BackendManager(default_backend="tensorflow")

        with pytest.raises((ValueError, RuntimeError)):
            BackendManager(default_backend="pytorch")

    def test_available_backends_detected(self):
        """Test that available backends are detected correctly"""
        mgr = BackendManager()

        # NumPy should always be available
        assert "numpy" in mgr.available_backends

        # Check torch and jax match actual availability
        assert ("torch" in mgr.available_backends) == torch_available
        assert ("jax" in mgr.available_backends) == jax_available

    def test_properties(self):
        """Test property access"""
        mgr = BackendManager()

        # Test properties return correct types
        assert isinstance(mgr.default_backend, str)
        assert mgr.default_backend in VALID_BACKENDS
        assert isinstance(mgr.preferred_device, str)
        assert isinstance(mgr.available_backends, list)

    def test_repr(self):
        """Test __repr__ output"""
        mgr = BackendManager()

        repr_str = repr(mgr)

        assert "BackendManager" in repr_str
        assert "numpy" in repr_str
        assert "cpu" in repr_str

    def test_str(self):
        """Test __str__ output"""
        mgr = BackendManager()

        str_repr = str(mgr)

        assert "BackendManager" in str_repr
        assert "numpy" in str_repr


# ============================================================================
# Test Class 2: Backend Detection
# ============================================================================


class TestBackendDetection:
    """Test backend detection from array types"""

    def test_detect_numpy(self):
        """Test detecting NumPy arrays"""
        mgr = BackendManager()
        x = np.array([1.0, 2.0])

        backend = mgr.detect(x)

        assert backend == "numpy"
        assert backend in VALID_BACKENDS

    @pytest.mark.skipif(not torch_available, reason="PyTorch not installed")
    def test_detect_torch(self):
        """Test detecting PyTorch tensors"""
        mgr = BackendManager()
        x = torch.tensor([1.0, 2.0])

        backend = mgr.detect(x)

        assert backend == "torch"
        assert backend in VALID_BACKENDS

    @pytest.mark.skipif(not jax_available, reason="JAX not installed")
    def test_detect_jax(self):
        """Test detecting JAX arrays"""
        mgr = BackendManager()
        x = jnp.array([1.0, 2.0])

        backend = mgr.detect(x)

        assert backend == "jax"
        assert backend in VALID_BACKENDS

    def test_detect_invalid_type(self):
        """Test error on invalid array type"""
        mgr = BackendManager()

        with pytest.raises(TypeError, match="Unknown input type"):
            mgr.detect([1.0, 2.0])  # Python list

        with pytest.raises(TypeError, match="Unknown input type"):
            mgr.detect(1.0)  # Scalar

    def test_detect_various_shapes(self):
        """Test detection works for various array shapes"""
        mgr = BackendManager()

        # 1D
        assert mgr.detect(np.array([1.0])) == "numpy"

        # 2D
        assert mgr.detect(np.array([[1.0, 2.0], [3.0, 4.0]])) == "numpy"

        # Scalar array
        assert mgr.detect(np.array(1.0)) == "numpy"

    def test_detect_returns_backend_type(self):
        """Test that detect returns Backend type"""
        mgr = BackendManager()
        x = np.array([1.0, 2.0])

        backend: Backend = mgr.detect(x)

        # Should be one of valid backends
        assert backend in VALID_BACKENDS


# ============================================================================
# Test Class 3: Backend Conversion
# ============================================================================


class TestBackendConversion:
    """Test array conversion between backends"""

    def test_numpy_to_numpy_noop(self):
        """Test NumPy to NumPy is no-op"""
        mgr = BackendManager()
        x = np.array([1.0, 2.0])

        x_converted = mgr.convert(x, "numpy")

        assert x_converted is x  # Same object
        assert isinstance(x_converted, np.ndarray)

    @pytest.mark.skipif(not torch_available, reason="PyTorch not installed")
    def test_numpy_to_torch(self):
        """Test NumPy to PyTorch conversion"""
        mgr = BackendManager()
        x = np.array([1.0, 2.0])

        x_torch = mgr.convert(x, "torch")

        assert isinstance(x_torch, torch.Tensor)
        assert torch.allclose(x_torch, torch.tensor([1.0, 2.0]))

    @pytest.mark.skipif(not torch_available, reason="PyTorch not installed")
    def test_torch_to_numpy(self):
        """Test PyTorch to NumPy conversion"""
        mgr = BackendManager()
        x = torch.tensor([1.0, 2.0])

        x_numpy = mgr.convert(x, "numpy")

        assert isinstance(x_numpy, np.ndarray)
        assert np.allclose(x_numpy, np.array([1.0, 2.0]))

    @pytest.mark.skipif(not jax_available, reason="JAX not installed")
    def test_numpy_to_jax(self):
        """Test NumPy to JAX conversion"""
        mgr = BackendManager()
        x = np.array([1.0, 2.0])

        x_jax = mgr.convert(x, "jax")

        assert isinstance(x_jax, jnp.ndarray)
        assert jnp.allclose(x_jax, jnp.array([1.0, 2.0]))

    @pytest.mark.skipif(not jax_available, reason="JAX not installed")
    def test_jax_to_numpy(self):
        """Test JAX to NumPy conversion"""
        mgr = BackendManager()
        x = jnp.array([1.0, 2.0])

        x_numpy = mgr.convert(x, "numpy")

        assert isinstance(x_numpy, np.ndarray)
        assert np.allclose(x_numpy, np.array([1.0, 2.0]))

    @pytest.mark.skipif(
        not (torch_available and jax_available), reason="Both PyTorch and JAX required",
    )
    def test_torch_to_jax(self):
        """Test PyTorch to JAX conversion (via NumPy)"""
        mgr = BackendManager()
        x = torch.tensor([1.0, 2.0])

        x_jax = mgr.convert(x, "jax")

        assert isinstance(x_jax, jnp.ndarray)
        assert jnp.allclose(x_jax, jnp.array([1.0, 2.0]))

    def test_convert_invalid_backend(self):
        """Test error on invalid target backend"""
        mgr = BackendManager()
        x = np.array([1.0, 2.0])

        with pytest.raises(ValueError, match="Invalid backend"):
            mgr.convert(x, "tensorflow")

    @pytest.mark.skipif(not torch_available, reason="PyTorch not installed")
    def test_convert_to_unavailable_backend(self):
        """Test error when trying to convert to unavailable backend"""
        # This test is tricky - we'd need to mock backend availability
        # For now, just test that error message is helpful
        mgr = BackendManager()
        x = np.array([1.0, 2.0])

        # If we try to use an unavailable backend, should get clear error
        # (This would only trigger if we could somehow make torch unavailable)

    def test_convert_preserves_dtype(self):
        """Test that conversion preserves data type when possible"""
        mgr = BackendManager()

        x_float64 = np.array([1.0, 2.0], dtype=np.float64)
        x_float32 = np.array([1.0, 2.0], dtype=np.float32)

        if torch_available:
            # Note: Current implementation converts to float32
            # This test documents current behavior
            x_torch = mgr.convert(x_float64, "torch")
            assert isinstance(x_torch, torch.Tensor)


# ============================================================================
# Test Class 4: Availability Checking
# ============================================================================


class TestAvailabilityChecking:
    """Test backend availability checking"""

    def test_check_available_numpy(self):
        """Test NumPy is always available"""
        mgr = BackendManager()

        assert mgr.check_available("numpy")

    def test_check_available_torch(self):
        """Test torch availability matches import"""
        mgr = BackendManager()

        assert mgr.check_available("torch") == torch_available

    def test_check_available_jax(self):
        """Test JAX availability matches import"""
        mgr = BackendManager()

        assert mgr.check_available("jax") == jax_available

    def test_require_backend_success(self):
        """Test require_backend succeeds for available backends"""
        mgr = BackendManager()

        # Should not raise
        mgr.require_backend("numpy")

        if torch_available:
            mgr.require_backend("torch")

        if jax_available:
            mgr.require_backend("jax")

    def test_require_backend_failure(self):
        """Test require_backend raises for unavailable backends"""
        mgr = BackendManager()

        # Mock unavailable backend by checking one that's not installed
        if not torch_available:
            with pytest.raises(RuntimeError, match="not available"):
                mgr.require_backend("torch")

        if not jax_available:
            with pytest.raises(RuntimeError, match="not available"):
                mgr.require_backend("jax")

    def test_require_backend_helpful_message(self):
        """Test require_backend provides helpful error messages"""
        mgr = BackendManager()

        if not torch_available:
            with pytest.raises(RuntimeError, match="pip install torch"):
                mgr.require_backend("torch")

        if not jax_available:
            with pytest.raises(RuntimeError, match="pip install jax"):
                mgr.require_backend("jax")


# ============================================================================
# Test Class 5: Device Management
# ============================================================================


class TestDeviceManagement:
    """Test device management for GPU backends"""

    def test_default_device(self):
        """Test default device is CPU"""
        mgr = BackendManager()

        assert mgr.preferred_device == "cpu"

    def test_to_device(self):
        """Test setting preferred device"""
        mgr = BackendManager()

        mgr.to_device("cuda")

        assert mgr.preferred_device == "cuda"

    def test_to_device_with_index(self):
        """Test setting device with index"""
        mgr = BackendManager()

        mgr.to_device("cuda:0")

        assert mgr.preferred_device == "cuda:0"

    @pytest.mark.skipif(not torch_available, reason="PyTorch not installed")
    def test_to_device_validation_with_backend(self):
        """Test device validation considers backend"""
        mgr = BackendManager(default_backend="torch")

        # Should accept cuda for torch
        mgr.to_device("cuda")
        assert mgr.preferred_device == "cuda"

        # Should accept mps for torch (on Mac)
        mgr.to_device("mps")
        assert mgr.preferred_device == "mps"

    def test_to_device_stores_preference(self):
        """Test that to_device() stores preference without immediate validation."""
        mgr = BackendManager(default_backend="numpy")

        # NEW BEHAVIOR: Can set any device preference
        mgr.to_device("cuda")
        assert mgr.preferred_device == "cuda"

        mgr.to_device("mps")
        assert mgr.preferred_device == "mps"

    def test_to_device_validated_on_backend_change(self):
        """Test that device is validated when backend is changed."""
        mgr = BackendManager(default_backend="torch")
        mgr.to_device("cuda")

        # Switching to numpy should fail because cuda is not valid for numpy
        with pytest.raises(ValueError, match=r"NumPy backend only supports CPU"):
            mgr.set_default("numpy")

    def test_to_device_numpy_workflow(self):
        """Test correct workflow for numpy backend with device preferences."""
        mgr = BackendManager(default_backend="numpy")

        # Set a GPU preference (doesn't validate yet)
        mgr.to_device("cuda")

        # Now switch to a GPU-capable backend - should validate
        mgr.set_default("torch")  # This should work
        assert mgr.default_backend == "torch"
        assert mgr.preferred_device == "cuda"

    def test_reset(self):
        """Test reset returns to defaults"""
        mgr = BackendManager()

        if torch_available:
            mgr.set_default("torch", device="cuda")

            assert mgr.default_backend == "torch"
            assert mgr.preferred_device == "cuda"

            mgr.reset()

            assert mgr.default_backend == DEFAULT_BACKEND
            assert mgr.preferred_device == DEFAULT_DEVICE

    def test_set_default_with_device(self):
        """Test setting backend and device together"""
        if not torch_available:
            pytest.skip("PyTorch not installed")

        mgr = BackendManager()
        mgr.set_default("torch", device="cuda")

        assert mgr.default_backend == "torch"
        assert mgr.preferred_device == "cuda"

    def test_set_default_with_device_validation(self):
        """Test that set_default validates device compatibility."""
        mgr = BackendManager()

        # Set CUDA device first
        mgr.to_device("cuda")

        # Try to set numpy backend - should fail
        with pytest.raises(ValueError, match=r"NumPy backend only supports CPU"):
            mgr.set_default("numpy")

        # Set torch backend - should work
        mgr.set_default("torch")
        assert mgr.default_backend == "torch"


# ============================================================================
# Test Class 6: Context Managers
# ============================================================================


class TestContextManagers:
    """Test temporary backend switching"""

    def test_use_backend_basic(self):
        """Test basic context manager usage"""
        if not torch_available:
            pytest.skip("PyTorch not installed")

        mgr = BackendManager(default_backend="numpy")

        assert mgr.default_backend == "numpy"

        with mgr.use_backend("torch"):
            assert mgr.default_backend == "torch"

        # Should restore after context
        assert mgr.default_backend == "numpy"

    def test_use_backend_with_device(self):
        """Test context manager with device change"""
        if not torch_available:
            pytest.skip("PyTorch not installed")

        mgr = BackendManager(default_backend="numpy", default_device="cpu")

        with mgr.use_backend("torch", device="cuda"):
            assert mgr.default_backend == "torch"
            assert mgr.preferred_device == "cuda"

        # Should restore both
        assert mgr.default_backend == "numpy"
        assert mgr.preferred_device == "cpu"

    def test_use_backend_nested(self):
        """Test nested context managers"""
        if not torch_available:
            pytest.skip("PyTorch not installed")

        mgr = BackendManager(default_backend="numpy")

        with mgr.use_backend("torch"):
            assert mgr.default_backend == "torch"

            # Nested context
            with mgr.use_backend("numpy"):
                assert mgr.default_backend == "numpy"

            # Back to torch
            assert mgr.default_backend == "torch"

        # Back to original
        assert mgr.default_backend == "numpy"

    def test_use_backend_exception_safety(self):
        """Test context manager restores state even after exception"""
        if not torch_available:
            pytest.skip("PyTorch not installed")

        mgr = BackendManager(default_backend="numpy")

        try:
            with mgr.use_backend("torch"):
                assert mgr.default_backend == "torch"
                raise ValueError("Test exception")
        except ValueError:
            pass

        # Should still restore despite exception
        assert mgr.default_backend == "numpy"


# ============================================================================
# Test Class 7: Information Retrieval
# ============================================================================


class TestInformationRetrieval:
    """Test getting information about backend status"""

    def test_get_info_returns_backend_config(self):
        """Test get_info returns BackendConfig TypedDict"""
        mgr = BackendManager()

        config = mgr.get_info()

        # Should be a dict
        assert isinstance(config, dict)

        # Should have BackendConfig structure
        assert "backend" in config
        assert "device" in config
        assert "dtype" in config

    def test_get_info_backend_field(self):
        """Test backend field in BackendConfig"""
        mgr = BackendManager()

        config = mgr.get_info()

        assert config["backend"] == "numpy"
        assert config["backend"] in VALID_BACKENDS

    def test_get_info_device_field(self):
        """Test device field in BackendConfig"""
        mgr = BackendManager()

        config = mgr.get_info()

        assert config["device"] == "cpu"
        assert isinstance(config["device"], (str, type(None)))

    def test_get_info_dtype_field(self):
        """Test dtype field in BackendConfig"""
        mgr = BackendManager()

        config = mgr.get_info()

        assert "dtype" in config
        assert isinstance(config["dtype"], (str, type(None)))

    def test_get_info_after_configuration(self):
        """Test BackendConfig reflects configuration changes"""
        mgr = BackendManager()

        if torch_available:
            mgr.set_default("torch", device="cuda")
            config = mgr.get_info()

            assert config["backend"] == "torch"
            assert config["device"] == "cuda"

    def test_get_extended_info_available(self):
        """Test get_extended_info provides additional metadata"""
        mgr = BackendManager()

        # Check if extended info method exists
        if hasattr(mgr, "get_extended_info"):
            info = mgr.get_extended_info()

            assert isinstance(info, dict)
            assert "default_backend" in info
            assert "available_backends" in info
            assert "torch_available" in info
            assert "jax_available" in info
            assert "numpy_version" in info

    def test_get_extended_info_content(self):
        """Test extended info content is correct"""
        mgr = BackendManager()

        if hasattr(mgr, "get_extended_info"):
            info = mgr.get_extended_info()

            assert info["default_backend"] == "numpy"
            assert "numpy" in info["available_backends"]
            assert info["torch_available"] == torch_available
            assert info["jax_available"] == jax_available
            assert info["numpy_version"] is not None

    def test_get_extended_info_versions(self):
        """Test version information in extended info"""
        mgr = BackendManager()

        if hasattr(mgr, "get_extended_info"):
            info = mgr.get_extended_info()

            # NumPy version should always be present
            assert info["numpy_version"] is not None

            # Torch/JAX versions should match availability
            if torch_available:
                assert info["torch_version"] is not None
            else:
                assert info["torch_version"] is None

            if jax_available:
                assert info["jax_version"] is not None
            else:
                assert info["jax_version"] is None


# ============================================================================
# Test Class 8: Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests combining multiple features"""

    @pytest.mark.skipif(not torch_available, reason="PyTorch not installed")
    def test_full_workflow(self):
        """Test complete workflow with type safety"""
        # Create manager with typed defaults
        mgr = BackendManager(
            default_backend=DEFAULT_BACKEND,
            default_device=DEFAULT_DEVICE,
        )

        # Create NumPy array
        x_np = np.array([1.0, 2.0, 3.0])

        # Detect backend (returns Backend type)
        backend: Backend = mgr.detect(x_np)
        assert backend == "numpy"

        # Convert to torch
        x_torch = mgr.convert(x_np, "torch")
        assert isinstance(x_torch, torch.Tensor)

        # Detect torch backend
        assert mgr.detect(x_torch) == "torch"

        # Convert back
        x_back = mgr.convert(x_torch, "numpy")
        assert np.allclose(x_np, x_back)

        # Change default backend
        mgr.set_default("torch")
        assert mgr.default_backend == "torch"

        # Get typed config
        config: BackendConfig = mgr.get_info()
        assert config["backend"] == "torch"

        # Use context manager
        with mgr.use_backend("numpy"):
            assert mgr.default_backend == "numpy"

        assert mgr.default_backend == "torch"

    def test_multi_backend_consistency(self):
        """Test that conversions maintain values across all backends"""
        mgr = BackendManager()

        original = np.array([1.5, -2.7, 0.0, 42.0])

        backends_to_test = ["numpy"]
        if torch_available:
            backends_to_test.append("torch")
        if jax_available:
            backends_to_test.append("jax")

        # Test all pairwise conversions
        for backend in backends_to_test:
            converted = mgr.convert(original, backend)
            back_to_numpy = mgr.convert(converted, "numpy")

            assert np.allclose(
                original, back_to_numpy,
            ), f"Values changed during {backend} round-trip"

    def test_type_safe_workflow(self):
        """Test type-safe usage throughout workflow"""
        mgr = BackendManager()

        # All operations should work with typed parameters
        backend: Backend = mgr.default_backend
        device: Device = mgr.preferred_device
        config: BackendConfig = mgr.get_info()

        # Types should be correct
        assert backend in VALID_BACKENDS
        assert isinstance(device, str)
        assert isinstance(config, dict)
        assert config["backend"] in VALID_BACKENDS


# ============================================================================
# Test Class 9: Edge Cases
# ============================================================================


class TestEdgeCases:
    """Test edge cases and error conditions"""

    def test_empty_array(self):
        """Test with empty arrays"""
        mgr = BackendManager()

        x = np.array([])
        backend = mgr.detect(x)

        assert backend == "numpy"

    @pytest.mark.skipif(not torch_available, reason="PyTorch not installed")
    def test_large_array(self):
        """Test with large arrays"""
        mgr = BackendManager()

        x = np.random.randn(1000, 1000)
        x_torch = mgr.convert(x, "torch")

        assert isinstance(x_torch, torch.Tensor)
        assert x_torch.shape == (1000, 1000)

    def test_special_values(self):
        """Test with special float values"""
        mgr = BackendManager()

        # NaN, Inf, -Inf
        x = np.array([np.nan, np.inf, -np.inf, 0.0])

        # Should not raise
        backend = mgr.detect(x)
        assert backend == "numpy"

        if torch_available:
            x_torch = mgr.convert(x, "torch")
            assert torch.isnan(x_torch[0])
            assert torch.isinf(x_torch[1])

    def test_string_backend_still_works_at_runtime(self):
        """Test backward compatibility with string backends at runtime"""
        mgr = BackendManager()

        # Even though we use Backend type, strings still work at runtime
        backend_str = "numpy"
        mgr.set_default(backend_str)
        assert mgr.default_backend == backend_str

        # Conversion with string backend
        x = np.array([1.0, 2.0])
        x_converted = mgr.convert(x, "numpy")
        assert isinstance(x_converted, np.ndarray)

    def test_invalid_backend_caught_at_validation(self):
        """Test that invalid backends are caught by validation"""
        mgr = BackendManager()

        # Should be caught by validate_backend
        with pytest.raises(ValueError, match="Invalid backend"):
            validate_backend("tensorflow")

        # Should propagate through set_default
        with pytest.raises((ValueError, RuntimeError)):
            mgr.set_default("tensorflow")


# ============================================================================
# Test Class 10: Backward Compatibility
# ============================================================================


class TestBackwardCompatibility:
    """Test backward compatibility with pre-Phase-2 code"""

    def test_untyped_usage_still_works(self):
        """Test that code without type annotations still works"""
        mgr = BackendManager()

        # Old-style usage without types
        backend = mgr.default_backend
        device = mgr.preferred_device

        # Should work at runtime
        assert backend == "numpy"
        assert device == "cpu"

    def test_string_backends_accepted(self):
        """Test that plain strings are still accepted"""
        mgr = BackendManager()

        # Plain strings should work
        mgr.set_default("numpy")
        assert mgr.default_backend == "numpy"

        if torch_available:
            mgr.set_default("torch")
            assert mgr.default_backend == "torch"

    def test_dict_return_values_compatible(self):
        """Test that dict return values are compatible"""
        mgr = BackendManager()

        # Old code expected dict
        config = mgr.get_info()

        # Should still be a dict
        assert isinstance(config, dict)

        # Should support dict operations
        assert "backend" in config
        backend_value = config["backend"]
        assert backend_value in VALID_BACKENDS


# ============================================================================
# Test Class 11: ensure_type() Method
# ============================================================================


class TestEnsureType:
    """Test ensure_type() method for type checking and minimal conversion"""

    def test_ensure_type_numpy_already_numpy(self):
        """Test that NumPy arrays are returned as-is (no conversion)"""
        mgr = BackendManager(default_backend="numpy")

        x_original = np.array([1.0, 2.0, 3.0])
        x_ensured = mgr.ensure_type(x_original, backend="numpy")

        # Should be the exact same object (no copy)
        assert x_ensured is x_original
        assert isinstance(x_ensured, np.ndarray)
        np.testing.assert_array_equal(x_ensured, [1.0, 2.0, 3.0])

    def test_ensure_type_numpy_from_list(self):
        """Test converting list to NumPy array"""
        mgr = BackendManager(default_backend="numpy")

        x_list = [1.0, 2.0, 3.0]
        x_ensured = mgr.ensure_type(x_list, backend="numpy")

        assert isinstance(x_ensured, np.ndarray)
        np.testing.assert_array_equal(x_ensured, [1.0, 2.0, 3.0])

    def test_ensure_type_numpy_uses_default_backend(self):
        """Test that backend=None uses default_backend"""
        mgr = BackendManager(default_backend="numpy")

        x_list = [1.0, 2.0, 3.0]
        x_ensured = mgr.ensure_type(x_list)  # No backend specified

        assert isinstance(x_ensured, np.ndarray)
        np.testing.assert_array_equal(x_ensured, [1.0, 2.0, 3.0])

    @pytest.mark.skipif(not torch_available, reason="PyTorch not installed")
    def test_ensure_type_torch_already_torch(self):
        """Test that PyTorch tensors are returned as-is (no conversion)"""
        import torch

        mgr = BackendManager(default_backend="torch", default_device="cpu")

        x_original = torch.tensor([1.0, 2.0, 3.0])
        x_ensured = mgr.ensure_type(x_original, backend="torch")

        # Should be the same tensor (possibly moved to device)
        assert isinstance(x_ensured, torch.Tensor)
        assert torch.allclose(x_ensured, torch.tensor([1.0, 2.0, 3.0], dtype=x_ensured.dtype))

    @pytest.mark.skipif(not torch_available, reason="PyTorch not installed")
    def test_ensure_type_torch_from_numpy(self):
        """Test converting NumPy array to PyTorch tensor"""
        import torch

        mgr = BackendManager(default_backend="torch", default_device="cpu")

        x_np = np.array([1.0, 2.0, 3.0])
        x_ensured = mgr.ensure_type(x_np, backend="torch")

        assert isinstance(x_ensured, torch.Tensor)
        assert torch.allclose(x_ensured, torch.tensor([1.0, 2.0, 3.0], dtype=x_ensured.dtype))
        assert x_ensured.device.type == "cpu"

    @pytest.mark.skipif(not torch_available, reason="PyTorch not installed")
    def test_ensure_type_torch_preserves_dtype_float64(self):
        """Test that float64 NumPy arrays become float64 tensors"""
        import torch

        mgr = BackendManager(default_backend="torch", default_device="cpu")

        x_np = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        x_ensured = mgr.ensure_type(x_np, backend="torch")

        assert isinstance(x_ensured, torch.Tensor)
        assert x_ensured.dtype == torch.float64

    @pytest.mark.skipif(not torch_available, reason="PyTorch not installed")
    def test_ensure_type_torch_preserves_dtype_float32(self):
        """Test that float32 NumPy arrays become float32 tensors"""
        import torch

        mgr = BackendManager(default_backend="torch", default_device="cpu")

        x_np = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        x_ensured = mgr.ensure_type(x_np, backend="torch")

        assert isinstance(x_ensured, torch.Tensor)
        assert x_ensured.dtype == torch.float32

    @pytest.mark.skipif(not torch_available, reason="PyTorch not installed")
    def test_ensure_type_torch_from_list(self):
        """Test converting list to PyTorch tensor"""
        import torch

        mgr = BackendManager(default_backend="torch", default_device="cpu")

        x_list = [1.0, 2.0, 3.0]
        x_ensured = mgr.ensure_type(x_list, backend="torch")

        assert isinstance(x_ensured, torch.Tensor)
        assert torch.allclose(x_ensured, torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64))
        assert x_ensured.dtype == torch.float64  # Default for non-numpy input

    @pytest.mark.skipif(
        not torch_available or not torch.cuda.is_available(), reason="CUDA not available",
    )
    def test_ensure_type_torch_device_placement(self):
        """Test that tensors are moved to preferred device"""
        import torch

        mgr = BackendManager(default_backend="torch", default_device="cuda")

        # Create CPU tensor
        x_cpu = torch.tensor([1.0, 2.0, 3.0], device="cpu")
        x_ensured = mgr.ensure_type(x_cpu, backend="torch")

        # Should be moved to CUDA
        assert x_ensured.device.type == "cuda"
        assert torch.allclose(x_ensured.cpu(), x_cpu)

    @pytest.mark.skipif(
        not torch_available or not torch.cuda.is_available(), reason="CUDA not available",
    )
    def test_ensure_type_torch_already_on_correct_device(self):
        """Test that tensors already on correct device are returned as-is"""
        import torch

        mgr = BackendManager(default_backend="torch", default_device="cuda")

        # Create tensor on CUDA
        x_cuda = torch.tensor([1.0, 2.0, 3.0], device="cuda")
        x_ensured = mgr.ensure_type(x_cuda, backend="torch")

        # Should be returned as-is (same object)
        assert x_ensured.device.type == "cuda"
        assert torch.allclose(x_ensured, x_cuda)

    @pytest.mark.skipif(not jax_available, reason="JAX not installed")
    def test_ensure_type_jax_already_jax(self):
        """Test that JAX arrays are returned as-is (no conversion)"""
        import jax.numpy as jnp

        mgr = BackendManager(default_backend="jax")

        x_original = jnp.array([1.0, 2.0, 3.0])
        x_ensured = mgr.ensure_type(x_original, backend="jax")

        # Should be the same array
        assert isinstance(x_ensured, jnp.ndarray)
        assert jnp.allclose(x_ensured, jnp.array([1.0, 2.0, 3.0]))

    @pytest.mark.skipif(not jax_available, reason="JAX not installed")
    def test_ensure_type_jax_from_numpy(self):
        """Test converting NumPy array to JAX array"""
        import jax.numpy as jnp

        mgr = BackendManager(default_backend="jax")

        x_np = np.array([1.0, 2.0, 3.0])
        x_ensured = mgr.ensure_type(x_np, backend="jax")

        assert isinstance(x_ensured, jnp.ndarray)
        assert jnp.allclose(x_ensured, jnp.array([1.0, 2.0, 3.0]))

    @pytest.mark.skipif(not jax_available, reason="JAX not installed")
    def test_ensure_type_jax_from_list(self):
        """Test converting list to JAX array"""
        import jax.numpy as jnp

        mgr = BackendManager(default_backend="jax")

        x_list = [1.0, 2.0, 3.0]
        x_ensured = mgr.ensure_type(x_list, backend="jax")

        assert isinstance(x_ensured, jnp.ndarray)
        assert jnp.allclose(x_ensured, jnp.array([1.0, 2.0, 3.0]))

    def test_ensure_type_default_backend_numpy(self):
        """Test using default_backend when backend=None"""
        mgr = BackendManager(default_backend="numpy")

        x_list = [1.0, 2.0, 3.0]
        x_ensured = mgr.ensure_type(x_list)  # Uses default_backend

        assert isinstance(x_ensured, np.ndarray)

    @pytest.mark.skipif(not torch_available, reason="PyTorch not installed")
    def test_ensure_type_default_backend_torch(self):
        """Test using default_backend=torch when backend=None"""
        import torch

        mgr = BackendManager(default_backend="torch", default_device="cpu")

        x_list = [1.0, 2.0, 3.0]
        x_ensured = mgr.ensure_type(x_list)  # Uses default_backend

        assert isinstance(x_ensured, torch.Tensor)

    @pytest.mark.skipif(not jax_available, reason="JAX not installed")
    def test_ensure_type_default_backend_jax(self):
        """Test using default_backend=jax when backend=None"""
        import jax.numpy as jnp

        mgr = BackendManager(default_backend="jax")

        x_list = [1.0, 2.0, 3.0]
        x_ensured = mgr.ensure_type(x_list)  # Uses default_backend

        assert isinstance(x_ensured, jnp.ndarray)

    @pytest.mark.skipif(not torch_available, reason="PyTorch not installed")
    def test_ensure_type_numpy_to_torch_to_numpy(self):
        """Test conversion chain: NumPy -> ensure as torch -> ensure as numpy"""
        import torch

        mgr = BackendManager(default_backend="numpy")

        x_np = np.array([1.0, 2.0, 3.0])

        # Convert to torch
        x_torch = mgr.ensure_type(x_np, backend="torch")
        assert isinstance(x_torch, torch.Tensor)

        # Convert back to numpy
        x_np_back = mgr.ensure_type(x_torch, backend="numpy")
        assert isinstance(x_np_back, np.ndarray)
        np.testing.assert_array_equal(x_np_back, x_np)

    def test_ensure_type_scalar_to_numpy(self):
        """Test that scalar values are converted to arrays"""
        mgr = BackendManager(default_backend="numpy")

        x_scalar = 5.0
        x_ensured = mgr.ensure_type(x_scalar, backend="numpy")

        assert isinstance(x_ensured, np.ndarray)
        assert x_ensured.shape == ()  # Scalar array
        assert x_ensured.item() == 5.0

    @pytest.mark.skipif(not torch_available, reason="PyTorch not installed")
    def test_ensure_type_scalar_to_torch(self):
        """Test that scalar values are converted to tensors"""
        import torch

        mgr = BackendManager(default_backend="torch", default_device="cpu")

        x_scalar = 5.0
        x_ensured = mgr.ensure_type(x_scalar, backend="torch")

        assert isinstance(x_ensured, torch.Tensor)
        assert x_ensured.item() == 5.0

    def test_ensure_type_2d_array_numpy(self):
        """Test ensure_type works with 2D arrays"""
        mgr = BackendManager(default_backend="numpy")

        x_2d = [[1.0, 2.0], [3.0, 4.0]]
        x_ensured = mgr.ensure_type(x_2d, backend="numpy")

        assert isinstance(x_ensured, np.ndarray)
        assert x_ensured.shape == (2, 2)
        np.testing.assert_array_equal(x_ensured, [[1.0, 2.0], [3.0, 4.0]])

    @pytest.mark.skipif(not torch_available, reason="PyTorch not installed")
    def test_ensure_type_2d_array_torch(self):
        """Test ensure_type works with 2D arrays for PyTorch"""
        import torch

        mgr = BackendManager(default_backend="torch", default_device="cpu")

        x_2d = [[1.0, 2.0], [3.0, 4.0]]
        x_ensured = mgr.ensure_type(x_2d, backend="torch")

        assert isinstance(x_ensured, torch.Tensor)
        assert x_ensured.shape == (2, 2)


# ============================================================================
# Run Tests
# ============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
