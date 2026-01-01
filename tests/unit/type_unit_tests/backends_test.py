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
Unit Tests for Backend Types Module

Tests cover:
- Backend literal types
- Device validation
- Method type definitions
- Noise and stochastic types
- Configuration TypedDicts
- Constants validation
- Utility function behavior
- Method selection logic
- Backend/device compatibility
- Default value correctness
"""

import numpy as np
import pytest

from src.types.backends import (  # Backend types; Method types; Noise types; Configuration; Constants; Utilities
    DEFAULT_BACKEND,
    DEFAULT_DEVICE,
    DEFAULT_DTYPE,
    VALID_BACKENDS,
    VALID_DEVICES,
    Backend,
    BackendConfig,
    ConvergenceType,
    Device,
    DiscretizationMethod,
    DiscretizerConfig,
    IntegrationMethod,
    IntegratorConfig,
    NoiseType,
    OptimizationMethod,
    SDEIntegrationMethod,
    SDEIntegratorConfig,
    SDEType,
    SystemConfig,
    get_backend_default_method,
    validate_backend,
    validate_device,
)

# ============================================================================
# Test Backend Type Definitions
# ============================================================================


class TestBackendTypes:
    """Test backend type definitions."""

    def test_backend_valid_values(self):
        """Test Backend literal accepts valid values."""
        backend1: Backend = "numpy"
        backend2: Backend = "torch"
        backend3: Backend = "jax"

        assert backend1 == "numpy"
        assert backend2 == "torch"
        assert backend3 == "jax"

    def test_device_is_string(self):
        """Test Device is string type."""
        device1: Device = "cpu"
        device2: Device = "cuda"
        device3: Device = "cuda:0"
        device4: Device = "mps"

        assert isinstance(device1, str)
        assert isinstance(device2, str)

    def test_backend_config_dict(self):
        """Test BackendConfig TypedDict."""
        config: BackendConfig = {"backend": "torch", "device": "cuda:0", "dtype": "float32"}

        assert config["backend"] == "torch"
        assert config["device"] == "cuda:0"
        assert config["dtype"] == "float32"

    def test_backend_config_minimal(self):
        """Test minimal BackendConfig (total=False)."""
        config: BackendConfig = {"backend": "numpy"}

        assert "backend" in config
        assert "device" not in config  # Optional


# ============================================================================
# Test Method Type Definitions
# ============================================================================


class TestMethodTypes:
    """Test method type definitions."""

    def test_integration_method_examples(self):
        """Test common IntegrationMethod values."""
        method1: IntegrationMethod = "RK45"
        method2: IntegrationMethod = "DOP853"
        method3: IntegrationMethod = "euler"
        method4: IntegrationMethod = "Radau"

        assert isinstance(method1, str)

    def test_discretization_method_examples(self):
        """Test common DiscretizationMethod values."""
        method1: DiscretizationMethod = "euler"
        method2: DiscretizationMethod = "exact"
        method3: DiscretizationMethod = "tustin"
        method4: DiscretizationMethod = "zoh"

        assert isinstance(method1, str)

    def test_sde_integration_method_examples(self):
        """Test common SDEIntegrationMethod values."""
        method1: SDEIntegrationMethod = "euler"
        method2: SDEIntegrationMethod = "milstein"
        method3: SDEIntegrationMethod = "EM"
        method4: SDEIntegrationMethod = "SEA"

        assert isinstance(method1, str)

    def test_optimization_method_examples(self):
        """Test common OptimizationMethod values."""
        method1: OptimizationMethod = "SLSQP"
        method2: OptimizationMethod = "L-BFGS-B"
        method3: OptimizationMethod = "trust-constr"

        assert isinstance(method1, str)


# ============================================================================
# Test Noise and Stochastic Types
# ============================================================================


# TODO: refactor
class TestNoiseTypes:
    """Test noise and stochastic type definitions."""

    def test_noise_type_valid_values(self):
        """Test NoiseType literal values."""
        noise1: NoiseType = NoiseType.ADDITIVE
        noise2: NoiseType = NoiseType.MULTIPLICATIVE
        noise3: NoiseType = NoiseType.DIAGONAL
        noise4: NoiseType = NoiseType.SCALAR
        noise5: NoiseType = NoiseType.GENERAL

        assert noise1 == NoiseType.ADDITIVE
        assert noise2 == NoiseType.MULTIPLICATIVE

    def test_sde_type_valid_values(self):
        """Test SDEType literal values."""
        sde1: SDEType = SDEType.ITO
        sde2: SDEType = SDEType.STRATONOVICH

        assert sde1 == SDEType.ITO
        assert sde2 == SDEType.STRATONOVICH

    def test_convergence_type_valid_values(self):
        """Test ConvergenceType literal values."""
        conv1: ConvergenceType = ConvergenceType.STRONG
        conv2: ConvergenceType = ConvergenceType.WEAK

        assert conv1 == ConvergenceType.STRONG
        assert conv2 == ConvergenceType.WEAK


# ============================================================================
# Test Configuration TypedDicts
# ============================================================================


class TestConfigurationTypes:
    """Test configuration TypedDict structures."""

    def test_system_config(self):
        """Test SystemConfig TypedDict."""
        config: SystemConfig = {
            "name": "Pendulum",
            "class_name": "InvertedPendulum",
            "nx": 2,
            "nu": 1,
            "ny": 2,
            "is_discrete": False,
            "is_stochastic": False,
            "is_autonomous": False,
            "backend": "numpy",
            "device": "cpu",
        }

        assert config["nx"] == 2
        assert config["nu"] == 1
        assert config["backend"] == "numpy"

    def test_integrator_config(self):
        """Test IntegratorConfig TypedDict."""
        config: IntegratorConfig = {
            "method": "RK45",
            "rtol": 1e-6,
            "atol": 1e-9,
            "max_step": 0.1,
            "vectorized": True,
            "dense_output": False,
        }

        assert config["method"] == "RK45"
        assert config["rtol"] == 1e-6
        assert config["atol"] == 1e-9

    def test_discretizer_config(self):
        """Test DiscretizerConfig TypedDict."""
        config: DiscretizerConfig = {
            "dt": 0.01,
            "method": "exact",
            "backend": "numpy",
            "preserve_stability": True,
        }

        assert config["dt"] == 0.01
        assert config["method"] == "exact"
        assert config["preserve_stability"] is True

    def test_sde_integrator_config(self):
        """Test SDEIntegratorConfig TypedDict."""
        config: SDEIntegratorConfig = {
            "method": "milstein",
            "dt": 0.01,
            "convergence_type": ConvergenceType.STRONG,
            "backend": "torch",
            "seed": 42,
        }

        assert config["method"] == "milstein"
        assert config["convergence_type"] == ConvergenceType.STRONG
        assert config["seed"] == 42


# ============================================================================
# Test Constants
# ============================================================================


class TestConstants:
    """Test constant values and tuples."""

    def test_valid_backends_tuple(self):
        """Test VALID_BACKENDS contains expected values."""
        assert "numpy" in VALID_BACKENDS
        assert "torch" in VALID_BACKENDS
        assert "jax" in VALID_BACKENDS
        assert len(VALID_BACKENDS) == 3

    def test_valid_devices_tuple(self):
        """Test VALID_DEVICES contains common devices."""
        assert "cpu" in VALID_DEVICES
        assert "cuda" in VALID_DEVICES

    def test_default_backend(self):
        """Test default backend is numpy."""
        assert DEFAULT_BACKEND == "numpy"

    def test_default_device(self):
        """Test default device is cpu."""
        assert DEFAULT_DEVICE == "cpu"

    def test_default_dtype(self):
        """Test default dtype is float64."""
        assert DEFAULT_DTYPE == np.float64


# ============================================================================
# Test Utility Functions
# ============================================================================


class TestUtilityFunctions:
    """Test backend utility functions."""

    def test_get_backend_default_method_deterministic(self):
        """Test default method for deterministic systems."""
        assert get_backend_default_method("numpy", is_stochastic=False) == "RK45"
        assert get_backend_default_method("torch", is_stochastic=False) == "rk4"
        assert get_backend_default_method("jax", is_stochastic=False) == "rk4"

    def test_get_backend_default_method_stochastic(self):
        """Test default method for stochastic systems."""
        assert get_backend_default_method("numpy", is_stochastic=True) == "EM"
        assert get_backend_default_method("torch", is_stochastic=True) == "euler"
        assert get_backend_default_method("jax", is_stochastic=True) == "Euler"

    def test_validate_backend_valid(self):
        """Test validate_backend accepts valid backends."""
        assert validate_backend("numpy") == "numpy"
        assert validate_backend("torch") == "torch"
        assert validate_backend("jax") == "jax"

    def test_validate_backend_invalid(self):
        """Test validate_backend rejects invalid backends."""
        with pytest.raises(ValueError, match="Invalid backend"):
            validate_backend("pytorch")

        with pytest.raises(ValueError, match="Invalid backend"):
            validate_backend("tensorflow")

    def test_validate_device_cpu_always_valid(self):
        """Test CPU device is valid for all backends."""
        assert validate_device("cpu", "numpy") == "cpu"
        assert validate_device("cpu", "torch") == "cpu"
        assert validate_device("cpu", "jax") == "cpu"

    def test_validate_device_cuda_requires_gpu_backend(self):
        """Test CUDA requires torch or jax."""
        # Valid
        assert validate_device("cuda", "torch") == "cuda"
        assert validate_device("cuda:0", "torch") == "cuda:0"
        assert validate_device("cuda", "jax") == "cuda"

        # Invalid - NumPy is CPU-only
        with pytest.raises(ValueError, match="NumPy backend only supports CPU"):
            validate_device("cuda", "numpy")

    def test_validate_device_mps_requires_torch(self):
        """Test MPS requires PyTorch."""
        # Valid
        assert validate_device("mps", "torch") == "mps"

        # Invalid
        with pytest.raises(ValueError, match="MPS device requires torch"):
            validate_device("mps", "jax")

        with pytest.raises(ValueError, match="NumPy backend only supports CPU"):
            validate_device("mps", "numpy")


# ============================================================================
# Test Configuration Validation
# ============================================================================


class TestConfigurationValidation:
    """Test configuration dictionary validation."""

    def test_integrator_config_complete(self):
        """Test complete IntegratorConfig."""
        config: IntegratorConfig = {
            "method": "RK45",
            "rtol": 1e-6,
            "atol": 1e-9,
            "max_step": 0.1,
            "first_step": 0.001,
            "vectorized": True,
            "dense_output": False,
        }

        # All fields present
        assert "method" in config
        assert "rtol" in config
        assert "atol" in config

    def test_integrator_config_partial(self):
        """Test partial IntegratorConfig (total=False)."""
        config: IntegratorConfig = {"method": "RK45"}

        # Only method specified
        assert config["method"] == "RK45"
        assert "rtol" not in config

    def test_discretizer_config_typical(self):
        """Test typical DiscretizerConfig usage."""
        config: DiscretizerConfig = {
            "dt": 0.01,
            "method": "exact",
            "backend": "numpy",
        }

        assert config["dt"] == 0.01
        assert config["method"] == "exact"

    def test_sde_integrator_config_with_seed(self):
        """Test SDEIntegratorConfig with reproducibility."""
        config: SDEIntegratorConfig = {
            "method": "euler",
            "dt": 0.01,
            "convergence_type": ConvergenceType.STRONG,
            "backend": "torch",
            "seed": 42,
        }

        assert config["seed"] == 42
        assert config["convergence_type"] == ConvergenceType.STRONG


# ============================================================================
# Test Realistic Usage Patterns
# ============================================================================


class TestRealisticUsage:
    """Test types in realistic scenarios."""

    def test_backend_selection_pattern(self):
        """Test backend selection workflow."""
        # User selects backend
        backend: Backend = "torch"
        device: Device = "cuda:0"

        # Create config
        config: BackendConfig = {"backend": backend, "device": device, "dtype": "float32"}

        assert config["backend"] == "torch"
        assert config["device"] == "cuda:0"

    def test_auto_select_defaults(self):
        """Test automatic default selection."""
        # For numpy backend, deterministic
        method1 = get_backend_default_method("numpy", is_stochastic=False)
        assert method1 == "RK45"

        # For torch backend, stochastic
        method2 = get_backend_default_method("torch", is_stochastic=True)
        assert method2 == "euler"


# ============================================================================
# Test Backend/Device Compatibility
# ============================================================================


class TestBackendDeviceCompatibility:
    """Test backend and device compatibility rules."""

    def test_numpy_only_cpu(self):
        """Test NumPy is CPU-only."""
        # Valid
        validate_device("cpu", "numpy")

        # Invalid
        with pytest.raises(ValueError):
            validate_device("cuda", "numpy")

        with pytest.raises(ValueError):
            validate_device("mps", "numpy")

    def test_torch_supports_multiple_devices(self):
        """Test PyTorch supports CPU, CUDA, MPS."""
        # All should be valid (actual availability checked at runtime)
        validate_device("cpu", "torch")
        validate_device("cuda", "torch")
        validate_device("cuda:0", "torch")
        validate_device("mps", "torch")

    def test_jax_supports_cpu_cuda(self):
        """Test JAX supports CPU and CUDA."""
        validate_device("cpu", "jax")
        validate_device("cuda", "jax")

        # MPS not supported by JAX
        with pytest.raises(ValueError):
            validate_device("mps", "jax")


# ============================================================================
# Test Edge Cases
# ============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_config_dict(self):
        """Test empty configuration is valid."""
        config: BackendConfig = {}
        assert isinstance(config, dict)

    def test_system_config_minimal(self):
        """Test minimal SystemConfig."""
        config: SystemConfig = {
            "nx": 1,
            "nu": 0,  # Autonomous
        }

        assert config["nx"] == 1
        assert config["nu"] == 0

    def test_discretizer_config_minimal(self):
        """Test minimal DiscretizerConfig."""
        config: DiscretizerConfig = {
            "dt": 0.01,
        }

        assert config["dt"] == 0.01

    def test_method_string_arbitrary(self):
        """Test method types accept arbitrary strings (at type level)."""
        # These are just strings, validation happens at runtime
        method: IntegrationMethod = "custom_method"
        assert isinstance(method, str)


# ============================================================================
# Test Documentation Examples
# ============================================================================


class TestDocumentationExamples:
    """Test examples from docstrings work correctly."""

    def test_backend_example(self):
        """Test Backend docstring example."""
        backend: Backend = "torch"
        assert backend == "torch"

    def test_device_example(self):
        """Test Device docstring example."""
        device: Device = "cuda:0"
        assert device == "cuda:0"

    def test_backend_config_example(self):
        """Test BackendConfig docstring example."""
        config: BackendConfig = {"backend": "torch", "device": "cuda:0", "dtype": "float32"}

        assert config["backend"] == "torch"

    def test_integrator_config_example(self):
        """Test IntegratorConfig docstring example."""
        config: IntegratorConfig = {"method": "RK45", "rtol": 1e-6, "atol": 1e-9, "max_step": 0.1}

        assert config["rtol"] == 1e-6

    def test_noise_type_conditional_example(self):
        """Test NoiseType conditional usage example."""
        noise_type: NoiseType = NoiseType.ADDITIVE

        if noise_type == NoiseType.ADDITIVE:
            # Optimization available
            can_optimize = True
        else:
            can_optimize = False

        assert can_optimize is True


# ============================================================================
# Test Type Consistency
# ============================================================================


class TestTypeConsistency:
    """Test type consistency across module."""

    def test_backend_in_valid_backends(self):
        """Test Backend values are in VALID_BACKENDS."""
        backends: list[Backend] = ["numpy", "torch", "jax"]

        for backend in backends:
            assert backend in VALID_BACKENDS

    def test_default_backend_is_valid(self):
        """Test DEFAULT_BACKEND is in VALID_BACKENDS."""
        assert DEFAULT_BACKEND in VALID_BACKENDS

    def test_default_device_is_valid(self):
        """Test DEFAULT_DEVICE is in VALID_DEVICES."""
        assert DEFAULT_DEVICE in VALID_DEVICES


# ============================================================================
# Test Integration Patterns
# ============================================================================


class TestIntegrationPatterns:
    """Test realistic integration patterns."""

    def test_production_configuration(self):
        """Test production-ready configuration."""
        # Production: NumPy, high accuracy
        backend: Backend = "numpy"

        integrator_config: IntegratorConfig = {
            "method": "DOP853",
            "rtol": 1e-10,
            "atol": 1e-12,
        }

        assert integrator_config["rtol"] <= 1e-6

    def test_gpu_acceleration_configuration(self):
        """Test GPU acceleration setup."""
        # GPU: PyTorch with CUDA
        backend: Backend = "torch"
        device: Device = "cuda:0"

        validate_device(device, backend)

        config: BackendConfig = {
            "backend": backend,
            "device": device,
            "dtype": "float32",  # Faster on GPU
        }

        assert config["backend"] == "torch"
        assert config["device"] == "cuda:0"

    def test_stochastic_simulation_configuration(self):
        """Test stochastic simulation setup."""
        backend: Backend = "numpy"

        sde_config: SDEIntegratorConfig = {
            "method": "EM",
            "dt": 0.01,
            "convergence_type": ConvergenceType.STRONG,
            "backend": backend,
            "seed": 42,
        }

        assert sde_config["seed"] == 42
        assert sde_config["convergence_type"] == ConvergenceType.STRONG

    def test_discretization_for_control_design(self):
        """Test discretization config for controller design."""
        # Control design: exact discretization
        config: DiscretizerConfig = {
            "dt": 0.01,
            "method": "exact",
            "backend": "numpy",
            "preserve_stability": True,
        }

        assert config["method"] == "exact"
        assert config["preserve_stability"] is True


# ============================================================================
# Test Error Messages
# ============================================================================


class TestErrorMessages:
    """Test that error messages are informative."""

    def test_invalid_backend_error_message(self):
        """Test validate_backend gives helpful error."""
        with pytest.raises(ValueError) as exc_info:
            validate_backend("invalid_backend")

        assert "Invalid backend" in str(exc_info.value)
        assert "numpy" in str(exc_info.value)  # Shows valid options

    def test_invalid_device_error_message(self):
        """Test validate_device gives helpful error."""
        with pytest.raises(ValueError) as exc_info:
            validate_device("cuda", "numpy")

        assert "NumPy" in str(exc_info.value)
        assert "CPU" in str(exc_info.value)


# ============================================================================
# Test Constants Immutability
# ============================================================================


class TestConstantsImmutability:
    """Test that constants are properly defined."""

    def test_valid_backends_is_tuple(self):
        """Test VALID_BACKENDS is immutable tuple."""
        assert isinstance(VALID_BACKENDS, tuple)
        # Tuples are immutable
        with pytest.raises(TypeError):
            VALID_BACKENDS[0] = "other"  # type: ignore


# ============================================================================
# Test Default Values
# ============================================================================


class TestDefaultValues:
    """Test default constant values are sensible."""

    def test_default_backend_is_universal(self):
        """Test default backend works everywhere."""
        assert DEFAULT_BACKEND == "numpy"
        # NumPy is always available (core dependency)

    def test_default_device_is_universal(self):
        """Test default device works everywhere."""
        assert DEFAULT_DEVICE == "cpu"
        # CPU is always available

    def test_default_dtype_is_precise(self):
        """Test default dtype is double precision."""
        assert DEFAULT_DTYPE == np.float64
        # Control/scientific computing needs precision


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
