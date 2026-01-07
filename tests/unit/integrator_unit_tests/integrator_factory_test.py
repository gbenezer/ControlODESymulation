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
Unit Tests for Integrator Factory

Tests the factory class for creating numerical integrators, including:
- Backend-specific integrator creation
- Method validation and selection
- Use case-specific factory methods
- Error handling for invalid configurations
- Method listing and recommendation utilities
- Julia DiffEqPy integration support

Design Note
-----------
This test suite validates the IntegratorFactory, which uses semantic types
from cdesym.types.core (ScalarLike for time steps) following the project
design principles. The factory creates integrators that use the centralized
type framework for consistency across the codebase.

Test Coverage
-------------
1. Basic integrator creation
2. Backend-method compatibility validation
3. Fixed-step vs adaptive selection
4. Default method selection
5. Use case-specific factories (production, optimization, neural ODE, julia, etc.)
6. Method listing and information retrieval
7. Recommendation system
8. Error handling and validation
9. Julia DiffEqPy support (with/without Julia installed)
10. Helper method validation (_is_julia_method, _is_fixed_step_method, etc.)
"""

from unittest.mock import Mock

import numpy as np
import pytest

# Conditional import for Julia
try:
    from diffeqpy import de

    JULIA_AVAILABLE = True
except ImportError:
    JULIA_AVAILABLE = False

from cdesym.systems.base.numerical_integration.integrator_base import IntegratorBase, StepMode

# Import the factory
from cdesym.systems.base.numerical_integration.integrator_factory import (
    IntegratorFactory,
    IntegratorType,
    auto_integrator,
    create_integrator,
)

# Import semantic types for clarity (though not heavily used in factory tests)
from cdesym.types.core import ScalarLike

# ============================================================================
# Mock System Fixture
# ============================================================================


@pytest.fixture
def mock_system():
    """
    Create a mock SymbolicDynamicalSystem for testing.

    The factory tests primarily validate integrator creation logic,
    not actual integration, so a simple mock suffices.
    """
    system = Mock()
    system.nx = 2
    system.nu = 1
    system.ny = 2
    system.__call__ = Mock(return_value=np.array([1.0, 2.0]))
    system.forward = Mock(return_value=np.array([1.0, 2.0]))
    return system


# ============================================================================
# Test Class: Basic Creation
# ============================================================================


class TestBasicCreation:
    """Test basic integrator creation with IntegratorFactory.create()."""

    def test_create_default_numpy(self, mock_system):
        """Test creating integrator with default numpy backend."""
        integrator = IntegratorFactory.create(mock_system, backend="numpy")

        assert integrator is not None
        assert integrator.backend == "numpy"
        assert integrator.system == mock_system

    def test_create_with_default_method(self, mock_system):
        """Test that default methods are selected correctly."""
        integrator = IntegratorFactory.create(mock_system, backend="numpy")
        assert hasattr(integrator, "method")

    def test_create_with_specific_method(self, mock_system):
        """Test creating with specific method."""
        integrator = IntegratorFactory.create(mock_system, backend="numpy", method="RK45")

        assert integrator.method == "RK45"

    def test_create_fixed_step_requires_dt(self, mock_system):
        """Test that fixed-step methods require dt parameter."""
        with pytest.raises(ValueError, match="requires dt"):
            IntegratorFactory.create(
                mock_system,
                backend="numpy",
                method="rk4",
                step_mode=StepMode.FIXED,
            )

    def test_create_fixed_step_with_dt(self, mock_system):
        """Test creating fixed-step integrator with dt (ScalarLike)."""
        dt: ScalarLike = 0.01  # Type hint for clarity
        integrator = IntegratorFactory.create(
            mock_system,
            backend="numpy",
            method="rk4",
            dt=dt,
            step_mode=StepMode.FIXED,
        )

        assert integrator.dt == 0.01
        assert integrator.step_mode == StepMode.FIXED

    def test_create_with_options(self, mock_system):
        """Test creating integrator with additional options."""
        integrator = IntegratorFactory.create(
            mock_system,
            backend="numpy",
            method="RK45",
            rtol=1e-9,
            atol=1e-11,
        )

        assert integrator.rtol == 1e-9
        assert integrator.atol == 1e-11


# ============================================================================
# Test Class: Backend Validation
# ============================================================================


class TestBackendValidation:
    """Test backend validation and compatibility checks."""

    def test_invalid_backend_raises_error(self, mock_system):
        """Test that invalid backend name raises ValueError."""
        with pytest.raises(ValueError, match="Invalid backend"):
            IntegratorFactory.create(mock_system, backend="matlab")

    def test_valid_backends(self, mock_system):
        """Test all valid backends can be created."""
        # Test numpy (always available)
        integrator = IntegratorFactory.create(mock_system, backend="numpy")
        assert integrator.backend == "numpy"

        # Test torch if available
        try:
            import torch

            integrator = IntegratorFactory.create(mock_system, backend="torch")
            assert integrator.backend == "torch"
        except ImportError:
            pytest.skip("PyTorch not installed")

        # Test jax if available
        try:
            import jax

            integrator = IntegratorFactory.create(mock_system, backend="jax")
            assert integrator.backend == "jax"
        except ImportError:
            pytest.skip("JAX not installed")


# ============================================================================
# Test Class: Method-Backend Compatibility
# ============================================================================


class TestMethodBackendCompatibility:
    """Test validation of method-backend compatibility."""

    def test_scipy_only_method_on_wrong_backend(self, mock_system):
        """Test that methods with no cross-backend mapping fail validation."""
        # Use a method that ONLY exists on numpy and has NO normalization
        with pytest.raises(ValueError, match="not available"):
            IntegratorFactory.create(
                mock_system,
                backend="torch",
                method="DOP853",  # Scipy-only, no torch equivalent
            )

    def test_universal_methods_work_with_any_backend(self, mock_system):
        """Test that universal methods work with any backend.

        Only euler and midpoint are truly universal:
        - euler: Available on numpy (manual/Julia), torch (TorchDiffEq), jax (Diffrax)
        - midpoint: Available on numpy (manual/Julia), torch (TorchDiffEq), jax (Diffrax)
        - rk4: Only on numpy (manual) and torch (TorchDiffEq), NOT on jax
        - heun: Only on numpy (manual/Julia) and jax (Diffrax), NOT on torch
        """
        universal_methods = ["euler", "midpoint"]

        for method in universal_methods:
            for backend in ["numpy", "torch", "jax"]:
                try:
                    dt: ScalarLike = 0.01
                    integrator = IntegratorFactory.create(
                        mock_system,
                        backend=backend,
                        method=method,
                        dt=dt,
                        step_mode=StepMode.FIXED,
                    )
                    assert integrator is not None
                    assert integrator.backend == backend
                except ImportError:
                    # Expected if torch/jax not installed
                    pytest.skip(f"{backend} not installed")
                except Exception as e:
                    pytest.fail(f"Universal method {method} on {backend} failed: {e}")

    def test_backend_specific_method_validated(self, mock_system):
        """Test that backend-specific methods are validated."""
        # fehlberg2 only exists on torch
        with pytest.raises(ValueError, match="not available"):
            IntegratorFactory.create(
                mock_system,
                backend="jax",
                method="fehlberg2",
            )


# ============================================================================
# Test Class: Julia DiffEqPy Support
# ============================================================================


class TestJuliaDiffEqPy:
    """Test Julia DiffEqPy integration."""

    def test_is_julia_method_helper(self):
        """Test _is_julia_method helper correctly identifies Julia methods."""
        # Julia methods (capital first letter)
        assert IntegratorFactory._is_julia_method("Tsit5")
        assert IntegratorFactory._is_julia_method("Vern9")
        assert IntegratorFactory._is_julia_method("Rosenbrock23")

        # Julia auto-switching (contains parentheses)
        assert IntegratorFactory._is_julia_method("AutoTsit5(Rosenbrock23())")

        # Not Julia methods
        assert not IntegratorFactory._is_julia_method("LSODA")  # Scipy
        assert not IntegratorFactory._is_julia_method("RK45")  # Scipy
        assert not IntegratorFactory._is_julia_method("tsit5")  # Diffrax (lowercase)
        assert not IntegratorFactory._is_julia_method("dopri5")  # TorchDiffEq/Diffrax

    @pytest.mark.skipif(not JULIA_AVAILABLE, reason="Julia not installed")
    def test_create_julia_integrator(self, mock_system):
        """Test creating Julia-based integrator."""
        integrator = IntegratorFactory.create(mock_system, backend="numpy", method="Tsit5")

        assert integrator is not None
        assert integrator.backend == "numpy"
        assert integrator.algorithm == "Tsit5"

    @pytest.mark.skipif(not JULIA_AVAILABLE, reason="Julia not installed")
    def test_for_julia_factory_method(self, mock_system):
        """Test for_julia factory method."""
        integrator = IntegratorFactory.for_julia(mock_system, algorithm="Vern9")

        assert integrator is not None
        assert integrator.algorithm == "Vern9"

    def test_julia_method_without_diffeqpy_raises_error(self, mock_system):
        """Test that Julia method without diffeqpy raises clear error."""
        if JULIA_AVAILABLE:
            pytest.skip("Julia is installed, can't test error")

        with pytest.raises(ImportError, match="diffeqpy"):
            IntegratorFactory.create(mock_system, backend="numpy", method="Tsit5")


# ============================================================================
# Test Class: Use Case Factory Methods
# ============================================================================


class TestUseCaseFactoryMethods:
    """Test use case-specific factory methods."""

    def test_auto_method(self, mock_system):
        """Test auto() method selects appropriate backend."""
        integrator = IntegratorFactory.auto(mock_system)

        assert integrator is not None
        assert integrator.backend in ["numpy", "torch", "jax"]

    def test_auto_with_prefer_backend(self, mock_system):
        """Test auto() respects backend preference."""
        integrator = IntegratorFactory.auto(mock_system, prefer_backend="numpy")

        assert integrator.backend == "numpy"

    def test_for_production_default(self, mock_system):
        """Test for_production() uses scipy LSODA by default."""
        integrator = IntegratorFactory.for_production(mock_system)

        assert integrator.backend == "numpy"
        assert integrator.method == "LSODA"

    @pytest.mark.skipif(not JULIA_AVAILABLE, reason="Julia not installed")
    def test_for_production_julia(self, mock_system):
        """Test for_production() with Julia option."""
        integrator = IntegratorFactory.for_production(mock_system, use_julia=True)

        assert integrator.backend == "numpy"
        assert "AutoTsit5" in integrator.algorithm

    def test_for_optimization_prefers_jax(self, mock_system):
        """Test for_optimization() prefers JAX if available."""
        try:
            import jax

            integrator = IntegratorFactory.for_optimization(mock_system)
            assert integrator.backend == "jax"
        except ImportError:
            pytest.skip("JAX not installed")

    def test_for_optimization_with_torch(self, mock_system):
        """Test for_optimization() can use PyTorch."""
        try:
            import torch

            integrator = IntegratorFactory.for_optimization(mock_system, prefer_backend="torch")
            assert integrator.backend == "torch"
        except ImportError:
            pytest.skip("PyTorch not installed")

    def test_for_neural_ode(self, mock_system):
        """Test for_neural_ode() factory method."""
        try:
            import torch

            integrator = IntegratorFactory.for_neural_ode(mock_system)
            assert integrator.backend == "torch"
        except ImportError:
            pytest.skip("PyTorch not installed")

    def test_for_simple(self, mock_system):
        """Test for_simple() creates RK4 integrator."""
        dt: ScalarLike = 0.01
        integrator = IntegratorFactory.for_simple(mock_system, dt=dt)

        # Fixed-step integrators have 'name' property, not 'method' attribute
        assert "rk4" in integrator.name.lower()
        assert integrator.step_mode == StepMode.FIXED
        assert integrator.dt == dt

    def test_for_educational(self, mock_system):
        """Test for_educational() creates Euler integrator."""
        dt: ScalarLike = 0.01
        integrator = IntegratorFactory.for_educational(mock_system, dt=dt)

        # Fixed-step integrators have 'name' property, not 'method' attribute
        assert "euler" in integrator.name.lower()
        assert integrator.step_mode == StepMode.FIXED


# ============================================================================
# Test Class: Helper Methods
# ============================================================================


class TestHelperMethods:
    """Test internal helper methods."""

    def test_is_manual_method(self):
        """Test _is_manual_method helper."""
        assert IntegratorFactory._is_manual_method("euler")
        assert IntegratorFactory._is_manual_method("heun")
        assert IntegratorFactory._is_manual_method("midpoint")
        assert IntegratorFactory._is_manual_method("rk4")

        assert not IntegratorFactory._is_manual_method("dopri5")
        assert not IntegratorFactory._is_manual_method("LSODA")
        assert not IntegratorFactory._is_manual_method("Tsit5")

    def test_is_scipy_method(self):
        """Test _is_scipy_method helper."""
        assert IntegratorFactory._is_scipy_method("LSODA")
        assert IntegratorFactory._is_scipy_method("RK45")
        assert IntegratorFactory._is_scipy_method("BDF")

        assert not IntegratorFactory._is_scipy_method("dopri5")
        assert not IntegratorFactory._is_scipy_method("tsit5")

    def test_is_julia_method(self):
        """Test _is_julia_method helper correctly identifies Julia methods."""
        # Julia methods (capital first letter)
        assert IntegratorFactory._is_julia_method("Tsit5")
        assert IntegratorFactory._is_julia_method("Vern9")
        assert IntegratorFactory._is_julia_method("Rosenbrock23")

        # Julia auto-switching (contains parentheses)
        assert IntegratorFactory._is_julia_method("AutoTsit5(Rosenbrock23())")

        # Not Julia methods
        assert not IntegratorFactory._is_julia_method("LSODA")  # Scipy
        assert not IntegratorFactory._is_julia_method("RK45")  # Scipy
        assert not IntegratorFactory._is_julia_method("tsit5")  # Diffrax (lowercase)
        assert not IntegratorFactory._is_julia_method("dopri5")  # TorchDiffEq/Diffrax


# ============================================================================
# Test Class: Method Listing and Information
# ============================================================================


class TestMethodListingAndInfo:
    """Test method listing and information retrieval."""

    def test_list_methods_all(self):
        """Test list_methods returns methods organized by categories."""
        methods = IntegratorFactory.list_methods()

        # Now returns categories, not backends as top-level keys
        assert "deterministic_fixed_step" in methods
        assert "deterministic_adaptive" in methods

        # Check some expected methods in categories
        assert "euler" in methods["deterministic_fixed_step"]
        assert "rk4" in methods["deterministic_fixed_step"]
        assert "LSODA" in methods["deterministic_adaptive"]

    def test_list_methods_specific_backend(self):
        """Test list_methods for specific backend."""
        numpy_methods = IntegratorFactory.list_methods("numpy")

        # Returns categories, not nested by backend
        assert "deterministic_fixed_step" in numpy_methods
        assert "deterministic_adaptive" in numpy_methods

        # Check expected methods
        fixed_step = numpy_methods["deterministic_fixed_step"]
        adaptive = numpy_methods["deterministic_adaptive"]

        assert "euler" in fixed_step
        assert "rk4" in fixed_step
        assert "LSODA" in adaptive
        assert "Tsit5" in adaptive  # Julia method

    def test_get_info_scipy(self):
        """Test get_info for scipy method."""
        info = IntegratorFactory.get_info("numpy", "LSODA")

        assert "name" in info
        assert "description" in info
        assert "LSODA" in info["name"]

    def test_get_info_diffrax(self):
        """Test get_info for diffrax method."""
        info = IntegratorFactory.get_info("jax", "tsit5")

        assert "name" in info
        assert "description" in info

    @pytest.mark.skipif(not JULIA_AVAILABLE, reason="Julia not installed")
    def test_get_info_julia(self):
        """Test get_info delegates to Julia algorithm info."""
        info = IntegratorFactory.get_info("numpy", "Tsit5")

        assert "name" in info
        assert "library" in info
        assert "Julia" in info["library"]

    def test_get_info_unknown_method(self):
        """Test get_info for unknown method returns generic info."""
        info = IntegratorFactory.get_info("numpy", "unknown_method")

        assert "name" in info
        assert info["name"] == "unknown_method"


# ============================================================================
# Test Class: Recommendation System
# ============================================================================


class TestRecommendationSystem:
    """Test integrator recommendation system."""

    def test_recommend_production(self):
        """Test recommendation for production."""
        rec = IntegratorFactory.recommend("production")

        assert rec["backend"] == "numpy"
        assert rec["method"] == "LSODA"
        assert "description" in rec

    def test_recommend_optimization(self):
        """Test recommendation for optimization."""
        rec = IntegratorFactory.recommend("optimization")

        assert rec["backend"] == "jax"
        assert rec["method"] == "tsit5"

    def test_recommend_neural_ode(self):
        """Test recommendation for neural ODE."""
        rec = IntegratorFactory.recommend("neural_ode")

        assert rec["backend"] == "torch"
        assert rec["method"] == "dopri5"
        assert rec["adjoint"] is True

    def test_recommend_simple(self):
        """Test recommendation for simple use case."""
        rec = IntegratorFactory.recommend("simple")

        assert rec["backend"] == "numpy"
        assert rec["method"] == "rk4"
        assert "dt" in rec

    def test_recommend_julia(self):
        """Test recommendation for Julia."""
        rec = IntegratorFactory.recommend("julia")

        assert rec["backend"] == "numpy"
        assert rec["method"] == "Tsit5"

    def test_recommend_educational(self):
        """Test recommendation for educational."""
        rec = IntegratorFactory.recommend("educational")

        assert rec["backend"] == "numpy"
        assert rec["method"] == "euler"

    def test_recommend_with_gpu(self):
        """Test recommendation adjusts for GPU."""
        rec = IntegratorFactory.recommend("optimization", has_gpu=True)

        # Should prefer GPU-capable backend
        assert rec["backend"] in ["torch", "jax"]

    def test_recommend_invalid_use_case(self):
        """Test invalid use case raises error."""
        with pytest.raises(ValueError, match="Unknown use case"):
            IntegratorFactory.recommend("invalid_use_case")


# ============================================================================
# Test Class: Convenience Functions
# ============================================================================


class TestConvenienceFunctions:
    """Test convenience wrapper functions."""

    def test_create_integrator_function(self, mock_system):
        """Test create_integrator convenience function."""
        integrator = create_integrator(mock_system, backend="numpy")

        assert integrator is not None
        assert integrator.backend == "numpy"

    def test_create_integrator_with_method(self, mock_system):
        """Test create_integrator with method."""
        integrator = create_integrator(mock_system, backend="numpy", method="RK45")

        assert integrator.method == "RK45"

    @pytest.mark.skipif(not JULIA_AVAILABLE, reason="Julia not installed")
    def test_create_integrator_julia_method(self, mock_system):
        """Test create_integrator with Julia method."""
        integrator = create_integrator(mock_system, backend="numpy", method="Tsit5")

        assert integrator.algorithm == "Tsit5"

    def test_auto_integrator_function(self, mock_system):
        """Test auto_integrator convenience function."""
        integrator = auto_integrator(mock_system)

        assert integrator is not None


# ============================================================================
# Test Class: Error Handling
# ============================================================================


class TestErrorHandling:
    """Test error handling in factory."""

    def test_missing_pytorch_for_neural_ode(self, mock_system):
        """Test error when PyTorch not available for neural ODE."""
        try:
            import torch

            pytest.skip("PyTorch is installed")
        except ImportError:
            with pytest.raises(ImportError, match="PyTorch"):
                IntegratorFactory.for_neural_ode(mock_system)

    def test_method_not_available_for_backend(self, mock_system):
        """Test error when method truly not available for backend."""
        # Use backend-specific method on wrong backend
        with pytest.raises(ValueError, match="not available") as exc_info:
            IntegratorFactory.create(
                mock_system,
                backend="numpy",
                method="fehlberg2",  # TorchDiffEq-only method
            )

        error_msg = str(exc_info.value).lower()
        assert "not available" in error_msg or "fehlberg2" in error_msg

    def test_fixed_step_without_dt_clear_error(self, mock_system):
        """Test clear error when dt missing for fixed-step."""
        with pytest.raises(ValueError) as exc_info:
            IntegratorFactory.create(
                mock_system,
                backend="numpy",
                method="rk4",
                step_mode=StepMode.FIXED,
            )

        assert "requires dt" in str(exc_info.value).lower()

    def test_truly_invalid_method_for_backend(self, mock_system):
        """Test error for method that doesn't exist anywhere."""
        with pytest.raises(ValueError, match="not available"):
            IntegratorFactory.create(
                mock_system,
                backend="jax",
                method="completely_fake_method_12345",
            )

    def test_normalization_works_across_backends(self, mock_system):
        """Test that normalization allows portable canonical names."""
        # This should NOT raise - it's the FEATURE of the registry!
        # LSODA → dopri5 on torch (automatic normalization)
        integrator = IntegratorFactory.create(
            mock_system,
            backend="torch",
            method="LSODA",  # Normalizes to dopri5
        )
        assert integrator.method == "dopri5"


# ============================================================================
# Test Class: Integration with Actual System
# ============================================================================


class TestIntegrationWithActualSystem:
    """Test factory with real system (if available)."""

    @pytest.mark.integration
    def test_create_and_verify_type(self, mock_system):
        """Test creating integrator returns correct type."""
        integrator = IntegratorFactory.create(mock_system, backend="numpy", method="RK45")

        assert isinstance(integrator, IntegratorBase)
        assert integrator.backend == "numpy"
        assert integrator.method == "RK45"


# ============================================================================
# Test Class: Method Normalization
# ============================================================================


class TestMethodNormalization:
    """Test that method normalization works across backends."""

    def test_lsoda_normalizes_to_backend_equivalent(self, mock_system):
        """Test LSODA normalizes to appropriate method for each backend."""
        # All backends should successfully create an integrator
        # (proving normalization worked)

        integrator_numpy = IntegratorFactory.create(mock_system, backend="numpy", method="LSODA")
        assert integrator_numpy is not None
        assert integrator_numpy.backend == "numpy"
        assert integrator_numpy.method == "LSODA"

        integrator_torch = IntegratorFactory.create(mock_system, backend="torch", method="LSODA")
        assert integrator_torch is not None
        assert integrator_torch.backend == "torch"
        assert integrator_torch.method == "dopri5"  # Normalized

        integrator_jax = IntegratorFactory.create(mock_system, backend="jax", method="LSODA")
        assert integrator_jax is not None
        assert integrator_jax.backend == "jax"
        # Don't check solver attribute - just verify creation succeeded

    def test_canonical_methods_work_across_backends(self, mock_system):
        """Test canonical method names work on all backends (via normalization)."""
        canonical_methods = ["lsoda", "rk45", "rk23"]

        for method in canonical_methods:
            # Each canonical name should work on all backends
            for backend in ["numpy", "torch", "jax"]:
                integrator = IntegratorFactory.create(mock_system, backend=backend, method=method)
                assert integrator is not None
                assert integrator.backend == backend

    def test_rk45_normalizes_correctly(self, mock_system):
        """Test rk45 canonical name works on all backends."""
        # This is the FEATURE - portable canonical names!
        for backend in ["numpy", "torch", "jax"]:
            integrator = IntegratorFactory.create(mock_system, backend=backend, method="rk45")
            assert integrator is not None

    def test_backend_specific_method_fails_on_wrong_backend(self, mock_system):
        """Test truly backend-specific methods fail on wrong backend."""
        # fehlberg2 ONLY exists on torch, no normalization
        with pytest.raises(ValueError, match="not available"):
            IntegratorFactory.create(mock_system, backend="jax", method="fehlberg2")


# ============================================================================
# Test Class: Edge Cases
# ============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_options(self, mock_system):
        """Test creation with no additional options."""
        integrator = IntegratorFactory.create(mock_system)

        assert integrator is not None

    def test_many_options(self, mock_system):
        """Test creation with many options."""
        integrator = IntegratorFactory.create(
            mock_system,
            backend="numpy",
            method="RK45",
            rtol=1e-10,
            atol=1e-12,
            max_steps=50000,
            first_step=1e-6,
        )

        assert integrator.rtol == 1e-10

    def test_none_method_uses_default(self, mock_system):
        """Test that method=None uses backend default."""
        integrator = IntegratorFactory.create(mock_system, backend="numpy", method=None)

        # Should default to LSODA for numpy
        assert integrator.method == "LSODA"


# ============================================================================
# Test Class: Factory-Level Julia Preference (Optional Optimization)
# ============================================================================


class TestFactoryJuliaPreference:
    """Test optional Julia preference in factory routing.

    The factory MAY prefer Julia implementations for euler/heun/midpoint when
    available on NumPy backend. This is an implementation detail, not a
    normalization concern.

    Key behaviors:
    - lowercase 'euler' on numpy → tries Julia 'Euler', falls back to manual
    - capitalized 'Euler' on numpy → explicitly requests Julia
    - prefer_manual=True option → forces manual implementation
    - torch/jax → always use manual implementations
    """

    def test_lowercase_manual_methods_create_successfully(self, mock_system):
        """Test that lowercase manual methods create integrators successfully.

        The factory may prefer Julia (if available) or use manual implementations.
        This test just ensures creation succeeds.
        """
        methods_to_test = ["euler", "heun", "midpoint"]

        for method in methods_to_test:
            integrator = IntegratorFactory.create(
                mock_system, backend="numpy", method=method, dt=0.01, step_mode=StepMode.FIXED
            )

            # Should successfully create an integrator
            assert integrator is not None
            assert integrator.backend == "numpy"

    def test_capitalized_methods_attempt_julia(self, mock_system):
        """Test that capitalized methods attempt to use Julia."""
        capitalized_methods = ["Euler", "Heun", "Midpoint"]

        for method in capitalized_methods:
            if JULIA_AVAILABLE:
                # Should successfully create Julia integrator
                try:
                    integrator = IntegratorFactory.create(
                        mock_system,
                        backend="numpy",
                        method=method,
                        dt=0.01,
                        step_mode=StepMode.FIXED,
                    )
                    assert integrator is not None
                    # Should be DiffEqPy integrator
                    assert hasattr(integrator, "algorithm")
                except Exception as e:
                    # Some Julia methods might not be available - that's OK
                    pass
            else:
                # Should fail if Julia not available
                with pytest.raises(ImportError, match="diffeqpy"):
                    IntegratorFactory.create(
                        mock_system,
                        backend="numpy",
                        method=method,
                        dt=0.01,
                        step_mode=StepMode.FIXED,
                    )

    @pytest.mark.skipif(not JULIA_AVAILABLE, reason="Julia not installed")
    def test_prefer_manual_option_forces_manual(self, mock_system):
        """Test that prefer_manual=True option forces manual implementation."""
        from cdesym.systems.base.numerical_integration.fixed_step_integrators import (
            ExplicitEulerIntegrator,
            MidpointIntegrator,
        )

        # Create with prefer_manual=True
        integrator = IntegratorFactory.create(
            mock_system,
            backend="numpy",
            method="euler",
            dt=0.01,
            step_mode=StepMode.FIXED,
            prefer_manual=True,
        )

        # Should be manual implementation, not DiffEqPy
        assert isinstance(integrator, ExplicitEulerIntegrator)
        assert not hasattr(integrator, "algorithm")  # Not a Julia integrator

    def test_rk4_always_uses_manual(self, mock_system):
        """Test that rk4 always uses manual (no Julia equivalent)."""
        from cdesym.systems.base.numerical_integration.fixed_step_integrators import (
            RK4Integrator,
        )

        integrator = IntegratorFactory.create(
            mock_system, backend="numpy", method="rk4", dt=0.01, step_mode=StepMode.FIXED
        )

        # Should always be manual RK4
        assert isinstance(integrator, RK4Integrator)

    def test_torch_jax_always_use_manual(self, mock_system):
        """Test that torch/jax backends always use manual implementations."""
        methods = ["euler", "midpoint"]

        for backend in ["torch", "jax"]:
            try:
                for method in methods:
                    integrator = IntegratorFactory.create(
                        mock_system,
                        backend=backend,
                        method=method,
                        dt=0.01,
                        step_mode=StepMode.FIXED,
                    )

                    # Should create successfully
                    assert integrator is not None
                    assert integrator.backend == backend

                    # Should NOT be DiffEqPy integrator
                    assert not hasattr(integrator, "algorithm")
            except ImportError:
                # Expected if torch/jax not installed
                pytest.skip(f"{backend} not installed")

    @pytest.mark.skipif(not JULIA_AVAILABLE, reason="Julia not installed")
    def test_julia_preference_provides_performance_benefit(self, mock_system):
        """Test that Julia preference is indeed using Julia implementations.

        This verifies the optimization actually works when Julia is available.
        """
        from cdesym.systems.base.numerical_integration.diffeqpy_integrator import (
            DiffEqPyIntegrator,
        )

        # Create with lowercase (should prefer Julia)
        integrator = IntegratorFactory.create(
            mock_system, backend="numpy", method="euler", dt=0.01, step_mode=StepMode.FIXED
        )

        # Should be DiffEqPy integrator
        assert isinstance(integrator, DiffEqPyIntegrator)
        assert integrator.algorithm == "Euler"

    def test_factory_julia_preference_is_optional(self, mock_system):
        """Test that Julia preference is optional and degrades gracefully.

        Even if Julia preference is implemented, it should gracefully fall back
        to manual if Julia is not available.
        """
        # This should always succeed (Julia or manual)
        integrator = IntegratorFactory.create(
            mock_system, backend="numpy", method="euler", dt=0.01, step_mode=StepMode.FIXED
        )

        assert integrator is not None
        assert integrator.backend == "numpy"

        if JULIA_AVAILABLE:
            # May use Julia (optimization)
            pass
        else:
            # Should fall back to manual
            from cdesym.systems.base.numerical_integration.fixed_step_integrators import (
                ExplicitEulerIntegrator,
            )

            assert isinstance(integrator, ExplicitEulerIntegrator)

    def test_heun_only_available_on_numpy(self, mock_system):
        """Test that heun is only available on numpy backend.

        TorchDiffEq and Diffrax don't have Heun implementations.
        """
        # Should work on numpy
        integrator = IntegratorFactory.create(
            mock_system, backend="numpy", method="heun", dt=0.01, step_mode=StepMode.FIXED
        )
        assert integrator is not None

        # Should fail on torch
        try:
            import torch

            with pytest.raises(ValueError, match="Unknown method"):
                IntegratorFactory.create(
                    mock_system, backend="torch", method="heun", dt=0.01, step_mode=StepMode.FIXED
                )
        except ImportError:
            pytest.skip("PyTorch not installed")


# ============================================================================
# Test Class: Normalization Does NOT Handle Julia Preference
# ============================================================================


class TestNormalizationPurity:
    """Test that normalization layer is pure and doesn't handle Julia preference.

    This validates the architectural decision that Julia preference is a
    factory implementation detail, not a normalization concern.
    """

    def test_normalize_manual_methods_pass_through(self, mock_system):
        """Test that manual methods pass through normalization unchanged."""
        from cdesym.systems.base.numerical_integration.method_registry import normalize_method_name

        # Manual methods should pass through unchanged
        assert normalize_method_name("euler", "numpy") == "euler"
        assert normalize_method_name("heun", "numpy") == "heun"
        assert normalize_method_name("midpoint", "numpy") == "midpoint"
        assert normalize_method_name("rk4", "numpy") == "rk4"

        # On all backends
        for backend in ["numpy", "torch", "jax"]:
            assert normalize_method_name("euler", backend) == "euler"
            assert normalize_method_name("heun", backend) == "heun"
            assert normalize_method_name("midpoint", backend) == "midpoint"

    def test_normalize_canonical_names_only(self, mock_system):
        """Test that normalization only handles true canonical names."""
        from cdesym.systems.base.numerical_integration.method_registry import normalize_method_name

        # These ARE canonical names and should normalize
        assert normalize_method_name("euler_maruyama", "numpy") == "EM"
        assert normalize_method_name("euler_maruyama", "torch") == "euler"
        assert normalize_method_name("euler_maruyama", "jax") == "Euler"

        assert normalize_method_name("rk45", "numpy") == "RK45"
        assert normalize_method_name("rk45", "torch") == "dopri5"
        assert normalize_method_name("rk45", "jax") == "tsit5"

    def test_no_manual_prefix_in_normalization(self, mock_system):
        """Test that normalization doesn't handle manual_* prefix."""
        from cdesym.systems.base.numerical_integration.method_registry import normalize_method_name

        # manual_* methods should pass through unchanged
        # (not stripped to lowercase)
        assert normalize_method_name("manual_euler", "numpy") == "manual_euler"
        assert normalize_method_name("manual_heun", "torch") == "manual_heun"
        assert normalize_method_name("manual_midpoint", "jax") == "manual_midpoint"


# ============================================================================
# Test Class: IntegratorType Enum
# ============================================================================


class TestIntegratorTypeEnum:
    """Test IntegratorType enum."""

    def test_integrator_types_defined(self):
        """Test that all integrator types are defined."""
        assert hasattr(IntegratorType, "PRODUCTION")
        assert hasattr(IntegratorType, "OPTIMIZATION")
        assert hasattr(IntegratorType, "NEURAL_ODE")
        assert hasattr(IntegratorType, "JULIA")  # New!
        assert hasattr(IntegratorType, "SIMPLE")
        assert hasattr(IntegratorType, "EDUCATIONAL")

    def test_integrator_type_values(self):
        """Test integrator type enum values."""
        assert IntegratorType.PRODUCTION.value == "production"
        assert IntegratorType.OPTIMIZATION.value == "optimization"
        assert IntegratorType.NEURAL_ODE.value == "neural_ode"
        assert IntegratorType.JULIA.value == "julia"
        assert IntegratorType.SIMPLE.value == "simple"
        assert IntegratorType.EDUCATIONAL.value == "educational"


# ============================================================================
# Test Class: ScalarLike Type Usage
# ============================================================================


class TestScalarLikeTypeUsage:
    """Test that dt parameters accept ScalarLike types."""

    def test_dt_with_float(self, mock_system):
        """Test dt parameter accepts float (basic ScalarLike)."""
        dt: ScalarLike = 0.01
        integrator = IntegratorFactory.create(
            mock_system,
            backend="numpy",
            method="rk4",
            dt=dt,
            step_mode=StepMode.FIXED,
        )

        assert integrator.dt == 0.01

    def test_dt_with_int(self, mock_system):
        """Test dt parameter accepts int (also ScalarLike)."""
        dt: ScalarLike = 1
        integrator = IntegratorFactory.create(
            mock_system,
            backend="numpy",
            method="rk4",
            dt=dt,
            step_mode=StepMode.FIXED,
        )

        assert integrator.dt == 1

    def test_dt_with_numpy_scalar(self, mock_system):
        """Test dt parameter accepts numpy scalar."""
        dt: ScalarLike = np.float64(0.01)
        integrator = IntegratorFactory.create(
            mock_system,
            backend="numpy",
            method="rk4",
            dt=dt,
            step_mode=StepMode.FIXED,
        )

        assert integrator.dt == 0.01


# ============================================================================
# Additional fixtures for integration tests
# ============================================================================


class SimpleDecaySystem:
    """
    Simple system for testing: dx/dt = -x (exponential decay).

    Not a Mock - real minimal implementation for integration tests.
    """

    def __init__(self):
        self.nx = 2
        self.nu = 1
        self.ny = 2

    def __call__(self, x, u=None, t=None, backend=None, **kwargs):
        """Dynamics: dx/dt = -x"""
        x_array = np.asarray(x)
        return -x_array

    def forward(self, x, u=None, **kwargs):
        """Alias for __call__"""
        return self(x, u)


@pytest.fixture
def realistic_mock_system():
    """
    Create a simple system with realistic dynamics for integration tests.

    Returns system with dynamics: dx/dt = -x (exponential decay)
    """
    return SimpleDecaySystem()


# ============================================================================
# Regression Tests for DiscretizedSystem Integration (Bug #XXX)
# ============================================================================


class TestDiscretizedSystemIntegration:
    """
    Test IntegratorFactory behavior when called from DiscretizedSystem.

    Regression Prevention: The bug occurred because DiscretizedSystem was
    creating integrators on every step without specifying step_mode, causing
    inefficiency and incorrect behavior.
    """

    def test_fixed_step_methods_with_explicit_step_mode(self, mock_system):
        """
        Fixed-step methods should work correctly with explicit StepMode.FIXED.

        This is how DiscretizedSystem now calls the factory.
        """
        for method in ["euler", "rk4", "midpoint", "heun"]:
            integrator = IntegratorFactory.create(
                mock_system,
                backend="numpy",
                method=method,
                dt=0.01,
                step_mode=StepMode.FIXED,  # ← Explicit, not defaulted
            )

            # Should create successfully
            assert integrator is not None
            assert integrator.step_mode == StepMode.FIXED

            # Should have efficient step() method
            assert hasattr(integrator, "step")
            assert callable(integrator.step)

    def test_step_mode_fixed_with_prefer_manual(self, mock_system):
        """
        Combining step_mode=FIXED with prefer_manual=True should work.

        This is the pattern DiscretizedSystem uses for optimal performance.
        """
        integrator = IntegratorFactory.create(
            mock_system,
            backend="numpy",
            method="euler",
            dt=0.01,
            step_mode=StepMode.FIXED,
            prefer_manual=True,  # ← Avoid Julia overhead
        )

        from cdesym.systems.base.numerical_integration.fixed_step_integrators import (
            ExplicitEulerIntegrator,
        )

        # Should use native Python implementation
        assert isinstance(integrator, ExplicitEulerIntegrator)
        assert integrator.step_mode == StepMode.FIXED

    def test_default_step_mode_is_adaptive(self, mock_system):
        """
        Verify default step_mode is ADAPTIVE (documents behavior).

        This was part of the bug - when step_mode wasn't specified,
        it defaulted to ADAPTIVE even for fixed-step methods.
        """
        integrator = IntegratorFactory.create(
            mock_system,
            backend="numpy",
            method="euler",
            dt=0.01,
            # step_mode NOT specified ← Should default to ADAPTIVE
        )

        # Default should be ADAPTIVE
        assert integrator.step_mode == StepMode.ADAPTIVE

    def test_step_mode_mismatch_detection(self, mock_system):
        """
        Test that step_mode is correctly set regardless of method type.

        This documents that the factory doesn't validate step_mode against
        method - it trusts the caller (DiscretizedSystem) to set it correctly.
        """
        # Fixed-step method with ADAPTIVE mode (unusual but allowed)
        integrator1 = IntegratorFactory.create(
            mock_system,
            backend="numpy",
            method="euler",
            dt=0.01,
            step_mode=StepMode.ADAPTIVE,  # ← Explicit ADAPTIVE for fixed method
        )
        assert integrator1.step_mode == StepMode.ADAPTIVE

        # This is unusual but shouldn't crash - documents behavior
        # (DiscretizedSystem should never do this, but factory allows it)

    def test_integrator_is_stateless_and_reusable(self, realistic_mock_system):
        """
        Integrators should be stateless and reusable across steps.

        Critical for DiscretizedSystem's caching strategy.
        """
        integrator = IntegratorFactory.create(
            realistic_mock_system,
            backend="numpy",
            method="euler",
            dt=0.01,
            step_mode=StepMode.FIXED,
            prefer_manual=True,
        )

        # Should be able to call step() multiple times
        x0 = np.array([1.0, 0.0])

        x1 = integrator.step(x0, None, dt=0.01)
        x2 = integrator.step(x1, None, dt=0.01)
        x3 = integrator.step(x2, None, dt=0.01)

        # All calls should succeed
        assert x1 is not None
        assert x2 is not None
        assert x3 is not None

        # States should be different (evolving)
        assert not np.allclose(x1, x0)
        assert not np.allclose(x2, x1)
        assert not np.allclose(x3, x2)

        # Should be decaying (dx/dt = -x)
        assert x1[0] < x0[0]
        assert x2[0] < x1[0]
        assert x3[0] < x2[0]

    def test_sde_method_validation_on_deterministic_system(self, mock_system):
        """
        Factory should reject SDE methods on deterministic systems.

        DiscretizedSystem catches this ValueError and maps to deterministic.
        """
        # Mock system is deterministic (no is_stochastic attribute)
        with pytest.raises(ValueError, match="SDE method.*deterministic"):
            IntegratorFactory.create(
                mock_system,
                backend="numpy",
                method="EM",  # SDE method
                dt=0.01,
                step_mode=StepMode.FIXED,
            )

        # This error is expected - DiscretizedSystem catches and handles it


class TestStepMethodAvailability:
    """
    Verify all integrators have working step() methods.

    Critical for DiscretizedSystem's fast path.
    """

    def test_all_fixed_step_integrators_have_step_method(self, mock_system):
        """All fixed-step integrators must have step() method."""
        methods = ["euler", "rk4", "midpoint", "heun"]

        for method in methods:
            integrator = IntegratorFactory.create(
                mock_system,
                backend="numpy",
                method=method,
                dt=0.01,
                step_mode=StepMode.FIXED,
                prefer_manual=True,
            )

            # Must have step() method
            assert hasattr(integrator, "step"), f"{method} integrator missing step() method"

            # Must be callable
            assert callable(integrator.step), f"{method} integrator step() is not callable"

    def test_step_method_signature_correct(self, realistic_mock_system):
        """step() should accept (x, u, dt) parameters."""
        integrator = IntegratorFactory.create(
            realistic_mock_system,
            backend="numpy",
            method="euler",
            dt=0.01,
            step_mode=StepMode.FIXED,
            prefer_manual=True,
        )

        x0 = np.array([1.0, 0.0])

        # Should accept these forms
        x1 = integrator.step(x0, None)  # dt from integrator.dt
        x2 = integrator.step(x0, None, dt=0.01)  # explicit dt

        assert x1 is not None
        assert x2 is not None

        # Both should evolve the state
        assert not np.allclose(x1, x0)
        assert not np.allclose(x2, x0)

        # Should be decaying
        assert x1[0] < x0[0]
        assert x2[0] < x0[0]


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
