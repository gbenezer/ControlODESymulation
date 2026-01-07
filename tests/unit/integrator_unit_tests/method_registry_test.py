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
Unit Tests for Integration Method Registry
===========================================

Comprehensive test suite for method_registry.py covering:
- Method classification (SDE vs ODE, fixed-step vs adaptive)
- Name normalization across backends
- Method validation
- Backend compatibility checking
- Method discovery and introspection
- Ambiguous method handling
- Edge cases
"""

import pytest

from cdesym.systems.base.numerical_integration.method_registry import (
    # Classification functions
    is_sde_method,
    is_fixed_step,
    normalize_method_name,
    validate_method,
    get_available_methods,
    get_method_info,
    list_all_methods,
    # Constants
    DETERMINISTIC_FIXED_STEP,
    DETERMINISTIC_ADAPTIVE,
    SDE_FIXED_STEP,
    SDE_ADAPTIVE,
    SDE_METHODS,
    NORMALIZATION_MAP,
    BACKEND_METHODS,
)


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def all_backends():
    """List of all supported backends."""
    return ["numpy", "torch", "jax"]


@pytest.fixture
def canonical_sde_methods():
    """Canonical SDE method names that are in SDE_FIXED_STEP."""
    return [
        "euler_maruyama",
        "milstein",
        "stratonovich_milstein",
        "sra1",
    ]


@pytest.fixture
def canonical_ode_methods():
    """Canonical ODE method names."""
    return [
        "rk45",
        "rk23",
        "dopri5",
        "dopri8",
        "tsit5",
        "implicit_euler",
        "bdf",
        "lsoda",
    ]


@pytest.fixture
def manual_implementations():
    """Manual implementations available on all backends."""
    return ["euler", "heun", "midpoint", "rk4"]


@pytest.fixture
def ambiguous_methods():
    """Methods that appear in both deterministic and SDE contexts."""
    return ["euler", "midpoint"]  # heun is NOT ambiguous - only deterministic


# ============================================================================
# Test Method Classification
# ============================================================================


class TestMethodClassification:
    """Test is_sde_method() and is_fixed_step() functions."""

    def test_is_sde_method_canonical_names(self, canonical_sde_methods):
        """Test SDE detection for canonical names."""
        for method in canonical_sde_methods:
            assert is_sde_method(method), f"{method} should be detected as SDE"

    def test_is_sde_method_reversible_heun(self):
        """Test that reversible_heun is detected as SDE (in SDE_ADAPTIVE)."""
        assert is_sde_method("reversible_heun")

    def test_is_sde_method_backend_specific(self):
        """Test SDE detection for backend-specific names."""
        # NumPy/Julia names
        assert is_sde_method("EM")
        assert is_sde_method("RKMil")
        assert is_sde_method("SRIW1")
        assert is_sde_method("EulerHeun")

        # PyTorch names (note: 'euler' and 'milstein' are in SDE_METHODS)
        assert is_sde_method("euler")
        assert is_sde_method("milstein")
        assert is_sde_method("srk")

        # JAX names
        assert is_sde_method("Euler")
        assert is_sde_method("ItoMilstein")
        assert is_sde_method("StratonovichMilstein")

    def test_is_sde_method_deterministic(self, canonical_ode_methods):
        """Test that deterministic methods are not detected as SDE."""
        for method in canonical_ode_methods:
            assert not is_sde_method(method), f"{method} should not be SDE"

        # Explicit deterministic methods
        assert not is_sde_method("rk4")
        assert not is_sde_method("RK45")
        assert not is_sde_method("LSODA")
        assert not is_sde_method("dopri5")

    def test_is_sde_method_unknown(self):
        """Test that unknown methods return False."""
        assert not is_sde_method("unknown_method")
        assert not is_sde_method("my_custom_integrator")
        assert not is_sde_method("")

    def test_is_fixed_step_deterministic(self, manual_implementations):
        """Test fixed-step detection for deterministic methods."""
        for method in manual_implementations:
            assert is_fixed_step(method), f"{method} should be fixed-step"

    def test_is_fixed_step_adaptive_deterministic(self):
        """Test adaptive detection for deterministic methods."""
        adaptive_methods = ["RK45", "RK23", "LSODA", "dopri5", "tsit5", "Tsit5"]
        for method in adaptive_methods:
            assert not is_fixed_step(method), f"{method} should be adaptive"

    def test_is_fixed_step_sde(self):
        """Test fixed-step detection for SDE methods."""
        # Most SDE methods are fixed-step
        fixed_sde = ["euler_maruyama", "milstein", "EM", "RKMil", "SRIW1"]
        for method in fixed_sde:
            assert is_fixed_step(method), f"{method} should be fixed-step"

    def test_is_fixed_step_adaptive_sde(self):
        """Test adaptive detection for rare adaptive SDE methods."""
        adaptive_sde = ["AutoEM", "LambaEM", "adaptive_heun", "reversible_heun"]
        for method in adaptive_sde:
            assert not is_fixed_step(method), f"{method} should be adaptive"

    def test_is_fixed_step_unknown_conservative_default(self):
        """Test that unknown methods default to adaptive (conservative)."""
        assert not is_fixed_step("unknown_method")
        assert not is_fixed_step("my_custom_integrator")
        assert not is_fixed_step("")

    def test_get_method_info_ambiguous_method(self):
        """Test info for truly ambiguous methods."""
        # euler and midpoint exist in both deterministic and SDE contexts
        for method in ["euler", "midpoint"]:
            info = get_method_info(method, "numpy")

            # Should successfully get info
            assert info["original_name"] == method
            assert info["is_available"] is True
            assert info["is_fixed_step"] is True

            # heun is NOT ambiguous - it's ONLY in DETERMINISTIC_FIXED_STEP
            assert not is_sde_method("heun"), "heun should NOT be SDE"
            assert is_fixed_step("heun"), "heun should be fixed-step"


# ============================================================================
# Test Ambiguous Methods
# ============================================================================


class TestAmbiguousMethods:
    """Test proper handling of ambiguous method names."""

    def test_euler_in_both_contexts(self):
        """Test that 'euler' appears in both deterministic and SDE."""
        assert "euler" in DETERMINISTIC_FIXED_STEP
        assert "euler" in SDE_FIXED_STEP
        assert is_sde_method("euler")
        assert is_fixed_step("euler")

    def test_midpoint_in_both_contexts(self):
        """Test that 'midpoint' appears in both deterministic and SDE."""
        assert "midpoint" in DETERMINISTIC_FIXED_STEP
        assert "midpoint" in SDE_FIXED_STEP
        assert is_sde_method("midpoint")
        assert is_fixed_step("midpoint")

    def test_ambiguous_validation_with_stochastic_flag(self):
        """Test that ambiguous methods validate correctly with is_stochastic flag."""
        # Ambiguous methods should validate in both contexts
        for method in ["euler", "midpoint"]:
            # Should be valid for deterministic systems
            is_valid, error = validate_method(method, "numpy", is_stochastic=False)
            assert is_valid, f"{method} should be valid for deterministic"

            # Should be valid for stochastic systems
            is_valid, error = validate_method(method, "numpy", is_stochastic=True)
            assert is_valid, f"{method} should be valid for stochastic"

    def test_reversible_heun_only_adaptive(self):
        """Test that reversible_heun is only in SDE_ADAPTIVE, not SDE_FIXED_STEP."""
        assert "reversible_heun" in SDE_ADAPTIVE
        assert "reversible_heun" not in SDE_FIXED_STEP
        assert not is_fixed_step("reversible_heun")


# ============================================================================
# Test Method Normalization
# ============================================================================


class TestMethodNormalization:
    """Test normalize_method_name() function."""

    def test_normalize_euler_maruyama(self, all_backends):
        """Test normalization of euler_maruyama across backends."""
        expected = {
            "numpy": "EM",
            "torch": "euler",
            "jax": "Euler",
        }
        for backend in all_backends:
            result = normalize_method_name("euler_maruyama", backend)
            assert (
                result == expected[backend]
            ), f"euler_maruyama on {backend} should normalize to {expected[backend]}"

    def test_normalize_milstein(self, all_backends):
        """Test normalization of milstein across backends."""
        expected = {
            "numpy": "RKMil",
            "torch": "milstein",
            "jax": "ItoMilstein",
        }
        for backend in all_backends:
            result = normalize_method_name("milstein", backend)
            assert result == expected[backend]

    def test_normalize_stratonovich_milstein(self, all_backends):
        """Test normalization of stratonovich_milstein across backends."""
        expected = {
            "numpy": "RKMil",
            "torch": "milstein",
            "jax": "StratonovichMilstein",
        }
        for backend in all_backends:
            result = normalize_method_name("stratonovich_milstein", backend)
            assert result == expected[backend]

    def test_normalize_reversible_heun(self, all_backends):
        """Test normalization of reversible_heun across backends."""
        expected = {
            "numpy": "EulerHeun",
            "torch": "reversible_heun",
            "jax": "ReversibleHeun",
        }
        for backend in all_backends:
            result = normalize_method_name("reversible_heun", backend)
            assert result == expected[backend]

    def test_normalize_rk45(self, all_backends):
        """Test normalization of rk45 across backends."""
        expected = {
            "numpy": "RK45",
            "torch": "dopri5",
            "jax": "tsit5",
        }
        for backend in all_backends:
            result = normalize_method_name("rk45", backend)
            assert result == expected[backend]

    def test_normalize_tsit5(self, all_backends):
        """Test normalization of tsit5 across backends."""
        expected = {
            "numpy": "Tsit5",
            "torch": "dopri5",  # No tsit5 in PyTorch, maps to dopri5
            "jax": "tsit5",
        }
        for backend in all_backends:
            result = normalize_method_name("tsit5", backend)
            assert result == expected[backend]

    def test_normalize_already_valid(self, all_backends):
        """Test that already-valid names pass through unchanged."""
        # NumPy-specific
        assert normalize_method_name("EM", "numpy") == "EM"
        assert normalize_method_name("RK45", "numpy") == "RK45"
        assert normalize_method_name("Tsit5", "numpy") == "Tsit5"

        # PyTorch-specific
        assert normalize_method_name("dopri5", "torch") == "dopri5"
        assert normalize_method_name("milstein", "torch") == "milstein"

        # JAX-specific
        assert normalize_method_name("ItoMilstein", "jax") == "ItoMilstein"
        assert normalize_method_name("tsit5", "jax") == "tsit5"

    def test_normalize_manual_implementations(self, all_backends, manual_implementations):
        """Test that manual implementations pass through unchanged.

        Normalization does NOT apply Julia preference - that's handled in the factory.
        """
        for method in manual_implementations:
            for backend in all_backends:
                result = normalize_method_name(method, backend)

                # Manual implementations pass through unchanged
                assert (
                    result == method
                ), f"Method {method} on {backend} should pass through as '{method}', got '{result}'"

    def test_normalize_case_insensitive(self):
        """Test case-insensitive normalization."""
        # Should handle different cases
        assert normalize_method_name("RK45", "torch") == "dopri5"
        assert normalize_method_name("rk45", "torch") == "dopri5"
        assert normalize_method_name("Rk45", "torch") == "dopri5"

    def test_normalize_unknown_method(self, all_backends):
        """Test that unknown methods pass through unchanged."""
        unknown = "my_custom_method"
        for backend in all_backends:
            result = normalize_method_name(unknown, backend)
            assert result == unknown

    def test_normalize_idempotency(self, all_backends):
        """Test that normalization is idempotent."""
        method = "euler_maruyama"
        for backend in all_backends:
            normalized_once = normalize_method_name(method, backend)
            normalized_twice = normalize_method_name(normalized_once, backend)
            assert (
                normalized_once == normalized_twice
            ), f"Normalization should be idempotent for {method} on {backend}"

    def test_normalize_all_canonical_names(self):
        """Test normalization of all canonical names in NORMALIZATION_MAP."""
        for canonical_name, backend_map in NORMALIZATION_MAP.items():
            for backend, expected in backend_map.items():
                result = normalize_method_name(canonical_name, backend)
                assert (
                    result == expected
                ), f"{canonical_name} should normalize to {expected} on {backend}"


# ============================================================================
# Test Method Validation
# ============================================================================


class TestMethodValidation:
    """Test validate_method() function."""

    def test_validate_valid_deterministic(self, all_backends):
        """Test validation of valid deterministic methods."""
        valid_combos = [
            ("rk4", "numpy", False),
            ("RK45", "numpy", False),
            ("dopri5", "torch", False),
            ("tsit5", "jax", False),
        ]
        for method, backend, is_stochastic in valid_combos:
            is_valid, error = validate_method(method, backend, is_stochastic)
            assert is_valid, f"{method} on {backend} should be valid: {error}"
            assert error is None

    def test_validate_valid_stochastic(self, all_backends):
        """Test validation of valid stochastic methods."""
        valid_combos = [
            ("euler_maruyama", "numpy", True),
            ("EM", "numpy", True),
            ("euler", "torch", True),
            ("Euler", "jax", True),
            ("milstein", "torch", True),
        ]
        for method, backend, is_stochastic in valid_combos:
            is_valid, error = validate_method(method, backend, is_stochastic)
            assert is_valid, f"{method} on {backend} should be valid: {error}"
            assert error is None

    def test_validate_invalid_method_for_backend(self):
        """Test validation fails for unavailable methods."""
        # Use Julia-specific methods that don't exist on other backends
        # and don't have normalization mappings

        # SRIW1 (Julia SDE method) not available on PyTorch
        is_valid, error = validate_method("SRIW1", "torch", True)
        assert not is_valid
        assert "not available" in error
        assert "torch" in error

        # Vern7 (Julia ODE method) not available on PyTorch
        is_valid, error = validate_method("Vern7", "torch", False)
        assert not is_valid
        assert "not available" in error

    def test_validate_sde_method_on_deterministic_system(self):
        """Test validation fails for EXCLUSIVE SDE method on deterministic system."""
        # euler_maruyama is exclusive SDE (not in deterministic sets)
        is_valid, error = validate_method("euler_maruyama", "numpy", is_stochastic=False)
        assert not is_valid
        assert "SDE method" in error
        assert "deterministic system" in error

        is_valid, error = validate_method("EM", "numpy", is_stochastic=False)
        assert not is_valid
        assert "SDE method" in error

    def test_validate_ambiguous_method_on_deterministic_system(self):
        """Test that ambiguous methods (euler, midpoint) pass validation on deterministic."""
        # These methods exist in DETERMINISTIC_FIXED_STEP, so should be valid
        for method in ["euler", "midpoint"]:
            is_valid, error = validate_method(method, "numpy", is_stochastic=False)
            assert is_valid, f"{method} should be valid on deterministic: {error}"

    def test_validate_deterministic_method_on_stochastic_system(self):
        """Test that deterministic methods ARE allowed on stochastic systems."""
        # This should be valid (warning handled elsewhere)
        is_valid, error = validate_method("rk4", "numpy", is_stochastic=True)
        assert is_valid
        assert error is None

        is_valid, error = validate_method("RK45", "numpy", is_stochastic=True)
        assert is_valid
        assert error is None

    def test_validate_invalid_backend(self):
        """Test validation fails for invalid backend."""
        is_valid, error = validate_method("rk4", "invalid_backend", False)
        assert not is_valid
        assert "Invalid backend" in error
        assert "invalid_backend" in error

    def test_validate_normalizes_before_checking(self):
        """Test that validation normalizes method names first."""
        # Canonical name should be normalized and validated
        is_valid, error = validate_method("euler_maruyama", "torch", True)
        assert is_valid  # Should normalize to 'euler' which is valid

        is_valid, error = validate_method("rk45", "torch", False)
        assert is_valid  # Should normalize to 'dopri5' which is valid

    def test_validate_error_messages_include_alternatives(self):
        """Test that error messages suggest available methods."""
        # Use a method that's truly unavailable after normalization
        # Julia-specific method on PyTorch
        is_valid, error = validate_method("SRIW1", "torch", True)
        assert not is_valid
        assert "Available methods:" in error or "Canonical aliases:" in error


# ============================================================================
# Test Method Discovery
# ============================================================================


class TestMethodDiscovery:
    """Test get_available_methods() function."""

    def test_get_available_methods_all(self, all_backends):
        """Test getting all methods for each backend."""
        for backend in all_backends:
            methods = get_available_methods(backend, method_type="all")

            # Check expected keys
            expected_keys = {
                "deterministic_fixed_step",
                "deterministic_adaptive",
                "sde_fixed_step",
                "sde_adaptive",
                "canonical_aliases",
            }
            assert set(methods.keys()) == expected_keys

            # Each category should be non-empty except possibly sde_adaptive for JAX
            for key in expected_keys:
                if key == "sde_adaptive" and backend == "jax":
                    # JAX has no adaptive SDE methods in current implementation
                    continue
                assert len(methods[key]) > 0, f"{key} should not be empty for {backend}"

    def test_get_available_methods_deterministic(self, all_backends):
        """Test filtering for deterministic methods only."""
        for backend in all_backends:
            methods = get_available_methods(backend, method_type="deterministic")

            # Should have deterministic categories
            assert "deterministic_fixed_step" in methods
            assert "deterministic_adaptive" in methods
            assert "canonical_aliases" in methods

            # Should NOT have SDE categories
            assert "sde_fixed_step" not in methods
            assert "sde_adaptive" not in methods

    def test_get_available_methods_stochastic(self, all_backends):
        """Test filtering for stochastic methods only."""
        for backend in all_backends:
            methods = get_available_methods(backend, method_type="stochastic")

            # Should have SDE categories
            assert "sde_fixed_step" in methods
            assert "sde_adaptive" in methods
            assert "canonical_aliases" in methods

            # Should NOT have deterministic categories
            assert "deterministic_fixed_step" not in methods
            assert "deterministic_adaptive" not in methods

    def test_get_available_methods_fixed_step(self, all_backends):
        """Test filtering for fixed-step methods only."""
        for backend in all_backends:
            methods = get_available_methods(backend, method_type="fixed_step")

            # Should have fixed-step categories
            assert "deterministic_fixed_step" in methods
            assert "sde_fixed_step" in methods

            # Should NOT have adaptive or canonical
            assert "deterministic_adaptive" not in methods
            assert "sde_adaptive" not in methods
            assert "canonical_aliases" not in methods

    def test_get_available_methods_adaptive(self, all_backends):
        """Test filtering for adaptive methods only."""
        for backend in all_backends:
            methods = get_available_methods(backend, method_type="adaptive")

            # Should have adaptive categories
            assert "deterministic_adaptive" in methods
            assert "sde_adaptive" in methods

            # Should NOT have fixed-step or canonical
            assert "deterministic_fixed_step" not in methods
            assert "sde_fixed_step" not in methods
            assert "canonical_aliases" not in methods

    def test_get_available_methods_invalid_type(self):
        """Test that invalid method_type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown method_type"):
            get_available_methods("numpy", method_type="invalid")

    def test_get_available_methods_manual_implementations(
        self, all_backends, manual_implementations
    ):
        """Test that manual implementations appear in all backends."""
        for backend in all_backends:
            methods = get_available_methods(backend, method_type="all")
            fixed_step = methods["deterministic_fixed_step"]

            for manual_method in manual_implementations:
                assert (
                    manual_method in fixed_step
                ), f"{manual_method} should be in {backend} fixed-step methods"

    def test_get_available_methods_canonical_aliases_consistent(self, all_backends):
        """Test that canonical aliases are consistent across backends."""
        # Get canonical aliases for all backends
        all_aliases = {}
        for backend in all_backends:
            methods = get_available_methods(backend, method_type="all")
            all_aliases[backend] = set(methods["canonical_aliases"])

        # NumPy should have the most (includes lsoda)
        # PyTorch and JAX should have similar sets
        assert len(all_aliases["numpy"]) >= len(all_aliases["torch"])
        assert len(all_aliases["numpy"]) >= len(all_aliases["jax"])

    def test_get_available_methods_sorted(self, all_backends):
        """Test that returned methods are sorted."""
        for backend in all_backends:
            methods = get_available_methods(backend, method_type="all")

            for category, method_list in methods.items():
                # Check if sorted
                assert method_list == sorted(
                    method_list
                ), f"{category} should be sorted for {backend}"


# ============================================================================
# Test Method Info
# ============================================================================


class TestMethodInfo:
    """Test get_method_info() function."""

    def test_get_method_info_canonical_sde(self):
        """Test getting info for canonical SDE method."""
        info = get_method_info("euler_maruyama", "torch")

        assert info["original_name"] == "euler_maruyama"
        assert info["normalized_name"] == "euler"
        assert info["backend"] == "torch"
        assert info["is_sde"] is True
        assert info["is_fixed_step"] is True
        assert info["is_adaptive"] is False
        assert info["is_available"] is True
        assert info["category"] == "sde_fixed_step"

    def test_get_method_info_canonical_ode(self):
        """Test getting info for canonical ODE method."""
        info = get_method_info("rk45", "numpy")

        assert info["original_name"] == "rk45"
        assert info["normalized_name"] == "RK45"
        assert info["backend"] == "numpy"
        assert info["is_sde"] is False
        assert info["is_fixed_step"] is False
        assert info["is_adaptive"] is True
        assert info["is_available"] is True
        assert info["category"] == "deterministic_adaptive"

    def test_get_method_info_backend_specific(self):
        """Test getting info for backend-specific method."""
        info = get_method_info("ItoMilstein", "jax")

        assert info["original_name"] == "ItoMilstein"
        assert info["normalized_name"] == "ItoMilstein"
        assert info["backend"] == "jax"
        assert info["is_sde"] is True
        assert info["is_fixed_step"] is True
        assert info["category"] == "sde_fixed_step"

    def test_get_method_info_fixed_step_deterministic(self):
        """Test info for fixed-step deterministic method."""
        info = get_method_info("rk4", "numpy")

        assert info["is_sde"] is False
        assert info["is_fixed_step"] is True
        assert info["is_adaptive"] is False
        assert info["category"] == "deterministic_fixed_step"

    def test_get_method_info_adaptive_sde(self):
        """Test info for rare adaptive SDE method."""
        info = get_method_info("AutoEM", "numpy")

        assert info["is_sde"] is True
        assert info["is_fixed_step"] is False
        assert info["is_adaptive"] is True
        assert info["category"] == "sde_adaptive"

    def test_get_method_info_reversible_heun(self):
        """Test info for reversible_heun (adaptive SDE)."""
        info = get_method_info("reversible_heun", "torch")

        assert info["is_sde"] is True
        assert info["is_fixed_step"] is False  # In SDE_ADAPTIVE
        assert info["is_adaptive"] is True
        assert info["category"] == "sde_adaptive"

    def test_get_method_info_unavailable_method(self):
        """Test info for method unavailable on backend after normalization."""
        # LSODA normalizes to dopri5 on torch, so it IS available
        # Use a Julia-specific method that doesn't normalize
        info = get_method_info("SRIW1", "torch")

        assert info["original_name"] == "SRIW1"
        assert info["normalized_name"] == "SRIW1"  # No normalization for this
        assert info["backend"] == "torch"
        assert info["is_available"] is False

    def test_get_method_info_unknown_method(self):
        """Test info for completely unknown method."""
        info = get_method_info("my_custom_method", "numpy")

        assert info["original_name"] == "my_custom_method"
        assert info["normalized_name"] == "my_custom_method"
        assert info["is_available"] is False

    def test_get_method_info_ambiguous_method(self):
        """Test info for truly ambiguous methods."""
        # Only euler and midpoint are ambiguous
        for method in ["euler", "midpoint"]:
            info = get_method_info(method, "numpy")

            # Should successfully get info
            assert info["original_name"] == method
            assert info["normalized_name"] == method  # No normalization for manual methods
            assert info["is_available"] is True
            assert info["is_fixed_step"] is True


# ============================================================================
# Test List All Methods
# ============================================================================


class TestListAllMethods:
    """Test list_all_methods() function."""

    def test_list_all_methods_structure(self):
        """Test that list_all_methods returns expected structure."""
        methods = list_all_methods()

        expected_keys = {
            "deterministic_fixed_step",
            "deterministic_adaptive",
            "sde_fixed_step",
            "sde_adaptive",
            "all_canonical",
        }
        assert set(methods.keys()) == expected_keys

    def test_list_all_methods_non_empty(self):
        """Test that all categories are non-empty."""
        methods = list_all_methods()

        for category, method_list in methods.items():
            assert len(method_list) > 0, f"{category} should not be empty"

    def test_list_all_methods_sorted(self):
        """Test that all methods are sorted."""
        methods = list_all_methods()

        for category, method_list in methods.items():
            assert method_list == sorted(method_list), f"{category} should be sorted"

    def test_list_all_methods_canonical_from_normalization_map(self):
        """Test that canonical methods come from NORMALIZATION_MAP."""
        methods = list_all_methods()
        canonical = set(methods["all_canonical"])

        # Should match keys in NORMALIZATION_MAP
        expected_canonical = set(NORMALIZATION_MAP.keys())
        assert canonical == expected_canonical

    def test_list_all_methods_deterministic_disjoint_from_sde(self):
        """Test that deterministic and SDE methods mostly don't overlap."""
        methods = list_all_methods()

        det_fixed = set(methods["deterministic_fixed_step"])
        det_adaptive = set(methods["deterministic_adaptive"])
        sde_fixed = set(methods["sde_fixed_step"])
        sde_adaptive = set(methods["sde_adaptive"])

        # The overlap should only be ambiguous names
        det_all = det_fixed | det_adaptive
        sde_all = sde_fixed | sde_adaptive
        overlap = det_all & sde_all

        # Known ambiguous methods (euler, midpoint in both contexts)
        expected_overlap = {"euler", "midpoint"}
        assert overlap == expected_overlap


# ============================================================================
# Test Constants Consistency
# ============================================================================


class TestConstantsConsistency:
    """Test that module-level constants are consistent."""

    def test_sde_methods_union(self):
        """Test that SDE_METHODS is union of fixed and adaptive."""
        assert SDE_METHODS == SDE_FIXED_STEP | SDE_ADAPTIVE

    def test_deterministic_and_sde_overlap(self):
        """Test overlap between deterministic and SDE methods."""
        det_all = DETERMINISTIC_FIXED_STEP | DETERMINISTIC_ADAPTIVE
        overlap = det_all & SDE_METHODS

        # Only ambiguous methods should overlap (euler, midpoint)
        # reversible_heun is NOT in DETERMINISTIC_ADAPTIVE (fixed)
        # adaptive_heun is NOT in DETERMINISTIC_ADAPTIVE (fixed)
        expected_overlap = {"euler", "midpoint"}
        assert overlap == expected_overlap

    def test_reversible_heun_not_in_fixed_step(self):
        """Test that reversible_heun is NOT in SDE_FIXED_STEP."""
        assert "reversible_heun" not in SDE_FIXED_STEP
        assert "reversible_heun" in SDE_ADAPTIVE

    def test_stratonovich_milstein_in_sde_fixed_step(self):
        """Test that stratonovich_milstein is in SDE_FIXED_STEP."""
        assert "stratonovich_milstein" in SDE_FIXED_STEP

    def test_backend_methods_contain_manual_implementations(self):
        """Test that all backends include manual implementations."""
        manual = DETERMINISTIC_FIXED_STEP  # euler, midpoint, rk4, heun

        for backend, methods in BACKEND_METHODS.items():
            # All manual implementations should be present
            assert manual.issubset(methods), f"{backend} should include all manual implementations"

    def test_normalization_map_canonical_names(self):
        """Test that NORMALIZATION_MAP contains canonical names."""
        canonical_methods = {
            # SDE
            "euler_maruyama",
            "milstein",
            "stratonovich_milstein",
            "sra1",
            "reversible_heun",
            # ODE
            "rk45",
            "rk23",
            "dopri5",
            "dopri8",
            "tsit5",
            "implicit_euler",
            "bdf",
            "lsoda",
        }

        map_methods = set(NORMALIZATION_MAP.keys())
        assert canonical_methods.issubset(map_methods)

    def test_normalization_map_all_backends(self):
        """Test that each normalization has all backends."""
        expected_backends = {"numpy", "torch", "jax"}

        for method, backend_map in NORMALIZATION_MAP.items():
            assert (
                set(backend_map.keys()) == expected_backends
            ), f"{method} should have mappings for all backends"

    def test_backend_methods_coverage(self):
        """Test that BACKEND_METHODS covers all methods."""
        all_backends = {"numpy", "torch", "jax"}
        assert set(BACKEND_METHODS.keys()) == all_backends

        # Each backend should have reasonable number of methods
        for backend, methods in BACKEND_METHODS.items():
            assert len(methods) >= 10, f"{backend} should have at least 10 methods"

    def test_frozensets_immutable(self):
        """Test that method sets are frozensets (immutable)."""
        assert isinstance(DETERMINISTIC_FIXED_STEP, frozenset)
        assert isinstance(DETERMINISTIC_ADAPTIVE, frozenset)
        assert isinstance(SDE_FIXED_STEP, frozenset)
        assert isinstance(SDE_ADAPTIVE, frozenset)
        assert isinstance(SDE_METHODS, frozenset)

        for backend, methods in BACKEND_METHODS.items():
            assert isinstance(methods, frozenset), f"{backend} methods should be frozenset"


# ============================================================================
# Test Edge Cases
# ============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_string_method(self):
        """Test handling of empty string as method."""
        assert not is_sde_method("")
        assert not is_fixed_step("")
        assert normalize_method_name("", "numpy") == ""

        is_valid, error = validate_method("", "numpy", False)
        assert not is_valid

    def test_whitespace_method(self):
        """Test handling of whitespace in method names."""
        methods_with_whitespace = [" rk4", "rk4 ", " rk4 ", "\trk4", "rk4\n"]

        for method in methods_with_whitespace:
            # Should not match (exact string matching)
            assert not is_sde_method(method)
            # Normalization should pass through
            assert normalize_method_name(method, "numpy") == method

    def test_case_sensitivity_classification(self):
        """Test case sensitivity in classification functions."""
        # Classification is case-sensitive
        assert is_sde_method("EM")  # Julia style
        assert not is_sde_method("em")  # Not in set
        assert not is_sde_method("Em")

        # RK45 is adaptive
        assert not is_fixed_step("RK45")
        # 'rk45' (lowercase) is not in any set, defaults to False (adaptive)
        assert not is_fixed_step("rk45")

    def test_normalize_with_special_characters(self):
        """Test normalization with special characters."""
        special_names = [
            "my-method",
            "my_method_123",
            "method.v2",
            "method@backend",
        ]

        for method in special_names:
            # Should pass through unchanged
            for backend in ["numpy", "torch", "jax"]:
                result = normalize_method_name(method, backend)
                assert result == method

    def test_very_long_method_name(self):
        """Test handling of very long method names."""
        long_name = "a" * 1000

        assert not is_sde_method(long_name)
        assert not is_fixed_step(long_name)
        assert normalize_method_name(long_name, "numpy") == long_name

        is_valid, error = validate_method(long_name, "numpy", False)
        assert not is_valid

    def test_method_names_with_numbers(self):
        """Test methods with numbers in names."""
        # These are real methods
        assert "SRIW1" in SDE_FIXED_STEP
        assert "SRIW2" in SDE_FIXED_STEP
        assert is_sde_method("SRIW1")
        assert is_fixed_step("SRIW1")

    def test_unicode_method_names(self):
        """Test handling of unicode in method names."""
        unicode_names = ["метод", "方法", "معادلة"]

        for method in unicode_names:
            assert not is_sde_method(method)
            assert normalize_method_name(method, "numpy") == method

    def test_method_with_underscores_and_numbers(self):
        """Test methods with both underscores and numbers."""
        test_methods = ["method_1", "test_method_123", "_private_method"]

        for method in test_methods:
            # Should not match any known method
            assert not is_sde_method(method)
            # Should pass through normalization
            assert normalize_method_name(method, "numpy") == method

    def test_none_as_method_name(self):
        """Test that None as method name raises ValueError."""
        # Python's 'in' operator with None and frozenset returns False
        # rather than raising TypeError
        assert not is_sde_method(None)
        assert not is_fixed_step(None)

        # normalize_method_name should raise ValueError for None
        with pytest.raises(ValueError, match="method cannot be None"):
            normalize_method_name(None, "numpy")

        # validate_method calls normalize_method_name internally,
        # so it will also raise ValueError
        with pytest.raises(ValueError, match="method cannot be None"):
            validate_method(None, "numpy", False)


# ============================================================================
# Additional Edge Case Tests
# ============================================================================


class TestAdditionalEdgeCases:
    """Additional edge case tests for robustness."""

    def test_normalization_with_none_backend(self):
        """Test that normalization with None backend is handled."""
        # This should either return the method unchanged or handle gracefully
        result = normalize_method_name("rk4", None)
        # If backend is None, it won't be in BACKEND_METHODS, so returns unchanged
        assert result == "rk4"

    def test_validate_with_none_backend(self):
        """Test validation with None backend."""
        is_valid, error = validate_method("rk4", None, False)
        assert not is_valid
        assert "Invalid backend" in error

    def test_methods_with_similar_names(self):
        """Test methods with similar but distinct names."""
        # Ensure we don't confuse similar method names
        assert is_sde_method("euler")  # SDE version
        assert is_sde_method("Euler")  # JAX SDE version
        assert not is_sde_method("euler_method")  # Not a real method

        assert is_sde_method("EulerHeun")  # Julia SDE
        assert not is_sde_method("EulerHuen")  # Misspelled

    def test_backend_case_sensitivity(self):
        """Test that backend names are case-sensitive."""
        # Backend names should be lowercase
        is_valid, error = validate_method("rk4", "NumPy", False)
        assert not is_valid
        assert "Invalid backend" in error

        is_valid, error = validate_method("rk4", "TORCH", False)
        assert not is_valid

    def test_empty_normalization_map_lookup(self):
        """Test normalization when method has no entry."""
        # Method that doesn't exist in NORMALIZATION_MAP
        for backend in ["numpy", "torch", "jax"]:
            result = normalize_method_name("nonexistent_method", backend)
            assert result == "nonexistent_method"

    def test_multiple_normalization_attempts(self):
        """Test that multiple normalizations are truly idempotent."""
        method = "euler_maruyama"
        for backend in ["numpy", "torch", "jax"]:
            norm1 = normalize_method_name(method, backend)
            norm2 = normalize_method_name(norm1, backend)
            norm3 = normalize_method_name(norm2, backend)

            assert norm1 == norm2 == norm3

    def test_validate_exclusive_sde_check(self):
        """Test that validation correctly identifies exclusive SDE methods."""
        # euler_maruyama is EXCLUSIVE SDE (not in deterministic sets)
        is_valid, error = validate_method("euler_maruyama", "numpy", False)
        assert not is_valid
        assert "SDE method" in error

        # EM is also EXCLUSIVE SDE
        is_valid, error = validate_method("EM", "numpy", False)
        assert not is_valid
        assert "SDE method" in error

        # But euler is NOT exclusive (in both deterministic and SDE)
        is_valid, error = validate_method("euler", "numpy", False)
        assert is_valid, "euler should be valid for deterministic (ambiguous method)"

    def test_get_info_for_all_canonical_methods(self):
        """Test get_method_info works for all canonical methods."""
        for canonical_name in NORMALIZATION_MAP.keys():
            for backend in ["numpy", "torch", "jax"]:
                info = get_method_info(canonical_name, backend)

                # Should always return a dict with expected keys
                assert "original_name" in info
                assert "normalized_name" in info
                assert "backend" in info
                assert "is_sde" in info
                assert "is_fixed_step" in info
                assert "is_adaptive" in info
                assert "is_available" in info
                assert "category" in info

                # Original name should match input
                assert info["original_name"] == canonical_name
                assert info["backend"] == backend

    def test_all_backend_methods_are_valid(self):
        """Test that all methods in BACKEND_METHODS validate correctly."""
        for backend, methods in BACKEND_METHODS.items():
            for method in methods:
                # Determine if method is SDE
                is_sde = is_sde_method(method)

                # Ambiguous methods (euler, midpoint) should validate in both contexts
                if method in ["euler", "midpoint"]:
                    # Test both contexts
                    is_valid_det, _ = validate_method(method, backend, False)
                    is_valid_sde, _ = validate_method(method, backend, True)
                    assert is_valid_det, f"{method} should be valid for deterministic"
                    assert is_valid_sde, f"{method} should be valid for stochastic"
                else:
                    # Test appropriate context
                    is_valid, error = validate_method(method, backend, is_sde)
                    assert is_valid, f"{method} should be valid on {backend}: {error}"


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests combining multiple functions."""

    def test_full_workflow_canonical_to_backend(self):
        """Test complete workflow: canonical -> normalize -> validate."""
        canonical_method = "euler_maruyama"
        backends = ["numpy", "torch", "jax"]

        for backend in backends:
            # Normalize
            normalized = normalize_method_name(canonical_method, backend)

            # Validate
            is_valid, error = validate_method(normalized, backend, is_stochastic=True)
            assert is_valid, f"{canonical_method} should work on {backend}: {error}"

            # Get info
            info = get_method_info(canonical_method, backend)
            assert info["is_sde"]
            assert info["is_available"]

    def test_full_workflow_discover_and_validate(self):
        """Test workflow: discover -> validate all methods."""
        for backend in ["numpy", "torch", "jax"]:
            methods = get_available_methods(backend, method_type="all")

            # Validate all SDE methods
            for method in methods["sde_fixed_step"]:
                is_valid, error = validate_method(method, backend, is_stochastic=True)
                assert is_valid, f"{method} should be valid SDE on {backend}: {error}"

            # Validate all deterministic methods
            for method in methods["deterministic_fixed_step"]:
                is_valid, error = validate_method(method, backend, is_stochastic=False)
                assert is_valid, f"{method} should be valid deterministic on {backend}: {error}"

    def test_cross_backend_portability(self):
        """Test that canonical names work across all backends."""
        canonical = ["euler_maruyama", "milstein", "rk45", "rk23"]
        backends = ["numpy", "torch", "jax"]

        for method in canonical:
            for backend in backends:
                # Should normalize successfully
                normalized = normalize_method_name(method, backend)
                assert normalized is not None

                # Should validate successfully
                is_stochastic = method in ["euler_maruyama", "milstein"]
                is_valid, error = validate_method(method, backend, is_stochastic)
                assert is_valid, f"{method} should be portable to {backend}: {error}"

    def test_method_info_matches_classification(self):
        """Test that get_method_info matches individual classification functions."""
        test_methods = [
            ("rk4", "numpy"),
            ("euler_maruyama", "torch"),
            ("RK45", "numpy"),
            ("ItoMilstein", "jax"),
        ]

        for method, backend in test_methods:
            info = get_method_info(method, backend)
            normalized = info["normalized_name"]

            # Check consistency
            assert info["is_sde"] == is_sde_method(normalized)
            assert info["is_fixed_step"] == is_fixed_step(normalized)
            assert info["is_adaptive"] == (not is_fixed_step(normalized))

    def test_all_normalization_map_entries_valid(self):
        """Test that all entries in NORMALIZATION_MAP are valid."""
        for canonical, backend_map in NORMALIZATION_MAP.items():
            for backend, normalized in backend_map.items():
                # Normalized method should exist in BACKEND_METHODS
                assert (
                    normalized in BACKEND_METHODS[backend]
                ), f"{canonical} -> {normalized} should be in {backend}"

                # Should validate successfully
                info = get_method_info(canonical, backend)
                is_valid, error = validate_method(canonical, backend, info["is_sde"])
                assert (
                    is_valid
                ), f"{canonical} -> {normalized} should validate on {backend}: {error}"


# ============================================================================
# Test Module Exports
# ============================================================================


class TestModuleExports:
    """Test that __all__ exports are correct."""

    def test_all_exports_exist(self):
        """Test that all names in __all__ are defined."""
        from cdesym.systems.base.numerical_integration import method_registry

        for name in method_registry.__all__:
            assert hasattr(method_registry, name), f"{name} should be exported"

    def test_all_exports_complete(self):
        """Test that __all__ includes expected exports."""
        from cdesym.systems.base.numerical_integration import method_registry

        expected_functions = [
            "is_sde_method",
            "is_fixed_step",
            "normalize_method_name",
            "validate_method",
            "get_available_methods",
            "get_method_info",
            "list_all_methods",
        ]

        for func in expected_functions:
            assert func in method_registry.__all__, f"{func} should be in __all__"

    def test_constants_exported(self):
        """Test that important constants are exported."""
        from cdesym.systems.base.numerical_integration import method_registry

        expected_constants = [
            "DETERMINISTIC_FIXED_STEP",
            "DETERMINISTIC_ADAPTIVE",
            "SDE_FIXED_STEP",
            "SDE_ADAPTIVE",
            "SDE_METHODS",
            "NORMALIZATION_MAP",
            "BACKEND_METHODS",
        ]

        for const in expected_constants:
            assert const in method_registry.__all__, f"{const} should be in __all__"


# ============================================================================
# Performance Tests (Optional)
# ============================================================================


class TestPerformance:
    """Optional performance tests."""

    def test_classification_performance(self):
        """Test that classification is fast (O(1) lookups)."""
        import time

        methods = list(DETERMINISTIC_FIXED_STEP | DETERMINISTIC_ADAPTIVE | SDE_METHODS)

        start = time.time()
        for _ in range(10000):
            for method in methods:
                is_sde_method(method)
                is_fixed_step(method)
        elapsed = time.time() - start

        # Should be very fast (< 1s for 10k * ~50 methods)
        assert elapsed < 1.0, f"Classification too slow: {elapsed:.3f}s"

    def test_normalization_performance(self):
        """Test that normalization is fast."""
        import time

        methods = list(NORMALIZATION_MAP.keys())
        backends = ["numpy", "torch", "jax"]

        start = time.time()
        for _ in range(10000):
            for method in methods:
                for backend in backends:
                    normalize_method_name(method, backend)
        elapsed = time.time() - start

        # Should be fast (< 2s for 10k * ~13 methods * 3 backends)
        # Threshold has been relaxed
        assert elapsed < 2.0, f"Normalization too slow: {elapsed:.3f}s"


# ============================================================================
# Documentation Tests
# ============================================================================


class TestDocumentation:
    """Test that documentation examples work."""

    def test_docstring_examples_is_sde_method(self):
        """Test examples from is_sde_method docstring."""
        # Canonical SDE names
        assert is_sde_method("euler_maruyama") is True
        assert is_sde_method("milstein") is True

        # Backend-specific SDE names
        assert is_sde_method("EM") is True
        assert is_sde_method("euler") is True
        assert is_sde_method("ItoMilstein") is True

        # Deterministic methods
        assert is_sde_method("rk4") is False
        assert is_sde_method("RK45") is False
        assert is_sde_method("dopri5") is False

    def test_docstring_examples_normalize(self):
        """Test examples from normalize_method_name docstring."""
        assert normalize_method_name("euler_maruyama", "numpy") == "EM"
        assert normalize_method_name("euler_maruyama", "torch") == "euler"
        assert normalize_method_name("euler_maruyama", "jax") == "Euler"

        assert normalize_method_name("rk45", "numpy") == "RK45"
        assert normalize_method_name("rk45", "torch") == "dopri5"
        assert normalize_method_name("rk45", "jax") == "tsit5"

    def test_docstring_examples_validate(self):
        """Test examples from validate_method docstring."""
        is_valid, error = validate_method("euler_maruyama", "torch", is_stochastic=True)
        assert is_valid is True
        assert error is None

        is_valid, error = validate_method("RK45", "numpy", is_stochastic=False)
        assert is_valid is True

        is_valid, error = validate_method("rk4", "jax", is_stochastic=False)
        assert is_valid is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
