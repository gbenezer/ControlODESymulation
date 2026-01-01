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
Comprehensive unit tests for SymbolicValidator

Tests cover all validation checks:
1. Required attributes
2. Type validation
3. Dimension consistency
4. Symbol validation
5. Parameter validation
6. Physical constraints
7. Naming conventions
8. Usage patterns
9. ValidationResult integration
10. Autonomous system validation
"""

import numpy as np
import pytest
import sympy as sp

from src.systems.base.utils.symbolic_validator import (
    SymbolicValidator,
    ValidationError,
)
from src.types.utilities import ValidationResult, SymbolicValidationResult

# ============================================================================
# Mock System Classes for Testing
# ============================================================================


class MockValidSystem:
    """Valid controlled system for testing"""

    def __init__(self):
        x = sp.symbols("x", real=True)
        u = sp.symbols("u", real=True)
        a = sp.symbols("a", real=True, positive=True)

        self.state_vars = [x]
        self.control_vars = [u]
        self._f_sym = sp.Matrix([-a * x + u])
        self._h_sym = None
        self.parameters = {a: 1.0}
        self.output_vars = []
        self.order = 1


class MockAutonomousSystem:
    """Valid autonomous system for testing"""

    def __init__(self):
        x = sp.symbols("x", real=True)
        a = sp.symbols("a", real=True, positive=True)

        self.state_vars = [x]
        self.control_vars = []  # AUTONOMOUS
        self._f_sym = sp.Matrix([-a * x])
        self._h_sym = None
        self.parameters = {a: 2.0}
        self.output_vars = []
        self.order = 1


class MockAutonomous2D:
    """Valid 2D autonomous system"""

    def __init__(self):
        x1, x2 = sp.symbols("x1 x2", real=True)
        k = sp.symbols("k", real=True, positive=True)

        self.state_vars = [x1, x2]
        self.control_vars = []  # AUTONOMOUS
        self._f_sym = sp.Matrix([x2, -k * x1])
        self._h_sym = None
        self.parameters = {k: 10.0}
        self.output_vars = []
        self.order = 1


class MockSecondOrderSystem:
    """Valid second-order controlled system"""

    def __init__(self):
        q, q_dot = sp.symbols("q q_dot", real=True)
        u = sp.symbols("u", real=True)
        k, c = sp.symbols("k c", real=True, positive=True)

        self.state_vars = [q, q_dot]
        self.control_vars = [u]
        self._f_sym = sp.Matrix([-k * q - c * q_dot + u])
        self._h_sym = None
        self.parameters = {k: 10.0, c: 0.5}
        self.output_vars = []
        self.order = 2


# ============================================================================
# Test ValidationResult Dataclass
# ============================================================================


class TestValidationResult:
    """Test ValidationResult dataclass."""

    def test_valid_result_creation(self):
        """Test creating valid ValidationResult."""
        result = ValidationResult(valid=True, errors=[], warnings=[], info={"nx": 1, "nu": 1})

        assert result["valid"]
        assert len(result["errors"]) == 0
        assert len(result["warnings"]) == 0
        assert result["info"]["nx"] == 1

    def test_invalid_result_creation(self):
        """Test creating invalid ValidationResult."""
        result = ValidationResult(
            valid=False, errors=["Dimension mismatch"], warnings=["Unusual order"], info={}
        )

        assert not result["valid"]
        assert len(result["errors"]) == 1
        assert len(result["warnings"]) == 1


# ============================================================================
# Test Class 1: Required Attributes (Controlled Systems)
# ============================================================================


class TestRequiredAttributesControlled:
    """Test validation of required attributes for controlled systems"""

    def test_valid_system_passes(self):
        """Test that valid system passes validation"""
        system = MockValidSystem()
        validator = SymbolicValidator(system)

        result = validator.validate(raise_on_error=False)
        assert result.is_valid

    def test_valid_system_with_raise(self):
        """Test valid system doesn't raise with raise_on_error=True"""
        system = MockValidSystem()
        validator = SymbolicValidator(system)

        result = validator.validate(raise_on_error=True)
        assert result.is_valid

    def test_missing_state_vars(self):
        """Test error when state_vars is missing"""
        system = MockValidSystem()
        delattr(system, "state_vars")

        validator = SymbolicValidator(system)

        with pytest.raises(ValidationError, match="Missing required attribute 'state_vars'"):
            validator.validate(raise_on_error=True)

    def test_missing_state_vars_no_raise(self):
        """Test error detection without raising"""
        system = MockValidSystem()
        delattr(system, "state_vars")

        validator = SymbolicValidator(system)
        result = validator.validate(raise_on_error=False)

        assert not result.is_valid
        assert any("state_vars" in err for err in result.errors)

    def test_empty_state_vars(self):
        """Test error when state_vars is empty"""
        system = MockValidSystem()
        system.state_vars = []

        validator = SymbolicValidator(system)

        with pytest.raises(ValidationError, match="state_vars is empty"):
            validator.validate(raise_on_error=True)

    def test_missing_control_vars(self):
        """Test error when control_vars is missing"""
        system = MockValidSystem()
        delattr(system, "control_vars")

        validator = SymbolicValidator(system)

        with pytest.raises(ValidationError, match="Missing required attribute 'control_vars'"):
            validator.validate(raise_on_error=True)

    def test_missing_f_sym(self):
        """Test error when _f_sym is missing"""
        system = MockValidSystem()
        delattr(system, "_f_sym")

        validator = SymbolicValidator(system)

        with pytest.raises(ValidationError, match="Missing required attribute '_f_sym'"):
            validator.validate(raise_on_error=True)

    def test_none_f_sym(self):
        """Test error when _f_sym is None"""
        system = MockValidSystem()
        system._f_sym = None

        validator = SymbolicValidator(system)

        with pytest.raises(ValidationError, match="_f_sym is not defined"):
            validator.validate(raise_on_error=True)

    def test_missing_parameters(self):
        """Test error when parameters is missing"""
        system = MockValidSystem()
        delattr(system, "parameters")

        validator = SymbolicValidator(system)

        with pytest.raises(ValidationError, match="Missing required attribute 'parameters'"):
            validator.validate(raise_on_error=True)

    def test_missing_order(self):
        """Test error when order is missing"""
        system = MockValidSystem()
        delattr(system, "order")

        validator = SymbolicValidator(system)

        with pytest.raises(ValidationError, match="Missing required attribute 'order'"):
            validator.validate(raise_on_error=True)


# ============================================================================
# Test Class 2: Required Attributes (Autonomous Systems)
# ============================================================================


class TestRequiredAttributesAutonomous:
    """Test validation of required attributes for autonomous systems"""

    def test_valid_autonomous_system_passes(self):
        """Test that valid autonomous system passes validation"""
        system = MockAutonomousSystem()
        validator = SymbolicValidator(system)

        result = validator.validate(raise_on_error=False)
        assert result.is_valid

    def test_empty_control_vars_allowed(self):
        """Test that empty control_vars is allowed for autonomous systems"""
        system = MockAutonomousSystem()
        # control_vars is already []

        validator = SymbolicValidator(system)
        result = validator.validate(raise_on_error=False)

        assert result.is_valid
        assert result.info["nu"] == 0
        assert result.info.get("is_autonomous", False)

    def test_autonomous_2d_system_passes(self):
        """Test that 2D autonomous system passes validation"""
        system = MockAutonomous2D()
        validator = SymbolicValidator(system)

        result = validator.validate(raise_on_error=False)

        assert result.is_valid
        assert result.info["nx"] == 2
        assert result.info["nu"] == 0


# ============================================================================
# Test Class 3: Type Validation
# ============================================================================


class TestTypeValidation:
    """Test type validation"""

    def test_state_vars_not_symbols(self):
        """Test error when state_vars contains non-Symbols"""

        class InvalidSystem:
            state_vars = ["x", "y"]  # Strings - INVALID
            control_vars = [sp.symbols("u")]
            _f_sym = sp.Matrix([sp.Integer(1), sp.Integer(2)])
            _h_sym = None
            parameters = {}
            output_vars = []
            order = 1

        system = InvalidSystem()
        validator = SymbolicValidator(system)

        with pytest.raises(ValidationError, match="is not a SymPy Symbol"):
            validator.validate(raise_on_error=True)

    def test_control_vars_not_symbols(self):
        """Test error when control_vars contains non-Symbols"""
        system = MockValidSystem()
        system.control_vars = ["u"]  # String instead of Symbol

        validator = SymbolicValidator(system)

        with pytest.raises(ValidationError, match="is not a SymPy Symbol"):
            validator.validate(raise_on_error=True)

    def test_output_vars_not_symbols(self):
        """Test error when output_vars contains non-Symbols"""
        system = MockValidSystem()
        system.output_vars = ["y"]  # String instead of Symbol

        validator = SymbolicValidator(system)

        with pytest.raises(ValidationError, match="is not a SymPy Symbol"):
            validator.validate(raise_on_error=True)

    def test_f_sym_not_matrix(self):
        """Test error when _f_sym is not a Matrix"""
        system = MockValidSystem()
        system._f_sym = [sp.symbols("x")]  # List instead of Matrix

        validator = SymbolicValidator(system)

        with pytest.raises(ValidationError, match="_f_sym must be sp.Matrix"):
            validator.validate(raise_on_error=True)

    def test_h_sym_not_matrix(self):
        """Test error when _h_sym is not a Matrix"""
        system = MockValidSystem()
        system._h_sym = [sp.symbols("x")]  # List instead of Matrix

        validator = SymbolicValidator(system)

        with pytest.raises(ValidationError, match="_h_sym must be sp.Matrix"):
            validator.validate(raise_on_error=True)

    def test_parameter_key_not_symbol(self):
        """Test error when parameter key is not a Symbol"""
        system = MockValidSystem()
        system.parameters = {"a": 1.0}  # String key instead of Symbol

        validator = SymbolicValidator(system)

        with pytest.raises(ValidationError, match="Parameter key.*is not a SymPy Symbol"):
            validator.validate(raise_on_error=True)

    def test_parameter_value_not_numeric(self):
        """Test error when parameter value is not numeric"""
        system = MockValidSystem()
        a = sp.symbols("a")
        x, u = sp.symbols("x u")

        # Need valid _f_sym that uses 'a'
        system.state_vars = [x]
        system.control_vars = [u]
        system._f_sym = sp.Matrix([a * x + u])
        system.parameters = {a: "1.0"}  # String value instead of number

        validator = SymbolicValidator(system)

        with pytest.raises(ValidationError, match="Parameter value.*must be numeric"):
            validator.validate(raise_on_error=True)

    def test_order_not_integer(self):
        """Test error when order is not integer"""
        system = MockValidSystem()
        system.order = 1.5  # Float instead of int

        validator = SymbolicValidator(system)

        with pytest.raises(ValidationError, match="order must be int"):
            validator.validate(raise_on_error=True)


# ============================================================================
# Test Class 4: Dimension Validation
# ============================================================================


class TestDimensionValidation:
    """Test dimension consistency checks"""

    def test_f_sym_wrong_rows_first_order(self):
        """Test error when _f_sym has wrong number of rows"""
        system = MockValidSystem()
        x, y = sp.symbols("x y")
        system.state_vars = [x, y]  # 2 states
        # But _f_sym still has 1 row!

        validator = SymbolicValidator(system)

        with pytest.raises(ValidationError, match="has.*rows but expected"):
            validator.validate(raise_on_error=True)

    def test_f_sym_not_column_vector(self):
        """Test error when _f_sym is not a column vector"""
        system = MockValidSystem()
        x = sp.symbols("x")
        system._f_sym = sp.Matrix([[x, x]])  # Row vector instead of column

        validator = SymbolicValidator(system)

        with pytest.raises(ValidationError, match="must be column vector"):
            validator.validate(raise_on_error=True)

    def test_h_sym_not_column_vector(self):
        """Test error when _h_sym is not a column vector"""
        system = MockValidSystem()
        x = sp.symbols("x")
        system._h_sym = sp.Matrix([[x, x]])  # Row vector

        validator = SymbolicValidator(system)

        with pytest.raises(ValidationError, match="must be column vector"):
            validator.validate(raise_on_error=True)

    def test_output_vars_h_sym_mismatch(self):
        """Test error when output_vars and _h_sym dimensions don't match"""
        system = MockValidSystem()
        y1, y2 = sp.symbols("y1 y2")
        x = sp.symbols("x")

        system.output_vars = [y1, y2]  # 2 outputs
        system._h_sym = sp.Matrix([x])  # But only 1 row in _h_sym

        validator = SymbolicValidator(system)

        with pytest.raises(ValidationError, match="output_vars has.*elements but _h_sym has.*rows"):
            validator.validate(raise_on_error=True)

    def test_order_not_divisible(self):
        """Test error when nx not divisible by order"""
        system = MockValidSystem()
        x1, x2, x3 = sp.symbols("x1 x2 x3")
        system.state_vars = [x1, x2, x3]  # 3 states
        system.order = 2  # But 3 is not divisible by 2!

        validator = SymbolicValidator(system)

        with pytest.raises(ValidationError, match="must be divisible by order"):
            validator.validate(raise_on_error=True)

    def test_second_order_system_valid(self):
        """Test that valid second-order system passes"""
        system = MockSecondOrderSystem()
        validator = SymbolicValidator(system)

        result = validator.validate(raise_on_error=False)
        assert result.is_valid

    def test_order_less_than_one(self):
        """Test error when order < 1"""
        system = MockValidSystem()
        system.order = 0

        validator = SymbolicValidator(system)

        with pytest.raises(ValidationError, match="order must be >= 1"):
            validator.validate(raise_on_error=True)

    def test_order_very_large_warning(self):
        """Test warning when order is unusually high"""
        system = MockValidSystem()
        # Need to create enough state vars to match the order
        x_vars = [sp.symbols(f"x{i}") for i in range(10)]
        u = sp.symbols("u")

        system.state_vars = x_vars
        system.control_vars = [u]
        system._f_sym = sp.Matrix([u])  # Only 1 row for order=10 system
        system.order = 10  # Very high order
        system.parameters = {}

        validator = SymbolicValidator(system)

        with pytest.warns(UserWarning, match="order.*unusually high"):
            result = validator.validate(raise_on_error=False)

        assert result.is_valid
        assert len(result.warnings) > 0


# ============================================================================
# Test Class 5: Symbol Validation (Controlled Systems)
# ============================================================================


class TestSymbolValidationControlled:
    """Test symbolic expression validation for controlled systems"""

    def test_undefined_symbol_in_f_sym(self):
        """Test error when _f_sym contains undefined symbol"""
        system = MockValidSystem()
        mystery = sp.symbols("mystery")
        x = sp.symbols("x")
        system._f_sym = sp.Matrix([x + mystery])  # mystery not defined!

        validator = SymbolicValidator(system)

        with pytest.raises(ValidationError, match="undefined symbols"):
            validator.validate(raise_on_error=True)

    def test_f_sym_no_state_dependency_warning(self):
        """Test warning when dynamics don't depend on states"""
        system = MockValidSystem()
        x, u = sp.symbols("x u")

        system.state_vars = [x]
        system.control_vars = [u]
        system._f_sym = sp.Matrix([u])  # Only depends on control!
        system.parameters = {}  # No parameters needed

        validator = SymbolicValidator(system)

        with pytest.warns(UserWarning, match="does not depend on any state"):
            result = validator.validate(raise_on_error=False)

        assert result.is_valid
        assert len(result.warnings) > 0

    def test_h_sym_contains_control(self):
        """Test error when output depends on control"""
        system = MockValidSystem()
        x, u = sp.symbols("x u")

        system.state_vars = [x]
        system.control_vars = [u]
        system._f_sym = sp.Matrix([x + u])
        system._h_sym = sp.Matrix([x + u])  # Depends on control!
        system.parameters = {}

        validator = SymbolicValidator(system)

        with pytest.raises(ValidationError, match="_h_sym contains control"):
            validator.validate(raise_on_error=True)

    def test_h_sym_undefined_symbol(self):
        """Test error when _h_sym contains undefined symbol"""
        system = MockValidSystem()
        x = sp.symbols("x")
        mystery = sp.symbols("mystery")
        system._h_sym = sp.Matrix([x + mystery])

        validator = SymbolicValidator(system)

        with pytest.raises(ValidationError, match="_h_sym contains undefined symbols"):
            validator.validate(raise_on_error=True)


# ============================================================================
# Test Class 6: Symbol Validation (Autonomous Systems)
# ============================================================================


class TestSymbolValidationAutonomous:
    """Test symbolic expression validation for autonomous systems"""

    def test_autonomous_no_control_dependency(self):
        """Test that autonomous system dynamics don't depend on control"""
        system = MockAutonomousSystem()
        validator = SymbolicValidator(system)

        result = validator.validate(raise_on_error=False)

        # Should pass - no control variables to depend on
        assert result.is_valid

    def test_autonomous_undefined_symbol_error(self):
        """Test error when autonomous system has undefined symbol"""
        system = MockAutonomousSystem()
        mystery = sp.symbols("mystery")
        x = sp.symbols("x")
        system._f_sym = sp.Matrix([x + mystery])

        validator = SymbolicValidator(system)

        with pytest.raises(ValidationError, match="undefined symbols"):
            validator.validate(raise_on_error=True)

    def test_autonomous_2d_valid(self):
        """Test that 2D autonomous system is symbolically valid"""
        system = MockAutonomous2D()
        validator = SymbolicValidator(system)

        result = validator.validate(raise_on_error=False)
        assert result.is_valid


# ============================================================================
# Test Class 7: Parameter Validation
# ============================================================================


class TestParameterValidation:
    """Test parameter validation"""

    def test_unused_parameter_warning(self):
        """Test warning for unused parameters"""
        system = MockValidSystem()
        unused = sp.symbols("unused")
        system.parameters[unused] = 5.0  # Not used in _f_sym

        validator = SymbolicValidator(system)

        with pytest.warns(UserWarning, match="defined but not used"):
            result = validator.validate(raise_on_error=False)

        assert result.is_valid
        assert len(result.warnings) > 0

    def test_parameter_used_in_h_sym_no_warning(self):
        """Test no warning when parameter is used in _h_sym"""
        system = MockValidSystem()
        x, u = sp.symbols("x u")
        a, b = sp.symbols("a b")

        system.state_vars = [x]
        system.control_vars = [u]
        system._f_sym = sp.Matrix([a * x + u])  # Uses 'a'
        system._h_sym = sp.Matrix([b * x])  # Uses 'b'
        system.parameters = {a: 1.0, b: 2.0}

        validator = SymbolicValidator(system)

        result = validator.validate(raise_on_error=False)
        assert result.is_valid


# ============================================================================
# Test Class 8: Physical Constraints
# ============================================================================


class TestPhysicalConstraints:
    """Test physical constraint validation"""

    def test_nan_parameter(self):
        """Test error for NaN parameter"""
        system = MockValidSystem()
        x, u = sp.symbols("x u")
        a = sp.symbols("a")

        system.state_vars = [x]
        system.control_vars = [u]
        system._f_sym = sp.Matrix([a * x + u])
        system.parameters = {a: np.nan}

        validator = SymbolicValidator(system)

        with pytest.raises(ValidationError, match="non-finite value"):
            validator.validate(raise_on_error=True)

    def test_inf_parameter(self):
        """Test error for infinite parameter"""
        system = MockValidSystem()
        x, u = sp.symbols("x u")
        a = sp.symbols("a")

        system.state_vars = [x]
        system.control_vars = [u]
        system._f_sym = sp.Matrix([a * x + u])
        system.parameters = {a: np.inf}

        validator = SymbolicValidator(system)

        with pytest.raises(ValidationError, match="non-finite value"):
            validator.validate(raise_on_error=True)

    def test_very_small_parameter_warning(self):
        """Test warning for very small parameters"""
        system = MockValidSystem()
        x, u = sp.symbols("x u")
        a = sp.symbols("a")

        system.state_vars = [x]
        system.control_vars = [u]
        system._f_sym = sp.Matrix([a * x + u])
        system.parameters = {a: 1e-15}

        validator = SymbolicValidator(system)

        with pytest.warns(UserWarning, match="very small"):
            result = validator.validate(raise_on_error=False)

        assert result.is_valid
        assert len(result.warnings) > 0

    def test_very_large_parameter_warning(self):
        """Test warning for very large parameters"""
        system = MockValidSystem()
        x, u = sp.symbols("x u")
        a = sp.symbols("a")

        system.state_vars = [x]
        system.control_vars = [u]
        system._f_sym = sp.Matrix([a * x + u])
        system.parameters = {a: 1e15}

        validator = SymbolicValidator(system)

        with pytest.warns(UserWarning, match="very large"):
            result = validator.validate(raise_on_error=False)

        assert result.is_valid


# ============================================================================
# Test Class 9: Naming Conventions
# ============================================================================


class TestNamingConventions:
    """Test naming convention validation"""

    def test_duplicate_variable_names(self):
        """Test error for duplicate variable names"""
        system = MockValidSystem()
        x1 = sp.symbols("x")
        x2 = sp.symbols("x")  # Same name!
        u = sp.symbols("u")

        system.state_vars = [x1, x2]
        system.control_vars = [u]

        validator = SymbolicValidator(system)

        with pytest.raises(ValidationError, match="Duplicate variable names"):
            validator.validate(raise_on_error=True)

    def test_second_order_bad_naming_warning(self):
        """Test warning for second-order system with poor naming"""
        system = MockSecondOrderSystem()
        x1, x2 = sp.symbols("x1 x2")  # No 'dot' in names
        u = sp.symbols("u")
        k, c = sp.symbols("k c")

        system.state_vars = [x1, x2]
        system.control_vars = [u]
        system._f_sym = sp.Matrix([-k * x1 - c * x2 + u])
        system.parameters = {k: 10.0, c: 0.5}
        system.order = 2

        validator = SymbolicValidator(system)

        with pytest.warns(UserWarning, match="don't follow.*pattern"):
            result = validator.validate(raise_on_error=False)

        assert result.is_valid

    def test_second_order_good_naming_no_warning(self):
        """Test no warning for properly named second-order system"""
        system = MockSecondOrderSystem()
        # Already uses q, q_dot naming

        validator = SymbolicValidator(system)

        result = validator.validate(raise_on_error=False)
        assert result.is_valid


# ============================================================================
# Test Class 10: ValidationResult Return
# ============================================================================


class TestValidationResultReturn:
    """Test ValidationResult return values"""

    def test_result_structure_valid(self):
        """Test ValidationResult structure for valid system"""
        system = MockValidSystem()
        validator = SymbolicValidator(system)

        result = validator.validate(raise_on_error=False)

        assert isinstance(result, SymbolicValidationResult)
        assert result.is_valid
        assert isinstance(result.errors, list)
        assert isinstance(result.warnings, list)
        assert isinstance(result.info, dict)
        assert len(result.errors) == 0

    def test_result_structure_invalid(self):
        """Test ValidationResult structure for invalid system"""
        system = MockValidSystem()
        system.state_vars = []  # Invalid

        validator = SymbolicValidator(system)
        result = validator.validate(raise_on_error=False)

        assert isinstance(result, SymbolicValidationResult)
        assert not result.is_valid
        assert len(result.errors) > 0

    def test_result_info_contains_dimensions(self):
        """Test that info contains dimension information"""
        system = MockValidSystem()
        validator = SymbolicValidator(system)

        result = validator.validate(raise_on_error=False)

        assert "nx" in result.info
        assert "nu" in result.info
        assert result.info["nx"] == 1
        assert result.info["nu"] == 1

    def test_result_info_autonomous_flag(self):
        """Test that info contains autonomous flag for autonomous systems"""
        system = MockAutonomousSystem()
        validator = SymbolicValidator(system)

        result = validator.validate(raise_on_error=False)

        assert "is_autonomous" in result.info
        assert result.info["is_autonomous"] is True
        assert result.info["nu"] == 0

    def test_result_info_contains_order(self):
        """Test that info contains order"""
        system = MockSecondOrderSystem()
        validator = SymbolicValidator(system)

        result = validator.validate(raise_on_error=False)

        assert "order" in result.info
        assert result.info["order"] == 2

    def test_result_info_linearity(self):
        """Test that info contains linearity flag"""
        system = MockValidSystem()
        validator = SymbolicValidator(system)

        result = validator.validate(raise_on_error=False)

        assert "is_linear" in result.info
        # -a*x + u is linear
        assert result.info["is_linear"] is True

    def test_result_info_nonlinear(self):
        """Test linearity detection for nonlinear system"""
        system = MockValidSystem()
        x, u = sp.symbols("x u")
        a = sp.symbols("a")

        system.state_vars = [x]
        system.control_vars = [u]
        system._f_sym = sp.Matrix([a * x**2 + u])  # Nonlinear
        system.parameters = {a: 1.0}

        validator = SymbolicValidator(system)
        result = validator.validate(raise_on_error=False)

        assert result.info["is_linear"] is False


# ============================================================================
# Test Class 11: Static Method
# ============================================================================


class TestStaticMethod:
    """Test static convenience method"""

    def test_static_validate_valid(self):
        """Test static method on valid system"""
        system = MockValidSystem()

        result = SymbolicValidator.validate_system(system, raise_on_error=False)
        assert result.is_valid

    def test_static_validate_valid_with_raise(self):
        """Test static method with raise_on_error=True"""
        system = MockValidSystem()

        result = SymbolicValidator.validate_system(system, raise_on_error=True)
        assert result.is_valid

    def test_static_validate_invalid(self):
        """Test static method on invalid system"""
        system = MockValidSystem()
        system.state_vars = []

        with pytest.raises(ValidationError):
            SymbolicValidator.validate_system(system, raise_on_error=True)

    def test_static_validate_invalid_no_raise(self):
        """Test static method returns result without raising"""
        system = MockValidSystem()
        system.state_vars = []

        result = SymbolicValidator.validate_system(system, raise_on_error=False)
        assert not result.is_valid

    def test_static_validate_autonomous(self):
        """Test static method on autonomous system"""
        system = MockAutonomousSystem()

        result = SymbolicValidator.validate_system(system, raise_on_error=False)
        assert result.is_valid
        assert result.info["is_autonomous"]


# ============================================================================
# Test Class 12: String Representations
# ============================================================================


class TestStringRepresentations:
    """Test __repr__ and __str__ methods"""

    def test_repr(self):
        """Test __repr__ output"""
        system = MockValidSystem()
        validator = SymbolicValidator(system)

        repr_str = repr(validator)

        assert "SymbolicValidator" in repr_str
        assert "MockValidSystem" in repr_str

    def test_str(self):
        """Test __str__ output"""
        system = MockValidSystem()
        validator = SymbolicValidator(system)

        str_repr = str(validator)

        assert "SymbolicValidator" in str_repr


# ============================================================================
# Test Class 13: Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests combining multiple validation checks"""

    def test_multiple_errors(self):
        """Test system with multiple errors"""
        system = MockValidSystem()

        # Introduce multiple errors
        system.state_vars = []  # Error 1
        system._f_sym = [1, 2, 3]  # Error 2: not a Matrix
        system.parameters = {"a": 1.0}  # Error 3: string key

        validator = SymbolicValidator(system)
        result = validator.validate(raise_on_error=False)

        assert not result.is_valid
        assert len(result.errors) >= 3

    def test_multiple_warnings(self):
        """Test system with multiple warnings"""
        system = MockValidSystem()
        x, u = sp.symbols("x u")
        a, b = sp.symbols("a b")

        system.state_vars = [x]
        system.control_vars = [u]
        system._f_sym = sp.Matrix([u])  # Warning 1: no state dependency
        system.parameters = {a: 1e-15, b: 1e15}  # Warning 2,3: extreme values

        validator = SymbolicValidator(system)
        result = validator.validate(raise_on_error=False)

        assert result.is_valid  # Valid despite warnings
        assert len(result.warnings) >= 2

    def test_complex_valid_system(self):
        """Test complex but valid system"""
        x1, x2 = sp.symbols("x1 x2", real=True)
        u1, u2 = sp.symbols("u1 u2", real=True)
        m, k, c = sp.symbols("m k c", real=True, positive=True)

        system = MockValidSystem()
        system.state_vars = [x1, x2]
        system.control_vars = [u1, u2]
        system._f_sym = sp.Matrix([x2, (-k * x1 - c * x2 + u1 + u2) / m])
        system.parameters = {m: 1.0, k: 10.0, c: 0.5}
        system.order = 1

        validator = SymbolicValidator(system)

        result = validator.validate(raise_on_error=False)
        assert result.is_valid
        assert len(result.errors) == 0

    def test_full_workflow_with_output(self):
        """Test system with output equation"""
        x1, x2 = sp.symbols("x1 x2", real=True)
        u = sp.symbols("u", real=True)
        y = sp.symbols("y", real=True)
        a = sp.symbols("a", real=True, positive=True)

        system = MockValidSystem()
        system.state_vars = [x1, x2]
        system.control_vars = [u]
        system.output_vars = [y]
        system._f_sym = sp.Matrix([x2, -a * x1 + u])
        system._h_sym = sp.Matrix([x1])  # Output is first state
        system.parameters = {a: 1.0}
        system.order = 1

        validator = SymbolicValidator(system)
        result = validator.validate(raise_on_error=False)

        assert result.is_valid
        assert result.info["has_output"]
        assert result.info["ny"] == 1

    def test_autonomous_system_integration(self):
        """Test autonomous system full validation"""
        system = MockAutonomous2D()
        validator = SymbolicValidator(system)

        result = validator.validate(raise_on_error=False)

        assert result.is_valid
        assert result.info["is_autonomous"]
        assert result.info["nu"] == 0
        assert result.info["nx"] == 2
        assert len(result.errors) == 0


# ============================================================================
# Test Class 14: Error Message Formatting
# ============================================================================


class TestErrorMessageFormatting:
    """Test error message formatting"""

    def test_error_message_contains_errors(self):
        """Test that error message contains all errors"""
        system = MockValidSystem()
        system.state_vars = []

        validator = SymbolicValidator(system)

        try:
            validator.validate(raise_on_error=True)
        except ValidationError as e:
            error_msg = str(e)
            assert "state_vars is empty" in error_msg
            assert "validation failed" in error_msg.lower()

    def test_error_message_contains_common_fixes(self):
        """Test that error message includes common fixes"""
        system = MockValidSystem()
        system.parameters = {"a": 1.0}  # String key

        validator = SymbolicValidator(system)

        try:
            validator.validate(raise_on_error=True)
        except ValidationError as e:
            error_msg = str(e)
            assert "COMMON FIXES" in error_msg


# ============================================================================
# Run Tests
# ============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
