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
"""

import pytest
import numpy as np
import sympy as sp

from src.systems.base.symbolic_validator import SymbolicValidator, ValidationError


# ============================================================================
# Mock System Classes for Testing
# ============================================================================


class MockValidSystem:
    """Valid system for testing"""
    
    def __init__(self):
        x = sp.symbols('x', real=True)
        u = sp.symbols('u', real=True)
        a = sp.symbols('a', real=True, positive=True)
        
        self.state_vars = [x]
        self.control_vars = [u]
        self._f_sym = sp.Matrix([-a * x + u])
        self._h_sym = None
        self.parameters = {a: 1.0}
        self.output_vars = []
        self.order = 1


class MockSecondOrderSystem:
    """Valid second-order system"""
    
    def __init__(self):
        q, q_dot = sp.symbols('q q_dot', real=True)
        u = sp.symbols('u', real=True)
        k, c = sp.symbols('k c', real=True, positive=True)
        
        self.state_vars = [q, q_dot]
        self.control_vars = [u]
        self._f_sym = sp.Matrix([-k * q - c * q_dot + u])
        self._h_sym = None
        self.parameters = {k: 10.0, c: 0.5}
        self.output_vars = []
        self.order = 2


# ============================================================================
# Test Class 1: Required Attributes
# ============================================================================


class TestRequiredAttributes:
    """Test validation of required attributes"""
    
    def test_valid_system_passes(self):
        """Test that valid system passes validation"""
        system = MockValidSystem()
        validator = SymbolicValidator()
        
        # Should not raise
        assert validator.validate(system) is True
    
    def test_missing_state_vars(self):
        """Test error when state_vars is missing"""
        system = MockValidSystem()
        delattr(system, 'state_vars')
        
        validator = SymbolicValidator()
        
        with pytest.raises(ValidationError, match="Missing required attribute 'state_vars'"):
            validator.validate(system)
    
    def test_empty_state_vars(self):
        """Test error when state_vars is empty"""
        system = MockValidSystem()
        system.state_vars = []
        
        validator = SymbolicValidator()
        
        with pytest.raises(ValidationError, match="state_vars is empty"):
            validator.validate(system)
    
    def test_missing_control_vars(self):
        """Test error when control_vars is missing"""
        system = MockValidSystem()
        delattr(system, 'control_vars')
        
        validator = SymbolicValidator()
        
        with pytest.raises(ValidationError, match="Missing required attribute 'control_vars'"):
            validator.validate(system)
    
    def test_empty_control_vars(self):
        """Test error when control_vars is empty"""
        system = MockValidSystem()
        system.control_vars = []
        
        validator = SymbolicValidator()
        
        with pytest.raises(ValidationError, match="control_vars is empty"):
            validator.validate(system)
    
    def test_missing_f_sym(self):
        """Test error when _f_sym is missing"""
        system = MockValidSystem()
        delattr(system, '_f_sym')
        
        validator = SymbolicValidator()
        
        with pytest.raises(ValidationError, match="Missing required attribute '_f_sym'"):
            validator.validate(system)
    
    def test_none_f_sym(self):
        """Test error when _f_sym is None"""
        system = MockValidSystem()
        system._f_sym = None
        
        validator = SymbolicValidator()
        
        with pytest.raises(ValidationError, match="_f_sym is not defined"):
            validator.validate(system)
    
    def test_missing_parameters(self):
        """Test error when parameters is missing"""
        system = MockValidSystem()
        delattr(system, 'parameters')
        
        validator = SymbolicValidator()
        
        with pytest.raises(ValidationError, match="Missing required attribute 'parameters'"):
            validator.validate(system)
    
    def test_missing_order(self):
        """Test error when order is missing"""
        system = MockValidSystem()
        delattr(system, 'order')
        
        validator = SymbolicValidator()
        
        with pytest.raises(ValidationError, match="Missing required attribute 'order'"):
            validator.validate(system)


# ============================================================================
# Test Class 2: Type Validation
# ============================================================================


class TestTypeValidation:
    """Test type validation"""
    
    def test_state_vars_not_symbols(self):
        """Test error when state_vars contains non-Symbols"""
        system = MockValidSystem()
        system.state_vars = ['x', 'y']  # Strings instead of Symbols
        # Also need to fix _f_sym to match
        system._f_sym = sp.Matrix([0, 0])
        
        validator = SymbolicValidator()
        
        with pytest.raises(ValidationError, match="is not a SymPy Symbol"):
            validator.validate(system)
    
    def test_control_vars_not_symbols(self):
        """Test error when control_vars contains non-Symbols"""
        system = MockValidSystem()
        system.control_vars = ['u']  # String instead of Symbol
        
        validator = SymbolicValidator()
        
        with pytest.raises(ValidationError, match="is not a SymPy Symbol"):
            validator.validate(system)
    
    def test_output_vars_not_symbols(self):
        """Test error when output_vars contains non-Symbols"""
        system = MockValidSystem()
        system.output_vars = ['y']  # String instead of Symbol
        
        validator = SymbolicValidator()
        
        with pytest.raises(ValidationError, match="is not a SymPy Symbol"):
            validator.validate(system)
    
    def test_f_sym_not_matrix(self):
        """Test error when _f_sym is not a Matrix"""
        system = MockValidSystem()
        system._f_sym = [sp.symbols('x')]  # List instead of Matrix
        
        validator = SymbolicValidator()
        
        with pytest.raises(ValidationError, match="_f_sym must be sp.Matrix"):
            validator.validate(system)
    
    def test_h_sym_not_matrix(self):
        """Test error when _h_sym is not a Matrix"""
        system = MockValidSystem()
        system._h_sym = [sp.symbols('x')]  # List instead of Matrix
        
        validator = SymbolicValidator()
        
        with pytest.raises(ValidationError, match="_h_sym must be sp.Matrix"):
            validator.validate(system)
    
    def test_parameter_key_not_symbol(self):
        """Test error when parameter key is not a Symbol"""
        system = MockValidSystem()
        system.parameters = {'a': 1.0}  # String key instead of Symbol
        
        validator = SymbolicValidator()
        
        with pytest.raises(ValidationError, match="Parameter key.*is not a SymPy Symbol"):
            validator.validate(system)
    
    def test_parameter_value_not_numeric(self):
        """Test error when parameter value is not numeric"""
        system = MockValidSystem()
        a = sp.symbols('a')
        x, u = sp.symbols('x u')
        
        # Need valid _f_sym that uses 'a'
        system.state_vars = [x]
        system.control_vars = [u]
        system._f_sym = sp.Matrix([a * x + u])
        system.parameters = {a: "1.0"}  # String value instead of number
        
        validator = SymbolicValidator()
        
        with pytest.raises(ValidationError, match="Parameter value.*must be numeric"):
            validator.validate(system)
    
    def test_order_not_integer(self):
        """Test error when order is not integer"""
        system = MockValidSystem()
        system.order = 1.5  # Float instead of int
        
        validator = SymbolicValidator()
        
        with pytest.raises(ValidationError, match="order must be int"):
            validator.validate(system)


# ============================================================================
# Test Class 3: Dimension Validation
# ============================================================================


class TestDimensionValidation:
    """Test dimension consistency checks"""
    
    def test_f_sym_wrong_rows_first_order(self):
        """Test error when _f_sym has wrong number of rows"""
        system = MockValidSystem()
        x, y = sp.symbols('x y')
        system.state_vars = [x, y]  # 2 states
        # But _f_sym still has 1 row!
        
        validator = SymbolicValidator()
        
        with pytest.raises(ValidationError, match="has.*rows but expected"):
            validator.validate(system)
    
    def test_f_sym_not_column_vector(self):
        """Test error when _f_sym is not a column vector"""
        system = MockValidSystem()
        x = sp.symbols('x')
        system._f_sym = sp.Matrix([[x, x]])  # Row vector instead of column
        
        validator = SymbolicValidator()
        
        with pytest.raises(ValidationError, match="must be column vector"):
            validator.validate(system)
    
    def test_h_sym_not_column_vector(self):
        """Test error when _h_sym is not a column vector"""
        system = MockValidSystem()
        x = sp.symbols('x')
        system._h_sym = sp.Matrix([[x, x]])  # Row vector
        
        validator = SymbolicValidator()
        
        with pytest.raises(ValidationError, match="must be column vector"):
            validator.validate(system)
    
    def test_output_vars_h_sym_mismatch(self):
        """Test error when output_vars and _h_sym dimensions don't match"""
        system = MockValidSystem()
        y1, y2 = sp.symbols('y1 y2')
        x = sp.symbols('x')
        
        system.output_vars = [y1, y2]  # 2 outputs
        system._h_sym = sp.Matrix([x])  # But only 1 row in _h_sym
        
        validator = SymbolicValidator()
        
        with pytest.raises(ValidationError, match="output_vars has.*elements but _h_sym has.*rows"):
            validator.validate(system)
    
    def test_order_not_divisible(self):
        """Test error when nx not divisible by order"""
        system = MockValidSystem()
        x1, x2, x3 = sp.symbols('x1 x2 x3')
        system.state_vars = [x1, x2, x3]  # 3 states
        system.order = 2  # But 3 is not divisible by 2!
        
        validator = SymbolicValidator()
        
        with pytest.raises(ValidationError, match="must be divisible by order"):
            validator.validate(system)
    
    def test_second_order_system_valid(self):
        """Test that valid second-order system passes"""
        system = MockSecondOrderSystem()
        validator = SymbolicValidator()
        
        # Should not raise
        assert validator.validate(system) is True
    
    def test_order_less_than_one(self):
        """Test error when order < 1"""
        system = MockValidSystem()
        system.order = 0
        
        validator = SymbolicValidator()
        
        with pytest.raises(ValidationError, match="order must be >= 1"):
            validator.validate(system)
    
    def test_order_very_large_warning(self):
        """Test warning when order is unusually high"""
        system = MockValidSystem()
        # Need to create enough state vars to match the order
        x_vars = [sp.symbols(f'x{i}') for i in range(10)]
        u = sp.symbols('u')
        
        system.state_vars = x_vars
        system.control_vars = [u]
        system._f_sym = sp.Matrix([u])  # Only 1 row for order=10 system
        system.order = 10  # Very high order
        system.parameters = {}
        
        validator = SymbolicValidator()
        
        with pytest.warns(UserWarning, match="order.*unusually high"):
            validator.validate(system)


# ============================================================================
# Test Class 4: Symbol Validation
# ============================================================================


class TestSymbolValidation:
    """Test symbolic expression validation"""
    
    def test_undefined_symbol_in_f_sym(self):
        """Test error when _f_sym contains undefined symbol"""
        system = MockValidSystem()
        mystery = sp.symbols('mystery')
        x = sp.symbols('x')
        system._f_sym = sp.Matrix([x + mystery])  # mystery not defined!
        
        validator = SymbolicValidator()
        
        with pytest.raises(ValidationError, match="undefined symbols"):
            validator.validate(system)
    
    def test_f_sym_no_state_dependency_warning(self):
        """Test warning when dynamics don't depend on states"""
        system = MockValidSystem()
        x, u = sp.symbols('x u')
        
        system.state_vars = [x]
        system.control_vars = [u]
        system._f_sym = sp.Matrix([u])  # Only depends on control!
        system.parameters = {}  # No parameters needed
        
        validator = SymbolicValidator()
        
        with pytest.warns(UserWarning, match="does not depend on any state"):
            validator.validate(system)
    
    def test_h_sym_contains_control(self):
        """Test error when output depends on control"""
        system = MockValidSystem()
        x, u = sp.symbols('x u')
        
        system.state_vars = [x]
        system.control_vars = [u]
        system._f_sym = sp.Matrix([x + u])
        system._h_sym = sp.Matrix([x + u])  # Depends on control!
        system.parameters = {}
        
        validator = SymbolicValidator()
        
        with pytest.raises(ValidationError, match="_h_sym contains control"):
            validator.validate(system)
    
    def test_h_sym_undefined_symbol(self):
        """Test error when _h_sym contains undefined symbol"""
        system = MockValidSystem()
        x = sp.symbols('x')
        mystery = sp.symbols('mystery')
        system._h_sym = sp.Matrix([x + mystery])
        
        validator = SymbolicValidator()
        
        with pytest.raises(ValidationError, match="_h_sym contains undefined symbols"):
            validator.validate(system)


# ============================================================================
# Test Class 5: Parameter Validation
# ============================================================================


class TestParameterValidation:
    """Test parameter validation"""
    
    def test_unused_parameter_warning(self):
        """Test warning for unused parameters"""
        system = MockValidSystem()
        unused = sp.symbols('unused')
        system.parameters[unused] = 5.0  # Not used in _f_sym
        
        validator = SymbolicValidator()
        
        with pytest.warns(UserWarning, match="defined but not used"):
            validator.validate(system)
    
    def test_parameter_used_in_h_sym_no_warning(self):
        """Test no warning when parameter is used in _h_sym"""
        system = MockValidSystem()
        x, u = sp.symbols('x u')
        a, b = sp.symbols('a b')
        
        system.state_vars = [x]
        system.control_vars = [u]
        system._f_sym = sp.Matrix([a * x + u])  # Uses 'a'
        system._h_sym = sp.Matrix([b * x])      # Uses 'b'
        system.parameters = {a: 1.0, b: 2.0}
        
        validator = SymbolicValidator()
        
        # Should not raise or warn (both parameters are used)
        validator.validate(system)


# ============================================================================
# Test Class 6: Physical Constraints
# ============================================================================


class TestPhysicalConstraints:
    """Test physical constraint validation"""
    
    def test_nan_parameter(self):
        """Test error for NaN parameter"""
        system = MockValidSystem()
        x, u = sp.symbols('x u')
        a = sp.symbols('a')
        
        system.state_vars = [x]
        system.control_vars = [u]
        system._f_sym = sp.Matrix([a * x + u])
        system.parameters = {a: np.nan}
        
        validator = SymbolicValidator()
        
        with pytest.raises(ValidationError, match="non-finite value"):
            validator.validate(system)
    
    def test_inf_parameter(self):
        """Test error for infinite parameter"""
        system = MockValidSystem()
        x, u = sp.symbols('x u')
        a = sp.symbols('a')
        
        system.state_vars = [x]
        system.control_vars = [u]
        system._f_sym = sp.Matrix([a * x + u])
        system.parameters = {a: np.inf}
        
        validator = SymbolicValidator()
        
        with pytest.raises(ValidationError, match="non-finite value"):
            validator.validate(system)
    
    def test_very_small_parameter_warning(self):
        """Test warning for very small parameters"""
        system = MockValidSystem()
        x, u = sp.symbols('x u')
        a = sp.symbols('a')
        
        system.state_vars = [x]
        system.control_vars = [u]
        system._f_sym = sp.Matrix([a * x + u])
        system.parameters = {a: 1e-15}
        
        validator = SymbolicValidator()
        
        with pytest.warns(UserWarning, match="very small"):
            validator.validate(system)
    
    def test_very_large_parameter_warning(self):
        """Test warning for very large parameters"""
        system = MockValidSystem()
        x, u = sp.symbols('x u')
        a = sp.symbols('a')
        
        system.state_vars = [x]
        system.control_vars = [u]
        system._f_sym = sp.Matrix([a * x + u])
        system.parameters = {a: 1e15}
        
        validator = SymbolicValidator()
        
        with pytest.warns(UserWarning, match="very large"):
            validator.validate(system)


# ============================================================================
# Test Class 7: Naming Conventions
# ============================================================================


class TestNamingConventions:
    """Test naming convention validation"""
    
    def test_duplicate_variable_names(self):
        """Test error for duplicate variable names"""
        system = MockValidSystem()
        x1 = sp.symbols('x')
        x2 = sp.symbols('x')  # Same name!
        u = sp.symbols('u')
        
        system.state_vars = [x1, x2]
        system.control_vars = [u]
        
        validator = SymbolicValidator()
        
        with pytest.raises(ValidationError, match="Duplicate variable names"):
            validator.validate(system)
    
    def test_second_order_bad_naming_warning(self):
        """Test warning for second-order system with poor naming"""
        system = MockSecondOrderSystem()
        x1, x2 = sp.symbols('x1 x2')  # No 'dot' in names
        u = sp.symbols('u')
        k, c = sp.symbols('k c')
        
        system.state_vars = [x1, x2]
        system.control_vars = [u]
        system._f_sym = sp.Matrix([-k * x1 - c * x2 + u])
        system.parameters = {k: 10.0, c: 0.5}
        system.order = 2
        
        validator = SymbolicValidator()
        
        with pytest.warns(UserWarning, match="don't follow.*pattern"):
            validator.validate(system)
    
    def test_second_order_good_naming_no_warning(self):
        """Test no warning for properly named second-order system"""
        system = MockSecondOrderSystem()
        # Already uses q, q_dot naming
        
        validator = SymbolicValidator()
        
        # Should not warn
        validator.validate(system)


# ============================================================================
# Test Class 8: Check Method
# ============================================================================


class TestCheckMethod:
    """Test non-raising check method"""
    
    def test_check_valid_system(self):
        """Test check method on valid system"""
        system = MockValidSystem()
        validator = SymbolicValidator(strict=False)
        
        is_valid, errors, warnings = validator.check(system)
        
        assert is_valid is True
        assert len(errors) == 0
        assert len(warnings) == 0
    
    def test_check_invalid_system(self):
        """Test check method on invalid system"""
        system = MockValidSystem()
        system.state_vars = []  # Invalid!
        
        validator = SymbolicValidator(strict=False)
        
        is_valid, errors, warnings = validator.check(system)
        
        assert is_valid is False
        assert len(errors) > 0
        assert "state_vars is empty" in errors[0]
    
    def test_check_with_warnings(self):
        """Test check method captures warnings"""
        system = MockValidSystem()
        # Create a system with warnings but no errors
        x_vars = [sp.symbols(f'x{i}') for i in range(10)]
        u = sp.symbols('u')
        
        system.state_vars = x_vars
        system.control_vars = [u]
        system._f_sym = sp.Matrix([u])  # Only 1 row for order=10
        system.order = 10  # Warning: unusually high
        system.parameters = {}
        
        validator = SymbolicValidator(strict=False)
        
        is_valid, errors, warnings = validator.check(system)
        
        assert is_valid is True  # No errors, only warnings
        assert len(errors) == 0
        assert len(warnings) > 0


# ============================================================================
# Test Class 9: Static Method
# ============================================================================


class TestStaticMethod:
    """Test static convenience method"""
    
    def test_static_validate_valid(self):
        """Test static method on valid system"""
        system = MockValidSystem()
        
        # Should not raise
        assert SymbolicValidator.validate_system(system) is True
    
    def test_static_validate_invalid(self):
        """Test static method on invalid system"""
        system = MockValidSystem()
        system.state_vars = []
        
        with pytest.raises(ValidationError):
            SymbolicValidator.validate_system(system)


# ============================================================================
# Test Class 10: String Representations
# ============================================================================


class TestStringRepresentations:
    """Test __repr__ and __str__ methods"""
    
    def test_repr(self):
        """Test __repr__ output"""
        validator = SymbolicValidator(strict=True)
        
        repr_str = repr(validator)
        
        assert 'SymbolicValidator' in repr_str
        assert 'strict=True' in repr_str
    
    def test_str(self):
        """Test __str__ output"""
        validator = SymbolicValidator(strict=False)
        
        str_repr = str(validator)
        
        assert 'SymbolicValidator' in str_repr
        assert 'strict=False' in str_repr


# ============================================================================
# Test Class 11: Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests combining multiple validation checks"""
    
    def test_multiple_errors(self):
        """Test system with multiple errors"""
        system = MockValidSystem()
        
        # Introduce multiple errors
        system.state_vars = []  # Error 1
        system._f_sym = [1, 2, 3]  # Error 2: not a Matrix
        system.parameters = {'a': 1.0}  # Error 3: string key
        
        validator = SymbolicValidator(strict=False)
        is_valid, errors, warnings = validator.check(system)
        
        assert is_valid is False
        assert len(errors) >= 3
    
    def test_complex_valid_system(self):
        """Test complex but valid system"""
        x1, x2 = sp.symbols('x1 x2', real=True)
        u1, u2 = sp.symbols('u1 u2', real=True)
        m, k, c = sp.symbols('m k c', real=True, positive=True)
        
        system = MockValidSystem()
        system.state_vars = [x1, x2]
        system.control_vars = [u1, u2]
        system._f_sym = sp.Matrix([
            x2,
            (-k * x1 - c * x2 + u1 + u2) / m
        ])
        system.parameters = {m: 1.0, k: 10.0, c: 0.5}
        system.order = 1
        
        validator = SymbolicValidator()
        
        # Should pass all validations
        assert validator.validate(system) is True


# ============================================================================
# Run Tests
# ============================================================================


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])