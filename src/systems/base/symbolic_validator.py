"""
Symbolic Validator for SymbolicDynamicalSystem

Validates that symbolic system definitions are correct, complete, and consistent.

Checks:
- Required attributes (state_vars, control_vars, _f_sym)
- Type correctness (Symbols, Matrix, numeric parameters)
- Dimension consistency (nx, nu, order relationships)
- Symbolic expression validity (no undefined symbols)
- Parameter usage (no unused parameters, all used parameters defined)
- Physical constraints (positive masses, finite values)
- Naming conventions (no duplicates, proper derivative naming)

This class is completely standalone and can validate any object with
symbolic system attributes.
"""

from typing import List, Dict, Tuple, Optional, TYPE_CHECKING
import warnings
import sympy as sp
import numpy as np

if TYPE_CHECKING:
    from src.systems.base.symbolic_dynamical_system import SymbolicDynamicalSystem


class ValidationError(ValueError):
    """Raised when system validation fails"""
    pass


class SymbolicValidator:
    """
    Validates symbolic dynamical system definitions.
    
    Performs comprehensive validation of symbolic system components including
    variable definitions, dimensions, symbolic expressions, and parameter values.
    
    Example:
        >>> validator = SymbolicValidator()
        >>> validator.validate(system)  # Raises ValidationError if invalid
        >>> 
        >>> # Or check without raising
        >>> is_valid, errors, warnings = validator.check(system)
    """
    
    def __init__(self, strict: bool = True):
        """
        Initialize validator.
        
        Args:
            strict: If True, raise exception on validation failure.
                   If False, return errors without raising.
        """
        self.strict = strict
        self._errors: List[str] = []
        self._warnings: List[str] = []
    
    # ========================================================================
    # Public API
    # ========================================================================
    
    def validate(self, system: 'SymbolicDynamicalSystem') -> bool:
        """
        Validate system definition and raise exception if invalid.
        
        Args:
            system: System to validate
            
        Returns:
            True if valid
            
        Raises:
            ValidationError: If system is invalid (with detailed error messages)
            
        Example:
            >>> validator = SymbolicValidator()
            >>> try:
            >>>     validator.validate(system)
            >>>     print("System is valid!")
            >>> except ValidationError as e:
            >>>     print(f"Validation failed: {e}")
        """
        is_valid, errors, warnings_list = self.check(system)
        
        # Issue warnings
        if warnings_list:
            self._issue_warnings(warnings_list)
        
        # Raise if invalid
        if not is_valid:
            raise ValidationError(self._format_error_message(errors))
        
        return True
    
    def check(self, system: 'SymbolicDynamicalSystem') -> Tuple[bool, List[str], List[str]]:
        """
        Check system validity without raising exceptions.
        
        Args:
            system: System to check
            
        Returns:
            Tuple of (is_valid, errors, warnings)
            
        Example:
            >>> validator = SymbolicValidator(strict=False)
            >>> is_valid, errors, warnings = validator.check(system)
            >>> if not is_valid:
            >>>     for error in errors:
            >>>         print(f"Error: {error}")
        """
        self._errors = []
        self._warnings = []
        
        # Run all validation checks
        # CRITICAL: Check required attributes and types FIRST
        # If these fail, skip checks that would try to USE the invalid data
        self._validate_required_attributes(system)
        self._validate_types(system)
        
        # Only continue with deeper checks if basics pass
        if len(self._errors) == 0:
            self._validate_dimensions(system)
            self._validate_symbols(system)
            self._validate_parameters(system)
            self._validate_physical_constraints(system)
            self._validate_naming_conventions(system)
            self._check_usage_patterns(system)
        
        is_valid = len(self._errors) == 0
        return is_valid, self._errors.copy(), self._warnings.copy()
    
    # ========================================================================
    # Validation Checks
    # ========================================================================
    
    def _validate_required_attributes(self, system: 'SymbolicDynamicalSystem'):
        """Check that all required attributes are present and non-empty"""
        
        # Check state_vars exists and is not empty
        if not hasattr(system, 'state_vars'):
            self._errors.append("Missing required attribute 'state_vars'")
        elif not system.state_vars:
            self._errors.append("state_vars is empty - at least one state variable required")
        
        # Check control_vars exists and is not empty
        if not hasattr(system, 'control_vars'):
            self._errors.append("Missing required attribute 'control_vars'")
        elif not system.control_vars:
            self._errors.append("control_vars is empty - at least one control variable required")
        
        # Check _f_sym exists and is defined
        if not hasattr(system, '_f_sym'):
            self._errors.append("Missing required attribute '_f_sym'")
        elif system._f_sym is None:
            self._errors.append("_f_sym is not defined (is None)")
        
        # Check parameters exists (can be empty dict)
        if not hasattr(system, 'parameters'):
            self._errors.append("Missing required attribute 'parameters'")
        
        # Check order exists
        if not hasattr(system, 'order'):
            self._errors.append("Missing required attribute 'order'")
    
    def _validate_types(self, system: 'SymbolicDynamicalSystem'):
        """Check that all attributes have correct types"""
        
        # Check state_vars contains only Symbols
        if hasattr(system, 'state_vars') and system.state_vars:
            for i, var in enumerate(system.state_vars):
                if not isinstance(var, sp.Symbol):
                    self._errors.append(
                        f"state_vars[{i}] = {var} is not a SymPy Symbol "
                        f"(got {type(var).__name__})"
                    )
        
        # Check control_vars contains only Symbols
        if hasattr(system, 'control_vars') and system.control_vars:
            for i, var in enumerate(system.control_vars):
                if not isinstance(var, sp.Symbol):
                    self._errors.append(
                        f"control_vars[{i}] = {var} is not a SymPy Symbol "
                        f"(got {type(var).__name__})"
                    )
        
        # Check output_vars contains only Symbols (if defined)
        if hasattr(system, 'output_vars') and system.output_vars:
            for i, var in enumerate(system.output_vars):
                if not isinstance(var, sp.Symbol):
                    self._errors.append(
                        f"output_vars[{i}] = {var} is not a SymPy Symbol "
                        f"(got {type(var).__name__})"
                    )
        
        # Check _f_sym is a Matrix
        if hasattr(system, '_f_sym') and system._f_sym is not None:
            if not isinstance(system._f_sym, sp.Matrix):
                self._errors.append(
                    f"_f_sym must be sp.Matrix, got {type(system._f_sym).__name__}"
                )
        
        # Check _h_sym is a Matrix (if defined)
        if hasattr(system, '_h_sym') and system._h_sym is not None:
            if not isinstance(system._h_sym, sp.Matrix):
                self._errors.append(
                    f"_h_sym must be sp.Matrix, got {type(system._h_sym).__name__}"
                )
        
        # Check parameters dict has Symbol keys and numeric values
        if hasattr(system, 'parameters') and system.parameters:
            for key, value in system.parameters.items():
                if not isinstance(key, sp.Symbol):
                    self._errors.append(
                        f"Parameter key {key} is not a SymPy Symbol "
                        f"(got {type(key).__name__}). "
                        f"Use Symbol objects as keys: {{m: 1.0}} not {{'m': 1.0}}"
                    )
                
                if not isinstance(value, (int, float, np.number)):
                    self._errors.append(
                        f"Parameter value for {key} must be numeric, "
                        f"got {type(value).__name__}"
                    )
        
        # Check order is an integer
        if hasattr(system, 'order'):
            if not isinstance(system.order, int):
                self._errors.append(
                    f"order must be int, got {type(system.order).__name__}"
                )
    
    def _validate_dimensions(self, system: 'SymbolicDynamicalSystem'):
        """Check dimensional consistency"""
        
        # Get dimensions (handle missing attributes gracefully)
        nx = len(system.state_vars) if hasattr(system, 'state_vars') else 0
        nu = len(system.control_vars) if hasattr(system, 'control_vars') else 0
        order = system.order if hasattr(system, 'order') else 1
        
        # Check _f_sym dimensions
        if hasattr(system, '_f_sym') and system._f_sym is not None:
            if isinstance(system._f_sym, sp.Matrix):
                expected_rows = nx // order if order > 1 else nx
                actual_rows = system._f_sym.shape[0]
                
                if actual_rows != expected_rows:
                    self._errors.append(
                        f"_f_sym has {actual_rows} rows but expected {expected_rows} "
                        f"(nx={nx}, order={order}, nq={nx//order if order > 1 else nx})"
                    )
                
                if system._f_sym.shape[1] != 1:
                    self._errors.append(
                        f"_f_sym must be column vector (shape[1]=1), "
                        f"got shape {system._f_sym.shape}"
                    )
        
        # Check _h_sym dimensions (if defined)
        if hasattr(system, '_h_sym') and system._h_sym is not None:
            if isinstance(system._h_sym, sp.Matrix):
                if system._h_sym.shape[1] != 1:
                    self._errors.append(
                        f"_h_sym must be column vector (shape[1]=1), "
                        f"got shape {system._h_sym.shape}"
                    )
                
                # Check consistency with output_vars if defined
                if hasattr(system, 'output_vars') and system.output_vars:
                    ny_from_vars = len(system.output_vars)
                    ny_from_h = system._h_sym.shape[0]
                    
                    if ny_from_vars != ny_from_h:
                        self._errors.append(
                            f"output_vars has {ny_from_vars} elements but "
                            f"_h_sym has {ny_from_h} rows"
                        )
        
        # Check system order vs state dimension
        if order > 1:
            if nx % order != 0:
                self._errors.append(
                    f"For order={order} system, nx={nx} must be divisible by order. "
                    f"Got nx % order = {nx % order}. "
                    f"State should be [q, q̇, q̈, ...] with balanced partitions."
                )
        
        # Check reasonable order
        if hasattr(system, 'order'):
            if order < 1:
                self._errors.append(f"order must be >= 1, got {order}")
            elif order > 5:
                self._warnings.append(
                    f"order = {order} is unusually high. "
                    f"Are you sure this is correct? Most systems are order 1 or 2."
                )
    
    def _validate_symbols(self, system: 'SymbolicDynamicalSystem'):
        """Validate symbolic expressions"""
        
        # Only proceed if we have all required components AND they passed type checks
        if not (hasattr(system, 'state_vars') and hasattr(system, 'control_vars') 
                and hasattr(system, '_f_sym') and hasattr(system, 'parameters')):
            return
        
        if not (system.state_vars and system.control_vars 
                and system._f_sym is not None 
                and isinstance(system._f_sym, sp.Matrix)):
            return
        
        # Verify state_vars and control_vars are actually Symbols before using them
        # (If not, type validation will catch it, don't fail here too)
        if not all(isinstance(v, sp.Symbol) for v in system.state_vars):
            return
        if not all(isinstance(v, sp.Symbol) for v in system.control_vars):
            return
        if system.parameters and not all(isinstance(k, sp.Symbol) for k in system.parameters.keys()):
            return
        
        # Get all symbols in _f_sym
        try:
            f_symbols = system._f_sym.free_symbols
        except Exception:
            # If we can't get free_symbols, the expression is malformed
            # Other validation will catch this
            return
        
        # Expected symbols: state + control + parameters
        expected_symbols = set(
            system.state_vars + 
            system.control_vars + 
            list(system.parameters.keys())
        )
        
        # Check for undefined symbols
        undefined_symbols = f_symbols - expected_symbols
        if undefined_symbols:
            self._errors.append(
                f"_f_sym contains undefined symbols: {undefined_symbols}. "
                f"All symbols must be declared in state_vars, control_vars, or parameters."
            )
        
        # Check that dynamics depend on states (warning only)
        state_symbols = set(system.state_vars)
        if not (f_symbols & state_symbols):
            self._warnings.append(
                "_f_sym does not depend on any state variables. "
                "Is this intentional? (e.g., integrator system)"
            )
        
        # Validate output expression (if defined)
        if hasattr(system, '_h_sym') and system._h_sym is not None:
            if isinstance(system._h_sym, sp.Matrix):
                try:
                    h_symbols = system._h_sym.free_symbols
                except Exception:
                    return
                
                state_and_params = set(system.state_vars + list(system.parameters.keys()))
                
                # Output should NOT depend on control
                control_symbols = set(system.control_vars)
                if h_symbols & control_symbols:
                    self._errors.append(
                        f"_h_sym contains control variables {h_symbols & control_symbols}. "
                        f"Output h(x) should only depend on states, not controls."
                    )
                
                # Check for undefined symbols in output
                undefined = h_symbols - state_and_params
                if undefined:
                    self._errors.append(
                        f"_h_sym contains undefined symbols: {undefined}. "
                        f"Output can only use state_vars and parameters."
                    )
    
    def _validate_parameters(self, system: 'SymbolicDynamicalSystem'):
        """Validate parameter definitions and usage"""
        
        if not hasattr(system, 'parameters'):
            return
        
        # Check for unused parameters
        if (hasattr(system, '_f_sym') and system._f_sym is not None 
            and isinstance(system._f_sym, sp.Matrix) and system.parameters):
            
            param_symbols = set(system.parameters.keys())
            f_symbols = system._f_sym.free_symbols
            used_params = f_symbols & param_symbols
            unused_params = param_symbols - used_params
            
            # Check if unused in _h_sym before warning
            if unused_params and hasattr(system, '_h_sym') and system._h_sym is not None:
                if isinstance(system._h_sym, sp.Matrix):
                    h_symbols = system._h_sym.free_symbols
                    unused_params = unused_params - h_symbols
            
            if unused_params:
                self._warnings.append(
                    f"Parameters {unused_params} are defined but not used in _f_sym or _h_sym. "
                    f"Consider removing them or checking for typos."
                )
    
    def _validate_physical_constraints(self, system: 'SymbolicDynamicalSystem'):
        """Validate physical constraints on parameter values"""
        
        if not hasattr(system, 'parameters') or not system.parameters:
            return
        
        for symbol, value in system.parameters.items():
            param_name = str(symbol).lower()
            
            # Check for NaN or Inf
            if not np.isfinite(value):
                self._errors.append(
                    f"Parameter {symbol} has non-finite value: {value}. "
                    f"Parameters must be finite numbers."
                )
                continue
            
            # Check physical parameters are positive
            physical_params = ['m', 'mass', 'l', 'length', 'I', 'inertia', 'k', 'c']
            if any(name in param_name for name in physical_params):
                if value <= 0:
                    self._errors.append(
                        f"Physical parameter {symbol} = {value} should be positive. "
                        f"Negative or zero values are physically invalid."
                    )
            
            # Warn about very small or very large values
            if abs(value) < 1e-10:
                self._warnings.append(
                    f"Parameter {symbol} = {value} is very small (< 1e-10). "
                    f"This may cause numerical issues."
                )
            elif abs(value) > 1e10:
                self._warnings.append(
                    f"Parameter {symbol} = {value} is very large (> 1e10). "
                    f"Consider rescaling for numerical stability."
                )
    
    def _validate_naming_conventions(self, system: 'SymbolicDynamicalSystem'):
        """Validate naming conventions and check for duplicates"""
        
        if not (hasattr(system, 'state_vars') and hasattr(system, 'control_vars')):
            return
        
        # Only check if variables are actually Symbols (otherwise type validation handles it)
        if system.state_vars and not all(isinstance(v, sp.Symbol) for v in system.state_vars):
            return
        if system.control_vars and not all(isinstance(v, sp.Symbol) for v in system.control_vars):
            return
        
        # Collect all variable names
        all_vars = []
        if system.state_vars:
            all_vars.extend(system.state_vars)
        if system.control_vars:
            all_vars.extend(system.control_vars)
        if hasattr(system, 'output_vars') and system.output_vars:
            if all(isinstance(v, sp.Symbol) for v in system.output_vars):
                all_vars.extend(system.output_vars)
        
        # Check for duplicate names
        all_var_names = [str(var) for var in all_vars]
        if len(all_var_names) != len(set(all_var_names)):
            from collections import Counter
            counts = Counter(all_var_names)
            duplicates = [name for name, count in counts.items() if count > 1]
            self._errors.append(
                f"Duplicate variable names found: {duplicates}. "
                f"Each variable must have a unique name."
            )
        
        # Check naming convention for second-order systems
        if hasattr(system, 'order') and system.order == 2:
            if hasattr(system, 'state_vars') and len(system.state_vars) >= 2:
                state_names = [str(var) for var in system.state_vars]
                nq = len(system.state_vars) // 2
                
                # Check if second half uses derivative notation
                has_dot_pattern = any(
                    'dot' in name.lower() or "'" in name 
                    for name in state_names[nq:]
                )
                
                if not has_dot_pattern:
                    self._warnings.append(
                        f"Second-order system but state_vars don't follow [q, q̇] pattern. "
                        f"State names: {state_names}. "
                        f"Consider naming derivatives with '_dot' suffix for clarity."
                    )
    
    def _check_usage_patterns(self, system: 'SymbolicDynamicalSystem'):
        """Check for unusual usage patterns (warnings only)"""
        
        # Check if dynamics are linear or nonlinear
        if (hasattr(system, '_f_sym') and system._f_sym is not None 
            and hasattr(system, 'state_vars') and system.state_vars):
            
            if isinstance(system._f_sym, sp.Matrix):
                # Check if system is linear (useful info)
                is_linear = True
                for expr in system._f_sym:
                    for state in system.state_vars:
                        if sp.degree(expr, state) > 1:
                            is_linear = False
                            break
                    if not is_linear:
                        break
                
                # Just informational, not a warning
                # Could be exposed via a property later
    
    # ========================================================================
    # Helper Methods
    # ========================================================================
    
    def _issue_warnings(self, warnings_list: List[str]):
        """Issue Python warnings for validation warnings"""
        for warning in warnings_list:
            warnings.warn(f"System validation warning: {warning}", UserWarning)
    
    def _format_error_message(self, errors: List[str]) -> str:
        """Format error messages in a readable way"""
        msg = "System validation failed:\n"
        msg += "\n".join(f"  • {error}" for error in errors)
        msg += "\n\n" + "=" * 70
        msg += "\nCOMMON FIXES:"
        msg += "\n  1. Use Symbol objects as parameter keys: {m: 1.0} not {'m': 1.0}"
        msg += "\n  2. Ensure _f_sym is a sp.Matrix, not a list"
        msg += "\n  3. Check that state_vars and control_vars are lists of Symbols"
        msg += "\n  4. Verify system order matches state dimension (nx % order == 0)"
        msg += "\n  5. Ensure all symbols in expressions are defined"
        msg += "\n" + "=" * 70
        return msg
    
    # ========================================================================
    # Convenience Methods
    # ========================================================================
    
    @staticmethod
    def validate_system(system: 'SymbolicDynamicalSystem') -> bool:
        """
        Static convenience method for one-off validation.
        
        Args:
            system: System to validate
            
        Returns:
            True if valid
            
        Raises:
            ValidationError: If invalid
            
        Example:
            >>> SymbolicValidator.validate_system(my_system)
        """
        validator = SymbolicValidator(strict=True)
        return validator.validate(system)
    
    def __repr__(self) -> str:
        """String representation"""
        return f"SymbolicValidator(strict={self.strict})"
    
    def __str__(self) -> str:
        """Human-readable string"""
        return f"SymbolicValidator(strict={self.strict})"