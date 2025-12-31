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
SDE Validator - Validation for Stochastic Systems

Validates SDE system definitions including dimension compatibility,
symbol resolution, and noise structure validation.

Validation Checks:
- Drift validation (reuses SymbolicValidator)
- Diffusion-specific validation
- Noise structure validation (claimed vs. actual)
- Zero diffusion detection (warning)

Reuses:
- SymbolicValidator for comprehensive drift validation
- NoiseCharacterizer for noise structure analysis
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set
from typing_extensions import TypedDict

import sympy as sp

from src.types.symbolic import ParameterDict, SymbolicDiffusionMatrix, SymbolicMatrix
from src.systems.base.utils.stochastic.noise_analysis import NoiseCharacterizer
from src.types.utilities import SymbolicValidationResult
from src.systems.base.utils.symbolic_validator import SymbolicValidator

# ============================================================================
# Exceptions
# ============================================================================


class ValidationError(Exception):
    """Raised when SDE validation fails."""

    pass

class SDEValidationInfo(TypedDict):
    """
    Type-safe info dictionary for SDE validation results.
    
    Contains system dimensions and noise characteristics.
    """
    
    # System dimensions
    nx: int
    nw: int
    num_states: int
    num_controls: int
    num_parameters: int
    has_time_var: bool
    # Noise characteristics
    noise_type: str
    is_additive: bool
    is_multiplicative: bool
    is_diagonal: bool
    is_scalar: bool
    depends_on_state: bool
    depends_on_control: bool
    depends_on_time: bool


# ============================================================================
# SDE Validator
# ============================================================================


class SDEValidator:
    """
    Validates stochastic dynamical system definitions.

    Performs comprehensive validation including:
    - Drift validation (reuses SymbolicValidator)
    - Diffusion dimension compatibility
    - Symbol resolution
    - Type checking
    - Noise structure validation
    - Zero diffusion detection
    - Parameter validation (via SymbolicValidator)
    - Physical constraints (via SymbolicValidator)
    - Naming conventions (via SymbolicValidator)

    Examples
    --------
    >>> from sympy import symbols, Matrix
    >>> x1, x2, u = symbols('x1 x2 u')
    >>>
    >>> drift = Matrix([[x2], [-x1 + u]])
    >>> diffusion = Matrix([[0.1], [0.2]])
    >>>
    >>> validator = SDEValidator(drift, diffusion, [x1, x2], [u])
    >>> result = validator.validate()
    >>>
    >>> if result.is_valid:
    ...     print("System is valid!")
    ... else:
    ...     print(f"Errors: {result.errors}")
    """

    def __init__(
        self,
        drift_expr: SymbolicMatrix,
        diffusion_expr: SymbolicDiffusionMatrix,
        state_vars: List[sp.Symbol],
        control_vars: List[sp.Symbol],
        time_var: Optional[sp.Symbol] = None,
        parameters: Optional[ParameterDict] = None,
    ):
        """
        Initialize SDE validator.

        Parameters
        ----------
        drift_expr : SymbolicMatrix
            Drift vector f(x, u), shape (nx, 1)
        diffusion_expr : SymbolicDiffusionMatrix
            Diffusion matrix g(x, u), shape (nx, nw)
        state_vars : List[sp.Symbol]
            State variable symbols
        control_vars : List[sp.Symbol]
            Control variable symbols
        time_var : sp.Symbol, optional
            Time variable symbol (if time-varying)
        parameters : ParameterDict, optional
            System parameters

        Raises
        ------
        TypeError
            If inputs have incorrect types
        """
        # Type validation (fail fast)
        self._validate_input_types(drift_expr, diffusion_expr, state_vars, control_vars)

        self.drift = drift_expr
        self.diffusion = diffusion_expr
        self.state_vars = state_vars
        self.control_vars = control_vars
        self.time_var = time_var
        self.parameters = parameters or {}

        # Extract dimensions
        self.nx = len(state_vars)
        self.nw = diffusion_expr.shape[1]

        # Validation state
        self._errors: List[str] = []
        self._warnings: List[str] = []

    def _validate_input_types(
        self,
        drift_expr,
        diffusion_expr,
        state_vars,
        control_vars,
    ):
        """Validate input types (fail fast)."""
        if not isinstance(drift_expr, sp.Matrix):
            raise TypeError(f"drift_expr must be sp.Matrix, got {type(drift_expr).__name__}")

        if not isinstance(diffusion_expr, sp.Matrix):
            raise TypeError(
                f"diffusion_expr must be sp.Matrix, got {type(diffusion_expr).__name__}"
            )

        if not isinstance(state_vars, list):
            raise TypeError(f"state_vars must be list, got {type(state_vars).__name__}")

        if not isinstance(control_vars, list):
            raise TypeError(f"control_vars must be list, got {type(control_vars).__name__}")

        # Validate all elements are symbols
        for i, var in enumerate(state_vars):
            if not isinstance(var, sp.Symbol):
                raise TypeError(f"state_vars[{i}] must be sp.Symbol, got {type(var).__name__}")

        for i, var in enumerate(control_vars):
            if not isinstance(var, sp.Symbol):
                raise TypeError(f"control_vars[{i}] must be sp.Symbol, got {type(var).__name__}")

    def validate(
        self,
        claimed_noise_type: Optional[str] = None,
        raise_on_error: bool = False,
    ) -> SymbolicValidationResult:
        """
        Perform comprehensive SDE validation.

        Parameters
        ----------
        claimed_noise_type : str, optional
            User's claim about noise type ('additive', 'diagonal', 'scalar')
            If provided, validates claim matches actual structure
        raise_on_error : bool
            If True, raise ValidationError on validation failure

        Returns
        -------
        SymbolicValidationResult
            Validation results with errors, warnings, and info

        Raises
        ------
        ValidationError
            If validation fails and raise_on_error=True

        Examples
        --------
        >>> result = validator.validate()
        >>> if not result.is_valid:
        ...     print(f"Validation failed: {result.errors}")
        >>>
        >>> # Validate claimed noise type
        >>> result = validator.validate(claimed_noise_type='additive')
        >>>
        >>> # Raise exception on error
        >>> try:
        ...     result = validator.validate(raise_on_error=True)
        ... except ValidationError as e:
        ...     print(f"Validation failed: {e}")
        """
        # Reset state
        self._errors = []
        self._warnings = []

        # ✅ REUSE: Validate drift using SymbolicValidator
        self._validate_drift_with_symbolic_validator()

        # SDE-specific validation checks
        self._validate_diffusion_dimensions()
        self._validate_diffusion_symbols()
        self._validate_zero_diffusion()

        # Validate noise type claim if provided
        if claimed_noise_type is not None:
            self._validate_noise_type_claim(claimed_noise_type)

        # Determine validity
        is_valid = len(self._errors) == 0

        # Build result
        result = SymbolicValidationResult(
            is_valid=is_valid,
            errors=self._errors.copy(),
            warnings=self._warnings.copy(),
            info=self._build_info(),
        )

        # Raise if requested
        if not is_valid and raise_on_error:
            raise ValidationError(self._format_error_message())

        return result

    # ========================================================================
    # Validation Checks - Drift (Reuses SymbolicValidator)
    # ========================================================================

    def _validate_drift_with_symbolic_validator(self):
        """
        Reuse SymbolicValidator for comprehensive drift validation.

        This gives us for free:
        - Type validation for drift
        - Dimension checking
        - Symbol validation (undefined symbols)
        - Parameter validation (unused parameters)
        - Physical constraints (NaN, Inf, positive masses)
        - Naming conventions (duplicates)

        Note: We handle time_var separately since SymbolicValidator
        doesn't support it directly.
        """
        # If time_var is used, add it to parameters as a workaround
        # This allows SymbolicValidator to see it as a valid symbol
        params_with_time = self.parameters.copy()
        if self.time_var is not None:
            # Add time as a dummy parameter so SymbolicValidator doesn't complain
            # We'll filter out any warnings about unused time variable
            params_with_time[self.time_var] = 0.0  # Dummy value

        # Create minimal system for drift validation
        class DriftSystem:
            """Minimal mock system for SymbolicValidator."""

            def __init__(self, drift, states, controls, params):
                self.state_vars = states
                self.control_vars = controls
                self._f_sym = drift
                self.parameters = params
                self.order = 1
                self._h_sym = None
                self.output_vars = []

        # Create drift system
        drift_system = DriftSystem(self.drift, self.state_vars, self.control_vars, params_with_time)

        # Validate using SymbolicValidator
        try:
            drift_validator = SymbolicValidator(drift_system)
            drift_result = drift_validator.validate(raise_on_error=False)

            # Filter out warnings about time variable being unused
            filtered_warnings = []
            for warn in drift_result.warnings:
                # Skip warning if it's about the time variable being unused
                if self.time_var and str(self.time_var) in warn and "not used" in warn:
                    continue
                filtered_warnings.append(warn)

            # Merge errors and filtered warnings from drift validation
            # Prefix with "Drift:" for clarity
            self._errors.extend([f"Drift: {err}" for err in drift_result.errors])
            self._warnings.extend([f"Drift: {warn}" for warn in filtered_warnings])

        except Exception as e:
            # If SymbolicValidator itself fails, catch it
            self._errors.append(f"Drift validation failed: {str(e)}")

    # ========================================================================
    # Validation Checks - Diffusion (SDE-specific)
    # ========================================================================

    def _validate_diffusion_dimensions(self):
        """Validate diffusion matrix dimensions."""
        # Diffusion rows must match state dimension
        if self.diffusion.shape[0] != self.nx:
            self._errors.append(
                f"Diffusion rows ({self.diffusion.shape[0]}) must match "
                f"state dimension ({self.nx})"
            )

        # Diffusion must have at least one column
        if self.nw < 1:
            self._errors.append("Diffusion must have at least 1 noise source (nw >= 1)")

        # Warn if noise dimension exceeds state dimension
        if self.nw > self.nx:
            self._warnings.append(
                f"Number of noise sources ({self.nw}) exceeds state dimension ({self.nx}). "
                f"This is unusual - typically nw <= nx."
            )

        # Warn if very high-dimensional noise
        if self.nw > 10:
            self._warnings.append(
                f"Very high-dimensional noise (nw={self.nw}). "
                f"SDE solving can be computationally expensive."
            )

    def _validate_diffusion_symbols(self):
        """Validate symbols in diffusion expression."""
        # Collect allowed symbols
        allowed_symbols: Set[sp.Symbol] = set()
        allowed_symbols.update(self.state_vars)
        allowed_symbols.update(self.control_vars)
        allowed_symbols.update(self.parameters.keys())

        if self.time_var is not None:
            allowed_symbols.add(self.time_var)

        # Check diffusion symbols
        diffusion_symbols = self.diffusion.free_symbols
        undefined_diffusion = diffusion_symbols - allowed_symbols

        if undefined_diffusion:
            self._errors.append(
                f"Undefined symbols in diffusion: {sorted(str(s) for s in undefined_diffusion)}"
            )

    def _validate_zero_diffusion(self):
        """Check for zero or near-zero diffusion."""
        # Check if all elements are zero
        is_all_zero = all(
            self.diffusion[i, j] == 0
            for i in range(self.diffusion.shape[0])
            for j in range(self.diffusion.shape[1])
        )

        if is_all_zero:
            self._warnings.append(
                "Diffusion matrix is all zeros - this is an ODE, not an SDE. "
                "Consider using a deterministic system type instead."
            )

    def _validate_noise_type_claim(self, claimed_type: str):
        """
        Validate claimed noise type matches actual structure.

        Uses NoiseCharacterizer to analyze actual noise structure.
        """
        # ✅ REUSE: NoiseCharacterizer for analysis
        characterizer = NoiseCharacterizer(
            diffusion_expr=self.diffusion,
            state_vars=self.state_vars,
            control_vars=self.control_vars,
            time_var=self.time_var,
        )

        # Validate claim
        try:
            characterizer.validate_noise_type_claim(claimed_type)
        except ValueError as e:
            self._errors.append(f"Noise type validation failed: {str(e)}")

    # ========================================================================
    # Info Building
    # ========================================================================

    def _build_info(self) -> SDEValidationInfo:
        """Build info dictionary with system characteristics."""
        # ✅ REUSE: NoiseCharacterizer for analysis
        characterizer = NoiseCharacterizer(
            diffusion_expr=self.diffusion,
            state_vars=self.state_vars,
            control_vars=self.control_vars,
            time_var=self.time_var,
        )

        char = characterizer.characteristics

        return {
            # System dimensions
            "nx": self.nx,
            "nw": self.nw,
            "num_states": len(self.state_vars),
            "num_controls": len(self.control_vars),
            "num_parameters": len(self.parameters),
            "has_time_var": self.time_var is not None,
            # Noise characteristics
            "noise_type": char.noise_type.value,
            "is_additive": char.is_additive,
            "is_multiplicative": char.is_multiplicative,
            "is_diagonal": char.is_diagonal,
            "is_scalar": char.is_scalar,
            "depends_on_state": char.depends_on_state,
            "depends_on_control": char.depends_on_control,
            "depends_on_time": char.depends_on_time,
        }

    # ========================================================================
    # Error Formatting
    # ========================================================================

    def _format_error_message(self) -> str:
        """Format comprehensive error message."""
        lines = ["SDE validation failed:"]
        lines.append("")
        lines.append("Errors:")
        for error in self._errors:
            lines.append(f"  • {error}")

        if self._warnings:
            lines.append("")
            lines.append("Warnings:")
            for warning in self._warnings:
                lines.append(f"  • {warning}")

        lines.append("")
        lines.append("=" * 70)
        lines.append("COMMON FIXES:")
        lines.append("  1. Ensure drift and diffusion have correct dimensions")
        lines.append(
            "  2. Check that all symbols are defined in state_vars, control_vars, or parameters"
        )
        lines.append("  3. Use Symbol objects as parameter keys: {m: 1.0} not {'m': 1.0}")
        lines.append("  4. Verify noise type claim matches actual structure")
        lines.append("  5. Check for undefined symbols in expressions")
        lines.append("=" * 70)

        return "\n".join(lines)

    def __repr__(self) -> str:
        """String representation."""
        return f"SDEValidator(nx={self.nx}, nw={self.nw})"


# ============================================================================
# Convenience Functions
# ============================================================================


def validate_sde_system(
    drift_expr: SymbolicMatrix,
    diffusion_expr: SymbolicDiffusionMatrix,
    state_vars: List[sp.Symbol],
    control_vars: List[sp.Symbol],
    **kwargs,
) -> SymbolicValidationResult:
    """
    Convenience function for validating SDE systems.

    Parameters
    ----------
    drift_expr : SymbolicMatrix
        Drift vector
    diffusion_expr : SymbolicDiffusionMatrix
        Diffusion matrix
    state_vars : List[sp.Symbol]
        State variables
    control_vars : List[sp.Symbol]
        Control variables
    **kwargs
        Additional options (time_var, parameters, claimed_noise_type, raise_on_error)

    Returns
    -------
    SymbolicValidationResult
        Validation results

    Examples
    --------
    >>> result = validate_sde_system(drift, diffusion, [x], [u])
    >>> if result.is_valid:
    ...     print("Valid SDE system!")
    """
    validator = SDEValidator(
        drift_expr,
        diffusion_expr,
        state_vars,
        control_vars,
        time_var=kwargs.get("time_var"),
        parameters=kwargs.get("parameters"),
    )

    return validator.validate(
        claimed_noise_type=kwargs.get("claimed_noise_type"),
        raise_on_error=kwargs.get("raise_on_error", False),
    )