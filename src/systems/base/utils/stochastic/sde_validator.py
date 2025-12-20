"""
SDE Validator - Validation for Stochastic Systems

Extends SymbolicValidator with SDE-specific validation rules.

Additional checks for SDEs:
- Diffusion expression validity
- Noise type consistency
- SDE type validity
- Drift-diffusion compatibility
- Dimensional consistency

Reuses: SymbolicValidator for all base validation
"""

from typing import List, Tuple, Optional
import sympy as sp

from src.systems.base.utils.symbolic_validator import SymbolicValidator, ValidationError
from src.systems.base.utils.stochastic.noise_analysis import NoiseCharacterizer


class SDEValidationError(ValidationError):
    """Raised when SDE validation fails."""
    pass


class SDEValidator:
    """
    Validates stochastic dynamical system definitions.
    
    Extends SymbolicValidator with SDE-specific checks while
    reusing all base ODE validation logic.
    
    Examples
    --------
    >>> validator = SDEValidator()
    >>> validator.validate_sde_system(
    ...     drift, diffusion, state_vars, control_vars, noise_type='additive'
    ... )
    """
    
    def __init__(self, strict: bool = True):
        """
        Initialize SDE validator.
        
        Parameters
        ----------
        strict : bool
            If True, raise exceptions on validation failure
        """
        # ✅ REUSE: Compose with base validator
        self.base_validator = SymbolicValidator(strict=strict)
        self.strict = strict
        self._errors: List[str] = []
        self._warnings: List[str] = []
    
    def validate_sde_system(
        self,
        drift_expr: sp.Matrix,
        diffusion_expr: sp.Matrix,
        state_vars: List[sp.Symbol],
        control_vars: List[sp.Symbol],
        sde_type: str,
        claimed_noise_type: Optional[str] = None,
        parameters: Optional[dict] = None
    ) -> bool:
        """
        Validate complete SDE system.
        
        Parameters
        ----------
        drift_expr : sp.Matrix
            Drift vector f(x, u)
        diffusion_expr : sp.Matrix
            Diffusion matrix g(x, u)
        state_vars : List[sp.Symbol]
            State variables
        control_vars : List[sp.Symbol]
            Control variables
        sde_type : str
            'ito' or 'stratonovich'
        claimed_noise_type : str, optional
            User's claim about noise type
        parameters : dict, optional
            System parameters
        
        Returns
        -------
        bool
            True if valid
        
        Raises
        ------
        SDEValidationError
            If validation fails and strict=True
        """
        self._errors = []
        self._warnings = []
        
        # Run all validation checks
        self._validate_drift_expression(drift_expr, state_vars, control_vars)
        self._validate_diffusion_expression(diffusion_expr, state_vars)
        self._validate_sde_type(sde_type)
        self._validate_drift_diffusion_compatibility(drift_expr, diffusion_expr)
        
        if claimed_noise_type is not None:
            self._validate_noise_type_claim(
                diffusion_expr, state_vars, control_vars, claimed_noise_type
            )
        
        # Check validity
        is_valid = len(self._errors) == 0
        
        # Raise if invalid and strict
        if not is_valid and self.strict:
            raise SDEValidationError(self._format_error_message())
        
        return is_valid
    
    def _validate_drift_expression(
        self,
        drift_expr: sp.Matrix,
        state_vars: List[sp.Symbol],
        control_vars: List[sp.Symbol]
    ):
        """Validate drift expression (reuse base validator logic)."""
        # Basic type check
        if not isinstance(drift_expr, sp.Matrix):
            self._errors.append(
                f"drift_expr must be sp.Matrix, got {type(drift_expr).__name__}"
            )
            return
        
        # Shape check
        if drift_expr.shape[1] != 1:
            self._errors.append(
                f"drift_expr must be column vector (shape[1]=1), "
                f"got shape {drift_expr.shape}"
            )
        
        if drift_expr.shape[0] != len(state_vars):
            self._errors.append(
                f"drift_expr rows ({drift_expr.shape[0]}) must match "
                f"number of states ({len(state_vars)})"
            )
    
    def _validate_diffusion_expression(
        self,
        diffusion_expr: sp.Matrix,
        state_vars: List[sp.Symbol]
    ):
        """Validate diffusion expression structure."""
        # Type check
        if not isinstance(diffusion_expr, sp.Matrix):
            self._errors.append(
                f"diffusion_expr must be sp.Matrix, got {type(diffusion_expr).__name__}"
            )
            return
        
        # Shape check - must be (nx, nw)
        nx = len(state_vars)
        nw = diffusion_expr.shape[1]
        
        if diffusion_expr.shape[0] != nx:
            self._errors.append(
                f"diffusion_expr rows ({diffusion_expr.shape[0]}) must match "
                f"state dimension ({nx})"
            )
        
        if nw < 1:
            self._errors.append(
                "diffusion_expr must have at least 1 column (nw >= 1)"
            )
        
        # Warn if very high-dimensional noise
        if nw > nx:
            self._warnings.append(
                f"Number of Wiener processes ({nw}) exceeds state dimension ({nx}). "
                f"This is unusual - typically nw <= nx."
            )
        
        if nw > 10:
            self._warnings.append(
                f"Very high-dimensional noise (nw={nw}). "
                f"SDE solving can be expensive for many noise sources."
            )
    
    def _validate_sde_type(self, sde_type: str):
        """Validate SDE interpretation type."""
        valid_types = ['ito', 'stratonovich']
        if sde_type not in valid_types:
            self._errors.append(
                f"Invalid sde_type '{sde_type}'. "
                f"Must be one of {valid_types}"
            )
    
    def _validate_drift_diffusion_compatibility(
        self,
        drift_expr: sp.Matrix,
        diffusion_expr: sp.Matrix
    ):
        """Check that drift and diffusion are compatible."""
        # Must have same number of rows
        if drift_expr.shape[0] != diffusion_expr.shape[0]:
            self._errors.append(
                f"drift and diffusion must have same number of rows. "
                f"Got drift: {drift_expr.shape[0]}, diffusion: {diffusion_expr.shape[0]}"
            )
    
    def _validate_noise_type_claim(
        self,
        diffusion_expr: sp.Matrix,
        state_vars: List[sp.Symbol],
        control_vars: List[sp.Symbol],
        claimed_type: str
    ):
        """
        Validate that claimed noise_type matches actual structure.
        
        Uses NoiseCharacterizer to analyze actual structure.
        """
        # ✅ REUSE: NoiseCharacterizer for analysis
        characterizer = NoiseCharacterizer(
            diffusion_expr, state_vars, control_vars
        )
        
        try:
            # This will raise ValueError if inconsistent
            characterizer.validate_noise_type_claim(claimed_type)
        except ValueError as e:
            self._errors.append(str(e))
    
    def _format_error_message(self) -> str:
        """Format error messages."""
        msg = "SDE validation failed:\n"
        msg += "\n".join(f"  • {error}" for error in self._errors)
        if self._warnings:
            msg += "\n\nWarnings:\n"
            msg += "\n".join(f"  • {warning}" for warning in self._warnings)
        return msg
    
    def __repr__(self) -> str:
        """String representation."""
        return f"SDEValidator(strict={self.strict})"