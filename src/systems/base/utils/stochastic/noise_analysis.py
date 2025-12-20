"""
Noise Structure Analysis for Stochastic Systems

Analyzes symbolic diffusion expressions to determine noise characteristics
and recommend efficient specialized solvers.

This module is PURE ANALYSIS - no code generation, no evaluation.
Just symbolic analysis using SymPy.

Components:
    - NoiseType: Enum for noise classifications
    - SDEType: Enum for SDE interpretations
    - NoiseCharacteristics: Dataclass with analysis results
    - NoiseCharacterizer: Analysis engine (pure SymPy)

Reuses: Nothing (self-contained SymPy analysis)
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Set
import sympy as sp


# ============================================================================
# Enumerations
# ============================================================================

class NoiseType(Enum):
    """
    Classification of stochastic noise structure.
    
    Determines which specialized SDE solvers can be used efficiently.
    """
    ADDITIVE = "additive"              # g(x,u,t) = constant (most efficient)
    MULTIPLICATIVE = "multiplicative"  # g(x,u,t) depends on x
    DIAGONAL = "diagonal"              # Independent noise sources
    SCALAR = "scalar"                  # Single Wiener process
    GENERAL = "general"                # Full matrix, state-dependent


class SDEType(Enum):
    """
    Stochastic differential equation interpretation.
    
    Attributes
    ----------
    ITO : str
        Itô calculus - standard in probability theory and finance
    STRATONOVICH : str
        Stratonovich calculus - more natural for physics applications
    """
    ITO = "ito"
    STRATONOVICH = "stratonovich"


# ============================================================================
# Analysis Results Container
# ============================================================================

@dataclass
class NoiseCharacteristics:
    """
    Container for noise structure analysis results.
    
    This dataclass stores all information about the noise structure,
    enabling automatic solver selection and optimization.
    
    Attributes
    ----------
    noise_type : NoiseType
        Classified noise type
    num_wiener : int
        Number of independent Wiener processes (nw)
    is_additive : bool
        True if diffusion is constant (doesn't depend on state/control/time)
    is_multiplicative : bool
        True if diffusion depends on state
    is_diagonal : bool
        True if noise sources don't couple (diagonal diffusion matrix)
    is_scalar : bool
        True if single Wiener process (nw = 1)
    depends_on_state : bool
        True if diffusion depends on any state variable
    depends_on_control : bool
        True if diffusion depends on any control variable
    depends_on_time : bool
        True if diffusion depends on time
    state_dependencies : Set[sp.Symbol]
        Which specific state variables appear in diffusion
    control_dependencies : Set[sp.Symbol]
        Which specific control variables appear in diffusion
    """
    noise_type: NoiseType
    num_wiener: int
    is_additive: bool
    is_multiplicative: bool
    is_diagonal: bool
    is_scalar: bool
    depends_on_state: bool
    depends_on_control: bool
    depends_on_time: bool
    state_dependencies: Set[sp.Symbol]
    control_dependencies: Set[sp.Symbol]
    
    def recommended_solvers(self, backend: str) -> List[str]:
        """
        Recommend efficient solvers based on noise structure.
        
        Parameters
        ----------
        backend : str
            Integration backend ('jax', 'torch', 'numpy')
        
        Returns
        -------
        List[str]
            Recommended solver names, ordered by efficiency/accuracy
        
        Examples
        --------
        >>> if char.is_additive:
        ...     solvers = char.recommended_solvers('jax')
        ...     print(solvers)  # ['sea', 'shark', 'sra1']
        """
        if backend == 'jax':
            # Diffrax recommendations
            if self.is_additive:
                # Specialized additive-noise solvers
                return ['sea', 'shark', 'sra1']
            elif self.is_scalar:
                # Single noise source
                return ['euler_heun', 'heun', 'reversible_heun']
            elif self.is_diagonal:
                # Independent noise sources
                return ['euler_heun', 'spark']
            else:
                # General noise
                return ['euler_heun', 'spark', 'general_shark']
        
        elif backend == 'torch':
            # TorchSDE recommendations
            if self.is_additive:
                return ['euler', 'milstein', 'srk']
            elif self.is_scalar or self.is_diagonal:
                return ['milstein', 'srk']
            else:
                return ['euler', 'srk', 'reversible_heun']
        
        elif backend == 'numpy':
            # Julia/DiffEqPy recommendations
            if self.is_additive:
                # High-order methods for additive noise
                return ['SRA1', 'SRA2', 'SRA3']
            else:
                # General solvers
                return ['SOSRI', 'SRI', 'SRIW1', 'SRIW2']
        
        return []


# ============================================================================
# Noise Characterizer - Pure Analysis Engine
# ============================================================================

class NoiseCharacterizer:
    """
    Analyzes symbolic diffusion expressions to determine noise structure.
    
    This is PURE ANALYSIS using only SymPy - no code generation,
    no numerical evaluation, no backend dependencies.
    
    The analysis results enable:
    - Automatic selection of efficient specialized solvers
    - Validation of noise type claims
    - Optimization opportunities identification
    
    Examples
    --------
    >>> # Additive noise (constant)
    >>> diffusion = sp.Matrix([[0.1], [0.2]])
    >>> char = NoiseCharacterizer(diffusion, [x1, x2], [u])
    >>> char.analyze()
    >>> print(char.characteristics.noise_type)
    NoiseType.ADDITIVE
    >>> 
    >>> # Multiplicative noise (state-dependent)
    >>> diffusion = sp.Matrix([[0.1 * x1]])
    >>> char = NoiseCharacterizer(diffusion, [x1], [u])
    >>> char.analyze()
    >>> print(char.characteristics.is_multiplicative)
    True
    """
    
    def __init__(
        self,
        diffusion_expr: sp.Matrix,
        state_vars: List[sp.Symbol],
        control_vars: List[sp.Symbol],
        time_var: Optional[sp.Symbol] = None
    ):
        """
        Initialize noise characterizer.
        
        Parameters
        ----------
        diffusion_expr : sp.Matrix
            Symbolic diffusion matrix g(x, u, t), shape (nx, nw)
        state_vars : List[sp.Symbol]
            State variable symbols
        control_vars : List[sp.Symbol]
            Control variable symbols
        time_var : sp.Symbol, optional
            Time variable symbol (if time-varying)
        """
        self.diffusion_expr = diffusion_expr
        self.state_vars = state_vars
        self.control_vars = control_vars
        self.time_var = time_var
        
        # Extract dimensions
        self.nx = diffusion_expr.shape[0]
        self.nw = diffusion_expr.shape[1]
        
        # Perform analysis (lazy - only when accessed)
        self._characteristics = None
    
    @property
    def characteristics(self) -> NoiseCharacteristics:
        """
        Get noise characteristics (cached after first analysis).
        
        Returns
        -------
        NoiseCharacteristics
            Analysis results
        """
        if self._characteristics is None:
            self._characteristics = self.analyze()
        return self._characteristics
    
    def analyze(self) -> NoiseCharacteristics:
        """
        Analyze diffusion expression structure.
        
        Returns
        -------
        NoiseCharacteristics
            Complete analysis of noise structure
        
        Examples
        --------
        >>> char = NoiseCharacterizer(diffusion, states, controls)
        >>> results = char.analyze()
        >>> if results.is_additive:
        ...     print("Use specialized additive-noise solver!")
        """
        # Extract free symbols from diffusion
        diffusion_syms = self.diffusion_expr.free_symbols
        
        # Create sets for comparison
        state_syms = set(self.state_vars)
        control_syms = set(self.control_vars)
        time_syms = {self.time_var} if self.time_var else set()
        
        # Find specific dependencies
        state_deps = diffusion_syms & state_syms
        control_deps = diffusion_syms & control_syms
        time_deps = diffusion_syms & time_syms
        
        # Classify dependencies
        depends_on_state = bool(state_deps)
        depends_on_control = bool(control_deps)
        depends_on_time = bool(time_deps)
        
        # Determine if additive (most important classification)
        is_additive = not (depends_on_state or depends_on_control or depends_on_time)
        is_multiplicative = depends_on_state
        
        # Check diagonal structure
        is_diagonal = self._check_diagonal()
        
        # Check scalar noise
        is_scalar = (self.nw == 1)
        
        # Determine overall noise type
        noise_type = self._classify_noise_type(
            is_additive, is_multiplicative, is_diagonal, is_scalar
        )
        
        return NoiseCharacteristics(
            noise_type=noise_type,
            num_wiener=self.nw,
            is_additive=is_additive,
            is_multiplicative=is_multiplicative,
            is_diagonal=is_diagonal,
            is_scalar=is_scalar,
            depends_on_state=depends_on_state,
            depends_on_control=depends_on_control,
            depends_on_time=depends_on_time,
            state_dependencies=state_deps,
            control_dependencies=control_deps,
        )
    
    def _check_diagonal(self) -> bool:
        """
        Check if diffusion matrix is diagonal.
        
        Returns
        -------
        bool
            True if all off-diagonal elements are zero
        """
        # Can only be diagonal if square
        if self.nw != self.nx:
            return False
        
        # Check all off-diagonal elements
        for i in range(self.nx):
            for j in range(self.nw):
                if i != j:
                    if self.diffusion_expr[i, j] != 0:
                        return False
        
        return True
    
    def _classify_noise_type(
        self,
        is_additive: bool,
        is_multiplicative: bool,
        is_diagonal: bool,
        is_scalar: bool
    ) -> NoiseType:
        """
        Classify overall noise type based on characteristics.
        
        Priority order:
        1. Additive (most specialized)
        2. Scalar (single noise source)
        3. Diagonal (independent sources)
        4. Multiplicative (state-dependent)
        5. General (fallback)
        """
        if is_additive:
            return NoiseType.ADDITIVE
        elif is_scalar:
            return NoiseType.SCALAR
        elif is_diagonal:
            return NoiseType.DIAGONAL
        elif is_multiplicative:
            return NoiseType.MULTIPLICATIVE
        else:
            return NoiseType.GENERAL
    
    def validate_noise_type_claim(self, claimed_type: str) -> bool:
        """
        Validate that a claimed noise_type matches actual structure.
        
        Parameters
        ----------
        claimed_type : str
            User's claim about noise type ('additive', 'diagonal', etc.)
        
        Returns
        -------
        bool
            True if claim matches analysis
        
        Raises
        ------
        ValueError
            If claimed type contradicts analysis
        
        Examples
        --------
        >>> char = NoiseCharacterizer(diffusion, states, controls)
        >>> char.validate_noise_type_claim('additive')  # Validates
        """
        char = self.characteristics
        
        if claimed_type == 'additive' and not char.is_additive:
            raise ValueError(
                f"Claimed noise_type='additive' but diffusion depends on: "
                f"state={char.depends_on_state}, "
                f"control={char.depends_on_control}, "
                f"time={char.depends_on_time}"
            )
        
        if claimed_type == 'diagonal' and not char.is_diagonal:
            raise ValueError(
                "Claimed noise_type='diagonal' but diffusion has "
                "off-diagonal elements (coupling between noise sources)"
            )
        
        if claimed_type == 'scalar' and not char.is_scalar:
            raise ValueError(
                f"Claimed noise_type='scalar' but nw={char.num_wiener} > 1"
            )
        
        return True
    
    def get_optimization_hints(self) -> Dict[str, any]:
        """
        Provide optimization hints based on noise structure.
        
        Returns
        -------
        Dict[str, any]
            Optimization opportunities and recommendations
        
        Examples
        --------
        >>> hints = char.get_optimization_hints()
        >>> print(hints['can_precompute'])
        True  # For additive noise
        """
        char = self.characteristics
        
        return {
            'can_precompute_diffusion': char.is_additive,
            'can_use_diagonal_solver': char.is_diagonal,
            'can_use_scalar_solver': char.is_scalar,
            'complexity': self._estimate_complexity(),
            'recommended_backends': self._recommend_backends(),
        }
    
    def _estimate_complexity(self) -> str:
        """Estimate computational complexity of diffusion evaluation."""
        char = self.characteristics
        
        if char.is_additive:
            return "O(1) - constant, precomputable"
        elif char.is_scalar:
            return "O(nx) - scalar multiplication"
        elif char.is_diagonal:
            return "O(nx) - element-wise"
        else:
            return "O(nx * nw) - full matrix"
    
    def _recommend_backends(self) -> List[str]:
        """Recommend backends based on noise structure."""
        char = self.characteristics
        
        # All backends support general noise
        recommended = ['jax', 'torch', 'numpy']
        
        # JAX has best specialized solvers for additive
        if char.is_additive:
            recommended = ['jax', 'numpy', 'torch']  # JAX first
        
        return recommended
    
    def __repr__(self) -> str:
        """String representation."""
        if self._characteristics is None:
            return "NoiseCharacterizer(not yet analyzed)"
        return f"NoiseCharacterizer(type={self.characteristics.noise_type.value})"