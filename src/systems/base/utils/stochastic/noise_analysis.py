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
Noise Structure Analysis for Stochastic Systems
================================================

Analyzes symbolic diffusion expressions to determine noise characteristics
and recommend efficient specialized solvers.

This module is PURE ANALYSIS - no code generation, no evaluation.
Just symbolic analysis using SymPy.

Components:
    - NoiseType: Enum for noise classifications
    - SDEType: Enum for SDE interpretations
    - NoiseCharacteristics: Dataclass with analysis results
    - NoiseCharacterizer: Analysis engine (pure SymPy)

Classification Hierarchy (CORRECTED)
------------------------------------
The noise type classification follows this priority order:

1. **ADDITIVE**: Constant diffusion (no state/control/time dependence)
   - Most optimizable: can precompute g once
   - Enables specialized solvers (SEA, SHARK, SRA)
   - Example: g = [[σ₁, 0], [0, σ₂]]

2. **SCALAR**: Single noise source (nw = 1), not constant
   - Simple structure: g ∈ ℝⁿˣ×¹
   - May be state-dependent: g = σ*x
   - Simpler than full matrix operations
   - Example: g = [[σ*x]]

3. **DIAGONAL**: Diagonal matrix (nw = nx), independent noise sources
   - No coupling between noise sources: g[i,j] = 0 for i ≠ j
   - Element-wise operations possible
   - May be state-dependent
   - Example: g = [[σ₁*x₁, 0], [0, σ₂*x₂]]

4. **MULTIPLICATIVE**: State-dependent, non-diagonal, non-scalar
   - Noise intensity varies with state
   - Cannot be precomputed
   - Requires re-evaluation each step
   - Example: g = [[σ₁*x₁, σ₂*x₂], [σ₃*x₁, σ₄*x₂]] (non-diagonal)

5. **GENERAL**: Fallback for other cases
   - Complex coupling or special structure
   - Least optimizable

Key Fix
-------
Previous version had bug where MULTIPLICATIVE was unreachable:

    # WRONG (old code):
    elif is_multiplicative and not is_diagonal and not is_scalar:
        return NoiseType.GENERAL  # ← Bug! Should be MULTIPLICATIVE
    elif is_multiplicative:
        return NoiseType.MULTIPLICATIVE  # ← Unreachable!

    # CORRECT (fixed):
    elif is_multiplicative:
        return NoiseType.MULTIPLICATIVE  # ← Now reachable!

This makes MULTIPLICATIVE meaningful for non-diagonal, non-scalar, state-dependent noise.

Reuses: Nothing (self-contained SymPy analysis)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set

import sympy as sp

from src.types.backends import Backend, NoiseType

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
        Classified noise type (ADDITIVE, SCALAR, DIAGONAL, MULTIPLICATIVE, GENERAL)
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

    Examples
    --------
    >>> char = NoiseCharacteristics(...)
    >>> if char.is_additive:
    ...     G = precompute_constant_noise()  # Optimization!
    >>>
    >>> solvers = char.recommended_solvers('jax')
    >>> print(solvers)  # ['sea', 'shark'] for additive
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

    def recommended_solvers(self, backend: Backend) -> List[str]:
        """
        Recommend efficient solvers based on noise structure.

        Parameters
        ----------
        backend : Backend
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
        if backend == "jax":
            # Diffrax recommendations
            if self.is_additive:
                # Specialized additive-noise solvers
                return ["sea", "shark", "sra1"]
            if self.is_scalar:
                # Single noise source
                return ["euler_heun", "heun", "reversible_heun"]
            if self.is_diagonal:
                # Independent noise sources
                return ["euler_heun", "spark"]
            if self.noise_type == NoiseType.MULTIPLICATIVE:
                # Multiplicative-specific solvers
                return ["euler_heun", "reversible_heun", "spark"]
            # General noise
            return ["euler_heun", "spark", "general_shark"]

        if backend == "torch":
            # TorchSDE recommendations
            if self.is_additive:
                return ["euler", "milstein", "srk"]
            if self.is_scalar or self.is_diagonal:
                return ["milstein", "srk"]
            if self.noise_type == NoiseType.MULTIPLICATIVE:
                return ["srk", "reversible_heun"]
            return ["euler", "srk", "reversible_heun"]

        if backend == "numpy":
            # Julia/DiffEqPy recommendations
            if self.is_additive:
                # High-order methods for additive noise
                return ["SRA1", "SRA2", "SRA3"]
            if self.noise_type == NoiseType.MULTIPLICATIVE:
                return ["SRIW1", "SRIW2", "SRI"]
            # General solvers
            return ["SOSRI", "SRI", "SRIW1", "SRIW2"]

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
    >>> results = char.analyze()
    >>> print(results.noise_type)
    NoiseType.ADDITIVE
    >>>
    >>> # Multiplicative noise (state-dependent, non-diagonal)
    >>> diffusion = sp.Matrix([[0.1 * x1, 0.05 * x2]])
    >>> char = NoiseCharacterizer(diffusion, [x1, x2], [u])
    >>> results = char.analyze()
    >>> print(results.noise_type)
    NoiseType.MULTIPLICATIVE
    >>> print(results.is_multiplicative)
    True
    """

    def __init__(
        self,
        diffusion_expr: sp.Matrix,
        state_vars: List[sp.Symbol],
        control_vars: List[sp.Symbol],
        time_var: Optional[sp.Symbol] = None,
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

        Examples
        --------
        >>> x1, x2 = sp.symbols('x1 x2')
        >>> u = sp.symbols('u')
        >>> diffusion = sp.Matrix([[0.1], [0.2]])
        >>> char = NoiseCharacterizer(diffusion, [x1, x2], [u])
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

    @classmethod
    def from_dict(cls, config: Dict) -> "NoiseCharacterizer":
        """
        Create NoiseCharacterizer from configuration dictionary.

        This is a convenience factory method for creating characterizers from
        dictionaries, such as test fixtures or saved configurations.

        Parameters
        ----------
        config : Dict
            Configuration dictionary with keys:
            - "diffusion" or "diffusion_expr": Symbolic diffusion matrix
            - "state_vars": List of state symbols
            - "control_vars": List of control symbols
            - "time_var" (optional): Time symbol

        Returns
        -------
        NoiseCharacterizer
            Initialized characterizer

        Examples
        --------
        >>> config = {
        ...     "diffusion": sp.Matrix([[0.1*x]]),
        ...     "state_vars": [x],
        ...     "control_vars": [u]
        ... }
        >>> char = NoiseCharacterizer.from_dict(config)
        >>> result = char.analyze()

        With test fixtures:
        >>> @pytest.fixture
        ... def my_noise():
        ...     return {"diffusion": ..., "state_vars": ..., "control_vars": ...}
        >>>
        >>> def test_something(my_noise):
        ...     char = NoiseCharacterizer.from_dict(my_noise)
        ...     # or equivalently:
        ...     char = NoiseCharacterizer(**my_noise)  # Now works!
        """
        # Handle both "diffusion" and "diffusion_expr" keys
        diffusion_expr = config.get("diffusion_expr") or config.get("diffusion")

        if diffusion_expr is None:
            raise ValueError(
                "Configuration dictionary must contain 'diffusion' or 'diffusion_expr' key",
            )

        return cls(
            diffusion_expr=diffusion_expr,
            state_vars=config["state_vars"],
            control_vars=config["control_vars"],
            time_var=config.get("time_var"),
        )

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
        is_scalar = self.nw == 1

        # Determine overall noise type (FIXED LOGIC)
        noise_type = self._classify_noise_type(
            is_additive, is_multiplicative, is_diagonal, is_scalar,
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

        Examples
        --------
        >>> # Diagonal: [[σ₁, 0], [0, σ₂]]
        >>> diffusion = sp.Matrix([[0.1, 0], [0, 0.2]])
        >>> char = NoiseCharacterizer(diffusion, [x1, x2], [])
        >>> char._check_diagonal()
        True
        >>>
        >>> # Non-diagonal: [[σ₁, σ₂], [σ₃, σ₄]]
        >>> diffusion = sp.Matrix([[0.1, 0.05], [0.03, 0.2]])
        >>> char = NoiseCharacterizer(diffusion, [x1, x2], [])
        >>> char._check_diagonal()
        False
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
        self, is_additive: bool, is_multiplicative: bool, is_diagonal: bool, is_scalar: bool,
    ) -> NoiseType:
        """
        Classify noise type with CORRECTED hierarchy.

        Priority (most specific to most general):
        1. ADDITIVE - constant, no dependencies (best for optimization)
        2. SCALAR - single noise source (nw=1), may be state-dependent
        3. DIAGONAL - diagonal structure (nw=nx), independent noise sources
        4. MULTIPLICATIVE - state-dependent, non-diagonal, non-scalar
        5. GENERAL - fallback for complex cases

        Key Changes from Previous Version
        ----------------------------------
        FIXED: Removed the buggy condition that made MULTIPLICATIVE unreachable.

        OLD (BUGGY):
            elif is_multiplicative and not is_diagonal and not is_scalar:
                return NoiseType.GENERAL  # ← BUG!
            elif is_multiplicative:
                return NoiseType.MULTIPLICATIVE  # ← Unreachable

        NEW (CORRECT):
            elif is_multiplicative:
                return NoiseType.MULTIPLICATIVE  # ← Now reachable!

        This means:
        - Diagonal + multiplicative → DIAGONAL (diagonal has higher priority)
        - Scalar + multiplicative → SCALAR (scalar has higher priority)
        - Non-diagonal, non-scalar, multiplicative → MULTIPLICATIVE ✓

        Examples by Category
        --------------------
        ADDITIVE:
        >>> g = [[0.3, 0.2]]  # Constant 1×2
        >>> # Classification: ADDITIVE

        SCALAR:
        >>> g = [[σ*x]]  # State-dependent, nw=1
        >>> # Classification: SCALAR (not MULTIPLICATIVE!)

        DIAGONAL:
        >>> g = [[σ₁*x₁, 0], [0, σ₂*x₂]]  # Diagonal, state-dependent
        >>> # Classification: DIAGONAL (not MULTIPLICATIVE!)

        MULTIPLICATIVE:
        >>> g = [[σ₁*x₁, σ₂*x₂]]  # Non-diagonal, non-scalar, state-dependent
        >>> # Classification: MULTIPLICATIVE ✓

        GENERAL:
        >>> g = complex non-standard structure
        >>> # Classification: GENERAL

        Parameters
        ----------
        is_additive : bool
            No dependencies on state/control/time
        is_multiplicative : bool
            Depends on state
        is_diagonal : bool
            Diagonal matrix structure
        is_scalar : bool
            Single noise source (nw = 1)

        Returns
        -------
        NoiseType
            Classified noise type
        """
        # Priority 1: Constant (most optimizable)
        if is_additive:
            return NoiseType.ADDITIVE

        # Priority 2: Single noise source (special structure)
        if is_scalar:
            return NoiseType.SCALAR

        # Priority 3: Diagonal (independent noise sources)
        if is_diagonal:
            return NoiseType.DIAGONAL

        # Priority 4: State-dependent non-diagonal non-scalar
        # FIXED: Removed buggy condition that caused MULTIPLICATIVE to be unreachable
        if is_multiplicative:
            return NoiseType.MULTIPLICATIVE

        # Priority 5: Fallback
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

        if claimed_type == "additive" and not char.is_additive:
            raise ValueError(
                f"Claimed noise_type='additive' but diffusion depends on: "
                f"state={char.depends_on_state}, "
                f"control={char.depends_on_control}, "
                f"time={char.depends_on_time}",
            )

        if claimed_type == "diagonal" and not char.is_diagonal:
            raise ValueError(
                "Claimed noise_type='diagonal' but diffusion has "
                "off-diagonal elements (coupling between noise sources)",
            )

        if claimed_type == "scalar" and not char.is_scalar:
            raise ValueError(f"Claimed noise_type='scalar' but nw={char.num_wiener} > 1")

        if claimed_type == "multiplicative" and not char.is_multiplicative:
            raise ValueError(
                "Claimed noise_type='multiplicative' but diffusion is constant "
                "(no state dependence detected)",
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
        >>> if hints['can_precompute']:
        ...     print("Can precompute constant noise!")
        """
        char = self.characteristics
        return {
            "can_precompute_diffusion": char.is_additive,
            "can_use_diagonal_solver": char.is_diagonal,
            "can_use_scalar_solver": char.is_scalar,
            "requires_reevaluation": char.is_multiplicative,
            "complexity": self._estimate_complexity(),
            "recommended_backends": self._recommend_backends(),
        }

    def _estimate_complexity(self) -> str:
        """Estimate computational complexity of diffusion evaluation."""
        char = self.characteristics
        if char.is_additive:
            return "O(1) - constant, precomputable"
        if char.is_scalar:
            return "O(nx) - scalar multiplication"
        if char.is_diagonal:
            return "O(nx) - element-wise"
        if char.noise_type == NoiseType.MULTIPLICATIVE:
            return "O(nx * nw) - full matrix, state-dependent"
        return "O(nx * nw) - full matrix, general"

    def _recommend_backends(self) -> List[str]:
        """Recommend backends based on noise structure."""
        char = self.characteristics
        # All backends support general noise
        recommended = ["jax", "torch", "numpy"]

        # JAX has best specialized solvers for additive
        if char.is_additive:
            recommended = ["jax", "numpy", "torch"]  # JAX first

        return recommended

    def __repr__(self) -> str:
        """String representation."""
        if self._characteristics is None:
            return "NoiseCharacterizer(not yet analyzed)"
        return f"NoiseCharacterizer(type={self.characteristics.noise_type.value})"


# ============================================================================
# Convenience Functions
# ============================================================================


def analyze_noise_structure(
    diffusion_expr: sp.Matrix,
    state_vars: List[sp.Symbol],
    control_vars: List[sp.Symbol],
    time_var: Optional[sp.Symbol] = None,
) -> NoiseCharacteristics:
    """
    Convenience function for analyzing noise structure.

    Parameters
    ----------
    diffusion_expr : sp.Matrix
        Symbolic diffusion matrix
    state_vars : List[sp.Symbol]
        State variables
    control_vars : List[sp.Symbol]
        Control variables
    time_var : sp.Symbol, optional
        Time variable

    Returns
    -------
    NoiseCharacteristics
        Analysis results

    Examples
    --------
    >>> x = sp.symbols('x')
    >>> diffusion = sp.Matrix([[0.1 * x, 0.05 * x]])
    >>> char = analyze_noise_structure(diffusion, [x], [])
    >>> print(char.noise_type)
    NoiseType.MULTIPLICATIVE
    """
    characterizer = NoiseCharacterizer(diffusion_expr, state_vars, control_vars, time_var)
    return characterizer.analyze()
