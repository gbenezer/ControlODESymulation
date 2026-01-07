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
Integration Method Registry and Normalization
==============================================

Centralized registry of integration methods across backends with support for:
- Canonical name normalization (e.g., 'euler_maruyama' → 'EM'/'euler'/'Euler')
- Method classification (SDE vs ODE, fixed-step vs adaptive)
- Backend compatibility checking
- Method discovery and validation

This module serves as the single source of truth for all integration methods
available across NumPy/Julia (DiffEqPy), PyTorch (TorchDiffEq/TorchSDE), and
JAX (Diffrax) backends.

Design Philosophy
-----------------
**Backend Naming Conventions:**
- NumPy/Julia (DiffEqPy): Capitalized (e.g., 'EM', 'Tsit5', 'SRIW1')
- PyTorch (TorchSDE/TorchDiffEq): lowercase (e.g., 'euler', 'dopri5')
- JAX (Diffrax): PascalCase (e.g., 'Euler', 'ItoMilstein', 'Tsit5')

**Canonical Names:**
User-friendly aliases that work across all backends (e.g., 'euler_maruyama',
'milstein', 'rk45'). These are automatically normalized to backend-specific
names via the normalization map.

**Method Classification:**
Methods are classified along two dimensions:
1. **System type**: Deterministic (ODE) vs Stochastic (SDE)
2. **Time stepping**: Fixed-step vs Adaptive

This classification determines:
- Which DiscretizationMode is appropriate
- Whether dense output is available
- Expected convergence properties

Usage Examples
--------------
**Basic normalization:**

>>> from cdesym.systems.base.numerical_integration.method_registry import (
...     normalize_method_name
... )
>>> 
>>> # Canonical name → backend-specific
>>> normalize_method_name('euler_maruyama', 'numpy')
'EM'
>>> normalize_method_name('euler_maruyama', 'torch')
'euler'
>>> normalize_method_name('euler_maruyama', 'jax')
'Euler'

**Method classification:**

>>> from cdesym.systems.base.numerical_integration.method_registry import (
...     is_sde_method, is_fixed_step
... )
>>> 
>>> is_sde_method('euler_maruyama')
True
>>> is_sde_method('rk4')
False
>>> 
>>> is_fixed_step('rk4')
True
>>> is_fixed_step('RK45')
False

**Discovering available methods:**

>>> from cdesym.systems.base.numerical_integration.method_registry import (
...     get_available_methods
... )
>>> 
>>> # All methods for PyTorch backend
>>> methods = get_available_methods('torch', method_type='all')
>>> print(methods['sde_fixed_step'])
['euler', 'milstein', 'srk', 'midpoint']
>>> 
>>> # Only stochastic methods
>>> sde_methods = get_available_methods('jax', method_type='stochastic')
>>> print(sde_methods['canonical_aliases'])
['euler_maruyama', 'milstein', 'rk45', 'rk23', 'tsit5']

**Validation:**

>>> from cdesym.systems.base.numerical_integration.method_registry import (
...     validate_method
... )
>>> 
>>> # Check if method is valid for backend
>>> is_valid, error = validate_method('euler_maruyama', 'torch', is_stochastic=True)
>>> if is_valid:
...     print("✓ Method is valid")
>>> else:
...     print(f"✗ {error}")
✓ Method is valid

**Integration with DiscretizedSystem:**

>>> from cdesym.systems.base.numerical_integration.method_registry import (
...     normalize_method_name, is_fixed_step
... )
>>> 
>>> # In DiscretizedSystem.__init__:
>>> backend = getattr(continuous_system, '_default_backend', 'numpy')
>>> method = normalize_method_name(user_method, backend)
>>> self._is_fixed_step = is_fixed_step(method)

Notes
-----
- Method availability depends on installed packages (diffeqpy, torchsde, diffrax)
- Manual implementations (euler, rk4, etc.) work on all backends
- Some method names appear in multiple categories (e.g., 'euler' for both
  deterministic and SDE in PyTorch context)
- Normalization is idempotent: normalize(normalize(x)) = normalize(x)
"""

from typing import Dict, FrozenSet, List, Literal, Optional, Tuple
from cdesym.types.backends import Backend


# ============================================================================
# Deterministic Methods - Fixed Step
# ============================================================================

DETERMINISTIC_FIXED_STEP: FrozenSet[str] = frozenset([
    # Manual implementations (available on all backends)
    # NOTE: 'euler' and 'midpoint' also appear in SDE context for some backends
    # When used as deterministic methods, they ignore any noise terms
    "euler",     # Forward Euler (1st order)
    "heun",      # Heun's method / Improved Euler / Explicit Trapezoid (2nd order)
    "midpoint",  # Midpoint/RK2 (2nd order)
    "rk4",       # Classic Runge-Kutta 4 (4th order)
])

# ============================================================================
# Deterministic Methods - Adaptive
# ============================================================================

DETERMINISTIC_ADAPTIVE: FrozenSet[str] = frozenset([
    # ========================================================================
    # Scipy methods (NumPy backend)
    # ========================================================================
    "RK45",      # Dormand-Prince 5(4) - general purpose
    "RK23",      # Bogacki-Shampine 3(2) - lower accuracy
    "DOP853",    # Dormand-Prince 8(5,3) - high accuracy
    "Radau",     # Implicit Runge-Kutta (stiff systems)
    "BDF",       # Backward Differentiation Formula (very stiff)
    "LSODA",     # Auto stiffness detection (switches between Adams/BDF)
    
    # ========================================================================
    # TorchDiffEq methods (PyTorch backend)
    # ========================================================================
    "dopri5",          # Dormand-Prince 5(4)
    "dopri8",          # Dormand-Prince 8
    "bosh3",           # Bogacki-Shampine 3(2)
    "fehlberg2",       # Fehlberg 2(1)
    # NOTE: 'adaptive_heun' removed - it's primarily an SDE method
    "explicit_adams",  # Explicit Adams-Bashforth
    "implicit_adams",  # Implicit Adams-Moulton
    
    # ========================================================================
    # Diffrax methods (JAX backend)
    # ========================================================================
    "tsit5",          # Tsitouras 5/4 (modern, efficient)
    "dopri5",         # Dormand-Prince 5(4)
    "dopri8",         # Dormand-Prince 8
    "bosh3",          # Bogacki-Shampine 3(2)
    "implicit_euler", # Implicit Euler (stiff)
    "kvaerno3",       # Kvaerno 3 (ESDIRK, stiff)
    "kvaerno4",       # Kvaerno 4 (ESDIRK, stiff)
    "kvaerno5",       # Kvaerno 5 (ESDIRK, stiff)
    
    # ========================================================================
    # Julia/DiffEqPy methods (NumPy backend via Julia)
    # ========================================================================
    "Tsit5",         # Tsitouras 5/4
    "Vern6",         # Verner 6/5
    "Vern7",         # Verner 7/6
    "Vern8",         # Verner 8/7
    "Vern9",         # Verner 9/8
    "DP5",           # Dormand-Prince 5
    "DP8",           # Dormand-Prince 8
    "Rosenbrock23",  # Rosenbrock 2/3 (stiff)
    "Rodas5",        # Rosenbrock 5 (stiff)
    "ROCK4",         # Stabilized explicit (moderately stiff)
])

# ============================================================================
# Stochastic (SDE) Methods - Fixed Step
# ============================================================================

SDE_FIXED_STEP: FrozenSet[str] = frozenset([
    # ========================================================================
    # Canonical names (user-friendly aliases)
    # ========================================================================
    "euler_maruyama",         # Most common SDE method (strong 0.5, weak 1.0)
    "milstein",               # Higher accuracy (strong 1.0)
    "stratonovich_milstein",  # Stratonovich interpretation
    "sra1",                   # Stochastic Runge-Kutta (strong 1.5, diagonal)
    
    # ========================================================================
    # NumPy/Julia (DiffEqPy) - Capitalized
    # ========================================================================
    "EM",          # Euler-Maruyama
    "EulerHeun",   # Euler-Heun (predictor-corrector)
    "SRIW1",       # Stochastic Runge-Kutta (diagonal noise, weak 1.5)
    "SRIW2",       # Stochastic Runge-Kutta (diagonal noise, weak 2.0)
    "SRA1",        # Stochastic Runge-Kutta (strong 1.5, diagonal)
    "SRA3",        # Stochastic Runge-Kutta (strong 1.5, general)
    "RKMil",       # Runge-Kutta-Milstein
    "ImplicitEM",  # Implicit Euler-Maruyama (stiff SDEs)
    
    # ========================================================================
    # PyTorch (TorchSDE) - lowercase
    # ========================================================================
    # NOTE: PyTorch uses 'euler' for SDE, which conflicts with deterministic
    # The context (stochastic vs deterministic system) determines usage
    "euler",           # Euler-Maruyama (TorchSDE naming)
    "milstein",        # Milstein method
    "srk",             # Stochastic Runge-Kutta
    "midpoint",        # Stochastic midpoint
    
    # ========================================================================
    # JAX (Diffrax) - PascalCase
    # ========================================================================
    "Euler",                  # Euler-Maruyama
    "EulerHeun",              # Euler-Heun
    "Heun",                   # Heun's method for SDEs
    "ItoMilstein",            # Milstein (Itô interpretation)
    "StratonovichMilstein",   # Milstein (Stratonovich interpretation)
    "SEA",                    # Split-step Exponential Adams
    "SHARK",                  # Split-step High-order Adams-Runge-Kutta
    "SRA1",                   # Stochastic Runge-Kutta (matches Julia)
    "ReversibleHeun",         # Time-reversible Heun
])

# ============================================================================
# Stochastic (SDE) Methods - Adaptive
# ============================================================================

SDE_ADAPTIVE: FrozenSet[str] = frozenset([
    # ========================================================================
    # NumPy/Julia (DiffEqPy)
    # ========================================================================
    "AutoEM",     # Automatic Euler-Maruyama with adaptive stepping
    "LambaEM",    # Lambda Euler-Maruyama (adaptive)
    
    # ========================================================================
    # PyTorch (TorchSDE)
    # ========================================================================
    "adaptive_heun",   # Adaptive Heun for SDEs
    "reversible_heun",  # Can be used in adaptive mode (PyTorch)
])

# Union of all SDE methods
SDE_METHODS: FrozenSet[str] = SDE_FIXED_STEP | SDE_ADAPTIVE

# ============================================================================
# Normalization Map: Canonical Names → Backend-Specific Names
# ============================================================================

NORMALIZATION_MAP: Dict[str, Dict[Backend, str]] = {
    # ========================================================================
    # Stochastic (SDE) Methods
    # ========================================================================
    
    # Euler-Maruyama (most common SDE method)
    "euler_maruyama": {
        "numpy": "EM",      # Julia/DiffEqPy
        "torch": "euler",   # TorchSDE
        "jax": "Euler",     # Diffrax
    },
    
    # Milstein method
    "milstein": {
        "numpy": "RKMil",      # Julia (Runge-Kutta-Milstein)
        "torch": "milstein",   # TorchSDE
        "jax": "ItoMilstein",  # Diffrax (Itô interpretation)
    },
    
    # Stratonovich Milstein
    "stratonovich_milstein": {
        "numpy": "RKMil",                # Julia (can handle Stratonovich)
        "torch": "milstein",             # TorchSDE (typically Itô)
        "jax": "StratonovichMilstein",   # Diffrax
    },
    
    # SRA1 (diagonal noise, order 1.5 strong)
    "sra1": {
        "numpy": "SRA1",  # Julia
        "torch": "srk",   # TorchSDE (closest equivalent)
        "jax": "SRA1",    # Diffrax
    },
    
    # Reversible/Symmetric Heun (for time-reversible integration)
    "reversible_heun": {
        "numpy": "EulerHeun",       # Julia (similar, but not identical)
        "torch": "reversible_heun", # TorchSDE
        "jax": "ReversibleHeun",    # Diffrax
    },
    
    # ========================================================================
    # Deterministic Methods - Adaptive
    # ========================================================================
    
    # RK45 family (Dormand-Prince 5(4))
    "rk45": {
        "numpy": "RK45",    # Scipy
        "torch": "dopri5",  # TorchDiffEq
        "jax": "tsit5",     # Diffrax (Tsitouras 5/4, similar performance)
    },
    
    # Dormand-Prince 5(4) - explicit
    "dopri5": {
        "numpy": "RK45",    # Scipy (DP5(4) = RK45)
        "torch": "dopri5",  # TorchDiffEq
        "jax": "dopri5",    # Diffrax also has dopri5
    },
    
    # RK23 family (Bogacki-Shampine 3(2))
    "rk23": {
        "numpy": "RK23",   # Scipy
        "torch": "bosh3",  # TorchDiffEq
        "jax": "bosh3",    # Diffrax
    },
    
    # High-order methods
    "dopri8": {
        "numpy": "DOP853",  # Scipy (8th order)
        "torch": "dopri8",  # TorchDiffEq
        "jax": "dopri8",    # Diffrax
    },
    
    # Tsitouras 5/4 (modern, efficient)
    "tsit5": {
        "numpy": "Tsit5",   # Julia
        "torch": "dopri5",  # TorchDiffEq (closest)
        "jax": "tsit5",     # Diffrax
    },
    
    # Implicit methods for stiff systems
    "implicit_euler": {
        "numpy": "Radau",          # Scipy (implicit RK)
        "torch": "implicit_adams", # TorchDiffEq
        "jax": "implicit_euler",   # Diffrax
    },
    
    # Stiff ODE solvers
    "bdf": {
        "numpy": "BDF",            # Scipy
        "torch": "implicit_adams", # TorchDiffEq (closest)
        "jax": "kvaerno5",         # Diffrax (ESDIRK for stiff)
    },
    
    # Auto stiffness detection
    "lsoda": {
        "numpy": "LSODA",   # Scipy
        "torch": "dopri5",  # TorchDiffEq (no LSODA equivalent)
        "jax": "tsit5",     # Diffrax (no LSODA equivalent)
    },
    
    # ========================================================================
    # Deterministic Methods - Fixed Step (prefer Julia on numpy)
    # ========================================================================
    
    # Euler method - prefer Julia 'Euler' on numpy, manual elsewhere
    "euler": {
        "numpy": "Euler",     # Use Julia's implementation
        "torch": "euler",     # Use manual implementation
        "jax": "euler",       # Use manual implementation
    },
    
    # Heun method - prefer Julia 'Heun' on numpy, manual elsewhere
    "heun": {
        "numpy": "Heun",      # Use Julia's implementation
        "torch": "heun",      # Use manual implementation
        "jax": "heun",        # Use manual implementation
    },
    
    # Midpoint method - prefer Julia 'Midpoint' on numpy, manual elsewhere
    "midpoint": {
        "numpy": "Midpoint",  # Use Julia's implementation
        "torch": "midpoint",  # Use manual implementation
        "jax": "midpoint",    # Use manual implementation
    },
    
    # Allow users to explicitly request manual implementations
    "manual_euler": {
        "numpy": "euler",     # Force manual (lowercase)
        "torch": "euler", 
        "jax": "euler",
    },
    "manual_heun": {
        "numpy": "heun",      # Force manual (lowercase)
        "torch": "heun",
        "jax": "heun", 
    },
    "manual_midpoint": {
        "numpy": "midpoint",  # Force manual (lowercase)
        "torch": "midpoint",
        "jax": "midpoint",
    },
}

# ============================================================================
# Backend-Specific Method Sets
# ============================================================================

BACKEND_METHODS: Dict[Backend, FrozenSet[str]] = {
    # ========================================================================
    # NumPy backend: Julia/DiffEqPy + Scipy
    # ========================================================================
    "numpy": frozenset([
        # Julia capitalized methods (SDE)
        "EM", "LambaEM", "EulerHeun", "SRIW1", "SRIW2", "SRA1", "SRA3",
        "RKMil", "ImplicitEM", "AutoEM",
        # Scipy methods (ODE)
        "RK45", "RK23", "DOP853", "Radau", "BDF", "LSODA",
        # Julia capitalized methods (ODE)
        "Tsit5", "Vern6", "Vern7", "Vern8", "Vern9", "DP5", "DP8",
        "Rosenbrock23", "Rodas5", "ROCK4",
        # Manual implementations (fixed-step)
        "euler", "heun", "midpoint", "rk4",
        # Julia low-order implementations (optional, capitalized)
        "Euler", "Heun", "Midpoint",
    ]),
    
    # ========================================================================
    # PyTorch backend: TorchSDE + TorchDiffEq
    # ========================================================================
    "torch": frozenset([
        # TorchSDE lowercase (SDE)
        "euler", "milstein", "srk", "midpoint", "reversible_heun",
        "adaptive_heun",
        # TorchDiffEq lowercase (ODE)
        "dopri5", "dopri8", "bosh3", "fehlberg2",
        "explicit_adams", "implicit_adams",
        # Manual implementations (fixed-step)
        "heun", "rk4",
    ]),
    
    # ========================================================================
    # JAX backend: Diffrax
    # ========================================================================
    "jax": frozenset([
        # Diffrax SDE methods (PascalCase)
        "Euler", "EulerHeun", "Heun", "ItoMilstein", "StratonovichMilstein",
        "SEA", "SHARK", "SRA1", "ReversibleHeun",
        # Diffrax ODE methods (lowercase/PascalCase mix)
        "tsit5", "dopri5", "dopri8", "bosh3", "implicit_euler",
        "kvaerno3", "kvaerno4", "kvaerno5",
        # Manual implementations (fixed-step)
        "euler", "heun", "midpoint", "rk4",
    ]),
}

# ============================================================================
# Method Classification Functions
# ============================================================================

def is_sde_method(method: str) -> bool:
    """
    Check if integration method is for stochastic differential equations.

    Determines whether a given method name is designed for SDE integration
    (stochastic systems) or deterministic ODE integration.

    Parameters
    ----------
    method : str
        Integration method name (normalized or original)

    Returns
    -------
    bool
        True if method is for stochastic systems (SDE), False otherwise

    Notes
    -----
    - Checks against `SDE_METHODS` frozenset (union of fixed-step and adaptive)
    - Works with both canonical names ('euler_maruyama') and backend-specific
      names ('EM', 'euler', 'Euler')
    - Method name should ideally be normalized first, but works with any name
      in the `SDE_METHODS` set
    - Returns False for deterministic methods (euler, rk4, RK45, etc.)

    Ambiguous Cases
    ---------------
    Some method names appear in both deterministic and stochastic contexts:

    - 'euler': Both a deterministic method (Forward Euler) and SDE method
      (Euler-Maruyama for PyTorch/TorchSDE). In `SDE_METHODS`, so returns True.
    - 'midpoint': Similar ambiguity. Returns True (in `SDE_METHODS`).

    For these cases, normalization to canonical names ('euler_maruyama' vs 'rk4')
    is recommended before calling this function.

    Examples
    --------
    >>> # Canonical SDE names
    >>> is_sde_method('euler_maruyama')
    True
    >>> is_sde_method('milstein')
    True

    >>> # Backend-specific SDE names
    >>> is_sde_method('EM')  # NumPy/Julia
    True
    >>> is_sde_method('euler')  # PyTorch (ambiguous!)
    True
    >>> is_sde_method('ItoMilstein')  # JAX
    True

    >>> # Deterministic methods
    >>> is_sde_method('rk4')
    False
    >>> is_sde_method('RK45')
    False
    >>> is_sde_method('dopri5')
    False

    >>> # Unknown methods
    >>> is_sde_method('my_custom_method')
    False

    See Also
    --------
    is_fixed_step : Classify method by time-stepping strategy
    normalize_method_name : Convert canonical names to backend-specific
    """
    return method in SDE_METHODS


def is_fixed_step(method: str) -> bool:
    """
    Check if integration method uses fixed time stepping.

    Classifies methods into fixed-step (constant dt throughout integration)
    or adaptive (dt adjusted based on error estimates). This classification
    is used to auto-select discretization mode and validate mode/method
    compatibility.

    Parameters
    ----------
    method : str
        Integration method name (normalized or original)

    Returns
    -------
    bool
        True if method uses fixed time steps, False if adaptive

    Classification Rules
    --------------------
    1. **Deterministic fixed-step**: euler, midpoint, rk4, heun → True
    2. **Deterministic adaptive**: RK45, LSODA, dopri5, tsit5, etc. → False
    3. **SDE fixed-step**: Most SDE methods (EM, euler_maruyama, etc.) → True
    4. **SDE adaptive**: Rare cases (LambaEM, AutoEM, adaptive_heun) → False
    5. **Unknown methods**: Conservative default → False (more flexible)

    Notes
    -----
    - Fixed-step methods take exactly `n_steps` integrations of size `dt`
    - Adaptive methods adjust step size internally for accuracy/efficiency
    - Most SDE methods are fixed-step (adaptive SDE solvers are rare)
    - Unknown methods default to False (adaptive mode works for both cases)
    - Used to auto-select DiscretizationMode.FIXED_STEP vs DENSE_OUTPUT

    Design Decision: Conservative Default
    -------------------------------------
    When method is unknown, returns False (assume adaptive) because:
    - DENSE_OUTPUT mode works for both fixed and adaptive methods
    - FIXED_STEP mode ONLY works for fixed-step methods
    - Better to be conservative than raise unexpected errors

    Examples
    --------
    >>> # Deterministic fixed-step methods
    >>> is_fixed_step('euler')
    True
    >>> is_fixed_step('rk4')
    True
    >>> is_fixed_step('heun')
    True

    >>> # Deterministic adaptive methods
    >>> is_fixed_step('RK45')
    False
    >>> is_fixed_step('LSODA')
    False
    >>> is_fixed_step('dopri5')  # PyTorch
    False
    >>> is_fixed_step('tsit5')  # JAX
    False

    >>> # SDE methods (mostly fixed-step)
    >>> is_fixed_step('euler_maruyama')
    True
    >>> is_fixed_step('EM')  # NumPy/Julia
    True
    >>> is_fixed_step('milstein')
    True
    >>> is_fixed_step('SRIW1')  # Julia SDE
    True

    >>> # Rare adaptive SDE methods
    >>> is_fixed_step('LambaEM')  # Julia adaptive
    False
    >>> is_fixed_step('AutoEM')  # Julia adaptive
    False
    >>> is_fixed_step('adaptive_heun')  # PyTorch
    False
    >>> is_fixed_step('reversible_heun')  # Can be adaptive
    False

    >>> # Unknown method (conservative default)
    >>> is_fixed_step('my_custom_method')
    False

    Notes on Ambiguous Methods
    ---------------------------
    Some methods appear in both deterministic and SDE contexts:
    
    - **'euler'**: In both DETERMINISTIC_FIXED_STEP and SDE_FIXED_STEP
      Classification: Fixed-step (True) for both contexts
      
    - **'midpoint'**: In both DETERMINISTIC_FIXED_STEP and SDE_FIXED_STEP
      Classification: Fixed-step (True) for both contexts
      
    - **'reversible_heun'**: In both SDE_FIXED_STEP and SDE_ADAPTIVE
      Classification: Adaptive (False) - prioritizes adaptive classification
      since it CAN be used in adaptive mode
    
    The ambiguity is resolved at runtime by the system type (stochastic vs
    deterministic) passed to validate_method().

    See Also
    --------
    is_sde_method : Check if method is for stochastic systems
    normalize_method_name : Normalize method names across backends
    """
    # Deterministic fixed-step
    if method in DETERMINISTIC_FIXED_STEP:
        return True

    # Deterministic adaptive
    if method in DETERMINISTIC_ADAPTIVE:
        return False

    # SDE methods - mostly fixed-step
    if method in SDE_METHODS:
        # Check if it's one of the rare adaptive SDE methods
        return method not in SDE_ADAPTIVE

    # Unknown method - conservative default
    # Assume adaptive (more flexible, works for both fixed and adaptive)
    return False


def normalize_method_name(method: str, backend: Backend = "numpy") -> str:
    """
    Normalize method names across backends to canonical form.

    Provides user-friendly aliases while maintaining backend compatibility.
    Handles both deterministic and stochastic (SDE) methods. This enables
    portable code: users can write 'euler_maruyama' and it automatically
    becomes 'EM' (NumPy), 'euler' (PyTorch), or 'Euler' (JAX).

    Parameters
    ----------
    method : str
        User-provided method name (canonical or backend-specific)
    backend : Backend, default='numpy'
        Target backend: 'numpy', 'torch', or 'jax'

    Returns
    -------
    str
        Normalized method name appropriate for backend

    Normalization Logic
    -------------------
    1. **Auto-switching methods** → Pass through unchanged
       - Julia methods with parentheses (e.g., 'AutoTsit5(Rosenbrock23())')

    2. **Already valid for backend?** → Return (with special upgrades)
       - Check if method is in BACKEND_METHODS[backend]
       - Special case: lowercase euler/heun/midpoint on numpy → upgrade to Julia
       - Avoids breaking SDE methods (e.g., 'Euler' on jax stays 'Euler')

    3. **In normalization map?** → Map to backend-specific name
       - Check both exact case and lowercase versions
       - Handles canonical names like 'euler_maruyama', 'rk45'
       - Handles explicit manual requests like 'manual_euler'

    4. **Unknown method?** → Pass through unchanged
       - Will fail at integrator level with appropriate error
       - Allows custom methods to pass through

    Backend Conventions
    -------------------
    - **NumPy/Julia (DiffEqPy)**: Capitalized (e.g., 'EM', 'Tsit5', 'SRIW1')
    - **PyTorch (TorchSDE/TorchDiffEq)**: lowercase (e.g., 'euler', 'dopri5')
    - **JAX (Diffrax)**: PascalCase (e.g., 'Euler', 'ItoMilstein', 'Tsit5')

    Julia Preference for ODE Methods
    ---------------------------------
    On NumPy backend, lowercase ODE methods automatically prefer Julia:
    - 'euler' → 'Euler' (Julia's implementation)
    - 'heun' → 'Heun' (Julia's implementation)
    - 'midpoint' → 'Midpoint' (Julia's implementation)
    
    To explicitly use manual implementations on NumPy:
    - Use 'manual_euler', 'manual_heun', 'manual_midpoint'

    Idempotency
    -----------
    Normalization is idempotent:
    >>> normalize_method_name(normalize_method_name('euler_maruyama', 'jax'), 'jax')
    'Euler'

    Once normalized, subsequent calls return the same result.

    Examples
    --------
    **Canonical SDE names → backend-specific:**

    >>> normalize_method_name('euler_maruyama', 'numpy')
    'EM'
    >>> normalize_method_name('euler_maruyama', 'torch')
    'euler'
    >>> normalize_method_name('euler_maruyama', 'jax')
    'Euler'

    **Canonical ODE names → backend-specific:**

    >>> normalize_method_name('rk45', 'numpy')
    'RK45'
    >>> normalize_method_name('rk45', 'torch')
    'dopri5'
    >>> normalize_method_name('rk45', 'jax')
    'tsit5'

    **Julia preference for ODE methods on NumPy:**

    >>> normalize_method_name('euler', 'numpy')
    'Euler'  # Upgraded to Julia
    >>> normalize_method_name('heun', 'numpy')
    'Heun'   # Upgraded to Julia
    >>> normalize_method_name('midpoint', 'numpy')
    'Midpoint'  # Upgraded to Julia

    **Manual implementations on NumPy:**

    >>> normalize_method_name('manual_euler', 'numpy')
    'euler'  # Explicit manual request
    >>> normalize_method_name('manual_heun', 'numpy')
    'heun'   # Explicit manual request

    **SDE methods preserved (critical for correctness):**

    >>> normalize_method_name('Euler', 'jax')
    'Euler'  # SDE method, NOT normalized to 'euler'
    >>> normalize_method_name('Heun', 'jax')
    'Heun'   # SDE method, NOT normalized to 'heun'

    **ODE methods on torch/jax use manual implementations:**

    >>> normalize_method_name('euler', 'torch')
    'euler'
    >>> normalize_method_name('heun', 'jax')
    'heun'

    **Cross-backend normalization (torch → jax):**

    >>> normalize_method_name('dopri5', 'jax')
    'dopri5'  # Note: 'dopri5' exists in JAX, so no mapping needed

    **Already correct for backend → unchanged:**

    >>> normalize_method_name('EM', 'numpy')
    'EM'
    >>> normalize_method_name('ItoMilstein', 'jax')
    'ItoMilstein'

    **Manual implementations (portable):**

    >>> normalize_method_name('rk4', 'numpy')
    'rk4'
    >>> normalize_method_name('rk4', 'torch')
    'rk4'
    >>> normalize_method_name('rk4', 'jax')
    'rk4'

    **Unknown method → pass through:**

    >>> normalize_method_name('my_custom_method', 'numpy')
    'my_custom_method'
    # Will be validated at integrator creation

    **Case-insensitive lookup (canonical names only):**

    >>> normalize_method_name('RK45', 'torch')
    'dopri5'
    >>> normalize_method_name('rk45', 'torch')
    'dopri5'

    Usage in DiscretizedSystem
    ---------------------------
    >>> # Automatic normalization in __init__
    >>> backend = getattr(continuous_system, '_default_backend', 'numpy')
    >>> method = normalize_method_name(user_method, backend)
    >>> # Now 'method' is guaranteed to use backend conventions

    Notes
    -----
    - Normalization happens before any method classification
    - Original method name should be preserved for debugging/logging
    - Some canonical names map to different methods on different backends
      (e.g., 'rk45' → 'RK45'/'dopri5'/'tsit5')
    - Not all methods can be perfectly mapped across backends (e.g., 'lsoda'
      doesn't exist on PyTorch/JAX, maps to nearest equivalent)
    - **Critical**: Methods already valid for a backend are preserved to avoid
      breaking SDE integration (e.g., 'Euler' on jax must stay 'Euler')

    See Also
    --------
    validate_method : Check if method is valid for backend
    is_sde_method : Check if method is for stochastic systems
    is_fixed_step : Check if method uses fixed time stepping
    """
    
    # Add None check
    if method is None:
        raise ValueError("method cannot be None")
    
    # ========================================================================
    # Handle Julia/DiffEqPy auto-switching methods FIRST
    # ========================================================================
    
    # DiffEqPy methods can have complex syntax with parentheses for auto-switching
    # e.g., AutoTsit5(Rosenbrock23()), AutoVern7(Rodas5())
    # These are valid on numpy backend and should be passed through as-is
    if backend == "numpy" and "(" in method:
        # This is likely a Julia auto-switching method
        # Don't try to normalize it - validation will handle it
        return method
    
    # ========================================================================
    # Check if already valid for backend
    # ========================================================================
    
    if backend in BACKEND_METHODS and method in BACKEND_METHODS[backend]:
        # Special case: Prefer Julia for lowercase euler/heun/midpoint on numpy
        # This upgrades manual ODE implementations to faster Julia versions
        if backend == "numpy" and method in ["euler", "heun", "midpoint"]:
            return method.capitalize()  # euler → Euler, heun → Heun, etc.
        
        # Otherwise, if it's already valid, don't change it!
        # This is CRITICAL for SDE methods - 'Euler' on jax must stay 'Euler'
        # not be converted to 'euler' which would break SDE integration
        return method
    
    # ========================================================================
    # Try normalization map (exact and case-insensitive)
    # ========================================================================
    
    method_lower = method.lower()
    
    # Try exact match first
    if method in NORMALIZATION_MAP and backend in NORMALIZATION_MAP[method]:
        return NORMALIZATION_MAP[method][backend]
    
    # Try case-insensitive match
    if method_lower in NORMALIZATION_MAP and backend in NORMALIZATION_MAP[method_lower]:
        return NORMALIZATION_MAP[method_lower][backend]
    
    # ========================================================================
    # No normalization found - return as-is
    # ========================================================================
    
    # Let integrator factory handle validation and error messages
    return method


def get_available_methods(
    backend: Backend = "numpy",
    method_type: Literal["all", "deterministic", "stochastic", "fixed_step", "adaptive"] = "all",
) -> Dict[str, List[str]]:
    """
    Get available integration methods for a backend.

    Returns a dictionary of methods organized by category (deterministic vs
    stochastic, fixed-step vs adaptive) along with canonical aliases that
    work across all backends.

    Parameters
    ----------
    backend : Backend, default='numpy'
        Backend to query: 'numpy', 'torch', or 'jax'
    method_type : str, default='all'
        Filter by method type:
        - 'all': All methods (default)
        - 'deterministic': Only ODE methods
        - 'stochastic': Only SDE methods
        - 'fixed_step': Only fixed-step methods (both ODE and SDE)
        - 'adaptive': Only adaptive methods (both ODE and SDE)

    Returns
    -------
    dict
        Dictionary with method categories as keys:
        
        - **deterministic_fixed_step** : list of str
            Fixed-step ODE methods (euler, rk4, heun, midpoint)
        
        - **deterministic_adaptive** : list of str
            Adaptive ODE methods (RK45, LSODA, dopri5, tsit5, etc.)
        
        - **sde_fixed_step** : list of str
            Fixed-step SDE methods (EM, euler_maruyama, milstein, etc.)
        
        - **sde_adaptive** : list of str
            Adaptive SDE methods (LambaEM, AutoEM, adaptive_heun)
        
        - **canonical_aliases** : list of str
            User-friendly canonical names that work on all backends
            (euler_maruyama, milstein, rk45, rk23, etc.)

    Method Availability
    -------------------
    Actual availability depends on installed packages:
    
    - **NumPy backend**:
      - Requires: scipy (always available)
      - Optional: diffeqpy (Julia integration, enables high-accuracy SDE solvers)
      - Without diffeqpy: Only scipy methods + manual implementations
    
    - **PyTorch backend**:
      - Requires: torch (for basic functionality)
      - Optional: torchdiffeq (ODE solvers), torchsde (SDE solvers)
      - Without packages: Only manual implementations (euler, rk4, heun)
    
    - **JAX backend**:
      - Requires: jax, jaxlib
      - Optional: diffrax (comprehensive ODE/SDE solvers)
      - Without diffrax: Only manual implementations

    This function returns ALL methods that COULD be available, not just
    those currently installed. Use validate_method() to check if a specific
    method is actually available.

    Examples
    --------
    **Get all methods for PyTorch:**

    >>> methods = get_available_methods('torch', method_type='all')
    >>> print(methods.keys())
    dict_keys(['deterministic_fixed_step', 'deterministic_adaptive',
               'sde_fixed_step', 'sde_adaptive', 'canonical_aliases'])
    >>> 
    >>> print(methods['sde_fixed_step'])
    ['euler', 'milstein', 'srk', 'midpoint']

    **Get only stochastic methods:**

    >>> sde_methods = get_available_methods('jax', method_type='stochastic')
    >>> print(sde_methods['sde_fixed_step'])
    ['Euler', 'EulerHeun', 'Heun', 'ItoMilstein', 'StratonovichMilstein',
     'SEA', 'SHARK', 'SRA1']
    >>> 
    >>> print(sde_methods['canonical_aliases'])
    ['euler_maruyama', 'milstein', 'rk45', 'rk23', 'tsit5']

    **Get only fixed-step methods (both ODE and SDE):**

    >>> fixed = get_available_methods('numpy', method_type='fixed_step')
    >>> print(fixed['deterministic_fixed_step'])
    ['euler', 'midpoint', 'rk4', 'heun']
    >>> print(fixed['sde_fixed_step'])
    ['EM', 'EulerHeun', 'SRIW1', 'SRIW2', 'SRA1', 'SRA3', 'RKMil', 'ImplicitEM']

    **Compare backends:**

    >>> for backend in ['numpy', 'torch', 'jax']:
    ...     methods = get_available_methods(backend, method_type='stochastic')
    ...     n_sde = len(methods['sde_fixed_step']) + len(methods['sde_adaptive'])
    ...     print(f"{backend:6s}: {n_sde} SDE methods")
    numpy : 10 SDE methods
    torch : 6 SDE methods
    jax   : 9 SDE methods

    **Discover canonical aliases:**

    >>> methods = get_available_methods('torch')
    >>> print("Portable canonical names:")
    >>> for alias in methods['canonical_aliases']:
    ...     print(f"  - {alias}")
    Portable canonical names:
      - euler_maruyama
      - milstein
      - rk45
      - rk23

    **Filter and format for user display:**

    >>> methods = get_available_methods('numpy', method_type='deterministic')
    >>> 
    >>> print("Fixed-step methods:")
    >>> for method in sorted(methods['deterministic_fixed_step']):
    ...     print(f"  - {method}")
    >>> 
    >>> print("\nAdaptive methods:")
    >>> for method in sorted(methods['deterministic_adaptive']):
    ...     print(f"  - {method}")

    **Usage in DiscretizedSystem.get_available_methods():**

    >>> # Delegate to this utility function
    >>> @staticmethod
    >>> def get_available_methods(*args, **kwargs):
    ...     from cdesym.systems.base.numerical_integration.method_registry import (
    ...         get_available_methods as _get_available_methods
    ...     )
    ...     return _get_available_methods(*args, **kwargs)

    Notes
    -----
    - Returns potential methods, not necessarily installed/working
    - Manual implementations (euler, rk4, heun) work on all backends
    - Some method names appear in multiple categories due to backend conventions
    - Canonical aliases provide portable names across backends
    - Filter results by method_type to reduce information overload

    See Also
    --------
    validate_method : Check if specific method is actually available
    normalize_method_name : Convert canonical to backend-specific names
    BACKEND_METHODS : Complete set of methods per backend
    """
    # ========================================================================
    # Build complete method dictionary for backend
    # ========================================================================
    
    all_methods = {
        "numpy": {
            "deterministic_fixed_step": sorted(list(DETERMINISTIC_FIXED_STEP)),
            "deterministic_adaptive": sorted([
                m for m in DETERMINISTIC_ADAPTIVE
                if m in BACKEND_METHODS["numpy"]
            ]),
            "sde_fixed_step": sorted([
                m for m in SDE_FIXED_STEP
                if m in BACKEND_METHODS["numpy"]
            ]),
            "sde_adaptive": sorted([
                m for m in SDE_ADAPTIVE
                if m in BACKEND_METHODS["numpy"]
            ]),
            "canonical_aliases": sorted([
                "euler_maruyama", "milstein", "stratonovich_milstein",
                "sra1", "reversible_heun",
                "rk45", "rk23", "dopri5", "dopri8", "tsit5",
                "implicit_euler", "bdf", "lsoda",
            ]),
        },
        "torch": {
            "deterministic_fixed_step": sorted(list(DETERMINISTIC_FIXED_STEP & BACKEND_METHODS["torch"])),
            "deterministic_adaptive": sorted([
                m for m in DETERMINISTIC_ADAPTIVE
                if m in BACKEND_METHODS["torch"]
            ]),
            "sde_fixed_step": sorted([
                m for m in SDE_FIXED_STEP
                if m in BACKEND_METHODS["torch"]
            ]),
            "sde_adaptive": sorted([
                m for m in SDE_ADAPTIVE
                if m in BACKEND_METHODS["torch"]
            ]),
            "canonical_aliases": sorted([
                "euler_maruyama", "milstein", "sra1", "reversible_heun",
                "rk45", "rk23", "dopri5", "dopri8",
            ]),
        },
        "jax": {
            "deterministic_fixed_step": sorted(list(DETERMINISTIC_FIXED_STEP & BACKEND_METHODS["jax"])),
            "deterministic_adaptive": sorted([
                m for m in DETERMINISTIC_ADAPTIVE
                if m in BACKEND_METHODS["jax"]
            ]),
            "sde_fixed_step": sorted([
                m for m in SDE_FIXED_STEP
                if m in BACKEND_METHODS["jax"]
            ]),
            "sde_adaptive": sorted([
                m for m in SDE_ADAPTIVE
                if m in BACKEND_METHODS["jax"]
            ]),
            "canonical_aliases": sorted([
                "euler_maruyama", "milstein", "stratonovich_milstein",
                "sra1", "reversible_heun",
                "rk45", "rk23", "dopri5", "dopri8", "tsit5",
                "implicit_euler", "bdf",
            ]),
        },
    }
    
    backend_methods = all_methods.get(backend, {})
    
    # ========================================================================
    # Apply filters based on method_type
    # ========================================================================
    
    if method_type == "all":
        return backend_methods
    
    if method_type == "deterministic":
        return {
            k: v
            for k, v in backend_methods.items()
            if "deterministic" in k or k == "canonical_aliases"
        }
    
    if method_type == "stochastic":
        return {
            k: v
            for k, v in backend_methods.items()
            if "sde" in k or k == "canonical_aliases"
        }
    
    if method_type == "fixed_step":
        return {
            k: v
            for k, v in backend_methods.items()
            if "fixed_step" in k
        }
    
    if method_type == "adaptive":
        return {
            k: v
            for k, v in backend_methods.items()
            if "adaptive" in k
        }
    
    raise ValueError(
        f"Unknown method_type: {method_type}. "
        f"Must be one of: 'all', 'deterministic', 'stochastic', 'fixed_step', 'adaptive'"
    )


def validate_method(
    method: str,
    backend: Backend,
    is_stochastic: bool = False,
) -> Tuple[bool, Optional[str]]:
    """
    Validate method for backend and system type.

    Checks if a method is valid for the specified backend and system type
    (deterministic vs stochastic). Provides detailed error messages for
    invalid configurations.

    Parameters
    ----------
    method : str
        Method name to validate (canonical or backend-specific)
    backend : Backend
        Target backend: 'numpy', 'torch', or 'jax'
    is_stochastic : bool, default=False
        Whether the system is stochastic (SDE) or deterministic (ODE)

    Returns
    -------
    is_valid : bool
        True if method is valid, False otherwise
    error_message : str or None
        Description of problem if invalid, None if valid

    Validation Checks
    -----------------
    1. **Normalization**: Convert canonical names to backend-specific
    2. **Backend availability**: Check if method exists for backend
    3. **Type mismatch**: Detect SDE method on deterministic system
    
    Does NOT check:
    - Whether required packages are installed (diffeqpy, torchsde, etc.)
    - Whether method will actually work (that's for integrator factory)
    - Performance characteristics or accuracy

    Error Messages
    --------------
    Error messages are designed to be user-friendly and actionable:

    - **Method not available**: Lists available methods for backend
    - **Type mismatch**: Explains SDE method used on deterministic system
    - **Invalid backend**: Lists valid backends

    Examples
    --------
    **Valid configurations:**

    >>> # Canonical name on appropriate backend
    >>> is_valid, error = validate_method('euler_maruyama', 'torch', is_stochastic=True)
    >>> print(is_valid)
    True
    >>> print(error)
    None

    >>> # Backend-specific name
    >>> is_valid, error = validate_method('RK45', 'numpy', is_stochastic=False)
    >>> print(is_valid)
    True

    >>> # Manual implementation (works everywhere)
    >>> is_valid, error = validate_method('rk4', 'jax', is_stochastic=False)
    >>> print(is_valid)
    True

    **Invalid configurations:**

    >>> # Method doesn't exist for backend
    >>> is_valid, error = validate_method('LSODA', 'torch', is_stochastic=False)
    >>> print(is_valid)
    False
    >>> print(error)
    Method 'LSODA' not available for torch backend. Available methods: ...

    >>> # SDE method on deterministic system
    >>> is_valid, error = validate_method('euler_maruyama', 'numpy', is_stochastic=False)
    >>> print(is_valid)
    False
    >>> print(error)
    SDE method 'euler_maruyama' used on deterministic system

    >>> # Deterministic method on SDE system (WARNING, not error)
    >>> is_valid, error = validate_method('rk4', 'numpy', is_stochastic=True)
    >>> print(is_valid)
    True  # Valid, but may ignore noise (handled elsewhere)

    **Usage in DiscretizedSystem:**

    >>> # Validate during initialization
    >>> normalized_method = normalize_method_name(user_method, backend)
    >>> is_valid, error = validate_method(normalized_method, backend, is_stochastic)
    >>> 
    >>> if not is_valid:
    ...     raise ValueError(f"Invalid method configuration: {error}")

    **Pre-flight check before expensive computation:**

    >>> # Check multiple configurations
    >>> configurations = [
    ...     ('rk4', 'numpy', False),
    ...     ('euler_maruyama', 'torch', True),
    ...     ('LSODA', 'jax', False),  # This will fail
    ... ]
    >>> 
    >>> for method, backend, is_stochastic in configurations:
    ...     is_valid, error = validate_method(method, backend, is_stochastic)
    ...     if is_valid:
    ...         print(f"✓ {method} on {backend}")
    ...     else:
    ...         print(f"✗ {method} on {backend}: {error}")

    **Generate user-friendly error messages:**

    >>> def create_discretization(system, method, backend):
    ...     is_valid, error = validate_method(
    ...         method, backend, system.is_stochastic
    ...     )
    ...     
    ...     if not is_valid:
    ...         # Show user what went wrong
    ...         print(f"Error: {error}")
    ...         
    ...         # Show alternatives
    ...         available = get_available_methods(
    ...             backend,
    ...             method_type='stochastic' if system.is_stochastic else 'deterministic'
    ...         )
    ...         print(f"Try one of: {available['canonical_aliases']}")
    ...         return None
    ...     
    ...     return DiscretizedSystem(system, dt=0.01, method=method)

    Notes
    -----
    - Normalization happens automatically before validation
    - Does not check if packages (diffeqpy, torchsde) are installed
    - Only validates logical consistency, not runtime availability
    - Original method name preserved in error messages for clarity
    
    **Ambiguous Method Handling:**
    Some methods appear in both deterministic and SDE categories:
    - 'euler': Valid for both contexts (determined by is_stochastic flag)
    - 'midpoint': Valid for both contexts (determined by is_stochastic flag)
    - When is_stochastic=False and method is SDE-only, validation fails
    - When is_stochastic=True and method is deterministic, validation passes
      (with warning handled elsewhere, as noise will be ignored)

    Limitations
    -----------
    - Cannot detect if specific method version is buggy
    - Cannot check if method is appropriate for problem stiffness
    - Cannot verify if tolerances are reasonable
    - Does not check hardware compatibility (GPU, TPU)

    See Also
    --------
    normalize_method_name : Normalize before validation
    get_available_methods : List all methods for backend
    is_sde_method : Check if method is for stochastic systems
    """
    # ========================================================================
    # Validate backend
    # ========================================================================
    
    if backend not in BACKEND_METHODS:
        valid_backends = list(BACKEND_METHODS.keys())
        return False, (
            f"Invalid backend: '{backend}'. "
            f"Must be one of: {valid_backends}"
        )
    
    # ========================================================================
    # Normalize method name
    # ========================================================================
    
    normalized = normalize_method_name(method, backend)
    
    # ========================================================================
    # Handle Julia/DiffEqPy auto-switching methods
    # ========================================================================
    
    # Methods like "AutoTsit5(Rosenbrock23())" are valid Julia/DiffEqPy methods
    # They won't be in BACKEND_METHODS but should be allowed on numpy backend
    if backend == "numpy" and "(" in normalized:
        # Julia auto-switching method - allow it through
        # The actual Julia backend will validate if it's a real method
        return True, None
    
    # ========================================================================
    # Check if method exists for backend
    # ========================================================================
    
    if normalized not in BACKEND_METHODS[backend]:
        # Get available methods for helpful error message
        available = get_available_methods(backend, method_type="all")
        
        # Flatten all categories for display
        all_available = set()
        for category, methods in available.items():
            if category != "canonical_aliases":
                all_available.update(methods)
        
        return False, (
            f"Method '{method}' (normalized to '{normalized}') not available "
            f"for {backend} backend. Available methods: "
            f"{sorted(list(all_available))}. "
            f"Canonical aliases: {available['canonical_aliases']}"
        )
    
    # ========================================================================
    # Check for type mismatch (SDE method on deterministic system)
    # ========================================================================
    
    # Only fail if method is EXCLUSIVELY SDE (not in deterministic sets)
    is_exclusively_sde = (
        is_sde_method(normalized) and
        normalized not in DETERMINISTIC_FIXED_STEP and
        normalized not in DETERMINISTIC_ADAPTIVE
    )
    
    if not is_stochastic and is_exclusively_sde:
        return False, (
            f"SDE method '{method}' (normalized to '{normalized}') used on "
            f"deterministic system. SDE methods are only for stochastic systems."
        )
    
    # Note: We allow deterministic methods on stochastic systems
    # (noise will be ignored, but that's handled elsewhere with warnings)
    
    # ========================================================================
    # All checks passed
    # ========================================================================
    
    return True, None


# ============================================================================
# Convenience Functions
# ============================================================================

def get_method_info(method: str, backend: Backend = "numpy") -> Dict[str, any]:
    """
    Get comprehensive information about a method.

    Parameters
    ----------
    method : str
        Method name (canonical or backend-specific)
    backend : Backend, default='numpy'
        Target backend

    Returns
    -------
    dict
        Dictionary with method information:
        - original_name : str - Method name as provided
        - normalized_name : str - Backend-specific normalized name
        - backend : str - Target backend
        - is_sde : bool - Whether method is for SDEs
        - is_fixed_step : bool - Whether method uses fixed time stepping
        - is_adaptive : bool - Whether method uses adaptive time stepping
        - is_available : bool - Whether method exists for backend
        - category : str - Method category (e.g., 'deterministic_fixed_step')

    Examples
    --------
    >>> info = get_method_info('euler_maruyama', 'torch')
    >>> print(info)
    {
        'original_name': 'euler_maruyama',
        'normalized_name': 'euler',
        'backend': 'torch',
        'is_sde': True,
        'is_fixed_step': True,
        'is_adaptive': False,
        'is_available': True,
        'category': 'sde_fixed_step'
    }
    """
    normalized = normalize_method_name(method, backend)
    is_sde = is_sde_method(normalized)
    is_fixed = is_fixed_step(normalized)
    is_available = normalized in BACKEND_METHODS.get(backend, set())
    
    # Determine category
    if is_sde:
        category = "sde_fixed_step" if is_fixed else "sde_adaptive"
    else:
        category = "deterministic_fixed_step" if is_fixed else "deterministic_adaptive"
    
    return {
        "original_name": method,
        "normalized_name": normalized,
        "backend": backend,
        "is_sde": is_sde,
        "is_fixed_step": is_fixed,
        "is_adaptive": not is_fixed,
        "is_available": is_available,
        "category": category,
    }


def list_all_methods() -> Dict[str, List[str]]:
    """
    List all integration methods across all categories.

    Returns
    -------
    dict
        Dictionary with all methods organized by category:
        - deterministic_fixed_step : list
        - deterministic_adaptive : list
        - sde_fixed_step : list
        - sde_adaptive : list
        - all_canonical : list

    Examples
    --------
    >>> methods = list_all_methods()
    >>> print(f"Total methods: {sum(len(v) for v in methods.values())}")
    >>> print(f"SDE methods: {len(methods['sde_fixed_step']) + len(methods['sde_adaptive'])}")
    """
    # Get unique canonical aliases across all normalization map entries
    canonical_aliases = set()
    for method_name in NORMALIZATION_MAP.keys():
        canonical_aliases.add(method_name)
    
    return {
        "deterministic_fixed_step": sorted(list(DETERMINISTIC_FIXED_STEP)),
        "deterministic_adaptive": sorted(list(DETERMINISTIC_ADAPTIVE)),
        "sde_fixed_step": sorted(list(SDE_FIXED_STEP)),
        "sde_adaptive": sorted(list(SDE_ADAPTIVE)),
        "all_canonical": sorted(list(canonical_aliases)),
    }
    
def get_implementing_library(method: str, backend: Backend, is_stochastic: bool = False) -> str:
    """
    Get which library/package implements this method.
    
    Returns
    -------
    str
        One of: 'scipy', 'diffeqpy', 'torchdiffeq', 'torchsde', 
        'diffrax', 'manual', 'unknown'
    
    Examples
    --------
    >>> get_implementing_library('LSODA', 'numpy', is_stochastic=False)
    'scipy'
    >>> get_implementing_library('Tsit5', 'numpy', is_stochastic=False)
    'diffeqpy'
    >>> get_implementing_library('euler', 'torch', is_stochastic=True)
    'torchsde'
    >>> get_implementing_library('euler', 'numpy', is_stochastic=False)
    'manual'
    >>> get_implementing_library('rk4', 'jax', is_stochastic=False)
    'manual'
    """
    # ========================================================================
    # Manual implementations (only for deterministic on NumPy)
    # ========================================================================
    
    manual_methods = {"euler", "heun", "midpoint", "rk4"}
    
    if method in manual_methods and backend == "numpy" and not is_stochastic:
        return "manual"
    
    # ========================================================================
    # Scipy (NumPy only, deterministic only, specific set)
    # ========================================================================
    
    if backend == "numpy" and not is_stochastic:
        scipy_methods = {"LSODA", "RK45", "RK23", "DOP853", "Radau", "BDF"}
        if method in scipy_methods:
            return "scipy"
    
    # ========================================================================
    # DiffEqPy (NumPy only, Capital letter heuristic)
    # ========================================================================
    
    if backend == "numpy":
        # Auto-switching methods (contain parentheses)
        if "(" in method:
            return "diffeqpy"
        
        # Capital first letter, excluding Scipy methods
        if method and method[0].isupper():
            scipy_methods = {"LSODA", "RK45", "RK23", "DOP853", "Radau", "BDF"}
            if method not in scipy_methods:
                return "diffeqpy"
    
    # ========================================================================
    # TorchDiffEq (PyTorch, deterministic)
    # ========================================================================
    
    if backend == "torch" and not is_stochastic:
        torchdiffeq_methods = {"dopri5", "dopri8", "bosh3", "fehlberg2", 
                               "explicit_adams", "implicit_adams"}
        if method in torchdiffeq_methods:
            return "torchdiffeq"
        
        # Manual methods on torch (deterministic only)
        if method in manual_methods:
            return "manual"
    
    # ========================================================================
    # TorchSDE (PyTorch, stochastic)
    # ========================================================================
    
    if backend == "torch" and is_stochastic:
        torchsde_methods = {"euler", "milstein", "srk", "midpoint", 
                           "reversible_heun", "adaptive_heun"}
        if method in torchsde_methods:
            return "torchsde"
    
    # ========================================================================
    # Diffrax (JAX, both ODE and SDE)
    # ========================================================================
    
    if backend == "jax":
        # Manual methods on JAX (deterministic only)
        if method in manual_methods and not is_stochastic:
            return "manual"
        
        # Everything else is Diffrax
        return "diffrax"
    
    # ========================================================================
    # Unknown/unsupported
    # ========================================================================
    
    return "unknown"


# ============================================================================
# Module-level convenience
# ============================================================================

__all__ = [
    # Classification functions
    "is_sde_method",
    "is_fixed_step",
    "normalize_method_name",
    "validate_method",
    "get_available_methods",
    "get_method_info",
    "list_all_methods",
    "get_implementing_library",
    
    # Constants (for advanced usage)
    "DETERMINISTIC_FIXED_STEP",
    "DETERMINISTIC_ADAPTIVE",
    "SDE_FIXED_STEP",
    "SDE_ADAPTIVE",
    "SDE_METHODS",
    "NORMALIZATION_MAP",
    "BACKEND_METHODS",
]