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
Discretized System - Numerical Discretization of Continuous Systems
====================================================================

Provides discrete-time approximation via three modes: FIXED_STEP, 
DENSE_OUTPUT, and BATCH_INTERPOLATION.

This module provides DiscretizedSystem, which wraps any ContinuousSystemBase
and provides a discrete interface through numerical integration.

Protocol Satisfaction
--------------------
DiscretizedSystem satisfies:
- ✓ DiscreteSystemProtocol (step, simulate)
- ✓ LinearizableDiscreteProtocol (linearize via ZOH)
- ✗ SymbolicDiscreteProtocol (no symbolic expressions - purely numerical)

This is CORRECT - discretization is numerical, not symbolic!

NOTE: DiscretizedSystem itself doesn't provide symbolic discrete-time expressions, 
even when wrapping symbolic continuous systems. 

See class docstring for complete documentation.
"""

import inspect
import time
from enum import Enum
from typing import Callable, Optional, Sequence, Tuple, List

import numpy as np
from scipy.interpolate import interp1d

from src.systems.base.core.continuous_system_base import ContinuousSystemBase
from src.systems.base.core.discrete_system_base import DiscreteSystemBase
from src.systems.base.numerical_integration.integrator_factory import IntegratorFactory
from src.systems.base.numerical_integration.stochastic.sde_integrator_factory import SDEIntegratorFactory

from src.types.core import ControlVector, DiscreteControlInput, StateVector
from src.types.backends import Backend
from src.types.linearization import DiscreteLinearization
from src.types.trajectories import DiscreteSimulationResult


class DiscretizationMode(Enum):
    """Three discretization modes balancing accuracy and efficiency."""
    FIXED_STEP = "fixed_step"
    DENSE_OUTPUT = "dense_output"
    BATCH_INTERPOLATION = "batch_interpolation"


class DiscretizedSystem(DiscreteSystemBase):
    """
    Pure wrapper providing discrete interface to continuous systems.
    
    Protocol Satisfaction
    --------------------
    This class satisfies:
    - DiscreteSystemProtocol: Has step(), simulate(), dt, nx, nu
    - LinearizableDiscreteProtocol: Has linearize() (wraps continuous)
    
    Does NOT satisfy:
    - SymbolicDiscreteProtocol: No symbolic machinery (purely numerical)
    
    This means it can be used in:
    - ✓ Any function expecting DiscreteSystemProtocol
    - ✓ Control design (LQR, MPC) expecting LinearizableDiscreteProtocol
    - ✗ Code generation expecting SymbolicDiscreteProtocol
    
    Examples
    --------
    >>> from src.types.protocols import LinearizableDiscreteProtocol
    >>> 
    >>> def lqr_design(system: LinearizableDiscreteProtocol, Q, R):
    ...     Ad, Bd = system.linearize(np.zeros(system.nx), np.zeros(system.nu))
    ...     # ... LQR computation
    >>> 
    >>> # DiscretizedSystem works here:
    >>> continuous = Pendulum(m=1.0, l=0.5)
    >>> discrete = DiscretizedSystem(continuous, dt=0.01)
    >>> K = lqr_design(discrete, Q, R)  # ✓ Type checks pass!
    """
    
    # ========================================================================
    # Deterministic Methods
    # ========================================================================
    
    # Fixed-step deterministic methods (manual implementations + some backends)
    _DETERMINISTIC_FIXED_STEP = frozenset([
        'euler', 'midpoint', 'rk4', 'heun',  # Manual implementations (all backends)
    ])
    
    # Adaptive deterministic methods (scipy + backends)
    _DETERMINISTIC_ADAPTIVE = frozenset([
        # Scipy methods
        'RK45', 'RK23', 'DOP853', 'Radau', 'BDF', 'LSODA',
        # TorchDiffEq adaptive
        'dopri5', 'dopri8', 'bosh3', 'fehlberg2', 'adaptive_heun',
        'explicit_adams', 'implicit_adams',
        # Diffrax adaptive (lowercase)
        'tsit5', 'dopri5', 'dopri8', 'bosh3',
        'implicit_euler', 'kvaerno3', 'kvaerno4', 'kvaerno5',
        # Julia/DiffEqPy adaptive (capitalized)
        'Tsit5', 'Vern6', 'Vern7', 'Vern8', 'Vern9',
        'DP5', 'DP8', 'Rosenbrock23', 'Rodas5', 'ROCK4',
    ])
    
    # ========================================================================
    # Stochastic (SDE) Methods
    # ========================================================================
    
    # SDE methods - organized by backend for clarity
    _SDE_METHODS = frozenset([
        # Canonical simplified names (preferred for user API)
        'euler_maruyama', 'milstein',
        
        # NumPy/Julia (DiffEqPy) - capitalized
        'EM', 'LambaEM', 'EulerHeun',
        'SRIW1', 'SRIW2', 'SRA1', 'SRA3',
        'RKMil', 'ImplicitEM',
        
        # PyTorch (TorchSDE) - lowercase
        'euler', 'milstein', 'srk', 'midpoint',  
        'reversible_heun', 'adaptive_heun',
        
        # JAX (Diffrax) - PascalCase
        'Euler', 'EulerHeun', 'Heun',
        'ItoMilstein', 'StratonovichMilstein',
        'SEA', 'SHARK', 'SRA1', 'ReversibleHeun',
    ])
    
    # Adaptive SDE methods (rare, but they exist)
    _SDE_ADAPTIVE = frozenset([
        'AutoEM',  # Julia
        'adaptive_heun', 'reversible_heun',  # TorchSDE
        'LambaEM',  # Julia adaptive
    ])
    
    # ========================================================================
    # Method Classification Helpers
    # ========================================================================
    
    @classmethod
    def _is_method_sde(cls, method: str) -> bool:
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
        - Checks against the class-level `_SDE_METHODS` frozenset
        - Works with both canonical names ('euler_maruyama') and backend-specific
        names ('EM', 'euler', 'Euler')
        - Method name should ideally be normalized first, but this works with
        any name in the _SDE_METHODS set
        - Returns False for deterministic methods (euler, rk4, RK45, etc.)
        
        Examples
        --------
        >>> # Canonical SDE names
        >>> DiscretizedSystem._is_method_sde('euler_maruyama')
        True
        >>> DiscretizedSystem._is_method_sde('milstein')
        True
        
        >>> # Backend-specific SDE names
        >>> DiscretizedSystem._is_method_sde('EM')  # NumPy/Julia
        True
        >>> DiscretizedSystem._is_method_sde('euler')  # PyTorch (ambiguous!)
        True
        >>> DiscretizedSystem._is_method_sde('ItoMilstein')  # JAX
        True
        
        >>> # Deterministic methods
        >>> DiscretizedSystem._is_method_sde('rk4')
        False
        >>> DiscretizedSystem._is_method_sde('RK45')
        False
        >>> DiscretizedSystem._is_method_sde('dopri5')
        False
        
        >>> # Unknown methods
        >>> DiscretizedSystem._is_method_sde('my_custom_method')
        False
        
        Ambiguous Cases
        ---------------
        Some method names appear in both deterministic and stochastic contexts:
        
        - 'euler': Both a deterministic method (Forward Euler) and SDE method
        (Euler-Maruyama for PyTorch/TorchSDE). In _SDE_METHODS, so returns True.
        - 'midpoint': Similar ambiguity. Returns True (in _SDE_METHODS).
        
        For these cases, normalization to canonical names ('euler_maruyama' vs 'rk4')
        is recommended before calling this method.
        
        See Also
        --------
        _is_method_fixed_step : Classify method by time-stepping strategy
        _normalize_method_name : Convert canonical names to backend-specific
        """
        return method in cls._SDE_METHODS

    @classmethod
    def _is_method_fixed_step(cls, method: str) -> bool:
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
        >>> DiscretizedSystem._is_method_fixed_step('euler')
        True
        >>> DiscretizedSystem._is_method_fixed_step('rk4')
        True
        >>> DiscretizedSystem._is_method_fixed_step('heun')
        True
        
        >>> # Deterministic adaptive methods
        >>> DiscretizedSystem._is_method_fixed_step('RK45')
        False
        >>> DiscretizedSystem._is_method_fixed_step('LSODA')
        False
        >>> DiscretizedSystem._is_method_fixed_step('dopri5')  # PyTorch
        False
        >>> DiscretizedSystem._is_method_fixed_step('tsit5')  # JAX
        False
        
        >>> # SDE methods (mostly fixed-step)
        >>> DiscretizedSystem._is_method_fixed_step('euler_maruyama')
        True
        >>> DiscretizedSystem._is_method_fixed_step('EM')  # NumPy/Julia
        True
        >>> DiscretizedSystem._is_method_fixed_step('milstein')
        True
        >>> DiscretizedSystem._is_method_fixed_step('SRIW1')  # Julia SDE
        True
        
        >>> # Rare adaptive SDE methods
        >>> DiscretizedSystem._is_method_fixed_step('LambaEM')  # Julia adaptive
        False
        >>> DiscretizedSystem._is_method_fixed_step('AutoEM')  # Julia adaptive
        False
        >>> DiscretizedSystem._is_method_fixed_step('adaptive_heun')  # PyTorch
        False
        
        >>> # Unknown method (conservative default)
        >>> DiscretizedSystem._is_method_fixed_step('my_custom_method')
        False
        
        Usage in __init__
        -----------------
        >>> # Auto-select mode based on method
        >>> discrete = DiscretizedSystem(system, dt=0.01, method='rk4')
        >>> # _is_method_fixed_step('rk4') = True
        >>> # → mode auto-selected as FIXED_STEP
        
        >>> discrete = DiscretizedSystem(system, dt=0.01, method='RK45')
        >>> # _is_method_fixed_step('RK45') = False
        >>> # → mode auto-selected as DENSE_OUTPUT
        
        Edge Cases
        ----------
        **Ambiguous names** (appear in multiple contexts):
        - 'euler': In both _DETERMINISTIC_FIXED_STEP and _SDE_METHODS
        - Returns True (deterministic check comes first)
        - 'midpoint': Similar ambiguity, returns True
        
        **Method not in any set**:
        - Returns False (conservative default)
        - User can still manually specify mode=FIXED_STEP if desired
        
        See Also
        --------
        _is_method_sde : Check if method is for stochastic systems
        DiscretizationMode : Enum defining discretization strategies
        """
        # Deterministic fixed-step
        if method in cls._DETERMINISTIC_FIXED_STEP:
            return True
        
        # Deterministic adaptive
        if method in cls._DETERMINISTIC_ADAPTIVE:
            return False
        
        # SDE methods - mostly fixed-step
        if method in cls._SDE_METHODS:
            # Check if it's one of the rare adaptive SDE methods
            return method not in cls._SDE_ADAPTIVE
        
        # Unknown method - conservative default
        # Assume adaptive (more flexible, works for both fixed and adaptive)
        return False
    
    # ========================================================================
    # Name Normalization
    # ========================================================================
    
    @staticmethod
    def _normalize_method_name(method: str, backend: Backend = 'numpy') -> str:
        """
        Normalize method names across backends to canonical form.
        
        This provides user-friendly aliases while maintaining backend compatibility.
        Handles both deterministic and stochastic (SDE) methods.
        
        Parameters
        ----------
        method : str
            User-provided method name (canonical or backend-specific)
        backend : str, default='numpy'
            Target backend: 'numpy', 'torch', 'jax'
        
        Returns
        -------
        str
            Normalized method name appropriate for backend
        
        Notes
        -----
        - If method is already backend-appropriate, returns unchanged
        - If method is canonical (simplified), maps to backend-specific name
        - If method is from different backend, attempts cross-backend mapping
        - Unknown methods pass through unchanged (will fail at integrator level)
        
        Backend Conventions
        -------------------
        - NumPy/Julia (DiffEqPy): Capitalized (e.g., 'EM', 'Tsit5', 'SRIW1')
        - PyTorch (TorchSDE/TorchDiffEq): lowercase (e.g., 'euler', 'dopri5')
        - JAX (Diffrax): PascalCase (e.g., 'Euler', 'ItoMilstein', 'Tsit5')
        
        Examples
        --------
        >>> # Canonical SDE names -> backend-specific
        >>> DiscretizedSystem._normalize_method_name('euler_maruyama', 'numpy')
        'EM'
        >>> DiscretizedSystem._normalize_method_name('euler_maruyama', 'torch')
        'euler'
        >>> DiscretizedSystem._normalize_method_name('euler_maruyama', 'jax')
        'Euler'
        
        >>> # Canonical ODE names -> backend-specific
        >>> DiscretizedSystem._normalize_method_name('rk45', 'numpy')
        'RK45'
        >>> DiscretizedSystem._normalize_method_name('rk45', 'torch')
        'dopri5'
        >>> DiscretizedSystem._normalize_method_name('rk45', 'jax')
        'tsit5'
        
        >>> # Cross-backend normalization (torch -> jax)
        >>> DiscretizedSystem._normalize_method_name('dopri5', 'jax')
        'tsit5'
        
        >>> # Already correct for backend -> unchanged
        >>> DiscretizedSystem._normalize_method_name('EM', 'numpy')
        'EM'
        >>> DiscretizedSystem._normalize_method_name('ItoMilstein', 'jax')
        'ItoMilstein'
        
        >>> # Unknown method -> pass through
        >>> DiscretizedSystem._normalize_method_name('my_custom_method', 'numpy')
        'my_custom_method'
        """
        # ====================================================================
        # Normalization Map: canonical/alias -> backend-specific
        # ====================================================================
        
        normalization_map = {
            # ================================================================
            # Stochastic (SDE) Methods
            # ================================================================
            
            # Euler-Maruyama (most common SDE method)
            'euler_maruyama': {
                'numpy': 'EM',           # Julia/DiffEqPy
                'torch': 'euler',        # TorchSDE
                'jax': 'Euler',          # Diffrax
            },
            
            # Milstein method
            'milstein': {
                'numpy': 'RKMil',        # Julia (Runge-Kutta-Milstein)
                'torch': 'milstein',     # TorchSDE
                'jax': 'ItoMilstein',    # Diffrax (Itô interpretation)
            },
            
            # Stratonovich Milstein
            'stratonovich_milstein': {
                'numpy': 'RKMil',        # Julia (can handle Stratonovich)
                'torch': 'milstein',     # TorchSDE (typically Itô)
                'jax': 'StratonovichMilstein',  # Diffrax
            },
            
            # SRA1 (diagonal noise, order 1.5 strong)
            'sra1': {
                'numpy': 'SRA1',         # Julia
                'torch': 'srk',          # TorchSDE (closest equivalent)
                'jax': 'SRA1',           # Diffrax
            },
            
            # Reversible/Symmetric Heun (for time-reversible integration)
            'reversible_heun': {
                'numpy': 'EulerHeun',    # Julia (similar)
                'torch': 'reversible_heun',  # TorchSDE
                'jax': 'ReversibleHeun',     # Diffrax
            },
            
            # ================================================================
            # Deterministic Methods - Fixed Step
            # ================================================================
            
            # These work across all backends (manual implementations)
            # 'euler', 'midpoint', 'rk4', 'heun' -> no normalization needed
            
            # ================================================================
            # Deterministic Methods - Adaptive
            # ================================================================
            
            # RK45 family (Dormand-Prince 5(4))
            'rk45': {
                'numpy': 'RK45',         # Scipy
                'torch': 'dopri5',       # TorchDiffEq
                'jax': 'tsit5',          # Diffrax (Tsitouras 5/4, similar performance)
            },
            'dopri5': {
                'numpy': 'RK45',         # Scipy (DP5(4) = RK45)
                'torch': 'dopri5',       # TorchDiffEq
                'jax': 'dopri5',         # Diffrax also has dopri5
            },
            
            # RK23 family (Bogacki-Shampine 3(2))
            'rk23': {
                'numpy': 'RK23',         # Scipy
                'torch': 'bosh3',        # TorchDiffEq
                'jax': 'bosh3',          # Diffrax
            },
            
            # High-order methods
            'dopri8': {
                'numpy': 'DOP853',       # Scipy (8th order)
                'torch': 'dopri8',       # TorchDiffEq
                'jax': 'dopri8',         # Diffrax
            },
            
            # Tsitouras 5/4 (modern, efficient)
            'tsit5': {
                'numpy': 'Tsit5',        # Julia
                'torch': 'dopri5',       # TorchDiffEq (closest)
                'jax': 'tsit5',          # Diffrax
            },
            
            # Implicit methods for stiff systems
            'implicit_euler': {
                'numpy': 'Radau',        # Scipy (implicit RK)
                'torch': 'implicit_adams',  # TorchDiffEq
                'jax': 'implicit_euler',    # Diffrax
            },
            
            # Stiff ODE solvers
            'bdf': {
                'numpy': 'BDF',          # Scipy
                'torch': 'implicit_adams',  # TorchDiffEq (closest)
                'jax': 'kvaerno5',       # Diffrax (ESDIRK for stiff)
            },
            
            # Auto stiffness detection
            'lsoda': {
                'numpy': 'LSODA',        # Scipy
                'torch': 'dopri5',       # TorchDiffEq (no LSODA equivalent)
                'jax': 'tsit5',          # Diffrax (no LSODA equivalent)
            },
        }
        
        # ====================================================================
        # Method Already Valid Check
        # ====================================================================
        
        # Check if method is already appropriate for backend
        # (avoids unnecessary mapping when user provides correct name)
        
        if backend == 'numpy':
            # NumPy backend uses Julia/DiffEqPy + Scipy
            valid_numpy = {
                # Julia capitalized methods
                'EM', 'LambaEM', 'EulerHeun', 'SRIW1', 'SRIW2', 'SRA1', 'SRA3',
                'RKMil', 'ImplicitEM', 'AutoEM',
                'Tsit5', 'Vern6', 'Vern7', 'Vern8', 'Vern9',
                'DP5', 'DP8', 'Rosenbrock23', 'Rodas5', 'ROCK4',
                # Scipy methods
                'RK45', 'RK23', 'DOP853', 'Radau', 'BDF', 'LSODA',
                # Manual implementations
                'euler', 'midpoint', 'rk4', 'heun',
            }
            if method in valid_numpy:
                return method
        
        elif backend == 'torch':
            # PyTorch backend uses TorchSDE + TorchDiffEq
            valid_torch = {
                # TorchSDE lowercase
                'euler', 'milstein', 'srk', 'midpoint',
                'reversible_heun', 'adaptive_heun',
                # TorchDiffEq lowercase
                'dopri5', 'dopri8', 'bosh3', 'fehlberg2', 'adaptive_heun',
                'explicit_adams', 'implicit_adams',
                # Manual implementations
                'euler', 'midpoint', 'rk4', 'heun',
            }
            if method in valid_torch:
                return method
        
        elif backend == 'jax':
            # JAX backend uses Diffrax
            valid_jax = {
                # Diffrax SDE methods (PascalCase)
                'Euler', 'EulerHeun', 'Heun',
                'ItoMilstein', 'StratonovichMilstein',
                'SEA', 'SHARK', 'SRA1', 'ReversibleHeun',
                # Diffrax ODE methods (lowercase/PascalCase mix)
                'tsit5', 'dopri5', 'dopri8', 'bosh3',
                'implicit_euler', 'kvaerno3', 'kvaerno4', 'kvaerno5',
                # Manual implementations
                'euler', 'midpoint', 'rk4', 'heun',
            }
            if method in valid_jax:
                return method
        
        # ====================================================================
        # Apply Normalization
        # ====================================================================
        
        # Convert method to lowercase for case-insensitive lookup
        method_lower = method.lower()
        
        # Try exact match first
        if method in normalization_map and backend in normalization_map[method]:
            return normalization_map[method][backend]
        
        # Try case-insensitive match
        if method_lower in normalization_map and backend in normalization_map[method_lower]:
            return normalization_map[method_lower][backend]
        
        # ====================================================================
        # No normalization found - return as-is
        # ====================================================================
        
        # Let integrator factory handle validation and error messages
        return method
    
    @staticmethod
    def get_available_methods(backend: Backend = 'numpy', method_type: str = 'all') -> dict:
        """
        Get available integration methods for a backend.
        
        Parameters
        ----------
        backend : str
            'numpy', 'torch', or 'jax'
        method_type : str
            'all', 'deterministic', 'stochastic', 'fixed_step', 'adaptive'
        
        Returns
        -------
        dict
            Dictionary with method categories and their available methods
        
        Examples
        --------
        >>> methods = DiscretizedSystem.get_available_methods('torch', 'stochastic')
        >>> print(methods['sde_fixed_step'])
        ['euler', 'milstein', 'srk', 'midpoint']
        """
        all_methods = {
            'numpy': {
                'deterministic_fixed_step': ['euler', 'midpoint', 'rk4', 'heun'],
                'deterministic_adaptive': [
                    'RK45', 'RK23', 'DOP853', 'Radau', 'BDF', 'LSODA',
                    'Tsit5', 'Vern6', 'Vern7', 'Vern8', 'Vern9',
                    'DP5', 'DP8', 'Rosenbrock23', 'Rodas5', 'ROCK4'
                ],
                'sde_fixed_step': [
                    'EM', 'EulerHeun', 'SRIW1', 'SRIW2', 'SRA1', 'SRA3',
                    'RKMil', 'ImplicitEM'
                ],
                'sde_adaptive': ['LambaEM', 'AutoEM'],
                'canonical_aliases': ['euler_maruyama', 'milstein', 'rk45', 'rk23', 'lsoda'],
            },
            'torch': {
                'deterministic_fixed_step': ['euler', 'midpoint', 'rk4', 'heun'],
                'deterministic_adaptive': [
                    'dopri5', 'dopri8', 'bosh3', 'fehlberg2',
                    'explicit_adams', 'implicit_adams', 'adaptive_heun'
                ],
                'sde_fixed_step': ['euler', 'milstein', 'srk', 'midpoint'],
                'sde_adaptive': ['reversible_heun', 'adaptive_heun'],
                'canonical_aliases': ['euler_maruyama', 'milstein', 'rk45', 'rk23'],
            },
            'jax': {
                'deterministic_fixed_step': ['euler', 'midpoint', 'rk4', 'heun'],
                'deterministic_adaptive': [
                    'tsit5', 'dopri5', 'dopri8', 'bosh3',
                    'implicit_euler', 'kvaerno3', 'kvaerno4', 'kvaerno5'
                ],
                'sde_fixed_step': [
                    'Euler', 'EulerHeun', 'Heun',
                    'ItoMilstein', 'StratonovichMilstein',
                    'SEA', 'SHARK', 'SRA1'
                ],
                'sde_adaptive': ['ReversibleHeun'],
                'canonical_aliases': ['euler_maruyama', 'milstein', 'rk45', 'rk23', 'tsit5'],
            }
        }
        
        backend_methods = all_methods.get(backend, {})
        
        if method_type == 'all':
            return backend_methods
        elif method_type == 'deterministic':
            return {
                k: v for k, v in backend_methods.items()
                if 'deterministic' in k or k == 'canonical_aliases'
            }
        elif method_type == 'stochastic':
            return {
                k: v for k, v in backend_methods.items()
                if 'sde' in k or k == 'canonical_aliases'
            }
        elif method_type == 'fixed_step':
            return {
                k: v for k, v in backend_methods.items()
                if 'fixed_step' in k
            }
        elif method_type == 'adaptive':
            return {
                k: v for k, v in backend_methods.items()
                if 'adaptive' in k
            }
        else:
            raise ValueError(f"Unknown method_type: {method_type}")
    
    def __init__(
        self,
        continuous_system: ContinuousSystemBase,
        dt: float = 0.01,
        method: str = 'rk4',
        mode: Optional[DiscretizationMode] = None,
        interpolation_kind: str = 'linear',
        auto_detect_sde: bool = True,
        sde_method: Optional[str] = None,
        **integrator_kwargs
    ):
        """
        Initialize discretized system wrapper.

        Parameters
        ----------
        continuous_system : ContinuousSystemBase
            Continuous system to discretize. Supports:
            - ContinuousSystemBase (deterministic)
            - ContinuousSymbolicSystem (symbolic)
            - ContinuousStochasticSystem (stochastic)
        dt : float, default=0.01
            Sampling time step (seconds)
        method : str, default='rk4'
            Integration method. Available methods depend on system type and backend.
            
            **Method names are automatically normalized** to match the backend's
            conventions. You can use either canonical names (recommended) or 
            backend-specific names - both work seamlessly.
            
            **Deterministic Fixed-Step Methods:**
            - 'euler': Forward Euler (1st order)
            - 'midpoint': Midpoint/RK2 (2nd order)
            - 'rk4': Classic Runge-Kutta 4 (4th order) [DEFAULT]
            - 'heun': Heun's method (2nd order)
            
            **Deterministic Adaptive Methods:**
            
            *Canonical names (work on all backends):*
            - 'rk45': Dormand-Prince 5(4) - general purpose
            - 'rk23': Bogacki-Shampine 3(2) - lower accuracy
            - 'dopri8': Dormand-Prince 8 - high accuracy
            - 'tsit5': Tsitouras 5(4) - modern, efficient
            - 'bdf': Backward Differentiation - very stiff
            - 'lsoda': Auto stiffness detection (NumPy/Scipy only)
            
            *Backend-specific names (also supported):*
            - NumPy/Julia: 'RK45', 'RK23', 'DOP853', 'Radau', 'BDF', 'LSODA',
              'Tsit5', 'Vern6', 'Vern7', 'Vern8', 'Vern9', 'DP5', 'DP8',
              'Rosenbrock23', 'Rodas5', 'ROCK4'
            - PyTorch: 'dopri5', 'dopri8', 'bosh3', 'fehlberg2',
              'explicit_adams', 'implicit_adams', 'adaptive_heun'
            - JAX: 'tsit5', 'dopri5', 'dopri8', 'bosh3', 'implicit_euler',
              'kvaerno3', 'kvaerno4', 'kvaerno5'
            
            **Stochastic (SDE) Methods:**
            
            *Canonical names (recommended - work on all backends):*
            - 'euler_maruyama': Euler-Maruyama (strong 0.5, weak 1.0)
            - 'milstein': Milstein method (strong 1.0)
            - 'stratonovich_milstein': Stratonovich interpretation
            - 'sra1': Stochastic Runge-Kutta (diagonal noise, order 1.5)
            - 'reversible_heun': Time-reversible integration
            
            *Backend-specific names (also supported):*
            - NumPy/Julia: 'EM', 'LambaEM', 'EulerHeun', 'SRIW1', 'SRIW2',
              'SRA1', 'SRA3', 'RKMil', 'ImplicitEM', 'AutoEM'
            - PyTorch: 'euler', 'milstein', 'srk', 'midpoint',
              'reversible_heun', 'adaptive_heun'
            - JAX: 'Euler', 'EulerHeun', 'Heun', 'ItoMilstein',
              'StratonovichMilstein', 'SEA', 'SHARK', 'SRA1', 'ReversibleHeun'
            
            **Note on method selection:**
            - For stochastic systems, prefer specifying `sde_method` explicitly
            - Canonical names ('euler_maruyama', 'milstein', 'rk45') are 
              automatically converted to backend-appropriate names
            - Use `DiscretizedSystem.get_available_methods(backend)` to see
              all methods for a specific backend
            
        mode : DiscretizationMode, optional
            Discretization mode. If None, auto-selected based on method:
            - FIXED_STEP for fixed-step methods (euler, rk4, euler_maruyama, etc.)
            - DENSE_OUTPUT for adaptive methods (rk45, lsoda, etc.)
            - BATCH_INTERPOLATION must be explicitly requested
        interpolation_kind : str, default='linear'
            Interpolation method for BATCH mode:
            - 'linear': Fast, robust (default)
            - 'cubic': Smoother, requires ≥4 adaptive points
        auto_detect_sde : bool, default=True
            For stochastic systems: warn if using deterministic method.
            Set to False to suppress warnings when intentionally using
            deterministic integration on stochastic systems (ignores noise).
        sde_method : str, optional
            Explicitly specify SDE method for stochastic systems.
            Overrides method parameter. Recommended options:
            - 'euler_maruyama': General purpose, fast (canonical name)
            - 'milstein': Higher accuracy (canonical name)
            - Backend-specific names also supported (e.g., 'EM', 'ItoMilstein')
            
            If specified, uses this method regardless of auto_detect_sde.
            Method name is automatically normalized to backend conventions.
        **integrator_kwargs
            Additional arguments passed to integrator:
            - rtol : float (default: 1e-6) - Relative tolerance (adaptive only)
            - atol : float (default: 1e-8) - Absolute tolerance (adaptive only)
            - max_steps : int - Maximum integration steps
            - seed : int - Random seed (SDE methods only)
            - adjoint : bool - Use adjoint method (JAX/PyTorch only)

        Raises
        ------
        TypeError
            If continuous_system is not ContinuousSystemBase
        ValueError
            If dt <= 0, or invalid mode/method combination

        Examples
        --------
        **Deterministic Systems:**

        >>> # Basic deterministic system with RK4
        >>> from src.systems.examples import Pendulum
        >>> continuous = Pendulum(m=1.0, l=0.5)
        >>> discrete = DiscretizedSystem(continuous, dt=0.01, method='rk4')
        >>> result = discrete.simulate(x0=np.array([0.1, 0.0]), n_steps=1000)

        >>> # High-accuracy adaptive integration using canonical name
        >>> discrete = DiscretizedSystem(continuous, dt=0.01, method='rk45', rtol=1e-9)
        >>> # 'rk45' automatically becomes 'RK45' (NumPy), 'dopri5' (PyTorch), 
        >>> # or 'tsit5' (JAX) depending on backend

        >>> # Stiff system with auto-detection
        >>> discrete = DiscretizedSystem(stiff_system, method='lsoda')

        **Stochastic Systems with Canonical Names:**

        >>> # Stochastic system with canonical SDE method (recommended)
        >>> from src.systems.examples import StochasticPendulum
        >>> stochastic = StochasticPendulum(m=1.0, l=0.5, sigma=0.1)
        >>> 
        >>> # Use canonical name - works on any backend
        >>> discrete = DiscretizedSystem(
        ...     stochastic, 
        ...     dt=0.01, 
        ...     sde_method='euler_maruyama'  # Canonical name
        ... )
        >>> # Automatically becomes 'EM' (NumPy), 'euler' (PyTorch), or 'Euler' (JAX)
        >>> 
        >>> # Single stochastic trajectory
        >>> result = discrete.simulate(x0=np.array([0.1, 0.0]), n_steps=1000)
        >>> 
        >>> # Monte Carlo simulation (100 trajectories)
        >>> mc_result = discrete.simulate_stochastic(
        ...     x0=np.array([0.1, 0.0]),
        ...     n_steps=1000,
        ...     n_trajectories=100
        ... )
        >>> print(f"Mean: {mc_result['mean_trajectory'][-1]}")
        >>> print(f"Std: {mc_result['std_trajectory'][-1]}")

        **Backend-Specific Method Names (Also Supported):**

        >>> # You can also use backend-specific names directly
        >>> discrete = DiscretizedSystem(
        ...     jax_stochastic_system,
        ...     dt=0.001,
        ...     sde_method='ItoMilstein'  # JAX/Diffrax-specific name
        ... )
        >>> # No normalization needed - already correct for JAX

        >>> # Julia/DiffEqPy high-accuracy SDE solver  
        >>> discrete = DiscretizedSystem(
        ...     numpy_stochastic_system,
        ...     dt=0.001,
        ...     sde_method='SRIW1'  # Julia-specific name for diagonal noise
        ... )

        **Intentionally Ignoring Noise:**

        >>> # Using deterministic method on stochastic system (ignores noise)
        >>> discrete = DiscretizedSystem(
        ...     stochastic, 
        ...     dt=0.01, 
        ...     method='rk4',
        ...     auto_detect_sde=False  # Suppress warning
        ... )
        >>> # Noise is ignored - useful for comparing with/without noise

        **Symbolic Systems:**

        >>> # Symbolic system works transparently
        >>> from src.systems.examples import SymbolicCartPole
        >>> symbolic = SymbolicCartPole()
        >>> discrete = DiscretizedSystem(symbolic, dt=0.01, method='rk4')
        >>> 
        >>> # Can use any integration method (canonical or backend-specific)
        >>> discrete = DiscretizedSystem(symbolic, method='rk45', rtol=1e-9)

        **Discovering Available Methods:**

        >>> # See what methods are available for your backend
        >>> methods = DiscretizedSystem.get_available_methods('jax', 'stochastic')
        >>> print(methods['sde_fixed_step'])
        ['Euler', 'EulerHeun', 'Heun', 'ItoMilstein', ...]
        >>> 
        >>> print(methods['canonical_aliases'])
        ['euler_maruyama', 'milstein', 'rk45', 'rk23', 'tsit5']

        **Checking Configuration:**

        >>> # View detailed information about the discretization
        >>> discrete.print_info()
        # Displays:
        #   - System type (deterministic/stochastic/symbolic)
        #   - Method selection (original vs normalized)
        #   - Backend detected
        #   - SDE integrator availability
        #   - Warnings if any
        
        Notes
        -----
        **Method Name Normalization:**
        Method names are automatically converted to backend-appropriate forms:
        
        - 'euler_maruyama' → 'EM' (NumPy/Julia), 'euler' (PyTorch), 'Euler' (JAX)
        - 'milstein' → 'RKMil' (NumPy/Julia), 'milstein' (PyTorch), 'ItoMilstein' (JAX)
        - 'rk45' → 'RK45' (NumPy/Scipy), 'dopri5' (PyTorch), 'tsit5' (JAX)
        - 'tsit5' → 'Tsit5' (NumPy/Julia), 'dopri5' (PyTorch), 'tsit5' (JAX)
        
        This normalization is automatic and transparent. You can use either
        canonical names (recommended for portability) or backend-specific names.
        The original method name is preserved in `_original_method` for debugging.
        
        **Backend Detection:**
        The backend is detected from `continuous_system._default_backend`.
        If not available, defaults to 'numpy'. Supported backends: 'numpy',
        'torch', 'jax'.
        """
        
        # ========================================================================
        # INITIALIZATION OVERVIEW
        # ========================================================================
        # 
        # High-level flow:
        # 1. Validate inputs (continuous_system type, dt > 0)
        # 2. Store basic attributes (system, dt, kwargs)
        # 3. **Normalize method names** for backend compatibility
        # 4. Detect system type (stochastic, symbolic)
        # 5. **Method selection logic** (complex - see decision tree below)
        # 6. Classify method (fixed-step vs adaptive)
        # 7. Determine discretization mode (FIXED_STEP, DENSE_OUTPUT, BATCH)
        # 8. Validate mode/method compatibility
        # 9. Handle symbolic system metadata
        # 10. Store metadata for get_info()
        # 
        # Most complexity is in step 5 (method selection for stochastic systems)
        # 
        # ========================================================================
        
        # ========================================================================
        # Validation
        # ========================================================================
        
        if not isinstance(continuous_system, ContinuousSystemBase):
            raise TypeError(
                f"Expected ContinuousSystemBase (or subclass), "
                f"got {type(continuous_system).__name__}"
            )
        
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt}")
        
        # ========================================================================
        # Store basic attributes
        # ========================================================================
        
        self._continuous_system = continuous_system
        self._dt = float(dt)
        self._interpolation_kind = interpolation_kind
        self._integrator_kwargs = integrator_kwargs
        self._original_method = method  # Store original before any modifications
        
        # ========================================================================
        # Normalize method names for backend
        # ========================================================================
        
        # Normalize method names to backend-specific conventions BEFORE
        # any classification or decision logic
        if hasattr(continuous_system, '_default_backend'):
            backend = continuous_system._default_backend
            method = self._normalize_method_name(method, backend)
            
            # Also normalize sde_method if provided
            if sde_method is not None:
                sde_method = self._normalize_method_name(sde_method, backend)
        else:
            # Default to numpy backend if not specified
            backend = 'numpy'
            method = self._normalize_method_name(method, backend)
            if sde_method is not None:
                sde_method = self._normalize_method_name(sde_method, backend)
        
        # ========================================================================
        # Detect system type
        # ========================================================================
        
        self._is_stochastic = continuous_system.is_stochastic
        self._is_symbolic = hasattr(continuous_system, '_f_sym')
        
        # Check if we can import stochastic system types
        try:
            from src.systems.base.core.continuous_stochastic_system import ContinuousStochasticSystem
            self._has_stochastic_module = True
        except ImportError:
            self._has_stochastic_module = False
        
        # ========================================================================
        # Handle stochastic systems
        # ========================================================================
        
        if self._is_stochastic:
            # ====================================================================
            # Method Selection Decision Flow for Stochastic Systems
            # ====================================================================
            # 
            # DECISION TREE:
            # 
            # 1. sde_method explicitly provided?
            #    ├─ YES: SDE integrator available?
            #    │   ├─ YES → Use sde_method (source: 'explicit')
            #    │   └─ NO  → Warn, use deterministic method (source: 'explicit_unavailable')
            #    │
            #    └─ NO: Continue to step 2
            # 
            # 2. auto_detect_sde=True AND method is deterministic?
            #    ├─ YES: SDE integrator available?
            #    │   ├─ YES → Warn, recommend SDE method, keep deterministic (source: 'deterministic_fallback')
            #    │   └─ NO  → Warn, recommend SDE method, keep deterministic (source: 'deterministic_fallback')
            #    │
            #    └─ NO: Continue to step 3
            # 
            # 3. Either:
            #    - auto_detect_sde=False (user explicitly disabled detection)
            #    - method is already an SDE method
            #    
            #    Action: Use method as-is
            #    ├─ If deterministic method + auto_detect_sde=False → Warn about noise being ignored
            #    └─ source: 'user_specified'
            # 
            # OUTCOMES:
            # - method = sde_method (if explicitly provided and available)
            # - method = original method (all other cases)
            # - self._method_source = one of:
            #   * 'explicit' - sde_method provided and used
            #   * 'explicit_unavailable' - sde_method requested but integrator missing
            #   * 'deterministic_fallback' - auto-detected stochastic but using deterministic
            #   * 'user_specified' - user explicitly chose method (or disabled auto-detect)
            # 
            # ====================================================================
            
            # Check if SDE integrator is actually available
            try:
                from src.systems.base.numerical_integration.stochastic.sde_integrator_factory import SDEIntegratorFactory
                self._has_sde_integrator = True
            except ImportError:
                self._has_sde_integrator = False
            
            # Determine method to use
            if sde_method is not None:
                # User explicitly requested SDE method
                if self._has_sde_integrator:
                    # SDE integrator available - use it
                    method = sde_method
                    self._method_source = 'explicit'
                else:
                    # SDE integrator not available - warn and keep deterministic
                    import warnings
                    warnings.warn(
                        f"Explicitly requested SDE method '{sde_method}', but SDE integration "
                        f"not available. Using deterministic method '{method}' - noise IGNORED.",
                        UserWarning,
                        stacklevel=2
                    )
                    self._method_source = 'explicit_unavailable'
            
            elif auto_detect_sde and not self._is_method_sde(method):
                # User gave deterministic method for stochastic system
                recommended_method = self._detect_sde_method(continuous_system)
                
                if self._has_sde_integrator:
                    # SDE integrator available - recommend but don't auto-switch
                    import warnings
                    warnings.warn(
                        f"Stochastic system with deterministic method '{method}'. "
                        f"Noise will be IGNORED. Consider using sde_method='{recommended_method}' "
                        f"for proper noise handling.",
                        UserWarning,
                        stacklevel=2
                    )
                    self._method_source = 'deterministic_fallback'
                else:
                    # SDE integrator not available - just warn
                    import warnings
                    warnings.warn(
                        f"Stochastic system with deterministic method '{method}'. "
                        f"SDEIntegratorFactory not available - noise will be IGNORED. "
                        f"Recommended: '{recommended_method}'.",
                        UserWarning,
                        stacklevel=2
                    )
                    self._method_source = 'deterministic_fallback'
            
            else:
                # User explicitly chose deterministic method or turned off auto-detect,
                # or already provided an SDE method
                if not self._is_method_sde(method) and not auto_detect_sde:
                    import warnings
                    warnings.warn(
                        f"Using deterministic method '{method}' on stochastic system. "
                        f"Noise will be IGNORED.",
                        UserWarning,
                        stacklevel=2
                    )
                self._method_source = 'user_specified'
        
        else:
            # Deterministic system
            self._has_sde_integrator = False
            self._method_source = 'deterministic_system'
        
        
        # CRITICAL: Set self._method AFTER all the logic above
        # At this point, 'method' variable contains the final method to use
        # (either normalized original method or sde_method if explicitly provided)
        self._method = method
        
        # ========================================================================
        # Determine if method is fixed-step (using class method)
        # ========================================================================
        
        self._is_fixed_step = self._is_method_fixed_step(method)
        
        # ========================================================================
        # Determine discretization mode
        # ========================================================================
        
        self._mode = mode if mode else (
            DiscretizationMode.FIXED_STEP if self._is_fixed_step 
            else DiscretizationMode.DENSE_OUTPUT
        )
        
        # Validate mode/method combination
        if self._mode == DiscretizationMode.FIXED_STEP and not self._is_fixed_step:
            raise ValueError(
                f"Cannot use adaptive method '{method}' with FIXED_STEP mode. "
                f"Use mode=DiscretizationMode.DENSE_OUTPUT or choose a fixed-step method."
            )
        
        # Warn about BATCH mode with stochastic systems
        if self._mode == DiscretizationMode.BATCH_INTERPOLATION and self._is_stochastic:
            import warnings
            warnings.warn(
                "BATCH_INTERPOLATION mode with stochastic system. "
                "Each call generates different trajectory due to randomness.",
                UserWarning,
                stacklevel=2
            )
        
        # ========================================================================
        # Handle symbolic systems (future optimization hooks)
        # ========================================================================
        
        if self._is_symbolic:
            # Check if symbolic Jacobian is available
            self._has_symbolic_jacobian = hasattr(continuous_system, 'get_jacobian_symbolic')
            
            # Store symbolic info for potential optimizations
            if hasattr(continuous_system, 'state_vars'):
                self._symbolic_state_vars = continuous_system.state_vars
            else:
                self._symbolic_state_vars = None
            
            if hasattr(continuous_system, 'control_vars'):
                self._symbolic_control_vars = continuous_system.control_vars
            else:
                self._symbolic_control_vars = None
        else:
            self._has_symbolic_jacobian = False
            self._symbolic_state_vars = None
            self._symbolic_control_vars = None
        
        # ========================================================================
        # Store metadata for get_info()
        # ========================================================================
        
        self._system_metadata = {
            'is_stochastic': self._is_stochastic,
            'is_symbolic': self._is_symbolic,
            'has_sde_integrator': self._has_sde_integrator,
            'has_symbolic_jacobian': self._has_symbolic_jacobian,
            'method_source': self._method_source,
            'original_method': self._original_method,
            'final_method': self._method
        }

    def _detect_sde_method(self, system) -> str:
        """
        Detect best SDE integration method for stochastic system.
        
        Parameters
        ----------
        system : ContinuousSystemBase
            System to analyze (should be stochastic)
        
        Returns
        -------
        str
            Recommended SDE method ('euler_maruyama' or 'milstein')
        
        Notes
        -----
        Decision logic:
        - Additive noise → 'euler_maruyama' (simplest, exact for additive)
        - Diagonal multiplicative noise → 'milstein' (better accuracy)
        - General multiplicative noise → 'euler_maruyama' (conservative)
        """
        if not system.is_stochastic:
            return self._method  # Not stochastic, return current method
        
        # Check for additive noise (Euler-Maruyama is exact)
        if hasattr(system, 'is_additive_noise'):
            try:
                if system.is_additive_noise():
                    return 'euler_maruyama'
            except:
                pass  # Method might not be implemented
        
        # Check for diagonal noise (Milstein is efficient)
        if hasattr(system, 'is_diagonal_noise'):
            try:
                if system.is_diagonal_noise():
                    return 'milstein'
            except:
                pass
        
        # Default: Euler-Maruyama (most general, conservative)
        return 'euler_maruyama'
    
    @property
    def dt(self) -> float:
        return self._dt
    
    @property
    def mode(self) -> DiscretizationMode:
        return self._mode
    
    @property
    def nx(self) -> int:
        return self._continuous_system.nx
    
    @property
    def nu(self) -> int:
        return self._continuous_system.nu
    
    @property
    def ny(self) -> int:
        return self._continuous_system.ny
    
    @property
    def is_stochastic(self) -> bool:
        return self._continuous_system.is_stochastic
    
    def step(self, x: StateVector, u: Optional[ControlVector] = None, k: int = 0) -> StateVector:
        if self._mode == DiscretizationMode.BATCH_INTERPOLATION:
            raise NotImplementedError("step() not supported in BATCH_INTERPOLATION mode")
        
        t_start, t_end = k * self._dt, (k + 1) * self._dt
        return self._step_fixed(x, u, t_start, t_end) if self._mode == DiscretizationMode.FIXED_STEP else self._step_dense(x, u, t_start, t_end)
    
    def _step_fixed(self, x, u, t_start, t_end):
        """
        Single fixed-step integration from t_start to t_end.
        
        Automatically selects between SDE and deterministic integration based on
        system type and method configuration. For stochastic systems with SDE
        methods, attempts SDE integration with fallback to deterministic.
        
        Parameters
        ----------
        x : StateVector
            Current state (nx,)
        u : ControlVector or None
            Control input (constant over [t_start, t_end])
        t_start : float
            Start time
        t_end : float
            End time (typically t_start + dt)
        
        Returns
        -------
        StateVector
            State at t_end (nx,)
        
        Integration Selection Logic
        ---------------------------
        1. **SDE integration** (if all conditions met):
        - System is stochastic (self._is_stochastic = True)
        - Method is SDE method (self._method in _SDE_METHODS)
        - SDE integrator available (self._has_sde_integrator = True)
        - Uses SDEIntegratorFactory with specified method
        
        2. **Deterministic integration** (otherwise):
        - Non-stochastic systems
        - Stochastic systems with deterministic methods
        - SDE integration unavailable or failed
        - Uses IntegratorFactory with specified method
        
        Fallback Behavior
        -----------------
        If SDE integration is attempted but fails, automatically falls back to
        deterministic integration (RK4) with a warning. Fallback occurs when:
        
        - **ImportError**: SDEIntegratorFactory module not available
        - Missing package: diffeqpy (NumPy), torchsde (PyTorch), diffrax (JAX)
        
        - **AttributeError**: System missing required SDE interface
        - System lacks `diffusion()` method
        - System lacks `_default_backend` attribute
        - Backend object missing expected attributes
        
        - **ValueError**: Invalid configuration for SDE integrator
        - Method not supported by backend
        - Invalid dt or integrator_kwargs
        - System dimensions mismatch (nx, nu, nw)
        
        **Warning**: When fallback occurs, noise is IGNORED and only drift term
        is integrated. This produces incorrect results for stochastic systems.
        
        Notes
        -----
        - Called by `step()` when mode is FIXED_STEP
        - Each call creates new integrator instance (may be inefficient for
        long simulations with many steps)
        - Control u is held constant over the integration interval
        - For stochastic systems, each call generates different noise realization
        
        Examples
        --------
        >>> # Typical usage through step() method
        >>> discrete = DiscretizedSystem(system, dt=0.01, method='rk4',
        ...                              mode=DiscretizationMode.FIXED_STEP)
        >>> x_next = discrete.step(x0, u0, k=0)
        >>> # Internally calls: _step_fixed(x0, u0, 0.0, 0.01)
        
        See Also
        --------
        _step_dense : Step using adaptive integration with dense output
        step : Public interface for single time step
        """
        # Check if we should use SDE integrator
        use_sde = (
            self._is_stochastic 
            and self._method in self._SDE_METHODS 
            and self._has_sde_integrator
        )
        
        if use_sde:
            # Use SDE integrator for stochastic systems
            try:
                from src.systems.base.numerical_integration.stochastic.sde_integrator_factory import SDEIntegratorFactory
                
                integrator = SDEIntegratorFactory.create(
                    sde_system=self._continuous_system,
                    backend=self._continuous_system._default_backend,
                    method=self._method,
                    dt=self._dt,
                    **self._integrator_kwargs
                )
            except (ImportError, AttributeError, ValueError) as e:
                # SDE integrator creation failed - fall back to deterministic
                import warnings
                warnings.warn(
                    f"Failed to create SDE integrator: {e}. "
                    f"Falling back to deterministic integration (noise ignored).",
                    UserWarning,
                    stacklevel=2
                )
                integrator = IntegratorFactory.create(
                    system=self._continuous_system,
                    backend=self._continuous_system._default_backend,
                    method='rk4',  # Safe deterministic fallback
                    dt=self._dt,
                    **self._integrator_kwargs
                )
        else:
            # Use regular integrator for deterministic systems or deterministic methods
            integrator = IntegratorFactory.create(
                system=self._continuous_system,
                backend=self._continuous_system._default_backend,
                method=self._method,
                dt=self._dt,
                **self._integrator_kwargs
            )
        
        result = integrator.integrate(x0=x, u_func=lambda t, xv: u, t_span=(t_start, t_end))
        return result['x'][-1, :] if 'x' in result else result['y'][:, -1]


    def _step_dense(self, x, u, t_start, t_end):
        """
        Single step using dense output (adaptive methods).
        
        Integrates from t_start to t_end using adaptive step size methods and
        evaluates solution at t_end using dense output (continuous solution
        representation). For stochastic systems, falls back to regular integration
        since most SDE methods don't support dense output.
        
        Parameters
        ----------
        x : StateVector
            Current state (nx,)
        u : ControlVector or None
            Control input (constant over [t_start, t_end])
        t_start : float
            Start time
        t_end : float
            End time (typically t_start + dt)
        
        Returns
        -------
        StateVector
            State at t_end (nx,)
        
        Integration Selection Logic
        ---------------------------
        1. **SDE integration** (if stochastic system with SDE method):
        - Attempts SDE integration WITHOUT dense output
        - Most SDE methods don't support dense output
        - Uses final point from integration result
        - Falls back to deterministic on failure
        
        2. **Deterministic integration with dense output** (otherwise):
        - Uses adaptive method (RK45, LSODA, etc.)
        - Computes continuous solution representation
        - Evaluates solution exactly at t_end
        - More accurate for adaptive methods than using final integration point
        
        Dense Output Behavior
        ---------------------
        **Deterministic systems:**
        - Adaptive integrator computes solution representation (e.g., Hermite
        interpolant for RK45)
        - Solution can be evaluated at any t in [t_start, t_end]
        - Result at t_end is typically more accurate than last integration point
        - Fallback: If dense output unavailable, uses final integration point
        
        **Stochastic systems:**
        - Dense output NOT SUPPORTED for SDE methods
        - Uses final integration point only
        - Each call produces different result due to random noise
        - Interpolation between SDE sample paths is not well-defined
        
        Fallback Behavior (SDE Path)
        ----------------------------
        If SDE integration is attempted but fails, falls back to deterministic
        integration with dense output. Fallback triggers:
        
        - **ImportError**: SDEIntegratorFactory not available
        - Missing: diffeqpy (NumPy), torchsde (PyTorch), diffrax (JAX)
        
        - **AttributeError**: System interface incomplete
        - Missing `diffusion()` method
        - Missing `_default_backend` attribute
        - Backend missing expected methods
        
        - **ValueError**: Invalid SDE integrator configuration
        - Method not supported by backend
        - Invalid parameters (dt, method, dimensions)
        
        **Warning**: Fallback ignores noise - produces incorrect results for
        stochastic systems.
        
        Fallback Behavior (Dense Output Path)
        -------------------------------------
        If dense output is unavailable (old scipy versions, custom integrators),
        uses final integration point instead:
        
        - Checks if result['sol'] exists and is not None
        - If available: evaluates sol(t_end)
        - If unavailable: uses result['x'][-1, :] or result['y'][:, -1]
        
        Notes
        -----
        - Called by `step()` when mode is DENSE_OUTPUT
        - Preferred for adaptive methods (more accurate than FIXED_STEP mode)
        - Less efficient than FIXED_STEP for fixed-step methods
        - For long simulations, BATCH_INTERPOLATION mode is more efficient
        
        Performance Considerations
        --------------------------
        - Creates new integrator instance per call (overhead for short intervals)
        - Dense output evaluation is cheap compared to integration cost
        - For N steps, BATCH mode does 1 integration vs N integrations here
        
        Examples
        --------
        >>> # Typical usage with adaptive method
        >>> discrete = DiscretizedSystem(system, dt=0.01, method='RK45',
        ...                              mode=DiscretizationMode.DENSE_OUTPUT)
        >>> x_next = discrete.step(x0, u0, k=0)
        >>> # Internally: integrates [0.0, 0.01] adaptively, evaluates sol(0.01)
        
        >>> # With stochastic system (no dense output)
        >>> discrete = DiscretizedSystem(stochastic_system, dt=0.01,
        ...                              sde_method='euler_maruyama',
        ...                              mode=DiscretizationMode.DENSE_OUTPUT)
        >>> x_next = discrete.step(x0, u0, k=0)
        >>> # Internally: SDE integration without dense output, uses final point
        
        See Also
        --------
        _step_fixed : Step using fixed-step integration
        _simulate_batch : Batch integration with interpolation (more efficient)
        step : Public interface for single time step
        """
        # Check if we should use SDE integrator
        use_sde = (
            self._is_stochastic 
            and self._method in self._SDE_METHODS 
            and self._has_sde_integrator
        )
        
        if use_sde:
            # Use SDE integrator (dense output not typically supported for SDEs)
            try:
                from src.systems.base.numerical_integration.stochastic.sde_integrator_factory import SDEIntegratorFactory
                
                integrator = SDEIntegratorFactory.create(
                    sde_system=self._continuous_system,
                    backend=self._continuous_system._default_backend,
                    method=self._method,
                    dt=self._dt,
                    **self._integrator_kwargs
                )
                
                result = integrator.integrate(
                    x0=x, 
                    u_func=lambda t, xv: u, 
                    t_span=(t_start, t_end),
                    dense_output=False  # SDEs typically don't support dense output
                )
                
                return result['x'][-1, :] if 'x' in result else result['y'][:, -1]
                
            except (ImportError, AttributeError, ValueError) as e:
                # Fall back to deterministic
                import warnings
                warnings.warn(
                    f"SDE integrator failed: {e}. Using deterministic integration.",
                    UserWarning,
                    stacklevel=2
                )
                # Fall through to deterministic path below
        
        # Use regular integrator with dense output
        integrator = IntegratorFactory.create(
            system=self._continuous_system,
            backend=self._continuous_system._default_backend,
            method=self._method,
            **self._integrator_kwargs
        )
        
        result = integrator.integrate(
            x0=x, 
            u_func=lambda t, xv: u, 
            t_span=(t_start, t_end), 
            dense_output=True
        )
        
        if 'sol' in result and result['sol'] is not None:
            x_end = result['sol'](t_end)
            return x_end.ravel() if x_end.ndim > 1 else x_end
        
        return result['x'][-1, :] if 'x' in result else result['y'][:, -1]
    
    def simulate(self, x0: StateVector, u_sequence: DiscreteControlInput = None, 
                 n_steps: int = 100, **kwargs) -> DiscreteSimulationResult:
        return self._simulate_batch(x0, u_sequence, n_steps) if self._mode == DiscretizationMode.BATCH_INTERPOLATION else self._simulate_step_by_step(x0, u_sequence, n_steps)
    
    def simulate_stochastic(
        self,
        x0: StateVector,
        u_sequence: DiscreteControlInput = None,
        n_steps: int = 100,
        n_trajectories: int = 100,
        **kwargs
    ) -> dict:
        """
        Simulate stochastic system with multiple Monte Carlo trajectories.
        
        Generates multiple independent realizations of the stochastic system,
        each with different random noise. Useful for uncertainty quantification,
        statistical analysis, and estimating expectations over noise distributions.
        
        Only available for stochastic systems with SDE integration methods.
        For deterministic systems, use the regular `simulate()` method.
        
        Parameters
        ----------
        x0 : StateVector
            Initial state (nx,). Same initial condition used for all trajectories.
        u_sequence : DiscreteControlInput, optional
            Control sequence applied to all trajectories. Can be:
            - None: Zero control (or no control if nu=0)
            - ndarray: Shape (n_steps, nu) - open-loop control sequence
            - callable(k): Time-dependent control u(k) - same for all trajectories
            - callable(x, k): **NOT SUPPORTED** - state feedback requires different
              control for each trajectory, which is not implemented
            
            **Important**: The same control sequence is applied to ALL trajectories.
            Each trajectory differs only in the random noise realization.
        n_steps : int, default=100
            Number of time steps to simulate
        n_trajectories : int, default=100
            Number of independent Monte Carlo realizations to generate.
            
            **Memory warning**: Total memory scales as:
            `n_trajectories × (n_steps+1) × nx × 8 bytes`
            
            Examples:
            - 100 trajectories × 1000 steps × 10 states ≈ 8 MB
            - 1000 trajectories × 10000 steps × 50 states ≈ 4 GB
            
            For large simulations, consider:
            - Processing trajectories in batches
            - Using lower precision (float32)
            - Computing statistics online without storing all trajectories
        **kwargs
            Additional arguments passed to integrator:
            - seed : int - Random seed for reproducibility. If not provided,
              each call produces different results. If provided, all n_trajectories
              use sequential seeds: seed, seed+1, seed+2, ...
            - Other integrator-specific kwargs (rtol, atol, etc.)
        
        Returns
        -------
        dict
            Dictionary containing Monte Carlo simulation results:
            
            - **states** : ndarray, shape (n_trajectories, n_steps+1, nx)
                All individual trajectories. states[i, k, :] is the state of
                trajectory i at time step k.
            
            - **controls** : ndarray, shape (n_steps, nu) or None
                Control sequence applied (same for all trajectories). None if
                nu=0 or u_sequence=None.
            
            - **mean_trajectory** : ndarray, shape (n_steps+1, nx)
                Sample mean over all trajectories at each time step.
                mean_trajectory[k, :] = mean(states[:, k, :], axis=0)
            
            - **std_trajectory** : ndarray, shape (n_steps+1, nx)
                Sample standard deviation over all trajectories at each time step.
                std_trajectory[k, :] = std(states[:, k, :], axis=0)
            
            - **time_steps** : ndarray, shape (n_steps+1,)
                Time step indices [0, 1, 2, ..., n_steps]
            
            - **dt** : float
                Sampling time step (seconds)
            
            - **success** : bool
                Always True (included for consistency with simulate())
            
            - **n_trajectories** : int
                Number of trajectories generated
            
            - **metadata** : dict
                Additional information:
                - 'method': Integration method used
                - 'mode': Discretization mode
                - 'is_stochastic': True
                - 'convergence_type': 'strong' (assumed for most SDE methods)
        
        Raises
        ------
        ValueError
            If system is not stochastic, or if SDE method not configured,
            or if SDE integrator not available
        NotImplementedError
            If u_sequence is state-feedback (callable with 2 arguments).
            State-feedback control requires per-trajectory control computation,
            which is not currently implemented.
        
        Notes
        -----
        **Random Seed Behavior:**
        - Without seed: Each call generates different random trajectories
        - With seed: Reproducible results across calls
        - Seed assignment: Trajectory i uses seed + i (if seed provided)
        
        **Control Sequence Behavior:**
        - Open-loop control: Same u(k) applied to all trajectories
        - State feedback: NOT SUPPORTED (would require per-trajectory control)
        - No control (None): Zero control vector applied
        
        **Convergence Properties:**
        - Sample mean converges to E[X(t)] as n_trajectories → ∞
        - Convergence rate typically O(1/√n_trajectories) (Central Limit Theorem)
        - For variance estimation, n_trajectories > 100 recommended
        - For tail probabilities, n_trajectories > 1000 recommended
        
        **Memory Management:**
        All trajectories are stored in memory simultaneously. For large-scale
        simulations, consider implementing custom batched processing or
        computing statistics online.
        
        **Statistical Estimators:**
        The returned statistics (mean_trajectory, std_trajectory) are sample
        statistics, not population parameters. They have estimation error
        that decreases with n_trajectories.
        
        Examples
        --------
        **Basic Monte Carlo Simulation:**
        
        >>> # Setup stochastic system
        >>> discrete = DiscretizedSystem(
        ...     stochastic_system, 
        ...     dt=0.01, 
        ...     sde_method='euler_maruyama'
        ... )
        >>> 
        >>> # Run 100 trajectories
        >>> result = discrete.simulate_stochastic(
        ...     x0=np.array([1.0, 0.0]),
        ...     u_sequence=None,
        ...     n_steps=1000,
        ...     n_trajectories=100
        ... )
        >>> 
        >>> print(f"Mean final state: {result['mean_trajectory'][-1]}")
        >>> print(f"Std final state: {result['std_trajectory'][-1]}")
        
        **Reproducible Simulation with Seed:**
        
        >>> # Same seed produces identical results
        >>> result1 = discrete.simulate_stochastic(
        ...     x0=np.array([1.0, 0.0]),
        ...     n_steps=1000,
        ...     n_trajectories=100,
        ...     seed=42
        ... )
        >>> 
        >>> result2 = discrete.simulate_stochastic(
        ...     x0=np.array([1.0, 0.0]),
        ...     n_steps=1000,
        ...     n_trajectories=100,
        ...     seed=42
        ... )
        >>> 
        >>> np.allclose(result1['states'], result2['states'])
        True
        
        **With Open-Loop Control:**
        
        >>> # Constant control applied to all trajectories
        >>> u_const = np.array([1.0])
        >>> result = discrete.simulate_stochastic(
        ...     x0=np.array([0.0, 0.0]),
        ...     u_sequence=u_const,  # Same control for all trajectories
        ...     n_steps=500,
        ...     n_trajectories=200
        ... )
        >>> 
        >>> # Time-varying control (same for all trajectories)
        >>> def u_func(k):
        ...     return np.array([np.sin(k * 0.1)])
        >>> 
        >>> result = discrete.simulate_stochastic(
        ...     x0=np.array([0.0, 0.0]),
        ...     u_sequence=u_func,
        ...     n_steps=500,
        ...     n_trajectories=200
        ... )
        
        **Plotting Confidence Intervals:**
        
        >>> import matplotlib.pyplot as plt
        >>> 
        >>> # Extract results
        >>> t = result['time_steps'] * result['dt']
        >>> mean = result['mean_trajectory'][:, 0]  # First state
        >>> std = result['std_trajectory'][:, 0]
        >>> 
        >>> # Plot mean ± 2σ (≈95% confidence)
        >>> plt.figure(figsize=(10, 6))
        >>> plt.plot(t, mean, 'b-', linewidth=2, label='Mean')
        >>> plt.fill_between(t, mean-2*std, mean+2*std, 
        ...                  alpha=0.3, label='95% CI')
        >>> plt.xlabel('Time (s)')
        >>> plt.ylabel('State')
        >>> plt.legend()
        >>> plt.grid(True)
        >>> plt.show()
        
        **Analyzing Individual Trajectories:**
        
        >>> # Access individual trajectories
        >>> traj_5 = result['states'][5, :, :]  # 6th trajectory
        >>> 
        >>> # Plot first 10 trajectories
        >>> plt.figure()
        >>> for i in range(10):
        ...     plt.plot(t, result['states'][i, :, 0], alpha=0.5)
        >>> plt.plot(t, result['mean_trajectory'][:, 0], 'k-', 
        ...          linewidth=2, label='Mean')
        >>> plt.legend()
        
        **Computing Additional Statistics:**
        
        >>> # Quantiles (median, 5th, 95th percentiles)
        >>> median = np.percentile(result['states'], 50, axis=0)
        >>> lower = np.percentile(result['states'], 5, axis=0)
        >>> upper = np.percentile(result['states'], 95, axis=0)
        >>> 
        >>> # Probability of exceeding threshold
        >>> threshold = 2.0
        >>> prob_exceed = np.mean(result['states'][:, -1, 0] > threshold)
        >>> print(f"P(x[0] > {threshold}) ≈ {prob_exceed:.2%}")
        
        **Memory-Conscious Batch Processing:**
        
        >>> # For very large simulations, process in batches
        >>> n_total = 10000
        >>> batch_size = 100
        >>> n_batches = n_total // batch_size
        >>> 
        >>> # Accumulate statistics online
        >>> sum_states = None
        >>> sum_sq_states = None
        >>> 
        >>> for batch in range(n_batches):
        ...     result = discrete.simulate_stochastic(
        ...         x0=x0, n_steps=n_steps, 
        ...         n_trajectories=batch_size,
        ...         seed=batch*batch_size  # Different seeds per batch
        ...     )
        ...     
        ...     if sum_states is None:
        ...         sum_states = result['states'].sum(axis=0)
        ...         sum_sq_states = (result['states']**2).sum(axis=0)
        ...     else:
        ...         sum_states += result['states'].sum(axis=0)
        ...         sum_sq_states += (result['states']**2).sum(axis=0)
        >>> 
        >>> # Compute overall statistics
        >>> mean = sum_states / n_total
        >>> variance = (sum_sq_states / n_total) - mean**2
        >>> std = np.sqrt(variance)
        
        See Also
        --------
        simulate : Single trajectory simulation (deterministic or stochastic)
        _detect_sde_method : Auto-detect appropriate SDE integration method
        """
        if not self._is_stochastic:
            raise ValueError(
                "simulate_stochastic() only available for stochastic systems. "
                "Use regular simulate() for deterministic systems."
            )
        
        if self._method not in self._SDE_METHODS:
            raise ValueError(
                f"simulate_stochastic() requires SDE method. "
                f"Current method '{self._method}' is deterministic. "
                f"Use sde_method parameter: DiscretizedSystem(system, sde_method='euler_maruyama')"
            )
        
        if not self._has_sde_integrator:
            raise ValueError(
                "SDE integrator not available. Install required package:\n"
                "  - NumPy backend: pip install diffeqpy\n"
                "  - PyTorch backend: pip install torchsde\n"
                "  - JAX backend: pip install diffrax"
            )
        
        # Run multiple trajectories
        all_trajectories = []
        controls_ref = None
        
        for traj_idx in range(n_trajectories):
            # Each trajectory uses different random noise
            result = self.simulate(x0, u_sequence, n_steps, **kwargs)
            all_trajectories.append(result['states'])
            
            if controls_ref is None:
                controls_ref = result['controls']
        
        # Stack trajectories: (n_trajectories, n_steps+1, nx)
        all_trajectories = np.array(all_trajectories)
        
        # Compute statistics
        mean_traj = np.mean(all_trajectories, axis=0)
        std_traj = np.std(all_trajectories, axis=0)
        
        return {
            'states': all_trajectories,
            'controls': controls_ref,
            'mean_trajectory': mean_traj,
            'std_trajectory': std_traj,
            'time_steps': result['time_steps'],
            'dt': self.dt,
            'success': True,
            'n_trajectories': n_trajectories,
            'metadata': {
                'method': self._method,
                'mode': self._mode.value,
                'is_stochastic': True,
                'convergence_type': 'strong'  # Assumed for most SDE methods
            }
        }
    
    def _simulate_step_by_step(self, x0, u_sequence, n_steps):
        states = np.zeros((n_steps + 1, self.nx))
        states[0, :] = x0
        controls = []
        u_func = self._prepare_control_sequence(u_sequence, n_steps)
        
        x = x0
        for k in range(n_steps):
            u = u_func(x, k)
            controls.append(u)
            x = self.step(x, u, k)
            states[k + 1, :] = x
        
        return {
            "states": states,
            "controls": np.array(controls) if controls and controls[0] is not None else None,
            "time_steps": np.arange(n_steps + 1),
            "dt": self.dt,
            "success": True,
            "metadata": {"method": self._method, "mode": self._mode.value}
        }
    
    def _simulate_batch(self, x0, u_sequence, n_steps):
        if callable(u_sequence) and len(inspect.signature(u_sequence).parameters) == 2:
            raise ValueError("State-feedback not supported in BATCH_INTERPOLATION mode")
        
        u_func_discrete = self._prepare_control_sequence(u_sequence, n_steps)
        u_func_continuous = lambda t, x: u_func_discrete(x, min(int(t / self.dt), n_steps - 1))
        
        result = self._continuous_system.integrate(
            x0=x0, u=u_func_continuous, t_span=(0.0, n_steps * self.dt),
            method=self._method, **self._integrator_kwargs
        )
        
        trajectory = result['x'] if 'x' in result else result['y'].T
        t_regular = np.arange(0, n_steps + 1) * self.dt
        states_regular = self._interpolate_trajectory(result['t'], trajectory, t_regular)
        
        return {
            "states": states_regular,
            "controls": None,
            "time_steps": np.arange(n_steps + 1),
            "dt": self.dt,
            "success": result.get('success', True),
            "metadata": {
                "method": self._method, "mode": self._mode.value,
                "nfev": result.get('nfev'), "adaptive_points": len(result['t'])
            }
        }
    
    def _interpolate_trajectory(self, t_adaptive, y_adaptive, t_regular):
        """
        Interpolate trajectory from adaptive time grid to regular time grid.
        
        Used in BATCH_INTERPOLATION mode to convert adaptive integrator output
        (irregular time points) to the requested regular time grid (uniform dt).
        
        Parameters
        ----------
        t_adaptive : ndarray, shape (n_adaptive,)
            Time points from adaptive integration (irregular spacing).
            Expected to span at least the range of t_regular.
        y_adaptive : ndarray, shape (n_adaptive, nx) or (nx, n_adaptive)
            State trajectory at adaptive time points.
            Automatically transposed if shape is (nx, n_adaptive).
        t_regular : ndarray, shape (n_regular,)
            Desired regular time grid (uniform spacing dt).
            Typically: [0, dt, 2*dt, ..., n_steps*dt]
        
        Returns
        -------
        y_regular : ndarray, shape (n_regular, nx)
            Interpolated state trajectory at regular time points
        
        Raises
        ------
        ValueError
            If t_regular extends significantly beyond t_adaptive range,
            indicating integration failure or early termination.
        
        Interpolation Method
        --------------------
        - Uses scipy.interpolate.interp1d
        - Kind determined by self._interpolation_kind ('linear' or 'cubic')
        - Automatic fallback: cubic → linear if n_adaptive < 4 points
        - Each state dimension interpolated independently
        
        Time Grid Validation
        --------------------
        The method validates that t_adaptive properly spans t_regular:
        
        1. **Floating-point tolerance** (tol = 1e-10):
        - Small mismatches due to floating-point arithmetic are allowed
        - Typical: t_adaptive[-1] = 9.999999999999998 vs t_regular[-1] = 10.0
        
        2. **Significant mismatch** (> 1% of dt):
        - Raises ValueError with diagnostic information
        - Indicates adaptive integrator stopped early due to:
            * max_steps limit reached
            * Numerical issues (stiffness, instability)
            * Integration failure
        
        3. **Minor mismatch** (> tol but < 1% of dt):
        - Issues warning but continues
        - Clips t_regular to t_adaptive range
        - Uses endpoint extrapolation
        
        **Why validation is important:**
        Previous implementation silently extrapolated when t_adaptive was shorter
        than expected, masking serious integration failures. This could produce
        incorrect results without any indication that something went wrong.
        
        Notes
        -----
        - Only used in BATCH_INTERPOLATION mode
        - Assumes t_adaptive is sorted (enforced with assume_sorted=True)
        - Transposition handling: Accepts either (n, nx) or (nx, n) for y_adaptive
        - Independent interpolation per state may introduce small artifacts at
        state constraints
        
        Performance
        -----------
        - O(nx * n_regular * log(n_adaptive)) for sorted interpolation
        - Cubic interpolation ~2-3x slower than linear
        - Negligible compared to integration cost for typical simulations
        - Validation overhead is O(1)
        
        Accuracy
        --------
        - Linear: O(h²) error where h = max spacing in t_adaptive
        - Cubic: O(h⁴) error
        - For adaptive integrators with dense output, evaluating dense output
        directly (DENSE_OUTPUT mode) is more accurate than interpolation
        
        Examples
        --------
        >>> # Normal case - adaptive grid spans regular grid
        >>> t_adaptive = np.array([0.0, 0.003, 0.011, 0.025, ..., 10.0])
        >>> y_adaptive = np.random.randn(len(t_adaptive), 4)  # 4 states
        >>> t_regular = np.arange(0, 10.01, 0.01)  # 1001 points
        >>> y_regular = discrete._interpolate_trajectory(
        ...     t_adaptive, y_adaptive, t_regular
        ... )
        >>> y_regular.shape
        (1001, 4)
        
        >>> # Floating-point mismatch (harmless)
        >>> t_adaptive = np.array([0.0, 0.5, 0.999999999999998])
        >>> t_regular = np.array([0.0, 0.5, 1.0])
        >>> y_regular = discrete._interpolate_trajectory(...)  # Works fine
        
        >>> # Minor mismatch (warning issued)
        >>> t_adaptive = np.array([0.0, 0.5, 0.99])  # 1% short
        >>> t_regular = np.array([0.0, 0.5, 1.0])
        >>> y_regular = discrete._interpolate_trajectory(...)
        # UserWarning: Time grid mismatch detected...
        
        >>> # Significant mismatch (error raised)
        >>> t_adaptive = np.array([0.0, 0.5, 0.8])  # 20% short
        >>> t_regular = np.array([0.0, 0.5, 1.0])
        >>> y_regular = discrete._interpolate_trajectory(...)
        # ValueError: Adaptive integration appears to have failed...
        
        >>> # Cubic interpolation requires ≥4 points
        >>> t_adaptive_short = np.array([0.0, 0.5, 1.0])  # Only 3 points
        >>> # Automatically falls back to linear (no warning needed)
        
        See Also
        --------
        _simulate_batch : Main caller of this method
        scipy.interpolate.interp1d : Underlying interpolation function
        """
        # Transpose if necessary
        if y_adaptive.shape[0] != len(t_adaptive):
            y_adaptive = y_adaptive.T
        
        nx = y_adaptive.shape[1]
        y_regular = np.zeros((len(t_regular), nx))
        
        # ========================================================================
        # Validate time grid compatibility
        # ========================================================================
        
        # Define tolerance for floating-point comparisons
        FP_TOL = 1e-10  # Floating-point epsilon tolerance
        MISMATCH_TOL = 0.01 * self._dt  # 1% of time step
        
        # Check start time
        t_start_diff = t_regular[0] - t_adaptive[0]
        if t_start_diff < -FP_TOL:  # t_regular starts before t_adaptive
            raise ValueError(
                f"Regular time grid starts before adaptive grid: "
                f"t_regular[0]={t_regular[0]:.10f}, t_adaptive[0]={t_adaptive[0]:.10f}. "
                f"Difference: {t_start_diff:.2e}"
            )
        
        # Check end time
        t_end_diff = t_regular[-1] - t_adaptive[-1]
        
        if t_end_diff > MISMATCH_TOL:
            # Significant mismatch - integration likely failed
            raise ValueError(
                f"Adaptive integration appears to have failed or stopped early. "
                f"Requested end time: t={t_regular[-1]:.6f}, "
                f"but integration stopped at t={t_adaptive[-1]:.6f}. "
                f"Shortfall: {t_end_diff:.6f} ({t_end_diff/self._dt:.1f} time steps). "
                f"This may indicate:\n"
                f"  - max_steps limit reached\n"
                f"  - Numerical instability in integration\n"
                f"  - Stiff system requiring different method\n"
                f"Check result['success'] and consider using a more robust integrator."
            )
        
        elif t_end_diff > FP_TOL:
            # Minor mismatch - warn but continue
            import warnings
            warnings.warn(
                f"Time grid mismatch detected. "
                f"Requested t_end={t_regular[-1]:.10f}, "
                f"but adaptive integration ended at t={t_adaptive[-1]:.10f}. "
                f"Difference: {t_end_diff:.2e}. "
                f"Clipping to adaptive range and extrapolating with endpoint values.",
                UserWarning,
                stacklevel=3  # Report from _simulate_batch
            )
        
        # If start time slightly before adaptive grid (within tolerance), clip it
        if t_start_diff > FP_TOL:
            import warnings
            warnings.warn(
                f"Regular grid starts slightly after adaptive grid "
                f"(Δt={t_start_diff:.2e}). Clipping to adaptive range.",
                UserWarning,
                stacklevel=3
            )
        
        # ========================================================================
        # Clip to safe range (after validation)
        # ========================================================================
        
        # Clip t_regular to be within t_adaptive range
        # After validation, this should only clip floating-point noise
        t_regular_clipped = np.clip(t_regular, t_adaptive[0], t_adaptive[-1])
        
        # ========================================================================
        # Choose interpolation kind
        # ========================================================================
        
        kind = self._interpolation_kind
        if len(t_adaptive) < 4 and kind == 'cubic':
            # Fallback to linear if not enough points for cubic
            # (cubic spline requires at least 4 points)
            kind = 'linear'
        
        # ========================================================================
        # Perform interpolation
        # ========================================================================
        
        for i in range(nx):
            interp = interp1d(
                t_adaptive, 
                y_adaptive[:, i], 
                kind=kind,
                bounds_error=False,  # Don't raise (we've already validated)
                fill_value=(y_adaptive[0, i], y_adaptive[-1, i]),  # Endpoint extrapolation
                assume_sorted=True  # Performance optimization
            )
            y_regular[:, i] = interp(t_regular_clipped)
        
        return y_regular
    
    def linearize(self, x_eq: StateVector, u_eq: Optional[ControlVector] = None) -> DiscreteLinearization:
        lin_result = self._continuous_system.linearize(x_eq, u_eq)
        A, B = lin_result[:2]  # Handle both (A,B) and (A,B,G)
        
        from scipy.linalg import expm
        nx, I = A.shape[0], np.eye(A.shape[0])
        Ad = expm(A * self.dt)
        
        try:
            if np.linalg.cond(A) > 1e10:
                Bd = self.dt * B
            else:
                Bd = np.linalg.inv(A) @ (Ad - I) @ B
        except np.linalg.LinAlgError:
            Bd = self.dt * B
        
        return (Ad, Bd)
    
    def _prepare_control_sequence(self, u_sequence, n_steps):
        if u_sequence is None:
            return lambda x, k: None if self.nu == 0 else np.zeros(self.nu)
        
        if callable(u_sequence):
            sig = inspect.signature(u_sequence)
            if len(sig.parameters) == 1:
                return lambda x, k: u_sequence(k)
            elif len(sig.parameters) == 2:
                names = list(sig.parameters.keys())
                if names[0] in ['x', 'state']:
                    return u_sequence
                elif names[0] in ['k', 'time']:
                    return lambda x, k: u_sequence(k, x)
                try:
                    u_sequence(np.zeros(self.nx), 0)
                    return u_sequence
                except:
                    return lambda x, k: u_sequence(k, x)
            else:
                raise TypeError(f"Control function must accept 1 or 2 parameters, got {len(sig.parameters)}")
        
        if isinstance(u_sequence, np.ndarray):
            if u_sequence.ndim == 1:
                # Constant control - validate dimension
                if u_sequence.size != self.nu:
                    raise ValueError(f"Control dimension mismatch: expected {self.nu}, got {u_sequence.size}")
                return lambda x, k: u_sequence
            if u_sequence.shape[0] == n_steps:
                return lambda x, k: u_sequence[k, :] if k < n_steps else u_sequence[-1, :]
            return lambda x, k: u_sequence[:, k] if k < u_sequence.shape[1] else u_sequence[:, -1]
        
        if isinstance(u_sequence, (list, tuple)):
            return lambda x, k: np.asarray(u_sequence[k] if k < len(u_sequence) else u_sequence[-1])
        
        raise TypeError(f"Invalid control type: {type(u_sequence)}")
    
    def compare_modes(self, x0, u_sequence, n_steps, reference_solution=None):
        if reference_solution is None:
            ref = DiscretizedSystem(self._continuous_system, dt=self.dt, method='LSODA',
                                   mode=DiscretizationMode.BATCH_INTERPOLATION, rtol=1e-12, atol=1e-14)
            reference_solution = ref.simulate(x0, u_sequence, n_steps)['states']
        
        results, timings, errors = {}, {}, {}
        
        for name, mode, method in [('fixed_step', DiscretizationMode.FIXED_STEP, 'rk4'),
                                   ('dense_output', DiscretizationMode.DENSE_OUTPUT, 'RK45'),
                                   ('batch', DiscretizationMode.BATCH_INTERPOLATION, 'RK45')]:
            sys = DiscretizedSystem(self._continuous_system, dt=self.dt, method=method, mode=mode)
            start = time.time()
            result = sys.simulate(x0, u_sequence, n_steps)
            timings[name] = time.time() - start
            results[name] = result
            errors[name] = np.sqrt(np.mean((result['states'] - reference_solution) ** 2))
        
        return {
            'results': results, 'timings': timings, 'errors': errors,
            'reference': reference_solution,
            'speedup_batch_vs_fixed': timings['fixed_step'] / timings['batch'],
            'speedup_batch_vs_dense': timings['dense_output'] / timings['batch']
        }
    
    def change_method(self, new_method: str, **new_kwargs) -> "DiscretizedSystem":
        """Create new DiscretizedSystem with different method."""
        merged_kwargs = {**self._integrator_kwargs, **new_kwargs}
        return DiscretizedSystem(
            self._continuous_system, dt=self.dt, method=new_method,
            mode=None, interpolation_kind=self._interpolation_kind, **merged_kwargs
        )
    
    def get_info(self) -> dict:
        """
        Get comprehensive discretization information.
        
        Returns
        -------
        dict
            Dictionary containing:
            - Basic info: class, mode, method, dt, dimensions
            - System type: is_stochastic, is_symbolic, continuous_system_type
            - Capabilities: supports_step, supports_closed_loop, interpolation
            - Stochastic info: noise type, SDE method, availability (if stochastic)
            - Symbolic info: symbolic variables, Jacobian availability (if symbolic)
            - Integrator settings: kwargs passed to integrator
        
        Examples
        --------
        >>> discrete = DiscretizedSystem(stochastic_system, dt=0.01)
        >>> info = discrete.get_info()
        >>> print(info['stochastic_info']['recommended_method'])
        'euler_maruyama'
        >>> print(info['method_selection']['source'])
        'auto_detected'
        """
        info = {
            # ====================================================================
            # Basic Information
            # ====================================================================
            "class": "DiscretizedSystem",
            "mode": self._mode.value,
            "method": self._method,
            "dt": self.dt,
            "is_fixed_step": self._is_fixed_step,
            "interpolation": self._interpolation_kind,
            
            # ====================================================================
            # Capabilities
            # ====================================================================
            "supports_step": self._mode != DiscretizationMode.BATCH_INTERPOLATION,
            "supports_closed_loop": self._mode != DiscretizationMode.BATCH_INTERPOLATION,
            "supports_linearization": True,
            
            # ====================================================================
            # System Information
            # ====================================================================
            "continuous_system_type": type(self._continuous_system).__name__,
            "is_stochastic": self.is_stochastic,
            "is_symbolic": self._is_symbolic,
            
            # ====================================================================
            # Dimensions
            # ====================================================================
            "dimensions": {
                "nx": self.nx,
                "nu": self.nu,
                "ny": self.ny
            },
            
            # ====================================================================
            # Integrator Settings
            # ====================================================================
            "integrator_kwargs": self._integrator_kwargs,
        }
        
        # ========================================================================
        # Method Selection Information
        # ========================================================================
        
        info['method_selection'] = {
            'source': self._method_source,
            'original_method': self._original_method,
            'final_method': self._method,
            'description': self._get_method_selection_description()
        }
        
        # ========================================================================
        # Stochastic System Information
        # ========================================================================
        
        if self.is_stochastic:
            stochastic_info = {
                'is_stochastic': True,
                'has_sde_integrator': self._has_sde_integrator,
                'recommended_method': self._detect_sde_method(self._continuous_system),
                'noise_ignored': self._method in self._DETERMINISTIC_FIXED_STEP and not self._has_sde_integrator,
            }
            
            # Get noise structure if available
            try:
                if hasattr(self._continuous_system, 'is_additive_noise'):
                    stochastic_info['is_additive_noise'] = self._continuous_system.is_additive_noise()
                else:
                    stochastic_info['is_additive_noise'] = None
            except:
                stochastic_info['is_additive_noise'] = None
            
            try:
                if hasattr(self._continuous_system, 'is_diagonal_noise'):
                    stochastic_info['is_diagonal_noise'] = self._continuous_system.is_diagonal_noise()
                else:
                    stochastic_info['is_diagonal_noise'] = None
            except:
                stochastic_info['is_diagonal_noise'] = None
            
            # Get SDE type if available
            if hasattr(self._continuous_system, 'sde_type'):
                stochastic_info['sde_type'] = self._continuous_system.sde_type
            else:
                stochastic_info['sde_type'] = 'unknown'
            
            # Get diffusion dimension if available
            if hasattr(self._continuous_system, 'nw'):
                stochastic_info['noise_dimension'] = self._continuous_system.nw
            else:
                stochastic_info['noise_dimension'] = None
            
            info['stochastic_info'] = stochastic_info
        
        # ========================================================================
        # Symbolic System Information
        # ========================================================================
        
        if self._is_symbolic:
            symbolic_info = {
                'is_symbolic': True,
                'has_symbolic_jacobian': self._has_symbolic_jacobian,
                'can_generate_code': True,  # Assume true if symbolic
            }
            
            # Get symbolic variable names
            if self._symbolic_state_vars is not None:
                try:
                    symbolic_info['state_vars'] = [str(v) for v in self._symbolic_state_vars]
                except:
                    symbolic_info['state_vars'] = None
            else:
                symbolic_info['state_vars'] = None
            
            if self._symbolic_control_vars is not None:
                try:
                    symbolic_info['control_vars'] = [str(v) for v in self._symbolic_control_vars]
                except:
                    symbolic_info['control_vars'] = None
            else:
                symbolic_info['control_vars'] = None
            
            # Check for parameters
            if hasattr(self._continuous_system, 'parameters'):
                try:
                    params = self._continuous_system.parameters
                    symbolic_info['parameters'] = {str(k): v for k, v in params.items()}
                except:
                    symbolic_info['parameters'] = None
            else:
                symbolic_info['parameters'] = None
            
            info['symbolic_info'] = symbolic_info
        
        # ========================================================================
        # Warnings/Recommendations
        # ========================================================================
        
        warnings = []
        
        # Check for stochastic + deterministic method
        if self.is_stochastic and self._method in self._DETERMINISTIC_FIXED_STEP:
            if not self._has_sde_integrator:
                warnings.append(
                    "Stochastic system with deterministic integrator - noise is IGNORED. "
                    "Install SDE integration support for proper noise handling."
                )
            elif self._method_source == 'user_specified':
                warnings.append(
                    f"Using deterministic method '{self._method}' on stochastic system - "
                    f"consider using '{self._detect_sde_method(self._continuous_system)}' instead."
                )
        
        # Check for BATCH mode with stochastic
        if self._mode == DiscretizationMode.BATCH_INTERPOLATION and self.is_stochastic:
            warnings.append(
                "BATCH_INTERPOLATION mode with stochastic system - "
                "each simulation produces different trajectory."
            )
        
        # Check for cubic interpolation with few expected points
        if self._interpolation_kind == 'cubic' and self._mode == DiscretizationMode.BATCH_INTERPOLATION:
            warnings.append(
                "Using cubic interpolation - automatic fallback to linear if <4 adaptive points."
            )
        
        if warnings:
            info['warnings'] = warnings
        
        return info


    def _get_method_selection_description(self) -> str:
        """Get human-readable description of how method was selected."""
        source = self._method_source
        
        descriptions = {
            'explicit': f"Explicitly set via sde_method='{self._method}'",
            'auto_detected': f"Auto-detected for stochastic system (was '{self._original_method}')",
            'deterministic_fallback': f"Deterministic fallback (SDE integrator unavailable)",
            'user_specified': f"User-specified (auto-detection disabled)",
            'deterministic_system': f"Deterministic system"
        }
        
        return descriptions.get(source, f"Unknown ({source})")


    def print_info(self):
        """
        Print formatted discretization information.
        
        Displays comprehensive information about the discretized system including
        system type, method selection, capabilities, and any warnings.
        """
        info = self.get_info()
        
        print("=" * 70)
        print("DiscretizedSystem")
        print("=" * 70)
        
        # Basic info
        print(f"Continuous System: {info['continuous_system_type']}")
        print(f"Discretization Method: {info['method']}")
        print(f"Mode: {info['mode'].upper()}")
        print(f"Time Step: {info['dt']}s ({1/info['dt']:.1f} Hz)")
        
        # Dimensions
        dims = info['dimensions']
        print(f"Dimensions: nx={dims['nx']}, nu={dims['nu']}, ny={dims['ny']}")
        
        # System type
        print(f"Stochastic: {info['is_stochastic']}")
        print(f"Symbolic: {info['is_symbolic']}")
        
        # Capabilities
        print(f"Supports step(): {info['supports_step']}")
        print(f"Supports closed-loop: {info['supports_closed_loop']}")
        
        # Method selection
        print("\nMethod Selection:")
        print(f"  Source: {info['method_selection']['description']}")
        if info['method_selection']['original_method'] != info['method_selection']['final_method']:
            print(f"  Original: {info['method_selection']['original_method']}")
            print(f"  Final: {info['method_selection']['final_method']}")
        
        # Stochastic info
        if 'stochastic_info' in info:
            print("\nStochastic System Info:")
            sinfo = info['stochastic_info']
            print(f"  SDE Type: {sinfo.get('sde_type', 'unknown')}")
            print(f"  Recommended Method: {sinfo['recommended_method']}")
            print(f"  Has SDE Integrator: {sinfo['has_sde_integrator']}")
            print(f"  Noise Ignored: {sinfo['noise_ignored']}")
            
            if sinfo.get('is_additive_noise') is not None:
                print(f"  Additive Noise: {sinfo['is_additive_noise']}")
            if sinfo.get('is_diagonal_noise') is not None:
                print(f"  Diagonal Noise: {sinfo['is_diagonal_noise']}")
            if sinfo.get('noise_dimension') is not None:
                print(f"  Noise Dimension: {sinfo['noise_dimension']}")
        
        # Symbolic info
        if 'symbolic_info' in info:
            print("\nSymbolic System Info:")
            sinfo = info['symbolic_info']
            print(f"  Has Symbolic Jacobian: {sinfo['has_symbolic_jacobian']}")
            print(f"  Can Generate Code: {sinfo['can_generate_code']}")
            
            if sinfo.get('state_vars'):
                print(f"  State Variables: {', '.join(sinfo['state_vars'])}")
            if sinfo.get('control_vars'):
                print(f"  Control Variables: {', '.join(sinfo['control_vars'])}")
            if sinfo.get('parameters'):
                print(f"  Parameters: {len(sinfo['parameters'])} defined")
        
        # Integrator options
        if info['integrator_kwargs']:
            print("\nIntegrator Options:")
            for key, val in info['integrator_kwargs'].items():
                print(f"  {key}: {val}")
        
        # Warnings
        if 'warnings' in info:
            print("\n" + "⚠" * 35)
            print("WARNINGS:")
            for i, warning in enumerate(info['warnings'], 1):
                print(f"{i}. {warning}")
            print("⚠" * 35)
        
        print("=" * 70)
        
    def __repr__(self):
        return f"DiscretizedSystem(dt={self.dt:.4f}, method={self._method}, mode={self._mode.value})"


def discretize(continuous_system, dt, method='rk4', **kwargs):
    """
    Convenience wrapper for creating a discretized system.
    
    Creates a DiscretizedSystem with automatic mode selection based on the
    chosen integration method. This is the recommended way to discretize
    systems for step-by-step simulation and control applications.
    
    Parameters
    ----------
    continuous_system : ContinuousSystemBase
        Continuous system to discretize
    dt : float
        Sampling time step (seconds)
    method : str, default='rk4'
        Integration method. Can use canonical names ('euler_maruyama', 'rk45')
        or backend-specific names. Method name is automatically normalized
        for the system's backend.
        
        Common choices:
        - 'rk4': Classic 4th-order Runge-Kutta (default, good all-around)
        - 'euler': Simple 1st-order (fast but less accurate)
        - 'RK45': Adaptive 5th-order (for smooth, non-stiff systems)
        - 'LSODA': Adaptive with auto stiffness detection
        - 'euler_maruyama': For stochastic systems (canonical SDE name)
    **kwargs
        Additional arguments passed to DiscretizedSystem:
        - mode : DiscretizationMode - Override automatic mode selection
        - sde_method : str - Explicit SDE method for stochastic systems
        - auto_detect_sde : bool - Enable SDE method auto-detection
        - rtol, atol : float - Tolerances for adaptive methods
        - seed : int - Random seed for stochastic systems
        - Other integrator-specific options
    
    Returns
    -------
    DiscretizedSystem
        Discretized system ready for simulation
    
    Mode Selection
    --------------
    The discretization mode is automatically selected based on method:
    - Fixed-step methods (euler, rk4, euler_maruyama, etc.) → FIXED_STEP
    - Adaptive methods (RK45, LSODA, tsit5, etc.) → DENSE_OUTPUT
    - For BATCH_INTERPOLATION mode, use `discretize_batch()` instead
    
    When to Use
    -----------
    Use `discretize()` for:
    
    ✓ **Step-by-step simulation**: Need to inspect/modify state at each step
    ✓ **Control design**: LQR, MPC, policy evaluation with state feedback
    ✓ **Real-time applications**: Online computation with tight timing
    ✓ **Event-driven simulation**: Need to detect events between steps
    ✓ **Learning algorithms**: Reinforcement learning, system identification
    
    Use `discretize_batch()` for:
    
    ✓ **Open-loop trajectory generation**: No state feedback needed
    ✓ **Batch processing**: Many trajectories with different parameters
    ✓ **Visualization**: Generate smooth trajectories for plotting
    ✓ **Performance-critical**: Need maximum speed for open-loop simulation
    
    Examples
    --------
    **Basic discretization:**
    
    >>> from src.systems.examples import Pendulum
    >>> continuous = Pendulum(m=1.0, l=0.5)
    >>> 
    >>> # Default: RK4 with automatic mode selection
    >>> discrete = discretize(continuous, dt=0.01)
    >>> # Equivalent to: DiscretizedSystem(continuous, dt=0.01, method='rk4',
    >>> #                                   mode=DiscretizationMode.FIXED_STEP)
    >>> 
    >>> # Simulate with step-by-step control
    >>> x = np.array([0.1, 0.0])
    >>> for k in range(100):
    ...     u = controller(x, k)  # State feedback
    ...     x = discrete.step(x, u, k)
    
    **Using canonical names (recommended for portability):**
    
    >>> # High-accuracy adaptive method
    >>> discrete = discretize(continuous, dt=0.01, method='rk45', rtol=1e-9)
    >>> # 'rk45' normalized to 'RK45' (NumPy), 'dopri5' (PyTorch), or 'tsit5' (JAX)
    
    **Stochastic system with canonical SDE method:**
    
    >>> from src.systems.examples import StochasticPendulum
    >>> stochastic = StochasticPendulum(m=1.0, l=0.5, sigma=0.1)
    >>> 
    >>> # Use canonical SDE method name
    >>> discrete = discretize(stochastic, dt=0.01, sde_method='euler_maruyama')
    >>> # Automatically normalized to backend: 'EM' (NumPy), 'euler' (PyTorch), 'Euler' (JAX)
    >>> 
    >>> # Simulate single trajectory
    >>> result = discrete.simulate(x0=np.array([0.1, 0.0]), n_steps=1000)
    
    **Override automatic mode selection:**
    
    >>> # Force DENSE_OUTPUT mode with fixed-step method (unusual but allowed)
    >>> discrete = discretize(continuous, dt=0.01, method='rk4',
    ...                      mode=DiscretizationMode.DENSE_OUTPUT)
    
    **Closed-loop control with state feedback:**
    
    >>> discrete = discretize(continuous, dt=0.01, method='rk4')
    >>> 
    >>> def lqr_controller(x, k):
    ...     return -K @ x  # LQR feedback gain
    >>> 
    >>> # Simulate with controller
    >>> result = discrete.simulate(
    ...     x0=np.array([0.1, 0.0]),
    ...     u_sequence=lqr_controller,  # State feedback function
    ...     n_steps=500
    ... )
    
    **Real-time capable simulation:**
    
    >>> # Check if fast enough for 100 Hz control
    >>> discrete = discretize(continuous, dt=0.01, method='euler')
    >>> quality = compute_discretization_quality(discrete, x0, None, 1000)
    >>> 
    >>> if quality['timing']['steps_per_second'] > 10000:  # 100× real-time
    ...     print("✓ Suitable for 100 Hz real-time control")
    
    **Compare with batch mode:**
    
    >>> # Step-by-step (for control)
    >>> discrete_step = discretize(continuous, dt=0.01, method='rk4')
    >>> result1 = discrete_step.simulate(x0, u_sequence=None, n_steps=1000)
    >>> 
    >>> # Batch mode (for visualization)  
    >>> discrete_batch = discretize_batch(continuous, dt=0.01, method='LSODA')
    >>> result2 = discrete_batch.simulate(x0, u_sequence=None, n_steps=1000)
    >>> 
    >>> # Batch mode typically 5-10× faster for open-loop simulation
    
    See Also
    --------
    discretize_batch : Optimized for batch/open-loop simulation
    DiscretizedSystem : Full constructor with all options
    DiscretizationMode : Available discretization modes
    
    Notes
    -----
    - Method names automatically normalized to backend conventions
    - Original method name preserved in discrete._original_method
    - For stochastic systems, prefer explicit sde_method parameter
    - Mode auto-selection can be overridden with mode= parameter
    """
    return DiscretizedSystem(continuous_system, dt=dt, method=method, **kwargs)


def discretize_batch(continuous_system, dt, method='LSODA', **kwargs):
    """
    Create discretized system optimized for batch/open-loop simulation.
    
    Uses BATCH_INTERPOLATION mode, which performs a single adaptive integration
    over the entire time span and interpolates to the regular grid. This is
    typically 5-10× faster than step-by-step integration for open-loop
    trajectories but does not support state feedback control.
    
    Parameters
    ----------
    continuous_system : ContinuousSystemBase
        Continuous system to discretize
    dt : float
        Sampling time step (seconds) for output trajectory.
        Note: Adaptive integrator uses variable internal steps, dt only
        affects the output grid spacing.
    method : str, default='LSODA'
        Adaptive integration method. Recommended options:
        - 'LSODA': Auto stiffness detection (default, robust)
        - 'RK45': General purpose, non-stiff
        - 'DOP853': High accuracy (8th order)
        - 'Radau': Implicit, for stiff systems
        - 'BDF': Backward differentiation, very stiff systems
        
        Fixed-step methods (euler, rk4) also work but defeat the purpose
        of batch mode - prefer step-by-step discretize() for these.
    **kwargs
        Additional arguments passed to DiscretizedSystem:
        - interpolation_kind : str - 'linear' (default) or 'cubic'
        - rtol, atol : float - Integration tolerances (e.g., rtol=1e-9)
        - max_steps : int - Maximum integration steps
        - Other integrator-specific options
    
    Returns
    -------
    DiscretizedSystem
        Discretized system in BATCH_INTERPOLATION mode
    
    Mode Behavior
    -------------
    BATCH_INTERPOLATION mode:
    1. Calls adaptive integrator once for entire time span [0, T]
    2. Integrator returns irregular time points (dense where needed)
    3. Interpolates to regular grid with spacing dt
    4. Returns trajectory on regular grid
    
    **Advantages:**
    - Much faster for long simulations (1 integration vs N steps)
    - Adaptive error control throughout
    - Natural handling of stiff systems
    
    **Limitations:**
    - No state feedback control (open-loop only)
    - No step() method available
    - Cannot inspect/modify state during integration
    - Fixed control sequence only
    
    When to Use
    -----------
    Use `discretize_batch()` for:
    
    ✓ **Trajectory generation**: Creating reference trajectories
    ✓ **Visualization**: Smooth plots for papers/presentations  
    ✓ **Parameter sweeps**: Many simulations with different parameters
    ✓ **Monte Carlo studies**: Large number of open-loop trajectories
    ✓ **Performance-critical**: Need maximum speed for open-loop simulation
    ✓ **High accuracy**: Tight tolerances with adaptive methods
    
    Do NOT use for:
    
    ✗ **State feedback control**: Use discretize() instead
    ✗ **Event detection**: Need step-by-step access
    ✗ **Real-time simulation**: Need predictable timing per step
    ✗ **Learning algorithms**: RL needs state feedback
    
    Performance Comparison
    ----------------------
    Typical speedup vs step-by-step for open-loop simulation:
    
    - Smooth system (pendulum): 5-10× faster
    - Stiff system (chemical reaction): 10-50× faster  
    - With tight tolerances (rtol=1e-9): 20-100× faster
    
    The speedup comes from:
    - Single integrator initialization (not N times)
    - Adaptive steps naturally adjust to difficulty
    - Efficient dense output evaluation
    
    Interpolation Quality
    ---------------------
    **Linear interpolation** (default):
    - Fast, robust, stable
    - Accuracy limited by adaptive integrator's step size
    - Recommended for most applications
    
    **Cubic interpolation**:
    - Smoother trajectories (C¹ continuous)
    - Better for visualization
    - Requires ≥4 adaptive points (automatic fallback if fewer)
    - Slightly slower
    
    **Note**: For best accuracy, use integrator's native dense output
    (DENSE_OUTPUT mode) rather than post-interpolation. However, DENSE_OUTPUT
    is much slower for long simulations.
    
    Examples
    --------
    **Basic batch simulation:**
    
    >>> from src.systems.examples import Pendulum
    >>> continuous = Pendulum(m=1.0, l=0.5)
    >>> 
    >>> # Create batch-mode discretization
    >>> discrete = discretize_batch(continuous, dt=0.01)
    >>> # Uses LSODA (auto stiffness detection) by default
    >>> 
    >>> # Simulate 1000 steps - single integration call
    >>> result = discrete.simulate(
    ...     x0=np.array([np.pi/4, 0.0]),
    ...     u_sequence=None,  # Open-loop (no control)
    ...     n_steps=1000
    ... )
    >>> 
    >>> # Plot smooth trajectory
    >>> import matplotlib.pyplot as plt
    >>> t = result['time_steps'] * result['dt']
    >>> plt.plot(t, result['states'][:, 0], label='θ(t)')
    
    **High-accuracy trajectory:**
    
    >>> # Very tight tolerances for publication-quality results
    >>> discrete = discretize_batch(
    ...     continuous, dt=0.01, 
    ...     method='DOP853',  # 8th order
    ...     rtol=1e-12, 
    ...     atol=1e-14
    ... )
    >>> result = discrete.simulate(x0, None, 1000)
    
    **With open-loop control:**
    
    >>> # Time-varying control (not state feedback!)
    >>> def u_func(k):
    ...     return np.array([np.sin(k * 0.1)])
    >>> 
    >>> discrete = discretize_batch(continuous, dt=0.01)
    >>> result = discrete.simulate(x0, u_sequence=u_func, n_steps=500)
    >>> # Control evaluated at each time step: u(0), u(1), ..., u(499)
    
    **Stiff system:**
    
    >>> from src.systems.examples import VanDerPol
    >>> stiff_system = VanDerPol(mu=1000)  # Very stiff
    >>> 
    >>> # Use implicit method for stiff systems
    >>> discrete = discretize_batch(stiff_system, dt=0.01, method='Radau')
    >>> result = discrete.simulate(x0, None, 10000)
    
    **Cubic interpolation for smooth plots:**
    
    >>> discrete = discretize_batch(
    ...     continuous, dt=0.01, 
    ...     interpolation_kind='cubic'
    ... )
    >>> result = discrete.simulate(x0, None, 1000)
    >>> # Smoother trajectory, better for publication figures
    
    **Parameter sweep (batch processing):**
    
    >>> masses = np.linspace(0.5, 2.0, 10)
    >>> trajectories = []
    >>> 
    >>> for m in masses:
    ...     system = Pendulum(m=m, l=0.5)
    ...     discrete = discretize_batch(system, dt=0.01, method='LSODA')
    ...     result = discrete.simulate(x0, None, 500)
    ...     trajectories.append(result['states'])
    >>> 
    >>> # Plot family of trajectories
    >>> for i, traj in enumerate(trajectories):
    ...     plt.plot(t, traj[:, 0], alpha=0.5, label=f'm={masses[i]:.1f}')
    
    **Performance comparison:**
    
    >>> import time
    >>> 
    >>> # Batch mode
    >>> discrete_batch = discretize_batch(continuous, dt=0.01, method='LSODA')
    >>> start = time.time()
    >>> result_batch = discrete_batch.simulate(x0, None, 10000)
    >>> time_batch = time.time() - start
    >>> 
    >>> # Step-by-step mode
    >>> discrete_step = discretize(continuous, dt=0.01, method='RK45')
    >>> start = time.time()
    >>> result_step = discrete_step.simulate(x0, None, 10000)
    >>> time_step = time.time() - start
    >>> 
    >>> print(f"Batch mode: {time_batch:.3f}s")
    >>> print(f"Step mode:  {time_step:.3f}s")
    >>> print(f"Speedup:    {time_step/time_batch:.1f}×")
    Batch mode: 0.012s
    Step mode:  0.089s
    Speedup:    7.4×
    
    **Checking adaptive integrator stats:**
    
    >>> discrete = discretize_batch(continuous, dt=0.01, method='RK45')
    >>> result = discrete.simulate(x0, None, 1000)
    >>> 
    >>> if 'metadata' in result and 'adaptive_points' in result['metadata']:
    ...     n_adaptive = result['metadata']['adaptive_points']
    ...     n_regular = len(result['states'])
    ...     print(f"Adaptive integrator used {n_adaptive} points")
    ...     print(f"Interpolated to {n_regular} regular points")
    ...     print(f"Compression: {n_regular/n_adaptive:.1f}× output points")
    
    **State feedback NOT supported:**
    
    >>> # This will NOT work as expected!
    >>> def feedback_controller(x, k):
    ...     return -K @ x
    >>> 
    >>> result = discrete_batch.simulate(x0, feedback_controller, 1000)
    >>> # ERROR: Batch mode doesn't support state feedback
    >>> # Use discretize() instead for control applications
    
    See Also
    --------
    discretize : Step-by-step discretization (supports state feedback)
    DiscretizedSystem : Full constructor with all options
    DiscretizationMode.BATCH_INTERPOLATION : Mode used by this function
    compare_modes : Compare performance of different modes
    
    Notes
    -----
    - Mode is always BATCH_INTERPOLATION (cannot be overridden)
    - step() method raises NotImplementedError in this mode
    - State feedback control not supported - use discretize() instead
    - Adaptive methods strongly recommended (defeats purpose otherwise)
    - For stochastic systems: Each call produces different trajectory due
      to different noise realization
    """
    return DiscretizedSystem(continuous_system, dt=dt, method=method,
                            mode=DiscretizationMode.BATCH_INTERPOLATION, **kwargs)


def analyze_discretization_error(continuous_system, x0, u_sequence, dt_values,
                                 method='rk4', n_steps=100, reference_dt=None):
    """
    Analyze discretization error vs time step for convergence study.
    
    Computes the numerical error for a sequence of time steps by comparing
    against a high-accuracy reference solution. Estimates the convergence
    rate (order of accuracy) from the error scaling with dt.
    
    This is essential for:
    - Verifying correct implementation of integration methods
    - Selecting appropriate dt for desired accuracy
    - Comparing different integration methods
    - Detecting implementation bugs (unexpected convergence rates)
    
    Parameters
    ----------
    continuous_system : ContinuousSystemBase
        System to analyze
    x0 : StateVector
        Initial condition (nx,)
    u_sequence : DiscreteControlInput, optional
        Control sequence (same for all dt values)
    dt_values : array_like
        Sequence of time steps to test (e.g., [0.1, 0.05, 0.025, 0.0125])
        Should span at least one order of magnitude for reliable convergence
        rate estimation.
    method : str, default='rk4'
        Integration method to analyze
    n_steps : int, default=100
        Number of time steps for largest dt. Smaller dt values simulate
        proportionally longer to reach same final time:
        T_final = n_steps * max(dt_values)
    reference_dt : float, optional
        Time step for reference solution. If None, uses min(dt_values) / 10.
        Reference solution uses LSODA with tight tolerances (rtol=1e-12,
        atol=1e-14) to approximate "exact" solution.
    
    Returns
    -------
    dict
        Dictionary containing convergence analysis results:
        
        - **dt_values** : list of float
            Time steps tested (copy of input)
        
        - **errors** : list of float
            RMS error for each dt value, computed as:
            error = sqrt(mean((y(t) - y_ref(t))^2))
            where mean is over all time points and state dimensions
        
        - **timings** : list of float
            Wall-clock time (seconds) for each simulation
        
        - **reference** : dict
            Reference simulation result from DiscretizedSystem.simulate()
        
        - **method** : str
            Integration method analyzed
        
        - **convergence_rate** : float
            Estimated order of accuracy (slope of log(error) vs log(dt)).
            
            **Expected values:**
            - Euler: ~1.0
            - Heun/Midpoint/RK2: ~2.0
            - RK4: ~4.0
            - RK45 (adaptive, tight tol): ~5.0
            
            **Interpretation:**
            - Rate < expected: Implementation bug or insufficient reference accuracy
            - Rate > expected: Asymptotic regime not reached (dt too large)
            - Rate ≈ expected: Correct implementation ✓
    
    Algorithm
    ---------
    1. Generate reference solution at dt_ref with high-accuracy LSODA
    2. For each dt in dt_values:
       a. Simulate system with method and dt
       b. Interpolate both trajectories to common time grid
       c. Compute RMS error between method and reference
       d. Record computation time
    3. Fit line to log(error) vs log(dt) to estimate convergence rate
    
    Convergence Rate Theory
    -----------------------
    For a pth-order method, the global error scales as:
    
        error ≈ C * dt^p
    
    Taking logarithms:
    
        log(error) ≈ log(C) + p * log(dt)
    
    The convergence rate p is the slope of the log-log plot. This is a
    fundamental property of numerical methods:
    
    - **Linear (p=1)**: Halving dt halves the error
    - **Quadratic (p=2)**: Halving dt quarters the error  
    - **Quartic (p=4)**: Halving dt reduces error by 16×
    
    Method Selection Guidance
    -------------------------
    Use convergence analysis to choose method and dt:
    
    **For real-time applications** (fixed computational budget):
    - Plot error vs timing
    - Choose method with lowest error for available time
    - Example: RK4 at dt=0.01 may beat RK45 at dt=0.001 if both take 0.1s
    
    **For accuracy requirements** (target error threshold):
    - Find dt where error < threshold
    - Choose method with largest viable dt (fastest)
    - Example: RK4 needs dt=0.001 but Euler needs dt=0.0001 for error < 1e-6
    
    **For long-term stability**:
    - Verify convergence rate matches expected order
    - Check that error doesn't grow catastrophically
    - Consider energy-preserving methods for conservative systems
    
    Interpreting Convergence Rates
    -------------------------------
    **Rate matches expected order** (e.g., RK4 gives ~4.0):
    - ✓ Implementation correct
    - ✓ dt range appropriate (in asymptotic regime)
    - ✓ Reference solution sufficiently accurate
    
    **Rate lower than expected** (e.g., RK4 gives ~2.5):
    - Reference solution not accurate enough (decrease reference_dt)
    - Implementation bug in method
    - System has discontinuities (rate limited by smoothness)
    - Floating-point roundoff dominates (dt too small)
    
    **Rate higher than expected** (e.g., RK4 gives ~5.0):
    - dt too large (not in asymptotic regime yet)
    - System has special structure method exploits
    - Insufficient dt values to establish trend
    
    **Rate near zero or negative**:
    - Serious implementation problem
    - Reference solution wrong
    - System behavior changes between dt values
    
    Notes
    -----
    - All simulations use same initial condition x0
    - Same control sequence u_sequence applied (scaled to final time)
    - Reference solution should have error << smallest dt error
    - Convergence rate assumes asymptotic regime (sufficiently small dt)
    - Results only valid for smooth systems (discontinuities reduce rate)
    
    Limitations
    -----------
    - Does not test adaptive methods at their natural tolerances
    - RMS error may hide localized large errors
    - Single initial condition may not be representative
    - Does not account for stiffness or stability limits
    
    Examples
    --------
    **Basic convergence study:**
    
    >>> from src.systems.examples import Pendulum
    >>> system = Pendulum(m=1.0, l=0.5)
    >>> x0 = np.array([np.pi/4, 0.0])  # 45° initial angle
    >>> 
    >>> # Test dt from 0.1 down to 0.00625 (4 halvings)
    >>> dt_values = 0.1 / 2**np.arange(5)  # [0.1, 0.05, 0.025, 0.0125, 0.00625]
    >>> 
    >>> results = analyze_discretization_error(
    ...     system, x0, u_sequence=None, 
    ...     dt_values=dt_values, 
    ...     method='rk4',
    ...     n_steps=100  # Simulate to t=10s
    ... )
    >>> 
    >>> print(f"Convergence rate: {results['convergence_rate']:.2f}")
    >>> print(f"Expected for RK4: ~4.0")
    Convergence rate: 3.98
    Expected for RK4: ~4.0
    
    **Compare multiple methods:**
    
    >>> methods = ['euler', 'midpoint', 'rk4']
    >>> dt_values = 0.1 / 2**np.arange(4)
    >>> 
    >>> for method in methods:
    ...     result = analyze_discretization_error(
    ...         system, x0, None, dt_values, method=method
    ...     )
    ...     print(f"{method:8s}: rate = {result['convergence_rate']:.2f}")
    euler   : rate = 0.99
    midpoint: rate = 2.01  
    rk4     : rate = 3.98
    
    **Visualize convergence:**
    
    >>> import matplotlib.pyplot as plt
    >>> 
    >>> result = analyze_discretization_error(system, x0, None, dt_values, 'rk4')
    >>> 
    >>> # Log-log plot
    >>> plt.figure(figsize=(10, 6))
    >>> plt.loglog(result['dt_values'], result['errors'], 'o-', label='RK4')
    >>> 
    >>> # Reference lines for different orders
    >>> dt_ref = np.array(result['dt_values'])
    >>> for order in [1, 2, 4]:
    ...     # Normalize to pass through first point
    ...     err_ref = result['errors'][0] * (dt_ref / dt_ref[0])**order
    ...     plt.loglog(dt_ref, err_ref, '--', alpha=0.5, 
    ...                label=f'Order {order}')
    >>> 
    >>> plt.xlabel('Time step dt')
    >>> plt.ylabel('RMS Error')
    >>> plt.title(f'Convergence Study (rate = {result["convergence_rate"]:.2f})')
    >>> plt.legend()
    >>> plt.grid(True, which='both', alpha=0.3)
    >>> plt.show()
    
    **Accuracy vs efficiency:**
    
    >>> # Plot error vs computation time
    >>> plt.figure()
    >>> plt.loglog(result['timings'], result['errors'], 'o-')
    >>> plt.xlabel('Computation Time (s)')
    >>> plt.ylabel('RMS Error')
    >>> plt.title('Error vs Computational Cost')
    >>> plt.grid(True)
    
    **Method selection for target accuracy:**
    
    >>> target_error = 1e-6
    >>> 
    >>> # Find required dt for each method
    >>> for method in ['euler', 'midpoint', 'rk4']:
    ...     result = analyze_discretization_error(
    ...         system, x0, None, dt_values, method=method
    ...     )
    ...     
    ...     # Interpolate to find dt for target error
    ...     # Assume error = C * dt^p
    ...     C = result['errors'][0] / dt_values[0]**result['convergence_rate']
    ...     dt_required = (target_error / C)**(1/result['convergence_rate'])
    ...     
    ...     print(f"{method:8s}: dt = {dt_required:.6f} for error < {target_error}")
    euler   : dt = 0.001000 for error < 1e-06
    midpoint: dt = 0.003162 for error < 1e-06
    rk4     : dt = 0.010000 for error < 1e-06
    
    **Detecting implementation bugs:**
    
    >>> # Custom method with bug
    >>> result = analyze_discretization_error(system, x0, None, dt_values, 
    ...                                       method='my_buggy_rk4')
    >>> 
    >>> if abs(result['convergence_rate'] - 4.0) > 0.5:
    ...     print(f"WARNING: Expected rate ~4.0, got {result['convergence_rate']:.2f}")
    ...     print("Check implementation!")
    
    See Also
    --------
    recommend_dt : Automatically recommend dt for target accuracy
    compare_modes : Compare discretization modes for a system
    DiscretizedSystem : The discretization class being analyzed
    """
    if reference_dt is None:
        reference_dt = min(dt_values) / 10
    
    ref = DiscretizedSystem(continuous_system, dt=reference_dt, method='LSODA',
                           mode=DiscretizationMode.BATCH_INTERPOLATION, rtol=1e-12, atol=1e-14)
    n_ref = int(n_steps * max(dt_values) / reference_dt)
    ref_result = ref.simulate(x0, u_sequence, n_ref)
    t_compare = np.arange(n_steps + 1) * max(dt_values)
    
    errors, timings = [], []
    
    for dt in dt_values:
        disc = DiscretizedSystem(continuous_system, dt=dt, method=method)
        start = time.time()
        n_curr = int(n_steps * max(dt_values) / dt)
        result = disc.simulate(x0, u_sequence, n_curr)
        timings.append(time.time() - start)
        
        # Interpolate both to comparison grid
        from scipy.interpolate import interp1d
        t_curr = result['time_steps'] * dt
        t_ref = ref_result['time_steps'] * reference_dt
        
        # Ensure comparison times are within both ranges
        t_min = max(t_curr[0], t_ref[0])
        t_max = min(t_curr[-1], t_ref[-1])
        t_compare_valid = t_compare[(t_compare >= t_min) & (t_compare <= t_max)]
        
        if len(t_compare_valid) == 0:
            # Fallback: just compare final points
            error = np.linalg.norm(result['states'][-1, :] - ref_result['states'][-1, :])
            errors.append(error)
            continue
        
        states_interp = np.zeros((len(t_compare_valid), continuous_system.nx))
        ref_interp = np.zeros((len(t_compare_valid), continuous_system.nx))
        
        # Use linear interpolation to avoid cubic spline issues
        for i in range(continuous_system.nx):
            states_interp[:, i] = interp1d(t_curr, result['states'][:, i], kind='linear')(t_compare_valid)
            ref_interp[:, i] = interp1d(t_ref, ref_result['states'][:, i], kind='linear')(t_compare_valid)
        
        errors.append(np.sqrt(np.mean((states_interp - ref_interp) ** 2)))
    
    # Estimate convergence rate
    log_dt, log_err = np.log(dt_values), np.log(np.array(errors) + 1e-16)
    convergence_rate = np.polyfit(log_dt, log_err, 1)[0]
    
    return {
        'dt_values': list(dt_values), 'errors': errors, 'timings': timings,
        'reference': ref_result, 'method': method, 'convergence_rate': convergence_rate
    }


def recommend_dt(continuous_system, x0, target_error=1e-6, method='rk4',
                dt_range=(1e-4, 0.1), n_test=10):
    """
    Recommend optimal time step for target discretization accuracy.
    
    Automatically selects the largest dt that achieves a specified error tolerance
    by testing logarithmically-spaced values and performing convergence analysis.
    This balances accuracy (meeting target_error) with computational efficiency
    (maximizing dt to minimize total steps).
    
    The function uses analyze_discretization_error() to evaluate discretization
    quality across a range of dt values, then recommends the largest dt that
    meets the target error threshold. If no dt achieves the target, it warns
    and returns the best available option.
    
    Parameters
    ----------
    continuous_system : ContinuousSystemBase
        Continuous system to discretize. Must implement dynamics() or f().
    x0 : StateVector
        Initial condition for test trajectory, shape (nx,).
        Should be representative of expected operating conditions.
    target_error : float, default=1e-6
        Target RMS error relative to high-accuracy reference solution.
        Defined as: error = sqrt(mean((x_approx - x_ref)²))
    method : str, default='rk4'
        Integration method to use. Common choices:
        - 'euler': First-order (simple, fast, less accurate)
        - 'midpoint': Second-order (moderate accuracy)
        - 'rk4': Fourth-order (good accuracy, industry standard)
        - 'RK45', 'RK23': Adaptive methods (efficient for smooth systems)
    dt_range : tuple of float, default=(1e-4, 0.1)
        Range of dt values to test: (dt_min, dt_max).
        Should span expected operating regime.
    n_test : int, default=10
        Number of dt values to test (logarithmically spaced).
        More points give finer resolution but increase computation time.
    
    Returns
    -------
    dict
        Dictionary with recommendation and supporting data:
        
        - recommended_dt : float
            Recommended time step (seconds). This is the LARGEST dt
            that achieves target_error, balancing accuracy and efficiency.
        
        - achieved_error : float
            RMS error at recommended_dt. Should be ≤ target_error
            (unless no dt achieves target, then this is best available).
        
        - timing : float
            Wall-clock time for test simulation at recommended_dt (seconds).
            Useful for estimating computational cost.
        
        - all_results : dict
            Complete output from analyze_discretization_error(), containing:
            * dt_values : List of all tested dt values
            * errors : List of corresponding RMS errors
            * timings : List of computation times
            * convergence_rate : Estimated order of accuracy
            * reference : High-accuracy reference trajectory
            * method : Integration method used
    
    Interpreting Results
    --------------------
    **recommended_dt Selection:**
    
    The algorithm selects dt using a "largest acceptable" strategy:
    
    1. Test n_test logarithmically-spaced dt values in dt_range
    2. For each dt, compute RMS error vs high-accuracy reference
    3. Find all dt where error < target_error
    4. Return the LARGEST such dt (most efficient)
    5. If none achieve target, return dt with smallest error + warning
    
    This maximizes efficiency while meeting accuracy requirements.
    
    **Accuracy Guarantees:**
    
    - achieved_error ≤ target_error: SUCCESS, recommendation is valid
    - achieved_error > target_error: WARNING issued, no dt achieves target
      * Consider: wider dt_range, more accurate method, or relax target
    
    **Efficiency Considerations:**
    
    - Larger dt → fewer steps → faster simulation
    - But larger dt → worse accuracy (eventually)
    - recommended_dt finds the sweet spot: maximum dt for acceptable error
    
    **Method Selection:**
    
    Different methods have different accuracy/cost tradeoffs:
    
    - euler: O(dt) error, very fast per step
      * Large dt needed for given error → many steps
    - rk4: O(dt⁴) error, ~4× cost per step
      * Much larger dt possible → often fewer total steps
    - RK45: Adaptive, very efficient for smooth systems
      * Can use larger dt on easy parts, smaller where needed
    
    Common Scenarios
    ----------------
    **Real-time control (dt constrained):**
    
    If you need a specific dt for hardware/control reasons:
    
    >>> # Check if dt=0.01 is accurate enough
    >>> result = recommend_dt(system, x0, target_error=1e-6, 
    ...                       dt_range=(0.01, 0.01), n_test=1)
    >>> if result['achieved_error'] > target_error:
    ...     print("Need more accurate method!")
    
    **Offline simulation (accuracy priority):**
    
    >>> # Find dt for very high accuracy
    >>> result = recommend_dt(system, x0, target_error=1e-9,
    ...                       dt_range=(1e-6, 1e-2), method='RK45')
    >>> dt = result['recommended_dt']  # Use for production runs
    
    **Quick prototyping (speed priority):**
    
    >>> # Find largest dt that's "good enough"
    >>> result = recommend_dt(system, x0, target_error=1e-4,
    ...                       dt_range=(1e-3, 0.1), method='euler')
    >>> # Fast iterations during development
    
    **Method comparison:**
    
    >>> # Which method gives largest usable dt?
    >>> for method in ['euler', 'midpoint', 'rk4']:
    ...     result = recommend_dt(system, x0, method=method)
    ...     print(f"{method:8s}: dt={result['recommended_dt']:.6f}, "
    ...           f"cost={result['timing']:.4f}s")
    
    Warnings
    --------
    - **Single trajectory test**: Recommendation based on one x0 and control
      sequence. May not generalize to all operating conditions.
    
    - **Reference accuracy**: Uses LSODA with tight tolerances (rtol=1e-12,
      atol=1e-14) as "truth". For extremely stiff systems, this may still
      have errors.
    
    - **Linear error model**: Assumes error ∝ dt^p. Valid for sufficiently
      small dt, but may break down for large dt or near singularities.
    
    - **No control sensitivity**: Recommendation doesn't account for how
      control performance degrades with larger dt. Use smaller dt for
      aggressive control or fast disturbance rejection.
    
    - **Computational overhead**: Testing n_test=10 values requires 10
      simulations. For expensive systems, consider reducing n_test or
      using coarser dt_range initially.
    
    Examples
    --------
    **Basic usage:**
    
    >>> from src.systems.examples import Pendulum
    >>> system = Pendulum(m=1.0, l=0.5, g=9.81)
    >>> x0 = np.array([0.1, 0.0])  # Small angle
    >>> 
    >>> result = recommend_dt(system, x0, target_error=1e-6)
    >>> print(f"Recommended dt: {result['recommended_dt']:.6f}s")
    >>> print(f"Achieved error: {result['achieved_error']:.2e}")
    >>> print(f"Convergence rate: {result['all_results']['convergence_rate']:.2f}")
    Recommended dt: 0.003162s
    Achieved error: 8.32e-07
    Convergence rate: 4.02
    
    **Compare methods:**
    
    >>> methods = ['euler', 'midpoint', 'rk4', 'RK45']
    >>> target = 1e-6
    >>> 
    >>> print(f"Target error: {target:.0e}
")
    >>> for method in methods:
    ...     result = recommend_dt(system, x0, target_error=target, method=method)
    ...     dt = result['recommended_dt']
    ...     err = result['achieved_error']
    ...     time = result['timing']
    ...     rate = result['all_results']['convergence_rate']
    ...     
    ...     print(f"{method:8s}: dt={dt:.6f}, error={err:.2e}, "
    ...           f"time={time:.4f}s, rate={rate:.2f}")
    Target error: 1e-06
    
    euler   : dt=0.000178, error=9.23e-07, time=0.1234s, rate=1.01
    midpoint: dt=0.001000, error=9.87e-07, time=0.0456s, rate=2.03
    rk4     : dt=0.003162, error=8.32e-07, time=0.0234s, rate=4.02
    RK45    : dt=0.010000, error=5.67e-07, time=0.0123s, rate=4.89
    
    **Analysis:** RK45 allows 31× larger dt than euler, with 10× less
    computation time! Higher-order methods pay off.
    
    **Handle insufficient accuracy:**
    
    >>> # Try to achieve very tight error with crude method
    >>> result = recommend_dt(system, x0, target_error=1e-10, method='euler',
    ...                       dt_range=(1e-5, 0.1))
    UserWarning: No dt achieves target 1.00e-10
    >>> 
    >>> print(f"Best available: dt={result['recommended_dt']:.6f}")
    >>> print(f"Achieved error: {result['achieved_error']:.2e}")
    >>> print("Consider: smaller dt_range[0] or better method")
    Best available: dt=0.000010
    Achieved error: 3.45e-09
    Consider: smaller dt_range[0] or better method
    
    **Visualize error vs dt:**
    
    >>> result = recommend_dt(system, x0, target_error=1e-6, method='rk4')
    >>> 
    >>> import matplotlib.pyplot as plt
    >>> dt_vals = result['all_results']['dt_values']
    >>> errors = result['all_results']['errors']
    >>> 
    >>> plt.loglog(dt_vals, errors, 'o-', label='Measured')
    >>> plt.axhline(1e-6, color='r', linestyle='--', label='Target')
    >>> plt.axvline(result['recommended_dt'], color='g', linestyle='--',
    ...            label='Recommended')
    >>> plt.xlabel('Time step dt (s)')
    >>> plt.ylabel('RMS Error')
    >>> plt.legend()
    >>> plt.grid(True)
    >>> plt.title(f"Recommended dt = {result['recommended_dt']:.6f}s")
    
    **Estimate total simulation cost:**
    
    >>> # How long will a 10-second simulation take?
    >>> result = recommend_dt(system, x0, target_error=1e-6, method='rk4')
    >>> 
    >>> dt = result['recommended_dt']
    >>> time_per_step = result['timing'] / 100  # Analysis used 100 steps
    >>> 
    >>> simulation_duration = 10.0  # seconds of simulated time
    >>> n_steps = int(simulation_duration / dt)
    >>> estimated_time = n_steps * time_per_step
    >>> 
    >>> print(f"Simulating {simulation_duration}s will take ~{estimated_time:.2f}s")
    >>> print(f"Real-time factor: {simulation_duration / estimated_time:.1f}×")
    Simulating 10.0s will take ~0.74s
    Real-time factor: 13.5×
    
    **Optimize for different accuracy requirements:**
    
    >>> # Rapid prototyping vs production
    >>> scenarios = [
    ...     ('prototype', 1e-4),  # Fast iterations
    ...     ('validation', 1e-6), # Standard accuracy
    ...     ('publication', 1e-9) # High accuracy
    ... ]
    >>> 
    >>> for name, target in scenarios:
    ...     result = recommend_dt(system, x0, target_error=target, method='RK45')
    ...     print(f"{name:12s}: dt={result['recommended_dt']:.6f}, "
    ...           f"error={result['achieved_error']:.2e}")
    prototype   : dt=0.031623, error=7.89e-05, 
    validation  : dt=0.010000, error=8.12e-07,
    publication : dt=0.001000, error=9.45e-10,
    
    **Production workflow:**
    
    >>> # 1. Find optimal dt for development
    >>> dev_result = recommend_dt(system, x0, target_error=1e-5, method='rk4')
    >>> dev_dt = dev_result['recommended_dt']
    >>> 
    >>> # 2. Do rapid iterations with this dt
    >>> dev_system = DiscretizedSystem(system, dt=dev_dt, method='rk4')
    >>> # ... design controller, tune parameters, etc.
    >>> 
    >>> # 3. For final validation, use tighter tolerance
    >>> prod_result = recommend_dt(system, x0, target_error=1e-8, method='RK45')
    >>> prod_dt = prod_result['recommended_dt']
    >>> 
    >>> # 4. Production runs with validated dt
    >>> prod_system = DiscretizedSystem(system, dt=prod_dt, method='RK45')
    >>> # ... final results, publications, deployment
    
    See Also
    --------
    analyze_discretization_error : Detailed convergence analysis used internally
    compute_discretization_quality : Quick quality check for configured system
    DiscretizedSystem : The discretization class being configured
    """
    
    dt_values = np.logspace(np.log10(dt_range[0]), np.log10(dt_range[1]), n_test)
    analysis = analyze_discretization_error(continuous_system, x0, None, dt_values, method, 100)
    
    errors = np.array(analysis['errors'])
    valid_mask = errors < target_error
    
    if not np.any(valid_mask):
        import warnings
        warnings.warn(f"No dt achieves target {target_error:.2e}", UserWarning)
        best_idx = np.argmin(errors)
    else:
        best_idx = np.where(valid_mask)[0][-1]
    
    return {
        'recommended_dt': float(dt_values[best_idx]),
        'achieved_error': float(errors[best_idx]),
        'timing': analysis['timings'][best_idx],
        'all_results': analysis
    }


def detect_sde_integrator(continuous_system):
    """Detect best SDE method for stochastic system."""
    if not continuous_system.is_stochastic:
        raise ValueError("System is not stochastic")
    
    if hasattr(continuous_system, 'is_additive_noise') and continuous_system.is_additive_noise():
        return 'euler_maruyama'
    if hasattr(continuous_system, 'is_diagonal_noise') and continuous_system.is_diagonal_noise():
        return 'milstein'
    return 'euler_maruyama'


def compute_discretization_quality(discrete_system, x0, u_sequence, n_steps, metrics=None):
    """
    Compute quality metrics for discretization configuration.
    
    Evaluates a configured DiscretizedSystem on timing and stability metrics
    to assess suitability for a given application. Useful for quick sanity
    checks before running expensive simulations.
    
    Parameters
    ----------
    discrete_system : DiscretizedSystem
        Configured discretization to evaluate
    x0 : StateVector
        Initial condition (nx,)
    u_sequence : DiscreteControlInput, optional
        Control sequence for simulation
    n_steps : int
        Number of time steps to simulate
    metrics : list of str, optional
        Metrics to compute. Available options:
        - 'timing': Measure computational performance
        - 'stability': Check numerical stability
        
        If None, computes ['timing', 'stability'] (all metrics).
    
    Returns
    -------
    dict
        Dictionary with requested metrics:
        
        **If 'timing' requested:**
        - timing : dict
            - total_time : float
                Wall-clock time for simulation (seconds)
            - time_per_step : float  
                Average time per step (seconds/step)
            - steps_per_second : float
                Throughput (steps/second)
        
        **If 'stability' requested:**
        - stability : dict
            - is_stable : bool
                True if final norm < 100× initial norm (heuristic)
            - final_norm : float
                ||x(T)||₂ (Euclidean norm of final state)
            - max_norm : float
                max_t ||x(t)||₂ (maximum norm over trajectory)
    
    Interpreting Results
    --------------------
    **Timing Metrics:**
    
    - **Real-time capable**: steps_per_second >> 1/dt
      - Example: dt=0.01 requires >100 steps/sec for 1× real-time
      - For 10× real-time: need >1000 steps/sec
    
    - **Method efficiency**: Compare time_per_step across methods
      - RK4: ~4 function evaluations per step
      - RK45: Variable, typically 6-10 evaluations per step
      - Adaptive methods faster per step on smooth systems
    
    - **Scalability**: time_per_step should be constant
      - Linear growth: Expected (fixed cost per step)
      - Superlinear growth: Memory/cache issues, numerical problems
    
    **Stability Metrics:**
    
    - **is_stable = True**: Trajectory bounded (not diverging)
      - Heuristic: final_norm < 100 * initial_norm
      - Does NOT guarantee physical correctness
      - May still have large numerical error
    
    - **is_stable = False**: Numerical instability likely
      - State norm grew by >100× 
      - Possible causes:
        * dt too large (violates CFL/stability conditions)
        * Method unsuitable for problem (explicit for stiff)
        * Physical instability (system actually unstable)
    
    - **max_norm >> final_norm**: Potential issues
      - Transient instability
      - Control saturation not modeled
      - Constraint violation
    
    Limitations
    -----------
    - Single trajectory test (may not be representative)
    - Stability check is heuristic (not rigorous)
    - No accuracy assessment (use analyze_discretization_error)
    - No convergence verification
    - Timing affected by system load, CPU throttling, etc.
    
    Examples
    --------
    **Basic quality check:**
    
    >>> from src.systems.examples import Pendulum
    >>> system = Pendulum(m=1.0, l=0.5)
    >>> discrete = DiscretizedSystem(system, dt=0.01, method='rk4')
    >>> 
    >>> quality = compute_discretization_quality(
    ...     discrete, 
    ...     x0=np.array([0.1, 0.0]),
    ...     u_sequence=None,
    ...     n_steps=1000
    ... )
    >>> 
    >>> print(f"Stable: {quality['stability']['is_stable']}")
    >>> print(f"Speed: {quality['timing']['steps_per_second']:.1f} steps/sec")
    Stable: True
    Speed: 15234.5 steps/sec
    
    **Check real-time capability:**
    
    >>> dt = 0.01  # 100 Hz
    >>> quality = compute_discretization_quality(discrete, x0, None, 1000)
    >>> 
    >>> realtime_factor = quality['timing']['steps_per_second'] * dt
    >>> print(f"Real-time factor: {realtime_factor:.1f}×")
    >>> 
    >>> if realtime_factor > 10:
    ...     print("✓ Suitable for real-time control (10× margin)")
    >>> elif realtime_factor > 1:
    ...     print("⚠ Marginal for real-time (consider faster method)")
    >>> else:
    ...     print("✗ Too slow for real-time")
    Real-time factor: 152.3×
    ✓ Suitable for real-time control (10× margin)
    
    **Compare methods:**
    
    >>> methods = ['euler', 'rk4', 'RK45']
    >>> x0 = np.array([0.1, 0.0])
    >>> 
    >>> for method in methods:
    ...     disc = DiscretizedSystem(system, dt=0.01, method=method)
    ...     quality = compute_discretization_quality(disc, x0, None, 1000)
    ...     
    ...     print(f"{method:8s}: {quality['timing']['steps_per_second']:8.1f} steps/s, "
    ...           f"stable={quality['stability']['is_stable']}")
    euler   :  25432.1 steps/s, stable=True
    rk4     :  15234.5 steps/s, stable=True
    RK45    :  12456.3 steps/s, stable=True
    
    **Stability diagnosis:**
    
    >>> # Try large dt that may be unstable
    >>> discrete = DiscretizedSystem(system, dt=0.1, method='euler')
    >>> quality = compute_discretization_quality(discrete, x0, None, 100)
    >>> 
    >>> if not quality['stability']['is_stable']:
    ...     print(f"⚠ UNSTABLE: norm grew to {quality['stability']['final_norm']:.2e}")
    ...     print(f"  Initial norm: {np.linalg.norm(x0):.2e}")
    ...     print(f"  Growth factor: {quality['stability']['final_norm'] / np.linalg.norm(x0):.1f}×")
    ...     print("  → Try smaller dt or different method")
    
    **Only compute timing (faster):**
    
    >>> quality = compute_discretization_quality(
    ...     discrete, x0, None, n_steps=10000,
    ...     metrics=['timing']  # Skip stability check
    ... )
    >>> print(f"Throughput: {quality['timing']['steps_per_second']:.1f} steps/sec")
    
    **Benchmark mode selection:**
    
    >>> modes = [
    ...     ('fixed', DiscretizationMode.FIXED_STEP),
    ...     ('dense', DiscretizationMode.DENSE_OUTPUT),
    ...     ('batch', DiscretizationMode.BATCH_INTERPOLATION)
    ... ]
    >>> 
    >>> for name, mode in modes:
    ...     disc = DiscretizedSystem(system, dt=0.01, method='RK45', mode=mode)
    ...     quality = compute_discretization_quality(disc, x0, None, 1000,
    ...                                              metrics=['timing'])
    ...     print(f"{name:8s}: {quality['timing']['total_time']:.4f}s")
    fixed   : 0.0856s
    dense   : 0.0834s
    batch   : 0.0124s  ← Fastest for batch simulation!
    
    See Also
    --------
    analyze_discretization_error : Comprehensive convergence analysis
    recommend_dt : Find dt for target accuracy
    compare_modes : Compare all discretization modes
    """
    if metrics is None:
        metrics = ['timing', 'stability']
    
    results = {}
    start = time.time()
    sim_result = discrete_system.simulate(x0, u_sequence, n_steps)
    elapsed = time.time() - start
    
    if 'timing' in metrics:
        results['timing'] = {
            'total_time': elapsed,
            'time_per_step': elapsed / n_steps,
            'steps_per_second': n_steps / elapsed
        }
    
    if 'stability' in metrics:
        norms = np.linalg.norm(sim_result['states'], axis=1)
        results['stability'] = {
            'is_stable': bool(norms[-1] < 100 * norms[0]),
            'final_norm': float(norms[-1]),
            'max_norm': float(np.max(norms))
        }
    
    return results


__all__ = [
    'DiscretizationMode', 'DiscretizedSystem', 'discretize', 'discretize_batch',
    'analyze_discretization_error', 'recommend_dt', 'detect_sde_integrator',
    'compute_discretization_quality'
]