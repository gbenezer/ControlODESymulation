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
Symbolic System Base - Core Symbolic Machinery (Time-Domain Agnostic)
======================================================================

This module provides the foundational abstract base class for all symbolic systems,
regardless of whether they are continuous-time or discrete-time. SymbolicSystemBase
extracts the ~1,800 lines of shared symbolic machinery that was previously duplicated
between continuous and discrete implementations.

Overview
--------
SymbolicSystemBase serves as the root of the symbolic system hierarchy, providing:
- Symbolic variable management (state_vars, control_vars, output_vars)
- Parameter handling and substitution
- Code generation via CodeGenerator (multi-backend)
- Backend management (NumPy, PyTorch, JAX)
- Equilibrium point management
- Configuration persistence (save/load)
- Performance tracking and statistics

This class is **time-domain agnostic** - it makes no assumptions about whether
dx/dt = f(x,u) represents continuous derivatives or x[k+1] = f(x[k], u[k])
represents discrete updates. Subclasses (via ContinuousSystemBase or DiscreteSystemBase)
provide the time-domain semantics.

Architecture
-----------
SymbolicSystemBase uses composition over inheritance for specialized functionality:

- **BackendManager**: Backend detection, conversion, and device placement
- **SymbolicValidator**: System definition validation
- **CodeGenerator**: Symbolic → numerical code generation with caching
- **EquilibriumHandler**: Multiple equilibrium point management

Concrete subclasses must inherit from BOTH SymbolicSystemBase and a time-domain
base (ContinuousSystemBase or DiscreteSystemBase) to get complete functionality.

Inheritance Hierarchy
--------------------
```
                    SymbolicSystemBase (abstract)
                            |
         +------------------+------------------+
         |                                     |
    ContinuousSymbolicSystem          DiscreteSymbolicSystem
    (+ ContinuousSystemBase)           (+ DiscreteSystemBase)
         |                                     |
         |                                     |
ContinuousStochasticSystem         DiscreteStochasticSystem
```

Key Design Decisions
-------------------
1. **Abstract define_system()**: Forces subclasses to define symbolic expressions
2. **Abstract print_equations()**: Notation differs (dx/dt vs x[k+1])
3. **No __call__ or integrate()**: Time-domain semantics belong in specialized bases
4. **Composition for utilities**: BackendManager, CodeGenerator, etc. are composed
5. **Template method pattern**: __init__ orchestrates: define → validate → initialize

What This Class Provides
------------------------
- Symbolic variable containers (state_vars, control_vars, etc.)
- Parameter dictionary with substitution
- System dimensions (nx, nu, ny, nq)
- Backend configuration and management
- Code generation and compilation
- Performance statistics
- Equilibrium point storage and retrieval
- Configuration serialization

What This Class Does NOT Provide
--------------------------------
- Forward dynamics evaluation (__call__, forward, step)
- Time integration (integrate, simulate, rollout)
- Linearization computation (linearize, linearized_dynamics)
- Output evaluation (observe)
- Time-domain specific semantics

These are provided by:
- ContinuousSystemBase + concrete implementations
- DiscreteSystemBase + concrete implementations

Mathematical Notation
--------------------
**Continuous Systems (ContinuousDynamicalSystem):**
- dx/dt = f(x, u, t)  [_f_sym represents dx/dt or q^(n)]
- y = h(x)           [_h_sym represents output]
- A = ∂f/∂x, B = ∂f/∂u  [continuous Jacobians]

**Discrete Systems (DiscreteDynamicalSystem):**
- x[k+1] = f(x[k], u[k])  [_f_sym represents x[k+1] or q[k+1]]
- y[k] = h(x[k])         [_h_sym represents output]
- Ad = ∂f/∂x, Bd = ∂f/∂u  [discrete Jacobians]

System Order
-----------
Systems can be defined in two equivalent ways:

**First-Order State-Space Form (order=1):**
- State: x = [q, q̇, q̈, ..., q^(n-1)] for nth-order physical system
- _f_sym returns ALL derivatives: [q̇, q̈, ..., q^(n)]
- Example: Pendulum with x = [θ, θ̇], f returns [θ̇, θ̈]

**Higher-Order Form (order=n):**
- State: x = [q, q̇, q̈, ..., q^(n-1)] for nth-order physical system
- _f_sym returns ONLY highest derivative q^(n)
- Example: Pendulum with x = [θ, θ̇], f returns only θ̈, order=2

Both are mathematically equivalent. The framework handles state-space
construction automatically during linearization.

Usage Example
-------------
```python
class MySystem(SymbolicSystemBase, ContinuousSystemBase):
    '''Concrete system combining symbolic machinery with continuous interface.'''
    
    def define_system(self, a=1.0, b=2.0):
        '''Define symbolic system.'''
        x = sp.symbols('x', real=True)
        u = sp.symbols('u', real=True)
        a_sym, b_sym = sp.symbols('a b', real=True, positive=True)
        
        self.state_vars = [x]
        self.control_vars = [u]
        self._f_sym = sp.Matrix([-a_sym * x + b_sym * u])
        self.parameters = {a_sym: a, b_sym: b}
        self.order = 1
    
    def print_equations(self, simplify=True):
        '''Implement with continuous notation.'''
        print("dx/dt = f(x, u)")
        # ... implementation details
    
    # Must also implement ContinuousSystemBase interface:
    # - __call__(x, u, t) → dx/dt
    # - integrate(x0, u, t_span)
    # - linearize(x_eq, u_eq) → (A, B)
    # - simulate(x0, controller, t_span, dt)
```

See Also
--------
- ContinuousSymbolicSystem: Concrete continuous-time implementation
- DiscreteSymbolicSystem: Concrete discrete-time implementation
- ContinuousSystemBase: Continuous-time interface
- DiscreteSystemBase: Discrete-time interface
- BackendManager: Multi-backend management
- CodeGenerator: Symbolic to numerical compilation
- SymbolicValidator: System definition validation
- EquilibriumHandler: Equilibrium point storage

Notes
-----
- State variables must be SymPy Symbol objects
- Parameters must use Symbol keys (not strings): {m: 1.0} not {'m': 1.0}
- System order must divide state dimension evenly: nx % order == 0
- Output functions h(x) should not depend on control u
- All backends produce numerically equivalent results

Authors
-------
Gil Benezer

License
-------
AGPL-3.0
"""

import json
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union, Any

import numpy as np
import sympy as sp

from src.systems.base.utils.backend_manager import BackendManager
from src.systems.base.utils.code_generator import CodeGenerator
from src.systems.base.utils.equilibrium_handler import EquilibriumHandler
from src.systems.base.utils.symbolic_validator import SymbolicValidator, ValidationError
from src.types.utilities import SymbolicValidationResult

if TYPE_CHECKING:
    import jax.numpy as jnp
    import torch

# Type aliases
from src.types.core import (
    ScalarLike,
    EquilibriumName,
    EquilibriumState,
    EquilibriumControl
)
from src.types.backends import Backend, Device


class SymbolicSystemBase(ABC):
    """
    Abstract base class for symbolic systems (time-domain agnostic).

    Provides symbolic machinery for ANY symbolic system, whether continuous
    or discrete time. This class extracts the ~1,800 lines of common code
    that was previously duplicated between SymbolicDynamicalSystem and
    DiscreteSymbolicSystem.

    Subclasses must:
    1. Inherit from BOTH SymbolicSystemBase AND a time-domain base
       (ContinuousSystemBase or DiscreteSystemBase)
    2. Implement the abstract define_system() method
    3. Implement the abstract print_equations() method
    4. Implement all methods from the time-domain base interface

    Attributes
    ----------
    state_vars : List[sp.Symbol]
        State variables as SymPy symbols (e.g., [x, v, theta])
    control_vars : List[sp.Symbol]
        Control variables as SymPy symbols (e.g., [u1, u2])
    output_vars : List[sp.Symbol]
        Output variable names (optional)
    parameters : Dict[sp.Symbol, float]
        System parameters with Symbol keys (e.g., {m: 1.0, k: 10.0})
    _f_sym : sp.Matrix
        Symbolic dynamics expression (interpretation depends on subclass)
    _h_sym : Optional[sp.Matrix]
        Symbolic output expression (None = identity output)
    order : int
        System order (1 = first-order, 2 = second-order, etc.)
    backend : BackendManager
        Backend management component
    equilibria : EquilibriumHandler
        Equilibrium point management

    Examples
    --------
    Concrete system combining symbolic base with continuous interface:
    >>> class LinearOscillator(SymbolicSystemBase, ContinuousSystemBase):
    ...     def define_system(self, k=1.0, c=0.1):
    ...         x, v = sp.symbols('x v', real=True)
    ...         u = sp.symbols('u', real=True)
    ...         k_sym, c_sym = sp.symbols('k c', positive=True)
    ...
    ...         self.state_vars = [x, v]
    ...         self.control_vars = [u]
    ...         self._f_sym = sp.Matrix([v, -k_sym*x - c_sym*v + u])
    ...         self.parameters = {k_sym: k, c_sym: c}
    ...         self.order = 1
    ...
    ...     def print_equations(self, simplify=True):
    ...         print("Continuous dynamics: dx/dt = f(x, u)")
    ...         # ... implementation
    ...
    ...     # Also implement ContinuousSystemBase interface
    ...     def __call__(self, x, u=None, t=0.0):
    ...         # ... implementation
    ...         pass
    ...
    >>> system = LinearOscillator(k=2.0, c=0.5)
    >>> system.nx  # Number of states
    2
    >>> system.parameters  # Numerical values
    {k: 2.0, c: 0.5}
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the symbolic system using template method pattern.

        CRITICAL: Uses cooperative multiple inheritance via super().__init__()
        to ensure all base classes in the MRO are properly initialized.

        This constructor follows a carefully orchestrated sequence:
        1. Initialize empty symbolic containers
        2. Create specialized components (validation, backend, etc.)
        3. **Call super().__init__() for cooperative inheritance**
        4. Call user-defined define_system() to populate symbolic expressions
        5. Validate the system definition
        6. Initialize components that depend on validated system

        The template method pattern ensures consistent initialization across
        all symbolic systems while allowing customization via define_system().

        Parameters
        ----------
        *args : tuple
            Positional arguments passed to define_system()
        **kwargs : dict
            Keyword arguments passed to define_system()

        Raises
        ------
        ValidationError
            If the system definition is invalid (missing required attributes,
            wrong types, inconsistent dimensions, etc.)

        Notes
        -----
        Subclasses should NOT override __init__. Instead, implement define_system()
        to specify the symbolic system definition.

        The initialization sequence is critical:
        - BackendManager created first (needed by other components)
        - EquilibriumHandler created early (define_system may add equilibria)
        - super().__init__() called BEFORE define_system (cooperative inheritance)
        - Validator created after define_system (validates populated system)
        - CodeGenerator created last (needs validated system)

        Examples
        --------
        >>> class MySystem(SymbolicSystemBase, ContinuousSystemBase):
        ...     def define_system(self, param=1.0):
        ...         # Define your system here
        ...         pass
        ...
        >>> system = MySystem(param=2.0)  # Calls __init__ → define_system → validate
        """

        # ====================================================================
        # Phase 1: Initialize Symbolic Containers
        # ====================================================================
        # These are populated by the user-defined define_system() method

        self.state_vars: List[sp.Symbol] = []
        """State variables as SymPy Symbol objects (e.g., [x, y, theta])"""

        self.control_vars: List[sp.Symbol] = []
        """Control variables as SymPy Symbol objects (e.g., [u1, u2])"""

        self.output_vars: List[sp.Symbol] = []
        """Output variables (optional, used if custom output defined)"""

        self.parameters: Dict[sp.Symbol, float] = {}
        """System parameters: Symbol → numeric value (e.g., {m: 1.0, k: 10.0})"""

        self._f_sym: Optional[sp.Matrix] = None
        """
        Symbolic dynamics (interpretation depends on subclass):
        - Continuous: dx/dt = f(x, u) or q^(n) = f(x, u) for nth-order
        - Discrete: x[k+1] = f(x[k], u[k]) or q[k+1] = f(x[k], u[k])
        """

        self._h_sym: Optional[sp.Matrix] = None
        """Symbolic output: y = h(x). If None, output is identity (y = x)"""

        self.order: int = 1
        """
        System order: 1 (first-order), 2 (second-order), etc.
        Controls interpretation of _f_sym (all derivatives vs highest only)
        """

        # ====================================================================
        # Phase 2: Initialize Components (Pre-Definition)
        # ====================================================================

        # COMPOSITION: Delegate backend management
        self.backend = BackendManager(default_backend="numpy", default_device="cpu")
        """Backend manager handles detection, conversion, and device placement"""

        # COMPOSITION: Delegate validation (created after define_system)
        self._validator: Optional[SymbolicValidator] = None
        """Validator checks system definition for correctness"""

        # COMPOSITION: Delegate equilibrium management
        # Note: Initialized early because define_system might add equilibria
        self.equilibria = EquilibriumHandler(nx=0, nu=0)
        """Equilibrium handler manages multiple equilibrium points"""

        # COMPOSITION: Delegate code generation (initialized after validation)
        self._code_gen: Optional[CodeGenerator] = None
        """Code generator creates numerical functions from symbolic expressions"""

        # Initialization flag
        self._initialized: bool = False
        """Tracks whether system has been successfully initialized and validated"""
        
        # ====================================================================
        # Phase 3: Cooperative Multiple Inheritance
        # ====================================================================
        # CRITICAL FIX: Call super().__init__() to ensure all base classes
        # in the Method Resolution Order (MRO) are properly initialized
        
        # This is essential for multiple inheritance to work correctly:
        # - If inheriting from (SymbolicSystemBase, ContinuousSystemBase),
        #   this ensures ContinuousSystemBase.__init__() is called (if it exists)
        # - If ContinuousSystemBase has no __init__, Python's object.__init__() is called
        # - This maintains the cooperative inheritance chain
        
        super().__init__()

        # ====================================================================
        # Phase 4: Template Method Pattern - Define → Validate → Initialize
        # ====================================================================

        # Step 1: Call user-defined system definition
        self.define_system(*args, **kwargs)

        # Step 2: Create validator and validate system definition
        self._validator = SymbolicValidator(self)
        try:
            validation_result: SymbolicValidationResult = self._validator.validate(raise_on_error=True)
        except ValidationError as e:
            # Re-raise with context about which system failed
            raise ValidationError(
                f"Validation failed for {self.__class__.__name__}:\n{str(e)}"
            ) from e

        # Step 3: Update equilibrium handler dimensions (now that we know nx, nu)
        self.equilibria.nx = self.nx
        self.equilibria.nu = self.nu

        # Step 4: Mark as initialized (validation passed)
        self._initialized = True

        # Step 5: Initialize code generator (depends on validated system)
        self._code_gen = CodeGenerator(self)

    # ========================================================================
    # String Representations
    # ========================================================================

    def __repr__(self) -> str:
        """
        Return detailed string representation for debugging.

        Returns
        -------
        str
            Detailed representation including dimensions, backend, and device

        Examples
        --------
        >>> system = SimplePendulum()
        >>> repr(system)
        'SimplePendulum(nx=2, nu=1, ny=2, order=2, backend=numpy, device=cpu)'
        """
        return (
            f"{self.__class__.__name__}("
            f"nx={self.nx}, nu={self.nu}, ny={self.ny}, order={self.order}, "
            f"backend={self._default_backend}, device={self._preferred_device})"
        )

    def __str__(self) -> str:
        """
        Return human-readable string representation.

        Returns
        -------
        str
            Concise representation with key information

        Examples
        --------
        >>> system = SimplePendulum()
        >>> str(system)
        'SimplePendulum(nx=2, nu=1, backend=numpy)'
        """
        equilibria_str = (
            f", {len(self.equilibria.list_names())} equilibria"
            if len(self.equilibria.list_names()) > 1
            else ""
        )
        return (
            f"{self.__class__.__name__}(nx={self.nx}, nu={self.nu}, "
            f"backend={self._default_backend}{equilibria_str})"
        )

    # ========================================================================
    # Abstract Methods - Must Be Implemented by Subclasses
    # ========================================================================

    @abstractmethod
    def define_system(self, *args, **kwargs):
        """
        Define the symbolic system (must be implemented by subclasses).

        This method must populate the following attributes:

        **Required Attributes:**

        - ``self.state_vars``: List[sp.Symbol]
            State variables (e.g., [x, y, theta])
            Cannot be empty
        - ``self.control_vars``: List[sp.Symbol]
            Control variables (e.g., [u1, u2])
            Empty list for autonomous systems
        - ``self._f_sym``: sp.Matrix
            Symbolic dynamics (column vector)
        - ``self.parameters``: Dict[sp.Symbol, float]
            Parameter values with Symbol keys (NOT strings!)

        **Optional Attributes:**

        - ``self.output_vars``: List[sp.Symbol]
            Output variable names (optional)
        - ``self._h_sym``: sp.Matrix
            Symbolic output function (None = identity)
        - ``self.order``: int
            System order (default: 1)

        Parameters
        ----------
        *args : tuple
            System-specific positional arguments (e.g., mass, length, damping)
        **kwargs : dict
            System-specific keyword arguments

        Raises
        ------
        ValidationError
            If the defined system is invalid (checked after this method returns)

        Notes
        -----
        **CRITICAL**: self.parameters must use SymPy Symbol objects as keys!

        Correct::

            {m: 1.0, l: 0.5}

        Incorrect::

            {'m': 1.0, 'l': 0.5}  # Strings won't work!

        **System Order - Two Equivalent Formulations:**

        1. **First-Order State-Space Form (order=1):**
           - State: x = [q, q̇] for a 2nd-order physical system
           - _f_sym returns ALL derivatives: [q̇, q̈]
           - Set: self.order = 1

           Example::

               self.state_vars = [theta, theta_dot]
               self._f_sym = sp.Matrix([
                   theta_dot,                      # dθ/dt = θ̇
                   -k*theta - c*theta_dot + u      # dθ̇/dt = θ̈
               ])
               self.order = 1

        2. **Higher-Order Form (order=n):**
           - State: x = [q, q̇] for a 2nd-order physical system
           - _f_sym returns ONLY highest derivative: q̈
           - Set: self.order = 2

           Example::

               self.state_vars = [theta, theta_dot]
               self._f_sym = sp.Matrix([
                   -k*theta - c*theta_dot + u  # Only θ̈
               ])
               self.order = 2

        Both formulations are mathematically equivalent! The framework handles
        state-space construction automatically during linearization.

        **When to use which:**

        - Use order=1 (state-space) for: simpler code, explicit derivatives
        - Use order=n (higher-order) for: physics-focused definitions, cleaner dynamics

        **Validation rules:**

        - For order=1: len(_f_sym) must equal nx
        - For order=n: len(_f_sym) must equal nq, and nx must be divisible by order

        Examples
        --------
        First-order system::

            def define_system(self, a=1.0):
                x = sp.symbols('x')
                u = sp.symbols('u')
                a_sym = sp.symbols('a', real=True, positive=True)

                self.state_vars = [x]
                self.control_vars = [u]
                self._f_sym = sp.Matrix([-a_sym * x + u])
                self.parameters = {a_sym: a}
                self.order = 1

        Second-order (state-space form)::

            def define_system(self, m=1.0, k=10.0, c=0.5):
                q, q_dot = sp.symbols('q q_dot')
                u = sp.symbols('u')
                m_sym, k_sym, c_sym = sp.symbols('m k c', positive=True)

                # Return both derivatives explicitly
                self.state_vars = [q, q_dot]
                self.control_vars = [u]
                self._f_sym = sp.Matrix([
                    q_dot,                                  # dq/dt
                    (-k_sym*q - c_sym*q_dot + u)/m_sym     # dq̇/dt = q̈
                ])
                self.parameters = {m_sym: m, k_sym: k, c_sym: c}
                self.order = 1  # First-order state-space

        Second-order (higher-order form)::

            def define_system(self, m=1.0, k=10.0, c=0.5):
                q, q_dot = sp.symbols('q q_dot')
                u = sp.symbols('u')
                m_sym, k_sym, c_sym = sp.symbols('m k c', positive=True)

                # Return only acceleration
                q_ddot = (-k_sym*q - c_sym*q_dot + u)/m_sym

                self.state_vars = [q, q_dot]
                self.control_vars = [u]
                self._f_sym = sp.Matrix([q_ddot])  # Only highest derivative
                self.parameters = {m_sym: m, k_sym: k, c_sym: c}
                self.order = 2  # Second-order form
        """
        pass

    @abstractmethod
    def print_equations(self, simplify: bool = True):
        """
        Print symbolic equations in human-readable format.

        This method is abstract because the notation differs between continuous
        and discrete systems:
        - Continuous: "dx/dt = f(x, u)" or "dθ/dt", "dθ̇/dt"
        - Discrete: "x[k+1] = f(x[k], u[k])" or "θ[k+1]", "θ̇[k+1]"

        Subclasses must implement this with appropriate notation for their
        time-domain semantics.

        Parameters
        ----------
        simplify : bool
            If True, simplify expressions before printing
            If False, print raw expressions

        Notes
        -----
        Typical implementation should display:
        - System name
        - State and control variables
        - System order and dimensions
        - Dynamics equations with proper notation
        - Output equations (if defined)

        Examples
        --------
        Continuous implementation::

            def print_equations(self, simplify=True):
                print("=" * 70)
                print(f"{self.__class__.__name__}")
                print("=" * 70)
                print(f"State Variables: {self.state_vars}")
                print(f"Control Variables: {self.control_vars}")
                print(f"System Order: {self.order}")
                print(f"Dimensions: nx={self.nx}, nu={self.nu}, ny={self.ny}")

                print("\\nDynamics: dx/dt = f(x, u)")
                for var, expr in zip(self.state_vars, self._f_sym):
                    expr_sub = self.substitute_parameters(expr)
                    if simplify:
                        expr_sub = sp.simplify(expr_sub)
                    print(f"  d{var}/dt = {expr_sub}")

                if self._h_sym is not None:
                    print("\\nOutput: y = h(x)")
                    for i, expr in enumerate(self._h_sym):
                        expr_sub = self.substitute_parameters(expr)
                        if simplify:
                            expr_sub = sp.simplify(expr_sub)
                        print(f"  y[{i}] = {expr_sub}")

                print("=" * 70)

        Discrete implementation::

            def print_equations(self, simplify=True):
                print("=" * 70)
                print(f"{self.__class__.__name__}")
                print("=" * 70)
                print(f"State Variables: {self.state_vars}")
                print(f"Control Variables: {self.control_vars}")
                print(f"System Order: {self.order}")
                print(f"Dimensions: nx={self.nx}, nu={self.nu}, ny={self.ny}")

                print("\\nDynamics: x[k+1] = f(x[k], u[k])")
                for var, expr in zip(self.state_vars, self._f_sym):
                    expr_sub = self.substitute_parameters(expr)
                    if simplify:
                        expr_sub = sp.simplify(expr_sub)
                    print(f"  {var}[k+1] = {expr_sub}")

                if self._h_sym is not None:
                    print("\\nOutput: y[k] = h(x[k])")
                    for i, expr in enumerate(self._h_sym):
                        expr_sub = self.substitute_parameters(expr)
                        if simplify:
                            expr_sub = sp.simplify(expr_sub)
                        print(f"  y[{i}] = {expr_sub}")

                print("=" * 70)
        """
        pass
    
    # ========================================================================
    # Hook Method for Equilibrium Verification (Template Method Pattern)
    # ========================================================================

    def _verify_equilibrium_numpy(
        self, x_eq: np.ndarray, u_eq: np.ndarray, tol: ScalarLike
    ) -> bool:
        """
        Verify equilibrium condition (hook method for concrete classes).

        This is a template method hook that concrete classes override to
        provide time-domain-specific equilibrium verification:
        - Continuous: ||f(x_eq, u_eq)|| < tol
        - Discrete: ||f(x_eq, u_eq) - x_eq|| < tol

        Base implementation raises NotImplementedError to force subclasses
        to provide their own verification logic.

        Parameters
        ----------
        x_eq : np.ndarray
            Equilibrium state (nx,)
        u_eq : np.ndarray
            Equilibrium control (nu,)
        tol : float
            Tolerance for verification

        Returns
        -------
        bool
            True if equilibrium condition is satisfied

        Raises
        ------
        NotImplementedError
            If subclass doesn't implement this method

        Notes
        -----
        This method is called by add_equilibrium() when verify=True.
        Concrete classes must override to provide appropriate verification.

        Examples
        --------
        In ContinuousDynamicalSystem:

        >>> def _verify_equilibrium_numpy(self, x_eq, u_eq, tol):
        ...     '''Continuous: f(x_eq, u_eq) ≈ 0'''
        ...     dx = self(x_eq, u_eq, t=0.0)
        ...     return np.linalg.norm(dx) < tol

        In DiscreteDynamicalSystem:

        >>> def _verify_equilibrium_numpy(self, x_eq, u_eq, tol):
        ...     '''Discrete: f(x_eq, u_eq) ≈ x_eq'''
        ...     x_next = self.step(x_eq, u_eq, k=0)
        ...     return np.linalg.norm(x_next - x_eq) < tol
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _verify_equilibrium_numpy(). "
            f"This method should check the equilibrium condition for your system type.\n"
            f"\n"
            f"For continuous systems:\n"
            f"    dx = self(x_eq, u_eq, t=0.0)\n"
            f"    return np.linalg.norm(dx) < tol\n"
            f"\n"
            f"For discrete systems:\n"
            f"    x_next = self.step(x_eq, u_eq, k=0)\n"
            f"    return np.linalg.norm(x_next - x_eq) < tol"
        )

    # ========================================================================
    # Properties - System Dimensions
    # ========================================================================

    @property
    def nx(self) -> int:
        """
        Number of states.

        Returns
        -------
        int
            Dimension of state vector x

        Examples
        --------
        >>> system = SimplePendulum()
        >>> system.nx
        2
        """
        return len(self.state_vars)

    @property
    def nu(self) -> int:
        """
        Number of controls.

        Returns
        -------
        int
            Dimension of control vector u

        Examples
        --------
        >>> system = SimplePendulum()
        >>> system.nu
        1
        """
        return len(self.control_vars)

    @property
    def ny(self) -> int:
        """
        Number of outputs.

        Returns
        -------
        int
            Dimension of output vector y

        Notes
        -----
        - If output_vars is defined: ny = len(output_vars)
        - Else if _h_sym is defined: ny = number of rows in _h_sym
        - Else (identity output): ny = nx

        Examples
        --------
        >>> system = SimplePendulum()  # No custom output
        >>> system.ny  # Same as nx
        2
        """
        if self.output_vars:
            return len(self.output_vars)
        elif self._h_sym is not None:
            return self._h_sym.shape[0]
        else:
            return self.nx

    @property
    def nq(self) -> int:
        """
        Number of generalized coordinates (for higher-order systems).

        Returns
        -------
        int
            Number of generalized coordinates

        Notes
        -----
        For an nth-order system with state x = [q, q̇, ..., q^(n-1)]:
        - nq = nx / order
        - For first-order systems: nq = nx
        - For second-order: nq = nx / 2

        Examples
        --------
        >>> # Second-order pendulum: x = [θ, θ̇]
        >>> system = SimplePendulum()
        >>> system.nx  # 2 states
        2
        >>> system.order  # 2nd order
        2
        >>> system.nq  # 1 generalized coordinate (θ)
        1
        """
        return self.nx // self.order if self.order > 1 else self.nx

    # ========================================================================
    # Properties - Backend Configuration (Backward Compatibility)
    # ========================================================================

    @property
    def _default_backend(self) -> Backend:
        """
        Get default backend (backward compatibility property).

        This property redirects to the BackendManager for actual storage.
        Kept for backward compatibility with code that accesses this attribute.

        Returns
        -------
        str
            Default backend name ('numpy', 'torch', or 'jax')

        Examples
        --------
        >>> system._default_backend
        'numpy'
        >>> system._default_backend = 'torch'  # Sets via BackendManager
        """
        return self.backend.default_backend

    @_default_backend.setter
    def _default_backend(self, value: Backend):
        """
        Set default backend (backward compatibility property).

        Parameters
        ----------
        value : str
            Backend name ('numpy', 'torch', or 'jax')
        """
        self.backend.set_default(value)

    @property
    def _preferred_device(self) -> Device:
        """
        Get preferred device (backward compatibility property).

        This property redirects to the BackendManager for actual storage.
        Kept for backward compatibility with code that accesses this attribute.

        Returns
        -------
        str
            Preferred device ('cpu', 'cuda', 'cuda:0', 'gpu:0', etc.)

        Examples
        --------
        >>> system._preferred_device
        'cpu'
        >>> system._preferred_device = 'cuda:0'  # Sets via BackendManager
        """
        return self.backend.preferred_device

    @_preferred_device.setter
    def _preferred_device(self, value: Device):
        """
        Set preferred device (backward compatibility property).

        Parameters
        ----------
        value : str
            Device string ('cpu', 'cuda', 'cuda:0', etc.)
        """
        self.backend.to_device(value)

    # ========================================================================
    # Backend Configuration Methods
    # ========================================================================

    def set_default_backend(
        self, backend: Backend, device: Optional[Device] = None
    ) -> "SymbolicSystemBase":
        """
        Set default backend and optionally device for this system.

        The default backend is used when backend='default' is passed to methods,
        or when no backend is specified and conversion is needed.

        Parameters
        ----------
        backend : str
            Backend name ('numpy', 'torch', or 'jax')
        device : Optional[str]
            Device for GPU backends ('cpu', 'cuda', 'cuda:0', 'gpu:0', etc.)
            If None, device is not changed.

        Returns
        -------
        SymbolicSystemBase
            Self (for method chaining)

        Raises
        ------
        ValueError
            If backend name is invalid
        RuntimeError
            If backend is not available (not installed)

        Examples
        --------
        >>> system.set_default_backend('torch', device='cuda:0')
        >>> system._default_backend
        'torch'
        >>> system._preferred_device
        'cuda:0'

        Method chaining:
        >>> system.set_default_backend('jax').compile(verbose=True)
        """
        self.backend.set_default(backend, device)
        return self

    def to_device(self, device: Device) -> "SymbolicSystemBase":
        """
        Set preferred device for PyTorch/JAX backends.

        Changes the device for all subsequent operations. Clears cached functions
        for backends that need recompilation (PyTorch, JAX) because device-specific
        code may differ.

        Parameters
        ----------
        device : str
            Device string ('cpu', 'cuda', 'cuda:0', 'gpu:0', 'tpu:0', etc.)

        Returns
        -------
        SymbolicSystemBase
            Self (for method chaining)

        Notes
        -----
        - NumPy always uses CPU (device setting ignored)
        - PyTorch and JAX respect device setting
        - Changing device clears cached functions for affected backends

        Examples
        --------
        >>> system.to_device('cuda:0')
        >>> system.set_default_backend('torch')
        >>> # All torch operations now use CUDA device 0

        Method chaining:
        >>> system.to_device('cuda').set_default_backend('torch')
        """
        self.backend.to_device(device)

        # Clear cached functions that need recompilation for new device
        if self.backend.default_backend in ["torch", "jax"]:
            self._clear_backend_cache(self.backend.default_backend)

        return self

    def _clear_backend_cache(self, backend: Backend):
        """
        Clear cached compiled functions for a specific backend.

        This is an internal method called when device changes require
        recompilation. Users should typically use reset_caches() instead.

        Parameters
        ----------
        backend : str
            Backend to clear ('numpy', 'torch', or 'jax')

        Notes
        -----
        This delegates to CodeGenerator.reset_cache()
        """
        self._code_gen.reset_cache([backend])

    @contextmanager
    def use_backend(self, backend: Backend, device: Optional[Device] = None):
        """
        Temporarily switch to a different backend and/or device.

        This context manager allows temporary backend changes without
        affecting the configured default. Useful for benchmarking or
        comparing backend performance.

        Parameters
        ----------
        backend : str
            Temporary backend to use ('numpy', 'torch', 'jax')
        device : Optional[str]
            Temporary device to use (None = keep current device)

        Yields
        ------
        SymbolicSystemBase
            Self with temporary backend configuration

        Examples
        --------
        >>> system.set_default_backend('numpy')
        >>>
        >>> # Temporarily use PyTorch
        >>> with system.use_backend('torch', device='cuda'):
        ...     dx = system(x, u, backend='default')  # Uses torch on CUDA
        >>>
        >>> # Back to NumPy after context
        >>> system._default_backend
        'numpy'

        Nested contexts:
        >>> with system.use_backend('torch'):
        ...     with system.use_backend('jax'):
        ...         # Uses JAX
        ...         pass
        ...     # Back to torch
        ...     pass
        >>> # Back to original
        """
        with self.backend.use_backend(backend, device):
            yield self

    def get_backend_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about backend configuration and status.

        Returns
        -------
        dict
            Dictionary containing:
            - 'default_backend': Current default backend
            - 'preferred_device': Current device setting
            - 'available_backends': List of installed backends
            - 'compiled_backends': List of backends with compiled functions
            - 'torch_available': Whether PyTorch is installed
            - 'jax_available': Whether JAX is installed
            - 'numpy_version': NumPy version string
            - 'torch_version': PyTorch version (or None)
            - 'jax_version': JAX version (or None)
            - 'initialized': Whether system is initialized

        Examples
        --------
        >>> info = system.get_backend_info()
        >>> print(f"Default: {info['default_backend']}")
        >>> print(f"Available: {info['available_backends']}")
        >>> print(f"Compiled: {info['compiled_backends']}")
        """
        # Get extended info from backend manager
        info = self.backend.get_extended_info()

        # Add code generation status
        compiled = [
            backend
            for backend in ["numpy", "torch", "jax"]
            if self._code_gen.is_compiled(backend)["f"]
        ]

        info["compiled_backends"] = compiled
        info["initialized"] = self._initialized

        return info

    # ========================================================================
    # Code Generation and Compilation
    # ========================================================================

    def compile(
        self, backends: Optional[List[Backend]] = None, verbose: bool = False, **kwargs
    ) -> Dict[str, float]:
        """
        Pre-compile dynamics functions for specified backends.

        Compilation happens lazily by default (on first use). This method
        allows eager compilation to reduce first-call latency and validate
        that code generation works correctly.

        Parameters
        ----------
        backends : Optional[List[str]]
            List of backends to compile ('numpy', 'torch', 'jax').
            If None, compiles for all available backends.
        verbose : bool
            If True, print compilation progress and timing
        **kwargs : dict
            Backend-specific compilation options

        Returns
        -------
        Dict[str, float]
            Compilation times per backend (seconds)

        Examples
        --------
        >>> # Compile for all available backends
        >>> times = system.compile(verbose=True)
        Compiling for numpy... 0.123s
        Compiling for torch... 0.456s
        Compiling for jax... 0.789s

        >>> # Compile only for specific backends
        >>> system.compile(backends=['numpy', 'torch'])

        >>> # Chain with other operations
        >>> system.compile().set_default_backend('torch')

        Notes
        -----
        - Compilation is cached - subsequent calls are no-ops unless cache is cleared
        - JAX compilation includes JIT, which may take longer initially
        - PyTorch compilation is optional (not JIT by default)
        - NumPy always uses regular Python functions (fastest to "compile")

        See Also
        --------
        reset_caches : Clear compiled function cache
        get_backend_info : Check compilation status
        """
        # Delegate to code generator's compile_all method
        return self._code_gen.compile_all(backends=backends, verbose=verbose, **kwargs)

    def reset_caches(self, backends: Optional[List[Backend]] = None):
        """
        Reset cached compiled functions for specified backends.

        Clears the code generation cache, forcing recompilation on next use.
        Useful when system parameters change or to free memory.

        Parameters
        ----------
        backends : Optional[List[str]]
            List of backends to reset ('numpy', 'torch', 'jax').
            If None, resets all backends.

        Examples
        --------
        >>> # Reset all cached functions
        >>> system.reset_caches()

        >>> # Reset only PyTorch cache
        >>> system.reset_caches(['torch'])

        >>> # After parameter update
        >>> system.parameters[m] = 2.0  # Changed mass
        >>> system.reset_caches()  # Force recompilation with new value

        Notes
        -----
        - Does not affect the system definition (state_vars, _f_sym, etc.)
        - Only clears the compiled numerical functions
        - Next function call will trigger recompilation
        - Use sparingly - compilation has overhead

        See Also
        --------
        compile : Pre-compile functions
        _clear_backend_cache : Clear single backend (internal use)
        """
        self._code_gen.reset_cache(backends)

    # ========================================================================
    # Performance Tracking
    # ========================================================================

    def get_performance_stats(self) -> Dict[str, float]:
        """
        Get performance statistics for system operations.

        Returns timing and call count information for key operations.
        Useful for profiling and optimization.

        Returns
        -------
        Dict[str, float]
            Dictionary containing:
            - 'forward_calls': Number of forward dynamics calls
            - 'forward_time': Total time in forward dynamics (seconds)
            - 'avg_forward_time': Average time per forward call (seconds)
            - (Other stats depend on concrete subclass implementation)

        Examples
        --------
        >>> # Run some operations
        >>> for _ in range(100):
        ...     dx = system(x, u)
        ...
        >>> stats = system.get_performance_stats()
        >>> print(f"Forward calls: {stats['forward_calls']}")
        >>> print(f"Avg time: {stats['avg_forward_time']:.6f}s")

        Notes
        -----
        - Statistics accumulate over system lifetime
        - Use reset_performance_stats() to clear counters
        - Timing includes overhead from backend detection/conversion
        - Concrete subclasses may add additional statistics

        See Also
        --------
        reset_performance_stats : Reset all performance counters
        """
        # Base class provides minimal stats
        # Concrete subclasses override to add dynamics/linearization stats
        return {
            "forward_calls": 0,
            "forward_time": 0.0,
            "avg_forward_time": 0.0,
        }

    def reset_performance_stats(self):
        """
        Reset all performance counters to zero.

        Clears timing and call count statistics across all components.

        Examples
        --------
        >>> system.reset_performance_stats()
        >>> stats = system.get_performance_stats()
        >>> stats['forward_calls']
        0

        Notes
        -----
        - Resets counters in all components (DynamicsEvaluator, etc.)
        - Does not affect compilation cache or system definition
        - Concrete subclasses override to reset their component stats
        """
        # Base class does nothing
        # Concrete subclasses override to reset their components
        pass

    # ========================================================================
    # Symbolic Utilities
    # ========================================================================

    def substitute_parameters(
        self, expr: Union[sp.Expr, sp.Matrix]
    ) -> Union[sp.Expr, sp.Matrix]:
        """
        Substitute numerical parameter values into symbolic expression.

        Replaces all parameter symbols with their numerical values
        from self.parameters dictionary.

        Parameters
        ----------
        expr : Union[sp.Expr, sp.Matrix]
            Symbolic expression or matrix

        Returns
        -------
        Union[sp.Expr, sp.Matrix]
            Expression with parameters substituted

        Examples
        --------
        >>> m, k = sp.symbols('m k')
        >>> expr = m * sp.symbols('x') + k
        >>> system.parameters = {m: 1.0, k: 10.0}
        >>> system.substitute_parameters(expr)
        x + 10.0

        Notes
        -----
        This is used internally by code generation to create
        parameter-specific numerical functions.
        """
        return expr.subs(self.parameters)

    # ========================================================================
    # Equilibrium Management
    # ========================================================================

    def add_equilibrium(
        self,
        name: EquilibriumName,
        x_eq: EquilibriumState,
        u_eq: EquilibriumControl,
        verify: bool = True,
        tol: float = 1e-6,
        **metadata,
    ):
        """
        Add an equilibrium point with optional verification.

        Equilibrium points are states where the dynamics are zero (or identity
        for discrete systems). The system can have multiple equilibria
        (e.g., pendulum upright vs downward).

        Parameters
        ----------
        name : str
            Unique name for this equilibrium (e.g., 'origin', 'upright', 'inverted')
        x_eq : np.ndarray
            Equilibrium state (nx,)
        u_eq : np.ndarray
            Equilibrium control (nu,)
        verify : bool
            If True, verify that equilibrium condition holds (default: True)
        tol : float
            Tolerance for verification (default: 1e-6)
        **metadata : dict
            Additional metadata to store (e.g., stability='stable')

        Raises
        ------
        ValueError
            If dimensions don't match system dimensions
        UserWarning
            If verification fails (not actually an equilibrium)

        Examples
        --------
        >>> # Pendulum downward equilibrium
        >>> system.add_equilibrium(
        ...     'downward',
        ...     x_eq=np.array([0.0, 0.0]),
        ...     u_eq=np.array([0.0]),
        ...     verify=True
        ... )

        >>> # Inverted pendulum (with metadata)
        >>> system.add_equilibrium(
        ...     'inverted',
        ...     x_eq=np.array([np.pi, 0.0]),
        ...     u_eq=np.array([0.0]),
        ...     stability='unstable',
        ...     notes='Requires active control'
        ... )

        >>> # Get equilibrium back
        >>> x_eq = system.equilibria.get_x('inverted')
        >>> u_eq = system.equilibria.get_u('inverted')

        Notes
        -----
        - Verification implementation depends on concrete subclass
        - If verification fails, a warning is issued but equilibrium is still added
        - Delegates to EquilibriumHandler.add() for storage and management
        - Concrete subclasses may override to provide custom verification

        See Also
        --------
        get_equilibrium : Retrieve equilibrium in specified backend
        list_equilibria : List all equilibrium names
        set_default_equilibrium : Set default equilibrium
        remove_equilibrium : Remove an equilibrium
        """
        # Base class: no verification function
        # Concrete subclasses override to provide verification via _verify_equilibrium_numpy
        
        # CRITICAL FIX: Properly handle verification function
        # FIXED IMPLEMENTATION: Verify first, then add
        
        # TODO: having a _verify_equilibrium_numpy() function seems ad-hoc
        # compared to having the verification be handled by the EquilibriumHandler;
        # may need minor refactoring
    
        if verify:
            # Check if concrete class implements verification
            try:
                is_valid = self._verify_equilibrium_numpy(x_eq, u_eq, tol)
                
                if not is_valid:
                    # Equilibrium verification failed
                    import warnings
                    warnings.warn(
                        f"Equilibrium '{name}' at x={x_eq}, u={u_eq} failed verification. "
                        f"The equilibrium condition is not satisfied within tolerance {tol}. "
                        f"Adding anyway, but this may not be a true equilibrium point.",
                        UserWarning,
                        stacklevel=2
                    )
                    
            except NotImplementedError:
                # Method not implemented - warn
                import warnings
                warnings.warn(
                    f"{self.__class__.__name__} does not implement _verify_equilibrium_numpy(). "
                    f"Equilibrium '{name}' will be added without verification.",
                    UserWarning,
                    stacklevel=2
                )
        
        # Add to handler (without verify_fn - we already checked)
        # Pass verify_fn=None to prevent EquilibriumHandler from trying to verify again
        self.equilibria.add(name, x_eq, u_eq, verify_fn=None, tol=tol, **metadata)

    def set_default_equilibrium(self, name: EquilibriumName) -> "SymbolicSystemBase":
        """
        Set default equilibrium for get operations without name.

        Parameters
        ----------
        name : str
            Name of equilibrium to use as default

        Returns
        -------
        SymbolicSystemBase
            Self for method chaining

        Examples
        --------
        >>> system.set_default_equilibrium('inverted')
        >>> x_eq = system.equilibria.get_x()  # Gets 'inverted' by default

        Method chaining:
        >>> system.set_default_equilibrium('upright').compile()
        """
        self.equilibria.set_default(name)
        return self

    def get_equilibrium(
        self, name: Optional[EquilibriumName] = None, backend: Optional[Backend] = None
    ) -> Tuple[EquilibriumState, EquilibriumControl]:
        """
        Get equilibrium state and control in specified backend.

        Parameters
        ----------
        name : Optional[str]
            Equilibrium name (None = default)
        backend : Optional[str]
            Backend for arrays (None = system default)

        Returns
        -------
        Tuple[ArrayLike, ArrayLike]
            (x_eq, u_eq) in requested backend

        Examples
        --------
        >>> x_eq, u_eq = system.get_equilibrium('inverted', backend='torch')
        >>> x_eq, u_eq = system.get_equilibrium()  # Default equilibrium, default backend
        """
        backend = backend or self._default_backend
        return self.equilibria.get_both(name, backend)

    def list_equilibria(self) -> List[EquilibriumName]:
        """
        List all equilibrium names.

        Returns
        -------
        List[str]
            Names of all defined equilibria

        Examples
        --------
        >>> system.list_equilibria()
        ['origin', 'downward', 'inverted']
        """
        return self.equilibria.list_names()

    def get_equilibrium_metadata(self, name: Optional[EquilibriumName] = None) -> Dict:
        """
        Get metadata for equilibrium.

        Parameters
        ----------
        name : Optional[str]
            Equilibrium name (None = default)

        Returns
        -------
        Dict
            Metadata dictionary

        Examples
        --------
        >>> meta = system.get_equilibrium_metadata('inverted')
        >>> print(meta['stability'])
        'unstable'
        """
        return self.equilibria.get_metadata(name)

    def remove_equilibrium(self, name: EquilibriumName):
        """
        Remove an equilibrium point.

        Parameters
        ----------
        name : str
            Equilibrium name to remove

        Raises
        ------
        ValueError
            If trying to remove 'origin' or nonexistent equilibrium

        Examples
        --------
        >>> system.remove_equilibrium('test_point')
        """
        if name == "origin":
            raise ValueError("Cannot remove origin equilibrium")

        if name not in self.equilibria._equilibria:
            raise ValueError(f"Unknown equilibrium '{name}'")

        del self.equilibria._equilibria[name]

        # Reset default if we removed it
        if self.equilibria._default == name:
            self.equilibria._default = "origin"

    # ========================================================================
    # Configuration Persistence
    # ========================================================================

    def get_config_dict(self) -> Dict:
        """
        Get system configuration as dictionary.

        Returns
        -------
        Dict
            Configuration dictionary containing:
            - 'class_name': System class name
            - 'state_vars': State variable names
            - 'control_vars': Control variable names
            - 'output_vars': Output variable names
            - 'parameters': Parameter values (as dict)
            - 'order': System order
            - 'nx', 'nu', 'ny': Dimensions
            - 'backend': Default backend
            - 'device': Preferred device

        Examples
        --------
        >>> config = system.get_config_dict()
        >>> config['nx']
        2
        >>> config['parameters']
        {'m': 1.0, 'k': 10.0}

        Notes
        -----
        - Useful for saving system configuration to file
        - Does not include compiled functions or cached data
        - Can be used with save_config() for persistence

        See Also
        --------
        save_config : Save configuration to JSON file
        """
        # Convert parameter Symbol keys to strings for JSON serialization
        params_dict = {str(k): float(v) for k, v in self.parameters.items()}

        return {
            "class_name": self.__class__.__name__,
            "state_vars": [str(v) for v in self.state_vars],
            "control_vars": [str(v) for v in self.control_vars],
            "output_vars": [str(v) for v in self.output_vars],
            "parameters": params_dict,
            "order": self.order,
            "nx": self.nx,
            "nu": self.nu,
            "ny": self.ny,
            "backend": self._default_backend,
            "device": self._preferred_device,
        }

    def save_config(self, filename: str):
        """
        Save system configuration to JSON file.

        Parameters
        ----------
        filename : str
            Path to output file (will be created/overwritten)

        Examples
        --------
        >>> system.save_config('pendulum_config.json')

        >>> # Load config (manually)
        >>> import json
        >>> with open('pendulum_config.json', 'r') as f:
        ...     config = json.load(f)
        >>> print(config['parameters'])

        Notes
        -----
        - Saves only configuration, not compiled functions
        - Use get_config_dict() to get config without saving
        - JSON format enables easy sharing and version control

        See Also
        --------
        get_config_dict : Get configuration dictionary
        """
        config = self.get_config_dict()
        with open(filename, "w") as f:
            json.dump(config, f, indent=2)
