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
Symbolic Dynamical System - Multi-Backend Framework
====================================================

This module provides the core abstraction for defining and simulating dynamical systems
using symbolic mathematics with automatic multi-backend code generation.

Overview
--------
The SymbolicDynamicalSystem class allows users to define continuous-time dynamical systems
symbolically using SymPy, then automatically generates efficient numerical implementations
for NumPy, PyTorch, and JAX backends. The framework handles:

- Symbolic system definition with parameter substitution
- Automatic code generation and caching for multiple backends
- Forward dynamics evaluation: dx/dt = f(x, u)
- System linearization: A = ∂f/∂x, B = ∂f/∂u
- Output functions: y = h(x) with linearization C = ∂h/∂x
- Equilibrium management and verification
- Performance tracking and optimization
- Device management (CPU/GPU) for accelerated backends

Architecture
-----------
The class uses a composition-based architecture, delegating specialized tasks to
focused components:

- **BackendManager**: Handles backend detection, conversion, and device placement
- **SymbolicValidator**: Validates system definitions for correctness
- **CodeGenerator**: Generates and caches numerical functions from symbolic expressions
- **DynamicsEvaluator**: Evaluates forward dynamics across backends
- **LinearizationEngine**: Computes linearized dynamics (A, B matrices)
- **ObservationEngine**: Evaluates output functions and their linearizations
- **EquilibriumHandler**: Manages multiple equilibrium points

Key Features
-----------
1. **Multi-Backend Support**: Write once, run on NumPy, PyTorch, or JAX
2. **Automatic Code Generation**: Symbolic → optimized numerical code
3. **Higher-Order Systems**: Native support for 2nd, 3rd, ... nth-order systems
4. **Zero Copy Overhead**: Direct backend operations, no unnecessary conversions
5. **JIT Compilation**: Automatic JIT for JAX, optional for PyTorch
6. **GPU Acceleration**: Seamless CPU/GPU execution for PyTorch and JAX
7. **Verification Tools**: Compare symbolic Jacobians against autodiff
8. **Performance Tracking**: Built-in timing and call counting

Mathematical Notation
--------------------
- **x**: State vector (nx × 1)
- **u**: Control vector (nu × 1)
- **y**: Output vector (ny × 1)
- **f(x, u)**: Dynamics function (returns dx/dt)
- **h(x)**: Output function
- **A = ∂f/∂x**: State Jacobian (nx × nx)
- **B = ∂f/∂u**: Control Jacobian (nx × nu)
- **C = ∂h/∂x**: Output Jacobian (ny × nx)

System Order and Formulations
-----------------------------
Systems can be defined in two equivalent ways:

**First-Order State-Space Form (order=1):**
For a physical system with n generalized coordinates q:
- State: x = [q, q̇, q̈, ..., q^(n-1)] (nx states)
- Dynamics: dx/dt = f(x, u) returns ALL derivatives [q̇, q̈, ..., q^(n)]
- Set: self.order = 1
- Example: Pendulum with x = [θ, θ̇], f returns [θ̇, θ̈]

**Higher-Order Form (order=n):**
For an nth-order system:
- State: x = [q, q̇, q̈, ..., q^(n-1)] (nx = n * nq states)
- Dynamics: f(x, u) returns ONLY highest derivative q^(n)
- Set: self.order = n
- Example: Pendulum with x = [θ, θ̇], f returns only θ̈, order=2

Both are equivalent! The framework automatically handles state-space construction
for higher-order systems during linearization.

Notes
-----
- State variables must be SymPy Symbol objects
- Parameters must use Symbol keys (not strings): {m: 1.0} not {'m': 1.0}
- System order must divide state dimension evenly: nx % order == 0
- Output functions h(x) should not depend on control u
- All backends produce numerically equivalent results

See Also
--------
- BackendManager: Backend detection and conversion
- SymbolicValidator: System definition validation
- CodeGenerator: Symbolic to numerical code generation
- DynamicsEvaluator: Forward dynamics evaluation
- LinearizationEngine: Linearization computation
- ObservationEngine: Output function evaluation
- EquilibriumHandler: Equilibrium point management

Authors
-------
Gil Benezer

License
-------
MIT License
"""

import copy
from contextlib import contextmanager
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Union, Optional, Tuple, List, Dict
import json
import sympy as sp
import numpy as np

# necessary sub-object import
from src.systems.base.utils.equilibrium_handler import EquilibriumHandler
from src.systems.base.utils.backend_manager import BackendManager
from src.systems.base.utils.symbolic_validator import SymbolicValidator, ValidationError
from src.systems.base.utils.code_generator import CodeGenerator
from src.systems.base.utils.dynamics_evaluator import DynamicsEvaluator
from src.systems.base.utils.linearization_engine import LinearizationEngine
from src.systems.base.utils.observation_engine import ObservationEngine

if TYPE_CHECKING:
    import torch
    import jax.numpy as jnp

# Type alias for backend-agnostic arrays
ArrayLike = Union[np.ndarray, "torch.Tensor", "jnp.ndarray"]


class SymbolicDynamicalSystem(ABC):
    """
    Abstract base class for symbolic dynamical systems with multi-backend execution.

    This class provides a framework for defining continuous-time dynamical systems
    symbolically and automatically generating efficient numerical implementations
    for NumPy, PyTorch, and JAX backends.

    The system is defined by:
        dx/dt = f(x, u)  - Dynamics (required)
        y = h(x)         - Output (optional, defaults to identity)

    Where:
        x ∈ ℝⁿˣ is the state vector
        u ∈ ℝⁿᵘ is the control input
        y ∈ ℝⁿʸ is the output vector

    Subclasses must implement the `define_system()` method to specify the
    symbolic dynamics and parameters.

    Attributes
    ----------
    state_vars : List[sp.Symbol]
        State variables as SymPy symbols
    control_vars : List[sp.Symbol]
        Control variables as SymPy symbols
    output_vars : List[sp.Symbol]
        Output variables (optional)
    parameters : Dict[sp.Symbol, float]
        System parameters (Symbol → numeric value)
    _f_sym : sp.Matrix
        Symbolic dynamics expression
    _h_sym : Optional[sp.Matrix]
        Symbolic output expression (None = identity)
    order : int
        System order (1 = first-order, 2 = second-order, etc.)
    backend : BackendManager
        Backend management component
    equilibria : EquilibriumHandler
        Equilibrium point management

    Examples
    --------
    >>> class LinearSystem(SymbolicDynamicalSystem):
    ...     def define_system(self, a=1.0):
    ...         x = sp.symbols('x', real=True)
    ...         u = sp.symbols('u', real=True)
    ...         a_sym = sp.symbols('a', real=True, positive=True)
    ...
    ...         self.state_vars = [x]
    ...         self.control_vars = [u]
    ...         self._f_sym = sp.Matrix([-a_sym * x + u])
    ...         self.parameters = {a_sym: a}
    ...         self.order = 1
    ...
    >>> system = LinearSystem(a=2.0)
    >>> x = np.array([1.0])
    >>> u = np.array([0.5])
    >>> dx = system(x, u)  # Evaluate dynamics
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the symbolic dynamical system.

        This constructor follows a template method pattern:
        1. Initialize empty symbolic containers
        2. Create specialized components for validation, code generation, etc.
        3. Call user-defined `define_system()` to populate symbolic expressions
        4. Validate the system definition
        5. Initialize remaining components that depend on validated system

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
        Subclasses should not override __init__. Instead, implement define_system()
        to specify the symbolic system definition.

        Examples
        --------
        >>> class MySystem(SymbolicDynamicalSystem):
        ...     def define_system(self, param=1.0):
        ...         # Define your system here
        ...         pass
        ...
        >>> system = MySystem(param=2.0)  # Calls __init__ → define_system
        """

        # ====================================================================
        # Symbolic Definition Containers
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
        """Symbolic dynamics: dx/dt = f(x, u) or q^(n) = f(x, u) for nth-order"""

        self._h_sym: Optional[sp.Matrix] = None
        """Symbolic output: y = h(x). If None, output is identity (y = x)"""

        self.order: int = 1
        """System order: 1 (first-order), 2 (second-order), etc. (defaults to first-order)"""

        # ====================================================================
        # Component Initialization
        # ====================================================================

        # COMPOSITION: Delegate backend management
        self.backend = BackendManager(default_backend="numpy", default_device="cpu")
        """Backend manager handles detection, conversion, and device placement"""

        # COMPOSITION: Delegate validation
        self._validator: Optional[SymbolicValidator] = None
        """Validator checks system definition for correctness"""

        # COMPOSITION: Delegate equilibrium management
        # Note: Initialized early because define_system might add equilibria
        self.equilibria = EquilibriumHandler(nx=0, nu=0)  # Will update dimensions after validation
        """Equilibrium handler manages multiple equilibrium points"""

        # COMPOSITION: Delegate code generation (initialized after validation)
        self._code_gen: Optional[CodeGenerator] = None
        """Code generator creates numerical functions from symbolic expressions"""

        # Initialization flag
        self._initialized: bool = False
        """Tracks whether system has been successfully initialized and validated"""

        # ====================================================================
        # Template Method Pattern: Define → Validate → Initialize
        # ====================================================================

        # Step 1: Call user-defined system definition
        self.define_system(*args, **kwargs)

        # Step 2: Create validator and validate system definition
        self._validator = SymbolicValidator(self)
        try:
            validation_result = self._validator.validate(raise_on_error=True)
        except ValidationError as e:
            # Re-raise with context about which system failed
            raise ValidationError(
                f"Validation failed for {self.__class__.__name__}:\n{str(e)}"
            ) from e

        # Step 3: Update equilibrium handler dimensions (now that we know nx, nu)
        self.equilibria.nx = self.nx  # Property setter
        self.equilibria.nu = self.nu  # Property setter

        # Step 3: Mark as initialized (validation passed)
        self._initialized = True

        # Step 4: Initialize components that depend on validated system
        self._code_gen = CodeGenerator(self)
        """Code generator (initialized after validation)"""

        self._dynamics = DynamicsEvaluator(self, self._code_gen, self.backend)
        """Dynamics evaluator handles forward dynamics evaluation"""

        self._linearization = LinearizationEngine(self, self._code_gen, self.backend)
        """Linearization engine computes A, B matrices"""

        self._observation = ObservationEngine(self, self._code_gen, self.backend)
        """Observation engine evaluates output functions and C matrix"""

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

    @abstractmethod
    def define_system(self, *args, **kwargs):
        """
        Define the symbolic system (must be implemented by subclasses).

        This method must populate the following attributes:

        Required Attributes
        ------------------
        - self.state_vars : List[sp.Symbol]
            State variables (e.g., [x, y, theta])
        - self.control_vars : List[sp.Symbol]
            Control variables (e.g., [u1, u2])
        - self._f_sym : sp.Matrix
            Symbolic dynamics (column vector)
        - self.parameters : Dict[sp.Symbol, float]
            Parameter values with Symbol keys (NOT strings!)

        Optional Attributes
        ------------------
        - self.output_vars : List[sp.Symbol]
            Output variable names (optional)
        - self._h_sym : sp.Matrix
            Symbolic output function (None = identity)
        - self.order : int
            System order (default: 1)

        Parameters
        ----------
        *args : tuple
            System-specific positional arguments (e.g., mass, length)
        **kwargs : dict
            System-specific keyword arguments

        Raises
        ------
        ValidationError
            If the defined system is invalid (checked after this method returns)

        Notes
        -----
        **CRITICAL**: self.parameters must use SymPy Symbol objects as keys!

        Correct:   {m: 1.0, l: 0.5}
        Incorrect: {'m': 1.0, 'l': 0.5}  # Strings won't work!

        **System Order - Two Equivalent Formulations:**

        You can define systems in two ways:

        1. **First-Order State-Space Form (order=1):**
           - State: x = [q, q̇] for a 2nd-order physical system
           - _f_sym returns ALL derivatives: [q̇, q̈]
           - Set: self.order = 1
           - Example:
             ```python
             self.state_vars = [theta, theta_dot]
             self._f_sym = sp.Matrix([
                 theta_dot,  # dθ/dt = θ̇
                 -k*theta - c*theta_dot + u  # dθ̇/dt = θ̈
             ])
             self.order = 1  # First-order state-space
             ```

        2. **Higher-Order Form (order=n):**
           - State: x = [q, q̇] for a 2nd-order physical system
           - _f_sym returns ONLY highest derivative: q̈
           - Set: self.order = 2
           - Example:
             ```python
             self.state_vars = [theta, theta_dot]
             self._f_sym = sp.Matrix([
                 -k*theta - c*theta_dot + u  # Only θ̈
             ])
             self.order = 2  # Second-order
             ```

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
        First-order system:
        >>> def define_system(self, a=1.0):
        ...     x = sp.symbols('x')
        ...     u = sp.symbols('u')
        ...     a_sym = sp.symbols('a', real=True, positive=True)
        ...
        ...     self.state_vars = [x]
        ...     self.control_vars = [u]
        ...     self._f_sym = sp.Matrix([-a_sym * x + u])
        ...     self.parameters = {a_sym: a}
        ...     self.order = 1

        Second-order (state-space form):
        >>> def define_system(self, m=1.0, k=10.0, c=0.5):
        ...     q, q_dot = sp.symbols('q q_dot')
        ...     u = sp.symbols('u')
        ...     m_sym, k_sym, c_sym = sp.symbols('m k c', positive=True)
        ...
        ...     # Return both derivatives explicitly
        ...     self.state_vars = [q, q_dot]
        ...     self.control_vars = [u]
        ...     self._f_sym = sp.Matrix([
        ...         q_dot,  # dq/dt
        ...         (-k_sym*q - c_sym*q_dot + u)/m_sym  # dq̇/dt = q̈
        ...     ])
        ...     self.parameters = {m_sym: m, k_sym: k, c_sym: c}
        ...     self.order = 1  # First-order state-space

        Second-order (higher-order form):
        >>> def define_system(self, m=1.0, k=10.0, c=0.5):
        ...     q, q_dot = sp.symbols('q q_dot')
        ...     u = sp.symbols('u')
        ...     m_sym, k_sym, c_sym = sp.symbols('m k c', positive=True)
        ...
        ...     # Return only acceleration
        ...     q_ddot = (-k_sym*q - c_sym*q_dot + u)/m_sym
        ...
        ...     self.state_vars = [q, q_dot]
        ...     self.control_vars = [u]
        ...     self._f_sym = sp.Matrix([q_ddot])  # Only highest derivative
        ...     self.parameters = {m_sym: m, k_sym: k, c_sym: c}
        ...     self.order = 2  # Second-order form
        """
        pass

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
    def _default_backend(self) -> str:
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
    def _default_backend(self, value: str):
        """
        Set default backend (backward compatibility property).

        Parameters
        ----------
        value : str
            Backend name ('numpy', 'torch', or 'jax')
        """
        self.backend.set_default(value)

    @property
    def _preferred_device(self) -> str:
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
    def _preferred_device(self, value: str):
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

    def set_default_backend(self, backend: str, device: Optional[str] = None):
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
        SymbolicDynamicalSystem
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

    def to_device(self, device: str) -> "SymbolicDynamicalSystem":
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
        SymbolicDynamicalSystem
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

    def _clear_backend_cache(self, backend: str):
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
    def use_backend(self, backend: str, device: Optional[str] = None):
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
        SymbolicDynamicalSystem
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

    def get_backend_info(self) -> Dict[str, any]:
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
        # Get base info from backend manager
        info = self.backend.get_info()

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
        self, backends: Optional[List[str]] = None, verbose: bool = False, **kwargs
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
            Dictionary mapping backend → compilation time (seconds)
            Returns None for backends that failed to compile

        Examples
        --------
        >>> # Compile for all available backends
        >>> timings = system.compile(verbose=True)
        Compiling numpy... 0.05s
        Compiling torch... 0.12s
        Compiling jax... 0.89s
        
        >>> # Compile specific backends
        >>> timings = system.compile(backends=['numpy', 'torch'])
        >>> print(f"NumPy: {timings['numpy']:.3f}s")

        Notes
        -----
        Delegates to CodeGenerator.compile_all() for actual compilation.
        Returns only dynamics ('f') timing for backward compatibility.
        """
        # Extract just the 'f' timings for backward compatibility
        all_timings = self._code_gen.compile_all(backends=backends, verbose=verbose, **kwargs)
        return {backend: timings.get("f") for backend, timings in all_timings.items()}

    def reset_caches(self, backends: Optional[List[str]] = None):
        """
        Clear cached compiled functions for specified backends.

        Useful when:
        - Freeing memory (cached functions can be large)
        - Forcing recompilation (e.g., after parameter changes)
        - Debugging code generation issues

        Parameters
        ----------
        backends : Optional[List[str]]
            List of backends to reset (None = reset all backends)

        Examples
        --------
        >>> # Clear all caches
        >>> system.reset_caches()
        
        >>> # Clear only torch cache
        >>> system.reset_caches(['torch'])

        Notes
        -----
        After clearing caches, functions will be regenerated on next use.
        This delegates to CodeGenerator.reset_cache().
        """
        self._code_gen.reset_cache(backends)

    def warmup(
        self,
        backend: Optional[str] = None,
        test_point: Optional[Tuple[ArrayLike, ArrayLike]] = None,
    ) -> bool:
        """
        Warm up backend by compiling and running test evaluation.

        Useful for:
        - JIT compilation warmup (especially JAX)
        - Reducing first-call latency in production
        - Validating backend works before critical operations

        Parameters
        ----------
        backend : Optional[str]
            Backend to warm up (None = use default backend)
        test_point : Optional[Tuple[ArrayLike, ArrayLike]]
            Optional (x, u) test point. If None, uses origin equilibrium.

        Returns
        -------
        bool
            True if warmup successful, False if failed

        Examples
        --------
        >>> # Warm up JAX with JIT compilation
        >>> system.set_default_backend('jax', device='gpu:0')
        >>> success = system.warmup()
        Warming up jax backend...
        ✓ jax backend ready (test evaluation successful)
        
        >>> # Warm up with custom test point
        >>> x_test = np.array([0.1, 0.0])
        >>> u_test = np.array([0.0])
        >>> system.warmup(test_point=(x_test, u_test))

        Notes
        -----
        For JAX, the first call triggers JIT compilation which can be slow.
        This method warms up the JIT cache so subsequent calls are fast.
        """
        backend = backend or self._default_backend

        print(f"Warming up {backend} backend...")
        self._code_gen.generate_dynamics(backend)

        if self._h_sym is not None:
            self._code_gen.generate_output(backend)

        # Test evaluation
        if test_point is not None:
            x_test, u_test = test_point
        else:
            # Use equilibrium point
            x_test = self.equilibria.get_x(backend=backend)
            u_test = self.equilibria.get_u(backend=backend)

        # Run test evaluation
        try:
            dx = self.forward(x_test, u_test, backend=backend)
            print(f"✓ {backend} backend ready (test evaluation successful)")
            return True
        except Exception as e:
            print(f"✗ {backend} backend warmup failed: {e}")
            return False

    # ========================================================================
    # Copying and Cloning
    # ========================================================================

    def clone(self, backend: Optional[str] = None, deep: bool = True) -> "SymbolicDynamicalSystem":
        """
        Create a copy of the system, optionally changing backend.

        Useful for creating system variants or comparing backends without
        modifying the original system.

        Parameters
        ----------
        backend : Optional[str]
            New default backend for cloned system (None = keep same)
        deep : bool
            If True, deep copy (independent object)
            If False, shallow copy (shares some data with original)

        Returns
        -------
        SymbolicDynamicalSystem
            Cloned system

        Examples
        --------
        >>> # Create independent copy
        >>> system2 = system.clone()
        >>> system2.parameters[m] = 2.0  # Doesn't affect original
        
        >>> # Clone and switch to JAX
        >>> jax_system = system.clone(backend='jax')
        >>> jax_system._default_backend
        'jax'

        Notes
        -----
        Deep copy is recommended for most use cases to avoid unintended
        side effects. Shallow copy may share cached functions and could
        cause unexpected behavior.
        """
        if deep:
            cloned = copy.deepcopy(self)
        else:
            cloned = copy.copy(self)

        if backend is not None:
            cloned.set_default_backend(backend)

        return cloned

    # ========================================================================
    # Configuration Persistence
    # ========================================================================

    def save_config(self, filename: str):
        """
        Save system configuration to file.

        Saves all system parameters, dimensions, backend settings, and
        equilibrium points to a JSON or PyTorch file for later reconstruction.

        Parameters
        ----------
        filename : str
            Output filename. Must end with '.json' or '.pt'

        Raises
        ------
        ValueError
            If filename has unsupported extension

        Examples
        --------
        >>> system.save_config('pendulum_config.json')
        Configuration saved to pendulum_config.json

        >>> system.save_config('system.pt')  # PyTorch format
        Configuration saved to system.pt

        Notes
        -----
        This saves configuration only, not the symbolic definitions or
        compiled functions. To fully reconstruct a system, you need both
        the saved config and the class definition.
        """
        config = {
            "class_name": self.__class__.__name__,
            "parameters": {str(k): float(v) for k, v in self.parameters.items()},
            "order": self.order,
            "nx": self.nx,
            "nu": self.nu,
            "ny": self.ny,
            "default_backend": self._default_backend,
            "preferred_device": self._preferred_device,
            "equilibria": {
                name: {"x": eq["x"].tolist(), "u": eq["u"].tolist(), "metadata": eq["metadata"]}
                for name, eq in self.equilibria._equilibria.items()
            },
        }

        if filename.endswith(".json"):
            with open(filename, "w") as f:
                json.dump(config, f, indent=2)
        elif filename.endswith(".pt"):
            import torch

            torch.save(config, filename)
        else:
            raise ValueError(f"Unsupported file format: {filename}. Use .json or .pt")

        print(f"Configuration saved to {filename}")

    def get_config_dict(self) -> Dict:
        """
        Get system configuration as a dictionary.

        Returns configuration without saving to file. Useful for
        serialization, logging, or programmatic access.

        Returns
        -------
        dict
            Configuration dictionary with keys:
            - 'class_name': System class name
            - 'parameters': Parameter names and values
            - 'order': System order
            - 'nx', 'nu', 'ny': Dimensions
            - 'default_backend': Current backend
            - 'preferred_device': Current device

        Examples
        --------
        >>> config = system.get_config_dict()
        >>> print(config['class_name'])
        'SimplePendulum'
        >>> print(config['parameters'])
        {'m': 1.0, 'l': 0.5, 'g': 9.81}
        """
        return {
            "class_name": self.__class__.__name__,
            "parameters": {str(k): float(v) for k, v in self.parameters.items()},
            "order": self.order,
            "nx": self.nx,
            "nu": self.nu,
            "ny": self.ny,
            "default_backend": self._default_backend,
            "preferred_device": self._preferred_device,
            "equilibria": self.equilibria.list_names(),  # Add this!
            "default_equilibrium": self.equilibria._default,  # Add this!
        }

    # ========================================================================
    # Performance Statistics
    # ========================================================================

    def get_performance_stats(self) -> Dict[str, float]:
        """
        Get performance statistics from all system components.

        Collects timing and call count statistics from DynamicsEvaluator
        and LinearizationEngine components.

        Returns
        -------
        dict
            Statistics dictionary with keys:
            - 'forward_calls': Number of forward dynamics evaluations
            - 'forward_time': Total time in forward evaluation (seconds)
            - 'avg_forward_time': Average forward evaluation time
            - 'linearization_calls': Number of linearization computations
            - 'linearization_time': Total linearization time
            - 'avg_linearization_time': Average linearization time

        Examples
        --------
        >>> for _ in range(100):
        ...     dx = system(x, u)
        ...
        >>> stats = system.get_performance_stats()
        >>> print(f"Forward calls: {stats['forward_calls']}")
        >>> print(f"Avg time: {stats['avg_forward_time']:.6f}s")

        See Also
        --------
        reset_performance_stats : Reset all performance counters
        """
        # Get stats from components
        dynamics_stats = self._dynamics.get_stats()
        linearization_stats = self._linearization.get_stats()

        return {
            "forward_calls": dynamics_stats["calls"],
            "forward_time": dynamics_stats["total_time"],
            "avg_forward_time": dynamics_stats["avg_time"],
            "linearization_calls": linearization_stats["calls"],
            "linearization_time": linearization_stats["total_time"],
            "avg_linearization_time": linearization_stats["avg_time"],
        }

    def reset_performance_stats(self):
        """
        Reset all performance counters to zero.

        Clears timing and call count statistics in all components
        (DynamicsEvaluator and LinearizationEngine).

        Examples
        --------
        >>> system.reset_performance_stats()
        >>> stats = system.get_performance_stats()
        >>> stats['forward_calls']
        0
        """
        self._dynamics.reset_stats()
        self._linearization.reset_stats()

    # ========================================================================
    # Symbolic Utilities
    # ========================================================================

    def substitute_parameters(self, expr: Union[sp.Expr, sp.Matrix]) -> Union[sp.Expr, sp.Matrix]:
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

    def print_equations(self, simplify: bool = True):
        """
        Print symbolic equations in human-readable format.

        Displays the system's symbolic dynamics, output equations,
        state/control variables, and dimensions.

        Parameters
        ----------
        simplify : bool
            If True, simplify expressions before printing
            If False, print raw expressions

        Examples
        --------
        >>> system.print_equations()
        ======================================================================
        SimplePendulum
        ======================================================================
        State Variables: [theta, theta_dot]
        Control Variables: [u]
        System Order: 2
        Dimensions: nx=2, nu=1, ny=2

        Dynamics: dx/dt = f(x, u)
          dtheta/dt = theta_dot
          dtheta_dot/dt = -19.62*sin(theta) - 0.1*theta_dot + 4.0*u
        ======================================================================

        Notes
        -----
        Simplification can make complex expressions more readable but
        may take time for large systems.
        """
        print("=" * 70)
        print(f"{self.__class__.__name__}")
        print("=" * 70)
        print(f"State Variables: {self.state_vars}")
        print(f"Control Variables: {self.control_vars}")
        print(f"System Order: {self.order}")
        print(f"Dimensions: nx={self.nx}, nu={self.nu}, ny={self.ny}")

        print("\nDynamics: dx/dt = f(x, u)")
        for i, (var, expr) in enumerate(zip(self.state_vars, self._f_sym)):
            expr_sub = self.substitute_parameters(expr)
            if simplify:
                expr_sub = sp.simplify(expr_sub)
            print(f"  d{var}/dt = {expr_sub}")

        if self._h_sym is not None:
            print("\nOutput: y = h(x)")
            for i, expr in enumerate(self._h_sym):
                expr_sub = self.substitute_parameters(expr)
                if simplify:
                    expr_sub = sp.simplify(expr_sub)
                print(f"  y[{i}] = {expr_sub}")

        print("=" * 70)

    # ========================================================================
    # Forward Dynamics
    # ========================================================================

    def forward(self, x: ArrayLike, u: Optional[ArrayLike] = None, backend: Optional[str] = None) -> ArrayLike:
        """
        Evaluate continuous-time dynamics: dx/dt = f(x, u) or dx/dt = f(x) for autonomous.

        Computes the state derivative at the given state and control.
        Automatically detects backend from input types or uses specified backend.
        Supports both controlled (nu > 0) and autonomous (nu = 0) systems.

        Parameters
        ----------
        x : ArrayLike
            State vector (nx,) or batched states (batch, nx)
        u : Optional[ArrayLike]
            Control vector (nu,) or batched controls (batch, nu)
            For autonomous systems (nu=0), u can be None or omitted
        backend : Optional[str]
            Backend selection:
            - None: Auto-detect from input type (default)
            - 'numpy', 'torch', 'jax': Force specific backend
            - 'default': Use configured default backend

        Returns
        -------
        ArrayLike
            State derivative dx/dt (same type as backend)
            - First-order: shape (nx,) or (batch, nx)
            - nth-order: shape (nq,) or (batch, nq) [highest derivative only]

        Raises
        ------
        ValueError
            If input dimensions don't match system dimensions, or if u is None
            for a non-autonomous system

        Examples
        --------
        Controlled system:
        >>> x = np.array([1.0, 0.0])
        >>> u = np.array([0.5])
        >>> dx = system.forward(x, u)  # Returns NumPy array

        Autonomous system:
        >>> x = np.array([1.0, 0.0])
        >>> dx = system.forward(x)  # u=None for autonomous
        >>> # or explicitly:
        >>> dx = system.forward(x, u=None)

        Force PyTorch backend (converts input):
        >>> dx = system.forward(x, u, backend='torch')  # Returns torch.Tensor

        Batched evaluation:
        >>> x_batch = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        >>> u_batch = np.array([[0.5], [0.5], [0.5]])
        >>> dx_batch = system.forward(x_batch, u_batch)  # Shape: (3, nx)

        Notes
        -----
        This method delegates to DynamicsEvaluator.evaluate() for actual computation.
        Backend conversion happens automatically if input type doesn't match target backend.
        For autonomous systems, passing u=None is allowed and recommended.
        """
        return self._dynamics.evaluate(x, u, backend)

    def __call__(self, x: ArrayLike, u: Optional[ArrayLike] = None, backend: Optional[str] = None) -> ArrayLike:
        """
        Make system callable: system(x, u) or system(x) for autonomous evaluates dynamics.

        This is the primary user interface for forward dynamics evaluation.
        Equivalent to calling system.forward(x, u, backend).

        Parameters
        ----------
        x : ArrayLike
            State vector
        u : Optional[ArrayLike]
            Control vector (None for autonomous systems)
        backend : Optional[str]
            Backend selection (None = auto-detect)

        Returns
        -------
        ArrayLike
            State derivative dx/dt

        Examples
        --------
        Controlled system:
        >>> dx = system(x, u)  # Pythonic interface
        
        Autonomous system:
        >>> dx = system(x)  # No control input
        
        # Equivalent to:
        >>> dx = system.forward(x, u)

        Notes
        -----
        This makes the system object behave like a function, which is
        intuitive for users familiar with functional programming or
        mathematical notation. Supports both controlled and autonomous systems.
        """
        return self.forward(x, u, backend)

    # ========================================================================
    # Linearization
    # ========================================================================

    def linearized_dynamics(
        self, 
        x: Union[ArrayLike, str], 
        u: Optional[ArrayLike] = None,  # Only ArrayLike, not string
        backend: Optional[str] = None
    ) -> Tuple[ArrayLike, ArrayLike]:
        """
        Compute numerical linearization of dynamics: A = ∂f/∂x, B = ∂f/∂u.

        Evaluates the Jacobian matrices at the given state and control point.
        For higher-order systems, returns the full state-space representation.
        For autonomous systems (nu=0), u can be None and B will be empty (nx, 0).

        Parameters
        ----------
        x : Union[ArrayLike, str]
            State to linearize at, OR equilibrium name
        u : Optional[ArrayLike]
            Control to linearize at
            Ignored if x is string (equilibrium name)
        backend : Optional[str]
            Backend selection

        Returns
        -------
        Tuple[ArrayLike, ArrayLike]
            (A, B) matrices where:
            - A: State Jacobian ∂f/∂x, shape (nx, nx) or (batch, nx, nx)
            - B: Control Jacobian ∂f/∂u, shape (nx, nu) or (batch, nx, nu)
            For autonomous systems: B has shape (nx, 0) - empty matrix

        Examples
        --------
        Controlled system:
        >>> x = np.array([0.1, 0.0])
        >>> u = np.array([0.0])
        >>> A, B = system.linearized_dynamics(x, u)
        >>> A.shape
        (2, 2)
        >>> B.shape
        (2, 1)

        Autonomous system:
        >>> x = np.array([0.1, 0.0])
        >>> A, B = system.linearized_dynamics(x)  # u=None
        >>> A.shape
        (2, 2)
        >>> B.shape
        (2, 0)  # Empty B matrix

        Batched linearization:
        >>> x_batch = np.array([[0.1, 0.0], [0.2, 0.0]])
        >>> u_batch = np.array([[0.0], [0.0]])
        >>> A, B = system.linearized_dynamics(x_batch, u_batch)
        >>> A.shape
        (2, 2, 2)  # (batch, nx, nx)

        >>> # Direct state/control
        >>> A, B = system.linearized_dynamics(x, u)
        
        >>> # Named equilibrium
        >>> A, B = system.linearized_dynamics('inverted')
        >>> A, B = system.linearized_dynamics('inverted', backend='torch')

        Notes
        -----
        For second-order systems with state x = [q, q̇]:
        - Returns full state-space form with ∂q̇/∂q relationships
        - A includes both kinematics (q̇) and dynamics (q̈)

        Uses cached Jacobian functions if available, otherwise computes
        symbolically. Delegates to LinearizationEngine.compute_dynamics().
        """
        # If x is a string, treat as equilibrium name
        if isinstance(x, str):
            equilibrium_name = x
            backend = backend or self._default_backend
            x, u = self.equilibria.get_both(equilibrium_name, backend)
        
        return self._linearization.compute_dynamics(x, u, backend)

    def linearized_dynamics_symbolic(
        self, 
        x_eq: Optional[Union[sp.Matrix, str]] = None, 
        u_eq: Optional[sp.Matrix] = None  # Only sp.Matrix, not string
    ) -> Tuple[sp.Matrix, sp.Matrix]:
        """
        Compute symbolic linearization: A = ∂f/∂x, B = ∂f/∂u.

        Returns symbolic matrices that can be used for analytical analysis,
        controller design, or verification.

        Parameters
        ----------
        x_eq : Optional[Union[sp.Matrix, str]]
            Equilibrium state (symbolic matrix), OR equilibrium name (string).
            If None, uses origin (zeros).
            If string, retrieves equilibrium from handler (u_eq is ignored).
        u_eq : Optional[sp.Matrix]
            Equilibrium control (symbolic matrix). If None, uses zeros.
            Ignored if x_eq is a string (equilibrium name).

        Returns
        -------
        Tuple[sp.Matrix, sp.Matrix]
            (A, B) symbolic matrices with parameters and equilibrium substituted

        Examples
        --------
        >>> # Linearize at origin
        >>> A_sym, B_sym = system.linearized_dynamics_symbolic()
        >>> print(A_sym)
        Matrix([[0, 1], [-10.0, -0.5]])

        >>> # Linearize at custom symbolic point
        >>> x_eq = sp.Matrix([sp.pi, 0])  # Upright pendulum
        >>> u_eq = sp.Matrix([0])
        >>> A_sym, B_sym = system.linearized_dynamics_symbolic(x_eq, u_eq)
        
        >>> # Linearize at named equilibrium (u_eq ignored)
        >>> A_sym, B_sym = system.linearized_dynamics_symbolic('inverted')

        Convert to NumPy:
        >>> A_np = np.array(A_sym, dtype=float)

        Notes
        -----
        For higher-order systems, automatically constructs the full
        state-space representation. Delegates to LinearizationEngine.compute_symbolic().
        """
        # If x_eq is a string, get equilibrium from handler
        if isinstance(x_eq, str):
            equilibrium_name = x_eq
            x_np, u_np = self.equilibria.get_both(equilibrium_name, backend='numpy')
            # Convert NumPy arrays to SymPy matrices
            x_eq = sp.Matrix(x_np.tolist())
            u_eq = sp.Matrix(u_np.tolist())
        
        return self._linearization.compute_symbolic(x_eq, u_eq)

    def verify_jacobians(
        self, x: ArrayLike, u: Optional[ArrayLike] = None, tol: float = 1e-3, backend: str = "torch"
    ) -> Dict[str, Union[bool, float]]:
        """
        Verify symbolic Jacobians against automatic differentiation.

        Compares analytically-derived Jacobians (from SymPy) against
        numerically-computed Jacobians (from PyTorch/JAX autodiff) to
        verify correctness of symbolic derivations.

        Parameters
        ----------
        x : ArrayLike
            State at which to verify
        u : Optional[ArrayLike]
            Control at which to verify (None for autonomous systems)
        tol : float
            Tolerance for considering Jacobians equal (default: 1e-3)
        backend : str
            Backend for autodiff ('torch' or 'jax', not 'numpy')

        Returns
        -------
        dict
            Verification results with keys:
            - 'A_match': bool, True if A Jacobian matches within tolerance
            - 'B_match': bool, True if B Jacobian matches within tolerance
            - 'A_error': float, Maximum absolute error in A
            - 'B_error': float, Maximum absolute error in B

        Raises
        ------
        ValueError
            If backend is 'numpy' (doesn't support autodiff)
        RuntimeError
            If specified backend is not available

        Examples
        --------
        Controlled system:
        >>> x = torch.tensor([0.1, 0.0])
        >>> u = torch.tensor([0.0])
        >>> results = system.verify_jacobians(x, u, backend='torch', tol=1e-6)
        >>>
        >>> if results['A_match'] and results['B_match']:
        ...     print("✓ Jacobians verified!")
        ... else:
        ...     print(f"✗ A error: {results['A_error']:.2e}")
        ...     print(f"✗ B error: {results['B_error']:.2e}")

        Autonomous system:
        >>> x = torch.tensor([0.1, 0.0])
        >>> results = system.verify_jacobians(x, backend='torch')  # u=None
        >>> # B_match will be True trivially (empty matrix comparison)

        Notes
        -----
        This is a powerful debugging tool for ensuring symbolic derivations
        are correct. Small errors (< 1e-6) are usually due to numerical precision.
        Large errors indicate bugs in symbolic Jacobian computation.

        Delegates to LinearizationEngine.verify_jacobians().
        """
        return self._linearization.verify_jacobians(x, u, backend, tol)

    # ========================================================================
    # Output Functions
    # ========================================================================

    def h(self, x: ArrayLike, backend: Optional[str] = None) -> ArrayLike:
        """
        Evaluate output equation: y = h(x).

        Computes the system output at the given state. If no custom output
        function is defined, returns the full state (identity map).

        Parameters
        ----------
        x : ArrayLike
            State vector (nx,) or batched states (batch, nx)
        backend : Optional[str]
            Backend selection (None = auto-detect)

        Returns
        -------
        ArrayLike
            Output vector y, shape (ny,) or (batch, ny)

        Examples
        --------
        >>> # System with identity output (h(x) = x)
        >>> y = system.h(x)
        >>> np.allclose(y, x)
        True

        >>> # System with custom output
        >>> # h(x) = x1^2 + x2^2 (energy)
        >>> x = np.array([1.0, 2.0])
        >>> y = system.h(x)
        >>> y
        array([5.0])

        Batched evaluation:
        >>> x_batch = np.array([[1.0, 2.0], [2.0, 3.0]])
        >>> y_batch = system.h(x_batch)
        >>> y_batch.shape
        (2, 1)

        Notes
        -----
        Output functions must only depend on states, not controls.
        Delegates to ObservationEngine.evaluate().
        """
        return self._observation.evaluate(x, backend)

    def linearized_observation(self, x: ArrayLike, backend: Optional[str] = None) -> ArrayLike:
        """
        Compute linearized observation matrix: C = ∂h/∂x.

        Evaluates the output Jacobian at the given state. If no custom
        output function is defined, returns identity matrix.

        Parameters
        ----------
        x : ArrayLike
            State at which to linearize (nx,) or (batch, nx)
        backend : Optional[str]
            Backend selection (None = auto-detect)

        Returns
        -------
        ArrayLike
            C matrix, shape (ny, nx) or (batch, ny, nx)

        Examples
        --------
        >>> # For custom output h(x) = [x1, x1^2 + x2^2]
        >>> x = np.array([1.0, 2.0])
        >>> C = system.linearized_observation(x)
        >>> C
        array([[1., 0.],
               [2., 4.]])  # C = [[1, 0], [2*x1, 2*x2]]

        Identity output:
        >>> # No custom output → C = I
        >>> C = system.linearized_observation(x)
        >>> np.allclose(C, np.eye(system.nx))
        True

        Notes
        -----
        Delegates to ObservationEngine.compute_jacobian().
        Uses cached Jacobian functions when available.
        """
        return self._observation.compute_jacobian(x, backend)

    def linearized_observation_symbolic(self, x_eq: Optional[sp.Matrix] = None) -> sp.Matrix:
        """
        Compute symbolic observation Jacobian: C = ∂h/∂x.

        Returns symbolic matrix for analytical analysis or verification.

        Parameters
        ----------
        x_eq : Optional[sp.Matrix]
            Equilibrium state (symbolic). If None, uses zeros.

        Returns
        -------
        sp.Matrix
            Symbolic C matrix with parameters and equilibrium substituted

        Examples
        --------
        >>> C_sym = system.linearized_observation_symbolic()
        >>> print(C_sym)
        Matrix([[1, 0], [0, 4]])

        >>> # At specific point
        >>> x_eq = sp.Matrix([1, 2])
        >>> C_sym = system.linearized_observation_symbolic(x_eq)

        Notes
        -----
        For identity output (no custom h), returns identity matrix.
        Delegates to ObservationEngine.compute_symbolic().
        """
        return self._observation.compute_symbolic(x_eq)

    # ========================================================================
    # Equilibrium Management
    # ========================================================================

    def _verify_equilibrium_numpy(self, x_eq: np.ndarray, u_eq: np.ndarray) -> np.ndarray:
        """
        Verify equilibrium using NumPy backend (internal helper).

        Evaluates f(x_eq, u_eq) to check if it's near zero (equilibrium condition).

        Parameters
        ----------
        x_eq : np.ndarray
            Candidate equilibrium state
        u_eq : np.ndarray
            Candidate equilibrium control

        Returns
        -------
        np.ndarray
            Dynamics at proposed equilibrium (should be ≈ 0)

        Notes
        -----
        This is an internal method used by add_equilibrium() for verification.
        Uses NumPy backend for consistency regardless of default backend.
        """
        return self.forward(x_eq, u_eq, backend="numpy")

    def add_equilibrium(
        self,
        name: str,
        x_eq: np.ndarray,
        u_eq: np.ndarray,
        verify: bool = True,
        tol: float = 1e-6,
        **metadata,
    ):
        """
        Add an equilibrium point with optional verification.

        Equilibrium points are states where dx/dt = f(x_eq, u_eq) ≈ 0.
        The system can have multiple equilibria (e.g., pendulum upright vs downward).

        Parameters
        ----------
        name : str
            Unique name for this equilibrium (e.g., 'origin', 'upright', 'inverted')
        x_eq : np.ndarray
            Equilibrium state (nx,)
        u_eq : np.ndarray
            Equilibrium control (nu,)
        verify : bool
            If True, verify that f(x_eq, u_eq) ≈ 0 (default: True)
        tol : float
            Tolerance for verification: ||f(x_eq, u_eq)|| < tol
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
        Verification uses the NumPy backend for consistency. If verification
        fails, a warning is issued but the equilibrium is still added.

        Delegates to EquilibriumHandler.add() for storage and management.
        """
        verify_fn = self._verify_equilibrium_numpy if verify else None
        self.equilibria.add(name, x_eq, u_eq, verify_fn=verify_fn, tol=tol, **metadata)

    def set_default_equilibrium(self, name: str) -> "SymbolicDynamicalSystem":
        """
        Set default equilibrium for get operations without name.
        
        Parameters
        ----------
        name : str
            Name of equilibrium to use as default
        
        Returns
        -------
        SymbolicDynamicalSystem
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
        self, 
        name: Optional[str] = None, 
        backend: Optional[str] = None
    ) -> Tuple[ArrayLike, ArrayLike]:
        """
        Get equilibrium state and control in specified backend,
        or default backend if not specified
        
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
        """
        backend = backend or self._default_backend
        return self.equilibria.get_both(name, backend)

    def list_equilibria(self) -> List[str]:
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

    def get_equilibrium_metadata(self, name: Optional[str] = None) -> Dict:
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

    def remove_equilibrium(self, name: str):
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
        if name == 'origin':
            raise ValueError("Cannot remove origin equilibrium")
        
        if name not in self.equilibria._equilibria:
            raise ValueError(f"Unknown equilibrium '{name}'")
        
        del self.equilibria._equilibria[name]
        
        # Reset default if we removed it
        if self.equilibria._default == name:
            self.equilibria._default = 'origin'

    # Convenience: Quick access to control design
    @property
    def control(self):
        """
        Get control designer for this system (lazy initialization).

        Provides access to control design tools like LQR, MPC, etc.
        Created on first access and cached for subsequent use.

        Returns
        -------
        ControlDesigner
            Control design interface for this system

        Examples
        --------
        >>> # Design LQR controller
        >>> K = system.control.lqr(Q, R)
        >>>
        >>> # Design MPC controller
        >>> controller = system.control.mpc(horizon=10)

        Notes
        -----
        Requires the control module to be available. The ControlDesigner
        is created lazily on first access to avoid unnecessary initialization
        if control design is not needed.
        """
        if not hasattr(self, "_control_designer"):
            from src.control.control_designer import ControlDesigner

            self._control_designer = ControlDesigner(self)
        return self._control_designer


# New name (preferred)
ContinuousSymbolicSystem = SymbolicDynamicalSystem