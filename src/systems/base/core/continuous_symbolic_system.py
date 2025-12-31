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
Continuous Symbolic System - Symbolic Continuous-Time Dynamical Systems
========================================================================

This module provides the concrete implementation for symbolic continuous-time
dynamical systems, combining the symbolic machinery from SymbolicSystemBase
with the continuous-time interface from ContinuousSystemBase.

This is the NEW refactored implementation that replaces the old
SymbolicDynamicalSystem class by using multiple inheritance to eliminate
the ~1,800 lines of code duplication.

Architecture
-----------
```
    SymbolicSystemBase          ContinuousSystemBase
    (symbolic machinery)        (continuous interface)
            |                            |
            +-------------+--------------+
                          |
              ContinuousSymbolicSystem
              (combines both via multiple inheritance)
```

Key Features
------------
- Multi-backend execution (NumPy, PyTorch, JAX)
- Automatic code generation from symbolic expressions
- Integration via IntegratorFactory (flexible solver selection)
- Linearization with symbolic and numerical Jacobians
- Output function evaluation and linearization
- Equilibrium verification for continuous systems
- Performance tracking and statistics

What This Class Provides
------------------------
- Forward dynamics: dx/dt = f(x, u, t)
- Numerical integration: Uses IntegratorFactory for flexible solver selection
- Linearization: A = ∂f/∂x, B = ∂f/∂u
- Output evaluation: y = h(x)
- Output linearization: C = ∂h/∂x
- Equilibrium verification: ||f(x_eq, u_eq)|| < tol

What's Inherited from SymbolicSystemBase
----------------------------------------
- Symbolic variable management (state_vars, control_vars, parameters)
- Code generation and compilation
- Backend configuration and management
- Equilibrium storage and retrieval
- Configuration persistence
- Performance statistics base

Usage Example
-------------
```python
class Pendulum(ContinuousSymbolicSystem):
    def define_system(self, m=1.0, l=0.5, g=9.81, b=0.1):
        '''Define simple pendulum with damping.'''
        theta, omega = sp.symbols('theta omega', real=True)
        u = sp.symbols('u', real=True)
        m_sym, l_sym, g_sym, b_sym = sp.symbols('m l g b', positive=True)
        
        self.state_vars = [theta, omega]
        self.control_vars = [u]
        self._f_sym = sp.Matrix([
            omega,
            -(g_sym/l_sym)*sp.sin(theta) - (b_sym/(m_sym*l_sym**2))*omega + u/(m_sym*l_sym**2)
        ])
        self.parameters = {m_sym: m, l_sym: l, g_sym: g, b_sym: b}
        self.order = 1

# Create and use
system = Pendulum(m=0.5, l=0.3)

# Evaluate dynamics
x = np.array([0.1, 0.0])
u = np.array([0.0])
dx = system(x, u)  # Returns dx/dt

# Integrate
result = system.integrate(
    x0=x,
    u=None,  # Zero control
    t_span=(0.0, 5.0),
    method='RK45'
)

# Linearize at origin
A, B = system.linearize(np.zeros(2), np.zeros(1))
```

See Also
--------
- SymbolicSystemBase: Base symbolic machinery
- ContinuousSystemBase: Continuous-time interface
- DynamicsEvaluator: Forward dynamics evaluation
- LinearizationEngine: Linearization computation
- ObservationEngine: Output function evaluation
- IntegratorFactory: Numerical integration

Authors
-------
Gil Benezer

License
-------
AGPL-3.0
"""

import inspect
from typing import TYPE_CHECKING, Callable, Dict, Optional, Tuple, Union

import numpy as np
import sympy as sp

from src.systems.base.core.continuous_system_base import ContinuousSystemBase
from src.systems.base.core.symbolic_system_base import SymbolicSystemBase
from src.systems.base.numerical_integration.integrator_factory import IntegratorFactory
from src.systems.base.utils.dynamics_evaluator import DynamicsEvaluator
from src.systems.base.utils.linearization_engine import LinearizationEngine
from src.systems.base.utils.observation_engine import ObservationEngine

# Type imports
from src.types.core import (
    ArrayLike,
    ControlInput,
    ControlVector,
    ScalarLike,
    StateVector,
)
from src.types.backends import Backend, IntegrationMethod
from src.types.linearization import LinearizationResult
from src.types.trajectories import IntegrationResult, SimulationResult, TimePoints, TimeSpan

if TYPE_CHECKING:
    import jax.numpy as jnp
    import torch


class ContinuousSymbolicSystem(SymbolicSystemBase, ContinuousSystemBase):
    """
    Concrete symbolic continuous-time dynamical system.

    Combines symbolic machinery from SymbolicSystemBase with the continuous-time
    interface from ContinuousSystemBase to provide a complete implementation for
    symbolic continuous-time systems.

    This class represents systems of the form:
        dx/dt = f(x, u, t)
        y = h(x)

    where:
        x ∈ ℝⁿˣ: State vector
        u ∈ ℝⁿᵘ: Control input
        y ∈ ℝⁿʸ: Output vector
        t ∈ ℝ: Time

    Users subclass and implement define_system() to specify symbolic dynamics.

    Examples
    --------
    Define a linear oscillator:

    >>> class Oscillator(ContinuousSymbolicSystem):
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
    >>> system = Oscillator(k=2.0, c=0.5)
    >>> x = np.array([0.1, 0.0])
    >>> dx = system(x, u=None)  # Evaluate dynamics
    >>> 
    >>> # Integrate
    >>> result = system.integrate(x, t_span=(0, 10), method='RK45')
    >>> 
    >>> # Linearize
    >>> A, B = system.linearize(np.zeros(2), np.zeros(1))
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize continuous symbolic system.

        The initialization follows cooperative multiple inheritance:
        1. SymbolicSystemBase.__init__() handles symbolic setup
        2. ContinuousSystemBase properties are available
        3. Time-domain-specific evaluators are initialized

        Parameters
        ----------
        *args : tuple
            Positional arguments passed to define_system()
        **kwargs : dict
            Keyword arguments passed to define_system()

        Notes
        -----
        Order matters for multiple inheritance - super().__init__() in
        SymbolicSystemBase ensures both bases are properly initialized.
        """
        # Initialize both bases via super() in SymbolicSystemBase
        # This calls: SymbolicSystemBase.__init__() → super().__init__() → ContinuousSystemBase
        super().__init__(*args, **kwargs)

        # Initialize continuous-time-specific evaluators
        # These require the symbolic system to be validated (done in super().__init__)
        
        self._dynamics = DynamicsEvaluator(self, self._code_gen, self.backend)
        """Evaluates forward dynamics: dx/dt = f(x, u)"""

        self._linearization = LinearizationEngine(self, self._code_gen, self.backend)
        """Computes linearization: A = ∂f/∂x, B = ∂f/∂u"""

        self._observation = ObservationEngine(self, self._code_gen, self.backend)
        """Evaluates output: y = h(x) and C = ∂h/∂x"""

    # ========================================================================
    # ContinuousSystemBase Interface Implementation
    # ========================================================================

    def __call__(
        self,
        x: StateVector,
        u: Optional[ControlVector] = None,
        t: ScalarLike = 0.0,
        backend: Optional[Backend] = None
    ) -> StateVector:
        """
        Evaluate continuous-time dynamics: dx/dt = f(x, u, t).

        Parameters
        ----------
        x : StateVector
            Current state (nx,) or batched (batch, nx)
        u : Optional[ControlVector]
            Control input (nu,) or batched (batch, nu)
            None for autonomous systems or zero control
        t : float
            Current time (currently ignored for time-invariant systems)

        Returns
        -------
        StateVector
            State derivative dx/dt, same shape and backend as input

        Examples
        --------
        >>> x = np.array([1.0, 0.0])
        >>> u = np.array([0.5])
        >>> dx = system(x, u)
        >>> 
        >>> # Autonomous system
        >>> dx = system(x)  # u=None
        >>> 
        >>> # Batched
        >>> x_batch = np.random.randn(100, 2)
        >>> u_batch = np.random.randn(100, 1)
        >>> dx_batch = system(x_batch, u_batch)
        """
        return self._dynamics.evaluate(x, u, backend=backend)

    def integrate(
        self,
        x0: StateVector,
        u: ControlInput = None,
        t_span: TimeSpan = (0.0, 10.0),
        method: IntegrationMethod = "RK45",
        t_eval: Optional[TimePoints] = None,
        dense_output: bool = False,
        **integrator_kwargs
    ) -> IntegrationResult:
        """
        Integrate continuous system using numerical ODE solver.

        This method creates an appropriate integrator via IntegratorFactory
        and delegates the integration. Different methods can be used for
        different calls without storing integrator state.

        Parameters
        ----------
        x0 : StateVector
            Initial state (nx,)
        u : Union[ControlVector, Callable, None]
            Control input:
            - None: Zero control (autonomous)
            - Array: Constant control u(t) = u_const
            - Callable: Time-varying u(t) or state-feedback u(t, x)
        t_span : TimeSpan
            Integration interval (t_start, t_end)
        method : IntegrationMethod
            Integration method. Options:
            
            **NumPy backend (scipy)**:
            - 'RK45': Explicit Runge-Kutta 4-5 (default, adaptive)
            - 'RK23': Explicit Runge-Kutta 2-3 (adaptive)
            - 'DOP853': High-order Dormand-Prince (adaptive)
            - 'LSODA': Auto stiff/non-stiff switching (adaptive)
            - 'Radau': Implicit Runge-Kutta (stiff, adaptive)
            - 'BDF': Backward Differentiation (stiff, adaptive)
            
            **NumPy backend (Julia via DiffEqPy)**:
            - 'Tsit5': Tsitouras 5/4 (excellent general-purpose)
            - 'Vern9': Verner 9/8 (very high accuracy)
            - 'Rosenbrock23': Implicit for moderately stiff
            - 'AutoTsit5(Rosenbrock23())': Auto-switching
            
            **PyTorch backend (torchdiffeq)**:
            - 'dopri5': Dormand-Prince 5 (adaptive)
            - 'dopri8': Dormand-Prince 8 (high accuracy)
            
            **JAX backend (diffrax)**:
            - 'tsit5': Tsitouras 5/4 (adaptive, JIT-compiled)
            - 'dopri5': Dormand-Prince 5 (adaptive)
            - 'dopri8': Dormand-Prince 8 (high accuracy)
            
            **All backends (manual)**:
            - 'rk4': Fixed-step Runge-Kutta 4
            - 'euler': Forward Euler (fixed-step)
            - 'midpoint': Midpoint method (fixed-step)
        
        t_eval : Optional[TimePoints]
            Specific times to return solution
            If None, uses solver's internal adaptive points
        
        dense_output : bool
            If True, return dense interpolated solution (adaptive only)
        
        **integrator_kwargs
            Additional integrator options:
            - dt : float (required for fixed-step methods)
            - rtol : float (relative tolerance, default: 1e-6)
            - atol : float (absolute tolerance, default: 1e-8)
            - max_steps : int (maximum steps, default: 10000)

        Returns
        -------
        IntegrationResult
            TypedDict containing:
            - t: Time points (T,)
            - x: State trajectory (T, nx) - **time-major ordering**
            - success: Integration succeeded
            - message: Status message
            - nfev: Number of function evaluations
            - nsteps: Number of steps taken
            - integration_time: Computation time (seconds)
            - solver: Integrator name
            - sol: Dense output object (if dense_output=True)

        Examples
        --------
        Basic integration:
        
        >>> result = system.integrate(
        ...     x0=np.array([1.0, 0.0]),
        ...     t_span=(0.0, 10.0)
        ... )
        >>> plt.plot(result['t'], result['x'][:, 0])
        
        Stiff system with tight tolerances:
        
        >>> result = system.integrate(
        ...     x0=x0,
        ...     t_span=(0, 100),
        ...     method='Radau',
        ...     rtol=1e-9,
        ...     atol=1e-11
        ... )
        
        State feedback control:
        
        >>> def controller(t, x):
        ...     K = np.array([[-1.0, -2.0]])
        ...     return -K @ x
        >>> result = system.integrate(x0, u=controller, t_span=(0, 5))
        
        Time-varying control:
        
        >>> def u_func(t, x):
        ...     return np.array([np.sin(t)])
        >>> result = system.integrate(x0, u=u_func, t_span=(0, 10))
        
        Constant control:
        
        >>> result = system.integrate(
        ...     x0, 
        ...     u=np.array([1.0]),  # Constant
        ...     t_span=(0, 10)
        ... )
        
        Autonomous system:
        
        >>> result = autonomous_system.integrate(
        ...     x0=np.array([1.0, 0.0]),
        ...     u=None,  # No control
        ...     t_span=(0, 10)
        ... )
        
        Fixed-step RK4:
        
        >>> result = system.integrate(
        ...     x0, 
        ...     t_span=(0, 10),
        ...     method='rk4',
        ...     dt=0.01
        ... )
        
        High-accuracy Julia solver:
        
        >>> result = system.integrate(
        ...     x0, 
        ...     t_span=(0, 10),
        ...     method='Vern9',
        ...     rtol=1e-12
        ... )
        
        Notes
        -----
        The integrator is created fresh for each call, allowing different
        methods to be used for different integration tasks without state
        management overhead. The factory pattern ensures the right integrator
        is selected based on backend and method.
        
        For autonomous systems (nu=0), u can be None or omitted entirely.
        """
        # Convert control input to standard function form
        u_func = self._prepare_control_input(u)

        # Create integrator via factory (no state stored - created on demand)
        integrator = IntegratorFactory.create(
            system=self,
            backend=self._default_backend,
            method=method,
            **integrator_kwargs
        )

        # Delegate to integrator
        return integrator.integrate(
            x0=x0,
            u_func=u_func,
            t_span=t_span,
            t_eval=t_eval,
            dense_output=dense_output
        )

    def linearize(
        self,
        x_eq: StateVector,
        u_eq: Optional[ControlVector] = None
    ) -> LinearizationResult:
        """
        Compute linearization of continuous dynamics: A = ∂f/∂x, B = ∂f/∂u.

        For continuous systems, this returns the continuous-time Jacobian matrices
        that define the linearized system: d(δx)/dt = A·δx + B·δu

        Parameters
        ----------
        x_eq : StateVector
            Equilibrium state (nx,)
        u_eq : Optional[ControlVector]
            Equilibrium control (nu,)
            If None, uses zero control

        Returns
        -------
        LinearizationResult
            Tuple (A, B) where:
            - A: State Jacobian ∂f/∂x, shape (nx, nx)
            - B: Control Jacobian ∂f/∂u, shape (nx, nu)

        Examples
        --------
        >>> x_eq = np.zeros(2)
        >>> u_eq = np.zeros(1)
        >>> A, B = system.linearize(x_eq, u_eq)
        >>> 
        >>> # Check continuous stability: Re(λ) < 0
        >>> eigenvalues = np.linalg.eigvals(A)
        >>> is_stable = np.all(np.real(eigenvalues) < 0)
        >>> 
        >>> # LQR design
        >>> from scipy.linalg import solve_continuous_are
        >>> P = solve_continuous_are(A, B, Q, R)
        >>> K = np.linalg.inv(R) @ B.T @ P
        
        Notes
        -----
        For higher-order systems, automatically constructs the full
        state-space representation with kinematic relationships.
        
        For autonomous systems (nu=0), B will be an empty (nx, 0) matrix.
        """
        return self._linearization.compute_dynamics(x_eq, u_eq, backend=None)

    def print_equations(self, simplify: bool = True):
        """
        Print symbolic equations using continuous-time notation.

        Displays the system's symbolic dynamics and output equations
        with d/dt notation appropriate for continuous systems.

        Parameters
        ----------
        simplify : bool
            If True, simplify expressions before printing

        Examples
        --------
        >>> system.print_equations()
        ======================================================================
        Pendulum
        ======================================================================
        State Variables: [theta, omega]
        Control Variables: [u]
        System Order: 1
        Dimensions: nx=2, nu=1, ny=2

        Dynamics: dx/dt = f(x, u)
          dtheta/dt = omega
          domega/dt = -19.62*sin(theta) - 0.1*omega + 4.0*u
        ======================================================================
        """
        print("=" * 70)
        print(f"{self.__class__.__name__} (Continuous-Time)")
        print("=" * 70)
        print(f"State Variables: {self.state_vars}")
        print(f"Control Variables: {self.control_vars}")
        print(f"System Order: {self.order}")
        print(f"Dimensions: nx={self.nx}, nu={self.nu}, ny={self.ny}")

        print("\nDynamics: dx/dt = f(x, u)")
        for var, expr in zip(self.state_vars, self._f_sym):
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
    # Hook Method Implementation for Equilibrium Verification
    # ========================================================================

    def _verify_equilibrium_numpy(
        self,
        x_eq: np.ndarray,
        u_eq: np.ndarray,
        tol: ScalarLike
    ) -> bool:
        """
        Verify continuous equilibrium condition: ||f(x_eq, u_eq)|| < tol.

        For continuous systems, an equilibrium satisfies dx/dt = 0, which
        means f(x_eq, u_eq) ≈ 0.

        Parameters
        ----------
        x_eq : np.ndarray
            Candidate equilibrium state (nx,)
        u_eq : np.ndarray
            Candidate equilibrium control (nu,)
        tol : float
            Tolerance for verification

        Returns
        -------
        bool
            True if ||f(x_eq, u_eq)|| < tol

        Examples
        --------
        >>> # Valid equilibrium: dx/dt = -x + u = 0 when x = u
        >>> is_valid = system._verify_equilibrium_numpy(
        ...     np.array([1.0]), 
        ...     np.array([1.0]), 
        ...     tol=1e-6
        ... )
        >>> assert is_valid
        
        Notes
        -----
        Uses NumPy backend for consistency regardless of default backend.
        Called by add_equilibrium() when verify=True.
        """
        # Evaluate dynamics at equilibrium point
        dx = self(x_eq, u_eq, t=0.0)
        
        # Check if near zero
        error = np.linalg.norm(dx)
        return error < tol

    # ========================================================================
    # Control Input Preparation (Helper for integrate())
    # ========================================================================

    def _prepare_control_input(
        self,
        u: ControlInput
    ) -> Callable[[float, StateVector], Optional[ControlVector]]:
        """
        Convert various control input formats to standard function.

        Converts the flexible ControlInput type to the standard (t, x) → u
        function signature expected by integrators.

        Parameters
        ----------
        u : ControlInput
            Control input in various formats:
            - None: Zero control or autonomous
            - Array: Constant control
            - Callable: Time-varying or state-feedback

        Returns
        -------
        Callable[[float, StateVector], Optional[ControlVector]]
            Standard control function: (t, x) → u

        Notes
        -----
        This method handles the impedance mismatch between the user-friendly
        ControlInput API and the integrator's requirement for a specific
        function signature.
        """
        if u is None:
            # Autonomous system or zero control
            if self.nu == 0:
                return lambda t, x: None
            else:
                # Zero control for non-autonomous system
                return lambda t, x: np.zeros(self.nu)

        elif callable(u):
            # Check function signature
            sig = inspect.signature(u)
            n_params = len(sig.parameters)

            if n_params == 1:
                # u(t) - time-varying only
                return lambda t, x: u(t)
            elif n_params == 2:
                # u(t, x) - state feedback (already correct form)
                # Try to determine parameter order
                try:
                    # Test with dummy values to see if (t, x) works
                    test_x = np.zeros(self.nx)
                    _ = u(0.0, test_x)
                    # Success - it's (t, x) order
                    return u
                except (TypeError, ValueError):
                    # Try (x, t) order
                    try:
                        _ = u(test_x, 0.0)
                        # It's (x, t) order - need to swap
                        return lambda t, x: u(x, t)
                    except:
                        # Can't determine - assume (t, x) and let it fail later
                        return u
            else:
                raise ValueError(
                    f"Control function must have signature u(t) or u(t, x), "
                    f"got {n_params} parameters"
                )

        else:
            # Constant control - convert array to function
            u_array = np.asarray(u)
            return lambda t, x: u_array

    # ========================================================================
    # Additional Dynamics Methods (Convenience/Aliases)
    # ========================================================================

    def forward(
        self,
        x: StateVector,
        u: Optional[ControlVector] = None,
        backend: Optional[Backend] = None
    ) -> StateVector:
        """
        Alias for dynamics evaluation with explicit backend specification.

        Equivalent to __call__ but allows explicit backend override.

        Parameters
        ----------
        x : StateVector
            State (nx,)
        u : Optional[ControlVector]
            Control (nu,)
        backend : Optional[Backend]
            Backend override (None = use default)

        Returns
        -------
        StateVector
            State derivative dx/dt

        Examples
        --------
        >>> dx = system.forward(x, u)  # Use default backend
        >>> dx = system.forward(x, u, backend='torch')  # Force PyTorch
        """
        return self._dynamics.evaluate(x, u, backend)

    # ========================================================================
    # Linearization Methods (Extended Interface)
    # ========================================================================

    def linearized_dynamics(
        self,
        x: Union[StateVector, str],
        u: Optional[ControlVector] = None,
        backend: Optional[Backend] = None,
    ) -> Tuple[ArrayLike, ArrayLike]:
        """
        Compute numerical linearization: A = ∂f/∂x, B = ∂f/∂u.

        Alias for linearize() with support for named equilibria and
        explicit backend specification.

        Parameters
        ----------
        x : Union[StateVector, str]
            State to linearize at, OR equilibrium name
        u : Optional[ControlVector]
            Control to linearize at (ignored if x is string)
        backend : Optional[Backend]
            Backend for result arrays

        Returns
        -------
        Tuple[ArrayLike, ArrayLike]
            (A, B) Jacobian matrices

        Examples
        --------
        >>> # By state/control
        >>> A, B = system.linearized_dynamics(
        ...     np.array([0.0, 0.0]),
        ...     np.array([0.0])
        ... )
        >>> 
        >>> # By equilibrium name
        >>> A, B = system.linearized_dynamics('inverted')
        >>> 
        >>> # Force PyTorch backend
        >>> A, B = system.linearized_dynamics(x, u, backend='torch')
        """
        # Handle named equilibrium
        if isinstance(x, str):
            equilibrium_name = x
            backend = backend or self._default_backend
            x, u = self.equilibria.get_both(equilibrium_name, backend)

        return self._linearization.compute_dynamics(x, u, backend)

    def linearized_dynamics_symbolic(
        self,
        x_eq: Optional[Union[sp.Matrix, str]] = None,
        u_eq: Optional[sp.Matrix] = None,
    ) -> Tuple[sp.Matrix, sp.Matrix]:
        """
        Compute symbolic linearization: A = ∂f/∂x, B = ∂f/∂u.

        Returns symbolic matrices for analytical analysis.

        Parameters
        ----------
        x_eq : Optional[Union[sp.Matrix, str]]
            Equilibrium state (symbolic) OR equilibrium name
            If None, uses zeros
        u_eq : Optional[sp.Matrix]
            Equilibrium control (symbolic)
            If None, uses zeros

        Returns
        -------
        Tuple[sp.Matrix, sp.Matrix]
            (A, B) symbolic matrices with parameters substituted

        Examples
        --------
        >>> A_sym, B_sym = system.linearized_dynamics_symbolic()
        >>> print(A_sym)
        Matrix([[0, 1], [-10.0, -0.5]])
        >>> 
        >>> # At named equilibrium
        >>> A_sym, B_sym = system.linearized_dynamics_symbolic('inverted')
        >>> 
        >>> # Convert to NumPy
        >>> A_np = np.array(A_sym, dtype=float)
        """
        # Handle named equilibrium
        if isinstance(x_eq, str):
            equilibrium_name = x_eq
            x_np, u_np = self.equilibria.get_both(equilibrium_name, backend="numpy")
            x_eq = sp.Matrix(x_np.tolist())
            u_eq = sp.Matrix(u_np.tolist())

        return self._linearization.compute_symbolic(x_eq, u_eq)

    def verify_jacobians(
        self,
        x: StateVector,
        u: Optional[ControlVector] = None,
        tol: float = 1e-3,
        backend: str = "torch"
    ) -> Dict[str, Union[bool, float]]:
        """
        Verify symbolic Jacobians against automatic differentiation.

        Compares analytically-derived Jacobians (from SymPy) against
        numerically-computed Jacobians (from PyTorch/JAX autodiff).

        Parameters
        ----------
        x : StateVector
            State at which to verify
        u : Optional[ControlVector]
            Control at which to verify (None for autonomous)
        tol : float
            Tolerance for considering Jacobians equal
        backend : str
            Backend for autodiff ('torch' or 'jax', not 'numpy')

        Returns
        -------
        Dict[str, Union[bool, float]]
            Verification results:
            - 'A_match': bool - True if A matches
            - 'B_match': bool - True if B matches
            - 'A_error': float - Maximum absolute error in A
            - 'B_error': float - Maximum absolute error in B

        Examples
        --------
        >>> x = np.array([0.1, 0.0])
        >>> u = np.array([0.0])
        >>> results = system.verify_jacobians(x, u, backend='torch')
        >>> 
        >>> if results['A_match'] and results['B_match']:
        ...     print("✓ Jacobians verified!")
        ... else:
        ...     print(f"✗ A error: {results['A_error']:.2e}")
        ...     print(f"✗ B error: {results['B_error']:.2e}")
        
        Notes
        -----
        Requires PyTorch or JAX for automatic differentiation.
        Small errors (< 1e-6) are usually numerical precision issues.
        Large errors indicate bugs in symbolic Jacobian computation.
        """
        return self._linearization.verify_jacobians(x, u, backend, tol)

    # ========================================================================
    # Output Function Methods
    # ========================================================================

    def h(
        self,
        x: StateVector,
        backend: Optional[Backend] = None
    ) -> ArrayLike:
        """
        Evaluate output equation: y = h(x).

        Computes the system output at the given state. If no custom output
        function is defined, returns the full state (identity map).

        Parameters
        ----------
        x : StateVector
            State vector (nx,) or batched (batch, nx)
        backend : Optional[Backend]
            Backend selection (None = auto-detect)

        Returns
        -------
        ArrayLike
            Output vector y, shape (ny,) or (batch, ny)

        Examples
        --------
        >>> # Identity output
        >>> y = system.h(x)
        >>> np.allclose(y, x)
        True
        >>> 
        >>> # Custom output
        >>> x = np.array([1.0, 2.0])
        >>> y = system.h(x)  # Might return energy, distance, etc.
        """
        return self._observation.evaluate(x, backend)

    def linearized_observation(
        self,
        x: StateVector,
        backend: Optional[Backend] = None
    ) -> ArrayLike:
        """
        Compute linearized observation matrix: C = ∂h/∂x.

        Parameters
        ----------
        x : StateVector
            State at which to linearize (nx,)
        backend : Optional[Backend]
            Backend selection

        Returns
        -------
        ArrayLike
            C matrix, shape (ny, nx)

        Examples
        --------
        >>> x = np.array([1.0, 2.0])
        >>> C = system.linearized_observation(x)
        >>> 
        >>> # For identity output
        >>> np.allclose(C, np.eye(system.nx))
        True
        """
        return self._observation.compute_jacobian(x, backend)

    def linearized_observation_symbolic(
        self,
        x_eq: Optional[sp.Matrix] = None
    ) -> sp.Matrix:
        """
        Compute symbolic observation Jacobian: C = ∂h/∂x.

        Parameters
        ----------
        x_eq : Optional[sp.Matrix]
            Equilibrium state (symbolic), None = zeros

        Returns
        -------
        sp.Matrix
            Symbolic C matrix

        Examples
        --------
        >>> C_sym = system.linearized_observation_symbolic()
        >>> print(C_sym)
        """
        return self._observation.compute_symbolic(x_eq)

    # ========================================================================
    # Performance Statistics (Override from SymbolicSystemBase)
    # ========================================================================

    def get_performance_stats(self) -> Dict[str, float]:
        """
        Get performance statistics from all components.

        Collects timing and call count from DynamicsEvaluator and
        LinearizationEngine.

        Returns
        -------
        Dict[str, float]
            Statistics:
            - 'forward_calls': Number of forward dynamics calls
            - 'forward_time': Total forward time (seconds)
            - 'avg_forward_time': Average forward time
            - 'linearization_calls': Number of linearizations
            - 'linearization_time': Total linearization time
            - 'avg_linearization_time': Average linearization time

        Examples
        --------
        >>> for _ in range(100):
        ...     dx = system(x, u)
        >>> 
        >>> stats = system.get_performance_stats()
        >>> print(f"Forward calls: {stats['forward_calls']}")
        >>> print(f"Avg time: {stats['avg_forward_time']:.6f}s")
        """
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

        Clears statistics in DynamicsEvaluator and LinearizationEngine.

        Examples
        --------
        >>> system.reset_performance_stats()
        >>> stats = system.get_performance_stats()
        >>> assert stats['forward_calls'] == 0
        """
        self._dynamics.reset_stats()
        self._linearization.reset_stats()

    # ========================================================================
    # Utility Methods
    # ========================================================================

    def warmup(
        self,
        backend: Optional[Backend] = None,
        test_point: Optional[Tuple[StateVector, ControlVector]] = None,
    ) -> bool:
        """
        Warm up backend by compiling and running test evaluation.

        Useful for JIT compilation warmup (especially JAX) and validating
        backend configuration before critical operations.

        Parameters
        ----------
        backend : Optional[Backend]
            Backend to warm up (None = default)
        test_point : Optional[Tuple[StateVector, ControlVector]]
            Test (x, u) point (None = use equilibrium)

        Returns
        -------
        bool
            True if warmup successful

        Examples
        --------
        >>> system.set_default_backend('jax', device='gpu:0')
        >>> success = system.warmup()
        >>> # First call triggers JIT compilation
        """
        backend = backend or self._default_backend

        # Generate code
        self._code_gen.generate_dynamics(backend)
        if self._h_sym is not None:
            self._code_gen.generate_output(backend)

        # Test evaluation
        if test_point is not None:
            x_test, u_test = test_point
        else:
            x_test = self.equilibria.get_x(backend=backend)
            u_test = self.equilibria.get_u(backend=backend)

        try:
            dx = self.forward(x_test, u_test, backend=backend)
            return True
        except Exception:
            return False


# ============================================================================
# Migration Alias (Backward Compatibility)
# ============================================================================

# Old name (deprecated but still works)
SymbolicDynamicalSystem = ContinuousSymbolicSystem
"""
Deprecated alias for backward compatibility.

Use ContinuousSymbolicSystem instead. This alias will be removed in a
future version after a deprecation period.

Examples
--------
>>> # Old code (still works)
>>> class MySystem(SymbolicDynamicalSystem):
...     pass
>>> 
>>> # New code (preferred)
>>> class MySystem(ContinuousSymbolicSystem):
...     pass
"""

# Also provide shorter alias
ContinuousDynamicalSystem = ContinuousSymbolicSystem
"""
Shorter alias for ContinuousSymbolicSystem.

Both names are equally valid - use whichever reads better in your context.
"""