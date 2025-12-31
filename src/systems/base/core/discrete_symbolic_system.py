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
Discrete Symbolic System - Symbolic Discrete-Time Dynamical Systems
====================================================================

This module provides the concrete implementation for symbolic discrete-time
dynamical systems, combining the symbolic machinery from SymbolicSystemBase
with the discrete-time interface from DiscreteSystemBase.

This is the NEW refactored implementation that uses multiple inheritance
to eliminate code duplication with continuous systems.

Architecture
-----------
```
    SymbolicSystemBase          DiscreteSystemBase
    (symbolic machinery)        (discrete interface)
            |                            |
            +-------------+--------------+
                          |
              DiscreteSymbolicSystem
              (combines both via multiple inheritance)
```

Key Features
------------
- Multi-backend execution (NumPy, PyTorch, JAX)
- Automatic code generation from symbolic expressions
- Direct state transition (no integration needed)
- Linearization: Ad = ∂f/∂x, Bd = ∂f/∂u
- Multi-step simulation with rollout support
- Equilibrium verification for discrete systems
- Performance tracking

What This Class Provides
------------------------
- State transition: x[k+1] = f(x[k], u[k])
- Multi-step simulation via rollout
- Linearization: Ad = ∂f/∂x, Bd = ∂f/∂u
- Output evaluation: y[k] = h(x[k])
- Equilibrium verification: ||f(x_eq, u_eq) - x_eq|| < tol
- Sampling period management (dt)

What's Inherited from SymbolicSystemBase
----------------------------------------
- Symbolic variable management
- Code generation and compilation
- Backend configuration
- Equilibrium storage
- Configuration persistence

Usage Example
-------------
```python
class DiscreteLinear(DiscreteSymbolicSystem):
    def define_system(self, a=0.9, b=0.1, dt=0.01):
        x = sp.symbols('x', real=True)
        u = sp.symbols('u', real=True)
        a_sym, b_sym = sp.symbols('a b', real=True)
        
        self.state_vars = [x]
        self.control_vars = [u]
        self._f_sym = sp.Matrix([a_sym*x + b_sym*u])
        self.parameters = {a_sym: a, b_sym: b}
        self._dt = dt  # Required for discrete systems!
        self.order = 1

# Create and use
system = DiscreteLinear(a=0.95, dt=0.1)

# Single step
x_k = np.array([1.0])
u_k = np.array([0.5])
x_next = system.step(x_k, u_k)

# Multi-step simulation
result = system.simulate(
    x0=np.array([1.0]),
    u_sequence=np.zeros((100, 1)),
    n_steps=100
)

# Linearize
Ad, Bd = system.linearize(np.zeros(1), np.zeros(1))
```

See Also
--------
- SymbolicSystemBase: Base symbolic machinery
- DiscreteSystemBase: Discrete-time interface
- DynamicsEvaluator: State transition evaluation
- LinearizationEngine: Linearization computation

Authors
-------
Gil Benezer

License
-------
AGPL-3.0
"""

from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np
import sympy as sp

from src.systems.base.core.discrete_system_base import DiscreteSystemBase
from src.systems.base.core.symbolic_system_base import SymbolicSystemBase
from src.systems.base.utils.dynamics_evaluator import DynamicsEvaluator
from src.systems.base.utils.linearization_engine import LinearizationEngine
from src.systems.base.utils.observation_engine import ObservationEngine

# Type imports
from src.types.core import (
    OutputMatrix,
    ControlVector,
    OutputVector,
    DiscreteControlInput,
    ScalarLike,
    StateVector,
)
from src.types.backends import Backend
from src.types.linearization import LinearizationResult
from src.types.trajectories import DiscreteSimulationResult


class DiscreteSymbolicSystem(SymbolicSystemBase, DiscreteSystemBase):
    """
    Concrete symbolic discrete-time dynamical system.

    Combines symbolic machinery from SymbolicSystemBase with discrete-time
    interface from DiscreteSystemBase.

    Represents difference equations:
        x[k+1] = f(x[k], u[k])
        y[k] = h(x[k])

    where:
        x[k] ∈ ℝⁿˣ: State at discrete time k
        u[k] ∈ ℝⁿᵘ: Control input at time k
        y[k] ∈ ℝⁿʸ: Output at time k
        k ∈ ℤ: Discrete time index

    Users subclass and implement define_system() to specify symbolic dynamics.

    **CRITICAL**: Discrete systems must set self._dt in define_system()!

    Examples
    --------
    >>> class DiscreteOscillator(DiscreteSymbolicSystem):
    ...     def define_system(self, a=0.95, b=0.05, dt=0.1):
    ...         x, v = sp.symbols('x v', real=True)
    ...         u = sp.symbols('u', real=True)
    ...         a_sym, b_sym = sp.symbols('a b', real=True)
    ...
    ...         self.state_vars = [x, v]
    ...         self.control_vars = [u]
    ...         self._f_sym = sp.Matrix([
    ...             a_sym*x + (1-a_sym)*v,
    ...             v + b_sym*u
    ...         ])
    ...         self.parameters = {a_sym: a, b_sym: b}
    ...         self._dt = dt  # REQUIRED!
    ...         self.order = 1
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize discrete symbolic system.

        Follows cooperative multiple inheritance:
        1. SymbolicSystemBase.__init__() handles symbolic setup
        2. DiscreteSystemBase properties available
        3. Discrete-specific evaluators initialized

        **IMPORTANT**: define_system() MUST set self._dt!
        """
        
        # Initialize both bases
        super().__init__(*args, **kwargs)

        # Verify dt was set in define_system()
        if not hasattr(self, '_dt'):
            raise ValueError(
                f"{self.__class__.__name__} must define self._dt in define_system(). "
                "Example: self._dt = 0.01"
            )

        # Initialize discrete-specific evaluators
        self._dynamics = DynamicsEvaluator(self, self._code_gen, self.backend)
        """Evaluates next state: x[k+1] = f(x[k], u[k])"""

        self._linearization = LinearizationEngine(self, self._code_gen, self.backend)
        """Computes discrete linearization: Ad = ∂f/∂x, Bd = ∂f/∂u"""

        self._observation = ObservationEngine(self, self._code_gen, self.backend)
        """Evaluates output: y[k] = h(x[k])"""
        

    # ========================================================================
    # DiscreteSystemBase Interface Implementation
    # ========================================================================

    @property
    def dt(self) -> ScalarLike:
        """
        Sampling period / time step.

        Returns
        -------
        float
            Time step in seconds (default is 1.0 seconds)

        Examples
        --------
        >>> system.dt
        0.01
        >>> print(f"Sampling rate: {1/system.dt} Hz")
        """
        return self._dt

    def step(
        self,
        x: StateVector,
        u: Optional[ControlVector] = None,
        k: int = 0,
        backend: Optional[Backend] = None
    ) -> StateVector:
        """
        Compute next state: x[k+1] = f(x[k], u[k]).

        Parameters
        ----------
        x : StateVector
            Current state x[k], shape (nx,)
        u : Optional[ControlVector]
            Control input u[k], shape (nu,)
            None for autonomous systems
        k : int
            Time step index (currently ignored for time-invariant systems)

        Returns
        -------
        StateVector
            Next state x[k+1], same shape and backend as input

        Examples
        --------
        >>> x_k = np.array([1.0, 0.0])
        >>> u_k = np.array([0.5])
        >>> x_next = system.step(x_k, u_k)
        >>> 
        >>> # Autonomous
        >>> x_next = system.step(x_k)  # u=None
        """
        return self._dynamics.evaluate(x, u, backend=backend)

    def simulate(
        self,
        x0: StateVector,
        u_sequence: DiscreteControlInput = None,
        n_steps: int = 100,
        **kwargs
    ) -> DiscreteSimulationResult:
        """
        Simulate discrete system for multiple steps.

        Repeatedly applies step() to generate state trajectory.

        Parameters
        ----------
        x0 : StateVector
            Initial state (nx,)
        u_sequence : Union[ControlVector, Sequence, Callable, None]
            Control sequence:
            - None: Zero control
            - Array (nu,): Constant control
            - Array (n_steps, nu): Pre-computed sequence
            - Callable(k): Time-indexed u[k] = u_func(k)
        n_steps : int
            Number of simulation steps

        Returns
        -------
        DiscreteSimulationResult
            TypedDict containing:
            - states: State trajectory (nx, n_steps+1) - includes x[0]
            - controls: Control sequence (nu, n_steps)
            - time_steps: [0, 1, 2, ..., n_steps]
            - dt: Sampling period
            - metadata: Additional info

        Examples
        --------
        >>> # Constant control
        >>> result = system.simulate(
        ...     x0=np.array([1.0]),
        ...     u_sequence=np.array([0.5]),
        ...     n_steps=100
        ... )
        >>> 
        >>> # Pre-computed sequence
        >>> u_seq = np.random.randn(100, 1)
        >>> result = system.simulate(x0, u_seq, n_steps=100)
        >>> 
        >>> # Time-indexed function
        >>> result = system.simulate(
        ...     x0, 
        ...     u_sequence=lambda k: np.array([np.sin(k*system.dt)]),
        ...     n_steps=100
        ... )
        """
        # Initialize storage
        states = np.zeros((self.nx, n_steps + 1))
        states[:, 0] = x0

        # Prepare control function
        u_func = self._prepare_control_sequence(u_sequence, n_steps)

        # Simulate forward
        x = x0
        controls = []
        for k in range(n_steps):
            u = u_func(k, x)
            controls.append(u)
            x = self.step(x, u, k)
            states[:, k + 1] = x

        # Format controls
        if controls and controls[0] is not None:
            controls_array = np.array(controls).T
        else:
            controls_array = None

        return {
            "states": states.T, # need to check that this is correct for all backends
            "controls": controls_array,
            "time_steps": np.arange(n_steps + 1),
            "dt": self._dt,
            "success": True,
            "metadata": {"method": "discrete_step"}
        }

    def linearize(
        self,
        x_eq: StateVector,
        u_eq: Optional[ControlVector] = None
    ) -> LinearizationResult:
        """
        Compute discrete linearization: Ad = ∂f/∂x, Bd = ∂f/∂u.

        For discrete systems: δx[k+1] = Ad·δx[k] + Bd·δu[k]

        Parameters
        ----------
        x_eq : StateVector
            Equilibrium state (nx,)
        u_eq : Optional[ControlVector]
            Equilibrium control (nu,)

        Returns
        -------
        DiscreteLinearization
            Tuple (Ad, Bd) where:
            - Ad: State transition matrix, shape (nx, nx)
            - Bd: Control matrix, shape (nx, nu)

        Examples
        --------
        >>> Ad, Bd = system.linearize(np.zeros(2), np.zeros(1))
        >>> 
        >>> # Check discrete stability: |λ| < 1
        >>> eigenvalues = np.linalg.eigvals(Ad)
        >>> is_stable = np.all(np.abs(eigenvalues) < 1.0)
        >>> 
        >>> # Discrete LQR
        >>> from scipy.linalg import solve_discrete_are
        >>> P = solve_discrete_are(Ad, Bd, Q, R)
        >>> K = np.linalg.inv(R + Bd.T @ P @ Bd) @ (Bd.T @ P @ Ad)
        """
        return self._linearization.compute_dynamics(x_eq, u_eq, backend=None)

    def print_equations(self, simplify: bool = True):
        """
        Print symbolic equations using discrete-time notation.

        Uses x[k+1] notation appropriate for difference equations.

        Parameters
        ----------
        simplify : bool
            If True, simplify expressions before printing

        Examples
        --------
        >>> system.print_equations()
        ======================================================================
        DiscreteLinear (Discrete-Time, dt=0.01)
        ======================================================================
        State Variables: [x]
        Control Variables: [u]
        Dimensions: nx=1, nu=1, ny=1

        Dynamics: x[k+1] = f(x[k], u[k])
          x[k+1] = 0.9*x[k] + 0.1*u[k]
        ======================================================================
        """
        print("=" * 70)
        print(f"{self.__class__.__name__} (Discrete-Time, dt={self.dt})")
        print("=" * 70)
        print(f"State Variables: {self.state_vars}")
        print(f"Control Variables: {self.control_vars}")
        print(f"System Order: {self.order}")
        print(f"Dimensions: nx={self.nx}, nu={self.nu}, ny={self.ny}")

        print("\nDynamics: x[k+1] = f(x[k], u[k])")
        for var, expr in zip(self.state_vars, self._f_sym):
            expr_sub = self.substitute_parameters(expr)
            if simplify:
                expr_sub = sp.simplify(expr_sub)
            print(f"  {var}[k+1] = {expr_sub}")

        if self._h_sym is not None:
            print("\nOutput: y[k] = h(x[k])")
            for i, expr in enumerate(self._h_sym):
                expr_sub = self.substitute_parameters(expr)
                if simplify:
                    expr_sub = sp.simplify(expr_sub)
                print(f"  y[{i}][k] = {expr_sub}")

        print("=" * 70)

    # ========================================================================
    # Hook Method for Equilibrium Verification
    # ========================================================================

    def _verify_equilibrium_numpy(
        self,
        x_eq: np.ndarray,
        u_eq: np.ndarray,
        tol: ScalarLike
    ) -> bool:
        """
        Verify discrete equilibrium condition: ||f(x_eq, u_eq) - x_eq|| < tol.

        For discrete systems, an equilibrium is a fixed point where
        f(x_eq, u_eq) = x_eq.

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
            True if ||f(x_eq, u_eq) - x_eq|| < tol

        Examples
        --------
        >>> # For x[k+1] = 0.9*x + 0.1*u, fixed point when x = 0.9*x + 0.1*u
        >>> # => 0.1*x = 0.1*u => x = u
        >>> is_valid = system._verify_equilibrium_numpy(
        ...     np.array([1.0]),
        ...     np.array([1.0]),
        ...     tol=1e-6
        ... )
        >>> assert is_valid
        """
        # Compute next state
        x_next = self.step(x_eq, u_eq, k=0)
        
        # Check if fixed point
        error = np.linalg.norm(x_next - x_eq)
        return error <= tol

    # ========================================================================
    # Control Sequence Preparation (Helper for simulate())
    # ========================================================================

    def _prepare_control_sequence(
        self,
        u_sequence: DiscreteControlInput,
        n_steps: int
    ) -> Callable[[int, StateVector], Optional[ControlVector]]:
        """
        Convert various control sequence formats to standard function.

        Converts DiscreteControlInput to (k, x) → u function.

        Parameters
        ----------
        u_sequence : DiscreteControlInput
            Control in various formats
        n_steps : int
            Number of steps (for validation)

        Returns
        -------
        Callable[[int, StateVector], Optional[ControlVector]]
            Standard function: (k, x) → u[k]
        """
        if u_sequence is None:
            # Zero control or autonomous
            if self.nu == 0:
                return lambda k, x: None
            else:
                return lambda k, x: np.zeros(self.nu)

        elif callable(u_sequence):
            # Time-indexed function or state feedback
            import inspect
            sig = inspect.signature(u_sequence)
            n_params = len(sig.parameters)

            if n_params == 1:
                # u(k) - time-indexed
                return lambda k, x: u_sequence(k)
            elif n_params == 2:
                # u(k, x) or u(x, k) - state feedback
                # Try to detect order
                try:
                    test_x = np.zeros(self.nx)
                    _ = u_sequence(0, test_x)
                    # Works with (k, x)
                    return u_sequence
                except:
                    # Try (x, k)
                    try:
                        _ = u_sequence(test_x, 0)
                        # Works with (x, k) - swap
                        return lambda k, x: u_sequence(x, k)
                    except:
                        # Assume (k, x) standard
                        return u_sequence
            else:
                raise ValueError(
                    f"Control function must accept 1 or 2 parameters, "
                    f"got {n_params}"
                )

        elif isinstance(u_sequence, np.ndarray):
            if u_sequence.ndim == 1:
                # Constant control
                if u_sequence.shape[0] != self.nu:
                    raise ValueError(
                        f"Control dimension mismatch. Expected (nu={self.nu},), "
                        f"got shape {u_sequence.shape}"
                    )
                return lambda k, x: u_sequence
            
            elif u_sequence.ndim == 2:
                # Pre-computed sequence (n_steps, nu) or (nu, n_steps)
                if u_sequence.shape[0] == n_steps:
                    # (n_steps, nu) - time-major
                    return lambda k, x: u_sequence[k] if k < len(u_sequence) else u_sequence[-1]
                elif u_sequence.shape[1] == n_steps:
                    # (nu, n_steps) - state-major
                    return lambda k, x: u_sequence[:, k] if k < u_sequence.shape[1] else u_sequence[:, -1]
                else:
                    raise ValueError(
                        f"Control sequence shape mismatch. Expected ({n_steps}, {self.nu}) or "
                        f"({self.nu}, {n_steps}), got {u_sequence.shape}"
                    )
            else:
                raise ValueError(
                    f"Control array must be 1D or 2D, got {u_sequence.ndim}D"
                )

        elif isinstance(u_sequence, (list, tuple)):
            # List of control values
            def list_control(k, x):
                if k < len(u_sequence):
                    return np.asarray(u_sequence[k])
                else:
                    return np.asarray(u_sequence[-1])
            return list_control

        else:
            raise TypeError(
                f"Invalid control sequence type: {type(u_sequence)}. "
                f"Expected None, array, callable, or sequence."
            )

    # ========================================================================
    # Additional Methods (Convenience/Aliases)
    # ========================================================================

    def forward(
        self,
        x: StateVector,
        u: Optional[ControlVector] = None,
        backend: Optional[Backend] = None
    ) -> StateVector:
        """
        Alias for step() with explicit backend specification.

        Parameters
        ----------
        x : StateVector
            Current state x[k]
        u : Optional[ControlVector]
            Control u[k]
        backend : Optional[Backend]
            Backend override

        Returns
        -------
        StateVector
            Next state x[k+1]

        Examples
        --------
        >>> x_next = system.forward(x, u)
        >>> x_next = system.forward(x, u, backend='torch')
        """
        return self._dynamics.evaluate(x, u, backend)

    # ========================================================================
    # Linearization Methods (Extended)
    # ========================================================================

    def linearized_dynamics(
        self,
        x: Union[StateVector, str],
        u: Optional[ControlVector] = None,
        backend: Optional[Backend] = None,
    ) -> LinearizationResult:
        """
        Compute discrete linearization (alias for linearize()).

        Parameters
        ----------
        x : Union[StateVector, str]
            State or equilibrium name
        u : Optional[ControlVector]
            Control
        backend : Optional[Backend]
            Backend for result

        Returns
        -------
        Tuple[ArrayLike, ArrayLike]
            (Ad, Bd) discrete Jacobians

        Examples
        --------
        >>> Ad, Bd = system.linearized_dynamics(np.zeros(2), np.zeros(1))
        >>> Ad, Bd = system.linearized_dynamics('origin')
        """
        if isinstance(x, str):
            backend = backend or self._default_backend
            x, u = self.equilibria.get_both(x, backend)

        return self._linearization.compute_dynamics(x, u, backend)

    def linearized_dynamics_symbolic(
        self,
        x_eq: Optional[Union[sp.Matrix, str]] = None,
        u_eq: Optional[sp.Matrix] = None,
    ) -> Tuple[sp.Matrix, sp.Matrix]:
        """
        Compute symbolic discrete linearization.

        Parameters
        ----------
        x_eq : Optional[Union[sp.Matrix, str]]
            Equilibrium state or name
        u_eq : Optional[sp.Matrix]
            Equilibrium control

        Returns
        -------
        Tuple[sp.Matrix, sp.Matrix]
            (Ad, Bd) symbolic matrices

        Examples
        --------
        >>> Ad_sym, Bd_sym = system.linearized_dynamics_symbolic()
        """
        if isinstance(x_eq, str):
            x_np, u_np = self.equilibria.get_both(x_eq, backend="numpy")
            x_eq = sp.Matrix(x_np.tolist())
            u_eq = sp.Matrix(u_np.tolist())

        return self._linearization.compute_symbolic(x_eq, u_eq)

    # ========================================================================
    # Output Functions
    # ========================================================================

    def h(
        self,
        x: StateVector,
        backend: Optional[Backend] = None
    ) -> OutputVector:
        """
        Evaluate output: y[k] = h(x[k]).

        Parameters
        ----------
        x : StateVector
            State x[k]
        backend : Optional[Backend]
            Backend selection

        Returns
        -------
        ArrayLike
            Output y[k]
        """
        return self._observation.evaluate(x, backend)

    def linearized_observation(
        self,
        x: StateVector,
        backend: Optional[Backend] = None
    ) -> OutputMatrix:
        """
        Compute C = ∂h/∂x.

        Parameters
        ----------
        x : StateVector
            State
        backend : Optional[Backend]
            Backend

        Returns
        -------
        ArrayLike
            C matrix (ny, nx)
        """
        return self._observation.compute_jacobian(x, backend)

    # ========================================================================
    # Performance Statistics
    # ========================================================================

    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics from evaluators."""
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
        """Reset performance counters."""
        self._dynamics.reset_stats()
        self._linearization.reset_stats()


# ============================================================================
# Migration Alias (Backward Compatibility)
# ============================================================================

DiscreteDynamicalSystem = DiscreteSymbolicSystem
"""
Alias for DiscreteSymbolicSystem.

Both names are equally valid.
"""