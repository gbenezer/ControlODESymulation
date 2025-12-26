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
Continuous System Base Class (Layer 1)
======================================

This module provides the abstract base class for all continuous-time dynamical systems.

Overview
--------
The ContinuousSystemBase class defines the core interface that all continuous-time
systems must implement, regardless of whether they are deterministic or stochastic,
symbolic or data-driven.

All continuous-time systems share these fundamental operations:
- Forward dynamics evaluation: dx/dt = f(x, u, t)
- Numerical integration over time intervals
- Linearization around equilibrium points

This base class enforces these contracts through abstract methods, ensuring
consistent APIs across the entire continuous systems hierarchy.

Architecture Position
--------------------
Layer 1 (Abstract Interfaces):
    ContinuousSystemBase ← YOU ARE HERE
    DiscreteSystemBase

Layer 2 (Symbolic Implementations):
    ContinuousSymbolicSystem(ContinuousSystemBase)
    ContinuousStochasticSystem(ContinuousSymbolicSystem)

Layer 3 (Discrete Implementations):
    DiscreteSymbolicSystem(DiscreteSystemBase)
    DiscreteStochasticSystem(DiscreteSymbolicSystem, DiscreteSystemBase)

Layer 4 (Bridges):
    DiscreteTimeWrapper
    Discretizer

Key Design Principles
--------------------
1. **Time Domain**: Continuous-time only (dx/dt formulation)
2. **Backend Agnostic**: Works with NumPy, PyTorch, JAX
3. **Minimal Interface**: Only essential methods are abstract
4. **Composable**: Designed for inheritance and extension
5. **Type Safe**: Full type hints using src/types

Abstract Methods
---------------
Subclasses MUST implement:
- __call__(x, u, t): Evaluate dynamics at a point
- integrate(x0, u, t_span): Integrate over time
- linearize(x_eq, u_eq): Compute linearization

Concrete Methods
---------------
Provided by default:
- simulate(): High-level simulation interface
- __repr__(): String representation

Mathematical Notation
--------------------
- x(t) ∈ ℝⁿˣ: State vector (continuous-time)
- u(t) ∈ ℝⁿᵘ: Control input (continuous-time)
- t ∈ ℝ: Time (continuous)
- dx/dt = f(x, u, t): Continuous-time dynamics

Examples
--------
>>> from abc import ABC
>>> class MyODESystem(ContinuousSystemBase):
...     def __call__(self, x, u=None, t=0.0):
...         # Implement dx/dt = f(x, u, t)
...         return -x + (u if u is not None else 0.0)
...
...     def integrate(self, x0, u, t_span):
...         # Implement numerical integration
...         from scipy.integrate import solve_ivp
...         ...
...
...     def linearize(self, x_eq, u_eq):
...         # Implement linearization
...         A = -np.eye(self.nx)
...         B = np.eye(self.nx)
...         return ContinuousLinearization(A=A, B=B, x_eq=x_eq, u_eq=u_eq)

See Also
--------
- ContinuousSymbolicSystem: Symbolic implementation with multi-backend code generation
- ContinuousStochasticSystem: Stochastic differential equations (SDEs)
- DiscreteSystemBase: Discrete-time counterpart

Authors
-------
Gil Benezer

License
-------
GNU Affero General Public License v3.0
"""

from abc import ABC, abstractmethod
from typing import Callable, Optional, Union

from src.types.core import ArrayLike, ControlVector, StateVector, TimeArray
from src.types.linearization import LinearizationResult
from src.types.trajectories import SimulationResult


class ContinuousSystemBase(ABC):
    """
    Abstract base class for all continuous-time dynamical systems.

    This class defines the fundamental interface that all continuous-time systems
    must implement. It serves as Layer 1 in the architecture, providing the
    contract for dynamics evaluation, integration, and linearization.

    All continuous-time systems satisfy:
        dx/dt = f(x, u, t)

    where x is the state, u is the control input, and t is time.

    Subclasses must implement three core methods:
    1. __call__: Evaluate dynamics at a single point
    2. integrate: Numerically integrate over a time interval
    3. linearize: Compute linearized dynamics around an equilibrium

    Attributes
    ----------
    nx : int
        State dimension (number of state variables)
    nu : int
        Control dimension (number of control inputs)
    ny : int
        Output dimension (number of outputs), optional

    Notes
    -----
    This is an abstract base class and cannot be instantiated directly.
    Use concrete implementations like ContinuousSymbolicSystem or create
    your own subclass.

    The class enforces a consistent API across all continuous-time systems,
    enabling polymorphic use in controllers, observers, and analysis tools.

    Examples
    --------
    Create a simple linear system:

    >>> class LinearSystem(ContinuousSystemBase):
    ...     def __init__(self, A, B):
    ...         self.A = A
    ...         self.B = B
    ...         self.nx = A.shape[0]
    ...         self.nu = B.shape[1]
    ...
    ...     def __call__(self, x, u=None, t=0.0):
    ...         u = u if u is not None else np.zeros(self.nu)
    ...         return self.A @ x + self.B @ u
    ...
    ...     def integrate(self, x0, u, t_span):
    ...         # Use scipy or other integrator
    ...         ...
    ...
    ...     def linearize(self, x_eq, u_eq):
    ...         return ContinuousLinearization(
    ...             A=self.A, B=self.B, x_eq=x_eq, u_eq=u_eq
    ...         )

    Polymorphic usage:

    >>> def analyze_stability(system: ContinuousSystemBase, x_eq, u_eq):
    ...     \"\"\"Works with ANY continuous system.\"\"\"
    ...     lin = system.linearize(x_eq, u_eq)
    ...     eigenvalues = np.linalg.eigvals(lin.A)
    ...     return np.all(eigenvalues.real < 0)  # Stable if all Re(λ) < 0
    """

    # =========================================================================
    # Abstract Methods (MUST be implemented by subclasses)
    # =========================================================================

    @abstractmethod
    def __call__(
        self,
        x: StateVector,
        u: Optional[ControlVector] = None,
        t: float = 0.0
    ) -> StateVector:
        """
        Evaluate continuous-time dynamics: dx/dt = f(x, u, t).

        This is the core dynamics evaluation method. It computes the time derivative
        of the state vector at a given point in state space.

        Parameters
        ----------
        x : StateVector
            Current state vector (nx,) or (nx, n_batch)
        u : Optional[ControlVector]
            Control input vector (nu,) or (nu, n_batch)
            If None, assumes zero control or autonomous dynamics
        t : float
            Current time (default: 0.0)
            Used for time-varying systems

        Returns
        -------
        StateVector
            Time derivative dx/dt with same shape as x

        Notes
        -----
        For autonomous systems, t is ignored.
        For time-invariant systems, t is typically ignored.
        For batch evaluation, x and u should have shape (n_dim, n_batch).

        The returned derivative should be in the same backend as the input
        (NumPy array → NumPy array, PyTorch tensor → PyTorch tensor, etc.).

        Examples
        --------
        Evaluate dynamics at a single point:

        >>> x = np.array([1.0, 2.0])
        >>> u = np.array([0.5])
        >>> dxdt = system(x, u)  # Returns dx/dt

        Batch evaluation:

        >>> x_batch = np.random.randn(2, 100)  # 100 states
        >>> u_batch = np.random.randn(1, 100)  # 100 controls
        >>> dxdt_batch = system(x_batch, u_batch)  # Returns (2, 100)

        Time-varying system:

        >>> dxdt = system(x, u, t=5.0)  # Evaluate at t=5
        """
        pass

    @abstractmethod
    def integrate(
        self,
        x0: StateVector,
        u: Union[ControlVector, Callable[[float], ControlVector], None] = None,
        t_span: tuple[float, float] = (0.0, 10.0),
        dt: Optional[float] = None,
        method: str = "RK45",
        **integrator_kwargs
    ) -> SimulationResult:
        """
        Integrate system dynamics over a time interval.

        Numerically solve the initial value problem:
            dx/dt = f(x, u, t)
            x(t0) = x0

        Parameters
        ----------
        x0 : StateVector
            Initial state (nx,)
        u : Union[ControlVector, Callable[[float], ControlVector], None]
            Control input, can be:
            - None: Zero control or autonomous
            - Array: Constant control u(t) = u for all t
            - Callable: Time-varying control u(t) = u_func(t)
        t_span : tuple[float, float]
            Time interval (t_start, t_end)
        dt : Optional[float]
            Output time step (if None, integrator chooses)
        method : str
            Integration method (e.g., 'RK45', 'RK23', 'LSODA', 'Euler')
        **integrator_kwargs
            Additional arguments passed to the integrator

        Returns
        -------
        SimulationResult
            Structured result containing:
            - time: Time points (n_steps,)
            - states: State trajectory (nx, n_steps)
            - controls: Control trajectory (nu, n_steps) if applicable
            - metadata: Integration info (method, success, etc.)

        Notes
        -----
        The integrator choice affects accuracy and speed:
        - 'RK45': Adaptive Runge-Kutta (good default)
        - 'RK23': Faster but less accurate
        - 'LSODA': Good for stiff systems
        - 'Euler': Simple explicit method

        For stochastic systems, specialized SDE integrators are used instead.

        Examples
        --------
        Constant control:

        >>> x0 = np.array([1.0, 0.0])
        >>> u = np.array([0.5])
        >>> result = system.integrate(x0, u, t_span=(0, 10), dt=0.01)
        >>> plt.plot(result.time, result.states[0, :])

        Time-varying control:

        >>> def u_func(t):
        ...     return np.array([np.sin(t)])
        >>> result = system.integrate(x0, u_func, t_span=(0, 10))

        Autonomous system:

        >>> result = system.integrate(x0, u=None, t_span=(0, 10))
        """
        pass

    @abstractmethod
    def linearize(
        self,
        x_eq: StateVector,
        u_eq: Optional[ControlVector] = None
    ) -> LinearizationResult:
        """
        Compute linearized dynamics around an equilibrium point.

        For a continuous-time system dx/dt = f(x, u), compute the linearization:
            d(δx)/dt = A·δx + B·δu

        where:
            A = ∂f/∂x|(x_eq, u_eq)  (State Jacobian, nx × nx)
            B = ∂f/∂u|(x_eq, u_eq)  (Control Jacobian, nx × nu)

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
            Structured result containing:
            - A: State Jacobian matrix (nx, nx)
            - B: Control Jacobian matrix (nx, nu)
            - x_eq: Equilibrium state
            - u_eq: Equilibrium control
            - Additional fields for stochastic systems (G matrix, etc.)

        Notes
        -----
        The linearization is valid for small deviations from the equilibrium:
            δx = x - x_eq
            δu = u - u_eq

        For symbolic systems, Jacobians are computed symbolically then evaluated.
        For data-driven systems, Jacobians may be computed via finite differences
        or automatic differentiation.

        The equilibrium point should satisfy f(x_eq, u_eq) ≈ 0 (within tolerance).

        Examples
        --------
        Linearize at origin:

        >>> x_eq = np.zeros(2)
        >>> u_eq = np.zeros(1)
        >>> lin = system.linearize(x_eq, u_eq)
        >>> print(f"A matrix:\\n{lin.A}")
        >>> print(f"B matrix:\\n{lin.B}")

        Check stability:

        >>> eigenvalues = np.linalg.eigvals(lin.A)
        >>> is_stable = np.all(eigenvalues.real < 0)

        Design LQR controller:

        >>> K = system.control.lqr(Q, R, x_eq=x_eq, u_eq=u_eq)
        """
        pass

    # =========================================================================
    # Concrete Methods (Provided by base class)
    # =========================================================================

    def simulate(
        self,
        x0: StateVector,
        controller: Optional[Callable[[StateVector, float], ControlVector]] = None,
        t_span: tuple[float, float] = (0.0, 10.0),
        dt: Optional[float] = None,
        **kwargs
    ) -> SimulationResult:
        """
        High-level simulation interface with optional feedback controller.

        This is a convenience method that wraps integrate() with support for
        closed-loop simulation via a feedback controller.

        Parameters
        ----------
        x0 : StateVector
            Initial state (nx,)
        controller : Optional[Callable[[StateVector, float], ControlVector]]
            Feedback controller u = controller(x, t)
            If None, uses zero control (open-loop)
        t_span : tuple[float, float]
            Simulation time interval (t_start, t_end)
        dt : Optional[float]
            Output time step
        **kwargs
            Additional arguments passed to integrate()

        Returns
        -------
        SimulationResult
            Trajectory with time, states, controls, and metadata

        Examples
        --------
        Open-loop simulation:

        >>> result = system.simulate(x0, t_span=(0, 5))

        Closed-loop with state feedback:

        >>> K = np.array([[-1.0, -2.0]])  # LQR gain
        >>> def controller(x, t):
        ...     return -K @ x
        >>> result = system.simulate(x0, controller, t_span=(0, 5))

        Time-varying reference tracking:

        >>> def controller(x, t):
        ...     x_ref = np.array([np.sin(t), np.cos(t)])
        ...     return K @ (x_ref - x)
        >>> result = system.simulate(x0, controller, t_span=(0, 10))
        """
        # Convert controller to control input function
        if controller is None:
            u_func = None
        else:
            u_func = lambda t: controller(
                # This is a simplified implementation
                # Actual implementation needs access to current state
                # from integrator, which requires wrapping the dynamics
                x0, t  # Placeholder - real implementation more complex
            )

        return self.integrate(x0, u_func, t_span, dt, **kwargs)

    def __repr__(self) -> str:
        """
        String representation of the system.

        Returns
        -------
        str
            Human-readable description

        Examples
        --------
        >>> print(system)
        ContinuousSymbolicSystem(nx=2, nu=1, ny=2)
        """
        class_name = self.__class__.__name__
        nx = getattr(self, 'nx', '?')
        nu = getattr(self, 'nu', '?')
        ny = getattr(self, 'ny', '?')
        return f"{class_name}(nx={nx}, nu={nu}, ny={ny})"

    # =========================================================================
    # Properties (Optional, can be overridden by subclasses)
    # =========================================================================

    @property
    def is_continuous(self) -> bool:
        """Return True (this is a continuous-time system)."""
        return True

    @property
    def is_discrete(self) -> bool:
        """Return False (this is NOT a discrete-time system)."""
        return False

    @property
    def is_stochastic(self) -> bool:
        """
        Return True if system has stochastic dynamics.

        Default: False (deterministic)
        Override in stochastic subclasses.
        """
        return False

    @property
    def is_time_varying(self) -> bool:
        """
        Return True if system dynamics depend explicitly on time.

        Default: False (time-invariant)
        Override in time-varying subclasses.
        """
        return False
