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
Discrete System Base Class (Layer 1)
====================================

This module provides the abstract base class for all discrete-time dynamical systems.

Overview
--------
The DiscreteSystemBase class defines the core interface that all discrete-time
systems must implement, regardless of whether they are deterministic or stochastic,
symbolic or data-driven.

All discrete-time systems share these fundamental operations:
- State update: x[k+1] = f(x[k], u[k], k)
- Multi-step simulation over discrete time steps
- Linearization around equilibrium points

This base class enforces these contracts through abstract methods, ensuring
consistent APIs across the entire discrete systems hierarchy.

Architecture Position
--------------------
Layer 1 (Abstract Interfaces):
    ContinuousSystemBase
    DiscreteSystemBase ← YOU ARE HERE

Layer 2 (Symbolic Implementations):
    ContinuousSymbolicSystem(ContinuousSystemBase)
    DiscreteSymbolicSystem(DiscreteSystemBase)

Layer 3 (Combined Implementations):
    DiscreteStochasticSystem(DiscreteSymbolicSystem, DiscreteSystemBase)

Layer 4 (Bridges):
    DiscreteTimeWrapper
    Discretizer (converts continuous → discrete)

Key Design Principles
--------------------
1. **Time Domain**: Discrete-time only (difference equations)
2. **Backend Agnostic**: Works with NumPy, PyTorch, JAX
3. **Minimal Interface**: Only essential methods are abstract
4. **Composable**: Designed for inheritance and extension
5. **Type Safe**: Full type hints using src/types

Abstract Methods
---------------
Subclasses MUST implement:
- step(x, u, k): Single time step update
- simulate(x0, u_sequence, n_steps): Multi-step simulation
- linearize(x_eq, u_eq): Compute linearization

Abstract Properties
------------------
Subclasses MUST provide:
- dt: Time step (sampling period)

Concrete Methods
---------------
Provided by default:
- rollout(): Alias for simulate with clearer name
- __repr__(): String representation

Mathematical Notation
--------------------
- x[k] ∈ ℝⁿˣ: State at discrete time k
- u[k] ∈ ℝⁿᵘ: Control input at time k
- k ∈ ℤ₊: Discrete time index
- x[k+1] = f(x[k], u[k], k): Discrete-time dynamics
- Δt: Sampling period (dt property)

Examples
--------
>>> class MyDiscreteSystem(DiscreteSystemBase):
...     def __init__(self, dt=0.1):
...         self._dt = dt
...         self.nx = 2
...         self.nu = 1
...
...     @property
...     def dt(self):
...         return self._dt
...
...     def step(self, x, u=None, k=0):
...         # Implement x[k+1] = f(x[k], u[k])
...         u = u if u is not None else np.zeros(self.nu)
...         return x + self.dt * (-x + u)
...
...     def simulate(self, x0, u_sequence, n_steps):
...         # Implement multi-step simulation
...         ...
...
...     def linearize(self, x_eq, u_eq):
...         # Implement linearization
...         Ad = np.eye(self.nx) - self.dt * np.eye(self.nx)
...         Bd = self.dt * np.eye(self.nx, self.nu)
...         return DiscreteLinearization(Ad=Ad, Bd=Bd, x_eq=x_eq, u_eq=u_eq)

See Also
--------
- DiscreteSymbolicSystem: Symbolic implementation from discretized continuous systems
- DiscreteStochasticSystem: Stochastic difference equations
- ContinuousSystemBase: Continuous-time counterpart
- Discretizer: Convert continuous systems to discrete

Authors
-------
Gil Benezer

License
-------
GNU Affero General Public License v3.0
"""

from abc import ABC, abstractmethod
from typing import Callable, Optional, Sequence, Union

import numpy as np

from src.types.core import ArrayLike, ControlVector, StateVector
from src.types.linearization import DiscreteLinearization, LinearizationResult
from src.types.trajectories import DiscreteSimulationResult


class DiscreteSystemBase(ABC):
    """
    Abstract base class for all discrete-time dynamical systems.

    This class defines the fundamental interface that all discrete-time systems
    must implement. It serves as Layer 1 in the architecture, providing the
    contract for state updates, simulation, and linearization.

    All discrete-time systems satisfy:
        x[k+1] = f(x[k], u[k], k)

    where x[k] is the state at time step k, u[k] is the control input, and k
    is the discrete time index.

    Subclasses must implement three core methods and one property:
    1. step: Update state by one time step
    2. simulate: Run multi-step simulation
    3. linearize: Compute linearized dynamics around an equilibrium
    4. dt (property): Sampling period / time step

    Attributes
    ----------
    nx : int
        State dimension (number of state variables)
    nu : int
        Control dimension (number of control inputs)
    ny : int
        Output dimension (number of outputs), optional
    dt : float
        Sampling period / time step (abstract property)

    Notes
    -----
    This is an abstract base class and cannot be instantiated directly.
    Use concrete implementations like DiscreteSymbolicSystem or create
    your own subclass.

    The class enforces a consistent API across all discrete-time systems,
    enabling polymorphic use in controllers, observers, and analysis tools.

    Discrete systems are often created by discretizing continuous systems
    using methods like Euler, Runge-Kutta, or exact discretization.

    Examples
    --------
    Create a simple discrete linear system:

    >>> class DiscreteLinearSystem(DiscreteSystemBase):
    ...     def __init__(self, Ad, Bd, dt):
    ...         self.Ad = Ad
    ...         self.Bd = Bd
    ...         self._dt = dt
    ...         self.nx = Ad.shape[0]
    ...         self.nu = Bd.shape[1]
    ...
    ...     @property
    ...     def dt(self):
    ...         return self._dt
    ...
    ...     def step(self, x, u=None, k=0):
    ...         u = u if u is not None else np.zeros(self.nu)
    ...         return self.Ad @ x + self.Bd @ u
    ...
    ...     def simulate(self, x0, u_sequence, n_steps):
    ...         states = [x0]
    ...         x = x0
    ...         for k in range(n_steps):
    ...             u = u_sequence[k] if u_sequence is not None else None
    ...             x = self.step(x, u, k)
    ...             states.append(x)
    ...         return DiscreteSimulationResult(
    ...             states=np.array(states).T,
    ...             controls=u_sequence,
    ...             time_steps=np.arange(n_steps + 1),
    ...             dt=self.dt
    ...         )
    ...
    ...     def linearize(self, x_eq, u_eq):
    ...         return DiscreteLinearization(
    ...             Ad=self.Ad, Bd=self.Bd, x_eq=x_eq, u_eq=u_eq, dt=self.dt
    ...         )

    Polymorphic usage:

    >>> def check_discrete_stability(system: DiscreteSystemBase, x_eq, u_eq):
    ...     \"\"\"Works with ANY discrete system.\"\"\"
    ...     lin = system.linearize(x_eq, u_eq)
    ...     eigenvalues = np.linalg.eigvals(lin.Ad)
    ...     return np.all(np.abs(eigenvalues) < 1)  # Stable if all |λ| < 1
    """

    # =========================================================================
    # Abstract Properties (MUST be implemented by subclasses)
    # =========================================================================

    @property
    @abstractmethod
    def dt(self) -> float:
        """
        Sampling period / time step of the discrete system.

        This is the time interval between consecutive state updates:
            t[k+1] = t[k] + dt

        Returns
        -------
        float
            Time step in seconds

        Notes
        -----
        For systems created by discretizing continuous systems, dt is the
        discretization time step.

        For naturally discrete systems (e.g., from data), dt represents
        the sampling period of the measurements.

        The time step must be positive and finite.

        Examples
        --------
        >>> print(f"System time step: {system.dt} seconds")
        System time step: 0.01 seconds

        >>> # Convert to frequency
        >>> freq = 1.0 / system.dt
        >>> print(f"Sampling frequency: {freq} Hz")
        Sampling frequency: 100.0 Hz
        """
        pass

    # =========================================================================
    # Abstract Methods (MUST be implemented by subclasses)
    # =========================================================================

    @abstractmethod
    def step(
        self,
        x: StateVector,
        u: Optional[ControlVector] = None,
        k: int = 0
    ) -> StateVector:
        """
        Compute next state: x[k+1] = f(x[k], u[k], k).

        This is the core state update method. It computes the next state
        given the current state and control input.

        Parameters
        ----------
        x : StateVector
            Current state vector (nx,) or (nx, n_batch)
        u : Optional[ControlVector]
            Control input vector (nu,) or (nu, n_batch)
            If None, assumes zero control or autonomous dynamics
        k : int
            Current discrete time step (default: 0)
            Used for time-varying systems

        Returns
        -------
        StateVector
            Next state x[k+1] with same shape as x

        Notes
        -----
        For autonomous systems, k is ignored.
        For time-invariant systems, k is typically ignored.
        For batch evaluation, x and u should have shape (n_dim, n_batch).

        The returned state should be in the same backend as the input
        (NumPy array → NumPy array, PyTorch tensor → PyTorch tensor, etc.).

        Examples
        --------
        Single step update:

        >>> x = np.array([1.0, 2.0])
        >>> u = np.array([0.5])
        >>> x_next = system.step(x, u)

        Batch evaluation:

        >>> x_batch = np.random.randn(2, 100)  # 100 states
        >>> u_batch = np.random.randn(1, 100)  # 100 controls
        >>> x_next_batch = system.step(x_batch, u_batch)

        Manual simulation loop:

        >>> x = x0
        >>> for k in range(100):
        ...     u = controller(x, k)
        ...     x = system.step(x, u, k)
        ...     # Log or visualize x
        """
        pass

    @abstractmethod
    def simulate(
        self,
        x0: StateVector,
        u_sequence: Optional[Union[ControlVector, Sequence[ControlVector], 
                                   Callable[[int], ControlVector]]] = None,
        n_steps: int = 100,
        **kwargs
    ) -> DiscreteSimulationResult:
        """
        Simulate system for multiple discrete time steps.

        Run the discrete dynamics forward in time:
            x[0] = x0
            x[k+1] = f(x[k], u[k], k)  for k = 0, 1, ..., n_steps-1

        Parameters
        ----------
        x0 : StateVector
            Initial state (nx,)
        u_sequence : Optional[Union[ControlVector, Sequence, Callable]]
            Control input sequence, can be:
            - None: Zero control for all steps
            - Array (nu,): Constant control u[k] = u for all k
            - Sequence: Pre-computed sequence u[0], u[1], ..., u[n_steps-1]
            - Callable: Control policy u[k] = u_func(k)
        n_steps : int
            Number of simulation steps (default: 100)
        **kwargs
            Additional simulation options (e.g., save_intermediate)

        Returns
        -------
        DiscreteSimulationResult
            Structured result containing:
            - states: State trajectory (nx, n_steps+1)
            - controls: Control sequence (nu, n_steps) if applicable
            - time_steps: Time step indices [0, 1, ..., n_steps]
            - dt: Sampling period
            - metadata: Additional info

        Notes
        -----
        The state trajectory includes n_steps+1 points (including x0).
        The control sequence has n_steps points (one for each transition).

        For closed-loop simulation, use a callable u_sequence that
        depends on the current state (see examples).

        Examples
        --------
        Open-loop with constant control:

        >>> x0 = np.array([1.0, 0.0])
        >>> u = np.array([0.5])
        >>> result = system.simulate(x0, u, n_steps=100)
        >>> plt.plot(result.time_steps, result.states[0, :])

        Pre-computed control sequence:

        >>> u_seq = [np.array([0.5 * np.sin(k * 0.1)]) for k in range(100)]
        >>> result = system.simulate(x0, u_seq, n_steps=100)

        State feedback control:

        >>> def controller(k):
        ...     # Access current state through closure or use step() manually
        ...     return -K @ x_current  # LQR control
        >>> result = system.simulate(x0, controller, n_steps=100)

        Time-varying reference tracking:

        >>> def controller(k):
        ...     x_ref = np.array([np.sin(k * system.dt), 
        ...                       np.cos(k * system.dt)])
        ...     return K @ (x_ref - x_current)
        >>> result = system.simulate(x0, controller, n_steps=100)
        """
        pass

    @abstractmethod
    def linearize(
        self,
        x_eq: StateVector,
        u_eq: Optional[ControlVector] = None
    ) -> DiscreteLinearization:
        """
        Compute linearized discrete dynamics around an equilibrium point.

        For a discrete system x[k+1] = f(x[k], u[k]), compute the linearization:
            δx[k+1] = Ad·δx[k] + Bd·δu[k]

        where:
            Ad = ∂f/∂x|(x_eq, u_eq)  (State Jacobian, nx × nx)
            Bd = ∂f/∂u|(x_eq, u_eq)  (Control Jacobian, nx × nu)

        Parameters
        ----------
        x_eq : StateVector
            Equilibrium state (nx,)
        u_eq : Optional[ControlVector]
            Equilibrium control (nu,)
            If None, uses zero control

        Returns
        -------
        DiscreteLinearization
            Structured result containing:
            - Ad: Discrete state Jacobian matrix (nx, nx)
            - Bd: Discrete control Jacobian matrix (nx, nu)
            - x_eq: Equilibrium state
            - u_eq: Equilibrium control
            - dt: Sampling period
            - Additional fields for stochastic systems

        Notes
        -----
        The linearization is valid for small deviations from the equilibrium:
            δx[k] = x[k] - x_eq
            δu[k] = u[k] - u_eq

        For symbolic systems, Jacobians are computed symbolically then evaluated.
        For data-driven systems, Jacobians may be computed via finite differences.

        The equilibrium point should satisfy f(x_eq, u_eq) = x_eq (fixed point).

        Stability analysis:
        - Stable if all |eigenvalues(Ad)| < 1
        - Unstable if any |eigenvalue(Ad)| > 1
        - Marginal if |eigenvalue(Ad)| = 1

        Examples
        --------
        Linearize at origin:

        >>> x_eq = np.zeros(2)
        >>> u_eq = np.zeros(1)
        >>> lin = system.linearize(x_eq, u_eq)
        >>> print(f"Ad matrix:\\n{lin.Ad}")
        >>> print(f"Bd matrix:\\n{lin.Bd}")

        Check discrete stability:

        >>> eigenvalues = np.linalg.eigvals(lin.Ad)
        >>> is_stable = np.all(np.abs(eigenvalues) < 1)
        >>> print(f"System stable: {is_stable}")

        Design discrete LQR controller:

        >>> from scipy.linalg import solve_discrete_are
        >>> P = solve_discrete_are(lin.Ad, lin.Bd, Q, R)
        >>> K = np.linalg.inv(R + lin.Bd.T @ P @ lin.Bd) @ (lin.Bd.T @ P @ lin.Ad)

        Relationship to continuous linearization:

        >>> # For Euler discretization: Ad ≈ I + dt * A
        >>> dt = system.dt
        >>> A_approx = (lin.Ad - np.eye(system.nx)) / dt
        """
        pass

    # =========================================================================
    # Concrete Methods (Provided by base class)
    # =========================================================================

    def rollout(
        self,
        x0: StateVector,
        policy: Optional[Callable[[StateVector, int], ControlVector]] = None,
        n_steps: int = 100,
        **kwargs
    ) -> DiscreteSimulationResult:
        """
        Rollout system trajectory with optional state-feedback policy.

        This is an alias for simulate() but with a clearer API for closed-loop
        simulation with state-dependent control policies.

        Parameters
        ----------
        x0 : StateVector
            Initial state (nx,)
        policy : Optional[Callable[[StateVector, int], ControlVector]]
            Control policy u = policy(x, k)
            If None, uses zero control (open-loop)
        n_steps : int
            Number of simulation steps
        **kwargs
            Additional arguments passed to simulate()

        Returns
        -------
        DiscreteSimulationResult
            Trajectory with states, controls, and metadata

        Examples
        --------
        Open-loop rollout:

        >>> result = system.rollout(x0, n_steps=100)

        State feedback policy:

        >>> K = np.array([[-1.0, -2.0]])  # LQR gain
        >>> def policy(x, k):
        ...     return -K @ x
        >>> result = system.rollout(x0, policy, n_steps=100)

        Time-varying policy with reference:

        >>> x_ref_trajectory = generate_reference()
        >>> def policy(x, k):
        ...     x_ref = x_ref_trajectory[k]
        ...     return K @ (x_ref - x)
        >>> result = system.rollout(x0, policy, n_steps=100)

        MPC policy:

        >>> mpc_controller = system.control.mpc(horizon=10, Q=Q, R=R)
        >>> def policy(x, k):
        ...     return mpc_controller.compute_control(x, k)
        >>> result = system.rollout(x0, policy, n_steps=100)
        """
        # Implementation: manually iterate with state feedback
        if policy is None:
            return self.simulate(x0, u_sequence=None, n_steps=n_steps, **kwargs)

        # Manual rollout with state-dependent control
        states = [x0]
        controls = []
        x = x0

        for k in range(n_steps):
            u = policy(x, k)
            controls.append(u)
            x = self.step(x, u, k)
            states.append(x)

        states_array = np.array(states).T  # Shape: (nx, n_steps+1)
        controls_array = np.array(controls).T if controls else None  # (nu, n_steps)

        return DiscreteSimulationResult(
            states=states_array,
            controls=controls_array,
            time_steps=np.arange(n_steps + 1),
            dt=self.dt,
            metadata={'method': 'rollout', 'closed_loop': True}
        )

    def __repr__(self) -> str:
        """
        String representation of the discrete system.

        Returns
        -------
        str
            Human-readable description

        Examples
        --------
        >>> print(system)
        DiscreteSymbolicSystem(nx=2, nu=1, dt=0.01)
        """
        class_name = self.__class__.__name__
        nx = getattr(self, 'nx', '?')
        nu = getattr(self, 'nu', '?')
        dt = getattr(self, 'dt', '?')
        return f"{class_name}(nx={nx}, nu={nu}, dt={dt})"

    # =========================================================================
    # Properties (Optional, can be overridden by subclasses)
    # =========================================================================

    @property
    def is_continuous(self) -> bool:
        """Return False (this is NOT a continuous-time system)."""
        return False

    @property
    def is_discrete(self) -> bool:
        """Return True (this is a discrete-time system)."""
        return True

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
        Return True if system dynamics depend explicitly on time step k.

        Default: False (time-invariant)
        Override in time-varying subclasses.
        """
        return False

    @property
    def sampling_frequency(self) -> float:
        """
        Get sampling frequency in Hz.

        Returns
        -------
        float
            Frequency = 1 / dt

        Examples
        --------
        >>> print(f"Sampling rate: {system.sampling_frequency} Hz")
        Sampling rate: 100.0 Hz
        """
        return 1.0 / self.dt
