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

Abstract base class for all continuous-time dynamical systems.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

import numpy as np

from src.types.backends import Backend, IntegrationMethod
from src.types.core import ControlInput, ControlVector, FeedbackController, ScalarLike, StateVector
from src.types.linearization import LinearizationResult
from src.types.trajectories import IntegrationResult, SimulationResult, TimeSpan

if TYPE_CHECKING:
    import plotly.graph_objects as go

    from src.control.control_synthesis import ControlSynthesis
    from src.control.system_analysis import SystemAnalysis
    from src.visualization.control_plots import ControlPlotter
    from src.visualization.phase_portrait import PhasePortraitPlotter
    from src.visualization.trajectory_plotter import TrajectoryPlotter


class ContinuousSystemBase(ABC):
    """
    Abstract base class for all continuous-time dynamical systems.

    This class defines the fundamental interface that all continuous-time systems
    must implement. All continuous-time systems satisfy:
        dx/dt = f(x, u, t)

    Subclasses must implement:
    1. __call__(x, u, t): Evaluate dynamics at a point
    2. integrate(x0, u, t_span): Low-level numerical integration with solver diagnostics
    3. linearize(x_eq, u_eq): Compute linearization

    Additional concrete methods provided:
    - simulate(): High-level simulation with regular time grid (wraps integrate())

    Examples
    --------
    >>> class MyODESystem(ContinuousSystemBase):
    ...     def __call__(self, x, u=None, t=0.0):
    ...         return -x + (u if u is not None else 0.0)
    ...
    ...     def integrate(self, x0, u, t_span, method="RK45", **kwargs):
    ...         # Use scipy.integrate.solve_ivp or similar
    ...         result = solve_ivp(...)
    ...         return {
    ...             "t": result.t,
    ...             "y": result.y,
    ...             "success": result.success,
    ...             "nfev": result.nfev,
    ...             ...
    ...         }
    ...
    ...     def linearize(self, x_eq, u_eq):
    ...         A = -np.eye(self.nx)
    ...         B = np.eye(self.nx, self.nu)
    ...         return (A, B)
    """

    # =========================================================================
    # Abstract Methods (MUST be implemented by subclasses)
    # =========================================================================

    @abstractmethod
    def __call__(
        self,
        x: StateVector,
        u: Optional[ControlVector] = None,
        t: ScalarLike = 0.0,
        backend: Optional[Backend] = None,
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
        - For autonomous systems, t is ignored
        - For time-invariant systems, t is typically ignored
        - For batch evaluation, x and u should have shape (n_dim, n_batch)
        - The returned derivative should be in the same backend as the input

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
        """

    @abstractmethod
    def integrate(
        self,
        x0: StateVector,
        u: ControlInput = None,
        t_span: TimeSpan = (0.0, 10.0),
        method: IntegrationMethod = "RK45",
        **integrator_kwargs,
    ) -> IntegrationResult:
        """
        Low-level numerical integration with ODE solver diagnostics.

        Numerically solve the initial value problem:
            dx/dt = f(x, u, t)
            x(t0) = x0

        **API Level**: This is a **low-level integration method** that exposes raw solver
        output including adaptive time points, convergence information, and performance
        metrics. For typical use cases, prefer `simulate()` which provides a cleaner
        interface with regular time grids and the intuitive (x, t) controller convention.

        **Control Input Handling**: This method accepts flexible control input formats
        and automatically converts them to the internal (t, x) → u function signature
        expected by numerical solvers. You can provide:
        - None for autonomous/zero control
        - Arrays for constant control
        - Functions with various signatures (see below)

        The conversion to solver convention is handled internally - you don't need to
        worry about the (t, x) vs (x, t) distinction at this level.

        Parameters
        ----------
        x0 : StateVector
            Initial state (nx,)
        u : Union[ControlVector, Callable[[float], ControlVector], None]
            Control input in flexible formats:

            - **None**: Zero control or autonomous system
            - **Array (nu,)**: Constant control u(t) = u_const for all t
            - **Callable u(t)**: Time-varying control, signature: t → u
            - **Callable u(t, x)**: State-feedback control (auto-detected)
            - **Callable u(x, t)**: State-feedback control (auto-detected)

            The method will automatically detect the function signature and convert
            to the internal (t, x) → u convention used by solvers. For callables with
            two parameters, it attempts to detect the order by inspecting parameter
            names or testing with dummy values.

            **Standard Convention**: Functions with two parameters should use (t, x) order
            to avoid ambiguity. If using (x, t) order, the wrapper will attempt detection
            but may fail on edge cases - prefer wrapping explicitly:
        ```python
            u_func = lambda t, x: my_controller(x, t)
        ```
        t_span : tuple[float, float]
            Time interval (t_start, t_end)
        method : str
            Integration method. Options:

        **integrator_kwargs
            Additional arguments passed to the ODE solver:

            - **dt** : float (required for fixed-step methods)
            - **rtol** : float (relative tolerance, default: 1e-6)
            - **atol** : float (absolute tolerance, default: 1e-8)
            - **max_steps** : int (maximum steps, default: 10000)
            - **t_eval** : ArrayLike (specific times to return solution)
            - **dense_output** : bool (return interpolant, default: False)
            - **first_step** : float (initial step size guess)
            - **max_step** : float (maximum step size)

        Returns
        -------
        IntegrationResult
            TypedDict containing:

            - **t**: Time points (T,) - adaptive, chosen by solver
            - **x** or **y**: State trajectory
            - scipy returns 'y' with shape (nx, T)
            - other backends return 'x' with shape (T, nx)
            - **success**: bool - whether integration succeeded
            - **message**: str - solver status message
            - **nfev**: int - number of function evaluations
            - **nsteps**: int - number of steps taken
            - **integration_time**: float - computation time (seconds)
            - **solver**: str - integrator name used
            - **njev**: int - number of Jacobian evaluations (if applicable)
            - **nlu**: int - number of LU decompositions (implicit methods)
            - **status**: int - termination status code

        Notes
        -----
        **Adaptive Time Points**: The time points in the result are chosen adaptively
        by the solver based on error control, NOT on a regular grid. This means:
        - Solver takes larger steps when dynamics are smooth
        - Solver takes smaller steps when dynamics change rapidly
        - Time points are NOT uniformly spaced

        For a regular time grid suitable for plotting or analysis, use `simulate()`
        instead, or provide `t_eval` parameter.

        **Backend Differences**:
        - **scipy**: Returns 'y' with shape (nx, T) - state-major
        - **torchdiffeq/diffrax**: Return 'x' with shape (T, nx) - time-major
        - This method handles both conventions automatically in `simulate()`

        Examples
        --------
        **Basic usage** - autonomous system:

        >>> x0 = np.array([1.0, 0.0])
        >>> result = system.integrate(x0, u=None, t_span=(0, 10))
        >>> print(f"Success: {result['success']}")
        >>> print(f"Function evaluations: {result['nfev']}")
        >>>
        >>> # Handle both scipy and other backends
        >>> if 'y' in result:
        >>>     trajectory = result['y']  # scipy: (nx, T)
        >>>     plt.plot(result['t'], trajectory[0, :])
        >>> else:
        >>>     trajectory = result['x']  # others: (T, nx)
        >>>     plt.plot(result['t'], trajectory[:, 0])

        **Constant control**:

        >>> u_const = np.array([0.5])
        >>> result = system.integrate(x0, u=u_const, t_span=(0, 10))

        **Time-varying control** - single parameter function:

        >>> def u_func(t):
        ...     return np.array([np.sin(t)])
        >>> result = system.integrate(x0, u=u_func, t_span=(0, 10))

        **State feedback** - two parameter function (auto-detected):

        >>> def u_func(t, x):
        ...     K = np.array([[1.0, 2.0]])
        ...     return -K @ x
        >>> result = system.integrate(x0, u=u_func, t_span=(0, 10))

        **Stiff system** with tight tolerances:

        >>> result = system.integrate(
        ...     x0,
        ...     u=None,
        ...     t_span=(0, 10),
        ...     method='Radau',
        ...     rtol=1e-8,
        ...     atol=1e-10
        ... )
        >>> print(f"Stiff solver steps: {result['nsteps']}")

        **High-accuracy Julia solver**:

        >>> result = system.integrate(
        ...     x0,
        ...     u=None,
        ...     t_span=(0, 10),
        ...     method='Vern9',  # Julia high-accuracy
        ...     rtol=1e-12,
        ...     atol=1e-14
        ... )

        **Regular time grid** for plotting:

        >>> t_eval = np.linspace(0, 10, 1001)  # 1001 points
        >>> result = system.integrate(
        ...     x0,
        ...     u=None,
        ...     t_span=(0, 10),
        ...     t_eval=t_eval
        ... )
        >>> assert len(result['t']) == 1001

        **Fixed-step integration** (RK4):

        >>> result = system.integrate(
        ...     x0,
        ...     u=None,
        ...     t_span=(0, 10),
        ...     method='rk4',
        ...     dt=0.01  # Required for fixed-step
        ... )

        **Check solver performance**:

        >>> result = system.integrate(x0, u=None, t_span=(0, 10))
        >>> if result['nfev'] > 10000:
        ...     print("⚠ Warning: Many function evaluations!")
        ...     print("Consider:")
        ...     print("  - Using stiff solver (Radau, BDF)")
        ...     print("  - Relaxing tolerances")
        ...     print("  - Checking for stiffness")

        **Dense output** (interpolation):

        >>> result = system.integrate(
        ...     x0,
        ...     u=None,
        ...     t_span=(0, 10),
        ...     dense_output=True
        ... )
        >>> if 'sol' in result:
        ...     # Evaluate at arbitrary times
        ...     t_fine = np.linspace(0, 10, 10000)
        ...     x_fine = result['sol'](t_fine)

        **Comparing backends**:

        >>> # NumPy (scipy)
        >>> result_np = system.integrate(x0, u=None, t_span=(0, 10), method='RK45')
        >>>
        >>> # Julia (DiffEqPy)
        >>> result_jl = system.integrate(x0, u=None, t_span=(0, 10), method='Tsit5')
        >>>
        >>> # JAX (diffrax)
        >>> system.set_default_backend('jax')
        >>> result_jax = system.integrate(x0, u=None, t_span=(0, 10), method='tsit5')

        **Error handling**:

        >>> try:
        ...     result = system.integrate(
        ...         x0,
        ...         u=None,
        ...         t_span=(0, 10),
        ...         method='RK45',
        ...         max_steps=100  # Very low limit
        ...     )
        ...     if not result['success']:
        ...         print(f"Integration failed: {result['message']}")
        ... except RuntimeError as e:
        ...     print(f"Runtime error: {e}")

        See Also
        --------
        simulate : High-level simulation with regular time grid (recommended)
        IntegratorFactory : Create custom integrators with specific methods
        linearize : Compute linearized dynamics at equilibrium
        """

    @abstractmethod
    def linearize(
        self,
        x_eq: StateVector,
        u_eq: Optional[ControlVector] = None,
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
            Tuple containing Jacobian matrices:
            - Deterministic systems: (A, B)
            - Stochastic systems: (A, B, G) where G is diffusion matrix

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
        >>> A, B = system.linearize(x_eq, u_eq)
        >>> print(f"A matrix:\\n{A}")
        >>> print(f"B matrix:\\n{B}")

        Check stability (continuous-time):

        >>> eigenvalues = np.linalg.eigvals(A)
        >>> is_stable = np.all(np.real(eigenvalues) < 0)
        >>> print(f"System stable: {is_stable}")

        Design LQR controller:

        >>> from scipy.linalg import solve_continuous_are
        >>> P = solve_continuous_are(A, B, Q, R)
        >>> K = np.linalg.inv(R) @ B.T @ P
        """

    # =========================================================================
    # Concrete Methods (Provided by base class)
    # =========================================================================

    def simulate(
        self,
        x0: StateVector,
        controller: Optional[FeedbackController] = None,
        t_span: TimeSpan = (0.0, 10.0),
        dt: ScalarLike = 0.01,
        method: IntegrationMethod = "RK45",
        **kwargs,
    ) -> SimulationResult:
        """
        High-level simulation interface with regular time grid.

        This method wraps integrate() and post-processes the result to provide
        a regular time grid and cleaner output. This is the recommended method
        for most use cases (currently, may be deprecated once DiscretizedSystem is
        developed).

        Parameters
        ----------
        x0 : StateVector
            Initial state (nx,)
        controller : Optional[Callable[[StateVector, float], ControlVector]]
            Feedback controller u = controller(x, t)
            **STANDARD CONVENTION**: State is primary argument, time is secondary
            This aligns with discrete systems' policy(x, k) signature
            If None, uses zero control (open-loop)
        t_span : tuple[float, float]
            Simulation time interval (t_start, t_end)
        dt : float
            Output time step for regular grid (default: 0.01)
        method : str
            Integration method passed to integrate()
        **kwargs
            Additional arguments passed to integrate()

        Returns
        -------
        SimulationResult
            TypedDict containing:
            - **time**: Time points array (T,) with uniform spacing dt
            - **states**: State trajectory (T, nx) - **TIME-MAJOR ordering**
            - **controls**: Control trajectory (T, nu) if controller provided
            - **metadata**: Additional information (method, dt, success, nfev)

        Notes
        -----
        **Time-Major Convention**: This method returns states in (T, nx) shape,
        which is the modern standard for time series data:
        - Compatible with pandas DataFrames
        - Natural for time-series analysis
        - Consistent with ML/data science conventions
        - Matches discrete systems' output format


        Unlike integrate(), this method:
        - Returns states on a regular time grid (not adaptive)
        - Supports state-feedback controllers
        - Hides solver diagnostics (cleaner output)
        - Is easier to use for plotting and analysis

        Controller Signature
        --------------------
        Controllers must have signature (x, t) -> u:
        - x: StateVector - current state (PRIMARY argument)
        - t: float - current time (secondary argument)
        - Returns: ControlVector - control input

        This matches the discrete systems' policy(x, k) convention where state
        is the primary argument. An internal adapter converts to scipy's (t, x)
        convention when calling integrate().

        Examples
        --------
        **Open-loop simulation**:

        >>> result = system.simulate(x0, t_span=(0, 5), dt=0.01)
        >>> # Time-major indexing
        >>> plt.plot(result["time"], result["states"][:, 0])  # First state
        >>> plt.plot(result["time"], result["states"][:, 1])  # Second state
        >>> plt.xlabel("Time (s)")
        >>> plt.ylabel("State")

        **Using with pandas** (natural with time-major):

        >>> import pandas as pd
        >>> result = system.simulate(x0, controller, t_span=(0, 10))
        >>> df = pd.DataFrame(
        ...     result["states"],
        ...     index=result["time"],
        ...     columns=[f"x{i}" for i in range(system.nx)]
        ... )
        >>> df.plot()

        **Closed-loop with state feedback**:

        >>> K = np.array([[-1.0, -2.0]])  # LQR gain
        >>> def controller(x, t):
        ...     return -K @ x
        >>> result = system.simulate(x0, controller, t_span=(0, 5))
        >>>
        >>> # Extract trajectory
        >>> t = result["time"]
        >>> x = result["states"]  # (T, nx)
        >>> u = result["controls"]  # (T, nu)

        **Time-varying reference tracking**:

        >>> def controller(x, t):
        ...     x_ref = np.array([np.sin(t), np.cos(t)])
        ...     K = np.array([[1.0, 0.5]])
        ...     return K @ (x_ref - x)
        >>> result = system.simulate(x0, controller, t_span=(0, 10))

        **Adaptive gain controller**:

        >>> def controller(x, t):
        ...     K = 1.0 + 0.1 * t  # Gain increases with time
        ...     return np.array([-K * x[0]])
        >>> result = system.simulate(x0, controller, t_span=(0, 10))

        **Saturated control**:

        >>> def controller(x, t):
        ...     u_raw = -2.0 * x[0] - 0.5 * x[1]
        ...     return np.array([np.clip(u_raw, -1.0, 1.0)])
        >>> result = system.simulate(x0, controller, t_span=(0, 10))

        **Time-only control** (uncommon, but supported):

        >>> def controller(x, t):
        ...     return np.array([np.sin(2 * np.pi * t)])
        >>> result = system.simulate(x0, controller, t_span=(0, 10))

        **Batch plotting multiple states**:

        >>> result = system.simulate(x0, controller, t_span=(0, 10))
        >>> fig, axes = plt.subplots(system.nx, 1, sharex=True)
        >>> for i, ax in enumerate(axes):
        ...     ax.plot(result["time"], result["states"][:, i])
        ...     ax.set_ylabel(f"$x_{i}$")
        >>> axes[-1].set_xlabel("Time (s)")

        **Phase portrait** (using time-major indexing):

        >>> result = system.simulate(x0, controller, t_span=(0, 10))
        >>> plt.plot(result["states"][:, 0], result["states"][:, 1])
        >>> plt.xlabel("$x_0$")
        >>> plt.ylabel("$x_1$")

        See Also
        --------
        integrate : Low-level integration with solver diagnostics
        rollout : Alternative name for simulation
        """

        # Create regular time grid
        t_regular = np.arange(t_span[0], t_span[1] + dt, dt)

        # Adapt controller signature from (x, t) to scipy's (t, x) if needed
        if controller is not None:

            def u_func_scipy(t, x):
                """Adapter: converts scipy's (t, x) to our (x, t) convention"""
                return controller(x, t)

            u_func = u_func_scipy
        else:
            u_func = None

        # Call low-level integrate() with regular time grid
        int_result = self.integrate(
            x0=x0,
            u=u_func,
            t_span=t_span,
            t_eval=t_regular,  # Request specific times
            method=method,
            **kwargs,
        )

        # Handle both 'x' and 'y' keys (different integrator conventions)
        if "x" in int_result:
            states_time_major = int_result["x"]  # (T, nx) - modern convention
        elif "y" in int_result:
            # scipy convention: (nx, T) - transpose to (T, nx)
            states_time_major = int_result["y"].T
        else:
            raise KeyError(
                "Integration result missing both 'x' and 'y' keys. "
                "This indicates an issue with the integrator backend.",
            )

        # If times don't exactly match requested grid, interpolate
        if not np.allclose(int_result["t"], t_regular, rtol=1e-10):
            # Need to interpolate to exact regular grid
            T, nx = states_time_major.shape
            states_interp = np.zeros((len(t_regular), nx))

            # Interpolate each state dimension
            for i in range(nx):
                states_interp[:, i] = np.interp(t_regular, int_result["t"], states_time_major[:, i])

            states_regular = states_interp  # (T, nx)
        else:
            states_regular = states_time_major  # (T, nx)

        # Reconstruct control trajectory if controller provided
        controls = None
        if controller is not None:
            # Evaluate controller at each time point
            controls_list = []
            for i, t in enumerate(t_regular):
                # Extract state at this time (states_regular is (T, nx))
                x_t = states_regular[i, :]

                # Call controller with (x, t) signature (our convention)
                u_t = controller(x_t, t)
                controls_list.append(u_t)

            controls = np.array(controls_list)  # (T, nu) - time-major

        return {
            "time": t_regular,
            "states": states_regular,  # (T, nx) - TIME-MAJOR
            "controls": controls,  # (T, nu) - TIME-MAJOR
            "success": int_result.get("success", True),
            "metadata": {
                "method": method,
                "dt": dt,
                "nfev": int_result.get("nfev", None),
                "integration_time": int_result.get("integration_time", None),
            },
        }

    def rollout(
        self,
        x0: StateVector,
        policy: Optional[FeedbackController] = None,
        t_span: TimeSpan = (0.0, 10.0),
        dt: ScalarLike = 0.01,
        method: IntegrationMethod = "RK45",
        **kwargs,
    ) -> SimulationResult:
        """
        Rollout system trajectory with optional state-feedback policy.

        This is an alias for simulate() that provides API consistency with discrete
        systems. The name "rollout" is commonly used in reinforcement learning and
        control theory for executing a policy over time.

        Parameters
        ----------
        x0 : StateVector
            Initial state (nx,)
        policy : Optional[Callable[[StateVector, float], ControlVector]]
            State-feedback policy u = policy(x, t)
            **STANDARD CONVENTION**: State is primary argument, time is secondary
            If None, uses zero control (open-loop)
        t_span : tuple[float, float]
            Simulation time interval (t_start, t_end)
        dt : float
            Output time step for regular grid (default: 0.01)
        method : str
            Integration method passed to integrate()
        **kwargs
            Additional arguments passed to integrate()

        Returns
        -------
        SimulationResult
            TypedDict containing:
            - **time**: Time points array (T,) with uniform spacing dt
            - **states**: State trajectory (T, nx) - TIME-MAJOR ordering
            - **controls**: Control trajectory (T, nu) if policy provided
            - **metadata**: Additional information with 'closed_loop' flag

        Notes
        -----
        **API Consistency**: This method provides the same interface as
        DiscreteSystemBase.rollout(), making it easier to work with both
        continuous and discrete systems using identical code patterns.

        The only difference from simulate() is:
        - Parameter name: "policy" instead of "controller" (same semantics)
        - Metadata includes 'closed_loop' flag for compatibility
        - Name emphasizes trajectory generation with state feedback

        **When to Use**:
        - Use rollout() when emphasizing policy execution (RL/control context)
        - Use simulate() for general-purpose simulation
        - Both methods are functionally equivalent for continuous systems

        Policy Signature
        ----------------
        Policies must have signature (x, t) -> u:
        - x: StateVector - current state (PRIMARY argument)
        - t: float - current time (secondary argument)
        - Returns: ControlVector - control input

        Examples
        --------
        **Open-loop rollout**:

        >>> result = system.rollout(x0, t_span=(0, 10), dt=0.01)
        >>> plt.plot(result["time"], result["states"][:, 0])

        **State feedback policy** (LQR):

        >>> K = np.array([[-1.0, -2.0]])  # LQR gain
        >>> def policy(x, t):
        ...     return -K @ x
        >>> result = system.rollout(x0, policy, t_span=(0, 10))

        **Time-varying policy with reference**:

        >>> x_ref_func = lambda t: np.array([np.sin(t), np.cos(t)])
        >>> def policy(x, t):
        ...     x_ref = x_ref_func(t)
        ...     K = np.array([[1.0, 0.5]])
        ...     return K @ (x_ref - x)
        >>> result = system.rollout(x0, policy, t_span=(0, 10))

        **Nonlinear policy** (e.g., neural network):

        >>> def neural_policy(x, t):
        ...     # Example: simple nonlinear policy
        ...     hidden = np.tanh(W1 @ x + b1)
        ...     u = W2 @ hidden + b2
        ...     return u
        >>> result = system.rollout(x0, neural_policy, t_span=(0, 5))

        **MPC-style receding horizon**:

        >>> def mpc_policy(x, t):
        ...     # Solve optimization at each step
        ...     u_opt = solve_mpc(x, horizon=10, Q=Q, R=R)
        ...     return u_opt
        >>> result = system.rollout(x0, mpc_policy, t_span=(0, 10), dt=0.1)

        **Comparing policies**:

        >>> policies = {
        ...     "LQR": lqr_policy,
        ...     "MPC": mpc_policy,
        ...     "Neural": neural_policy
        ... }
        >>>
        >>> results = {}
        >>> for name, policy in policies.items():
        ...     results[name] = system.rollout(x0, policy, t_span=(0, 10))
        ...
        >>> # Plot comparison
        >>> for name, result in results.items():
        ...     plt.plot(result["time"], result["states"][:, 0], label=name)
        >>> plt.legend()

        **Trajectory optimization context**:

        >>> # Generate initial trajectory
        >>> result = system.rollout(x0, initial_policy, t_span=(0, 5))
        >>>
        >>> # Extract trajectory for optimization
        >>> trajectory = result["states"]  # (T, nx)
        >>>
        >>> # Optimize policy
        >>> optimized_policy = optimize_policy(trajectory)
        >>>
        >>> # Re-rollout with optimized policy
        >>> final_result = system.rollout(x0, optimized_policy, t_span=(0, 5))

        **Reinforcement learning context**:

        >>> # Collect rollout for policy gradient
        >>> def stochastic_policy(x, t):
        ...     mu = policy_network(x)
        ...     u = mu + np.random.randn(*mu.shape) * sigma
        ...     return u
        >>>
        >>> rollouts = []
        >>> for episode in range(num_episodes):
        ...     result = system.rollout(x0, stochastic_policy, t_span=(0, 10))
        ...     reward = compute_reward(result["states"], result["controls"])
        ...     rollouts.append((result, reward))

        **Monte Carlo evaluation**:

        >>> # Evaluate policy robustness
        >>> results = []
        >>> for _ in range(100):
        ...     x0_perturbed = x0 + np.random.randn(len(x0)) * 0.1
        ...     result = system.rollout(x0_perturbed, policy, t_span=(0, 10))
        ...     results.append(result)
        >>>
        >>> # Analyze performance distribution
        >>> final_errors = [np.linalg.norm(r["states"][-1, :]) for r in results]
        >>> print(f"Mean final error: {np.mean(final_errors):.3f}")
        >>> print(f"Std final error: {np.std(final_errors):.3f}")

        **Using metadata closed_loop flag**:

        >>> result = system.rollout(x0, policy, t_span=(0, 10))
        >>> if result["metadata"]["closed_loop"]:
        ...     print("Closed-loop rollout with state feedback")
        ... else:
        ...     print("Open-loop rollout")

        See Also
        --------
        simulate : Equivalent method with "controller" parameter name
        integrate : Low-level integration with solver diagnostics
        DiscreteSystemBase.rollout : Discrete-time analog
        """
        # Call simulate() with same functionality
        result = self.simulate(
            x0=x0,
            controller=policy,
            t_span=t_span,
            dt=dt,
            method=method,
            **kwargs,
        )

        # Add closed_loop flag to metadata for consistency with discrete systems
        result["metadata"]["closed_loop"] = policy is not None
        result["metadata"]["method_type"] = "rollout"

        return result

    # =========================================================================
    # Control Framework Integration
    # =========================================================================

    @property
    def control(self) -> "ControlSynthesis":
        """
        Access control synthesis utilities.

        Provides classical control design algorithms including LQR, Kalman filter,
        and LQG for continuous-time systems.

        Returns
        -------
        ControlSynthesis
            Control design utilities with methods:
            - design_lqr(A, B, Q, R, N=None, system_type='continuous')
            - design_kalman(A, C, Q, R, system_type='continuous')
            - design_lqg(A, B, C, Q_state, R_control, Q_process, R_measurement, ...)

        Examples
        --------
        >>> # Linearize and design LQR
        >>> A, B = system.linearize(x_eq, u_eq)
        >>> Q = np.diag([10, 1])
        >>> R = np.array([[0.1]])
        >>> result = system.control.design_lqr(A, B, Q, R, system_type='continuous')
        >>> K = result['gain']
        >>>
        >>> # Design Kalman filter
        >>> C = np.array([[1, 0]])
        >>> kalman = system.control.design_kalman(
        ...     A, C, Q_proc=0.01*np.eye(2), R_meas=np.array([[0.1]]),
        ...     system_type='continuous'
        ... )
        >>> L = kalman['gain']

        See Also
        --------
        analysis : System analysis utilities
        design_lqr : Convenience method (linearizes + designs in one call)
        """
        if not hasattr(self, "_control_synthesis"):
            self._control_synthesis = None

        if self._control_synthesis is None:
            from src.control.control_synthesis import ControlSynthesis

            backend = getattr(self, "backend", None)
            if backend is not None and hasattr(backend, "default_backend"):
                backend_str = backend.default_backend
            else:
                backend_str = "numpy"

            self._control_synthesis = ControlSynthesis(backend=backend_str)

        return self._control_synthesis

    @property
    def analysis(self) -> "SystemAnalysis":
        """
        Access system analysis utilities.

        Provides stability, controllability, and observability analysis
        for continuous-time linear systems.

        Returns
        -------
        SystemAnalysis
            System analysis utilities with methods:
            - stability(A, system_type='continuous', tolerance=1e-10)
            - controllability(A, B, tolerance=1e-10)
            - observability(A, C, tolerance=1e-10)
            - analyze_linearization(A, B, C, system_type='continuous')

        Examples
        --------
        >>> # Check stability
        >>> A, B = system.linearize(x_eq, u_eq)
        >>> stability = system.analysis.stability(A, system_type='continuous')
        >>> print(f"Stable: {stability['is_stable']}")
        >>>
        >>> # Check controllability
        >>> ctrl_info = system.analysis.controllability(A, B)
        >>> if ctrl_info['is_controllable']:
        ...     result = system.design_lqr(Q, R)

        See Also
        --------
        control : Control synthesis utilities
        linearize : Compute linearization matrices
        """
        if not hasattr(self, "_system_analysis"):
            self._system_analysis = None

        if self._system_analysis is None:
            from src.control.system_analysis import SystemAnalysis

            backend = getattr(self, "backend", None)
            if backend is not None and hasattr(backend, "default_backend"):
                backend_str = backend.default_backend
            else:
                backend_str = "numpy"

            self._system_analysis = SystemAnalysis(backend=backend_str)

        return self._system_analysis

    # =========================================================================
    # Plotting Framework Integration
    # =========================================================================

    @property
    def plotter(self) -> "TrajectoryPlotter":
        """
        Access trajectory plotting utilities.

        Provides time-domain visualization for state and control trajectories
        with automatic handling of batched data and backend conversion.

        Returns
        -------
        TrajectoryPlotter
            Trajectory plotting utilities with methods:
            - plot_trajectory(t, x, u=None, state_names=None)
            - plot_state_and_control(t, x, u, state_names=None)
            - plot_comparison(t, trajectories, state_names=None)

        Examples
        --------
        >>> # Plot integration result
        >>> result = system.integrate(x0, u=None, t_span=(0, 10))
        >>> fig = system.plotter.plot_trajectory(
        ...     result['t'],
        ...     result['x'],
        ...     state_names=['Position', 'Velocity']
        ... )
        >>> fig.show()
        >>>
        >>> # With control inputs
        >>> fig = system.plotter.plot_state_and_control(
        ...     result['t'],
        ...     result['x'],
        ...     result['u']
        ... )
        >>>
        >>> # Compare multiple simulations
        >>> trajectories = {
        ...     'Controlled': result_controlled['x'],
        ...     'Uncontrolled': result_uncontrolled['x'],
        ... }
        >>> fig = system.plotter.plot_comparison(result['t'], trajectories)
        >>>
        >>> # Or use convenience method
        >>> fig = system.plot(result, state_names=['θ', 'ω'])

        See Also
        --------
        phase_plotter : Phase space visualization
        control_plotter : Control analysis plots
        plot : Convenience method

        Notes
        -----
        - Automatically converts PyTorch/JAX arrays to NumPy for plotting
        - Handles batched trajectories for Monte Carlo simulations
        - Adaptive subplot layout based on number of states
        """
        if not hasattr(self, "_trajectory_plotter"):
            self._trajectory_plotter = None

        if self._trajectory_plotter is None:
            from src.visualization.trajectory_plotter import TrajectoryPlotter

            # Safe backend access (works for symbolic and non-symbolic systems)
            backend = getattr(self, "backend", None)
            if backend is not None and hasattr(backend, "default_backend"):
                backend_str = backend.default_backend
            else:
                backend_str = "numpy"

            self._trajectory_plotter = TrajectoryPlotter(backend=backend_str)

        return self._trajectory_plotter

    @property
    def phase_plotter(self) -> "PhasePortraitPlotter":
        """
        Access phase portrait plotting utilities.

        Provides state space visualization including 2D/3D phase portraits,
        vector fields, and equilibrium point markers.

        Returns
        -------
        PhasePortraitPlotter
            Phase space plotting utilities with methods:
            - plot_2d(x, state_names=None, vector_field=None, equilibria=None)
            - plot_3d(x, state_names=None, show_direction=True)
            - plot_limit_cycle(x, state_names=None, period_estimate=None)

        Examples
        --------
        >>> # 2D phase portrait with vector field
        >>> result = system.integrate(x0, u=None, t_span=(0, 20))
        >>> fig = system.phase_plotter.plot_2d(
        ...     result['x'],
        ...     state_names=('Position', 'Velocity'),
        ...     vector_field=system.__call__,
        ...     equilibria=[np.zeros(2)],
        ...     show_direction=True
        ... )
        >>> fig.show()
        >>>
        >>> # Limit cycle (Van der Pol, Lorenz, etc.)
        >>> fig = system.phase_plotter.plot_limit_cycle(
        ...     result['x'],
        ...     state_names=('x', 'ẋ'),
        ...     period_estimate=6.28  # Approx period in time units
        ... )
        >>>
        >>> # 3D phase portrait (Lorenz attractor, etc.)
        >>> fig = system.phase_plotter.plot_3d(
        ...     result['x'],
        ...     state_names=('x', 'y', 'z'),
        ...     show_direction=True
        ... )
        >>>
        >>> # Multiple trajectories (different ICs)
        >>> x_batch = np.stack([sim1['x'], sim2['x'], sim3['x']])
        >>> fig = system.phase_plotter.plot_2d(x_batch)

        See Also
        --------
        plotter : Time-domain trajectory plotting
        control_plotter : Control analysis plots

        Notes
        -----
        - Interactive Plotly plots with zoom, pan, hover
        - Vector field overlay shows system dynamics
        - Start point marked with green circle
        - End point marked with red square
        - Direction arrows show flow along trajectory
        """
        if not hasattr(self, "_phase_plotter"):
            self._phase_plotter = None

        if self._phase_plotter is None:
            from src.visualization.phase_portrait import PhasePortraitPlotter

            backend = getattr(self, "backend", None)
            if backend is not None and hasattr(backend, "default_backend"):
                backend_str = backend.default_backend
            else:
                backend_str = "numpy"

            self._phase_plotter = PhasePortraitPlotter(backend=backend_str)

        return self._phase_plotter

    @property
    def control_plotter(self) -> "ControlPlotter":
        """
        Access control system analysis plotting utilities.

        Provides control-specific visualizations including eigenvalue maps,
        frequency response, Nyquist plots, root locus, and performance metrics.

        Returns
        -------
        ControlPlotter
            Control analysis plotting utilities with methods:
            - plot_eigenvalue_map(eigenvalues, system_type='continuous')
            - plot_frequency_response(w, mag_dB, phase_deg)
            - plot_nyquist(real, imag, frequencies=None)
            - plot_root_locus(root_locus_data, system_type='continuous')
            - plot_step_response(t, y, reference=1.0)
            - plot_impulse_response(t, y)
            - plot_gain_comparison(gains, labels=None)
            - plot_controllability_gramian(W_c, state_names=None)
            - plot_observability_gramian(W_o, state_names=None)
            - plot_riccati_convergence(P_history)

        Examples
        --------
        >>> # Eigenvalue stability map (continuous: Re(λ) < 0)
        >>> lqr = system.design_lqr(Q, R)
        >>> fig = system.control_plotter.plot_eigenvalue_map(
        ...     lqr['closed_loop_eigenvalues'],
        ...     system_type='continuous'
        ... )
        >>> fig.show()
        >>>
        >>> # Impulse response
        >>> from scipy.linalg import expm
        >>> A_cl = A - B @ lqr['gain']
        >>> t = np.linspace(0, 10, 1000)
        >>> y_imp = np.array([C @ expm(A_cl * t_i) @ B for t_i in t]).flatten()
        >>> fig = system.control_plotter.plot_impulse_response(t, y_imp)
        >>>
        >>> # Frequency response (Bode plot)
        >>> from scipy import signal
        >>> sys_cl = signal.StateSpace(A_cl, B, C, D)
        >>> w, H = signal.freqresp(sys_cl, w=np.logspace(-2, 2, 1000))
        >>> mag_dB = 20 * np.log10(np.abs(H).flatten())
        >>> phase_deg = np.angle(H, deg=True).flatten()
        >>> fig = system.control_plotter.plot_frequency_response(w, mag_dB, phase_deg)
        >>>
        >>> # Nyquist plot (open-loop for stability analysis)
        >>> sys_ol = signal.StateSpace(A, B, C, D)
        >>> w, H = signal.freqresp(sys_ol, w=np.logspace(-2, 2, 1000))
        >>> fig = system.control_plotter.plot_nyquist(
        ...     np.real(H).flatten(),
        ...     np.imag(H).flatten(),
        ...     frequencies=w
        ... )
        >>>
        >>> # Root locus (Q weight variation)
        >>> Q_values = np.logspace(-1, 3, 50)
        >>> poles_list = [system.design_lqr(q*np.eye(2), R)['closed_loop_eigenvalues']
        ...               for q in Q_values]
        >>> fig = system.control_plotter.plot_root_locus({
        ...     'gains': Q_values,
        ...     'poles': np.array(poles_list)
        ... }, system_type='continuous')
        >>>
        >>> # Gain comparison
        >>> gains = {
        ...     'Q=10': system.design_lqr(10*np.eye(2), R)['gain'],
        ...     'Q=100': system.design_lqr(100*np.eye(2), R)['gain'],
        ... }
        >>> fig = system.control_plotter.plot_gain_comparison(gains, labels=['θ', 'ω'])

        See Also
        --------
        plotter : Time-domain trajectory plotting
        phase_plotter : Phase space visualization
        design_lqr : LQR controller design (continuous-time)

        Notes
        -----
        Continuous system stability:
        - **Continuous: Re(λ) < 0 (left half-plane)** ← Use system_type='continuous'
        - Discrete: |λ| < 1 (inside unit circle)

        Always specify `system_type='continuous'` for continuous systems!
        """
        if not hasattr(self, "_control_plotter"):
            self._control_plotter = None

        if self._control_plotter is None:
            from src.visualization.control_plots import ControlPlotter

            backend = getattr(self, "backend", None)
            if backend is not None and hasattr(backend, "default_backend"):
                backend_str = backend.default_backend
            else:
                backend_str = "numpy"

            self._control_plotter = ControlPlotter(backend=backend_str)

        return self._control_plotter

    def plot(
        self,
        result: IntegrationResult,
        state_names: Optional[list] = None,
        **kwargs,
    ) -> "go.Figure":
        """
        Plot integration result (convenience method).

        Wrapper around plotter.plot_trajectory() for quick visualization
        of continuous-time integration results.

        Parameters
        ----------
        result : IntegrationResult
            Integration result dictionary with 't' and 'x' keys
            from integrate() or simulate()
        state_names : Optional[list]
            Names for state variables (e.g., ['Position', 'Velocity'])
            If None, uses generic labels ['x₁', 'x₂', ...]
        **kwargs
            Additional arguments passed to plot_trajectory():
            - title : str - Plot title
            - color_scheme : str - Color scheme name
            - show_legend : bool - Show legend for batched trajectories

        Returns
        -------
        go.Figure
            Interactive Plotly figure object

        Examples
        --------
        >>> # Simple usage
        >>> result = system.integrate(x0, u=None, t_span=(0, 10))
        >>> fig = system.plot(result)
        >>> fig.show()
        >>>
        >>> # With state names and custom title
        >>> fig = system.plot(
        ...     result,
        ...     state_names=['θ', 'ω'],
        ...     title='Pendulum Dynamics'
        ... )
        >>>
        >>> # Export to HTML
        >>> fig.write_html('simulation.html')
        >>>
        >>> # Apply publication theme
        >>> from src.visualization.themes import PlotThemes
        >>> fig = system.plot(result)
        >>> fig = PlotThemes.apply_theme(fig, theme='publication')
        >>> fig.show()
        >>>
        >>> # Batched trajectories (Monte Carlo for stochastic systems)
        >>> results = []
        >>> for trial in range(10):
        ...     results.append(system.integrate(x0, u=None, t_span=(0, 10)))
        >>> x_batch = np.stack([r['x'] for r in results])
        >>> result_batch = {'t': results[0]['t'], 'x': x_batch}
        >>> fig = system.plot(result_batch)  # Plots all 10 trajectories

        See Also
        --------
        plotter.plot_trajectory : Full trajectory plotting method
        plotter.plot_state_and_control : Plot states and controls together
        phase_plotter.plot_2d : Phase space visualization
        control_plotter : Control analysis plots

        Notes
        -----
        This is a convenience wrapper that:
        - Extracts time and state from result dictionary
        - Calls plotter.plot_trajectory() with appropriate arguments
        - Returns Plotly figure for further customization

        For more control over plotting, use plotter methods directly.
        """
        return self.plotter.plot_trajectory(
            result["t"],
            result["x"],
            state_names=state_names,
            **kwargs,
        )

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
        nx = getattr(self, "nx", "?")
        nu = getattr(self, "nu", "?")
        ny = getattr(self, "ny", "?")
        return f"{class_name}(nx={nx}, nu={nu}, ny={ny})"
