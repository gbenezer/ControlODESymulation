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

Abstract base class for all discrete-time dynamical systems.

This module should be placed at:
    src/systems/base/discrete_system_base.py
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

import numpy as np

from src.types.core import (
    ControlVector,
    DiscreteControlInput,
    DiscreteFeedbackPolicy,
    StateVector,
)
from src.types.linearization import DiscreteLinearization
from src.types.trajectories import DiscreteSimulationResult

if TYPE_CHECKING:
    import plotly.graph_objects as go

    from src.control.control_synthesis import ControlSynthesis
    from src.control.system_analysis import SystemAnalysis
    from src.visualization.control_plots import ControlPlotter
    from src.visualization.phase_portrait import PhasePortraitPlotter
    from src.visualization.trajectory_plotter import TrajectoryPlotter


class DiscreteSystemBase(ABC):
    """
    Abstract base class for all discrete-time dynamical systems.

    All discrete-time systems satisfy:
        x[k+1] = f(x[k], u[k], k)

    Subclasses must implement:
    1. dt (property): Sampling period
    2. step(x, u, k): Single time step update
    3. simulate(x0, u_sequence, n_steps): Multi-step simulation
    4. linearize(x_eq, u_eq): Compute linearization

    Additional concrete methods provided:
    - rollout(): Closed-loop simulation with state-feedback policy

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
    ...         u = u if u is not None else np.zeros(self.nu)
    ...         return 0.9 * x + 0.1 * u
    ...
    ...     def simulate(self, x0, u_sequence, n_steps):
    ...         # Implement multi-step simulation
    ...         ...
    ...
    ...     def linearize(self, x_eq, u_eq):
    ...         Ad = 0.9 * np.eye(self.nx)
    ...         Bd = 0.1 * np.eye(self.nx, self.nu)
    ...         return (Ad, Bd)
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

    # =========================================================================
    # Abstract Methods (MUST be implemented by subclasses)
    # =========================================================================

    @abstractmethod
    def step(self, x: StateVector, u: Optional[ControlVector] = None, k: int = 0) -> StateVector:
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
        - For autonomous systems, k is ignored
        - For time-invariant systems, k is typically ignored
        - For batch evaluation, x and u should have shape (n_dim, n_batch)
        - The returned state should be in the same backend as the input

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

    @abstractmethod
    def simulate(
        self,
        x0: StateVector,
        u_sequence: DiscreteControlInput = None,
        n_steps: int = 100,
        **kwargs,
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
            TypedDict (returns as dict) containing:
            - states: State trajectory (nx, n_steps+1) - includes x[0]
            - controls: Control sequence (nu, n_steps) if applicable
            - time_steps: Time step indices [0, 1, ..., n_steps]
            - dt: Sampling period
            - metadata: Additional info (method, success, etc.)

        Notes
        -----
        The state trajectory includes n_steps+1 points (including x0).
        The control sequence has n_steps points (one for each transition).

        For closed-loop simulation with state-dependent control, you can use
        rollout() instead, which provides a cleaner interface for state feedback.

        Examples
        --------
        Open-loop with constant control:

        >>> x0 = np.array([1.0, 0.0])
        >>> u = np.array([0.5])
        >>> result = system.simulate(x0, u, n_steps=100)
        >>> plt.step(result["time_steps"], result["states"][0, :])

        Pre-computed control sequence:

        >>> u_seq = [np.array([0.5 * np.sin(k * 0.1)]) for k in range(100)]
        >>> result = system.simulate(x0, u_seq, n_steps=100)

        Time-indexed control function:

        >>> def u_func(k):
        ...     return np.array([0.5 * np.sin(k * system.dt)])
        >>> result = system.simulate(x0, u_func, n_steps=100)

        Autonomous system (no control):

        >>> result = system.simulate(x0, u_sequence=None, n_steps=100)
        """

    @abstractmethod
    def linearize(
        self,
        x_eq: StateVector,
        u_eq: Optional[ControlVector] = None,
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
            Tuple containing Jacobian matrices:
            - Deterministic systems: (Ad, Bd)
            - Stochastic systems: (Ad, Bd, Gd) where Gd is diffusion matrix

        Notes
        -----
        The linearization is valid for small deviations from the equilibrium:
            δx[k] = x[k] - x_eq
            δu[k] = u[k] - u_eq

        For symbolic systems, Jacobians are computed symbolically then evaluated.
        For data-driven systems, Jacobians may be computed via finite differences.

        The equilibrium point should satisfy f(x_eq, u_eq) = x_eq (fixed point).

        Stability analysis for discrete systems:
        - Stable if all |eigenvalues(Ad)| < 1
        - Unstable if any |eigenvalue(Ad)| > 1
        - Marginal if |eigenvalue(Ad)| = 1

        Examples
        --------
        Linearize at origin:

        >>> x_eq = np.zeros(2)
        >>> u_eq = np.zeros(1)
        >>> Ad, Bd = system.linearize(x_eq, u_eq)
        >>> print(f"Ad matrix:\\n{Ad}")
        >>> print(f"Bd matrix:\\n{Bd}")

        Check discrete stability:

        >>> eigenvalues = np.linalg.eigvals(Ad)
        >>> is_stable = np.all(np.abs(eigenvalues) < 1)
        >>> print(f"System stable: {is_stable}")

        Design discrete LQR controller:

        >>> from scipy.linalg import solve_discrete_are
        >>> P = solve_discrete_are(Ad, Bd, Q, R)
        >>> K = np.linalg.inv(R + Bd.T @ P @ Bd) @ (Bd.T @ P @ Ad)

        Relationship to continuous linearization:

        >>> # For Euler discretization: Ad ≈ I + dt * A
        >>> dt = system.dt
        >>> A_approx = (Ad - np.eye(system.nx)) / dt
        """

    # =========================================================================
    # Concrete Methods (Provided by base class)
    # =========================================================================

    def rollout(
        self,
        x0: StateVector,
        policy: Optional[DiscreteFeedbackPolicy] = None,
        n_steps: int = 100,
        **kwargs,
    ) -> DiscreteSimulationResult:
        """
        Rollout system trajectory with optional state-feedback policy.

        This is a higher-level alternative to simulate() that provides a cleaner
        interface for closed-loop simulation with state-dependent policies.

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
            Additional arguments (stored in metadata)

        Returns
        -------
        DiscreteSimulationResult
            TypedDict (returns as dict) containing trajectory and metadata
            - states: (n_steps+1, nx) - TIME-MAJOR
            - controls: (n_steps, nu) - TIME-MAJOR
            - time_steps: (n_steps+1,)
            - dt: float
            - metadata: dict with closed_loop flag

        Examples
        --------
        Open-loop rollout:

        >>> result = system.rollout(x0, n_steps=100)

        State feedback policy (LQR):

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

        # Manual rollout with state feedback
        if policy is None:
            return self.simulate(x0, u_sequence=None, n_steps=n_steps, **kwargs)

        # Implement closed-loop rollout with TIME-MAJOR ordering
        states = np.zeros((n_steps + 1, getattr(self, "nx", x0.shape[0])))
        states[0, :] = x0
        controls = []
        x = x0

        for k in range(n_steps):
            u = policy(x, k)
            controls.append(u)
            x = self.step(x, u, k)
            states[k + 1, :] = x

        controls_array = np.array(controls) if controls else None  # (n_steps, nu)

        return {
            "states": states,  # (n_steps+1, nx)
            "controls": controls_array,  # (n_steps, nu)
            "time_steps": np.arange(n_steps + 1),
            "dt": self.dt,
            "metadata": {**kwargs, "closed_loop": True, "method": "rollout"},
        }

    # =========================================================================
    # Control Framework Integration
    # =========================================================================

    @property
    def control(self) -> "ControlSynthesis":
        """
        Access control synthesis utilities.

        Provides classical control design algorithms including LQR, Kalman filter,
        and LQG for discrete-time systems.

        Returns
        -------
        ControlSynthesis
            Control design utilities with methods:
            - design_lqr(A, B, Q, R, N=None, system_type='discrete')
            - design_kalman(A, C, Q, R, system_type='discrete')
            - design_lqg(A, B, C, Q_state, R_control, Q_process, R_measurement, ...)

        Examples
        --------
        >>> # Linearize and design discrete LQR
        >>> Ad, Bd = system.linearize(x_eq, u_eq)
        >>> Q = np.diag([10, 1])
        >>> R = np.array([[0.1]])
        >>> result = system.control.design_lqr(Ad, Bd, Q, R, system_type='discrete')
        >>> K = result['gain']
        >>> print(f"Discrete gain: {K}")
        >>>
        >>> # Apply control in simulation
        >>> x = x0
        >>> for k in range(N):
        ...     u = -K @ (x - x_eq)
        ...     x = system.step(x, u)
        >>>
        >>> # Design discrete Kalman filter
        >>> C = np.array([[1, 0]])
        >>> kalman = system.control.design_kalman(
        ...     Ad, C, Q_proc=0.01*np.eye(2), R_meas=np.array([[0.1]]),
        ...     system_type='discrete'
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
        for discrete-time linear systems.

        Returns
        -------
        SystemAnalysis
            System analysis utilities with methods:
            - stability(A, system_type='discrete', tolerance=1e-10)
            - controllability(A, B, tolerance=1e-10)
            - observability(A, C, tolerance=1e-10)
            - analyze_linearization(A, B, C, system_type='discrete')

        Examples
        --------
        >>> # Check stability (discrete: |λ| < 1)
        >>> Ad, Bd = system.linearize(x_eq, u_eq)
        >>> stability = system.analysis.stability(Ad, system_type='discrete')
        >>> print(f"Stable: {stability['is_stable']}")
        >>> print(f"Spectral radius: {stability['spectral_radius']:.3f}")
        >>>
        >>> # Check controllability
        >>> ctrl_info = system.analysis.controllability(Ad, Bd)
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
        >>> # Plot simulation result
        >>> result = system.simulate(x0, u_sequence, n_steps=100)
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
        ...     'IC 1': result1['x'],
        ...     'IC 2': result2['x'],
        ... }
        >>> fig = system.plotter.plot_comparison(result1['t'], trajectories)
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
        >>> # 2D phase portrait
        >>> result = system.simulate(x0, u_sequence, n_steps=200)
        >>> fig = system.phase_plotter.plot_2d(
        ...     result['x'],
        ...     state_names=('Position', 'Velocity'),
        ...     show_direction=True,
        ...     show_start_end=True
        ... )
        >>> fig.show()
        >>>
        >>> # With equilibrium points
        >>> fig = system.phase_plotter.plot_2d(
        ...     result['x'],
        ...     state_names=('x₁', 'x₂'),
        ...     equilibria=[np.zeros(2), np.array([1, 0])]
        ... )
        >>>
        >>> # 3D phase portrait (for 3+ state systems)
        >>> fig = system.phase_plotter.plot_3d(
        ...     result['x'][:, :3],
        ...     state_names=('x', 'y', 'z'),
        ...     show_direction=True
        ... )
        >>>
        >>> # Multiple initial conditions (batched)
        >>> x_batch = np.stack([sim1['x'], sim2['x'], sim3['x']])
        >>> fig = system.phase_plotter.plot_2d(x_batch)

        See Also
        --------
        plotter : Time-domain trajectory plotting
        control_plotter : Control analysis plots

        Notes
        -----
        - Interactive Plotly plots with zoom, pan, hover
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
            - plot_eigenvalue_map(eigenvalues, system_type='discrete')
            - plot_frequency_response(w, mag_dB, phase_deg)
            - plot_nyquist(real, imag, frequencies=None)
            - plot_root_locus(root_locus_data, system_type='discrete')
            - plot_step_response(t, y, reference=1.0)
            - plot_impulse_response(t, y)
            - plot_gain_comparison(gains, labels=None)
            - plot_controllability_gramian(W_c, state_names=None)
            - plot_observability_gramian(W_o, state_names=None)
            - plot_riccati_convergence(P_history)

        Examples
        --------
        >>> # Eigenvalue stability map (discrete: |λ| < 1)
        >>> lqr = system.design_lqr(Q, R)
        >>> fig = system.control_plotter.plot_eigenvalue_map(
        ...     lqr['closed_loop_eigenvalues'],
        ...     system_type='discrete'
        ... )
        >>> fig.show()
        >>>
        >>> # Step response
        >>> A_cl = Ad - Bd @ lqr['gain']
        >>> x = np.zeros(system.nx)
        >>> y_step = []
        >>> for k in range(100):
        ...     y_step.append(C @ x)
        ...     x = A_cl @ x + Bd * (1.0 if k == 0 else 0.0)
        >>> t = np.arange(100) * system.dt
        >>> fig = system.control_plotter.plot_step_response(t, np.array(y_step))
        >>>
        >>> # Frequency response (Bode plot)
        >>> from scipy import signal
        >>> sys_cl = signal.StateSpace(A_cl, Bd, C, D, dt=system.dt)
        >>> w, H = signal.freqresp(sys_cl, w=np.logspace(-2, 2, 1000))
        >>> mag_dB = 20 * np.log10(np.abs(H).flatten())
        >>> phase_deg = np.angle(H, deg=True).flatten()
        >>> fig = system.control_plotter.plot_frequency_response(w, mag_dB, phase_deg)
        >>>
        >>> # Root locus (Q weight variation)
        >>> Q_values = np.logspace(-1, 3, 50)
        >>> poles_list = [system.design_lqr(q*np.eye(2), R)['closed_loop_eigenvalues']
        ...               for q in Q_values]
        >>> fig = system.control_plotter.plot_root_locus({
        ...     'gains': Q_values,
        ...     'poles': np.array(poles_list)
        ... }, system_type='discrete')
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
        design_lqr : LQR controller design (discrete-time)

        Notes
        -----
        Discrete system stability:
        - Continuous: Re(λ) < 0 (left half-plane)
        - **Discrete: |λ| < 1 (inside unit circle)** ← Use system_type='discrete'

        Always specify `system_type='discrete'` for discrete systems!
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
        result: DiscreteSimulationResult,
        state_names: Optional[list] = None,
        **kwargs,
    ) -> "go.Figure":
        """
        Plot simulation result (convenience method).

        Wrapper around plotter.plot_trajectory() for quick visualization
        of discrete-time simulation results.

        Parameters
        ----------
        result : DiscreteSimulationResult
            Simulation result dictionary with 't' and 'x' keys
            from simulate() or rollout()
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
        >>> result = system.simulate(x0, u_sequence, n_steps=100)
        >>> fig = system.plot(result)
        >>> fig.show()
        >>>
        >>> # With state names and custom title
        >>> fig = system.plot(
        ...     result,
        ...     state_names=['θ', 'ω'],
        ...     title='Discrete Pendulum Dynamics'
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
        >>> # Batched trajectories (Monte Carlo)
        >>> results = []
        >>> for x0 in initial_conditions:
        ...     results.append(system.simulate(x0, u_sequence, n_steps=100))
        >>> x_batch = np.stack([r['x'] for r in results])
        >>> result_batch = {'t': results[0]['t'], 'x': x_batch}
        >>> fig = system.plot(result_batch)  # Plots all trajectories

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
        nx = getattr(self, "nx", "?")
        nu = getattr(self, "nu", "?")
        dt = getattr(self, "dt", "?")
        return f"{class_name}(nx={nx}, nu={nu}, dt={dt})"
