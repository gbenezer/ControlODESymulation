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
Trajectory and Sequence Types

Defines types for time series data in control and dynamical systems:
- State trajectories (time series of states)
- Control sequences (time series of control inputs)
- Output sequences (time series of measurements)
- Noise sequences (time series of stochastic disturbances)
- Time arrays and spans
- Integration and simulation results

These types represent the fundamental data structures for storing
and analyzing system behavior over time.

Mathematical Context
-------------------
Trajectories represent solutions to differential/difference equations:

Continuous: x(t) where dx/dt = f(x(t), u(t), t)
Discrete: {x[k]} where x[k+1] = f(x[k], u[k])

Shape Conventions:
- Single trajectory: (n_steps, nx)
- Batched trajectories: (n_steps, batch, nx)
- Multiple trials: (n_trials, n_steps, nx)

Usage
-----
>>> from src.types.trajectories import (
...     StateTrajectory,
...     ControlSequence,
...     TimePoints,
...     IntegrationResult,
... )
>>>
>>> # Simulate system
>>> trajectory: StateTrajectory = system.simulate(x0, u_seq, steps=100)
>>> print(trajectory.shape)  # (101, nx) - includes t=0
>>>
>>> # Analyze trajectory
>>> time: TimePoints = np.linspace(0, 10, 101)
>>> for t, x in zip(time, trajectory):
...     print(f"t={t:.2f}, x={x}")
"""

from typing import Any, Dict, Optional, Tuple

from typing_extensions import TypedDict

from .core import ArrayLike

# ============================================================================
# Trajectory and Sequence Types
# ============================================================================

StateTrajectory = ArrayLike
"""
State trajectory over time.

Time series of state vectors representing system evolution.

Shapes:
- Single trajectory: (n_steps, nx)
  Each row is x[k] or x(t_k)
  
- Batched trajectories: (n_steps, batch, nx)
  Multiple trajectories simulated in parallel
  
- Multiple trials: (n_trials, n_steps, nx)
  Independent simulation runs

Indexing:
- trajectory[k] -> state at step k (nx,)
- trajectory[:, i] -> i-th state component over time (n_steps,)
- trajectory[k, b] -> state at step k, batch b (nx,)

Examples
--------
>>> # Single trajectory from simulation
>>> x0 = np.array([1.0, 0.0])
>>> trajectory: StateTrajectory = system.simulate(x0, u_seq, steps=100)
>>> print(trajectory.shape)  # (101, 2) - includes initial state
>>> 
>>> # Extract position and velocity
>>> position = trajectory[:, 0]  # First state component
>>> velocity = trajectory[:, 1]  # Second state component
>>> 
>>> # Batched simulation (Monte Carlo)
>>> x0_batch = np.random.randn(1000, 2)  # 1000 initial conditions
>>> trajectories: StateTrajectory = system.simulate_batch(x0_batch, u_seq)
>>> print(trajectories.shape)  # (101, 1000, 2)
>>> 
>>> # Multiple independent trials
>>> trials: StateTrajectory = np.array([
...     system.simulate(x0, u_seq, steps=100)
...     for _ in range(10)
... ])
>>> print(trials.shape)  # (10, 101, 2)
"""

ControlSequence = ArrayLike
"""
Control input sequence over time.

Time series of control vectors applied to system.

Shapes:
- Single sequence: (n_steps, nu)
  Control at each time step
  
- Batched sequences: (n_steps, batch, nu)
  Different controls for each batch
  
- Open-loop: (n_steps, nu)
  Pre-computed control sequence

Note: Length is typically n_steps (not n_steps+1) since u[k] affects
x[k] → x[k+1], and we don't need u at final state.

Examples
--------
>>> # Zero control
>>> u_seq: ControlSequence = np.zeros((100, 1))
>>> 
>>> # Sinusoidal control
>>> t = np.linspace(0, 10, 100)
>>> u_seq: ControlSequence = np.sin(t).reshape(-1, 1)
>>> 
>>> # MPC generates control sequence
>>> u_optimal: ControlSequence = mpc.solve(x0, horizon=20)
>>> print(u_optimal.shape)  # (20, nu)
>>> 
>>> # Feedback control sequence (computed online)
>>> trajectory = [x0]
>>> controls = []
>>> x = x0
>>> for k in range(100):
...     u = controller(x)  # Feedback policy
...     controls.append(u)
...     x = system.step(x, u)
...     trajectory.append(x)
>>> u_seq: ControlSequence = np.array(controls)
>>> 
>>> # Batched control for multiple systems
>>> u_batch: ControlSequence = np.random.randn(100, 50, 2)  # 50 systems
"""

OutputSequence = ArrayLike
"""
Output/measurement sequence over time.

Time series of sensor measurements or system outputs.

Shapes:
- Single sequence: (n_steps, ny)
  Measurements at each time
  
- Batched sequences: (n_steps, batch, ny)
  Multiple measurement streams

Common uses:
- Sensor data for state estimation
- System identification datasets
- Validation data for learned models

Examples
--------
>>> # Simulate and observe
>>> trajectory: StateTrajectory = system.simulate(x0, u_seq, steps=100)
>>> observations: OutputSequence = system.observe(trajectory)
>>> print(observations.shape)  # (101, ny)
>>> 
>>> # Noisy measurements
>>> y_clean: OutputSequence = C @ trajectory.T  # (ny, n_steps)
>>> y_noisy: OutputSequence = y_clean + np.random.randn(*y_clean.shape) * 0.1
>>> y_noisy = y_noisy.T  # (n_steps, ny)
>>> 
>>> # Kalman filter with measurements
>>> x_estimates = []
>>> for k in range(len(observations)):
...     y_k = observations[k]
...     x_hat = kalman_filter.update(y_k, u_seq[k])
...     x_estimates.append(x_hat)
>>> 
>>> # System identification dataset
>>> dataset = {
...     'inputs': u_seq,      # (n_steps, nu)
...     'outputs': y_seq,     # (n_steps, ny)
...     'time': t_points,     # (n_steps,)
... }
"""

NoiseSequence = ArrayLike
"""
Noise/disturbance sequence for stochastic simulation.

Time series of random disturbances in stochastic systems.

Shapes:
- Single sequence: (n_steps, nw)
  IID noise samples
  
- Batched sequences: (n_steps, batch, nw)
  Independent noise for each trajectory

Distribution:
- Discrete stochastic: w[k] ~ N(0, I)
- Continuous SDE: dW[k] ~ N(0, dt*I) (Brownian increments)

Examples
--------
>>> # Standard normal noise for discrete system
>>> w_seq: NoiseSequence = np.random.randn(100, 2)
>>> trajectory = system.simulate_stochastic(x0, u_seq, w_seq)
>>> 
>>> # Brownian increments for continuous SDE
>>> dt = 0.01
>>> n_steps = 1000
>>> dW: NoiseSequence = np.random.randn(n_steps, 2) * np.sqrt(dt)
>>> 
>>> # Reproducible simulation with seed
>>> np.random.seed(42)
>>> w_seq: NoiseSequence = np.random.randn(100, 3)
>>> 
>>> # Batched Monte Carlo simulation
>>> n_trials = 1000
>>> w_batch: NoiseSequence = np.random.randn(100, n_trials, 2)
>>> trajectories_mc = system.simulate_stochastic_batch(x0, u_seq, w_batch)
>>> 
>>> # Colored noise (correlated)
>>> # First generate white noise, then filter
>>> w_white = np.random.randn(1000, 2)
>>> # Apply low-pass filter for colored noise
>>> from scipy.signal import lfilter
>>> b, a = [0.1, 0.9], [1.0]
>>> w_colored: NoiseSequence = lfilter(b, a, w_white, axis=0)
"""


# ============================================================================
# Time Array Types
# ============================================================================

TimePoints = ArrayLike
"""
Array of time points for simulation or evaluation.

Discrete time instants at which system is evaluated.

Shape: (n_points,)

Types:
- Regular grid: t = [0, dt, 2*dt, ..., T]
- Irregular grid: t = [0, 0.1, 0.15, 0.3, ...]
- Adaptive: From adaptive integrator (irregular)

Examples
--------
>>> # Regular time grid
>>> t: TimePoints = np.linspace(0, 10, 101)
>>> dt = t[1] - t[0]
>>> print(f"dt = {dt:.3f}")  # 0.100
>>> 
>>> # Irregular time grid
>>> t: TimePoints = np.array([0, 0.1, 0.15, 0.5, 1.0, 2.0, 5.0, 10.0])
>>> 
>>> # Logarithmic spacing (for stiff systems)
>>> t: TimePoints = np.logspace(-3, 1, 100)  # 0.001 to 10
>>> 
>>> # From simulation result
>>> result: IntegrationResult = integrator.solve(x0, u, t_span)
>>> t: TimePoints = result['t']
>>> trajectory: StateTrajectory = result['y']
>>> 
>>> # Discrete-time steps (as floats)
>>> k = np.arange(0, 100)
>>> t: TimePoints = k * dt  # Convert to continuous time
"""

TimeSpan = Tuple[float, float]
"""
Time interval for continuous integration: (t_start, t_end).

Defines initial and final times for ODE/SDE integration.

Format: (t_start, t_end) where t_start < t_end

Examples
--------
>>> # Standard interval [0, T]
>>> t_span: TimeSpan = (0.0, 10.0)
>>> 
>>> # Non-zero start time
>>> t_span: TimeSpan = (5.0, 15.0)
>>> 
>>> # Short interval for testing
>>> t_span: TimeSpan = (0.0, 0.1)
>>> 
>>> # Use in integration
>>> from scipy.integrate import solve_ivp
>>> result = solve_ivp(
...     fun=dynamics,
...     t_span=t_span,
...     y0=x0,
...     method='RK45'
... )
>>> 
>>> # Extract t_eval from t_span
>>> t_start, t_end = t_span
>>> t_eval: TimePoints = np.linspace(t_start, t_end, 1000)
"""


# ============================================================================
# Integration and Simulation Result Types
# ============================================================================


class IntegrationResult(TypedDict, total=False):
    """
    Result from continuous-time integration (ODE/SDE solver).
    
    Contains trajectory, time points, and solver diagnostics.
    
    Shape Convention
    ----------------
    Time-major ordering for easy analysis and plotting:
    - t: (T,) - Time points
    - x: (T, nx) - State at each time point
    
    This differs from scipy's (nx, T) convention but is more natural
    for analysis: x[:, i] gives i-th component over time.
    
    Attributes
    ----------
    t : ArrayLike
        Time points (T,)
    x : ArrayLike
        State trajectory (T, nx) - time-major ordering
    success : bool
        Whether integration succeeded
    message : str
        Status message
    nfev : int
        Number of function evaluations
    nsteps : int
        Number of integration steps
    integration_time : float
        Computation time in seconds
    solver : str
        Name of solver used
    
    Optional Fields
    ---------------
    njev : int
        Number of Jacobian evaluations
    nlu : int
        Number of LU decompositions
    status : int
        Solver-specific status code
    sol : Any
        Dense output object (solver-specific)
    dense_output : bool
        Whether dense output is available
    
    Examples
    --------
    >>> # Integrate system
    >>> result: IntegrationResult = integrator.integrate(
    ...     x0=np.array([1.0, 0.0]),
    ...     u_func=lambda t, x: np.zeros(1),
    ...     t_span=(0.0, 10.0)
    ... )
    >>>
    >>> # Access results
    >>> t = result["t"]        # Time points (T,)
    >>> x = result["x"]        # States (T, nx)
    >>> 
    >>> # Plot first state component
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(t, x[:, 0], label='x1')
    >>> plt.plot(t, x[:, 1], label='x2')
    >>>
    >>> # Check success
    >>> if result["success"]:
    ...     print(f"Completed in {result['integration_time']:.3f}s")
    ...     print(f"Function evals: {result['nfev']}")
    """
    t: ArrayLike
    x: ArrayLike
    success: bool
    message: str
    nfev: int
    nsteps: int
    integration_time: float
    solver: str
    # Optional fields
    njev: int
    nlu: int
    status: int
    sol: Any
    dense_output: bool


class SimulationResult(TypedDict, total=False):
    """
    Result from discretized continuous-time simulation.

    Contains trajectories, control sequences, and metadata.

    Attributes
    ----------
    states : StateTrajectory
        State trajectory (n_steps+1, nx) - includes initial state
    controls : Optional[ControlSequence]
        Control sequence used (n_steps, nu)
    outputs : Optional[OutputSequence]
        Output sequence (n_steps+1, ny) if computed
    noise : Optional[NoiseSequence]
        Noise sequence (n_steps, nw) for stochastic systems
    time : Optional[TimePoints]
        Time points (n_steps+1,) if applicable
    metadata : Dict[str, Any]
        Additional information (cost, constraints, etc.)

    Examples
    --------
    >>> # Basic discrete simulation
    >>> result: SimulationResult = system.simulate(
    ...     x0=np.zeros(3),
    ...     u_seq=np.zeros((100, 2)),
    ...     steps=100,
    ...     return_all=True
    ... )
    >>>
    >>> states = result['states']      # (101, 3)
    >>> controls = result['controls']  # (100, 2)
    >>>
    >>> # Stochastic simulation
    >>> result: SimulationResult = sde_system.simulate(
    ...     x0=np.zeros(2),
    ...     u_seq=np.zeros((100, 1)),
    ...     w_seq=np.random.randn(100, 2),
    ...     steps=100
    ... )
    >>>
    >>> if 'noise' in result:
    ...     print("Stochastic simulation")
    ...     noise_used = result['noise']
    >>>
    >>> # With time information
    >>> dt = 0.01
    >>> result: SimulationResult = system.simulate(
    ...     x0=x0, u_seq=u_seq, steps=100, dt=dt
    ... )
    >>> time = result['time']  # [0, dt, 2*dt, ..., 100*dt]
    >>>
    >>> # Access metadata
    >>> if 'metadata' in result:
    ...     metadata = result['metadata']
    ...     if 'cost' in metadata:
    ...         print(f"Trajectory cost: {metadata['cost']:.2f}")
    """

    states: ArrayLike
    controls: Optional[ArrayLike]
    outputs: Optional[ArrayLike]
    noise: Optional[ArrayLike]
    time: Optional[ArrayLike]
    metadata: Dict[str, Any]

class DiscreteSimulationResult(TypedDict, total=False):
    """
    Result from discrete-time system simulation.

    Contains state trajectory, control sequence, and metadata for
    discrete-time systems (difference equations).

    Attributes
    ----------
    states : StateTrajectory
        State trajectory (n_steps+1, nx) - includes initial state x[0]
    controls : Optional[ControlSequence]
        Control sequence applied (n_steps, nu)
    outputs : Optional[OutputSequence]
        Output sequence (n_steps+1, ny) if computed
    noise : Optional[NoiseSequence]
        Noise sequence (n_steps, nw) for stochastic discrete systems
    time_steps : ArrayLike
        Time step indices [0, 1, 2, ..., n_steps]
    dt : float
        Time step / sampling period
    metadata : Dict[str, Any]
        Additional information (method, success, closed_loop, etc.)

    Notes
    -----
    The discrete result differs from continuous simulation results:
    - Uses integer time_steps instead of continuous time points
    - Includes dt (sampling period) as a scalar
    - State trajectory has n_steps+1 points (includes x[0])
    - Control sequence has n_steps points (u[0] through u[n_steps-1])

    Examples
    --------
    >>> # Discrete simulation
    >>> result: DiscreteSimulationResult = discrete_system.simulate(
    ...     x0=np.array([1.0, 0.0]),
    ...     u_sequence=np.zeros((100, 1)),
    ...     n_steps=100
    ... )
    >>>
    >>> states = result['states']          # (101, 2) - includes x[0]
    >>> controls = result['controls']      # (100, 1)
    >>> time_steps = result['time_steps']  # [0, 1, ..., 100]
    >>> dt = result['dt']                  # 0.01
    >>>
    >>> # Convert to continuous time if needed
    >>> time_continuous = result['time_steps'] * result['dt']
    >>>
    >>> # Plot discrete trajectory
    >>> import matplotlib.pyplot as plt
    >>> plt.step(time_steps, states[:, 0], where='post', label='x1')
    >>> plt.xlabel('Time Step k')
    >>> plt.ylabel('State')
    """

    states: ArrayLike
    controls: Optional[ArrayLike]
    outputs: Optional[ArrayLike]
    noise: Optional[ArrayLike]
    time_steps: ArrayLike
    dt: float
    metadata: Dict[str, Any]

# ============================================================================
# Trajectory Analysis Types
# ============================================================================


class TrajectoryStatistics(TypedDict, total=False):
    """
    Statistical summary of trajectory.

    Computed statistics over time series data.

    Attributes
    ----------
    mean : ArrayLike
        Mean state over trajectory (nx,)
    std : ArrayLike
        Standard deviation over trajectory (nx,)
    min : ArrayLike
        Minimum values (nx,)
    max : ArrayLike
        Maximum values (nx,)
    initial : ArrayLike
        Initial state x[0] (nx,)
    final : ArrayLike
        Final state x[-1] (nx,)
    length : int
        Number of time steps
    duration : float
        Time duration (t_end - t_start)

    Examples
    --------
    >>> def compute_trajectory_stats(trajectory: StateTrajectory) -> TrajectoryStatistics:
    ...     '''Compute statistics of trajectory.'''
    ...     return TrajectoryStatistics(
    ...         mean=np.mean(trajectory, axis=0),
    ...         std=np.std(trajectory, axis=0),
    ...         min=np.min(trajectory, axis=0),
    ...         max=np.max(trajectory, axis=0),
    ...         initial=trajectory[0],
    ...         final=trajectory[-1],
    ...         length=len(trajectory),
    ...     )
    >>>
    >>> stats: TrajectoryStatistics = compute_trajectory_stats(trajectory)
    >>> print(f"Mean state: {stats['mean']}")
    >>> print(f"Final state: {stats['final']}")
    >>> print(f"Max deviation: {np.max(stats['std'])}")
    """

    mean: ArrayLike
    std: ArrayLike
    min: ArrayLike
    max: ArrayLike
    initial: ArrayLike
    final: ArrayLike
    length: int
    duration: float


class TrajectorySegment(TypedDict):
    """
    Segment of trajectory between two time points.

    Extracted portion of full trajectory for analysis.

    Attributes
    ----------
    states : StateTrajectory
        State trajectory segment
    controls : Optional[ControlSequence]
        Control sequence segment
    time : TimePoints
        Time points for segment
    start_index : int
        Index in original trajectory where segment starts
    end_index : int
        Index in original trajectory where segment ends

    Examples
    --------
    >>> def extract_segment(
    ...     result: SimulationResult,
    ...     t_start: float,
    ...     t_end: float
    ... ) -> TrajectorySegment:
    ...     '''Extract trajectory segment.'''
    ...     time = result['time']
    ...     mask = (time >= t_start) & (time <= t_end)
    ...     indices = np.where(mask)[0]
    ...
    ...     return TrajectorySegment(
    ...         states=result['states'][mask],
    ...         controls=result['controls'][mask[:-1]] if 'controls' in result else None,
    ...         time=time[mask],
    ...         start_index=indices[0],
    ...         end_index=indices[-1],
    ...     )
    >>>
    >>> # Extract transient response (first 2 seconds)
    >>> segment: TrajectorySegment = extract_segment(result, 0.0, 2.0)
    >>> transient_states = segment['states']
    """

    states: ArrayLike
    controls: Optional[ArrayLike]
    time: ArrayLike
    start_index: int
    end_index: int


# ============================================================================
# Export All
# ============================================================================

__all__ = [
    # Trajectory and sequence types
    "StateTrajectory",
    "ControlSequence",
    "OutputSequence",
    "NoiseSequence",
    # Time types
    "TimePoints",
    "TimeSpan",
    # Result types
    "IntegrationResult",
    "SimulationResult",
    # Analysis types
    "TrajectoryStatistics",
    "TrajectorySegment",
]
