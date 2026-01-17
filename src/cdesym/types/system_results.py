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

# System Result Types

from typing import Any, Dict, Optional, Union
from typing_extensions import TypedDict
from .core import ArrayLike


# ============================================================================
# BASE TYPES (shared fields via inheritance)
# ============================================================================


class IntegrationResultBase(TypedDict, total=False):
    """
    Base type for all integration results (adaptive time stepping).

    Contains fields common to both ODE and SDE integration.
    """

    t: ArrayLike  # Time points (may be irregular)
    x: ArrayLike  # State trajectory
    success: bool
    message: str
    nfev: int  # Function evaluations (drift)
    nsteps: int
    solver: str
    integration_time: float
    # Optional solver diagnostics
    njev: int
    nlu: int
    status: int
    sol: Any
    dense_output: bool


class SimulationResultBase(TypedDict, total=False):
    """
    Base type for all simulation results (regular time grid).

    Contains fields common to deterministic and stochastic simulation.
    """

    t: ArrayLike  # Regular time grid
    x: ArrayLike  # State trajectory
    success: bool
    message: str
    method: str
    dt: float
    metadata: Dict[str, Any]
    # Optional fields
    u: Optional[ArrayLike]  # Control sequence
    y: Optional[ArrayLike]  # Output sequence


class RolloutResultBase(TypedDict, total=False):
    """
    Base type for all rollout results (closed-loop simulation).

    Contains fields common to deterministic and stochastic rollout.
    """

    t: ArrayLike  # Regular time grid
    x: ArrayLike  # State trajectory
    u: ArrayLike  # Control (always present - from feedback)
    success: bool
    message: str
    method: str
    dt: float
    metadata: Dict[str, Any]
    controller_type: str
    closed_loop: bool  # Always True
    # Optional fields
    y: Optional[ArrayLike]


class DiscreteSimulationResultBase(TypedDict, total=False):
    """
    Base type for all discrete simulation results.

    Contains fields common to deterministic and stochastic discrete simulation.
    """

    t: ArrayLike  # Integer time steps [0, 1, 2, ...]
    x: ArrayLike  # State trajectory
    dt: float  # Sampling period
    success: bool
    message: str
    method: str
    metadata: Dict[str, Any]
    # Optional fields
    u: Optional[ArrayLike]  # Control sequence
    y: Optional[ArrayLike]  # Output sequence


# ============================================================================
# CONTINUOUS DETERMINISTIC RESULTS
# ============================================================================


class IntegrationResult(IntegrationResultBase):
    """
    Result from ODE integration with adaptive time stepping.

    Low-level integration interface with full solver diagnostics.
    Returned by ContinuousSystemBase.integrate() for deterministic systems.

    **NOTE**: This type is UNCHANGED from current version.

    Shape Convention
    ----------------
    Time-major ordering:
    - t: (T,) - Time points (irregular spacing from adaptive solver)
    - x: (T, nx) - State trajectory

    Examples
    --------
    >>> result: IntegrationResult = system.integrate(
    ...     x0=np.array([1.0, 0.0]),
    ...     u=None,
    ...     t_span=(0.0, 10.0),
    ...     method='RK45'
    ... )
    >>>
    >>> t = result['t']        # (T,) - irregular spacing
    >>> x = result['x']        # (T, nx)
    >>> nfev = result['nfev']  # Function evaluations
    """

    pass  # Inherits all fields from IntegrationResultBase


class SimulationResult(SimulationResultBase):
    """
    Result from ODE simulation with regular time grid.

    High-level simulation interface for deterministic continuous systems.
    Returned by ContinuousSystemBase.simulate() for open-loop control.

    **BREAKING CHANGES from v0.x**:
    - Keys: 'time' → 't', 'states' → 'x', 'controls' → 'u'
    - Shape: (nx, T) → (T, nx) for time-major ordering

    Shape Convention
    ----------------
    Time-major ordering:
    - t: (T,) - Regular time grid with spacing dt
    - x: (T, nx) - State trajectory
    - u: (T, nu) - Control sequence (if provided)

    Examples
    --------
    >>> result: SimulationResult = system.simulate(
    ...     x0=np.array([1.0, 0.0]),
    ...     u=None,
    ...     t_span=(0.0, 10.0),
    ...     dt=0.01
    ... )
    >>>
    >>> t = result['t']        # (1001,)
    >>> x = result['x']        # (1001, 2)
    >>> dt = result['dt']      # 0.01

    See Also
    --------
    RolloutResult : For closed-loop simulation
    SDESimulationResult : For stochastic simulation
    """

    pass  # Inherits all fields from SimulationResultBase


class RolloutResult(RolloutResultBase):
    """
    Result from ODE closed-loop simulation (state feedback).

    High-level interface for deterministic continuous systems with controller.
    Returned by ContinuousSystemBase.rollout() (NEW method in v1.0).

    Shape Convention
    ----------------
    Time-major ordering:
    - t: (T,) - Regular time grid
    - x: (T, nx) - State trajectory
    - u: (T, nu) - Feedback control applied

    Examples
    --------
    >>> def controller(x, t):
    ...     return -K @ x
    >>>
    >>> result: RolloutResult = system.rollout(
    ...     x0=np.array([1.0, 0.0]),
    ...     controller=controller,
    ...     t_span=(0.0, 10.0),
    ...     dt=0.01
    ... )
    >>>
    >>> t = result['t']        # (1001,)
    >>> x = result['x']        # (1001, 2)
    >>> u = result['u']        # (1001, 1) - always present

    See Also
    --------
    SimulationResult : For open-loop simulation
    SDERolloutResult : For stochastic rollout
    """

    pass  # Inherits all fields from RolloutResultBase


# ============================================================================
# CONTINUOUS STOCHASTIC RESULTS
# ============================================================================


class SDEIntegrationResult(IntegrationResultBase):
    """
    Result from SDE integration with adaptive time stepping.

    Low-level SDE integration with Brownian motion and diffusion diagnostics.
    Returned by ContinuousStochasticSystem.integrate().

    **NOTE**: This type is UNCHANGED from current version - already correct.

    Shape Convention
    ----------------
    Single trajectory (n_paths=1):
    - t: (T,)
    - x: (T, nx)

    Multiple trajectories (n_paths>1):
    - t: (T,)
    - x: (n_paths, T, nx)

    Attributes (SDE-specific extensions)
    ------------------------------------
    diffusion_evals : int
        Number of diffusion function evaluations
    noise_samples : ArrayLike
        Brownian motion samples used
    n_paths : int
        Number of Monte Carlo paths
    noise_type : str
        'additive' or 'multiplicative'
    sde_type : str
        'ito' or 'stratonovich'
    convergence_type : str
        'strong' or 'weak'

    Examples
    --------
    >>> # Single path
    >>> result: SDEIntegrationResult = sde_system.integrate(
    ...     x0=np.array([1.0, 0.0]),
    ...     u=None,
    ...     t_span=(0.0, 10.0),
    ...     method='euler_maruyama',
    ...     n_paths=1
    ... )
    >>> x = result['x']        # (T, nx)
    >>>
    >>> # Monte Carlo
    >>> result: SDEIntegrationResult = sde_system.integrate(
    ...     x0=np.array([1.0, 0.0]),
    ...     u=None,
    ...     t_span=(0.0, 10.0),
    ...     method='euler_maruyama',
    ...     n_paths=500,
    ...     seed=42
    ... )
    >>> x = result['x']        # (500, T, nx)
    >>> mean = x.mean(axis=0)  # (T, nx)
    """

    # SDE-specific fields
    diffusion_evals: int
    noise_samples: ArrayLike
    n_paths: int
    noise_type: str
    sde_type: str
    convergence_type: str


class SDESimulationResult(SimulationResultBase):
    """
    Result from SDE simulation with regular time grid (NEW in v1.0).

    High-level SDE simulation for stochastic continuous systems.
    Returned by ContinuousStochasticSystem.simulate() when n_paths > 1.

    Shape Convention
    ----------------
    Multiple trajectories:
    - t: (T,)
    - x: (n_paths, T, nx)
    - u: (n_paths, T, nu) if control varies by path

    Attributes (SDE-specific extensions)
    ------------------------------------
    n_paths : int
        Number of Monte Carlo paths (required, always >1)
    noise_type : str
        'additive' or 'multiplicative'
    sde_type : str
        'ito' or 'stratonovich'
    seed : Optional[int]
        Random seed used for reproducibility
    noise_samples : ArrayLike
        Brownian increments used
    diffusion_evals : int
        Number of diffusion function evaluations

    Examples
    --------
    >>> result: SDESimulationResult = sde_system.simulate(
    ...     x0=np.array([1.0, 0.0]),
    ...     u=None,
    ...     t_span=(0.0, 10.0),
    ...     dt=0.01,
    ...     n_paths=500,
    ...     seed=42
    ... )
    >>>
    >>> t = result['t']            # (1001,)
    >>> x = result['x']            # (500, 1001, 2)
    >>> n_paths = result['n_paths']  # 500
    >>>
    >>> # Compute statistics
    >>> mean_traj = x.mean(axis=0)  # (1001, 2)
    >>> std_traj = x.std(axis=0)    # (1001, 2)

    See Also
    --------
    SimulationResult : For deterministic simulation
    SDEIntegrationResult : For adaptive-grid SDE integration
    """

    # Required SDE fields
    n_paths: int
    noise_type: str
    sde_type: str
    # Optional SDE fields
    seed: Optional[int]
    noise_samples: ArrayLike
    diffusion_evals: int


class SDERolloutResult(RolloutResultBase):
    """
    Result from SDE closed-loop simulation with state feedback (NEW in v1.0).

    High-level SDE rollout for stochastic continuous systems with controller.
    Returned by ContinuousStochasticSystem.rollout() when n_paths > 1.

    Shape Convention
    ----------------
    Multiple trajectories:
    - t: (T,)
    - x: (n_paths, T, nx)
    - u: (n_paths, T, nu) - feedback control for each path

    Attributes (SDE-specific extensions)
    ------------------------------------
    Same as SDESimulationResult plus controller_type and closed_loop.

    Examples
    --------
    >>> def controller(x, t):
    ...     return -K @ x
    >>>
    >>> result: SDERolloutResult = sde_system.rollout(
    ...     x0=np.array([1.0, 0.0]),
    ...     controller=controller,
    ...     t_span=(0.0, 10.0),
    ...     dt=0.01,
    ...     n_paths=500,
    ...     seed=42
    ... )
    >>>
    >>> x = result['x']  # (500, 1001, 2)
    >>> u = result['u']  # (500, 1001, 1) - feedback for each path

    See Also
    --------
    RolloutResult : For deterministic rollout
    SDESimulationResult : For open-loop SDE simulation
    """

    # Required SDE fields
    n_paths: int
    noise_type: str
    sde_type: str
    # Optional SDE fields
    seed: Optional[int]
    noise_samples: ArrayLike
    diffusion_evals: int


# ============================================================================
# DISCRETE DETERMINISTIC RESULTS
# ============================================================================


class DiscreteSimulationResult(DiscreteSimulationResultBase):
    """
    Result from discrete-time simulation (deterministic).

    Returned by DiscreteSystemBase.simulate() for open-loop control.

    **BREAKING CHANGES from v0.x**:
    - Keys: 'time_steps' → 't', 'states' → 'x', 'controls' → 'u'
    - Shape: (nx, T) → (T, nx) for time-major ordering

    Shape Convention
    ----------------
    Time-major ordering:
    - t: (T,) - Integer time steps [0, 1, 2, ..., n_steps]
    - x: (T, nx) where T = n_steps+1
    - u: (T-1, nu) where T-1 = n_steps

    Examples
    --------
    >>> result: DiscreteSimulationResult = discrete_system.simulate(
    ...     x0=np.array([1.0, 0.0]),
    ...     u_sequence=np.zeros((100, 1)),
    ...     n_steps=100
    ... )
    >>>
    >>> t = result['t']        # [0, 1, ..., 100]
    >>> x = result['x']        # (101, 2)
    >>> u = result['u']        # (100, 1)

    See Also
    --------
    DiscreteRolloutResult : For closed-loop simulation
    """

    pass  # Inherits all fields from DiscreteSimulationResultBase


class DiscreteRolloutResult(DiscreteSimulationResultBase):
    """
    Result from discrete-time closed-loop simulation (state feedback).

    Returned by DiscreteSystemBase.rollout() for policy-based control.

    **NOTE**: v0.x rollout() already used time-major, so less change here.

    Shape Convention
    ----------------
    Time-major ordering:
    - t: (T,) - Integer time steps
    - x: (T, nx)
    - u: (T-1, nu) - feedback control applied

    Attributes (rollout-specific)
    ------------------------------
    policy_type : str
        Type of policy used
    closed_loop : bool
        Always True

    Examples
    --------
    >>> def policy(x, k):
    ...     return -K @ x
    >>>
    >>> result: DiscreteRolloutResult = discrete_system.rollout(
    ...     x0=np.array([1.0, 0.0]),
    ...     policy=policy,
    ...     n_steps=100
    ... )
    >>>
    >>> x = result['x']  # (101, 2)
    >>> u = result['u']  # (100, 1) - always present

    See Also
    --------
    DiscreteSimulationResult : For open-loop simulation
    """

    u: ArrayLike  # Required (not Optional) - feedback control
    policy_type: str
    closed_loop: bool


# ============================================================================
# DISCRETE STOCHASTIC RESULTS
# ============================================================================


class DiscreteStochasticSimulationResult(DiscreteSimulationResultBase):
    """
    Result from discrete stochastic simulation (NEW in v1.0).

    Returned by DiscreteStochasticSystem.simulate_stochastic().

    Shape Convention
    ----------------
    Multiple trajectories:
    - t: (T,) - Integer time steps
    - x: (n_paths, T, nx)
    - u: (n_paths, T-1, nu) if control varies

    Attributes (stochastic-specific)
    ---------------------------------
    n_paths : int
        Number of Monte Carlo paths
    noise_type : str
        Noise structure
    seed : Optional[int]
        Random seed used
    noise_samples : ArrayLike
        Noise realizations used

    Examples
    --------
    >>> result: DiscreteStochasticSimulationResult = discrete_stochastic.simulate_stochastic(
    ...     x0=np.array([1.0, 0.0]),
    ...     u_sequence=None,
    ...     n_steps=100,
    ...     n_paths=500,
    ...     seed=42
    ... )
    >>>
    >>> x = result['x']            # (500, 101, 2)
    >>> n_paths = result['n_paths']  # 500
    >>> mean_traj = x.mean(axis=0)  # (101, 2)

    See Also
    --------
    DiscreteSimulationResult : For deterministic discrete simulation
    """

    # Required stochastic fields
    n_paths: int
    noise_type: str
    # Optional stochastic fields
    seed: Optional[int]
    noise_samples: ArrayLike


class DiscreteStochasticRolloutResult(DiscreteSimulationResultBase):
    """
    Result from discrete stochastic closed-loop simulation (NEW in v1.0).

    Returned by DiscreteStochasticSystem.rollout() when supporting
    stochastic feedback policies with Monte Carlo.

    Shape Convention
    ----------------
    Multiple trajectories:
    - t: (T,)
    - x: (n_paths, T, nx)
    - u: (n_paths, T-1, nu) - feedback for each path

    Examples
    --------
    >>> def policy(x, k):
    ...     return -K @ x
    >>>
    >>> result: DiscreteStochasticRolloutResult = discrete_stochastic.rollout(
    ...     x0=np.array([1.0, 0.0]),
    ...     policy=policy,
    ...     n_steps=100,
    ...     n_paths=500,
    ...     seed=42
    ... )
    >>>
    >>> x = result['x']  # (500, 101, 2)
    >>> u = result['u']  # (500, 100, 1)

    See Also
    --------
    DiscreteRolloutResult : For deterministic rollout
    DiscreteStochasticSimulationResult : For open-loop stochastic
    """

    # Rollout-specific
    u: ArrayLike  # Required - feedback control
    policy_type: str
    closed_loop: bool
    # Stochastic-specific
    n_paths: int
    noise_type: str
    seed: Optional[int]
    noise_samples: ArrayLike


# ============================================================================
# UNION TYPES (for polymorphic code)
# ============================================================================

ContinuousIntegrationResultUnion = Union[IntegrationResult, SDEIntegrationResult]
"""Union of continuous integration results (deterministic or stochastic)."""

ContinuousSimulationResultUnion = Union[SimulationResult, SDESimulationResult]
"""Union of continuous simulation results (deterministic or stochastic)."""

ContinuousRolloutResultUnion = Union[RolloutResult, SDERolloutResult]
"""Union of continuous rollout results (deterministic or stochastic)."""

DiscreteSimulationResultUnion = Union[
    DiscreteSimulationResult,
    DiscreteStochasticSimulationResult,
]
"""Union of discrete simulation results (deterministic or stochastic)."""

DiscreteRolloutResultUnion = Union[
    DiscreteRolloutResult,
    DiscreteStochasticRolloutResult,
]
"""Union of discrete rollout results (deterministic or stochastic)."""

SystemResult = Union[
    # Continuous deterministic
    IntegrationResult,
    SimulationResult,
    RolloutResult,
    # Continuous stochastic
    SDEIntegrationResult,
    SDESimulationResult,
    SDERolloutResult,
    # Discrete deterministic
    DiscreteSimulationResult,
    DiscreteRolloutResult,
    # Discrete stochastic
    DiscreteStochasticSimulationResult,
    DiscreteStochasticRolloutResult,
]
"""
Union of all dynamical system execution result types.

Represents any result from executing a dynamical system via:
- integrate() - Adaptive-grid ODE/SDE integration
- simulate() - Regular-grid open-loop execution
- rollout() - Regular-grid closed-loop execution

All types share common fields: 't', 'x', 'success', 'message', 'metadata'

Use for maximally polymorphic functions that work with any system result.

Examples
--------
>>> def extract_final_state(result: SystemResult) -> np.ndarray:
...     '''Extract final state from any system result type.'''
...     x = result['x']
...     if x.ndim == 2:
...         return x[-1]  # Single trajectory
...     else:
...         return x[:, -1]  # Multiple paths (n_paths, T, nx)
>>> 
>>> def compute_trajectory_duration(result: SystemResult) -> float:
...     '''Compute time duration of any system result.'''
...     t = result['t']
...     if t.dtype == np.integer:
...         # Discrete system - use dt
...         return len(t) * result['dt']
...     else:
...         # Continuous system - use time array
...         return float(t[-1] - t[0])
"""


# ============================================================================
# EXPORT ALL
# ============================================================================

__all__ = [
    # Base types
    "IntegrationResultBase",
    "SimulationResultBase",
    "RolloutResultBase",
    "DiscreteSimulationResultBase",
    # Continuous deterministic
    "IntegrationResult",
    "SimulationResult",
    "RolloutResult",
    # Continuous stochastic
    "SDEIntegrationResult",
    "SDESimulationResult",
    "SDERolloutResult",
    # Discrete deterministic
    "DiscreteSimulationResult",
    "DiscreteRolloutResult",
    # Discrete stochastic
    "DiscreteStochasticSimulationResult",
    "DiscreteStochasticRolloutResult",
    # Union types
    "ContinuousIntegrationResultUnion",
    "ContinuousSimulationResultUnion",
    "ContinuousRolloutResultUnion",
    "DiscreteSimulationResultUnion",
    "DiscreteRolloutResultUnion",
    "SystemResult",
]
