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
Differential Flatness Types

Result types for differential flatness and trajectory planning:
- Flatness analysis and verification
- Trajectory planning via flatness
- Flat output parameterization

Differential flatness enables exact trajectory planning and feedforward
control design.

Mathematical Background
----------------------
Differential Flatness:
    System ẋ = f(x, u) is differentially flat if ∃ flat output y_flat:

    1. y_flat = σ(x)  (function of state)
    2. x = φ_x(y_flat, ẏ_flat, ÿ_flat, ..., y_flat^(r))
    3. u = φ_u(y_flat, ẏ_flat, ÿ_flat, ..., y_flat^(r))

    State and control are algebraic functions of y_flat and derivatives.
    No integration required!

Properties:
    - dim(y_flat) = m (number of inputs)
    - Every state trajectory ↔ unique flat output trajectory
    - Enables motion planning in flat output space
    - Feedforward control via inversion

Common Flat Systems:
    1. Linear controllable: y_flat = Cx (any output)
    2. Quadrotor: y_flat = [x, y, z, ψ] (position + yaw)
    3. Unicycle: y_flat = [x, y] (position)
    4. PVTOL aircraft: y_flat = [x, z] (horizontal + vertical)
    5. Crane: y_flat = [cart_position, load_angle]

Trajectory Planning:
    1. Define desired flat trajectory: y_flat(t)
    2. Compute derivatives: ẏ_flat(t), ÿ_flat(t), ...
    3. Invert: x(t) = φ_x(...), u(t) = φ_u(...)
    4. Result: Open-loop trajectory (x(t), u(t))

Example - Quadrotor:
    Flat output: y_flat = [x, y, z, ψ]

    From y_flat, compute:
        - Roll φ, Pitch θ from [ẍ, ÿ, z̈]
        - Thrust T from desired acceleration
        - Angular rates from time derivatives

    Plan trajectory in (x, y, z, ψ) space → get full state + control

Usage
-----
>>> from src.types.flatness import (
...     DifferentialFlatnessResult,
...     TrajectoryPlanningResult,
... )
>>>
>>> # Check flatness
>>> result: DifferentialFlatnessResult = check_flatness(system)
>>> if result['is_flat']:
...     y_flat_func = result['flat_output']
...     phi_x = result['state_from_flat']
...     phi_u = result['control_from_flat']
>>>
>>> # Plan trajectory using flatness
>>> traj: TrajectoryPlanningResult = plan_flat_trajectory(
...     system, x_initial, x_final, time_horizon=5.0
... )
>>> x_ref = traj['state_trajectory']
>>> u_ff = traj['control_trajectory']
"""

from typing import Callable, Optional

from typing_extensions import TypedDict

from .core import (
    ArrayLike,
)
from .trajectories import (
    ControlSequence,
    StateTrajectory,
    TimePoints,
)

# ============================================================================
# Type Aliases for Flatness
# ============================================================================

FlatnessOutput = ArrayLike
"""
Flat output for differentially flat systems.

Special output y_flat such that state and control can be expressed
algebraically in terms of y_flat and its derivatives.

Algebraic relationships:
    x = φ_x(y_flat, ẏ_flat, ÿ_flat, ..., y^(r)_flat)
    u = φ_u(y_flat, ẏ_flat, ÿ_flat, ..., y^(r)_flat)

Dimension: (m,) where m = number of inputs

Examples
--------
>>> # Quadrotor: position + yaw
>>> y_flat: FlatnessOutput = np.array([x, y, z, psi])
>>> 
>>> # Unicycle: position
>>> y_flat: FlatnessOutput = np.array([x, y])
>>> 
>>> # Double integrator: position
>>> y_flat: FlatnessOutput = np.array([x])
"""


# ============================================================================
# Differential Flatness Analysis
# ============================================================================


class DifferentialFlatnessResult(TypedDict):
    """
    Differential flatness analysis result.

    Determines if system is differentially flat and provides
    flat output mappings.

    Fields
    ------
    is_flat : bool
        True if system is differentially flat
    flat_output : Optional[Callable]
        Flat output function y_flat = σ(x)
    flat_dimension : int
        Dimension of flat output (typically = nu)
    differential_order : int
        Maximum derivative order r needed
    state_from_flat : Optional[Callable]
        Inverse map x = φ_x(y, ẏ, ÿ, ...)
    control_from_flat : Optional[Callable]
        Inverse map u = φ_u(y, ẏ, ÿ, ...)
    verification_method : str
        How flatness was verified ('analytic', 'symbolic', 'numerical')

    Examples
    --------
    >>> # Check if system is flat
    >>> result: DifferentialFlatnessResult = check_flatness(
    ...     system=quadrotor,
    ...     method='analytic'
    ... )
    >>>
    >>> if result['is_flat']:
    ...     print(f"System is differentially flat!")
    ...     print(f"Flat output dimension: {result['flat_dimension']}")
    ...     print(f"Max derivative order: {result['differential_order']}")
    ...
    ...     # Extract mappings
    ...     sigma = result['flat_output']        # y = σ(x)
    ...     phi_x = result['state_from_flat']    # x = φ_x(y, ẏ, ...)
    ...     phi_u = result['control_from_flat']  # u = φ_u(y, ẏ, ...)
    ...
    ...     # Use for trajectory planning
    ...     # 1. Plan desired y_flat(t)
    ...     t = np.linspace(0, 5, 100)
    ...     y_flat_traj = plan_flat_output(t)
    ...
    ...     # 2. Compute derivatives
    ...     dy_flat = compute_derivatives(y_flat_traj, dt)
    ...     ddy_flat = compute_derivatives(dy_flat, dt)
    ...
    ...     # 3. Invert to get x(t) and u(t)
    ...     x_traj = phi_x(y_flat_traj, dy_flat, ddy_flat)
    ...     u_traj = phi_u(y_flat_traj, dy_flat, ddy_flat)
    ... else:
    ...     print("System is not differentially flat")
    """

    is_flat: bool
    flat_output: Optional[Callable]
    flat_dimension: int
    differential_order: int
    state_from_flat: Optional[Callable]
    control_from_flat: Optional[Callable]
    verification_method: str


# ============================================================================
# Trajectory Planning
# ============================================================================


class TrajectoryPlanningResult(TypedDict, total=False):
    """
    Trajectory planning result.

    Plans feasible trajectory from initial to final state,
    often using differential flatness.

    Fields
    ------
    state_trajectory : StateTrajectory
        Planned state trajectory x(t) or x[k]
    control_trajectory : ControlSequence
        Required control trajectory u(t) or u[k]
    flat_trajectory : Optional[ArrayLike]
        Flat output trajectory y_flat(t) if flat system
    time_points : TimePoints
        Time discretization (N+1,)
    cost : float
        Trajectory cost J
    feasible : bool
        Satisfies all constraints
    method : str
        Planning method used ('flatness', 'optimization', 'RRT', etc.)
    computation_time : float
        Planning time in seconds

    Examples
    --------
    >>> # Plan trajectory using flatness
    >>> result: TrajectoryPlanningResult = plan_flat_trajectory(
    ...     system=quadrotor,
    ...     x_initial=np.zeros(12),
    ...     x_final=np.array([10, 10, 5, 0, ...]),
    ...     time_horizon=5.0,
    ...     dt=0.01
    ... )
    >>>
    >>> if result['feasible']:
    ...     # Extract trajectory
    ...     x_ref = result['state_trajectory']
    ...     u_ff = result['control_trajectory']
    ...     t = result['time_points']
    ...
    ...     print(f"Planning time: {result['computation_time']:.3f}s")
    ...     print(f"Trajectory cost: {result['cost']:.2f}")
    ...     print(f"Method: {result['method']}")
    ...
    ...     # Execute with feedforward + feedback
    ...     x = x_initial
    ...     for k in range(len(t)-1):
    ...         # Feedforward from plan
    ...         u_ff_k = u_ff[k]
    ...
    ...         # Feedback correction
    ...         e = x - x_ref[k]
    ...         u_fb = -K @ e
    ...
    ...         # Combined control
    ...         u = u_ff_k + u_fb
    ...         x = system.step(x, u, dt)
    ...
    ...     # Visualize flat trajectory
    ...     if 'flat_trajectory' in result:
    ...         y_flat = result['flat_trajectory']
    ...         import matplotlib.pyplot as plt
    ...         plt.plot(y_flat[:, 0], y_flat[:, 1], label='Flat trajectory')
    ...         plt.xlabel('x position')
    ...         plt.ylabel('y position')
    ... else:
    ...     print("No feasible trajectory found")
    """

    state_trajectory: StateTrajectory
    control_trajectory: ControlSequence
    flat_trajectory: Optional[ArrayLike]
    time_points: TimePoints
    cost: float
    feasible: bool
    method: str
    computation_time: float


# ============================================================================
# Export All
# ============================================================================

__all__ = [
    # Type aliases
    "FlatnessOutput",
    # Results
    "DifferentialFlatnessResult",
    "TrajectoryPlanningResult",
]
