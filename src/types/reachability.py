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
Reachability and Safety Analysis Types

Result types for reachability analysis, safety verification, and
barrier certificate methods:
- Forward/backward reachable sets
- Region of Attraction (ROA)
- Barrier certificates
- Control Barrier Functions (CBF)
- Control Lyapunov Functions (CLF)
- Formal verification

These methods ensure safety and stability in autonomous systems.

Mathematical Background
----------------------
Reachable Set:
    Forward reachable set from x0:
        Reach(x0, [0,T]) = {x(T) : x(0) = x0, u(t) ∈ U, t ∈ [0,T]}

    Backward reachable set to target:
        Reach^(-1)(X_T, [0,T]) = {x0 : ∃u(·) s.t. x(T) ∈ X_T}

    Reachable tube (all intermediate times):
        Tube(x0, [0,T]) = ⋃_{t∈[0,T]} Reach(x0, [0,t])

Region of Attraction (ROA):
    For equilibrium x_eq with Lyapunov V(x):
        ROA = {x : V(x) ≤ c, V̇(x) < 0}

    Largest invariant set where trajectories converge to x_eq

Barrier Certificates:
    Safety: separate safe set S from unsafe U
    Barrier function B(x) satisfies:
        1. B(x) > 0  ∀x ∈ S  (positive on safe)
        2. B(x) < 0  ∀x ∈ U  (negative on unsafe)
        3. Ḃ(x) ≤ 0  ∀x ∈ ∂S (decreasing on boundary)

    If exists → S ∩ U = ∅ (safety guaranteed)

Control Barrier Functions (CBF):
    For safety-critical control:
        B̈(x) + γ Ḃ(x) ≥ 0  (exponential decrease condition)

    Safety filter:
        u* = argmin ||u - u_nom||²
             s.t. Ḃ(x,u) ≥ -γB(x)

Control Lyapunov Functions (CLF):
    For stabilization:
        V̇(x,u) ≤ -αV(x)  (exponential decrease)

    Stabilizing control:
        u* = argmin ||u||²
             s.t. V̇(x,u) ≤ -αV(x)

Usage
-----
>>> from src.types.reachability import (
...     ReachabilityResult,
...     ROAResult,
...     CBFResult,
... )
>>>
>>> # Reachability analysis
>>> result: ReachabilityResult = compute_reachable_set(
...     system, x0, u_bounds, horizon=10
... )
>>> tube = result['reachable_tube']  # Set at each time
>>>
>>> # Safety via CBF
>>> cbf_result: CBFResult = cbf.filter_control(x, u_desired)
>>> u_safe = cbf_result['safe_control']
>>> if cbf_result['constraint_active']:
...     print("Safety constraint engaged!")
"""

from typing import Callable, List, Optional

from typing_extensions import TypedDict

from .core import (
    ArrayLike,
    ControlVector,
    CovarianceMatrix,
    StateVector,
)
from .trajectories import (
    StateTrajectory,
)

# ============================================================================
# Type Aliases for Sets
# ============================================================================

ReachableSet = ArrayLike
"""
Reachable set representation.

Can be represented as:
- Polytope: vertices (n_vertices, nx)
- Ellipsoid: {x : (x-c)'P^(-1)(x-c) ≤ 1}
- Zonotope: center + generators
- Grid samples: (n_samples, nx)

Examples
--------
>>> # Polytope vertices
>>> reachable: ReachableSet = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
>>> 
>>> # Grid sampling
>>> reachable: ReachableSet = np.random.randn(1000, 2)
"""

SafeSet = ArrayLike
"""
Safe set representation (invariant/viability).

Regions guaranteed safe under system dynamics.

Examples
--------
>>> # Polytope safe region
>>> safe: SafeSet = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])
"""


# ============================================================================
# Reachability Analysis
# ============================================================================


class ReachabilityResult(TypedDict, total=False):
    """
    Reachability analysis result.

    Computes forward/backward reachable sets over time horizon.

    Fields
    ------
    reachable_set : ReachableSet
        Reachable set at final time T
    reachable_tube : List[ReachableSet]
        Reachable set at each time t ∈ [0,T]
    volume : float
        Volume/measure of reachable set
    representation : str
        Set representation type ('polytope', 'ellipsoid', 'zonotope', 'grid')
    method : str
        Method used ('Hamilton-Jacobi', 'Ellipsoidal', 'Zonotope', 'sampling')
    computation_time : float
        Computation time in seconds

    Examples
    --------
    >>> # Forward reachability
    >>> result: ReachabilityResult = compute_reachable_set(
    ...     system=pendulum,
    ...     x0=np.array([0.1, 0.0]),
    ...     u_bounds=[-1, 1],
    ...     horizon=10,
    ...     dt=0.1
    ... )
    >>>
    >>> # Visualize reachable tube
    >>> import matplotlib.pyplot as plt
    >>> for i, reach_set in enumerate(result['reachable_tube']):
    ...     plt.plot(reach_set[:, 0], reach_set[:, 1],
    ...              alpha=0.3, color='blue')
    >>>
    >>> print(f"Final volume: {result['volume']:.3f}")
    >>> print(f"Method: {result['method']}")
    >>>
    >>> # Check if target reached
    >>> x_target = np.array([0, 0])
    >>> final_set = result['reachable_set']
    >>> # Check if x_target in convex hull of final_set
    """

    reachable_set: ReachableSet
    reachable_tube: List[ReachableSet]
    volume: float
    representation: str
    method: str
    computation_time: float


# ============================================================================
# Region of Attraction
# ============================================================================


class ROAResult(TypedDict):
    """
    Region of Attraction (ROA) analysis result.

    Estimates basin of attraction for equilibrium point via Lyapunov analysis.

    Fields
    ------
    region_of_attraction : SafeSet
        ROA estimate (level set of Lyapunov function)
    lyapunov_function : Callable
        V(x) Lyapunov function
    lyapunov_matrix : CovarianceMatrix
        P matrix if V(x) = x'Px (quadratic)
    level_set : float
        c where {x : V(x) ≤ c} defines ROA
    volume_estimate : float
        Estimated volume of ROA
    verification_samples : int
        Number of samples used for verification
    certification_method : str
        Method used ('SOS', 'sampling', 'LMI', 'bisection')

    Examples
    --------
    >>> # Quadratic Lyapunov function
    >>> P = np.array([[2, 0], [0, 1]])
    >>> V = lambda x: x.T @ P @ x
    >>>
    >>> result: ROAResult = compute_roa(
    ...     system=pendulum,
    ...     equilibrium=np.array([0, 0]),
    ...     lyapunov_function=V,
    ...     method='SOS'
    ... )
    >>>
    >>> # Extract ROA
    >>> roa = result['region_of_attraction']
    >>> c = result['level_set']
    >>>
    >>> # Check if state in ROA
    >>> x_test = np.array([0.5, 0.1])
    >>> if V(x_test) <= c:
    ...     print("State in ROA - will converge to equilibrium")
    >>>
    >>> print(f"ROA volume: {result['volume_estimate']:.3f}")
    >>> print(f"Certified by: {result['certification_method']}")
    """

    region_of_attraction: SafeSet
    lyapunov_function: Callable
    lyapunov_matrix: CovarianceMatrix
    level_set: float
    volume_estimate: float
    verification_samples: int
    certification_method: str


# ============================================================================
# Formal Verification
# ============================================================================


class VerificationResult(TypedDict, total=False):
    """
    Formal verification result.

    Proves or disproves safety/reachability properties.

    Fields
    ------
    verified : bool
        True if property verified
    property_type : str
        Type of property ('safety', 'reachability', 'liveness', 'invariance')
    confidence : float
        Confidence level (0-1), 1.0 = formal proof
    counterexample : Optional[StateTrajectory]
        Trajectory violating property (if not verified)
    certification_method : str
        Method used ('SOS', 'SMT', 'abstract-interpretation', 'reachability')
    computation_time : float
        Verification time in seconds

    Examples
    --------
    >>> # Verify safety property
    >>> safe_region = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])
    >>>
    >>> result: VerificationResult = verify_safety(
    ...     system=quad rotor,
    ...     initial_set=x0_bounds,
    ...     safe_set=safe_region,
    ...     horizon=50,
    ...     method='reachability'
    ... )
    >>>
    >>> if result['verified']:
    ...     print(f"Safety certified! (confidence: {result['confidence']})")
    ... else:
    ...     print("Safety violation found:")
    ...     counterex = result['counterexample']
    ...     plt.plot(counterex[:, 0], counterex[:, 1])
    >>>
    >>> # Verify reachability (can we reach target?)
    >>> reach_result: VerificationResult = verify_reachability(
    ...     system, x0, target_set, horizon=100
    ... )
    >>> if reach_result['verified']:
    ...     print("Target is reachable!")
    """

    verified: bool
    property_type: str
    confidence: float
    counterexample: Optional[StateTrajectory]
    certification_method: str
    computation_time: float


# ============================================================================
# Barrier Certificates
# ============================================================================


class BarrierCertificateResult(TypedDict):
    """
    Barrier certificate result.

    Barrier function separating safe and unsafe regions.

    Fields
    ------
    barrier_function : Callable[[StateVector], float]
        B(x) barrier function
    barrier_matrix : Optional[CovarianceMatrix]
        P if B(x) = x'Px (quadratic barrier)
    valid : bool
        Barrier conditions satisfied
    safe_set : SafeSet
        Certified safe region {x : B(x) > 0}
    unsafe_set : Optional[ArrayLike]
        Unsafe region {x : B(x) < 0}
    method : str
        Synthesis method ('SOS', 'LP', 'neural', 'convex')

    Examples
    --------
    >>> # Find barrier separating safe/unsafe
    >>> safe = np.array([[0, 0], [5, 0], [5, 5], [0, 5]])
    >>> unsafe = np.array([[8, 8], [10, 8], [10, 10], [8, 10]])
    >>>
    >>> result: BarrierCertificateResult = find_barrier_certificate(
    ...     system=robot,
    ...     safe_set=safe,
    ...     unsafe_set=unsafe,
    ...     method='SOS'
    ... )
    >>>
    >>> if result['valid']:
    ...     B = result['barrier_function']
    ...
    ...     # Verify safety
    ...     x_test = np.array([2, 2])
    ...     if B(x_test) > 0:
    ...         print("State is safe")
    ...
    ...     # Visualize barrier
    ...     x1 = np.linspace(0, 10, 50)
    ...     x2 = np.linspace(0, 10, 50)
    ...     X1, X2 = np.meshgrid(x1, x2)
    ...     Z = np.array([[B(np.array([x, y]))
    ...                    for x, y in zip(x1_row, x2_row)]
    ...                   for x1_row, x2_row in zip(X1, X2)])
    ...     plt.contour(X1, X2, Z, levels=[0], colors='red')
    """

    barrier_function: Callable[[StateVector], float]
    barrier_matrix: Optional[CovarianceMatrix]
    valid: bool
    safe_set: SafeSet
    unsafe_set: Optional[ArrayLike]
    method: str


# ============================================================================
# Control Barrier Functions
# ============================================================================


class CBFResult(TypedDict):
    """
    Control Barrier Function (CBF) result.

    Safety filter for control ensuring forward invariance of safe set.

    Fields
    ------
    safe_control : ControlVector
        Safety-filtered control u_safe
    barrier_value : float
        B(x) at current state
    barrier_derivative : float
        dB/dt or ΔB under current dynamics
    constraint_active : bool
        True if safety constraint is active
    nominal_control : ControlVector
        Original desired control u_nom
    modification_magnitude : float
        ||u_safe - u_nom|| control modification

    Examples
    --------
    >>> # Define barrier function (distance to obstacle)
    >>> obstacle_center = np.array([5, 5])
    >>> obstacle_radius = 2.0
    >>> B = lambda x: np.linalg.norm(x - obstacle_center)**2 - obstacle_radius**2
    >>>
    >>> # Create CBF controller
    >>> cbf = ControlBarrierFunction(barrier=B, system=robot, gamma=1.0)
    >>>
    >>> # Filter control at each step
    >>> x = np.array([4, 3])
    >>> u_desired = np.array([1, 1])  # Nominal control (move toward obstacle)
    >>>
    >>> result: CBFResult = cbf.filter_control(x, u_desired)
    >>>
    >>> u_safe = result['safe_control']
    >>> print(f"Barrier value: {result['barrier_value']:.3f}")
    >>>
    >>> if result['constraint_active']:
    ...     print("Safety filter active!")
    ...     print(f"Control modified by: {result['modification_magnitude']:.3f}")
    ... else:
    ...     print("Nominal control is safe")
    >>>
    >>> # Apply safe control
    >>> x_next = robot.step(x, u_safe)
    """

    safe_control: ControlVector
    barrier_value: float
    barrier_derivative: float
    constraint_active: bool
    nominal_control: ControlVector
    modification_magnitude: float


# ============================================================================
# Control Lyapunov Functions
# ============================================================================


class CLFResult(TypedDict):
    """
    Control Lyapunov Function (CLF) result.

    Stabilizing control with guaranteed convergence.

    Fields
    ------
    stabilizing_control : ControlVector
        Control ensuring V̇ < 0
    lyapunov_value : float
        V(x) at current state
    lyapunov_derivative : float
        V̇(x, u) under current control
    stability_margin : float
        How much V̇ < 0 (larger = faster convergence)
    convergence_rate : float
        Exponential rate α in V̇ ≤ -αV
    feasible : bool
        CLF condition feasible

    Examples
    --------
    >>> # Quadratic Lyapunov function
    >>> P = np.array([[2, 0], [0, 1]])
    >>> V = lambda x: x.T @ P @ x
    >>>
    >>> # Create CLF controller
    >>> clf = ControlLyapunovFunction(lyapunov=V, system=pendulum, alpha=0.5)
    >>>
    >>> # Compute stabilizing control
    >>> x = np.array([0.5, 0.1])
    >>>
    >>> result: CLFResult = clf.compute_control(x)
    >>>
    >>> if result['feasible']:
    ...     u = result['stabilizing_control']
    ...     print(f"V(x) = {result['lyapunov_value']:.3f}")
    ...     print(f"V̇(x,u) = {result['lyapunov_derivative']:.3f}")
    ...     print(f"Convergence rate: {result['convergence_rate']:.2f}")
    ... else:
    ...     print("CLF condition infeasible")
    """

    stabilizing_control: ControlVector
    lyapunov_value: float
    lyapunov_derivative: float
    stability_margin: float
    convergence_rate: float
    feasible: bool


# ============================================================================
# Export All
# ============================================================================

__all__ = [
    # Set representations
    "ReachableSet",
    "SafeSet",
    # Analysis results
    "ReachabilityResult",
    "ROAResult",
    "VerificationResult",
    "BarrierCertificateResult",
    "CBFResult",
    "CLFResult",
]
