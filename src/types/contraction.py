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
Contraction Theory Types

Result types for contraction analysis and control:
- Contraction analysis
- Control Contraction Metrics (CCM)
- Funnel control
- Incremental stability

Contraction theory provides powerful tools for analyzing convergence
and designing stabilizing controllers.

Mathematical Background
----------------------
Contraction Theory:
    Virtual displacement: δx = x₁ - x₂
    Riemannian metric: M(x) ≻ 0
    Metric norm: ||δx||²_M = δx' M(x) δx

    Contraction condition:
        d/dt ||δx||_M ≤ -β ||δx||_M

    Implies: ||δx(t)||_M ≤ e^(-βt) ||δx(0)||_M

    Consequence: All trajectories converge exponentially

Infinitesimal Contraction:
    For ẋ = f(x, t), define:
        F(x, t) = ∂f/∂x (Jacobian)

    Contraction if ∃M(x) ≻ 0:
        Ṁ + MF + F'M ≺ -2βM

    Or in steady-state coordinates:
        F + F' ≺ -2βI (Θ = M^(-1/2))

Control Contraction Metrics (CCM):
    For ẋ = f(x, u), find M(x) and u(x):
        Ṁ + M(∂f/∂x) + (∂f/∂x)'M ≺ -2βM

    Controller: u = k(x) such that closed-loop contracts

    Solved via convex optimization (LMI/SDP)

Funnel Control:
    Track reference x_d(t) within funnel:
        ||x(t) - x_d(t)||_M ≤ ρ(t)

    Funnel ρ(t): time-varying bound
    Prescribes transient performance

    Example: ρ(t) = (ρ₀ - ρ_∞)e^(-λt) + ρ_∞

Incremental Stability:
    System is δ-GAS if:
        ||x₁(t) - x₂(t)|| → 0  as  t → ∞

    For any initial conditions x₁(0), x₂(0)

    Stronger than GAS: all trajectories converge to each other

Usage
-----
>>> from src.types.contraction import (
...     ContractionAnalysisResult,
...     CCMResult,
...     FunnelingResult,
... )
>>>
>>> # Contraction analysis
>>> result: ContractionAnalysisResult = analyze_contraction(system)
>>> if result['is_contracting']:
...     print(f"Rate β = {result['contraction_rate']}")
>>>
>>> # CCM controller design
>>> ccm: CCMResult = design_ccm_controller(
...     system, contraction_rate=0.5
... )
>>> K = ccm['feedback_gain']
>>>
>>> # Funnel control
>>> funnel: FunnelingResult = design_funnel_controller(
...     system, reference, funnel_shape='exponential'
... )
>>> u = funnel['controller'](x, t)
"""

from typing import Callable, Optional

from typing_extensions import TypedDict

from .core import (
    ArrayLike,
    GainMatrix,
)
from .trajectories import (
    StateTrajectory,
)

# ============================================================================
# Type Aliases for Contraction
# ============================================================================

ContractionMetric = ArrayLike
"""
Contraction metric M(x) defining Riemannian geometry.

Symmetric positive definite matrix M(x) ≻ 0 defining metric:
    ||δx||²_M = δx' M(x) δx

Can be:
- Constant: M(x) = M (uniform metric)
- State-dependent: M(x) varies with state
- Matrix-valued function

Shape: (nx, nx) symmetric positive definite

Examples
--------
>>> # Euclidean metric (identity)
>>> M: ContractionMetric = np.eye(3)
>>> 
>>> # Weighted metric
>>> M: ContractionMetric = np.diag([2.0, 1.0, 0.5])
>>> 
>>> # State-dependent (as function)
>>> def metric(x: StateVector) -> ContractionMetric:
...     return np.eye(len(x)) * (1 + x[0]**2)
"""

ContractionRate = float
"""
Contraction rate β > 0.

Exponential convergence rate:
    ||δx(t)||_M ≤ e^(-βt) ||δx(0)||_M

Larger β → faster convergence

Examples
--------
>>> beta: ContractionRate = 0.5
>>> # 63% convergence in t = 1/β = 2 seconds
>>> # 95% convergence in t = 3/β = 6 seconds
"""


# ============================================================================
# Contraction Analysis
# ============================================================================


class ContractionAnalysisResult(TypedDict, total=False):
    """
    Contraction analysis result.

    Determines if system is contracting and computes contraction rate.

    Fields
    ------
    is_contracting : bool
        True if system is contracting
    contraction_rate : ContractionRate
        Exponential rate β
    metric : ContractionMetric
        Contraction metric M(x)
    metric_type : str
        Metric type ('constant', 'state_dependent', 'control_dependent')
    verification_method : str
        Method used ('LMI', 'SOS', 'optimization', 'analytic')
    convergence_bound : Optional[Callable]
        Upper bound ||δx(t)|| ≤ bound(t, ||δx(0)||)
    exponential_convergence : bool
        True if exponential (vs polynomial)
    incremental_stability : bool
        True if incrementally asymptotically stable
    condition_number : Optional[float]
        Condition number κ(M) if constant metric

    Examples
    --------
    >>> # Analyze system contraction
    >>> result: ContractionAnalysisResult = analyze_contraction(
    ...     dynamics=lambda x: -A @ x,
    ...     method='LMI'
    ... )
    >>>
    >>> if result['is_contracting']:
    ...     beta = result['contraction_rate']
    ...     M = result['metric']
    ...
    ...     print(f"System is contracting!")
    ...     print(f"Rate β = {beta:.3f}")
    ...     print(f"Method: {result['verification_method']}")
    ...
    ...     # Convergence bound
    ...     if 'convergence_bound' in result:
    ...         bound = result['convergence_bound']
    ...         t = np.linspace(0, 10, 100)
    ...         delta_bound = [bound(ti, 1.0) for ti in t]
    ...
    ...         import matplotlib.pyplot as plt
    ...         plt.plot(t, delta_bound)
    ...         plt.xlabel('Time')
    ...         plt.ylabel('||δx|| bound')
    ... else:
    ...     print("System is not contracting")
    """

    is_contracting: bool
    contraction_rate: ContractionRate
    metric: ContractionMetric
    metric_type: str
    verification_method: str
    convergence_bound: Optional[Callable]
    exponential_convergence: bool
    incremental_stability: bool
    condition_number: Optional[float]


# ============================================================================
# Control Contraction Metrics (CCM)
# ============================================================================


class CCMResult(TypedDict, total=False):
    """
    Control Contraction Metrics (CCM) result.

    Controller synthesis ensuring contraction of closed-loop system.

    Fields
    ------
    feedback_gain : GainMatrix
        State-dependent K(x) or constant K
    metric : ContractionMetric
        Contraction metric M(x)
    contraction_rate : ContractionRate
        Guaranteed contraction rate β
    metric_condition_number : ArrayLike
        Condition number κ(M(x)) over state space
    contraction_verified : bool
        LMI/SOS verification succeeded
    robustness_margin : float
        Margin in contraction condition
    geodesic_distance : Optional[Callable]
        Distance in M-metric: d_M(x₁, x₂)

    Examples
    --------
    >>> # Design CCM controller
    >>> result: CCMResult = design_ccm_controller(
    ...     system=pendulum,
    ...     contraction_rate=0.5,
    ...     method='SDP'
    ... )
    >>>
    >>> if result['contraction_verified']:
    ...     K = result['feedback_gain']
    ...     M = result['metric']
    ...     beta = result['contraction_rate']
    ...
    ...     print(f"CCM controller designed!")
    ...     print(f"Contraction rate: {beta:.3f}")
    ...     print(f"Robustness margin: {result['robustness_margin']:.3f}")
    ...
    ...     # Apply controller
    ...     def controller(x):
    ...         if callable(K):
    ...             return K(x)  # State-dependent
    ...         else:
    ...             return K @ x  # Constant
    ...
    ...     # Geodesic distance (if available)
    ...     if 'geodesic_distance' in result:
    ...         d_M = result['geodesic_distance']
    ...         x1 = np.array([1.0, 0.5])
    ...         x2 = np.array([0.5, 0.3])
    ...         dist = d_M(x1, x2)
    ...         print(f"Distance in M-metric: {dist:.3f}")
    """

    feedback_gain: GainMatrix
    metric: ContractionMetric
    contraction_rate: ContractionRate
    metric_condition_number: ArrayLike
    contraction_verified: bool
    robustness_margin: float
    geodesic_distance: Optional[Callable]


# ============================================================================
# Funnel Control
# ============================================================================


class FunnelingResult(TypedDict):
    """
    Funnel control result.

    Tracking controller with prescribed performance bounds.

    Fields
    ------
    controller : Callable
        Funnel controller u(x, t)
    tracking_funnel : Callable
        Funnel boundary ρ(t)
    funnel_shape : str
        Funnel type ('exponential', 'polynomial', 'custom')
    reference_trajectory : StateTrajectory
        Desired trajectory x_d(t)
    performance_bound : Callable
        Tracking bound ||x(t) - x_d(t)|| ≤ bound(t)
    transient_bound : float
        Initial error amplification factor
    contraction_rate : ContractionRate
        Asymptotic contraction rate

    Examples
    --------
    >>> # Design funnel controller
    >>> result: FunnelingResult = design_funnel_controller(
    ...     system=robot,
    ...     reference_trajectory=x_desired,
    ...     funnel_shape='exponential',
    ...     initial_funnel_width=1.0,
    ...     final_funnel_width=0.1,
    ...     convergence_rate=0.5
    ... )
    >>>
    >>> controller = result['controller']
    >>> funnel = result['tracking_funnel']
    >>>
    >>> # Simulate closed-loop
    >>> t_sim = np.linspace(0, 10, 1000)
    >>> x = x0
    >>> x_traj = [x]
    >>>
    >>> for t in t_sim[1:]:
    ...     u = controller(x, t)
    ...     x = system.step(x, u, dt)
    ...     x_traj.append(x)
    >>>
    >>> x_traj = np.array(x_traj)
    >>> x_d = result['reference_trajectory']
    >>>
    >>> # Verify funnel constraint
    >>> import matplotlib.pyplot as plt
    >>> error = np.linalg.norm(x_traj - x_d, axis=1)
    >>> bound = np.array([funnel(t) for t in t_sim])
    >>>
    >>> plt.plot(t_sim, error, label='Actual error')
    >>> plt.plot(t_sim, bound, 'r--', label='Funnel bound')
    >>> plt.fill_between(t_sim, 0, bound, alpha=0.3)
    >>> plt.xlabel('Time')
    >>> plt.ylabel('Tracking error')
    >>> plt.legend()
    >>>
    >>> # Check: error should stay within funnel
    >>> assert np.all(error <= bound + 1e-6)
    """

    controller: Callable
    tracking_funnel: Callable
    funnel_shape: str
    reference_trajectory: StateTrajectory
    performance_bound: Callable
    transient_bound: float
    contraction_rate: ContractionRate


# ============================================================================
# Incremental Stability
# ============================================================================


class IncrementalStabilityResult(TypedDict):
    """
    Incremental stability analysis result.

    Analyzes convergence of trajectories to each other.

    Fields
    ------
    incrementally_stable : bool
        True if δ-GAS (incremental global asymptotic stability)
    contraction_rate : Optional[ContractionRate]
        Exponential rate (if contracting)
    metric : Optional[ContractionMetric]
        Metric M(x) used for analysis
    kl_bound : Optional[Callable]
        KL stability bound β(||δx(0)||, t)
    convergence_type : str
        Type of convergence ('exponential', 'asymptotic', 'finite_time')

    Examples
    --------
    >>> # Check incremental stability
    >>> result: IncrementalStabilityResult = check_incremental_stability(
    ...     system=nonlinear_system,
    ...     method='contraction'
    ... )
    >>>
    >>> if result['incrementally_stable']:
    ...     print("All trajectories converge to each other!")
    ...
    ...     conv_type = result['convergence_type']
    ...     print(f"Convergence type: {conv_type}")
    ...
    ...     if conv_type == 'exponential':
    ...         beta = result['contraction_rate']
    ...         print(f"Exponential rate: {beta:.3f}")
    ...         print(f"Bound: ||x₁(t) - x₂(t)|| ≤ e^(-{beta}t) ||x₁(0) - x₂(0)||")
    ...
    ...     # KL bound (if available)
    ...     if 'kl_bound' in result:
    ...         kl = result['kl_bound']
    ...         # β(||δx(0)||, t) → 0 as t → ∞
    ... else:
    ...     print("System is not incrementally stable")
    """

    incrementally_stable: bool
    contraction_rate: Optional[ContractionRate]
    metric: Optional[ContractionMetric]
    kl_bound: Optional[Callable]
    convergence_type: str


# ============================================================================
# Export All
# ============================================================================

__all__ = [
    # Type aliases
    "ContractionMetric",
    "ContractionRate",
    # Analysis results
    "ContractionAnalysisResult",
    "CCMResult",
    "FunnelingResult",
    "IncrementalStabilityResult",
]
