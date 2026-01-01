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
Robustness and Uncertainty Quantification Types

Result types for robust control and uncertainty analysis:
- Robust stability analysis
- Structured Singular Value (μ-analysis)
- Tube-based MPC
- Stochastic MPC
- Risk-sensitive control

These methods handle model uncertainty and disturbances.

Mathematical Background
----------------------
Parametric Uncertainty:
    Uncertain system: ẋ = A(θ)x + B(θ)u
    Parameter set: θ ∈ Θ ⊂ ℝ^p

    Robust stability: stable ∀θ ∈ Θ
    Worst-case performance: sup_{θ∈Θ} J(θ)

Structured Singular Value (μ):
    For M(s) and structured uncertainty Δ:
        μ_Δ(M) = 1/min{σ̄(Δ) : det(I - MΔ) = 0}

    Robust stability: μ(M) < 1
    Robust performance: μ(N) < 1

    Upper bound (from H∞): μ ≤ σ̄(M)
    Computing μ is NP-hard → upper/lower bounds

Tube MPC:
    Nominal system: x̄[k+1] = Ax̄[k] + Bu[k]
    True system: x[k+1] = Ax[k] + Bu[k] + w[k]

    Invariant tube: ||x[k] - x̄[k]|| ≤ α_k

    Control: u[k] = v[k] + K(x[k] - x̄[k])
    - v[k]: nominal control (MPC)
    - K: ancillary feedback (pre-stabilizing)

    Constraints tightened by tube size

Stochastic MPC:
    Chance constraints: P(g(x,u) ≤ 0) ≥ 1 - ε

    Gaussian: g(x,u) + Φ^{-1}(1-ε)σ_g ≤ 0

    Cost: E[J] or E[J] + λ·Var[J] (risk-sensitive)

Usage
-----
>>> from src.types.robustness import (
...     RobustStabilityResult,
...     TubeMPCResult,
...     StochasticMPCResult,
... )
>>>
>>> # Robust stability analysis
>>> result: RobustStabilityResult = analyze_robust_stability(
...     system, uncertainty_set
... )
>>> if result['robustly_stable']:
...     print(f"Margin: {result['stability_margin']:.3f}")
>>>
>>> # Tube MPC
>>> tube_result: TubeMPCResult = tube_mpc.solve(x0, w_bound)
>>> u_nominal = tube_result['nominal_control'][0]
>>> K = tube_result['feedback_control']
"""

from typing import List, Optional, Tuple, Union

from typing_extensions import TypedDict

from .core import (
    ArrayLike,
    CovarianceMatrix,
    GainMatrix,
    ParameterVector,
)
from .trajectories import (
    ControlSequence,
    StateTrajectory,
)

# ============================================================================
# Type Aliases for Uncertainty
# ============================================================================

UncertaintySet = Union[ArrayLike, Tuple[ArrayLike, ArrayLike]]
"""
Parametric uncertainty set representation.

Can be:
- Polytope: vertices (n_vertices, n_params)
- Ellipsoid: (center, shape_matrix)
- Box: (lower_bounds, upper_bounds)

Examples
--------
>>> # Box uncertainty: θ ∈ [θ_min, θ_max]
>>> uncertainty: UncertaintySet = (
...     np.array([0.8, 1.8]),  # lower
...     np.array([1.2, 2.2])   # upper
... )
>>> 
>>> # Ellipsoid: (θ - θ_nom)'P^{-1}(θ - θ_nom) ≤ 1
>>> uncertainty: UncertaintySet = (
...     np.array([1.0, 2.0]),  # center
...     np.diag([0.1, 0.2])    # shape matrix
... )
"""


# ============================================================================
# Robust Stability Analysis
# ============================================================================


class RobustStabilityResult(TypedDict, total=False):
    """
    Robust stability analysis result.

    Analyzes stability over parametric uncertainty set.

    Fields
    ------
    robustly_stable : bool
        True if stable for all θ ∈ Θ
    worst_case_eigenvalue : complex
        Eigenvalue closest to instability
    stability_margin : float
        Minimum distance to instability boundary
    critical_parameter : Optional[ParameterVector]
        Worst-case parameter θ*
    method : str
        Analysis method ('polytope', 'Lyapunov', 'gridding')
    conservatism : Optional[float]
        Conservatism estimate (0 = exact, 1 = very conservative)

    Examples
    --------
    >>> # Define uncertain system
    >>> # A(θ) = A_nom + θ₁*ΔA₁ + θ₂*ΔA₂
    >>> A_nom = np.array([[0, 1], [-2, -1]])
    >>> uncertainty = ([-0.2, -0.1], [0.2, 0.1])
    >>>
    >>> result: RobustStabilityResult = analyze_robust_stability(
    ...     A_nom, uncertainty, method='polytope'
    ... )
    >>>
    >>> if result['robustly_stable']:
    ...     print(f"System is robustly stable!")
    ...     print(f"Stability margin: {result['stability_margin']:.3f}")
    ... else:
    ...     print(f"Instability at θ = {result['critical_parameter']}")
    ...     print(f"Critical eigenvalue: {result['worst_case_eigenvalue']}")
    >>>
    >>> # Check conservatism
    >>> if result.get('conservatism', 0) > 0.5:
    ...     print("Analysis is conservative - true margin may be larger")
    """

    robustly_stable: bool
    worst_case_eigenvalue: complex
    stability_margin: float
    critical_parameter: Optional[ParameterVector]
    method: str
    conservatism: Optional[float]


# ============================================================================
# Structured Singular Value (μ-analysis)
# ============================================================================


class StructuredSingularValueResult(TypedDict):
    """
    Structured Singular Value (μ-analysis) result.

    Analyzes robust stability and performance with structured uncertainty.

    Fields
    ------
    mu_value : float
        Structured singular value μ
    robustness_margin : float
        Margin = 1/μ (system stable for ||Δ|| < 1/μ)
    worst_case_uncertainty : ArrayLike
        Critical perturbation Δ*
    frequency_grid : Optional[ArrayLike]
        Frequency points ω where μ computed
    upper_bound : float
        μ upper bound
    lower_bound : float
        μ lower bound

    Examples
    --------
    >>> # μ-analysis for robust stability
    >>> result: StructuredSingularValueResult = mu_analysis(
    ...     M_matrix, delta_structure, frequency_grid
    ... )
    >>>
    >>> mu = result['mu_value']
    >>> margin = result['robustness_margin']
    >>>
    >>> if mu < 1:
    ...     print(f"Robustly stable!")
    ...     print(f"Can tolerate ||Δ|| < {margin:.3f}")
    ... else:
    ...     print(f"Instability possible with ||Δ|| = {1/mu:.3f}")
    ...     Delta_crit = result['worst_case_uncertainty']
    >>>
    >>> # Plot μ vs frequency
    >>> import matplotlib.pyplot as plt
    >>> if 'frequency_grid' in result:
    ...     omega = result['frequency_grid']
    ...     plt.plot(omega, result['mu_value'])
    ...     plt.axhline(1, color='r', linestyle='--')
    ...     plt.xlabel('Frequency (rad/s)')
    ...     plt.ylabel('μ')
    """

    mu_value: float
    robustness_margin: float
    worst_case_uncertainty: ArrayLike
    frequency_grid: Optional[ArrayLike]
    upper_bound: float
    lower_bound: float


# ============================================================================
# Tube-based MPC
# ============================================================================


class TubeDefinition(TypedDict):
    """
    Tube definition for robust MPC.

    Defines robust invariant tube around nominal trajectory.

    Fields
    ------
    shape : str
        Tube shape ('ellipsoid', 'polytope', 'box')
    center_trajectory : StateTrajectory
        Nominal center trajectory (N+1, nx)
    tube_radii : ArrayLike
        Tube size at each time (N+1,)
    shape_matrices : Optional[List[CovarianceMatrix]]
        Shape matrices for ellipsoidal tubes

    Examples
    --------
    >>> # Ellipsoidal tube
    >>> tube: TubeDefinition = {
    ...     'shape': 'ellipsoid',
    ...     'center_trajectory': x_nominal,
    ...     'tube_radii': np.linspace(0.1, 0.5, N+1),
    ...     'shape_matrices': [P_k for k in range(N+1)],
    ... }
    >>>
    >>> # Box tube (simpler)
    >>> tube: TubeDefinition = {
    ...     'shape': 'box',
    ...     'center_trajectory': x_nominal,
    ...     'tube_radii': np.ones(N+1) * 0.2,
    ... }
    """

    shape: str
    center_trajectory: StateTrajectory
    tube_radii: ArrayLike
    shape_matrices: Optional[List[CovarianceMatrix]]


class TubeMPCResult(TypedDict, total=False):
    """
    Tube-based MPC result.

    Robust MPC using disturbance-invariant tubes.

    Fields
    ------
    nominal_control : ControlSequence
        Nominal control sequence v (N, nu)
    feedback_control : GainMatrix
        Ancillary feedback gain K (nu, nx)
    actual_control : ControlSequence
        Applied control u = v + K(x - x̄) (N, nu)
    nominal_trajectory : StateTrajectory
        Nominal state trajectory x̄ (N+1, nx)
    tube_definition : TubeDefinition
        Robust invariant tube
    tightened_constraints : ArrayLike
        Tightened constraints accounting for tube

    Examples
    --------
    >>> # Setup tube MPC
    >>> tube_mpc = TubeMPC(
    ...     system=A, B,
    ...     Q=Q, R=R,
    ...     disturbance_bound=0.1,
    ...     horizon=20
    ... )
    >>>
    >>> # Solve at current state
    >>> x_current = np.array([1.0, 0.5])
    >>> result: TubeMPCResult = tube_mpc.solve(x_current)
    >>>
    >>> # Extract control
    >>> v_nom = result['nominal_control'][0]
    >>> K = result['feedback_control']
    >>> x_nom = result['nominal_trajectory'][0]
    >>>
    >>> # Actual control with feedback
    >>> u = v_nom + K @ (x_current - x_nom)
    >>>
    >>> # Visualize tube
    >>> tube = result['tube_definition']
    >>> center = tube['center_trajectory']
    >>> radii = tube['tube_radii']
    >>>
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(center[:, 0], center[:, 1], 'b-', label='Nominal')
    >>> plt.fill_between(
    ...     center[:, 0] - radii,
    ...     center[:, 0] + radii,
    ...     alpha=0.3, label='Tube'
    ... )
    """

    nominal_control: ControlSequence
    feedback_control: GainMatrix
    actual_control: ControlSequence
    nominal_trajectory: StateTrajectory
    tube_definition: TubeDefinition
    tightened_constraints: ArrayLike


# ============================================================================
# Stochastic MPC
# ============================================================================


class StochasticMPCResult(TypedDict, total=False):
    """
    Stochastic MPC result.

    MPC with chance constraints and distributional robustness.

    Fields
    ------
    control_sequence : ControlSequence
        Optimal control sequence (N, nu)
    predicted_mean : StateTrajectory
        Mean state prediction (N+1, nx)
    predicted_covariance : List[CovarianceMatrix]
        State covariance at each time
    constraint_violation_probability : float
        Maximum constraint violation probability
    expected_cost : float
        Expected cost E[J]
    cost_variance : float
        Cost variance Var[J]
    robust_feasible : bool
        Robustly feasible w.r.t. chance constraints
    chance_constraint_levels : ArrayLike
        Achieved confidence levels (N,)

    Examples
    --------
    >>> # Stochastic MPC with chance constraints
    >>> # P(||x|| ≤ x_max) ≥ 1 - ε = 0.95
    >>> smpc = StochasticMPC(
    ...     system=(A, B),
    ...     Q=Q, R=R,
    ...     process_noise_cov=W,
    ...     chance_constraint_level=0.95,
    ...     horizon=30
    ... )
    >>>
    >>> result: StochasticMPCResult = smpc.solve(x0, Sigma0)
    >>>
    >>> if result['robust_feasible']:
    ...     u = result['control_sequence'][0]
    ...
    ...     # Expected performance
    ...     print(f"Expected cost: {result['expected_cost']:.2f}")
    ...     print(f"Cost std: {np.sqrt(result['cost_variance']):.2f}")
    ...
    ...     # Risk
    ...     risk = result['constraint_violation_probability']
    ...     print(f"Constraint violation risk: {risk*100:.1f}%")
    ...
    ...     # Uncertainty propagation
    ...     mean_traj = result['predicted_mean']
    ...     cov_traj = result['predicted_covariance']
    ...
    ...     # Plot with confidence bounds
    ...     import matplotlib.pyplot as plt
    ...     std_traj = np.array([np.sqrt(np.diag(P)) for P in cov_traj])
    ...     plt.plot(mean_traj[:, 0], label='Mean')
    ...     plt.fill_between(
    ...         range(len(mean_traj)),
    ...         mean_traj[:, 0] - 2*std_traj[:, 0],
    ...         mean_traj[:, 0] + 2*std_traj[:, 0],
    ...         alpha=0.3, label='95% confidence'
    ...     )
    """

    control_sequence: ControlSequence
    predicted_mean: StateTrajectory
    predicted_covariance: List[CovarianceMatrix]
    constraint_violation_probability: float
    expected_cost: float
    cost_variance: float
    robust_feasible: bool
    chance_constraint_levels: ArrayLike


# ============================================================================
# Risk-Sensitive Control
# ============================================================================


class RiskSensitiveResult(TypedDict):
    """
    Risk-sensitive control result.

    Balances expected cost and cost variance.

    Fields
    ------
    gain : GainMatrix
        Risk-sensitive gain K (nu, nx)
    cost_to_go_matrix : CovarianceMatrix
        Risk-sensitive cost-to-go P (nx, nx)
    risk_parameter : float
        Risk aversion parameter θ
    expected_cost : float
        Expected cost E[J]
    cost_variance : float
        Cost variance Var[J]
    certainty_equivalent : float
        Certainty equivalent cost

    Examples
    --------
    >>> # Risk-sensitive LQR
    >>> # Minimize: E[J] + θ·Var[J]
    >>> result: RiskSensitiveResult = design_risk_sensitive_lqr(
    ...     A, B, Q, R,
    ...     process_noise=W,
    ...     risk_parameter=0.1  # θ > 0: risk-averse
    ... )
    >>>
    >>> K = result['gain']
    >>> theta = result['risk_parameter']
    >>>
    >>> print(f"Risk parameter: {theta}")
    >>> print(f"Expected cost: {result['expected_cost']:.2f}")
    >>> print(f"Cost std: {np.sqrt(result['cost_variance']):.2f}")
    >>>
    >>> # Compare with risk-neutral (θ=0)
    >>> lqr_result = design_lqr(A, B, Q, R)
    >>> # Risk-sensitive gain is more conservative
    """

    gain: GainMatrix
    cost_to_go_matrix: CovarianceMatrix
    risk_parameter: float
    expected_cost: float
    cost_variance: float
    certainty_equivalent: float


# ============================================================================
# Export All
# ============================================================================

__all__ = [
    # Uncertainty sets
    "UncertaintySet",
    # Robust analysis
    "RobustStabilityResult",
    "StructuredSingularValueResult",
    # Tube MPC
    "TubeDefinition",
    "TubeMPCResult",
    # Stochastic control
    "StochasticMPCResult",
    "RiskSensitiveResult",
]
