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
Advanced Control Theory Types

Result types for advanced control algorithms:
- Model Predictive Control (MPC)
- Moving Horizon Estimation (MHE)
- H₂ and H∞ Optimal Control
- Linear Matrix Inequalities (LMI)
- Adaptive Control
- Sliding Mode Control

These types extend classical control with modern optimal, robust,
and adaptive techniques suitable for constrained and uncertain systems.

Mathematical Background
----------------------
Model Predictive Control (MPC):
    Solve at each time k:
        min  Σᵢ₌₀ᴺ (‖xᵢ-xᵣₑf‖²Q + ‖uᵢ‖²R) + ‖xN-xᵣₑf‖²P
        s.t. xᵢ₊₁ = Axᵢ + Buᵢ
             uₘᵢₙ ≤ uᵢ ≤ uₘₐₓ
             xₘᵢₙ ≤ xᵢ ≤ xₘₐₓ
    Apply u₀*, repeat at k+1 (receding horizon)

H₂ Optimal Control:
    Minimize H₂ norm: ‖G‖₂² = (1/2π) ∫ tr(G(jω)*G(jω)) dω
    RMS response to white noise disturbances
    Solved via algebraic Riccati equation

H∞ Robust Control:
    Minimize H∞ norm: ‖G‖∞ = supω σ̄(G(jω))
    Worst-case gain from disturbance to output
    Solved via γ-iteration and Riccati equations
    Provides guaranteed robustness to uncertainties

Sliding Mode Control (SMC):
    Sliding surface: s(x) = Cx
    Control law: u = -K sgn(s) (discontinuous)
    Properties: Finite-time convergence, robustness
    Challenges: Chattering (high-frequency switching)

Adaptive Control:
    Parameter estimation: θ̂˙ = -Γ e ϕ (gradient adaptation)
    Certainty equivalence: Use θ̂ as if true
    Properties: Handles unknown parameters
    Stability: Lyapunov-based design ensures convergence

Usage
-----
>>> from src.types.control_advanced import (
...     MPCResult,
...     H2ControlResult,
...     HInfControlResult,
... )
>>>
>>> # MPC optimization
>>> result: MPCResult = mpc.solve(x0, reference, horizon=20)
>>> u_optimal = result['control_sequence'][0]  # Apply first control
>>>
>>> # H∞ design
>>> result: HInfControlResult = design_hinf(system, gamma=2.0)
>>> K = result['gain']
>>> print(f"Achieved γ: {result['hinf_norm']:.3f}")
"""

from typing import Dict, Optional

import numpy as np
from typing_extensions import TypedDict

from .core import (
    ArrayLike,
    ControlVector,
    CovarianceMatrix,
    GainMatrix,
    ParameterVector,
    StateVector,
)
from .trajectories import (
    ControlSequence,
    OutputSequence,
    StateTrajectory,
)

# ============================================================================
# Model Predictive Control
# ============================================================================


class MPCResult(TypedDict, total=False):
    """
    Model Predictive Control (MPC) solution result.

    MPC solves a finite-horizon optimal control problem at each time step,
    applying only the first control in the sequence (receding horizon).

    Fields
    ------
    control_sequence : ControlSequence
        Optimal control trajectory u*[0:N-1] (N, nu)
    predicted_trajectory : StateTrajectory
        Predicted state trajectory x*[0:N] (N+1, nx)
    cost : float
        Optimal objective value J*
    success : bool
        Whether optimization converged successfully
    iterations : int
        Number of optimization iterations
    solve_time : float
        Computation time in seconds
    constraint_violations : Optional[ArrayLike]
        Slack variable values (if soft constraints used)
    dual_variables : Optional[ArrayLike]
        Lagrange multipliers (sensitivity to constraints)

    Examples
    --------
    >>> # Setup MPC
    >>> mpc = ModelPredictiveController(
    ...     system, horizon=20, Q=np.diag([10, 1]), R=np.array([[0.1]])
    ... )
    >>>
    >>> # Solve at current state
    >>> x_current = np.array([1.0, 0.5])
    >>> x_ref = np.zeros(2)
    >>> result: MPCResult = mpc.solve(x_current, x_ref)
    >>>
    >>> # Apply first control (receding horizon)
    >>> u_apply = result['control_sequence'][0]
    >>>
    >>> # Check solution quality
    >>> if result['success']:
    ...     print(f"Cost: {result['cost']:.3f}")
    ...     print(f"Solve time: {result['solve_time']*1000:.1f} ms")
    >>>
    >>> # Examine predicted trajectory
    >>> x_pred = result['predicted_trajectory']
    >>> print(x_pred.shape)  # (21, 2) for horizon=20
    """

    control_sequence: ControlSequence
    predicted_trajectory: StateTrajectory
    cost: float
    success: bool
    iterations: int
    solve_time: float
    constraint_violations: Optional[ArrayLike]
    dual_variables: Optional[ArrayLike]


class MHEResult(TypedDict, total=False):
    """
    Moving Horizon Estimation (MHE) result.

    MHE is the dual of MPC - optimal state estimation over a
    receding horizon using past measurements and control inputs.

    Fields
    ------
    state_estimate : StateVector
        Current optimal state estimate x̂[k] (nx,)
    covariance_estimate : CovarianceMatrix
        Estimated covariance P̂[k] (nx, nx)
    state_trajectory : StateTrajectory
        Estimated trajectory over horizon (N, nx)
    cost : float
        Estimation objective value
    success : bool
        Whether optimization succeeded
    solve_time : float
        Computation time in seconds
    innovation_sequence : OutputSequence
        Measurement residuals y - Cx̂ (N, ny)

    Examples
    --------
    >>> # Setup MHE
    >>> mhe = MovingHorizonEstimator(
    ...     system, horizon=10, Q_process=0.01*np.eye(2), R_meas=0.1*np.eye(1)
    ... )
    >>>
    >>> # Update with new measurement
    >>> y_meas = np.array([1.2])
    >>> u_applied = np.array([0.5])
    >>> result: MHEResult = mhe.update(y_meas, u_applied)
    >>>
    >>> # Use state estimate
    >>> x_hat = result['state_estimate']
    >>> P_hat = result['covariance_estimate']
    >>>
    >>> # Check innovation
    >>> innovations = result['innovation_sequence']
    >>> innovation_norm = np.linalg.norm(innovations[-1])
    """

    state_estimate: StateVector
    covariance_estimate: CovarianceMatrix
    state_trajectory: StateTrajectory
    cost: float
    success: bool
    solve_time: float
    innovation_sequence: OutputSequence


# ============================================================================
# Optimal and Robust Control
# ============================================================================


class H2ControlResult(TypedDict):
    """
    H₂ optimal control result.

    H₂ control minimizes the RMS (root-mean-square) response to
    white noise disturbances, equivalent to LQG for certain problem setups.

    Fields
    ------
    gain : GainMatrix
        H₂ optimal controller gain K (nu, nx)
    h2_norm : float
        Achieved H₂ norm ‖G‖₂
    cost_to_go : CovarianceMatrix
        Riccati solution P (nx, nx)
    closed_loop_stable : bool
        Closed-loop system is stable
    closed_loop_poles : np.ndarray
        Eigenvalues of (A - BK)

    Examples
    --------
    >>> # Design H₂ controller
    >>> A = np.array([[0, 1], [-2, -3]])
    >>> B = np.array([[0], [1]])
    >>> C_z = np.eye(2)  # Performance output
    >>> D_zu = np.zeros((2, 1))
    >>>
    >>> result: H2ControlResult = design_h2_controller(A, B, C_z, D_zu)
    >>>
    >>> K = result['gain']
    >>> print(f"H₂ norm: {result['h2_norm']:.3f}")
    >>> print(f"Stable: {result['closed_loop_stable']}")
    >>>
    >>> # Apply control
    >>> u = -K @ x
    """

    gain: GainMatrix
    h2_norm: float
    cost_to_go: CovarianceMatrix
    closed_loop_stable: bool
    closed_loop_poles: np.ndarray


class HInfControlResult(TypedDict):
    """
    H∞ robust control result.

    H∞ control minimizes the worst-case gain from disturbances to
    performance outputs, providing guaranteed robustness.

    Fields
    ------
    gain : GainMatrix
        H∞ controller gain K (nu, nx)
    hinf_norm : float
        Achieved H∞ norm ‖G‖∞
    gamma : float
        Performance bound γ (hinf_norm ≤ γ)
    central_solution : CovarianceMatrix
        Central Riccati solution (nx, nx)
    feasible : bool
        Whether γ was achievable
    robustness_margin : float
        Stability margin (how much uncertainty tolerated)

    Examples
    --------
    >>> # Design H∞ controller with γ = 2.0
    >>> A = np.array([[0, 1], [-2, -3]])
    >>> B = np.array([[0], [1]])
    >>> C_z = np.eye(2)
    >>> D_zu = np.zeros((2, 1))
    >>>
    >>> result: HInfControlResult = design_hinf_controller(
    ...     A, B, C_z, D_zu, gamma=2.0
    ... )
    >>>
    >>> if result['feasible']:
    ...     K = result['gain']
    ...     print(f"Achieved γ: {result['hinf_norm']:.3f}")
    ...     print(f"Robustness margin: {result['robustness_margin']:.3f}")
    ... else:
    ...     print("γ = 2.0 not achievable, try larger γ")
    >>>
    >>> # Verify worst-case performance
    >>> # ‖G_cl‖∞ ≤ γ guarantees robustness to uncertainties
    """

    gain: GainMatrix
    hinf_norm: float
    gamma: float
    central_solution: CovarianceMatrix
    feasible: bool
    robustness_margin: float


class LMIResult(TypedDict, total=False):
    """
    Linear Matrix Inequality (LMI) solver result.

    LMIs enable convex formulation of many control problems:
    - Lyapunov stability (P > 0, A'P + PA < 0)
    - H∞ synthesis, polytopic systems, etc.

    Fields
    ------
    decision_variables : Dict[str, ArrayLike]
        Solved matrix variables (P, Y, etc.)
    objective_value : float
        Optimal objective function value
    feasible : bool
        Whether LMI constraints are feasible
    solver : str
        Solver used ('cvxpy', 'mosek', 'sedumi', etc.)
    solve_time : float
        Computation time in seconds
    condition_number : float
        Condition number of solution (numerical health)

    Examples
    --------
    >>> # Lyapunov stability LMI: find P > 0 s.t. A'P + PA < 0
    >>> import cvxpy as cp
    >>> P = cp.Variable((2, 2), symmetric=True)
    >>> constraints = [
    ...     P >> 0,  # P positive definite
    ...     A.T @ P + P @ A << 0  # Lyapunov inequality
    ... ]
    >>> problem = cp.Problem(cp.Minimize(0), constraints)
    >>> problem.solve()
    >>>
    >>> result: LMIResult = {
    ...     'decision_variables': {'P': P.value},
    ...     'objective_value': problem.value,
    ...     'feasible': problem.status == 'optimal',
    ...     'solver': 'cvxpy',
    ...     'solve_time': problem.solver_stats.solve_time,
    ...     'condition_number': np.linalg.cond(P.value),
    ... }
    >>>
    >>> if result['feasible']:
    ...     P_lyap = result['decision_variables']['P']
    ...     print("System is stable")
    """

    decision_variables: Dict[str, ArrayLike]
    objective_value: float
    feasible: bool
    solver: str
    solve_time: float
    condition_number: float


# ============================================================================
# Adaptive and Robust Nonlinear Control
# ============================================================================


class AdaptiveControlResult(TypedDict, total=False):
    """
    Adaptive control result.

    Adaptive controllers adjust gains online to handle unknown or
    time-varying parameters.

    Fields
    ------
    current_gain : GainMatrix
        Current adapted controller gain K(t) (nu, nx)
    parameter_estimate : ParameterVector
        Current parameter estimate θ̂(t) (nθ,)
    parameter_covariance : CovarianceMatrix
        Parameter uncertainty P_θ(t) (nθ, nθ)
    adaptation_rate : float
        Current learning rate Γ
    tracking_error : float
        Output tracking error ‖y - y_ref‖
    parameter_error : Optional[ParameterVector]
        True error θ̂ - θ (if θ known, for testing)

    Examples
    --------
    >>> # Model Reference Adaptive Control (MRAC)
    >>> adaptive_ctrl = AdaptiveController(
    ...     reference_model, adaptation_rate=0.1
    ... )
    >>>
    >>> # Update at each time step
    >>> result: AdaptiveControlResult = adaptive_ctrl.update(
    ...     x_current, y_measured, y_reference
    ... )
    >>>
    >>> # Apply adapted control
    >>> K = result['current_gain']
    >>> u = -K @ x_current
    >>>
    >>> # Monitor adaptation
    >>> theta_hat = result['parameter_estimate']
    >>> tracking_err = result['tracking_error']
    >>> print(f"Tracking error: {tracking_err:.3f}")
    >>> print(f"Parameter estimate: {theta_hat}")
    >>>
    >>> # Check convergence (if true parameters known)
    >>> if result['parameter_error'] is not None:
    ...     param_err_norm = np.linalg.norm(result['parameter_error'])
    ...     print(f"Parameter error: {param_err_norm:.3f}")
    """

    current_gain: GainMatrix
    parameter_estimate: ParameterVector
    parameter_covariance: CovarianceMatrix
    adaptation_rate: float
    tracking_error: float
    parameter_error: Optional[ParameterVector]


class SlidingModeResult(TypedDict):
    """
    Sliding Mode Control (SMC) result.

    SMC uses discontinuous control to drive system to a sliding
    surface in finite time, providing robustness to uncertainties.

    Fields
    ------
    control : ControlVector
        SMC control signal u (nu,)
    sliding_variable : ArrayLike
        Sliding surface variable s = Cx (ns,)
    on_sliding_surface : bool
        Whether |s| < ε (in sliding mode)
    reaching_time_estimate : Optional[float]
        Estimated time to reach surface (if not on surface)
    chattering_magnitude : float
        Control chattering level (high-frequency switching)

    Examples
    --------
    >>> # Sliding mode controller
    >>> smc = SlidingModeController(
    ...     sliding_surface_gain=np.array([[1, 1]]),
    ...     switching_gain=5.0,
    ...     boundary_layer=0.1
    ... )
    >>>
    >>> # Compute control
    >>> x = np.array([1.0, 0.5])
    >>> x_desired = np.zeros(2)
    >>> result: SlidingModeResult = smc.compute_control(x, x_desired)
    >>>
    >>> # Apply control
    >>> u = result['control']
    >>>
    >>> # Check sliding mode
    >>> s = result['sliding_variable']
    >>> if result['on_sliding_surface']:
    ...     print("In sliding mode - tracking achieved")
    ... else:
    ...     t_reach = result['reaching_time_estimate']
    ...     print(f"Reaching surface in ~{t_reach:.2f} seconds")
    >>>
    >>> # Monitor chattering
    >>> chattering = result['chattering_magnitude']
    >>> if chattering > 1.0:
    ...     print("High chattering - consider boundary layer")
    """

    control: ControlVector
    sliding_variable: ArrayLike
    on_sliding_surface: bool
    reaching_time_estimate: Optional[float]
    chattering_magnitude: float


# ============================================================================
# Export All
# ============================================================================

__all__ = [
    # Predictive control
    "MPCResult",
    "MHEResult",
    # Optimal/robust control
    "H2ControlResult",
    "HInfControlResult",
    "LMIResult",
    # Adaptive/nonlinear control
    "AdaptiveControlResult",
    "SlidingModeResult",
]
