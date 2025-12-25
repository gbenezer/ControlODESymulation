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
Classical Control Theory Types

Result types for classical control theory algorithms:
- Linear Quadratic Regulator (LQR)
- Linear Quadratic Gaussian (LQG)
- Kalman Filter
- Stability, controllability, and observability analysis

These types provide structured return values from control design functions,
enabling type-safe controller and estimator design.

Mathematical Background
----------------------
Classical control theory for linear systems:

LQR (Optimal State Feedback):
    Minimize: J = ∫(x'Qx + u'Ru)dt
    Solution: u = -Kx where K = R⁻¹B'P
    P satisfies: A'P + PA - PBR⁻¹B'P + Q = 0 (Riccati)

Kalman Filter (Optimal State Estimation):
    System: x[k+1] = Ax[k] + Bu[k] + w[k], w ~ N(0,Q)
            y[k] = Cx[k] + v[k], v ~ N(0,R)
    Update: x̂[k+1] = Ax̂[k] + Bu[k] + L(y[k] - Cx̂[k])
    Gain: L = APC'(CPC' + R)⁻¹
    P satisfies: P = A(P - PC'(CPC'+R)⁻¹CP)A' + Q

LQG (LQR + Kalman):
    Separation principle: Design independently
    Controller: u = -K(x̂ - x_ref)
    Estimator: x̂[k+1] = Ax̂[k] + Bu[k] + L(y[k] - Cx̂[k])

Usage
-----
>>> from src.types.control_classical import (
...     LQRResult,
...     KalmanFilterResult,
...     StabilityInfo,
... )
>>> 
>>> # LQR design
>>> result: LQRResult = design_lqr(A, B, Q, R)
>>> K = result['gain']
>>> is_stable = np.all(np.real(result['closed_loop_eigenvalues']) < 0)
>>> 
>>> # Kalman filter design
>>> kalman: KalmanFilterResult = design_kalman(A, C, Q_process, R_meas)
>>> L = kalman['gain']
>>> 
>>> # Check stability
>>> stability: StabilityInfo = check_stability(A)
>>> if stability['is_stable']:
...     print(f"Stable with margin {stability['stability_margin']:.3f}")
"""

from typing import Optional
from typing_extensions import TypedDict
import numpy as np

from .core import (
    GainMatrix,
    CovarianceMatrix,
    ControllabilityMatrix,
    ObservabilityMatrix,
)


# ============================================================================
# Stability Analysis Types
# ============================================================================

class StabilityInfo(TypedDict):
    """
    Stability analysis result dictionary.
    
    Contains eigenvalue-based stability information for linear systems.
    
    Stability Criteria:
    - Continuous: All Re(λ) < 0 (left half-plane)
    - Discrete: All |λ| < 1 (inside unit circle)
    
    Fields
    ------
    eigenvalues : np.ndarray
        Eigenvalues of system matrix (complex)
    magnitudes : np.ndarray
        Absolute values |λ| of eigenvalues
    max_magnitude : float
        Maximum |λ| (spectral radius)
    spectral_radius : float
        Same as max_magnitude (discrete systems)
    is_stable : bool
        True if system is asymptotically stable
    is_marginally_stable : bool
        True if max|λ| ≈ 1 or Re(λ) ≈ 0
    is_unstable : bool
        True if any |λ| > 1 or Re(λ) > 0
    
    Examples
    --------
    >>> # Continuous system
    >>> A = np.array([[0, 1], [-2, -3]])
    >>> stability: StabilityInfo = analyze_stability(A, system_type='continuous')
    >>> print(stability['is_stable'])  # True
    >>> print(stability['eigenvalues'])  # [-1, -2]
    >>> 
    >>> # Discrete system
    >>> Ad = np.array([[0.9, 0.1], [0, 0.8]])
    >>> stability: StabilityInfo = analyze_stability(Ad, system_type='discrete')
    >>> print(stability['is_stable'])  # True (both |λ| < 1)
    >>> print(stability['spectral_radius'])  # 0.9
    >>> 
    >>> # Unstable system
    >>> A_unstable = np.array([[1, 1], [0, 1]])
    >>> stability: StabilityInfo = analyze_stability(A_unstable, system_type='continuous')
    >>> print(stability['is_unstable'])  # True
    """
    eigenvalues: np.ndarray
    magnitudes: np.ndarray
    max_magnitude: float
    spectral_radius: float
    is_stable: bool
    is_marginally_stable: bool
    is_unstable: bool


class ControllabilityInfo(TypedDict, total=False):
    """
    Controllability analysis result.
    
    A system (A, B) is controllable if all states can be driven to any
    desired value in finite time using appropriate control inputs.
    
    Controllability Test:
    - Rank of controllability matrix C = [B AB A²B ... Aⁿ⁻¹B] equals nx
    
    Fields
    ------
    controllability_matrix : ControllabilityMatrix
        C = [B, AB, A²B, ..., Aⁿ⁻¹B] of shape (nx, nx*nu)
    rank : int
        Rank of controllability matrix
    is_controllable : bool
        True if rank == nx (full rank)
    uncontrollable_modes : Optional[np.ndarray]
        Eigenvalues of uncontrollable subsystem (if any)
    
    Examples
    --------
    >>> A = np.array([[0, 1], [-2, -3]])
    >>> B = np.array([[0], [1]])
    >>> 
    >>> info: ControllabilityInfo = analyze_controllability(A, B)
    >>> print(info['is_controllable'])  # True
    >>> print(info['rank'])  # 2
    >>> print(info['controllability_matrix'].shape)  # (2, 2)
    >>> 
    >>> # Uncontrollable system
    >>> B_bad = np.array([[1], [1]])  # Both states affected equally
    >>> A_diag = np.array([[1, 0], [0, 2]])
    >>> info: ControllabilityInfo = analyze_controllability(A_diag, B_bad)
    >>> print(info['is_controllable'])  # False
    >>> print(info['uncontrollable_modes'])  # Some eigenvalues
    """
    controllability_matrix: ControllabilityMatrix
    rank: int
    is_controllable: bool
    uncontrollable_modes: Optional[np.ndarray]


class ObservabilityInfo(TypedDict, total=False):
    """
    Observability analysis result.
    
    A system (A, C) is observable if the initial state can be determined
    from output measurements over a finite time interval.
    
    Observability Test:
    - Rank of observability matrix O = [C; CA; CA²; ...; CAⁿ⁻¹] equals nx
    
    Fields
    ------
    observability_matrix : ObservabilityMatrix
        O = [C; CA; CA²; ...; CAⁿ⁻¹] of shape (nx*ny, nx)
    rank : int
        Rank of observability matrix
    is_observable : bool
        True if rank == nx (full rank)
    unobservable_modes : Optional[np.ndarray]
        Eigenvalues of unobservable subsystem (if any)
    
    Examples
    --------
    >>> A = np.array([[0, 1], [-2, -3]])
    >>> C = np.array([[1, 0]])  # Measure position only
    >>> 
    >>> info: ObservabilityInfo = analyze_observability(A, C)
    >>> print(info['is_observable'])  # True
    >>> print(info['rank'])  # 2
    >>> print(info['observability_matrix'].shape)  # (2, 2)
    >>> 
    >>> # Unobservable system
    >>> C_bad = np.array([[1, 1]])  # Can't distinguish states
    >>> A_diag = np.array([[1, 0], [0, 2]])
    >>> info: ObservabilityInfo = analyze_observability(A_diag, C_bad)
    >>> print(info['is_observable'])  # False
    """
    observability_matrix: ObservabilityMatrix
    rank: int
    is_observable: bool
    unobservable_modes: Optional[np.ndarray]


# ============================================================================
# Classical Control Design Result Types
# ============================================================================

class LQRResult(TypedDict):
    """
    Linear Quadratic Regulator (LQR) design result.
    
    LQR computes optimal state feedback gain K that minimizes:
        J = ∫₀^∞ (x'Qx + u'Ru) dt  (continuous)
        J = Σₖ₌₀^∞ (x'Qx + u'Ru)     (discrete)
    
    Optimal control law: u = -Kx
    
    Fields
    ------
    gain : GainMatrix
        Optimal feedback gain K of shape (nu, nx)
    cost_to_go : CovarianceMatrix
        Solution P to algebraic Riccati equation (nx, nx)
    closed_loop_eigenvalues : np.ndarray
        Eigenvalues of (A - BK) - indicates stability and response
    stability_margin : float
        Distance from stability boundary (positive = stable)
    
    Examples
    --------
    >>> # Continuous LQR
    >>> A = np.array([[0, 1], [-2, -3]])
    >>> B = np.array([[0], [1]])
    >>> Q = np.diag([10, 1])  # Penalize position more than velocity
    >>> R = np.array([[0.1]])  # Control cost
    >>> 
    >>> result: LQRResult = design_lqr_continuous(A, B, Q, R)
    >>> K = result['gain']
    >>> print(K.shape)  # (1, 2)
    >>> 
    >>> # Apply control
    >>> x = np.array([1.0, 0.0])
    >>> u = -K @ x
    >>> 
    >>> # Check stability
    >>> print(np.all(np.real(result['closed_loop_eigenvalues']) < 0))  # True
    >>> print(result['stability_margin'])  # Positive value
    >>> 
    >>> # Discrete LQR
    >>> Ad = np.array([[1, 0.1], [0, 0.9]])
    >>> Bd = np.array([[0], [0.1]])
    >>> result_d: LQRResult = design_lqr_discrete(Ad, Bd, Q, R)
    >>> print(np.all(np.abs(result_d['closed_loop_eigenvalues']) < 1))  # True
    """
    gain: GainMatrix
    cost_to_go: CovarianceMatrix
    closed_loop_eigenvalues: np.ndarray
    stability_margin: float


class KalmanFilterResult(TypedDict):
    """
    Kalman Filter (optimal state estimator) design result.
    
    Kalman filter provides optimal state estimate for linear system:
        x[k+1] = Ax[k] + Bu[k] + w[k],  w ~ N(0, Q)
        y[k] = Cx[k] + v[k],            v ~ N(0, R)
    
    Estimator dynamics: x̂[k+1] = Ax̂[k] + Bu[k] + L(y[k] - Cx̂[k])
    
    Fields
    ------
    gain : GainMatrix
        Kalman gain L of shape (nx, ny)
    error_covariance : CovarianceMatrix
        Steady-state error covariance P (nx, nx)
    innovation_covariance : CovarianceMatrix
        Innovation covariance S = CPC' + R (ny, ny)
    observer_eigenvalues : np.ndarray
        Eigenvalues of (A - LC) - determines convergence rate
    
    Examples
    --------
    >>> # Design Kalman filter
    >>> A = np.array([[1, 0.1], [0, 0.9]])
    >>> C = np.array([[1, 0]])  # Measure position only
    >>> Q_process = 0.01 * np.eye(2)  # Process noise covariance
    >>> R_meas = 0.1 * np.eye(1)      # Measurement noise covariance
    >>> 
    >>> result: KalmanFilterResult = design_kalman_filter(A, C, Q_process, R_meas)
    >>> L = result['gain']
    >>> print(L.shape)  # (2, 1)
    >>> 
    >>> # State estimation loop
    >>> x_hat = np.zeros(2)
    >>> for k in range(N):
    ...     # Prediction
    ...     x_hat_pred = A @ x_hat + B @ u[k]
    ...     
    ...     # Update (correction)
    ...     innovation = y[k] - C @ x_hat_pred
    ...     x_hat = x_hat_pred + L @ innovation
    >>> 
    >>> # Check observer stability
    >>> print(np.all(np.abs(result['observer_eigenvalues']) < 1))  # True
    >>> 
    >>> # Innovation statistics
    >>> S = result['innovation_covariance']
    >>> print(S.shape)  # (1, 1)
    """
    gain: GainMatrix
    error_covariance: CovarianceMatrix
    innovation_covariance: CovarianceMatrix
    observer_eigenvalues: np.ndarray


class LQGResult(TypedDict):
    """
    Linear Quadratic Gaussian (LQG) controller design result.
    
    LQG combines optimal control (LQR) with optimal estimation (Kalman):
    - LQR: Optimal state feedback u = -Kx (if state known)
    - Kalman: Optimal state estimate x̂ from measurements
    - LQG: Certainty equivalence u = -Kx̂
    
    Separation Principle:
    - Design LQR and Kalman independently
    - Combine for optimal performance under Gaussian noise
    
    Fields
    ------
    control_gain : GainMatrix
        LQR feedback gain K (nu, nx)
    estimator_gain : GainMatrix
        Kalman gain L (nx, ny)
    control_cost_to_go : CovarianceMatrix
        Controller Riccati solution P_control
    estimation_error_covariance : CovarianceMatrix
        Estimator Riccati solution P_estimate
    separation_verified : bool
        Confirms separation principle holds
    closed_loop_stable : bool
        Combined controller-estimator system is stable
    controller_eigenvalues : np.ndarray
        Eigenvalues of (A - BK) - control loop
    estimator_eigenvalues : np.ndarray
        Eigenvalues of (A - LC) - estimation loop
    
    Examples
    --------
    >>> # System matrices
    >>> A = np.array([[1, 0.1], [0, 0.9]])
    >>> B = np.array([[0], [0.1]])
    >>> C = np.array([[1, 0]])  # Measure position only
    >>> 
    >>> # Design weights
    >>> Q_control = np.diag([10, 1])   # State cost
    >>> R_control = np.array([[0.1]])  # Control cost
    >>> Q_process = 0.01 * np.eye(2)   # Process noise
    >>> R_meas = 0.1 * np.eye(1)       # Measurement noise
    >>> 
    >>> result: LQGResult = design_lqg(A, B, C, Q_control, R_control, Q_process, R_meas)
    >>> 
    >>> K = result['control_gain']
    >>> L = result['estimator_gain']
    >>> print(result['closed_loop_stable'])  # True
    >>> print(result['separation_verified'])  # True
    >>> 
    >>> # Implement LQG controller
    >>> x_hat = np.zeros(2)  # Initial estimate
    >>> for k in range(N):
    ...     # Control (certainty equivalence)
    ...     u[k] = -K @ x_hat
    ...     
    ...     # Prediction
    ...     x_hat = A @ x_hat + B @ u[k]
    ...     
    ...     # Measurement update
    ...     innovation = y[k] - C @ x_hat
    ...     x_hat = x_hat + L @ innovation
    >>> 
    >>> # Check eigenvalues
    >>> print("Controller poles:", result['controller_eigenvalues'])
    >>> print("Estimator poles:", result['estimator_eigenvalues'])
    """
    control_gain: GainMatrix
    estimator_gain: GainMatrix
    control_cost_to_go: CovarianceMatrix
    estimation_error_covariance: CovarianceMatrix
    separation_verified: bool
    closed_loop_stable: bool
    controller_eigenvalues: np.ndarray
    estimator_eigenvalues: np.ndarray


# ============================================================================
# Additional Classical Control Types
# ============================================================================

class PolePlacementResult(TypedDict):
    """
    Pole placement (eigenvalue assignment) result.
    
    Design state feedback gain K such that closed-loop system
    (A - BK) has desired eigenvalues (poles).
    
    Fields
    ------
    gain : GainMatrix
        State feedback gain K (nu, nx)
    desired_poles : np.ndarray
        Desired closed-loop eigenvalues
    achieved_poles : np.ndarray
        Actual achieved eigenvalues of (A - BK)
    is_controllable : bool
        System must be controllable for arbitrary placement
    
    Examples
    --------
    >>> A = np.array([[0, 1], [-2, -3]])
    >>> B = np.array([[0], [1]])
    >>> 
    >>> # Place poles for faster response
    >>> desired_poles = np.array([-5, -6])
    >>> result: PolePlacementResult = pole_placement(A, B, desired_poles)
    >>> 
    >>> K = result['gain']
    >>> print(result['is_controllable'])  # True
    >>> print(np.allclose(result['achieved_poles'], result['desired_poles']))  # True
    >>> 
    >>> # Verify
    >>> A_cl = A - B @ K
    >>> actual_poles = np.linalg.eigvals(A_cl)
    >>> print(np.sort(actual_poles))  # [-6, -5]
    """
    gain: GainMatrix
    desired_poles: np.ndarray
    achieved_poles: np.ndarray
    is_controllable: bool


class LuenbergerObserverResult(TypedDict):
    """
    Luenberger observer (deterministic state estimator) design result.
    
    Observer dynamics: x̂˙ = Ax̂ + Bu + L(y - Cx̂)
    Error dynamics: e˙ = (A - LC)e
    
    Choose L to place observer poles (eigenvalues of A - LC).
    
    Fields
    ------
    gain : GainMatrix
        Observer gain L (nx, ny)
    desired_poles : np.ndarray
        Desired observer eigenvalues
    achieved_poles : np.ndarray
        Actual eigenvalues of (A - LC)
    is_observable : bool
        System must be observable for arbitrary placement
    
    Examples
    --------
    >>> A = np.array([[0, 1], [-2, -3]])
    >>> C = np.array([[1, 0]])  # Measure position
    >>> 
    >>> # Place observer poles faster than controller
    >>> desired_poles = np.array([-10, -12])
    >>> result: LuenbergerObserverResult = design_observer(A, C, desired_poles)
    >>> 
    >>> L = result['gain']
    >>> print(result['is_observable'])  # True
    >>> 
    >>> # Observer-based control
    >>> x_hat = np.zeros(2)
    >>> for k in range(N):
    ...     u = -K @ x_hat  # Control law
    ...     y_meas = C @ x_true + noise
    ...     
    ...     # Observer update
    ...     x_hat_dot = A @ x_hat + B @ u + L @ (y_meas - C @ x_hat)
    ...     x_hat = x_hat + dt * x_hat_dot  # Euler integration
    """
    gain: GainMatrix
    desired_poles: np.ndarray
    achieved_poles: np.ndarray
    is_observable: bool


# ============================================================================
# Export All
# ============================================================================

__all__ = [
    # Analysis types
    'StabilityInfo',
    'ControllabilityInfo',
    'ObservabilityInfo',
    
    # Control design
    'LQRResult',
    'KalmanFilterResult',
    'LQGResult',
    'PolePlacementResult',
    'LuenbergerObserverResult',
]