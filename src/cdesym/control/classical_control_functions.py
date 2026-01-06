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
Classical Control Theory Functions

Pure stateless functions for classical control design and analysis:

**Control Design:**
- Linear Quadratic Regulator (LQR) - continuous and discrete
- Kalman Filter - optimal state estimation
- Linear Quadratic Gaussian (LQG) - combined LQR + Kalman

**System Analysis:**
- Stability analysis - eigenvalue-based
- Controllability - rank test
- Observability - rank test

All functions are pure (no side effects, no state) and work like scipy.
Backend conversion is handled internally.

Mathematical Background
-----------------------
LQR minimizes:
    J = ∫₀^∞ (x'Qx + u'Ru) dt  (continuous)
    J = Σₖ₌₀^∞ (x'Qx + u'Ru)     (discrete)

Solution via algebraic Riccati equation (ARE):
    Continuous: A'P + PA - PBR⁻¹B'P + Q = 0
    Discrete:   P = A'PA - A'PB(R + B'PB)⁻¹B'PA + Q

Optimal gain: K = R⁻¹B'P (continuous), K = (R + B'PB)⁻¹B'PA (discrete)

Kalman Filter for:
    x[k+1] = Ax[k] + Bu[k] + w[k],  w ~ N(0,Q)
    y[k] = Cx[k] + v[k],            v ~ N(0,R)

Estimator: x̂[k+1] = Ax̂[k] + Bu[k] + L(y[k] - Cx̂[k])

Stability:
    Continuous: All Re(λ) < 0 (left half-plane)
    Discrete:   All |λ| < 1 (inside unit circle)

Controllability: rank([B AB A²B ... Aⁿ⁻¹B]) = n
Observability:   rank([C; CA; CA²; ...; CAⁿ⁻¹]) = n

Usage
-----
>>> from controldesymulation.control.classical import design_lqr_continuous, analyze_stability
>>> import numpy as np
>>>
>>> # LQR design
>>> A = np.array([[0, 1], [-2, -3]])
>>> B = np.array([[0], [1]])
>>> Q = np.diag([10, 1])
>>> R = np.array([[0.1]])
>>>
>>> result = design_lqr_continuous(A, B, Q, R)
>>> K = result['gain']
>>> print(f"Optimal gain: {K}")
>>> print(f"Stability margin: {result['stability_margin']:.3f}")
>>>
>>> # Stability analysis
>>> stability = analyze_stability(A, system_type='continuous')
>>> print(f"Stable: {stability['is_stable']}")
"""

from typing import Optional

import numpy as np
from scipy import linalg

from cdesym.types.backends import Backend
from cdesym.types.control_classical import (
    ControllabilityInfo,
    KalmanFilterResult,
    LQGResult,
    LQRResult,
    ObservabilityInfo,
    StabilityInfo,
)
from cdesym.types.core import (
    InputMatrix,
    OutputMatrix,
    StateMatrix,
)

# ============================================================================
# Backend Conversion Utilities (Internal)
# ============================================================================


def _to_numpy(arr, backend: Backend):
    """
    Convert array to NumPy for scipy operations.

    Args:
        arr: Array in any backend
        backend: Source backend identifier

    Returns:
        NumPy array
    """
    if isinstance(arr, np.ndarray):
        return arr

    if backend == "torch" or hasattr(arr, "cpu"):
        # PyTorch tensor
        return arr.detach().cpu().numpy()
    if backend == "jax" or hasattr(arr, "__array__"):
        # JAX array
        return np.array(arr)
    # Try generic conversion
    return np.asarray(arr)


def _from_numpy(arr: np.ndarray, backend: Backend):
    """
    Convert NumPy array back to target backend.

    Args:
        arr: NumPy array
        backend: Target backend

    Returns:
        Array in target backend
    """
    if backend == "numpy":
        return arr
    if backend == "torch":
        import torch

        return torch.from_numpy(arr)
    if backend == "jax":
        import jax.numpy as jnp

        return jnp.array(arr)
    return arr


# ============================================================================
# LQR - Linear Quadratic Regulator
# ============================================================================


def design_lqr(
    A: StateMatrix,
    B: InputMatrix,
    Q: StateMatrix,
    R: InputMatrix,
    N: Optional[InputMatrix] = None,
    system_type: str = "discrete",
    backend: Backend = "numpy",
) -> LQRResult:
    """
    Design Linear Quadratic Regulator (LQR) controller.

    Minimizes cost functional:
        Continuous: J = ∫₀^∞ (x'Qx + u'Ru + 2x'Nu) dt
        Discrete:   J = Σₖ₌₀^∞ (x[k]'Qx[k] + u[k]'Ru[k] + 2x[k]'Nu[k])

    Solves algebraic Riccati equation (ARE):
        Continuous (CARE): A'P + PA - (PB + N)R⁻¹(B'P + N') + Q = 0
        Discrete (DARE):   P = A'PA - (A'PB + N)(R + B'PB)⁻¹(B'PA + N') + Q

    Optimal control law:
        Continuous: u = -Kx where K = R⁻¹(B'P + N')
        Discrete:   u[k] = -Kx[k] where K = (R + B'PB)⁻¹(B'PA + N')

    Parameters
    ----------
    A : StateMatrix
        State matrix (nx, nx)
    B : InputMatrix
        Input matrix (nx, nu)
    Q : StateMatrix
        State cost matrix (nx, nx), must be positive semi-definite (Q ≥ 0)
    R : InputMatrix
        Control cost matrix (nu, nu), must be positive definite (R > 0)
    N : Optional[InputMatrix]
        Cross-coupling matrix (nx, nu), optional. Default is zero.
        Allows for non-quadratic objectives.
    system_type : str
        'continuous' or 'discrete', default 'discrete'
    backend : Backend
        Computational backend ('numpy', 'torch', 'jax'), default 'numpy'

    Returns
    -------
    LQRResult
        Dictionary containing:
            - gain: Optimal feedback gain K (nu, nx)
            - cost_to_go: Riccati solution P (nx, nx)
            - controller_eigenvalues: Eigenvalues of (A - BK)
            - stability_margin: Distance from stability boundary
              * Continuous: -max(Re(λ)) (positive = stable)
              * Discrete: 1 - max(|λ|) (positive = stable)

    Raises
    ------
    ValueError
        If matrices have incompatible shapes or invalid system_type
    LinAlgError
        If Riccati equation has no solution (system may be unstabilizable)

    Examples
    --------
    Continuous-time double integrator:

    >>> A = np.array([[0, 1], [0, 0]])
    >>> B = np.array([[0], [1]])
    >>> Q = np.diag([10, 1])  # Penalize position more
    >>> R = np.array([[0.1]])
    >>>
    >>> result = design_lqr(A, B, Q, R, system_type='continuous')
    >>> K = result['gain']
    >>> print(f"Gain: {K}")
    >>> print(f"Stable: {result['stability_margin'] > 0}")

    Discrete-time system:

    >>> Ad = np.array([[1, 0.1], [0, 1]])
    >>> Bd = np.array([[0.005], [0.1]])
    >>> Q = np.diag([10, 1])
    >>> R = np.array([[0.1]])
    >>>
    >>> result = design_lqr(Ad, Bd, Q, R, system_type='discrete')
    >>> K = result['gain']
    >>>
    >>> # Apply control in simulation
    >>> x = np.array([1.0, 0.0])
    >>> for k in range(100):
    ...     u = -K @ x
    ...     x = Ad @ x + Bd @ u

    With cross-coupling term:

    >>> N = np.array([[0.5], [0.1]])
    >>> result = design_lqr(A, B, Q, R, N=N, system_type='continuous')

    Using PyTorch backend:

    >>> import torch
    >>> A_torch = torch.tensor(A, dtype=torch.float64)
    >>> B_torch = torch.tensor(B, dtype=torch.float64)
    >>> Q_torch = torch.tensor(Q, dtype=torch.float64)
    >>> R_torch = torch.tensor(R, dtype=torch.float64)
    >>>
    >>> result = design_lqr(
    ...     A_torch, B_torch, Q_torch, R_torch,
    ...     system_type='continuous',
    ...     backend='torch'
    ... )
    >>> K = result['gain']  # Returns torch.Tensor

    Notes
    -----
    **Controllability Requirements:**
    - Full controllability: (A, B) must be controllable for arbitrary pole placement
    - Stabilizability: Unstable modes must be controllable (weaker, sufficient for LQR)

    **Cost Matrix Requirements:**
    - Q must be positive semi-definite (all eigenvalues ≥ 0)
    - R must be positive definite (all eigenvalues > 0)
    - (Q, A) should be detectable for finite-horizon convergence

    **Stability:**
    - Continuous: Closed-loop stable if all Re(λ) < 0 (left half-plane)
    - Discrete: Closed-loop stable if all |λ| < 1 (inside unit circle)
    - stability_margin > 0 indicates asymptotic stability

    **Cross-Coupling Term N:**
    - Allows non-standard quadratic costs
    - Useful for systems with control-state coupling
    - Set to None (default) for standard LQR

    **Numerical Considerations:**
    - Uses scipy's solve_continuous_are / solve_discrete_are
    - Numerical issues may arise for ill-conditioned systems
    - Consider scaling states/controls for better conditioning

    See Also
    --------
    design_kalman_filter : Optimal state estimator (dual to LQR)
    design_lqg : Combined LQR + Kalman filter
    analyze_controllability : Test controllability of (A, B)
    analyze_stability : Analyze closed-loop stability
    """
    # Validate system_type first
    if system_type not in ["continuous", "discrete"]:
        raise ValueError(
            f"system_type must be 'continuous' or 'discrete', got '{system_type}'",
        )

    # Convert to NumPy for scipy
    A_np = _to_numpy(A, backend)
    B_np = _to_numpy(B, backend)
    Q_np = _to_numpy(Q, backend)
    R_np = _to_numpy(R, backend)

    # Validate shapes
    nx = A_np.shape[0]
    nu = B_np.shape[1]

    if A_np.shape != (nx, nx):
        raise ValueError(f"A must be square, got shape {A_np.shape}")
    if B_np.shape[0] != nx:
        raise ValueError(f"B must have {nx} rows, got {B_np.shape[0]}")
    if Q_np.shape != (nx, nx):
        raise ValueError(f"Q must be ({nx}, {nx}), got {Q_np.shape}")
    if R_np.shape != (nu, nu):
        raise ValueError(f"R must be ({nu}, {nu}), got {R_np.shape}")

    # Process cross-coupling term N if provided
    N_np = None
    if N is not None:
        N_np = _to_numpy(N, backend)
        if N_np.shape != (nx, nu):
            raise ValueError(f"N must be ({nx}, {nu}), got {N_np.shape}")

    # Solve appropriate Riccati equation and compute gain
    if system_type == "continuous":
        # Continuous-time Algebraic Riccati Equation (CARE)
        if N_np is not None:
            P = linalg.solve_continuous_are(A_np, B_np, Q_np, R_np, s=N_np)
            # K = R^{-1}(B'P + N')
            K = linalg.solve(R_np, B_np.T @ P + N_np.T)
        else:
            P = linalg.solve_continuous_are(A_np, B_np, Q_np, R_np)
            # K = R^{-1}B'P
            K = linalg.solve(R_np, B_np.T @ P)

        # Closed-loop system: A_cl = A - BK
        A_cl = A_np - B_np @ K
        eigenvalues = np.linalg.eigvals(A_cl)

        # Stability margin for continuous: -max(Re(λ))
        # Positive margin = stable (all Re(λ) < 0)
        stability_margin = -np.max(np.real(eigenvalues))

    else:  # discrete
        # Discrete-time Algebraic Riccati Equation (DARE)
        if N_np is not None:
            P = linalg.solve_discrete_are(A_np, B_np, Q_np, R_np, s=N_np)
            # K = (R + B'PB)^{-1}(B'PA + N')
            K = linalg.solve(R_np + B_np.T @ P @ B_np, B_np.T @ P @ A_np + N_np.T)
        else:
            P = linalg.solve_discrete_are(A_np, B_np, Q_np, R_np)
            # K = (R + B'PB)^{-1}B'PA
            K = linalg.solve(R_np + B_np.T @ P @ B_np, B_np.T @ P @ A_np)

        # Closed-loop system
        A_cl = A_np - B_np @ K
        eigenvalues = np.linalg.eigvals(A_cl)

        # Stability margin for discrete: 1 - max(|λ|)
        # Positive margin = stable (all |λ| < 1)
        max_magnitude = np.max(np.abs(eigenvalues))
        stability_margin = 1.0 - max_magnitude

    # Convert back to target backend
    result: LQRResult = {
        "gain": _from_numpy(K, backend),
        "cost_to_go": _from_numpy(P, backend),
        "controller_eigenvalues": _from_numpy(eigenvalues, backend),
        "stability_margin": float(stability_margin),
    }

    return result


# ============================================================================
# Kalman Filter - Optimal State Estimation
# ============================================================================


def design_kalman_filter(
    A: StateMatrix,
    C: OutputMatrix,
    Q: StateMatrix,
    R: OutputMatrix,
    system_type: str = "discrete",
    backend: Backend = "numpy",
) -> KalmanFilterResult:
    """
    Design Kalman filter for optimal state estimation.

    For linear system with Gaussian noise:
        x[k+1] = Ax[k] + Bu[k] + w[k],  w ~ N(0, Q)  (process noise)
        y[k] = Cx[k] + v[k],            v ~ N(0, R)  (measurement noise)

    Kalman filter provides optimal state estimate:
        Discrete: x̂[k+1] = Ax̂[k] + Bu[k] + L(y[k] - Cx̂[k])
        Continuous: ˙x̂ = Ax̂ + Bu + L(y - Cx̂)

    Minimizes steady-state estimation error covariance.

    Args:
        A: State matrix (nx, nx)
        C: Output matrix (ny, nx)
        Q: Process noise covariance (nx, nx), Q ≥ 0
        R: Measurement noise covariance (ny, ny), R > 0
        system_type: 'continuous' or 'discrete'
        backend: Computational backend

    Returns:
        KalmanFilterResult containing:
            - gain: Kalman gain L (nx, ny)
            - error_covariance: Steady-state error covariance P (nx, nx)
            - innovation_covariance: Innovation covariance S = CPC' + R (ny, ny)
            - estimator_eigenvalues: Eigenvalues of (A - LC)

    Raises:
        ValueError: If matrices have incompatible shapes or invalid system_type
        LinAlgError: If Riccati equation has no solution

    Examples
    --------
    >>> # Discrete Kalman filter
    >>> A = np.array([[1, 0.1], [0, 0.95]])  # Slightly unstable
    >>> C = np.array([[1, 0]])  # Measure position only
    >>> Q = 0.01 * np.eye(2)    # Small process noise
    >>> R = np.array([[0.1]])   # Measurement noise
    >>>
    >>> result = design_kalman_filter(A, C, Q, R, system_type='discrete')
    >>> L = result['gain']
    >>> print(f"Kalman gain: {L}")
    >>>
    >>> # Use in estimation loop
    >>> x_hat = np.zeros(2)
    >>> for k in range(N):
    ...     # Prediction
    ...     x_hat_pred = A @ x_hat + B @ u[k]
    ...
    ...     # Correction
    ...     innovation = y[k] - C @ x_hat_pred
    ...     x_hat = x_hat_pred + L @ innovation
    >>>
    >>> # Check estimator stability
    >>> print(f"Estimator stable: {np.all(np.abs(result['estimator_eigenvalues']) < 1)}")
    >>>
    >>> # Continuous Kalman filter
    >>> result_c = design_kalman_filter(A, C, Q, R, system_type='continuous')
    >>> L_c = result_c['gain']

    Notes
    -----
    - For observability, (A, C) must be observable
    - For detectability, unstable modes must be observable
    - Q must be positive semi-definite (process noise)
    - R must be positive definite (measurement noise)
    - Kalman filter is optimal for linear Gaussian systems
    - For nonlinear systems, use Extended Kalman Filter (EKF) or Unscented Kalman Filter (UKF)
    """
    # Convert to NumPy
    A_np = _to_numpy(A, backend)
    C_np = _to_numpy(C, backend)
    Q_np = _to_numpy(Q, backend)
    R_np = _to_numpy(R, backend)

    # Validate shapes
    nx = A_np.shape[0]
    ny = C_np.shape[0]

    if A_np.shape != (nx, nx):
        raise ValueError(f"A must be square, got shape {A_np.shape}")
    if C_np.shape[1] != nx:
        raise ValueError(f"C must have {nx} columns, got {C_np.shape[1]}")
    if Q_np.shape != (nx, nx):
        raise ValueError(f"Q must be ({nx}, {nx}), got {Q_np.shape}")
    if R_np.shape != (ny, ny):
        raise ValueError(f"R must be ({ny}, {ny}), got {R_np.shape}")

    if system_type not in ["continuous", "discrete"]:
        raise ValueError(f"system_type must be 'continuous' or 'discrete', got '{system_type}'")

    if system_type == "continuous":
        # Continuous-time Kalman filter
        # Solve: PA' + AP - PC'R^{-1}CP + Q = 0 (dual of LQR)
        P = linalg.solve_continuous_are(A_np.T, C_np.T, Q_np, R_np)
        # Kalman gain: L = PC'R^{-1}
        L = P @ C_np.T @ linalg.inv(R_np)
        # Innovation covariance
        S = C_np @ P @ C_np.T + R_np
        # Estimator dynamics: A - LC
        A_estimator = A_np - L @ C_np
    else:
        # Discrete-time Kalman filter
        # Solve: P = APA' - APC'(CPC' + R)^{-1}CPA' + Q (dual of LQR)
        P = linalg.solve_discrete_are(A_np.T, C_np.T, Q_np, R_np)
        # Innovation covariance: S = CPC' + R
        S = C_np @ P @ C_np.T + R_np
        # Kalman gain: L = APC'S^{-1}
        L = A_np @ P @ C_np.T @ linalg.inv(S)
        # estimator dynamics: A - LC
        A_estimator = A_np - L @ C_np

    # estimator eigenvalues (for convergence rate)
    estimator_eigenvalues = np.linalg.eigvals(A_estimator)

    # Convert back to target backend
    result: KalmanFilterResult = {
        "gain": _from_numpy(L, backend),
        "error_covariance": _from_numpy(P, backend),
        "innovation_covariance": _from_numpy(S, backend),
        "estimator_eigenvalues": _from_numpy(estimator_eigenvalues, backend),
    }

    return result


# ============================================================================
# LQG - Linear Quadratic Gaussian (LQR + Kalman)
# ============================================================================


def design_lqg(
    A: StateMatrix,
    B: InputMatrix,
    C: OutputMatrix,
    Q_state: StateMatrix,
    R_control: InputMatrix,
    Q_process: StateMatrix,
    R_measurement: OutputMatrix,
    N: Optional[InputMatrix] = None,
    system_type: str = "discrete",
    backend: Backend = "numpy",
) -> LQGResult:
    """
    Design Linear Quadratic Gaussian (LQG) controller.

    Combines LQR controller with Kalman filter estimator.
    Separation principle allows independent design of controller and estimator.

    System:
        x[k+1] = Ax[k] + Bu[k] + w[k],  w ~ N(0, Q_process)
        y[k] = Cx[k] + v[k],            v ~ N(0, R_measurement)

    Controller: u[k] = -Kx̂[k] (feedback on estimate)
    Estimator: x̂[k+1] = Ax̂[k] + Bu[k] + L(y[k] - Cx̂[k])

    Args:
        A: State matrix (nx, nx)
        B: Input matrix (nx, nu)
        C: Output matrix (ny, nx)
        Q_state: LQR state cost matrix (nx, nx)
        R_control: LQR control cost matrix (nu, nu)
        Q_process: Process noise covariance (nx, nx)
        R_measurement: Measurement noise covariance (ny, ny)
        system_type: 'continuous' or 'discrete'
        backend: Computational backend

    Returns:
        LQGResult containing:
            - controller_gain: LQR gain K (nu, nx)
            - estimator_gain: Kalman gain L (nx, ny)
            - controller_riccati: LQR Riccati solution P_c (nx, nx)
            - estimator_covariance: Kalman covariance P_e (nx, nx)
            - controller_eigenvalues: Controller eigenvalues of (A - BK)
            - estimator_eigenvalues: Estimator eigenvalues of (A - LC)

    Examples
    --------
    >>> # Design LQG controller
    >>> A = np.array([[1, 0.1], [0, 0.9]])
    >>> B = np.array([[0], [0.1]])
    >>> C = np.array([[1, 0]])  # Measure position
    >>>
    >>> # LQR weights
    >>> Q_state = np.diag([10, 1])
    >>> R_control = np.array([[0.1]])
    >>>
    >>> # Noise covariances
    >>> Q_process = 0.01 * np.eye(2)
    >>> R_measurement = np.array([[0.1]])
    >>>
    >>> result = design_lqg(
    ...     A, B, C,
    ...     Q_state, R_control,
    ...     Q_process, R_measurement,
    ...     system_type='discrete'
    ... )
    >>>
    >>> K = result['controller_gain']
    >>> L = result['estimator_gain']
    >>>
    >>> # Implementation
    >>> x_hat = np.zeros(2)
    >>> for k in range(N):
    ...     # Control
    ...     u = -K @ x_hat
    ...
    ...     # Estimation
    ...     x_hat_pred = A @ x_hat + B @ u
    ...     innovation = y[k] - C @ x_hat_pred
    ...     x_hat = x_hat_pred + L @ innovation

    Notes
    -----
    - Separation principle: LQR and Kalman can be designed independently
    - LQG is optimal for linear systems with Gaussian noise
    - Closed-loop has eigenvalues of both controller and estimator
    - Controller must stabilize system
    - Estimator must converge faster than controller for good performance
    - Trade-off: Lower Q_process/R_measurement → more aggressive estimator
    """
    
    # Design LQR controller
    lqr_result = design_lqr(
        A=A,
        B=B,
        Q=Q_state,
        R=R_control,
        N=N,
        system_type=system_type,
        backend=backend,
    )

    # Design Kalman filter estimator
    kalman_result = design_kalman_filter(
        A, C, Q_process, R_measurement, system_type, backend
    )

    # Check stability and separation
    closed_loop_stable = bool(
        lqr_result["stability_margin"] > 0 
        and (np.max(np.abs(kalman_result["estimator_eigenvalues"])) < 1 
             if system_type == "discrete" 
             else np.max(np.real(kalman_result["estimator_eigenvalues"])) < 0)
    )
    
    # Separation principle always holds for linear systems
    separation_verified = True

    # Construct LQG result - match TypedDict field names exactly
    result: LQGResult = {
        "control_gain": lqr_result["gain"],
        "estimator_gain": kalman_result["gain"],
        "control_cost_to_go": lqr_result["cost_to_go"],
        "estimation_error_covariance": kalman_result["error_covariance"],  # Changed from estimator_covariance
        "separation_verified": separation_verified,
        "closed_loop_stable": closed_loop_stable,
        "controller_eigenvalues": lqr_result["controller_eigenvalues"],
        "estimator_eigenvalues": kalman_result["estimator_eigenvalues"],
    }

    return result


# ============================================================================
# Stability Analysis
# ============================================================================


def analyze_stability(
    A: StateMatrix,
    system_type: str = "continuous",
    tolerance: float = 1e-10,
) -> StabilityInfo:
    """
    Analyze system stability via eigenvalue analysis.

    Stability criteria:
        Continuous (dx/dt = Ax): All Re(λ) < 0 (left half-plane)
        Discrete (x[k+1] = Ax): All |λ| < 1 (inside unit circle)

    Args:
        A: State matrix (nx, nx)
        system_type: 'continuous' or 'discrete'
        tolerance: Tolerance for marginal stability detection

    Returns:
        StabilityInfo containing:
            - eigenvalues: Eigenvalues of A (complex array)
            - magnitudes: |λ| for all eigenvalues
            - max_magnitude: max(|λ|) = spectral radius
            - spectral_radius: Same as max_magnitude
            - is_stable: True if asymptotically stable
            - is_marginally_stable: True if critically stable
            - is_unstable: True if unstable

    Examples
    --------
    >>> # Stable continuous system
    >>> A = np.array([[0, 1], [-2, -3]])
    >>> stability = analyze_stability(A, system_type='continuous')
    >>> print(stability['is_stable'])  # True
    >>> print(stability['eigenvalues'])  # [-1, -2]
    >>>
    >>> # Unstable continuous system
    >>> A_unstable = np.array([[1, 1], [0, 1]])
    >>> stability = analyze_stability(A_unstable, system_type='continuous')
    >>> print(stability['is_unstable'])  # True
    >>>
    >>> # Stable discrete system
    >>> Ad = np.array([[0.9, 0.1], [0, 0.8]])
    >>> stability = analyze_stability(Ad, system_type='discrete')
    >>> print(stability['is_stable'])  # True
    >>> print(stability['spectral_radius'])  # 0.9
    >>>
    >>> # Marginally stable (on boundary)
    >>> A_marginal = np.array([[0, 1], [-1, 0]])  # Pure oscillation
    >>> stability = analyze_stability(A_marginal, system_type='continuous')
    >>> print(stability['is_marginally_stable'])  # True

    Notes
    -----
    - Marginal stability: Eigenvalues on stability boundary
        - Continuous: Re(λ) = 0 (imaginary axis)
        - Discrete: |λ| = 1 (unit circle)
    - Asymptotic stability: All trajectories converge to zero
    - Lyapunov stability: Bounded trajectories (includes marginal)
    """
    # Convert to NumPy
    A_np = np.asarray(A)

    if A_np.ndim != 2 or A_np.shape[0] != A_np.shape[1]:
        raise ValueError(f"A must be square matrix, got shape {A_np.shape}")

    # Compute eigenvalues
    eigenvalues = np.linalg.eigvals(A_np)
    magnitudes = np.abs(eigenvalues)
    max_magnitude = np.max(magnitudes)

    if system_type == "continuous":
        # Continuous: check real parts
        max_real = np.max(np.real(eigenvalues))
        is_stable = max_real < -tolerance
        is_marginally_stable = np.abs(max_real) <= tolerance
        is_unstable = max_real > tolerance
    elif system_type == "discrete":
        # Discrete: check magnitudes
        is_stable = max_magnitude < 1.0 - tolerance
        is_marginally_stable = np.abs(max_magnitude - 1.0) <= tolerance
        is_unstable = max_magnitude > 1.0 + tolerance
    else:
        raise ValueError(f"system_type must be 'continuous' or 'discrete', got '{system_type}'")

    result: StabilityInfo = {
        "eigenvalues": eigenvalues,
        "magnitudes": magnitudes,
        "max_magnitude": float(max_magnitude),
        "spectral_radius": float(max_magnitude),
        "is_stable": bool(is_stable),
        "is_marginally_stable": bool(is_marginally_stable),
        "is_unstable": bool(is_unstable),
    }

    return result


# ============================================================================
# Controllability Analysis
# ============================================================================


def analyze_controllability(
    A: StateMatrix,
    B: InputMatrix,
    tolerance: float = 1e-10,
) -> ControllabilityInfo:
    """
    Test controllability of linear system (A, B).

    A system is controllable if all states can be driven to any desired
    value in finite time using appropriate control inputs.

    Controllability test:
        rank(C) = n, where C = [B, AB, A²B, ..., Aⁿ⁻¹B]

    Args:
        A: State matrix (nx, nx)
        B: Input matrix (nx, nu)
        tolerance: Tolerance for rank computation

    Returns:
        ControllabilityInfo containing:
            - controllability_matrix: C = [B, AB, ...] (nx, nx*nu)
            - rank: Rank of controllability matrix
            - is_controllable: True if rank = nx (full rank)
            - uncontrollable_modes: Eigenvalues of uncontrollable subspace (if any)

    Examples
    --------
    >>> # Fully controllable
    >>> A = np.array([[0, 1], [-2, -3]])
    >>> B = np.array([[0], [1]])
    >>> info = analyze_controllability(A, B)
    >>> print(info['is_controllable'])  # True
    >>> print(info['rank'])  # 2
    >>>
    >>> # Uncontrollable system (diagonal with identical input)
    >>> A = np.array([[1, 0], [0, 2]])
    >>> B = np.array([[1], [1]])  # Can't control modes independently
    >>> info = analyze_controllability(A, B)
    >>> print(info['is_controllable'])  # False
    >>> print(info['rank'])  # 1
    >>>
    >>> # Single-input controllable
    >>> A = np.array([[0, 1, 0], [0, 0, 1], [-1, -2, -3]])
    >>> B = np.array([[0], [0], [1]])
    >>> info = analyze_controllability(A, B)
    >>> print(info['is_controllable'])  # True

    Notes
    -----
    - Controllability is necessary for pole placement
    - Stabilizability: Unstable modes must be controllable (weaker condition)
    - Numerical issues: Use SVD for better numerical stability
    - For large systems, consider PBH test or Gram matrix
    """
    # Convert to NumPy
    A_np = np.asarray(A)
    B_np = np.asarray(B)

    nx = A_np.shape[0]
    nu = B_np.shape[1]

    if A_np.shape != (nx, nx):
        raise ValueError(f"A must be square, got shape {A_np.shape}")
    if B_np.shape[0] != nx:
        raise ValueError(f"B must have {nx} rows, got {B_np.shape[0]}")

    # Build controllability matrix: C = [B, AB, A²B, ..., Aⁿ⁻¹B]
    C = np.zeros((nx, nx * nu))
    C[:, :nu] = B_np

    AB = B_np.copy()
    for i in range(1, nx):
        AB = A_np @ AB
        C[:, i * nu : (i + 1) * nu] = AB

    # Compute rank
    rank = np.linalg.matrix_rank(C, tol=tolerance)
    is_controllable = rank == nx

    # Find uncontrollable modes (if any)
    # TODO: Implement PBH test or controllability decomposition
    uncontrollable_modes = None
    if not is_controllable:
        # For now, just note that some modes are uncontrollable
        # Full implementation would use Kalman decomposition
        pass

    result: ControllabilityInfo = {
        "controllability_matrix": C,
        "rank": int(rank),
        "is_controllable": bool(is_controllable),
        "uncontrollable_modes": uncontrollable_modes,
    }

    return result


# ============================================================================
# Observability Analysis
# ============================================================================


def analyze_observability(
    A: StateMatrix,
    C: OutputMatrix,
    tolerance: float = 1e-10,
) -> ObservabilityInfo:
    """
    Test observability of linear system (A, C).

    A system is observable if the initial state can be determined from
    output measurements over a finite time interval.

    Observability test:
        rank(O) = n, where O = [C; CA; CA²; ...; CAⁿ⁻¹]

    Args:
        A: State matrix (nx, nx)
        C: Output matrix (ny, nx)
        tolerance: Tolerance for rank computation

    Returns:
        ObservabilityInfo containing:
            - observability_matrix: O = [C; CA; ...] (nx*ny, nx)
            - rank: Rank of observability matrix
            - is_observable: True if rank = nx (full rank)
            - unobservable_modes: Eigenvalues of unobservable subspace (if any)

    Examples
    --------
    >>> # Fully observable
    >>> A = np.array([[0, 1], [-2, -3]])
    >>> C = np.array([[1, 0]])  # Measure position only
    >>> info = analyze_observability(A, C)
    >>> print(info['is_observable'])  # True
    >>> print(info['rank'])  # 2
    >>>
    >>> # Unobservable system
    >>> A = np.array([[1, 0], [0, 2]])
    >>> C = np.array([[1, 1]])  # Can't distinguish states
    >>> info = analyze_observability(A, C)
    >>> print(info['is_observable'])  # False
    >>>
    >>> # Full state measurement
    >>> A = np.array([[0, 1, 0], [0, 0, 1], [-1, -2, -3]])
    >>> C = np.eye(3)  # Measure all states
    >>> info = analyze_observability(A, C)
    >>> print(info['is_observable'])  # True

    Notes
    -----
    - Observability is necessary for state estimation (Kalman filter)
    - Detectability: Unstable modes must be observable (weaker condition)
    - Dual to controllability: (A, C) observable ⟺ (A', C') controllable
    - For large systems, use dual controllability test
    """
    # Convert to NumPy
    A_np = np.asarray(A)
    C_np = np.asarray(C)

    nx = A_np.shape[0]
    ny = C_np.shape[0]

    if A_np.shape != (nx, nx):
        raise ValueError(f"A must be square, got shape {A_np.shape}")
    if C_np.shape[1] != nx:
        raise ValueError(f"C must have {nx} columns, got {C_np.shape[1]}")

    # Build observability matrix: O = [C; CA; CA²; ...; CAⁿ⁻¹]
    O = np.zeros((nx * ny, nx))
    O[:ny, :] = C_np

    CA = C_np.copy()
    for i in range(1, nx):
        CA = CA @ A_np
        O[i * ny : (i + 1) * ny, :] = CA

    # Compute rank
    rank = np.linalg.matrix_rank(O, tol=tolerance)
    is_observable = rank == nx

    # Find unobservable modes (if any)
    # TODO: Implement PBH test or observability decomposition
    unobservable_modes = None
    if not is_observable:
        # For now, just note that some modes are unobservable
        # Full implementation would use Kalman decomposition
        pass

    result: ObservabilityInfo = {
        "observability_matrix": O,
        "rank": int(rank),
        "is_observable": bool(is_observable),
        "unobservable_modes": unobservable_modes,
    }

    return result


# ============================================================================
# Export All
# ============================================================================

__all__ = [
    # LQR
    "design_lqr",
    # Kalman Filter
    "design_kalman_filter",
    # LQG
    "design_lqg",
    # Analysis
    "analyze_stability",
    "analyze_controllability",
    "analyze_observability",
]
