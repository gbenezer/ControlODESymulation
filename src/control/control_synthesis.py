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
Control Synthesis Wrapper

Thin wrapper around classical control functions for system composition.

Provides backend consistency with parent system while delegating to
pure functions in classical.py. This is NOT a heavy utility - it simply
routes to stateless algorithms with the appropriate backend setting.

Design Philosophy
-----------------
- Composition not inheritance
- Thin wrapper (no state, no caching)
- Routes to pure functions
- Backend consistency with parent system
- Clean integration with system classes

Architecture
------------
ControlSynthesis is a lightweight utility that:
1. Stores only the backend setting from parent system
2. Delegates all work to pure functions in classical.py
3. Provides clean method names for composition

Usage
-----
>>> # Direct instantiation (rare - usually via system)
>>> from src.control.synthesis import ControlSynthesis
>>> import numpy as np
>>>
>>> synthesis = ControlSynthesis(backend='numpy')
>>> A = np.array([[0, 1], [-2, -3]])
>>> B = np.array([[0], [1]])
>>> Q = np.diag([10, 1])
>>> R = np.array([[0.1]])
>>>
>>> result = synthesis.design_lqr_continuous(A, B, Q, R)
>>> K = result['gain']
>>>
>>> # Typical usage - via system composition
>>> system = Pendulum()
>>> A, B = system.linearize(x_eq, u_eq)
>>> result = system.control.design_lqr_continuous(A, B, Q, R)
"""

from typing import Optional

from src.types.backends import Backend
from src.types.control_classical import (
    KalmanFilterResult,
    LQGResult,
    LQRResult,
)
from src.types.core import (
    InputMatrix,
    OutputMatrix,
    StateMatrix,
)


class ControlSynthesis:
    """
    Control synthesis wrapper for system composition.

    Thin wrapper that routes to pure control design functions while
    maintaining backend consistency with parent system.

    This class holds minimal state (just backend setting) and delegates
    all computation to pure functions in classical.py.

    Attributes
    ----------
    backend : Backend
        Computational backend ('numpy', 'torch', 'jax')

    Examples
    --------
    >>> # Via system composition (typical usage)
    >>> system = Pendulum()
    >>>
    >>> # Design LQR controller
    >>> Q = np.diag([10, 1])
    >>> R = np.array([[0.1]])
    >>> result = system.control.design_lqr_continuous(
    ...     *system.linearize(x_eq, u_eq),
    ...     Q, R
    ... )
    >>> K = result['gain']
    >>>
    >>> # Design Kalman filter
    >>> C = np.array([[1, 0]])
    >>> Q_proc = 0.01 * np.eye(2)
    >>> R_meas = np.array([[0.1]])
    >>> A, _ = system.linearize(x_eq, u_eq)
    >>> kalman = system.control.design_kalman(A, C, Q_proc, R_meas)
    >>> L = kalman['gain']
    >>>
    >>> # Design LQG (combined)
    >>> lqg = system.control.design_lqg(
    ...     *system.linearize(x_eq, u_eq),
    ...     C,
    ...     Q, R,  # LQR weights
    ...     Q_proc, R_meas  # Kalman noise
    ... )

    Notes
    -----
    This is a thin wrapper - all algorithms are in classical.py.
    The wrapper only provides:
    1. Backend consistency with parent system
    2. Clean composition interface
    3. Convenience for system integration
    """

    def __init__(self, backend: Backend = "numpy"):
        """
        Initialize control synthesis wrapper.

        Args:
            backend: Computational backend from parent system
                     ('numpy', 'torch', 'jax')

        Examples
        --------
        >>> # Usually created by system, not directly
        >>> synthesis = ControlSynthesis(backend='torch')
        """
        self.backend = backend

    def design_lqr_continuous(
        self,
        A: StateMatrix,
        B: InputMatrix,
        Q: StateMatrix,
        R: InputMatrix,
        N: Optional[InputMatrix] = None,
    ) -> LQRResult:
        """
        Design continuous-time LQR controller.

        Routes to classical.design_lqr_continuous() with system backend.

        Minimizes: J = ∫₀^∞ (x'Qx + u'Ru + 2x'Nu) dt
        Control law: u = -Kx

        Args:
            A: State matrix (nx, nx)
            B: Input matrix (nx, nu)
            Q: State cost matrix (nx, nx), Q ≥ 0
            R: Control cost matrix (nu, nu), R > 0
            N: Cross-coupling matrix (nx, nu), optional

        Returns:
            LQRResult with gain, cost-to-go, eigenvalues, stability margin

        Examples
        --------
        >>> # Via system
        >>> A, B = system.linearize(x_eq, u_eq)
        >>> Q = np.diag([10, 1])  # Penalize position more
        >>> R = np.array([[0.1]])
        >>>
        >>> result = system.control.design_lqr_continuous(A, B, Q, R)
        >>> K = result['gain']
        >>> print(f"Gain shape: {K.shape}")  # (nu, nx)
        >>> print(f"Stable: {result['stability_margin'] > 0}")
        >>>
        >>> # Apply control
        >>> def control_policy(x):
        ...     return -K @ (x - x_eq)
        >>>
        >>> # Simulate closed-loop
        >>> result = system.integrate(
        ...     x0,
        ...     u_func=lambda t, x: control_policy(x),
        ...     t_span=(0, 10)
        ... )

        See Also
        --------
        design_lqr_discrete : Discrete-time LQR
        design_lqg : Combined LQR + Kalman filter
        """
        from .classical import design_lqr_continuous

        return design_lqr_continuous(A, B, Q, R, N, backend=self.backend)

    def design_lqr_discrete(
        self,
        A: StateMatrix,
        B: InputMatrix,
        Q: StateMatrix,
        R: InputMatrix,
        N: Optional[InputMatrix] = None,
    ) -> LQRResult:
        """
        Design discrete-time LQR controller.

        Routes to classical.design_lqr_discrete() with system backend.

        Minimizes: J = Σₖ₌₀^∞ (x[k]'Qx[k] + u[k]'Ru[k] + 2x[k]'Nu[k])
        Control law: u[k] = -Kx[k]

        Args:
            A: State matrix (nx, nx)
            B: Input matrix (nx, nu)
            Q: State cost matrix (nx, nx), Q ≥ 0
            R: Control cost matrix (nu, nu), R > 0
            N: Cross-coupling matrix (nx, nu), optional

        Returns:
            LQRResult with gain, cost-to-go, eigenvalues, stability margin

        Examples
        --------
        >>> # For discrete system
        >>> A, B = discrete_system.linearize(x_eq, u_eq)
        >>> Q = np.diag([10, 1])
        >>> R = np.array([[0.1]])
        >>>
        >>> result = discrete_system.control.design_lqr_discrete(A, B, Q, R)
        >>> K = result['gain']
        >>>
        >>> # Apply in simulation
        >>> x = x0
        >>> for k in range(100):
        ...     u = -K @ (x - x_eq)
        ...     x = discrete_system.step(x, u)

        See Also
        --------
        design_lqr_continuous : Continuous-time LQR
        """
        from .classical import design_lqr_discrete

        return design_lqr_discrete(A, B, Q, R, N, backend=self.backend)

    def design_kalman(
        self,
        A: StateMatrix,
        C: OutputMatrix,
        Q: StateMatrix,
        R: OutputMatrix,
        system_type: str = "discrete",
    ) -> KalmanFilterResult:
        """
        Design Kalman filter for optimal state estimation.

        Routes to classical.design_kalman_filter() with system backend.

        System:
            x[k+1] = Ax[k] + Bu[k] + w[k],  w ~ N(0, Q)
            y[k] = Cx[k] + v[k],            v ~ N(0, R)

        Estimator:
            x̂[k+1] = Ax̂[k] + Bu[k] + L(y[k] - Cx̂[k])

        Args:
            A: State matrix (nx, nx)
            C: Output matrix (ny, nx)
            Q: Process noise covariance (nx, nx), Q ≥ 0
            R: Measurement noise covariance (ny, ny), R > 0
            system_type: 'continuous' or 'discrete'

        Returns:
            KalmanFilterResult with gain, covariances, observer eigenvalues

        Examples
        --------
        >>> # Design Kalman filter
        >>> A, _ = system.linearize(x_eq, u_eq)
        >>> C = np.array([[1, 0]])  # Measure position only
        >>> Q_proc = 0.01 * np.eye(2)  # Process noise
        >>> R_meas = np.array([[0.1]])  # Measurement noise
        >>>
        >>> kalman = system.control.design_kalman(
        ...     A, C, Q_proc, R_meas,
        ...     system_type='discrete'
        ... )
        >>> L = kalman['gain']
        >>> print(f"Kalman gain shape: {L.shape}")  # (nx, ny)
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
        >>> # Check observer convergence
        >>> obs_stable = np.all(np.abs(kalman['observer_eigenvalues']) < 1)
        >>> print(f"Observer stable: {obs_stable}")

        See Also
        --------
        design_lqg : Combined LQR + Kalman (full LQG controller)
        """
        from .classical import design_kalman_filter

        return design_kalman_filter(A, C, Q, R, system_type, backend=self.backend)

    def design_lqg(
        self,
        A: StateMatrix,
        B: InputMatrix,
        C: OutputMatrix,
        Q_state: StateMatrix,
        R_control: InputMatrix,
        Q_process: StateMatrix,
        R_measurement: OutputMatrix,
        system_type: str = "discrete",
    ) -> LQGResult:
        """
        Design Linear Quadratic Gaussian (LQG) controller.

        Routes to classical.design_lqg() with system backend.

        Combines LQR controller with Kalman filter estimator via
        separation principle.

        Controller: u[k] = -Kx̂[k]
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

        Returns:
            LQGResult with controller gain, estimator gain, Riccati solutions,
            and eigenvalues

        Examples
        --------
        >>> # Design complete LQG controller
        >>> A, B = system.linearize(x_eq, u_eq)
        >>> C = np.array([[1, 0]])  # Partial state measurement
        >>>
        >>> # LQR weights
        >>> Q_state = np.diag([10, 1])
        >>> R_control = np.array([[0.1]])
        >>>
        >>> # Noise covariances
        >>> Q_process = 0.01 * np.eye(2)
        >>> R_measurement = np.array([[0.1]])
        >>>
        >>> lqg = system.control.design_lqg(
        ...     A, B, C,
        ...     Q_state, R_control,
        ...     Q_process, R_measurement,
        ...     system_type='discrete'
        ... )
        >>>
        >>> K = lqg['controller_gain']  # LQR gain
        >>> L = lqg['estimator_gain']   # Kalman gain
        >>>
        >>> # Implementation
        >>> x_hat = np.zeros(2)
        >>> for k in range(N):
        ...     # Control based on estimate
        ...     u = -K @ (x_hat - x_ref)
        ...
        ...     # State estimation
        ...     x_hat_pred = A @ x_hat + B @ u
        ...     innovation = y[k] - C @ x_hat_pred
        ...     x_hat = x_hat_pred + L @ innovation
        >>>
        >>> # Check closed-loop stability
        >>> ctrl_stable = lqg['closed_loop_eigenvalues']
        >>> obs_stable = lqg['observer_eigenvalues']

        Notes
        -----
        - Separation principle: LQR and Kalman designed independently
        - LQG optimal for linear Gaussian systems
        - Controller eigenvalues determine regulation performance
        - Observer eigenvalues determine estimation convergence
        - Estimator should converge faster than controller

        See Also
        --------
        design_lqr_continuous : LQR controller only
        design_lqr_discrete : Discrete LQR controller only
        design_kalman : Kalman filter only
        """
        from .classical import design_lqg

        return design_lqg(
            A,
            B,
            C,
            Q_state,
            R_control,
            Q_process,
            R_measurement,
            system_type,
            backend=self.backend,
        )


# ============================================================================
# Export
# ============================================================================

__all__ = [
    "ControlSynthesis",
]
