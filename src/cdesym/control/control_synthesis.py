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
pure functions in classical_control_functions.py. This is NOT a heavy
utility - it simply routes to stateless algorithms with the appropriate
backend setting.

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
2. Delegates all work to pure functions in classical_control_functions.py
3. Provides clean method names for composition

Usage
-----
>>> # Direct instantiation (rare - usually via system)
>>> from controldesymulation.control.synthesis import ControlSynthesis
>>> import numpy as np
>>>
>>> synthesis = ControlSynthesis(backend='numpy')
>>> A = np.array([[0, 1], [-2, -3]])
>>> B = np.array([[0], [1]])
>>> Q = np.diag([10, 1])
>>> R = np.array([[0.1]])
>>>
>>> # Unified interface (recommended)
>>> result = synthesis.design_lqr(A, B, Q, R, system_type='continuous')
>>> K = result['gain']
>>>
>>> # Typical usage - via system composition
>>> system = Pendulum()
>>> A, B = system.linearize(x_eq, u_eq)
>>> result = system.control.design_lqr(A, B, Q, R, system_type='continuous')
"""

from typing import Optional

from cdesym.types.backends import Backend
from cdesym.types.control_classical import (
    KalmanFilterResult,
    LQGResult,
    LQRResult,
)
from cdesym.types.core import (
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
    all computation to pure functions in classical_control_functions.py.

    Attributes
    ----------
    backend : Backend
        Computational backend ('numpy', 'torch', 'jax')

    Examples
    --------
    >>> # Via system composition (typical usage)
    >>> system = Pendulum()
    >>>
    >>> # Design LQR controller (unified interface)
    >>> Q = np.diag([10, 1])
    >>> R = np.array([[0.1]])
    >>> result = system.control.design_lqr(
    ...     *system.linearize(x_eq, u_eq),
    ...     Q, R,
    ...     system_type='continuous'
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
    This is a thin wrapper - all algorithms are in classical_control_functions.py.
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

    def design_lqr(
        self,
        A: StateMatrix,
        B: InputMatrix,
        Q: StateMatrix,
        R: InputMatrix,
        N: Optional[InputMatrix] = None,
        system_type: str = "discrete",
    ) -> LQRResult:
        """
        Design LQR controller (unified interface).

        Routes to classical.design_lqr() with system backend.

        Minimizes cost functional:
            Continuous: J = ∫₀^∞ (x'Qx + u'Ru + 2x'Nu) dt
            Discrete:   J = Σₖ₌₀^∞ (x[k]'Qx[k] + u[k]'Ru[k] + 2x[k]'Nu[k])

        Control law:
            Continuous: u = -Kx
            Discrete:   u[k] = -Kx[k]

        Args:
            A: State matrix (nx, nx)
            B: Input matrix (nx, nu)
            Q: State cost matrix (nx, nx), Q ≥ 0
            R: Control cost matrix (nu, nu), R > 0
            N: Cross-coupling matrix (nx, nu), optional
            system_type: 'continuous' or 'discrete', default 'discrete'

        Returns:
            LQRResult with gain, cost-to-go, eigenvalues, stability margin

        Examples
        --------
        >>> # Via system - continuous
        >>> A, B = system.linearize(x_eq, u_eq)
        >>> Q = np.diag([10, 1])
        >>> R = np.array([[0.1]])
        >>>
        >>> result = system.control.design_lqr(
        ...     A, B, Q, R,
        ...     system_type='continuous'
        ... )
        >>> K = result['gain']
        >>>
        >>> # Via system - discrete (default)
        >>> result = discrete_system.control.design_lqr(A, B, Q, R)
        >>> K = result['gain']
        >>>
        >>> # With cross-coupling term
        >>> N = np.array([[0.5], [0.1]])
        >>> result = system.control.design_lqr(
        ...     A, B, Q, R, N=N,
        ...     system_type='continuous'
        ... )

        See Also
        --------
        design_kalman : Kalman filter design
        design_lqg : Combined LQR + Kalman filter
        """
        from cdesym.control.classical_control_functions import design_lqr

        return design_lqr(A, B, Q, R, N, system_type=system_type, backend=self.backend)

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
        >>> obs_stable = np.all(np.abs(kalman['estimator_eigenvalues']) < 1)
        >>> print(f"Observer stable: {obs_stable}")

        See Also
        --------
        design_lqg : Combined LQR + Kalman (full LQG controller)
        """
        from cdesym.control.classical_control_functions import design_kalman_filter

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
        N: Optional[InputMatrix] = None,
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
            N: Cross-coupling matrix (nx, nu), optional
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
        >>> K = lqg['control_gain']  # LQR gain
        >>> L = lqg['estimator_gain']   # Kalman gain
        >>>
        >>> # With cross-coupling term
        >>> N = np.array([[0.5], [0.1]])
        >>> lqg = system.control.design_lqg(
        ...     A, B, C,
        ...     Q_state, R_control,
        ...     Q_process, R_measurement,
        ...     N=N,
        ...     system_type='discrete'
        ... )

        Notes
        -----
        - Separation principle: LQR and Kalman designed independently
        - LQG optimal for linear Gaussian systems
        - Controller eigenvalues determine regulation performance
        - Observer eigenvalues determine estimation convergence
        - Estimator should converge faster than controller

        See Also
        --------
        design_lqr : Unified LQR controller design
        design_kalman : Kalman filter only
        """
        from cdesym.control.classical_control_functions import design_lqg

        return design_lqg(
            A,
            B,
            C,
            Q_state,
            R_control,
            Q_process,
            R_measurement,
            N,
            system_type,
            backend=self.backend,
        )


# ============================================================================
# Export
# ============================================================================

__all__ = [
    "ControlSynthesis",
]
