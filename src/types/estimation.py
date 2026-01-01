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
Nonlinear State Estimation Types

Result types for nonlinear state estimation algorithms:
- Extended Kalman Filter (EKF)
- Unscented Kalman Filter (UKF)
- Particle Filter (PF)

These estimators extend the linear Kalman filter to handle nonlinear
system dynamics and measurement models.

Mathematical Background
----------------------
Extended Kalman Filter (EKF):
    Linearizes nonlinear functions via first-order Taylor expansion:

    Prediction:
        x̂⁻[k+1] = f(x̂[k], u[k])
        P⁻[k+1] = F P[k] F' + Q
        where F = ∂f/∂x|x̂[k]

    Update:
        K[k] = P⁻[k] H'(H P⁻[k] H' + R)⁻¹
        x̂[k] = x̂⁻[k] + K[k](y[k] - h(x̂⁻[k]))
        P[k] = (I - K[k]H)P⁻[k]
        where H = ∂h/∂x|x̂⁻[k]

    Limitations: Assumes small nonlinearity, requires Jacobians

Unscented Kalman Filter (UKF):
    Uses deterministic sigma points to capture mean and covariance:

    Sigma Points (for nx-dimensional state):
        χ₀ = x̂
        χᵢ = x̂ + (√((nx+λ)P))ᵢ,  i = 1,...,nx
        χᵢ = x̂ - (√((nx+λ)P))ᵢ₋ₙₓ,  i = nx+1,...,2nx

    Propagation:
        γᵢ = f(χᵢ, u)
        x̂⁻ = Σ wᵢᵐ γᵢ
        P⁻ = Σ wᵢᶜ (γᵢ - x̂⁻)(γᵢ - x̂⁻)' + Q

    Advantages: No Jacobians, better for highly nonlinear systems

Particle Filter (Sequential Monte Carlo):
    Represents posterior as weighted particles:

    Prediction:
        xᵢ[k+1] ~ p(x[k+1]|xᵢ[k], u[k])  (sample from dynamics)

    Update (Importance Sampling):
        wᵢ[k] ∝ wᵢ[k-1] · p(y[k]|xᵢ[k])  (measurement likelihood)

    Resampling (when ESS < threshold):
        Draw N particles from {xᵢ, wᵢ} distribution

    Advantages: No linearity/Gaussian assumptions, arbitrary posteriors
    Challenges: Particle degeneracy, computational cost

Usage
-----
>>> from src.types.estimation import (
...     EKFResult,
...     UKFResult,
...     ParticleFilterResult,
... )
>>>
>>> # Extended Kalman Filter
>>> ekf = ExtendedKalmanFilter(nonlinear_system, Q, R)
>>> result: EKFResult = ekf.update(y_measured, u_applied)
>>> x_hat = result['state_estimate']
>>> P = result['covariance']
>>> innovation = result['innovation']
>>>
>>> # Unscented Kalman Filter
>>> ukf = UnscentedKalmanFilter(nonlinear_system, Q, R, alpha=0.1, beta=2.0)
>>> result: UKFResult = ukf.update(y_measured, u_applied)
>>> sigma_points = result['sigma_points']  # Inspect sigma points
>>>
>>> # Particle Filter
>>> pf = ParticleFilter(nonlinear_system, n_particles=1000)
>>> result: ParticleFilterResult = pf.update(y_measured, u_applied)
>>> if result['effective_sample_size'] < 500:
...     print("Particle degeneracy detected")
"""

from typing_extensions import TypedDict

from .core import (
    ArrayLike,
    CovarianceMatrix,
    GainMatrix,
    OutputVector,
    StateVector,
)
from .trajectories import (
    StateTrajectory,
)

# ============================================================================
# Extended Kalman Filter
# ============================================================================


class EKFResult(TypedDict, total=False):
    """
    Extended Kalman Filter (EKF) state and result.

    EKF extends the Kalman filter to nonlinear systems by linearizing
    the dynamics and measurement functions at each time step via Jacobians.

    Fields
    ------
    state_estimate : StateVector
        Current state estimate x̂[k] (nx,)
    covariance : CovarianceMatrix
        Current error covariance P[k] (nx, nx)
    innovation : OutputVector
        Measurement innovation y[k] - h(x̂⁻[k]) (ny,)
    innovation_covariance : CovarianceMatrix
        Innovation covariance S[k] = H P⁻ H' + R (ny, ny)
    kalman_gain : GainMatrix
        Kalman gain K[k] = P⁻ H' S⁻¹ (nx, ny)
    likelihood : float
        Log-likelihood of measurement log p(y[k]|y[1:k-1])

    Examples
    --------
    >>> # Nonlinear pendulum system
    >>> def dynamics(x, u):
    ...     theta, omega = x
    ...     return np.array([omega, -np.sin(theta) + u[0]])
    >>>
    >>> def measurement(x):
    ...     return np.array([x[0]])  # Measure angle only
    >>>
    >>> # Create EKF
    >>> ekf = ExtendedKalmanFilter(
    ...     dynamics_fn=dynamics,
    ...     measurement_fn=measurement,
    ...     Q=0.01 * np.eye(2),
    ...     R=0.1 * np.eye(1)
    ... )
    >>>
    >>> # Initialize
    >>> ekf.initialize(x0=np.array([0.1, 0.0]), P0=np.eye(2))
    >>>
    >>> # Update loop
    >>> for k in range(N):
    ...     # Predict
    ...     ekf.predict(u[k])
    ...
    ...     # Update with measurement
    ...     result: EKFResult = ekf.update(y[k])
    ...
    ...     # Extract estimate
    ...     x_hat = result['state_estimate']
    ...     P = result['covariance']
    ...
    ...     # Check innovation
    ...     innovation = result['innovation']
    ...     if np.linalg.norm(innovation) > 3.0:
    ...         print(f"Large innovation at k={k}")
    >>>
    >>> # Examine likelihood for outlier detection
    >>> log_likelihood = result['likelihood']
    >>> if log_likelihood < -10:
    ...     print("Possible outlier measurement")
    """

    state_estimate: StateVector
    covariance: CovarianceMatrix
    innovation: OutputVector
    innovation_covariance: CovarianceMatrix
    kalman_gain: GainMatrix
    likelihood: float


# ============================================================================
# Unscented Kalman Filter
# ============================================================================


class UKFResult(TypedDict, total=False):
    """
    Unscented Kalman Filter (UKF) result.

    UKF uses the unscented transform to propagate mean and covariance
    through nonlinear functions without computing Jacobians.

    Fields
    ------
    state_estimate : StateVector
        State estimate x̂[k] (nx,)
    covariance : CovarianceMatrix
        Error covariance P[k] (nx, nx)
    innovation : OutputVector
        Measurement innovation y[k] - ŷ[k] (ny,)
    sigma_points : StateTrajectory
        Sigma points used χᵢ (2*nx+1, nx)
    weights_mean : ArrayLike
        Weights for mean computation wᵢᵐ (2*nx+1,)
    weights_covariance : ArrayLike
        Weights for covariance computation wᵢᶜ (2*nx+1,)

    Examples
    --------
    >>> # Highly nonlinear system (bearings-only tracking)
    >>> def dynamics(x, u):
    ...     # State: [x_pos, y_pos, x_vel, y_vel]
    ...     dt = 0.1
    ...     F = np.array([[1, 0, dt, 0],
    ...                   [0, 1, 0, dt],
    ...                   [0, 0, 1, 0],
    ...                   [0, 0, 0, 1]])
    ...     return F @ x
    >>>
    >>> def measurement(x):
    ...     # Measure bearing angle only
    ...     return np.array([np.arctan2(x[1], x[0])])
    >>>
    >>> # Create UKF with tuned parameters
    >>> ukf = UnscentedKalmanFilter(
    ...     dynamics_fn=dynamics,
    ...     measurement_fn=measurement,
    ...     Q=0.01 * np.eye(4),
    ...     R=0.05 * np.eye(1),
    ...     alpha=0.001,  # Spread of sigma points
    ...     beta=2.0,     # Prior knowledge (2 = Gaussian)
    ...     kappa=0.0     # Secondary scaling parameter
    ... )
    >>>
    >>> # Initialize
    >>> ukf.initialize(x0=np.array([1.0, 1.0, 0.1, 0.1]), P0=np.eye(4))
    >>>
    >>> # Update
    >>> result: UKFResult = ukf.update(y_measured)
    >>>
    >>> # Inspect sigma points (useful for debugging)
    >>> sigma_points = result['sigma_points']
    >>> print(f"Sigma points shape: {sigma_points.shape}")  # (9, 4)
    >>>
    >>> # Visualize sigma point spread
    >>> import matplotlib.pyplot as plt
    >>> plt.scatter(sigma_points[:, 0], sigma_points[:, 1])
    >>> plt.title("UKF Sigma Points")
    >>>
    >>> # Check weights
    >>> w_mean = result['weights_mean']
    >>> print(f"Weight sum: {np.sum(w_mean):.6f}")  # Should be 1.0
    """

    state_estimate: StateVector
    covariance: CovarianceMatrix
    innovation: OutputVector
    sigma_points: StateTrajectory
    weights_mean: ArrayLike
    weights_covariance: ArrayLike


# ============================================================================
# Particle Filter
# ============================================================================


class ParticleFilterResult(TypedDict, total=False):
    """
    Particle Filter (Sequential Monte Carlo) result.

    Particle filters represent the posterior distribution as a set of
    weighted samples (particles), enabling estimation for arbitrary
    nonlinear/non-Gaussian systems.

    Fields
    ------
    state_estimate : StateVector
        Mean of particle distribution x̂[k] = Σ wᵢ xᵢ (nx,)
    covariance : CovarianceMatrix
        Sample covariance P[k] (nx, nx)
    particles : StateTrajectory
        All particle states (n_particles, nx)
    weights : ArrayLike
        Normalized particle weights wᵢ (n_particles,)
    effective_sample_size : float
        ESS = 1/Σwᵢ² ∈ [1, n_particles]
    resampled : bool
        Whether resampling occurred this step

    Examples
    --------
    >>> # Highly nonlinear system with non-Gaussian noise
    >>> def dynamics(x, u, noise):
    ...     # Chaotic dynamics with state-dependent noise
    ...     return np.array([
    ...         x[1],
    ...         -np.sin(x[0]) + u[0] + noise[0] * (1 + x[0]**2)
    ...     ])
    >>>
    >>> def measurement(x, noise):
    ...     # Nonlinear measurement with outliers
    ...     return np.array([x[0]**2 + noise[0]])
    >>>
    >>> # Create Particle Filter
    >>> pf = ParticleFilter(
    ...     dynamics_fn=dynamics,
    ...     measurement_fn=measurement,
    ...     n_particles=1000,
    ...     resampling_threshold=0.5  # Resample when ESS < 500
    ... )
    >>>
    >>> # Initialize particles
    >>> pf.initialize(
    ...     x0_mean=np.array([0.0, 0.0]),
    ...     x0_cov=np.eye(2)
    ... )
    >>>
    >>> # Update loop
    >>> for k in range(N):
    ...     # Predict (propagate particles through dynamics)
    ...     pf.predict(u[k], Q=0.1*np.eye(2))
    ...
    ...     # Update (weight by measurement likelihood)
    ...     result: ParticleFilterResult = pf.update(y[k], R=0.5*np.eye(1))
    ...
    ...     # Extract estimate
    ...     x_hat = result['state_estimate']
    ...     P = result['covariance']
    ...
    ...     # Monitor particle degeneracy
    ...     ess = result['effective_sample_size']
    ...     if ess < 100:
    ...         print(f"Warning: Low ESS = {ess:.0f} at k={k}")
    ...
    ...     # Check if resampling occurred
    ...     if result['resampled']:
    ...         print(f"Resampling at k={k}")
    >>>
    >>> # Visualize particle distribution
    >>> particles = result['particles']
    >>> weights = result['weights']
    >>>
    >>> import matplotlib.pyplot as plt
    >>> plt.scatter(particles[:, 0], particles[:, 1],
    ...            s=weights*1000, alpha=0.5)
    >>> plt.title(f"Particle Distribution (ESS={ess:.0f})")
    >>>
    >>> # Compute credible intervals from particles
    >>> x_samples = particles[:, 0]
    >>> x_lower = np.percentile(x_samples, 2.5, weights=weights)
    >>> x_upper = np.percentile(x_samples, 97.5, weights=weights)
    >>> print(f"95% credible interval: [{x_lower:.3f}, {x_upper:.3f}]")
    """

    state_estimate: StateVector
    covariance: CovarianceMatrix
    particles: StateTrajectory
    weights: ArrayLike
    effective_sample_size: float
    resampled: bool


# ============================================================================
# Export All
# ============================================================================

__all__ = [
    "EKFResult",
    "ParticleFilterResult",
    "UKFResult",
]
