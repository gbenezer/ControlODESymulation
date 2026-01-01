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
Conformal Prediction Types

Result types for conformal inference and uncertainty quantification:
- Conformal prediction sets
- Distribution-free coverage guarantees
- Adaptive conformal inference
- Split conformal and full conformal

Conformal prediction provides rigorous uncertainty quantification
without distributional assumptions.

Mathematical Background
----------------------
Conformal Prediction:
    Goal: Predict set C(x) such that:
        P(y ∈ C(x)) ≥ 1 - α

    For ANY data distribution (distribution-free!)

Split Conformal:
    1. Split data: Train (n₁), Calibration (n₂)
    2. Train model: ŷ = f(x)
    3. Compute calibration scores: sᵢ = score(xᵢ, yᵢ, ŷᵢ)
    4. Quantile: q = Quantile(s₁,...,sₙ₂; (1-α)(1 + 1/n₂))
    5. Prediction set: C(x_new) = {y : score(x_new, y, ŷ_new) ≤ q}

    Guarantee: P(y_new ∈ C(x_new)) ≥ 1 - α

Nonconformity Scores:
    Regression:
        - Absolute residual: s = |y - ŷ|
        - Normalized: s = |y - ŷ| / σ̂(x)
        - Quantile: s = max(ŷ_lo - y, y - ŷ_hi)

    Classification:
        - Margin: s = 1 - p_y (probability of true class)
        - APS: s = Σ_{i: p_i ≥ p_y} p_i

Adaptive Conformal Inference (ACI):
    For online/time-series with distribution shift:
        q_t = q_{t-1} + γ(α - err_t)

    Where err_t = 1{y_t ∉ C_t(x_t)} (coverage error)

    Asymptotic guarantee: lim_{t→∞} coverage_t = 1 - α

Usage
-----
>>> from src.types.conformal import (
...     ConformalPredictionResult,
...     AdaptiveConformalResult,
... )
>>>
>>> # Calibrate conformal predictor
>>> cp = ConformalPredictor(model, calibration_data)
>>>
>>> # Predict with coverage guarantee
>>> result: ConformalPredictionResult = cp.predict(
...     x_test, alpha=0.1  # 90% coverage
... )
>>>
>>> prediction_set = result['prediction_set']
>>> coverage = result['coverage_guarantee']
>>> print(f"Guaranteed coverage: {coverage*100:.1f}%")
"""

from typing import Tuple, Union

from typing_extensions import TypedDict

from .core import (
    ArrayLike,
    StateVector,
)

# ============================================================================
# Type Aliases for Conformal Prediction
# ============================================================================

ConformalPredictionSet = Union[ArrayLike, Tuple[ArrayLike, ArrayLike]]
"""
Conformal prediction set for future states/outputs.

Provides distribution-free prediction intervals:
    P(y_true ∈ prediction_set) ≥ 1 - α

Different representations:
- Interval: (lower_bounds, upper_bounds) element-wise
- Ball: (center, radius) scalar
- Polytope: vertices array (n_vertices, n_dims)

Examples
--------
>>> # Interval prediction set
>>> prediction_set: ConformalPredictionSet = (
...     np.array([1.5, 2.0]),  # lower bounds
...     np.array([2.5, 3.0])   # upper bounds
... )
>>> 
>>> # Ball prediction set
>>> prediction_set: ConformalPredictionSet = (
...     np.array([2.0, 2.5]),  # center
...     0.5                     # radius
... )
"""

NonconformityScore = ArrayLike
"""
Nonconformity scores measuring prediction "strangeness".

Lower score → prediction more conforming to calibration data

Common scores:
- Absolute residual: |y - ŷ|
- Normalized residual: |y - ŷ| / σ̂(x)
- Mahalanobis distance: √((y - ŷ)' Σ^(-1) (y - ŷ))

Shape: (n_samples,) for batch scoring

Examples
--------
>>> # Absolute residual scores
>>> scores: NonconformityScore = np.abs(y_pred - y_true)
>>> 
>>> # Normalized scores
>>> scores: NonconformityScore = np.abs(y_pred - y_true) / std_pred
"""


# ============================================================================
# Conformal Calibration
# ============================================================================


class ConformalCalibrationResult(TypedDict):
    """
    Conformal prediction calibration result.

    Result from calibrating conformal predictor on calibration set.

    Fields
    ------
    quantile : float
        Calibrated quantile threshold q
    alpha : float
        Target miscoverage level (1 - coverage)
    empirical_coverage : float
        Observed coverage on calibration set
    calibration_scores : NonconformityScore
        Nonconformity scores from calibration data
    n_calibration : int
        Size of calibration set
    prediction_set_type : str
        Type of prediction set ('interval', 'ball', 'polytope')

    Examples
    --------
    >>> # Calibrate on residuals
    >>> residuals = np.abs(y_cal - y_pred_cal)
    >>>
    >>> result: ConformalCalibrationResult = calibrate_conformal(
    ...     scores=residuals,
    ...     alpha=0.1  # Target 90% coverage
    ... )
    >>>
    >>> q = result['quantile']
    >>> coverage = result['empirical_coverage']
    >>>
    >>> print(f"Calibrated quantile: {q:.3f}")
    >>> print(f"Empirical coverage: {coverage*100:.1f}%")
    >>> print(f"Calibration set size: {result['n_calibration']}")
    >>>
    >>> # Use quantile for prediction
    >>> # Prediction set: {y : |y - ŷ| ≤ q}
    >>> y_lower = y_pred_new - q
    >>> y_upper = y_pred_new + q
    """

    quantile: float
    alpha: float
    empirical_coverage: float
    calibration_scores: NonconformityScore
    n_calibration: int
    prediction_set_type: str


# ============================================================================
# Conformal Prediction
# ============================================================================


class ConformalPredictionResult(TypedDict, total=False):
    """
    Conformal prediction result for test points.

    Provides prediction sets with finite-sample coverage guarantees.

    Fields
    ------
    prediction_set : ConformalPredictionSet
        Prediction set(s) for test point(s)
    point_prediction : StateVector
        Point prediction (e.g., set center)
    coverage_guarantee : float
        Guaranteed coverage level (1 - α)
    average_set_size : float
        Average size of prediction sets
    nonconformity_score : NonconformityScore
        Nonconformity score(s) for test point(s)
    threshold : float
        Threshold q for set construction
    adaptive : bool
        Whether adaptive to input x

    Examples
    --------
    >>> # Create conformal predictor
    >>> cp = ConformalPredictor(
    ...     model=my_model,
    ...     calibration_data=(x_cal, y_cal)
    ... )
    >>>
    >>> # Predict with 90% coverage guarantee
    >>> result: ConformalPredictionResult = cp.predict(
    ...     x_test=np.array([1.5, 2.0]),
    ...     alpha=0.1
    ... )
    >>>
    >>> # Extract prediction set
    >>> if isinstance(result['prediction_set'], tuple):
    ...     lower, upper = result['prediction_set']
    ...     print(f"Prediction interval: [{lower}, {upper}]")
    ...
    >>> print(f"Coverage guarantee: {result['coverage_guarantee']*100:.1f}%")
    >>> print(f"Average set size: {result['average_set_size']:.3f}")
    >>>
    >>> # For multiple test points
    >>> result_batch: ConformalPredictionResult = cp.predict(
    ...     x_test=np.random.randn(100, 2),
    ...     alpha=0.05  # 95% coverage
    ... )
    >>>
    >>> avg_size = result_batch['average_set_size']
    >>> print(f"Average prediction set size: {avg_size:.3f}")
    """

    prediction_set: ConformalPredictionSet
    point_prediction: StateVector
    coverage_guarantee: float
    average_set_size: float
    nonconformity_score: NonconformityScore
    threshold: float
    adaptive: bool


# ============================================================================
# Adaptive Conformal Inference
# ============================================================================


class AdaptiveConformalResult(TypedDict):
    """
    Adaptive Conformal Inference (ACI) result.

    For online/sequential prediction with distribution shift.
    Adapts threshold to maintain target coverage over time.

    Fields
    ------
    threshold : float
        Current adaptive threshold q_t
    coverage_history : ArrayLike
        Coverage rate over time steps
    miscoverage_rate : float
        Current miscoverage rate
    target_alpha : float
        Target miscoverage level (1 - coverage)
    adaptation_rate : float
        Learning rate γ for threshold updates
    effective_sample_size : int
        Effective size of calibration window

    Examples
    --------
    >>> # Initialize adaptive conformal predictor
    >>> aci = AdaptiveConformalInference(
    ...     model=my_model,
    ...     target_alpha=0.1,     # Target 90% coverage
    ...     adaptation_rate=0.01  # Learning rate γ
    ... )
    >>>
    >>> # Online updates
    >>> results = []
    >>> for t in range(1000):
    ...     # Get new data point
    ...     x_t, y_t = data_stream[t]
    ...
    ...     # Predict with current threshold
    ...     result: AdaptiveConformalResult = aci.update(x_t, y_t)
    ...     results.append(result)
    ...
    ...     # Check current performance
    ...     if t % 100 == 0:
    ...         coverage = 1 - result['miscoverage_rate']
    ...         print(f"Step {t}: Coverage = {coverage*100:.1f}%")
    >>>
    >>> # Plot coverage over time
    >>> import matplotlib.pyplot as plt
    >>> coverage_hist = np.array([r['coverage_history'] for r in results])
    >>> plt.plot(coverage_hist)
    >>> plt.axhline(0.9, color='r', linestyle='--', label='Target')
    >>> plt.xlabel('Time step')
    >>> plt.ylabel('Coverage')
    >>> plt.legend()
    >>>
    >>> # Final statistics
    >>> final_result = results[-1]
    >>> print(f"Final threshold: {final_result['threshold']:.3f}")
    >>> print(f"Final miscoverage: {final_result['miscoverage_rate']*100:.1f}%")
    """

    threshold: float
    coverage_history: ArrayLike
    miscoverage_rate: float
    target_alpha: float
    adaptation_rate: float
    effective_sample_size: int


# ============================================================================
# Export All
# ============================================================================

__all__ = [
    # Type aliases
    "ConformalPredictionSet",
    "NonconformityScore",
    # Results
    "ConformalCalibrationResult",
    "ConformalPredictionResult",
    "AdaptiveConformalResult",
]
