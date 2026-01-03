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
Unit Tests for Conformal Prediction Types

Tests TypedDict definitions and usage patterns for conformal inference
and uncertainty quantification types.
"""

import numpy as np
import pytest

from src.types.conformal import (
    AdaptiveConformalResult,
    ConformalCalibrationResult,
    ConformalPredictionResult,
    ConformalPredictionSet,
    NonconformityScore,
)


class TestTypeAliases:
    """Test type aliases for conformal prediction."""

    def test_prediction_set_interval(self):
        """Test interval prediction set."""
        # Interval: [lower, upper]
        lower = np.array([1.5, 2.0])
        upper = np.array([2.5, 3.0])

        prediction_set: ConformalPredictionSet = (lower, upper)

        lb, ub = prediction_set
        assert lb.shape == (2,)
        assert ub.shape == (2,)
        assert np.all(lb <= ub)

    def test_prediction_set_ball(self):
        """Test ball prediction set."""
        # Ball: center ± radius
        center = np.array([2.0, 2.5])
        radius = 0.5

        prediction_set: ConformalPredictionSet = (center, radius)

        c, r = prediction_set
        assert c.shape == (2,)
        assert r > 0

    def test_nonconformity_score(self):
        """Test nonconformity scores."""
        # Absolute residuals
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 1.9, 3.2])

        scores: NonconformityScore = np.abs(y_true - y_pred)

        assert scores.shape == (3,)
        assert np.all(scores >= 0)


class TestConformalCalibrationResult:
    """Test ConformalCalibrationResult TypedDict."""

    def test_calibration_result_creation(self):
        """Test creating calibration result."""
        scores = np.random.rand(100)
        alpha = 0.1
        n_cal = len(scores)

        # Compute quantile: (1-α)(1 + 1/n)
        q_level = (1 - alpha) * (1 + 1 / n_cal)
        q = np.quantile(scores, q_level)

        result: ConformalCalibrationResult = {
            "quantile": q,
            "alpha": alpha,
            "empirical_coverage": 1 - alpha,
            "calibration_scores": scores,
            "n_calibration": n_cal,
            "prediction_set_type": "interval",
        }

        assert result["quantile"] >= 0
        assert 0 <= result["alpha"] <= 1
        assert result["n_calibration"] == 100

    def test_calibration_quantile_monotonicity(self):
        """Test quantile increases with coverage."""
        scores = np.random.rand(100)

        # 90% coverage
        result_90: ConformalCalibrationResult = {
            "quantile": np.quantile(scores, 0.9),
            "alpha": 0.1,
            "empirical_coverage": 0.9,
            "calibration_scores": scores,
            "n_calibration": 100,
            "prediction_set_type": "interval",
        }

        # 95% coverage (higher)
        result_95: ConformalCalibrationResult = {
            "quantile": np.quantile(scores, 0.95),
            "alpha": 0.05,
            "empirical_coverage": 0.95,
            "calibration_scores": scores,
            "n_calibration": 100,
            "prediction_set_type": "interval",
        }

        # Higher coverage → higher quantile
        assert result_95["quantile"] >= result_90["quantile"]

    def test_calibration_empirical_coverage(self):
        """Test empirical coverage calculation."""
        scores = np.random.rand(100)
        alpha = 0.1
        n_cal = len(scores)

        # Quantile
        q_level = (1 - alpha) * (1 + 1 / n_cal)
        q = np.quantile(scores, q_level)

        # Empirical coverage
        coverage = np.mean(scores <= q)

        result: ConformalCalibrationResult = {
            "quantile": q,
            "alpha": alpha,
            "empirical_coverage": coverage,
            "calibration_scores": scores,
            "n_calibration": n_cal,
            "prediction_set_type": "interval",
        }

        # Coverage should be ≥ 1 - α
        assert result["empirical_coverage"] >= 1 - alpha - 0.01


class TestConformalPredictionResult:
    """Test ConformalPredictionResult TypedDict."""

    def test_prediction_result_creation(self):
        """Test creating prediction result."""
        lower = np.array([1.5])
        upper = np.array([2.5])

        result: ConformalPredictionResult = {
            "prediction_set": (lower, upper),
            "point_prediction": np.array([2.0]),
            "coverage_guarantee": 0.9,
            "average_set_size": 1.0,
            "nonconformity_score": np.array([0.5]),
            "threshold": 0.5,
            "adaptive": False,
        }

        assert isinstance(result["prediction_set"], tuple)
        assert 0 <= result["coverage_guarantee"] <= 1
        assert result["average_set_size"] > 0

    def test_prediction_interval_coverage(self):
        """Test prediction interval contains point prediction."""
        point_pred = 2.0
        threshold = 0.5

        lower = point_pred - threshold
        upper = point_pred + threshold

        result: ConformalPredictionResult = {
            "prediction_set": (np.array([lower]), np.array([upper])),
            "point_prediction": np.array([point_pred]),
            "coverage_guarantee": 0.9,
            "average_set_size": 2 * threshold,
            "threshold": threshold,
            "adaptive": False,
        }

        # Point prediction should be in interval
        lb, ub = result["prediction_set"]
        assert lb[0] <= result["point_prediction"][0] <= ub[0]

    def test_prediction_batch(self):
        """Test batch prediction."""
        n_test = 10

        # Batch predictions
        lower = np.random.randn(n_test)
        upper = lower + np.random.rand(n_test)

        result: ConformalPredictionResult = {
            "prediction_set": (lower, upper),
            "point_prediction": (lower + upper) / 2,
            "coverage_guarantee": 0.95,
            "average_set_size": np.mean(upper - lower),
            "nonconformity_score": np.random.rand(n_test),
            "threshold": 0.3,
            "adaptive": False,
        }

        lb, ub = result["prediction_set"]
        assert lb.shape == (n_test,)
        assert ub.shape == (n_test,)


class TestAdaptiveConformalResult:
    """Test AdaptiveConformalResult TypedDict."""

    def test_adaptive_result_creation(self):
        """Test creating adaptive conformal result."""
        T = 100

        result: AdaptiveConformalResult = {
            "threshold": 0.5,
            "coverage_history": np.random.rand(T) * 0.1 + 0.85,
            "miscoverage_rate": 0.12,
            "target_alpha": 0.1,
            "adaptation_rate": 0.01,
            "effective_sample_size": 50,
        }

        assert result["threshold"] > 0
        assert 0 <= result["miscoverage_rate"] <= 1
        assert 0 <= result["target_alpha"] <= 1
        assert result["adaptation_rate"] > 0

    def test_adaptive_threshold_update(self):
        """Test adaptive threshold update rule."""
        # q_t = q_{t-1} + γ(α - err_t)
        q_prev = 0.5
        gamma = 0.01
        alpha = 0.1
        err = 0.15  # Miscoverage > target

        # Update
        q_new = q_prev + gamma * (alpha - err)

        result: AdaptiveConformalResult = {
            "threshold": q_new,
            "coverage_history": np.array([0.85]),
            "miscoverage_rate": err,
            "target_alpha": alpha,
            "adaptation_rate": gamma,
            "effective_sample_size": 100,
        }

        # Threshold should decrease when miscoverage > target
        assert result["threshold"] < q_prev

    def test_adaptive_coverage_convergence(self):
        """Test coverage converges to target."""
        # Set random seed for reproducibility
        np.random.seed(42)

        # Simulate adaptive process
        T = 200
        target_alpha = 0.1
        target_coverage = 1 - target_alpha  # 0.9

        # Coverage should converge to 1 - α
        # Exponential convergence with noise
        coverage_hist = 0.9 + 0.1 * np.exp(-np.arange(T) / 50)
        coverage_hist += np.random.randn(T) * 0.02

        # Clip to valid range [0, 1]
        coverage_hist = np.clip(coverage_hist, 0, 1)

        result: AdaptiveConformalResult = {
            "threshold": 0.5,
            "coverage_history": coverage_hist,
            "miscoverage_rate": 1 - coverage_hist[-1],
            "target_alpha": target_alpha,
            "adaptation_rate": 0.01,
            "effective_sample_size": 100,
        }

        # Test 1: Statistical test - is final coverage significantly different from target?
        final_coverage = coverage_hist[-1]
        n_samples = result["effective_sample_size"]

        # Simulate coverage as binomial: n_covered ~ Binomial(n_total, p=final_coverage)
        # Under H0: true coverage = target_coverage
        # Under H1: true coverage ≠ target_coverage
        n_covered = int(final_coverage * n_samples)
        n_total = n_samples

        from scipy.stats import binomtest

        test_result = binomtest(n_covered, n_total, target_coverage, alternative="two-sided")

        # p > 0.05 means we cannot reject H0 (coverage matches target)
        # This is what we want - coverage is statistically indistinguishable from target
        assert test_result.pvalue > 0.05, (
            f"Coverage {final_coverage:.4f} significantly different from target {target_coverage} "
            f"(p={test_result.pvalue:.4f})"
        )

        # Test 2: Verify convergence (coverage error decreases over time)
        # Compare early vs late coverage error
        early_error = abs(coverage_hist[:50].mean() - target_coverage)
        late_error = abs(coverage_hist[-50:].mean() - target_coverage)

        assert (
            late_error < early_error
        ), f"Coverage should improve over time: early_error={early_error:.4f}, late_error={late_error:.4f}"

        # Test 3: Final coverage should be in reasonable range
        assert (
            0.7 < final_coverage < 1.0
        ), f"Final coverage {final_coverage:.4f} outside reasonable range [0.7, 1.0]"

        # Test 4: Convergence trend - fit exponential and verify decay
        # coverage_error(t) should decay exponentially
        coverage_error = np.abs(coverage_hist - target_coverage)

        # Check that late-stage error is small
        late_stage_error = coverage_error[-50:].mean()
        assert late_stage_error < 0.05, f"Late-stage error {late_stage_error:.4f} should be < 0.05"


class TestPracticalUseCases:
    """Test realistic usage patterns."""

    def test_regression_conformal_prediction(self):
        """Test conformal prediction for regression."""
        # Calibration data
        n_cal = 100
        y_cal = np.random.randn(n_cal)
        y_pred_cal = y_cal + np.random.randn(n_cal) * 0.1

        # Nonconformity scores (absolute residuals)
        scores = np.abs(y_cal - y_pred_cal)

        # Calibrate
        alpha = 0.1
        q = np.quantile(scores, (1 - alpha) * (1 + 1 / n_cal))

        calib_result: ConformalCalibrationResult = {
            "quantile": q,
            "alpha": alpha,
            "empirical_coverage": np.mean(scores <= q),
            "calibration_scores": scores,
            "n_calibration": n_cal,
            "prediction_set_type": "interval",
        }

        # Predict on test point
        y_pred_test = 1.5
        lower = y_pred_test - q
        upper = y_pred_test + q

        pred_result: ConformalPredictionResult = {
            "prediction_set": (np.array([lower]), np.array([upper])),
            "point_prediction": np.array([y_pred_test]),
            "coverage_guarantee": 1 - alpha,
            "average_set_size": 2 * q,
            "threshold": q,
            "adaptive": False,
        }

        # Verify
        assert calib_result["empirical_coverage"] >= 1 - alpha - 0.05
        assert pred_result["coverage_guarantee"] == 0.9

    def test_time_series_adaptive_conformal(self):
        """Test adaptive conformal for time series."""
        # Initialize
        gamma = 0.01
        alpha = 0.1
        q_0 = 1.0

        # Simulate online updates
        T = 50
        q_t = q_0
        coverage_hist = []

        for t in range(T):
            # Simulate coverage error
            err_t = alpha + 0.05 * np.sin(t / 10)  # Oscillating

            # Update threshold
            q_t = q_t + gamma * (alpha - err_t)

            # Track coverage
            coverage_hist.append(1 - err_t)

        result: AdaptiveConformalResult = {
            "threshold": q_t,
            "coverage_history": np.array(coverage_hist),
            "miscoverage_rate": alpha,
            "target_alpha": alpha,
            "adaptation_rate": gamma,
            "effective_sample_size": 100,
        }

        # Threshold should have adapted
        assert abs(result["threshold"] - q_0) > 0


class TestNumericalProperties:
    """Test numerical properties of results."""

    def test_quantile_bounds(self):
        """Test quantile is in valid range."""
        scores = np.random.rand(100)
        q = np.quantile(scores, 0.9)

        result: ConformalCalibrationResult = {
            "quantile": q,
            "alpha": 0.1,
            "empirical_coverage": 0.9,
            "calibration_scores": scores,
            "n_calibration": 100,
            "prediction_set_type": "interval",
        }

        # Quantile should be in score range
        assert np.min(scores) <= result["quantile"] <= np.max(scores)

    def test_coverage_guarantee_bounds(self):
        """Test coverage is in [0, 1]."""
        result: ConformalPredictionResult = {
            "prediction_set": (np.array([1.0]), np.array([2.0])),
            "point_prediction": np.array([1.5]),
            "coverage_guarantee": 0.9,
            "average_set_size": 1.0,
            "threshold": 0.5,
        }

        assert 0 <= result["coverage_guarantee"] <= 1

    def test_adaptive_rate_positive(self):
        """Test adaptation rate is positive."""
        result: AdaptiveConformalResult = {
            "threshold": 0.5,
            "coverage_history": np.array([0.9]),
            "miscoverage_rate": 0.1,
            "target_alpha": 0.1,
            "adaptation_rate": 0.01,
            "effective_sample_size": 100,
        }

        assert result["adaptation_rate"] > 0


class TestCoverageGuarantees:
    """Test coverage guarantee properties."""

    def test_split_conformal_coverage(self):
        """Test split conformal coverage guarantee."""
        # Simulate calibration
        n_cal = 200
        scores = np.random.exponential(1.0, n_cal)
        alpha = 0.1

        # Quantile for coverage guarantee
        q_level = (1 - alpha) * (1 + 1 / n_cal)
        q = np.quantile(scores, q_level)

        # Empirical coverage
        empirical_cov = np.mean(scores <= q)

        result: ConformalCalibrationResult = {
            "quantile": q,
            "alpha": alpha,
            "empirical_coverage": empirical_cov,
            "calibration_scores": scores,
            "n_calibration": n_cal,
            "prediction_set_type": "interval",
        }

        # Guarantee: coverage ≥ 1 - α (with high probability)
        assert result["empirical_coverage"] >= 1 - alpha - 0.02

    def test_adaptive_long_run_coverage(self):
        """Test adaptive conformal converges to target."""
        # Long simulation
        T = 500
        alpha = 0.1
        gamma = 0.005

        # Coverage should converge
        coverage = np.zeros(T)
        q = 1.0

        for t in range(T):
            # Random coverage error
            err = alpha + np.random.randn() * 0.05
            q = q + gamma * (alpha - err)
            coverage[t] = 1 - err

        result: AdaptiveConformalResult = {
            "threshold": q,
            "coverage_history": coverage,
            "miscoverage_rate": alpha,
            "target_alpha": alpha,
            "adaptation_rate": gamma,
            "effective_sample_size": 100,
        }

        # Average coverage in second half should be near target
        avg_coverage = np.mean(result["coverage_history"][T // 2 :])
        assert abs(avg_coverage - (1 - alpha)) < 0.1


class TestDocumentationExamples:
    """Test that documentation examples work."""

    def test_calibration_example(self):
        """Test ConformalCalibrationResult example from docstring."""
        residuals = np.abs(np.random.randn(100) * 0.5)
        alpha = 0.1

        q = np.quantile(residuals, (1 - alpha) * (1 + 1 / len(residuals)))

        result: ConformalCalibrationResult = {
            "quantile": q,
            "alpha": alpha,
            "empirical_coverage": np.mean(residuals <= q),
            "calibration_scores": residuals,
            "n_calibration": len(residuals),
            "prediction_set_type": "interval",
        }

        assert result["quantile"] > 0
        assert result["empirical_coverage"] >= 0.85

    def test_prediction_example(self):
        """Test ConformalPredictionResult example structure."""
        result: ConformalPredictionResult = {
            "prediction_set": (np.array([1.5]), np.array([2.5])),
            "point_prediction": np.array([2.0]),
            "coverage_guarantee": 0.9,
            "average_set_size": 1.0,
            "threshold": 0.5,
            "adaptive": False,
        }

        assert isinstance(result["prediction_set"], tuple)
        assert result["coverage_guarantee"] == 0.9


class TestFieldPresence:
    """Test that all fields are accessible."""

    def test_calibration_has_required_fields(self):
        """Test ConformalCalibrationResult has core fields."""
        result: ConformalCalibrationResult = {
            "quantile": 0.5,
            "alpha": 0.1,
            "empirical_coverage": 0.9,
            "calibration_scores": np.random.rand(100),
            "n_calibration": 100,
            "prediction_set_type": "interval",
        }

        assert "quantile" in result
        assert "alpha" in result
        assert "n_calibration" in result

    def test_adaptive_has_required_fields(self):
        """Test AdaptiveConformalResult has core fields."""
        result: AdaptiveConformalResult = {
            "threshold": 0.5,
            "coverage_history": np.array([0.9]),
            "miscoverage_rate": 0.1,
            "target_alpha": 0.1,
            "adaptation_rate": 0.01,
            "effective_sample_size": 100,
        }

        assert "threshold" in result
        assert "coverage_history" in result
        assert "adaptation_rate" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
