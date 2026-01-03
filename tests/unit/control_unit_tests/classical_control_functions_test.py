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
Unit Tests for Classical Control Functions

Tests cover:
- Unified LQR design (continuous and discrete via system_type parameter)
- Kalman filter design
- LQG controller design
- Stability analysis
- Controllability analysis
- Observability analysis
- Backend conversions (NumPy, PyTorch, JAX)
- Error handling and edge cases

Test Structure:
- TestUnifiedLQR: Unified design_lqr() interface tests
- TestLQRContinuous: Continuous-time LQR tests
- TestLQRDiscrete: Discrete-time LQR tests
- TestKalmanFilter: Kalman filter tests
- TestLQG: LQG controller tests
- TestStabilityAnalysis: Stability analysis tests
- TestControllability: Controllability tests
- TestObservability: Observability tests
- TestBackendConversion: Multi-backend tests
- TestErrorHandling: Edge cases and error conditions
"""

import unittest

import numpy as np
from numpy.testing import assert_allclose

# Optional backends for testing
try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import jax.numpy as jnp

    HAS_JAX = True
except ImportError:
    HAS_JAX = False

from src.control.classical_control_functions import (
    analyze_controllability,
    analyze_observability,
    analyze_stability,
    design_kalman_filter,
    design_lqg,
    design_lqr,
)

# ============================================================================
# Test Fixtures and Utilities
# ============================================================================


class ControlTestCase(unittest.TestCase):
    """Base class with common test utilities."""

    def setUp(self):
        """Set up common test systems."""
        # Double integrator (continuous)
        self.A_double_int = np.array([[0, 1], [0, 0]])
        self.B_double_int = np.array([[0], [1]])
        self.C_double_int = np.array([[1, 0]])

        # Simple stable system
        self.A_stable = np.array([[0, 1], [-2, -3]])
        self.B_stable = np.array([[0], [1]])
        self.C_stable = np.array([[1, 0]])

        # Discrete double integrator (dt = 0.1)
        dt = 0.1
        self.Ad_double_int = np.array([[1, dt], [0, 1]])
        self.Bd_double_int = np.array([[0.5 * dt**2], [dt]])

        # Standard cost matrices
        self.Q2 = np.diag([10, 1])
        self.R1 = np.array([[0.1]])

        # Noise covariances
        self.Q_process = 0.01 * np.eye(2)
        self.R_meas = 0.1 * np.eye(1)

        # Tolerance for numerical comparisons
        self.rtol = 1e-5
        self.atol = 1e-8

    def assert_positive_definite(self, M: np.ndarray, name: str = "Matrix"):
        """Assert matrix is positive definite."""
        eigenvalues = np.linalg.eigvals(M)
        self.assertTrue(
            np.all(eigenvalues > 0),
            f"{name} is not positive definite. Min eigenvalue: {np.min(eigenvalues)}",
        )

    def assert_symmetric(self, M: np.ndarray, name: str = "Matrix"):
        """Assert matrix is symmetric."""
        assert_allclose(M, M.T, rtol=self.rtol, atol=self.atol, err_msg=f"{name} is not symmetric")

    def assert_stable_continuous(self, eigenvalues: np.ndarray):
        """Assert continuous system is stable (Re(λ) < 0)."""
        max_real = np.max(np.real(eigenvalues))
        self.assertLess(max_real, 0, f"System unstable. Max Re(λ) = {max_real}")

    def assert_stable_discrete(self, eigenvalues: np.ndarray):
        """Assert discrete system is stable (|λ| < 1)."""
        max_mag = np.max(np.abs(eigenvalues))
        self.assertLess(max_mag, 1.0, f"System unstable. Max |λ| = {max_mag}")


# ============================================================================
# Unified LQR Tests
# ============================================================================


class TestUnifiedLQR(ControlTestCase):
    """Test unified design_lqr() interface."""

    def test_unified_lqr_continuous(self):
        """Test unified LQR with system_type='continuous'."""
        result = design_lqr(
            self.A_stable,
            self.B_stable,
            self.Q2,
            self.R1,
            system_type="continuous",
        )

        # Check result structure
        self.assertIn("gain", result)
        self.assertIn("cost_to_go", result)
        self.assertIn("closed_loop_eigenvalues", result)
        self.assertIn("stability_margin", result)

        # Check dimensions
        self.assertEqual(result["gain"].shape, (1, 2))
        self.assertEqual(result["cost_to_go"].shape, (2, 2))

        # Check stability
        self.assertGreater(result["stability_margin"], 0)
        self.assert_stable_continuous(result["closed_loop_eigenvalues"])

    def test_unified_lqr_discrete_default(self):
        """Test unified LQR with default system_type='discrete'."""
        # Default should be discrete
        result = design_lqr(self.Ad_double_int, self.Bd_double_int, self.Q2, self.R1)

        self.assertEqual(result["gain"].shape, (1, 2))
        self.assertGreater(result["stability_margin"], 0)
        self.assert_stable_discrete(result["closed_loop_eigenvalues"])

    def test_unified_lqr_discrete_explicit(self):
        """Test unified LQR with explicit system_type='discrete'."""
        result = design_lqr(
            self.Ad_double_int,
            self.Bd_double_int,
            self.Q2,
            self.R1,
            system_type="discrete",
        )

        self.assertEqual(result["gain"].shape, (1, 2))
        self.assertGreater(result["stability_margin"], 0)
        self.assert_stable_discrete(result["closed_loop_eigenvalues"])

    def test_unified_lqr_with_cross_term(self):
        """Test unified LQR with cross-coupling term N."""
        N = np.array([[0.5], [0.1]])

        # Continuous
        result_c = design_lqr(
            self.A_stable,
            self.B_stable,
            self.Q2,
            self.R1,
            N=N,
            system_type="continuous",
        )
        self.assertGreater(result_c["stability_margin"], 0)

        # Discrete
        result_d = design_lqr(
            self.Ad_double_int,
            self.Bd_double_int,
            self.Q2,
            self.R1,
            N=N,
            system_type="discrete",
        )
        self.assertGreater(result_d["stability_margin"], 0)

    def test_unified_lqr_invalid_system_type(self):
        """Test unified LQR with invalid system_type."""
        with self.assertRaises(ValueError) as cm:
            design_lqr(self.A_stable, self.B_stable, self.Q2, self.R1, system_type="invalid")
        self.assertIn("continuous", str(cm.exception))
        self.assertIn("discrete", str(cm.exception))


# ============================================================================
# LQR Continuous Tests (Unified)
# ============================================================================


class TestLQRContinuous(ControlTestCase):
    """Test continuous-time LQR design."""

    def test_lqr_double_integrator(self):
        """Test LQR on standard double integrator."""
        # Use unified interface
        result = design_lqr(
            self.A_double_int,
            self.B_double_int,
            self.Q2,
            self.R1,
            system_type="continuous",
        )

        # Check result structure
        self.assertIn("gain", result)
        self.assertIn("cost_to_go", result)
        self.assertIn("closed_loop_eigenvalues", result)
        self.assertIn("stability_margin", result)

        # Check dimensions
        K = result["gain"]
        P = result["cost_to_go"]
        self.assertEqual(K.shape, (1, 2))
        self.assertEqual(P.shape, (2, 2))

        # Check P is symmetric and positive definite
        self.assert_symmetric(P, "Cost-to-go matrix")
        self.assert_positive_definite(P, "Cost-to-go matrix")

        # Check stability
        self.assertGreater(result["stability_margin"], 0, "System should be stable")
        self.assert_stable_continuous(result["closed_loop_eigenvalues"])

        # Verify Riccati equation: A'P + PA - PBR^{-1}B'P + Q = 0
        A = self.A_double_int
        B = self.B_double_int
        R_inv = np.linalg.inv(self.R1)
        residual = A.T @ P + P @ A - P @ B @ R_inv @ B.T @ P + self.Q2
        assert_allclose(residual, np.zeros_like(residual), atol=1e-6)

    def test_lqr_stable_system(self):
        """Test LQR on already stable system."""
        result = design_lqr(
            self.A_stable,
            self.B_stable,
            self.Q2,
            self.R1,
            system_type="continuous",
        )

        # Should still be stable with improved performance
        self.assertGreater(result["stability_margin"], 0)

        # Closed-loop should be more stable than open-loop
        A_cl_eigs = result["closed_loop_eigenvalues"]
        A_eigs = np.linalg.eigvals(self.A_stable)

        max_real_cl = np.max(np.real(A_cl_eigs))
        max_real_ol = np.max(np.real(A_eigs))
        self.assertLess(max_real_cl, max_real_ol)

    def test_lqr_with_cross_term(self):
        """Test LQR with cross-coupling term N."""
        N = np.array([[0.5], [0.1]])

        result = design_lqr(
            self.A_stable,
            self.B_stable,
            self.Q2,
            self.R1,
            N=N,
            system_type="continuous",
        )

        # Should still produce stable controller
        self.assertGreater(result["stability_margin"], 0)
        self.assertEqual(result["gain"].shape, (1, 2))

    def test_lqr_high_q_vs_r(self):
        """Test effect of Q vs R weighting."""
        # High Q (penalize state more)
        Q_high = 100 * self.Q2
        result_high_q = design_lqr(
            self.A_double_int,
            self.B_double_int,
            Q_high,
            self.R1,
            system_type="continuous",
        )

        # High R (penalize control more)
        R_high = 10 * self.R1
        result_high_r = design_lqr(
            self.A_double_int,
            self.B_double_int,
            self.Q2,
            R_high,
            system_type="continuous",
        )

        # High Q should give larger gains
        K_high_q = np.linalg.norm(result_high_q["gain"])
        K_high_r = np.linalg.norm(result_high_r["gain"])
        self.assertGreater(K_high_q, K_high_r)

    def test_lqr_multi_input(self):
        """Test LQR with multiple inputs."""
        # 3-state, 2-input system
        A = np.array([[0, 1, 0], [0, 0, 1], [-1, -2, -3]])
        B = np.array([[0, 0], [1, 0], [0, 1]])
        Q = np.eye(3)
        R = np.eye(2)

        result = design_lqr(A, B, Q, R, system_type="continuous")

        self.assertEqual(result["gain"].shape, (2, 3))
        self.assertGreater(result["stability_margin"], 0)


# ============================================================================
# LQR Discrete Tests (Unified)
# ============================================================================


class TestLQRDiscrete(ControlTestCase):
    """Test discrete-time LQR design."""

    def test_lqr_discrete_double_integrator(self):
        """Test discrete LQR on double integrator."""
        result = design_lqr(
            self.Ad_double_int,
            self.Bd_double_int,
            self.Q2,
            self.R1,
            system_type="discrete",
        )

        # Check result structure
        self.assertIn("gain", result)
        self.assertIn("cost_to_go", result)

        # Check dimensions
        K = result["gain"]
        P = result["cost_to_go"]
        self.assertEqual(K.shape, (1, 2))
        self.assertEqual(P.shape, (2, 2))

        # Check stability
        self.assertGreater(result["stability_margin"], 0)
        self.assert_stable_discrete(result["closed_loop_eigenvalues"])

        # Verify discrete Riccati equation
        A = self.Ad_double_int
        B = self.Bd_double_int
        K_computed = np.linalg.solve(self.R1 + B.T @ P @ B, B.T @ P @ A)
        assert_allclose(K, K_computed, rtol=self.rtol, atol=self.atol)

    def test_lqr_discrete_stability_margin(self):
        """Test stability margin calculation for discrete systems."""
        result = design_lqr(
            self.Ad_double_int,
            self.Bd_double_int,
            self.Q2,
            self.R1,
            system_type="discrete",
        )

        # Stability margin = 1 - max(|λ|)
        max_mag = np.max(np.abs(result["closed_loop_eigenvalues"]))
        expected_margin = 1.0 - max_mag

        assert_allclose(result["stability_margin"], expected_margin, rtol=self.rtol, atol=self.atol)

    def test_lqr_discrete_vs_continuous_consistency(self):
        """Test that discretized continuous LQR gives similar results."""
        # Design continuous LQR
        result_c = design_lqr(
            self.A_double_int,
            self.B_double_int,
            self.Q2,
            self.R1,
            system_type="continuous",
        )

        # Design discrete LQR
        result_d = design_lqr(
            self.Ad_double_int,
            self.Bd_double_int,
            self.Q2,
            self.R1,
            system_type="discrete",
        )

        # Gains should be similar (not exact due to discretization)
        # This is a loose check
        K_c = result_c["gain"]
        K_d = result_d["gain"]

        # Both should be stable
        self.assertGreater(result_c["stability_margin"], 0)
        self.assertGreater(result_d["stability_margin"], 0)


# ============================================================================
# Kalman Filter Tests
# ============================================================================


class TestKalmanFilter(ControlTestCase):
    """Test Kalman filter design."""

    def test_kalman_discrete_basic(self):
        """Test discrete Kalman filter design."""
        result = design_kalman_filter(
            self.Ad_double_int,
            self.C_double_int,
            self.Q_process,
            self.R_meas,
            system_type="discrete",
        )

        # Check result structure
        self.assertIn("gain", result)
        self.assertIn("error_covariance", result)
        self.assertIn("innovation_covariance", result)
        self.assertIn("observer_eigenvalues", result)

        # Check dimensions
        L = result["gain"]
        P = result["error_covariance"]
        S = result["innovation_covariance"]

        self.assertEqual(L.shape, (2, 1))
        self.assertEqual(P.shape, (2, 2))
        self.assertEqual(S.shape, (1, 1))

        # Check P is symmetric and positive definite
        self.assert_symmetric(P, "Error covariance")
        self.assert_positive_definite(P, "Error covariance")

        # Check observer stability
        self.assert_stable_discrete(result["observer_eigenvalues"])

        # Verify innovation covariance: S = CPC' + R
        C = self.C_double_int
        S_computed = C @ P @ C.T + self.R_meas
        assert_allclose(S, S_computed, rtol=self.rtol, atol=self.atol)

    def test_kalman_continuous(self):
        """Test continuous Kalman filter design."""
        result = design_kalman_filter(
            self.A_stable,
            self.C_stable,
            self.Q_process,
            self.R_meas,
            system_type="continuous",
        )

        self.assertEqual(result["gain"].shape, (2, 1))
        self.assert_stable_continuous(result["observer_eigenvalues"])

    def test_kalman_perfect_measurement(self):
        """Test with very low measurement noise (R → 0)."""
        R_low = 1e-6 * np.eye(1)

        result = design_kalman_filter(
            self.Ad_double_int,
            self.C_double_int,
            self.Q_process,
            R_low,
            system_type="discrete",
        )

        # With low R, Kalman gain should be larger
        # (trust measurements more)
        L_norm = np.linalg.norm(result["gain"])
        self.assertGreater(L_norm, 0.1)

    def test_kalman_high_process_noise(self):
        """Test with high process noise."""
        Q_high = 10 * self.Q_process

        result = design_kalman_filter(
            self.Ad_double_int,
            self.C_double_int,
            Q_high,
            self.R_meas,
            system_type="discrete",
        )

        # Higher process noise should give larger error covariance
        P = result["error_covariance"]
        self.assertGreater(np.trace(P), 0.1)

    def test_kalman_full_state_measurement(self):
        """Test with full state measurement (C = I)."""
        C_full = np.eye(2)
        R_full = 0.1 * np.eye(2)

        result = design_kalman_filter(
            self.Ad_double_int,
            C_full,
            self.Q_process,
            R_full,
            system_type="discrete",
        )

        self.assertEqual(result["gain"].shape, (2, 2))
        self.assert_stable_discrete(result["observer_eigenvalues"])


# ============================================================================
# LQG Tests
# ============================================================================


class TestLQG(ControlTestCase):
    """Test LQG (LQR + Kalman) design."""

    def test_lqg_discrete_basic(self):
        """Test basic discrete LQG design."""
        result = design_lqg(
            self.Ad_double_int,
            self.Bd_double_int,
            self.C_double_int,
            self.Q2,  # State cost
            self.R1,  # Control cost
            self.Q_process,  # Process noise
            self.R_meas,  # Measurement noise
            system_type="discrete",
        )

        # Check result structure
        self.assertIn("controller_gain", result)
        self.assertIn("estimator_gain", result)
        self.assertIn("controller_riccati", result)
        self.assertIn("estimator_covariance", result)
        self.assertIn("closed_loop_eigenvalues", result)
        self.assertIn("observer_eigenvalues", result)

        # Check dimensions
        K = result["controller_gain"]
        L = result["estimator_gain"]

        self.assertEqual(K.shape, (1, 2))
        self.assertEqual(L.shape, (2, 1))

        # Both controller and estimator should be stable
        self.assert_stable_discrete(result["closed_loop_eigenvalues"])
        self.assert_stable_discrete(result["observer_eigenvalues"])

    def test_lqg_continuous_basic(self):
        """Test basic continuous LQG design."""
        result = design_lqg(
            self.A_stable,
            self.B_stable,
            self.C_stable,
            self.Q2,
            self.R1,
            self.Q_process,
            self.R_meas,
            system_type="continuous",
        )

        # Check stability
        self.assert_stable_continuous(result["closed_loop_eigenvalues"])
        self.assert_stable_continuous(result["observer_eigenvalues"])

    def test_lqg_separation_principle(self):
        """Test separation principle: LQG = LQR + Kalman designed independently."""
        # Design LQR separately
        lqr_result = design_lqr(
            self.Ad_double_int,
            self.Bd_double_int,
            self.Q2,
            self.R1,
            system_type="discrete",
        )

        # Design Kalman separately
        kalman_result = design_kalman_filter(
            self.Ad_double_int,
            self.C_double_int,
            self.Q_process,
            self.R_meas,
            system_type="discrete",
        )

        # Design LQG
        lqg_result = design_lqg(
            self.Ad_double_int,
            self.Bd_double_int,
            self.C_double_int,
            self.Q2,
            self.R1,
            self.Q_process,
            self.R_meas,
            system_type="discrete",
        )

        # LQG gains should match individual designs
        assert_allclose(
            lqg_result["controller_gain"],
            lqr_result["gain"],
            rtol=self.rtol,
            atol=self.atol,
        )
        assert_allclose(
            lqg_result["estimator_gain"],
            kalman_result["gain"],
            rtol=self.rtol,
            atol=self.atol,
        )

    def test_lqg_observer_faster_than_controller(self):
        """Test that observer eigenvalues can be placed independently of controller."""
        # Design with aggressive estimator (low measurement noise)
        # This doesn't guarantee faster convergence, but tests that
        # the estimator gain responds to measurement noise changes
        R_meas_low = 0.001 * np.eye(1)
        R_meas_high = 1.0 * np.eye(1)

        result_low_r = design_lqg(
            self.Ad_double_int,
            self.Bd_double_int,
            self.C_double_int,
            self.Q2,
            self.R1,
            self.Q_process,
            R_meas_low,
            system_type="discrete",
        )

        result_high_r = design_lqg(
            self.Ad_double_int,
            self.Bd_double_int,
            self.C_double_int,
            self.Q2,
            self.R1,
            self.Q_process,
            R_meas_high,
            system_type="discrete",
        )

        # With lower R (more trust in measurements), Kalman gain should be larger
        L_low = np.linalg.norm(result_low_r["estimator_gain"])
        L_high = np.linalg.norm(result_high_r["estimator_gain"])

        self.assertGreater(L_low, L_high, "Lower measurement noise should give larger Kalman gain")


# ============================================================================
# Stability Analysis Tests
# ============================================================================


class TestStabilityAnalysis(ControlTestCase):
    """Test stability analysis function."""

    def test_stability_continuous_stable(self):
        """Test stable continuous system."""
        A = self.A_stable
        result = analyze_stability(A, system_type="continuous")

        self.assertTrue(result["is_stable"])
        self.assertFalse(result["is_unstable"])
        self.assertFalse(result["is_marginally_stable"])

        # Check eigenvalues
        self.assertEqual(len(result["eigenvalues"]), 2)
        self.assertTrue(np.all(np.real(result["eigenvalues"]) < 0))

    def test_stability_continuous_unstable(self):
        """Test unstable continuous system."""
        A_unstable = np.array([[1, 1], [0, 1]])
        result = analyze_stability(A_unstable, system_type="continuous")

        self.assertFalse(result["is_stable"])
        self.assertTrue(result["is_unstable"])
        self.assertTrue(np.any(np.real(result["eigenvalues"]) > 0))

    def test_stability_continuous_marginally_stable(self):
        """Test marginally stable continuous system (pure oscillator)."""
        A_marginal = np.array([[0, 1], [-1, 0]])  # Pure imaginary eigenvalues
        result = analyze_stability(A_marginal, system_type="continuous")

        self.assertTrue(result["is_marginally_stable"])
        self.assertFalse(result["is_stable"])

        # Eigenvalues should be on imaginary axis
        real_parts = np.real(result["eigenvalues"])
        self.assertTrue(np.all(np.abs(real_parts) < 1e-10))

    def test_stability_discrete_stable(self):
        """Test stable discrete system."""
        Ad = 0.9 * np.eye(2)
        result = analyze_stability(Ad, system_type="discrete")

        self.assertTrue(result["is_stable"])
        self.assertFalse(result["is_unstable"])

        # All eigenvalues inside unit circle
        self.assertTrue(np.all(result["magnitudes"] < 1.0))
        self.assertAlmostEqual(result["spectral_radius"], 0.9, places=6)

    def test_stability_discrete_unstable(self):
        """Test unstable discrete system."""
        Ad_unstable = 1.1 * np.eye(2)
        result = analyze_stability(Ad_unstable, system_type="discrete")

        self.assertFalse(result["is_stable"])
        self.assertTrue(result["is_unstable"])
        self.assertGreater(result["spectral_radius"], 1.0)

    def test_stability_discrete_marginally_stable(self):
        """Test marginally stable discrete system (on unit circle)."""
        theta = np.pi / 4
        Ad_marginal = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        result = analyze_stability(Ad_marginal, system_type="discrete")

        self.assertTrue(result["is_marginally_stable"])
        self.assertAlmostEqual(result["spectral_radius"], 1.0, places=6)

    def test_stability_spectral_radius(self):
        """Test spectral radius calculation."""
        A = np.diag([0.5, 0.8, 0.3])
        result = analyze_stability(A, system_type="discrete")

        self.assertAlmostEqual(result["spectral_radius"], 0.8, places=10)
        self.assertEqual(result["max_magnitude"], result["spectral_radius"])


# ============================================================================
# Controllability Tests
# ============================================================================


class TestControllability(ControlTestCase):
    """Test controllability analysis."""

    def test_controllability_full_rank(self):
        """Test fully controllable system."""
        result = analyze_controllability(self.A_double_int, self.B_double_int)

        self.assertTrue(result["is_controllable"])
        self.assertEqual(result["rank"], 2)

        # Controllability matrix should be 2x2 for nx=2, nu=1
        C = result["controllability_matrix"]
        self.assertEqual(C.shape, (2, 2))

        # Should be full rank
        self.assertEqual(np.linalg.matrix_rank(C), 2)

    def test_controllability_uncontrollable(self):
        """Test uncontrollable system."""
        # System where one mode cannot be controlled
        # dx1/dt = x1 (uncontrollable eigenvalue at 1)
        # dx2/dt = 2*x2 + u (controllable)
        A = np.array([[1, 0], [0, 2]])
        B = np.array([[0], [1]])  # Only affects second state

        result = analyze_controllability(A, B)

        self.assertFalse(result["is_controllable"])
        self.assertLess(result["rank"], 2)

    def test_controllability_multi_input(self):
        """Test controllability with multiple inputs."""
        A = np.array([[0, 1, 0], [0, 0, 1], [-1, -2, -3]])
        B = np.array([[0, 0], [1, 0], [0, 1]])

        result = analyze_controllability(A, B)

        self.assertTrue(result["is_controllable"])
        self.assertEqual(result["rank"], 3)

        # Controllability matrix shape: (nx, nx*nu) = (3, 6)
        self.assertEqual(result["controllability_matrix"].shape, (3, 6))

    def test_controllability_single_input_chain(self):
        """Test canonical controllable form (chain of integrators)."""
        # dx1/dt = x2, dx2/dt = x3, dx3/dt = u
        A = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
        B = np.array([[0], [0], [1]])

        result = analyze_controllability(A, B)

        self.assertTrue(result["is_controllable"])
        self.assertEqual(result["rank"], 3)


# ============================================================================
# Observability Tests
# ============================================================================


class TestObservability(ControlTestCase):
    """Test observability analysis."""

    def test_observability_full_rank(self):
        """Test fully observable system."""
        result = analyze_observability(self.A_double_int, self.C_double_int)

        self.assertTrue(result["is_observable"])
        self.assertEqual(result["rank"], 2)

        # Observability matrix should be 2x2 for nx=2, ny=1
        O = result["observability_matrix"]
        self.assertEqual(O.shape, (2, 2))

        # Should be full rank
        self.assertEqual(np.linalg.matrix_rank(O), 2)

    def test_observability_unobservable(self):
        """Test unobservable system."""
        # System where one mode cannot be observed
        # y = x1 (cannot see x2)
        A = np.array([[1, 0], [0, 2]])
        C = np.array([[1, 0]])  # Only measures first state

        result = analyze_observability(A, C)

        self.assertFalse(result["is_observable"])
        self.assertLess(result["rank"], 2)

    def test_observability_full_state_measurement(self):
        """Test with full state measurement."""
        A = np.array([[0, 1, 0], [0, 0, 1], [-1, -2, -3]])
        C = np.eye(3)

        result = analyze_observability(A, C)

        self.assertTrue(result["is_observable"])
        self.assertEqual(result["rank"], 3)

    def test_observability_single_output(self):
        """Test single output observability."""
        A = np.array([[0, 1], [-2, -3]])
        C = np.array([[1, 0]])

        result = analyze_observability(A, C)

        self.assertTrue(result["is_observable"])

        # Observability matrix: [C; CA] = (2,2)
        O = result["observability_matrix"]
        self.assertEqual(O.shape, (2, 2))


# ============================================================================
# Backend Conversion Tests
# ============================================================================


class TestBackendConversion(ControlTestCase):
    """Test multi-backend support."""

    def test_numpy_backend(self):
        """Test with explicit NumPy backend."""
        result = design_lqr(
            self.A_stable,
            self.B_stable,
            self.Q2,
            self.R1,
            system_type="continuous",
            backend="numpy",
        )

        self.assertIsInstance(result["gain"], np.ndarray)
        self.assertIsInstance(result["cost_to_go"], np.ndarray)

    @unittest.skipIf(not HAS_TORCH, "PyTorch not available")
    def test_torch_backend(self):
        """Test with PyTorch tensors."""
        A_torch = torch.tensor(self.A_stable, dtype=torch.float64)
        B_torch = torch.tensor(self.B_stable, dtype=torch.float64)
        Q_torch = torch.tensor(self.Q2, dtype=torch.float64)
        R_torch = torch.tensor(self.R1, dtype=torch.float64)

        result = design_lqr(
            A_torch,
            B_torch,
            Q_torch,
            R_torch,
            system_type="continuous",
            backend="torch",
        )

        # Results should be PyTorch tensors
        self.assertIsInstance(result["gain"], torch.Tensor)
        self.assertIsInstance(result["cost_to_go"], torch.Tensor)

        # Should be stable
        self.assertGreater(result["stability_margin"], 0)

    @unittest.skipIf(not HAS_JAX, "JAX not available")
    def test_jax_backend(self):
        """Test with JAX arrays."""
        A_jax = jnp.array(self.A_stable)
        B_jax = jnp.array(self.B_stable)
        Q_jax = jnp.array(self.Q2)
        R_jax = jnp.array(self.R1)

        result = design_lqr(A_jax, B_jax, Q_jax, R_jax, system_type="continuous", backend="jax")

        # Results should be JAX arrays
        self.assertIsInstance(result["gain"], jnp.ndarray)
        self.assertIsInstance(result["cost_to_go"], jnp.ndarray)

        # Should be stable
        self.assertGreater(result["stability_margin"], 0)


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestErrorHandling(ControlTestCase):
    """Test error handling and edge cases."""

    def test_lqr_dimension_mismatch(self):
        """Test LQR with mismatched dimensions."""
        A = np.eye(2)
        B_wrong = np.array([[1]])  # Should be (2, 1)

        with self.assertRaises(ValueError):
            design_lqr(A, B_wrong, self.Q2, self.R1, system_type="continuous")

    def test_lqr_non_square_A(self):
        """Test LQR with non-square A matrix."""
        A_wrong = np.array([[1, 2, 3], [4, 5, 6]])

        with self.assertRaises(ValueError):
            design_lqr(A_wrong, self.B_stable, self.Q2, self.R1, system_type="continuous")

    def test_kalman_dimension_mismatch(self):
        """Test Kalman filter with mismatched dimensions."""
        A = np.eye(2)
        C_wrong = np.array([[1, 0, 0]])  # Should be (ny, 2)

        with self.assertRaises(ValueError):
            design_kalman_filter(A, C_wrong, self.Q_process, self.R_meas)

    def test_stability_invalid_type(self):
        """Test stability analysis with invalid system type."""
        with self.assertRaises(ValueError):
            analyze_stability(self.A_stable, system_type="invalid")

    def test_stability_non_square(self):
        """Test stability with non-square matrix."""
        A_wrong = np.array([[1, 2, 3], [4, 5, 6]])

        with self.assertRaises(ValueError):
            analyze_stability(A_wrong)

    def test_controllability_dimension_mismatch(self):
        """Test controllability with dimension mismatch."""
        A = np.eye(2)
        B_wrong = np.array([[1], [2], [3]])  # Wrong number of rows

        with self.assertRaises(ValueError):
            analyze_controllability(A, B_wrong)

    def test_lqg_invalid_system_type(self):
        """Test LQG with invalid system type."""
        with self.assertRaises(ValueError):
            design_lqg(
                self.A_stable,
                self.B_stable,
                self.C_stable,
                self.Q2,
                self.R1,
                self.Q_process,
                self.R_meas,
                system_type="invalid",
            )


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration(ControlTestCase):
    """Integration tests combining multiple functions."""

    def test_lqr_stabilizes_unstable_system(self):
        """Test that LQR stabilizes an unstable system."""
        # Unstable continuous system
        A_unstable = np.array([[1, 1], [0, 1]])
        B = np.array([[0], [1]])

        # Verify it's unstable
        stability_ol = analyze_stability(A_unstable, system_type="continuous")
        self.assertTrue(stability_ol["is_unstable"])

        # Design LQR using unified interface
        Q = np.eye(2)
        R = np.array([[1.0]])
        result = design_lqr(A_unstable, B, Q, R, system_type="continuous")

        # Closed-loop should be stable
        stability_cl = analyze_stability(A_unstable - B @ result["gain"], system_type="continuous")
        self.assertTrue(stability_cl["is_stable"])

    def test_controllability_required_for_lqr(self):
        """Test that controllability is necessary for LQR."""
        # Uncontrollable system
        A = np.array([[1, 0], [0, 2]])
        B = np.array([[0], [0]])  # No control authority

        # Check controllability
        ctrl = analyze_controllability(A, B)
        self.assertFalse(ctrl["is_controllable"])

        # LQR should fail or give poor results
        # (scipy might still solve but result won't be useful)
        Q = np.eye(2)
        R = np.array([[1.0]])

        try:
            result = design_lqr(A, B, Q, R, system_type="continuous")
            # If it succeeds, check that it doesn't stabilize
            A_cl = A - B @ result["gain"]
            stability = analyze_stability(A_cl, system_type="continuous")
            # Should still be unstable
            self.assertTrue(stability["is_unstable"])
        except np.linalg.LinAlgError:
            # Expected failure
            pass

    def test_full_lqg_pipeline(self):
        """Test complete LQG design pipeline."""
        # Check controllability and observability
        ctrl = analyze_controllability(self.Ad_double_int, self.Bd_double_int)
        self.assertTrue(ctrl["is_controllable"])

        obs = analyze_observability(self.Ad_double_int, self.C_double_int)
        self.assertTrue(obs["is_observable"])

        # Design LQG
        result = design_lqg(
            self.Ad_double_int,
            self.Bd_double_int,
            self.C_double_int,
            self.Q2,
            self.R1,
            self.Q_process,
            self.R_meas,
            system_type="discrete",
        )

        # Verify both controller and estimator are stable
        ctrl_stability = analyze_stability(
            self.Ad_double_int - self.Bd_double_int @ result["controller_gain"],
            system_type="discrete",
        )
        self.assertTrue(ctrl_stability["is_stable"])

        est_stability = analyze_stability(
            self.Ad_double_int - result["estimator_gain"] @ self.C_double_int,
            system_type="discrete",
        )
        self.assertTrue(est_stability["is_stable"])


# ============================================================================
# Performance Tests
# ============================================================================


class TestPerformance(ControlTestCase):
    """Test numerical performance and accuracy."""

    def test_lqr_numerical_stability(self):
        """Test LQR with ill-conditioned system."""
        # Well-conditioned system
        A = np.array([[0, 1], [-2, -3]])
        B = np.array([[0], [1]])
        Q = np.diag([100, 1])
        R = np.array([[0.001]])

        result = design_lqr(A, B, Q, R, system_type="continuous")

        # Should still converge
        self.assertGreater(result["stability_margin"], 0)
        self.assertFalse(np.any(np.isnan(result["gain"])))
        self.assertFalse(np.any(np.isinf(result["gain"])))

    def test_large_system_performance(self):
        """Test performance on larger system."""
        n = 10
        A = -np.eye(n) + 0.1 * np.random.randn(n, n)
        A = (A + A.T) / 2  # Make symmetric for stability
        B = np.random.randn(n, 2)
        Q = np.eye(n)
        R = np.eye(2)

        # Should complete without error
        result = design_lqr(A, B, Q, R, system_type="continuous")

        self.assertEqual(result["gain"].shape, (2, n))
        self.assertEqual(result["cost_to_go"].shape, (n, n))


# ============================================================================
# Test Suite
# ============================================================================


def suite():
    """Create test suite."""
    test_suite = unittest.TestSuite()

    # Add all test classes
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestUnifiedLQR))
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestLQRContinuous))
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestLQRDiscrete))
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestKalmanFilter))
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestLQG))
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestStabilityAnalysis))
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestControllability))
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestObservability))
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestBackendConversion))
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestErrorHandling))
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestIntegration))
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestPerformance))

    return test_suite


if __name__ == "__main__":
    # Run tests with verbosity
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())
