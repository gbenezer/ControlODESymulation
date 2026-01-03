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
Unit Tests for Reachability and Safety Types

Tests TypedDict definitions and usage patterns for reachability analysis,
safety verification, and barrier certificate methods.
"""

import numpy as np
import pytest

from src.types.reachability import (
    BarrierCertificateResult,
    CBFResult,
    CLFResult,
    ReachabilityResult,
    ReachableSet,
    ROAResult,
    SafeSet,
    VerificationResult,
)


class TestSetRepresentations:
    """Test set representation type aliases."""

    def test_reachable_set_polytope(self):
        """Test reachable set as polytope vertices."""
        # Square polytope
        vertices: ReachableSet = np.array(
            [
                [0, 0],
                [1, 0],
                [1, 1],
                [0, 1],
            ],
        )

        assert vertices.shape == (4, 2)

    def test_safe_set_grid(self):
        """Test safe set as grid samples."""
        # Grid sampling
        safe: SafeSet = np.random.randn(100, 2)

        assert safe.shape == (100, 2)


class TestReachabilityResult:
    """Test ReachabilityResult TypedDict."""

    def test_reachability_result_creation(self):
        """Test creating ReachabilityResult instance."""
        # Forward reachability over 10 steps
        N = 10
        nx = 2

        result: ReachabilityResult = {
            "reachable_set": np.random.randn(50, nx),
            "reachable_tube": [np.random.randn(20 + i * 3, nx) for i in range(N + 1)],
            "volume": 5.2,
            "representation": "polytope",
            "method": "Ellipsoidal",
            "computation_time": 0.5,
        }

        assert result["reachable_set"].shape[1] == nx
        assert len(result["reachable_tube"]) == N + 1
        assert result["volume"] > 0

    def test_reachability_tube_growth(self):
        """Test that reachable tube generally grows over time."""
        # Simulate growing reachable set
        tube = [
            np.random.randn(10, 2),  # t=0
            np.random.randn(15, 2),  # t=1
            np.random.randn(20, 2),  # t=2
            np.random.randn(25, 2),  # t=3
        ]

        result: ReachabilityResult = {
            "reachable_set": tube[-1],
            "reachable_tube": tube,
            "volume": 10.0,
            "representation": "zonotope",
            "method": "Zonotope",
            "computation_time": 0.2,
        }

        # Check tube grows (more vertices over time)
        sizes = [r.shape[0] for r in result["reachable_tube"]]
        assert all(sizes[i] <= sizes[i + 1] for i in range(len(sizes) - 1))


class TestROAResult:
    """Test ROAResult TypedDict."""

    def test_roa_result_creation(self):
        """Test creating ROAResult instance."""
        # Quadratic Lyapunov V(x) = x'Px
        P = np.array([[2, 0], [0, 1]])
        V = lambda x: x.T @ P @ x

        result: ROAResult = {
            "region_of_attraction": np.random.randn(100, 2),
            "lyapunov_function": V,
            "lyapunov_matrix": P,
            "level_set": 1.0,
            "volume_estimate": 3.14,
            "verification_samples": 1000,
            "certification_method": "SOS",
        }

        assert result["lyapunov_matrix"].shape == (2, 2)
        assert result["level_set"] > 0
        assert result["verification_samples"] > 0

    def test_roa_lyapunov_properties(self):
        """Test Lyapunov function properties."""
        # V(x) = x'Px must be positive definite
        P = np.array([[3, 0.5], [0.5, 2]])

        # Check P is positive definite
        eigenvalues = np.linalg.eigvals(P)
        assert np.all(eigenvalues > 0)

        result: ROAResult = {
            "region_of_attraction": np.random.randn(50, 2),
            "lyapunov_function": lambda x: x.T @ P @ x,
            "lyapunov_matrix": P,
            "level_set": 2.0,
            "volume_estimate": 6.28,
            "verification_samples": 500,
            "certification_method": "LMI",
        }

        # V(0) = 0
        V = result["lyapunov_function"]
        assert np.isclose(V(np.zeros(2)), 0.0)

    def test_roa_level_set_membership(self):
        """Test checking if state is in ROA."""
        P = np.diag([1.0, 1.0])
        V = lambda x: x.T @ P @ x
        c = 1.0  # Level set

        result: ROAResult = {
            "region_of_attraction": np.random.randn(100, 2),
            "lyapunov_function": V,
            "lyapunov_matrix": P,
            "level_set": c,
            "volume_estimate": np.pi,  # Circle of radius 1
            "verification_samples": 100,
            "certification_method": "sampling",
        }

        # Check states
        x_inside = np.array([0.5, 0.5])
        x_outside = np.array([2.0, 2.0])

        assert V(x_inside) <= c  # Inside ROA
        assert V(x_outside) > c  # Outside ROA


class TestVerificationResult:
    """Test VerificationResult TypedDict."""

    def test_verification_result_success(self):
        """Test verification result when property holds."""
        result: VerificationResult = {
            "verified": True,
            "property_type": "safety",
            "confidence": 1.0,
            "certification_method": "reachability",
            "computation_time": 1.5,
        }

        assert result["verified"] == True
        assert result["confidence"] == 1.0
        assert result.get("counterexample") is None

    def test_verification_result_failure(self):
        """Test verification result when property violated."""
        # Counterexample trajectory
        violation_traj = np.array(
            [
                [0, 0],
                [0.5, 0.5],
                [1.0, 1.0],
                [2.0, 2.0],  # Violates safety
            ],
        )

        result: VerificationResult = {
            "verified": False,
            "property_type": "safety",
            "confidence": 1.0,
            "counterexample": violation_traj,
            "certification_method": "SMT",
            "computation_time": 0.8,
        }

        assert result["verified"] == False
        assert result["counterexample"] is not None
        assert result["counterexample"].shape == (4, 2)


class TestBarrierCertificateResult:
    """Test BarrierCertificateResult TypedDict."""

    def test_barrier_certificate_creation(self):
        """Test creating BarrierCertificateResult."""
        # Simple barrier: B(x) = x1 - 5 (x1 < 5 is safe)
        B = lambda x: 5 - x[0]

        result: BarrierCertificateResult = {
            "barrier_function": B,
            "barrier_matrix": None,  # Not quadratic
            "valid": True,
            "safe_set": np.random.randn(50, 2),
            "unsafe_set": np.random.randn(30, 2) + np.array([10, 0]),
            "method": "LP",
        }

        assert result["valid"] == True
        assert result["barrier_function"] is not None

    def test_barrier_separation(self):
        """Test barrier separates safe and unsafe."""
        # Quadratic barrier
        P = np.eye(2)
        center = np.array([5, 5])
        B = lambda x: 4.0 - (x - center).T @ P @ (x - center)

        result: BarrierCertificateResult = {
            "barrier_function": B,
            "barrier_matrix": P,
            "valid": True,
            "safe_set": np.random.randn(50, 2) + center * 0.5,
            "unsafe_set": np.random.randn(30, 2) + center * 1.5,
            "method": "SOS",
        }

        # Safe point (B > 0)
        x_safe = center
        assert B(x_safe) > 0

        # Unsafe point (B < 0)
        x_unsafe = center + np.array([5, 5])
        assert B(x_unsafe) < 0


class TestCBFResult:
    """Test CBFResult TypedDict."""

    def test_cbf_result_creation(self):
        """Test creating CBFResult instance."""
        result: CBFResult = {
            "safe_control": np.array([0.8, 0.5]),
            "barrier_value": 0.5,
            "barrier_derivative": -0.1,
            "constraint_active": True,
            "nominal_control": np.array([1.0, 1.0]),
            "modification_magnitude": 0.58,
        }

        assert result["safe_control"].shape == (2,)
        assert result["constraint_active"] == True
        assert result["modification_magnitude"] > 0

    def test_cbf_safety_filter(self):
        """Test CBF safety filtering."""
        # Nominal control (might be unsafe)
        u_nom = np.array([2.0, 2.0])

        # Safety filter modifies control
        u_safe = np.array([1.0, 1.0])

        result: CBFResult = {
            "safe_control": u_safe,
            "barrier_value": 0.1,  # Close to boundary
            "barrier_derivative": -0.05,
            "constraint_active": True,
            "nominal_control": u_nom,
            "modification_magnitude": np.linalg.norm(u_safe - u_nom),
        }

        # Safe control should be different from nominal
        assert not np.allclose(result["safe_control"], result["nominal_control"])
        assert result["constraint_active"] == True

    def test_cbf_inactive_constraint(self):
        """Test CBF when constraint is inactive."""
        u_nom = np.array([0.5, 0.5])

        result: CBFResult = {
            "safe_control": u_nom,  # Same as nominal
            "barrier_value": 5.0,  # Far from boundary
            "barrier_derivative": -0.01,
            "constraint_active": False,
            "nominal_control": u_nom,
            "modification_magnitude": 0.0,
        }

        # No modification when constraint inactive
        assert np.allclose(result["safe_control"], result["nominal_control"])
        assert result["modification_magnitude"] == 0.0


class TestCLFResult:
    """Test CLFResult TypedDict."""

    def test_clf_result_creation(self):
        """Test creating CLFResult instance."""
        result: CLFResult = {
            "stabilizing_control": np.array([-0.5]),
            "lyapunov_value": 0.8,
            "lyapunov_derivative": -0.4,
            "stability_margin": 0.4,
            "convergence_rate": 0.5,
            "feasible": True,
        }

        assert result["feasible"] == True
        assert result["lyapunov_derivative"] < 0  # V̇ < 0
        assert result["convergence_rate"] > 0

    def test_clf_convergence(self):
        """Test CLF ensures exponential convergence."""
        V_current = 1.0
        alpha = 0.5  # Convergence rate

        result: CLFResult = {
            "stabilizing_control": np.array([-1.0, -0.5]),
            "lyapunov_value": V_current,
            "lyapunov_derivative": -alpha * V_current,  # V̇ = -αV
            "stability_margin": 0.5,
            "convergence_rate": alpha,
            "feasible": True,
        }

        # Check exponential decrease condition
        assert result["lyapunov_derivative"] <= -alpha * V_current + 1e-10

    def test_clf_infeasible(self):
        """Test CLF when no feasible control exists."""
        result: CLFResult = {
            "stabilizing_control": np.array([0.0]),
            "lyapunov_value": 2.0,
            "lyapunov_derivative": 0.1,  # V̇ > 0 (can't decrease)
            "stability_margin": 0.0,
            "convergence_rate": 0.0,
            "feasible": False,
        }

        assert result["feasible"] == False
        assert result["lyapunov_derivative"] >= 0


class TestPracticalUseCases:
    """Test realistic usage patterns."""

    def test_obstacle_avoidance_cbf(self):
        """Test CBF for obstacle avoidance."""
        # Obstacle at (5, 5) with radius 2
        obstacle_center = np.array([5, 5])
        obstacle_radius = 2.0

        # Barrier: B(x) = ||x - x_obs||² - r²
        B = lambda x: np.linalg.norm(x - obstacle_center) ** 2 - obstacle_radius**2

        # Robot approaching obstacle
        x = np.array([4, 3])
        u_desired = np.array([1, 1])  # Toward obstacle

        # Safety filter engages
        u_safe = np.array([0.5, 0.8])  # Modified to avoid

        result: CBFResult = {
            "safe_control": u_safe,
            "barrier_value": B(x),
            "barrier_derivative": -0.5,
            "constraint_active": True,
            "nominal_control": u_desired,
            "modification_magnitude": np.linalg.norm(u_safe - u_desired),
        }

        assert result["constraint_active"] == True
        assert result["barrier_value"] > 0  # Still safe

    def test_reachability_for_planning(self):
        """Test reachability analysis for motion planning."""
        # Compute reachable set over horizon
        result: ReachabilityResult = {
            "reachable_set": np.random.randn(100, 2),
            "reachable_tube": [np.random.randn(50, 2) for _ in range(11)],
            "volume": 12.5,
            "representation": "polytope",
            "method": "Zonotope",
            "computation_time": 0.8,
        }

        # Check if target in reachable set
        target = np.array([3, 2])
        final_set = result["reachable_set"]

        # Simple containment check (convex hull)
        # In practice, use proper polytope containment
        distances = np.linalg.norm(final_set - target, axis=1)
        min_dist = np.min(distances)

        # If target close to reachable set, likely reachable
        assert min_dist < 5.0  # Threshold


class TestNumericalProperties:
    """Test numerical properties of results."""

    def test_barrier_value_sign(self):
        """Test barrier function sign convention."""
        # B > 0 in safe, B < 0 in unsafe
        B = lambda x: 5 - x[0]

        x_safe = np.array([2, 0])
        x_unsafe = np.array([8, 0])

        assert B(x_safe) > 0
        assert B(x_unsafe) < 0

    def test_lyapunov_positive_definite(self):
        """Test Lyapunov function is positive definite."""
        P = np.array([[2, 0.5], [0.5, 1]])
        V = lambda x: x.T @ P @ x

        # V(0) = 0
        assert np.isclose(V(np.zeros(2)), 0.0)

        # V(x) > 0 for x ≠ 0
        x_nonzero = np.array([1, 1])
        assert V(x_nonzero) > 0

        # P is positive definite
        eigenvalues = np.linalg.eigvals(P)
        assert np.all(eigenvalues > 0)


class TestDocumentationExamples:
    """Test that documentation examples work."""

    def test_cbf_example(self):
        """Test CBFResult example from docstring."""
        B = lambda x: np.linalg.norm(x - np.array([5, 5])) ** 2 - 4.0

        x = np.array([4, 3])
        u_desired = np.array([1, 1])
        u_safe = np.array([0.8, 0.9])

        result: CBFResult = {
            "safe_control": u_safe,
            "barrier_value": B(x),
            "barrier_derivative": -0.2,
            "constraint_active": True,
            "nominal_control": u_desired,
            "modification_magnitude": np.linalg.norm(u_safe - u_desired),
        }

        assert result["constraint_active"] == True

    def test_roa_example(self):
        """Test ROAResult example structure."""
        P = np.array([[2, 0], [0, 1]])
        V = lambda x: x.T @ P @ x

        result: ROAResult = {
            "region_of_attraction": np.random.randn(100, 2),
            "lyapunov_function": V,
            "lyapunov_matrix": P,
            "level_set": 1.0,
            "volume_estimate": 1.57,  # π/2 for ellipse
            "verification_samples": 1000,
            "certification_method": "SOS",
        }

        assert result["lyapunov_matrix"].shape == (2, 2)


class TestFieldPresence:
    """Test that all fields are accessible."""

    def test_reachability_has_required_fields(self):
        """Test ReachabilityResult has core fields."""
        result: ReachabilityResult = {
            "reachable_set": np.zeros((10, 2)),
            "reachable_tube": [np.zeros((5, 2))],
            "volume": 1.0,
        }

        assert "reachable_set" in result
        assert "reachable_tube" in result

    def test_cbf_has_required_fields(self):
        """Test CBFResult has core fields."""
        result: CBFResult = {
            "safe_control": np.zeros(2),
            "barrier_value": 1.0,
            "barrier_derivative": -0.1,
            "constraint_active": False,
            "nominal_control": np.zeros(2),
            "modification_magnitude": 0.0,
        }

        assert "safe_control" in result
        assert "barrier_value" in result
        assert "constraint_active" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
