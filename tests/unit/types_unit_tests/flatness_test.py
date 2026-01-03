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
Unit Tests for Differential Flatness Types

Tests TypedDict definitions and usage patterns for differential flatness
and trajectory planning types.
"""

import numpy as np
import pytest

from src.types.flatness import (
    DifferentialFlatnessResult,
    FlatnessOutput,
    TrajectoryPlanningResult,
)


class TestTypeAliases:
    """Test type aliases for flatness."""

    def test_flatness_output_quadrotor(self):
        """Test flat output for quadrotor."""
        # Flat output: [x, y, z, ψ]
        y_flat: FlatnessOutput = np.array([1.0, 2.0, 3.0, 0.5])

        assert y_flat.shape == (4,)

    def test_flatness_output_unicycle(self):
        """Test flat output for unicycle."""
        # Flat output: [x, y]
        y_flat: FlatnessOutput = np.array([1.5, 2.5])

        assert y_flat.shape == (2,)


class TestDifferentialFlatnessResult:
    """Test DifferentialFlatnessResult TypedDict."""

    def test_flatness_result_creation(self):
        """Test creating flatness result."""
        # Flat output function
        sigma = lambda x: x[:2]  # Position

        # Inverse maps
        phi_x = lambda y, dy, ddy: np.concatenate([y, dy])
        phi_u = lambda y, dy, ddy: ddy

        result: DifferentialFlatnessResult = {
            "is_flat": True,
            "flat_output": sigma,
            "flat_dimension": 2,
            "differential_order": 2,
            "state_from_flat": phi_x,
            "control_from_flat": phi_u,
            "verification_method": "analytic",
        }

        assert result["is_flat"] == True
        assert result["flat_dimension"] == 2
        assert result["differential_order"] > 0

    def test_flatness_double_integrator(self):
        """Test flatness of double integrator."""
        # ẍ = u, flat output: y = x

        # Flat output: position
        sigma = lambda x: np.array([x[0]])

        # State from flat: x = [y, ẏ]
        phi_x = lambda y, dy: np.array([y[0], dy[0]])

        # Control from flat: u = ÿ
        phi_u = lambda y, dy, ddy: np.array([ddy[0]])

        result: DifferentialFlatnessResult = {
            "is_flat": True,
            "flat_output": sigma,
            "flat_dimension": 1,
            "differential_order": 2,
            "state_from_flat": phi_x,
            "control_from_flat": phi_u,
            "verification_method": "analytic",
        }

        # Test mappings
        x = np.array([1.0, 0.5])
        y = result["flat_output"](x)
        assert y.shape == (1,)

    def test_flatness_unicycle(self):
        """Test flatness of unicycle."""
        # Flat output: [x, y] (position)
        sigma = lambda x: x[:2]

        # State from flat: x, y, θ
        def phi_x(y, dy):
            theta = np.arctan2(dy[1], dy[0])
            return np.array([y[0], y[1], theta])

        # Control from flat: v, ω
        def phi_u(y, dy, ddy):
            v = np.linalg.norm(dy)
            omega = (dy[0] * ddy[1] - dy[1] * ddy[0]) / (dy[0] ** 2 + dy[1] ** 2)
            return np.array([v, omega])

        result: DifferentialFlatnessResult = {
            "is_flat": True,
            "flat_output": sigma,
            "flat_dimension": 2,
            "differential_order": 2,
            "state_from_flat": phi_x,
            "control_from_flat": phi_u,
            "verification_method": "analytic",
        }

        assert result["is_flat"] == True
        assert result["flat_dimension"] == 2

    def test_flatness_negative_case(self):
        """Test non-flat system."""
        result: DifferentialFlatnessResult = {
            "is_flat": False,
            "flat_output": None,
            "flat_dimension": 0,
            "differential_order": 0,
            "state_from_flat": None,
            "control_from_flat": None,
            "verification_method": "symbolic",
        }

        assert result["is_flat"] == False
        assert result["flat_output"] is None


class TestTrajectoryPlanningResult:
    """Test TrajectoryPlanningResult TypedDict."""

    def test_trajectory_planning_result_creation(self):
        """Test creating trajectory planning result."""
        N = 100
        nx, nu = 4, 2

        result: TrajectoryPlanningResult = {
            "state_trajectory": np.random.randn(N + 1, nx),
            "control_trajectory": np.random.randn(N, nu),
            "flat_trajectory": np.random.randn(N + 1, 2),
            "time_points": np.linspace(0, 5, N + 1),
            "cost": 25.3,
            "feasible": True,
            "method": "flatness",
            "computation_time": 0.15,
        }

        assert result["state_trajectory"].shape == (N + 1, nx)
        assert result["control_trajectory"].shape == (N, nu)
        assert result["feasible"] == True

    def test_trajectory_planning_dimensions(self):
        """Test trajectory dimensions consistency."""
        N = 50

        result: TrajectoryPlanningResult = {
            "state_trajectory": np.random.randn(N + 1, 3),
            "control_trajectory": np.random.randn(N, 1),
            "time_points": np.linspace(0, 2, N + 1),
            "cost": 10.5,
            "feasible": True,
            "method": "flatness",
            "computation_time": 0.05,
        }

        # States: N+1 points, Controls: N points
        assert len(result["state_trajectory"]) == len(result["time_points"])
        assert len(result["control_trajectory"]) == len(result["time_points"]) - 1

    def test_trajectory_planning_flat_output(self):
        """Test trajectory planning with flat output."""
        N = 30
        t = np.linspace(0, 3, N + 1)

        # Flat trajectory (e.g., quadrotor position)
        y_flat = np.column_stack(
            [
                np.sin(t),
                np.cos(t),
                t / 3,
            ],
        )

        result: TrajectoryPlanningResult = {
            "state_trajectory": np.random.randn(N + 1, 12),  # Full state
            "control_trajectory": np.random.randn(N, 4),  # Quad controls
            "flat_trajectory": y_flat,
            "time_points": t,
            "cost": 15.0,
            "feasible": True,
            "method": "flatness",
            "computation_time": 0.08,
        }

        assert result["flat_trajectory"].shape == (N + 1, 3)

    def test_trajectory_planning_infeasible(self):
        """Test infeasible trajectory planning."""
        result: TrajectoryPlanningResult = {
            "state_trajectory": np.array([]),
            "control_trajectory": np.array([]),
            "time_points": np.array([]),
            "cost": np.inf,
            "feasible": False,
            "method": "optimization",
            "computation_time": 1.5,
        }

        assert result["feasible"] == False
        assert result["cost"] == np.inf


class TestPracticalUseCases:
    """Test realistic usage patterns."""

    def test_double_integrator_trajectory(self):
        """Test trajectory planning for double integrator."""
        # System: ẍ = u
        # Flat output: y = x

        # Plan trajectory
        N = 50
        t = np.linspace(0, 2, N + 1)

        # Desired flat trajectory (5th order polynomial)
        # Normalized time s = t/T ∈ [0, 1]
        s = t / 2.0
        # Polynomial that goes from 0 to 1: p(s) = 10s³ - 15s⁴ + 6s⁵
        # Scale to go from 0 to 3
        y_flat = 3.0 * (10 * s**3 - 15 * s**4 + 6 * s**5)
        dy_flat = np.gradient(y_flat, t)
        ddy_flat = np.gradient(dy_flat, t)

        # State from flat
        x_traj = np.column_stack([y_flat, dy_flat])

        # Control from flat
        u_traj = ddy_flat[:-1]  # N points

        result: TrajectoryPlanningResult = {
            "state_trajectory": x_traj,
            "control_trajectory": u_traj.reshape(-1, 1),
            "flat_trajectory": y_flat.reshape(-1, 1),
            "time_points": t,
            "cost": np.sum(u_traj**2),
            "feasible": True,
            "method": "flatness",
            "computation_time": 0.02,
        }

        # Verify boundary conditions
        assert np.isclose(result["state_trajectory"][0, 0], 0.0, atol=0.1)
        assert np.isclose(result["state_trajectory"][-1, 0], 3.0, atol=0.1)
        # Velocity should be zero at endpoints
        assert np.isclose(result["state_trajectory"][0, 1], 0.0, atol=0.2)
        assert np.isclose(result["state_trajectory"][-1, 1], 0.0, atol=0.2)

    def test_quadrotor_trajectory_planning(self):
        """Test quadrotor trajectory planning using flatness."""
        # Flat output: [x, y, z, ψ]
        N = 100
        t = np.linspace(0, 5, N + 1)

        # Circular trajectory at constant height
        radius = 2.0
        height = 3.0
        omega = 2 * np.pi / 5  # One revolution

        y_flat = np.column_stack(
            [
                radius * np.cos(omega * t),  # x
                radius * np.sin(omega * t),  # y
                height * np.ones(N + 1),  # z
                omega * t,  # ψ (yaw)
            ],
        )

        result: TrajectoryPlanningResult = {
            "state_trajectory": np.random.randn(N + 1, 12),  # Full state
            "control_trajectory": np.random.randn(N, 4),  # 4 motors
            "flat_trajectory": y_flat,
            "time_points": t,
            "cost": 50.0,
            "feasible": True,
            "method": "flatness",
            "computation_time": 0.12,
        }

        assert result["flat_trajectory"].shape == (N + 1, 4)
        assert result["feasible"] == True

    def test_unicycle_path_planning(self):
        """Test unicycle path planning."""
        # Flat output: [x, y]
        N = 60
        t = np.linspace(0, 3, N + 1)

        # S-curve trajectory
        y_flat = np.column_stack(
            [
                t,  # x
                np.sin(2 * np.pi * t / 3),  # y
            ],
        )

        result: TrajectoryPlanningResult = {
            "state_trajectory": np.random.randn(N + 1, 3),  # [x, y, θ]
            "control_trajectory": np.random.randn(N, 2),  # [v, ω]
            "flat_trajectory": y_flat,
            "time_points": t,
            "cost": 20.0,
            "feasible": True,
            "method": "flatness",
            "computation_time": 0.05,
        }

        assert result["flat_trajectory"].shape == (N + 1, 2)


class TestNumericalProperties:
    """Test numerical properties of results."""

    def test_flat_dimension_positive(self):
        """Test flat dimension is positive for flat systems."""
        result: DifferentialFlatnessResult = {
            "is_flat": True,
            "flat_output": lambda x: x[:2],
            "flat_dimension": 2,
            "differential_order": 2,
            "state_from_flat": lambda y, dy: y,
            "control_from_flat": lambda y, dy, ddy: ddy,
            "verification_method": "analytic",
        }

        if result["is_flat"]:
            assert result["flat_dimension"] > 0

    def test_differential_order_non_negative(self):
        """Test differential order is non-negative."""
        result: DifferentialFlatnessResult = {
            "is_flat": True,
            "flat_output": lambda x: x[0:1],
            "flat_dimension": 1,
            "differential_order": 2,
            "state_from_flat": lambda y, dy: y,
            "control_from_flat": lambda y, dy, ddy: ddy,
            "verification_method": "analytic",
        }

        assert result["differential_order"] >= 0

    def test_trajectory_time_monotonic(self):
        """Test time points are monotonically increasing."""
        t = np.linspace(0, 5, 101)

        result: TrajectoryPlanningResult = {
            "state_trajectory": np.random.randn(101, 4),
            "control_trajectory": np.random.randn(100, 2),
            "time_points": t,
            "cost": 10.0,
            "feasible": True,
            "method": "flatness",
            "computation_time": 0.1,
        }

        # Time should be monotonically increasing
        assert np.all(np.diff(result["time_points"]) > 0)


class TestFlatnessVerification:
    """Test flatness verification methods."""

    def test_analytic_verification(self):
        """Test analytic flatness verification."""
        result: DifferentialFlatnessResult = {
            "is_flat": True,
            "flat_output": lambda x: x[:2],
            "flat_dimension": 2,
            "differential_order": 2,
            "state_from_flat": lambda y, dy: y,
            "control_from_flat": lambda y, dy, ddy: ddy,
            "verification_method": "analytic",
        }

        assert result["verification_method"] == "analytic"

    def test_symbolic_verification(self):
        """Test symbolic flatness verification."""
        result: DifferentialFlatnessResult = {
            "is_flat": False,
            "flat_output": None,
            "flat_dimension": 0,
            "differential_order": 0,
            "state_from_flat": None,
            "control_from_flat": None,
            "verification_method": "symbolic",
        }

        assert result["verification_method"] == "symbolic"


class TestDocumentationExamples:
    """Test that documentation examples work."""

    def test_flatness_result_example(self):
        """Test DifferentialFlatnessResult example from docstring."""
        sigma = lambda x: x[:2]
        phi_x = lambda y, dy, ddy: np.concatenate([y, dy])
        phi_u = lambda y, dy, ddy: ddy

        result: DifferentialFlatnessResult = {
            "is_flat": True,
            "flat_output": sigma,
            "flat_dimension": 2,
            "differential_order": 2,
            "state_from_flat": phi_x,
            "control_from_flat": phi_u,
            "verification_method": "analytic",
        }

        assert result["is_flat"] == True
        assert callable(result["flat_output"])

    def test_trajectory_planning_example(self):
        """Test TrajectoryPlanningResult example structure."""
        N = 50

        result: TrajectoryPlanningResult = {
            "state_trajectory": np.random.randn(N + 1, 4),
            "control_trajectory": np.random.randn(N, 2),
            "time_points": np.linspace(0, 5, N + 1),
            "cost": 25.0,
            "feasible": True,
            "method": "flatness",
            "computation_time": 0.1,
        }

        assert result["feasible"] == True
        assert result["method"] == "flatness"


class TestFieldPresence:
    """Test that all fields are accessible."""

    def test_flatness_result_has_required_fields(self):
        """Test DifferentialFlatnessResult has core fields."""
        result: DifferentialFlatnessResult = {
            "is_flat": True,
            "flat_output": lambda x: x,
            "flat_dimension": 2,
            "differential_order": 2,
            "state_from_flat": lambda y, dy: y,
            "control_from_flat": lambda y, dy, ddy: ddy,
            "verification_method": "analytic",
        }

        assert "is_flat" in result
        assert "flat_dimension" in result
        assert "differential_order" in result

    def test_trajectory_planning_has_required_fields(self):
        """Test TrajectoryPlanningResult has core fields."""
        result: TrajectoryPlanningResult = {
            "state_trajectory": np.zeros((10, 2)),
            "control_trajectory": np.zeros((9, 1)),
            "time_points": np.linspace(0, 1, 10),
            "cost": 5.0,
            "feasible": True,
            "method": "flatness",
            "computation_time": 0.05,
        }

        assert "state_trajectory" in result
        assert "control_trajectory" in result
        assert "feasible" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
