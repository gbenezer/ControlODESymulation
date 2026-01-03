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
Unit Tests for Trajectories Types Module

Tests cover:
- State trajectory types and shapes
- Control sequence types and indexing
- Output and noise sequences
- Time array types (TimePoints, TimeSpan)
- IntegrationResult TypedDict structure
- SimulationResult TypedDict structure
- TrajectoryStatistics computation
- TrajectorySegment extraction
- Single vs batched vs multi-trial patterns
- Edge cases and realistic usage
"""


import numpy as np
import pytest

from src.types.trajectories import (  # Trajectory types; Time types; Result types; Analysis types
    ControlSequence,
    IntegrationResult,
    NoiseSequence,
    OutputSequence,
    SimulationResult,
    StateTrajectory,
    TimePoints,
    TimeSpan,
    TrajectorySegment,
    TrajectoryStatistics,
)

# ============================================================================
# Test State Trajectory Types
# ============================================================================


class TestStateTrajectory:
    """Test StateTrajectory type."""

    def test_single_trajectory_shape(self):
        """Test single trajectory has correct shape."""
        n_steps = 100
        nx = 3

        trajectory: StateTrajectory = np.random.randn(n_steps, nx)

        assert trajectory.shape == (n_steps, nx)
        assert trajectory.ndim == 2

    def test_single_trajectory_indexing(self):
        """Test indexing single trajectory."""
        trajectory: StateTrajectory = np.random.randn(100, 3)

        # Get state at time k
        x_k = trajectory[10]
        assert x_k.shape == (3,)

        # Get state component over time
        x1_traj = trajectory[:, 0]
        assert x1_traj.shape == (100,)

    def test_batched_trajectory_shape(self):
        """Test batched trajectories have correct shape."""
        n_steps = 100
        batch = 50
        nx = 3

        trajectories: StateTrajectory = np.random.randn(n_steps, batch, nx)

        assert trajectories.shape == (n_steps, batch, nx)
        assert trajectories.ndim == 3

    def test_batched_trajectory_indexing(self):
        """Test indexing batched trajectories."""
        trajectories: StateTrajectory = np.random.randn(100, 50, 3)

        # Get all states at time k
        x_k_all = trajectories[10]
        assert x_k_all.shape == (50, 3)

        # Get single trajectory
        traj_b = trajectories[:, 5, :]
        assert traj_b.shape == (100, 3)

        # Get state at time k, batch b
        x_k_b = trajectories[10, 5]
        assert x_k_b.shape == (3,)

    def test_multi_trial_trajectory_shape(self):
        """Test multi-trial trajectories."""
        n_trials = 10
        n_steps = 100
        nx = 2

        trials: StateTrajectory = np.random.randn(n_trials, n_steps, nx)

        assert trials.shape == (n_trials, n_steps, nx)
        assert trials.ndim == 3

    def test_trajectory_includes_initial_state(self):
        """Test trajectory includes initial state at t=0."""
        n_steps = 100
        nx = 3

        # Trajectory should have n_steps+1 points (including t=0)
        trajectory: StateTrajectory = np.random.randn(n_steps + 1, nx)

        x0 = trajectory[0]  # Initial state
        x_final = trajectory[-1]  # Final state

        assert x0.shape == (nx,)
        assert x_final.shape == (nx,)
        assert len(trajectory) == n_steps + 1


# ============================================================================
# Test Control Sequence Types
# ============================================================================


class TestControlSequence:
    """Test ControlSequence type."""

    def test_single_control_sequence_shape(self):
        """Test single control sequence shape."""
        n_steps = 100
        nu = 2

        u_seq: ControlSequence = np.random.randn(n_steps, nu)

        assert u_seq.shape == (n_steps, nu)

    def test_control_sequence_indexing(self):
        """Test indexing control sequence."""
        u_seq: ControlSequence = np.random.randn(100, 2)

        # Get control at time k
        u_k = u_seq[10]
        assert u_k.shape == (2,)

        # Get control component over time
        u1_seq = u_seq[:, 0]
        assert u1_seq.shape == (100,)

    def test_zero_control_sequence(self):
        """Test zero control sequence."""
        n_steps = 100
        nu = 1

        u_seq: ControlSequence = np.zeros((n_steps, nu))

        assert np.all(u_seq == 0)
        assert u_seq.shape == (n_steps, nu)

    def test_batched_control_sequence(self):
        """Test batched control sequences."""
        n_steps = 100
        batch = 50
        nu = 2

        u_batch: ControlSequence = np.random.randn(n_steps, batch, nu)

        assert u_batch.shape == (n_steps, batch, nu)

    def test_control_sequence_shorter_than_trajectory(self):
        """Test control sequence is typically n_steps, not n_steps+1."""
        n_steps = 100

        # State trajectory: (n_steps+1, nx) - includes initial state
        x_traj: StateTrajectory = np.random.randn(n_steps + 1, 3)

        # Control sequence: (n_steps, nu) - u[k] affects x[k] -> x[k+1]
        u_seq: ControlSequence = np.random.randn(n_steps, 2)

        assert len(x_traj) == len(u_seq) + 1


# ============================================================================
# Test Output Sequence Types
# ============================================================================


class TestOutputSequence:
    """Test OutputSequence type."""

    def test_output_sequence_shape(self):
        """Test output sequence shape."""
        n_steps = 100
        ny = 2

        y_seq: OutputSequence = np.random.randn(n_steps, ny)

        assert y_seq.shape == (n_steps, ny)

    def test_output_from_state_trajectory(self):
        """Test generating output from state trajectory."""
        n_steps = 101
        nx = 3
        ny = 2

        # State trajectory
        x_traj: StateTrajectory = np.random.randn(n_steps, nx)

        # Observation matrix
        C = np.random.randn(ny, nx)

        # Generate output
        y_seq: OutputSequence = (C @ x_traj.T).T

        assert y_seq.shape == (n_steps, ny)

    def test_noisy_measurements(self):
        """Test adding noise to measurements."""
        n_steps = 100
        ny = 2

        # Clean output
        y_clean: OutputSequence = np.random.randn(n_steps, ny)

        # Add noise
        noise_std = 0.1
        y_noisy: OutputSequence = y_clean + np.random.randn(n_steps, ny) * noise_std

        assert y_noisy.shape == y_clean.shape
        assert not np.allclose(y_clean, y_noisy)

    def test_batched_output_sequence(self):
        """Test batched output sequences."""
        n_steps = 100
        batch = 50
        ny = 3

        y_batch: OutputSequence = np.random.randn(n_steps, batch, ny)

        assert y_batch.shape == (n_steps, batch, ny)


# ============================================================================
# Test Noise Sequence Types
# ============================================================================


class TestNoiseSequence:
    """Test NoiseSequence type."""

    def test_discrete_noise_sequence(self):
        """Test discrete stochastic noise (standard normal)."""
        n_steps = 100
        nw = 2

        # Standard normal: w[k] ~ N(0, I)
        w_seq: NoiseSequence = np.random.randn(n_steps, nw)

        assert w_seq.shape == (n_steps, nw)

        # Check statistics (approximately N(0, 1))
        assert np.abs(np.mean(w_seq)) < 0.2  # Mean ≈ 0
        assert np.abs(np.std(w_seq) - 1.0) < 0.2  # Std ≈ 1

    def test_brownian_increments(self):
        """Test Brownian increments for continuous SDE."""
        n_steps = 1000
        nw = 2
        dt = 0.01

        # Brownian increments: dW ~ N(0, dt*I)
        dW: NoiseSequence = np.random.randn(n_steps, nw) * np.sqrt(dt)

        assert dW.shape == (n_steps, nw)

        # Check scaling
        assert np.abs(np.std(dW) - np.sqrt(dt)) < 0.02

    def test_reproducible_noise_with_seed(self):
        """Test reproducible noise generation."""
        n_steps = 100
        nw = 2
        seed = 42

        # Generate twice with same seed
        np.random.seed(seed)
        w1: NoiseSequence = np.random.randn(n_steps, nw)

        np.random.seed(seed)
        w2: NoiseSequence = np.random.randn(n_steps, nw)

        assert np.array_equal(w1, w2)

    def test_batched_noise_monte_carlo(self):
        """Test batched noise for Monte Carlo simulation."""
        n_steps = 100
        n_trials = 1000
        nw = 2

        w_batch: NoiseSequence = np.random.randn(n_steps, n_trials, nw)

        assert w_batch.shape == (n_steps, n_trials, nw)


# ============================================================================
# Test Time Types
# ============================================================================


class TestTimeTypes:
    """Test time array types."""

    def test_time_points_regular_grid(self):
        """Test regular time grid."""
        t_start = 0.0
        t_end = 10.0
        n_points = 101

        t: TimePoints = np.linspace(t_start, t_end, n_points)

        assert t.shape == (n_points,)
        assert t[0] == t_start
        assert t[-1] == t_end

        # Check uniform spacing
        dt = np.diff(t)
        assert np.allclose(dt, dt[0])

    def test_time_points_irregular_grid(self):
        """Test irregular time grid."""
        t: TimePoints = np.array([0, 0.1, 0.15, 0.5, 1.0, 2.0, 5.0, 10.0])

        assert len(t) == 8
        assert t[0] == 0
        assert t[-1] == 10.0

        # Check non-uniform spacing
        dt = np.diff(t)
        assert not np.allclose(dt, dt[0])

    def test_time_points_logarithmic(self):
        """Test logarithmic time spacing."""
        t: TimePoints = np.logspace(-3, 1, 100)  # 0.001 to 10

        assert len(t) == 100
        assert t[0] == pytest.approx(0.001, rel=1e-6)
        assert t[-1] == pytest.approx(10.0, rel=1e-6)

    def test_time_span_tuple(self):
        """Test TimeSpan tuple structure."""
        t_span: TimeSpan = (0.0, 10.0)

        assert isinstance(t_span, tuple)
        assert len(t_span) == 2

        t_start, t_end = t_span
        assert t_start < t_end

    def test_time_span_non_zero_start(self):
        """Test TimeSpan with non-zero start."""
        t_span: TimeSpan = (5.0, 15.0)

        t_start, t_end = t_span
        assert t_start == 5.0
        assert t_end == 15.0
        assert t_end - t_start == 10.0


# ============================================================================
# Test Integration Result
# ============================================================================


class TestIntegrationResult:
    """Test IntegrationResult TypedDict."""

    def test_integration_result_success(self):
        """Test successful integration result."""
        result: IntegrationResult = {
            "t": np.linspace(0, 10, 101),
            "y": np.random.randn(101, 3),
            "success": True,
            "message": "Integration successful",
            "nfev": 523,
            "njev": 0,
        }

        assert result["success"] is True
        assert result["t"].shape == (101,)
        assert result["y"].shape == (101, 3)
        assert result["nfev"] > 0

    def test_integration_result_failure(self):
        """Test failed integration result."""
        result: IntegrationResult = {
            "t": np.array([0.0]),
            "y": np.array([[1.0, 0.0]]),
            "success": False,
            "message": "Integration stopped: step size too small",
            "nfev": 10000,
        }

        assert result["success"] is False
        assert "message" in result

    def test_integration_result_partial_fields(self):
        """Test IntegrationResult with partial fields (total=False)."""
        result: IntegrationResult = {
            "t": np.linspace(0, 1, 11),
            "y": np.random.randn(11, 2),
            "success": True,
        }

        assert "t" in result
        assert "y" in result
        assert "success" in result
        assert "nfev" not in result  # Optional

    def test_integration_result_with_jacobian_info(self):
        """Test IntegrationResult with Jacobian evaluations."""
        result: IntegrationResult = {
            "t": np.linspace(0, 10, 101),
            "y": np.random.randn(101, 3),
            "success": True,
            "message": "Success",
            "nfev": 200,
            "njev": 50,  # Jacobian evaluations
            "nlu": 30,  # LU decompositions
        }

        assert result["njev"] == 50
        assert result["nlu"] == 30


# ============================================================================
# Test Simulation Result
# ============================================================================


class TestSimulationResult:
    """Test SimulationResult TypedDict."""

    def test_simulation_result_basic(self):
        """Test basic simulation result."""
        n_steps = 100
        nx = 3
        nu = 2

        result: SimulationResult = {
            "states": np.random.randn(n_steps + 1, nx),
            "controls": np.random.randn(n_steps, nu),
        }

        assert result["states"].shape == (n_steps + 1, nx)
        assert result["controls"].shape == (n_steps, nu)

    def test_simulation_result_with_outputs(self):
        """Test simulation result with outputs."""
        n_steps = 100
        nx = 3
        nu = 2
        ny = 2

        result: SimulationResult = {
            "states": np.random.randn(n_steps + 1, nx),
            "controls": np.random.randn(n_steps, nu),
            "outputs": np.random.randn(n_steps + 1, ny),
        }

        assert "outputs" in result
        assert result["outputs"].shape == (n_steps + 1, ny)

    def test_simulation_result_stochastic(self):
        """Test stochastic simulation result with noise."""
        n_steps = 100
        nx = 2
        nu = 1
        nw = 2

        result: SimulationResult = {
            "states": np.random.randn(n_steps + 1, nx),
            "controls": np.random.randn(n_steps, nu),
            "noise": np.random.randn(n_steps, nw),
        }

        assert "noise" in result
        assert result["noise"].shape == (n_steps, nw)

    def test_simulation_result_with_time(self):
        """Test simulation result with time information."""
        n_steps = 100
        dt = 0.01

        result: SimulationResult = {
            "states": np.random.randn(n_steps + 1, 3),
            "controls": np.random.randn(n_steps, 2),
            "time": np.arange(n_steps + 1) * dt,
        }

        assert "time" in result
        assert result["time"].shape == (n_steps + 1,)
        assert result["time"][-1] == pytest.approx(n_steps * dt)

    def test_simulation_result_with_info(self):
        """Test simulation result with metadata."""
        result: SimulationResult = {
            "states": np.random.randn(101, 3),
            "controls": np.random.randn(100, 2),
            "info": {
                "cost": 123.45,
                "constraint_violations": 0,
                "method": "euler",
            },
        }

        assert "info" in result
        assert result["info"]["cost"] == 123.45
        assert result["info"]["method"] == "euler"


# ============================================================================
# Test Trajectory Statistics
# ============================================================================


class TestTrajectoryStatistics:
    """Test TrajectoryStatistics TypedDict."""

    def test_trajectory_statistics_computation(self):
        """Test computing trajectory statistics."""
        trajectory: StateTrajectory = np.random.randn(100, 3)

        stats: TrajectoryStatistics = {
            "mean": np.mean(trajectory, axis=0),
            "std": np.std(trajectory, axis=0),
            "min": np.min(trajectory, axis=0),
            "max": np.max(trajectory, axis=0),
            "initial": trajectory[0],
            "final": trajectory[-1],
            "length": len(trajectory),
        }

        assert stats["mean"].shape == (3,)
        assert stats["std"].shape == (3,)
        assert stats["length"] == 100

    def test_trajectory_statistics_with_duration(self):
        """Test trajectory statistics with duration."""
        trajectory: StateTrajectory = np.random.randn(101, 2)
        time: TimePoints = np.linspace(0, 10, 101)

        stats: TrajectoryStatistics = {
            "mean": np.mean(trajectory, axis=0),
            "std": np.std(trajectory, axis=0),
            "min": np.min(trajectory, axis=0),
            "max": np.max(trajectory, axis=0),
            "initial": trajectory[0],
            "final": trajectory[-1],
            "length": len(trajectory),
            "duration": time[-1] - time[0],
        }

        assert stats["duration"] == pytest.approx(10.0)

    def test_trajectory_statistics_partial(self):
        """Test partial trajectory statistics."""
        trajectory: StateTrajectory = np.random.randn(100, 3)

        stats: TrajectoryStatistics = {
            "mean": np.mean(trajectory, axis=0),
            "initial": trajectory[0],
            "final": trajectory[-1],
        }

        assert "mean" in stats
        assert "initial" in stats
        assert "std" not in stats  # Optional


# ============================================================================
# Test Trajectory Segment
# ============================================================================


class TestTrajectorySegment:
    """Test TrajectorySegment TypedDict."""

    def test_trajectory_segment_extraction(self):
        """Test extracting trajectory segment."""
        n_steps = 100
        trajectory: StateTrajectory = np.random.randn(n_steps, 3)
        time: TimePoints = np.linspace(0, 10, n_steps)

        # Extract segment from t=2 to t=5
        mask = (time >= 2.0) & (time <= 5.0)
        indices = np.where(mask)[0]

        segment: TrajectorySegment = {
            "states": trajectory[mask],
            "controls": None,
            "time": time[mask],
            "start_index": indices[0],
            "end_index": indices[-1],
        }

        assert segment["start_index"] < segment["end_index"]
        assert len(segment["states"]) == len(segment["time"])

    def test_trajectory_segment_with_controls(self):
        """Test trajectory segment with controls."""
        n_steps = 100
        trajectory: StateTrajectory = np.random.randn(n_steps + 1, 3)
        controls: ControlSequence = np.random.randn(n_steps, 2)
        time: TimePoints = np.linspace(0, 10, n_steps + 1)

        # Extract first half
        mid = n_steps // 2

        segment: TrajectorySegment = {
            "states": trajectory[: mid + 1],
            "controls": controls[:mid],
            "time": time[: mid + 1],
            "start_index": 0,
            "end_index": mid,
        }

        assert segment["states"].shape[0] == segment["controls"].shape[0] + 1

    def test_extract_transient_response(self):
        """Test extracting transient response segment."""
        n_steps = 1000
        dt = 0.01
        trajectory: StateTrajectory = np.random.randn(n_steps + 1, 2)
        time: TimePoints = np.arange(n_steps + 1) * dt

        # Extract first 2 seconds (transient)
        mask = time <= 2.0
        indices = np.where(mask)[0]

        transient: TrajectorySegment = {
            "states": trajectory[mask],
            "controls": None,
            "time": time[mask],
            "start_index": 0,
            "end_index": indices[-1],
        }

        assert transient["time"][-1] <= 2.0


# ============================================================================
# Test Realistic Usage Patterns
# ============================================================================


class TestRealisticUsage:
    """Test types in realistic scenarios."""

    def test_discrete_simulation_workflow(self):
        """Test complete discrete simulation workflow."""
        # Setup
        nx, nu = 3, 2
        n_steps = 100
        dt = 0.01

        # Initial condition
        x0: StateTrajectory = np.array([1.0, 0.0, 0.5])

        # Control sequence
        u_seq: ControlSequence = np.zeros((n_steps, nu))

        # Simulate (mock)
        trajectory: StateTrajectory = np.random.randn(n_steps + 1, nx)
        trajectory[0] = x0  # Set initial state

        # Time points
        time: TimePoints = np.arange(n_steps + 1) * dt

        # Result
        result: SimulationResult = {
            "states": trajectory,
            "controls": u_seq,
            "time": time,
        }

        assert result["states"][0][0] == x0[0]
        assert result["time"][-1] == pytest.approx(n_steps * dt)

    def test_continuous_integration_workflow(self):
        """Test continuous integration workflow."""
        # Integration
        t_span: TimeSpan = (0.0, 10.0)
        t_eval: TimePoints = np.linspace(0, 10, 101)

        # Mock integration result
        result: IntegrationResult = {
            "t": t_eval,
            "y": np.random.randn(101, 3),
            "success": True,
            "message": "Integration successful",
            "nfev": 500,
        }

        # Extract trajectory
        trajectory: StateTrajectory = result["y"]
        time: TimePoints = result["t"]

        assert len(trajectory) == len(time)
        assert result["success"] is True

    def test_monte_carlo_simulation(self):
        """Test Monte Carlo simulation with batched noise."""
        n_steps = 100
        n_trials = 1000
        nx = 2
        nw = 2

        # Initial condition
        x0 = np.zeros(nx)

        # Batched noise
        w_batch: NoiseSequence = np.random.randn(n_steps, n_trials, nw)

        # Mock batched simulation
        trajectories: StateTrajectory = np.random.randn(n_steps + 1, n_trials, nx)

        # Compute statistics across trials
        mean_traj = np.mean(trajectories, axis=1)
        std_traj = np.std(trajectories, axis=1)

        assert mean_traj.shape == (n_steps + 1, nx)
        assert std_traj.shape == (n_steps + 1, nx)

    def test_trajectory_analysis_pipeline(self):
        """Test complete trajectory analysis pipeline."""
        # Generate trajectory
        n_steps = 100
        trajectory: StateTrajectory = np.random.randn(n_steps, 3)
        time: TimePoints = np.linspace(0, 10, n_steps)

        # Compute statistics
        stats: TrajectoryStatistics = {
            "mean": np.mean(trajectory, axis=0),
            "std": np.std(trajectory, axis=0),
            "min": np.min(trajectory, axis=0),
            "max": np.max(trajectory, axis=0),
            "initial": trajectory[0],
            "final": trajectory[-1],
            "length": len(trajectory),
            "duration": time[-1] - time[0],
        }

        # Extract transient segment (first 20%)
        transient_end = int(0.2 * n_steps)
        transient: TrajectorySegment = {
            "states": trajectory[:transient_end],
            "controls": None,
            "time": time[:transient_end],
            "start_index": 0,
            "end_index": transient_end - 1,
        }

        # Verify analysis
        assert stats["duration"] == pytest.approx(10.0)
        assert len(transient["states"]) == transient_end


# ============================================================================
# Test Edge Cases
# ============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_step_trajectory(self):
        """Test trajectory with single time step."""
        trajectory: StateTrajectory = np.random.randn(2, 3)  # t=0 and t=1

        assert len(trajectory) == 2
        assert trajectory[0].shape == (3,)

    def test_scalar_system_trajectory(self):
        """Test trajectory for scalar system (nx=1)."""
        n_steps = 100

        trajectory: StateTrajectory = np.random.randn(n_steps, 1)

        assert trajectory.shape == (n_steps, 1)

    def test_autonomous_system_empty_control(self):
        """Test autonomous system with no control."""
        n_steps = 100

        # Empty control (nu=0)
        u_seq: ControlSequence = np.zeros((n_steps, 0))

        assert u_seq.shape == (n_steps, 0)

    def test_instantaneous_time_span(self):
        """Test time span with zero duration."""
        t_span: TimeSpan = (5.0, 5.0)

        t_start, t_end = t_span
        duration = t_end - t_start

        assert duration == 0.0

    def test_large_batch_size(self):
        """Test with very large batch size."""
        n_steps = 100
        batch = 10000
        nx = 3

        trajectories: StateTrajectory = np.random.randn(n_steps, batch, nx)

        assert trajectories.shape == (n_steps, batch, nx)


# ============================================================================
# Test Documentation Examples
# ============================================================================


class TestDocumentationExamples:
    """Test examples from docstrings work correctly."""

    def test_state_trajectory_example(self):
        """Test StateTrajectory docstring example."""
        # Single trajectory from simulation
        trajectory: StateTrajectory = np.random.randn(101, 2)

        # Extract position and velocity
        position = trajectory[:, 0]
        velocity = trajectory[:, 1]

        assert position.shape == (101,)
        assert velocity.shape == (101,)

    def test_control_sequence_example(self):
        """Test ControlSequence docstring example."""
        # Zero control
        u_seq: ControlSequence = np.zeros((100, 1))

        assert np.all(u_seq == 0)
        assert u_seq.shape == (100, 1)

    def test_time_points_example(self):
        """Test TimePoints docstring example."""
        # Regular time grid
        t: TimePoints = np.linspace(0, 10, 101)
        dt = t[1] - t[0]

        assert dt == pytest.approx(0.1)

    def test_time_span_example(self):
        """Test TimeSpan docstring example."""
        t_span: TimeSpan = (0.0, 10.0)

        t_start, t_end = t_span
        assert t_start == 0.0
        assert t_end == 10.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
