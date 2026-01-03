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
Unit Tests for Trajectory Plotter

Tests trajectory visualization functionality including single/batched
trajectories, theme integration, and layout determination.
"""

import numpy as np
import plotly.graph_objects as go
import pytest

from src.visualization.trajectory_plotter import TrajectoryPlotter

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def plotter():
    """Create default trajectory plotter."""
    return TrajectoryPlotter(backend="numpy")


@pytest.fixture
def simple_trajectory():
    """Create simple 1D trajectory."""
    t = np.linspace(0, 10, 100)
    x = np.sin(t)[:, None]  # (100, 1)
    return t, x


@pytest.fixture
def multi_state_trajectory():
    """Create 2D trajectory."""
    t = np.linspace(0, 10, 100)
    x = np.column_stack([np.sin(t), np.cos(t)])  # (100, 2)
    return t, x


@pytest.fixture
def batched_trajectory():
    """Create batched trajectories."""
    t = np.linspace(0, 10, 100)
    x_batch = np.stack(
        [np.column_stack([np.sin(t + phi), np.cos(t + phi)]) for phi in [0, 0.5, 1.0]],
    )  # (3, 100, 2)
    return t, x_batch


@pytest.fixture
def control_input():
    """Create control input."""
    u = 0.1 * np.random.randn(100, 1)
    return u


# ============================================================================
# Initialization Tests
# ============================================================================


class TestInitialization:
    """Test TrajectoryPlotter initialization."""

    def test_default_initialization(self):
        """Test default initialization."""
        plotter = TrajectoryPlotter()
        assert plotter.backend == "numpy"
        assert plotter.default_theme == "default"

    def test_backend_initialization(self):
        """Test initialization with different backends."""
        plotter_numpy = TrajectoryPlotter(backend="numpy")
        assert plotter_numpy.backend == "numpy"

        plotter_torch = TrajectoryPlotter(backend="torch")
        assert plotter_torch.backend == "torch"

        plotter_jax = TrajectoryPlotter(backend="jax")
        assert plotter_jax.backend == "jax"

    def test_theme_initialization(self):
        """Test initialization with different default themes."""
        plotter_pub = TrajectoryPlotter(default_theme="publication")
        assert plotter_pub.default_theme == "publication"

        plotter_dark = TrajectoryPlotter(default_theme="dark")
        assert plotter_dark.default_theme == "dark"


# ============================================================================
# plot_trajectory Tests
# ============================================================================


class TestPlotTrajectory:
    """Test plot_trajectory method."""

    def test_simple_trajectory(self, plotter, simple_trajectory):
        """Test plotting simple 1D trajectory."""
        t, x = simple_trajectory
        fig = plotter.plot_trajectory(t, x)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
        assert fig.layout.title.text == "State Trajectories"

    def test_multi_state_trajectory(self, plotter, multi_state_trajectory):
        """Test plotting multi-state trajectory."""
        t, x = multi_state_trajectory
        fig = plotter.plot_trajectory(t, x)

        assert isinstance(fig, go.Figure)
        # Should have 2 traces (one per state)
        assert len(fig.data) == 2

    def test_batched_trajectory(self, plotter, batched_trajectory):
        """Test plotting batched trajectories."""
        t, x_batch = batched_trajectory
        fig = plotter.plot_trajectory(t, x_batch)

        assert isinstance(fig, go.Figure)
        # Should have 6 traces (3 batches × 2 states)
        assert len(fig.data) == 6
        assert fig.layout.showlegend is True

    def test_custom_state_names(self, plotter, multi_state_trajectory):
        """Test with custom state names."""
        t, x = multi_state_trajectory
        state_names = ["Position", "Velocity"]
        fig = plotter.plot_trajectory(t, x, state_names=state_names)

        assert isinstance(fig, go.Figure)
        # Check subplot titles contain custom names
        subplot_titles = [anno.text for anno in fig.layout.annotations if hasattr(anno, "text")]
        assert "Position" in subplot_titles
        assert "Velocity" in subplot_titles

    def test_custom_title(self, plotter, simple_trajectory):
        """Test with custom title."""
        t, x = simple_trajectory
        custom_title = "My Custom Trajectory"
        fig = plotter.plot_trajectory(t, x, title=custom_title)

        assert fig.layout.title.text == custom_title

    def test_color_scheme(self, plotter, batched_trajectory):
        """Test with different color schemes."""
        t, x_batch = batched_trajectory

        # Test colorblind safe
        fig = plotter.plot_trajectory(t, x_batch, color_scheme="colorblind_safe")
        assert isinstance(fig, go.Figure)

        # Test tableau
        fig = plotter.plot_trajectory(t, x_batch, color_scheme="tableau")
        assert isinstance(fig, go.Figure)

    def test_theme_application(self, plotter, simple_trajectory):
        """Test theme application."""
        t, x = simple_trajectory

        # Test default theme
        fig_default = plotter.plot_trajectory(t, x, theme="default")
        assert isinstance(fig_default, go.Figure)

        # Test publication theme
        fig_pub = plotter.plot_trajectory(t, x, theme="publication")
        assert isinstance(fig_pub, go.Figure)
        assert fig_pub.layout.font.size == 14  # Publication font size

        # Test dark theme
        fig_dark = plotter.plot_trajectory(t, x, theme="dark")
        assert isinstance(fig_dark, go.Figure)

    def test_default_theme_fallback(self):
        """Test that default_theme is used when theme not specified."""
        plotter = TrajectoryPlotter(default_theme="publication")
        t = np.linspace(0, 10, 100)
        x = np.sin(t)[:, None]

        fig = plotter.plot_trajectory(t, x)
        assert fig.layout.font.size == 14  # Publication theme font

    def test_show_legend_false(self, plotter, batched_trajectory):
        """Test disabling legend."""
        t, x_batch = batched_trajectory
        fig = plotter.plot_trajectory(t, x_batch, show_legend=False)

        assert fig.layout.showlegend is False

    def test_incompatible_time_shape(self, plotter):
        """Test error on incompatible time shape."""
        t = np.linspace(0, 10, 50)  # 50 points
        x = np.random.randn(100, 2)  # 100 points

        with pytest.raises(ValueError, match="Time shape"):
            plotter.plot_trajectory(t, x)

    def test_state_names_length_mismatch(self, plotter, multi_state_trajectory):
        """Test error on state_names length mismatch."""
        t, x = multi_state_trajectory
        state_names = ["Position"]  # Only 1 name for 2 states

        with pytest.raises(ValueError, match="state_names length"):
            plotter.plot_trajectory(t, x, state_names=state_names)

    def test_1d_array_conversion(self, plotter):
        """Test that 1D arrays are converted to 2D."""
        t = np.linspace(0, 10, 100)
        x = np.sin(t)  # 1D array

        fig = plotter.plot_trajectory(t, x)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1


# ============================================================================
# plot_state_and_control Tests
# ============================================================================


class TestPlotStateAndControl:
    """Test plot_state_and_control method."""

    def test_basic_state_and_control(self, plotter, multi_state_trajectory, control_input):
        """Test plotting states and controls."""
        t, x = multi_state_trajectory
        u = control_input

        fig = plotter.plot_state_and_control(t, x, u)

        assert isinstance(fig, go.Figure)
        # Should have traces for 2 states + 1 control
        assert len(fig.data) == 3

    def test_custom_names(self, plotter, multi_state_trajectory, control_input):
        """Test with custom state and control names."""
        t, x = multi_state_trajectory
        u = control_input

        state_names = ["θ", "ω"]
        control_names = ["Torque"]

        fig = plotter.plot_state_and_control(
            t, x, u, state_names=state_names, control_names=control_names,
        )

        assert isinstance(fig, go.Figure)

    def test_batched_state_and_control(self, plotter, batched_trajectory):
        """Test batched states and controls."""
        t, x_batch = batched_trajectory
        u_batch = 0.1 * np.random.randn(3, 100, 1)

        fig = plotter.plot_state_and_control(t, x_batch, u_batch)

        assert isinstance(fig, go.Figure)
        assert fig.layout.showlegend is True

    def test_batching_mismatch_error(self, plotter, batched_trajectory, control_input):
        """Test error when batching doesn't match."""
        t, x_batch = batched_trajectory  # Batched (3, 100, 2)
        u = control_input  # Not batched (100, 1)

        with pytest.raises(ValueError, match="both be batched"):
            plotter.plot_state_and_control(t, x_batch, u)

    def test_theme_application(self, plotter, multi_state_trajectory, control_input):
        """Test theme application."""
        t, x = multi_state_trajectory
        u = control_input

        fig = plotter.plot_state_and_control(t, x, u, theme="publication")

        assert fig.layout.font.size == 14


# ============================================================================
# plot_comparison Tests
# ============================================================================


class TestPlotComparison:
    """Test plot_comparison method."""

    def test_basic_comparison(self, plotter):
        """Test basic trajectory comparison."""
        t = np.linspace(0, 10, 100)
        x1 = np.column_stack([np.sin(t), np.cos(t)])
        x2 = np.column_stack([np.sin(t + 0.5), np.cos(t + 0.5)])

        trajectories = {
            "Trajectory 1": x1,
            "Trajectory 2": x2,
        }

        fig = plotter.plot_comparison(t, trajectories)

        assert isinstance(fig, go.Figure)
        # Should have 4 traces (2 states × 2 trajectories)
        assert len(fig.data) == 4
        assert fig.layout.showlegend is True

    def test_comparison_with_state_names(self, plotter):
        """Test comparison with custom state names."""
        t = np.linspace(0, 10, 100)
        x1 = np.column_stack([np.sin(t), np.cos(t)])
        x2 = np.column_stack([np.sin(t + 0.5), np.cos(t + 0.5)])

        trajectories = {"Controlled": x1, "Uncontrolled": x2}
        state_names = ["Position", "Velocity"]

        fig = plotter.plot_comparison(t, trajectories, state_names=state_names)

        assert isinstance(fig, go.Figure)

    def test_comparison_shape_mismatch(self, plotter):
        """Test error on trajectory shape mismatch."""
        t = np.linspace(0, 10, 100)
        x1 = np.random.randn(100, 2)
        x2 = np.random.randn(100, 3)  # Different number of states

        trajectories = {"Traj1": x1, "Traj2": x2}

        with pytest.raises(ValueError, match="shape"):
            plotter.plot_comparison(t, trajectories)

    def test_comparison_1d_arrays(self, plotter):
        """Test comparison with 1D arrays."""
        t = np.linspace(0, 10, 100)
        x1 = np.sin(t)
        x2 = np.cos(t)

        trajectories = {"Sin": x1, "Cos": x2}

        fig = plotter.plot_comparison(t, trajectories)
        assert isinstance(fig, go.Figure)

    def test_comparison_theme(self, plotter):
        """Test comparison with themes."""
        t = np.linspace(0, 10, 100)
        x1 = np.sin(t)[:, None]
        x2 = np.cos(t)[:, None]

        trajectories = {"A": x1, "B": x2}

        fig = plotter.plot_comparison(t, trajectories, color_scheme="colorblind_safe", theme="dark")

        assert isinstance(fig, go.Figure)


# ============================================================================
# Helper Method Tests
# ============================================================================


class TestHelperMethods:
    """Test internal helper methods."""

    def test_to_numpy_passthrough(self, plotter):
        """Test _to_numpy with NumPy array."""
        arr = np.array([1, 2, 3])
        result = plotter._to_numpy(arr)
        assert isinstance(result, np.ndarray)
        assert np.array_equal(result, arr)

    def test_to_numpy_none(self, plotter):
        """Test _to_numpy with None."""
        result = plotter._to_numpy(None)
        assert result is None

    def test_is_batched_true(self, plotter):
        """Test _is_batched with 3D array."""
        x_batch = np.random.randn(3, 100, 2)
        assert plotter._is_batched(x_batch) is True

    def test_is_batched_false_2d(self, plotter):
        """Test _is_batched with 2D array."""
        x = np.random.randn(100, 2)
        assert plotter._is_batched(x) is False

    def test_is_batched_false_1d(self, plotter):
        """Test _is_batched with 1D array."""
        x = np.random.randn(100)
        assert plotter._is_batched(x) is False

    def test_is_batched_none(self, plotter):
        """Test _is_batched with None."""
        assert plotter._is_batched(None) is False

    def test_determine_layout_single(self, plotter):
        """Test layout determination for single plot."""
        n_rows, n_cols = plotter._determine_layout(1)
        assert (n_rows, n_cols) == (1, 1)

    def test_determine_layout_few(self, plotter):
        """Test layout determination for few plots."""
        # 2 plots: 1 row, 2 cols
        assert plotter._determine_layout(2) == (1, 2)
        # 4 plots: 1 row, 4 cols
        assert plotter._determine_layout(4) == (1, 4)

    def test_determine_layout_medium(self, plotter):
        """Test layout determination for medium number of plots."""
        # 5 plots: 2 rows, 3 cols
        assert plotter._determine_layout(5) == (2, 3)
        # 8 plots: 2 rows, 4 cols
        assert plotter._determine_layout(8) == (2, 4)

    def test_determine_layout_many(self, plotter):
        """Test layout determination for many plots."""
        # 12 plots: 3 rows, 4 cols
        assert plotter._determine_layout(12) == (3, 4)
        # 16 plots: 4 rows, 4 cols
        n_rows, n_cols = plotter._determine_layout(16)
        assert n_rows * n_cols >= 16


# ============================================================================
# Utility Method Tests
# ============================================================================


class TestUtilityMethods:
    """Test static utility methods."""

    def test_list_available_themes(self):
        """Test listing available themes."""
        themes = TrajectoryPlotter.list_available_themes()

        assert isinstance(themes, list)
        assert len(themes) == 4
        assert "default" in themes
        assert "publication" in themes
        assert "dark" in themes
        assert "presentation" in themes

    def test_list_available_color_schemes(self):
        """Test listing available color schemes."""
        schemes = TrajectoryPlotter.list_available_color_schemes()

        assert isinstance(schemes, list)
        assert len(schemes) == 9
        assert "plotly" in schemes
        assert "colorblind_safe" in schemes
        assert "tableau" in schemes
        assert "sequential_blue" in schemes
        assert "diverging_red_blue" in schemes


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests combining multiple features."""

    def test_complete_workflow(self, plotter):
        """Test complete workflow from data to plot."""
        # Generate data
        t = np.linspace(0, 10, 100)
        x = np.column_stack([np.sin(t), np.cos(t)])
        u = 0.1 * np.random.randn(100, 1)

        # Create plots
        fig1 = plotter.plot_trajectory(
            t, x, state_names=["sin", "cos"], color_scheme="colorblind_safe", theme="publication",
        )

        fig2 = plotter.plot_state_and_control(
            t, x, u, state_names=["sin", "cos"], control_names=["noise"], theme="publication",
        )

        # Both should be valid figures
        assert isinstance(fig1, go.Figure)
        assert isinstance(fig2, go.Figure)

        # Both should have publication theme applied
        assert fig1.layout.font.size == 14
        assert fig2.layout.font.size == 14

    def test_batched_workflow(self, plotter):
        """Test workflow with batched trajectories."""
        # Generate batched data
        t = np.linspace(0, 10, 100)
        x_batch = np.stack(
            [np.column_stack([np.sin(t + phi), np.cos(t + phi)]) for phi in np.linspace(0, 1, 5)],
        )

        # Plot with different themes
        fig_default = plotter.plot_trajectory(t, x_batch, theme="default")
        fig_dark = plotter.plot_trajectory(t, x_batch, theme="dark")

        assert isinstance(fig_default, go.Figure)
        assert isinstance(fig_dark, go.Figure)

    def test_theme_consistency(self, plotter):
        """Test theme consistency across different plot types."""
        t = np.linspace(0, 10, 100)
        x = np.sin(t)[:, None]
        u = 0.1 * np.random.randn(100, 1)

        # Create different plot types with same theme
        fig1 = plotter.plot_trajectory(t, x, theme="publication")
        fig2 = plotter.plot_state_and_control(t, x, u, theme="publication")

        trajectories = {"A": x, "B": x * 0.5}
        fig3 = plotter.plot_comparison(t, trajectories, theme="publication")

        # All should have consistent font size
        assert fig1.layout.font.size == 14
        assert fig2.layout.font.size == 14
        assert fig3.layout.font.size == 14


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_time_point(self, plotter):
        """Test with single time point."""
        t = np.array([0.0])
        x = np.array([[1.0]])

        fig = plotter.plot_trajectory(t, x)
        assert isinstance(fig, go.Figure)

    def test_large_batch_size(self, plotter):
        """Test with large batch size."""
        t = np.linspace(0, 10, 100)
        x_batch = np.random.randn(20, 100, 2)  # 20 batches

        fig = plotter.plot_trajectory(t, x_batch)
        assert isinstance(fig, go.Figure)
        # Should have 40 traces (20 batches × 2 states)
        assert len(fig.data) == 40

    def test_many_states(self, plotter):
        """Test with many state variables."""
        t = np.linspace(0, 10, 100)
        x = np.random.randn(100, 10)  # 10 states

        fig = plotter.plot_trajectory(t, x)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 10

    def test_empty_trajectories_dict(self, plotter):
        """Test comparison with empty dict should handle gracefully."""
        t = np.linspace(0, 10, 100)
        trajectories = {}

        # Should handle empty dict without crashing
        # (actual behavior depends on implementation)
        try:
            fig = plotter.plot_comparison(t, trajectories)
        except (ValueError, IndexError, KeyError):
            # Expected to raise an error with empty dict
            pass


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    # Run tests with pytest when executed directly
    pytest.main([__file__, "-v", "--tb=short"])
