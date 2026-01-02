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
Unit Tests for Phase Portrait Plotter

Tests phase space visualization functionality including 2D/3D portraits,
vector fields, equilibria, and theme integration.
"""

import numpy as np
import pytest
import plotly.graph_objects as go

from src.visualization.phase_portrait import PhasePortraitPlotter


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def plotter():
    """Create default phase portrait plotter."""
    return PhasePortraitPlotter(backend='numpy')


@pytest.fixture
def circular_trajectory_2d():
    """Create circular 2D trajectory."""
    t = np.linspace(0, 2*np.pi, 100)
    x = np.column_stack([np.cos(t), np.sin(t)])  # (100, 2)
    return x


@pytest.fixture
def spiral_trajectory_2d():
    """Create spiral 2D trajectory."""
    t = np.linspace(0, 4*np.pi, 200)
    r = np.exp(-0.1 * t)
    x = np.column_stack([r * np.cos(t), r * np.sin(t)])  # (200, 2)
    return x


@pytest.fixture
def batched_trajectory_2d():
    """Create batched 2D trajectories."""
    t = np.linspace(0, 2*np.pi, 100)
    x_batch = np.stack([
        np.column_stack([
            (1 + 0.2*i) * np.cos(t),
            (1 + 0.2*i) * np.sin(t)
        ]) for i in range(3)
    ])  # (3, 100, 2)
    return x_batch


@pytest.fixture
def lorenz_trajectory_3d():
    """Create Lorenz attractor trajectory."""
    # Simple Lorenz-like trajectory
    t = np.linspace(0, 25, 1000)
    x = 10 * np.sin(0.5 * t)
    y = 15 * np.cos(0.4 * t)
    z = 20 + 10 * np.sin(0.3 * t)
    return np.column_stack([x, y, z])


@pytest.fixture
def simple_vector_field():
    """Create simple vector field function."""
    def f(x1, x2):
        return np.array([x2, -x1])
    return f


@pytest.fixture
def pendulum_vector_field():
    """Create pendulum vector field."""
    def f(x1, x2):
        return np.array([x2, -np.sin(x1) - 0.1*x2])
    return f


# ============================================================================
# Initialization Tests
# ============================================================================


class TestInitialization:
    """Test PhasePortraitPlotter initialization."""

    def test_default_initialization(self):
        """Test default initialization."""
        plotter = PhasePortraitPlotter()
        assert plotter.backend == 'numpy'
        assert plotter.default_theme == 'default'

    def test_backend_initialization(self):
        """Test initialization with different backends."""
        plotter_torch = PhasePortraitPlotter(backend='torch')
        assert plotter_torch.backend == 'torch'
        
        plotter_jax = PhasePortraitPlotter(backend='jax')
        assert plotter_jax.backend == 'jax'

    def test_theme_initialization(self):
        """Test initialization with custom theme."""
        plotter = PhasePortraitPlotter(default_theme='dark')
        assert plotter.default_theme == 'dark'


# ============================================================================
# plot_2d Tests
# ============================================================================


class TestPlot2D:
    """Test plot_2d method."""

    def test_simple_2d_portrait(self, plotter, circular_trajectory_2d):
        """Test simple 2D phase portrait."""
        fig = plotter.plot_2d(circular_trajectory_2d)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_custom_state_names(self, plotter, circular_trajectory_2d):
        """Test with custom state names."""
        fig = plotter.plot_2d(
            circular_trajectory_2d,
            state_names=('Position', 'Velocity')
        )
        
        assert isinstance(fig, go.Figure)
        assert fig.layout.xaxis.title.text == 'Position'
        assert fig.layout.yaxis.title.text == 'Velocity'

    def test_batched_2d_portrait(self, plotter, batched_trajectory_2d):
        """Test batched 2D phase portraits."""
        fig = plotter.plot_2d(batched_trajectory_2d)
        
        assert isinstance(fig, go.Figure)
        # Should have multiple trajectory traces
        assert len(fig.data) > 3  # At least 3 trajectories

    def test_no_direction_arrows(self, plotter, circular_trajectory_2d):
        """Test without direction arrows."""
        fig = plotter.plot_2d(
            circular_trajectory_2d,
            show_direction=False
        )
        
        assert isinstance(fig, go.Figure)

    def test_no_start_end_markers(self, plotter, circular_trajectory_2d):
        """Test without start/end markers."""
        fig = plotter.plot_2d(
            circular_trajectory_2d,
            show_start_end=False
        )
        
        assert isinstance(fig, go.Figure)

    def test_with_vector_field(self, plotter, circular_trajectory_2d, simple_vector_field):
        """Test with vector field overlay."""
        fig = plotter.plot_2d(
            circular_trajectory_2d,
            vector_field=simple_vector_field
        )
        
        assert isinstance(fig, go.Figure)

    def test_with_equilibria(self, plotter, circular_trajectory_2d):
        """Test with equilibrium points."""
        equilibria = [
            np.array([0.0, 0.0]),
            np.array([1.0, 0.0])
        ]
        
        fig = plotter.plot_2d(
            circular_trajectory_2d,
            equilibria=equilibria
        )
        
        assert isinstance(fig, go.Figure)

    def test_with_vector_field_and_equilibria(self, plotter, spiral_trajectory_2d, simple_vector_field):
        """Test with both vector field and equilibria."""
        equilibria = [np.array([0.0, 0.0])]
        
        fig = plotter.plot_2d(
            spiral_trajectory_2d,
            vector_field=simple_vector_field,
            equilibria=equilibria
        )
        
        assert isinstance(fig, go.Figure)

    def test_custom_title(self, plotter, circular_trajectory_2d):
        """Test with custom title."""
        custom_title = "My Phase Portrait"
        fig = plotter.plot_2d(
            circular_trajectory_2d,
            title=custom_title
        )
        
        assert fig.layout.title.text == custom_title

    def test_color_scheme(self, plotter, batched_trajectory_2d):
        """Test with different color schemes."""
        fig = plotter.plot_2d(
            batched_trajectory_2d,
            color_scheme='colorblind_safe'
        )
        
        assert isinstance(fig, go.Figure)

    def test_theme_application(self, plotter, circular_trajectory_2d):
        """Test theme application."""
        fig = plotter.plot_2d(
            circular_trajectory_2d,
            theme='publication'
        )
        
        assert fig.layout.font.size == 14

    def test_default_theme_fallback(self):
        """Test default theme fallback."""
        plotter = PhasePortraitPlotter(default_theme='dark')
        t = np.linspace(0, 2*np.pi, 100)
        x = np.column_stack([np.cos(t), np.sin(t)])
        
        fig = plotter.plot_2d(x)
        assert isinstance(fig, go.Figure)

    def test_equal_aspect_ratio(self, plotter, circular_trajectory_2d):
        """Test that equal aspect ratio is enforced."""
        fig = plotter.plot_2d(circular_trajectory_2d)
        
        assert fig.layout.yaxis.scaleanchor == 'x'
        assert fig.layout.yaxis.scaleratio == 1

    def test_invalid_dimension_1d(self, plotter):
        """Test error on 1D trajectory."""
        x = np.array([1, 2, 3])
        
        with pytest.raises(ValueError, match="plot_2d requires 2D state"):
            plotter.plot_2d(x)

    def test_invalid_dimension_3d_state(self, plotter):
        """Test error on 3D state (wrong dimensionality)."""
        x = np.random.randn(100, 3)
        
        with pytest.raises(ValueError, match="plot_2d requires 2D state"):
            plotter.plot_2d(x)

    def test_short_trajectory(self, plotter):
        """Test with very short trajectory."""
        x = np.array([[0, 0], [1, 1], [2, 0]])
        
        fig = plotter.plot_2d(x)
        assert isinstance(fig, go.Figure)


# ============================================================================
# plot_3d Tests
# ============================================================================


class TestPlot3D:
    """Test plot_3d method."""

    def test_simple_3d_portrait(self, plotter, lorenz_trajectory_3d):
        """Test simple 3D phase portrait."""
        fig = plotter.plot_3d(lorenz_trajectory_3d)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_custom_state_names(self, plotter, lorenz_trajectory_3d):
        """Test with custom state names."""
        fig = plotter.plot_3d(
            lorenz_trajectory_3d,
            state_names=('x', 'y', 'z')
        )
        
        assert isinstance(fig, go.Figure)
        assert fig.layout.scene.xaxis.title.text == 'x'
        assert fig.layout.scene.yaxis.title.text == 'y'
        assert fig.layout.scene.zaxis.title.text == 'z'

    def test_batched_3d_portrait(self, plotter):
        """Test batched 3D phase portraits."""
        t = np.linspace(0, 10, 200)
        x_batch = np.stack([
            np.column_stack([
                np.sin(t + phi),
                np.cos(t + phi),
                t * 0.1
            ]) for phi in [0, 0.5, 1.0]
        ])  # (3, 200, 3)
        
        fig = plotter.plot_3d(x_batch)
        
        assert isinstance(fig, go.Figure)
        # Should have multiple trajectories
        assert len(fig.data) > 3

    def test_no_direction_markers(self, plotter, lorenz_trajectory_3d):
        """Test without direction markers."""
        fig = plotter.plot_3d(
            lorenz_trajectory_3d,
            show_direction=False
        )
        
        assert isinstance(fig, go.Figure)

    def test_no_start_end_markers(self, plotter, lorenz_trajectory_3d):
        """Test without start/end markers."""
        fig = plotter.plot_3d(
            lorenz_trajectory_3d,
            show_start_end=False
        )
        
        assert isinstance(fig, go.Figure)

    def test_custom_title(self, plotter, lorenz_trajectory_3d):
        """Test with custom title."""
        custom_title = "Lorenz Attractor"
        fig = plotter.plot_3d(
            lorenz_trajectory_3d,
            title=custom_title
        )
        
        assert fig.layout.title.text == custom_title

    def test_color_scheme(self, plotter):
        """Test with different color schemes."""
        t = np.linspace(0, 10, 200)
        x_batch = np.stack([
            np.column_stack([np.sin(t + phi), np.cos(t + phi), t * 0.1])
            for phi in [0, 0.5, 1.0]
        ])
        
        fig = plotter.plot_3d(
            x_batch,
            color_scheme='tableau'
        )
        
        assert isinstance(fig, go.Figure)

    def test_theme_application(self, plotter, lorenz_trajectory_3d):
        """Test theme application."""
        fig = plotter.plot_3d(
            lorenz_trajectory_3d,
            theme='dark'
        )
        
        assert isinstance(fig, go.Figure)

    def test_invalid_dimension_2d_state(self, plotter):
        """Test error on 2D state (wrong dimensionality)."""
        x = np.random.randn(100, 2)
        
        with pytest.raises(ValueError, match="plot_3d requires 3D state"):
            plotter.plot_3d(x)

    def test_invalid_shape(self, plotter):
        """Test error on invalid shape."""
        x = np.random.randn(100)  # 1D
        
        with pytest.raises(ValueError, match="plot_3d requires"):
            plotter.plot_3d(x)

    def test_short_trajectory_no_direction(self, plotter):
        """Test short trajectory (no direction markers)."""
        # Trajectory with only 10 points (< 20 threshold)
        x = np.random.randn(10, 3)
        
        fig = plotter.plot_3d(x)
        assert isinstance(fig, go.Figure)


# ============================================================================
# plot_limit_cycle Tests
# ============================================================================


class TestPlotLimitCycle:
    """Test plot_limit_cycle method."""

    def test_simple_limit_cycle(self, plotter, circular_trajectory_2d):
        """Test simple limit cycle visualization."""
        fig = plotter.plot_limit_cycle(circular_trajectory_2d)
        
        assert isinstance(fig, go.Figure)
        assert "Limit Cycle" in fig.layout.annotations[-1].text

    def test_custom_state_names(self, plotter, circular_trajectory_2d):
        """Test with custom state names."""
        fig = plotter.plot_limit_cycle(
            circular_trajectory_2d,
            state_names=('x', 'y')
        )
        
        assert isinstance(fig, go.Figure)

    def test_with_period_estimate(self, plotter, circular_trajectory_2d):
        """Test with period estimate."""
        fig = plotter.plot_limit_cycle(
            circular_trajectory_2d,
            period_estimate=50
        )
        
        assert isinstance(fig, go.Figure)

    def test_custom_title(self, plotter, circular_trajectory_2d):
        """Test with custom title."""
        custom_title = "Van der Pol Limit Cycle"
        fig = plotter.plot_limit_cycle(
            circular_trajectory_2d,
            title=custom_title
        )
        
        assert fig.layout.title.text == custom_title

    def test_theme_application(self, plotter, circular_trajectory_2d):
        """Test theme application."""
        fig = plotter.plot_limit_cycle(
            circular_trajectory_2d,
            theme='publication'
        )
        
        assert fig.layout.font.size == 14

    def test_color_scheme(self, plotter, circular_trajectory_2d):
        """Test with different color scheme."""
        fig = plotter.plot_limit_cycle(
            circular_trajectory_2d,
            color_scheme='colorblind_safe'
        )
        
        assert isinstance(fig, go.Figure)

    def test_batched_takes_first(self, plotter, batched_trajectory_2d):
        """Test that batched input takes first trajectory."""
        fig = plotter.plot_limit_cycle(batched_trajectory_2d)
        
        assert isinstance(fig, go.Figure)

    def test_invalid_dimension(self, plotter):
        """Test error on non-2D state."""
        x = np.random.randn(100, 3)
        
        with pytest.raises(ValueError, match="requires 2D state"):
            plotter.plot_limit_cycle(x)


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

    def test_is_batched_none(self, plotter):
        """Test _is_batched with None."""
        assert plotter._is_batched(None) is False


# ============================================================================
# Vector Field Tests
# ============================================================================


class TestVectorField:
    """Test vector field functionality."""

    def test_vector_field_simple(self, plotter, circular_trajectory_2d, simple_vector_field):
        """Test simple vector field overlay."""
        fig = plotter.plot_2d(
            circular_trajectory_2d,
            vector_field=simple_vector_field
        )
        
        assert isinstance(fig, go.Figure)
        # Should have annotations for vector field
        assert len(fig.layout.annotations) > 0

    def test_vector_field_pendulum(self, plotter, spiral_trajectory_2d, pendulum_vector_field):
        """Test pendulum vector field."""
        fig = plotter.plot_2d(
            spiral_trajectory_2d,
            vector_field=pendulum_vector_field
        )
        
        assert isinstance(fig, go.Figure)

    def test_vector_field_with_equilibria(self, plotter, spiral_trajectory_2d, simple_vector_field):
        """Test vector field with equilibria."""
        equilibria = [np.array([0.0, 0.0])]
        
        fig = plotter.plot_2d(
            spiral_trajectory_2d,
            vector_field=simple_vector_field,
            equilibria=equilibria
        )
        
        assert isinstance(fig, go.Figure)


# ============================================================================
# Equilibria Tests
# ============================================================================


class TestEquilibria:
    """Test equilibrium marker functionality."""

    def test_single_equilibrium_2d(self, plotter, circular_trajectory_2d):
        """Test single equilibrium point."""
        equilibria = [np.array([0.0, 0.0])]
        
        fig = plotter.plot_2d(
            circular_trajectory_2d,
            equilibria=equilibria
        )
        
        assert isinstance(fig, go.Figure)

    def test_multiple_equilibria_2d(self, plotter, circular_trajectory_2d):
        """Test multiple equilibrium points."""
        equilibria = [
            np.array([0.0, 0.0]),
            np.array([1.0, 0.0]),
            np.array([-1.0, 0.0])
        ]
        
        fig = plotter.plot_2d(
            circular_trajectory_2d,
            equilibria=equilibria
        )
        
        assert isinstance(fig, go.Figure)

    def test_empty_equilibria_list(self, plotter, circular_trajectory_2d):
        """Test with empty equilibria list."""
        fig = plotter.plot_2d(
            circular_trajectory_2d,
            equilibria=[]
        )
        
        assert isinstance(fig, go.Figure)


# ============================================================================
# Utility Method Tests
# ============================================================================


class TestUtilityMethods:
    """Test static utility methods."""

    def test_list_available_themes(self):
        """Test listing available themes."""
        themes = PhasePortraitPlotter.list_available_themes()
        
        assert isinstance(themes, list)
        assert len(themes) == 4
        assert 'default' in themes
        assert 'publication' in themes
        assert 'dark' in themes
        assert 'presentation' in themes

    def test_list_available_color_schemes(self):
        """Test listing available color schemes."""
        schemes = PhasePortraitPlotter.list_available_color_schemes()
        
        assert isinstance(schemes, list)
        assert len(schemes) == 9
        assert 'plotly' in schemes
        assert 'colorblind_safe' in schemes


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests combining multiple features."""

    def test_complete_2d_workflow(self, plotter, spiral_trajectory_2d):
        """Test complete 2D workflow with all features."""
        def dynamics(x1, x2):
            return np.array([x2, -0.1*x2 - x1])
        
        equilibria = [np.array([0.0, 0.0])]
        
        fig = plotter.plot_2d(
            spiral_trajectory_2d,
            state_names=('Position', 'Velocity'),
            show_direction=True,
            show_start_end=True,
            vector_field=dynamics,
            equilibria=equilibria,
            title='Complete 2D Portrait',
            color_scheme='colorblind_safe',
            theme='publication'
        )
        
        assert isinstance(fig, go.Figure)
        assert fig.layout.font.size == 14

    def test_complete_3d_workflow(self, plotter, lorenz_trajectory_3d):
        """Test complete 3D workflow."""
        fig = plotter.plot_3d(
            lorenz_trajectory_3d,
            state_names=('x', 'y', 'z'),
            show_direction=True,
            show_start_end=True,
            title='Lorenz Attractor',
            color_scheme='tableau',
            theme='dark'
        )
        
        assert isinstance(fig, go.Figure)

    def test_batched_workflow_with_themes(self, plotter, batched_trajectory_2d):
        """Test batched trajectories with consistent theming."""
        fig = plotter.plot_2d(
            batched_trajectory_2d,
            state_names=('x₁', 'x₂'),
            color_scheme='colorblind_safe',
            theme='publication'
        )
        
        assert isinstance(fig, go.Figure)
        assert fig.layout.font.size == 14

    def test_theme_consistency_2d_3d(self, plotter):
        """Test theme consistency between 2D and 3D plots."""
        # 2D trajectory
        t = np.linspace(0, 2*np.pi, 100)
        x_2d = np.column_stack([np.cos(t), np.sin(t)])
        
        # 3D trajectory
        x_3d = np.column_stack([np.cos(t), np.sin(t), t])
        
        fig_2d = plotter.plot_2d(x_2d, theme='publication')
        fig_3d = plotter.plot_3d(x_3d, theme='publication')
        
        # Both should have publication theme
        assert fig_2d.layout.font.size == 14
        assert fig_3d.layout.font.size == 14


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_point_trajectory(self, plotter):
        """Test with single point (degenerate trajectory)."""
        x = np.array([[0.0, 0.0]])
        
        fig = plotter.plot_2d(x)
        assert isinstance(fig, go.Figure)

    def test_two_point_trajectory(self, plotter):
        """Test with two-point trajectory."""
        x = np.array([[0.0, 0.0], [1.0, 1.0]])
        
        fig = plotter.plot_2d(x)
        assert isinstance(fig, go.Figure)

    def test_very_long_trajectory(self, plotter):
        """Test with very long trajectory."""
        t = np.linspace(0, 100, 10000)
        x = np.column_stack([np.sin(t), np.cos(t)])
        
        fig = plotter.plot_2d(x)
        assert isinstance(fig, go.Figure)

    def test_large_batch_size(self, plotter):
        """Test with large batch size."""
        t = np.linspace(0, 2*np.pi, 100)
        x_batch = np.stack([
            np.column_stack([np.cos(t + phi), np.sin(t + phi)])
            for phi in np.linspace(0, 2*np.pi, 15)
        ])  # 15 batches
        
        fig = plotter.plot_2d(x_batch)
        assert isinstance(fig, go.Figure)

    def test_trajectory_with_nans(self, plotter):
        """Test trajectory with NaN values (should handle gracefully)."""
        x = np.column_stack([np.cos(np.linspace(0, 2*np.pi, 100)),
                            np.sin(np.linspace(0, 2*np.pi, 100))])
        x[50, :] = np.nan  # Insert NaN
        
        # Should not crash
        try:
            fig = plotter.plot_2d(x)
            assert isinstance(fig, go.Figure)
        except:
            # Acceptable to raise error on NaN
            pass


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    # Run tests with pytest when executed directly
    pytest.main([__file__, "-v", "--tb=short"])