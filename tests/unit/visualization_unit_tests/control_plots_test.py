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
Unit Tests for Control Plotter

Tests control system visualization functionality including eigenvalue maps,
gain comparison, Riccati convergence, Gramians, and frequency response plots.
"""

import numpy as np
import plotly.graph_objects as go
import pytest

from src.visualization.control_plots import ControlPlotter

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def plotter():
    """Create default control plotter."""
    return ControlPlotter(backend="numpy")


@pytest.fixture
def stable_eigenvalues_continuous():
    """Create stable continuous-time eigenvalues."""
    return np.array([-1.0 + 2.0j, -1.0 - 2.0j, -2.0, -3.0])


@pytest.fixture
def unstable_eigenvalues_continuous():
    """Create unstable continuous-time eigenvalues."""
    return np.array([0.5 + 1.0j, 0.5 - 1.0j, -2.0])


@pytest.fixture
def stable_eigenvalues_discrete():
    """Create stable discrete-time eigenvalues."""
    return np.array([0.5 + 0.3j, 0.5 - 0.3j, 0.7, 0.4])


@pytest.fixture
def unstable_eigenvalues_discrete():
    """Create unstable discrete-time eigenvalues."""
    return np.array([1.2 + 0.5j, 1.2 - 0.5j, 0.8])


@pytest.fixture
def lqr_gains():
    """Create sample LQR gains for comparison."""
    return {
        "Q=10*I": np.array([[1.5, 2.0]]),
        "Q=100*I": np.array([[4.5, 6.0]]),
        "Q=1000*I": np.array([[14.2, 18.5]]),
    }


@pytest.fixture
def riccati_history():
    """Create sample Riccati convergence history."""
    # Simulate convergence to solution
    P_final = np.array([[10.0, 2.0], [2.0, 5.0]])
    history = []
    for i in range(20):
        # Exponential convergence
        factor = 1 - np.exp(-i / 3)
        P = P_final * factor + np.eye(2) * (1 - factor)
        history.append(P)
    return history


@pytest.fixture
def controllability_gramian():
    """Create sample controllability Gramian."""
    return np.array([[5.0, 1.0], [1.0, 3.0]])


@pytest.fixture
def step_response_data():
    """Create sample step response data."""
    t = np.linspace(0, 10, 200)
    # Underdamped second-order system
    zeta = 0.5
    omega_n = 2.0
    y = 1 - np.exp(-zeta * omega_n * t) * (
        np.cos(omega_n * np.sqrt(1 - zeta**2) * t)
        + (zeta / np.sqrt(1 - zeta**2)) * np.sin(omega_n * np.sqrt(1 - zeta**2) * t)
    )
    return t, y


@pytest.fixture
def frequency_response_data():
    """Create sample frequency response data."""
    w = np.logspace(-2, 2, 100)
    # Simple first-order system: G(s) = 1/(s+1)
    H = 1 / (1j * w + 1)
    mag_dB = 20 * np.log10(np.abs(H))
    phase_deg = np.angle(H, deg=True)
    return w, mag_dB, phase_deg


# ============================================================================
# Initialization Tests
# ============================================================================


class TestInitialization:
    """Test ControlPlotter initialization."""

    def test_default_initialization(self):
        """Test default initialization."""
        plotter = ControlPlotter()
        assert plotter.backend == "numpy"
        assert plotter.default_theme == "default"

    def test_backend_initialization(self):
        """Test initialization with different backends."""
        plotter = ControlPlotter(backend="torch")
        assert plotter.backend == "torch"

    def test_theme_initialization(self):
        """Test initialization with custom theme."""
        plotter = ControlPlotter(default_theme="publication")
        assert plotter.default_theme == "publication"


# ============================================================================
# plot_eigenvalue_map Tests
# ============================================================================


class TestPlotEigenvalueMap:
    """Test plot_eigenvalue_map method."""

    def test_continuous_stable(self, plotter, stable_eigenvalues_continuous):
        """Test continuous stable eigenvalues."""
        fig = plotter.plot_eigenvalue_map(stable_eigenvalues_continuous, system_type="continuous")

        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_continuous_unstable(self, plotter, unstable_eigenvalues_continuous):
        """Test continuous unstable eigenvalues."""
        fig = plotter.plot_eigenvalue_map(
            unstable_eigenvalues_continuous,
            system_type="continuous",
            show_stability_margin=True,
        )

        assert isinstance(fig, go.Figure)

    def test_discrete_stable(self, plotter, stable_eigenvalues_discrete):
        """Test discrete stable eigenvalues."""
        fig = plotter.plot_eigenvalue_map(stable_eigenvalues_discrete, system_type="discrete")

        assert isinstance(fig, go.Figure)

    def test_discrete_unstable(self, plotter, unstable_eigenvalues_discrete):
        """Test discrete unstable eigenvalues."""
        fig = plotter.plot_eigenvalue_map(unstable_eigenvalues_discrete, system_type="discrete")

        assert isinstance(fig, go.Figure)

    def test_custom_title(self, plotter, stable_eigenvalues_continuous):
        """Test with custom title."""
        custom_title = "My Eigenvalue Map"
        fig = plotter.plot_eigenvalue_map(stable_eigenvalues_continuous, title=custom_title)

        assert fig.layout.title.text == custom_title

    def test_no_stability_margin(self, plotter, stable_eigenvalues_continuous):
        """Test without stability margin annotation."""
        fig = plotter.plot_eigenvalue_map(
            stable_eigenvalues_continuous,
            show_stability_margin=False,
        )

        assert isinstance(fig, go.Figure)

    def test_theme_application(self, plotter, stable_eigenvalues_continuous):
        """Test theme application."""
        fig = plotter.plot_eigenvalue_map(stable_eigenvalues_continuous, theme="publication")

        assert fig.layout.font.size == 14

    def test_equal_aspect_ratio(self, plotter, stable_eigenvalues_continuous):
        """Test that equal aspect ratio is applied."""
        fig = plotter.plot_eigenvalue_map(stable_eigenvalues_continuous)

        # Check that yaxis has scaleanchor set to 'x'
        assert fig.layout.yaxis.scaleanchor == "x"
        assert fig.layout.yaxis.scaleratio == 1


# ============================================================================
# plot_gain_comparison Tests
# ============================================================================


class TestPlotGainComparison:
    """Test plot_gain_comparison method."""

    def test_single_input_gains(self, plotter, lqr_gains):
        """Test comparison of single-input gains."""
        fig = plotter.plot_gain_comparison(lqr_gains)

        assert isinstance(fig, go.Figure)
        # Should have 3 bar traces (one per gain)
        assert len(fig.data) == 3

    def test_multi_input_gains(self, plotter):
        """Test comparison of multi-input gains."""
        gains = {
            "Design A": np.array([[1.0, 2.0], [3.0, 4.0]]),
            "Design B": np.array([[1.5, 2.5], [3.5, 4.5]]),
        }

        fig = plotter.plot_gain_comparison(gains)

        assert isinstance(fig, go.Figure)

    def test_custom_labels(self, plotter, lqr_gains):
        """Test with custom state labels."""
        labels = ["Position", "Velocity"]
        fig = plotter.plot_gain_comparison(lqr_gains, labels=labels)

        assert isinstance(fig, go.Figure)

    def test_labels_length_mismatch(self, plotter, lqr_gains):
        """Test error on labels length mismatch."""
        labels = ["Position"]  # Only 1 label for 2 states

        with pytest.raises(ValueError, match="labels length"):
            plotter.plot_gain_comparison(lqr_gains, labels=labels)

    def test_gain_shape_mismatch(self, plotter):
        """Test error on gain shape mismatch."""
        gains = {
            "Gain1": np.array([[1.0, 2.0]]),
            "Gain2": np.array([[1.0, 2.0, 3.0]]),  # Different shape
        }

        with pytest.raises(ValueError, match="shape"):
            plotter.plot_gain_comparison(gains)

    def test_color_scheme(self, plotter, lqr_gains):
        """Test with different color schemes."""
        fig = plotter.plot_gain_comparison(lqr_gains, color_scheme="colorblind_safe")

        assert isinstance(fig, go.Figure)

    def test_theme_application(self, plotter, lqr_gains):
        """Test theme application."""
        fig = plotter.plot_gain_comparison(lqr_gains, theme="dark")

        assert isinstance(fig, go.Figure)

    def test_1d_gain_conversion(self, plotter):
        """Test that 1D gains are converted to 2D."""
        gains = {
            "Gain1": np.array([1.0, 2.0]),
            "Gain2": np.array([1.5, 2.5]),
        }

        fig = plotter.plot_gain_comparison(gains)
        assert isinstance(fig, go.Figure)


# ============================================================================
# plot_riccati_convergence Tests
# ============================================================================


class TestPlotRiccatiConvergence:
    """Test plot_riccati_convergence method."""

    def test_basic_convergence(self, plotter, riccati_history):
        """Test basic Riccati convergence plot."""
        fig = plotter.plot_riccati_convergence(riccati_history)

        assert isinstance(fig, go.Figure)
        # Should have 2 traces (norm and difference)
        assert len(fig.data) == 2

    def test_custom_title(self, plotter, riccati_history):
        """Test with custom title."""
        custom_title = "My Convergence Plot"
        fig = plotter.plot_riccati_convergence(riccati_history, title=custom_title)

        assert fig.layout.title.text == custom_title

    def test_theme_application(self, plotter, riccati_history):
        """Test theme application."""
        fig = plotter.plot_riccati_convergence(riccati_history, theme="publication")

        assert fig.layout.font.size == 14

    def test_single_iteration(self, plotter):
        """Test with single iteration."""
        P_history = [np.eye(2)]
        fig = plotter.plot_riccati_convergence(P_history)

        assert isinstance(fig, go.Figure)

    def test_many_iterations(self, plotter):
        """Test with many iterations."""
        P_history = [np.eye(2) * (1 - np.exp(-i / 10)) for i in range(100)]
        fig = plotter.plot_riccati_convergence(P_history)

        assert isinstance(fig, go.Figure)


# ============================================================================
# plot_controllability_gramian Tests
# ============================================================================


class TestPlotControllabilityGramian:
    """Test plot_controllability_gramian method."""

    def test_basic_gramian(self, plotter, controllability_gramian):
        """Test basic Gramian plot."""
        fig = plotter.plot_controllability_gramian(controllability_gramian)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1  # Heatmap

    def test_custom_state_names(self, plotter, controllability_gramian):
        """Test with custom state names."""
        state_names = ["x", "y"]
        fig = plotter.plot_controllability_gramian(controllability_gramian, state_names=state_names)

        assert isinstance(fig, go.Figure)

    def test_state_names_mismatch(self, plotter, controllability_gramian):
        """Test error on state_names length mismatch."""
        state_names = ["x"]  # Only 1 name for 2x2 matrix

        with pytest.raises(ValueError, match="state_names length"):
            plotter.plot_controllability_gramian(controllability_gramian, state_names=state_names)

    def test_non_square_gramian(self, plotter):
        """Test error on non-square Gramian."""
        W = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        with pytest.raises(ValueError, match="square"):
            plotter.plot_controllability_gramian(W)

    def test_large_gramian(self, plotter):
        """Test with larger Gramian."""
        W = np.random.randn(5, 5)
        W = W @ W.T  # Make symmetric positive semi-definite

        fig = plotter.plot_controllability_gramian(W)
        assert isinstance(fig, go.Figure)

    def test_theme_application(self, plotter, controllability_gramian):
        """Test theme application."""
        fig = plotter.plot_controllability_gramian(controllability_gramian, theme="dark")

        assert isinstance(fig, go.Figure)


# ============================================================================
# plot_observability_gramian Tests
# ============================================================================


class TestPlotObservabilityGramian:
    """Test plot_observability_gramian method."""

    def test_basic_observability_gramian(self, plotter, controllability_gramian):
        """Test observability Gramian (uses controllability plotter)."""
        fig = plotter.plot_observability_gramian(
            controllability_gramian,
            title="Observability Gramian",
        )

        assert isinstance(fig, go.Figure)
        assert fig.layout.title.text == "Observability Gramian"


# ============================================================================
# plot_step_response Tests
# ============================================================================


class TestPlotStepResponse:
    """Test plot_step_response method."""

    def test_basic_step_response(self, plotter, step_response_data):
        """Test basic step response plot."""
        t, y = step_response_data
        fig = plotter.plot_step_response(t, y)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 2  # Response + reference

    def test_no_metrics(self, plotter, step_response_data):
        """Test without performance metrics."""
        t, y = step_response_data
        fig = plotter.plot_step_response(t, y, show_metrics=False)

        assert isinstance(fig, go.Figure)

    def test_custom_reference(self, plotter, step_response_data):
        """Test with custom reference value."""
        t, y = step_response_data
        fig = plotter.plot_step_response(t, y, reference=2.0)

        assert isinstance(fig, go.Figure)

    def test_multidimensional_output(self, plotter):
        """Test with multidimensional output (takes first)."""
        t = np.linspace(0, 10, 100)
        y = np.column_stack([np.ones(100), np.ones(100) * 0.5])

        fig = plotter.plot_step_response(t, y)
        assert isinstance(fig, go.Figure)

    def test_theme_application(self, plotter, step_response_data):
        """Test theme application."""
        t, y = step_response_data
        fig = plotter.plot_step_response(t, y, theme="publication")

        assert fig.layout.font.size == 14


# ============================================================================
# plot_impulse_response Tests
# ============================================================================


class TestPlotImpulseResponse:
    """Test plot_impulse_response method."""

    def test_basic_impulse_response(self, plotter):
        """Test basic impulse response plot."""
        t = np.linspace(0, 10, 200)
        y = np.exp(-t) * np.sin(2 * np.pi * t)

        fig = plotter.plot_impulse_response(t, y)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 2  # Response + zero line

    def test_no_metrics(self, plotter):
        """Test without performance metrics."""
        t = np.linspace(0, 10, 200)
        y = np.exp(-t)

        fig = plotter.plot_impulse_response(t, y, show_metrics=False)

        assert isinstance(fig, go.Figure)

    def test_decaying_response(self, plotter):
        """Test with decaying response (should compute decay rate)."""
        t = np.linspace(0, 10, 200)
        y = np.exp(-0.5 * t)

        fig = plotter.plot_impulse_response(t, y, show_metrics=True)

        assert isinstance(fig, go.Figure)

    def test_theme_application(self, plotter):
        """Test theme application."""
        t = np.linspace(0, 10, 200)
        y = np.exp(-t)

        fig = plotter.plot_impulse_response(t, y, theme="dark")

        assert isinstance(fig, go.Figure)


# ============================================================================
# plot_frequency_response Tests
# ============================================================================


class TestPlotFrequencyResponse:
    """Test plot_frequency_response method."""

    def test_basic_frequency_response(self, plotter, frequency_response_data):
        """Test basic frequency response (Bode) plot."""
        w, mag_dB, phase_deg = frequency_response_data
        fig = plotter.plot_frequency_response(w, mag_dB, phase_deg)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 2  # Magnitude + phase

    def test_no_margins(self, plotter, frequency_response_data):
        """Test without margin annotations."""
        w, mag_dB, phase_deg = frequency_response_data
        fig = plotter.plot_frequency_response(w, mag_dB, phase_deg, show_margins=False)

        assert isinstance(fig, go.Figure)

    def test_custom_title(self, plotter, frequency_response_data):
        """Test with custom title."""
        w, mag_dB, phase_deg = frequency_response_data
        custom_title = "My Bode Plot"

        fig = plotter.plot_frequency_response(w, mag_dB, phase_deg, title=custom_title)

        assert fig.layout.title.text == custom_title

    def test_theme_application(self, plotter, frequency_response_data):
        """Test theme application."""
        w, mag_dB, phase_deg = frequency_response_data
        fig = plotter.plot_frequency_response(w, mag_dB, phase_deg, theme="publication")

        assert fig.layout.font.size == 14


# ============================================================================
# plot_nyquist Tests
# ============================================================================


class TestPlotNyquist:
    """Test plot_nyquist method."""

    def test_basic_nyquist(self, plotter):
        """Test basic Nyquist plot."""
        w = np.logspace(-2, 2, 100)
        H = 1 / (1j * w + 1)

        fig = plotter.plot_nyquist(np.real(H), np.imag(H))

        assert isinstance(fig, go.Figure)
        # Should have positive freq, negative freq, critical point, circle
        assert len(fig.data) >= 4

    def test_no_critical_point(self, plotter):
        """Test without critical point marker."""
        w = np.logspace(-2, 2, 100)
        H = 1 / (1j * w + 1)

        fig = plotter.plot_nyquist(np.real(H), np.imag(H), show_critical_point=False)

        assert isinstance(fig, go.Figure)
        # Should have fewer traces without critical point
        assert len(fig.data) == 2  # Just positive and negative freq

    def test_with_frequencies(self, plotter):
        """Test with frequency array for hover info."""
        w = np.logspace(-2, 2, 100)
        H = 1 / (1j * w + 1)

        fig = plotter.plot_nyquist(np.real(H), np.imag(H), frequencies=w)

        assert isinstance(fig, go.Figure)

    def test_equal_aspect_ratio(self, plotter):
        """Test that equal aspect ratio is applied."""
        w = np.logspace(-2, 2, 100)
        H = 1 / (1j * w + 1)

        fig = plotter.plot_nyquist(np.real(H), np.imag(H))

        assert fig.layout.yaxis.scaleanchor == "x"
        assert fig.layout.yaxis.scaleratio == 1

    def test_theme_application(self, plotter):
        """Test theme application."""
        w = np.logspace(-2, 2, 100)
        H = 1 / (1j * w + 1)

        fig = plotter.plot_nyquist(np.real(H), np.imag(H), theme="dark")

        assert isinstance(fig, go.Figure)


# ============================================================================
# plot_root_locus Tests
# ============================================================================


class TestPlotRootLocus:
    """Test plot_root_locus method."""

    def test_basic_root_locus(self, plotter):
        """Test basic root locus plot."""
        gains = np.logspace(-1, 2, 20)
        # Simple pole migration
        poles = np.array([[-1 - k, -1 + k] for k in np.linspace(0, 2, 20)])

        root_locus_data = {"gains": gains, "poles": poles}

        fig = plotter.plot_root_locus(root_locus_data)

        assert isinstance(fig, go.Figure)

    def test_with_zeros(self, plotter):
        """Test root locus with zeros."""
        gains = np.logspace(-1, 2, 20)
        poles = np.array([[-1 - k, -1 + k] for k in np.linspace(0, 2, 20)])
        zeros = np.array([-0.5])

        root_locus_data = {"gains": gains, "poles": poles, "zeros": zeros}

        fig = plotter.plot_root_locus(root_locus_data)

        assert isinstance(fig, go.Figure)

    def test_discrete_system(self, plotter):
        """Test root locus for discrete system."""
        gains = np.logspace(-1, 2, 20)
        poles = np.array([[0.9 - 0.01 * k, 0.8 + 0.01 * k] for k in range(20)])

        root_locus_data = {"gains": gains, "poles": poles}

        fig = plotter.plot_root_locus(root_locus_data, system_type="discrete")

        assert isinstance(fig, go.Figure)

    def test_no_grid(self, plotter):
        """Test without stability grid."""
        gains = np.logspace(-1, 2, 20)
        poles = np.array([[-1 - k, -1 + k] for k in np.linspace(0, 2, 20)])

        root_locus_data = {"gains": gains, "poles": poles}

        fig = plotter.plot_root_locus(root_locus_data, show_grid=False)

        assert isinstance(fig, go.Figure)

    def test_color_scheme(self, plotter):
        """Test with different color schemes."""
        gains = np.logspace(-1, 2, 20)
        poles = np.array([[-1 - k, -1 + k, -2 + 0.5 * k] for k in np.linspace(0, 2, 20)])

        root_locus_data = {"gains": gains, "poles": poles}

        fig = plotter.plot_root_locus(root_locus_data, color_scheme="colorblind_safe")

        assert isinstance(fig, go.Figure)

    def test_invalid_poles_shape(self, plotter):
        """Test error on invalid poles shape."""
        gains = np.logspace(-1, 2, 20)
        poles = np.array([-1, -2, -3])  # 1D array

        root_locus_data = {"gains": gains, "poles": poles}

        with pytest.raises(ValueError, match="2D"):
            plotter.plot_root_locus(root_locus_data)

    def test_theme_application(self, plotter):
        """Test theme application."""
        gains = np.logspace(-1, 2, 20)
        poles = np.array([[-1 - k, -1 + k] for k in np.linspace(0, 2, 20)])

        root_locus_data = {"gains": gains, "poles": poles}

        fig = plotter.plot_root_locus(root_locus_data, theme="publication")

        assert fig.layout.font.size == 14


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


# ============================================================================
# Utility Method Tests
# ============================================================================


class TestUtilityMethods:
    """Test static utility methods."""

    def test_list_available_themes(self):
        """Test listing available themes."""
        themes = ControlPlotter.list_available_themes()

        assert isinstance(themes, list)
        assert len(themes) == 4
        assert "default" in themes
        assert "publication" in themes
        assert "dark" in themes
        assert "presentation" in themes

    def test_list_available_color_schemes(self):
        """Test listing available color schemes."""
        schemes = ControlPlotter.list_available_color_schemes()

        assert isinstance(schemes, list)
        assert len(schemes) == 9
        assert "plotly" in schemes
        assert "colorblind_safe" in schemes


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests combining multiple features."""

    def test_complete_control_analysis(self, plotter):
        """Test complete control system analysis workflow."""
        # Eigenvalues
        eigs = np.array([-1.0 + 2.0j, -1.0 - 2.0j, -2.0])
        fig1 = plotter.plot_eigenvalue_map(eigs, theme="publication")

        # Gains
        gains = {
            "Q=10": np.array([[1.0, 2.0]]),
            "Q=100": np.array([[3.0, 4.0]]),
        }
        fig2 = plotter.plot_gain_comparison(gains, theme="publication")

        # Step response
        t = np.linspace(0, 10, 100)
        y = 1 - np.exp(-t)
        fig3 = plotter.plot_step_response(t, y, theme="publication")

        # All should be valid figures with consistent theme
        assert all(isinstance(f, go.Figure) for f in [fig1, fig2, fig3])
        assert all(f.layout.font.size == 14 for f in [fig1, fig2, fig3])

    def test_theme_consistency(self, plotter, stable_eigenvalues_continuous):
        """Test theme consistency across different plot types."""
        # Create different plot types with same theme
        fig1 = plotter.plot_eigenvalue_map(stable_eigenvalues_continuous, theme="dark")

        t = np.linspace(0, 10, 100)
        y = 1 - np.exp(-t)
        fig2 = plotter.plot_step_response(t, y, theme="dark")

        # Both should have dark theme applied
        assert isinstance(fig1, go.Figure)
        assert isinstance(fig2, go.Figure)


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_eigenvalue(self, plotter):
        """Test with single eigenvalue."""
        eigs = np.array([-1.0])
        fig = plotter.plot_eigenvalue_map(eigs)

        assert isinstance(fig, go.Figure)

    def test_many_eigenvalues(self, plotter):
        """Test with many eigenvalues."""
        eigs = np.random.randn(50) - 1.0 + 1j * np.random.randn(50)
        fig = plotter.plot_eigenvalue_map(eigs)

        assert isinstance(fig, go.Figure)

    def test_short_time_series(self, plotter):
        """Test with very short time series."""
        t = np.array([0.0, 1.0])
        y = np.array([0.0, 1.0])

        fig = plotter.plot_step_response(t, y)
        assert isinstance(fig, go.Figure)


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    # Run tests with pytest when executed directly
    pytest.main([__file__, "-v", "--tb=short"])
