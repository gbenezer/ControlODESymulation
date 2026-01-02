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
Unit Tests for Plotting Themes Module

Tests color schemes, plot themes, and color manipulation utilities.
"""

import pytest
import plotly.graph_objects as go

from src.visualization.themes import (
    ColorSchemes,
    PlotThemes,
    hex_to_rgb,
    rgb_to_hex,
    lighten_color,
)


# ============================================================================
# ColorSchemes Tests
# ============================================================================


class TestColorSchemes:
    """Test ColorSchemes class functionality."""

    def test_plotly_colors_count(self):
        """Verify PLOTLY palette has correct number of colors."""
        assert len(ColorSchemes.PLOTLY) == 10

    def test_d3_colors_count(self):
        """Verify D3 palette has correct number of colors."""
        assert len(ColorSchemes.D3) == 10

    def test_colorblind_safe_colors_count(self):
        """Verify COLORBLIND_SAFE palette has correct number of colors."""
        assert len(ColorSchemes.COLORBLIND_SAFE) == 8

    def test_tableau_colors_count(self):
        """Verify TABLEAU palette has correct number of colors."""
        assert len(ColorSchemes.TABLEAU) == 10

    def test_sequential_blue_colors_count(self):
        """Verify SEQUENTIAL_BLUE palette has correct number of colors."""
        assert len(ColorSchemes.SEQUENTIAL_BLUE) == 9

    def test_sequential_green_colors_count(self):
        """Verify SEQUENTIAL_GREEN palette has correct number of colors."""
        assert len(ColorSchemes.SEQUENTIAL_GREEN) == 9

    def test_sequential_orange_colors_count(self):
        """Verify SEQUENTIAL_ORANGE palette has correct number of colors."""
        assert len(ColorSchemes.SEQUENTIAL_ORANGE) == 9

    def test_diverging_red_blue_colors_count(self):
        """Verify DIVERGING_RED_BLUE palette has correct number of colors."""
        assert len(ColorSchemes.DIVERGING_RED_BLUE) == 9

    def test_diverging_purple_green_colors_count(self):
        """Verify DIVERGING_PURPLE_GREEN palette has correct number of colors."""
        assert len(ColorSchemes.DIVERGING_PURPLE_GREEN) == 9

    def test_colors_are_valid_hex(self):
        """Verify all colors are valid hex codes."""
        palettes = [
            ColorSchemes.PLOTLY,
            ColorSchemes.D3,
            ColorSchemes.COLORBLIND_SAFE,
            ColorSchemes.TABLEAU,
            ColorSchemes.SEQUENTIAL_BLUE,
            ColorSchemes.SEQUENTIAL_GREEN,
            ColorSchemes.SEQUENTIAL_ORANGE,
            ColorSchemes.DIVERGING_RED_BLUE,
            ColorSchemes.DIVERGING_PURPLE_GREEN,
        ]
        
        for palette in palettes:
            for color in palette:
                assert color.startswith("#"), f"Color {color} missing #"
                assert len(color) == 7, f"Color {color} wrong length"
                # Verify valid hex characters
                int(color[1:], 16)

    def test_get_colors_default(self):
        """Test get_colors with default parameters."""
        colors = ColorSchemes.get_colors()
        assert len(colors) == len(ColorSchemes.PLOTLY)
        assert colors == ColorSchemes.PLOTLY

    def test_get_colors_plotly(self):
        """Test get_colors with plotly scheme."""
        colors = ColorSchemes.get_colors("plotly")
        assert colors == ColorSchemes.PLOTLY

    def test_get_colors_d3(self):
        """Test get_colors with d3 scheme."""
        colors = ColorSchemes.get_colors("d3")
        assert colors == ColorSchemes.D3

    def test_get_colors_colorblind_safe(self):
        """Test get_colors with colorblind_safe scheme."""
        colors = ColorSchemes.get_colors("colorblind_safe")
        assert colors == ColorSchemes.COLORBLIND_SAFE

    def test_get_colors_wong_alias(self):
        """Test get_colors with wong alias for colorblind_safe."""
        colors = ColorSchemes.get_colors("wong")
        assert colors == ColorSchemes.COLORBLIND_SAFE

    def test_get_colors_tableau(self):
        """Test get_colors with tableau scheme."""
        colors = ColorSchemes.get_colors("tableau")
        assert colors == ColorSchemes.TABLEAU

    def test_get_colors_sequential_blue(self):
        """Test get_colors with sequential_blue scheme."""
        colors = ColorSchemes.get_colors("sequential_blue")
        assert colors == ColorSchemes.SEQUENTIAL_BLUE

    def test_get_colors_sequential_green(self):
        """Test get_colors with sequential_green scheme."""
        colors = ColorSchemes.get_colors("sequential_green")
        assert colors == ColorSchemes.SEQUENTIAL_GREEN

    def test_get_colors_sequential_orange(self):
        """Test get_colors with sequential_orange scheme."""
        colors = ColorSchemes.get_colors("sequential_orange")
        assert colors == ColorSchemes.SEQUENTIAL_ORANGE

    def test_get_colors_diverging_red_blue(self):
        """Test get_colors with diverging_red_blue scheme."""
        colors = ColorSchemes.get_colors("diverging_red_blue")
        assert colors == ColorSchemes.DIVERGING_RED_BLUE

    def test_get_colors_diverging_purple_green(self):
        """Test get_colors with diverging_purple_green scheme."""
        colors = ColorSchemes.get_colors("diverging_purple_green")
        assert colors == ColorSchemes.DIVERGING_PURPLE_GREEN

    def test_get_colors_case_insensitive(self):
        """Test get_colors with various case variations."""
        colors1 = ColorSchemes.get_colors("PLOTLY")
        colors2 = ColorSchemes.get_colors("Plotly")
        colors3 = ColorSchemes.get_colors("plotly")
        assert colors1 == colors2 == colors3 == ColorSchemes.PLOTLY

    def test_get_colors_with_underscores(self):
        """Test get_colors handles underscores, hyphens, and spaces."""
        colors1 = ColorSchemes.get_colors("colorblind_safe")
        colors2 = ColorSchemes.get_colors("colorblind-safe")
        colors3 = ColorSchemes.get_colors("colorblind safe")
        assert colors1 == colors2 == colors3 == ColorSchemes.COLORBLIND_SAFE

    def test_get_colors_n_colors_less_than_palette(self):
        """Test get_colors with n_colors less than palette size."""
        colors = ColorSchemes.get_colors("plotly", n_colors=5)
        assert len(colors) == 5
        assert colors == ColorSchemes.PLOTLY[:5]

    def test_get_colors_n_colors_equal_to_palette(self):
        """Test get_colors with n_colors equal to palette size."""
        colors = ColorSchemes.get_colors("colorblind_safe", n_colors=8)
        assert len(colors) == 8
        assert colors == ColorSchemes.COLORBLIND_SAFE

    def test_get_colors_n_colors_greater_than_palette(self):
        """Test get_colors cycles when n_colors exceeds palette size."""
        colors = ColorSchemes.get_colors("colorblind_safe", n_colors=10)
        assert len(colors) == 10
        # First 8 should match palette
        assert colors[:8] == ColorSchemes.COLORBLIND_SAFE
        # Next 2 should cycle from beginning
        assert colors[8] == ColorSchemes.COLORBLIND_SAFE[0]
        assert colors[9] == ColorSchemes.COLORBLIND_SAFE[1]

    def test_get_colors_n_colors_cycling(self):
        """Test get_colors cycles correctly for large n_colors."""
        colors = ColorSchemes.get_colors("plotly", n_colors=25)
        assert len(colors) == 25
        # Check cycling pattern
        for i in range(25):
            expected_color = ColorSchemes.PLOTLY[i % 10]
            assert colors[i] == expected_color

    def test_get_colors_n_colors_zero(self):
        """Test get_colors with n_colors=0."""
        colors = ColorSchemes.get_colors("plotly", n_colors=0)
        assert len(colors) == 0
        assert colors == []

    def test_get_colors_n_colors_one(self):
        """Test get_colors with n_colors=1."""
        colors = ColorSchemes.get_colors("plotly", n_colors=1)
        assert len(colors) == 1
        assert colors[0] == ColorSchemes.PLOTLY[0]

    def test_get_colors_invalid_scheme(self):
        """Test get_colors raises ValueError for invalid scheme."""
        with pytest.raises(ValueError, match="Unknown color scheme"):
            ColorSchemes.get_colors("invalid_scheme")

    def test_get_colors_returns_copy(self):
        """Test get_colors returns a copy, not reference."""
        colors = ColorSchemes.get_colors("plotly")
        colors[0] = "#FFFFFF"
        # Original should be unchanged
        assert ColorSchemes.PLOTLY[0] != "#FFFFFF"

    def test_interpolate_colors_red_to_blue(self):
        """Test interpolate_colors from red to blue."""
        gradient = ColorSchemes.interpolate_colors("#FF0000", "#0000FF", n_steps=5)
        assert len(gradient) == 5
        assert gradient[0] == "#ff0000"  # Red
        assert gradient[-1] == "#0000ff"  # Blue
        # Middle should be purple-ish
        assert gradient[2] == "#7f007f"

    def test_interpolate_colors_two_steps(self):
        """Test interpolate_colors with n_steps=2 (endpoints only)."""
        gradient = ColorSchemes.interpolate_colors("#FF0000", "#00FF00", n_steps=2)
        assert len(gradient) == 2
        assert gradient[0] == "#ff0000"
        assert gradient[1] == "#00ff00"

    def test_interpolate_colors_one_step(self):
        """Test interpolate_colors with n_steps=1."""
        gradient = ColorSchemes.interpolate_colors("#FF0000", "#0000FF", n_steps=1)
        assert len(gradient) == 1
        assert gradient[0] == "#ff0000"

    def test_interpolate_colors_many_steps(self):
        """Test interpolate_colors with many steps."""
        gradient = ColorSchemes.interpolate_colors("#000000", "#FFFFFF", n_steps=101)
        assert len(gradient) == 101
        assert gradient[0] == "#000000"  # Black
        assert gradient[50] == "#7f7f7f"  # Gray
        assert gradient[100] == "#ffffff"  # White

    def test_interpolate_colors_monotonic(self):
        """Test interpolate_colors produces monotonic gradient."""
        gradient = ColorSchemes.interpolate_colors("#000000", "#FF0000", n_steps=10)
        # Red channel should increase monotonically
        red_values = [int(c[1:3], 16) for c in gradient]
        assert red_values == sorted(red_values)


# ============================================================================
# PlotThemes Tests
# ============================================================================


class TestPlotThemes:
    """Test PlotThemes class functionality."""

    def test_default_theme_keys(self):
        """Verify DEFAULT theme has expected keys."""
        expected_keys = {
            "color_scheme",
            "template",
            "font_family",
            "font_size",
            "line_width",
            "marker_size",
            "grid",
        }
        assert set(PlotThemes.DEFAULT.keys()) == expected_keys

    def test_publication_theme_keys(self):
        """Verify PUBLICATION theme has expected keys."""
        assert "color_scheme" in PlotThemes.PUBLICATION
        assert "template" in PlotThemes.PUBLICATION
        assert "showlegend" in PlotThemes.PUBLICATION

    def test_dark_theme_keys(self):
        """Verify DARK theme has expected keys."""
        assert "color_scheme" in PlotThemes.DARK
        assert "template" in PlotThemes.DARK
        assert PlotThemes.DARK["template"] == "plotly_dark"

    def test_presentation_theme_keys(self):
        """Verify PRESENTATION theme has expected keys."""
        assert "color_scheme" in PlotThemes.PRESENTATION
        assert "font_size" in PlotThemes.PRESENTATION
        # Presentation should have larger fonts
        assert PlotThemes.PRESENTATION["font_size"] > PlotThemes.DEFAULT["font_size"]

    def test_apply_theme_default(self):
        """Test apply_theme with default theme."""
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[1, 2, 3], y=[1, 2, 3]))
        
        result = PlotThemes.apply_theme(fig, theme="default")
        
        assert isinstance(result, go.Figure)
        assert result.layout.template.layout.plot_bgcolor == "white"

    def test_apply_theme_publication(self):
        """Test apply_theme with publication theme."""
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[1, 2, 3], y=[1, 2, 3]))
        
        result = PlotThemes.apply_theme(fig, theme="publication")
        
        assert isinstance(result, go.Figure)
        assert result.layout.font.size == 14
        assert "Times New Roman" in result.layout.font.family

    def test_apply_theme_dark(self):
        """Test apply_theme with dark theme."""
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[1, 2, 3], y=[1, 2, 3]))
        
        result = PlotThemes.apply_theme(fig, theme="dark")
        
        assert isinstance(result, go.Figure)
        # Dark theme should have dark background
        assert result.layout.template.layout.plot_bgcolor != "white"

    def test_apply_theme_presentation(self):
        """Test apply_theme with presentation theme."""
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[1, 2, 3], y=[1, 2, 3]))
        
        result = PlotThemes.apply_theme(fig, theme="presentation")
        
        assert isinstance(result, go.Figure)
        assert result.layout.font.size == 18

    def test_apply_theme_case_insensitive(self):
        """Test apply_theme is case insensitive."""
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=[1, 2, 3], y=[1, 2, 3]))
        result1 = PlotThemes.apply_theme(fig1, theme="DEFAULT")
        
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=[1, 2, 3], y=[1, 2, 3]))
        result2 = PlotThemes.apply_theme(fig2, theme="default")
        
        assert result1.layout.template == result2.layout.template

    def test_apply_theme_custom_dict(self):
        """Test apply_theme with custom theme dictionary."""
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[1, 2, 3], y=[1, 2, 3]))
        
        custom_theme = {
            "template": "plotly_white",
            "font_family": "Helvetica",
            "font_size": 20,
        }
        
        result = PlotThemes.apply_theme(fig, theme=custom_theme)
        
        assert result.layout.font.size == 20
        assert "Helvetica" in result.layout.font.family

    def test_apply_theme_updates_line_width(self):
        """Test apply_theme updates line widths."""
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[1, 2, 3], y=[1, 2, 3], mode="lines"))
        
        result = PlotThemes.apply_theme(fig, theme="publication")
        
        assert result.data[0].line.width == 2.5

    def test_apply_theme_updates_marker_size(self):
        """Test apply_theme updates marker sizes."""
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[1, 2, 3], y=[1, 2, 3], mode="markers"))
        
        result = PlotThemes.apply_theme(fig, theme="presentation")
        
        assert result.data[0].marker.size == 12

    def test_apply_theme_updates_legend(self):
        """Test apply_theme updates legend visibility."""
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[1, 2, 3], y=[1, 2, 3], name="Test"))
        
        result = PlotThemes.apply_theme(fig, theme="publication")
        
        assert result.layout.showlegend is True

    def test_apply_theme_invalid_theme_name(self):
        """Test apply_theme raises ValueError for invalid theme name."""
        fig = go.Figure()
        
        with pytest.raises(ValueError, match="Unknown theme"):
            PlotThemes.apply_theme(fig, theme="invalid_theme")

    def test_apply_theme_invalid_theme_type(self):
        """Test apply_theme raises TypeError for invalid theme type."""
        fig = go.Figure()
        
        with pytest.raises(TypeError, match="theme must be str or dict"):
            PlotThemes.apply_theme(fig, theme=123)

    def test_apply_theme_modifies_in_place(self):
        """Test apply_theme modifies figure in place."""
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[1, 2, 3], y=[1, 2, 3]))
        
        original_id = id(fig)
        result = PlotThemes.apply_theme(fig, theme="default")
        
        assert id(result) == original_id

    def test_apply_theme_empty_figure(self):
        """Test apply_theme works with empty figure."""
        fig = go.Figure()
        
        result = PlotThemes.apply_theme(fig, theme="default")
        
        assert isinstance(result, go.Figure)

    def test_apply_theme_multiple_traces(self):
        """Test apply_theme works with multiple traces."""
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[1, 2], y=[1, 2], mode="lines"))
        fig.add_trace(go.Scatter(x=[1, 2], y=[2, 1], mode="lines"))
        
        result = PlotThemes.apply_theme(fig, theme="publication")
        
        assert result.data[0].line.width == 2.5
        assert result.data[1].line.width == 2.5

    def test_get_line_styles(self):
        """Test get_line_styles returns expected styles."""
        styles = PlotThemes.get_line_styles()
        
        assert isinstance(styles, list)
        assert len(styles) == 6
        assert "solid" in styles
        assert "dot" in styles
        assert "dash" in styles
        assert "longdash" in styles
        assert "dashdot" in styles
        assert "longdashdot" in styles


# ============================================================================
# Color Manipulation Tests
# ============================================================================


class TestColorManipulation:
    """Test color manipulation utility functions."""

    def test_hex_to_rgb_red(self):
        """Test hex_to_rgb with pure red."""
        rgb = hex_to_rgb("#FF0000")
        assert rgb == (255, 0, 0)

    def test_hex_to_rgb_green(self):
        """Test hex_to_rgb with pure green."""
        rgb = hex_to_rgb("#00FF00")
        assert rgb == (0, 255, 0)

    def test_hex_to_rgb_blue(self):
        """Test hex_to_rgb with pure blue."""
        rgb = hex_to_rgb("#0000FF")
        assert rgb == (0, 0, 255)

    def test_hex_to_rgb_white(self):
        """Test hex_to_rgb with white."""
        rgb = hex_to_rgb("#FFFFFF")
        assert rgb == (255, 255, 255)

    def test_hex_to_rgb_black(self):
        """Test hex_to_rgb with black."""
        rgb = hex_to_rgb("#000000")
        assert rgb == (0, 0, 0)

    def test_hex_to_rgb_lowercase(self):
        """Test hex_to_rgb with lowercase hex."""
        rgb = hex_to_rgb("#ff0000")
        assert rgb == (255, 0, 0)

    def test_hex_to_rgb_mixed_case(self):
        """Test hex_to_rgb with mixed case hex."""
        rgb = hex_to_rgb("#FfAa00")
        assert rgb == (255, 170, 0)

    def test_hex_to_rgb_no_hash(self):
        """Test hex_to_rgb without leading hash."""
        rgb = hex_to_rgb("FF0000")
        assert rgb == (255, 0, 0)

    def test_rgb_to_hex_red(self):
        """Test rgb_to_hex with pure red."""
        hex_color = rgb_to_hex(255, 0, 0)
        assert hex_color == "#ff0000"

    def test_rgb_to_hex_green(self):
        """Test rgb_to_hex with pure green."""
        hex_color = rgb_to_hex(0, 255, 0)
        assert hex_color == "#00ff00"

    def test_rgb_to_hex_blue(self):
        """Test rgb_to_hex with pure blue."""
        hex_color = rgb_to_hex(0, 0, 255)
        assert hex_color == "#0000ff"

    def test_rgb_to_hex_white(self):
        """Test rgb_to_hex with white."""
        hex_color = rgb_to_hex(255, 255, 255)
        assert hex_color == "#ffffff"

    def test_rgb_to_hex_black(self):
        """Test rgb_to_hex with black."""
        hex_color = rgb_to_hex(0, 0, 0)
        assert hex_color == "#000000"

    def test_rgb_to_hex_gray(self):
        """Test rgb_to_hex with gray."""
        hex_color = rgb_to_hex(128, 128, 128)
        assert hex_color == "#808080"

    def test_hex_to_rgb_roundtrip(self):
        """Test hex_to_rgb and rgb_to_hex roundtrip conversion."""
        original = "#A3B5C7"
        rgb = hex_to_rgb(original)
        result = rgb_to_hex(*rgb)
        assert result.upper() == original.upper()

    def test_lighten_color_positive(self):
        """Test lighten_color with positive factor (lighten)."""
        result = lighten_color("#808080", factor=0.5)
        # Should move halfway to white
        assert result == "#bfbfbf"

    def test_lighten_color_negative(self):
        """Test lighten_color with negative factor (darken)."""
        result = lighten_color("#808080", factor=-0.5)
        # Should move halfway to black
        assert result == "#404040"

    def test_lighten_color_zero(self):
        """Test lighten_color with zero factor (no change)."""
        original = "#FF0000"
        result = lighten_color(original, factor=0.0)
        assert result.upper() == original.upper()

    def test_lighten_color_max_lighten(self):
        """Test lighten_color with factor=1.0 (white)."""
        result = lighten_color("#000000", factor=1.0)
        assert result == "#ffffff"

    def test_lighten_color_max_darken(self):
        """Test lighten_color with factor=-1.0 (black)."""
        result = lighten_color("#FFFFFF", factor=-1.0)
        assert result == "#000000"

    def test_lighten_color_clamping_upper(self):
        """Test lighten_color clamps to valid range (upper)."""
        # Already bright, lighten further
        result = lighten_color("#F0F0F0", factor=0.5)
        rgb = hex_to_rgb(result)
        # All values should be <= 255
        assert all(v <= 255 for v in rgb)

    def test_lighten_color_clamping_lower(self):
        """Test lighten_color clamps to valid range (lower)."""
        # Already dark, darken further
        result = lighten_color("#0F0F0F", factor=-0.5)
        rgb = hex_to_rgb(result)
        # All values should be >= 0
        assert all(v >= 0 for v in rgb)

    def test_lighten_color_red_channel(self):
        """Test lighten_color affects red channel correctly."""
        original = "#FF0000"
        lightened = lighten_color(original, factor=0.5)
        r, g, b = hex_to_rgb(lightened)
        # Red should move toward 255 (already there)
        assert r == 255
        # Green and blue should move toward 255
        assert g > 0
        assert b > 0

    def test_lighten_color_preserves_ratios(self):
        """Test lighten_color preserves color ratios approximately."""
        original = "#FF8000"  # Orange
        lightened = lighten_color(original, factor=0.3)
        
        r_orig, g_orig, b_orig = hex_to_rgb(original)
        r_light, g_light, b_light = hex_to_rgb(lightened)
        
        # Lightened color should still have more red than green
        assert r_light > g_light
        assert g_light > b_light


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_color_scheme_in_theme(self):
        """Test that theme color schemes are valid."""
        for theme_name, theme_config in [
            ("DEFAULT", PlotThemes.DEFAULT),
            ("PUBLICATION", PlotThemes.PUBLICATION),
            ("DARK", PlotThemes.DARK),
            ("PRESENTATION", PlotThemes.PRESENTATION),
        ]:
            color_scheme = theme_config["color_scheme"]
            # Should not raise
            colors = ColorSchemes.get_colors(color_scheme)
            assert len(colors) > 0

    def test_complete_workflow(self):
        """Test complete workflow: create figure, apply theme, verify colors."""
        # Create figure
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[1, 2, 3], y=[1, 2, 3], mode="lines+markers"))
        
        # Apply theme
        fig = PlotThemes.apply_theme(fig, theme="publication")
        
        # Get colors for legend
        colors = ColorSchemes.get_colors("colorblind_safe", n_colors=3)
        
        # Verify everything works together
        assert isinstance(fig, go.Figure)
        assert len(colors) == 3
        assert fig.layout.font.size == 14

    def test_custom_theme_with_color_manipulation(self):
        """Test custom theme using color manipulation utilities."""
        base_color = "#0000FF"
        light_color = lighten_color(base_color, factor=0.3)
        dark_color = lighten_color(base_color, factor=-0.3)
        
        # Create gradient
        gradient = ColorSchemes.interpolate_colors(dark_color, light_color, n_steps=5)
        
        # Use in custom theme
        custom_theme = PlotThemes.DEFAULT.copy()
        custom_theme["font_size"] = 16
        
        fig = go.Figure()
        for i, color in enumerate(gradient):
            fig.add_trace(
                go.Scatter(
                    x=[1, 2, 3],
                    y=[i, i + 1, i + 2],
                    line=dict(color=color),
                )
            )
        
        fig = PlotThemes.apply_theme(fig, theme=custom_theme)
        
        assert len(fig.data) == 5
        assert fig.layout.font.size == 16


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    # Run tests with pytest when executed directly
    pytest.main([__file__, "-v", "--tb=short"])
