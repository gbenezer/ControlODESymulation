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
Plotting Themes and Color Schemes

Centralized color palettes and styling configuration for consistent visualization.

Key Features
------------
- Predefined color schemes: Plotly, D3, ColorblindSafe, Sequential, Diverging
- Complete plot themes: Default, Publication, Dark
- Color manipulation utilities: Lighten, darken, interpolate
- Easy customization and extension

Main Classes
------------
ColorSchemes : Color palette definitions
    PLOTLY : Default Plotly colors
    D3 : D3.js categorical colors
    COLORBLIND_SAFE : Wong palette (colorblind accessible)
    SEQUENTIAL_BLUE : Blue gradient for heatmaps
    DIVERGING_RED_BLUE : Red-blue diverging scale

PlotThemes : Complete theme configurations
    DEFAULT : Standard Plotly white theme
    PUBLICATION : Publication-ready styling
    DARK : Dark mode theme

Usage
-----
>>> from src.plotting.themes import ColorSchemes, PlotThemes
>>> 
>>> # Get color palette
>>> colors = ColorSchemes.get_colors('colorblind_safe', n_colors=5)
>>> 
>>> # Apply theme to figure
>>> fig = plotter.plot_trajectory(t, x)
>>> fig = PlotThemes.apply_theme(fig, theme='publication')
>>> fig.show()
"""

from typing import List, Optional

import plotly.graph_objects as go


class ColorSchemes:
    """
    Predefined color palettes for plotting.

    Provides various color schemes suitable for different visualization types
    and accessibility requirements.

    Attributes
    ----------
    PLOTLY : List[str]
        Default Plotly color sequence (10 colors)
    D3 : List[str]
        D3.js Category10 colors (10 colors)
    COLORBLIND_SAFE : List[str]
        Wong palette - colorblind accessible (8 colors)
    TABLEAU : List[str]
        Tableau 10 color palette (10 colors)
    SEQUENTIAL_BLUE : List[str]
        Blue sequential scale for heatmaps (9 colors)
    SEQUENTIAL_GREEN : List[str]
        Green sequential scale (9 colors)
    DIVERGING_RED_BLUE : List[str]
        Red-blue diverging scale (9 colors)
    DIVERGING_PURPLE_GREEN : List[str]
        Purple-green diverging scale (9 colors)

    Examples
    --------
    >>> # Get Plotly colors
    >>> colors = ColorSchemes.PLOTLY
    >>> print(colors[0])  # '#636EFA'
    >>>
    >>> # Get colorblind-safe palette
    >>> colors = ColorSchemes.get_colors('colorblind_safe', n_colors=3)
    """

    # Categorical color schemes (for distinct categories)
    PLOTLY = [
        "#636EFA",  # Blue
        "#EF553B",  # Red
        "#00CC96",  # Green
        "#AB63FA",  # Purple
        "#FFA15A",  # Orange
        "#19D3F3",  # Cyan
        "#FF6692",  # Pink
        "#B6E880",  # Light green
        "#FF97FF",  # Light purple
        "#FECB52",  # Yellow
    ]

    D3 = [
        "#1f77b4",  # Blue
        "#ff7f0e",  # Orange
        "#2ca02c",  # Green
        "#d62728",  # Red
        "#9467bd",  # Purple
        "#8c564b",  # Brown
        "#e377c2",  # Pink
        "#7f7f7f",  # Gray
        "#bcbd22",  # Yellow-green
        "#17becf",  # Cyan
    ]

    COLORBLIND_SAFE = [
        "#0173B2",  # Blue
        "#DE8F05",  # Orange
        "#029E73",  # Green
        "#CC78BC",  # Pink
        "#CA9161",  # Tan
        "#949494",  # Gray
        "#ECE133",  # Yellow
        "#56B4E9",  # Sky blue
    ]

    TABLEAU = [
        "#4E79A7",  # Blue
        "#F28E2B",  # Orange
        "#E15759",  # Red
        "#76B7B2",  # Teal
        "#59A14F",  # Green
        "#EDC948",  # Yellow
        "#B07AA1",  # Purple
        "#FF9DA7",  # Pink
        "#9C755F",  # Brown
        "#BAB0AC",  # Gray
    ]

    # Sequential color schemes (for continuous/ordered data)
    SEQUENTIAL_BLUE = [
        "#f7fbff",  # Lightest
        "#deebf7",
        "#c6dbef",
        "#9ecae1",
        "#6baed6",
        "#4292c6",
        "#2171b5",
        "#08519c",
        "#08306b",  # Darkest
    ]

    SEQUENTIAL_GREEN = [
        "#f7fcf5",
        "#e5f5e0",
        "#c7e9c0",
        "#a1d99b",
        "#74c476",
        "#41ab5d",
        "#238b45",
        "#006d2c",
        "#00441b",
    ]

    SEQUENTIAL_ORANGE = [
        "#fff5eb",
        "#fee6ce",
        "#fdd0a2",
        "#fdae6b",
        "#fd8d3c",
        "#f16913",
        "#d94801",
        "#a63603",
        "#7f2704",
    ]

    # Diverging color schemes (for data with meaningful center)
    DIVERGING_RED_BLUE = [
        "#b2182b",  # Dark red
        "#d6604d",
        "#f4a582",
        "#fddbc7",
        "#f7f7f7",  # White center
        "#d1e5f0",
        "#92c5de",
        "#4393c3",
        "#2166ac",  # Dark blue
    ]

    DIVERGING_PURPLE_GREEN = [
        "#762a83",  # Dark purple
        "#9970ab",
        "#c2a5cf",
        "#e7d4e8",
        "#f7f7f7",  # White center
        "#d9f0d3",
        "#a6dba0",
        "#5aae61",
        "#1b7837",  # Dark green
    ]

    @staticmethod
    def get_colors(scheme: str = "plotly", n_colors: Optional[int] = None) -> List[str]:
        """
        Get color palette by name.

        Parameters
        ----------
        scheme : str
            Name of color scheme:
            - 'plotly': Default Plotly colors
            - 'd3': D3.js colors
            - 'colorblind_safe': Colorblind accessible
            - 'tableau': Tableau 10 palette
            - 'sequential_blue': Blue gradient
            - 'sequential_green': Green gradient
            - 'diverging_red_blue': Red-blue scale
            - 'diverging_purple_green': Purple-green scale
        n_colors : Optional[int]
            Number of colors needed
            If more than available, cycles through palette

        Returns
        -------
        List[str]
            List of hex color codes

        Examples
        --------
        >>> colors = ColorSchemes.get_colors('plotly', n_colors=5)
        >>> print(len(colors))  # 5
        >>>
        >>> # Cycles if more colors needed
        >>> colors = ColorSchemes.get_colors('colorblind_safe', n_colors=10)
        >>> print(len(colors))  # 10 (cycles through 8-color palette)

        Raises
        ------
        ValueError
            If scheme name is not recognized
        """
        # Get base palette
        scheme_lower = scheme.lower().replace("-", "_").replace(" ", "_")

        if scheme_lower == "plotly":
            palette = ColorSchemes.PLOTLY
        elif scheme_lower == "d3":
            palette = ColorSchemes.D3
        elif scheme_lower in ["colorblind_safe", "wong"]:
            palette = ColorSchemes.COLORBLIND_SAFE
        elif scheme_lower == "tableau":
            palette = ColorSchemes.TABLEAU
        elif scheme_lower == "sequential_blue":
            palette = ColorSchemes.SEQUENTIAL_BLUE
        elif scheme_lower == "sequential_green":
            palette = ColorSchemes.SEQUENTIAL_GREEN
        elif scheme_lower == "sequential_orange":
            palette = ColorSchemes.SEQUENTIAL_ORANGE
        elif scheme_lower == "diverging_red_blue":
            palette = ColorSchemes.DIVERGING_RED_BLUE
        elif scheme_lower == "diverging_purple_green":
            palette = ColorSchemes.DIVERGING_PURPLE_GREEN
        else:
            raise ValueError(
                f"Unknown color scheme '{scheme}'. "
                f"Available: plotly, d3, colorblind_safe, tableau, "
                f"sequential_blue, sequential_green, diverging_red_blue, diverging_purple_green"
            )

        # Return appropriate number of colors
        if n_colors is None:
            return palette.copy()
        else:
            # Cycle through palette if more colors needed
            colors = []
            for i in range(n_colors):
                colors.append(palette[i % len(palette)])
            return colors

    @staticmethod
    def interpolate_colors(
        color1: str, color2: str, n_steps: int = 10
    ) -> List[str]:
        """
        Generate color gradient between two colors.

        Parameters
        ----------
        color1 : str
            Starting color (hex code)
        color2 : str
            Ending color (hex code)
        n_steps : int
            Number of intermediate colors

        Returns
        -------
        List[str]
            List of hex color codes

        Examples
        --------
        >>> # Gradient from red to blue
        >>> gradient = ColorSchemes.interpolate_colors('#FF0000', '#0000FF', n_steps=5)
        >>> print(gradient)
        ['#FF0000', '#BF003F', '#7F007F', '#3F00BF', '#0000FF']
        """
        # Parse hex colors
        r1 = int(color1[1:3], 16)
        g1 = int(color1[3:5], 16)
        b1 = int(color1[5:7], 16)

        r2 = int(color2[1:3], 16)
        g2 = int(color2[3:5], 16)
        b2 = int(color2[5:7], 16)

        # Generate gradient
        colors = []
        for i in range(n_steps):
            t = i / (n_steps - 1) if n_steps > 1 else 0
            r = int(r1 + (r2 - r1) * t)
            g = int(g1 + (g2 - g1) * t)
            b = int(b1 + (b2 - b1) * t)
            colors.append(f"#{r:02x}{g:02x}{b:02x}")

        return colors


class PlotThemes:
    """
    Complete plotting theme configurations.

    Provides preset themes that combine colors, fonts, templates, and styling
    for consistent professional visualization.

    Attributes
    ----------
    DEFAULT : dict
        Standard Plotly white theme
    PUBLICATION : dict
        Publication-ready styling (clean, high-contrast)
    DARK : dict
        Dark mode theme
    PRESENTATION : dict
        Large fonts and high contrast for presentations

    Examples
    --------
    >>> # Apply theme to figure
    >>> fig = plotter.plot_trajectory(t, x)
    >>> fig = PlotThemes.apply_theme(fig, theme='publication')
    >>> fig.show()
    >>>
    >>> # Custom theme
    >>> custom = PlotThemes.DEFAULT.copy()
    >>> custom['font_size'] = 16
    >>> fig = PlotThemes.apply_theme(fig, theme=custom)
    """

    DEFAULT = {
        "color_scheme": "plotly",
        "template": "plotly_white",
        "font_family": "Arial, sans-serif",
        "font_size": 12,
        "line_width": 2,
        "marker_size": 8,
        "grid": True,
    }

    PUBLICATION = {
        "color_scheme": "colorblind_safe",
        "template": "simple_white",
        "font_family": "Times New Roman, serif",
        "font_size": 14,
        "line_width": 2.5,
        "marker_size": 10,
        "grid": True,
        "showlegend": True,
    }

    DARK = {
        "color_scheme": "plotly",
        "template": "plotly_dark",
        "font_family": "Arial, sans-serif",
        "font_size": 12,
        "line_width": 2,
        "marker_size": 8,
        "grid": True,
    }

    PRESENTATION = {
        "color_scheme": "tableau",
        "template": "plotly_white",
        "font_family": "Arial, sans-serif",
        "font_size": 18,
        "line_width": 3,
        "marker_size": 12,
        "grid": True,
    }

    @staticmethod
    def apply_theme(fig: go.Figure, theme: str = "default") -> go.Figure:
        """
        Apply complete theme to Plotly figure.

        Parameters
        ----------
        fig : go.Figure
            Plotly figure to style
        theme : str or dict
            Theme name ('default', 'publication', 'dark', 'presentation')
            or custom theme dictionary

        Returns
        -------
        go.Figure
            Styled figure

        Examples
        --------
        >>> # Apply publication theme
        >>> fig = plotter.plot_trajectory(t, x)
        >>> fig = PlotThemes.apply_theme(fig, theme='publication')
        >>> fig.write_html('figure.html')
        >>>
        >>> # Custom theme
        >>> custom = {
        ...     'template': 'plotly_white',
        ...     'font_family': 'Helvetica',
        ...     'font_size': 16,
        ... }
        >>> fig = PlotThemes.apply_theme(fig, theme=custom)
        """
        # Get theme config
        if isinstance(theme, str):
            theme_lower = theme.lower()
            if theme_lower == "default":
                config = PlotThemes.DEFAULT
            elif theme_lower == "publication":
                config = PlotThemes.PUBLICATION
            elif theme_lower == "dark":
                config = PlotThemes.DARK
            elif theme_lower == "presentation":
                config = PlotThemes.PRESENTATION
            else:
                raise ValueError(
                    f"Unknown theme '{theme}'. "
                    f"Available: default, publication, dark, presentation"
                )
        elif isinstance(theme, dict):
            config = theme
        else:
            raise TypeError("theme must be str or dict")

        # Apply template
        if "template" in config:
            fig.update_layout(template=config["template"])

        # Apply fonts
        if "font_family" in config or "font_size" in config:
            font = {}
            if "font_family" in config:
                font["family"] = config["font_family"]
            if "font_size" in config:
                font["size"] = config["font_size"]
            fig.update_layout(font=font)

        # Apply legend settings
        if "showlegend" in config:
            fig.update_layout(showlegend=config["showlegend"])

        # Update line widths (if traces exist)
        if "line_width" in config:
            for trace in fig.data:
                if hasattr(trace, "line"):
                    trace.line.width = config["line_width"]

        # Update marker sizes (if traces exist)
        if "marker_size" in config:
            for trace in fig.data:
                if hasattr(trace, "marker") and hasattr(trace.marker, "size"):
                    trace.marker.size = config["marker_size"]

        return fig

    @staticmethod
    def get_line_styles() -> List[str]:
        """
        Get available line dash patterns.

        Returns
        -------
        List[str]
            List of Plotly line dash patterns

        Examples
        --------
        >>> styles = PlotThemes.get_line_styles()
        >>> print(styles)
        ['solid', 'dot', 'dash', 'longdash', 'dashdot', 'longdashdot']
        """
        return ["solid", "dot", "dash", "longdash", "dashdot", "longdashdot"]


# ============================================================================
# Color Manipulation Utilities
# ============================================================================


def hex_to_rgb(hex_color: str) -> tuple:
    """
    Convert hex color to RGB tuple.

    Parameters
    ----------
    hex_color : str
        Hex color code (e.g., '#FF0000')

    Returns
    -------
    tuple
        (r, g, b) values in range [0, 255]

    Examples
    --------
    >>> rgb = hex_to_rgb('#FF0000')
    >>> print(rgb)  # (255, 0, 0)
    """
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


def rgb_to_hex(r: int, g: int, b: int) -> str:
    """
    Convert RGB values to hex color.

    Parameters
    ----------
    r : int
        Red value [0, 255]
    g : int
        Green value [0, 255]
    b : int
        Blue value [0, 255]

    Returns
    -------
    str
        Hex color code

    Examples
    --------
    >>> hex_color = rgb_to_hex(255, 0, 0)
    >>> print(hex_color)  # '#ff0000'
    """
    return f"#{r:02x}{g:02x}{b:02x}"


def lighten_color(hex_color: str, factor: float = 0.2) -> str:
    """
    Lighten or darken a hex color.

    Parameters
    ----------
    hex_color : str
        Hex color code
    factor : float
        Lightening factor
        Positive: lighten (move toward white)
        Negative: darken (move toward black)
        Range: [-1, 1]

    Returns
    -------
    str
        Modified hex color

    Examples
    --------
    >>> # Lighten blue
    >>> light_blue = lighten_color('#0000FF', factor=0.3)
    >>>
    >>> # Darken red
    >>> dark_red = lighten_color('#FF0000', factor=-0.3)
    """
    r, g, b = hex_to_rgb(hex_color)

    if factor > 0:
        # Lighten: move toward white (255, 255, 255)
        r = int(r + (255 - r) * factor)
        g = int(g + (255 - g) * factor)
        b = int(b + (255 - b) * factor)
    else:
        # Darken: move toward black (0, 0, 0)
        r = int(r * (1 + factor))
        g = int(g * (1 + factor))
        b = int(b * (1 + factor))

    # Clamp to valid range
    r = max(0, min(255, r))
    g = max(0, min(255, g))
    b = max(0, min(255, b))

    return rgb_to_hex(r, g, b)


# ============================================================================
# Export
# ============================================================================

__all__ = [
    "ColorSchemes",
    "PlotThemes",
    "hex_to_rgb",
    "rgb_to_hex",
    "lighten_color",
]
