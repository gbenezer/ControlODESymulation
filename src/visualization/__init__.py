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
Visualization Tools
===================

This module provides visualization utilities for dynamical systems analysis,
including trajectory plotting, phase portraits, and control system visualization.

Plotting Classes
----------------
>>> from src.visualization import (
...     TrajectoryPlotter,
...     PhasePortraitPlotter,
...     ControlPlotter,
... )
>>>
>>> # Plot state trajectories
>>> plotter = TrajectoryPlotter()
>>> plotter.plot(result)
>>>
>>> # Phase portrait analysis
>>> phase = PhasePortraitPlotter(system)
>>> phase.plot(xlim=(-2, 2), ylim=(-2, 2))
>>>
>>> # Control input visualization
>>> ctrl_plotter = ControlPlotter()
>>> ctrl_plotter.plot(t, u)

Themes and Styling
------------------
>>> from src.visualization import ColorSchemes, PlotThemes
>>>
>>> # Use predefined color schemes
>>> colors = ColorSchemes.CONTROL_THEORY
>>>
>>> # Apply consistent themes
>>> PlotThemes.apply_default()

Authors
-------
Gil Benezer

License
-------
GNU Affero General Public License v3.0
"""

# Plotters
from .control_plots import ControlPlotter
from .phase_portrait import PhasePortraitPlotter
from .trajectory_plotter import TrajectoryPlotter

# Themes and styling
from .themes import (
    ColorSchemes,
    PlotThemes,
    hex_to_rgb,
    lighten_color,
    rgb_to_hex,
)

# Export public API
__all__ = [
    # Plotters
    "TrajectoryPlotter",
    "PhasePortraitPlotter",
    "ControlPlotter",
    # Themes and styling
    "ColorSchemes",
    "PlotThemes",
    "hex_to_rgb",
    "rgb_to_hex",
    "lighten_color",
]