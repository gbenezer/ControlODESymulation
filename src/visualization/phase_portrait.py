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
Phase Portrait Plotter - State Space Visualization

Interactive Plotly-based phase space visualization for dynamical systems.

Key Features
------------
- 2D/3D phase portraits: Visualize trajectories in state space
- Vector field overlay: Show system dynamics with arrow fields
- Equilibrium markers: Highlight fixed points and their stability
- Direction arrows: Show trajectory flow direction
- Batched support: Multiple trajectories in same phase space
- Interactive 3D: Rotate and zoom 3D phase portraits
- Consistent theming: Integrates with centralized themes module

Main Class
----------
PhasePortraitPlotter : Phase space visualization
    plot_2d() : 2D phase portrait (x₁ vs x₂)
    plot_3d() : 3D phase portrait (x₁ vs x₂ vs x₃)
    plot_limit_cycle() : Highlight periodic orbits

Usage
-----
>>> from src.plotting import PhasePortraitPlotter
>>> 
>>> # Create plotter
>>> plotter = PhasePortraitPlotter()
>>> 
>>> # 2D phase portrait
>>> fig = plotter.plot_2d(
...     x=trajectory,  # (T, 2)
...     state_names=('Position', 'Velocity'),
...     show_direction=True
... )
>>> fig.show()
>>> 
>>> # With vector field and custom theme
>>> def dynamics(x1, x2):
...     return np.array([x2, -x1])
>>> 
>>> fig = plotter.plot_2d(
...     x=trajectory,
...     vector_field=dynamics,
...     equilibria=[np.zeros(2)],
...     color_scheme='colorblind_safe',
...     theme='publication'
... )
>>> 
>>> # 3D phase portrait
>>> fig = plotter.plot_3d(
...     x=trajectory_3d,  # (T, 3)
...     state_names=('x', 'y', 'z'),
...     theme='dark'
... )
"""

from typing import Callable, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go

from src.visualization.themes import ColorSchemes, PlotThemes
from src.types.backends import Backend


class PhasePortraitPlotter:
    """
    Phase space visualization for dynamical systems.

    Provides interactive Plotly-based plotting for phase portraits in 2D and 3D,
    with optional vector fields, equilibrium points, and direction indicators.

    Attributes
    ----------
    backend : Backend
        Default computational backend for array conversion
    default_theme : str
        Default plot theme to apply

    Examples
    --------
    2D phase portrait:

    >>> plotter = PhasePortraitPlotter()
    >>> x = np.column_stack([np.sin(t), np.cos(t)])  # (T, 2)
    >>> fig = plotter.plot_2d(x, state_names=('sin', 'cos'))
    >>> fig.show()

    With vector field and equilibrium:

    >>> def f(x1, x2):
    ...     return np.array([x2, -x1 - 0.1*x2])
    >>> 
    >>> fig = plotter.plot_2d(
    ...     x,
    ...     vector_field=f,
    ...     equilibria=[np.zeros(2)],
    ...     show_direction=True,
    ...     theme='publication'
    ... )

    3D phase portrait (Lorenz attractor):

    >>> fig = plotter.plot_3d(
    ...     x_lorenz,  # (T, 3)
    ...     state_names=('x', 'y', 'z'),
    ...     color_scheme='tableau',
    ...     theme='dark'
    ... )
    """

    def __init__(self, backend: Backend = "numpy", default_theme: str = "default"):
        """
        Initialize phase portrait plotter.

        Parameters
        ----------
        backend : Backend
            Default backend for array conversion ('numpy', 'torch', 'jax')
        default_theme : str
            Default theme to apply to plots
            Options: 'default', 'publication', 'dark', 'presentation'

        Examples
        --------
        >>> plotter = PhasePortraitPlotter(backend='numpy')
        >>> plotter = PhasePortraitPlotter(backend='jax', default_theme='dark')
        """
        self.backend = backend
        self.default_theme = default_theme

    # =========================================================================
    # Main Plotting Methods
    # =========================================================================

    def plot_2d(
        self,
        x: np.ndarray,
        state_names: Tuple[str, str] = ("x₁", "x₂"),
        trajectory_names: Optional[List[str]] = None,
        show_direction: bool = True,
        show_start_end: bool = True,
        vector_field: Optional[Callable] = None,
        equilibria: Optional[List[np.ndarray]] = None,
        title: str = "2D Phase Portrait",
        color_scheme: str = "plotly",
        theme: Optional[str] = None,
        **kwargs,
    ) -> go.Figure:
        """
        Create 2D phase portrait (x₂ vs x₁).

        Plots trajectory through 2D state space with optional vector field
        overlay and equilibrium point markers.

        Parameters
        ----------
        x : np.ndarray
            State trajectory, shape (T, 2) or (n_batch, T, 2)
            Must be 2-dimensional state
        state_names : Tuple[str, str]
            Names for horizontal and vertical axes
            Default: ('x₁', 'x₂')
        trajectory_names : Optional[List[str]]
            Custom names for each trajectory (for batched trajectories)
            If None, uses "Trajectory 1", "Trajectory 2", etc.
            Length must equal number of batches
        show_direction : bool
            If True, add arrows showing trajectory direction
        show_start_end : bool
            If True, mark initial and final points
        vector_field : Optional[Callable]
            Function f(x1, x2) -> [dx1, dx2] for vector field
            If provided, overlays arrow field showing dynamics
        equilibria : Optional[List[np.ndarray]]
            List of equilibrium points to mark, each shape (2,)
        title : str
            Plot title
        color_scheme : str
            Color scheme name
            Options: 'plotly', 'd3', 'colorblind_safe', 'tableau',
                     'sequential_blue', 'diverging_red_blue', etc.
            Default: 'plotly'
        theme : Optional[str]
            Plot theme to apply
            Options: 'default', 'publication', 'dark', 'presentation'
            If None, uses self.default_theme
        **kwargs
            Additional customization arguments

        Returns
        -------
        go.Figure
            Plotly figure object

        Examples
        --------
        >>> # Simple phase portrait
        >>> t = np.linspace(0, 10, 100)
        >>> x = np.column_stack([np.sin(t), np.cos(t)])
        >>> fig = plotter.plot_2d(x, state_names=('Position', 'Velocity'))
        >>> fig.show()
        >>>
        >>> # With vector field (pendulum) and publication theme
        >>> def pendulum_dynamics(x1, x2):
        ...     return np.array([x2, -np.sin(x1) - 0.1*x2])
        >>>
        >>> fig = plotter.plot_2d(
        ...     x,
        ...     vector_field=pendulum_dynamics,
        ...     equilibria=[np.array([0, 0]), np.array([np.pi, 0])],
        ...     show_direction=True,
        ...     color_scheme='colorblind_safe',
        ...     theme='publication'
        ... )
        >>>
        >>> # Batched trajectories (multiple initial conditions)
        >>> x_batch = np.random.randn(5, 100, 2)
        >>> fig = plotter.plot_2d(
        ...     x_batch,
        ...     trajectory_names=['IC 1', 'IC 2', 'IC 3', 'IC 4', 'IC 5'],
        ...     theme='dark'
        ... )

        Notes
        -----
        - Automatically handles batched trajectories
        - Vector field computed on grid if provided
        - Start point: green circle, End point: red square
        - Direction arrows added at regular intervals
        """
        # Use default theme if not specified
        if theme is None:
            theme = self.default_theme

        # Convert to NumPy
        x_np = self._to_numpy(x)

        # Validate dimensions
        if x_np.shape[-1] != 2:
            raise ValueError(
                f"plot_2d requires 2D state, got shape {x_np.shape} (last dim should be 2)"
            )

        # Detect batching
        is_batched = self._is_batched(x_np)

        if is_batched:
            # x: (n_batch, T, 2)
            n_batch, T, _ = x_np.shape
        else:
            # x: (T, 2)
            if x_np.ndim == 1:
                raise ValueError("plot_2d requires at least 2D array (T, 2)")
            T, _ = x_np.shape
            n_batch = 1
            # Reshape to (1, T, 2) for uniform processing
            x_np = x_np[np.newaxis, :, :]

        # Get colors from centralized color schemes
        colors = ColorSchemes.get_colors(color_scheme, n_batch)

        # Generate trajectory names if not provided
        if trajectory_names is None:
            trajectory_names = [f"Trajectory {i + 1}" for i in range(n_batch)]
        elif len(trajectory_names) != n_batch:
            raise ValueError(f"trajectory_names length {len(trajectory_names)} != n_batch {n_batch}")

        # Create figure
        fig = go.Figure()

        # Add start/end markers first (for legend ordering)
        if show_start_end and n_batch > 0:
            # Add start marker (will be first in legend)
            fig.add_trace(
                go.Scatter(
                    x=[x_np[0, 0, 0]],
                    y=[x_np[0, 0, 1]],
                    mode="markers",
                    name="Start",
                    marker=dict(color="green", size=10, symbol="circle"),
                    showlegend=True,
                    hovertemplate=f"<b>Start: {trajectory_names[0]}</b><br>x₁: %{{x:.3f}}<br>x₂: %{{y:.3f}}<extra></extra>",
                )
            )
            
            # Add end marker (will be second in legend)
            fig.add_trace(
                go.Scatter(
                    x=[x_np[0, -1, 0]],
                    y=[x_np[0, -1, 1]],
                    mode="markers",
                    name="End",
                    marker=dict(color="red", size=10, symbol="square"),
                    showlegend=True,
                    hovertemplate=f"<b>End: {trajectory_names[0]}</b><br>x₁: %{{x:.3f}}<br>x₂: %{{y:.3f}}<extra></extra>",
                )
            )

        # Add equilibria if provided (third in legend)
        if equilibria is not None:
            self._add_equilibria_markers(fig, equilibria)

        # Add vector field if provided
        if vector_field is not None:
            self._add_vector_field_2d(fig, vector_field, x_np, grid_density=15)

        # Add trajectories (last in legend)
        for batch_idx in range(n_batch):
            x_traj = x_np[batch_idx]  # (T, 2)

            # Main trajectory line
            fig.add_trace(
                go.Scatter(
                    x=x_traj[:, 0],
                    y=x_traj[:, 1],
                    mode="lines",
                    name=trajectory_names[batch_idx],
                    line=dict(color=colors[batch_idx], width=2),
                    showlegend=True,
                )
            )

            # Add start/end markers for additional trajectories (beyond first)
            if show_start_end and batch_idx > 0:
                # Start marker (not in legend)
                fig.add_trace(
                    go.Scatter(
                        x=[x_traj[0, 0]],
                        y=[x_traj[0, 1]],
                        mode="markers",
                        marker=dict(color="green", size=10, symbol="circle"),
                        showlegend=False,
                        hovertemplate=f"<b>Start: {trajectory_names[batch_idx]}</b><br>x₁: %{{x:.3f}}<br>x₂: %{{y:.3f}}<extra></extra>",
                    )
                )

                # End marker (not in legend)
                fig.add_trace(
                    go.Scatter(
                        x=[x_traj[-1, 0]],
                        y=[x_traj[-1, 1]],
                        mode="markers",
                        marker=dict(color="red", size=10, symbol="square"),
                        showlegend=False,
                        hovertemplate=f"<b>End: {trajectory_names[batch_idx]}</b><br>x₁: %{{x:.3f}}<br>x₂: %{{y:.3f}}<extra></extra>",
                    )
                )

            # Direction arrows
            if show_direction and T > 10:
                self._add_direction_arrows_2d(
                    fig, x_traj, colors[batch_idx], n_arrows=5
                )

        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title=state_names[0],
            yaxis_title=state_names[1],
            width=700,
            height=600,
            showlegend=True,
        )

        # Equal aspect ratio for phase portraits
        fig.update_yaxes(scaleanchor="x", scaleratio=1)

        # Apply theme
        fig = PlotThemes.apply_theme(fig, theme=theme)

        return fig

    def plot_3d(
        self,
        x: np.ndarray,
        state_names: Tuple[str, str, str] = ("x₁", "x₂", "x₃"),
        trajectory_names: Optional[List[str]] = None,
        show_direction: bool = True,
        show_start_end: bool = True,
        title: str = "3D Phase Portrait",
        color_scheme: str = "plotly",
        direction_colorscale: str = "Viridis",
        theme: Optional[str] = None,
        **kwargs,
    ) -> go.Figure:
        """
        Create 3D phase portrait.

        Plots trajectory through 3D state space with interactive rotation.

        Parameters
        ----------
        x : np.ndarray
            State trajectory, shape (T, 3) or (n_batch, T, 3)
            Must be 3-dimensional state
        state_names : Tuple[str, str, str]
            Names for x, y, z axes
            Default: ('x₁', 'x₂', 'x₃')
        trajectory_names : Optional[List[str]]
            Custom names for each trajectory (for batched trajectories)
            If None, uses "Trajectory 1", "Trajectory 2", etc.
            Length must equal number of batches
        show_direction : bool
            If True, show temporal direction via line color gradient (single trajectory only)
        show_start_end : bool
            If True, mark initial and final points
        title : str
            Plot title
        color_scheme : str
            Color scheme name
            Options: 'plotly', 'd3', 'colorblind_safe', 'tableau', etc.
            Default: 'plotly'
            Note: Only used for batched trajectories; single trajectories use direction_colorscale
        direction_colorscale : str
            Colorscale for temporal gradient (single trajectory only)
            Options: 'Viridis', 'Plasma', 'Inferno', 'Magma', 'Cividis',
                     'Turbo', 'Rainbow', 'Jet', 'Hot', 'Cool', 'Blues', 'Reds',
                     'Greens', 'Portland', 'Picnic', 'Electric', 'Blackbody'
            Default: 'Viridis'
        theme : Optional[str]
            Plot theme to apply
            Options: 'default', 'publication', 'dark', 'presentation'
            If None, uses self.default_theme
        **kwargs
            Additional customization arguments

        Returns
        -------
        go.Figure
            Interactive 3D Plotly figure

        Examples
        --------
        >>> # Lorenz attractor with temporal gradient
        >>> x_lorenz = solve_lorenz(...)  # (T, 3)
        >>> fig = plotter.plot_3d(
        ...     x_lorenz,
        ...     state_names=('x', 'y', 'z'),
        ...     title='Lorenz Attractor',
        ...     show_direction=True,
        ...     theme='dark'
        ... )
        >>> fig.show()
        >>>
        >>> # Multiple trajectories with custom names
        >>> x_batch = np.random.randn(3, 1000, 3)
        >>> fig = plotter.plot_3d(
        ...     x_batch,
        ...     trajectory_names=['Low Energy', 'Medium Energy', 'High Energy'],
        ...     color_scheme='colorblind_safe',
        ...     theme='publication'
        ... )

        Notes
        -----
        - Interactive: Click and drag to rotate
        - Scroll to zoom
        - Single trajectory: Line color shows time progression (dark→light)
        - Batched trajectories: Each trajectory has distinct solid color
        - Start: green sphere, End: red cube
        """
        # Use default theme if not specified
        if theme is None:
            theme = self.default_theme

        # Convert to NumPy
        x_np = self._to_numpy(x)

        # Validate dimensions
        if x_np.shape[-1] != 3:
            raise ValueError(
                f"plot_3d requires 3D state, got shape {x_np.shape} (last dim should be 3)"
            )

        # Detect batching
        is_batched = self._is_batched(x_np)

        if is_batched:
            # x: (n_batch, T, 3)
            n_batch, T, _ = x_np.shape
        else:
            # x: (T, 3)
            if x_np.ndim != 2:
                raise ValueError("plot_3d requires shape (T, 3)")
            T, _ = x_np.shape
            n_batch = 1
            x_np = x_np[np.newaxis, :, :]

        # Get colors from centralized color schemes
        colors = ColorSchemes.get_colors(color_scheme, n_batch)

        # Create figure
        fig = go.Figure()

        # Add start/end markers first (for legend ordering)
        if show_start_end and n_batch > 0:
            # Add start marker (will be first in legend)
            fig.add_trace(
                go.Scatter3d(
                    x=[x_np[0, 0, 0]],
                    y=[x_np[0, 0, 1]],
                    z=[x_np[0, 0, 2]],
                    mode="markers",
                    name="Start",
                    marker=dict(color="green", size=8, symbol="circle"),
                    showlegend=True,
                )
            )
            
            # Add end marker (will be second in legend)
            fig.add_trace(
                go.Scatter3d(
                    x=[x_np[0, -1, 0]],
                    y=[x_np[0, -1, 1]],
                    z=[x_np[0, -1, 2]],
                    mode="markers",
                    name="End",
                    marker=dict(color="red", size=8, symbol="square"),
                    showlegend=True,
                )
            )

        # Add trajectories (last in legend)
        for batch_idx in range(n_batch):
            x_traj = x_np[batch_idx]  # (T, 3)

            # For single trajectory, use color gradient to show time
            # For batched, use solid colors to distinguish trajectories
            if n_batch == 1 and show_direction:
                # Single trajectory: gradient shows temporal evolution
                time_points = np.linspace(0, 1, T)
                
                fig.add_trace(
                    go.Scatter3d(
                        x=x_traj[:, 0],
                        y=x_traj[:, 1],
                        z=x_traj[:, 2],
                        mode="lines",
                        name="Trajectory",
                        line=dict(
                            color=time_points,
                            colorscale=direction_colorscale,
                            width=4,
                            showscale=True,
                            colorbar=dict(
                                title="Time",
                                x=-0.15,  # Position on left side
                                xanchor="right",
                                len=0.75,
                                thickness=15
                            )
                        ),
                        showlegend=True,
                    )
                )
            else:
                # Batched trajectories: use solid colors
                fig.add_trace(
                    go.Scatter3d(
                        x=x_traj[:, 0],
                        y=x_traj[:, 1],
                        z=x_traj[:, 2],
                        mode="lines",
                        name=f"Trajectory {batch_idx + 1}" if is_batched else "Trajectory",
                        line=dict(color=colors[batch_idx], width=3),
                        showlegend=True,
                    )
                )

            # Add start/end markers for additional trajectories (beyond first)
            if show_start_end and batch_idx > 0:
                # Start marker (not in legend)
                fig.add_trace(
                    go.Scatter3d(
                        x=[x_traj[0, 0]],
                        y=[x_traj[0, 1]],
                        z=[x_traj[0, 2]],
                        mode="markers",
                        marker=dict(color="green", size=8, symbol="circle"),
                        showlegend=False,
                    )
                )

                # End marker (not in legend)
                fig.add_trace(
                    go.Scatter3d(
                        x=[x_traj[-1, 0]],
                        y=[x_traj[-1, 1]],
                        z=[x_traj[-1, 2]],
                        mode="markers",
                        marker=dict(color="red", size=8, symbol="square"),
                        showlegend=False,
                    )
                )

        # Update layout
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title=state_names[0],
                yaxis_title=state_names[1],
                zaxis_title=state_names[2],
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
            ),
            width=800,
            height=700,
            showlegend=True,
        )

        # Apply theme
        fig = PlotThemes.apply_theme(fig, theme=theme)

        return fig

    def plot_limit_cycle(
        self,
        x: np.ndarray,
        state_names: Tuple[str, str] = ("x₁", "x₂"),
        trajectory_names: Optional[List[str]] = None,
        period_estimate: Optional[float] = None,
        title: str = "Limit Cycle",
        color_scheme: str = "plotly",
        theme: Optional[str] = None,
        **kwargs,
    ) -> go.Figure:
        """
        Visualize limit cycle (periodic orbit).

        Highlights periodic behavior in phase space by overlaying multiple
        periods if detected.

        Parameters
        ----------
        x : np.ndarray
            State trajectory, shape (T, 2)
            Should contain at least one full period
        state_names : Tuple[str, str]
            Names for axes
        trajectory_names : Optional[List[str]]
            Custom names for trajectories
            If None, uses "Trajectory 1"
        period_estimate : Optional[float]
            Estimated period (in samples)
            If None, attempts auto-detection
        title : str
            Plot title
        color_scheme : str
            Color scheme name
            Default: 'plotly'
        theme : Optional[str]
            Plot theme to apply
            Options: 'default', 'publication', 'dark', 'presentation'
            If None, uses self.default_theme
        **kwargs
            Additional arguments

        Returns
        -------
        go.Figure
            Phase portrait with limit cycle highlighted

        Examples
        --------
        >>> # Van der Pol oscillator
        >>> x_vdp = solve_van_der_pol(...)  # (T, 2)
        >>> fig = plotter.plot_limit_cycle(
        ...     x_vdp,
        ...     period_estimate=100,
        ...     theme='publication'
        ... )
        >>> fig.show()

        Notes
        -----
        - Limit cycle detection is heuristic
        - Works best with long trajectories (many periods)
        - Highlights the periodic attractor
        """
        # Use default theme if not specified
        if theme is None:
            theme = self.default_theme

        # Convert to NumPy
        x_np = self._to_numpy(x)

        # Validate
        if x_np.shape[-1] != 2:
            raise ValueError("plot_limit_cycle requires 2D state")

        if x_np.ndim == 3:
            # Take first trajectory if batched
            x_np = x_np[0]

        # Simple limit cycle visualization
        # In future: could add Poincaré section, period detection, etc.
        fig = self.plot_2d(
            x_np,
            state_names=state_names,
            trajectory_names=trajectory_names,
            show_direction=True,
            show_start_end=True,
            title=title,
            color_scheme=color_scheme,
            theme=theme,
        )

        # Add annotation
        fig.add_annotation(
            text="Limit Cycle (Periodic Orbit)",
            xref="paper",
            yref="paper",
            x=0.5,
            y=1.05,
            showarrow=False,
            font=dict(size=14, color="blue"),
        )

        return fig

    # =========================================================================
    # Helper Methods (Internal)
    # =========================================================================

    def _to_numpy(self, arr) -> np.ndarray:
        """
        Convert array from any backend to NumPy.

        Parameters
        ----------
        arr : array-like
            Input array (any backend)

        Returns
        -------
        np.ndarray
            NumPy array
        """
        if arr is None:
            return None

        if isinstance(arr, np.ndarray):
            return arr

        # PyTorch
        if hasattr(arr, "cpu") and hasattr(arr, "detach"):
            return arr.detach().cpu().numpy()

        # JAX
        if hasattr(arr, "__array__"):
            return np.array(arr)

        # Generic fallback
        return np.asarray(arr)

    def _is_batched(self, x: np.ndarray) -> bool:
        """
        Detect if trajectory is batched.

        Parameters
        ----------
        x : np.ndarray
            State trajectory

        Returns
        -------
        bool
            True if batched (3D), False otherwise
        """
        if x is None:
            return False
        return x.ndim == 3

    def _add_vector_field_2d(
        self,
        fig: go.Figure,
        f: Callable,
        x_trajectories: np.ndarray,
        grid_density: int = 15,
    ) -> None:
        """
        Add vector field to 2D phase portrait.

        Parameters
        ----------
        fig : go.Figure
            Figure to add arrows to
        f : Callable
            Dynamics function f(x1, x2) -> [dx1, dx2]
        x_trajectories : np.ndarray
            Trajectories to determine plot bounds, shape (n_batch, T, 2)
        grid_density : int
            Number of arrows per dimension
        """
        # Determine bounds from trajectories
        x1_all = x_trajectories[:, :, 0].flatten()
        x2_all = x_trajectories[:, :, 1].flatten()

        x1_min, x1_max = x1_all.min(), x1_all.max()
        x2_min, x2_max = x2_all.min(), x2_all.max()

        # Add padding
        x1_range = x1_max - x1_min
        x2_range = x2_max - x2_min
        x1_min -= 0.1 * x1_range
        x1_max += 0.1 * x1_range
        x2_min -= 0.1 * x2_range
        x2_max += 0.1 * x2_range

        # Create grid
        x1_grid = np.linspace(x1_min, x1_max, grid_density)
        x2_grid = np.linspace(x2_min, x2_max, grid_density)

        # Compute vector field
        for x1 in x1_grid:
            for x2 in x2_grid:
                try:
                    vec = f(x1, x2)
                    dx1, dx2 = vec[0], vec[1]

                    # Normalize for visualization
                    mag = np.sqrt(dx1**2 + dx2**2)
                    if mag > 1e-10:
                        scale = 0.03 * min(x1_range, x2_range)
                        dx1_norm = dx1 / mag * scale
                        dx2_norm = dx2 / mag * scale

                        # Add arrow
                        fig.add_annotation(
                            x=x1 + dx1_norm,
                            y=x2 + dx2_norm,
                            ax=x1,
                            ay=x2,
                            xref="x",
                            yref="y",
                            axref="x",
                            ayref="y",
                            showarrow=True,
                            arrowhead=2,
                            arrowsize=1,
                            arrowwidth=1,
                            arrowcolor="rgba(128, 128, 128, 0.4)",
                        )
                except:
                    # Skip points where dynamics evaluation fails
                    pass

    def _add_equilibria_markers(
        self, fig: go.Figure, equilibria: List[np.ndarray]
    ) -> None:
        """
        Add markers for equilibrium points.

        Parameters
        ----------
        fig : go.Figure
            Figure to add markers to
        equilibria : List[np.ndarray]
            List of equilibrium points, each shape (2,) or (3,)
        """
        if not equilibria:
            return

        # Determine dimensionality from first equilibrium
        ndim = equilibria[0].shape[0]

        if ndim == 2:
            # 2D equilibria
            x_eq = [eq[0] for eq in equilibria]
            y_eq = [eq[1] for eq in equilibria]

            fig.add_trace(
                go.Scatter(
                    x=x_eq,
                    y=y_eq,
                    mode="markers",
                    name="Equilibria",
                    marker=dict(
                        color="black",
                        size=12,
                        symbol="x",
                        line=dict(color="black", width=2),
                    ),
                    showlegend=True,
                )
            )
        elif ndim == 3:
            # 3D equilibria
            x_eq = [eq[0] for eq in equilibria]
            y_eq = [eq[1] for eq in equilibria]
            z_eq = [eq[2] for eq in equilibria]

            fig.add_trace(
                go.Scatter3d(
                    x=x_eq,
                    y=y_eq,
                    z=z_eq,
                    mode="markers",
                    name="Equilibria",
                    marker=dict(color="black", size=8, symbol="x"),
                    showlegend=True,
                )
            )

    def _add_direction_arrows_2d(
        self, fig: go.Figure, x_traj: np.ndarray, color: str, n_arrows: int = 5
    ) -> None:
        """
        Add direction arrows to 2D trajectory.

        Parameters
        ----------
        fig : go.Figure
            Figure to add arrows to
        x_traj : np.ndarray
            Trajectory, shape (T, 2)
        color : str
            Arrow color
        n_arrows : int
            Number of arrows to add
        """
        T = x_traj.shape[0]
        indices = np.linspace(0, T - 2, n_arrows, dtype=int)

        for idx in indices:
            x1_start, x2_start = x_traj[idx]
            x1_end, x2_end = x_traj[idx + 1]

            fig.add_annotation(
                x=x1_end,
                y=x2_end,
                ax=x1_start,
                ay=x2_start,
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                showarrow=True,
                arrowhead=2,
                arrowsize=1.5,
                arrowwidth=2,
                arrowcolor=color,
            )

    # =========================================================================
    # Utility Methods
    # =========================================================================

    @staticmethod
    def list_available_themes() -> List[str]:
        """
        List available plot themes.

        Returns
        -------
        List[str]
            Available theme names

        Examples
        --------
        >>> themes = PhasePortraitPlotter.list_available_themes()
        >>> print(themes)
        ['default', 'publication', 'dark', 'presentation']
        """
        return ["default", "publication", "dark", "presentation"]

    @staticmethod
    def list_available_color_schemes() -> List[str]:
        """
        List available color schemes.

        Returns
        -------
        List[str]
            Available color scheme names

        Examples
        --------
        >>> schemes = PhasePortraitPlotter.list_available_color_schemes()
        >>> print(schemes)
        ['plotly', 'd3', 'colorblind_safe', 'tableau', ...]
        """
        return [
            "plotly",
            "d3",
            "colorblind_safe",
            "tableau",
            "sequential_blue",
            "sequential_green",
            "sequential_orange",
            "diverging_red_blue",
            "diverging_purple_green",
        ]


# ============================================================================
# Export
# ============================================================================

__all__ = [
    "PhasePortraitPlotter",
]