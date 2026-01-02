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
Trajectory Plotter - Time-Domain Visualization

Interactive Plotly-based visualization for state and control trajectories.

Key Features
------------
- Backend agnostic: Works with NumPy, PyTorch, JAX arrays
- Batched trajectory support: Handles Monte Carlo simulations automatically
- Adaptive layouts: Automatically determines optimal subplot arrangement
- Interactive: Zoom, pan, hover tooltips via Plotly
- Publication ready: Customizable styling and export to HTML

Main Class
----------
TrajectoryPlotter : Time-domain trajectory visualization
    plot_trajectory() : Plot state variables vs time
    plot_state_and_control() : Combined state + control visualization
    plot_comparison() : Compare multiple simulation runs

Usage
-----
>>> from src.plotting import TrajectoryPlotter
>>> 
>>> # Create plotter
>>> plotter = TrajectoryPlotter(backend='numpy')
>>> 
>>> # Plot single trajectory
>>> t = np.linspace(0, 10, 100)
>>> x = np.random.randn(100, 2)
>>> fig = plotter.plot_trajectory(t, x, state_names=['θ', 'ω'])
>>> fig.show()
>>> 
>>> # Plot batched trajectories (Monte Carlo)
>>> x_batch = np.random.randn(10, 100, 2)  # (n_batch, T, nx)
>>> fig = plotter.plot_trajectory(t, x_batch)
>>> fig.show()
>>> 
>>> # Via system integration
>>> system = Pendulum()
>>> result = system.integrate(x0, u=None, t_span=(0, 10))
>>> fig = system.plotter.plot_trajectory(
...     result['t'], 
...     result['x'],
...     state_names=['Angle', 'Angular Velocity']
... )
"""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.types.backends import Backend


class TrajectoryPlotter:
    """
    Time-domain trajectory visualization.

    Provides interactive Plotly-based plotting for state and control trajectories
    with automatic handling of batched data, backend conversion, and adaptive layouts.

    Attributes
    ----------
    backend : Backend
        Default computational backend for array conversion

    Examples
    --------
    Basic trajectory plotting:

    >>> plotter = TrajectoryPlotter()
    >>> t = np.linspace(0, 10, 100)
    >>> x = np.sin(t)[:, None]  # (100, 1)
    >>> fig = plotter.plot_trajectory(t, x, state_names=['sin(t)'])
    >>> fig.show()

    Batched trajectories (Monte Carlo):

    >>> x_batch = np.stack([np.sin(t + phi)[:, None] for phi in [0, 0.5, 1.0]])
    >>> # x_batch shape: (3, 100, 1)
    >>> fig = plotter.plot_trajectory(t, x_batch)
    >>> fig.show()  # Shows all 3 trajectories

    State and control:

    >>> u = 0.1 * np.random.randn(100, 1)
    >>> fig = plotter.plot_state_and_control(t, x, u)
    >>> fig.show()
    """

    def __init__(self, backend: Backend = "numpy"):
        """
        Initialize trajectory plotter.

        Parameters
        ----------
        backend : Backend
            Default backend for array conversion ('numpy', 'torch', 'jax')

        Examples
        --------
        >>> plotter = TrajectoryPlotter(backend='numpy')
        >>> plotter = TrajectoryPlotter(backend='torch')
        """
        self.backend = backend

    # =========================================================================
    # Main Plotting Methods
    # =========================================================================

    def plot_trajectory(
        self,
        t: np.ndarray,
        x: np.ndarray,
        u: Optional[np.ndarray] = None,
        state_names: Optional[List[str]] = None,
        control_names: Optional[List[str]] = None,
        title: str = "State Trajectories",
        color_scheme: str = "plotly",
        show_legend: bool = True,
        **kwargs,
    ) -> go.Figure:
        """
        Plot state trajectories over time.

        Creates subplots for each state variable showing evolution over time.
        Automatically handles single or batched trajectories.

        Parameters
        ----------
        t : np.ndarray
            Time points, shape (T,) or (n_batch, T)
        x : np.ndarray
            State trajectories, shape (T, nx) or (n_batch, T, nx)
        u : Optional[np.ndarray]
            Control inputs, shape (T, nu) or (n_batch, T, nu)
            If provided, adds control subplot(s)
        state_names : Optional[List[str]]
            Names for state variables, length nx
            Default: ['x₁', 'x₂', ...]
        control_names : Optional[List[str]]
            Names for control inputs, length nu
        title : str
            Overall plot title
        color_scheme : str
            Color scheme name (default: 'plotly')
        show_legend : bool
            Whether to show legend for batched trajectories
        **kwargs
            Additional arguments for customization

        Returns
        -------
        go.Figure
            Plotly figure object

        Examples
        --------
        >>> # Single trajectory
        >>> t = np.linspace(0, 10, 100)
        >>> x = np.column_stack([np.sin(t), np.cos(t)])
        >>> fig = plotter.plot_trajectory(t, x, state_names=['sin', 'cos'])
        >>> fig.show()
        >>>
        >>> # Batched trajectories
        >>> x_batch = np.random.randn(5, 100, 2)
        >>> fig = plotter.plot_trajectory(t, x_batch)
        >>>
        >>> # With control
        >>> u = 0.1 * np.random.randn(100, 1)
        >>> fig = plotter.plot_trajectory(t, x, u=u)

        Notes
        -----
        - Automatically detects batched vs single trajectories
        - Converts from PyTorch/JAX to NumPy internally
        - Adaptive subplot layout based on number of states
        - Interactive: zoom, pan, hover tooltips
        """
        # Convert to NumPy
        t_np = self._to_numpy(t)
        x_np = self._to_numpy(x)
        u_np = self._to_numpy(u) if u is not None else None

        # Detect if batched
        is_batched = self._is_batched(x_np)

        # Validate and reshape
        if is_batched:
            # x: (n_batch, T, nx)
            n_batch, T, nx = x_np.shape
            if t_np.ndim == 1:
                # Broadcast time to all batches
                t_np = np.tile(t_np, (n_batch, 1))
            elif t_np.shape != (n_batch, T):
                raise ValueError(f"Time shape {t_np.shape} incompatible with x shape {x_np.shape}")
        else:
            # x: (T, nx)
            if x_np.ndim == 1:
                x_np = x_np[:, None]  # Make (T, 1)
            T, nx = x_np.shape
            n_batch = 1
            if t_np.shape != (T,):
                raise ValueError(f"Time shape {t_np.shape} incompatible with x shape {x_np.shape}")

        # Generate state names if not provided
        if state_names is None:
            state_names = [f"x_{i+1}" for i in range(nx)]
        elif len(state_names) != nx:
            raise ValueError(f"state_names length {len(state_names)} != nx {nx}")

        # Get colors
        colors = self._get_colors(color_scheme, n_batch)

        # Determine subplot layout
        n_rows, n_cols = self._determine_layout(nx)

        # Create subplots
        fig = make_subplots(
            rows=n_rows,
            cols=n_cols,
            subplot_titles=state_names,
            vertical_spacing=0.08,
            horizontal_spacing=0.08,
        )

        # Add trajectory traces
        self._add_trajectory_traces(
            fig, t_np, x_np, state_names, colors, is_batched, n_rows, n_cols
        )

        # Update layout
        fig.update_layout(
            title=title,
            showlegend=show_legend and is_batched,
            height=max(300, 200 * n_rows),
            template="plotly_white",
        )

        # Update axes
        for i in range(nx):
            row = i // n_cols + 1
            col = i % n_cols + 1
            fig.update_xaxes(title_text="Time (s)", row=row, col=col)
            fig.update_yaxes(title_text=state_names[i], row=row, col=col)

        return fig

    def plot_state_and_control(
        self,
        t: np.ndarray,
        x: np.ndarray,
        u: np.ndarray,
        state_names: Optional[List[str]] = None,
        control_names: Optional[List[str]] = None,
        title: str = "State and Control Trajectories",
        color_scheme: str = "plotly",
        **kwargs,
    ) -> go.Figure:
        """
        Plot states and controls in synchronized subplots.

        Creates two subplot groups: states (top) and controls (bottom),
        with synchronized time axes.

        Parameters
        ----------
        t : np.ndarray
            Time points, shape (T,) or (n_batch, T)
        x : np.ndarray
            State trajectories, shape (T, nx) or (n_batch, T, nx)
        u : np.ndarray
            Control inputs, shape (T, nu) or (n_batch, T, nu)
        state_names : Optional[List[str]]
            Names for state variables
        control_names : Optional[List[str]]
            Names for control inputs
        title : str
            Overall plot title
        color_scheme : str
            Color scheme name
        **kwargs
            Additional arguments

        Returns
        -------
        go.Figure
            Plotly figure with state and control subplots

        Examples
        --------
        >>> t = np.linspace(0, 10, 100)
        >>> x = np.column_stack([np.sin(t), np.cos(t)])
        >>> u = 0.1 * np.random.randn(100, 1)
        >>> fig = plotter.plot_state_and_control(
        ...     t, x, u,
        ...     state_names=['Position', 'Velocity'],
        ...     control_names=['Force']
        ... )
        >>> fig.show()
        """
        # Convert to NumPy
        t_np = self._to_numpy(t)
        x_np = self._to_numpy(x)
        u_np = self._to_numpy(u)

        # Detect batching
        is_batched_x = self._is_batched(x_np)
        is_batched_u = self._is_batched(u_np)

        if is_batched_x != is_batched_u:
            raise ValueError("State and control must both be batched or both single")

        is_batched = is_batched_x

        # Get dimensions
        if is_batched:
            n_batch, T, nx = x_np.shape
            _, _, nu = u_np.shape
            if t_np.ndim == 1:
                t_np = np.tile(t_np, (n_batch, 1))
        else:
            if x_np.ndim == 1:
                x_np = x_np[:, None]
            if u_np.ndim == 1:
                u_np = u_np[:, None]
            T, nx = x_np.shape
            _, nu = u_np.shape
            n_batch = 1

        # Generate names
        if state_names is None:
            state_names = [f"x_{i+1}" for i in range(nx)]
        if control_names is None:
            control_names = [f"u_{i+1}" for i in range(nu)]

        # Colors
        colors = self._get_colors(color_scheme, n_batch)

        # Layout: nx states + nu controls
        n_total = nx + nu
        n_rows, n_cols = self._determine_layout(n_total)

        # Create subplots with section titles
        subplot_titles = state_names + control_names

        fig = make_subplots(
            rows=n_rows,
            cols=n_cols,
            subplot_titles=subplot_titles,
            vertical_spacing=0.08,
            horizontal_spacing=0.08,
        )

        # Add state traces
        self._add_trajectory_traces(
            fig, t_np, x_np, state_names, colors, is_batched, n_rows, n_cols, offset=0
        )

        # Add control traces
        self._add_trajectory_traces(
            fig, t_np, u_np, control_names, colors, is_batched, n_rows, n_cols, offset=nx
        )

        # Update layout
        fig.update_layout(
            title=title,
            showlegend=is_batched,
            height=max(400, 200 * n_rows),
            template="plotly_white",
        )

        # Update axes
        for i in range(n_total):
            row = i // n_cols + 1
            col = i % n_cols + 1
            fig.update_xaxes(title_text="Time (s)", row=row, col=col)
            if i < nx:
                fig.update_yaxes(title_text=state_names[i], row=row, col=col)
            else:
                fig.update_yaxes(title_text=control_names[i - nx], row=row, col=col)

        return fig

    def plot_comparison(
        self,
        t: np.ndarray,
        trajectories: Dict[str, np.ndarray],
        state_names: Optional[List[str]] = None,
        title: str = "Trajectory Comparison",
        mode: str = "overlay",
        **kwargs,
    ) -> go.Figure:
        """
        Compare multiple simulation runs.

        Parameters
        ----------
        t : np.ndarray
            Time points, shape (T,)
        trajectories : Dict[str, np.ndarray]
            Dictionary mapping labels to trajectories
            Each trajectory: (T, nx) array
        state_names : Optional[List[str]]
            Names for state variables
        title : str
            Overall plot title
        mode : str
            Comparison mode: 'overlay' or 'side-by-side'
        **kwargs
            Additional arguments

        Returns
        -------
        go.Figure
            Comparison plot

        Examples
        --------
        >>> # Compare controlled vs uncontrolled
        >>> trajectories = {
        ...     'Controlled': x_controlled,
        ...     'Uncontrolled': x_uncontrolled,
        ... }
        >>> fig = plotter.plot_comparison(t, trajectories, mode='overlay')
        >>> fig.show()
        """
        # Convert to NumPy
        t_np = self._to_numpy(t)
        trajectories_np = {label: self._to_numpy(x) for label, x in trajectories.items()}

        # Get dimensions from first trajectory
        first_traj = list(trajectories_np.values())[0]
        if first_traj.ndim == 1:
            first_traj = first_traj[:, None]
        T, nx = first_traj.shape

        # Validate all trajectories have same shape
        for label, x in trajectories_np.items():
            if x.ndim == 1:
                x = x[:, None]
                trajectories_np[label] = x
            if x.shape != (T, nx):
                raise ValueError(
                    f"Trajectory '{label}' shape {x.shape} != expected {(T, nx)}"
                )

        # Generate state names
        if state_names is None:
            state_names = [f"x_{i+1}" for i in range(nx)]

        # Get colors (one per trajectory)
        n_traj = len(trajectories)
        colors = self._get_colors("plotly", n_traj)

        # Determine layout
        n_rows, n_cols = self._determine_layout(nx)

        # Create subplots
        fig = make_subplots(
            rows=n_rows,
            cols=n_cols,
            subplot_titles=state_names,
            vertical_spacing=0.08,
            horizontal_spacing=0.08,
        )

        # Add traces for each trajectory
        for traj_idx, (label, x) in enumerate(trajectories_np.items()):
            for state_idx in range(nx):
                row = state_idx // n_cols + 1
                col = state_idx % n_cols + 1

                fig.add_trace(
                    go.Scatter(
                        x=t_np,
                        y=x[:, state_idx],
                        mode="lines",
                        name=label,
                        line=dict(color=colors[traj_idx], width=2),
                        showlegend=(state_idx == 0),  # Only show legend once per trajectory
                        legendgroup=label,
                    ),
                    row=row,
                    col=col,
                )

        # Update layout
        fig.update_layout(
            title=title,
            showlegend=True,
            height=max(400, 200 * n_rows),
            template="plotly_white",
        )

        # Update axes
        for i in range(nx):
            row = i // n_cols + 1
            col = i % n_cols + 1
            fig.update_xaxes(title_text="Time (s)", row=row, col=col)
            fig.update_yaxes(title_text=state_names[i], row=row, col=col)

        return fig

    # =========================================================================
    # Helper Methods (Internal)
    # =========================================================================

    def _to_numpy(self, arr) -> np.ndarray:
        """
        Convert array from any backend to NumPy.

        Handles NumPy, PyTorch, JAX arrays transparently.

        Parameters
        ----------
        arr : array-like
            Input array (any backend)

        Returns
        -------
        np.ndarray
            NumPy array

        Examples
        --------
        >>> # NumPy (passthrough)
        >>> arr_np = plotter._to_numpy(np.array([1, 2, 3]))
        >>>
        >>> # PyTorch
        >>> import torch
        >>> arr_torch = torch.tensor([1, 2, 3])
        >>> arr_np = plotter._to_numpy(arr_torch)
        >>>
        >>> # JAX
        >>> import jax.numpy as jnp
        >>> arr_jax = jnp.array([1, 2, 3])
        >>> arr_np = plotter._to_numpy(arr_jax)
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
        Detect if trajectory is batched (Monte Carlo).

        Parameters
        ----------
        x : np.ndarray
            State trajectory

        Returns
        -------
        bool
            True if batched (shape: (n_batch, T, nx))
            False if single (shape: (T, nx) or (T,))

        Notes
        -----
        Heuristic: If 3D array, assume batched. If 2D or 1D, assume single.
        """
        if x is None:
            return False

        # 3D array → batched
        if x.ndim == 3:
            return True
        # 2D or 1D → single
        elif x.ndim <= 2:
            return False
        else:
            # 4D+ is unexpected but treat as batched
            return True

    def _determine_layout(self, n_plots: int) -> Tuple[int, int]:
        """
        Determine optimal subplot layout.

        Parameters
        ----------
        n_plots : int
            Number of subplots needed

        Returns
        -------
        Tuple[int, int]
            (n_rows, n_cols) for subplot grid

        Examples
        --------
        >>> plotter._determine_layout(1)
        (1, 1)
        >>> plotter._determine_layout(4)
        (1, 4)
        >>> plotter._determine_layout(5)
        (2, 3)
        >>> plotter._determine_layout(12)
        (3, 4)
        """
        if n_plots == 1:
            return (1, 1)
        elif n_plots <= 4:
            # Single row
            return (1, n_plots)
        elif n_plots <= 8:
            # Two rows
            n_cols = math.ceil(n_plots / 2)
            return (2, n_cols)
        elif n_plots <= 12:
            # Three rows, 4 columns max
            n_cols = min(4, math.ceil(n_plots / 3))
            n_rows = math.ceil(n_plots / n_cols)
            return (n_rows, n_cols)
        else:
            # Square-ish grid
            n_cols = math.ceil(math.sqrt(n_plots))
            n_rows = math.ceil(n_plots / n_cols)
            return (n_rows, n_cols)

    def _get_colors(self, scheme: str, n_colors: int) -> List[str]:
        """
        Get color palette.

        Parameters
        ----------
        scheme : str
            Color scheme name
        n_colors : int
            Number of colors needed

        Returns
        -------
        List[str]
            List of hex color codes

        Notes
        -----
        Currently uses Plotly default colors. In future, will integrate
        with themes.py for more schemes.
        """
        # Plotly default color sequence
        plotly_colors = [
            "#636EFA",
            "#EF553B",
            "#00CC96",
            "#AB63FA",
            "#FFA15A",
            "#19D3F3",
            "#FF6692",
            "#B6E880",
            "#FF97FF",
            "#FECB52",
        ]

        # Cycle through colors if more needed
        colors = []
        for i in range(n_colors):
            colors.append(plotly_colors[i % len(plotly_colors)])

        return colors

    def _add_trajectory_traces(
        self,
        fig: go.Figure,
        t: np.ndarray,
        x: np.ndarray,
        state_names: List[str],
        colors: List[str],
        is_batched: bool,
        n_rows: int,
        n_cols: int,
        offset: int = 0,
    ) -> None:
        """
        Add trajectory traces to figure.

        Parameters
        ----------
        fig : go.Figure
            Plotly figure to add traces to
        t : np.ndarray
            Time points
        x : np.ndarray
            State trajectories
        state_names : List[str]
            Names for states
        colors : List[str]
            Color palette
        is_batched : bool
            Whether trajectories are batched
        n_rows : int
            Number of subplot rows
        n_cols : int
            Number of subplot columns
        offset : int
            Subplot offset (for combined plots)
        """
        if is_batched:
            # x: (n_batch, T, nx)
            n_batch, T, nx = x.shape

            for batch_idx in range(n_batch):
                for state_idx in range(nx):
                    subplot_idx = state_idx + offset
                    row = subplot_idx // n_cols + 1
                    col = subplot_idx % n_cols + 1

                    fig.add_trace(
                        go.Scatter(
                            x=t[batch_idx] if t.ndim == 2 else t,
                            y=x[batch_idx, :, state_idx],
                            mode="lines",
                            name=f"Trajectory {batch_idx + 1}",
                            line=dict(color=colors[batch_idx], width=1.5),
                            showlegend=(state_idx == 0),  # Only show in legend once
                            legendgroup=f"traj_{batch_idx}",
                        ),
                        row=row,
                        col=col,
                    )
        else:
            # x: (T, nx)
            T, nx = x.shape

            for state_idx in range(nx):
                subplot_idx = state_idx + offset
                row = subplot_idx // n_cols + 1
                col = subplot_idx % n_cols + 1

                fig.add_trace(
                    go.Scatter(
                        x=t,
                        y=x[:, state_idx],
                        mode="lines",
                        line=dict(color=colors[0], width=2),
                        showlegend=False,
                    ),
                    row=row,
                    col=col,
                )


# ============================================================================
# Export
# ============================================================================

__all__ = [
    "TrajectoryPlotter",
]
