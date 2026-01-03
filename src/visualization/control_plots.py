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
Control Plotter - Control-Specific Visualizations

Interactive Plotly-based visualization for control system analysis.

Key Features
------------
- Eigenvalue maps: Visualize stability with continuous/discrete regions
- Gain comparison: Compare LQR/control gains across designs
- Riccati convergence: Monitor solver convergence
- Gramian visualization: Controllability/observability heatmaps
- Step response: Closed-loop response with metrics
- Frequency analysis: Bode plots, Nyquist diagrams, root locus
- Consistent theming: Integrates with centralized themes module

Main Class
----------
ControlPlotter : Control system analysis visualization
    plot_eigenvalue_map() : Eigenvalue location with stability regions
    plot_gain_comparison() : Compare feedback gains
    plot_riccati_convergence() : Riccati equation solver convergence
    plot_controllability_gramian() : Gramian heatmap
    plot_observability_gramian() : Observability Gramian heatmap
    plot_step_response() : Step response with performance metrics
    plot_impulse_response() : Impulse response with decay analysis
    plot_frequency_response() : Bode plot (magnitude and phase)
    plot_nyquist() : Nyquist diagram for stability analysis
    plot_root_locus() : Root locus (pole migration vs gain)

Usage
-----
>>> from src.plotting import ControlPlotter
>>>
>>> # Create plotter
>>> plotter = ControlPlotter()
>>>
>>> # Eigenvalue map with custom theme
>>> lqr_result = system.design_lqr(Q, R)
>>> fig = plotter.plot_eigenvalue_map(
...     lqr_result['closed_loop_eigenvalues'],
...     system_type='continuous',
...     theme='publication'
... )
>>> fig.show()
>>>
>>> # Compare gains with colorblind-safe palette
>>> gains = {
...     'Q=10': design_lqr(10*np.eye(2), R)['gain'],
...     'Q=100': design_lqr(100*np.eye(2), R)['gain'],
... }
>>> fig = plotter.plot_gain_comparison(
...     gains,
...     color_scheme='colorblind_safe',
...     theme='publication'
... )
"""

from typing import Dict, List, Optional

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.types.backends import Backend
from src.visualization.themes import ColorSchemes, PlotThemes


class ControlPlotter:
    """
    Control system analysis visualization.

    Provides interactive Plotly-based plotting for control-specific analysis
    including eigenvalue maps, gain comparisons, and performance metrics.

    Attributes
    ----------
    backend : Backend
        Default computational backend for array conversion
    default_theme : str
        Default plot theme to apply

    Examples
    --------
    Eigenvalue stability map:

    >>> plotter = ControlPlotter()
    >>> result = system.design_lqr(Q, R)
    >>> fig = plotter.plot_eigenvalue_map(
    ...     result['closed_loop_eigenvalues'],
    ...     system_type='continuous',
    ...     theme='publication'
    ... )
    >>> fig.show()

    Compare LQR gains:

    >>> gains = {
    ...     'Light Q': K1,
    ...     'Heavy Q': K2,
    ... }
    >>> fig = plotter.plot_gain_comparison(
    ...     gains,
    ...     color_scheme='colorblind_safe',
    ...     theme='dark'
    ... )
    """

    def __init__(self, backend: Backend = "numpy", default_theme: str = "default"):
        """
        Initialize control plotter.

        Parameters
        ----------
        backend : Backend
            Default backend for array conversion ('numpy', 'torch', 'jax')
        default_theme : str
            Default theme to apply to plots
            Options: 'default', 'publication', 'dark', 'presentation'

        Examples
        --------
        >>> plotter = ControlPlotter(backend='numpy')
        >>> plotter = ControlPlotter(backend='torch', default_theme='publication')
        """
        self.backend = backend
        self.default_theme = default_theme

    # =========================================================================
    # Main Plotting Methods
    # =========================================================================

    def plot_eigenvalue_map(
        self,
        eigenvalues: np.ndarray,
        system_type: str = "continuous",
        title: str = "Closed-Loop Eigenvalues",
        show_stability_margin: bool = True,
        theme: Optional[str] = None,
        **kwargs,
    ) -> go.Figure:
        """
        Plot eigenvalues with stability region.

        Creates complex plane plot showing eigenvalue locations with
        stability region highlighted.

        Parameters
        ----------
        eigenvalues : np.ndarray
            Complex eigenvalues, shape (n,)
        system_type : str
            'continuous' or 'discrete'
            Determines stability criterion:
            - Continuous: Re(λ) < 0 (left half-plane)
            - Discrete: |λ| < 1 (inside unit circle)
        title : str
            Plot title
        show_stability_margin : bool
            If True, annotate stability margin
        theme : Optional[str]
            Plot theme to apply
            Options: 'default', 'publication', 'dark', 'presentation'
            If None, uses self.default_theme
        **kwargs
            Additional arguments

        Returns
        -------
        go.Figure
            Eigenvalue map with stability region

        Examples
        --------
        >>> # Continuous system with publication theme
        >>> lqr = system.design_lqr(Q, R)
        >>> fig = plotter.plot_eigenvalue_map(
        ...     lqr['closed_loop_eigenvalues'],
        ...     system_type='continuous',
        ...     theme='publication'
        ... )
        >>> print(f"Margin: {lqr['stability_margin']:.3f}")
        >>>
        >>> # Discrete system with dark theme
        >>> lqr_d = discrete_system.design_lqr(Q, R)
        >>> fig = plotter.plot_eigenvalue_map(
        ...     lqr_d['closed_loop_eigenvalues'],
        ...     system_type='discrete',
        ...     theme='dark'
        ... )

        Notes
        -----
        - Continuous: Stable if all eigenvalues in left half-plane (Re(λ) < 0)
        - Discrete: Stable if all eigenvalues inside unit circle (|λ| < 1)
        - Stability region shown in green
        - Unstable region shown in red
        - Eigenvalues plotted as blue circles
        """
        # Use default theme if not specified
        if theme is None:
            theme = self.default_theme

        # Convert to NumPy
        eigs = self._to_numpy(eigenvalues)

        # Create figure
        fig = go.Figure()

        # Draw stability region
        self._draw_stability_region(fig, system_type)

        # Plot eigenvalues (use ColorSchemes for consistency)
        eigenvalue_color = ColorSchemes.PLOTLY[0]

        fig.add_trace(
            go.Scatter(
                x=np.real(eigs),
                y=np.imag(eigs),
                mode="markers",
                name="Eigenvalues",
                marker=dict(
                    color=eigenvalue_color,
                    size=12,
                    symbol="circle",
                    line=dict(color="white", width=2),
                ),
                hovertemplate="λ = %{x:.4f} + %{y:.4f}j<br>|λ| = %{text:.4f}<extra></extra>",
                text=np.abs(eigs),
            ),
        )

        # Determine plot limits
        if system_type == "continuous":
            # Focus on left half-plane
            real_min = min(-1, np.min(np.real(eigs)) - 0.5)
            real_max = max(1, np.max(np.real(eigs)) + 0.5)
            imag_range = max(2, np.max(np.abs(np.imag(eigs))) + 0.5)
            imag_min, imag_max = -imag_range, imag_range

            # Add stability margin annotation
            if show_stability_margin:
                max_real = np.max(np.real(eigs))
                stability_margin = -max_real

                fig.add_annotation(
                    x=max_real,
                    y=0,
                    text=f"Margin = {stability_margin:.3f}",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    ax=-50,
                    ay=0,
                )
        else:
            # Discrete: focus on unit circle
            max_mag = max(1.2, np.max(np.abs(eigs)) + 0.2)
            real_min, real_max = -max_mag, max_mag
            imag_min, imag_max = -max_mag, max_mag

            # Add stability margin annotation
            if show_stability_margin:
                max_magnitude = np.max(np.abs(eigs))
                stability_margin = 1.0 - max_magnitude

                # Find eigenvalue with max magnitude
                max_idx = np.argmax(np.abs(eigs))
                max_eig = eigs[max_idx]

                fig.add_annotation(
                    x=np.real(max_eig),
                    y=np.imag(max_eig),
                    text=f"Margin = {stability_margin:.3f}",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    ax=50,
                    ay=-50,
                )

        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title="Real Part" if system_type == "continuous" else "Real",
            yaxis_title="Imaginary Part" if system_type == "continuous" else "Imaginary",
            width=700,
            height=600,
            xaxis=dict(range=[real_min, real_max], zeroline=True),
            yaxis=dict(range=[imag_min, imag_max], zeroline=True),
            showlegend=True,
        )

        # Equal aspect ratio
        fig.update_yaxes(scaleanchor="x", scaleratio=1)

        # Apply theme
        fig = PlotThemes.apply_theme(fig, theme=theme)

        return fig

    def plot_gain_comparison(
        self,
        gains: Dict[str, np.ndarray],
        labels: Optional[List[str]] = None,
        title: str = "Feedback Gain Comparison",
        color_scheme: str = "plotly",
        theme: Optional[str] = None,
        **kwargs,
    ) -> go.Figure:
        """
        Compare feedback gains across different designs.

        Creates grouped bar chart or heatmap showing gain values
        for different control designs.

        Parameters
        ----------
        gains : Dict[str, np.ndarray]
            Dictionary mapping design names to gain matrices
            Each gain: (nu, nx) array
        labels : Optional[List[str]]
            Labels for gain entries (state names)
            If None, uses generic labels
        title : str
            Plot title
        color_scheme : str
            Color scheme name for bar charts
            Options: 'plotly', 'd3', 'colorblind_safe', 'tableau', etc.
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
            Gain comparison plot

        Examples
        --------
        >>> # Compare different Q weights with colorblind-safe colors
        >>> gains = {
        ...     'Q=10*I': system.design_lqr(10*np.eye(2), R)['gain'],
        ...     'Q=100*I': system.design_lqr(100*np.eye(2), R)['gain'],
        ...     'Q=1000*I': system.design_lqr(1000*np.eye(2), R)['gain'],
        ... }
        >>> fig = plotter.plot_gain_comparison(
        ...     gains,
        ...     color_scheme='colorblind_safe',
        ...     theme='publication'
        ... )
        >>> fig.show()
        >>>
        >>> # With state labels
        >>> fig = plotter.plot_gain_comparison(
        ...     gains,
        ...     labels=['Position', 'Velocity'],
        ...     theme='dark'
        ... )

        Notes
        -----
        - Each design shown as separate bar group
        - Useful for parameter studies
        - Shows effect of Q/R tuning on gains
        """
        # Use default theme if not specified
        if theme is None:
            theme = self.default_theme

        # Convert all gains to NumPy
        gains_np = {name: self._to_numpy(K) for name, K in gains.items()}

        # Get dimensions from first gain
        first_gain = list(gains_np.values())[0]
        if first_gain.ndim == 1:
            first_gain = first_gain[np.newaxis, :]
            gains_np = {
                name: K[np.newaxis, :] if K.ndim == 1 else K for name, K in gains_np.items()
            }

        nu, nx = first_gain.shape

        # Validate all gains have same shape
        for name, K in gains_np.items():
            if K.shape != (nu, nx):
                raise ValueError(f"Gain '{name}' has shape {K.shape} != expected {(nu, nx)}")

        # Generate labels
        if labels is None:
            labels = [f"x_{i+1}" for i in range(nx)]
        elif len(labels) != nx:
            raise ValueError(f"labels length {len(labels)} != nx {nx}")

        # Get colors from centralized color schemes
        n_designs = len(gains_np)
        colors = ColorSchemes.get_colors(color_scheme, n_designs)

        # Create figure
        if nu == 1:
            # Single control input: bar chart
            fig = go.Figure()

            for idx, (name, K) in enumerate(gains_np.items()):
                fig.add_trace(
                    go.Bar(
                        name=name,
                        x=labels,
                        y=K.flatten(),
                        text=[f"{val:.3f}" for val in K.flatten()],
                        textposition="outside",
                        marker=dict(color=colors[idx]),
                    ),
                )

            fig.update_layout(
                title=title,
                xaxis_title="State",
                yaxis_title="Gain Value",
                barmode="group",
                width=max(500, 100 * nx),
                height=500,
            )
        else:
            # Multiple controls: heatmap for each design
            fig = make_subplots(
                rows=1,
                cols=n_designs,
                subplot_titles=list(gains_np.keys()),
                horizontal_spacing=0.1,
            )

            for col_idx, (name, K) in enumerate(gains_np.items(), start=1):
                fig.add_trace(
                    go.Heatmap(
                        z=K,
                        x=labels,
                        y=[f"u_{i+1}" for i in range(nu)],
                        colorscale="RdBu",
                        zmid=0,
                        showscale=(col_idx == n_designs),
                        text=[[f"{val:.2f}" for val in row] for row in K],
                        texttemplate="%{text}",
                        textfont={"size": 10},
                    ),
                    row=1,
                    col=col_idx,
                )

            fig.update_layout(
                title=title,
                width=max(600, 300 * n_designs),
                height=400,
            )

        # Apply theme
        fig = PlotThemes.apply_theme(fig, theme=theme)

        return fig

    def plot_riccati_convergence(
        self,
        P_history: List[np.ndarray],
        title: str = "Riccati Equation Convergence",
        theme: Optional[str] = None,
        **kwargs,
    ) -> go.Figure:
        """
        Plot Riccati equation solver convergence.

        Shows how Riccati matrix P converges during iterative solution.

        Parameters
        ----------
        P_history : List[np.ndarray]
            List of P matrices at each iteration
            Each P: (nx, nx) array
        title : str
            Plot title
        theme : Optional[str]
            Plot theme to apply
            Options: 'default', 'publication', 'dark', 'presentation'
            If None, uses self.default_theme
        **kwargs
            Additional arguments

        Returns
        -------
        go.Figure
            Convergence plot

        Examples
        --------
        >>> # During iterative Riccati solving
        >>> P_history = []
        >>> P = np.eye(nx)
        >>> for iter in range(100):
        ...     P = riccati_iteration(P, A, B, Q, R)
        ...     P_history.append(P.copy())
        >>>
        >>> fig = plotter.plot_riccati_convergence(
        ...     P_history,
        ...     theme='publication'
        ... )
        >>> fig.show()

        Notes
        -----
        - Plots Frobenius norm vs iteration
        - Shows convergence rate
        - Useful for debugging custom solvers
        """
        # Use default theme if not specified
        if theme is None:
            theme = self.default_theme

        # Convert to NumPy
        P_history_np = [self._to_numpy(P) for P in P_history]

        n_iterations = len(P_history_np)

        # Compute norms
        norms = [np.linalg.norm(P, "fro") for P in P_history_np]

        # Compute differences (convergence)
        diffs = [0.0]  # First iteration has no previous
        for i in range(1, n_iterations):
            diff = np.linalg.norm(P_history_np[i] - P_history_np[i - 1], "fro")
            diffs.append(diff)

        # Create figure with two subplots
        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=("Riccati Matrix Norm", "Iteration-to-Iteration Change"),
        )

        # Get colors
        norm_color = ColorSchemes.PLOTLY[0]
        diff_color = ColorSchemes.PLOTLY[1]

        # Plot norm
        fig.add_trace(
            go.Scatter(
                x=list(range(n_iterations)),
                y=norms,
                mode="lines+markers",
                name="||P||_F",
                line=dict(color=norm_color, width=2),
            ),
            row=1,
            col=1,
        )

        # Plot differences (log scale)
        fig.add_trace(
            go.Scatter(
                x=list(range(n_iterations)),
                y=diffs,
                mode="lines+markers",
                name="||P_k - P_{k-1}||_F",
                line=dict(color=diff_color, width=2),
            ),
            row=1,
            col=2,
        )

        # Update layout
        fig.update_xaxes(title_text="Iteration", row=1, col=1)
        fig.update_xaxes(title_text="Iteration", row=1, col=2)
        fig.update_yaxes(title_text="Frobenius Norm", row=1, col=1)
        fig.update_yaxes(title_text="Change", type="log", row=1, col=2)

        fig.update_layout(
            title=title,
            width=1000,
            height=400,
            showlegend=False,
        )

        # Apply theme
        fig = PlotThemes.apply_theme(fig, theme=theme)

        return fig

    def plot_controllability_gramian(
        self,
        W_c: np.ndarray,
        state_names: Optional[List[str]] = None,
        title: str = "Controllability Gramian",
        theme: Optional[str] = None,
        **kwargs,
    ) -> go.Figure:
        """
        Visualize controllability Gramian as heatmap.

        Shows coupling between states for controllability analysis.

        Parameters
        ----------
        W_c : np.ndarray
            Controllability Gramian, shape (nx, nx)
            Symmetric positive semi-definite matrix
        state_names : Optional[List[str]]
            Names for states
        title : str
            Plot title
        theme : Optional[str]
            Plot theme to apply
            Options: 'default', 'publication', 'dark', 'presentation'
            If None, uses self.default_theme
        **kwargs
            Additional arguments

        Returns
        -------
        go.Figure
            Gramian heatmap

        Examples
        --------
        >>> # Compute controllability Gramian
        >>> from scipy.linalg import solve_continuous_lyapunov
        >>> W_c = solve_continuous_lyapunov(A, -B @ B.T)
        >>>
        >>> fig = plotter.plot_controllability_gramian(
        ...     W_c,
        ...     state_names=['Position', 'Velocity'],
        ...     theme='publication'
        ... )
        >>> fig.show()

        Notes
        -----
        - Diagonal elements: controllability of individual states
        - Off-diagonal: coupling between states
        - Small eigenvalues → difficult to control
        - Can also visualize observability Gramian W_o
        """
        # Use default theme if not specified
        if theme is None:
            theme = self.default_theme

        # Convert to NumPy
        W_np = self._to_numpy(W_c)

        # Get dimension
        nx = W_np.shape[0]

        # Validate
        if W_np.shape != (nx, nx):
            raise ValueError(f"Gramian must be square, got shape {W_np.shape}")

        # Generate state names
        if state_names is None:
            state_names = [f"x_{i+1}" for i in range(nx)]
        elif len(state_names) != nx:
            raise ValueError(f"state_names length {len(state_names)} != nx {nx}")

        # Create heatmap
        fig = go.Figure(
            data=go.Heatmap(
                z=W_np,
                x=state_names,
                y=state_names,
                colorscale="Viridis",
                text=[[f"{val:.2e}" for val in row] for row in W_np],
                texttemplate="%{text}",
                textfont={"size": 10},
                colorbar=dict(title="Value"),
            ),
        )

        fig.update_layout(
            title=title,
            xaxis_title="State",
            yaxis_title="State",
            width=max(500, 80 * nx),
            height=max(450, 80 * nx),
        )

        # Apply theme
        fig = PlotThemes.apply_theme(fig, theme=theme)

        return fig

    def plot_observability_gramian(
        self,
        W_o: np.ndarray,
        state_names: Optional[List[str]] = None,
        title: str = "Observability Gramian",
        theme: Optional[str] = None,
        **kwargs,
    ) -> go.Figure:
        """
        Visualize observability Gramian as heatmap.

        Shows coupling between states for observability analysis.

        Parameters
        ----------
        W_o : np.ndarray
            Observability Gramian, shape (nx, nx)
        state_names : Optional[List[str]]
            Names for states
        title : str
            Plot title
        theme : Optional[str]
            Plot theme to apply
            Options: 'default', 'publication', 'dark', 'presentation'
            If None, uses self.default_theme
        **kwargs
            Additional arguments

        Returns
        -------
        go.Figure
            Gramian heatmap

        Examples
        --------
        >>> # Compute observability Gramian
        >>> from scipy.linalg import solve_continuous_lyapunov
        >>> W_o = solve_continuous_lyapunov(A.T, -C.T @ C)
        >>>
        >>> fig = plotter.plot_observability_gramian(
        ...     W_o,
        ...     theme='publication'
        ... )
        """
        # Just use controllability Gramian plotter with different title
        return self.plot_controllability_gramian(W_o, state_names, title, theme, **kwargs)

    def plot_step_response(
        self,
        t: np.ndarray,
        y: np.ndarray,
        reference: float = 1.0,
        show_metrics: bool = True,
        title: str = "Step Response",
        theme: Optional[str] = None,
        **kwargs,
    ) -> go.Figure:
        """
        Plot step response with performance metrics.

        Shows closed-loop step response with annotations for
        rise time, settling time, overshoot, etc.

        Parameters
        ----------
        t : np.ndarray
            Time points, shape (T,)
        y : np.ndarray
            Output response, shape (T,) or (T, ny)
        reference : float
            Reference value (step height)
        show_metrics : bool
            If True, annotate performance metrics
        title : str
            Plot title
        theme : Optional[str]
            Plot theme to apply
            Options: 'default', 'publication', 'dark', 'presentation'
            If None, uses self.default_theme
        **kwargs
            Additional arguments

        Returns
        -------
        go.Figure
            Step response plot with metrics

        Examples
        --------
        >>> # Simulate closed-loop step response
        >>> A_cl = A - B @ K  # Closed-loop A matrix
        >>> result = system.integrate(x0, u=None, A_override=A_cl, t_span=(0, 10))
        >>> y = result['x'][:, 0]  # First state
        >>>
        >>> fig = plotter.plot_step_response(
        ...     result['t'],
        ...     y,
        ...     reference=1.0,
        ...     show_metrics=True,
        ...     theme='publication'
        ... )

        Notes
        -----
        - Rise time: 10% to 90% of final value
        - Settling time: Within 2% of final value
        - Overshoot: Peak value above reference
        - Steady-state error: Final value vs reference
        """
        # Use default theme if not specified
        if theme is None:
            theme = self.default_theme

        # Convert to NumPy
        t_np = self._to_numpy(t)
        y_np = self._to_numpy(y)

        # Handle multidimensional output
        if y_np.ndim > 1:
            y_np = y_np[:, 0]  # Take first output

        # Create figure
        fig = go.Figure()

        # Get colors
        response_color = ColorSchemes.PLOTLY[0]

        # Plot response
        fig.add_trace(
            go.Scatter(
                x=t_np,
                y=y_np,
                mode="lines",
                name="Response",
                line=dict(color=response_color, width=2),
            ),
        )

        # Plot reference
        fig.add_trace(
            go.Scatter(
                x=t_np,
                y=[reference] * len(t_np),
                mode="lines",
                name="Reference",
                line=dict(color="gray", width=2, dash="dash"),
            ),
        )

        # Calculate and annotate metrics
        if show_metrics:
            final_value = y_np[-1]
            peak_value = np.max(y_np)
            peak_time = t_np[np.argmax(y_np)]

            # Overshoot
            overshoot = (peak_value - reference) / reference * 100

            # Rise time (10% to 90%)
            threshold_10 = 0.1 * reference
            threshold_90 = 0.9 * reference
            idx_10 = np.where(y_np >= threshold_10)[0]
            idx_90 = np.where(y_np >= threshold_90)[0]
            if len(idx_10) > 0 and len(idx_90) > 0:
                rise_time = t_np[idx_90[0]] - t_np[idx_10[0]]
            else:
                rise_time = None

            # Settling time (2% criterion)
            settling_band = 0.02 * reference
            settled = np.abs(y_np - reference) < settling_band
            if np.any(settled):
                settling_idx = np.where(settled)[0][0]
                # Check if stays settled
                if np.all(settled[settling_idx:]):
                    settling_time = t_np[settling_idx]
                else:
                    settling_time = None
            else:
                settling_time = None

            # Steady-state error
            ss_error = reference - final_value

            # Add metrics text
            metrics_text = "<b>Performance Metrics:</b><br>"
            if rise_time is not None:
                metrics_text += f"Rise Time: {rise_time:.3f} s<br>"
            if settling_time is not None:
                metrics_text += f"Settling Time: {settling_time:.3f} s<br>"
            metrics_text += f"Overshoot: {overshoot:.2f}%<br>"
            metrics_text += f"Peak Time: {peak_time:.3f} s<br>"
            metrics_text += f"SS Error: {ss_error:.3f}"

            fig.add_annotation(
                text=metrics_text,
                xref="paper",
                yref="paper",
                x=0.98,
                y=0.02,
                xanchor="right",
                yanchor="bottom",
                showarrow=False,
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="black",
                borderwidth=1,
            )

        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title="Time (s)",
            yaxis_title="Output",
            width=800,
            height=500,
            showlegend=True,
        )

        # Apply theme
        fig = PlotThemes.apply_theme(fig, theme=theme)

        return fig

    def plot_impulse_response(
        self,
        t: np.ndarray,
        y: np.ndarray,
        show_metrics: bool = True,
        title: str = "Impulse Response",
        theme: Optional[str] = None,
        **kwargs,
    ) -> go.Figure:
        """
        Plot impulse response with performance metrics.

        Shows closed-loop impulse response (response to Dirac delta input)
        with annotations for peak, decay rate, and settling characteristics.

        Parameters
        ----------
        t : np.ndarray
            Time points, shape (T,)
        y : np.ndarray
            Output response, shape (T,) or (T, ny)
        show_metrics : bool
            If True, annotate performance metrics
        title : str
            Plot title
        theme : Optional[str]
            Plot theme to apply
            Options: 'default', 'publication', 'dark', 'presentation'
            If None, uses self.default_theme
        **kwargs
            Additional arguments

        Returns
        -------
        go.Figure
            Impulse response plot with metrics

        Examples
        --------
        >>> # Simulate closed-loop impulse response
        >>> # For continuous systems: y(t) = C @ expm(A_cl*t) @ B
        >>> from scipy.linalg import expm
        >>> A_cl = A - B @ K
        >>> t = np.linspace(0, 10, 1000)
        >>> y = np.array([C @ expm(A_cl * t_i) @ B for t_i in t]).flatten()
        >>>
        >>> fig = plotter.plot_impulse_response(
        ...     t, y,
        ...     show_metrics=True,
        ...     theme='publication'
        ... )
        >>>
        >>> # For discrete systems: simulate with impulse at k=0
        >>> x = np.zeros(nx)
        >>> y_discrete = []
        >>> for k in range(N):
        ...     u_k = 1.0 if k == 0 else 0.0  # Impulse at k=0
        ...     y_discrete.append(C @ x)
        ...     x = A_cl @ x + B * u_k
        >>> fig = plotter.plot_impulse_response(t_discrete, np.array(y_discrete))

        Notes
        -----
        - Peak value: Maximum absolute response
        - Peak time: Time to peak
        - Decay rate: Exponential decay constant (if applicable)
        - Settling time: Time to settle within 2% of zero
        - Energy: Integral of squared response (L2 norm)
        """
        # Use default theme if not specified
        if theme is None:
            theme = self.default_theme

        # Convert to NumPy
        t_np = self._to_numpy(t)
        y_np = self._to_numpy(y)

        # Handle multidimensional output
        if y_np.ndim > 1:
            y_np = y_np[:, 0]  # Take first output

        # Create figure
        fig = go.Figure()

        # Get colors
        response_color = ColorSchemes.PLOTLY[0]

        # Plot response
        fig.add_trace(
            go.Scatter(
                x=t_np,
                y=y_np,
                mode="lines",
                name="Impulse Response",
                line=dict(color=response_color, width=2),
            ),
        )

        # Plot zero line
        fig.add_trace(
            go.Scatter(
                x=t_np,
                y=[0] * len(t_np),
                mode="lines",
                name="Zero",
                line=dict(color="gray", width=1, dash="dash"),
                showlegend=False,
            ),
        )

        # Calculate and annotate metrics
        if show_metrics:
            # Peak value and time
            abs_y = np.abs(y_np)
            peak_value = np.max(abs_y)
            peak_idx = np.argmax(abs_y)
            peak_time = t_np[peak_idx]

            # Settling time (2% criterion - settling to near zero)
            settling_threshold = 0.02 * peak_value
            settled = abs_y < settling_threshold
            if np.any(settled):
                settling_idx = np.where(settled)[0][0]
                # Check if stays settled
                if np.all(settled[settling_idx:]):
                    settling_time = t_np[settling_idx]
                else:
                    settling_time = None
            else:
                settling_time = None

            # Energy (L2 norm - integral of squared response)
            dt = np.mean(np.diff(t_np)) if len(t_np) > 1 else 1.0
            energy = np.sum(y_np**2) * dt

            # Estimate decay rate (if response decays exponentially)
            decay_rate = None
            if len(t_np) > 20 and peak_idx < len(t_np) - 10:
                # Use data after peak
                t_decay = t_np[peak_idx:]
                y_decay = abs_y[peak_idx:]

                # Only fit if response is actually decaying
                if y_decay[-1] < 0.5 * y_decay[0]:
                    # Avoid log of zero/negative
                    y_decay_safe = np.maximum(y_decay, 1e-10)
                    try:
                        # Linear fit in log space: log(y) = log(A) - alpha*t
                        coeffs = np.polyfit(t_decay, np.log(y_decay_safe), 1)
                        decay_rate = -coeffs[0]  # -alpha
                        if decay_rate < 0:  # Growing, not decaying
                            decay_rate = None
                    except:
                        decay_rate = None

            # Add metrics text
            metrics_text = "<b>Impulse Response Metrics:</b><br>"
            metrics_text += f"Peak: {peak_value:.3f}<br>"
            metrics_text += f"Peak Time: {peak_time:.3f} s<br>"
            if settling_time is not None:
                metrics_text += f"Settling Time: {settling_time:.3f} s<br>"
            if decay_rate is not None and decay_rate > 0:
                metrics_text += f"Decay Rate: {decay_rate:.3f} 1/s<br>"
                metrics_text += f"Time Constant: {1/decay_rate:.3f} s<br>"
            metrics_text += f"Energy (L²): {energy:.3f}"

            fig.add_annotation(
                text=metrics_text,
                xref="paper",
                yref="paper",
                x=0.98,
                y=0.98,
                xanchor="right",
                yanchor="top",
                showarrow=False,
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="black",
                borderwidth=1,
            )

            # Mark peak with annotation
            fig.add_annotation(
                x=peak_time,
                y=y_np[peak_idx],
                text=f"Peak: {y_np[peak_idx]:.3f}",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                ax=0,
                ay=-40,
            )

        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title="Time (s)",
            yaxis_title="Output",
            width=800,
            height=500,
            showlegend=True,
        )

        # Apply theme
        fig = PlotThemes.apply_theme(fig, theme=theme)

        return fig

    def plot_frequency_response(
        self,
        frequencies: np.ndarray,
        magnitude: np.ndarray,
        phase: np.ndarray,
        title: str = "Frequency Response (Bode Plot)",
        show_margins: bool = True,
        theme: Optional[str] = None,
        **kwargs,
    ) -> go.Figure:
        """
        Plot frequency response (Bode plot).

        Creates magnitude and phase plots vs frequency, with optional
        gain and phase margin annotations.

        Parameters
        ----------
        frequencies : np.ndarray
            Frequency points in rad/s, shape (n_freq,)
        magnitude : np.ndarray
            Magnitude response in dB, shape (n_freq,)
        phase : np.ndarray
            Phase response in degrees, shape (n_freq,)
        title : str
            Plot title
        show_margins : bool
            If True, annotate gain and phase margins
        theme : Optional[str]
            Plot theme to apply
            Options: 'default', 'publication', 'dark', 'presentation'
            If None, uses self.default_theme
        **kwargs
            Additional arguments

        Returns
        -------
        go.Figure
            Bode plot with magnitude and phase subplots

        Examples
        --------
        >>> # Compute frequency response
        >>> from scipy import signal
        >>>
        >>> # Closed-loop transfer function
        >>> A_cl = A - B @ K
        >>> sys = signal.StateSpace(A_cl, B, C, D)
        >>>
        >>> # Frequency response
        >>> w = np.logspace(-2, 2, 1000)  # rad/s
        >>> w, H = signal.freqresp(sys, w)
        >>>
        >>> # Convert to dB and degrees
        >>> mag_dB = 20 * np.log10(np.abs(H).flatten())
        >>> phase_deg = np.angle(H, deg=True).flatten()
        >>>
        >>> # Plot with publication theme
        >>> fig = plotter.plot_frequency_response(
        ...     w, mag_dB, phase_deg,
        ...     theme='publication'
        ... )
        >>> fig.show()

        Notes
        -----
        - Magnitude in dB: 20*log10(|H(jω)|)
        - Phase in degrees: ∠H(jω)
        - Gain margin: Amount gain can increase before instability
        - Phase margin: Amount phase can decrease before instability
        - Crossover frequencies automatically detected
        """
        # Use default theme if not specified
        if theme is None:
            theme = self.default_theme

        # Convert to NumPy
        freq_np = self._to_numpy(frequencies)
        mag_np = self._to_numpy(magnitude)
        phase_np = self._to_numpy(phase)

        # Create subplots
        fig = make_subplots(
            rows=2,
            cols=1,
            subplot_titles=("Magnitude", "Phase"),
            vertical_spacing=0.12,
            shared_xaxes=True,
        )

        # Get colors
        mag_color = ColorSchemes.PLOTLY[0]
        phase_color = ColorSchemes.PLOTLY[1]

        # Magnitude plot
        fig.add_trace(
            go.Scatter(
                x=freq_np,
                y=mag_np,
                mode="lines",
                name="Magnitude",
                line=dict(color=mag_color, width=2),
                showlegend=False,
            ),
            row=1,
            col=1,
        )

        # Phase plot
        fig.add_trace(
            go.Scatter(
                x=freq_np,
                y=phase_np,
                mode="lines",
                name="Phase",
                line=dict(color=phase_color, width=2),
                showlegend=False,
            ),
            row=2,
            col=1,
        )

        # Add 0 dB line to magnitude plot
        fig.add_hline(
            y=0,
            line_dash="dash",
            line_color="gray",
            row=1,
            col=1,
            opacity=0.5,
        )

        # Add -180° line to phase plot
        fig.add_hline(
            y=-180,
            line_dash="dash",
            line_color="gray",
            row=2,
            col=1,
            opacity=0.5,
        )

        # Calculate and annotate margins
        if show_margins:
            # Find gain crossover (where |H| = 1, i.e., mag_dB = 0)
            zero_crossings = np.where(np.diff(np.sign(mag_np)))[0]
            if len(zero_crossings) > 0:
                # Use first crossing
                idx_gc = zero_crossings[0]
                freq_gc = freq_np[idx_gc]
                phase_gc = phase_np[idx_gc]

                # Phase margin = 180° + phase at gain crossover
                phase_margin = 180 + phase_gc

                # Annotate gain crossover frequency
                fig.add_annotation(
                    x=np.log10(freq_gc),
                    y=0,
                    text=f"ω_gc = {freq_gc:.2f} rad/s<br>PM = {phase_margin:.1f}°",
                    showarrow=True,
                    arrowhead=2,
                    ax=0,
                    ay=-40,
                    row=1,
                    col=1,
                )

            # Find phase crossover (where phase = -180°)
            phase_cross = np.where(np.diff(np.sign(phase_np + 180)))[0]
            if len(phase_cross) > 0:
                # Use first crossing
                idx_pc = phase_cross[0]
                freq_pc = freq_np[idx_pc]
                mag_pc = mag_np[idx_pc]

                # Gain margin = -mag_dB at phase crossover
                gain_margin = -mag_pc

                # Annotate phase crossover frequency
                fig.add_annotation(
                    x=np.log10(freq_pc),
                    y=-180,
                    text=f"ω_pc = {freq_pc:.2f} rad/s<br>GM = {gain_margin:.1f} dB",
                    showarrow=True,
                    arrowhead=2,
                    ax=0,
                    ay=40,
                    row=2,
                    col=1,
                )

        # Update axes
        fig.update_xaxes(
            title_text="Frequency (rad/s)",
            type="log",
            showgrid=True,
            row=2,
            col=1,
        )
        fig.update_xaxes(type="log", showgrid=True, row=1, col=1)

        fig.update_yaxes(title_text="Magnitude (dB)", showgrid=True, row=1, col=1)
        fig.update_yaxes(title_text="Phase (deg)", showgrid=True, row=2, col=1)

        # Update layout
        fig.update_layout(
            title=title,
            width=800,
            height=700,
            showlegend=False,
        )

        # Apply theme
        fig = PlotThemes.apply_theme(fig, theme=theme)

        return fig

    def plot_nyquist(
        self,
        real: np.ndarray,
        imag: np.ndarray,
        frequencies: Optional[np.ndarray] = None,
        title: str = "Nyquist Plot",
        show_critical_point: bool = True,
        theme: Optional[str] = None,
        **kwargs,
    ) -> go.Figure:
        """
        Plot Nyquist diagram.

        Shows frequency response in complex plane, useful for stability analysis
        via Nyquist stability criterion.

        Parameters
        ----------
        real : np.ndarray
            Real part of frequency response, shape (n_freq,)
        imag : np.ndarray
            Imaginary part of frequency response, shape (n_freq,)
        frequencies : Optional[np.ndarray]
            Frequency points in rad/s (for hover info)
        title : str
            Plot title
        show_critical_point : bool
            If True, mark critical point (-1, 0j)
        theme : Optional[str]
            Plot theme to apply
            Options: 'default', 'publication', 'dark', 'presentation'
            If None, uses self.default_theme
        **kwargs
            Additional arguments

        Returns
        -------
        go.Figure
            Nyquist plot

        Examples
        --------
        >>> # Compute open-loop frequency response
        >>> from scipy import signal
        >>>
        >>> # Open-loop system: G(s) = C(sI - A)^(-1)B
        >>> sys_ol = signal.StateSpace(A, B, C, D)
        >>>
        >>> # Frequency response
        >>> w = np.logspace(-2, 2, 1000)
        >>> w, H = signal.freqresp(sys_ol, w)
        >>> H = H.flatten()
        >>>
        >>> # Plot Nyquist with dark theme
        >>> fig = plotter.plot_nyquist(
        ...     np.real(H), np.imag(H), frequencies=w,
        ...     theme='dark'
        ... )
        >>> fig.show()

        Notes
        -----
        - Nyquist plot: H(jω) in complex plane as ω varies
        - Critical point: (-1, 0j)
        - Stability: Number of encirclements of (-1, 0j) determines stability
        - Gain margin: Distance from curve to (-1, 0j)
        - Phase margin: Angle from curve to (-1, 0j)
        """
        # Use default theme if not specified
        if theme is None:
            theme = self.default_theme

        # Convert to NumPy
        real_np = self._to_numpy(real)
        imag_np = self._to_numpy(imag)

        # Create figure
        fig = go.Figure()

        # Get colors
        nyquist_color = ColorSchemes.PLOTLY[0]

        # Plot Nyquist curve (positive frequencies)
        fig.add_trace(
            go.Scatter(
                x=real_np,
                y=imag_np,
                mode="lines+markers",
                name="H(jω) (ω > 0)",
                line=dict(color=nyquist_color, width=2),
                marker=dict(size=4),
                hovertemplate="Re: %{x:.3f}<br>Im: %{y:.3f}<extra></extra>",
            ),
        )

        # Plot mirror for negative frequencies (complex conjugate)
        fig.add_trace(
            go.Scatter(
                x=real_np,
                y=-imag_np,
                mode="lines+markers",
                name="H(jω) (ω < 0)",
                line=dict(color=nyquist_color, width=2, dash="dot"),
                marker=dict(size=4),
                hovertemplate="Re: %{x:.3f}<br>Im: %{y:.3f}<extra></extra>",
            ),
        )

        # Mark critical point (-1, 0j)
        if show_critical_point:
            fig.add_trace(
                go.Scatter(
                    x=[-1],
                    y=[0],
                    mode="markers",
                    name="Critical Point",
                    marker=dict(
                        color="red",
                        size=15,
                        symbol="x",
                        line=dict(color="red", width=2),
                    ),
                ),
            )

            # Add circle around critical point
            theta = np.linspace(0, 2 * np.pi, 100)
            radius = 0.3
            circle_x = -1 + radius * np.cos(theta)
            circle_y = radius * np.sin(theta)

            fig.add_trace(
                go.Scatter(
                    x=circle_x,
                    y=circle_y,
                    mode="lines",
                    line=dict(color="red", width=1, dash="dash"),
                    showlegend=False,
                    hoverinfo="skip",
                ),
            )

        # Add real and imaginary axes
        fig.add_hline(y=0, line_dash="solid", line_color="black", line_width=1)
        fig.add_vline(x=0, line_dash="solid", line_color="black", line_width=1)

        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title="Real Part",
            yaxis_title="Imaginary Part",
            width=700,
            height=700,
            showlegend=True,
        )

        # Equal aspect ratio
        fig.update_yaxes(scaleanchor="x", scaleratio=1)

        # Apply theme
        fig = PlotThemes.apply_theme(fig, theme=theme)

        return fig

    def plot_root_locus(
        self,
        root_locus_data: Dict[str, np.ndarray],
        title: str = "Root Locus",
        show_grid: bool = True,
        system_type: str = "continuous",
        color_scheme: str = "plotly",
        theme: Optional[str] = None,
        **kwargs,
    ) -> go.Figure:
        """
        Plot root locus (pole migration as gain varies).

        Shows how closed-loop poles move in complex plane as control
        gain varies from 0 to infinity.

        Parameters
        ----------
        root_locus_data : Dict[str, np.ndarray]
            Dictionary with:
            - 'gains': Array of gain values, shape (n_gains,)
            - 'poles': Array of poles, shape (n_gains, n_poles)
            - Optional 'zeros': Open-loop zeros
        title : str
            Plot title
        show_grid : bool
            If True, show stability grid
        system_type : str
            'continuous' or 'discrete' (affects stability region)
        color_scheme : str
            Color scheme for pole branches
            Options: 'plotly', 'd3', 'colorblind_safe', 'tableau', etc.
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
            Root locus plot

        Examples
        --------
        >>> # Compute root locus for LQR as Q varies
        >>> from scipy import signal
        >>>
        >>> gains = np.logspace(-1, 3, 50)  # Q weight values
        >>> poles_list = []
        >>>
        >>> for q in gains:
        ...     lqr = system.design_lqr(q * np.eye(nx), R)
        ...     poles_list.append(lqr['closed_loop_eigenvalues'])
        >>>
        >>> root_locus_data = {
        ...     'gains': gains,
        ...     'poles': np.array(poles_list)
        ... }
        >>>
        >>> fig = plotter.plot_root_locus(
        ...     root_locus_data,
        ...     system_type='continuous',
        ...     color_scheme='colorblind_safe',
        ...     theme='publication'
        ... )
        >>> fig.show()

        Notes
        -----
        - Each branch shows one pole's trajectory
        - Starts at open-loop pole (K=0)
        - Ends at zero or infinity (K→∞)
        - Stability: poles must stay in stable region
        - Continuous: left half-plane (Re < 0)
        - Discrete: inside unit circle (|z| < 1)
        """
        # Use default theme if not specified
        if theme is None:
            theme = self.default_theme

        # Extract data
        gains = self._to_numpy(root_locus_data["gains"])
        poles = self._to_numpy(root_locus_data["poles"])

        # Validate
        if poles.ndim != 2:
            raise ValueError(f"poles must be 2D (n_gains, n_poles), got {poles.shape}")

        n_gains, n_poles = poles.shape

        # Get colors for pole branches
        colors = ColorSchemes.get_colors(color_scheme, n_poles)

        # Create figure
        fig = go.Figure()

        # Draw stability region
        if show_grid:
            self._draw_stability_region(fig, system_type)

        # Plot each pole branch
        for pole_idx in range(n_poles):
            pole_branch = poles[:, pole_idx]

            # Main trajectory
            fig.add_trace(
                go.Scatter(
                    x=np.real(pole_branch),
                    y=np.imag(pole_branch),
                    mode="lines+markers",
                    name=f"Pole {pole_idx + 1}",
                    line=dict(color=colors[pole_idx], width=2),
                    marker=dict(size=4),
                    showlegend=(pole_idx < 5),  # Limit legend clutter
                ),
            )

            # Mark start (K=0, open-loop pole)
            fig.add_trace(
                go.Scatter(
                    x=[np.real(pole_branch[0])],
                    y=[np.imag(pole_branch[0])],
                    mode="markers",
                    marker=dict(
                        color="green",
                        size=10,
                        symbol="circle",
                        line=dict(width=2),
                    ),
                    showlegend=(pole_idx == 0),
                    name="K=0" if pole_idx == 0 else None,
                ),
            )

            # Mark end (K→∞)
            fig.add_trace(
                go.Scatter(
                    x=[np.real(pole_branch[-1])],
                    y=[np.imag(pole_branch[-1])],
                    mode="markers",
                    marker=dict(
                        color="red",
                        size=10,
                        symbol="square",
                        line=dict(width=2),
                    ),
                    showlegend=(pole_idx == 0),
                    name="K→∞" if pole_idx == 0 else None,
                ),
            )

        # Add zeros if provided
        if "zeros" in root_locus_data:
            zeros = self._to_numpy(root_locus_data["zeros"])
            fig.add_trace(
                go.Scatter(
                    x=np.real(zeros),
                    y=np.imag(zeros),
                    mode="markers",
                    name="Zeros",
                    marker=dict(color="black", size=12, symbol="circle-open"),
                ),
            )

        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title="Real Part" if system_type == "continuous" else "Real",
            yaxis_title="Imaginary Part" if system_type == "continuous" else "Imaginary",
            width=800,
            height=700,
            showlegend=True,
        )

        # Equal aspect ratio
        fig.update_yaxes(scaleanchor="x", scaleratio=1)

        # Apply theme
        fig = PlotThemes.apply_theme(fig, theme=theme)

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

    def _draw_stability_region(self, fig: go.Figure, system_type: str) -> None:
        """
        Draw stability region on eigenvalue plot.

        Parameters
        ----------
        fig : go.Figure
            Figure to add region to
        system_type : str
            'continuous' or 'discrete'
        """
        if system_type == "continuous":
            # Continuous: left half-plane (Re(λ) < 0)
            # Draw vertical line at Re(λ) = 0 (imaginary axis)
            fig.add_vline(
                x=0,
                line_width=2,
                line_dash="solid",
                line_color="black",
                annotation_text="Stability Boundary",
                annotation_position="top",
            )

            # Shade stable region (left half-plane, green)
            fig.add_vrect(
                x0=-10,
                x1=0,
                fillcolor="green",
                opacity=0.1,
                layer="below",
                line_width=0,
                annotation_text="Stable",
                annotation_position="top left",
            )

            # Shade unstable region (right half-plane, red)
            fig.add_vrect(
                x0=0,
                x1=10,
                fillcolor="red",
                opacity=0.1,
                layer="below",
                line_width=0,
                annotation_text="Unstable",
                annotation_position="top right",
            )

        elif system_type == "discrete":
            # Discrete: unit circle (|λ| < 1)
            # Draw unit circle
            theta = np.linspace(0, 2 * np.pi, 100)
            x_circle = np.cos(theta)
            y_circle = np.sin(theta)

            fig.add_trace(
                go.Scatter(
                    x=x_circle,
                    y=y_circle,
                    mode="lines",
                    name="Unit Circle",
                    line=dict(color="black", width=2, dash="solid"),
                    showlegend=True,
                ),
            )

            # Annotate regions
            fig.add_annotation(
                x=0,
                y=0,
                text="Stable<br>(inside)",
                showarrow=False,
                font=dict(size=12, color="green"),
            )

            fig.add_annotation(
                x=1.5,
                y=0,
                text="Unstable<br>(outside)",
                showarrow=False,
                font=dict(size=12, color="red"),
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
        >>> themes = ControlPlotter.list_available_themes()
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
        >>> schemes = ControlPlotter.list_available_color_schemes()
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
    "ControlPlotter",
]
