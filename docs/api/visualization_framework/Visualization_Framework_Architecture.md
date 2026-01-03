# Visualization Framework Architecture

## Overview

The **Visualization Framework** provides interactive, publication-ready plotting for dynamical systems analysis. It consists of **4 core modules** totaling ~4,422 lines organized into a **2-layer architecture**: centralized theming + specialized plotters.

## Architecture Philosophy

**Themeable Interactive Visualization** - The visualization framework enables:

1. **Interactive Plots** - Plotly-based with zoom, pan, hover tooltips
2. **Centralized Theming** - Consistent colors and styles across all plots
3. **Backend Agnostic** - Works with NumPy, PyTorch, JAX seamlessly
4. **Specialized Plotters** - Dedicated classes for different visualization types
5. **Publication Ready** - High-quality output for papers and presentations
6. **System Integration** - Clean `system.plotter` and `system.control_plotter` APIs
7. **Accessible Design** - Colorblind-safe palettes available

```python
# Consistent theming across all plots
plotter = TrajectoryPlotter()
fig = plotter.plot_trajectory(t, x, theme='publication', color_scheme='colorblind_safe')
fig.show()
```

## Framework Layers

```
┌────────────────────────────────────────────────────────────┐
│                   APPLICATION LAYER                         │
│         (ContinuousSystemBase, DiscreteSystemBase)          │
│                                                              │
│  system.plotter          ──────► TrajectoryPlotter          │
│  system.phase_plotter    ──────► PhasePortraitPlotter       │
│  system.control_plotter  ──────► ControlPlotter             │
└──────────────────┬─────────────────────────────────────────┘
                   │ use theming from
                   ↓
┌────────────────────────────────────────────────────────────┐
│                   THEMING LAYER                             │
│                   themes.py (623 lines)                     │
│                                                              │
│  ColorSchemes:            PlotThemes:                       │
│  • PLOTLY                • DEFAULT                          │
│  • D3                    • PUBLICATION                      │
│  • COLORBLIND_SAFE       • DARK                             │
│  • TABLEAU                                                  │
│  • SEQUENTIAL_BLUE                                          │
│  • DIVERGING_RED_BLUE                                       │
│                                                              │
│  Utilities: lighten_color(), darken_color(), interpolate() │
└──────────────────┬─────────────────────────────────────────┘
                   │ used by
                   ↓
┌────────────────────────────────────────────────────────────┐
│                   PLOTTER LAYER                             │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ TrajectoryPlotter (914 lines)                        │  │
│  │ • plot_trajectory()        - State vs time           │  │
│  │ • plot_state_and_control() - Combined view           │  │
│  │ • plot_comparison()        - Multiple runs           │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ PhasePortraitPlotter (1,027 lines)                   │  │
│  │ • plot_2d()                - 2D phase portrait       │  │
│  │ • plot_3d()                - 3D phase portrait       │  │
│  │ • plot_limit_cycle()       - Periodic orbits         │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ ControlPlotter (1,858 lines)                         │  │
│  │ • plot_eigenvalue_map()         - Stability regions  │  │
│  │ • plot_gain_comparison()        - Compare gains      │  │
│  │ • plot_riccati_convergence()    - Solver conv        │  │
│  │ • plot_controllability_gramian()- Gramian heatmap    │  │
│  │ • plot_observability_gramian()  - Gramian heatmap    │  │
│  │ • plot_step_response()          - Step response      │  │
│  │ • plot_impulse_response()       - Impulse response   │  │
│  │ • plot_frequency_response()     - Bode plots         │  │
│  │ • plot_nyquist()                - Nyquist diagram    │  │
│  │ • plot_root_locus()             - Root locus         │  │
│  └──────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────┘
```

## Module Breakdown

### Theming Layer

#### themes.py
**File:** `themes.py` (623 lines)

**Purpose:** Centralized color palettes and plot styling

**Design Philosophy:**

- Single source of truth for all colors and styles
- Easy customization and extension
- Accessibility (colorblind-safe options)
- Publication-ready defaults

**Categories:**

**1. Color Schemes**

```python
class ColorSchemes:
    """Predefined color palettes for plotting."""
    
    # Categorical schemes (for distinct categories)
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
        # ... 10 colors total
    ]
    
    COLORBLIND_SAFE = [
        "#0173B2",  # Blue (Wong palette)
        "#DE8F05",  # Orange
        "#029E73",  # Green
        # ... 8 colors (colorblind accessible)
    ]
    
    TABLEAU = [
        "#4E79A7",  # Blue
        "#F28E2B",  # Orange
        # ... 10 colors
    ]
    
    # Sequential schemes (for heatmaps, continuous data)
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
    
    SEQUENTIAL_GREEN = [...]  # Similar structure
    
    # Diverging schemes (for signed data)
    DIVERGING_RED_BLUE = [
        "#67001f",  # Dark red
        "#b2182b",
        "#d6604d",
        "#f4a582",
        "#fddbc7",
        "#f7f7f7",  # White (center)
        "#d1e5f0",
        "#92c5de",
        "#4393c3",
        "#2166ac",
        "#053061",  # Dark blue
    ]
    
    DIVERGING_PURPLE_GREEN = [...]  # Similar structure
    
    @staticmethod
    def get_colors(scheme: str, n_colors: Optional[int] = None) -> List[str]:
        """
        Get colors from a named scheme.
        
        Args:
            scheme: Name of color scheme ('plotly', 'd3', 'colorblind_safe', etc.)
            n_colors: Number of colors to return (cycles if exceeds palette size)
        
        Returns:
            List of color hex codes
        
        Examples:
            >>> colors = ColorSchemes.get_colors('colorblind_safe', n_colors=3)
            >>> print(colors)  # ['#0173B2', '#DE8F05', '#029E73']
        """
```

**2. Plot Themes**

```python
class PlotThemes:
    """Complete theme configurations for plots."""
    
    DEFAULT = {
        'template': 'plotly',
        'font_family': 'Arial, sans-serif',
        'font_size': 12,
        'title_font_size': 16,
        'axis_line_color': '#444',
        'grid_color': '#EEE',
        'background_color': '#FFF',
        'plot_background_color': '#FFF',
    }
    
    PUBLICATION = {
        'template': 'simple_white',
        'font_family': 'Times New Roman, serif',
        'font_size': 14,
        'title_font_size': 18,
        'axis_line_color': '#000',
        'axis_line_width': 1.5,
        'grid_color': '#CCC',
        'grid_width': 0.5,
        'background_color': '#FFF',
        'plot_background_color': '#FFF',
        'showlegend': True,
        'legend_bgcolor': 'rgba(255,255,255,0.8)',
        'legend_borderwidth': 1,
    }
    
    DARK = {
        'template': 'plotly_dark',
        'font_family': 'Arial, sans-serif',
        'font_size': 12,
        'title_font_size': 16,
        'axis_line_color': '#AAA',
        'grid_color': '#444',
        'background_color': '#111',
        'plot_background_color': '#1E1E1E',
    }
    
    @staticmethod
    def apply_theme(fig: go.Figure, theme: str = 'default') -> go.Figure:
        """
        Apply complete theme to figure.
        
        Args:
            fig: Plotly figure to style
            theme: Theme name ('default', 'publication', 'dark')
        
        Returns:
            Styled figure
        
        Examples:
            >>> fig = go.Figure()
            >>> fig = PlotThemes.apply_theme(fig, theme='publication')
        """
```

**3. Color Utilities**

```python
def lighten_color(color: str, amount: float = 0.5) -> str:
    """
    Lighten a hex color.
    
    Args:
        color: Hex color code '#RRGGBB'
        amount: Amount to lighten (0-1), 0 = no change, 1 = white
    
    Returns:
        Lightened hex color
    
    Examples:
        >>> lighten_color('#1f77b4', 0.5)  # Light blue
        '#8FBBDA'
    """

def darken_color(color: str, amount: float = 0.5) -> str:
    """Darken a hex color."""

def interpolate_colors(color1: str, color2: str, n: int) -> List[str]:
    """
    Create gradient between two colors.
    
    Args:
        color1: Start color hex code
        color2: End color hex code
        n: Number of colors in gradient
    
    Returns:
        List of n hex colors
    
    Examples:
        >>> gradient = interpolate_colors('#1f77b4', '#ff7f0e', n=5)
    """
```

**Key Design Features:**

- **Accessibility** - Colorblind-safe palettes (Wong palette)
- **Variety** - Categorical, sequential, diverging schemes
- **Consistency** - All plotters use same theme system
- **Publication quality** - Professional defaults
- **Customization** - Easy to add new themes/schemes

---

### Plotter Layer

#### trajectory_plotter.py
**File:** `trajectory_plotter.py` (914 lines)

**Purpose:** Time-domain visualization (state and control vs time)

**Design Philosophy:**

- Backend agnostic (NumPy/PyTorch/JAX)
- Automatic batch handling (Monte Carlo simulations)
- Adaptive layouts (optimal subplot arrangement)
- Interactive tooltips

**Architecture:**

```python
class TrajectoryPlotter:
    """
    Time-domain trajectory visualization.
    
    Attributes:
        backend: Default computational backend
        default_theme: Default plot theme
    """
    
    def __init__(self, backend: Backend = 'numpy', theme: str = 'default'):
        self.backend = backend
        self.default_theme = theme
```

**Methods:**

**1. plot_trajectory()**

```python
def plot_trajectory(
    self,
    t: TimePoints,
    x: StateTrajectory,
    state_names: Optional[List[str]] = None,
    show_std: bool = True,
    color_scheme: str = 'plotly',
    theme: str = 'default',
    title: Optional[str] = None
) -> go.Figure:
    """
    Plot state variables vs time.
    
    Automatically handles:
    - Single trajectory: (T, nx)
    - Batched trajectories: (n_batch, T, nx) → shows mean ± std
    - Adaptive subplot layout
    
    Args:
        t: Time points (T,)
        x: State trajectory (T, nx) or (n_batch, T, nx)
        state_names: State variable names for axis labels
        show_std: Show std deviation bands for batched data
        color_scheme: Color scheme name
        theme: Plot theme name
        title: Plot title
    
    Returns:
        Interactive Plotly figure with subplots for each state
    
    Examples:
        >>> # Single trajectory
        >>> t = np.linspace(0, 10, 100)
        >>> x = np.random.randn(100, 2)
        >>> fig = plotter.plot_trajectory(t, x, state_names=['x₁', 'x₂'])
        >>> 
        >>> # Batched (Monte Carlo)
        >>> x_batch = np.random.randn(10, 100, 2)
        >>> fig = plotter.plot_trajectory(t, x_batch, show_std=True)
        >>> 
        >>> # Publication style
        >>> fig = plotter.plot_trajectory(
        ...     t, x,
        ...     theme='publication',
        ...     color_scheme='colorblind_safe'
        ... )
    """
```

**2. plot_state_and_control()**

```python
def plot_state_and_control(
    self,
    t: TimePoints,
    x: StateTrajectory,
    u: ControlSequence,
    state_names: Optional[List[str]] = None,
    control_names: Optional[List[str]] = None,
    color_scheme: str = 'plotly',
    theme: str = 'default'
) -> go.Figure:
    """
    Plot states and controls together.
    
    Creates subplot grid with states on top, controls on bottom.
    
    Args:
        t: Time points (T,)
        x: State trajectory (T, nx)
        u: Control sequence (T, nu)
        state_names: State variable names
        control_names: Control input names
        color_scheme: Color scheme name
        theme: Plot theme name
    
    Returns:
        Interactive Plotly figure with state + control subplots
    
    Examples:
        >>> result = system.simulate(x0, controller, t_span=(0, 10))
        >>> fig = plotter.plot_state_and_control(
        ...     result['t'],
        ...     result['x'],
        ...     result['u'],
        ...     state_names=['Position', 'Velocity'],
        ...     control_names=['Force']
        ... )
    """
```

**3. plot_comparison()**

```python
def plot_comparison(
    self,
    results: Dict[str, Dict],
    state_names: Optional[List[str]] = None,
    color_scheme: str = 'plotly',
    theme: str = 'default'
) -> go.Figure:
    """
    Compare multiple simulation runs.
    
    Args:
        results: Dict mapping run names to integration results
                 Each result must have 't' and 'x' keys
        state_names: State variable names
        color_scheme: Color scheme name
        theme: Plot theme name
    
    Returns:
        Interactive Plotly figure comparing runs
    
    Examples:
        >>> results = {
        ...     'LQR Q=10': system.simulate(...),
        ...     'LQR Q=100': system.simulate(...),
        ...     'Open-loop': system.simulate(...)
        ... }
        >>> fig = plotter.plot_comparison(results)
    """
```

**Key Features:**

- ✅ Backend conversion handled automatically
- ✅ Batch dimension detected and processed
- ✅ Adaptive subplot layouts (optimal rows×cols)
- ✅ Mean ± std bands for Monte Carlo
- ✅ Interactive tooltips with exact values
- ✅ Publication-ready styling

---

#### phase_portrait.py
**File:** `phase_portrait.py` (1,027 lines)

**Purpose:** State space visualization (phase portraits)

**Design Philosophy:**

- Visualize dynamics in state space (not time domain)
- Support 2D and 3D phase portraits
- Optional vector field overlays
- Equilibrium point highlighting

**Architecture:**

```python
class PhasePortraitPlotter:
    """
    Phase space visualization.
    
    Attributes:
        backend: Default computational backend
        default_theme: Default plot theme
    """
    
    def __init__(self, backend: Backend = 'numpy', theme: str = 'default'):
        self.backend = backend
        self.default_theme = theme
```

**Methods:**

**1. plot_2d()**

```python
def plot_2d(
    self,
    x: StateTrajectory,
    state_indices: Tuple[int, int] = (0, 1),
    state_names: Optional[Tuple[str, str]] = None,
    vector_field: Optional[Callable] = None,
    equilibria: Optional[List[np.ndarray]] = None,
    show_direction: bool = False,
    color_scheme: str = 'plotly',
    theme: str = 'default'
) -> go.Figure:
    """
    2D phase portrait.
    
    Plots trajectories in state space (x₁ vs x₂).
    
    Args:
        x: State trajectory (T, nx) or (n_batch, T, nx)
        state_indices: Which states to plot (default first two)
        state_names: Axis labels
        vector_field: Function (x1, x2) → (dx1, dx2) for arrows
        equilibria: List of equilibrium points to mark
        show_direction: Add arrows showing trajectory direction
        color_scheme: Color scheme name
        theme: Plot theme name
    
    Returns:
        Interactive 2D phase portrait
    
    Examples:
        >>> # Basic phase portrait
        >>> fig = plotter.plot_2d(
        ...     x=trajectory,  # (T, 2)
        ...     state_names=('θ', 'ω')
        ... )
        >>> 
        >>> # With vector field
        >>> def pendulum_field(theta, omega):
        ...     return omega, -np.sin(theta)
        >>> 
        >>> fig = plotter.plot_2d(
        ...     x=trajectory,
        ...     vector_field=pendulum_field,
        ...     equilibria=[np.array([0, 0]), np.array([np.pi, 0])],
        ...     show_direction=True
        ... )
    """
```

**2. plot_3d()**

```python
def plot_3d(
    self,
    x: StateTrajectory,
    state_indices: Tuple[int, int, int] = (0, 1, 2),
    state_names: Optional[Tuple[str, str, str]] = None,
    equilibria: Optional[List[np.ndarray]] = None,
    color_scheme: str = 'plotly',
    theme: str = 'default'
) -> go.Figure:
    """
    3D phase portrait.
    
    Interactive 3D visualization with rotation and zoom.
    
    Args:
        x: State trajectory (T, nx)
        state_indices: Which states to plot (3 indices)
        state_names: Axis labels
        equilibria: Equilibrium points to mark
        color_scheme: Color scheme name
        theme: Plot theme name
    
    Returns:
        Interactive 3D phase portrait
    
    Examples:
        >>> # Lorenz attractor
        >>> fig = plotter.plot_3d(
        ...     x=lorenz_trajectory,  # (T, 3)
        ...     state_names=('x', 'y', 'z'),
        ...     theme='dark'
        ... )
    """
```

**3. plot_limit_cycle()**

```python
def plot_limit_cycle(
    self,
    x: StateTrajectory,
    state_indices: Tuple[int, int] = (0, 1),
    highlight_period: Optional[float] = None,
    color_scheme: str = 'plotly',
    theme: str = 'default'
) -> go.Figure:
    """
    Highlight periodic orbits (limit cycles).
    
    Args:
        x: State trajectory (T, nx)
        state_indices: Which states to plot
        highlight_period: If known, highlight one period
        color_scheme: Color scheme name
        theme: Plot theme name
    
    Returns:
        Phase portrait with limit cycle highlighted
    
    Examples:
        >>> # Van der Pol oscillator
        >>> fig = plotter.plot_limit_cycle(
        ...     x=vdp_trajectory,
        ...     highlight_period=6.28,
        ...     theme='publication'
        ... )
    """
```

**Key Features:**

- ✅ 2D and 3D phase space visualization
- ✅ Vector field overlays (quiver plots)
- ✅ Equilibrium point markers (color-coded by stability)
- ✅ Direction arrows on trajectories
- ✅ Interactive 3D rotation/zoom
- ✅ Limit cycle detection and highlighting

---

#### control_plots.py
**File:** `control_plots.py` (1,858 lines)

**Purpose:** Control-specific visualizations

**Design Philosophy:**

- Specialized plots for control analysis
- Frequency domain analysis (Bode, Nyquist)
- Stability analysis (eigenvalue maps, root locus)
- Performance metrics (step response, gramians)

**Architecture:**

```python
class ControlPlotter:
    """
    Control system analysis visualization.
    
    Attributes:
        backend: Default computational backend
        default_theme: Default plot theme
    """
    
    def __init__(self, backend: Backend = 'numpy', theme: str = 'default'):
        self.backend = backend
        self.default_theme = theme
```

**Methods (10 total):**

**1. plot_eigenvalue_map()**

```python
def plot_eigenvalue_map(
    self,
    eigenvalues: np.ndarray,
    system_type: str = 'continuous',
    show_stability_region: bool = True,
    theme: str = 'default'
) -> go.Figure:
    """
    Eigenvalue map with stability regions.
    
    Continuous: Plots Re(λ) vs Im(λ) with left half-plane shaded
    Discrete: Plots eigenvalues on complex plane with unit circle
    
    Args:
        eigenvalues: Complex eigenvalues
        system_type: 'continuous' or 'discrete'
        show_stability_region: Shade stability region
        theme: Plot theme name
    
    Returns:
        Interactive eigenvalue map
    
    Examples:
        >>> result = system.control.design_lqr(A, B, Q, R)
        >>> fig = plotter.plot_eigenvalue_map(
        ...     result['closed_loop_eigenvalues'],
        ...     system_type='continuous'
        ... )
    """
```

**2. plot_gain_comparison()**

```python
def plot_gain_comparison(
    self,
    gains: Dict[str, np.ndarray],
    color_scheme: str = 'plotly',
    theme: str = 'default'
) -> go.Figure:
    """
    Compare feedback gains across designs.
    
    Args:
        gains: Dict mapping design names to gain matrices
        color_scheme: Color scheme name
        theme: Plot theme name
    
    Returns:
        Heatmap comparing gains
    
    Examples:
        >>> gains = {
        ...     'Q=10': design_lqr(10*Q, R)['gain'],
        ...     'Q=100': design_lqr(100*Q, R)['gain'],
        ... }
        >>> fig = plotter.plot_gain_comparison(gains)
    """
```

**3. plot_riccati_convergence()**

```python
def plot_riccati_convergence(
    self,
    P_history: List[np.ndarray],
    theme: str = 'default'
) -> go.Figure:
    """
    Riccati equation solver convergence.
    
    Args:
        P_history: List of P matrices during iteration
        theme: Plot theme name
    
    Returns:
        Convergence plot (Frobenius norm vs iteration)
    """
```

**4. plot_controllability_gramian()**

```python
def plot_controllability_gramian(
    self,
    W_c: np.ndarray,
    state_names: Optional[List[str]] = None,
    theme: str = 'default'
) -> go.Figure:
    """
    Controllability Gramian heatmap.
    
    Args:
        W_c: Controllability Gramian matrix (nx, nx)
        state_names: State variable names
        theme: Plot theme name
    
    Returns:
        Heatmap of Gramian
    """
```

**5. plot_observability_gramian()**

```python
def plot_observability_gramian(
    self,
    W_o: np.ndarray,
    state_names: Optional[List[str]] = None,
    theme: str = 'default'
) -> go.Figure:
    """
    Observability Gramian heatmap.
    
    Args:
        W_o: Observability Gramian matrix (nx, nx)
        state_names: State variable names
        theme: Plot theme name
    
    Returns:
        Heatmap of Gramian
    """
```

**6. plot_step_response()**

```python
def plot_step_response(
    self,
    t: TimePoints,
    y: OutputSequence,
    show_metrics: bool = True,
    theme: str = 'default'
) -> go.Figure:
    """
    Step response with performance metrics.
    
    Computes and displays:
    - Rise time (10% → 90%)
    - Settling time (within 2% of final value)
    - Overshoot percentage
    - Steady-state error
    
    Args:
        t: Time points
        y: Output response (T, ny)
        show_metrics: Annotate metrics on plot
        theme: Plot theme name
    
    Returns:
        Step response plot with metrics
    """
```

**7. plot_impulse_response()**

```python
def plot_impulse_response(
    self,
    t: TimePoints,
    y: OutputSequence,
    show_decay: bool = True,
    theme: str = 'default'
) -> go.Figure:
    """
    Impulse response with decay analysis.
    
    Args:
        t: Time points
        y: Output response (T, ny)
        show_decay: Show exponential decay envelope
        theme: Plot theme name
    
    Returns:
        Impulse response plot
    """
```

**8. plot_frequency_response()**

```python
def plot_frequency_response(
    self,
    omega: np.ndarray,
    magnitude_db: np.ndarray,
    phase_deg: np.ndarray,
    show_margins: bool = True,
    theme: str = 'default'
) -> go.Figure:
    """
    Bode plot (magnitude and phase).
    
    Args:
        omega: Frequency points (rad/s)
        magnitude_db: Magnitude in dB
        phase_deg: Phase in degrees
        show_margins: Show gain/phase margins
        theme: Plot theme name
    
    Returns:
        Two-subplot Bode plot
    
    Examples:
        >>> from scipy import signal
        >>> sys = signal.lti(A, B, C, D)
        >>> omega, mag, phase = signal.bode(sys)
        >>> fig = plotter.plot_frequency_response(omega, mag, phase)
    """
```

**9. plot_nyquist()**

```python
def plot_nyquist(
    self,
    real: np.ndarray,
    imag: np.ndarray,
    show_unit_circle: bool = True,
    theme: str = 'default'
) -> go.Figure:
    """
    Nyquist diagram for stability analysis.
    
    Args:
        real: Real part of frequency response
        imag: Imaginary part of frequency response
        show_unit_circle: Show unit circle and critical point (-1, 0)
        theme: Plot theme name
    
    Returns:
        Nyquist plot
    """
```

**10. plot_root_locus()**

```python
def plot_root_locus(
    self,
    pole_paths: np.ndarray,
    gains: np.ndarray,
    system_type: str = 'continuous',
    theme: str = 'default'
) -> go.Figure:
    """
    Root locus (pole migration vs gain).
    
    Args:
        pole_paths: Pole locations (n_gains, n_poles)
        gains: Feedback gain values
        system_type: 'continuous' or 'discrete'
        theme: Plot theme name
    
    Returns:
        Root locus plot with stability regions
    """
```

**Key Features:**

- ✅ 10 specialized control analysis plots
- ✅ Frequency domain (Bode, Nyquist)
- ✅ Time domain (step, impulse)
- ✅ Stability analysis (eigenvalues, root locus)
- ✅ Gramian visualizations
- ✅ Performance metrics
- ✅ Interactive annotations

---

## Integration with Systems

### System Properties

All system classes (ContinuousSystemBase, DiscreteSystemBase) provide plotter access:

```python
# In ContinuousSystemBase / DiscreteSystemBase

@property
def plotter(self) -> TrajectoryPlotter:
    """Access trajectory plotting utilities."""
    if not hasattr(self, '_trajectory_plotter'):
        from src.visualization.trajectory_plotter import TrajectoryPlotter
        self._trajectory_plotter = TrajectoryPlotter(
            backend=self._default_backend,
            theme='default'
        )
    return self._trajectory_plotter

@property
def phase_plotter(self) -> PhasePortraitPlotter:
    """Access phase portrait plotting utilities."""
    if not hasattr(self, '_phase_portrait_plotter'):
        from src.visualization.phase_portrait import PhasePortraitPlotter
        self._phase_portrait_plotter = PhasePortraitPlotter(
            backend=self._default_backend,
            theme='default'
        )
    return self._phase_portrait_plotter

@property
def control_plotter(self) -> ControlPlotter:
    """Access control plotting utilities."""
    if not hasattr(self, '_control_plotter'):
        from src.visualization.control_plots import ControlPlotter
        self._control_plotter = ControlPlotter(
            backend=self._default_backend,
            theme='default'
        )
    return self._control_plotter
```

### Usage Pattern

```python
# Create system
system = Pendulum()

# Simulate
result = system.integrate(x0, u=None, t_span=(0, 10))

# Plot trajectory (via system)
fig = system.plotter.plot_trajectory(
    result['t'],
    result['x'],
    state_names=['θ', 'ω'],
    theme='publication'
)
fig.show()

# Plot phase portrait (via system)
fig = system.phase_plotter.plot_2d(
    result['x'],
    state_names=('θ', 'ω'),
    show_direction=True
)
fig.show()

# Plot eigenvalues (via system)
A, B = system.linearize(x_eq, u_eq)
lqr = system.control.design_lqr(A, B, Q, R)
fig = system.control_plotter.plot_eigenvalue_map(
    lqr['closed_loop_eigenvalues'],
    system_type='continuous'
)
fig.show()
```

---

## Design Patterns

### Pattern 1: Centralized Theming

**Why centralized themes?**

```python
# ANTI-PATTERN: Hardcoded colors in each plotter
class BadPlotter:
    def plot(self):
        fig.add_trace(go.Scatter(line=dict(color='#1f77b4')))  # Hardcoded!

# GOOD PATTERN: Centralized theming
class GoodPlotter:
    def plot(self, color_scheme='plotly', theme='default'):
        colors = ColorSchemes.get_colors(color_scheme)
        fig.add_trace(go.Scatter(line=dict(color=colors[0])))
        fig = PlotThemes.apply_theme(fig, theme)
```

**Benefits:**

- ✅ Consistent colors across all plots
- ✅ Easy to switch themes globally
- ✅ Accessibility (colorblind-safe)
- ✅ Publication mode with one parameter

### Pattern 2: Backend Agnostic Plotting

```python
def plot_trajectory(self, t, x, **kwargs):
    """Works with NumPy, PyTorch, JAX."""
    
    # Convert to NumPy for Plotly
    t_np = self._to_numpy(t)
    x_np = self._to_numpy(x)
    
    # Plot with Plotly (requires NumPy)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t_np, y=x_np[:, 0]))
    
    return fig

def _to_numpy(self, arr):
    """Convert any backend to NumPy."""
    if hasattr(arr, 'cpu'):  # PyTorch
        return arr.cpu().numpy()
    elif hasattr(arr, '__array__'):  # JAX
        return np.array(arr)
    else:  # NumPy
        return arr
```

### Pattern 3: Adaptive Layouts

```python
def _determine_subplot_layout(self, n_plots: int) -> Tuple[int, int]:
    """
    Determine optimal subplot grid.
    
    Examples:
        1-2 plots: (n, 1) vertical
        3-4 plots: (2, 2) square
        5-6 plots: (2, 3)
        7-9 plots: (3, 3)
    """
    if n_plots <= 2:
        return n_plots, 1
    elif n_plots <= 4:
        return 2, 2
    else:
        cols = int(np.ceil(np.sqrt(n_plots)))
        rows = int(np.ceil(n_plots / cols))
        return rows, cols
```

### Pattern 4: Batch Dimension Detection

```python
def _process_batch(self, x: np.ndarray) -> Tuple[np.ndarray, bool]:
    """
    Detect and process batch dimension.
    
    Args:
        x: Could be (T, nx) or (n_batch, T, nx)
    
    Returns:
        (data, is_batched)
        - If batched: returns mean trajectory
        - If single: returns as-is
    """
    if x.ndim == 3:  # Batched
        return np.mean(x, axis=0), True
    elif x.ndim == 2:  # Single
        return x, False
    else:
        raise ValueError(f"Invalid shape: {x.shape}")
```

## Key Strengths

1. **Centralized Theming** - Single source of truth for colors/styles
2. **Backend Agnostic** - NumPy/PyTorch/JAX transparent
3. **Interactive** - Plotly-based with zoom, pan, hover
4. **Publication Ready** - Professional defaults and themes
5. **Accessible** - Colorblind-safe palettes
6. **Specialized Plotters** - Right tool for each visualization type
7. **System Integration** - Clean `system.plotter` APIs
8. **Adaptive Layouts** - Optimal subplot arrangements
9. **Batch Support** - Monte Carlo visualization automatic
10. **Comprehensive** - 16 plot types covering all needs

This visualization framework provides **publication-quality interactive plotting** for ControlDESymulation!
