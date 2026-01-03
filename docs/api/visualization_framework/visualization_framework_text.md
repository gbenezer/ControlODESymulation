# Visualization Framework Architecture (Text Diagram)

```
═══════════════════════════════════════════════════════════════════════
               VISUALIZATION FRAMEWORK ARCHITECTURE
═══════════════════════════════════════════════════════════════════════

                    ┌──────────────────────────┐
                    │  APPLICATION LAYER       │
                    │                          │
                    │  ContinuousSystemBase    │
                    │  DiscreteSystemBase      │
                    └──────────┬───────────────┘
                               │
                ┌──────────────┴───────────────┐
                │                              │
                ↓                              ↓
    ┌───────────────────────┐      ┌───────────────────────┐
    │  system.plotter       │      │  system.phase_plotter  │
    │  TrajectoryPlotter    │      │  PhasePortraitPlotter  │
    └───────────┬───────────┘      └───────────┬───────────┘
                │                              │
                │  ┌──────────────────────────────┐
                └──┤  system.control_plotter      │
                   │  ControlPlotter              │
                   └──────────┬───────────────────┘
                              │
                              │ use theming from
                              ↓
            ┌──────────────────────────────────────┐
            │  THEMING LAYER                       │
            │  themes.py (623 lines)               │
            │                                      │
            │  ┌────────────────────────────────┐ │
            │  │ ColorSchemes                   │ │
            │  ├────────────────────────────────┤ │
            │  │ • PLOTLY (10 colors)           │ │
            │  │ • D3 (10 colors)               │ │
            │  │ • COLORBLIND_SAFE (8 colors)   │ │
            │  │ • TABLEAU (10 colors)          │ │
            │  │ • SEQUENTIAL_BLUE (9 colors)   │ │
            │  │ • SEQUENTIAL_GREEN (9 colors)  │ │
            │  │ • DIVERGING_RED_BLUE (11)      │ │
            │  │ • DIVERGING_PURPLE_GREEN (11)  │ │
            │  └────────────────────────────────┘ │
            │                                      │
            │  ┌────────────────────────────────┐ │
            │  │ PlotThemes                     │ │
            │  ├────────────────────────────────┤ │
            │  │ • DEFAULT (white, clean)       │ │
            │  │ • PUBLICATION (serif, formal)  │ │
            │  │ • DARK (dark background)       │ │
            │  └────────────────────────────────┘ │
            │                                      │
            │  ┌────────────────────────────────┐ │
            │  │ Color Utilities                │ │
            │  ├────────────────────────────────┤ │
            │  │ • lighten_color()              │ │
            │  │ • darken_color()               │ │
            │  │ • interpolate_colors()         │ │
            │  └────────────────────────────────┘ │
            └──────────────┬───────────────────────┘
                           │ used by
                           ↓
            ┌──────────────────────────────────────┐
            │  PLOTTER LAYER                       │
            │                                      │
            │  ┌────────────────────────────────┐ │
            │  │ TrajectoryPlotter (914 lines)  │ │
            │  ├────────────────────────────────┤ │
            │  │ Time-Domain Visualization      │ │
            │  ├────────────────────────────────┤ │
            │  │ • plot_trajectory()            │ │
            │  │   - Single: (T, nx)            │ │
            │  │   - Batch: (n_batch, T, nx)    │ │
            │  │   - Mean ± std bands           │ │
            │  │                                │ │
            │  │ • plot_state_and_control()     │ │
            │  │   - Combined view              │ │
            │  │   - Adaptive layout            │ │
            │  │                                │ │
            │  │ • plot_comparison()            │ │
            │  │   - Multiple runs              │ │
            │  │   - Overlay plots              │ │
            │  └────────────────────────────────┘ │
            │                                      │
            │  ┌────────────────────────────────┐ │
            │  │ PhasePortraitPlotter (1027)    │ │
            │  ├────────────────────────────────┤ │
            │  │ State Space Visualization      │ │
            │  ├────────────────────────────────┤ │
            │  │ • plot_2d()                    │ │
            │  │   - x₁ vs x₂                   │ │
            │  │   - Vector fields              │ │
            │  │   - Equilibria                 │ │
            │  │   - Direction arrows           │ │
            │  │                                │ │
            │  │ • plot_3d()                    │ │
            │  │   - Interactive 3D             │ │
            │  │   - Rotate/zoom                │ │
            │  │                                │ │
            │  │ • plot_limit_cycle()           │ │
            │  │   - Periodic orbits            │ │
            │  │   - Period highlighting        │ │
            │  └────────────────────────────────┘ │
            │                                      │
            │  ┌────────────────────────────────┐ │
            │  │ ControlPlotter (1858 lines)    │ │
            │  ├────────────────────────────────┤ │
            │  │ Control-Specific Plots         │ │
            │  ├────────────────────────────────┤ │
            │  │ Stability Analysis:            │ │
            │  │ • plot_eigenvalue_map()        │ │
            │  │ • plot_root_locus()            │ │
            │  │                                │ │
            │  │ Design Comparison:             │ │
            │  │ • plot_gain_comparison()       │ │
            │  │ • plot_riccati_convergence()   │ │
            │  │                                │ │
            │  │ Gramian Analysis:              │ │
            │  │ • plot_controllability_gramian │ │
            │  │ • plot_observability_gramian   │ │
            │  │                                │ │
            │  │ Time Responses:                │ │
            │  │ • plot_step_response()         │ │
            │  │ • plot_impulse_response()      │ │
            │  │                                │ │
            │  │ Frequency Analysis:            │ │
            │  │ • plot_frequency_response()    │ │
            │  │ • plot_nyquist()               │ │
            │  └────────────────────────────────┘ │
            └──────────────────────────────────────┘


═══════════════════════════════════════════════════════════════════════
                        MODULE BREAKDOWN
═══════════════════════════════════════════════════════════════════════

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ THEMING LAYER: themes.py (623 lines)                            ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

ColorSchemes Class
├─ Categorical Schemes (for distinct categories)
│  ├─ PLOTLY: ['#636EFA', '#EF553B', '#00CC96', ...]  (10 colors)
│  ├─ D3: ['#1f77b4', '#ff7f0e', '#2ca02c', ...]      (10 colors)
│  ├─ COLORBLIND_SAFE: ['#0173B2', '#DE8F05', ...]   (8 colors)
│  └─ TABLEAU: ['#4E79A7', '#F28E2B', '#E15759', ...] (10 colors)
│
├─ Sequential Schemes (for heatmaps, continuous data)
│  ├─ SEQUENTIAL_BLUE: ['#f7fbff', ..., '#08306b']   (9 colors)
│  └─ SEQUENTIAL_GREEN: ['#f7fcf5', ..., '#00441b']  (9 colors)
│
├─ Diverging Schemes (for signed data, ±)
│  ├─ DIVERGING_RED_BLUE: ['#67001f', ..., '#053061'] (11 colors)
│  └─ DIVERGING_PURPLE_GREEN: ['#40004b', ..., '#00441b'] (11)
│
└─ Methods
   └─> get_colors(scheme, n_colors) → List[str]

PlotThemes Class
├─ DEFAULT
│  ├─ template: 'plotly'
│  ├─ font: Arial, sans-serif, 12pt
│  ├─ background: White
│  └─ grid: #EEE
│
├─ PUBLICATION
│  ├─ template: 'simple_white'
│  ├─ font: Times New Roman, serif, 14pt
│  ├─ background: White
│  ├─ grid: #CCC
│  └─ borders: Black, thicker lines
│
├─ DARK
│  ├─ template: 'plotly_dark'
│  ├─ font: Arial, sans-serif, 12pt
│  ├─ background: #111
│  └─ grid: #444
│
└─ Methods
   └─> apply_theme(fig, theme) → go.Figure

Color Utilities
├─> lighten_color(color, amount) → str
├─> darken_color(color, amount) → str
└─> interpolate_colors(color1, color2, n) → List[str]


┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ PLOTTER LAYER: TrajectoryPlotter (914 lines)                    ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

TrajectoryPlotter.__init__(backend, theme)
├─> self.backend = backend
└─> self.default_theme = theme

plot_trajectory(t, x, state_names, show_std, color_scheme, theme)
├─> Detects batch dimension
│   ├─ Single: (T, nx) → plot directly
│   └─ Batch: (n_batch, T, nx) → compute mean, std
├─> Determines subplot layout
│   └─> _determine_subplot_layout(nx) → (rows, cols)
├─> Creates subplots
│   └─> make_subplots(rows, cols)
├─> Adds traces for each state
│   ├─ Mean line
│   └─ Std bands (if batched and show_std=True)
├─> Applies theme
│   └─> PlotThemes.apply_theme(fig, theme)
└─> Returns: go.Figure

plot_state_and_control(t, x, u, state_names, control_names, ...)
├─> Creates combined subplot grid
│   ├─ Top rows: States
│   └─ Bottom rows: Controls
├─> Adds state traces
├─> Adds control traces
└─> Returns: go.Figure

plot_comparison(results, state_names, color_scheme, theme)
├─> Iterates over results dict
├─> Adds trace for each result
│   └─> Different color per result
├─> Creates legend
└─> Returns: go.Figure

Internal Methods
├─> _to_numpy(arr) → np.ndarray
│   ├─ PyTorch: arr.cpu().numpy()
│   ├─ JAX: np.array(arr)
│   └─ NumPy: arr
│
└─> _determine_subplot_layout(n_plots) → (rows, cols)
    ├─ 1-2 plots: (n, 1)
    ├─ 3-4 plots: (2, 2)
    └─ 5+ plots: optimal grid


┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ PLOTTER LAYER: PhasePortraitPlotter (1027 lines)                ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

PhasePortraitPlotter.__init__(backend, theme)
├─> self.backend = backend
└─> self.default_theme = theme

plot_2d(x, state_indices, state_names, vector_field, equilibria, ...)
├─> Extracts relevant states
│   └─> x[:, state_indices]
├─> Creates main trajectory trace
│   └─> go.Scatter(x=x₁, y=x₂, mode='lines')
├─> Optional: Add vector field
│   ├─> Creates meshgrid
│   ├─> Evaluates dynamics at grid points
│   └─> Adds quiver plot (arrows)
├─> Optional: Add equilibria
│   └─> go.Scatter(x, y, mode='markers', marker_size=12)
├─> Optional: Add direction arrows
│   └─> Annotations with arrows along trajectory
└─> Returns: go.Figure

plot_3d(x, state_indices, state_names, equilibria, ...)
├─> Creates 3D scatter trace
│   └─> go.Scatter3d(x=x₁, y=x₂, z=x₃, mode='lines')
├─> Optional: Add equilibria
│   └─> go.Scatter3d (markers)
├─> Sets 3D layout
│   ├─> scene.xaxis.title
│   ├─> scene.yaxis.title
│   └─> scene.zaxis.title
└─> Returns: go.Figure

plot_limit_cycle(x, state_indices, highlight_period, ...)
├─> Plots trajectory
├─> Optional: Highlights one period
│   └─> Different color/style for [0, T_period]
├─> Adds limit cycle annotation
└─> Returns: go.Figure


┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ PLOTTER LAYER: ControlPlotter (1858 lines)                      ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

ControlPlotter.__init__(backend, theme)
├─> self.backend = backend
└─> self.default_theme = theme

Stability Analysis Plots
├─> plot_eigenvalue_map(eigenvalues, system_type, ...)
│   ├─> Continuous: Plots Re(λ) vs Im(λ)
│   │   ├─> Shades left half-plane (stable region)
│   │   └─> Adds imaginary axis line
│   └─> Discrete: Plots complex plane
│       ├─> Draws unit circle
│       └─> Shades interior (stable region)
│
└─> plot_root_locus(pole_paths, gains, system_type, ...)
    ├─> Plots pole trajectories vs gain
    ├─> Shows stability boundary
    └─> Annotates gain values

Design Comparison Plots
├─> plot_gain_comparison(gains, ...)
│   ├─> Creates heatmap
│   │   └─> go.Heatmap(z=gain_matrix)
│   └─> Annotates with values
│
└─> plot_riccati_convergence(P_history, ...)
    ├─> Computes Frobenius norm for each P
    ├─> Plots norm vs iteration
    └─> Shows convergence threshold

Gramian Analysis Plots
├─> plot_controllability_gramian(W_c, state_names, ...)
│   ├─> Creates heatmap of W_c
│   └─> Annotates diagonal (controllability)
│
└─> plot_observability_gramian(W_o, state_names, ...)
    ├─> Creates heatmap of W_o
    └─> Annotates diagonal (observability)

Time Response Plots
├─> plot_step_response(t, y, show_metrics, ...)
│   ├─> Plots step response
│   ├─> Computes metrics:
│   │   ├─ Rise time (10% → 90%)
│   │   ├─ Settling time (±2% band)
│   │   ├─ Overshoot (%)
│   │   └─ Steady-state error
│   └─> Annotates metrics on plot
│
└─> plot_impulse_response(t, y, show_decay, ...)
    ├─> Plots impulse response
    ├─> Optional: Exponential decay envelope
    └─> Annotates decay rate

Frequency Analysis Plots
├─> plot_frequency_response(omega, mag_db, phase_deg, ...)
│   ├─> Creates 2-subplot figure
│   │   ├─ Top: Magnitude (dB) vs ω
│   │   └─ Bottom: Phase (deg) vs ω
│   ├─> Semilog x-axis
│   └─> Optional: Gain/phase margins
│
└─> plot_nyquist(real, imag, show_unit_circle, ...)
    ├─> Plots Nyquist diagram
    ├─> Shows critical point (-1, 0)
    ├─> Optional: Unit circle
    └─> Annotates encirclements


═══════════════════════════════════════════════════════════════════════
                        INTEGRATION WITH SYSTEMS
═══════════════════════════════════════════════════════════════════════

ContinuousSystemBase / DiscreteSystemBase
│
├─> @property plotter(self) → TrajectoryPlotter
│   └─> if not hasattr(self, '_trajectory_plotter'):
│       └─> self._trajectory_plotter = TrajectoryPlotter(
│               backend=self._default_backend,
│               theme='default'
│           )
│   └─> return self._trajectory_plotter
│
├─> @property phase_plotter(self) → PhasePortraitPlotter
│   └─> if not hasattr(self, '_phase_portrait_plotter'):
│       └─> self._phase_portrait_plotter = PhasePortraitPlotter(
│               backend=self._default_backend,
│               theme='default'
│           )
│   └─> return self._phase_portrait_plotter
│
└─> @property control_plotter(self) → ControlPlotter
    └─> if not hasattr(self, '_control_plotter'):
        └─> self._control_plotter = ControlPlotter(
                backend=self._default_backend,
                theme='default'
            )
    └─> return self._control_plotter

Usage Example:
    system = Pendulum()
    system.set_default_backend('numpy')
    
    # Via system properties
    result = system.integrate(x0, u=None, t_span=(0, 10))
    
    fig = system.plotter.plot_trajectory(result['t'], result['x'])
    # ↑ Uses TrajectoryPlotter with 'numpy' backend
    
    fig = system.phase_plotter.plot_2d(result['x'])
    # ↑ Uses PhasePortraitPlotter with 'numpy' backend


═══════════════════════════════════════════════════════════════════════
                        PLOTTING WORKFLOW
═══════════════════════════════════════════════════════════════════════

Typical Workflow:
┌─────────────────────────────────────────────────────────────────┐
│ 1. User calls: system.plotter.plot_trajectory(t, x)            │
│    ↓                                                             │
│ 2. TrajectoryPlotter receives backend='numpy' from system      │
│    ↓                                                             │
│ 3. Plotter detects data shape                                  │
│    ├─ (T, nx): Single trajectory                               │
│    └─ (n_batch, T, nx): Batched → compute mean, std            │
│    ↓                                                             │
│ 4. Convert backend to NumPy (Plotly requirement)               │
│    ├─ PyTorch: arr.cpu().numpy()                               │
│    ├─ JAX: np.array(arr)                                       │
│    └─ NumPy: as-is                                             │
│    ↓                                                             │
│ 5. Determine optimal subplot layout                            │
│    └─> _determine_subplot_layout(nx) → (rows, cols)            │
│    ↓                                                             │
│ 6. Create Plotly figure with subplots                          │
│    └─> make_subplots(rows, cols)                               │
│    ↓                                                             │
│ 7. Get colors from ColorSchemes                                │
│    └─> ColorSchemes.get_colors(color_scheme, nx)               │
│    ↓                                                             │
│ 8. Add traces for each state                                   │
│    ├─ Mean line (or single trajectory)                         │
│    └─ Std bands (if batched)                                   │
│    ↓                                                             │
│ 9. Apply theme                                                  │
│    └─> PlotThemes.apply_theme(fig, theme)                      │
│    ↓                                                             │
│ 10. Return interactive Plotly figure                           │
└─────────────────────────────────────────────────────────────────┘

Phase Portrait Workflow:
┌─────────────────────────────────────────────────────────────────┐
│ 1. User calls: system.phase_plotter.plot_2d(x)                 │
│    ↓                                                             │
│ 2. Extract relevant states: x[:, state_indices]                │
│    ↓                                                             │
│ 3. Create main trajectory trace                                │
│    └─> go.Scatter(x=x₁, y=x₂, mode='lines')                    │
│    ↓                                                             │
│ 4. Optional: Add vector field                                  │
│    ├─> Create meshgrid of state space                          │
│    ├─> Evaluate dynamics at each point                         │
│    └─> Add quiver plot (arrows)                                │
│    ↓                                                             │
│ 5. Optional: Add equilibrium points                            │
│    └─> go.Scatter with markers                                 │
│    ↓                                                             │
│ 6. Optional: Add direction arrows                              │
│    └─> Annotations along trajectory                            │
│    ↓                                                             │
│ 7. Apply theme and return                                      │
└─────────────────────────────────────────────────────────────────┘


═══════════════════════════════════════════════════════════════════════
                        FILE SIZE SUMMARY
═══════════════════════════════════════════════════════════════════════

┌──────────────────────────────┬────────┬──────────────────────────┐
│ Module                       │ Lines  │ Purpose                  │
├──────────────────────────────┼────────┼──────────────────────────┤
│ THEMING LAYER                │        │                          │
├──────────────────────────────┼────────┼──────────────────────────┤
│ themes.py                    │   623  │ Colors and styles        │
├──────────────────────────────┼────────┼──────────────────────────┤
│ PLOTTER LAYER                │        │                          │
├──────────────────────────────┼────────┼──────────────────────────┤
│ trajectory_plotter.py        │   914  │ Time-domain plots        │
│ phase_portrait.py            │ 1,027  │ State space plots        │
│ control_plots.py             │ 1,858  │ Control analysis plots   │
├──────────────────────────────┼────────┼──────────────────────────┤
│ TOTAL                        │ 4,422  │ Complete framework       │
└──────────────────────────────┴────────┴──────────────────────────┘


═══════════════════════════════════════════════════════════════════════
                        DESIGN PHILOSOPHY
═══════════════════════════════════════════════════════════════════════

✓ CENTRALIZED THEMING
  Single source of truth for colors and styles
  Easy to maintain consistency

✓ BACKEND AGNOSTIC
  NumPy/PyTorch/JAX handled transparently
  Automatic conversion for Plotly

✓ INTERACTIVE
  Plotly-based plots with zoom, pan, hover
  Rich user experience

✓ PUBLICATION READY
  Professional themes and color schemes
  Export to PDF, PNG, HTML

✓ ACCESSIBLE
  Colorblind-safe palettes (Wong palette)
  WCAG compliance

✓ SPECIALIZED PLOTTERS
  Dedicated class for each visualization type
  Right tool for the job

✓ SYSTEM INTEGRATION
  Clean system.plotter properties
  Automatic backend inheritance

✓ ADAPTIVE LAYOUTS
  Optimal subplot arrangements
  Smart defaults

✓ BATCH SUPPORT
  Monte Carlo visualization automatic
  Mean ± std bands

✓ COMPREHENSIVE
  16 plot types covering all needs
  Time, frequency, state space


═══════════════════════════════════════════════════════════════════════
                        PLOT TYPE SUMMARY
═══════════════════════════════════════════════════════════════════════

Time-Domain Plots (TrajectoryPlotter)
  ├─ plot_trajectory()           → State vs time
  ├─ plot_state_and_control()    → Combined state + control
  └─ plot_comparison()           → Multiple simulation runs

State-Space Plots (PhasePortraitPlotter)
  ├─ plot_2d()                   → 2D phase portrait
  ├─ plot_3d()                   → 3D phase portrait
  └─ plot_limit_cycle()          → Periodic orbits

Control Analysis (ControlPlotter)
  ├─ Stability:
  │  ├─ plot_eigenvalue_map()    → Pole locations
  │  └─ plot_root_locus()        → Pole migration vs gain
  ├─ Design:
  │  ├─ plot_gain_comparison()   → Compare feedback gains
  │  └─ plot_riccati_convergence() → Solver convergence
  ├─ Gramians:
  │  ├─ plot_controllability_gramian() → W_c heatmap
  │  └─ plot_observability_gramian()   → W_o heatmap
  ├─ Time Response:
  │  ├─ plot_step_response()     → Step with metrics
  │  └─ plot_impulse_response()  → Impulse with decay
  └─ Frequency:
     ├─ plot_frequency_response() → Bode plots
     └─ plot_nyquist()           → Nyquist diagram

Total: 16 plot types


═══════════════════════════════════════════════════════════════════════
                        COLOR SCHEME REFERENCE
═══════════════════════════════════════════════════════════════════════

Categorical Schemes (for distinct items):
  PLOTLY           → 10 colors, default Plotly palette
  D3               → 10 colors, D3.js standard
  COLORBLIND_SAFE  → 8 colors, Wong palette (accessible)
  TABLEAU          → 10 colors, Tableau default

Sequential Schemes (for heatmaps):
  SEQUENTIAL_BLUE  → 9 colors, light → dark blue
  SEQUENTIAL_GREEN → 9 colors, light → dark green

Diverging Schemes (for signed data):
  DIVERGING_RED_BLUE      → 11 colors, red ← white → blue
  DIVERGING_PURPLE_GREEN  → 11 colors, purple ← white → green


═══════════════════════════════════════════════════════════════════════
                        THEME REFERENCE
═══════════════════════════════════════════════════════════════════════

DEFAULT Theme:
  Background:   White (#FFF)
  Font:         Arial, sans-serif, 12pt
  Grid:         Light gray (#EEE)
  Template:     'plotly'
  Use case:     Interactive exploration, web displays

PUBLICATION Theme:
  Background:   White (#FFF)
  Font:         Times New Roman, serif, 14pt
  Grid:         Medium gray (#CCC)
  Template:     'simple_white'
  Lines:        Thicker, more contrast
  Use case:     Papers, formal presentations

DARK Theme:
  Background:   Dark gray (#111)
  Plot area:    Slightly lighter (#1E1E1E)
  Font:         Arial, sans-serif, 12pt
  Grid:         Dark lines (#444)
  Template:     'plotly_dark'
  Use case:     Dark mode, presentations on dark backgrounds


═══════════════════════════════════════════════════════════════════════
```

## Summary

**Total Lines:** 4,422 (framework)

**Architecture:** Centralized theming + specialized plotters

**Plot Types:** 16 total (3 time-domain, 3 state-space, 10 control)

**Color Schemes:** 8 predefined (categorical, sequential, diverging)

**Themes:** 3 complete themes (default, publication, dark)

**Integration:** Via `system.plotter`, `system.phase_plotter`, `system.control_plotter`

**Philosophy:** Interactive, accessible, publication-ready visualization

**Result:** Comprehensive plotting framework for dynamical systems analysis!
