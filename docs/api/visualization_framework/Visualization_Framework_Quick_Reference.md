# Visualization Framework Quick Reference

## Quick Start

### 1. Time-Domain Trajectory Plot

```python
from src.systems.examples import Pendulum
import numpy as np

# Create system
system = Pendulum()

# Simulate
result = system.integrate(
    x0=np.array([0.5, 0]),
    u=None,
    t_span=(0, 10)
)

# Plot trajectory
fig = system.plotter.plot_trajectory(
    result['t'],
    result['x'],
    state_names=['θ', 'ω'],
    theme='publication'
)
fig.show()

# Save to HTML
fig.write_html('trajectory.html')
```

### 2. Phase Portrait (2D)

```python
# 2D phase portrait
fig = system.phase_plotter.plot_2d(
    x=result['x'],
    state_names=('θ', 'ω'),
    show_direction=True,
    theme='publication'
)
fig.show()

# With vector field
def pendulum_field(theta, omega):
    return omega, -np.sin(theta) - 0.1*omega

fig = system.phase_plotter.plot_2d(
    x=result['x'],
    vector_field=pendulum_field,
    equilibria=[np.array([0, 0]), np.array([np.pi, 0])],
    theme='publication'
)
```

### 3. Control Analysis (Eigenvalue Map)

```python
# Linearize and design LQR
x_eq = np.array([np.pi, 0])
u_eq = np.zeros(1)
A, B = system.linearize(x_eq, u_eq)

Q = np.diag([10, 1])
R = np.array([[0.1]])
lqr = system.control.design_lqr(A, B, Q, R, system_type='continuous')

# Plot eigenvalues
fig = system.control_plotter.plot_eigenvalue_map(
    lqr['closed_loop_eigenvalues'],
    system_type='continuous',
    theme='publication'
)
fig.show()
```

### 4. State and Control Together

```python
# Simulate with controller
def lqr_controller(x, t):
    return -lqr['gain'] @ (x - x_eq)

result = system.simulate(
    x0=np.array([np.pi + 0.2, 0]),
    controller=lqr_controller,
    t_span=(0, 10),
    dt=0.01
)

# Plot state and control
fig = system.plotter.plot_state_and_control(
    result['t'],
    result['x'],
    result['u'],
    state_names=['θ', 'ω'],
    control_names=['Torque'],
    theme='publication'
)
```

---

## Trajectory Plotting

### Basic Trajectory Plot

```python
from src.visualization.trajectory_plotter import TrajectoryPlotter

plotter = TrajectoryPlotter(backend='numpy')

# Single trajectory
t = np.linspace(0, 10, 100)
x = np.random.randn(100, 2)

fig = plotter.plot_trajectory(
    t, x,
    state_names=['x₁', 'x₂'],
    theme='default'
)
fig.show()
```

### Batched Trajectories (Monte Carlo)

```python
# Multiple trajectories
n_batch = 20
x_batch = np.random.randn(n_batch, 100, 2)

# Automatically shows mean ± std
fig = plotter.plot_trajectory(
    t, x_batch,
    state_names=['x₁', 'x₂'],
    show_std=True,  # Show std deviation bands
    color_scheme='colorblind_safe',
    theme='publication'
)
```

### State and Control Combined

```python
# State + control subplots
fig = plotter.plot_state_and_control(
    t=result['t'],
    x=result['x'],  # (T, nx)
    u=result['u'],  # (T, nu)
    state_names=['Position', 'Velocity'],
    control_names=['Force'],
    color_scheme='plotly',
    theme='default'
)
```

### Comparing Multiple Runs

```python
# Compare different controllers
results = {
    'LQR Q=10': system.simulate(...),
    'LQR Q=100': system.simulate(...),
    'Open-loop': system.simulate(...)
}

fig = plotter.plot_comparison(
    results,
    state_names=['θ', 'ω'],
    color_scheme='colorblind_safe',
    theme='publication'
)
```

---

## Phase Portrait Plotting

### 2D Phase Portrait

```python
from src.visualization.phase_portrait import PhasePortraitPlotter

plotter = PhasePortraitPlotter(backend='numpy')

# Basic 2D phase portrait
fig = plotter.plot_2d(
    x=trajectory,  # (T, 2) or (T, nx) with state_indices
    state_names=('x₁', 'x₂'),
    theme='default'
)
fig.show()
```

### With Direction Arrows

```python
# Show trajectory direction
fig = plotter.plot_2d(
    x=trajectory,
    state_names=('θ', 'ω'),
    show_direction=True,  # Adds arrows
    theme='publication'
)
```

### With Vector Field

```python
# Add vector field (quiver plot)
def dynamics(x1, x2):
    """Return (dx1/dt, dx2/dt)"""
    return x2, -x1 - 0.1*x2

fig = plotter.plot_2d(
    x=trajectory,
    vector_field=dynamics,
    state_names=('Position', 'Velocity'),
    theme='publication'
)
```

### With Equilibria

```python
# Mark equilibrium points
equilibria = [
    np.array([0, 0]),      # Stable equilibrium
    np.array([np.pi, 0])   # Unstable equilibrium
]

fig = plotter.plot_2d(
    x=trajectory,
    equilibria=equilibria,
    vector_field=dynamics,
    show_direction=True,
    theme='publication'
)
```

### 3D Phase Portrait

```python
# 3D phase space
fig = plotter.plot_3d(
    x=trajectory_3d,  # (T, 3)
    state_names=('x', 'y', 'z'),
    theme='dark'
)
fig.show()

# Lorenz attractor example
fig = plotter.plot_3d(
    x=lorenz_trajectory,
    state_names=('x', 'y', 'z'),
    equilibria=[np.zeros(3)],
    theme='publication'
)
```

### Limit Cycle Visualization

```python
# Highlight periodic orbits
fig = plotter.plot_limit_cycle(
    x=vdp_trajectory,
    state_indices=(0, 1),
    highlight_period=6.28,  # If known
    theme='publication'
)
```

---

## Control Analysis Plots

### Eigenvalue Map

```python
from src.visualization.control_plots import ControlPlotter

plotter = ControlPlotter(backend='numpy')

# Continuous system
fig = plotter.plot_eigenvalue_map(
    eigenvalues=lqr['closed_loop_eigenvalues'],
    system_type='continuous',  # Shows left half-plane
    show_stability_region=True,
    theme='publication'
)

# Discrete system
fig = plotter.plot_eigenvalue_map(
    eigenvalues=discrete_poles,
    system_type='discrete',  # Shows unit circle
    theme='publication'
)
```

### Gain Comparison

```python
# Compare LQR gains for different Q weights
gains = {}
for q_scale in [1, 10, 100, 1000]:
    Q_scaled = q_scale * Q
    result = system.control.design_lqr(A, B, Q_scaled, R)
    gains[f'Q={q_scale}'] = result['gain']

fig = plotter.plot_gain_comparison(
    gains,
    color_scheme='colorblind_safe',
    theme='publication'
)
```

### Riccati Convergence

```python
# Monitor iterative Riccati solver
P_history = []  # Collect P at each iteration

fig = plotter.plot_riccati_convergence(
    P_history,
    theme='publication'
)
```

### Controllability Gramian

```python
# Compute controllability Gramian
from scipy.linalg import solve_continuous_lyapunov

W_c = solve_continuous_lyapunov(A, -B @ B.T)

fig = plotter.plot_controllability_gramian(
    W_c,
    state_names=['x₁', 'x₂'],
    theme='publication'
)
```

### Observability Gramian

```python
# Compute observability Gramian
W_o = solve_continuous_lyapunov(A.T, -C.T @ C)

fig = plotter.plot_observability_gramian(
    W_o,
    state_names=['x₁', 'x₂'],
    theme='publication'
)
```

### Step Response

```python
from scipy import signal

# Closed-loop system
A_cl = A - B @ K
C = np.array([[1, 0]])  # Measure first state
D = np.zeros((1, 1))

sys = signal.StateSpace(A_cl, B, C, D)
t, y = signal.step(sys)

fig = plotter.plot_step_response(
    t, y,
    show_metrics=True,  # Annotate rise time, overshoot, etc.
    theme='publication'
)
```

### Impulse Response

```python
t, y = signal.impulse(sys)

fig = plotter.plot_impulse_response(
    t, y,
    show_decay=True,  # Show exponential decay envelope
    theme='publication'
)
```

### Bode Plot

```python
omega, mag, phase = signal.bode(sys)

fig = plotter.plot_frequency_response(
    omega, mag, phase,
    show_margins=True,  # Show gain/phase margins
    theme='publication'
)
```

### Nyquist Diagram

```python
# Frequency response
omega = np.logspace(-2, 2, 1000)
s = 1j * omega
H = np.array([C @ np.linalg.solve(s[i]*np.eye(2) - A, B) for i in range(len(s))])

fig = plotter.plot_nyquist(
    real=np.real(H.flatten()),
    imag=np.imag(H.flatten()),
    show_unit_circle=True,
    theme='publication'
)
```

### Root Locus

```python
# Compute pole paths for varying gain
gains = np.linspace(0, 10, 100)
pole_paths = np.zeros((len(gains), 2), dtype=complex)

for i, k in enumerate(gains):
    pole_paths[i, :] = np.linalg.eigvals(A - k * B @ K)

fig = plotter.plot_root_locus(
    pole_paths,
    gains,
    system_type='continuous',
    theme='publication'
)
```

---

## Theming and Customization

### Available Themes

```python
from src.visualization.themes import PlotThemes

# Default theme (white background)
fig = plotter.plot_trajectory(t, x, theme='default')

# Publication theme (Times New Roman, high contrast)
fig = plotter.plot_trajectory(t, x, theme='publication')

# Dark theme (dark background)
fig = plotter.plot_trajectory(t, x, theme='dark')

# Apply theme to existing figure
fig = PlotThemes.apply_theme(fig, theme='publication')
```

### Color Schemes

```python
from src.visualization.themes import ColorSchemes

# Get colors from predefined scheme
colors = ColorSchemes.get_colors('plotly', n_colors=5)
colors = ColorSchemes.get_colors('d3', n_colors=5)
colors = ColorSchemes.get_colors('colorblind_safe', n_colors=5)
colors = ColorSchemes.get_colors('tableau', n_colors=5)

# Use in plots
fig = plotter.plot_trajectory(
    t, x,
    color_scheme='colorblind_safe'
)
```

### Available Color Schemes

**Categorical (for distinct categories):**
- `'plotly'` - Default Plotly colors (10 colors)
- `'d3'` - D3.js Category10 (10 colors)
- `'colorblind_safe'` - Wong palette (8 colors, accessible)
- `'tableau'` - Tableau 10 palette (10 colors)

**Sequential (for heatmaps):**
- `'sequential_blue'` - Blue gradient (9 colors)
- `'sequential_green'` - Green gradient (9 colors)

**Diverging (for signed data):**
- `'diverging_red_blue'` - Red-blue scale (11 colors)
- `'diverging_purple_green'` - Purple-green scale (11 colors)

### Color Utilities

```python
from src.visualization.themes import lighten_color, darken_color, interpolate_colors

# Lighten a color
light_blue = lighten_color('#1f77b4', amount=0.5)

# Darken a color
dark_blue = darken_color('#1f77b4', amount=0.5)

# Create gradient
gradient = interpolate_colors('#1f77b4', '#ff7f0e', n=10)
```

---

## Common Patterns

### Pattern 1: Publication-Ready Figure

```python
# Single configuration for publication
fig = system.plotter.plot_trajectory(
    result['t'],
    result['x'],
    state_names=['θ', 'ω'],
    color_scheme='colorblind_safe',  # Accessible
    theme='publication',              # Professional style
    title='Pendulum Response'
)

# Export
fig.write_html('figure.html')
fig.write_image('figure.pdf')  # Requires kaleido
fig.write_image('figure.png', width=1200, height=800, scale=2)
```

### Pattern 2: Multi-Subplot Analysis

```python
from plotly.subplots import make_subplots

# Create custom subplot layout
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=('State 1', 'State 2', 'Control', 'Phase Portrait'),
    specs=[[{'type': 'scatter'}, {'type': 'scatter'}],
           [{'type': 'scatter'}, {'type': 'scatter'}]]
)

# Add traces
fig.add_trace(
    go.Scatter(x=result['t'], y=result['x'][:, 0], name='x₁'),
    row=1, col=1
)
fig.add_trace(
    go.Scatter(x=result['t'], y=result['x'][:, 1], name='x₂'),
    row=1, col=2
)
fig.add_trace(
    go.Scatter(x=result['t'], y=result['u'][:, 0], name='u'),
    row=2, col=1
)
fig.add_trace(
    go.Scatter(x=result['x'][:, 0], y=result['x'][:, 1], name='Phase'),
    row=2, col=2
)

# Apply theme
fig = PlotThemes.apply_theme(fig, theme='publication')
```

### Pattern 3: Batch Visualization

```python
# Monte Carlo simulation
n_runs = 50
x_batch = np.zeros((n_runs, len(t), 2))

for i in range(n_runs):
    x0_noisy = x0 + 0.1 * np.random.randn(2)
    result = system.integrate(x0_noisy, u=None, t_span=(0, 10))
    x_batch[i] = result['x']

# Plot all runs (mean ± std automatically)
fig = system.plotter.plot_trajectory(
    t, x_batch,
    state_names=['θ', 'ω'],
    show_std=True,
    theme='publication'
)

# Or plot all individual runs
fig = go.Figure()
for i in range(n_runs):
    fig.add_trace(go.Scatter(
        x=t,
        y=x_batch[i, :, 0],
        mode='lines',
        line=dict(color='blue', width=0.5),
        opacity=0.3,
        showlegend=(i == 0),
        name='Individual runs' if i == 0 else ''
    ))

fig = PlotThemes.apply_theme(fig, theme='publication')
```

### Pattern 4: Interactive Dashboard

```python
# Create interactive dashboard with multiple views
from plotly.subplots import make_subplots

fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=(
        'Time-Domain Response',
        'Phase Portrait',
        'Control Input',
        'Eigenvalue Map'
    )
)

# Time domain
fig.add_trace(
    go.Scatter(x=result['t'], y=result['x'][:, 0], name='State'),
    row=1, col=1
)

# Phase portrait
fig.add_trace(
    go.Scatter(x=result['x'][:, 0], y=result['x'][:, 1], 
               mode='lines+markers', name='Trajectory'),
    row=1, col=2
)

# Control
fig.add_trace(
    go.Scatter(x=result['t'], y=result['u'][:, 0], name='Control'),
    row=2, col=1
)

# Eigenvalues
eigs = lqr['closed_loop_eigenvalues']
fig.add_trace(
    go.Scatter(x=np.real(eigs), y=np.imag(eigs),
               mode='markers', marker=dict(size=10), name='Poles'),
    row=2, col=2
)

fig.update_layout(height=800, showlegend=True)
fig = PlotThemes.apply_theme(fig, theme='publication')
fig.show()
```

### Pattern 5: Animation (Time Evolution)

```python
# Animated trajectory
frames = []
for i in range(0, len(t), 5):  # Every 5th point
    frame = go.Frame(
        data=[go.Scatter(
            x=result['x'][:i, 0],
            y=result['x'][:i, 1],
            mode='lines+markers',
            marker=dict(size=8)
        )],
        name=str(i)
    )
    frames.append(frame)

fig = go.Figure(
    data=[go.Scatter(x=[], y=[], mode='lines+markers')],
    frames=frames
)

fig.update_layout(
    updatemenus=[{
        'type': 'buttons',
        'buttons': [
            {'label': 'Play', 'method': 'animate',
             'args': [None, {'frame': {'duration': 50}}]},
            {'label': 'Pause', 'method': 'animate',
             'args': [[None], {'frame': {'duration': 0}, 'mode': 'immediate'}]}
        ]
    }]
)

fig = PlotThemes.apply_theme(fig, theme='dark')
```

### Pattern 6: Backend-Agnostic Plotting

```python
import torch
import jax.numpy as jnp

# NumPy system
system_np = Pendulum()
result_np = system_np.integrate(x0_np, u=None, t_span=(0, 10))
fig = system_np.plotter.plot_trajectory(result_np['t'], result_np['x'])

# PyTorch system (on GPU)
system_torch = Pendulum()
system_torch.set_default_backend('torch')
system_torch.set_default_device('cuda:0')
x0_torch = torch.tensor(x0_np, device='cuda:0')
result_torch = system_torch.integrate(x0_torch, u=None, t_span=(0, 10))

# Plotter handles conversion automatically
fig = system_torch.plotter.plot_trajectory(
    result_torch['t'],  # Automatically converts GPU tensor to NumPy
    result_torch['x'],
    theme='publication'
)

# JAX system
system_jax = Pendulum()
system_jax.set_default_backend('jax')
x0_jax = jnp.array(x0_np)
result_jax = system_jax.integrate(x0_jax, u=None, t_span=(0, 10))

# Also handles JAX arrays
fig = system_jax.plotter.plot_trajectory(
    result_jax['t'],
    result_jax['x'],
    theme='publication'
)
```

---

## Troubleshooting

### Issue: Plot appears blank

**Possible Causes:**
- Data shape incorrect
- All NaN/Inf values
- Empty arrays

**Solutions:**
```python
# Check data
print(f"t shape: {t.shape}")
print(f"x shape: {x.shape}")
print(f"t range: [{t[0]}, {t[-1]}]")
print(f"x range: [{x.min()}, {x.max()}]")
print(f"NaN count: {np.isnan(x).sum()}")

# Verify dimensions
assert t.ndim == 1, "t must be 1D"
assert x.ndim in [2, 3], "x must be 2D or 3D"
if x.ndim == 2:
    assert x.shape[0] == len(t), "First dimension must match time"
```

### Issue: Theme not applying

**Problem:**
```python
# This doesn't work
fig = plotter.plot_trajectory(t, x)
PlotThemes.apply_theme(fig, theme='publication')  # Returns new figure!
fig.show()  # Shows old figure without theme
```

**Solution:**
```python
# Correct: Capture returned figure
fig = plotter.plot_trajectory(t, x)
fig = PlotThemes.apply_theme(fig, theme='publication')
fig.show()

# Or use theme parameter directly
fig = plotter.plot_trajectory(t, x, theme='publication')
fig.show()
```

### Issue: Batched data not showing std bands

**Check:**
```python
# Ensure show_std=True
fig = plotter.plot_trajectory(
    t, x_batch,
    show_std=True  # Must be True for std bands
)

# Verify batch dimension
print(f"Shape: {x_batch.shape}")
# Should be (n_batch, T, nx), not (T, nx)
```

### Issue: Colors cycling too fast

**Problem:** Too many traces, colors repeat

**Solution:**
```python
# Use larger color scheme
fig = plotter.plot_comparison(
    many_results,
    color_scheme='plotly'  # 10 colors instead of default
)

# Or create custom colors
from src.visualization.themes import interpolate_colors

custom_colors = interpolate_colors('#1f77b4', '#ff7f0e', n=20)
# Use in manual plotting
```

### Issue: 3D plot not interactive

**Check browser compatibility:**
```python
# Ensure using modern browser
# Chrome, Firefox, Safari all support WebGL

# Try different renderer
fig.show(renderer='browser')  # Opens in default browser
fig.show(renderer='notebook')  # For Jupyter
```

### Issue: Export to PDF fails

**Solution:**
```python
# Install kaleido
# pip install kaleido

# Then export
fig.write_image('figure.pdf')
fig.write_image('figure.png', width=1200, height=800, scale=2)
```

---

## Direct Plotter Usage (Without System)

For cases where you need plotting without a system object:

```python
from src.visualization.trajectory_plotter import TrajectoryPlotter
from src.visualization.phase_portrait import PhasePortraitPlotter
from src.visualization.control_plots import ControlPlotter

# Create plotters
traj_plotter = TrajectoryPlotter(backend='numpy', theme='publication')
phase_plotter = PhasePortraitPlotter(backend='numpy', theme='publication')
control_plotter = ControlPlotter(backend='numpy', theme='publication')

# Use directly
fig = traj_plotter.plot_trajectory(t, x, state_names=['x₁', 'x₂'])
fig = phase_plotter.plot_2d(x, state_names=('x₁', 'x₂'))
fig = control_plotter.plot_eigenvalue_map(eigenvalues, system_type='continuous')
```

---

## Advanced Customization

### Custom Color Scheme

```python
# Define custom colors
CUSTOM_COLORS = [
    '#1B9E77',  # Teal
    '#D95F02',  # Orange
    '#7570B3',  # Purple
    '#E7298A',  # Pink
    '#66A61E',  # Green
]

# Use in plotting
fig = go.Figure()
for i, (name, result) in enumerate(results.items()):
    fig.add_trace(go.Scatter(
        x=result['t'],
        y=result['x'][:, 0],
        name=name,
        line=dict(color=CUSTOM_COLORS[i % len(CUSTOM_COLORS)])
    ))
```

### Custom Theme

```python
# Define custom theme dictionary
CUSTOM_THEME = {
    'template': 'simple_white',
    'font_family': 'Helvetica, sans-serif',
    'font_size': 13,
    'title_font_size': 17,
    'axis_line_color': '#333',
    'grid_color': '#DDD',
    'background_color': '#FAFAFA',
}

# Apply manually
fig.update_layout(
    template=CUSTOM_THEME['template'],
    font=dict(
        family=CUSTOM_THEME['font_family'],
        size=CUSTOM_THEME['font_size']
    ),
    title_font_size=CUSTOM_THEME['title_font_size'],
    # ... etc
)
```

### Subplot Customization

```python
# Fine-grained control over subplots
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=('State 1', 'State 2', 'Control', 'Phase'),
    horizontal_spacing=0.15,
    vertical_spacing=0.12,
    row_heights=[0.6, 0.4],
    column_widths=[0.5, 0.5]
)

# Update individual subplot axes
fig.update_xaxes(title_text="Time (s)", row=1, col=1)
fig.update_yaxes(title_text="Angle (rad)", row=1, col=1)
```

---

## Best Practices

### 1. Always Use Themes for Consistency

```python
# Good: Consistent theming
fig1 = plotter.plot_trajectory(t, x, theme='publication')
fig2 = plotter.plot_2d(x, theme='publication')

# Bad: Inconsistent default themes
fig1 = plotter.plot_trajectory(t, x)  # Default theme
fig2 = plotter.plot_2d(x, theme='dark')  # Different theme
```

### 2. Use Colorblind-Safe Palettes

```python
# For publications and presentations
fig = plotter.plot_comparison(
    results,
    color_scheme='colorblind_safe',
    theme='publication'
)
```

### 3. Add Informative Labels

```python
# Good: Clear labels
fig = plotter.plot_trajectory(
    t, x,
    state_names=['Angle (rad)', 'Angular Velocity (rad/s)'],
    title='Pendulum Response to Initial Condition'
)

# Bad: Generic labels
fig = plotter.plot_trajectory(t, x)  # x1, x2, etc.
```

### 4. Export Multiple Formats

```python
# Save interactive HTML
fig.write_html('figure_interactive.html')

# Save static images
fig.write_image('figure.pdf')  # For papers
fig.write_image('figure.png', width=1200, height=800, scale=2)  # For slides
```

---

## References

- **Architecture:** See `Visualization_Framework_Architecture.md` for complete details
- **Theming:** See `src/visualization/themes.py`
- **Plotters:** See `src/visualization/trajectory_plotter.py`, `phase_portrait.py`, `control_plots.py`
- **Plotly Documentation:** https://plotly.com/python/
