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
Visual Test Suite for Phase Portrait Plotter

Generates HTML files for visual inspection of phase portrait plotting functionality.
Run this script to create a gallery of test plots.

Usage:
    python visual_test_phase_portrait.py

Output:
    Creates HTML files in ./visual_tests/phase_portrait/
"""

import numpy as np
from pathlib import Path

from src.visualization.phase_portrait import PhasePortraitPlotter


def setup_output_directory():
    """Create output directory for visual tests."""
    output_dir = Path("visual_tests/phase_portrait")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def solve_pendulum(t, theta0, omega0, damping=0.1):
    """Solve damped pendulum equations."""
    dt = t[1] - t[0]
    theta = np.zeros_like(t)
    omega = np.zeros_like(t)
    theta[0] = theta0
    omega[0] = omega0
    
    for i in range(len(t) - 1):
        theta[i+1] = theta[i] + dt * omega[i]
        omega[i+1] = omega[i] + dt * (-np.sin(theta[i]) - damping * omega[i])
    
    return np.column_stack([theta, omega])


def solve_van_der_pol(t, x0, y0, mu=1.0):
    """Solve Van der Pol oscillator."""
    dt = t[1] - t[0]
    x = np.zeros_like(t)
    y = np.zeros_like(t)
    x[0] = x0
    y[0] = y0
    
    for i in range(len(t) - 1):
        x[i+1] = x[i] + dt * y[i]
        y[i+1] = y[i] + dt * (mu * (1 - x[i]**2) * y[i] - x[i])
    
    return np.column_stack([x, y])


def solve_lorenz(t, x0, y0, z0, sigma=10, rho=28, beta=8/3):
    """Solve Lorenz attractor."""
    dt = t[1] - t[0]
    x = np.zeros_like(t)
    y = np.zeros_like(t)
    z = np.zeros_like(t)
    x[0], y[0], z[0] = x0, y0, z0
    
    for i in range(len(t) - 1):
        x[i+1] = x[i] + dt * sigma * (y[i] - x[i])
        y[i+1] = y[i] + dt * (x[i] * (rho - z[i]) - y[i])
        z[i+1] = z[i] + dt * (x[i] * y[i] - beta * z[i])
    
    return np.column_stack([x, y, z])


def test_1_simple_circle(output_dir):
    """Test 1: Simple circular trajectory."""
    print("Generating Test 1: Simple circle...")
    
    plotter = PhasePortraitPlotter()
    t = np.linspace(0, 2*np.pi, 200)
    x = np.column_stack([np.cos(t), np.sin(t)])
    
    fig = plotter.plot_2d(
        x,
        state_names=('x₁', 'x₂'),
        title='Test 1: Simple Circular Trajectory',
        show_direction=True,
        show_start_end=True
    )
    
    fig.write_html(output_dir / "01_simple_circle.html")
    print("  ✓ Saved: 01_simple_circle.html")


def test_2_inward_spiral(output_dir):
    """Test 2: Inward spiral (damped oscillator)."""
    print("Generating Test 2: Inward spiral...")
    
    plotter = PhasePortraitPlotter()
    t = np.linspace(0, 20, 500)
    r = np.exp(-0.1 * t)
    x = np.column_stack([r * np.cos(t), r * np.sin(t)])
    
    fig = plotter.plot_2d(
        x,
        state_names=('Position', 'Velocity'),
        title='Test 2: Inward Spiral (Damped Oscillator)',
        show_direction=True,
        show_start_end=True,
        theme='publication'
    )
    
    fig.write_html(output_dir / "02_inward_spiral.html")
    print("  ✓ Saved: 02_inward_spiral.html")


def test_3_pendulum_no_damping(output_dir):
    """Test 3: Pendulum without damping."""
    print("Generating Test 3: Pendulum (no damping)...")
    
    plotter = PhasePortraitPlotter()
    t = np.linspace(0, 20, 1000)
    x = solve_pendulum(t, theta0=0.5, omega0=0, damping=0.0)
    
    def pendulum_dynamics(theta, omega):
        return np.array([omega, -np.sin(theta)])
    
    equilibria = [
        np.array([0.0, 0.0]),
        np.array([np.pi, 0.0]),
        np.array([-np.pi, 0.0])
    ]
    
    fig = plotter.plot_2d(
        x,
        state_names=('θ (rad)', 'ω (rad/s)'),
        vector_field=pendulum_dynamics,
        equilibria=equilibria,
        title='Test 3: Pendulum Phase Portrait (Undamped)',
        show_direction=True
    )
    
    fig.write_html(output_dir / "03_pendulum_no_damping.html")
    print("  ✓ Saved: 03_pendulum_no_damping.html")


def test_4_pendulum_with_damping(output_dir):
    """Test 4: Damped pendulum."""
    print("Generating Test 4: Damped pendulum...")
    
    plotter = PhasePortraitPlotter()
    t = np.linspace(0, 30, 1500)
    x = solve_pendulum(t, theta0=2.0, omega0=0, damping=0.2)
    
    def damped_pendulum_dynamics(theta, omega):
        return np.array([omega, -np.sin(theta) - 0.2*omega])
    
    equilibria = [np.array([0.0, 0.0])]
    
    fig = plotter.plot_2d(
        x,
        state_names=('θ (rad)', 'ω (rad/s)'),
        vector_field=damped_pendulum_dynamics,
        equilibria=equilibria,
        title='Test 4: Damped Pendulum (ζ=0.2)',
        show_direction=True,
        color_scheme='colorblind_safe',
        theme='publication'
    )
    
    fig.write_html(output_dir / "04_pendulum_damped.html")
    print("  ✓ Saved: 04_pendulum_damped.html")


def test_5_van_der_pol(output_dir):
    """Test 5: Van der Pol oscillator."""
    print("Generating Test 5: Van der Pol oscillator...")
    
    plotter = PhasePortraitPlotter()
    t = np.linspace(0, 50, 2000)
    x = solve_van_der_pol(t, x0=0.1, y0=0.1, mu=1.0)
    
    def van_der_pol_dynamics(x1, x2):
        mu = 1.0
        return np.array([x2, mu * (1 - x1**2) * x2 - x1])
    
    equilibria = [np.array([0.0, 0.0])]
    
    fig = plotter.plot_2d(
        x,
        state_names=('x', 'dx/dt'),
        vector_field=van_der_pol_dynamics,
        equilibria=equilibria,
        title='Test 5: Van der Pol Oscillator (μ=1.0)',
        show_direction=True
    )
    
    fig.write_html(output_dir / "05_van_der_pol.html")
    print("  ✓ Saved: 05_van_der_pol.html")


def test_6_limit_cycle(output_dir):
    """Test 6: Limit cycle visualization."""
    print("Generating Test 6: Limit cycle...")
    
    plotter = PhasePortraitPlotter()
    t = np.linspace(0, 50, 2000)
    x = solve_van_der_pol(t, x0=2.0, y0=0, mu=2.0)
    
    fig = plotter.plot_limit_cycle(
        x,
        state_names=('x', 'dx/dt'),
        title='Test 6: Limit Cycle (Van der Pol, μ=2.0)',
        theme='publication'
    )
    
    fig.write_html(output_dir / "06_limit_cycle.html")
    print("  ✓ Saved: 06_limit_cycle.html")


def test_7_batched_trajectories(output_dir):
    """Test 7: Multiple initial conditions."""
    print("Generating Test 7: Batched trajectories...")
    
    plotter = PhasePortraitPlotter()
    t = np.linspace(0, 30, 1000)
    
    # Multiple initial conditions for pendulum
    initial_conditions = [
        (0.5, 0),
        (1.0, 0),
        (1.5, 0),
        (2.0, 0),
        (2.5, 0)
    ]
    
    x_batch = np.stack([
        solve_pendulum(t, theta0=theta0, omega0=omega0, damping=0.15)
        for theta0, omega0 in initial_conditions
    ])
    
    def damped_pendulum_dynamics(theta, omega):
        return np.array([omega, -np.sin(theta) - 0.15*omega])
    
    equilibria = [np.array([0.0, 0.0])]
    
    fig = plotter.plot_2d(
        x_batch,
        state_names=('θ (rad)', 'ω (rad/s)'),
        vector_field=damped_pendulum_dynamics,
        equilibria=equilibria,
        title='Test 7: Multiple Initial Conditions',
        color_scheme='colorblind_safe'
    )
    
    fig.write_html(output_dir / "07_batched_trajectories.html")
    print("  ✓ Saved: 07_batched_trajectories.html")


def test_8_lorenz_attractor(output_dir):
    """Test 8: Lorenz attractor (3D)."""
    print("Generating Test 8: Lorenz attractor...")
    
    plotter = PhasePortraitPlotter()
    t = np.linspace(0, 25, 5000)
    x = solve_lorenz(t, x0=1.0, y0=1.0, z0=1.0)
    
    fig = plotter.plot_3d(
        x,
        state_names=('x', 'y', 'z'),
        title='Test 8: Lorenz Attractor',
        show_direction=True,
        show_start_end=True,
        theme='dark'
    )
    
    fig.write_html(output_dir / "08_lorenz_attractor.html")
    print("  ✓ Saved: 08_lorenz_attractor.html")


def test_9_lorenz_batched(output_dir):
    """Test 9: Lorenz attractor with multiple ICs."""
    print("Generating Test 9: Lorenz with multiple ICs...")
    
    plotter = PhasePortraitPlotter()
    t = np.linspace(0, 20, 3000)
    
    # Slightly different initial conditions
    initial_conditions = [
        (1.0, 1.0, 1.0),
        (1.01, 1.0, 1.0),
        (1.0, 1.01, 1.0),
    ]
    
    x_batch = np.stack([
        solve_lorenz(t, x0=x0, y0=y0, z0=z0)
        for x0, y0, z0 in initial_conditions
    ])
    
    fig = plotter.plot_3d(
        x_batch,
        state_names=('x', 'y', 'z'),
        title='Test 9: Lorenz Sensitivity to ICs',
        show_direction=True,
        color_scheme='colorblind_safe',
        theme='publication'
    )
    
    fig.write_html(output_dir / "09_lorenz_batched.html")
    print("  ✓ Saved: 09_lorenz_batched.html")


def test_10_harmonic_oscillator(output_dir):
    """Test 10: Simple harmonic oscillator."""
    print("Generating Test 10: Harmonic oscillator...")
    
    plotter = PhasePortraitPlotter()
    t = np.linspace(0, 20, 400)
    x = np.column_stack([np.cos(t), -np.sin(t)])
    
    def harmonic_dynamics(x1, x2):
        return np.array([x2, -x1])
    
    equilibria = [np.array([0.0, 0.0])]
    
    fig = plotter.plot_2d(
        x,
        state_names=('Position', 'Velocity'),
        vector_field=harmonic_dynamics,
        equilibria=equilibria,
        title='Test 10: Simple Harmonic Oscillator',
        show_direction=True
    )
    
    fig.write_html(output_dir / "10_harmonic_oscillator.html")
    print("  ✓ Saved: 10_harmonic_oscillator.html")


def test_11_duffing_oscillator(output_dir):
    """Test 11: Duffing oscillator."""
    print("Generating Test 11: Duffing oscillator...")
    
    plotter = PhasePortraitPlotter()
    
    # Duffing equation: x'' + delta*x' + alpha*x + beta*x^3 = 0
    t = np.linspace(0, 50, 2000)
    dt = t[1] - t[0]
    
    alpha, beta, delta = -1.0, 1.0, 0.1
    x = np.zeros_like(t)
    v = np.zeros_like(t)
    x[0], v[0] = 0.5, 0.5
    
    for i in range(len(t) - 1):
        x[i+1] = x[i] + dt * v[i]
        v[i+1] = v[i] + dt * (-delta*v[i] - alpha*x[i] - beta*x[i]**3)
    
    trajectory = np.column_stack([x, v])
    
    def duffing_dynamics(x1, x2):
        return np.array([x2, -0.1*x2 + x1 - x1**3])
    
    equilibria = [
        np.array([0.0, 0.0]),
        np.array([1.0, 0.0]),
        np.array([-1.0, 0.0])
    ]
    
    fig = plotter.plot_2d(
        trajectory,
        state_names=('x', 'dx/dt'),
        vector_field=duffing_dynamics,
        equilibria=equilibria,
        title='Test 11: Duffing Oscillator (α=-1, β=1)',
        show_direction=True,
        theme='publication'
    )
    
    fig.write_html(output_dir / "11_duffing_oscillator.html")
    print("  ✓ Saved: 11_duffing_oscillator.html")


def test_12_predator_prey(output_dir):
    """Test 12: Predator-prey (Lotka-Volterra)."""
    print("Generating Test 12: Predator-prey model...")
    
    plotter = PhasePortraitPlotter()
    
    # Lotka-Volterra equations
    t = np.linspace(0, 50, 2000)
    dt = t[1] - t[0]
    
    alpha, beta, gamma, delta = 1.0, 0.1, 0.1, 0.02
    prey = np.zeros_like(t)
    pred = np.zeros_like(t)
    prey[0], pred[0] = 10.0, 5.0
    
    for i in range(len(t) - 1):
        prey[i+1] = prey[i] + dt * (alpha*prey[i] - beta*prey[i]*pred[i])
        pred[i+1] = pred[i] + dt * (-gamma*pred[i] + delta*prey[i]*pred[i])
    
    x = np.column_stack([prey, pred])
    
    def lotka_volterra_dynamics(x1, x2):
        return np.array([
            x1 - 0.1*x1*x2,
            -0.1*x2 + 0.02*x1*x2
        ])
    
    equilibria = [np.array([10.0, 10.0])]
    
    fig = plotter.plot_2d(
        x,
        state_names=('Prey', 'Predator'),
        vector_field=lotka_volterra_dynamics,
        equilibria=equilibria,
        title='Test 12: Predator-Prey Dynamics (Lotka-Volterra)',
        show_direction=True
    )
    
    fig.write_html(output_dir / "12_predator_prey.html")
    print("  ✓ Saved: 12_predator_prey.html")


def test_13_3d_helix(output_dir):
    """Test 13: 3D helix."""
    print("Generating Test 13: 3D helix...")
    
    plotter = PhasePortraitPlotter()
    t = np.linspace(0, 4*np.pi, 500)
    x = np.column_stack([
        np.cos(t),
        np.sin(t),
        t * 0.2
    ])
    
    fig = plotter.plot_3d(
        x,
        state_names=('x', 'y', 'z'),
        title='Test 13: 3D Helix',
        show_direction=True,
        show_start_end=True
    )
    
    fig.write_html(output_dir / "13_3d_helix.html")
    print("  ✓ Saved: 13_3d_helix.html")


def test_14_rossler_attractor(output_dir):
    """Test 14: Rössler attractor."""
    print("Generating Test 14: Rössler attractor...")
    
    plotter = PhasePortraitPlotter()
    
    # Rössler system
    t = np.linspace(0, 100, 5000)
    dt = t[1] - t[0]
    
    a, b, c = 0.2, 0.2, 5.7
    x = np.zeros_like(t)
    y = np.zeros_like(t)
    z = np.zeros_like(t)
    x[0], y[0], z[0] = 1.0, 1.0, 1.0
    
    for i in range(len(t) - 1):
        x[i+1] = x[i] + dt * (-y[i] - z[i])
        y[i+1] = y[i] + dt * (x[i] + a*y[i])
        z[i+1] = z[i] + dt * (b + z[i]*(x[i] - c))
    
    trajectory = np.column_stack([x, y, z])
    
    fig = plotter.plot_3d(
        trajectory,
        state_names=('x', 'y', 'z'),
        title='Test 14: Rössler Attractor',
        show_direction=True,
        theme='dark'
    )
    
    fig.write_html(output_dir / "14_rossler_attractor.html")
    print("  ✓ Saved: 14_rossler_attractor.html")


def test_15_batched_circles(output_dir):
    """Test 15: Batched circular trajectories."""
    print("Generating Test 15: Batched circles...")
    
    plotter = PhasePortraitPlotter()
    t = np.linspace(0, 2*np.pi, 200)
    
    radii = [0.5, 1.0, 1.5, 2.0]
    x_batch = np.stack([
        np.column_stack([r * np.cos(t), r * np.sin(t)])
        for r in radii
    ])
    
    fig = plotter.plot_2d(
        x_batch,
        state_names=('x', 'y'),
        title='Test 15: Concentric Circles',
        show_direction=True,
        color_scheme='tableau'
    )
    
    fig.write_html(output_dir / "15_batched_circles.html")
    print("  ✓ Saved: 15_batched_circles.html")


def test_16_saddle_point(output_dir):
    """Test 16: Saddle point dynamics."""
    print("Generating Test 16: Saddle point...")
    
    plotter = PhasePortraitPlotter()
    
    # Linear saddle: x' = x, y' = -y
    t = np.linspace(0, 3, 300)
    dt = t[1] - t[0]
    
    # Multiple trajectories showing saddle behavior
    initial_conditions = [
        (1.0, 0.1),
        (1.0, -0.1),
        (0.1, 1.0),
        (-0.1, 1.0),
    ]
    
    trajectories = []
    for x0, y0 in initial_conditions:
        x_traj = np.zeros_like(t)
        y_traj = np.zeros_like(t)
        x_traj[0], y_traj[0] = x0, y0
        
        for i in range(len(t) - 1):
            x_traj[i+1] = x_traj[i] + dt * x_traj[i]
            y_traj[i+1] = y_traj[i] + dt * (-y_traj[i])
        
        trajectories.append(np.column_stack([x_traj, y_traj]))
    
    x_batch = np.stack(trajectories)
    
    def saddle_dynamics(x1, x2):
        return np.array([x1, -x2])
    
    equilibria = [np.array([0.0, 0.0])]
    
    fig = plotter.plot_2d(
        x_batch,
        state_names=('x', 'y'),
        vector_field=saddle_dynamics,
        equilibria=equilibria,
        title='Test 16: Saddle Point Dynamics',
        show_direction=True,
        color_scheme='d3'
    )
    
    fig.write_html(output_dir / "16_saddle_point.html")
    print("  ✓ Saved: 16_saddle_point.html")


def test_17_theme_default(output_dir):
    """Test 17: Default theme."""
    print("Generating Test 17: Default theme...")
    
    plotter = PhasePortraitPlotter()
    t = np.linspace(0, 10, 500)
    x = solve_van_der_pol(t, x0=0.5, y0=0, mu=1.5)
    
    fig = plotter.plot_2d(
        x,
        state_names=('x', 'dx/dt'),
        title='Test 17: Default Theme',
        theme='default'
    )
    
    fig.write_html(output_dir / "17_theme_default.html")
    print("  ✓ Saved: 17_theme_default.html")


def test_18_theme_publication(output_dir):
    """Test 18: Publication theme."""
    print("Generating Test 18: Publication theme...")
    
    plotter = PhasePortraitPlotter()
    t = np.linspace(0, 10, 500)
    x = solve_van_der_pol(t, x0=0.5, y0=0, mu=1.5)
    
    fig = plotter.plot_2d(
        x,
        state_names=('x', 'dx/dt'),
        title='Test 18: Publication Theme',
        color_scheme='colorblind_safe',
        theme='publication'
    )
    
    fig.write_html(output_dir / "18_theme_publication.html")
    print("  ✓ Saved: 18_theme_publication.html")


def test_19_theme_dark(output_dir):
    """Test 19: Dark theme."""
    print("Generating Test 19: Dark theme...")
    
    plotter = PhasePortraitPlotter()
    t = np.linspace(0, 10, 500)
    x = solve_van_der_pol(t, x0=0.5, y0=0, mu=1.5)
    
    fig = plotter.plot_2d(
        x,
        state_names=('x', 'dx/dt'),
        title='Test 19: Dark Theme',
        theme='dark'
    )
    
    fig.write_html(output_dir / "19_theme_dark.html")
    print("  ✓ Saved: 19_theme_dark.html")


def test_20_theme_presentation(output_dir):
    """Test 20: Presentation theme."""
    print("Generating Test 20: Presentation theme...")
    
    plotter = PhasePortraitPlotter()
    t = np.linspace(0, 10, 500)
    x = solve_van_der_pol(t, x0=0.5, y0=0, mu=1.5)
    
    fig = plotter.plot_2d(
        x,
        state_names=('x', 'dx/dt'),
        title='Test 20: Presentation Theme',
        theme='presentation'
    )
    
    fig.write_html(output_dir / "20_theme_presentation.html")
    print("  ✓ Saved: 20_theme_presentation.html")


def test_21_color_schemes(output_dir):
    """Test 21: Color scheme comparison."""
    print("Generating Test 21: Color scheme comparison...")
    
    t = np.linspace(0, 20, 800)
    
    # Multiple pendulum trajectories
    initial_angles = [0.5, 1.0, 1.5, 2.0]
    x_batch = np.stack([
        solve_pendulum(t, theta0=theta0, omega0=0, damping=0.1)
        for theta0 in initial_angles
    ])
    
    schemes = ['plotly', 'colorblind_safe', 'tableau', 'd3']
    
    for i, scheme in enumerate(schemes, start=1):
        plotter = PhasePortraitPlotter()
        
        fig = plotter.plot_2d(
            x_batch,
            state_names=('θ', 'ω'),
            title=f'Test 21.{i}: Color Scheme "{scheme}"',
            color_scheme=scheme,
            show_direction=False
        )
        
        fig.write_html(output_dir / f"21_{i}_color_{scheme}.html")
        print(f"  ✓ Saved: 21_{i}_color_{scheme}.html")


def test_22_spiral_gallery(output_dir):
    """Test 22: Gallery of different spiral types."""
    print("Generating Test 22: Spiral gallery...")
    
    plotter = PhasePortraitPlotter()
    
    t = np.linspace(0, 20, 1000)
    
    # Different damping values
    dampings = [0.05, 0.1, 0.2, 0.4]
    x_batch = np.stack([
        solve_pendulum(t, theta0=2.0, omega0=0, damping=d)
        for d in dampings
    ])
    
    fig = plotter.plot_2d(
        x_batch,
        state_names=('θ', 'ω'),
        title='Test 22: Effect of Damping on Phase Portrait',
        show_direction=False,
        color_scheme='colorblind_safe',
        theme='publication'
    )
    
    fig.write_html(output_dir / "22_spiral_gallery.html")
    print("  ✓ Saved: 22_spiral_gallery.html")


def test_23_3d_torus(output_dir):
    """Test 23: 3D torus trajectory."""
    print("Generating Test 23: 3D torus...")
    
    plotter = PhasePortraitPlotter()
    
    # Parametric torus
    t = np.linspace(0, 4*np.pi, 800)
    R, r = 3.0, 1.0  # Major and minor radius
    
    x = (R + r * np.cos(5*t)) * np.cos(t)
    y = (R + r * np.cos(5*t)) * np.sin(t)
    z = r * np.sin(5*t)
    
    trajectory = np.column_stack([x, y, z])
    
    fig = plotter.plot_3d(
        trajectory,
        state_names=('x', 'y', 'z'),
        title='Test 23: 3D Torus Trajectory',
        show_direction=True,
        theme='default'
    )
    
    fig.write_html(output_dir / "23_3d_torus.html")
    print("  ✓ Saved: 23_3d_torus.html")


def test_24_separatrix(output_dir):
    """Test 24: Separatrix in pendulum."""
    print("Generating Test 24: Separatrix...")
    
    plotter = PhasePortraitPlotter()
    
    t = np.linspace(0, 20, 1000)
    
    # Trajectories on both sides of separatrix
    initial_conditions = [
        (0.5, 0),   # Oscillation
        (1.5, 0),   # Oscillation
        (2.5, 0),   # Oscillation
        (3.0, 0),   # Near separatrix
        (3.5, 0),   # Rotation
    ]
    
    x_batch = np.stack([
        solve_pendulum(t, theta0=theta0, omega0=omega0, damping=0.0)
        for theta0, omega0 in initial_conditions
    ])
    
    def pendulum_dynamics(theta, omega):
        return np.array([omega, -np.sin(theta)])
    
    equilibria = [
        np.array([0.0, 0.0]),
        np.array([np.pi, 0.0]),
        np.array([-np.pi, 0.0])
    ]
    
    fig = plotter.plot_2d(
        x_batch,
        state_names=('θ (rad)', 'ω (rad/s)'),
        vector_field=pendulum_dynamics,
        equilibria=equilibria,
        title='Test 24: Pendulum Separatrix',
        show_direction=False,
        color_scheme='tableau'
    )
    
    fig.write_html(output_dir / "24_separatrix.html")
    print("  ✓ Saved: 24_separatrix.html")


def generate_index_html(output_dir):
    """Generate index.html for easy navigation."""
    print("\nGenerating index.html...")
    
    html_files = sorted(output_dir.glob("*.html"))
    
    html_content = """<!DOCTYPE html>
<html>
<head>
    <title>Phase Portrait Plotter Visual Tests</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        h1 {
            color: #333;
            border-bottom: 3px solid #00CC96;
            padding-bottom: 10px;
        }
        .category {
            background: #e8f8f4;
            padding: 15px;
            border-radius: 8px;
            margin: 20px 0;
        }
        .category h2 {
            margin: 0 0 10px 0;
            color: #00AA77;
        }
        .test-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
            gap: 20px;
            margin-top: 15px;
        }
        .test-card {
            background: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            transition: transform 0.2s;
        }
        .test-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }
        .test-card h3 {
            margin-top: 0;
            color: #00CC96;
            font-size: 1.1em;
        }
        .test-card p {
            color: #666;
            font-size: 0.95em;
            margin: 10px 0;
        }
        .test-card .badge {
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.75em;
            margin: 5px 5px 5px 0;
        }
        .badge-2d {
            background: #d4edff;
            color: #0066cc;
        }
        .badge-3d {
            background: #ffe4d4;
            color: #cc6600;
        }
        .badge-vector {
            background: #e4d4ff;
            color: #6600cc;
        }
        .test-card a {
            display: inline-block;
            background: #00CC96;
            color: white;
            padding: 10px 20px;
            text-decoration: none;
            border-radius: 4px;
            margin-top: 10px;
            font-size: 0.9em;
        }
        .test-card a:hover {
            background: #00AA77;
        }
        .stats {
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 20px;
        }
        .stat-item {
            text-align: center;
        }
        .stat-number {
            font-size: 2em;
            font-weight: bold;
            color: #00CC96;
        }
        .stat-label {
            color: #666;
            font-size: 0.9em;
        }
    </style>
</head>
<body>
    <h1>🌀 Phase Portrait Plotter Visual Test Suite</h1>
    <p>Visual inspection gallery for phase space visualization functionality.</p>
    <p><strong>Generated:</strong> """ + str(Path.cwd() / output_dir) + """</p>
    
    <div class="stats">
        <div class="stat-item">
            <div class="stat-number">28</div>
            <div class="stat-label">Test Cases</div>
        </div>
        <div class="stat-item">
            <div class="stat-number">8</div>
            <div class="stat-label">Dynamical Systems</div>
        </div>
        <div class="stat-item">
            <div class="stat-number">4</div>
            <div class="stat-label">Themes</div>
        </div>
        <div class="stat-item">
            <div class="stat-number">4</div>
            <div class="stat-label">Color Schemes</div>
        </div>
    </div>
"""
    
    categories = {
        '2D Phase Portraits - Basic': [
            (1, 'Simple circular trajectory', ['2D']),
            (2, 'Inward spiral (damped)', ['2D']),
            (10, 'Harmonic oscillator', ['2D', 'Vector']),
        ],
        '2D Phase Portraits - Nonlinear Systems': [
            (3, 'Pendulum (no damping)', ['2D', 'Vector']),
            (4, 'Damped pendulum', ['2D', 'Vector']),
            (5, 'Van der Pol oscillator', ['2D', 'Vector']),
            (11, 'Duffing oscillator', ['2D', 'Vector']),
            (12, 'Predator-prey (Lotka-Volterra)', ['2D', 'Vector']),
        ],
        'Limit Cycles & Special Behaviors': [
            (6, 'Limit cycle detection', ['2D']),
            (16, 'Saddle point dynamics', ['2D', 'Vector']),
            (24, 'Pendulum separatrix', ['2D', 'Vector']),
        ],
        '3D Phase Portraits': [
            (8, 'Lorenz attractor', ['3D']),
            (9, 'Lorenz with multiple ICs', ['3D']),
            (13, '3D helix', ['3D']),
            (14, 'Rössler attractor', ['3D']),
            (23, '3D torus', ['3D']),
        ],
        'Batched Trajectories': [
            (7, 'Multiple initial conditions', ['2D', 'Vector']),
            (15, 'Concentric circles', ['2D']),
            (22, 'Effect of damping', ['2D']),
        ],
        'Themes & Colors': [
            (17, 'Default theme', ['2D']),
            (18, 'Publication theme', ['2D']),
            (19, 'Dark theme', ['2D']),
            (20, 'Presentation theme', ['2D']),
            (21, 'Color scheme comparison (4 schemes)', ['2D']),
        ]
    }
    
    for category, tests in categories.items():
        html_content += f"""
    <div class="category">
        <h2>{category}</h2>
        <div class="test-grid">
"""
        
        for test_info in tests:
            test_num = test_info[0]
            desc = test_info[1]
            badges = test_info[2] if len(test_info) > 2 else []
            
            # Find matching files
            if test_num in [21]:
                # Multiple files for color scheme comparison
                pattern = f"{test_num:02d}_*"
                matching_files = sorted(output_dir.glob(f"{pattern}.html"))
                for file in matching_files:
                    if file.name == "index.html":
                        continue
                    file_desc = file.stem.replace('_', ' ').title()
                    
                    badges_html = ''
                    for badge in badges:
                        badge_class = f'badge-{badge.lower()}'
                        badges_html += f'<span class="badge {badge_class}">{badge}</span>'
                    if 'vector' in file.name.lower() or 'Vector' in badges:
                        badges_html += '<span class="badge badge-vector">Vector Field</span>'
                    
                    html_content += f"""
            <div class="test-card">
                <h3>{file_desc}</h3>
                <p>{desc}</p>
                {badges_html}
                <a href="{file.name}" target="_blank">View Plot →</a>
            </div>
"""
            else:
                # Single file
                pattern = f"{test_num:02d}_*.html"
                matching_files = list(output_dir.glob(pattern))
                if matching_files:
                    file = matching_files[0]
                    
                    badges_html = ''
                    for badge in badges:
                        badge_class = f'badge-{badge.lower()}'
                        badges_html += f'<span class="badge {badge_class}">{badge}</span>'
                    
                    html_content += f"""
            <div class="test-card">
                <h3>Test {test_num}</h3>
                <p>{desc}</p>
                {badges_html}
                <a href="{file.name}" target="_blank">View Plot →</a>
            </div>
"""
        
        html_content += """
        </div>
    </div>
"""
    
    html_content += """
</body>
</html>
"""
    
    index_path = output_dir / "index.html"
    index_path.write_text(html_content)
    print(f"  ✓ Saved: index.html")


def main():
    """Run all visual tests."""
    print("=" * 70)
    print("Phase Portrait Plotter Visual Test Suite")
    print("=" * 70)
    print()
    
    output_dir = setup_output_directory()
    print(f"Output directory: {output_dir.absolute()}\n")
    
    # Run all tests
    test_1_simple_circle(output_dir)
    test_2_inward_spiral(output_dir)
    test_3_pendulum_no_damping(output_dir)
    test_4_pendulum_with_damping(output_dir)
    test_5_van_der_pol(output_dir)
    test_6_limit_cycle(output_dir)
    test_7_batched_trajectories(output_dir)
    test_8_lorenz_attractor(output_dir)
    test_9_lorenz_batched(output_dir)
    test_10_harmonic_oscillator(output_dir)
    test_11_duffing_oscillator(output_dir)
    test_12_predator_prey(output_dir)
    test_13_3d_helix(output_dir)
    test_14_rossler_attractor(output_dir)
    test_15_batched_circles(output_dir)
    test_16_saddle_point(output_dir)
    test_17_theme_default(output_dir)
    test_18_theme_publication(output_dir)
    test_19_theme_dark(output_dir)
    test_20_theme_presentation(output_dir)
    test_21_color_schemes(output_dir)
    test_22_spiral_gallery(output_dir)
    test_23_3d_torus(output_dir)
    test_24_separatrix(output_dir)
    
    # Generate index
    generate_index_html(output_dir)
    
    print("\n" + "=" * 70)
    print("✓ All visual tests generated successfully!")
    print("=" * 70)
    print(f"\nOpen this file in your browser:")
    print(f"  {(output_dir / 'index.html').absolute()}")
    print()


if __name__ == "__main__":
    main()