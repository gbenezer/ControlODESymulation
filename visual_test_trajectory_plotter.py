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
Visual Test Suite for Trajectory Plotter

Generates HTML files for visual inspection of trajectory plotting functionality.
Run this script to create a gallery of test plots.

Usage:
    python visual_test_trajectory_plotter.py

Output:
    Creates HTML files in ./visual_tests/trajectory_plotter/
"""

import numpy as np
from pathlib import Path

from src.visualization.trajectory_plotter import TrajectoryPlotter


def setup_output_directory():
    """Create output directory for visual tests."""
    output_dir = Path("visual_tests/trajectory_plotter")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def test_1_simple_trajectory(output_dir):
    """Test 1: Simple single-state trajectory."""
    print("Generating Test 1: Simple trajectory...")
    
    plotter = TrajectoryPlotter()
    t = np.linspace(0, 10, 200)
    x = np.sin(t)[:, None]
    
    fig = plotter.plot_trajectory(
        t, x,
        state_names=['sin(t)'],
        title='Test 1: Simple Trajectory'
    )
    
    fig.write_html(output_dir / "01_simple_trajectory.html")
    print("  ✓ Saved: 01_simple_trajectory.html")


def test_2_multi_state_trajectory(output_dir):
    """Test 2: Multi-state trajectory."""
    print("Generating Test 2: Multi-state trajectory...")
    
    plotter = TrajectoryPlotter()
    t = np.linspace(0, 10, 200)
    x = np.column_stack([
        np.sin(t),
        np.cos(t),
        np.sin(2*t),
        np.cos(2*t)
    ])
    
    fig = plotter.plot_trajectory(
        t, x,
        state_names=['sin(t)', 'cos(t)', 'sin(2t)', 'cos(2t)'],
        title='Test 2: Multi-State Trajectory'
    )
    
    fig.write_html(output_dir / "02_multi_state_trajectory.html")
    print("  ✓ Saved: 02_multi_state_trajectory.html")


def test_3_batched_trajectories(output_dir):
    """Test 3: Batched trajectories (Monte Carlo)."""
    print("Generating Test 3: Batched trajectories...")
    
    plotter = TrajectoryPlotter()
    t = np.linspace(0, 10, 200)
    
    # 5 trajectories with different phase shifts
    x_batch = np.stack([
        np.column_stack([np.sin(t + phi), np.cos(t + phi)])
        for phi in np.linspace(0, np.pi, 5)
    ])
    
    fig = plotter.plot_trajectory(
        t, x_batch,
        state_names=['sin(t+φ)', 'cos(t+φ)'],
        title='Test 3: Batched Trajectories',
        show_legend=True
    )
    
    fig.write_html(output_dir / "03_batched_trajectories.html")
    print("  ✓ Saved: 03_batched_trajectories.html")


def test_4_state_and_control(output_dir):
    """Test 4: State and control visualization."""
    print("Generating Test 4: State and control...")
    
    plotter = TrajectoryPlotter()
    t = np.linspace(0, 10, 200)
    
    # Damped oscillator
    omega = 2.0
    zeta = 0.1
    x = np.column_stack([
        np.exp(-zeta*omega*t) * np.sin(omega*t),
        -zeta*omega*np.exp(-zeta*omega*t)*np.sin(omega*t) + omega*np.exp(-zeta*omega*t)*np.cos(omega*t)
    ])
    
    # Control input
    u = -0.5 * x[:, 0:1] - 0.3 * x[:, 1:2]
    
    fig = plotter.plot_state_and_control(
        t, x, u,
        state_names=['Position', 'Velocity'],
        control_names=['Force'],
        title='Test 4: State and Control'
    )
    
    fig.write_html(output_dir / "04_state_and_control.html")
    print("  ✓ Saved: 04_state_and_control.html")


def test_5_comparison(output_dir):
    """Test 5: Trajectory comparison."""
    print("Generating Test 5: Trajectory comparison...")
    
    plotter = TrajectoryPlotter()
    t = np.linspace(0, 10, 200)
    
    # Controlled vs uncontrolled
    zeta_controlled = 0.7
    zeta_uncontrolled = 0.1
    omega = 2.0
    
    x_controlled = np.column_stack([
        np.exp(-zeta_controlled*omega*t) * np.sin(omega*t),
        -zeta_controlled*omega*np.exp(-zeta_controlled*omega*t)*np.sin(omega*t)
    ])
    
    x_uncontrolled = np.column_stack([
        np.exp(-zeta_uncontrolled*omega*t) * np.sin(omega*t),
        -zeta_uncontrolled*omega*np.exp(-zeta_uncontrolled*omega*t)*np.sin(omega*t)
    ])
    
    trajectories = {
        'Controlled (ζ=0.7)': x_controlled,
        'Uncontrolled (ζ=0.1)': x_uncontrolled,
    }
    
    fig = plotter.plot_comparison(
        t, trajectories,
        state_names=['Position', 'Velocity'],
        title='Test 5: Controlled vs Uncontrolled'
    )
    
    fig.write_html(output_dir / "05_comparison.html")
    print("  ✓ Saved: 05_comparison.html")


def test_6_themes_default(output_dir):
    """Test 6: Default theme."""
    print("Generating Test 6: Default theme...")
    
    plotter = TrajectoryPlotter()
    t = np.linspace(0, 10, 200)
    x = np.column_stack([np.sin(t), np.cos(t)])
    
    fig = plotter.plot_trajectory(
        t, x,
        state_names=['sin(t)', 'cos(t)'],
        title='Test 6: Default Theme',
        theme='default'
    )
    
    fig.write_html(output_dir / "06_theme_default.html")
    print("  ✓ Saved: 06_theme_default.html")


def test_7_themes_publication(output_dir):
    """Test 7: Publication theme."""
    print("Generating Test 7: Publication theme...")
    
    plotter = TrajectoryPlotter()
    t = np.linspace(0, 10, 200)
    x = np.column_stack([np.sin(t), np.cos(t)])
    
    fig = plotter.plot_trajectory(
        t, x,
        state_names=['sin(t)', 'cos(t)'],
        title='Test 7: Publication Theme',
        color_scheme='colorblind_safe',
        theme='publication'
    )
    
    fig.write_html(output_dir / "07_theme_publication.html")
    print("  ✓ Saved: 07_theme_publication.html")


def test_8_themes_dark(output_dir):
    """Test 8: Dark theme."""
    print("Generating Test 8: Dark theme...")
    
    plotter = TrajectoryPlotter()
    t = np.linspace(0, 10, 200)
    x = np.column_stack([np.sin(t), np.cos(t)])
    
    fig = plotter.plot_trajectory(
        t, x,
        state_names=['sin(t)', 'cos(t)'],
        title='Test 8: Dark Theme',
        theme='dark'
    )
    
    fig.write_html(output_dir / "08_theme_dark.html")
    print("  ✓ Saved: 08_theme_dark.html")


def test_9_themes_presentation(output_dir):
    """Test 9: Presentation theme."""
    print("Generating Test 9: Presentation theme...")
    
    plotter = TrajectoryPlotter()
    t = np.linspace(0, 10, 200)
    x = np.column_stack([np.sin(t), np.cos(t)])
    
    fig = plotter.plot_trajectory(
        t, x,
        state_names=['sin(t)', 'cos(t)'],
        title='Test 9: Presentation Theme',
        color_scheme='tableau',
        theme='presentation'
    )
    
    fig.write_html(output_dir / "09_theme_presentation.html")
    print("  ✓ Saved: 09_theme_presentation.html")


def test_10_color_schemes(output_dir):
    """Test 10: Different color schemes."""
    print("Generating Test 10: Color schemes comparison...")
    
    t = np.linspace(0, 10, 200)
    x_batch = np.stack([
        np.column_stack([np.sin(t + phi), np.cos(t + phi)])
        for phi in np.linspace(0, 1, 4)
    ])
    
    schemes = ['plotly', 'colorblind_safe', 'tableau', 'd3']
    
    for i, scheme in enumerate(schemes, start=1):
        plotter = TrajectoryPlotter()
        fig = plotter.plot_trajectory(
            t, x_batch,
            state_names=['sin', 'cos'],
            title=f'Test 10.{i}: Color Scheme "{scheme}"',
            color_scheme=scheme,
            show_legend=True
        )
        
        fig.write_html(output_dir / f"10_{i}_color_scheme_{scheme}.html")
        print(f"  ✓ Saved: 10_{i}_color_scheme_{scheme}.html")


def test_11_large_state_space(output_dir):
    """Test 11: Many state variables."""
    print("Generating Test 11: Large state space...")
    
    plotter = TrajectoryPlotter()
    t = np.linspace(0, 10, 200)
    
    # 8 state variables
    x = np.column_stack([
        np.sin(t),
        np.cos(t),
        np.sin(2*t),
        np.cos(2*t),
        np.sin(3*t),
        np.cos(3*t),
        np.sin(0.5*t),
        np.cos(0.5*t),
    ])
    
    state_names = [f'x_{i+1}' for i in range(8)]
    
    fig = plotter.plot_trajectory(
        t, x,
        state_names=state_names,
        title='Test 11: Large State Space (8 states)'
    )
    
    fig.write_html(output_dir / "11_large_state_space.html")
    print("  ✓ Saved: 11_large_state_space.html")


def test_12_noisy_data(output_dir):
    """Test 12: Noisy measurements."""
    print("Generating Test 12: Noisy data...")
    
    plotter = TrajectoryPlotter()
    t = np.linspace(0, 10, 200)
    
    # Clean signal
    x_clean = np.column_stack([np.sin(t), np.cos(t)])
    
    # Noisy signal
    noise = 0.1 * np.random.randn(*x_clean.shape)
    x_noisy = x_clean + noise
    
    trajectories = {
        'Clean Signal': x_clean,
        'Noisy Signal': x_noisy,
    }
    
    fig = plotter.plot_comparison(
        t, trajectories,
        state_names=['sin(t)', 'cos(t)'],
        title='Test 12: Clean vs Noisy Data',
        color_scheme='colorblind_safe'
    )
    
    fig.write_html(output_dir / "12_noisy_data.html")
    print("  ✓ Saved: 12_noisy_data.html")


def test_13_step_response(output_dir):
    """Test 13: Step response."""
    print("Generating Test 13: Step response...")
    
    plotter = TrajectoryPlotter()
    t = np.linspace(0, 10, 200)
    
    # First-order step response
    tau = 1.0
    x_step = 1 - np.exp(-t/tau)
    
    fig = plotter.plot_trajectory(
        t, x_step[:, None],
        state_names=['Output'],
        title='Test 13: Step Response (τ=1.0s)'
    )
    
    fig.write_html(output_dir / "13_step_response.html")
    print("  ✓ Saved: 13_step_response.html")


def test_14_phase_plane_data(output_dir):
    """Test 14: Phase plane trajectory."""
    print("Generating Test 14: Phase plane trajectory...")
    
    plotter = TrajectoryPlotter()
    
    # Van der Pol oscillator
    mu = 1.0
    t = np.linspace(0, 20, 500)
    
    # Simple simulation (Euler method)
    x1 = np.zeros_like(t)
    x2 = np.zeros_like(t)
    x1[0], x2[0] = 2.0, 0.0
    dt = t[1] - t[0]
    
    for i in range(len(t)-1):
        x1[i+1] = x1[i] + dt * x2[i]
        x2[i+1] = x2[i] + dt * (mu*(1-x1[i]**2)*x2[i] - x1[i])
    
    x = np.column_stack([x1, x2])
    
    fig = plotter.plot_trajectory(
        t, x,
        state_names=['Position', 'Velocity'],
        title='Test 14: Van der Pol Oscillator (μ=1.0)'
    )
    
    fig.write_html(output_dir / "14_phase_plane_data.html")
    print("  ✓ Saved: 14_phase_plane_data.html")


def generate_index_html(output_dir):
    """Generate index.html for easy navigation."""
    print("\nGenerating index.html...")
    
    html_files = sorted(output_dir.glob("*.html"))
    
    html_content = """<!DOCTYPE html>
<html>
<head>
    <title>Trajectory Plotter Visual Tests</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        h1 {
            color: #333;
            border-bottom: 3px solid #636EFA;
            padding-bottom: 10px;
        }
        .test-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 20px;
            margin-top: 20px;
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
            color: #636EFA;
        }
        .test-card a {
            display: inline-block;
            background: #636EFA;
            color: white;
            padding: 10px 20px;
            text-decoration: none;
            border-radius: 4px;
            margin-top: 10px;
        }
        .test-card a:hover {
            background: #4c5fd8;
        }
        .category {
            background: #e8f4f8;
            padding: 10px;
            border-radius: 4px;
            margin-bottom: 20px;
        }
        .category h2 {
            margin: 0;
            color: #0066cc;
        }
    </style>
</head>
<body>
    <h1>📊 Trajectory Plotter Visual Test Suite</h1>
    <p>Visual inspection gallery for trajectory plotting functionality.</p>
    <p><strong>Generated:</strong> """ + str(Path.cwd() / output_dir) + """</p>
    
    <div class="category">
        <h2>Basic Functionality</h2>
    </div>
    <div class="test-grid">
"""
    
    test_categories = {
        'Basic': list(range(1, 6)),
        'Themes': list(range(6, 10)),
        'Color Schemes': [10],
        'Advanced': list(range(11, 15)),
    }
    
    test_descriptions = {
        1: "Simple single-state trajectory",
        2: "Multi-state trajectory with 4 states",
        3: "Batched trajectories (Monte Carlo)",
        4: "State and control visualization",
        5: "Trajectory comparison",
        6: "Default theme",
        7: "Publication theme",
        8: "Dark theme",
        9: "Presentation theme",
        10: "Color scheme comparison (4 schemes)",
        11: "Large state space (8 states)",
        12: "Clean vs noisy data",
        13: "Step response",
        14: "Van der Pol oscillator",
    }
    
    for file in html_files:
        if file.name == "index.html":
            continue
            
        # Extract test number
        test_num = int(file.stem.split('_')[0])
        desc = test_descriptions.get(test_num, "Test case")
        
        html_content += f"""
        <div class="test-card">
            <h3>Test {file.stem.replace('_', ' ').title()}</h3>
            <p>{desc}</p>
            <a href="{file.name}" target="_blank">View Plot →</a>
        </div>
"""
    
    html_content += """
    </div>
</body>
</html>
"""
    
    index_path = output_dir / "index.html"
    index_path.write_text(html_content)
    print(f"  ✓ Saved: index.html")


def main():
    """Run all visual tests."""
    print("=" * 70)
    print("Trajectory Plotter Visual Test Suite")
    print("=" * 70)
    print()
    
    output_dir = setup_output_directory()
    print(f"Output directory: {output_dir.absolute()}\n")
    
    # Run all tests
    test_1_simple_trajectory(output_dir)
    test_2_multi_state_trajectory(output_dir)
    test_3_batched_trajectories(output_dir)
    test_4_state_and_control(output_dir)
    test_5_comparison(output_dir)
    test_6_themes_default(output_dir)
    test_7_themes_publication(output_dir)
    test_8_themes_dark(output_dir)
    test_9_themes_presentation(output_dir)
    test_10_color_schemes(output_dir)
    test_11_large_state_space(output_dir)
    test_12_noisy_data(output_dir)
    test_13_step_response(output_dir)
    test_14_phase_plane_data(output_dir)
    
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