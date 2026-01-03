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
Visual Test Suite for Control Plotter

Generates HTML files for visual inspection of control system plotting functionality.
Run this script to create a gallery of test plots.

Usage:
    python visual_test_control_plotter.py

Output:
    Creates HTML files in ./visual_tests/control_plotter/
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from pathlib import Path

import numpy as np
from scipy.linalg import solve_continuous_are

from src.visualization.control_plots import ControlPlotter


def setup_output_directory():
    """Create output directory for visual tests."""
    output_dir = Path("tests/visual_tests/control_plotter")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def test_1_eigenvalue_map_continuous_stable(output_dir):
    """Test 1: Continuous stable eigenvalues."""
    print("Generating Test 1: Continuous stable eigenvalues...")

    plotter = ControlPlotter()

    # LQR-like eigenvalues
    eigenvalues = np.array([-1.0 + 3.0j, -1.0 - 3.0j, -2.5 + 1.0j, -2.5 - 1.0j, -4.0, -5.0])

    fig = plotter.plot_eigenvalue_map(
        eigenvalues,
        system_type="continuous",
        title="Test 1: Continuous Stable System",
        show_stability_margin=True,
    )

    fig.write_html(output_dir / "01_eigenvalue_continuous_stable.html")
    print("  ✓ Saved: 01_eigenvalue_continuous_stable.html")


def test_2_eigenvalue_map_continuous_marginally_stable(output_dir):
    """Test 2: Continuous marginally stable eigenvalues."""
    print("Generating Test 2: Continuous marginally stable eigenvalues...")

    plotter = ControlPlotter()

    # Near imaginary axis
    eigenvalues = np.array([-0.1 + 4.0j, -0.1 - 4.0j, -0.5 + 2.0j, -0.5 - 2.0j, -1.0])

    fig = plotter.plot_eigenvalue_map(
        eigenvalues,
        system_type="continuous",
        title="Test 2: Marginally Stable (Small Stability Margin)",
        show_stability_margin=True,
        theme="publication",
    )

    fig.write_html(output_dir / "02_eigenvalue_continuous_marginal.html")
    print("  ✓ Saved: 02_eigenvalue_continuous_marginal.html")


def test_3_eigenvalue_map_discrete_stable(output_dir):
    """Test 3: Discrete stable eigenvalues."""
    print("Generating Test 3: Discrete stable eigenvalues...")

    plotter = ControlPlotter()

    # Inside unit circle
    eigenvalues = np.array([0.7 + 0.3j, 0.7 - 0.3j, 0.5 + 0.5j, 0.5 - 0.5j, 0.9, 0.6])

    fig = plotter.plot_eigenvalue_map(
        eigenvalues,
        system_type="discrete",
        title="Test 3: Discrete Stable System",
        show_stability_margin=True,
    )

    fig.write_html(output_dir / "03_eigenvalue_discrete_stable.html")
    print("  ✓ Saved: 03_eigenvalue_discrete_stable.html")


def test_4_gain_comparison_lqr(output_dir):
    """Test 4: LQR gain comparison."""
    print("Generating Test 4: LQR gain comparison...")

    plotter = ControlPlotter()

    # Simulate different Q weights
    gains = {
        "Q = 10·I": np.array([[1.58, 2.24]]),
        "Q = 100·I": np.array([[5.00, 7.07]]),
        "Q = 1000·I": np.array([[15.81, 22.36]]),
        "Q = 10000·I": np.array([[50.00, 70.71]]),
    }

    fig = plotter.plot_gain_comparison(
        gains,
        labels=["Position", "Velocity"],
        title="Test 4: LQR Feedback Gain vs Q Weight",
        color_scheme="colorblind_safe",
        theme="publication",
    )

    fig.write_html(output_dir / "04_gain_comparison_lqr.html")
    print("  ✓ Saved: 04_gain_comparison_lqr.html")


def test_5_gain_comparison_mimo(output_dir):
    """Test 5: MIMO gain comparison."""
    print("Generating Test 5: MIMO gain comparison...")

    plotter = ControlPlotter()

    # Multi-input multi-output gains
    gains = {
        "Conservative": np.array([[1.0, 2.0, 3.0], [0.5, 1.0, 1.5]]),
        "Moderate": np.array([[2.0, 4.0, 6.0], [1.0, 2.0, 3.0]]),
        "Aggressive": np.array([[4.0, 8.0, 12.0], [2.0, 4.0, 6.0]]),
    }

    fig = plotter.plot_gain_comparison(
        gains, labels=["x₁", "x₂", "x₃"], title="Test 5: MIMO Feedback Gains", theme="default",
    )

    fig.write_html(output_dir / "05_gain_comparison_mimo.html")
    print("  ✓ Saved: 05_gain_comparison_mimo.html")


def test_6_riccati_convergence(output_dir):
    """Test 6: Riccati equation convergence."""
    print("Generating Test 6: Riccati convergence...")

    plotter = ControlPlotter()

    # Simulate iterative Riccati solver
    A = np.array([[0, 1], [-2, -3]])
    B = np.array([[0], [1]])
    Q = np.eye(2)
    R = np.array([[1]])

    # True solution
    P_final = solve_continuous_are(A, B, Q, R)

    # Simulate convergence
    P_history = []
    P = np.eye(2) * 10  # Initial guess
    for i in range(50):
        # Simple iteration (not actual Riccati iteration, just for demo)
        alpha = 1 - np.exp(-i / 5)
        P = P_final * alpha + P * (1 - alpha)
        P_history.append(P.copy())

    fig = plotter.plot_riccati_convergence(
        P_history, title="Test 6: Riccati Equation Convergence", theme="publication",
    )

    fig.write_html(output_dir / "06_riccati_convergence.html")
    print("  ✓ Saved: 06_riccati_convergence.html")


def test_7_controllability_gramian(output_dir):
    """Test 7: Controllability Gramian."""
    print("Generating Test 7: Controllability Gramian...")

    plotter = ControlPlotter()

    # Generate a controllability Gramian
    # For demo: symmetric positive definite matrix
    np.random.seed(42)
    A = np.random.randn(4, 4)
    W_c = A @ A.T + np.eye(4) * 0.5

    fig = plotter.plot_controllability_gramian(
        W_c,
        state_names=["x₁", "x₂", "x₃", "x₄"],
        title="Test 7: Controllability Gramian",
        theme="default",
    )

    fig.write_html(output_dir / "07_controllability_gramian.html")
    print("  ✓ Saved: 07_controllability_gramian.html")


def test_8_observability_gramian(output_dir):
    """Test 8: Observability Gramian."""
    print("Generating Test 8: Observability Gramian...")

    plotter = ControlPlotter()

    # Generate an observability Gramian
    np.random.seed(43)
    A = np.random.randn(3, 3)
    W_o = A @ A.T + np.eye(3) * 1.0

    fig = plotter.plot_observability_gramian(
        W_o,
        state_names=["Position", "Velocity", "Acceleration"],
        title="Test 8: Observability Gramian",
        theme="dark",
    )

    fig.write_html(output_dir / "08_observability_gramian.html")
    print("  ✓ Saved: 08_observability_gramian.html")


def test_9_step_response_underdamped(output_dir):
    """Test 9: Underdamped step response."""
    print("Generating Test 9: Underdamped step response...")

    plotter = ControlPlotter()

    # Second-order underdamped system
    t = np.linspace(0, 10, 500)
    zeta = 0.3  # Underdamped
    omega_n = 2.0
    omega_d = omega_n * np.sqrt(1 - zeta**2)

    y = 1 - np.exp(-zeta * omega_n * t) * (
        np.cos(omega_d * t) + (zeta / np.sqrt(1 - zeta**2)) * np.sin(omega_d * t)
    )

    fig = plotter.plot_step_response(
        t,
        y,
        reference=1.0,
        show_metrics=True,
        title="Test 9: Underdamped Step Response (ζ=0.3)",
        theme="publication",
    )

    fig.write_html(output_dir / "09_step_response_underdamped.html")
    print("  ✓ Saved: 09_step_response_underdamped.html")


def test_10_step_response_critically_damped(output_dir):
    """Test 10: Critically damped step response."""
    print("Generating Test 10: Critically damped step response...")

    plotter = ControlPlotter()

    # Second-order critically damped system
    t = np.linspace(0, 10, 500)
    omega_n = 2.0

    y = 1 - (1 + omega_n * t) * np.exp(-omega_n * t)

    fig = plotter.plot_step_response(
        t,
        y,
        reference=1.0,
        show_metrics=True,
        title="Test 10: Critically Damped Step Response (ζ=1.0)",
        theme="default",
    )

    fig.write_html(output_dir / "10_step_response_critically_damped.html")
    print("  ✓ Saved: 10_step_response_critically_damped.html")


def test_11_impulse_response(output_dir):
    """Test 11: Impulse response."""
    print("Generating Test 11: Impulse response...")

    plotter = ControlPlotter()

    # Damped oscillator impulse response
    t = np.linspace(0, 10, 500)
    zeta = 0.1
    omega_n = 2.0

    y = (
        (omega_n / np.sqrt(1 - zeta**2))
        * np.exp(-zeta * omega_n * t)
        * np.sin(omega_n * np.sqrt(1 - zeta**2) * t)
    )

    fig = plotter.plot_impulse_response(
        t, y, show_metrics=True, title="Test 11: Impulse Response (ζ=0.1)", theme="publication",
    )

    fig.write_html(output_dir / "11_impulse_response.html")
    print("  ✓ Saved: 11_impulse_response.html")


def test_12_frequency_response_first_order(output_dir):
    """Test 12: First-order frequency response."""
    print("Generating Test 12: First-order frequency response...")

    plotter = ControlPlotter()

    # First-order system: G(s) = 1/(s+1)
    w = np.logspace(-2, 2, 200)
    H = 1 / (1j * w + 1)

    mag_dB = 20 * np.log10(np.abs(H))
    phase_deg = np.angle(H, deg=True)

    fig = plotter.plot_frequency_response(
        w,
        mag_dB,
        phase_deg,
        title="Test 12: First-Order System Bode Plot",
        show_margins=False,
        theme="publication",
    )

    fig.write_html(output_dir / "12_frequency_response_first_order.html")
    print("  ✓ Saved: 12_frequency_response_first_order.html")


def test_13_frequency_response_second_order(output_dir):
    """Test 13: Second-order frequency response."""
    print("Generating Test 13: Second-order frequency response...")

    plotter = ControlPlotter()

    # Second-order system with resonance
    w = np.logspace(-1, 2, 200)
    zeta = 0.2
    omega_n = 5.0

    H = omega_n**2 / (omega_n**2 - w**2 + 2j * zeta * omega_n * w)

    mag_dB = 20 * np.log10(np.abs(H))
    phase_deg = np.angle(H, deg=True)

    fig = plotter.plot_frequency_response(
        w,
        mag_dB,
        phase_deg,
        title="Test 13: Second-Order System with Resonance (ζ=0.2)",
        show_margins=True,
        theme="default",
    )

    fig.write_html(output_dir / "13_frequency_response_second_order.html")
    print("  ✓ Saved: 13_frequency_response_second_order.html")


def test_14_nyquist_stable(output_dir):
    """Test 14: Nyquist plot - stable system."""
    print("Generating Test 14: Nyquist plot (stable)...")

    plotter = ControlPlotter()

    # Stable system: G(s) = 1/(s+1)^2
    w = np.logspace(-2, 2, 300)
    H = 1 / (1j * w + 1) ** 2

    fig = plotter.plot_nyquist(
        np.real(H),
        np.imag(H),
        frequencies=w,
        title="Test 14: Nyquist Diagram (Stable System)",
        show_critical_point=True,
        theme="publication",
    )

    fig.write_html(output_dir / "14_nyquist_stable.html")
    print("  ✓ Saved: 14_nyquist_stable.html")


def test_15_nyquist_marginally_stable(output_dir):
    """Test 15: Nyquist plot - marginally stable."""
    print("Generating Test 15: Nyquist plot (marginally stable)...")

    plotter = ControlPlotter()

    # System that passes near critical point
    w = np.logspace(-2, 2, 300)
    H = 2 / ((1j * w + 1) * (1j * w + 0.5))

    fig = plotter.plot_nyquist(
        np.real(H),
        np.imag(H),
        frequencies=w,
        title="Test 15: Nyquist Diagram (Marginally Stable)",
        show_critical_point=True,
        theme="dark",
    )

    fig.write_html(output_dir / "15_nyquist_marginal.html")
    print("  ✓ Saved: 15_nyquist_marginal.html")


def test_16_root_locus_continuous(output_dir):
    """Test 16: Root locus for continuous system."""
    print("Generating Test 16: Root locus (continuous)...")

    plotter = ControlPlotter()

    # Simple root locus: poles migrate as gain varies
    gains = np.logspace(-1, 2, 50)

    # Simulate pole movement
    poles = []
    for k in gains:
        # Two poles moving toward each other then splitting
        if k < 1:
            p = np.array([-2 + k, -4 - k])
        else:
            spread = np.sqrt(k - 1)
            p = np.array([-3 + 1j * spread, -3 - 1j * spread])
        poles.append(p)

    root_locus_data = {"gains": gains, "poles": np.array(poles), "zeros": np.array([-1.0])}

    fig = plotter.plot_root_locus(
        root_locus_data,
        title="Test 16: Root Locus (Continuous System)",
        system_type="continuous",
        color_scheme="colorblind_safe",
        theme="publication",
    )

    fig.write_html(output_dir / "16_root_locus_continuous.html")
    print("  ✓ Saved: 16_root_locus_continuous.html")


def test_17_root_locus_discrete(output_dir):
    """Test 17: Root locus for discrete system."""
    print("Generating Test 17: Root locus (discrete)...")

    plotter = ControlPlotter()

    # Discrete system root locus
    gains = np.linspace(0, 2, 40)

    # Poles moving inside unit circle
    poles = []
    for k in gains:
        angle = np.pi / 4 * (1 - k / 2)
        radius = 0.9 * (1 - k / 3)
        p = np.array([radius * np.exp(1j * angle), radius * np.exp(-1j * angle), 0.7 * (1 - k / 4)])
        poles.append(p)

    root_locus_data = {"gains": gains, "poles": np.array(poles)}

    fig = plotter.plot_root_locus(
        root_locus_data,
        title="Test 17: Root Locus (Discrete System)",
        system_type="discrete",
        color_scheme="tableau",
        theme="default",
    )

    fig.write_html(output_dir / "17_root_locus_discrete.html")
    print("  ✓ Saved: 17_root_locus_discrete.html")


def test_18_theme_comparison(output_dir):
    """Test 18: Theme comparison on eigenvalue map."""
    print("Generating Test 18: Theme comparison...")

    eigenvalues = np.array([-1.0 + 2.0j, -1.0 - 2.0j, -2.0 + 1.0j, -2.0 - 1.0j, -3.0])

    themes = ["default", "publication", "dark", "presentation"]

    for i, theme in enumerate(themes, start=1):
        plotter = ControlPlotter(default_theme=theme)

        fig = plotter.plot_eigenvalue_map(
            eigenvalues,
            system_type="continuous",
            title=f"Test 18.{i}: Eigenvalue Map - {theme.title()} Theme",
        )

        fig.write_html(output_dir / f"18_{i}_theme_{theme}.html")
        print(f"  ✓ Saved: 18_{i}_theme_{theme}.html")


def test_19_color_scheme_comparison(output_dir):
    """Test 19: Color scheme comparison."""
    print("Generating Test 19: Color scheme comparison...")

    gains = {
        "Design A": np.array([[1.0, 2.0]]),
        "Design B": np.array([[2.0, 3.0]]),
        "Design C": np.array([[3.0, 4.0]]),
        "Design D": np.array([[4.0, 5.0]]),
    }

    schemes = ["plotly", "colorblind_safe", "tableau", "d3"]

    for i, scheme in enumerate(schemes, start=1):
        plotter = ControlPlotter()

        fig = plotter.plot_gain_comparison(
            gains,
            labels=["x₁", "x₂"],
            title=f'Test 19.{i}: Color Scheme "{scheme}"',
            color_scheme=scheme,
            theme="default",
        )

        fig.write_html(output_dir / f"19_{i}_colors_{scheme}.html")
        print(f"  ✓ Saved: 19_{i}_colors_{scheme}.html")


def test_20_overdamped_step_response(output_dir):
    """Test 20: Overdamped step response."""
    print("Generating Test 20: Overdamped step response...")

    plotter = ControlPlotter()

    # Overdamped system (two real poles)
    t = np.linspace(0, 10, 500)
    tau1, tau2 = 1.0, 2.0

    y = 1 - (tau2 / (tau2 - tau1)) * np.exp(-t / tau1) + (tau1 / (tau2 - tau1)) * np.exp(-t / tau2)

    fig = plotter.plot_step_response(
        t,
        y,
        reference=1.0,
        show_metrics=True,
        title="Test 20: Overdamped Step Response (ζ>1)",
        theme="presentation",
    )

    fig.write_html(output_dir / "20_step_response_overdamped.html")
    print("  ✓ Saved: 20_step_response_overdamped.html")


def generate_index_html(output_dir):
    """Generate index.html for easy navigation."""
    print("\nGenerating index.html...")

    html_files = sorted(output_dir.glob("*.html"))

    html_content = (
        """<!DOCTYPE html>
<html>
<head>
    <title>Control Plotter Visual Tests</title>
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
            border-bottom: 3px solid #636EFA;
            padding-bottom: 10px;
        }
        .category {
            background: #e8f4f8;
            padding: 15px;
            border-radius: 8px;
            margin: 20px 0;
        }
        .category h2 {
            margin: 0 0 10px 0;
            color: #0066cc;
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
            color: #636EFA;
            font-size: 1.1em;
        }
        .test-card p {
            color: #666;
            font-size: 0.95em;
            margin: 10px 0;
        }
        .test-card a {
            display: inline-block;
            background: #636EFA;
            color: white;
            padding: 10px 20px;
            text-decoration: none;
            border-radius: 4px;
            margin-top: 10px;
            font-size: 0.9em;
        }
        .test-card a:hover {
            background: #4c5fd8;
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
            color: #636EFA;
        }
        .stat-label {
            color: #666;
            font-size: 0.9em;
        }
    </style>
</head>
<body>
    <h1>🎛️ Control Plotter Visual Test Suite</h1>
    <p>Visual inspection gallery for control system plotting functionality.</p>
    <p><strong>Generated:</strong> """
        + str(Path.cwd() / output_dir)
        + """</p>
    
    <div class="stats">
        <div class="stat-item">
            <div class="stat-number">24</div>
            <div class="stat-label">Test Cases</div>
        </div>
        <div class="stat-item">
            <div class="stat-number">10</div>
            <div class="stat-label">Plot Types</div>
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
    )

    categories = {
        "Eigenvalue Maps": [
            (1, "Continuous stable system"),
            (2, "Continuous marginally stable"),
            (3, "Discrete stable system"),
        ],
        "Gain Comparison": [
            (4, "LQR gain vs Q weight"),
            (5, "MIMO feedback gains"),
        ],
        "Convergence & Gramians": [
            (6, "Riccati equation convergence"),
            (7, "Controllability Gramian"),
            (8, "Observability Gramian"),
        ],
        "Step Responses": [
            (9, "Underdamped (ζ=0.3)"),
            (10, "Critically damped (ζ=1.0)"),
            (20, "Overdamped (ζ>1)"),
        ],
        "Impulse Response": [
            (11, "Damped oscillator"),
        ],
        "Frequency Response (Bode)": [
            (12, "First-order system"),
            (13, "Second-order with resonance"),
        ],
        "Nyquist Diagrams": [
            (14, "Stable system"),
            (15, "Marginally stable system"),
        ],
        "Root Locus": [
            (16, "Continuous system"),
            (17, "Discrete system"),
        ],
        "Themes & Colors": [
            (18, "Theme comparison (4 themes)"),
            (19, "Color scheme comparison (4 schemes)"),
        ],
    }

    for category, tests in categories.items():
        html_content += f"""
    <div class="category">
        <h2>{category}</h2>
        <div class="test-grid">
"""

        for test_num, desc in tests:
            # Find matching files
            if test_num in [18, 19]:
                # Multiple files for these tests
                pattern = f"{test_num:02d}_*"
                matching_files = sorted(output_dir.glob(f"{pattern}.html"))
                for file in matching_files:
                    if file.name == "index.html":
                        continue
                    file_desc = file.stem.replace("_", " ").title()
                    html_content += f"""
            <div class="test-card">
                <h3>{file_desc}</h3>
                <p>{desc}</p>
                <a href="{file.name}" target="_blank">View Plot →</a>
            </div>
"""
            else:
                # Single file
                pattern = f"{test_num:02d}_*.html"
                matching_files = list(output_dir.glob(pattern))
                if matching_files:
                    file = matching_files[0]
                    html_content += f"""
            <div class="test-card">
                <h3>Test {test_num}</h3>
                <p>{desc}</p>
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
    print("  ✓ Saved: index.html")


def main():
    """Run all visual tests."""
    print("=" * 70)
    print("Control Plotter Visual Test Suite")
    print("=" * 70)
    print()

    output_dir = setup_output_directory()
    print(f"Output directory: {output_dir.absolute()}\n")

    # Run all tests
    test_1_eigenvalue_map_continuous_stable(output_dir)
    test_2_eigenvalue_map_continuous_marginally_stable(output_dir)
    test_3_eigenvalue_map_discrete_stable(output_dir)
    test_4_gain_comparison_lqr(output_dir)
    test_5_gain_comparison_mimo(output_dir)
    test_6_riccati_convergence(output_dir)
    test_7_controllability_gramian(output_dir)
    test_8_observability_gramian(output_dir)
    test_9_step_response_underdamped(output_dir)
    test_10_step_response_critically_damped(output_dir)
    test_11_impulse_response(output_dir)
    test_12_frequency_response_first_order(output_dir)
    test_13_frequency_response_second_order(output_dir)
    test_14_nyquist_stable(output_dir)
    test_15_nyquist_marginally_stable(output_dir)
    test_16_root_locus_continuous(output_dir)
    test_17_root_locus_discrete(output_dir)
    test_18_theme_comparison(output_dir)
    test_19_color_scheme_comparison(output_dir)
    test_20_overdamped_step_response(output_dir)

    # Generate index
    generate_index_html(output_dir)

    print("\n" + "=" * 70)
    print("✓ All visual tests generated successfully!")
    print("=" * 70)
    print("\nOpen this file in your browser:")
    print(f"  {(output_dir / 'index.html').absolute()}")
    print()


if __name__ == "__main__":
    main()
