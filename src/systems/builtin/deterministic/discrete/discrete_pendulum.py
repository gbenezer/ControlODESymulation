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
Discrete-time Simple Pendulum - Classic Nonlinear Dynamics.

This module provides discrete-time models of the simple pendulum, one of the
most fundamental and well-studied systems in physics and engineering. It serves as:
- The canonical example of nonlinear oscillatory dynamics
- A model for understanding phase space structure and separatrices
- A benchmark for energy-based control methods
- An illustration of small-angle linearization and its limits
- A testbed for nonlinear observers and estimators

The simple pendulum represents:
- Clock mechanisms and timekeeping devices
- Playground swings and their dynamics
- Seismometer suspensions
- Ship stabilization systems (inverted pendulum)
- Satellite attitude dynamics (gravity gradient stabilization)
- Any rigid body rotating under gravity

This is the discrete-time version, appropriate for:
- Digital control systems with periodic updates
- Sampled-data implementations
- Educational demonstrations with real hardware
- Comparison with continuous-time theory
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import sympy as sp
from scipy.integrate import odeint

from src.systems.base.core.discrete_symbolic_system import DiscreteSymbolicSystem


class DiscretePendulum(DiscreteSymbolicSystem):
    """
    Discrete-time simple pendulum with friction and optional control.

    Physical System:
    ---------------
    A point mass m suspended by a massless, rigid rod of length L, swinging
    in a vertical plane under the influence of gravity. The pendulum can be:
    - Free (no control): Natural oscillatory motion
    - Forced (with control): External torque applied at pivot

    **Mechanical Configuration:**
```
              ● Pivot (fixed)
              |
              | L (rod length)
              |
              ● m (point mass)
              ↓
              g (gravity)
```

    Angle θ measured from downward vertical:
    - θ = 0: Hanging down (stable equilibrium)
    - θ = π: Standing up (unstable equilibrium)
    - θ = π/2: Horizontal position (maximum potential energy during swing)

    **Equation of Motion (from Newton's Second Law):**
    The continuous-time dynamics are:

        m·L²·θ̈ = -m·g·L·sin(θ) - b·θ̇ + τ

    Dividing by m·L²:

        θ̈ = -(g/L)·sin(θ) - (b/m·L²)·θ̇ + τ/(m·L²)

    Define:
        ω₀² = g/L: Natural frequency squared [rad²/s²]
        β = b/(m·L²): Damping coefficient [1/s]
        u = τ/(m·L²): Normalized torque [rad/s²]

    Then:
        θ̈ = -ω₀²·sin(θ) - β·θ̇ + u

    **Discrete-Time Dynamics:**
    Multiple discretization methods available (see parameters).

    State Space:
    -----------
    State: x[k] = [θ[k], ω[k]]
        Angular position:
        - θ: Angle from vertical [rad]
          * θ = 0: Downward equilibrium (lowest energy)
          * θ = π: Upward equilibrium (highest energy)
          * θ = ±π/2: Horizontal (maximum torque)
          * Periodic: sin(θ) and cos(θ) are 2π-periodic
          * Can be wrapped to [-π, π] or left unwrapped

        Angular velocity:
        - ω: Rate of change of angle [rad/s]
          * ω > 0: Counterclockwise rotation
          * ω < 0: Clockwise rotation
          * ω = 0: Instantaneous rest
          * In phase space: forms closed curves (periodic) or rotations

    Control: u[k] = [τ[k]]
        Applied torque (optional):
        - τ: External torque at pivot [N·m]
          * Normalized: u = τ/(m·L²) [rad/s²]
          * τ > 0: Pushes pendulum counterclockwise
          * τ < 0: Pushes pendulum clockwise
          * For swing-up or stabilization control

    Output: y[k] = [θ[k]] or [θ[k], ω[k]]
        - Position-only measurement (typical)
        - Full state if velocity sensor available
        - In practice: encoder for θ, gyroscope for ω

    Dynamics (Physical Regimes):
    ----------------------------
    **Small Angle Approximation (|θ| << 1):**
    For small displacements, sin(θ) ≈ θ:
        θ̈ ≈ -ω₀²·θ - β·θ̇

    This is a LINEAR harmonic oscillator! (See DiscreteOscillator)
    - Valid for θ < 0.2 rad (~10°)
    - Error: O(θ³)

    **Nonlinear Regime (arbitrary θ):**
    Must use full sin(θ) term:
    - Period depends on amplitude (not constant like linear!)
    - Large swings take longer
    - Complete rotation possible if sufficient energy

    **Phase Space Structure:**
    The (θ, ω) phase portrait has rich structure:

    1. **Fixed points:**
       - (0, 0): Stable focus/center (depends on damping)
       - (±π, 0): Saddle points (unstable)

    2. **Periodic orbits (β = 0):**
       - Closed curves around (0, 0)
       - Period increases with amplitude

    3. **Separatrix (β = 0):**
       - Special trajectory through saddle points
       - Separates oscillations from rotations
       - Homoclinic orbit (starts and ends at saddle)

    4. **Rotations (high energy):**
       - Open curves (pendulum goes over the top)
       - ω doesn't change sign
       - Continuous rotation in one direction

    Parameters:
    ----------
    m : float, default=1.0
        Mass of the pendulum bob [kg]
        - Affects inertia (m·L²)
        - Typical: 0.1-10 kg

    L : float, default=1.0
        Length of the pendulum rod [m]
        - Determines natural frequency: ω₀ = √(g/L)
        - Longer pendulum → slower oscillations
        - Typical: 0.1-2.0 m

    g : float, default=9.81
        Gravitational acceleration [m/s²]
        - Earth: 9.81
        - Moon: 1.62
        - Mars: 3.71

    b : float, default=0.1
        Damping coefficient [N·m·s/rad]
        - Air resistance + friction at pivot
        - Larger b → faster decay
        - b = 0: Undamped (conservative, energy-preserving)
        - Typical: 0.01-1.0

    dt : float, default=0.01
        Sampling period [s]
        - Digital control update rate
        - Must satisfy Nyquist criterion
        - Smaller dt → better accuracy
        - Typical: 0.001-0.1 s

    method : str, default='zoh'
        Discretization method:
        - 'zoh': Zero-order hold (exact for constant τ)
        - 'euler': Forward Euler (simple, O(dt))
        - 'rk4': Runge-Kutta 4th order (accurate, O(dt⁴))
        - 'exact': Exact discretization for linear approximation

    use_control : bool, default=True
        If True, includes control input τ
        If False, free pendulum (autonomous)

    Equilibria:
    ----------
    **Downward Equilibrium (θ = 0, ω = 0):**
    Stability: STABLE (center if undamped, stable focus if damped)
    - Minimum potential energy
    - Small perturbations oscillate (if β > 0: with decay)
    - Linearization: θ̈ = -ω₀²·θ - β·θ̇ (harmonic oscillator)

    **Upward Equilibrium (θ = ±π, ω = 0):**
    Stability: UNSTABLE (saddle point)
    - Maximum potential energy
    - Small perturbations grow exponentially
    - Linearization: θ̈ = +ω₀²·(θ - π) - β·θ̇ (inverted)
    - Famous "inverted pendulum" control problem

    **Separatrix Energy (undamped):**
    The energy separating oscillations from rotations:
        E_sep = 2·m·g·L

    If E < E_sep: Oscillation (back and forth)
    If E > E_sep: Rotation (goes over the top)
    If E = E_sep: Asymptotic approach to upward position

    Controllability:
    ---------------
    **With Control (u ≠ 0):**
    Completely controllable - can reach any state from any other state.
    
    **Without Control (Free Pendulum):**
    NOT controllable - energy determines accessible states.
    Can only reach states with same or lower energy (due to damping).

    Observability:
    -------------
    **Position-only measurement (y = θ):**
    Observable almost everywhere.
    - Can reconstruct ω from θ measurements over time
    - Singularity at equilibria (θ = constant → ω ambiguous)

    **Full state measurement:**
    Trivially observable.

    Energy and Integrals of Motion:
    -------------------------------
    **Total Mechanical Energy:**
        E = (1/2)·m·L²·ω² + m·g·L·(1 - cos(θ))

    Kinetic: K = (1/2)·m·L²·ω²
    Potential: V = m·g·L·(1 - cos(θ))

    **For Undamped, Unforced Pendulum (β = 0, u = 0):**
    Energy is conserved: dE/dt = 0
    Phase space trajectories are level sets of E(θ, ω).

    **For Damped Pendulum (β > 0, u = 0):**
    Energy decreases: dE/dt = -b·ω² ≤ 0
    All trajectories asymptotically approach (0, 0).

    **For Forced Pendulum (u ≠ 0):**
    Energy can increase or decrease:
        dE/dt = (m·L²·u - b)·ω²

    Control Objectives:
    ------------------
    **1. Stabilization at Downward Position:**
       Goal: θ → 0, ω → 0
       Method: PD control works well (linear regime)
       Challenge: Minimal (natural equilibrium)

    **2. Swing-Up Control:**
       Goal: Move from θ = 0 to θ = π
       Methods:
       - Energy-based: Pump energy until E ≈ E_target
       - Bang-bang: Apply maximum torque
       - Trajectory optimization
       Challenge: Large control effort, multiple rotations

    **3. Inverted Stabilization:**
       Goal: Stabilize at θ = π, ω = 0
       Method: LQR around linearization
       Challenge: Unstable equilibrium, requires continuous control

    **4. Trajectory Tracking:**
       Goal: Follow θ_ref(t), ω_ref(t)
       Method: Feedforward + feedback
       Application: Robotic manipulation

    **5. Limit Cycle Creation:**
       Goal: Create stable periodic oscillation
       Method: Nonlinear feedback
       Application: Rhythmic motion generation

    State Constraints:
    -----------------
    **1. Angle Limits (if physical stops exist):**
       θ_min ≤ θ ≤ θ_max
       Otherwise: θ ∈ ℝ (unbounded rotations possible)

    **2. Velocity Limits:**
       |ω| ≤ ω_max
       Physical: Limited by energy or mechanism

    **3. Torque Limits:**
       |τ| ≤ τ_max
       Most critical practical constraint
       Determines controllability

    Numerical Considerations:
    ------------------------
    **Discretization Accuracy:**
    - ZOH: Good for control applications
    - Euler: Simple but may be inaccurate/unstable for large dt
    - RK4: High accuracy, recommended for simulation

    **Angle Wrapping:**
    For visualization and analysis:
    - Wrap θ to [-π, π]: θ_wrapped = atan2(sin(θ), cos(θ))
    - Or leave unwrapped to count rotations

    **Energy Verification:**
    For undamped case, check energy conservation:
        |E[k+1] - E[k]| should be small (numerical precision)

    Example Usage:
    -------------
    >>> # Create pendulum with realistic parameters
    >>> system = DiscretePendulum(
    ...     m=0.5,      # 500g mass
    ...     L=1.0,      # 1m length
    ...     g=9.81,
    ...     b=0.05,     # Light damping
    ...     dt=0.01,
    ...     method='rk4'
    ... )
    >>> 
    >>> print(f"Natural frequency: {system.natural_frequency:.3f} rad/s")
    >>> print(f"Period: {system.period:.3f} s")
    >>> print(f"Damping ratio: {system.damping_ratio:.4f}")
    >>> 
    >>> # Free oscillation from initial displacement
    >>> x0 = np.array([np.pi/4, 0.0])  # 45° release from rest
    >>> result_free = system.simulate(
    ...     x0=x0,
    ...     u_sequence=None,
    ...     n_steps=1000
    ... )
    >>> 
    >>> # Plot phase portrait
    >>> import plotly.graph_objects as go
    >>> fig = go.Figure()
    >>> fig.add_trace(go.Scatter(
    ...     x=result_free['states'][:, 0],
    ...     y=result_free['states'][:, 1],
    ...     mode='lines',
    ...     name='Trajectory'
    ... ))
    >>> fig.update_layout(
    ...     title='Pendulum Phase Portrait',
    ...     xaxis_title='θ [rad]',
    ...     yaxis_title='ω [rad/s]'
    ... )
    >>> fig.show()
    >>> 
    >>> # Energy analysis
    >>> energies = np.array([
    ...     system.compute_total_energy(x[0], x[1])
    ...     for x in result_free['states']
    ... ])
    >>> 
    >>> fig_energy = go.Figure()
    >>> fig_energy.add_trace(go.Scatter(
    ...     x=result_free['time_steps'] * system.dt,
    ...     y=energies,
    ...     name='Total Energy'
    ... ))
    >>> fig_energy.update_layout(
    ...     title='Energy vs Time (should decrease with damping)',
    ...     xaxis_title='Time [s]',
    ...     yaxis_title='Energy [J]'
    ... )
    >>> fig_energy.show()
    >>> 
    >>> # Swing-up control (energy-based)
    >>> def swing_up_energy_control(x, k):
    ...     theta, omega = x
    ...     
    ...     # Current energy
    ...     E = system.compute_total_energy(theta, omega)
    ...     
    ...     # Target energy (upward position)
    ...     E_target = system.m * system.g * system.L * 2
    ...     
    ...     # Energy error
    ...     E_error = E - E_target
    ...     
    ...     # If close to upward, switch to stabilization
    ...     if abs(theta - np.pi) < 0.3 and abs(omega) < 1.0:
    ...         # LQR around upward
    ...         K_up = system.design_upward_stabilizer()
    ...         return -K_up @ np.array([theta - np.pi, omega])
    ...     else:
    ...         # Energy pumping: add energy when moving in right direction
    ...         k_swing = 5.0
    ...         return k_swing * E_error * np.sign(omega * np.cos(theta))
    >>> 
    >>> result_swing = system.rollout(
    ...     x0=np.array([0.0, 0.0]),
    ...     policy=swing_up_energy_control,
    ...     n_steps=2000
    ... )
    >>> 
    >>> # Visualize swing-up in phase space
    >>> fig_swing = system.plot_phase_portrait_with_separatrix()
    >>> fig_swing.add_trace(go.Scatter(
    ...     x=result_swing['states'][:, 0],
    ...     y=result_swing['states'][:, 1],
    ...     mode='lines',
    ...     line=dict(color='red', width=2),
    ...     name='Swing-up trajectory'
    ... ))
    >>> fig_swing.show()
    >>> 
    >>> # Compare small vs large angle dynamics
    >>> # Small angle (linear regime)
    >>> x0_small = np.array([0.1, 0.0])  # ~6°
    >>> result_small = system.simulate(x0_small, None, n_steps=500)
    >>> 
    >>> # Large angle (nonlinear regime)
    >>> x0_large = np.array([2.0, 0.0])  # ~115°
    >>> result_large = system.simulate(x0_large, None, n_steps=500)
    >>> 
    >>> # Compare periods
    >>> def find_period(states, dt):
    ...     # Find zero crossings
    ...     theta = states[:, 0]
    ...     crossings = np.where(np.diff(np.sign(theta)))[0]
    ...     if len(crossings) >= 2:
    ...         period = 2 * (crossings[1] - crossings[0]) * dt
    ...         return period
    ...     return None
    >>> 
    >>> period_small = find_period(result_small['states'], system.dt)
    >>> period_large = find_period(result_large['states'], system.dt)
    >>> 
    >>> print(f"Small angle period: {period_small:.3f} s")
    >>> print(f"Large angle period: {period_large:.3f} s")
    >>> print(f"Linear theory: {system.period:.3f} s")
    >>> print("Note: Large angle period > small angle period (nonlinearity)")

    Physical Insights:
    -----------------
    **Isochronism (Small Angles Only):**
    For small oscillations, period is independent of amplitude:
        T = 2π√(L/g) = 2π/ω₀

    This is why pendulum clocks work! (But only for small swings)

    **Anharmonicity (Large Angles):**
    For large amplitudes, period increases with amplitude:
        T(θ₀) ≈ T₀·(1 + θ₀²/16 + ...) for initial angle θ₀

    This breaks isochronism → pendulum clocks must limit amplitude.

    **Separatrix Dynamics:**
    Trajectories on the separatrix (E = E_sep) take infinite time to reach
    the unstable equilibrium. This is a homoclinic orbit.

    **Conservation vs Dissipation:**
    - No damping: Phase space filled with nested closed curves
    - With damping: Spiral inward to stable equilibrium
    - Energy landscape funnels trajectories toward rest

    **Chaotic Forcing:**
    Add periodic forcing: θ̈ = -ω₀²·sin(θ) + A·cos(Ω·t)
    Can produce chaos (sensitive dependence on initial conditions)!
    This is the "driven damped pendulum" - route to chaos.

    Common Pitfalls:
    ---------------
    1. **Using linear approximation for large angles:**
       sin(θ) ≈ θ only valid for |θ| < 0.2 rad
       Large errors for θ > π/4

    2. **Forgetting angle periodicity:**
       θ = 0 and θ = 2π are the same state
       Important for phase portraits

    3. **Ignoring separatrix:**
       Qualitative dynamics change across separatrix
       Control strategies differ for oscillations vs rotations

    4. **Wrong linearization for upward:**
       Linearizing at θ = π gives θ̈ = +ω₀²·(θ - π)
       Note: POSITIVE coefficient (unstable)

    5. **Energy not conserved numerically:**
       Discretization introduces energy errors
       Use symplectic integrators for better conservation

    6. **Insufficient damping modeling:**
       Real pendulums have complex friction
       Viscous approximation b·ω may be inadequate

    Extensions:
    ----------
    1. **Driven pendulum:**
       Add periodic forcing: u[k] = A·cos(Ω·k·dt)
       Can produce chaos and strange attractors

    2. **Double pendulum:**
       Two coupled pendula
       Exhibits deterministic chaos

    3. **Spherical pendulum:**
       3D motion (θ, φ)
       Rich dynamics, Coriolis effects

    4. **Elastic pendulum:**
       Pendulum with spring (varying L)
       Coupled radial-angular motion

    5. **Pendulum on cart:**
       Inverted pendulum (classic underactuated system)
       Noncollocated control problem

    6. **Parametric excitation:**
       Varying L sinusoidally
       Parametric resonance effects

    See Also:
    --------
    DiscreteRobotArm : Similar dynamics (different notation)
    DiscreteOscillator : Linear limit (small angles)
    DoublePendulum : Chaotic extension
    """

    def define_system(
        self,
        m: float = 1.0,
        L: float = 1.0,
        g: float = 9.81,
        b: float = 0.1,
        dt: float = 0.01,
        method: str = 'zoh',
        use_control: bool = True,
    ):
        """
        Define discrete-time pendulum dynamics.

        Parameters
        ----------
        m : float
            Mass [kg]
        L : float
            Length [m]
        g : float
            Gravity [m/s²]
        b : float
            Damping coefficient [N·m·s/rad]
        dt : float
            Sampling period [s]
        method : str
            Discretization method ('zoh', 'euler', 'rk4')
        use_control : bool
            If True, include control input
        """
        # Store parameters
        self.m = m
        self.L = L
        self.g = g
        self.b = b
        self._method = method
        self._use_control = use_control

        # Derived quantities
        self.I = m * L**2  # Moment of inertia
        self.omega_0 = np.sqrt(g / L)  # Natural frequency
        self.beta = b / self.I  # Normalized damping

        # State variables
        theta, omega = sp.symbols('theta omega', real=True)

        # Symbolic parameters
        m_sym, L_sym, g_sym, b_sym = sp.symbols('m L g b', real=True, positive=True)

        self.state_vars = [theta, omega]
        self._dt = dt
        self.order = 1

        self.parameters = {
            m_sym: m,
            L_sym: L,
            g_sym: g,
            b_sym: b,
        }

        # Control input (optional)
        if use_control:
            tau = sp.symbols('tau', real=True)
            self.control_vars = [tau]
            # Normalized control
            u_normalized = tau / (m_sym * L_sym**2)
        else:
            self.control_vars = []
            u_normalized = 0

        # Angular acceleration from equation of motion
        # θ̈ = -(g/L)·sin(θ) - (b/(m·L²))·θ̇ + τ/(m·L²)
        alpha = (
            -(g_sym / L_sym) * sp.sin(theta)
            - (b_sym / (m_sym * L_sym**2)) * omega
            + u_normalized
        )

        # Discretize
        if method == 'zoh':
            # Zero-order hold
            theta_next = theta + dt * omega + 0.5 * dt**2 * alpha
            omega_next = omega + dt * alpha

        elif method == 'euler':
            # Forward Euler
            theta_next = theta + dt * omega
            omega_next = omega + dt * alpha

        elif method == 'rk4':
            # Runge-Kutta 4th order
            # k1
            k1_theta = omega
            k1_omega = alpha

            # k2
            theta_k2 = theta + 0.5 * dt * k1_theta
            omega_k2 = omega + 0.5 * dt * k1_omega
            alpha_k2 = (
                -(g_sym / L_sym) * sp.sin(theta_k2)
                - (b_sym / (m_sym * L_sym**2)) * omega_k2
                + u_normalized
            )
            k2_theta = omega_k2
            k2_omega = alpha_k2

            # k3
            theta_k3 = theta + 0.5 * dt * k2_theta
            omega_k3 = omega + 0.5 * dt * k2_omega
            alpha_k3 = (
                -(g_sym / L_sym) * sp.sin(theta_k3)
                - (b_sym / (m_sym * L_sym**2)) * omega_k3
                + u_normalized
            )
            k3_theta = omega_k3
            k3_omega = alpha_k3

            # k4
            theta_k4 = theta + dt * k3_theta
            omega_k4 = omega + dt * k3_omega
            alpha_k4 = (
                -(g_sym / L_sym) * sp.sin(theta_k4)
                - (b_sym / (m_sym * L_sym**2)) * omega_k4
                + u_normalized
            )
            k4_theta = omega_k4
            k4_omega = alpha_k4

            # Combine
            theta_next = theta + (dt / 6.0) * (k1_theta + 2*k2_theta + 2*k3_theta + k4_theta)
            omega_next = omega + (dt / 6.0) * (k1_omega + 2*k2_omega + 2*k3_omega + k4_omega)

        else:
            raise ValueError(f"Unknown discretization method: {method}")

        self._f_sym = sp.Matrix([theta_next, omega_next])
        self._h_sym = sp.Matrix([theta])  # Measure angle only
        self.output_vars = []

    def setup_equilibria(self):
        """Set up downward and upward equilibria."""
        # Downward (stable)
        self.add_equilibrium(
            "downward",
            x_eq=np.array([0.0, 0.0]),
            u_eq=np.array([0.0]) if self._use_control else np.array([]),
            verify=True,
            stability="stable",
            notes=f"Stable equilibrium. Natural frequency: {self.natural_frequency:.3f} rad/s, "
                  f"Period: {self.period:.3f} s"
        )

        # Upward (unstable)
        self.add_equilibrium(
            "upward",
            x_eq=np.array([np.pi, 0.0]),
            u_eq=np.array([0.0]) if self._use_control else np.array([]),
            verify=True,
            stability="unstable",
            notes="Unstable inverted equilibrium. Requires active stabilization."
        )

        self.set_default_equilibrium("downward")

    @property
    def natural_frequency(self) -> float:
        """Natural frequency ω₀ = √(g/L) [rad/s]."""
        return self.omega_0

    @property
    def period(self) -> float:
        """Period of small oscillations T = 2π/ω₀ [s]."""
        return 2.0 * np.pi / self.omega_0

    @property
    def damping_ratio(self) -> float:
        """Damping ratio ζ = β/(2ω₀) [-]."""
        return self.beta / (2.0 * self.omega_0)

    def compute_kinetic_energy(self, omega: float) -> float:
        """Kinetic energy K = (1/2)·I·ω² [J]."""
        return 0.5 * self.I * omega**2

    def compute_potential_energy(self, theta: float) -> float:
        """Potential energy V = m·g·L·(1 - cos(θ)) [J]."""
        return self.m * self.g * self.L * (1.0 - np.cos(theta))

    def compute_total_energy(self, theta: float, omega: float) -> float:
        """Total mechanical energy E = K + V [J]."""
        return self.compute_kinetic_energy(omega) + self.compute_potential_energy(theta)

    def compute_separatrix_energy(self) -> float:
        """Energy of separatrix (boundary between oscillation/rotation) [J]."""
        return 2.0 * self.m * self.g * self.L

    def design_upward_stabilizer(self) -> np.ndarray:
        """
        Design LQR controller for upward stabilization.

        Returns
        -------
        np.ndarray
            LQR gain K
        """
        # Linearize around upward
        x_up = np.array([np.pi, 0.0])
        u_up = np.array([0.0]) if self._use_control else np.array([])

        Ad, Bd = self.linearize(x_up, u_up)

        # LQR design
        Q = np.diag([100.0, 10.0])
        R = np.array([[1.0]])

        lqr_result = self.control.design_lqr(Ad, Bd, Q, R, system_type='discrete')
        return lqr_result['gain']

    def compute_separatrix(
        self,
        n_points: int = 1000,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute separatrix trajectory (for undamped case).

        Returns
        -------
        tuple
            (theta, omega) points on separatrix
        """
        if self.b > 1e-10:
            import warnings
            warnings.warn("Separatrix only well-defined for undamped case (b=0)")

        # Separatrix has energy E = 2mgL
        E_sep = self.compute_separatrix_energy()

        # Solve for omega as function of theta
        theta_vals = np.linspace(-np.pi, np.pi, n_points)
        omega_vals = np.zeros_like(theta_vals)

        for i, theta in enumerate(theta_vals):
            V = self.compute_potential_energy(theta)
            K = E_sep - V
            if K >= 0:
                omega_vals[i] = np.sqrt(2 * K / self.I)
            else:
                omega_vals[i] = 0

        # Mirror for negative velocities
        theta_full = np.concatenate([theta_vals, theta_vals[::-1]])
        omega_full = np.concatenate([omega_vals, -omega_vals[::-1]])

        return theta_full, omega_full

    # def print_equations(self, simplify: bool = True):
    #     """Print symbolic equations."""
    #     print("=" * 70)
    #     print(f"{self.__class__.__name__} (Discrete-Time, dt={self.dt} s)")
    #     print("=" * 70)
    #     print("Simple Pendulum: Classic Nonlinear Oscillator")
    #     print(f"Discretization Method: {self._method.upper()}")

    #     print("\nPhysical Parameters:")
    #     print(f"  Mass: m = {self.m} kg")
    #     print(f"  Length: L = {self.L} m")
    #     print(f"  Gravity: g = {self.g} m/s²")
    #     print(f"  Damping: b = {self.b} N·m·s/rad")
    #     print(f"  Moment of inertia: I = m·L² = {self.I:.4f} kg·m²")

    #     print("\nDynamic Characteristics:")
    #     print(f"  Natural frequency: ω₀ = {self.natural_frequency:.3f} rad/s")
    #     print(f"  Natural period: T = {self.period:.3f} s")
    #     print(f"  Damping ratio: ζ = {self.damping_ratio:.4f}")
        
    #     if self.damping_ratio < 1.0:
    #         print(f"  Classification: Underdamped")
    #     elif abs(self.damping_ratio - 1.0) < 0.01:
    #         print(f"  Classification: Critically damped")
    #     else:
    #         print(f"  Classification: Overdamped")

    #     print(f"\nState: x = [θ, ω] (angle, angular velocity)")
    #     if self._use_control:
    #         print(f"Control: u = [τ] (applied torque)")
    #     else:
    #         print("No control (free pendulum)")
    #     print(f"Dimensions: nx={self.nx}, nu={self.nu}")

    #     print("\nContinuous-Time Equation of Motion:")
    #     print("  m·L²·θ̈ = -m·g·L·sin(θ) - b·θ̇ + τ")
    #     print("\nNormalized form:")
    #     print("  θ̈ = -ω₀²·sin(θ) - β·θ̇ + u")
    #     print(f"  where ω₀² = g/L = {self.natural_frequency**2:.3f}")
    #     print(f"        β = b/(m·L²) = {self.beta:.4f}")

    #     print("\nDiscrete-Time Dynamics: x[k+1] = f(x[k], u[k])")
    #     for i, (var, expr) in enumerate(zip(self.state_vars, self._f_sym)):
    #         expr_sub = self.substitute_parameters(expr)
    #         if simplify:
    #             expr_sub = sp.simplify(expr_sub)
    #         label = ['θ[k+1]', 'ω[k+1]'][i]
    #         print(f"  {label} = {expr_sub}")

    #     print("\nOutput: y[k] = θ[k] (angle measurement)")

    #     print("\nEquilibria:")
    #     print("  1. Downward (θ=0, ω=0): STABLE")
    #     print("     - Minimum potential energy")
    #     print("     - Small oscillations have period T = 2π/ω₀")
    #     print("  2. Upward (θ=±π, ω=0): UNSTABLE")
    #     print("     - Maximum potential energy")
    #     print("     - Inverted pendulum configuration")

    #     E_sep = self.compute_separatrix_energy()
    #     print(f"\nSeparatrix Energy (oscillation/rotation boundary):")
    #     print(f"  E_sep = 2·m·g·L = {E_sep:.4f} J")

    #     print("\nSmall Angle Approximation (|θ| << 1):")
    #     print("  sin(θ) ≈ θ → Linear harmonic oscillator")
    #     print("  Valid for θ < 0.2 rad (~10°)")

    #     print("\nPhysical Interpretation:")
    #     print("  - θ: Angle from downward vertical [rad]")
    #     print("  - ω: Angular velocity [rad/s]")
    #     print("  - Nonlinear due to sin(θ) term")
    #     print("  - Period increases with amplitude (anharmonicity)")

    #     print("\nApplications:")
    #     print("  - Clock mechanisms (Galileo's discovery)")
    #     print("  - Seismometers and accelerometers")
    #     print("  - Ship stabilization systems")
    #     print("  - Educational demonstrations")
    #     print("  - Nonlinear dynamics benchmark")

    #     print("=" * 70)


# Aliases
SimplePendulum = DiscretePendulum
DiscretePendulumWithFriction = DiscretePendulum