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
Discrete-time Single-Link Robot Arm - Nonlinear Control Benchmark.

This module provides discrete-time models of a single-link robot arm (revolute
joint), one of the fundamental systems in robotics and control theory. It serves as:
- A canonical example of underactuated mechanical systems
- A benchmark for nonlinear control design
- A model for robot manipulator dynamics (single degree of freedom)
- An illustration of gravity compensation and friction effects
- A testbed for trajectory tracking and motion planning

The single-link robot arm represents:
- Satellite attitude control (simplified)
- Robotic joint dynamics (shoulder, elbow, wrist)
- Inverted pendulum on cart (rotational analog)
- Crane or boom control systems
- Any rotating rigid body under torque control

This is the discrete-time version, appropriate for:
- Digital control implementation (microcontrollers, PLCs)
- Sampled-data systems (periodic sensing and actuation)
- Real-time control with fixed update rates
- Model predictive control (MPC)
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import sympy as sp
from scipy.optimize import fsolve

from src.systems.base.core.discrete_symbolic_system import DiscreteSymbolicSystem


class DiscreteRobotArm(DiscreteSymbolicSystem):
    """
    Discrete-time single-link robot arm with gravity and friction.

    Physical System:
    ---------------
    A rigid link of length L and mass m rotating about a fixed pivot (revolute
    joint) under the influence of:
    - Applied torque τ (control input)
    - Gravitational torque (proportional to sin(θ))
    - Viscous friction (proportional to angular velocity)
    - Coulomb friction (optional, velocity-dependent sign)

    **Mechanical Configuration:**
```
              τ (motor torque)
              ↓
         ┌────●──── Pivot (fixed)
         │    
         │ L (link length)
         │    
         ●    ← m (mass at center of gravity)
         │    
         │    
         ↓ g (gravity)
```

    The link rotates in a vertical plane with angle θ measured from the downward
    vertical position (θ = 0 is straight down, θ = π is straight up).

    **Continuous-Time Dynamics (Euler-Lagrange):**
    The equation of motion derived from Lagrangian mechanics:

        I·θ̈ = τ - m·g·L_c·sin(θ) - b·θ̇

    where:
        I: Moment of inertia about pivot [kg·m²]
        τ: Applied torque (control input) [N·m]
        m·g·L_c·sin(θ): Gravitational torque [N·m]
        b·θ̇: Viscous friction torque [N·m]

    For a uniform rod rotating about one end:
        I = (1/3)·m·L²

    For point mass at distance L_c:
        I = m·L_c²

    **Discrete-Time Dynamics (Zero-Order Hold):**
    Exact discretization assuming constant torque between samples:

        θ[k+1] = θ[k] + dt·ω[k] + 0.5·dt²·α[k]
        ω[k+1] = ω[k] + dt·α[k]

    where angular acceleration:
        α[k] = (τ[k] - m·g·L_c·sin(θ[k]) - b·ω[k]) / I

    This is the ZOH discretization of the second-order mechanical system.

    State Space:
    -----------
    State: x[k] = [θ[k], ω[k]]
        Angular position:
        - θ: Joint angle [rad]
          * θ = 0: Link pointing down (equilibrium, stable)
          * θ = π: Link pointing up (equilibrium, unstable)
          * θ = ±π/2: Horizontal positions
          * Periodic: θ and θ + 2π are equivalent
          * Typically unwrapped for control (track rotations)

        Angular velocity:
        - ω: Joint angular velocity [rad/s]
          * ω > 0: Counterclockwise rotation
          * ω < 0: Clockwise rotation
          * ω = 0: Instantaneous rest
          * Typical range: -10 to +10 rad/s for industrial robots

    Control: u[k] = [τ[k]]
        Applied torque:
        - τ: Motor/actuator torque [N·m]
          * τ > 0: Counterclockwise torque (lifts link)
          * τ < 0: Clockwise torque (lowers link)
          * Bounded by actuator: τ_min ≤ τ ≤ τ_max
          * Typical: |τ| ≤ 100 N·m for industrial arms

    Output: y[k] = [θ[k]] or [θ[k], ω[k]]
        - Position-only measurement (most common):
          y[k] = θ[k]
          Measured via encoder, resolver, or potentiometer
        - Full state measurement:
          y[k] = [θ[k], ω[k]]
          Velocity from tachometer or differentiation

    Dynamics (Physical Interpretation):
    -----------------------------------
    The discrete dynamics combine three effects:

    **1. Inertial Term (I·α):**
       - Resistance to angular acceleration
       - Larger I → slower response to torque
       - Depends on mass distribution (I = ∫r²·dm)

    **2. Gravity Term (-m·g·L_c·sin(θ)):**
       - Restoring torque toward θ = 0 (downward)
       - Destabilizing torque near θ = π (upward)
       - Maximum at θ = ±π/2 (horizontal)
       - Zero at θ = 0, π (vertical positions)
       - Creates nonlinearity (sinusoidal)

    **3. Friction Term (-b·ω):**
       - Opposes motion (always negative sign relative to ω)
       - Dissipates energy
       - Causes exponential decay of free oscillations
       - Linear approximation (Coulomb friction more realistic)

    **Energy Considerations:**
    Total mechanical energy:
        E = E_kinetic + E_potential
        E = (1/2)·I·ω² + m·g·L_c·(1 - cos(θ))

    Potential energy reference at θ = 0 (downward).
    
    Energy changes due to:
    - Control input: P_control = τ·ω (power delivered)
    - Friction: P_friction = -b·ω² (power dissipated, always negative)

    Parameters:
    ----------
    m : float, default=1.0
        Mass of the link [kg]
        - Typical robot arms: 0.5-50 kg
        - Satellite appendages: 1-100 kg
        - Industrial robots: 5-500 kg

    L : float, default=1.0
        Length of the link [m]
        - Total physical length
        - Typical: 0.1-3.0 m for robot arms
        - Satellite booms: 1-20 m

    L_c : float, default=0.5
        Distance from pivot to center of mass [m]
        - For uniform rod: L_c = L/2
        - For point mass at end: L_c = L
        - For complex shapes: depends on mass distribution
        - Must satisfy: 0 < L_c ≤ L

    g : float, default=9.81
        Gravitational acceleration [m/s²]
        - Earth: 9.81
        - Moon: 1.62
        - Mars: 3.71
        - Space (microgravity): ~0

    b : float, default=0.1
        Viscous friction coefficient [N·m·s/rad]
        - Models bearing friction, air resistance
        - Typical: 0.01-1.0 for robot joints
        - Larger values → more damping
        - b = 0: No friction (conservative system)

    dt : float, default=0.01
        Sampling period [s]
        - Digital control update rate
        - Typical: 0.001-0.1 s (10 Hz - 1 kHz)
        - Must satisfy Nyquist criterion
        - Smaller dt → better tracking, higher computation

    method : str, default='zoh'
        Discretization method:
        - 'zoh': Zero-order hold (exact for constant τ)
        - 'euler': Forward Euler (simple, less accurate)
        - 'rk4': Runge-Kutta 4th order (high accuracy)

    inertia_type : str, default='uniform_rod'
        How to compute moment of inertia:
        - 'uniform_rod': I = (1/3)·m·L² (rod about end)
        - 'point_mass': I = m·L_c² (mass at L_c)
        - 'thin_rod_center': I = (1/12)·m·L² (rod about center)
        - 'custom': Use provided I value

    I_custom : Optional[float]
        Custom moment of inertia [kg·m²] if inertia_type='custom'

    Equilibria:
    ----------
    The system has two equilibrium points (where α = 0 with τ = 0):

    **1. Downward Equilibrium (θ = 0, ω = 0):**
       Stability: STABLE (locally asymptotically stable)
       - Link hanging down (lowest potential energy)
       - Small perturbations oscillate and decay (if b > 0)
       - Eigenvalues in unit circle (|λ| < 1)
       - Natural resting position
       - Requires no torque to maintain

    **2. Upward Equilibrium (θ = π, ω = 0):**
       Stability: UNSTABLE (saddle point)
       - Link pointing up (highest potential energy)
       - Small perturbations grow exponentially
       - Eigenvalues outside unit circle (|λ| > 1)
       - Requires active control to maintain (like inverted pendulum)
       - Challenging control problem

    **Required Torque for Equilibrium at Arbitrary θ:**
    To hold link at angle θ* with ω* = 0:
        τ_eq = m·g·L_c·sin(θ*)

    Examples:
    - θ* = 0 (down): τ_eq = 0 (no torque needed)
    - θ* = π/2 (horizontal): τ_eq = m·g·L_c (maximum)
    - θ* = π (up): τ_eq = 0 (but unstable!)

    Controllability:
    ---------------
    The discrete robot arm is COMPLETELY CONTROLLABLE from any state to any
    other state (in finite time) as long as τ is unbounded.

    **Proof:** The system is a discretization of:
        ẍ = f(x, ẋ) + (1/I)·u

    where control u appears directly in acceleration. Since the system is
    second-order with full-rank controllability matrix, it's controllable.

    **Practical Controllability:**
    With bounded control |τ| ≤ τ_max:
    - Some states unreachable in finite time
    - Upward position (θ = π) requires minimum τ_max > m·g·L_c
    - Fast motions require large torques (I·α_max = τ_max)

    **Minimum Time Control:**
    For point-to-point motion, optimal control is typically bang-bang:
    1. Apply maximum torque τ_max (accelerate)
    2. Coast (τ = 0) or partial torque
    3. Apply minimum torque -τ_max (decelerate)
    4. Arrive at target with ω = 0

    Observability:
    -------------
    **Position-only measurement: y[k] = θ[k]**
        Observable for almost all trajectories.
        Velocity can be estimated from position differences:
            ω[k] ≈ (θ[k] - θ[k-1]) / dt

        Better: Use Kalman filter or observer for ω estimation.

    **Full state measurement: y[k] = [θ[k], ω[k]]**
        Trivially observable (direct measurement).

    **Nonlinear Observability:**
    The system is nonlinear, so observability depends on trajectory.
    For most motions, position measurement sufficient to reconstruct full state.

    Control Objectives:
    ------------------
    **1. Regulation (Stabilization):**
       Goal: Drive to equilibrium (typically θ = 0, ω = 0)
       Methods:
       - PD control: τ = -k_p·θ - k_d·ω
       - LQR (linearized): Optimal gains for quadratic cost
       - Energy-based: Swing-up then stabilize
       - Sliding mode: Robust to uncertainty

    **2. Trajectory Tracking:**
       Goal: Follow reference trajectory θ_ref(t), ω_ref(t)
       Control law:
           τ = I·α_ref + m·g·L_c·sin(θ) + b·ω - K·e

       where e = [θ - θ_ref, ω - ω_ref] is tracking error.

       Applications:
       - Pick-and-place operations
       - Assembly tasks
       - Welding/cutting paths

    **3. Swing-Up Control:**
       Goal: Move from downward (θ = 0) to upward (θ = π)
       Challenge: Requires energy injection, then stabilization
       Methods:
       - Energy-based control (pump energy until E = E_target)
       - Partial feedback linearization
       - Model predictive control (MPC)

    **4. Disturbance Rejection:**
       Goal: Maintain position despite external forces
       Disturbances:
       - Unknown loads (changing m)
       - Wind/vibrations
       - Model uncertainties
       Methods:
       - Integral action (PI/PID)
       - Disturbance observer
       - Adaptive control

    **5. Optimal Control:**
       Goal: Minimize cost functional (time, energy, jerk)
       Examples:
       - Minimum time: J = ∫ dt
       - Minimum energy: J = ∫ τ² dt
       - Minimum jerk: J = ∫ (dα/dt)² dt (smooth motion)

    State Constraints:
    -----------------
    **1. Joint Limits: θ_min ≤ θ[k] ≤ θ_max**
       - Physical stops prevent full rotation
       - Typical: -π ≤ θ ≤ π or 0 ≤ θ ≤ 2π
       - Some joints: limited range (e.g., ±90°)
       - Must respect in trajectory planning

    **2. Velocity Limits: |ω[k]| ≤ ω_max**
       - Motor speed limitations
       - Safety considerations
       - Typical: ω_max = 5-50 rad/s
       - Affects trajectory feasibility

    **3. Torque Limits: |τ[k]| ≤ τ_max**
       - Motor torque saturation (most critical)
       - Typical: τ_max = 10-1000 N·m
       - Causes anti-windup issues in PI/PID
       - Must account for in MPC

    **4. Acceleration Limits: |α[k]| ≤ α_max**
       - Mechanical stress limitations
       - Passenger/payload comfort
       - Jerk limits: |dα/dt| ≤ jerk_max

    Numerical Considerations:
    ------------------------
    **Discretization Accuracy:**
    - ZOH: Exact for piecewise-constant τ
    - Euler: O(dt) error, may be unstable for large dt
    - RK4: O(dt⁴) error, excellent accuracy

    **Stability Condition:**
    For explicit methods, stability requires:
        dt < 2/√(ω_n²)
    
    where ω_n ≈ √(m·g·L_c/I) is natural frequency.

    For typical parameters: dt < 0.1 s is safe.

    **Angle Wrapping:**
    For continuous rotation (no joint limits):
    - Wrap θ to [-π, π] or [0, 2π]
    - Avoid discontinuities in controller
    - Use sin(θ), cos(θ) when possible (smooth)

    **Singularities:**
    - θ = ±π: Linearization singular (sin(θ) ≈ -θ not valid)
    - Requires nonlinear control near upward position

    Example Usage:
    -------------
    >>> # Create robot arm with realistic parameters
    >>> system = DiscreteRobotArm(
    ...     m=2.0,        # 2 kg link
    ...     L=0.5,        # 50 cm long
    ...     L_c=0.25,     # Center of mass at midpoint
    ...     g=9.81,       # Earth gravity
    ...     b=0.1,        # Light damping
    ...     dt=0.01       # 100 Hz control rate
    ... )
    >>> 
    >>> # Initial condition: hanging down, at rest
    >>> x0 = np.array([0.0, 0.0])
    >>> 
    >>> # Design PD controller for downward stabilization
    >>> Ad, Bd = system.linearize(np.zeros(2), np.zeros(1))
    >>> 
    >>> # LQR design
    >>> Q = np.diag([100.0, 1.0])  # Care more about position
    >>> R = np.array([[1.0]])
    >>> lqr_result = system.control.design_lqr(
    ...     Ad, Bd, Q, R, system_type='discrete'
    ... )
    >>> K = lqr_result['gain']
    >>> print(f"LQR gain: K = {K}")
    >>> 
    >>> # Simulate with LQR control from perturbed initial condition
    >>> x0_perturbed = np.array([0.5, 0.0])  # 30° displacement
    >>> 
    >>> def lqr_controller(x, k):
    ...     return -K @ x
    >>> 
    >>> result_lqr = system.rollout(
    ...     x0_perturbed,
    ...     lqr_controller,
    ...     n_steps=500
    ... )
    >>> 
    >>> # Plot results
    >>> import plotly.graph_objects as go
    >>> from plotly.subplots import make_subplots
    >>> 
    >>> fig = make_subplots(
    ...     rows=3, cols=1,
    ...     subplot_titles=['Angle', 'Angular Velocity', 'Control Torque']
    ... )
    >>> 
    >>> t = result_lqr['time_steps'] * system.dt
    >>> 
    >>> # Angle
    >>> fig.add_trace(
    ...     go.Scatter(x=t, y=result_lqr['states'][:, 0], name='θ'),
    ...     row=1, col=1
    ... )
    >>> fig.add_hline(y=0, line_dash='dash', line_color='red', row=1, col=1)
    >>> 
    >>> # Angular velocity
    >>> fig.add_trace(
    ...     go.Scatter(x=t, y=result_lqr['states'][:, 1], name='ω'),
    ...     row=2, col=1
    ... )
    >>> 
    >>> # Control torque
    >>> fig.add_trace(
    ...     go.Scatter(x=t, y=result_lqr['controls'][:, 0], name='τ'),
    ...     row=3, col=1
    ... )
    >>> 
    >>> fig.update_xaxes(title_text='Time [s]', row=3, col=1)
    >>> fig.update_yaxes(title_text='θ [rad]', row=1, col=1)
    >>> fig.update_yaxes(title_text='ω [rad/s]', row=2, col=1)
    >>> fig.update_yaxes(title_text='τ [N·m]', row=3, col=1)
    >>> fig.update_layout(height=800, showlegend=False)
    >>> fig.show()
    >>> 
    >>> # Gravity compensation control
    >>> def gravity_comp_pd(x, k):
    ...     theta, omega = x
    ...     # PD gains
    ...     k_p, k_d = 50.0, 10.0
    ...     # Desired position (horizontal)
    ...     theta_d = np.pi / 2
    ...     # PD + gravity compensation
    ...     tau_pd = -k_p * (theta - theta_d) - k_d * omega
    ...     tau_gravity = system.compute_gravity_torque(theta)
    ...     return tau_pd + tau_gravity
    >>> 
    >>> result_grav = system.rollout(
    ...     x0=np.array([0.0, 0.0]),
    ...     policy=gravity_comp_pd,
    ...     n_steps=1000
    ... )
    >>> 
    >>> # Swing-up control (energy-based)
    >>> def swing_up_controller(x, k):
    ...     theta, omega = x
    ...     
    ...     # Compute current energy
    ...     E_current = system.compute_total_energy(theta, omega)
    ...     E_target = system.compute_potential_energy(np.pi)  # Upward
    ...     
    ...     # If near upward, switch to stabilization
    ...     if abs(theta - np.pi) < 0.2 and abs(omega) < 0.5:
    ...         # LQR around upward equilibrium
    ...         x_error = np.array([theta - np.pi, omega])
    ...         K_up = system.design_upward_controller()
    ...         return -K_up @ x_error
    ...     else:
    ...         # Energy pumping
    ...         E_error = E_current - E_target
    ...         k_E = 2.0
    ...         return -k_E * E_error * omega * np.cos(theta)
    >>> 
    >>> result_swing = system.rollout(
    ...     x0=np.array([0.0, 0.0]),
    ...     policy=swing_up_controller,
    ...     n_steps=2000
    ... )
    >>> 
    >>> # Trajectory tracking
    >>> # Generate smooth trajectory
    >>> n_steps_traj = 500
    >>> t_traj = np.arange(n_steps_traj) * system.dt
    >>> 
    >>> # Polynomial trajectory (quintic for smooth accel/decel)
    >>> theta_ref, omega_ref, alpha_ref = system.generate_quintic_trajectory(
    ...     theta_0=0.0,
    ...     theta_f=np.pi/2,
    ...     T=t_traj[-1],
    ...     n_points=n_steps_traj
    ... )
    >>> 
    >>> def tracking_controller(x, k):
    ...     if k >= len(theta_ref):
    ...         k = len(theta_ref) - 1
    ...     
    ...     theta, omega = x
    ...     
    ...     # Feedforward (inverse dynamics)
    ...     tau_ff = (system.I * alpha_ref[k] + 
    ...               system.m * system.g * system.L_c * np.sin(theta_ref[k]) +
    ...               system.b * omega_ref[k])
    ...     
    ...     # Feedback (PD on error)
    ...     e_theta = theta - theta_ref[k]
    ...     e_omega = omega - omega_ref[k]
    ...     tau_fb = -50.0 * e_theta - 10.0 * e_omega
    ...     
    ...     return tau_ff + tau_fb
    >>> 
    >>> result_track = system.rollout(
    ...     x0=np.array([0.0, 0.0]),
    ...     policy=tracking_controller,
    ...     n_steps=n_steps_traj
    ... )
    >>> 
    >>> # Plot tracking performance
    >>> fig_track = go.Figure()
    >>> fig_track.add_trace(go.Scatter(
    ...     x=t_traj, y=theta_ref,
    ...     name='Reference', line=dict(dash='dash', width=3)
    ... ))
    >>> fig_track.add_trace(go.Scatter(
    ...     x=t_traj, y=result_track['states'][:, 0],
    ...     name='Actual'
    ... ))
    >>> fig_track.update_layout(
    ...     title='Trajectory Tracking',
    ...     xaxis_title='Time [s]',
    ...     yaxis_title='Angle [rad]'
    ... )
    >>> fig_track.show()
    >>> 
    >>> # Compute tracking error metrics
    >>> error = result_track['states'][:, 0] - theta_ref
    >>> rmse = np.sqrt(np.mean(error**2))
    >>> max_error = np.max(np.abs(error))
    >>> print(f"RMSE: {rmse:.4f} rad")
    >>> print(f"Max error: {max_error:.4f} rad")

    Physical Insights:
    -----------------
    **Underactuation:**
    The robot arm is NOT underactuated - it has 2 states (θ, ω) and 1 control (τ),
    but since τ directly affects acceleration (second derivative), it's
    fully actuated. Compare to cart-pole (4 states, 1 control = underactuated).

    **Gravity Compensation:**
    The term m·g·L_c·sin(θ) represents gravitational torque. To hold the arm
    at angle θ without moving:
        τ_hold = m·g·L_c·sin(θ)

    This is feed-forward compensation - cancels gravity exactly.
    
    At θ = π/2 (horizontal): Maximum torque = m·g·L_c
    At θ = 0 or π (vertical): Zero torque needed

    **Energy Perspective:**
    Total energy: E = (1/2)·I·ω² + m·g·L_c·(1 - cos(θ))

    For swing-up without friction:
    - Pump energy until E = m·g·L_c·2 (upward position)
    - Strategy: Add energy when ω and (θ - θ_target) have same sign

    **Passivity:**
    The robot arm (without control) is passive:
    - Energy dissipated by friction: dE/dt = -b·ω² ≤ 0
    - No energy created spontaneously
    - Useful property for stability analysis

    **Friction Effects:**
    - Viscous friction (b·ω): Opposes motion, smooth
    - Coulomb friction (F_c·sign(ω)): Constant magnitude, causes stick-slip
    - Static friction: Prevents motion until τ > τ_static
    - Often modeled as: τ_friction = b·ω + F_c·sign(ω) + F_s·δ(ω)

    **Coriolis and Centrifugal Forces:**
    For single link, these are zero. For multi-link arms:
    - Coriolis: velocity-dependent coupling between joints
    - Centrifugal: velocity-squared terms
    - Complicate dynamics significantly

    Common Pitfalls:
    ---------------
    1. **Forgetting gravity compensation:**
       PD control alone unstable at non-zero angles
       Always add τ_gravity = m·g·L_c·sin(θ)

    2. **Wrong linearization point:**
       Linearizing at θ = π (upward) gives UNSTABLE system
       Different controller needed for upward stabilization

    3. **Angle wrapping issues:**
       θ = π and θ = -π are same position
       Controller must handle discontinuity

    4. **Ignoring torque saturation:**
       Real motors have τ_max
       Anti-windup essential for integral control

    5. **Insufficient discretization rate:**
       dt too large → poor tracking, instability
       Nyquist: Sample at least 10× natural frequency

    6. **Velocity estimation noise:**
       Numerical differentiation amplifies noise
       Use Kalman filter or low-pass filter

    Extensions:
    ----------
    1. **Multi-link arm:**
       Couple multiple single-link dynamics
       Adds Coriolis, centrifugal terms

    2. **Flexible link:**
       Model link as elastic beam
       Infinite-dimensional system (PDE)

    3. **With payload:**
       Variable mass at end-effector
       Adaptive control needed

    4. **Friction models:**
       Coulomb, Stribeck, LuGre models
       More realistic friction behavior

    5. **Joint compliance:**
       Series elastic actuator
       Adds flexibility, impedance control

    6. **Cooperative manipulation:**
       Multiple arms holding object
       Force control, coordination

    See Also:
    --------
    DiscreteDoubleIntegrator : Simpler system (no gravity)
    DiscretePendulum : Similar (different notation)
    DiscreteCartPole : Underactuated extension
    """

    def define_system(
        self,
        m: float = 1.0,
        L: float = 1.0,
        L_c: Optional[float] = None,
        g: float = 9.81,
        b: float = 0.1,
        dt: float = 0.01,
        method: str = 'zoh',
        inertia_type: str = 'uniform_rod',
        I_custom: Optional[float] = None,
    ):
        """
        Define discrete-time robot arm dynamics.

        Parameters
        ----------
        m : float
            Link mass [kg]
        L : float
            Link length [m]
        L_c : Optional[float]
            Distance to center of mass [m] (default: L/2)
        g : float
            Gravitational acceleration [m/s²]
        b : float
            Viscous friction coefficient [N·m·s/rad]
        dt : float
            Sampling period [s]
        method : str
            Discretization method ('zoh', 'euler', 'rk4')
        inertia_type : str
            How to compute inertia ('uniform_rod', 'point_mass', 'custom')
        I_custom : Optional[float]
            Custom moment of inertia [kg·m²]
        """
        # Store configuration
        self._method = method
        self._inertia_type = inertia_type

        # Default center of mass at midpoint
        if L_c is None:
            L_c = L / 2.0

        # Validate
        if L_c > L or L_c <= 0:
            raise ValueError(f"Center of mass L_c must satisfy 0 < L_c ≤ L")

        # Compute moment of inertia
        if inertia_type == 'uniform_rod':
            # Rod rotating about one end
            I = (1.0 / 3.0) * m * L**2
        elif inertia_type == 'point_mass':
            # Point mass at distance L_c
            I = m * L_c**2
        elif inertia_type == 'thin_rod_center':
            # Rod rotating about center
            I = (1.0 / 12.0) * m * L**2
        elif inertia_type == 'custom':
            if I_custom is None:
                raise ValueError("Must provide I_custom for inertia_type='custom'")
            I = I_custom
        else:
            raise ValueError(f"Unknown inertia_type: {inertia_type}")

        # Store physical parameters
        self.m = m
        self.L = L
        self.L_c = L_c
        self.g = g
        self.b = b
        self.I = I

        # State variables
        theta, omega = sp.symbols('theta omega', real=True)
        tau = sp.symbols('tau', real=True)

        # Symbolic parameters
        m_sym, L_c_sym, g_sym, b_sym, I_sym = sp.symbols(
            'm L_c g b I', real=True, positive=True
        )

        self.state_vars = [theta, omega]
        self.control_vars = [tau]
        self.output_vars = []
        self._dt = dt
        self.order = 1

        self.parameters = {
            m_sym: m,
            L_c_sym: L_c,
            g_sym: g,
            b_sym: b,
            I_sym: I,
        }

        # Angular acceleration (from equation of motion)
        alpha = (tau - m_sym * g_sym * L_c_sym * sp.sin(theta) - b_sym * omega) / I_sym

        # Discretize based on method
        if method == 'zoh':
            # Zero-order hold (exact for constant torque)
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
            alpha_k2 = (tau - m_sym * g_sym * L_c_sym * sp.sin(theta_k2) - b_sym * omega_k2) / I_sym
            k2_theta = omega_k2
            k2_omega = alpha_k2

            # k3
            theta_k3 = theta + 0.5 * dt * k2_theta
            omega_k3 = omega + 0.5 * dt * k2_omega
            alpha_k3 = (tau - m_sym * g_sym * L_c_sym * sp.sin(theta_k3) - b_sym * omega_k3) / I_sym
            k3_theta = omega_k3
            k3_omega = alpha_k3

            # k4
            theta_k4 = theta + dt * k3_theta
            omega_k4 = omega + dt * k3_omega
            alpha_k4 = (tau - m_sym * g_sym * L_c_sym * sp.sin(theta_k4) - b_sym * omega_k4) / I_sym
            k4_theta = omega_k4
            k4_omega = alpha_k4

            # Combine
            theta_next = theta + (dt / 6.0) * (k1_theta + 2*k2_theta + 2*k3_theta + k4_theta)
            omega_next = omega + (dt / 6.0) * (k1_omega + 2*k2_omega + 2*k3_omega + k4_omega)

        else:
            raise ValueError(f"Unknown discretization method: {method}")

        self._f_sym = sp.Matrix([theta_next, omega_next])
        self._h_sym = sp.Matrix([theta])  # Measure position only

    def setup_equilibria(self):
        """
        Set up equilibrium points (downward and upward).

        Adds both stable (downward) and unstable (upward) equilibria.
        """
        # Downward equilibrium (stable)
        self.add_equilibrium(
            "downward",
            x_eq=np.array([0.0, 0.0]),
            u_eq=np.array([0.0]),
            verify=True,
            stability="stable",
            notes=f"Stable equilibrium (link hanging down). "
                  f"Natural frequency: ω_n = {self.natural_frequency:.3f} rad/s"
        )

        # Upward equilibrium (unstable)
        self.add_equilibrium(
            "upward",
            x_eq=np.array([np.pi, 0.0]),
            u_eq=np.array([0.0]),
            verify=True,
            stability="unstable",
            notes="Unstable equilibrium (link pointing up). Requires active control."
        )

        self.set_default_equilibrium("downward")

    @property
    def natural_frequency(self) -> float:
        """Natural frequency of small oscillations around downward position [rad/s]."""
        return np.sqrt(self.m * self.g * self.L_c / self.I)

    @property
    def damping_ratio(self) -> float:
        """Damping ratio (dimensionless)."""
        omega_n = self.natural_frequency
        return self.b / (2.0 * self.I * omega_n)

    def compute_gravity_torque(self, theta: float) -> float:
        """
        Compute gravitational torque at angle theta.

        Parameters
        ----------
        theta : float
            Joint angle [rad]

        Returns
        -------
        float
            Gravitational torque [N·m]

        Examples
        --------
        >>> system = DiscreteRobotArm()
        >>> tau_g = system.compute_gravity_torque(np.pi/2)  # Horizontal
        >>> print(f"Gravity torque at horizontal: {tau_g:.2f} N·m")
        """
        return self.m * self.g * self.L_c * np.sin(theta)

    def compute_kinetic_energy(self, omega: float) -> float:
        """
        Compute kinetic energy.

        Parameters
        ----------
        omega : float
            Angular velocity [rad/s]

        Returns
        -------
        float
            Kinetic energy [J]
        """
        return 0.5 * self.I * omega**2

    def compute_potential_energy(self, theta: float) -> float:
        """
        Compute gravitational potential energy.

        Parameters
        ----------
        theta : float
            Joint angle [rad]

        Returns
        -------
        float
            Potential energy [J]

        Notes
        -----
        Reference (zero potential): θ = 0 (downward)
        """
        return self.m * self.g * self.L_c * (1.0 - np.cos(theta))

    def compute_total_energy(self, theta: float, omega: float) -> float:
        """
        Compute total mechanical energy.

        Parameters
        ----------
        theta : float
            Joint angle [rad]
        omega : float
            Angular velocity [rad/s]

        Returns
        -------
        float
            Total energy [J]
        """
        return self.compute_kinetic_energy(omega) + self.compute_potential_energy(theta)

    def generate_quintic_trajectory(
        self,
        theta_0: float,
        theta_f: float,
        T: float,
        n_points: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate smooth quintic polynomial trajectory.

        Quintic ensures:
        - Smooth position, velocity, and acceleration
        - Zero velocity/acceleration at start and end

        Parameters
        ----------
        theta_0 : float
            Initial angle [rad]
        theta_f : float
            Final angle [rad]
        T : float
            Total time [s]
        n_points : int
            Number of points

        Returns
        -------
        tuple
            (theta_ref, omega_ref, alpha_ref) arrays

        Examples
        --------
        >>> system = DiscreteRobotArm(dt=0.01)
        >>> theta_traj, omega_traj, alpha_traj = system.generate_quintic_trajectory(
        ...     theta_0=0.0,
        ...     theta_f=np.pi/2,
        ...     T=5.0,
        ...     n_points=500
        ... )
        """
        t = np.linspace(0, T, n_points)

        # Quintic polynomial coefficients (boundary conditions)
        # θ(0) = θ_0, θ(T) = θ_f
        # ω(0) = 0, ω(T) = 0
        # α(0) = 0, α(T) = 0

        a0 = theta_0
        a1 = 0
        a2 = 0
        a3 = 10 * (theta_f - theta_0) / T**3
        a4 = -15 * (theta_f - theta_0) / T**4
        a5 = 6 * (theta_f - theta_0) / T**5

        # Position
        theta_ref = (a0 + a1*t + a2*t**2 + a3*t**3 + a4*t**4 + a5*t**5)

        # Velocity
        omega_ref = (a1 + 2*a2*t + 3*a3*t**2 + 4*a4*t**3 + 5*a5*t**4)

        # Acceleration
        alpha_ref = (2*a2 + 6*a3*t + 12*a4*t**2 + 20*a5*t**3)

        return theta_ref, omega_ref, alpha_ref

    def design_upward_controller(self) -> np.ndarray:
        """
        Design LQR controller for upward equilibrium stabilization.

        Returns
        -------
        np.ndarray
            LQR gain matrix K for upward position

        Examples
        --------
        >>> system = DiscreteRobotArm()
        >>> K_up = system.design_upward_controller()
        >>> print(f"Upward stabilization gain: {K_up}")
        """
        # Linearize around upward position
        x_up = np.array([np.pi, 0.0])
        u_up = np.array([0.0])

        Ad, Bd = self.linearize(x_up, u_up)

        # Check if unstable (should be)
        eigenvalues = np.linalg.eigvals(Ad)
        if not np.any(np.abs(eigenvalues) > 1):
            import warnings
            warnings.warn("Upward equilibrium appears stable (unexpected!)")

        # Design LQR
        Q = np.diag([100.0, 10.0])  # Penalize angle heavily
        R = np.array([[1.0]])

        lqr_result = self.control.design_lqr(Ad, Bd, Q, R, system_type='discrete')

        return lqr_result['gain']

    # def print_equations(self, simplify: bool = True):
    #     """
    #     Print symbolic equations using discrete-time notation.

    #     Parameters
    #     ----------
    #     simplify : bool
    #         If True, simplify expressions before printing
    #     """
    #     print("=" * 70)
    #     print(f"{self.__class__.__name__} (Discrete-Time, dt={self.dt} s)")
    #     print("=" * 70)
    #     print("Single-Link Robot Arm with Gravity and Friction")
    #     print(f"Discretization Method: {self._method.upper()}")

    #     print("\nPhysical Parameters:")
    #     print(f"  Mass: m = {self.m} kg")
    #     print(f"  Link length: L = {self.L} m")
    #     print(f"  Center of mass: L_c = {self.L_c} m")
    #     print(f"  Moment of inertia: I = {self.I:.4f} kg·m²")
    #     print(f"  Gravity: g = {self.g} m/s²")
    #     print(f"  Friction: b = {self.b} N·m·s/rad")
    #     print(f"  Inertia type: {self._inertia_type}")

    #     print("\nDynamic Characteristics:")
    #     print(f"  Natural frequency: ω_n = {self.natural_frequency:.3f} rad/s ({self.natural_frequency/(2*np.pi):.3f} Hz)")
    #     print(f"  Natural period: T_n = {2*np.pi/self.natural_frequency:.3f} s")
    #     print(f"  Damping ratio: ζ = {self.damping_ratio:.4f}")

    #     damping_class = "underdamped"
    #     if self.damping_ratio >= 1.0:
    #         damping_class = "overdamped"
    #     elif abs(self.damping_ratio - 1.0) < 0.01:
    #         damping_class = "critically damped"
    #     print(f"  Damping classification: {damping_class}")

    #     print(f"\nState: x = [θ, ω] (angle, angular velocity)")
    #     print(f"Control: u = [τ] (applied torque)")
    #     print(f"Dimensions: nx={self.nx}, nu={self.nu}")
    #     print(f"Sampling Period: {self.dt} s")

    #     print("\nContinuous-Time Equation of Motion:")
    #     print("  I·θ̈ = τ - m·g·L_c·sin(θ) - b·θ̇")

    #     print("\nDiscrete-Time Dynamics: x[k+1] = f(x[k], u[k])")
    #     for i, (var, expr) in enumerate(zip(self.state_vars, self._f_sym)):
    #         expr_sub = self.substitute_parameters(expr)
    #         if simplify:
    #             expr_sub = sp.simplify(expr_sub)
    #         label = ['θ[k+1]', 'ω[k+1]'][i]
    #         print(f"  {label} = {expr_sub}")

    #     print("\nOutput: y[k] = h(x[k])")
    #     print(f"  y[k] = θ[k] (position measurement)")

    #     print("\nEquilibria:")
    #     print("  1. Downward (θ=0, ω=0): STABLE")
    #     print("     - Natural resting position")
    #     print("     - Small oscillations decay to zero")
    #     print("  2. Upward (θ=π, ω=0): UNSTABLE")
    #     print("     - Inverted position")
    #     print("     - Requires active stabilization")

    #     print("\nGravity Compensation:")
    #     print("  τ_gravity(θ) = m·g·L_c·sin(θ)")
    #     print(f"  Maximum (horizontal): {self.m * self.g * self.L_c:.2f} N·m")

    #     print("\nPhysical Interpretation:")
    #     print("  - θ: Joint angle (0=down, π=up) [rad]")
    #     print("  - ω: Angular velocity [rad/s]")
    #     print("  - τ: Motor/actuator torque [N·m]")
    #     print("  - Nonlinear due to sin(θ) gravity term")
    #     print("  - Second-order mechanical system")

    #     print("\nTypical Applications:")
    #     print("  - Robot manipulator (single joint)")
    #     print("  - Satellite/spacecraft appendage")
    #     print("  - Crane boom control")
    #     print("  - Rotary inverted pendulum")
    #     print("  - Nonlinear control benchmark")

    #     print("=" * 70)


# Alias for backward compatibility
DiscreteSingleLinkArm = DiscreteRobotArm
DiscreteRobotJoint = DiscreteRobotArm