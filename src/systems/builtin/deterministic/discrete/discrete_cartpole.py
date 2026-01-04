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
Discrete-time Cart-Pole System - Classic Underactuated Control Problem.

This module provides discrete-time models of the cart-pole (inverted pendulum
on cart), one of the most important benchmark systems in control theory and
reinforcement learning. It serves as:
- The canonical example of underactuated mechanical systems
- A benchmark for nonlinear control design (swing-up and stabilization)
- The standard testbed for reinforcement learning algorithms
- An illustration of non-minimum phase systems
- A model for balancing and stabilization problems

The cart-pole represents:
- Inverted pendulum balancing (Segway, balance robots)
- Rocket/missile stabilization during launch
- Ship stabilization with ballast control
- Human balance and postural control
- Crane anti-swing control (inverted dynamics)

This is the discrete-time version, appropriate for:
- Digital control implementation
- Reinforcement learning (OpenAI Gym environment)
- Model predictive control (MPC)
- Real-time embedded control systems
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import sympy as sp

from src.systems.base.core.discrete_symbolic_system import DiscreteSymbolicSystem


class DiscreteCartPole(DiscreteSymbolicSystem):
    """
    Discrete-time cart-pole (inverted pendulum on cart) system.

    Physical System:
    ---------------
    A cart of mass M moving horizontally on a frictionless track, with a
    pole (pendulum) of mass m and length L attached via a frictionless pivot.
    The cart is actuated by a horizontal force F, while the pole is free to
    rotate under gravity.

    **Mechanical Configuration:**
```
                  ● m (pole mass)
                 /|
                / | L (pole length)
               /  |
              /   |
         ====●====●==== Cart (mass M)
         ←---F    |
         _____|____|_____ Track (frictionless)
              x (cart position)
              θ (pole angle from vertical)
```

    **Key Features:**
    - **Underactuated:** 4 states (x, ẋ, θ, θ̇), 1 control (F)
    - **Nonlinear:** Coupled nonlinear dynamics
    - **Non-minimum phase:** Zero at s = +√(g/L) (RHP for inverted)
    - **Unstable:** Upright equilibrium is unstable without control

    **This is THE classic underactuated system!**

    **Equations of Motion (Lagrangian Mechanics):**
    Using Euler-Lagrange equations with generalized coordinates q = [x, θ]:

    Cart equation:
        (M + m)·ẍ + m·L·θ̈·cos(θ) - m·L·θ̇²·sin(θ) = F

    Pole equation:
        m·L²·θ̈ + m·L·ẍ·cos(θ) - m·g·L·sin(θ) = 0

    Solving for accelerations (ẍ, θ̈):
        
        θ̈ = [g·sin(θ) - cos(θ)·(F + m·L·θ̇²·sin(θ))/(M + m)] / 
             [L·(4/3 - m·cos²(θ)/(M + m))]

        ẍ = [F + m·L·(θ̇²·sin(θ) - θ̈·cos(θ))] / (M + m)

    **Discrete-Time Dynamics:**
    Zero-order hold discretization with numerical integration.

    State Space:
    -----------
    State: x[k] = [x[k], ẋ[k], θ[k], θ̇[k]]
        Cart position:
        - x: Horizontal position of cart [m]
          * Unbounded in principle: -∞ < x < ∞
          * Typically constrained: |x| ≤ x_max (track length)
          * x = 0: Reference position (often center of track)

        Cart velocity:
        - ẋ: Horizontal velocity of cart [m/s]
          * Can be positive or negative
          * Typical: |ẋ| < 5 m/s

        Pole angle:
        - θ: Angle from upward vertical [rad]
          * θ = 0: Pole upright (inverted, unstable)
          * θ = π: Pole hanging down (stable)
          * θ = ±π/2: Pole horizontal
          * For control: typically keep |θ| < π/4 (±45°)

        Pole angular velocity:
        - θ̇: Rate of change of pole angle [rad/s]
          * Positive: Falling to the right
          * Negative: Falling to the left
          * Large |θ̇| indicates imminent fall

    Control: u[k] = [F[k]]
        Applied force on cart:
        - F: Horizontal force [N]
          * F > 0: Push cart to the right
          * F < 0: Push cart to the left
          * Bounded: |F| ≤ F_max (actuator limit)
          * Typical: |F| ≤ 50 N

    **Note on Convention:**
    Some formulations measure θ from downward vertical (θ = 0 down, θ = π up).
    This implementation uses θ = 0 for UPRIGHT (inverted), which is more
    common in control literature and RL benchmarks.

    Output: y[k] = [x[k], θ[k]] or full state
        Typical measurements:
        - x: Cart position (encoder, potentiometer)
        - θ: Pole angle (encoder at pivot, IMU on pole)
        - Velocities: Computed from differences or direct measurement

    Dynamics (Physical Interpretation):
    -----------------------------------
    **Coupling Between Cart and Pole:**
    The dynamics are strongly coupled:
    1. Moving cart → creates inertial force on pole (affects θ̈)
    2. Pole falling → pulls cart in that direction (affects ẍ)
    3. Centrifugal force: θ̇² term in cart equation
    4. Gravitational torque: sin(θ) term in pole equation

    **Underactuation:**
    - 4 state variables (x, ẋ, θ, θ̇)
    - 1 control input (F)
    - Cannot independently control cart and pole
    - Must use internal dynamics (coupling) to control pole

    **Non-minimum Phase:**
    If we try to move cart to the right (F > 0):
    - Initially, pole falls to the LEFT (opposite direction!)
    - This is non-minimum phase behavior
    - Complicates control design
    - Cannot simply use output feedback

    **Zero Dynamics:**
    If we constrain x = constant (cart position fixed):
    - Remaining dynamics are pole oscillations
    - Zero dynamics unstable (pole falls down)
    - This is why system is non-minimum phase

    Parameters:
    ----------
    M : float, default=1.0
        Mass of cart [kg]
        - Typical: 0.5-5.0 kg
        - Heavier cart: More inertia, slower response
        - Lighter cart: Faster response, more sensitive to pole

    m : float, default=0.1
        Mass of pole [kg]
        - Typical: 0.05-0.5 kg (much less than cart)
        - Ratio m/M affects coupling strength
        - Larger m: Stronger coupling, harder to balance

    L : float, default=0.5
        Half-length of pole [m]
        - Distance from pivot to center of mass
        - Full length = 2L
        - Typical: 0.3-1.0 m
        - Longer pole: Slower dynamics, easier to balance
        - Shorter pole: Faster dynamics, harder to balance

    g : float, default=9.81
        Gravitational acceleration [m/s²]
        - Earth: 9.81
        - Moon: 1.62 (much easier to balance!)
        - Mars: 3.71

    b_cart : float, default=0.1
        Cart friction coefficient [N·s/m]
        - Linear viscous friction on cart
        - Typical: 0.0-1.0
        - b = 0: Frictionless (ideal)

    b_pole : float, default=0.0
        Pole friction coefficient [N·m·s/rad]
        - Friction at pivot joint
        - Typically negligible: b_pole ≈ 0
        - Can include for realism

    dt : float, default=0.02
        Sampling period [s]
        - Typical control rates: 20-100 Hz (dt = 0.01-0.05 s)
        - Must be fast enough to stabilize unstable pole
        - Rule of thumb: dt < 0.1/ω_pole where ω_pole ~ √(g/L)

    Equilibria:
    ----------
    **Four Equilibrium Points:**

    1. **Downward, Center (x=0, ẋ=0, θ=π, θ̇=0):**
       - STABLE: Pole hanging down, cart centered
       - Natural resting position
       - Easy to maintain (no control needed)
       - Lowest potential energy

    2. **Upward, Center (x=0, ẋ=0, θ=0, θ̇=0):**
       - UNSTABLE: Pole inverted, cart centered
       - Target for balancing control
       - Requires continuous active control
       - Highest potential energy
       - Classic inverted pendulum problem

    3. **Downward, Offset (x≠0, ẋ=0, θ=π, θ̇=0):**
       - Family of stable equilibria
       - Cart at any position, pole down
       - Less interesting for control

    4. **Upward, Offset (x≠0, ẋ=0, θ=0, θ̇=0):**
       - Family of unstable equilibria
       - Requires control to maintain both position and balance

    **Linearization at Upright:**
    Around (0, 0, 0, 0), with small angles (sin(θ) ≈ θ, cos(θ) ≈ 1):

        θ̈ ≈ [(M + m)·g·θ - F] / [L·(M + m) - m·L]
        ẍ ≈ [F - m·g·θ] / (M + m)

    This gives a LINEAR system for control design.

    Controllability:
    ---------------
    **The cart-pole is CONTROLLABLE:**
    Despite being underactuated (4 states, 1 input), the system is completely
    controllable. This is verified by checking rank of controllability matrix.

    **Why controllable despite underactuation?**
    - The coupling between cart and pole creates internal dynamics
    - Moving cart affects pole angle (and vice versa)
    - Can exploit this coupling to control both

    **Lie Bracket Analysis:**
    The system satisfies sufficient conditions for controllability:
    - Control vector field and its Lie brackets span ℝ⁴

    **Practical Controllability:**
    With bounded control |F| ≤ F_max:
    - Cannot balance pole beyond certain initial angles
    - Cannot swing up from downward if F_max too small
    - Reachable set depends on F_max and initial conditions

    Observability:
    -------------
    **Full State Measurement:**
    Trivially observable (measure all 4 states directly).

    **Partial Observations:**
    - (x, θ) only: Observable (can reconstruct velocities)
    - θ only: NOT observable (cart position decoupled)
    - x only: NOT observable (pole angle decoupled)

    Control Objectives:
    ------------------
    **1. Balance Control (Stabilization at Upright):**
       Goal: Stabilize at (0, 0, 0, 0) from small perturbations
       Methods:
       - LQR (linearized around upright)
       - Pole placement
       - PID control
       - Sliding mode control
       Challenge: Unstable equilibrium, requires fast feedback

    **2. Swing-Up Control:**
       Goal: Move from downward (θ=π) to upright (θ=0)
       Methods:
       - Energy-based control
       - Partial feedback linearization
       - Trajectory optimization
       - Reinforcement learning
       Challenge: Large control effort, nonlinear dynamics

    **3. Swing-Up + Balance (Full Problem):**
       Goal: Swing up from any initial condition, then stabilize
       Method: Switch between swing-up and balance controllers
       - Swing-up when |θ - π| > threshold
       - Balance when |θ| < threshold
       Challenge: Smooth switching, mode transitions

    **4. Trajectory Tracking:**
       Goal: Follow reference trajectory (x_ref(t), θ_ref(t))
       Application: Moving while balancing
       Example: Segway following path

    **5. Disturbance Rejection:**
       Goal: Maintain balance despite pushes/bumps
       Methods: Robust control, H∞, adaptive control

    State Constraints:
    -----------------
    **1. Cart Position Limits: |x[k]| ≤ x_max**
       - Physical track length
       - Typical: x_max = 2-5 m
       - Failure if cart hits end
       - Must be enforced by controller

    **2. Pole Angle Limits: |θ[k]| ≤ θ_max**
       - Balancing region: θ_max ≈ π/12 (15°)
       - Beyond this, typically considered "fallen"
       - RL episode terminates if exceeded

    **3. Velocity Limits:**
       |ẋ[k]| ≤ ẋ_max: Cart velocity limit
       |θ̇[k]| ≤ θ̇_max: Pole angular velocity limit

    **4. Force Limits: |F[k]| ≤ F_max**
       - Actuator saturation (most critical)
       - Typical: F_max = 10-50 N
       - Affects controllability region

    **5. Success Criteria (Reinforcement Learning):**
       - |x| < x_max (typically 2.4 m)
       - |θ| < θ_max (typically 12° ≈ 0.2 rad)
       - Episode fails if either violated

    Numerical Considerations:
    ------------------------
    **Singularity at θ = ±π/2:**
    The equations have division by [1 - m·cos²(θ)/(M+m)]
    - Denominator → 0 as θ → ±π/2 for certain mass ratios
    - Typically not an issue (pole doesn't reach horizontal when balancing)

    **Angle Wrapping:**
    For swing-up control, θ may exceed ±π:
    - Can wrap to [-π, π] or leave unwrapped
    - Unwrapped better for tracking rotations
    - sin(θ), cos(θ) automatically handle periodicity

    Example Usage:
    -------------
    >>> # Create cart-pole with OpenAI Gym-like parameters
    >>> system = DiscreteCartPole(
    ...     M=1.0,        # 1 kg cart
    ...     m=0.1,        # 100g pole
    ...     L=0.5,        # 1m pole (half-length)
    ...     g=9.81,
    ...     dt=0.02,      # 50 Hz
    ...     method='rk4'
    ... )
    >>> 
    >>> print(f"Mass ratio m/M: {system.m/system.M:.2f}")
    >>> print(f"Pole natural frequency: {system.pole_frequency:.2f} rad/s")
    >>> 
    >>> # Initial condition: upright with small perturbation
    >>> x0_upright = np.array([0.0, 0.0, 0.1, 0.0])  # Small angle
    >>> 
    >>> # Design LQR controller for balancing
    >>> x_eq = np.array([0.0, 0.0, 0.0, 0.0])
    >>> u_eq = np.array([0.0])
    >>> 
    >>> Ad, Bd = system.linearize(x_eq, u_eq)
    >>> 
    >>> # LQR weights
    >>> Q = np.diag([1.0, 0.1, 100.0, 10.0])  # Care most about pole angle
    >>> R = np.array([[0.01]])
    >>> 
    >>> lqr_result = system.control.design_lqr(
    ...     Ad, Bd, Q, R, system_type='discrete'
    ... )
    >>> K = lqr_result['gain']
    >>> print(f"\nLQR gain: K = {K}")
    >>> 
    >>> # Check closed-loop stability
    >>> eigenvalues = lqr_result['closed_loop_eigenvalues']
    >>> print(f"Closed-loop eigenvalues: {eigenvalues}")
    >>> print(f"All stable: {np.all(np.abs(eigenvalues) < 1)}")
    >>> 
    >>> # Simulate balancing
    >>> def lqr_balance(x, k):
    ...     return -K @ x
    >>> 
    >>> result_balance = system.rollout(
    ...     x0=x0_upright,
    ...     policy=lqr_balance,
    ...     n_steps=500
    ... )
    >>> 
    >>> # Plot balancing performance
    >>> import plotly.graph_objects as go
    >>> from plotly.subplots import make_subplots
    >>> 
    >>> fig = make_subplots(
    ...     rows=3, cols=1,
    ...     subplot_titles=['Cart Position', 'Pole Angle', 'Control Force']
    ... )
    >>> 
    >>> t = result_balance['time_steps'] * system.dt
    >>> 
    >>> # Cart position
    >>> fig.add_trace(go.Scatter(x=t, y=result_balance['states'][:, 0], 
    ...                          name='x'), row=1, col=1)
    >>> fig.add_hline(y=0, line_dash='dash', row=1, col=1)
    >>> 
    >>> # Pole angle (convert to degrees)
    >>> fig.add_trace(go.Scatter(x=t, y=np.rad2deg(result_balance['states'][:, 2]),
    ...                          name='θ'), row=2, col=1)
    >>> fig.add_hline(y=0, line_dash='dash', row=2, col=1)
    >>> fig.add_hline(y=12, line_dash='dot', line_color='red', row=2, col=1)
    >>> fig.add_hline(y=-12, line_dash='dot', line_color='red', row=2, col=1)
    >>> 
    >>> # Control force
    >>> fig.add_trace(go.Scatter(x=t[:-1], y=result_balance['controls'][:, 0],
    ...                          name='F'), row=3, col=1)
    >>> 
    >>> fig.update_xaxes(title_text='Time [s]', row=3, col=1)
    >>> fig.update_yaxes(title_text='x [m]', row=1, col=1)
    >>> fig.update_yaxes(title_text='θ [deg]', row=2, col=1)
    >>> fig.update_yaxes(title_text='F [N]', row=3, col=1)
    >>> fig.update_layout(height=900, showlegend=False, 
    ...                   title_text='Cart-Pole Balancing Control (LQR)')
    >>> fig.show()
    >>> 
    >>> # Energy-based swing-up control
    >>> x0_down = np.array([0.0, 0.0, np.pi, 0.0])  # Pole hanging down
    >>> 
    >>> def swing_up_controller(x, k):
    ...     x_pos, x_vel, theta, theta_vel = x
    ...     
    ...     # Current energy
    ...     E = system.compute_total_energy(theta, theta_vel)
    ...     
    ...     # Target energy (upright position)
    ...     E_target = system.m * system.g * system.L
    ...     
    ...     # Energy error
    ...     E_error = E - E_target
    ...     
    ...     # Switch to LQR when close to upright
    ...     if abs(theta) < 0.3 and abs(theta_vel) < 1.0:
    ...         # Stabilize using LQR
    ...         return -K @ x
    ...     else:
    ...         # Energy pumping
    ...         k_swing = 10.0
    ...         # Pump energy: apply force in direction that increases energy
    ...         return k_swing * E_error * np.sign(theta_vel * np.cos(theta))
    >>> 
    >>> result_swing = system.rollout(
    ...     x0=x0_down,
    ...     policy=swing_up_controller,
    ...     n_steps=1000
    ... )
    >>> 
    >>> # Visualize swing-up
    >>> fig_swing = system.plot_animation_frames(result_swing, frames=[0, 200, 400, 600, 800])
    >>> fig_swing.show()
    >>> 
    >>> # Reinforcement learning evaluation
    >>> def check_episode_success(states, x_threshold=2.4, theta_threshold=0.2):
    ...     """Check if episode stayed within success criteria."""
    ...     x_ok = np.all(np.abs(states[:, 0]) <= x_threshold)
    ...     theta_ok = np.all(np.abs(states[:, 2]) <= theta_threshold)
    ...     return x_ok and theta_ok
    >>> 
    >>> success = check_episode_success(result_balance['states'])
    >>> print(f"\nBalancing success: {success}")
    >>> 
    >>> # Compare different mass ratios
    >>> mass_ratios = [0.05, 0.1, 0.2, 0.5]
    >>> 
    >>> fig_compare = go.Figure()
    >>> 
    >>> for m_ratio in mass_ratios:
    ...     sys_temp = DiscreteCartPole(M=1.0, m=m_ratio, L=0.5, dt=0.02)
    ...     
    ...     # Design LQR
    ...     Ad_temp, Bd_temp = sys_temp.linearize(x_eq, u_eq)
    ...     lqr_temp = sys_temp.control.design_lqr(Ad_temp, Bd_temp, Q, R,
    ...                                             system_type='discrete')
    ...     K_temp = lqr_temp['gain']
    ...     
    ...     # Simulate
    ...     result_temp = sys_temp.rollout(
    ...         x0=x0_upright,
    ...         policy=lambda x, k, K_=K_temp: -K_ @ x,
    ...         n_steps=500
    ...     )
    ...     
    ...     fig_compare.add_trace(go.Scatter(
    ...         x=t,
    ...         y=np.rad2deg(result_temp['states'][:, 2]),
    ...         name=f'm/M = {m_ratio:.2f}'
    ...     ))
    >>> 
    >>> fig_compare.update_layout(
    ...     title='Effect of Mass Ratio on Balancing',
    ...     xaxis_title='Time [s]',
    ...     yaxis_title='Pole Angle [deg]'
    ... )
    >>> fig_compare.show()

    Physical Insights:
    -----------------
    **Why Is It Hard to Balance?**
    The upright position is an unstable saddle point:
    - Any small disturbance causes pole to fall
    - Falls faster as it tilts more (sin(θ) nonlinearity)
    - Must apply corrective force before falling too far
    - Human balancing uses same principles!

    **The Balancing Strategy:**
    To balance a falling pole:
    1. Pole starts falling (θ ≠ 0)
    2. Move cart in direction pole is falling
    3. This creates inertial force opposing fall
    4. Pole rights itself due to cart acceleration
    5. Stop cart before oscillation grows

    This is exactly what LQR does automatically!

    **Energy Perspective for Swing-Up:**
    Total energy: E = KE_cart + KE_pole + PE_pole

    To swing up:
    1. Pump energy by pushing cart back and forth
    2. Exploit coupling: cart motion → pole motion
    3. When E ≈ E_target, pole near vertical
    4. Switch to stabilizing controller

    **Non-Minimum Phase Behavior:**
    Initial response is in "wrong" direction:
    - Push cart right (F > 0)
    - Pole initially tilts LEFT
    - Then cart movement brings pole back
    - This is why simple controllers fail

    **Mass Ratio Effects:**
    The ratio m/M affects difficulty:
    - Small m/M (light pole): Easier to balance, less coupling
    - Large m/M (heavy pole): Harder to balance, strong coupling
    - m/M = 1: Equal masses, maximum challenge
    - Typical: m/M ≈ 0.1

    **Pole Length Effects:**
    Longer pole (larger L):
    - Slower natural frequency: ω ~ √(g/L)
    - Easier to balance (more reaction time)
    - Larger moment of inertia
    - But requires more space

    Common Pitfalls:
    ---------------
    1. **Wrong angle convention:**
       Some use θ = 0 for DOWN, others for UP
       Check carefully! This implementation: θ = 0 is UPRIGHT

    2. **Forgetting non-minimum phase:**
       Cannot use simple output feedback
       Need state feedback (or observer)

    3. **Too slow sampling:**
       Unstable pole requires fast control
       dt > 0.1 s typically too slow

    4. **Linearization over-reach:**
       Linear controller only works near upright
       Breaks down for |θ| > π/4
       Need nonlinear control for swing-up

    5. **Ignoring constraints:**
       Real system has |x| < x_max, |F| < F_max
       Must handle saturation and limits

    6. **Switching controller discontinuities:**
       Abrupt switch between swing-up and balance
       Can cause chattering or instability
       Use smooth transitions (hysteresis, blending)

    Extensions:
    ----------
    1. **Rotary inverted pendulum (Furuta):**
       Cart moves in circle instead of line
       Different coupling dynamics

    2. **Double inverted pendulum:**
       Two poles in series (acrobot)
       Much harder to control

    3. **3D cart-pole:**
       Pole can fall in any direction
       Spherical coordinates needed

    4. **Flexible pole:**
       Pole bends (elastic)
       Infinite-dimensional system

    5. **With obstacles:**
       Must avoid obstacles while balancing
       Combines planning and control

    6. **Multi-cart cooperation:**
       Two carts balancing shared pole
       Coordination problem

    Reinforcement Learning Context:
    ------------------------------
    The cart-pole is THE standard RL benchmark:
    - **OpenAI Gym:** 'CartPole-v1' environment
    - **State:** [x, ẋ, θ, θ̇]
    - **Action:** Discrete {left, right} or continuous force
    - **Reward:** +1 per timestep balanced
    - **Done:** |x| > 2.4 or |θ| > 0.2 rad
    - **Success:** Average reward > 195 over 100 episodes

    **Why Popular for RL?**
    - Simple enough to learn quickly
    - Complex enough to be interesting
    - Unstable → requires intelligent control
    - Fast simulation
    - Well-understood theoretically (can validate)

    See Also:
    --------
    DiscretePendulum : Simpler (just pole, no cart)
    CartPole : Continuous-time version
    """

    def define_system(
        self,
        M: float = 1.0,
        m: float = 0.1,
        L: float = 0.5,
        g: float = 9.81,
        b_cart: float = 0.1,
        b_pole: float = 0.0,
        dt: float = 0.02,
    ):
        """
        Define discrete-time cart-pole dynamics.

        Parameters
        ----------
        M : float
            Cart mass [kg]
        m : float
            Pole mass [kg]
        L : float
            Pole half-length [m] (distance from pivot to center of mass)
        g : float
            Gravity [m/s²]
        b_cart : float
            Cart friction [N·s/m]
        b_pole : float
            Pole friction [N·m·s/rad]
        dt : float
            Sampling period [s]
        method : str
            Discretization method ('euler', 'rk4')
        """
        # Store parameters
        self.M = M
        self.m = m
        self.L = L
        self.g = g
        self.b_cart = b_cart
        self.b_pole = b_pole

        # Derived quantities
        self.total_mass = M + m
        self.pole_mass_moment = m * L

        # State variables (cart position, cart velocity, pole angle, pole angular velocity)
        x, x_dot, theta, theta_dot = sp.symbols('x x_dot theta theta_dot', real=True)
        F = sp.symbols('F', real=True)

        # Symbolic parameters
        M_sym, m_sym, L_sym, g_sym = sp.symbols('M m L g', real=True, positive=True)
        b_cart_sym, b_pole_sym = sp.symbols('b_cart b_pole', real=True, nonnegative=True)

        self.state_vars = [x, x_dot, theta, theta_dot]
        self.control_vars = [F]
        self._dt = dt
        self.order = 1

        self.parameters = {
            M_sym: M,
            m_sym: m,
            L_sym: L,
            g_sym: g,
            b_cart_sym: b_cart,
            b_pole_sym: b_pole,
        }

        # Equations of motion (derived from Lagrangian)
        # See: Åström & Murray, "Feedback Systems"
        
        # Temporary variables for readability
        sin_theta = sp.sin(theta)
        cos_theta = sp.cos(theta)
        
        # Common denominator term
        denom = M_sym + m_sym - m_sym * cos_theta**2

        # Angular acceleration of pole
        theta_ddot = (
            (M_sym + m_sym) * g_sym * sin_theta
            - cos_theta * (F - b_cart_sym * x_dot + m_sym * L_sym * theta_dot**2 * sin_theta)
            - b_pole_sym * theta_dot / L_sym
        ) / (L_sym * denom)

        # Linear acceleration of cart
        x_ddot = (
            F - b_cart_sym * x_dot
            + m_sym * L_sym * (theta_dot**2 * sin_theta - theta_ddot * cos_theta)
        ) / (M_sym + m_sym)

        self._f_sym = sp.Matrix([x, x_dot, theta, theta_dot])
        self._h_sym = sp.Matrix([x, theta])  # Measure position and angle
        self.output_vars = []

    def setup_equilibria(self):
        """Set up equilibrium points."""
        # Upright (inverted) - unstable
        self.add_equilibrium(
            "upright",
            x_eq=np.array([0.0, 0.0, 0.0, 0.0]),
            u_eq=np.array([0.0]),
            verify=True,
            stability="unstable",
            notes="Unstable inverted equilibrium. Target for balancing control."
        )

        # Downward (hanging) - stable
        self.add_equilibrium(
            "downward",
            x_eq=np.array([0.0, 0.0, np.pi, 0.0]),
            u_eq=np.array([0.0]),
            verify=True,
            stability="stable",
            notes="Stable hanging equilibrium. Natural resting position."
        )

        self.set_default_equilibrium("upright")

    @property
    def pole_frequency(self) -> float:
        """Natural frequency of pole oscillations √(g/L) [rad/s]."""
        return np.sqrt(self.g / self.L)

    @property
    def pole_period(self) -> float:
        """Period of small pole oscillations [s]."""
        return 2.0 * np.pi / self.pole_frequency

    def compute_total_energy(self, theta: float, theta_dot: float) -> float:
        """
        Compute total mechanical energy of pole.

        Parameters
        ----------
        theta : float
            Pole angle [rad]
        theta_dot : float
            Pole angular velocity [rad/s]

        Returns
        -------
        float
            Total energy [J]

        Notes
        -----
        Cart kinetic energy ignored (only pole energy for swing-up).
        """
        # Pole kinetic energy
        KE = 0.5 * self.m * (self.L * theta_dot)**2

        # Pole potential energy (reference at θ = π, hanging down)
        PE = self.m * self.g * self.L * (np.cos(theta) + 1)

        return KE + PE

    # def print_equations(self, simplify: bool = False):
    #     """Print symbolic equations."""
    #     print("=" * 70)
    #     print(f"{self.__class__.__name__} (Discrete-Time, dt={self.dt} s)")
    #     print("=" * 70)
    #     print("Cart-Pole (Inverted Pendulum on Cart)")
    #     print("Classic Underactuated Control Benchmark")
    #     print(f"Discretization Method: {self._method.upper()}")

    #     print("\nPhysical Parameters:")
    #     print(f"  Cart mass: M = {self.M} kg")
    #     print(f"  Pole mass: m = {self.m} kg")
    #     print(f"  Mass ratio: m/M = {self.m/self.M:.3f}")
    #     print(f"  Pole half-length: L = {self.L} m (full length = {2*self.L} m)")
    #     print(f"  Gravity: g = {self.g} m/s²")
    #     print(f"  Cart friction: b_cart = {self.b_cart} N·s/m")
    #     print(f"  Pole friction: b_pole = {self.b_pole} N·m·s/rad")

    #     print("\nDynamic Characteristics:")
    #     print(f"  Pole frequency: ω_pole = {self.pole_frequency:.3f} rad/s")
    #     print(f"  Pole period: T_pole = {self.pole_period:.3f} s")
    #     print(f"  Total mass: M_total = {self.total_mass} kg")

    #     print(f"\nState: x = [x, ẋ, θ, θ̇]")
    #     print(f"  - x: Cart position [m]")
    #     print(f"  - ẋ: Cart velocity [m/s]")
    #     print(f"  - θ: Pole angle from upward vertical [rad]")
    #     print(f"  - θ̇: Pole angular velocity [rad/s]")
    #     print(f"\nControl: u = [F] (horizontal force on cart) [N]")
    #     print(f"Dimensions: nx={self.nx}, nu={self.nu}")
    #     print(f"Sampling Period: {self.dt} s ({1/self.dt:.0f} Hz)")

    #     print("\nContinuous-Time Equations of Motion:")
    #     print("  (M + m)·ẍ + m·L·θ̈·cos(θ) - m·L·θ̇²·sin(θ) = F - b_cart·ẋ")
    #     print("  m·L²·θ̈ + m·L·ẍ·cos(θ) - m·g·L·sin(θ) = -b_pole·θ̇")

    #     print("\nSolved for accelerations:")
    #     print("  θ̈ = [(M+m)·g·sin(θ) - cos(θ)·(F - b_cart·ẋ + m·L·θ̇²·sin(θ)) - b_pole·θ̇/L]")
    #     print("      / [L·(M + m - m·cos²(θ))]")
    #     print("  ẍ = [F - b_cart·ẋ + m·L·(θ̇²·sin(θ) - θ̈·cos(θ))] / (M + m)")

    #     print("\nDiscrete-Time Dynamics: x[k+1] = f(x[k], u[k])")
    #     print("  (Symbolic expressions omitted for brevity - very complex)")
    #     print(f"  Method: {self._method}")

    #     print("\nEquilibria:")
    #     print("  1. Upright (x=0, ẋ=0, θ=0, θ̇=0): UNSTABLE")
    #     print("     - Inverted pendulum configuration")
    #     print("     - Target for balancing control")
    #     print("     - Linearization gives unstable pole (eigenvalue > 1)")
    #     print("  2. Downward (x=0, ẋ=0, θ=π, θ̇=0): STABLE")
    #     print("     - Natural hanging position")
    #     print("     - Starting point for swing-up control")

    #     print("\nKey Properties:")
    #     print("  - UNDERACTUATED: 4 states, 1 control")
    #     print("  - NONLINEAR: Coupled trigonometric dynamics")
    #     print("  - NON-MINIMUM PHASE: Zero in RHP")
    #     print("  - UNSTABLE: Requires feedback for balancing")
    #     print("  - CONTROLLABLE: Despite underactuation")

    #     print("\nControl Challenges:")
    #     print("  1. Balancing: Stabilize at upright (LQR, pole placement)")
    #     print("  2. Swing-up: Move from down to up (energy-based, trajectory)")
    #     print("  3. Combined: Swing-up then balance (mode switching)")
    #     print("  4. Constraints: Track limits, force limits")

    #     print("\nReinforcement Learning Benchmark:")
    #     print("  - OpenAI Gym 'CartPole-v1' environment")
    #     print("  - Success: Keep |x| < 2.4m, |θ| < 12°")
    #     print("  - Reward: +1 per timestep balanced")
    #     print("  - Solved: Average reward ≥ 195 over 100 episodes")

    #     print("\nApplications:")
    #     print("  - Segway/balance robots")
    #     print("  - Inverted pendulum stabilization")
    #     print("  - Rocket/missile control during launch")
    #     print("  - Human posture control modeling")
    #     print("  - RL algorithm benchmarking")

    #     print("=" * 70)


# Aliases
CartPole = DiscreteCartPole
InvertedPendulumOnCart = DiscreteCartPole