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
Discrete-time double integrator system.

This module provides discrete-time implementations of the double integrator,
one of the most fundamental systems in control theory. It serves as:
- A pedagogical example for learning discrete-time control
- A benchmark for testing control algorithms
- A simplified model for many mechanical systems (e.g., cart, satellite, robot joint)

The double integrator represents the discretized motion equations of a point mass
under force control, making it applicable to:
- Position control of motors and actuators
- Satellite attitude control (simplified)
- Robotic manipulator joint control (single degree of freedom)
- Automotive cruise control (simplified longitudinal dynamics)
- Quadcopter position control (per axis)
"""

from typing import Optional

import numpy as np
import sympy as sp

from src.systems.base.core.discrete_symbolic_system import DiscreteSymbolicSystem


class DiscreteDoubleIntegrator(DiscreteSymbolicSystem):
    """
    Discrete-time double integrator (position-controlled point mass).

    Physical System:
    ---------------
    The double integrator models the simplest mechanical system: a point mass
    that can be directly controlled by force or acceleration. It represents
    the discretized continuous-time system:

        d²p/dt² = u  (continuous time)

    where p is position and u is acceleration (force/mass). The discrete-time
    approximation depends on the discretization method:

    **Zero-Order Hold (ZOH) - Exact Discretization:**
    Assumes control input is held constant between samples, leading to:
        p[k+1] = p[k] + dt·v[k] + 0.5·dt²·u[k]
        v[k+1] = v[k] + dt·u[k]

    This is the EXACT solution to the continuous system with piecewise-constant
    control, making it the preferred discretization for control design.

    **Forward Euler - First-Order Approximation:**
    Simple but less accurate:
        p[k+1] = p[k] + dt·v[k]
        v[k+1] = v[k] + dt·u[k]

    **Backward Euler - Implicit Method:**
    More stable but requires solving implicit equation:
        p[k+1] = p[k] + dt·v[k+1]
        v[k+1] = v[k] + dt·u[k]

    **Tustin/Trapezoidal - Second-Order Accurate:**
    Bilinear transform, preserves frequency response:
        p[k+1] = p[k] + dt·(v[k] + v[k+1])/2
        v[k+1] = v[k] + dt·u[k]

    This class uses ZOH by default as it's exact and most common in control.

    State Space:
    -----------
    State: x[k] = [p[k], v[k]]
        Position state:
        - p: Position [m] (or [rad] for rotational systems)
          * Can be unbounded: -∞ < p < ∞
          * Typically constrained in practice: p_min ≤ p ≤ p_max
          * Examples: cart position, joint angle, satellite position

        Velocity state:
        - v: Velocity [m/s] (or [rad/s] for rotational)
          * Can be unbounded: -∞ < v < ∞
          * Typically constrained: v_min ≤ v ≤ v_max (actuator limits)
          * Examples: cart speed, joint angular velocity, satellite velocity

    Control: u[k] = [a[k]]
        - a: Acceleration [m/s²] (or angular acceleration [rad/s²])
          * Proportional to applied force: a = F/m
          * Bounded by actuator capacity: a_min ≤ a ≤ a_max
          * Typical range: -10 to +10 m/s² for mechanical systems
          * For satellite: ±0.01 rad/s² (small thrusters)
          * For robot joint: ±100 rad/s² (powerful servos)

    Output: y[k] = [p[k]] or [p[k], v[k]]
        - Position-only measurement (most common):
          y[k] = p[k]
          Examples: encoder, GPS, camera
        - Full state measurement:
          y[k] = [p[k], v[k]]
          Examples: encoder + tachometer, IMU

    Dynamics (Zero-Order Hold):
    ---------------------------
    The ZOH discretization gives EXACT discrete-time dynamics:

        p[k+1] = p[k] + dt·v[k] + 0.5·dt²·u[k]
        v[k+1] = v[k] + dt·u[k]

    **Matrix Form:**
        x[k+1] = Ad·x[k] + Bd·u[k]

    where:
        Ad = [1   dt ]    (State transition matrix)
             [0   1  ]

        Bd = [0.5·dt²]    (Control input matrix)
             [  dt   ]

    **Physical Interpretation:**

    Position update:
    - Current position: p[k]
    - Displacement due to velocity: dt·v[k] (drift term)
    - Displacement due to acceleration: 0.5·dt²·u[k] (control term)
    - New position: p[k+1]

    The 0.5·dt² factor comes from integration:
        ∫₀^dt ∫₀^τ u dσ dτ = 0.5·dt²·u

    Velocity update:
    - Current velocity: v[k]
    - Change due to acceleration: dt·u[k]
    - New velocity: v[k+1]

    **Linearity:**
    The double integrator is LINEAR - superposition applies:
        f(αx₁ + βx₂, αu₁ + βu₂) = αf(x₁, u₁) + βf(x₂, u₂)

    This makes it ideal for:
    - Linear control design (LQR, pole placement)
    - Analytical stability analysis
    - Frequency domain analysis
    - Teaching/learning control theory

    Parameters:
    ----------
    dt : float, default=0.1
        Sampling/discretization time step [s]
        Critical parameter affecting:
        - System dynamics approximation accuracy
        - Control bandwidth and performance
        - Computational requirements
        - Stability margins

        Guidelines:
        - Nyquist: dt < π/ω_max (sample faster than fastest frequency)
        - Shannon: dt < 1/(2·f_bandwidth)
        - Rule of thumb: 10-20 samples per desired closed-loop rise time
        - Typical values: 0.001-0.1 s for mechanical systems

    method : str, default='zoh'
        Discretization method:
        - 'zoh': Zero-order hold (exact, recommended)
        - 'euler': Forward Euler (simple, less accurate)
        - 'backward_euler': Implicit (more stable)
        - 'tustin': Trapezoidal/bilinear (frequency-preserving)

    mass : float, default=1.0
        Mass of the system [kg]
        Only used if control is force (not acceleration)
        Affects control authority: a = F/m
        Typical: 0.1-1000 kg depending on application

    use_force_input : bool, default=False
        If True, control input is force F [N] instead of acceleration
        Dynamics become: x[k+1] = Ad·x[k] + Bd·(u[k]/m)
        Useful for systems where force is the natural actuator variable

    Equilibria:
    ----------
    **Origin (zero position, zero velocity):**
        x_eq = [0, 0]
        u_eq = 0

    This is a MARGINALLY STABLE equilibrium:
    - Not asymptotically stable (doesn't return to origin naturally)
    - Any initial displacement persists forever (constant velocity)
    - Requires feedback control for stabilization

    **Eigenvalue Analysis:**
    The system matrix Ad has eigenvalues:
        λ₁ = 1, λ₂ = 1

    Both eigenvalues are on the unit circle (|λ| = 1), confirming marginal stability.
    - |λ| < 1 would be stable (decays to zero)
    - |λ| > 1 would be unstable (grows unbounded)
    - |λ| = 1 means neutral stability (persists)

    **Any constant velocity is an equilibrium:**
        x_eq = [p*, v*] for any p*, v*
        u_eq = 0

    These form a family of equilibria parameterized by v*. The system naturally
    preserves any velocity when no control is applied (Newton's first law).

    Controllability:
    ---------------
    The discrete double integrator is COMPLETELY CONTROLLABLE from any state
    to any other state in finite time.

    **Controllability Matrix:**
        C = [Bd, Ad·Bd] = [0.5·dt²   0.5·dt³]
                          [  dt      dt²    ]

    **Controllability Test:**
        rank(C) = 2 = nx  ✓ Fully controllable

    **Physical Meaning:**
    - Can reach ANY position and velocity from ANY initial condition
    - Requires at most 2 time steps with unbounded control
    - With bounded control |u| ≤ u_max, requires more time
    - Minimum time to reach (p_target, 0) from (0, 0):
          t_min = √(2·|p_target|/u_max)

    **Bang-Bang Control:**
    Time-optimal control is bang-bang (maximum effort switching once):
    1. Apply u = +u_max until midpoint
    2. Switch to u = -u_max to brake
    3. Arrive at target with v = 0

    Observability:
    -------------
    **Position-only measurement: y[k] = p[k]**
        C = [1  0]

        Observability matrix:
        O = [C   ] = [1   0 ]
            [C·Ad]   [1   dt]

        rank(O) = 2 = nx  ✓ Fully observable

    Can reconstruct full state (position AND velocity) from position
    measurements over time. Velocity is estimated from position differences.

    **Full state measurement: y[k] = [p[k], v[k]]**
        C = [1  0]
            [0  1]

        Trivially observable - direct measurement of all states.

    Control Objectives:
    ------------------
    Common control goals for double integrator:

    1. **Regulation: Drive to origin**
       Goal: x → [0, 0]
       LQR optimal control:
           u[k] = -K·x[k]
           K = [k_p, k_d]  (PD-like gains)

       Closed-loop dynamics:
           x[k+1] = (Ad - Bd·K)·x[k]

       Design K to place eigenvalues inside unit circle.

    2. **Tracking: Follow reference trajectory**
       Goal: x[k] → x_ref[k]
       Control law:
           u[k] = -K·(x[k] - x_ref[k]) + u_ff[k]

       Feedforward term:
           u_ff[k] = (x_ref[k+1] - Ad·x_ref[k]) / Bd

    3. **Point-to-point motion: (0,0) → (p_target, 0)**
       Goal: Move to target position with zero final velocity
       Minimum-time control: bang-bang
       Minimum-energy control: sinusoidal profile

    4. **Setpoint stabilization: p → p_target**
       Goal: Regulate position while allowing nonzero velocity
       Requires integral action or feedforward:
           u[k] = -K·[p[k]-p_target, v[k]] + u_ss

    5. **Velocity regulation: v → v_target**
       Goal: Maintain constant velocity (cruise control)
       Simple proportional control:
           u[k] = k_v·(v_target - v[k])

    State Constraints:
    -----------------
    Physical constraints that must be enforced:

    1. **Position limits: p_min ≤ p[k] ≤ p_max**
       - Physical workspace boundaries
       - Track length, joint angle limits
       - Must be respected by controller
       - Model Predictive Control (MPC) handles naturally

    2. **Velocity limits: v_min ≤ v[k] ≤ v_max**
       - Actuator/motor speed limits
       - Safety considerations
       - Energy/power constraints
       - Typical: |v| ≤ 10 m/s for cart, |v| ≤ 100 rad/s for motor

    3. **Acceleration limits: a_min ≤ u[k] ≤ a_max**
       - Actuator force/torque saturation
       - Most critical practical constraint
       - Causes nonlinearity (saturation)
       - Anti-windup required for integral control
       - Typical: |u| ≤ 10 m/s² for cart, |u| ≤ 100 rad/s² for servo

    4. **Jerk limits: |u[k+1] - u[k]|/dt ≤ jerk_max** (advanced)
       - Smoothness requirement
       - Reduces mechanical wear
       - Passenger comfort (elevators, vehicles)
       - Requires higher-order planning

    Numerical Considerations:
    ------------------------
    **Stability:**
    The ZOH discretization is EXACT - no numerical stability issues
    regardless of dt (assuming arithmetic precision).

    For other methods:
    - Euler: Stable for any dt (system is already stable)
    - Backward Euler: Always stable (A-stable method)
    - Tustin: Stable for any dt

    **Accuracy:**
    - ZOH: Exact for piecewise-constant control
    - Euler: O(dt) error (first-order)
    - Tustin: O(dt²) error (second-order)

    For control design, ZOH is preferred because:
    - Matches real digital control (sample-and-hold)
    - Preserves controllability exactly
    - No approximation error in discrete control law

    **Conditioning:**
    For very small dt, the controllability matrix becomes ill-conditioned:
        cond(C) ≈ O(1/dt²)

    This is rarely a problem in practice since dt is typically 0.001-0.1 s.

    **Computation:**
    State update is trivial - just 2 additions and 3 multiplications per step.
    This makes it ideal for real-time control on embedded systems.

    Control Design Examples:
    -----------------------
    **1. Pole Placement (Deadbeat Control - Fastest Response):**
        Place both eigenvalues at origin → reaches target in 2 steps

        Desired characteristic equation: z² = 0
        Control law: u[k] = -K·x[k]
        Gain: K = [1/dt², 2/dt]

        For dt = 0.1: K = [100, 20]

        Warning: Requires very large control effort!

    **2. LQR (Optimal Quadratic Cost):**
        Minimize: J = Σ (x'·Q·x + u'·R·u)

        Choose Q, R to balance:
        - Q large: Fast response, aggressive control
        - R large: Smooth control, slower response

        Example: Q = diag([10, 1]), R = 1
        Result: K ≈ [3.2, 2.6] for dt = 0.1

        This gives good compromise between speed and control effort.

    **3. PD-Like Control (Intuitive):**
        u[k] = -k_p·p[k] - k_d·v[k]

        - k_p: Position gain (stiffness)
        - k_d: Velocity gain (damping)

        Design rules:
        - Larger k_p → faster position correction
        - Larger k_d → more damping (less overshoot)
        - Critical damping: k_d = 2·√k_p
        - Under-damped: k_d < 2·√k_p (oscillatory)
        - Over-damped: k_d > 2·√k_p (sluggish)

    **4. Model Predictive Control (MPC):**
        Solve optimization at each step:
            min_{u[k:k+N]} Σ (‖x[i]-x_ref[i]‖² + R‖u[i]‖²)
            subject to:
                x[i+1] = Ad·x[i] + Bd·u[i]
                x_min ≤ x[i] ≤ x_max
                u_min ≤ u[i] ≤ u_max

        Advantages:
        - Handles constraints explicitly
        - Optimal preview control
        - Can incorporate feedforward

        Disadvantage: Requires online optimization (computational cost)

    Example Usage:
    -------------
    >>> # Create double integrator with 10 Hz sampling (dt=0.1s)
    >>> system = DiscreteDoubleIntegrator(dt=0.1)
    >>> 
    >>> # Initial condition: 1m displaced, at rest
    >>> x0 = np.array([1.0, 0.0])  # [position, velocity]
    >>> 
    >>> # Open-loop simulation (no control)
    >>> result = system.simulate(
    ...     x0=x0,
    ...     u_sequence=np.zeros(100),  # No control
    ...     n_steps=100
    ... )
    >>> # Result: constant position at p=1.0 (neutral stability)
    >>> 
    >>> # Design LQR controller for regulation
    >>> Ad, Bd = system.linearize(np.zeros(2), np.zeros(1))
    >>> Q = np.diag([10.0, 1.0])  # Care more about position
    >>> R = np.array([[1.0]])
    >>> lqr_result = system.control.design_lqr(
    ...     Ad, Bd, Q, R, system_type='discrete'
    ... )
    >>> K = lqr_result['gain']
    >>> print(f"LQR gain: K = {K}")
    >>> 
    >>> # Closed-loop eigenvalues (should be inside unit circle)
    >>> eigenvalues = lqr_result['closed_loop_eigenvalues']
    >>> print(f"Closed-loop eigenvalues: {eigenvalues}")
    >>> print(f"Stable: {np.all(np.abs(eigenvalues) < 1)}")
    >>> 
    >>> # Simulate with LQR control
    >>> def lqr_controller(x, k):
    ...     return -K @ x
    >>> 
    >>> result_lqr = system.rollout(x0, lqr_controller, n_steps=100)
    >>> 
    >>> # Plot results
    >>> import matplotlib.pyplot as plt
    >>> fig, axes = plt.subplots(3, 1, figsize=(10, 8))
    >>> 
    >>> # Position
    >>> axes[0].plot(result_lqr['time_steps'] * system.dt, 
    ...             result_lqr['states'][:, 0])
    >>> axes[0].set_ylabel('Position [m]')
    >>> axes[0].axhline(0, color='r', linestyle='--', label='Target')
    >>> axes[0].legend()
    >>> axes[0].grid()
    >>> 
    >>> # Velocity
    >>> axes[1].plot(result_lqr['time_steps'] * system.dt,
    ...             result_lqr['states'][:, 1])
    >>> axes[1].set_ylabel('Velocity [m/s]')
    >>> axes[1].axhline(0, color='r', linestyle='--')
    >>> axes[1].grid()
    >>> 
    >>> # Control effort
    >>> axes[2].plot(result_lqr['time_steps'][:-1] * system.dt,
    ...             result_lqr['controls'])
    >>> axes[2].set_ylabel('Acceleration [m/s²]')
    >>> axes[2].set_xlabel('Time [s]')
    >>> axes[2].grid()
    >>> 
    >>> plt.tight_layout()
    >>> plt.show()
    >>> 
    >>> # Point-to-point motion with MPC
    >>> from scipy.optimize import minimize
    >>> 
    >>> def mpc_controller(x, k):
    ...     '''Simple MPC with horizon N=10'''
    ...     N = 10  # Prediction horizon
    ...     p_target = 2.0  # Target position
    ...     u_max = 5.0  # Acceleration limit
    ...     
    ...     def cost(u_seq):
    ...         x_pred = x.copy()
    ...         total_cost = 0.0
    ...         for i in range(N):
    ...             u_i = u_seq[i]
    ...             # State update
    ...             x_pred = Ad @ x_pred + Bd * u_i
    ...             # Stage cost
    ...             error = x_pred[0] - p_target
    ...             total_cost += 10*error**2 + x_pred[1]**2 + 0.1*u_i**2
    ...         return total_cost
    ...     
    ...     # Optimize
    ...     u_init = np.zeros(N)
    ...     bounds = [(-u_max, u_max)] * N
    ...     result = minimize(cost, u_init, bounds=bounds, method='SLSQP')
    ...     
    ...     return result.x[0]  # Return first control action
    >>> 
    >>> result_mpc = system.rollout(x0, mpc_controller, n_steps=50)
    >>> 
    >>> # Check trajectory constraints
    >>> print(f"Max velocity: {np.max(np.abs(result_mpc['states'][:, 1])):.2f} m/s")
    >>> print(f"Max acceleration: {np.max(np.abs(result_mpc['controls'])):.2f} m/s²")
    >>> print(f"Final position error: {result_mpc['states'][-1, 0] - 2.0:.4f} m")

    Physical Insights:
    -----------------
    **Newton's First Law:**
    The double integrator embodies inertia - objects in motion stay in motion
    unless acted upon by a force (control). This is why it's marginally stable:
    any velocity persists forever without control.

    **Minimum-Time Control:**
    To move from (0, 0) to (p_target, 0) in minimum time with |u| ≤ u_max:

    1. Accelerate at u_max for time t₁
    2. Decelerate at -u_max for time t₂
    3. Arrive at target with v = 0

    Solution: t₁ = t₂ = √(|p_target|/u_max)
    Total time: t_total = 2·√(|p_target|/u_max)

    This creates a parabolic position profile (constant acceleration).

    **Energy Considerations:**
    Kinetic energy: E_k = 0.5·m·v²
    Control energy: E_c = ∫ u² dt

    LQR with R > 0 trades off reaching target quickly (minimize E_k)
    versus using less control effort (minimize E_c).

    **Frequency Response:**
    For continuous system (ω = frequency):
        H(s) = 1/s²  →  |H(jω)| = 1/ω²  (-40 dB/decade)

    This is a double integrator in frequency domain:
    - Integrates input twice to get output
    - Phase lag: -180° (always lags by half cycle)
    - Very sensitive to low frequencies (drift)
    - Requires feedback for stability

    **Discrete Frequency Response:**
    For discrete system:
        H(z) = Bd / (z - 1)²

    Resonance at z = 1 (ω = 0) - infinite DC gain without feedback.

    **Relationship to Other Systems:**
    - Cart on frictionless track
    - Satellite position (1D)
    - Inverted pendulum (linearized about upright)
    - Mass-spring system (with k = 0, b = 0)
    - Robot joint (simplified, no gravity or friction)

    Common Pitfalls:
    ---------------
    1. **Forgetting the 0.5 factor:**
       p[k+1] = p[k] + dt·v[k] + 0.5·dt²·u[k]  ✓
       NOT: p[k+1] = p[k] + dt·v[k] + dt²·u[k]  ✗

    2. **Ignoring actuator saturation:**
       Real actuators saturate! Linear control assumes unlimited u.
       Use anti-windup or MPC for constrained control.

    3. **Too small dt for available computation:**
       Faster sampling requires faster computation.
       Ensure control law executes within dt.

    4. **Forgetting neutral stability:**
       Without control, ANY velocity persists forever.
       Always close the loop for practical systems.

    5. **Not handling measurement noise:**
       Real position sensors have noise.
       Use Kalman filter to estimate velocity from noisy position.

    Extensions:
    ----------
    This basic double integrator can be extended to:

    1. **Damped double integrator:**
       Add friction/drag term: v[k+1] = v[k] + dt·(u[k] - b·v[k])
       Now asymptotically stable even without control

    2. **Mass-spring-damper:**
       Add spring force: p[k+1] includes -k·p[k] term
       Creates oscillatory dynamics

    3. **Multi-dimensional:**
       3D motion: x[k] = [p_x, v_x, p_y, v_y, p_z, v_z]
       Each axis is independent double integrator

    4. **Coupled system:**
       Multiple masses connected by springs
       Creates higher-order system

    5. **With disturbances:**
       Add process noise: x[k+1] = Ad·x[k] + Bd·u[k] + w[k]
       Requires stochastic control (LQG)

    See Also:
    --------
    ContinuousDoubleIntegrator : Continuous-time version
    DiscreteOscillator : With spring force added
    DiscreteDampedIntegrator : With friction
    InvertedPendulum : Nonlinear extension (with sin(θ))
    """

    def define_system(
        self,
        dt: float = 0.1,
        method: str = 'zoh',
        mass: float = 1.0,
        use_force_input: bool = False,
    ):
        """
        Define discrete-time double integrator dynamics.

        Parameters
        ----------
        dt : float
            Sampling time step [s]
        method : str
            Discretization method:
            - 'zoh': Zero-order hold (exact, recommended)
            - 'euler': Forward Euler (simple)
            - 'backward_euler': Backward Euler (implicit)
            - 'tustin': Trapezoidal/bilinear transform
        mass : float
            System mass [kg] (only used if use_force_input=True)
        use_force_input : bool
            If True, control input is force [N] instead of acceleration [m/s²]
        """
        # Store configuration
        self._discretization_method = method
        self._mass = mass
        self._use_force_input = use_force_input

        # State variables
        p, v = sp.symbols('p v', real=True)
        u = sp.symbols('u', real=True)

        self.state_vars = [p, v]
        self.control_vars = [u]
        self.output_vars = []
        self._dt = dt
        self.order = 1

        # No symbolic parameters for basic double integrator
        self.parameters = {}

        # Define dynamics based on discretization method
        if method == 'zoh':
            # Zero-order hold (exact discretization)
            # This is the EXACT solution for piecewise-constant control
            if use_force_input:
                # Control is force, divide by mass
                p_next = p + dt * v + 0.5 * dt**2 * u / mass
                v_next = v + dt * u / mass
            else:
                # Control is acceleration (default)
                p_next = p + dt * v + 0.5 * dt**2 * u
                v_next = v + dt * u

        elif method == 'euler':
            # Forward Euler (first-order approximation)
            if use_force_input:
                p_next = p + dt * v
                v_next = v + dt * u / mass
            else:
                p_next = p + dt * v
                v_next = v + dt * u

        elif method == 'backward_euler':
            # Backward Euler (implicit method)
            # For double integrator, can solve explicitly
            if use_force_input:
                # v[k+1] = v[k] + dt * u[k]/m
                # p[k+1] = p[k] + dt * v[k+1] = p[k] + dt*(v[k] + dt*u[k]/m)
                v_next = v + dt * u / mass
                p_next = p + dt * v_next
            else:
                v_next = v + dt * u
                p_next = p + dt * v_next

        elif method == 'tustin':
            # Trapezoidal/bilinear (second-order accurate)
            if use_force_input:
                # v[k+1] = v[k] + dt * u[k]/m
                # p[k+1] = p[k] + dt * (v[k] + v[k+1])/2
                v_next = v + dt * u / mass
                p_next = p + dt * (v + v_next) / 2
            else:
                v_next = v + dt * u
                p_next = p + dt * (v + v_next) / 2

        else:
            raise ValueError(
                f"Unknown discretization method '{method}'. "
                f"Choose from: 'zoh', 'euler', 'backward_euler', 'tustin'"
            )

        self._f_sym = sp.Matrix([p_next, v_next])
        self._h_sym = sp.Matrix([p])  # Measure position only

    def setup_equilibria(self):
        """
        Set up equilibrium points.

        Adds the origin (zero position, zero velocity) as default equilibrium.
        Note: ANY constant velocity is technically an equilibrium with u=0,
        but we only add the origin for simplicity.
        """
        # Origin is automatically added to systems and set as default
        # by default
        pass

    def compute_system_matrices(self) -> tuple:
        """
        Compute discrete-time state-space matrices Ad, Bd.

        Returns
        -------
        tuple
            (Ad, Bd) where:
            - Ad: State transition matrix (2×2)
            - Bd: Control input matrix (2×1)

        Examples
        --------
        >>> system = DiscreteDoubleIntegrator(dt=0.1)
        >>> Ad, Bd = system.compute_system_matrices()
        >>> print(f"Ad =\\n{Ad}")
        >>> print(f"Bd =\\n{Bd}")

        Notes
        -----
        For ZOH discretization:
            Ad = [1   dt]
                 [0   1 ]

            Bd = [0.5*dt²]
                 [  dt   ]
        """
        # Get linearization at origin (system is linear, so same everywhere)
        Ad, Bd = self.linearize(np.zeros(2), np.zeros(1))
        return Ad, Bd

    def compute_controllability_matrix(self) -> np.ndarray:
        """
        Compute controllability matrix C = [Bd, Ad·Bd].

        Returns
        -------
        np.ndarray
            Controllability matrix (2×2)

        Examples
        --------
        >>> system = DiscreteDoubleIntegrator(dt=0.1)
        >>> C = system.compute_controllability_matrix()
        >>> rank = np.linalg.matrix_rank(C)
        >>> print(f"Controllability matrix rank: {rank}")
        >>> print(f"System is controllable: {rank == 2}")

        Notes
        -----
        The double integrator is always completely controllable.
        rank(C) = 2 for any dt > 0.
        """
        Ad, Bd = self.compute_system_matrices()
        C = self.analysis.controllability(Ad, Bd)
        return C

    def compute_observability_matrix(self, C_output: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute observability matrix O = [C; C·Ad].

        Parameters
        ----------
        C_output : Optional[np.ndarray]
            Output matrix (1×2 or 2×2)
            If None, uses position-only measurement: C = [1, 0]

        Returns
        -------
        np.ndarray
            Observability matrix

        Examples
        --------
        >>> system = DiscreteDoubleIntegrator(dt=0.1)
        >>> # Position-only measurement
        >>> O = system.compute_observability_matrix()
        >>> rank = np.linalg.matrix_rank(O)
        >>> print(f"Observable: {rank == 2}")
        >>>
        >>> # Full state measurement
        >>> C_full = np.eye(2)
        >>> O_full = system.compute_observability_matrix(C_full)

        Notes
        -----
        For position-only measurement C = [1, 0]:
            O = [1    0 ]
                [1   dt ]

        rank(O) = 2 → fully observable (can estimate velocity from position)
        """
        Ad, _ = self.compute_system_matrices()

        if C_output is None:
            # Default: position-only measurement
            C_output = np.array([[1.0, 0.0]])

        O = self.analysis.observability(Ad, C_output)
        return O

    def design_deadbeat_controller(self) -> np.ndarray:
        """
        Design deadbeat controller (fastest possible response).

        Places both closed-loop eigenvalues at origin, giving finite-time
        settling in 2 steps.

        Returns
        -------
        np.ndarray
            Deadbeat control gain K (1×2)

        Examples
        --------
        >>> system = DiscreteDoubleIntegrator(dt=0.1)
        >>> K_db = system.design_deadbeat_controller()
        >>> print(f"Deadbeat gain: K = {K_db}")
        >>>
        >>> # Simulate
        >>> def deadbeat_control(x, k):
        ...     return -K_db @ x
        >>>
        >>> x0 = np.array([1.0, 0.0])
        >>> result = system.rollout(x0, deadbeat_control, n_steps=10)
        >>>
        >>> # Check settling time
        >>> settling_idx = np.where(np.abs(result['states'][:, 0]) < 0.01)[0]
        >>> print(f"Settled in {settling_idx[0]} steps")

        Notes
        -----
        Deadbeat control gives fastest response but requires VERY large
        control effort. The gain magnitude scales as 1/dt², so smaller
        time steps require exponentially larger control.

        For dt = 0.1: K ≈ [100, 20]
        For dt = 0.01: K ≈ [10000, 200]

        Use with caution in systems with actuator limits!
        """
        dt = self._dt

        # Desired closed-loop poles at origin: (z-0)² = z² = 0
        # Characteristic equation: det(zI - (Ad - Bd·K)) = 0
        # This gives: z² - (2 - K₁·dt² - K₂·dt)·z + (1 - K₁·dt² - K₂·dt) = 0
        # For deadbeat: z² = 0, so coefficients of z¹ and z⁰ must be zero

        # Solution (for ZOH discretization):
        if self._discretization_method == 'zoh':
            K = np.array([[1.0 / dt**2, 2.0 / dt]])
        else:
            # For other methods, use pole placement
            from scipy.signal import place_poles

            Ad, Bd = self.compute_system_matrices()
            desired_poles = [0.0, 0.0]
            result = place_poles(Ad, Bd, desired_poles)
            K = result.gain_matrix

        return K

    def design_pd_controller(
        self,
        k_p: float,
        k_d: Optional[float] = None,
        damping_ratio: float = 1.0,
    ) -> np.ndarray:
        """
        Design PD (proportional-derivative) controller.

        Control law: u[k] = -k_p·p[k] - k_d·v[k]

        Parameters
        ----------
        k_p : float
            Position gain (stiffness)
            Larger k_p → faster position response
        k_d : Optional[float]
            Velocity gain (damping)
            If None, computed from damping_ratio
        damping_ratio : float
            Desired damping ratio (only used if k_d is None)
            - ζ = 1.0: Critical damping (no overshoot)
            - ζ < 1.0: Under-damped (oscillatory)
            - ζ > 1.0: Over-damped (sluggish)

        Returns
        -------
        np.ndarray
            PD gain matrix K = [k_p, k_d]

        Examples
        --------
        >>> system = DiscreteDoubleIntegrator(dt=0.1)
        >>>
        >>> # Critical damping (no overshoot)
        >>> K_critical = system.design_pd_controller(k_p=10.0, damping_ratio=1.0)
        >>> print(f"Critical damping: K = {K_critical}")
        >>>
        >>> # Under-damped (faster but oscillatory)
        >>> K_underdamped = system.design_pd_controller(k_p=10.0, damping_ratio=0.5)
        >>>
        >>> # Over-damped (slow but smooth)
        >>> K_overdamped = system.design_pd_controller(k_p=10.0, damping_ratio=2.0)
        >>>
        >>> # Manual k_d specification
        >>> K_manual = system.design_pd_controller(k_p=10.0, k_d=5.0)

        Notes
        -----
        The relationship between continuous and discrete PD gains:
            Continuous: k_d = 2·ζ·√k_p (for ζ = 1: k_d = 2√k_p)
            Discrete: Similar but depends on dt

        For discrete double integrator, the closed-loop poles are:
            z = 1 - k_p·dt²/2 ± √((k_p·dt²/2)² - k_d·dt·k_p·dt²/2)
        """
        if k_d is None:
            # Compute k_d from damping ratio
            # For critical damping in continuous time: k_d = 2·√k_p
            # In discrete time, this is approximate
            k_d = 2.0 * damping_ratio * np.sqrt(k_p)

        K = np.array([[k_p, k_d]])
        return K

    def compute_minimum_time(
        self,
        p_target: float,
        u_max: float,
    ) -> float:
        """
        Compute minimum time to reach target position from origin with bounded control.

        For bang-bang control: |u| ≤ u_max

        Parameters
        ----------
        p_target : float
            Target position [m]
        u_max : float
            Maximum acceleration [m/s²]

        Returns
        -------
        float
            Minimum time [s]

        Examples
        --------
        >>> system = DiscreteDoubleIntegrator(dt=0.1)
        >>> t_min = system.compute_minimum_time(p_target=10.0, u_max=2.0)
        >>> print(f"Minimum time to reach 10m with 2 m/s² limit: {t_min:.2f} s")

        Notes
        -----
        Minimum-time trajectory is bang-bang:
        1. Accelerate at u_max until midpoint
        2. Decelerate at -u_max until target
        3. Arrive with v = 0

        Time to reach target:
            t_min = 2·√(|p_target|/u_max)

        This assumes starting from rest: (p, v) = (0, 0) → (p_target, 0)
        """
        t_min = 2.0 * np.sqrt(np.abs(p_target) / u_max)
        return t_min

    def generate_minimum_time_control(
        self,
        p_target: float,
        u_max: float,
        n_steps: int,
    ) -> np.ndarray:
        """
        Generate minimum-time control sequence (bang-bang).

        Parameters
        ----------
        p_target : float
            Target position [m]
        u_max : float
            Maximum acceleration [m/s²]
        n_steps : int
            Number of discrete time steps

        Returns
        -------
        np.ndarray
            Control sequence (n_steps,)

        Examples
        --------
        >>> system = DiscreteDoubleIntegrator(dt=0.1)
        >>> u_seq = system.generate_minimum_time_control(
        ...     p_target=10.0,
        ...     u_max=2.0,
        ...     n_steps=100
        ... )
        >>>
        >>> # Simulate with bang-bang control
        >>> result = system.simulate(
        ...     x0=np.zeros(2),
        ...     u_sequence=u_seq,
        ...     n_steps=100
        ... )
        >>>
        >>> print(f"Final position: {result['states'][-1, 0]:.2f} m")
        >>> print(f"Final velocity: {result['states'][-1, 1]:.2f} m/s")

        Notes
        -----
        The switching time occurs at the midpoint:
            t_switch = √(|p_target|/u_max)
        """
        # Compute switching time
        t_switch = np.sqrt(np.abs(p_target) / u_max)
        k_switch = int(t_switch / self._dt)

        # Generate bang-bang sequence
        u_seq = np.ones(n_steps) * np.sign(p_target) * u_max
        u_seq[k_switch:] = -np.sign(p_target) * u_max

        return u_seq

    def generate_minimum_energy_control(
        self,
        p_target: float,
        t_final: float,
        n_steps: int,
    ) -> np.ndarray:
        """
        Generate minimum-energy control sequence for given time.

        Minimizes ∫ u² dt subject to reaching target at t_final.

        Parameters
        ----------
        p_target : float
            Target position [m]
        t_final : float
            Final time [s]
        n_steps : int
            Number of discrete steps

        Returns
        -------
        np.ndarray
            Control sequence (n_steps,)

        Examples
        --------
        >>> system = DiscreteDoubleIntegrator(dt=0.1)
        >>> u_seq = system.generate_minimum_energy_control(
        ...     p_target=10.0,
        ...     t_final=10.0,
        ...     n_steps=100
        ... )

        Notes
        -----
        For double integrator, minimum-energy control is sinusoidal:
            u(t) = A·sin(πt/t_final)

        where A is chosen to reach p_target at t_final.
        """
        # Time vector
        t = np.linspace(0, t_final, n_steps)

        # Sinusoidal profile
        # From calculus of variations: u(t) = A·sin(πt/t_final)
        # where A = 4·p_target·π² / t_final²
        A = 4.0 * p_target * np.pi**2 / t_final**2
        u_seq = A * np.sin(np.pi * t / t_final)

        return u_seq

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
    #     print(f"Discretization Method: {self._discretization_method.upper()}")
    #     if self._use_force_input:
    #         print(f"Mass: {self._mass} kg")
    #         print("Control Input: Force [N]")
    #     else:
    #         print("Control Input: Acceleration [m/s²]")

    #     print(f"\nState: x = [p, v]")
    #     print(f"Control: u = [a]")
    #     print(f"Dimensions: nx={self.nx}, nu={self.nu}")
    #     print(f"Sampling Period: {self.dt} s")

    #     print("\nDynamics: x[k+1] = f(x[k], u[k])")
    #     state_labels = ['p[k+1]', 'v[k+1]']
    #     for label, expr in zip(state_labels, self._f_sym):
    #         expr_sub = self.substitute_parameters(expr)
    #         if simplify:
    #             expr_sub = sp.simplify(expr_sub)
    #         print(f"  {label} = {expr_sub}")

    #     print("\nOutput: y[k] = h(x[k])")
    #     print(f"  y[k] = {self._h_sym[0]}")

    #     # Display system matrices
    #     Ad, Bd = self.compute_system_matrices()
    #     print("\nState-Space Form: x[k+1] = Ad·x[k] + Bd·u[k]")
    #     print(f"Ad = {Ad}")
    #     print(f"Bd = {Bd}")

    #     # Eigenvalues
    #     eigenvalues = np.linalg.eigvals(Ad)
    #     print(f"\nOpen-Loop Eigenvalues: {eigenvalues}")
    #     print(f"Stability: Marginally stable (|λ| = 1)")

    #     # Controllability
    #     C = self.compute_controllability_matrix()
    #     rank_C = np.linalg.matrix_rank(C)
    #     print(f"\nControllability: rank(C) = {rank_C} (fully controllable: {rank_C == 2})")

    #     # Observability
    #     O = self.compute_observability_matrix()
    #     rank_O = np.linalg.matrix_rank(O)
    #     print(f"Observability (pos only): rank(O) = {rank_O} (fully observable: {rank_O == 2})")

    #     print("\nPhysical Interpretation:")
    #     print("  - p[k]: Position [m]")
    #     print("  - v[k]: Velocity [m/s]")
    #     print("  - u[k]: Acceleration [m/s²]" if not self._use_force_input else "  - u[k]: Force [N]")

    #     print("\nKey Properties:")
    #     print("  - Linear system (superposition applies)")
    #     print("  - Marginally stable (requires feedback control)")
    #     print("  - Fully controllable (can reach any state)")
    #     print("  - Fully observable (from position measurements)")
    #     print("  - Represents inertial dynamics (Newton's laws)")

    #     print("\nTypical Applications:")
    #     print("  - Cart position control")
    #     print("  - Satellite/spacecraft positioning")
    #     print("  - Robot joint control")
    #     print("  - Servo motor control")
    #     print("  - Any point mass under force control")

    #     print("=" * 70)


class DiscreteDoubleIntegratorWithForce(DiscreteSymbolicSystem):
    """
    Discrete double integrator with explicit force input and mass.

    This is a convenience alias that creates a DiscreteDoubleIntegrator
    with use_force_input=True, making the mass parameter explicit in
    the dynamics.

    Dynamics: p[k+1] = p[k] + dt·v[k] + 0.5·dt²·F[k]/m
              v[k+1] = v[k] + dt·F[k]/m

    See DiscreteDoubleIntegrator for full documentation.
    """

    def define_system(self, dt: float = 0.1, mass: float = 1.0, method: str = 'zoh'):
        """
        Define double integrator with force input.

        Parameters
        ----------
        dt : float
            Sampling time [s]
        mass : float
            System mass [kg]
        method : str
            Discretization method
        """
        # Delegate to parent with force input enabled
        parent = DiscreteDoubleIntegrator(dt=dt, method=method, mass=mass, use_force_input=True)

        # Copy all attributes
        self.state_vars = parent.state_vars
        self.control_vars = parent.control_vars
        self.output_vars = parent.output_vars
        self._f_sym = parent._f_sym
        self._h_sym = parent._h_sym
        self.parameters = parent.parameters
        self._dt = parent._dt
        self.order = parent.order

        # Store configuration
        self._discretization_method = method
        self._mass = mass
        self._use_force_input = True

    # def print_equations(self, simplify: bool = True):
    #     """Print equations with explicit mass term."""
    #     print("=" * 70)
    #     print(f"{self.__class__.__name__} (Discrete-Time, dt={self.dt} s)")
    #     print("=" * 70)
    #     print(f"Mass: m = {self._mass} kg")
    #     print("Control Input: Force [N]")
    #     print("\nNewton's Second Law: F = m·a")
    #     print(f"Acceleration: a = F/m = u/{self._mass}")

    #     # Call parent print method
    #     super().print_equations(simplify)