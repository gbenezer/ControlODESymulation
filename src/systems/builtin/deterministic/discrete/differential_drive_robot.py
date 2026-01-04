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
Discrete-time Differential Drive Robot - Nonholonomic Mobile Robotics.

This module provides discrete-time models of a differential drive mobile robot,
one of the most common configurations in mobile robotics. It serves as:
- A canonical example of nonholonomic constraints (rolling without slipping)
- A model for wheeled mobile robots (TurtleBot, warehouse robots, etc.)
- An illustration of underactuated systems with drift
- A benchmark for path planning and trajectory tracking
- A testbed for nonlinear control in robotics

The differential drive robot represents:
- Two-wheeled robots (left and right wheels independently controlled)
- Wheelchair kinematics
- Segway/balance robot base (without balancing dynamics)
- Tank-like vehicles (skid steering)
- Vacuum cleaning robots, delivery robots, AGVs

This is the discrete-time kinematic model, appropriate for:
- Path planning algorithms (RRT, A*, Dijkstra)
- Model predictive control (MPC)
- Trajectory tracking controllers
- Real-time implementation on embedded systems
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import sympy as sp

from src.systems.base.core.discrete_symbolic_system import DiscreteSymbolicSystem


class DifferentialDriveRobot(DiscreteSymbolicSystem):
    """
    Discrete-time differential drive mobile robot with nonholonomic constraints.

    Physical System:
    ---------------
    A mobile robot with two independently driven wheels on a common axle,
    plus one or more passive casters for stability. Control is achieved by
    varying the velocities of the left and right wheels.

    **Mechanical Configuration:**
```
              Front
                ↑
         ╔══════╪══════╗
         ║      │      ║
      L  ║    ──●──    ║  θ (heading)
         ║   /  │  \   ║
         ╚══●═══╧═══●══╝
           Left    Right
           wheel   wheel
           
         ←─── 2L ───→
         (wheel separation)
```

    The robot configuration:
    - Two drive wheels separated by distance 2L (wheelbase)
    - Each wheel can be controlled independently
    - Robot body is rigid (no deformation)
    - Wheels roll without slipping (nonholonomic constraint)
    - Position (x, y) in global frame
    - Orientation θ (heading angle)

    **Nonholonomic Constraint:**
    The key feature is the "rolling without slipping" constraint:
    The robot CANNOT move sideways (perpendicular to heading).

    Mathematically:
        ẋ·sin(θ) - ẏ·cos(θ) = 0

    This means the robot has only 2 velocity degrees of freedom (forward, turn)
    despite having 3 configuration variables (x, y, θ).

    **Kinematic Model:**
    The robot velocity is determined by wheel velocities:
        v = (v_L + v_R) / 2  (forward velocity)
        ω = (v_R - v_L) / (2L)  (angular velocity)

    Where:
        v_L: Left wheel velocity [m/s]
        v_R: Right wheel velocity [m/s]
        L: Half of wheelbase [m]

    The robot motion in global frame:
        ẋ = v·cos(θ)
        ẏ = v·sin(θ)
        θ̇ = ω

    **Discrete-Time Dynamics:**
    Exact discretization (assuming constant wheel velocities):
        x[k+1] = x[k] + dt·v[k]·cos(θ[k])
        y[k+1] = y[k] + dt·v[k]·sin(θ[k])
        θ[k+1] = θ[k] + dt·ω[k]

    Or directly in terms of wheel velocities:
        x[k+1] = x[k] + dt·(v_L + v_R)/2·cos(θ[k])
        y[k+1] = y[k] + dt·(v_L + v_R)/2·sin(θ[k])
        θ[k+1] = θ[k] + dt·(v_R - v_L)/(2L)

    State Space:
    -----------
    State: x[k] = [x[k], y[k], θ[k]]
        Position coordinates (global frame):
        - x: Position along world x-axis [m]
          * Unbounded: -∞ < x < ∞
          * Typically confined to workspace
          * Can be arbitrarily large

        - y: Position along world y-axis [m]
          * Unbounded: -∞ < y < ∞
          * Typically confined to workspace
          * Can be arbitrarily large

        Heading angle:
        - θ: Orientation/heading angle [rad]
          * θ = 0: Facing along +x axis (East)
          * θ = π/2: Facing along +y axis (North)
          * θ = π: Facing along -x axis (West)
          * θ = 3π/2 or -π/2: Facing along -y axis (South)
          * Periodic: θ and θ + 2π are equivalent
          * Can be wrapped to [-π, π] or [0, 2π]

    Control: u[k] = [v_L[k], v_R[k]]
        Wheel velocities (independent control):
        - v_L: Left wheel velocity [m/s]
          * v_L > 0: Left wheel forward
          * v_L < 0: Left wheel backward
          * Bounded: |v_L| ≤ v_max

        - v_R: Right wheel velocity [m/s]
          * v_R > 0: Right wheel forward
          * v_R < 0: Right wheel backward
          * Bounded: |v_R| ≤ v_max

    **Control Modes (derived from v_L, v_R):**

    1. **Straight line (v_L = v_R = v):**
       - Robot moves forward/backward
       - No turning
       - Fastest point-to-point if aligned

    2. **Pure rotation (v_L = -v_R):**
       - Robot spins in place
       - No translation
       - Used for reorientation

    3. **Arc motion (v_L ≠ v_R):**
       - Robot follows circular arc
       - Radius: R = L·(v_L + v_R)/(v_R - v_L)
       - Most general motion

    4. **Stop (v_L = v_R = 0):**
       - Robot stationary
       - Maintain position

    Output: y[k] = [x[k], y[k], θ[k]]
        - Full state measurement (typical for indoor robots)
        - x, y from wheel odometry or external localization
        - θ from IMU/gyroscope or magnetometer
        - In practice: odometry has drift, requires sensor fusion

    Dynamics (Kinematic Model):
    ---------------------------
    The discrete dynamics are:

        x[k+1] = x[k] + dt·(v_L[k] + v_R[k])/2·cos(θ[k])
        y[k+1] = y[k] + dt·(v_L[k] + v_R[k])/2·sin(θ[k])
        θ[k+1] = θ[k] + dt·(v_R[k] - v_L[k])/(2L)

    **Alternative Control Parametrization:**
    Can also use (v, ω) as control:
        v = (v_L + v_R) / 2  (linear velocity)
        ω = (v_R - v_L) / (2L)  (angular velocity)

    Then:
        x[k+1] = x[k] + dt·v[k]·cos(θ[k])
        y[k+1] = y[k] + dt·v[k]·sin(θ[k])
        θ[k+1] = θ[k] + dt·ω[k]

    This is the **unicycle model** - mathematically equivalent.

    **Motion Primitives:**

    Forward:
        v_L = v_R = v_desired
        Result: Straight line in current heading direction

    Backward:
        v_L = v_R = -v_desired
        Result: Straight line backward

    Turn in place (left):
        v_L = -v_turn, v_R = +v_turn
        Result: Counterclockwise rotation about center

    Turn in place (right):
        v_L = +v_turn, v_R = -v_turn
        Result: Clockwise rotation about center

    Arc (general):
        v_L ≠ v_R
        Result: Circular arc with radius R

    Parameters:
    ----------
    L : float, default=0.5
        Half-width of wheelbase [m]
        - Distance from center to each wheel
        - Total wheelbase = 2L
        - Typical values: 0.1-0.5 m
        - Larger L → larger turning radius
        - Affects maneuverability

    dt : float, default=0.1
        Sampling period [s]
        - Control update rate
        - Sensor measurement period
        - Typical: 0.01-0.1 s (10-100 Hz)
        - Affects trajectory smoothness

    max_wheel_velocity : Optional[float]
        Maximum wheel velocity [m/s]
        - Physical motor speed limit
        - Typical: 0.5-5.0 m/s
        - Affects reachable velocities

    control_mode : str, default='wheel_velocities'
        Control input type:
        - 'wheel_velocities': u = [v_L, v_R]
        - 'unicycle': u = [v, ω] (linear and angular velocity)

    Equilibria:
    ----------
    **No traditional equilibria exist!**

    Unlike typical dynamical systems, the differential drive robot has
    NO equilibrium points in the usual sense:
    - Every state (x, y, θ) with v_L = v_R = 0 is an "equilibrium"
    - These form a 3D manifold (entire configuration space)
    - The robot can stop at any position and orientation

    This is characteristic of kinematic systems without restoring forces.

    **Configuration Space:**
    The set of all possible poses: SE(2) = ℝ² × S¹
    - ℝ²: Position (x, y)
    - S¹: Orientation θ (circle)

    Controllability (Nonholonomic):
    -------------------------------
    **Controllability vs Nonholonomic Constraints:**

    The differential drive robot is:
    - **Controllable:** Can reach any configuration (x, y, θ) from any other
    - **Nonholonomic:** Cannot move in all directions instantaneously

    This seems contradictory but isn't:
    - Can reach any point, but must follow curved path
    - Cannot drive sideways or diagonally
    - Requires sequences of forward/turn motions

    **Chow's Theorem (Controllability of Nonholonomic Systems):**
    The system is controllable if Lie brackets of control vector fields
    span the tangent space. For differential drive:
    
    Vector fields:
        f₁ = [cos(θ), sin(θ), 0]ᵀ  (forward motion)
        f₂ = [0, 0, 1]ᵀ  (pure rotation)
    
    Lie bracket:
        [f₁, f₂] = [sin(θ), -cos(θ), 0]ᵀ  (sideways motion!)

    This shows sideways motion achievable via sequences, proving controllability.

    **Minimum Maneuvers:**
    To reach any (x_f, y_f, θ_f) from (x₀, y₀, θ₀):
    - At most 3 motion primitives needed (Dubins paths)
    - Types: CSC (Curve-Straight-Curve), CCC (Curve-Curve-Curve)
    - Minimum turning radius: R_min = v_max·L/v_max = L

    **Reeds-Shepp Paths (with reverse):**
    If backward motion allowed, even more efficient paths exist.

    Observability:
    -------------
    **Full state measurement:**
    Trivially observable - directly measure (x, y, θ).

    **Partial observations:**
    - GPS only: (x, y) measured, θ must be estimated
    - Compass only: θ measured, position must be integrated
    - Wheel odometry: Integrate wheel velocities (drift over time)

    **Sensor Fusion:**
    In practice, combine multiple sensors:
    - Wheel encoders (dead reckoning)
    - IMU (gyroscope for θ̇, accelerometer for validation)
    - GPS/external localization (absolute position)
    - Kalman filter or particle filter for fusion

    Control Objectives:
    ------------------
    **1. Point-to-Point Navigation:**
       Goal: Move from (x₀, y₀, θ₀) to (x_f, y_f, θ_f)
       Methods:
       - Feedback linearization (near goal)
       - Pure pursuit (follow carrot point)
       - Potential field methods
       - Sampling-based planning (RRT)

    **2. Trajectory Tracking:**
       Goal: Follow reference path (x_ref(t), y_ref(t), θ_ref(t))
       Control laws:
       - Kinematic controller (backstepping)
       - Input-output linearization
       - Model predictive control (MPC)
       Applications: Warehouse navigation, assembly lines

    **3. Path Following:**
       Goal: Converge to and follow geometric path
       - Less strict than tracking (no time parametrization)
       - Pure pursuit, Stanley controller, LOS guidance
       - Applications: Lane following, contour tracking

    **4. Formation Control:**
       Goal: Maintain relative positions with other robots
       - Leader-follower, virtual structure, behavioral
       - Applications: Multi-robot coordination, swarms

    **5. Obstacle Avoidance:**
       Goal: Reach target while avoiding obstacles
       - Dynamic window approach (DWA)
       - Velocity obstacles
       - Artificial potential fields

    State Constraints:
    -----------------
    **1. Workspace Boundaries:**
       (x, y) ∈ Workspace ⊂ ℝ²
       - Room boundaries, allowed regions
       - Avoid obstacles, walls, hazards

    **2. Velocity Limits:**
       |v_L|, |v_R| ≤ v_max
       - Motor speed limitations
       - Typical: v_max = 0.5-2.0 m/s

    **3. Acceleration Limits:**
       |v_L[k+1] - v_L[k]|/dt ≤ a_max
       |v_R[k+1] - v_R[k]|/dt ≤ a_max
       - Motor acceleration capacity
       - Prevents wheel slip
       - Typical: a_max = 1-5 m/s²

    **4. Curvature/Turning Radius:**
       |ω/v| ≤ 1/R_min
       - Minimum turning radius: R_min
       - Depends on wheelbase and velocity limits
       - R_min = L·v_max/v_max = L (at maximum speeds)

    **5. Nonholonomic Constraint:**
       Velocity perpendicular to heading = 0
       - Cannot move sideways
       - Must follow curved paths
       - Restricts instantaneous motion

    Numerical Considerations:
    ------------------------

    **Angle Wrapping:**
    Heading angle θ should be wrapped to [-π, π]:
        θ_wrapped = atan2(sin(θ), cos(θ))

    Prevents numerical overflow and simplifies control.

    **Odometry Drift:**
    Integrating wheel velocities accumulates errors:
    - Wheel slip (especially on turns)
    - Uneven surfaces
    - Wheelbase calibration errors
    - Typical drift: 5-10% of distance traveled

    Prevention: Use external localization (GPS, SLAM, markers).

    **Singularities:**
    - No kinematic singularities (well-defined everywhere)
    - Control singularity at v = 0 for some controllers
    - Planning singularity for backward motion (if not allowed)

    Example Usage:
    -------------
    >>> # Create differential drive robot
    >>> robot = DifferentialDriveRobot(
    ...     L=0.15,          # 30cm wheelbase
    ...     dt=0.1,          # 10 Hz control
    ...     max_wheel_velocity=1.0
    ... )
    >>> 
    >>> # Initial pose: origin, facing East
    >>> x0 = np.array([0.0, 0.0, 0.0])  # [x, y, θ]
    >>> 
    >>> # Drive straight forward
    >>> v_forward = 0.5  # 0.5 m/s
    >>> u_forward = np.array([v_forward, v_forward])
    >>> 
    >>> result_straight = robot.simulate(
    ...     x0=x0,
    ...     u_sequence=u_forward,  # Constant velocity
    ...     n_steps=50
    ... )
    >>> 
    >>> print(f"Final position: ({result_straight['states'][-1, 0]:.2f}, "
    ...       f"{result_straight['states'][-1, 1]:.2f})")
    >>> # Should be approximately (2.5, 0) after 5 seconds at 0.5 m/s
    >>> 
    >>> # Turn in place (spin)
    >>> omega_spin = 1.0  # 1 rad/s
    >>> v_L_spin = -omega_spin * robot.L
    >>> v_R_spin = +omega_spin * robot.L
    >>> u_spin = np.array([v_L_spin, v_R_spin])
    >>> 
    >>> result_spin = robot.simulate(
    ...     x0=x0,
    ...     u_sequence=u_spin,
    ...     n_steps=31  # π/ω seconds ≈ 3.14s
    >>> )
    >>> 
    >>> print(f"Final heading: {result_spin['states'][-1, 2]:.2f} rad")
    >>> # Should be approximately π (180° turn)
    >>> 
    >>> # Circular arc motion
    >>> v_circle = 0.5  # m/s
    >>> omega_circle = 0.5  # rad/s
    >>> radius = v_circle / omega_circle
    >>> print(f"Circle radius: {radius:.2f} m")
    >>> 
    >>> v_L_circle = v_circle - omega_circle * robot.L
    >>> v_R_circle = v_circle + omega_circle * robot.L
    >>> u_circle = np.array([v_L_circle, v_R_circle])
    >>> 
    >>> result_circle = robot.simulate(
    ...     x0=x0,
    ...     u_sequence=u_circle,
    ...     n_steps=100
    ... )
    >>> 
    >>> # Plot trajectory
    >>> import plotly.graph_objects as go
    >>> fig = go.Figure()
    >>> fig.add_trace(go.Scatter(
    ...     x=result_circle['states'][:, 0],
    ...     y=result_circle['states'][:, 1],
    ...     mode='lines+markers',
    ...     name='Robot path',
    ...     marker=dict(size=3)
    ... ))
    >>> 
    >>> # Add orientation arrows
    >>> for i in range(0, len(result_circle['states']), 10):
    ...     x_i, y_i, theta_i = result_circle['states'][i]
    ...     dx = 0.1 * np.cos(theta_i)
    ...     dy = 0.1 * np.sin(theta_i)
    ...     fig.add_annotation(
    ...         x=x_i, y=y_i,
    ...         ax=x_i + dx, ay=y_i + dy,
    ...         xref='x', yref='y',
    ...         axref='x', ayref='y',
    ...         showarrow=True,
    ...         arrowhead=2,
    ...         arrowsize=1,
    ...         arrowwidth=2,
    ...         arrowcolor='red'
    ...     )
    >>> 
    >>> fig.update_layout(
    ...     title='Differential Drive Robot Trajectory',
    ...     xaxis_title='x [m]',
    ...     yaxis_title='y [m]',
    ...     yaxis_scaleanchor='x',
    ...     width=800,
    ...     height=800
    ... )
    >>> fig.show()
    >>> 
    >>> # Point-to-point controller
    >>> def goto_controller(x, k):
    ...     # Current pose
    ...     x_curr, y_curr, theta_curr = x
    ...     
    ...     # Target
    ...     x_target, y_target, theta_target = 5.0, 3.0, np.pi/4
    ...     
    ...     # Distance and angle to target
    ...     dx = x_target - x_curr
    ...     dy = y_target - y_curr
    ...     distance = np.sqrt(dx**2 + dy**2)
    ...     angle_to_target = np.arctan2(dy, dx)
    ...     
    ...     # Heading error
    ...     theta_error = angle_to_target - theta_curr
    ...     # Wrap to [-π, π]
    ...     theta_error = np.arctan2(np.sin(theta_error), np.cos(theta_error))
    ...     
    ...     # If far from target, move toward it
    ...     if distance > 0.1:
    ...         # Proportional control
    ...         v = min(0.5, 2.0 * distance)  # Slow down near target
    ...         omega = 3.0 * theta_error  # Turn to face target
    ...     else:
    ...         # Near target, just adjust orientation
    ...         v = 0.0
    ...         final_theta_error = theta_target - theta_curr
    ...         final_theta_error = np.arctan2(np.sin(final_theta_error), 
    ...                                        np.cos(final_theta_error))
    ...         omega = 2.0 * final_theta_error
    ...     
    ...     # Convert to wheel velocities
    ...     v_L = v - omega * robot.L
    ...     v_R = v + omega * robot.L
    ...     
    ...     return np.array([v_L, v_R])
    >>> 
    >>> result_goto = robot.rollout(
    ...     x0=np.array([0.0, 0.0, 0.0]),
    ...     policy=goto_controller,
    ...     n_steps=200
    ... )
    >>> 
    >>> # Visualize with goal
    >>> fig_goto = robot.plot_trajectory_with_robot(result_goto)
    >>> fig_goto.add_trace(go.Scatter(
    ...     x=[5.0], y=[3.0],
    ...     mode='markers',
    ...     marker=dict(size=20, color='red', symbol='star'),
    ...     name='Goal'
    ... ))
    >>> fig_goto.show()
    >>> 
    >>> def path_following_controller(x, k):
    ...     # Current pose
    ...     x_curr, y_curr, theta_curr = x
    ...     
    ...     # Current time
    ...     t = k * robot.dt
    ...     
    ...     # Desired position on path
    ...     x_des, y_des = lemniscate_path(0.5 * t)
    ...     
    ...     # Lookahead point (pure pursuit)
    ...     lookahead = 0.3  # meters
    ...     t_lookahead = t + lookahead / 0.5
    ...     x_look, y_look = lemniscate_path(0.5 * t_lookahead)
    ...     
    ...     # Control toward lookahead
    ...     dx = x_look - x_curr
    ...     dy = y_look - y_curr
    ...     distance = np.sqrt(dx**2 + dy**2)
    ...     angle_to_lookahead = np.arctan2(dy, dx)
    ...     
    ...     theta_error = angle_to_lookahead - theta_curr
    ...     theta_error = np.arctan2(np.sin(theta_error), np.cos(theta_error))
    ...     
    ...     # Pure pursuit control
    ...     v = 0.5  # Constant forward velocity
    ...     omega = 2.0 * v * np.sin(theta_error) / lookahead
    ...     
    ...     v_L = v - omega * robot.L
    ...     v_R = v + omega * robot.L
    ...     
    ...     return np.array([v_L, v_R])
    >>> 
    >>> result_path = robot.rollout(
    ...     x0=np.array([0.0, 0.0, 0.0]),
    ...     policy=path_following_controller,
    ...     n_steps=800
    ... )
    >>> 
    >>> # Plot path following
    >>> t_plot = np.linspace(0, 40, 1000)
    >>> x_path, y_path = lemniscate_path(0.5 * t_plot)
    >>> 
    >>> fig_path = go.Figure()
    >>> fig_path.add_trace(go.Scatter(
    ...     x=x_path, y=y_path,
    ...     mode='lines',
    ...     line=dict(dash='dash', color='gray', width=2),
    ...     name='Desired path'
    ... ))
    >>> fig_path.add_trace(go.Scatter(
    ...     x=result_path['states'][:, 0],
    ...     y=result_path['states'][:, 1],
    ...     mode='lines',
    ...     line=dict(color='blue', width=2),
    ...     name='Actual path'
    ... ))
    >>> fig_path.update_layout(
    ...     title='Figure-8 Path Following',
    ...     xaxis_title='x [m]',
    ...     yaxis_title='y [m]',
    ...     yaxis_scaleanchor='x'
    ... )
    >>> fig_path.show()

    Physical Insights:
    -----------------
    **Nonholonomic Constraints Explained:**
    A car cannot move sideways because wheels enforce "rolling without slipping".
    This is a velocity constraint (not a position constraint):
    - Instantaneous motion restricted
    - But can reach any configuration via maneuvers
    - Like parallel parking: can't move sideways, but can reach any spot

    **Differential Drive Advantages:**
    - Simple mechanical design (two motors)
    - Can turn in place (zero turning radius)
    - Good maneuverability in tight spaces
    - Easy to control (decoupled v and ω)

    **Differential Drive Disadvantages:**
    - Cannot move sideways (nonholonomic)
    - Sensitive to wheel diameter differences
    - Odometry drift from wheel slip
    - Poor traction on uneven terrain

    **Kinematic vs Dynamic Models:**
    This is a KINEMATIC model (velocities → positions).
    
    Dynamic model would include:
    - Motor dynamics (voltage → wheel velocity)
    - Friction forces
    - Inertia effects
    - Slip dynamics

    Kinematic model assumes:
    - Instantaneous velocity changes (infinite acceleration)
    - Perfect velocity tracking by motors
    - No slip, no dynamics

    Valid when:
    - Velocities change slowly (quasi-static)
    - Motor control bandwidth >> motion bandwidth
    - Low speeds (slip negligible)

    **Dubins Paths (Shortest Paths):**
    For car-like robots (forward only, bounded curvature), shortest paths
    between poses are combinations of:
    - L: Left turn (maximum curvature)
    - R: Right turn (maximum curvature)
    - S: Straight line

    Six types: LSL, LSR, RSL, RSR, RLR, LRL

    **Reeds-Shepp Paths (with reverse):**
    If backward motion allowed, 48 possible path types!
    Often shorter than Dubins paths.

    Common Pitfalls:
    ---------------
    1. **Forgetting nonholonomic constraint:**
       Cannot use methods for holonomic systems
       Cannot command arbitrary (ẋ, ẏ, θ̇)

    2. **Direct position control:**
       Cannot directly control x, y independently
       Must go through velocity → position

    3. **Linearization limitations:**
       System is NOT feedback linearizable everywhere
       Linearization fails at v = 0

    4. **Odometry as ground truth:**
       Wheel odometry drifts significantly
       Always validate with external sensors

    5. **Ignoring wheel velocity limits:**
       Commanded v_L, v_R may exceed limits
       Must saturate or use MPC with constraints

    6. **Backward motion confusion:**
       Some controllers assume forward-only
       Backward motion requires sign handling

    Extensions:
    ----------
    1. **Car-like (Ackermann steering):**
       Front wheels steer, rear wheels drive
       Different kinematics, cannot turn in place

    2. **Omnidirectional (mecanum/swerve):**
       Can move in any direction instantly
       Holonomic (no constraint)

    3. **Dynamic model:**
       Include motor dynamics, slip, inertia
       Second-order system

    4. **Skid-steering:**
       Four or six wheels, intentional slip
       More complex dynamics

    5. **Tracked vehicle:**
       Tank-like, continuous contact
       Slip modeling essential

    6. **Multi-robot system:**
       Fleet coordination, formation control
    """

    def define_system(
        self,
        L: float = 0.5,
        dt: float = 0.1,
        method: str = 'euler',
        max_wheel_velocity: Optional[float] = None,
        control_mode: str = 'wheel_velocities',
    ):
        """
        Define discrete-time differential drive robot kinematics.

        Parameters
        ----------
        L : float
            Half of wheelbase [m] (distance from center to wheel)
        dt : float
            Sampling period [s]
        max_wheel_velocity : Optional[float]
            Maximum wheel velocity [m/s]
        control_mode : str
            'wheel_velocities' or 'unicycle'
        """
        # Store configuration
        self.L = L
        self._method = method
        self._max_wheel_velocity = max_wheel_velocity
        self._control_mode = control_mode

        # State variables
        x, y, theta = sp.symbols('x y theta', real=True)

        self.state_vars = [x, y, theta]
        self._dt = dt
        self.order = 1
        self.output_vars = [x, y, theta]

        # Parameters
        L_sym = sp.symbols('L', positive=True, real=True)
        self.parameters = {L_sym: L}

        # Control variables (two options)
        if control_mode == 'wheel_velocities':
            v_L, v_R = sp.symbols('v_L v_R', real=True)
            self.control_vars = [v_L, v_R]

            # Compute linear and angular velocities
            v = (v_L + v_R) / 2
            omega = (v_R - v_L) / (2 * L_sym)

        elif control_mode == 'unicycle':
            v, omega = sp.symbols('v omega', real=True)
            self.control_vars = [v, omega]

        else:
            raise ValueError(f"Unknown control_mode: {control_mode}")

        # Kinematic equations
        
        # Exact solution for circular arc (constant v, ω)
        # This is the exact solution of the continuous kinematics
        
        # Handle special case: straight line (ω = 0)
        # Use Piecewise for symbolic expression
        R = v / omega  # Radius of curvature (undefined if ω = 0)
        
        # # For straight line: simple displacement
        # x_straight = x + dt * v * sp.cos(theta)
        # y_straight = y + dt * v * sp.sin(theta)
        # theta_straight = theta
        
        # For arc motion:
        x_arc = x + R * (sp.sin(theta + omega*dt) - sp.sin(theta))
        y_arc = y - R * (sp.cos(theta + omega*dt) - sp.cos(theta))
        theta_arc = theta + omega * dt
        
        # Use arc solution by default (straight is limit)
        x_next = x_arc
        y_next = y_arc
        theta_next = theta_arc

        self._f_sym = sp.Matrix([x_next, y_next, theta_next])
        self._h_sym = sp.Matrix([x, y, theta])

    def setup_equilibria(self):
        """
        Set up reference configurations.

        Note: No true equilibria exist (can stop anywhere).
        """
        # Origin is always automatically added
        pass

    def convert_unicycle_to_wheel(
        self,
        v: float,
        omega: float,
    ) -> Tuple[float, float]:
        """
        Convert unicycle controls (v, ω) to wheel velocities (v_L, v_R).

        Parameters
        ----------
        v : float
            Linear velocity [m/s]
        omega : float
            Angular velocity [rad/s]

        Returns
        -------
        tuple
            (v_L, v_R) wheel velocities [m/s]

        Examples
        --------
        >>> robot = DifferentialDriveRobot(L=0.2)
        >>> v_L, v_R = robot.convert_unicycle_to_wheel(v=1.0, omega=0.5)
        >>> print(f"Left: {v_L:.2f}, Right: {v_R:.2f}")
        """
        v_L = v - omega * self.L
        v_R = v + omega * self.L
        return v_L, v_R

    def convert_wheel_to_unicycle(
        self,
        v_L: float,
        v_R: float,
    ) -> Tuple[float, float]:
        """
        Convert wheel velocities (v_L, v_R) to unicycle controls (v, ω).

        Parameters
        ----------
        v_L : float
            Left wheel velocity [m/s]
        v_R : float
            Right wheel velocity [m/s]

        Returns
        -------
        tuple
            (v, ω) linear and angular velocities

        Examples
        --------
        >>> robot = DifferentialDriveRobot(L=0.2)
        >>> v, omega = robot.convert_wheel_to_unicycle(v_L=0.8, v_R=1.2)
        >>> print(f"Linear: {v:.2f} m/s, Angular: {omega:.2f} rad/s")
        """
        v = (v_L + v_R) / 2.0
        omega = (v_R - v_L) / (2.0 * self.L)
        return v, omega

    def compute_instantaneous_radius(
        self,
        v_L: float,
        v_R: float,
    ) -> Optional[float]:
        """
        Compute instantaneous radius of curvature.

        Parameters
        ----------
        v_L : float
            Left wheel velocity [m/s]
        v_R : float
            Right wheel velocity [m/s]

        Returns
        -------
        float or None
            Radius of curvature [m], None if straight line

        Notes
        -----
        R = L·(v_L + v_R)/(v_R - v_L)

        Special cases:
        - v_L = v_R: R = ∞ (straight line)
        - v_L = -v_R: R = 0 (turn in place)

        Examples
        --------
        >>> robot = DifferentialDriveRobot(L=0.2)
        >>> R = robot.compute_instantaneous_radius(v_L=0.8, v_R=1.2)
        >>> print(f"Turning radius: {R:.2f} m")
        """
        if abs(v_R - v_L) < 1e-10:
            return None  # Straight line (infinite radius)

        R = self.L * (v_L + v_R) / (v_R - v_L)
        return R

    def compute_minimum_turning_radius(
        self,
        v_max: Optional[float] = None,
    ) -> float:
        """
        Compute minimum turning radius.

        Parameters
        ----------
        v_max : Optional[float]
            Maximum wheel velocity (default: use stored value)

        Returns
        -------
        float
            Minimum radius [m]

        Notes
        -----
        Minimum radius when one wheel at +v_max, other at -v_max:
            R_min = L·(v_max - (-v_max))/(v_max - (-v_max)) = L

        Actually, for differential drive: R_min → 0 (can spin in place!)
        But at max forward speed: R_min = L·v_max/v_max = L

        Examples
        --------
        >>> robot = DifferentialDriveRobot(L=0.2)
        >>> R_min = robot.compute_minimum_turning_radius(v_max=1.0)
        >>> print(f"Min turning radius at full speed: {R_min:.2f} m")
        """
        if v_max is None:
            if self._max_wheel_velocity is None:
                raise ValueError("Must specify v_max or set max_wheel_velocity")
            v_max = self._max_wheel_velocity

        # At maximum forward speed with maximum turn rate
        return self.L

    # def print_equations(self, simplify: bool = True):
    #     """Print symbolic equations."""
    #     print("=" * 70)
    #     print(f"{self.__class__.__name__} (Discrete-Time, dt={self.dt} s)")
    #     print("=" * 70)
    #     print("Differential Drive Mobile Robot (Nonholonomic)")
    #     print(f"Integration Method: {self._method.upper()}")
    #     print(f"Control Mode: {self._control_mode}")

    #     print("\nPhysical Parameters:")
    #     print(f"  Half-wheelbase: L = {self.L} m")
    #     print(f"  Full wheelbase: 2L = {2*self.L} m")
    #     if self._max_wheel_velocity is not None:
    #         print(f"  Max wheel velocity: {self._max_wheel_velocity} m/s")

    #     print(f"\nState: x = [x, y, θ] (position, heading)")
    #     if self._control_mode == 'wheel_velocities':
    #         print(f"Control: u = [v_L, v_R] (left/right wheel velocities)")
    #     else:
    #         print(f"Control: u = [v, ω] (linear/angular velocity)")
    #     print(f"Dimensions: nx={self.nx}, nu={self.nu}")

    #     print("\nKinematic Equations:")
    #     print("  Linear velocity: v = (v_L + v_R) / 2")
    #     print("  Angular velocity: ω = (v_R - v_L) / (2L)")
    #     print("\nContinuous-time kinematics:")
    #     print("  ẋ = v·cos(θ)")
    #     print("  ẏ = v·sin(θ)")
    #     print("  θ̇ = ω")

    #     print("\nNonholonomic Constraint:")
    #     print("  ẋ·sin(θ) - ẏ·cos(θ) = 0")
    #     print("  (Cannot move perpendicular to heading)")

    #     print("\nDiscrete-Time Dynamics: x[k+1] = f(x[k], u[k])")
    #     labels = ['x[k+1]', 'y[k+1]', 'θ[k+1]']
    #     for label, expr in zip(labels, self._f_sym):
    #         expr_sub = self.substitute_parameters(expr)
    #         if simplify:
    #             expr_sub = sp.simplify(expr_sub)
    #         print(f"  {label} = {expr_sub}")

    #     print("\nMotion Primitives:")
    #     print("  Forward:       v_L = v_R > 0  → Straight ahead")
    #     print("  Backward:      v_L = v_R < 0  → Straight back")
    #     print("  Spin left:     v_L < 0, v_R > 0, |v_L|=|v_R| → Turn in place CCW")
    #     print("  Spin right:    v_L > 0, v_R < 0, |v_L|=|v_R| → Turn in place CW")
    #     print("  Arc left:      v_L < v_R  → Curve left")
    #     print("  Arc right:     v_L > v_R  → Curve right")

    #     print("\nKey Properties:")
    #     print("  - Nonholonomic (cannot move sideways)")
    #     print("  - 3 configuration DOF, 2 velocity DOF")
    #     print("  - Controllable (can reach any pose)")
    #     print("  - No equilibria (can stop anywhere)")
    #     print("  - Kinematic model (no dynamics)")

    #     print("\nApplications:")
    #     print("  - Warehouse/logistics robots")
    #     print("  - Vacuum/cleaning robots")
    #     print("  - Educational robots (TurtleBot, Pioneer)")
    #     print("  - Wheelchairs and mobility devices")
    #     print("  - AGVs (Automated Guided Vehicles)")

    #     print("=" * 70)


# Aliases
DifferentialDrive = DifferentialDriveRobot
TwoWheeledRobot = DifferentialDriveRobot
UnicycleModel = DifferentialDriveRobot  # When control_mode='unicycle'