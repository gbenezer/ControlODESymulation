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

import sympy as sp

from src.systems.base.core.continuous_symbolic_system import ContinuousSymbolicSystem


class DubinsVehicle(ContinuousSymbolicSystem):
    """
    Dubins vehicle - kinematic car model with unicycle dynamics.

    Physical System:
    ---------------
    A simplified model of a car or mobile robot that can move forward and
    rotate, but cannot move sideways (nonholonomic constraint).

    The vehicle is modeled as a point with:
    - Position (x, y) in the plane
    - Heading angle θ
    - Forward velocity v (control input)
    - Angular velocity ω (control input)

    **Key constraint**: The vehicle must move in the direction it's pointing
    (no lateral sliding). This is called a nonholonomic constraint.

    Coordinate Frame:
    ----------------
    - Inertial frame: Fixed (x, y) coordinates
    - Body frame: Moves and rotates with vehicle
    - Heading θ: Angle from x-axis to vehicle's forward direction

    State Space:
    -----------
    State: x = [x, y, θ]
        - x: Horizontal position [m]
        - y: Vertical position [m]
        - θ (theta): Heading angle [rad]
          * θ = 0: pointing right (along +x axis)
          * θ = π/2: pointing up (along +y axis)
          * θ = π: pointing left
          * θ = 3π/2 or -π/2: pointing down

    Control: u = [v, ω]
        - v: Forward velocity [m/s]
          * v > 0: move forward
          * v < 0: move backward
          * v = 0: stopped
        - ω (omega): Angular velocity [rad/s]
          * ω > 0: turn left (counterclockwise)
          * ω < 0: turn right (clockwise)
          * ω = 0: straight motion

    Output: y = [x, y, θ]
        - Full state observation (position and heading)

    Dynamics:
    --------
    The kinematic equations (Dubins car model):

        ẋ = v·cos(θ)
        ẏ = v·sin(θ)
        θ̇ = ω

    **Position dynamics**:
    - Vehicle moves in direction θ at speed v
    - cos(θ) and sin(θ) project velocity onto x and y axes
    - No motion perpendicular to heading (nonholonomic constraint)

    **Heading dynamics**:
    - Directly controlled by angular velocity ω
    - Independent of forward velocity (can rotate in place if v=0)

    Physical interpretation:
    - The vehicle is like a bicycle: must point where it's going
    - Cannot slide sideways (like a car on dry pavement)
    - Minimum turning radius determined by maximum ω/v ratio

    Turning Radius:
    --------------
    When moving in a circle (v constant, ω constant):
        R = v/ω  (radius of circular path)

    - Tighter turn: increase ω or decrease v
    - Larger turn: decrease ω or increase v
    - Straight line: ω = 0

    Parameters:
    ----------
    This implementation has no physical parameters - it's a pure kinematic
    model. Further modifications to this model may include:
    - Maximum speed v_max
    - Maximum angular velocity ω_max
    - Minimum turning radius R_min = v_max/ω_max

    Equilibria:
    ----------
    **Stationary at origin**:
        x_eq = [0, 0, θ*]  (any heading θ*)
        u_eq = [0, 0]      (no velocity)

    Note: Equilibria form a manifold - any (x*, y*, θ*) with u = [0, 0].
    The system is marginally stable (doesn't return to equilibrium on its own).

    See Also:
    --------
    PathTracking : Error dynamics for path following
    PVTOL : Flying vehicle with similar kinematics
    CartPole : Another nonholonomic system
    """

    def define_system(self):
        x, y, theta = sp.symbols("x y theta", real=True)
        v, omega = sp.symbols("v omega", real=True)

        self.parameters = {}
        self.state_vars = [x, y, theta]
        self.control_vars = [v, omega]
        self.output_vars = []
        self.order = 1

        # Kinematic equations
        dx = v * sp.cos(theta)
        dy = v * sp.sin(theta)
        dtheta = omega

        self._f_sym = sp.Matrix([dx, dy, dtheta])
        self._h_sym = sp.Matrix([x, y, theta])

    def setup_equilibria(self):
        # method to add equilibria to the system automatically after initialization
        # origin is always added regardless, so nothing needs to be done
        pass
