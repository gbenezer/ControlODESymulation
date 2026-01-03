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

import numpy as np
import sympy as sp

from src.systems.base.core.continuous_symbolic_system import ContinuousSymbolicSystem


class PathTracking(ContinuousSymbolicSystem):
    """
    Path tracking error dynamics for a vehicle following a circular reference path.

    Physical System:
    ---------------
    Models the error dynamics of a kinematic vehicle (car, robot, boat) as it
    attempts to follow a circular trajectory.

    The vehicle uses a bicycle model (front-wheel steering) and the error
    coordinates are relative to the closest point on the reference circle.

    Coordinate Frames:
    -----------------
    - **Reference path**: Circular trajectory with radius R
    - **Path frame**: Moving frame tangent to reference path
    - **Vehicle frame**: Body-fixed frame of the vehicle
    - **Error coordinates**: Deviations from reference path in path frame

    State Space:
    -----------
    State: x = [d_e, θ_e]
        - d_e: Lateral (cross-track) error [m]
          * d_e > 0: vehicle is to the left of the path
          * d_e < 0: vehicle is to the right of the path
          * d_e = 0: vehicle is exactly on the path

        - θ_e: Heading error [rad]
          * θ_e > 0: vehicle heading points left of desired direction
          * θ_e < 0: vehicle heading points right of desired direction
          * θ_e = 0: vehicle heading is tangent to path

    Control: u = [δ]
        - δ (delta): Front wheel steering angle [rad]
          * δ > 0: steer left
          * δ < 0: steer right
          * δ = 0: straight ahead

    Output: y = [d_e, θ_e]
        - Full state observation (both errors measured)

    Dynamics:
    --------
    The error dynamics describe how tracking errors evolve:

        ḋ_e = v·sin(θ_e)

        θ̇_e = (v·δ)/L - cos(θ_e)/(R/v - sin(θ_e))

    **Lateral error rate ḋ_e**:
    - Proportional to forward speed v
    - Depends on heading error through sin(θ_e)
    - When θ_e > 0 (heading left), d_e increases (moves left)
    - When θ_e < 0 (heading right), d_e decreases (moves right)

    **Heading error rate θ̇_e**:
    - First term (v·δ)/L: Vehicle's turning rate (Ackermann steering)
    - Second term: Path's curvature rate projection
    - At equilibrium, these balance to track the circle

    Physical interpretation:
    - If vehicle steers more than needed → heading error increases
    - If vehicle steers less than needed → heading error decreases
    - Coupling: lateral error affects required steering through geometry

    Parameters:
    ----------
    speed : float, default=1.0
        Constant forward speed of vehicle [m/s]. Assumed to be maintained
        by a low-level speed controller. Higher speed → faster error dynamics.
    length : float, default=1.0
        Vehicle wheelbase [m]. Distance between front and rear axles.
        Longer wheelbase → less maneuverable (smaller turning rate).
    radius : float, default=10.0
        Radius of the circular reference path [m]. Larger radius → gentler
        turn, easier to track. radius → ∞ approaches straight line tracking.

    Equilibrium:
    -----------
    Perfect tracking equilibrium:
        x_eq = [0, 0]  (no lateral error, no heading error)
        u_eq = L/R     (steady-state steering angle for circle)

    At equilibrium:
    - Vehicle is on the path (d_e = 0)
    - Vehicle heading is tangent to path (θ_e = 0)
    - Steering angle exactly matches path curvature
    - Steady-state steering: δ = L/R = wheelbase/radius

    See Also:
    --------
    DubinsVehicle : Full kinematic model (not error dynamics)
    PVTOL : Another vehicle with reference tracking
    CartPole : Another system with error dynamics formulation
    """

    def define_system(self, speed_val=1.0, length_val=1.0, radius_val=10.0):
        d_e, theta_e = sp.symbols("d_e theta_e", real=True)
        delta = sp.symbols("delta", real=True)
        v, L, R = sp.symbols("v L R", real=True, positive=True)

        self.parameters = {v: speed_val, L: length_val, R: radius_val}

        self.state_vars = [d_e, theta_e]
        self.control_vars = [delta]
        self.output_vars = [d_e, theta_e]
        self.order = 1
        
        # Storing value for equilibrium control
        self.eq_control = length_val / radius_val

        # Error dynamics
        sin_theta_e = sp.sin(theta_e)
        cos_theta_e = sp.cos(theta_e)

        # Lateral error rate
        d_e_dot = v * sin_theta_e

        # Heading error rate
        coef = R / v
        theta_e_dot = (v * delta / L) - (cos_theta_e / (coef - sin_theta_e))

        self._f_sym = sp.Matrix([d_e_dot, theta_e_dot])
        self._h_sym = sp.Matrix([d_e, theta_e])
        
    def setup_equilibria(self):
        # method to add equilibria to the system automatically after initialization

        self.add_equilibrium(
            'center',
            x_eq=np.array([0.0, 0.0]),
            u_eq=np.array([self.eq_control]),
            verify=True
            )
