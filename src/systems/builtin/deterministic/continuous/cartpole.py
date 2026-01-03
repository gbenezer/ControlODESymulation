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

from typing import Optional

import numpy as np
import sympy as sp

from src.systems.base.core.continuous_symbolic_system import ContinuousSymbolicSystem


class CartPole(ContinuousSymbolicSystem):
    """
    Cart-pole system (inverted pendulum on cart) - classic underactuated system.

    Physical System:
    ---------------
    A pole (inverted pendulum) attached to a cart that moves horizontally.
    - Cart can slide freely on a horizontal track
    - Pole is attached to cart via a frictionless pivot
    - System is underactuated: 1 control input, 2 degrees of freedom
    - Nonlinear coupling between cart motion and pole angle

    State Space:
    -----------
    State: x = [x, θ, ẋ, θ̇]
        Position coordinates:
        - x: Cart position [m] (positive right)
        - θ: Pole angle from vertical [rad]
          * θ = 0: upright (unstable equilibrium)
          * θ = π: hanging down (stable equilibrium)

        Velocity coordinates:
        - ẋ (x_dot): Cart velocity [m/s]
        - θ̇ (theta_dot): Pole angular velocity [rad/s]

    Control: u = [F]
        - F: Horizontal force applied to cart [N]
        Can be positive (push right) or negative (push left)

    Output: y = [x, θ]
        - Measures cart position and pole angle

    Dynamics:
    --------
    The equations of motion (derived from Lagrangian mechanics):

    Let M = m_cart + m_pole, then:

        ẍ = (F - b·ẋ + m_pole·l·θ̇²·sin(θ) - m_pole·g·sin(θ)·cos(θ)) / (M - m_pole·cos²(θ))

        θ̈ = (F·cos(θ) - b·ẋ·cos(θ) + m_pole·l·θ̇²·sin(θ)·cos(θ) - M·g·sin(θ)) / (l·(M - m_pole·cos²(θ)))

    Physical interpretation:
    - Cart accelerates from applied force F
    - Pole motion creates reaction forces on cart
    - Centrifugal force (θ̇² term) affects both cart and pole
    - Gravity pulls pole down (sin(θ) term)
    - Friction opposes cart motion (b·ẋ)

    Parameters:
    ----------
    m_cart : float, default=1.0
        Mass of the cart [kg]
    m_pole : float, default=0.1
        Mass of the pole [kg]
        Typical: m_pole << m_cart (light pole, heavy cart)
    length : float, default=0.5
        Length from pivot to pole's center of mass [m]
    gravity : float, default=9.81
        Gravitational acceleration [m/s²]
    friction : float, default=0.1
        Cart friction coefficient [N⋅s/m]
        Models bearing friction and air resistance

    Equilibria:
    ----------
    1. **Upright (unstable)**:
       x_eq = [x*, 0, 0, 0]  (any cart position, pole vertical)
       u_eq = 0 (no force needed)

    2. **Hanging (stable)**:
       x_eq = [x*, π, 0, 0]  (any cart position, pole hanging)
       u_eq = 0 (no force needed)

    NOTE: this is a different convention than that of the SymbolicPendulum
    class; that system is defined with π radians being inverted/upward

    See Also:
    --------
    SymbolicPendulum : Simpler version without cart
    PVTOL : Another underactuated system with similar challenges
    """

    def define_system(
        self,
        m_cart_val: float = 1.0,
        m_pole_val: float = 0.1,
        length_val: float = 0.5,
        gravity_val: float = 9.81,
        friction_val: float = 0.1,
        equilibrium_position: Optional[float] = None,
    ):
        # setting equilibrium position for setup_equilibria
        self.eq_pos = equilibrium_position

        # State variables
        x, theta, x_dot, theta_dot = sp.symbols("x theta x_dot theta_dot", real=True)
        F = sp.symbols("F", real=True)

        # Parameters
        mc, mp, l, g, b = sp.symbols("mc mp l g b", real=True, positive=True)

        self.parameters = {
            mc: m_cart_val,
            mp: m_pole_val,
            l: length_val,
            g: gravity_val,
            b: friction_val,
        }

        self.state_vars = [x, theta, x_dot, theta_dot]
        self.control_vars = [F]
        self.output_vars = [x, theta]
        self.order = 2

        # Dynamics (derived from Euler-Lagrange equations)
        # Total mass
        M = mc + mp

        # Sin and cos of theta
        sin_theta = sp.sin(theta)
        cos_theta = sp.cos(theta)

        # Denominator for both equations
        denom = M - mp * cos_theta**2

        # Cart acceleration
        x_ddot = (
            F - b * x_dot + mp * l * theta_dot**2 * sin_theta - mp * g * sin_theta * cos_theta
        ) / denom

        # Pole angular acceleration
        theta_ddot = (
            F * cos_theta
            - b * x_dot * cos_theta
            + mp * l * theta_dot**2 * sin_theta * cos_theta
            - M * g * sin_theta
        ) / (l * denom)

        # Second-order system: forward() returns accelerations
        self._f_sym = sp.Matrix([x_ddot, theta_ddot])
        self._h_sym = sp.Matrix([x, theta])

    def setup_equilibria(self):
        # method to add equilibria to the system automatically after initialization
        # origin is always added regardless, so another equilibrium could be defined
        # at cart origin with the pole hanging down.
        self.add_equilibrium(
            "hanging origin",
            x_eq=np.array([0.0, np.pi, 0.0, 0.0]),
            u_eq=np.array([0.0]),
            stability="stable",
        )

        # if the user specified a particular equilibrium position
        if self.eq_pos is not None:
            self.add_equilibrium(
                "hanging",
                x_eq=np.array([self.eq_pos, np.pi, 0.0, 0.0]),
                u_eq=np.array([0.0]),
                stability="stable",
            )
            self.add_equilibrium(
                "inverted",
                x_eq=np.array([self.eq_pos, 0.0, 0.0, 0.0]),
                u_eq=np.array([0.0]),
                stability="unstable",
            )
            self.set_default_equilibrium(name="inverted")
