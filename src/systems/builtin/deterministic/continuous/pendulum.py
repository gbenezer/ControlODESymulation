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


class SymbolicPendulum(ContinuousSymbolicSystem):
    """
    Simple inverted pendulum system - first-order state-space formulation.

    Physical System:
    ---------------
    A point mass attached to a massless rigid rod, free to rotate about a fixed pivot.
    The pendulum experiences:
    - Gravitational torque (proportional to sin(θ))
    - Viscous damping (proportional to angular velocity)
    - External control torque

    State Space:
    -----------
    State: x = [θ, θ̇]
        - θ (theta): Angular position from upward vertical [rad]
          * θ = 0: upright (unstable equilibrium)
          * θ = π: hanging down (stable equilibrium)
        - θ̇ (theta_dot): Angular velocity [rad/s]

    Control: u = [τ]
        - τ (torque): Applied torque at pivot [N⋅m]

    Output: y = [θ]
        - Measures only the angle (partial observation)

    Dynamics:
    --------
    The equations of motion are:
        θ̇ = θ̇
        θ̈ = -(β/I)θ̇ + (g/l)sin(θ) + τ/I

    where I = ml² is the moment of inertia.

    Rewritten as first-order system:
        dx/dt = [θ̇, -(β/ml²)θ̇ + (g/l)sin(θ) + τ/(ml²)]ᵀ

    Parameters:
    ----------
    m : float, default=1.0
        Mass of the bob [kg]. Larger mass → more inertia, slower response.
    l : float, default=1.0
        Length of the rod [m]. Longer rod → more gravity torque, slower dynamics.
    beta : float, default=1.0
        Damping coefficient [N⋅m⋅s/rad]. Larger β → more energy dissipation.
    g : float, default=9.81
        Gravitational acceleration [m/s²].
    """

    def define_system(
        self,
        m_val: float = 1.0,
        l_val: float = 1.0,
        beta_val: float = 1.0,
        g_val: float = 9.81,
    ):
        theta, theta_dot = sp.symbols("theta theta_dot", real=True)
        u = sp.symbols("u", real=True)
        m, l, beta, g = sp.symbols("m l beta g", real=True, positive=True)

        self.parameters = {m: m_val, l: l_val, beta: beta_val, g: g_val}
        self.state_vars = [theta, theta_dot]
        self.control_vars = [u]
        # self.output_vars = [theta]  # ❌ REMOVE THIS LINE
        # validation does not allow the same symbol to be used
        # as a state and output variable
        # if creating a system with partial state feedback, put symbols
        # in _h_sym only
        # use output_variables only for truly unique symbolic variables
        self.order = 1

        ml2 = m * l * l
        self._f_sym = sp.Matrix(
            [theta_dot, (-beta / ml2) * theta_dot + (g / l) * sp.sin(theta) + u / ml2],
        )
        self._h_sym = sp.Matrix([theta])  # Output is theta (partial observability)

    def setup_equilibria(self):
        # method to add equilibria to the system automatically after initialization

        # add the stable equilibrium where the pendulum is hanging down
        self.add_equilibrium(
            "downward",
            x_eq=np.array([0.0, 0.0]),
            u_eq=np.array([0.0]),
            verify=True,
        )

        # add the unstable equilibrium where the pendulum is inverted
        self.add_equilibrium(
            "inverted",
            x_eq=np.array([np.pi, 0.0]),
            u_eq=np.array([0.0]),
            stability="unstable",
            notes="Requires active control",
        )
        self.set_default_equilibrium(name="inverted")


class SymbolicPendulum2ndOrder(ContinuousSymbolicSystem):
    """
    Inverted pendulum - second-order formulation (returns ONLY acceleration).

    **CRITICAL BEHAVIOR**: The forward() method returns ONLY θ̈, not [θ̇, θ̈].
    This is the correct formulation for second-order systems.

    Physical System:
    ---------------
    Identical physics to SymbolicPendulum, but formulated as a second-order
    differential equation rather than a first-order state-space system.

    State Space:
    -----------
    State: x = [θ, θ̇]  (same as first-order variant)

    Dynamics Representation:
    -----------------------
    Second-order form:
        θ̈ = -(β/I)θ̇ + (g/l)sin(θ) + τ/I

    The forward() method computes and returns ONLY the acceleration θ̈.

    State-space conversion (handled automatically):
        dx/dt = [θ̇    ] = [      θ̇       ]
                [θ̈    ]   [f(θ, θ̇, τ)]

    The GenericDiscreteTimeSystem integrator handles the conversion:
    1. Calls forward(x, u) to get θ̈
    2. Integrates θ̈ to get θ̇_{k+1}
    3. Integrates θ̇ to get θ_{k+1}
    4. Returns x_{k+1} = [θ_{k+1}, θ̇_{k+1}]

    Parameters:
    ----------
    m, l, beta, g : Same as SymbolicPendulum

    Properties:
    ----------
    order : int = 2
        Marks system as second-order (changes integration behavior)
    nq : int = 1
        Number of generalized coordinates (just θ)

    Notes:
    -----
    - forward() output shape is (1,) for scalar acceleration, NOT (2,)
    - The state x is still (2,) = [θ, θ̇]
    - Integrators automatically handle the state-space conversion
    - Can use different integration methods for position vs velocity
    - Linearization returns full 2×2 state-space matrices

    See Also:
    --------
    SymbolicPendulum : First-order state-space formulation
    SymbolicQuadrotor2D : Another second-order system (3 accelerations)
    """

    def define_system(self, m_val=1.0, l_val=1.0, beta_val=1.0, g_val=9.81):

        # State: [theta, theta_dot]
        theta, theta_dot = sp.symbols("theta theta_dot", real=True)
        u = sp.symbols("u", real=True)
        m, l, beta, g = sp.symbols("m l beta g", real=True, positive=True)

        self.parameters = {m: m_val, l: l_val, beta: beta_val, g: g_val}
        self.state_vars = [theta, theta_dot]
        self.control_vars = [u]
        self.order = 2  # Mark as second-order

        # Second-order system: return ONLY acceleration
        ml2 = m * l * l
        theta_ddot = (-beta / ml2) * theta_dot + (g / l) * sp.sin(theta) + u / ml2

        self._f_sym = sp.Matrix([theta_ddot])  # Single element
        self._h_sym = sp.Matrix([theta])

    def setup_equilibria(self):
        # method to add equilibria to the system automatically after initialization

        # add the stable equilibrium where the pendulum is hanging down
        self.add_equilibrium(
            "downward",
            x_eq=np.array([0.0, 0.0]),
            u_eq=np.array([0.0]),
            verify=True,
        )

        # add the unstable equilibrium where the pendulum is inverted
        self.add_equilibrium(
            "inverted",
            x_eq=np.array([np.pi, 0.0]),
            u_eq=np.array([0.0]),
            stability="unstable",
            notes="Requires active control",
        )
        self.set_default_equilibrium(name="inverted")
