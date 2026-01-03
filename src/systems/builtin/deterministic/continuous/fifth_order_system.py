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


class FifthOrderMechanicalSystem(ContinuousSymbolicSystem):
    """
    Fifth-order mechanical system - extremely high-order dynamics.

    **WARNING**: This is an artificially complex system designed for testing
    high-order integration schemes. Physical systems rarely exceed third order.

    Physical Interpretation:
    -----------------------
    Could represent:
    - Flexible manipulator with multiple vibration modes
    - Actuator with nested control loops (each adding an order)
    - Academic test case for high-order integration

    Mathematical Formulation:
    ------------------------
    State: x = [q, q', q'', q''', q⁽⁴⁾]
    where:
        - q: Position [m]
        - q': Velocity [m/s]
        - q'': Acceleration [m/s²]
        - q''': Jerk [m/s³]
        - q⁽⁴⁾: Snap (fourth derivative) [m/s⁴]

    The system evolves according to:
        q⁽⁵⁾ = f(q, q', q'', q''', q⁽⁴⁾, u)

    Dynamics:
    --------
    q⁽⁵⁾ = -(k/m)q - c₁q' - c₂q'' - c₃q''' - 0.01q⁽⁴⁾ - g + u/m

    This includes:
    - Stiffness term: -kq (like a spring)
    - Multiple damping terms at each derivative level
    - Gravity: -g
    - Control input: u/m

    Parameters:
    ----------
    m : float, default=1.0
        Mass [kg]
    k : float, default=1.0
        Stiffness coefficient [N/m]
    c1 : float, default=0.1
        First-order damping (velocity damping) [N⋅s/m]
    c2 : float, default=0.05
        Second-order damping (acceleration damping) [N⋅s³/m]
    c3 : float, default=0.01
        Third-order damping (jerk damping) [N⋅s⁵/m]
    g : float, default=9.81
        Gravitational acceleration [m/s²]

    State Space:
    -----------
    State: x = [q, q', q'', q''', q⁽⁴⁾]  (5D)
    Control: u = [force]  (1D)
    Output: y = [q, q']  (position and velocity)

    Equilibrium:
    -----------
    Static equilibrium (balancing gravity):
        q_eq = -mg/k  (compressed by gravity)
        All derivatives zero
        u_eq = mg  (supporting weight)

    See Also:
    --------
    SymbolicPendulum2ndOrder : More typical second-order system
    CoupledOscillatorSystem : More realistic multi-DOF system
    """

    def define_system(
        self,
        m_val: float = 1.0,
        k_val: float = 1.0,
        c1_val: float = 0.1,
        c2_val: float = 0.05,
        c3_val: float = 0.01,
        g_val: float = 9.81,
    ):
        q, q1, q2, q3, q4 = sp.symbols("q q1 q2 q3 q4", real=True)
        u = sp.symbols("u", real=True)
        m, k, c1, c2, c3, g = sp.symbols("m k c1 c2 c3 g", real=True, positive=True)

        self.parameters = {
            m: m_val,
            k: k_val,
            c1: c1_val,
            c2: c2_val,
            c3: c3_val,
            g: g_val,
        }

        self.state_vars = [q, q1, q2, q3, q4]
        self.control_vars = [u]
        self.output_vars = [q, q1]
        self.order = 5

        # equilibrium value for setup method
        self.u_eq = m_val * g_val
        self.q_eq = -1.0 * (self.u_eq / k_val)

        # Fifth derivative: complex dynamics with multiple damping terms
        q5 = -k / m * q - c1 * q1 - c2 * q2 - c3 * q3 - 0.01 * q4 - g + u / m

        self._f_sym = sp.Matrix([q5])
        self._h_sym = sp.Matrix([q, q1])

    def setup_equilibria(self):
        # method to add equilibria to the system automatically after initialization

        self.add_equilibrium(
            "compressed",
            x_eq=np.array([self.q_eq, 0.0, 0.0, 0.0, 0.0]),
            u_eq=np.array([self.u_eq]),
            verify=True,
        )
        self.set_default_equilibrium(name="compressed")
