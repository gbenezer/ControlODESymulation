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


class CoupledOscillatorSystem(ContinuousSymbolicSystem):
    """
    Coupled mass-spring-damper system with rotational coupling - first-order formulation.

    Physical System:
    ---------------
    Two masses connected by springs to fixed walls and to each other, with an
    additional rotational degree of freedom that couples to the second mass.

    The system consists of:
    - Two point masses (m₁, m₂) that can move horizontally
    - Springs connecting each mass to ground (k₁, k₂)
    - Coupling spring between the masses (k_coupling)
    - Viscous dampers on both masses (shared coefficient c)
    - Rotational element (moment of inertia J) coupled to mass 2

    State Space:
    -----------
    State: x = [x₁, x₂, v₁, v₂, θ]
        Position coordinates:
        - x₁: Position of mass 1 [m]
        - x₂: Position of mass 2 [m]
        - θ (theta): Rotational angle [rad]

        Velocity coordinates:
        - v₁: Velocity of mass 1 [m/s]
        - v₂: Velocity of mass 2 [m/s]

    Control: u = [u₁, u₂]
        - u₁: Force applied to mass 1 [N]
        - u₂: Combined force/torque applied to mass 2 and rotational element

    Output: y = [x₁, x₂, θ]
        - Measures positions of both masses and rotational angle

    Dynamics:
    --------
    The equations of motion are:

    Mass 1 (standard spring-mass-damper):
        dx₁/dt = v₁
        dv₁/dt = -(k₁/m₁)x₁ - (k_c/m₁)(x₁ - x₂) - (c/m₁)v₁ + u₁/m₁

    Mass 2 (coupled to rotation):
        dx₂/dt = v₂
        dv₂/dt = -(k₂/m₂)x₂ - (k_c/m₂)(x₂ - x₁) - (c/m₂)v₂ + sin(θ)/m₂ + u₂/m₂

    Rotational element:
        dθ/dt = -θ/J - x₂/J + u₂/(2J)

    Physical interpretation:
    - Springs create restoring forces proportional to displacement
    - Coupling spring connects the two masses
    - Dampers dissipate energy proportionally to velocity
    - Rotation affects mass 2 through sin(θ) term (nonlinear coupling)
    - Control u₂ affects both mass 2 translation and rotation

    Parameters:
    ----------
    m1 : float, default=1.0
        Mass of first oscillator [kg]. Larger m₁ → slower response to forces.
    m2 : float, default=0.5
        Mass of second oscillator [kg]. Typically different from m₁ to create
        interesting modal behavior.
    k1 : float, default=2.0
        Spring stiffness connecting mass 1 to ground [N/m]. Higher k₁ →
        higher natural frequency for mass 1.
    k2 : float, default=1.0
        Spring stiffness connecting mass 2 to ground [N/m].
    k_coupling : float, default=0.5
        Coupling spring stiffness between masses [N/m]. Controls strength of
        interaction between oscillators. Higher k_c → stronger coupling.
    c : float, default=0.1
        Damping coefficient [N·s/m]. Applied to both masses. Higher c →
        more energy dissipation.
    J : float, default=0.1
        Moment of inertia for rotational element [kg·m²]. Affects rotational
        response time.

    Equilibrium:
    -----------
    Origin equilibrium (all zeros):
        x_eq = [0, 0, 0, 0, 0]  (masses at rest, no rotation)
        u_eq = [0, 0]  (no external forces)

    This equilibrium is stable due to spring restoring forces and damping.

    See Also:
    --------
    NonlinearChainSystem : Chain of coupled oscillators
    Manipulator2Link : Another coupled multi-body system
    """

    def define_system(
        self,
        m1_val=1.0,
        m2_val=0.5,
        k1_val=2.0,
        k2_val=1.0,
        k_coupling_val=0.5,
        c_val=0.1,
        J_val=0.1,
    ):
        x1, x2, v1, v2, theta = sp.symbols("x1 x2 v1 v2 theta", real=True)
        u1, u2 = sp.symbols("u1 u2", real=True)
        m1, m2, k1, k2, k_c, c, J = sp.symbols("m1 m2 k1 k2 k_c c J", real=True, positive=True)

        self.parameters = {
            m1: m1_val,
            m2: m2_val,
            k1: k1_val,
            k2: k2_val,
            k_c: k_coupling_val,
            c: c_val,
            J: J_val,
        }

        self.state_vars = [x1, x2, v1, v2, theta]
        self.control_vars = [u1, u2]
        self.output_vars = [x1, x2, theta]
        self.order = 1

        # Coupled dynamics
        dx1 = v1
        dv1 = -k1 / m1 * x1 - k_c / m1 * (x1 - x2) - c / m1 * v1 + u1 / m1
        dx2 = v2
        dv2 = -k2 / m2 * x2 - k_c / m2 * (x2 - x1) - c / m2 * v2 + sp.sin(theta) / m2 + u2 / m2
        dtheta = -theta / J - x2 / J + u2 / (2 * J)

        self._f_sym = sp.Matrix([dx1, dx2, dv1, dv2, dtheta])
        self._h_sym = sp.Matrix([x1, x2, theta])

    def setup_equilibria(self):
        # method to add equilibria to the system automatically after initialization
        # origin is always added regardless, so nothing needs to be done
        pass


class NonlinearChainSystem(ContinuousSymbolicSystem):
    """
    Chain of five coupled nonlinear oscillators - first-order formulation.

    Physical System:
    ---------------
    A one-dimensional chain of five oscillators where each element influences
    its neighbors through nonlinear coupling.

    Each oscillator has:
    - Linear restoring force (spring-like: -kx)
    - Linear damping (viscous: -cx)
    - Nonlinear coupling to neighbors via sin(x_j - x_i)
    - Only the first oscillator receives external control

    State Space:
    -----------
    State: x = [x₁, x₂, x₃, x₄, x₅]
        - x₁: State of oscillator 1 [rad or m]
        - x₂: State of oscillator 2 [rad or m]
        - x₃: State of oscillator 3 [rad or m]
        - x₄: State of oscillator 4 [rad or m]
        - x₅: State of oscillator 5 [rad or m]

    Control: u = [u]
        - u: External force/torque applied only to first oscillator
        - Influence propagates to other oscillators through coupling

    Output: y = [x₁, x₃, x₅]
        - Sparse observation: only odd-numbered oscillators measured

    Dynamics:
    --------
    The equations of motion form a nearest-neighbor coupling structure:

    Oscillator 1 (left boundary, receives control):
        dx₁/dt = -k·x₁ - c·x₁ + α·sin(x₂ - x₁) + u

    Oscillator 2 (interior, coupled to neighbors):
        dx₂/dt = -k·x₂ - c·x₂ + α·sin(x₁ - x₂) + α·sin(x₃ - x₂)

    Oscillator 3 (interior, coupled to neighbors):
        dx₃/dt = -k·x₃ - c·x₃ + α·sin(x₂ - x₃) + α·sin(x₄ - x₃)

    Oscillator 4 (interior, coupled to neighbors):
        dx₄/dt = -k·x₄ - c·x₄ + α·sin(x₃ - x₄) + α·sin(x₅ - x₄)

    Oscillator 5 (right boundary, no control):
        dx₅/dt = -k·x₅ - c·x₅ + α·sin(x₄ - x₅)

    Physical interpretation:
    - Linear terms (-kx, -cx): individual oscillator wants to return to zero
    - Nonlinear coupling α·sin(x_j - x_i): synchronization force
      * When x_j > x_i: positive force on i (speeds it up)
      * When x_j < x_i: negative force on i (slows it down)
      * Maximum coupling at π/2 phase difference
    - Control u propagates through chain via coupling

    Parameters:
    ----------
    k : float, default=1.0
        Linear stiffness/restoring coefficient [1/s]. Higher k → stronger
        individual oscillator dynamics, weaker relative coupling influence.
    c : float, default=0.1
        Damping coefficient [1/s]. Higher c → faster energy dissipation.
        Damps out transients and oscillations.
    alpha : float, default=0.1
        Nonlinear coupling strength. Controls interaction between neighbors:
        - α = 0: Uncoupled oscillators
        - Small α: Weak coupling, local behavior dominates
        - Large α: Strong coupling, collective behavior emerges
        - α > k+c: Coupling-dominated dynamics, synchronization possible

    Equilibrium:
    -----------
    Synchronous equilibrium (all at origin):
        x_eq = [0, 0, 0, 0, 0]  (all oscillators aligned at zero)
        u_eq = 0  (no external force)

    This equilibrium is stable due to damping. Other synchronized states
    (all x_i equal) are also equilibria for u=0.

    See Also:
    --------
    CoupledOscillatorSystem : Smaller coupled system with different structure
    VanDerPolOscillator : Single nonlinear oscillator with limit cycle
    Lorenz : Another system with complex nonlinear dynamics
    """

    def define_system(self, k_val=1.0, c_val=0.1, alpha_val=0.1):
        x1, x2, x3, x4, x5 = sp.symbols("x1 x2 x3 x4 x5", real=True)
        u = sp.symbols("u", real=True)
        k, c, alpha = sp.symbols("k c alpha", real=True, positive=True)

        self.parameters = {k: k_val, c: c_val, alpha: alpha_val}

        self.state_vars = [x1, x2, x3, x4, x5]
        self.control_vars = [u]
        self.output_vars = [x1, x3, x5]
        self.order = 1

        # Chain dynamics with nonlinear coupling
        dx1 = -k * x1 - c * x1 + alpha * sp.sin(x2 - x1) + u
        dx2 = -k * x2 - c * x2 + alpha * sp.sin(x1 - x2) + alpha * sp.sin(x3 - x2)
        dx3 = -k * x3 - c * x3 + alpha * sp.sin(x2 - x3) + alpha * sp.sin(x4 - x3)
        dx4 = -k * x4 - c * x4 + alpha * sp.sin(x3 - x4) + alpha * sp.sin(x5 - x4)
        dx5 = -k * x5 - c * x5 + alpha * sp.sin(x4 - x5)

        self._f_sym = sp.Matrix([dx1, dx2, dx3, dx4, dx5])
        self._h_sym = sp.Matrix([x1, x3, x5])

    def setup_equilibria(self):
        # method to add equilibria to the system automatically after initialization
        # origin is always added regardless, so nothing needs to be done
        pass
