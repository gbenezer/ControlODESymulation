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

class Manipulator2Link(ContinuousSymbolicSystem):
    """
    Two-link planar robotic manipulator - second-order formulation.

    Physical System:
    ---------------
    A planar robot arm with two revolute joints, each driven by a motor that
    applies torque.

    The system consists of:
    - Two rigid links connected in series
    - Two revolute (rotational) joints at base and elbow
    - Actuators (motors) at each joint providing control torques
    - Gravity acting downward
    - Joint friction opposing motion

    Configuration:
    -------------
    - Link 1: Attached to fixed base, length l₁, mass m₁
    - Link 2: Attached to end of link 1, length l₂, mass m₂
    - Joint 1 (base): Angle q₁ from horizontal
    - Joint 2 (elbow): Angle q₂ relative to link 1

    State Space:
    -----------
    State: x = [q₁, q₂, q̇₁, q̇₂]
        Joint angles (configuration):
        - q₁: Base joint angle [rad]
          * q₁ = 0: link 1 horizontal to the right
          * q₁ = π/2: link 1 pointing up
        - q₂: Elbow joint angle [rad]
          * q₂ = 0: links aligned (straight arm)
          * q₂ = π: fully bent (folded back)

        Joint velocities:
        - q̇₁: Base angular velocity [rad/s]
        - q̇₂: Elbow angular velocity [rad/s]

    Control: u = [τ₁, τ₂]
        - τ₁: Torque applied at base joint [N·m]
        - τ₂: Torque applied at elbow joint [N·m]

    Output: y = [q₁, q₂]
        - Measures joint angles (typical for robots with encoders)
        - Does not directly measure end-effector position

    Dynamics:
    --------
    The manipulator dynamics follow the standard robot equation:
        M(q)q̈ + C(q,q̇)q̇ + G(q) + F(q̇) = τ

    where:
    - M(q): Configuration-dependent inertia matrix (2×2)
    - C(q,q̇): Coriolis and centrifugal terms
    - G(q): Gravity terms
    - F(q̇): Friction terms
    - τ: Applied joint torques (control)

    **Inertia Matrix M(q)**:
    The inertia matrix captures how joint accelerations relate to torques.
    It depends on configuration due to changing mass distribution:

        M₁₁ = m₁·lc₁² + m₂·(l₁² + lc₂² + 2·l₁·lc₂·cos(q₂)) + I₁ + I₂
        M₁₂ = m₂·(lc₂² + l₁·lc₂·cos(q₂)) + I₂
        M₂₁ = M₁₂  (symmetric)
        M₂₂ = m₂·lc₂² + I₂

    Key features:
    - M is symmetric and positive definite
    - Diagonal terms: self-inertia of each joint
    - Off-diagonal: coupling between joints
    - M₁₁ maximized when arm is extended (q₂ = 0)

    **Coriolis and Centrifugal Terms C(q,q̇)**:
    These arise from coordinate system rotation and create coupling:

        h = -m₂·l₁·lc₂·sin(q₂)
        C₁ = h·(2·q̇₁·q̇₂ + q̇₂²)
        C₂ = -h·q̇₁²

    Physical interpretation:
    - When joint 2 moves, it creates forces on joint 1 (and vice versa)
    - Centrifugal: pushing outward when rotating
    - Coriolis: deflection perpendicular to motion

    **Gravity Terms G(q)**:
    Gravitational torques trying to pull arm downward:

        G₁ = (m₁·lc₁ + m₂·l₁)·g·cos(q₁) + m₂·lc₂·g·cos(q₁ + q₂)
        G₂ = m₂·lc₂·g·cos(q₁ + q₂)

    Key features:
    - Maximum when arm horizontal (cos = 1)
    - Zero when arm vertical (cos = 0)
    - Both links contribute to joint 1 torque
    - Only link 2 affects joint 2 torque

    **Friction F(q̇)**:
    Simple viscous friction model:

        F₁ = b₁·q̇₁
        F₂ = b₂·q̇₂

    **Solving for Accelerations**:
    The forward dynamics gives:
        q̈ = M⁻¹(τ - C - G - F)

    Parameters:
    ----------
    m1 : float, default=1.0
        Mass of link 1 [kg]. Affects inertia and gravity torques.
    m2 : float, default=1.0
        Mass of link 2 [kg]. Lighter distal link → faster motion.
    l1 : float, default=1.0
        Length of link 1 [m]. Distance from base to elbow.
    l2 : float, default=1.0
        Length of link 2 [m]. Distance from elbow to end-effector.
    lc1 : float, default=0.5
        Distance from base joint to center of mass of link 1 [m].
        Typically lc₁ = l₁/2 for uniform link.
    lc2 : float, default=0.5
        Distance from elbow joint to center of mass of link 2 [m].
        Typically lc₂ = l₂/2 for uniform link.
    I1 : float, default=0.1
        Moment of inertia of link 1 about its center of mass [kg·m²].
        For uniform rod: I = (1/12)·m·l²
    I2 : float, default=0.1
        Moment of inertia of link 2 about its center of mass [kg·m²].
    gravity : float, default=9.81
        Gravitational acceleration [m/s²].
    friction1 : float, default=0.1
        Viscous friction coefficient at joint 1 [N·m·s/rad].
    friction2 : float, default=0.1
        Viscous friction coefficient at joint 2 [N·m·s/rad].

    Equilibria:
    ----------
    **Hanging down (stable)**:
        q_eq = [π, 0]  (link 1 down, link 2 aligned)
        q̇_eq = [0, 0]  (at rest)
        τ_eq = [0, 0]  (gravity balances)

    **Horizontal (unstable without control)**:
        q_eq = [0, 0]  (both links horizontal)
        Requires active control due to gravity

    **Upright (highly unstable)**:
        q_eq = [π/2, 0]  (both links pointing up)
        Requires fast, precise control

    Forward Kinematics:
    ------------------
    End-effector position in Cartesian space:
        x_ee = l₁·cos(q₁) + l₂·cos(q₁ + q₂)
        y_ee = l₁·sin(q₁) + l₂·sin(q₁ + q₂)

    Workspace:
    - Circle of radius l₁ + l₂ (maximum reach)
    - Inner circle of radius |l₁ - l₂| (unreachable)
    - Singularities when arm fully extended or folded

    See Also:
    --------
    CartPole : Similar dynamics but with prismatic (sliding) joint
    SymbolicPendulum2ndOrder : Single-link version
    PVTOL : Flying robot with similar multi-body coupling
    """

    def __init__(
        self,
        m1: float = 1.0,
        m2: float = 1.0,
        l1: float = 1.0,
        l2: float = 1.0,
        lc1: float = 0.5,
        lc2: float = 0.5,
        I1: float = 0.1,
        I2: float = 0.1,
        gravity: float = 9.81,
        friction1: float = 0.1,
        friction2: float = 0.1,
    ):
        super().__init__(m1, m2, l1, l2, lc1, lc2, I1, I2, gravity, friction1, friction2)

    def define_system(
        self,
        m1_val,
        m2_val,
        l1_val,
        l2_val,
        lc1_val,
        lc2_val,
        I1_val,
        I2_val,
        gravity_val,
        friction1_val,
        friction2_val,
    ):
        # State variables
        q1, q2, q1_dot, q2_dot = sp.symbols("q1 q2 q1_dot q2_dot", real=True)
        tau1, tau2 = sp.symbols("tau1 tau2", real=True)

        # Parameters
        m1, m2, l1, l2, lc1, lc2 = sp.symbols("m1 m2 l1 l2 lc1 lc2", real=True, positive=True)
        I1, I2, g, b1, b2 = sp.symbols("I1 I2 g b1 b2", real=True, positive=True)

        self.parameters = {
            m1: m1_val,
            m2: m2_val,
            l1: l1_val,
            l2: l2_val,
            lc1: lc1_val,
            lc2: lc2_val,
            I1: I1_val,
            I2: I2_val,
            g: gravity_val,
            b1: friction1_val,
            b2: friction2_val,
        }

        self.state_vars = [q1, q2, q1_dot, q2_dot]
        self.control_vars = [tau1, tau2]
        self.output_vars = [q1, q2]
        self.order = 2

        # Mass matrix M(q)
        M11 = m1 * lc1**2 + m2 * (l1**2 + lc2**2 + 2 * l1 * lc2 * sp.cos(q2)) + I1 + I2
        M12 = m2 * (lc2**2 + l1 * lc2 * sp.cos(q2)) + I2
        M21 = M12
        M22 = m2 * lc2**2 + I2

        # Coriolis and centrifugal terms C(q, q_dot)
        h = -m2 * l1 * lc2 * sp.sin(q2)
        C1 = h * (2 * q1_dot * q2_dot + q2_dot**2)
        C2 = -h * q1_dot**2

        # Gravity terms G(q)
        G1 = (m1 * lc1 + m2 * l1) * g * sp.cos(q1) + m2 * lc2 * g * sp.cos(q1 + q2)
        G2 = m2 * lc2 * g * sp.cos(q1 + q2)

        # Friction
        F1 = b1 * q1_dot
        F2 = b2 * q2_dot

        # Solve for accelerations: M * q_ddot = tau - C - G - F
        # q_ddot = M^(-1) * (tau - C - G - F)
        det_M = M11 * M22 - M12 * M21

        # Inverse of M
        M_inv_11 = M22 / det_M
        M_inv_12 = -M12 / det_M
        M_inv_21 = -M21 / det_M
        M_inv_22 = M11 / det_M

        # Right-hand side
        rhs1 = tau1 - C1 - G1 - F1
        rhs2 = tau2 - C2 - G2 - F2

        # Accelerations
        q1_ddot = M_inv_11 * rhs1 + M_inv_12 * rhs2
        q2_ddot = M_inv_21 * rhs1 + M_inv_22 * rhs2

        self._f_sym = sp.Matrix([q1_ddot, q2_ddot])
        self._h_sym = sp.Matrix([q1, q2])

    def setup_equilibria(self):
        # method to add equilibria to the system automatically after initialization
        # adds the hanging and inverted equilibria (I didn't care to calculate the torque
        # for the horizontal one)
        
        self.add_equilibrium(
            'stable hanging',
            x_eq=np.array([np.pi, 0.0, 0.0, 0.0]),
            u_eq=np.array([0.0, 0.0]),
            verify=True,
            stability='stable'
            )

        self.add_equilibrium(
            'unstable upright',
            x_eq=np.array([np.pi/2, 0.0, 0.0, 0.0]),
            u_eq=np.array([0.0, 0.0]),
            stability='unstable',
            )
        
        self.set_default_equilibrium(name="unstable upright")