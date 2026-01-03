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


class ControlledDuffingOscillator(ContinuousSymbolicSystem):
    """
    Duffing oscillator - nonlinear oscillator with cubic stiffness term.
    # NOTE: Implementation is incomplete because there is currently no way
    # to specify time symbolically as a variable within the library frameworks
    # TODO: Modify codebase to allow for this

    Physical System:
    ---------------
    A mass-spring-damper system where the spring force is nonlinear,
    containing both linear and cubic terms.

    The system consists of:
    - A mass attached to a nonlinear spring
    - Linear viscous damping
    - Optional periodic forcing
    - Additional optional external forcing

    **Key feature**: Depending on parameters, can exhibit:
    - Bistability (two stable equilibria)
    - Jump phenomena (sudden changes in amplitude)
    - Chaos (for certain forcing parameters)
    - Multiple periodic solutions for same forcing

    State Space:
    -----------
    State: x = [x, v]
        - x: Displacement from equilibrium [m or dimensionless]
          * x = 0: Neutral position (for symmetric case)
          * Multiple equilibria possible depending on α, β

        - v: Velocity [m/s or dimensionless]
          * v = ẋ (rate of change of displacement)

    # NOTE: not fully implemented due to framework limitations
    Control: u = [u]
        - u: External forcing [N or dimensionless]
        - This control specifies custom external forcing
        - Additional periodic forcing for studying resonance controlled by
            - γ, periodic forcing amplitude
            - ω, periodic forcing phase
            - u_tot(t) = γ·cos(ω·t) + u
        - Can be feedback control for stabilization

    Output: y = [x]
        - Measures displacement only (typical for position sensors)
        - Partial observation (velocity not directly measured)

    Dynamics:
    --------
    The Duffing equation in first-order form:

        ẋ = v
        v̇ = -δv - αx - βx³ + γ·cos(ω·t) + u

    Or as a second-order ODE:
        ẍ + δẋ + αx + βx³ = γ·cos(ω·t) + u

    **Velocity equation (v̇)**:
    - -δv: Linear damping (energy dissipation)
      * δ > 0: Positive damping (stable)
      * δ = 0: Undamped (conservative)
      * δ < 0: Negative damping (unstable, pumps energy in)

    - -αx: Linear restoring force (like Hooke's law)
      * α > 0: Hardening spring at origin (stable)
      * α < 0: Softening spring at origin (unstable) → bistable
      * α = 0: Purely cubic spring

    - -βx³: Cubic nonlinear term
      * β > 0: Hardening nonlinearity (stiffens at large x)
      * β < 0: Softening nonlinearity (weakens at large x)
      * Dominates at large displacements

    - γ·cos(ω·t) + u: Periodic forcing and additional external forcing/control

    Spring Force Types:
    ------------------
    The total spring force is F_spring = αx + βx³

    1. **Hardening spring (α > 0, β > 0)**:
       - Gets stiffer with displacement
       - Single stable equilibrium at origin
       - Natural frequency increases with amplitude

    2. **Softening spring (α > 0, β < 0)**:
       - Gets softer with displacement
       - Can lose stability at large amplitude
       - Natural frequency decreases with amplitude

    3. **Bistable (α < 0, β > 0)**:
       - Double-well potential
       - Three equilibria: unstable origin + two stable wells
       - Can "snap through" between wells
       - Classic case: α = -1, β = 1

    Parameters:
    ----------
    alpha : float, default=-1.0
        Linear stiffness coefficient [1/s² or dimensionless].
        - α > 0: Monostable (single well)
        - α < 0: Bistable (double well)
        Standard Duffing: α = -1 (bistable)

    beta : float, default=1.0
        Cubic stiffness coefficient [1/(m²·s²) or dimensionless].
        - β > 0: Hardening spring
        - β < 0: Softening spring
        Standard Duffing: β = 1 (hardening)
        Together with α < 0, creates double-well potential.

    delta : float, default=0.3
        Damping coefficient [1/s or dimensionless].
        - δ = 0: Conservative (Hamiltonian)
        - δ > 0: Dissipative (stable attractors)
        - δ < 0: Negative damping (self-excited oscillations)
        Standard: δ = 0.3 (light damping)

    # NOTE: Not currently used due to framework limitations
    gamma : float, default=0.0
        Forcing amplitude [N or dimensionless].
        For studying forced response: set γ > 0 and use
        u(t) = γ·cos(ωt) as control input.
        Standard chaotic case: γ ≈ 0.3 - 0.5

    # NOTE: Not currently used due to framework limitations
    omega : float, default=1.0
        Forcing frequency [rad/s].
        For periodic forcing u(t) = γ·cos(ωt).
        Chaos often occurs near ω ≈ 1.0 (near resonance)

    Potential Energy:
    ----------------
    For unforced, undamped case (γ·cos(ω·t) + u = 0, δ = 0), the system is conservative
    with potential:
        V(x) = (α/2)x² + (β/4)x⁴

    **Bistable case (α = -1, β = 1)**:
        V(x) = -x²/2 + x⁴/4 = (x² - 1)²/4 - 1/4

    This creates a double-well potential:
    - Two minima (stable): x = ±1
    - One maximum (unstable): x = 0
    - Barrier height: V(0) - V(±1) = 1/4

    Equilibria:
    ----------
    For unforced system (γ·cos(ω·t) + u = 0):

    **Monostable case (α > 0, β > 0)**:
        x_eq = [0, 0]  (only equilibrium, stable)

    **Bistable case (α < 0, β > 0)**:
        Origin (unstable):
            x_eq = [0, 0]

        Two stable wells:
            x_eq = [±√(-α/β), 0]

        For α = -1, β = 1:
            x_eq = [±1, 0]  (stable)

    Behavior Regimes:
    ----------------
    **1. Unforced (γ·cos(ω·t) + u = 0)**:
    - Monostable: Oscillations decay to origin
    - Bistable: Oscillations decay to one of two wells
    - Basin boundary separates initial conditions

    **2. Periodic forcing (u = 0, γ·cos(ω·t) != 0)**:
    - **Periodic response**: For small γ, system responds at ω
    - **Subharmonics**: Response at ω/n (frequency division)
    - **Superharmonics**: Response at n·ω (frequency multiplication)
    - **Jump phenomenon**: Sudden amplitude change at certain ω
    - **Hysteresis**: Different response for increasing vs. decreasing ω

    **3. Chaotic forcing (moderate γ, ω near 1)**:
    - **Sensitive dependence**: Small changes → large differences
    - **Strange attractor**: Fractal structure in phase space
    - **Unpredictable**: Despite deterministic equations
    - **Window of chaos**: Chaos between periodic windows

    **Classic chaotic parameters**:
    - α = -1, β = 1, δ = 0.3, γ = 0.3, ω = 1.0

    Jump Phenomenon:
    ---------------
    For softening springs (β < 0) or hardening with forcing:
    - As forcing frequency ω is slowly varied, amplitude changes smoothly
    - At critical frequency, amplitude suddenly jumps
    - Hysteresis: different response for sweep up vs. sweep down
    - Bistability: two stable periodic solutions for same ω

    See Also:
    --------
    VanDerPolOscillator : Self-excited oscillator with limit cycle
    Lorenz : Another famous chaotic system
    SymbolicPendulum : Related but with sin(θ) nonlinearity
    """

    def define_system(
        self,
        alpha_val: float = -1.0,
        beta_val: float = 1.0,
        delta_val: float = 0.3,
        gamma_val: float = 0.0,
        omega_val: float = 1.0,
    ):
        x, v = sp.symbols("x v", real=True)
        u = sp.symbols("u", real=True)
        alpha, beta, delta, gamma, omega = sp.symbols("alpha beta delta gamma omega", real=True)

        self.parameters = {
            alpha: alpha_val,
            beta: beta_val,
            delta: delta_val,
            gamma: gamma_val,
            omega: omega_val,
        }

        self.state_vars = [x, v]
        self.control_vars = [u]
        self.output_vars = []
        self.order = 1

        # Duffing equation: d²x/dt² + delta*dx/dt + alpha*x + beta*x³ = gamma*cos(omega*t) + u
        # First-order form
        dx = v
        dv = -delta * v - alpha * x - beta * x**3 + u

        self._f_sym = sp.Matrix([dx, dv])
        self._h_sym = sp.Matrix([x])

    def setup_equilibria(self):
        # method to add equilibria to the system automatically after initialization
        # origin is always added regardless.
        # TODO: once time can be used symbolically, implement parameter-dependent method
        # for equilibria specification
        pass
