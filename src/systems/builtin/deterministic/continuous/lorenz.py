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


class Lorenz(ContinuousSymbolicSystem):
    """
    Lorenz system - famous chaotic dynamical system from atmospheric convection.

    Physical System:
    ---------------
    A simplified model of atmospheric convection. The system models:
    - Fluid circulation in a heated layer between two plates
    - Rate of convective overturning (x)
    - Horizontal temperature variation (y)
    - Vertical temperature variation (z)

    State Space:
    -----------
    State: x = [x, y, z]
        - x: Rate of convective motion [dimensionless]
          * x > 0: clockwise circulation
          * x < 0: counterclockwise circulation
          * Proportional to velocity of fluid flow

        - y: Horizontal temperature variation [dimensionless]
          * y > 0: warmer on one side
          * y < 0: warmer on other side
          * Temperature difference driving convection

        - z: Vertical temperature variation from linearity [dimensionless]
          * z > 0: more stratified (stable)
          * z < 0: less stratified (unstable)
          * Deviation from conductive temperature profile

    Output: y = [x, y]
        - Partial observation: measures x and y, not z
        - Models limited sensor availability
        - Creates observability challenges for state estimation

    Dynamics:
    --------
    The Lorenz equations:

        ẋ = σ(y - x)
        ẏ = x(ρ - z) - y
        ż = xy - βz

    **First equation (convection rate)**:
    - σ(y - x): Proportional to temperature difference
    - σ (sigma): Prandtl number - ratio of viscosity to thermal diffusivity
    - Drives x toward y at rate σ

    **Second equation (horizontal temperature)**:
    - x(ρ - z): Nonlinear coupling - convection affects temperature
    - ρ (rho): Rayleigh number - ratio of buoyancy to viscous forces
    - -y: Damping term (heat diffusion)
    - When z < ρ, convection x amplifies y

    **Third equation (vertical temperature)**:
    - xy: Nonlinear product - convection creates temperature gradients
    - -βz: Damping/relaxation toward linear profile
    - β (beta): Geometric factor (aspect ratio of convection cell)

    Parameters:
    ----------
    sigma : float, default=10.0
        Prandtl number [dimensionless]. Ratio of momentum diffusivity
        (viscosity) to thermal diffusivity. Standard Chaotic Lorenz: σ = 10
        Higher σ → faster adjustment of x to y

    rho : float, default=28.0
        Rayleigh number [dimensionless]. Measures temperature difference
        driving convection relative to dissipative effects. Critical values:
        - ρ < 1: No convection (conduction only)
        - 1 < ρ < 24.74: Steady convection
        - ρ > 24.74: Chaotic behavior possible
        - ρ = 28: Classic chaotic Lorenz attractor
        Higher ρ → stronger driving force

    beta : float, default=8/3
        Geometric factor [dimensionless]. Related to aspect ratio of
        convection cell (width/height). Standard value 8/3 ≈ 2.667 gives
        the classic "butterfly" attractor shape.
        - Affects dissipation rate in z
        - Controls attractor shape and size

    Equilibria:
    ----------
    **Origin (unstable for ρ > 1)**:
        x_eq = [0, 0, 0]  (no convection)

    Stable when ρ < 1 (conduction dominates).
    Unstable when ρ > 1 (convection develops).

    **Convective equilibria (for ρ > 1)**:
        C+ = [√(β(ρ-1)), √(β(ρ-1)), ρ-1]
        C- = [-√(β(ρ-1)), -√(β(ρ-1)), ρ-1]

    These represent steady clockwise (C+) and counterclockwise (C-)
    convection cells. Both become unstable for ρ > 24.74, leading to chaos.

    Behavior Regimes:
    ----------------
    1. **ρ < 1 (No convection)**:
       - Origin is stable
       - All trajectories decay to zero
       - Heat transported by conduction only

    2. **1 < ρ < 13.926 (Steady convection)**:
       - Origin becomes unstable
       - C+ or C- are stable (bistable system)
       - Steady convection cells form

    3. **13.926 < ρ < 24.74 (Periodic/complex)**:
       - C+ and C- lose stability
       - Can have limit cycles or complex behavior

    4. **ρ > 24.74 (Chaos)**:
       - Chaotic behavior emerges
       - Sensitive dependence on initial conditions
       - Strange attractor (Lorenz butterfly)

    5. **ρ = 28 (Classic chaos)**:
       - Well-studied chaotic attractor
       - Fractal structure
       - Positive Lyapunov exponent

    The Lorenz Attractor:
    --------------------
    For standard parameters (σ=10, ρ=28, β=8/3):
    - **Shape**: Two wing-like lobes (butterfly shape)
    - **Structure**: Strange attractor (fractal dimension ≈ 2.06)
    - **Behavior**: Trajectories spiral around C+ or C-, occasionally
      switching between wings
    - **Predictability**: Initial condition error doubles ~every 2 time units
    - **Volume contraction**: Phase space volume shrinks → dissipative system

    Physical Interpretation:
    -------------------------------------------------
    - x: Velocity of convection roll
    - y: Temperature difference between ascending and descending fluid
    - z: Deviation from linear temperature profile
    - ρ: Driving force (heating from below)
    - σ: Fluid properties (viscosity vs. thermal conductivity)
    - β: Cell geometry

    See Also:
    --------
    DuffingOscillator : Another chaotic system (forced oscillator)
    VanDerPolOscillator : Limit cycle oscillator
    NonlinearChainSystem : Coupled oscillators with complex dynamics
    """

    def define_system(
        self, sigma_val: float = 10.0, rho_val: float = 28.0, beta_val: float = 8.0 / 3.0
    ):
        x, y, z = sp.symbols("x y z", real=True)
        sigma, rho, beta = sp.symbols("sigma rho beta", real=True, positive=True)

        self.parameters = {sigma: sigma_val, rho: rho_val, beta: beta_val}

        self.state_vars = [x, y, z]
        self.control_vars = []
        self.output_vars = []
        self.order = 1

        # Lorenz dynamics
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z

        self._f_sym = sp.Matrix([dx, dy, dz])
        self._h_sym = sp.Matrix([x, y])

    def setup_equilibria(self):
        # method to add equilibria to the system automatically after initialization
        # origin is always added regardless, so nothing needs to be done
        pass


class ControlledLorenz(ContinuousSymbolicSystem):
    """
    Lorenz system - famous chaotic dynamical system from atmospheric convection with a forcing term

    Physical System:
    ---------------
    A simplified model of atmospheric convection. The system models:
    - Fluid circulation in a heated layer between two plates
    - Rate of convective overturning (x)
    - Horizontal temperature variation (y)
    - Vertical temperature variation (z)

    State Space:
    -----------
    State: x = [x, y, z]
        - x: Rate of convective motion [dimensionless]
          * x > 0: clockwise circulation
          * x < 0: counterclockwise circulation
          * Proportional to velocity of fluid flow

        - y: Horizontal temperature variation [dimensionless]
          * y > 0: warmer on one side
          * y < 0: warmer on other side
          * Temperature difference driving convection

        - z: Vertical temperature variation from linearity [dimensionless]
          * z > 0: more stratified (stable)
          * z < 0: less stratified (unstable)
          * Deviation from conductive temperature profile

    Control: u = [u]
        - u: External forcing/perturbation [dimensionless]
        - Typically u = 0 for studying natural chaos
        - Can be used to control or suppress chaos

    Output: y = [x, y]
        - Partial observation: measures x and y, not z
        - Models limited sensor availability
        - Creates observability challenges for state estimation

    Dynamics:
    --------
    The Lorenz equations with control:

        ẋ = σ(y - x) + u
        ẏ = x(ρ - z) - y
        ż = xy - βz

    **First equation (convection rate)**:
    - σ(y - x): Proportional to temperature difference
    - σ (sigma): Prandtl number - ratio of viscosity to thermal diffusivity
    - Drives x toward y at rate σ
    - Control u added here for external forcing

    **Second equation (horizontal temperature)**:
    - x(ρ - z): Nonlinear coupling - convection affects temperature
    - ρ (rho): Rayleigh number - ratio of buoyancy to viscous forces
    - -y: Damping term (heat diffusion)
    - When z < ρ, convection x amplifies y

    **Third equation (vertical temperature)**:
    - xy: Nonlinear product - convection creates temperature gradients
    - -βz: Damping/relaxation toward linear profile
    - β (beta): Geometric factor (aspect ratio of convection cell)

    Parameters:
    ----------
    sigma : float, default=10.0
        Prandtl number [dimensionless]. Ratio of momentum diffusivity
        (viscosity) to thermal diffusivity. Standard Chaotic Lorenz: σ = 10
        Higher σ → faster adjustment of x to y

    rho : float, default=28.0
        Rayleigh number [dimensionless]. Measures temperature difference
        driving convection relative to dissipative effects. Critical values:
        - ρ < 1: No convection (conduction only)
        - 1 < ρ < 24.74: Steady convection
        - ρ > 24.74: Chaotic behavior possible
        - ρ = 28: Classic chaotic Lorenz attractor
        Higher ρ → stronger driving force

    beta : float, default=8/3
        Geometric factor [dimensionless]. Related to aspect ratio of
        convection cell (width/height). Standard value 8/3 ≈ 2.667 gives
        the classic "butterfly" attractor shape.
        - Affects dissipation rate in z
        - Controls attractor shape and size

    Equilibria:
    ----------
    **Origin (unstable for ρ > 1)**:
        x_eq = [0, 0, 0]  (no convection)
        u_eq = 0

    Stable when ρ < 1 (conduction dominates).
    Unstable when ρ > 1 (convection develops).

    **Convective equilibria (for ρ > 1)**:
        C+ = [√(β(ρ-1)), √(β(ρ-1)), ρ-1]
        C- = [-√(β(ρ-1)), -√(β(ρ-1)), ρ-1]

    These represent steady clockwise (C+) and counterclockwise (C-)
    convection cells. Both become unstable for ρ > 24.74, leading to chaos.

    Behavior Regimes:
    ----------------
    1. **ρ < 1 (No convection)**:
       - Origin is stable
       - All trajectories decay to zero
       - Heat transported by conduction only

    2. **1 < ρ < 13.926 (Steady convection)**:
       - Origin becomes unstable
       - C+ or C- are stable (bistable system)
       - Steady convection cells form

    3. **13.926 < ρ < 24.74 (Periodic/complex)**:
       - C+ and C- lose stability
       - Can have limit cycles or complex behavior

    4. **ρ > 24.74 (Chaos)**:
       - Chaotic behavior emerges
       - Sensitive dependence on initial conditions
       - Strange attractor (Lorenz butterfly)

    5. **ρ = 28 (Classic chaos)**:
       - Well-studied chaotic attractor
       - Fractal structure
       - Positive Lyapunov exponent

    The Lorenz Attractor:
    --------------------
    For standard parameters (σ=10, ρ=28, β=8/3):
    - **Shape**: Two wing-like lobes (butterfly shape)
    - **Structure**: Strange attractor (fractal dimension ≈ 2.06)
    - **Behavior**: Trajectories spiral around C+ or C-, occasionally
      switching between wings
    - **Predictability**: Initial condition error doubles ~every 2 time units
    - **Volume contraction**: Phase space volume shrinks → dissipative system

    Physical Interpretation:
    -------------------------------------------------
    - x: Velocity of convection roll
    - y: Temperature difference between ascending and descending fluid
    - z: Deviation from linear temperature profile
    - ρ: Driving force (heating from below)
    - σ: Fluid properties (viscosity vs. thermal conductivity)
    - β: Cell geometry

    See Also:
    --------
    DuffingOscillator : Another chaotic system (forced oscillator)
    VanDerPolOscillator : Limit cycle oscillator
    NonlinearChainSystem : Coupled oscillators with complex dynamics
    """

    def define_system(
        self, sigma_val: float = 10.0, rho_val: float = 28.0, beta_val: float = 8.0 / 3.0
    ):
        x, y, z = sp.symbols("x y z", real=True)
        u = sp.symbols("u", real=True)
        sigma, rho, beta = sp.symbols("sigma rho beta", real=True, positive=True)

        self.parameters = {sigma: sigma_val, rho: rho_val, beta: beta_val}

        self.state_vars = [x, y, z]
        self.control_vars = [u]
        self.output_vars = []
        self.order = 1

        # Lorenz dynamics with control
        dx = sigma * (y - x) + u
        dy = x * (rho - z) - y
        dz = x * y - beta * z

        self._f_sym = sp.Matrix([dx, dy, dz])
        self._h_sym = sp.Matrix([x, y])

    def setup_equilibria(self):
        # method to add equilibria to the system automatically after initialization
        # origin is always added regardless, so nothing needs to be done
        pass
