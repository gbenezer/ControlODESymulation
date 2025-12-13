import sympy as sp
import numpy as np
import torch
from symbolic_dynamical_system import SymbolicDynamicalSystem


class VanDerPolOscillator(SymbolicDynamicalSystem):
    """
    Van der Pol oscillator - self-excited nonlinear oscillator with limit cycle.

    Physical System:
    ---------------
    Originally meant to model electronic oscillator circuits. The system exhibits self-sustained
    oscillations.

    The key feature is **nonlinear damping**:
    - Near origin: negative damping (pumps energy in)
    - Far from origin: positive damping (dissipates energy)
    - Result: stable limit cycle (periodic orbit)

    State Space:
    -----------
    State: x = [x, y]
        - x: Primary variable [V or dimensionless]
          * In electrical circuit: voltage or current
          * In general: oscillating quantity

        - y: Derivative-related variable [V/s or dimensionless]
          * y ≈ ẋ for μ → 0
          * Not exactly velocity for μ > 0 (includes nonlinear term)

    Control: u = [u]
        - u: External forcing/input [V or dimensionless]
        - Can perturb the natural oscillation
        - Can be used for synchronization or frequency control

    Output: y_out = [x]
        - Measures only x (the oscillating variable)
        - Partial observation (y not directly measured)

    Dynamics:
    --------
    The Van der Pol equation in standard form:

        ẋ = y
        ẏ = μ(1 - x²)y - x + u

    Or as a second-order ODE:
        ẍ - μ(1 - x²)ẋ + x = u

    **First equation**: Simply defines y ≈ ẋ

    **Second equation**:
    - μ(1 - x²)y: Nonlinear damping (Van der Pol term)
      * When |x| < 1: (1 - x²) > 0 → negative damping (adds energy)
      * When |x| > 1: (1 - x²) < 0 → positive damping (removes energy)
      * Balance creates stable limit cycle

    - -x: Linear restoring force (like harmonic oscillator)
      * Provides natural frequency ω₀ ≈ 1

    - u: External forcing/control

    Parameters:
    ----------
    mu : float, default=1.0
        Nonlinearity parameter [dimensionless].
        Controls strength of nonlinear damping and oscillation shape:

        - **μ → 0**: Nearly sinusoidal (harmonic oscillator)
          * Period T ≈ 2π
          * Smooth, sinusoidal limit cycle

        - **μ = 1**: Standard Van der Pol
          * Period T ≈ 6.7
          * Mildly distorted sinusoid

        - **μ >> 1**: Relaxation oscillations
          * Period T ≈ (3 - 2ln(2))μ ≈ 1.614μ
          * Sharp "fast" and "slow" phases
          * Almost discontinuous (spikes and plateaus)

    Behavior Regimes:
    ----------------
    **1. Small μ (μ < 0.1): Harmonic-like**
    - Nearly sinusoidal oscillations
    - Frequency ≈ 1 rad/s
    - Smooth limit cycle
    - Weak nonlinearity

    **2. Moderate μ (0.1 < μ < 3): Nonlinear oscillations**
    - Visible waveform distortion
    - Frequency slightly reduced
    - Standard Van der Pol behavior

    **3. Large μ (μ > 3): Relaxation oscillations**
    - Two-timescale dynamics
    - Fast jumps between slow plateaus
    - Very non-sinusoidal
    - Period proportional to μ

    Equilibrium:
    -----------
    **Origin (unstable)**:
        x_eq = [0, 0]
        u_eq = 0

    For u = 0, the origin is:
    - Unstable focus (spiral): trajectories spiral outward
    - All trajectories (except origin) approach the limit cycle
    - Eigenvalues: λ = μ/2 ± i√(4-μ²)/2
      * Real part positive (unstable)
      * Imaginary part gives oscillation frequency

    Limit Cycle:
    -----------
    For u = 0, the system has a unique **stable limit cycle**:

    **Properties**:
    - Globally attracting (except from origin)
    - Isolated (no nearby periodic orbits)
    - Amplitude ≈ 2 for all μ (approximately)
    - Period depends on μ:
      * μ → 0: T → 2π (harmonic)
      * μ = 1: T ≈ 6.7
      * μ >> 1: T ≈ 1.614μ

    **Basin of attraction**: Entire plane except origin
    - Any non-zero initial condition → limit cycle
    - Time to converge depends on distance from cycle

    Relaxation Oscillations (μ >> 1):
    ---------------------------------
    For large μ, the system exhibits relaxation oscillations:

    **Mechanism**:
    1. **Slow phase**: x grows slowly along stable manifold
    2. **Jump**: At x ≈ 1, rapid transition (fast manifold)
    3. **Slow phase**: x decreases slowly along stable manifold
    4. **Jump**: At x ≈ -1, rapid transition back
    5. Repeat

    **Characteristics**:
    - Distinct timescales (ε = 1/μ is small parameter)
    - Almost piecewise linear trajectory
    - Useful model for on-off systems (heart beats, neurons)

    See Also:
    --------
    DuffingOscillator : Another nonlinear oscillator (can be chaotic)
    Lorenz : 3D system that exhibits chaos
    NonlinearChainSystem : Multiple coupled oscillators
    """

    def __init__(self, mu: float = 1.0):
        super().__init__()
        self.order = 1
        self.mu_val = mu
        self.define_system(mu)

    def define_system(self, mu_val):
        x, y = sp.symbols("x y", real=True)
        u = sp.symbols("u", real=True)
        mu = sp.symbols("mu", real=True, positive=True)

        self.parameters = {mu: mu_val}
        self.state_vars = [x, y]
        self.control_vars = [u]
        self.output_vars = [x]

        # Van der Pol dynamics: d²x/dt² - μ(1-x²)dx/dt + x = u
        # Rewritten as first-order system
        dx = y
        dy = mu * (1 - x**2) * y - x + u

        self._f_sym = sp.Matrix([dx, dy])
        self._h_sym = sp.Matrix([x])

    @property
    def x_equilibrium(self) -> torch.Tensor:
        return torch.zeros(2)

    @property
    def u_equilibrium(self) -> torch.Tensor:
        return torch.zeros(1)


class Lorenz(SymbolicDynamicalSystem):
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

    def __init__(self, sigma: float = 10.0, rho: float = 28.0, beta: float = 8.0 / 3.0):
        super().__init__()
        self.order = 1
        self.sigma_val = sigma
        self.rho_val = rho
        self.beta_val = beta
        self.define_system(sigma, rho, beta)

    def define_system(self, sigma_val, rho_val, beta_val):
        x, y, z = sp.symbols("x y z", real=True)
        u = sp.symbols("u", real=True)
        sigma, rho, beta = sp.symbols("sigma rho beta", real=True, positive=True)

        self.parameters = {sigma: sigma_val, rho: rho_val, beta: beta_val}

        self.state_vars = [x, y, z]
        self.control_vars = [u]
        self.output_vars = [x, y]

        # Lorenz dynamics with control
        dx = sigma * (y - x) + u
        dy = x * (rho - z) - y
        dz = x * y - beta * z

        self._f_sym = sp.Matrix([dx, dy, dz])
        self._h_sym = sp.Matrix([x, y])

    @property
    def x_equilibrium(self) -> torch.Tensor:
        """Origin (unstable for standard parameters)"""
        return torch.zeros(3)

    @property
    def u_equilibrium(self) -> torch.Tensor:
        return torch.zeros(1)


class DuffingOscillator(SymbolicDynamicalSystem):
    """
    Duffing oscillator - nonlinear oscillator with cubic stiffness term.

    Physical System:
    ---------------
    A mass-spring-damper system where the spring force is nonlinear,
    containing both linear and cubic terms.

    The system consists of:
    - A mass attached to a nonlinear spring
    - Linear viscous damping
    - Optional periodic forcing (through control input)

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

    Control: u = [u]
        - u: External forcing [N or dimensionless]
        - Often periodic: u(t) = γ·cos(ω·t) for studying resonance
        - Can be feedback control for stabilization

    Output: y = [x]
        - Measures displacement only (typical for position sensors)
        - Partial observation (velocity not directly measured)

    Dynamics:
    --------
    The Duffing equation in first-order form:

        ẋ = v
        v̇ = -δv - αx - βx³ + u

    Or as a second-order ODE:
        ẍ + δẋ + αx + βx³ = u

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

    - u: External forcing/control

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

    gamma : float, default=0.0
        Forcing amplitude [N or dimensionless].
        For studying forced response: set γ > 0 and use
        u(t) = γ·cos(ωt) as control input.
        Standard chaotic case: γ ≈ 0.3 - 0.5

    omega : float, default=1.0
        Forcing frequency [rad/s].
        For periodic forcing u(t) = γ·cos(ωt).
        Chaos often occurs near ω ≈ 1.0 (near resonance)

    Potential Energy:
    ----------------
    For unforced, undamped case (u=0, δ=0), the system is conservative
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
    For unforced system (u = 0):

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
    **1. Unforced (γ = 0 or u = 0)**:
    - Monostable: Oscillations decay to origin
    - Bistable: Oscillations decay to one of two wells
    - Basin boundary separates initial conditions

    **2. Periodic forcing (u = γ·cos(ωt))**:
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

    def __init__(
        self,
        alpha: float = -1.0,
        beta: float = 1.0,
        delta: float = 0.3,
        gamma: float = 0.0,
        omega: float = 1.0,
    ):
        super().__init__()
        self.order = 1
        self.alpha_val = alpha
        self.beta_val = beta
        self.delta_val = delta
        self.gamma_val = gamma
        self.omega_val = omega
        self.define_system(alpha, beta, delta, gamma, omega)

    def define_system(self, alpha_val, beta_val, delta_val, gamma_val, omega_val):
        x, v = sp.symbols("x v", real=True)
        u = sp.symbols("u", real=True)
        alpha, beta, delta, gamma, omega = sp.symbols(
            "alpha beta delta gamma omega", real=True
        )

        self.parameters = {
            alpha: alpha_val,
            beta: beta_val,
            delta: delta_val,
            gamma: gamma_val,
            omega: omega_val,
        }

        self.state_vars = [x, v]
        self.control_vars = [u]
        self.output_vars = [x]

        # Duffing equation: d²x/dt² + delta*dx/dt + alpha*x + beta*x³ = gamma*cos(omega*t) + u
        # First-order form
        dx = v
        dv = -delta * v - alpha * x - beta * x**3 + u

        self._f_sym = sp.Matrix([dx, dv])
        self._h_sym = sp.Matrix([x])

    @property
    def x_equilibrium(self) -> torch.Tensor:
        return torch.zeros(2)

    @property
    def u_equilibrium(self) -> torch.Tensor:
        return torch.zeros(1)
