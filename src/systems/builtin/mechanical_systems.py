import sympy as sp
import numpy as np
import torch
from symbolic_dynamical_system import SymbolicDynamicalSystem


class SymbolicPendulum(SymbolicDynamicalSystem):
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

    def __init__(
        self, m: float = 1.0, l: float = 1.0, beta: float = 1.0, g: float = 9.81
    ):
        super().__init__()
        self.order = 1
        # Store values for backward compatibility
        self.m_val = m
        self.l_val = l
        self.beta_val = beta
        self.g_val = g
        self.inertia_val = m * l**2
        self.define_system(m, l, beta, g)

    def define_system(self, m_val, l_val, beta_val, g_val):
        theta, theta_dot = sp.symbols("theta theta_dot", real=True)
        u = sp.symbols("u", real=True)
        m, l, beta, g = sp.symbols("m l beta g", real=True, positive=True)

        self.parameters = {m: m_val, l: l_val, beta: beta_val, g: g_val}
        self.state_vars = [theta, theta_dot]
        self.control_vars = [u]
        self.output_vars = [theta]

        ml2 = m * l * l
        self._f_sym = sp.Matrix(
            [theta_dot, (-beta / ml2) * theta_dot + (g / l) * sp.sin(theta) + u / ml2]
        )
        self._h_sym = sp.Matrix([theta])

    @property
    def inertia(self):
        """For backward compatibility"""
        return self.inertia_val


class SymbolicPendulum2ndOrder(SymbolicDynamicalSystem):
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

    def __init__(self, m=1.0, l=1.0, beta=1.0, g=9.81):
        super().__init__()
        self.order = 2  # CRITICAL: Mark as second-order
        self.m_val = m
        self.l_val = l
        self.beta_val = beta
        self.g_val = g
        self.inertia_val = m * l**2
        self.define_system(m, l, beta, g)

    def define_system(self, m_val, l_val, beta_val, g_val):
        # State: [theta, theta_dot]
        theta, theta_dot = sp.symbols("theta theta_dot", real=True)
        u = sp.symbols("u", real=True)
        m, l, beta, g = sp.symbols("m l beta g", real=True, positive=True)

        self.parameters = {m: m_val, l: l_val, beta: beta_val, g: g_val}
        self.state_vars = [theta, theta_dot]
        self.control_vars = [u]
        self.output_vars = [theta]  # Observe angle only

        ml2 = m * l * l

        # Second-order system: return ONLY acceleration
        theta_ddot = (-beta / ml2) * theta_dot + (g / l) * sp.sin(theta) + u / ml2

        self._f_sym = sp.Matrix([theta_ddot])  # ← Single element!
        self._h_sym = sp.Matrix([theta])

    @property
    def inertia(self):
        """For backward compatibility"""
        return self.inertia_val

    @property
    def nq(self) -> int:
        """Number of generalized coordinates"""
        return 1  # One angle

class FifthOrderMechanicalSystem(SymbolicDynamicalSystem):
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

    def __init__(
        self,
        m: float = 1.0,
        k: float = 1.0,
        c1: float = 0.1,
        c2: float = 0.05,
        c3: float = 0.01,
        g: float = 9.81,
    ):
        super().__init__()
        self.order = 5
        # Store values for backward compatibility
        self.m_val = m
        self.k_val = k
        self.c1_val = c1
        self.c2_val = c2
        self.c3_val = c3
        self.g_val = g
        self.define_system(m, k, c1, c2, c3, g)

    def define_system(self, m_val, k_val, c1_val, c2_val, c3_val, g_val):
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

        # Fifth derivative: complex dynamics with multiple damping terms
        q5 = -k / m * q - c1 * q1 - c2 * q2 - c3 * q3 - 0.01 * q4 - g + u / m

        self._f_sym = sp.Matrix([q5])
        self._h_sym = sp.Matrix([q, q1])

    @property
    def x_equilibrium(self) -> torch.Tensor:
        q_eq = -self.g_val * self.m_val / self.k_val
        return torch.tensor([q_eq, 0.0, 0.0, 0.0, 0.0])

    @property
    def u_equilibrium(self) -> torch.Tensor:
        return torch.tensor([self.m_val * self.g_val])


class CoupledOscillatorSystem(SymbolicDynamicalSystem):
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

    def __init__(
        self,
        m1: float = 1.0,
        m2: float = 0.5,
        k1: float = 2.0,
        k2: float = 1.0,
        k_coupling: float = 0.5,
        c: float = 0.1,
        J: float = 0.1,
    ):
        super().__init__()
        self.order = 1
        # Store values
        self.m1_val = m1
        self.m2_val = m2
        self.k1_val = k1
        self.k2_val = k2
        self.k_coupling_val = k_coupling
        self.c_val = c
        self.J_val = J
        self.define_system(m1, m2, k1, k2, k_coupling, c, J)

    def define_system(
        self, m1_val, m2_val, k1_val, k2_val, k_coupling_val, c_val, J_val
    ):
        x1, x2, v1, v2, theta = sp.symbols("x1 x2 v1 v2 theta", real=True)
        u1, u2 = sp.symbols("u1 u2", real=True)
        m1, m2, k1, k2, k_c, c, J = sp.symbols(
            "m1 m2 k1 k2 k_c c J", real=True, positive=True
        )

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

        # Coupled dynamics
        dx1 = v1
        dv1 = -k1 / m1 * x1 - k_c / m1 * (x1 - x2) - c / m1 * v1 + u1 / m1
        dx2 = v2
        dv2 = (
            -k2 / m2 * x2
            - k_c / m2 * (x2 - x1)
            - c / m2 * v2
            + sp.sin(theta) / m2
            + u2 / m2
        )
        dtheta = -theta / J - x2 / J + u2 / (2 * J)

        self._f_sym = sp.Matrix([dx1, dx2, dv1, dv2, dtheta])
        self._h_sym = sp.Matrix([x1, x2, theta])

    @property
    def x_equilibrium(self) -> torch.Tensor:
        return torch.zeros(5)

    @property
    def u_equilibrium(self) -> torch.Tensor:
        return torch.zeros(2)


class NonlinearChainSystem(SymbolicDynamicalSystem):
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

    def __init__(self, k: float = 1.0, c: float = 0.1, alpha: float = 0.1):
        super().__init__()
        self.order = 1
        # Store values
        self.k_val = k
        self.c_val = c
        self.alpha_val = alpha
        self.define_system(k, c, alpha)

    def define_system(self, k_val, c_val, alpha_val):
        x1, x2, x3, x4, x5 = sp.symbols("x1 x2 x3 x4 x5", real=True)
        u = sp.symbols("u", real=True)
        k, c, alpha = sp.symbols("k c alpha", real=True, positive=True)

        self.parameters = {k: k_val, c: c_val, alpha: alpha_val}

        self.state_vars = [x1, x2, x3, x4, x5]
        self.control_vars = [u]
        self.output_vars = [x1, x3, x5]

        # Chain dynamics with nonlinear coupling
        dx1 = -k * x1 - c * x1 + alpha * sp.sin(x2 - x1) + u
        dx2 = -k * x2 - c * x2 + alpha * sp.sin(x1 - x2) + alpha * sp.sin(x3 - x2)
        dx3 = -k * x3 - c * x3 + alpha * sp.sin(x2 - x3) + alpha * sp.sin(x4 - x3)
        dx4 = -k * x4 - c * x4 + alpha * sp.sin(x3 - x4) + alpha * sp.sin(x5 - x4)
        dx5 = -k * x5 - c * x5 + alpha * sp.sin(x4 - x5)

        self._f_sym = sp.Matrix([dx1, dx2, dx3, dx4, dx5])
        self._h_sym = sp.Matrix([x1, x3, x5])

    @property
    def x_equilibrium(self) -> torch.Tensor:
        return torch.zeros(5)

    @property
    def u_equilibrium(self) -> torch.Tensor:
        return torch.zeros(1)


class CartPole(SymbolicDynamicalSystem):
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

    See Also:
    --------
    SymbolicPendulum : Simpler version without cart
    PVTOL : Another underactuated system with similar challenges
    """

    def __init__(
        self,
        m_cart: float = 1.0,
        m_pole: float = 0.1,
        length: float = 0.5,
        gravity: float = 9.81,
        friction: float = 0.1,
    ):
        super().__init__()
        self.order = 2
        # Store values
        self.m_cart_val = m_cart
        self.m_pole_val = m_pole
        self.length_val = length
        self.gravity_val = gravity
        self.friction_val = friction
        self.define_system(m_cart, m_pole, length, gravity, friction)

    def define_system(
        self, m_cart_val, m_pole_val, length_val, gravity_val, friction_val
    ):
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
            F
            - b * x_dot
            + mp * l * theta_dot**2 * sin_theta
            - mp * g * sin_theta * cos_theta
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

    @property
    def x_equilibrium(self) -> torch.Tensor:
        """Upright equilibrium at origin"""
        return torch.zeros(4)

    @property
    def u_equilibrium(self) -> torch.Tensor:
        """No force needed at equilibrium"""
        return torch.zeros(1)

class DubinsVehicle(SymbolicDynamicalSystem):
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

    def __init__(self):
        super().__init__()
        self.order = 1
        self.define_system()

    def define_system(self):
        x, y, theta = sp.symbols("x y theta", real=True)
        v, omega = sp.symbols("v omega", real=True)

        self.parameters = {}
        self.state_vars = [x, y, theta]
        self.control_vars = [v, omega]
        self.output_vars = [x, y, theta]

        # Kinematic equations
        dx = v * sp.cos(theta)
        dy = v * sp.sin(theta)
        dtheta = omega

        self._f_sym = sp.Matrix([dx, dy, dtheta])
        self._h_sym = sp.Matrix([x, y, theta])

    @property
    def x_equilibrium(self) -> torch.Tensor:
        return torch.zeros(3)

    @property
    def u_equilibrium(self) -> torch.Tensor:
        return torch.zeros(2)


class Manipulator2Link(SymbolicDynamicalSystem):
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
        super().__init__()
        self.order = 2
        # Store values
        self.m1_val = m1
        self.m2_val = m2
        self.l1_val = l1
        self.l2_val = l2
        self.lc1_val = lc1
        self.lc2_val = lc2
        self.I1_val = I1
        self.I2_val = I2
        self.gravity_val = gravity
        self.friction1_val = friction1
        self.friction2_val = friction2
        self.define_system(
            m1, m2, l1, l2, lc1, lc2, I1, I2, gravity, friction1, friction2
        )

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
        m1, m2, l1, l2, lc1, lc2 = sp.symbols(
            "m1 m2 l1 l2 lc1 lc2", real=True, positive=True
        )
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

    @property
    def x_equilibrium(self) -> torch.Tensor:
        """Hanging down equilibrium"""
        return torch.tensor([sp.pi, 0.0, 0.0, 0.0])

    @property
    def u_equilibrium(self) -> torch.Tensor:
        """Zero torque at hanging equilibrium"""
        return torch.zeros(2)


class PathTracking(SymbolicDynamicalSystem):
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

    def __init__(self, speed: float = 1.0, length: float = 1.0, radius: float = 10.0):
        super().__init__()
        self.order = 1
        self.speed_val = speed
        self.length_val = length
        self.radius_val = radius
        self.define_system(speed, length, radius)

    def define_system(self, speed_val, length_val, radius_val):
        d_e, theta_e = sp.symbols("d_e theta_e", real=True)
        delta = sp.symbols("delta", real=True)
        v, L, R = sp.symbols("v L R", real=True, positive=True)

        self.parameters = {v: speed_val, L: length_val, R: radius_val}

        self.state_vars = [d_e, theta_e]
        self.control_vars = [delta]
        self.output_vars = [d_e, theta_e]

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

    @property
    def x_equilibrium(self) -> torch.Tensor:
        return torch.zeros(2)

    @property
    def u_equilibrium(self) -> torch.Tensor:
        return torch.tensor([self.length_val / self.radius_val])