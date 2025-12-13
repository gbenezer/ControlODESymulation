import sympy as sp
import numpy as np
import torch
from symbolic_dynamical_system import SymbolicDynamicalSystem


class SymbolicQuadrotor2D(SymbolicDynamicalSystem):
    """
    Planar quadrotor (quadcopter) - second-order formulation.

    Physical System:
    ---------------
    A quadrotor constrained to move in a 2D vertical plane with two rotors
    providing thrust. The system has:
    - 3 degrees of freedom: (x, y) position and pitch angle θ
    - 2 control inputs: thrust forces from left and right rotors
    - Underactuated: 3 DOF controlled by 2 inputs
    - Nonlinear coupling between rotation and translation

    State Space:
    -----------
    State: x = [x, y, θ, ẋ, ẏ, θ̇]
        Position coordinates:
        - x: Horizontal position [m] (positive right)
        - y: Vertical position [m] (positive up)
        - θ (theta): Pitch angle [rad] (positive counterclockwise)
          * θ = 0: level orientation

        Velocity coordinates:
        - ẋ (x_dot): Horizontal velocity [m/s]
        - ẏ (y_dot): Vertical velocity [m/s]
        - θ̇ (theta_dot): Angular velocity [rad/s]

    Control: u = [u₁, u₂]
        - u₁: Left rotor thrust [N]
        - u₂: Right rotor thrust [N]
        Both must be non-negative in physical systems (thrust-only)

    Output: y = [x, y, θ]
        - Measures position and orientation

    Dynamics:
    --------
    The equations of motion are:
        ẍ = -(u₁ + u₂)/m · sin(θ)
        ÿ = (u₁ + u₂)/m · cos(θ) - g
        θ̈ = L/I · (u₁ - u₂)

    Physical interpretation:
    - Total thrust (u₁ + u₂) provides vertical lift and horizontal acceleration
    - Differential thrust (u₁ - u₂) creates torque for rotation
    - Gravity acts downward with acceleration g
    - Thrust direction rotates with pitch angle θ

    Parameters:
    ----------
    length : float, default=0.25
        Half-distance between rotors [m]. Larger L → more control authority
        for rotation (more torque from differential thrust).
    mass : float, default=0.486
        Total mass of quadrotor [kg]. Based on Crazyflie 2.0 specs.
    inertia : float, default=0.00383
        Moment of inertia about center of mass [kg⋅m²].
    gravity : float, default=9.81
        Gravitational acceleration [m/s²].

    Equilibrium:
    -----------
    Hovering equilibrium (level flight):
        x_eq = [x*, y*, 0, 0, 0, 0]  (any (x*, y*), level, stationary)
        u_eq = [mg/2, mg/2]  (each rotor supports half the weight)

    Default Physical Parameters:
    -----------------------------------
    - Mass: 0.027 kg (27 grams)
    - Length: 0.046 m (rotor arm)
    - Inertia: 0.00383 kg⋅m²
    - Gravity: 9.81 m/s²

    See Also:
    --------
    SymbolicQuadrotor2DState : Full-state observation variant
    PVTOL : Similar dynamics but different parameterization
    CartPole : Another underactuated 2D system
    """

    def __init__(
        self,
        length: float = 0.25,
        mass: float = 0.486,
        inertia: float = 0.00383,
        gravity: float = 9.81,
    ):
        super().__init__()
        self.order = 2
        # Store values for backward compatibility
        self.length_val = length
        self.mass_val = mass
        self.inertia_val = inertia
        self.gravity_val = gravity
        self.define_system(length, mass, inertia, gravity)

    def define_system(self, length_val, mass_val, inertia_val, gravity_val):
        x, y, theta, x_dot, y_dot, theta_dot = sp.symbols(
            "x y theta x_dot y_dot theta_dot", real=True
        )
        u1, u2 = sp.symbols("u1 u2", real=True)
        L, m, I, g = sp.symbols("L m I g", real=True, positive=True)

        self.parameters = {L: length_val, m: mass_val, I: inertia_val, g: gravity_val}
        self.state_vars = [x, y, theta, x_dot, y_dot, theta_dot]
        self.control_vars = [u1, u2]
        self.output_vars = [x, y, theta]

        # For second-order system, forward() returns acceleration
        dx_dot = (-1 / m) * sp.sin(theta) * (u1 + u2)
        dy_dot = (1 / m) * sp.cos(theta) * (u1 + u2) - g
        dtheta_dot = (L / I) * (u1 - u2)

        self._f_sym = sp.Matrix([dx_dot, dy_dot, dtheta_dot])
        self._h_sym = sp.Matrix([x, y, theta])

    @property
    def u_equilibrium(self) -> torch.Tensor:
        mg = self.mass_val * self.gravity_val
        return torch.tensor([mg / 2, mg / 2])

    @property
    def length(self):
        """For backward compatibility"""
        return self.length_val

    @property
    def mass(self):
        """For backward compatibility"""
        return self.mass_val

    @property
    def inertia(self):
        """For backward compatibility"""
        return self.inertia_val

    @property
    def gravity(self):
        """For backward compatibility"""
        return self.gravity_val


class SymbolicQuadrotor2DLidar(SymbolicDynamicalSystem):
    """
    Symbolic representation of a planar (2D) quadrotor with lidar-based partial observations.

    Models a quadrotor constrained to move in the y-z plane with dynamics derived from
    first principles. The system has 4 states (vertical position, pitch angle, and their
    derivatives) and 2 control inputs (thrust from each rotor). Unlike full-state feedback,
    this system uses a lidar sensor that measures distances to the ground at 4 different
    angles, providing partial observability that requires state estimation (e.g., Kalman
    filtering) for control. This implementation uses symbolic computation via SymPy to
    enable automatic Jacobian derivation for neural Lyapunov control synthesis.

    Based on the Stanford ASL neural-network-lyapunov quadrotor2d example:
    https://github.com/StanfordASL/neural-network-lyapunov/blob/master/neural_network_lyapunov/examples/quadrotor2d/quadrotor_2d.py

    State Vector (nx=4):
        - y: vertical position [m]
        - theta: pitch angle [rad]
        - y_dot: vertical velocity [m/s]
        - theta_dot: angular velocity [rad/s]

    Control Inputs (nu=2):
        - u1: thrust from rotor 1 [N]
        - u2: thrust from rotor 2 [N]

    Output Vector (ny=4):
        - Lidar ray distances at 4 different angles [m]
        - Measured from quadrotor to ground, ranging from [0, H]
        - Angles span from theta - angle_max to theta + angle_max

    Dynamics:
        The system is second-order, so forward() returns accelerations:
        - dy_dot = (1/m) * cos(theta) * (u1 + u2) - g - b * y_dot
        - dtheta_dot = (L/I) * (u1 - u2) - b * theta_dot

        where b is an optional damping coefficient (default: 0).

    Observation Model:
        Lidar rays measure distance to ground at different angles:
        - phi_i = theta - angle_offset_i
        - ray_i = (y + origin_height) / cos(phi_i)
        - Clamped to [0, H] and masked when out of valid range

    Equilibrium:
        - State: [0, 0, 0, 0] (hovering at origin)
        - Control: [mg/2, mg/2] (equal thrust counteracting gravity)
        - Output: Lidar readings at equilibrium depend on ray angles. Center rays
          measure approximately origin_height, while angled rays measure slightly
          longer distances (e.g., ~1.12m for rays at ±26.8° when origin_height=1.0)

    Parameters:
        length: Distance from center of mass to rotor [m]. Default: 0.25
        mass: Total quadrotor mass [kg]. Default: 0.486
        inertia: Moment of inertia about pitch axis [kg⋅m²]. Default: 0.00383
        gravity: Gravitational acceleration [m/s²]. Default: 9.81
        b: Damping coefficient for both translational and angular velocities. Default: 0.0
        H: Maximum lidar range [m]. Default: 5.0
        angle_max: Maximum angle offset for lidar rays [rad]. Default: 0.149π
        origin_height: Height offset added to vertical position [m]. Default: 1.0


    Note:
        This symbolic implementation is compatible with the hardcoded Quadrotor2DLidarDynamics
        class when using matching parameters. The observation function h(x) uses smooth
        approximations (via tanh and smooth_clamp) instead of hard thresholding to maintain
        differentiability for automatic Jacobian computation.
    """

    def __init__(
        self,
        length: float = 0.25,
        mass: float = 0.486,
        inertia: float = 0.00383,
        gravity: float = 9.81,
        b: float = 0.0,
        H: float = 5.0,
        angle_max: float = 0.149 * np.pi,
        origin_height: float = 1.0,
    ):
        super().__init__()
        self.order = 2
        # Store values for backward compatibility
        self.length_val = length
        self.mass_val = mass
        self.inertia_val = inertia
        self.gravity_val = gravity
        self.b_val = b
        self.H = H
        self.angle_max = angle_max
        self.origin_height = origin_height
        self.define_system(
            length, mass, inertia, gravity, b, H, angle_max, origin_height
        )

    def define_system(
        self,
        length_val,
        mass_val,
        inertia_val,
        gravity_val,
        b_val,
        H_val,
        angle_max_val,
        origin_height_val,
    ):
        y, theta, y_dot, theta_dot = sp.symbols("y theta y_dot theta_dot", real=True)
        u1, u2 = sp.symbols("u1 u2", real=True)
        L, m, I, g, b = sp.symbols("L m I g b", real=True, positive=True)

        self.parameters = {
            L: length_val,
            m: mass_val,
            I: inertia_val,
            g: gravity_val,
            b: b_val,
        }
        self.state_vars = [y, theta, y_dot, theta_dot]  # nx = 4
        self.control_vars = [u1, u2]

        # Dynamics (same as before)
        dy_dot = (1 / m) * sp.cos(theta) * (u1 + u2) - g - b * y_dot
        dtheta_dot = (L / I) * (u1 - u2) - b * theta_dot

        self._f_sym = sp.Matrix([dy_dot, dtheta_dot])

        # Lidar observation model
        # Create 4 lidar rays at different angles
        ny = 4
        lidar_rays = []

        for i in range(ny):
            # Linearly spaced angles from -angle_max to +angle_max
            angle_offset = -angle_max_val + i * (2 * angle_max_val / (ny - 1))
            phi = theta - angle_offset

            # Basic ray calculation: distance = (height) / cos(angle)
            ray_distance = (y + origin_height_val) / sp.cos(phi)

            # Smooth approximation of clamping to [0, H]
            # Use smooth functions to maintain differentiability
            # smooth_clamp(x, 0, H) ≈ max(0, min(x, H))

            # Soft ReLU for lower bound, soft minimum for upper bound
            # soft_relu(x) = ln(1 + exp(k*x))/k  (approaches ReLU as k→∞)
            # For symbolic computation, use: (x + sqrt(x^2 + eps))/2 which approximates ReLU

            eps = 1e-6  # Small constant for numerical stability

            # Soft ReLU: max(0, x) ≈ (x + sqrt(x^2 + eps))/2
            soft_relu = (ray_distance + sp.sqrt(ray_distance**2 + eps)) / 2

            # Soft min(x, H): H - soft_relu(H - x)
            clamped_ray = (
                H_val
                - (H_val - soft_relu + sp.sqrt((H_val - soft_relu) ** 2 + eps)) / 2
            )

            lidar_rays.append(clamped_ray)

        self._h_sym = sp.Matrix(lidar_rays)
        self.output_vars = [
            sp.Symbol(f"lidar_{i}", real=True) for i in range(ny)
        ]  # ny = 4

    @property
    def u_equilibrium(self) -> torch.Tensor:
        mg = self.mass_val * self.gravity_val
        return torch.tensor([mg / 2, mg / 2])

    @property
    def length(self):
        """For backward compatibility"""
        return self.length_val

    @property
    def mass(self):
        """For backward compatibility"""
        return self.mass_val

    @property
    def inertia(self):
        """For backward compatibility"""
        return self.inertia_val

    @property
    def gravity(self):
        """For backward compatibility"""
        return self.gravity_val

    @property
    def b(self):
        """For backward compatibility"""
        return self.b_val

class PVTOL(SymbolicDynamicalSystem):
    """
    Planar Vertical Take-Off and Landing (PVTOL) aircraft - second-order formulation.

    Physical System:
    ---------------
    A simplified model of a VTOL aircraft (helicopter, drone)
    constrained to move in a vertical plane.

    The aircraft has:
    - Two thrust actuators (left and right, or front and back)
    - Ability to rotate (pitch) and translate (x, y)
    - Gravity acting downward
    - Thrust-vectoring through body rotation

    **Key feature**: Underactuated with 2 inputs controlling 3 degrees of freedom.
    Must use rotation (θ) to control horizontal motion (x).

    Coordinate Frame:
    ----------------
    This implementation uses body-fixed velocity coordinates, which is common
    in aircraft dynamics:
    - Position (x, y): In inertial (world) frame
    - Velocities (ẋ, ẏ): In body frame (rotates with aircraft)
    - Angle θ: Pitch angle in inertial frame

    State Space:
    -----------
    State: x = [x, y, θ, ẋ, ẏ, θ̇]
        Position coordinates (inertial frame):
        - x: Horizontal position [m] (positive right)
        - y: Vertical position [m] (positive up)
        - θ (theta): Pitch angle [rad]
          * θ = 0: level (horizontal orientation)
          * θ > 0: nose up (pitched back)
          * θ < 0: nose down (pitched forward)

        Velocity coordinates (body frame):
        - ẋ (x_dot): Velocity in body x-direction [m/s]
        - ẏ (y_dot): Velocity in body y-direction [m/s]
        - θ̇ (theta_dot): Angular velocity [rad/s]

    Control: u = [u₁, u₂]
        - u₁: Left/front thrust [N]
        - u₂: Right/back thrust [N]
        Both must be non-negative in physical systems (thrust-only)

    Output: y = [x, y, θ]
        - Measures position and orientation

    Dynamics:
    --------
    The PVTOL dynamics in body frame are:

        ẍ_body = ẏ·θ̇ - g·sin(θ)
        ÿ_body = -ẋ·θ̇ - g·cos(θ) + (u₁ + u₂)/m
        θ̈ = d/I · (u₁ - u₂)

    **Horizontal acceleration (ẍ_body)**:
    - ẏ·θ̇: Centrifugal effect from rotation
    - -g·sin(θ): Gravity component in body x-direction
    - Controlled indirectly through angle θ

    **Vertical acceleration (ÿ_body)**:
    - -ẋ·θ̇: Coriolis effect from rotation
    - -g·cos(θ): Gravity component in body y-direction
    - (u₁ + u₂)/m: Total thrust divided by mass

    **Angular acceleration (θ̈)**:
    - d/I · (u₁ - u₂): Torque from differential thrust
    - d: Distance from center of mass to thrusters
    - I: Moment of inertia

    Parameters:
    ----------
    length : float, default=0.25
        Half-distance between thrusters [m]. Also interpreted as distance
        from center of mass to each thruster. Larger L → more control
        authority for rotation (more torque per thrust difference).
    mass : float, default=4.0
        Total mass of aircraft [kg]. Larger mass → slower acceleration
        response, more thrust needed to hover.
    inertia : float, default=0.0475
        Moment of inertia about center of mass [kg·m²]. Larger I →
        slower rotational response.
    gravity : float, default=9.8
        Gravitational acceleration [m/s²].
    dist : float, default=0.25
        Lever arm for torque generation [m]. Often equals length.
        Determines θ̈ = (dist/inertia)·(u₁ - u₂).

    Equilibria:
    ----------
    **Hovering (level flight)**:
        x_eq = [x*, y*, 0, 0, 0, 0]  (any position, level, stationary)
        u_eq = [mg/2, mg/2]  (equal thrust, each supporting half weight)

    At hover:
    - Total thrust balances gravity: u₁ + u₂ = mg
    - Differential thrust is zero: u₁ - u₂ = 0
    - No rotation: θ = 0

    **Tilted hover** (advanced):
        For x_eq = [x*, y*, θ*, 0, 0, 0] with θ* ≠ 0:
        Requires different thrust distribution to maintain position

    See Also:
    --------
    SymbolicQuadrotor2D : Similar flying vehicle, different parameterization
    CartPole : Another underactuated system
    Manipulator2Link : Multi-body system with coupling
    """

    def __init__(
        self,
        length: float = 0.25,
        mass: float = 4.0,
        inertia: float = 0.0475,
        gravity: float = 9.8,
        dist: float = 0.25,
    ):
        super().__init__()
        self.order = 2
        # Store values
        self.length_val = length
        self.mass_val = mass
        self.inertia_val = inertia
        self.gravity_val = gravity
        self.dist_val = dist
        self.define_system(length, mass, inertia, gravity, dist)

    def define_system(self, length_val, mass_val, inertia_val, gravity_val, dist_val):
        # State variables (position and velocity in body frame)
        x, y, theta, x_dot, y_dot, theta_dot = sp.symbols(
            "x y theta x_dot y_dot theta_dot", real=True
        )
        u1, u2 = sp.symbols("u1 u2", real=True)

        # Parameters
        L, m, I, g, d = sp.symbols("L m I g d", real=True, positive=True)

        self.parameters = {
            L: length_val,
            m: mass_val,
            I: inertia_val,
            g: gravity_val,
            d: dist_val,
        }

        self.state_vars = [x, y, theta, x_dot, y_dot, theta_dot]
        self.control_vars = [u1, u2]
        self.output_vars = [x, y, theta]

        # Rotation from body to world frame
        sin_theta = sp.sin(theta)
        cos_theta = sp.cos(theta)

        # The original code has velocities in a rotated frame
        # Position derivatives in world frame
        # x_change = x_dot * cos_theta - y_dot * sin_theta
        # y_change = x_dot * sin_theta + y_dot * cos_theta

        # Acceleration dynamics in body frame
        x_ddot = y_dot * theta_dot - g * sin_theta
        y_ddot = -x_dot * theta_dot - g * cos_theta + (u1 + u2) / m
        theta_ddot = (u1 - u2) * d / I

        # For second-order system, forward() returns accelerations
        self._f_sym = sp.Matrix([x_ddot, y_ddot, theta_ddot])
        self._h_sym = sp.Matrix([x, y, theta])

    @property
    def x_equilibrium(self) -> torch.Tensor:
        return torch.zeros(6)

    @property
    def u_equilibrium(self) -> torch.Tensor:
        return torch.full((2,), self.mass_val * self.gravity_val / 2)

    @property
    def length(self):
        return self.length_val

    @property
    def mass(self):
        return self.mass_val

    @property
    def inertia(self):
        return self.inertia_val

    @property
    def gravity(self):
        return self.gravity_val

    @property
    def dist(self):
        return self.dist_val