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


"""
Core Types - Fundamental Building Blocks

Defines the most basic types used throughout the framework:
- Multi-backend array types (NumPy, PyTorch, JAX)
- Semantic vector types (state, control, output, noise)
- Matrix types (dynamics, control, covariance, gains)
- System dimensions and equilibria
- Function signatures for dynamics and control
- Callback types

These are the foundation upon which all other type modules build.

Design Philosophy
----------------
- **Backend Agnostic**: Support NumPy, PyTorch, JAX equally
- **Semantic Clarity**: Names convey mathematical meaning
- **Composition Ready**: Types compose into higher-level structures
- **Type Safety**: Enable static type checking

Usage
-----
>>> from src.systems.base.types.core import (
...     StateVector,
...     ControlVector,
...     StateMatrix,
...     GainMatrix,
... )
>>> 
>>> def compute_control(x: StateVector, K: GainMatrix) -> ControlVector:
...     return -K @ x
"""

from typing import Union, Tuple, Dict, List, Callable, Optional, TypeVar, Any
import numpy as np

# Conditional imports for type checking
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import torch
    import jax.numpy as jnp
    import sympy as sp


# ============================================================================
# Basic Array Types - Multi-Backend Support
# ============================================================================

ArrayLike = Union[np.ndarray, 'torch.Tensor', 'jnp.ndarray']
"""
Array-like type supporting multiple backends.

Can be NumPy array, PyTorch tensor, or JAX array.
Use this for maximum flexibility across backends.

Shape conventions:
- Scalars: ()
- Vectors: (n,)
- Matrices: (m, n)
- Batched: (batch, ...)

Examples
--------
>>> def process(data: ArrayLike) -> ArrayLike:
...     # Works with all backends
...     return data * 2
"""

NumpyArray = np.ndarray
"""
Pure NumPy array.

Use when backend is definitely NumPy (no conversion needed).

Examples
--------
>>> def numpy_only(x: NumpyArray) -> NumpyArray:
...     return np.linalg.solve(A, x)
"""

# Conditional backend types (only if libraries available)
try:
    import torch
    TorchTensor = torch.Tensor
    """PyTorch tensor type."""
except ImportError:
    TorchTensor = type(None)

try:
    import jax.numpy as jnp
    JaxArray = jnp.ndarray
    """JAX array type."""
except ImportError:
    JaxArray = type(None)

ScalarLike = Union[float, int, np.number, 'torch.Tensor', 'jnp.ndarray']
"""
Scalar value in any backend.

Can be Python float/int, NumPy scalar, or 0-d tensor.

Examples
--------
>>> dt: ScalarLike = 0.01
>>> learning_rate: ScalarLike = 1e-3
"""

IntegerLike = Union[int, np.integer]
"""
Integer value (for dimensions, indices).

Examples
--------
>>> nx: IntegerLike = 3
>>> batch_size: IntegerLike = np.int64(100)
"""


# ============================================================================
# Vector Types - Semantic Naming by Role
# ============================================================================

StateVector = ArrayLike
"""
State vector x ∈ ℝⁿˣ.

The fundamental variable describing system configuration.

Shapes:
- Single state: (nx,)
- Batched states: (batch, nx)
- Trajectory: (n_steps, nx)
- Batched trajectory: (n_steps, batch, nx)

Examples
--------
>>> # Single state
>>> x: StateVector = np.array([1.0, 0.0, 0.5])
>>> 
>>> # Batched (100 states)
>>> x_batch: StateVector = np.random.randn(100, 3)
>>> 
>>> # Trajectory (101 time steps)
>>> x_traj: StateVector = np.random.randn(101, 3)
"""

ControlVector = ArrayLike
"""
Control input vector u ∈ ℝⁿᵘ.

Actuator commands that influence system dynamics.

Shapes:
- Single control: (nu,)
- Batched controls: (batch, nu)
- Control sequence: (n_steps, nu)
- Batched sequence: (n_steps, batch, nu)

Examples
--------
>>> # Single control
>>> u: ControlVector = np.array([0.5])
>>> 
>>> # Control sequence (100 steps)
>>> u_seq: ControlVector = np.zeros((100, 1))
>>> 
>>> # For autonomous systems (nu=0)
>>> u_autonomous: Optional[ControlVector] = None
"""

OutputVector = ArrayLike
"""
Output/observation/measurement vector y ∈ ℝⁿʸ.

Sensor measurements or system outputs.

Shapes:
- Single output: (ny,)
- Batched outputs: (batch, ny)
- Output sequence: (n_steps, ny)

Examples
--------
>>> # Full state observation
>>> y: OutputVector = np.array([1.0, 0.0, 0.5])  # y = x
>>> 
>>> # Partial observation (position only)
>>> y_partial: OutputVector = np.array([1.0])  # y = C*x where C = [1, 0, 0]
"""

NoiseVector = ArrayLike
"""
Noise/disturbance vector w ∈ ℝⁿʷ (stochastic systems only).

For continuous SDEs: Brownian motion increment dW
For discrete stochastic: Standard normal w[k] ~ N(0, I)

Shapes:
- Single sample: (nw,)
- Batched samples: (batch, nw)
- Noise sequence: (n_steps, nw)

Examples
--------
>>> # Standard normal noise
>>> w: NoiseVector = np.random.randn(2)
>>> 
>>> # Brownian increment (continuous)
>>> dW: NoiseVector = np.random.randn(2) * np.sqrt(dt)
>>> 
>>> # Discrete-time IID noise
>>> w_k: NoiseVector = np.random.randn(2)  # w[k] ~ N(0, I)
"""

ParameterVector = ArrayLike
"""
Parameter vector θ ∈ ℝⁿᵖ.

System parameters for identification, adaptation, or learning.

Examples
--------
>>> # Physical parameters
>>> theta: ParameterVector = np.array([mass, damping, stiffness])
>>> 
>>> # Neural network parameters (flattened)
>>> theta_nn: ParameterVector = flatten(model.parameters())
"""

ResidualVector = ArrayLike
"""
Residual/error vector (for optimization, estimation).

Difference between prediction and measurement.

Examples
--------
>>> # Measurement residual
>>> residual: ResidualVector = y_measured - y_predicted
>>> 
>>> # Tracking error
>>> error: ResidualVector = x_desired - x_current
"""


# ============================================================================
# Matrix Types - Semantic Naming by Role  
# ============================================================================

StateMatrix = ArrayLike
"""
State matrix (nx, nx).

Represents state-to-state coupling in linear(ized) systems.

Uses:
- Continuous: Ac (drift Jacobian ∂f/∂x)
- Discrete: Ad (state transition matrix)
- Covariance: P (state covariance E[(x-x̄)(x-x̄)'])
- Cost: Q (state cost weight in LQR)

Examples
--------
>>> # Continuous linearization
>>> Ac: StateMatrix = np.array([[0, 1], [-1, 0]])
>>> 
>>> # Discrete transition
>>> Ad: StateMatrix = np.eye(2) + dt*Ac
>>> 
>>> # Covariance
>>> P: StateMatrix = np.eye(2)
>>> 
>>> # LQR cost
>>> Q: StateMatrix = np.diag([10, 1])
"""

InputMatrix = ArrayLike
"""
Input matrix B (nx, nu).

Maps control/input vector to state derivatives in state-space models:
    Continuous: ẋ = Ax + Bu
    Discrete:   x[k+1] = Ax[k] + Bu[k]

Standard terminology in control theory literature. Also called:
- B matrix (state-space notation)
- Input gain matrix
- Control-to-state matrix

The B matrix structure reveals actuation architecture:
- Row i: how all inputs affect state i
- Column j: how input j affects all states
- Rank deficiency: underactuated system

Common structures:
- Full rank: all states directly actuated
- Partial: only some states directly affected
- Sparse: localized actuation (e.g., torque only on joints)

Examples
--------
>>> # Simple integrator: ẋ = u
>>> B: InputMatrix = np.array([[1.0]])
>>> 
>>> # Double integrator (position-velocity)
>>> # Only velocity is directly actuated
>>> B: InputMatrix = np.array([[0.0],   # position
...                            [1.0]])  # velocity
>>> 
>>> # Quadrotor (multi-input)
>>> # 4 motors, 6 states (x, y, z, roll, pitch, yaw)
>>> B: InputMatrix = np.zeros((6, 4))
>>> B[2, :] = [1, 1, 1, 1]      # z affected by all motors
>>> B[3, :] = [1, -1, -1, 1]    # roll differential
>>> B[4, :] = [1, 1, -1, -1]    # pitch differential
>>> B[5, :] = [1, -1, 1, -1]    # yaw differential
>>> 
>>> # Linearized from Jacobian
>>> # B = ∂f/∂u|_(x_eq, u_eq)
>>> def dynamics(x, u):
...     return np.array([x[1], -np.sin(x[0]) + u[0]])
>>> 
>>> # Jacobian at equilibrium
>>> B_lin: InputMatrix = np.array([[0.0],    # ∂f₁/∂u = 0
...                                [1.0]])   # ∂f₂/∂u = 1
"""

# Backward compatibility (deprecated in favor of InputMatrix)
ControlMatrix = InputMatrix
"""
Deprecated: Use InputMatrix instead.

This alias is maintained for backward compatibility with existing code.
New code should use InputMatrix to match standard control theory terminology.
"""

OutputMatrix = ArrayLike
"""
Output/observation matrix (ny, nx).

Maps state to output: y = C*x.

Uses:
- Observation: C (measurement model)
- Kalman filter: H (observation matrix)

Examples
--------
>>> # Full state observation
>>> C: OutputMatrix = np.eye(3)
>>> 
>>> # Partial observation (position only)
>>> C_partial: OutputMatrix = np.array([[1, 0, 0]])
>>> 
>>> # Multi-sensor observation
>>> C_multi: OutputMatrix = np.array([[1, 0, 0], [0, 1, 0]])
"""

DiffusionMatrix = ArrayLike
"""
Diffusion/noise gain matrix (nx, nw) - stochastic systems only.

Scales noise in stochastic dynamics.

Uses:
- Continuous: Gc (diffusion Jacobian ∂g/∂x, scales dW)
- Discrete: Gd (noise gain, scales w[k])
- Process noise: Q = G*W*G' where W is noise intensity

Examples
--------
>>> # Continuous SDE
>>> Gc: DiffusionMatrix = np.array([[0.1], [0.2]])
>>> 
>>> # Discrete (Euler-Maruyama)
>>> Gd: DiffusionMatrix = np.sqrt(dt) * Gc
>>> 
>>> # Additive noise (constant)
>>> G_additive: DiffusionMatrix = 0.1 * np.eye(nx)
"""

FeedthroughMatrix = ArrayLike
"""
Feedthrough/direct transmission matrix (ny, nu).

Direct control-to-output coupling: y = C*x + D*u.

Typically D = 0 (no direct feedthrough).

Examples
--------
>>> # No feedthrough (most common)
>>> D: FeedthroughMatrix = np.zeros((2, 1))
>>> 
>>> # Direct feedthrough (sensor measures control)
>>> D_direct: FeedthroughMatrix = np.array([[1.0]])
"""

CovarianceMatrix = ArrayLike
"""
Covariance matrix (symmetric, positive semidefinite).

Represents uncertainty or cost weighting.

Uses:
- State uncertainty: P (nx, nx)
- Measurement noise: R (ny, ny)
- Process noise: Q (nx, nx) or (nw, nw)
- Parameter uncertainty: Σ_θ (np, np)

Always symmetric: Σ = Σ'
Always PSD: v'Σv ≥ 0 for all v

Examples
--------
>>> # State covariance (isotropic)
>>> P: CovarianceMatrix = np.eye(3)
>>> 
>>> # Measurement noise (diagonal)
>>> R: CovarianceMatrix = np.diag([0.1, 0.05, 0.02])
>>> 
>>> # Process noise (correlated)
>>> Q: CovarianceMatrix = np.array([[0.1, 0.05], [0.05, 0.2]])
"""

GainMatrix = ArrayLike
"""
Gain matrix for control or estimation.

Maps state/output to control/estimate.

Uses:
- LQR gain: K (nu, nx) where u = -K*x
- Kalman gain: L (nx, ny) where x̂̇ = ... + L*(y - C*x̂)
- Observer gain: K_obs (nx, ny)

Examples
--------
>>> # LQR feedback gain
>>> K_lqr: GainMatrix = np.array([[1.0, 0.5]])  # (nu=1, nx=2)
>>> u = -K_lqr @ x
>>> 
>>> # Kalman gain
>>> K_kalman: GainMatrix = np.array([[0.1], [0.2]])  # (nx=2, ny=1)
>>> innovation = y - C @ x_hat
>>> x_hat_update = x_hat + K_kalman @ innovation
"""

ControllabilityMatrix = ArrayLike
"""
Controllability matrix (nx, nx*nu).

C = [B, AB, A²B, ..., A^(n-1)B]

System is controllable iff rank(C) = nx.

Examples
--------
>>> # Build controllability matrix
>>> C: ControllabilityMatrix = np.hstack([
...     B,
...     A @ B,
...     A @ A @ B,
...     # ... A^(n-1) @ B
... ])
>>> 
>>> # Check controllability
>>> rank = np.linalg.matrix_rank(C)
>>> is_controllable = (rank == nx)
"""

ObservabilityMatrix = ArrayLike
"""
Observability matrix (nx*ny, nx).

O = [C; CA; CA²; ...; CA^(n-1)]

System is observable iff rank(O) = nx.

Examples
--------
>>> # Build observability matrix
>>> O: ObservabilityMatrix = np.vstack([
...     C,
...     C @ A,
...     C @ A @ A,
...     # ... C @ A^(n-1)
... ])
>>> 
>>> # Check observability
>>> rank = np.linalg.matrix_rank(O)
>>> is_observable = (rank == nx)
"""

CostMatrix = ArrayLike
"""
Cost/weight matrix for optimal control.

Defines quadratic cost: J = x'Qx + u'Ru.

Types:
- Q: State cost (nx, nx) - penalizes state deviation
- R: Control cost (nu, nu) - penalizes control effort
- S: Cross cost (nx, nu) - couples state and control
- N: Terminal cost (nx, nx) - final state penalty

All must be symmetric and positive (semi)definite.

Examples
--------
>>> # State cost (emphasize position over velocity)
>>> Q: CostMatrix = np.diag([10, 1, 1])
>>> 
>>> # Control cost (penalize large inputs)
>>> R: CostMatrix = 0.1 * np.eye(nu)
>>> 
>>> # Terminal cost (large penalty at end)
>>> N: CostMatrix = 100 * Q
"""


# ============================================================================
# System Dimension Types
# ============================================================================

from typing_extensions import TypedDict

class SystemDimensions(TypedDict, total=False):
    """
    System dimensions as dictionary.
    
    All fields optional to support partial specifications.
    
    Attributes
    ----------
    nx : int
        State dimension
    nu : int
        Control dimension
    ny : int
        Output dimension
    nw : int
        Noise dimension (stochastic systems only)
    np : int
        Parameter dimension (for learning/adaptation)
    
    Examples
    --------
    >>> # Deterministic system
    >>> dims: SystemDimensions = {'nx': 3, 'nu': 2, 'ny': 3}
    >>> 
    >>> # Stochastic system
    >>> dims_stochastic: SystemDimensions = {
    ...     'nx': 2, 'nu': 1, 'ny': 2, 'nw': 2
    ... }
    >>> 
    >>> # With parameters
    >>> dims_adaptive: SystemDimensions = {
    ...     'nx': 3, 'nu': 1, 'ny': 3, 'np': 5
    ... }
    """
    nx: int
    nu: int
    ny: int
    nw: int
    np: int


DimensionTuple = Tuple[int, int, int]
"""
System dimensions as simple tuple: (nx, nu, ny).

Lightweight alternative to SystemDimensions dict.

Examples
--------
>>> dims: DimensionTuple = (3, 2, 3)
>>> nx, nu, ny = dims
"""


# ============================================================================
# Equilibrium Types
# ============================================================================

EquilibriumState = StateVector
"""
Equilibrium state point x_eq.

State where system has no net change:
- Continuous: dx/dt = f(x_eq, u_eq) = 0
- Discrete: x[k+1] = f(x_eq, u_eq) = x_eq

Examples
--------
>>> # Origin equilibrium
>>> x_eq: EquilibriumState = np.zeros(nx)
>>> 
>>> # Inverted pendulum equilibrium
>>> x_eq_up: EquilibriumState = np.array([np.pi, 0])
>>> 
>>> # Hovering quadrotor
>>> x_eq_hover: EquilibriumState = np.array([0, 0, 5, 0, 0, 0])  # (x, y, z, vx, vy, vz)
"""

EquilibriumControl = ControlVector
"""
Equilibrium control input u_eq.

Control that maintains equilibrium state.

For autonomous systems (nu=0), u_eq is typically None or empty array.

Examples
--------
>>> # Zero control at origin
>>> u_eq: EquilibriumControl = np.zeros(nu)
>>> 
>>> # Gravity compensation for hover
>>> u_eq_hover: EquilibriumControl = np.array([m * g])
>>> 
>>> # Autonomous system
>>> u_eq_autonomous: Optional[EquilibriumControl] = None
"""

EquilibriumPoint = Tuple[EquilibriumState, EquilibriumControl]
"""
Complete equilibrium specification: (x_eq, u_eq).

Bundles state and control equilibrium together.

Examples
--------
>>> # Origin with zero control
>>> equilibrium: EquilibriumPoint = (np.zeros(3), np.zeros(1))
>>> x_eq, u_eq = equilibrium
>>> 
>>> # Named unpacking
>>> origin: EquilibriumPoint = (np.zeros(nx), np.zeros(nu))
>>> x_eq_origin, u_eq_origin = origin
"""

EquilibriumName = str
"""
Named equilibrium identifier.

Human-readable string identifying an equilibrium.

Common names:
- 'origin': Zero state and control
- 'upright': Inverted pendulum up
- 'downward': Pendulum down
- 'hover': Aircraft hovering
- 'cruise': Vehicle cruising

Examples
--------
>>> name: EquilibriumName = 'origin'
>>> name: EquilibriumName = 'upright'
>>> name: EquilibriumName = 'hover'
>>> 
>>> # Use in API
>>> Ad, Bd = system.linearize('origin')  # By name
"""

EquilibriumIdentifier = Union[EquilibriumName, EquilibriumState]
"""
Equilibrium specified by name or state vector.

Enables flexible API: use name if available, state otherwise.

Examples
--------
>>> # By name (preferred if equilibrium is registered)
>>> identifier: EquilibriumIdentifier = 'origin'
>>> Ad, Bd = system.linearize('origin')
>>> 
>>> # By state (for unregistered points)
>>> identifier: EquilibriumIdentifier = np.zeros(nx)
>>> Ad, Bd = system.linearize(np.zeros(nx), np.zeros(nu))
"""


# ============================================================================
# Function Types - Dynamics and Control
# ============================================================================

DynamicsFunction = Callable[[StateVector, Optional[ControlVector]], StateVector]
"""
Dynamics function f(x, u).

Continuous: f(x, u, t) → dx/dt (rate of change)
Discrete: f(x, u) → x[k+1] (next state)

Parameters
----------
x : StateVector
    Current state (nx,)
u : Optional[ControlVector]
    Control input (nu,) or None for autonomous

Returns
-------
StateVector
    Dynamics evaluation (nx,)

Examples
--------
>>> # Linear dynamics
>>> def f_linear(x: StateVector, u: ControlVector) -> StateVector:
...     return A @ x + B @ u
>>> 
>>> # Nonlinear dynamics
>>> def f_nonlinear(x: StateVector, u: ControlVector) -> StateVector:
...     x1, x2 = x
...     return np.array([x2, -np.sin(x1) + u[0]])
>>> 
>>> # Autonomous
>>> def f_autonomous(x: StateVector, u: Optional[ControlVector] = None) -> StateVector:
...     return -alpha * x
"""

OutputFunction = Callable[[StateVector], OutputVector]
"""
Output/observation function h(x).

Maps state to measurement: y = h(x).

Parameters
----------
x : StateVector
    State (nx,)

Returns
-------
OutputVector
    Output (ny,)

Examples
--------
>>> # Full state observation
>>> def h_full(x: StateVector) -> OutputVector:
...     return x  # y = x
>>> 
>>> # Partial observation (position only)
>>> def h_partial(x: StateVector) -> OutputVector:
...     return x[0:2]  # First two states
>>> 
>>> # Nonlinear observation
>>> def h_nonlinear(x: StateVector) -> OutputVector:
...     return np.array([np.linalg.norm(x)])  # Distance from origin
"""

DiffusionFunction = Callable[[StateVector, Optional[ControlVector]], DiffusionMatrix]
"""
Diffusion function g(x, u) for stochastic systems.

Returns noise gain matrix.

Parameters
----------
x : StateVector
    State (nx,)
u : Optional[ControlVector]
    Control (nu,) or None

Returns
-------
DiffusionMatrix
    Diffusion matrix (nx, nw)

Examples
--------
>>> # Additive noise (constant)
>>> def g_additive(x: StateVector, u: ControlVector) -> DiffusionMatrix:
...     return 0.1 * np.eye(nx)
>>> 
>>> # Multiplicative noise (state-dependent)
>>> def g_multiplicative(x: StateVector, u: ControlVector) -> DiffusionMatrix:
...     return np.diag(0.1 * np.abs(x))
>>> 
>>> # Geometric Brownian motion
>>> def g_gbm(x: StateVector, u: ControlVector) -> DiffusionMatrix:
...     return np.array([[sigma * x[0]]])
"""

ControlPolicy = Callable[[StateVector], ControlVector]
"""
Control policy/controller π(x).

Maps state to control action: u = π(x).

Parameters
----------
x : StateVector
    Current state (nx,)

Returns
-------
ControlVector
    Control action (nu,)

Examples
--------
>>> # LQR policy
>>> def lqr_policy(x: StateVector) -> ControlVector:
...     return -K @ x
>>> 
>>> # Nonlinear policy
>>> def nonlinear_policy(x: StateVector) -> ControlVector:
...     return -np.tanh(K @ x)
>>> 
>>> # Neural network policy
>>> def nn_policy(x: StateVector) -> ControlVector:
...     return model.forward(torch.tensor(x)).numpy()
"""

StateEstimator = Callable[[OutputVector], StateVector]
"""
State estimator/observer L(y).

Estimates state from measurements: x̂ = L(y).

Parameters
----------
y : OutputVector
    Measurement (ny,)

Returns
-------
StateVector
    State estimate (nx,)

Examples
--------
>>> # Simple observer (assumes y = C*x)
>>> def observer(y: OutputVector) -> StateVector:
...     return np.linalg.pinv(C) @ y
>>> 
>>> # Kalman filter estimate
>>> def kalman_estimator(y: OutputVector) -> StateVector:
...     # Internal state update logic
...     return x_hat
"""

CostFunction = Callable[[StateVector, ControlVector], float]
"""
Cost/objective function J(x, u).

Scalar cost for optimization.

Parameters
----------
x : StateVector
    State (nx,)
u : ControlVector
    Control (nu,)

Returns
-------
float
    Cost value

Examples
--------
>>> # Quadratic cost
>>> def quadratic_cost(x: StateVector, u: ControlVector) -> float:
...     return x.T @ Q @ x + u.T @ R @ u
>>> 
>>> # Nonlinear cost
>>> def nonlinear_cost(x: StateVector, u: ControlVector) -> float:
...     tracking_error = np.linalg.norm(x - x_desired)
...     control_effort = np.linalg.norm(u)
...     return tracking_error**2 + 0.1 * control_effort**2
"""

Constraint = Callable[[StateVector, ControlVector], ArrayLike]
"""
Constraint function c(x, u).

Returns constraint values that should satisfy c ≤ 0.

Parameters
----------
x : StateVector
    State (nx,)
u : ControlVector
    Control (nu,)

Returns
-------
ArrayLike
    Constraint values (n_constraints,)
    Should satisfy c ≤ 0 for feasibility

Examples
--------
>>> # Input constraints: |u| ≤ u_max
>>> def input_constraint(x: StateVector, u: ControlVector) -> ArrayLike:
...     return np.abs(u) - u_max  # c ≤ 0
>>> 
>>> # State constraints: x in box
>>> def state_box(x: StateVector, u: ControlVector) -> ArrayLike:
...     return np.concatenate([
...         x - x_max,   # x ≤ x_max
...         -x + x_min,  # x ≥ x_min
...     ])
>>> 
>>> # Nonlinear constraint: avoid obstacle
>>> def obstacle_avoidance(x: StateVector, u: ControlVector) -> ArrayLike:
...     distance = np.linalg.norm(x - x_obstacle)
...     return r_safe - distance  # Stay away by r_safe
"""


# ============================================================================
# Callback Types
# ============================================================================

IntegrationCallback = Callable[[float, StateVector], Optional[bool]]
"""
Callback during continuous integration.

Called at each integration step.

Parameters
----------
t : float
    Current time
x : StateVector
    Current state

Returns
-------
Optional[bool]
    True to stop integration, False/None to continue

Examples
--------
>>> def stop_on_divergence(t: float, x: StateVector) -> bool:
...     if np.linalg.norm(x) > 100:
...         print(f"Stopping at t={t:.2f}: state diverged")
...         return True  # Stop
...     return False  # Continue
>>> 
>>> # Event detection
>>> def detect_event(t: float, x: StateVector) -> bool:
...     return x[0] < 0  # Stop when first state goes negative
"""

SimulationCallback = Callable[[int, StateVector, ControlVector], None]
"""
Callback during discrete simulation.

Called at each time step.

Parameters
----------
k : int
    Current time step
x : StateVector
    State at time k
u : ControlVector
    Control at time k

Returns
-------
None

Examples
--------
>>> def logger(k: int, x: StateVector, u: ControlVector):
...     if k % 10 == 0:
...         print(f"Step {k}: ||x||={np.linalg.norm(x):.3f}")
>>> 
>>> # Data collection
>>> data = {'states': [], 'controls': []}
>>> def collect_data(k: int, x: StateVector, u: ControlVector):
...     data['states'].append(x.copy())
...     data['controls'].append(u.copy())
"""

OptimizationCallback = Callable[[ArrayLike], None]
"""
Callback during optimization.

Called at each optimization iteration.

Parameters
----------
x : ArrayLike
    Current optimization variable

Returns
-------
None

Examples
--------
>>> def monitor_convergence(x: ArrayLike):
...     cost = cost_function(x)
...     print(f"Current cost: {cost:.6f}")
>>> 
>>> # Early stopping
>>> best_cost = [float('inf')]
>>> def early_stop(x: ArrayLike):
...     cost = cost_function(x)
...     if cost < best_cost[0]:
...         best_cost[0] = cost
...     elif cost > 1.1 * best_cost[0]:
...         raise StopIteration("Cost increasing - stop")
"""


# ============================================================================
# Generic Type Variables
# ============================================================================

T = TypeVar('T', bound=ArrayLike)
"""
Generic array type variable.

Use for functions that preserve array type.

Examples
--------
>>> def scale(x: T, factor: float) -> T:
...     '''Returns same type as input.'''
...     return x * factor
"""

S = TypeVar('S', bound='DiscreteSystemBase')
"""
Generic discrete system type variable.

Use for functions operating on discrete systems.

Examples
--------
>>> def simulate_system(system: S, x0: StateVector, steps: int) -> StateTrajectory:
...     '''Works with any discrete system.'''
...     return system.simulate(x0, steps=steps)
"""

C = TypeVar('C', bound='ContinuousSystemBase')
"""
Generic continuous system type variable.

Use for functions operating on continuous systems.

Examples
--------
>>> def integrate_system(system: C, x0: StateVector, t_span: TimeSpan) -> StateTrajectory:
...     '''Works with any continuous system.'''
...     return system.integrate(x0, t_span=t_span)
"""

MatrixT = TypeVar('MatrixT', StateMatrix, InputMatrix, DiffusionMatrix)
"""
Generic matrix type variable.

Use for functions that work with any matrix type.

Examples
--------
>>> def check_symmetry(M: MatrixT) -> bool:
...     M_np = ensure_numpy(M)
...     return np.allclose(M_np, M_np.T)
"""


# ============================================================================
# Export All
# ============================================================================

__all__ = [
    # Basic arrays
    'ArrayLike',
    'NumpyArray',
    'TorchTensor',
    'JaxArray',
    'ScalarLike',
    'IntegerLike',
    
    # Vectors
    'StateVector',
    'ControlVector',
    'OutputVector',
    'NoiseVector',
    'ParameterVector',
    'ResidualVector',
    
    # Matrices
    'StateMatrix',
    'InputMatrix',
    'OutputMatrix',
    'DiffusionMatrix',
    'FeedthroughMatrix',
    'CovarianceMatrix',
    'GainMatrix',
    'ControllabilityMatrix',
    'ObservabilityMatrix',
    'CostMatrix',
    
    # Dimensions
    'SystemDimensions',
    'DimensionTuple',
    
    # Equilibria
    'EquilibriumState',
    'EquilibriumControl',
    'EquilibriumPoint',
    'EquilibriumName',
    'EquilibriumIdentifier',
    
    # Functions
    'DynamicsFunction',
    'OutputFunction',
    'DiffusionFunction',
    'ControlPolicy',
    'StateEstimator',
    'CostFunction',
    'Constraint',
    
    # Callbacks
    'IntegrationCallback',
    'SimulationCallback',
    'OptimizationCallback',
    
    # Type variables
    'T',
    'S',
    'C',
    'MatrixT',
]