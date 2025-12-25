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
Comprehensive Type Definitions for ControlDESymulation Framework

Centralized type aliases, protocols, and type utilities for the entire framework.
Provides consistent typing across all modules for:
- Arrays and tensors (multi-backend)
- Vectors (state, control, output, noise)
- Matrices (dynamics, linearization, covariance)
- System dimensions and properties
- Linearization results
- Trajectories and sequences
- Symbolic expressions
- Integration and discretization
- Stability and controllability
- Control design (LQR, LQG, MPC, H∞, etc.)
- State estimation (Kalman, EKF, UKF, Particle Filter, MHE)
- System identification (ERA, DMD, SINDy, N4SID, Koopman)
- Safety and reachability (CBF, HJI, Barriers)
- Contraction analysis (CCM, Funneling)
- Conformal prediction (distribution-free inference)
- Robustness (μ-analysis, Tube MPC)
- Learning (Neural dynamics, GP)

Design Principles
----------------
1. **Single Source of Truth**: All type definitions in one place
2. **Semantic Naming**: Types convey meaning (StateVector vs ArrayLike)
3. **Backend Agnostic**: Support NumPy, PyTorch, JAX equally
4. **Extensible**: Easy to add new backends or types
5. **Well Documented**: Every type has clear docstring and examples

Usage Pattern
------------
>>> from src.systems.base.types import (
...     StateVector,
...     ControlVector,
...     LinearizationResult,
...     Backend,
... )
>>> 
>>> def my_controller(
...     x: StateVector,
...     u: ControlVector,
...     backend: Backend = 'numpy'
... ) -> StateVector:
...     '''Type checker knows these are array-like.'''
...     return x + u

Migration Guide
--------------
Replace local type aliases with centralized imports:

Before:
    from typing import Union
    import numpy as np
    ArrayLike = Union[np.ndarray, 'torch.Tensor', 'jnp.ndarray']

After:
    from src.systems.base.types import ArrayLike, StateVector

This reduces duplication and ensures consistency.

Statistics
----------
- 165+ type definitions
- 20+ TypedDict result types
- 15+ utility functions
- 8+ Protocol definitions
- Covers all major control/estimation/identification algorithms
"""

from typing import (
    Union,
    Tuple,
    Optional,
    Dict,
    List,
    Callable,
    TypeVar,
    Protocol,
    Literal,
    Any,
)
from typing_extensions import TypedDict
import numpy as np

# Conditional imports for type checking only
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import torch
    import jax.numpy as jnp
    import sympy as sp
    from src.systems.base.discrete_system_base import DiscreteSystemBase
    from src.systems.base.continuous_system_base import ContinuousSystemBase


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
"""Pure NumPy array (when backend is definitely NumPy)."""

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
"""Scalar value in any backend (0-dimensional)."""

IntegerLike = Union[int, np.integer]
"""Integer value (for dimensions, indices)."""


# ============================================================================
# Vector Types - Semantic Naming by Role
# ============================================================================

StateVector = ArrayLike
"""
State vector x ∈ ℝⁿˣ.

Shapes:
- Single: (nx,)
- Batched: (batch, nx)
- Trajectory: (n_steps, nx)
- Batched trajectory: (n_steps, batch, nx)

Examples
--------
>>> x: StateVector = np.array([1.0, 0.0, 0.5])  # Single state
>>> x_batch: StateVector = np.random.randn(100, 3)  # Batched
"""

ControlVector = ArrayLike
"""
Control input vector u ∈ ℝⁿᵘ.

Shapes:
- Single: (nu,)
- Batched: (batch, nu)
- Sequence: (n_steps, nu)
- Batched sequence: (n_steps, batch, nu)

Examples
--------
>>> u: ControlVector = np.array([0.5])  # Single control
>>> u_seq: ControlVector = np.zeros((100, 1))  # Control sequence
"""

OutputVector = ArrayLike
"""
Output/observation vector y ∈ ℝⁿʸ.

Shapes:
- Single: (ny,)
- Batched: (batch, ny)
- Sequence: (n_steps, ny)

Examples
--------
>>> y: OutputVector = np.array([1.0, 0.0])  # Measurement
"""

NoiseVector = ArrayLike
"""
Noise/disturbance vector w ∈ ℝⁿʷ (stochastic systems only).

For continuous SDEs: Brownian motion increment dW
For discrete stochastic: Standard normal w[k] ~ N(0, I)

Shapes:
- Single: (nw,)
- Batched: (batch, nw)
- Sequence: (n_steps, nw)

Examples
--------
>>> w: NoiseVector = np.random.randn(2)  # Standard normal
>>> dW: NoiseVector = np.random.randn(2) * np.sqrt(dt)  # Brownian increment
"""

ParameterVector = ArrayLike
"""
Parameter vector θ ∈ ℝⁿᵖ (for parameter estimation/learning).

Examples
--------
>>> theta: ParameterVector = np.array([1.0, 0.5, 2.0])
"""

ResidualVector = ArrayLike
"""
Residual/error vector (for optimization, estimation).

Examples
--------
>>> residual: ResidualVector = y_measured - y_predicted
"""


# ============================================================================
# Matrix Types - Semantic Naming by Role  
# ============================================================================

StateMatrix = ArrayLike
"""
State matrix (nx, nx).

Examples:
- Continuous: Ac (drift Jacobian)
- Discrete: Ad (state transition)
- Covariance: P (state covariance)
- Cost: Q (state cost matrix)

Examples
--------
>>> Ac: StateMatrix = np.array([[0, 1], [-1, 0]])  # Continuous
>>> Ad: StateMatrix = np.eye(2) + dt*Ac  # Discrete
>>> P: StateMatrix = np.eye(2)  # Covariance
"""

ControlMatrix = ArrayLike
"""
Control matrix (nx, nu).

Examples:
- Continuous: Bc (control Jacobian)
- Discrete: Bd (control gain)
- Cost: R (control cost matrix, nu×nu)

Examples
--------
>>> Bc: ControlMatrix = np.array([[0], [1]])  # Continuous
>>> Bd: ControlMatrix = dt * Bc  # Discrete
"""

OutputMatrix = ArrayLike
"""
Output/observation matrix (ny, nx).

Examples:
- Observation: C (y = C*x)
- Measurement: H (in Kalman filter notation)

Examples
--------
>>> C: OutputMatrix = np.eye(3)  # Full state observation
>>> C_partial: OutputMatrix = np.array([[1, 0, 0]])  # Partial observation
"""

DiffusionMatrix = ArrayLike
"""
Diffusion/noise gain matrix (nx, nw) - stochastic systems only.

Examples:
- Continuous: Gc (diffusion Jacobian, scales dW)
- Discrete: Gd (noise gain, scales w[k])

Examples
--------
>>> Gc: DiffusionMatrix = np.array([[0.1], [0.2]])  # Continuous
>>> Gd: DiffusionMatrix = np.sqrt(dt) * Gc  # Discrete (Euler-Maruyama)
"""

FeedthroughMatrix = ArrayLike
"""
Feedthrough/direct transmission matrix (ny, nu).

For output: y = C*x + D*u

Examples
--------
>>> D: FeedthroughMatrix = np.zeros((2, 1))  # No feedthrough
"""

CovarianceMatrix = ArrayLike
"""
Covariance matrix (symmetric, positive semidefinite).

Examples:
- State: P (nx, nx)
- Measurement noise: R (ny, ny)
- Process noise: Q (nx, nx) or (nw, nw)

Examples
--------
>>> P: CovarianceMatrix = np.eye(3)  # State covariance
>>> Q: CovarianceMatrix = 0.01 * np.eye(2)  # Process noise
"""

GainMatrix = ArrayLike
"""
Gain matrix for control or estimation.

Examples:
- LQR gain: K (nu, nx)
- Kalman gain: L (nx, ny)
- Observer gain: K_obs (nx, ny)

Examples
--------
>>> K_lqr: GainMatrix = np.array([[1.0, 0.5]])  # (nu, nx)
>>> K_kalman: GainMatrix = np.array([[0.1], [0.2]])  # (nx, ny)
"""

ControllabilityMatrix = ArrayLike
"""Controllability matrix C = [B, AB, A²B, ...] (nx, nx*nu)."""

ObservabilityMatrix = ArrayLike
"""Observability matrix O = [C; CA; CA²; ...] (nx*ny, nx)."""

CostMatrix = ArrayLike
"""
Cost matrix for optimal control.

Examples:
- Q: State cost (nx, nx)
- R: Control cost (nu, nu)
- S: Cross cost (nx, nu)
- N: Terminal cost (nx, nx)

Examples
--------
>>> Q: CostMatrix = np.diag([10, 1, 1])
>>> R: CostMatrix = 0.1 * np.eye(nu)
"""


# ============================================================================
# Linearization Result Types
# ============================================================================

DeterministicLinearization = Tuple[StateMatrix, ControlMatrix]
"""
Linearization for deterministic systems: (A, B).

Continuous: (Ac, Bc) where dx/dt ≈ Ac*δx + Bc*δu
Discrete: (Ad, Bd) where δx[k+1] ≈ Ad*δx[k] + Bd*δu[k]

Examples
--------
>>> Ac, Bc = continuous_system.linearize(x_eq, u_eq)
>>> Ad, Bd = discrete_system.linearize(x_eq, u_eq)
"""

StochasticLinearization = Tuple[StateMatrix, ControlMatrix, DiffusionMatrix]
"""
Linearization for stochastic systems: (A, B, G).

Continuous: (Ac, Bc, Gc) where dx ≈ (Ac*δx + Bc*δu)dt + Gc*dW
Discrete: (Ad, Bd, Gd) where δx[k+1] ≈ Ad*δx[k] + Bd*δu[k] + Gd*w[k]

Examples
--------
>>> Ac, Bc, Gc = sde_system.linearize(x_eq, u_eq)
>>> Ad, Bd, Gd = discrete_stochastic.linearize(x_eq, u_eq)
"""

LinearizationResult = Union[DeterministicLinearization, StochasticLinearization]
"""
Flexible linearization result type.

Can be (A, B) or (A, B, G) depending on system type.
Enables polymorphic code that handles both.

Examples
--------
>>> result = system.linearize(x_eq, u_eq)
>>> A, B = result[0], result[1]
>>> if len(result) == 3:
...     G = result[2]  # Stochastic
"""

ObservationLinearization = Tuple[OutputMatrix, FeedthroughMatrix]
"""
Output linearization: (C, D) where y ≈ C*δx + D*δu.

Examples
--------
>>> C, D = system.linearized_observation(x_eq, u_eq)
"""


# ============================================================================
# Trajectory and Sequence Types
# ============================================================================

StateTrajectory = ArrayLike
"""
State trajectory over time.

Shapes:
- Single trajectory: (n_steps, nx)
- Batched trajectories: (n_steps, batch, nx)
- Multiple trials: (n_trials, n_steps, nx)

Examples
--------
>>> trajectory: StateTrajectory = system.simulate(x0, u_seq, steps=100)
>>> print(trajectory.shape)  # (101, nx) - includes t=0
"""

ControlSequence = ArrayLike
"""
Control input sequence over time.

Shapes:
- Single sequence: (n_steps, nu)
- Batched sequences: (n_steps, batch, nu)

Examples
--------
>>> u_seq: ControlSequence = np.zeros((100, nu))
>>> u_seq: ControlSequence = controller.plan(x0, horizon=100)
"""

OutputSequence = ArrayLike
"""
Output/measurement sequence over time.

Shapes:
- Single sequence: (n_steps, ny)
- Batched: (n_steps, batch, ny)

Examples
--------
>>> y_seq: OutputSequence = sensor.measure(state_trajectory)
"""

NoiseSequence = ArrayLike
"""
Noise sequence for stochastic simulation.

Shapes:
- Single: (n_steps, nw)
- Batched: (n_steps, batch, nw)

Examples
--------
>>> w_seq: NoiseSequence = np.random.randn(100, nw)
>>> dW_seq: NoiseSequence = np.random.randn(100, nw) * np.sqrt(dt)
"""

TimePoints = ArrayLike
"""
Array of time points for simulation.

Shape: (n_points,)

Examples
--------
>>> t_span: TimePoints = np.linspace(0, 10, 1000)
>>> t_eval: TimePoints = np.arange(0, 10, 0.01)
"""

TimeSpan = Tuple[float, float]
"""
Time interval (t_start, t_end) for continuous integration.

Examples
--------
>>> t_span: TimeSpan = (0.0, 10.0)
"""


# ============================================================================
# Symbolic Types (SymPy)
# ============================================================================

if TYPE_CHECKING:
    SymbolicExpression = 'sp.Expr'
    SymbolicMatrix = 'sp.Matrix'
    SymbolicSymbol = 'sp.Symbol'
else:
    SymbolicExpression = Any
    SymbolicMatrix = Any
    SymbolicSymbol = Any

SymbolDict = Dict[SymbolicSymbol, float]
"""
Mapping from symbolic parameters to numerical values.

Examples
--------
>>> params: SymbolDict = {alpha: 1.0, beta: 0.5, gamma: 2.0}
"""

SymbolicStateEquations = SymbolicMatrix
"""Symbolic state equations: f(x, u, t) as SymPy matrix."""

SymbolicOutputEquations = SymbolicMatrix
"""Symbolic output equations: h(x, u, t) as SymPy matrix."""

SymbolicDiffusionMatrix = SymbolicMatrix
"""Symbolic diffusion matrix: g(x, u, t) for stochastic systems."""


# ============================================================================
# System Dimension Types
# ============================================================================

class SystemDimensions(TypedDict, total=False):
    """
    System dimensions as dictionary.
    
    All fields optional to support partial specifications.
    
    Examples
    --------
    >>> dims: SystemDimensions = {'nx': 3, 'nu': 2, 'ny': 3}
    >>> dims_stochastic: SystemDimensions = {'nx': 2, 'nu': 1, 'nw': 2}
    """
    nx: int  # State dimension
    nu: int  # Control dimension
    ny: int  # Output dimension
    nw: int  # Noise dimension (stochastic only)
    np: int  # Parameter dimension (for learning)


DimensionTuple = Tuple[int, int, int]
"""
System dimensions as tuple: (nx, nu, ny).

Examples
--------
>>> dims: DimensionTuple = (3, 2, 3)
"""


# ============================================================================
# Equilibrium Types
# ============================================================================

EquilibriumState = StateVector
"""
Equilibrium state point x_eq.

State where dx/dt = 0 (continuous) or x[k+1] = x[k] (discrete).

Examples
--------
>>> x_eq: EquilibriumState = np.zeros(nx)
>>> x_eq: EquilibriumState = np.array([np.pi, 0])  # Inverted pendulum
"""

EquilibriumControl = ControlVector
"""
Equilibrium control input u_eq.

Control that maintains equilibrium state.

Examples
--------
>>> u_eq: EquilibriumControl = np.zeros(nu)
>>> u_eq: EquilibriumControl = np.array([9.81 * m])  # Hover control
"""

EquilibriumPoint = Tuple[EquilibriumState, EquilibriumControl]
"""
Complete equilibrium specification: (x_eq, u_eq).

Examples
--------
>>> equilibrium: EquilibriumPoint = (np.zeros(3), np.zeros(1))
>>> x_eq, u_eq = equilibrium
"""

EquilibriumName = str
"""
Named equilibrium identifier.

Examples
--------
>>> name: EquilibriumName = 'origin'
>>> name: EquilibriumName = 'upright'
>>> name: EquilibriumName = 'hover'
"""

EquilibriumIdentifier = Union[EquilibriumName, EquilibriumState]
"""
Equilibrium specified by name or state vector.

Enables flexible API: use name if available, state otherwise.

Examples
--------
>>> # By name
>>> Ad, Bd = system.linearize('origin')
>>> 
>>> # By state
>>> Ad, Bd = system.linearize(np.zeros(nx), np.zeros(nu))
"""


# ============================================================================
# Backend and Device Types
# ============================================================================

Backend = Literal['numpy', 'torch', 'jax']
"""
Backend identifier.

Valid values: 'numpy', 'torch', 'jax'

Examples
--------
>>> backend: Backend = 'torch'
>>> system.set_default_backend('jax')
"""

Device = str
"""
Device identifier for PyTorch/JAX.

Examples:
- 'cpu'
- 'cuda'
- 'cuda:0', 'cuda:1', ...
- 'mps' (Apple Silicon)

Examples
--------
>>> device: Device = 'cuda:0'
>>> system.set_preferred_device('cpu')
"""

class BackendConfig(TypedDict, total=False):
    """
    Backend configuration dictionary.
    
    Examples
    --------
    >>> config: BackendConfig = {
    ...     'backend': 'torch',
    ...     'device': 'cuda:0',
    ...     'dtype': 'float32'
    ... }
    """
    backend: Backend
    device: Optional[Device]
    dtype: Optional[str]


# ============================================================================
# Method and Algorithm Types
# ============================================================================

IntegrationMethod = str
"""
Integration method for continuous systems.

Common values:
- 'RK45', 'RK23', 'DOP853' (adaptive Runge-Kutta)
- 'Radau', 'BDF', 'LSODA' (stiff solvers)
- 'euler', 'rk4', 'rk2' (fixed-step)

Examples
--------
>>> method: IntegrationMethod = 'RK45'
>>> integrator = Integrator(system, method='DOP853')
"""

DiscretizationMethod = str
"""
Discretization method for continuous → discrete.

Common values:
- 'euler': Forward Euler
- 'exact': Matrix exponential
- 'tustin': Bilinear/Tustin
- 'matched': Pole-zero matched

Examples
--------
>>> method: DiscretizationMethod = 'exact'
>>> discretizer = Discretizer(system, dt=0.01, method='tustin')
"""

SDEIntegrationMethod = str
"""
SDE integration method for stochastic systems.

Common values:
- 'euler', 'EM', 'euler-maruyama': Euler-Maruyama
- 'milstein', 'ItoMilstein': Milstein scheme
- 'srk', 'SRIW1': Stochastic Runge-Kutta
- 'SEA', 'SHARK', 'SRA1': Additive noise optimized

Examples
--------
>>> method: SDEIntegrationMethod = 'milstein'
>>> sde_integrator = SDEIntegrator(sde_system, method='SRIW1')
"""

OptimizationMethod = str
"""
Optimization method for control/estimation.

Examples: 'SLSQP', 'L-BFGS-B', 'trust-constr', 'Newton-CG'

Examples
--------
>>> method: OptimizationMethod = 'SLSQP'
"""


# ============================================================================
# Stability Analysis Types
# ============================================================================

class StabilityInfo(TypedDict):
    """
    Stability analysis result dictionary.
    
    Contains eigenvalue-based stability information.
    
    Examples
    --------
    >>> stability: StabilityInfo = system.check_stability(x_eq, u_eq)
    >>> if stability['is_stable']:
    ...     print(f"Stable with ρ={stability['spectral_radius']:.3f}")
    """
    eigenvalues: np.ndarray              # Eigenvalues of linearization
    magnitudes: np.ndarray               # Absolute values |λ|
    max_magnitude: float                 # max|λ|
    spectral_radius: float               # Same as max_magnitude
    is_stable: bool                      # All |λ| < 1 (discrete) or Re(λ) < 0 (continuous)
    is_marginally_stable: bool           # max|λ| ≈ 1 or Re(λ) ≈ 0
    is_unstable: bool                    # Any |λ| > 1 or Re(λ) > 0


class ControllabilityInfo(TypedDict, total=False):
    """
    Controllability analysis result.
    
    Examples
    --------
    >>> info: ControllabilityInfo = analyze_controllability(system, x_eq, u_eq)
    >>> if info['is_controllable']:
    ...     print(f"Controllable with rank {info['rank']}")
    """
    controllability_matrix: ControllabilityMatrix  # C = [B, AB, A²B, ...]
    rank: int                            # Rank of C
    is_controllable: bool                # rank == nx
    uncontrollable_modes: Optional[np.ndarray]  # Eigenvalues of uncontrollable modes


class ObservabilityInfo(TypedDict, total=False):
    """
    Observability analysis result.
    
    Examples
    --------
    >>> info: ObservabilityInfo = analyze_observability(system, x_eq)
    >>> if info['is_observable']:
    ...     print("System is observable")
    """
    observability_matrix: ObservabilityMatrix  # O = [C; CA; CA²; ...]
    rank: int                            # Rank of O
    is_observable: bool                  # rank == nx
    unobservable_modes: Optional[np.ndarray]  # Eigenvalues of unobservable modes


# ============================================================================
# Integration and Simulation Types
# ============================================================================

class IntegrationResult(TypedDict, total=False):
    """
    Result from continuous integration.
    
    Examples
    --------
    >>> result: IntegrationResult = integrator.solve(x0, u, t_span)
    >>> trajectory = result['y']
    >>> success = result['success']
    """
    t: TimePoints                        # Time points
    y: StateTrajectory                   # State trajectory
    success: bool                        # Integration successful
    message: str                         # Status message
    nfev: int                            # Number of function evaluations
    njev: int                            # Number of Jacobian evaluations


class SimulationResult(TypedDict, total=False):
    """
    Result from discrete simulation.
    
    Examples
    --------
    >>> result: SimulationResult = system.simulate(
    ...     x0, u_seq, steps=100, return_all=True
    ... )
    >>> states = result['states']
    >>> controls = result['controls']
    """
    states: StateTrajectory              # State trajectory
    controls: ControlSequence            # Control sequence used
    outputs: Optional[OutputSequence]    # Output sequence (if computed)
    noise: Optional[NoiseSequence]       # Noise sequence (stochastic)
    time: Optional[TimePoints]           # Time points (if applicable)
    info: Dict[str, Any]                 # Additional information


# ============================================================================
# Noise and Stochastic Types
# ============================================================================

NoiseType = Literal['additive', 'multiplicative', 'diagonal', 'scalar', 'general']
"""
Noise structure classification.

Values:
- 'additive': g(x,u,t) = constant
- 'multiplicative': g(x,u,t) depends on state
- 'diagonal': Independent noise sources
- 'scalar': Single noise source (nw=1)
- 'general': Full coupling

Examples
--------
>>> noise_type: NoiseType = system.get_noise_type()
"""

SDEType = Literal['ito', 'stratonovich']
"""
SDE interpretation type.

Values:
- 'ito': Itô interpretation
- 'stratonovich': Stratonovich interpretation

Examples
--------
>>> sde_type: SDEType = 'ito'
>>> system.sde_type = 'stratonovich'
"""

ConvergenceType = Literal['strong', 'weak']
"""
SDE convergence type.

Values:
- 'strong': Pathwise convergence (sample paths)
- 'weak': Weak convergence (distributions/moments)

Examples
--------
>>> conv_type: ConvergenceType = 'strong'
"""


# ============================================================================
# Control Design Result Types
# ============================================================================

class LQRResult(TypedDict):
    """
    LQR controller design result.
    
    Examples
    --------
    >>> result: LQRResult = design_lqr(system, Q, R)
    >>> K = result['gain']
    >>> P = result['cost_to_go']
    """
    gain: GainMatrix                     # Feedback gain K
    cost_to_go: CovarianceMatrix         # Solution to Riccati equation P
    closed_loop_eigenvalues: np.ndarray  # Eigenvalues of (A - B*K)
    stability_margin: float              # How far from instability


class KalmanFilterResult(TypedDict):
    """
    Kalman filter design result.
    
    Examples
    --------
    >>> result: KalmanFilterResult = design_kalman(system, Q, R)
    >>> L = result['gain']
    >>> P = result['error_covariance']
    """
    gain: GainMatrix                     # Kalman gain L
    error_covariance: CovarianceMatrix   # Steady-state P
    innovation_covariance: CovarianceMatrix  # Innovation S
    observer_eigenvalues: np.ndarray     # Eigenvalues of (A - L*C)


class LQGResult(TypedDict):
    """
    LQG (LQR + Kalman) controller design result.
    
    Combines optimal control and optimal estimation.
    
    Examples
    --------
    >>> result: LQGResult = design_lqg(system, Q_control, R_control, Q_noise, R_noise)
    >>> K_lqr = result['control_gain']
    >>> L_kalman = result['estimator_gain']
    >>> controller = LQGController(K_lqr, L_kalman)
    """
    control_gain: GainMatrix             # LQR gain K
    estimator_gain: GainMatrix           # Kalman gain L
    control_cost_to_go: CovarianceMatrix # Controller Riccati solution P_c
    estimation_error_covariance: CovarianceMatrix  # Estimator Riccati solution P_e
    separation_verified: bool            # Separation principle holds
    closed_loop_stable: bool             # Combined system stable
    controller_eigenvalues: np.ndarray   # Controller poles
    estimator_eigenvalues: np.ndarray    # Observer poles


class MPCResult(TypedDict, total=False):
    """
    MPC solution result.
    
    Examples
    --------
    >>> result: MPCResult = mpc.solve(x0, reference)
    >>> u_optimal = result['control_sequence']
    >>> predicted = result['predicted_trajectory']
    """
    control_sequence: ControlSequence    # Optimal control u*
    predicted_trajectory: StateTrajectory  # Predicted x
    cost: float                          # Optimal cost
    success: bool                        # Optimization succeeded
    iterations: int                      # Number of iterations
    solve_time: float                    # Computation time (seconds)
    constraint_violations: Optional[ArrayLike]  # Constraint slack variables
    dual_variables: Optional[ArrayLike]  # Lagrange multipliers


class MHEResult(TypedDict, total=False):
    """
    Moving Horizon Estimation result.
    
    Optimal state estimation over receding horizon.
    
    Examples
    --------
    >>> result: MHEResult = mhe.estimate(measurements, controls)
    >>> x_estimated = result['state_estimate']
    >>> P = result['covariance_estimate']
    """
    state_estimate: StateVector          # Optimal state estimate x̂
    covariance_estimate: CovarianceMatrix  # Estimated covariance P̂
    state_trajectory: StateTrajectory    # Estimated trajectory over horizon
    cost: float                          # Estimation cost
    success: bool                        # Optimization succeeded
    solve_time: float                    # Computation time
    innovation_sequence: OutputSequence  # Measurement innovations


class EKFResult(TypedDict, total=False):
    """
    Extended Kalman Filter state and result.
    
    For nonlinear systems with linearization at each step.
    
    Examples
    --------
    >>> ekf = ExtendedKalmanFilter(system, Q, R)
    >>> result: EKFResult = ekf.update(y_measured, u_applied)
    >>> x_hat = result['state_estimate']
    >>> P = result['covariance']
    """
    state_estimate: StateVector          # Current state estimate x̂[k]
    covariance: CovarianceMatrix         # Current covariance P[k]
    innovation: OutputVector             # Innovation y - ŷ
    innovation_covariance: CovarianceMatrix  # Innovation covariance S
    kalman_gain: GainMatrix              # Current Kalman gain K[k]
    likelihood: float                    # Log-likelihood of measurement


class UKFResult(TypedDict, total=False):
    """
    Unscented Kalman Filter result.
    
    Sigma-point based filter for nonlinear systems.
    
    Examples
    --------
    >>> ukf = UnscentedKalmanFilter(system, Q, R)
    >>> result: UKFResult = ukf.update(y_measured, u_applied)
    >>> x_hat = result['state_estimate']
    """
    state_estimate: StateVector          # State estimate
    covariance: CovarianceMatrix         # Covariance
    innovation: OutputVector             # Innovation
    sigma_points: StateTrajectory        # Sigma points used (2*nx+1, nx)
    weights_mean: ArrayLike              # Weights for mean
    weights_covariance: ArrayLike        # Weights for covariance


class ParticleFilterResult(TypedDict, total=False):
    """
    Particle Filter result.
    
    Sequential Monte Carlo estimation.
    
    Examples
    --------
    >>> pf = ParticleFilter(system, n_particles=1000)
    >>> result: ParticleFilterResult = pf.update(y_measured, u_applied)
    >>> x_hat = result['state_estimate']
    """
    state_estimate: StateVector          # Mean of particle distribution
    covariance: CovarianceMatrix         # Sample covariance
    particles: StateTrajectory           # All particles (n_particles, nx)
    weights: ArrayLike                   # Particle weights (n_particles,)
    effective_sample_size: float         # ESS (measure of degeneracy)
    resampled: bool                      # Whether resampling occurred


# ============================================================================
# Advanced Control Result Types
# ============================================================================

class H2ControlResult(TypedDict):
    """
    H₂ optimal control result.
    
    Minimizes H₂ norm (RMS response to white noise).
    
    Examples
    --------
    >>> result: H2ControlResult = design_h2_controller(system, W_input, W_output)
    >>> K = result['gain']
    >>> h2_norm = result['h2_norm']
    """
    gain: GainMatrix                     # H₂ optimal gain K
    h2_norm: float                       # Achieved H₂ norm
    cost_to_go: CovarianceMatrix         # Riccati solution
    closed_loop_stable: bool             # Stability of closed-loop
    closed_loop_poles: np.ndarray        # Closed-loop eigenvalues


class HInfControlResult(TypedDict):
    """
    H∞ robust control result.
    
    Minimizes H∞ norm (worst-case gain).
    
    Examples
    --------
    >>> result: HInfControlResult = design_hinf_controller(system, gamma=1.5)
    >>> K = result['gain']
    >>> hinf_norm = result['hinf_norm']
    """
    gain: GainMatrix                     # H∞ controller gain K
    hinf_norm: float                     # Achieved H∞ norm
    gamma: float                         # Performance bound γ
    central_solution: CovarianceMatrix   # Central Riccati solution
    feasible: bool                       # γ achievable
    robustness_margin: float             # Stability margin


class LMIResult(TypedDict, total=False):
    """
    Linear Matrix Inequality solver result.
    
    For synthesis via convex optimization.
    
    Examples
    --------
    >>> result: LMIResult = solve_lmi(constraints, objective)
    >>> P = result['decision_variables']['P']
    """
    decision_variables: Dict[str, ArrayLike]  # Solved variables (P, Y, etc.)
    objective_value: float               # Optimal objective
    feasible: bool                       # LMI feasible
    solver: str                          # Solver used ('cvxpy', 'mosek', etc.)
    solve_time: float                    # Computation time
    condition_number: float              # Numerical conditioning


class AdaptiveControlResult(TypedDict, total=False):
    """
    Adaptive control result.
    
    For systems with parameter uncertainty.
    
    Examples
    --------
    >>> result: AdaptiveControlResult = adaptive_controller.update(x, y, u)
    >>> K_adapted = result['current_gain']
    >>> theta_hat = result['parameter_estimate']
    """
    current_gain: GainMatrix             # Current adapted gain
    parameter_estimate: ParameterVector  # Estimated parameters θ̂
    parameter_covariance: CovarianceMatrix  # Parameter uncertainty
    adaptation_rate: float               # Current learning rate
    tracking_error: float                # Output tracking error
    parameter_error: Optional[ParameterVector]  # θ̂ - θ_true (if known)


class SlidingModeResult(TypedDict):
    """
    Sliding Mode Control result.
    
    For robust nonlinear control.
    
    Examples
    --------
    >>> result: SlidingModeResult = smc.compute_control(x, x_desired)
    >>> u = result['control']
    >>> on_surface = result['on_sliding_surface']
    """
    control: ControlVector               # SMC control signal
    sliding_variable: ArrayLike          # Sliding surface variable s
    on_sliding_surface: bool             # |s| < ε
    reaching_time_estimate: Optional[float]  # Estimated time to surface
    chattering_magnitude: float          # Control chattering level


# ============================================================================
# System Identification Types
# ============================================================================

HankelMatrix = ArrayLike
"""
Hankel matrix for system identification.

Block Hankel matrix constructed from input-output data:
    H = [y(0)   y(1)   y(2)   ... y(L-1)  ]
        [y(1)   y(2)   y(3)   ... y(L)    ]
        [y(2)   y(3)   y(4)   ... y(L+1)  ]
        [  ⋮      ⋮      ⋮    ⋱    ⋮      ]
        [y(N-1) y(N)   y(N+1) ... y(N+L-2)]

Used in:
- ERA (Eigensystem Realization Algorithm)
- Ho-Kalman algorithm  
- Subspace identification (N4SID, MOESP)
- DMD (Dynamic Mode Decomposition)

Shape: (rows, cols) where rows = observability index, cols = controllability index

Examples
--------
>>> # Build Hankel from measurements
>>> H: HankelMatrix = build_hankel_matrix(y_data, rows=10, cols=20)
>>> print(H.shape)  # (10*ny, 20)
>>> 
>>> # SVD for order selection
>>> U, s, Vt = np.linalg.svd(H)
>>> n_states = np.sum(s > threshold)
"""

ToeplitzMatrix = ArrayLike
"""
Toeplitz matrix (constant diagonals).

Used in:
- Impulse response analysis
- Convolution operations
- System identification

Examples
--------
>>> from scipy.linalg import toeplitz
>>> T: ToeplitzMatrix = toeplitz(impulse_response)
"""

TrajectoryMatrix = ArrayLike
"""
Data matrix for trajectory-based methods.

Shapes:
- Input matrix: (n_samples, nu)
- Output matrix: (n_samples, ny)
- State matrix: (n_samples, nx)

Used in:
- DMD (Dynamic Mode Decomposition)
- SINDy (Sparse Identification of Nonlinear Dynamics)
- Koopman operator methods

Examples
--------
>>> X: TrajectoryMatrix = np.column_stack([x[:-1] for x in trajectories])
>>> Y: TrajectoryMatrix = np.column_stack([x[1:] for x in trajectories])
>>> # Learn dynamics: Y ≈ A @ X
"""

MarkovParameters = ArrayLike
"""
Markov parameters (impulse response coefficients).

For discrete LTI: y[k] = Σ M[i] * u[k-i]
Where M[i] = C * A^(i-1) * B

Shape: (n_parameters, ny, nu)

Used in ERA, subspace ID.

Examples
--------
>>> markov: MarkovParameters = system.compute_markov_parameters(n=20)
>>> print(markov.shape)  # (20, ny, nu)
"""

CorrelationMatrix = ArrayLike
"""
Correlation or covariance matrix for data analysis.

Examples:
- R_uu: Input autocorrelation
- R_yy: Output autocorrelation  
- R_yu: Input-output cross-correlation

Examples
--------
>>> R_yy: CorrelationMatrix = np.correlate(y_data, y_data, mode='full')
"""

class SystemIDResult(TypedDict, total=False):
    """
    System identification result.
    
    Result from identifying system from data.
    
    Examples
    --------
    >>> result: SystemIDResult = identify_system(u_data, y_data, order=3)
    >>> A_identified = result['A']
    >>> fit_percentage = result['fit_percentage']
    """
    A: StateMatrix                       # Identified state matrix
    B: ControlMatrix                     # Identified control matrix
    C: OutputMatrix                      # Identified output matrix
    D: FeedthroughMatrix                 # Identified feedthrough
    G: Optional[DiffusionMatrix]         # Noise gain (stochastic ID)
    Q: Optional[CovarianceMatrix]        # Process noise covariance
    R: Optional[CovarianceMatrix]        # Measurement noise covariance
    order: int                           # Model order (nx)
    fit_percentage: float                # Model fit quality (0-100%)
    residuals: ArrayLike                 # Prediction residuals
    method: str                          # ID method used ('n4sid', 'era', etc.)
    hankel_matrix: Optional[HankelMatrix]  # Hankel matrix (if applicable)
    singular_values: Optional[ArrayLike] # SVD singular values


class SubspaceIDResult(TypedDict, total=False):
    """
    Subspace identification specific result.
    
    Methods: N4SID, MOESP, CVA, etc.
    
    Examples
    --------
    >>> result: SubspaceIDResult = n4sid(u_data, y_data, order=5)
    >>> A = result['A']
    >>> observability_matrix = result['observability_matrix']
    """
    A: StateMatrix                       # State matrix
    B: ControlMatrix                     # Control matrix
    C: OutputMatrix                      # Output matrix
    D: FeedthroughMatrix                 # Feedthrough
    observability_matrix: ObservabilityMatrix  # Extended observability
    controllability_matrix: ControllabilityMatrix  # Extended controllability
    hankel_matrix: HankelMatrix          # Data Hankel matrix
    projection_matrix: ArrayLike         # Oblique/orthogonal projection
    singular_values: ArrayLike           # For order selection
    order: int                           # Selected model order
    fit_quality: float                   # VAF (Variance Accounted For)


class ERAResult(TypedDict):
    """
    Eigensystem Realization Algorithm result.
    
    Identifies system from impulse response data.
    
    Examples
    --------
    >>> result: ERAResult = era(markov_parameters, order=10)
    >>> A = result['A']
    >>> modes = result['system_modes']
    """
    A: StateMatrix                       # Identified A matrix
    B: ControlMatrix                     # Identified B matrix
    C: OutputMatrix                      # Identified C matrix
    D: FeedthroughMatrix                 # Identified D matrix
    hankel_matrix: HankelMatrix          # Built from Markov parameters
    singular_values: ArrayLike           # Hankel SVD singular values
    system_modes: np.ndarray             # Identified system modes (eigenvalues)
    order: int                           # Model order
    observability_matrix: ObservabilityMatrix
    controllability_matrix: ControllabilityMatrix


class DMDResult(TypedDict, total=False):
    """
    Dynamic Mode Decomposition result.
    
    Data-driven modal decomposition and prediction.
    
    Examples
    --------
    >>> result: DMDResult = dmd(X, Y, rank=10)
    >>> A_dmd = result['dynamics_matrix']
    >>> modes = result['modes']
    >>> eigenvalues = result['eigenvalues']
    """
    dynamics_matrix: StateMatrix         # DMD dynamics matrix Ã
    modes: ArrayLike                     # DMD modes (spatial patterns)
    eigenvalues: np.ndarray              # DMD eigenvalues (temporal behavior)
    amplitudes: ArrayLike                # Mode amplitudes
    frequencies: ArrayLike               # Mode frequencies (continuous-time)
    growth_rates: ArrayLike              # Mode growth/decay rates
    rank: int                            # Truncation rank
    singular_values: ArrayLike           # SVD singular values


class SINDyResult(TypedDict, total=False):
    """
    SINDy (Sparse Identification of Nonlinear Dynamics) result.
    
    Discovers governing equations from data.
    
    Examples
    --------
    >>> result: SINDyResult = sindy(X, dX, library_functions)
    >>> coefficients = result['coefficients']
    >>> identified_terms = result['active_terms']
    """
    coefficients: ArrayLike              # Sparse coefficient matrix (n_features, nx)
    active_terms: List[str]              # Identified active terms
    library_functions: List[Callable]    # Basis functions used
    sparsity_level: float                # Fraction of zero coefficients
    reconstruction_error: float          # ||dX - Θ*Ξ||
    condition_number: float              # Library matrix condition number
    selected_features: List[int]         # Indices of selected features


class KoopmanResult(TypedDict, total=False):
    """
    Koopman operator approximation result.
    
    Linear representation of nonlinear dynamics in lifted space.
    
    Examples
    --------
    >>> result: KoopmanResult = koopman_approximation(trajectories, observables)
    >>> K_operator = result['koopman_operator']
    >>> lifted_dim = result['lifted_dimension']
    """
    koopman_operator: StateMatrix        # Koopman matrix K
    lifting_functions: List[Callable]    # Observable functions φ(x)
    lifted_dimension: int                # Dimension of lifted space
    eigenvalues: np.ndarray              # Koopman eigenvalues
    eigenfunctions: ArrayLike            # Koopman eigenfunctions
    reconstruction_error: float          # Prediction error
    method: str                          # 'EDMD', 'DMD', 'kernel', 'neural'


# ============================================================================
# Model Reduction Types
# ============================================================================

class BalancedRealizationResult(TypedDict):
    """
    Balanced realization result.
    
    Balances controllability and observability gramians.
    
    Examples
    --------
    >>> result: BalancedRealizationResult = balanced_realization(system)
    >>> A_bal = result['A_balanced']
    >>> hsv = result['hankel_singular_values']
    """
    A_balanced: StateMatrix              # Balanced state matrix
    B_balanced: ControlMatrix            # Balanced control matrix
    C_balanced: OutputMatrix             # Balanced output matrix
    transformation: ArrayLike            # Balancing transformation T
    hankel_singular_values: ArrayLike    # HSVs (for truncation)
    controllability_gramian: CovarianceMatrix  # Wc
    observability_gramian: CovarianceMatrix    # Wo


class ReducedOrderModelResult(TypedDict):
    """
    Model order reduction result.
    
    Reduced-order approximation of high-order system.
    
    Examples
    --------
    >>> result: ReducedOrderModelResult = reduce_model(system, target_order=5)
    >>> system_reduced = result['reduced_system']
    >>> error_bound = result['approximation_error']
    """
    A_reduced: StateMatrix               # Reduced A (n_r, n_r)
    B_reduced: ControlMatrix             # Reduced B (n_r, nu)
    C_reduced: OutputMatrix              # Reduced C (ny, n_r)
    D_reduced: FeedthroughMatrix         # Reduced D (ny, nu)
    original_order: int                  # Original nx
    reduced_order: int                   # Reduced n_r
    approximation_error: float           # H∞ or H₂ error bound
    method: str                          # Method used ('balanced', 'modal', etc.)
    preserved_modes: np.ndarray          # Eigenvalues kept in reduction


# ============================================================================
# Reachability and Safety Types
# ============================================================================

ReachableSet = ArrayLike
"""
Reachable set representation.

Can be:
- Polytope vertices: (n_vertices, nx)
- Ellipsoid parameters: (center, shape_matrix)
- Zonotope generators: (n_generators, nx)
- Grid samples: (n_samples, nx)

Examples
--------
>>> reachable: ReachableSet = compute_reachable_set(system, x0, u_bounds, steps=10)
"""

SafeSet = ArrayLike
"""
Safe set representation (invariant/viability).

Similar representations to ReachableSet.

Examples
--------
>>> safe: SafeSet = compute_safe_set(system, constraints)
"""

class ReachabilityResult(TypedDict, total=False):
    """
    General reachability analysis result.
    
    Examples
    --------
    >>> result: ReachabilityResult = analyze_reachability(system, x0, horizon=10)
    >>> reachable_set = result['reachable_set']
    """
    reachable_set: ReachableSet          # Reachable set at final time
    reachable_tube: List[ReachableSet]   # Reachable set at each time
    volume: float                        # Volume of reachable set
    representation: str                  # 'polytope', 'ellipsoid', 'zonotope'
    method: str                          # Method used
    computation_time: float              # Time to compute


class ROAResult(TypedDict):
    """
    Region of Attraction analysis result.
    
    For Lyapunov-based stability analysis.
    
    Examples
    --------
    >>> result: ROAResult = compute_roa(system, V_lyapunov)
    >>> roa_estimate = result['region_of_attraction']
    >>> volume = result['volume_estimate']
    """
    region_of_attraction: SafeSet        # ROA estimate
    lyapunov_function: Callable          # V(x) Lyapunov function
    lyapunov_matrix: CovarianceMatrix    # P in V(x) = x'Px
    level_set: float                     # c where V(x) ≤ c defines ROA
    volume_estimate: float               # ROA volume
    verification_samples: int            # Samples used for verification
    certification_method: str            # 'SOS', 'sampling', 'LMI'


class VerificationResult(TypedDict, total=False):
    """
    Formal verification result.
    
    For proving safety/reachability properties.
    
    Examples
    --------
    >>> result: VerificationResult = verify_safety(system, safe_set, horizon=10)
    >>> if result['verified']:
    ...     print(f"Safety certified with confidence {result['confidence']}")
    """
    verified: bool                       # Property verified
    property_type: str                   # 'safety', 'reachability', 'liveness'
    confidence: float                    # Confidence level (0-1)
    counterexample: Optional[StateTrajectory]  # Violating trajectory if not verified
    certification_method: str            # Method used
    computation_time: float              # Time to verify


class BarrierCertificateResult(TypedDict):
    """
    Barrier certificate result.
    
    For safety verification via barrier functions.
    
    Examples
    --------
    >>> result: BarrierCertificateResult = find_barrier_certificate(
    ...     system, safe_set, unsafe_set
    ... )
    >>> B = result['barrier_function']
    """
    barrier_function: Callable[[StateVector], float]  # B(x)
    barrier_matrix: Optional[CovarianceMatrix]  # P if B(x) = x'Px
    valid: bool                          # Barrier conditions satisfied
    safe_set: SafeSet                    # Certified safe region
    unsafe_set: Optional[ArrayLike]      # Unsafe region
    method: str                          # 'SOS', 'LP', 'neural'


class CBFResult(TypedDict):
    """
    Control Barrier Function result.
    
    For safety-critical control synthesis.
    
    Examples
    --------
    >>> cbf = ControlBarrierFunction(barrier, system)
    >>> result: CBFResult = cbf.filter_control(x, u_desired)
    >>> u_safe = result['safe_control']
    """
    safe_control: ControlVector          # Safety-filtered control
    barrier_value: float                 # B(x) at current state
    barrier_derivative: float            # dB/dt or ΔB
    constraint_active: bool              # Safety constraint active
    nominal_control: ControlVector       # Original desired control
    modification_magnitude: float        # ||u_safe - u_desired||


class CLFResult(TypedDict):
    """
    Control Lyapunov Function result.
    
    For stability-guaranteed control synthesis.
    
    Examples
    --------
    >>> clf = ControlLyapunovFunction(lyapunov, system)
    >>> result: CLFResult = clf.synthesize_control(x)
    >>> u = result['stabilizing_control']
    """
    stabilizing_control: ControlVector   # Stabilizing control
    lyapunov_value: float                # V(x)
    lyapunov_derivative: float           # dV/dt or ΔV
    descent_rate: float                  # -dV/dt (should be positive)
    feasible: bool                       # Stabilizing control exists


# ============================================================================
# HJI Reachability and Differential Games Types
# ============================================================================

ValueFunction = Callable[[StateVector, float], float]
"""
Value function for optimal control or reachability.

For HJI: V(x, t) satisfies Hamilton-Jacobi-Isaacs equation.

Examples
--------
>>> V: ValueFunction = lambda x, t: x.T @ P(t) @ x
>>> value = V(x_current, t_current)
"""

LevelSet = float
"""
Level set value for reachability analysis.

The set {x : V(x, t) ≤ c} defines reachable/safe region.

Examples
--------
>>> target_level: LevelSet = 0.0
>>> safe_region = {x : V(x, t) <= target_level}
"""

class HJIReachabilityResult(TypedDict, total=False):
    """
    Hamilton-Jacobi-Isaacs reachability result.
    
    Computes backward/forward reachable sets via HJI PDE.
    
    Examples
    --------
    >>> result: HJIReachabilityResult = hji_reachability(
    ...     system, target_set, time_horizon=5.0
    ... )
    >>> brt = result['backward_reachable_tube']
    >>> value_function = result['value_function']
    """
    value_function: ValueFunction        # Solution to HJI: V(x, t)
    value_grid: ArrayLike                # Discretized value function
    grid_points: ArrayLike               # Grid for level set method
    backward_reachable_tube: List[ArrayLike]  # BRT at each time
    backward_reachable_set: ArrayLike    # BRS at t=0
    optimal_control: Optional[Callable]  # u*(x, t) from HJI
    optimal_disturbance: Optional[Callable]  # d*(x, t) for games
    time_points: TimePoints              # Time discretization
    target_set: ArrayLike                # Target/goal set
    avoid_set: Optional[ArrayLike]       # Avoid/obstacle set
    method: str                          # 'level_set', 'viability_kernel'


class BackwardReachableResult(TypedDict):
    """
    Backward reachability result.
    
    What states can reach target within time horizon?
    
    Examples
    --------
    >>> result: BackwardReachableResult = backward_reachable(
    ...     system, target_set, horizon=10
    ... )
    >>> brs = result['backward_reachable_set']
    >>> if x0 in brs:
    ...     print("Can reach target!")
    """
    backward_reachable_set: ReachableSet  # BRS at initial time
    backward_reachable_tube: List[ReachableSet]  # BRT over time
    capture_basin: ReachableSet          # Guaranteed capture region
    time_to_reach: Optional[Callable]    # T(x) minimum time to target
    value_function: Optional[ValueFunction]  # HJI value function
    method: str                          # Computation method


class ForwardReachableResult(TypedDict):
    """
    Forward reachability result.
    
    What states can be reached from initial set?
    
    Examples
    --------
    >>> result: ForwardReachableResult = forward_reachable(
    ...     system, initial_set, horizon=10
    ... )
    >>> frs = result['forward_reachable_set']
    """
    forward_reachable_set: ReachableSet  # FRS at final time
    forward_reachable_tube: List[ReachableSet]  # FRT over time
    bloating_bound: Optional[float]      # Approximation error bound
    method: str                          # 'lagrangian', 'eulerian', 'zonotope'
    computation_time: float              # Time to compute


class ViabilityKernelResult(TypedDict):
    """
    Viability kernel result.
    
    Largest controlled-invariant set within constraints.
    
    Examples
    --------
    >>> result: ViabilityKernelResult = compute_viability_kernel(
    ...     system, constraint_set
    ... )
    >>> viab_kernel = result['viability_kernel']
    >>> if x0 in viab_kernel:
    ...     print("Can stay safe forever!")
    """
    viability_kernel: SafeSet            # Largest viable set
    discriminating_kernel: SafeSet       # Can avoid avoid_set forever
    capture_basin: Optional[ReachableSet]  # For reach-avoid
    value_function: Optional[ValueFunction]  # Viability value function
    control_strategy: Optional[Callable]  # Viable control policy
    iterations: int                      # Algorithm iterations
    convergence: bool                    # Whether converged


class DifferentialGameResult(TypedDict, total=False):
    """
    Differential game solution result.
    
    Two-player (control vs disturbance) optimal strategies.
    
    Examples
    --------
    >>> result: DifferentialGameResult = solve_differential_game(
    ...     system, target_set, avoid_set
    ... )
    >>> u_star = result['control_strategy']
    >>> d_star = result['disturbance_strategy']
    """
    value_function: ValueFunction        # Game value V(x, t)
    control_strategy: Callable           # Optimal control u*(x, t)
    disturbance_strategy: Callable       # Optimal disturbance d*(x, t)
    winning_region: SafeSet              # Where control wins
    losing_region: Optional[ArrayLike]   # Where disturbance wins
    saddle_point_verified: bool          # Nash equilibrium exists
    game_type: str                       # 'reach-avoid', 'pursuit-evasion'


# ============================================================================
# Robustness and Uncertainty Types
# ============================================================================

UncertaintySet = Union[ArrayLike, Tuple[ArrayLike, ArrayLike]]
"""
Uncertainty set representation for parametric uncertainty.

Can be:
- Polytope: vertices array (n_vertices, n_params)
- Ellipsoid: (center, shape_matrix)
- Box: (lower_bounds, upper_bounds)

Note: This is for parametric uncertainty (m, k, damping).
For prediction uncertainty, see ConformalPredictionSet.

Examples
--------
>>> # Box uncertainty
>>> uncertainty: UncertaintySet = (theta_min, theta_max)
>>> 
>>> # Ellipsoid uncertainty  
>>> uncertainty: UncertaintySet = (theta_nominal, Sigma)
"""

class RobustStabilityResult(TypedDict, total=False):
    """
    Robust stability analysis result.
    
    Stability under parametric uncertainty.
    
    Examples
    --------
    >>> result: RobustStabilityResult = analyze_robust_stability(
    ...     system, uncertainty_set
    ... )
    >>> is_robust = result['robustly_stable']
    """
    robustly_stable: bool                # Stable for all uncertainties
    worst_case_eigenvalue: complex       # Closest to instability
    stability_margin: float              # Minimum stability margin
    critical_parameter: Optional[ParameterVector]  # Worst-case parameters
    method: str                          # Analysis method
    conservatism: Optional[float]        # Conservatism estimate


class StructuredSingularValueResult(TypedDict):
    """
    Structured Singular Value (μ-analysis) result.
    
    For robust performance analysis.
    
    Examples
    --------
    >>> result: StructuredSingularValueResult = mu_analysis(system, Delta)
    >>> mu = result['mu_value']
    >>> margin = 1 / mu
    """
    mu_value: float                      # Structured singular value μ
    robustness_margin: float             # 1/μ (stability margin)
    worst_case_uncertainty: ArrayLike    # Critical Δ
    frequency_grid: Optional[ArrayLike]  # Frequency points analyzed
    upper_bound: float                   # μ upper bound
    lower_bound: float                   # μ lower bound


class TubeDefinition(TypedDict):
    """
    Tube for robust control.
    
    Parameterizes reachable set tube over time.
    
    Examples
    --------
    >>> tube: TubeDefinition = {
    ...     'shape': 'ellipsoid',
    ...     'center_trajectory': nominal_trajectory,
    ...     'tube_radii': radii_over_time
    ... }
    """
    shape: str                           # 'ellipsoid', 'polytope', 'box'
    center_trajectory: StateTrajectory   # Nominal center
    tube_radii: ArrayLike                # Size at each time (n_steps,)
    shape_matrices: Optional[List[CovarianceMatrix]]  # For ellipsoids


class TubeMPCResult(TypedDict, total=False):
    """
    Tube-based MPC result.
    
    Robust MPC with disturbance-invariant tubes.
    
    Examples
    --------
    >>> result: TubeMPCResult = tube_mpc.solve(x0, disturbance_bound)
    >>> u_nominal = result['nominal_control']
    >>> u_feedback = result['feedback_control']
    """
    nominal_control: ControlSequence     # Nominal control v
    feedback_control: GainMatrix         # Ancillary feedback K
    actual_control: ControlSequence      # Applied u = v + K(x - x_nom)
    nominal_trajectory: StateTrajectory  # Nominal state trajectory
    tube_definition: TubeDefinition      # Robust invariant tube
    tightened_constraints: ArrayLike     # Constraints - tube size


# ============================================================================
# Stochastic Control Types
# ============================================================================

class StochasticMPCResult(TypedDict, total=False):
    """
    Stochastic MPC result.
    
    Handles chance constraints and distributional robustness.
    
    Examples
    --------
    >>> result: StochasticMPCResult = stochastic_mpc.solve(x0, disturbance_model)
    >>> u = result['control_sequence']
    >>> risk = result['constraint_violation_probability']
    """
    control_sequence: ControlSequence    # Optimal control
    predicted_mean: StateTrajectory      # Mean prediction
    predicted_covariance: List[CovarianceMatrix]  # Covariance at each step
    constraint_violation_probability: float  # Risk level
    expected_cost: float                 # Expected cost
    cost_variance: float                 # Cost variance
    robust_feasible: bool                # Robustly feasible
    chance_constraint_levels: ArrayLike  # Confidence levels achieved


class RiskSensitiveResult(TypedDict):
    """
    Risk-sensitive control result.
    
    Penalizes cost variance in addition to mean.
    
    Examples
    --------
    >>> result: RiskSensitiveResult = design_risk_sensitive(system, Q, R, theta)
    >>> K = result['gain']
    >>> risk_level = result['risk_parameter']
    """
    gain: GainMatrix                     # Risk-sensitive gain
    cost_to_go: CovarianceMatrix         # Modified Riccati solution
    risk_parameter: float                # θ (risk aversion)
    expected_cost: float                 # E[J]
    cost_variance: float                 # Var[J]
    exponential_cost: float              # E[exp(θ*J)]


# ============================================================================
# Contraction Analysis Types
# ============================================================================

ContractionMetric = ArrayLike
"""
Contraction metric M(x) for contraction analysis.

Defines Riemannian metric M(x) ≻ 0 such that:
    d/dt ||δx||_M ≤ -β ||δx||_M

where ||δx||_M² = δx' M(x) δx

Can be:
- Constant: M(x) = M (uniform metric)
- State-dependent: M(x) varies with x
- Matrix-valued function

Shape: (nx, nx) symmetric positive definite

Examples
--------
>>> # Constant metric
>>> M: ContractionMetric = np.eye(nx)
>>> 
>>> # State-dependent (function)
>>> M: Callable[[StateVector], ContractionMetric] = lambda x: compute_metric(x)
"""

ContractionRate = float
"""
Contraction rate β > 0.

System contracts if d/dt ||δx|| ≤ -β ||δx||.
Larger β means faster contraction.

Examples
--------
>>> beta: ContractionRate = 0.5
>>> # Trajectories converge exponentially: ||δx(t)|| ≤ e^(-β*t) ||δx(0)||
"""

class ContractionAnalysisResult(TypedDict, total=False):
    """
    Contraction analysis result.
    
    Determines if system is contracting and computes contraction rate.
    
    Examples
    --------
    >>> result: ContractionAnalysisResult = analyze_contraction(system)
    >>> if result['is_contracting']:
    ...     print(f"Contracting with rate β={result['contraction_rate']}")
    ...     print(f"Convergence bound: {result['convergence_bound']}")
    """
    is_contracting: bool                 # Whether system is contracting
    contraction_rate: ContractionRate    # Rate β (if contracting)
    metric: ContractionMetric            # Contraction metric M(x)
    metric_type: str                     # 'constant', 'state_dependent', 'control_dependent'
    verification_method: str             # 'LMI', 'SOS', 'optimization', 'analytic'
    convergence_bound: Optional[Callable]  # ||δx(t)|| ≤ bound(t, ||δx(0)||)
    exponential_convergence: bool        # Exponential vs polynomial
    incremental_stability: bool          # Incremental asymptotic stability
    condition_number: Optional[float]    # cond(M) if applicable


class CCMResult(TypedDict, total=False):
    """
    Control Contraction Metrics result.
    
    Designing controllers via contraction + control.
    
    Examples
    --------
    >>> result: CCMResult = design_ccm_controller(system, contraction_rate=0.5)
    >>> K = result['feedback_gain']
    >>> M = result['metric']
    >>> verified = result['contraction_verified']
    """
    feedback_gain: GainMatrix            # State-dependent K(x) or constant K
    metric: ContractionMetric            # Contraction metric M(x)
    contraction_rate: ContractionRate    # Guaranteed rate β
    metric_condition_number: ArrayLike   # κ(M(x)) over state space
    contraction_verified: bool           # LMI/SOS verification succeeded
    robustness_margin: float             # How much margin in contraction
    geodesic_distance: Optional[Callable]  # Distance in M-metric


class FunnelingResult(TypedDict):
    """
    Funnel control result (contraction-based tracking).
    
    Guarantees tracking within time-varying funnel.
    
    Examples
    --------
    >>> result: FunnelingResult = design_funnel_controller(
    ...     system, reference_trajectory, funnel_shape
    ... )
    >>> controller = result['controller']
    >>> funnel = result['tracking_funnel']
    """
    controller: Callable                 # Funnel controller u(x, t)
    tracking_funnel: Callable            # Funnel boundary ρ(t)
    funnel_shape: str                    # 'exponential', 'polynomial', 'custom'
    reference_trajectory: StateTrajectory  # Desired x_d(t)
    performance_bound: Callable          # ||x - x_d|| ≤ bound(t)
    transient_bound: float               # Initial error amplification
    contraction_rate: ContractionRate    # Asymptotic contraction rate


class IncrementalStabilityResult(TypedDict):
    """
    Incremental stability analysis result.
    
    Convergence of trajectories to each other (not just to equilibrium).
    
    Examples
    --------
    >>> result: IncrementalStabilityResult = check_incremental_stability(system)
    >>> if result['incrementally_stable']:
    ...     print("All trajectories converge to each other")
    """
    incrementally_stable: bool           # δ-GAS (global asymptotic stability)
    contraction_rate: Optional[ContractionRate]  # If contracting
    metric: Optional[ContractionMetric]  # M(x) for analysis
    kl_bound: Optional[Callable]         # KL stability bound
    convergence_type: str                # 'exponential', 'asymptotic', 'finite_time'


# ============================================================================
# Differential Flatness Types
# ============================================================================

FlatnessOutput = ArrayLike
"""
Flat output for differentially flat systems.

Special output y_flat such that:
    x = φ_x(y_flat, ẏ_flat, ÿ_flat, ...)
    u = φ_u(y_flat, ẏ_flat, ÿ_flat, ...)

Dimension: (n_flat,) where n_flat ≤ nx

Examples
--------
>>> y_flat: FlatnessOutput = np.array([x_pos, y_pos])  # Quadrotor: position is flat
"""

class DifferentialFlatnessResult(TypedDict):
    """
    Differential flatness analysis result.
    
    Determines if system is differentially flat.
    
    Examples
    --------
    >>> result: DifferentialFlatnessResult = check_flatness(system)
    >>> if result['is_flat']:
    ...     y_flat = result['flat_output']
    ...     print(f"Flat output dimension: {result['flat_dimension']}")
    """
    is_flat: bool                        # Whether system is flat
    flat_output: Optional[Callable]      # y_flat = σ(x)
    flat_dimension: int                  # Dimension of flat output
    differential_order: int              # Max derivative order needed
    state_from_flat: Optional[Callable]  # x = φ_x(y, ẏ, ÿ, ...)
    control_from_flat: Optional[Callable]  # u = φ_u(y, ẏ, ÿ, ...)
    verification_method: str             # How flatness was verified


class TrajectoryPlanningResult(TypedDict, total=False):
    """
    Trajectory planning result (often using flatness).
    
    Plans feasible trajectory from initial to final state.
    
    Examples
    --------
    >>> result: TrajectoryPlanningResult = plan_trajectory(
    ...     system, x_initial, x_final, time_horizon=5.0
    ... )
    >>> reference = result['state_trajectory']
    >>> controls = result['control_trajectory']
    """
    state_trajectory: StateTrajectory    # Planned state x(t) or x[k]
    control_trajectory: ControlSequence  # Required control u(t) or u[k]
    flat_trajectory: Optional[ArrayLike]  # y_flat(t) if flat system
    time_points: TimePoints              # Time discretization
    cost: float                          # Trajectory cost
    feasible: bool                       # Satisfies constraints
    method: str                          # Planning method
    computation_time: float              # Planning time


# ============================================================================
# Conformal Prediction Types
# ============================================================================

ConformalPredictionSet = Union[ArrayLike, Tuple[ArrayLike, ArrayLike]]
"""
Conformal prediction set for future states.

Provides distribution-free prediction intervals with coverage guarantees:
    P(x_true ∈ prediction_set) ≥ 1 - α

Different from UncertaintySet (parametric) - this is for prediction uncertainty.

Can be:
- Interval: (lower_bounds, upper_bounds) for each dimension
- Ball: (center, radius)
- Polytope: vertices array
- General: indicator function

Examples
--------
>>> # Calibrate on past residuals
>>> prediction_set: ConformalPredictionSet = (x_lower, x_upper)
>>> 
>>> # Guarantee: 90% coverage
>>> alpha = 0.1
>>> set_90 = conformal_calibrator.predict(x_test, alpha=alpha)
"""

NonconformityScore = ArrayLike
"""
Nonconformity scores for conformal prediction.

Measures "strangeness" of each prediction.
Lower score = more conforming to training data.

Shape: (n_samples,)

Examples
--------
>>> scores: NonconformityScore = np.abs(y_pred - y_true)  # Absolute residual
>>> scores: NonconformityScore = mahalanobis_distance(y_pred, y_true, Sigma)
"""

class ConformalCalibrationResult(TypedDict):
    """
    Conformal prediction calibration result.
    
    Result from calibrating conformal predictor on data.
    
    Examples
    --------
    >>> result: ConformalCalibrationResult = calibrate_conformal(
    ...     residuals, alpha=0.1
    ... )
    >>> quantile = result['quantile']
    >>> coverage = result['empirical_coverage']
    """
    quantile: float                      # Calibrated quantile for level α
    alpha: float                         # Target miscoverage level
    empirical_coverage: float            # Observed coverage on calibration set
    calibration_scores: NonconformityScore  # Nonconformity scores used
    n_calibration: int                   # Calibration set size
    prediction_set_type: str             # 'interval', 'ball', 'polytope'


class ConformalPredictionResult(TypedDict, total=False):
    """
    Conformal prediction result for test points.
    
    Provides prediction sets with coverage guarantees.
    
    Examples
    --------
    >>> cp = ConformalPredictor(model, calibration_data)
    >>> result: ConformalPredictionResult = cp.predict(x_test, alpha=0.1)
    >>> prediction_set = result['prediction_set']
    >>> set_size = result['average_set_size']
    """
    prediction_set: ConformalPredictionSet  # Prediction set(s)
    point_prediction: StateVector        # Point prediction (set center)
    coverage_guarantee: float            # Guaranteed coverage (1 - α)
    average_set_size: float              # Average prediction set size
    nonconformity_score: NonconformityScore  # Score for test point
    threshold: float                     # Threshold for set construction
    adaptive: bool                       # Whether adaptive to x


class AdaptiveConformalResult(TypedDict):
    """
    Adaptive Conformal Inference result.
    
    For time series with distribution shift.
    
    Examples
    --------
    >>> aci = AdaptiveConformalInference(model, gamma=0.01)
    >>> result: AdaptiveConformalResult = aci.update(x_new, y_new)
    >>> current_threshold = result['threshold']
    """
    threshold: float                     # Current adaptive threshold
    coverage_history: ArrayLike          # Coverage over time
    miscoverage_rate: float              # Current miscoverage
    target_alpha: float                  # Target miscoverage
    adaptation_rate: float               # Learning rate γ
    effective_sample_size: int           # Effective calibration size


# ============================================================================
# Learning and Neural Network Types
# ============================================================================

Dataset = Tuple[StateTrajectory, ControlSequence, OutputSequence]
"""
System identification dataset: (states, controls, outputs).

Examples
--------
>>> dataset: Dataset = (x_data, u_data, y_data)
>>> states, controls, outputs = dataset
"""

TrainingBatch = Tuple[StateVector, ControlVector, StateVector]
"""
Single training batch: (x[k], u[k], x[k+1]).

For learning dynamics models.

Examples
--------
>>> batch: TrainingBatch = (x_current, u_current, x_next)
"""

LearningRate = float
"""Learning rate for optimization/training."""

LossValue = float
"""Loss/cost function value."""

class NeuralNetworkConfig(TypedDict, total=False):
    """
    Neural network configuration for learning.
    
    Examples
    --------
    >>> config: NeuralNetworkConfig = {
    ...     'hidden_layers': [64, 64, 32],
    ...     'activation': 'relu',
    ...     'learning_rate': 1e-3
    ... }
    """
    hidden_layers: List[int]             # Hidden layer sizes
    activation: str                      # Activation function
    learning_rate: float                 # Learning rate
    batch_size: int                      # Training batch size
    epochs: int                          # Number of epochs
    optimizer: str                       # Optimizer type
    regularization: Optional[float]      # L2 regularization strength
    dropout: Optional[float]             # Dropout probability


class TrainingResult(TypedDict, total=False):
    """
    Neural network training result.
    
    Examples
    --------
    >>> result: TrainingResult = train_neural_ode(model, data, epochs=100)
    >>> final_loss = result['final_loss']
    >>> history = result['loss_history']
    """
    final_loss: float                    # Final training loss
    best_loss: float                     # Best loss achieved
    loss_history: List[float]            # Loss per epoch
    validation_loss: Optional[float]     # Validation set loss
    training_time: float                 # Total training time
    epochs_trained: int                  # Epochs completed
    early_stopped: bool                  # Whether early stopping triggered
    model_state: Optional[Dict]          # Saved model parameters


class NeuralNetworkDynamicsResult(TypedDict, total=False):
    """
    Neural network dynamics model result.
    
    Learned f(x, u) from data.
    
    Examples
    --------
    >>> result: NeuralNetworkDynamicsResult = learn_dynamics(
    ...     training_data, architecture
    ... )
    >>> dynamics_model = result['model']
    >>> validation_error = result['validation_mse']
    """
    model: Any                           # Trained neural network (framework-specific)
    training_mse: float                  # Training mean squared error
    validation_mse: float                # Validation MSE
    test_mse: Optional[float]            # Test MSE
    training_time: float                 # Training duration
    architecture: List[int]              # Network architecture
    n_parameters: int                    # Number of trainable parameters
    lipschitz_constant: Optional[float]  # Estimated Lipschitz constant


class GPDynamicsResult(TypedDict, total=False):
    """
    Gaussian Process dynamics model result.
    
    Probabilistic dynamics learning.
    
    Examples
    --------
    >>> result: GPDynamicsResult = learn_gp_dynamics(training_data)
    >>> mean_function = result['mean_function']
    >>> variance_function = result['variance_function']
    """
    mean_function: Callable              # Predictive mean μ(x, u)
    variance_function: Callable          # Predictive variance σ²(x, u)
    kernel: str                          # Kernel type used
    hyperparameters: Dict[str, float]    # Learned hyperparameters
    training_points: Dataset             # Training data
    log_marginal_likelihood: float       # Model evidence
    prediction_intervals: Optional[ArrayLike]  # Confidence bounds


# ============================================================================
# Cost and Objective Types
# ============================================================================

CostFunction = Callable[[StateVector, ControlVector], float]
"""
Cost/objective function: J(x, u) → ℝ.

Examples
--------
>>> def quadratic_cost(x: StateVector, u: ControlVector) -> float:
...     return x.T @ Q @ x + u.T @ R @ u
>>> 
>>> cost: CostFunction = quadratic_cost
"""

Constraint = Callable[[StateVector, ControlVector], ArrayLike]
"""
Constraint function: c(x, u) ≤ 0.

Examples
--------
>>> def input_constraint(x: StateVector, u: ControlVector) -> ArrayLike:
...     return np.abs(u) - u_max
>>> 
>>> constraint: Constraint = input_constraint
"""


# ============================================================================
# Function Types (Callables)
# ============================================================================

DynamicsFunction = Callable[[StateVector, Optional[ControlVector]], StateVector]
"""
Dynamics function.

Continuous: f(x, u, t) → dx/dt
Discrete: f(x, u) → x[k+1]

Examples
--------
>>> def dynamics(x: StateVector, u: ControlVector) -> StateVector:
...     return A @ x + B @ u
>>> 
>>> f: DynamicsFunction = dynamics
"""

OutputFunction = Callable[[StateVector], OutputVector]
"""
Output/observation function: h(x) → y.

Examples
--------
>>> def output(x: StateVector) -> OutputVector:
...     return C @ x
>>> 
>>> h: OutputFunction = output
"""

DiffusionFunction = Callable[[StateVector, Optional[ControlVector]], DiffusionMatrix]
"""
Diffusion function for stochastic systems: g(x, u, t).

Examples
--------
>>> def diffusion(x: StateVector, u: ControlVector) -> DiffusionMatrix:
...     return sigma * np.eye(nx)
>>> 
>>> g: DiffusionFunction = diffusion
"""

ControlPolicy = Callable[[StateVector], ControlVector]
"""
Control policy/controller: π(x) → u.

Examples
--------
>>> def lqr_policy(x: StateVector) -> ControlVector:
...     return -K @ x
>>> 
>>> policy: ControlPolicy = lqr_policy
"""

StateEstimator = Callable[[OutputVector], StateVector]
"""
State estimator/observer: L(y) → x_hat.

Examples
--------
>>> def kalman_estimator(y: OutputVector) -> StateVector:
...     # Kalman filter logic
...     return x_hat
>>> 
>>> estimator: StateEstimator = kalman_estimator
"""


# ============================================================================
# Callback Types
# ============================================================================

IntegrationCallback = Callable[[float, StateVector], Optional[bool]]
"""
Callback during integration.

Parameters: (t, x)
Returns: True to stop integration, False/None to continue

Examples
--------
>>> def callback(t: float, x: StateVector) -> bool:
...     if np.linalg.norm(x) > 100:
...         return True  # Stop - state too large
...     return False
"""

SimulationCallback = Callable[[int, StateVector, ControlVector], None]
"""
Callback during discrete simulation.

Parameters: (k, x[k], u[k])
Returns: None

Examples
--------
>>> def callback(k: int, x: StateVector, u: ControlVector):
...     if k % 10 == 0:
...         print(f"Step {k}: x={x}")
"""

OptimizationCallback = Callable[[ArrayLike], None]
"""
Callback during optimization.

Parameters: (x_current,)

Examples
--------
>>> def callback(x: ArrayLike):
...     print(f"Current cost: {cost_function(x):.3f}")
"""


# ============================================================================
# Protocol Definitions (Structural Subtyping)
# ============================================================================

class LinearizableProtocol(Protocol):
    """
    Protocol for systems that can be linearized.
    
    Any class with linearize() method satisfies this protocol.
    """
    
    def linearize(
        self,
        x_eq: StateVector,
        u_eq: Optional[ControlVector] = None,
        **kwargs
    ) -> LinearizationResult:
        """Compute linearization."""
        ...


class SimulatableProtocol(Protocol):
    """
    Protocol for systems that can be simulated.
    
    Any class with step() or integrate() satisfies this.
    """
    
    def step(
        self,
        x: StateVector,
        u: Optional[ControlVector] = None,
        **kwargs
    ) -> StateVector:
        """Compute next state."""
        ...


class StochasticProtocol(Protocol):
    """
    Protocol for stochastic systems.
    
    Any class with is_stochastic property and diffusion.
    """
    
    @property
    def is_stochastic(self) -> bool:
        """Whether system is stochastic."""
        ...
    
    def diffusion(
        self,
        x: StateVector,
        u: Optional[ControlVector] = None,
        **kwargs
    ) -> DiffusionMatrix:
        """Evaluate diffusion matrix."""
        ...


# ============================================================================
# Validation and Analysis Types
# ============================================================================

class ValidationResult(TypedDict, total=False):
    """
    System validation result.
    
    Examples
    --------
    >>> result: ValidationResult = validator.validate(system)
    >>> if result['valid']:
    ...     print("System is valid")
    """
    valid: bool                          # Overall validity
    errors: List[str]                    # Error messages
    warnings: List[str]                  # Warning messages
    checks_passed: int                   # Number of checks passed
    checks_total: int                    # Total number of checks


class PerformanceMetrics(TypedDict, total=False):
    """
    Performance metrics for simulation/control.
    
    Examples
    --------
    >>> metrics: PerformanceMetrics = analyze_performance(trajectory)
    >>> print(f"Settling time: {metrics['settling_time']:.2f}s")
    """
    settling_time: float                 # Time to settle within tolerance
    rise_time: float                     # 10% to 90% rise time
    overshoot: float                     # Maximum overshoot (%)
    steady_state_error: float            # Steady-state tracking error
    control_effort: float                # Integral of |u|²
    trajectory_cost: float               # Integral of cost function


# ============================================================================
# Cache and Metadata Types
# ============================================================================

CacheKey = str
"""
Cache key for memoization.

Examples
--------
>>> key: CacheKey = "x_eq=[0.0,0.0]_u_eq=[0.0]_method=euler"
"""

class CacheStatistics(TypedDict):
    """
    Cache performance statistics.
    
    Examples
    --------
    >>> stats: CacheStatistics = linearizer.get_stats()
    >>> print(f"Hit rate: {stats['hit_rate']:.1%}")
    """
    computes: int                        # Number of computations
    cache_hits: int                      # Number of cache hits
    cache_misses: int                    # Number of cache misses
    total_requests: int                  # Total requests
    cache_size: int                      # Number of cached items
    hit_rate: float                      # cache_hits / total_requests


Metadata = Dict[str, Any]
"""
General metadata dictionary.

Examples
--------
>>> metadata: Metadata = {
...     'timestamp': '2025-01-01T00:00:00',
...     'version': '1.0.0',
...     'author': 'user'
... }
"""


# ============================================================================
# Configuration Types
# ============================================================================

class SystemConfig(TypedDict, total=False):
    """
    System configuration dictionary.
    
    Examples
    --------
    >>> config: SystemConfig = system.get_config_dict()
    >>> print(config['nx'], config['nu'])
    """
    name: str                            # System name
    class_name: str                      # Class name
    nx: int                              # State dimension
    nu: int                              # Control dimension
    ny: int                              # Output dimension
    nw: int                              # Noise dimension
    is_discrete: bool                    # Discrete vs continuous
    is_stochastic: bool                  # Stochastic vs deterministic
    is_autonomous: bool                  # Autonomous vs controlled
    backend: Backend                     # Default backend
    device: Device                       # Preferred device
    parameters: SymbolDict               # Parameter values


class IntegratorConfig(TypedDict, total=False):
    """
    Integrator configuration.
    
    Examples
    --------
    >>> config: IntegratorConfig = {
    ...     'method': 'RK45',
    ...     'rtol': 1e-6,
    ...     'atol': 1e-9,
    ...     'max_step': 0.1
    ... }
    """
    method: IntegrationMethod            # Integration method
    rtol: float                          # Relative tolerance
    atol: float                          # Absolute tolerance
    max_step: float                      # Maximum step size
    first_step: Optional[float]          # Initial step size
    vectorized: bool                     # Vectorized evaluation
    dense_output: bool                   # Dense output mode


class DiscretizerConfig(TypedDict, total=False):
    """
    Discretizer configuration.
    
    Examples
    --------
    >>> config: DiscretizerConfig = {
    ...     'dt': 0.01,
    ...     'method': 'exact',
    ...     'backend': 'numpy'
    ... }
    """
    dt: float                            # Time step
    method: DiscretizationMethod         # Discretization method
    backend: Backend                     # Backend to use
    order: int                           # Approximation order
    preserve_stability: bool             # Preserve continuous stability


# ============================================================================
# Optimization Types
# ============================================================================

class OptimizationBounds(TypedDict):
    """
    Optimization variable bounds.
    
    Examples
    --------
    >>> bounds: OptimizationBounds = {
    ...     'lower': np.array([-10, -5]),
    ...     'upper': np.array([10, 5])
    ... }
    """
    lower: ArrayLike                     # Lower bounds
    upper: ArrayLike                     # Upper bounds


class OptimizationResult(TypedDict, total=False):
    """
    Optimization result.
    
    Examples
    --------
    >>> result: OptimizationResult = optimizer.solve()
    >>> optimal_u = result['x']
    >>> cost = result['fun']
    """
    x: ArrayLike                         # Optimal solution
    fun: float                           # Optimal cost
    success: bool                        # Optimization succeeded
    message: str                         # Status message
    nit: int                             # Number of iterations
    nfev: int                            # Function evaluations
    njev: int                            # Jacobian evaluations


# ============================================================================
# Utility Type Guards and Converters
# ============================================================================

def is_batched(x: ArrayLike) -> bool:
    """
    Check if array is batched (has batch dimension).
    
    Parameters
    ----------
    x : ArrayLike
        Array to check
    
    Returns
    -------
    bool
        True if x.ndim > 1 (has batch dimension)
    
    Examples
    --------
    >>> x_single = np.array([1, 2, 3])
    >>> is_batched(x_single)
    False
    >>> 
    >>> x_batch = np.array([[1, 2, 3], [4, 5, 6]])
    >>> is_batched(x_batch)
    True
    """
    if hasattr(x, 'ndim'):
        return x.ndim > 1
    elif hasattr(x, 'shape'):
        return len(x.shape) > 1
    return False


def get_batch_size(x: ArrayLike) -> Optional[int]:
    """
    Get batch size from batched array.
    
    Parameters
    ----------
    x : ArrayLike
        Array to check
    
    Returns
    -------
    Optional[int]
        Batch size if batched, None otherwise
    
    Examples
    --------
    >>> x_batch = np.random.randn(50, 3)
    >>> get_batch_size(x_batch)
    50
    >>> 
    >>> x_single = np.array([1, 2, 3])
    >>> get_batch_size(x_single)
    None
    """
    if is_batched(x):
        return x.shape[0]
    return None


def is_numpy(x: ArrayLike) -> bool:
    """Check if array is NumPy."""
    return isinstance(x, np.ndarray)


def is_torch(x: ArrayLike) -> bool:
    """Check if array is PyTorch tensor."""
    try:
        import torch
        return isinstance(x, torch.Tensor)
    except ImportError:
        return False


def is_jax(x: ArrayLike) -> bool:
    """Check if array is JAX array."""
    try:
        import jax.numpy as jnp
        return isinstance(x, jnp.ndarray)
    except ImportError:
        return False


def get_backend(x: ArrayLike) -> Backend:
    """
    Detect backend from array.
    
    Parameters
    ----------
    x : ArrayLike
        Array to check
    
    Returns
    -------
    Backend
        'numpy', 'torch', or 'jax'
    
    Raises
    ------
    TypeError
        If backend cannot be determined
    
    Examples
    --------
    >>> x_np = np.array([1, 2, 3])
    >>> get_backend(x_np)
    'numpy'
    >>> 
    >>> x_torch = torch.tensor([1, 2, 3])
    >>> get_backend(x_torch)
    'torch'
    """
    if is_numpy(x):
        return 'numpy'
    elif is_torch(x):
        return 'torch'
    elif is_jax(x):
        return 'jax'
    else:
        raise TypeError(f"Unknown backend for type {type(x)}")


def ensure_numpy(x: ArrayLike) -> np.ndarray:
    """
    Convert to NumPy array regardless of backend.
    
    Parameters
    ----------
    x : ArrayLike
        Array in any backend
    
    Returns
    -------
    np.ndarray
        NumPy array
    
    Examples
    --------
    >>> x_torch = torch.tensor([1.0, 2.0])
    >>> x_np = ensure_numpy(x_torch)
    >>> type(x_np)
    <class 'numpy.ndarray'>
    """
    if isinstance(x, np.ndarray):
        return x
    
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except ImportError:
        pass
    
    try:
        import jax.numpy as jnp
        if isinstance(x, jnp.ndarray):
            return np.array(x)
    except ImportError:
        pass
    
    # Fallback
    return np.asarray(x)


def ensure_backend(x: ArrayLike, backend: Backend) -> ArrayLike:
    """
    Convert array to specified backend.
    
    Parameters
    ----------
    x : ArrayLike
        Input array
    backend : Backend
        Target backend
    
    Returns
    -------
    ArrayLike
        Array in target backend
    
    Examples
    --------
    >>> x_np = np.array([1, 2, 3])
    >>> x_torch = ensure_backend(x_np, 'torch')
    >>> type(x_torch)
    <class 'torch.Tensor'>
    """
    current_backend = get_backend(x)
    
    if current_backend == backend:
        return x
    
    # Convert to numpy first (common intermediate)
    x_np = ensure_numpy(x)
    
    # Convert to target backend
    if backend == 'numpy':
        return x_np
    elif backend == 'torch':
        import torch
        return torch.tensor(x_np)
    elif backend == 'jax':
        import jax.numpy as jnp
        return jnp.array(x_np)
    else:
        raise ValueError(f"Unknown backend: {backend}")


def check_state_shape(x: StateVector, nx: int, name: str = "state"):
    """
    Validate state vector shape.
    
    Parameters
    ----------
    x : StateVector
        State to validate
    nx : int
        Expected state dimension
    name : str
        Parameter name for error messages
    
    Raises
    ------
    ValueError
        If shape is incorrect
    
    Examples
    --------
    >>> x = np.array([1, 2, 3])
    >>> check_state_shape(x, nx=3, name="initial_state")  # OK
    >>> check_state_shape(x, nx=2, name="state")  # ValueError
    """
    x_arr = ensure_numpy(x)
    
    if is_batched(x_arr):
        # Batched: (batch, nx)
        if x_arr.shape[1] != nx:
            raise ValueError(
                f"{name} has incorrect dimension. "
                f"Expected (..., {nx}), got shape {x_arr.shape}"
            )
    else:
        # Single: (nx,)
        if x_arr.shape[0] != nx:
            raise ValueError(
                f"{name} has incorrect dimension. "
                f"Expected ({nx},), got shape {x_arr.shape}"
            )


def check_control_shape(u: ControlVector, nu: int, name: str = "control"):
    """
    Validate control vector shape.
    
    Similar to check_state_shape but for control inputs.
    """
    u_arr = ensure_numpy(u)
    
    if is_batched(u_arr):
        if u_arr.shape[1] != nu:
            raise ValueError(
                f"{name} has incorrect dimension. "
                f"Expected (..., {nu}), got shape {u_arr.shape}"
            )
    else:
        if u_arr.shape[0] != nu:
            raise ValueError(
                f"{name} has incorrect dimension. "
                f"Expected ({nu},), got shape {u_arr.shape}"
            )


def get_array_shape(x: ArrayLike) -> Tuple[int, ...]:
    """
    Get shape of array regardless of backend.
    
    Examples
    --------
    >>> x = np.random.randn(10, 3)
    >>> get_array_shape(x)
    (10, 3)
    """
    if hasattr(x, 'shape'):
        return tuple(x.shape)
    else:
        return ()


def extract_dimensions(
    x: Optional[StateVector] = None,
    u: Optional[ControlVector] = None,
    y: Optional[OutputVector] = None,
) -> SystemDimensions:
    """
    Extract system dimensions from example vectors.
    
    Parameters
    ----------
    x : Optional[StateVector]
        Example state vector
    u : Optional[ControlVector]
        Example control vector
    y : Optional[OutputVector]
        Example output vector
    
    Returns
    -------
    SystemDimensions
        Dimensions extracted from vectors
    
    Examples
    --------
    >>> x = np.array([1, 2, 3])
    >>> u = np.array([0.5])
    >>> dims = extract_dimensions(x, u)
    >>> print(dims)
    SystemDimensions(nx=3, nu=1, ny=3, nw=0)
    """
    nx = x.shape[-1] if x is not None else 0
    nu = u.shape[-1] if u is not None else 0
    ny = y.shape[-1] if y is not None else nx
    
    return SystemDimensions(nx=nx, nu=nu, ny=ny, nw=0)


# ============================================================================
# Type Conversion Utilities
# ============================================================================

class ArrayConverter:
    """
    Utility class for array conversions.
    
    Provides methods for converting between backends efficiently.
    
    Examples
    --------
    >>> converter = ArrayConverter()
    >>> x_torch = converter.to_torch(x_numpy)
    >>> x_jax = converter.to_jax(x_numpy)
    """
    
    @staticmethod
    def to_numpy(x: ArrayLike) -> np.ndarray:
        """Convert to NumPy."""
        return ensure_numpy(x)
    
    @staticmethod
    def to_torch(x: ArrayLike, device: Optional[Device] = None) -> 'torch.Tensor':
        """
        Convert to PyTorch tensor.
        
        Parameters
        ----------
        x : ArrayLike
            Input array
        device : Optional[Device]
            Target device ('cpu', 'cuda', etc.)
        
        Returns
        -------
        torch.Tensor
            PyTorch tensor
        """
        import torch
        x_np = ensure_numpy(x)
        tensor = torch.tensor(x_np)
        if device is not None:
            tensor = tensor.to(device)
        return tensor
    
    @staticmethod
    def to_jax(x: ArrayLike) -> 'jnp.ndarray':
        """Convert to JAX array."""
        import jax.numpy as jnp
        x_np = ensure_numpy(x)
        return jnp.array(x_np)
    
    @staticmethod
    def convert(x: ArrayLike, target_backend: Backend) -> ArrayLike:
        """Convert to specified backend."""
        return ensure_backend(x, target_backend)


# ============================================================================
# Generic Type Variables
# ============================================================================

T = TypeVar('T', bound=ArrayLike)
"""Generic array type variable."""

S = TypeVar('S', bound='DiscreteSystemBase')
"""Generic discrete system type variable."""

C = TypeVar('C', bound='ContinuousSystemBase')
"""Generic continuous system type variable."""

MatrixT = TypeVar('MatrixT', StateMatrix, ControlMatrix, DiffusionMatrix)
"""Generic matrix type variable."""


# ============================================================================
# Enumerations and Constants
# ============================================================================

VALID_BACKENDS = ('numpy', 'torch', 'jax')
"""Tuple of valid backend names."""

VALID_DEVICES = ('cpu', 'cuda', 'mps')
"""Common device identifiers (extend as needed)."""

DEFAULT_BACKEND: Backend = 'numpy'
"""Default backend if not specified."""

DEFAULT_DEVICE: Device = 'cpu'
"""Default device if not specified."""

DEFAULT_DTYPE = np.float64
"""Default numerical precision."""


# ============================================================================
# Export All Public Types
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
    'ControlMatrix',
    'OutputMatrix',
    'DiffusionMatrix',
    'FeedthroughMatrix',
    'CovarianceMatrix',
    'GainMatrix',
    'ControllabilityMatrix',
    'ObservabilityMatrix',
    'CostMatrix',
    
    # Linearization
    'DeterministicLinearization',
    'StochasticLinearization',
    'LinearizationResult',
    'ObservationLinearization',
    
    # Trajectories
    'StateTrajectory',
    'ControlSequence',
    'OutputSequence',
    'NoiseSequence',
    'TimePoints',
    'TimeSpan',
    
    # Symbolic
    'SymbolicExpression',
    'SymbolicMatrix',
    'SymbolicSymbol',
    'SymbolDict',
    'SymbolicStateEquations',
    'SymbolicOutputEquations',
    'SymbolicDiffusionMatrix',
    
    # Dimensions
    'SystemDimensions',
    'DimensionTuple',
    
    # Equilibria
    'EquilibriumState',
    'EquilibriumControl',
    'EquilibriumPoint',
    'EquilibriumName',
    'EquilibriumIdentifier',
    
    # Backend
    'Backend',
    'Device',
    'BackendConfig',
    
    # Methods
    'IntegrationMethod',
    'DiscretizationMethod',
    'SDEIntegrationMethod',
    'OptimizationMethod',
    
    # Analysis
    'StabilityInfo',
    'ControllabilityInfo',
    'ObservabilityInfo',
    'PerformanceMetrics',
    
    # Results
    'IntegrationResult',
    'SimulationResult',
    'LQRResult',
    'KalmanFilterResult',
    'LQGResult',
    'MPCResult',
    'MHEResult',
    'EKFResult',
    'UKFResult',
    'ParticleFilterResult',
    'OptimizationResult',
    'ValidationResult',
    
    # Advanced control
    'H2ControlResult',
    'HInfControlResult',
    'LMIResult',
    'AdaptiveControlResult',
    'SlidingModeResult',
    
    # Robust control
    'UncertaintySet',
    'RobustStabilityResult',
    'StructuredSingularValueResult',
    'TubeDefinition',
    'TubeMPCResult',
    
    # Stochastic control
    'StochasticMPCResult',
    'RiskSensitiveResult',
    
    # Model reduction
    'BalancedRealizationResult',
    'ReducedOrderModelResult',
    
    # System identification
    'HankelMatrix',
    'ToeplitzMatrix',
    'TrajectoryMatrix',
    'MarkovParameters',
    'CorrelationMatrix',
    'SystemIDResult',
    'SubspaceIDResult',
    'ERAResult',
    'DMDResult',
    'SINDyResult',
    'KoopmanResult',
    
    # Reachability and safety
    'ReachableSet',
    'SafeSet',
    'ReachabilityResult',
    'ROAResult',
    'VerificationResult',
    'BarrierCertificateResult',
    'CBFResult',
    'CLFResult',
    
    # HJI reachability
    'ValueFunction',
    'LevelSet',
    'HJIReachabilityResult',
    'BackwardReachableResult',
    'ForwardReachableResult',
    'ViabilityKernelResult',
    'DifferentialGameResult',
    
    # Contraction analysis
    'ContractionMetric',
    'ContractionRate',
    'ContractionAnalysisResult',
    'CCMResult',
    'FunnelingResult',
    'IncrementalStabilityResult',
    
    # Differential flatness
    'FlatnessOutput',
    'DifferentialFlatnessResult',
    'TrajectoryPlanningResult',
    
    # Conformal prediction
    'ConformalPredictionSet',
    'NonconformityScore',
    'ConformalCalibrationResult',
    'ConformalPredictionResult',
    'AdaptiveConformalResult',
    
    # Learning
    'Dataset',
    'TrainingBatch',
    'LearningRate',
    'LossValue',
    'NeuralNetworkConfig',
    'TrainingResult',
    'NeuralNetworkDynamicsResult',
    'GPDynamicsResult',
    
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
    
    # Configs
    'SystemConfig',
    'IntegratorConfig',
    'DiscretizerConfig',
    
    # Optimization
    'OptimizationBounds',
    
    # Cache
    'CacheKey',
    'CacheStatistics',
    'Metadata',
    
    # Noise
    'NoiseType',
    'SDEType',
    'ConvergenceType',
    
    # Protocols
    'LinearizableProtocol',
    'SimulatableProtocol',
    'StochasticProtocol',
    
    # Utilities
    'is_batched',
    'get_batch_size',
    'is_numpy',
    'is_torch',
    'is_jax',
    'get_backend',
    'ensure_numpy',
    'ensure_backend',
    'check_state_shape',
    'check_control_shape',
    'get_array_shape',
    'extract_dimensions',
    'ArrayConverter',
    
    # Type variables
    'T',
    'S',
    'C',
    'MatrixT',
    
    # Constants
    'VALID_BACKENDS',
    'VALID_DEVICES',
    'DEFAULT_BACKEND',
    'DEFAULT_DEVICE',
    'DEFAULT_DTYPE',
]