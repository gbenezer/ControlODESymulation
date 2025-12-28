# Type System Reference

> **ControlDESymulation Type System Documentation**
>
> A comprehensive reference for the modular type system powering the ControlDESymulation framework.

---

## Overview

The `src/types` directory contains a comprehensive, modular type system with:

- **225+ type definitions** for arrays, vectors, matrices, and functions
- **55+ TypedDict result types** for structured return values
- **15+ utility functions** for type guards, converters, and validators
- **8+ Protocol definitions** for structural subtyping
- **5 constant tuples** for method groupings
- **19 Python modules** organized by domain-specific functionality

**Complete Documentation**: All 19 modules fully documented with examples and mathematical background.

---

## Table of Contents

1. [Directory Structure](#directory-structure)
2. [Core Types](#1-core-types-corepy)
3. [Backend Configuration](#2-backend-configuration-backendspy)
4. [Symbolic Types](#3-symbolic-types-symbolicpy)
5. [Trajectory Types](#4-trajectory-types-trajectoriespy)
6. [Linearization Types](#5-linearization-types-linearizationpy)
7. [Classical Control Types](#6-classical-control-types-control_classicalpy)
8. [Advanced Control Types](#7-advanced-control-types-control_advancedpy)
9. [Estimation Types](#8-estimation-types-estimationpy)
10. [System Identification Types](#9-system-identification-types-identificationpy)
11. [Reachability & Safety Types](#10-reachability--safety-types-reachabilitypy)
12. [Robustness Types](#11-robustness-types-robustnesspy)
13. [Optimization Types](#12-optimization-types-optimizationpy)
14. [Learning Types](#13-learning-types-learningpy)
15. [Utility Types & Functions](#14-utility-types--functions-utilitiespy)
16. [Contraction Theory Types](#15-contraction-theory-types-contractionpy)
17. [Conformal Prediction Types](#16-conformal-prediction-types-conformalpy)
18. [Differential Flatness Types](#17-differential-flatness-types-flatnesspy)
19. [Model Reduction Types](#18-model-reduction-types-model_reductionpy)
20. [Design Principles](#design-principles)
21. [Usage Patterns](#usage-patterns)
22. [Integration Pipelines](#integration-pipelines)

---

## Directory Structure

| File | Purpose | Key Exports |
|------|---------|-------------|
| `core.py` | Fundamental types | `StateVector`, `ControlVector`, `DynamicsFunction` |
| `backends.py` | Backend/method selection | `Backend`, `Device`, `IntegrationMethod` |
| `symbolic.py` | SymPy symbolic math | `SymbolicExpression`, `SymbolicMatrix` |
| `trajectories.py` | Time series data | `StateTrajectory`, `SimulationResult`, `DiscreteSimulationResult` |
| `linearization.py` | Linearization results | `LinearizationResult`, `StateJacobian` |
| `control_classical.py` | LQR, LQG, Kalman | `LQRResult`, `KalmanFilterResult` |
| `control_advanced.py` | MPC, H2/H-infinity | `MPCResult`, `HInfControlResult` |
| `estimation.py` | State estimation | `EKFResult`, `UKFResult`, `ParticleFilterResult` |
| `identification.py` | System ID methods | `DMDResult`, `SINDyResult`, `KoopmanResult` |
| `reachability.py` | Safety analysis | `ReachabilityResult`, `CBFResult` |
| `robustness.py` | Robust control | `RobustStabilityResult`, `TubeMPCResult` |
| `optimization.py` | Optimization results | `OptimizationResult`, `TrajectoryOptimizationResult` |
| `learning.py` | Neural nets, RL | `TrainingResult`, `NeuralDynamicsResult` |
| `utilities.py` | Guards, converters | `is_numpy()`, `ensure_backend()` |
| `contraction.py` | Contraction analysis | Contraction metric types |
| `conformal.py` | Conformal prediction | Prediction interval types |
| `flatness.py` | Differential flatness | Flat output types |
| `model_reduction.py` | Model reduction | Reduced-order model types |
| `__init__.py` | Central imports | All public types |

---

## 1. Core Types (`core.py`)

### Array Types (Multi-Backend Support)

```python
ArrayLike = Union[np.ndarray, torch.Tensor, jax.Array]
NumpyArray = np.ndarray
TorchTensor = torch.Tensor
JaxArray = jax.Array
ScalarLike = Union[float, int, np.floating, torch.Tensor, jax.Array]
IntegerLike = Union[int, np.integer]
```

### Vector Types (Semantic Naming)

| Type | Description | Typical Shape |
|------|-------------|---------------|
| `StateVector` | System state x in R^nx | `(nx,)` or `(batch, nx)` |
| `ControlVector` | Control input u in R^nu | `(nu,)` or `(batch, nu)` |
| `OutputVector` | Measurements y in R^ny | `(ny,)` or `(batch, ny)` |
| `NoiseVector` | Disturbances w in R^nw | `(nw,)` or `(batch, nw)` |
| `ParameterVector` | Parameters theta in R^np | `(np,)` |
| `ResidualVector` | Prediction errors | `(n,)` |

### Matrix Types (Semantic Naming)

| Type | Description | Shape | Common Uses |
|------|-------------|-------|-------------|
| `StateMatrix` | State coupling | `(nx, nx)` | A, Ad, P matrices |
| `InputMatrix` | Control mapping | `(nx, nu)` | B, Bd matrices |
| `OutputMatrix` | Output mapping | `(ny, nx)` | C matrix |
| `DiffusionMatrix` | Noise gain | `(nx, nw)` | G matrix |
| `FeedthroughMatrix` | Direct feedthrough | `(ny, nu)` | D matrix |
| `CovarianceMatrix` | Symmetric PSD | `(n, n)` | P, Q, R, Sigma |
| `GainMatrix` | Control/observer gains | varies | K, L matrices |
| `ControllabilityMatrix` | Controllability | `(nx, nx*nu)` | [B AB ... A^(n-1)B] |
| `ObservabilityMatrix` | Observability | `(nx*ny, nx)` | [C; CA; ...] |
| `CostMatrix` | Quadratic weights | `(n, n)` | Q, R cost matrices |

### Dimension Types

```python
class SystemDimensions(TypedDict):
    nx: int  # State dimension
    nu: int  # Control dimension
    ny: int  # Output dimension
    nw: int  # Noise dimension (optional)
    np: int  # Parameter dimension (optional)

DimensionTuple = Tuple[int, int, int]  # (nx, nu, ny)
```

### Equilibrium Types

```python
EquilibriumState = StateVector      # x_eq
EquilibriumControl = ControlVector  # u_eq
EquilibriumPoint = Tuple[EquilibriumState, EquilibriumControl]
EquilibriumName = str               # 'origin', 'hover', etc.
EquilibriumIdentifier = Union[EquilibriumName, EquilibriumState]
```

### Function Types (Callable Signatures)

```python
DynamicsFunction = Callable[[StateVector, ControlVector], StateVector]
# f(x, u) -> dx/dt or x[k+1]

OutputFunction = Callable[[StateVector], OutputVector]
# h(x) -> y

DiffusionFunction = Callable[[StateVector, ControlVector], DiffusionMatrix]
# g(x, u) -> noise scaling matrix

ControlPolicy = Callable[[StateVector], ControlVector]
# pi(x) -> u (feedback control law)

TimeVaryingControl = Callable[[float], ControlVector]
# u(t) -> time-varying control (continuous)

FeedbackController = Callable[[StateVector, float], ControlVector]
# pi(x, t) -> time-varying state feedback

ControlInput = Union[ControlVector, TimeVaryingControl, None]
# Flexible control input for integration

TimeIndexedControl = Callable[[int], ControlVector]
# u[k] -> time-indexed control (discrete)

DiscreteFeedbackPolicy = Callable[[StateVector, int], ControlVector]
# pi(x[k], k) -> discrete-time state feedback

DiscreteControlInput = Union[ControlVector, Sequence[ControlVector], TimeIndexedControl, None]
# Flexible control input for discrete simulation

StateEstimator = Callable[[OutputVector], StateVector]
# L(y) -> x_hat

CostFunction = Callable[[StateVector, ControlVector], float]
# J(x, u) -> scalar cost

Constraint = Callable[[StateVector, ControlVector], ArrayLike]
# c(x, u) -> constraint values (c <= 0 for feasibility)
```

### Callback Types

```python
IntegrationCallback = Callable[[float, StateVector], None]
# Called during continuous integration: callback(t, x)

SimulationCallback = Callable[[int, StateVector, ControlVector], None]
# Called at each discrete step: callback(k, x, u)

OptimizationCallback = Callable[[int, ArrayLike, float], None]
# Called at each iteration: callback(iter, x, cost)
```

### Type Variables

```python
T = TypeVar('T', bound=ArrayLike)    # Generic array (preserves backend)
S = TypeVar('S')                      # Generic discrete system
C = TypeVar('C')                      # Generic continuous system
MatrixT = TypeVar('MatrixT')          # Generic matrix type
```

---

## 2. Backend Configuration (`backends.py`)

### Backend and Device Types

```python
Backend = Literal["numpy", "torch", "jax"]
Device = str  # "cpu", "cuda", "cuda:0", "mps", "tpu"

VALID_BACKENDS = ("numpy", "torch", "jax")
VALID_DEVICES = ("cpu", "cuda", "mps", "tpu")
DEFAULT_BACKEND = "numpy"
DEFAULT_DEVICE = "cpu"
DEFAULT_DTYPE = np.float64
```

### Configuration TypedDicts

```python
class BackendConfig(TypedDict, total=False):
    backend: Backend
    device: Device
    dtype: type

class SystemConfig(TypedDict, total=False):
    backend: Backend
    device: Device
    dtype: type
    # Additional system metadata

class IntegratorConfig(TypedDict, total=False):
    method: IntegrationMethod
    rtol: float
    atol: float
    max_step: float
    # Continuous integration settings

class DiscretizerConfig(TypedDict, total=False):
    method: DiscretizationMethod
    dt: float
    # Discretization settings

class SDEIntegratorConfig(TypedDict, total=False):
    method: SDEIntegrationMethod
    noise_type: NoiseType
    sde_type: SDEType
    # Stochastic integration settings
```

### Integration, Discretization, and Optimization Methods

```python
IntegrationMethod = str
DiscretizationMethod = str
SDEIntegrationMethod = str
OptimizationMethod = str
```

### Stochastic Types

```python
NoiseType = Literal[
    "additive",        # g(x,u) = G (constant)
    "multiplicative",  # g(x,u) = G(x,u)
    "diagonal",        # Diagonal noise covariance
    "scalar",          # Single noise source
    "general"          # Full noise structure
]

SDEType = Literal["ito", "stratonovich"]
ConvergenceType = Literal["strong", "weak"]
```

### Utility Functions

```python
def get_backend_default_method(backend: Backend) -> IntegrationMethod:
    """Get default integration method for backend."""

def validate_backend(backend: str) -> Backend:
    """Validate and return canonical backend name."""

def validate_device(device: str) -> Device:
    """Validate device string."""
```

---

## 3. Symbolic Types (`symbolic.py`)

### Basic Symbolic Types

```python
SymbolicExpression = sp.Expr       # Single symbolic expression
SymbolicMatrix = sp.Matrix         # Matrix of expressions
SymbolicSymbol = sp.Symbol         # Single symbolic variable
SymbolDict = Dict[str, sp.Symbol]  # Name -> symbol mapping
SymbolicExpressionInput = Union[sp.Expr, sp.Matrix, List[sp.Expr]]  # Flexible input
```

### System Equation Types

```python
SymbolicStateEquations = sp.Matrix
# f(x, u) as column vector [f1; f2; ...; fn]

SymbolicOutputEquations = sp.Matrix
# h(x, u) as column vector [h1; h2; ...; hm]

SymbolicDiffusionMatrix = sp.Matrix
# g(x, u) for SDE noise scaling
```

### Parameter and Substitution Types

```python
ParameterDict = Dict[sp.Symbol, float]
# {m: 1.0, g: 9.81, L: 0.5} for numeric evaluation

SubstitutionDict = Dict[sp.Symbol, Union[float, sp.Expr]]
# General substitution (can substitute expressions)
```

### Derivative Types

```python
SymbolicJacobian = sp.Matrix
# Jacobian matrix: df_i/dx_j

SymbolicGradient = sp.Matrix
# Gradient vector: [df/dx1, df/dx2, ...]

SymbolicHessian = sp.Matrix
# Hessian matrix: d^2f/(dx_i dx_j)
```

---

## 4. Trajectory Types (`trajectories.py`)

### Sequence Types

| Type | Shape | Description |
|------|-------|-------------|
| `StateTrajectory` | `(n_steps, nx)` | Time series of states |
| `ControlSequence` | `(n_steps, nu)` | Control inputs over time |
| `OutputSequence` | `(n_steps, ny)` | Measurements over time |
| `NoiseSequence` | `(n_steps, nw)` | Noise realizations |

### Time Types

```python
TimePoints = ArrayLike  # Shape: (n_points,) array of time instants
TimeSpan = Tuple[float, float]  # (t_start, t_end)
```

### Result TypedDicts

```python
class IntegrationResult(TypedDict, total=False):
    t: TimePoints           # Time points
    y: StateTrajectory      # State trajectory
    success: bool           # Integration succeeded
    message: str            # Status message
    nfev: int               # Function evaluations
    njev: int               # Jacobian evaluations
    nlu: int                # LU decompositions
    status: int             # Termination status code

class SimulationResult(TypedDict, total=False):
    states: StateTrajectory           # (n_steps+1, nx)
    controls: Optional[ControlSequence]  # (n_steps, nu)
    outputs: Optional[OutputSequence]    # (n_steps+1, ny)
    noise: Optional[NoiseSequence]       # (n_steps, nw)
    time: Optional[TimePoints]           # (n_steps+1,)
    metadata: Dict[str, Any]          # Additional information

class DiscreteSimulationResult(TypedDict, total=False):
    states: StateTrajectory           # (n_steps+1, nx) - includes x[0]
    controls: Optional[ControlSequence]  # (n_steps, nu)
    outputs: Optional[OutputSequence]    # (n_steps+1, ny)
    noise: Optional[NoiseSequence]       # (n_steps, nw)
    time_steps: ArrayLike             # [0, 1, 2, ..., n_steps]
    dt: float                         # Sampling period
    metadata: Dict[str, Any]          # Additional information

class TrajectoryStatistics(TypedDict, total=False):
    mean: ArrayLike       # Mean value
    std: ArrayLike        # Standard deviation
    min: ArrayLike        # Minimum value
    max: ArrayLike        # Maximum value
    initial: ArrayLike    # Initial state
    final: ArrayLike      # Final state
    length: int           # Number of points
    duration: float       # Total time span

class TrajectorySegment(TypedDict):
    states: ArrayLike           # State trajectory segment
    controls: Optional[ArrayLike]  # Control sequence segment
    time: ArrayLike             # Time points for segment
    start_index: int
    end_index: int
```

---

## 5. Linearization Types (`linearization.py`)

### Main Linearization Types

```python
DeterministicLinearization = Tuple[StateMatrix, InputMatrix]
# (A, B) for deterministic systems: dx/dt = Ax + Bu

StochasticLinearization = Tuple[StateMatrix, InputMatrix, DiffusionMatrix]
# (A, B, G) for stochastic systems: dx = (Ax + Bu)dt + G dW

LinearizationResult = Union[DeterministicLinearization, StochasticLinearization]
# Polymorphic: check len(result) to determine type

ObservationLinearization = Tuple[OutputMatrix, FeedthroughMatrix]
# (C, D) for output equation: y = Cx + Du
```

### Time-Domain Aliases

```python
ContinuousLinearization = DeterministicLinearization      # (Ac, Bc)
DiscreteLinearization = DeterministicLinearization        # (Ad, Bd)
ContinuousStochasticLinearization = StochasticLinearization  # (Ac, Bc, Gc)
DiscreteStochasticLinearization = StochasticLinearization    # (Ad, Bd, Gd)
```

### Complete Linearization

```python
FullLinearization = Tuple[StateMatrix, InputMatrix, OutputMatrix, FeedthroughMatrix]
# (A, B, C, D) complete state-space representation

FullStochasticLinearization = Tuple[
    StateMatrix, InputMatrix, DiffusionMatrix, OutputMatrix, FeedthroughMatrix
]
# (A, B, G, C, D) complete stochastic state-space
```

### Jacobian Types

```python
StateJacobian = StateMatrix      # df/dx: state sensitivity
ControlJacobian = InputMatrix    # df/du: control sensitivity
OutputJacobian = OutputMatrix    # dh/dx: output sensitivity
DiffusionJacobian = DiffusionMatrix  # dg/dx or g(x) itself

LinearizationCacheKey = str
# Format: "x0_hash:u0_hash" for caching linearizations
```

---

## 6. Classical Control Types (`control_classical.py`)

### Analysis Types

```python
class StabilityInfo(TypedDict):
    eigenvalues: ArrayLike        # System eigenvalues
    magnitudes: ArrayLike         # |lambda_i| (discrete) or Re(lambda_i) (continuous)
    is_stable: bool               # All eigenvalues in stable region
    stability_margin: float       # Distance to instability boundary
    dominant_eigenvalue: complex  # Slowest/least damped mode
    damping_ratios: ArrayLike     # Damping for complex pairs
    natural_frequencies: ArrayLike  # Undamped natural frequencies

class ControllabilityInfo(TypedDict):
    controllability_matrix: ControllabilityMatrix  # [B AB A^2B ...]
    rank: int                     # Rank of controllability matrix
    is_controllable: bool         # rank == nx
    controllable_modes: ArrayLike # Eigenvalues of controllable subspace
    uncontrollable_modes: ArrayLike  # Eigenvalues outside controllable subspace

class ObservabilityInfo(TypedDict):
    observability_matrix: ObservabilityMatrix  # [C; CA; CA^2; ...]
    rank: int                     # Rank of observability matrix
    is_observable: bool           # rank == nx
    observable_modes: ArrayLike   # Eigenvalues of observable subspace
    unobservable_modes: ArrayLike # Hidden modes
```

### Control Design Results

```python
class LQRResult(TypedDict):
    K: GainMatrix                 # Optimal feedback gain: u = -Kx
    P: CovarianceMatrix           # Cost-to-go matrix (Riccati solution)
    eigenvalues: ArrayLike        # Closed-loop eigenvalues
    gain_margin: float            # Gain margin (dB)
    phase_margin: float           # Phase margin (degrees)
    cost: float                   # Optimal cost J* = x0' P x0

class KalmanFilterResult(TypedDict):
    L: GainMatrix                 # Kalman gain: x_hat += L(y - C*x_hat)
    P: CovarianceMatrix           # Steady-state error covariance
    innovation_covariance: CovarianceMatrix  # E[(y - y_hat)(y - y_hat)']
    eigenvalues: ArrayLike        # Estimator eigenvalues

class LQGResult(TypedDict):
    control_gain: GainMatrix      # LQR gain K
    estimator_gain: GainMatrix    # Kalman gain L
    control_riccati: CovarianceMatrix  # LQR cost-to-go P
    estimator_riccati: CovarianceMatrix  # Kalman error covariance
    closed_loop_eigenvalues: ArrayLike  # Combined system eigenvalues

class PolePlacementResult(TypedDict):
    K: GainMatrix                 # Feedback gain
    desired_poles: ArrayLike      # Requested pole locations
    achieved_poles: ArrayLike     # Actual closed-loop poles
    conditioning: float           # Numerical conditioning

class LuenbergerObserverResult(TypedDict):
    L: GainMatrix                 # Observer gain
    desired_poles: ArrayLike      # Requested estimator poles
    achieved_poles: ArrayLike     # Actual estimator poles
```

---

## 7. Advanced Control Types (`control_advanced.py`)

### Predictive Control

```python
class MPCResult(TypedDict):
    control_sequence: ControlSequence  # Optimal u[0:N]
    predicted_trajectory: StateTrajectory  # Predicted x[0:N+1]
    cost: float                    # Optimal cost
    success: bool                  # Solver converged
    iterations: int                # Solver iterations
    solve_time: float              # Computation time (seconds)
    active_constraints: List[str]  # Binding constraints

class MHEResult(TypedDict):
    state_estimate: StateVector    # Current state estimate
    state_trajectory: StateTrajectory  # Estimated x[t-N:t]
    parameter_estimate: ParameterVector  # Estimated parameters (optional)
    cost: float                    # Estimation cost
    success: bool                  # Solver converged
```

### Robust Optimal Control

```python
class H2ControlResult(TypedDict):
    K: GainMatrix                  # H2 optimal controller gain
    P: CovarianceMatrix            # Riccati solution
    h2_norm: float                 # ||T_zw||_2 closed-loop norm
    eigenvalues: ArrayLike         # Closed-loop poles

class HInfControlResult(TypedDict):
    K: GainMatrix                  # H-infinity controller gain
    gamma: float                   # Achieved performance level
    P: CovarianceMatrix            # Riccati solution
    eigenvalues: ArrayLike         # Closed-loop poles
    iterations: int                # Gamma iteration count

class LMIResult(TypedDict):
    decision_variables: Dict[str, ArrayLike]  # Solved variables
    objective_value: float         # Optimal objective
    feasible: bool                 # Problem feasible
    solver: str                    # Solver used (CVXPY, etc.)
    solve_time: float              # Computation time
```

### Adaptive and Nonlinear Control

```python
class AdaptiveControlResult(TypedDict):
    current_gain: GainMatrix       # Current adaptive gain
    parameter_estimate: ParameterVector  # Estimated parameters
    covariance: CovarianceMatrix   # Parameter estimation covariance
    tracking_error: float          # Current tracking error norm
    adaptation_rate: float         # Learning rate

class SlidingModeResult(TypedDict):
    control: ControlVector         # Sliding mode control input
    sliding_variable: float        # s(x) value
    on_sliding_surface: bool       # |s| < epsilon
    chattering_magnitude: float    # High-frequency oscillation measure
    equivalent_control: ControlVector  # Continuous approximation
```

---

## 8. Estimation Types (`estimation.py`)

### Extended Kalman Filter

```python
class EKFResult(TypedDict):
    state_estimate: StateVector    # x_hat
    covariance: CovarianceMatrix   # P (estimation error covariance)
    innovation: OutputVector       # y - h(x_hat) (measurement residual)
    innovation_covariance: CovarianceMatrix  # S = HPH' + R
    kalman_gain: GainMatrix        # K = PH'S^{-1}
    likelihood: float              # p(y|x_hat) for consistency check
```

### Unscented Kalman Filter

```python
class UKFResult(TypedDict):
    state_estimate: StateVector    # x_hat (weighted mean of sigma points)
    covariance: CovarianceMatrix   # P
    sigma_points: ArrayLike        # (2*nx+1, nx) sigma point set
    weights_mean: ArrayLike        # Weights for mean calculation
    weights_covariance: ArrayLike  # Weights for covariance calculation
    innovation: OutputVector       # y - y_hat
    cross_covariance: ArrayLike    # P_xy for gain calculation
```

### Particle Filter

```python
class ParticleFilterResult(TypedDict):
    state_estimate: StateVector    # Weighted particle mean
    covariance: CovarianceMatrix   # Weighted particle covariance
    particles: ArrayLike           # (n_particles, nx) particle set
    weights: ArrayLike             # (n_particles,) normalized weights
    effective_sample_size: float   # ESS = 1/sum(w^2), indicates degeneracy
    resampled: bool                # Whether resampling was triggered
    log_likelihood: float          # log p(y|particles)
```

---

## 9. System Identification Types (`identification.py`)

### Data Matrix Types

```python
HankelMatrix = ArrayLike
# Block Hankel matrix from time series data
# H = [y[0]   y[1]   ... y[j]  ]
#     [y[1]   y[2]   ... y[j+1]]
#     [...]

ToeplitzMatrix = ArrayLike
# Constant along diagonals

TrajectoryMatrix = ArrayLike
# Data snapshot matrix for DMD/SINDy
# X = [x[0] x[1] ... x[m-1]]
# Y = [x[1] x[2] ... x[m]]

MarkovParameters = ArrayLike
# Impulse response sequence: H[k] = C A^{k-1} B
```

### Result Types

```python
class SystemIDResult(TypedDict):
    A: StateMatrix                 # Identified state matrix
    B: InputMatrix                 # Identified input matrix
    C: OutputMatrix                # Identified output matrix
    D: FeedthroughMatrix           # Identified feedthrough
    fit_percentage: float          # Model fit (0-100%)
    residuals: ArrayLike           # Prediction errors
    method: str                    # Identification method used
    singular_values: ArrayLike     # SVD singular values (model order)
    order: int                     # Selected model order

class SubspaceIDResult(SystemIDResult):
    observability_matrix: ObservabilityMatrix
    controllability_matrix: ControllabilityMatrix
    hankel_singular_values: ArrayLike

class ERAResult(TypedDict):
    A: StateMatrix
    B: InputMatrix
    C: OutputMatrix
    D: FeedthroughMatrix
    hankel_singular_values: ArrayLike  # For model order selection
    markov_parameters: MarkovParameters
    fit_percentage: float

class DMDResult(TypedDict):
    A: StateMatrix                 # DMD matrix (or Koopman approx)
    modes: ArrayLike               # DMD modes (eigenvectors)
    eigenvalues: ArrayLike         # DMD eigenvalues
    amplitudes: ArrayLike          # Mode amplitudes
    frequencies: ArrayLike         # Oscillation frequencies (Hz)
    growth_rates: ArrayLike        # Exponential growth/decay rates
    reconstruction_error: float    # ||X' - A*X||

class SINDyResult(TypedDict):
    coefficients: ArrayLike        # Sparse coefficient matrix
    active_terms: List[str]        # Non-zero basis function names
    basis_functions: List[Callable]  # Library of candidate functions
    sparsity_level: float          # Fraction of zero coefficients
    reconstruction_error: float    # Training error
    cross_validation_error: float  # Held-out error

class KoopmanResult(TypedDict):
    K: ArrayLike                   # Koopman operator (lifted space)
    eigenfunctions: List[Callable] # Koopman eigenfunctions
    eigenvalues: ArrayLike         # Koopman eigenvalues
    lifting_function: Callable     # g: x -> lifted coordinates
    projection: Callable           # P: lifted -> original
    reconstruction_error: float
```

---

## 10. Reachability & Safety Types (`reachability.py`)

### Set Representation Types

```python
ReachableSet = ArrayLike
# Set representation (vertices, half-spaces, or samples)

SafeSet = ArrayLike
# Invariant safe region specification
```

### Analysis Results

```python
class ReachabilityResult(TypedDict):
    forward_set: ReachableSet      # States reachable from initial set
    backward_set: ReachableSet     # States that can reach target
    time_horizon: float            # Analysis time horizon
    volume: float                  # Set volume/measure
    representation: str            # 'polytope', 'ellipsoid', 'zonotope'
    method: str                    # 'hamilton-jacobi', 'polytope', 'sampling'
    vertices: ArrayLike            # Polytope vertices (if applicable)

class ROAResult(TypedDict):
    lyapunov_function: Callable    # V(x) Lyapunov function
    sublevel_set: float            # c such that {x: V(x) <= c} is ROA
    volume: float                  # Estimated ROA volume
    certified: bool                # Formally verified
    boundary_points: ArrayLike     # Points on ROA boundary

class VerificationResult(TypedDict):
    verified: bool                 # Property holds
    property_type: str             # 'safety', 'reachability', 'invariance'
    confidence: float              # Confidence level (0-1)
    counterexample: Optional[StateTrajectory]  # Violating trajectory
    computation_time: float

class BarrierCertificateResult(TypedDict):
    barrier_function: Callable     # B(x) barrier function
    safe_set: SafeSet              # {x: B(x) >= 0}
    unsafe_set: ArrayLike          # {x: B(x) < 0}
    certified: bool                # Barrier conditions verified
    margin: float                  # Safety margin

class CBFResult(TypedDict):
    cbf: Callable                  # Control Barrier Function h(x)
    safe_control_set: Callable     # K_cbf(x) = {u: dh/dt + alpha*h >= 0}
    alpha: float                   # Class-K function parameter
    qp_formulation: Dict           # QP problem structure

class CLFResult(TypedDict):
    clf: Callable                  # Control Lyapunov Function V(x)
    stabilizing_control: Callable  # u(x) from CLF-QP
    convergence_rate: float        # Exponential convergence rate
```

---

## 11. Robustness Types (`robustness.py`)

### Uncertainty Representation

```python
UncertaintySet = ArrayLike
# Parametric uncertainty representation
# Types: 'polytope', 'ellipsoid', 'box', 'norm-bounded'
```

### Analysis Results

```python
class RobustStabilityResult(TypedDict):
    robustly_stable: bool          # Stable for all uncertainties
    worst_case_eigenvalue: complex # Most destabilizing eigenvalue
    stability_margin: float        # Distance to instability
    critical_parameter: ParameterVector  # Parameter at margin

class StructuredSingularValueResult(TypedDict):
    mu_value: float                # Structured singular value
    robustness_margin: float       # 1/mu (uncertainty tolerance)
    upper_bound: float             # mu upper bound
    lower_bound: float             # mu lower bound
    d_scales: ArrayLike            # D-K iteration scales
    frequency: float               # Critical frequency (if freq-domain)

class TubeDefinition(TypedDict):
    nominal_trajectory: StateTrajectory
    tube_bounds: ArrayLike         # Tube cross-section at each time
    invariant: bool                # Tube is robustly invariant
    representation: str            # 'polytope', 'ellipsoid', 'zonotope'

class TubeMPCResult(TypedDict):
    nominal_control: ControlSequence  # Nominal MPC solution
    ancillary_gain: GainMatrix     # K for u = u_nom + K(x - x_nom)
    tube: TubeDefinition           # Robust invariant tube
    constraint_tightening: ArrayLike  # Tightened constraints
    cost: float
    success: bool
```

---

## 12. Optimization Types (`optimization.py`)

### Bounds and Constraints

```python
OptimizationBounds = Tuple[ArrayLike, ArrayLike]
# (lower_bounds, upper_bounds) for decision variables
```

### Result Types

```python
class OptimizationResult(TypedDict):
    x: ArrayLike                   # Optimal solution x*
    fun: float                     # Optimal objective f(x*)
    success: bool                  # Solver converged
    message: str                   # Termination message
    nit: int                       # Number of iterations
    nfev: int                      # Function evaluations
    njev: int                      # Jacobian evaluations
    nhev: int                      # Hessian evaluations

class ConstrainedOptimizationResult(OptimizationResult):
    lagrange_multipliers: ArrayLike  # Dual variables
    constraint_violations: ArrayLike # |c(x*)| for active constraints
    kkt_residual: float            # KKT condition residual
    active_set: List[int]          # Indices of active constraints

class TrajectoryOptimizationResult(TypedDict):
    states: StateTrajectory        # Optimal state trajectory
    controls: ControlSequence      # Optimal control sequence
    time: TimePoints               # Time grid
    cost: float                    # Optimal cost
    costates: ArrayLike            # Costate/adjoint trajectory (optional)
    success: bool
    method: str                    # 'direct_collocation', 'shooting', etc.
```

---

## 13. Learning Types (`learning.py`)

### Data Types

```python
Dataset = Tuple[StateTrajectory, ControlSequence, OutputSequence]
# (X, U, Y) training data

TrainingBatch = Tuple[StateVector, ControlVector, StateVector]
# (x[k], u[k], x[k+1]) for dynamics learning
```

### Configuration and Results

```python
class NeuralNetworkConfig(TypedDict, total=False):
    hidden_layers: List[int]       # [64, 64] layer sizes
    activation: str                # 'relu', 'tanh', 'silu'
    learning_rate: float           # Optimizer learning rate
    batch_size: int                # Training batch size
    epochs: int                    # Training epochs
    optimizer: str                 # 'adam', 'sgd', 'lbfgs'
    regularization: float          # L2 weight decay
    dropout: float                 # Dropout rate

class TrainingResult(TypedDict):
    final_loss: float              # Final training loss
    best_loss: float               # Best validation loss
    loss_history: List[float]      # Loss per epoch
    validation_loss: List[float]   # Validation loss per epoch
    training_time: float           # Total training time (seconds)
    early_stopped: bool            # Training stopped early
    best_epoch: int                # Epoch with best validation

class NeuralDynamicsResult(TypedDict):
    model: Any                     # Trained neural network
    training_result: TrainingResult
    prediction_error: float        # Test set error
    architecture: NeuralNetworkConfig

class RLTrainingResult(TypedDict):
    policy: Callable               # Learned policy pi(x) -> u
    value_function: Callable       # V(x) or Q(x,u)
    episode_rewards: List[float]   # Reward per episode
    training_time: float
    algorithm: str                 # 'ppo', 'sac', 'ddpg', etc.
```

---

## 14. Utility Types & Functions (`utilities.py`)

### Protocol Definitions

```python
class LinearizableProtocol(Protocol):
    """Any system that can be linearized."""
    def linearize(
        self,
        x_eq: StateVector,
        u_eq: ControlVector
    ) -> LinearizationResult: ...

class SimulatableProtocol(Protocol):
    """Any system that can be stepped forward."""
    def step(
        self,
        x: StateVector,
        u: ControlVector
    ) -> StateVector: ...

class StochasticProtocol(Protocol):
    """Systems with stochastic dynamics."""
    is_stochastic: bool
    def diffusion(
        self,
        x: StateVector,
        u: ControlVector
    ) -> DiffusionMatrix: ...
```

### Type Guard Functions

```python
def is_batched(x: ArrayLike) -> bool:
    """Check if array has batch dimension."""

def get_batch_size(x: ArrayLike) -> int:
    """Extract batch size from batched array."""

def is_numpy(x: ArrayLike) -> TypeGuard[np.ndarray]:
    """Check if array is NumPy."""

def is_torch(x: ArrayLike) -> TypeGuard[torch.Tensor]:
    """Check if array is PyTorch tensor."""

def is_jax(x: ArrayLike) -> TypeGuard[jax.Array]:
    """Check if array is JAX array."""

def get_backend(x: ArrayLike) -> Backend:
    """Detect and return array backend."""
```

### Converter Functions

```python
def ensure_numpy(x: ArrayLike) -> np.ndarray:
    """Convert any array to NumPy."""

def ensure_backend(x: ArrayLike, backend: Backend) -> ArrayLike:
    """Convert array to specified backend."""

class ArrayConverter:
    """Systematic array conversion between backends."""
    def to_numpy(self, x: ArrayLike) -> np.ndarray: ...
    def to_torch(self, x: ArrayLike, device: Device = "cpu") -> torch.Tensor: ...
    def to_jax(self, x: ArrayLike) -> jax.Array: ...
```

### Validator Functions

```python
def check_state_shape(x: ArrayLike, nx: int) -> bool:
    """Validate state vector dimensions."""

def check_control_shape(u: ArrayLike, nu: int) -> bool:
    """Validate control vector dimensions."""

def get_array_shape(x: ArrayLike) -> Tuple[int, ...]:
    """Get shape regardless of backend."""

def extract_dimensions(system: Any) -> SystemDimensions:
    """Extract nx, nu, ny from system object."""
```

### Cache and Metadata Types

```python
CacheKey = str
# Format for caching computed results

class CacheStatistics(TypedDict):
    hits: int
    misses: int
    size: int
    max_size: int

Metadata = Dict[str, Any]
# Arbitrary metadata dictionary

class ExecutionStats(TypedDict, total=False):
    """Execution time and resource statistics."""
    execution_time: float
    memory_used: int
    function_calls: int
    backend: str

class ValidationResult(TypedDict):
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    info: Dict

class SymbolicValidationResult:
    """Validation result for symbolic systems."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    info: Dict[str, Any]

class PerformanceMetrics(TypedDict, total=False):
    """Performance metrics for simulation/control."""
    settling_time: float
    rise_time: float
    overshoot: float
    steady_state_error: float
    control_effort: float
    trajectory_cost: float
```

---

## 15. Contraction Theory Types (`contraction.py`)

Types for contraction analysis and control - powerful tools for proving exponential convergence of dynamical systems.

### Type Aliases

```python
ContractionMetric = ArrayLike  # M(x) ≻ 0, defines Riemannian metric
ContractionRate = float        # β > 0, exponential convergence rate
```

**Mathematical Background**:
- Virtual displacement: δx = x₁ - x₂
- Metric norm: ||δx||²_M = δx' M(x) δx
- Contraction: d/dt ||δx||_M ≤ -β ||δx||_M
- Result: ||δx(t)||_M ≤ e^(-βt) ||δx(0)||_M

**Usage**:
```python
beta: ContractionRate = 0.5  # 63% convergence in t=1/β=2 seconds
M: ContractionMetric = np.eye(3)  # Euclidean metric
```

### Analysis Results

#### ContractionAnalysisResult

```python
class ContractionAnalysisResult(TypedDict, total=False):
    is_contracting: bool                    # System contracts
    contraction_rate: ContractionRate       # Exponential rate β
    metric: ContractionMetric               # M(x) ≻ 0
    metric_type: str                        # 'constant', 'state_dependent'
    verification_method: str                # 'LMI', 'SOS', 'optimization'
    convergence_bound: Optional[Callable]   # Upper bound ||δx(t)||
    exponential_convergence: bool           # True if exponential
    incremental_stability: bool             # All trajectories converge
    condition_number: Optional[float]       # κ(M) if constant metric
```

**Usage**:
```python
result: ContractionAnalysisResult = analyze_contraction(
    dynamics=lambda x: -A @ x,
    method='LMI'
)

if result['is_contracting']:
    β = result['contraction_rate']
    M = result['metric']
    print(f"System contracts with rate β={β:.3f}")
    
    # Convergence bound
    bound = result['convergence_bound']
    t = np.linspace(0, 10, 100)
    max_deviation = [bound(ti, 1.0) for ti in t]
```

#### CCMResult (Control Contraction Metrics)

```python
class CCMResult(TypedDict, total=False):
    feedback_gain: GainMatrix               # K(x) or constant K
    metric: ContractionMetric               # M(x)
    contraction_rate: ContractionRate       # Guaranteed β
    metric_condition_number: ArrayLike      # κ(M(x)) over state space
    contraction_verified: bool              # LMI/SDP succeeded
    robustness_margin: float                # Margin in contraction condition
    geodesic_distance: Optional[Callable]   # d_M(x₁, x₂)
```

**Usage**:
```python
result: CCMResult = design_ccm_controller(
    system=pendulum,
    contraction_rate=0.5,
    method='SDP'
)

if result['contraction_verified']:
    K = result['feedback_gain']
    
    # Apply controller
    def controller(x):
        return K(x) if callable(K) else K @ x
    
    print(f"Robustness margin: {result['robustness_margin']:.3f}")
```

#### FunnelingResult

```python
class FunnelingResult(TypedDict):
    controller: Callable                    # u(x, t)
    tracking_funnel: Callable               # ρ(t) boundary
    funnel_shape: str                       # 'exponential', 'polynomial'
    reference_trajectory: StateTrajectory   # x_d(t)
    performance_bound: Callable             # ||x - x_d|| ≤ bound(t)
    transient_bound: float                  # Initial amplification
    contraction_rate: ContractionRate       # Asymptotic rate
```

**Usage**:
```python
result: FunnelingResult = design_funnel_controller(
    system=robot,
    reference_trajectory=x_desired,
    funnel_shape='exponential',
    initial_funnel_width=1.0,
    final_funnel_width=0.1
)

controller = result['controller']
funnel = result['tracking_funnel']

# Verify tracking stays within funnel
error = np.linalg.norm(x_traj - x_desired, axis=1)
bound = np.array([funnel(t) for t in t_sim])
assert np.all(error <= bound + 1e-6)
```

#### IncrementalStabilityResult

```python
class IncrementalStabilityResult(TypedDict):
    incrementally_stable: bool              # δ-GAS property
    contraction_rate: Optional[ContractionRate]
    metric: Optional[ContractionMetric]
    kl_bound: Optional[Callable]            # β(||δx(0)||, t)
    convergence_type: str                   # 'exponential', 'asymptotic'
```

**Usage**:
```python
result: IncrementalStabilityResult = check_incremental_stability(
    system=nonlinear_system,
    method='contraction'
)

if result['incrementally_stable']:
    print("All trajectories converge to each other!")
    if result['convergence_type'] == 'exponential':
        β = result['contraction_rate']
        print(f"Exponential rate: {β:.3f}")
```

---

## 16. Conformal Prediction Types (`conformal.py`)

Types for distribution-free uncertainty quantification with finite-sample coverage guarantees.

### Type Aliases

```python
ConformalPredictionSet = Union[ArrayLike, Tuple[ArrayLike, ArrayLike]]
# Interval: (lower_bounds, upper_bounds)
# Ball: (center, radius)
# Polytope: vertices array

NonconformityScore = ArrayLike
# Measures prediction "strangeness"
# Lower score → more conforming to calibration data
```

**Mathematical Background**:
- Split conformal: Train/calibration split
- Coverage guarantee: P(y_new ∈ C(x_new)) ≥ 1-α (distribution-free!)
- Nonconformity score: s_i = score(x_i, y_i, ŷ_i)
- Quantile: q = Quantile(s₁,...,s_n; (1-α)(1 + 1/n))

### Calibration Results

#### ConformalCalibrationResult

```python
class ConformalCalibrationResult(TypedDict):
    quantile: float                         # Calibrated threshold q
    alpha: float                            # Target miscoverage (1-coverage)
    empirical_coverage: float               # Observed coverage on cal set
    calibration_scores: NonconformityScore  # From calibration data
    n_calibration: int                      # Calibration set size
    prediction_set_type: str                # 'interval', 'ball', 'polytope'
```

**Usage**:
```python
result: ConformalCalibrationResult = calibrate_conformal(
    scores=residuals,
    alpha=0.1  # Target 90% coverage
)

q = result['quantile']
coverage = result['empirical_coverage']
print(f"Calibrated quantile: {q:.3f}")
print(f"Empirical coverage: {coverage*100:.1f}%")

# Prediction set: {y : |y - ŷ| ≤ q}
y_lower = y_pred - q
y_upper = y_pred + q
```

### Prediction Results

#### ConformalPredictionResult

```python
class ConformalPredictionResult(TypedDict, total=False):
    prediction_set: ConformalPredictionSet  # Set(s) for test point(s)
    point_prediction: StateVector           # Point estimate
    coverage_guarantee: float               # Guaranteed coverage (1-α)
    average_set_size: float                 # Average prediction set size
    nonconformity_score: NonconformityScore # For test point(s)
    threshold: float                        # Threshold q
    adaptive: bool                          # Input-adaptive
```

**Usage**:
```python
cp = ConformalPredictor(model=my_model, calibration_data=(x_cal, y_cal))

result: ConformalPredictionResult = cp.predict(
    x_test=np.array([1.5, 2.0]),
    alpha=0.1  # 90% coverage guarantee
)

if isinstance(result['prediction_set'], tuple):
    lower, upper = result['prediction_set']
    print(f"Prediction interval: [{lower}, {upper}]")

print(f"Coverage guarantee: {result['coverage_guarantee']*100:.1f}%")
print(f"Average set size: {result['average_set_size']:.3f}")
```

#### AdaptiveConformalResult

```python
class AdaptiveConformalResult(TypedDict):
    threshold: float                        # Current adaptive q_t
    coverage_history: ArrayLike             # Coverage over time
    miscoverage_rate: float                 # Current miscoverage
    target_alpha: float                     # Target (1-coverage)
    adaptation_rate: float                  # Learning rate γ
    effective_sample_size: int              # Effective calibration window
```

**Usage**:
```python
aci = AdaptiveConformalInference(
    model=my_model,
    target_alpha=0.1,
    adaptation_rate=0.01
)

# Online updates
for t in range(1000):
    x_t, y_t = data_stream[t]
    result: AdaptiveConformalResult = aci.update(x_t, y_t)
    
    if t % 100 == 0:
        coverage = 1 - result['miscoverage_rate']
        print(f"Step {t}: Coverage = {coverage*100:.1f}%")
        print(f"Threshold: {result['threshold']:.3f}")
```

---

## 17. Differential Flatness Types (`flatness.py`)

Types for differential flatness - enables exact trajectory planning via algebraic inversion.

### Type Aliases

```python
FlatnessOutput = ArrayLike  # Flat output y_flat (m,)
# State and control algebraically expressible:
#   x = φ_x(y_flat, ẏ_flat, ÿ_flat, ...)
#   u = φ_u(y_flat, ẏ_flat, ÿ_flat, ...)
```

**Mathematical Background**:
- System is flat if ∃ y_flat such that:
  - y_flat = σ(x) (function of state)
  - x = φ_x(y_flat, ẏ_flat, ..., y_flat^(r))
  - u = φ_u(y_flat, ẏ_flat, ..., y_flat^(r))
- dim(y_flat) = m (number of inputs)
- Every trajectory ↔ unique flat output trajectory
- Enables planning in flat space + algebraic inversion

### Flatness Analysis

#### DifferentialFlatnessResult

```python
class DifferentialFlatnessResult(TypedDict):
    is_flat: bool                           # System is differentially flat
    flat_output: Optional[Callable]         # y_flat = σ(x)
    flat_dimension: int                     # dim(y_flat) = nu typically
    differential_order: int                 # Max derivative order needed
    state_from_flat: Optional[Callable]     # x = φ_x(y, ẏ, ÿ, ...)
    control_from_flat: Optional[Callable]   # u = φ_u(y, ẏ, ÿ, ...)
    verification_method: str                # 'analytic', 'symbolic', 'numerical'
```

**Usage**:
```python
result: DifferentialFlatnessResult = check_flatness(
    system=quadrotor,
    method='analytic'
)

if result['is_flat']:
    print(f"System is flat! Flat output dim: {result['flat_dimension']}")
    print(f"Derivative order: {result['differential_order']}")
    
    σ = result['flat_output']           # y = σ(x)
    φ_x = result['state_from_flat']     # x = φ_x(y, ẏ, ...)
    φ_u = result['control_from_flat']   # u = φ_u(y, ẏ, ...)
    
    # Plan trajectory in flat output space
    y_flat_traj = plan_flat_output(t)
    dy_flat = compute_derivatives(y_flat_traj, dt)
    ddy_flat = compute_derivatives(dy_flat, dt)
    
    # Invert to get x(t) and u(t)
    x_traj = φ_x(y_flat_traj, dy_flat, ddy_flat)
    u_traj = φ_u(y_flat_traj, dy_flat, ddy_flat)
```

### Trajectory Planning

#### TrajectoryPlanningResult

```python
class TrajectoryPlanningResult(TypedDict, total=False):
    state_trajectory: StateTrajectory       # x(t) or x[k]
    control_trajectory: ControlSequence     # u(t) or u[k]
    flat_trajectory: Optional[ArrayLike]    # y_flat(t) if flat system
    time_points: TimePoints                 # Time discretization
    cost: float                             # Trajectory cost J
    feasible: bool                          # Satisfies constraints
    method: str                             # 'flatness', 'optimization', 'RRT'
    computation_time: float                 # Planning time (seconds)
```

**Usage**:
```python
result: TrajectoryPlanningResult = plan_flat_trajectory(
    system=quadrotor,
    x_initial=np.zeros(12),
    x_final=np.array([10, 10, 5, 0, ...]),
    time_horizon=5.0
)

if result['feasible']:
    x_ref = result['state_trajectory']
    u_ff = result['control_trajectory']
    
    print(f"Planning time: {result['computation_time']:.3f}s")
    print(f"Trajectory cost: {result['cost']:.2f}")
    print(f"Method: {result['method']}")
    
    # Execute with feedforward + feedback
    for k in range(len(result['time_points'])-1):
        u_ff_k = u_ff[k]
        e = x - x_ref[k]
        u = u_ff_k - K @ e  # Feedforward + feedback
        x = system.step(x, u, dt)
    
    # Visualize flat trajectory
    if 'flat_trajectory' in result:
        y_flat = result['flat_trajectory']
        import matplotlib.pyplot as plt
        plt.plot(y_flat[:, 0], y_flat[:, 1], label='Flat trajectory')
        plt.xlabel('x position')
        plt.ylabel('y position')
```

---

## 18. Model Reduction Types (`model_reduction.py`)

Types for creating lower-order approximations with guaranteed error bounds.

### Balanced Realization

#### BalancedRealizationResult

```python
class BalancedRealizationResult(TypedDict):
    A_balanced: StateMatrix                 # Balanced A_bal (nx, nx)
    B_balanced: InputMatrix                 # Balanced B_bal (nx, nu)
    C_balanced: OutputMatrix                # Balanced C_bal (ny, nx)
    transformation: ArrayLike               # Balancing transform T
    hankel_singular_values: ArrayLike       # HSVs σ₁ ≥ σ₂ ≥ ... ≥ 0
    controllability_gramian: CovarianceMatrix  # W_c
    observability_gramian: CovarianceMatrix    # W_o
```

**Mathematical Background**:
- Balanced coordinates: W_c = W_o = Σ = diag(σ₁, σ₂, ..., σ_n)
- σ_i are Hankel singular values (HSVs)
- HSVs measure state importance (controllability + observability)
- States with small HSVs can be truncated

**Usage**:
```python
result: BalancedRealizationResult = balanced_realization(
    A=A, B=B, C=C
)

A_bal = result['A_balanced']
B_bal = result['B_balanced']
C_bal = result['C_balanced']
hsv = result['hankel_singular_values']

# Plot HSVs to choose reduced order
import matplotlib.pyplot as plt
plt.semilogy(hsv, 'o-')
plt.axhline(1e-3, color='r', linestyle='--', label='Threshold')
plt.xlabel('State index')
plt.ylabel('Hankel singular value')
plt.grid(True)

# Choose order (keep HSVs > threshold)
r = np.sum(hsv > 1e-3)
print(f"Reduced order: {r}/{len(hsv)}")

# Verify balanced gramians
W_c = result['controllability_gramian']
W_o = result['observability_gramian']
assert np.allclose(W_c, W_o)
assert np.allclose(W_c, np.diag(hsv))
```

### Reduced Order Models

#### ReducedOrderModelResult

```python
class ReducedOrderModelResult(TypedDict):
    A_reduced: StateMatrix                  # Reduced A_r (n_r, n_r)
    B_reduced: InputMatrix                  # Reduced B_r (n_r, nu)
    C_reduced: OutputMatrix                 # Reduced C_r (ny, n_r)
    D_reduced: FeedthroughMatrix            # Reduced D_r (ny, nu)
    original_order: int                     # Original nx
    reduced_order: int                      # Reduced n_r
    approximation_error: float              # Error bound ||G - G_r||_∞
    method: str                             # 'balanced', 'modal', 'hankel', 'POD'
    preserved_modes: np.ndarray             # Eigenvalues kept
```

**Mathematical Background**:
- Balanced truncation error bound: ||G - G_r||_∞ ≤ 2(σ_r+1 + ... + σ_n)
- Preserves stability (if original stable)
- Guarantees quality of approximation

**Usage**:
```python
result: ReducedOrderModelResult = reduce_model(
    system=(A, B, C, D),
    target_order=10,
    method='balanced'
)

A_r = result['A_reduced']
B_r = result['B_reduced']
C_r = result['C_reduced']
D_r = result['D_reduced']

print(f"Original: {result['original_order']} states")
print(f"Reduced: {result['reduced_order']} states")
print(f"Error bound: {result['approximation_error']:.2e}")

# Compare transfer functions
from scipy import signal
sys_orig = signal.StateSpace(A, B, C, D)
sys_red = signal.StateSpace(A_r, B_r, C_r, D_r)

omega = np.logspace(-2, 2, 100)
_, G_orig = signal.freqresp((A, B, C, D), omega)
_, G_red = signal.freqresp((A_r, B_r, C_r, D_r), omega)

# Plot Bode comparison
import matplotlib.pyplot as plt
plt.loglog(omega, np.abs(G_orig[0, 0]), label='Original')
plt.loglog(omega, np.abs(G_red[0, 0]), '--', label='Reduced')
plt.xlabel('Frequency (rad/s)')
plt.ylabel('Magnitude')
plt.legend()

# Verify error bound
error = np.max(np.abs(G_orig - G_red))
print(f"Max frequency error: {error:.2e}")
print(f"Error bound: {result['approximation_error']:.2e}")
assert error <= result['approximation_error']
```

---

## Design Principles

### 1. Backend Agnostic Design

All array types support NumPy, PyTorch, and JAX equally:

```python
def compute_gain(A: StateMatrix, B: InputMatrix) -> GainMatrix:
    # Works with any backend
    backend = get_backend(A)
    # ... computation ...
    return ensure_backend(K, backend)
```

### 2. Semantic Naming

Types are named by their role in control theory, not their representation:

```python
# Clear intent from type names
def lqr(A: StateMatrix, B: InputMatrix,
        Q: CostMatrix, R: CostMatrix) -> LQRResult:
    ...

# vs ambiguous
def lqr(A: np.ndarray, B: np.ndarray,
        Q: np.ndarray, R: np.ndarray) -> Tuple:
    ...
```

### 3. Polymorphic Types

Union types handle both deterministic and stochastic cases:

```python
def analyze_system(lin: LinearizationResult) -> StabilityInfo:
    if len(lin) == 2:
        A, B = lin  # Deterministic
    else:
        A, B, G = lin  # Stochastic
    # ... analysis ...
```

### 4. Protocol-Based Interfaces

Structural subtyping without inheritance:

```python
def simulate(system: SimulatableProtocol, x0: StateVector) -> StateTrajectory:
    # Works with any object that has a step() method
    ...
```

### 5. TypedDict for Results

Rich, type-safe return values:

```python
result = lqr_design(A, B, Q, R)
K = result["K"]           # Type-checked access
eigenvalues = result["eigenvalues"]
print(f"Gain margin: {result['gain_margin']:.2f} dB")
```

---

## Usage Patterns

### Pattern 1: Type-Safe Control Design

```python
from src.types import (
    StateMatrix, InputMatrix, CostMatrix,
    LQRResult, GainMatrix
)

def design_controller(
    A: StateMatrix,
    B: InputMatrix,
    Q: CostMatrix,
    R: CostMatrix
) -> GainMatrix:
    result: LQRResult = lqr_solve(A, B, Q, R)
    return result["K"]
```

### Pattern 2: Backend-Agnostic Functions

```python
from src.types import ArrayLike, Backend, ensure_backend, get_backend

def process_trajectory(
    states: ArrayLike,
    target_backend: Backend = "numpy"
) -> ArrayLike:
    # Detect input backend
    source = get_backend(states)

    # Process (backend-agnostic)
    result = some_computation(states)

    # Convert to target
    return ensure_backend(result, target_backend)
```

### Pattern 3: Protocol-Based Dispatch

```python
from src.types import LinearizableProtocol, SimulatableProtocol

def analyze_and_simulate(
    system: LinearizableProtocol & SimulatableProtocol,
    x0: StateVector
) -> Dict[str, Any]:
    # Linearize at equilibrium
    A, B = system.linearize(x0, np.zeros(system.nu))

    # Simulate
    trajectory = []
    x = x0
    for _ in range(100):
        x = system.step(x, -K @ x)
        trajectory.append(x)

    return {"linearization": (A, B), "trajectory": trajectory}
```

### Pattern 4: Result Unpacking

```python
from src.types import MPCResult

def run_mpc(system, x0) -> Tuple[ControlVector, float]:
    result: MPCResult = solve_mpc(system, x0)

    if not result["success"]:
        raise RuntimeError(result.get("message", "MPC failed"))

    # Apply first control
    u0 = result["control_sequence"][0]
    cost = result["cost"]

    print(f"Solved in {result['solve_time']:.3f}s, "
          f"{result['iterations']} iterations")

    return u0, cost
```

---

## Integration Pipelines

### Control Design Pipeline

```
System Definition
       │
       ▼
┌──────────────────┐
│   Linearization  │ → LinearizationResult (A, B) or (A, B, G)
└──────────────────┘
       │
       ▼
┌──────────────────┐
│ Classical Control│ → LQRResult, KalmanFilterResult
└──────────────────┘
       │
       ▼
┌──────────────────┐
│ Advanced Control │ → MPCResult, HInfControlResult
└──────────────────┘
```

### State Estimation Pipeline

```
Measurements
       │
       ▼
┌──────────────────┐
│    EKF / UKF     │ → EKFResult, UKFResult
└──────────────────┘
       │
       ▼
┌──────────────────┐
│ Particle Filter  │ → ParticleFilterResult
└──────────────────┘
       │
       ▼
State Estimate + Covariance
```

### System Identification Pipeline

```
Input/Output Data
       │
       ▼
┌──────────────────┐
│   Subspace ID    │ → SubspaceIDResult, ERAResult
└──────────────────┘
       │
       ▼
┌──────────────────┐
│    DMD / SINDy   │ → DMDResult, SINDyResult
└──────────────────┘
       │
       ▼
┌──────────────────┐
│     Koopman      │ → KoopmanResult
└──────────────────┘
       │
       ▼
Identified Model (A, B, C, D)
```

### Safety Verification Pipeline

```
System + Constraints
       │
       ▼
┌──────────────────┐
│  Reachability    │ → ReachabilityResult
└──────────────────┘
       │
       ▼
┌──────────────────┐
│ Barrier Function │ → BarrierCertificateResult, CBFResult
└──────────────────┘
       │
       ▼
┌──────────────────┐
│   Verification   │ → VerificationResult
└──────────────────┘
       │
       ▼
Safety Certificate
```

---

## Quick Reference Tables

### Vector/Matrix Dimension Conventions

| Type | Typical Variable | Shape |
|------|------------------|-------|
| `StateVector` | `x` | `(nx,)` |
| `ControlVector` | `u` | `(nu,)` |
| `OutputVector` | `y` | `(ny,)` |
| `StateMatrix` | `A` | `(nx, nx)` |
| `InputMatrix` | `B` | `(nx, nu)` |
| `OutputMatrix` | `C` | `(ny, nx)` |
| `FeedthroughMatrix` | `D` | `(ny, nu)` |
| `DiffusionMatrix` | `G` | `(nx, nw)` |
| `CovarianceMatrix` | `P, Q, R` | `(n, n)` |
| `GainMatrix` | `K, L` | varies |

### Common TypedDict Fields

| Field | Type | Description |
|-------|------|-------------|
| `success` | `bool` | Operation completed successfully |
| `cost` | `float` | Objective function value |
| `eigenvalues` | `ArrayLike` | System poles |
| `iterations` | `int` | Solver iterations |
| `solve_time` | `float` | Computation time (seconds) |
| `method` | `str` | Algorithm used |

### Backend Detection

```python
x = some_array
if is_numpy(x):
    # NumPy operations
elif is_torch(x):
    # PyTorch operations
elif is_jax(x):
    # JAX operations
```
