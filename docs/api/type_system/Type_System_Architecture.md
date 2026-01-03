# ControlDESymulation Type System Architecture

## Overview

The **Type System** is the **foundational layer** that provides semantic types, structured results, and type-safe interfaces for the entire framework. It consists of **8 focused modules** totaling ~7,000 lines that define over **200 type aliases** and **structured dictionaries**.

## Architecture Philosophy

**Type-Driven Design** - The type system enables:

1. **Semantic Clarity** - Names convey mathematical meaning (`StateVector`, not `ArrayLike`)
2. **Type Safety** - Static type checking via mypy/pyright
3. **IDE Support** - Autocomplete and inline documentation
4. **Backend Agnosticism** - Same types work with NumPy/PyTorch/JAX
5. **Structured Results** - TypedDict for dictionaries (not plain `dict`)
6. **Self-Documenting** - Types encode constraints and invariants

```python
# Compare:
def bad(x, u):  # What are x and u? Arrays? Scalars? Dimensions?
    return x + u

# vs:
def good(x: StateVector, u: ControlVector) -> StateVector:
    """Clear intent: state in, control in, state out."""
    return x + u
```

## Type System Layers

```
┌────────────────────────────────────────────────────────────┐
│                    APPLICATION LAYER                        │
│  (UI Framework, Delegation Layer, Integration Framework)   │
└──────────────────┬─────────────────────────────────────────┘
                   │ uses types from
                   ↓
┌────────────────────────────────────────────────────────────┐
│                    TYPE SYSTEM                              │
│                                                              │
│  ┌─────────────────────────────────────────────────┐       │
│  │  FOUNDATIONAL TYPES                             │       │
│  │  • core.py           - Vectors, matrices        │       │
│  │  • backends.py       - Backend enums            │       │
│  └─────────────────────────────────────────────────┘       │
│                                                              │
│  ┌─────────────────────────────────────────────────┐       │
│  │  DOMAIN TYPES                                   │       │
│  │  • trajectories.py   - Time series results      │       │
│  │  • linearization.py  - Jacobian tuples          │       │
│  │  • symbolic.py       - SymPy types              │       │
│  │  • control_classical.py - Control design results│       │
│  └─────────────────────────────────────────────────┘       │
│                                                              │
│  ┌─────────────────────────────────────────────────┐       │
│  │  STRUCTURAL TYPES                               │       │
│  │  • protocols.py      - Abstract interfaces      │       │
│  │  • utilities.py      - Helper types             │       │
│  └─────────────────────────────────────────────────┘       │
└────────────────────────────────────────────────────────────┘
```

## Module Breakdown

### Foundational Types

#### core.py
**File:** `core.py` (1,501 lines)

**Purpose:** Fundamental building blocks for all other types

**Categories:**

**1. Multi-Backend Arrays (20+ types)**
```python
ArrayLike = Union[np.ndarray, torch.Tensor, jnp.ndarray]
NumpyArray = np.ndarray
TorchTensor = torch.Tensor
JaxArray = jnp.ndarray
ScalarLike = Union[float, int, np.number, torch.Tensor, jnp.ndarray]
IntegerLike = Union[int, np.integer]
```

**2. Semantic Vector Types (15+ types)**
```python
StateVector         # x ∈ ℝⁿˣ - State
ControlVector       # u ∈ ℝⁿᵘ - Control input
OutputVector        # y ∈ ℝⁿʸ - Measured output
NoiseVector         # w ∈ ℝⁿʷ - Stochastic noise
EquilibriumState    # x_eq - Equilibrium state
EquilibriumControl  # u_eq - Equilibrium control
TimeDerivative      # dx/dt - State derivative
StateIncrement      # δx - State deviation
ControlIncrement    # δu - Control deviation
```

**3. Matrix Types (30+ types)**
```python
# Dynamics matrices
StateMatrix         # A ∈ ℝⁿˣˣⁿˣ - ∂f/∂x
InputMatrix         # B ∈ ℝⁿˣˣⁿᵘ - ∂f/∂u  
DiffusionMatrix     # G ∈ ℝⁿˣˣⁿʷ - Noise intensity

# Observation matrices
OutputMatrix        # C ∈ ℝⁿʸˣⁿˣ - ∂h/∂x
FeedthroughMatrix   # D ∈ ℝⁿʸˣⁿᵘ - Direct feedthrough

# Control matrices
GainMatrix          # K ∈ ℝⁿᵘˣⁿˣ - Feedback gain
CostMatrix          # Q ∈ ℝⁿˣˣⁿˣ - State cost
ControlCostMatrix   # R ∈ ℝⁿᵘˣⁿᵘ - Control cost

# Stochastic matrices
CovarianceMatrix    # P ∈ ℝⁿˣˣⁿˣ - Covariance
ProcessNoiseMatrix  # Q ∈ ℝⁿˣˣⁿˣ - Process noise cov
MeasurementNoiseMatrix  # R ∈ ℝⁿʸˣⁿʸ - Measurement noise cov

# Special matrices
IdentityMatrix      # I ∈ ℝⁿˣⁿ
ZeroMatrix         # 0 ∈ ℝᵐˣⁿ
GramianMatrix      # Controllability/observability
```

**4. Function Signatures (10+ types)**
```python
DynamicsFunction    # (x, u) → dx/dt
OutputFunction      # (x) → y
ControlPolicy       # (t, x) → u
CostFunction        # (x, u) → scalar
ObserverFunction    # (y, u) → x_hat
```

**5. System Properties (15+ types)**
```python
StateDimension      # nx - Number of states
ControlDimension    # nu - Number of controls
OutputDimension     # ny - Number of outputs
NoiseDimension      # nw - Number of Wiener processes
SystemOrder         # order - Differential order

EquilibriumPoint    # (x_eq, u_eq) - Tuple
EquilibriumName     # str - Named equilibrium
```

**Key Design:**

- **Semantic naming** - Type names encode mathematical meaning
- **Multi-backend** - All types support NumPy/PyTorch/JAX
- **Composition** - Types compose into higher-level structures
- **Documentation** - Each type has examples and constraints

---

#### backends.py
**File:** `backends.py` (735 lines)

**Purpose:** Backend configuration and method selection types

**Categories:**

**1. Backend Types**
```python
Backend = Literal["numpy", "torch", "jax"]
Device = str  # 'cpu', 'cuda:0', 'mps', 'tpu'

class BackendConfig(TypedDict, total=False):
    backend: Backend
    device: Optional[Device]
    dtype: Optional[str]  # 'float32', 'float64'
```

**2. Integration Methods**
```python
IntegrationMethod = str  # 'RK45', 'dopri5', 'tsit5', etc.

# Specific categories
OdeMethod = str          # Deterministic methods
SdeMethod = str          # Stochastic methods
FixedStepMethod = str    # Fixed-step methods
AdaptiveMethod = str     # Adaptive methods
```

**3. Discretization Methods**
```python
DiscretizationMethod = Literal[
    "exact",      # Matrix exponential
    "euler",      # Forward Euler
    "tustin",     # Bilinear transform
    "backward",   # Backward Euler
    "matched",    # Zero-order hold
]
```

**4. SDE Types**
```python
SDEType = Literal["ito", "stratonovich"]

class NoiseType(Enum):
    ADDITIVE = "additive"
    MULTIPLICATIVE = "multiplicative"
    MULTIPLICATIVE_DIAGONAL = "multiplicative_diagonal"
    MULTIPLICATIVE_SCALAR = "multiplicative_scalar"
    MULTIPLICATIVE_GENERAL = "multiplicative_general"
    UNKNOWN = "unknown"

class ConvergenceType(Enum):
    STRONG = "strong"  # Pathwise convergence
    WEAK = "weak"      # Distribution convergence
```

**5. System Configuration**
```python
class SystemConfig(TypedDict, total=False):
    """Complete system configuration."""
    nx: int              # State dimension
    nu: int              # Control dimension
    ny: int              # Output dimension
    nw: int              # Noise dimension
    order: int           # System order
    dt: Optional[float]  # Time step (discrete)
    backend: Backend
    device: Device
```

**Key Design:**

- **Type-safe enums** - Literal types prevent typos
- **Structured configs** - TypedDict for configuration
- **Extensible** - Easy to add new methods
- **Self-documenting** - Clear valid values

---

### Domain Types

#### trajectories.py
**File:** `trajectories.py** (879 lines)

**Purpose:** Time series data and simulation results

**Categories:**

**1. Trajectory Types**
```python
StateTrajectory = ArrayLike      # (T, nx) or (T, batch, nx)
ControlSequence = ArrayLike      # (T, nu) or (T, batch, nu)
OutputSequence = ArrayLike       # (T, ny)
NoiseSequence = ArrayLike        # (T, nw)
```

**2. Time Types**
```python
TimePoints = ArrayLike           # (T,) - Time grid
TimeSpan = Tuple[float, float]   # (t0, tf) - Interval
TimeStep = float                 # dt - Step size
```

**3. Integration Results (TypedDict)**
```python
class IntegrationResult(TypedDict, total=False):
    """ODE integration result."""
    t: TimePoints              # Time points
    x: StateTrajectory         # State trajectory (T, nx)
    success: bool              # Integration succeeded
    message: str               # Status message
    nfev: int                  # Function evaluations
    nsteps: int                # Integration steps
    integration_time: float    # Wall time (seconds)
    solver: str                # Integrator name
    
    # Optional fields (adaptive methods)
    njev: int                  # Jacobian evaluations
    nlu: int                   # LU decompositions
    status: int                # Solver status code
    sol: Any                   # Dense output object
    dense_output: bool         # Dense output available
```

**4. SDE Integration Results**
```python
class SDEIntegrationResult(TypedDict, total=False):
    """SDE integration result (extends IntegrationResult)."""
    # All IntegrationResult fields, plus:
    diffusion_evals: int       # Diffusion function calls
    noise_samples: NoiseVector # Brownian increments used
    n_paths: int               # Number of trajectories
    convergence_type: str      # 'strong' or 'weak'
    sde_type: str              # 'ito' or 'stratonovich'
    noise_type: str            # Noise structure
```

**5. Batch Results**
```python
class BatchSimulationResult(TypedDict):
    """Batched simulation result."""
    t: TimePoints                    # (T,)
    x: StateTrajectory               # (T, batch, nx)
    u: ControlSequence               # (T, batch, nu)
    batch_size: int
    statistics: Dict[str, ArrayLike] # Mean, std, etc.
```

**Key Design:**

- **Time-major ordering** - (T, nx) not (nx, T)
- **TypedDict results** - Structured, type-safe dictionaries
- **Optional fields** - Use `total=False` for flexibility
- **Consistent shapes** - All trajectories follow same conventions

---

#### linearization.py
**File:** `linearization.py` (502 lines)

**Purpose:** Linearization results and Jacobian types

**Categories:**

**1. Basic Linearization**
```python
DeterministicLinearization = Tuple[StateMatrix, InputMatrix]
# Returns: (A, B) where
#   A = ∂f/∂x - State Jacobian
#   B = ∂f/∂u - Control Jacobian

StochasticLinearization = Tuple[StateMatrix, InputMatrix, DiffusionMatrix]
# Returns: (A, B, G) where
#   A = ∂f/∂x
#   B = ∂f/∂u  
#   G = ∂g/∂x or g(x_eq) - Diffusion

LinearizationResult = Union[DeterministicLinearization, StochasticLinearization]
# Polymorphic: works with both
```

**2. Output Linearization**
```python
ObservationLinearization = Tuple[OutputMatrix, FeedthroughMatrix]
# Returns: (C, D) where
#   C = ∂h/∂x - Output Jacobian
#   D = ∂h/∂u - Feedthrough (usually 0)
```

**3. Full State-Space**
```python
FullLinearization = Tuple[StateMatrix, InputMatrix, OutputMatrix, FeedthroughMatrix]
# Returns: (A, B, C, D)

FullStochasticLinearization = Tuple[
    StateMatrix, InputMatrix, DiffusionMatrix, OutputMatrix, FeedthroughMatrix
]
# Returns: (A, B, G, C, D)
```

**4. Time-Domain Aliases**
```python
ContinuousLinearization = DeterministicLinearization
DiscreteLinearization = DeterministicLinearization
ContinuousStochasticLinearization = StochasticLinearization
DiscreteStochasticLinearization = StochasticLinearization
```

**5. Jacobian-Specific**
```python
StateJacobian = StateMatrix       # A = ∂f/∂x
ControlJacobian = InputMatrix     # B = ∂f/∂u
OutputJacobian = OutputMatrix     # C = ∂h/∂x
DiffusionJacobian = DiffusionMatrix  # G = ∂g/∂x
```

**Key Design:**

- **Tuple returns** - Natural unpacking: `A, B = linearize()`
- **Polymorphic types** - Union handles deterministic/stochastic
- **Semantic aliases** - Time-domain context clear
- **Mathematical clarity** - Names match theory

---

#### symbolic.py
**File:** `symbolic.py` (646 lines)

**Purpose:** SymPy symbolic types

**Categories:**

**1. Symbolic Variables**
```python
SymbolicVariable = sp.Symbol        # Single variable
SymbolicVector = sp.Matrix          # Vector of symbols
SymbolicMatrix = sp.Matrix          # Matrix expression
SymbolicExpression = sp.Expr        # General expression
```

**2. System Components**
```python
DynamicsExpression = sp.Matrix      # f(x, u) symbolic
OutputExpression = sp.Matrix        # h(x) symbolic
DiffusionExpression = sp.Matrix     # g(x, u) symbolic
ParameterDict = Dict[sp.Symbol, float]  # Parameter values
```

**3. Jacobian Expressions**
```python
JacobianExpression = sp.Matrix      # ∂f/∂x symbolic
HessianExpression = sp.Matrix       # ∂²f/∂x² symbolic
GradientExpression = sp.Matrix      # ∇f symbolic
```

**4. Substitution Types**
```python
SubstitutionDict = Dict[sp.Symbol, Union[float, sp.Expr]]
SimplificationStrategy = Literal["simplify", "expand", "factor", "cancel"]
```

**Key Design:**

- **SymPy integration** - Bridge symbolic ↔ numerical
- **Type annotations** - Clarify SymPy usage
- **Code generation** - Input to compilation pipeline
- **Parameter substitution** - Clear parameter handling

---

#### control_classical.py
**File:** `control_classical.py` (542 lines)

**Purpose:** Classical control theory result types

**Categories:**

**1. System Analysis Results**
```python
class StabilityInfo(TypedDict):
    """Stability analysis result.
    
    Stability Criteria:
    - Continuous: All Re(λ) < 0 (left half-plane)
    - Discrete: All |λ| < 1 (inside unit circle)
    """
    eigenvalues: np.ndarray          # Complex eigenvalues
    magnitudes: np.ndarray           # |λ| values
    max_magnitude: float             # Spectral radius
    spectral_radius: float           # Same as max_magnitude
    is_stable: bool                  # Asymptotically stable
    is_marginally_stable: bool       # On stability boundary
    is_unstable: bool                # Unstable

class ControllabilityInfo(TypedDict, total=False):
    """Controllability analysis result.
    
    Test: rank(C) = nx where C = [B AB A²B ... Aⁿ⁻¹B]
    """
    controllability_matrix: ControllabilityMatrix  # (nx, nx*nu)
    rank: int                        # Rank of C
    is_controllable: bool            # rank == nx
    uncontrollable_modes: Optional[np.ndarray]  # Eigenvalues

class ObservabilityInfo(TypedDict, total=False):
    """Observability analysis result.
    
    Test: rank(O) = nx where O = [C; CA; CA²; ...; CAⁿ⁻¹]
    """
    observability_matrix: ObservabilityMatrix  # (nx*ny, nx)
    rank: int                        # Rank of O
    is_observable: bool              # rank == nx
    unobservable_modes: Optional[np.ndarray]  # Eigenvalues
```

**2. Control Design Results**
```python
class LQRResult(TypedDict):
    """Linear Quadratic Regulator result.
    
    Minimizes: J = ∫(x'Qx + u'Ru)dt  (continuous)
               J = Σ(x'Qx + u'Ru)     (discrete)
    
    Control law: u = -Kx
    """
    gain: GainMatrix                 # Feedback gain K (nu, nx)
    cost_to_go: CovarianceMatrix     # Riccati solution P (nx, nx)
    closed_loop_eigenvalues: np.ndarray  # eig(A - BK)
    stability_margin: float          # Phase/gain margin

class KalmanFilterResult(TypedDict):
    """Kalman Filter (optimal estimator) result.
    
    System:
        x[k+1] = Ax[k] + Bu[k] + w[k],  w ~ N(0,Q)
        y[k] = Cx[k] + v[k],            v ~ N(0,R)
    
    Estimator: x̂[k+1] = Ax̂[k] + Bu[k] + L(y[k] - Cx̂[k])
    """
    gain: GainMatrix                 # Kalman gain L (nx, ny)
    error_covariance: CovarianceMatrix  # Error cov P (nx, nx)
    innovation_covariance: CovarianceMatrix  # Innovation S (ny, ny)
    observer_eigenvalues: np.ndarray  # eig(A - LC)

class LQGResult(TypedDict):
    """Linear Quadratic Gaussian controller result.
    
    Combines LQR (optimal control) + Kalman (optimal estimation)
    via separation principle.
    
    Controller: u = -Kx̂
    Estimator: x̂[k+1] = Ax̂[k] + Bu[k] + L(y[k] - Cx̂[k])
    """
    control_gain: GainMatrix         # LQR gain K (nu, nx)
    estimator_gain: GainMatrix       # Kalman gain L (nx, ny)
    control_cost_to_go: CovarianceMatrix  # Controller Riccati P
    estimation_error_covariance: CovarianceMatrix  # Estimator Riccati P
    separation_verified: bool        # Separation principle holds
    closed_loop_stable: bool         # Overall stability
    controller_eigenvalues: np.ndarray  # eig(A - BK)
    estimator_eigenvalues: np.ndarray   # eig(A - LC)
```

**3. Additional Controllers**
```python
class PolePlacementResult(TypedDict):
    """Pole placement (eigenvalue assignment) result.
    
    Design K such that eig(A - BK) = desired poles
    """
    gain: GainMatrix                 # State feedback gain K
    desired_poles: np.ndarray        # Desired eigenvalues
    achieved_poles: np.ndarray       # Actual eig(A - BK)
    is_controllable: bool            # Arbitrary placement possible

class LuenbergerObserverResult(TypedDict):
    """Luenberger observer (deterministic estimator) result.
    
    Observer: x̂˙ = Ax̂ + Bu + L(y - Cx̂)
    Error dynamics: e˙ = (A - LC)e
    """
    gain: GainMatrix                 # Observer gain L (nx, ny)
    desired_poles: np.ndarray        # Desired observer poles
    achieved_poles: np.ndarray       # Actual eig(A - LC)
    is_observable: bool              # Arbitrary placement possible
```

**Usage Examples:**
```python
# Stability analysis
stability: StabilityInfo = analyze_stability(A, system_type='continuous')
if stability['is_stable']:
    print(f"Stable with spectral radius {stability['spectral_radius']:.3f}")

# LQR design
lqr: LQRResult = system.control.design_lqr(A, B, Q, R, system_type='continuous')
K = lqr['gain']
closed_loop_A = A - B @ K

# Kalman filter design
kalman: KalmanFilterResult = system.control.design_kalman(
    A, C, Q_process, R_measurement, system_type='discrete'
)
L = kalman['gain']

# LQG controller (combined)
lqg: LQGResult = system.control.design_lqg(
    A, B, C, Q, R, Q_process, R_measurement, system_type='discrete'
)
K = lqg['control_gain']
L = lqg['estimator_gain']

# Controllability check
ctrl: ControllabilityInfo = analyze_controllability(A, B)
if ctrl['is_controllable']:
    print(f"System is controllable with rank {ctrl['rank']}")
```

**Key Design:**

- **TypedDict results** - Structured, type-safe returns
- **Mathematical clarity** - Names match control theory
- **Complete information** - All relevant analysis data
- **Separation principle** - LQG designed independently
- **Stability guarantees** - Eigenvalues included
- **IDE support** - Autocomplete for all fields

---

### Structural Types

#### protocols.py
**File:** `protocols.py` (1,086 lines)

**Purpose:** Abstract interfaces via Protocol

**Categories:**

**1. System Protocols**
```python
class DynamicalSystemProtocol(Protocol):
    """Abstract interface for dynamical systems."""
    @property
    def nx(self) -> int: ...
    @property
    def nu(self) -> int: ...
    def __call__(self, x: StateVector, u: ControlVector) -> StateVector: ...

class ContinuousSystemProtocol(DynamicalSystemProtocol, Protocol):
    """Continuous-time system interface."""
    def integrate(
        self,
        x0: StateVector,
        u_func: Callable,
        t_span: TimeSpan
    ) -> IntegrationResult: ...

class DiscreteSystemProtocol(DynamicalSystemProtocol, Protocol):
    """Discrete-time system interface."""
    @property
    def dt(self) -> float: ...
    def step(self, x: StateVector, u: ControlVector) -> StateVector: ...
    def simulate(
        self,
        x0: StateVector,
        u_seq: ControlSequence,
        steps: int
    ) -> StateTrajectory: ...

class StochasticSystemProtocol(Protocol):
    """Stochastic system interface."""
    @property
    def nw(self) -> int: ...
    def drift(self, x: StateVector, u: ControlVector) -> StateVector: ...
    def diffusion(self, x: StateVector, u: ControlVector) -> DiffusionMatrix: ...
```

**2. Observer Protocols**
```python
class ObserverProtocol(Protocol):
    """State observer interface."""
    def observe(self, x: StateVector) -> OutputVector: ...
    def estimate(self, y: OutputVector, u: ControlVector) -> StateVector: ...
```

**3. Controller Protocols**
```python
class ControllerProtocol(Protocol):
    """Controller interface."""
    def compute_control(self, x: StateVector) -> ControlVector: ...

class FeedbackControllerProtocol(ControllerProtocol, Protocol):
    """Linear feedback controller."""
    @property
    def K(self) -> GainMatrix: ...
```

**Key Design:**

- **Structural subtyping** - Duck typing with type safety
- **Interface documentation** - Clear contracts
- **Composition** - Protocols compose naturally
- **No inheritance** - Structural not nominal

---

#### utilities.py
**File:** `utilities.py` (1,132 lines)

**Purpose:** Helper types and utilities

**Categories:**

**1. Type Guards**
```python
def is_numpy(arr: ArrayLike) -> bool:
    """Check if array is NumPy."""
    return isinstance(arr, np.ndarray)

def is_torch(arr: ArrayLike) -> bool:
    """Check if array is PyTorch."""
    return hasattr(arr, '__module__') and 'torch' in arr.__module__

def is_jax(arr: ArrayLike) -> bool:
    """Check if array is JAX."""
    return hasattr(arr, '__module__') and 'jax' in arr.__module__
```

**2. Shape Utilities**
```python
def is_batched(arr: ArrayLike, expected_dims: int = 1) -> bool:
    """Check if array is batched."""
    return arr.ndim > expected_dims

def get_batch_size(arr: ArrayLike) -> Optional[int]:
    """Get batch size if batched."""
    return arr.shape[0] if is_batched(arr) else None

def get_state_dim(x: StateVector) -> int:
    """Get state dimension."""
    return x.shape[-1] if x.ndim > 0 else 1
```

**3. Performance Types**
```python
class ExecutionStats(TypedDict):
    """Performance statistics."""
    count: int              # Number of calls
    total_time: float       # Total time (seconds)
    avg_time: float         # Average time
    min_time: float         # Fastest call
    max_time: float         # Slowest call
```

**4. Validation Types**
```python
class ValidationResult(TypedDict):
    """Validation result."""
    valid: bool
    errors: List[str]
    warnings: List[str]
    info: Dict[str, Any]
```

**Key Design:**

- **Runtime utilities** - Complement static types
- **Type guards** - Enable type narrowing
- **Performance tracking** - Structured metrics
- **Validation** - Structured error reporting

---

## Type System Design Principles

### 1. Semantic Over Structural
```python
# Bad: Structural (what it is)
def compute(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
    pass

# Good: Semantic (what it means)
def compute_control(x: StateVector, K: GainMatrix) -> ControlVector:
    pass
```

### 2. Backend Agnosticism
```python
# Same function signature works for all backends
def dynamics(x: StateVector, u: ControlVector) -> StateVector:
    # Works with NumPy, PyTorch, JAX
    return f(x, u)
```

### 3. TypedDict for Structured Results
```python
# Bad: Plain dict (no type safety)
def integrate() -> dict:
    return {'t': t, 'x': x, 'success': True}

# Good: TypedDict (type-safe, documented)
def integrate() -> IntegrationResult:
    return {'t': t, 'x': x, 'success': True, ...}
```

### 4. Optional Fields with total=False
```python
class IntegrationResult(TypedDict, total=False):
    # Required fields
    t: TimePoints
    x: StateTrajectory
    success: bool
    
    # Optional fields (adaptive methods only)
    njev: int  # May not be present
    sol: Any   # Dense output (optional)
```

### 5. Polymorphic Types via Union
```python
LinearizationResult = Union[
    Tuple[StateMatrix, InputMatrix],           # Deterministic
    Tuple[StateMatrix, InputMatrix, DiffusionMatrix]  # Stochastic
]

# Single function handles both
def analyze(result: LinearizationResult):
    A, B = result[0], result[1]
    if len(result) == 3:
        G = result[2]  # Stochastic
```

### 6. Protocol for Interfaces
```python
# No inheritance needed - structural typing
class MySystem:
    def __call__(self, x: StateVector, u: ControlVector) -> StateVector:
        return x + u

# Satisfies DynamicalSystemProtocol structurally
system: DynamicalSystemProtocol = MySystem()
```

## Usage Throughout the Framework

### In UI Framework
```python
class ContinuousSymbolicSystem(SymbolicSystemBase, ContinuousSystemBase):
    def __call__(self, x: StateVector, u: Optional[ControlVector] = None) -> StateVector:
        """Evaluate dynamics."""
        return self._dynamics.evaluate(x, u, backend=self._default_backend)
    
    def linearize(
        self,
        x_eq: EquilibriumState,
        u_eq: EquilibriumControl,
        backend: Backend = "numpy"
    ) -> DeterministicLinearization:
        """Compute linearization."""
        return self._linearization.linearize_continuous(x_eq, u_eq, backend)
```

### In Delegation Layer
```python
class DynamicsEvaluator:
    def evaluate(
        self,
        x: StateVector,
        u: Optional[ControlVector],
        backend: Backend
    ) -> StateVector:
        """Evaluate forward dynamics."""
        f_func: DynamicsFunction = self.code_gen.generate_dynamics(backend)
        return f_func(x, u)
    
    def get_stats(self) -> ExecutionStats:
        """Get performance statistics."""
        return {
            'count': self._call_count,
            'total_time': self._total_time,
            'avg_time': self._total_time / self._call_count,
            'min_time': self._min_time,
            'max_time': self._max_time
        }
```

### In Integration Framework
```python
class ScipyIntegrator(IntegratorBase):
    def integrate(
        self,
        x0: StateVector,
        u_func: Callable[[ScalarLike, StateVector], Optional[ControlVector]],
        t_span: TimeSpan,
        t_eval: Optional[TimePoints] = None
    ) -> IntegrationResult:
        """Integrate using scipy."""
        # ... implementation ...
        
        result: IntegrationResult = {
            't': sol.t,
            'x': sol.y.T,
            'success': sol.success,
            'message': sol.message,
            'nfev': sol.nfev,
            'nsteps': sol.nfev,
            'integration_time': elapsed,
            'solver': self.name
        }
        
        # Optional fields
        if hasattr(sol, 'njev'):
            result['njev'] = sol.njev
        
        return result
```

## Key Strengths

1. **Semantic Clarity** - Names convey mathematical meaning
2. **Type Safety** - Static checking prevents errors
3. **IDE Support** - Autocomplete and documentation
4. **Backend Agnostic** - Works with NumPy/PyTorch/JAX
5. **Structured Results** - TypedDict not plain dict
6. **Self-Documenting** - Types encode constraints
7. **Composition** - Types compose naturally
8. **Extensible** - Easy to add new types
9. **Consistent** - Same conventions throughout
10. **Testable** - Type-driven testing

This type system is the **foundation** that enables the clean, type-safe architecture of the entire ControlDESymulation framework!
