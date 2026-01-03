# Type System Quick Reference

## Quick Start

### Import Types

```python
# Core types
from src.types.core import (
    StateVector,
    ControlVector,
    StateMatrix,
    InputMatrix,
    GainMatrix,
)

# Backend types
from src.types.backends import Backend, Device, IntegrationMethod

# Result types
from src.types.trajectories import (
    IntegrationResult,
    StateTrajectory,
    TimePoints,
)

# Linearization types
from src.types.linearization import (
    LinearizationResult,
    DeterministicLinearization,
)

# Control types
from src.types.control_classical import (
    LQRResult,
    KalmanFilterResult,
    LQGResult,
    StabilityInfo,
    ControllabilityInfo,
    ObservabilityInfo,
)
```

## Foundational Types

### Vector Types

```python
# State vector (x)
x: StateVector = np.array([1.0, 0.0])

# Control vector (u)
u: ControlVector = np.array([0.5])

# Output vector (y)
y: OutputVector = system.observe(x)

# Noise vector (w)
w: NoiseVector = np.random.randn(nw)

# Equilibrium point
x_eq: EquilibriumState = np.zeros(nx)
u_eq: EquilibriumControl = np.zeros(nu)

# State derivative
dx: TimeDerivative = system(x, u)

# State deviation
δx: StateIncrement = x - x_eq
δu: ControlIncrement = u - u_eq
```

### Matrix Types

```python
# Dynamics matrices
A: StateMatrix = system.linearize(x_eq, u_eq)[0]  # ∂f/∂x (nx, nx)
B: InputMatrix = system.linearize(x_eq, u_eq)[1]  # ∂f/∂u (nx, nu)

# Observation matrices
C: OutputMatrix = np.eye(nx)                       # ∂h/∂x (ny, nx)
D: FeedthroughMatrix = np.zeros((ny, nu))         # Direct feedthrough

# Control matrices
K: GainMatrix = design_lqr(A, B, Q, R)            # Feedback gain (nu, nx)
Q: CostMatrix = np.eye(nx)                        # State cost (nx, nx)
R: ControlCostMatrix = np.eye(nu)                 # Control cost (nu, nu)

# Stochastic matrices
G: DiffusionMatrix = sde_system.diffusion(x, u)   # Noise (nx, nw)
P: CovarianceMatrix = np.eye(nx)                  # Covariance (nx, nx)
```

### Scalar Types

```python
# Time values
t: ScalarLike = 0.0
dt: ScalarLike = 0.01

# Dimensions
nx: IntegerLike = 3
nu: IntegerLike = 1
ny: IntegerLike = 2
nw: IntegerLike = 1

# System properties
order: int = 2  # Second-order system
```

## Backend Types

### Backend Selection

```python
# Backend type
backend: Backend = 'numpy'  # or 'torch', 'jax'

# Device type
device: Device = 'cuda:0'   # or 'cpu', 'mps'

# Configuration
config: BackendConfig = {
    'backend': 'torch',
    'device': 'cuda:0',
    'dtype': 'float32'
}
```

### Integration Methods

```python
# ODE methods
method: IntegrationMethod = 'RK45'     # scipy
method: IntegrationMethod = 'dopri5'   # torch/jax
method: IntegrationMethod = 'Tsit5'    # Julia

# SDE methods
sde_method: str = 'euler-maruyama'
sde_method: str = 'milstein'
sde_method: str = 'heun'

# Fixed-step methods
fixed_method: str = 'rk4'
fixed_method: str = 'euler'
```

### Noise Types

```python
from src.types.backends import NoiseType, SDEType, ConvergenceType

# Noise structure
noise_type = NoiseType.ADDITIVE
noise_type = NoiseType.MULTIPLICATIVE_DIAGONAL

# SDE interpretation
sde_type: SDEType = 'ito'
sde_type: SDEType = 'stratonovich'

# Convergence type
convergence: ConvergenceType = ConvergenceType.STRONG
convergence: ConvergenceType = ConvergenceType.WEAK
```

## Domain Types

### Trajectories

```python
# State trajectory
trajectory: StateTrajectory = system.simulate(x0, u_seq, steps=100)
# Shape: (101, nx) - includes t=0

# Batched trajectories
batch_traj: StateTrajectory = system.simulate_batch(x0_batch, u_seq)
# Shape: (T, batch, nx)

# Control sequence
u_seq: ControlSequence = np.zeros((100, nu))
# Shape: (T, nu)

# Time points
time: TimePoints = np.linspace(0, 10, 101)
# Shape: (T,)

# Time span
t_span: TimeSpan = (0.0, 10.0)

# Extract components
position = trajectory[:, 0]  # First state component
velocity = trajectory[:, 1]  # Second state component
```

### Integration Results

```python
# ODE integration
result: IntegrationResult = integrator.integrate(
    x0=np.array([1.0, 0.0]),
    u_func=lambda t, x: np.zeros(1),
    t_span=(0.0, 10.0)
)

# Access fields
t: TimePoints = result['t']
x: StateTrajectory = result['x']
success: bool = result['success']
nfev: int = result['nfev']
solver: str = result['solver']

# Optional fields (adaptive)
if 'njev' in result:
    njev: int = result['njev']

# SDE integration
sde_result: SDEIntegrationResult = sde_integrator.integrate(
    x0, u_func, t_span
)

# SDE-specific fields
diffusion_evals: int = sde_result['diffusion_evals']
noise_samples: NoiseVector = sde_result['noise_samples']
n_paths: int = sde_result['n_paths']
```

### Linearization Results

```python
# Deterministic linearization
A, B = system.linearize(x_eq, u_eq)
# A: StateMatrix (nx, nx)
# B: InputMatrix (nx, nu)

# Type annotation
result: DeterministicLinearization = system.linearize(x_eq, u_eq)

# Stochastic linearization
A, B, G = sde_system.linearize(x_eq, u_eq)
# G: DiffusionMatrix (nx, nw)

# Type annotation
result: StochasticLinearization = sde_system.linearize(x_eq, u_eq)

# Polymorphic handling
result: LinearizationResult = system.linearize(x_eq, u_eq)
A, B = result[0], result[1]
if len(result) == 3:
    G = result[2]  # Stochastic

# Output linearization
C, D = system.linearized_observation(x_eq, u_eq)
# C: OutputMatrix (ny, nx)
# D: FeedthroughMatrix (ny, nu)

# Full state-space
A, B, C, D = system.full_linearization(x_eq, u_eq)
```

### Control Design Results

```python
from src.types.control_classical import (
    LQRResult,
    KalmanFilterResult,
    LQGResult,
    StabilityInfo,
    ControllabilityInfo,
    ObservabilityInfo,
)

# LQR controller design
A, B = system.linearize(x_eq, u_eq)
Q = np.diag([10, 1])
R = np.array([[0.1]])

result: LQRResult = system.control.design_lqr(
    A, B, Q, R,
    system_type='continuous'
)

# Access LQR result fields
K: GainMatrix = result['gain']                    # Feedback gain (nu, nx)
P: CovarianceMatrix = result['cost_to_go']        # Riccati solution (nx, nx)
eigs: np.ndarray = result['closed_loop_eigenvalues']  # (A-BK) eigenvalues
margin: float = result['stability_margin']        # Phase/gain margin

# Check stability
is_stable = np.all(np.real(eigs) < 0)  # Continuous
is_stable = np.all(np.abs(eigs) < 1)   # Discrete

# Kalman filter design
C = np.array([[1, 0]])
Q_proc = 0.01 * np.eye(2)
R_meas = np.array([[0.1]])

kalman: KalmanFilterResult = system.control.design_kalman(
    A, C, Q_proc, R_meas,
    system_type='discrete'
)

# Access Kalman result fields
L: GainMatrix = kalman['gain']                     # Kalman gain (nx, ny)
P_est: CovarianceMatrix = kalman['error_covariance']  # Error cov (nx, nx)
S: CovarianceMatrix = kalman['innovation_covariance']  # Innovation (ny, ny)
obs_eigs: np.ndarray = kalman['observer_eigenvalues']  # (A-LC) eigenvalues

# LQG controller (combined)
lqg: LQGResult = system.control.design_lqg(
    A, B, C,
    Q, R,           # LQR weights
    Q_proc, R_meas,  # Kalman noise
    system_type='discrete'
)

# Access LQG result fields
K_lqr: GainMatrix = lqg['control_gain']           # LQR gain (nu, nx)
L_kf: GainMatrix = lqg['estimator_gain']          # Kalman gain (nx, ny)
P_ctrl: CovarianceMatrix = lqg['control_cost_to_go']  # Controller Riccati
P_est: CovarianceMatrix = lqg['estimation_error_covariance']  # Estimator Riccati
stable: bool = lqg['closed_loop_stable']          # Overall stability
separated: bool = lqg['separation_verified']      # Separation principle
ctrl_eigs: np.ndarray = lqg['controller_eigenvalues']  # (A-BK) eigenvalues
est_eigs: np.ndarray = lqg['estimator_eigenvalues']    # (A-LC) eigenvalues
```

### System Analysis Results

```python
# Stability analysis
stability: StabilityInfo = analyze_stability(A, system_type='continuous')

# Access stability fields
eigs: np.ndarray = stability['eigenvalues']          # Complex eigenvalues
mags: np.ndarray = stability['magnitudes']           # |λ| values
max_mag: float = stability['max_magnitude']          # Spectral radius
rho: float = stability['spectral_radius']            # Same as max_magnitude
is_stable: bool = stability['is_stable']             # Asymptotically stable
is_marginal: bool = stability['is_marginally_stable']  # On stability boundary
is_unstable: bool = stability['is_unstable']         # Unstable

# Controllability analysis
ctrl: ControllabilityInfo = analyze_controllability(A, B)

# Access controllability fields
C_matrix: ControllabilityMatrix = ctrl['controllability_matrix']  # (nx, nx*nu)
rank: int = ctrl['rank']                             # Rank of C_matrix
controllable: bool = ctrl['is_controllable']         # rank == nx
if 'uncontrollable_modes' in ctrl:
    uncontrol_modes: np.ndarray = ctrl['uncontrollable_modes']  # Eigenvalues

# Observability analysis
obs: ObservabilityInfo = analyze_observability(A, C)

# Access observability fields
O_matrix: ObservabilityMatrix = obs['observability_matrix']  # (nx*ny, nx)
rank: int = obs['rank']                              # Rank of O_matrix
observable: bool = obs['is_observable']              # rank == nx
if 'unobservable_modes' in obs:
    unobs_modes: np.ndarray = obs['unobservable_modes']  # Eigenvalues
```

## Symbolic Types

```python
import sympy as sp
from src.types.symbolic import (
    SymbolicVariable,
    SymbolicVector,
    DynamicsExpression,
    ParameterDict,
)

# Variables
x, v = sp.symbols('x v', real=True)
u = sp.symbols('u', real=True)

# Parameters
m, k = sp.symbols('m k', positive=True)
params: ParameterDict = {m: 1.0, k: 10.0}

# Dynamics expression
f_sym: DynamicsExpression = sp.Matrix([
    v,
    -k*x/m + u/m
])

# Output expression
h_sym: OutputExpression = sp.Matrix([x])

# Diffusion expression
g_sym: DiffusionExpression = sp.Matrix([[0], [0.1]])
```

## Structural Types

### Protocols

```python
from src.types.protocols import (
    DynamicalSystemProtocol,
    ContinuousSystemProtocol,
    StochasticSystemProtocol,
)

# Use in type hints
def analyze(system: DynamicalSystemProtocol):
    """Works with any system satisfying the protocol."""
    print(f"State dim: {system.nx}")
    print(f"Control dim: {system.nu}")
    dx = system(np.zeros(system.nx), np.zeros(system.nu))

# No inheritance needed!
class MySystem:
    @property
    def nx(self) -> int:
        return 2
    
    @property
    def nu(self) -> int:
        return 1
    
    def __call__(self, x: StateVector, u: ControlVector) -> StateVector:
        return x + u

# Satisfies protocol structurally
system: DynamicalSystemProtocol = MySystem()
analyze(system)  # Works!
```

### Utilities

```python
from src.types.utilities import (
    is_numpy,
    is_torch,
    is_jax,
    is_batched,
    get_batch_size,
    ExecutionStats,
)

# Type guards
x = torch.tensor([1.0, 0.0])
if is_torch(x):
    # Type narrowed to torch.Tensor
    x_cuda = x.cuda()

# Shape utilities
x_batch = np.random.randn(100, 2)
if is_batched(x_batch):
    batch_size = get_batch_size(x_batch)  # 100

# Performance tracking
stats: ExecutionStats = {
    'count': 1000,
    'total_time': 0.5,
    'avg_time': 0.0005,
    'min_time': 0.0003,
    'max_time': 0.001
}
```

## Common Patterns

### Pattern 1: Type-Safe Function Signatures

```python
def compute_control(
    x: StateVector,
    K: GainMatrix,
    x_eq: EquilibriumState
) -> ControlVector:
    """Compute LQR feedback control."""
    return -K @ (x - x_eq)

# Usage
x = np.array([1.0, 0.5])
K = np.array([[1.0, 0.5]])
x_eq = np.zeros(2)
u = compute_control(x, K, x_eq)  # Type-safe!
```

### Pattern 2: Backend-Agnostic Code

```python
def dynamics(
    x: StateVector,
    u: ControlVector,
    backend: Backend
) -> StateVector:
    """Works with NumPy, PyTorch, JAX."""
    # Backend-specific operations auto-detected
    if backend == 'torch':
        return torch.sin(x) + u
    elif backend == 'jax':
        return jnp.sin(x) + u
    else:
        return np.sin(x) + u
```

### Pattern 3: Structured Results

```python
def integrate_and_analyze(
    system,
    x0: StateVector,
    t_span: TimeSpan
) -> Dict[str, Any]:
    """Integration with structured output."""
    result: IntegrationResult = system.integrate(
        x0, lambda t, x: np.zeros(system.nu), t_span
    )
    
    return {
        'trajectory': result['x'],
        'time': result['t'],
        'success': result['success'],
        'performance': {
            'nfev': result['nfev'],
            'time': result['integration_time']
        }
    }
```

### Pattern 4: Polymorphic Linearization

```python
def stability_analysis(
    system: Union[ContinuousSystemProtocol, StochasticSystemProtocol],
    x_eq: EquilibriumState,
    u_eq: EquilibriumControl
) -> Dict[str, Any]:
    """Works with deterministic AND stochastic systems."""
    result: LinearizationResult = system.linearize(x_eq, u_eq)
    
    A = result[0]
    eigenvalues = np.linalg.eigvals(A)
    
    analysis = {
        'eigenvalues': eigenvalues,
        'stable': np.all(np.real(eigenvalues) < 0),
        'is_stochastic': len(result) == 3
    }
    
    if len(result) == 3:
        G = result[2]
        analysis['process_noise_cov'] = G @ G.T
    
    return analysis
```

### Pattern 5: Type Guards for Runtime Checks

```python
from src.types.utilities import is_batched

def process(x: StateVector) -> StateVector:
    """Handle both single and batched inputs."""
    if is_batched(x):
        # x has shape (batch, nx)
        batch_size = get_batch_size(x)
        result = np.array([process_single(x[i]) for i in range(batch_size)])
    else:
        # x has shape (nx,)
        result = process_single(x)
    
    return result
```

## Type Checking

### With mypy

```bash
# Install mypy
pip install mypy

# Check types
mypy src/

# Strict mode
mypy --strict src/systems/
```

### With pyright

```bash
# Install pyright
npm install -g pyright

# Check types
pyright src/
```

### Common Type Errors

**Error: Incompatible types**
```python
# Bad
x: StateVector = [1.0, 0.0]  # List, not array

# Good
x: StateVector = np.array([1.0, 0.0])
```

**Error: Missing required key**
```python
# Bad (missing 'success')
result: IntegrationResult = {'t': t, 'x': x}

# Good
result: IntegrationResult = {
    't': t,
    'x': x,
    'success': True,
    'message': 'Integration succeeded',
    'nfev': 100,
    'nsteps': 50,
    'integration_time': 0.5,
    'solver': 'RK45'
}
```

**Error: Wrong shape**
```python
# Type hint documents expected shape
def compute(A: StateMatrix, x: StateVector) -> StateVector:
    # A should be (nx, nx), x should be (nx,)
    return A @ x

# Bad
A = np.array([[1, 2]])  # (1, 2) - wrong shape!

# Good
A = np.array([[1, 2], [3, 4]])  # (2, 2) - correct!
```

## Advanced Usage

### Custom TypedDict

```python
from typing_extensions import TypedDict

class SimulationConfig(TypedDict, total=False):
    """Custom simulation configuration."""
    x0: StateVector
    t_span: TimeSpan
    method: IntegrationMethod
    rtol: float
    atol: float

# Usage
config: SimulationConfig = {
    'x0': np.array([1.0, 0.0]),
    't_span': (0.0, 10.0),
    'method': 'RK45'
}
```

### Generic Types

```python
from typing import TypeVar, Generic

T = TypeVar('T', np.ndarray, torch.Tensor, jnp.ndarray)

class Trajectory(Generic[T]):
    """Generic trajectory for any backend."""
    def __init__(self, data: T):
        self.data = data
    
    def get_final_state(self) -> T:
        return self.data[-1]

# Usage
np_traj = Trajectory[np.ndarray](np_data)
torch_traj = Trajectory[torch.Tensor](torch_data)
```

### Protocol Composition

```python
from src.types.protocols import (
    DynamicalSystemProtocol,
    ObservableSystemProtocol,
)

class FullSystem(DynamicalSystemProtocol, ObservableSystemProtocol, Protocol):
    """Combined protocol."""
    pass

def process(system: FullSystem):
    """Requires both dynamics and observation."""
    x = np.zeros(system.nx)
    dx = system(x, np.zeros(system.nu))
    y = system.observe(x)
```

## Testing with Types

```python
def test_dynamics_signature():
    """Test that dynamics has correct signature."""
    system = MySystem()
    
    x: StateVector = np.array([1.0, 0.0])
    u: ControlVector = np.array([0.5])
    
    # Type checker ensures this is valid
    dx: StateVector = system(x, u)
    
    # Runtime checks
    assert isinstance(dx, np.ndarray)
    assert dx.shape == (system.nx,)

def test_integration_result():
    """Test integration result structure."""
    result: IntegrationResult = integrator.integrate(x0, u_func, t_span)
    
    # Type checker ensures these keys exist
    assert result['success']
    assert result['t'].shape[0] == result['x'].shape[0]
    assert result['solver'] == 'RK45'
```

## References

- **Architecture:** See `Type_System_Architecture.md`
- **Core Types:** See `src/types/core.py`
- **Backend Types:** See `src/types/backends.py`
- **Result Types:** See `src/types/trajectories.py`
- **Control Types:** See `src/types/control_classical.py`
- **Protocols:** See `src/types/protocols.py`
