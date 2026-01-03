# ControlDESymulation Numerical Integration Framework Architecture

## Overview

The numerical integration framework provides **multi-backend**, **multi-method** support for integrating both **deterministic ODEs** and **stochastic SDEs**. The framework consists of **13 core files** organized into a clean 2-track architecture: deterministic and stochastic.

## Architecture Tracks

```
Track 1: Deterministic ODE Integration
├── IntegratorBase (abstract)
├── IntegratorFactory (creates integrators)
├── Backend-Specific Implementations:
│   ├── ScipyIntegrator (NumPy)
│   ├── TorchDiffEqIntegrator (PyTorch)
│   ├── DiffraxIntegrator (JAX)
│   └── DiffEqPyIntegrator (Julia)
└── FixedStepIntegrators (RK4, Euler, Midpoint)

Track 2: Stochastic SDE Integration
├── SDEIntegratorBase (extends IntegratorBase)
├── SDEIntegratorFactory (creates SDE integrators)
├── Backend-Specific Implementations:
│   ├── TorchSDEIntegrator (PyTorch)
│   ├── DiffraxSDEIntegrator (JAX)
│   └── DiffEqPySDEIntegrator (Julia)
└── CustomBrownianPath (custom noise support)
```

## File Breakdown

### Track 1: Deterministic ODE Integration

#### IntegratorBase
**File:** `integrator_base.py` (512 lines)

**Purpose:** Abstract base class for all numerical integrators

**Key Features:**
- Defines unified interface for all integrators
- StepMode enum (FIXED vs ADAPTIVE)
- Performance statistics tracking
- TypedDict-based IntegrationResult

**Abstract Methods:**
```python
- step(x, u, dt) → x_next          # Single integration step
- integrate(x0, u_func, t_span) → IntegrationResult  # Multi-step
- name: str                         # Integrator identifier
```

**Attributes:**
```python
- system: ContinuousSystemBase     # System to integrate
- dt: float                        # Time step (or initial guess)
- step_mode: StepMode              # FIXED or ADAPTIVE
- backend: Backend                 # 'numpy', 'torch', 'jax'
- rtol, atol: float                # Error tolerances (adaptive)
- _stats: dict                     # Performance tracking
```

---

#### IntegratorFactory
**File:** `integrator_factory.py` (1,267 lines)

**Purpose:** Factory for creating appropriate integrators

**Key Features:**
- Automatic backend/method selection
- Method-to-backend routing
- Use case-specific helpers
- Comprehensive method registry

**Available Backends:**
- **NumPy:** scipy, DiffEqPy (Julia), manual methods
- **PyTorch:** torchdiffeq
- **JAX:** diffrax

**Factory Methods:**
```python
# Main creation
create(system, backend, method, dt, **options) → IntegratorBase

# Convenience methods
auto(system, backend=None) → IntegratorBase           # Best for backend
for_production(system) → IntegratorBase               # LSODA or AutoTsit5
for_optimization(system) → IntegratorBase             # Diffrax tsit5
for_neural_ode(system) → IntegratorBase              # TorchDiffEq dopri5
for_julia(system, algorithm='Tsit5') → IntegratorBase # Julia solver

# Backend-specific
scipy(system, method='RK45', **opts) → ScipyIntegrator
julia(system, algorithm='Tsit5', **opts) → DiffEqPyIntegrator
torch(system, method='dopri5', **opts) → TorchDiffEqIntegrator
jax(system, method='tsit5', **opts) → DiffraxIntegrator
```

**Method Registry:**

| Method | Backend | Type | Best For |
|--------|---------|------|----------|
| LSODA | NumPy (scipy) | Adaptive | General (auto-stiffness) |
| RK45 | NumPy (scipy) | Adaptive | Non-stiff ODEs |
| Radau | NumPy (scipy) | Adaptive | Stiff ODEs |
| BDF | NumPy (scipy) | Adaptive | Very stiff ODEs |
| Tsit5 | NumPy (Julia) | Adaptive | High performance |
| Vern9 | NumPy (Julia) | Adaptive | High accuracy |
| Rodas5 | NumPy (Julia) | Adaptive | Stiff (Rosenbrock) |
| dopri5 | PyTorch/JAX | Adaptive | Neural ODEs |
| dopri8 | PyTorch/JAX | Adaptive | High accuracy |
| tsit5 | JAX (diffrax) | Adaptive | Optimization |
| euler | Any | Fixed | Educational |
| rk4 | Any | Fixed | Simple systems |

---

#### ScipyIntegrator
**File:** `scipy_integrator.py** (~620 lines estimate)

**Purpose:** Adaptive integration using scipy.integrate.solve_ivp

**Supported Methods:**
- **RK45** (default): Dormand-Prince 5(4) - general purpose
- **RK23**: Bogacki-Shampine 3(2) - fast, low accuracy
- **DOP853**: Dormand-Prince 8(5,3) - high accuracy
- **Radau**: Implicit RK - stiff systems
- **BDF**: Backward differentiation - very stiff
- **LSODA**: Auto stiffness detection

**Key Features:**
- Professional-grade adaptive stepping
- Error control (rtol/atol)
- Dense output (interpolation)
- Event detection
- Both controlled and autonomous systems

**Example:**
```python
integrator = ScipyIntegrator(
    system,
    method='RK45',
    rtol=1e-6,
    atol=1e-8
)
result = integrator.integrate(x0, u_func, t_span=(0, 10))
```

---

#### TorchDiffEqIntegrator
**File:** `torchdiffeq_integrator.py` (estimated ~800 lines)

**Purpose:** PyTorch integration with GPU acceleration and autograd

**Supported Methods:**
- **dopri5**: Dormand-Prince 5(4)
- **dopri8**: Dormand-Prince 8
- **adaptive_heun**: Heun's method
- **bosh3**: Bogacki-Shampine 3
- **fehlberg2**: Fehlberg 2(1)
- **explicit_adams**, **implicit_adams**: Multi-step methods

**Key Features:**
- GPU acceleration
- Automatic differentiation
- Adjoint method for memory efficiency
- Neural ODE support

**Example:**
```python
integrator = TorchDiffEqIntegrator(
    system,
    method='dopri5',
    backend='torch',
    device='cuda:0'
)
result = integrator.integrate(x0, u_func, t_span=(0, 10))
```

---

#### DiffraxIntegrator
**File:** `diffrax_integrator.py` (estimated ~700 lines)

**Purpose:** JAX integration with XLA compilation and autograd

**Supported Methods:**
- **tsit5**: Tsitouras 5(4) - recommended
- **dopri5**: Dormand-Prince 5(4)
- **dopri8**: Dormand-Prince 8
- **heun**: Heun's method
- **ralston**: Ralston's method
- **reversible_heun**: Reversible Heun

**Key Features:**
- XLA compilation
- JAX transformations (jit, vmap, grad)
- Efficient for optimization
- Functional programming style

**Example:**
```python
integrator = DiffraxIntegrator(
    system,
    method='tsit5',
    backend='jax'
)
result = integrator.integrate(x0, u_func, t_span=(0, 10))
```

---

#### DiffEqPyIntegrator
**File:** `diffeqpy_integrator.py` (estimated ~900 lines)

**Purpose:** Julia's DifferentialEquations.jl via Python bindings

**Supported Methods:**
Extensive Julia solver ecosystem:
- **Explicit RK:** Tsit5, Vern6, Vern7, Vern8, Vern9, DP5, DP8
- **Rosenbrock:** Rosenbrock23, Rosenbrock32, Rodas4, Rodas5
- **BDF:** TRBDF2, KenCarp3, KenCarp4, KenCarp5
- **Radau:** RadauIIA5
- **Stabilized:** ROCK2, ROCK4
- **Symplectic:** VelocityVerlet, SymplecticEuler
- **Auto-switching:** AutoTsit5(Rosenbrock23()), AutoVern7(Rodas5())

**Key Features:**
- Highest performance
- Automatic stiffness detection
- Extensive method library
- Production-grade reliability

**Example:**
```python
integrator = DiffEqPyIntegrator(
    system,
    algorithm='Vern9',
    backend='numpy',
    reltol=1e-12,
    abstol=1e-14
)
result = integrator.integrate(x0, u_func, t_span=(0, 10))
```

---

#### FixedStepIntegrators
**File:** `fixed_step_integrators.py` (estimated ~600 lines)

**Purpose:** Manual implementation of fixed-step methods

**Supported Methods:**
- **euler**: Forward Euler (order 1)
- **midpoint**: Midpoint method (order 2)
- **rk4**: Runge-Kutta 4 (order 4)

**Key Features:**
- Backend-agnostic (NumPy, PyTorch, JAX)
- Simple, transparent implementations
- Educational value
- Constant time step

**Example:**
```python
integrator = RK4Integrator(
    system,
    dt=0.01,
    backend='numpy'
)
result = integrator.integrate(x0, u_func, t_span=(0, 10))
```

---

### Track 2: Stochastic SDE Integration

#### SDEIntegratorBase
**File:** `sde_integrator_base.py` (1,080 lines)

**Purpose:** Abstract base for SDE integrators

**Mathematical Form:**
```
dx = f(x, u, t)dt + g(x, u, t)dW
```
where:
- f: Drift (deterministic)
- g: Diffusion (stochastic intensity)
- dW: Brownian motion increments

**Key Differences from ODE:**
- Random noise generation
- Weak vs strong convergence
- Noise structure exploitation
- Monte Carlo simulation support
- Itô vs Stratonovich interpretation

**Abstract Methods:**
```python
- step(x, u, dt, dW) → x_next      # Single SDE step with noise
- integrate(x0, u_func, t_span) → SDEIntegrationResult
- integrate_monte_carlo(x0, u_func, t_span, n_paths) → SDEIntegrationResult
```

**Result Type:**
```python
SDEIntegrationResult = {
    't': TimePoints,               # Time grid
    'x': StateVector,              # Trajectory (T, nx) or (n_paths, T, nx)
    'success': bool,
    'message': str,
    'nfev': int,                   # Drift evaluations
    'diffusion_evals': int,        # Diffusion evaluations
    'noise_samples': NoiseVector,  # Brownian increments
    'n_paths': int,                # Number of trajectories
    'convergence_type': str,       # 'strong' or 'weak'
    'sde_type': str,               # 'ito' or 'stratonovich'
    ...
}
```

---

#### SDEIntegratorFactory
**File:** `sde_integrator_factory.py` (estimated ~1,000 lines)

**Purpose:** Factory for creating SDE integrators

**Factory Methods:**
```python
create(sde_system, backend, method, dt, **options) → SDEIntegratorBase
auto(sde_system, backend=None) → SDEIntegratorBase
for_monte_carlo(sde_system, n_paths) → SDEIntegratorBase
```

**Available Methods:**

| Method | Backend | Convergence | Noise Type |
|--------|---------|-------------|------------|
| euler-maruyama | NumPy/PyTorch/JAX | Strong (0.5) | General |
| milstein | NumPy | Strong (1.0) | Diagonal |
| heun | PyTorch/JAX | Strong (1.0) | Additive |
| srk | PyTorch | Strong | General |
| reversible_heun | PyTorch | Strong | Additive |

---

#### TorchSDEIntegrator
**File:** `torchsde_integrator.py` (estimated ~800 lines)

**Purpose:** PyTorch SDE integration with torchsde

**Supported Methods:**
- **euler**: Euler-Maruyama (strong order 0.5)
- **heun**: Heun's method (strong order 1.0 for additive)
- **srk**: Stochastic Runge-Kutta
- **reversible_heun**: Reversible Heun (strong order 1.0)

**Key Features:**
- GPU acceleration
- Adaptive stepping
- Noise structure exploitation
- Adjoint method for gradients

**Example:**
```python
integrator = TorchSDEIntegrator(
    sde_system,
    method='heun',
    dt=0.01,
    backend='torch'
)
result = integrator.integrate(x0, u_func, t_span=(0, 10))
```

---

#### DiffraxSDEIntegrator
**File:** `diffrax_sde_integrator.py` (estimated ~750 lines)

**Purpose:** JAX SDE integration with diffrax

**Supported Methods:**
- **euler**: Euler-Maruyama
- **heun**: Heun's method
- **reversible_heun**: Reversible Heun

**Key Features:**
- JAX transformations
- XLA compilation
- Custom noise support
- Efficient for optimization

**Example:**
```python
integrator = DiffraxSDEIntegrator(
    sde_system,
    method='euler',
    dt=0.01,
    backend='jax',
    seed=42
)
result = integrator.integrate(x0, u_func, t_span=(0, 10))
```

---

#### DiffEqPySDEIntegrator
**File:** `diffeqpy_sde_integrator.py` (estimated ~850 lines)

**Purpose:** Julia SDE solvers via DiffEqPy

**Supported Methods:**
- Euler-Maruyama variants
- Milstein
- Stochastic Rosenbrock
- Advanced Julia SDE methods

**Key Features:**
- Production-grade SDE solvers
- Automatic noise structure detection
- High-performance algorithms

---

#### CustomBrownianPath
**File:** `custom_brownian.py` (160 lines)

**Purpose:** Custom Brownian motion for deterministic testing

**Key Features:**
- User-provided noise increments
- Diffrax AbstractPath interface
- Deterministic testing support
- Custom noise patterns

**Example:**
```python
# Zero noise for testing
dW = jnp.zeros(1)
brownian = CustomBrownianPath(0.0, 0.01, dW)

# Or random noise
brownian = create_custom_or_random_brownian(
    key, 0.0, 0.01, shape=(nw,), dW=None
)
```

---

## Design Principles

### 1. Backend Abstraction
All integrators work across backends:
- **NumPy:** CPU, scipy, Julia integration
- **PyTorch:** GPU, autograd, neural ODEs
- **JAX:** XLA, functional, optimization

### 2. Factory Pattern
IntegratorFactory and SDEIntegratorFactory provide:
- Automatic method selection
- Backend-specific routing
- Convenience constructors
- Use case optimization

### 3. Unified Result Types
All integrators return TypedDict results:
- **IntegrationResult** for ODEs
- **SDEIntegrationResult** for SDEs
- Consistent fields across backends
- Optional fields for extra diagnostics

### 4. Composition Not Inheritance
- Integrators compose with systems
- No deep inheritance hierarchies
- Clear separation of concerns
- Easy to extend

### 5. Performance Tracking
Built-in statistics:
```python
integrator._stats = {
    'total_fev': int,      # Function evaluations
    'total_steps': int,    # Integration steps
    'total_time': float,   # Computation time
}
```

## Integration Result Types

### IntegrationResult (ODE)
```python
{
    't': array,              # Time points (T,)
    'x': array,              # States (T, nx) - time-major
    'success': bool,         # Integration succeeded
    'message': str,          # Status message
    'nfev': int,             # Function evaluations
    'nsteps': int,           # Steps taken
    'integration_time': float,  # Wall time (seconds)
    'solver': str,           # Integrator name
    
    # Optional (adaptive methods):
    'njev': int,             # Jacobian evaluations
    'nlu': int,              # LU decompositions
    'status': int,           # Solver status code
    'sol': object,           # Dense output (if requested)
    'dense_output': bool,    # Dense output available
}
```

### SDEIntegrationResult (SDE)
```python
{
    # All IntegrationResult fields, plus:
    'diffusion_evals': int,     # Diffusion function calls
    'noise_samples': array,     # Brownian increments used
    'n_paths': int,             # Number of trajectories
    'convergence_type': str,    # 'strong' or 'weak'
    'sde_type': str,            # 'ito' or 'stratonovich'
    'noise_type': str,          # 'additive', 'multiplicative', etc.
    
    # For Monte Carlo (n_paths > 1):
    'x': array,                 # (n_paths, T, nx)
    'statistics': dict,         # mean, std, quantiles
}
```

## Usage Examples

### Example 1: Simple ODE Integration (NumPy)
```python
from src.systems.base import ContinuousSymbolicSystem
from src.systems.base.numerical_integration import IntegratorFactory
import numpy as np

# Create system (e.g., pendulum)
system = Pendulum(m=1.0, l=0.5)

# Create integrator automatically
integrator = IntegratorFactory.auto(system)

# Integrate
result = integrator.integrate(
    x0=np.array([0.1, 0.0]),
    u_func=lambda t, x: np.zeros(1),
    t_span=(0.0, 10.0)
)

print(f"Method: {result['solver']}")
print(f"Steps: {result['nsteps']}")
print(f"Success: {result['success']}")
```

### Example 2: GPU-Accelerated Neural ODE (PyTorch)
```python
import torch
from src.systems.base.numerical_integration import IntegratorFactory

# Create system and move to GPU
system = MyNeuralODE()
system.set_default_backend('torch')
system.set_default_device('cuda:0')

# Create Neural ODE integrator
integrator = IntegratorFactory.for_neural_ode(system)

# Integrate on GPU
x0 = torch.tensor([[1.0, 0.0]], device='cuda:0')
result = integrator.integrate(
    x0=x0,
    u_func=lambda t, x: torch.zeros(1, device='cuda:0'),
    t_span=(0.0, 10.0)
)
```

### Example 3: High-Accuracy Julia Integration
```python
from src.systems.base.numerical_integration import IntegratorFactory

# Create Julia integrator with high accuracy
integrator = IntegratorFactory.for_julia(
    system,
    algorithm='Vern9',  # 9th order
    reltol=1e-12,
    abstol=1e-14
)

result = integrator.integrate(x0, u_func, t_span=(0, 100))
print(f"Accurate to {result['nfev']} function evaluations")
```

### Example 4: Stochastic SDE Integration
```python
from src.systems.base import ContinuousStochasticSystem
from src.systems.base.numerical_integration.stochastic import SDEIntegratorFactory

# Create SDE system (e.g., Ornstein-Uhlenbeck)
sde_system = OrnsteinUhlenbeck(alpha=2.0, sigma=0.3)

# Create SDE integrator
integrator = SDEIntegratorFactory.create(
    sde_system,
    backend='torch',
    method='heun',
    dt=0.01,
    seed=42
)

# Single trajectory
result = integrator.integrate(x0, u_func, t_span=(0, 10))
print(f"Noise type: {result['noise_type']}")
print(f"SDE type: {result['sde_type']}")
```

### Example 5: Monte Carlo SDE Simulation
```python
# Multiple trajectories for uncertainty quantification
result = integrator.integrate_monte_carlo(
    x0=np.array([1.0]),
    u_func=lambda t, x: None,
    t_span=(0, 10),
    n_paths=1000
)

# Get statistics
from src.systems.base.numerical_integration.sde_integrator_base import (
    get_trajectory_statistics
)
stats = get_trajectory_statistics(result)

print(f"Mean at t=10: {stats['mean'][-1]}")
print(f"Std at t=10: {stats['std'][-1]}")
print(f"95% CI: [{stats['q25'][-1]}, {stats['q75'][-1]}]")
```

### Example 6: Custom Noise (Deterministic Testing)
```python
import jax.numpy as jnp
from src.systems.base.numerical_integration.stochastic.custom_brownian import (
    CustomBrownianPath
)

# Zero noise for deterministic testing
dW = jnp.zeros((nw,))
brownian = CustomBrownianPath(0.0, 0.01, dW)

# Use in integration
result = integrator.integrate(
    x0, u_func, t_span=(0, 1),
    brownian_path=brownian
)
```

## Integrator Selection Guide

### By Use Case

| Use Case | Recommended Integrator | Reason |
|----------|----------------------|---------|
| General ODE | `IntegratorFactory.for_production(system)` | LSODA auto-stiffness |
| Neural ODE | `IntegratorFactory.for_neural_ode(system)` | TorchDiffEq adjoint |
| Optimization | `IntegratorFactory.for_optimization(system)` | Diffrax JAX |
| High Accuracy | `IntegratorFactory.for_julia(system, 'Vern9')` | Julia 9th order |
| Stiff ODE | `ScipyIntegrator(system, method='BDF')` | Implicit BDF |
| Simple ODE | `RK4Integrator(system, dt=0.01)` | Classic RK4 |
| SDE (general) | `SDEIntegratorFactory.auto(sde_system)` | Best for noise type |
| Monte Carlo | `SDEIntegratorFactory.for_monte_carlo(...)` | Parallelized |

### By Backend

| Backend | ODE Integrator | SDE Integrator |
|---------|---------------|----------------|
| NumPy | ScipyIntegrator, DiffEqPyIntegrator | (limited support) |
| PyTorch | TorchDiffEqIntegrator | TorchSDEIntegrator |
| JAX | DiffraxIntegrator | DiffraxSDEIntegrator |

### By System Properties

| System Type | Best Integrator |
|-------------|-----------------|
| Non-stiff | RK45, Tsit5, dopri5 |
| Stiff | BDF, Radau, Rodas5 |
| High accuracy | Vern9, DOP853 |
| Real-time | RK4, euler (fixed-step) |
| Additive noise SDE | Heun (strong order 1.0) |
| General SDE | Euler-Maruyama |

## File Size Summary

### Deterministic Track
| File | Lines | Purpose |
|------|-------|---------|
| integrator_base.py | 512 | Abstract base |
| integrator_factory.py | 1,267 | Factory pattern |
| scipy_integrator.py | ~620 | NumPy (scipy) |
| torchdiffeq_integrator.py | ~800 | PyTorch |
| diffrax_integrator.py | ~700 | JAX |
| diffeqpy_integrator.py | ~900 | Julia |
| fixed_step_integrators.py | ~600 | Manual methods |

### Stochastic Track
| File | Lines | Purpose |
|------|-------|---------|
| sde_integrator_base.py | 1,080 | SDE abstract base |
| sde_integrator_factory.py | ~1,000 | SDE factory |
| torchsde_integrator.py | ~800 | PyTorch SDE |
| diffrax_sde_integrator.py | ~750 | JAX SDE |
| diffeqpy_sde_integrator.py | ~850 | Julia SDE |
| custom_brownian.py | 160 | Custom noise |

**Total: ~10,000+ lines** of production-ready integration code

## Key Strengths

1. **Multi-Backend Support** - Seamless NumPy/PyTorch/JAX switching
2. **Extensive Method Library** - 40+ integration methods
3. **Factory Pattern** - Automatic method selection
4. **Type Safety** - TypedDict results with IDE support
5. **Performance** - GPU acceleration, XLA compilation, Julia performance
6. **Flexibility** - Fixed and adaptive stepping
7. **Stochastic Support** - Full SDE integration framework
8. **Noise Structure** - Exploits additive/diagonal/scalar noise
9. **Monte Carlo** - Built-in multi-trajectory simulation
10. **Testing** - Custom noise for deterministic testing

This framework enables state-of-the-art numerical integration for control theory and machine learning applications!
