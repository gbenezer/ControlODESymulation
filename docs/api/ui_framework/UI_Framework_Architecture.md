# ControlDESymulation UI Framework Architecture

## Overview

The UI (User Interface) framework consists of **8 core files** organized into a 4-layer architecture that eliminates ~1,800 lines of code duplication while maintaining clean separation of concerns.

## Architecture Layers

```
Layer 0: SymbolicSystemBase (time-domain agnostic)
         |
         +------------------+
         |                  |
Layer 1: ContinuousSystemBase  DiscreteSystemBase
         (continuous)       (discrete)
         |                  |
         +--------+         +--------+
         |        |         |        |
Layer 2: Continuous Discrete  Continuous Discrete
         Symbolic   Symbolic  Stochastic Stochastic
         |          |         |          |
Layer 3: ContinuousStochasticSystem  DiscreteStochasticSystem
         (extends Layer 2)            (extends Layer 2)
```

## Layer Breakdown

### Layer 0: Foundation - SymbolicSystemBase
**File:** `symbolic_system_base.py`

**Purpose:** Time-domain agnostic symbolic machinery

**Responsibilities:**

- Symbolic variable management (state_vars, control_vars, output_vars)
- Parameter handling and substitution
- Code generation via CodeGenerator (multi-backend)
- Backend management (NumPy, PyTorch, JAX)
- Equilibrium point management (add, get, remove, list)
- Configuration persistence (save/load JSON)
- Performance tracking and statistics
- Abstract methods: `define_system()`, `print_equations()`

**Key Design:**

- Uses **composition** for utilities (BackendManager, CodeGenerator, EquilibriumHandler, SymbolicValidator)
- Template method pattern: `__init__` orchestrates define → validate → initialize
- Makes NO assumptions about continuous vs discrete time

**What it does NOT provide:**

- Forward dynamics evaluation (`__call__`, `step`)
- Time integration (`integrate`, `simulate`)
- Linearization computation
- These are left to time-domain-specific layers

---

### Layer 1: Time-Domain Bases

#### ContinuousSystemBase
**File:** `continuous_system_base.py`

**Purpose:** Abstract interface for continuous-time systems

**Core Interface:**
```python
dx/dt = f(x, u, t)  # Continuous dynamics

# Abstract methods (must implement):
- __call__(x, u, t) → dx/dt          # Evaluate dynamics
- integrate(x0, u, t_span) → result  # Numerical integration
- linearize(x_eq, u_eq) → (A, B)     # Jacobian matrices

# Concrete method (provided):
- simulate(x0, controller, t_span, dt) → result  # High-level simulation
```

**Key Features:**

- Flexible control input handling (None, arrays, callables)
- Multi-backend integration support
- Dense output and adaptive stepping
- Comprehensive solver diagnostics

#### DiscreteSystemBase  
**File:** `discrete_system_base.py`

**Purpose:** Abstract interface for discrete-time systems

**Core Interface:**
```python
x[k+1] = f(x[k], u[k], k)  # Discrete dynamics

# Abstract property:
- dt: float  # Sampling period (must implement)

# Abstract methods:
- step(x, u, k) → x_next              # Single step
- simulate(x0, u_sequence, n_steps)   # Multi-step simulation
- linearize(x_eq, u_eq) → (Ad, Bd)    # Discrete Jacobians

# Concrete method (provided):
- rollout(x0, policy, n_steps)  # Closed-loop simulation
```

**Key Features:**

- Time-major array convention: (n_steps, nx)
- State-feedback policy support
- Sampling frequency properties

---

### Layer 2: Concrete Symbolic Implementations

#### ContinuousSymbolicSystem
**File:** `continuous_symbolic_system.py`

**Inheritance:** `SymbolicSystemBase + ContinuousSystemBase`

**Purpose:** Combines symbolic machinery with continuous-time execution

**Key Components:**

- **DynamicsEvaluator:** Evaluates dx/dt = f(x, u)
- **LinearizationEngine:** Computes A = ∂f/∂x, B = ∂f/∂u
- **ObservationEngine:** Evaluates y = h(x), C = ∂h/∂x
- **IntegratorFactory:** Creates appropriate ODE solvers

**Usage Pattern:**
```python
class MySystem(ContinuousSymbolicSystem):
    def define_system(self, param1=1.0):
        # Define state_vars, control_vars
        # Set self._f_sym (drift)
        # Set self.parameters
        # Set self.order
```

#### DiscreteSymbolicSystem
**File:** `discrete_symbolic_system.py` (1,020 lines)

**Inheritance:** `SymbolicSystemBase + DiscreteSystemBase`

**Purpose:** Combines symbolic machinery with discrete-time execution

**Key Components:**

- **DynamicsEvaluator:** Evaluates x[k+1] = f(x[k], u[k])
- **LinearizationEngine:** Computes Ad = ∂f/∂x, Bd = ∂f/∂u
- **ObservationEngine:** Evaluates y[k] = h(x[k])

**Critical Requirement:** Must set `self._dt` in `define_system()`

**Usage Pattern:**
```python
class MyDiscreteSystem(DiscreteSymbolicSystem):
    def define_system(self, dt=0.01, param1=1.0):
        # Define state_vars, control_vars
        # Set self._f_sym (next state function)
        # Set self.parameters
        # Set self._dt  # REQUIRED!
        # Set self.order
```

---

### Layer 3: Stochastic Extensions

#### ContinuousStochasticSystem
**File:** `continuous_stochastic_system.py` (1,103 lines)

**Inheritance:** `ContinuousSymbolicSystem` (single inheritance)

**Purpose:** Adds stochastic differential equation (SDE) support

**Mathematical Form:**
```
dx = f(x, u, t)dt + g(x, u, t)dW

where:
- f: Drift (inherited from parent)
- g: Diffusion matrix (added here)
- dW: Brownian motion increments
```

**Key Components:**

- **DiffusionHandler:** Generates and caches diffusion functions
- **NoiseCharacterizer:** Automatic noise structure analysis
- **SDEValidator:** SDE-specific validation
- **SDEIntegratorFactory:** Stochastic integration methods

**Noise Types (Auto-Detected):**

- ADDITIVE: g(x,u,t) = constant
- MULTIPLICATIVE: g(x,u,t) depends on state
- DIAGONAL: Independent noise sources
- SCALAR: Single Wiener process
- GENERAL: Full coupling

**Usage Pattern:**
```python
class MySDESystem(ContinuousStochasticSystem):
    def define_system(self, sigma=0.5):
        # Define drift (same as continuous symbolic)
        # ... state_vars, _f_sym, parameters, order ...
        
        # ADD: Define diffusion
        self.diffusion_expr = sp.Matrix([[sigma_sym]])
        self.sde_type = 'ito'  # or 'stratonovich'
```

#### DiscreteStochasticSystem
**File:** `discrete_stochastic_system.py` (1,383 lines)

**Inheritance:** `DiscreteSymbolicSystem` (single inheritance)

**Purpose:** Adds stochastic difference equation support

**Mathematical Form:**
```
x[k+1] = f(x[k], u[k]) + g(x[k], u[k])·w[k]

where:
- f: Deterministic dynamics (inherited from parent)
- g: Diffusion matrix
- w[k]: Discrete-time noise
```

**Similar structure to continuous stochastic but for discrete-time**

---

### Special System: DiscretizedSystem
**File:** `discretized_system.py` (4,916 lines!)

**Purpose:** Creates discrete-time approximations from continuous systems

**Key Features:**

- Multiple discretization methods (Euler, RK4, Tustin, etc.)
- Preserves symbolic structure when possible
- Handles both deterministic and stochastic systems
- Automatic validation of discretization accuracy

---

## Design Principles

### 1. Cooperative Multiple Inheritance

- **Layer 2** uses multiple inheritance to combine:
  - Symbolic machinery (SymbolicSystemBase)
  - Time-domain interface (ContinuousSystemBase or DiscreteSystemBase)
- Uses `super().__init__()` for proper MRO (Method Resolution Order)

### 2. Composition Over Inheritance

- Specialized functionality delegated to utility classes:
  - BackendManager
  - CodeGenerator
  - EquilibriumHandler
  - DynamicsEvaluator
  - LinearizationEngine
  - ObservationEngine

### 3. Template Method Pattern

- Base classes define workflow
- Subclasses fill in specific implementations
- `define_system()` is the key extension point

### 4. Separation of Concerns

- **Layer 0:** Symbolic manipulation (time-agnostic)
- **Layer 1:** Time-domain semantics (abstract)
- **Layer 2:** Concrete execution (symbolic + time-domain)
- **Layer 3:** Specialized extensions (stochastic)

### 5. Zero Code Duplication

- The ~1,800 lines previously duplicated between continuous and discrete are now in SymbolicSystemBase
- Stochastic systems extend deterministic via single inheritance
- No repeated code for backend management, code generation, etc.

## Key Abstractions

### Abstract Methods Users Must Implement

**In SymbolicSystemBase subclasses:**
```python
def define_system(self, **params):
    """Define symbolic system.
    
    Must set:
    - self.state_vars: List[sp.Symbol]
    - self.control_vars: List[sp.Symbol]
    - self._f_sym: sp.Matrix (dynamics)
    - self.parameters: Dict[sp.Symbol, float]
    - self.order: int
    
    For discrete: Must also set self._dt
    For stochastic: Must also set self.diffusion_expr
    """

def print_equations(self, simplify=True):
    """Print system equations in readable form."""
```

### System Properties (Automatic)

All systems provide:
```python
- nx: int          # State dimension
- nu: int          # Control dimension  
- ny: int          # Output dimension
- nq: int          # Physical dimension (nx / order)
- order: int       # System order

# Discrete only:
- dt: float        # Sampling period
- sampling_frequency: float

# Stochastic only:
- nw: int          # Number of Wiener processes
- is_additive_noise(): bool
- is_multiplicative_noise(): bool
```

## Backend Support

All systems support multi-backend execution:

- **NumPy:** Default, CPU-based
- **PyTorch:** GPU acceleration, automatic differentiation
- **JAX:** XLA compilation, automatic differentiation
- **Julia (via DiffEqPy):** High-performance ODE/SDE solvers

## Integration Methods

### Continuous Systems (via IntegratorFactory)

- **scipy:** RK45, RK23, DOP853, Radau, BDF, LSODA
- **Julia (DiffEqPy):** Tsit5, Vern7, Vern9, Rodas5, etc.
- **JAX (diffrax):** dopri5, tsit5, heun, etc.
- **PyTorch (torchdiffeq):** dopri5, euler, rk4, etc.

### Stochastic Systems (via SDEIntegratorFactory)

- **sdeint:** euler-maruyama, milstein, etc.
- **torchsde:** euler, heun, srk, reversible_heun
- **JAX implementations:** euler-maruyama with noise analysis

## Usage Examples

### Continuous Deterministic System
```python
from src.systems.base import ContinuousSymbolicSystem
import sympy as sp
import numpy as np

class Pendulum(ContinuousSymbolicSystem):
    def define_system(self, m=1.0, l=0.5, g=9.81, b=0.1):
        theta, omega = sp.symbols('theta omega', real=True)
        u = sp.symbols('u', real=True)
        m_sym, l_sym, g_sym, b_sym = sp.symbols('m l g b', positive=True)
        
        self.state_vars = [theta, omega]
        self.control_vars = [u]
        self._f_sym = sp.Matrix([
            omega,
            -(g_sym/l_sym)*sp.sin(theta) - (b_sym/(m_sym*l_sym**2))*omega + u/(m_sym*l_sym**2)
        ])
        self.parameters = {m_sym: m, l_sym: l, g_sym: g, b_sym: b}
        self.order = 1

# Use
system = Pendulum(m=0.5, l=0.3)
x = np.array([0.1, 0.0])
dx = system(x, u=np.array([0.0]))  # Evaluate dynamics
result = system.integrate(x, u=None, t_span=(0, 5), method='RK45')
A, B = system.linearize(np.zeros(2), np.zeros(1))
```

### Discrete System
```python
class DiscreteLinear(DiscreteSymbolicSystem):
    def define_system(self, a=0.9, b=0.1, dt=0.01):
        x = sp.symbols('x', real=True)
        u = sp.symbols('u', real=True)
        a_sym, b_sym = sp.symbols('a b', real=True)
        
        self.state_vars = [x]
        self.control_vars = [u]
        self._f_sym = sp.Matrix([a_sym*x + b_sym*u])
        self.parameters = {a_sym: a, b_sym: b}
        self._dt = dt  # REQUIRED!
        self.order = 1

# Use
system = DiscreteLinear(a=0.95, dt=0.1)
x_next = system.step(x, u)
result = system.simulate(x0, u_sequence=None, n_steps=100)
```

### Stochastic System (SDE)
```python
class OrnsteinUhlenbeck(ContinuousStochasticSystem):
    def define_system(self, alpha=1.0, sigma=0.5):
        x = sp.symbols('x', real=True)
        u = sp.symbols('u', real=True)
        alpha_sym = sp.symbols('alpha', positive=True)
        sigma_sym = sp.symbols('sigma', positive=True)
        
        # Drift
        self.state_vars = [x]
        self.control_vars = [u]
        self._f_sym = sp.Matrix([[-alpha_sym * x + u]])
        self.parameters = {alpha_sym: alpha, sigma_sym: sigma}
        self.order = 1
        
        # Diffusion
        self.diffusion_expr = sp.Matrix([[sigma_sym]])
        self.sde_type = 'ito'

# Use
system = OrnsteinUhlenbeck(alpha=2.0, sigma=0.3)
print(system.is_additive_noise())  # True
result = system.integrate(x0, u=None, t_span=(0, 10), method='euler-maruyama')
```

## Strengths of This Architecture

1. **Clean separation of concerns** - each layer has a single responsibility
2. **Zero code duplication** - symbolic machinery shared across all systems
3. **Type safety** - comprehensive TypedDict definitions throughout
4. **Backend flexibility** - seamlessly switch between NumPy/PyTorch/JAX/Julia
5. **Extensibility** - easy to add new system types via inheritance
6. **Mathematical rigor** - proper handling of ODEs, SDEs, difference equations
7. **Performance** - multi-backend support enables GPU acceleration
8. **Documentation** - extensive docstrings with mathematical notation

This is production-quality systems engineering with excellent abstraction layers!
