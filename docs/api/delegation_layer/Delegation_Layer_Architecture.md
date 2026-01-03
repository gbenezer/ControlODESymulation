# ControlDESymulation Delegation Layer Architecture

## Overview

The **Delegation Layer** (also called the **Support Framework**) provides **specialized services** to the UI framework through **composition rather than inheritance**. This layer consists of **11 focused utility classes** totaling ~7,200 lines that handle backend management, code generation, dynamics evaluation, and stochastic analysis.

## Architecture Philosophy

**Composition Over Inheritance** - Instead of creating deep inheritance hierarchies, the UI framework composes these specialized utilities:

```python
# In ContinuousSymbolicSystem.__init__():
self.backend = BackendManager(default_backend, default_device)
self._code_gen = CodeGenerator(self, self.backend)
self._dynamics = DynamicsEvaluator(self, self._code_gen, self.backend)
self._linearization = LinearizationEngine(self, self._code_gen, self.backend)
self._observation = ObservationEngine(self, self._code_gen, self.backend)
self.equilibria = EquilibriumHandler(nx, nu)
```

This design provides:

- **Single Responsibility** - Each class does one thing well
- **Reusability** - Utilities can be used independently
- **Testability** - Each component tested in isolation
- **Flexibility** - Easy to swap implementations
- **Clarity** - Clear separation of concerns

## Architecture Layers

```
┌──────────────────────────────────────────────────────────────┐
│                     UI FRAMEWORK                              │
│  (SymbolicSystemBase, ContinuousSymbolicSystem, etc.)        │
└────────────────────┬─────────────────────────────────────────┘
                     │ uses (composition)
                     ↓
┌──────────────────────────────────────────────────────────────┐
│                  DELEGATION LAYER                             │
│                                                                │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  CORE UTILITIES (Universal)                         │    │
│  │  • BackendManager      - Multi-backend support      │    │
│  │  • CodeGenerator       - Symbolic → numerical       │    │
│  │  • EquilibriumHandler  - Equilibrium management     │    │
│  │  • SymbolicValidator   - System validation          │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                                │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  DETERMINISTIC EVALUATION (ODE Systems)             │    │
│  │  • DynamicsEvaluator    - Forward dynamics          │    │
│  │  • LinearizationEngine  - Jacobian computation      │    │
│  │  • ObservationEngine    - Output evaluation         │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                                │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  STOCHASTIC SUPPORT (SDE Systems)                   │    │
│  │  • DiffusionHandler    - Diffusion generation       │    │
│  │  • NoiseCharacterizer  - Noise analysis             │    │
│  │  • SDEValidator        - SDE validation             │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                                │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  LOW-LEVEL UTILITIES                                │    │
│  │  • codegen_utils       - SymPy code generation      │    │
│  └─────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────┘
```

## Component Breakdown

### Core Utilities (Universal)

#### BackendManager
**File:** `backend_manager.py` (545 lines)

**Purpose:** Multi-backend array handling and device management

**Responsibilities:**

- Backend detection from array types
- Array conversion between backends (NumPy ↔ PyTorch ↔ JAX)
- Backend availability checking
- Device management (CPU, CUDA, TPU)
- Default backend configuration
- Temporary backend switching (context manager)

**Key Features:**
```python
# Detect backend from array
backend = mgr.detect(array)  # Returns 'numpy', 'torch', or 'jax'

# Convert between backends
x_torch = mgr.to_backend(x_numpy, 'torch')
x_jax = mgr.to_backend(x_torch, 'jax')

# Device placement
x_cuda = mgr.to_device(x, 'cuda:0')

# Temporary backend switching
with mgr.use_backend('jax'):
    # Operations use JAX
    result = compute()
```

**Supported Backends:**

- **NumPy:** CPU-based, universal compatibility
- **PyTorch:** GPU acceleration, autograd, neural networks
- **JAX:** XLA compilation, functional programming, optimization

**Properties:**
```python
mgr.default_backend     # Current default backend
mgr.preferred_device    # Current device (cpu, cuda:0, etc.)
mgr.available_backends  # List of available backends
```

---

#### CodeGenerator
**File:** `code_generator.py` (565 lines)

**Purpose:** Orchestrates symbolic → numerical code generation with caching

**Responsibilities:**

- Generate dynamics functions: f(x, u) → dx/dt
- Generate output functions: h(x) → y
- Generate Jacobian functions: A, B, C
- Per-backend function caching
- Compilation and warmup
- Cache invalidation on parameter changes

**Key Features:**
```python
# Generate dynamics function (cached)
f_numpy = code_gen.generate_dynamics('numpy')
f_torch = code_gen.generate_dynamics('torch')
f_jax = code_gen.generate_dynamics('jax', jit=True)

# Generate Jacobians
A_func = code_gen.generate_jacobian_A('numpy')
B_func = code_gen.generate_jacobian_B('numpy')

# Output function
h_func = code_gen.generate_output('numpy')
C_func = code_gen.generate_jacobian_C('numpy')

# Compile all backends at once
timings = code_gen.compile_all(backends=['numpy', 'torch', 'jax'])
```

**Caching Strategy:**

- Functions cached per backend
- Symbolic Jacobians computed once, then compiled per backend
- Cache invalidated on parameter changes
- Automatic recompilation when needed

**Backend-Specific Optimizations:**

- **NumPy:** Uses `lambdify` with CSE (common subexpression elimination)
- **PyTorch:** Simplifies expressions before generation
- **JAX:** JIT compilation via `jax.jit`

---

#### EquilibriumHandler
**File:** `equilibrium_handler.py` (221 lines)

**Purpose:** Manages multiple named equilibrium points

**Responsibilities:**

- Store equilibria as NumPy arrays (backend-neutral)
- Convert to any backend on demand
- Named equilibrium management
- Equilibrium verification
- Metadata storage (stability, description, etc.)
- Default equilibrium selection

**Key Features:**
```python
# Add equilibrium with verification
equilibria.add(
    'upright',
    x_eq=np.array([np.pi, 0]),
    u_eq=np.zeros(1),
    verify_fn=lambda x, u: system(x, u),
    metadata={'stability': 'unstable'}
)

# Get in any backend
x_eq_numpy = equilibria.get_x('upright', backend='numpy')
x_eq_torch = equilibria.get_x('upright', backend='torch')

# Set default
equilibria.set_default('upright')

# Get both state and control
x_eq, u_eq = equilibria.get_both('upright', backend='jax')

# List all
names = equilibria.list_names()  # ['origin', 'upright', ...]

# Get metadata
meta = equilibria.get_metadata('upright')
print(meta['stability'])  # 'unstable'
```

**Automatic Features:**

- Origin equilibrium always present
- Dimension validation on add
- Finite value checking (no NaN/Inf)
- Optional verification with tolerance

---

#### SymbolicValidator
**File:** `symbolic_validator.py` (718 lines)

**Purpose:** Validates symbolic system definitions

**Responsibilities:**

- Check state/control variable consistency
- Validate symbolic expression dimensions
- Verify parameter keys (Symbol, not string)
- Check system order compatibility
- Ensure output function validity
- Detect common errors early

**Validation Checks:**
```python
# Required validations
✓ state_vars is list of sp.Symbol
✓ control_vars is list of sp.Symbol  
✓ _f_sym is sp.Matrix with correct dimensions
✓ parameters keys are sp.Symbol (not strings)
✓ order divides nx evenly (nx % order == 0)

# Output function validations (if defined)
✓ _h_sym is sp.Matrix
✓ _h_sym only depends on state_vars (not control)
✓ output_vars matches _h_sym dimensions

# Error messages are clear and actionable
```

**Example Errors Caught:**
```python
# Bad: String keys
self.parameters = {'m': 1.0}  # ❌ String key
# Good: Symbol keys  
self.parameters = {m_sym: 1.0}  # ✓ Symbol key

# Bad: Wrong dimensions
self._f_sym = sp.Matrix([x, y])  # nx=2, but system has nx=3
# Good: Correct dimensions
self._f_sym = sp.Matrix([x, y, z])  # ✓ nx=3
```

---

### Deterministic Evaluation

#### DynamicsEvaluator
**File:** `dynamics_evaluator.py` (576 lines)

**Purpose:** Forward dynamics evaluation across backends

**Responsibilities:**

- Evaluate dx/dt = f(x, u) for controlled systems
- Evaluate dx/dt = f(x) for autonomous systems (u=None)
- Handle batched vs single evaluation
- Backend-specific dispatch
- Input shape validation
- Performance tracking

**Key Features:**
```python
# Single evaluation
dx = evaluator.evaluate(x, u, backend='numpy')

# Autonomous system (u=None)
dx = evaluator.evaluate(x, backend='numpy')

# Batched evaluation
x_batch = np.random.randn(100, nx)  # (batch, nx)
u_batch = np.random.randn(100, nu)  # (batch, nu)
dx_batch = evaluator.evaluate(x_batch, u_batch)  # (batch, nx)

# Get performance stats
stats = evaluator.get_stats()
print(f"Total evaluations: {stats['count']}")
print(f"Average time: {stats['avg_time']:.6f}s")
```

**Backend Dispatch:**

- Automatically detects input backend
- Uses cached compiled functions
- Handles shape mismatches gracefully
- Converts outputs to match input backend

**Performance Tracking:**
```python
{
    'count': int,        # Number of evaluations
    'total_time': float, # Total time (seconds)
    'avg_time': float,   # Average per evaluation
    'min_time': float,   # Fastest evaluation
    'max_time': float,   # Slowest evaluation
}
```

---

#### LinearizationEngine
**File:** `linearization_engine.py` (907 lines)

**Purpose:** Compute linearizations (Jacobians) at equilibria

**Responsibilities:**

- Compute continuous Jacobians: A = ∂f/∂x, B = ∂f/∂u
- Compute discrete Jacobians: Ad = ∂f/∂x, Bd = ∂f/∂u
- Handle higher-order systems (order > 1)
- Automatic state-space construction
- Symbolic and numerical Jacobians
- Backend-agnostic evaluation

**Mathematical Forms:**

**Continuous Systems:**
```
Linearization at (x_eq, u_eq):
δẋ = A·δx + B·δu

where:
A = ∂f/∂x|(x_eq, u_eq)  ∈ ℝ^(nx × nx)
B = ∂f/∂u|(x_eq, u_eq)  ∈ ℝ^(nx × nu)
```

**Discrete Systems:**
```
Linearization at (x_eq, u_eq):
δx[k+1] = Ad·δx[k] + Bd·δu[k]

where:
Ad = ∂f/∂x|(x_eq, u_eq)  ∈ ℝ^(nx × nx)
Bd = ∂f/∂u|(x_eq, u_eq)  ∈ ℝ^(nx × nu)
```

**Key Features:**
```python
# Continuous linearization
A, B = engine.linearize_continuous(x_eq, u_eq, backend='numpy')

# Discrete linearization
Ad, Bd = engine.linearize_discrete(x_eq, u_eq, backend='numpy')

# Higher-order systems (automatic state-space)
# For order=2 system: q̈ = f(q, q̇, u)
# Automatically constructs: [q̇, q̈]ᵀ = A[q, q̇]ᵀ + Bu
A, B = engine.linearize_continuous(x_eq, u_eq)

# Get symbolic Jacobians (one-time computation)
A_sym = engine.get_symbolic_A()
B_sym = engine.get_symbolic_B()
```

**Higher-Order Handling:**

For `order=n` systems where state is `[q, q̇, ..., q^(n-1)]` and only `q^(n)` is returned:

1. Automatically constructs full state derivative
2. Computes Jacobian of full state-space form
3. Returns proper (nx × nx) and (nx × nu) matrices

---

#### ObservationEngine
**File:** `observation_engine.py` (628 lines)

**Purpose:** Output function evaluation and linearization

**Responsibilities:**

- Evaluate output: y = h(x)
- Compute output Jacobian: C = ∂h/∂x
- Batched output evaluation
- Multi-backend support
- Performance tracking

**Mathematical Form:**
```
Output equation:
y = h(x)

Output linearization:
δy = C·δx

where:
C = ∂h/∂x|(x_eq)  ∈ ℝ^(ny × nx)
```

**Key Features:**
```python
# Evaluate output
y = engine.observe(x, backend='numpy')

# Batched evaluation
x_batch = np.random.randn(100, nx)
y_batch = engine.observe(x_batch)  # (100, ny)

# Output Jacobian
C = engine.linearize_output(x_eq, backend='numpy')

# Check if output defined
if engine.has_output():
    y = engine.observe(x)
```

**Validation:**

- Ensures h(x) doesn't depend on control u
- Checks dimension consistency
- Validates symbolic expressions

---

### Stochastic Support

#### DiffusionHandler
**File:** `diffusion_handler.py` (1,069 lines)

**Purpose:** Generate and cache diffusion functions for SDEs

**Responsibilities:**

- Generate diffusion matrix: g(x, u) ∈ ℝ^(nx × nw)
- Automatic noise structure detection
- Specialized functions for additive/diagonal/scalar noise
- Multi-backend diffusion compilation
- Stratonovich correction computation
- Performance optimization

**Mathematical Form:**
```
SDE: dx = f(x, u)dt + g(x, u)dW

Diffusion matrix g(x, u):
- Shape: (nx, nw)
- nx: state dimension
- nw: number of independent Wiener processes

Noise types (auto-detected):
- ADDITIVE: g(x, u) = constant
- MULTIPLICATIVE: g depends on x or u
- DIAGONAL: g is diagonal (independent noise)
- SCALAR: nw = 1 (single Wiener process)
```

**Key Features:**
```python
# Generate diffusion function
g_func = handler.generate_diffusion('numpy')

# Evaluate diffusion
g_matrix = handler.evaluate_diffusion(x, u, backend='numpy')

# Automatic noise detection
noise_type = handler.get_noise_type()
# Returns: ADDITIVE, MULTIPLICATIVE_DIAGONAL, etc.

# Stratonovich correction
correction = handler.compute_stratonovich_correction(x, u)

# Optimized for additive noise
if handler.is_additive():
    g_const = handler.get_additive_matrix()  # Constant matrix
```

**Noise Structure Exploitation:**

- **Additive (constant):** Returns constant matrix, no recomputation needed
- **Multiplicative types:**

  - **Diagonal:** Independent noise channels, specialized handling
  - **Scalar:** Single Wiener process (nw=1), simplified operations
  - **General:** Full matrix coupling, complete computation

---

#### NoiseCharacterizer
**File:** `noise_analysis.py` (692 lines)

**Purpose:** Automatic noise structure analysis

**Responsibilities:**

- Detect noise types (additive, multiplicative, etc.)
- Identify noise structure (diagonal, scalar, general)
- Recommend optimal SDE solvers
- Compute noise statistics
- Validate noise properties

**Noise Classification:**
```python
NoiseType:
├─ ADDITIVE               # g(x,u) = constant
├─ MULTIPLICATIVE         # g depends on x or u
│  ├─ DIAGONAL            # Independent noise sources
│  ├─ SCALAR              # Single Wiener process
│  └─ GENERAL             # Full coupling
└─ UNKNOWN                # Cannot determine
```

**Key Features:**
```python
# Automatic characterization
characteristics = characterizer.analyze()

# Access properties
noise_type = characteristics.noise_type
is_additive = characteristics.is_additive
is_diagonal = characteristics.is_diagonal
nw = characteristics.nw  # Number of Wiener processes

# Solver recommendations
solvers = characteristics.recommend_solvers('torch')
# Returns: ['heun', 'reversible_heun'] for additive
# Returns: ['euler-maruyama'] for multiplicative

# Validation
if not characteristics.is_valid:
    print(characteristics.validation_message)
```

**Solver Recommendations:**

| Noise Type | Backend | Recommended Solvers |
|------------|---------|-------------------|
| Additive | PyTorch | heun, reversible_heun |
| Additive | JAX | heun, reversible_heun |
| Multiplicative Diagonal | NumPy | milstein |
| General | All | euler-maruyama |

---

#### SDEValidator
**File:** `sde_validator.py` (544 lines)

**Purpose:** SDE-specific validation

**Responsibilities:**

- Validate diffusion matrix dimensions
- Check SDE type (Itô vs Stratonovich)
- Ensure compatibility with drift
- Verify noise structure claims
- Validate stochastic system definition

**Validation Checks:**
```python
✓ diffusion_expr is sp.Matrix
✓ Dimensions: (nx, nw) where nw ≥ 1
✓ diffusion_expr only uses state_vars and control_vars
✓ sde_type is 'ito' or 'stratonovich'
✓ Compatibility with drift _f_sym
✓ No division by zero in diffusion terms
✓ Finite symbolic expressions
```

**Example Errors Caught:**
```python
# Bad: Wrong dimensions
self.diffusion_expr = sp.Matrix([[sigma]])  # nx=2, but only 1 row
# Good: Correct dimensions
self.diffusion_expr = sp.Matrix([[sigma], [0]])  # (2, 1)

# Bad: Invalid SDE type
self.sde_type = 'unknown'  # ❌
# Good: Valid type
self.sde_type = 'ito'  # ✓
```

---

### Low-Level Utilities

#### codegen_utils
**File:** `codegen_utils.py` (733 lines)

**Purpose:** Low-level SymPy → executable code conversion

**Responsibilities:**

- SymPy Matrix → callable function
- Backend-specific code generation
- Common subexpression elimination (CSE)
- Symbolic simplification strategies
- NumPy/PyTorch/JAX code generation
- Function signature handling

**Key Functions:**
```python
# Main generation function
func = generate_function(
    expr,           # sp.Matrix symbolic expression
    input_vars,     # List of sp.Symbol variables
    backend='numpy', # Target backend
    simplify=True,  # Apply simplification
    cse=True,       # Common subexpression elimination
    jit=False       # JIT compile (JAX only)
)

# Backend-specific optimizations
func_numpy = generate_function(expr, vars, backend='numpy', cse=True)
func_torch = generate_function(expr, vars, backend='torch', simplify=True)
func_jax = generate_function(expr, vars, backend='jax', jit=True)
```

**Optimization Strategies:**

**NumPy:**

- Common subexpression elimination (CSE)
- Fast numerical modules ('numpy', 'scipy')
- Matrix operations optimization

**PyTorch:**

- Symbolic simplification before generation
- Automatic differentiation compatibility
- GPU tensor operations

**JAX:**

- JIT compilation via `jax.jit`
- Pure functional style
- XLA optimization
- Automatic vectorization

**Generated Function Signatures:**
```python
# Dynamics function
f(x, u) → dx/dt
# where x: array(nx,), u: array(nu,) → array(nx,)

# Jacobian function
A(x, u) → matrix(nx, nx)
B(x, u) → matrix(nx, nu)

# Output function
h(x) → y
# where x: array(nx,) → array(ny,)
```

---

## Composition Patterns

### Pattern 1: Core System Utilities
**Used by:** All symbolic systems

```python
# In SymbolicSystemBase.__init__():
self.backend = BackendManager(default_backend, default_device)
self._code_gen = CodeGenerator(self, self.backend)
self._validator = SymbolicValidator()
self.equilibria = EquilibriumHandler(nx, nu)
```

### Pattern 2: Deterministic Extensions
**Used by:** ContinuousSymbolicSystem, DiscreteSymbolicSystem

```python
# In ContinuousSymbolicSystem.__init__():
super().__init__(*args, **kwargs)  # Sets up core utilities

# Add deterministic evaluators
self._dynamics = DynamicsEvaluator(self, self._code_gen, self.backend)
self._linearization = LinearizationEngine(self, self._code_gen, self.backend)
self._observation = ObservationEngine(self, self._code_gen, self.backend)
```

### Pattern 3: Stochastic Extensions
**Used by:** ContinuousStochasticSystem, DiscreteStochasticSystem

```python
# In ContinuousStochasticSystem.__init__():
super().__init__(*args, **kwargs)  # Sets up deterministic utilities

# Add stochastic support
self.diffusion_handler = DiffusionHandler(self, self._code_gen, self.backend)
self.noise_characteristics = NoiseCharacterizer().analyze(self.diffusion_expr)
self._sde_validator = SDEValidator()
```

## Design Principles

### 1. Single Responsibility

Each class has one clear purpose:

- BackendManager → Backend management ONLY
- CodeGenerator → Code generation ONLY
- DynamicsEvaluator → Dynamics evaluation ONLY

### 2. Composition Not Inheritance
Systems compose utilities rather than inherit:
```python
# NOT: class System(BackendManager, CodeGenerator, ...)
# YES: class System:
#          self.backend = BackendManager()
#          self._code_gen = CodeGenerator()
```

### 3. Dependency Injection
Utilities receive dependencies via constructor:
```python
DynamicsEvaluator(system, code_gen, backend_mgr)
```

### 4. Interface Segregation

Each utility has focused, minimal interface:

- BackendManager: detect, convert, to_backend
- CodeGenerator: generate_dynamics, generate_jacobian_A/B/C
- DynamicsEvaluator: evaluate

### 5. Lazy Initialization
Functions generated and cached on first use:
```python
# First call: generates and caches
f = code_gen.generate_dynamics('numpy')

# Subsequent calls: returns cached
f_again = code_gen.generate_dynamics('numpy')
assert f is f_again  # Same object
```

### 6. Backend Agnosticism
Utilities work with all backends transparently:
```python
# Same interface, different backends
dx_numpy = evaluator.evaluate(x_np, u_np, backend='numpy')
dx_torch = evaluator.evaluate(x_torch, u_torch, backend='torch')
dx_jax = evaluator.evaluate(x_jax, u_jax, backend='jax')
```

## Usage Examples

### Example 1: Creating a System with Full Delegation

```python
from src.systems.base import ContinuousSymbolicSystem
import sympy as sp
import numpy as np

class MySystem(ContinuousSymbolicSystem):
    def define_system(self, m=1.0, k=10.0):
        x, v = sp.symbols('x v', real=True)
        u = sp.symbols('u', real=True)
        m_sym, k_sym = sp.symbols('m k', positive=True)
        
        self.state_vars = [x, v]
        self.control_vars = [u]
        self._f_sym = sp.Matrix([v, -k_sym*x/m_sym + u/m_sym])
        self.parameters = {m_sym: m, k_sym: k}
        self.order = 1

# System automatically sets up:
system = MySystem()
# system.backend         (BackendManager)
# system._code_gen       (CodeGenerator)
# system._dynamics       (DynamicsEvaluator)
# system._linearization  (LinearizationEngine)
# system.equilibria      (EquilibriumHandler)

# Use the delegated functionality
x = np.array([1.0, 0.0])
u = np.array([0.5])

# Dynamics (via DynamicsEvaluator)
dx = system(x, u)

# Linearization (via LinearizationEngine)
A, B = system.linearize(np.zeros(2), np.zeros(1))

# Equilibria (via EquilibriumHandler)
system.add_equilibrium('test', x_eq=np.zeros(2), u_eq=np.zeros(1))
```

### Example 2: Backend Switching

```python
import torch

# Switch backend for entire system
system.set_default_backend('torch')
system.set_default_device('cuda:0')

# Backend manager handles conversions
x_torch = torch.tensor([[1.0, 0.0]], device='cuda:0')
u_torch = torch.tensor([[0.5]], device='cuda:0')

# Automatically uses PyTorch backend
dx_torch = system(x_torch, u_torch)

# Equilibria converted on demand
x_eq_torch = system.equilibria.get_x('origin', backend='torch')
```

### Example 3: Code Generation and Caching

```python
# First call: generates and caches
import time

start = time.time()
f_numpy = system._code_gen.generate_dynamics('numpy')
t1 = time.time() - start
print(f"First generation: {t1:.3f}s")

# Second call: returns cached (instant)
start = time.time()
f_numpy_again = system._code_gen.generate_dynamics('numpy')
t2 = time.time() - start
print(f"Cached retrieval: {t2:.6f}s")  # ~0.000001s

assert f_numpy is f_numpy_again  # Same function object
```

### Example 4: Stochastic System with Full Delegation

```python
from src.systems.base import ContinuousStochasticSystem

class StochasticOscillator(ContinuousStochasticSystem):
    def define_system(self, k=10.0, sigma=0.5):
        x, v = sp.symbols('x v', real=True)
        u = sp.symbols('u', real=True)
        k_sym, sigma_sym = sp.symbols('k sigma', positive=True)
        
        # Drift
        self.state_vars = [x, v]
        self.control_vars = [u]
        self._f_sym = sp.Matrix([v, -k_sym*x + u])
        self.parameters = {k_sym: k, sigma_sym: sigma}
        self.order = 1
        
        # Diffusion (additive noise)
        self.diffusion_expr = sp.Matrix([[0], [sigma_sym]])
        self.sde_type = 'ito'

# System sets up additional stochastic delegation:
sde_system = StochasticOscillator()
# sde_system.diffusion_handler       (DiffusionHandler)
# sde_system.noise_characteristics   (NoiseCharacteristics)
# (plus all deterministic utilities)

# Use stochastic functionality
x = np.array([1.0, 0.0])
u = np.array([0.5])

# Drift (via DynamicsEvaluator)
f = sde_system.drift(x, u)

# Diffusion (via DiffusionHandler)
g = sde_system.diffusion(x, u)

# Noise analysis (automatic)
print(f"Noise type: {sde_system.noise_characteristics.noise_type}")
print(f"Additive: {sde_system.is_additive_noise()}")
print(f"Recommended solvers: {sde_system.recommend_solvers('torch')}")
```

### Example 5: Performance Tracking

```python
# DynamicsEvaluator tracks performance
for i in range(1000):
    dx = system(np.random.randn(2), np.zeros(1))

# Get statistics
stats = system._dynamics.get_stats()
print(f"Evaluations: {stats['count']}")
print(f"Average time: {stats['avg_time']:.6f}s")
print(f"Total time: {stats['total_time']:.3f}s")
```

## Key Strengths

1. **Clean Separation** - Each utility has one responsibility
2. **Reusability** - Can use utilities independently
3. **Testability** - Easy to test in isolation
4. **Flexibility** - Easy to add new utilities
5. **Performance** - Caching and lazy initialization
6. **Multi-Backend** - Seamless backend switching
7. **Type Safety** - TypedDict and semantic types
8. **Documentation** - Clear purpose and interface
9. **Maintainability** - Easy to understand and modify
10. **Extensibility** - Composition enables new features

This delegation layer is the foundation that makes the UI framework clean, powerful, and maintainable!
