# Delegation Layer Quick Reference

## Overview

The Delegation Layer provides specialized services through **composition**. Systems compose these utilities rather than inheriting from them.

## Core Utilities

### BackendManager

**Purpose:** Multi-backend array handling

```python
# Access via system
mgr = system.backend

# Detect backend from array
backend = mgr.detect(array)  # 'numpy', 'torch', 'jax'

# Convert between backends
x_torch = mgr.to_backend(x_numpy, 'torch')
x_jax = mgr.to_backend(x_torch, 'jax')
x_numpy = mgr.to_backend(x_jax, 'numpy')

# Device placement (GPU)
x_cuda = mgr.to_device(x_torch, 'cuda:0')

# Set defaults
mgr.set_default('torch', device='cuda:0')

# Temporary backend switch
with mgr.use_backend('jax'):
    # Operations use JAX
    result = compute()

# Check available backends
print(mgr.available_backends)  # ['numpy', 'torch', 'jax']
```

**Properties:**
```python
mgr.default_backend     # Current default
mgr.preferred_device    # Current device
mgr.available_backends  # List of available
```

---

### CodeGenerator

**Purpose:** Symbolic → numerical code generation with caching

```python
# Access via system (private)
code_gen = system._code_gen

# Generate dynamics function (cached)
f_numpy = code_gen.generate_dynamics('numpy')
f_torch = code_gen.generate_dynamics('torch')
f_jax = code_gen.generate_dynamics('jax', jit=True)

# Generate Jacobians
A_func = code_gen.generate_jacobian_A('numpy')
B_func = code_gen.generate_jacobian_B('numpy')

# Output function (if system has outputs)
h_func = code_gen.generate_output('numpy')
C_func = code_gen.generate_jacobian_C('numpy')

# Compile all backends at once
timings = code_gen.compile_all(backends=['numpy', 'torch', 'jax'])
print(f"NumPy: {timings['numpy']:.3f}s")
print(f"PyTorch: {timings['torch']:.3f}s")
print(f"JAX: {timings['jax']:.3f}s")

# Check cache status
has_numpy = code_gen.has_dynamics('numpy')
has_torch = code_gen.has_dynamics('torch')

# Clear cache (e.g., after parameter change)
code_gen.invalidate_cache()
```

**Caching:**

- Functions cached per backend
- Automatic on first call
- Invalidated on parameter changes
- Symbolic Jacobians computed once, compiled per backend

---

### EquilibriumHandler

**Purpose:** Named equilibrium management

```python
# Access via system
eq = system.equilibria

# Add equilibrium
eq.add(
    'upright',
    x_eq=np.array([np.pi, 0]),
    u_eq=np.zeros(1),
    metadata={'stability': 'unstable'}
)

# Add with verification
eq.add(
    'inverted',
    x_eq=np.array([0, 0]),
    u_eq=np.zeros(1),
    verify_fn=lambda x, u: system(x, u),
    tol=1e-6,
    metadata={'stability': 'stable'}
)

# Get equilibrium in any backend
x_eq_numpy = eq.get_x('upright', backend='numpy')
x_eq_torch = eq.get_x('upright', backend='torch')
x_eq_jax = eq.get_x('upright', backend='jax')

# Get control
u_eq = eq.get_u('upright', backend='numpy')

# Get both
x_eq, u_eq = eq.get_both('upright', backend='torch')

# Set default
eq.set_default('upright')

# Get default (no name needed)
x_eq, u_eq = eq.get_both()  # Uses default

# List all
names = eq.list_names()  # ['origin', 'upright', 'inverted']

# Get metadata
meta = eq.get_metadata('upright')
print(meta['stability'])  # 'unstable'
print(meta.get('verified', False))  # True if verified
```

**Automatic Features:**

- Origin always present at (0, 0)
- Dimension validation on add
- Finite value checking (no NaN/Inf)
- Backend-neutral storage (NumPy)

---

### SymbolicValidator

**Purpose:** System definition validation

```python
# Access via system (used automatically)
validator = system._validator

# Validate system (called automatically in __init__)
result = validator.validate(system)

if result['valid']:
    print("System is valid")
else:
    print(f"Errors: {result['errors']}")
    print(f"Warnings: {result['warnings']}")
```

**What It Checks:**
```python
✓ state_vars is list of sp.Symbol
✓ control_vars is list of sp.Symbol
✓ _f_sym is sp.Matrix
✓ _f_sym has correct dimensions
✓ parameters keys are sp.Symbol (not strings!)
✓ order divides nx evenly
✓ _h_sym dimensions (if defined)
✓ _h_sym doesn't depend on control
```

**Common Errors:**
```python
# ❌ String keys in parameters
self.parameters = {'m': 1.0}  # WRONG!

# ✓ Symbol keys
self.parameters = {m_sym: 1.0}  # Correct

# ❌ Wrong dimensions
self._f_sym = sp.Matrix([x, y])  # nx=3, wrong!

# ✓ Correct dimensions
self._f_sym = sp.Matrix([x, y, z])  # nx=3, correct
```

---

## Deterministic Evaluation

### DynamicsEvaluator

**Purpose:** Forward dynamics evaluation

```python
# Access via system (private)
dynamics = system._dynamics

# Single evaluation (usually via system.__call__)
dx = dynamics.evaluate(x, u, backend='numpy')

# Autonomous system
dx = dynamics.evaluate(x, backend='numpy')  # u=None

# Batched evaluation
x_batch = np.random.randn(100, nx)
u_batch = np.random.randn(100, nu)
dx_batch = dynamics.evaluate(x_batch, u_batch)

# Get performance statistics
stats = dynamics.get_stats()
print(f"Evaluations: {stats['count']}")
print(f"Average time: {stats['avg_time']:.6f}s")
print(f"Total time: {stats['total_time']:.3f}s")
```

**Performance Tracking:**
```python
{
    'count': int,        # Number of evaluations
    'total_time': float, # Total time (seconds)
    'avg_time': float,   # Average per evaluation
    'min_time': float,   # Fastest
    'max_time': float,   # Slowest
}
```

---

### LinearizationEngine

**Purpose:** Jacobian computation

```python
# Access via system (private)
lin = system._linearization

# Continuous linearization (via system.linearize)
A, B = lin.linearize_continuous(x_eq, u_eq, backend='numpy')

# Discrete linearization
Ad, Bd = lin.linearize_discrete(x_eq, u_eq, backend='numpy')

# Get symbolic Jacobians (computed once)
A_sym = lin.get_symbolic_A()
B_sym = lin.get_symbolic_B()

# Pretty print symbolic Jacobians
lin.print_symbolic_jacobians()
```

**Higher-Order Systems:**
For `order=2` systems where state is `[q, q̇]` but only `q̈` is returned:
```python
# System automatically constructs full state-space:
# [q̇, q̈]ᵀ = A[q, q̇]ᵀ + Bu

A, B = system.linearize(x_eq, u_eq)
# A is (nx, nx) = (2, 2)
# B is (nx, nu) = (2, nu)
```

---

### ObservationEngine

**Purpose:** Output evaluation

```python
# Access via system (private)
obs = system._observation

# Check if output defined
if obs.has_output():
    # Evaluate output
    y = obs.observe(x, backend='numpy')
    
    # Batched output
    x_batch = np.random.randn(100, nx)
    y_batch = obs.observe(x_batch)  # (100, ny)
    
    # Output Jacobian
    C = obs.linearize_output(x_eq, backend='numpy')
```

---

## Stochastic Support

### DiffusionHandler

**Purpose:** Diffusion matrix generation for SDEs

```python
# Access via stochastic system
diffusion = system.diffusion_handler

# Generate diffusion function (cached)
g_func = diffusion.generate_diffusion('numpy')

# Evaluate diffusion matrix
g = diffusion.evaluate_diffusion(x, u, backend='numpy')
# Returns: (nx, nw) matrix

# Check noise type
if diffusion.is_additive():
    # Optimized for additive noise
    g_const = diffusion.get_additive_matrix()
    # g doesn't depend on x, u - constant!

if diffusion.is_diagonal():
    # Diagonal diffusion matrix
    print("Independent noise sources")

# Stratonovich correction (if sde_type='stratonovich')
correction = diffusion.compute_stratonovich_correction(x, u)

# Number of Wiener processes
nw = diffusion.nw
```

---

### NoiseCharacterizer

**Purpose:** Automatic noise analysis

```python
# Access via stochastic system
characteristics = system.noise_characteristics

# Noise properties
noise_type = characteristics.noise_type
# Returns: ADDITIVE, MULTIPLICATIVE_DIAGONAL, etc.

is_additive = characteristics.is_additive
is_diagonal = characteristics.is_diagonal
nw = characteristics.nw  # Number of Wiener processes

# Solver recommendations
solvers = characteristics.recommend_solvers('torch')
# For additive: ['heun', 'reversible_heun']
# For multiplicative: ['euler-maruyama']

# Validation
if characteristics.is_valid:
    print("Noise structure valid")
else:
    print(characteristics.validation_message)
```

**Noise Types:**
```
ADDITIVE                # g(x,u) = constant
MULTIPLICATIVE_DIAGONAL # g(x,u) diagonal
MULTIPLICATIVE_SCALAR   # Single Wiener process
MULTIPLICATIVE_GENERAL  # Full matrix
```

---

### SDEValidator

**Purpose:** SDE-specific validation

```python
# Access via stochastic system (used automatically)
sde_val = system._sde_validator

# Validate (called automatically in __init__)
result = sde_val.validate(system)

if result['valid']:
    print("SDE system valid")
else:
    print(f"Errors: {result['errors']}")
```

**What It Checks:**
```python
✓ diffusion_expr is sp.Matrix
✓ Dimensions: (nx, nw)
✓ diffusion_expr uses only state_vars, control_vars
✓ sde_type is 'ito' or 'stratonovich'
✓ Compatibility with drift
✓ No division by zero
```

---

## Common Patterns

### Pattern 1: Backend Switching

```python
import torch

# Switch backend for entire system
system.set_default_backend('torch')
system.set_default_device('cuda:0')

# All operations now use PyTorch/GPU
x = torch.tensor([[1.0, 0.0]], device='cuda:0')
dx = system(x, u=None)  # Returns torch.Tensor on cuda:0

# Equilibria automatically converted
x_eq = system.equilibria.get_x('origin', backend='torch')
# Returns: torch.Tensor on cuda:0
```

### Pattern 2: Performance Profiling

```python
# Run many evaluations
for i in range(10000):
    x = np.random.randn(nx)
    u = np.random.randn(nu)
    dx = system(x, u)

# Get statistics
stats = system._dynamics.get_stats()
print(f"Total evaluations: {stats['count']}")
print(f"Average time: {stats['avg_time']*1e6:.1f} μs")
print(f"Evaluations/sec: {stats['count']/stats['total_time']:.0f}")
```

### Pattern 3: Compilation and Warmup

```python
# Compile all backends upfront
timings = system._code_gen.compile_all(
    backends=['numpy', 'torch', 'jax']
)

# Warmup each backend
for backend in ['numpy', 'torch', 'jax']:
    x = system.backend.to_backend(np.zeros(nx), backend)
    u = system.backend.to_backend(np.zeros(nu), backend)
    _ = system._dynamics.evaluate(x, u, backend=backend)

print("All backends compiled and warmed up")
```

### Pattern 4: Equilibrium Management

```python
# Add multiple equilibria
equilibria_data = [
    ('origin', np.zeros(nx), np.zeros(nu), {'stability': 'unstable'}),
    ('upright', np.array([0, 0]), np.zeros(nu), {'stability': 'stable'}),
    ('inverted', np.array([np.pi, 0]), np.zeros(nu), {'stability': 'unstable'}),
]

for name, x_eq, u_eq, metadata in equilibria_data:
    system.add_equilibrium(name, x_eq, u_eq, **metadata)

# Linearize at each
for name in system.list_equilibria():
    x_eq, u_eq = system.get_equilibrium(name)
    A, B = system.linearize(x_eq, u_eq)
    
    # Stability analysis
    eigenvalues = np.linalg.eigvals(A)
    stable = np.all(np.real(eigenvalues) < 0)  # Continuous
    
    print(f"{name}: {'Stable' if stable else 'Unstable'}")
```

### Pattern 5: Stochastic Noise Analysis

```python
# Create stochastic system
sde_system = MyStochasticSystem()

# Automatic noise characterization
noise = sde_system.noise_characteristics

print(f"Noise type: {noise.noise_type}")
print(f"Additive: {noise.is_additive}")
print(f"Diagonal: {noise.is_diagonal}")
print(f"Number of Wiener processes: {noise.nw}")

# Get solver recommendations
for backend in ['numpy', 'torch', 'jax']:
    if backend in sde_system.backend.available_backends:
        solvers = noise.recommend_solvers(backend)
        print(f"{backend}: {solvers}")

# Use optimized evaluation for additive noise
if sde_system.is_additive_noise():
    # Diffusion is constant - evaluate once
    g_const = sde_system.diffusion_handler.get_additive_matrix()
    print(f"Constant diffusion matrix:\n{g_const}")
```

## Troubleshooting

### Issue: "Parameter keys must be Symbol, not str"

**Problem:**
```python
self.parameters = {'m': 1.0}  # ❌ String key
```

**Solution:**
```python
m_sym = sp.symbols('m', positive=True)
self.parameters = {m_sym: 1.0}  # ✓ Symbol key
```

### Issue: Slow first call, fast subsequent calls

**Explanation:** This is normal - code generation and compilation happen on first call, then results are cached.

**Solution:** Compile upfront if needed:
```python
system._code_gen.compile_all(backends=['numpy', 'torch', 'jax'])
```

### Issue: Backend mismatch errors

**Problem:** Mixing backends in operations

**Solution:** Convert to same backend:
```python
x_torch = system.backend.to_backend(x_numpy, 'torch')
u_torch = system.backend.to_backend(u_numpy, 'torch')
dx_torch = system(x_torch, u_torch)
```

### Issue: Equilibrium verification fails

**Problem:** Equilibrium not actually at equilibrium

**Check:**
```python
x_eq = np.array([...])
u_eq = np.array([...])

# For continuous: should be near zero
dx = system(x_eq, u_eq)
print(f"Max |f(x,u)|: {np.abs(dx).max()}")  # Should be < 1e-6

# For discrete: should equal x_eq
x_next = system.step(x_eq, u_eq)
print(f"Max |f(x,u) - x|: {np.abs(x_next - x_eq).max()}")
```

## Advanced Usage

### Custom Backend Configuration

```python
# Configure backend preferences
system.backend.set_default('torch', device='cuda:0')

# Use different devices for different operations
with system.backend.use_device('cuda:1'):
    # Operations on GPU 1
    dx = system(x, u)

# Check CUDA availability
if 'torch' in system.backend.available_backends:
    import torch
    if torch.cuda.is_available():
        system.backend.set_default('torch', device='cuda:0')
```

### Symbolic Jacobian Analysis

```python
# Get symbolic Jacobians
A_sym = system._linearization.get_symbolic_A()
B_sym = system._linearization.get_symbolic_B()

# Simplify
A_simplified = sp.simplify(A_sym)

# Substitute specific parameter values
params = {m_sym: 1.0, k_sym: 10.0}
A_numerical = A_sym.subs(params)

# Pretty print
from sympy import pprint
pprint(A_sym)
```

### Cache Management

```python
# Check what's cached
has_numpy_f = system._code_gen.has_dynamics('numpy')
has_torch_A = system._code_gen.has_jacobian_A('torch')

# Invalidate cache (forces recompilation)
system._code_gen.invalidate_cache()

# Selective invalidation
system._code_gen._f_funcs['numpy'] = None  # Only NumPy
```

## Testing Your Utilities

```python
def test_delegation_layer():
    """Test that delegation layer is properly set up."""
    system = MySystem()
    
    # Check core utilities exist
    assert hasattr(system, 'backend')
    assert hasattr(system, '_code_gen')
    assert hasattr(system, 'equilibria')
    
    # Check deterministic evaluators
    assert hasattr(system, '_dynamics')
    assert hasattr(system, '_linearization')
    
    # Test backend manager
    x = np.array([1.0, 0.0])
    backend = system.backend.detect(x)
    assert backend == 'numpy'
    
    # Test code generator
    f = system._code_gen.generate_dynamics('numpy')
    assert callable(f)
    
    # Test equilibrium handler
    assert 'origin' in system.list_equilibria()
    
    print("✓ Delegation layer tests passed")

test_delegation_layer()
```

## References

- **Architecture:** See `Delegation_Layer_Architecture.md`
- **Type Definitions:** See `src/types/`
- **UI Framework:** See `UI_Framework_Architecture.md`
