# ControlDESymulation UI Framework Quick Reference

## Quick Start Guide

### 1. Continuous Deterministic System

```python
from src.systems.base import ContinuousSymbolicSystem
import sympy as sp
import numpy as np

class MySystem(ContinuousSymbolicSystem):
    """Your custom continuous system."""
    
    def define_system(self, param1=1.0, param2=2.0):
        """Define symbolic dynamics."""
        # Step 1: Define symbolic variables
        x1, x2 = sp.symbols('x1 x2', real=True)
        u = sp.symbols('u', real=True)
        p1, p2 = sp.symbols('p1 p2', positive=True)
        
        # Step 2: Set state and control variables
        self.state_vars = [x1, x2]
        self.control_vars = [u]
        
        # Step 3: Define dynamics: dx/dt = f(x, u)
        self._f_sym = sp.Matrix([
            x2,
            -p1*x1 - p2*x2 + u
        ])
        
        # Step 4: Set parameters
        self.parameters = {p1: param1, p2: param2}
        
        # Step 5: Set system order (1 for first-order state-space)
        self.order = 1
        
        # Optional: Define output function y = h(x)
        self._h_sym = sp.Matrix([x1])  # Measure position only
        self.output_vars = ['y']

# Create and use
system = MySystem(param1=10.0, param2=0.5)

# Evaluate dynamics at a point
x = np.array([1.0, 0.0])
u = np.array([0.5])
dx = system(x, u)  # Returns dx/dt

# Integrate over time
result = system.integrate(
    x0=x,
    u=None,  # Zero control (autonomous)
    t_span=(0.0, 10.0),
    method='RK45'
)
t = result['t']
x_traj = result['y']  # scipy returns (nx, T)

# Linearize at equilibrium
x_eq = np.zeros(2)
u_eq = np.zeros(1)
A, B = system.linearize(x_eq, u_eq)
print(f"A matrix:\n{A}")
print(f"B matrix:\n{B}")

# Simulate with controller
def controller(x, t):
    K = np.array([[-1.0, -2.0]])  # LQR gain
    return -K @ x

sim_result = system.simulate(
    x0=x,
    controller=controller,
    t_span=(0, 10),
    dt=0.01
)
```

### 2. Discrete System

```python
from src.systems.base import DiscreteSymbolicSystem
import sympy as sp
import numpy as np

class MyDiscreteSystem(DiscreteSymbolicSystem):
    """Your custom discrete system."""
    
    def define_system(self, a=0.9, b=0.1, dt=0.01):
        """Define discrete dynamics."""
        # Define variables
        x = sp.symbols('x', real=True)
        u = sp.symbols('u', real=True)
        a_sym, b_sym = sp.symbols('a b', real=True)
        
        # State and control
        self.state_vars = [x]
        self.control_vars = [u]
        
        # Discrete dynamics: x[k+1] = f(x[k], u[k])
        self._f_sym = sp.Matrix([a_sym*x + b_sym*u])
        
        # Parameters
        self.parameters = {a_sym: a, b_sym: b}
        
        # CRITICAL: Must set dt for discrete systems!
        self._dt = dt
        
        # System order
        self.order = 1

# Create and use
system = MyDiscreteSystem(a=0.95, b=0.05, dt=0.1)

# Single step
x_k = np.array([1.0])
u_k = np.array([0.5])
x_next = system.step(x_k, u_k)

# Multi-step simulation
result = system.simulate(
    x0=np.array([1.0]),
    u_sequence=np.zeros((100, 1)),  # Constant zero control
    n_steps=100
)
states = result['states']  # (n_steps+1, nx)
controls = result['controls']  # (n_steps, nu)

# Rollout with state feedback
def policy(x, k):
    K = np.array([[-0.8]])
    return -K @ x

result = system.rollout(
    x0=np.array([1.0]),
    policy=policy,
    n_steps=100
)

# Linearize
Ad, Bd = system.linearize(np.zeros(1), np.zeros(1))
eigenvalues = np.linalg.eigvals(Ad)
is_stable = np.all(np.abs(eigenvalues) < 1)
```

### 3. Stochastic System (SDE)

```python
from src.systems.base import ContinuousStochasticSystem
import sympy as sp
import numpy as np

class OrnsteinUhlenbeck(ContinuousStochasticSystem):
    """Ornstein-Uhlenbeck process: mean-reverting stochastic process."""
    
    def define_system(self, alpha=1.0, sigma=0.5):
        """Define SDE: dx = -alpha*x*dt + sigma*dW"""
        # Define variables
        x = sp.symbols('x', real=True)
        u = sp.symbols('u', real=True)
        alpha_sym = sp.symbols('alpha', positive=True)
        sigma_sym = sp.symbols('sigma', positive=True)
        
        # State and control
        self.state_vars = [x]
        self.control_vars = [u]
        
        # Drift term: f(x, u) in dx = f*dt + g*dW
        self._f_sym = sp.Matrix([[-alpha_sym * x + u]])
        
        # Parameters
        self.parameters = {alpha_sym: alpha, sigma_sym: sigma}
        self.order = 1
        
        # STOCHASTIC: Define diffusion term g(x, u)
        self.diffusion_expr = sp.Matrix([[sigma_sym]])
        
        # SDE type: 'ito' or 'stratonovich'
        self.sde_type = 'ito'

# Create and use
system = OrnsteinUhlenbeck(alpha=2.0, sigma=0.3)

# Check noise type (automatic analysis)
print(f"Additive noise: {system.is_additive_noise()}")  # True
print(f"Noise characteristics: {system.noise_characteristics}")

# Evaluate drift and diffusion
x = np.array([1.0])
u = np.array([0.0])
f = system.drift(x, u)  # Drift term
g = system.diffusion(x, u)  # Diffusion matrix

# Integrate SDE
result = system.integrate(
    x0=x,
    u=None,
    t_span=(0, 10),
    method='euler-maruyama',  # SDE integrator
    dt=0.01  # Required for SDE methods
)

# Get solver recommendations
solvers = system.recommend_solvers(backend='jax')
print(f"Recommended solvers: {solvers}")
```

### 4. Higher-Order Systems

```python
class SecondOrderSystem(ContinuousSymbolicSystem):
    """Example: Double integrator (2nd order)."""
    
    def define_system(self, m=1.0):
        # Physical variable
        q = sp.symbols('q', real=True)
        u = sp.symbols('u', real=True)
        m_sym = sp.symbols('m', positive=True)
        
        # State is [q, q_dot] but we define just the physical variable
        self.state_vars = [q]  # Will be expanded to [q, q_dot]
        self.control_vars = [u]
        
        # Define ONLY highest derivative: q_ddot
        self._f_sym = sp.Matrix([u / m_sym])
        
        self.parameters = {m_sym: m}
        self.order = 2  # Second-order system!
```

## Common Operations

### Backend Management

```python
# Set default backend
system.set_default_backend('torch')  # 'numpy', 'torch', 'jax'

# Set device
system.set_default_device('cuda:0')  # For GPU

# Check backend
print(system._default_backend)

# Convert arrays
x_torch = system.backend.to_backend(x_numpy, 'torch')
```

### Equilibrium Management

```python
# Add equilibrium point
system.add_equilibrium(
    'upright',
    x_eq=np.array([np.pi, 0]),
    u_eq=np.zeros(1),
    metadata={'stability': 'unstable'}
)

# Get equilibrium
x_eq, u_eq = system.get_equilibrium('upright')

# List all equilibria
names = system.list_equilibria()  # ['origin', 'upright']

# Set default
system.set_default_equilibrium('upright')

# Get metadata
meta = system.get_equilibrium_metadata('upright')
print(meta['stability'])  # 'unstable'
```

### Configuration Persistence

```python
# Save configuration
system.save_config('my_system_config.json')

# Get config dict
config = system.get_config_dict()
print(config['parameters'])
```

### Performance Statistics

```python
# Get statistics
stats = system.stats.summary()
print(stats)

# Reset statistics
system.stats.reset()
```

### Code Compilation

```python
# Force recompilation (e.g., after parameter change)
system.compile()

# Check if compiled
if system._is_compiled:
    print("System is compiled")
```

## Common Patterns

### Pattern 1: Parameter Study

```python
# Create systems with different parameters
systems = [MySystem(param1=p) for p in np.linspace(1, 10, 10)]

# Simulate each
results = []
for sys in systems:
    result = sys.integrate(x0, u=None, t_span=(0, 10))
    results.append(result)
```

### Pattern 2: Closed-Loop Control

```python
# Design LQR controller
A, B = system.linearize(x_eq, u_eq)
Q = np.eye(system.nx)
R = np.eye(system.nu)

from scipy.linalg import solve_continuous_are
P = solve_continuous_are(A, B, Q, R)
K = np.linalg.inv(R) @ B.T @ P

# Define controller
def lqr_controller(x, t):
    return -K @ (x - x_eq)

# Simulate
result = system.simulate(
    x0=x0,
    controller=lqr_controller,
    t_span=(0, 10),
    dt=0.01
)
```

### Pattern 3: Batch Evaluation

```python
# Generate batch of initial conditions
x_batch = np.random.randn(1000, system.nx)
u_batch = np.zeros((1000, system.nu))

# Evaluate dynamics for all (vectorized)
dx_batch = system(x_batch, u_batch)  # Returns (1000, nx)
```

### Pattern 4: Multi-Backend Comparison

```python
import time

backends = ['numpy', 'torch', 'jax']
times = {}

for backend in backends:
    system.set_default_backend(backend)
    
    start = time.time()
    result = system.integrate(x0, u=None, t_span=(0, 10))
    times[backend] = time.time() - start

print(times)  # Compare performance
```

## Troubleshooting

### Issue: "System must define self._dt"

**Solution:** For discrete systems, add `self._dt = dt` in `define_system()`

### Issue: "diffusion_expr is None"

**Solution:** For stochastic systems, add `self.diffusion_expr = sp.Matrix([...])` in `define_system()`

### Issue: NumPy boolean comparison warning

**Solution:** This is handled internally. If you see it, ensure you're using the latest version.

### Issue: Integration fails

**Possible causes:**

- Stiff dynamics → Try stiff solver: `method='Radau'` or `method='BDF'`
- Too tight tolerances → Relax: `rtol=1e-6, atol=1e-8`
- Bad initial condition → Check `x0` is physically reasonable

### Issue: Linearization returns NaN

**Possible causes:**

- Equilibrium is not valid → Verify `f(x_eq, u_eq) ≈ 0` (or `= x_eq` for discrete)
- Symbolic expressions have division by zero → Check parameter values

## Testing Your System

```python
def test_my_system():
    """Basic tests for custom system."""
    system = MySystem()
    
    # Test 1: Dimensions
    assert system.nx == 2
    assert system.nu == 1
    
    # Test 2: Evaluation
    x = np.zeros(system.nx)
    u = np.zeros(system.nu)
    dx = system(x, u)
    assert dx.shape == (system.nx,)
    
    # Test 3: Linearization
    A, B = system.linearize(x, u)
    assert A.shape == (system.nx, system.nx)
    assert B.shape == (system.nx, system.nu)
    
    # Test 4: Integration
    result = system.integrate(x, u=None, t_span=(0, 1))
    assert result['success']
    
    print("All tests passed!")

test_my_system()
```

## Advanced Features

### Custom Output Functions

```python
def define_system(self):
    # ... state variables ...
    
    # Define custom output
    self._h_sym = sp.Matrix([
        self.state_vars[0]**2,  # Nonlinear output
        sp.sin(self.state_vars[1])
    ])
    self.output_vars = ['y1', 'y2']

# Use
y = system.observe(x)  # Evaluate output
C = system.linearize_output(x_eq)  # Output Jacobian
```

### Time-Varying Parameters

```python
# This requires system to be time-varying
# Currently, parameters are fixed at initialization
# For time-varying, use control input u to modulate behavior
```

### Hybrid Systems

```python
# Create continuous and discrete versions
cont_sys = MyContinuousSystem()
disc_sys = cont_sys.discretize(dt=0.01, method='rk4')

# disc_sys is now a DiscreteSymbolicSystem
```

## References

- **Architecture Doc:** See `UI_Framework_Architecture.md` for full details
- **Type Definitions:** See `src/types/` for TypedDict definitions
- **Examples:** See `examples/` directory for more complete examples
- **Tests:** See `tests/` directory for comprehensive test suites
