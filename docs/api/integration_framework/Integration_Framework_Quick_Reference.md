# Numerical Integration Framework Quick Reference

## Quick Start

### 1. Automatic Integration (Easiest)

```python
from src.systems.base.numerical_integration import IntegratorFactory

# Let the factory choose the best integrator
integrator = IntegratorFactory.auto(system)

# Integrate
result = integrator.integrate(
    x0=np.array([1.0, 0.0]),
    u_func=lambda t, x: np.zeros(1),
    t_span=(0.0, 10.0)
)

print(f"Success: {result['success']}")
print(f"Method: {result['solver']}")
```

### 2. Specific Method Selection

```python
# Create with specific method
integrator = IntegratorFactory.create(
    system,
    backend='numpy',
    method='RK45',
    rtol=1e-6,
    atol=1e-8
)
```

### 3. Use Case Helpers

```python
# For production
integrator = IntegratorFactory.for_production(system)

# For optimization (JAX)
integrator = IntegratorFactory.for_optimization(system)

# For neural ODEs (PyTorch)
integrator = IntegratorFactory.for_neural_ode(system)

# For Julia (highest performance)
integrator = IntegratorFactory.for_julia(system, algorithm='Vern9')
```

## Available Methods

### NumPy (scipy)

```python
# Adaptive non-stiff
IntegratorFactory.create(system, backend='numpy', method='RK45')   # General
IntegratorFactory.create(system, backend='numpy', method='RK23')   # Fast
IntegratorFactory.create(system, backend='numpy', method='DOP853') # Accurate

# Adaptive stiff
IntegratorFactory.create(system, backend='numpy', method='Radau')  # Stiff
IntegratorFactory.create(system, backend='numpy', method='BDF')    # Very stiff
IntegratorFactory.create(system, backend='numpy', method='LSODA')  # Auto-stiffness ⭐
```

### NumPy (Julia via DiffEqPy)

```python
# High-order explicit
IntegratorFactory.create(system, backend='numpy', method='Tsit5')  # 5th order ⭐
IntegratorFactory.create(system, backend='numpy', method='Vern9')  # 9th order
IntegratorFactory.create(system, backend='numpy', method='DP8')    # Dormand-Prince 8

# Rosenbrock (stiff)
IntegratorFactory.create(system, backend='numpy', method='Rodas5')  # Best stiff ⭐
IntegratorFactory.create(system, backend='numpy', method='Rosenbrock23')

# Auto-switching
IntegratorFactory.create(system, backend='numpy', method='AutoTsit5(Rosenbrock23())')
```

### PyTorch (torchdiffeq)

```python
# Adaptive methods
IntegratorFactory.create(system, backend='torch', method='dopri5')  # 5(4) ⭐
IntegratorFactory.create(system, backend='torch', method='dopri8')  # 8
IntegratorFactory.create(system, backend='torch', method='bosh3')   # 3

# With adjoint for memory efficiency
integrator = TorchDiffEqIntegrator(
    system,
    method='dopri5',
    adjoint=True  # For neural ODEs
)
```

### JAX (diffrax)

```python
# Adaptive methods
IntegratorFactory.create(system, backend='jax', method='tsit5')   # Best ⭐
IntegratorFactory.create(system, backend='jax', method='dopri5')  # Alternative
IntegratorFactory.create(system, backend='jax', method='dopri8')  # High accuracy
```

### Manual (Fixed-Step)

```python
# Simple methods
IntegratorFactory.create(system, backend='numpy', method='rk4', dt=0.01)
IntegratorFactory.create(system, backend='numpy', method='euler', dt=0.01)
IntegratorFactory.create(system, backend='numpy', method='midpoint', dt=0.01)
```

## Stochastic SDE Integration

### Create SDE Integrator

```python
from src.systems.base.numerical_integration.stochastic import SDEIntegratorFactory

# Automatic selection based on noise type
integrator = SDEIntegratorFactory.auto(sde_system)

# Specific method
integrator = SDEIntegratorFactory.create(
    sde_system,
    backend='torch',
    method='heun',
    dt=0.01,
    seed=42
)
```

### Single Trajectory

```python
result = integrator.integrate(
    x0=np.array([1.0]),
    u_func=lambda t, x: None,  # Autonomous
    t_span=(0.0, 10.0)
)

print(f"SDE type: {result['sde_type']}")
print(f"Noise type: {result['noise_type']}")
print(f"Diffusion evals: {result['diffusion_evals']}")
```

### Monte Carlo Simulation

```python
# Multiple trajectories for statistics
result = integrator.integrate_monte_carlo(
    x0=np.array([1.0]),
    u_func=lambda t, x: None,
    t_span=(0.0, 10.0),
    n_paths=1000  # 1000 trajectories
)

# Get statistics
from src.systems.base.numerical_integration.sde_integrator_base import (
    get_trajectory_statistics
)
stats = get_trajectory_statistics(result)

print(f"Mean: {stats['mean'][-1]}")
print(f"Std: {stats['std'][-1]}")
print(f"Median: {stats['median'][-1]}")
```

### Custom Noise (Deterministic Testing)

```python
import jax.numpy as jnp
from src.systems.base.numerical_integration.stochastic.custom_brownian import (
    CustomBrownianPath
)

# Zero noise for testing
dW = jnp.zeros((nw,))
brownian = CustomBrownianPath(0.0, 0.01, dW)

# Use in integration
result = integrator.integrate(
    x0, u_func, t_span=(0, 1),
    brownian_path=brownian
)
```

## Common Patterns

### Pattern 1: GPU Acceleration (PyTorch)

```python
import torch

# Set system to use PyTorch backend
system.set_default_backend('torch')
system.set_default_device('cuda:0')

# Create integrator
integrator = IntegratorFactory.create(
    system,
    backend='torch',
    method='dopri5'
)

# Data on GPU
x0 = torch.tensor([[1.0, 0.0]], device='cuda:0')
result = integrator.integrate(x0, u_func, t_span=(0, 10))

# Result is on GPU
assert result['x'].device.type == 'cuda'
```

### Pattern 2: High Accuracy

```python
# Julia Vern9 with tight tolerances
integrator = IntegratorFactory.for_julia(
    system,
    algorithm='Vern9',  # 9th order
    reltol=1e-12,
    abstol=1e-14
)

result = integrator.integrate(x0, u_func, t_span=(0, 100))
print(f"Function evaluations: {result['nfev']}")
```

### Pattern 3: Stiff Systems

```python
# Option 1: scipy BDF
integrator = IntegratorFactory.create(
    system,
    backend='numpy',
    method='BDF',
    rtol=1e-8,
    atol=1e-10
)

# Option 2: Julia Rodas5 (better performance)
integrator = IntegratorFactory.for_julia(
    system,
    algorithm='Rodas5'
)

# Option 3: Auto-switching
integrator = IntegratorFactory.create(
    system,
    backend='numpy',
    method='AutoTsit5(Rosenbrock23())'
)
```

### Pattern 4: Dense Output (Interpolation)

```python
# Request dense output
result = integrator.integrate(
    x0, u_func, t_span=(0, 10),
    dense_output=True
)

if result.get('dense_output'):
    # Evaluate at arbitrary times
    t_fine = np.linspace(0, 10, 10000)
    x_fine = result['sol'](t_fine)
```

### Pattern 5: Specific Time Points

```python
# Evaluate at specific times
t_eval = np.linspace(0, 10, 1001)  # 1001 uniform points

result = integrator.integrate(
    x0, u_func, t_span=(0, 10),
    t_eval=t_eval
)

assert len(result['t']) == 1001
```

### Pattern 6: Controlled vs Autonomous Systems

```python
# Controlled system
u_func = lambda t, x: -K @ x  # State feedback
result = integrator.integrate(x0, u_func, t_span=(0, 10))

# Autonomous system (nu=0)
u_func = lambda t, x: None  # No control
result = integrator.integrate(x0, u_func, t_span=(0, 10))
```

### Pattern 7: Time-Varying Control

```python
# Time-varying control
def u_func(t, x):
    return np.array([np.sin(t)])

result = integrator.integrate(x0, u_func, t_span=(0, 10))
```

### Pattern 8: Neural ODE Training

```python
import torch
from torch import nn

class NeuralODE(nn.Module):
    def __init__(self, system):
        super().__init__()
        self.system = system
        self.integrator = IntegratorFactory.for_neural_ode(system)
    
    def forward(self, x0, t_span):
        result = self.integrator.integrate(
            x0, 
            u_func=lambda t, x: torch.zeros(1),
            t_span=t_span
        )
        return result['x']

# Use in training loop
model = NeuralODE(system)
optimizer = torch.optim.Adam(model.parameters())

for epoch in range(100):
    x_pred = model(x0, t_span=(0, 1))
    loss = criterion(x_pred, x_true)
    loss.backward()
    optimizer.step()
```

## Integrator Selection Guide

### By System Type

| System | Recommended |
|--------|-------------|
| Non-stiff ODE | `RK45`, `Tsit5`, `dopri5` |
| Stiff ODE | `BDF`, `Radau`, `Rodas5` |
| Unknown stiffness | `LSODA`, `AutoTsit5(...)` |
| High accuracy | `Vern9`, `DOP853` |
| Simple/Educational | `rk4`, `euler` |
| Additive noise SDE | `heun` (strong order 1.0) |
| General SDE | `euler-maruyama` |

### By Backend

| Backend | Best For | Best Method |
|---------|----------|-------------|
| NumPy (scipy) | General use | `LSODA` |
| NumPy (Julia) | Performance | `Tsit5`, `Vern9` |
| PyTorch | Neural ODEs | `dopri5` (adjoint) |
| JAX | Optimization | `tsit5` |

### By Use Case

| Use Case | Factory Method |
|----------|----------------|
| General | `IntegratorFactory.auto(system)` |
| Production | `IntegratorFactory.for_production(system)` |
| Optimization | `IntegratorFactory.for_optimization(system)` |
| Neural ODE | `IntegratorFactory.for_neural_ode(system)` |
| Highest accuracy | `IntegratorFactory.for_julia(system, 'Vern9')` |
| Monte Carlo SDE | `SDEIntegratorFactory.for_monte_carlo(...)` |

## Integration Result Fields

### ODE Integration
```python
result = {
    't': array,              # Time points (T,)
    'x': array,              # State trajectory (T, nx)
    'success': bool,         # Integration succeeded
    'message': str,          # Status message
    'nfev': int,             # Function evaluations
    'nsteps': int,           # Integration steps
    'integration_time': float,  # Wall time (seconds)
    'solver': str,           # Integrator name
    
    # Optional (adaptive):
    'njev': int,             # Jacobian evaluations
    'nlu': int,              # LU decompositions
    'status': int,           # Status code
    'sol': object,           # Dense output
}
```

### SDE Integration
```python
result = {
    # All ODE fields, plus:
    'diffusion_evals': int,     # Diffusion calls
    'noise_samples': array,     # Brownian increments
    'n_paths': int,             # Number of trajectories
    'convergence_type': str,    # 'strong' or 'weak'
    'sde_type': str,            # 'ito' or 'stratonovich'
    'noise_type': str,          # 'additive', etc.
}
```

## Troubleshooting

### Issue: Integration fails with "maximum steps exceeded"
**Solutions:**
- Stiff system → Use stiff solver: `method='BDF'` or `method='Radau'`
- Increase max steps: `max_steps=100000`
- Loosen tolerances: `rtol=1e-5, atol=1e-7`

### Issue: Integration too slow
**Solutions:**
- Use Julia: `IntegratorFactory.for_julia(system, 'Tsit5')`
- Use GPU: `backend='torch'` with `device='cuda:0'`
- Loosen tolerances: `rtol=1e-4, atol=1e-6`
- Try fixed-step: `method='rk4', dt=0.01`

### Issue: Integration not accurate enough
**Solutions:**
- Tighten tolerances: `rtol=1e-9, atol=1e-11`
- Use high-order method: `method='Vern9'` or `method='DOP853'`
- Provide `t_eval` with dense grid

### Issue: SDE results vary too much
**Solutions:**
- Increase Monte Carlo paths: `n_paths=10000`
- Reduce time step: `dt=0.001`
- Use higher-order method: `method='heun'` (for additive noise)
- Set seed for reproducibility: `seed=42`

### Issue: Out of memory (GPU)
**Solutions:**
- Use adjoint method: `adjoint=True` (PyTorch)
- Reduce batch size / number of paths
- Use checkpointing (if available)
- Switch to CPU: `backend='numpy'`

## Advanced Features

### Event Detection (scipy only)

```python
def impact_event(t, x):
    """Detect when velocity crosses zero."""
    return x[1]  # velocity

impact_event.terminal = True  # Stop at event
impact_event.direction = -1   # Negative crossing

result = integrator.integrate(
    x0, u_func, t_span=(0, 10),
    events=impact_event
)

if result.get('t_events'):
    print(f"Impact at t = {result['t_events'][0]}")
```

### Custom Integration Options

```python
# Backend-specific options
result = integrator.integrate(
    x0, u_func, t_span=(0, 10),
    
    # Scipy options
    max_step=0.1,          # Maximum step size
    first_step=0.001,      # Initial step
    vectorized=True,       # Vectorized evaluation
    
    # JAX options
    solver_kwargs={
        'max_steps': 10000,
        'adjoint': 'checkpoint'
    }
)
```

### Performance Profiling

```python
# Check integrator statistics
print(f"Function evaluations: {integrator._stats['total_fev']}")
print(f"Total steps: {integrator._stats['total_steps']}")
print(f"Total time: {integrator._stats['total_time']:.3f}s")
```

## Testing Your Integration

```python
def test_integration():
    """Basic integration test."""
    system = MySystem()
    integrator = IntegratorFactory.auto(system)
    
    x0 = np.array([1.0, 0.0])
    u_func = lambda t, x: np.zeros(1)
    
    result = integrator.integrate(x0, u_func, t_span=(0, 1))
    
    # Check success
    assert result['success'], f"Integration failed: {result['message']}"
    
    # Check shape
    assert result['x'].shape[1] == system.nx
    
    # Check time points
    assert result['t'][0] == 0.0
    assert result['t'][-1] <= 1.0
    
    print("✓ Integration test passed")

test_integration()
```

## References

- **Architecture:** See `Integration_Framework_Architecture.md`
- **Type Definitions:** See `src/types/trajectories.py`
- **Backend Info:** See `src/types/backends.py`
- **System Interface:** See `continuous_system_base.py`
