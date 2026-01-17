# ContinuousStochasticSystem Method Design Analysis

## Current State

**What ContinuousStochasticSystem has**:
- ✅ Overrides `integrate()` - adds `n_paths`, `seed` parameters
- ❌ Does NOT override `simulate()` - inherits from ContinuousSystemBase
- ❌ Does NOT have `simulate_stochastic()` method

**Comparison with DiscreteStochasticSystem**:
- ✅ Has `simulate()` (inherited, single path)
- ✅ Has `simulate_stochastic()` (new method, Monte Carlo with `n_paths`, `seed`)

---

## Design Philosophy Comparison

### Continuous: "Parameter Extension" Approach

**Current `integrate()` implementation**:
```python
def integrate(
    self,
    x0,
    u=None,
    t_span=(0, 10),
    method='euler_maruyama',
    n_paths=1,        # ← Extends base method
    seed=None,        # ← Extends base method
    **kwargs
) -> SDEIntegrationResult:
    if n_paths == 1:
        # Single trajectory
    else:
        # Monte Carlo
```

**Philosophy**: Same method handles both single path and Monte Carlo by varying `n_paths`

**Pros**:
- Single method to learn
- Natural progression: n_paths=1 → n_paths=100
- No duplicate code
- Seamless transition between use cases

**Cons**:
- Parameter explosion (many optional params)
- Less explicit about stochastic nature
- Type signature complexity

---

### Discrete: "Separate Methods" Approach

**DiscreteStochasticSystem implementation**:
```python
def simulate(self, x0, u_sequence, n_steps):
    # Single path only (or deterministic-like)
    # Standard discrete simulation
    
def simulate_stochastic(self, x0, u_sequence, n_steps, n_paths, seed):
    # Explicitly Monte Carlo
    # Requires stochastic parameters
```

**Philosophy**: Different methods for different use cases

**Pros**:
- Explicit about deterministic vs stochastic
- Cleaner type signatures
- No ambiguity

**Cons**:
- More methods to learn
- Some code duplication
- Need to choose which to call

---

## Recommendation for ContinuousStochasticSystem

### 1. Override `simulate()` - YES ✅

**Reason**: Users who want regular grids deserve stochastic support

**Signature**:
```python
def simulate(
    self,
    x0: StateVector,
    controller: Optional[FeedbackController] = None,
    t_span: TimeSpan = (0.0, 10.0),
    dt: ScalarLike = 0.01,
    method: SDEIntegrationMethod = "euler_maruyama",
    n_paths: int = 1,           # ← ADD
    seed: Optional[int] = None, # ← ADD
    **kwargs
) -> SimulationResult:  # Or Union[SimulationResult, SDESimulationResult]
```

**Implementation**:
```python
def simulate(self, x0, controller, t_span, dt, method='euler_maruyama', 
             n_paths=1, seed=None, **kwargs):
    """
    High-level SDE simulation with regular time grid.
    
    Extends parent's simulate() to support Monte Carlo simulation
    with reproducible random seeds.
    
    Parameters
    ----------
    n_paths : int
        Number of Monte Carlo paths (default: 1)
        - n_paths=1: Single trajectory
        - n_paths>1: Monte Carlo ensemble
    seed : Optional[int]
        Random seed for reproducibility
    
    Returns
    -------
    SimulationResult or SDESimulationResult
        For n_paths=1:
            - time: (T,) regular grid
            - states: (T, nx)
        For n_paths>1:
            - time: (T,) regular grid  
            - states: (n_paths, T, nx)
    """
    # Convert controller(x, t) to u function if needed
    if controller is not None:
        u_func = lambda t, x: controller(x, t)
    else:
        u_func = None
    
    # Call integrate() with regular grid via dt parameter
    result = self.integrate(
        x0=x0,
        u=u_func,
        t_span=t_span,
        method=method,
        dt=dt,          # Forces regular grid
        n_paths=n_paths,
        seed=seed,
        **kwargs
    )
    
    # Convert IntegrationResult to SimulationResult format
    # (or keep same format for plotting compatibility)
    return {
        'time': result['t'],      # or keep as 't'
        'states': result['x'],    # or keep as 'x'
        'metadata': {
            'method': method,
            'dt': dt,
            'n_paths': n_paths,
            'seed': seed,
            'success': result['success'],
            'noise_type': result.get('noise_type'),
        }
    }
```

**Why this is good**:
- Consistent with `integrate()` - both support n_paths
- Natural for continuous systems (noise is just a parameter)
- Works for both single path and Monte Carlo
- Provides regular grid that `integrate()` doesn't guarantee

---

### 2. Add `simulate_stochastic()` - NO ❌

**Reason**: Redundant with parameter-extended `simulate()`

**Why NOT add it**:

1. **Redundancy**: Would do exactly what `simulate(..., n_paths=N, seed=S)` does
2. **API bloat**: Two methods for same thing confuses users
3. **Against continuous philosophy**: The parameter extension works well here
4. **Different from discrete**: Discrete systems have algorithmic differences; continuous don't

**When you'd call it**:
```python
# Option A: Using simulate() with parameters (BETTER)
result = system.simulate(x0, t_span=(0, 10), dt=0.01, 
                        n_paths=500, seed=42)

# Option B: Using simulate_stochastic() (REDUNDANT)
result = system.simulate_stochastic(x0, t_span=(0, 10), dt=0.01,
                                   n_paths=500, seed=42)
```

There's no benefit to Option B - it's the same call with different name.

---

## Why Discrete Has Both But Continuous Shouldn't

### Discrete Systems

**Reason for separation**:
```python
# simulate() - might use deterministic stepping algorithm
def simulate(self, x0, u_sequence, n_steps):
    for k in range(n_steps):
        x = self.step(x, u, k)  # Deterministic step
        
# simulate_stochastic() - uses stochastic stepping
def simulate_stochastic(self, x0, u_sequence, n_steps, n_paths, seed):
    for k in range(n_steps):
        w = sample_noise(seed)
        x = self.stochastic_step(x, u, k, w)  # Different algorithm
```

**Different algorithms** warrant separate methods.

### Continuous Systems

**No algorithmic difference**:
```python
# integrate() handles BOTH via same integrator
def integrate(self, x0, u, t_span, method, n_paths, seed):
    integrator = SDEIntegratorFactory.create(method, seed)
    
    if n_paths == 1:
        return integrator.integrate(x0, u, t_span)
    else:
        return integrator.integrate_monte_carlo(x0, u, t_span, n_paths)
```

**Same integrator, just batched** - no reason for separate method.

---

## Complete Recommendation

### What ContinuousStochasticSystem Should Have

```python
class ContinuousStochasticSystem(ContinuousSymbolicSystem):
    
    def integrate(
        self,
        x0, u, t_span,
        method='euler_maruyama',
        n_paths=1,    # ✅ Already has this
        seed=None,    # ✅ Already has this
        **kwargs
    ) -> SDEIntegrationResult:
        """Low-level SDE integration (adaptive or regular grid)."""
        # ✅ Already implemented correctly
        
    def simulate(
        self,
        x0, controller, t_span, dt,
        method='euler_maruyama',
        n_paths=1,    # ❌ NEEDS TO ADD
        seed=None,    # ❌ NEEDS TO ADD
        **kwargs
    ) -> SimulationResult:
        """High-level SDE simulation (regular grid guaranteed)."""
        # ❌ Currently NOT overridden - SHOULD BE
        # Should wrap integrate() with dt parameter
        # Should convert controller(x,t) to u(t,x)
        # Should return regular grid result
```

### What Should NOT Be Added

```python
# ❌ DON'T ADD THIS - Redundant
def simulate_stochastic(self, x0, controller, t_span, dt, 
                       n_paths, seed, **kwargs):
    # This would just call simulate() anyway
    return self.simulate(x0, controller, t_span, dt,
                        n_paths=n_paths, seed=seed, **kwargs)
```

---

## Implementation Priority

### High Priority ✅

**Override `simulate()` in ContinuousStochasticSystem**:
```python
def simulate(
    self,
    x0: StateVector,
    controller: Optional[FeedbackController] = None,
    t_span: TimeSpan = (0.0, 10.0),
    dt: ScalarLike = 0.01,
    method: SDEIntegrationMethod = "euler_maruyama",
    n_paths: int = 1,
    seed: Optional[int] = None,
    **kwargs
) -> SimulationResult:
    """
    Simulate stochastic system with regular time grid.
    
    Wraps integrate() to provide:
    - Regular time grid (dt spacing)
    - State-feedback controller support
    - Cleaner output format
    - Monte Carlo support via n_paths
    
    Parameters
    ----------
    controller : Optional[Callable[[StateVector, float], ControlVector]]
        Feedback controller u = controller(x, t)
        State is primary argument (matches discrete convention)
    n_paths : int
        Number of Monte Carlo paths (default: 1)
    seed : Optional[int]
        Random seed for reproducibility
    
    Returns
    -------
    SimulationResult
        Dictionary containing:
        - time: Regular time grid (T,)
        - states: Trajectory (T, nx) or (n_paths, T, nx)
        - metadata: {method, dt, n_paths, seed, noise_type, ...}
        
    Examples
    --------
    Single trajectory with seed:
    >>> result = system.simulate(x0, t_span=(0, 10), dt=0.01, seed=42)
    
    Monte Carlo ensemble:
    >>> result = system.simulate(x0, t_span=(0, 10), dt=0.01,
    ...                          n_paths=500, seed=42)
    >>> mean = result['states'].mean(axis=0)  # (T, nx)
    
    With state feedback:
    >>> def controller(x, t):
    ...     return -K @ x
    >>> result = system.simulate(x0, controller, t_span=(0, 10), dt=0.01)
    """
    # Convert controller (x, t) -> u  to  u_func (t, x) -> u
    if controller is not None:
        u_func = lambda t, x: controller(x, t)
    else:
        u_func = None
    
    # Call integrate with regular grid (dt parameter)
    int_result = self.integrate(
        x0=x0,
        u=u_func,
        t_span=t_span,
        method=method,
        dt=dt,           # Ensures regular grid
        n_paths=n_paths,
        seed=seed,
        **kwargs
    )
    
    # Return in SimulationResult format
    # NOTE: Consider keeping 't', 'x' for plotting compatibility
    return {
        't': int_result['t'],      # or 'time'
        'x': int_result['x'],      # or 'states'
        'success': int_result['success'],
        'metadata': {
            'method': method,
            'dt': dt,
            'n_paths': n_paths,
            'seed': seed,
            'noise_type': int_result.get('noise_type'),
            'sde_type': int_result.get('sde_type'),
        }
    }
```

### Don't Implement ❌

**`simulate_stochastic()` method** - not needed, would be redundant

---

## Summary Table

| Method | Should Override? | Should Add? | Reason |
|--------|------------------|-------------|---------|
| `integrate()` | ✅ Already done | - | SDE-specific with n_paths, seed |
| `simulate()` | ✅ YES | - | Provide regular grid version with n_paths, seed |
| `simulate_stochastic()` | - | ❌ NO | Redundant with parameter-extended simulate() |

---

## Key Design Principle

**For continuous systems**: Extend existing methods with optional stochastic parameters
- `integrate(..., n_paths=1, seed=None)` - works for both ODE and SDE
- `simulate(..., n_paths=1, seed=None)` - works for both ODE and SDE

**For discrete systems**: Separate methods make sense due to algorithmic differences
- `simulate()` - deterministic stepping
- `simulate_stochastic()` - stochastic stepping

**Continuous is parameter-driven, discrete is method-driven.**
