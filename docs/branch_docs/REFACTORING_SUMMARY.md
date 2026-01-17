# ControlDESymulation Refactoring Summary

## Overview

This document summarizes the breaking changes required to standardize the API before v1.0 release on PyPI. Since there are no public users yet, this is the optimal time for breaking changes.

---

## 1. Standardize Array Shape Convention (CRITICAL)

### Problem
- **Continuous systems**: Time-major `(T, nx)` - rows are time points
- **Discrete systems**: State-major `(nx, T)` - columns are time points
- Inconsistent indexing confuses users and breaks generic code

### Solution
**Standardize ALL systems on time-major convention: `(T, nx)`**

### Changes Required

#### DiscreteSystemBase.simulate()
```python
# OLD return format
{
    'states': np.ndarray,      # (nx, n_steps+1) ❌
    'controls': np.ndarray,    # (nu, n_steps) ❌
}

# NEW return format
{
    'states': np.ndarray,      # (n_steps+1, nx) ✅
    'controls': np.ndarray,    # (n_steps, nu) ✅
}
```

#### Impact
- DiscreteSymbolicSystem.simulate() implementation
- DiscreteStochasticSystem.simulate_stochastic() implementation
- DiscretizedSystem.simulate() and simulate_stochastic()
- All discrete plotting code
- All discrete examples and tests

---

## 2. Add rollout() Method to Continuous Systems

### Problem
- Discrete systems have `simulate()` (open-loop) and `rollout()` (closed-loop)
- Continuous systems only have `simulate()` with controller parameter
- Inconsistent API between continuous and discrete

### Solution
**Add `rollout()` to ContinuousSystemBase for state-feedback control**

### Changes Required

#### ContinuousSystemBase - Refactor Methods

**OLD `simulate()` signature**:
```python
def simulate(
    self,
    x0: StateVector,
    controller: Optional[FeedbackController] = None,  # ❌ Remove
    t_span: TimeSpan = (0.0, 10.0),
    dt: ScalarLike = 0.01,
    method: IntegrationMethod = "RK45",
    **kwargs
) -> SimulationResult:
```

**NEW `simulate()` signature** (open-loop only):
```python
def simulate(
    self,
    x0: StateVector,
    u: Optional[Union[ControlVector, TimeVaryingControl]] = None,  # ✅ Open-loop
    t_span: TimeSpan = (0.0, 10.0),
    dt: ScalarLike = 0.01,
    method: IntegrationMethod = "RK45",
    **kwargs
) -> SimulationResult:
    """
    Open-loop simulation with time-varying control u(t).
    
    Parameters
    ----------
    u : Optional[Union[ControlVector, Callable[[float], ControlVector]]]
        Control input:
        - None: Zero control
        - Array (nu,): Constant control
        - Callable u(t): Time-varying control (NO state feedback)
    """
```

**NEW `rollout()` method** (closed-loop):
```python
def rollout(
    self,
    x0: StateVector,
    controller: Optional[FeedbackController] = None,  # ✅ State feedback
    t_span: TimeSpan = (0.0, 10.0),
    dt: ScalarLike = 0.01,
    method: IntegrationMethod = "RK45",
    **kwargs
) -> SimulationResult:
    """
    Closed-loop simulation with state-feedback controller(x, t).
    
    Parameters
    ----------
    controller : Optional[Callable[[StateVector, float], ControlVector]]
        Feedback controller u = controller(x, t)
        State is primary argument (matches discrete convention)
    """
```

#### Implementation Pattern
```python
def rollout(self, x0, controller, t_span, dt, method, **kwargs):
    # Convert controller(x, t) to u(t) for simulate()
    # (This requires integration, so actually call integrate() directly)
    u_func = lambda t, x: controller(x, t) if controller else None
    return self.integrate(x0, u_func, t_span, method=method, dt=dt, **kwargs)
```

---

## 3. Override simulate() in ContinuousStochasticSystem

### Problem
- ContinuousStochasticSystem.integrate() supports `n_paths` and `seed`
- ContinuousStochasticSystem.simulate() does NOT (inherits from parent)
- No way to do regular-grid SDE simulation with Monte Carlo

### Solution
**Override `simulate()` to add `n_paths` and `seed` parameters**

### Changes Required

```python
class ContinuousStochasticSystem(ContinuousSymbolicSystem):
    
    def simulate(
        self,
        x0: StateVector,
        u: Optional[Union[ControlVector, TimeVaryingControl]] = None,
        t_span: TimeSpan = (0.0, 10.0),
        dt: ScalarLike = 0.01,
        method: SDEIntegrationMethod = "euler_maruyama",
        n_paths: int = 1,           # ✅ ADD
        seed: Optional[int] = None, # ✅ ADD
        **kwargs
    ) -> SimulationResult:
        """
        Simulate stochastic system with regular time grid.
        
        Supports Monte Carlo simulation via n_paths parameter.
        """
        # Delegate to integrate() with dt parameter
        return self.integrate(
            x0=x0,
            u=u,
            t_span=t_span,
            method=method,
            dt=dt,           # Ensures regular grid
            n_paths=n_paths,
            seed=seed,
            **kwargs
        )
    
    def rollout(
        self,
        x0: StateVector,
        controller: Optional[FeedbackController] = None,
        t_span: TimeSpan = (0.0, 10.0),
        dt: ScalarLike = 0.01,
        method: SDEIntegrationMethod = "euler_maruyama",
        n_paths: int = 1,           # ✅ ADD
        seed: Optional[int] = None, # ✅ ADD
        **kwargs
    ) -> SimulationResult:
        """
        Rollout stochastic system with state feedback.
        
        Supports Monte Carlo simulation via n_paths parameter.
        """
        # Convert controller to u function
        u_func = lambda t, x: controller(x, t) if controller else None
        return self.integrate(
            x0=x0,
            u=u_func,
            t_span=t_span,
            method=method,
            dt=dt,
            n_paths=n_paths,
            seed=seed,
            **kwargs
        )
```

**DO NOT add `simulate_stochastic()`** - redundant with parameter-extended `simulate()`

---

## 4. Standardize Return Dictionary Keys

### Problem
- `integrate()` returns: `{'t': ..., 'x': ..., ...}`
- `simulate()` returns: `{'time': ..., 'states': ..., ...}`
- Plotting only works with `integrate()` format

### Solution
**Standardize on `integrate()` format for all methods**

### Changes Required

All `simulate()` and `rollout()` methods should return:
```python
{
    't': np.ndarray,       # Time points (T,) - NOT 'time'
    'x': np.ndarray,       # States (T, nx) - NOT 'states'
    'u': np.ndarray,       # Controls (T, nu) - NOT 'controls'
    'success': bool,
    'message': str,
    'method': str,
    'dt': float,
    'metadata': dict
}
```

**Rationale**: Plotting infrastructure expects `result['t']` and `result['x']`

**Alternative**: Update plotting to accept both formats (more work, less clean)

---

## 5. Discrete Systems: Update Return Format

### Current Issues
- Key name: `'time_steps'` → change to `'t'`
- Shape: `(nx, T)` → change to `(T, nx)`
- Key name: `'states'` → change to `'x'`
- Key name: `'controls'` → change to `'u'` (optional)

### Updated DiscreteSimulationResult

```python
# OLD
DiscreteSimulationResult = TypedDict('DiscreteSimulationResult', {
    'states': NDArray,        # (nx, n_steps+1) ❌
    'controls': NDArray,      # (nu, n_steps) ❌
    'time_steps': NDArray,    # ❌
    'dt': float,
    'metadata': dict
})

# NEW  
DiscreteSimulationResult = TypedDict('DiscreteSimulationResult', {
    'x': NDArray,             # (n_steps+1, nx) ✅ time-major
    'u': NDArray,             # (n_steps, nu) ✅ time-major
    't': NDArray,             # ✅ matches continuous
    'dt': float,
    'success': bool,          # ✅ add for consistency
    'message': str,           # ✅ add for consistency
    'method': str,
    'metadata': dict
})
```

---

## 6. Update All Implementations

### Files to Modify

#### Core System Files
- `continuous_system_base.py` - add `rollout()`, refactor `simulate()`
- `continuous_stochastic_system.py` - override `simulate()` and `rollout()`
- `discrete_system_base.py` - update return format to time-major
- `discrete_symbolic_system.py` - update `simulate()` implementation
- `discrete_stochastic_system.py` - update `simulate_stochastic()`
- `discretized_system.py` - update both `simulate()` methods

#### Type Definitions
- Update `SimulationResult` TypedDict
- Update `DiscreteSimulationResult` TypedDict
- Update `SDEIntegrationResult` TypedDict (if needed)

#### Plotting Infrastructure
- Verify `plotter.plot_trajectory()` works with new format
- Verify `phase_plotter.plot_2d()` works with new format
- Update any plotting that assumes state-major

#### Tests
- All discrete system tests (shape assertions)
- All continuous system tests (controller → rollout)
- All plotting tests
- Integration tests across system types

#### Examples
- Update all example scripts
- Update all tutorial notebooks

---

## 7. Tutorial Updates

### Basic Usage Tutorial
```python
# Use integrate() for now (already works)
result = pendulum.integrate(x0, t_span=(0, 20), dt=0.05)
pendulum.plot(result)
```

**After refactoring**:
```python
# Open-loop with simulate()
result = pendulum.simulate(x0, u=None, t_span=(0, 20), dt=0.05)

# Closed-loop with rollout()
result = pendulum.rollout(x0, controller=my_controller, t_span=(0, 20), dt=0.05)

# Both work with plotting!
pendulum.plot(result)
```

### Stochastic Tutorial
```python
# Already uses integrate() - no change needed
result = reactor.integrate(
    x0, t_span=(0, 50), dt=0.05,
    method='euler_maruyama',
    n_paths=500,
    seed=42
)
```

**After refactoring** - can also use simulate():
```python
# Open-loop stochastic
result = reactor.simulate(
    x0, u=None, t_span=(0, 50), dt=0.05,
    n_paths=500,
    seed=42
)

# Closed-loop stochastic
result = reactor.rollout(
    x0, controller=feedback_controller, t_span=(0, 50), dt=0.05,
    n_paths=500,
    seed=42
)
```

---

## Summary of Breaking Changes

### API Changes (User-Facing)

1. **Discrete result indexing** (BREAKING):
   ```python
   # OLD
   result['states'][i, :]  # i-th state over time
   result['states'][:, k]  # All states at time k
   
   # NEW
   result['x'][k, :]       # All states at time k
   result['x'][:, i]       # i-th state over time
   ```

2. **Continuous simulate() signature** (BREAKING):
   ```python
   # OLD
   result = system.simulate(x0, controller=my_controller, ...)
   
   # NEW - open-loop
   result = system.simulate(x0, u=my_u_func, ...)
   
   # NEW - closed-loop
   result = system.rollout(x0, controller=my_controller, ...)
   ```

3. **Result dictionary keys** (BREAKING):
   ```python
   # OLD
   result['time'], result['states'], result['controls']
   
   # NEW
   result['t'], result['x'], result['u']
   ```

### Internal Changes

1. All discrete `simulate()` implementations return time-major
2. All `simulate()` methods support open-loop only
3. All `rollout()` methods support closed-loop only
4. Stochastic systems override both `simulate()` and `rollout()`
5. Plotting works uniformly across all result types

---

## Benefits

### Consistency
- ✅ Same shape convention everywhere: `(T, nx)`
- ✅ Same method names: `simulate()` and `rollout()`
- ✅ Same semantics: open-loop vs closed-loop
- ✅ Same return format: `{'t', 'x', 'u', ...}`

### Usability
- ✅ No mental overhead remembering which convention applies
- ✅ Generic plotting code works for all systems
- ✅ Pandas integration: `pd.DataFrame(result['x'], index=result['t'])`
- ✅ Clear separation of concerns: simulate vs rollout

### Compatibility
- ✅ Aligns with modern Python ecosystem (pandas, sklearn, PyTorch)
- ✅ Makes library easier to learn and teach
- ✅ Reduces documentation burden

---

## Migration Checklist

- [ ] Update type definitions (SimulationResult, DiscreteSimulationResult)
- [ ] Add ContinuousSystemBase.rollout()
- [ ] Refactor ContinuousSystemBase.simulate() (remove controller param)
- [ ] Override ContinuousStochasticSystem.simulate() and rollout()
- [ ] Update DiscreteSymbolicSystem.simulate() to time-major
- [ ] Update DiscreteStochasticSystem.simulate_stochastic() to time-major
- [ ] Update DiscretizedSystem.simulate() to time-major
- [ ] Update DiscretizedSystem.simulate_stochastic() to time-major
- [ ] Update all plotting code
- [ ] Update all tests
- [ ] Update all examples
- [ ] Update all tutorials
- [ ] Update documentation
- [ ] Run full test suite
- [ ] Review all docstrings

---

## Risk Assessment

**Risk**: LOW - No public users yet

**Complexity**: MODERATE - Touches many files but changes are mechanical

**Testing Burden**: HIGH - Must verify all tests pass with new conventions

**Documentation Burden**: MODERATE - Update examples and tutorials

**Recommendation**: Execute before v1.0 release to PyPI
