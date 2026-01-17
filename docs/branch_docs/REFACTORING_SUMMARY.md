# ControlDESymulation Refactoring Summary

## Overview

This document summarizes the breaking changes required to standardize the API before v1.0 release on PyPI. Since there are no public users yet, this is the optimal time for breaking changes.

**Key Changes**:
1. Time-major array ordering everywhere: `(T, nx)` not `(nx, T)`
2. Consistent key names: `'t'`, `'x'`, `'u'` (not `'time'`, `'states'`, `'controls'`)
3. Method separation: `simulate()` (open-loop) vs `rollout()` (closed-loop)
4. **Complete type system**: 10 result types covering deterministic + stochastic variants

**Estimated effort**: ~78 hours (~2 weeks)

---

## 1. Standardize Array Shape Convention (CRITICAL)

### Problem
- **Continuous systems**: Time-major `(T, nx)` - rows are time points
- **Discrete systems**: State-major `(nx, T)` - columns are time points
- **Inconsistent** indexing confuses users and breaks generic code

### Solution
**Standardize ALL systems on time-major convention: `(T, nx)`**

### Changes Required

#### Continuous: SimulationResult
```python
# OLD return format
{
    'time': np.ndarray,      # (T,) ❌
    'states': np.ndarray,    # (nx, T) ❌ STATE-MAJOR
    'controls': np.ndarray,  # (nu, T) ❌
}

# NEW return format
{
    't': np.ndarray,         # (T,) ✅
    'x': np.ndarray,         # (T, nx) ✅ TIME-MAJOR
    'u': np.ndarray,         # (T, nu) ✅
}
```

#### Discrete: DiscreteSimulationResult
```python
# OLD return format
{
    'time_steps': np.ndarray,  # (T,) ❌
    'states': np.ndarray,      # (nx, T) ❌ STATE-MAJOR
    'controls': np.ndarray,    # (nu, T) ❌
}

# NEW return format
{
    't': np.ndarray,           # (T,) ✅
    'x': np.ndarray,           # (T, nx) ✅ TIME-MAJOR
    'u': np.ndarray,           # (T, nu) ✅
}
```

### Impact
- All system implementations
- All discrete plotting code
- All discrete examples and tests
- User code that accesses result arrays

---

## 2. Create Comprehensive Type System (NEW)

### Problem
- Old: Only basic TypedDict definitions in `trajectories.py`
- Missing: Stochastic result types, rollout types, union types
- No clear organization of result types

### Solution
**Create dedicated `system_results.py` module with complete type hierarchy**

### Type Hierarchy

```
system_results.py (NEW MODULE)
├── Base Types (4)
│   ├── IntegrationResultBase
│   ├── SimulationResultBase
│   ├── RolloutResultBase
│   └── DiscreteSimulationResultBase
│
├── Continuous Deterministic (3)
│   ├── IntegrationResult (ODE, adaptive grid) ✅ already correct
│   ├── SimulationResult (ODE, regular grid) ❌ needs update
│   └── RolloutResult (ODE, closed-loop) ❌ NEW
│
├── Continuous Stochastic (3)
│   ├── SDEIntegrationResult (SDE, adaptive grid) ✅ already correct
│   ├── SDESimulationResult (SDE, regular grid) ❌ NEW
│   └── SDERolloutResult (SDE, closed-loop) ❌ NEW
│
├── Discrete Deterministic (2)
│   ├── DiscreteSimulationResult (open-loop) ❌ needs update
│   └── DiscreteRolloutResult (closed-loop) ❌ needs update
│
├── Discrete Stochastic (2)
│   ├── DiscreteStochasticSimulationResult ❌ NEW
│   └── DiscreteStochasticRolloutResult ❌ NEW
│
└── Union Types (6)
    ├── ContinuousIntegrationResultUnion
    ├── ContinuousSimulationResultUnion
    ├── ContinuousRolloutResultUnion
    ├── DiscreteSimulationResultUnion
    ├── DiscreteRolloutResultUnion
    └── SystemResult (union of ALL types) ❌ NEW
```

### Total: 10 Concrete Types + 4 Base Types + 6 Union Types = 20 exports

### Changes Required

1. **Create** `src/cdesym/types/system_results.py` with all types
2. **Update** `src/cdesym/types/trajectories.py` to import from system_results
3. **Update** `src/cdesym/types/__init__.py` to export all types
4. **Use** appropriate types in all system implementations

---

## 3. Add rollout() Method to Continuous Systems

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
) -> RolloutResult:
    """
    Closed-loop simulation with state-feedback controller(x, t).
    
    Parameters
    ----------
    controller : Optional[Callable[[StateVector, float], ControlVector]]
        Feedback controller u = controller(x, t)
        State is primary argument (matches discrete convention)
    
    Returns
    -------
    RolloutResult
        Includes 'controller_type' and 'closed_loop' fields
    """
```

---

## 4. Override simulate() and rollout() in ContinuousStochasticSystem

### Problem
- `ContinuousStochasticSystem.integrate()` supports `n_paths` and `seed`
- `ContinuousStochasticSystem.simulate()` does NOT (inherits from parent)
- No way to do regular-grid SDE simulation with Monte Carlo

### Solution
**Override `simulate()` AND `rollout()` to add `n_paths` and `seed` parameters**

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
    ) -> Union[SimulationResult, SDESimulationResult]:
        """
        Simulate stochastic system with regular time grid.
        
        Supports Monte Carlo simulation via n_paths parameter.
        
        Returns
        -------
        SimulationResult (if n_paths=1) or SDESimulationResult (if n_paths>1)
        """
    
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
    ) -> Union[RolloutResult, SDERolloutResult]:
        """
        Rollout stochastic system with state feedback.
        
        Supports Monte Carlo simulation via n_paths parameter.
        
        Returns
        -------
        RolloutResult (if n_paths=1) or SDERolloutResult (if n_paths>1)
        """
```

**DO NOT add `simulate_stochastic()`** - redundant with parameter-extended `simulate()`

---

## 5. Standardize Return Dictionary Keys

### Problem
- `integrate()` returns: `{'t': ..., 'x': ..., ...}` ✅
- `simulate()` returns: `{'time': ..., 'states': ..., ...}` ❌
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

**Benefit**: After refactoring, `integrate()`, `simulate()`, and `rollout()` all return compatible formats!

---

## 6. Discrete Systems: Update Return Format

### Current Issues
- Key name: `'time_steps'` → change to `'t'`
- Shape: `(nx, T)` → change to `(T, nx)`
- Key name: `'states'` → change to `'x'`
- Key name: `'controls'` → change to `'u'`

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
    't': NDArray,             # (n_steps+1,) ✅
    'x': NDArray,             # (n_steps+1, nx) ✅ time-major
    'u': NDArray,             # (n_steps, nu) ✅ time-major
    'dt': float,
    'success': bool,          # ✅ add for consistency
    'message': str,           # ✅ add for consistency
    'method': str,
    'metadata': dict
})
```

---

## 7. Add Stochastic Discrete Result Types

### Problem
- Discrete stochastic systems have `simulate_stochastic()` method
- No proper return type defined for Monte Carlo results

### Solution
**Create `DiscreteStochasticSimulationResult` and `DiscreteStochasticRolloutResult`**

```python
class DiscreteStochasticSimulationResult(DiscreteSimulationResultBase):
    """Monte Carlo discrete simulation result."""
    # Required stochastic fields
    n_paths: int              # Number of paths
    noise_type: str          # Noise structure
    # Optional stochastic fields
    seed: Optional[int]
    noise_samples: ArrayLike
```

**Shape convention**:
- Single path (n_paths=1): `x` is `(T, nx)`
- Multiple paths: `x` is `(n_paths, T, nx)`

---

## 8. Update All Implementations

### Files to Modify

#### Core System Files
- `continuous_system_base.py` - add `rollout()`, refactor `simulate()`
- `continuous_stochastic_system.py` - override `simulate()` and `rollout()`
- `discrete_system_base.py` - update return format to time-major
- `discrete_symbolic_system.py` - update `simulate()` implementation
- `discrete_stochastic_system.py` - update `simulate_stochastic()`
- `discretized_system.py` - update both `simulate()` methods

#### Type Definitions
- **Create** `src/cdesym/types/system_results.py` with all 10 types
- **Update** `src/cdesym/types/trajectories.py` to import from system_results
- **Update** `src/cdesym/types/__init__.py` to export all types

#### Plotting Infrastructure
- Verify `plotter.plot_trajectory()` works with new format (should already work!)
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

## 9. Summary of Breaking Changes

### API Changes (User-Facing)

**1. Discrete result indexing** (BREAKING):
```python
# OLD
result['states'][i, :]  # i-th state over time
result['states'][:, k]  # All states at time k

# NEW
result['x'][:, i]       # i-th state over time
result['x'][k, :]       # All states at time k
```

**2. Continuous simulate() signature** (BREAKING):
```python
# OLD
result = system.simulate(x0, controller=my_controller, ...)

# NEW - open-loop
result = system.simulate(x0, u=my_u_func, ...)

# NEW - closed-loop
result = system.rollout(x0, controller=my_controller, ...)
```

**3. Result dictionary keys** (BREAKING):
```python
# OLD
result['time'], result['states'], result['controls']

# NEW
result['t'], result['x'], result['u']
```

**4. Stochastic simulation** (NEW CAPABILITY):
```python
# Single path (returns SimulationResult)
result = sde_system.simulate(x0, u=None, t_span=(0, 10), dt=0.01, n_paths=1)

# Monte Carlo (returns SDESimulationResult)
result = sde_system.simulate(x0, u=None, t_span=(0, 10), dt=0.01, 
                             n_paths=500, seed=42)
assert result['x'].shape == (500, 1001, 2)  # (n_paths, T, nx)
```

---

## 10. Benefits

### Consistency
- ✅ Same shape convention everywhere: `(T, nx)`
- ✅ Same method names: `simulate()` and `rollout()`
- ✅ Same semantics: open-loop vs closed-loop
- ✅ Same return format: `{'t', 'x', 'u', ...}`
- ✅ Complete type system with proper hierarchy

### Usability
- ✅ No mental overhead remembering which convention applies
- ✅ Generic plotting code works for all systems
- ✅ Pandas integration: `pd.DataFrame(result['x'], index=result['t'])`
- ✅ Clear separation of concerns: simulate vs rollout
- ✅ Polymorphic code via `SystemResult` union type

### Compatibility
- ✅ Aligns with modern Python ecosystem (pandas, sklearn, PyTorch)
- ✅ Makes library easier to learn and teach
- ✅ Reduces documentation burden
- ✅ Type safety with comprehensive TypedDict hierarchy

---

## 11. Migration Checklist

### Type System
- [ ] Create `src/cdesym/types/system_results.py` with all 20 types
- [ ] Update `src/cdesym/types/trajectories.py` to import from system_results
- [ ] Update `src/cdesym/types/__init__.py` to export all types
- [ ] Add module docstring to system_results.py

### Continuous Systems
- [ ] Add `ContinuousSystemBase.rollout()` method
- [ ] Refactor `ContinuousSystemBase.simulate()` (remove controller param)
- [ ] Override `ContinuousStochasticSystem.simulate()` (add n_paths, seed)
- [ ] Override `ContinuousStochasticSystem.rollout()` (add n_paths, seed)

### Discrete Systems
- [ ] Update `DiscreteSymbolicSystem.simulate()` to time-major
- [ ] Update `DiscreteStochasticSystem.simulate_stochastic()` to time-major
- [ ] Update `DiscretizedSystem.simulate()` to time-major
- [ ] Update `DiscretizedSystem.simulate_stochastic()` to time-major
- [ ] Verify `DiscreteSystemBase.rollout()` uses correct keys

### Infrastructure
- [ ] Update all plotting code (should be minimal!)
- [ ] Update all tests (~50 test files)
- [ ] Update all examples (~20 examples)
- [ ] Update all tutorials (~5 tutorials)
- [ ] Update documentation

### Verification
- [ ] Run full test suite
- [ ] Review all docstrings
- [ ] Test all examples
- [ ] Build documentation
- [ ] Create migration guide

---

## 12. Effort Estimate

### Breakdown

| Task | Hours | Notes |
|------|-------|-------|
| **Type System** | 3 | Create system_results.py, update trajectories.py, __init__.py |
| **Continuous Base** | 4 | Add rollout(), refactor simulate() |
| **Continuous Stochastic** | 6 | Override simulate() and rollout() with n_paths/seed |
| **Discrete Base** | 5 | Convert to time-major ordering |
| **Discrete Stochastic** | 4 | Update stochastic methods, add new types |
| **Tests** | 25 | ~50 test files × 30 min average |
| **Examples** | 5 | ~20 examples × 15 min |
| **Tutorials** | 5 | ~5 tutorials × 1 hour |
| **Plotting** | 2 | Verify compatibility (should be minimal) |
| **Documentation** | 5 | Docstrings, migration guide, CHANGELOG |
| **Verification** | 3 | Final testing and review |
| **Migration Guide** | 2 | Write comprehensive guide |

**Total**: ~78 hours (~2 weeks at 40 hours/week)

### Risk Assessment

**Risk**: LOW - No public users yet

**Complexity**: MODERATE-HIGH - Touches many files but changes are mechanical

**Testing Burden**: HIGH - Must verify all tests pass with new conventions

**Documentation Burden**: MODERATE - Update examples and tutorials

**Recommendation**: ✅ Execute before v1.0 release to PyPI

---

## 13. Type System Summary

### The 10 Concrete Result Types

**Continuous** (6 types):
1. `IntegrationResult` - ODE, adaptive grid ✅ already correct
2. `SimulationResult` - ODE, regular grid
3. `RolloutResult` - ODE, closed-loop
4. `SDEIntegrationResult` - SDE, adaptive grid ✅ already correct
5. `SDESimulationResult` - SDE, regular grid
6. `SDERolloutResult` - SDE, closed-loop

**Discrete** (4 types):
7. `DiscreteSimulationResult` - Deterministic, open-loop
8. `DiscreteRolloutResult` - Deterministic, closed-loop
9. `DiscreteStochasticSimulationResult` - Stochastic, open-loop
10. `DiscreteStochasticRolloutResult` - Stochastic, closed-loop

### The 6 Union Types

1. `ContinuousIntegrationResultUnion` = `IntegrationResult | SDEIntegrationResult`
2. `ContinuousSimulationResultUnion` = `SimulationResult | SDESimulationResult`
3. `ContinuousRolloutResultUnion` = `RolloutResult | SDERolloutResult`
4. `DiscreteSimulationResultUnion` = `DiscreteSimulationResult | DiscreteStochasticSimulationResult`
5. `DiscreteRolloutResultUnion` = `DiscreteRolloutResult | DiscreteStochasticRolloutResult`
6. `SystemResult` = Union of ALL 10 types

### Usage Example

```python
from cdesym.types import SystemResult

def analyze_any_result(result: SystemResult) -> dict:
    """Works with ANY system execution result."""
    t = result['t']
    x = result['x']
    
    # Handle different dimensions
    if x.ndim == 2:
        # Single trajectory: (T, nx)
        final_state = x[-1]
    else:
        # Multiple paths: (n_paths, T, nx)
        final_state = x[:, -1]  # Final state of each path
    
    return {
        'duration': t[-1] - t[0],
        'final_state': final_state,
        'is_stochastic': 'n_paths' in result,
    }
```

---

## Conclusion

This refactoring establishes a **clean, consistent, and comprehensive** API for v1.0:

- ✅ **Time-major everywhere** - aligns with modern Python ecosystem
- ✅ **Clear method separation** - simulate (open-loop) vs rollout (closed-loop)
- ✅ **Complete type system** - 10 result types + 6 union types
- ✅ **Stochastic support** - proper types for Monte Carlo simulation
- ✅ **Plotting compatibility** - all results work with same plotting code
- ✅ **Type safety** - comprehensive TypedDict hierarchy with inheritance
- ✅ **No public users** - perfect timing for breaking changes

**Total effort**: ~78 hours, well worth it for a stable v1.0 foundation.
