# ControlDESymulation Refactoring Implementation Plan

## Overview

This refactoring standardizes the API before v1.0 release by:

1. **Time-major ordering** for all arrays: `(T, nx)` not `(nx, T)`
2. **Consistent key names**: `'t'`, `'x'`, `'u'` (not `'time'`, `'states'`, `'controls'`)
3. **Method separation**: `simulate()` for open-loop, `rollout()` for closed-loop
4. **Complete type system**: 10 result types (deterministic + stochastic) organized in `system_results.py`

**Estimated effort**: ~78 hours (~2 weeks)

**Breaking changes**: YES - but no public users yet, so this is the optimal time

---

## Phase 1: Type System Update (NEW)

### 1.3.1 Update trajectories.py unit test suite and debug

### 1.3.2 Add system_result.py unit test suite and debug

### 1.4 Commit Type Changes

```bash
git add src/cdesym/types/system_results.py
git add src/cdesym/types/trajectories.py
git add src/cdesym/types/__init__.py
git commit -m "refactor: Create comprehensive type system for execution results

- Add system_results.py with 10 result types (deterministic + stochastic)
- Update trajectories.py to import from system_results
- Export all types from types/__init__.py
- Maintain backward compatibility via re-exports"
```

---

## Phase 2: Continuous System Base (Deterministic)

**Estimated time**: 4 hours

### 2.1 Add rollout() Method

**File**: `src/cdesym/systems/base/core/continuous_system_base.py`

Add new method:

```python
def rollout(
    self,
    x0: StateVector,
    controller: Optional[FeedbackController] = None,
    t_span: TimeSpan = (0.0, 10.0),
    dt: ScalarLike = 0.01,
    method: IntegrationMethod = "RK45",
    **kwargs
) -> RolloutResult:
    """
    Closed-loop simulation with state-feedback controller.
    
    Parameters
    ----------
    controller : Optional[Callable[[StateVector, float], ControlVector]]
        Feedback controller u = controller(x, t)
        State is primary argument (matches discrete convention)
        If None, uses zero control
    t_span : TimeSpan
        Time interval (t_start, t_end)
    dt : ScalarLike
        Time step for regular grid
    method : IntegrationMethod
        Integration method to use
    
    Returns
    -------
    RolloutResult
        Dictionary with keys: 't', 'x', 'u', 'success', 'message', 
        'method', 'dt', 'metadata', 'controller_type', 'closed_loop'
    
    Examples
    --------
    >>> def controller(x, t):
    ...     K = np.array([[-1.0, -2.0]])
    ...     return -K @ x
    >>> 
    >>> result = system.rollout(
    ...     x0=np.array([1.0, 0.0]),
    ...     controller=controller,
    ...     t_span=(0.0, 10.0),
    ...     dt=0.01
    ... )
    """
    # Convert controller(x, t) to u_func(t, x) for integrate()
    if controller is not None:
        def u_func(t, x):
            return controller(x, t)
    else:
        u_func = None
    
    # Call integrate with converted control
    result = self.integrate(
        x0=x0,
        u=u_func,
        t_span=t_span,
        method=method,
        dt=dt,
        **kwargs
    )
    
    # Convert to RolloutResult format
    return {
        't': result['t'],
        'x': result['x'],
        'u': result.get('u'),  # May not be present
        'success': result['success'],
        'message': result['message'],
        'method': result.get('method', method),
        'dt': dt,
        'metadata': result.get('metadata', {}),
        'controller_type': 'feedback' if controller else 'zero',
        'closed_loop': True,
    }
```

### 2.2 Refactor simulate() Method

**Remove** `controller` parameter:

```python
# OLD signature
def simulate(self, x0, controller=None, t_span=(0, 10), dt=0.01, **kwargs):
    
# NEW signature
def simulate(
    self,
    x0: StateVector,
    u: Optional[Union[ControlVector, TimeVaryingControl]] = None,
    t_span: TimeSpan = (0.0, 10.0),
    dt: ScalarLike = 0.01,
    method: IntegrationMethod = "RK45",
    **kwargs
) -> SimulationResult:
    """
    Open-loop simulation with time-varying control.
    
    Parameters
    ----------
    u : Optional[Union[ControlVector, Callable[[float], ControlVector]]]
        Control input:
        - None: Zero control
        - Array (nu,): Constant control
        - Callable u(t): Time-varying control (NO state feedback)
    
    Returns
    -------
    SimulationResult
        Dictionary with keys: 't', 'x', 'u', 'success', 'message',
        'method', 'dt', 'metadata'
    
    Examples
    --------
    >>> # Open-loop with no control
    >>> result = system.simulate(x0, u=None, t_span=(0, 10), dt=0.01)
    >>> 
    >>> # Open-loop with time-varying control
    >>> u_func = lambda t: np.array([np.sin(t)])
    >>> result = system.simulate(x0, u=u_func, t_span=(0, 10), dt=0.01)
    """
    # Call integrate with dt to ensure regular grid
    result = self.integrate(
        x0=x0,
        u=u,
        t_span=t_span,
        method=method,
        dt=dt,
        **kwargs
    )
    
    # Convert to SimulationResult format (should already match)
    return {
        't': result['t'],
        'x': result['x'],
        'u': result.get('u'),
        'success': result['success'],
        'message': result['message'],
        'method': result.get('method', method),
        'dt': dt,
        'metadata': result.get('metadata', {}),
    }
```

### 2.3 Commit Continuous Base Changes

```bash
git add src/cdesym/systems/base/core/continuous_system_base.py
git commit -m "refactor: Add rollout() and refactor simulate() in ContinuousSystemBase

- Add rollout() method for closed-loop simulation
- Remove controller parameter from simulate() (now open-loop only)
- Add u parameter to simulate() for time-varying control
- Update return types to SimulationResult and RolloutResult"
```

---

## Phase 3: Continuous Stochastic System

**Estimated time**: 6 hours

### 3.1 Override simulate() for Stochastic

**File**: `src/cdesym/systems/base/core/continuous_stochastic_system.py`

```python
def simulate(
    self,
    x0: StateVector,
    u: Optional[Union[ControlVector, TimeVaryingControl]] = None,
    t_span: TimeSpan = (0.0, 10.0),
    dt: ScalarLike = 0.01,
    method: SDEIntegrationMethod = "euler_maruyama",
    n_paths: int = 1,           # NEW
    seed: Optional[int] = None, # NEW
    **kwargs
) -> Union[SimulationResult, SDESimulationResult]:
    """
    Simulate stochastic system with regular time grid.
    
    Supports Monte Carlo simulation via n_paths parameter.
    
    Parameters
    ----------
    n_paths : int
        Number of Monte Carlo paths (default: 1)
        - n_paths=1: Returns SimulationResult
        - n_paths>1: Returns SDESimulationResult
    seed : Optional[int]
        Random seed for reproducibility
    
    Returns
    -------
    SimulationResult or SDESimulationResult
        For n_paths=1: SimulationResult with shape (T, nx)
        For n_paths>1: SDESimulationResult with shape (n_paths, T, nx)
    """
    # Call integrate with n_paths and seed
    result = self.integrate(
        x0=x0,
        u=u,
        t_span=t_span,
        method=method,
        dt=dt,
        n_paths=n_paths,
        seed=seed,
        **kwargs
    )
    
    # Return appropriate type based on n_paths
    if n_paths == 1:
        return {
            't': result['t'],
            'x': result['x'],
            'u': result.get('u'),
            'success': result['success'],
            'message': result['message'],
            'method': method,
            'dt': dt,
            'metadata': result.get('metadata', {}),
        }
    else:
        return {
            't': result['t'],
            'x': result['x'],
            'u': result.get('u'),
            'success': result['success'],
            'message': result['message'],
            'method': method,
            'dt': dt,
            'metadata': result.get('metadata', {}),
            'n_paths': n_paths,
            'noise_type': result.get('noise_type'),
            'sde_type': result.get('sde_type'),
            'seed': seed,
            'noise_samples': result.get('noise_samples'),
            'diffusion_evals': result.get('diffusion_evals'),
        }
```

### 3.2 Override rollout() for Stochastic

```python
def rollout(
    self,
    x0: StateVector,
    controller: Optional[FeedbackController] = None,
    t_span: TimeSpan = (0.0, 10.0),
    dt: ScalarLike = 0.01,
    method: SDEIntegrationMethod = "euler_maruyama",
    n_paths: int = 1,           # NEW
    seed: Optional[int] = None, # NEW
    **kwargs
) -> Union[RolloutResult, SDERolloutResult]:
    """
    Rollout stochastic system with state feedback.
    
    Supports Monte Carlo simulation via n_paths parameter.
    
    Returns
    -------
    RolloutResult or SDERolloutResult
        For n_paths=1: RolloutResult
        For n_paths>1: SDERolloutResult
    """
    # Convert controller to u function
    if controller is not None:
        u_func = lambda t, x: controller(x, t)
    else:
        u_func = None
    
    # Call integrate
    result = self.integrate(
        x0=x0,
        u=u_func,
        t_span=t_span,
        method=method,
        dt=dt,
        n_paths=n_paths,
        seed=seed,
        **kwargs
    )
    
    # Return appropriate type
    if n_paths == 1:
        return {
            't': result['t'],
            'x': result['x'],
            'u': result.get('u'),
            'success': result['success'],
            'message': result['message'],
            'method': method,
            'dt': dt,
            'metadata': result.get('metadata', {}),
            'controller_type': 'feedback' if controller else 'zero',
            'closed_loop': True,
        }
    else:
        return {
            't': result['t'],
            'x': result['x'],
            'u': result.get('u'),
            'success': result['success'],
            'message': result['message'],
            'method': method,
            'dt': dt,
            'metadata': result.get('metadata', {}),
            'controller_type': 'feedback' if controller else 'zero',
            'closed_loop': True,
            'n_paths': n_paths,
            'noise_type': result.get('noise_type'),
            'sde_type': result.get('sde_type'),
            'seed': seed,
            'noise_samples': result.get('noise_samples'),
            'diffusion_evals': result.get('diffusion_evals'),
        }
```

### 3.3 Commit Stochastic Changes

```bash
git add src/cdesym/systems/base/core/continuous_stochastic_system.py
git commit -m "refactor: Override simulate() and rollout() for stochastic systems

- Add n_paths and seed parameters to simulate()
- Add n_paths and seed parameters to rollout()
- Return SDESimulationResult/SDERolloutResult when n_paths > 1
- Maintain backward compatibility with n_paths=1"
```

---

## Phase 4: Discrete System Base

**Estimated time**: 5 hours

### 4.1 Update simulate() to Time-Major

**File**: `src/cdesym/systems/base/core/discrete_system_base.py`

```python
def simulate(
    self,
    x0: StateVector,
    u_sequence: Optional[DiscreteControlInput] = None,
    n_steps: int = 100,
    **kwargs
) -> DiscreteSimulationResult:
    """
    Simulate discrete system forward in time.
    
    **BREAKING CHANGE**: Now returns time-major arrays (T, nx) not (nx, T)
    
    Returns
    -------
    DiscreteSimulationResult
        Dictionary with:
        - 't': Time steps [0, 1, ..., n_steps]
        - 'x': States (n_steps+1, nx) - TIME-MAJOR
        - 'u': Controls (n_steps, nu) - TIME-MAJOR
        - 'dt': Sampling period
        - 'success': bool
        - 'message': str
        - 'method': str
        - 'metadata': dict
    """
    # Allocate arrays - TIME-MAJOR
    states = np.zeros((n_steps + 1, self.nx))  # (T, nx)
    controls = np.zeros((n_steps, self.nu)) if self.nu > 0 else None
    
    # Set initial state
    states[0] = x0
    
    # Forward simulation
    x = x0
    for k in range(n_steps):
        # Get control
        u = self._get_control_at_step(u_sequence, k)
        
        # Store control
        if controls is not None:
            controls[k] = u
        
        # Step forward
        x = self.step(x, u, k)
        states[k + 1] = x
    
    return {
        't': np.arange(n_steps + 1),
        'x': states,  # (n_steps+1, nx)
        'u': controls,  # (n_steps, nu) or None
        'dt': self._dt,
        'success': True,
        'message': 'Simulation completed',
        'method': 'discrete_step',
        'metadata': {},
    }
```

### 4.2 Verify rollout() is Already Time-Major

**Check**: `rollout()` should already return time-major. Just update key names if needed:

- `'time_steps'` → `'t'`
- Confirm shape is already `(T, nx)`

### 4.3 Commit Discrete Base Changes

```bash
git add src/cdesym/systems/base/core/discrete_system_base.py
git commit -m "refactor: Convert discrete simulate() to time-major ordering

- Change states from (nx, T) to (T, nx)
- Change controls from (nu, T) to (T, nu)
- Update key names: 'time_steps' -> 't'
- Maintain same logic, just transposed output"
```

---

## Phase 5: Discrete Stochastic System

**Estimated time**: 4 hours

### 5.1 Update simulate_stochastic()

**File**: `src/cdesym/systems/base/core/discrete_stochastic_system.py`

```python
def simulate_stochastic(
    self,
    x0: StateVector,
    u_sequence: Optional[DiscreteControlInput] = None,
    n_steps: int = 100,
    n_paths: int = 1,
    seed: Optional[int] = None,
    **kwargs
) -> Union[DiscreteSimulationResult, DiscreteStochasticSimulationResult]:
    """
    Monte Carlo simulation of stochastic discrete system.
    
    Returns
    -------
    DiscreteSimulationResult or DiscreteStochasticSimulationResult
        For n_paths=1: DiscreteSimulationResult
        For n_paths>1: DiscreteStochasticSimulationResult with (n_paths, T, nx)
    """
    if seed is not None:
        np.random.seed(seed)
    
    if n_paths == 1:
        # Single path - use regular simulate with noise
        return self._simulate_single_path(x0, u_sequence, n_steps, **kwargs)
    else:
        # Multiple paths - Monte Carlo
        states = np.zeros((n_paths, n_steps + 1, self.nx))
        
        for i in range(n_paths):
            result = self._simulate_single_path(x0, u_sequence, n_steps, **kwargs)
            states[i] = result['x']
        
        return {
            't': np.arange(n_steps + 1),
            'x': states,  # (n_paths, T, nx)
            'u': result.get('u'),  # Assumes same control for all paths
            'dt': self._dt,
            'success': True,
            'message': 'Stochastic simulation completed',
            'method': 'discrete_stochastic',
            'metadata': {},
            'n_paths': n_paths,
            'noise_type': self.noise_type,
            'seed': seed,
        }
```

### 5.2 Commit Discrete Stochastic Changes

```bash
git add src/cdesym/systems/base/core/discrete_stochastic_system.py
git commit -m "refactor: Update discrete stochastic to return proper result types

- Return DiscreteStochasticSimulationResult for n_paths > 1
- Use time-major ordering (n_paths, T, nx)
- Add proper type annotations"
```

---

## Phase 6: Update All Tests

**Estimated time**: 25 hours

### 6.1 Update Continuous System Tests

**Files**: `tests/test_continuous_system.py`, `tests/test_continuous_symbolic_system.py`

**Changes needed**:

```python
# OLD
result = system.simulate(x0, controller=my_controller, ...)
states = result['states']  # (nx, T)
x1 = states[0, :]

# NEW - open-loop
result = system.simulate(x0, u=None, ...)
x = result['x']  # (T, nx)
x1 = x[:, 0]

# NEW - closed-loop
result = system.rollout(x0, controller=my_controller, ...)
x = result['x']  # (T, nx)
x1 = x[:, 0]
```

### 6.2 Update Discrete System Tests

**Files**: `tests/test_discrete_system.py`

```python
# OLD
result = discrete_sys.simulate(x0, u_seq, n_steps=100)
states = result['states']  # (nx, 101)
x1 = states[0, :]

# NEW
result = discrete_sys.simulate(x0, u_seq, n_steps=100)
x = result['x']  # (101, nx)
x1 = x[:, 0]
```

### 6.3 Update Stochastic Tests

**Files**: `tests/test_stochastic_system.py`

```python
# NEW - single path
result = sde_system.simulate(x0, u=None, t_span=(0, 10), dt=0.01, n_paths=1)
assert result['x'].shape == (1001, 2)

# NEW - Monte Carlo
result = sde_system.simulate(x0, u=None, t_span=(0, 10), dt=0.01, n_paths=500, seed=42)
assert result['x'].shape == (500, 1001, 2)
assert 'n_paths' in result
```

### 6.4 Test Strategy

For each test file:

1. Search for `result['time']` → replace with `result['t']`
2. Search for `result['states']` → replace with `result['x']`
3. Search for `result['controls']` → replace with `result['u']`
4. Search for `.shape[0]` patterns → verify indexing is correct
5. Search for `[:, k]` or `[i, :]` → transpose as needed

### 6.5 Run Tests Incrementally

```bash
# After updating each test file
pytest tests/test_continuous_system.py -v

# Fix any failures before moving on
# Repeat for all test files
```

### 6.6 Commit Test Updates

```bash
git add tests/
git commit -m "refactor: Update all tests for time-major convention and new API

- Update key names: 'time' -> 't', 'states' -> 'x', 'controls' -> 'u'
- Update indexing for time-major: (T, nx) not (nx, T)
- Update method calls: simulate() vs rollout()
- Verify all tests pass"
```

---

## Phase 7: Update Examples and Tutorials

**Estimated time**: 10 hours

### 7.1 Update Examples

**Files**: `examples/*.py`

For each example:

1. Update `simulate()` calls to use new API
2. Update result key access
3. Update array indexing
4. Test that example runs without errors

### 7.2 Update Tutorials

**Files**: `docs/tutorials/*.qmd`

Since tutorials will be redone after refactoring, just:

1. Add note at top: "Tutorial being updated for v1.0 API"
2. Or create new versions with v1.0 API
3. Keep old versions as `*_legacy.qmd` temporarily

### 7.3 Update README Examples

**File**: `README.md`

Update code snippets to use new API:

```python
# Basic usage example
result = system.simulate(x0, u=None, t_span=(0, 10), dt=0.01)
plt.plot(result['t'], result['x'][:, 0])

# Closed-loop example
result = system.rollout(x0, controller=controller, t_span=(0, 10), dt=0.01)
plt.plot(result['t'], result['x'][:, 0])
```

### 7.4 Commit Documentation Updates

```bash
git add examples/ docs/ README.md
git commit -m "docs: Update examples and tutorials for v1.0 API

- Update all example scripts
- Add notes to tutorials about API changes
- Update README code snippets"
```

---

## Phase 8: Update Plotting Infrastructure

**Estimated time**: 2 hours

### 8.1 Verify Plotting Compatibility

**Files**: `src/cdesym/plotting/plotter.py`

**Good news**: Plotting should already work! It expects:

```python
def plot_trajectory(result):
    t = result['t']
    x = result['x']  # (T, nx)
    plt.plot(t, x[:, i])
```

This matches our new format.

**If there are any issues**:

- Remove any checks for `'time'` or `'states'` keys
- Update to use `'t'` and `'x'` universally

### 8.2 Test Plotting

```python
# Create simple test
result = pendulum.simulate(x0, u=None, t_span=(0, 10), dt=0.01)
pendulum.plot(result)  # Should work!

result = pendulum.rollout(x0, controller=controller, t_span=(0, 10), dt=0.01)
pendulum.plot(result)  # Should also work!
```

### 8.3 Commit Plotting Updates (if any)

```bash
git add src/cdesym/plotting/
git commit -m "refactor: Update plotting to use standardized result format"
```

---

## Phase 9: Final Verification

**Estimated time**: 3 hours

### 9.1 Run Full Test Suite

```bash
pytest -v > tests_after_refactoring.log 2>&1

# Compare with baseline
diff tests_before_refactoring.log tests_after_refactoring.log
```

**Expected**: Same number of passing tests (or more if you added tests)

### 9.2 Manual Testing Checklist

- [ ] Continuous simulate() works (open-loop)
- [ ] Continuous rollout() works (closed-loop)
- [ ] Stochastic simulate() works with n_paths=1
- [ ] Stochastic simulate() works with n_paths>1
- [ ] Stochastic rollout() works
- [ ] Discrete simulate() works
- [ ] Discrete rollout() works
- [ ] Discrete stochastic works
- [ ] Plotting works with all result types
- [ ] Examples run without errors

### 9.3 Run Examples

```bash
for example in examples/*.py; do
    echo "Running $example"
    python "$example" || echo "FAILED: $example"
done
```

### 9.4 Documentation Build

```bash
cd docs
make html
# Check for broken links or missing references
```

---

## Phase 10: Migration Guide and Release

**Estimated time**: 2 hours

### 10.1 Create Migration Guide

**File**: `MIGRATION_v1.0.md`

Document all breaking changes with examples:

```markdown
# Migration Guide to v1.0

## Breaking Changes

### 1. Result Key Names
- `'time'` → `'t'`
- `'states'` → `'x'`
- `'controls'` → `'u'`
- `'time_steps'` → `'t'`

### 2. Array Shapes
All results now use time-major ordering:
- Continuous: `(T, nx)` instead of `(nx, T)`
- Discrete: `(T, nx)` instead of `(nx, T)`

### 3. Method Signatures
`simulate()` is now open-loop only:
```python
# OLD
result = system.simulate(x0, controller=my_controller, ...)

# NEW - open-loop
result = system.simulate(x0, u=u_func, ...)

# NEW - closed-loop
result = system.rollout(x0, controller=my_controller, ...)
```

### 4. Migration Examples

[Include comprehensive before/after examples]

```

### 10.2 Update CHANGELOG
**File**: `CHANGELOG.md`

```markdown
## [1.0.0] - 2025-XX-XX

### Breaking Changes
- **Result format standardization**: All result dictionaries now use consistent keys ('t', 'x', 'u') and time-major shape convention (T, nx)
- **Method separation**: `simulate()` is now open-loop only; use `rollout()` for closed-loop simulation with state feedback
- **Type system**: Comprehensive result type hierarchy with 10 types for deterministic/stochastic variants

### Added
- `rollout()` method for closed-loop simulation on all system types
- Stochastic variants: `SDESimulationResult`, `SDERolloutResult`, `DiscreteStochasticSimulationResult`, etc.
- Union type `SystemResult` for polymorphic code
- Complete type system in `cdesym.types.system_results`

### Changed
- All result arrays use time-major ordering: (T, nx) not (nx, T)
- Result dictionary keys standardized: 't', 'x', 'u' (not 'time', 'states', 'controls')
- `simulate()` signature: removed `controller` parameter, added `u` parameter

### Migration
See MIGRATION_v1.0.md for detailed migration guide.
```

### 10.3 Final Commit and Tag

```bash
git add MIGRATION_v1.0.md CHANGELOG.md
git commit -m "docs: Add migration guide and changelog for v1.0"

# Tag the release
git tag -a v1.0.0 -m "Release v1.0.0: API standardization"

# Merge to main
git checkout main
git merge refactor/v1.0-api-standardization

# Push everything
git push origin main
git push origin v1.0.0
```

---

## Troubleshooting Common Issues

### Issue: Tests fail with "KeyError: 'states'"

**Solution**: Search for `result['states']` and replace with `result['x']`

### Issue: Shape assertions fail

**Solution**: Update shape checks from `(nx, T)` to `(T, nx)`:

```python
# OLD
assert result['states'].shape == (2, 1001)

# NEW
assert result['x'].shape == (1001, 2)
```

### Issue: Plotting shows transposed data

**Solution**: Check that you're accessing `result['x'][:, i]` not `result['x'][i, :]`

### Issue: Controller signature errors

**Solution**: Make sure controller signature is `controller(x, t)` not `controller(t, x)`

---

## Summary Checklist

Before merging to main, verify:

- [ ] All 10 result types defined in `system_results.py`
- [ ] `trajectories.py` updated to import from `system_results`
- [ ] `types/__init__.py` exports all new types
- [ ] `rollout()` added to continuous systems
- [ ] `simulate()` refactored to remove controller parameter
- [ ] Stochastic systems override `simulate()` and `rollout()`
- [ ] Discrete systems use time-major ordering
- [ ] All tests updated and passing
- [ ] All examples updated and working
- [ ] Plotting infrastructure verified
- [ ] Documentation updated
- [ ] Migration guide created
- [ ] CHANGELOG updated
- [ ] Release tagged

**Estimated total time**: ~78 hours (~2 weeks at 40 hours/week)
