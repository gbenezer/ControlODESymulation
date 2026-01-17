# Discrete vs Continuous `simulate()` Methods: Inconsistency Analysis

## Side-by-Side Comparison

### ContinuousSystemBase.simulate()

```python
def simulate(
    self,
    x0: StateVector,
    controller: Optional[FeedbackController] = None,  # (x, t) -> u
    t_span: TimeSpan = (0.0, 10.0),
    dt: ScalarLike = 0.01,
    method: IntegrationMethod = "RK45",
    **kwargs
) -> SimulationResult
```

**Returns**: `SimulationResult` TypedDict
```python
{
    'time': np.ndarray,      # (T,) - regular time grid
    'states': np.ndarray,    # (T, nx) - TIME-MAJOR
    'controls': np.ndarray,  # (T, nu) or None
    'metadata': dict         # {method, dt, success, nfev}
}
```

---

### DiscreteSystemBase.simulate()

```python
def simulate(
    self,
    x0: StateVector,
    u_sequence: DiscreteControlInput = None,  # Various formats
    n_steps: int = 100,
    **kwargs
) -> DiscreteSimulationResult
```

**Returns**: `DiscreteSimulationResult` TypedDict
```python
{
    'states': np.ndarray,      # (nx, n_steps+1) - STATE-MAJOR
    'controls': np.ndarray,    # (nu, n_steps) or None
    'time_steps': np.ndarray,  # [0, 1, 2, ..., n_steps]
    'dt': float,               # Sampling period
    'metadata': dict           # {method, success, ...}
}
```

---

## Critical Inconsistencies

### 1. **Array Shape Convention** ⚠️ MAJOR

| Aspect | Continuous | Discrete | Consistent? |
|--------|------------|----------|-------------|
| States shape | `(T, nx)` | `(nx, T)` | ❌ NO |
| Controls shape | `(T, nu)` | `(nu, T)` | ❌ NO |
| Convention | **TIME-MAJOR** | **STATE-MAJOR** | ❌ NO |

**Example**:
```python
# Continuous: time-major
cont_result = cont_sys.simulate(x0, t_span=(0, 10), dt=0.1)
cont_result['states'].shape  # (100, nx) - rows are time points
cont_result['states'][50, 0]  # State x[0] at t=50*0.1

# Discrete: state-major  
disc_result = disc_sys.simulate(x0, n_steps=100)
disc_result['states'].shape  # (nx, 101) - columns are time points
disc_result['states'][0, 50]  # State x[0] at k=50
```

**Impact**:
- User must remember which convention applies
- Can't write generic plotting code
- Transpose gymnastics when converting between types
- Error-prone indexing

**Why this happened**:
- Continuous: Modern ML/data science uses time-major (pandas, scikit-learn)
- Discrete: Legacy control theory uses state-major (MATLAB convention)

---

### 2. **Control Input Philosophy** ⚠️ MODERATE

| Aspect | Continuous | Discrete |
|--------|------------|----------|
| Parameter name | `controller` | `u_sequence` |
| Type | `FeedbackController` | `DiscreteControlInput` |
| Signature | `(x, t) -> u` | Various (see below) |
| Primary use | State feedback | Open-loop / time-indexed |

**Continuous control**:
```python
# Only accepts state-feedback callable
def controller(x, t):
    return -K @ x

result = system.simulate(x0, controller=controller, ...)
```

**Discrete control**:
```python
# Accepts multiple formats:
# 1. None (zero control)
result = system.simulate(x0, u_sequence=None, ...)

# 2. Constant array
result = system.simulate(x0, u_sequence=np.array([1.0]), ...)

# 3. Pre-computed sequence
u_seq = np.array([[0.1], [0.2], [0.3], ...])  # (n_steps, nu)
result = system.simulate(x0, u_sequence=u_seq, ...)

# 4. Time-indexed function
def u_func(k):  # NOT (x, k) !
    return np.array([np.sin(k * dt)])
result = system.simulate(x0, u_sequence=u_func, ...)
```

**Problem**: Continuous `simulate()` doesn't handle open-loop (time-varying without state feedback) naturally:
```python
# Continuous: awkward for time-only control
def controller(x, t):  # Must accept x even if not used
    return np.array([np.sin(t)])  # Ignore x
    
result = cont_sys.simulate(x0, controller=controller, ...)
```

**Discrete has `rollout()`** for state feedback:
```python
def policy(x, k):
    return -K @ x
    
result = disc_sys.rollout(x0, policy=policy, n_steps=100)
```

So discrete has:
- `simulate()` → open-loop focus
- `rollout()` → closed-loop focus

Continuous only has:
- `simulate()` → closed-loop focus (no open-loop method!)

---

### 3. **Time Specification** ⚠️ MODERATE

| Aspect | Continuous | Discrete |
|--------|------------|----------|
| Time parameter | `t_span=(t0, tf)` | `n_steps=100` |
| Grid parameter | `dt=0.01` | (uses `self._dt`) |
| Flexibility | Arbitrary start/end | Always starts at k=0 |

**Continuous**:
```python
# Can simulate any time interval
result = system.simulate(x0, t_span=(5.0, 15.0), dt=0.01)
# Returns times [5.0, 5.01, 5.02, ..., 15.0]
```

**Discrete**:
```python
# Always starts at k=0
result = system.simulate(x0, n_steps=100)
# Returns time_steps [0, 1, 2, ..., 100]
```

**Inconsistency**: Can't specify arbitrary start time for discrete systems

**Also**: Discrete uses object's `self._dt` attribute, continuous takes `dt` as parameter

---

### 4. **Return Type Keys** ⚠️ MINOR

| Key | Continuous | Discrete | Notes |
|-----|------------|----------|-------|
| Time | `'time'` | `'time_steps'` | Different names |
| States | `'states'` | `'states'` | ✅ Same |
| Controls | `'controls'` | `'controls'` | ✅ Same |
| Sampling | (none) | `'dt'` | Discrete includes dt |
| Metadata | `'metadata'` | `'metadata'` | ✅ Same |

**Minor inconsistency**: Time array has different key names

---

### 5. **Number of Points Returned** ⚠️ MINOR

| Aspect | Continuous | Discrete |
|--------|------------|----------|
| Input | `t_span=(0, 10)`, `dt=0.1` | `n_steps=100` |
| Output length | `T = (10-0)/0.1 + 1 = 101` | `n_steps + 1 = 101` |
| Includes x0? | ✅ Yes | ✅ Yes |

**Actually consistent** - both include initial state

But:
- Continuous: Length determined by `(tf - t0) / dt`
- Discrete: Length is `n_steps + 1`

---

### 6. **Docstring Philosophy** ⚠️ MODERATE

**Continuous docstring says**:
> "This method wraps integrate() and post-processes the result to provide a regular time grid and cleaner output. This is the recommended method for most use cases **(currently, may be deprecated once DiscretizedSystem is developed)**."

**Questions**:
1. Why would discrete systems deprecate continuous `simulate()`?
2. Is there a plan to unify?
3. What's the end goal?

---

## Summary of Inconsistencies

### Major Issues ❌

1. **Array shape convention completely opposite**
   - Continuous: `(T, nx)` time-major
   - Discrete: `(nx, T)` state-major
   - **Impact**: HIGH - affects all user code

2. **Control input philosophy differs**
   - Continuous: state-feedback focused (`controller`)
   - Discrete: open-loop focused (`u_sequence`) + separate `rollout()` for feedback
   - **Impact**: MODERATE - confusing mental model

### Moderate Issues ⚠️

3. **Time specification different**
   - Continuous: `t_span`, `dt` parameters
   - Discrete: `n_steps`, implicit `self._dt`
   - **Impact**: MODERATE - different calling patterns

4. **Return dict key names**
   - `'time'` vs `'time_steps'`
   - **Impact**: LOW - just documentation/convention

### Minor Issues ⚡

5. **Metadata organization**
   - Both have metadata but organize differently
   - **Impact**: LOW

---

## Root Causes

### 1. **Historical Decisions**

**Discrete systems** came from MATLAB control theory:
- State-major: `x = [x1; x2; x3]` is column vector
- Trajectories: `X = [x0, x1, x2, ...]` columns are time points
- Control sequences: `U = [u0, u1, u2, ...]`

**Continuous systems** adopted modern Python/ML conventions:
- Time-major: Rows are samples (time points)
- Compatible with pandas: `pd.DataFrame(states, index=time)`
- Compatible with sklearn, PyTorch, etc.

### 2. **Different Primary Use Cases**

**Continuous `simulate()`**:
- Primary: Closed-loop control with state feedback
- Secondary: Time-varying open-loop (awkward)

**Discrete `simulate()`**:
- Primary: Open-loop with pre-computed sequences
- Secondary: Time-indexed functions
- Separate `rollout()` for state feedback

### 3. **Design Evolution**

Systems evolved separately without sufficient coordination:
- Continuous added `simulate()` after `integrate()` existed
- Discrete had `simulate()` from the start
- No early standardization discussion

---

## Recommendations for Consistency

### Option A: Keep Both, Document Clearly ⚡

**Pros**:
- No breaking changes
- Each optimized for its domain
- Users specialize in one or the other

**Cons**:
- Confusing for users working with both
- Hard to build generic tools
- Code duplication in utilities

**Implementation**:
- Add prominent warnings in docs about convention differences
- Provide converter utilities
- Clear migration guides

---

### Option B: Standardize on Time-Major 🎯 RECOMMENDED

**Change discrete to match continuous**:
```python
# New discrete.simulate() return format
{
    'time_steps': np.ndarray,  # (T,) - keep name or change to 'time'
    'states': np.ndarray,      # (T, nx) - CHANGE from (nx, T)
    'controls': np.ndarray,    # (T, nu) - CHANGE from (nu, T)
    'dt': float,
    'metadata': dict
}
```

**Migration path**:
1. Add deprecation warning for accessing `result['states'][i, :]` (state-major)
2. Provide both conventions for 1-2 versions
3. Eventually remove state-major

**Pros**:
- ✅ Consistent with modern Python ecosystem
- ✅ Pandas-compatible
- ✅ Easier to write generic code
- ✅ Matches continuous convention

**Cons**:
- ❌ Breaking change for existing discrete users
- ❌ Different from MATLAB
- ❌ Requires migration

---

### Option C: Unify Control Input Handling 🎯 RECOMMENDED

**Make both accept flexible control**:

```python
# Continuous.simulate() - ADD support for open-loop
def simulate(self, x0, control=None, ...):
    # control can be:
    # - None → zero
    # - array(nu,) → constant
    # - callable(t) → time-varying open-loop
    # - callable(x, t) → state feedback
    pass

# Discrete.simulate() - ALREADY supports flexible
def simulate(self, x0, u_sequence=None, ...):
    # Already supports all these!
    pass
```

**OR** standardize on names:

```python
# Both use 'control' parameter (not 'controller' or 'u_sequence')
cont_sys.simulate(x0, control=my_controller, ...)
disc_sys.simulate(x0, control=my_sequence, ...)
```

**Pros**:
- More flexible continuous systems
- Consistent naming
- Easier to remember

**Cons**:
- Parameter name change is breaking
- May need type hints for clarity

---

### Option D: Add Conversion Utilities ⚡

Regardless of standardization, provide helpers:

```python
# cdesym.utils.results
def to_time_major(result: DiscreteSimulationResult) -> DiscreteSimulationResult:
    """Convert discrete result to time-major convention."""
    return {
        'time_steps': result['time_steps'],
        'states': result['states'].T,  # (nx, T) -> (T, nx)
        'controls': result['controls'].T if result['controls'] is not None else None,
        'dt': result['dt'],
        'metadata': result['metadata']
    }

def to_state_major(result: SimulationResult) -> SimulationResult:
    """Convert continuous result to state-major convention."""
    return {
        'time': result['time'],
        'states': result['states'].T,  # (T, nx) -> (nx, T)
        'controls': result['controls'].T if result['controls'] is not None else None,
        'metadata': result['metadata']
    }
```

---

## Proposed Action Plan

### Phase 1: Document (Immediate) 📝

1. Add clear warnings in docstrings about shape conventions
2. Create comparison guide in documentation
3. Add examples showing both conventions

### Phase 2: Utilities (Short-term) 🛠️

1. Add `to_time_major()` and `to_state_major()` converters
2. Add `is_time_major()` detector
3. Update plotting to handle both (or convert internally)

### Phase 3: Standardization (Medium-term) 🎯

1. **Decision point**: Time-major for all?
2. Add deprecation warnings to discrete state-major
3. Support both for 2-3 releases
4. Eventually remove deprecated convention

### Phase 4: Unification (Long-term) 🔮

1. Consider unified `ControlInput` type for both
2. Standardize parameter names (`control` vs `controller` vs `u_sequence`)
3. Align time specification (continuous could support `n_steps`?)

---

## Immediate Recommendations

### For Your Current Codebase

**Keep current behavior** but:

1. ✅ **Document clearly** - add big callout boxes in docs
2. ✅ **Provide converters** - make it easy to switch
3. ✅ **Fix plotting** - make it work with both formats
4. ⚠️ **Consider time-major as default** - for new development

### For Tutorials

**Be explicit about conventions**:

```python
# Continuous tutorial
result = system.simulate(x0, controller, t_span=(0, 10), dt=0.01)
# Note: states shape is (T, nx) - rows are time points
plt.plot(result['time'], result['states'][:, 0])  # First state

# Discrete tutorial  
result = system.simulate(x0, u_sequence, n_steps=100)
# Note: states shape is (nx, T) - columns are time points
plt.plot(result['time_steps'], result['states'][0, :])  # First state
```

---

## Decision Matrix

| Issue | Keep As-Is | Standardize | Priority |
|-------|-----------|-------------|----------|
| Shape convention | Easy | Breaking change | HIGH |
| Control input | Easy | Minor breaking | MEDIUM |
| Time specification | Easy | Breaking change | LOW |
| Return keys | Easy | Non-breaking | LOW |

**Recommendation**: 
1. **High priority**: Standardize shape to time-major (breaking but worth it)
2. **Medium priority**: Unify control input naming
3. **Low priority**: Keep time specification as-is (domain-appropriate)
4. **Low priority**: Standardize return key names (minor improvement)
