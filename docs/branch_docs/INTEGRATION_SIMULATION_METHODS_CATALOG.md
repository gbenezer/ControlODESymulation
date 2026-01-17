# Integration and Simulation Methods Catalog

Complete overview of all integration and simulation methods in ControlDESymulation's UI framework.

---

## Continuous Systems

### ContinuousSystemBase (Abstract Base)

#### `integrate(x0, u, t_span, method, **kwargs) -> IntegrationResult`

**Purpose**: Low-level ODE solver interface with full diagnostics

**Signature**:
```python
def integrate(
    self,
    x0: StateVector,
    u: ControlInput = None,  # None, array, or callable
    t_span: TimeSpan = (0.0, 10.0),
    method: IntegrationMethod = "RK45",
    **integrator_kwargs
) -> IntegrationResult
```

**Returns**: `IntegrationResult` TypedDict
- `t`: Adaptive time points (irregular spacing)
- `x` or `y`: State trajectory (backend-dependent shape)
- `success`, `message`, `nfev`, `njev`, `nlu`, `nsteps`
- `integration_time`, `solver`, `status`

**Control Input Types**:
- `None` → Zero/autonomous
- `array(nu,)` → Constant control
- `callable(t)` → Time-varying
- `callable(t, x)` or `callable(x, t)` → State feedback (auto-detected)

**When to Use**:
- Need solver diagnostics (nfev, convergence info)
- Want adaptive time stepping
- Need dense output interpolant
- Expert use cases

**Key Features**:
- Adaptive time points (irregular grid)
- Exposes all solver parameters
- Returns raw solver output
- Backend-agnostic (scipy, PyTorch, JAX)

---

#### `simulate(x0, controller, t_span, dt, method, **kwargs) -> SimulationResult`

**Purpose**: High-level interface with regular time grid and cleaner output

**Signature**:
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
- `time`: Regular time grid (uniform spacing = dt)
- `states`: State trajectory (T, nx) - **time-major**
- `controls`: Control trajectory (T, nu) if controller provided
- `metadata`: {'method', 'dt', 'success', 'nfev'}

**Controller Signature**: `(x, t) -> u`
- **Primary argument**: State (x)
- **Secondary argument**: Time (t)
- Matches discrete systems convention: `policy(x, k)`

**When to Use**:
- Regular time grid needed for plotting
- Closed-loop simulation with state feedback
- Don't need solver internals
- **Recommended for most users**

**Key Features**:
- Regular time grid (easy plotting)
- Time-major convention: `(T, nx)`
- Pandas-compatible output
- Wraps `integrate()` internally
- Cleaner output (hides solver details)

**Current Status**: Docstring says "may be deprecated once DiscretizedSystem is developed"

---

### ContinuousSymbolicSystem (extends ContinuousSystemBase)

#### `integrate()` - **Overrides parent**

**What's Different**:
- Same signature as parent
- Uses `IntegratorFactory` to create integrators
- Adds symbolic system-specific features
- Supports `t_eval` and `dense_output` explicitly

**Additional Parameters**:
- `t_eval`: Specific times to evaluate solution
- `dense_output`: Return interpolant object

**Implementation**: Delegates to factory-created integrators

---

### ContinuousStochasticSystem (extends ContinuousSymbolicSystem)

#### `integrate()` - **Overrides for SDEs**

**Signature**:
```python
def integrate(
    self,
    x0: StateVector,
    u: ControlInput = None,
    t_span: TimeSpan = (0.0, 10.0),
    method: SDEIntegrationMethod = "euler_maruyama",  # SDE-specific
    t_eval: Optional[TimePoints] = None,
    n_paths: int = 1,  # ← Monte Carlo support
    seed: Optional[int] = None,  # ← Reproducibility
    **integrator_kwargs
) -> SDEIntegrationResult
```

**Returns**: `SDEIntegrationResult` TypedDict
- `t`: Time points (T,)
- `x`: State trajectory
  - Single path (n_paths=1): `(T, nx)`
  - Multiple paths: `(n_paths, T, nx)`
- `success`, `message`
- `n_paths`: Number of paths
- `noise_type`: Detected noise structure
- `sde_type`: 'ito' or 'stratonovich'
- `nfev`: Drift evaluations
- `diffusion_evals`: Diffusion evaluations
- `integration_time`

**SDE-Specific Parameters**:
- `n_paths`: Monte Carlo ensemble size (1 to 1000+)
- `seed`: Random seed for reproducibility
- `method`: SDE methods only ('euler_maruyama', 'milstein', 'heun', etc.)

**When to Use**:
- **Always for stochastic systems**
- Monte Carlo simulation (`n_paths > 1`)
- Reproducible noise (`seed`)
- SDE integrators required

**Key Features**:
- Built-in Monte Carlo support
- Uses `SDEIntegratorFactory` instead of `IntegratorFactory`
- Returns noise type and SDE type
- Diffusion evaluation count

**No `simulate()` override** - inherits from parent but may not work well with SDEs

---

## Discrete Systems

### DiscreteSystemBase (Abstract Base)

#### `simulate(x0, u_sequence, n_steps, **kwargs) -> DiscreteSimulationResult`

**Purpose**: Forward simulation of discrete-time dynamics

**Signature**:
```python
def simulate(
    self,
    x0: StateVector,
    u_sequence: DiscreteControlInput = None,
    n_steps: int = 100,
    **kwargs
) -> DiscreteSimulationResult
```

**Returns**: `DiscreteSimulationResult` TypedDict
- `states`: State trajectory (nx, n_steps+1) - includes x[0]
- `controls`: Control sequence (nu, n_steps)
- `time_steps`: [0, 1, 2, ..., n_steps]
- `dt`: Sampling period
- `metadata`: {'method', 'success', ...}

**Control Input Types** (`DiscreteControlInput`):
- `None` → Zero control
- `array(nu,)` → Constant control
- `Sequence[array]` → Pre-computed sequence
- `Callable(k)` → Time-indexed u[k] = u_func(k)

**When to Use**:
- Open-loop discrete simulation
- Pre-computed control sequences
- Time-indexed control functions

**Key Features**:
- Integer time steps
- State-major convention: `(nx, n_steps+1)`
- Includes initial state x[0]
- n_steps transitions → n_steps+1 states

---

#### `rollout(x0, policy, n_steps, **kwargs) -> DiscreteSimulationResult`

**Purpose**: Closed-loop simulation with state-feedback policy

**Signature**:
```python
def rollout(
    self,
    x0: StateVector,
    policy: Optional[DiscreteFeedbackPolicy] = None,  # (x, k) -> u
    n_steps: int = 100,
    **kwargs
) -> DiscreteSimulationResult
```

**Policy Signature**: `(x, k) -> u`
- **Primary argument**: State (x)
- **Secondary argument**: Time step (k)

**When to Use**:
- State feedback control
- Reinforcement learning policies
- Adaptive/nonlinear control
- Clearer API than `simulate()` with callable

**Relationship to `simulate()`**:
- Wrapper around `simulate()`
- Provides cleaner interface for state feedback
- Same return type

---

### DiscreteSymbolicSystem (extends DiscreteSystemBase)

#### `simulate()` - **Implements abstract method**

**Implementation**:
- Repeatedly calls `step(x, u, k)` for each time step
- Stores trajectory in arrays
- Returns standard `DiscreteSimulationResult`

**Key Detail**: Returns `states.T` for shape `(nx, n_steps+1)`

---

### DiscreteStochasticSystem (extends DiscreteSymbolicSystem)

#### `simulate_stochastic(x0, u_sequence, n_steps, n_paths, seed, **kwargs)`

**Purpose**: Monte Carlo simulation of discrete stochastic systems

**Signature**:
```python
def simulate_stochastic(
    self,
    x0: StateVector,
    u_sequence: Optional[DiscreteControlInput] = None,
    n_steps: int = 100,
    n_paths: int = 1,  # ← Monte Carlo
    seed: Optional[int] = None,  # ← Reproducibility
    **kwargs
) -> DiscreteSimulationResult
```

**Returns**: `DiscreteSimulationResult` (extended)
- `states`: Trajectory
  - Single path: `(n_steps+1, nx)`
  - Multiple paths: `(n_paths, n_steps+1, nx)`
- `controls`: Control sequence (n_steps, nu)
- `time_steps`: [0, 1, ..., n_steps]
- `dt`: Sampling period
- `metadata`: Includes `n_paths`, `noise_type`, `seed`

**When to Use**:
- **For discrete stochastic systems** (not deterministic)
- Monte Carlo analysis
- Uncertainty propagation
- Reproducible random trajectories

**Note**: Parent `simulate()` still exists for single deterministic trajectory

---

### DiscretizedSystem (extends DiscreteSymbolicSystem)

**Special Case**: Discretizes a continuous system for discrete-time control

#### `simulate()` - **Overrides with two modes**

**Implementation Modes**:

1. **STEP_BY_STEP** (default):
   ```python
   def simulate(x0, u_sequence, n_steps):
       # Repeatedly call step()
       for k in range(n_steps):
           x = step(x, u, k)
   ```

2. **BATCH_INTERPOLATION**:
   ```python
   def simulate(x0, u_sequence, n_steps):
       # Single continuous integration + interpolation
       result = continuous_system.integrate(...)
       return interpolate_to_discrete_grid(result)
   ```

**Which Mode**:
- Set via `DiscretizationMode` enum
- BATCH faster for long simulations
- STEP_BY_STEP more accurate for control

---

#### `simulate_stochastic()`

**Purpose**: Monte Carlo for discretized stochastic systems

**Signature**:
```python
def simulate_stochastic(
    self,
    x0: StateVector,
    u_sequence: DiscreteControlInput = None,
    n_steps: int = 100,
    n_trajectories: int = 100,  # Note: different parameter name
    **kwargs
) -> dict
```

**Returns** (custom dict, not TypedDict):
- `states`: (n_trajectories, n_steps+1, nx)
- `controls`: Control sequence
- `mean_trajectory`: Mean across trajectories
- `std_trajectory`: Std dev across trajectories
- `time_steps`: [0, 1, ..., n_steps]
- `dt`, `success`, `n_trajectories`
- `metadata`: Includes convergence info

**Requirements**:
- Must have stochastic continuous system
- Must use SDE method ('euler_maruyama', etc.)
- Requires SDE integrator backend

---

## Summary Table

| Class | Method | Purpose | Returns | Key Parameters |
|-------|--------|---------|---------|----------------|
| **Continuous** | | | | |
| ContinuousSystemBase | `integrate()` | Low-level ODE solver | `IntegrationResult` | `method`, `t_span` |
| ContinuousSystemBase | `simulate()` | High-level regular grid | `SimulationResult` | `controller`, `dt` |
| ContinuousStochasticSystem | `integrate()` | SDE solver + Monte Carlo | `SDEIntegrationResult` | `n_paths`, `seed` |
| **Discrete** | | | | |
| DiscreteSystemBase | `simulate()` | Forward discrete simulation | `DiscreteSimulationResult` | `u_sequence`, `n_steps` |
| DiscreteSystemBase | `rollout()` | State-feedback simulation | `DiscreteSimulationResult` | `policy`, `n_steps` |
| DiscreteStochasticSystem | `simulate_stochastic()` | Discrete Monte Carlo | `DiscreteSimulationResult` | `n_paths`, `seed` |
| DiscretizedSystem | `simulate()` | Discretized continuous | `DiscreteSimulationResult` | Mode-dependent |
| DiscretizedSystem | `simulate_stochastic()` | Discretized SDE Monte Carlo | Custom dict | `n_trajectories` |

---

## Key Observations

### 1. **Two Parallel Hierarchies**

**Continuous**:
- `integrate()` = low-level (adaptive grid, diagnostics)
- `simulate()` = high-level (regular grid, cleaner)

**Discrete**:
- `simulate()` = open-loop (pre-computed u)
- `rollout()` = closed-loop (state feedback)

### 2. **Stochastic Extensions**

**Continuous**: Same `integrate()` method with `n_paths` parameter

**Discrete**: Separate `simulate_stochastic()` method

### 3. **Control Input Conventions**

**Continuous**:
- `integrate()`: `u` can be `callable(t)` or `callable(t, x)` or `callable(x, t)`
- `simulate()`: `controller(x, t)` - state first

**Discrete**:
- `simulate()`: `u_sequence` - time-indexed or pre-computed
- `rollout()`: `policy(x, k)` - state first

### 4. **Return Format Inconsistencies**

**Problem**: Different return types for similar operations
- `integrate()` → `IntegrationResult` (has `'t'`, `'x'`)
- `simulate()` → `SimulationResult` (has `'time'`, `'states'`)
- Plotting methods expect `IntegrationResult` format

**Impact**: Can't easily plot `simulate()` results

### 5. **Missing Functionality**

1. **No `simulate()` for stochastic continuous systems**
   - Must use `integrate()` directly
   - No regular-grid version for SDEs

2. **Plotting expects `integrate()` results**
   - `plot()` method calls `plotter.plot_trajectory(result['t'], result['x'])`
   - Breaks with `simulate()` which has `'time'`, `'states'`

3. **Inconsistent Monte Carlo parameter names**
   - Continuous SDE: `n_paths`
   - Discrete stochastic: `n_paths`
   - Discretized stochastic: `n_trajectories`

---

## Recommendations for Tutorial

### For Basic Usage Tutorial

**Use `integrate()` because**:
1. ✅ Works directly with plotting infrastructure
2. ✅ Consistent across continuous/stochastic systems
3. ✅ Passing `dt` gives regular grid anyway
4. ✅ Don't need to explain two different result formats

```python
# Basic tutorial - use this
result = pendulum.integrate(
    x0=x0,
    t_span=(0, 20),
    dt=0.05  # Regular grid
)
pendulum.plot(result)  # Works!
```

### When to Introduce `simulate()`

**Save for advanced tutorials**:
- Tutorial on closed-loop control
- State-feedback controller design
- Discrete system tutorial (where `simulate()` is the primary method)

### For Stochastic Tutorial

**Always use `integrate()`**:
```python
# Stochastic tutorial
result = reactor.integrate(
    x0=x0,
    t_span=(0, 50),
    method='euler_maruyama',
    dt=0.05,
    n_paths=500,
    seed=42
)
```

No alternative - `simulate()` doesn't work for SDEs.
