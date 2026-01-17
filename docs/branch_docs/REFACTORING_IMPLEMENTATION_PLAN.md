# ControlDESymulation Refactoring Implementation Plan

## Phase 0: Preparation (Before Starting)

### 0.1 Ensure Clean Working Directory
```bash
# Check status
git status

# Commit or stash any uncommitted changes
git add -A
git commit -m "Pre-refactoring checkpoint"

# Verify you're on main/master
git branch
```

### 0.2 Create Refactoring Branch
```bash
# Create and switch to new branch
git switch -c refactor/time-major-rollout

# Push branch to remote (optional, for backup)
git push -u origin refactor/time-major-rollout
```

---

## Phase 1: Type Definitions

### 1.1 Update SimulationResult TypedDict
**File**: `src/cdesym/types/results.py` (or wherever TypedDicts are defined)

```python
# Update to use consistent keys
SimulationResult = TypedDict('SimulationResult', {
    't': NDArray,          # Changed from 'time'
    'x': NDArray,          # Changed from 'states', shape (T, nx)
    'u': Optional[NDArray],# Changed from 'controls', shape (T, nu)
    'success': bool,
    'message': str,
    'method': str,
    'dt': float,
    'metadata': dict
})
```

### 1.2 Update DiscreteSimulationResult TypedDict
```python
DiscreteSimulationResult = TypedDict('DiscreteSimulationResult', {
    't': NDArray,          # Changed from 'time_steps'
    'x': NDArray,          # Changed from 'states', NOW (T, nx) not (nx, T)
    'u': Optional[NDArray],# Changed from 'controls', NOW (T, nu) not (nu, T)
    'success': bool,
    'message': str,
    'method': str,
    'dt': float,
    'metadata': dict
})
```

### 1.3 Commit Type Changes
```bash
git add src/cdesym/types/
git commit -m "refactor: Standardize result TypedDicts to time-major with consistent keys"
```

---

## Phase 2: Continuous System Base

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
) -> SimulationResult:
    """
    Closed-loop simulation with state-feedback controller.
    
    Parameters
    ----------
    controller : Optional[Callable[[StateVector, float], ControlVector]]
        Feedback controller u = controller(x, t)
        If None, uses zero control
    ...
    
    Returns
    -------
    SimulationResult
        Dictionary with keys: 't', 'x', 'u', 'success', 'message', 'method', 'dt', 'metadata'
    """
    # Convert controller(x, t) to u(t, x) for integrate()
    if controller is not None:
        u_func = lambda t, x: controller(x, t)
    else:
        u_func = None
    
    # Call integrate() with regular grid
    result = self.integrate(
        x0=x0,
        u=u_func,
        t_span=t_span,
        method=method,
        dt=dt,
        **kwargs
    )
    
    # Return with standardized keys (already has 't', 'x')
    return result
```

### 2.2 Refactor simulate() Method
**Same file**: Modify existing `simulate()` to remove `controller` parameter

```python
def simulate(
    self,
    x0: StateVector,
    u: Optional[Union[ControlVector, TimeVaryingControl]] = None,  # CHANGED
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
    ...
    
    Returns
    -------
    SimulationResult
        Dictionary with keys: 't', 'x', 'u', 'success', 'message', 'method', 'dt', 'metadata'
        
    Notes
    -----
    For state-feedback control, use rollout() instead.
    """
    # Call integrate() with regular grid
    result = self.integrate(
        x0=x0,
        u=u,
        t_span=t_span,
        method=method,
        dt=dt,
        **kwargs
    )
    
    # Return with standardized keys (already has 't', 'x')
    return result
```

### 2.3 Commit Continuous Base Changes
```bash
git add src/cdesym/systems/base/core/continuous_system_base.py
git commit -m "refactor: Add rollout() and refactor simulate() in ContinuousSystemBase"
```

---

## Phase 3: Continuous Stochastic System

### 3.1 Override simulate() and rollout()
**File**: `src/cdesym/systems/continuous_stochastic_system.py`

Add overrides:
```python
def simulate(
    self,
    x0: StateVector,
    u: Optional[Union[ControlVector, TimeVaryingControl]] = None,
    t_span: TimeSpan = (0.0, 10.0),
    dt: ScalarLike = 0.01,
    method: SDEIntegrationMethod = "euler_maruyama",
    n_paths: int = 1,           # ADDED
    seed: Optional[int] = None, # ADDED
    **kwargs
) -> SimulationResult:
    """Open-loop SDE simulation with optional Monte Carlo."""
    return self.integrate(
        x0=x0, u=u, t_span=t_span, method=method, dt=dt,
        n_paths=n_paths, seed=seed, **kwargs
    )

def rollout(
    self,
    x0: StateVector,
    controller: Optional[FeedbackController] = None,
    t_span: TimeSpan = (0.0, 10.0),
    dt: ScalarLike = 0.01,
    method: SDEIntegrationMethod = "euler_maruyama",
    n_paths: int = 1,           # ADDED
    seed: Optional[int] = None, # ADDED
    **kwargs
) -> SimulationResult:
    """Closed-loop SDE simulation with optional Monte Carlo."""
    u_func = lambda t, x: controller(x, t) if controller else None
    return self.integrate(
        x0=x0, u=u_func, t_span=t_span, method=method, dt=dt,
        n_paths=n_paths, seed=seed, **kwargs
    )
```

### 3.2 Commit Stochastic Changes
```bash
git add src/cdesym/systems/continuous_stochastic_system.py
git commit -m "refactor: Override simulate() and rollout() in ContinuousStochasticSystem with Monte Carlo support"
```

---

## Phase 4: Discrete System Base

### 4.1 Update simulate() Return Format
**File**: `src/cdesym/systems/base/core/discrete_system_base.py`

Update abstract method signature and docstring:
```python
@abstractmethod
def simulate(
    self,
    x0: StateVector,
    u_sequence: DiscreteControlInput = None,
    n_steps: int = 100,
    **kwargs
) -> DiscreteSimulationResult:
    """
    Simulate system for multiple discrete time steps.
    
    Returns
    -------
    DiscreteSimulationResult
        TypedDict containing:
        - t: Time step array (n_steps+1,)
        - x: State trajectory (n_steps+1, nx) - TIME-MAJOR
        - u: Control sequence (n_steps, nu) - TIME-MAJOR
        - dt: Sampling period
        - success: bool
        - message: str
        - method: str
        - metadata: dict
    """
```

### 4.2 Commit Discrete Base Changes
```bash
git add src/cdesym/systems/base/core/discrete_system_base.py
git commit -m "refactor: Update DiscreteSystemBase.simulate() to time-major convention"
```

---

## Phase 5: Discrete Symbolic System

### 5.1 Update simulate() Implementation
**File**: `src/cdesym/systems/discrete_symbolic_system.py`

Change implementation to return time-major:
```python
def simulate(
    self,
    x0: StateVector,
    u_sequence: DiscreteControlInput = None,
    n_steps: int = 100,
    **kwargs
) -> DiscreteSimulationResult:
    """Simulate discrete system for multiple steps."""
    
    # Initialize storage - TIME-MAJOR
    states = np.zeros((n_steps + 1, self.nx))  # CHANGED from (self.nx, n_steps + 1)
    states[0, :] = x0
    
    # Prepare control function
    u_func = self._prepare_control_sequence(u_sequence, n_steps)
    
    # Simulate forward
    x = x0
    controls = []
    for k in range(n_steps):
        u = u_func(x, k)
        controls.append(u)
        x = self.step(x, u, k)
        states[k + 1, :] = x
    
    # Format controls - TIME-MAJOR
    if controls and controls[0] is not None:
        controls_array = np.array(controls)  # CHANGED from np.array(controls).T
    else:
        controls_array = None
    
    return {
        "t": np.arange(n_steps + 1),  # CHANGED from 'time_steps'
        "x": states,                   # CHANGED from 'states', already time-major
        "u": controls_array,           # CHANGED from 'controls'
        "dt": self._dt,
        "success": True,
        "message": "Discrete simulation completed",
        "method": "discrete_step",
        "metadata": {"method": "discrete_step"}
    }
```

### 5.2 Commit Discrete Symbolic Changes
```bash
git add src/cdesym/systems/discrete_symbolic_system.py
git commit -m "refactor: Update DiscreteSymbolicSystem.simulate() to time-major with standardized keys"
```

---

## Phase 6: Discrete Stochastic System

### 6.1 Update simulate_stochastic() Implementation
**File**: `src/cdesym/systems/discrete_stochastic_system.py`

Update to return time-major:
```python
def simulate_stochastic(
    self,
    x0: StateVector,
    u_sequence: Optional[Union[ControlVector, DiscreteControlInput]] = None,
    n_steps: int = 100,
    n_paths: int = 1,
    seed: Optional[int] = None,
    **kwargs
) -> DiscreteSimulationResult:
    """Simulate stochastic discrete system with optional Monte Carlo."""
    
    if seed is not None:
        np.random.seed(seed)
    
    u_func = self._prepare_control_sequence(u_sequence, n_steps)
    
    if n_paths == 1:
        # Single trajectory - TIME-MAJOR
        states = np.zeros((n_steps + 1, self.nx))  # CHANGED
        states[0, :] = x0
        controls = []
        
        x = x0
        for k in range(n_steps):
            u = u_func(x, k)
            controls.append(u)
            x_next = self.step(x, u, k)
            states[k + 1, :] = x_next
            x = x_next
        
        controls_array = np.array(controls) if controls else None  # CHANGED
        
        return {
            "t": np.arange(n_steps + 1),  # CHANGED
            "x": states,                   # CHANGED, time-major
            "u": controls_array,           # CHANGED
            "dt": self._dt,
            "success": True,
            "message": "Stochastic simulation completed",
            "method": "stochastic_step",
            "metadata": {
                "n_paths": 1,
                "noise_type": self.get_noise_type(),
                "seed": seed
            }
        }
    
    else:
        # Monte Carlo - TIME-MAJOR
        all_states = np.zeros((n_paths, n_steps + 1, self.nx))  # CHANGED
        
        for path_idx in range(n_paths):
            x = x0
            all_states[path_idx, 0, :] = x0
            
            for k in range(n_steps):
                u = u_func(x, k)
                x_next = self.step(x, u, k)
                all_states[path_idx, k + 1, :] = x_next
                x = x_next
        
        return {
            "t": np.arange(n_steps + 1),  # CHANGED
            "x": all_states,               # CHANGED, (n_paths, T, nx)
            "u": None,                     # Could store but skipping for now
            "dt": self._dt,
            "success": True,
            "message": f"Monte Carlo simulation with {n_paths} paths",
            "method": "stochastic_step",
            "metadata": {
                "n_paths": n_paths,
                "noise_type": self.get_noise_type(),
                "seed": seed
            }
        }
```

### 6.2 Commit Discrete Stochastic Changes
```bash
git add src/cdesym/systems/discrete_stochastic_system.py
git commit -m "refactor: Update DiscreteStochasticSystem.simulate_stochastic() to time-major"
```

---

## Phase 7: Discretized System

### 7.1 Update Both simulate() Methods
**File**: `src/cdesym/systems/discretized_system.py`

Update `_simulate_step_by_step()`:
```python
def _simulate_step_by_step(self, x0, u_sequence, n_steps):
    # TIME-MAJOR
    states = np.zeros((n_steps + 1, self.nx))  # CHANGED
    states[0, :] = x0
    controls = []
    u_func = self._prepare_control_sequence(u_sequence, n_steps)
    
    x = x0
    for k in range(n_steps):
        u = u_func(x, k)
        controls.append(u)
        x = self.step(x, u, k)
        states[k + 1, :] = x
    
    return {
        "t": np.arange(n_steps + 1),  # CHANGED
        "x": states,                   # CHANGED
        "u": np.array(controls) if controls and controls[0] is not None else None,
        "dt": self.dt,
        "success": True,
        "message": "Step-by-step simulation completed",
        "method": self._method,
        "metadata": {"method": self._method, "mode": self._mode.value}
    }
```

Update `_simulate_batch()`:
```python
def _simulate_batch(self, x0, u_sequence, n_steps):
    # ... existing code ...
    
    # Convert continuous result to discrete grid
    trajectory = result["x"] if "x" in result else result["y"].T
    t_regular = np.arange(0, n_steps + 1) * self.dt
    states_regular = self._interpolate_trajectory(result["t"], trajectory, t_regular)
    
    # Ensure time-major
    if states_regular.shape[0] != n_steps + 1:
        states_regular = states_regular.T
    
    return {
        "t": t_regular,      # CHANGED
        "x": states_regular, # CHANGED, (T, nx)
        "u": None,           # Could reconstruct but skipping
        "dt": self.dt,
        "success": True,
        "message": "Batch simulation completed",
        "method": self._method,
        "metadata": {"method": self._method, "mode": self._mode.value}
    }
```

Update `simulate_stochastic()`:
```python
def simulate_stochastic(
    self,
    x0: StateVector,
    u_sequence: DiscreteControlInput = None,
    n_steps: int = 100,
    n_trajectories: int = 100,
    **kwargs
) -> dict:
    """Simulate stochastic system with multiple Monte Carlo trajectories."""
    
    # ... validation code ...
    
    all_trajectories = []
    for traj_idx in range(n_trajectories):
        result = self.simulate(x0, u_sequence, n_steps, **kwargs)
        all_trajectories.append(result["x"])  # CHANGED from result["states"]
    
    # Stack: (n_trajectories, T, nx) - already time-major
    all_trajectories = np.array(all_trajectories)
    
    # Compute statistics
    mean_traj = np.mean(all_trajectories, axis=0)  # (T, nx)
    std_traj = np.std(all_trajectories, axis=0)    # (T, nx)
    
    return {
        "t": result["t"],           # CHANGED
        "x": all_trajectories,      # CHANGED
        "u": result.get("u"),       # CHANGED
        "mean": mean_traj,          # Added for convenience
        "std": std_traj,            # Added for convenience
        "dt": self.dt,
        "success": True,
        "message": f"Monte Carlo with {n_trajectories} paths",
        "method": self._method,
        "metadata": {
            "n_trajectories": n_trajectories,
            "method": self._method,
            "mode": self._mode.value,
            "is_stochastic": True,
        }
    }
```

### 7.2 Commit Discretized System Changes
```bash
git add src/cdesym/systems/discretized_system.py
git commit -m "refactor: Update DiscretizedSystem to time-major convention"
```

---

## Phase 8: Update Plotting Infrastructure

### 8.1 Verify Plotting Compatibility
**Files**: 
- `src/cdesym/ui/plotting/trajectory_plotter.py`
- `src/cdesym/ui/plotting/phase_plotter.py`

Check if plotting assumes specific shapes. Update if needed:
```python
# Should handle both (T, nx) and (nx, T) gracefully
def plot_trajectory(self, t, x, ...):
    # Ensure time-major
    if x.ndim == 2 and x.shape[0] != len(t):
        x = x.T  # Convert state-major to time-major
    
    # Now x is (T, nx)
    # ... plotting code ...
```

### 8.2 Update plot() Convenience Method
**File**: `src/cdesym/systems/base/core/continuous_system_base.py`

Ensure it works with new result format:
```python
def plot(self, result: Union[IntegrationResult, SimulationResult], ...):
    """Plot integration or simulation result."""
    # Both now use 't' and 'x' keys
    return self.plotter.plot_trajectory(
        result["t"],
        result["x"],
        ...
    )
```

### 8.3 Commit Plotting Changes
```bash
git add src/cdesym/ui/plotting/
git commit -m "refactor: Update plotting to handle time-major convention uniformly"
```

---

## Phase 9: Update Tests

### 9.1 Update Continuous System Tests
**Files**: `tests/systems/test_continuous_*.py`

Update tests that use `simulate()` with controller:
```python
# OLD
result = system.simulate(x0, controller=my_controller, ...)

# NEW
result = system.rollout(x0, controller=my_controller, ...)
```

### 9.2 Update Discrete System Tests
**Files**: `tests/systems/test_discrete_*.py`

Update shape assertions:
```python
# OLD
assert result['states'].shape == (nx, n_steps + 1)
assert result['states'][0, :].shape == (n_steps + 1,)

# NEW
assert result['x'].shape == (n_steps + 1, nx)
assert result['x'][:, 0].shape == (n_steps + 1,)
```

### 9.3 Update Plotting Tests
**Files**: `tests/ui/test_plotting.py`

Verify tests pass with new format.

### 9.4 Run All Tests
```bash
# Run full test suite
pytest tests/ -v

# Fix any failures iteratively
# Commit fixes as you go
git add tests/
git commit -m "test: Update all tests for time-major convention and API changes"
```

---

## Phase 10: Update Examples and Tutorials

### 10.1 Update Example Scripts
**Files**: `examples/**/*.py`

Update all example code:
- Change `controller=` to `rollout()`
- Update indexing for discrete systems
- Update key access (`states` → `x`, `time` → `t`)

```bash
git add examples/
git commit -m "docs: Update examples for refactored API"
```

### 10.2 Update Tutorial Notebooks
**Files**: `tutorials/**/*.qmd` or `*.ipynb`

Same changes as examples.

```bash
git add tutorials/
git commit -m "docs: Update tutorials for refactored API"
```

### 10.3 Update basic_usage.qmd
```python
# Now can use either
result = pendulum.simulate(x0, u=None, t_span=(0, 20), dt=0.05)
# or
result = pendulum.rollout(x0, controller=my_controller, t_span=(0, 20), dt=0.05)

# Both work with plotting
pendulum.plot(result)
```

---

## Phase 11: Update Documentation

### 11.1 Update API Reference
Update docstrings if needed (should be done in phases above).

### 11.2 Update README
**File**: `README.md`

Update quick start examples:
```python
# Open-loop simulation
result = system.simulate(x0, u=my_control_func, t_span=(0, 10), dt=0.01)

# Closed-loop control
result = system.rollout(x0, controller=feedback_controller, t_span=(0, 10), dt=0.01)

# Access results (time-major)
plt.plot(result['t'], result['x'][:, 0])  # First state
```

### 11.3 Add Migration Guide
**File**: `docs/MIGRATION.md` or similar

Document breaking changes and how to migrate.

```bash
git add README.md docs/
git commit -m "docs: Update documentation for API refactoring"
```

---

## Phase 12: Final Verification

### 12.1 Run Complete Test Suite
```bash
# Run all tests
pytest tests/ -v --cov=cdesym

# Check coverage
coverage report
```

### 12.2 Build Documentation
```bash
# Build docs locally
cd docs
make html

# Verify no warnings
```

### 12.3 Test Example Scripts
```bash
# Run all example scripts
for script in examples/**/*.py; do
    echo "Testing $script"
    python $script || exit 1
done
```

### 12.4 Render Tutorials
```bash
# Render Quarto tutorials
cd tutorials
quarto render

# Check for errors
```

---

## Phase 13: Merge to Main

### 13.1 Final Review
```bash
# Review all changes
git log --oneline origin/main..HEAD

# Review diff
git diff origin/main
```

### 13.2 Push Branch
```bash
# Push all commits
git push origin refactor/time-major-rollout
```

### 13.3 Create Pull Request (Optional)
If using GitHub/GitLab:
- Create PR from refactor branch to main
- Review changes
- Request review if working with team

### 13.4 Merge to Main
```bash
# Switch to main
git switch main

# Pull latest (if collaborative)
git pull origin main

# Merge refactor branch
git merge refactor/time-major-rollout

# Or use squash merge for cleaner history
git merge --squash refactor/time-major-rollout
git commit -m "refactor: Standardize API to time-major convention with rollout() method

Major breaking changes:
- All systems now use time-major (T, nx) shape convention
- Added rollout() method for closed-loop simulation
- Refactored simulate() for open-loop only
- Standardized result dict keys: 't', 'x', 'u'
- ContinuousStochasticSystem now supports Monte Carlo in simulate()

See REFACTORING_SUMMARY.md for complete details."

# Push to remote
git push origin main
```

### 13.5 Tag Release
```bash
# Tag as pre-release (before v1.0)
git tag -a v0.9.0 -m "Pre-release with API standardization"
git push origin v0.9.0

# Or tag as v1.0.0 if ready
git tag -a v1.0.0 -m "First stable release with standardized API"
git push origin v1.0.0
```

### 13.6 Delete Refactor Branch (Optional)
```bash
# Delete local branch
git branch -d refactor/time-major-rollout

# Delete remote branch
git push origin --delete refactor/time-major-rollout
```

---

## Rollback Plan (If Needed)

If something goes wrong:

### Option 1: Reset Branch
```bash
# On refactor branch, undo all commits
git reset --hard origin/main

# Start over
```

### Option 2: Revert Merge (If Already Merged)
```bash
# On main, revert the merge commit
git revert -m 1 <merge-commit-hash>

# Or reset to before merge (dangerous if pushed)
git reset --hard HEAD~1
```

### Option 3: Cherry-Pick Good Commits
```bash
# Create new branch from main
git switch -c refactor/time-major-rollout-v2 main

# Cherry-pick commits that worked
git cherry-pick <commit-hash>
```

---

## Estimated Timeline

| Phase | Description | Estimated Time |
|-------|-------------|----------------|
| 0 | Preparation | 15 min |
| 1 | Type definitions | 30 min |
| 2 | Continuous base | 1 hour |
| 3 | Continuous stochastic | 30 min |
| 4 | Discrete base | 30 min |
| 5 | Discrete symbolic | 1 hour |
| 6 | Discrete stochastic | 1 hour |
| 7 | Discretized system | 1.5 hours |
| 8 | Plotting | 1 hour |
| 9 | Tests | 2-3 hours |
| 10 | Examples/tutorials | 2 hours |
| 11 | Documentation | 1 hour |
| 12 | Verification | 1 hour |
| 13 | Merge | 30 min |
| **Total** | | **~14-15 hours** |

---

## Success Criteria

- [ ] All tests pass
- [ ] All examples run without errors
- [ ] All tutorials render correctly
- [ ] Documentation builds without warnings
- [ ] No regressions in functionality
- [ ] Code coverage maintained or improved
- [ ] Git history is clean and logical

---

## Post-Merge Tasks

### Immediate
- [ ] Update CHANGELOG.md
- [ ] Announce breaking changes (if any users)
- [ ] Update PyPI description (when releasing)

### Before v1.0 PyPI Release
- [ ] Final documentation review
- [ ] Create comprehensive migration guide
- [ ] Prepare release notes
- [ ] Version bump to 1.0.0
- [ ] Build and test package distribution
- [ ] Upload to PyPI

---

## Notes

- **Commit often**: Small, logical commits are easier to review and debug
- **Test frequently**: Run tests after each major phase
- **Document as you go**: Update docstrings alongside code changes
- **Keep backups**: Push to remote regularly
- **Be patient**: This is a significant refactoring - take breaks!
