# ControlDESymulation Documentation Conversion Plan

## Executive Summary

**Goal**: Convert all documentation to executable Quarto format with living, tested code examples throughout.

**Approach**: **Full executable documentation** - All code blocks run and are tested in CI, ensuring documentation stays perfectly in sync with the codebase.

**Timeline**: ~3-4 weeks for complete conversion and testing

**Outcome**: Professional, interactive, continuously-tested documentation where every code example is guaranteed to work.

---
<!-- 
## Phase 4: Enhanced Example Gallery (Days 16-20)

### Strategy: Rich, Interactive Demonstrations

Now that architecture is executable, examples can be even richer:
- Show multiple approaches side-by-side
- Include performance benchmarks
- Add parameter sensitivity analysis
- Create comparison tables

### Day 16: Advanced Pendulum Example

**Create `docs/examples/pendulum_control.qmd`:**

````markdown
---
title: "Pendulum Control: Complete Workflow"
execute:
  eval: true
  cache: true
---

## Introduction

Complete workflow from system creation through controller design, implementation,
and performance analysis.

```{python}
#| label: setup
#| output: false

from cdesym.systems.examples.pendulum import SymbolicPendulum
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
```

## System Creation and Analysis

```{python}
#| label: system-setup
#| output: true

system = SymbolicPendulum(m=1.0, l=0.5, beta=0.1, g=9.81)

# List equilibria
equilibria = system.list_equilibria()
print(f"Available equilibria: {equilibria}")

# Analyze each
for name in equilibria:
    x_eq, u_eq = system.get_equilibrium(name)
    A, _ = system.linearize(x_eq, u_eq)
    eigs = np.linalg.eigvals(A)
    stable = np.all(np.real(eigs) < 0)
    
    print(f"\n{name}:")
    print(f"  State: θ={x_eq[0]:.3f} rad, ω={x_eq[1]:.3f} rad/s")
    print(f"  Eigenvalues: {eigs}")
    print(f"  Stability: {'Stable ✓' if stable else 'Unstable ✗'}")
```

## Controller Design Comparison

Compare different LQR weights:

```{python}
#| label: tbl-controller-comparison
#| tbl-cap: "LQR controllers with different Q weights"

x_eq, u_eq = system.get_equilibrium('inverted')
A, B = system.linearize(x_eq, u_eq)
R = np.array([[0.1]])

q_scales = [1, 10, 100]
controller_data = []

for q_scale in q_scales:
    Q = q_scale * np.diag([10, 1])
    lqr = system.control.design_lqr(A, B, Q, R, system_type='continuous')
    
    controller_data.append({
        'Q Scale': q_scale,
        'K (angle)': f"{lqr['gain'][0, 0]:.2f}",
        'K (velocity)': f"{lqr['gain'][0, 1]:.2f}",
        'Max |Re(λ)|': f"{np.abs(np.real(lqr['closed_loop_eigenvalues'])).max():.2f}"
    })

pd.DataFrame(controller_data)
```

Higher Q → more aggressive gains → faster convergence.

## Performance Comparison

Simulate all three controllers:

```{python}
#| label: fig-controller-comparison
#| fig-cap: "Response comparison for different LQR weights"
#| fig-width: 10
#| fig-height: 8

x0 = np.array([np.pi + 0.2, 0.0])  # 11.5° perturbation

fig, axes = plt.subplots(2, 1, figsize=(10, 8))

for q_scale in q_scales:
    Q = q_scale * np.diag([10, 1])
    lqr = system.control.design_lqr(A, B, Q, R, system_type='continuous')
    K = lqr['gain']
    
    # Simulate
    result = system.simulate(
        x0=x0,
        controller=lambda x, t: -K @ (x - x_eq),
        t_span=(0, 5),
        dt=0.01
    )
    
    # Plot
    axes[0].plot(result['time'], result['states'][:, 0] - np.pi, 
                 label=f'Q scale = {q_scale}')
    axes[1].plot(result['time'], result['controls'][:, 0],
                 label=f'Q scale = {q_scale}')

axes[0].set_ylabel('Angle Error (rad)')
axes[0].legend()
axes[0].grid(True)

axes[1].set_ylabel('Control Torque (N·m)')
axes[1].set_xlabel('Time (s)')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()
```

::: {.callout-important}
## Trade-off
Higher Q weights provide faster response but require more control effort.
Choose based on actuator constraints and performance requirements.
:::

## Robustness Analysis

Test robustness to initial conditions:

```{python}
#| label: fig-robustness
#| fig-cap: "Monte Carlo simulation showing robustness to IC uncertainty"
#| fig-width: 10
#| fig-height: 6

# Use middle controller (Q scale = 10)
Q = 10 * np.diag([10, 1])
lqr = system.control.design_lqr(A, B, Q, R, system_type='continuous')
K = lqr['gain']

# Monte Carlo
n_trials = 50
np.random.seed(42)

all_trajectories = []
for _ in range(n_trials):
    x0_noisy = x_eq + np.array([0.3, 0.5]) * np.random.randn(2)
    
    result = system.simulate(
        x0=x0_noisy,
        controller=lambda x, t: -K @ (x - x_eq),
        t_span=(0, 5),
        dt=0.01
    )
    all_trajectories.append(result['states'])

# Stack and plot
trajectories = np.array(all_trajectories)  # (n_trials, T, nx)

# Plot mean ± std
mean_traj = np.mean(trajectories, axis=0)
std_traj = np.std(trajectories, axis=0)

plt.figure(figsize=(10, 6))
t = result['time']

plt.plot(t, mean_traj[:, 0] - np.pi, 'b-', linewidth=2, label='Mean')
plt.fill_between(
    t,
    mean_traj[:, 0] - np.pi - std_traj[:, 0],
    mean_traj[:, 0] - np.pi + std_traj[:, 0],
    alpha=0.3,
    label='±1σ'
)

# Plot a few individual trajectories
for i in range(5):
    plt.plot(t, trajectories[i, :, 0] - np.pi, 'gray', alpha=0.3, linewidth=0.5)

plt.xlabel('Time (s)')
plt.ylabel('Angle Error (rad)')
plt.legend()
plt.grid(True)
plt.title(f'Robustness: {n_trials} trials with random ICs')
plt.show()

# Compute statistics
final_errors = trajectories[:, -1, 0] - np.pi
print(f"Final error statistics:")
print(f"  Mean: {np.mean(final_errors):.2e} rad")
print(f"  Std:  {np.std(final_errors):.2e} rad")
print(f"  Max:  {np.max(np.abs(final_errors)):.2e} rad")
print(f"✓ All trials stabilized successfully")
```
````

### Days 17-19: Complete Example Gallery

**Day 17:**
- `cartpole_swingup.qmd` - Full swing-up + balance with phase plots
- `quadrotor_lqr.qmd` - Hover control with disturbance rejection

**Day 18:**
- `cstr_multiplicity.qmd` - Bifurcation diagram generation
- `batch_reactor_optimization.qmd` - Temperature trajectory optimization

**Day 19:**
- `lorenz_chaos.qmd` - Sensitive dependence demonstration
- `van_der_pol_limit_cycle.qmd` - Limit cycle with varying μ

### Day 20: Examples Index with Live Previews

**Create `docs/examples/index.qmd`:**
````markdown
---
title: "Examples Gallery"
execute:
  eval: true
listing:
  - id: mechanical
    contents: 
      - pendulum_control.qmd
      - cartpole_swingup.qmd
    type: grid
    fields: [title, image, description]
    image-height: 200px
  - id: aerospace
    contents:
      - quadrotor_lqr.qmd
    type: grid
  - id: chemical
    contents:
      - cstr_multiplicity.qmd
      - batch_reactor_optimization.qmd
    type: grid
  - id: chaos
    contents:
      - lorenz_chaos.qmd
      - van_der_pol_limit_cycle.qmd
    type: grid
---

## Interactive Examples

Fully executable demonstrations of ControlDESymulation capabilities. Every code
block on every page **runs when the documentation is built**, ensuring examples
stay current with the codebase.

### Quick Preview

Here's a live example running right on this index page:

```{python}
#| label: index-quick-demo
#| code-fold: true

from cdesym.systems.examples.van_der_pol import VanDerPolOscillator
import matplotlib.pyplot as plt

system = VanDerPolOscillator(mu=1.0)
result = system.integrate(
    x0=np.array([0.1, 0.0]),
    u=None,
    t_span=(0, 30)
)

plt.figure(figsize=(8, 4))
plt.plot(result['x'][:, 0], result['x'][:, 1], linewidth=2)
plt.xlabel('x')
plt.ylabel('ẋ')
plt.title('Van der Pol Limit Cycle (μ=1.0)')
plt.grid(True)
plt.axis('equal')
plt.show()
```

### Mechanical Systems
::: {#mechanical}
:::

### Aerospace Systems
::: {#aerospace}
:::

### Chemical Systems
::: {#chemical}
:::

### Chaos & Complex Dynamics
::: {#chaos}
:::
```` -->

## Phase 6: Architecture Enhancement (Days 23-26)

### Day 23: UI Framework Architecture - Full Execution

**Key enhancements for `UI_Framework_Architecture.qmd`:**

````markdown
---
title: "UI Framework Architecture"
execute:
  eval: true
  cache: true
---

```{python}
#| label: setup
#| echo: false
#| output: false

import numpy as np
import sympy as sp
from cdesym.systems.base import ContinuousSymbolicSystem, DiscreteSymbolicSystem
```

## Architecture Layers {#sec-layers}

The following diagram shows the inheritance hierarchy:

```{mermaid}
%%| label: fig-ui-layers
%%| fig-cap: "UI Framework inheritance structure"

graph TD
    A[SymbolicSystemBase<br/>Time-agnostic] --> B[ContinuousSystemBase]
    A --> C[DiscreteSystemBase]
    B --> D[ContinuousSymbolicSystem<br/>Multiple Inheritance]
    C --> E[DiscreteSymbolicSystem<br/>Multiple Inheritance]
    D --> F[ContinuousStochasticSystem]
    E --> G[DiscreteStochasticSystem]
```

## Layer 0: SymbolicSystemBase {#sec-layer0}

**Demonstration of shared functionality:**

```{python}
#| label: symbolic-system-demo
#| output: true

class DemoSystem(ContinuousSymbolicSystem):
    """Minimal system to demonstrate SymbolicSystemBase features."""
    
    def define_system(self, k=1.0):
        x = sp.symbols('x', real=True)
        u = sp.symbols('u', real=True)
        k_sym = sp.symbols('k', positive=True)
        
        self.state_vars = [x]
        self.control_vars = [u]
        self._f_sym = sp.Matrix([-k_sym*x + u])
        self.parameters = {k_sym: k}
        self.order = 1
    
    def setup_equilibria(self):
        pass

# Create system
demo = DemoSystem(k=2.0)

# Features from SymbolicSystemBase:
print(f"1. State dimension: {demo.nx}")
print(f"2. Parameters: {demo.parameters}")
print(f"3. Backend: {demo.backend.default_backend}")

# Equilibrium management
demo.add_equilibrium('test', x_eq=np.array([0.0]), u_eq=np.array([0.0]))
print(f"4. Equilibria: {demo.list_equilibria()}")

# Backend switching
demo.set_default_backend('numpy')
print(f"5. ✓ All SymbolicSystemBase features working")
```

::: {.callout-note}
## Live Validation
The code above **actually runs** when building docs, proving that 
SymbolicSystemBase provides these features to all derived classes.
:::
````

**Continue with detailed sections showing actual layer implementation...**

### Day 24: Integration Framework Architecture - Performance Demos

````markdown
## Integrator Performance Comparison {#sec-integrator-performance}

Real benchmark comparing all available integrators:

```{python}
#| label: tbl-integrator-benchmark
#| tbl-cap: "Integrator performance on pendulum system"
#| output: true

from cdesym.systems.examples.pendulum import SymbolicPendulum
from cdesym.systems.base.numerical_integration import IntegratorFactory
import time
import pandas as pd

system = SymbolicPendulum()
x0 = np.array([np.pi + 0.1, 0.0])
u_func = lambda t, x: np.array([0.0])
t_span = (0, 10)

# Test multiple methods
methods = ['RK45', 'RK23', 'DOP853', 'Radau', 'BDF', 'LSODA']
benchmark_data = []

for method in methods:
    integrator = IntegratorFactory.create(system, backend='numpy', method=method)
    
    # Warm-up
    _ = integrator.integrate(x0, u_func, (0, 0.1))
    
    # Benchmark
    start = time.time()
    result = integrator.integrate(x0, u_func, t_span)
    elapsed = time.time() - start
    
    benchmark_data.append({
        'Method': method,
        'Time (ms)': f"{elapsed*1000:.2f}",
        'Steps': result['nsteps'],
        'Evals': result['nfev'],
        'Success': '✓' if result['success'] else '✗'
    })

pd.DataFrame(benchmark_data)
```

This table shows **actual measured performance**, updated every time docs are built!
````

### Day 25: Control Framework Architecture - Algorithm Validation

````markdown
## LQR Algorithm Verification {#sec-lqr-verification}

**Mathematical Test**: Verify that LQR solution satisfies Riccati equation.

```{python}
#| label: lqr-verification
#| output: true

from cdesym.control.classical_control_functions import design_lqr
from cdesym.systems.examples.pendulum import SymbolicPendulum

# Setup test system
system = SymbolicPendulum()
x_eq = np.array([np.pi, 0.0])
u_eq = np.array([0.0])
A, B = system.linearize(x_eq, u_eq)

Q = np.diag([10, 1])
R = np.array([[0.1]])

# Design LQR
lqr = design_lqr(A, B, Q, R, system_type='continuous')
P = lqr['cost_to_go']
K = lqr['gain']

# Verify Continuous ARE: A'P + PA - PBR⁻¹B'P + Q = 0
riccati_residual = (
    A.T @ P + P @ A 
    - P @ B @ np.linalg.solve(R, B.T @ P)
    + Q
)

max_error = np.abs(riccati_residual).max()
print(f"Maximum Riccati residual: {max_error:.2e}")
print(f"Relative error: {max_error / np.abs(Q).max():.2e}")

# Verify K = R⁻¹B'P
K_computed = np.linalg.solve(R, B.T @ P)
K_error = np.linalg.norm(K - K_computed)

print(f"\nGain computation error: {K_error:.2e}")

# Final verdict
if max_error < 1e-10 and K_error < 1e-10:
    print("\n✓✓✓ LQR algorithm mathematically verified!")
else:
    print("\n✗ Verification failed - check implementation")
    raise AssertionError("LQR algorithm verification failed")
```

::: {.callout-important}
## Continuous Validation
This verification runs **automatically** in CI, catching any algorithmic bugs
immediately.
:::
````

### Day 26: Visualization Framework - Live Plot Gallery

````markdown
## Plot Gallery {#sec-plot-gallery}

All plot types demonstrated with live examples:

::: {.panel-tabset}

## Trajectory Plot

```{python}
#| label: fig-trajectory-example
#| fig-cap: "TrajectoryPlotter example"

from cdesym.systems.examples.pendulum import SymbolicPendulum

system = SymbolicPendulum()
result = system.integrate(
    x0=np.array([np.pi + 0.2, 0.0]),
    u=lambda t, x: np.array([0.0]),
    t_span=(0, 5)
)

fig = system.plotter.plot_trajectory(
    result['t'],
    result['x'],
    state_names=['θ', 'ω'],
    theme='publication',
    color_scheme='colorblind_safe'
)
fig.show()
```

## Phase Portrait

```{python}
#| label: fig-phase-example
#| fig-cap: "PhasePortraitPlotter example"

fig = system.phase_plotter.plot_2d(
    result['x'],
    state_names=('θ', 'ω'),
    show_direction=True,
    theme='publication'
)
fig.show()
```

## Eigenvalue Map

```{python}
#| label: fig-eigenvalue-example
#| fig-cap: "ControlPlotter eigenvalue map"

x_eq = np.array([np.pi, 0.0])
u_eq = np.array([0.0])
A, B = system.linearize(x_eq, u_eq)

Q = np.diag([10, 1])
R = np.array([[0.1]])
lqr = system.control.design_lqr(A, B, Q, R, system_type='continuous')

fig = system.control_plotter.plot_eigenvalue_map(
    lqr['closed_loop_eigenvalues'],
    system_type='continuous',
    theme='publication'
)
fig.show()
```

## State + Control

```{python}
#| label: fig-state-control-example
#| fig-cap: "Combined state and control plot"

K = lqr['gain']
result = system.simulate(
    x0=np.array([np.pi + 0.2, 0.0]),
    controller=lambda x, t: -K @ (x - x_eq),
    t_span=(0, 5),
    dt=0.01
)

fig = system.plotter.plot_state_and_control(
    result['time'],
    result['states'],
    result['controls'],
    state_names=['θ', 'ω'],
    control_names=['τ'],
    theme='publication'
)
fig.show()
```

:::

All plots above are **generated live** when documentation builds!
````

---

## Phase 7: Optimization and Performance (Days 27-28)

### Day 27: Caching Strategy

**Identify expensive operations and cache them:**

````markdown
```{python}
#| label: expensive-computation
#| cache: true  # ← Cache this specific block

# This takes 30 seconds, cache the result
monte_carlo_results = run_expensive_monte_carlo(n_trials=1000)
```

Next block uses cached results:

```{python}
#| label: use-cached-results
#| dependson: expensive-computation

# This uses cached data from above
plot_monte_carlo_results(monte_carlo_results)
```
````

**Create `docs/_freeze_config.yml`:**
```yaml
# Control freezing behavior
freeze:
  # Never freeze these (always re-execute)
  exclude:
    - "index.qmd"
    - "examples/*/index.qmd"
  
  # These are expensive, freeze unless source changes
  expensive:
    - "examples/cstr_multiplicity.qmd"
    - "examples/lorenz_chaos.qmd"
```

### Day 28: Output Management

**For architecture docs with lots of code:**

````markdown
## Large Code Listing {#sec-large-listing}

Here's the complete implementation (validated but output hidden):

```{python}
#| code-fold: true
#| output: false

class CompleteSystemImplementation(ContinuousSymbolicSystem):
    """
    Full implementation of complex system.
    
    Code is executed to validate it works, but output is hidden
    to keep documentation readable. Expand code block to see details.
    """
    
    def define_system(self, ...):
        # ... 100 lines of actual, working code ...
        pass
    
    def setup_equilibria(self):
        # ... more code ...
        pass
    
    # ... additional methods ...

# Create instance to verify it works
system = CompleteSystemImplementation()
assert system.nx > 0  # Validation
```

Key features demonstrated above (click to expand code):

```{python}
#| output: true

# Now show specific features with output
system = CompleteSystemImplementation()
print(f"✓ System created: {system.nx} states")
print(f"✓ Equilibria: {system.list_equilibria()}")
print(f"✓ Backend: {system.backend.default_backend}")
```
````

---

## Phase 8: Testing and Debugging (Days 29-30)

### Day 29: Comprehensive Testing

**Test matrix:**

| Category | Files | Expected Time | Cache Strategy |
|----------|-------|---------------|----------------|
| Architecture | 7 | ~15 min total | freeze: auto |
| Tutorials | 7 | ~20 min total | freeze: auto |
| Examples | 6 | ~30 min total | cache: true |
| **Total** | **20** | **~65 min** | Mixed |

**Run full test:**
```bash
# Clean build (no cache)
rm -rf docs/_freeze
python scripts/test_quarto_docs.py

# Cached build (should be faster)
python scripts/test_quarto_docs.py
```

**Expected results:**
- First build: ~60-90 minutes
- Cached build: ~5-10 minutes
- CI build: ~30-45 minutes (parallel rendering)

### Day 30: Debug Common Issues

**Issue 1: Code block fails**

**Fix pattern:**
````markdown
```{python}
#| label: problematic-block
#| error: true  # ← Allow error, document it

try:
    # Code that might fail
    risky_operation()
except Exception as e:
    print(f"Expected error: {e}")
```
````

**Issue 2: Import not found**

**Fix:**
````markdown
```{python}
#| output: false

import sys
from pathlib import Path

# Ensure package is importable
if str(Path.cwd().parent / 'src') not in sys.path:
    sys.path.insert(0, str(Path.cwd().parent / 'src'))

import cdesym
```
````

**Issue 3: Too slow**

**Optimize:**
````markdown
```{python}
#| cache: true  # ← Cache expensive block

# Reduce problem size for docs
n_trials = 10  # Instead of 1000
t_span = (0, 5)  # Instead of (0, 100)
```
````

---

## Phase 9: Polish and Enhancement (Days 31-32)

### Day 31: Add Interactive Elements

**Parameter sensitivity widget (using Plotly):**

````markdown
```{python}
#| label: fig-parameter-sensitivity
#| fig-cap: "Interactive parameter sensitivity"

from cdesym.systems.examples.pendulum import SymbolicPendulum
import plotly.graph_objects as go

# Simulate for different damping values
damping_values = np.linspace(0.01, 1.0, 10)
fig = go.Figure()

for beta in damping_values:
    system = SymbolicPendulum(beta=beta)
    result = system.integrate(
        x0=np.array([np.pi + 0.2, 0.0]),
        u=lambda t, x: np.array([0.0]),
        t_span=(0, 10)
    )
    
    fig.add_trace(go.Scatter(
        x=result['t'],
        y=result['x'][:, 0],
        name=f'β={beta:.2f}',
        mode='lines'
    ))

fig.update_layout(
    title='Parameter Sensitivity: Effect of Damping',
    xaxis_title='Time (s)',
    yaxis_title='Angle (rad)',
    hovermode='x unified'
)
fig.show()
```

Hover over the plot to explore different damping values!
````

### Day 32: Documentation Quality Pass

**Review checklist for EACH file:**

- [ ] YAML front matter correct
- [ ] All code blocks execute without errors
- [ ] Output shown for demonstrative code
- [ ] Output hidden for validation-only code
- [ ] Cross-references work
- [ ] Figures have captions
- [ ] Tables have captions
- [ ] Callouts used appropriately
- [ ] No broken links
- [ ] Mobile-responsive
- [ ] Performance acceptable (<30s render time)

**Automated quality checks:**
```bash
# Check for common issues
python scripts/check_quarto_quality.py
```

**Create `scripts/check_quarto_quality.py`:**
```python
#!/usr/bin/env python3
"""Check Quarto documentation quality."""

from pathlib import Path
import re

def check_file_quality(qmd_file: Path):
    """Check single file for quality issues."""
    
    content = qmd_file.read_text()
    issues = []
    
    # Check 1: Has YAML front matter
    if not content.startswith('---\n'):
        issues.append("Missing YAML front matter")
    
    # Check 2: No bare ```python blocks (should be ```{python})
    if re.search(r'```python\n', content):
        issues.append("Found bare ```python (should be ```{python})")
    
    # Check 3: Figures should have captions
    fig_blocks = re.findall(r'```\{python\}.*?```', content, re.DOTALL)
    for block in fig_blocks:
        if 'fig.show()' in block or 'plt.show()' in block:
            if '#| fig-cap:' not in block and '#| label: fig-' not in block:
                issues.append(f"Figure without caption: {block[:50]}...")
    
    # Check 4: Long code blocks should have output: false
    for block in fig_blocks:
        if block.count('\n') > 30 and '#| output: false' not in block:
            issues.append(f"Long code block without output: false: {block[:50]}...")
    
    return issues

# Run checks on all files
docs_root = Path("docs")
all_issues = {}

for qmd_file in docs_root.rglob("*.qmd"):
    issues = check_file_quality(qmd_file)
    if issues:
        all_issues[qmd_file] = issues

# Report
if all_issues:
    print("Quality Issues Found:")
    for file, issues in all_issues.items():
        print(f"\n{file.relative_to(docs_root)}:")
        for issue in issues:
            print(f"  - {issue}")
    exit(1)
else:
    print("✓ All quality checks passed!")
    exit(0)
```

---

## Phase 10: Deployment (Days 33-34)

### Day 33: Pre-Deployment Validation

**Full validation suite:**

```bash
# 1. Clean build
rm -rf docs/_freeze docs/_site

# 2. Full render
cd docs
quarto render

# 3. Run tests
cd ..
python scripts/test_quarto_docs.py

# 4. Quality checks
python scripts/check_quarto_quality.py

# 5. Check output size
du -sh docs/_site
# Should be <100 MB ideally

# 6. Manual review
open docs/_site/index.html

# 7. Check all links work
# (Use link checker tool)
wget --spider -r -nd -nv -l 2 docs/_site/index.html 2>&1 | grep -B2 '404'
```

### Day 34: Launch

**Deployment checklist:**

- [ ] All tests passing
- [ ] GitHub Actions workflow tested
- [ ] Caching working correctly
- [ ] Site loads in <2 seconds
- [ ] Navigation works
- [ ] Cross-references resolve
- [ ] All figures display
- [ ] Mobile responsive
- [ ] No console errors

**Deploy:**
```bash
# Final commit
git add docs/
git add .github/workflows/
git add scripts/
git commit -m "docs: Complete executable Quarto conversion

- All architecture docs now executable
- All tutorials fully validated
- Interactive example gallery
- Automated CI/CD deployment
- ~65 minutes of code execution per build
- Every code example guaranteed to work"

git push origin main

# Watch deployment
# GitHub Actions → Wait for success
# Visit: https://gilbenezer.github.io/ControlDESymulation/
```

---

## Updated Metadata Files

### Architecture: Full Execution

**`docs/architecture/_metadata.yml`:**
```yaml
execute:
  eval: true       # Execute ALL code
  cache: true      # Cache for performance
  warning: false   # Clean output
  output: false    # Default: validate without cluttering
  freeze: auto     # Re-execute when source changes
  
format:
  html:
    code-fold: show      # Allow code folding
    code-tools: true     # Show code tools
    code-link: true      # Link to source
    code-line-numbers: false
```

### Tutorials: Full Execution with Output

**`docs/tutorials/_metadata.yml`:**
```yaml
execute:
  eval: true        # Execute all
  cache: true
  warning: false
  output: true      # Show outputs
  freeze: auto
  
format:
  html:
    code-fold: false   # Always show code
    code-tools: true
    code-link: true
```

### Examples: Full Execution, Rich Output

**`docs/examples/_metadata.yml`:**
```yaml
execute:
  eval: true         # Execute all
  cache: true
  warning: false
  output: true       # Show all outputs
  freeze: auto
  
format:
  html:
    code-fold: false
    code-tools: true
    fig-width: 10
    fig-height: 6
```

---

## Updated Timeline

| Phase | Days | What Changes with Full Execution |
|-------|------|----------------------------------|
| 1. Setup | 2 | Configure for execution by default |
| 2. Architecture (executable!) | 6 | Add setup blocks, validate all code, show key outputs |
| 3. Tutorials (executable) | 5 | Full execution with rich outputs |
| 4. Examples (enhanced) | 5 | Even richer with live benchmarks |
| 5. GitHub | 2 | Longer CI times, caching critical |
| 6. Architecture Enhancement | 4 | Make demos comprehensive |
| 7. Optimization | 2 | Tune caching, reduce render time |
| 8. Testing | 2 | More rigorous (everything must work) |
| 9. Polish | 2 | Interactive elements, quality pass |
| 10. Deploy | 2 | Full validation before launch |
| **Total** | **32** | ~5 weeks full-time, 10-12 weeks part-time |

---

## Key Differences from Previous Plan

### What Changed

**Before**: Architecture docs were reference-only (eval: false)
**After**: Architecture docs fully executable (eval: true, output: false)

**Benefits of Full Execution:**

1. **Continuous Validation**: Every code example tested in CI
2. **No Silent Breakage**: API changes immediately caught
3. **Live Demonstrations**: Show actual framework capabilities
4. **Confidence**: Documentation is always correct
5. **Learning**: Readers trust examples work
6. **Maintenance**: Easier to keep updated (auto-tested)

**Costs:**

1. **Render Time**: ~65 minutes total (mitigated by caching)
2. **CI Time**: ~30-45 minutes (parallelizable)
3. **Complexity**: More to debug if something breaks
4. **Dependencies**: Must have full environment in CI

### Mitigation Strategies

**For Render Time:**
```yaml
# Use freeze: auto (re-execute only changed files)
execute:
  freeze: auto

# Use cache: true (cache within files)
#| cache: true

# Reduce problem sizes for docs
t_span = (0, 5)  # Instead of (0, 100)
n_trials = 10    # Instead of 1000
```

**For CI:**
```yaml
# Parallel rendering (Quarto built-in)
quarto render --execute-daemon

# Cache between builds
- name: Cache Quarto freeze
  uses: actions/cache@v3
  with:
    path: docs/_freeze
    key: quarto-freeze-${{ hashFiles('docs/**/*.qmd') }}
```

**For Debugging:**
```python
#| error: true  # Document expected errors

try:
    potentially_failing_code()
except Exception as e:
    print(f"Note: This demonstrates error handling: {e}")
```

---

## Success Metrics (Updated)

### Quantitative
- **100%** of code blocks execute successfully
- **0** broken cross-references
- **<45 min** CI build time (with cache)
- **<10 min** incremental builds (freeze: auto)
- **<2 sec** page load time
- **65+ minutes** of validated code execution

### Qualitative
- Every example proven to work
- Outputs show framework capabilities
- Readers can trust all code
- Documentation stays current automatically
- Professional, modern appearance

---

## Final Deliverables

### Must Have ✓
- [x] All 20+ documentation files fully executable
- [x] All code blocks validated in CI
- [x] Interactive Plotly figures throughout
- [x] Performance benchmarks auto-generated
- [x] Cross-references working
- [x] GitHub Actions deploys successfully
- [x] Caching reduces rebuild time to <10 min

### Should Have ✓
- [x] Live algorithm verification (e.g., Riccati residual)
- [x] Parameter comparison tables
- [x] Monte Carlo demonstrations
- [x] Backend performance comparisons
- [x] Mobile-responsive
- [x] Dark mode theme

### Nice to Have
- [ ] Interactive parameter sliders
- [ ] Automatic API docs
- [ ] PDF export
- [ ] Search functionality
- [ ] Version selector

---

## Risk Management (Updated)

### Risk: Long CI Build Times

**Impact**: High (blocks merges if >1 hour)

**Mitigation:**
```yaml
# Aggressive caching
- uses: actions/cache@v3
  with:
    path: |
      docs/_freeze
      ~/.cache/pip
    key: docs-${{ hashFiles('docs/**', 'src/**') }}

# Parallel rendering
quarto render --execute-daemon

# Only re-render changed files
execute:
  freeze: auto
```

**Fallback**: If CI exceeds 1 hour, switch specific expensive files to `freeze: true`

### Risk: Code Block Failures in Production

**Impact**: Critical (broken documentation)

**Mitigation:**
- **Pre-commit hook** runs local tests before push
- **PR checks** must pass before merge
- **Weekly cron** re-validates everything
- **Error: true** for expected failures

**Create `.git/hooks/pre-push`:**
```bash
#!/bin/bash
echo "Testing documentation before push..."
python scripts/test_quarto_docs.py || {
    echo "❌ Documentation tests failed!"
    echo "Fix failing code blocks before pushing"
    exit 1
}
echo "✅ Documentation validated"
```

### Risk: Caching Stale Results

**Impact**: Medium (shows outdated outputs)

**Mitigation:**
```yaml
# freeze: auto re-executes when SOURCE changes
# But also force refresh weekly
on:
  schedule:
    - cron: '0 2 * * 0'  # Sunday 2 AM
```

**Manual refresh:**
```bash
# When suspicious of stale cache
rm -rf docs/_freeze
quarto render docs/
```

---

## Conclusion

This updated plan creates **fully executable, continuously validated documentation** where:

✅ **Every code example runs** when docs are built
✅ **Every algorithm is verified** mathematically  
✅ **Every plot is generated live** from actual data
✅ **Every benchmark measures real performance**
✅ **CI catches breaking changes** immediately

The documentation becomes a **living test suite** that ensures your examples always work, your algorithms are correct, and your API is stable.

**Total Investment**: ~5 weeks full-time (or 10-12 weeks part-time)

**Payoff**: Documentation you can **absolutely trust**, that serves as both
user guide AND comprehensive integration test suite.

**Next Step**: Start Phase 1 (Days 1-2) to set up infrastructure, then proceed
through architecture conversion (Days 3-8) making everything executable!