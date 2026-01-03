# Control Framework Architecture

## Overview

The **Control Framework** provides classical control theory algorithms for analysis and synthesis of dynamical systems. It consists of **3 core modules** totaling ~1,786 lines organized into a clean **2-layer architecture**: pure stateless functions and thin composition wrappers.

## Architecture Philosophy

**Functional Design with Composition** - The control framework enables:

1. **Pure Functions** - Stateless algorithms like scipy (design_lqr, analyze_stability)
2. **Thin Wrappers** - Minimal composition layer for system integration
3. **Type Safety** - TypedDict results for all algorithms
4. **Backend Consistency** - Automatic backend handling from parent system
5. **Separation of Concerns** - Analysis vs synthesis clearly separated
6. **Mathematical Rigor** - Implements classical control theory correctly
7. **System Integration** - Clean `system.control` and `system.analysis` APIs

```python
# Pure functions - stateless, reusable
result = design_lqr(A, B, Q, R, system_type='continuous')

# Composition wrappers - integrate with systems
result = system.control.design_lqr(A, B, Q, R, system_type='continuous')
```

## Framework Layers

```
┌────────────────────────────────────────────────────────────┐
│                   APPLICATION LAYER                         │
│              (ContinuousSystemBase, DiscreteSystemBase)     │
│                                                              │
│  system.control   ──────► ControlSynthesis (388 lines)     │
│  system.analysis  ──────► SystemAnalysis   (431 lines)     │
└──────────────────┬─────────────────────────────────────────┘
                   │ delegates to
                   ↓
┌────────────────────────────────────────────────────────────┐
│              PURE FUNCTION LAYER                            │
│          classical_control_functions.py (967 lines)         │
│                                                              │
│  Control Design:          System Analysis:                  │
│  • design_lqr()          • analyze_stability()             │
│  • design_kalman()       • analyze_controllability()       │
│  • design_lqg()          • analyze_observability()         │
│                                                              │
│  All functions are stateless, pure, backend-agnostic       │
└──────────────────┬─────────────────────────────────────────┘
                   │ returns
                   ↓
┌────────────────────────────────────────────────────────────┐
│                   TYPE LAYER                                │
│             control_classical.py (542 lines)                │
│                                                              │
│  • LQRResult           • StabilityInfo                      │
│  • KalmanFilterResult  • ControllabilityInfo                │
│  • LQGResult           • ObservabilityInfo                  │
└────────────────────────────────────────────────────────────┘
```

## Module Breakdown

### Pure Function Layer

#### classical_control_functions.py
**File:** `classical_control_functions.py` (967 lines)

**Purpose:** Stateless algorithms for control design and system analysis

**Design Philosophy:**

- Pure functions (no side effects, no state)
- Works like scipy - takes matrices in, returns TypedDict
- Backend conversion handled internally
- Mathematical correctness guaranteed
- Comprehensive error checking

**Categories:**

**1. Control Design Functions**

```python
def design_lqr(
    A: StateMatrix,
    B: InputMatrix,
    Q: StateMatrix,
    R: InputMatrix,
    N: Optional[InputMatrix] = None,
    system_type: str = "discrete",
    backend: Backend = "numpy"
) -> LQRResult:
    """
    Design Linear Quadratic Regulator (LQR) controller.
    
    Unified interface for continuous and discrete LQR.
    
    Mathematical Background:
        Continuous: Minimize J = ∫₀^∞ (x'Qx + u'Ru + 2x'Nu) dt
        Discrete:   Minimize J = Σₖ₌₀^∞ (x'Qx + u'Ru + 2x'Nu)
    
    Solution via Algebraic Riccati Equation (ARE):
        Continuous: A'P + PA - PBR⁻¹B'P + Q = 0
        Discrete:   P = A'PA - A'PB(R + B'PB)⁻¹B'PA + Q
    
    Optimal Gain:
        Continuous: K = R⁻¹B'P
        Discrete:   K = (R + B'PB)⁻¹B'PA
    
    Args:
        A: State matrix (nx, nx)
        B: Input matrix (nx, nu)
        Q: State cost matrix (nx, nx), Q ≥ 0, (Q,A) detectable
        R: Control cost matrix (nu, nu), R > 0
        N: Cross-coupling matrix (nx, nu), default None
        system_type: 'continuous' or 'discrete'
        backend: Computational backend
    
    Returns:
        LQRResult with:
            - gain: Optimal feedback gain K (nu, nx)
            - cost_to_go: Riccati solution P (nx, nx)
            - closed_loop_eigenvalues: eig(A - BK)
            - stability_margin: Stability robustness measure
    
    Raises:
        ValueError: If Q, R dimensions incompatible
        LinAlgError: If Riccati equation has no stabilizing solution
    
    Examples:
        >>> A = np.array([[0, 1], [-2, -3]])
        >>> B = np.array([[0], [1]])
        >>> Q = np.diag([10, 1])
        >>> R = np.array([[0.1]])
        >>> 
        >>> result = design_lqr(A, B, Q, R, system_type='continuous')
        >>> K = result['gain']
        >>> A_cl = A - B @ K
        >>> print(f"Closed-loop eigenvalues: {result['closed_loop_eigenvalues']}")
    """
```

```python
def design_kalman_filter(
    A: StateMatrix,
    C: OutputMatrix,
    Q: StateMatrix,
    R: OutputMatrix,
    system_type: str = "discrete",
    backend: Backend = "numpy"
) -> KalmanFilterResult:
    """
    Design Kalman filter for optimal state estimation.
    
    System Model:
        x[k+1] = Ax[k] + Bu[k] + w[k],  w ~ N(0, Q)
        y[k] = Cx[k] + v[k],            v ~ N(0, R)
    
    Estimator Dynamics:
        x̂[k+1] = Ax̂[k] + Bu[k] + L(y[k] - Cx̂[k])
    
    Optimal Gain:
        L = APC'(CPC' + R)⁻¹
    
    Riccati Equation:
        P = A(P - PC'(CPC'+R)⁻¹CP)A' + Q
    
    Args:
        A: State matrix (nx, nx)
        C: Output matrix (ny, nx)
        Q: Process noise covariance (nx, nx), Q ≥ 0
        R: Measurement noise covariance (ny, ny), R > 0
        system_type: 'continuous' or 'discrete'
        backend: Computational backend
    
    Returns:
        KalmanFilterResult with:
            - gain: Kalman gain L (nx, ny)
            - error_covariance: Steady-state P (nx, nx)
            - innovation_covariance: S = CPC' + R (ny, ny)
            - observer_eigenvalues: eig(A - LC)
    
    Examples:
        >>> A = np.array([[1, 0.1], [0, 0.9]])
        >>> C = np.array([[1, 0]])
        >>> Q_proc = 0.01 * np.eye(2)
        >>> R_meas = np.array([[0.1]])
        >>> 
        >>> kalman = design_kalman_filter(A, C, Q_proc, R_meas)
        >>> L = kalman['gain']
        >>> print(f"Observer poles: {kalman['observer_eigenvalues']}")
    """
```

```python
def design_lqg(
    A: StateMatrix,
    B: InputMatrix,
    C: OutputMatrix,
    Q_state: StateMatrix,
    R_control: InputMatrix,
    Q_process: StateMatrix,
    R_measurement: OutputMatrix,
    N: Optional[InputMatrix] = None,
    system_type: str = "discrete",
    backend: Backend = "numpy"
) -> LQGResult:
    """
    Design Linear Quadratic Gaussian (LQG) controller.
    
    Combines LQR (optimal control) with Kalman filter (optimal estimation)
    via the separation principle.
    
    Controller: u[k] = -Kx̂[k]
    Estimator: x̂[k+1] = Ax̂[k] + Bu[k] + L(y[k] - Cx̂[k])
    
    Separation Principle:
        - Design LQR assuming full state available
        - Design Kalman filter independently
        - Combine: optimal for linear Gaussian systems
        - Closed-loop poles = controller poles ∪ observer poles
    
    Args:
        A: State matrix (nx, nx)
        B: Input matrix (nx, nu)
        C: Output matrix (ny, nx)
        Q_state: LQR state cost (nx, nx)
        R_control: LQR control cost (nu, nu)
        Q_process: Process noise covariance (nx, nx)
        R_measurement: Measurement noise covariance (ny, ny)
        N: Cross-coupling (nx, nu), optional
        system_type: 'continuous' or 'discrete'
        backend: Computational backend
    
    Returns:
        LQGResult with:
            - control_gain: LQR gain K (nu, nx)
            - estimator_gain: Kalman gain L (nx, ny)
            - control_cost_to_go: Controller Riccati P_c
            - estimation_error_covariance: Estimator Riccati P_e
            - separation_verified: bool
            - closed_loop_stable: bool
            - controller_eigenvalues: eig(A - BK)
            - estimator_eigenvalues: eig(A - LC)
    
    Examples:
        >>> A = np.array([[1, 0.1], [0, 0.9]])
        >>> B = np.array([[0], [0.1]])
        >>> C = np.array([[1, 0]])
        >>> 
        >>> lqg = design_lqg(
        ...     A, B, C,
        ...     Q_state=np.diag([10, 1]),
        ...     R_control=np.array([[0.1]]),
        ...     Q_process=0.01*np.eye(2),
        ...     R_measurement=np.array([[0.1]])
        ... )
        >>> 
        >>> K = lqg['control_gain']
        >>> L = lqg['estimator_gain']
        >>> print(f"LQG stable: {lqg['closed_loop_stable']}")
    """
```

**2. System Analysis Functions**

```python
def analyze_stability(
    A: StateMatrix,
    system_type: str = "continuous",
    backend: Backend = "numpy"
) -> StabilityInfo:
    """
    Analyze system stability via eigenvalue placement.
    
    Stability Criteria:
        Continuous: All Re(λ) < 0 (left half-plane)
        Discrete:   All |λ| < 1 (inside unit circle)
    
    Marginal Stability:
        Continuous: max Re(λ) ≈ 0
        Discrete:   max |λ| ≈ 1
    
    Args:
        A: State matrix (nx, nx)
        system_type: 'continuous' or 'discrete'
        backend: Computational backend
    
    Returns:
        StabilityInfo with:
            - eigenvalues: Complex eigenvalues of A
            - magnitudes: |λ| for each eigenvalue
            - max_magnitude: max |λ| (spectral radius)
            - spectral_radius: Same as max_magnitude
            - is_stable: Asymptotic stability
            - is_marginally_stable: On boundary
            - is_unstable: At least one unstable mode
    
    Examples:
        >>> A = np.array([[0, 1], [-2, -3]])
        >>> stability = analyze_stability(A, system_type='continuous')
        >>> 
        >>> if stability['is_stable']:
        ...     print(f"Stable! Eigenvalues: {stability['eigenvalues']}")
        ...     print(f"Spectral radius: {stability['spectral_radius']:.3f}")
    """
```

```python
def analyze_controllability(
    A: StateMatrix,
    B: InputMatrix,
    backend: Backend = "numpy"
) -> ControllabilityInfo:
    """
    Test system controllability via rank condition.
    
    Controllability Test:
        rank(C) = nx where C = [B, AB, A²B, ..., Aⁿ⁻¹B]
    
    Interpretation:
        - Controllable: All states can be driven to any value
        - Uncontrollable: Some states cannot be influenced by control
    
    Args:
        A: State matrix (nx, nx)
        B: Input matrix (nx, nu)
        backend: Computational backend
    
    Returns:
        ControllabilityInfo with:
            - controllability_matrix: C = [B AB ... Aⁿ⁻¹B] (nx, nx*nu)
            - rank: Rank of controllability matrix
            - is_controllable: rank == nx
            - uncontrollable_modes: Eigenvalues of uncontrollable subsystem
    
    Examples:
        >>> A = np.array([[0, 1], [-2, -3]])
        >>> B = np.array([[0], [1]])
        >>> 
        >>> ctrl = analyze_controllability(A, B)
        >>> print(f"Controllable: {ctrl['is_controllable']}")
        >>> print(f"Rank: {ctrl['rank']} / {A.shape[0]}")
    """
```

```python
def analyze_observability(
    A: StateMatrix,
    C: OutputMatrix,
    backend: Backend = "numpy"
) -> ObservabilityInfo:
    """
    Test system observability via rank condition.
    
    Observability Test:
        rank(O) = nx where O = [C; CA; CA²; ...; CAⁿ⁻¹]
    
    Interpretation:
        - Observable: Initial state can be determined from outputs
        - Unobservable: Some states hidden from measurements
    
    Args:
        A: State matrix (nx, nx)
        C: Output matrix (ny, nx)
        backend: Computational backend
    
    Returns:
        ObservabilityInfo with:
            - observability_matrix: O = [C; CA; ...] (nx*ny, nx)
            - rank: Rank of observability matrix
            - is_observable: rank == nx
            - unobservable_modes: Eigenvalues of unobservable subsystem
    
    Examples:
        >>> A = np.array([[0, 1], [-2, -3]])
        >>> C = np.array([[1, 0]])
        >>> 
        >>> obs = analyze_observability(A, C)
        >>> print(f"Observable: {obs['is_observable']}")
        >>> print(f"Rank: {obs['rank']} / {A.shape[0]}")
    """
```

**Key Design Features:**

- **Pure functions** - No state, no side effects
- **Scipy-like API** - Familiar interface for control engineers
- **Backend agnostic** - Internal conversion to/from NumPy
- **Comprehensive validation** - Dimension checks, positive-definiteness
- **Riccati solvers** - scipy.linalg.solve_continuous_are/solve_discrete_are
- **Error handling** - Clear exceptions for infeasible problems
- **TypedDict returns** - Type-safe structured results

**Internal Utilities:**
```python
def _to_numpy(arr, backend: Backend) -> np.ndarray:
    """Convert from any backend to NumPy for scipy."""

def _from_numpy(arr: np.ndarray, backend: Backend):
    """Convert from NumPy back to original backend."""
```

---

### Composition Wrapper Layer

#### control_synthesis.py
**File:** `control_synthesis.py` (388 lines)

**Purpose:** Thin wrapper for control design algorithms

**Design Philosophy:**

- **Composition not inheritance** - Utility held by system, not base class
- **No state** - Only stores backend setting from parent
- **No caching** - Delegates immediately to pure functions
- **Clean API** - Methods match control theory terminology

**Architecture:**
```python
class ControlSynthesis:
    """
    Control synthesis wrapper for system composition.
    
    Thin wrapper that routes to pure control design functions while
    maintaining backend consistency with parent system.
    
    Attributes:
        backend: Backend setting from parent system
    """
    
    def __init__(self, backend: Backend = "numpy"):
        self.backend = backend
    
    def design_lqr(self, A, B, Q, R, N=None, system_type='discrete'):
        """Route to classical_control_functions.design_lqr()"""
        from src.control.classical_control_functions import design_lqr
        return design_lqr(A, B, Q, R, N, system_type, self.backend)
    
    def design_kalman(self, A, C, Q, R, system_type='discrete'):
        """Route to classical_control_functions.design_kalman_filter()"""
        from src.control.classical_control_functions import design_kalman_filter
        return design_kalman_filter(A, C, Q, R, system_type, self.backend)
    
    def design_lqg(self, A, B, C, Q_state, R_control, Q_process, R_measurement, N=None, system_type='discrete'):
        """Route to classical_control_functions.design_lqg()"""
        from src.control.classical_control_functions import design_lqg
        return design_lqg(A, B, C, Q_state, R_control, Q_process, R_measurement, N, system_type, self.backend)
```

**Integration with Systems:**
```python
# In ContinuousSystemBase / DiscreteSystemBase
@property
def control(self) -> ControlSynthesis:
    """Access control synthesis utilities."""
    if not hasattr(self, '_control_synthesis'):
        from src.control.control_synthesis import ControlSynthesis
        self._control_synthesis = ControlSynthesis(backend=self._default_backend)
    return self._control_synthesis
```

**Usage Pattern:**
```python
# Via system composition (typical)
system = Pendulum()
A, B = system.linearize(x_eq, u_eq)

# Design controller - backend handled automatically
result = system.control.design_lqr(A, B, Q, R, system_type='continuous')
K = result['gain']

# Control law
u = -K @ (x - x_eq)
```

---

#### system_analysis.py
**File:** `system_analysis.py` (431 lines)

**Purpose:** Thin wrapper for system analysis algorithms

**Design Philosophy:**

- **Identical to ControlSynthesis** - Same thin wrapper pattern
- **Separation of concerns** - Analysis separate from synthesis
- **Consistent interface** - Matches control_synthesis.py design

**Architecture:**
```python
class SystemAnalysis:
    """
    System analysis wrapper for composition.
    
    Thin wrapper that routes to pure system analysis functions while
    maintaining backend consistency with parent system.
    
    Attributes:
        backend: Backend setting from parent system
    """
    
    def __init__(self, backend: Backend = "numpy"):
        self.backend = backend
    
    def stability(self, A, system_type='continuous'):
        """Route to classical_control_functions.analyze_stability()"""
        from src.control.classical_control_functions import analyze_stability
        return analyze_stability(A, system_type, self.backend)
    
    def controllability(self, A, B):
        """Route to classical_control_functions.analyze_controllability()"""
        from src.control.classical_control_functions import analyze_controllability
        return analyze_controllability(A, B, self.backend)
    
    def observability(self, A, C):
        """Route to classical_control_functions.analyze_observability()"""
        from src.control.classical_control_functions import analyze_observability
        return analyze_observability(A, C, self.backend)
```

**Integration with Systems:**
```python
# In ContinuousSystemBase / DiscreteSystemBase
@property
def analysis(self) -> SystemAnalysis:
    """Access system analysis utilities."""
    if not hasattr(self, '_system_analysis'):
        from src.control.system_analysis import SystemAnalysis
        self._system_analysis = SystemAnalysis(backend=self._default_backend)
    return self._system_analysis
```

**Usage Pattern:**
```python
# Via system composition (typical)
system = Pendulum()
A, B = system.linearize(x_eq, u_eq)
C = np.array([[1, 0]])  # Measure position

# Analyze system properties
stability = system.analysis.stability(A, system_type='continuous')
controllability = system.analysis.controllability(A, B)
observability = system.analysis.observability(A, C)

# Check results
if stability['is_stable']:
    print(f"Stable with margin: {stability['stability_margin']:.3f}")

if controllability['is_controllable'] and observability['is_observable']:
    print("System is minimal (controllable and observable)")
```

---

### Type Layer

#### control_classical.py
**File:** `control_classical.py` (542 lines)

**Purpose:** TypedDict result types for control algorithms

See `Type_System_Architecture.md` for complete documentation.

**Key Types:**

- `LQRResult` - LQR controller design result
- `KalmanFilterResult` - Kalman filter design result
- `LQGResult` - LQG controller design result
- `StabilityInfo` - Stability analysis result
- `ControllabilityInfo` - Controllability test result
- `ObservabilityInfo` - Observability test result

---

## Design Patterns

### Pattern 1: Pure Functions + Thin Wrappers

**Why this architecture?**

```python
# ANTI-PATTERN: Methods on system class (violates SRP)
class ContinuousSystemBase:
    def design_lqr(self, Q, R):
        # 200+ lines of LQR implementation
        # Mixes system concerns with control algorithm
        pass

# GOOD PATTERN: Pure function + composition
# Pure function (classical_control_functions.py)
def design_lqr(A, B, Q, R, system_type, backend):
    """Stateless, testable, reusable."""
    # 200+ lines focused solely on LQR algorithm
    return LQRResult(...)

# Thin wrapper (control_synthesis.py)
class ControlSynthesis:
    def design_lqr(self, A, B, Q, R, system_type):
        return design_lqr(A, B, Q, R, system_type, self.backend)

# System integration (continuous_system_base.py)
@property
def control(self) -> ControlSynthesis:
    return ControlSynthesis(backend=self._default_backend)
```

**Benefits:**

- ✅ **Single Responsibility** - Pure functions do one thing
- ✅ **Testability** - Functions easy to unit test
- ✅ **Reusability** - Functions work standalone
- ✅ **Composition** - System uses utilities, doesn't inherit them
- ✅ **Maintainability** - Changes isolated to function layer

### Pattern 2: Backend Agnosticism

```python
def design_lqr(..., backend: Backend):
    """Works with NumPy, PyTorch, JAX transparently."""
    
    # Convert to NumPy for scipy
    A_np = _to_numpy(A, backend)
    B_np = _to_numpy(B, backend)
    
    # Solve in NumPy (scipy.linalg)
    P = solve_continuous_are(A_np, B_np, Q_np, R_np)
    K = np.linalg.solve(R_np, B_np.T @ P)
    
    # Convert back to original backend
    K_result = _from_numpy(K, backend)
    P_result = _from_numpy(P, backend)
    
    return LQRResult(gain=K_result, cost_to_go=P_result, ...)
```

### Pattern 3: TypedDict Results

```python
# All functions return structured TypedDict
result: LQRResult = design_lqr(A, B, Q, R, system_type='continuous')

# IDE autocomplete knows all fields
K = result['gain']                    # ✓ Valid
P = result['cost_to_go']              # ✓ Valid
eigs = result['closed_loop_eigenvalues']  # ✓ Valid
bad = result['nonexistent_key']       # ✗ Type error!

# Type checking prevents errors
def apply_control(result: LQRResult) -> np.ndarray:
    return result['gain']  # ✓ Type checker verifies
```

### Pattern 4: Unified Continuous/Discrete Interface

```python
# Same function handles both continuous and discrete
def design_lqr(A, B, Q, R, N=None, system_type='discrete', backend='numpy'):
    """Unified interface - system_type selects algorithm."""
    
    if system_type == 'continuous':
        # Continuous-time algebraic Riccati equation
        P = solve_continuous_are(A, B, Q, R, s=N)
        K = np.linalg.solve(R, B.T @ P)
    elif system_type == 'discrete':
        # Discrete-time algebraic Riccati equation
        P = solve_discrete_are(A, B, Q, R, s=N)
        K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    else:
        raise ValueError(f"Invalid system_type: {system_type}")
    
    # Rest of implementation identical
    closed_loop_eigs = np.linalg.eigvals(A - B @ K)
    
    return LQRResult(gain=K, cost_to_go=P, ...)
```

**Benefits:**

- ✅ Single function for both cases
- ✅ Less code duplication
- ✅ Easier to maintain
- ✅ Consistent API across system types

---

## Mathematical Algorithms

### LQR Design

**Continuous-Time LQR:**

**Cost Functional:**
```
J = ∫₀^∞ (x'Qx + u'Ru + 2x'Nu) dt
```

**Algebraic Riccati Equation (ARE):**
```
A'P + PA - PBR⁻¹B'P + Q - N'R⁻¹N = 0
```

**Optimal Gain:**
```
K = R⁻¹(B'P + N')
```

**Closed-Loop Dynamics:**
```
ẋ = (A - BK)x
```

**Discrete-Time LQR:**

**Cost Functional:**
```
J = Σₖ₌₀^∞ (x[k]'Qx[k] + u[k]'Ru[k] + 2x[k]'Nu[k])
```

**Discrete ARE:**
```
P = A'PA - (A'PB + N)(R + B'PB)⁻¹(B'PA + N') + Q
```

**Optimal Gain:**
```
K = (R + B'PB)⁻¹(B'PA + N')
```

**Closed-Loop Dynamics:**
```
x[k+1] = (A - BK)x[k]
```

### Kalman Filter Design

**System Model:**
```
x[k+1] = Ax[k] + Bu[k] + w[k],  w ~ N(0, Q)
y[k] = Cx[k] + v[k],            v ~ N(0, R)
```

**Estimator:**
```
x̂[k+1] = Ax̂[k] + Bu[k] + L(y[k] - Cx̂[k])
```

**Kalman Gain:**
```
L = APC'(CPC' + R)⁻¹
```

**Error Covariance Riccati Equation:**
```
P = A(P - PC'(CPC' + R)⁻¹CP)A' + Q
```

**Error Dynamics:**
```
e[k+1] = (A - LC)e[k] + w[k] - Lv[k]
```

### LQG Controller

**Separation Principle:**

1. Design LQR assuming full state feedback: `u = -Kx`
2. Design Kalman filter for state estimation: `x̂[k+1] = Ax̂[k] + Bu[k] + L(y[k] - Cx̂[k])`
3. Combine: `u = -Kx̂` (certainty equivalence)

**Closed-Loop System:**
```
[x[k+1]]   [A - BK    BK  ] [x[k] ]   [w[k]]
[e[k+1]] = [  0     A - LC] [e[k] ] + [w[k] - Lv[k]]
```

**Eigenvalues:**

- Controller poles: eig(A - BK)
- Observer poles: eig(A - LC)
- Combined poles: eig(A - BK) ∪ eig(A - LC)

**Optimality:**

- Optimal for linear systems with Gaussian noise
- Minimizes steady-state error covariance
- Separation holds for linear systems

---

## Usage Workflows

### Workflow 1: Design LQR Controller

```python
from src.systems.examples import Pendulum
import numpy as np

# 1. Create system
system = Pendulum(m=1.0, l=0.5, g=9.81, b=0.1)

# 2. Define equilibrium (upright position)
x_eq = np.array([np.pi, 0])  # [theta, omega]
u_eq = np.zeros(1)

# 3. Linearize at equilibrium
A, B = system.linearize(x_eq, u_eq)

# 4. Design LQR controller
Q = np.diag([10, 1])  # Penalize angle more than velocity
R = np.array([[0.1]])  # Small control cost

result = system.control.design_lqr(
    A, B, Q, R,
    system_type='continuous'
)

# 5. Extract gain and verify stability
K = result['gain']
print(f"LQR gain: {K}")
print(f"Closed-loop eigenvalues: {result['closed_loop_eigenvalues']}")
print(f"Stable: {np.all(np.real(result['closed_loop_eigenvalues']) < 0)}")

# 6. Implement controller
def lqr_controller(x, t):
    return -K @ (x - x_eq)

# 7. Simulate closed-loop
result_sim = system.simulate(
    x0=np.array([np.pi + 0.2, 0]),  # Start near upright
    controller=lqr_controller,
    t_span=(0, 10),
    dt=0.01
)
```

### Workflow 2: Design Kalman Filter

```python
# 1. Same system and linearization
A, B = system.linearize(x_eq, u_eq)

# 2. Define measurement model (measure angle only)
C = np.array([[1, 0]])  # Observe theta, not omega

# 3. Define noise covariances
Q_process = 0.01 * np.eye(2)  # Process noise
R_measurement = np.array([[0.1]])  # Measurement noise

# 4. Design Kalman filter
kalman = system.control.design_kalman(
    A, C, Q_process, R_measurement,
    system_type='continuous'
)

# 5. Extract gain
L = kalman['gain']
print(f"Kalman gain: {L}")
print(f"Observer eigenvalues: {kalman['observer_eigenvalues']}")

# 6. Implement estimator
x_hat = np.zeros(2)
for k in range(N):
    # Prediction
    x_hat_pred = A @ x_hat + B @ u[k]
    
    # Measurement update
    y_meas = C @ x_true[k] + np.random.randn() * np.sqrt(0.1)
    innovation = y_meas - C @ x_hat_pred
    x_hat = x_hat_pred + L @ innovation
```

### Workflow 3: Design LQG Controller

```python
# 1. Define all matrices
A, B = system.linearize(x_eq, u_eq)
C = np.array([[1, 0]])  # Partial state measurement

# 2. Design weights
Q_state = np.diag([10, 1])    # LQR state cost
R_control = np.array([[0.1]])  # LQR control cost
Q_process = 0.01 * np.eye(2)  # Kalman process noise
R_measurement = np.array([[0.1]])  # Kalman measurement noise

# 3. Design LQG
lqg = system.control.design_lqg(
    A, B, C,
    Q_state, R_control,
    Q_process, R_measurement,
    system_type='continuous'
)

# 4. Extract both gains
K = lqg['control_gain']
L = lqg['estimator_gain']

print(f"LQG stable: {lqg['closed_loop_stable']}")
print(f"Separation verified: {lqg['separation_verified']}")

# 5. Implement LQG controller
x_hat = np.zeros(2)
for k in range(N):
    # Control (certainty equivalence)
    u[k] = -K @ (x_hat - x_eq)
    
    # Prediction
    x_hat = A @ x_hat + B @ u[k]
    
    # Measurement update
    y_meas = C @ x_true[k] + measurement_noise[k]
    innovation = y_meas - C @ x_hat
    x_hat = x_hat + L @ innovation
```

### Workflow 4: System Analysis

```python
# 1. Linearize system
A, B = system.linearize(x_eq, u_eq)
C = np.array([[1, 0]])

# 2. Check stability
stability = system.analysis.stability(A, system_type='continuous')
print(f"Stable: {stability['is_stable']}")
print(f"Eigenvalues: {stability['eigenvalues']}")
print(f"Spectral radius: {stability['spectral_radius']:.3f}")

# 3. Check controllability
ctrl = system.analysis.controllability(A, B)
print(f"Controllable: {ctrl['is_controllable']}")
print(f"Rank: {ctrl['rank']} / {A.shape[0]}")

if not ctrl['is_controllable']:
    print(f"Uncontrollable modes: {ctrl['uncontrollable_modes']}")

# 4. Check observability
obs = system.analysis.observability(A, C)
print(f"Observable: {obs['is_observable']}")
print(f"Rank: {obs['rank']} / {A.shape[0]}")

# 5. Verify conditions for LQR/Kalman
if ctrl['is_controllable']:
    print("✓ Can design LQR controller")
else:
    print("✗ Cannot design LQR - system not controllable")

if obs['is_observable']:
    print("✓ Can design Kalman filter")
else:
    print("✗ Cannot design Kalman filter - system not observable")

if ctrl['is_controllable'] and obs['is_observable']:
    print("✓ System is minimal - can design LQG controller")
```

## Key Strengths

1. **Pure Functional Core** - Stateless algorithms, easy to test
2. **Thin Wrappers** - Minimal composition layer, no business logic
3. **Type Safety** - TypedDict results throughout
4. **Backend Agnostic** - NumPy/PyTorch/JAX transparent
5. **Separation of Concerns** - Analysis vs synthesis clearly separated
6. **Mathematical Rigor** - Correct implementation of classical control
7. **Clean Integration** - `system.control` and `system.analysis` APIs
8. **Unified Interface** - Single function for continuous/discrete
9. **Comprehensive** - LQR, Kalman, LQG, stability, controllability, observability
10. **Scipy-like** - Familiar API for control engineers

This control framework is the **foundation** for classical control theory in ControlDESymulation!
