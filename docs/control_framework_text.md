# Control Framework Architecture (Text Diagram)

```
═══════════════════════════════════════════════════════════════════════
                   CONTROL FRAMEWORK ARCHITECTURE
═══════════════════════════════════════════════════════════════════════

                    ┌──────────────────────────┐
                    │  APPLICATION LAYER       │
                    │                          │
                    │  ContinuousSystemBase    │
                    │  DiscreteSystemBase      │
                    └──────────┬───────────────┘
                               │
                ┌──────────────┴───────────────┐
                │                              │
                ↓                              ↓
    ┌───────────────────────┐      ┌───────────────────────┐
    │  system.control       │      │  system.analysis      │
    │  ControlSynthesis     │      │  SystemAnalysis       │
    │  (388 lines)          │      │  (431 lines)          │
    └───────────┬───────────┘      └───────────┬───────────┘
                │                              │
                │  delegates to                │  delegates to
                │                              │
                └──────────────┬───────────────┘
                               │
                               ↓
            ┌──────────────────────────────────────┐
            │  PURE FUNCTION LAYER                 │
            │  classical_control_functions.py      │
            │  (967 lines)                         │
            │                                      │
            │  ┌────────────────────────────────┐ │
            │  │ Control Design Functions       │ │
            │  ├────────────────────────────────┤ │
            │  │ • design_lqr()                 │ │
            │  │ • design_kalman_filter()       │ │
            │  │ • design_lqg()                 │ │
            │  └────────────────────────────────┘ │
            │                                      │
            │  ┌────────────────────────────────┐ │
            │  │ System Analysis Functions      │ │
            │  ├────────────────────────────────┤ │
            │  │ • analyze_stability()          │ │
            │  │ • analyze_controllability()    │ │
            │  │ • analyze_observability()      │ │
            │  └────────────────────────────────┘ │
            │                                      │
            │  All functions are:                  │
            │  - Pure (no state)                   │
            │  - Backend agnostic                  │
            │  - TypedDict results                 │
            └──────────────┬───────────────────────┘
                           │ returns
                           ↓
            ┌──────────────────────────────────────┐
            │  TYPE LAYER                          │
            │  control_classical.py (542 lines)    │
            │                                      │
            │  ┌────────────────────────────────┐ │
            │  │ Analysis Result Types          │ │
            │  ├────────────────────────────────┤ │
            │  │ • StabilityInfo                │ │
            │  │ • ControllabilityInfo          │ │
            │  │ • ObservabilityInfo            │ │
            │  └────────────────────────────────┘ │
            │                                      │
            │  ┌────────────────────────────────┐ │
            │  │ Control Design Result Types    │ │
            │  ├────────────────────────────────┤ │
            │  │ • LQRResult                    │ │
            │  │ • KalmanFilterResult           │ │
            │  │ • LQGResult                    │ │
            │  └────────────────────────────────┘ │
            └──────────────────────────────────────┘


═══════════════════════════════════════════════════════════════════════
                        MODULE BREAKDOWN
═══════════════════════════════════════════════════════════════════════

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ PURE FUNCTION LAYER: classical_control_functions.py (967 lines) ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

Control Design Functions
├─ design_lqr()
│  ├─ Continuous-time LQR
│  │  └─> solve_continuous_are(A, B, Q, R)
│  ├─ Discrete-time LQR
│  │  └─> solve_discrete_are(A, B, Q, R)
│  └─> Returns: LQRResult
│
├─ design_kalman_filter()
│  ├─ Continuous-time Kalman
│  │  └─> solve_continuous_are(A.T, C.T, Q, R)
│  ├─ Discrete-time Kalman
│  │  └─> solve_discrete_are(A.T, C.T, Q, R)
│  └─> Returns: KalmanFilterResult
│
└─ design_lqg()
   ├─> Calls: design_lqr() + design_kalman_filter()
   ├─> Verifies: separation_principle
   ├─> Checks: closed_loop_stability
   └─> Returns: LQGResult

System Analysis Functions
├─ analyze_stability()
│  ├─> Computes: eigenvalues of A
│  ├─> Tests:
│  │  • Continuous: Re(λ) < 0 ?
│  │  • Discrete:   |λ| < 1 ?
│  └─> Returns: StabilityInfo
│
├─ analyze_controllability()
│  ├─> Constructs: C = [B AB A²B ... Aⁿ⁻¹B]
│  ├─> Computes: rank(C)
│  ├─> Tests: rank(C) == nx ?
│  └─> Returns: ControllabilityInfo
│
└─ analyze_observability()
   ├─> Constructs: O = [C; CA; CA²; ...; CAⁿ⁻¹]
   ├─> Computes: rank(O)
   ├─> Tests: rank(O) == nx ?
   └─> Returns: ObservabilityInfo

Internal Utilities
├─ _to_numpy(arr, backend)
│  └─> Converts: torch/jax → numpy (for scipy)
│
└─ _from_numpy(arr, backend)
   └─> Converts: numpy → torch/jax (back to original)


┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ WRAPPER LAYER                                                    ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

ControlSynthesis (388 lines)
├─ __init__(backend)
│  └─> Stores: self.backend
│
├─ design_lqr(A, B, Q, R, N, system_type)
│  └─> Routes: design_lqr(..., backend=self.backend)
│
├─ design_kalman(A, C, Q, R, system_type)
│  └─> Routes: design_kalman_filter(..., backend=self.backend)
│
└─ design_lqg(A, B, C, Q_state, R_control, Q_process, R_measurement, N, system_type)
   └─> Routes: design_lqg(..., backend=self.backend)

SystemAnalysis (431 lines)
├─ __init__(backend)
│  └─> Stores: self.backend
│
├─ stability(A, system_type)
│  └─> Routes: analyze_stability(..., backend=self.backend)
│
├─ controllability(A, B)
│  └─> Routes: analyze_controllability(..., backend=self.backend)
│
└─ observability(A, C)
   └─> Routes: analyze_observability(..., backend=self.backend)


═══════════════════════════════════════════════════════════════════════
                        INTEGRATION WITH SYSTEMS
═══════════════════════════════════════════════════════════════════════

ContinuousSystemBase / DiscreteSystemBase
│
├─> @property control(self) -> ControlSynthesis
│   └─> if not hasattr(self, '_control_synthesis'):
│       └─> self._control_synthesis = ControlSynthesis(self._default_backend)
│   └─> return self._control_synthesis
│
└─> @property analysis(self) -> SystemAnalysis
    └─> if not hasattr(self, '_system_analysis'):
        └─> self._system_analysis = SystemAnalysis(self._default_backend)
    └─> return self._system_analysis

Usage:
    system = Pendulum()
    system.set_default_backend('torch')
    
    # Automatic backend handling
    result = system.control.design_lqr(A, B, Q, R)
    # ↑ Uses 'torch' backend automatically
    
    stability = system.analysis.stability(A)
    # ↑ Uses 'torch' backend automatically


═══════════════════════════════════════════════════════════════════════
                        ALGORITHM FLOW
═══════════════════════════════════════════════════════════════════════

LQR Design Flow:
┌─────────────────────────────────────────────────────────────────┐
│ 1. User calls: system.control.design_lqr(A, B, Q, R)           │
│    ↓                                                             │
│ 2. ControlSynthesis routes to: design_lqr(..., backend='torch')│
│    ↓                                                             │
│ 3. design_lqr() converts: A, B, Q, R  →  NumPy                 │
│    ↓                                                             │
│ 4. Solve Riccati: solve_continuous_are(A_np, B_np, Q_np, R_np) │
│    ↓                                                             │
│ 5. Compute gain: K = R⁻¹B'P                                    │
│    ↓                                                             │
│ 6. Compute eigenvalues: eig(A - BK)                            │
│    ↓                                                             │
│ 7. Convert back: K_np, P_np  →  torch.Tensor                   │
│    ↓                                                             │
│ 8. Return: LQRResult with all fields                           │
└─────────────────────────────────────────────────────────────────┘

LQG Design Flow:
┌─────────────────────────────────────────────────────────────────┐
│ 1. User calls: system.control.design_lqg(A, B, C, ...)         │
│    ↓                                                             │
│ 2. design_lqg() internally calls:                              │
│    ├─> lqr_result = design_lqr(A, B, Q_state, R_control)       │
│    └─> kalman_result = design_kalman_filter(A, C, Q_proc, R_m) │
│    ↓                                                             │
│ 3. Extract gains:                                               │
│    ├─> K = lqr_result['gain']                                  │
│    └─> L = kalman_result['gain']                               │
│    ↓                                                             │
│ 4. Verify separation principle:                                │
│    └─> closed_loop_stable = all(eig(A-BK) ∪ eig(A-LC) stable) │
│    ↓                                                             │
│ 5. Return: LQGResult with combined information                 │
└─────────────────────────────────────────────────────────────────┘


═══════════════════════════════════════════════════════════════════════
                        FILE SIZE SUMMARY
═══════════════════════════════════════════════════════════════════════

┌──────────────────────────────┬────────┬──────────────────────────┐
│ Module                       │ Lines  │ Purpose                  │
├──────────────────────────────┼────────┼──────────────────────────┤
│ PURE FUNCTION LAYER          │        │                          │
├──────────────────────────────┼────────┼──────────────────────────┤
│ classical_control_functions  │   967  │ Stateless algorithms     │
├──────────────────────────────┼────────┼──────────────────────────┤
│ WRAPPER LAYER                │        │                          │
├──────────────────────────────┼────────┼──────────────────────────┤
│ control_synthesis            │   388  │ Control design wrapper   │
│ system_analysis              │   431  │ System analysis wrapper  │
├──────────────────────────────┼────────┼──────────────────────────┤
│ TYPE LAYER                   │        │                          │
├──────────────────────────────┼────────┼──────────────────────────┤
│ control_classical            │   542  │ TypedDict results        │
├──────────────────────────────┼────────┼──────────────────────────┤
│ TOTAL                        │ 2,328  │ Complete framework       │
└──────────────────────────────┴────────┴──────────────────────────┘


═══════════════════════════════════════════════════════════════════════
                        DESIGN PHILOSOPHY
═══════════════════════════════════════════════════════════════════════

✓ PURE FUNCTIONS
  Stateless algorithms like scipy
  Easy to test, easy to reason about

✓ THIN WRAPPERS
  Minimal composition layer
  No business logic in wrappers

✓ COMPOSITION OVER INHERITANCE
  Systems use utilities via properties
  Not inherited methods

✓ TYPE SAFETY
  All results are TypedDict
  IDE autocomplete support

✓ BACKEND AGNOSTIC
  Internal conversion to/from NumPy
  Transparent to user

✓ SEPARATION OF CONCERNS
  Analysis ≠ Synthesis
  Clear boundaries

✓ MATHEMATICAL RIGOR
  Correct classical control theory
  Verified against textbooks

✓ SCIPY-LIKE API
  Familiar to control engineers
  Takes matrices, returns dicts


═══════════════════════════════════════════════════════════════════════
                        KEY ALGORITHMS
═══════════════════════════════════════════════════════════════════════

Continuous-Time LQR:
    Minimize: J = ∫₀^∞ (x'Qx + u'Ru) dt
    Riccati:  A'P + PA - PBR⁻¹B'P + Q = 0
    Gain:     K = R⁻¹B'P
    Control:  u = -Kx

Discrete-Time LQR:
    Minimize: J = Σₖ (x'Qx + u'Ru)
    Riccati:  P = A'PA - A'PB(R+B'PB)⁻¹B'PA + Q
    Gain:     K = (R+B'PB)⁻¹B'PA
    Control:  u[k] = -Kx[k]

Kalman Filter:
    System:   x[k+1] = Ax[k] + Bu[k] + w[k]
              y[k] = Cx[k] + v[k]
    Gain:     L = APC'(CPC'+R)⁻¹
    Estimate: x̂[k+1] = Ax̂[k] + Bu[k] + L(y[k] - Cx̂[k])

LQG Controller:
    Controller: u = -Kx̂
    Estimator:  x̂[k+1] = Ax̂[k] + Bu[k] + L(y[k] - Cx̂[k])
    Separation: Design K and L independently
    Poles:      eig(A-BK) ∪ eig(A-LC)


═══════════════════════════════════════════════════════════════════════
```

## Summary

**Total Lines:** 2,328 (framework) + 542 (types) = **2,870 lines**

**Architecture:** Pure functions + thin wrappers + composition

**Core Functions:** 6 (3 design + 3 analysis)

**Integration:** Via `system.control` and `system.analysis` properties

**Philosophy:** Functional design, type safety, backend agnosticism

**Result:** Production-ready classical control theory framework!
