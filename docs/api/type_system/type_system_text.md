# Type System Architecture (Text Diagram)

```
═══════════════════════════════════════════════════════════════════════
                      TYPE SYSTEM ARCHITECTURE
═══════════════════════════════════════════════════════════════════════

                    ┌──────────────────────────┐
                    │   APPLICATION LAYER      │
                    │                          │
                    │  • UI Framework          │
                    │  • Delegation Layer      │
                    │  • Integration Framework │
                    └──────────┬───────────────┘
                               │
                               │ uses types from
                               │
                ┌──────────────┴───────────────┐
                ↓                              ↓
┏━━━━━━━━━━━━━━━━━━━━━━━━━┓      ┏━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ FOUNDATIONAL TYPES      ┃      ┃ DOMAIN TYPES            ┃
┣━━━━━━━━━━━━━━━━━━━━━━━━━┫      ┣━━━━━━━━━━━━━━━━━━━━━━━━━┫
┃ core.py (1,501 lines)   ┃      ┃ trajectories.py (879)   ┃
┃ ├─ Multi-Backend Arrays ┃      ┃ ├─ StateTrajectory      ┃
┃ │  • ArrayLike          ┃      ┃ │  • ControlSequence    ┃
┃ │  • NumpyArray         ┃      ┃ │  • OutputSequence     ┃
┃ │  • TorchTensor        ┃      ┃ │  • TimePoints         ┃
┃ │  • JaxArray           ┃      ┃ └─ IntegrationResult    ┃
┃ │  • ScalarLike         ┃      ┃    SDEIntegrationResult ┃
┃ ├─ Semantic Vectors     ┃      ┣━━━━━━━━━━━━━━━━━━━━━━━━━┫
┃ │  • StateVector        ┃      ┃ linearization.py (502)  ┃
┃ │  • ControlVector      ┃      ┃ ├─ DeterministicLin     ┃
┃ │  • OutputVector       ┃      ┃ │  • StochasticLin      ┃
┃ │  • NoiseVector        ┃      ┃ │  • LinearizationResult┃
┃ │  • EquilibriumState   ┃      ┃ ├─ ObservationLin       ┃
┃ ├─ Matrix Types         ┃      ┃ └─ Jacobians           ┃
┃ │  • StateMatrix        ┃      ┃    • StateJacobian     ┃
┃ │  • InputMatrix        ┃      ┃    • ControlJacobian   ┃
┃ │  • OutputMatrix       ┃      ┃    • OutputJacobian    ┃
┃ │  • GainMatrix         ┃      ┣━━━━━━━━━━━━━━━━━━━━━━━━━┫
┃ │  • CovarianceMatrix   ┃      ┃ symbolic.py (646)       ┃
┃ └─ Function Signatures  ┃      ┃ ├─ SymbolicVariable    ┃
┃    • DynamicsFunction   ┃      ┃ │  • SymbolicVector     ┃
┃    • OutputFunction     ┃      ┃ │  • SymbolicExpression ┃
┃    • ControlPolicy      ┃      ┃ ├─ DynamicsExpression  ┃
┣━━━━━━━━━━━━━━━━━━━━━━━━━┫      ┃ │  • OutputExpression   ┃
┃ backends.py (735 lines) ┃      ┃ │  • DiffusionExpression┃
┃ ├─ Backend Types        ┃      ┃ └─ ParameterDict       ┃
┃ │  • Backend            ┃      ┣━━━━━━━━━━━━━━━━━━━━━━━━━┫
┃ │  • Device             ┃      ┃ control_classical.py    ┃
┃ │  • BackendConfig      ┃      ┃ ├─ Analysis Results     ┃
┃ ├─ Integration Methods  ┃      ┃ │  • StabilityInfo      ┃
┃ │  • IntegrationMethod  ┃      ┃ │  • ControllabilityInfo┃
┃ │  • OdeMethod          ┃      ┃ │  • ObservabilityInfo  ┃
┃ │  • SdeMethod          ┃      ┃ ├─ Control Design       ┃
┃ ├─ Noise Types          ┃      ┃ │  • LQRResult          ┃
┃ │  • NoiseType          ┃      ┃ │  • KalmanFilterResult ┃
┃ │  • SDEType            ┃      ┃ │  • LQGResult          ┃
┃ │  • ConvergenceType    ┃      ┃ └─ Other Controllers    ┃
┃ └─ Configuration        ┃      ┃    • PolePlacementResult┃
┃    • SystemConfig       ┃      ┃    • LuenbergerObserver ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━┛      ┗━━━━━━━━━━━━━━━━━━━━━━━━━┛
                                 ┣━━━━━━━━━━━━━━━━━━━━━━━━━┫
                                 ┃ utilities.py (1,132)    ┃
                                 ┃ ├─ Type Guards          ┃
                                 ┃ │  • is_numpy()         ┃
                                 ┃ │  • is_torch()         ┃
                                 ┃ │  • is_jax()           ┃
                                 ┃ ├─ Shape Utilities      ┃
                                 ┃ │  • is_batched()       ┃
                                 ┃ │  • get_batch_size()   ┃
                                 ┃ ├─ Performance Types    ┃
                                 ┃ │  • ExecutionStats     ┃
                                 ┃ └─ Validation Types     ┃
                                 ┃    • ValidationResult   ┃
                                 ┗━━━━━━━━━━━━━━━━━━━━━━━━━┛


═══════════════════════════════════════════════════════════════════════
                        TYPE CATEGORIES
═══════════════════════════════════════════════════════════════════════

┌────────────────────────┬────────┬──────────────────────────────────┐
│ Category               │ Count  │ Examples                         │
├────────────────────────┼────────┼──────────────────────────────────┤
│ VECTOR TYPES           │  15+   │ StateVector, ControlVector,      │
│                        │        │ NoiseVector, EquilibriumState    │
├────────────────────────┼────────┼──────────────────────────────────┤
│ MATRIX TYPES           │  30+   │ StateMatrix, GainMatrix,         │
│                        │        │ CovarianceMatrix, DiffusionMatrix│
├────────────────────────┼────────┼──────────────────────────────────┤
│ FUNCTION TYPES         │  10+   │ DynamicsFunction, ControlPolicy, │
│                        │        │ ObserverFunction                 │
├────────────────────────┼────────┼──────────────────────────────────┤
│ BACKEND TYPES          │  20+   │ Backend, Device, IntegrationMethod│
│                        │        │ NoiseType, SDEType               │
├────────────────────────┼────────┼──────────────────────────────────┤
│ TRAJECTORY TYPES       │  15+   │ StateTrajectory, TimePoints,     │
│                        │        │ IntegrationResult                │
├────────────────────────┼────────┼──────────────────────────────────┤
│ LINEARIZATION TYPES    │  15+   │ DeterministicLinearization,      │
│                        │        │ StateJacobian, LinearizationResult│
├────────────────────────┼────────┼──────────────────────────────────┤
│ CONTROL TYPES          │  8     │ LQRResult, KalmanFilterResult,   │
│                        │        │ LQGResult, StabilityInfo         │
├────────────────────────┼────────┼──────────────────────────────────┤
│ SYMBOLIC TYPES         │  10+   │ SymbolicExpression, ParameterDict│
│                        │        │ DynamicsExpression               │
├────────────────────────┼────────┼──────────────────────────────────┤
│ PROTOCOL TYPES         │  20+   │ DynamicalSystemProtocol,         │
│                        │        │ ContinuousSystemProtocol         │
├────────────────────────┼────────┼──────────────────────────────────┤
│ UTILITY TYPES          │  20+   │ ExecutionStats, ValidationResult,│
│                        │        │ Type guards, shape utilities     │
├────────────────────────┼────────┼──────────────────────────────────┤
│ TYPEDDICT RESULTS      │  20+   │ IntegrationResult,               │
│                        │        │ SDEIntegrationResult,            │
│                        │        │ LQRResult, LQGResult             │
├────────────────────────┼────────┼──────────────────────────────────┤
│ TOTAL TYPES            │ 200+   │                                  │
└────────────────────────┴────────┴──────────────────────────────────┘


═══════════════════════════════════════════════════════════════════════
                        DESIGN PRINCIPLES
═══════════════════════════════════════════════════════════════════════

1. SEMANTIC OVER STRUCTURAL
   ┌────────────────────────────────────────┐
   │ Bad:  arr1: np.ndarray                 │
   │ Good: x: StateVector                   │
   │                                        │
   │ Name conveys MEANING not just TYPE     │
   └────────────────────────────────────────┘

2. BACKEND AGNOSTICISM
   ┌────────────────────────────────────────┐
   │ ArrayLike = Union[                     │
   │   np.ndarray,                          │
   │   torch.Tensor,                        │
   │   jnp.ndarray                          │
   │ ]                                      │
   │                                        │
   │ Same types work across all backends    │
   └────────────────────────────────────────┘

3. TYPEDDICT FOR STRUCTURES
   ┌────────────────────────────────────────┐
   │ Bad:  def f() -> dict                  │
   │ Good: def f() -> IntegrationResult     │
   │                                        │
   │ class IntegrationResult(TypedDict):    │
   │     t: TimePoints                      │
   │     x: StateTrajectory                 │
   │     success: bool                      │
   │                                        │
   │ Type-safe, IDE-friendly, documented    │
   └────────────────────────────────────────┘

4. OPTIONAL FIELDS VIA total=False
   ┌────────────────────────────────────────┐
   │ class IntegrationResult(               │
   │     TypedDict, total=False             │
   │ ):                                     │
   │     # Required                         │
   │     t: TimePoints                      │
   │     success: bool                      │
   │                                        │
   │     # Optional                         │
   │     njev: int  # Adaptive only         │
   │     sol: Any   # Dense output          │
   └────────────────────────────────────────┘

5. PROTOCOLS NOT INHERITANCE
   ┌────────────────────────────────────────┐
   │ class System:  # No inheritance!       │
   │     def __call__(self, x, u): ...      │
   │                                        │
   │ # Satisfies protocol structurally:     │
   │ system: DynamicalSystemProtocol = ...  │
   │                                        │
   │ Structural subtyping (duck typing)     │
   └────────────────────────────────────────┘

6. POLYMORPHIC VIA UNION
   ┌────────────────────────────────────────┐
   │ LinearizationResult = Union[           │
   │     Tuple[A, B],        # Deterministic│
   │     Tuple[A, B, G]      # Stochastic   │
   │ ]                                      │
   │                                        │
   │ result = system.linearize(...)         │
   │ A, B = result[0], result[1]            │
   │ if len(result) == 3:                   │
   │     G = result[2]  # Stochastic        │
   └────────────────────────────────────────┘


═══════════════════════════════════════════════════════════════════════
                        USAGE FLOW
═══════════════════════════════════════════════════════════════════════

1. IMPORT TYPES
   from src.types.core import StateVector, ControlVector
   from src.types.backends import Backend
   from src.types.trajectories import IntegrationResult

2. TYPE ANNOTATIONS
   def dynamics(x: StateVector, u: ControlVector) -> StateVector:
       return f(x, u)

3. STRUCTURED RESULTS
   result: IntegrationResult = integrator.integrate(x0, u_func, t_span)

4. TYPE-SAFE ACCESS
   t: TimePoints = result['t']
   x: StateTrajectory = result['x']
   success: bool = result['success']

5. IDE AUTOCOMPLETE
   result['  # IDE shows: t, x, success, nfev, ...

6. STATIC TYPE CHECKING
   $ mypy src/
   Success: no issues found in 50 source files


═══════════════════════════════════════════════════════════════════════
                        FILE STATISTICS
═══════════════════════════════════════════════════════════════════════

┌─────────────────────┬─────────┬────────────────────────────────┐
│ Module              │ Lines   │ Purpose                        │
├─────────────────────┼─────────┼────────────────────────────────┤
│ FOUNDATIONAL        │         │                                │
├─────────────────────┼─────────┼────────────────────────────────┤
│ core.py             │  1,501  │ Vectors, matrices, functions   │
│ backends.py         │    735  │ Backends, configs, methods     │
├─────────────────────┼─────────┼────────────────────────────────┤
│ Subtotal            │  2,236  │                                │
├─────────────────────┼─────────┼────────────────────────────────┤
│ DOMAIN              │         │                                │
├─────────────────────┼─────────┼────────────────────────────────┤
│ trajectories.py     │    879  │ Time series, results           │
│ linearization.py    │    502  │ Jacobians, tuples              │
│ symbolic.py         │    646  │ SymPy types                    │
│ control_classical.py│    542  │ Control design results         │
├─────────────────────┼─────────┼────────────────────────────────┤
│ Subtotal            │  2,569  │                                │
├─────────────────────┼─────────┼────────────────────────────────┤
│ STRUCTURAL          │         │                                │
├─────────────────────┼─────────┼────────────────────────────────┤
│ protocols.py        │  1,086  │ Abstract interfaces            │
│ utilities.py        │  1,132  │ Helpers, guards                │
├─────────────────────┼─────────┼────────────────────────────────┤
│ Subtotal            │  2,218  │                                │
├─────────────────────┼─────────┼────────────────────────────────┤
│ TOTAL               │  7,023  │ 200+ types                     │
└─────────────────────┴─────────┴────────────────────────────────┘


═══════════════════════════════════════════════════════════════════════
                        KEY BENEFITS
═══════════════════════════════════════════════════════════════════════

✓ SEMANTIC CLARITY
  Names encode mathematical meaning

✓ TYPE SAFETY
  Static checking prevents errors

✓ IDE SUPPORT
  Autocomplete and inline docs

✓ BACKEND AGNOSTIC
  NumPy/PyTorch/JAX seamless

✓ STRUCTURED RESULTS
  TypedDict not plain dict

✓ SELF-DOCUMENTING
  Types encode constraints

✓ COMPOSITION
  Types compose naturally

✓ EXTENSIBLE
  Easy to add new types

✓ CONSISTENT
  Same patterns throughout

✓ TESTABLE
  Type-driven testing


═══════════════════════════════════════════════════════════════════════
```

## Type System Summary

**Foundation:** 7,023 lines defining 200+ types

**Philosophy:** Type-driven design for clarity and safety

**Architecture:** 3 layers (Foundational, Domain, Structural)

**Impact:** Enables clean, type-safe code throughout framework

**Result:** Self-documenting, IDE-friendly, statically checkable codebase
