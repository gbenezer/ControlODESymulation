# Delegation Layer Architecture (Text Diagram)

```
═══════════════════════════════════════════════════════════════════════
                    DELEGATION LAYER ARCHITECTURE
═══════════════════════════════════════════════════════════════════════

                         UI FRAMEWORK SYSTEMS
                    ┌──────────────────────────────┐
                    │  SymbolicSystemBase          │
                    │  ContinuousSymbolicSystem    │
                    │  DiscreteSymbolicSystem      │
                    │  StochasticSystems           │
                    └──────────┬───────────────────┘
                               │
                               │ COMPOSES (not inherits!)
                               │
                ┌──────────────┴───────────────┐
                ↓                              ↓
┏━━━━━━━━━━━━━━━━━━━━━━━━━┓      ┏━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃   CORE UTILITIES        ┃      ┃ DETERMINISTIC EVAL      ┃
┃   (Universal)           ┃      ┃ (ODE Systems)           ┃
┣━━━━━━━━━━━━━━━━━━━━━━━━━┫      ┣━━━━━━━━━━━━━━━━━━━━━━━━━┫
┃ BackendManager          ┃      ┃ DynamicsEvaluator       ┃
┃ • Multi-backend support ┃      ┃ • Forward dynamics      ┃
┃ • Device management     ┃      ┃ • dx/dt = f(x, u)       ┃
┃ • Array conversion      ┃      ┃ • Batched evaluation    ┃
┃   545 lines              ┃      ┃   576 lines             ┃
┣━━━━━━━━━━━━━━━━━━━━━━━━━┫      ┣━━━━━━━━━━━━━━━━━━━━━━━━━┫
┃ CodeGenerator           ┃      ┃ LinearizationEngine     ┃
┃ • Symbolic → numerical  ┃      ┃ • Jacobian computation  ┃
┃ • Function caching      ┃      ┃ • A = ∂f/∂x, B = ∂f/∂u  ┃
┃ • Multi-backend compile ┃      ┃ • Higher-order systems  ┃
┃   565 lines              ┃      ┃   907 lines             ┃
┣━━━━━━━━━━━━━━━━━━━━━━━━━┫      ┣━━━━━━━━━━━━━━━━━━━━━━━━━┫
┃ EquilibriumHandler      ┃      ┃ ObservationEngine       ┃
┃ • Named equilibria      ┃      ┃ • Output evaluation     ┃
┃ • Verification          ┃      ┃ • y = h(x)              ┃
┃ • Metadata storage      ┃      ┃ • C = ∂h/∂x             ┃
┃   221 lines              ┃      ┃   628 lines             ┃
┣━━━━━━━━━━━━━━━━━━━━━━━━━┫      ┗━━━━━━━━━━━━━━━━━━━━━━━━━┛
┃ SymbolicValidator       ┃
┃ • System validation     ┃
┃ • Dimension checks      ┃
┃ • Error detection       ┃
┃   718 lines              ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━┛

                ↓
┏━━━━━━━━━━━━━━━━━━━━━━━━━┓      ┏━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ STOCHASTIC SUPPORT      ┃      ┃ LOW-LEVEL UTILITIES     ┃
┃ (SDE Systems)           ┃      ┃                         ┃
┣━━━━━━━━━━━━━━━━━━━━━━━━━┫      ┣━━━━━━━━━━━━━━━━━━━━━━━━━┫
┃ DiffusionHandler        ┃      ┃ codegen_utils           ┃
┃ • Diffusion generation  ┃      ┃ • SymPy → code          ┃
┃ • g(x, u) matrix        ┃      ┃ • CSE optimization      ┃
┃ • Noise optimization    ┃      ┃ • Backend-specific      ┃
┃   1069 lines             ┃      ┃   733 lines             ┃
┣━━━━━━━━━━━━━━━━━━━━━━━━━┫      ┗━━━━━━━━━━━━━━━━━━━━━━━━━┛
┃ NoiseCharacterizer      ┃
┃ • Automatic analysis    ┃
┃ • Noise type detection  ┃
┃ • Solver recommendations┃
┃   692 lines              ┃
┣━━━━━━━━━━━━━━━━━━━━━━━━━┫
┃ SDEValidator            ┃
┃ • SDE validation        ┃
┃ • Diffusion checks      ┃
┃ • Type verification     ┃
┃   544 lines              ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━┛

═══════════════════════════════════════════════════════════════════════
                        COMPOSITION PATTERN
═══════════════════════════════════════════════════════════════════════

class ContinuousSymbolicSystem(SymbolicSystemBase, ContinuousSystemBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Core utilities (from SymbolicSystemBase)
        self.backend = BackendManager(...)          ✓
        self._code_gen = CodeGenerator(...)         ✓
        self.equilibria = EquilibriumHandler(...)   ✓
        self._validator = SymbolicValidator()       ✓
        
        # Deterministic evaluators (added here)
        self._dynamics = DynamicsEvaluator(...)     ✓
        self._linearization = LinearizationEngine(...)  ✓
        self._observation = ObservationEngine(...)  ✓


class ContinuousStochasticSystem(ContinuousSymbolicSystem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Inherits all deterministic utilities above
        
        # Add stochastic support
        self.diffusion_handler = DiffusionHandler(...)      ✓
        self.noise_characteristics = NoiseCharacterizer()   ✓
        self._sde_validator = SDEValidator()               ✓


═══════════════════════════════════════════════════════════════════════
                        DEPENDENCY GRAPH
═══════════════════════════════════════════════════════════════════════

┌────────────────┐
│ UI Framework   │
│ Systems        │
└───────┬────────┘
        │
        ├──> BackendManager
        │
        ├──> CodeGenerator ──┬──> BackendManager
        │                    └──> codegen_utils
        │
        ├──> EquilibriumHandler
        │
        ├──> SymbolicValidator
        │
        ├──> DynamicsEvaluator ──┬──> CodeGenerator
        │                        └──> BackendManager
        │
        ├──> LinearizationEngine ──┬──> CodeGenerator
        │                          └──> BackendManager
        │
        ├──> ObservationEngine ──┬──> CodeGenerator
        │                        └──> BackendManager
        │
        └──> (if stochastic)
             ├──> DiffusionHandler ──┬──> CodeGenerator
             │                       └──> BackendManager
             │
             ├──> NoiseCharacterizer
             │
             └──> SDEValidator


═══════════════════════════════════════════════════════════════════════
                        LAYER BREAKDOWN
═══════════════════════════════════════════════════════════════════════

┌─────────────────────────┬────────┬─────────────────────────────────┐
│ Component               │ Lines  │ Purpose                         │
├─────────────────────────┼────────┼─────────────────────────────────┤
│ CORE UTILITIES          │        │                                 │
├─────────────────────────┼────────┼─────────────────────────────────┤
│ BackendManager          │   545  │ Multi-backend array handling    │
│ CodeGenerator           │   565  │ Symbolic → numerical compile    │
│ EquilibriumHandler      │   221  │ Named equilibrium management    │
│ SymbolicValidator       │   718  │ System definition validation    │
├─────────────────────────┼────────┼─────────────────────────────────┤
│ Subtotal                │ 2,049  │                                 │
├─────────────────────────┼────────┼─────────────────────────────────┤
│ DETERMINISTIC EVAL      │        │                                 │
├─────────────────────────┼────────┼─────────────────────────────────┤
│ DynamicsEvaluator       │   576  │ Forward dynamics evaluation     │
│ LinearizationEngine     │   907  │ Jacobian computation            │
│ ObservationEngine       │   628  │ Output evaluation               │
├─────────────────────────┼────────┼─────────────────────────────────┤
│ Subtotal                │ 2,111  │                                 │
├─────────────────────────┼────────┼─────────────────────────────────┤
│ STOCHASTIC SUPPORT      │        │                                 │
├─────────────────────────┼────────┼─────────────────────────────────┤
│ DiffusionHandler        │ 1,069  │ Diffusion matrix generation     │
│ NoiseCharacterizer      │   692  │ Automatic noise analysis        │
│ SDEValidator            │   544  │ SDE-specific validation         │
├─────────────────────────┼────────┼─────────────────────────────────┤
│ Subtotal                │ 2,305  │                                 │
├─────────────────────────┼────────┼─────────────────────────────────┤
│ LOW-LEVEL UTILITIES     │        │                                 │
├─────────────────────────┼────────┼─────────────────────────────────┤
│ codegen_utils           │   733  │ SymPy code generation           │
├─────────────────────────┼────────┼─────────────────────────────────┤
│ Subtotal                │   733  │                                 │
├─────────────────────────┼────────┼─────────────────────────────────┤
│ TOTAL                   │ 7,198  │                                 │
└─────────────────────────┴────────┴─────────────────────────────────┘


═══════════════════════════════════════════════════════════════════════
                        KEY RESPONSIBILITIES
═══════════════════════════════════════════════════════════════════════

BackendManager:
    ✓ Detect backend from array type
    ✓ Convert arrays between backends
    ✓ Manage devices (CPU, CUDA, TPU)
    ✓ Backend availability checking

CodeGenerator:
    ✓ Generate f(x, u) → dx/dt functions
    ✓ Generate h(x) → y functions
    ✓ Generate Jacobians: A, B, C
    ✓ Per-backend caching
    ✓ Symbolic optimization

EquilibriumHandler:
    ✓ Store named equilibria
    ✓ Backend-neutral storage
    ✓ On-demand conversion
    ✓ Verification with tolerance
    ✓ Metadata management

SymbolicValidator:
    ✓ Validate state/control variables
    ✓ Check symbolic dimensions
    ✓ Verify parameter types
    ✓ Ensure system order compatibility
    ✓ Clear error messages

DynamicsEvaluator:
    ✓ Evaluate dx/dt = f(x, u)
    ✓ Handle autonomous systems (u=None)
    ✓ Batched evaluation
    ✓ Performance tracking
    ✓ Backend dispatch

LinearizationEngine:
    ✓ Compute A = ∂f/∂x, B = ∂f/∂u
    ✓ Continuous and discrete Jacobians
    ✓ Higher-order state-space construction
    ✓ Symbolic and numerical modes
    ✓ Backend-agnostic

ObservationEngine:
    ✓ Evaluate y = h(x)
    ✓ Compute C = ∂h/∂x
    ✓ Batched output evaluation
    ✓ Multi-backend support

DiffusionHandler:
    ✓ Generate g(x, u) diffusion matrix
    ✓ Noise structure optimization
    ✓ Stratonovich corrections
    ✓ Additive/diagonal/scalar specialization

NoiseCharacterizer:
    ✓ Automatic noise type detection
    ✓ Structure analysis (additive, diagonal, etc.)
    ✓ Solver recommendations per backend
    ✓ Noise statistics

SDEValidator:
    ✓ Validate diffusion dimensions
    ✓ Check SDE type (Itô vs Stratonovich)
    ✓ Verify compatibility with drift
    ✓ Ensure finite expressions

codegen_utils:
    ✓ SymPy Matrix → callable function
    ✓ Common subexpression elimination
    ✓ Backend-specific optimizations
    ✓ NumPy/PyTorch/JAX code generation


═══════════════════════════════════════════════════════════════════════
                        DESIGN BENEFITS
═══════════════════════════════════════════════════════════════════════

✓ SINGLE RESPONSIBILITY
  Each utility does ONE thing well

✓ COMPOSITION OVER INHERITANCE
  Systems compose utilities, don't inherit

✓ REUSABILITY
  Utilities can be used independently

✓ TESTABILITY
  Each component tested in isolation

✓ FLEXIBILITY
  Easy to add new utilities

✓ PERFORMANCE
  Caching and lazy initialization

✓ MULTI-BACKEND
  Seamless NumPy/PyTorch/JAX switching

✓ TYPE SAFETY
  TypedDict and semantic types throughout

✓ MAINTAINABILITY
  Clear separation of concerns

✓ EXTENSIBILITY
  Composition enables new features


═══════════════════════════════════════════════════════════════════════
```

## Summary

**Total:** 7,198 lines across 11 focused utility classes

**Core Philosophy:** Composition over inheritance

**Key Pattern:** Each system composes the utilities it needs rather than inheriting a monolithic base

**Result:** Clean, testable, flexible architecture with clear separation of concerns
