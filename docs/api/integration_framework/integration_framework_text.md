# Numerical Integration Framework Architecture

```
═══════════════════════════════════════════════════════════════════════
                    INTEGRATION FRAMEWORK ARCHITECTURE
═══════════════════════════════════════════════════════════════════════

                          ┌──────────────────────┐
                          │  IntegratorBase      │
                          │  (Abstract Base)     │
                          │  512 lines           │
                          └──────────┬───────────┘
                                     │
                     ┌───────────────┼───────────────┐
                     │                               │
         ┌───────────▼───────────┐      ┌───────────▼───────────┐
         │  IntegratorFactory    │      │  SDEIntegratorBase    │
         │  (Creates ODE)        │      │  (SDE Extension)      │
         │  1,267 lines          │      │  1,080 lines          │
         └───────────┬───────────┘      └───────────┬───────────┘
                     │                               │
         ┌───────────┴────────────┐      ┌───────────▼───────────┐
         │                        │      │  SDEIntegratorFactory │
         ▼                        │      │  (Creates SDE)        │
  ┌─────────────┐                │      │  1,000 lines          │
  │ Scipy       │                │      └───────────┬───────────┘
  │ Integrator  │                │                  │
  │ 620 lines   │                │      ┌───────────┴────────────┐
  └─────────────┘                │      │                        │
                                 ▼      ▼                        ▼
  ┌─────────────┐         ┌─────────────┐         ┌─────────────┐
  │ TorchDiffEq │         │ TorchSDE    │         │ DiffraxSDE  │
  │ Integrator  │         │ Integrator  │         │ Integrator  │
  │ 800 lines   │         │ 800 lines   │         │ 750 lines   │
  └─────────────┘         └─────────────┘         └─────────────┘
         │                        ▲                        ▲
  ┌─────────────┐                │                        │
  │ Diffrax     │         ┌─────────────┐         ┌─────────────┐
  │ Integrator  │         │ DiffEqPySDE │         │ Custom      │
  │ 700 lines   │         │ Integrator  │         │ Brownian    │
  └─────────────┘         │ 850 lines   │         │ 160 lines   │
         │                └─────────────┘         └─────────────┘
  ┌─────────────┐
  │ DiffEqPy    │
  │ Integrator  │
  │ 900 lines   │
  └─────────────┘
         │
  ┌─────────────┐
  │ FixedStep   │
  │ Integrators │
  │ 600 lines   │
  └─────────────┘


═══════════════════════════════════════════════════════════════════════
                        TRACK BREAKDOWN
═══════════════════════════════════════════════════════════════════════

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ TRACK 1: DETERMINISTIC ODE INTEGRATION                            ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

IntegratorBase (512 lines)
│
├─> IntegratorFactory (1,267 lines) ────> Creates integrators
│   │
│   ├─> ScipyIntegrator (620 lines)
│   │   └─> Methods: RK45, RK23, DOP853, Radau, BDF, LSODA
│   │
│   ├─> TorchDiffEqIntegrator (800 lines)
│   │   └─> Methods: dopri5, dopri8, adaptive_heun, bosh3
│   │
│   ├─> DiffraxIntegrator (700 lines)
│   │   └─> Methods: tsit5, dopri5, dopri8, heun, ralston
│   │
│   ├─> DiffEqPyIntegrator (900 lines)
│   │   └─> Methods: Tsit5, Vern9, Rodas5, AutoTsit5(...)
│   │
│   └─> FixedStepIntegrators (600 lines)
│       └─> Methods: euler, midpoint, rk4


┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ TRACK 2: STOCHASTIC SDE INTEGRATION                               ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

SDEIntegratorBase (1,080 lines) ─extends─> IntegratorBase
│
├─> SDEIntegratorFactory (1,000 lines) ────> Creates SDE integrators
│   │
│   ├─> TorchSDEIntegrator (800 lines)
│   │   └─> Methods: euler, heun, srk, reversible_heun
│   │
│   ├─> DiffraxSDEIntegrator (750 lines)
│   │   └─> Methods: euler, heun, reversible_heun
│   │   └─> Uses: CustomBrownianPath (160 lines)
│   │
│   └─> DiffEqPySDEIntegrator (850 lines)
│       └─> Methods: Euler-Maruyama, Milstein, etc.


═══════════════════════════════════════════════════════════════════════
                        BACKEND SUPPORT
═══════════════════════════════════════════════════════════════════════

┌─────────────┬──────────────────────────┬────────────────────────┐
│ Backend     │ ODE Integrators          │ SDE Integrators        │
├─────────────┼──────────────────────────┼────────────────────────┤
│ NumPy       │ • ScipyIntegrator        │ (Limited)              │
│             │ • DiffEqPyIntegrator     │ • DiffEqPySDEIntegrator│
│             │ • FixedStepIntegrators   │                        │
├─────────────┼──────────────────────────┼────────────────────────┤
│ PyTorch     │ • TorchDiffEqIntegrator  │ • TorchSDEIntegrator   │
│             │ • FixedStepIntegrators   │                        │
├─────────────┼──────────────────────────┼────────────────────────┤
│ JAX         │ • DiffraxIntegrator      │ • DiffraxSDEIntegrator │
│             │ • FixedStepIntegrators   │                        │
└─────────────┴──────────────────────────┴────────────────────────┘


═══════════════════════════════════════════════════════════════════════
                        METHOD CATEGORIES
═══════════════════════════════════════════════════════════════════════

ADAPTIVE (ODE)                    FIXED-STEP (ODE)
├─ Non-Stiff                     ├─ euler
│  • RK45 (scipy)                │  • Order 1
│  • Tsit5 (Julia)               │  • All backends
│  • dopri5 (PyTorch/JAX)        ├─ midpoint
├─ Stiff                         │  • Order 2
│  • BDF (scipy)                 │  • All backends
│  • Radau (scipy)               └─ rk4
│  • Rodas5 (Julia)                 • Order 4
└─ Auto-Stiffness                   • All backends
   • LSODA (scipy)
   • AutoTsit5 (Julia)          STOCHASTIC (SDE)
                                 ├─ Euler-Maruyama
HIGH ACCURACY                    │  • Strong order 0.5
├─ Vern9 (Julia)                 │  • All noise types
│  • 9th order                   ├─ Heun
├─ DOP853 (scipy)                │  • Strong order 1.0
│  • 8th order                   │  • Additive noise
└─ dopri8 (PyTorch/JAX)          ├─ Milstein
   • 8th order                   │  • Strong order 1.0
                                 │  • Diagonal noise
                                 └─ SRK
                                    • Various orders


═══════════════════════════════════════════════════════════════════════
                        RESULT TYPES
═══════════════════════════════════════════════════════════════════════

IntegrationResult (ODE):         SDEIntegrationResult (SDE):
├─ t: Time points (T,)           ├─ (All ODE fields)
├─ x: States (T, nx)             ├─ diffusion_evals: int
├─ success: bool                 ├─ noise_samples: array
├─ message: str                  ├─ n_paths: int
├─ nfev: int                     ├─ convergence_type: str
├─ nsteps: int                   ├─ sde_type: str
├─ integration_time: float       └─ noise_type: str
├─ solver: str
└─ (Optional: njev, nlu, sol)


═══════════════════════════════════════════════════════════════════════
                        USAGE FLOW
═══════════════════════════════════════════════════════════════════════

1. CREATE SYSTEM
   └─> system = MySystem()

2. CREATE INTEGRATOR
   ├─> IntegratorFactory.auto(system)              [Automatic]
   ├─> IntegratorFactory.for_production(system)    [LSODA/AutoTsit5]
   ├─> IntegratorFactory.for_optimization(system)  [JAX tsit5]
   └─> IntegratorFactory.create(system, ...)       [Custom]

3. INTEGRATE
   └─> result = integrator.integrate(x0, u_func, t_span)

4. ANALYZE RESULT
   ├─> result['t']          # Time points
   ├─> result['x']          # State trajectory
   ├─> result['success']    # Success flag
   └─> result['solver']     # Method used


═══════════════════════════════════════════════════════════════════════
```

## Key Design Features

1. **Factory Pattern** - Automatic integrator selection
2. **Multi-Backend** - NumPy, PyTorch, JAX seamlessly
3. **Dual Track** - Separate ODE and SDE implementations
4. **TypedDict Results** - Type-safe, IDE-friendly
5. **Performance** - GPU, XLA, Julia for speed
6. **Flexibility** - 40+ methods across backends
7. **Monte Carlo** - Built-in multi-trajectory simulation

## Total Line Count: ~10,000 lines

**Deterministic:** ~5,400 lines
**Stochastic:** ~4,640 lines
