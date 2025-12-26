# Phase 0.2 Dependency Analysis Results

Generated: 2025-12-26

This document maps all files that import the four core system classes being refactored.

---

## Summary Statistics

| Class | Import Count | Source Files | Test Files | Doc Files |
|-------|-------------|--------------|------------|-----------|
| `SymbolicDynamicalSystem` | 28 | 18 | 6 | 1 |
| `StochasticDynamicalSystem` | 20 | 10 | 8 | 0 |
| `DiscreteSymbolicSystem` | 9 | 4 | 4 | 0 |
| `DiscreteStochasticSystem` | 9 | 5 | 3 | 0 |

---

## 1. Files Importing `SymbolicDynamicalSystem`

### Source Files (18 files)

**Core System Files:**
- `src/systems/base/stochastic_dynamical_system.py:214` - Parent class dependency
- `src/systems/base/discrete_symbolic_system.py:27` - Parent class dependency
- `src/systems/base/generic_discrete_time_system.py:29`

**Built-in Systems:**
- `src/systems/builtin/linear_systems.py:36`
- `src/systems/builtin/mechanical_systems.py:20`
- `src/systems/builtin/abstract_symbolic_systems.py:20`
- `src/systems/builtin/aerial_systems.py:20`

**Control Module:**
- `src/control/control_designer.py:22`

**Discretization Module:**
- `src/systems/base/discretization/discretizer.py:108`
- `src/systems/base/discretization/discrete_simulator.py:70`
- `src/systems/base/discretization/discrete_linearization.py:100`

**Numerical Integration Module:**
- `src/systems/base/numerical_integration/fixed_step_integrators.py:43`
- `src/systems/base/numerical_integration/diffeqpy_integrator.py:138`
- `src/systems/base/numerical_integration/scipy_integrator.py:46`
- `src/systems/base/numerical_integration/integrator_factory.py:61`
- `src/systems/base/numerical_integration/integrator_base.py:37`

**Utilities Module:**
- `src/systems/base/utils/observation_engine.py:42`
- `src/systems/base/utils/code_generator.py:39`
- `src/systems/base/utils/dynamics_evaluator.py:43`
- `src/systems/base/utils/symbolic_validator.py:42`
- `src/systems/base/utils/linearization_engine.py:45`

### Test Files (6 files)
- `tests/unit/core_class_unit_tests/symbolic_dynamical_system_test.py:52`
- `tests/unit/discretization_unit_tests/discrete_simulator_test.py:56`
- `tests/unit/discretization_unit_tests/discretizer_test.py:408, 1007`
- `tests/unit/discretization_unit_tests/discrete_linearization_test.py:59`
- `tests/unit/integrator_unit_tests/test_autonomous_system_integration.py:50`
- `tests/unit/integrator_unit_tests/stochastic/sde_integrator_base_test.py:386`

### Documentation (1 file)
- `README.md:57` - Example code showing class usage

---

## 2. Files Importing `StochasticDynamicalSystem`

### Source Files (10 files)

**Core System Files:**
- `src/systems/base/discrete_stochastic_system.py:30` - Parent class dependency

**Built-in Stochastic Systems:**
- `src/systems/builtin/stochastic/ornstein_uhlenbeck.py:109`
- `src/systems/builtin/stochastic/brownian_motion.py:112`
- `src/systems/builtin/stochastic/geometric_brownian_motion.py:111`

**Discretization Module:**
- `src/systems/base/discretization/discrete_linearization.py:189`
- `src/systems/base/discretization/stochastic/stochastic_discrete_linearization.py:194`
- `src/systems/base/discretization/stochastic/stochastic_discretizer.py:108, 262`
- `src/systems/base/discretization/stochastic/stochastic_discrete_simulator.py:76`

**Numerical Integration Module:**
- `src/systems/base/numerical_integration/stochastic/sde_integrator_base.py:69, 313`
- `src/systems/base/numerical_integration/stochastic/sde_integrator_factory.py:61`

### Test Files (8 files)
- `tests/unit/core_class_unit_tests/stochastic_dynamical_system_test.py:60`
- `tests/unit/integrator_unit_tests/stochastic/torchsde_integrator_test.py:54`
- `tests/unit/integrator_unit_tests/stochastic/sde_integrator_factory_test.py:46`
- `tests/unit/integrator_unit_tests/stochastic/sde_integrator_base_test.py:45`
- `tests/unit/integrator_unit_tests/stochastic/diffeqpy_sde_integrator_test.py:65`
- `tests/unit/integrator_unit_tests/stochastic/diffrax_sde_integrator_test.py:60`
- `tests/unit/discretization_unit_tests/stochastic/stochastic_discrete_linearization_test.py:62`
- `tests/unit/discretization_unit_tests/stochastic/stochastic_discretizer_test.py:128`

---

## 3. Files Importing `DiscreteSymbolicSystem`

### Source Files (4 files)
- `src/systems/base/discretization/discrete_linearization.py:98, 166`
- `src/systems/base/discretization/discrete_simulator.py:68, 131, 191`

### Test Files (4 files)
- `tests/unit/core_class_unit_tests/discrete_symbolic_system_test.py:51`
- `tests/unit/discretization_unit_tests/discrete_simulator_test.py:53`
- `tests/unit/discretization_unit_tests/discrete_linearization_test.py:56`
- `tests/unit/discretization_unit_tests/stochastic/stochastic_discrete_simulator_test.py:157`

---

## 4. Files Importing `DiscreteStochasticSystem`

### Source Files (5 files)

**Built-in Discrete Stochastic Systems:**
- `src/systems/builtin/stochastic/discrete_random_walk.py:20`
- `src/systems/builtin/stochastic/discrete_white_noise.py:20`
- `src/systems/builtin/stochastic/discrete_ar1.py:20`

**Discretization Module:**
- `src/systems/base/discretization/discrete_linearization.py:181`
- `src/systems/base/discretization/stochastic/stochastic_discrete_linearization.py:186`
- `src/systems/base/discretization/stochastic/stochastic_discrete_simulator.py:75`

### Test Files (3 files)
- `tests/unit/core_class_unit_tests/discrete_stochastic_system_test.py:53`
- `tests/unit/discretization_unit_tests/stochastic/stochastic_discrete_linearization_test.py:57`
- `tests/unit/discretization_unit_tests/stochastic/stochastic_discrete_simulator_test.py:56`

---

## 5. External Dependencies

### Documentation Files
| File | References |
|------|------------|
| `README.md` | `SymbolicDynamicalSystem`, `DiscreteSymbolicSystem`, `DiscreteStochasticSystem` (example code) |
| `CHANGELOG.md` | All four classes (migration notes) |
| `type_system_integration_details.md` | `DiscreteSymbolicSystem`, `DiscreteStochasticSystem` |

### Examples
- No example files found (`examples/` directory does not exist)

### Jupyter Notebooks
- No notebooks found (`.ipynb` files do not exist)

---

## 6. Dependency Graph

```
SymbolicDynamicalSystem (ROOT)
├── StochasticDynamicalSystem (inherits)
│   └── DiscreteStochasticSystem (inherits)
├── DiscreteSymbolicSystem (inherits)
└── GenericDiscreteTimeSystem (uses)

Downstream Dependencies:
├── Built-in Systems (4 files)
│   ├── linear_systems.py
│   ├── mechanical_systems.py
│   ├── abstract_symbolic_systems.py
│   └── aerial_systems.py
├── Stochastic Built-in Systems (3 continuous + 3 discrete)
├── Discretization Module (7 files)
├── Numerical Integration Module (8 files)
├── Utilities Module (5 files)
├── Control Module (1 file)
└── Test Suite (21 test files)
```

---

## 7. Impact Analysis

### High Impact Files (require careful refactoring)
1. `src/systems/base/stochastic_dynamical_system.py` - Inherits from `SymbolicDynamicalSystem`
2. `src/systems/base/discrete_symbolic_system.py` - Inherits from `SymbolicDynamicalSystem`
3. `src/systems/base/discrete_stochastic_system.py` - Inherits from `StochasticDynamicalSystem`

### Medium Impact Files (type hints and imports)
- All discretization module files
- All numerical integration module files
- All utility module files

### Low Impact Files (simple import updates)
- All built-in system files
- Control designer
- Test files

---

## 8. Recommendations for Phase 2

1. **Start with base class creation (Phase 1)** before touching any imports
2. **Update inheritance chain first**:
   - Create `ContinuousSystemBase`
   - Rename `SymbolicDynamicalSystem` → `ContinuousSymbolicSystem`
   - Update `StochasticDynamicalSystem` → `ContinuousStochasticSystem`
3. **Add backward compatibility aliases immediately** to avoid breaking downstream
4. **Update tests incrementally** - run test suite after each major change
5. **Update README.md last** after all code changes are validated

---

## 9. Checklist Status

- [x] Map all files that import `SymbolicDynamicalSystem`
- [x] Map all files that import `StochasticDynamicalSystem`
- [x] Map all files that import `DiscreteSymbolicSystem`
- [x] Map all files that import `DiscreteStochasticSystem`
- [x] Identify external dependencies (tests, examples, documentation)

**Phase 0.2 Complete**
