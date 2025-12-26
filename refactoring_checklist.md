# Refactoring Implementation Checklist
## Unified Discrete-Time Interface with Consistent Naming

---

## Overview

This checklist provides a step-by-step implementation plan for refactoring the symbolic dynamical systems library. The refactoring introduces:

1. **Consistent naming**: `StochasticDynamicalSystem` → `ContinuousStochasticSystem`
2. **Clear time-domain separation**: `ContinuousSystemBase` and `DiscreteSystemBase`
3. **Type system integration**: Structured types throughout from `src/types/`
4. **Four-layer architecture** for better organization and extensibility

---

## Phase 0: Pre-Implementation Preparation

### 0.1 Repository Backup & Branching
- [ ] Create full repository backup
- [ ] Create feature branch: `refactor/unified-discrete-interface`
- [ ] Document current API for backward compatibility reference
- [ ] Set up CI/CD to run on feature branch

### 0.2 Dependency Analysis
- [ ] Map all files that import `SymbolicDynamicalSystem`
- [ ] Map all files that import `StochasticDynamicalSystem`
- [ ] Map all files that import `DiscreteSymbolicSystem`
- [ ] Map all files that import `DiscreteStochasticSystem`
- [ ] Identify external dependencies (tests, examples, documentation)

### 0.3 Test Baseline
- [ ] Run full test suite and record results
- [ ] Document any existing test failures
- [ ] Create test coverage report
- [ ] Identify tests that will need updates

**Dependencies**: None  
**Estimated Time**: 2-4 hours  
**Risk**: Low

---

## Phase 1: Create Abstract Base Classes (Layer 1)

### 1.1 Create `ContinuousSystemBase`
**File**: `src/systems/base/core/continuous_system_base.py`

- [ ] Define abstract class with ABC
- [ ] Add abstract method: `__call__(self, x, u) -> ArrayLike`
- [ ] Add abstract method: `integrate(self, x0, u, t_span) -> Trajectory`
- [ ] Add abstract method: `linearize(self, x_eq, u_eq) -> LinearizationResult`
- [ ] Add docstrings with examples
- [ ] Add type hints for all methods
- [ ] Create unit test: `tests/core_class_unit_tests/continuous_system_base_test.py`

**Dependencies**: None  
**Estimated Time**: 2-3 hours  
**Risk**: Low

### 1.2 Create `DiscreteSystemBase`
**File**: `src/systems/base/core/discrete_system_base.py`

- [ ] Define abstract class with ABC
- [ ] Add abstract method: `step(self, x, u) -> ArrayLike`
- [ ] Add abstract method: `linearize(self, x_eq, u_eq) -> LinearizationResult`
- [ ] Add abstract property: `dt` (time step)
- [ ] Add docstrings with examples
- [ ] Add type hints for all methods
- [ ] Create unit test: `tests/core_class_unit_tests/discrete_system_base_test.py`

**Dependencies**: None  
**Estimated Time**: 2-3 hours  
**Risk**: Low

### 1.3 Update Core Module Exports
**File**: `src/systems/base/core/__init__.py`

- [ ] Export `ContinuousSystemBase`
- [ ] Export `DiscreteSystemBase`
- [ ] Update module docstring

**Dependencies**: 1.1, 1.2  
**Estimated Time**: 15 minutes  
**Risk**: Low

---

## Phase 2: Rename and Refactor Continuous Systems (Layer 2)

### 2.1 Rename `SymbolicDynamicalSystem` → `ContinuousSymbolicSystem`
**File**: `src/systems/base/core/symbolic_dynamical_system.py` → `continuous_symbolic_system.py`

- [ ] Create new file: `continuous_symbolic_system.py`
- [ ] Copy entire `SymbolicDynamicalSystem` class
- [ ] Rename class to `ContinuousSymbolicSystem`
- [ ] Inherit from `ContinuousSystemBase`
- [ ] Verify all methods satisfy `ContinuousSystemBase` interface
- [ ] Add backward compatibility alias: `SymbolicDynamicalSystem = ContinuousSymbolicSystem`
- [ ] Update all internal docstrings and references
- [ ] Run existing tests with new class name
- [ ] Create deprecation warning for old name

**Dependencies**: 1.1  
**Estimated Time**: 3-4 hours  
**Risk**: Medium (many downstream dependencies)

### 2.2 Rename `StochasticDynamicalSystem` → `ContinuousStochasticSystem`
**File**: `src/systems/base/core/stochastic_dynamical_system.py` → `continuous_stochastic_system.py`

- [ ] Create new file: `continuous_stochastic_system.py`
- [ ] Copy entire `StochasticDynamicalSystem` class
- [ ] Rename class to `ContinuousStochasticSystem`
- [ ] Change parent class to `ContinuousSymbolicSystem`
- [ ] Verify inheritance chain: `ContinuousStochasticSystem` → `ContinuousSymbolicSystem` → `ContinuousSystemBase`
- [ ] Update diffusion-related methods
- [ ] Add backward compatibility alias: `StochasticDynamicalSystem = ContinuousStochasticSystem`
- [ ] Update all internal docstrings and references
- [ ] Run existing SDE tests with new class name
- [ ] Create deprecation warning for old name

**Dependencies**: 2.1  
**Estimated Time**: 3-4 hours  
**Risk**: Medium

### 2.3 Update Continuous System Tests
**Files**: 
- `tests/core_class_unit_tests/symbolic_dynamical_system_test.py`
- `tests/core_class_unit_tests/stochastic_dynamical_system_test.py`

- [ ] Update imports to use new class names
- [ ] Add tests for `ContinuousSystemBase` interface compliance
- [ ] Add tests for backward compatibility aliases
- [ ] Add deprecation warning tests
- [ ] Verify all existing tests pass

**Dependencies**: 2.1, 2.2  
**Estimated Time**: 2-3 hours  
**Risk**: Low

---

## Phase 3: Type System Integration

**⚠️ SEE SEPARATE DOCUMENT: `phase_3_type_system_integration.md` ⚠️**

This phase refactors all existing code to use the types and utilities from `src/types/`:
- Core types (StateVector, ControlVector, ArrayLike, etc.)
- Linearization result types (structured returns instead of tuples)
- Trajectory types (SimulationResult instead of tuple returns)
- Backend types (BackendType, BackendArray)
- Symbolic types (SymbolicMatrix, SymbolicExpression)
- Control, estimation, and optimization types

### 3.1 Core Types Adoption (`src/types/core.py`)
- [ ] Update all system classes to use `StateVector`, `ControlVector`, `OutputVector`
- [ ] Update integrators to use `StateVector`, `TimeArray`
- [ ] Update discretizers to use core types
- [ ] Run mypy validation

**Estimated Time**: 9 hours

### 3.2 Backend Types Adoption (`src/types/backends.py`)
- [ ] Update `backend_manager.py` to use `BackendType`, `BackendArray`
- [ ] Update code generator
- [ ] Run mypy validation

**Estimated Time**: 4 hours

### 3.3 Symbolic Types Adoption (`src/types/symbolic.py`)
- [ ] Update all symbolic system classes
- [ ] Update symbolic utilities
- [ ] Run mypy validation

**Estimated Time**: 5 hours

### 3.4 Linearization Types Adoption (`src/types/linearization.py`)
- [ ] All `linearize()` methods return structured types
- [ ] Update linearization engine
- [ ] Update control modules to accept structured types
- [ ] Add backward compatibility via `__iter__`

**Estimated Time**: 11 hours

### 3.5 Trajectory Types Adoption (`src/types/trajectories.py`)
- [ ] All `integrate()` methods return `SimulationResult`
- [ ] All `simulate()` methods return structured types
- [ ] Update trajectory plotter
- [ ] Add backward compatibility via `__iter__`

**Estimated Time**: 12 hours

### 3.6 Estimation Types Adoption (`src/types/estimation.py`)
- [ ] Update all observer modules
- [ ] Return structured estimation results

**Estimated Time**: 4 hours

### 3.7 Control Types Adoption (`src/types/control_*.py`)
- [ ] Update classical control modules
- [ ] Update advanced control modules
- [ ] Return structured control results

**Estimated Time**: 6 hours

### 3.8 Other Types (Optimization, Learning, Identification)
- [ ] Update MPC internals
- [ ] Update learning modules (if applicable)
- [ ] Update identification modules (if applicable)

**Estimated Time**: 6 hours

### 3.9 Comprehensive Type Checking
- [ ] Set up mypy configuration
- [ ] Run `mypy --strict` on entire codebase
- [ ] Fix all type errors
- [ ] Add type checking to CI/CD

**Estimated Time**: 8 hours

**Phase 3 Total Estimated Time**: ~68 hours  
**Dependencies**: Phase 2  
**Risk**: Medium (introduces breaking changes to return types)

**Key Breaking Changes**:
- `linearize()` returns structured types instead of tuples
- `integrate()` returns `SimulationResult` instead of `(t, x)`
- `simulate()` returns structured result objects

**Backward Compatibility**: Add `__iter__` methods to result classes for tuple unpacking support

---

## Phase 4: Refactor Discrete Systems (Layer 3)

### 4.1 Update `DiscreteSymbolicSystem`
**File**: `src/systems/base/core/discrete_symbolic_system.py`

- [ ] Change inheritance: from `SymbolicDynamicalSystem` to `ContinuousSymbolicSystem`
- [ ] Add `DiscreteSystemBase` as additional parent (multiple inheritance)
- [ ] Implement `step(self, x, u)` method (wrapper for `__call__`)
- [ ] Implement `linearize(self, x_eq, u_eq)` to return discrete (Ad, Bd)
- [ ] Add `dt` property
- [ ] Add `is_stochastic` property returning `False`
- [ ] Verify `_is_discrete = True` flag is set
- [ ] Update docstrings
- [ ] Run existing discrete tests

**Dependencies**: 1.2, 2.1, 3.1, 3.3, 3.4  
**Estimated Time**: 3-4 hours  
**Risk**: Medium

### 4.2 Update `DiscreteStochasticSystem`
**File**: `src/systems/base/core/discrete_stochastic_system.py`

- [ ] Change inheritance: from `StochasticDynamicalSystem` to `ContinuousStochasticSystem`
- [ ] Add `DiscreteSystemBase` as additional parent (multiple inheritance)
- [ ] Implement `step(self, x, u, w)` method for stochastic stepping
- [ ] Implement `linearize(self, x_eq, u_eq)` to return (Ad, Bd, Gd)
- [ ] Add `dt` property
- [ ] Add `is_stochastic` property returning `True`
- [ ] Verify `_is_discrete = True` flag is set
- [ ] Update diffusion handling for discrete time
- [ ] Update docstrings
- [ ] Run existing discrete stochastic tests

**Dependencies**: 1.2, 2.2, 3.1, 3.4  
**Estimated Time**: 4-5 hours  
**Risk**: Medium-High

### 4.3 Update Discrete System Tests
**Files**:
- `tests/core_class_unit_tests/discrete_symbolic_system_test.py`
- `tests/core_class_unit_tests/discrete_stochastic_system_test.py`

- [ ] Update imports if needed
- [ ] Add tests for `DiscreteSystemBase` interface compliance
- [ ] Add tests for `step()` method
- [ ] Add tests for `linearize()` returning discrete matrices
- [ ] Add tests for `dt` property
- [ ] Add tests for `is_stochastic` property
- [ ] Verify all existing tests pass

**Dependencies**: 4.1, 4.2  
**Estimated Time**: 3-4 hours  
**Risk**: Low

---

## Phase 5: Create Discretization Wrapper (Layer 4)

### 5.1 Design `DiscretizationWrapper`
**File**: `src/systems/base/core/discretization_wrapper.py`

- [ ] Create class inheriting from `DiscreteSystemBase`
- [ ] Add `__init__(self, continuous_system, discretizer)`
- [ ] Add type checking to reject `DiscreteSystemBase` inputs
- [ ] Store reference to continuous system
- [ ] Store reference to discretizer
- [ ] Implement `step(self, x, u)` using discretizer
- [ ] Implement `linearize(self, x_eq, u_eq)` for discrete linearization
- [ ] Add `dt` property from discretizer
- [ ] Add `is_stochastic` property based on continuous system
- [ ] Add `continuous_system` property for access
- [ ] Add comprehensive docstrings

**Dependencies**: 1.2, 2.1, 2.2, 3.1, 3.5  
**Estimated Time**: 4-5 hours  
**Risk**: Medium

### 5.2 Integrate with Existing Discretizers
**Files**: 
- `src/systems/base/discretization/discretizer.py`
- `src/systems/base/discretization/stochastic/stochastic_discretizer.py`

- [ ] Update discretizers to work with `DiscretizationWrapper`
- [ ] Ensure discretizers can handle `ContinuousSymbolicSystem`
- [ ] Ensure stochastic discretizers can handle `ContinuousStochasticSystem`
- [ ] Add factory method: `continuous_system.discretize(dt, method)` returns wrapper
- [ ] Update discretizer docstrings

**Dependencies**: 5.1  
**Estimated Time**: 3-4 hours  
**Risk**: Medium

### 5.3 Create Wrapper Tests
**File**: `tests/core_class_unit_tests/discretization_wrapper_test.py`

- [ ] Test wrapper creation with continuous systems
- [ ] Test rejection of discrete systems (TypeError)
- [ ] Test `step()` method
- [ ] Test `linearize()` method
- [ ] Test `dt` property
- [ ] Test `is_stochastic` property
- [ ] Test integration with various discretizers
- [ ] Test stochastic wrapper behavior

**Dependencies**: 5.1, 5.2  
**Estimated Time**: 3-4 hours  
**Risk**: Low

---

## Phase 6: Update Discretization Module

### 6.1 Update Discretizer Classes
**Files**:
- `src/systems/base/discretization/discretizer.py`
- `src/systems/base/discretization/discrete_simulator.py`
- `src/systems/base/discretization/discrete_linearization.py`

- [ ] Update to work with new base classes
- [ ] Accept `ContinuousSystemBase` instead of `SymbolicDynamicalSystem`
- [ ] Update type hints
- [ ] Update docstrings
- [ ] Run discretization tests

**Dependencies**: 2.1, 2.2, 3.4, 3.5  
**Estimated Time**: 3-4 hours  
**Risk**: Medium

### 6.2 Update Stochastic Discretizers
**Files**:
- `src/systems/base/discretization/stochastic/stochastic_discretizer.py`
- `src/systems/base/discretization/stochastic/stochastic_discrete_simulator.py`
- `src/systems/base/discretization/stochastic/stochastic_discrete_linearization.py`

- [ ] Update to work with `ContinuousStochasticSystem`
- [ ] Update type hints
- [ ] Update docstrings
- [ ] Run stochastic discretization tests

**Dependencies**: 2.2, 3.4, 3.5  
**Estimated Time**: 3-4 hours  
**Risk**: Medium

### 6.3 Update Discretization Tests
**Files**: All tests in `tests/discretization_unit_tests/`

- [ ] Update imports to use new class names
- [ ] Update tests to use `ContinuousSymbolicSystem` and `ContinuousStochasticSystem`
- [ ] Add tests for `DiscretizationWrapper` integration
- [ ] Verify all existing tests pass

**Dependencies**: 6.1, 6.2  
**Estimated Time**: 2-3 hours  
**Risk**: Low

---

## Phase 7: Update Numerical Integration Module

### 7.1 Update ODE Integrators
**Files**:
- `src/systems/base/numerical_integration/integrator_base.py`
- `src/systems/base/numerical_integration/scipy_integrator.py`
- `src/systems/base/numerical_integration/torchdiffeq_integrator.py`
- `src/systems/base/numerical_integration/diffeqpy_integrator.py`
- `src/systems/base/numerical_integration/diffrax_integrator.py`
- `src/systems/base/numerical_integration/fixed_step_integrators.py`

- [ ] Update to accept `ContinuousSystemBase`
- [ ] Update type hints
- [ ] Update docstrings
- [ ] Run integrator tests

**Dependencies**: 1.1, 2.1, 3.5  
**Estimated Time**: 2-3 hours  
**Risk**: Low

### 7.2 Update SDE Integrators
**Files**:
- `src/systems/base/numerical_integration/stochastic/sde_integrator_base.py`
- `src/systems/base/numerical_integration/stochastic/torchsde_integrator.py`
- `src/systems/base/numerical_integration/stochastic/diffeqpy_sde_integrator.py`
- `src/systems/base/numerical_integration/stochastic/diffrax_sde_integrator.py`

- [ ] Update to accept `ContinuousStochasticSystem`
- [ ] Update type hints
- [ ] Update docstrings
- [ ] Run SDE integrator tests

**Dependencies**: 1.1, 2.2, 3.5  
**Estimated Time**: 2-3 hours  
**Risk**: Low

### 7.3 Update Integration Tests
**Files**: All tests in `tests/integrator_unit_tests/`

- [ ] Update imports to use new class names
- [ ] Verify all integrator tests pass
- [ ] Add tests for base class interface compliance

**Dependencies**: 7.1, 7.2  
**Estimated Time**: 1-2 hours  
**Risk**: Low

---

## Phase 8: Update Utility Modules

### 8.1 Update Core Utilities
**Files**:
- `src/systems/base/utils/backend_manager.py`
- `src/systems/base/utils/code_generator.py`
- `src/systems/base/utils/dynamics_evaluator.py`
- `src/systems/base/utils/equilibrium_handler.py`
- `src/systems/base/utils/linearization_engine.py`
- `src/systems/base/utils/observation_engine.py`

- [ ] Update type hints to accept new base classes
- [ ] Update docstrings
- [ ] Run utility tests

**Dependencies**: 1.1, 1.2, 2.1, 2.2, 3.2, 3.3  
**Estimated Time**: 2-3 hours  
**Risk**: Low

### 8.2 Update Stochastic Utilities
**Files**:
- `src/systems/base/utils/stochastic/diffusion_handler.py`
- `src/systems/base/utils/stochastic/noise_analysis.py`
- `src/systems/base/utils/stochastic/sde_validator.py`

- [ ] Update to work with `ContinuousStochasticSystem`
- [ ] Update type hints
- [ ] Update docstrings
- [ ] Run stochastic utility tests

**Dependencies**: 2.2, 3.3  
**Estimated Time**: 1-2 hours  
**Risk**: Low

### 8.3 Update Utility Tests
**Files**: All tests in `tests/utils_unit_tests/`

- [ ] Update imports to use new class names
- [ ] Verify all utility tests pass

**Dependencies**: 8.1, 8.2  
**Estimated Time**: 1-2 hours  
**Risk**: Low

---

## Phase 9: Update Built-in Systems Library

### 9.1 Update Continuous Built-in Systems
**Files**:
- `src/systems/builtin/mechanical_systems.py`
- `src/systems/builtin/aerial_systems.py`
- `src/systems/builtin/linear_systems.py`
- `src/systems/builtin/abstract_symbolic_systems.py`

- [ ] Update to inherit from `ContinuousSymbolicSystem`
- [ ] Update imports
- [ ] Update docstrings
- [ ] Run system-specific tests

**Dependencies**: 2.1  
**Estimated Time**: 2-3 hours  
**Risk**: Low

### 9.2 Update Stochastic Built-in Systems
**Files**:
- `src/systems/builtin/stochastic/brownian_motion.py`
- `src/systems/builtin/stochastic/geometric_brownian_motion.py`
- `src/systems/builtin/stochastic/ornstein_uhlenbeck.py`

- [ ] Update to inherit from `ContinuousStochasticSystem`
- [ ] Update imports
- [ ] Update docstrings
- [ ] Run stochastic system tests

**Dependencies**: 2.2  
**Estimated Time**: 1-2 hours  
**Risk**: Low

### 9.3 Update Discrete Built-in Systems
**Files**:
- `src/systems/builtin/stochastic/discrete_ar1.py`
- `src/systems/builtin/stochastic/discrete_random_walk.py`
- `src/systems/builtin/stochastic/discrete_white_noise.py`

- [ ] Update to inherit from `DiscreteSymbolicSystem` or `DiscreteStochasticSystem`
- [ ] Update imports
- [ ] Update docstrings
- [ ] Run discrete system tests

**Dependencies**: 4.1, 4.2  
**Estimated Time**: 1-2 hours  
**Risk**: Low

---

## Phase 10: Update Type Definitions

### 10.1 Update Core Types
**File**: `src/types/core.py`

- [ ] Add type aliases for new base classes
- [ ] Update system type unions
- [ ] Update docstrings

**Dependencies**: 1.1, 1.2, 2.1, 2.2  
**Estimated Time**: 1 hour  
**Risk**: Low

### 10.2 Update Symbolic Types
**File**: `src/types/symbolic.py`

- [ ] Update symbolic system type definitions
- [ ] Add deprecation notices for old type names
- [ ] Update docstrings

**Dependencies**: 2.1, 2.2  
**Estimated Time**: 30 minutes  
**Risk**: Low

### 10.3 Update Linearization Types
**File**: `src/types/linearization.py`

- [ ] Ensure types work with both continuous and discrete linearization
- [ ] Add discrete-specific linearization types if needed
- [ ] Update docstrings

**Dependencies**: 1.1, 1.2, 4.1, 4.2  
**Estimated Time**: 1 hour  
**Risk**: Low

---

## Phase 11: Update Control and Estimation Modules

### 11.1 Update Control Modules
**Files**:
- `src/control/lqr.py`
- `src/control/mpc.py`
- `src/control/pid.py`
- `src/control/feedback_linearization.py`

- [ ] Update to accept `ContinuousSystemBase` or `DiscreteSystemBase`
- [ ] Update type hints
- [ ] Update docstrings
- [ ] Run control tests

**Dependencies**: 1.1, 1.2, 3.7  
**Estimated Time**: 2-3 hours  
**Risk**: Low-Medium

### 11.2 Update Observer Modules
**Files**:
- `src/observers/kalman_filter.py`
- `src/observers/extended_kalman_filter.py`
- `src/observers/luenberger_observer.py`

- [ ] Update to accept new base classes
- [ ] Update type hints
- [ ] Update docstrings
- [ ] Run observer tests

**Dependencies**: 1.1, 1.2, 2.2, 3.6  
**Estimated Time**: 2-3 hours  
**Risk**: Low-Medium

---

## Phase 12: Update Module Exports and Documentation

### 12.1 Update Main Package Exports
**File**: `src/systems/__init__.py`

- [ ] Export new base classes
- [ ] Export renamed classes
- [ ] Add deprecation warnings for old names
- [ ] Update package docstring
- [ ] Create migration guide

**Dependencies**: All previous phases  
**Estimated Time**: 1-2 hours  
**Risk**: Low

### 12.2 Update Core Module Exports
**File**: `src/systems/base/core/__init__.py`

- [ ] Export `ContinuousSystemBase`
- [ ] Export `DiscreteSystemBase`
- [ ] Export `ContinuousSymbolicSystem`
- [ ] Export `ContinuousStochasticSystem`
- [ ] Export backward compatibility aliases
- [ ] Update module docstring

**Dependencies**: 2.1, 2.2  
**Estimated Time**: 30 minutes  
**Risk**: Low

### 12.3 Create Migration Guide
**File**: `MIGRATION_GUIDE.md`

- [ ] Document all renamed classes
- [ ] Provide before/after code examples
- [ ] List breaking changes
- [ ] Provide upgrade instructions
- [ ] Document deprecation timeline

**Dependencies**: All previous phases  
**Estimated Time**: 2-3 hours  
**Risk**: Low

---

## Phase 13: Documentation and Examples

### 13.1 Update API Documentation
- [ ] Update docstrings for all modified classes
- [ ] Update type hints throughout
- [ ] Generate new API documentation
- [ ] Review for consistency

**Dependencies**: All previous phases  
**Estimated Time**: 3-4 hours  
**Risk**: Low

### 13.2 Update Examples and Tutorials
- [ ] Update all example scripts
- [ ] Update Jupyter notebooks
- [ ] Create examples showcasing new architecture
- [ ] Test all examples

**Dependencies**: All previous phases  
**Estimated Time**: 4-6 hours  
**Risk**: Low

### 13.3 Update README and Documentation
- [ ] Update main README.md
- [ ] Update architecture diagrams
- [ ] Update quickstart guide
- [ ] Update contribution guidelines

**Dependencies**: 13.1, 13.2  
**Estimated Time**: 2-3 hours  
**Risk**: Low

---

## Phase 14: Final Testing and Validation

### 14.1 Comprehensive Test Suite
- [ ] Run full test suite
- [ ] Verify 100% of tests pass
- [ ] Check test coverage (aim for >90%)
- [ ] Add missing tests if coverage dropped
- [ ] Run tests with deprecation warnings enabled

**Dependencies**: All previous phases  
**Estimated Time**: 2-3 hours  
**Risk**: Medium

### 14.2 Integration Testing
- [ ] Test end-to-end workflows
- [ ] Test discretization pipelines
- [ ] Test simulation pipelines
- [ ] Test control design pipelines
- [ ] Test estimation pipelines

**Dependencies**: 14.1  
**Estimated Time**: 3-4 hours  
**Risk**: Medium

### 14.3 Performance Testing
- [ ] Benchmark critical operations
- [ ] Compare performance with baseline
- [ ] Ensure no significant regressions
- [ ] Document any performance changes

**Dependencies**: 14.1  
**Estimated Time**: 2-3 hours  
**Risk**: Low

### 14.4 Backward Compatibility Testing
- [ ] Test all deprecated aliases work
- [ ] Test deprecation warnings appear
- [ ] Create backward compatibility test suite
- [ ] Document any breaking changes

**Dependencies**: 14.1  
**Estimated Time**: 2-3 hours  
**Risk**: Medium

---

## Phase 15: Deployment Preparation

### 15.1 Version Management
- [ ] Update version number (follow semantic versioning)
- [ ] Create changelog entry
- [ ] Tag release candidate
- [ ] Update release notes

**Dependencies**: Phase 14  
**Estimated Time**: 1 hour  
**Risk**: Low

### 15.2 Release Candidate Testing
- [ ] Deploy RC to test environment
- [ ] Run extended test suite
- [ ] Test in real-world scenarios
- [ ] Gather feedback from beta testers

**Dependencies**: 15.1  
**Estimated Time**: Ongoing (1 week)  
**Risk**: Low-Medium

### 15.3 Final Review
- [ ] Code review all changes
- [ ] Review all documentation updates
- [ ] Final spell check and proofreading
- [ ] Verify all deprecation warnings are appropriate

**Dependencies**: 15.2  
**Estimated Time**: 2-3 hours  
**Risk**: Low

---

## Phase 16: Merge and Release

### 16.1 Merge to Main
- [ ] Merge feature branch to main/master
- [ ] Resolve any merge conflicts
- [ ] Run full test suite on main
- [ ] Tag release version

**Dependencies**: Phase 15  
**Estimated Time**: 1-2 hours  
**Risk**: Low

### 16.2 Release
- [ ] Create GitHub release
- [ ] Publish to PyPI (if applicable)
- [ ] Update documentation website
- [ ] Announce release

**Dependencies**: 16.1  
**Estimated Time**: 1-2 hours  
**Risk**: Low

### 16.3 Post-Release Monitoring
- [ ] Monitor issue tracker for bug reports
- [ ] Monitor deprecation warning usage
- [ ] Prepare hotfix process if needed
- [ ] Plan deprecation removal timeline

**Dependencies**: 16.2  
**Estimated Time**: Ongoing  
**Risk**: Low

---

## Phase 17: Future Extensions (Post-Refactoring Roadmap)

This phase outlines longer-term goals and extensions for the library after the core refactoring is complete. These are from the original README Phase 5 and represent the library's ultimate vision.

### 17.1 Reinforcement Learning Environment Integration
**Goal**: Enable automatic RL environment construction from symbolic systems

**Tasks**:
- [ ] Gymnasium environment wrapper for dynamical systems
- [ ] PyBullet integration for physics simulation
- [ ] Brax integration for JAX-based RL
- [ ] Automatic action/observation space definition
- [ ] Reward function specification interface
- [ ] Episode termination condition handling

**Use Case**: Convert any `ContinuousSymbolicSystem` or `DiscreteSymbolicSystem` into a standard RL environment

**Dependencies**: Phase 16  
**Estimated Time**: 2-3 weeks  
**Risk**: Medium

### 17.2 Synthetic Data Generation and Export
**Goal**: Streamline data generation for ML applications

**Tasks**:
- [ ] Batch simulation orchestration
- [ ] Parallel trajectory generation
- [ ] Data export formats (HDF5, CSV, NumPy, PyTorch)
- [ ] Monte Carlo sampling for stochastic systems
- [ ] Trajectory augmentation (noise injection, resampling)
- [ ] Dataset splitting utilities (train/val/test)
- [ ] Data validation and quality checks

**Use Case**: Generate training datasets for neural network controllers or system identification

**Dependencies**: Phase 16  
**Estimated Time**: 1-2 weeks  
**Risk**: Low

### 17.3 Neural Network Verification Integration
**Goal**: Connect with NN verification tools for safe learning-based control

**Tasks**:
- [ ] VNN-Lib format export for systems and specifications
- [ ] Auto-LiRPA/CROWN integration for robustness verification
- [ ] NFL-Veripy integration for neural feedback loop verification
- [ ] Safety specification language for control systems
- [ ] Barrier certificate generation interface
- [ ] Counterexample-guided refinement support

**Use Case**: Verify neural network controllers satisfy safety constraints on symbolic systems

**Dependencies**: Phase 16  
**Estimated Time**: 3-4 weeks  
**Risk**: High (complex external integrations)

### 17.4 Lyapunov-Based Controller Synthesis
**Goal**: Implement neural Lyapunov controller design and verification

**Based on**:
- [Lyapunov-stable Neural Control for State and Output Feedback: A Novel Formulation](https://proceedings.mlr.press/v235/yang24f.html)
- [Certifying Stability of Reinforcement Learning Policies using Generalized Lyapunov Functions](https://arxiv.org/abs/2505.10947v3)

**Tasks**:
- [ ] Generalized Lyapunov function class definitions
- [ ] Neural Lyapunov network architectures
- [ ] Lyapunov derivative verification (symbolic + numerical)
- [ ] Controller synthesis with stability guarantees
- [ ] Observer synthesis with Lyapunov methods
- [ ] Region of attraction estimation
- [ ] Stability certification for RL policies

**Use Case**: Design provably stable neural controllers for nonlinear systems

**Dependencies**: Phase 16  
**Estimated Time**: 4-6 weeks  
**Risk**: High (research-grade implementation)

### 17.5 Model Predictive Control Integration
**Goal**: Add state-of-the-art MPC capabilities

**Libraries under consideration**: do-mpc, CasADi, acados, nMPyC, GEKKO

**Tasks**:
- [ ] Evaluate and select MPC backend(s)
- [ ] Create unified MPC interface for symbolic systems
- [ ] Implement nonlinear MPC (NMPC)
- [ ] Implement economic MPC (EMPC)
- [ ] Support for multi-stage MPC
- [ ] Robust MPC with uncertainty
- [ ] Stochastic MPC for SDEs
- [ ] Real-time MPC code generation
- [ ] MPC warm-starting strategies

**Use Case**: Optimal control for constrained nonlinear systems

**Dependencies**: Phase 16  
**Estimated Time**: 3-4 weeks  
**Risk**: Medium

### 17.6 Parameter Sensitivity Analysis
**Goal**: Analyze how parameter variations affect system behavior

**Tasks**:
- [ ] Local sensitivity analysis (derivatives w.r.t. parameters)
- [ ] Global sensitivity analysis (Sobol indices, variance-based)
- [ ] Parameter uncertainty propagation
- [ ] Morris screening for high-dimensional parameters
- [ ] Automatic sensitivity plot generation
- [ ] Parameter importance ranking
- [ ] Identifiability analysis

**Use Case**: Determine which physical parameters most affect system dynamics

**Dependencies**: Phase 16  
**Estimated Time**: 1-2 weeks  
**Risk**: Low

### 17.7 Advanced Control Theory Techniques
**Goal**: Expand control capabilities beyond LQR/LQG

**Techniques to consider** (based on community interest):
- [ ] Robust control (H∞, H2, μ-synthesis)
- [ ] Adaptive control (MRAC, L1 adaptive)
- [ ] Sliding mode control
- [ ] Backstepping control
- [ ] Stochastic optimal control
- [ ] Risk-sensitive control
- [ ] Tube MPC (robust MPC)

**Use Case**: Advanced controller design for systems with uncertainty

**Dependencies**: Phase 16  
**Estimated Time**: Varies by technique (2-4 weeks each)  
**Risk**: Medium-High

### 17.8 Composite System Construction
**Goal**: Build large systems from interconnected subsystems

**Tasks**:
- [ ] System composition operators (series, parallel, feedback)
- [ ] Port-based modeling (power flow, bond graphs)
- [ ] Algebraic loop detection and resolution
- [ ] Hierarchical system definition
- [ ] Modularity with component libraries
- [ ] Automatic dimension matching and validation
- [ ] Block diagram visualization

**Use Case**: Model complex systems like aircraft (propulsion + aerodynamics + control surfaces)

**Dependencies**: Phase 16  
**Estimated Time**: 2-3 weeks  
**Risk**: Medium

### 17.9 Stochastic Observation Dynamics
**Goal**: Support stochastic measurement models

**Tasks**:
- [ ] Define stochastic observation functions (y = h(x) + measurement noise)
- [ ] Diffusion + drift for output variables
- [ ] Correlated state/measurement noise
- [ ] Non-Gaussian measurement noise
- [ ] Extended Kalman filter for nonlinear stochastic observations
- [ ] Unscented Kalman filter support
- [ ] Particle filter integration

**Use Case**: Realistic sensor models with noise for estimation problems

**Dependencies**: Phase 16  
**Estimated Time**: 2-3 weeks  
**Risk**: Low-Medium

---

## Risk Matrix

| Phase | Risk Level | Mitigation Strategy |
|-------|-----------|---------------------|
| Phase 0 | Low | Full backup, separate branch |
| Phase 1 | Low | Standalone base classes, comprehensive tests |
| Phase 2 | Medium | Backward compatibility aliases, extensive testing |
| Phase 3 | Medium | Incremental adoption, backward compatibility via `__iter__` |
| Phase 4 | Medium-High | Multiple inheritance testing, discrete-specific tests |
| Phase 5 | Medium | Type checking, integration tests |
| Phase 6-8 | Low-Medium | Incremental updates, test after each file |
| Phase 9 | Low | Built-in systems have good test coverage |
| Phase 10-11 | Low-Medium | Type system validation, control tests |
| Phase 12-13 | Low | Documentation review, example testing |
| Phase 14 | Medium | Comprehensive testing, performance benchmarks |
| Phase 15-16 | Low-Medium | RC testing, gradual rollout |
| Phase 17 | Varies | Community-driven, incremental implementation |

---

## Critical Dependencies Graph

```
Phase 0 (Prep)
    ↓
Phase 1 (Base Classes)
    ↓
Phase 2 (Rename Continuous Classes)
    ↓
Phase 3 (Type System Integration) ← CRITICAL PHASE
    ↓
    ├─→ Phase 4 (Discrete Systems)
    │        ↓
    ├─→ Phase 5 (Wrapper)
    │        ↓
    ├─→ Phase 6 (Discretization)
    │        ↓
    ├─→ Phase 7 (Integration)
    │        ↓
    ├─→ Phase 8 (Utilities)
    │        ↓
    └─→ Phase 11 (Control/Observers)
             ↓
        Phase 9-10, 12-13 (Systems, Types, Docs)
             ↓
        Phase 14 (Final Testing)
             ↓
        Phase 15 (Deployment Prep)
             ↓
        Phase 16 (Release)
             ↓
        Phase 17 (Future Extensions)
```

---

## Estimated Timeline

| Phase | Estimated Time | Cumulative |
|-------|---------------|------------|
| Phase 0 | 4 hours | 4 hours |
| Phase 1 | 7 hours | 11 hours |
| Phase 2 | 12 hours | 23 hours |
| Phase 3 | 68 hours | 91 hours |
| Phase 4 | 15 hours | 106 hours |
| Phase 5 | 12 hours | 118 hours |
| Phase 6 | 8 hours | 126 hours |
| Phase 7 | 7 hours | 133 hours |
| Phase 8 | 6 hours | 139 hours |
| Phase 9 | 6 hours | 145 hours |
| Phase 10 | 2.5 hours | 147.5 hours |
| Phase 11 | 5 hours | 152.5 hours |
| Phase 12 | 4 hours | 156.5 hours |
| Phase 13 | 10 hours | 166.5 hours |
| Phase 14 | 12 hours | 178.5 hours |
| Phase 15 | 1 week + 6 hours | ~1 week |
| Phase 16 | 4 hours | ~1 week |
| Phase 17 | Ongoing | N/A |

**Total Estimated Time (Phases 0-16)**: ~178 hours of active development + 1 week RC testing  
**Realistic Timeline**: 4-5 weeks with dedicated effort  
**Phase 17**: Community-driven, implemented incrementally based on demand

---

## Success Criteria

✅ All 61 test files pass  
✅ Test coverage maintained or improved (>90%)  
✅ No performance regressions (within 5% of baseline)  
✅ All backward compatibility aliases work  
✅ Deprecation warnings fire appropriately  
✅ `mypy --strict` passes with 0 errors  
✅ All structured return types implemented  
✅ Documentation fully updated  
✅ Migration guide complete  
✅ All examples run successfully  
✅ Code review approved  
✅ Zero critical bugs in RC testing  

---

## Rollback Plan

If critical issues are discovered:

1. **Immediate**: Revert merge, restore from backup
2. **Investigation**: Identify root cause in feature branch
3. **Fix**: Apply targeted fixes
4. **Re-test**: Run full test suite again
5. **Re-deploy**: Attempt merge again

---

## Notes and Considerations

### Multiple Inheritance Concerns (Phase 4)
- Python's MRO (Method Resolution Order) must be carefully managed
- Use `super()` appropriately in all methods
- Test MRO explicitly: `DiscreteStochasticSystem.__mro__`
- Consider using composition over inheritance if MRO becomes problematic

### Backward Compatibility Strategy
- Keep old class names as aliases for at least 2 major versions
- Use `warnings.warn()` with `DeprecationWarning`
- Provide clear migration path in warnings
- Remove aliases in version N+2 after deprecation in version N
- Add `__iter__` to result classes for tuple unpacking backward compatibility

### Type Checking Compatibility
- Ensure type hints work with mypy, pyright, and pyre
- Use `typing.Protocol` for structural subtyping if needed
- Test with `--strict` mode in mypy
- All return types should be structured (no naked tuples for multi-value returns)

### Performance Considerations
- Profile before and after refactoring
- Watch for overhead from multiple inheritance
- Ensure no unnecessary method calls in hot paths
- Consider using `__slots__` for base classes if memory is concern
- Structured return types should have minimal overhead

### Documentation Strategy
- Use Sphinx autodoc for API documentation
- Include mermaid diagrams for class hierarchies
- Provide migration examples for every renamed class
- Create video tutorial for major refactoring changes
- Document all breaking changes prominently

---

## Maintenance Plan Post-Refactoring

### Deprecation Timeline
- **Version N** (Current): Introduce new classes, deprecate old names
- **Version N+1**: Keep deprecated aliases, strengthen warnings
- **Version N+2**: Remove deprecated aliases, breaking change

### Monitoring
- Track usage of deprecated classes via logging (if telemetry available)
- Monitor GitHub issues for migration problems
- Create FAQ for common migration questions

### Support
- Provide migration assistance on issue tracker
- Create detailed migration examples for complex use cases
- Offer compatibility shim package if needed

---

## Phase 17 Success Criteria

✅ At least 3 of 9 sub-phases implemented  
✅ Integration tests for new capabilities  
✅ Examples/tutorials for each feature  
✅ Documentation updated  
✅ Community feedback incorporated  
✅ No regression in core functionality  

**Note**: Phase 17 is aspirational and can be implemented incrementally based on user demand and contribution.

---

## Appendix: Key Files to Review

### High Priority (Direct Impact)
1. `src/systems/base/core/symbolic_dynamical_system.py`
2. `src/systems/base/core/stochastic_dynamical_system.py`
3. `src/systems/base/core/discrete_symbolic_system.py`
4. `src/systems/base/core/discrete_stochastic_system.py`

### Medium Priority (Indirect Dependencies)
5. `src/systems/base/discretization/discretizer.py`
6. `src/systems/base/discretization/stochastic/stochastic_discretizer.py`
7. `src/systems/base/numerical_integration/integrator_base.py`
8. `src/systems/base/numerical_integration/stochastic/sde_integrator_base.py`

### Lower Priority (Utilities)
9. `src/systems/base/utils/linearization_engine.py`
10. `src/systems/base/utils/stochastic/diffusion_handler.py`

---

**END OF CHECKLIST**
