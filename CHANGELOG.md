# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Refactoring with unified discrete-time interface
- Abstract base classes: `ContinuousSystemBase`, `DiscreteSystemBase`
- Type system integration from `src/types/`
- Structured return types (`LinearizationResult`, `SimulationResult`, etc.)
- Comprehensive pyproject.toml with all tool configurations
- Pre-commit hooks for code quality
- GitHub Actions CI/CD pipeline
- Comprehensive documentation and examples

### Changed
- **BREAKING**: Renamed `StochasticDynamicalSystem` → `ContinuousStochasticSystem`
- **BREAKING**: Renamed `SymbolicDynamicalSystem` → `ContinuousSymbolicSystem`
- **BREAKING**: `linearize()` methods now return structured types instead of tuples
- **BREAKING**: `integrate()` methods now return `SimulationResult` instead of `(t, x)` tuple
- **BREAKING**: `simulate()` methods now return structured result objects
- Improved package structure and organization
- Enhanced type hints throughout codebase
- Better separation of concerns (continuous vs discrete systems)

### Deprecated
- Old class names (`SymbolicDynamicalSystem`, `StochasticDynamicalSystem`)
  - **Backward compatible aliases provided**
  - Will emit DeprecationWarning
  - Will be removed in v1.0.0

### Fixed
- Improved handling of stochastic system parameters
- Better error messages for invalid system definitions
- Fixed edge cases in discretization

### Migration Guide
See [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) for detailed upgrade instructions.

---

## [0.1.0] - TBD (Post-Refactoring Release)

This will be the first release after the major refactoring effort.

### Highlights
- Cleaner API with abstract base classes
- Full type hint coverage
- Structured return types for better IDE support
- Backward compatibility maintained with deprecation warnings

---

## [0.2.0] - 2024-XX-XX

### Added
- Initial release of ControlDESymulation
- Symbolic dynamical system framework using SymPy
- Multi-backend support: NumPy, PyTorch, JAX
- Continuous-time ODE systems
- Stochastic differential equations (SDEs)
- Discrete-time system support via discretization
- Basic control utilities (LQR, linearization)
- Visualization tools (Plotly-based trajectory plotting)
- Built-in systems library (pendulum, mass-spring-damper, etc.)
- Basic numerical integrators (SciPy, torchdiffeq, torchsde)

### Features
- Define systems symbolically once, execute on any backend
- Automatic code generation from symbolic expressions
- GPU acceleration via PyTorch and JAX
- Classical control theory tools
- State-space modeling and analysis

---

## Development Philosophy

### Versioning Strategy
- **Major version (X.0.0)**: Breaking API changes
- **Minor version (0.X.0)**: New features, backward compatible
- **Patch version (0.0.X)**: Bug fixes, documentation

### Breaking Changes Policy
- Breaking changes announced at least one minor version in advance
- Deprecation warnings added before removal
- Migration guides provided
- Backward compatibility maintained where possible

### Release Cycle
- Patch releases: As needed for critical bugs
- Minor releases: Every 2-3 months
- Major releases: When significant breaking changes accumulate

---

## Links
- [GitHub Repository](https://github.com/gbenezer/ControlDESymulation)
- [Issue Tracker](https://github.com/gbenezer/ControlDESymulation/issues)
- [Documentation](https://github.com/gbenezer/ControlDESymulation)