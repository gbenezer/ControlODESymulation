# Copyright (C) 2025 Gil Benezer
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
Numerical Integration
=====================

This module provides numerical integrators for both deterministic (ODE) and
stochastic (SDE) differential equations. Supports multiple backends including
NumPy/SciPy, PyTorch, JAX, and Julia's DifferentialEquations.jl.

Deterministic Integration (ODEs)
--------------------------------
For ordinary differential equations: dx/dt = f(x, u, t)

>>> from controldesymulation.systems.base.numerical_integration import (
...     IntegratorFactory,
...     create_integrator,
...     auto_integrator,
... )
>>>
>>> # Create integrator via factory
>>> integrator = IntegratorFactory.create(system, backend='numpy', method='RK45')
>>>
>>> # Or use convenience functions
>>> integrator = auto_integrator(system)  # Auto-select best backend

Stochastic Integration (SDEs)
-----------------------------
For stochastic differential equations: dx = f(x,u,t)dt + g(x,u,t)dW

>>> from controldesymulation.systems.base.numerical_integration.stochastic import (
...     SDEIntegratorFactory,
...     create_sde_integrator,
... )
>>>
>>> integrator = SDEIntegratorFactory.create(sde_system, backend='jax')

Supported Backends
------------------
**Deterministic (ODE):**
- NumPy/SciPy: LSODA, RK45, BDF, Radau (adaptive)
- NumPy/DiffEqPy: Tsit5, Vern9, Rosenbrock23 (Julia solvers)
- PyTorch/TorchDiffEq: dopri5, dopri8 (neural ODEs)
- JAX/Diffrax: tsit5, dopri5, dopri8 (optimization)

**Stochastic (SDE):**
- NumPy/DiffEqPy: EM, SRIW1, SRA1 (Julia solvers)
- PyTorch/TorchSDE: euler, milstein, srk (neural SDEs)
- JAX/Diffrax: Euler, ItoMilstein, SEA (optimization)

Authors
-------
Gil Benezer

License
-------
GNU Affero General Public License v3.0
"""

# Base classes and enums
# Re-export stochastic module for convenience
from . import stochastic

# Fixed-step integrators (always available, no external dependencies)
from .fixed_step_integrators import (
    ExplicitEulerIntegrator,
    MidpointIntegrator,
    RK4Integrator,
    HeunIntegrator,
    create_fixed_step_integrator,
)
from .integrator_base import IntegratorBase, StepMode

# Factory and convenience functions
from .integrator_factory import (
    IntegratorFactory,
    IntegratorType,
    auto_integrator,
    create_integrator,
)

# Export public API
__all__ = [
    # Base classes and enums
    "IntegratorBase",
    "StepMode",
    # Factory
    "IntegratorFactory",
    "IntegratorType",
    # Convenience functions
    "create_integrator",
    "auto_integrator",
    # Fixed-step integrators
    "ExplicitEulerIntegrator",
    "MidpointIntegrator",
    "HeunIntegrator",
    "RK4Integrator",
    "create_fixed_step_integrator",
    # Stochastic submodule
    "stochastic",
]
