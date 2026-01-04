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
System Utilities
================

This module provides utility classes and functions for working with dynamical systems,
including backend management, dynamics evaluation, linearization, code generation,
and symbolic validation.

Core Utilities
--------------
>>> from src.systems.base.utils import (
...     BackendManager,
...     DynamicsEvaluator,
...     LinearizationEngine,
...     EquilibriumHandler,
... )
>>>
>>> # Manage backend-specific operations
>>> backend = BackendManager(system, backend='jax')
>>>
>>> # Evaluate dynamics efficiently
>>> evaluator = DynamicsEvaluator(system)
>>> dx = evaluator.evaluate(x, u)
>>>
>>> # Linearize around equilibrium
>>> engine = LinearizationEngine(system)
>>> A, B = engine.linearize(x_eq, u_eq)

Code Generation
---------------
>>> from src.systems.base.utils import CodeGenerator, generate_function
>>>
>>> # Generate optimized numerical functions
>>> generator = CodeGenerator(system)
>>> f_numpy = generator.generate('numpy')
>>>
>>> # Or use convenience function
>>> f_jax = generate_function(expr, symbols, backend='jax')

Stochastic Utilities
--------------------
>>> from src.systems.base.utils.stochastic import (
...     SDEValidator,
...     NoiseCharacterizer,
...     DiffusionHandler,
... )

Authors
-------
Gil Benezer

License
-------
GNU Affero General Public License v3.0
"""

# Core utility classes
from .backend_manager import BackendManager
from .dynamics_evaluator import DynamicsEvaluator
from .equilibrium_handler import EquilibriumHandler
from .linearization_engine import LinearizationEngine
from .observation_engine import ObservationEngine

# Validation
# NOTE: ValidationError name collision
# Both symbolic_validator and stochastic.sde_validator define ValidationError.
# To avoid ambiguity, neither is exported at package level. Access them explicitly:
#   - from src.systems.base.utils.symbolic_validator import ValidationError  # for symbolic validation
#   - from src.systems.base.utils.stochastic.sde_validator import ValidationError  # for SDE validation
from .symbolic_validator import SymbolicValidator

# Code generation
from .code_generator import CodeGenerator
from .codegen_utils import (
    generate_function,
    generate_jacobian_function,
    generate_jax_function,
    generate_numpy_function,
    generate_torch_function,
)

# Re-export stochastic submodule for convenience
from . import stochastic

# Export public API
__all__ = [
    # Core utility classes
    "BackendManager",
    "DynamicsEvaluator",
    "EquilibriumHandler",
    "LinearizationEngine",
    "ObservationEngine",
    # Validation
    "SymbolicValidator",
    # Code generation
    "CodeGenerator",
    "generate_function",
    "generate_jacobian_function",
    "generate_numpy_function",
    "generate_torch_function",
    "generate_jax_function",
    # Stochastic submodule
    "stochastic",
]