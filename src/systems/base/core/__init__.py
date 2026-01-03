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
Core System Classes
==================================

This module provides the abstract base classes that define the fundamental
interfaces for all dynamical systems in the ControlDESymulation framework.

Authors
-------
Gil Benezer

License
-------
GNU Affero General Public License v3.0
"""

# Layer 3: Stochastic systems
from .continuous_stochastic_system import ContinuousStochasticSystem, StochasticDynamicalSystem

# Layer 2: Deterministic symbolic systems
from .continuous_symbolic_system import (
    ContinuousDynamicalSystem,
    ContinuousSymbolicSystem,
    SymbolicDynamicalSystem,
)

# Layer 1: Abstract base classes
from .continuous_system_base import ContinuousSystemBase
from .discrete_stochastic_system import DiscreteStochasticSystem
from .discrete_symbolic_system import DiscreteDynamicalSystem, DiscreteSymbolicSystem
from .discrete_system_base import DiscreteSystemBase

# Layer 4: Discretized systems (bridge)
from .discretized_system import (
    DiscretizationMode,
    DiscretizedSystem,
    analyze_discretization_error,
    compute_discretization_quality,
    discretize,
    discretize_batch,
    recommend_dt,
)
from .symbolic_system_base import SymbolicSystemBase

# Export public API
__all__ = [
    # Layer 1: Abstract base classes
    "SymbolicSystemBase",
    "ContinuousSystemBase",
    "DiscreteSystemBase",
    # Layer 2: Deterministic symbolic systems and aliases
    "ContinuousSymbolicSystem",
    "ContinuousDynamicalSystem",
    "SymbolicDynamicalSystem",
    "DiscreteDynamicalSystem",
    "DiscreteSymbolicSystem",
    # Layer 3: Stochastic systems and aliases
    "ContinuousStochasticSystem",
    "StochasticDynamicalSystem",
    "DiscreteStochasticSystem",
    # Layer 4: Discretized systems
    "DiscretizationMode",
    "DiscretizedSystem",
    "discretize",
    "discretize_batch",
    "analyze_discretization_error",
    "recommend_dt",
    "compute_discretization_quality",
]
