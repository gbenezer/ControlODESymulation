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
Stochastic Numerical Integration
================================

This module provides numerical integrators for stochastic differential equations (SDEs).
Supports multiple backends (NumPy/Julia, PyTorch, JAX) with both Ito and Stratonovich
interpretations.

Mathematical Form
-----------------
Stochastic differential equations:

    dx = f(x, u, t)dt + g(x, u, t)dW

where:
    - f(x, u, t): Drift vector (deterministic dynamics)
    - g(x, u, t): Diffusion matrix (stochastic intensity)
    - dW: Brownian motion increments

Supported Backends
------------------
- **NumPy/Julia (DiffEqPy)**: Access to Julia's extensive SDE solver ecosystem
- **PyTorch (TorchSDE)**: GPU acceleration, neural SDEs, adjoint methods
- **JAX (Diffrax)**: JIT compilation, autodiff, custom noise support

Authors
-------
Gil Benezer

License
-------
GNU Affero General Public License v3.0
"""

# Base class and utilities
from .sde_integrator_base import SDEIntegratorBase, get_trajectory_statistics

# Factory and convenience functions
from .sde_integrator_factory import (
    SDEIntegratorFactory,
    SDEIntegratorType,
    auto_sde_integrator,
    create_sde_integrator,
)

# Backend-specific SDE integrators
from .diffeqpy_sde_integrator import DiffEqPySDEIntegrator
from .torchsde_integrator import TorchSDEIntegrator
from .diffrax_sde_integrator import DiffraxSDEIntegrator

# Custom Brownian path utilities
from .custom_brownian import CustomBrownianPath, create_custom_or_random_brownian

# Export public API
__all__ = [
    # Base class
    "SDEIntegratorBase",
    # Factory
    "SDEIntegratorFactory",
    "SDEIntegratorType",
    # Convenience functions
    "create_sde_integrator",
    "auto_sde_integrator",
    # Utilities
    "get_trajectory_statistics",
    # Backend-specific integrators
    "DiffEqPySDEIntegrator",
    "TorchSDEIntegrator",
    "DiffraxSDEIntegrator",
    # Custom Brownian path
    "CustomBrownianPath",
    "create_custom_or_random_brownian",
]
