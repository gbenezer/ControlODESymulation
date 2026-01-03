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

import warnings
from typing import Callable, Dict, List, Optional

import numpy as np

from src.types.backends import Backend
from src.types.core import EquilibriumControl, EquilibriumName, EquilibriumPoint, EquilibriumState


class EquilibriumHandler:
    """
    Manages multiple equilibrium points for a dynamical system.

    Stores equilibria as backend-neutral NumPy arrays and provides
    conversion to any backend on demand.
    """

    def __init__(self, nx: int, nu: int):
        """
        Initialize equilibrium handler.

        Args:
            nx: Number of states
            nu: Number of controls
        """
        self._nx = nx
        self._nu = nu

        self._equilibria: Dict[str, Dict[str, np.ndarray]] = {}
        self._default: str = "origin"

        # Initialize origin
        self._equilibria["origin"] = {
            "x": np.zeros(nx),
            "u": np.zeros(nu),
            "metadata": {},  # For storing stability info, etc.
        }

    @property
    def nx(self) -> int:
        """Number of states"""
        return self._nx

    @nx.setter
    def nx(self, value: int):
        """Update state dimension and validate existing equilibria"""
        if self._nx != 0 and value != self._nx:
            # Validate all existing equilibria still have correct dimensions
            for name, eq in self._equilibria.items():
                if eq["x"].shape[0] != value:
                    raise ValueError(
                        f"Cannot change nx from {self._nx} to {value}: "
                        f"equilibrium '{name}' has wrong dimension",
                    )

        # Update dimension
        old_nx = self._nx
        self._nx = value

        # Recreate origin if dimensions changed from initialization
        if old_nx == 0 and value > 0:
            self._equilibria["origin"]["x"] = np.zeros(value)

    @property
    def nu(self) -> int:
        """Number of controls"""
        return self._nu

    @nu.setter
    def nu(self, value: int):
        """Update control dimension and validate existing equilibria"""
        if self._nu != 0 and value != self._nu:
            for name, eq in self._equilibria.items():
                if eq["u"].shape[0] != value:
                    raise ValueError(
                        f"Cannot change nu from {self._nu} to {value}: "
                        f"equilibrium '{name}' has wrong dimension",
                    )

        # Update dimension
        old_nu = self._nu
        self._nu = value

        # Recreate origin if dimensions changed from initialization
        if old_nu == 0 and value > 0:
            self._equilibria["origin"]["u"] = np.zeros(value)

    def add(
        self,
        name: EquilibriumName,
        x_eq: np.ndarray,
        u_eq: np.ndarray,
        verify_fn: Optional[Callable] = None,
        tol: float = 1e-6,
        **metadata,
    ) -> None:
        """
        Add named equilibrium point.

        Args:
            name: Equilibrium name
            x_eq: Equilibrium state
            u_eq: Equilibrium control
            verify_fn: Optional function(x, u) -> dx to verify equilibrium
            tol: Tolerance for verification
            **metadata: Additional info (stability, description, etc.)
        """
        x_eq = np.atleast_1d(np.asarray(x_eq))
        u_eq = np.atleast_1d(np.asarray(u_eq))

        if x_eq.shape[0] != self._nx:
            raise ValueError(f"x_eq must have shape ({self._nx},), got {x_eq.shape}")
        if u_eq.shape[0] != self._nu:
            raise ValueError(f"u_eq must have shape ({self._nu},), got {u_eq.shape}")

        # Verify if function provided
        if verify_fn is not None:
            dx = verify_fn(x_eq, u_eq)
            max_deriv = np.abs(dx).max() if isinstance(dx, np.ndarray) else abs(dx).max()

            # CHECK FOR NaN/Inf BEFORE COMPARISON
            if not np.isfinite(max_deriv):
                warnings.warn(
                    f"Equilibrium '{name}' is invalid: max|f(x,u)| = {max_deriv} (not finite)",
                )
                metadata["verified"] = False
                metadata["max_residual"] = float(max_deriv)
            elif max_deriv > tol:
                warnings.warn(
                    f"Equilibrium '{name}' may not be valid: "
                    f"max|f(x,u)| = {max_deriv:.2e} > {tol:.2e}",
                )
                metadata["verified"] = False
                metadata["max_residual"] = float(max_deriv)
            else:
                metadata["verified"] = True
                metadata["max_residual"] = float(max_deriv)

        self._equilibria[name] = {"x": x_eq, "u": u_eq, "metadata": metadata}

    def get_x(
        self,
        name: Optional[EquilibriumName] = None,
        backend: Backend = "numpy",
    ) -> EquilibriumState:
        """Get equilibrium state in specified backend"""
        name = name or self._default

        if name not in self._equilibria:
            available = list(self._equilibria.keys())
            raise ValueError(f"Unknown equilibrium '{name}'. Available: {available}")

        x_eq = self._equilibria[name]["x"]
        return self._convert_to_backend(x_eq, backend)

    def get_u(
        self,
        name: Optional[EquilibriumName] = None,
        backend: Backend = "numpy",
    ) -> EquilibriumControl:
        """Get equilibrium control in specified backend"""
        name = name or self._default

        if name not in self._equilibria:
            raise ValueError(f"Unknown equilibrium '{name}'")

        u_eq = self._equilibria[name]["u"]
        return self._convert_to_backend(u_eq, backend)

    def get_both(
        self,
        name: Optional[EquilibriumName] = None,
        backend: Backend = "numpy",
    ) -> EquilibriumPoint:
        """Get both state and control equilibria"""
        return self.get_x(name, backend), self.get_u(name, backend)

    def set_default(self, name: EquilibriumName):
        """Set default equilibrium"""
        if name not in self._equilibria:
            raise ValueError(f"Unknown equilibrium '{name}'")
        self._default = name

    def list_names(self) -> List[str]:
        """List all equilibrium names"""
        return list(self._equilibria.keys())

    def get_metadata(self, name: Optional[EquilibriumName] = None) -> Dict:
        """Get metadata for equilibrium"""
        name = name or self._default
        if name not in self._equilibria:
            raise ValueError(f"Unknown equilibrium '{name}'")
        return self._equilibria[name]["metadata"]

    def _convert_to_backend(self, arr: np.ndarray, backend: Backend) -> EquilibriumState:
        """Convert NumPy array to target backend"""
        if backend == "numpy":
            return arr
        if backend == "torch":
            import torch

            # Preserve dtype: float64 -> float64, float32 -> float32
            dtype = torch.float64 if arr.dtype == np.float64 else torch.float32
            return torch.tensor(arr, dtype=dtype)
        if backend == "jax":
            import jax.numpy as jnp

            return jnp.array(arr)  # JAX preserves dtype by default
        raise ValueError(f"Unknown backend: {backend}")

    def __repr__(self) -> str:
        return f"EquilibriumHandler({len(self._equilibria)} equilibria: {self.list_names()})"
