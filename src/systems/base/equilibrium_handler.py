from typing import Dict, List, Optional, Tuple
import numpy as np
import warnings


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
        self.nx = nx
        self.nu = nu

        self._equilibria: Dict[str, Dict[str, np.ndarray]] = {}
        self._default: str = "origin"

        # Initialize origin
        self._equilibria["origin"] = {
            "x": np.zeros(nx),
            "u": np.zeros(nu),
            "metadata": {},  # For storing stability info, etc.
        }

    def add(
        self,
        name: str,
        x_eq: np.ndarray,
        u_eq: np.ndarray,
        verify_fn: Optional[callable] = None,
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

        if x_eq.shape[0] != self.nx:
            raise ValueError(f"x_eq must have shape ({self.nx},), got {x_eq.shape}")
        if u_eq.shape[0] != self.nu:
            raise ValueError(f"u_eq must have shape ({self.nu},), got {u_eq.shape}")

        # Verify if function provided
        if verify_fn is not None:
            dx = verify_fn(x_eq, u_eq)
            max_deriv = np.abs(dx).max() if isinstance(dx, np.ndarray) else abs(dx).max()

            if max_deriv > tol:
                warnings.warn(
                    f"Equilibrium '{name}' may not be valid: "
                    f"max|f(x,u)| = {max_deriv:.2e} > {tol:.2e}"
                )
                metadata["verified"] = False
                metadata["max_residual"] = float(max_deriv)
            else:
                metadata["verified"] = True
                metadata["max_residual"] = float(max_deriv)

        self._equilibria[name] = {"x": x_eq, "u": u_eq, "metadata": metadata}

    def get_x(self, name: Optional[str] = None, backend: str = "numpy"):
        """Get equilibrium state in specified backend"""
        name = name or self._default

        if name not in self._equilibria:
            available = list(self._equilibria.keys())
            raise ValueError(f"Unknown equilibrium '{name}'. Available: {available}")

        x_eq = self._equilibria[name]["x"]
        return self._convert_to_backend(x_eq, backend)

    def get_u(self, name: Optional[str] = None, backend: str = "numpy"):
        """Get equilibrium control in specified backend"""
        name = name or self._default

        if name not in self._equilibria:
            raise ValueError(f"Unknown equilibrium '{name}'")

        u_eq = self._equilibria[name]["u"]
        return self._convert_to_backend(u_eq, backend)

    def get_both(self, name: Optional[str] = None, backend: str = "numpy"):
        """Get both state and control equilibria"""
        return self.get_x(name, backend), self.get_u(name, backend)

    def set_default(self, name: str):
        """Set default equilibrium"""
        if name not in self._equilibria:
            raise ValueError(f"Unknown equilibrium '{name}'")
        self._default = name

    def list_names(self) -> List[str]:
        """List all equilibrium names"""
        return list(self._equilibria.keys())

    def get_metadata(self, name: Optional[str] = None) -> Dict:
        """Get metadata for equilibrium"""
        name = name or self._default
        if name not in self._equilibria:
            raise ValueError(f"Unknown equilibrium '{name}'")
        return self._equilibria[name]["metadata"]

    def _convert_to_backend(self, arr: np.ndarray, backend: str):
        """Convert NumPy array to target backend"""
        if backend == "numpy":
            return arr
        elif backend == "torch":
            import torch

            return torch.tensor(arr, dtype=torch.float32)
        elif backend == "jax":
            import jax.numpy as jnp

            return jnp.array(arr)
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def __repr__(self) -> str:
        return f"EquilibriumHandler({len(self._equilibria)} equilibria: {self.list_names()})"
