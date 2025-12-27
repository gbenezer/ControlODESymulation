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
Observation Engine for SymbolicDynamicalSystem

Handles output/observation function evaluation across multiple backends.

Responsibilities:
- Output evaluation: y = h(x)
- Linearized observation: C = ∂h/∂x
- Backend-specific implementations (NumPy, PyTorch, JAX)
- Input validation and shape handling
- Batched vs single evaluation

This class manages all h(x)-related operations including both
the nonlinear output and its linearization.
"""

from typing import TYPE_CHECKING, Optional

import numpy as np
import sympy as sp

# Import from centralized type system
from src.types.backends import Backend
from src.types.core import (
    ArrayLike,
    FeedthroughMatrix,
    OutputMatrix,
    OutputVector,
    StateVector,
)
from src.types.linearization import ObservationLinearization
from src.types.utilities import get_batch_size, is_batched

if TYPE_CHECKING:
    import jax
    import jax.numpy as jnp
    import torch

    from src.systems.base.symbolic_dynamical_system import SymbolicDynamicalSystem
    from src.systems.base.utils.backend_manager import BackendManager
    from src.systems.base.utils.code_generator import CodeGenerator


class ObservationEngine:
    """
    Evaluates observation/output functions across backends.

    Handles the evaluation of y = h(x) and C = ∂h/∂x for all backends
    with proper shape handling and batching.

    Type System Integration:
        - StateVector: Input state
        - OutputVector: Evaluated output y = h(x)
        - OutputMatrix: Linearized observation C = ∂h/∂x
        - ObservationLinearization: (C, D) tuple for full output linearization
        - Backend: Type-safe backend selection

    Batching:
        Supports both single and batched evaluation using centralized
        utilities from the type framework (is_batched, get_batch_size).

    Example:
        >>> engine = ObservationEngine(system, code_gen, backend_mgr)
        >>> y: OutputVector = engine.evaluate(x, backend='numpy')
        >>> C: OutputMatrix = engine.compute_jacobian(x, backend='numpy')
    """

    def __init__(
        self,
        system: "SymbolicDynamicalSystem",
        code_gen: "CodeGenerator",
        backend_mgr: "BackendManager",
    ):
        """
        Initialize observation engine.

        Args:
            system: The dynamical system
            code_gen: Code generator for accessing compiled functions
            backend_mgr: Backend manager for detection/conversion
        """
        self.system = system
        self.code_gen = code_gen
        self.backend_mgr = backend_mgr

    # ========================================================================
    # Output Evaluation: y = h(x)
    # ========================================================================

    def evaluate(
        self, x: StateVector, backend: Optional[Backend] = None
    ) -> OutputVector:
        """
        Evaluate output equation: y = h(x).

        If no custom output function is defined, returns the full state (identity).

        Args:
            x: State vector
            backend: Backend selection:
                - None: Auto-detect from input type (default)
                - 'numpy', 'torch', 'jax': Force specific backend
                - 'default': Use system's default backend

        Returns:
            Output vector (type matches backend)

        Example:
            >>> y: OutputVector = engine.evaluate(x_numpy)  # Auto-detect NumPy
            >>> y: OutputVector = engine.evaluate(x_numpy, backend='torch')  # Convert to torch
        """
        # If no custom output, return identity
        if self.system._h_sym is None:
            return x

        # Determine target backend
        if backend == "default":
            target_backend = self.backend_mgr.default_backend
        elif backend is None:
            target_backend = self.backend_mgr.detect(x)
        else:
            target_backend = backend

        # Convert input if needed
        input_backend = self.backend_mgr.detect(x)
        if input_backend != target_backend:
            x = self.backend_mgr.convert(x, target_backend)

        # Dispatch to backend-specific implementation
        if target_backend == "numpy":
            return self._evaluate_numpy(x)
        elif target_backend == "torch":
            return self._evaluate_torch(x)
        elif target_backend == "jax":
            return self._evaluate_jax(x)
        else:
            raise ValueError(f"Unknown backend: {target_backend}")

    def _evaluate_numpy(self, x: np.ndarray) -> np.ndarray:
        """
        NumPy implementation of output evaluation.

        Uses centralized batching utilities for consistent shape handling.

        Raises:
            ValueError: If batch is empty (batch_size=0)
        """

        # If no custom output, return full state
        if self.system._h_sym is None:
            return x

        # Generate output function
        h_numpy = self.code_gen.generate_output("numpy")
        if h_numpy is None:
            return x

        # Handle batched input using batching utilities
        if not is_batched(x):
            x = np.expand_dims(x, 0)
            squeeze_output = True
        else:
            squeeze_output = False

        batch_size = get_batch_size(x)

        # Check for empty batch BEFORE processing
        if batch_size == 0:
            raise ValueError(
                f"Empty batch detected in output evaluation (batch_size=0). "
                f"Cannot compute outputs for zero samples. "
                f"Received x.shape={x.shape}. "
                f"This usually indicates a bug in data preparation or loop logic. "
                f"Check your data loading, filtering, or iteration code."
            )

        results = []

        for i in range(batch_size):
            x_list = [x[i, j] for j in range(self.system.nx)]
            result = h_numpy(*x_list)
            result = np.atleast_1d(np.array(result))
            results.append(result)

        # Defensive check (should never happen after batch_size check above)
        if len(results) == 0:
            raise RuntimeError(
                "Internal error: No results generated despite non-empty input validation. "
                "This is a bug in the observation engine - please report this."
            )

        result = np.stack(results)

        if squeeze_output:
            result = np.squeeze(result, 0)

        # Ensure correct return type
        result = self.backend_mgr.ensure_type(result, "numpy")

        return result

    def _evaluate_torch(self, x: "torch.Tensor") -> "torch.Tensor":
        """
        PyTorch implementation of output evaluation.

        Uses centralized batching utilities for consistent shape handling.

        Raises:
            ValueError: If batch is empty (batch_size=0)
        """

        if self.system._h_sym is None:
            return x

        # Generate output function
        h_torch = self.code_gen.generate_output("torch")
        if h_torch is None:
            return x

        # Handle batched input using batching utilities
        if not is_batched(x):
            x = x.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        batch_size = get_batch_size(x)

        # Check for empty batch BEFORE processing
        if batch_size == 0:
            raise ValueError(
                f"Empty batch detected in output evaluation (batch_size=0). "
                f"Cannot compute outputs for zero samples. "
                f"Received x.shape={tuple(x.shape)}. "
                f"This usually indicates a bug in data preparation or loop logic. "
                f"Check your DataLoader, filtering, or iteration code."
            )

        # Prepare input arguments
        x_list = [x[:, i] for i in range(self.system.nx)]

        # Call function
        result = h_torch(*x_list)

        # Handle output shape
        if squeeze_output:
            if result.ndim > 1:
                result = result.squeeze(0)
            elif result.ndim == 0:
                result = result.reshape(1)
        else:
            # Batched output - ensure shape is (batch, ny)
            if result.ndim == 1 and self.system.ny == 1:
                result = result.unsqueeze(1)

        # Final safety: ensure at least 1D
        if result.ndim == 0:
            result = result.reshape(1)

        # Ensure correct return type
        result = self.backend_mgr.ensure_type(result, "torch")

        return result

    def _evaluate_jax(self, x: "jnp.ndarray") -> "jnp.ndarray":
        """
        JAX implementation of output evaluation.

        Uses centralized batching utilities for consistent shape handling.

        Raises:
            ValueError: If batch is empty (batch_size=0)
        """
        import jax
        import jax.numpy as jnp

        if self.system._h_sym is None:
            return x

        # Generate output function
        h_jax = self.code_gen.generate_output("jax", jit=True)
        if h_jax is None:
            return x

        # Handle batched input using batching utilities
        if not is_batched(x):
            x = jnp.expand_dims(x, 0)
            squeeze_output = True
        else:
            squeeze_output = False

        batch_size = get_batch_size(x)

        # Check for empty batch BEFORE processing
        if batch_size == 0:
            raise ValueError(
                f"Empty batch detected in output evaluation (batch_size=0). "
                f"Cannot compute outputs for zero samples. "
                f"Received x.shape={x.shape}. "
                f"This usually indicates a bug in data preparation or loop logic. "
                f"Check your data loading, filtering, or vmap usage."
            )

        # For batched computation, use vmap
        if batch_size > 1:

            @jax.vmap
            def batched_observation(x_i):
                x_list = [x_i[j] for j in range(self.system.nx)]
                return h_jax(*x_list)

            result = batched_observation(x)
        else:
            # Single evaluation
            x_list = [x[0, i] for i in range(self.system.nx)]
            result = h_jax(*x_list)
            result = jnp.expand_dims(result, 0)

        if squeeze_output:
            result = jnp.squeeze(result, 0)

        # Ensure correct return type
        result = self.backend_mgr.ensure_type(result, "jax")

        return result

    # ========================================================================
    # Observation Linearization: C = ∂h/∂x
    # ========================================================================

    def compute_jacobian(
        self, x: StateVector, backend: Optional[Backend] = None
    ) -> OutputMatrix:
        """
        Compute linearized observation: C = ∂h/∂x.

        If no custom output function, returns identity matrix.

        Args:
            x: State at which to linearize
            backend: Backend selection:
                - None: Auto-detect from input type (default)
                - 'numpy', 'torch', 'jax': Force specific backend
                - 'default': Use system's default backend

        Returns:
            OutputMatrix
                C matrix (ny, nx) - output Jacobian (type matches backend)

        Example:
            >>> C: OutputMatrix = engine.compute_jacobian(x, backend='numpy')
            >>> print(C.shape)  # (ny, nx)
        """
        # If no custom output, return identity
        if self.system._h_sym is None:
            # Return identity in appropriate backend
            if backend is None:
                backend = self.backend_mgr.detect(x)

            if backend == "numpy" or backend is None:
                if not is_batched(x):
                    return np.eye(self.system.nx)
                else:
                    batch_size = get_batch_size(x)
                    return np.tile(np.eye(self.system.nx), (batch_size, 1, 1))
            elif backend == "torch":
                import torch

                if not is_batched(x):
                    return torch.eye(self.system.nx, dtype=x.dtype, device=x.device)
                else:
                    batch_size = get_batch_size(x)
                    return (
                        torch.eye(self.system.nx, dtype=x.dtype, device=x.device)
                        .unsqueeze(0)
                        .expand(batch_size, -1, -1)
                    )
            elif backend == "jax":
                import jax.numpy as jnp

                if not is_batched(x):
                    return jnp.eye(self.system.nx)
                else:
                    batch_size = get_batch_size(x)
                    return jnp.tile(jnp.eye(self.system.nx), (batch_size, 1, 1))

        # Determine target backend
        if backend == "default":
            target_backend = self.backend_mgr.default_backend
        elif backend is None:
            target_backend = self.backend_mgr.detect(x)
        else:
            target_backend = backend

        # Convert input if needed
        input_backend = self.backend_mgr.detect(x)
        if input_backend != target_backend:
            x = self.backend_mgr.convert(x, target_backend)

        # Dispatch to backend-specific implementation
        if target_backend == "numpy":
            return self._compute_jacobian_numpy(x)
        elif target_backend == "torch":
            return self._compute_jacobian_torch(x)
        elif target_backend == "jax":
            return self._compute_jacobian_jax(x)
        else:
            raise ValueError(f"Unknown backend: {target_backend}")

    def compute_symbolic(self, x_eq: Optional[sp.Matrix] = None) -> sp.Matrix:
        """
        Compute symbolic linearization C = ∂h/∂x.

        Args:
            x_eq: Equilibrium state (zeros if None)

        Returns:
            C: Symbolic Jacobian matrix

        Example:
            >>> C_sym = engine.compute_symbolic(x_eq=sp.Matrix([0, 0]))
        """
        if self.system._h_sym is None:
            return sp.eye(self.system.nx)

        if x_eq is None:
            x_eq = sp.Matrix([0] * self.system.nx)

        # Get symbolic Jacobian from CodeGenerator
        self.code_gen._compute_symbolic_jacobians()
        C_sym_cached = self.code_gen._C_sym_cache

        subs_dict = dict(zip(self.system.state_vars, list(x_eq)))
        C = C_sym_cached.subs(subs_dict)
        C = self.system.substitute_parameters(C)

        return C

    def _compute_jacobian_numpy(self, x: np.ndarray) -> np.ndarray:
        """
        NumPy implementation of observation Jacobian.

        Uses centralized batching utilities for consistent shape handling.

        Raises:
            ValueError: If batch is empty (batch_size=0)
        """

        # Handle batched input using batching utilities
        if not is_batched(x):
            x = np.expand_dims(x, 0)
            squeeze_output = True
        else:
            squeeze_output = False

        batch_size = get_batch_size(x)

        # Check for empty batch BEFORE processing
        if batch_size == 0:
            raise ValueError(
                f"Empty batch detected in observation linearization (batch_size=0). "
                f"Cannot compute observation Jacobian for zero samples. "
                f"Received x.shape={x.shape}. "
                f"This usually indicates a bug in data preparation or loop logic. "
                f"Check your data loading, filtering, or iteration code."
            )

        C_batch = np.zeros((batch_size, self.system.ny, self.system.nx))

        # Try to get cached Jacobian function
        _, _, C_func = self.code_gen.get_jacobians("numpy")

        if C_func is not None:
            # Use cached function
            for i in range(batch_size):
                x_i = x[i]
                x_list = [x_i[j] for j in range(self.system.nx)]

                C_result = C_func(*x_list)
                C_batch[i] = np.array(C_result, dtype=np.float64)
        else:
            # Fall back to symbolic evaluation
            for i in range(batch_size):
                x_np = np.atleast_1d(x[i])
                C_sym = self.compute_symbolic(sp.Matrix(x_np))
                C_batch[i] = np.array(C_sym, dtype=np.float64)

        if squeeze_output:
            C_batch = np.squeeze(C_batch, 0)

        # Ensure correct return type
        C_batch = self.backend_mgr.ensure_type(C_batch, "numpy")

        return C_batch

    def _compute_jacobian_torch(self, x: "torch.Tensor") -> "torch.Tensor":
        """
        PyTorch implementation of observation Jacobian.

        Uses centralized batching utilities for consistent shape handling.

        Raises:
            ValueError: If batch is empty (batch_size=0)
        """
        import torch

        # Handle batched input using batching utilities
        if not is_batched(x):
            x = x.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        batch_size = get_batch_size(x)

        # Check for empty batch BEFORE processing
        if batch_size == 0:
            raise ValueError(
                f"Empty batch detected in observation linearization (batch_size=0). "
                f"Cannot compute observation Jacobian for zero samples. "
                f"Received x.shape={tuple(x.shape)}. "
                f"This usually indicates a bug in data preparation or loop logic. "
                f"Check your DataLoader, filtering, or iteration code."
            )

        device = x.device
        dtype = x.dtype

        C_batch = torch.zeros(
            batch_size, self.system.ny, self.system.nx, dtype=dtype, device=device
        )

        # Try to get cached Jacobian function
        _, _, C_func = self.code_gen.get_jacobians("torch")

        if C_func is not None:
            # Use cached function
            for i in range(batch_size):
                x_i = x[i]
                x_list = [x_i[j] for j in range(self.system.nx)]
                C_batch[i] = C_func(*x_list)
        else:
            # Fall back to symbolic evaluation
            for i in range(batch_size):
                x_i = x[i] if batch_size > 1 else x.squeeze(0)
                x_np = x_i.detach().cpu().numpy()
                x_np = np.atleast_1d(x_np)

                C_sym = self.compute_symbolic(sp.Matrix(x_np))
                C_batch[i] = torch.tensor(
                    np.array(C_sym, dtype=np.float64), dtype=dtype, device=device
                )

        if squeeze_output:
            C_batch = C_batch.squeeze(0)

        # Ensure correct return type
        C_batch = self.backend_mgr.ensure_type(C_batch, "torch")

        return C_batch

    def _compute_jacobian_jax(self, x: "jnp.ndarray") -> "jnp.ndarray":
        """
        JAX implementation using automatic differentiation.

        Uses centralized batching utilities for consistent shape handling.

        Raises:
            ValueError: If batch is empty (batch_size=0)
        """
        import jax
        import jax.numpy as jnp

        # Ensure observation function is generated
        h_jax = self.code_gen.generate_output("jax", jit=True)

        # Handle batched input using batching utilities
        if not is_batched(x):
            x = jnp.expand_dims(x, 0)
            squeeze_output = True
        else:
            squeeze_output = False

        batch_size = get_batch_size(x)

        # Check for empty batch BEFORE processing
        if batch_size == 0:
            raise ValueError(
                f"Empty batch detected in observation linearization (batch_size=0). "
                f"Cannot compute observation Jacobian for zero samples. "
                f"Received x.shape={x.shape}. "
                f"This usually indicates a bug in data preparation or loop logic. "
                f"Check your data loading, filtering, or vmap usage."
            )

        # Define observation function for Jacobian computation
        def observation_fn(x_i):
            x_list = [x_i[j] for j in range(self.system.nx)]
            return h_jax(*x_list)

        # Compute Jacobian using JAX autodiff (vmap for batching)
        @jax.vmap
        def compute_jacobian(x_i):
            return jax.jacobian(observation_fn)(x_i)

        C_batch = compute_jacobian(x)

        if squeeze_output:
            C_batch = jnp.squeeze(C_batch, 0)

        # Ensure correct return type
        C_batch = self.backend_mgr.ensure_type(C_batch, "jax")

        return C_batch

    # ========================================================================
    # String Representations
    # ========================================================================

    def __repr__(self) -> str:
        """String representation for debugging"""
        has_custom = self.system._h_sym is not None
        return f"ObservationEngine(ny={self.system.ny}, custom_output={has_custom})"

    def __str__(self) -> str:
        """Human-readable string"""
        return f"ObservationEngine(ny={self.system.ny})"
