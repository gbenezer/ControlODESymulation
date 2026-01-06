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
Dynamics Evaluator for SymbolicDynamicalSystem

Handles forward dynamics evaluation across multiple backends.

Responsibilities:
- Forward dynamics evaluation: dx/dt = f(x, u) or dx/dt = f(x) for autonomous
- Backend-specific implementations (NumPy, PyTorch, JAX)
- Input validation and shape handling
- Batched vs single evaluation
- Performance tracking
- Backend dispatch
- Support for both controlled and autonomous systems
- Automatic type conversion for control inputs (handles None, numpy arrays, etc.)

IMPORTANT: Backend-specific methods (_evaluate_torch, _evaluate_jax) now include
automatic type conversion for control inputs. This is critical for SDE integration
and other cases where control functions may return numpy arrays instead of
backend-native tensors.

This class manages the evaluation of the system dynamics using
generated functions from CodeGenerator.
"""

import time
from typing import TYPE_CHECKING, Optional

import numpy as np

# Import from centralized type system
from cdesym.types.backends import Backend
from cdesym.types.core import ControlVector, StateVector
from cdesym.types.utilities import ExecutionStats, get_batch_size, is_batched

if TYPE_CHECKING:
    import jax.numpy as jnp
    import torch

    from cdesym.systems.base.symbolic_dynamical_system import SymbolicDynamicalSystem
    from cdesym.systems.base.utils.backend_manager import BackendManager
    from cdesym.systems.base.utils.code_generator import CodeGenerator


class DynamicsEvaluator:
    """
    Evaluates forward dynamics across backends.

    Handles the evaluation of dx/dt = f(x, u) for controlled systems or
    dx/dt = f(x) for autonomous systems. Supports NumPy, PyTorch, and JAX
    backends with proper shape handling, batching, and performance tracking.

    Type System Integration:
        - StateVector: Input state and output derivative
        - ControlVector: Input control (Optional for autonomous)
        - Backend: Type-safe backend selection
        - ExecutionStats: Structured performance metrics

    Batching:
        Supports both single and batched evaluation using centralized
        utilities from the type framework (is_batched, get_batch_size).

    Example:
        >>> # Controlled system
        >>> evaluator = DynamicsEvaluator(system, code_gen, backend_mgr)
        >>> dx: StateVector = evaluator.evaluate(x, u, backend='numpy')
        >>>
        >>> # Autonomous system (u=None)
        >>> dx: StateVector = evaluator.evaluate(x, backend='numpy')
        >>>
        >>> # Get performance stats
        >>> stats: ExecutionStats = evaluator.get_stats()
        >>> print(f"Average time: {stats['avg_time']:.6f}s")
    """

    def __init__(
        self,
        system: "SymbolicDynamicalSystem",
        code_gen: "CodeGenerator",
        backend_mgr: "BackendManager",
    ):
        """
        Initialize dynamics evaluator.

        Args:
            system: The dynamical system
            code_gen: Code generator for accessing compiled functions
            backend_mgr: Backend manager for detection/conversion
        """
        self.system = system
        self.code_gen = code_gen
        self.backend_mgr = backend_mgr

        # Performance tracking
        self._stats = {
            "calls": 0,
            "time": 0.0,
        }

    # ========================================================================
    # Main Evaluation API
    # ========================================================================

    def evaluate(
        self,
        x: StateVector,
        u: Optional[ControlVector] = None,
        backend: Optional[Backend] = None,
    ) -> StateVector:
        """
        Evaluate forward dynamics: dx/dt = f(x, u) or dx/dt = f(x).

        Args:
            x: State vector
            u: Control vector (None for autonomous systems)
            backend: Backend selection:
                - None: Auto-detect from input type (default)
                - 'numpy', 'torch', 'jax': Force specific backend
                - 'default': Use system's default backend

        Returns:
            State derivative vector (type matches backend)

        Raises:
            ValueError: If u is None for non-autonomous system

        Example:
            >>> # Controlled system - auto-detect backend
            >>> dx = evaluator.evaluate(x_numpy, u_numpy)  # Returns NumPy
            >>>
            >>> # Autonomous system
            >>> dx = evaluator.evaluate(x_numpy)  # u=None
            >>>
            >>> # Force specific backend (converts input)
            >>> dx = evaluator.evaluate(x_numpy, u_numpy, backend='torch')  # Returns PyTorch
        """
        # Handle autonomous systems
        if u is None:
            if self.system.nu > 0:
                raise ValueError(
                    f"Non-autonomous system requires control input u. "
                    f"System has {self.system.nu} control input(s).",
                )
            # Create empty control array in appropriate backend
            if backend == "default":
                target_backend = self.backend_mgr.default_backend
            elif backend is None:
                target_backend = self.backend_mgr.detect(x)
            else:
                target_backend = backend

            # Create empty array in appropriate backend
            if target_backend == "numpy":
                u = np.array([])
            elif target_backend == "torch":
                import torch

                u = torch.tensor([], dtype=torch.float32)
            elif target_backend == "jax":
                import jax.numpy as jnp

                u = jnp.array([])

        # Determine target backend
        if backend == "default":
            target_backend = self.backend_mgr.default_backend
        elif backend is None:
            target_backend = self.backend_mgr.detect(x)
        else:
            target_backend = backend

        # Convert inputs if needed
        input_backend = self.backend_mgr.detect(x)
        if input_backend != target_backend:
            x = self.backend_mgr.convert(x, target_backend)
            if self.system.nu > 0:  # Only convert u if not autonomous
                u = self.backend_mgr.convert(u, target_backend)

        # Dispatch to backend-specific implementation
        if target_backend == "numpy":
            return self._evaluate_numpy(x, u)
        if target_backend == "torch":
            return self._evaluate_torch(x, u)
        if target_backend == "jax":
            return self._evaluate_jax(x, u)
        raise ValueError(f"Unknown backend: {target_backend}")

    # ========================================================================
    # Backend-Specific Implementations
    # ========================================================================

    def _evaluate_numpy(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        NumPy backend implementation.

        Uses centralized batching utilities for consistent shape handling.
        Supports both controlled (nu > 0) and autonomous (nu = 0) systems.

        Raises:
            ValueError: If batch is empty (batch_size=0)
        """
        start_time = time.time()

        # Input validation
        if x.ndim == 0:
            raise ValueError("State array must be at least 1D")

        if x.ndim >= 1 and x.shape[-1] != self.system.nx:
            raise ValueError(f"Expected state dimension {self.system.nx}, got {x.shape[-1]}")

        # Check for empty batch using batching utilities
        if is_batched(x):
            batch_size = get_batch_size(x)
            if batch_size == 0:
                u_shape_str = f"{u.shape}" if u is not None and u.size > 0 else "None"
                raise ValueError(
                    f"Empty batch detected (batch_size=0). "
                    f"System evaluation requires at least one sample. "
                    f"Received x.shape={x.shape}, u.shape={u_shape_str}. "
                    f"This usually indicates a bug in data preparation or loop logic. "
                    f"Check your data loading, filtering, or iteration code.",
                )

        # For autonomous systems, u should be empty
        if self.system.nu == 0:
            if u.size != 0:
                raise ValueError("Autonomous system does not accept control input")
        else:
            # Non-autonomous: validate u
            if u.ndim == 0:
                raise ValueError("Control array must be at least 1D")
            if u.ndim >= 1 and u.shape[-1] != self.system.nu:
                raise ValueError(f"Expected control dimension {self.system.nu}, got {u.shape[-1]}")

            # Check for mismatched batch sizes using batching utilities
            if is_batched(x) and is_batched(u):
                batch_size_x = get_batch_size(x)
                batch_size_u = get_batch_size(u)
                if batch_size_x != batch_size_u:
                    raise ValueError(
                        f"Batch size mismatch: x has {batch_size_x} samples, "
                        f"u has {batch_size_u} samples",
                    )

        # Generate function (uses cache if available)
        f_numpy = self.code_gen.generate_dynamics("numpy")

        # Handle batched vs single evaluation using batching utilities
        if not is_batched(x):
            # Single evaluation
            x_list = [x[i] for i in range(self.system.nx)]

            if self.system.nu > 0:
                u_list = [u[i] for i in range(self.system.nu)]
                result = f_numpy(*(x_list + u_list))
            else:
                # Autonomous: no control inputs
                result = f_numpy(*x_list)

            result = np.array(result).flatten()
        else:
            # Batched evaluation
            batch_size = get_batch_size(x)
            results = []
            for i in range(batch_size):
                x_list = [x[i, j] for j in range(self.system.nx)]

                if self.system.nu > 0:
                    u_list = [u[i, j] for j in range(self.system.nu)]
                    result = f_numpy(*(x_list + u_list))
                else:
                    # Autonomous: no control inputs
                    result = f_numpy(*x_list)

                results.append(np.array(result).flatten())

            # Defensive check (should never happen after batch_size check above)
            if len(results) == 0:
                raise RuntimeError(
                    "Internal error: No results generated despite non-empty input validation. "
                    "This is a bug in the dynamics evaluator - please report this.",
                )

            result = np.stack(results)

        # Update performance stats
        self._stats["calls"] += 1
        self._stats["time"] += time.time() - start_time

        # Ensure correct backend type
        result = self.backend_mgr.ensure_type(result, "numpy")

        return result

    def _evaluate_torch(self, x: "torch.Tensor", u: "torch.Tensor") -> "torch.Tensor":
        """
        PyTorch backend implementation.

        Handles both single and batched evaluation with GPU support.
        Supports both controlled (nu > 0) and autonomous (nu = 0) systems.

        Raises:
            ValueError: If batch is empty (batch_size=0)
        """

        start_time = time.time()

        # CRITICAL FIX: Convert inputs to torch tensors if needed
        import torch
        import numpy as np
        
        # Convert x if it's a numpy array - PRESERVE DTYPE
        if isinstance(x, np.ndarray):
            # Preserve dtype: float64 → float64, float32 → float32
            if x.dtype == np.float64:
                x = torch.from_numpy(x).double()  # Use .double() for float64
            else:
                x = torch.from_numpy(x).float()   # Use .float() for float32
        
        # Convert u if it's a numpy array, or create empty tensor if None
        if u is None:
            # Autonomous system: create empty tensor matching x's dtype and device
            u = torch.tensor([], dtype=x.dtype, device=x.device)
        elif isinstance(u, np.ndarray):
            # Convert numpy array to torch tensor with same dtype and device as x
            u_torch = torch.from_numpy(u)
            # Match x's dtype and device
            u = u_torch.to(dtype=x.dtype, device=x.device)

        # Input validation
        if len(x.shape) == 0:
            raise ValueError("State tensor must be at least 1D")

        if len(x.shape) >= 1 and x.shape[-1] != self.system.nx:
            raise ValueError(f"Expected state dimension {self.system.nx}, got {x.shape[-1]}")

        # Check for empty batch using batching utilities
        if is_batched(x):
            batch_size = get_batch_size(x)
            if batch_size == 0:
                u_shape_str = f"{tuple(u.shape)}" if u is not None and u.numel() > 0 else "None"
                raise ValueError(
                    f"Empty batch detected (batch_size=0). "
                    f"System evaluation requires at least one sample. "
                    f"Received x.shape={tuple(x.shape)}, u.shape={u_shape_str}. "
                    f"This usually indicates a bug in data preparation or loop logic. "
                    f"Check your DataLoader, filtering, or iteration code.",
                )

        # For autonomous systems, u should be empty
        if self.system.nu == 0:
            if u.numel() != 0:
                raise ValueError("Autonomous system does not accept control input")
        else:
            # Non-autonomous: validate u
            if len(u.shape) == 0:
                raise ValueError("Control tensor must be at least 1D")
            if len(u.shape) >= 1 and u.shape[-1] != self.system.nu:
                raise ValueError(f"Expected control dimension {self.system.nu}, got {u.shape[-1]}")

            # Check for mismatched batch sizes using batching utilities
            if is_batched(x) and is_batched(u):
                batch_size_x = get_batch_size(x)
                batch_size_u = get_batch_size(u)
                if batch_size_x != batch_size_u:
                    raise ValueError(
                        f"Batch size mismatch: x has {batch_size_x} samples, "
                        f"u has {batch_size_u} samples",
                    )

        # Generate function (uses cache if available)
        f_torch = self.code_gen.generate_dynamics("torch")

        # Handle batched vs single evaluation using batching utilities
        if not is_batched(x):
            x = x.unsqueeze(0)
            if self.system.nu > 0:
                u = u.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        # Prepare arguments
        x_list = [x[:, i] for i in range(self.system.nx)]

        if self.system.nu > 0:
            u_list = [u[:, i] for i in range(self.system.nu)]
            all_args = x_list + u_list
        else:
            # Autonomous: only state arguments
            all_args = x_list

        # Call generated function
        result = f_torch(*all_args)

        # Handle output shape
        if squeeze_output:
            # Single input case - squeeze batch dimension
            result = result.squeeze(0)

            # Ensure at least 1D
            if result.ndim == 0:
                result = result.unsqueeze(0)
        # Batched input case - ensure proper 2D shape (batch, nq)
        elif result.ndim == 1:
            # If result is (batch,), reshape to (batch, 1) for single output systems
            if self.system.order > 1 or self.system.nx == 1:
                result = result.unsqueeze(1)

        # Update performance stats
        self._stats["calls"] += 1
        self._stats["time"] += time.time() - start_time

        # Ensure correct backend type
        result = self.backend_mgr.ensure_type(result, "torch")

        return result

    def _evaluate_jax(self, x: "jnp.ndarray", u: "jnp.ndarray") -> "jnp.ndarray":
        """
        JAX backend implementation.

        Handles both single and batched evaluation with vmap for efficiency.
        Supports both controlled (nu > 0) and autonomous (nu = 0) systems.

        Raises:
            ValueError: If batch is empty (batch_size=0)
        """
        import jax
        import jax.numpy as jnp
        import numpy as np

        start_time = time.time()

        # CRITICAL FIX: Convert inputs to jax arrays if needed
        # This handles cases where control functions return numpy arrays or None
        if isinstance(x, np.ndarray):
            x = jnp.array(x)  # JAX preserves dtype automatically
        
        # Convert u if it's a numpy array, or create empty array if None
        if u is None:
            # Autonomous system: create empty array with same dtype as x
            u = jnp.array([], dtype=x.dtype)
        elif isinstance(u, np.ndarray):
            # Convert numpy array to jax array (preserves dtype)
            u = jnp.array(u)

        # Input validation
        if x.ndim == 0:
            raise ValueError("State array must be at least 1D")

        if x.ndim >= 1 and x.shape[-1] != self.system.nx:
            raise ValueError(f"Expected state dimension {self.system.nx}, got {x.shape[-1]}")

        # Check for empty batch using batching utilities
        if is_batched(x):
            batch_size = get_batch_size(x)
            if batch_size == 0:
                u_shape_str = f"{u.shape}" if u is not None and u.size > 0 else "None"
                raise ValueError(
                    f"Empty batch detected (batch_size=0). "
                    f"System evaluation requires at least one sample. "
                    f"Received x.shape={x.shape}, u.shape={u_shape_str}. "
                    f"This usually indicates a bug in data preparation or loop logic. "
                    f"Check your data loading, filtering, or vmap usage.",
                )

        # For autonomous systems, u should be empty
        if self.system.nu == 0:
            if u.size != 0:
                raise ValueError("Autonomous system does not accept control input")
        else:
            # Non-autonomous: validate u
            if u.ndim == 0:
                raise ValueError("Control array must be at least 1D")
            if u.ndim >= 1 and u.shape[-1] != self.system.nu:
                raise ValueError(f"Expected control dimension {self.system.nu}, got {u.shape[-1]}")

            # Check for mismatched batch sizes using batching utilities
            if is_batched(x) and is_batched(u):
                batch_size_x = get_batch_size(x)
                batch_size_u = get_batch_size(u)
                if batch_size_x != batch_size_u:
                    raise ValueError(
                        f"Batch size mismatch: x has {batch_size_x} samples, "
                        f"u has {batch_size_u} samples",
                    )

        # Generate function (uses cache if available)
        f_jax = self.code_gen.generate_dynamics("jax", jit=True)

        # Handle batched vs single evaluation using batching utilities
        if not is_batched(x):
            x = jnp.expand_dims(x, 0)
            if self.system.nu > 0:
                u = jnp.expand_dims(u, 0)
            squeeze_output = True
        else:
            squeeze_output = False

        # For batched computation, use vmap
        batch_size = get_batch_size(x) if is_batched(x) else 1
        if batch_size > 1:
            if self.system.nu > 0:

                @jax.vmap
                def batched_dynamics(x_i, u_i):
                    x_list = [x_i[j] for j in range(self.system.nx)]
                    u_list = [u_i[j] for j in range(self.system.nu)]
                    return f_jax(*(x_list + u_list))

                result = batched_dynamics(x, u)
            else:
                # Autonomous: no control inputs
                @jax.vmap
                def batched_dynamics(x_i):
                    x_list = [x_i[j] for j in range(self.system.nx)]
                    return f_jax(*x_list)

                result = batched_dynamics(x)
        else:
            # Single evaluation
            x_list = [x[0, i] for i in range(self.system.nx)]

            if self.system.nu > 0:
                u_list = [u[0, i] for i in range(self.system.nu)]
                result = f_jax(*(x_list + u_list))
            else:
                # Autonomous: no control inputs
                result = f_jax(*x_list)

            result = jnp.expand_dims(result, 0)

        # Handle output shape
        if squeeze_output:
            result = result.squeeze(0)

            # Ensure at least 1D
            if result.ndim == 0:
                result = jnp.expand_dims(result, 0)
        # Batched case - ensure proper 2D shape
        elif result.ndim == 1:
            if self.system.order > 1 or self.system.nx == 1:
                result = jnp.expand_dims(result, 1)

        # Update performance stats
        self._stats["calls"] += 1
        self._stats["time"] += time.time() - start_time

        # Ensure correct backend type
        result = self.backend_mgr.ensure_type(result, "jax")

        return result

    # ========================================================================
    # Performance Tracking
    # ========================================================================

    def get_stats(self) -> ExecutionStats:
        """
        Get performance statistics.

        Returns:
            ExecutionStats
                Structured performance metrics with call count and timing

        Example:
            >>> stats: ExecutionStats = evaluator.get_stats()
            >>> print(f"Calls: {stats['calls']}")
            >>> print(f"Avg time: {stats['avg_time']:.6f}s")
        """
        return {
            "calls": self._stats["calls"],
            "total_time": self._stats["time"],
            "avg_time": self._stats["time"] / max(1, self._stats["calls"]),
        }

    def reset_stats(self):
        """
        Reset performance counters.

        Example:
            >>> evaluator.reset_stats()
            >>> # Stats are now zero
        """
        self._stats["calls"] = 0
        self._stats["time"] = 0.0

    # ========================================================================
    # String Representations
    # ========================================================================

    def __repr__(self) -> str:
        """String representation for debugging"""
        autonomous_str = " (autonomous)" if self.system.nu == 0 else ""
        return (
            f"DynamicsEvaluator("
            f"nx={self.system.nx}, nu={self.system.nu}{autonomous_str}, "
            f"calls={self._stats['calls']})"
        )

    def __str__(self) -> str:
        """Human-readable string"""
        return f"DynamicsEvaluator(calls={self._stats['calls']})"
