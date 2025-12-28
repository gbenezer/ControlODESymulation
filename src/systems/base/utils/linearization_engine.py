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
Linearization Engine for SymbolicDynamicalSystem

Handles linearization of system dynamics across multiple backends.

Responsibilities:
- Linearized dynamics computation: A = ∂f/∂x, B = ∂f/∂u
- Symbolic linearization (state-space form for higher-order systems)
- Numerical linearization (all backends)
- Jacobian verification against autodiff
- Performance tracking
- Support for both controlled and autonomous systems

This class focuses ONLY on dynamics linearization (A, B matrices).
Output linearization (C matrix) is handled by ObservationEngine.

For autonomous systems (nu=0), B matrix has shape (nx, 0).
"""

import time
from typing import TYPE_CHECKING, Dict, Optional, Tuple, Union

import numpy as np
import sympy as sp

# Import from centralized type system
from src.types import ArrayLike
from src.types.backends import Backend
from src.types.core import ControlVector, InputMatrix, StateMatrix, StateVector
from src.types.linearization import DeterministicLinearization
from src.types.utilities import ExecutionStats

if TYPE_CHECKING:
    import jax.numpy as jnp
    import torch

    from src.systems.base.symbolic_dynamical_system import SymbolicDynamicalSystem
    from src.systems.base.utils.backend_manager import BackendManager
    from src.systems.base.utils.code_generator import CodeGenerator


class LinearizationEngine:
    """
    Computes linearized dynamics across backends.

    Handles computation of A = ∂f/∂x and B = ∂f/∂u matrices both symbolically
    and numerically, with support for higher-order systems and autonomous systems.

    For autonomous systems (nu=0):
    - B matrix has shape (nx, 0)
    - u can be None or empty array
    - Only A matrix contains meaningful information

    Example:
        >>> # Controlled system
        >>> engine = LinearizationEngine(system, code_gen, backend_mgr)
        >>> lin: DeterministicLinearization = engine.compute_dynamics(x, u, backend='numpy')
        >>> A, B = lin
        >>>
        >>> # Autonomous system
        >>> A, B = engine.compute_dynamics(x, backend='numpy')  # u=None
        >>> B.shape  # (nx, 0)
        >>>
        >>> # Symbolic linearization
        >>> A_sym, B_sym = engine.compute_symbolic(x_eq, u_eq)
        >>>
        >>> # Verify against autodiff
        >>> results = engine.verify_jacobians(x, u, backend='torch')
    """

    def __init__(
        self,
        system: "SymbolicDynamicalSystem",
        code_gen: "CodeGenerator",
        backend_mgr: "BackendManager",
    ):
        """
        Initialize linearization engine.

        Args:
            system: The dynamical system
            code_gen: Code generator for accessing Jacobian functions
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
    # Main Linearization API
    # ========================================================================

    def compute_dynamics(
        self,
        x: StateVector,
        u: Optional[ControlVector] = None,
        backend: Optional[Backend] = None,
    ) -> DeterministicLinearization:
        """
        Compute linearized dynamics: A = ∂f/∂x, B = ∂f/∂u.

        Automatically detects backend from input types.
        Supports both controlled (nu > 0) and autonomous (nu = 0) systems.

        Args:
            x: State at which to linearize
            u: Control at which to linearize (None for autonomous systems)
            backend: Backend selection (None = auto-detect)

        Returns:
            DeterministicLinearization
                Tuple of (A, B) Jacobian matrices:
                - A: StateMatrix (nx, nx) - state Jacobian ∂f/∂x
                - B: InputMatrix (nx, nu) - control Jacobian ∂f/∂u, or (nx, 0) if autonomous

        Raises:
            ValueError: If u is None for non-autonomous system

        Example:
            >>> # Controlled system
            >>> lin: DeterministicLinearization = engine.compute_dynamics(x, u)
            >>> A, B = lin  # Unpack Jacobians
            >>> A.shape  # (nx, nx)
            >>> B.shape  # (nx, nu)
            >>>
            >>> # Autonomous system (nu=0)
            >>> A, B = engine.compute_dynamics(x_np)  # u=None
            >>> A.shape  # (nx, nx)
            >>> B.shape  # (nx, 0) - empty but valid
        """
        # Handle autonomous systems
        if u is None:
            if self.system.nu > 0:
                raise ValueError(
                    "Non-autonomous system requires control input u. "
                    f"System has {self.system.nu} control input(s)."
                )
            # Create empty control for autonomous system
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
            return self._compute_dynamics_numpy(x, u)
        elif target_backend == "torch":
            return self._compute_dynamics_torch(x, u)
        elif target_backend == "jax":
            return self._compute_dynamics_jax(x, u)
        else:
            raise ValueError(f"Unknown backend: {target_backend}")

    # TODO: Symbolic TypedDict Return-Type?
    def compute_symbolic(
        self, x_eq: Optional[sp.Matrix] = None, u_eq: Optional[sp.Matrix] = None
    ) -> Tuple[sp.Matrix, sp.Matrix]:
        """
        Compute symbolic linearization A = ∂f/∂x, B = ∂f/∂u.

        For higher-order systems, constructs the full state-space linearization.
        For autonomous systems (nu=0), B is an empty matrix (nx, 0).

        Args:
            x_eq: Equilibrium state (zeros if None)
            u_eq: Equilibrium control (zeros if None, empty if autonomous)

        Returns:
            Tuple of (A, B) symbolic matrices where:
            - A: (nx, nx) state Jacobian
            - B: (nx, nu) control Jacobian, or (nx, 0) if autonomous

        Example:
            >>> # Controlled system
            >>> A_sym, B_sym = engine.compute_symbolic(
            ...     x_eq=sp.Matrix([0, 0]),
            ...     u_eq=sp.Matrix([0])
            ... )
            >>>
            >>> # Autonomous system
            >>> A_sym, B_sym = engine.compute_symbolic(
            ...     x_eq=sp.Matrix([0, 0])
            ... )  # u_eq is None/empty for autonomous
            >>> B_sym.shape  # (2, 0)
        """
        if x_eq is None:
            x_eq = sp.Matrix([0] * self.system.nx)

        # Ensure x_eq is a column matrix
        if x_eq.shape[1] != 1:
            x_eq = x_eq.reshape(self.system.nx, 1)

        if u_eq is None:
            if self.system.nu > 0:
                u_eq = sp.Matrix([0] * self.system.nu)
            else:
                u_eq = sp.Matrix(0, 1, [])  # Empty column matrix for autonomous systems

        # Ensure u_eq is a column matrix (if not empty)
        if self.system.nu > 0 and u_eq.shape[1] != 1:
            u_eq = u_eq.reshape(self.system.nu, 1)

        # Compute symbolic Jacobians (cached in CodeGenerator)
        self.code_gen._compute_symbolic_jacobians()
        A_sym_cached = self.code_gen._A_sym_cache
        B_sym_cached = self.code_gen._B_sym_cache

        if self.system.order == 1:
            # First-order system: straightforward Jacobian
            A_sym = A_sym_cached
            B_sym = B_sym_cached
        elif self.system.order == 2:
            # Second-order system: x = [q, q̇], q̈ = f(x, u)
            # Construct state-space form:
            # d/dt [q]   = [0   I] [q]   + [0]    u
            #      [q̇]     [A_q A_q̇] [q̇]    [B_accel]

            nq = self.system.nq

            # Compute Jacobians of acceleration w.r.t. q and q̇
            A_accel = self.system._f_sym.jacobian(self.system.state_vars)
            B_accel = B_sym_cached

            # Construct full state-space matrices
            A_sym = sp.zeros(self.system.nx, self.system.nx)
            A_sym[:nq, nq:] = sp.eye(nq)  # dq/dt = q̇
            A_sym[nq:, :] = A_accel  # dq̇/dt = f(q, q̇, u)

            B_sym = sp.zeros(self.system.nx, self.system.nu)
            if self.system.nu > 0:  # Only add B_accel if system has controls
                B_sym[nq:, :] = B_accel  # Control affects acceleration
            # else: B_sym remains zeros with shape (nx, 0)
        else:
            # Higher-order systems: x = [q, q', q'', ..., q^(n-1)]
            nq = self.system.nq
            order = self.system.order

            A_highest = self.system._f_sym.jacobian(self.system.state_vars)
            B_highest = B_sym_cached

            A_sym = sp.zeros(self.system.nx, self.system.nx)
            # Each derivative becomes the next one
            for i in range(order - 1):
                A_sym[i * nq : (i + 1) * nq, (i + 1) * nq : (i + 2) * nq] = sp.eye(nq)
            # Highest derivative
            A_sym[(order - 1) * nq :, :] = A_highest

            B_sym = sp.zeros(self.system.nx, self.system.nu)
            if self.system.nu > 0:  # Only add B_highest if system has controls
                B_sym[(order - 1) * nq :, :] = B_highest
            # else: B_sym remains zeros with shape (nx, 0)

        # Substitute equilibrium point
        if self.system.nu > 0:
            # Controlled system: substitute both state and control
            subs_dict = {}
            for i, var in enumerate(self.system.state_vars):
                subs_dict[var] = x_eq[i, 0]  # Extract scalar from column matrix
            for i, var in enumerate(self.system.control_vars):
                subs_dict[var] = u_eq[i, 0]  # Extract scalar from column matrix
        else:
            # Autonomous system: only substitute state variables
            subs_dict = {}
            for i, var in enumerate(self.system.state_vars):
                subs_dict[var] = x_eq[i, 0]  # Extract scalar from column matrix

        A = A_sym.subs(subs_dict)
        B = B_sym.subs(subs_dict)

        # Substitute parameters
        A = self.system.substitute_parameters(A)
        B = self.system.substitute_parameters(B)

        return A, B

    # ========================================================================
    # Backend-Specific Implementations
    # ========================================================================

    def _compute_dynamics_numpy(
        self, x: np.ndarray, u: np.ndarray
    ) -> DeterministicLinearization:
        """
        NumPy implementation using cached functions or symbolic evaluation.

        Supports both controlled (nu > 0) and autonomous (nu = 0) systems.

        Raises:
            ValueError: If batch is empty (batch_size=0)
        """
        start_time = time.time()

        # Handle batched input
        if x.ndim == 1:
            x = np.expand_dims(x, 0)
            if self.system.nu > 0:
                u = np.expand_dims(u, 0)
            else:
                # Autonomous: u should already be empty, ensure shape (1, 0)
                u = np.empty((1, 0))
            squeeze_output = True
        else:
            squeeze_output = False

        batch_size = x.shape[0]

        # Check for empty batch BEFORE processing
        if batch_size == 0:
            u_shape_str = f"{u.shape}" if self.system.nu > 0 else "None"
            raise ValueError(
                f"Empty batch detected in linearization (batch_size=0). "
                f"Cannot compute Jacobian matrices for zero samples. "
                f"Received x.shape={x.shape}, u.shape={u_shape_str}. "
                f"This usually indicates a bug in data preparation or loop logic. "
                f"Check your data loading, filtering, or iteration code."
            )

        A_batch = np.zeros((batch_size, self.system.nx, self.system.nx))
        B_batch = np.zeros((batch_size, self.system.nx, self.system.nu))  # (nx, 0) if nu=0

        # Try to get cached Jacobian functions
        A_func, B_func, _ = self.code_gen.get_jacobians("numpy")

        if A_func is not None and (B_func is not None or self.system.nu == 0):
            # Use cached functions
            for i in range(batch_size):
                x_i = x[i]

                x_list = [x_i[j] for j in range(self.system.nx)]

                if self.system.nu > 0:
                    u_i = u[i]
                    u_list = [u_i[j] for j in range(self.system.nu)]
                    all_args = x_list + u_list

                    A_result = A_func(*all_args)
                    B_result = B_func(*all_args)

                    A_batch[i] = np.array(A_result, dtype=np.float64)
                    B_batch[i] = np.array(B_result, dtype=np.float64)
                else:
                    # Autonomous: only pass state variables
                    A_result = A_func(*x_list)
                    A_batch[i] = np.array(A_result, dtype=np.float64)
                    # B_batch[i] remains zeros with shape (nx, 0)
        else:
            # Fall back to symbolic evaluation
            for i in range(batch_size):
                x_np = np.atleast_1d(x[i])

                if self.system.nu > 0:
                    u_np = np.atleast_1d(u[i])
                    A_sym, B_sym = self.compute_symbolic(
                        sp.Matrix(x_np.reshape(-1, 1)), sp.Matrix(u_np.reshape(-1, 1))
                    )
                else:
                    # Autonomous: pass None for u_eq
                    A_sym, B_sym = self.compute_symbolic(sp.Matrix(x_np.reshape(-1, 1)), u_eq=None)

                A_batch[i] = np.array(A_sym, dtype=np.float64)
                B_batch[i] = np.array(B_sym, dtype=np.float64)

        if squeeze_output:
            A_batch = np.squeeze(A_batch, 0)
            B_batch = np.squeeze(B_batch, 0)

        # Update performance stats
        self._stats["calls"] += 1
        self._stats["time"] += time.time() - start_time

        # Ensure correct return type
        A_batch = self.backend_mgr.ensure_type(A_batch, "numpy")
        B_batch = self.backend_mgr.ensure_type(B_batch, "numpy")

        return A_batch, B_batch

    def _compute_dynamics_torch(
        self, x: "torch.Tensor", u: "torch.Tensor"
    ) -> DeterministicLinearization:
        """
        PyTorch implementation using cached functions or symbolic evaluation.

        Supports both controlled (nu > 0) and autonomous (nu = 0) systems.

        Raises:
            ValueError: If batch is empty (batch_size=0)
        """
        import torch

        start_time = time.time()

        # Handle batched input
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
            if self.system.nu > 0:
                u = u.unsqueeze(0)
            else:
                # Autonomous: ensure u has shape (1, 0)
                u = torch.empty((1, 0), dtype=x.dtype, device=x.device)
            squeeze_output = True
        else:
            squeeze_output = False

        batch_size = x.shape[0]

        # Check for empty batch BEFORE processing
        if batch_size == 0:
            u_shape_str = f"{tuple(u.shape)}" if self.system.nu > 0 else "None"
            raise ValueError(
                f"Empty batch detected in linearization (batch_size=0). "
                f"Cannot compute Jacobian matrices for zero samples. "
                f"Received x.shape={tuple(x.shape)}, u.shape={u_shape_str}. "
                f"This usually indicates a bug in data preparation or loop logic. "
                f"Check your DataLoader, filtering, or iteration code."
            )

        device = x.device
        dtype = x.dtype

        A_batch = torch.zeros(
            batch_size, self.system.nx, self.system.nx, dtype=dtype, device=device
        )
        B_batch = torch.zeros(
            batch_size, self.system.nx, self.system.nu, dtype=dtype, device=device
        )

        # Try to get cached Jacobian functions
        A_func, B_func, _ = self.code_gen.get_jacobians("torch")

        if A_func is not None and (B_func is not None or self.system.nu == 0):
            # Use cached functions
            for i in range(batch_size):
                x_i = x[i]

                x_list = [x_i[j] for j in range(self.system.nx)]

                if self.system.nu > 0:
                    u_i = u[i]
                    u_list = [u_i[j] for j in range(self.system.nu)]
                    all_args = x_list + u_list

                    A_batch[i] = A_func(*all_args)
                    B_batch[i] = B_func(*all_args)
                else:
                    # Autonomous: only pass state variables
                    A_batch[i] = A_func(*x_list)
                    # B_batch[i] remains zeros with shape (nx, 0)
        else:
            # Fall back to symbolic evaluation
            for i in range(batch_size):
                x_i = x[i] if batch_size > 1 else x.squeeze(0)

                x_np = x_i.detach().cpu().numpy()
                x_np = np.atleast_1d(x_np)

                if self.system.nu > 0:
                    u_i = u[i] if batch_size > 1 else u.squeeze(0)
                    u_np = u_i.detach().cpu().numpy()
                    u_np = np.atleast_1d(u_np)

                    A_sym, B_sym = self.compute_symbolic(
                        sp.Matrix(x_np.reshape(-1, 1)), sp.Matrix(u_np.reshape(-1, 1))
                    )
                else:
                    # Autonomous: pass None for u_eq
                    A_sym, B_sym = self.compute_symbolic(sp.Matrix(x_np.reshape(-1, 1)), u_eq=None)

                A_batch[i] = torch.tensor(
                    np.array(A_sym, dtype=np.float64), dtype=dtype, device=device
                )
                B_batch[i] = torch.tensor(
                    np.array(B_sym, dtype=np.float64), dtype=dtype, device=device
                )

        if squeeze_output:
            A_batch = A_batch.squeeze(0)
            B_batch = B_batch.squeeze(0)

        # Update performance stats
        self._stats["calls"] += 1
        self._stats["time"] += time.time() - start_time

        # Ensure correct return type
        A_batch = self.backend_mgr.ensure_type(A_batch, "torch")
        B_batch = self.backend_mgr.ensure_type(B_batch, "torch")

        return A_batch, B_batch

    def _compute_dynamics_jax(
        self, x: "jnp.ndarray", u: "jnp.ndarray"
    ) -> DeterministicLinearization:
        """
        JAX implementation using automatic differentiation.

        Supports both controlled (nu > 0) and autonomous (nu = 0) systems.
        For autonomous systems, uses only state variables for differentiation.

        Raises:
            ValueError: If batch is empty (batch_size=0)
        """
        import jax
        import jax.numpy as jnp

        start_time = time.time()

        # Ensure dynamics function is available
        f_jax = self.code_gen.generate_dynamics("jax", jit=True)

        # Handle batched input
        if x.ndim == 1:
            x = jnp.expand_dims(x, 0)
            if self.system.nu > 0:
                u = jnp.expand_dims(u, 0)
            else:
                # Autonomous: ensure u has shape (1, 0)
                u = jnp.empty((1, 0))
            squeeze_output = True
        else:
            squeeze_output = False

        batch_size = x.shape[0]

        # Check for empty batch BEFORE processing
        if batch_size == 0:
            u_shape_str = f"{u.shape}" if self.system.nu > 0 else "None"
            raise ValueError(
                f"Empty batch detected in linearization (batch_size=0). "
                f"Cannot compute Jacobian matrices for zero samples. "
                f"Received x.shape={x.shape}, u.shape={u_shape_str}. "
                f"This usually indicates a bug in data preparation or loop logic. "
                f"Check your data loading, filtering, or vmap usage."
            )

        # Define dynamics function for Jacobian computation
        def dynamics_fn(x_i, u_i):
            x_list = [x_i[j] for j in range(self.system.nx)]
            u_list = [u_i[j] for j in range(self.system.nu)]  # Empty if nu=0
            return f_jax(*(x_list + u_list))

        # Compute Jacobians using JAX autodiff (vmap for batching)
        if self.system.nu > 0:
            # Non-autonomous: compute both A and B
            @jax.vmap
            def compute_jacobians(x_i, u_i):
                A = jax.jacobian(lambda x: dynamics_fn(x, u_i))(x_i)
                B = jax.jacobian(lambda u: dynamics_fn(x_i, u))(u_i)
                return A, B

            A_batch, B_batch = compute_jacobians(x, u)
        else:
            # Autonomous: only compute A (B is empty)
            @jax.vmap
            def compute_jacobian_A(x_i, u_i):
                A = jax.jacobian(lambda x: dynamics_fn(x, u_i))(x_i)
                return A

            A_batch = compute_jacobian_A(x, u)
            # B is empty matrix (batch, nx, 0)
            B_batch = jnp.empty((x.shape[0], self.system.nx, 0))

        if squeeze_output:
            A_batch = jnp.squeeze(A_batch, 0)
            B_batch = jnp.squeeze(B_batch, 0)

        # Update performance stats
        self._stats["calls"] += 1
        self._stats["time"] += time.time() - start_time

        # Ensure correct return type
        A_batch = self.backend_mgr.ensure_type(A_batch, "jax")
        B_batch = self.backend_mgr.ensure_type(B_batch, "jax")

        return A_batch, B_batch

    # ========================================================================
    # Jacobian Verification
    # ========================================================================

    # TODO: TypedDict for JacobianVerification?
    def verify_jacobians(
        self,
        x: StateVector,
        u: Optional[ControlVector] = None,
        backend: Backend = "torch",
        tol: float = 1e-4
    ) -> Dict[str, Union[bool, float]]:
        """
        Verify symbolic Jacobians against automatic differentiation.

        Uses autodiff to numerically compute Jacobians and compares against
        symbolic derivation. Requires autodiff backend (torch or jax).

        Args:
            x: State at which to verify
            u: Control at which to verify (None for autonomous systems)
            backend: Backend for autodiff ('torch' or 'jax', not 'numpy')
            tol: Tolerance for considering Jacobians equal

        Returns:
            Dict with 'A_match', 'B_match' booleans and error magnitudes

        Raises:
            ValueError: If backend doesn't support autodiff or u is None for non-autonomous

        Example:
            >>> # Controlled system
            >>> results = engine.verify_jacobians(x, u, backend='torch', tol=1e-4)
            >>> assert results['A_match'] is True
            >>> assert results['B_match'] is True
            >>>
            >>> # Autonomous system
            >>> results = engine.verify_jacobians(x, backend='torch')  # u=None
            >>> assert results['A_match'] is True
            >>> # B_match will be True trivially for empty matrix
        """
        if backend not in ["torch", "jax"]:
            raise ValueError(
                f"Jacobian verification requires autodiff backend ('torch' or 'jax'), "
                f"got '{backend}'. NumPy doesn't support automatic differentiation."
            )

        # Handle autonomous systems
        if u is None:
            if self.system.nu > 0:
                raise ValueError("Non-autonomous system requires control input u")
            # Create empty control
            if backend == "torch":
                import torch

                u = torch.tensor([])
            else:  # jax
                import jax.numpy as jnp

                u = jnp.array([])

        # Check backend availability
        self.backend_mgr.require_backend(backend)

        # Dispatch to backend-specific verification
        if backend == "torch":
            return self._verify_jacobians_torch(x, u, tol)
        else:  # jax
            return self._verify_jacobians_jax(x, u, tol)

    def _verify_jacobians_torch(
        self, x: ArrayLike, u: ArrayLike, tol: float
    ) -> Dict[str, Union[bool, float]]:
        """
        PyTorch-based Jacobian verification.

        Supports both controlled and autonomous systems.
        """
        import torch

        # Convert to torch if needed
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(np.asarray(x), dtype=torch.float32)
        if not isinstance(u, torch.Tensor):
            u = torch.tensor(np.asarray(u), dtype=torch.float32)

        # Ensure proper 2D shape
        x_2d = x.reshape(1, -1) if len(x.shape) <= 1 else x

        if self.system.nu > 0:
            u_2d = u.reshape(1, -1) if len(u.shape) <= 1 else u
        else:
            # Autonomous: u should be empty tensor
            u_2d = u.reshape(1, 0) if u.numel() == 0 else u

        # Clone for autograd
        x_grad = x_2d.clone().requires_grad_(True)

        if self.system.nu > 0:
            u_grad = u_2d.clone().requires_grad_(True)
        else:
            u_grad = u_2d  # No gradients needed for empty tensor

        # Compute symbolic Jacobians
        A_sym, B_sym = self.compute_dynamics(x_2d.detach(), u_2d.detach(), backend="torch")

        # Ensure 3D shape for batch processing
        if len(A_sym.shape) == 2:
            A_sym = A_sym.unsqueeze(0)
            B_sym = B_sym.unsqueeze(0)

        # Compute numerical Jacobians via autograd
        from src.systems.base.utils.dynamics_evaluator import DynamicsEvaluator

        dynamics_eval = DynamicsEvaluator(self.system, self.code_gen, self.backend_mgr)
        fx = dynamics_eval.evaluate(x_grad, u_grad, backend="torch")

        # Ensure fx is 2D for consistent indexing
        if fx.ndim == 1:
            fx = fx.unsqueeze(1)
        elif fx.ndim == 0:
            fx = fx.reshape(1, 1)

        # Determine output dimension
        n_outputs = self.system.nq if self.system.order > 1 else self.system.nx

        # Compute gradients
        A_num = torch.zeros_like(A_sym)
        B_num = torch.zeros_like(B_sym)

        if self.system.order == 1:
            # First-order: verify full A and B
            for i in range(n_outputs):
                if fx[0, i].requires_grad:
                    grad_x = torch.autograd.grad(
                        fx[0, i], x_grad, retain_graph=True, create_graph=False
                    )[0]
                    A_num[0, i] = grad_x[0]

                    # Only compute B gradient for non-autonomous
                    if self.system.nu > 0:
                        grad_u = torch.autograd.grad(
                            fx[0, i], u_grad, retain_graph=True, create_graph=False
                        )[0]
                        B_num[0, i] = grad_u[0]
        else:
            # Higher-order: verify acceleration part
            for i in range(n_outputs):
                if fx[0, i].requires_grad:
                    grad_x = torch.autograd.grad(
                        fx[0, i], x_grad, retain_graph=True, create_graph=False
                    )[0]

                    row_idx = (self.system.order - 1) * self.system.nq + i
                    A_num[0, row_idx] = grad_x[0]

                    # Only compute B gradient for non-autonomous
                    if self.system.nu > 0:
                        grad_u = torch.autograd.grad(
                            fx[0, i], u_grad, retain_graph=True, create_graph=False
                        )[0]
                        B_num[0, row_idx] = grad_u[0]

            # Copy derivative relationships
            for i in range((self.system.order - 1) * self.system.nq):
                A_num[0, i] = A_sym[0, i]
                if self.system.nu > 0:
                    B_num[0, i] = B_sym[0, i]

        # Compute errors
        A_error = (A_sym - A_num).abs().max().item()

        if self.system.nu > 0:
            B_error = (B_sym - B_num).abs().max().item()
        else:
            # Autonomous: B is empty, error is trivially zero
            B_error = 0.0

        return {
            "A_match": bool(A_error < tol),
            "B_match": bool(B_error < tol),
            "A_error": float(A_error),
            "B_error": float(B_error),
        }

    def _verify_jacobians_jax(
        self, x: ArrayLike, u: ArrayLike, tol: float
    ) -> Dict[str, Union[bool, float]]:
        """
        JAX-based Jacobian verification.

        Supports both controlled and autonomous systems.
        """
        import jax.numpy as jnp

        # Convert to JAX if needed
        if not isinstance(x, jnp.ndarray):
            x = jnp.array(np.asarray(x))
        if not isinstance(u, jnp.ndarray):
            u = jnp.array(np.asarray(u))

        # Ensure proper shape
        x_2d = x.reshape(1, -1) if x.ndim <= 1 else x

        if self.system.nu > 0:
            u_2d = u.reshape(1, -1) if u.ndim <= 1 else u
        else:
            # Autonomous: u should be empty
            u_2d = u.reshape(1, 0) if u.size == 0 else u

        # Compute Jacobians using JAX autodiff
        A_jax, B_jax = self.compute_dynamics(x_2d, u_2d, backend="jax")

        # Compute symbolic Jacobians as ground truth
        x_np = np.array(x_2d[0])

        if self.system.nu > 0:
            u_np = np.array(u_2d[0])
            A_sym, B_sym = self.compute_symbolic(
                sp.Matrix(x_np.reshape(-1, 1)), sp.Matrix(u_np.reshape(-1, 1))
            )
        else:
            # Autonomous: pass None for u_eq
            A_sym, B_sym = self.compute_symbolic(sp.Matrix(x_np.reshape(-1, 1)), u_eq=None)

        A_sym_np = np.array(A_sym, dtype=np.float64)
        B_sym_np = np.array(B_sym, dtype=np.float64)

        # Convert JAX results to NumPy for comparison
        A_jax_np = np.array(A_jax)
        B_jax_np = np.array(B_jax)

        # Compute errors
        A_error = np.abs(A_sym_np - A_jax_np).max()

        if self.system.nu > 0:
            B_error = np.abs(B_sym_np - B_jax_np).max()
        else:
            # Autonomous: B is empty, error is trivially zero
            B_error = 0.0

        return {
            "A_match": bool(A_error < tol),
            "B_match": bool(B_error < tol),
            "A_error": float(A_error),
            "B_error": float(B_error),
        }

    # ========================================================================
    # Performance Tracking
    # ========================================================================

    def get_stats(self) -> ExecutionStats:
        """
        Get performance statistics.

        Returns:
            Dict with call count, total time, and average time
        """
        return {
            "calls": self._stats["calls"],
            "total_time": self._stats["time"],
            "avg_time": self._stats["time"] / max(1, self._stats["calls"]),
        }

    def reset_stats(self):
        """Reset performance counters."""
        self._stats["calls"] = 0
        self._stats["time"] = 0.0

    # ========================================================================
    # String Representations
    # ========================================================================

    def __repr__(self) -> str:
        """String representation for debugging"""
        autonomous_str = " (autonomous)" if self.system.nu == 0 else ""
        return (
            f"LinearizationEngine("
            f"nx={self.system.nx}, nu={self.system.nu}{autonomous_str}, "
            f"calls={self._stats['calls']})"
        )

    def __str__(self) -> str:
        """Human-readable string"""
        return f"LinearizationEngine(calls={self._stats['calls']})"
