import copy
from contextlib import contextmanager
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Union, Optional, Tuple, List, Dict
import json
import time
import sympy as sp
import numpy as np

# TODO:
# refactor linearization into LinearizationEngine and
# output/observation into ObservationEngine
# decide whether to put linearized observation into
# LinearizationEngine or ObservationEngine

# necessary sub-object import
from src.systems.base.equilibrium_handler import EquilibriumHandler
from src.systems.base.backend_manager import BackendManager
from src.systems.base.symbolic_validator import SymbolicValidator
from src.systems.base.code_generator import CodeGenerator
from src.systems.base.dynamics_evaluator import DynamicsEvaluator

if TYPE_CHECKING:
    import torch
    import jax.numpy as jnp

# Type alias for backend-agnostic arrays
ArrayLike = Union[np.ndarray, "torch.Tensor", "jnp.ndarray"]


class SymbolicDynamicalSystem(ABC):
    """
    Symbolic dynamical system with multi-backend execution.

    Core responsibilities:
    - Symbolic definition
    - Code generation
    - Multi-backend dispatch
    """

    def __init__(self, *args, **kwargs):
        # Symbolic definition
        self.state_vars: List[sp.Symbol] = []
        self.control_vars: List[sp.Symbol] = []
        self.output_vars: List[sp.Symbol] = []
        self.parameters: Dict[sp.Symbol, float] = {}
        self._f_sym: Optional[sp.Matrix] = None
        self._h_sym: Optional[sp.Matrix] = None
        self.order: int = 1

        # COMPOSITION: Delegate backend management
        self.backend = BackendManager(default_backend="numpy", default_device="cpu")

        # Backend configuration (to be deprecated)
        self._default_backend = "numpy"
        self._preferred_device = "cpu"

        # COMPOSITION: Delegate validation
        self._validator = SymbolicValidator(strict=True)

        # COMPOSITION: Delegate code generation (MUST be after validator init)
        # (Will be properly initialized after define_system)
        self._code_gen: Optional[CodeGenerator] = None

        # NOTE: _initialized is set to False initially, then True after validation
        self._initialized: bool = False

        # Call template method
        self.define_system(*args, **kwargs)

        # Validate using validator component
        self._validator.validate(self)

        # Mark as initialized AFTER successful validation
        self._initialized = True

        # Initialize code generator AFTER validation
        self._code_gen = CodeGenerator(self)

        # COMPOSITION: Delegate dynamics evaluation
        self._dynamics = DynamicsEvaluator(self, self._code_gen, self.backend)

        # COMPOSITION: Delegate equilibrium management
        self.equilibria = EquilibriumHandler(self.nx, self.nu)

        # for backward compatibility
        self._perf_stats = {
            "linearization_calls": 0,
            "linearization_time": 0.0,
        }

    def __repr__(self) -> str:
        """Detailed string representation for debugging"""
        return (
            f"{self.__class__.__name__}("
            f"nx={self.nx}, nu={self.nu}, ny={self.ny}, order={self.order}, "
            f"backend={self._default_backend}, device={self._preferred_device})"
        )

    def __str__(self) -> str:
        """Human-readable string representation"""
        equilibria_str = (
            f", {len(self.equilibria.list_names())} equilibria"
            if len(self.equilibria.list_names()) > 1
            else ""
        )
        return (
            f"{self.__class__.__name__}(nx={self.nx}, nu={self.nu}, "
            f"backend={self._default_backend}{equilibria_str})"
        )

    @abstractmethod
    def define_system(self, *args, **kwargs):
        """
        Define the symbolic system. Must set:
        - self.state_vars: List of state symbols
        - self.control_vars: List of control symbols
        - self.parameters: Dict with Symbol keys (not strings!)
        - self._f_sym: Symbolic dynamics matrix

        Technically optional settings:
        - self.output_vars: List of output symbols (optional, defaults to full state)
        - self._h_sym: Symbolic output matrix (optional, defaults to full state)
        - self.order: System order (default: 1)

        CRITICAL: self.parameters must use SymPy Symbol objects as keys!
        Example: {m: 1.0, l: 0.5} NOT {'m': 1.0, 'l': 0.5}

        Args:
            *args, **kwargs: System-specific parameters
        """
        pass

    # Properties
    @property
    def nx(self) -> int:
        """Number of states"""
        return len(self.state_vars)

    @property
    def nu(self) -> int:
        """Number of controls"""
        return len(self.control_vars)

    @property
    def ny(self) -> int:
        """Number of outputs"""
        if self.output_vars:
            return len(self.output_vars)
        elif self._h_sym is not None:
            return self._h_sym.shape[0]
        else:
            return self.nx

    @property
    def nq(self) -> int:
        """Number of generalized coordinates (for higher-order systems)"""
        return self.nx // self.order if self.order > 1 else self.nx

    @property
    def _default_backend(self):
        """Backward compatibility: redirect to backend manager"""
        return self.backend.default_backend

    @_default_backend.setter
    def _default_backend(self, value):
        """Backward compatibility: redirect to backend manager"""
        self.backend.set_default(value)

    @property
    def _preferred_device(self):
        """Backward compatibility: redirect to backend manager"""
        return self.backend.preferred_device

    @_preferred_device.setter
    def _preferred_device(self, value):
        """Backward compatibility: redirect to backend manager"""
        self.backend.to_device(value)

    # Utility functions
    def set_default_backend(self, backend: str, device: Optional[str] = None):
        """Set default backend for this system."""
        self.backend.set_default(backend, device)
        return self

    def _detect_backend(self, x) -> str:
        """Detect backend from input type"""
        return self.backend.detect(x)

    def _check_backend_available(self, backend: str):
        """Raise error if backend not available"""
        self.backend.require_backend(backend)

    def _convert_to_backend(self, arr: ArrayLike, backend: str):
        """Convert array to target backend"""
        return self.backend.convert(arr, backend)

    def to_device(self, device: str) -> "SymbolicDynamicalSystem":
        """Set preferred device for PyTorch/JAX backends."""
        self.backend.to_device(device)

        # Clear cached functions that need recompilation for new device
        if self.backend.default_backend in ["torch", "jax"]:
            self._clear_backend_cache(self.backend.default_backend)

        return self

    def _clear_backend_cache(self, backend: str):
        """Clear cached functions for a backend"""
        self._code_gen.reset_cache([backend])

    def _dispatch_to_backend(self, method_prefix: str, backend: Optional[str], *args):
        """
        Generic backend dispatcher.

        Args:
            method_prefix: Method name prefix (e.g., '_forward', '_h', '_linearized_dynamics')
            backend: Backend selection
            *args: Arguments to pass to backend method

        Returns:
            Result from backend-specific method
        """
        if backend == "default":
            target_backend = self.backend.default_backend
        elif backend is None:
            target_backend = self.backend.detect(args[0])
        else:
            target_backend = backend

        # Check availability
        self.backend.require_backend(target_backend)

        # Get backend-specific method
        method_name = f"{method_prefix}_{target_backend}"
        method = getattr(self, method_name)

        # Call it
        return method(*args)

    def get_backend_info(self) -> Dict[str, any]:
        """Get information about current backend configuration."""
        # Get base info from backend manager
        info = self.backend.get_info()

        # Get compiled status from code generator
        compiled = [
            backend
            for backend in ["numpy", "torch", "jax"]
            if self._code_gen.is_compiled(backend)["f"]
        ]

        info["compiled_backends"] = compiled
        info["initialized"] = self._initialized

        return info

    def clone(self, backend: Optional[str] = None, deep: bool = True) -> "SymbolicDynamicalSystem":
        """
        Create a copy of the system, optionally changing backend.

        Args:
            backend: New default backend for cloned system (None = keep same)
            deep: If True, deep copy; if False, shallow copy

        Returns:
            Cloned system

        Example:
            >>> # Clone with same backend
            >>> system2 = system.clone()
            >>>
            >>> # Clone and switch to JAX
            >>> jax_system = system.clone(backend='jax')
        """
        if deep:
            cloned = copy.deepcopy(self)
        else:
            cloned = copy.copy(self)

        if backend is not None:
            cloned.set_default_backend(backend)

        return cloned

    def reset_caches(self, backends: Optional[List[str]] = None):
        """Clear cached compiled functions"""
        self._code_gen.reset_cache(backends)

    def warmup(
        self,
        backend: Optional[str] = None,
        test_point: Optional[Tuple[ArrayLike, ArrayLike]] = None,
    ):
        """
        Warm up backend by compiling and optionally running test evaluation.

        Useful for:
        - JIT compilation (JAX)
        - Reducing first-call latency
        - Validating backend works before production use

        Args:
            backend: Backend to warm up (None = default backend)
            test_point: Optional (x, u) to test with (uses zeros if None)

        Example:
            >>> system.set_default_backend('jax', device='gpu:0')
            >>> system.warmup()  # Compile and warm up JIT
            >>> # First real call is now fast
        """
        backend = backend or self._default_backend

        print(f"Warming up {backend} backend...")
        self._code_gen.generate_dynamics(backend)

        if self._h_sym is not None:
            self._code_gen.generate_output(backend)

        # Test evaluation if requested
        if test_point is not None:
            x_test, u_test = test_point
        else:
            # Use zeros
            x_test = self.equilibria.get_x(backend=backend)
            u_test = self.equilibria.get_u(backend=backend)

        # Run test evaluation
        try:
            dx = self.forward(x_test, u_test, backend=backend)
            print(f"✓ {backend} backend ready (test evaluation successful)")
            return True
        except Exception as e:
            print(f"✗ {backend} backend warmup failed: {e}")
            return False

    @contextmanager
    def use_backend(self, backend: str, device: Optional[str] = None):
        """Temporarily switch backend and device."""
        with self.backend.use_backend(backend, device):
            yield self

    def save_config(self, filename: str):
        """Save system configuration including backend settings"""
        config = {
            "class_name": self.__class__.__name__,
            "parameters": {str(k): float(v) for k, v in self.parameters.items()},
            "order": self.order,
            "nx": self.nx,
            "nu": self.nu,
            "ny": self.ny,
            "default_backend": self._default_backend,  # ← Add
            "preferred_device": self._preferred_device,  # ← Add
            "equilibria": {  # ← Add equilibria
                name: {"x": eq["x"].tolist(), "u": eq["u"].tolist(), "metadata": eq["metadata"]}
                for name, eq in self.equilibria._equilibria.items()
            },
        }

        if filename.endswith(".json"):
            with open(filename, "w") as f:
                json.dump(config, f, indent=2)
        elif filename.endswith(".pt"):
            import torch

            torch.save(config, filename)
        else:
            raise ValueError(f"Unsupported file format: {filename}. Use .json or .pt")

        print(f"Configuration saved to {filename}")

    def get_config_dict(self) -> Dict:
        """Get configuration as dictionary"""
        return {
            "class_name": self.__class__.__name__,
            "parameters": {str(k): float(v) for k, v in self.parameters.items()},
            "order": self.order,
            "nx": self.nx,
            "nu": self.nu,
            "ny": self.ny,
            "default_backend": self._default_backend,  # ← Add
            "preferred_device": self._preferred_device,  # ← Add
        }

    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics."""
        # Get dynamics stats from evaluator
        dynamics_stats = self._dynamics.get_stats()

        return {
            "forward_calls": dynamics_stats["calls"],
            "forward_time": dynamics_stats["total_time"],
            "avg_forward_time": dynamics_stats["avg_time"],
            "linearization_calls": self._perf_stats.get("linearization_calls", 0),
            "linearization_time": self._perf_stats.get("linearization_time", 0.0),
            "avg_linearization_time": (
                self._perf_stats.get("linearization_time", 0.0)
                / max(1, self._perf_stats.get("linearization_calls", 1))
            ),
        }

    def reset_performance_stats(self):
        """Reset performance counters"""
        # Reset dynamics evaluator stats
        self._dynamics.reset_stats()

        # Reset linearization stats (kept in main class for now)
        for key in self._perf_stats:
            self._perf_stats[key] = 0.0 if "time" in key else 0

    # Symbolic methods
    def substitute_parameters(self, expr: Union[sp.Expr, sp.Matrix]) -> Union[sp.Expr, sp.Matrix]:
        """
        Substitute numerical parameter values into symbolic expression

        Args:
            expr: SymPy expression or matrix

        Returns:
            Expression with parameters substituted
        """
        return expr.subs(self.parameters)

    def linearized_dynamics_symbolic(
        self, x_eq: Optional[sp.Matrix] = None, u_eq: Optional[sp.Matrix] = None
    ) -> Tuple[sp.Matrix, sp.Matrix]:
        """
        Compute symbolic linearization A = df/dx, B = df/du

        For second-order systems, constructs the full state-space linearization
        from the acceleration dynamics.

        Args:
            x_eq: Equilibrium state (zeros if None)
            u_eq: Equilibrium control (zeros if None)

        Returns:
            (A, B): Linearized state and control matrices
        """
        if x_eq is None:
            x_eq = sp.Matrix([0] * self.nx)
        if u_eq is None:
            u_eq = sp.Matrix([0] * self.nu)

        # Get symbolic Jacobians from CodeGenerator (it handles caching)
        self._code_gen._compute_symbolic_jacobians()
        A_sym_cached = self._code_gen._A_sym_cache
        B_sym_cached = self._code_gen._B_sym_cache

        if self.order == 1:
            # First-order system: straightforward Jacobian
            A_sym = A_sym_cached
            B_sym = B_sym_cached
        elif self.order == 2:
            # Second-order system: x = [q, qdot], qddot = f(x, u)
            nq = self.nq

            # Compute Jacobians of acceleration w.r.t. q and qdot
            A_accel = self._f_sym.jacobian(self.state_vars)  # (nq, nx)
            B_accel = B_sym_cached  # (nq, nu)

            # Construct full state-space matrices
            A_sym = sp.zeros(self.nx, self.nx)
            A_sym[:nq, nq:] = sp.eye(nq)  # dq/dt = qdot
            A_sym[nq:, :] = A_accel  # dqdot/dt = f(q, qdot, u)

            B_sym = sp.zeros(self.nx, self.nu)
            B_sym[nq:, :] = B_accel  # Control affects acceleration
        else:
            # Higher-order systems
            nq = self.nq
            order = self.order

            A_highest = self._f_sym.jacobian(self.state_vars)
            B_highest = B_sym_cached

            A_sym = sp.zeros(self.nx, self.nx)
            # Each derivative becomes the next one
            for i in range(order - 1):
                A_sym[i * nq : (i + 1) * nq, (i + 1) * nq : (i + 2) * nq] = sp.eye(nq)
            # Highest derivative
            A_sym[(order - 1) * nq :, :] = A_highest

            B_sym = sp.zeros(self.nx, self.nu)
            B_sym[(order - 1) * nq :, :] = B_highest

        # Substitute equilibrium point
        subs_dict = dict(zip(self.state_vars + self.control_vars, list(x_eq) + list(u_eq)))
        A = A_sym.subs(subs_dict)
        B = B_sym.subs(subs_dict)

        # Substitute parameters
        A = self.substitute_parameters(A)
        B = self.substitute_parameters(B)

        return A, B

    def linearized_observation_symbolic(self, x_eq: Optional[sp.Matrix] = None) -> sp.Matrix:
        """
        Compute symbolic linearization C = dh/dx

        Args:
            x_eq: Equilibrium state (zeros if None)

        Returns:
            C: Linearized output matrix
        """
        if self._h_sym is None:
            return sp.eye(self.nx)

        if x_eq is None:
            x_eq = sp.Matrix([0] * self.nx)

        # Get symbolic Jacobian from CodeGenerator
        self._code_gen._compute_symbolic_jacobians()
        C_sym_cached = self._code_gen._C_sym_cache

        subs_dict = dict(zip(self.state_vars, list(x_eq)))
        C = C_sym_cached.subs(subs_dict)
        C = self.substitute_parameters(C)

        return C

    def verify_jacobians(
        self, x: ArrayLike, u: ArrayLike, tol: float = 1e-3, backend: str = "torch"
    ) -> Dict[str, Union[bool, float]]:
        """
        Verify symbolic Jacobians against automatic differentiation.

        Uses autodiff to numerically compute Jacobians and compares against
        symbolic derivation. Requires a backend with autodiff (torch or jax).

        Args:
            x: State at which to verify
            u: Control at which to verify
            tol: Tolerance for considering Jacobians equal
            backend: Backend for autodiff ('torch' or 'jax')
                    NumPy doesn't support autodiff

        Returns:
            Dict with 'A_match', 'B_match' booleans and error magnitudes

        Raises:
            RuntimeError: If backend doesn't support autodiff

        Example:
            >>> # Using PyTorch
            >>> x = torch.tensor([0.1, 0.0])
            >>> u = torch.tensor([0.0])
            >>> results = system.verify_jacobians(x, u, backend='torch')
            >>>
            >>> # Using JAX
            >>> x = jnp.array([0.1, 0.0])
            >>> u = jnp.array([0.0])
            >>> results = system.verify_jacobians(x, u, backend='jax')
        """
        if backend not in ["torch", "jax"]:
            raise ValueError(
                f"Jacobian verification requires autodiff backend ('torch' or 'jax'), "
                f"got '{backend}'. NumPy doesn't support automatic differentiation."
            )

        # Check backend availability
        self._check_backend_available(backend)

        # Dispatch to backend-specific implementation
        if backend == "torch":
            return self._verify_jacobians_torch(x, u, tol)
        else:  # jax
            return self._verify_jacobians_jax(x, u, tol)

    def _verify_jacobians_torch(
        self, x: ArrayLike, u: ArrayLike, tol: float
    ) -> Dict[str, Union[bool, float]]:
        """PyTorch-based Jacobian verification"""
        import torch

        # Convert to torch if needed
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(np.asarray(x), dtype=torch.float32)
        if not isinstance(u, torch.Tensor):
            u = torch.tensor(np.asarray(u), dtype=torch.float32)

        # Ensure proper 2D shape (batch_size=1, dim)
        x_2d = x.reshape(1, -1) if len(x.shape) <= 1 else x
        u_2d = u.reshape(1, -1) if len(u.shape) <= 1 else u

        # Clone for autograd
        x_grad = x_2d.clone().requires_grad_(True)
        u_grad = u_2d.clone().requires_grad_(True)

        # Compute symbolic Jacobians
        A_sym, B_sym = self.linearized_dynamics(x_2d.detach(), u_2d.detach(), backend="torch")

        # Ensure 3D shape for batch processing
        if len(A_sym.shape) == 2:
            A_sym = A_sym.unsqueeze(0)
            B_sym = B_sym.unsqueeze(0)

        ## Compute numerical Jacobians via autograd
        fx = self.forward(x_grad, u_grad, backend="torch")

        # CRITICAL FIX: Ensure fx is 2D for consistent indexing
        if fx.ndim == 1:
            fx = fx.unsqueeze(1)  # (n,) → (n, 1)
        elif fx.ndim == 0:
            fx = fx.reshape(1, 1)  # scalar → (1, 1)

        # Determine output dimension
        if self.order == 1:
            n_outputs = self.nx
        else:
            n_outputs = self.nq

        # Compute gradients
        A_num = torch.zeros_like(A_sym)
        B_num = torch.zeros_like(B_sym)

        if self.order == 1:
            # First-order: verify full A and B
            for i in range(n_outputs):
                if fx[0, i].requires_grad:
                    grad_x = torch.autograd.grad(
                        fx[0, i], x_grad, retain_graph=True, create_graph=False
                    )[0]
                    grad_u = torch.autograd.grad(
                        fx[0, i], u_grad, retain_graph=True, create_graph=False
                    )[0]
                    A_num[0, i] = grad_x[0]
                    B_num[0, i] = grad_u[0]
        else:
            # Higher-order: verify acceleration part
            for i in range(n_outputs):
                if fx[0, i].requires_grad:
                    grad_x = torch.autograd.grad(
                        fx[0, i], x_grad, retain_graph=True, create_graph=False
                    )[0]
                    grad_u = torch.autograd.grad(
                        fx[0, i], u_grad, retain_graph=True, create_graph=False
                    )[0]

                    row_idx = (self.order - 1) * self.nq + i
                    A_num[0, row_idx] = grad_x[0]
                    B_num[0, row_idx] = grad_u[0]

            # Copy derivative relationships
            for i in range((self.order - 1) * self.nq):
                A_num[0, i] = A_sym[0, i]
                B_num[0, i] = B_sym[0, i]

        # Compute errors
        A_error = (A_sym - A_num).abs().max().item()
        B_error = (B_sym - B_num).abs().max().item()

        return {
            "A_match": bool(A_error < tol),
            "B_match": bool(B_error < tol),
            "A_error": float(A_error),
            "B_error": float(B_error),
        }

    def _verify_jacobians_jax(
        self, x: ArrayLike, u: ArrayLike, tol: float
    ) -> Dict[str, Union[bool, float]]:
        """JAX-based Jacobian verification"""
        import jax
        import jax.numpy as jnp

        # Convert to JAX if needed
        if not isinstance(x, jnp.ndarray):
            x = jnp.array(np.asarray(x))
        if not isinstance(u, jnp.ndarray):
            u = jnp.array(np.asarray(u))

        # Ensure proper shape
        x_2d = x.reshape(1, -1) if x.ndim <= 1 else x
        u_2d = u.reshape(1, -1) if u.ndim <= 1 else u

        # Compute symbolic Jacobians
        A_sym, B_sym = self.linearized_dynamics(x_2d, u_2d, backend="jax")

        # For JAX, use the autodiff Jacobian computation directly
        # (which is what _linearized_dynamics_jax already does!)
        # So we can just compare against symbolic evaluation

        # Get symbolic Jacobians as NumPy
        x_np = np.array(x_2d[0])
        u_np = np.array(u_2d[0])
        A_sym_np, B_sym_np = self.linearized_dynamics_symbolic(sp.Matrix(x_np), sp.Matrix(u_np))
        A_sym_np = np.array(A_sym_np, dtype=np.float64)
        B_sym_np = np.array(B_sym_np, dtype=np.float64)

        # Convert JAX results to NumPy for comparison
        A_jax_np = np.array(A_sym)
        B_jax_np = np.array(B_sym)

        # Compute errors
        A_error = np.abs(A_sym_np - A_jax_np).max()
        B_error = np.abs(B_sym_np - B_jax_np).max()

        return {
            "A_match": bool(A_error < tol),
            "B_match": bool(B_error < tol),
            "A_error": float(A_error),
            "B_error": float(B_error),
        }

    # numerical function generation
    def compile(
        self, backends: Optional[List[str]] = None, verbose: bool = False, **kwargs
    ) -> Dict[str, float]:
        """
        Pre-compile dynamics functions for specified backends.

        Delegates to CodeGenerator for actual compilation.
        """
        # Extract just the 'f' timings for backward compatibility
        all_timings = self._code_gen.compile_all(backends=backends, verbose=verbose, **kwargs)
        return {backend: timings.get("f") for backend, timings in all_timings.items()}

    # forward methods
    def forward(self, x: ArrayLike, u: ArrayLike, backend: Optional[str] = None) -> ArrayLike:
        """Evaluate continuous-time dynamics: dx/dt = f(x, u)"""
        return self._dynamics.evaluate(x, u, backend)

    def __call__(self, x: ArrayLike, u: ArrayLike, backend: Optional[str] = None) -> ArrayLike:
        """
        Make system callable: system(x, u) calls forward(x, u).

        This is the primary user interface.
        """
        return self.forward(x, u, backend)

    # Linearized dynamics methods
    def linearized_dynamics(
        self, x: ArrayLike, u: ArrayLike, backend: Optional[str] = None
    ) -> Tuple[ArrayLike, ArrayLike]:
        """
        Numerical evaluation of linearized dynamics at point (x, u)

        Automatically detects backend from input types:
        - torch.Tensor → PyTorch computation
        - jax.Array → JAX computation (uses autodiff)
        - np.ndarray → NumPy computation (uses symbolic)

        Args:
            x: State tensor/array
            u: Control tensor/array

        Returns:
            (A, B): Linearized dynamics matrices (same type as input)

        TODO: needs to be adapted to equilibrium handling
        """
        if backend is not None:  # ← Changed
            target_backend = self._default_backend if backend == "default" else backend
            input_backend = self._detect_backend(x)
            if input_backend != target_backend:
                x = self._convert_to_backend(x, target_backend)
                u = self._convert_to_backend(u, target_backend)

        return self._dispatch_to_backend("_linearized_dynamics", backend, x, u)

    def _linearized_dynamics_torch(
        self, x: "torch.Tensor", u: "torch.Tensor"
    ) -> Tuple["torch.Tensor", "torch.Tensor"]:
        """PyTorch implementation using cached Jacobian functions or symbolic evaluation"""
        import torch

        start_time = time.time()

        # Handle batched input
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
            u = u.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        batch_size = x.shape[0]
        device = x.device
        dtype = x.dtype

        # Allocate output tensors
        A_batch = torch.zeros(batch_size, self.nx, self.nx, dtype=dtype, device=device)
        B_batch = torch.zeros(batch_size, self.nx, self.nu, dtype=dtype, device=device)

        # Try to get cached Jacobian functions from CodeGenerator
        A_func, B_func, _ = self._code_gen.get_jacobians("torch")

        if A_func is not None and B_func is not None:
            # Use cached functions (faster)
            for i in range(batch_size):
                x_i = x[i]
                u_i = u[i]

                # Prepare arguments
                x_list = [x_i[j] for j in range(self.nx)]
                u_list = [u_i[j] for j in range(self.nu)]
                all_args = x_list + u_list

                # Call cached Jacobian functions
                A_batch[i] = A_func(*all_args)
                B_batch[i] = B_func(*all_args)
        else:
            # Fall back to symbolic evaluation
            for i in range(batch_size):
                x_i = x[i] if batch_size > 1 else x.squeeze(0)
                u_i = u[i] if batch_size > 1 else u.squeeze(0)

                x_np = x_i.detach().cpu().numpy()
                u_np = u_i.detach().cpu().numpy()

                # Ensure arrays are at least 1D for SymPy Matrix
                x_np = np.atleast_1d(x_np)
                u_np = np.atleast_1d(u_np)

                A_sym, B_sym = self.linearized_dynamics_symbolic(sp.Matrix(x_np), sp.Matrix(u_np))
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
        if "linearization_calls" in self._perf_stats:
            self._perf_stats["linearization_calls"] += 1
            self._perf_stats["linearization_time"] += time.time() - start_time

        return A_batch, B_batch

    def _linearized_dynamics_jax(
        self, x: "jnp.ndarray", u: "jnp.ndarray"
    ) -> Tuple["jnp.ndarray", "jnp.ndarray"]:
        """JAX implementation using automatic differentiation"""
        import jax
        import jax.numpy as jnp

        # Ensure dynamics function is available
        f_jax = self._code_gen.generate_dynamics("jax", jit=True)

        # Handle batched input
        if x.ndim == 1:
            x = jnp.expand_dims(x, 0)
            u = jnp.expand_dims(u, 0)
            squeeze_output = True
        else:
            squeeze_output = False

        # Define dynamics function for Jacobian computation
        def dynamics_fn(x_i, u_i):
            x_list = [x_i[j] for j in range(self.nx)]
            u_list = [u_i[j] for j in range(self.nu)]
            return f_jax(*(x_list + u_list))

        # Compute Jacobians using JAX autodiff (vmap for batching)
        @jax.vmap
        def compute_jacobians(x_i, u_i):
            A = jax.jacobian(lambda x: dynamics_fn(x, u_i))(x_i)
            B = jax.jacobian(lambda u: dynamics_fn(x_i, u))(u_i)
            return A, B

        A_batch, B_batch = compute_jacobians(x, u)

        if squeeze_output:
            A_batch = jnp.squeeze(A_batch, 0)
            B_batch = jnp.squeeze(B_batch, 0)

        return A_batch, B_batch

    def _linearized_dynamics_numpy(
        self, x: np.ndarray, u: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """NumPy implementation using symbolic evaluation"""

        # Handle batched input
        if x.ndim == 1:
            x = np.expand_dims(x, 0)
            u = np.expand_dims(u, 0)
            squeeze_output = True
        else:
            squeeze_output = False

        batch_size = x.shape[0]

        A_batch = np.zeros((batch_size, self.nx, self.nx))
        B_batch = np.zeros((batch_size, self.nx, self.nu))

        # Try to get cached NumPy Jacobian functions from CodeGenerator
        A_func, B_func, _ = self._code_gen.get_jacobians("numpy")

        if A_func is not None and B_func is not None:
            # Use cached functions
            for i in range(batch_size):
                x_i = x[i]
                u_i = u[i]

                # Prepare arguments
                x_list = [x_i[j] for j in range(self.nx)]
                u_list = [u_i[j] for j in range(self.nu)]
                all_args = x_list + u_list

                # Call cached Jacobian functions
                A_result = A_func(*all_args)
                B_result = B_func(*all_args)

                # Handle different output types from lambdify
                A_batch[i] = np.array(A_result, dtype=np.float64)
                B_batch[i] = np.array(B_result, dtype=np.float64)
        else:
            # Fall back to symbolic evaluation
            for i in range(batch_size):
                x_np = np.atleast_1d(x[i])
                u_np = np.atleast_1d(u[i])

                # Use symbolic Jacobians
                A_sym, B_sym = self.linearized_dynamics_symbolic(sp.Matrix(x_np), sp.Matrix(u_np))
                A_batch[i] = np.array(A_sym, dtype=np.float64)
                B_batch[i] = np.array(B_sym, dtype=np.float64)

        if squeeze_output:
            A_batch = np.squeeze(A_batch, 0)
            B_batch = np.squeeze(B_batch, 0)

        return A_batch, B_batch

    # Linearized observation methods
    def linearized_observation(self, x: ArrayLike, backend: Optional[str] = None) -> ArrayLike:
        """
        Numerical evaluation of output linearization C = dh/dx

        Automatically detects backend from input type:
        - torch.Tensor → PyTorch computation
        - jax.Array → JAX computation (uses autodiff)
        - np.ndarray → NumPy computation (uses symbolic)

        Args:
            x: State tensor/array

        Returns:
            C: Linearized observation matrix (same type as input)

        TODO: needs to be adapted to current equilibrium handling mechanism
        """
        if backend is not None:
            target_backend = self._default_backend if backend == "default" else backend
            input_backend = self._detect_backend(x)
            if input_backend != target_backend:
                x = self._convert_to_backend(x, target_backend)

        return self._dispatch_to_backend("_linearized_observation", backend, x)

    def _linearized_observation_torch(self, x: "torch.Tensor") -> "torch.Tensor":
        """PyTorch implementation using cached Jacobian function or symbolic evaluation"""
        import torch

        # If no custom output function, return identity
        if self._h_sym is None:
            batch_size = x.shape[0] if len(x.shape) > 1 else 1
            if len(x.shape) == 1:
                return torch.eye(self.nx, dtype=x.dtype, device=x.device)
            else:
                return (
                    torch.eye(self.nx, dtype=x.dtype, device=x.device)
                    .unsqueeze(0)
                    .expand(batch_size, -1, -1)
                )

        # Handle batched input
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        batch_size = x.shape[0]
        device = x.device
        dtype = x.dtype

        C_batch = torch.zeros(batch_size, self.ny, self.nx, dtype=dtype, device=device)

        # Try to get cached Jacobian function from CodeGenerator
        _, _, C_func = self._code_gen.get_jacobians("torch")

        if C_func is not None:
            # Use cached function (faster)
            for i in range(batch_size):
                x_i = x[i]

                # Prepare arguments (only state variables for observation)
                x_list = [x_i[j] for j in range(self.nx)]

                # Call cached Jacobian function
                C_batch[i] = C_func(*x_list)
        else:
            # Fall back to symbolic evaluation
            for i in range(batch_size):
                x_i = x[i] if batch_size > 1 else x.squeeze(0)
                x_np = x_i.detach().cpu().numpy()

                # Ensure at least 1D
                x_np = np.atleast_1d(x_np)

                C_sym = self.linearized_observation_symbolic(sp.Matrix(x_np))
                C_batch[i] = torch.tensor(
                    np.array(C_sym, dtype=np.float64), dtype=dtype, device=device
                )

        if squeeze_output:
            C_batch = C_batch.squeeze(0)

        return C_batch

    def _linearized_observation_jax(self, x: "jnp.ndarray") -> "jnp.ndarray":
        """JAX implementation using automatic differentiation"""
        import jax
        import jax.numpy as jnp

        # If no custom output function, return identity
        if self._h_sym is None:
            batch_size = x.shape[0] if x.ndim > 1 else 1
            if x.ndim == 1:
                return jnp.eye(self.nx)
            else:
                return jnp.tile(jnp.eye(self.nx), (batch_size, 1, 1))

        # Ensure observation function is generated
        h_jax = self._code_gen.generate_output("jax", jit=True)

        # Handle batched input
        if x.ndim == 1:
            x = jnp.expand_dims(x, 0)
            squeeze_output = True
        else:
            squeeze_output = False

        # Define observation function for Jacobian computation
        def observation_fn(x_i):
            x_list = [x_i[j] for j in range(self.nx)]
            return h_jax(*x_list)

        # Compute Jacobian using JAX autodiff (vmap for batching)
        @jax.vmap
        def compute_jacobian(x_i):
            return jax.jacobian(observation_fn)(x_i)

        C_batch = compute_jacobian(x)

        if squeeze_output:
            C_batch = jnp.squeeze(C_batch, 0)

        return C_batch

    def _linearized_observation_numpy(self, x: np.ndarray) -> np.ndarray:
        """NumPy implementation using symbolic evaluation"""

        # If no custom output function, return identity
        if self._h_sym is None:
            batch_size = x.shape[0] if x.ndim > 1 else 1
            if x.ndim == 1:
                return np.eye(self.nx)
            else:
                return np.tile(np.eye(self.nx), (batch_size, 1, 1))

        # Handle batched input
        if x.ndim == 1:
            x = np.expand_dims(x, 0)
            squeeze_output = True
        else:
            squeeze_output = False

        batch_size = x.shape[0]
        C_batch = np.zeros((batch_size, self.ny, self.nx))

        # Try to get cached NumPy Jacobian function from CodeGenerator
        _, _, C_func = self._code_gen.get_jacobians("numpy")

        if C_func is not None:
            # Use cached function
            for i in range(batch_size):
                x_i = x[i]

                # Prepare arguments
                x_list = [x_i[j] for j in range(self.nx)]

                # Call cached Jacobian function
                C_result = C_func(*x_list)
                C_batch[i] = np.array(C_result, dtype=np.float64)
        else:
            # Fall back to symbolic evaluation
            for i in range(batch_size):
                x_np = np.atleast_1d(x[i])

                # Use symbolic Jacobian
                C_sym = self.linearized_observation_symbolic(sp.Matrix(x_np))
                C_batch[i] = np.array(C_sym, dtype=np.float64)

        if squeeze_output:
            C_batch = np.squeeze(C_batch, 0)

        return C_batch

    # Nonlinear output observation methods
    def h(self, x: ArrayLike, backend: Optional[str] = None) -> ArrayLike:
        """
        Evaluate output equation: y = h(x)

        Automatically detects backend from input type:
        - torch.Tensor → PyTorch computation
        - jax.Array → JAX computation
        - np.ndarray → NumPy computation

        Args:
            x: State tensor/array
            backend: Backend selection (optional)
                - None: Auto-detect from input type (default)
                - 'numpy', 'torch', 'jax': Force specific backend
                - 'default': Use self._default_backend

        Returns:
            Output tensor/array (type determined by backend)
        """
        if backend is not None:
            # Determine target
            target_backend = self._default_backend if backend == "default" else backend

            # Convert if needed
            input_backend = self._detect_backend(x)
            if input_backend != target_backend:
                x = self._convert_to_backend(x, target_backend)

        return self._dispatch_to_backend("_h", backend, x)

    def _h_torch(self, x: "torch.Tensor") -> "torch.Tensor":
        """PyTorch implementation of output equation"""

        if self._h_sym is None:
            return x

        try:
            h_torch = self._code_gen.generate_output("torch")
            if h_torch is None:
                return x  # No custom output
        except Exception as e:
            raise RuntimeError(f"Failed to generate torch output function: {e}") from e

        # Verify function was generated
        if h_torch is None:
            raise RuntimeError("Torch output function is still None after generation")

        # Verify it's callable
        if not callable(h_torch):
            raise TypeError(f"Generated torch output function is not callable: {type(h_torch)}")

        # Handle batched input
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        # Prepare input arguments
        x_list = [x[:, i] for i in range(self.nx)]

        # Call function
        try:
            result = h_torch(*x_list)
        except Exception as e:
            raise RuntimeError(f"Error evaluating torch output function: {e}") from e

        # For single (unbatched) input, squeeze batch dimension but keep output dimension
        if squeeze_output:
            if result.ndim > 1:
                result = result.squeeze(0)
            elif result.ndim == 0:
                result = result.reshape(1)
        else:
            # Batched output - ensure shape is (batch, ny)
            # If result is (batch,) but should be (batch, 1), add dimension
            if result.ndim == 1 and self.ny == 1:
                result = result.unsqueeze(1)

        # Final safety: ensure at least 1D
        if result.ndim == 0:
            result = result.reshape(1)

        return result

    def _h_numpy(self, x: np.ndarray) -> np.ndarray:
        """NumPy implementation of output equation"""

        # If no custom output function, return full state
        if self._h_sym is None:
            return x

        # Generate NumPy function for h if not cached
        try:
            h_numpy = self._code_gen.generate_output("numpy")
            if h_numpy is None:  # No custom output
                return x
        except Exception as e:
            raise RuntimeError(f"Failed to generate output function: {e}") from e

        # Verify function was generated
        if h_numpy is None:
            raise RuntimeError("Output function is still None after generation attempt")

        # Verify it's callable
        if not callable(h_numpy):
            raise TypeError(f"Generated output function is not callable: {type(h_numpy)}")

        # Handle batched input
        if x.ndim == 1:
            x = np.expand_dims(x, 0)
            squeeze_output = True
        else:
            squeeze_output = False

        batch_size = x.shape[0]
        results = []

        try:
            for i in range(batch_size):
                x_list = [x[i, j] for j in range(self.nx)]
                result = h_numpy(*x_list)

                # Convert to array
                result = np.atleast_1d(np.array(result))
                results.append(result)
        except Exception as e:
            raise RuntimeError(f"Error evaluating output function: {e}") from e

        result = np.stack(results)

        if squeeze_output:
            result = np.squeeze(result, 0)

        return result

    def _h_jax(self, x: "jnp.ndarray") -> "jnp.ndarray":
        """JAX implementation of output equation"""
        import jax.numpy as jnp

        # If no custom output function, return full state
        if self._h_sym is None:
            return x

        # Generate JAX function for h if not cached
        try:
            h_jax = self._code_gen.generate_output("jax", jit=True)
            if h_jax is None:
                return x  # No custom output
        except Exception as e:
            raise RuntimeError(f"Failed to generate torch output function: {e}") from e

        # Handle batched input
        if x.ndim == 1:
            x = jnp.expand_dims(x, 0)
            squeeze_output = True
        else:
            squeeze_output = False

        # For batched computation, use vmap
        if x.shape[0] > 1:
            import jax

            @jax.vmap
            def batched_observation(x_i):
                x_list = [x_i[j] for j in range(self.nx)]
                return h_jax(*x_list)

            result = batched_observation(x)
        else:
            # Single evaluation
            x_list = [x[0, i] for i in range(self.nx)]
            result = h_jax(*x_list)
            result = jnp.expand_dims(result, 0)

        if squeeze_output:
            result = jnp.squeeze(result, 0)

        return result

    # Utility methods
    def print_equations(self, simplify: bool = True):
        """
        Print symbolic equations in human-readable format

        Args:
            simplify: Whether to simplify expressions before printing
        """
        print("=" * 70)
        print(f"{self.__class__.__name__}")
        print("=" * 70)
        print(f"State Variables: {self.state_vars}")
        print(f"Control Variables: {self.control_vars}")
        print(f"System Order: {self.order}")
        print(f"Dimensions: nx={self.nx}, nu={self.nu}, ny={self.ny}")

        print("\nDynamics: dx/dt = f(x, u)")
        for i, (var, expr) in enumerate(zip(self.state_vars, self._f_sym)):
            expr_sub = self.substitute_parameters(expr)
            if simplify:
                expr_sub = sp.simplify(expr_sub)
            print(f"  d{var}/dt = {expr_sub}")

        if self._h_sym is not None:
            print("\nOutput: y = h(x)")
            for i, expr in enumerate(self._h_sym):
                expr_sub = self.substitute_parameters(expr)
                if simplify:
                    expr_sub = sp.simplify(expr_sub)
                print(f"  y[{i}] = {expr_sub}")

        print("=" * 70)

    # Convenience: Quick access to control design
    @property
    def control(self):
        """Get control designer for this system"""
        if not hasattr(self, "_control_designer"):
            from src.control.control_designer import ControlDesigner

            self._control_designer = ControlDesigner(self)
        return self._control_designer

    # Convenience: Quick access to adding new equilibria

    def _verify_equilibrium_numpy(self, x_eq: np.ndarray, u_eq: np.ndarray) -> np.ndarray:
        """Helper for equilibrium verification using NumPy backend"""
        return self.forward(x_eq, u_eq, backend="numpy")

    def add_equilibrium(
        self,
        name: str,
        x_eq: np.ndarray,
        u_eq: np.ndarray,
        verify: bool = True,
        tol: float = 1e-6,
        **metadata,
    ):
        """
        Convenience wrapper for adding equilibria with automatic verification.

        Args:
            name: Equilibrium name
            x_eq: Equilibrium state
            u_eq: Equilibrium control
            verify: Whether to verify it's actually an equilibrium
            tol: Tolerance for verification
            **metadata: Additional metadata

        Example:
            >>> system.add_equilibrium('inverted',
            ...                        x_eq=np.array([np.pi, 0.0]),
            ...                        u_eq=np.array([0.0]))
        """
        verify_fn = self._verify_equilibrium_numpy if verify else None
        self.equilibria.add(name, x_eq, u_eq, verify_fn=verify_fn, tol=tol, **metadata)
