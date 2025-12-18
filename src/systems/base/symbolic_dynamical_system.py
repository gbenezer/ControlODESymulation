import copy
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Union, Optional, Callable
import json
import time
import sympy as sp
from sympy import pycode
import numpy as np
import scipy
import torch
import torch.nn as nn


class SymbolicDynamicalSystem(ABC, nn.Module):
    """
    (Currently) Torch-based class for dynamical systems defined symbolically with SymPy.
    
    **Current Backend Support**:
    This class inherits from nn.Module (PyTorch) but supports multi-backend
    computation through automatic dispatch:
    - forward() accepts torch.Tensor, np.ndarray, or jax.Array
    - Returns same type as input
    - Equilibrium points are torch.Tensor (convert if using other backends)
    
    TODO: Make class fully backend-agnostic by implementing backend-setting/backend-passing logic

    Attributes:
        state_vars: List of symbolic state variables
        control_vars: List of symbolic control variables
        output_vars: List of symbolic output variables
        parameters: Dict mapping SymPy symbols to numerical values
        order: System order (1=first-order, 2=second-order, etc.)
        backend: Initial/current backend ("numpy", "jax", or "torch") (TODO)
    """

    def __init__(self):
        super().__init__()
        # To be defined by subclasses
        self.state_vars: List[sp.Symbol] = []
        self.control_vars: List[sp.Symbol] = []
        self.output_vars: List[sp.Symbol] = []
        self.parameters: Dict[sp.Symbol, float] = {}  # Symbols as keys!

        # Symbolic expressions (to be defined)
        self._f_sym: Optional[sp.Matrix] = None  # State dynamics: dx/dt = f(x, u)
        self._h_sym: Optional[sp.Matrix] = None  # Output: y = h(x)

        # System order (1 for first-order, 2 for second-order, etc.)
        self.order: int = 1

        # System backend
        self.backend = "numpy"

        # Cached numerical functions
        self._f_numpy: Optional[Callable] = None
        self._h_numpy: Optional[Callable] = None
        self._f_torch: Optional[Callable] = None
        self._h_torch: Optional[Callable] = None
        self._f_jax: Optional[Callable] = None
        self._h_jax: Optional[Callable] = None

        # Cached Jacobians for efficiency
        self._A_sym_cached: Optional[sp.Matrix] = None
        self._B_sym_cached: Optional[sp.Matrix] = None
        self._C_sym_cached: Optional[sp.Matrix] = None

        # Flag to track if system has been properly initialized
        self._initialized: bool = False

        # Performance statistics
        self._perf_stats = {
            "forward_calls": 0,
            "forward_time": 0.0,
            "linearization_calls": 0,
            "linearization_time": 0.0,
        }

    @abstractmethod
    def define_system(self, *args, **kwargs):
        """
        Define the symbolic system. Must set:
        - self.state_vars: List of state symbols
        - self.control_vars: List of control symbols
        - self.output_vars: List of output symbols (optional)
        - self.parameters: Dict with Symbol keys (not strings!)
        - self._f_sym: Symbolic dynamics matrix
        - self._h_sym: Symbolic output matrix (optional)
        - self.order: System order (default: 1)
        - self.backend: System backend (default: numpy)

        CRITICAL: self.parameters must use SymPy Symbol objects as keys!
        Example: {m: 1.0, l: 0.5} NOT {'m': 1.0, 'l': 0.5}

        Args:
            *args, **kwargs: System-specific parameters
        """
        pass

    def _validate_system(self) -> bool:
        """Validate that the system is properly defined"""
        errors = []

        if not self.state_vars:
            errors.append("state_vars is empty")

        if not self.control_vars:
            errors.append("control_vars is empty")

        if self._f_sym is None:
            errors.append("_f_sym is not defined")

        if self.parameters:
            for key in self.parameters.keys():
                if not isinstance(key, sp.Symbol):
                    errors.append(f"Parameter key {key} is not a SymPy Symbol")

        if errors:
            error_msg = "System validation failed:\n" + "\n".join(
                f"  - {e}" for e in errors
            )
            error_msg += "\n\nHINT: Did you use Symbol objects as parameter keys?"
            error_msg += "\n  Correct:   {m: 1.0, l: 0.5}"
            error_msg += "\n  Incorrect: {'m': 1.0, 'l': 0.5}"
            raise ValueError(error_msg)

        self._initialized = True
        return True

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
    def x_equilibrium(self) -> torch.Tensor:
        """Equilibrium state (override in subclass if needed)"""
        return torch.zeros(self.nx)

    @property
    def u_equilibrium(self) -> torch.Tensor:
        """Equilibrium control (override in subclass if needed)"""
        return torch.zeros(self.nu)

    def substitute_parameters(
        self, expr: Union[sp.Expr, sp.Matrix]
    ) -> Union[sp.Expr, sp.Matrix]:
        """
        Substitute numerical parameter values into symbolic expression

        Args:
            expr: SymPy expression or matrix

        Returns:
            Expression with parameters substituted
        """
        return expr.subs(self.parameters)

    def _cache_jacobians(self, backend="torch"):
        """
        Cache symbolic Jacobians and generate numerical functions for improved performance

        Args:
            backend: Target backend ('torch', 'jax', 'numpy')
        """

        from src.systems.base.codegen_utils import generate_function

        all_vars = self.state_vars + self.control_vars

        # Set backend-specific kwargs
        if backend == "torch":
            backend_kwargs = {}
        elif backend == "jax":
            backend_kwargs = {"jit": True}
        else:  # numpy
            backend_kwargs = {}

        # Cache dynamics Jacobians
        if self._f_sym is not None and self._A_sym_cached is None:
            self._A_sym_cached = self._f_sym.jacobian(self.state_vars)
            self._B_sym_cached = self._f_sym.jacobian(self.control_vars)

            A_with_params = self.substitute_parameters(self._A_sym_cached)
            B_with_params = self.substitute_parameters(self._B_sym_cached)

            if backend == "torch":
                self._A_torch_fn = generate_function(
                    A_with_params, all_vars, backend="torch", **backend_kwargs
                )
                self._B_torch_fn = generate_function(
                    B_with_params, all_vars, backend="torch", **backend_kwargs
                )
            elif backend == "jax":
                self._A_jax_fn = generate_function(
                    A_with_params, all_vars, backend="jax", **backend_kwargs
                )
                self._B_jax_fn = generate_function(
                    B_with_params, all_vars, backend="jax", **backend_kwargs
                )
            elif backend == "numpy":
                self._A_numpy_fn = generate_function(
                    A_with_params, all_vars, backend="numpy", **backend_kwargs
                )
                self._B_numpy_fn = generate_function(
                    B_with_params, all_vars, backend="numpy", **backend_kwargs
                )

        # Cache observation Jacobian
        if self._h_sym is not None and self._C_sym_cached is None:
            self._C_sym_cached = self._h_sym.jacobian(self.state_vars)
            C_with_params = self.substitute_parameters(self._C_sym_cached)

            if backend == "torch":
                self._C_torch_fn = generate_function(
                    C_with_params, self.state_vars, backend="torch", **backend_kwargs
                )
            elif backend == "jax":
                self._C_jax_fn = generate_function(
                    C_with_params, self.state_vars, backend="jax", **backend_kwargs
                )
            elif backend == "numpy":
                self._C_numpy_fn = generate_function(
                    C_with_params, self.state_vars, backend="numpy", **backend_kwargs
                )

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

        # Use cached Jacobians if available
        if self._A_sym_cached is None:
            self._cache_jacobians()

        if self.order == 1:
            # First-order system: straightforward Jacobian
            A_sym = self._A_sym_cached
            B_sym = self._B_sym_cached
        elif self.order == 2:
            # Second-order system: x = [q, qdot], qddot = f(x, u)
            # Need to construct full state-space form:
            # d/dt [q]    = [      0       I ]  [q]    + [  0  ] u
            #      [qdot]   [df/dq   df/dqdot]  [qdot]   [df/du]

            nq = self.nq

            # Compute Jacobians of acceleration w.r.t. q and qdot
            A_accel = self._f_sym.jacobian(self.state_vars)  # (nq, nx)
            B_accel = self._B_sym_cached  # (nq, nu)

            # Construct full state-space matrices
            A_sym = sp.zeros(self.nx, self.nx)
            A_sym[:nq, nq:] = sp.eye(nq)  # dq/dt = qdot
            A_sym[nq:, :] = A_accel  # dqdot/dt = f(q, qdot, u)

            B_sym = sp.zeros(self.nx, self.nu)
            B_sym[nq:, :] = B_accel  # Control affects acceleration
        else:
            # Higher-order systems
            # x = [q, q', q'', ..., q^(n-1)], q^(n) = f(x, u)
            # State-space form has similar structure
            nq = self.nq
            order = self.order

            A_highest = self._f_sym.jacobian(
                self.state_vars
            )  # Jacobian of highest derivative
            B_highest = self._B_sym_cached

            A_sym = sp.zeros(self.nx, self.nx)
            # Each derivative becomes the next one
            for i in range(order - 1):
                A_sym[i * nq : (i + 1) * nq, (i + 1) * nq : (i + 2) * nq] = sp.eye(nq)
            # Highest derivative
            A_sym[(order - 1) * nq :, :] = A_highest

            B_sym = sp.zeros(self.nx, self.nu)
            B_sym[(order - 1) * nq :, :] = B_highest

        # Substitute equilibrium point
        subs_dict = dict(
            zip(self.state_vars + self.control_vars, list(x_eq) + list(u_eq))
        )
        A = A_sym.subs(subs_dict)
        B = B_sym.subs(subs_dict)

        # Substitute parameters
        A = self.substitute_parameters(A)
        B = self.substitute_parameters(B)

        return A, B

    def linearized_observation_symbolic(
        self, x_eq: Optional[sp.Matrix] = None
    ) -> sp.Matrix:
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

        # Use cached Jacobian if available
        if self._C_sym_cached is None:
            self._cache_jacobians()

        subs_dict = dict(zip(self.state_vars, list(x_eq)))
        C = self._C_sym_cached.subs(subs_dict)
        C = self.substitute_parameters(C)

        return C

    def generate_numpy_function(self) -> Callable:
        """
        Generate lambdified NumPy function for dynamics

        Returns:
            Callable function compatible with NumPy
        """
        from src.systems.base.codegen_utils import generate_numpy_function

        f_with_params = self.substitute_parameters(self._f_sym)
        all_vars = self.state_vars + self.control_vars
        self._f_numpy = generate_numpy_function(f_with_params, all_vars)
        return self._f_numpy

    def generate_torch_function(self) -> Callable:
        """
        Generate PyTorch-compatible function for dynamics using code generation

        This method generates Python code as a string and executes it to create
        a function that uses PyTorch operations. This approach avoids issues with
        SymPy's lambdify and PyTorch tensor operations.

        Returns:
            Callable function compatible with PyTorch tensors
        """
        from src.systems.base.codegen_utils import generate_torch_function

        f_with_params = self.substitute_parameters(self._f_sym)
        f_with_params = sp.simplify(f_with_params)

        all_vars = self.state_vars + self.control_vars

        self._f_torch = generate_torch_function(f_with_params, all_vars)

        return self._f_torch

    def generate_jax_function(self, jit: bool = True) -> Callable:
        """
        Generate JAX-compatible function for dynamics

        Args:
            jit: Whether to JIT-compile the function (default: True)

        Returns:
            Callable function compatible with JAX arrays
        """
        from src.systems.base.codegen_utils import generate_jax_function

        f_with_params = self.substitute_parameters(self._f_sym)
        all_vars = self.state_vars + self.control_vars
        self._f_jax = generate_jax_function(f_with_params, all_vars, jit=jit)
        return self._f_jax

    def generate_dynamics_function(self, backend: str = "torch", **kwargs) -> Callable:
        """
        Generate dynamics function for specified backend

        Args:
            backend: 'numpy', 'torch', or 'jax'
            **kwargs: Backend-specific options (e.g., method='lambdify', jit=True)

        Returns:
            Callable function compatible with the specified backend
        """
        from src.systems.base.codegen_utils import generate_function

        f_with_params = self.substitute_parameters(self._f_sym)
        all_vars = self.state_vars + self.control_vars

        func = generate_function(f_with_params, all_vars, backend=backend, **kwargs)

        # Cache in appropriate attribute
        if backend == "numpy":
            self._f_numpy = func
        elif backend == "torch":
            self._f_torch = func
        elif backend == "jax":
            self._f_jax = func

        return func

    def forward(self, x, u):
        """
        Evaluate continuous-time dynamics: compute state derivative dx/dt = f(x, u)

        Automatically detects backend from input tensor types:
        - torch.Tensor → PyTorch computation
        - jax.Array → JAX computation
        - np.ndarray → NumPy computation

        **CRITICAL DISTINCTION**: This method returns the DERIVATIVE (rate of change)
        of the state, NOT the next state value. This is fundamentally different from
        discrete-time systems.

        Args:
            x: State tensor/array (batch_size, nx) or (nx,)
            u: Control tensor/array (batch_size, nu) or (nu,)

        Returns:
            State derivative (same shape and type as input)

        Raises:
            ValueError: If input dimensions don't match system dimensions
            TypeError: If input type is not supported
        """
        import torch

        # Detect backend from input type and dispatch
        if isinstance(x, torch.Tensor):
            return self._forward_torch(x, u)

        # Check for JAX arrays
        try:
            import jax.numpy as jnp

            if isinstance(x, jnp.ndarray):
                return self._forward_jax(x, u)
        except ImportError:
            pass

        # Default to NumPy
        if isinstance(x, np.ndarray):
            return self._forward_numpy(x, u)

        raise TypeError(
            f"Unsupported input type: {type(x)}. Expected torch.Tensor, jax.Array, or np.ndarray"
        )

    def _forward_torch(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """PyTorch backend implementation"""
        import torch

        start_time = time.time()

        # Input validation - handle edge cases
        if len(x.shape) == 0 or len(u.shape) == 0:
            raise ValueError("Input tensors must be at least 1D")

        # Check dimensions only if tensors are at least 1D
        if len(x.shape) >= 1 and x.shape[-1] != self.nx:
            raise ValueError(f"Expected state dimension {self.nx}, got {x.shape[-1]}")
        if len(u.shape) >= 1 and u.shape[-1] != self.nu:
            raise ValueError(f"Expected control dimension {self.nu}, got {u.shape[-1]}")

        if self._f_torch is None:
            self.generate_torch_function()

        # Handle batched vs single evaluation
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
            u = u.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        # Prepare arguments
        x_list = [x[:, i] for i in range(self.nx)]
        u_list = [u[:, i] for i in range(self.nu)]
        all_args = x_list + u_list

        # Call generated function
        result = self._f_torch(*all_args)

        if squeeze_output:
            result = result.squeeze(0)

        # Update performance stats
        self._perf_stats["forward_calls"] += 1
        self._perf_stats["forward_time"] += time.time() - start_time

        return result

    def _forward_jax(self, x, u):
        """JAX backend implementation"""
        import jax
        import jax.numpy as jnp

        # Input validation
        if x.ndim == 0 or u.ndim == 0:
            raise ValueError("Input arrays must be at least 1D")

        if x.ndim >= 1 and x.shape[-1] != self.nx:
            raise ValueError(f"Expected state dimension {self.nx}, got {x.shape[-1]}")
        if u.ndim >= 1 and u.shape[-1] != self.nu:
            raise ValueError(f"Expected control dimension {self.nu}, got {u.shape[-1]}")

        if not hasattr(self, "_f_jax") or self._f_jax is None:
            self.generate_jax_function()

        # Handle batched vs single evaluation
        if x.ndim == 1:
            x = jnp.expand_dims(x, 0)
            u = jnp.expand_dims(u, 0)
            squeeze_output = True
        else:
            squeeze_output = False

        # For batched computation, use vmap for efficiency
        if x.shape[0] > 1:

            @jax.vmap
            def batched_dynamics(x_i, u_i):
                x_list = [x_i[j] for j in range(self.nx)]
                u_list = [u_i[j] for j in range(self.nu)]
                return self._f_jax(*(x_list + u_list))

            result = batched_dynamics(x, u)
        else:
            # Single evaluation
            x_list = [x[0, i] for i in range(self.nx)]
            u_list = [u[0, i] for i in range(self.nu)]
            result = self._f_jax(*(x_list + u_list))
            result = jnp.expand_dims(result, 0)

        if squeeze_output:
            result = jnp.squeeze(result, 0)

        return result

    def _forward_numpy(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """NumPy backend implementation"""

        # Input validation
        if x.ndim == 0 or u.ndim == 0:
            raise ValueError("Input arrays must be at least 1D")

        if x.ndim >= 1 and x.shape[-1] != self.nx:
            raise ValueError(f"Expected state dimension {self.nx}, got {x.shape[-1]}")
        if u.ndim >= 1 and u.shape[-1] != self.nu:
            raise ValueError(f"Expected control dimension {self.nu}, got {u.shape[-1]}")

        if self._f_numpy is None:
            self.generate_numpy_function()

        # Handle batched vs single evaluation
        if x.ndim == 1:
            x_list = [x[i] for i in range(self.nx)]
            u_list = [u[i] for i in range(self.nu)]
            result = self._f_numpy(*(x_list + u_list))
            return np.array(result).flatten()
        else:
            # Batched numpy evaluation
            results = []
            for i in range(x.shape[0]):
                x_list = [x[i, j] for j in range(self.nx)]
                u_list = [u[i, j] for j in range(self.nu)]
                result = self._f_numpy(*(x_list + u_list))
                results.append(np.array(result).flatten())
            return np.stack(results)

    def linearized_dynamics(self, x, u):
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
        """
        import torch

        # Detect backend from input type and dispatch
        if isinstance(x, torch.Tensor):
            return self._linearized_dynamics_torch(x, u)

        # Check for JAX arrays
        try:
            import jax.numpy as jnp

            if isinstance(x, jnp.ndarray):
                return self._linearized_dynamics_jax(x, u)
        except ImportError:
            pass

        # Default to NumPy
        if isinstance(x, np.ndarray):
            return self._linearized_dynamics_numpy(x, u)

        raise TypeError(f"Unsupported input type: {type(x)}")

    def _linearized_dynamics_torch(self, x: torch.Tensor, u: torch.Tensor):
        """
        PyTorch implementation using cached Jacobian functions or symbolic evaluation
        """
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

        # Check if we have cached Jacobian functions
        if hasattr(self, "_A_torch_fn") and self._A_torch_fn is not None:
            # Use cached functions (faster)
            for i in range(batch_size):
                x_i = x[i]
                u_i = u[i]

                # Prepare arguments
                x_list = [x_i[j] for j in range(self.nx)]
                u_list = [u_i[j] for j in range(self.nu)]
                all_args = x_list + u_list

                # Call cached Jacobian functions
                A_batch[i] = self._A_torch_fn(*all_args)
                B_batch[i] = self._B_torch_fn(*all_args)
        else:
            # Fall back to symbolic evaluation (your existing implementation)
            for i in range(batch_size):
                # Convert to numpy - handle both 1D and potential 0D cases
                x_i = x[i] if batch_size > 1 else x.squeeze(0)
                u_i = u[i] if batch_size > 1 else u.squeeze(0)

                x_np = x_i.detach().cpu().numpy()
                u_np = u_i.detach().cpu().numpy()

                # Ensure arrays are at least 1D for SymPy Matrix
                x_np = np.atleast_1d(x_np)
                u_np = np.atleast_1d(u_np)

                A_sym, B_sym = self.linearized_dynamics_symbolic(
                    sp.Matrix(x_np), sp.Matrix(u_np)
                )
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

    def _linearized_dynamics_jax(self, x, u):
        """
        JAX implementation using automatic differentiation

        This is more efficient than symbolic -> lambdify for JAX
        """
        import jax
        import jax.numpy as jnp

        # Ensure dynamics function is available
        if not hasattr(self, "_f_jax") or self._f_jax is None:
            self.generate_jax_function()

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
            return self._f_jax(*(x_list + u_list))

        # Compute Jacobians using JAX autodiff (vmap for batching)
        @jax.vmap
        def compute_jacobians(x_i, u_i):
            # Jacobian w.r.t. state
            A = jax.jacobian(lambda x: dynamics_fn(x, u_i))(x_i)
            # Jacobian w.r.t. control
            B = jax.jacobian(lambda u: dynamics_fn(x_i, u))(u_i)
            return A, B

        A_batch, B_batch = compute_jacobians(x, u)

        if squeeze_output:
            A_batch = jnp.squeeze(A_batch, 0)
            B_batch = jnp.squeeze(B_batch, 0)

        return A_batch, B_batch

    def _linearized_dynamics_numpy(self, x: np.ndarray, u: np.ndarray):
        """
        NumPy implementation using symbolic evaluation

        This is the most reliable approach for NumPy since it doesn't have autodiff
        """

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

        # Check if we have cached NumPy Jacobian functions
        if hasattr(self, "_A_numpy_fn") and self._A_numpy_fn is not None:
            # Use cached functions
            for i in range(batch_size):
                x_i = x[i]
                u_i = u[i]

                # Prepare arguments
                x_list = [x_i[j] for j in range(self.nx)]
                u_list = [u_i[j] for j in range(self.nu)]
                all_args = x_list + u_list

                # Call cached Jacobian functions
                A_result = self._A_numpy_fn(*all_args)
                B_result = self._B_numpy_fn(*all_args)

                # Handle different output types from lambdify
                A_batch[i] = np.array(A_result, dtype=np.float64)
                B_batch[i] = np.array(B_result, dtype=np.float64)
        else:
            # Fall back to symbolic evaluation
            for i in range(batch_size):
                x_np = np.atleast_1d(x[i])
                u_np = np.atleast_1d(u[i])

                # Use symbolic Jacobians (cached)
                A_sym, B_sym = self.linearized_dynamics_symbolic(
                    sp.Matrix(x_np), sp.Matrix(u_np)
                )
                A_batch[i] = np.array(A_sym, dtype=np.float64)
                B_batch[i] = np.array(B_sym, dtype=np.float64)

        if squeeze_output:
            A_batch = np.squeeze(A_batch, 0)
            B_batch = np.squeeze(B_batch, 0)

        return A_batch, B_batch

    def linearized_observation(self, x):
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
        """
        import torch

        # Detect backend from input type and dispatch
        if isinstance(x, torch.Tensor):
            return self._linearized_observation_torch(x)

        # Check for JAX arrays
        try:
            import jax.numpy as jnp

            if isinstance(x, jnp.ndarray):
                return self._linearized_observation_jax(x)
        except ImportError:
            pass

        # Default to NumPy
        if isinstance(x, np.ndarray):
            return self._linearized_observation_numpy(x)

        raise TypeError(f"Unsupported input type: {type(x)}")

    def _linearized_observation_torch(self, x: torch.Tensor) -> torch.Tensor:
        """
        PyTorch implementation using cached Jacobian function or symbolic evaluation
        """
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

        # Check if we have cached Jacobian function
        if hasattr(self, "_C_torch_fn") and self._C_torch_fn is not None:
            # Use cached function (faster)
            for i in range(batch_size):
                x_i = x[i]

                # Prepare arguments (only state variables for observation)
                x_list = [x_i[j] for j in range(self.nx)]

                # Call cached Jacobian function
                C_batch[i] = self._C_torch_fn(*x_list)
        else:
            # Fall back to symbolic evaluation
            for i in range(batch_size):
                # Handle indexing properly
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

    def _linearized_observation_jax(self, x):
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
        if not hasattr(self, "_h_jax") or self._h_jax is None:
            from src.systems.base.codegen_utils import generate_jax_function

            h_with_params = self.substitute_parameters(self._h_sym)
            self._h_jax = generate_jax_function(
                h_with_params, self.state_vars, jit=True
            )

        # Handle batched input
        if x.ndim == 1:
            x = jnp.expand_dims(x, 0)
            squeeze_output = True
        else:
            squeeze_output = False

        # Define observation function for Jacobian computation
        def observation_fn(x_i):
            x_list = [x_i[j] for j in range(self.nx)]
            return self._h_jax(*x_list)

        # Compute Jacobian using JAX autodiff (vmap for batching)
        @jax.vmap
        def compute_jacobian(x_i):
            return jax.jacobian(observation_fn)(x_i)

        C_batch = compute_jacobian(x)

        if squeeze_output:
            C_batch = jnp.squeeze(C_batch, 0)

        return C_batch

    def _linearized_observation_numpy(self, x: np.ndarray) -> np.ndarray:
        """
        NumPy implementation using symbolic evaluation
        """

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

        # Check if we have cached NumPy Jacobian function
        if hasattr(self, "_C_numpy_fn") and self._C_numpy_fn is not None:
            # Use cached function
            for i in range(batch_size):
                x_i = x[i]

                # Prepare arguments
                x_list = [x_i[j] for j in range(self.nx)]

                # Call cached Jacobian function
                C_result = self._C_numpy_fn(*x_list)
                C_batch[i] = np.array(C_result, dtype=np.float64)
        else:
            # Fall back to symbolic evaluation
            for i in range(batch_size):
                x_np = np.atleast_1d(x[i])

                # Use symbolic Jacobian (cached)
                C_sym = self.linearized_observation_symbolic(sp.Matrix(x_np))
                C_batch[i] = np.array(C_sym, dtype=np.float64)

        if squeeze_output:
            C_batch = np.squeeze(C_batch, 0)

        return C_batch

    def h(self, x):
        """
        Evaluate output equation: y = h(x)

        Automatically detects backend from input type:
        - torch.Tensor → PyTorch computation
        - jax.Array → JAX computation
        - np.ndarray → NumPy computation

        Args:
            x: State tensor/array

        Returns:
            Output tensor/array (same type as input)
        """
        import torch

        # Detect backend from input type and dispatch
        if isinstance(x, torch.Tensor):
            return self._h_torch_eval(x)

        # Check for JAX arrays
        try:
            import jax.numpy as jnp

            if isinstance(x, jnp.ndarray):
                return self._h_jax_eval(x)
        except ImportError:
            pass

        # Default to NumPy
        if isinstance(x, np.ndarray):
            return self._h_numpy_eval(x)

        raise TypeError(f"Unsupported input type: {type(x)}")

    def _h_torch_eval(self, x: torch.Tensor) -> torch.Tensor:
        """PyTorch implementation of output equation"""

        if self._h_sym is None:
            return x

        if self._h_torch is None:
            from src.systems.base.codegen_utils import generate_torch_function

            h_with_params = self.substitute_parameters(self._h_sym)
            self._h_torch = generate_torch_function(h_with_params, self.state_vars)

        # Handle batched input
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        x_list = [x[:, i] for i in range(self.nx)]
        result = self._h_torch(*x_list)

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

    def _h_numpy_eval(self, x: np.ndarray) -> np.ndarray:
        """NumPy implementation of output equation"""

        # If no custom output function, return full state
        if self._h_sym is None:
            return x

        # Generate NumPy function for h if not cached
        if self._h_numpy is None:
            from src.systems.base.codegen_utils import generate_numpy_function

            h_with_params = self.substitute_parameters(self._h_sym)
            self._h_numpy = generate_numpy_function(h_with_params, self.state_vars)

        # Handle batched input
        if x.ndim == 1:
            x = np.expand_dims(x, 0)
            squeeze_output = True
        else:
            squeeze_output = False

        batch_size = x.shape[0]
        results = []

        for i in range(batch_size):
            x_list = [x[i, j] for j in range(self.nx)]
            result = self._h_numpy(*x_list)

            # Convert to array
            result = np.atleast_1d(np.array(result))
            results.append(result)

        result = np.stack(results)

        if squeeze_output:
            result = np.squeeze(result, 0)

        return result

    def _h_jax_eval(self, x):
        """JAX implementation of output equation"""
        import jax.numpy as jnp

        # If no custom output function, return full state
        if self._h_sym is None:
            return x

        # Generate JAX function for h if not cached
        if not hasattr(self, "_h_jax") or self._h_jax is None:
            from src.systems.base.codegen_utils import generate_jax_function

            h_with_params = self.substitute_parameters(self._h_sym)
            self._h_jax = generate_jax_function(
                h_with_params, self.state_vars, jit=True
            )

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
                return self._h_jax(*x_list)

            result = batched_observation(x)
        else:
            # Single evaluation
            x_list = [x[0, i] for i in range(self.nx)]
            result = self._h_jax(*x_list)
            result = jnp.expand_dims(result, 0)

        if squeeze_output:
            result = jnp.squeeze(result, 0)

        return result

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

    def check_equilibrium(
        self, x_eq: torch.Tensor, u_eq: torch.Tensor, tol: float = 1e-6
    ) -> Tuple[bool, float]:
        """
        Check if (x_eq, u_eq) is an equilibrium point

        Args:
            x_eq: Candidate equilibrium state
            u_eq: Candidate equilibrium control
            tol: Tolerance for considering derivative as zero

        Returns:
            (is_equilibrium, max_derivative): Boolean and max derivative magnitude
        """
        with torch.no_grad():
            dx = self.forward(
                x_eq.unsqueeze(0) if len(x_eq.shape) == 1 else x_eq,
                u_eq.unsqueeze(0) if len(u_eq.shape) == 1 else u_eq,
            )
            max_deriv = torch.abs(dx).max().item()
            is_eq = max_deriv < tol
        return is_eq, max_deriv

    def eigenvalues_at_equilibrium(self) -> np.ndarray:
        """
        Compute eigenvalues of linearization at equilibrium

        Returns:
            Eigenvalues as complex numpy array
        """
        x_eq = self.x_equilibrium.unsqueeze(0)
        u_eq = self.u_equilibrium.unsqueeze(0)
        A, _ = self.linearized_dynamics(x_eq, u_eq)
        A_np = A.squeeze().detach().cpu().numpy()
        eigenvalues = np.linalg.eigvals(A_np)
        return eigenvalues

    def is_stable_equilibrium(self, discrete_time: bool = False) -> bool:
        """
        Check if equilibrium is stable based on eigenvalues

        Args:
            discrete_time: If True, check |λ| < 1; if False, check Re(λ) < 0

        Returns:
            True if equilibrium is stable
        """
        eigs = self.eigenvalues_at_equilibrium()
        if discrete_time:
            return bool(np.all(np.abs(eigs) < 1.0))
        else:
            return bool(np.all(np.real(eigs) < 0.0))

    def clone(self):
        """Create a deep copy of the system"""
        return copy.deepcopy(self)

    def to_device(self, device: Union[str, torch.device]):
        """
        Move system to specified device

        Args:
            device: Target device ('cpu', 'cuda', or torch.device)

        Returns:
            Self for chaining
        """
        if isinstance(device, str):
            device = torch.device(device)

        # Move equilibrium points
        if hasattr(self, "_x_eq_cached"):
            self._x_eq_cached = self._x_eq_cached.to(device)
        if hasattr(self, "_u_eq_cached"):
            self._u_eq_cached = self._u_eq_cached.to(device)

        return self

    def verify_jacobians(
        self, x: torch.Tensor, u: torch.Tensor, tol: float = 1e-3
    ) -> Dict[str, Union[bool, float]]:
        """
        Verify symbolic Jacobians against numerical finite differences

        Checks:
        - A_match: Does ∂f/∂x from SymPy match autograd?
        - B_match: Does ∂f/∂u from SymPy match autograd?

        Use for:
        - Debugging symbolic derivations after system modifications
        - Ensuring code generation correctness
        - Validating against hardcoded implementations

        Args:
            x: State at which to verify (can be 1D or 2D)
            u: Control at which to verify (can be 1D or 2D)
            tol: Tolerance for considering Jacobians equal

        Returns:
            Dict with 'A_match', 'B_match' booleans and error magnitudes
        """
        # Ensure proper 2D shape (batch_size=1, dim)
        x_2d = x.reshape(1, -1) if len(x.shape) <= 1 else x
        u_2d = u.reshape(1, -1) if len(u.shape) <= 1 else u

        # Clone for autograd - keep 2D shape
        x_grad = x_2d.clone().requires_grad_(True)
        u_grad = u_2d.clone().requires_grad_(True)

        # Compute symbolic Jacobians
        A_sym, B_sym = self.linearized_dynamics(x_2d.detach(), u_2d.detach())

        # Ensure 3D shape for batch processing
        if len(A_sym.shape) == 2:
            A_sym = A_sym.unsqueeze(0)
            B_sym = B_sym.unsqueeze(0)

        # Compute numerical Jacobians via autograd
        fx = self.forward(x_grad, u_grad)  # fx shape: (1, n_outputs)

        # - First-order: n_outputs = nx (all state derivatives)
        # - Second-order: n_outputs = nq (only accelerations)
        # - Higher-order: n_outputs = nq (highest derivative only)
        if self.order == 1:
            n_outputs = self.nx
        else:
            n_outputs = self.nq

        # For higher-order systems, the Jacobians A and B are of full state-space form
        # but forward() only returns the highest derivative. We need to verify only
        # the relevant part of the Jacobians.
        A_num = torch.zeros_like(A_sym)
        B_num = torch.zeros_like(B_sym)

        if self.order == 1:
            # First-order: forward() returns dx/dt, verify full A and B
            for i in range(n_outputs):
                if fx[0, i].requires_grad:
                    grad_x = torch.autograd.grad(
                        fx[0, i], x_grad, retain_graph=True, create_graph=False
                    )[0]
                    grad_u = torch.autograd.grad(
                        fx[0, i], u_grad, retain_graph=True, create_graph=False
                    )[0]
                    A_num[0, i] = grad_x[0]  # grad_x shape: (1, nx)
                    B_num[0, i] = grad_u[0]  # grad_u shape: (1, nu)
        else:
            # Higher-order: forward() returns highest derivative only
            # The full state-space A matrix has structure:
            # For second-order: A = [[0, I], [A_accel]]
            # We verify only the A_accel part (rows nq:nx)

            for i in range(n_outputs):
                if fx[0, i].requires_grad:
                    grad_x = torch.autograd.grad(
                        fx[0, i], x_grad, retain_graph=True, create_graph=False
                    )[0]
                    grad_u = torch.autograd.grad(
                        fx[0, i], u_grad, retain_graph=True, create_graph=False
                    )[0]

                    # Place in the acceleration rows of the full state-space matrix
                    row_idx = (self.order - 1) * self.nq + i
                    A_num[0, row_idx] = grad_x[0]
                    B_num[0, row_idx] = grad_u[0]

            # For the derivative relationships (upper rows), we verify analytically
            # These should be identity blocks: dq/dt = qdot, etc.
            # The symbolic linearization already includes these, so we just copy them
            for i in range((self.order - 1) * self.nq):
                A_num[0, i] = A_sym[0, i]
                B_num[0, i] = B_sym[0, i]

        A_error = (A_sym - A_num).abs().max().item()
        B_error = (B_sym - B_num).abs().max().item()
        A_match = A_error < tol
        B_match = B_error < tol

        return {
            "A_match": bool(A_match),
            "B_match": bool(B_match),
            "A_error": float(A_error),
            "B_error": float(B_error),
        }

    def get_performance_stats(self) -> Dict[str, float]:
        """
        Get performance statistics

        Returns:
            Dict with timing and call count statistics
        """
        return {
            **self._perf_stats,
            "avg_forward_time": self._perf_stats["forward_time"]
            / max(1, self._perf_stats["forward_calls"]),
            "avg_linearization_time": self._perf_stats["linearization_time"]
            / max(1, self._perf_stats["linearization_calls"]),
        }

    def reset_performance_stats(self):
        """Reset performance counters"""
        for key in self._perf_stats:
            self._perf_stats[key] = 0.0 if "time" in key else 0

    def check_numerical_stability(
        self, x: torch.Tensor, u: torch.Tensor
    ) -> Dict[str, Union[bool, float]]:
        """
        Check for numerical issues (NaN, Inf, extreme values)

        Args:
            x: State to check (any shape)
            u: Control to check (any shape)

        Returns:
            Dict with stability indicators
        """
        # Ensure proper shape
        x_2d = x.reshape(1, -1) if len(x.shape) <= 1 else x
        u_2d = u.reshape(1, -1) if len(u.shape) <= 1 else u

        with torch.no_grad():
            dx = self.forward(x_2d, u_2d)
            return {
                "has_nan": bool(torch.isnan(dx).any().item()),
                "has_inf": bool(torch.isinf(dx).any().item()),
                "max_derivative": float(dx.abs().max().item()),
                "is_stable": bool(not (torch.isnan(dx).any() or torch.isinf(dx).any())),
            }

    def __repr__(self) -> str:
        """String representation for debugging"""
        return (
            f"{self.__class__.__name__}("
            f"nx={self.nx}, nu={self.nu}, ny={self.ny}, order={self.order})"
        )

    def __str__(self) -> str:
        """Human-readable string representation"""
        return (
            f"{self.__class__.__name__}(nx={self.nx}, nu={self.nu}, order={self.order})"
        )

    def save_config(self, filename: str):
        """
        Save system configuration to file

        Args:
            filename: Path to save configuration (supports .json, .yaml, .pt)
        """

        config = {
            "class_name": self.__class__.__name__,
            "parameters": {str(k): float(v) for k, v in self.parameters.items()},
            "order": self.order,
            "nx": self.nx,
            "nu": self.nu,
            "ny": self.ny,
        }

        if filename.endswith(".json"):
            with open(filename, "w") as f:
                json.dump(config, f, indent=2)
        elif filename.endswith(".pt"):
            torch.save(config, filename)
        else:
            raise ValueError(f"Unsupported file format: {filename}. Use .json or .pt")

        print(f"Configuration saved to {filename}")

    def get_config_dict(self) -> Dict:
        """
        Get configuration as dictionary

        Returns:
            Dict with system configuration
        """
        return {
            "class_name": self.__class__.__name__,
            "parameters": {str(k): float(v) for k, v in self.parameters.items()},
            "order": self.order,
            "nx": self.nx,
            "nu": self.nu,
            "ny": self.ny,
        }

    def lqr_control(
        self,
        Q: np.ndarray,
        R: np.ndarray,
        x_eq: Optional[torch.Tensor] = None,
        u_eq: Optional[torch.Tensor] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute LQR control gain for continuous-time system.

        **IMPORTANT**: This method linearizes the nonlinear system around the
        equilibrium point and computes the optimal gain for the linearized system.
        The resulting controller is:
        - Globally optimal for linear systems
        - Locally optimal near equilibrium for nonlinear systems
        - Performance degrades as state moves away from equilibrium

        Theory:
        ------
        Solves the continuous-time algebraic Riccati equation (CARE):
            A^T S + S A - S B R^{-1} B^T S + Q = 0

        The optimal gain is:
            K = -R^{-1} B^T S

        Control law:
            u(t) = K @ (x(t) - x_eq) + u_eq

        Cost function minimized (for linearized system):
            J = ∫[0,∞] [(x-x_eq)^T Q (x-x_eq) + (u-u_eq)^T R (u-u_eq)] dt

        Args:
            Q: State cost matrix (nx, nx). Must be positive semi-definite.
            Larger values penalize state deviations more heavily.
            R: Control cost matrix (nu, nu) or scalar for single input.
            Must be positive definite. Larger values penalize control effort.
            x_eq: Equilibrium state (uses self.x_equilibrium if None)
            u_eq: Equilibrium control (uses self.u_equilibrium if None)

        Returns:
            K: Control gain matrix (nu, nx). Control law is u = K @ (x - x_eq) + u_eq
            S: Solution to continuous-time Riccati equation (nx, nx)

        Raises:
            ValueError: If matrix dimensions are incompatible
            LinAlgError: If Riccati equation has no stabilizing solution

        Example:
            >>> # Design LQR for pendulum
            >>> pendulum = SymbolicPendulum(m=0.15, l=0.5, beta=0.1, g=9.81)
            >>> Q = np.diag([10.0, 1.0])  # Penalize angle more than velocity
            >>> R = np.array([[0.1]])      # Small control cost
            >>> K, S = pendulum.lqr_control(Q, R)
            >>>
            >>> # Apply control in simulation
            >>> controller = lambda x: K @ (x - pendulum.x_equilibrium) + pendulum.u_equilibrium

        Notes:
            - The linearization is computed at (x_eq, u_eq) using symbolic differentiation
            - For second-order systems, the full state-space linearization is used
            - The method assumes (x_eq, u_eq) is a valid equilibrium point
            - Stability is only guaranteed in a neighborhood of the equilibrium

        See Also:
            kalman_gain: Design optimal observer
            lqg_control: Combined controller and observer design
            linearized_dynamics: View the linearization used
        """
        if x_eq is None:
            x_eq = self.x_equilibrium
        if u_eq is None:
            u_eq = self.u_equilibrium

        # Ensure proper shape
        if len(x_eq.shape) == 1:
            x_eq = x_eq.unsqueeze(0)
        if len(u_eq.shape) == 1:
            u_eq = u_eq.unsqueeze(0)

        # Get linearized dynamics at equilibrium
        A, B = self.linearized_dynamics(x_eq, u_eq)
        A = A.squeeze(0).detach().cpu().numpy()
        B = B.squeeze(0).detach().cpu().numpy()

        # Ensure B is 2D (nx, nu)
        if B.ndim == 1:
            B = B.reshape(-1, 1)

        # Ensure R is 2D
        if isinstance(R, (int, float)):
            R = np.array([[R]])
        elif R.ndim == 1:
            R = np.diag(R)

        # Validate dimensions
        nx, nu = B.shape
        if A.shape != (nx, nx):
            raise ValueError(f"A must be ({nx}, {nx}), got {A.shape}")
        if Q.shape != (nx, nx):
            raise ValueError(f"Q must be ({nx}, {nx}), got {Q.shape}")
        if R.shape != (nu, nu):
            raise ValueError(f"R must be ({nu}, {nu}), got {R.shape}")

        # Solve continuous-time algebraic Riccati equation
        S = scipy.linalg.solve_continuous_are(A, B, Q, R)

        # Compute optimal gain
        K = -np.linalg.solve(R, B.T @ S)

        return K, S

    def kalman_gain(
        self,
        Q_process: Optional[np.ndarray] = None,
        R_measurement: Optional[np.ndarray] = None,
        x_eq: Optional[torch.Tensor] = None,
        u_eq: Optional[torch.Tensor] = None,
    ) -> np.ndarray:
        """
        Compute Kalman filter gain for continuous-time system.

        **IMPORTANT**: This method linearizes the nonlinear system around the
        equilibrium point and computes the optimal observer gain for the linearized
        system. The resulting observer is:
        - Globally optimal for linear systems with Gaussian noise
        - Locally optimal near equilibrium for nonlinear systems
        - Performance degrades as state moves away from equilibrium

        Theory:
        ------
        Solves the continuous-time dual Riccati equation:
            A P + P A^T - P C^T R^{-1} C P + Q = 0

        The optimal gain is:
            L = P C^T R^{-1}

        Observer dynamics:
            d x̂/dt = f(x̂, u) + L(y - h(x̂))

        For linearized system:
            d x̂/dt = A x̂ + B u + L(y - C x̂)

        Args:
            Q_process: Process noise covariance (nx, nx). Must be positive
                    semi-definite. Represents uncertainty in dynamics.
                    Default: 0.001 * I
            R_measurement: Measurement noise covariance (ny, ny) or scalar.
                        Must be positive definite. Represents sensor noise.
                        Default: 0.001 * I
            x_eq: Equilibrium state for linearization (uses self.x_equilibrium if None)
            u_eq: Equilibrium control for linearization (uses self.u_equilibrium if None)

        Returns:
            L: Kalman gain matrix (nx, ny). Observer correction term is L @ innovation

        Raises:
            ValueError: If matrix dimensions are incompatible
            LinAlgError: If dual Riccati equation has no stabilizing solution

        Example:
            >>> # Design Kalman filter for pendulum (measuring only angle)
            >>> pendulum = SymbolicPendulum(m=0.15, l=0.5, beta=0.1, g=9.81)
            >>> Q_process = np.diag([0.001, 0.01])      # Process noise
            >>> R_measurement = np.array([[0.1]])        # Measurement noise
            >>> L = pendulum.kalman_gain(Q_process, R_measurement)
            >>>
            >>> # Use in observer
            >>> observer = LinearObserver(pendulum, L)
            >>> observer.update(u, y_measured, dt=0.01)
            >>> x_estimate = observer.x_hat

        Notes:
            - The linearization is computed at (x_eq, u_eq) using symbolic differentiation
            - Q_process represents model uncertainty and unmodeled disturbances
            - R_measurement represents sensor noise characteristics
            - Larger Q_process → trust measurements more (higher gain)
            - Larger R_measurement → trust model more (lower gain)
            - The observer is guaranteed stable for the linearized system

        See Also:
            lqr_control: Design optimal controller
            lqg_control: Combined controller and observer design
            ExtendedKalmanFilter: Nonlinear state estimation
        """
        if Q_process is None:
            Q_process = np.eye(self.nx) * 1e-3
        if R_measurement is None:
            R_measurement = np.eye(self.ny) * 1e-3
        if x_eq is None:
            x_eq = self.x_equilibrium
        if u_eq is None:
            u_eq = self.u_equilibrium

        # Ensure proper shape
        if len(x_eq.shape) == 1:
            x_eq = x_eq.unsqueeze(0)

        # Get linearized dynamics
        A, _ = self.linearized_dynamics(
            x_eq, u_eq if len(u_eq.shape) > 1 else u_eq.unsqueeze(0)
        )
        A = A.squeeze(0).detach().cpu().numpy()

        C = self.linearized_observation(x_eq)
        C = C.squeeze(0).detach().cpu().numpy()

        # Ensure C is 2D (ny, nx)
        if C.ndim == 1:
            C = C.reshape(1, -1)

        # Ensure R_measurement is 2D
        if isinstance(R_measurement, (int, float)):
            R_measurement = np.array([[R_measurement]])
        elif R_measurement.ndim == 1:
            R_measurement = np.diag(R_measurement)

        # Validate dimensions
        nx = A.shape[0]
        ny = C.shape[0]

        if A.shape != (nx, nx):
            raise ValueError(f"A must be square, got {A.shape}")
        if C.shape[1] != nx:
            raise ValueError(f"C must have {nx} columns, got {C.shape}")
        if Q_process.shape != (nx, nx):
            raise ValueError(f"Q_process must be ({nx}, {nx}), got {Q_process.shape}")
        if R_measurement.shape != (ny, ny):
            raise ValueError(
                f"R_measurement must be ({ny}, {ny}), got {R_measurement.shape}"
            )

        # Solve continuous-time algebraic Riccati equation (dual problem)
        P = scipy.linalg.solve_continuous_are(A.T, C.T, Q_process, R_measurement)

        # Compute Kalman gain
        L = P @ C.T @ np.linalg.inv(R_measurement)

        return L

    def lqg_control(
        self,
        Q_lqr: np.ndarray,
        R_lqr: np.ndarray,
        Q_process: Optional[np.ndarray] = None,
        R_measurement: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute LQG controller (combined LQR controller + Kalman filter).

        **IMPORTANT**: This method designs an output feedback controller by combining:
        1. LQR controller for the linearized dynamics
        2. Kalman filter for the linearized observations

        The separation principle guarantees that for LINEAR systems:
        - Designing K and L separately is optimal
        - The closed-loop stability equals the product of controller/observer poles

        For NONLINEAR systems:
        - This provides a locally optimal solution near equilibrium
        - The separation principle does NOT hold globally
        - Performance degrades away from the equilibrium point
        - Consider adaptive or gain-scheduled approaches for large operating regions

        Theory:
        ------
        Output feedback control law:
            d x̂/dt = f(x̂, u) + L(y - h(x̂))    [Observer]
            u = K @ (x̂ - x_eq) + u_eq         [Controller based on estimate]

        For linearized system:
            Closed-loop poles = {eig(A + BK)} ∪ {eig(A - LC)}

        Args:
            Q_lqr: State cost for LQR (nx, nx)
            R_lqr: Control cost for LQR (nu, nu) or scalar
            Q_process: Process noise covariance (nx, nx). Default: 0.001 * I
            R_measurement: Measurement noise covariance (ny, ny) or scalar. Default: 0.001 * I

        Returns:
            K: LQR control gain (nu, nx)
            L: Kalman observer gain (nx, ny)

        Example:
            >>> # Design LQG controller for pendulum with noisy measurements
            >>> pendulum = SymbolicPendulum(m=0.15, l=0.5, beta=0.1, g=9.81)
            >>>
            >>> # Controller costs
            >>> Q_lqr = np.diag([10.0, 1.0])
            >>> R_lqr = np.array([[0.1]])
            >>>
            >>> # Noise covariances
            >>> Q_process = np.diag([0.001, 0.01])
            >>> R_measurement = np.array([[0.1]])
            >>>
            >>> K, L = pendulum.lqg_control(Q_lqr, R_lqr, Q_process, R_measurement)
            >>>
            >>> # Simulate with observer-based control
            >>> controller = LinearController(K, pendulum.x_equilibrium, pendulum.u_equilibrium)
            >>> observer = LinearObserver(pendulum, L)
            >>>
            >>> for t in range(steps):
            >>>     y = measure(x_true)  # Noisy measurement
            >>>     observer.update(u, y, dt)
            >>>     u = controller(observer.x_hat)
            >>>     x_true = step_dynamics(x_true, u, dt)

        Notes:
            - Separation principle: K and L can be designed independently (for linear systems)
            - The controller never sees the true state, only the estimate x̂
            - Closed-loop has 2*nx states: [x, x̂] (true state and estimate)
            - For nonlinear systems, consider EKF for the observer instead

        See Also:
            lqr_control: Controller design only
            kalman_gain: Observer design only
            lqg_closed_loop_matrix: Analyze closed-loop stability
            ExtendedKalmanFilter: Nonlinear observer alternative
        """
        K, _ = self.lqr_control(Q_lqr, R_lqr)
        L = self.kalman_gain(Q_process, R_measurement)
        return K, L

    def lqg_closed_loop_matrix(self, K: np.ndarray, L: np.ndarray) -> np.ndarray:
        """
        Compute closed-loop system matrix for LQG control.

        Returns the linearized dynamics of the augmented system [x, x̂] where:
        - x is the true state
        - x̂ is the observer's state estimate

        Theory:
        ------
        Closed-loop dynamics (linearized):
            d/dt [x  ]  = [A + BK    -BK   ] [x  ]
                 [x̂ ]   = [LC      A+BK-LC ] [x̂ ]

        Or equivalently in terms of state and estimation error e = x - x̂:
            d/dt [x]  = [A + BK   -BK ] [x]
                 [e]    [0      A - LC] [e]

        Eigenvalues:
            eig(A_cl) = {eig(A + BK)} ∪ {eig(A - LC)}

        This shows the separation principle: closed-loop poles are the union
        of controller poles and observer poles (for linear systems).

        Args:
            K: LQR control gain (nu, nx) from lqr_control()
            L: Kalman filter gain (nx, ny) from kalman_gain()

        Returns:
            A_cl: Closed-loop system matrix (2*nx, 2*nx)
                **State ordering**: [x[0], ..., x[nx-1], x̂[0], ..., x̂[nx-1]]
                    First nx elements: true state
                    Last nx elements:  estimate

        Example:
            >>> # Design LQG and analyze stability
            >>> K, L = system.lqg_control(Q_lqr, R_lqr, Q_process, R_measurement)
            >>> A_cl = system.lqg_closed_loop_matrix(K, L)
            >>>
            >>> # Check stability
            >>> eigenvalues = np.linalg.eigvals(A_cl)
            >>> is_stable = np.all(np.real(eigenvalues) < 0)
            >>> print(f"Closed-loop stable: {is_stable}")
            >>>
            >>> # Compare with open-loop
            >>> A, B = system.linearized_dynamics(x_eq, u_eq)
            >>> open_loop_eigs = np.linalg.eigvals(A)
            >>> print(f"Open-loop poles: {open_loop_eigs}")
            >>> print(f"Closed-loop poles: {eigenvalues}")

        Notes:
            - All eigenvalues should have negative real parts for stability
            - The matrix has block structure showing separation principle
            - Small entries (< 1e-6) are zeroed for numerical cleanliness
            - This is the linearized closed-loop; nonlinear behavior may differ

        See Also:
            lqg_control: Design the gains K and L
            eigenvalues_at_equilibrium: Open-loop eigenvalues
            is_stable_equilibrium: Check open-loop stability
        """
        x_eq = self.x_equilibrium.unsqueeze(0)
        u_eq = self.u_equilibrium.unsqueeze(0)

        A, B = self.linearized_dynamics(x_eq, u_eq)
        A = A.squeeze(0).detach().cpu().numpy()  # Only squeeze batch dim
        B = B.squeeze(0).detach().cpu().numpy()  # Only squeeze batch dim

        # Ensure B is 2D - if it got squeezed to 1D, reshape
        if B.ndim == 1:
            B = B.reshape(-1, 1)

        C = (
            self.linearized_observation(x_eq).squeeze(0).detach().cpu().numpy()
        )  # Only squeeze batch dim

        # Ensure C is 2D
        if C.ndim == 1:
            C = C.reshape(1, -1)

        # Ensure K is 2D (nu, nx)
        if K.ndim == 1:
            K = K.reshape(1, -1)

        # Ensure L is 2D (nx, ny)
        if L.ndim == 1:
            L = L.reshape(-1, 1)

        # Closed-loop system: [x, x̂]
        # dx/dt = Ax + B K x̂
        # dx̂/dt = A x̂ + B K x̂ + L(Cx - C x̂) = (A + B K - L C) x̂ + L C x
        A_cl = np.vstack(
            [
                np.hstack([A + B @ K, -B @ K]),  # dx/dt
                np.hstack([L @ C, A + B @ K - L @ C]),  # dx̂/dt
            ]
        )

        # Clean up near-zero entries
        A_cl[np.abs(A_cl) <= 1e-6] = 0

        return A_cl
