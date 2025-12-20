import copy
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Union, Optional, Tuple, List, Dict, Callable
import json
import time
import sympy as sp
import numpy as np
from contextlib import contextmanager

# necessary sub-object import
from src.systems.base.equilibrium_handler import EquilibriumHandler

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

        # Backend configuration
        self._default_backend = "numpy"
        self._preferred_device = "cpu"

        # Cached functions
        self._f_numpy: Optional[Callable] = None
        self._f_torch: Optional[Callable] = None
        self._f_jax: Optional[Callable] = None
        self._h_numpy_func: Optional[Callable] = None
        self._h_torch_func: Optional[Callable] = None
        self._h_jax_func: Optional[Callable] = None

        # Cached Jacobians
        self._A_sym_cached: Optional[sp.Matrix] = None
        self._B_sym_cached: Optional[sp.Matrix] = None
        self._C_sym_cached: Optional[sp.Matrix] = None

        self._initialized: bool = False

        # Call template method
        self.define_system(*args, **kwargs)
        self._validate_system()

        # COMPOSITION: Delegate equilibrium management
        self.equilibria = EquilibriumHandler(self.nx, self.nu)

        # for backward compatibility
        self._perf_stats = {
            "forward_calls": 0,
            "forward_time": 0.0,
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

    # Validation and properties
    def _validate_system(self) -> bool:
        """
        Validate that the system is properly defined.

        Checks:
        1. Required attributes are set
        2. Types are correct
        3. Dimensions are consistent
        4. Symbolic expressions are valid
        5. Parameters appear in expressions
        6. No duplicate symbols
        7. System order is consistent with state dimension

        Returns:
            True if valid

        Raises:
            ValueError: With detailed error messages if invalid
        """
        errors = []
        warnings_list = []

        # ================================================================
        # 1. REQUIRED ATTRIBUTES
        # ================================================================

        if not self.state_vars:
            errors.append("state_vars is empty")

        if not self.control_vars:
            errors.append("control_vars is empty")

        if self._f_sym is None:
            errors.append("_f_sym is not defined")

        # ================================================================
        # 2. TYPE VALIDATION
        # ================================================================

        # Check state_vars are Symbols
        if self.state_vars:
            for i, var in enumerate(self.state_vars):
                if not isinstance(var, sp.Symbol):
                    errors.append(
                        f"state_vars[{i}] = {var} is not a SymPy Symbol (got {type(var)})"
                    )

        # Check control_vars are Symbols
        if self.control_vars:
            for i, var in enumerate(self.control_vars):
                if not isinstance(var, sp.Symbol):
                    errors.append(
                        f"control_vars[{i}] = {var} is not a SymPy Symbol (got {type(var)})"
                    )

        # Check output_vars are Symbols (if defined)
        if self.output_vars:
            for i, var in enumerate(self.output_vars):
                if not isinstance(var, sp.Symbol):
                    errors.append(
                        f"output_vars[{i}] = {var} is not a SymPy Symbol (got {type(var)})"
                    )

        # Check _f_sym is Matrix
        if self._f_sym is not None and not isinstance(self._f_sym, sp.Matrix):
            errors.append(f"_f_sym must be sp.Matrix, got {type(self._f_sym)}")

        # Check _h_sym is Matrix (if defined)
        if self._h_sym is not None and not isinstance(self._h_sym, sp.Matrix):
            errors.append(f"_h_sym must be sp.Matrix, got {type(self._h_sym)}")

        # Check parameters dict has Symbol keys
        if self.parameters:
            for key, value in self.parameters.items():
                if not isinstance(key, sp.Symbol):
                    errors.append(f"Parameter key {key} is not a SymPy Symbol")
                if not isinstance(value, (int, float, np.number)):
                    errors.append(f"Parameter value for {key} must be numeric, got {type(value)}")

        # ================================================================
        # 3. DIMENSION CONSISTENCY
        # ================================================================

        # Check _f_sym dimensions (if it is a valid symbolic Matrix,
        # otherwise the above check will get it)
        if self._f_sym is not None and isinstance(self._f_sym, sp.Matrix):
            expected_rows = self.nq if self.order > 1 else self.nx
            actual_rows = self._f_sym.shape[0]

            if actual_rows != expected_rows:
                errors.append(
                    f"_f_sym has {actual_rows} rows but expected {expected_rows} "
                    f"(order={self.order}, nx={self.nx}, nq={self.nq})"
                )

            if self._f_sym.shape[1] != 1:
                errors.append(f"_f_sym must be column vector, got shape {self._f_sym.shape}")

        # Check _h_sym dimensions (if defined)
        if self._h_sym is not None and isinstance(self._h_sym, sp.Matrix):
            if self._h_sym.shape[1] != 1:
                errors.append(f"_h_sym must be column vector, got shape {self._h_sym.shape}")

            if self.output_vars and len(self.output_vars) != self._h_sym.shape[0]:
                errors.append(
                    f"output_vars has {len(self.output_vars)} elements but "
                    f"_h_sym has {self._h_sym.shape[0]} rows"
                )

        # Check system order vs state dimension
        if self.order > 1:
            if self.nx % self.order != 0:
                errors.append(
                    f"For order={self.order} system, nx={self.nx} must be divisible by order. "
                    f"Got nx % order = {self.nx % self.order}"
                )

        # ================================================================
        # 4. SYMBOLIC EXPRESSION VALIDATION
        # ================================================================

        if (
            self._f_sym is not None
            and self.state_vars
            and self.control_vars
            and isinstance(self._f_sym, sp.Matrix)
        ):
            # Get all symbols that appear in _f_sym
            f_symbols = self._f_sym.free_symbols

            # Expected symbols: state_vars + control_vars + parameter keys
            expected_symbols = set(
                self.state_vars + self.control_vars + list(self.parameters.keys())
            )

            # Check for undefined symbols
            undefined_symbols = f_symbols - expected_symbols
            if undefined_symbols:
                errors.append(
                    f"_f_sym contains undefined symbols: {undefined_symbols}. "
                    f"All symbols must be in state_vars, control_vars, or parameters."
                )

            # Check that dynamics actually depend on states (warning, not error)
            state_symbols = set(self.state_vars)
            if not (f_symbols & state_symbols):
                warnings_list.append(
                    "_f_sym does not depend on any state variables. "
                    "Is this intentional? (e.g., autonomous system)"
                )

        # Check output expression (if defined)
        if self._h_sym is not None and self.state_vars and isinstance(self._h_sym, sp.Matrix):
            h_symbols = self._h_sym.free_symbols
            state_and_params = set(self.state_vars + list(self.parameters.keys()))

            # Output should NOT depend on control
            control_symbols = set(self.control_vars)
            if h_symbols & control_symbols:
                errors.append(
                    f"_h_sym contains control variables {h_symbols & control_symbols}. "
                    f"Output h(x) should only depend on states, not controls."
                )

            # Check for undefined symbols in output
            undefined = h_symbols - state_and_params
            if undefined:
                errors.append(f"_h_sym contains undefined symbols: {undefined}")

        # ================================================================
        # 5. DUPLICATE SYMBOL DETECTION
        # ================================================================

        all_vars = self.state_vars + self.control_vars + self.output_vars
        all_var_names = [str(var) for var in all_vars]

        # Check for duplicate variable names
        if len(all_var_names) != len(set(all_var_names)):
            from collections import Counter

            counts = Counter(all_var_names)
            duplicates = [name for name, count in counts.items() if count > 1]
            errors.append(f"Duplicate variable names found: {duplicates}")

        # ================================================================
        # 6. PARAMETER USAGE VALIDATION
        # ================================================================

        if self.parameters and self._f_sym is not None and isinstance(self._f_sym, sp.Matrix):
            param_symbols = set(self.parameters.keys())
            used_params = self._f_sym.free_symbols & param_symbols
            unused_params = param_symbols - used_params

            # Warn about unused parameters
            if unused_params:
                warnings_list.append(
                    f"Parameters {unused_params} are defined but not used in _f_sym. "
                    f"This might be intentional (used in _h_sym) or an error."
                )

        # ================================================================
        # 7. NUMERICAL VALIDITY
        # ================================================================

        if self.parameters:
            for symbol, value in self.parameters.items():
                # Check for NaN or Inf
                if not np.isfinite(value):
                    errors.append(f"Parameter {symbol} has non-finite value: {value}")

                # Check for negative values where they might be problematic
                # (This is domain-specific, could be customized per system)
                param_name = str(symbol)
                if param_name in ["m", "mass", "l", "length", "I", "inertia"]:
                    if value <= 0:
                        errors.append(f"Physical parameter {symbol} = {value} should be positive")

        # ================================================================
        # 8. ORDER VALIDATION
        # ================================================================

        if not isinstance(self.order, int):
            errors.append(f"order must be int, got {type(self.order)}")
        elif self.order < 1:
            errors.append(f"order must be >= 1, got {self.order}")
        elif self.order > 5:
            warnings_list.append(
                f"order = {self.order} is unusually high. " f"Are you sure this is correct?"
            )

        # ================================================================
        # 9. BACKEND VALIDATION
        # ================================================================

        if hasattr(self, "_default_backend"):
            valid_backends = ["numpy", "torch", "jax"]
            if self._default_backend not in valid_backends:
                errors.append(
                    f"_default_backend = '{self._default_backend}' is invalid. "
                    f"Must be one of {valid_backends}"
                )

        # ================================================================
        # 10. CONSISTENCY CHECKS
        # ================================================================

        # For second-order systems, check state_vars structure
        if self.order == 2 and len(self.state_vars) >= 2:
            # Common pattern: [q, q_dot] or [q1, q2, q1_dot, q2_dot]
            state_names = [str(var) for var in self.state_vars]
            nq = self.nq

            # Check if first half are positions, second half velocities
            has_dot_pattern = any("dot" in name.lower() or "'" in name for name in state_names[nq:])
            if not has_dot_pattern:
                warnings_list.append(
                    f"Second-order system but state_vars don't follow [q, q̇] pattern. "
                    f"State names: {state_names}. Consider naming derivatives with '_dot' suffix."
                )

        # ================================================================
        # 11. DEVICE VALIDATION
        # ================================================================

        if hasattr(self, "_preferred_device") and self._preferred_device:
            device_str = self._preferred_device.lower()

            # Basic validation - device string format
            valid_device_prefixes = ["cpu", "cuda", "gpu", "tpu"]

            if not any(device_str.startswith(prefix) for prefix in valid_device_prefixes):
                warnings_list.append(
                    f"_preferred_device = '{self._preferred_device}' has unusual format. "
                    f"Expected formats: 'cpu', 'cuda', 'cuda:0', 'gpu:0', 'tpu:0'"
                )

            # Warn if GPU device specified but default backend is NumPy
            if self._default_backend == "numpy" and device_str != "cpu":
                warnings_list.append(
                    f"NumPy backend doesn't support devices. "
                    f"_preferred_device = '{self._preferred_device}' will be ignored. "
                    f"Consider setting backend to 'torch' or 'jax' for GPU support."
                )

        # # ================================================================
        # # 12. EQUILIBRIUM VALIDATION (Optional)
        # # ================================================================

        # # Verify the default "origin" equilibrium is actually an equilibrium
        # try:
        #     x_origin = self.equilibria.get_x('origin', backend='numpy')
        #     u_origin = self.equilibria.get_u('origin', backend='numpy')

        #     # This requires _f_numpy to be generated, so only check if already cached
        #     if self._f_numpy is not None:
        #         dx = self._forward_numpy(x_origin, u_origin)
        #         max_deriv = np.abs(dx).max()

        #         if max_deriv > 1e-6:
        #             warnings_list.append(
        #                 f"Origin equilibrium may not be valid: max|f(0,0)| = {max_deriv:.2e}. "
        #                 f"For systems where origin is not an equilibrium, add a custom equilibrium."
        #             )
        # except Exception:
        #     # Skip if equilibria not properly initialized yet
        #     pass

        # ================================================================
        # REPORT ERRORS AND WARNINGS
        # ================================================================

        # Print warnings (don't fail validation)
        if warnings_list:
            import warnings as warn_module

            for warning in warnings_list:
                warn_module.warn(f"System validation warning: {warning}", UserWarning)

        # Fail if errors found
        if errors:
            error_msg = "System validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
            error_msg += "\n\n" + "=" * 70
            error_msg += "\nCOMMON FIXES:"
            error_msg += "\n  1. Use Symbol objects as parameter keys: {m: 1.0} not {'m': 1.0}"
            error_msg += "\n  2. Ensure _f_sym is a sp.Matrix, not a list"
            error_msg += "\n  3. Check that state_vars and control_vars are lists of Symbols"
            error_msg += "\n  4. Verify system order matches state dimension"
            error_msg += "\n" + "=" * 70
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

    # Utility functions
    def set_default_backend(self, backend: str, device: Optional[str] = None):
        """
        Set default backend for this system.

        Args:
            backend: 'numpy', 'torch', or 'jax'
            device: Optional device ('cpu', 'cuda', 'gpu:0', etc.)

        Example:
            >>> system.set_default_backend('jax', device='gpu:0')
            >>> dx = system(x, u, backend='default')  # Uses JAX on GPU
        """
        valid_backends = ["numpy", "torch", "jax"]
        if backend not in valid_backends:
            raise ValueError(f"Invalid backend '{backend}'. Must be one of {valid_backends}")

        # Check if backend is available
        self._check_backend_available(backend)

        self._default_backend = backend
        if device is not None:
            self._preferred_device = device

        return self

    def _detect_backend(self, x) -> str:
        """Detect backend from input type"""
        try:
            import torch

            if isinstance(x, torch.Tensor):
                return "torch"
        except ImportError:
            pass

        try:
            import jax.numpy as jnp

            if isinstance(x, jnp.ndarray):
                return "jax"
        except ImportError:
            pass

        if isinstance(x, np.ndarray):
            return "numpy"

        raise TypeError(f"Unknown input type: {type(x)}")

    def _check_backend_available(self, backend: str):
        """Raise error if backend not available"""
        if backend == "torch":
            try:
                import torch
            except ImportError:
                raise RuntimeError("PyTorch backend not available. Install with: pip install torch")
        elif backend == "jax":
            try:
                import jax
            except ImportError:
                raise RuntimeError("JAX backend not available. Install with: pip install jax")

    def _convert_to_backend(self, arr: ArrayLike, backend: str):
        """Convert array to target backend"""
        # Already correct backend
        current_backend = self._detect_backend(arr)
        if current_backend == backend:
            return arr

        # Convert to NumPy first (common intermediate)
        if current_backend == "torch":
            arr_np = arr.detach().cpu().numpy()
        elif current_backend == "jax":
            arr_np = np.array(arr)
        else:
            arr_np = arr

        # Convert to target backend
        if backend == "numpy":
            return arr_np
        elif backend == "torch":
            import torch

            return torch.tensor(arr_np, dtype=torch.float32, device=self._preferred_device)
        elif backend == "jax":
            import jax.numpy as jnp
            from jax import device_put, devices

            arr_jax = jnp.array(arr_np)
            if self._preferred_device != "cpu":
                target_device = devices(self._preferred_device)[0]
                arr_jax = device_put(arr_jax, target_device)
            return arr_jax

        raise ValueError(f"Unknown backend: {backend}")

    def to_device(self, device: str) -> "SymbolicDynamicalSystem":
        """
        Set preferred device for PyTorch/JAX backends.

        Args:
            device: Device string ('cpu', 'cuda', 'cuda:0', 'gpu:0', etc.)

        Returns:
            Self for method chaining

        Example:
            >>> system.to_device('cuda')
            >>> system.set_default_backend('torch')
            >>> # All torch operations now use CUDA
            >>> dx = system(x, u, backend='default')

        Note:
            This only affects backends that support devices (PyTorch, JAX).
            NumPy always uses CPU.
        """
        self._preferred_device = device

        # Clear cached functions that need recompilation for new device
        # (Functions are device-agnostic for NumPy but device-specific for torch/jax)
        if self._default_backend in ["torch", "jax"]:
            self._clear_backend_cache(self._default_backend)

        return self

    def _clear_backend_cache(self, backend: str):
        """Clear cached functions for a backend (needed after device change)"""
        if backend == "torch":
            self._f_torch = None
            self._h_torch_func = None
            # Note: Keep symbolic Jacobian caches (_A_torch_fn, etc.)
            # They're device-agnostic after generation
        elif backend == "jax":
            self._f_jax = None
            self._h_jax_func = None

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
        # Determine backend from first argument (usually x)
        if backend == "default":
            target_backend = self._default_backend
        elif backend is None:
            target_backend = self._detect_backend(args[0])
        else:
            target_backend = backend

        # Check availability
        self._check_backend_available(target_backend)

        # Get backend-specific method
        method_name = f"{method_prefix}_{target_backend}"
        method = getattr(self, method_name)

        # Call it
        return method(*args)

    def get_backend_info(self) -> Dict[str, any]:
        """
        Get information about current backend configuration and availability.

        Returns:
            Dict with backend status, device info, and compiled functions

        Example:
            >>> info = system.get_backend_info()
            >>> print(info)
            {
                'default_backend': 'torch',
                'preferred_device': 'cuda:0',
                'available_backends': ['numpy', 'torch', 'jax'],
                'compiled_backends': ['numpy', 'torch'],
                'torch_available': True,
                'jax_available': True
            }
        """
        # Check which backends are available
        available = ["numpy"]  # Always available

        try:
            import torch

            available.append("torch")
            torch_available = True
        except ImportError:
            torch_available = False

        try:
            import jax

            available.append("jax")
            jax_available = True
        except ImportError:
            jax_available = False

        # Check which backends have compiled functions
        compiled = []
        if self._f_numpy is not None:
            compiled.append("numpy")
        if self._f_torch is not None:
            compiled.append("torch")
        if self._f_jax is not None:
            compiled.append("jax")

        return {
            "default_backend": self._default_backend,
            "preferred_device": self._preferred_device,
            "available_backends": available,
            "compiled_backends": compiled,
            "torch_available": torch_available,
            "jax_available": jax_available,
            "initialized": self._initialized,
        }

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
        """
        Clear cached compiled functions for specified backends.

        Useful when:
        - Changing devices
        - Freeing memory
        - Debugging code generation

        Args:
            backends: List of backends to reset (None = all)

        Example:
            >>> # Clear all caches
            >>> system.reset_caches()
            >>>
            >>> # Clear only PyTorch cache
            >>> system.reset_caches(['torch'])
        """
        if backends is None:
            backends = ["numpy", "torch", "jax"]

        for backend in backends:
            # Clear dynamics functions
            setattr(self, f"_f_{backend}", None)
            setattr(self, f"_h_{backend}_func", None)

            # Optionally clear Jacobian caches too
            # (Usually don't need to since they're symbolic)

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

        # Compile functions
        print(f"Warming up {backend} backend...")
        self._generate_dynamics_function(backend)

        if self._h_sym is not None:
            self._generate_output_function(backend)

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
        """
        Temporarily switch backend and device.

        Example:
            >>> # Default is NumPy
            >>> dx1 = system(x, u)  # NumPy
            >>>
            >>> # Temporarily use JAX
            >>> with system.use_backend('jax', device='gpu:0'):
            >>>     dx2 = system(x, u, backend='default')  # JAX on GPU
            >>>
            >>> # Back to NumPy
            >>> dx3 = system(x, u)  # NumPy again
        """
        # Save current state
        old_backend = self._default_backend
        old_device = self._preferred_device

        try:
            # Set new backend/device
            self.set_default_backend(backend, device)
            yield self
        finally:
            # Restore original state
            self._default_backend = old_backend
            self._preferred_device = old_device

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
        """
        Get performance statistics.

        Returns:
            Dict with timing and call count statistics
        """
        return {
            **self._perf_stats,
            "avg_forward_time": (
                self._perf_stats["forward_time"] / max(1, self._perf_stats["forward_calls"])
            ),
            "avg_linearization_time": (
                self._perf_stats["linearization_time"]
                / max(1, self._perf_stats["linearization_calls"])
            ),
        }

    def reset_performance_stats(self):
        """Reset performance counters"""
        for key in self._perf_stats:
            self._perf_stats[key] = 0.0 if "time" in key else 0

    # Symbolic methods
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

        TODO: needs adaptation to equilibrium handling
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

            A_highest = self._f_sym.jacobian(self.state_vars)  # Jacobian of highest derivative
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

        TODO: needs adaptation to equilibrium handling
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

        Functions are generated automatically on first use (lazy initialization),
        but pre-compilation can:
        - Reduce first-call latency
        - Validate code generation
        - Warm up JIT compilers

        Args:
            backends: List of backends ('numpy', 'torch', 'jax').
                     If None, compiles for all available backends.
            verbose: Print compilation progress
            **kwargs: Backend-specific options

        Returns:
            Dict mapping backend names to compilation times

        Example:
            >>> # Compile everything upfront
            >>> system = SymbolicPendulum()
            >>> timings = system.compile(verbose=True)
            Compiling numpy... 0.05s
            Compiling torch... 0.12s
            Compiling jax... 0.89s (includes JIT warmup)
            >>>
            >>> # Now all backends are ready
            >>> dx = system(x_numpy, u_numpy)  # Instant
            >>> dx = system(x_torch, u_torch)  # Instant
        """
        import time

        if backends is None:
            # Auto-detect available backends
            backends = []
            backends.append("numpy")  # Always available

            try:
                import torch

                backends.append("torch")
            except ImportError:
                pass

            try:
                import jax

                backends.append("jax")
            except ImportError:
                pass

        timings = {}

        for backend in backends:
            start = time.time()

            if verbose:
                print(f"Compiling {backend}...", end=" ", flush=True)

            try:
                self._generate_dynamics_function(backend, **kwargs)
                elapsed = time.time() - start
                timings[backend] = elapsed

                if verbose:
                    print(f"{elapsed:.2f}s")

            except Exception as e:
                if verbose:
                    print(f"FAILED: {e}")
                timings[backend] = None

        return timings

    def _generate_dynamics_function(self, backend: str, **kwargs) -> Callable:
        """
        INTERNAL: Generate dynamics function for backend.

        Called automatically on first use or explicitly via compile().
        """
        from src.systems.base.codegen_utils import generate_function

        # Check if already cached
        cache_attr = f"_f_{backend}"
        if getattr(self, cache_attr, None) is not None:
            return getattr(self, cache_attr)

        # Substitute parameters
        f_with_params = self.substitute_parameters(self._f_sym)

        # Backend-specific preprocessing
        if backend == "torch":
            # PyTorch benefits from symbolic simplification
            f_with_params = sp.simplify(f_with_params)

        all_vars = self.state_vars + self.control_vars

        # Generate function
        func = generate_function(f_with_params, all_vars, backend=backend, **kwargs)

        # Cache
        setattr(self, cache_attr, func)

        return func

    # forward methods
    def forward(self, x: ArrayLike, u: ArrayLike, backend: Optional[str] = None) -> ArrayLike:
        """
        Evaluate continuous-time dynamics: compute state derivative dx/dt = f(x, u) with optional backend override.

        Args:
            x: State (array/tensor)
            u: Control (array/tensor)
            backend: Override backend ('numpy', 'torch', 'jax')
                    If None, auto-detect from input type

        Returns:
            State derivative (type matches backend, not necessarily input)

        Examples:
            >>> # Auto-detect
            >>> dx = system(x_numpy, u_numpy)  # Returns NumPy
            >>>
            >>> # Explicit backend (converts input)
            >>> dx = system(x_numpy, u_numpy, backend='torch')  # Returns PyTorch!
            >>>
            >>> # Use configured default
            >>> system.set_default_backend('jax')
            >>> dx = system(x_numpy, u_numpy, backend='default')  # Returns JAX
        """
        # Convert inputs if forcing different backend
        if backend is not None:
            # Determine target backend
            if backend == "default":
                target_backend = self._default_backend
            else:
                target_backend = backend

            # Convert if input type doesn't match target
            input_backend = self._detect_backend(x)
            if input_backend != target_backend:
                x = self._convert_to_backend(x, target_backend)
                u = self._convert_to_backend(u, target_backend)

        return self._dispatch_to_backend("_forward", backend, x, u)

    def _forward_torch(self, x: "torch.Tensor", u: "torch.Tensor") -> "torch.Tensor":
        """PyTorch backend implementation"""
        import torch

        start_time = time.time()

        # Input validation
        if len(x.shape) == 0 or len(u.shape) == 0:
            raise ValueError("Input tensors must be at least 1D")

        if len(x.shape) >= 1 and x.shape[-1] != self.nx:
            raise ValueError(f"Expected state dimension {self.nx}, got {x.shape[-1]}")
        if len(u.shape) >= 1 and u.shape[-1] != self.nu:
            raise ValueError(f"Expected control dimension {self.nu}, got {u.shape[-1]}")

        # Generate function if not cached
        if self._f_torch is None:
            self._generate_dynamics_function("torch")  # ← Use consolidated method

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

            # Ensure at least 1D
            if result.ndim == 0:
                result = result.unsqueeze(0)

        return result

        # Update performance stats
        self._perf_stats["forward_calls"] += 1
        self._perf_stats["forward_time"] += time.time() - start_time

        return result

    def _forward_jax(self, x: "jnp.ndarray", u: "jnp.ndarray") -> "jnp.ndarray":
        """JAX backend implementation"""
        import jax
        import jax.numpy as jnp

        start_time = time.time()

        # Input validation
        if x.ndim == 0 or u.ndim == 0:
            raise ValueError("Input arrays must be at least 1D")

        if x.ndim >= 1 and x.shape[-1] != self.nx:
            raise ValueError(f"Expected state dimension {self.nx}, got {x.shape[-1]}")
        if u.ndim >= 1 and u.shape[-1] != self.nu:
            raise ValueError(f"Expected control dimension {self.nu}, got {u.shape[-1]}")

        # Generate function if not cached
        if self._f_jax is None:
            self._generate_dynamics_function("jax", jit=True)  # ← Use consolidated method

        # Handle batched vs single evaluation
        if x.ndim == 1:
            x = jnp.expand_dims(x, 0)
            u = jnp.expand_dims(u, 0)
            squeeze_output = True
        else:
            squeeze_output = False

        # For batched computation, use vmap
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
            result = result.squeeze(0)

            # Ensure at least 1D
            if result.ndim == 0:
                result = result.unsqueeze(0)

        # Update performance stats
        self._perf_stats["forward_calls"] += 1
        self._perf_stats["forward_time"] += time.time() - start_time

        return result

    def _forward_numpy(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """NumPy backend implementation"""

        start_time = time.time()

        # Input validation
        if x.ndim == 0 or u.ndim == 0:
            raise ValueError("Input arrays must be at least 1D")

        if x.ndim >= 1 and x.shape[-1] != self.nx:
            raise ValueError(f"Expected state dimension {self.nx}, got {x.shape[-1]}")
        if u.ndim >= 1 and u.shape[-1] != self.nu:
            raise ValueError(f"Expected control dimension {self.nu}, got {u.shape[-1]}")

        # Generate function if not cached
        if self._f_numpy is None:
            self._generate_dynamics_function("numpy")  # ← Use consolidated method

        # Handle batched vs single evaluation
        if x.ndim == 1:
            x_list = [x[i] for i in range(self.nx)]
            u_list = [u[i] for i in range(self.nu)]
            result = self._f_numpy(*(x_list + u_list))
            result = np.array(result).flatten()
        else:
            # Batched evaluation
            results = []
            for i in range(x.shape[0]):
                x_list = [x[i, j] for j in range(self.nx)]
                u_list = [u[i, j] for j in range(self.nu)]
                result = self._f_numpy(*(x_list + u_list))
                results.append(np.array(result).flatten())
            result = np.stack(results)

        # Update performance stats (ADD THIS)
        self._perf_stats["forward_calls"] += 1
        self._perf_stats["forward_time"] += time.time() - start_time

        return result

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
        """
        JAX implementation using automatic differentiation

        This is more efficient than symbolic -> lambdify for JAX
        """
        import jax
        import jax.numpy as jnp

        # Ensure dynamics function is available
        if not hasattr(self, "_f_jax") or self._f_jax is None:
            self._generate_dynamics_function("jax", jit=True)

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

    def _linearized_dynamics_numpy(
        self, x: np.ndarray, u: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
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
                A_sym, B_sym = self.linearized_dynamics_symbolic(sp.Matrix(x_np), sp.Matrix(u_np))
                A_batch[i] = np.array(A_sym, dtype=np.float64)
                B_batch[i] = np.array(B_sym, dtype=np.float64)

        if squeeze_output:
            A_batch = np.squeeze(A_batch, 0)
            B_batch = np.squeeze(B_batch, 0)

        return A_batch, B_batch

    def _generate_output_function(self, backend: str, **kwargs):
        """Generate output function h(x) for backend"""
        from src.systems.base.codegen_utils import generate_function

        # Check if already cached
        cache_attr = f"_h_{backend}_func"
        if getattr(self, cache_attr, None) is not None:
            return getattr(self, cache_attr)

        # No custom output - return identity
        if self._h_sym is None:
            return None

        # Generate function
        h_with_params = self.substitute_parameters(self._h_sym)
        func = generate_function(h_with_params, self.state_vars, backend=backend, **kwargs)

        # Cache
        setattr(self, cache_attr, func)

        return func

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
        if not hasattr(self, "_h_jax_func") or self._h_jax_func is None:
            self._generate_output_function("jax", jit=True)

        # Handle batched input
        if x.ndim == 1:
            x = jnp.expand_dims(x, 0)
            squeeze_output = True
        else:
            squeeze_output = False

        # Define observation function for Jacobian computation
        def observation_fn(x_i):
            x_list = [x_i[j] for j in range(self.nx)]
            return self._h_jax_func(*x_list)

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

        if self._h_torch_func is None:
            try:
                self._generate_output_function("torch")
            except Exception as e:
                raise RuntimeError(f"Failed to generate torch output function: {e}") from e

        # Verify function was generated
        if self._h_torch_func is None:
            raise RuntimeError("Torch output function is still None after generation")

        # Verify it's callable
        if not callable(self._h_torch_func):
            raise TypeError(
                f"Generated torch output function is not callable: {type(self._h_torch_func)}"
            )

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
            result = self._h_torch_func(*x_list)
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
        if self._h_numpy_func is None:
            try:
                self._generate_output_function("numpy")
            except Exception as e:
                raise RuntimeError(f"Failed to generate output function: {e}") from e

        # Verify function was generated
        if self._h_numpy_func is None:
            raise RuntimeError("Output function is still None after generation attempt")

        # Verify it's callable
        if not callable(self._h_numpy_func):
            raise TypeError(
                f"Generated output function is not callable: {type(self._h_numpy_func)}"
            )

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
                result = self._h_numpy_func(*x_list)

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
        if self._h_jax_func is None:
            self._generate_output_function("jax", jit=True)

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
                return self._h_jax_func(*x_list)

            result = batched_observation(x)
        else:
            # Single evaluation
            x_list = [x[0, i] for i in range(self.nx)]
            result = self._h_jax_func(*x_list)
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
