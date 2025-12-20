import copy
from contextlib import contextmanager
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Union, Optional, Tuple, List, Dict
import json
import sympy as sp
import numpy as np

# necessary sub-object import
from src.systems.base.utils.equilibrium_handler import EquilibriumHandler
from src.systems.base.utils.backend_manager import BackendManager
from src.systems.base.utils.symbolic_validator import SymbolicValidator
from src.systems.base.utils.code_generator import CodeGenerator
from src.systems.base.utils.dynamics_evaluator import DynamicsEvaluator
from src.systems.base.utils.linearization_engine import LinearizationEngine
from src.systems.base.utils.observation_engine import ObservationEngine

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

        # COMPOSITION: Delegate linearization
        self._linearization = LinearizationEngine(self, self._code_gen, self.backend)

        # COMPOSITION: Delegate observation
        self._observation = ObservationEngine(self, self._code_gen, self.backend)

        # COMPOSITION: Delegate equilibrium management
        self.equilibria = EquilibriumHandler(self.nx, self.nu)

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

    def _clear_backend_cache(self, backend: str):
        """Clear cached functions for a backend"""
        self._code_gen.reset_cache([backend])

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
        # Get stats from components
        dynamics_stats = self._dynamics.get_stats()
        linearization_stats = self._linearization.get_stats()

        return {
            "forward_calls": dynamics_stats["calls"],
            "forward_time": dynamics_stats["total_time"],
            "avg_forward_time": dynamics_stats["avg_time"],
            "linearization_calls": linearization_stats["calls"],
            "linearization_time": linearization_stats["total_time"],
            "avg_linearization_time": linearization_stats["avg_time"],
        }

    def reset_performance_stats(self):
        """Reset performance counters"""
        self._dynamics.reset_stats()
        self._linearization.reset_stats()

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
        """Compute symbolic linearization A = df/dx, B = df/du"""
        return self._linearization.compute_symbolic(x_eq, u_eq)

    def linearized_observation_symbolic(self, x_eq: Optional[sp.Matrix] = None) -> sp.Matrix:
        """Compute symbolic linearization C = dh/dx"""
        return self._observation.compute_symbolic(x_eq)

    def verify_jacobians(
        self, x: ArrayLike, u: ArrayLike, tol: float = 1e-3, backend: str = "torch"
    ) -> Dict[str, Union[bool, float]]:
        """Verify symbolic Jacobians against automatic differentiation."""
        return self._linearization.verify_jacobians(x, u, backend, tol)

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
        """Numerical evaluation of linearized dynamics: A = ∂f/∂x, B = ∂f/∂u"""
        return self._linearization.compute_dynamics(x, u, backend)

    # Linearized observation methods
    def linearized_observation(self, x: ArrayLike, backend: Optional[str] = None) -> ArrayLike:
        """Numerical evaluation of output linearization: C = ∂h/∂x"""
        return self._observation.compute_jacobian(x, backend)

    # Nonlinear output observation methods
    def h(self, x: ArrayLike, backend: Optional[str] = None) -> ArrayLike:
        """Evaluate output equation: y = h(x)"""
        return self._observation.evaluate(x, backend)

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
