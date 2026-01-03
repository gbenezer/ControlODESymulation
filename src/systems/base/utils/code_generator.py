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
Code Generator for SymbolicDynamicalSystem

Orchestrates generation and caching of numerical functions from symbolic expressions.

Manages:
- Forward dynamics functions: f(x, u)
- Output functions: h(x)
- Jacobian matrices: A, B (dynamics), C (observation)
- Function caching per backend
- Compilation and warmup
- Cache invalidation

This class is the high-level orchestrator that uses codegen_utils for
the low-level SymPy → executable code conversion.
"""

import time
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

import sympy as sp

# Import from centralized type system
from src.types.backends import Backend
from src.types.core import DynamicsFunction, OutputFunction

if TYPE_CHECKING:
    from src.systems.base.symbolic_dynamical_system import SymbolicDynamicalSystem


class CodeGenerator:
    """
    Orchestrates code generation and caching for a dynamical system.

    Manages generation of numerical functions from symbolic expressions,
    with per-backend caching to avoid redundant compilation.

    Example:
        >>> code_gen = CodeGenerator(system)
        >>>
        >>> # Generate dynamics function (with caching)
        >>> f_numpy = code_gen.generate_dynamics('numpy')
        >>> f_numpy_again = code_gen.generate_dynamics('numpy')  # Returns cached
        >>> assert f_numpy is f_numpy_again
        >>>
        >>> # Compile all backends
        >>> timings = code_gen.compile_all(backends=['numpy', 'torch', 'jax'])
    """

    def __init__(self, system: "SymbolicDynamicalSystem"):
        """
        Initialize code generator for a system.

        Args:
            system: The dynamical system to generate code for
        """
        self.system = system

        # Cache for generated functions (backend → function)
        self._f_funcs: Dict[str, Optional[DynamicsFunction]] = {
            "numpy": None,
            "torch": None,
            "jax": None,
        }

        self._h_funcs: Dict[str, Optional[OutputFunction]] = {
            "numpy": None,
            "torch": None,
            "jax": None,
        }

        # Cache for Jacobian functions
        self._A_funcs: Dict[str, Optional[Callable]] = {"numpy": None, "torch": None, "jax": None}

        self._B_funcs: Dict[str, Optional[Callable]] = {"numpy": None, "torch": None, "jax": None}

        self._C_funcs: Dict[str, Optional[Callable]] = {"numpy": None, "torch": None, "jax": None}

        # Symbolic Jacobian cache (shared across backends)
        self._A_sym_cache: Optional[sp.Matrix] = None
        self._B_sym_cache: Optional[sp.Matrix] = None
        self._C_sym_cache: Optional[sp.Matrix] = None

    # ========================================================================
    # Dynamics Function Generation
    # ========================================================================

    def generate_dynamics(self, backend: Backend, **kwargs) -> DynamicsFunction:
        """
        Generate f(x, u) function for specified backend.

        Uses caching - if function already generated, returns cached version.

        Args:
            backend: Target backend ('numpy', 'torch', 'jax')
            **kwargs: Backend-specific options (e.g., jit=True for JAX)

        Returns:
            DynamicsFunction: (x, u) → dx/dt

        Example:
            >>> f_numpy = code_gen.generate_dynamics('numpy')
            >>> dx = f_numpy(x_vals, u_vals)
        """
        # Check cache
        if self._f_funcs[backend] is not None:
            return self._f_funcs[backend]

        # Import low-level utility
        from src.systems.base.utils.codegen_utils import generate_function

        # Substitute parameters into symbolic expression
        f_with_params = self.system.substitute_parameters(self.system._f_sym)

        # Backend-specific preprocessing
        if backend == "torch":
            # PyTorch benefits from simplification
            f_with_params = sp.simplify(f_with_params)

        # Prepare input variables (state + control)
        all_vars = self.system.state_vars + self.system.control_vars

        # Generate function using low-level utility
        func = generate_function(f_with_params, all_vars, backend=backend, **kwargs)

        # Verify it's callable
        if not callable(func):
            raise RuntimeError(f"generate_function returned non-callable object: {type(func)}")

        # Cache it
        self._f_funcs[backend] = func

        return func

    def get_dynamics(self, backend: Backend) -> Optional[DynamicsFunction]:
        """
        Get cached dynamics function without generating.

        Args:
            backend: Target backend

        Returns:
            Cached function or None if not yet generated
        """
        return self._f_funcs.get(backend)

    # ========================================================================
    # Output Function Generation
    # ========================================================================

    def generate_output(self, backend: Backend, **kwargs) -> Optional[OutputFunction]:
        """
        Generate h(x) function for specified backend.

        Returns None if system has no custom output function (uses identity).

        Args:
            backend: Target backend ('numpy', 'torch', 'jax')
            **kwargs: Backend-specific options

        Returns:
            OutputFunction: (x) → y, or None if no custom output

        Example:
            >>> h_numpy = code_gen.generate_output('numpy')
            >>> if h_numpy is not None:
            >>>     y = h_numpy(x_vals)
        """
        # No custom output function
        if self.system._h_sym is None:
            return None

        # Check cache
        if self._h_funcs[backend] is not None:
            return self._h_funcs[backend]

        # Import low-level utility
        from src.systems.base.utils.codegen_utils import generate_function

        # Substitute parameters
        h_with_params = self.system.substitute_parameters(self.system._h_sym)

        # Generate function using only state variables
        func = generate_function(h_with_params, self.system.state_vars, backend=backend, **kwargs)

        # Verify it's callable
        if not callable(func):
            raise RuntimeError(f"generate_function returned non-callable object: {type(func)}")

        # Cache it
        self._h_funcs[backend] = func

        return func

    def get_output(self, backend: Backend) -> Optional[OutputFunction]:
        """
        Get cached output function without generating.

        Args:
            backend: Target backend

        Returns:
            Cached function or None if not yet generated
        """
        return self._h_funcs.get(backend)

    # ========================================================================
    # Jacobian Generation
    # ========================================================================

    def _compute_symbolic_jacobians(self):
        """
        Compute and cache symbolic Jacobians (if not already done).

        Computes:
        - A = ∂f/∂x (dynamics w.r.t. state)
        - B = ∂f/∂u (dynamics w.r.t. control)
        - C = ∂h/∂x (output w.r.t. state)
        """
        # Compute A and B if not cached
        if self._A_sym_cache is None:
            self._A_sym_cache = self.system._f_sym.jacobian(self.system.state_vars)

            # For autonomous systems (nu=0 or no control_vars), create empty B matrix directly
            if hasattr(self.system, "nu") and self.system.nu == 0:
                # B should be (nq, 0) for first-order or (nq, 0) for higher-order
                # The actual state-space form will be constructed in LinearizationEngine
                n_outputs = self.system.nq if self.system.order > 1 else self.system.nx
                self._B_sym_cache = sp.zeros(n_outputs, 0)
            elif len(self.system.control_vars) == 0:
                # Fallback: check if control_vars is empty (for systems without nu attribute)
                n_outputs = (
                    self.system.nq
                    if hasattr(self.system, "order") and self.system.order > 1
                    else self.system.nx
                )
                self._B_sym_cache = sp.zeros(n_outputs, 0)
            else:
                # Non-autonomous: compute Jacobian w.r.t. control variables
                self._B_sym_cache = self.system._f_sym.jacobian(self.system.control_vars)

        # Compute C if not cached and output function exists
        if self._C_sym_cache is None and self.system._h_sym is not None:
            self._C_sym_cache = self.system._h_sym.jacobian(self.system.state_vars)

    # TODO: Change to a TypedDict-based return?
    def generate_dynamics_jacobians(self, backend: Backend, **kwargs) -> Tuple[Callable, Callable]:
        """
        Generate A and B Jacobian functions.

        Args:
            backend: Target backend
            **kwargs: Backend-specific options

        Returns:
            Tuple of (A_func, B_func) where:
                A_func: (x, u) → ∂f/∂x
                B_func: (x, u) → ∂f/∂u

        Example:
            >>> A_func, B_func = code_gen.generate_dynamics_jacobians('numpy')
            >>> A = A_func(x_vals, u_vals)
            >>> B = B_func(x_vals, u_vals)
        """
        # Check cache
        if self._A_funcs[backend] is not None and self._B_funcs[backend] is not None:
            return self._A_funcs[backend], self._B_funcs[backend]

        # Compute symbolic Jacobians if needed
        self._compute_symbolic_jacobians()

        # Import low-level utility
        from src.systems.base.utils.codegen_utils import generate_function

        # Substitute parameters
        A_with_params = self.system.substitute_parameters(self._A_sym_cache)
        B_with_params = self.system.substitute_parameters(self._B_sym_cache)

        # Prepare input variables (state + control for evaluation point)
        all_vars = self.system.state_vars + self.system.control_vars

        # Generate functions
        A_func = generate_function(A_with_params, all_vars, backend=backend, **kwargs)
        B_func = generate_function(B_with_params, all_vars, backend=backend, **kwargs)

        # Cache them
        self._A_funcs[backend] = A_func
        self._B_funcs[backend] = B_func

        return A_func, B_func

    # TODO: Change to a TypedDict-based return?
    def generate_observation_jacobian(self, backend: Backend, **kwargs) -> Optional[Callable]:
        """
        Generate C Jacobian function (∂h/∂x).

        Returns None if no custom output function.

        Args:
            backend: Target backend
            **kwargs: Backend-specific options

        Returns:
            C_func: (x) → ∂h/∂x, or None if no custom output
        """
        # No custom output function
        if self.system._h_sym is None:
            return None

        # Check cache
        if self._C_funcs[backend] is not None:
            return self._C_funcs[backend]

        # Compute symbolic Jacobian if needed
        self._compute_symbolic_jacobians()

        # Import low-level utility
        from src.systems.base.utils.codegen_utils import generate_function

        # Substitute parameters
        C_with_params = self.system.substitute_parameters(self._C_sym_cache)

        # Generate function (only state variables needed)
        C_func = generate_function(C_with_params, self.system.state_vars, backend=backend, **kwargs)

        # Cache it
        self._C_funcs[backend] = C_func

        return C_func

    # TODO: Change to a TypedDict-based return?
    def get_jacobians(
        self,
        backend: Backend,
    ) -> Tuple[Optional[Callable], Optional[Callable], Optional[Callable]]:
        """
        Get cached Jacobian functions without generating.

        Args:
            backend: Target backend

        Returns:
            Tuple of (A_func, B_func, C_func) or None if not yet generated
        """
        return (self._A_funcs.get(backend), self._B_funcs.get(backend), self._C_funcs.get(backend))

    # ========================================================================
    # Compilation and Warmup
    # ========================================================================

    # TODO: Change to a TypedDict-based return?
    def compile_all(
        self,
        backends: Optional[List[Backend]] = None,
        include_jacobians: bool = False,
        verbose: bool = False,
        **kwargs,
    ) -> Dict[str, Dict[str, float]]:
        """
        Pre-compile functions for multiple backends.

        Useful for:
        - Reducing first-call latency
        - Validating code generation
        - Warming up JIT compilers (JAX)

        Args:
            backends: List of backends to compile (None = all available)
            include_jacobians: Also compile Jacobian functions
            verbose: Print compilation progress
            **kwargs: Backend-specific options

        Returns:
            Dict mapping backend → function_name → compilation_time

        Example:
            >>> timings = code_gen.compile_all(
            ...     backends=['numpy', 'torch'],
            ...     include_jacobians=True,
            ...     verbose=True
            ... )
            Compiling numpy...
              f: 0.05s
              h: 0.03s
              A, B: 0.08s
            Compiling torch...
              f: 0.12s
        """
        if backends is None:
            backends = ["numpy", "torch", "jax"]

        all_timings = {}

        for backend in backends:
            if verbose:
                print(f"Compiling {backend}...")

            backend_timings = {}

            # Compile dynamics
            try:
                start = time.time()
                self.generate_dynamics(backend, **kwargs)
                backend_timings["f"] = time.time() - start

                if verbose:
                    print(f"  f: {backend_timings['f']:.3f}s")
            except Exception as e:
                if verbose:
                    print(f"  f: FAILED ({e})")
                backend_timings["f"] = None

            # Compile output (if exists)
            if self.system._h_sym is not None:
                try:
                    start = time.time()
                    self.generate_output(backend, **kwargs)
                    backend_timings["h"] = time.time() - start

                    if verbose:
                        print(f"  h: {backend_timings['h']:.3f}s")
                except Exception as e:
                    if verbose:
                        print(f"  h: FAILED ({e})")
                    backend_timings["h"] = None

            # Compile Jacobians (if requested)
            if include_jacobians:
                try:
                    start = time.time()
                    self.generate_dynamics_jacobians(backend, **kwargs)
                    backend_timings["A_B"] = time.time() - start

                    if verbose:
                        print(f"  A, B: {backend_timings['A_B']:.3f}s")
                except Exception as e:
                    if verbose:
                        print(f"  A, B: FAILED ({e})")
                    backend_timings["A_B"] = None

                # Observation Jacobian
                if self.system._h_sym is not None:
                    try:
                        start = time.time()
                        self.generate_observation_jacobian(backend, **kwargs)
                        backend_timings["C"] = time.time() - start

                        if verbose:
                            print(f"  C: {backend_timings['C']:.3f}s")
                    except Exception as e:
                        if verbose:
                            print(f"  C: FAILED ({e})")
                        backend_timings["C"] = None

            all_timings[backend] = backend_timings

        return all_timings

    # ========================================================================
    # Cache Management
    # ========================================================================

    def reset_cache(self, backends: Optional[List[Backend]] = None):
        """
        Clear cached functions for specified backends.

        Useful when:
        - Changing devices (need recompilation)
        - Freeing memory
        - Debugging

        Args:
            backends: List of backends to reset (None = all)

        Example:
            >>> # Clear all caches
            >>> code_gen.reset_cache()
            >>>
            >>> # Clear only torch cache (e.g., after device change)
            >>> code_gen.reset_cache(['torch'])
        """
        if backends is None:
            backends = ["numpy", "torch", "jax"]

        for backend in backends:
            self._f_funcs[backend] = None
            self._h_funcs[backend] = None
            self._A_funcs[backend] = None
            self._B_funcs[backend] = None
            self._C_funcs[backend] = None

    # TODO: Change to a TypedDict-based return?
    def is_compiled(self, backend: Backend) -> Dict[str, bool]:
        """
        Check which functions are compiled for a backend.

        Args:
            backend: Backend to check

        Returns:
            Dict mapping function_name → is_compiled

        Example:
            >>> status = code_gen.is_compiled('numpy')
            >>> print(status)
            {'f': True, 'h': False, 'A': True, 'B': True, 'C': False}
        """
        return {
            "f": self._f_funcs[backend] is not None,
            "h": self._h_funcs[backend] is not None,
            "A": self._A_funcs[backend] is not None,
            "B": self._B_funcs[backend] is not None,
            "C": self._C_funcs[backend] is not None,
        }

    # TODO: Change to a TypedDict-based return?
    def get_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about code generation status.

        Returns:
            Dict with compilation status for all backends

        Example:
            >>> info = code_gen.get_info()
            >>> print(info)
            {
                'numpy': {'f': True, 'h': True, 'A': True, 'B': True, 'C': False},
                'torch': {'f': True, 'h': False, 'A': False, 'B': False, 'C': False},
                'jax': {'f': False, 'h': False, 'A': False, 'B': False, 'C': False}
            }
        """
        return {backend: self.is_compiled(backend) for backend in ["numpy", "torch", "jax"]}

    # ========================================================================
    # String Representations
    # ========================================================================

    def __repr__(self) -> str:
        """String representation for debugging"""
        compiled_backends = [
            backend for backend in ["numpy", "torch", "jax"] if self._f_funcs[backend] is not None
        ]
        return f"CodeGenerator(compiled_backends={compiled_backends})"

    def __str__(self) -> str:
        """Human-readable string"""
        compiled_count = sum(
            1 for backend in ["numpy", "torch", "jax"] if self._f_funcs[backend] is not None
        )
        return f"CodeGenerator({compiled_count}/3 backends compiled)"
