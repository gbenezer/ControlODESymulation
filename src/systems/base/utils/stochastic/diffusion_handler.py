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
Diffusion Handler - Code Generation for Stochastic Terms

Generates backend-specific numerical functions for evaluating diffusion terms.
Mirrors the drift handling in SymbolicDynamicalSystem's CodeGenerator.

Architecture:
    - Composes with NoiseCharacterizer for automatic analysis
    - Reuses codegen_utils.generate_function() for code generation
    - Mirrors CodeGenerator's caching and API patterns
    - Provides constant noise optimization for additive noise

This handler is responsible ONLY for code generation, not numerical evaluation.
Evaluation is handled by StochasticDynamicalSystem or integrators.

Reuses:
    - codegen_utils.generate_function() - 100% reuse for lambdify logic
    - CodeGenerator patterns - caching, reset, get_info API
    - NoiseCharacterizer - composition for automatic analysis
"""

import time
from typing import Callable, Dict, List, Optional

import numpy as np
import sympy as sp

from src.types.backends import Backend
from src.types.core import DiffusionFunction, DiffusionMatrix
from src.types.symbolic import ParameterDict, SymbolicDiffusionMatrix
from src.systems.base.utils.codegen_utils import generate_function
from src.systems.base.utils.stochastic.noise_analysis import (
    NoiseCharacteristics,
    NoiseCharacterizer,
    NoiseType,
)


class DiffusionHandler:
    """
    Handles code generation and caching for diffusion terms.

    Responsibilities:
    - Generate backend-specific g(x, u) functions via codegen_utils
    - Cache generated functions (avoid recompilation)
    - Compose with NoiseCharacterizer for automatic analysis
    - Provide constant noise matrices for additive case
    - Track generation statistics

    This class mirrors CodeGenerator's interface and patterns for consistency.

    Examples
    --------
    >>> from sympy import symbols, Matrix
    >>> x1, x2, u = symbols('x1 x2 u')
    >>>
    >>> # State-independent (additive) noise
    >>> diffusion = Matrix([[0.1], [0.2]])
    >>> handler = DiffusionHandler(diffusion, [x1, x2], [u])
    >>>
    >>> # Automatic analysis
    >>> print(handler.characteristics.noise_type)
    NoiseType.ADDITIVE
    >>>
    >>> print(handler.characteristics.recommended_solvers('jax'))
    ['sea', 'shark', 'sra1']
    >>>
    >>> # Generate function with caching
    >>> g_numpy = handler.generate_function('numpy')
    >>> g_val = g_numpy(1.0, 0.5, 0.0)  # g(x1=1.0, x2=0.5, u=0.0)
    >>>
    >>> # Second call returns cached
    >>> g_numpy_cached = handler.generate_function('numpy')
    >>> assert g_numpy is g_numpy_cached
    >>>
    >>> # For additive noise, get constant matrix
    >>> if handler.characteristics.is_additive:
    ...     G = handler.get_constant_noise('numpy')
    ...     print(G)  # [[0.1], [0.2]] - precomputed!
    """

    def __init__(
        self,
        diffusion_expr: SymbolicDiffusionMatrix,
        state_vars: List[sp.Symbol],
        control_vars: List[sp.Symbol],
        time_var: Optional[sp.Symbol] = None,
        parameters: Optional[ParameterDict] = None,
    ):
        """
        Initialize diffusion handler.

        Parameters
        ----------
        diffusion_expr : SymbolicDiffusionMatrix
            Symbolic diffusion matrix g(x, u, t), shape (nx, nw)
        state_vars : List[sp.Symbol]
            State variable symbols
        control_vars : List[sp.Symbol]
            Control variable symbols
        time_var : sp.Symbol, optional
            Time variable symbol (if time-varying diffusion)
        parameters : ParameterDict, optional
            Parameter values to substitute before code generation

        Examples
        --------
        >>> diffusion = Matrix([[sigma * x]])  # Multiplicative noise
        >>> handler = DiffusionHandler(
        ...     diffusion,
        ...     state_vars=[x],
        ...     control_vars=[u],
        ...     parameters={sigma: 0.2}
        ... )
        """
        self.diffusion_expr = diffusion_expr
        self.state_vars = state_vars
        self.control_vars = control_vars
        self.time_var = time_var
        self.parameters = parameters or {}

        # Extract dimensions
        self.nx = diffusion_expr.shape[0]
        self.nw = diffusion_expr.shape[1]

        # COMPOSE: Automatic noise analysis via NoiseCharacterizer
        self.characterizer = NoiseCharacterizer(
            diffusion_expr=diffusion_expr,
            state_vars=state_vars,
            control_vars=control_vars,
            time_var=time_var,
        )

        # Cache for generated functions (mirrors CodeGenerator pattern)
        self._diffusion_funcs: Dict[str, Optional[DiffusionFunction]] = {
            "numpy": None,
            "torch": None,
            "jax": None,
        }

        # Cache for constant noise (additive case only)
        self._constant_noise_cache: Dict[str, Optional[np.ndarray]] = {}

        # Statistics (mirrors CodeGenerator pattern)
        self._generation_stats = {"generations": 0, "cache_hits": 0, "total_time": 0.0}

    @property
    def characteristics(self) -> NoiseCharacteristics:
        """
        Get noise characteristics from automatic analysis.

        Returns
        -------
        NoiseCharacteristics
            Analysis results (cached after first access)

        Examples
        --------
        >>> char = handler.characteristics
        >>> if char.is_additive:
        ...     print("Use specialized additive-noise solver!")
        """
        return self.characterizer.characteristics

    # ========================================================================
    # Code Generation (Mirrors CodeGenerator.generate_dynamics)
    # ========================================================================

    def generate_function(self, backend: Backend, **kwargs) -> DiffusionFunction:
        """
        Generate g(x, u) function for specified backend.
        
        Parameters
        ----------
        backend : Backend
            Target backend ('numpy', 'torch', 'jax')
        **kwargs
            Additional arguments for generate_function
            
        Returns
        -------
        DiffusionFunction
            Callable g(x, u) → diffusion matrix
        """
        start_time = time.time()

        # Check cache (mirrors CodeGenerator)
        if self._diffusion_funcs[backend] is not None:
            self._generation_stats["cache_hits"] += 1
            return self._diffusion_funcs[backend]

        # Substitute parameters into expression (mirrors CodeGenerator)
        diffusion_with_params = self._substitute_parameters(self.diffusion_expr)

        # Backend-specific preprocessing (mirrors CodeGenerator)
        if backend == "torch":
            # PyTorch benefits from simplification
            diffusion_with_params = sp.simplify(diffusion_with_params)

        # Prepare input variables (mirrors CodeGenerator)
        all_vars = self.state_vars + self.control_vars
        if self.time_var is not None:
            all_vars = [self.time_var] + all_vars

        # REUSE: Use codegen_utils.generate_function()
        base_func = generate_function(
            expr=diffusion_with_params, symbols=all_vars, backend=backend, **kwargs
        )

        # Verify callable (mirrors CodeGenerator)
        if not callable(base_func):
            raise RuntimeError(f"generate_function returned non-callable: {type(base_func)}")

        # Wrap to ensure 2D output
        func = self._wrap_to_ensure_2d(base_func, backend)

        # Cache it (mirrors CodeGenerator)
        self._diffusion_funcs[backend] = func

        # Update stats
        elapsed = time.time() - start_time
        self._generation_stats["generations"] += 1
        self._generation_stats["total_time"] += elapsed

        return func

    def _wrap_to_ensure_2d(self, func: Callable, backend: Backend) -> DiffusionFunction:
        """
        Wrap function to ensure output has correct shape.

        Handles both single and batched evaluation:
        - Single input (x is 1D): Returns (nx, nw)
        - Batched input (x is 2D): Returns (batch, nx, nw) for multiplicative
                                        or (nx, nw) for additive

        The underlying generate_function() may return various shapes depending
        on input, so we intelligently reshape based on input characteristics.

        Parameters
        ----------
        func : Callable
            Base lambdified function from codegen_utils
        backend : str
            Backend name ('numpy', 'torch', 'jax')

        Returns
        -------
        Callable
            Wrapped function with proper output shape

        Notes
        -----
        For multiplicative noise with batched input:
            Input: x is (batch_size,) per state variable
            Output: Should be (batch_size, nx, nw)

        For additive noise with batched input:
            Input: x is (batch_size,) per state variable
            Output: (nx, nw) - constant, not batched
        """
        if backend == "numpy":

            def wrapped(*args):
                result = func(*args)
                result = np.atleast_2d(result)

                # Detect if batched input (first arg is array with len > 1)
                is_batched = False
                batch_size = 1

                if len(args) > 0:
                    first_arg = args[0]
                    if hasattr(first_arg, "shape") and len(first_arg.shape) > 0:
                        arg_size = len(first_arg)
                        if arg_size > 1:
                            is_batched = True
                            batch_size = arg_size

                # For additive noise, result is constant regardless of batch
                if self.characteristics.is_additive:
                    # Force to (nx, nw) even if batched input
                    if result.shape != (self.nx, self.nw):
                        result = result.reshape(self.nx, self.nw)
                    return result

                # For multiplicative noise with batched input
                if is_batched:
                    # Result from lambdify might be:
                    # - (batch_size,) for scalar output
                    # - (batch_size, nx*nw) flattened
                    # - Already correct shape

                    target_shape = (batch_size, self.nx, self.nw)

                    if result.shape == target_shape:
                        # Already correct
                        return result
                    elif result.size == batch_size * self.nx * self.nw:
                        # Reshape to target
                        return result.reshape(target_shape)
                    elif result.shape == (self.nx, self.nw):
                        # Constant result despite batched input (edge case)
                        return result
                    else:
                        # Try to reshape
                        return result.reshape(target_shape)
                else:
                    # Single input: ensure (nx, nw)
                    if result.shape != (self.nx, self.nw):
                        result = result.reshape(self.nx, self.nw)
                    return result

            return wrapped

        elif backend == "torch":

            def wrapped(*args):
                import torch

                result = func(*args)

                # Detect if batched
                is_batched = False
                batch_size = 1

                if len(args) > 0:
                    first_arg = args[0]
                    if hasattr(first_arg, "shape") and len(first_arg.shape) > 0:
                        if first_arg.shape[0] > 1:
                            is_batched = True
                            batch_size = first_arg.shape[0]

                # For additive noise, return (nx, nw) regardless
                if self.characteristics.is_additive:
                    if result.dim() == 1:
                        result = result.reshape(self.nx, self.nw)
                    elif result.dim() == 0:
                        result = result.reshape(1, 1)
                    elif result.shape != torch.Size([self.nx, self.nw]):
                        result = result.reshape(self.nx, self.nw)
                    return result

                # For multiplicative noise with batched input
                if is_batched:
                    target_shape = (batch_size, self.nx, self.nw)

                    if result.shape == target_shape:
                        return result
                    elif result.numel() == batch_size * self.nx * self.nw:
                        return result.reshape(target_shape)
                    elif result.shape == torch.Size([self.nx, self.nw]):
                        return result
                    else:
                        return result.reshape(target_shape)
                else:
                    # Single input
                    if result.dim() == 1:
                        result = result.reshape(self.nx, self.nw)
                    elif result.dim() == 0:
                        result = result.reshape(1, 1)
                    return result

            return wrapped

        elif backend == "jax":

            def wrapped(*args):
                import jax.numpy as jnp

                result = func(*args)

                # Detect if batched
                is_batched = False
                batch_size = 1

                if len(args) > 0:
                    first_arg = args[0]
                    if hasattr(first_arg, "shape") and len(first_arg.shape) > 0:
                        if first_arg.shape[0] > 1:
                            is_batched = True
                            batch_size = first_arg.shape[0]

                # For additive noise, return (nx, nw) regardless
                if self.characteristics.is_additive:
                    if result.ndim == 1:
                        result = result.reshape(self.nx, self.nw)
                    elif result.ndim == 0:
                        result = result.reshape(1, 1)
                    elif result.shape != (self.nx, self.nw):
                        result = result.reshape(self.nx, self.nw)
                    return result

                # For multiplicative noise with batched input
                if is_batched:
                    target_shape = (batch_size, self.nx, self.nw)

                    if result.shape == target_shape:
                        return result
                    elif result.size == batch_size * self.nx * self.nw:
                        return result.reshape(target_shape)
                    elif result.shape == (self.nx, self.nw):
                        return result
                    else:
                        return result.reshape(target_shape)
                else:
                    # Single input
                    if result.ndim == 1:
                        result = result.reshape(self.nx, self.nw)
                    elif result.ndim == 0:
                        result = result.reshape(1, 1)
                    return result

            return wrapped

        else:
            raise ValueError(f"Unknown backend: {backend}")

    def get_function(self, backend: Backend) -> Optional[DiffusionFunction]:
        """
        Get cached diffusion function without generating.

        Mirrors CodeGenerator.get_dynamics().

        Parameters
        ----------
        backend : Backend
            Target backend

        Returns
        -------
        Optional[DiffusionFunction]
            Cached function or None if not yet generated

        Examples
        --------
        >>> g = handler.get_function('numpy')
        >>> if g is None:
        ...     g = handler.generate_function('numpy')
        """
        return self._diffusion_funcs.get(backend)

    # ========================================================================
    # Constant Noise Optimization (Additive Noise)
    # ========================================================================

    def get_constant_noise(self, backend: Backend = "numpy") -> DiffusionMatrix:
        """
        Get constant noise matrix for additive noise.

        For additive noise, diffusion doesn't depend on state, control, or time,
        so it can be precomputed once and reused. This is a significant
        performance optimization for SDE solving.

        Parameters
        ----------
        backend : Backend
            Backend for array type

        Returns
        -------
        DiffusionMatrix
            Constant diffusion matrix, shape (nx, nw)

        Raises
        ------
        ValueError
            If noise is not additive (use generate_function instead)

        Examples
        --------
        >>> if handler.characteristics.is_additive:
        ...     G = handler.get_constant_noise('numpy')
        ...     # G is constant - precompute once, reuse everywhere!
        ...     for t in range(1000):
        ...         # No need to evaluate diffusion - just use G
        ...         pass
        """
        # Validate noise is additive
        if not self.characteristics.is_additive:
            raise ValueError(
                "get_constant_noise() only valid for additive noise.\n"
                f"Current noise type: {self.characteristics.noise_type.value}\n"
                "For state-dependent noise, use generate_function() and evaluate at each point."
            )

        # Check cache
        cache_key = f"{backend}_constant"
        if cache_key in self._constant_noise_cache:
            cached = self._constant_noise_cache[cache_key]
            # Convert cached NumPy to requested backend before returning
            return self._convert_to_backend(cached, backend)

        # Generate function (will use cache if available)
        func = self.generate_function(backend)

        # For additive noise, evaluate at arbitrary point (result is constant!)
        # Use zeros for simplicity
        if backend == "numpy":
            x_dummy = [0.0] * self.nx
            u_dummy = [0.0] * len(self.control_vars)
        elif backend == "torch":
            import torch

            x_dummy = [torch.tensor(0.0)] * self.nx
            u_dummy = [torch.tensor(0.0)] * len(self.control_vars)
        elif backend == "jax":
            import jax.numpy as jnp

            x_dummy = [jnp.array(0.0)] * self.nx
            u_dummy = [jnp.array(0.0)] * len(self.control_vars)
        else:
            raise ValueError(f"Unknown backend: {backend}")

        # Evaluate to get constant matrix
        constant_noise = func(*(x_dummy + u_dummy))

        # Convert to NumPy for universal caching
        if backend == "torch":
            constant_noise = constant_noise.detach().cpu().numpy()
        elif backend == "jax":
            constant_noise = np.array(constant_noise)

        # Ensure 2D array (nx, nw)
        constant_noise = np.atleast_2d(constant_noise)
        if constant_noise.shape[0] == 1 and self.nx > 1:
            constant_noise = constant_noise.T

        # Cache it (as NumPy)
        self._constant_noise_cache[cache_key] = constant_noise

        # Convert to requested backend before returning
        return self._convert_to_backend(constant_noise, backend)

    def _convert_to_backend(self, arr: np.ndarray, backend: Backend) -> DiffusionMatrix:
        """
        Convert NumPy array to target backend.

        Parameters
        ----------
        arr : np.ndarray
            NumPy array to convert
        backend : Backend
            Target backend ('numpy', 'torch', 'jax')

        Returns
        -------
        DiffusionMatrix
            Array in target backend format
        """
        if backend == "numpy":
            return arr
        elif backend == "torch":
            import torch

            dtype = torch.float64 if arr.dtype == np.float64 else torch.float32
            return torch.tensor(arr, dtype=dtype)
        elif backend == "jax":
            import jax.numpy as jnp

            return jnp.array(arr)
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def has_constant_noise(self) -> bool:
        """
        Check if constant noise has been computed and cached.

        Returns
        -------
        bool
            True if any backend has cached constant noise
        """
        return len(self._constant_noise_cache) > 0

    # ========================================================================
    # Parameter Substitution (Mirrors CodeGenerator)
    # ========================================================================

    def _substitute_parameters(self, expr: SymbolicDiffusionMatrix) -> SymbolicDiffusionMatrix:
        """
        Substitute parameter values into symbolic expression.

        This mirrors the parameter substitution in CodeGenerator.

        Parameters
        ----------
        expr : SymbolicDiffusionMatrix
            Expression containing parameter symbols

        Returns
        -------
        SymbolicDiffusionMatrix
            Expression with parameters substituted with numeric values

        Examples
        --------
        >>> # diffusion = [[sigma * x]]
        >>> # parameters = {sigma: 0.2}
        >>> result = handler._substitute_parameters(diffusion)
        >>> # result = [[0.2 * x]]
        """
        if not self.parameters:
            return expr

        # Substitute all parameters
        expr_with_params = expr.subs(self.parameters)

        return expr_with_params

    # ========================================================================
    # Compilation and Warmup (Mirrors CodeGenerator)
    # ========================================================================

    def compile_all(
        self, backends: Optional[List[Backend]] = None, verbose: bool = False, **kwargs
    ) -> Dict[str, float]:
        """
        Pre-compile diffusion functions for multiple backends.

        Mirrors CodeGenerator.compile_all() for consistency.

        Parameters
        ----------
        backends : List[Backend], optional
            Backends to compile (None = all available)
        verbose : bool
            Print compilation progress
        **kwargs
            Backend-specific options

        Returns
        -------
        Dict[str, float]
            Mapping backend → compilation_time

        Examples
        --------
        >>> timings = handler.compile_all(verbose=True)
        Compiling diffusion for numpy: 0.05s
        Compiling diffusion for torch: 0.12s
        Compiling diffusion for jax: 0.08s
        >>>
        >>> print(timings)
        {'numpy': 0.05, 'torch': 0.12, 'jax': 0.08}
        """
        if backends is None:
            backends = ["numpy", "torch", "jax"]

        timings = {}

        for backend in backends:
            if verbose:
                print(f"Compiling diffusion for {backend}...", end=" ")

            try:
                start = time.time()
                self.generate_function(backend, **kwargs)
                elapsed = time.time() - start
                timings[backend] = elapsed

                if verbose:
                    print(f"{elapsed:.3f}s")

            except Exception as e:
                timings[backend] = None
                if verbose:
                    print(f"FAILED ({e})")

        return timings

    def warmup(self, backend: Backend, **kwargs):
        """
        Warm up function compilation (especially useful for JAX JIT).

        Parameters
        ----------
        backend : Backend
            Backend to warm up
        **kwargs
            Test inputs for warmup call

        Examples
        --------
        >>> # JAX JIT warmup
        >>> handler.warmup('jax')
        >>> # First call compiled, subsequent calls fast
        """
        func = self.generate_function(backend)

        # Create dummy inputs for warmup
        if backend == "numpy":
            test_inputs = [0.0] * (self.nx + len(self.control_vars))
        elif backend == "torch":
            import torch

            test_inputs = [torch.tensor(0.0)] * (self.nx + len(self.control_vars))
        elif backend == "jax":
            import jax.numpy as jnp

            test_inputs = [jnp.array(0.0)] * (self.nx + len(self.control_vars))

        # Call once to trigger compilation
        _ = func(*test_inputs)

    # ========================================================================
    # Cache Management (Mirrors CodeGenerator)
    # ========================================================================

    def reset_cache(self, backends: Optional[List[Backend]] = None):
        """
        Clear cached functions for specified backends.

        Mirrors CodeGenerator.reset_cache().

        Parameters
        ----------
        backends : List[Backend], optional
            Backends to reset (None = all)

        Examples
        --------
        >>> # Clear all caches
        >>> handler.reset_cache()
        >>>
        >>> # Clear only torch (e.g., after device change)
        >>> handler.reset_cache(['torch'])
        """
        if backends is None:
            backends = ["numpy", "torch", "jax"]

        for backend in backends:
            self._diffusion_funcs[backend] = None

            # Also clear constant noise cache for this backend
            cache_key = f"{backend}_constant"
            if cache_key in self._constant_noise_cache:
                del self._constant_noise_cache[cache_key]

    def is_compiled(self, backend: Backend) -> bool:
        """
        Check if diffusion function is compiled for backend.

        Mirrors CodeGenerator.is_compiled() but simpler (only one function type).

        Parameters
        ----------
        backend : Backend
            Backend to check

        Returns
        -------
        bool
            True if function is cached/compiled

        Examples
        --------
        >>> if not handler.is_compiled('jax'):
        ...     handler.generate_function('jax')
        """
        return self._diffusion_funcs.get(backend) is not None

    # ========================================================================
    # Information & Statistics (Mirrors CodeGenerator)
    # ========================================================================

    def get_info(self) -> Dict[str, any]:
        """
        Get comprehensive information about diffusion handler state.

        Mirrors CodeGenerator.get_info() for consistent API.

        Returns
        -------
        Dict[str, any]
            Status, characteristics, compilation state, statistics

        Examples
        --------
        >>> info = handler.get_info()
        >>> print(info)
        {
            'dimensions': {'nx': 2, 'nw': 1},
            'noise_type': 'additive',
            'characteristics': {...},
            'compiled': {'numpy': True, 'torch': False, 'jax': True},
            'constant_noise_cached': True,
            'statistics': {...}
        }
        """
        char = self.characteristics

        return {
            "dimensions": {"nx": self.nx, "nw": self.nw, "num_parameters": len(self.parameters)},
            "noise_type": char.noise_type.value,
            "characteristics": {
                "is_additive": char.is_additive,
                "is_multiplicative": char.is_multiplicative,
                "is_diagonal": char.is_diagonal,
                "is_scalar": char.is_scalar,
                "depends_on": {
                    "state": char.depends_on_state,
                    "control": char.depends_on_control,
                    "time": char.depends_on_time,
                },
                "state_dependencies": [str(s) for s in char.state_dependencies],
                "control_dependencies": [str(s) for s in char.control_dependencies],
            },
            "compiled": {
                backend: self.is_compiled(backend) for backend in ["numpy", "torch", "jax"]
            },
            "constant_noise_cached": self.has_constant_noise(),
            "statistics": self.get_stats(),
        }

    def get_stats(self) -> Dict[str, any]:
        """
        Get generation statistics.

        Returns
        -------
        Dict[str, any]
            Generation count, cache hits, timing

        Examples
        --------
        >>> stats = handler.get_stats()
        >>> print(f"Cache hit rate: {stats['cache_hit_rate']:.1%}")
        """
        total_calls = self._generation_stats["generations"] + self._generation_stats["cache_hits"]

        return {
            "generations": self._generation_stats["generations"],
            "cache_hits": self._generation_stats["cache_hits"],
            "total_calls": total_calls,
            "cache_hit_rate": (self._generation_stats["cache_hits"] / max(1, total_calls)),
            "total_time": self._generation_stats["total_time"],
            "avg_generation_time": (
                self._generation_stats["total_time"] / max(1, self._generation_stats["generations"])
            ),
        }

    def reset_stats(self):
        """
        Reset generation statistics.

        Examples
        --------
        >>> handler.reset_stats()
        >>> stats = handler.get_stats()
        >>> assert stats['generations'] == 0
        """
        self._generation_stats = {"generations": 0, "cache_hits": 0, "total_time": 0.0}

    # ========================================================================
    # Optimization Hints
    # ========================================================================

    def can_optimize_for_additive(self) -> bool:
        """
        Check if additive-noise optimizations are applicable.

        Returns
        -------
        bool
            True if diffusion is constant and can be precomputed
        """
        return self.characteristics.is_additive

    def get_optimization_opportunities(self) -> Dict[str, bool]:
        """
        Identify optimization opportunities based on noise structure.

        Returns
        -------
        Dict[str, bool]
            Flags for various optimization strategies

        Examples
        --------
        >>> opts = handler.get_optimization_opportunities()
        >>> if opts['precompute_diffusion']:
        ...     G = handler.get_constant_noise()
        ...     # Use G directly instead of calling diffusion(x, u)
        """
        char = self.characteristics

        return {
            "precompute_diffusion": char.is_additive,
            "use_diagonal_solver": char.is_diagonal,
            "use_scalar_solver": char.is_scalar,
            "vectorize_easily": char.is_additive or char.is_diagonal,
            "cache_diffusion": not char.depends_on_state,  # Time-varying control only
        }

    # ========================================================================
    # String Representations (Mirrors CodeGenerator)
    # ========================================================================

    def __repr__(self) -> str:
        """String representation for debugging."""
        compiled_backends = [
            b for b in ["numpy", "torch", "jax"] if self._diffusion_funcs[b] is not None
        ]

        return (
            f"DiffusionHandler("
            f"nx={self.nx}, nw={self.nw}, "
            f"type={self.characteristics.noise_type.value}, "
            f"compiled={compiled_backends})"
        )

    def __str__(self) -> str:
        """Human-readable string."""
        compiled_count = sum(
            1 for b in ["numpy", "torch", "jax"] if self._diffusion_funcs[b] is not None
        )

        return (
            f"DiffusionHandler("
            f"({self.nx}, {self.nw}), "
            f"{self.characteristics.noise_type.value}, "
            f"{compiled_count}/3 backends)"
        )


# ============================================================================
# Utility Functions
# ============================================================================


def create_diffusion_handler(
    diffusion_expr: SymbolicDiffusionMatrix, state_vars: List[sp.Symbol], control_vars: List[sp.Symbol], **kwargs
) -> DiffusionHandler:
    """
    Convenience function for creating diffusion handlers.

    Parameters
    ----------
    diffusion_expr : SymbolicDiffusionMatrix
        Symbolic diffusion matrix
    state_vars : List[sp.Symbol]
        State variables
    control_vars : List[sp.Symbol]
        Control variables
    **kwargs
        Additional options (time_var, parameters)

    Returns
    -------
    DiffusionHandler
        Configured handler

    Examples
    --------
    >>> handler = create_diffusion_handler(diffusion, [x1, x2], [u])
    """
    return DiffusionHandler(diffusion_expr, state_vars, control_vars, **kwargs)
