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

from typing import Callable, Dict, List, Optional
import time
import sympy as sp
import numpy as np

from src.systems.base.utils.codegen_utils import generate_function
from src.systems.base.utilsstochastic.noise_analysis import (
    NoiseCharacterizer,
    NoiseCharacteristics,
    NoiseType
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
        diffusion_expr: sp.Matrix,
        state_vars: List[sp.Symbol],
        control_vars: List[sp.Symbol],
        time_var: Optional[sp.Symbol] = None,
        parameters: Optional[Dict[sp.Symbol, float]] = None
    ):
        """
        Initialize diffusion handler.
        
        Parameters
        ----------
        diffusion_expr : sp.Matrix
            Symbolic diffusion matrix g(x, u, t), shape (nx, nw)
        state_vars : List[sp.Symbol]
            State variable symbols
        control_vars : List[sp.Symbol]
            Control variable symbols
        time_var : sp.Symbol, optional
            Time variable symbol (if time-varying diffusion)
        parameters : Dict[sp.Symbol, float], optional
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
        
        # ✅ COMPOSE: Automatic noise analysis via NoiseCharacterizer
        self.characterizer = NoiseCharacterizer(
            diffusion_expr=diffusion_expr,
            state_vars=state_vars,
            control_vars=control_vars,
            time_var=time_var
        )
        
        # Cache for generated functions (mirrors CodeGenerator pattern)
        self._diffusion_funcs: Dict[str, Optional[Callable]] = {
            'numpy': None,
            'torch': None,
            'jax': None
        }
        
        # Cache for constant noise (additive case only)
        self._constant_noise_cache: Dict[str, Optional[np.ndarray]] = {}
        
        # Statistics (mirrors CodeGenerator pattern)
        self._generation_stats = {
            'generations': 0,
            'cache_hits': 0,
            'total_time': 0.0
        }
    
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
    
    def generate_function(self, backend: str, **kwargs) -> Callable:
        """
        Generate g(x, u) function for specified backend.
        
        Uses caching - if already generated, returns cached version.
        Mirrors CodeGenerator.generate_dynamics() API.
        
        Parameters
        ----------
        backend : str
            Target backend ('numpy', 'torch', 'jax')
        **kwargs
            Backend-specific options:
            - jit : bool (JAX only) - Whether to JIT compile
        
        Returns
        -------
        Callable
            Function with signature:
            - g(x1, x2, ..., xn, u1, u2, ..., um) → diffusion matrix
            
            Returns diffusion matrix as 2D array, shape (nx, nw)
        
        Examples
        --------
        >>> g_numpy = handler.generate_function('numpy')
        >>> 
        >>> # Call with unpacked scalars
        >>> import numpy as np
        >>> x_vals = [1.0, 0.5]  # x1=1.0, x2=0.5
        >>> u_vals = [0.0]       # u=0.0
        >>> g_matrix = g_numpy(*x_vals, *u_vals)
        >>> print(g_matrix.shape)  # (2, 1)
        >>> 
        >>> # JAX with JIT
        >>> g_jax = handler.generate_function('jax', jit=True)
        """
        start_time = time.time()
        
        # Check cache (mirrors CodeGenerator)
        if self._diffusion_funcs[backend] is not None:
            self._generation_stats['cache_hits'] += 1
            return self._diffusion_funcs[backend]
        
        # Substitute parameters into expression (mirrors CodeGenerator)
        diffusion_with_params = self._substitute_parameters(self.diffusion_expr)
        
        # Backend-specific preprocessing (mirrors CodeGenerator)
        if backend == 'torch':
            # PyTorch benefits from simplification
            diffusion_with_params = sp.simplify(diffusion_with_params)
        
        # Prepare input variables (mirrors CodeGenerator)
        all_vars = self.state_vars + self.control_vars
        if self.time_var is not None:
            all_vars = [self.time_var] + all_vars
        
        # ✅ REUSE: Use codegen_utils.generate_function()
        # This is the SAME function used by CodeGenerator for drift!
        func = generate_function(
            expr=diffusion_with_params,
            symbols=all_vars,
            backend=backend,
            **kwargs
        )
        
        # Verify callable (mirrors CodeGenerator)
        if not callable(func):
            raise RuntimeError(
                f"generate_function returned non-callable: {type(func)}"
            )
        
        # Cache it (mirrors CodeGenerator)
        self._diffusion_funcs[backend] = func
        
        # Update stats
        elapsed = time.time() - start_time
        self._generation_stats['generations'] += 1
        self._generation_stats['total_time'] += elapsed
        
        return func
    
    def get_function(self, backend: str) -> Optional[Callable]:
        """
        Get cached diffusion function without generating.
        
        Mirrors CodeGenerator.get_dynamics().
        
        Parameters
        ----------
        backend : str
            Target backend
        
        Returns
        -------
        Callable or None
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
    
    def get_constant_noise(self, backend: str = 'numpy') -> np.ndarray:
        """
        Get constant noise matrix for additive noise.
        
        For additive noise, diffusion doesn't depend on state, control, or time,
        so it can be precomputed once and reused. This is a significant
        performance optimization for SDE solving.
        
        Parameters
        ----------
        backend : str
            Backend for array type
        
        Returns
        -------
        np.ndarray
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
            return self._constant_noise_cache[cache_key]
        
        # Generate function (will use cache if available)
        func = self.generate_function(backend)
        
        # For additive noise, evaluate at arbitrary point (result is constant!)
        # Use zeros for simplicity
        if backend == 'numpy':
            x_dummy = [0.0] * self.nx
            u_dummy = [0.0] * len(self.control_vars)
        elif backend == 'torch':
            import torch
            x_dummy = [torch.tensor(0.0)] * self.nx
            u_dummy = [torch.tensor(0.0)] * len(self.control_vars)
        elif backend == 'jax':
            import jax.numpy as jnp
            x_dummy = [jnp.array(0.0)] * self.nx
            u_dummy = [jnp.array(0.0)] * len(self.control_vars)
        else:
            raise ValueError(f"Unknown backend: {backend}")
        
        # Evaluate to get constant matrix
        constant_noise = func(*(x_dummy + u_dummy))
        
        # Convert to NumPy for universal caching
        if backend == 'torch':
            constant_noise = constant_noise.detach().cpu().numpy()
        elif backend == 'jax':
            constant_noise = np.array(constant_noise)
        
        # Ensure 2D array (nx, nw)
        constant_noise = np.atleast_2d(constant_noise)
        if constant_noise.shape[0] == 1 and self.nx > 1:
            constant_noise = constant_noise.T
        
        # Cache it
        self._constant_noise_cache[cache_key] = constant_noise
        
        return constant_noise
    
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
    
    def _substitute_parameters(self, expr: sp.Matrix) -> sp.Matrix:
        """
        Substitute parameter values into symbolic expression.
        
        This mirrors the parameter substitution in CodeGenerator.
        
        Parameters
        ----------
        expr : sp.Matrix
            Expression containing parameter symbols
        
        Returns
        -------
        sp.Matrix
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
        self,
        backends: Optional[List[str]] = None,
        verbose: bool = False,
        **kwargs
    ) -> Dict[str, float]:
        """
        Pre-compile diffusion functions for multiple backends.
        
        Mirrors CodeGenerator.compile_all() for consistency.
        
        Parameters
        ----------
        backends : List[str], optional
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
            backends = ['numpy', 'torch', 'jax']
        
        timings = {}
        
        for backend in backends:
            if verbose:
                print(f"Compiling diffusion for {backend}...", end=' ')
            
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
    
    def warmup(self, backend: str, **kwargs):
        """
        Warm up function compilation (especially useful for JAX JIT).
        
        Parameters
        ----------
        backend : str
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
        if backend == 'numpy':
            test_inputs = [0.0] * (self.nx + len(self.control_vars))
        elif backend == 'torch':
            import torch
            test_inputs = [torch.tensor(0.0)] * (self.nx + len(self.control_vars))
        elif backend == 'jax':
            import jax.numpy as jnp
            test_inputs = [jnp.array(0.0)] * (self.nx + len(self.control_vars))
        
        # Call once to trigger compilation
        _ = func(*test_inputs)
    
    # ========================================================================
    # Cache Management (Mirrors CodeGenerator)
    # ========================================================================
    
    def reset_cache(self, backends: Optional[List[str]] = None):
        """
        Clear cached functions for specified backends.
        
        Mirrors CodeGenerator.reset_cache().
        
        Parameters
        ----------
        backends : List[str], optional
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
            backends = ['numpy', 'torch', 'jax']
        
        for backend in backends:
            self._diffusion_funcs[backend] = None
            
            # Also clear constant noise cache for this backend
            cache_key = f"{backend}_constant"
            if cache_key in self._constant_noise_cache:
                del self._constant_noise_cache[cache_key]
    
    def is_compiled(self, backend: str) -> bool:
        """
        Check if diffusion function is compiled for backend.
        
        Mirrors CodeGenerator.is_compiled() but simpler (only one function type).
        
        Parameters
        ----------
        backend : str
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
            'dimensions': {
                'nx': self.nx,
                'nw': self.nw,
                'num_parameters': len(self.parameters)
            },
            'noise_type': char.noise_type.value,
            'characteristics': {
                'is_additive': char.is_additive,
                'is_multiplicative': char.is_multiplicative,
                'is_diagonal': char.is_diagonal,
                'is_scalar': char.is_scalar,
                'depends_on': {
                    'state': char.depends_on_state,
                    'control': char.depends_on_control,
                    'time': char.depends_on_time,
                },
                'state_dependencies': [str(s) for s in char.state_dependencies],
                'control_dependencies': [str(s) for s in char.control_dependencies],
            },
            'compiled': {
                backend: self.is_compiled(backend)
                for backend in ['numpy', 'torch', 'jax']
            },
            'constant_noise_cached': self.has_constant_noise(),
            'statistics': self.get_stats(),
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
        total_calls = self._generation_stats['generations'] + self._generation_stats['cache_hits']
        
        return {
            'generations': self._generation_stats['generations'],
            'cache_hits': self._generation_stats['cache_hits'],
            'total_calls': total_calls,
            'cache_hit_rate': (
                self._generation_stats['cache_hits'] / max(1, total_calls)
            ),
            'total_time': self._generation_stats['total_time'],
            'avg_generation_time': (
                self._generation_stats['total_time'] / 
                max(1, self._generation_stats['generations'])
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
        self._generation_stats = {
            'generations': 0,
            'cache_hits': 0,
            'total_time': 0.0
        }
    
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
            'precompute_diffusion': char.is_additive,
            'use_diagonal_solver': char.is_diagonal,
            'use_scalar_solver': char.is_scalar,
            'vectorize_easily': char.is_additive or char.is_diagonal,
            'cache_diffusion': not char.depends_on_state,  # Time-varying control only
        }
    
    # ========================================================================
    # String Representations (Mirrors CodeGenerator)
    # ========================================================================
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        compiled_backends = [
            b for b in ['numpy', 'torch', 'jax']
            if self._diffusion_funcs[b] is not None
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
            1 for b in ['numpy', 'torch', 'jax']
            if self._diffusion_funcs[b] is not None
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
    diffusion_expr: sp.Matrix,
    state_vars: List[sp.Symbol],
    control_vars: List[sp.Symbol],
    **kwargs
) -> DiffusionHandler:
    """
    Convenience function for creating diffusion handlers.
    
    Parameters
    ----------
    diffusion_expr : sp.Matrix
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