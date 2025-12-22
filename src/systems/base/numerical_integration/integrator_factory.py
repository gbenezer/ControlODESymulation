"""
Integrator Factory - Unified Interface for Creating Numerical Integrators

Provides a convenient factory class for creating the appropriate integrator
based on backend, method, and requirements. Simplifies integrator selection
and configuration.

Now includes support for Julia's DifferentialEquations.jl via DiffEqPy!

Examples
--------
>>> # Automatic selection
>>> integrator = IntegratorFactory.create(system, backend='numpy')
>>>
>>> # Specific method
>>> integrator = IntegratorFactory.create(
...     system, backend='jax', method='tsit5'
... )
>>>
>>> # Julia DiffEqPy solver
>>> integrator = IntegratorFactory.create(
...     system, backend='numpy', method='Tsit5'
... )
>>>
>>> # Quick helpers
>>> integrator = IntegratorFactory.auto(system)  # Best for system
>>> integrator = IntegratorFactory.for_optimization(system)  # Best for gradients
>>> integrator = IntegratorFactory.for_julia(system)  # Best Julia solver
"""

from typing import Optional, Dict, Any, TYPE_CHECKING
from enum import Enum

from src.systems.base.numerical_integration.integrator_base import IntegratorBase, StepMode

if TYPE_CHECKING:
    from src.systems.base.symbolic_dynamical_system import SymbolicDynamicalSystem


class IntegratorType(Enum):
    """
    Integrator type categories.

    Used for automatic selection based on use case.
    """

    PRODUCTION = "production"  # Scipy (LSODA) or DiffEqPy (AutoTsit5)
    OPTIMIZATION = "optimization"  # Diffrax (tsit5)
    NEURAL_ODE = "neural_ode"  # TorchDiffEq (dopri5, adjoint)
    JULIA = "julia"  # DiffEqPy (Tsit5 or specialized)
    SIMPLE = "simple"  # RK4 fixed-step
    EDUCATIONAL = "educational"  # Euler fixed-step


class IntegratorFactory:
    """
    Factory for creating numerical integrators.

    Provides convenient methods for creating integrators based on:
    - Backend (numpy, torch, jax)
    - Method (RK45, dopri5, tsit5, Tsit5, etc.)
    - Use case (production, optimization, neural ODE, julia)

    Supports:
    - Scipy (numpy): LSODA, RK45, BDF, Radau, etc.
    - DiffEqPy (numpy): Tsit5, Vern9, Rosenbrock23, etc. (Julia solvers)
    - TorchDiffEq (torch): dopri5, dopri8, etc.
    - Diffrax (jax): tsit5, dopri5, etc.
    - Manual (any): euler, midpoint, rk4

    Examples
    --------
    >>> # Create integrator by backend and method
    >>> integrator = IntegratorFactory.create(
    ...     system,
    ...     backend='numpy',
    ...     method='LSODA'
    ... )
    >>>
    >>> # Julia solver
    >>> integrator = IntegratorFactory.create(
    ...     system,
    ...     backend='numpy',
    ...     method='Tsit5'  # Capital T = Julia
    ... )
    >>>
    >>> # Automatic selection
    >>> integrator = IntegratorFactory.auto(system)
    >>>
    >>> # Use case-specific
    >>> integrator = IntegratorFactory.for_optimization(system)
    >>> integrator = IntegratorFactory.for_production(system)
    >>> integrator = IntegratorFactory.for_julia(system, algorithm='Vern9')
    """

    # Default methods for each backend
    _BACKEND_DEFAULTS = {
        "numpy": "LSODA",
        "torch": "dopri5",
        "jax": "tsit5",
    }

    # Method to backend/integrator mapping
    _METHOD_TO_BACKEND = {
        # Scipy methods (numpy only)
        "RK45": "numpy",
        "RK23": "numpy",
        "DOP853": "numpy",
        "Radau": "numpy",
        "BDF": "numpy",
        "LSODA": "numpy",
        
        # DiffEqPy (Julia) methods (numpy only)
        # Convention: Capital first letter = Julia solver
        "Tsit5": "numpy",
        "Vern6": "numpy",
        "Vern7": "numpy",
        "Vern8": "numpy",
        "Vern9": "numpy",
        "DP5": "numpy",
        "DP8": "numpy",
        "Rosenbrock23": "numpy",
        "Rosenbrock32": "numpy",
        "Rodas4": "numpy",
        "Rodas4P": "numpy",
        "Rodas5": "numpy",
        "TRBDF2": "numpy",
        "KenCarp3": "numpy",
        "KenCarp4": "numpy",
        "KenCarp5": "numpy",
        "RadauIIA5": "numpy",
        "ROCK2": "numpy",
        "ROCK4": "numpy",
        "VelocityVerlet": "numpy",
        "SymplecticEuler": "numpy",
        # Special Julia auto-switching (contains parentheses)
        "AutoTsit5(Rosenbrock23())": "numpy",
        "AutoVern7(Rodas5())": "numpy",
        
        # TorchDiffEq-only methods (torch only)
        "adaptive_heun": "torch",
        "fehlberg2": "torch",
        "explicit_adams": "torch",
        "implicit_adams": "torch",
        "fixed_adams": "torch",
        "scipy_solver": "torch",
        
        # Shared adaptive methods (available in BOTH torch and jax)
        "dopri5": ["torch", "jax"],
        "dopri8": ["torch", "jax"],
        "bosh3": ["torch", "jax"],
        
        # Diffrax-only explicit methods (jax only)
        "tsit5": "jax",  # lowercase = Diffrax
        "heun": "jax",
        "ralston": "jax",
        "reversible_heun": "jax",
        
        # Fixed-step methods (available in all backends via manual implementation)
        "euler": "any",
        "midpoint": "any",
        "rk4": "any",
    }

    @classmethod
    def create(
        cls,
        system: "SymbolicDynamicalSystem",
        backend: str = "numpy",
        method: Optional[str] = None,
        dt: Optional[float] = None,
        step_mode: StepMode = StepMode.ADAPTIVE,
        **options,
    ) -> IntegratorBase:
        """
        Create an integrator with specified backend and method.

        Parameters
        ----------
        system : SymbolicDynamicalSystem
            System to integrate
        backend : str, optional
            Backend: 'numpy', 'torch', 'jax'. Default: 'numpy'
        method : Optional[str]
            Solver method. If None, uses backend default.
            - numpy: 'LSODA' (scipy, auto-stiffness)
            - numpy with capital: 'Tsit5' (Julia via DiffEqPy)
            - torch: 'dopri5' (general adaptive)
            - jax: 'tsit5' (general adaptive)
        dt : Optional[float]
            Time step (required for FIXED mode)
        step_mode : StepMode
            FIXED or ADAPTIVE stepping
        **options
            Additional integrator options (rtol, atol, etc.)

        Returns
        -------
        IntegratorBase
            Configured integrator

        Raises
        ------
        ValueError
            If backend/method combination is invalid
        ImportError
            If required package not installed

        Examples
        --------
        >>> # Use defaults (scipy)
        >>> integrator = IntegratorFactory.create(system)
        >>>
        >>> # Julia solver
        >>> integrator = IntegratorFactory.create(
        ...     system, backend='numpy', method='Tsit5'
        ... )
        >>>
        >>> # Specify JAX method
        >>> integrator = IntegratorFactory.create(
        ...     system, backend='jax', method='dopri5'
        ... )
        >>>
        >>> # Fixed-step
        >>> integrator = IntegratorFactory.create(
        ...     system,
        ...     backend='numpy',
        ...     method='rk4',
        ...     dt=0.01,
        ...     step_mode=StepMode.FIXED
        ... )
        """
        # Use default method if not specified
        if method is None:
            method = cls._BACKEND_DEFAULTS.get(backend, "LSODA")

        # Validate backend
        valid_backends = ["numpy", "torch", "jax"]
        if backend not in valid_backends:
            raise ValueError(f"Invalid backend '{backend}'. Choose from: {valid_backends}")

        # Check if method requires specific backend
        if method in cls._METHOD_TO_BACKEND:
            required_backend = cls._METHOD_TO_BACKEND[method]

            # Handle methods available in multiple backends
            if isinstance(required_backend, list):
                if backend not in required_backend:
                    raise ValueError(
                        f"Method '{method}' requires backend in {required_backend}, "
                        f"got backend='{backend}'"
                    )
            elif required_backend != "any" and required_backend != backend:
                raise ValueError(
                    f"Method '{method}' requires backend='{required_backend}', "
                    f"got backend='{backend}'"
                )

        # Create appropriate integrator
        if backend == "numpy":
            return cls._create_numpy_integrator(system, method, dt, step_mode, **options)
        elif backend == "torch":
            return cls._create_torch_integrator(system, method, dt, step_mode, **options)
        elif backend == "jax":
            return cls._create_jax_integrator(system, method, dt, step_mode, **options)

    @classmethod
    def _create_numpy_integrator(
        cls, system, method: str, dt: Optional[float], step_mode: StepMode, **options
    ):
        """
        Create NumPy-based integrator.
        
        Routes to appropriate integrator based on method type:
        - Manual fixed-step: euler, midpoint, rk4
        - Julia DiffEqPy: Tsit5, Vern9, Rosenbrock23, etc.
        - Scipy: LSODA, RK45, BDF, Radau, etc.
        """
        
        # 1. Manual fixed-step implementations
        if cls._is_fixed_step_method(method):
            if dt is None:
                raise ValueError(f"Fixed-step method '{method}' requires dt")

            from src.systems.base.numerical_integration.fixed_step_integrators import (
                ExplicitEulerIntegrator,
                MidpointIntegrator,
                RK4Integrator,
            )

            integrator_map = {
                "euler": ExplicitEulerIntegrator,
                "midpoint": MidpointIntegrator,
                "rk4": RK4Integrator,
            }

            integrator_class = integrator_map[method]
            return integrator_class(system, dt=dt, backend="numpy", **options)
        
        # 2. Julia DiffEqPy methods
        elif cls._is_julia_method(method):
            try:
                from src.systems.base.numerical_integration.diffeqpy_integrator import (
                    DiffEqPyIntegrator
                )
                
                return DiffEqPyIntegrator(
                    system,
                    dt=dt,
                    step_mode=step_mode,
                    backend='numpy',
                    algorithm=method,
                    **options
                )
            except ImportError:
                raise ImportError(
                    f"Julia method '{method}' requires diffeqpy. "
                    f"Install Julia + DifferentialEquations.jl + diffeqpy, "
                    f"or use a scipy method instead."
                )
        
        # 3. Scipy methods (default fallback)
        else:
            from src.systems.base.numerical_integration.scipy_integrator import ScipyIntegrator

            return ScipyIntegrator(system, dt=dt, method=method, backend="numpy", **options)

    @classmethod
    def _is_julia_method(cls, method: str) -> bool:
        """
        Determine if a method name refers to a Julia/DiffEqPy solver.
        
        Uses list_algorithms() from diffeqpy_integrator as single source of truth.
        
        Parameters
        ----------
        method : str
            Method name
            
        Returns
        -------
        bool
            True if Julia method, False otherwise
        """
        if not method:
            return False
        
        # Auto-switching algorithms contain parentheses
        if '(' in method:
            return True
        
        # Check against known Julia algorithms
        try:
            from src.systems.base.numerical_integration.diffeqpy_integrator import list_algorithms
            
            all_algorithms = list_algorithms()
            known_julia_algorithms = set()
            for category, algos in all_algorithms.items():
                known_julia_algorithms.update(algos)
            
            return method in known_julia_algorithms
        except ImportError:
            # diffeqpy not installed - fall back to heuristic
            # Julia algorithms start with capital letter but aren't all uppercase
            if method[0].isupper() and not method.isupper():
                return True
            return False
    
    @classmethod
    def _is_fixed_step_method(cls, method: str) -> bool:
        """
        Check if method is a manual fixed-step implementation.
        
        These methods are backend-agnostic and available everywhere.
        
        Parameters
        ----------
        method : str
            Method name
            
        Returns
        -------
        bool
            True if manual fixed-step method, False otherwise
        """
        return method in ['euler', 'midpoint', 'rk4']
    
    @classmethod
    def _is_scipy_method(cls, method: str) -> bool:
        """
        Check if method is a scipy solver.
        
        Scipy methods are typically all-caps or specific well-known names.
        
        Parameters
        ----------
        method : str
            Method name
            
        Returns
        -------
        bool
            True if scipy method, False otherwise
        """
        scipy_methods = {'LSODA', 'RK45', 'RK23', 'DOP853', 'Radau', 'BDF'}
        return method in scipy_methods

    @classmethod
    def _create_torch_integrator(cls, system, method: str, dt, step_mode, **options):
        """Create PyTorch-based integrator using TorchDiffEq."""
        from src.systems.base.numerical_integration.torchdiffeq_integrator import TorchDiffEqIntegrator
        
        # Always use TorchDiffEq for torch backend
        return TorchDiffEqIntegrator(
            system,
            dt=dt,
            step_mode=step_mode,
            backend='torch',
            method=method,
            **options
        )

    @classmethod
    def _create_jax_integrator(cls, system, method: str, dt, step_mode, **options):
        """Create JAX-based integrator - always use Diffrax."""
        from src.systems.base.numerical_integration.diffrax_integrator import DiffraxIntegrator
        
        # Let Diffrax handle ALL methods, including euler/midpoint
        return DiffraxIntegrator(
            system,
            dt=dt,
            step_mode=step_mode,
            backend='jax',
            solver=method,
            **options
        )

    # ========================================================================
    # Convenience Methods - Use Case-Specific Creation
    # ========================================================================

    @classmethod
    def auto(
        cls, system: "SymbolicDynamicalSystem", prefer_backend: Optional[str] = None, **options
    ) -> IntegratorBase:
        """
        Automatically select best integrator for system.

        Selection logic:
        1. If JAX available and no backend preference → Diffrax (fast + accurate)
        2. If PyTorch available and no backend preference → TorchDiffEq
        3. Otherwise → Scipy (always available)

        Parameters
        ----------
        system : SymbolicDynamicalSystem
            System to integrate
        prefer_backend : Optional[str]
            Preferred backend if available
        **options
            Additional options

        Returns
        -------
        IntegratorBase
            Best available integrator

        Examples
        --------
        >>> integrator = IntegratorFactory.auto(system)
        >>> integrator = IntegratorFactory.auto(system, prefer_backend='jax')
        """
        # Check backend availability
        backends_available = []

        try:
            import jax
            backends_available.append("jax")
        except ImportError:
            pass

        try:
            import torch
            backends_available.append("torch")
        except ImportError:
            pass

        backends_available.append("numpy")  # Always available

        # Select backend
        if prefer_backend and prefer_backend in backends_available:
            backend = prefer_backend
        elif "jax" in backends_available:
            backend = "jax"  # Prefer JAX (best for optimization)
        elif "torch" in backends_available:
            backend = "torch"
        else:
            backend = "numpy"

        return cls.create(system, backend=backend, **options)

    @classmethod
    def for_production(
        cls, 
        system: "SymbolicDynamicalSystem",
        use_julia: bool = False,
        **options
    ) -> IntegratorBase:
        """
        Create integrator for production use.

        Uses scipy.LSODA (default) or Julia's AutoTsit5 (if use_julia=True)
        with automatic stiffness detection. Most reliable choices.

        Parameters
        ----------
        system : SymbolicDynamicalSystem
            System to integrate
        use_julia : bool
            If True, use Julia's AutoTsit5. Default: False (scipy)
        **options
            Additional options (rtol, atol, etc.)

        Returns
        -------
        IntegratorBase
            Production-grade integrator

        Examples
        --------
        >>> # Scipy (default)
        >>> integrator = IntegratorFactory.for_production(
        ...     system, rtol=1e-9, atol=1e-11
        ... )
        >>>
        >>> # Julia (if installed)
        >>> integrator = IntegratorFactory.for_production(
        ...     system, use_julia=True
        ... )
        """
        if use_julia:
            try:
                from src.systems.base.numerical_integration.diffeqpy_integrator import (
                    DiffEqPyIntegrator
                )
                
                # Set conservative defaults
                default_options = {
                    "rtol": 1e-8,
                    "atol": 1e-10,
                }
                default_options.update(options)
                
                return DiffEqPyIntegrator(
                    system,
                    backend='numpy',
                    algorithm='AutoTsit5(Rosenbrock23())',  # Auto-stiffness
                    **default_options
                )
            except ImportError:
                raise ImportError(
                    "use_julia=True requires diffeqpy. "
                    "Install Julia + DifferentialEquations.jl + diffeqpy, "
                    "or use use_julia=False for scipy."
                )
        else:
            from src.systems.base.numerical_integration.scipy_integrator import ScipyIntegrator

            # Set conservative defaults
            default_options = {
                "rtol": 1e-8,
                "atol": 1e-10,
            }
            default_options.update(options)

            return ScipyIntegrator(
                system, method="LSODA", backend="numpy", **default_options
            )

    @classmethod
    def for_julia(
        cls,
        system: "SymbolicDynamicalSystem",
        algorithm: str = 'Tsit5',
        **options
    ) -> IntegratorBase:
        """
        Create Julia DiffEqPy integrator.
        
        Provides access to Julia's extensive ODE solver ecosystem.
        Best for difficult ODEs, high accuracy, or specialized solvers.

        Parameters
        ----------
        system : SymbolicDynamicalSystem
            System to integrate
        algorithm : str
            Julia algorithm name. Default: 'Tsit5'
            Options: Tsit5, Vern9, Rosenbrock23, AutoTsit5(Rosenbrock23()), etc.
        **options
            Additional options (rtol, atol, etc.)

        Returns
        -------
        IntegratorBase
            DiffEqPy integrator

        Raises
        ------
        ImportError
            If Julia/diffeqpy not installed

        Examples
        --------
        >>> # Default (Tsit5)
        >>> integrator = IntegratorFactory.for_julia(system)
        >>>
        >>> # High accuracy
        >>> integrator = IntegratorFactory.for_julia(
        ...     system,
        ...     algorithm='Vern9',
        ...     rtol=1e-12
        ... )
        >>>
        >>> # Auto-switching
        >>> integrator = IntegratorFactory.for_julia(
        ...     system,
        ...     algorithm='AutoTsit5(Rosenbrock23())'
        ... )
        >>>
        >>> # Stiff system
        >>> integrator = IntegratorFactory.for_julia(
        ...     system,
        ...     algorithm='Rodas5'
        ... )
        """
        try:
            from src.systems.base.numerical_integration.diffeqpy_integrator import (
                DiffEqPyIntegrator
            )
        except ImportError:
            raise ImportError(
                "Julia integration requires diffeqpy.\n\n"
                "Installation:\n"
                "1. Install Julia from https://julialang.org/downloads/\n"
                "2. Install DifferentialEquations.jl:\n"
                "   julia> using Pkg\n"
                "   julia> Pkg.add('DifferentialEquations')\n"
                "3. Install Python package:\n"
                "   pip install diffeqpy"
            )
        
        return DiffEqPyIntegrator(
            system,
            backend='numpy',
            algorithm=algorithm,
            **options
        )

    @classmethod
    def for_optimization(
        cls, system: "SymbolicDynamicalSystem", prefer_backend: str = "jax", **options
    ) -> IntegratorBase:
        """
        Create integrator for optimization/parameter estimation.

        Prioritizes gradient computation and JIT compilation.
        """
        # Try preferred backend first
        try:
            if prefer_backend == "jax":
                import jax
                from src.systems.base.numerical_integration.diffrax_integrator import (
                    DiffraxIntegrator,
                )

                return DiffraxIntegrator(
                    system,
                    dt=options.pop("dt", None),  # dt optional for adaptive
                    step_mode=options.pop("step_mode", StepMode.ADAPTIVE),
                    backend="jax",
                    solver="tsit5",
                    adjoint="recursive_checkpoint",
                    **options,
                )
            elif prefer_backend == "torch":
                import torch
                from src.systems.base.numerical_integration.torchdiffeq_integrator import (
                    TorchDiffEqIntegrator,
                )

                return TorchDiffEqIntegrator(
                    system,
                    dt=options.pop("dt", None),  # dt optional for adaptive
                    step_mode=options.pop("step_mode", StepMode.ADAPTIVE),
                    backend="torch",
                    method="dopri5",
                    adjoint=options.pop("adjoint", False),  # Default False
                    **options,
                )
        except ImportError:
            pass

        # Fallback: try JAX, then torch, then numpy
        try:
            import jax
            from src.systems.base.numerical_integration.diffrax_integrator import DiffraxIntegrator

            return DiffraxIntegrator(
                system,
                dt=options.pop("dt", None),
                step_mode=StepMode.ADAPTIVE,
                backend="jax",
                solver="tsit5",
                **options,
            )
        except ImportError:
            pass

        try:
            import torch
            from src.systems.base.numerical_integration.torchdiffeq_integrator import (
                TorchDiffEqIntegrator,
            )

            return TorchDiffEqIntegrator(
                system,
                dt=options.pop("dt", None),
                step_mode=StepMode.ADAPTIVE,
                backend="torch",
                method="dopri5",
                **options,
            )
        except ImportError:
            pass

        # Last resort: scipy (no gradient support)
        from src.systems.base.numerical_integration.scipy_integrator import ScipyIntegrator

        return ScipyIntegrator(system, method="RK45", backend="numpy", **options)

    @classmethod
    def for_neural_ode(cls, neural_system, **options) -> IntegratorBase:
        """
        Create integrator for Neural ODE training.

        Uses PyTorch with adjoint method for memory efficiency.

        Parameters
        ----------
        neural_system : nn.Module
            Neural network defining ODE dynamics
        **options
            Additional options

        Returns
        -------
        IntegratorBase
            TorchDiffEq integrator with adjoint method

        Raises
        ------
        ImportError
            If PyTorch not installed

        Examples
        --------
        >>> class NeuralODE(nn.Module):
        ...     def forward(self, t, x):
        ...         return self.net(x)
        >>>
        >>> neural_ode = NeuralODE()
        >>> integrator = IntegratorFactory.for_neural_ode(neural_ode)
        """
        try:
            from src.systems.base.numerical_integration.torchdiffeq_integrator import (
                TorchDiffEqIntegrator,
            )
        except ImportError:
            raise ImportError(
                "PyTorch is required for Neural ODE integration. "
                "Install with: pip install torch torchdiffeq"
            )

        return TorchDiffEqIntegrator(
            neural_system,
            backend="torch",
            method="dopri5",
            adjoint=True,  # Memory-efficient for neural networks
            **options,
        )

    @classmethod
    def for_simple_simulation(
        cls, system: "SymbolicDynamicalSystem", dt: float = 0.01, backend: str = "numpy", **options
    ) -> IntegratorBase:
        """
        Create simple fixed-step RK4 integrator.

        Good for prototyping and educational purposes.

        Parameters
        ----------
        system : SymbolicDynamicalSystem
            System to integrate
        dt : float
            Time step. Default: 0.01
        backend : str
            Backend to use. Default: 'numpy'
        **options
            Additional options

        Returns
        -------
        IntegratorBase
            RK4 fixed-step integrator

        Examples
        --------
        >>> integrator = IntegratorFactory.for_simple_simulation(
        ...     system, dt=0.01
        ... )
        """
        from src.systems.base.numerical_integration.fixed_step_integrators import RK4Integrator

        return RK4Integrator(system, dt=dt, backend=backend, **options)

    @classmethod
    def for_real_time(
        cls, system: "SymbolicDynamicalSystem", dt: float, backend: str = "numpy", **options
    ) -> IntegratorBase:
        """
        Create integrator for real-time systems.

        Uses fixed-step RK4 for predictable timing.

        Parameters
        ----------
        system : SymbolicDynamicalSystem
            System to integrate
        dt : float
            Fixed time step (must match real-time clock)
        backend : str
            Backend to use
        **options
            Additional options

        Returns
        -------
        IntegratorBase
            Fixed-step integrator

        Examples
        --------
        >>> # Real-time control at 100 Hz
        >>> integrator = IntegratorFactory.for_real_time(
        ...     system, dt=0.01  # 10ms = 100 Hz
        ... )
        """
        from src.systems.base.numerical_integration.fixed_step_integrators import RK4Integrator

        return RK4Integrator(system, dt=dt, backend=backend, **options)

    # ========================================================================
    # Utility Methods
    # ========================================================================

    @staticmethod
    def list_methods(backend: Optional[str] = None) -> Dict[str, list]:
        """
        List available methods for each backend.
        
        Returns
        -------
        Dict[str, list]
            Methods available for each backend
            
        Examples
        --------
        >>> methods = IntegratorFactory.list_methods()
        >>> print(methods['numpy'][:5])
        ['LSODA', 'RK45', 'Tsit5', 'Vern9', ...]
        >>>
        >>> # Just one backend
        >>> numpy_methods = IntegratorFactory.list_methods('numpy')
        """
        all_methods = {
            "numpy": [
                # Scipy methods
                "LSODA",
                "RK45",
                "RK23",
                "DOP853",
                "Radau",
                "BDF",
                # Julia methods (if diffeqpy installed)
                "Tsit5",
                "Vern6",
                "Vern7",
                "Vern8",
                "Vern9",
                "DP5",
                "DP8",
                "Rosenbrock23",
                "Rodas5",
                "AutoTsit5(Rosenbrock23())",
                "ROCK4",
                "VelocityVerlet",
                # Manual implementations
                "euler",
                "midpoint",
                "rk4",
            ],
            "torch": [
                "dopri5",
                "dopri8",
                "bosh3",
                "adaptive_heun",
                "fehlberg2",
                "euler",
                "midpoint",
                "rk4",
                "explicit_adams",
                "implicit_adams",
                "fixed_adams",
                "scipy_solver",
            ],
            "jax": [
                # Only solvers available in Diffrax 0.7.0
                "tsit5",
                "dopri5",
                "dopri8",
                "bosh3",
                "euler",
                "heun",
                "midpoint",
                "ralston",
                "reversible_heun",
                # NOTE: 'rk4' NOT in Diffrax 0.7.0
                # NOTE: Implicit/IMEX require Diffrax 0.8.0+
            ],
        }

        if backend:
            return {backend: all_methods.get(backend, [])}
        return all_methods

    @staticmethod
    def recommend(
        use_case: str, has_jax: bool = False, has_torch: bool = False, has_gpu: bool = False
    ) -> Dict[str, Any]:
        """
        Get integrator recommendation based on use case.

        Parameters
        ----------
        use_case : str
            One of: 'production', 'optimization', 'neural_ode',
            'prototype', 'educational', 'real_time', 'julia'
        has_jax : bool
            Whether JAX is available
        has_torch : bool
            Whether PyTorch is available
        has_gpu : bool
            Whether GPU is available

        Returns
        -------
        Dict[str, Any]
            Recommendation with 'backend', 'method', 'step_mode'

        Examples
        --------
        >>> rec = IntegratorFactory.recommend('production')
        >>> print(rec)
        {'backend': 'numpy', 'method': 'LSODA', 'step_mode': 'ADAPTIVE'}
        """
        recommendations = {
            "production": {
                "backend": "numpy",
                "method": "LSODA",
                "step_mode": StepMode.ADAPTIVE,
                "reason": "Most reliable, auto-stiffness detection",
            },
            "optimization": {
                "backend": "jax" if has_jax else "torch" if has_torch else "numpy",
                "method": "tsit5" if has_jax else "dopri5" if has_torch else "RK45",
                "step_mode": StepMode.ADAPTIVE,
                "reason": "Gradient support, JIT compilation",
            },
            "neural_ode": {
                "backend": "torch",
                "method": "dopri5",
                "step_mode": StepMode.ADAPTIVE,
                "adjoint": True,
                "reason": "Memory-efficient backprop through ODE",
            },
            "julia": {
                "backend": "numpy",
                "method": "Tsit5",
                "step_mode": StepMode.ADAPTIVE,
                "reason": "Access to Julia's powerful solver ecosystem",
            },
            "simple": {
                "backend": "numpy",
                "method": "rk4",
                "step_mode": StepMode.FIXED,
                "dt": 0.01,
                "reason": "Simple, fast, easy to debug",
            },
            "prototype": {
                "backend": "numpy",
                "method": "rk4",
                "step_mode": StepMode.FIXED,
                "dt": 0.01,
                "reason": "Simple, fast, easy to debug",
            },
            "educational": {
                "backend": "numpy",
                "method": "euler",
                "step_mode": StepMode.FIXED,
                "dt": 0.001,
                "reason": "Easiest to understand",
            },
            "real_time": {
                "backend": "numpy",
                "method": "rk4",
                "step_mode": StepMode.FIXED,
                "reason": "Predictable timing",
            },
        }

        if use_case not in recommendations:
            raise ValueError(
                f"Unknown use case '{use_case}'. " f"Choose from: {list(recommendations.keys())}"
            )

        rec = recommendations[use_case].copy()

        # Adjust for GPU availability
        if has_gpu and use_case == "optimization":
            if has_torch:
                rec["backend"] = "torch"
                rec["method"] = "dopri5"
            elif has_jax:
                rec["backend"] = "jax"
                rec["method"] = "tsit5"

        return rec

    @staticmethod
    def get_info(backend: str, method: str) -> Dict[str, Any]:
        """
        Get information about a specific integrator configuration.
        
        Delegates to integrator-specific info functions where available.

        Parameters
        ----------
        backend : str
            Backend name
        method : str
            Method name

        Returns
        -------
        Dict[str, Any]
            Information about the integrator

        Examples
        --------
        >>> info = IntegratorFactory.get_info('jax', 'tsit5')
        >>> print(info['description'])
        'Excellent general purpose, JAX-optimized'
        >>>
        >>> info = IntegratorFactory.get_info('numpy', 'Tsit5')
        >>> print(info['description'])
        'Excellent general-purpose solver with good efficiency'
        >>>
        >>> info = IntegratorFactory.get_info('numpy', 'Vern7')
        >>> print(info['description'])  # Works even if not in hardcoded list!
        """
        # For Julia methods, delegate to get_algorithm_info()
        if backend == 'numpy' and IntegratorFactory._is_julia_method(method):
            try:
                from src.systems.base.numerical_integration.diffeqpy_integrator import (
                    get_algorithm_info
                )
                
                julia_info = get_algorithm_info(method)
                # Add backend/library metadata
                julia_info['backend'] = 'numpy'
                julia_info['library'] = 'Julia DifferentialEquations.jl'
                return julia_info
            except ImportError:
                # diffeqpy not installed - return generic info
                return {
                    'name': f'Julia: {method}',
                    'backend': 'numpy',
                    'library': 'Julia DifferentialEquations.jl (not installed)',
                    'description': f'Julia algorithm {method} (diffeqpy not available for details)'
                }
        
        # Hardcoded info for scipy, diffrax, torchdiffeq
        method_info = {
            # Scipy
            "LSODA": {
                "name": "LSODA",
                "order": "Variable (1-12)",
                "type": "Adaptive",
                "library": "scipy",
                "description": "Auto-detects stiffness, switches Adams↔BDF",
                "best_for": "Production, unknown stiffness",
                "function_evals_per_step": "1-4",
            },
            "RK45": {
                "name": "Dormand-Prince 5(4)",
                "order": 5,
                "type": "Adaptive",
                "library": "scipy",
                "description": "General purpose, robust",
                "best_for": "Non-stiff systems",
                "function_evals_per_step": "6",
            },
            "DOP853": {
                "name": "Dormand-Prince 8(5,3)",
                "order": 8,
                "type": "Adaptive",
                "library": "scipy",
                "description": "Very high accuracy",
                "best_for": "Precision requirements",
                "function_evals_per_step": "12",
            },
            "BDF": {
                "name": "Backward Differentiation Formula",
                "order": "Variable (1-5)",
                "type": "Implicit",
                "library": "scipy",
                "description": "For very stiff systems",
                "best_for": "Chemistry, circuits",
                "function_evals_per_step": "1 + Jacobian",
            },
            "Radau": {
                "name": "Radau IIA",
                "order": 5,
                "type": "Implicit",
                "library": "scipy",
                "description": "Implicit Runge-Kutta for stiff systems",
                "best_for": "Stiff DAEs",
                "function_evals_per_step": "Variable + Jacobian",
            },
            
            # Diffrax (JAX)
            "tsit5": {
                "name": "Tsitouras 5(4)",
                "order": 5,
                "type": "Adaptive",
                "library": "Diffrax",
                "description": "Excellent general purpose, JAX-optimized",
                "best_for": "JAX optimization workflows",
                "function_evals_per_step": "7",
            },
            "dopri5": {
                "name": "Dormand-Prince 5(4)",
                "order": 5,
                "type": "Adaptive",
                "library": "Diffrax/TorchDiffEq",
                "description": "Classic robust solver",
                "best_for": "General purpose",
                "function_evals_per_step": "6",
            },
            "dopri8": {
                "name": "Dormand-Prince 8(7)",
                "order": 8,
                "type": "Adaptive",
                "library": "Diffrax/TorchDiffEq",
                "description": "High accuracy",
                "best_for": "Precision requirements",
                "function_evals_per_step": "12",
            },
            "bosh3": {
                "name": "Bogacki-Shampine 3(2)",
                "order": 3,
                "type": "Adaptive",
                "library": "Diffrax/TorchDiffEq",
                "description": "Lower order adaptive",
                "best_for": "Fast simulations",
                "function_evals_per_step": "4",
            },
            
            # Fixed-step
            "euler": {
                "name": "Explicit Euler",
                "order": 1,
                "type": "Fixed-step",
                "library": "Manual implementation",
                "description": "Simplest method, educational",
                "best_for": "Learning, prototyping",
                "function_evals_per_step": "1",
            },
            "midpoint": {
                "name": "Explicit Midpoint (RK2)",
                "order": 2,
                "type": "Fixed-step",
                "library": "Manual implementation",
                "description": "Second-order accuracy",
                "best_for": "Simple simulations",
                "function_evals_per_step": "2",
            },
            "rk4": {
                "name": "Classic Runge-Kutta 4",
                "order": 4,
                "type": "Fixed-step",
                "library": "Manual implementation",
                "description": "Excellent accuracy/cost trade-off",
                "best_for": "Fixed-step simulations",
                "function_evals_per_step": "4",
            },
        }

        return method_info.get(
            method, 
            {
                "name": method, 
                "description": "No information available",
                "backend": backend
            }
        )


# ============================================================================
# Convenience Functions
# ============================================================================


def create_integrator(
    system: "SymbolicDynamicalSystem",
    backend: str = "numpy",
    method: Optional[str] = None,
    **options,
) -> IntegratorBase:
    """
    Convenience function for creating integrators.

    Alias for IntegratorFactory.create().

    Examples
    --------
    >>> integrator = create_integrator(system)
    >>> integrator = create_integrator(system, backend='jax', method='tsit5')
    >>> integrator = create_integrator(system, backend='numpy', method='Tsit5')  # Julia
    """
    return IntegratorFactory.create(system, backend, method, **options)


def auto_integrator(system: "SymbolicDynamicalSystem", **options) -> IntegratorBase:
    """
    Automatically select best available integrator.

    Alias for IntegratorFactory.auto().

    Examples
    --------
    >>> integrator = auto_integrator(system, rtol=1e-8)
    """
    return IntegratorFactory.auto(system, **options)