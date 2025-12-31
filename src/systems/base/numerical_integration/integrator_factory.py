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
Integrator Factory - Unified Interface for Creating Numerical Integrators

Provides a convenient factory class for creating the appropriate integrator
based on backend, method, and requirements. Simplifies integrator selection
and configuration.

Now includes support for Julia's DifferentialEquations.jl via DiffEqPy!
Supports both controlled and autonomous systems (nu=0).

Design Note
-----------
This module uses semantic types from src.types.core for time step parameters,
following the project design principle: "Use semantic types for clarity".
This enables better type safety, IDE autocomplete, and consistency across
the codebase.

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
>>> # Autonomous system
>>> integrator = IntegratorFactory.create(autonomous_system, backend='jax')
>>> result = integrator.integrate(
...     x0=jnp.array([1.0, 0.0]),
...     u_func=lambda t, x: None,  # No control
...     t_span=(0.0, 10.0)
... )
>>>
>>> # Quick helpers
>>> integrator = IntegratorFactory.auto(system)  # Best for system
>>> integrator = IntegratorFactory.for_optimization(system)  # Best for gradients
>>> integrator = IntegratorFactory.for_julia(system)  # Best Julia solver
"""

from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, Optional

from src.systems.base.numerical_integration.integrator_base import IntegratorBase, StepMode

# Import semantic types from centralized type system
from src.types.core import ScalarLike
from src.types.backends import Backend, IntegrationMethod

if TYPE_CHECKING:
    from src.systems.base.core.continuous_system_base import ContinuousSystemBase


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

    All integrators support autonomous systems (nu=0) by passing u=None.

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
        system: "ContinuousSystemBase",
        backend: Backend = "numpy",
        method: Optional[IntegrationMethod] = None,
        dt: Optional[ScalarLike] = None,
        step_mode: StepMode = StepMode.ADAPTIVE,
        **options,
    ) -> IntegratorBase:
        """
        Create an integrator with specified backend and method.

        Parameters
        ----------
        system : SymbolicDynamicalSystem
            System to integrate (controlled or autonomous)
        backend : str, optional
            Backend: 'numpy', 'torch', 'jax'. Default: 'numpy'
        method : Optional[str]
            Solver method. If None, uses backend default.
            - numpy: 'LSODA' (scipy, auto-stiffness)
            - numpy with capital: 'Tsit5' (Julia via DiffEqPy)
            - torch: 'dopri5' (general adaptive)
            - jax: 'tsit5' (general adaptive)
        dt : Optional[ScalarLike]
            Time step (required for FIXED mode)
        step_mode : StepMode
            FIXED or ADAPTIVE stepping
        **options
            Additional integrator options (rtol, atol, etc.)
            Note: For JAX backend, 'solver' in options will be treated as 'method'

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
        >>> # Specify JAX method (both calling styles work)
        >>> integrator = IntegratorFactory.create(
        ...     system, backend='jax', method='dopri5'
        ... )
        >>> # OR
        >>> integrator = IntegratorFactory.create(
        ...     system, backend='jax', solver='dopri5'
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
        >>>
        >>> # Autonomous system
        >>> integrator = IntegratorFactory.create(autonomous_system)
        >>> result = integrator.integrate(
        ...     x0=np.array([1.0, 0.0]),
        ...     u_func=lambda t, x: None,
        ...     t_span=(0.0, 10.0)
        ... )
        """
        # Handle 'solver' parameter for JAX backend (backward compatibility)
        # If user passes 'solver' instead of 'method', use it
        if backend == "jax" and "solver" in options and method is None:
            method = options.pop("solver")  # Remove from options to avoid duplicate

        # Use default method if not specified
        if method is None:
            method = cls._BACKEND_DEFAULTS.get(backend, "LSODA")

        # Validate backend
        valid_backends = ["numpy", "torch", "jax"]
        if backend not in valid_backends:
            raise ValueError(f"Invalid backend '{backend}'. Choose from: {valid_backends}")

        # Validate method-backend compatibility
        if method in cls._METHOD_TO_BACKEND:
            allowed = cls._METHOD_TO_BACKEND[method]

            if allowed != "any":
                # Method has specific backend requirements
                if isinstance(allowed, list):
                    # Multiple backends allowed
                    if backend not in allowed:
                        raise ValueError(
                            f"Method '{method}' requires backend in {allowed}, " f"got '{backend}'"
                        )
                elif allowed != backend:
                    # Single backend required
                    raise ValueError(
                        f"Method '{method}' requires backend '{allowed}', " f"got '{backend}'"
                    )

        # Check if fixed-step method requires dt
        if cls._is_fixed_step_method(method) or step_mode == StepMode.FIXED:
            if dt is None:
                raise ValueError(
                    f"Fixed-step method '{method}' or FIXED step mode " f"requires dt parameter"
                )

        # Create appropriate integrator based on backend
        if backend == "numpy":
            return cls._create_numpy_integrator(system, method, dt, step_mode, **options)
        elif backend == "torch":
            return cls._create_torch_integrator(system, method, dt, step_mode, **options)
        elif backend == "jax":
            return cls._create_jax_integrator(system, method, dt, step_mode, **options)
        else:
            raise ValueError(f"Unknown backend: {backend}")

    @classmethod
    def _create_numpy_integrator(
        cls,
        system: "ContinuousSystemBase",
        method: IntegrationMethod,
        dt: Optional[ScalarLike],
        step_mode: StepMode,
        **options,
    ):
        """
        Create NumPy-based integrator.

        Routes to:
        - DiffEqPy for Julia methods (Capital first letter)
        - Scipy for standard methods (LSODA, RK45, etc.)
        - Fixed-step manual implementations (euler, midpoint, rk4)
        """
        # Check if Julia method (capital first letter or contains parentheses)
        if cls._is_julia_method(method):
            try:
                from src.systems.base.numerical_integration.diffeqpy_integrator import (
                    DiffEqPyIntegrator,
                )

                return DiffEqPyIntegrator(
                    system, dt=dt, step_mode=step_mode, backend="numpy", algorithm=method, **options
                )
            except ImportError as e:
                raise ImportError(
                    f"Julia method '{method}' requires diffeqpy. "
                    f"Install Julia and run: julia> using Pkg; Pkg.add(\"DifferentialEquations\")\n"
                    f"Then: pip install diffeqpy\n"
                    f"Error: {e}"
                )

        # Check if manual fixed-step method
        elif cls._is_fixed_step_method(method):
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

        # Otherwise, use scipy
        else:
            from src.systems.base.numerical_integration.scipy_integrator import ScipyIntegrator

            # Note: ScipyIntegrator is always adaptive, doesn't accept step_mode parameter
            return ScipyIntegrator(system, dt=dt, method=method, backend="numpy", **options)

    @classmethod
    def _is_julia_method(cls, method: IntegrationMethod) -> bool:
        """
        Check if method is a Julia DiffEqPy method.

        Julia methods are identified by:
        1. Capital first letter (e.g., Tsit5, Vern9)
        2. Contains parentheses (e.g., AutoTsit5(Rosenbrock23()))

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

        # Check for parentheses (auto-switching methods)
        if "(" in method:
            return True

        # Check for capital first letter (standard Julia methods)
        if method[0].isupper():
            # Exclude scipy methods (which also start with capitals)
            scipy_methods = {"LSODA", "RK45", "RK23", "DOP853", "Radau", "BDF"}
            if method not in scipy_methods:
                return True
            return False

    @classmethod
    def _is_fixed_step_method(cls, method: IntegrationMethod) -> bool:
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
        return method in ["euler", "midpoint", "rk4"]

    @classmethod
    def _is_scipy_method(cls, method: IntegrationMethod) -> bool:
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
        scipy_methods = {"LSODA", "RK45", "RK23", "DOP853", "Radau", "BDF"}
        return method in scipy_methods

    @classmethod
    def _create_torch_integrator(
        cls,
        system: "ContinuousSystemBase",
        method: IntegrationMethod,
        dt: Optional[ScalarLike],
        step_mode: StepMode,
        **options,
    ):
        """Create PyTorch-based integrator using TorchDiffEq."""
        from src.systems.base.numerical_integration.torchdiffeq_integrator import (
            TorchDiffEqIntegrator,
        )

        # Always use TorchDiffEq for torch backend
        return TorchDiffEqIntegrator(
            system, dt=dt, step_mode=step_mode, backend="torch", method=method, **options
        )

    @classmethod
    def _create_jax_integrator(
        cls,
        system: "ContinuousSystemBase",
        method: IntegrationMethod,
        dt: Optional[ScalarLike],
        step_mode: StepMode,
        **options,
    ):
        """
        Create JAX-based integrator - always use Diffrax.

        Notes
        -----
        The 'solver' parameter is removed from options to avoid duplicate
        keyword argument errors, as it's passed explicitly via the 'method' parameter.
        This handles both calling styles:
        - create(backend='jax', method='tsit5')  # Preferred
        - create(backend='jax', solver='tsit5')  # Also works (handled in create())
        """
        from src.systems.base.numerical_integration.diffrax_integrator import DiffraxIntegrator

        # Remove 'solver' from options if present to avoid duplicate keyword argument
        # This can happen if user calls create(backend='jax', solver='tsit5')
        options_clean = {k: v for k, v in options.items() if k != "solver"}

        # Let Diffrax handle ALL methods, including euler/midpoint
        return DiffraxIntegrator(
            system,
            dt=dt,
            step_mode=step_mode,
            backend="jax",
            solver=method,  # Pass as explicit keyword argument
            **options_clean,  # Pass remaining options without 'solver'
        )

    # ========================================================================
    # Convenience Methods - Use Case-Specific Creation
    # ========================================================================

    @classmethod
    def auto(
        cls, system: "ContinuousSystemBase", prefer_backend: Optional[Backend] = None, **options
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
            System to integrate (controlled or autonomous)
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
        >>>
        >>> # Works with autonomous systems
        >>> integrator = IntegratorFactory.auto(autonomous_system)
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
        cls, system: "ContinuousSystemBase", use_julia: bool = False, **options
    ) -> IntegratorBase:
        """
        Create integrator for production use.

        Uses scipy.LSODA (default) or Julia's AutoTsit5 (if use_julia=True)
        with automatic stiffness detection. Most reliable choices.

        Parameters
        ----------
        system : SymbolicDynamicalSystem
            System to integrate (controlled or autonomous)
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
        >>>
        >>> # Autonomous system
        >>> integrator = IntegratorFactory.for_production(autonomous_system)
        """
        if use_julia:
            try:
                from src.systems.base.numerical_integration.diffeqpy_integrator import (
                    DiffEqPyIntegrator,
                )

                # Set conservative defaults
                default_options = {
                    "rtol": 1e-8,
                    "atol": 1e-10,
                }
                default_options.update(options)

                return DiffEqPyIntegrator(
                    system, backend="numpy", algorithm="AutoTsit5(Rosenbrock23())", **default_options
                )
            except ImportError:
                raise ImportError(
                    "Julia integration requires diffeqpy. "
                    "Install Julia and run: julia> using Pkg; Pkg.add(\"DifferentialEquations\")\n"
                    "Then: pip install diffeqpy"
                )
        else:
            # Use scipy LSODA
            default_options = {
                "rtol": 1e-8,
                "atol": 1e-10,
            }
            default_options.update(options)

            return cls.create(system, backend="numpy", method="LSODA", **default_options)

    @classmethod
    def for_optimization(
        cls, system: "ContinuousSystemBase", prefer_backend: Optional[Backend] = None, **options
    ) -> IntegratorBase:
        """
        Create integrator optimized for gradient-based optimization.

        Prefers JAX (Diffrax) for best performance with gradients.
        Falls back to PyTorch if JAX unavailable.

        Parameters
        ----------
        system : SymbolicDynamicalSystem
            System to integrate (controlled or autonomous)
        prefer_backend : Optional[str]
            Force specific backend ('jax' or 'torch')
        **options
            Additional options

        Returns
        -------
        IntegratorBase
            Optimization-ready integrator

        Examples
        --------
        >>> integrator = IntegratorFactory.for_optimization(system)
        >>> integrator = IntegratorFactory.for_optimization(system, prefer_backend='torch')
        >>>
        >>> # Autonomous system
        >>> integrator = IntegratorFactory.for_optimization(autonomous_system)
        """
        if prefer_backend == "torch":
            try:
                import torch

                return cls.create(system, backend="torch", method="dopri5", **options)
            except ImportError:
                raise ImportError("PyTorch backend requires: pip install torch torchdiffeq")

        elif prefer_backend == "jax":
            try:
                import jax

                return cls.create(system, backend="jax", method="tsit5", **options)
            except ImportError:
                raise ImportError("JAX backend requires: pip install jax diffrax")

        else:
            # Auto-select: prefer JAX, fall back to PyTorch
            try:
                import jax

                return cls.create(system, backend="jax", method="tsit5", **options)
            except ImportError:
                try:
                    import torch

                    return cls.create(system, backend="torch", method="dopri5", **options)
                except ImportError:
                    raise ImportError(
                        "Optimization requires JAX or PyTorch. Install either:\n"
                        "  pip install jax diffrax\n"
                        "  pip install torch torchdiffeq"
                    )

    @classmethod
    def for_neural_ode(
        cls, system: "ContinuousSystemBase", use_adjoint: bool = True, **options
    ) -> IntegratorBase:
        """
        Create integrator for Neural ODE training.

        Uses PyTorch with adjoint method for memory-efficient backpropagation.

        Parameters
        ----------
        system : SymbolicDynamicalSystem
            Neural ODE system (should be torch.nn.Module)
        use_adjoint : bool
            Use adjoint method for backprop. Default: True
        **options
            Additional options

        Returns
        -------
        IntegratorBase
            Neural ODE integrator

        Examples
        --------
        >>> neural_ode = MyNeuralODE()  # torch.nn.Module
        >>> integrator = IntegratorFactory.for_neural_ode(neural_ode)
        """
        try:
            import torch
        except ImportError:
            raise ImportError("Neural ODE requires PyTorch: pip install torch torchdiffeq")

        return cls.create(
            system, backend="torch", method="dopri5", adjoint=use_adjoint, **options
        )

    @classmethod
    def for_julia(
        cls,
        system: "ContinuousSystemBase",
        algorithm: str = "Tsit5",
        **options,
    ) -> IntegratorBase:
        """
        Create Julia-based integrator using DiffEqPy.

        Provides access to Julia's extensive solver library.

        Parameters
        ----------
        system : SymbolicDynamicalSystem
            System to integrate (controlled or autonomous)
        algorithm : str
            Julia algorithm name. Default: 'Tsit5'
            Examples: 'Vern9', 'Rosenbrock23', 'AutoTsit5(Rosenbrock23())'
        **options
            Additional options (reltol, abstol, etc.)

        Returns
        -------
        IntegratorBase
            Julia-powered integrator

        Examples
        --------
        >>> # High-accuracy solver
        >>> integrator = IntegratorFactory.for_julia(system, algorithm='Vern9')
        >>>
        >>> # Stiff system
        >>> integrator = IntegratorFactory.for_julia(
        ...     system, algorithm='Rosenbrock23'
        ... )
        >>>
        >>> # Autonomous system
        >>> integrator = IntegratorFactory.for_julia(autonomous_system)
        """
        try:
            from src.systems.base.numerical_integration.diffeqpy_integrator import (
                DiffEqPyIntegrator,
            )

            return DiffEqPyIntegrator(system, backend="numpy", algorithm=algorithm, **options)
        except ImportError:
            raise ImportError(
                "Julia integration requires diffeqpy. "
                "Install Julia and run: julia> using Pkg; Pkg.add(\"DifferentialEquations\")\n"
                "Then: pip install diffeqpy"
            )

    @classmethod
    def for_simple(
        cls,
        system: "ContinuousSystemBase",
        dt: ScalarLike = 0.01,
        backend: Backend = "numpy",
        **options,
    ) -> IntegratorBase:
        """
        Create simple RK4 fixed-step integrator.

        Good for prototyping and educational purposes.

        Parameters
        ----------
        system : SymbolicDynamicalSystem
            System to integrate (controlled or autonomous)
        dt : ScalarLike
            Time step
        backend : str
            Backend to use
        **options
            Additional options

        Returns
        -------
        IntegratorBase
            RK4 integrator

        Examples
        --------
        >>> integrator = IntegratorFactory.for_simple(system, dt=0.01)
        >>>
        >>> # Autonomous system
        >>> integrator = IntegratorFactory.for_simple(autonomous_system)
        """
        return cls.create(
            system, backend=backend, method="rk4", dt=dt, step_mode=StepMode.FIXED, **options
        )

    @classmethod
    def for_educational(
        cls,
        system: "ContinuousSystemBase",
        dt: ScalarLike = 0.01,
        backend: Backend = "numpy",
        **options,
    ) -> IntegratorBase:
        """
        Create Euler fixed-step integrator.

        Simplest method for learning and debugging.

        Parameters
        ----------
        system : SymbolicDynamicalSystem
            System to integrate (controlled or autonomous)
        dt : ScalarLike
            Time step
        backend : str
            Backend to use
        **options
            Additional options

        Returns
        -------
        IntegratorBase
            Euler integrator

        Examples
        --------
        >>> integrator = IntegratorFactory.for_educational(system, dt=0.001)
        >>>
        >>> # Autonomous system
        >>> integrator = IntegratorFactory.for_educational(autonomous_system)
        """
        return cls.create(
            system, backend=backend, method="euler", dt=dt, step_mode=StepMode.FIXED, **options
        )

    # ========================================================================
    # Information and Recommendation Methods
    # ========================================================================

    @staticmethod
    def list_methods(backend: Optional[Backend] = None) -> Dict[str, list]:
        """
        List available methods for each backend.

        Parameters
        ----------
        backend : Optional[str]
            If specified, list methods for that backend only

        Returns
        -------
        Dict[str, list]
            Methods organized by backend

        Examples
        --------
        >>> methods = IntegratorFactory.list_methods()
        >>> print(methods['numpy'])
        >>> print(methods['jax'])
        >>>
        >>> jax_methods = IntegratorFactory.list_methods('jax')
        """
        all_methods = {
            "numpy": [
                "LSODA",
                "RK45",
                "RK23",
                "DOP853",
                "Radau",
                "BDF",
                "euler",
                "midpoint",
                "rk4",
                # Julia methods
                "Tsit5",
                "Vern6",
                "Vern7",
                "Vern8",
                "Vern9",
                "Rosenbrock23",
                "AutoTsit5(Rosenbrock23())",
            ],
            "torch": [
                "dopri5",
                "dopri8",
                "bosh3",
                "euler",
                "midpoint",
                "rk4",
                "adaptive_heun",
                "fehlberg2",
            ],
            "jax": [
                "tsit5",
                "dopri5",
                "dopri8",
                "bosh3",
                "euler",
                "midpoint",
                "rk4",
                "heun",
                "ralston",
            ],
        }

        if backend:
            return {backend: all_methods.get(backend, [])}
        return all_methods

    @staticmethod
    def recommend(use_case: str, has_gpu: bool = False) -> Dict[str, Any]:
        """
        Get recommended integrator configuration for a use case.

        Parameters
        ----------
        use_case : str
            Use case: 'production', 'optimization', 'neural_ode', 'simple',
            'julia', 'educational'
        has_gpu : bool
            Whether GPU is available

        Returns
        -------
        Dict[str, Any]
            Recommended configuration with 'backend', 'method', 'description'

        Examples
        --------
        >>> rec = IntegratorFactory.recommend('optimization')
        >>> print(rec['backend'], rec['method'])
        'jax' 'tsit5'
        >>>
        >>> rec = IntegratorFactory.recommend('production')
        >>> print(rec['description'])
        """
        recommendations = {
            "production": {
                "backend": "numpy",
                "method": "LSODA",
                "description": "Most reliable, auto-detects stiffness",
            },
            "optimization": {
                "backend": "jax",
                "method": "tsit5",
                "description": "Best for gradient-based optimization",
            },
            "neural_ode": {
                "backend": "torch",
                "method": "dopri5",
                "adjoint": True,
                "description": "Memory-efficient for neural networks",
            },
            "simple": {
                "backend": "numpy",
                "method": "rk4",
                "dt": 0.01,
                "description": "Simple, reliable, educational",
            },
            "julia": {
                "backend": "numpy",
                "method": "Tsit5",
                "description": "Julia's excellent general-purpose solver",
            },
            "educational": {
                "backend": "numpy",
                "method": "euler",
                "dt": 0.01,
                "description": "Simplest method for learning",
            },
        }

        if use_case not in recommendations:
            raise ValueError(
                f"Unknown use case '{use_case}'. " f"Choose from: {list(recommendations.keys())}"
            )

        rec = recommendations[use_case].copy()

        # Adjust for GPU availability
        if has_gpu and use_case == "optimization":
            try:
                import torch

                rec["backend"] = "torch"
                rec["method"] = "dopri5"
            except ImportError:
                try:
                    import jax

                    rec["backend"] = "jax"
                    rec["method"] = "tsit5"
                except ImportError:
                    pass

        return rec

    @staticmethod
    def get_info(backend: Backend, method: IntegrationMethod) -> Dict[str, Any]:
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
        if backend == "numpy" and IntegratorFactory._is_julia_method(method):
            try:
                from src.systems.base.numerical_integration.diffeqpy_integrator import (
                    get_algorithm_info,
                )

                julia_info = get_algorithm_info(method)
                # Add backend/library metadata
                julia_info["backend"] = "numpy"
                julia_info["library"] = "Julia DifferentialEquations.jl"
                return julia_info
            except ImportError:
                # diffeqpy not installed - return generic info
                return {
                    "name": f"Julia: {method}",
                    "backend": "numpy",
                    "library": "Julia DifferentialEquations.jl (not installed)",
                    "description": f"Julia algorithm {method} (diffeqpy not available for details)",
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
            method, {"name": method, "description": "No information available", "backend": backend}
        )


# ============================================================================
# Convenience Functions
# ============================================================================


def create_integrator(
    system: "ContinuousSystemBase",
    backend: Backend = "numpy",
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
    >>>
    >>> # Autonomous system
    >>> integrator = create_integrator(autonomous_system)
    """
    return IntegratorFactory.create(system, backend, method, **options)


def auto_integrator(system: "ContinuousSystemBase", **options) -> IntegratorBase:
    """
    Automatically select best available integrator.

    Alias for IntegratorFactory.auto().

    Examples
    --------
    >>> integrator = auto_integrator(system, rtol=1e-8)
    >>>
    >>> # Autonomous system
    >>> integrator = auto_integrator(autonomous_system)
    """
    return IntegratorFactory.auto(system, **options)
