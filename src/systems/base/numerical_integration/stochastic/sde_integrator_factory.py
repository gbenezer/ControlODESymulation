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
SDE Integrator Factory - Unified Interface for Creating SDE Integrators

Provides a convenient factory class for creating the appropriate SDE integrator
based on backend, method, and requirements. Simplifies integrator selection
and configuration for stochastic differential equations.

Supports Julia's DifferentialEquations.jl (NumPy), Diffrax (JAX), and TorchSDE (PyTorch).
All integrators support both controlled and autonomous systems (nu=0).

Design Note
-----------
This module uses types from the centralized type system:
- ScalarLike for time values and parameters
- Backend for backend specifications
- SDEType, ConvergenceType, NoiseType from backends module
- SDEIntegrationMethod for algorithm names

This ensures consistency across the entire codebase and enables proper
type checking during factory operations.

Examples
--------
>>> # Automatic selection
>>> integrator = SDEIntegratorFactory.create(sde_system, backend='jax')
>>>
>>> # Specific method
>>> integrator = SDEIntegratorFactory.create(
...     sde_system, backend='numpy', method='SRIW1'
... )
>>>
>>> # Autonomous system
>>> integrator = SDEIntegratorFactory.create(autonomous_sde_system, backend='torch')
>>> result = integrator.integrate(
...     x0=torch.tensor([1.0, 0.0]),
...     u_func=lambda t, x: None,  # No control
...     t_span=(0.0, 10.0)
... )
>>>
>>> # Quick helpers
>>> integrator = SDEIntegratorFactory.auto(sde_system)  # Best for system
>>> integrator = SDEIntegratorFactory.for_optimization(sde_system)  # Best for gradients
>>> integrator = SDEIntegratorFactory.for_julia(sde_system)  # Best Julia solver
"""

from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional

# Import base classes and utilities
from src.systems.base.numerical_integration.integrator_base import StepMode
from src.systems.base.numerical_integration.stochastic.sde_integrator_base import (
    SDEIntegratorBase,
)
from src.types.backends import (
    Backend,
    ConvergenceType,
    SDEIntegrationMethod,
    SDEType,
    validate_backend,
)

# Import from centralized type system
from src.types.core import ScalarLike

if TYPE_CHECKING:
    from src.systems.base.core.continuous_stochastic_system import ContinuousStochasticSystem


class SDEIntegratorType(Enum):
    """
    SDE integrator type categories.

    Used for automatic selection based on use case.
    """

    PRODUCTION = "production"  # Julia DiffEqPy (SRIW1, EM)
    OPTIMIZATION = "optimization"  # Diffrax (Euler, SEA)
    NEURAL_SDE = "neural_sde"  # TorchSDE (euler with adjoint)
    JULIA = "julia"  # DiffEqPy (specialized algorithms)
    MONTE_CARLO = "monte_carlo"  # High-order weak methods
    SIMPLE = "simple"  # Euler-Maruyama


class SDEIntegratorFactory:
    """
    Factory for creating SDE numerical integrators.

    Provides convenient methods for creating SDE integrators based on:
    - Backend (numpy, torch, jax)
    - Method (EM, SRIW1, euler, etc.)
    - Use case (production, optimization, neural SDE, Julia)
    - Noise structure (additive, diagonal, general)

    Supports:
    - DiffEqPy (numpy): EM, SRIW1, SRA1, ImplicitEM, etc. (Julia solvers)
    - TorchSDE (torch): euler, milstein, srk, etc.
    - Diffrax (jax): Euler, ItoMilstein, SEA, SHARK, etc.

    All integrators support autonomous systems (nu=0) by passing u=None.

    Examples
    --------
    >>> # Create integrator by backend and method
    >>> integrator = SDEIntegratorFactory.create(
    ...     sde_system,
    ...     backend='numpy',
    ...     method='EM'
    ... )
    >>>
    >>> # Julia solver for high accuracy
    >>> integrator = SDEIntegratorFactory.create(
    ...     sde_system,
    ...     backend='numpy',
    ...     method='SRIW1'
    ... )
    >>>
    >>> # JAX for optimization
    >>> integrator = SDEIntegratorFactory.create(
    ...     sde_system,
    ...     backend='jax',
    ...     method='Euler'
    ... )
    >>>
    >>> # Automatic selection
    >>> integrator = SDEIntegratorFactory.auto(sde_system)
    >>>
    >>> # Use case-specific
    >>> integrator = SDEIntegratorFactory.for_optimization(sde_system)
    >>> integrator = SDEIntegratorFactory.for_neural_sde(neural_sde)
    >>> integrator = SDEIntegratorFactory.for_julia(sde_system, algorithm='SRA1')
    """

    # Default methods for each backend
    _BACKEND_DEFAULTS: Dict[Backend, SDEIntegrationMethod] = {
        "numpy": "EM",  # Julia Euler-Maruyama
        "torch": "euler",  # TorchSDE euler
        "jax": "Euler",  # Diffrax Euler
    }

    # Method to backend mapping
    _METHOD_TO_BACKEND: Dict[SDEIntegrationMethod, Backend] = {
        # Julia DiffEqPy methods (numpy only)
        # Euler-Maruyama family
        "EM": "numpy",
        "LambaEM": "numpy",
        "EulerHeun": "numpy",
        # Stochastic Runge-Kutta
        "SRIW1": "numpy",
        "SRIW2": "numpy",
        "SOSRI": "numpy",
        "SOSRI2": "numpy",
        "SRA": "numpy",
        "SRA1": "numpy",
        "SRA2": "numpy",
        "SRA3": "numpy",
        "SOSRA": "numpy",
        "SOSRA2": "numpy",
        # Milstein family
        "RKMil": "numpy",
        "RKMilCommute": "numpy",
        "RKMilGeneral": "numpy",
        # Implicit methods
        "ImplicitEM": "numpy",
        "ImplicitEulerHeun": "numpy",
        "ImplicitRKMil": "numpy",
        # IMEX methods
        "SKenCarp": "numpy",
        # Adaptive
        "AutoEM": "numpy",
        # Optimized
        "SRI": "numpy",
        "SRIW1Optimized": "numpy",
        "SRIW2Optimized": "numpy",
        # TorchSDE methods (torch only)
        "euler": "torch",
        "milstein": "torch",
        "srk": "torch",
        "midpoint": "torch",
        "reversible_heun": "torch",
        "adaptive_heun": "torch",
        # Diffrax methods (jax only)
        "Euler": "jax",
        "EulerHeun": "jax",
        "Heun": "jax",
        "ItoMilstein": "jax",
        "StratonovichMilstein": "jax",
        "SEA": "jax",
        "SHARK": "jax",
        "SRA1": "jax",  # Note: SRA1 exists in both Julia and Diffrax
        "ReversibleHeun": "jax",
    }

    @classmethod
    def create(
        cls,
        sde_system: "ContinuousStochasticSystem",
        backend: Backend = "numpy",
        method: Optional[SDEIntegrationMethod] = None,
        dt: Optional[ScalarLike] = 0.01,
        step_mode: StepMode = StepMode.FIXED,
        sde_type: Optional[SDEType] = None,
        convergence_type: ConvergenceType = ConvergenceType.STRONG,
        seed: Optional[int] = None,
        **options,
    ) -> SDEIntegratorBase:
        """
        Create an SDE integrator with specified backend and method.

        Parameters
        ----------
        sde_system : ContinuousStochasticSystem
            SDE system to integrate (controlled or autonomous)
        backend : Backend
            Backend: 'numpy', 'torch', 'jax'. Default: 'numpy'
        method : Optional[SDEIntegrationMethod]
            Solver method. If None, uses backend default.
            - numpy: 'EM' (Julia Euler-Maruyama)
            - torch: 'euler' (TorchSDE euler)
            - jax: 'Euler' (Diffrax Euler)
        dt : Optional[ScalarLike]
            Time step (default: 0.01)
        step_mode : StepMode
            FIXED or ADAPTIVE stepping (most SDE solvers use FIXED)
        sde_type : Optional[SDEType]
            SDE interpretation (None = use system's type)
        convergence_type : ConvergenceType
            Strong or weak convergence
        seed : Optional[int]
            Random seed for reproducibility
        **options
            Additional integrator options (rtol, atol, adjoint, etc.)

        Returns
        -------
        SDEIntegratorBase
            Configured SDE integrator

        Raises
        ------
        ValueError
            If backend/method combination is invalid
        ImportError
            If required package not installed

        Examples
        --------
        >>> # Use defaults (Julia EM)
        >>> integrator = SDEIntegratorFactory.create(sde_system)
        >>>
        >>> # Julia high-accuracy solver
        >>> integrator = SDEIntegratorFactory.create(
        ...     sde_system,
        ...     backend='numpy',
        ...     method='SRIW1',
        ...     dt=0.001
        ... )
        >>>
        >>> # JAX for optimization
        >>> integrator = SDEIntegratorFactory.create(
        ...     sde_system,
        ...     backend='jax',
        ...     method='Euler',
        ...     dt=0.01,
        ...     seed=42
        ... )
        >>>
        >>> # PyTorch for neural SDEs
        >>> integrator = SDEIntegratorFactory.create(
        ...     neural_sde,
        ...     backend='torch',
        ...     method='euler',
        ...     adjoint=True  # Memory-efficient backprop
        ... )
        >>>
        >>> # Autonomous system (no control)
        >>> integrator = SDEIntegratorFactory.create(autonomous_sde_system)
        >>> result = integrator.integrate(
        ...     x0=np.array([1.0, 0.0]),
        ...     u_func=lambda t, x: None,
        ...     t_span=(0.0, 10.0)
        ... )
        """
        # Validate backend
        backend = validate_backend(backend)

        # Select method
        if method is None:
            method = cls._BACKEND_DEFAULTS[backend]
        else:
            # Verify method is compatible with backend
            expected_backend = cls._METHOD_TO_BACKEND.get(method)
            if expected_backend and expected_backend != backend:
                raise ValueError(
                    f"Method '{method}' requires backend '{expected_backend}', "
                    f"but backend '{backend}' was specified. "
                    f"Either change backend or use a compatible method.",
                )

        # Create integrator based on backend
        if backend == "numpy":
            return cls._create_diffeqpy(
                sde_system,
                method,
                dt,
                step_mode,
                sde_type,
                convergence_type,
                seed,
                **options,
            )
        if backend == "torch":
            return cls._create_torchsde(
                sde_system,
                method,
                dt,
                step_mode,
                sde_type,
                convergence_type,
                seed,
                **options,
            )
        if backend == "jax":
            return cls._create_diffrax(
                sde_system,
                method,
                dt,
                step_mode,
                sde_type,
                convergence_type,
                seed,
                **options,
            )
        raise ValueError(f"Unknown backend: {backend}")

    @classmethod
    def _create_diffeqpy(
        cls,
        sde_system: "ContinuousStochasticSystem",
        method: SDEIntegrationMethod,
        dt: Optional[ScalarLike],
        step_mode: StepMode,
        sde_type: Optional[SDEType],
        convergence_type: ConvergenceType,
        seed: Optional[int],
        **options,
    ) -> SDEIntegratorBase:
        """Create Julia DiffEqPy integrator."""
        try:
            from src.systems.base.numerical_integration.stochastic.diffeqpy_sde_integrator import (
                DiffEqPySDEIntegrator,
            )
        except ImportError as e:
            raise ImportError(
                "DiffEqPy integration requires diffeqpy package. "
                "Install: pip install diffeqpy && python -c 'from diffeqpy import install; install()'",
            ) from e

        return DiffEqPySDEIntegrator(
            sde_system,
            dt=dt,
            step_mode=step_mode,
            backend="numpy",
            algorithm=method,
            sde_type=sde_type,
            convergence_type=convergence_type,
            seed=seed,
            **options,
        )

    @classmethod
    def _create_torchsde(
        cls,
        sde_system: "ContinuousStochasticSystem",
        method: SDEIntegrationMethod,
        dt: Optional[ScalarLike],
        step_mode: StepMode,
        sde_type: Optional[SDEType],
        convergence_type: ConvergenceType,
        seed: Optional[int],
        **options,
    ) -> SDEIntegratorBase:
        """Create TorchSDE integrator."""
        try:
            from src.systems.base.numerical_integration.stochastic.torchsde_integrator import (
                TorchSDEIntegrator,
            )
        except ImportError as e:
            raise ImportError(
                "TorchSDE integration requires torchsde package. "
                "Install: pip install torch torchsde",
            ) from e

        return TorchSDEIntegrator(
            sde_system,
            dt=dt,
            step_mode=step_mode,
            backend="torch",
            method=method,
            sde_type=sde_type,
            convergence_type=convergence_type,
            seed=seed,
            **options,
        )

    @classmethod
    def _create_diffrax(
        cls,
        sde_system: "ContinuousStochasticSystem",
        method: SDEIntegrationMethod,
        dt: Optional[ScalarLike],
        step_mode: StepMode,
        sde_type: Optional[SDEType],
        convergence_type: ConvergenceType,
        seed: Optional[int],
        **options,
    ) -> SDEIntegratorBase:
        """Create Diffrax integrator."""
        try:
            from src.systems.base.numerical_integration.stochastic.diffrax_sde_integrator import (
                DiffraxSDEIntegrator,
            )
        except ImportError as e:
            raise ImportError(
                "Diffrax integration requires jax and diffrax packages. "
                "Install: pip install jax diffrax",
            ) from e

        return DiffraxSDEIntegrator(
            sde_system,
            dt=dt,
            step_mode=step_mode,
            backend="jax",
            solver=method,
            sde_type=sde_type,
            convergence_type=convergence_type,
            seed=seed,
            **options,
        )

    @classmethod
    def auto(
        cls,
        sde_system: "ContinuousStochasticSystem",
        seed: Optional[int] = None,
        **options,
    ) -> SDEIntegratorBase:
        """
        Automatically select best available SDE integrator.

        Tries backends in order of preference:
        1. JAX (Diffrax) - best for gradients and performance
        2. PyTorch (TorchSDE) - good for neural SDEs
        3. NumPy (DiffEqPy) - fallback, most features

        Parameters
        ----------
        sde_system : ContinuousStochasticSystem
            SDE system to integrate
        seed : Optional[int]
            Random seed for reproducibility
        **options
            Additional integrator options

        Returns
        -------
        SDEIntegratorBase
            Best available integrator

        Examples
        --------
        >>> # Auto-select based on what's installed
        >>> integrator = SDEIntegratorFactory.auto(sde_system)
        >>>
        >>> # With reproducibility
        >>> integrator = SDEIntegratorFactory.auto(sde_system, seed=42)
        >>>
        >>> # Autonomous system
        >>> integrator = SDEIntegratorFactory.auto(autonomous_sde_system)
        """
        # Try backends in order of preference
        for backend in ["jax", "torch", "numpy"]:
            try:
                return cls.create(sde_system, backend=backend, seed=seed, **options)
            except ImportError:
                continue

        raise ImportError(
            "No SDE integrator backend available. "
            "Install at least one of: diffeqpy, torchsde, or diffrax",
        )

    @classmethod
    def for_optimization(
        cls,
        sde_system: "ContinuousStochasticSystem",
        backend: Optional[Backend] = None,
        **options,
    ) -> SDEIntegratorBase:
        """
        Create SDE integrator optimized for gradient-based optimization.

        Prefers JAX (Diffrax) for JIT compilation and autodiff,
        fallback to PyTorch (TorchSDE).

        Parameters
        ----------
        sde_system : ContinuousStochasticSystem
            SDE system to integrate
        backend : Optional[Backend]
            Preferred backend (None = auto-select)
        **options
            Additional integrator options

        Returns
        -------
        SDEIntegratorBase
            Optimization-ready integrator

        Examples
        --------
        >>> # Auto-select best optimization backend
        >>> integrator = SDEIntegratorFactory.for_optimization(sde_system)
        >>>
        >>> # Force specific backend
        >>> integrator = SDEIntegratorFactory.for_optimization(
        ...     sde_system,
        ...     backend='jax',
        ...     dt=0.001
        ... )
        """
        if backend:
            return cls.create(sde_system, backend=backend, **options)

        # Try JAX first, then PyTorch
        for preferred_backend in ["jax", "torch"]:
            try:
                return cls.create(sde_system, backend=preferred_backend, **options)
            except ImportError:
                continue

        raise ImportError(
            "Optimization requires JAX (diffrax) or PyTorch (torchsde). "
            "Install: pip install jax diffrax  OR  pip install torch torchsde",
        )

    @classmethod
    def for_neural_sde(
        cls,
        sde_system: "ContinuousStochasticSystem",
        adjoint: bool = True,
        **options,
    ) -> SDEIntegratorBase:
        """
        Create SDE integrator for neural SDEs.

        Uses TorchSDE with adjoint method for memory-efficient backpropagation.

        Parameters
        ----------
        sde_system : ContinuousStochasticSystem
            Neural SDE system to integrate
        adjoint : bool
            Use adjoint method (default: True)
        **options
            Additional integrator options

        Returns
        -------
        SDEIntegratorBase
            Neural SDE integrator

        Examples
        --------
        >>> # Neural SDE with adjoint
        >>> integrator = SDEIntegratorFactory.for_neural_sde(neural_sde)
        >>>
        >>> # Without adjoint (more memory, faster forward)
        >>> integrator = SDEIntegratorFactory.for_neural_sde(
        ...     neural_sde,
        ...     adjoint=False
        ... )
        """
        return cls.create(sde_system, backend="torch", method="euler", adjoint=adjoint, **options)

    @classmethod
    def for_julia(
        cls,
        sde_system: "ContinuousStochasticSystem",
        algorithm: SDEIntegrationMethod = "SRIW1",
        **options,
    ) -> SDEIntegratorBase:
        """
        Create Julia-based SDE integrator (DiffEqPy).

        Access to Julia's extensive SDE solver ecosystem.

        Parameters
        ----------
        sde_system : ContinuousStochasticSystem
            SDE system to integrate
        algorithm : SDEIntegrationMethod
            Julia algorithm (default: 'SRIW1')
        **options
            Additional integrator options

        Returns
        -------
        SDEIntegratorBase
            Julia-powered integrator

        Examples
        --------
        >>> # High-accuracy diagonal noise
        >>> integrator = SDEIntegratorFactory.for_julia(sde_system, algorithm='SRIW1')
        >>>
        >>> # Simple and fast
        >>> integrator = SDEIntegratorFactory.for_julia(sde_system, algorithm='EM')
        >>>
        >>> # Stiff drift
        >>> integrator = SDEIntegratorFactory.for_julia(
        ...     stiff_sde,
        ...     algorithm='ImplicitEM'
        ... )
        """
        return cls.create(sde_system, backend="numpy", method=algorithm, **options)

    @classmethod
    def for_monte_carlo(
        cls,
        sde_system: "ContinuousStochasticSystem",
        noise_type: str = "general",
        **options,
    ) -> SDEIntegratorBase:
        """
        Create SDE integrator optimized for Monte Carlo simulation.

        Uses weak convergence methods optimized for moment accuracy.

        Parameters
        ----------
        sde_system : ContinuousStochasticSystem
            SDE system to integrate
        noise_type : str
            'additive', 'diagonal', or 'general'
        **options
            Additional integrator options

        Returns
        -------
        SDEIntegratorBase
            Monte Carlo integrator with weak convergence

        Examples
        --------
        >>> # Additive noise (fastest)
        >>> integrator = SDEIntegratorFactory.for_monte_carlo(
        ...     sde_system,
        ...     noise_type='additive'
        ... )
        >>>
        >>> # General noise
        >>> integrator = SDEIntegratorFactory.for_monte_carlo(
        ...     sde_system,
        ...     noise_type='general'
        ... )
        """
        # Select method based on noise type
        if noise_type == "additive":
            method = "SRA3"  # Order 2.0 weak for additive noise
        elif noise_type == "diagonal":
            method = "SRA1"  # Order 2.0 weak for diagonal noise
        else:
            method = "EM"  # General fallback

        return cls.create(
            sde_system,
            backend="numpy",
            method=method,
            convergence_type=ConvergenceType.WEAK,
            **options,
        )

    @staticmethod
    def list_methods(
        backend: Optional[Backend] = None,
    ) -> Dict[Backend, List[SDEIntegrationMethod]]:
        """
        List available SDE methods for each backend.

        Parameters
        ----------
        backend : Optional[Backend]
            If specified, return only methods for that backend

        Returns
        -------
        Dict[Backend, List[SDEIntegrationMethod]]
            Methods available for each backend

        Examples
        --------
        >>> # All methods
        >>> methods = SDEIntegratorFactory.list_methods()
        >>> print(methods['jax'])
        ['Euler', 'EulerHeun', 'Heun', 'ItoMilstein', ...]
        >>>
        >>> # Specific backend
        >>> torch_methods = SDEIntegratorFactory.list_methods('torch')
        >>> print(torch_methods['torch'])
        ['euler', 'milstein', 'srk', ...]
        """
        all_methods: Dict[Backend, List[SDEIntegrationMethod]] = {
            "numpy": [
                # Julia DiffEqPy methods
                # Euler-Maruyama family
                "EM",
                "LambaEM",
                "EulerHeun",
                # Stochastic RK (high accuracy)
                "SRIW1",
                "SRIW2",
                "SOSRI",
                "SRA1",
                "SRA3",
                "SOSRA",
                # Milstein family
                "RKMil",
                "RKMilCommute",
                # Implicit (for stiff)
                "ImplicitEM",
                "ImplicitRKMil",
                # IMEX
                "SKenCarp",
                # Adaptive
                "AutoEM",
            ],
            "torch": [
                # TorchSDE methods
                "euler",
                "milstein",
                "srk",
                "midpoint",
                "reversible_heun",
                "adaptive_heun",
            ],
            "jax": [
                # Diffrax methods
                "Euler",
                "EulerHeun",
                "Heun",
                "ItoMilstein",
                "StratonovichMilstein",
                "SEA",
                "SHARK",
                "SRA1",
                "ReversibleHeun",
            ],
        }

        if backend:
            return {backend: all_methods.get(backend, [])}
        return all_methods

    @staticmethod
    def recommend(
        use_case: str,
        noise_type: str = "general",
        has_jax: bool = False,
        has_torch: bool = False,
        has_gpu: bool = False,
    ) -> Dict[str, Any]:
        """
        Get SDE integrator recommendation based on use case.

        Parameters
        ----------
        use_case : str
            One of: 'production', 'optimization', 'neural_sde',
            'monte_carlo', 'prototype', 'simple', 'julia'
        noise_type : str
            'additive', 'diagonal', 'general'
        has_jax : bool
            Whether JAX is available
        has_torch : bool
            Whether PyTorch is available
        has_gpu : bool
            Whether GPU is available

        Returns
        -------
        Dict[str, Any]
            Recommendation with 'backend', 'method', 'convergence_type'

        Examples
        --------
        >>> rec = SDEIntegratorFactory.recommend('production')
        >>> print(rec)
        {'backend': 'numpy', 'method': 'SRIW1', 'convergence_type': 'STRONG'}
        """
        recommendations = {
            "production": {
                "backend": "numpy",
                "method": "SRIW1" if noise_type == "diagonal" else "EM",
                "convergence_type": ConvergenceType.STRONG,
                "reason": "Julia DiffEqPy - most reliable, high accuracy",
            },
            "optimization": {
                "backend": "jax" if has_jax else "torch" if has_torch else "numpy",
                "method": "Euler" if has_jax else "euler" if has_torch else "EM",
                "convergence_type": ConvergenceType.STRONG,
                "reason": "Gradient support, JIT compilation",
            },
            "neural_sde": {
                "backend": "torch",
                "method": "euler",
                "convergence_type": ConvergenceType.STRONG,
                "adjoint": True,
                "reason": "Memory-efficient backprop through SDE",
            },
            "julia": {
                "backend": "numpy",
                "method": "SRIW1",
                "convergence_type": ConvergenceType.STRONG,
                "reason": "Access to Julia's powerful SDE solver ecosystem",
            },
            "monte_carlo": {
                "backend": "numpy",
                "method": "SRA3" if noise_type == "additive" else "SRA1",
                "convergence_type": ConvergenceType.WEAK,
                "reason": "Weak convergence optimized for moment accuracy",
            },
            "simple": {
                "backend": "numpy",
                "method": "EM",
                "convergence_type": ConvergenceType.STRONG,
                "dt": 0.01,
                "reason": "Simple, fast, easy to debug",
            },
            "prototype": {
                "backend": "numpy",
                "method": "EM",
                "convergence_type": ConvergenceType.STRONG,
                "dt": 0.01,
                "reason": "Simple, fast, easy to debug",
            },
        }

        if use_case not in recommendations:
            raise ValueError(
                f"Unknown use case '{use_case}'. Choose from: {list(recommendations.keys())}",
            )

        rec = recommendations[use_case].copy()

        # Adjust for GPU availability
        if has_gpu and use_case == "optimization":
            if has_torch:
                rec["backend"] = "torch"
                rec["method"] = "euler"
            elif has_jax:
                rec["backend"] = "jax"
                rec["method"] = "Euler"

        return rec

    @staticmethod
    def get_info(backend: Backend, method: SDEIntegrationMethod) -> Dict[str, Any]:
        """
        Get information about a specific SDE integrator configuration.

        Parameters
        ----------
        backend : Backend
            Backend name
        method : SDEIntegrationMethod
            Method name

        Returns
        -------
        Dict[str, Any]
            Information about the SDE integrator

        Examples
        --------
        >>> info = SDEIntegratorFactory.get_info('jax', 'Euler')
        >>> print(info['description'])
        'Basic Euler-Maruyama, fast and robust'
        """
        # Delegate to integrator-specific info functions
        if backend == "numpy":
            try:
                from src.systems.base.numerical_integration.stochastic.diffeqpy_sde_integrator import (
                    DiffEqPySDEIntegrator,
                )

                return DiffEqPySDEIntegrator.get_algorithm_info(method)
            except ImportError:
                return {
                    "name": f"Julia: {method}",
                    "description": "Julia SDE algorithm (diffeqpy not installed)",
                }

        elif backend == "torch":
            from src.systems.base.numerical_integration.stochastic.torchsde_integrator import (
                TorchSDEIntegrator,
            )

            return TorchSDEIntegrator.get_method_info(method)

        elif backend == "jax":
            from src.systems.base.numerical_integration.stochastic.diffrax_sde_integrator import (
                DiffraxSDEIntegrator,
            )

            return DiffraxSDEIntegrator.get_solver_info(method)

        return {"name": method, "description": "No information available", "backend": backend}


# ============================================================================
# Convenience Functions
# ============================================================================


def create_sde_integrator(
    sde_system: "ContinuousStochasticSystem",
    backend: Backend = "numpy",
    method: Optional[SDEIntegrationMethod] = None,
    **options,
) -> SDEIntegratorBase:
    """
    Convenience function for creating SDE integrators.

    Alias for SDEIntegratorFactory.create().

    Examples
    --------
    >>> integrator = create_sde_integrator(sde_system)
    >>> integrator = create_sde_integrator(sde_system, backend='jax', method='Euler')
    >>> integrator = create_sde_integrator(sde_system, backend='numpy', method='SRIW1')
    >>>
    >>> # Autonomous system
    >>> integrator = create_sde_integrator(autonomous_sde_system)
    """
    return SDEIntegratorFactory.create(sde_system, backend, method, **options)


def auto_sde_integrator(sde_system: "ContinuousStochasticSystem", **options) -> SDEIntegratorBase:
    """
    Automatically select best available SDE integrator.

    Alias for SDEIntegratorFactory.auto().

    Examples
    --------
    >>> integrator = auto_sde_integrator(sde_system, seed=42)
    >>>
    >>> # Autonomous system
    >>> integrator = auto_sde_integrator(autonomous_sde_system)
    """
    return SDEIntegratorFactory.auto(sde_system, **options)
