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
Continuous Stochastic System - Stochastic Differential Equations (SDEs)
========================================================================

Extends ContinuousSymbolicSystem with comprehensive SDE support through modular composition.

This class coordinates specialized handlers to provide:
    - Drift dynamics (via parent ContinuousSymbolicSystem)
    - Diffusion code generation (via DiffusionHandler)
    - Automatic noise analysis (via NoiseCharacterizer)
    - SDE-specific validation (via SDEValidator)
    - Multi-backend execution (inherited from parent)
    - SDE-specific integration via SDEIntegratorFactory

Mathematical Form
-----------------
Stochastic differential equations in Itô or Stratonovich form:

    dx = f(x, u, t)dt + g(x, u, t)dW

where:
    - f(x, u, t): Drift vector (nx × 1) - deterministic dynamics
    - g(x, u, t): Diffusion matrix (nx × nw) - stochastic intensity
    - dW: Brownian motion increments (nw independent Wiener processes)
    - x ∈ ℝⁿˣ: State vector
    - u ∈ ℝⁿᵘ: Control input
    - t ∈ ℝ: Time

Architecture
-----------
This class is THIN because it delegates to:
    - ContinuousSymbolicSystem (parent): ALL drift dynamics logic
    - DiffusionHandler: Diffusion code generation and caching
    - NoiseCharacterizer: Automatic noise structure analysis (via DiffusionHandler)
    - SDEValidator: SDE-specific validation
    - BackendManager: Type conversions (inherited)
    - SDEIntegratorFactory: Stochastic integration (replaces deterministic integrators)

Noise Types Detected Automatically
---------------------------------
- ADDITIVE: g(x,u,t) = constant (most efficient)
- MULTIPLICATIVE: g(x,u,t) depends on state
- DIAGONAL: Independent noise sources
- SCALAR: Single Wiener process (nw = 1)
- GENERAL: Full coupling

The framework automatically selects efficient specialized solvers based on
detected noise structure.

Usage Pattern
-------------
Users subclass ContinuousStochasticSystem and implement define_system() to
specify both drift and diffusion:

    class OrnsteinUhlenbeck(ContinuousStochasticSystem):
        '''Ornstein-Uhlenbeck process with mean reversion.'''

        def define_system(self, alpha=1.0, sigma=0.5):
            # Define symbolic variables
            x = sp.symbols('x', real=True)
            u = sp.symbols('u', real=True)

            # Define symbolic parameters
            alpha_sym = sp.symbols('alpha', positive=True)
            sigma_sym = sp.symbols('sigma', positive=True)

            # Define drift (deterministic part)
            self.state_vars = [x]
            self.control_vars = [u]
            self._f_sym = sp.Matrix([[-alpha_sym * x + u]])
            self.parameters = {alpha_sym: alpha, sigma_sym: sigma}
            self.order = 1

            # Define diffusion (stochastic part)
            self.diffusion_expr = sp.Matrix([[sigma_sym]])
            self.sde_type = 'ito'  # String-based API (or omit for default)

    system = OrnsteinUhlenbeck(alpha=2.0, sigma=0.3)

This pattern is consistent with ContinuousSymbolicSystem.

Pure Diffusion Processes
------------------------
Systems with zero drift (f = 0) are fully supported. These are useful for:
- Brownian motion (pure random walk)
- Diffusion-only processes in physics
- Noise models for stochastic control

To define a pure diffusion process, set the drift to zero:

    class PureBrownianMotion(ContinuousStochasticSystem):
        '''2D Brownian motion with no drift.'''

        def define_system(self, sigma1=0.5, sigma2=0.3):
            x1, x2 = sp.symbols('x1 x2', real=True)
            sigma1_sym = sp.symbols('sigma1', positive=True)
            sigma2_sym = sp.symbols('sigma2', positive=True)

            # Zero drift
            self.state_vars = [x1, x2]
            self.control_vars = []  # Autonomous
            self._f_sym = sp.Matrix([[0], [0]])  # Zero drift!
            self.parameters = {sigma1_sym: sigma1, sigma2_sym: sigma2}
            self.order = 1

            # Diffusion only
            self.diffusion_expr = sp.Matrix([
                [sigma1_sym, 0],
                [0, sigma2_sym]
            ])
            self.sde_type = 'ito'

Examples
--------
>>> # Ornstein-Uhlenbeck process (additive noise)
>>> class OrnsteinUhlenbeck(ContinuousStochasticSystem):
...     def define_system(self, alpha=1.0, sigma=0.5):
...         x = sp.symbols('x')
...         u = sp.symbols('u')
...         alpha_sym = sp.symbols('alpha', positive=True)
...         sigma_sym = sp.symbols('sigma', positive=True)
...
...         self.state_vars = [x]
...         self.control_vars = [u]
...         self._f_sym = sp.Matrix([[-alpha_sym * x + u]])
...         self.parameters = {alpha_sym: alpha, sigma_sym: sigma}
...         self.order = 1
...
...         self.diffusion_expr = sp.Matrix([[sigma_sym]])
>>>
>>> system = OrnsteinUhlenbeck(alpha=2.0, sigma=0.3)
>>> system.is_additive_noise()
True
>>>
>>> # Geometric Brownian motion (multiplicative noise)
>>> class GeometricBrownianMotion(ContinuousStochasticSystem):
...     def define_system(self, mu=0.1, sigma=0.2):
...         x = sp.symbols('x', positive=True)
...         u = sp.symbols('u')
...         mu_sym, sigma_sym = sp.symbols('mu sigma', positive=True)
...
...         self.state_vars = [x]
...         self.control_vars = [u]
...         self._f_sym = sp.Matrix([[mu_sym * x + u]])
...         self.parameters = {mu_sym: mu, sigma_sym: sigma}
...         self.order = 1
...
...         self.diffusion_expr = sp.Matrix([[sigma_sym * x]])
>>>
>>> gbm = GeometricBrownianMotion(mu=0.15, sigma=0.25)
>>> gbm.is_multiplicative_noise()
True
>>> gbm.recommend_solvers('jax')
['euler_heun', 'heun', 'reversible_heun']

See Also
--------
- ContinuousSymbolicSystem: Parent class for drift dynamics
- DiffusionHandler: Diffusion code generation
- NoiseCharacterizer: Automatic noise analysis
- SDEValidator: SDE validation
- SDEIntegratorFactory: Stochastic integration

Authors
-------
Gil Benezer

License
-------
AGPL-3.0
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np
import sympy as sp

from src.systems.base.core.continuous_symbolic_system import ContinuousSymbolicSystem
from src.systems.base.numerical_integration.stochastic.sde_integrator_factory import (
    SDEIntegratorFactory,
)
from src.systems.base.utils.stochastic.diffusion_handler import DiffusionHandler
from src.systems.base.utils.stochastic.noise_analysis import (
    NoiseCharacteristics,
    NoiseType,
)
from src.systems.base.utils.stochastic.sde_validator import SDEValidator, ValidationError
from src.types.backends import Backend, SDEIntegrationMethod, SDEType

# Type imports
from src.types.core import ControlInput, ControlVector, ScalarLike, StateVector
from src.types.trajectories import TimePoints, TimeSpan


class ContinuousStochasticSystem(ContinuousSymbolicSystem):
    """
    Concrete symbolic continuous-time stochastic dynamical system (SDE).

    Extends ContinuousSymbolicSystem to handle stochastic differential equations.
    Users subclass this and implement define_system() to specify both drift
    and diffusion terms.

    Represents systems of the form:
        dx = f(x, u, t)dt + g(x, u, t)dW
        y = h(x)

    where:
        f: Drift (deterministic part) - inherited from parent
        g: Diffusion (stochastic part) - added by this class
        dW: Brownian motion increments

    Attributes (Set by User in define_system)
    -----------------------------------------
    diffusion_expr : sp.Matrix
        Symbolic diffusion matrix g(x, u), shape (nx, nw)
        REQUIRED - must be set in define_system()
    sde_type : SDEType or str
        SDE interpretation ('ito' or 'stratonovich')
        Optional - defaults to Itô

    Attributes (Created Automatically)
    ----------------------------------
    diffusion_handler : DiffusionHandler
        Generates and caches diffusion functions
    noise_characteristics : NoiseCharacteristics
        Automatic noise structure analysis results
    nw : int
        Number of independent Wiener processes
    is_stochastic : bool
        Always True for this class

    Examples
    --------
    >>> class OrnsteinUhlenbeck(ContinuousStochasticSystem):
    ...     '''Ornstein-Uhlenbeck process with mean reversion.'''
    ...
    ...     def define_system(self, alpha=1.0, sigma=0.5):
    ...         x = sp.symbols('x')
    ...         u = sp.symbols('u')
    ...         alpha_sym = sp.symbols('alpha', positive=True)
    ...         sigma_sym = sp.symbols('sigma', positive=True)
    ...
    ...         # Drift (deterministic part)
    ...         self.state_vars = [x]
    ...         self.control_vars = [u]
    ...         self._f_sym = sp.Matrix([[-alpha_sym * x + u]])
    ...         self.parameters = {alpha_sym: alpha, sigma_sym: sigma}
    ...         self.order = 1
    ...
    ...         # Diffusion (stochastic part)
    ...         self.diffusion_expr = sp.Matrix([[sigma_sym]])
    ...         self.sde_type = 'ito'
    >>>
    >>> # Instantiate system
    >>> system = OrnsteinUhlenbeck(alpha=2.0, sigma=0.3)
    >>>
    >>> # Automatic noise analysis
    >>> print(system.noise_characteristics.noise_type)
    NoiseType.ADDITIVE
    >>>
    >>> # Evaluate drift and diffusion
    >>> x = np.array([1.0])
    >>> u = np.array([0.0])
    >>> f = system.drift(x, u)  # Drift term
    >>> g = system.diffusion(x, u)  # Diffusion matrix
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize continuous stochastic system.

        Follows the template method pattern:
        1. Initialize SDE-specific containers
        2. Call parent __init__ (which calls define_system and validates)
        3. Validate SDE-specific attributes
        4. Initialize SDE-specific components

        Parameters
        ----------
        *args : tuple
            Positional arguments passed to define_system()
        **kwargs : dict
            Keyword arguments passed to define_system()

        Raises
        ------
        ValidationError
            If SDE system definition is invalid
        ValueError
            If diffusion_expr is not set in define_system()
        """

        # ====================================================================
        # SDE-Specific Containers (Before Parent Init)
        # ====================================================================

        self.diffusion_expr: Optional[sp.Matrix] = None
        """Symbolic diffusion matrix g(x, u) - MUST be set in define_system()"""

        self.sde_type: Union[str, SDEType] = "ito"
        """SDE interpretation - can be 'ito' or 'stratonovich' (string or enum)"""

        # Placeholders for components (created after parent init)
        self.diffusion_handler: Optional[DiffusionHandler] = None
        self.noise_characteristics: Optional[NoiseCharacteristics] = None
        self.nw: int = 0

        # ====================================================================
        # Parent Class Initialization
        # ====================================================================

        # CRITICAL: Call parent __init__ which handles all drift logic
        # This calls define_system(), validates drift, initializes dynamics evaluator
        super().__init__(*args, **kwargs)

        # ====================================================================
        # SDE-Specific Validation and Initialization
        # ====================================================================

        # Validate that diffusion_expr was set by user
        if self.diffusion_expr is None:
            raise ValueError(
                f"{self.__class__.__name__}.define_system() must set self.diffusion_expr.\n"
                "Example:\n"
                "  def define_system(self):\n"
                "      # ... set drift attributes ...\n"
                "      self.diffusion_expr = sp.Matrix([[sigma * x]])",
            )

        # Run SDE-specific validation
        self._validate_sde_system()

        # Initialize SDE-specific components
        self._initialize_sde_components()

    def _validate_sde_system(self):
        """
        Validate SDE-specific constraints.

        Uses SDEValidator to check:
        - Drift-diffusion dimension compatibility
        - Symbol resolution in diffusion
        - Noise type claims (if provided)

        Raises
        ------
        ValidationError
            If SDE validation fails
        """
        validator = SDEValidator(
            drift_expr=self._f_sym,
            diffusion_expr=self.diffusion_expr,
            state_vars=self.state_vars,
            control_vars=self.control_vars,
            time_var=getattr(self, "time_var", None),
            parameters=self.parameters,
        )

        try:
            validator.validate(raise_on_error=True)
        except ValidationError as e:
            raise ValidationError(
                f"SDE validation failed for {self.__class__.__name__}:\n{e!s}",
            ) from e

    def _initialize_sde_components(self):
        """
        Initialize SDE-specific components after validation.

        Creates:
        - DiffusionHandler for code generation
        - Extracts NoiseCharacteristics from handler
        - Sets noise dimension (nw)
        - Normalizes sde_type to enum
        """
        # Normalize sde_type to enum
        if isinstance(self.sde_type, str):
            sde_type_lower = self.sde_type.lower()
            if sde_type_lower not in ["ito", "stratonovich"]:
                raise ValueError(
                    f"Invalid sde_type '{self.sde_type}'. Must be 'ito' or 'stratonovich'",
                )
            self.sde_type = SDEType(sde_type_lower)
        elif not isinstance(self.sde_type, SDEType):
            raise TypeError(
                f"sde_type must be string or SDEType enum, got {type(self.sde_type).__name__}",
            )

        # Create DiffusionHandler for code generation
        self.diffusion_handler = DiffusionHandler(
            diffusion_expr=self.diffusion_expr,
            state_vars=self.state_vars,
            control_vars=self.control_vars,
            time_var=getattr(self, "time_var", None),
            parameters=self.parameters,
        )

        # Extract noise characteristics
        self.noise_characteristics = self.diffusion_handler.characteristics

        # Set noise dimension
        self.nw = self.diffusion_expr.shape[1]

    # ========================================================================
    # Override ContinuousSystemBase Interface for SDE Integration
    # ========================================================================

    def integrate(
        self,
        x0: StateVector,
        u: ControlInput = None,
        t_span: TimeSpan = (0.0, 10.0),
        method: SDEIntegrationMethod = "euler_maruyama",
        t_eval: Optional[TimePoints] = None,
        n_paths: int = 1,
        seed: Optional[int] = None,
        **integrator_kwargs,
    ):
        """
        Integrate stochastic system using SDE solver.

        **CRITICAL OVERRIDE**: This method overrides the parent's deterministic
        integrate() to use SDEIntegratorFactory instead of IntegratorFactory.
        This ensures proper handling of Brownian motion and noise structure.

        Parameters
        ----------
        x0 : StateVector
            Initial state (nx,)
        u : ControlInput
            Control input (constant, callable, or None)
        t_span : TimeSpan
            Integration interval (t_start, t_end)
        method : str
            SDE integration method (default: 'euler_maruyama')
        t_eval : Optional[TimePoints]
            Specific times to return solution
        n_paths : int
            Number of Monte Carlo paths to simulate (default: 1)
            For n_paths > 1, performs Monte Carlo simulation
        seed : Optional[int]
            Random seed for reproducibility
        **integrator_kwargs
            Additional options:
            - dt : float (required for most SDE methods)
            - rtol, atol : float (adaptive methods only)
            - convergence_type : ConvergenceType ('strong' or 'weak')

        Returns
        -------
        SDEIntegrationResult
            TypedDict containing:
            - t: Time points (T,)
            - x: State trajectories
            - Single path: (T, nx)
            - Multiple paths: (n_paths, T, nx)
            - success: Integration success
            - n_paths: Number of paths
            - noise_type: Detected noise type
            - sde_type: Itô or Stratonovich
            - nfev: Drift function evaluations
            - diffusion_evals: Diffusion evaluations
            - integration_time: Computation time

        Examples
        --------
        Single trajectory:
        >>> result = system.integrate(
        ...     x0=np.array([1.0, 0.0]),
        ...     u=None,
        ...     t_span=(0.0, 10.0),
        ...     method='euler_maruyama',
        ...     dt=0.01,
        ...     seed=42
        ... )

        Monte Carlo simulation:
        >>> result = system.integrate(
        ...     x0=x0,
        ...     t_span=(0, 10),
        ...     n_paths=1000,
        ...     dt=0.01,
        ...     seed=42
        ... )
        >>> mean_traj = result['x'].mean(axis=0)

        State feedback:
        >>> def controller(x, t):
        ...     return -K @ x
        >>> result = system.integrate(x0, controller, t_span=(0, 10), dt=0.01)
        """
        # Convert control input to standard (t, x) -> u form
        u_func = self._prepare_control_input(u)

        # CRITICAL FIX: Use positional argument for sde_system
        # The factory signature is: create(sde_system, backend, method, ...)
        integrator = SDEIntegratorFactory.create(
            sde_system=self,  # First positional argument
            backend=self._default_backend,
            method=method,
            seed=seed,
            **integrator_kwargs,
        )

        # Single path vs Monte Carlo
        if n_paths == 1:
            # Single trajectory
            return integrator.integrate(x0=x0, u_func=u_func, t_span=t_span, t_eval=t_eval)
        # Monte Carlo simulation
        if hasattr(integrator, "integrate_monte_carlo"):
            return integrator.integrate_monte_carlo(
                x0=x0,
                u_func=u_func,
                t_span=t_span,
                n_paths=n_paths,
                t_eval=t_eval,
            )
        # Manual Monte Carlo (run n_paths separate integrations)
        import warnings

        warnings.warn(
            f"Integrator '{method}' does not have native Monte Carlo support. "
            f"Running {n_paths} separate integrations (may be slow).",
            UserWarning,
        )

        all_paths = []
        for i in range(n_paths):
            # Set different seed for each path
            path_seed = seed + i if seed is not None else None
            path_integrator = SDEIntegratorFactory.create(
                sde_system=self,
                backend=self._default_backend,
                method=method,
                seed=path_seed,
                **integrator_kwargs,
            )

            result = path_integrator.integrate(
                x0=x0,
                u_func=u_func,
                t_span=t_span,
                t_eval=t_eval,
            )
            all_paths.append(result["x"])

        # Stack all paths
        x_all = np.stack(all_paths, axis=0)  # (n_paths, T, nx)

        # Return combined result
        return {
            **result,  # Use last result for metadata
            "x": x_all,
            "n_paths": n_paths,
            "message": f"Monte Carlo with {n_paths} paths (manual mode)",
        }

    # ========================================================================
    # Primary Interface - Drift and Diffusion Evaluation
    # ========================================================================

    def drift(
        self,
        x: StateVector,
        u: Optional[ControlVector] = None,
        t: ScalarLike = 0.0,
        backend: Optional[Backend] = None,
    ) -> StateVector:
        """
        Evaluate drift term f(x, u, t) or f(x, t) for autonomous.

        Parameters
        ----------
        x : StateVector
            State vector (nx,) or batch (batch, nx)
        u : Optional[ControlVector]
            Control vector (nu,) or batch (batch, nu)
            For autonomous systems (nu=0), u can be None
        t : float
            Time (currently ignored for time-invariant systems)
        backend : Optional[Backend]
            Backend selection (None = auto-detect)

        Returns
        -------
        StateVector
            Drift vector f(x, u), shape (nx,) or (batch, nx)

        Notes
        -----
        Delegates to parent class - reuses ALL drift evaluation logic.
        Supports both controlled and autonomous SDEs.

        Examples
        --------
        Controlled SDE:
        >>> f = system.drift(np.array([1.0]), np.array([0.0]))
        >>> print(f)
        [-1.0]

        Autonomous SDE:
        >>> f = system.drift(np.array([1.0]))  # u=None
        >>> print(f)
        [-2.0]
        """
        # DELEGATE: Parent class handles drift evaluation
        return super().__call__(x, u, t, backend=backend)

    def diffusion(
        self,
        x: StateVector,
        u: Optional[ControlVector] = None,
        t: ScalarLike = 0.0,
        backend: Optional[Backend] = None,
    ):
        """
        Evaluate diffusion term g(x, u, t) or g(x, t) for autonomous.

        Parameters
        ----------
        x : StateVector
            State vector (nx,) or batch (batch, nx)
        u : Optional[ControlVector]
            Control vector (nu,) or batch (batch, nu)
            For autonomous systems (nu=0), u can be None
        t : float
            Time (currently ignored)
        backend : Optional[Backend]
            Backend selection (None = auto-detect)

        Returns
        -------
        ArrayLike
            Diffusion matrix g(x, u), shape (nx, nw) or (batch, nx, nw)

        Examples
        --------
        Controlled SDE:
        >>> g = system.diffusion(np.array([2.0]), np.array([0.0]))
        >>> print(g.shape)
        (1, 1)

        Autonomous SDE:
        >>> g = system.diffusion(np.array([2.0]))  # u=None
        >>> print(g.shape)
        (1, 1)

        For additive noise (precompute once):
        >>> if system.is_additive_noise():
        ...     G = system.get_constant_noise()  # Precompute once
        """
        # Determine backend
        backend_to_use = backend if backend else self._default_backend

        # Generate/get cached diffusion function
        func = self.diffusion_handler.generate_function(backend_to_use)

        # Handle autonomous systems
        if u is None:
            if self.nu > 0:
                raise ValueError(
                    f"Non-autonomous system requires control input u. "
                    f"System has {self.nu} control input(s).",
                )
            # Create empty control array
            if backend_to_use == "numpy":
                u = np.array([])
            elif backend_to_use == "torch":
                import torch

                u = torch.tensor([])
            elif backend_to_use == "jax":
                import jax.numpy as jnp

                u = jnp.array([])
        elif u is not None and self.nu == 0:
            raise ValueError("Autonomous system cannot take control input")

        # Convert to appropriate backend
        if backend_to_use == "numpy":
            x_arr = np.atleast_1d(np.asarray(x))
            u_arr = np.atleast_1d(np.asarray(u)) if self.nu > 0 else np.array([])

            # Check for empty batch
            if x_arr.ndim > 1 and x_arr.shape[0] == 0:
                raise ValueError("Empty batch detected in diffusion evaluation (batch_size=0).")

        elif backend_to_use == "torch":
            import torch

            x_arr = torch.atleast_1d(torch.as_tensor(x))
            u_arr = torch.atleast_1d(torch.as_tensor(u)) if self.nu > 0 else torch.tensor([])

            if len(x_arr.shape) > 1 and x_arr.shape[0] == 0:
                raise ValueError("Empty batch detected in diffusion evaluation (batch_size=0).")

        elif backend_to_use == "jax":
            import jax.numpy as jnp

            x_arr = jnp.atleast_1d(jnp.asarray(x))
            u_arr = jnp.atleast_1d(jnp.asarray(u)) if self.nu > 0 else jnp.array([])

            if x_arr.ndim > 1 and x_arr.shape[0] == 0:
                raise ValueError("Empty batch detected in diffusion evaluation (batch_size=0).")
        else:
            raise ValueError(f"Unknown backend: {backend_to_use}")

        # Unpack arrays for lambdified function
        if x_arr.ndim == 1:
            # Single evaluation
            x_list = [x_arr[i] for i in range(self.nx)]
            u_list = [u_arr[i] for i in range(self.nu)] if self.nu > 0 else []
        else:
            # Batched evaluation
            x_list = [x_arr[:, i] for i in range(self.nx)]
            u_list = [u_arr[:, i] for i in range(self.nu)] if self.nu > 0 else []

        # Evaluate diffusion function
        result = func(*(x_list + u_list))
        result = self.backend.ensure_type(result, backend_to_use)

        return result

    def __call__(
        self,
        x: StateVector,
        u: Optional[ControlVector] = None,
        t: ScalarLike = 0.0,
        backend: Optional[Backend] = None,
    ) -> StateVector:
        """
        Evaluate drift dynamics: dx/dt = f(x, u, t) or dx/dt = f(x, t).

        Makes the system callable like the parent class.
        For diffusion, use system.diffusion(x, u, t).

        Parameters
        ----------
        x : StateVector
            State vector
        u : Optional[ControlVector]
            Control vector (None for autonomous systems)
        t : float
            Time
        backend : Optional[Backend]
            Backend selection

        Returns
        -------
        StateVector
            Drift term f(x, u, t) or f(x, t)

        Notes
        -----
        This evaluates ONLY the drift term for consistency with parent class.
        Use system.diffusion(x, u, t) for the diffusion term.

        Examples
        --------
        >>> # Drift evaluation (deterministic part)
        >>> f = system(x, u)
        >>>
        >>> # Diffusion evaluation (stochastic part)
        >>> g = system.diffusion(x, u)
        >>>
        >>> # For SDE simulation:
        >>> dt = 0.01
        >>> dW = np.random.randn(nw) * np.sqrt(dt)
        >>> dx = f * dt + g @ dW
        """
        return self.drift(x, u, t, backend)

    # ========================================================================
    # Override Linearization to Include Diffusion Matrix
    # ========================================================================

    def linearize(self, x_eq: StateVector, u_eq: Optional[ControlVector] = None):
        """
        Compute linearization including diffusion: A = ∂f/∂x, B = ∂f/∂u, G = g(x_eq).

        For stochastic systems, linearization returns three matrices:
        - A: State Jacobian ∂f/∂x
        - B: Control Jacobian ∂f/∂u
        - G: Diffusion matrix evaluated at equilibrium g(x_eq, u_eq)

        Parameters
        ----------
        x_eq : StateVector
            Equilibrium state (nx,)
        u_eq : Optional[ControlVector]
            Equilibrium control (nu,)

        Returns
        -------
        Tuple[ArrayLike, ArrayLike, ArrayLike]
            (A, B, G) where:
            - A: State Jacobian (nx, nx)
            - B: Control Jacobian (nx, nu)
            - G: Diffusion matrix (nx, nw)

        Examples
        --------
        >>> x_eq = np.zeros(2)
        >>> u_eq = np.zeros(1)
        >>> A, B, G = system.linearize(x_eq, u_eq)
        >>>
        >>> # Check continuous stability: Re(λ) < 0
        >>> eigenvalues = np.linalg.eigvals(A)
        >>> is_stable = np.all(np.real(eigenvalues) < 0)
        """
        # Get drift linearization from parent
        A, B = super().linearize(x_eq, u_eq)

        # Evaluate diffusion at equilibrium
        G = self.diffusion(x_eq, u_eq)

        return (A, B, G)

    # ========================================================================
    # Query Methods
    # ========================================================================

    def is_additive_noise(self) -> bool:
        """Check if noise is additive (constant, state-independent)."""
        return self.noise_characteristics.is_additive

    def is_multiplicative_noise(self) -> bool:
        """Check if noise is multiplicative (state-dependent)."""
        return self.noise_characteristics.is_multiplicative

    def is_diagonal_noise(self) -> bool:
        """Check if noise sources are independent (diagonal diffusion)."""
        return self.noise_characteristics.is_diagonal

    def is_scalar_noise(self) -> bool:
        """Check if system has single noise source."""
        return self.noise_characteristics.is_scalar

    def is_pure_diffusion(self) -> bool:
        """Check if system is pure diffusion (zero drift)."""
        return all(expr == 0 for expr in self._f_sym)

    def get_noise_type(self) -> NoiseType:
        """Get classified noise type."""
        return self.noise_characteristics.noise_type

    def get_sde_type(self) -> SDEType:
        return self.sde_type

    # May need to make this more elaborate or otherwise replace with
    # cleaner implementation
    def get_diffusion_matrix(self, x, u=None, backend: Optional[Backend] = None):
        return self.diffusion(x, u, backend=backend)

    def depends_on_state(self) -> bool:
        """Check if diffusion depends on state variables."""
        return self.noise_characteristics.depends_on_state

    def depends_on_control(self) -> bool:
        """Check if diffusion depends on control inputs."""
        return self.noise_characteristics.depends_on_control

    def depends_on_time(self) -> bool:
        """Check if diffusion depends on time."""
        return self.noise_characteristics.depends_on_time

    # ========================================================================
    # Solver Recommendations
    # ========================================================================

    def recommend_solvers(self, backend: Backend = "jax") -> List[str]:
        """
        Recommend efficient SDE solvers based on noise structure.

        Parameters
        ----------
        backend : str
            Integration backend ('jax', 'torch', 'numpy')

        Returns
        -------
        List[str]
            Recommended solver names, ordered by efficiency/accuracy

        Examples
        --------
        >>> solvers = system.recommend_solvers('jax')
        >>> print(solvers)
        ['sea', 'shark', 'sra1']  # For additive noise
        """
        return self.noise_characteristics.recommended_solvers(backend)

    def get_optimization_opportunities(self) -> Dict[str, bool]:
        """Get optimization opportunities based on noise structure."""
        return self.diffusion_handler.get_optimization_opportunities()

    # ========================================================================
    # Constant Noise Optimization (Additive Only)
    # ========================================================================

    def get_constant_noise(self, backend: Backend = "numpy"):
        """
        Get constant noise matrix for additive noise.

        For additive noise, diffusion is constant and can be precomputed
        once for significant performance gains.

        Parameters
        ----------
        backend : str
            Backend for array type

        Returns
        -------
        ArrayLike
            Constant diffusion matrix (nx, nw)

        Raises
        ------
        ValueError
            If noise is not additive
        """
        return self.diffusion_handler.get_constant_noise(backend)

    def can_optimize_for_additive(self) -> bool:
        """Check if additive-noise optimizations are applicable."""
        return self.diffusion_handler.can_optimize_for_additive()

    # ========================================================================
    # Compilation and Code Generation
    # ========================================================================

    def compile_diffusion(
        self,
        backends: Optional[List[Backend]] = None,
        verbose: bool = False,
        **kwargs,
    ) -> Dict[Backend, float]:
        """Pre-compile diffusion functions for specified backends."""
        return self.diffusion_handler.compile_all(backends=backends, verbose=verbose, **kwargs)

    def compile_all(
        self,
        backends: Optional[List[Backend]] = None,
        verbose: bool = False,
        **kwargs,
    ) -> Dict[Backend, Dict[str, float]]:
        """
        Compile both drift and diffusion for all backends.

        Returns
        -------
        Dict[str, Dict[str, float]]
            Nested dict: backend → {'drift': time, 'diffusion': time}
        """
        if backends is None:
            backends = ["numpy", "torch", "jax"]

        results = {}

        for backend in backends:
            if verbose:
                print(f"\nCompiling {backend} backend...")

            # Compile drift (via parent)
            drift_timings = super().compile(backends=[backend], verbose=verbose, **kwargs)

            # Compile diffusion
            diffusion_timings = self.compile_diffusion(
                backends=[backend],
                verbose=verbose,
                **kwargs,
            )

            results[backend] = {
                "drift": drift_timings.get(backend),
                "diffusion": diffusion_timings.get(backend),
            }

        return results

    def reset_diffusion_cache(self, backends: Optional[List[Backend]] = None):
        """Clear cached diffusion functions."""
        self.diffusion_handler.reset_cache(backends)

    def reset_all_caches(self, backends: Optional[List[Backend]] = None):
        """Clear both drift and diffusion caches."""
        super().reset_caches(backends)  # Drift
        self.reset_diffusion_cache(backends)  # Diffusion

    # ========================================================================
    # Information and Diagnostics
    # ========================================================================

    def get_info(self) -> Dict[str, Any]:
        """Get comprehensive system information."""
        # Get base info from parent
        base_info = super().get_backend_info()

        # Get diffusion info
        diffusion_info = self.diffusion_handler.get_info()

        # Build comprehensive SDE info
        return {
            **base_info,
            "system_type": "ContinuousStochasticSystem",
            "is_stochastic": True,
            "sde_type": self.sde_type.value,
            "dimensions": {
                "nx": self.nx,
                "nu": self.nu,
                "nw": self.nw,
            },
            "diffusion": diffusion_info,
            "noise": {
                "type": self.noise_characteristics.noise_type.value,
                "is_additive": self.noise_characteristics.is_additive,
                "is_multiplicative": self.noise_characteristics.is_multiplicative,
                "is_diagonal": self.noise_characteristics.is_diagonal,
                "is_scalar": self.noise_characteristics.is_scalar,
                "depends_on": {
                    "state": self.noise_characteristics.depends_on_state,
                    "control": self.noise_characteristics.depends_on_control,
                    "time": self.noise_characteristics.depends_on_time,
                },
            },
            "recommended_solvers": {
                "jax": self.recommend_solvers("jax"),
                "torch": self.recommend_solvers("torch"),
                "julia": self.recommend_solvers("numpy"),
            },
            "optimization_opportunities": self.get_optimization_opportunities(),
        }

    def print_sde_info(self):
        """Print formatted SDE system information."""
        print("=" * 70)
        print(f"Continuous Stochastic System: {self.__class__.__name__}")
        print("=" * 70)
        print(f"Dimensions: nx={self.nx}, nu={self.nu}, nw={self.nw}")
        print(f"SDE Type: {self.sde_type.value}")
        print(f"Noise Type: {self.noise_characteristics.noise_type.value}")

        print("\nNoise Characteristics:")
        print(f"  • Additive: {self.noise_characteristics.is_additive}")
        print(f"  • Multiplicative: {self.noise_characteristics.is_multiplicative}")
        print(f"  • Diagonal: {self.noise_characteristics.is_diagonal}")
        print(f"  • Scalar: {self.noise_characteristics.is_scalar}")

        print("\nDependencies:")
        print(f"  • State: {self.noise_characteristics.depends_on_state}")
        print(f"  • Control: {self.noise_characteristics.depends_on_control}")
        print(f"  • Time: {self.noise_characteristics.depends_on_time}")

        print("\nRecommended Solvers (JAX):")
        solvers = self.recommend_solvers("jax")
        print(f"  • {', '.join(solvers)}")

        print("\nOptimization Opportunities:")
        opts = self.get_optimization_opportunities()
        print(f"  • Precompute diffusion: {opts['precompute_diffusion']}")
        print(f"  • Use diagonal solver: {opts['use_diagonal_solver']}")
        print(f"  • Vectorize easily: {opts['vectorize_easily']}")

        print("=" * 70)

    # ========================================================================
    # Override Properties for Stochastic Systems
    # ========================================================================

    @property
    def is_stochastic(self) -> bool:
        """Return True (this is a stochastic system)."""
        return True

    # ========================================================================
    # String Representations
    # ========================================================================

    def __repr__(self) -> str:
        """Detailed representation for debugging."""
        return (
            f"{self.__class__.__name__}("
            f"nx={self.nx}, nu={self.nu}, nw={self.nw}, "
            f"noise={self.noise_characteristics.noise_type.value}, "
            f"sde={self.sde_type.value}, "
            f"backend={self._default_backend})"
        )

    def __str__(self) -> str:
        """Human-readable representation."""
        return (
            f"{self.__class__.__name__}: "
            f"{self.nx} state{'s' if self.nx != 1 else ''}, "
            f"{self.nu} control{'s' if self.nu != 1 else ''}, "
            f"{self.nw} noise source{'s' if self.nw != 1 else ''} "
            f"({self.noise_characteristics.noise_type.value})"
        )


# Backward compatibility alias (will be deprecated)
StochasticDynamicalSystem = ContinuousStochasticSystem
