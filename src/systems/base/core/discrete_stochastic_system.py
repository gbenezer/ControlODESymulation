# Location: src/systems/discrete_stochastic_system.py

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
Discrete Stochastic System - Stochastic Difference Equations
=============================================================

Extends DiscreteSymbolicSystem with comprehensive stochastic support through modular composition.

This class coordinates specialized handlers to provide:
    - Deterministic dynamics (via parent DiscreteSymbolicSystem)
    - Diffusion code generation (via DiffusionHandler)
    - Automatic noise analysis (via NoiseCharacterizer)
    - SDE-specific validation (via SDEValidator)
    - Multi-backend execution (inherited from parent)

Mathematical Form
-----------------
Stochastic difference equations:

    x[k+1] = f(x[k], u[k]) + g(x[k], u[k]) * w[k]

where:
    - f(x[k], u[k]): Deterministic next state (nx × 1) - mean dynamics
    - g(x[k], u[k]): Noise gain matrix (nx × nw) - stochastic intensity
    - w[k] ~ N(0, I): IID standard normal noise (nw independent sources)
    - x[k] ∈ ℝⁿˣ: State at discrete time k
    - u[k] ∈ ℝⁿᵘ: Control input at time k
    - k ∈ ℤ: Discrete time index

Key Differences from Continuous SDEs
-------------------------------------
**Time Domain**:
- Continuous: t ∈ ℝ (real-valued time, dx = f dt + g dW)
- Discrete: k ∈ ℤ (integer time steps, x[k+1] = f + g*w)

**Noise Process**:
- Continuous: dW ~ N(0, dt) (Brownian motion, correlated increments)
- Discrete: w[k] ~ N(0, I) (IID standard normal, independent samples)

**Deterministic Part**:
- Continuous: _f_sym = dx/dt (drift, state derivative)
- Discrete: _f_sym = f(x[k], u[k]) (next state mean, no dt scaling)

**Stochastic Part**:
- Continuous: g(x,u) scales sqrt(dt) * dW
- Discrete: g(x,u) is the direct noise gain (no dt scaling)

**Itô vs Stratonovich**:
- Continuous: Different interpretations, different conversion rules
- Discrete: No distinction (both equivalent, conventionally use Itô)

Architecture
-----------
This class is THIN because it delegates to:
    - DiscreteSymbolicSystem (parent): ALL deterministic dynamics logic
    - DiffusionHandler: Diffusion code generation and caching
    - NoiseCharacterizer: Automatic noise structure analysis (via DiffusionHandler)
    - SDEValidator: Stochastic-specific validation
    - BackendManager: Type conversions (inherited)

Noise Types Detected Automatically
---------------------------------
- ADDITIVE: g(x,u) = constant (most efficient)
- MULTIPLICATIVE: g(x,u) depends on state
- DIAGONAL: Independent noise sources
- SCALAR: Single noise source (nw = 1)
- GENERAL: Full coupling

The framework automatically identifies noise structure for optimization.

Usage Pattern
-------------
Users subclass DiscreteStochasticSystem and implement define_system() to
specify both deterministic and stochastic parts:

    class DiscreteOU(DiscreteStochasticSystem):
        '''Discrete-time Ornstein-Uhlenbeck (AR(1) process).'''

        def define_system(self, alpha=1.0, sigma=0.5, dt=0.1):
            # Define symbolic variables
            x = sp.symbols('x', real=True)
            u = sp.symbols('u', real=True)

            # Define symbolic parameters
            alpha_sym = sp.symbols('alpha', positive=True)
            sigma_sym = sp.symbols('sigma', positive=True)
            dt_sym = sp.symbols('dt', positive=True)

            # Define deterministic part (Euler discretization of OU)
            self.state_vars = [x]
            self.control_vars = [u]
            self._f_sym = sp.Matrix([(1 - alpha_sym*dt_sym) * x + u])
            self.parameters = {alpha_sym: alpha, sigma_sym: sigma, dt_sym: dt}
            self._dt = dt  # REQUIRED for discrete systems!
            self.order = 1

            # Define stochastic part (additive noise)
            self.diffusion_expr = sp.Matrix([[sigma_sym]])
            self.sde_type = 'ito'  # Convention for discrete

    system = DiscreteOU(alpha=2.0, sigma=0.3, dt=0.1)

This pattern is consistent with DiscreteSymbolicSystem and ContinuousStochasticSystem.

Examples
--------
>>> # Discrete-time Ornstein-Uhlenbeck (additive noise)
>>> class DiscreteOU(DiscreteStochasticSystem):
...     def define_system(self, alpha=1.0, sigma=0.5, dt=0.1):
...         x = sp.symbols('x')
...         u = sp.symbols('u')
...         alpha_sym = sp.symbols('alpha', positive=True)
...         sigma_sym = sp.symbols('sigma', positive=True)
...         dt_sym = sp.symbols('dt', positive=True)
...
...         self.state_vars = [x]
...         self.control_vars = [u]
...         self._f_sym = sp.Matrix([(1 - alpha_sym*dt_sym) * x + u])
...         self.parameters = {alpha_sym: alpha, sigma_sym: sigma, dt_sym: dt}
...         self._dt = dt
...         self.order = 1
...
...         self.diffusion_expr = sp.Matrix([[sigma_sym]])
...         self.sde_type = 'ito'
>>>
>>> system = DiscreteOU(alpha=2.0, sigma=0.3, dt=0.1)
>>> system.is_additive_noise()
True
>>>
>>> # Geometric random walk (multiplicative noise)
>>> class GeometricRandomWalk(DiscreteStochasticSystem):
...     def define_system(self, mu=0.1, sigma=0.2, dt=1.0):
...         x = sp.symbols('x', positive=True)
...         u = sp.symbols('u')
...         mu_sym, sigma_sym = sp.symbols('mu sigma', positive=True)
...
...         self.state_vars = [x]
...         self.control_vars = [u]
...         self._f_sym = sp.Matrix([(1 + mu_sym) * x + u])
...         self.parameters = {mu_sym: mu, sigma_sym: sigma}
...         self._dt = dt
...         self.order = 1
...
...         self.diffusion_expr = sp.Matrix([[sigma_sym * x]])
...         self.sde_type = 'ito'
>>>
>>> grw = GeometricRandomWalk(mu=0.15, sigma=0.25)
>>> grw.is_multiplicative_noise()
True

See Also
--------
- DiscreteSymbolicSystem: Parent class for deterministic dynamics
- ContinuousStochasticSystem: Continuous-time analog
- DiffusionHandler: Diffusion code generation
- NoiseCharacterizer: Automatic noise analysis
- SDEValidator: Stochastic system validation

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

from src.systems.base.core.discrete_symbolic_system import DiscreteSymbolicSystem
from src.systems.base.utils.stochastic.diffusion_handler import DiffusionHandler
from src.systems.base.utils.stochastic.noise_analysis import (
    NoiseCharacteristics,
    NoiseType,
    SDEType,
)
from src.systems.base.utils.stochastic.sde_validator import SDEValidator, ValidationError
from src.types.backends import Backend

# Type imports
from src.types.core import ControlVector, DiscreteControlInput, NoiseVector, StateVector
from src.types.trajectories import DiscreteSimulationResult


class DiscreteStochasticSystem(DiscreteSymbolicSystem):
    """
    Concrete symbolic discrete-time stochastic dynamical system.

    Extends DiscreteSymbolicSystem to handle stochastic difference equations.
    Users subclass this and implement define_system() to specify both deterministic
    and stochastic terms.

    Represents systems of the form:
        x[k+1] = f(x[k], u[k]) + g(x[k], u[k]) * w[k]
        y[k] = h(x[k])

    where:
        f: Deterministic next state (inherited from parent)
        g: Noise gain matrix (diffusion)
        w[k]: IID standard normal noise ~ N(0, I)

    The noise enters additively through the gain matrix g, which can be:
    - Constant (additive noise): g(x, u) = G
    - State-dependent (multiplicative): g(x, u) = G(x, u)
    - Control-dependent: g(x, u) = G(x, u)

    Users must define both dynamics (_f_sym) and diffusion (diffusion_expr)
    in define_system().

    Attributes (Set by User in define_system)
    -----------------------------------------
    diffusion_expr : sp.Matrix
        Symbolic diffusion matrix g(x, u), shape (nx, nw)
        REQUIRED - must be set in define_system()
    sde_type : SDEType or str
        SDE interpretation ('ito' or 'stratonovich')
        Optional - defaults to Itô (convention for discrete time)
        Note: In discrete time, both interpretations are equivalent

    Attributes (Created Automatically)
    ----------------------------------
    diffusion_handler : DiffusionHandler
        Generates and caches diffusion functions
    noise_characteristics : NoiseCharacteristics
        Automatic noise structure analysis results
    nw : int
        Number of independent noise sources
    is_stochastic : bool
        Always True for this class

    Examples
    --------
    Discrete-time Ornstein-Uhlenbeck process:

    >>> class DiscreteOU(DiscreteStochasticSystem):
    ...     '''AR(1) process with mean reversion.'''
    ...
    ...     def define_system(self, alpha=1.0, sigma=0.5, dt=0.1):
    ...         x = sp.symbols('x')
    ...         u = sp.symbols('u')
    ...         alpha_sym = sp.symbols('alpha', positive=True)
    ...         sigma_sym = sp.symbols('sigma', positive=True)
    ...         dt_sym = sp.symbols('dt', positive=True)
    ...
    ...         # Deterministic part (Euler discretization)
    ...         self.state_vars = [x]
    ...         self.control_vars = [u]
    ...         self._f_sym = sp.Matrix([(1 - alpha_sym*dt_sym) * x + u])
    ...         self.parameters = {alpha_sym: alpha, sigma_sym: sigma, dt_sym: dt}
    ...         self._dt = dt  # REQUIRED!
    ...         self.order = 1
    ...
    ...         # Stochastic part (additive noise)
    ...         self.diffusion_expr = sp.Matrix([[sigma_sym]])
    ...         self.sde_type = 'ito'
    >>>
    >>> # Instantiate system
    >>> system = DiscreteOU(alpha=2.0, sigma=0.3, dt=0.1)
    >>>
    >>> # Automatic noise analysis
    >>> print(system.noise_characteristics.noise_type)
    NoiseType.ADDITIVE
    >>>
    >>> # Evaluate deterministic and stochastic parts
    >>> x_k = np.array([1.0])
    >>> u_k = np.array([0.0])
    >>> f = system(x_k, u_k)  # Deterministic next state mean
    >>> g = system.diffusion(x_k, u_k)  # Noise gain
    >>>
    >>> # Full stochastic step
    >>> w_k = np.random.randn(1)
    >>> x_next = f + g @ w_k
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize discrete-time stochastic system.

        Follows the template method pattern:
        1. Initialize stochastic-specific containers
        2. Call parent __init__ (which calls define_system and validates)
        3. Validate stochastic-specific attributes
        4. Initialize stochastic-specific components

        Parameters
        ----------
        *args : tuple
            Positional arguments passed to define_system()
        **kwargs : dict
            Keyword arguments passed to define_system()

        Raises
        ------
        ValidationError
            If stochastic system definition is invalid
        ValueError
            If diffusion_expr is not set in define_system()

        Notes
        -----
        Subclasses must implement define_system() and set:
        - All parent class attributes (state_vars, control_vars, _f_sym, _dt, etc.)
        - self.diffusion_expr (required)
        - self.sde_type (optional, defaults to Itô)

        Examples
        --------
        >>> class MyStochasticSystem(DiscreteStochasticSystem):
        ...     def define_system(self, a=0.9, sigma=0.1, dt=0.1):
        ...         # Define deterministic part
        ...         x = sp.symbols('x')
        ...         self.state_vars = [x]
        ...         self.control_vars = []
        ...         self._f_sym = sp.Matrix([a * x])
        ...         self.parameters = {sp.symbols('a'): a, sp.symbols('sigma'): sigma}
        ...         self._dt = dt
        ...         self.order = 1
        ...
        ...         # Define stochastic part
        ...         self.diffusion_expr = sp.Matrix([[sigma]])
        ...         self.sde_type = 'ito'
        >>>
        >>> system = MyStochasticSystem(a=0.95, sigma=0.2, dt=0.1)
        """

        # ====================================================================
        # Stochastic-Specific Containers (Before Parent Init)
        # ====================================================================

        self.diffusion_expr: Optional[sp.Matrix] = None
        """Symbolic diffusion matrix g(x, u) - MUST be set in define_system()"""

        self.sde_type: Union[str, SDEType] = "ito"
        """SDE interpretation - 'ito' or 'stratonovich' (equivalent in discrete time)"""

        # Placeholders for components (created after parent init)
        self.diffusion_handler: Optional[DiffusionHandler] = None
        self.noise_characteristics: Optional[NoiseCharacteristics] = None
        self.nw: int = 0

        # ====================================================================
        # Parent Class Initialization
        # ====================================================================

        # CRITICAL: Call parent __init__ which handles all deterministic logic
        # This calls define_system(), validates deterministic part, initializes evaluators
        super().__init__(*args, **kwargs)

        # ====================================================================
        # Stochastic-Specific Validation and Initialization
        # ====================================================================

        # Validate that diffusion_expr was set by user
        if self.diffusion_expr is None:
            raise ValueError(
                f"{self.__class__.__name__}.define_system() must set self.diffusion_expr.\n"
                "Example:\n"
                "  def define_system(self, dt=0.1):\n"
                "      # ... set deterministic attributes and self._dt ...\n"
                "      self.diffusion_expr = sp.Matrix([[sigma]])",
            )

        # Run stochastic-specific validation
        self._validate_stochastic_system()

        # Initialize stochastic-specific components
        self._initialize_stochastic_components()

    def _validate_stochastic_system(self):
        """
        Validate stochastic-specific constraints.

        Uses SDEValidator to check:
        - Deterministic-diffusion dimension compatibility
        - Symbol resolution in diffusion
        - Noise type claims (if provided)

        Raises
        ------
        ValidationError
            If stochastic validation fails
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
                f"Stochastic validation failed for {self.__class__.__name__}:\n{e!s}",
            ) from e

    def _initialize_stochastic_components(self):
        """
        Initialize stochastic-specific components after validation.

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
    # Primary Interface - Deterministic and Stochastic Evaluation
    # ========================================================================

    def __call__(
        self,
        x: StateVector,
        u: Optional[ControlVector] = None,
        k: int = 0,
        backend: Optional[Backend] = None,
    ) -> StateVector:
        """
        Evaluate deterministic part: f(x[k], u[k]).

        Returns the mean next state E[x[k+1] | x[k], u[k]] = f(x[k], u[k]).
        For the full stochastic step, use step_stochastic().

        Parameters
        ----------
        x : StateVector
            Current state x[k] (nx,) or batched (batch, nx)
        u : Optional[ControlVector]
            Control input u[k] (nu,) or batched (batch, nu)
            None for autonomous systems
        k : int
            Time step index (currently ignored for time-invariant systems)
        backend : Optional[Backend]
            Backend selection (None = auto-detect)

        Returns
        -------
        StateVector
            Deterministic next state f(x[k], u[k]), same shape and backend as input

        Notes
        -----
        This evaluates ONLY the deterministic part for consistency with parent class.
        Use system.diffusion(x, u) for the noise gain matrix.
        Use system.step_stochastic(x, u, w) for the full stochastic update.

        Examples
        --------
        >>> # Mean next state (no noise)
        >>> x_k = np.array([1.0])
        >>> u_k = np.array([0.0])
        >>> f = system(x_k, u_k)
        >>>
        >>> # Full stochastic step
        >>> g = system.diffusion(x_k, u_k)
        >>> w_k = np.random.randn(system.nw)
        >>> x_next = f + g @ w_k
        """
        # Call parent's step method (NOT __call__)
        return self.step(x, u, k, backend)

    def diffusion(
        self,
        x: StateVector,
        u: Optional[ControlVector] = None,
        k: int = 0,
        backend: Optional[Backend] = None,
    ):
        """
        Evaluate diffusion term g(x[k], u[k]).

        Parameters
        ----------
        x : StateVector
            State vector (nx,) or batched (batch, nx)
        u : Optional[ControlVector]
            Control vector (nu,) or batched (batch, nu)
            None for autonomous systems
        k : int
            Time step (currently ignored for time-invariant systems)
        backend : Optional[Backend]
            Backend selection (None = auto-detect)

        Returns
        -------
        ArrayLike
            Diffusion matrix g(x, u), shape (nx, nw) or (batch, nx, nw)

        Examples
        --------
        Controlled system:
        >>> g = system.diffusion(np.array([2.0]), np.array([0.0]))
        >>> print(g.shape)
        (1, 1)

        Autonomous system:
        >>> g = system.diffusion(np.array([2.0]))  # u=None
        >>> print(g.shape)
        (1, 1)

        For additive noise (precompute once):
        >>> if system.is_additive_noise():
        ...     G = system.get_constant_noise()  # Precompute once
        ...     # Use G directly in simulation - huge speedup!
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

    def step_stochastic(
        self,
        x: StateVector,
        u: Optional[ControlVector] = None,
        w: Optional[NoiseVector] = None,
        k: int = 0,
        backend: Optional[Backend] = None,
    ) -> StateVector:
        """
        Compute full stochastic step: x[k+1] = f(x[k], u[k]) + g(x[k], u[k]) * w[k].

        This is the primary method for stochastic simulation, computing the complete
        state update including both deterministic and stochastic components.

        Parameters
        ----------
        x : StateVector
            Current state (nx,) or batched (batch, nx)
        u : Optional[ControlVector]
            Control (nu,) or batched (batch, nu), None for autonomous
        w : Optional[ArrayLike]
            Standard normal noise (nw,) or batched (batch, nw)
            If None, generated automatically
        k : int
            Time step (currently ignored for time-invariant systems)
        backend : Optional[Backend]
            Backend selection (None = auto-detect)

        Returns
        -------
        StateVector
            Next state x[k+1], same shape and backend as input

        Examples
        --------
        Automatic noise generation:
        >>> x_next = system.step_stochastic(x_k, u_k)

        Custom noise (for reproducibility):
        >>> w_k = np.random.randn(system.nw)
        >>> x_next = system.step_stochastic(x_k, u_k, w=w_k)

        Deterministic (w=0):
        >>> x_next = system.step_stochastic(x_k, u_k, w=np.zeros(system.nw))

        Batched inputs:
        >>> x_batch = np.random.randn(100, 2)  # 100 states
        >>> u_batch = np.random.randn(100, 1)  # 100 controls
        >>> w_batch = np.random.randn(100, 1)  # 100 noise samples
        >>> x_next_batch = system.step_stochastic(x_batch, u_batch, w_batch)

        Notes
        -----
        For batched inputs:
            x: (batch, nx)
            u: (batch, nu) or None
            w: (batch, nw) or None
            → x_next: (batch, nx)

        If w is None, noise is generated per sample in batch.

        With DiffusionHandler batching support:
        - Additive noise: g returns (nx, nw) - constant
        - Multiplicative noise: g returns (batch, nx, nw) - state-dependent

        The function automatically handles both cases efficiently.
        """
        backend = backend or self._default_backend

        # Deterministic part: f(x[k], u[k])
        f = self(x, u, k, backend)

        # Stochastic part: g(x[k], u[k])
        g = self.diffusion(x, u, k, backend)

        # Generate noise if not provided
        if w is None:
            # Detect if batched
            if self._is_batched(x):
                batch_size = x.shape[0]
                noise_shape = (batch_size, self.nw)
            else:
                noise_shape = (self.nw,)

            w = self._generate_noise(noise_shape, backend)

        # Full stochastic step: x[k+1] = f + g*w
        # Handle different diffusion shapes based on noise type and batching

        is_batched = self._is_batched(x)
        g_ndim = self._get_ndim(g)

        if is_batched:
            # Batched input
            if g_ndim == 3:
                # Multiplicative: g is (batch, nx, nw), w is (batch, nw)
                if backend == "numpy":
                    stochastic_term = np.einsum("ijk,ik->ij", g, w)
                elif backend == "torch":
                    import torch

                    stochastic_term = torch.bmm(g, w.unsqueeze(-1)).squeeze(-1)
                elif backend == "jax":
                    import jax.numpy as jnp

                    stochastic_term = jnp.einsum("ijk,ik->ij", g, w)

            elif g_ndim == 2:
                # Additive: g is (nx, nw) constant, w is (batch, nw)
                # Broadcast: (nx, nw) @ (batch, nw).T → (nx, batch) → T → (batch, nx)
                if backend == "numpy" or backend == "torch" or backend == "jax":
                    stochastic_term = (g @ w.T).T

            else:
                raise ValueError(
                    f"Unexpected diffusion shape for batched input: g.shape={g.shape}. "
                    f"Expected (batch, nx, nw) for multiplicative or (nx, nw) for additive.",
                )

        # Single trajectory: g is (nx, nw), w is (nw,)
        elif backend == "numpy" or backend == "torch" or backend == "jax":
            stochastic_term = g @ w

        return f + stochastic_term

    def simulate_stochastic(
        self,
        x0: StateVector,
        u_sequence: Optional[Union[ControlVector, DiscreteControlInput]] = None,
        n_steps: int = 100,
        n_paths: int = 1,
        seed: Optional[int] = None,
        **kwargs,
    ) -> DiscreteSimulationResult:
        """
        Simulate stochastic discrete system with optional Monte Carlo.

        Performs either single-path or Monte Carlo simulation of the stochastic
        difference equation.

        Parameters
        ----------
        x0 : StateVector
            Initial state (nx,)
        u_sequence : Optional[Union[ControlVector, DiscreteControlInput]]
            Control sequence (same format as parent simulate())
        n_steps : int
            Number of simulation steps
        n_paths : int
            Number of Monte Carlo paths (default: 1)
        seed : Optional[int]
            Random seed for reproducibility
        **kwargs
            Additional simulation options

        Returns
        -------
        DiscreteSimulationResult
            TypedDict containing:
            - states: State trajectories
              - Single path: (n_steps+1, nx)
              - Multiple paths: (n_paths, n_steps+1, nx)
            - controls: Control sequence (n_steps, nu)
            - time_steps: [0, 1, ..., n_steps]
            - dt: Sampling period
            - metadata: Additional info including:
              - n_paths: Number of paths
              - noise_type: Detected noise type
              - seed: Random seed used

        Examples
        --------
        Single trajectory:
        >>> result = system.simulate_stochastic(
        ...     x0=np.array([1.0]),
        ...     u_sequence=None,
        ...     n_steps=1000,
        ...     seed=42
        ... )
        >>> plt.plot(result['time_steps'], result['states'][:, 0])

        Monte Carlo simulation:
        >>> result = system.simulate_stochastic(
        ...     x0=np.array([1.0]),
        ...     u_sequence=None,
        ...     n_steps=1000,
        ...     n_paths=100,
        ...     seed=42
        ... )
        >>> # result['states'] has shape (100, 1001, 1)
        >>> mean_traj = result['states'].mean(axis=0)
        >>> std_traj = result['states'].std(axis=0)
        >>>
        >>> plt.plot(result['time_steps'], mean_traj[:, 0], label='Mean')
        >>> plt.fill_between(
        ...     result['time_steps'],
        ...     mean_traj[:, 0] - std_traj[:, 0],
        ...     mean_traj[:, 0] + std_traj[:, 0],
        ...     alpha=0.3
        ... )

        State feedback with stochastic dynamics:
        >>> def policy(x, k):
        ...     return -0.5 * x
        >>> result = system.simulate_stochastic(
        ...     x0=np.array([1.0]),
        ...     u_sequence=policy,
        ...     n_steps=1000
        ... )
        """
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)

        # Prepare control function
        u_func = self._prepare_control_sequence(u_sequence, n_steps)

        if n_paths == 1:
            # Single trajectory
            states = np.zeros((n_steps + 1, self.nx))
            states[0, :] = x0
            controls = []

            x = x0
            for k in range(n_steps):
                u = u_func(x, k)
                controls.append(u)

                # Generate noise
                w_k = np.random.randn(self.nw)

                # Stochastic step
                x = self.step_stochastic(x, u, w_k, k)
                states[k + 1, :] = x

            controls_array = np.array(controls) if controls and controls[0] is not None else None

            return {
                "states": states,  # (n_steps+1, nx)
                "controls": controls_array,  # (n_steps, nu)
                "time_steps": np.arange(n_steps + 1),
                "dt": self.dt,
                "success": True,
                "metadata": {
                    "method": "discrete_stochastic_step",
                    "n_paths": 1,
                    "noise_type": self.noise_characteristics.noise_type.value,
                    "seed": seed,
                },
            }

        # Monte Carlo simulation
        all_paths = np.zeros((n_paths, n_steps + 1, self.nx))

        for path_idx in range(n_paths):
            states = np.zeros((n_steps + 1, self.nx))
            states[0, :] = x0

            x = x0
            for k in range(n_steps):
                u = u_func(x, k)

                # Generate noise
                w_k = np.random.randn(self.nw)

                # Stochastic step
                x = self.step_stochastic(x, u, w_k, k)
                states[k + 1, :] = x

            all_paths[path_idx, :, :] = states

        return {
            "states": all_paths,  # (n_paths, n_steps+1, nx)
            "controls": None,  # Reconstructing for all paths is complex
            "time_steps": np.arange(n_steps + 1),
            "dt": self.dt,
            "success": True,
            "metadata": {
                "method": "discrete_stochastic_monte_carlo",
                "n_paths": n_paths,
                "noise_type": self.noise_characteristics.noise_type.value,
                "seed": seed,
            },
        }

    # ========================================================================
    # Override Linearization to Include Diffusion Matrix
    # ========================================================================

    def linearize(self, x_eq: StateVector, u_eq: Optional[ControlVector] = None):
        """
        Compute linearization including diffusion: Ad = ∂f/∂x, Bd = ∂f/∂u, Gd = g(x_eq).

        For stochastic systems, linearization returns three matrices:
        - Ad: State Jacobian ∂f/∂x
        - Bd: Control Jacobian ∂f/∂u
        - Gd: Diffusion matrix evaluated at equilibrium g(x_eq, u_eq)

        Parameters
        ----------
        x_eq : StateVector
            Equilibrium state (nx,)
        u_eq : Optional[ControlVector]
            Equilibrium control (nu,)

        Returns
        -------
        Tuple[ArrayLike, ArrayLike, ArrayLike]
            (Ad, Bd, Gd) where:
            - Ad: State Jacobian (nx, nx)
            - Bd: Control Jacobian (nx, nu)
            - Gd: Diffusion matrix (nx, nw)

        Examples
        --------
        >>> x_eq = np.zeros(2)
        >>> u_eq = np.zeros(1)
        >>> Ad, Bd, Gd = system.linearize(x_eq, u_eq)
        >>>
        >>> # Check discrete stability: |λ| < 1
        >>> eigenvalues = np.linalg.eigvals(Ad)
        >>> is_stable = np.all(np.abs(eigenvalues) < 1)
        >>>
        >>> # Stochastic covariance propagation
        >>> # P[k+1] = Ad @ P[k] @ Ad.T + Gd @ Gd.T
        """
        # Get deterministic linearization from parent
        Ad, Bd = super().linearize(x_eq, u_eq)

        # Evaluate diffusion at equilibrium
        Gd = self.diffusion(x_eq, u_eq)

        return (Ad, Bd, Gd)

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

    def get_noise_type(self) -> NoiseType:
        """Get classified noise type."""
        return self.noise_characteristics.noise_type

    def get_sde_type(self) -> SDEType:
        """Get SDE interpretation type (Itô convention for discrete)."""
        return self.sde_type

    def depends_on_state(self) -> bool:
        """Check if diffusion depends on state variables."""
        return self.noise_characteristics.depends_on_state

    def depends_on_control(self) -> bool:
        """Check if diffusion depends on control inputs."""
        return self.noise_characteristics.depends_on_control

    def depends_on_time(self) -> bool:
        """Check if diffusion depends on time (always False for time-invariant)."""
        return self.noise_characteristics.depends_on_time

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
        backend : Backend
            Backend for array type

        Returns
        -------
        ArrayLike
            Constant diffusion matrix (nx, nw)

        Raises
        ------
        ValueError
            If noise is not additive

        Examples
        --------
        >>> if system.is_additive_noise():
        ...     G = system.get_constant_noise('numpy')
        ...     print(G)
        ...     [[0.3]]
        ...
        ...     # Simulation loop with precomputed noise
        ...     for k in range(1000):
        ...         w_k = np.random.randn(nw)
        ...         x_next = system(x, u) + G @ w_k
        ...         x = x_next
        """
        return self.diffusion_handler.get_constant_noise(backend)

    def can_optimize_for_additive(self) -> bool:
        """Check if additive-noise optimizations are applicable."""
        return self.diffusion_handler.can_optimize_for_additive()

    # ========================================================================
    # Compilation and Code Generation
    # ========================================================================

    def compile_diffusion(
        self, backends: Optional[List[Backend]] = None, verbose: bool = False, **kwargs,
    ) -> Dict[Backend, float]:
        """Pre-compile diffusion functions for specified backends."""
        return self.diffusion_handler.compile_all(backends=backends, verbose=verbose, **kwargs)

    def compile_all(
        self, backends: Optional[List[Backend]] = None, verbose: bool = False, **kwargs,
    ) -> Dict[Backend, Dict[str, float]]:
        """
        Compile both deterministic and diffusion for all backends.

        Returns
        -------
        Dict[Backend, Dict[str, float]]
            Nested dict: backend → {'deterministic': time, 'diffusion': time}
        """
        if backends is None:
            backends = ["numpy", "torch", "jax"]

        results = {}

        for backend in backends:
            if verbose:
                print(f"\nCompiling {backend} backend...")

            # Compile deterministic (via parent)
            det_timings = super().compile(backends=[backend], verbose=verbose, **kwargs)

            # Compile diffusion
            diffusion_timings = self.compile_diffusion(
                backends=[backend], verbose=verbose, **kwargs,
            )

            results[backend] = {
                "deterministic": det_timings.get(backend),
                "diffusion": diffusion_timings.get(backend),
            }

        return results

    def reset_diffusion_cache(self, backends: Optional[List[Backend]] = None):
        """Clear cached diffusion functions."""
        self.diffusion_handler.reset_cache(backends)

    def reset_all_caches(self, backends: Optional[List[Backend]] = None):
        """Clear both deterministic and diffusion caches."""
        super().reset_caches(backends)  # Deterministic
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

        # Build comprehensive stochastic info
        return {
            **base_info,
            "system_type": "DiscreteStochasticSystem",
            "is_discrete": True,
            "is_stochastic": True,
            "sde_type": self.sde_type.value,
            "dimensions": {
                "nx": self.nx,
                "nu": self.nu,
                "nw": self.nw,
                "dt": self.dt,
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
            "optimization_opportunities": self.get_optimization_opportunities(),
        }

    def get_optimization_opportunities(self) -> Dict[str, bool]:
        """Get optimization opportunities based on noise structure."""
        return self.diffusion_handler.get_optimization_opportunities()

    def print_stochastic_info(self):
        """Print formatted stochastic system information."""
        print("=" * 70)
        print(f"Discrete Stochastic System: {self.__class__.__name__}")
        print("=" * 70)
        print(f"Dimensions: nx={self.nx}, nu={self.nu}, nw={self.nw}, dt={self.dt}")
        print(f"SDE Type: {self.sde_type.value} (convention)")
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

        print("\nOptimization Opportunities:")
        opts = self.get_optimization_opportunities()
        print(f"  • Precompute diffusion: {opts['precompute_diffusion']}")
        print(f"  • Use diagonal solver: {opts['use_diagonal_solver']}")
        print(f"  • Vectorize easily: {opts['vectorize_easily']}")

        print("=" * 70)

    def print_equations(self, simplify: bool = True):
        """
        Print symbolic equations using discrete-time stochastic notation.

        Overrides parent to show both deterministic and stochastic parts.

        Parameters
        ----------
        simplify : bool
            If True, simplify expressions before printing

        Examples
        --------
        >>> system.print_equations()
        ======================================================================
        DiscreteOU (Discrete-Time Stochastic, dt=0.1)
        ======================================================================
        State Variables: [x]
        Control Variables: [u]
        Dimensions: nx=1, nu=1, nw=1
        Noise Type: additive

        Deterministic Part: f(x[k], u[k])
          f_0 = 0.9*x + u

        Stochastic Part: g(x[k], u[k])
          g_0 = [0.3]

        Full Dynamics:
          x[k+1] = f(x[k], u[k]) + g(x[k], u[k]) * w[k]
          where w[k] ~ N(0, I)
        ======================================================================
        """
        print("=" * 70)
        print(f"{self.__class__.__name__} (Discrete-Time Stochastic, dt={self.dt})")
        print("=" * 70)
        print(f"State Variables: {self.state_vars}")
        print(f"Control Variables: {self.control_vars}")
        print(f"System Order: {self.order}")
        print(f"Dimensions: nx={self.nx}, nu={self.nu}, nw={self.nw}")
        print(f"Noise Type: {self.noise_characteristics.noise_type.value}")

        print("\nDeterministic Part: f(x[k], u[k])")
        for i, (var, expr) in enumerate(zip(self.state_vars, self._f_sym)):
            expr_sub = self.substitute_parameters(expr)
            if simplify:
                expr_sub = sp.simplify(expr_sub)
            print(f"  f_{i} = {expr_sub}")

        print("\nStochastic Part: g(x[k], u[k])")
        for i in range(self.nx):
            row_exprs = []
            for j in range(self.nw):
                expr = self.diffusion_expr[i, j]
                expr_sub = self.substitute_parameters(expr)
                if simplify:
                    expr_sub = sp.simplify(expr_sub)
                row_exprs.append(str(expr_sub))
            print(f"  g_{i} = [{', '.join(row_exprs)}]")

        print("\nFull Dynamics:")
        print("  x[k+1] = f(x[k], u[k]) + g(x[k], u[k]) * w[k]")
        print("  where w[k] ~ N(0, I) are IID standard normal")

        if self._h_sym is not None:
            print("\nOutput: y[k] = h(x[k])")
            for i, expr in enumerate(self._h_sym):
                expr_sub = self.substitute_parameters(expr)
                if simplify:
                    expr_sub = sp.simplify(expr_sub)
                print(f"  y[{i}] = {expr_sub}")

        print("=" * 70)

    # ========================================================================
    # Override Properties for Stochastic Systems
    # ========================================================================

    @property
    def is_stochastic(self) -> bool:
        """Return True (this is a stochastic system)."""
        return True

    # ========================================================================
    # Utility Methods (Internal)
    # ========================================================================

    def _is_batched(self, x: Any) -> bool:
        """Check if input is batched (2D)."""
        if isinstance(x, np.ndarray) or hasattr(x, "ndim"):
            return x.ndim > 1
        if hasattr(x, "shape"):
            return len(x.shape) > 1
        return False

    def _get_ndim(self, arr) -> int:
        """Get number of dimensions of array."""
        if isinstance(arr, np.ndarray) or hasattr(arr, "ndim"):
            return arr.ndim
        if hasattr(arr, "shape"):
            return len(arr.shape)
        return 0

    def _generate_noise(self, shape, backend: Backend):
        """Generate standard normal noise in specified backend."""
        if backend == "numpy":
            return np.random.randn(*shape)
        if backend == "torch":
            import torch

            return torch.randn(*shape)
        if backend == "jax":
            import jax

            # JAX needs explicit key management
            # For simplicity, use random key (users should set seed externally)
            key = jax.random.PRNGKey(np.random.randint(0, 2**31))
            return jax.random.normal(key, shape)
        raise ValueError(f"Unknown backend: {backend}")

    # ========================================================================
    # String Representations
    # ========================================================================

    def __repr__(self) -> str:
        """Detailed representation for debugging."""
        return (
            f"{self.__class__.__name__}("
            f"nx={self.nx}, nu={self.nu}, nw={self.nw}, dt={self.dt}, "
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
            f"{self.nw} noise source{'s' if self.nw != 1 else ''}, "
            f"dt={self.dt} "
            f"({self.noise_characteristics.noise_type.value})"
        )
