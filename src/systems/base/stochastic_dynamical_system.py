"""
Stochastic Dynamical System - SDE Framework

Extends SymbolicDynamicalSystem with comprehensive SDE support through modular composition.

This class coordinates specialized handlers to provide:
    - Drift dynamics (via parent SymbolicDynamicalSystem)
    - Diffusion code generation (via DiffusionHandler)
    - Automatic noise analysis (via NoiseCharacterizer)
    - SDE-specific validation (via SDEValidator)
    - Multi-backend execution (inherited from parent)

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
    - SymbolicDynamicalSystem (parent): ALL drift dynamics logic
    - DiffusionHandler: Diffusion code generation and caching
    - NoiseCharacterizer: Automatic noise structure analysis
    - SDEValidator: SDE-specific validation
    - BackendManager: Type conversions (inherited)

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
Users subclass StochasticDynamicalSystem and implement define_system() to
specify both drift and diffusion:

    class OrnsteinUhlenbeck(StochasticDynamicalSystem):
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

This pattern is consistent with SymbolicDynamicalSystem!

Examples
--------
>>> # Ornstein-Uhlenbeck process (additive noise)
>>> class OrnsteinUhlenbeck(StochasticDynamicalSystem):
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
>>> class GeometricBrownianMotion(StochasticDynamicalSystem):
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
"""

from typing import List, Optional, Dict, Any, Union
import sympy as sp
import numpy as np

from src.systems.base.symbolic_dynamical_system import SymbolicDynamicalSystem
from src.systems.base.utils.stochastic.diffusion_handler import DiffusionHandler
from src.systems.base.utils.stochastic.noise_analysis import (
    NoiseCharacteristics,
    SDEType,
    NoiseType,
)
from src.systems.base.utils.stochastic.sde_validator import SDEValidator, ValidationError


class StochasticDynamicalSystem(SymbolicDynamicalSystem):
    """
    Abstract base class for stochastic dynamical systems (SDEs).
    
    Extends SymbolicDynamicalSystem to handle stochastic differential equations.
    Users subclass this and implement define_system() to specify both drift
    and diffusion terms.
    
    Attributes (Set by User in define_system)
    -----------------------------------------
    diffusion_expr : sp.Matrix
        Symbolic diffusion matrix g(x, u), shape (nx, nw)
        REQUIRED - must be set in define_system()
    sde_type : SDEType
        SDE interpretation (ITO or STRATONOVICH)
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
    >>> class OrnsteinUhlenbeck(StochasticDynamicalSystem):
    ...     '''Ornstein-Uhlenbeck process with mean reversion.'''
    ...     
    ...     def define_system(self, alpha=1.0, sigma=0.5):
    ...         x = sp.symbols('x')
    ...         u = sp.symbols('u')
    ...         alpha_sym = sp.symbols('alpha', positive=True)
    ...         
    ...         # Drift (deterministic part)
    ...         self.state_vars = [x]
    ...         self.control_vars = [u]
    ...         self._f_sym = sp.Matrix([[-alpha_sym * x + u]])
    ...         self.parameters = {alpha_sym: alpha}
    ...         self.order = 1
    ...         
    ...         # Diffusion (stochastic part) - SDE-specific
    ...         self.diffusion_expr = sp.Matrix([[sigma]])
    ...         self.sde_type = SDEType.ITO
    >>> 
    >>> # Instantiate system
    >>> system = OrnsteinUhlenbeck(alpha=2.0, sigma=0.3)
    >>> 
    >>> # Automatic noise analysis
    >>> print(system.noise_characteristics.noise_type)
    NoiseType.ADDITIVE
    >>> 
    >>> # Get solver recommendations
    >>> print(system.recommend_solvers('jax'))
    ['sea', 'shark', 'sra1']
    >>> 
    >>> # Evaluate drift and diffusion
    >>> import numpy as np
    >>> x = np.array([1.0])
    >>> u = np.array([0.0])
    >>> 
    >>> f = system.drift(x, u)  # Drift term
    >>> g = system.diffusion(x, u)  # Diffusion matrix
    >>> 
    >>> # For additive noise, precompute constant matrix
    >>> if system.is_additive_noise():
    ...     G = system.get_constant_noise('numpy')
    ...     # Use G in simulation - no need to evaluate diffusion!
    """
    
    def __init__(self, *args, **kwargs):
        """
        Initialize stochastic dynamical system.
        
        This follows the same template method pattern as the parent class:
        1. Initialize SDE-specific containers
        2. Call parent __init__ (which calls define_system)
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
        
        Notes
        -----
        Subclasses must implement define_system() and set:
        - All parent class attributes (state_vars, control_vars, _f_sym, etc.)
        - self.diffusion_expr (required)
        - self.sde_type (optional, defaults to Itô)
        
        Examples
        --------
        >>> class MySystem(StochasticDynamicalSystem):
        ...     def define_system(self):
        ...         # Define drift (parent requirements)
        ...         self.state_vars = [sp.symbols('x')]
        ...         self.control_vars = [sp.symbols('u')]
        ...         self._f_sym = sp.Matrix([[-x + u]])
        ...         self.parameters = {}
        ...         self.order = 1
        ...         
        ...         # Define diffusion (SDE requirement)
        ...         self.diffusion_expr = sp.Matrix([[0.5]])
        >>> 
        >>> system = MySystem()
        """
        
        # ====================================================================
        # SDE-Specific Containers (Before Parent Init)
        # ====================================================================
        
        self.diffusion_expr: Optional[sp.Matrix] = None
        """Symbolic diffusion matrix g(x, u) - MUST be set in define_system()"""
        
        self.sde_type: Union[str, SDEType] = 'ito'
        """SDE interpretation - can be 'ito' or 'stratonovich' (string or enum)"""
        
        # Placeholders for components (created after parent init)
        self.diffusion_handler: Optional[DiffusionHandler] = None
        self.noise_characteristics: Optional[NoiseCharacteristics] = None
        self.nw: int = 0
        self.is_stochastic: bool = True
        
        # ====================================================================
        # Parent Class Initialization
        # ====================================================================
        
        # ✅ REUSE: Parent class handles drift logic and calls define_system()
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
                "      self.diffusion_expr = sp.Matrix([[sigma * x]])"
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
        # ✅ REUSE: SDEValidator for comprehensive SDE validation
        validator = SDEValidator(
            drift_expr=self._f_sym,
            diffusion_expr=self.diffusion_expr,
            state_vars=self.state_vars,
            control_vars=self.control_vars,
            time_var=getattr(self, 'time_var', None),
            parameters=self.parameters,
        )
        
        # Validate (raises on error)
        try:
            result = validator.validate(raise_on_error=True)
        except ValidationError as e:
            raise ValidationError(
                f"SDE validation failed for {self.__class__.__name__}:\n{str(e)}"
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
        # Normalize sde_type to enum (accept strings or enums for UX)
        if isinstance(self.sde_type, str):
            sde_type_lower = self.sde_type.lower()
            if sde_type_lower not in ['ito', 'stratonovich']:
                raise ValueError(
                    f"Invalid sde_type '{self.sde_type}'. "
                    f"Must be 'ito' or 'stratonovich'"
                )
            self.sde_type = SDEType(sde_type_lower)
        elif not isinstance(self.sde_type, SDEType):
            raise TypeError(
                f"sde_type must be string or SDEType enum, "
                f"got {type(self.sde_type).__name__}"
            )
        
        # ✅ COMPOSE: Create DiffusionHandler for code generation
        self.diffusion_handler = DiffusionHandler(
            diffusion_expr=self.diffusion_expr,
            state_vars=self.state_vars,
            control_vars=self.control_vars,
            time_var=getattr(self, 'time_var', None),
            parameters=self.parameters,
        )
        
        # ✅ DELEGATE: Extract noise characteristics (automatic analysis)
        self.noise_characteristics = self.diffusion_handler.characteristics
        
        # Set noise dimension
        self.nw = self.diffusion_expr.shape[1]
    
    # ========================================================================
    # Primary Interface - Drift and Diffusion Evaluation
    # ========================================================================
    
    def drift(self, x, u=None, backend: Optional[str] = None):
        """
        Evaluate drift term f(x, u) or f(x) for autonomous.
        
        Parameters
        ----------
        x : ArrayLike
            State vector (nx,) or batch (batch, nx)
        u : ArrayLike, optional
            Control vector (nu,) or batch (batch, nu)
            For autonomous systems (nu=0), u can be None
        backend : str, optional
            Backend selection (None = auto-detect)
        
        Returns
        -------
        ArrayLike
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
        # ✅ DELEGATE: Parent class handles drift evaluation (including autonomous)
        return super().__call__(x, u, backend=backend)
    
    def diffusion(self, x, u=None, backend: Optional[str] = None):
        """
        Evaluate diffusion term g(x, u) or g(x) for autonomous.
        
        Parameters
        ----------
        x : ArrayLike
            State vector (nx,) or batch (batch, nx)
        u : ArrayLike, optional
            Control vector (nu,) or batch (batch, nu)
            For autonomous systems (nu=0), u can be None
        backend : str, optional
            Backend selection (None = auto-detect)
        
        Returns
        -------
        ArrayLike
            Diffusion matrix g(x, u), shape (nx, nw) or (batch, nx, nw)
        
        Notes
        -----
        Delegates to DiffusionHandler for code generation and evaluation.
        For additive noise, consider using get_constant_noise() for efficiency.
        Supports both controlled and autonomous SDEs.
        
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
        
        >>> # For additive noise, this is constant
        >>> if system.is_additive_noise():
        ...     G = system.get_constant_noise()  # Precompute once
        """
        # Determine backend
        backend_to_use = backend if backend else self._default_backend
        
        # ✅ DELEGATE: DiffusionHandler generates and caches function
        func = self.diffusion_handler.generate_function(backend_to_use)
        
        # Handle autonomous systems - create empty control if needed
        # and also reject non-empty control
        if u is None:
            if self.nu > 0:
                raise ValueError(
                    f"Non-autonomous system requires control input u. "
                    f"System has {self.nu} control input(s)."
                )
            # Create empty control array in appropriate backend
            if backend_to_use == 'numpy':
                import numpy as np
                u = np.array([])
            elif backend_to_use == 'torch':
                import torch
                u = torch.tensor([])
            elif backend_to_use == 'jax':
                import jax.numpy as jnp
                u = jnp.array([])
        # Assess if the system is capable of taking control input
        elif u is not None:
            if self.nu == 0:
                raise ValueError(
                    f"Autonomous system cannot take control input"
                    )
        
        # Convert inputs to appropriate backend type
        # ✅ REUSE: Use parent's backend manager for conversions if needed
        if backend_to_use == 'numpy':
            import numpy as np
            x_arr = np.atleast_1d(np.asarray(x))
            u_arr = np.atleast_1d(np.asarray(u)) if self.nu > 0 else np.array([])
        elif backend_to_use == 'torch':
            import torch
            x_arr = torch.atleast_1d(torch.as_tensor(x))
            u_arr = torch.atleast_1d(torch.as_tensor(u)) if self.nu > 0 else torch.tensor([])
        elif backend_to_use == 'jax':
            import jax.numpy as jnp
            x_arr = jnp.atleast_1d(jnp.asarray(x))
            u_arr = jnp.atleast_1d(jnp.asarray(u)) if self.nu > 0 else jnp.array([])
        else:
            raise ValueError(f"Unknown backend: {backend_to_use}")
        
        # Unpack arrays for lambdified function call
        # Handle both single and batched inputs
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
        
        return result
    
    def __call__(self, x, u=None, backend: Optional[str] = None):
        """
        Evaluate drift dynamics: dx/dt = f(x, u) or dx/dt = f(x).
        
        Makes the system callable like the parent class.
        For diffusion, use system.diffusion(x, u).
        
        Parameters
        ----------
        x : ArrayLike
            State vector
        u : ArrayLike, optional
            Control vector (None for autonomous systems)
        backend : str, optional
            Backend selection
        
        Returns
        -------
        ArrayLike
            Drift term f(x, u) or f(x)
        
        Notes
        -----
        This evaluates ONLY the drift term for consistency with parent class.
        Use system.diffusion(x, u) for the diffusion term.
        Supports both controlled and autonomous SDEs.
        
        Examples
        --------
        Controlled SDE:
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
        
        Autonomous SDE:
        >>> # Drift evaluation (no control)
        >>> f = system(x)
        >>> 
        >>> # Diffusion evaluation (no control)
        >>> g = system.diffusion(x)
        >>> 
        >>> # For SDE simulation:
        >>> dt = 0.01
        >>> dW = np.random.randn(nw) * np.sqrt(dt)
        >>> dx = f * dt + g @ dW
        """
        return self.drift(x, u, backend)
    
    # ========================================================================
    # Noise Query Methods - Delegate to NoiseCharacteristics
    # ========================================================================
    
    def is_additive_noise(self) -> bool:
        """
        Check if noise is additive (constant, state-independent).
        
        Returns
        -------
        bool
            True if g(x,u,t) is constant
        
        Notes
        -----
        Additive noise enables specialized solvers and precomputation.
        
        Examples
        --------
        >>> if system.is_additive_noise():
        ...     G = system.get_constant_noise()  # Precompute once
        ...     # Use G directly in simulation - huge performance gain!
        """
        return self.noise_characteristics.is_additive
    
    def is_multiplicative_noise(self) -> bool:
        """
        Check if noise is multiplicative (state-dependent).
        
        Returns
        -------
        bool
            True if g(x,u,t) depends on state
        
        Examples
        --------
        >>> if system.is_multiplicative_noise():
        ...     # Must evaluate diffusion at each timestep
        ...     g = system.diffusion(x, u)
        """
        return self.noise_characteristics.is_multiplicative
    
    def is_diagonal_noise(self) -> bool:
        """
        Check if noise sources are independent (diagonal diffusion).
        
        Returns
        -------
        bool
            True if diffusion matrix is diagonal
        
        Notes
        -----
        Diagonal noise enables efficient element-wise solvers.
        """
        return self.noise_characteristics.is_diagonal
    
    def is_scalar_noise(self) -> bool:
        """
        Check if system has single noise source.
        
        Returns
        -------
        bool
            True if nw = 1
        """
        return self.noise_characteristics.is_scalar
    
    def get_noise_type(self) -> NoiseType:
        """
        Get classified noise type.
        
        Returns
        -------
        NoiseType
            Classified noise structure (ADDITIVE, MULTIPLICATIVE, etc.)
        
        Examples
        --------
        >>> noise_type = system.get_noise_type()
        >>> print(noise_type.value)
        'additive'
        >>> 
        >>> # Use in conditional logic
        >>> if noise_type == NoiseType.ADDITIVE:
        ...     use_specialized_solver()
        """
        return self.noise_characteristics.noise_type
    
    def depends_on_state(self) -> bool:
        """
        Check if diffusion depends on state variables.
        
        Returns
        -------
        bool
            True if any state variable appears in diffusion
        """
        return self.noise_characteristics.depends_on_state
    
    def depends_on_control(self) -> bool:
        """
        Check if diffusion depends on control inputs.
        
        Returns
        -------
        bool
            True if any control variable appears in diffusion
        """
        return self.noise_characteristics.depends_on_control
    
    def depends_on_time(self) -> bool:
        """
        Check if diffusion depends on time.
        
        Returns
        -------
        bool
            True if time variable appears in diffusion
        """
        return self.noise_characteristics.depends_on_time
    
    # ========================================================================
    # Solver Recommendations
    # ========================================================================
    
    def recommend_solvers(self, backend: str = 'jax') -> List[str]:
        """
        Recommend efficient SDE solvers based on noise structure.
        
        Automatically analyzes noise type and suggests specialized solvers
        that exploit the structure for better performance/accuracy.
        
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
        >>> # For additive noise
        >>> solvers = system.recommend_solvers('jax')
        >>> print(solvers)
        ['sea', 'shark', 'sra1']  # Specialized additive-noise solvers
        >>> 
        >>> # Use with integrator
        >>> from src.integrators.diffrax_sde import DiffraxSDEIntegrator
        >>> integrator = DiffraxSDEIntegrator(system, solver=solvers[0])
        >>> 
        >>> # For multiplicative noise
        >>> solvers = system.recommend_solvers('torch')
        >>> print(solvers)
        ['euler', 'milstein', 'srk']  # General SDE solvers
        """
        return self.noise_characteristics.recommended_solvers(backend)
    
    def get_optimization_opportunities(self) -> Dict[str, bool]:
        """
        Get optimization opportunities based on noise structure.
        
        Returns dictionary of boolean flags indicating which optimizations
        are applicable for this system's noise structure.
        
        Returns
        -------
        Dict[str, bool]
            Optimization flags:
            - 'precompute_diffusion': Can precompute g (additive only)
            - 'use_diagonal_solver': Can use diagonal-optimized solver
            - 'use_scalar_solver': Can use scalar-optimized solver
            - 'vectorize_easily': Easy to vectorize operations
            - 'cache_diffusion': Can cache diffusion values
        
        Examples
        --------
        >>> opts = system.get_optimization_opportunities()
        >>> if opts['precompute_diffusion']:
        ...     G = system.get_constant_noise()
        ...     # Use precomputed G in simulation loop
        >>> 
        >>> if opts['use_diagonal_solver']:
        ...     # Use element-wise operations
        ...     pass
        """
        return self.diffusion_handler.get_optimization_opportunities()
    
    # ========================================================================
    # Constant Noise Optimization (Additive Only)
    # ========================================================================
    
    def get_constant_noise(self, backend: str = 'numpy'):
        """
        Get constant noise matrix for additive noise.
        
        For additive noise, diffusion is constant and can be precomputed
        once for significant performance gains in simulation.
        
        Parameters
        ----------
        backend : str
            Backend for array type ('numpy', 'torch', 'jax')
        
        Returns
        -------
        np.ndarray
            Constant diffusion matrix, shape (nx, nw)
        
        Raises
        ------
        ValueError
            If noise is not additive
        
        Examples
        --------
        >>> if system.is_additive_noise():
        ...     G = system.get_constant_noise('numpy')
        ...     print(G)
        ...     [[0.5]]
        ...     
        ...     # Simulation loop with precomputed noise
        ...     dt = 0.01
        ...     for t in range(1000):
        ...         dW = np.random.randn(nw) * np.sqrt(dt)
        ...         dx = system.drift(x, u) * dt + G @ dW
        ...         x = x + dx
        """
        return self.diffusion_handler.get_constant_noise(backend)
    
    def can_optimize_for_additive(self) -> bool:
        """
        Check if additive-noise optimizations are applicable.
        
        Returns
        -------
        bool
            True if diffusion can be precomputed
        
        Examples
        --------
        >>> if system.can_optimize_for_additive():
        ...     G = system.get_constant_noise()
        ... else:
        ...     # Must evaluate diffusion at each point
        ...     g = system.diffusion(x, u)
        """
        return self.diffusion_handler.can_optimize_for_additive()
    
    # ========================================================================
    # Compilation and Code Generation
    # ========================================================================
    
    def compile_diffusion(
        self,
        backends: Optional[List[str]] = None,
        verbose: bool = False,
        **kwargs
    ) -> Dict[str, float]:
        """
        Pre-compile diffusion functions for specified backends.
        
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
        >>> timings = system.compile_diffusion(verbose=True)
        Compiling diffusion for numpy: 0.05s
        Compiling diffusion for torch: 0.12s
        Compiling diffusion for jax: 0.08s
        >>> 
        >>> print(f"NumPy: {timings['numpy']:.3f}s")
        """
        return self.diffusion_handler.compile_all(
            backends=backends,
            verbose=verbose,
            **kwargs
        )
    
    def compile_all(
        self,
        backends: Optional[List[str]] = None,
        verbose: bool = False,
        **kwargs
    ) -> Dict[str, Dict[str, float]]:
        """
        Compile both drift and diffusion for all backends.
        
        Pre-compiles all numerical functions to reduce first-call latency
        and validate that code generation works.
        
        Parameters
        ----------
        backends : List[str], optional
            Backends to compile (None = all available)
        verbose : bool
            Print compilation progress
        **kwargs
            Backend-specific compilation options
        
        Returns
        -------
        Dict[str, Dict[str, float]]
            Nested dict: backend → {'drift': time, 'diffusion': time}
        
        Examples
        --------
        >>> timings = system.compile_all(verbose=True)
        
        Compiling numpy backend...
        Compiling dynamics for numpy: 0.05s
        Compiling diffusion for numpy: 0.03s
        
        >>> print(f"NumPy drift: {timings['numpy']['drift']:.3f}s")
        >>> print(f"NumPy diffusion: {timings['numpy']['diffusion']:.3f}s")
        """
        if backends is None:
            backends = ['numpy', 'torch', 'jax']
        
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
                **kwargs
            )
            
            results[backend] = {
                'drift': drift_timings.get(backend),
                'diffusion': diffusion_timings.get(backend),
            }
        
        return results
    
    def reset_diffusion_cache(self, backends: Optional[List[str]] = None):
        """
        Clear cached diffusion functions.
        
        Parameters
        ----------
        backends : List[str], optional
            Backends to reset (None = all)
        
        Examples
        --------
        >>> # Clear only torch diffusion cache
        >>> system.reset_diffusion_cache(['torch'])
        >>> 
        >>> # Clear all diffusion caches
        >>> system.reset_diffusion_cache()
        """
        self.diffusion_handler.reset_cache(backends)
    
    def reset_all_caches(self, backends: Optional[List[str]] = None):
        """
        Clear both drift and diffusion caches.
        
        Parameters
        ----------
        backends : List[str], optional
            Backends to reset (None = all)
        
        Examples
        --------
        >>> # Free memory by clearing all cached functions
        >>> system.reset_all_caches()
        >>> 
        >>> # Clear only JAX caches
        >>> system.reset_all_caches(['jax'])
        """
        super().reset_caches(backends)  # Drift
        self.reset_diffusion_cache(backends)  # Diffusion
    
    # ========================================================================
    # Information and Diagnostics
    # ========================================================================
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get comprehensive system information.
        
        Combines information from:
        - Parent class (drift, dimensions, backend)
        - DiffusionHandler (diffusion compilation status)
        - NoiseCharacteristics (automatic analysis)
        
        Returns
        -------
        Dict[str, Any]
            Complete system description including:
            - System type and stochastic flag
            - Dimensions (nx, nu, nw)
            - SDE type (ito/stratonovich)
            - Noise characteristics (type, dependencies)
            - Recommended solvers (per backend)
            - Compilation status
            - Optimization opportunities
        
        Examples
        --------
        >>> info = system.get_info()
        >>> print(f"System type: {info['system_type']}")
        >>> print(f"Noise type: {info['noise']['type']}")
        >>> print(f"Recommended JAX solvers: {info['recommended_solvers']['jax']}")
        >>> 
        >>> # Check optimization opportunities
        >>> if info['optimization_opportunities']['precompute_diffusion']:
        ...     print("Can use constant noise optimization!")
        """
        # Get base info from parent
        base_info = super().get_backend_info()
        
        # Get diffusion handler info
        diffusion_info = self.diffusion_handler.get_info()
        
        # Build comprehensive SDE info
        sde_info = {
            **base_info,
            'system_type': 'StochasticDynamicalSystem',
            'is_stochastic': True,
            'sde_type': self.sde_type.value,
            'dimensions': {
                'nx': self.nx,
                'nu': self.nu,
                'nw': self.nw,
            },
            'diffusion': diffusion_info,
            'noise': {
                'type': self.noise_characteristics.noise_type.value,
                'is_additive': self.noise_characteristics.is_additive,
                'is_multiplicative': self.noise_characteristics.is_multiplicative,
                'is_diagonal': self.noise_characteristics.is_diagonal,
                'is_scalar': self.noise_characteristics.is_scalar,
                'depends_on': {
                    'state': self.noise_characteristics.depends_on_state,
                    'control': self.noise_characteristics.depends_on_control,
                    'time': self.noise_characteristics.depends_on_time,
                },
                'state_dependencies': [str(s) for s in self.noise_characteristics.state_dependencies],
                'control_dependencies': [str(s) for s in self.noise_characteristics.control_dependencies],
            },
            'recommended_solvers': {
                'jax': self.recommend_solvers('jax'),
                'torch': self.recommend_solvers('torch'),
                'julia': self.recommend_solvers('numpy'),
            },
            'optimization_opportunities': self.get_optimization_opportunities(),
        }
        
        return sde_info
    
    def print_sde_info(self):
        """
        Print formatted SDE system information.
        
        Displays comprehensive information about the SDE system including
        dimensions, noise type, dependencies, and solver recommendations.
        
        Examples
        --------
        >>> system.print_sde_info()
        ======================================================================
        Stochastic Dynamical System: OrnsteinUhlenbeck
        ======================================================================
        Dimensions: nx=1, nu=1, nw=1
        SDE Type: ito
        Noise Type: additive
        
        Noise Characteristics:
          • Additive: True
          • Multiplicative: False
          • Diagonal: False
          • Scalar: True
        
        Dependencies:
          • Depends on state: False
          • Depends on control: False
          • Depends on time: False
        
        Recommended Solvers (JAX):
          • sea, shark, sra1
        
        Optimization Opportunities:
          • Precompute diffusion: True
          • Use diagonal solver: False
          • Vectorize easily: True
        ======================================================================
        """
        print("=" * 70)
        print(f"Stochastic Dynamical System: {self.__class__.__name__}")
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
        print(f"  • Depends on state: {self.noise_characteristics.depends_on_state}")
        print(f"  • Depends on control: {self.noise_characteristics.depends_on_control}")
        print(f"  • Depends on time: {self.noise_characteristics.depends_on_time}")
        
        print("\nRecommended Solvers (JAX):")
        solvers = self.recommend_solvers('jax')
        print(f"  • {', '.join(solvers)}")
        
        print("\nOptimization Opportunities:")
        opts = self.get_optimization_opportunities()
        print(f"  • Precompute diffusion: {opts['precompute_diffusion']}")
        print(f"  • Use diagonal solver: {opts['use_diagonal_solver']}")
        print(f"  • Vectorize easily: {opts['vectorize_easily']}")
        
        print("=" * 70)
    
    # ========================================================================
    # Conversion Utilities
    # ========================================================================
    
    def to_deterministic(self):
        """
        Extract deterministic (drift-only) system.
        
        Creates a SymbolicDynamicalSystem with only the drift dynamics,
        effectively removing the stochastic component.
        
        Returns
        -------
        SymbolicDynamicalSystem
            Deterministic version (ODE) with same drift
        
        Examples
        --------
        >>> # Get deterministic approximation
        >>> ode_system = sde_system.to_deterministic()
        >>> 
        >>> # Compare mean trajectories
        >>> dx_deterministic = ode_system(x, u)
        >>> dx_stochastic_mean = sde_system.drift(x, u)
        >>> # These are identical (both are drift term)
        >>> 
        >>> # But SDE adds noise:
        >>> g = sde_system.diffusion(x, u)
        >>> # In SDE: dx = dx_stochastic_mean * dt + g @ dW
        
        Notes
        -----
        This is useful for:
        - Comparing stochastic vs deterministic behavior
        - Initial controller design on deterministic approximation
        - Analyzing mean dynamics without noise
        """
        # Create wrapper class that properly initializes as deterministic system
        class DeterministicWrapper(SymbolicDynamicalSystem):
            """Deterministic version of stochastic system."""
            
            def __init__(self, sde_system):
                # Store SDE system reference
                self._source_sde = sde_system
                
                # Call parent __init__ which will call define_system
                super().__init__()
            
            def define_system(self):
                """Define system using drift from SDE system."""
                # Copy drift attributes from SDE system
                self.state_vars = self._source_sde.state_vars
                self.control_vars = self._source_sde.control_vars
                self._f_sym = self._source_sde._f_sym
                self.parameters = self._source_sde.parameters
                self.order = self._source_sde.order
                self._h_sym = self._source_sde._h_sym if hasattr(self._source_sde, '_h_sym') else None
                self.output_vars = self._source_sde.output_vars
        
        return DeterministicWrapper(self)
    
    # ========================================================================
    # String Representations
    # ========================================================================
    
    def __repr__(self) -> str:
        """
        Detailed string representation for debugging.
        
        Returns
        -------
        str
            Detailed representation
        
        Examples
        --------
        >>> repr(system)
        'OrnsteinUhlenbeck(nx=1, nu=1, nw=1, noise=additive, sde=ito)'
        """
        return (
            f"{self.__class__.__name__}("
            f"nx={self.nx}, nu={self.nu}, nw={self.nw}, "
            f"noise={self.noise_characteristics.noise_type.value}, "
            f"sde={self.sde_type.value}, "
            f"backend={self._default_backend})"
        )
    
    def __str__(self) -> str:
        """
        Human-readable string representation.
        
        Returns
        -------
        str
            Concise representation
        
        Examples
        --------
        >>> str(system)
        'OrnsteinUhlenbeck: 1 state, 1 control, 1 noise source (additive)'
        """
        return (
            f"{self.__class__.__name__}: "
            f"{self.nx} state{'s' if self.nx != 1 else ''}, "
            f"{self.nu} control{'s' if self.nu != 1 else ''}, "
            f"{self.nw} noise source{'s' if self.nw != 1 else ''} "
            f"({self.noise_characteristics.noise_type.value})"
        )