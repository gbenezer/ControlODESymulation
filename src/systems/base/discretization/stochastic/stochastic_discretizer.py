"""
Stochastic Discretizer - Converts Continuous-Time SDEs to Discrete-Time
========================================================================

Provides numerical discretization of stochastic differential equations using
the existing SDE integrator framework. Supports multiple discretization schemes,
noise handling, and backends.

Mathematical Form
-----------------
Continuous SDE: dx = f(x, u, t)dt + g(x, u, t)dW
Discrete SDE:   x[k+1] = f_d(x[k], u[k]) + g_d(x[k], u[k]) * w[k]

where:
    - f_d: Discretized drift term
    - g_d: Discretized diffusion term (includes sqrt(dt))
    - w[k] ~ N(0, I): Standard normal random variable

Examples
--------
>>> # Create stochastic discretizer
>>> ou_process = OrnsteinUhlenbeck(alpha=2.0, sigma=0.5)
>>> discretizer = StochasticDiscretizer(ou_process, dt=0.01, method='euler')
>>> 
>>> # Single step (automatic noise generation)
>>> x_next = discretizer.step(x, u)
>>> 
>>> # Single step with custom noise (reproducibility)
>>> w = np.random.randn(discretizer.nw)
>>> x_next = discretizer.step(x, u, w=w)
>>> 
>>> # Linearize (returns Ad, Bd, Gd where Gd is noise gain)
>>> Ad, Bd, Gd = discretizer.linearize(x_eq, u_eq)
"""

from typing import Optional, Tuple, Union, Dict, Any, TYPE_CHECKING
import numpy as np

from src.systems.base.numerical_integration.stochastic.sde_integrator_base import (
    SDEIntegratorBase,
    ConvergenceType,
    SDEIntegrationResult
)
from src.systems.base.numerical_integration.stochastic.sde_integrator_factory import (
    SDEIntegratorFactory
)
from src.systems.base.numerical_integration.integrator_base import (
    StepMode,
    ArrayLike
)

if TYPE_CHECKING:
    from src.systems.base.stochastic_dynamical_system import StochasticDynamicalSystem
    import torch
    import jax.numpy as jnp


class StochasticDiscretizationMethod:
    """
    Enumeration of stochastic discretization methods.
    
    Maps to underlying SDE integrator methods.
    """
    # Euler-Maruyama (most common, order 0.5 strong)
    EULER = 'euler'              # All backends
    EM = 'EM'                    # Julia (same as Euler)
    
    # Milstein (order 1.0 strong)
    MILSTEIN = 'milstein'        # torch
    ITO_MILSTEIN = 'ItoMilstein' # jax
    
    # Stochastic Runge-Kutta (higher order)
    SRK = 'srk'                  # torch
    SRIW1 = 'SRIW1'              # Julia (order 1.5 strong for diagonal)
    
    # Additive noise optimized
    SEA = 'SEA'                  # jax (order 1.5 weak for additive)
    SHARK = 'SHARK'              # jax (order 2.0 weak for additive)
    SRA1 = 'SRA1'                # Julia/jax (order 2.0 weak)


class StochasticDiscretizer:
    """
    Converts continuous-time SDEs to discrete-time.
    
    Delegates to the SDE integrator framework for numerical integration,
    providing a discrete-time specific interface with noise handling,
    linearization, and Monte Carlo support.
    
    Attributes
    ----------
    continuous_system : StochasticDynamicalSystem
        The continuous-time SDE system to discretize
    dt : float
        Time step for discretization
    integrator : SDEIntegratorBase
        Numerical integrator for computing discrete dynamics
    method : str
        Discretization method name
    backend : str
        Backend being used ('numpy', 'torch', 'jax')
    nw : int
        Number of independent Wiener processes
    seed : Optional[int]
        Random seed for reproducibility
    
    Examples
    --------
    >>> # Basic usage
    >>> ou_process = OrnsteinUhlenbeck(alpha=2.0, sigma=0.5)
    >>> discretizer = StochasticDiscretizer(ou_process, dt=0.01, method='euler')
    >>> 
    >>> # Single step (automatic noise)
    >>> x_next = discretizer.step(x, u)
    >>> 
    >>> # Single step with custom noise
    >>> w = np.random.randn(1)  # nw=1 for OU process
    >>> x_next = discretizer.step(x, u, w=w)
    >>> 
    >>> # Controlled SDE
    >>> x_next = discretizer.step(np.array([1.0]), np.array([0.5]))
    >>> 
    >>> # Autonomous SDE
    >>> x_next = discretizer.step(np.array([1.0]), u=None)
    >>> 
    >>> # Linearize (includes noise gain)
    >>> Ad, Bd, Gd = discretizer.linearize(x_eq, u_eq)
    """

    # Add class-level defaults for each backend
    _BACKEND_DEFAULTS = {
        'numpy': 'EM',      # Julia Euler-Maruyama
        'torch': 'euler',   # TorchSDE euler
        'jax': 'Euler',     # Diffrax Euler (capital E)
    }
    
    def __init__(
        self,
        continuous_system: 'StochasticDynamicalSystem',
        dt: float,
        integrator: Optional[SDEIntegratorBase] = None,
        method: Optional[str] = None,
        backend: Optional[str] = None,
        seed: Optional[int] = None,
        convergence_type: ConvergenceType = ConvergenceType.STRONG,
        **integrator_options
    ):
        """
        Initialize stochastic discretizer.
        
        Parameters
        ----------
        continuous_system : StochasticDynamicalSystem
            Continuous-time SDE to discretize (controlled or autonomous)
        dt : float
            Time step for discretization (must be positive)
        integrator : Optional[SDEIntegratorBase]
            Custom SDE integrator instance. If provided, overrides method/backend.
        method : str
            Integration method name. Default: 'euler'
            Options: 'euler', 'milstein', 'srk', 'EM', 'SRIW1', etc.
        backend : Optional[str]
            Backend to use ('numpy', 'torch', 'jax').
            If None, uses system's default backend.
        seed : Optional[int]
            Random seed for reproducibility
        convergence_type : ConvergenceType
            Strong (pathwise) or weak (moment) convergence
        **integrator_options
            Additional options passed to integrator factory
            (sde_type, cache_noise, etc.)
        
        Raises
        ------
        ValueError
            If dt <= 0 or if system is not initialized
        TypeError
            If continuous_system is not StochasticDynamicalSystem
        
        Examples
        --------
        >>> # Default (Euler-Maruyama)
        >>> discretizer = StochasticDiscretizer(sde_system, dt=0.01)
        >>> 
        >>> # Milstein (higher order)
        >>> discretizer = StochasticDiscretizer(
        ...     sde_system, dt=0.01, method='milstein', backend='torch'
        ... )
        >>> 
        >>> # Julia high-accuracy
        >>> discretizer = StochasticDiscretizer(
        ...     sde_system, dt=0.01, method='SRIW1', backend='numpy'
        ... )
        >>> 
        >>> # With reproducible seed
        >>> discretizer = StochasticDiscretizer(
        ...     sde_system, dt=0.01, method='euler', seed=42
        ... )
        >>> 
        >>> # Autonomous SDE
        >>> discretizer = StochasticDiscretizer(autonomous_sde, dt=0.01)
        """
        # Validate inputs
        if dt <= 0:
            raise ValueError(f"Time step dt must be positive, got {dt}")
        
        # Validate system type
        from src.systems.base.stochastic_dynamical_system import StochasticDynamicalSystem
        
        if not isinstance(continuous_system, StochasticDynamicalSystem):
            raise TypeError(
                f"continuous_system must be StochasticDynamicalSystem, "
                f"got {type(continuous_system).__name__}"
            )
        
        if not hasattr(continuous_system, '_initialized') or not continuous_system._initialized:
            raise ValueError(
                f"System {continuous_system.__class__.__name__} is not initialized. "
                f"Ensure define_system() has been called."
            )
        
        self.continuous_system = continuous_system
        self.dt = dt
        self.seed = seed
        self.convergence_type = convergence_type
        
        # Determine backend
        if backend is None:
            backend = continuous_system._default_backend
        self.backend = backend
        
        # Determine method (use backend-appropriate default if not specified)
        if method is None:
            method = self._BACKEND_DEFAULTS.get(backend, 'EM')
        self.method = method
        
        # Create or store integrator
        if integrator is not None:
            # User provided custom integrator
            self.integrator = integrator
            self.method = integrator.name
            
            # Validate integrator's dt matches
            if hasattr(integrator, 'dt') and integrator.dt is not None:
                if abs(integrator.dt - dt) > 1e-12:
                    import warnings
                    warnings.warn(
                        f"Integrator dt={integrator.dt} doesn't match "
                        f"Discretizer dt={dt}. Using Discretizer dt."
                    )
        else:
            # Create integrator from method
            self.method = method
            
            # REUSE: Delegate to existing SDE integrator factory
            self.integrator = SDEIntegratorFactory.create(
                continuous_system,
                backend=backend,
                method=method,
                dt=dt,
                step_mode=StepMode.FIXED,  # Discrete time uses fixed steps
                seed=seed,
                convergence_type=convergence_type,
                **integrator_options
            )
        
        # Cache system dimensions
        self.nx = continuous_system.nx
        self.nu = continuous_system.nu
        self.ny = continuous_system.ny
        self.nw = continuous_system.nw  # Number of Wiener processes
        self.order = continuous_system.order
        
        # Cache noise properties
        self._is_additive = continuous_system.is_additive_noise()
        self._is_multiplicative = continuous_system.is_multiplicative_noise()
        self._is_diagonal = continuous_system.is_diagonal_noise()
        self._is_scalar = continuous_system.is_scalar_noise()
        
        # Cache for constant diffusion (additive noise only)
        self._cached_diffusion = None
        if self._is_additive:
            self._cache_constant_diffusion()
    
    def _cache_constant_diffusion(self):
        """
        Cache constant diffusion for additive noise systems.
        
        For additive noise, g(x,u,t) is constant, so compute once
        and reuse for efficiency.
        """
        if self._is_additive:
            self._cached_diffusion = self.continuous_system.get_constant_noise(
                backend=self.backend
            )
    
    # ========================================================================
    # Primary Interface - Discrete Stochastic Dynamics
    # ========================================================================
    
    def step(
        self,
        x: ArrayLike,
        u: Optional[ArrayLike] = None,
        w: Optional[ArrayLike] = None,
        dt: Optional[float] = None
    ) -> ArrayLike:
        """
        Compute one discrete stochastic step: x[k+1] = f_d(x[k], u[k], w[k]).
        
        Uses numerical SDE integration with Brownian motion increments.
        
        Parameters
        ----------
        x : ArrayLike
            Current state (nx,) or (batch, nx)
        u : Optional[ArrayLike]
            Control input (nu,) or (batch, nu)
            For autonomous systems (nu=0), u can be None
        w : Optional[ArrayLike]
            Standard normal random variable (nw,) or (batch, nw)
            If None, generated automatically using system's RNG
            If provided, used directly (for reproducibility)
        dt : Optional[float]
            Time step (uses self.dt if None)
        
        Returns
        -------
        ArrayLike
            Next state x[k+1], same shape and backend as input
        
        Examples
        --------
        >>> # Automatic noise generation
        >>> x_next = discretizer.step(np.array([1.0]), np.array([0.5]))
        >>> 
        >>> # Custom noise for reproducibility
        >>> w = np.random.randn(1)
        >>> x_next = discretizer.step(np.array([1.0]), np.array([0.5]), w=w)
        >>> 
        >>> # Autonomous SDE
        >>> x_next = discretizer.step(np.array([1.0]), u=None)
        >>> 
        >>> # Batched
        >>> x_batch = np.array([[1.0], [2.0]])
        >>> u_batch = np.array([[0.5], [0.3]])
        >>> x_next_batch = discretizer.step(x_batch, u_batch)
        
        Notes
        -----
        The noise w is a **standard normal** N(0, I), not the Brownian increment.
        The integrator internally converts: dW = sqrt(dt) * w
        
        For reproducibility, use the same sequence of w values or set seed.
        """
        # Use default dt if not specified
        if dt is None:
            dt = self.dt
        
        # Convert w to Brownian increment dW if provided
        # The integrator expects dW ~ N(0, dt*I), but we accept w ~ N(0, I)
        if w is not None:
            # Scale by sqrt(dt) to get Brownian increment
            if self.backend == 'numpy':
                import numpy as np
                dW = w * np.sqrt(dt)
            elif self.backend == 'torch':
                import torch
                dW = w * torch.sqrt(torch.tensor(dt))
            elif self.backend == 'jax':
                import jax.numpy as jnp
                dW = w * jnp.sqrt(dt)
        else:
            dW = None  # Let integrator generate
        
        # Delegate to SDE integrator
        return self.integrator.step(x, u, dt=dt, dW=dW)
    
    def __call__(
        self,
        x: ArrayLike,
        u: Optional[ArrayLike] = None,
        w: Optional[ArrayLike] = None
    ) -> ArrayLike:
        """
        Make discretizer callable: discretizer(x, u, w) → x[k+1].
        
        Examples
        --------
        >>> x_next = discretizer(x, u)
        >>> # Equivalent to:
        >>> x_next = discretizer.step(x, u)
        """
        return self.step(x, u, w)
    
    # ========================================================================
    # Linearization - Stochastic Discrete-Time Jacobians
    # ========================================================================
    
    def linearize(
        self,
        x_eq: ArrayLike,
        u_eq: Optional[ArrayLike] = None,
        method: str = 'euler',
        Q_noise: Optional[ArrayLike] = None
    ) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
        """
        Compute stochastic discrete-time linearization: (Ad, Bd, Gd).
        
        Linearized discrete SDE around equilibrium:
            δx[k+1] = Ad*δx[k] + Bd*δu[k] + Gd*w[k]
        
        where w[k] ~ N(0, I) is standard normal.
        
        Parameters
        ----------
        x_eq : ArrayLike
            Equilibrium state for linearization (nx,)
        u_eq : Optional[ArrayLike]
            Equilibrium control (nu,)
            For autonomous systems (nu=0), u_eq can be None
        method : str
            Discretization method for drift/control linearization:
            - 'euler': Ad = I + dt*Ac, Bd = dt*Bc (default)
            - 'exact': Ad = expm(Ac*dt), Bd = exact
        Q_noise : Optional[ArrayLike]
            Process noise covariance (nx, nx) for augmented linearization
            If None, uses identity (standard formulation)
        
        Returns
        -------
        Ad : ArrayLike
            Discrete-time state matrix (nx, nx)
        Bd : ArrayLike
            Discrete-time control matrix (nx, nu)
            For autonomous systems: shape (nx, 0)
        Gd : ArrayLike
            Discrete-time noise gain matrix (nx, nw)
            This is g(x_eq, u_eq) * sqrt(dt) for Euler method
        
        Examples
        --------
        >>> # Controlled SDE
        >>> Ad, Bd, Gd = discretizer.linearize(x_eq, u_eq, method='euler')
        >>> 
        >>> # Autonomous SDE
        >>> Ad, Bd, Gd = discretizer.linearize(x_eq, u=None, method='euler')
        >>> print(Bd.shape)  # (nx, 0) - empty
        >>> print(Gd.shape)  # (nx, nw)
        >>> 
        >>> # Exact discretization
        >>> Ad_exact, Bd_exact, Gd_exact = discretizer.linearize(
        ...     x_eq, u_eq, method='exact'
        ... )
        >>> 
        >>> # Use in stochastic control design
        >>> K = design_lqg_controller(Ad, Bd, Gd, Q, R, Q_noise, R_noise)
        
        Notes
        -----
        **Linearized Discrete SDE:**
        
        For Euler-Maruyama discretization:
            Ad = I + dt * ∂f/∂x
            Bd = dt * ∂f/∂u
            Gd = g(x_eq, u_eq) * sqrt(dt)
        
        For exact discretization (drift only):
            Ad = expm(∂f/∂x * dt)
            Bd = integral form
            Gd = g(x_eq, u_eq) * sqrt(dt)  (diffusion not affected)
        
        The noise term in discrete SDE:
            x[k+1] = Ad*x[k] + Bd*u[k] + Gd*w[k]
        where w[k] ~ N(0, I).
        
        Covariance propagation:
            Cov[x[k+1]] = Ad*Cov[x[k]]*Ad^T + Gd*Gd^T
        """
        # Get continuous-time drift linearization (Ac, Bc)
        Ac, Bc = self.continuous_system.linearized_dynamics(x_eq, u_eq, backend=self.backend)
        
        # Get diffusion at equilibrium
        g_eq = self.continuous_system.diffusion(x_eq, u_eq, backend=self.backend)
        
        # Discretize drift terms
        if method == 'euler':
            Ad, Bd = self._linearize_drift_euler(Ac, Bc)
        elif method == 'exact':
            Ad, Bd = self._linearize_drift_exact(Ac, Bc)
        elif method == 'tustin':
            Ad, Bd = self._linearize_drift_tustin(Ac, Bc)
        else:
            raise ValueError(
                f"Unknown linearization method '{method}'. "
                f"Choose from: 'euler', 'exact', 'tustin'"
            )
        
        # Discretize diffusion term
        # For Euler-Maruyama: Gd = g * sqrt(dt)
        Gd = self._linearize_diffusion(g_eq, method)
        
        return Ad, Bd, Gd
    
    def _linearize_drift_euler(
        self,
        Ac: ArrayLike,
        Bc: ArrayLike
    ) -> Tuple[ArrayLike, ArrayLike]:
        """Euler discretization of drift: Ad = I + dt*Ac, Bd = dt*Bc"""
        if self.backend == 'numpy':
            import numpy as np
            Ad = np.eye(self.nx) + self.dt * Ac
            Bd = self.dt * Bc
        elif self.backend == 'torch':
            import torch
            Ad = torch.eye(self.nx, device=Ac.device, dtype=Ac.dtype) + self.dt * Ac
            Bd = self.dt * Bc
        elif self.backend == 'jax':
            import jax.numpy as jnp
            Ad = jnp.eye(self.nx) + self.dt * Ac
            Bd = self.dt * Bc
        
        return Ad, Bd
    
    def _linearize_drift_exact(
        self,
        Ac: ArrayLike,
        Bc: ArrayLike
    ) -> Tuple[ArrayLike, ArrayLike]:
        """Exact discretization of drift using matrix exponential."""
        if self.backend == 'numpy':
            import numpy as np
            from scipy.linalg import expm
            
            Ad = expm(self.dt * Ac)
            
            # Compute Bd
            if np.linalg.matrix_rank(Ac) < self.nx:
                Bd = self.dt * Bc
            else:
                Ac_inv = np.linalg.inv(Ac)
                Bd = Ac_inv @ (Ad - np.eye(self.nx)) @ Bc
        
        elif self.backend == 'torch':
            import torch
            
            Ad = torch.linalg.matrix_exp(self.dt * Ac)
            
            try:
                Ac_inv = torch.linalg.inv(Ac)
                Bd = Ac_inv @ (Ad - torch.eye(self.nx, device=Ac.device, dtype=Ac.dtype)) @ Bc
            except RuntimeError:
                Bd = self.dt * Bc
        
        elif self.backend == 'jax':
            import jax.numpy as jnp
            from jax.scipy.linalg import expm as jax_expm
            
            Ad = jax_expm(self.dt * Ac)
            
            try:
                Ac_inv = jnp.linalg.inv(Ac)
                Bd = Ac_inv @ (Ad - jnp.eye(self.nx)) @ Bc
            except:
                Bd = self.dt * Bc
        
        return Ad, Bd
    
    def _linearize_drift_tustin(
        self,
        Ac: ArrayLike,
        Bc: ArrayLike
    ) -> Tuple[ArrayLike, ArrayLike]:
        """Tustin (bilinear) discretization of drift."""
        if self.backend == 'numpy':
            import numpy as np
            I = np.eye(self.nx)
            factor = self.dt / 2.0
            inv_term = np.linalg.inv(I - factor * Ac)
            Ad = inv_term @ (I + factor * Ac)
            Bd = np.sqrt(self.dt) * inv_term @ Bc
        
        elif self.backend == 'torch':
            import torch
            I = torch.eye(self.nx, device=Ac.device, dtype=Ac.dtype)
            factor = self.dt / 2.0
            inv_term = torch.linalg.inv(I - factor * Ac)
            Ad = inv_term @ (I + factor * Ac)
            Bd = torch.sqrt(torch.tensor(self.dt)) * inv_term @ Bc
        
        elif self.backend == 'jax':
            import jax.numpy as jnp
            I = jnp.eye(self.nx)
            factor = self.dt / 2.0
            inv_term = jnp.linalg.inv(I - factor * Ac)
            Ad = inv_term @ (I + factor * Ac)
            Bd = jnp.sqrt(self.dt) * inv_term @ Bc
        
        return Ad, Bd
    
    def _linearize_diffusion(
        self,
        g_eq: ArrayLike,
        method: str
    ) -> ArrayLike:
        """
        Discretize diffusion term: Gd = g * sqrt(dt).
        
        For Euler-Maruyama and most SDE schemes, the discrete
        noise gain is simply the continuous diffusion scaled by sqrt(dt).
        """
        if self.backend == 'numpy':
            import numpy as np
            Gd = g_eq * np.sqrt(self.dt)
        elif self.backend == 'torch':
            import torch
            Gd = g_eq * torch.sqrt(torch.tensor(self.dt))
        elif self.backend == 'jax':
            import jax.numpy as jnp
            Gd = g_eq * jnp.sqrt(self.dt)
        
        return Gd
    
    # ========================================================================
    # Observation - Delegates to Continuous System
    # ========================================================================
    
    def h(self, x: ArrayLike, backend: Optional[str] = None) -> ArrayLike:
        """
        Evaluate discrete output: y[k] = h(x[k]).
        
        For discrete-time systems, output is same as continuous-time.
        
        Examples
        --------
        >>> y = discretizer.h(x)
        """
        return self.continuous_system.h(x, backend or self.backend)
    
    def linearized_observation(
        self,
        x: ArrayLike,
        backend: Optional[str] = None
    ) -> ArrayLike:
        """
        Compute discrete observation Jacobian: Cd = dh/dx.
        
        Same as continuous-time since h(x) doesn't depend on discretization.
        
        Examples
        --------
        >>> Cd = discretizer.linearized_observation(x_eq)
        """
        return self.continuous_system.linearized_observation(x, backend or self.backend)
    
    # ========================================================================
    # Noise Analysis and Optimization
    # ========================================================================
    
    def is_additive_noise(self) -> bool:
        """Check if noise is additive (constant)."""
        return self._is_additive
    
    def is_multiplicative_noise(self) -> bool:
        """Check if noise is multiplicative (state-dependent)."""
        return self._is_multiplicative
    
    def is_diagonal_noise(self) -> bool:
        """Check if noise sources are independent."""
        return self._is_diagonal
    
    def is_scalar_noise(self) -> bool:
        """Check if system has single noise source."""
        return self._is_scalar
    
    def get_constant_noise_gain(self) -> ArrayLike:
        """
        Get constant noise gain for additive noise systems.
        
        Returns Gd = g * sqrt(dt) which is constant for additive noise.
        
        Returns
        -------
        ArrayLike
            Constant noise gain matrix (nx, nw)
        
        Raises
        ------
        ValueError
            If noise is not additive
        
        Examples
        --------
        >>> if discretizer.is_additive_noise():
        ...     Gd = discretizer.get_constant_noise_gain()
        ...     # Use Gd in simulation - huge performance gain!
        """
        if not self._is_additive:
            raise ValueError(
                "get_constant_noise_gain() only valid for additive noise. "
                f"System has {self.continuous_system.get_noise_type().value} noise."
            )
        
        if self._cached_diffusion is None:
            self._cache_constant_diffusion()
        
        # Scale by sqrt(dt)
        if self.backend == 'numpy':
            import numpy as np
            return self._cached_diffusion * np.sqrt(self.dt)
        elif self.backend == 'torch':
            import torch
            return self._cached_diffusion * torch.sqrt(torch.tensor(self.dt))
        elif self.backend == 'jax':
            import jax.numpy as jnp
            return self._cached_diffusion * jnp.sqrt(self.dt)
    
    # ========================================================================
    # Utility Methods
    # ========================================================================
    
    def set_seed(self, seed: int):
        """
        Set random seed for reproducibility.
        
        Parameters
        ----------
        seed : int
            Random seed
        
        Examples
        --------
        >>> discretizer.set_seed(42)
        >>> x1 = discretizer.step(x, u)
        >>> discretizer.set_seed(42)
        >>> x2 = discretizer.step(x, u)
        >>> # x1 and x2 should be identical (if backend supports it)
        """
        self.seed = seed
        self.integrator.set_seed(seed)
    
    def set_dt(self, new_dt: float):
        """
        Change the time step.
        
        Recreates the integrator with the new time step.
        
        Parameters
        ----------
        new_dt : float
            New time step (must be positive)
        
        Examples
        --------
        >>> discretizer.set_dt(0.005)
        >>> discretizer.dt
        0.005
        """
        if new_dt <= 0:
            raise ValueError(f"Time step must be positive, got {new_dt}")
        
        old_dt = self.dt
        self.dt = new_dt
        
        # Recreate integrator with new dt
        self.integrator = SDEIntegratorFactory.create(
            self.continuous_system,
            backend=self.backend,
            method=self.method,
            dt=new_dt,
            step_mode=StepMode.FIXED,
            seed=self.seed,
            convergence_type=self.convergence_type
        )
        
        # Update cached diffusion if applicable
        if self._is_additive:
            self._cache_constant_diffusion()
        
        print(f"Time step changed: {old_dt:.6f} → {new_dt:.6f}")
    
    def get_noise_info(self) -> Dict[str, Any]:
        """
        Get information about noise structure.
        
        Returns
        -------
        dict
            Noise properties including type, structure, optimizations
        
        Examples
        --------
        >>> info = discretizer.get_noise_info()
        >>> print(info['noise_type'])
        'additive'
        >>> if info['can_precompute']:
        ...     Gd = discretizer.get_constant_noise_gain()
        """
        return {
            'nw': self.nw,
            'noise_type': self.continuous_system.get_noise_type().value,
            'is_additive': self._is_additive,
            'is_multiplicative': self._is_multiplicative,
            'is_diagonal': self._is_diagonal,
            'is_scalar': self._is_scalar,
            'can_precompute': self._is_additive,
            'cached_diffusion': self._cached_diffusion is not None,
        }
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get comprehensive stochastic discretizer information.
        
        Returns
        -------
        dict
            Information about discretizer configuration
        
        Examples
        --------
        >>> info = discretizer.get_info()
        >>> print(info['method'])
        'euler'
        >>> print(info['nw'])
        1
        """
        base_info = {
            'continuous_system': self.continuous_system.__class__.__name__,
            'system_type': 'StochasticDynamicalSystem',
            'dt': self.dt,
            'method': self.method,
            'backend': self.backend,
            'integrator': self.integrator.name,
            'seed': self.seed,
            'convergence_type': self.convergence_type.value,
            'dimensions': {
                'nx': self.nx,
                'nu': self.nu,
                'ny': self.ny,
                'nw': self.nw,
            },
            'order': self.order,
            'is_autonomous': self.nu == 0,
        }
        
        # Add noise info
        noise_info = self.get_noise_info()
        base_info['noise'] = noise_info
        
        return base_info
    
    def __repr__(self) -> str:
        return (
            f"StochasticDiscretizer({self.continuous_system.__class__.__name__}, "
            f"dt={self.dt}, method={self.method}, backend={self.backend}, "
            f"nw={self.nw}, seed={self.seed})"
        )
    
    def __str__(self) -> str:
        autonomous_str = " (autonomous)" if self.nu == 0 else ""
        noise_str = f" ({self.continuous_system.get_noise_type().value} noise)"
        return (
            f"StochasticDiscretizer: {self.continuous_system.__class__.__name__} → "
            f"dt={self.dt:.4f}, {self.method}{noise_str}{autonomous_str}"
        )


# ============================================================================
# Utility Functions
# ============================================================================

def create_stochastic_discretizer(
    sde_system: 'StochasticDynamicalSystem',
    dt: float,
    method: Optional[str] = None,
    **kwargs
) -> StochasticDiscretizer:
    """
    Convenience function for creating stochastic discretizers.

    If method is None, uses backend-appropriate default.
    
    Examples
    --------
    >>> discretizer = create_stochastic_discretizer(sde_system, dt=0.01)
    >>> discretizer = create_stochastic_discretizer(
    ...     sde_system, dt=0.01, method='milstein', backend='torch'
    ... )
    """
    return StochasticDiscretizer(sde_system, dt, method=method, **kwargs)