"""
Discretizer - Converts Continuous-Time Systems to Discrete-Time

Provides numerical discretization of continuous-time dynamical systems using
the existing integrator framework. Supports multiple discretization methods
and backends.

Mathematical Form
-----------------
Continuous system: dx/dt = f(x, u)
Discrete system:   x[k+1] = f_d(x[k], u[k])

where f_d is computed via numerical integration over time step dt.

Examples
--------
>>> # Create discretizer
>>> pendulum_ct = SymbolicPendulum(m=1.0, l=0.5, g=9.81)
>>> discretizer = Discretizer(pendulum_ct, dt=0.01, method='rk4')
>>> 
>>> # Single step
>>> x_next = discretizer.step(x, u)
>>> 
>>> # Linearize
>>> Ad, Bd = discretizer.linearize(x_eq, u_eq)
"""

from typing import Optional, Tuple, Union, Dict, Any, TYPE_CHECKING
import numpy as np

from src.systems.base.numerical_integration.integrator_base import (
    IntegratorBase,
    StepMode,
    ArrayLike
)
from src.systems.base.numerical_integration.integrator_factory import (
    IntegratorFactory
)

if TYPE_CHECKING:
    from src.systems.base.symbolic_dynamical_system import SymbolicDynamicalSystem
    import torch
    import jax.numpy as jnp


class DiscretizationMethod:
    """
    Enumeration of discretization methods.
    
    Maps to underlying integrator methods but provides
    discrete-time specific context.
    """
    # Fixed-step methods (simple, predictable)
    EULER = 'euler'              # First-order, O(dt)
    MIDPOINT = 'midpoint'        # Second-order, O(dt²)
    RK4 = 'rk4'                  # Fourth-order, O(dt⁴)
    
    # Adaptive methods (higher accuracy)
    SCIPY_RK45 = 'RK45'          # Adaptive Runge-Kutta (scipy)
    SCIPY_LSODA = 'LSODA'        # Adaptive with stiffness detection
    
    # Julia methods (high performance)
    JULIA_TSIT5 = 'Tsit5'        # Julia Tsitouras 5(4)
    JULIA_VERN9 = 'Vern9'        # Julia Verner 9th order
    
    # Backend-specific methods
    JAX_TSIT5 = 'tsit5'          # Diffrax Tsitouras (lowercase)
    JAX_DOPRI5 = 'dopri5'        # Diffrax Dormand-Prince
    TORCH_DOPRI5 = 'dopri5'      # TorchDiffEq (same name)
    
    # Exact discretization (future)
    MATRIX_EXPONENTIAL = 'expm'  # Exact for linear systems


class Discretizer:
    """
    Converts continuous-time dynamics to discrete-time.
    
    Delegates to the integrator framework for numerical integration,
    providing a discrete-time specific interface with additional
    capabilities like linearization.
    
    Attributes
    ----------
    continuous_system : SymbolicDynamicalSystem
        The continuous-time system to discretize
    dt : float
        Time step for discretization
    integrator : IntegratorBase
        Numerical integrator for computing discrete dynamics
    method : str
        Discretization method name
    backend : str
        Backend being used ('numpy', 'torch', 'jax')
    
    Examples
    --------
    >>> # Basic usage
    >>> system_ct = LinearSystem(a=2.0)
    >>> discretizer = Discretizer(system_ct, dt=0.01, method='rk4')
    >>> 
    >>> # Single step
    >>> x_next = discretizer.step(x, u)
    >>> 
    >>> # Controlled system
    >>> x_next = discretizer.step(np.array([1.0, 0.0]), np.array([0.5]))
    >>> 
    >>> # Autonomous system
    >>> x_next = discretizer.step(np.array([1.0, 0.0]), u=None)
    >>> 
    >>> # Custom integrator
    >>> custom_integrator = IntegratorFactory.create(
    ...     system_ct, backend='jax', method='tsit5', dt=0.01
    ... )
    >>> discretizer = Discretizer(system_ct, dt=0.01, integrator=custom_integrator)
    """
    
    def __init__(
        self,
        continuous_system: 'SymbolicDynamicalSystem',
        dt: float,
        integrator: Optional[IntegratorBase] = None,
        method: str = 'rk4',
        backend: Optional[str] = None,
        **integrator_options
    ):
        """
        Initialize discretizer.
        
        Parameters
        ----------
        continuous_system : SymbolicDynamicalSystem
            Continuous-time system to discretize (controlled or autonomous)
        dt : float
            Time step for discretization (must be positive)
        integrator : Optional[IntegratorBase]
            Custom integrator instance. If provided, overrides method/backend.
        method : str
            Integration method name. Default: 'rk4'
            Options: 'euler', 'midpoint', 'rk4', 'RK45', 'LSODA', 'Tsit5', etc.
        backend : Optional[str]
            Backend to use ('numpy', 'torch', 'jax').
            If None, uses system's default backend.
        **integrator_options
            Additional options passed to integrator factory
            (rtol, atol, etc. for adaptive methods)
        
        Raises
        ------
        ValueError
            If dt <= 0 or if system is not initialized
        TypeError
            If continuous_system is not SymbolicDynamicalSystem
        
        Examples
        --------
        >>> # Default (RK4, system's backend)
        >>> discretizer = Discretizer(system, dt=0.01)
        >>> 
        >>> # Euler method (faster, less accurate)
        >>> discretizer = Discretizer(system, dt=0.001, method='euler')
        >>> 
        >>> # High accuracy with Julia
        >>> discretizer = Discretizer(
        ...     system, dt=0.01, method='Tsit5', backend='numpy'
        ... )
        >>> 
        >>> # Adaptive stepping (for smooth discrete simulation)
        >>> discretizer = Discretizer(
        ...     system, dt=0.01, method='RK45', rtol=1e-8, atol=1e-10
        ... )
        >>> 
        >>> # JAX for optimization
        >>> discretizer = Discretizer(system, dt=0.01, method='tsit5', backend='jax')
        >>> 
        >>> # Custom integrator
        >>> integrator = IntegratorFactory.for_optimization(system)
        >>> discretizer = Discretizer(system, dt=0.01, integrator=integrator)
        """
        # Validate inputs
        if dt <= 0:
            raise ValueError(f"Time step dt must be positive, got {dt}")
        
        if not hasattr(continuous_system, '_initialized') or not continuous_system._initialized:
            raise ValueError(
                f"System {continuous_system.__class__.__name__} is not initialized. "
                f"Ensure define_system() has been called."
            )
        
        self.continuous_system = continuous_system
        self.dt = dt
        
        # Determine backend
        if backend is None:
            backend = continuous_system._default_backend
        self.backend = backend
        
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
            
            # REUSE: Delegate to existing integrator factory
            self.integrator = IntegratorFactory.create(
                continuous_system,
                backend=backend,
                method=method,
                dt=dt,
                step_mode=StepMode.FIXED,  # Discrete time uses fixed steps
                **integrator_options
            )
        
        # Cache system dimensions
        self.nx = continuous_system.nx
        self.nu = continuous_system.nu
        self.ny = continuous_system.ny
        self.order = continuous_system.order
        
        # Cache for linearization (optional optimization)
        self._linearization_cache = {}
        self._cache_enabled = False
    
    # ========================================================================
    # Primary Interface - Discrete Dynamics
    # ========================================================================
    
    def step(
        self,
        x: ArrayLike,
        u: Optional[ArrayLike] = None,
        dt: Optional[float] = None
    ) -> ArrayLike:
        """
        Compute one discrete-time step: x[k+1] = f_d(x[k], u[k]).
        
        Uses numerical integration to approximate the continuous-time
        evolution over the time step dt.
        
        Parameters
        ----------
        x : ArrayLike
            Current state (nx,) or (batch, nx)
        u : Optional[ArrayLike]
            Control input (nu,) or (batch, nu)
            For autonomous systems (nu=0), u can be None
        dt : Optional[float]
            Time step (uses self.dt if None)
            Provided for flexibility, but typically should match self.dt
        
        Returns
        -------
        ArrayLike
            Next state x[k+1], same shape and backend as input
        
        Examples
        --------
        >>> # Controlled system
        >>> x_next = discretizer.step(np.array([1.0, 0.0]), np.array([0.5]))
        >>> 
        >>> # Autonomous system
        >>> x_next = discretizer.step(np.array([1.0, 0.0]), u=None)
        >>> 
        >>> # Batched
        >>> x_batch = np.array([[1.0, 0.0], [2.0, 0.0]])
        >>> u_batch = np.array([[0.5], [0.3]])
        >>> x_next_batch = discretizer.step(x_batch, u_batch)
        >>> 
        >>> # Custom time step (not recommended - use for variable-rate control)
        >>> x_next = discretizer.step(x, u, dt=0.005)
        
        Notes
        -----
        For autonomous systems (nu=0), passing u=None is required.
        The integrator will handle the autonomous case internally.
        """
        # Use default dt if not specified
        if dt is None:
            dt = self.dt
        
        # Delegate to integrator
        # The integrator handles autonomous systems (u=None) automatically
        return self.integrator.step(x, u, dt=dt)
    
    def __call__(
        self,
        x: ArrayLike,
        u: Optional[ArrayLike] = None
    ) -> ArrayLike:
        """
        Make discretizer callable: discretizer(x, u) → x[k+1].
        
        Provides convenient function-like interface.
        
        Examples
        --------
        >>> x_next = discretizer(x, u)
        >>> # Equivalent to:
        >>> x_next = discretizer.step(x, u)
        """
        return self.step(x, u)
    
    # ========================================================================
    # Linearization - Discrete-Time Jacobians
    # ========================================================================
    
    def linearize(
        self,
        x_eq: ArrayLike,
        u_eq: Optional[ArrayLike] = None,
        method: str = 'euler'
    ) -> Tuple[ArrayLike, ArrayLike]:
        """
        Compute discrete-time linearization: (Ad, Bd).
        
        Converts continuous-time Jacobians (Ac, Bc) to discrete-time
        using the specified method.
        
        Parameters
        ----------
        x_eq : ArrayLike
            Equilibrium state for linearization (nx,)
        u_eq : Optional[ArrayLike]
            Equilibrium control (nu,)
            For autonomous systems (nu=0), u_eq can be None
        method : str
            Discretization method for linearization:
            - 'euler': Ad = I + dt*Ac, Bd = dt*Bc (default)
            - 'exact': Ad = expm(Ac*dt), Bd = integral (for linear systems)
            - 'tustin': Bilinear (Tustin) transformation
            - 'matched': Zero-order hold with matched DC gain
        
        Returns
        -------
        Ad : ArrayLike
            Discrete-time state matrix (nx, nx)
        Bd : ArrayLike
            Discrete-time control matrix (nx, nu)
            For autonomous systems: shape (nx, 0) - empty matrix
        
        Raises
        ------
        ValueError
            If method is unknown or not implemented
        
        Examples
        --------
        >>> # Controlled system - Euler discretization
        >>> Ad, Bd = discretizer.linearize(x_eq, u_eq, method='euler')
        >>> 
        >>> # Autonomous system
        >>> Ad, Bd = discretizer.linearize(x_eq, u=None, method='euler')
        >>> print(Bd.shape)  # (nx, 0) - empty control matrix
        >>> 
        >>> # Exact discretization (for linear systems)
        >>> Ad_exact, Bd_exact = discretizer.linearize(x_eq, u_eq, method='exact')
        >>> 
        >>> # Use in discrete LQR
        >>> K, S = self._solve_dare(Ad, Bd, Q, R)
        
        Notes
        -----
        **Discretization Methods:**
        
        1. **Euler (Forward Difference)** - First order approximation
           Ad = I + dt * Ac
           Bd = dt * Bc
           
           Pros: Simple, fast
           Cons: Only accurate for small dt, can be unstable
        
        2. **Exact (Matrix Exponential)** - Exact for linear systems
           Ad = expm(Ac * dt)
           Bd = ∫[0,dt] expm(Ac*τ) dτ * Bc
           
           Pros: Exact for linear systems, stable
           Cons: Expensive for large systems, requires matrix exponential
        
        3. **Tustin (Bilinear Transform)** - Second order approximation
           s = 2/dt * (z-1)/(z+1)
           
           Pros: Better stability properties than Euler
           Cons: Frequency warping at high frequencies
        
        4. **Matched (Zero-Order Hold)** - Preserves DC gain
           Matches steady-state behavior between continuous and discrete
           
           Pros: Good for control applications
           Cons: More complex computation
        
        For most applications, 'euler' with small dt is sufficient.
        For critical applications or large dt, use 'exact'.
        """
        # Get continuous-time linearization
        Ac, Bc = self.continuous_system.linearized_dynamics(x_eq, u_eq, backend=self.backend)
        
        # Convert to appropriate backend arrays if needed
        if self.backend == 'numpy':
            import numpy as np
            Ac = np.asarray(Ac)
            Bc = np.asarray(Bc)
        elif self.backend == 'torch':
            import torch
            if not isinstance(Ac, torch.Tensor):
                Ac = torch.as_tensor(Ac)
            if not isinstance(Bc, torch.Tensor):
                Bc = torch.as_tensor(Bc)
        elif self.backend == 'jax':
            import jax.numpy as jnp
            Ac = jnp.asarray(Ac)
            Bc = jnp.asarray(Bc)
        
        # Apply discretization method
        if method == 'euler':
            Ad, Bd = self._linearize_euler(Ac, Bc)
        elif method == 'exact':
            Ad, Bd = self._linearize_exact(Ac, Bc)
        elif method == 'tustin':
            Ad, Bd = self._linearize_tustin(Ac, Bc)
        elif method == 'matched':
            Ad, Bd = self._linearize_matched(Ac, Bc)
        else:
            raise ValueError(
                f"Unknown linearization method '{method}'. "
                f"Choose from: 'euler', 'exact', 'tustin', 'matched'"
            )
        
        return Ad, Bd
    
    def _linearize_euler(
        self,
        Ac: ArrayLike,
        Bc: ArrayLike
    ) -> Tuple[ArrayLike, ArrayLike]:
        """
        Euler (forward difference) discretization.
        
        Ad = I + dt * Ac
        Bd = dt * Bc
        """
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
    
    def _linearize_exact(
        self,
        Ac: ArrayLike,
        Bc: ArrayLike
    ) -> Tuple[ArrayLike, ArrayLike]:
        """
        Exact discretization using matrix exponential.
        
        Ad = expm(Ac * dt)
        Bd = ∫[0,dt] expm(Ac*τ) dτ * Bc
           = Ad * inv(Ac) * (Ad - I) * Bc  (if Ac invertible)
           = dt * Bc                        (if Ac ≈ 0)
        
        This is exact for linear systems.
        """
        if self.backend == 'numpy':
            import numpy as np
            from scipy.linalg import expm
            
            # Compute Ad = expm(Ac * dt)
            Ad = expm(self.dt * Ac)
            
            # Compute Bd - handle singular Ac
            if np.linalg.matrix_rank(Ac) < self.nx:
                # Ac is singular - use Euler approximation for Bd
                Bd = self.dt * Bc
            else:
                # Ac is invertible - use exact formula
                Ac_inv = np.linalg.inv(Ac)
                Bd = Ac_inv @ (Ad - np.eye(self.nx)) @ Bc
        
        elif self.backend == 'torch':
            import torch
            
            # PyTorch matrix exponential
            Ad = torch.linalg.matrix_exp(self.dt * Ac)
            
            # Compute Bd - check if Ac is invertible
            try:
                Ac_inv = torch.linalg.inv(Ac)
                Bd = Ac_inv @ (Ad - torch.eye(self.nx, device=Ac.device, dtype=Ac.dtype)) @ Bc
            except RuntimeError:
                # Singular Ac - use Euler approximation
                Bd = self.dt * Bc
        
        elif self.backend == 'jax':
            import jax.numpy as jnp
            from jax.scipy.linalg import expm as jax_expm
            
            # JAX matrix exponential
            Ad = jax_expm(self.dt * Ac)
            
            # Compute Bd - handle singular Ac
            try:
                Ac_inv = jnp.linalg.inv(Ac)
                Bd = Ac_inv @ (Ad - jnp.eye(self.nx)) @ Bc
            except:
                # Singular Ac - use Euler approximation
                Bd = self.dt * Bc
        
        return Ad, Bd
    
    def _linearize_tustin(
        self,
        Ac: ArrayLike,
        Bc: ArrayLike
    ) -> Tuple[ArrayLike, ArrayLike]:
        """
        Tustin (bilinear) transformation.
        
        Uses the bilinear transform: s = 2/dt * (z-1)/(z+1)
        
        Ad = (I + dt/2 * Ac) * inv(I - dt/2 * Ac)
        Bd = sqrt(dt) * inv(I - dt/2 * Ac) * Bc
        """
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
    
    def _linearize_matched(
        self,
        Ac: ArrayLike,
        Bc: ArrayLike
    ) -> Tuple[ArrayLike, ArrayLike]:
        """
        Matched (zero-order hold) discretization.
        
        Preserves DC gain and step response characteristics.
        Uses exact method but with specific handling for control input.
        """
        # For now, use exact method
        # TODO: Implement proper ZOH matched transformation
        return self._linearize_exact(Ac, Bc)
    
    # ========================================================================
    # Observation - Delegates to Continuous System
    # ========================================================================
    
    def h(self, x: ArrayLike, backend: Optional[str] = None) -> ArrayLike:
        """
        Evaluate discrete output: y[k] = h(x[k]).
        
        For discrete-time systems, the output function is the same as
        continuous-time since h(x) doesn't depend on the discretization.
        
        Parameters
        ----------
        x : ArrayLike
            State (nx,) or (batch, nx)
        backend : Optional[str]
            Backend override
        
        Returns
        -------
        ArrayLike
            Output y, shape (ny,) or (batch, ny)
        
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
        
        For discrete-time systems, Cd = Cc (same as continuous) since
        h(x) doesn't depend on discretization.
        
        Parameters
        ----------
        x : ArrayLike
            State at which to linearize (nx,)
        backend : Optional[str]
            Backend override
        
        Returns
        -------
        ArrayLike
            C matrix, shape (ny, nx)
        
        Examples
        --------
        >>> Cd = discretizer.linearized_observation(x_eq)
        """
        return self.continuous_system.linearized_observation(x, backend or self.backend)
    
    # ========================================================================
    # Utility Methods
    # ========================================================================
    
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
        self.integrator = IntegratorFactory.create(
            self.continuous_system,
            backend=self.backend,
            method=self.method,
            dt=new_dt,
            step_mode=StepMode.FIXED
        )
        
        # Clear linearization cache since dt changed
        self._linearization_cache.clear()
        
        print(f"Time step changed: {old_dt:.6f} → {new_dt:.6f}")
    
    def enable_linearization_cache(self, enable: bool = True):
        """
        Enable or disable linearization caching.
        
        When enabled, linearizations at the same point are cached.
        Useful for repeated linearization at equilibrium.
        
        Parameters
        ----------
        enable : bool
            Whether to enable caching
        
        Examples
        --------
        >>> discretizer.enable_linearization_cache(True)
        >>> # First call computes
        >>> Ad1, Bd1 = discretizer.linearize(x_eq, u_eq)
        >>> # Second call uses cache
        >>> Ad2, Bd2 = discretizer.linearize(x_eq, u_eq)
        """
        self._cache_enabled = enable
        if not enable:
            self._linearization_cache.clear()
    
    def reset_cache(self):
        """Clear the linearization cache."""
        self._linearization_cache.clear()
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get comprehensive discretizer information.
        
        Returns
        -------
        dict
            Information about discretizer configuration
        
        Examples
        --------
        >>> info = discretizer.get_info()
        >>> print(info['method'])
        'rk4'
        >>> print(info['dt'])
        0.01
        """
        return {
            'continuous_system': self.continuous_system.__class__.__name__,
            'dt': self.dt,
            'method': self.method,
            'backend': self.backend,
            'integrator': self.integrator.name,
            'dimensions': {
                'nx': self.nx,
                'nu': self.nu,
                'ny': self.ny,
            },
            'order': self.order,
            'is_autonomous': self.nu == 0,
        }
    
    def __repr__(self) -> str:
        return (
            f"Discretizer({self.continuous_system.__class__.__name__}, "
            f"dt={self.dt}, method={self.method}, backend={self.backend})"
        )
    
    def __str__(self) -> str:
        autonomous_str = " (autonomous)" if self.nu == 0 else ""
        return (
            f"Discretizer: {self.continuous_system.__class__.__name__} → "
            f"dt={self.dt:.4f}, {self.method}{autonomous_str}"
        )