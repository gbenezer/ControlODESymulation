"""
Integrator Base - Abstract Interface for Numerical Integration

Provides a unified interface for numerical integration methods across
different backends (NumPy, PyTorch, JAX) with both fixed and adaptive
time stepping.

This module defines the abstract base class that all integrators must implement,
along with the StepMode enum for specifying integration behavior.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Callable, Dict, Any, TYPE_CHECKING
from enum import Enum
import numpy as np

if TYPE_CHECKING:
    from src.systems.base.symbolic_dynamical_system import SymbolicDynamicalSystem
    import torch
    import jax.numpy as jnp

# Type alias for backend-agnostic arrays
from typing import Union
ArrayLike = Union[np.ndarray, "torch.Tensor", "jnp.ndarray"]


class StepMode(Enum):
    """
    Integration step mode.
    
    Attributes
    ----------
    FIXED : str
        Fixed time step - integrator uses constant dt
        Best for: Real-time systems, discrete controllers, simple ODEs
        
    ADAPTIVE : str
        Adaptive time step - integrator adjusts dt based on error estimates
        Best for: Stiff systems, high accuracy requirements, variable dynamics
    """
    FIXED = "fixed"
    ADAPTIVE = "adaptive"


class IntegrationResult:
    """
    Container for integration results.
    
    Stores time points, states, and metadata from integration.
    
    Attributes
    ----------
    t : ArrayLike
        Time points (T,)
    x : ArrayLike  
        State trajectory (T, nx)
    success : bool
        Whether integration succeeded
    message : str
        Status message
    nfev : int
        Number of function evaluations
    nsteps : int
        Number of integration steps taken
    """
    
    def __init__(
        self,
        t: ArrayLike,
        x: ArrayLike,
        success: bool = True,
        message: str = "Integration successful",
        nfev: int = 0,
        nsteps: int = 0,
        **metadata
    ):
        self.t = t
        self.x = x
        self.success = success
        self.message = message
        self.nfev = nfev
        self.nsteps = nsteps
        self.metadata = metadata
    
    def __repr__(self) -> str:
        return (
            f"IntegrationResult(success={self.success}, "
            f"nsteps={self.nsteps}, nfev={self.nfev})"
        )


class IntegratorBase(ABC):
    """
    Abstract base class for numerical integrators.
    
    Provides a unified interface for integrating continuous-time dynamical
    systems with multiple backends and both fixed/adaptive step sizes.
    
    All integrators must implement:
    - step(): Single integration step
    - integrate(): Multi-step integration over interval
    - name: Integrator name for display
    
    Subclasses handle backend-specific implementations for NumPy, PyTorch, JAX.
    
    Examples
    --------
    >>> # Create integrator
    >>> integrator = RK4Integrator(system, dt=0.01, backend='numpy')
    >>> 
    >>> # Single step
    >>> x_next = integrator.step(x, u)
    >>> 
    >>> # Multi-step integration
    >>> result = integrator.integrate(
    ...     x0=np.array([1.0, 0.0]),
    ...     u_func=lambda t, x: np.zeros(1),
    ...     t_span=(0.0, 10.0)
    ... )
    >>> t, x_traj = result.t, result.x
    """
    
    def __init__(
        self,
        system: 'SymbolicDynamicalSystem',
        dt: Optional[float] = None,
        step_mode: StepMode = StepMode.FIXED,
        backend: str = 'numpy',
        **options
    ):
        """
        Initialize integrator.
        
        Parameters
        ----------
        system : SymbolicDynamicalSystem
            Continuous-time system to integrate
        dt : Optional[float]
            Time step:
            - FIXED mode: Required, constant step size
            - ADAPTIVE mode: Initial guess, will be adjusted
        step_mode : StepMode
            FIXED or ADAPTIVE stepping
        backend : str
            Backend to use ('numpy', 'torch', 'jax')
        **options : dict
            Integrator-specific options:
            - rtol : float
                Relative tolerance (adaptive only, default: 1e-6)
            - atol : float  
                Absolute tolerance (adaptive only, default: 1e-8)
            - max_steps : int
                Maximum number of steps (adaptive only, default: 10000)
            - min_step : float
                Minimum step size (adaptive only)
            - max_step : float
                Maximum step size (adaptive only)
        
        Raises
        ------
        ValueError
            If FIXED mode specified without dt
            If backend is invalid
        RuntimeError
            If backend is not available
        
        Examples
        --------
        >>> # Fixed-step integrator
        >>> integrator = RK4Integrator(system, dt=0.01, backend='numpy')
        >>> 
        >>> # Adaptive integrator
        >>> integrator = ScipyIntegrator(
        ...     system,
        ...     dt=0.01,  # Initial guess
        ...     step_mode=StepMode.ADAPTIVE,
        ...     backend='numpy',
        ...     rtol=1e-8,
        ...     atol=1e-10
        ... )
        """
        self.system = system
        self.dt = dt
        self.step_mode = step_mode
        self.backend = backend
        self.options = options
        
        # Validate
        if step_mode == StepMode.FIXED and dt is None:
            raise ValueError(
                "Time step dt is required for FIXED step mode. "
                "Specify dt in constructor."
            )
        
        if step_mode == StepMode.ADAPTIVE and dt is None:
            # Provide reasonable default initial guess
            self.dt = 0.01
        
        # Validate backend
        valid_backends = ['numpy', 'torch', 'jax']
        if backend not in valid_backends:
            raise ValueError(
                f"Invalid backend '{backend}'. Must be one of {valid_backends}"
            )
        
        # Extract common options
        self.rtol = options.get('rtol', 1e-6)
        self.atol = options.get('atol', 1e-8)
        self.max_steps = options.get('max_steps', 10000)
        
        # Statistics
        self._stats = {
            'total_steps': 0,
            'total_fev': 0,  # Function evaluations
            'total_time': 0.0,
        }
    
    @abstractmethod
    def step(
        self,
        x: ArrayLike,
        u: ArrayLike,
        dt: Optional[float] = None
    ) -> ArrayLike:
        """
        Take one integration step: x(t) → x(t + dt).
        
        Parameters
        ----------
        x : ArrayLike
            Current state (nx,) or (batch, nx)
        u : ArrayLike
            Control input (nu,) or (batch, nu)
        dt : Optional[float]
            Step size (uses self.dt if None)
            
        Returns
        -------
        ArrayLike
            Next state x(t + dt), same shape and type as input
            
        Notes
        -----
        For fixed-step integrators, dt should match self.dt.
        For adaptive integrators, dt may be adjusted internally.
        
        Examples
        --------
        >>> x = np.array([1.0, 0.0])
        >>> u = np.array([0.5])
        >>> x_next = integrator.step(x, u)
        >>> x_next.shape
        (2,)
        """
        pass
    
    @abstractmethod
    def integrate(
        self,
        x0: ArrayLike,
        u_func: Callable[[float, ArrayLike], ArrayLike],
        t_span: Tuple[float, float],
        t_eval: Optional[ArrayLike] = None,
        dense_output: bool = False
    ) -> IntegrationResult:
        """
        Integrate over time interval with control policy.
        
        Parameters
        ----------
        x0 : ArrayLike
            Initial state (nx,)
        u_func : Callable[[float, ArrayLike], ArrayLike]
            Control policy: (t, x) → u
            Can be:
            - Constant control: lambda t, x: u_const
            - State feedback: lambda t, x: -K @ x
            - Time-varying: lambda t, x: u(t)
        t_span : Tuple[float, float]
            Integration interval (t_start, t_end)
        t_eval : Optional[ArrayLike]
            Specific times at which to store solution
            If None:
            - FIXED mode: Uses t = t_start + k*dt for k=0,1,2,...
            - ADAPTIVE mode: Uses solver's internal time points
        dense_output : bool
            If True, return dense interpolated solution (adaptive only)
            
        Returns
        -------
        IntegrationResult
            Object containing:
            - t: Time points (T,)
            - x: State trajectory (T, nx)
            - success: Whether integration succeeded
            - nfev: Number of function evaluations
            - nsteps: Number of steps taken
            
        Raises
        ------
        RuntimeError
            If integration fails (e.g., step size too small, max steps exceeded)
        
        Examples
        --------
        >>> # Zero control
        >>> result = integrator.integrate(
        ...     x0=np.array([1.0, 0.0]),
        ...     u_func=lambda t, x: np.zeros(1),
        ...     t_span=(0.0, 10.0)
        ... )
        >>> 
        >>> # State feedback controller
        >>> K = np.array([[1.0, 2.0]])
        >>> result = integrator.integrate(
        ...     x0=x0,
        ...     u_func=lambda t, x: -K @ x,
        ...     t_span=(0.0, 10.0)
        ... )
        >>> 
        >>> # Evaluate at specific times
        >>> t_eval = np.linspace(0, 10, 1001)
        >>> result = integrator.integrate(x0, u_func, (0, 10), t_eval=t_eval)
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """
        Get integrator name for display and logging.
        
        Returns
        -------
        str
            Human-readable integrator name
            
        Examples
        --------
        >>> integrator.name
        'RK4 (Fixed Step)'
        >>> adaptive_integrator.name
        'scipy.RK45 (Adaptive)'
        """
        pass
    
    # ========================================================================
    # Common Utilities (Shared by All Integrators)
    # ========================================================================
    
    def _evaluate_dynamics(
        self,
        x: ArrayLike,
        u: ArrayLike
    ) -> ArrayLike:
        """
        Evaluate system dynamics with statistics tracking.
        
        Parameters
        ----------
        x : ArrayLike
            State
        u : ArrayLike
            Control
            
        Returns
        -------
        ArrayLike
            State derivative dx/dt
            
        Notes
        -----
        This wrapper counts function evaluations for performance analysis.
        """
        self._stats['total_fev'] += 1
        return self.system(x, u, backend=self.backend)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get integration statistics.
        
        Returns
        -------
        dict
            Statistics with keys:
            - 'total_steps': Total integration steps taken
            - 'total_fev': Total function evaluations
            - 'total_time': Total integration time
            - 'avg_fev_per_step': Average function evaluations per step
            
        Examples
        --------
        >>> result = integrator.integrate(x0, u_func, (0, 10))
        >>> stats = integrator.get_stats()
        >>> print(f"Steps: {stats['total_steps']}")
        >>> print(f"Function evals: {stats['total_fev']}")
        >>> print(f"Evals/step: {stats['avg_fev_per_step']:.1f}")
        """
        avg_fev = (
            self._stats['total_fev'] / max(1, self._stats['total_steps'])
        )
        
        return {
            **self._stats,
            'avg_fev_per_step': avg_fev,
        }
    
    def reset_stats(self):
        """
        Reset integration statistics to zero.
        
        Examples
        --------
        >>> integrator.reset_stats()
        >>> integrator.get_stats()['total_steps']
        0
        """
        self._stats['total_steps'] = 0
        self._stats['total_fev'] = 0
        self._stats['total_time'] = 0.0
    
    def __repr__(self) -> str:
        """String representation for debugging"""
        return (
            f"{self.__class__.__name__}("
            f"dt={self.dt}, mode={self.step_mode.value}, "
            f"backend={self.backend})"
        )
    
    def __str__(self) -> str:
        """Human-readable string"""
        return f"{self.name} (dt={self.dt:.4f}, {self.backend})"