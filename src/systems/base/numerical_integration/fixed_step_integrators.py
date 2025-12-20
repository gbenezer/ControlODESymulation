"""
Fixed-Step Integrators

Implements classic fixed time-step integration methods:
- Explicit Euler (1st order)
- Midpoint/RK2 (2nd order)
- RK4 (4th order)

These are manual implementations that work across all backends
(NumPy, PyTorch, JAX) using the system's multi-backend interface.
"""

import time
import numpy as np
from typing import Optional, Tuple, Callable, TYPE_CHECKING

from src.systems.integration.integrator_base import (
    IntegratorBase,
    StepMode,
    IntegrationResult,
    ArrayLike
)

if TYPE_CHECKING:
    from src.systems.base.symbolic_dynamical_system import SymbolicDynamicalSystem


class ExplicitEulerIntegrator(IntegratorBase):
    """
    Explicit Euler integrator (Forward Euler).
    
    First-order method: x_{k+1} = x_k + dt * f(x_k, u_k)
    
    Characteristics:
    - Order: 1 (error ∝ dt)
    - Stability: Conditionally stable (small dt required)
    - Simplicity: Easiest to understand and implement
    - Performance: Fastest per step, but needs many steps
    
    Best for:
    - Prototyping and debugging
    - Very smooth dynamics
    - Real-time systems where speed > accuracy
    
    Not recommended for:
    - Stiff systems (requires very small dt)
    - High accuracy requirements
    - Production simulations
    
    Examples
    --------
    >>> integrator = ExplicitEulerIntegrator(system, dt=0.01, backend='numpy')
    >>> x_next = integrator.step(x, u)
    >>> 
    >>> # Integrate trajectory
    >>> result = integrator.integrate(
    ...     x0=np.array([1.0, 0.0]),
    ...     u_func=lambda t, x: np.zeros(1),
    ...     t_span=(0.0, 10.0)
    ... )
    """
    
    def __init__(
        self,
        system: 'SymbolicDynamicalSystem',
        dt: float,
        backend: str = 'numpy',
        **options
    ):
        """
        Initialize Explicit Euler integrator.
        
        Parameters
        ----------
        system : SymbolicDynamicalSystem
            System to integrate
        dt : float
            Fixed time step
        backend : str
            Backend ('numpy', 'torch', 'jax')
        """
        super().__init__(system, dt, StepMode.FIXED, backend, **options)
    
    def step(
        self,
        x: ArrayLike,
        u: ArrayLike,
        dt: Optional[float] = None
    ) -> ArrayLike:
        """
        Take one Euler step: x_{k+1} = x_k + dt * f(x_k, u_k).
        
        Parameters
        ----------
        x : ArrayLike
            Current state
        u : ArrayLike
            Control input
        dt : Optional[float]
            Time step (uses self.dt if None)
            
        Returns
        -------
        ArrayLike
            Next state
        """
        dt = dt if dt is not None else self.dt
        
        # Evaluate dynamics
        dx = self._evaluate_dynamics(x, u)
        
        # Euler update
        x_next = x + dt * dx
        
        self._stats['total_steps'] += 1
        
        return x_next
    
    def integrate(
        self,
        x0: ArrayLike,
        u_func: Callable[[float, ArrayLike], ArrayLike],
        t_span: Tuple[float, float],
        t_eval: Optional[ArrayLike] = None,
        dense_output: bool = False
    ) -> IntegrationResult:
        """
        Integrate using fixed Euler steps.
        
        Parameters
        ----------
        x0 : ArrayLike
            Initial state
        u_func : Callable
            Control policy (t, x) → u
        t_span : Tuple[float, float]
            (t_start, t_end)
        t_eval : Optional[ArrayLike]
            Specific times to evaluate (if None, uses uniform grid)
        dense_output : bool
            Ignored for fixed-step methods
            
        Returns
        -------
        IntegrationResult
            Integration results
        """
        start_time = time.time()
        
        t0, tf = t_span
        
        # Create time grid
        if t_eval is None:
            num_steps = int(np.ceil((tf - t0) / self.dt))
            t_eval = np.linspace(t0, tf, num_steps + 1)
        
        # Convert to appropriate backend
        if self.backend == 'numpy':
            t_points = np.asarray(t_eval)
        elif self.backend == 'torch':
            import torch
            t_points = torch.as_tensor(t_eval) if not isinstance(t_eval, torch.Tensor) else t_eval
        elif self.backend == 'jax':
            import jax.numpy as jnp
            t_points = jnp.asarray(t_eval)
        
        # Initialize trajectory storage
        trajectory = [x0]
        x = x0
        
        # Integrate
        for i in range(len(t_points) - 1):
            t = t_points[i]
            dt_step = float(t_points[i+1] - t_points[i])
            
            u = u_func(float(t), x)
            x = self.step(x, u, dt=dt_step)
            trajectory.append(x)
        
        # Stack trajectory
        if self.backend == 'numpy':
            x_traj = np.stack(trajectory)
        elif self.backend == 'torch':
            import torch
            x_traj = torch.stack(trajectory)
        elif self.backend == 'jax':
            import jax.numpy as jnp
            x_traj = jnp.stack(trajectory)
        
        elapsed = time.time() - start_time
        self._stats['total_time'] += elapsed
        
        return IntegrationResult(
            t=t_points,
            x=x_traj,
            success=True,
            nfev=self._stats['total_fev'],
            nsteps=len(t_points) - 1
        )
    
    @property
    def name(self) -> str:
        return "Explicit Euler"


class MidpointIntegrator(IntegratorBase):
    """
    Midpoint integrator (RK2).
    
    Second-order method using midpoint evaluation.
    
    Algorithm:
        k1 = f(x_k, u_k)
        k2 = f(x_k + 0.5*dt*k1, u_k)
        x_{k+1} = x_k + dt * k2
    
    Characteristics:
    - Order: 2 (error ∝ dt²)
    - Stability: Better than Euler, still conditional
    - Function evaluations: 2 per step
    - Accuracy: Good balance of speed and accuracy
    
    Best for:
    - General-purpose simulation
    - Moderate accuracy requirements
    - When RK4 is overkill
    
    Examples
    --------
    >>> integrator = MidpointIntegrator(system, dt=0.01, backend='torch')
    >>> x_next = integrator.step(x_torch, u_torch)
    """
    
    def __init__(
        self,
        system: 'SymbolicDynamicalSystem',
        dt: float,
        backend: str = 'numpy',
        **options
    ):
        super().__init__(system, dt, StepMode.FIXED, backend, **options)
    
    def step(
        self,
        x: ArrayLike,
        u: ArrayLike,
        dt: Optional[float] = None
    ) -> ArrayLike:
        """
        Take one midpoint step.
        
        Uses two function evaluations per step for 2nd-order accuracy.
        """
        dt = dt if dt is not None else self.dt
        
        # Stage 1: Evaluate at current point
        k1 = self._evaluate_dynamics(x, u)
        
        # Stage 2: Evaluate at midpoint
        x_mid = x + 0.5 * dt * k1
        k2 = self._evaluate_dynamics(x_mid, u)
        
        # Update using midpoint derivative
        x_next = x + dt * k2
        
        self._stats['total_steps'] += 1
        
        return x_next
    
    def integrate(
        self,
        x0: ArrayLike,
        u_func: Callable[[float, ArrayLike], ArrayLike],
        t_span: Tuple[float, float],
        t_eval: Optional[ArrayLike] = None,
        dense_output: bool = False
    ) -> IntegrationResult:
        """Integrate using fixed midpoint steps."""
        start_time = time.time()
        
        t0, tf = t_span
        
        # Create time grid
        if t_eval is None:
            num_steps = int(np.ceil((tf - t0) / self.dt))
            t_eval = np.linspace(t0, tf, num_steps + 1)
        
        # Convert to backend
        if self.backend == 'numpy':
            t_points = np.asarray(t_eval)
        elif self.backend == 'torch':
            import torch
            t_points = torch.as_tensor(t_eval) if not isinstance(t_eval, torch.Tensor) else t_eval
        elif self.backend == 'jax':
            import jax.numpy as jnp
            t_points = jnp.asarray(t_eval)
        
        # Initialize
        trajectory = [x0]
        x = x0
        
        # Integrate
        for i in range(len(t_points) - 1):
            t = t_points[i]
            dt_step = float(t_points[i+1] - t_points[i])
            
            u = u_func(float(t), x)
            x = self.step(x, u, dt=dt_step)
            trajectory.append(x)
        
        # Stack
        if self.backend == 'numpy':
            x_traj = np.stack(trajectory)
        elif self.backend == 'torch':
            import torch
            x_traj = torch.stack(trajectory)
        elif self.backend == 'jax':
            import jax.numpy as jnp
            x_traj = jnp.stack(trajectory)
        
        elapsed = time.time() - start_time
        self._stats['total_time'] += elapsed
        
        return IntegrationResult(
            t=t_points,
            x=x_traj,
            success=True,
            nfev=self._stats['total_fev'],
            nsteps=len(t_points) - 1
        )
    
    @property
    def name(self) -> str:
        return "Midpoint (RK2)"


class RK4Integrator(IntegratorBase):
    """
    Classic 4th-order Runge-Kutta integrator.
    
    Fourth-order method with excellent accuracy/cost trade-off.
    
    Algorithm:
        k1 = f(x_k, u_k)
        k2 = f(x_k + 0.5*dt*k1, u_k)
        k3 = f(x_k + 0.5*dt*k2, u_k)
        k4 = f(x_k + dt*k3, u_k)
        x_{k+1} = x_k + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
    
    Characteristics:
    - Order: 4 (error ∝ dt⁴)
    - Stability: Good stability region
    - Function evaluations: 4 per step
    - Accuracy: Excellent for smooth dynamics
    
    Best for:
    - General-purpose simulation
    - Smooth, non-stiff systems
    - When accuracy matters more than speed
    - Production simulations
    
    Not recommended for:
    - Stiff systems (use BDF/Radau instead)
    - Very simple dynamics (Euler/Midpoint faster)
    - Real-time systems (too expensive per step)
    
    Examples
    --------
    >>> # NumPy backend
    >>> integrator = RK4Integrator(system, dt=0.01, backend='numpy')
    >>> 
    >>> # PyTorch backend (GPU-capable)
    >>> integrator = RK4Integrator(system, dt=0.01, backend='torch')
    >>> x_torch = torch.tensor([1.0, 0.0], device='cuda')
    >>> u_torch = torch.tensor([0.0], device='cuda')
    >>> x_next = integrator.step(x_torch, u_torch)
    """
    
    def __init__(
        self,
        system: 'SymbolicDynamicalSystem',
        dt: float,
        backend: str = 'numpy',
        **options
    ):
        """
        Initialize RK4 integrator.
        
        Parameters
        ----------
        system : SymbolicDynamicalSystem
            System to integrate
        dt : float
            Fixed time step
        backend : str
            Backend ('numpy', 'torch', 'jax')
        """
        super().__init__(system, dt, StepMode.FIXED, backend, **options)
    
    def step(
        self,
        x: ArrayLike,
        u: ArrayLike,
        dt: Optional[float] = None
    ) -> ArrayLike:
        """
        Take one RK4 step using four function evaluations.
        
        Parameters
        ----------
        x : ArrayLike
            Current state
        u : ArrayLike
            Control input (assumed constant over step)
        dt : Optional[float]
            Time step (uses self.dt if None)
            
        Returns
        -------
        ArrayLike
            Next state after RK4 step
            
        Notes
        -----
        Assumes control is constant over the time step. For time-varying
        control, use integrate() with a control policy u(t, x).
        """
        dt = dt if dt is not None else self.dt
        
        # RK4 stages
        k1 = self._evaluate_dynamics(x, u)
        k2 = self._evaluate_dynamics(x + 0.5 * dt * k1, u)
        k3 = self._evaluate_dynamics(x + 0.5 * dt * k2, u)
        k4 = self._evaluate_dynamics(x + dt * k3, u)
        
        # Weighted combination
        x_next = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        self._stats['total_steps'] += 1
        
        return x_next
    
    def integrate(
        self,
        x0: ArrayLike,
        u_func: Callable[[float, ArrayLike], ArrayLike],
        t_span: Tuple[float, float],
        t_eval: Optional[ArrayLike] = None,
        dense_output: bool = False
    ) -> IntegrationResult:
        """
        Integrate using fixed RK4 steps.
        
        Parameters
        ----------
        x0 : ArrayLike
            Initial state (nx,)
        u_func : Callable
            Control policy (t, x) → u
        t_span : Tuple[float, float]
            Integration interval (t_start, t_end)
        t_eval : Optional[ArrayLike]
            Evaluation times (if None, uses uniform grid with dt)
        dense_output : bool
            Ignored (fixed-step methods don't support dense output)
            
        Returns
        -------
        IntegrationResult
            Contains time points, trajectory, and statistics
            
        Examples
        --------
        >>> # Integrate with zero control
        >>> result = integrator.integrate(
        ...     x0=np.array([1.0, 0.0]),
        ...     u_func=lambda t, x: np.zeros(1),
        ...     t_span=(0.0, 10.0)
        ... )
        >>> print(f"Solved {result.nsteps} steps")
        >>> print(f"Function evaluations: {result.nfev}")
        """
        start_time = time.time()
        
        t0, tf = t_span
        
        # Create time grid
        if t_eval is None:
            num_steps = int(np.ceil((tf - t0) / self.dt))
            t_eval = np.linspace(t0, tf, num_steps + 1)
        
        # Convert to backend
        if self.backend == 'numpy':
            t_points = np.asarray(t_eval)
        elif self.backend == 'torch':
            import torch
            t_points = torch.as_tensor(t_eval) if not isinstance(t_eval, torch.Tensor) else t_eval
        elif self.backend == 'jax':
            import jax.numpy as jnp
            t_points = jnp.asarray(t_eval)
        
        # Initialize storage
        trajectory = [x0]
        x = x0
        
        # Integration loop
        for i in range(len(t_points) - 1):
            t = float(t_points[i])
            dt_step = float(t_points[i+1] - t_points[i])
            
            # Evaluate control
            u = u_func(t, x)
            
            # Take RK4 step
            x = self.step(x, u, dt=dt_step)
            trajectory.append(x)
        
        # Stack trajectory
        if self.backend == 'numpy':
            x_traj = np.stack(trajectory)
        elif self.backend == 'torch':
            import torch
            x_traj = torch.stack(trajectory)
        elif self.backend == 'jax':
            import jax.numpy as jnp
            x_traj = jnp.stack(trajectory)
        
        elapsed = time.time() - start_time
        self._stats['total_time'] += elapsed
        
        return IntegrationResult(
            t=t_points,
            x=x_traj,
            success=True,
            message="RK4 integration completed",
            nfev=self._stats['total_fev'],
            nsteps=len(t_points) - 1,
            integration_time=elapsed
        )
    
    @property
    def name(self) -> str:
        return "RK4 (Classic)"


# ============================================================================
# Utility: Quick Integrator Creation
# ============================================================================

def create_fixed_step_integrator(
    method: str,
    system: 'SymbolicDynamicalSystem',
    dt: float,
    backend: str = 'numpy'
) -> IntegratorBase:
    """
    Quick factory for fixed-step integrators.
    
    Parameters
    ----------
    method : str
        'euler', 'midpoint', or 'rk4'
    system : SymbolicDynamicalSystem
        System to integrate
    dt : float
        Time step
    backend : str
        Backend to use
        
    Returns
    -------
    IntegratorBase
        Configured integrator
        
    Examples
    --------
    >>> integrator = create_fixed_step_integrator('rk4', system, dt=0.01)
    """
    method_map = {
        'euler': ExplicitEulerIntegrator,
        'midpoint': MidpointIntegrator,
        'rk4': RK4Integrator,
    }
    
    if method not in method_map:
        raise ValueError(
            f"Unknown method '{method}'. Choose from: {list(method_map.keys())}"
        )
    
    integrator_class = method_map[method]
    return integrator_class(system, dt, backend)