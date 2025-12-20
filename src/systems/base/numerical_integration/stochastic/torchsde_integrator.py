"""
TorchSDEIntegrator: PyTorch-based SDE integration using torchsde library.

Provides GPU-accelerated SDE solving with adjoint sensitivity for
Neural SDEs and stochastic control systems.
"""

from typing import Optional, Callable, Tuple
import torch
from torch import Tensor
import torchsde

from src.systems.base.numerical_integration.integrator_base import (
    IntegratorBase,
    IntegrationResult,
    StepMode,
    ArrayLike
)
from src.systems.base.stochastic_dynamical_system import StochasticDynamicalSystem


class TorchSDEIntegrator(IntegratorBase):
    """
    PyTorch-based SDE integrator using the torchsde library.
    
    Supports Itô and Stratonovich SDEs with efficient GPU implementation
    and adjoint sensitivity for Neural SDEs.
    
    Parameters
    ----------
    system : StochasticDynamicalSystem or torch.nn.Module
        Stochastic system to integrate
    dt : Optional[float]
        Initial time step
    backend : str
        Must be 'torch'
    method : str, optional
        SDE solver method:
        - 'euler': Euler-Maruyama (strong order 0.5)
        - 'milstein': Milstein method (strong order 1.0)
        - 'srk': Stochastic Runge-Kutta
        - 'reversible_heun': Reversible Heun
        Default: 'euler'
    adjoint : bool, optional
        Use adjoint method (requires system to be nn.Module)
        Default: False
    **options
        Additional options
    
    Examples
    --------
    >>> # Regular stochastic system
    >>> system = StochasticDynamicalSystem(drift, diffusion, ...)
    >>> integrator = TorchSDEIntegrator(
    ...     system,
    ...     dt=0.01,
    ...     backend='torch',
    ...     method='euler',
    ...     adjoint=False
    ... )
    >>> 
    >>> # Neural SDE
    >>> class NeuralSDE(nn.Module):
    ...     noise_type = 'general'
    ...     sde_type = 'ito'
    ...     
    ...     def f(self, t, y):  # drift
    ...         return self.drift_net(y)
    ...     
    ...     def g(self, t, y):  # diffusion
    ...         return self.diffusion_net(y)
    >>> 
    >>> neural_sde = NeuralSDE()
    >>> integrator = TorchSDEIntegrator(
    ...     neural_sde,
    ...     backend='torch',
    ...     method='srk',
    ...     adjoint=True  # Memory efficient
    ... )
    """
    
    def __init__(
        self,
        system,
        dt: Optional[float] = None,
        step_mode: StepMode = StepMode.ADAPTIVE,
        backend: str = 'torch',
        method: str = 'euler',
        adjoint: bool = False,
        **options
    ):
        if backend != 'torch':
            raise ValueError(
                f"TorchSDEIntegrator requires backend='torch', got '{backend}'"
            )
        
        # Check if system is StochasticDynamicalSystem or nn.Module
        self.is_neural_sde = isinstance(system, torch.nn.Module)
        
        if not self.is_neural_sde and not isinstance(system, StochasticDynamicalSystem):
            raise TypeError(
                "System must be StochasticDynamicalSystem or nn.Module with f/g methods"
            )
        
        super().__init__(system, dt, step_mode, backend, **options)
        
        self.method = method
        self.use_adjoint = adjoint
        self._integrator_name = f"TorchSDE-{method}"
        
        # Available methods
        self.available_methods = ['euler', 'milstein', 'srk', 'reversible_heun']
        
        if method not in self.available_methods:
            raise ValueError(
                f"Unknown method '{method}'. Available: {self.available_methods}"
            )
        
        # Select integration function
        if self.use_adjoint:
            self._sdeint = torchsde.sdeint_adjoint
        else:
            self._sdeint = torchsde.sdeint
        
        # Create wrapper if system is not nn.Module
        if not self.is_neural_sde:
            self._sde_wrapper = self._create_sde_wrapper()
    
    def _create_sde_wrapper(self):
        """Create nn.Module wrapper for non-neural systems."""
        system = self.system
        
        class SDEWrapper(torch.nn.Module):
            """Wraps StochasticDynamicalSystem as nn.Module for torchsde."""
            
            def __init__(self, stochastic_system):
                super().__init__()
                self.system = stochastic_system
                self.noise_type = stochastic_system.noise_type
                self.sde_type = stochastic_system.sde_type
                # Store u_func as instance variable (set before integration)
                self.u_func = None
            
            def f(self, t, y):
                """Drift term."""
                t_val = float(t.item()) if isinstance(t, torch.Tensor) else float(t)
                u = self.u_func(t_val, y)
                return self.system.drift(y, u, backend='torch')
            
            def g(self, t, y):
                """Diffusion term."""
                t_val = float(t.item()) if isinstance(t, torch.Tensor) else float(t)
                u = self.u_func(t_val, y)
                return self.system.diffusion(y, u, backend='torch')
        
        return SDEWrapper(system)
    
    @property
    def name(self) -> str:
        """Return integrator name."""
        adjoint_str = " (Adjoint)" if self.use_adjoint else ""
        return f"{self._integrator_name}{adjoint_str}"
    
    def step(self, x: ArrayLike, u: ArrayLike, dt: Optional[float] = None) -> ArrayLike:
        """Single SDE step."""
        step_size = dt if dt is not None else self.dt
        if step_size is None:
            raise ValueError("Step size dt must be specified")
        
        # Use step count as seed
        noise_seed = self._stats.get('total_steps', 0)
        
        result = self.integrate_sde(
            x,
            lambda t, x: u,
            (0.0, step_size),
            noise_seed=noise_seed
        )
        
        self._stats['total_steps'] += 1
        
        return result.x[-1]
    
    def integrate(
        self,
        x0: ArrayLike,
        u_func: Callable,
        t_span: Tuple[float, float],
        t_eval: Optional[ArrayLike] = None,
        dense_output: bool = False
    ) -> IntegrationResult:
        """Integrate SDE with default seed."""
        return self.integrate_sde(x0, u_func, t_span, noise_seed=0, t_eval=t_eval)
    
    def integrate_sde(
        self,
        x0: ArrayLike,
        u_func: Callable,
        t_span: Tuple[float, float],
        noise_seed: int = 0,
        t_eval: Optional[ArrayLike] = None,
    ) -> IntegrationResult:
        """
        Integrate SDE using torchsde.
        
        Parameters
        ----------
        x0 : ArrayLike
            Initial state
        u_func : Callable
            Control policy
        t_span : Tuple[float, float]
            Time interval
        noise_seed : int
            Random seed for reproducibility
        t_eval : ArrayLike, optional
            Evaluation times
        
        Returns
        -------
        IntegrationResult
            SDE solution
        """
        t0, tf = t_span
        
        # Convert to tensors
        if not isinstance(x0, torch.Tensor):
            x0 = torch.tensor(x0, dtype=torch.float32)
        
        # Handle edge cases
        if t0 == tf:
            return IntegrationResult(
                t=torch.tensor([t0]),
                x=x0.unsqueeze(0) if x0.ndim == 1 else x0,
                success=True,
                message="Zero time span",
                nfev=0,
                nsteps=0,
            )
        
        # Prepare time points
        if t_eval is not None:
            if not isinstance(t_eval, torch.Tensor):
                t_eval = torch.tensor(t_eval, dtype=x0.dtype, device=x0.device)
            t_points = t_eval
        else:
            n_points = max(2, int((tf - t0) / self.dt) + 1)
            t_points = torch.linspace(t0, tf, n_points, dtype=x0.dtype, device=x0.device)
        
        # Setup SDE wrapper
        if self.is_neural_sde:
            sde = self.system
        else:
            self._sde_wrapper.u_func = u_func
            sde = self._sde_wrapper
        
        # Set random seed
        torch.manual_seed(noise_seed)
        
        # Integrate
        try:
            with torchsde.brownian_interval(
                t0=t0,
                t1=tf,
                size=(self.system.nw,) if hasattr(self.system, 'nw') else (x0.shape[0],),
                dtype=x0.dtype,
                device=x0.device
            ):
                ys = self._sdeint(
                    sde,
                    x0,
                    t_points,
                    method=self.method,
                    dt=self.dt,
                    rtol=self.rtol if self.step_mode == StepMode.ADAPTIVE else None,
                    atol=self.atol if self.step_mode == StepMode.ADAPTIVE else None,
                )
            
            success = torch.all(torch.isfinite(ys)).item()
            
            nsteps = len(t_points) - 1
            self._stats['total_steps'] += nsteps
            self._stats['total_fev'] += nsteps
            
            return IntegrationResult(
                t=t_points,
                x=ys,
                success=success,
                message="SDE integration successful" if success else "SDE failed",
                nfev=nsteps,
                nsteps=nsteps,
                method=self.method,
                noise_seed=noise_seed,
            )
            
        except Exception as e:
            return IntegrationResult(
                t=torch.tensor([t0]),
                x=x0.unsqueeze(0) if x0.ndim == 1 else x0,
                success=False,
                message=f"SDE integration failed: {str(e)}",
                nfev=0,
                nsteps=0,
            )
    
    def monte_carlo_simulation(
        self,
        x0: ArrayLike,
        u_func: Callable,
        t_span: Tuple[float, float],
        num_samples: int = 100,
        t_eval: Optional[ArrayLike] = None,
        base_seed: int = 0,
    ):
        """
        Run Monte Carlo simulations.
        
        Returns
        -------
        Tuple[Tensor, Tensor, Tensor]
            (time_points, mean_trajectory, std_trajectory)
        """
        trajectories = []
        
        for i in range(num_samples):
            result = self.integrate_sde(
                x0, u_func, t_span,
                noise_seed=base_seed + i,
                t_eval=t_eval
            )
            if result.success:
                trajectories.append(result.x)
        
        traj_stack = torch.stack(trajectories)  # (num_samples, T, nx)
        
        mean_traj = torch.mean(traj_stack, dim=0)
        std_traj = torch.std(traj_stack, dim=0)
        
        return result.t, mean_traj, std_traj