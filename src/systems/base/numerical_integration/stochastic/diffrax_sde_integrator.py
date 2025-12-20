"""
DiffraxSDEIntegrator: JAX-based SDE integration using Diffrax library.

Supports both Itô and Stratonovich SDEs with various specialized solvers
for additive, diagonal, and general noise structures.
"""

from typing import Optional, Callable, Tuple
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import Array
import diffrax as dfx

from src.systems.base.numerical_integration.integrator_base import (
    IntegratorBase,
    IntegrationResult,
    StepMode,
    ArrayLike
)
from src.systems.base.stochastic_dynamical_system import StochasticDynamicalSystem


class DiffraxSDEIntegrator(IntegratorBase):
    """
    JAX-based SDE integrator using the Diffrax library.
    
    Supports Itô and Stratonovich SDEs with specialized solvers for
    different noise types (additive, diagonal, general).
    
    Parameters
    ----------
    system : StochasticDynamicalSystem
        Stochastic system to integrate
    dt : Optional[float]
        Initial time step
    backend : str
        Must be 'jax'
    sde_solver : str, optional
        SDE solver. Options:
        - For general noise: 'euler_heun', 'spark', 'general_shark'
        - For additive noise: 'sea', 'shark', 'sra1'
        Default: 'euler_heun'
    levy_area : str, optional
        Lévy area approximation: 'space-time', 'davie', None
        Default: None
    **options
        Additional options
    
    Examples
    --------
    >>> system = StochasticDynamicalSystem(drift, diffusion, ...)
    >>> integrator = DiffraxSDEIntegrator(
    ...     system,
    ...     dt=0.01,
    ...     backend='jax',
    ...     sde_solver='euler_heun'
    ... )
    >>> result = integrator.integrate_sde(
    ...     x0, u_func, (0, 10), noise_seed=42
    ... )
    """
    
    def __init__(
        self,
        system: StochasticDynamicalSystem,
        dt: Optional[float] = None,
        step_mode: StepMode = StepMode.FIXED,
        backend: str = 'jax',
        sde_solver: str = 'euler_heun',
        levy_area: Optional[str] = None,
        **options
    ):
        if backend != 'jax':
            raise ValueError(
                f"DiffraxSDEIntegrator requires backend='jax', got '{backend}'"
            )
        
        if not isinstance(system, StochasticDynamicalSystem):
            raise TypeError(
                "System must be StochasticDynamicalSystem for SDE integration"
            )
        
        super().__init__(system, dt, step_mode, backend, **options)
        
        self.sde_solver_name = sde_solver
        self.levy_area = levy_area
        self._integrator_name = f"Diffrax-SDE-{sde_solver}"
        
        # Map SDE solver names
        self._sde_solver_map = {
            # General noise (Itô)
            'euler': dfx.Euler,
            'euler_heun': dfx.EulerHeun,
            'heun': dfx.Heun,
            'midpoint': dfx.Midpoint,
            
            # General noise (Stratonovich)
            'spark': dfx.SPaRK,
            'general_shark': dfx.GeneralShARK,
            'slow_rk': dfx.SlowRK,
            'sra1': dfx.SRA1,
            
            # Additive noise (efficient)
            'sea': dfx.SEA,
            'shark': dfx.ShARK,
            
            # Reversible
            'reversible_heun': dfx.ReversibleHeun,
        }
        
        if self.sde_solver_name not in self._sde_solver_map:
            raise ValueError(
                f"Unknown SDE solver '{sde_solver}'. "
                f"Available: {list(self._sde_solver_map.keys())}"
            )
    
    @property
    def name(self) -> str:
        """Return integrator name."""
        return f"{self._integrator_name} ({self.system.sde_type})"
    
    def step(self, x: ArrayLike, u: ArrayLike, dt: Optional[float] = None) -> ArrayLike:
        """
        Single SDE step (uses integrate_sde internally).
        
        Note: Each call generates new noise realization.
        """
        step_size = dt if dt is not None else self.dt
        if step_size is None:
            raise ValueError("Step size dt must be specified")
        
        # Use random seed based on step count for reproducibility
        noise_seed = self._stats.get('total_steps', 0)
        
        result = self.integrate_sde(
            x, lambda t, x: u, (0.0, step_size),
            noise_seed=noise_seed,
            t_eval=jnp.array([step_size])
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
        """
        Integrate SDE with default noise seed.
        
        For multiple realizations, use integrate_sde() with different seeds.
        """
        return self.integrate_sde(
            x0, u_func, t_span,
            noise_seed=0,
            t_eval=t_eval
        )
    
    def integrate_sde(
        self,
        x0: ArrayLike,
        u_func: Callable[[float, ArrayLike], ArrayLike],
        t_span: Tuple[float, float],
        noise_seed: int = 0,
        t_eval: Optional[ArrayLike] = None,
        **kwargs
    ) -> IntegrationResult:
        """
        Integrate stochastic differential equation.
        
        Parameters
        ----------
        x0 : ArrayLike
            Initial state (nx,)
        u_func : Callable
            Control policy (t, x) → u
        t_span : Tuple[float, float]
            Time interval (t0, tf)
        noise_seed : int
            Random seed for Brownian motion
        t_eval : ArrayLike, optional
            Evaluation times
        
        Returns
        -------
        IntegrationResult
            SDE solution (one realization)
        
        Examples
        --------
        >>> # Single realization
        >>> result = integrator.integrate_sde(x0, u_func, (0, 10), noise_seed=42)
        >>> 
        >>> # Monte Carlo (multiple realizations)
        >>> results = [
        ...     integrator.integrate_sde(x0, u_func, (0, 10), noise_seed=i)
        ...     for i in range(100)
        ... ]
        """
        t0, tf = t_span
        x0 = jnp.asarray(x0)
        
        # Handle edge cases
        if t0 == tf:
            return IntegrationResult(
                t=jnp.array([t0]),
                x=x0[None, :] if x0.ndim == 1 else x0,
                success=True,
                message="Zero time span",
                nfev=0,
                nsteps=0,
            )
        
        # Define drift term
        def drift_func(t, state, args):
            u = u_func(t, state)
            return self.system.drift(state, u, backend='jax')
        
        # Define diffusion term
        def diffusion_func(t, state, args):
            u = u_func(t, state)
            return self.system.diffusion(state, u, backend='jax')
        
        # Create Brownian motion
        brownian_motion = dfx.VirtualBrownianTree(
            t0=t0,
            t1=tf,
            tol=kwargs.get('brownian_tol', 1e-3),
            shape=(self.system.nw,),
            key=jr.PRNGKey(noise_seed)
        )
        
        # Create terms
        drift_term = dfx.ODETerm(drift_func)
        diffusion_term = dfx.ControlTerm(diffusion_func, brownian_motion)
        terms = dfx.MultiTerm(drift_term, diffusion_term)
        
        # Create solver
        solver = self._sde_solver_map[self.sde_solver_name]()
        
        # Setup time points
        if t_eval is not None:
            t_points = jnp.asarray(t_eval)
            saveat = dfx.SaveAt(ts=t_points)
        else:
            n_points = max(2, int((tf - t0) / self.dt) + 1)
            t_points = jnp.linspace(t0, tf, n_points)
            saveat = dfx.SaveAt(ts=t_points)
        
        # Stepsize controller
        if self.step_mode == StepMode.FIXED:
            stepsize_controller = dfx.ConstantStepSize()
        else:
            stepsize_controller = dfx.PIDController(
                rtol=self.rtol,
                atol=self.atol
            )
        
        # Solve SDE
        try:
            solution = dfx.diffeqsolve(
                terms,
                solver,
                t0=t0,
                t1=tf,
                dt0=self.dt,
                y0=x0,
                saveat=saveat,
                stepsize_controller=stepsize_controller,
                max_steps=self.max_steps,
                throw=False,
            )
            
            success = jnp.all(jnp.isfinite(solution.ys))
            
            nsteps = int(solution.stats.get('num_steps', 0))
            self._stats['total_steps'] += nsteps
            self._stats['total_fev'] += nsteps
            
            return IntegrationResult(
                t=solution.ts,
                x=solution.ys,
                success=bool(success),
                message="SDE integration successful" if success else "SDE integration failed",
                nfev=nsteps,
                nsteps=nsteps,
                solver=self.sde_solver_name,
                noise_seed=noise_seed,
            )
            
        except Exception as e:
            return IntegrationResult(
                t=jnp.array([t0]),
                x=x0[None, :] if x0.ndim == 1 else x0,
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
    ) -> Tuple[Array, Array, Array]:
        """
        Run Monte Carlo simulations with different noise realizations.
        
        Parameters
        ----------
        x0 : ArrayLike
            Initial state
        u_func : Callable
            Control policy
        t_span : Tuple[float, float]
            Time span
        num_samples : int
            Number of Monte Carlo samples
        t_eval : ArrayLike, optional
            Evaluation times
        base_seed : int
            Base random seed
        
        Returns
        -------
        Tuple[Array, Array, Array]
            (time_points, mean_trajectory, std_trajectory)
            - time_points: (T,)
            - mean_trajectory: (T, nx)
            - std_trajectory: (T, nx)
        
        Examples
        --------
        >>> t, mean, std = integrator.monte_carlo_simulation(
        ...     x0, u_func, (0, 10), num_samples=1000
        ... )
        >>> # Plot confidence intervals
        >>> plt.plot(t, mean[:, 0])
        >>> plt.fill_between(t, mean[:, 0] - 2*std[:, 0], 
        ...                     mean[:, 0] + 2*std[:, 0], alpha=0.3)
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
        
        # Stack and compute statistics
        traj_stack = jnp.stack(trajectories)  # (num_samples, T, nx)
        
        mean_traj = jnp.mean(traj_stack, axis=0)  # (T, nx)
        std_traj = jnp.std(traj_stack, axis=0)    # (T, nx)
        
        return result.t, mean_traj, std_traj