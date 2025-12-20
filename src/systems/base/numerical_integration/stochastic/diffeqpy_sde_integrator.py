"""
DiffEqPySDEIntegrator: Julia DifferentialEquations.jl SDE solver via diffeqpy.

Provides access to Julia's comprehensive SDE solver suite with high-order
methods and adaptive time-stepping.
"""

from typing import Optional, Callable, Tuple, Dict, Any
import numpy as np

from src.systems.base.numerical_integration.integrator_base import (
    IntegratorBase,
    IntegrationResult,
    StepMode,
    ArrayLike
)
from src.systems.base.stochastic_dynamical_system import StochasticDynamicalSystem


class DiffEqPySDEIntegrator(IntegratorBase):
    """
    SDE integrator using Julia's DifferentialEquations.jl via diffeqpy.
    
    Accesses the most comprehensive SDE solver suite available:
    - Low-order: EM (Euler-Maruyama), RKMil
    - High-order: SRI, SRIW1, SRIW2, SRA, SRA1, SRA2, SRA3
    - Adaptive: All solvers support adaptive stepping
    - Commutative noise: Optimized methods
    
    Parameters
    ----------
    system : StochasticDynamicalSystem
        Stochastic system to integrate
    dt : Optional[float]
        Initial time step
    backend : str
        Must be 'numpy' (Julia arrays → NumPy)
    algorithm : str, optional
        Julia SDE algorithm. Default: 'SOSRI'
        Recommendations:
        - General: 'SOSRI', 'SRI', 'SRIW1'
        - Commutative: 'SRA1', 'SRA2'
        - Simple: 'EM' (Euler-Maruyama)
        - High accuracy: 'SRA3', 'SRIW2'
    noise_rate_prototype : Optional[np.ndarray]
        Prototype for noise term shape (nx, nw)
        If None, inferred from system
    **options
        Additional Julia solver options
    
    Examples
    --------
    >>> system = StochasticDynamicalSystem(drift, diffusion, ...)
    >>> 
    >>> # Default (adaptive high-order)
    >>> integrator = DiffEqPySDEIntegrator(
    ...     system,
    ...     backend='numpy',
    ...     algorithm='SOSRI'
    ... )
    >>> 
    >>> # High accuracy
    >>> integrator = DiffEqPySDEIntegrator(
    ...     system,
    ...     algorithm='SRA3',
    ...     reltol=1e-10,
    ...     abstol=1e-12
    ... )
    """
    
    def __init__(
        self,
        system: StochasticDynamicalSystem,
        dt: Optional[float] = None,
        step_mode: StepMode = StepMode.ADAPTIVE,
        backend: str = 'numpy',
        algorithm: str = 'SOSRI',
        noise_rate_prototype: Optional[np.ndarray] = None,
        **options
    ):
        if backend != 'numpy':
            raise ValueError(
                f"DiffEqPySDEIntegrator requires backend='numpy', got '{backend}'"
            )
        
        if not isinstance(system, StochasticDynamicalSystem):
            raise TypeError(
                "System must be StochasticDynamicalSystem for SDE integration"
            )
        
        super().__init__(system, dt, step_mode, backend, **options)
        
        self.algorithm = algorithm
        self._integrator_name = f"DiffEqPy-SDE-{algorithm}"
        
        # Noise prototype
        if noise_rate_prototype is None:
            # Default: (nx, nw) matrix
            self.noise_rate_prototype = np.zeros((system.nx, system.nw))
        else:
            self.noise_rate_prototype = noise_rate_prototype
        
        # Import diffeqpy
        try:
            from diffeqpy import de
            self.de = de
        except ImportError:
            raise ImportError(
                "diffeqpy is required for DiffEqPySDEIntegrator.\n"
                "Install Julia, then:\n"
                "  julia> using Pkg\n"
                "  julia> Pkg.add('DifferentialEquations')\n"
                "Then install Python package:\n"
                "  pip install diffeqpy"
            )
    
    @property
    def name(self) -> str:
        """Return integrator name."""
        return f"{self._integrator_name} ({self.system.sde_type})"
    
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
        Integrate SDE using Julia's solvers.
        
        Parameters
        ----------
        x0 : ArrayLike
            Initial state
        u_func : Callable
            Control policy
        t_span : Tuple[float, float]
            Time interval
        noise_seed : int
            Random seed
        t_eval : ArrayLike, optional
            Evaluation times
        
        Returns
        -------
        IntegrationResult
            SDE solution
        """
        t0, tf = t_span
        x0 = np.asarray(x0)
        
        if t0 == tf:
            return IntegrationResult(
                t=np.array([t0]),
                x=x0[None, :],
                success=True,
                message="Zero time span",
                nfev=0,
                nsteps=0,
            )
        
        # Define drift function for Julia
        def drift_func(du, u, p, t):
            """Julia signature: f!(du, u, p, t) - in-place"""
            x_np = np.array(u)
            u_control = u_func(t, x_np)
            u_np = np.asarray(u_control)
            
            drift = self.system.drift(x_np, u_np, backend='numpy')
            du[:] = drift
            self._stats['total_fev'] += 1
        
        # Define diffusion function for Julia
        def diffusion_func(du, u, p, t):
            """Julia signature: g!(du, u, p, t) - in-place"""
            x_np = np.array(u)
            u_control = u_func(t, x_np)
            u_np = np.asarray(u_control)
            
            diff = self.system.diffusion(x_np, u_np, backend='numpy')
            du[:] = diff
        
        # Setup SDE problem
        tspan = (t0, tf)
        
        prob = self.de.SDEProblem(
            drift_func,
            diffusion_func,
            x0,
            tspan,
            noise_rate_prototype=self.noise_rate_prototype
        )
        
        # Prepare save points
        if t_eval is not None:
            saveat = list(t_eval)
        elif self.step_mode == StepMode.FIXED:
            n_steps = int((tf - t0) / self.dt) + 1
            saveat = list(np.linspace(t0, tf, n_steps))
        else:
            saveat = []
        
        # Solve
        try:
            # Set random seed in Julia
            import julia
            julia.Main.eval(f"using Random; Random.seed!({noise_seed})")
            
            sol = self.de.solve(
                prob,
                self._get_algorithm(),
                reltol=self.rtol,
                abstol=self.atol,
                saveat=saveat if saveat else None,
                save_everystep=(len(saveat) == 0),
                dt=self.dt if self.step_mode == StepMode.FIXED else None,
            )
            
            # Extract solution
            t_out = np.array(sol.t)
            x_out = np.array(sol.u).T
            
            success = True
            nsteps = len(t_out) - 1
            self._stats['total_steps'] += nsteps
            
            return IntegrationResult(
                t=t_out,
                x=x_out,
                success=success,
                message="SDE integration successful",
                nfev=self._stats['total_fev'],
                nsteps=nsteps,
                algorithm=self.algorithm,
                noise_seed=noise_seed,
            )
            
        except Exception as e:
            return IntegrationResult(
                t=np.array([t0]),
                x=x0[None, :],
                success=False,
                message=f"SDE integration failed: {str(e)}",
                nfev=0,
                nsteps=0,
            )
    
    def _get_algorithm(self):
        """Get Julia SDE algorithm object."""
        try:
            return eval(f"self.de.{self.algorithm}")
        except:
            return getattr(self.de, self.algorithm)()
    
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
        Tuple[np.ndarray, np.ndarray, np.ndarray]
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
        
        traj_stack = np.stack(trajectories)
        
        mean_traj = np.mean(traj_stack, axis=0)
        std_traj = np.std(traj_stack, axis=0)
        
        return result.t, mean_traj, std_traj


def list_diffeqpy_sde_algorithms() -> Dict[str, list]:
    """
    List available Julia SDE algorithms.
    
    Returns
    -------
    Dict[str, list]
        Categories of SDE algorithms
    """
    return {
        'low_order': [
            'EM',  # Euler-Maruyama (strong order 0.5)
            'EulerHeun',  # Euler-Heun
            'RKMil',  # Runge-Kutta Milstein
        ],
        'adaptive_high_order': [
            'SOSRI',  # Adaptive Rossler SRI (recommended)
            'SRI', 'SRIW1', 'SRIW2',  # Stochastic RK Itô
            'SRA', 'SRA1', 'SRA2', 'SRA3',  # Rossler methods
        ],
        'commutative_noise': [
            'SRA1', 'SRA2', 'SRA3',  # Optimized for commutative
        ],
        'stratonovich': [
            'RKMilCommute',  # For commutative Stratonovich
        ],
        'stabilized': [
            'SOSRI2',  # Second order stabilized
        ],
    }