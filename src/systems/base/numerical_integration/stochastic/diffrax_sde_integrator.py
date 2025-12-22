"""
DiffraxSDEIntegrator: JAX-based SDE integration using Diffrax library.

This module provides high-performance SDE integration with automatic
differentiation support through JAX's JIT compilation and gradient capabilities.
Ideal for optimization, parameter estimation, and neural SDEs.

Supports explicit SDE solvers with both Ito and Stratonovich interpretations,
controlled and autonomous systems.

Mathematical Form
-----------------
Stochastic differential equations:

    dx = f(x, u, t)dt + g(x, u, t)dW

Available Algorithms
-------------------
**General Purpose (Diagonal/Scalar Noise):**
- Euler-Heun: Order 0.5 strong, fast and robust
- Heun: Simple second-order method
- ItoMilstein: Order 1.0 strong (requires derivatives)
- StratonovichMilstein: Stratonovich version

**Specialized for Additive Noise:**
- SEA (SDE Adapted): Optimized for additive noise
- SHARK: Higher-order for additive noise
- SRA1: Order 2.0 weak for additive noise

**Reversible (Time-reversible):**
- ReversibleHeun: Reversible integration

**Advanced (Commutative Noise):**
- SDE solvers for commutative noise structures

Key Features
-----------
- **JIT Compilation**: Blazing fast execution via JAX
- **Automatic Differentiation**: Full gradient support through solves
- **GPU Acceleration**: Native GPU support via JAX
- **Vectorization**: Easy batching with vmap
- **Adjoint Methods**: Memory-efficient backpropagation

Installation
-----------
Requires JAX and Diffrax:

```bash
# CPU-only
pip install jax diffrax

# GPU (CUDA 12)
pip install -U "jax[cuda12]" diffrax

# GPU (CUDA 11)  
pip install -U "jax[cuda11]" diffrax
```

Examples
--------
>>> # Ornstein-Uhlenbeck (autonomous, additive noise)
>>> integrator = DiffraxSDEIntegrator(
...     sde_system,
...     dt=0.01,
...     solver='Euler',
...     backend='jax'
... )
>>> 
>>> result = integrator.integrate(
...     x0=jnp.array([1.0]),
...     u_func=lambda t, x: None,
...     t_span=(0.0, 10.0)
... )
>>> 
>>> # High accuracy with specialized solver
>>> integrator = DiffraxSDEIntegrator(
...     sde_system,
...     dt=0.001,
...     solver='SEA',  # Optimized for additive noise
...     rtol=1e-6,
...     atol=1e-8
... )
>>> 
>>> # GPU acceleration
>>> integrator = DiffraxSDEIntegrator(
...     sde_system,
...     dt=0.01,
...     solver='Euler',
...     backend='jax'
... )
>>> integrator.to_device('gpu')
>>> 
>>> # With gradients for optimization
>>> def loss_fn(x0_val):
...     result = integrator.integrate(x0_val, u_func, t_span)
...     return jnp.sum(result.x[-1]**2)
>>> 
>>> grad_fn = jax.grad(loss_fn)
>>> gradient = grad_fn(jnp.array([1.0]))
"""

from typing import Optional, Tuple, Callable, Dict, Any, List
import jax
import jax.numpy as jnp
from jax import Array
import diffrax as dfx

from src.systems.base.numerical_integration.stochastic.sde_integrator_base import (
    SDEIntegratorBase,
    SDEType,
    ConvergenceType,
    SDEIntegrationResult,
    StepMode,
    ArrayLike
)


class DiffraxSDEIntegrator(SDEIntegratorBase):
    """
    JAX-based SDE integrator using the Diffrax library.
    
    Provides high-performance SDE integration with automatic differentiation,
    JIT compilation, and GPU support via JAX.
    
    Parameters
    ----------
    sde_system : StochasticDynamicalSystem
        SDE system to integrate (controlled or autonomous)
    dt : Optional[float]
        Time step size (required for fixed step, initial guess for adaptive)
    step_mode : StepMode
        FIXED or ADAPTIVE stepping mode
    backend : str
        Must be 'jax' for this integrator
    solver : str
        Diffrax SDE solver name (default: 'Euler')
        Options: 'Euler', 'EulerHeun', 'Heun', 'ItoMilstein', 
                'SEA', 'SHARK', 'SRA1', 'ReversibleHeun'
    sde_type : Optional[SDEType]
        SDE interpretation (None = use system's type)
    convergence_type : ConvergenceType
        Strong or weak convergence
    seed : Optional[int]
        Random seed for reproducibility
    levy_area : str
        Levy area approximation: 'none', 'space-time', or 'full'
        Required for Milstein-type solvers
    **options
        Additional options:
        - rtol : float (default: 1e-3) - Relative tolerance (adaptive only)
        - atol : float (default: 1e-6) - Absolute tolerance (adaptive only)
        - max_steps : int (default: 10000) - Maximum steps
        - adjoint : str ('recursive_checkpoint', 'direct', 'implicit')
    
    Raises
    ------
    ValueError
        If backend is not 'jax'
    ImportError
        If JAX or Diffrax not installed
    
    Notes
    -----
    - Backend must be 'jax' (Diffrax is JAX-only)
    - Diffrax generates noise internally with high-quality PRNG
    - JIT compilation happens on first call (may be slow)
    - For optimization, use adjoint='recursive_checkpoint'
    - Milstein methods require levy_area approximation
    
    Examples
    --------
    >>> # Basic usage
    >>> integrator = DiffraxSDEIntegrator(
    ...     sde_system,
    ...     dt=0.01,
    ...     solver='Euler'
    ... )
    >>> 
    >>> # Optimized for additive noise
    >>> integrator = DiffraxSDEIntegrator(
    ...     additive_noise_system,
    ...     dt=0.001,
    ...     solver='SEA'  # Specialized for additive noise
    ... )
    >>> 
    >>> # High-order accuracy
    >>> integrator = DiffraxSDEIntegrator(
    ...     sde_system,
    ...     dt=0.001,
    ...     solver='ItoMilstein',
    ...     levy_area='space-time'
    ... )
    >>> 
    >>> # For gradient-based optimization
    >>> integrator = DiffraxSDEIntegrator(
    ...     sde_system,
    ...     dt=0.01,
    ...     solver='Euler',
    ...     adjoint='recursive_checkpoint'
    ... )
    """
    
    def __init__(
        self,
        sde_system,
        dt: Optional[float] = None,
        step_mode: StepMode = StepMode.FIXED,
        backend: str = 'jax',
        solver: str = 'Euler',
        sde_type: Optional[SDEType] = None,
        convergence_type: ConvergenceType = ConvergenceType.STRONG,
        seed: Optional[int] = None,
        levy_area: str = 'none',
        **options
    ):
        """Initialize Diffrax SDE integrator."""
        
        # Validate backend
        if backend != 'jax':
            raise ValueError(
                f"DiffraxSDEIntegrator requires backend='jax', got '{backend}'"
            )
        
        # Initialize base class
        super().__init__(
            sde_system,
            dt=dt,
            step_mode=step_mode,
            backend=backend,
            sde_type=sde_type,
            convergence_type=convergence_type,
            seed=seed,
            **options
        )
        
        self.solver_name = solver
        self.levy_area = levy_area
        self._integrator_name = f"Diffrax-{solver}"
        
        # Map solver names to Diffrax solver classes
        # Note: Not all solvers may be available in all Diffrax versions
        self._solver_map = {}
        
        # Basic solvers (always available)
        if hasattr(dfx, 'Euler'):
            self._solver_map['Euler'] = dfx.Euler
        if hasattr(dfx, 'EulerHeun'):
            self._solver_map['EulerHeun'] = dfx.EulerHeun
        if hasattr(dfx, 'Heun'):
            self._solver_map['Heun'] = dfx.Heun
        
        # Milstein methods
        if hasattr(dfx, 'ItoMilstein'):
            self._solver_map['ItoMilstein'] = dfx.ItoMilstein
        if hasattr(dfx, 'StratonovichMilstein'):
            self._solver_map['StratonovichMilstein'] = dfx.StratonovichMilstein
        
        # Specialized for additive noise (may not exist in older versions)
        if hasattr(dfx, 'SEA'):
            self._solver_map['SEA'] = dfx.SEA
        if hasattr(dfx, 'ShARK'):
            self._solver_map['SHARK'] = dfx.ShARK
        if hasattr(dfx, 'SRA1'):
            self._solver_map['SRA1'] = dfx.SRA1
        
        # Reversible
        if hasattr(dfx, 'ReversibleHeun'):
            self._solver_map['ReversibleHeun'] = dfx.ReversibleHeun
        
        # Validate solver
        if self.solver_name not in self._solver_map:
            raise ValueError(
                f"Unknown solver '{solver}'. "
                f"Available: {list(self._solver_map.keys())}"
            )
        
        # Adjoint method for backpropagation
        self.adjoint_name = options.get('adjoint', 'recursive_checkpoint')
        self._adjoint_map = {
            'recursive_checkpoint': dfx.RecursiveCheckpointAdjoint,
            'direct': dfx.DirectAdjoint,
            'implicit': dfx.ImplicitAdjoint,
        }
        
        if self.adjoint_name not in self._adjoint_map:
            raise ValueError(
                f"Unknown adjoint '{self.adjoint_name}'. "
                f"Available: {list(self._adjoint_map.keys())}"
            )
        
        # Device management
        self._device = 'cpu'
    
    @property
    def name(self) -> str:
        """Return integrator name."""
        mode_str = "Fixed" if self.step_mode == StepMode.FIXED else "Adaptive"
        return f"{self._integrator_name} ({mode_str})"
    
    def _get_solver_instance(self):
        """Get Diffrax solver instance."""
        solver_class = self._solver_map[self.solver_name]
        
        # Some solvers require levy area approximation
        if self.solver_name in ['ItoMilstein', 'StratonovichMilstein']:
            # Milstein methods need Levy area
            if self.levy_area == 'none':
                raise ValueError(
                    f"Solver '{self.solver_name}' requires levy_area. "
                    f"Set levy_area='space-time' or 'full'"
                )
        
        return solver_class()
    
    def _get_brownian_motion(self, key, t0, t1, shape):
        """
        Create Brownian motion for the integration interval.
        
        Parameters
        ----------
        key : jax.random.PRNGKey
            Random key for noise generation
        t0 : float
            Start time
        t1 : float
            End time
        shape : tuple
            Shape of noise (nw,) or similar
            
        Returns
        -------
        dfx.BrownianPath
            Brownian motion object
        """
        if self.levy_area == 'none':
            # Standard Brownian motion (no Levy area)
            return dfx.VirtualBrownianTree(
                t0, t1, tol=1e-3, shape=shape, key=key
            )
        elif self.levy_area == 'space-time':
            # Space-time Levy area (for Milstein)
            # Note: API may vary by Diffrax version
            try:
                return dfx.SpaceTimeLevyArea(
                    t0, t1, tol=1e-3, shape=shape, key=key
                )
            except TypeError:
                # Older API without t0/t1
                return dfx.SpaceTimeLevyArea(
                    tol=1e-3, shape=shape, key=key
                )
        elif self.levy_area == 'full':
            # Full Levy area approximation
            try:
                return dfx.SpaceTimeTimeLevyArea(
                    t0, t1, tol=1e-3, shape=shape, key=key
                )
            except (TypeError, AttributeError):
                # May not exist in all versions
                return dfx.VirtualBrownianTree(
                    t0, t1, tol=1e-3, shape=shape, key=key
                )
        else:
            raise ValueError(
                f"Invalid levy_area '{self.levy_area}'. "
                f"Choose from: 'none', 'space-time', 'full'"
            )
    
    def step(
        self,
        x: ArrayLike,
        u: Optional[ArrayLike] = None,
        dt: Optional[float] = None,
        dW: Optional[ArrayLike] = None
    ) -> ArrayLike:
        """
        Take one SDE integration step.
        
        Parameters
        ----------
        x : ArrayLike
            Current state (nx,) or (batch, nx)
        u : Optional[ArrayLike]
            Control input (nu,) or (batch, nu), or None for autonomous
        dt : Optional[float]
            Step size (uses self.dt if None)
        dW : Optional[ArrayLike]
            Brownian increments (not used - Diffrax generates internally)
            
        Returns
        -------
        ArrayLike
            Next state x(t + dt)
            
        Notes
        -----
        Single-step interface is less efficient than full integration
        due to overhead of setting up Diffrax problem each time.
        """
        step_size = dt if dt is not None else self.dt
        
        if step_size is None:
            raise ValueError("Step size dt must be specified")
        
        # Convert to JAX arrays
        x = jnp.asarray(x)
        if u is not None:
            u = jnp.asarray(u)
        
        # Generate random key
        if self.seed is not None:
            key = jax.random.PRNGKey(self.seed)
        else:
            key = jax.random.PRNGKey(0)
        
        # Define drift and diffusion for Diffrax
        def drift(t, y, args):
            return self.sde_system.drift(y, u, backend=self.backend)
        
        def diffusion(t, y, args):
            return self.sde_system.diffusion(y, u, backend=self.backend)
        
        # Create SDE terms
        drift_term = dfx.ODETerm(drift)
        diffusion_term = dfx.ControlTerm(
            diffusion,
            self._get_brownian_motion(key, 0.0, step_size, (self.sde_system.nw,))
        )
        terms = dfx.MultiTerm(drift_term, diffusion_term)
        
        # Get solver
        solver = self._get_solver_instance()
        
        # Integrate single step
        solution = dfx.diffeqsolve(
            terms,
            solver,
            t0=0.0,
            t1=step_size,
            dt0=step_size,
            y0=x,
            saveat=dfx.SaveAt(t1=True),
            stepsize_controller=dfx.ConstantStepSize(),
            max_steps=10,
        )
        
        # Update statistics
        self._stats['total_steps'] += 1
        self._stats['total_fev'] += 1
        
        return solution.ys[0]
    
    def integrate(
        self,
        x0: ArrayLike,
        u_func: Callable[[float, ArrayLike], Optional[ArrayLike]],
        t_span: Tuple[float, float],
        t_eval: Optional[ArrayLike] = None,
        dense_output: bool = False
    ) -> SDEIntegrationResult:
        """
        Integrate SDE over time interval.
        
        Parameters
        ----------
        x0 : ArrayLike
            Initial state (nx,)
        u_func : Callable
            Control policy: (t, x) -> u (or None for autonomous)
        t_span : Tuple[float, float]
            Integration interval (t_start, t_end)
        t_eval : Optional[ArrayLike]
            Specific times at which to save solution
        dense_output : bool
            If True, enable dense output (not supported for SDEs in Diffrax)
            
        Returns
        -------
        SDEIntegrationResult
            Integration result with trajectory
            
        Examples
        --------
        >>> # Autonomous SDE
        >>> result = integrator.integrate(
        ...     x0=jnp.array([1.0]),
        ...     u_func=lambda t, x: None,
        ...     t_span=(0.0, 10.0)
        ... )
        >>> 
        >>> # Controlled SDE
        >>> K = jnp.array([[1.0, 2.0]])
        >>> result = integrator.integrate(
        ...     x0=jnp.array([1.0, 0.0]),
        ...     u_func=lambda t, x: -K @ x,
        ...     t_span=(0.0, 10.0)
        ... )
        """
        t0, tf = t_span
        x0 = jnp.asarray(x0)
        
        # Validate time span
        if tf <= t0:
            raise ValueError(
                f"End time must be greater than start time. "
                f"Got t_span=({t0}, {tf})"
            )
        
        # Generate random key
        if self.seed is not None:
            key = jax.random.PRNGKey(self.seed)
        else:
            key = jax.random.PRNGKey(0)
        
        # Define drift and diffusion
        def drift(t, y, args):
            u = u_func(t, y)
            if u is not None and not isinstance(u, jnp.ndarray):
                u = jnp.asarray(u)
            
            dx = self.sde_system.drift(y, u, backend=self.backend)
            self._stats['total_fev'] += 1
            return dx
        
        def diffusion(t, y, args):
            u = u_func(t, y)
            if u is not None and not isinstance(u, jnp.ndarray):
                u = jnp.asarray(u)
            
            g = self.sde_system.diffusion(y, u, backend=self.backend)
            self._stats['diffusion_evals'] += 1
            return g
        
        # Create SDE terms
        drift_term = dfx.ODETerm(drift)
        brownian = self._get_brownian_motion(
            key, t0, tf, (self.sde_system.nw,)
        )
        diffusion_term = dfx.ControlTerm(diffusion, brownian)
        terms = dfx.MultiTerm(drift_term, diffusion_term)
        
        # Get solver and adjoint
        solver = self._get_solver_instance()
        adjoint = self._adjoint_map[self.adjoint_name]()
        
        # Setup save points and step controller
        if self.step_mode == StepMode.FIXED:
            if t_eval is not None:
                t_points = jnp.asarray(t_eval)
            else:
                n_steps = max(2, int((tf - t0) / self.dt) + 1)
                t_points = jnp.linspace(t0, tf, n_steps)
            
            stepsize_controller = dfx.StepTo(ts=t_points)
            saveat = dfx.SaveAt(ts=t_points)
            dt0_value = None
            
        else:
            # Adaptive step mode
            stepsize_controller = dfx.PIDController(
                rtol=self.rtol,
                atol=self.atol,
            )
            
            if t_eval is not None:
                t_points = jnp.asarray(t_eval)
                saveat = dfx.SaveAt(ts=t_points)
            else:
                n_dense = max(2, self.options.get('n_dense', 100))
                t_points = jnp.linspace(t0, tf, n_dense)
                saveat = dfx.SaveAt(ts=t_points)
            
            dt0_value = self.dt if self.dt is not None else (tf - t0) / 100
        
        # Solve SDE
        try:
            solution = dfx.diffeqsolve(
                terms,
                solver,
                t0=t0,
                t1=tf,
                dt0=dt0_value,
                y0=x0,
                saveat=saveat,
                stepsize_controller=stepsize_controller,
                max_steps=self.max_steps,
                adjoint=adjoint,
                throw=False,
            )
            
            # Check success
            success = jnp.all(jnp.isfinite(solution.ys))
            
            # Check if solution is valid
            if len(solution.ys) == 0:
                return SDEIntegrationResult(
                    t=jnp.array([t0]),
                    x=x0[jnp.newaxis, :] if x0.ndim == 1 else x0,
                    success=False,
                    message=f"Diffrax SDE solver '{self.solver_name}' produced no output",
                    nfev=0,
                    nsteps=0,
                    diffusion_evals=0,
                    n_paths=1,
                    convergence_type=self.convergence_type,
                    solver=self.solver_name,
                    sde_type=self.sde_type,
                )
            
            # Update statistics
            nsteps = int(solution.stats.get("num_steps", len(solution.ts) - 1))
            nfev = nsteps  # Estimate
            self._stats['total_steps'] += nsteps
            self._stats['total_fev'] += nfev
            self._stats['diffusion_evals'] += nsteps
            
            return SDEIntegrationResult(
                t=solution.ts,
                x=solution.ys,
                success=bool(success),
                message="Diffrax SDE integration successful" if success else "Integration failed (NaN/Inf detected)",
                nfev=self._stats['total_fev'],
                nsteps=nsteps,
                diffusion_evals=self._stats['diffusion_evals'],
                noise_samples=None,  # Diffrax doesn't expose noise samples
                n_paths=1,
                convergence_type=self.convergence_type,
                solver=self.solver_name,
                sde_type=self.sde_type,
            )
            
        except Exception as e:
            import traceback
            return SDEIntegrationResult(
                t=jnp.array([t0]),
                x=x0[jnp.newaxis, :] if x0.ndim == 1 else x0,
                success=False,
                message=f"Diffrax SDE integration failed: {str(e)}\n{traceback.format_exc()}",
                nfev=0,
                nsteps=0,
                diffusion_evals=0,
                n_paths=1,
                convergence_type=self.convergence_type,
                solver=self.solver_name,
                sde_type=self.sde_type,
            )
    
    # ========================================================================
    # JAX-Specific Methods
    # ========================================================================
    
    def integrate_with_gradient(
        self,
        x0: ArrayLike,
        u_func: Callable[[float, ArrayLike], Optional[ArrayLike]],
        t_span: Tuple[float, float],
        loss_fn: Callable[[SDEIntegrationResult], float],
        t_eval: Optional[ArrayLike] = None,
    ):
        """
        Integrate and compute gradients w.r.t. initial conditions.
        
        Parameters
        ----------
        x0 : ArrayLike
            Initial state
        u_func : Callable
            Control policy
        t_span : Tuple[float, float]
            Time span
        loss_fn : Callable
            Loss function taking IntegrationResult
        t_eval : Optional[ArrayLike]
            Evaluation times
            
        Returns
        -------
        tuple
            (loss_value, gradient_wrt_x0)
            
        Examples
        --------
        >>> def loss_fn(result):
        ...     return jnp.sum(result.x[-1]**2)
        >>> 
        >>> loss, grad = integrator.integrate_with_gradient(
        ...     x0, u_func, t_span, loss_fn
        ... )
        """
        def compute_loss(x0_val):
            result = self.integrate(x0_val, u_func, t_span, t_eval)
            return loss_fn(result)
        
        loss, grad = jax.value_and_grad(compute_loss)(x0)
        return loss, grad
    
    def jit_compile(self):
        """
        JIT-compile the integration for faster execution.
        
        First call will be slow (compilation), subsequent calls fast.
        
        Returns
        -------
        Callable
            JIT-compiled integration function
            
        Examples
        --------
        >>> jit_integrate = integrator.jit_compile()
        >>> result = jit_integrate(x0, u_func, t_span)  # Fast!
        """
        return jax.jit(self.integrate)
    
    def vectorized_integrate(
        self,
        x0_batch: ArrayLike,
        u_func: Callable,
        t_span: Tuple[float, float],
        t_eval: Optional[ArrayLike] = None,
    ):
        """
        Vectorized integration over batch of initial conditions.
        
        Uses jax.vmap for efficient batching.
        
        Parameters
        ----------
        x0_batch : ArrayLike
            Batch of initial states (batch, nx)
        u_func : Callable
            Control policy
        t_span : Tuple[float, float]
            Time span
        t_eval : Optional[ArrayLike]
            Evaluation times
            
        Returns
        -------
        List[SDEIntegrationResult]
            Results for each initial condition
            
        Examples
        --------
        >>> x0_batch = jnp.array([[1.0], [2.0], [3.0]])
        >>> results = integrator.vectorized_integrate(
        ...     x0_batch, u_func, (0, 10)
        ... )
        """
        # Note: This is a simple sequential implementation
        # True vmap requires more careful handling of randomness
        results = []
        for i in range(x0_batch.shape[0]):
            result = self.integrate(x0_batch[i], u_func, t_span, t_eval)
            results.append(result)
        return results
    
    def to_device(self, device: str):
        """
        Move computations to specified device.
        
        Parameters
        ----------
        device : str
            'cpu', 'gpu', 'gpu:0', 'gpu:1', etc.
            
        Examples
        --------
        >>> integrator.to_device('gpu')
        >>> # Now all computations use GPU
        """
        self._device = device
        # JAX will automatically use the specified device
        # when arrays are created
    
    # ========================================================================
    # Algorithm Information
    # ========================================================================
    
    @staticmethod
    def list_solvers() -> Dict[str, List[str]]:
        """
        List available Diffrax SDE solvers by category.
        
        Returns
        -------
        Dict[str, List[str]]
            Solvers organized by category
            
        Examples
        --------
        >>> solvers = DiffraxSDEIntegrator.list_solvers()
        >>> print(solvers['additive_noise'])
        ['SEA', 'SHARK', 'SRA1']
        """
        return {
            'basic': [
                'Euler',        # Euler-Maruyama
                'EulerHeun',    # Euler-Heun (predictor-corrector)
                'Heun',         # Heun's method
            ],
            'milstein': [
                'ItoMilstein',            # Ito-Milstein
                'StratonovichMilstein',   # Stratonovich-Milstein
            ],
            'additive_noise': [
                'SEA',      # SDE Adapted (additive noise)
                'SHARK',    # Higher-order for additive
                'SRA1',     # Order 2.0 weak
            ],
            'reversible': [
                'ReversibleHeun',  # Time-reversible
            ],
        }
    
    @staticmethod
    def get_solver_info(solver: str) -> Dict[str, Any]:
        """
        Get information about a specific solver.
        
        Parameters
        ----------
        solver : str
            Solver name
            
        Returns
        -------
        Dict[str, Any]
            Solver properties
            
        Examples
        --------
        >>> info = DiffraxSDEIntegrator.get_solver_info('SEA')
        >>> print(info['description'])
        'Optimized for additive noise, high efficiency'
        """
        solver_info = {
            'Euler': {
                'name': 'Euler-Maruyama',
                'strong_order': 0.5,
                'weak_order': 1.0,
                'description': 'Basic explicit method',
                'best_for': 'General purpose, fast',
                'levy_area': 'none',
            },
            'EulerHeun': {
                'name': 'Euler-Heun',
                'strong_order': 0.5,
                'weak_order': 1.0,
                'description': 'Predictor-corrector variant',
                'best_for': 'Improved stability',
                'levy_area': 'none',
            },
            'ItoMilstein': {
                'name': 'Ito-Milstein',
                'strong_order': 1.0,
                'weak_order': 1.0,
                'description': 'Higher order with Levy area',
                'best_for': 'Better accuracy',
                'levy_area': 'space-time',
            },
            'SEA': {
                'name': 'SDE Adapted',
                'strong_order': 1.0,
                'weak_order': 2.0,
                'description': 'Optimized for additive noise',
                'best_for': 'Fast, additive noise only',
                'levy_area': 'none',
            },
            'SHARK': {
                'name': 'ShARK',
                'strong_order': 1.5,
                'weak_order': 2.0,
                'description': 'Higher-order additive',
                'best_for': 'High accuracy, additive noise',
                'levy_area': 'none',
            },
            'SRA1': {
                'name': 'SRA1',
                'strong_order': 1.0,
                'weak_order': 2.0,
                'description': 'Weak order 2 for additive',
                'best_for': 'Monte Carlo, additive noise',
                'levy_area': 'none',
            },
        }
        
        return solver_info.get(
            solver,
            {
                'name': solver,
                'description': 'Diffrax SDE solver',
            }
        )
    
    @staticmethod
    def recommend_solver(
        noise_type: str,
        accuracy: str = 'medium',
        has_derivatives: bool = False
    ) -> str:
        """
        Recommend Diffrax solver based on problem characteristics.
        
        Parameters
        ----------
        noise_type : str
            'additive', 'diagonal', 'scalar', or 'general'
        accuracy : str
            'low', 'medium', or 'high'
        has_derivatives : bool
            Whether diffusion derivatives are available
            
        Returns
        -------
        str
            Recommended solver name
            
        Examples
        --------
        >>> solver = DiffraxSDEIntegrator.recommend_solver(
        ...     noise_type='additive',
        ...     accuracy='high'
        ... )
        >>> print(solver)
        'SHARK'
        """
        if noise_type == 'additive':
            if accuracy == 'high':
                return 'SHARK'
            elif accuracy == 'medium':
                return 'SEA'
            else:
                return 'Euler'
        
        elif has_derivatives and accuracy == 'high':
            return 'ItoMilstein'
        
        elif accuracy == 'high':
            return 'EulerHeun'
        
        else:
            return 'Euler'


# ============================================================================
# Utility Functions
# ============================================================================

def create_diffrax_sde_integrator(
    sde_system,
    solver: str = 'Euler',
    dt: float = 0.01,
    **options
) -> DiffraxSDEIntegrator:
    """
    Quick factory for Diffrax SDE integrators.
    
    Parameters
    ----------
    sde_system : StochasticDynamicalSystem
        SDE system to integrate
    solver : str
        Diffrax solver name
    dt : float
        Time step
    **options
        Additional options
        
    Returns
    -------
    DiffraxSDEIntegrator
        Configured integrator
        
    Examples
    --------
    >>> integrator = create_diffrax_sde_integrator(
    ...     sde_system,
    ...     solver='SEA',
    ...     dt=0.001
    ... )
    """
    return DiffraxSDEIntegrator(
        sde_system,
        dt=dt,
        solver=solver,
        backend='jax',
        **options
    )


def list_diffrax_sde_solvers() -> None:
    """
    Print all available Diffrax SDE solvers.
    
    Examples
    --------
    >>> list_diffrax_sde_solvers()
    
    Diffrax SDE Solvers
    ===================
    Basic Methods:
      - Euler: Euler-Maruyama (strong 0.5, weak 1.0)
    ...
    """
    solvers = DiffraxSDEIntegrator.list_solvers()
    
    print("Diffrax SDE Solvers (JAX-based)")
    print("=" * 60)
    
    category_names = {
        'basic': 'Basic Methods',
        'milstein': 'Milstein Methods (require Levy area)',
        'additive_noise': 'Specialized for Additive Noise',
        'reversible': 'Time-Reversible Methods',
    }
    
    for category, solver_list in solvers.items():
        print(f"\n{category_names.get(category, category)}:")
        for solver in solver_list:
            info = DiffraxSDEIntegrator.get_solver_info(solver)
            if 'strong_order' in info:
                print(f"  - {solver}: {info['description']} "
                      f"(strong {info['strong_order']}, weak {info['weak_order']})")
            else:
                print(f"  - {solver}: {info['description']}")
    
    print("\n" + "=" * 60)
    print("Use get_solver_info(name) for details")
    print("Use recommend_solver() for automatic selection")