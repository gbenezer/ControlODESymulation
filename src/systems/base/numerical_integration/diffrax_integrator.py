"""
DiffraxIntegrator: JAX-based ODE integration using Diffrax library.

This module provides adaptive and fixed-step ODE integration with automatic
differentiation support through JAX's JIT compilation.

Supports explicit, implicit, IMEX, and special solvers from Diffrax.
Supports both controlled and autonomous systems (nu=0).

Known Limitations:
- Backward time integration (tf < t0) is not currently supported due to
  complexity in handling time transformations and result ordering with Diffrax.
  For reverse-time integration, integrate forward and reverse the arrays.
"""

from typing import Optional, Callable, Tuple
import jax
import jax.numpy as jnp
from jax import Array
import diffrax as dfx

from src.systems.base.numerical_integration.integrator_base import (
    IntegratorBase,
    IntegrationResult,
    StepMode,
    ArrayLike
)


class DiffraxIntegrator(IntegratorBase):
    """
    JAX-based ODE integrator using the Diffrax library.

    Supports adaptive and fixed-step integration with various solvers
    including explicit, implicit (0.8.0+), IMEX (0.8.0+), and special methods.

    Parameters
    ----------
    system : SymbolicDynamicalSystem
        Continuous-time system to integrate (controlled or autonomous)
    dt : Optional[float]
        Time step size
    step_mode : StepMode
        FIXED or ADAPTIVE stepping mode
    backend : str
        Must be 'jax' for this integrator
    solver : str, optional
        Solver name. See _solver_map for available options.
        Default: 'tsit5'
    adjoint : str, optional
        Adjoint method for backpropagation. Options: 'recursive_checkpoint',
        'direct', 'implicit'. Default: 'recursive_checkpoint'
    **options
        Additional options including rtol, atol, max_steps
    
    Available Solvers (Diffrax 0.7.0)
    ----------------------------------
    Explicit Runge-Kutta:
        - tsit5: Tsitouras 5(4) - recommended for most problems
        - dopri5: Dormand-Prince 5(4)
        - dopri8: Dormand-Prince 8(7) - high accuracy
        - bosh3: Bogacki-Shampine 3(2)
        - euler: Forward Euler (1st order)
        - midpoint: Explicit midpoint (2nd order)
        - heun: Heun's method (2nd order)
        - ralston: Ralston's method (2nd order)
        - reversible_heun: Reversible Heun (2nd order, reversible)
        
    Additional Solvers (Diffrax 0.8.0+)
    ------------------------------------
    Implicit methods (for stiff systems):
        - implicit_euler: Backward Euler (1st order, A-stable)
        - kvaerno3: Kvaerno 3(2) ESDIRK
        - kvaerno4: Kvaerno 4(3) ESDIRK
        - kvaerno5: Kvaerno 5(4) ESDIRK
        
    IMEX methods (for semi-stiff systems):
        - sil3: 3rd order IMEX
        - kencarp3: Kennedy-Carpenter IMEX 3
        - kencarp4: Kennedy-Carpenter IMEX 4
        - kencarp5: Kennedy-Carpenter IMEX 5
        
    Special methods:
        - semi_implicit_euler: Semi-implicit Euler (symplectic)
    
    Notes
    -----
    - Implicit and IMEX solvers require Diffrax 0.8.0 or later
    - The integrator will automatically detect which solvers are available
    - Use `integrator._solver_map.keys()` to see available solvers
    - Supports autonomous systems (nu=0) by passing u=None
    - Backward time integration (t_span with tf < t0) is not supported
    
    Examples
    --------
    >>> # Controlled system
    >>> integrator = DiffraxIntegrator(system, backend='jax', solver='tsit5')
    >>> result = integrator.integrate(
    ...     x0=jnp.array([1.0, 0.0]),
    ...     u_func=lambda t, x: jnp.array([0.5]),
    ...     t_span=(0.0, 10.0)
    ... )
    >>> 
    >>> # Autonomous system
    >>> integrator = DiffraxIntegrator(autonomous_system, backend='jax')
    >>> result = integrator.integrate(
    ...     x0=jnp.array([1.0, 0.0]),
    ...     u_func=lambda t, x: None,
    ...     t_span=(0.0, 10.0)
    ... )
    """

    def __init__(
        self,
        system,
        dt: Optional[float] = None,
        step_mode: StepMode = StepMode.ADAPTIVE,
        backend: str = 'jax',
        solver: str = "tsit5",
        adjoint: str = "recursive_checkpoint",
        **options
    ):
        # Validate backend
        if backend != 'jax':
            raise ValueError(
                f"DiffraxIntegrator requires backend='jax', got '{backend}'"
            )
        
        # Initialize base class
        super().__init__(system, dt, step_mode, backend, **options)
        
        self.solver_name = solver.lower()
        self.adjoint_name = adjoint
        self._integrator_name = f"Diffrax-{solver}"

        # Map solver names to Diffrax solver classes
        # Organized by category for clarity
        self._solver_map = {
            # Explicit Runge-Kutta methods
            "tsit5": dfx.Tsit5,
            "dopri5": dfx.Dopri5,
            "dopri8": dfx.Dopri8,
            "bosh3": dfx.Bosh3,
            "euler": dfx.Euler,
            "midpoint": dfx.Midpoint,
            "heun": dfx.Heun,
            "ralston": dfx.Ralston,
            
            # Implicit methods (for stiff ODEs)
            "implicit_euler": dfx.ImplicitEuler,
            "kvaerno3": dfx.Kvaerno3,
            "kvaerno4": dfx.Kvaerno4,
            "kvaerno5": dfx.Kvaerno5,
            
            # IMEX methods (for semi-stiff ODEs)
            # Note: IMEX methods require splitting the ODE into stiff and non-stiff parts
            # For now, we'll mark them as available but may need special handling
            
            # Special methods
            "reversible_heun": dfx.ReversibleHeun,
            "semi_implicit_euler": dfx.SemiImplicitEuler,
        }
        
        # Additional explicit methods
        try:
            # RK4 might be available as a specific class
            self._solver_map["rk4"] = dfx.RK4
        except AttributeError:
            pass
        
        # IMEX solvers - these need special handling
        # They require the system to be split into explicit and implicit parts
        self._imex_solver_map = {
            "sil3": dfx.Sil3,
            "kencarp3": dfx.KenCarp3,
            "kencarp4": dfx.KenCarp4,
            "kencarp5": dfx.KenCarp5,
        }
        
        # Check if solver is IMEX
        self.is_imex = self.solver_name in self._imex_solver_map
        
        if not self.is_imex:
            if self.solver_name not in self._solver_map:
                raise ValueError(
                    f"Unknown solver '{solver}'. Available: {list(self._solver_map.keys()) + list(self._imex_solver_map.keys())}"
                )
        
        # Check if solver is implicit (requires Jacobian for efficiency)
        # Only relevant if implicit solvers are available (Diffrax 0.8.0+)
        implicit_solvers = {"implicit_euler", "kvaerno3", "kvaerno4", "kvaerno5"}
        self.is_implicit = self.solver_name in implicit_solvers and hasattr(dfx, 'ImplicitEuler')

        # Map adjoint names to Diffrax adjoint classes
        self._adjoint_map = {
            "recursive_checkpoint": dfx.RecursiveCheckpointAdjoint,
            "direct": dfx.DirectAdjoint,
            "implicit": dfx.ImplicitAdjoint,
        }

        if self.adjoint_name not in self._adjoint_map:
            raise ValueError(
                f"Unknown adjoint '{adjoint}'. Available: {list(self._adjoint_map.keys())}"
            )

    @property
    def name(self) -> str:
        """Return the name of the integrator."""
        mode_str = "Fixed Step" if self.step_mode == StepMode.FIXED else "Adaptive"
        solver_type = "IMEX" if self.is_imex else "Implicit" if self.is_implicit else "Explicit"
        return f"{self._integrator_name} ({solver_type}, {mode_str})"

    def _get_solver_instance(self):
        """Get the appropriate solver instance."""
        if self.is_imex:
            # IMEX solvers require explicit and implicit terms
            return self._imex_solver_map[self.solver_name]()
        else:
            return self._solver_map[self.solver_name]()
    
    def _create_ode_term(self, ode_func, use_implicit=False):
        """
        Create appropriate ODE term based on solver type.
        
        Parameters
        ----------
        ode_func : callable
            Function with signature (t, y, args) -> dy/dt
        use_implicit : bool
            If True and solver is implicit, create implicit term
            
        Returns
        -------
        ODETerm or MultiTerm
            Appropriate term for the solver
        """
        if self.is_imex:
            # IMEX solvers need explicit and implicit parts
            # For general systems, we use the full dynamics as explicit
            # and empty implicit part (user can override if needed)
            explicit_term = dfx.ODETerm(ode_func)
            implicit_term = dfx.ODETerm(lambda t, y, args: jnp.zeros_like(y))
            return dfx.MultiTerm(explicit_term, implicit_term)
        
        elif self.is_implicit and use_implicit:
            # For implicit solvers, we can optionally provide the Jacobian
            # This is more efficient but requires linearization support
            return dfx.ODETerm(ode_func)
        
        else:
            # Standard explicit term
            return dfx.ODETerm(ode_func)

    def step(
        self,
        x: ArrayLike,
        u: Optional[ArrayLike] = None,
        dt: Optional[float] = None
    ) -> ArrayLike:
        """
        Take one integration step: x(t) → x(t + dt).
        
        Parameters
        ----------
        x : ArrayLike
            Current state (nx,) or (batch, nx)
        u : Optional[ArrayLike]
            Control input (nu,) or (batch, nu), or None for autonomous systems
        dt : Optional[float]
            Step size (uses self.dt if None)
            
        Returns
        -------
        ArrayLike
            Next state x(t + dt)
        """
        step_size = dt if dt is not None else self.dt
        
        if step_size is None:
            raise ValueError("Step size dt must be specified")
        
        # Convert to JAX arrays if needed
        x = jnp.asarray(x)
        
        # Handle autonomous systems - keep None as None
        # Do NOT convert None to jax array
        if u is not None:
            u = jnp.asarray(u)
        
        # Define ODE function - MUST accept (t, y, args) even if args unused
        def ode_func(t, state, args):
            return self.system(state, u, backend=self.backend)
        
        # Create appropriate ODE term
        term = self._create_ode_term(ode_func)
        solver = self._get_solver_instance()
        
        # Set up solver kwargs
        solver_kwargs = {
            'terms': term,
            'solver': solver,
            't0': 0.0,
            't1': step_size,
            'dt0': step_size,
            'y0': x,
            'saveat': dfx.SaveAt(t1=True),
            'stepsize_controller': dfx.ConstantStepSize(),
            'max_steps': 10,
        }
        
        # Implicit solvers handle nonlinear solving internally
        # No additional configuration needed
        
        # Single step integration
        solution = dfx.diffeqsolve(**solver_kwargs)
        
        # Update stats
        self._stats['total_steps'] += 1
        self._stats['total_fev'] += int(solution.stats.get('num_steps', 1))
        
        return solution.ys[0]

    def integrate(
        self,
        x0: ArrayLike,
        u_func: Callable[[float, ArrayLike], Optional[ArrayLike]],
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
        u_func : Callable[[float, ArrayLike], Optional[ArrayLike]]
            Control policy: (t, x) → u (or None for autonomous systems)
        t_span : Tuple[float, float]
            Integration interval (t_start, t_end)
            Note: t_end must be > t_start (backward integration not supported)
        t_eval : Optional[ArrayLike]
            Specific times at which to store solution (must be increasing)
        dense_output : bool
            If True, return dense interpolated solution
            
        Returns
        -------
        IntegrationResult
            Object containing t, x, success, and metadata
            
        Raises
        ------
        ValueError
            If t_span has tf < t0 (backward integration not supported)
            
        Examples
        --------
        >>> # Controlled system
        >>> result = integrator.integrate(
        ...     x0=jnp.array([1.0, 0.0]),
        ...     u_func=lambda t, x: jnp.array([0.5]),
        ...     t_span=(0.0, 10.0)
        ... )
        >>> 
        >>> # Autonomous system
        >>> result = integrator.integrate(
        ...     x0=jnp.array([1.0, 0.0]),
        ...     u_func=lambda t, x: None,
        ...     t_span=(0.0, 10.0)
        ... )
        
        Notes
        -----
        Backward time integration (tf < t0) is not supported. If you need
        reverse-time integration, integrate forward and reverse the output:
        
        >>> # Instead of integrate(x0, u_func, (10.0, 0.0))
        >>> result = integrate(x0, u_func, (0.0, 10.0))
        >>> t_reversed = jnp.flip(result.t)
        >>> x_reversed = jnp.flip(result.x, axis=0)
        """
        t0, tf = t_span
        x0 = jnp.asarray(x0)
        
        # Validate time span
        if tf < t0:
            raise ValueError(
                f"Backward integration (tf < t0) is not supported. "
                f"Got t_span=({t0}, {tf}). "
                f"For reverse-time integration, integrate forward and flip results."
            )
        
        # Handle edge case
        if t0 == tf:
            return IntegrationResult(
                t=jnp.array([t0]),
                x=x0[None, :] if x0.ndim == 1 else x0,
                success=True,
                message="Zero time span",
                nfev=0,
                nsteps=0,
            )
        
        # Define ODE function - MUST accept (t, y, args) even if args unused
        def ode_func(t, state, args):
            u = u_func(t, state)
            
            # Handle autonomous systems - keep None as None
            # Do NOT convert None to jax array
            if u is not None and not isinstance(u, jnp.ndarray):
                u = jnp.asarray(u)
            
            return self.system(state, u, backend=self.backend)
        
        # Create appropriate ODE term
        term = self._create_ode_term(ode_func)
        solver = self._get_solver_instance()
        
        # Set up step size controller and save points
        if self.step_mode == StepMode.FIXED:
            if t_eval is not None:
                t_points = jnp.asarray(t_eval)
            else:
                n_steps = max(2, int((tf - t0) / self.dt) + 1)
                t_points = jnp.linspace(t0, tf, n_steps)
            
            stepsize_controller = dfx.StepTo(ts=t_points)
            saveat = dfx.SaveAt(ts=t_points)
            dt0_value = None  # StepTo determines step locations
            
        else:
            # Adaptive step mode
            stepsize_controller = dfx.PIDController(
                rtol=self.rtol,
                atol=self.atol,
                dtmin=self.options.get("dtmin", None),
                dtmax=self.options.get("dtmax", None),
            )
            
            if t_eval is not None:
                t_points = jnp.asarray(t_eval)
                saveat = dfx.SaveAt(ts=t_points)
            else:
                n_dense = max(2, self.options.get("n_dense", 100))
                t_points = jnp.linspace(t0, tf, n_dense)
                saveat = dfx.SaveAt(ts=t_points)
            
            dt0_value = self.dt if self.dt is not None else (tf - t0) / 100
        
        # Set up adjoint method
        adjoint_method = self._adjoint_map[self.adjoint_name]()
        
        # Build solver kwargs
        solver_kwargs = {
            'terms': term,
            'solver': solver,
            't0': t0,
            't1': tf,
            'dt0': dt0_value,
            'y0': x0,
            'saveat': saveat,
            'stepsize_controller': stepsize_controller,
            'max_steps': self.max_steps,
            'adjoint': adjoint_method,
            'throw': False,
        }
        
        # Solve ODE
        try:
            solution = dfx.diffeqsolve(**solver_kwargs)
            
            # Check success
            success = jnp.all(jnp.isfinite(solution.ys))
            
            # Update statistics
            nsteps = int(solution.stats.get("num_steps", 0))
            nfev = int(solution.stats.get("num_steps", 0))
            self._stats['total_steps'] += nsteps
            self._stats['total_fev'] += nfev
            
            return IntegrationResult(
                t=solution.ts,
                x=solution.ys,
                success=bool(success),
                message="Integration successful" if success else "Integration failed (NaN/Inf detected)",
                nfev=nfev,
                nsteps=nsteps,
                solver=self.solver_name,
                njev=int(solution.stats.get("num_jacobian_evals", 0)),
            )
            
        except Exception as e:
            return IntegrationResult(
                t=jnp.array([t0]),
                x=x0[None, :] if x0.ndim == 1 else x0,
                success=False,
                message=f"Integration failed: {str(e)}",
                nfev=0,
                nsteps=0,
            )

    # ========================================================================
    # JAX-Specific Methods
    # ========================================================================

    def integrate_with_gradient(
        self,
        x0: ArrayLike,
        u_func: Callable[[float, ArrayLike], Optional[ArrayLike]],
        t_span: Tuple[float, float],
        loss_fn: Callable[[IntegrationResult], float],
        t_eval: Optional[ArrayLike] = None,
    ):
        """Integrate and compute gradients w.r.t. initial conditions."""
        def compute_loss(x0_val):
            result = self.integrate(x0_val, u_func, t_span, t_eval)
            return loss_fn(result)
        
        loss, grad = jax.value_and_grad(compute_loss)(x0)
        return loss, grad

    def jit_compile_step(self):
        """Return a JIT-compiled version of the step function."""
        # Store reference to system and solver outside of JIT
        system = self.system
        backend = self.backend
        solver_class = self._get_solver_instance
        is_imex = self.is_imex
        is_implicit = self.is_implicit
        
        @jax.jit
        def jitted_step(x, u, dt):
            # Pure JIT-compatible step (no stats tracking)
            x = jnp.asarray(x)
            
            # Handle autonomous systems
            if u is not None:
                u = jnp.asarray(u)
            
            def ode_func(t, state, args):
                return system(state, u, backend=backend)
            
            # Create appropriate term
            if is_imex:
                explicit_term = dfx.ODETerm(ode_func)
                implicit_term = dfx.ODETerm(lambda t, y, args: jnp.zeros_like(y))
                terms = dfx.MultiTerm(explicit_term, implicit_term)
            else:
                terms = dfx.ODETerm(ode_func)
            
            solver = solver_class()
            
            solution = dfx.diffeqsolve(
                terms, solver,
                t0=0.0, t1=dt,
                dt0=dt, y0=x,
                saveat=dfx.SaveAt(t1=True),
                stepsize_controller=dfx.ConstantStepSize(),
                max_steps=10
            )
            
            return solution.ys[0]
        
        return jitted_step

    def vectorized_step(self, x_batch: ArrayLike, u_batch: Optional[ArrayLike] = None, 
                       dt: Optional[float] = None):
        """Vectorized step over batch of states and controls."""
        return jax.vmap(lambda x, u: self.step(x, u, dt))(x_batch, u_batch)

    def vectorized_integrate(
        self,
        x0_batch: ArrayLike,
        u_func: Callable[[float, ArrayLike], Optional[ArrayLike]],
        t_span: Tuple[float, float],
        t_eval: Optional[ArrayLike] = None,
    ):
        """Vectorized integration over batch of initial conditions."""
        results = []
        for i in range(x0_batch.shape[0]):
            results.append(self.integrate(x0_batch[i], u_func, t_span, t_eval))
        return results
    
    # ========================================================================
    # Helper Methods for IMEX Systems
    # ========================================================================
    
    def integrate_imex(
        self,
        x0: ArrayLike,
        explicit_func: Callable[[float, ArrayLike], ArrayLike],
        implicit_func: Callable[[float, ArrayLike], ArrayLike],
        t_span: Tuple[float, float],
        t_eval: Optional[ArrayLike] = None,
    ) -> IntegrationResult:
        """
        Integrate IMEX system with separate explicit and implicit parts.
        
        For systems of the form:
            dx/dt = f_explicit(t, x) + f_implicit(t, x)
        
        where f_explicit is non-stiff and f_implicit is stiff.
        
        Parameters
        ----------
        x0 : ArrayLike
            Initial state
        explicit_func : Callable
            Non-stiff part: (t, x) -> dx/dt_explicit
        implicit_func : Callable
            Stiff part: (t, x) -> dx/dt_implicit
        t_span : Tuple[float, float]
            Integration interval (must have tf > t0)
        t_eval : Optional[ArrayLike]
            Evaluation times
            
        Returns
        -------
        IntegrationResult
            Integration result
            
        Raises
        ------
        ValueError
            If solver is not IMEX, IMEX solvers not available, or tf < t0
        
        Notes
        -----
        IMEX solvers require Diffrax 0.8.0 or later
        """
        if not self._imex_solver_map:
            raise ValueError(
                "IMEX solvers are not available. "
                "Requires Diffrax 0.8.0 or later. "
                f"Current Diffrax version: {dfx.__version__}"
            )
        
        if not self.is_imex:
            raise ValueError(
                f"Solver '{self.solver_name}' is not an IMEX solver. "
                f"Use one of: {list(self._imex_solver_map.keys())}"
            )
        
        t0, tf = t_span
        
        if tf < t0:
            raise ValueError(
                f"Backward integration not supported for IMEX. Got t_span=({t0}, {tf})"
            )
        
        x0 = jnp.asarray(x0)
        
        # Define explicit and implicit ODE functions
        def explicit_ode(t, state, args):
            return explicit_func(t, state)
        
        def implicit_ode(t, state, args):
            return implicit_func(t, state)
        
        # Create IMEX terms
        explicit_term = dfx.ODETerm(explicit_ode)
        implicit_term = dfx.ODETerm(implicit_ode)
        terms = dfx.MultiTerm(explicit_term, implicit_term)
        
        # Get IMEX solver
        solver = self._imex_solver_map[self.solver_name]()
        
        # Set up integration
        if t_eval is not None:
            t_points = jnp.asarray(t_eval)
            saveat = dfx.SaveAt(ts=t_points)
        else:
            n_dense = max(2, self.options.get("n_dense", 100))
            t_points = jnp.linspace(t0, tf, n_dense)
            saveat = dfx.SaveAt(ts=t_points)
        
        stepsize_controller = dfx.PIDController(
            rtol=self.rtol,
            atol=self.atol,
        )
        
        dt0_value = self.dt if self.dt is not None else (tf - t0) / 100
        adjoint_method = self._adjoint_map[self.adjoint_name]()
        
        # Solve
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
                adjoint=adjoint_method,
                throw=False,
            )
            
            success = jnp.all(jnp.isfinite(solution.ys))
            nsteps = int(solution.stats.get("num_steps", 0))
            nfev = int(solution.stats.get("num_steps", 0))
            
            return IntegrationResult(
                t=solution.ts,
                x=solution.ys,
                success=bool(success),
                message="IMEX integration successful" if success else "IMEX integration failed",
                nfev=nfev,
                nsteps=nsteps,
                solver=self.solver_name,
            )
            
        except Exception as e:
            return IntegrationResult(
                t=jnp.array([t0]),
                x=x0[None, :],
                success=False,
                message=f"IMEX integration failed: {str(e)}",
                nfev=0,
                nsteps=0,
            )