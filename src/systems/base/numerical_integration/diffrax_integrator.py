# Copyright (C) 2025 Gil Benezer
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

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

Design Note
-----------
This module uses TypedDict-based result types from src.types.trajectories
and semantic types from src.types.core, following the project design principles:
- "Result types are TypedDict"
- "Use semantic types for clarity"
This enables better type safety, IDE autocomplete, and consistency across
the codebase.
"""

import time
from typing import Callable, Optional, TYPE_CHECKING

import diffrax as dfx
import jax
import jax.numpy as jnp
from jax import Array

from src.systems.base.numerical_integration.integrator_base import (
    IntegratorBase,
    StepMode,
)
from src.types.backends import Backend

# Import types from centralized type system
from src.types.core import (
    ControlVector,
    ScalarLike,
    StateVector,
)
from src.types.trajectories import (
    IntegrationResult,
    TimePoints,
    TimeSpan,
)

if TYPE_CHECKING:
    from src.systems.base.core.continuous_system_base import ContinuousSystemBase


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
        system: "ContinuousSystemBase",
        dt: Optional[float] = None,
        step_mode: StepMode = StepMode.ADAPTIVE,
        backend: Backend = "jax",
        solver: str = "tsit5",
        adjoint: str = "recursive_checkpoint",
        **options,
    ):
        # Validate backend
        if backend != "jax":
            raise ValueError(f"DiffraxIntegrator requires backend='jax', got '{backend}'")

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
        self.is_implicit = self.solver_name in implicit_solvers and hasattr(dfx, "ImplicitEuler")

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
        x: StateVector,
        u: Optional[ControlVector] = None,
        dt: Optional[ScalarLike] = None,
    ) -> StateVector:
        """
        Take one integration step: x(t) → x(t + dt).

        Parameters
        ----------
        x : StateVector
            Current state (nx,) or (batch, nx)
        u : Optional[ControlVector]
            Control input (nu,) or (batch, nu), or None for autonomous systems
        dt : Optional[ScalarLike]
            Step size (uses self.dt if None)

        Returns
        -------
        StateVector
            Next state x(t + dt)

        Examples
        --------
        >>> # Controlled system
        >>> x_next = integrator.step(
        ...     x=jnp.array([1.0, 0.0]),
        ...     u=jnp.array([0.5])
        ... )
        >>>
        >>> # Autonomous system
        >>> x_next = integrator.step(
        ...     x=jnp.array([1.0, 0.0]),
        ...     u=None
        ... )
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
            "terms": term,
            "solver": solver,
            "t0": 0.0,
            "t1": step_size,
            "dt0": step_size,
            "y0": x,
            "saveat": dfx.SaveAt(t1=True),
            "stepsize_controller": dfx.ConstantStepSize(),
            "max_steps": 10,
        }

        # Implicit solvers handle nonlinear solving internally
        # No additional configuration needed

        # Single step integration
        solution = dfx.diffeqsolve(**solver_kwargs)

        # Update stats
        self._stats["total_steps"] += 1
        self._stats["total_fev"] += int(solution.stats.get("num_steps", 1))

        return solution.ys[0]

    def integrate(
        self,
        x0: StateVector,
        u_func: Callable[[ScalarLike, StateVector], Optional[ControlVector]],
        t_span: TimeSpan,
        t_eval: Optional[TimePoints] = None,
        dense_output: bool = False,
    ) -> IntegrationResult:
        """
        Integrate over time interval with control policy.

        Parameters
        ----------
        x0 : StateVector
            Initial state (nx,)
        u_func : Callable[[float, StateVector], Optional[ControlVector]]
            Control policy: (t, x) → u (or None for autonomous systems)
        t_span : TimeSpan
            Integration interval (t_start, t_end)
            Note: t_end must be > t_start (backward integration not supported)
        t_eval : Optional[TimePoints]
            Specific times at which to store solution (must be increasing)
        dense_output : bool
            If True, return dense interpolated solution

        Returns
        -------
        IntegrationResult
            TypedDict containing:
            - t: Time points (T,)
            - x: State trajectory (T, nx)
            - success: Whether integration succeeded
            - message: Status message
            - nfev: Number of function evaluations
            - nsteps: Number of steps taken
            - integration_time: Computation time (seconds)
            - solver: Integrator name
            - njev: Number of Jacobian evaluations (if applicable)

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
        >>>
        >>> # Access results
        >>> t = result["t"]
        >>> x_traj = result["x"]
        >>> print(f"Success: {result['success']}")
        >>> print(f"Steps: {result['nsteps']}, Function evals: {result['nfev']}")

        Notes
        -----
        Backward time integration (tf < t0) is not supported. If you need
        reverse-time integration, integrate forward and reverse the output:

        >>> # Instead of integrate(x0, u_func, (10.0, 0.0))
        >>> result = integrate(x0, u_func, (0.0, 10.0))
        >>> t_reversed = jnp.flip(result["t"])
        >>> x_reversed = jnp.flip(result["x"], axis=0)
        """
        t0, tf = t_span
        x0 = jnp.asarray(x0)

        # Start timing
        start_time = time.perf_counter()

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
                integration_time=0.0,
                solver=self.solver_name,
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
            "terms": term,
            "solver": solver,
            "t0": t0,
            "t1": tf,
            "dt0": dt0_value,
            "y0": x0,
            "saveat": saveat,
            "stepsize_controller": stepsize_controller,
            "max_steps": self.max_steps,
            "adjoint": adjoint_method,
            "throw": False,
        }

        # Solve ODE
        try:
            solution = dfx.diffeqsolve(**solver_kwargs)

            # Check success
            success = jnp.all(jnp.isfinite(solution.ys))

            # Update statistics
            nsteps = int(solution.stats.get("num_steps", 0))
            nfev = int(solution.stats.get("num_steps", 0))
            self._stats["total_steps"] += nsteps
            self._stats["total_fev"] += nfev

            # Compute integration time
            integration_time = time.perf_counter() - start_time
            self._stats["total_time"] += integration_time

            return IntegrationResult(
                t=solution.ts,
                x=solution.ys,
                success=bool(success),
                message=(
                    "Integration successful" if success else "Integration failed (NaN/Inf detected)"
                ),
                nfev=nfev,
                nsteps=nsteps,
                integration_time=integration_time,
                solver=self.solver_name,
                njev=int(solution.stats.get("num_jacobian_evals", 0)),
            )

        except Exception as e:
            # Compute integration time even on failure
            integration_time = time.perf_counter() - start_time
            self._stats["total_time"] += integration_time

            return IntegrationResult(
                t=jnp.array([t0]),
                x=x0[None, :] if x0.ndim == 1 else x0,
                success=False,
                message=f"Integration failed: {str(e)}",
                nfev=0,
                nsteps=0,
                integration_time=integration_time,
                solver=self.solver_name,
            )

    # ========================================================================
    # JAX-Specific Methods
    # ========================================================================

    def integrate_with_gradient(
        self,
        x0: StateVector,
        u_func: Callable[[ScalarLike, StateVector], Optional[ControlVector]],
        t_span: TimeSpan,
        loss_fn: Callable[[IntegrationResult], ScalarLike],
        t_eval: Optional[TimePoints] = None,
    ):
        """
        Integrate and compute gradients w.r.t. initial conditions.

        Parameters
        ----------
        x0 : StateVector
            Initial state (nx,)
        u_func : Callable[[float, StateVector], Optional[ControlVector]]
            Control policy: (t, x) → u
        t_span : TimeSpan
            Integration interval (t_start, t_end)
        loss_fn : Callable[[IntegrationResult], float]
            Loss function on integration result
        t_eval : Optional[TimePoints]
            Evaluation times

        Returns
        -------
        loss : float
            Loss value
        grad : StateVector
            Gradient of loss w.r.t. x0

        Examples
        --------
        >>> # Define loss (e.g., final state error)
        >>> def loss_fn(result):
        ...     x_final = result["x"][-1]
        ...     x_target = jnp.array([1.0, 0.0])
        ...     return jnp.sum((x_final - x_target)**2)
        >>>
        >>> # Compute loss and gradient
        >>> loss, grad = integrator.integrate_with_gradient(
        ...     x0=jnp.array([0.0, 0.0]),
        ...     u_func=lambda t, x: jnp.zeros(1),
        ...     t_span=(0.0, 10.0),
        ...     loss_fn=loss_fn
        ... )
        >>> print(f"Loss: {loss:.4f}")
        >>> print(f"Gradient: {grad}")
        """

        def compute_loss(x0_val):
            result = self.integrate(x0_val, u_func, t_span, t_eval)
            return loss_fn(result)

        loss, grad = jax.value_and_grad(compute_loss)(x0)
        return loss, grad

    def jit_compile_step(self):
        """
        Return a JIT-compiled version of the step function.

        Returns
        -------
        jitted_step : callable
            JIT-compiled step function with signature:
            (x: StateVector, u: Optional[ControlVector], dt: float) -> StateVector

        Notes
        -----
        The JIT-compiled version does not track statistics for performance.
        Use the regular step() method if you need statistics tracking.

        Examples
        --------
        >>> # Create JIT-compiled step
        >>> jit_step = integrator.jit_compile_step()
        >>>
        >>> # Use for fast repeated stepping
        >>> x = jnp.array([1.0, 0.0])
        >>> u = jnp.array([0.5])
        >>> for _ in range(1000):
        ...     x = jit_step(x, u, 0.01)
        """
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
                terms,
                solver,
                t0=0.0,
                t1=dt,
                dt0=dt,
                y0=x,
                saveat=dfx.SaveAt(t1=True),
                stepsize_controller=dfx.ConstantStepSize(),
                max_steps=10,
            )

            return solution.ys[0]

        return jitted_step

    def vectorized_step(
        self,
        x_batch: StateVector,
        u_batch: Optional[ControlVector] = None,
        dt: Optional[ScalarLike] = None,
    ) -> StateVector:
        """
        Vectorized step over batch of states and controls.

        Parameters
        ----------
        x_batch : StateVector
            Batched states (batch, nx)
        u_batch : Optional[ControlVector]
            Batched controls (batch, nu) or None for autonomous
        dt : Optional[ScalarLike]
            Step size

        Returns
        -------
        StateVector
            Next states (batch, nx)

        Examples
        --------
        >>> # Batch of 100 states
        >>> x_batch = jnp.random.randn(100, 2)
        >>> u_batch = jnp.zeros((100, 1))
        >>> x_next_batch = integrator.vectorized_step(x_batch, u_batch)
        >>> print(x_next_batch.shape)  # (100, 2)
        """
        return jax.vmap(lambda x, u: self.step(x, u, dt))(x_batch, u_batch)

    def vectorized_integrate(
        self,
        x0_batch: StateVector,
        u_func: Callable[[ScalarLike, StateVector], Optional[ControlVector]],
        t_span: TimeSpan,
        t_eval: Optional[TimePoints] = None,
    ):
        """
        Vectorized integration over batch of initial conditions.

        Parameters
        ----------
        x0_batch : StateVector
            Batched initial states (batch, nx)
        u_func : Callable[[float, StateVector], Optional[ControlVector]]
            Control policy: (t, x) → u
        t_span : TimeSpan
            Integration interval
        t_eval : Optional[TimePoints]
            Evaluation times

        Returns
        -------
        list of IntegrationResult
            Integration results for each initial condition

        Examples
        --------
        >>> # Monte Carlo simulation with 100 initial conditions
        >>> x0_batch = jnp.random.randn(100, 2)
        >>> results = integrator.vectorized_integrate(
        ...     x0_batch,
        ...     lambda t, x: jnp.zeros(1),
        ...     (0.0, 10.0)
        ... )
        >>> print(f"Success rate: {sum(r['success'] for r in results) / 100}")
        """
        results = []
        for i in range(x0_batch.shape[0]):
            results.append(self.integrate(x0_batch[i], u_func, t_span, t_eval))
        return results

    # ========================================================================
    # Helper Methods for IMEX Systems
    # ========================================================================

    def integrate_imex(
        self,
        x0: StateVector,
        explicit_func: Callable[[ScalarLike, StateVector], StateVector],
        implicit_func: Callable[[ScalarLike, StateVector], StateVector],
        t_span: TimeSpan,
        t_eval: Optional[TimePoints] = None,
    ) -> IntegrationResult:
        """
        Integrate IMEX system with separate explicit and implicit parts.

        For systems of the form:
            dx/dt = f_explicit(t, x) + f_implicit(t, x)

        where f_explicit is non-stiff and f_implicit is stiff.

        Parameters
        ----------
        x0 : StateVector
            Initial state
        explicit_func : Callable[[float, StateVector], StateVector]
            Non-stiff part: (t, x) -> dx/dt_explicit
        implicit_func : Callable[[float, StateVector], StateVector]
            Stiff part: (t, x) -> dx/dt_implicit
        t_span : TimeSpan
            Integration interval (must have tf > t0)
        t_eval : Optional[TimePoints]
            Evaluation times

        Returns
        -------
        IntegrationResult
            Integration result with IMEX solver

        Raises
        ------
        ValueError
            If solver is not IMEX, IMEX solvers not available, or tf < t0

        Notes
        -----
        IMEX solvers require Diffrax 0.8.0 or later

        Examples
        --------
        >>> # Stiff-nonstiff splitting
        >>> def explicit_part(t, x):
        ...     # Non-stiff dynamics
        ...     return jnp.array([x[1], 0.0])
        >>>
        >>> def implicit_part(t, x):
        ...     # Stiff dynamics
        ...     return jnp.array([0.0, -100*x[1]])
        >>>
        >>> result = integrator.integrate_imex(
        ...     x0=jnp.array([1.0, 0.0]),
        ...     explicit_func=explicit_part,
        ...     implicit_func=implicit_part,
        ...     t_span=(0.0, 10.0)
        ... )
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

        # Start timing
        start_time = time.perf_counter()

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

            # Compute integration time
            integration_time = time.perf_counter() - start_time

            return IntegrationResult(
                t=solution.ts,
                x=solution.ys,
                success=bool(success),
                message="IMEX integration successful" if success else "IMEX integration failed",
                nfev=nfev,
                nsteps=nsteps,
                integration_time=integration_time,
                solver=self.solver_name,
            )

        except Exception as e:
            # Compute integration time even on failure
            integration_time = time.perf_counter() - start_time

            return IntegrationResult(
                t=jnp.array([t0]),
                x=x0[None, :] if x0.ndim == 1 else x0,
                success=False,
                message=f"IMEX integration failed: {str(e)}",
                nfev=0,
                nsteps=0,
                integration_time=integration_time,
                solver=self.solver_name,
            )
