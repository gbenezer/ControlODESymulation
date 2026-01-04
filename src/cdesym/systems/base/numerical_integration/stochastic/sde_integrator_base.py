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
SDE Integrator Base - Abstract Interface for Stochastic Differential Equation Integration

Provides a unified interface for numerical integration of SDEs across different
backends (NumPy, PyTorch, JAX) with both fixed and adaptive time stepping.

This module extends the integrator framework to handle stochastic systems,
supporting both Itô and Stratonovich interpretations.

Mathematical Form
-----------------
Stochastic differential equations:

    dx = f(x, u, t)dt + g(x, u, t)dW

where:
    - f(x, u, t): Drift vector (nx × 1) - deterministic dynamics
    - g(x, u, t): Diffusion matrix (nx × nw) - stochastic intensity
    - dW: Brownian motion increments (nw independent Wiener processes)

Key Differences from ODE Integration
------------------------------------
1. **Random Noise**: Each step requires Brownian motion increments
2. **Weak vs Strong Convergence**: Different accuracy measures
3. **Noise Structure**: Additive, multiplicative, diagonal, scalar
4. **Multiple Trajectories**: Monte Carlo simulation support
5. **SDE Type**: Itô vs Stratonovich interpretation

Architecture Consistency
-----------------------
- Inherits from IntegratorBase for consistency
- Adds SDE-specific methods (drift, diffusion, noise handling)
- Maintains same backend-agnostic interface
- Compatible with StochasticDynamicalSystem

Design Note
-----------
This module now uses TypedDict-based result types, following the project
design principle: "Result types are TypedDict". This enables better type
safety, IDE autocomplete, and consistency across the codebase.
"""

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple

import numpy as np

from cdesym.systems.base.numerical_integration.integrator_base import (
    IntegratorBase,
    StepMode,
)
from cdesym.types.backends import ConvergenceType, SDEType
from cdesym.types.core import (
    ArrayLike,
    ControlVector,
    DiffusionMatrix,
    NoiseVector,
    ScalarLike,
    StateVector,
)
from cdesym.types.trajectories import (
    SDEIntegrationResult,
    TimePoints,
    TimeSpan,
)

if TYPE_CHECKING:

    from cdesym.systems.base.core.continuous_stochastic_system import ContinuousStochasticSystem


def get_trajectory_statistics(result: SDEIntegrationResult) -> Dict[str, Any]:
    """
    Compute trajectory statistics for Monte Carlo analysis.

    Parameters
    ----------
    result : SDEIntegrationResult
        SDE integration result with single or multiple paths

    Returns
    -------
    Dict[str, Any]
        Statistics including mean, std, quantiles across paths

    Examples
    --------
    >>> mc_result = integrator.integrate_monte_carlo(x0, u_func, (0, 10), n_paths=1000)
    >>> stats = get_trajectory_statistics(mc_result)
    >>> print(f"Mean at final time: {stats['mean'][-1]}")
    >>> print(f"Std at final time: {stats['std'][-1]}")
    """
    n_paths = result.get("n_paths", 1)
    x = result["x"]

    if n_paths == 1:
        return {
            "mean": x,
            "std": None,
            "n_paths": 1,
            "note": "Single trajectory - no statistics available",
        }

    # Detect backend from array type
    if hasattr(x, '__array__'):
        # NumPy array or convertible
        x_np = np.asarray(x)
        backend = 'numpy'
    elif hasattr(x, 'cpu'):
        # PyTorch tensor
        x_np = x.detach().cpu().numpy()
        backend = 'torch'
    elif hasattr(x, 'device'):
        # JAX array
        x_np = np.asarray(x)
        backend = 'jax'
    else:
        # Fallback to NumPy conversion
        x_np = np.asarray(x)
        backend = 'numpy'

    # Multiple paths: compute statistics on NumPy arrays
    # x_np has shape (n_paths, T, nx)
    return {
        "mean": np.mean(x_np, axis=0),  # (T, nx)
        "std": np.std(x_np, axis=0),  # (T, nx)
        "min": np.min(x_np, axis=0),
        "max": np.max(x_np, axis=0),
        "median": np.median(x_np, axis=0),
        "q25": np.quantile(x_np, 0.25, axis=0),
        "q75": np.quantile(x_np, 0.75, axis=0),
        "n_paths": n_paths,
        "backend": backend,
    }


class SDEIntegratorBase(IntegratorBase):
    """
    Abstract base class for SDE integrators.

    Extends IntegratorBase to handle stochastic differential equations with
    drift and diffusion terms. Provides unified interface for both Ito and
    Stratonovich SDEs across multiple backends.

    All SDE integrators must implement:
    - step(): Single integration step with noise
    - integrate(): Multi-step integration with trajectories
    - name: Integrator name

    Additional SDE-specific capabilities:
    - Multiple trajectory simulation (Monte Carlo)
    - Noise structure exploitation (additive, diagonal, etc.)
    - Weak and strong convergence schemes
    - Stratonovich correction terms

    Attributes
    ----------
    sde_system : StochasticDynamicalSystem
        SDE system to integrate
    sde_type : SDEType
        SDE interpretation (Ito or Stratonovich)
    convergence_type : ConvergenceType
        Convergence criterion (strong or weak)
    seed : Optional[int]
        Random seed for reproducibility

    Result Types
    ------------
    Returns SDEIntegrationResult TypedDict with:
    - t: Time points (T,)
    - x: State trajectory (T, nx) or (n_paths, T, nx)
    - diffusion_evals: Number of diffusion evaluations
    - noise_samples: Brownian increments used
    - n_paths: Number of trajectories
    - convergence_type: 'strong' or 'weak'
    - sde_type: 'ito' or 'stratonovich'

    Examples
    --------
    >>> # Create SDE integrator
    >>> integrator = EulerMaruyamaIntegrator(
    ...     sde_system,
    ...     dt=0.01,
    ...     backend='numpy',
    ...     seed=42
    ... )
    >>>
    >>> # Single trajectory
    >>> result = integrator.integrate(
    ...     x0=np.array([1.0, 0.0]),
    ...     u_func=lambda t, x: np.zeros(1),
    ...     t_span=(0.0, 10.0)
    ... )
    >>> print(f"State shape: {result['x'].shape}")
    >>>
    >>> # Multiple trajectories (Monte Carlo)
    >>> result = integrator.integrate_monte_carlo(
    ...     x0=np.array([1.0, 0.0]),
    ...     u_func=lambda t, x: np.zeros(1),
    ...     t_span=(0.0, 10.0),
    ...     n_paths=1000
    ... )
    >>> stats = get_trajectory_statistics(result)
    >>> print(f"Mean trajectory: {stats['mean']}")
    >>> print(f"Standard deviation: {stats['std']}")
    >>>
    >>> # Autonomous system
    >>> result = integrator.integrate(
    ...     x0=np.array([1.0, 0.0]),
    ...     u_func=lambda t, x: None,
    ...     t_span=(0.0, 10.0)
    ... )
    """

    def __init__(
        self,
        sde_system: "ContinuousStochasticSystem",
        dt: Optional[ScalarLike] = None,
        step_mode: StepMode = StepMode.FIXED,
        backend: str = "numpy",
        sde_type: Optional[SDEType] = None,
        convergence_type: ConvergenceType = ConvergenceType.STRONG,
        seed: Optional[int] = None,
        **options,
    ):
        """
        Initialize SDE integrator.

        Parameters
        ----------
        sde_system : StochasticDynamicalSystem
            SDE system to integrate (must have drift and diffusion)
        dt : Optional[float]
            Time step:
            - FIXED mode: Required, constant step size
            - ADAPTIVE mode: Initial guess (if supported)
        step_mode : StepMode
            FIXED or ADAPTIVE stepping (most SDE solvers are FIXED)
        backend : str
            Backend to use ('numpy', 'torch', 'jax')
        sde_type : Optional[SDEType]
            SDE interpretation (None = use system's type)
        convergence_type : ConvergenceType
            Strong (pathwise) or weak (moment) convergence
        seed : Optional[int]
            Random seed for reproducibility
        **options : dict
            Additional options:
            - cache_noise : bool
                Cache noise samples (default: False)
            - noise_samples : ArrayLike
                Pre-generated noise (for deterministic testing)
        """
        # Validate system type - check for required SDE methods
        if not hasattr(sde_system, "get_sde_type"):
            raise TypeError(
                f"sde_system must be a StochasticDynamicalSystem with get_sde_type() method, "
                f"got {type(sde_system).__name__}",
            )
        if not hasattr(sde_system, "get_diffusion_matrix"):
            raise TypeError(
                f"sde_system must be a StochasticDynamicalSystem with get_diffusion_matrix() method, "
                f"got {type(sde_system).__name__}",
            )

        # Initialize base integrator
        super().__init__(sde_system, dt, step_mode, backend, **options)

        self.sde_system = sde_system

        # SDE type (use system's if not specified)
        if sde_type is None:
            self.sde_type = sde_system.get_sde_type()
        else:
            self.sde_type = sde_type

        self.convergence_type = convergence_type
        self.seed = seed

        # Random number generator
        self._rng = self._initialize_rng(backend, seed)

        # SDE-specific statistics
        self._stats["diffusion_evals"] = 0
        self._stats["noise_samples"] = 0

        # Noise dimension
        self.nw = sde_system.nw

        # Analyze noise structure for optimization
        self._is_additive = sde_system.is_additive_noise()
        self._is_multiplicative = not self._is_additive
        self._is_diagonal = sde_system.is_diagonal_noise()
        self._is_scalar = sde_system.is_scalar_noise()

        # Cache diffusion matrix if additive
        self._cached_diffusion = None
        if self._is_additive:
            # For additive noise, diffusion is constant
            dummy_x = np.zeros(sde_system.nx)
            dummy_u = np.zeros(sde_system.nu) if sde_system.nu > 0 else None
            self._cached_diffusion = sde_system.get_diffusion_matrix(
                dummy_x,
                dummy_u,
                backend=backend,
            )

    def _initialize_rng(self, backend: str, seed: Optional[int]):
        """Initialize random number generator for backend."""
        if backend == "numpy":
            return np.random.default_rng(seed)
        if backend == "torch":
            import torch

            if seed is not None:
                torch.manual_seed(seed)
            return None  # PyTorch uses global state
        if backend == "jax":
            import jax

            if seed is not None:
                return jax.random.PRNGKey(seed)
            return jax.random.PRNGKey(0)
        return None

    def _generate_noise(
        self,
        shape: Tuple[int, ...],
        dt: Optional[ScalarLike] = None,
    ) -> NoiseVector:
        """
        Generate Brownian motion increments.

        Parameters
        ----------
        shape : tuple
            Shape of noise to generate
        dt : Optional[float]
            Time step for scaling (uses self.dt if None)

        Returns
        -------
        ArrayLike
            Brownian increments ~ N(0, dt*I)
        """
        self._stats["noise_samples"] += 1

        if self.backend == "numpy":
            dW = self._rng.standard_normal(shape)
        elif self.backend == "torch":
            import torch

            dW = torch.randn(shape, dtype=torch.float64)
        elif self.backend == "jax":
            import jax

            self._rng, subkey = jax.random.split(self._rng)
            dW = jax.random.normal(subkey, shape)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

        # Scale by sqrt(dt) for Brownian motion
        dt_to_use = dt if dt is not None else self.dt
        if dt_to_use is not None:
            dW = dW * np.sqrt(dt_to_use)

        return dW

    def _evaluate_diffusion(
        self,
        x: StateVector,
        u: Optional[ControlVector] = None,
    ) -> DiffusionMatrix:
        """
        Evaluate diffusion matrix with caching for additive noise.

        Parameters
        ----------
        x : ArrayLike
            State
        u : Optional[ArrayLike]
            Control

        Returns
        -------
        ArrayLike
            Diffusion matrix G(x, u)
        """
        # Use cached value for additive noise
        if self._is_additive and self._cached_diffusion is not None:
            return self._cached_diffusion

        # Evaluate diffusion
        self._stats["diffusion_evals"] += 1
        return self.sde_system.get_diffusion_matrix(x, u, backend=self.backend)

    def _evaluate_drift(self, x: StateVector, u: Optional[ControlVector] = None) -> ArrayLike:
        """
        Evaluate drift term (deterministic dynamics).

        This is an alias for _evaluate_dynamics() providing clearer semantics
        in the SDE context where we distinguish between drift and diffusion.

        Parameters
        ----------
        x : ArrayLike
            State
        u : Optional[ArrayLike]
            Control

        Returns
        -------
        ArrayLike
            Drift vector f(x, u)

        Notes
        -----
        For SDE: dx = f(x,u) dt + g(x,u) dW
        - f(x,u) is the drift (this method)
        - g(x,u) is the diffusion (_evaluate_diffusion)

        This method is semantically equivalent to _evaluate_dynamics() from
        the base IntegratorBase class, but provides clearer naming for SDE code.

        Examples
        --------
        >>> # Evaluate drift and diffusion separately
        >>> f = integrator._evaluate_drift(x, u)      # Deterministic part
        >>> g = integrator._evaluate_diffusion(x, u)  # Stochastic part
        >>> # Euler-Maruyama step: x_next = x + f*dt + g*dW
        """
        return self._evaluate_dynamics(x, u)

    # ========================================================================
    # Abstract Methods (Must be Implemented by Subclasses)
    # ========================================================================

    @abstractmethod
    def step(
        self,
        x: StateVector,
        u: Optional[ControlVector] = None,
        dt: Optional[ScalarLike] = None,
        dW: Optional[NoiseVector] = None,
    ) -> ArrayLike:
        """
        Take one SDE integration step with noise.

        Parameters
        ----------
        x : ArrayLike
            Current state (nx,)
        u : Optional[ArrayLike]
            Control input (nu,) or None for autonomous
        dt : Optional[float]
            Step size (uses self.dt if None)
        dW : Optional[ArrayLike]
            Brownian increment (nw,) - if None, generates random

        Returns
        -------
        ArrayLike
            Next state x(t + dt)

        Notes
        -----
        Subclasses implement specific SDE methods:
        - Euler-Maruyama (order 0.5 strong, order 1 weak)
        - Milstein (order 1 strong)
        - Runge-Kutta SDE methods
        - etc.

        Examples
        --------
        >>> x_next = integrator.step(x, u, dt=0.01)
        >>> # With custom noise
        >>> dW = np.random.randn(nw) * np.sqrt(0.01)
        >>> x_next = integrator.step(x, u, dt=0.01, dW=dW)
        """

    @abstractmethod
    def integrate(
        self,
        x0: StateVector,
        u_func: Callable[[ScalarLike, StateVector], Optional[ControlVector]],
        t_span: TimeSpan,
        t_eval: Optional[TimePoints] = None,
        dense_output: bool = False,
    ) -> SDEIntegrationResult:
        """
        Integrate SDE over time interval.

        **API Level**: This is a **low-level stochastic integration method** that directly
        interfaces with numerical SDE solvers. For typical use cases, prefer the high-level
        `simulate()` method if available in your stochastic system class.

        **Control Function Convention**: This method uses the **scipy/ODE solver convention**
        where control functions have signature **(t, x) → u**, with time as the FIRST argument.
        This differs from high-level simulation APIs which use **(x, t) → u** with state as
        the primary argument. The difference is intentional:

        - **Low-level** `integrate()`: Uses (t, x) for direct solver compatibility
        - **High-level** `simulate()`: Uses (x, t) for intuitive control-theoretic API

        **Stochastic Integration**: Unlike deterministic ODE integration, SDE integration
        involves random Brownian motion increments. Each call with the same initial condition
        will produce different trajectories unless the random seed is fixed. For statistical
        analysis, use `integrate_monte_carlo()` to simulate multiple paths.

        Parameters
        ----------
        x0 : ArrayLike
            Initial state (nx,)
        u_func : Callable[[float, ArrayLike], Optional[ArrayLike]]
            Control policy with **low-level convention**: (t, x) → u
            - t: float - current time (**FIRST** argument, scipy convention)
            - x: ArrayLike - current state (**SECOND** argument)
            - Returns: Optional[ArrayLike] - control input u, or None for autonomous

            Can be:
            - Autonomous: lambda t, x: None
            - Constant control: lambda t, x: u_const
            - State feedback: lambda t, x: -K @ x
            - Time-varying: lambda t, x: u(t)
            - Stochastic policy: lambda t, x: policy(x) + noise
        t_span : Tuple[float, float]
            Integration interval (t_start, t_end)
        t_eval : Optional[ArrayLike]
            Specific times at which to store solution
            If None, uses solver's internal time points (typically uniform grid)
            For SDEs, irregular grids may affect statistical properties
        dense_output : bool
            If True, return dense interpolated solution (if supported by solver)
            Note: Most SDE solvers do not support dense output

        Returns
        -------
        SDEIntegrationResult
            TypedDict containing:
            - t: Time points (T,)
            - x: State trajectory (T, nx) - **time-major ordering**
            - success: Whether integration succeeded
            - message: Status message
            - nfev: Number of drift function evaluations
            - diffusion_evals: Number of diffusion function evaluations
            - noise_samples: Number of Brownian motion samples generated
            - nsteps: Number of integration steps taken
            - integration_time: Computation time (seconds)
            - solver: Integrator name
            - convergence_type: 'strong' or 'weak' convergence
            - sde_type: 'ito' or 'stratonovich' interpretation
            - n_paths: Number of trajectories (1 for single path)

        Raises
        ------
        RuntimeError
            If integration fails (numerical instability, step size issues, etc.)

        Notes
        -----
        **Stochastic Nature**: Each call generates a new random trajectory. For reproducible
        results, set the integrator's random seed via `set_seed()` before calling integrate().

        **Convergence Types**:
        - **Strong convergence**: Pathwise accuracy - each trajectory is accurate
        - **Weak convergence**: Moment accuracy - statistics (mean, variance) are accurate

        **SDE Interpretations**:
        - **Itô**: Most common, natural for stochastic calculus
        - **Stratonovich**: Physics-based, matches ordinary calculus rules

        **Time-Major Ordering**: Unlike some high-level APIs that use (nx, T) for backward
        compatibility, integrate() returns (T, nx) time-major ordering for consistency with
        numerical solver conventions and efficient time-series operations.

        Examples
        --------
        **Low-level integrate() usage** (uses (t, x) convention):

        Autonomous stochastic system:

        >>> result = integrator.integrate(
        ...     x0=np.array([1.0, 0.0]),
        ...     u_func=lambda t, x: None,  # Autonomous
        ...     t_span=(0.0, 10.0)
        ... )
        >>> print(f"Final state: {result['x'][-1]}")
        >>> print(f"Convergence: {result['convergence_type']}")

        Controlled stochastic system with constant control:

        >>> result = integrator.integrate(
        ...     x0=np.array([1.0, 0.0]),
        ...     u_func=lambda t, x: np.array([0.5]),  # Note: (t, x) order
        ...     t_span=(0.0, 10.0)
        ... )
        >>> print(f"Diffusion evaluations: {result['diffusion_evals']}")

        State feedback for stochastic stabilization:

        >>> K = np.array([[1.0, 2.0]])
        >>> result = integrator.integrate(
        ...     x0=np.array([1.0, 0.0]),
        ...     u_func=lambda t, x: -K @ x,  # (t, x) order for integrate()
        ...     t_span=(0.0, 10.0)
        ... )

        Time-varying stochastic control:

        >>> result = integrator.integrate(
        ...     x0=np.array([1.0, 0.0]),
        ...     u_func=lambda t, x: np.array([0.5 * np.sin(t)]),
        ...     t_span=(0.0, 10.0)
        ... )

        Reproducible stochastic simulation:

        >>> integrator.set_seed(42)
        >>> result1 = integrator.integrate(
        ...     x0=np.array([1.0, 0.0]),
        ...     u_func=lambda t, x: None,
        ...     t_span=(0.0, 10.0)
        ... )
        >>> integrator.set_seed(42)
        >>> result2 = integrator.integrate(
        ...     x0=np.array([1.0, 0.0]),
        ...     u_func=lambda t, x: None,
        ...     t_span=(0.0, 10.0)
        ... )
        >>> np.allclose(result1['x'], result2['x'])  # True - same random trajectory
        True

        Evaluate at specific times:

        >>> t_eval = np.linspace(0, 10, 101)
        >>> result = integrator.integrate(
        ...     x0=np.array([1.0, 0.0]),
        ...     u_func=lambda t, x: None,
        ...     t_span=(0, 10),
        ...     t_eval=t_eval
        ... )
        >>> assert len(result['t']) == 101

        **Monte Carlo simulation** (multiple trajectories):

        For statistical analysis, use `integrate_monte_carlo()`:

        >>> # Simulate 1000 paths
        >>> mc_result = integrator.integrate_monte_carlo(
        ...     x0=np.array([1.0, 0.0]),
        ...     u_func=lambda t, x: None,
        ...     t_span=(0.0, 10.0),
        ...     n_paths=1000
        ... )
        >>> # Result has shape (n_paths, T, nx)
        >>> print(mc_result['x'].shape)  # (1000, 101, 2)
        >>>
        >>> # Compute statistics
        >>> from controldesymulation.systems.base.numerical_integration.stochastic.sde_integrator_base import get_trajectory_statistics
        >>> stats = get_trajectory_statistics(mc_result)
        >>> print(f"Mean at t=10: {stats['mean'][-1]}")
        >>> print(f"Std at t=10: {stats['std'][-1]}")

        **High-level simulate() usage** (if available, uses (x, t) convention):

        If your stochastic system provides a high-level `simulate()` method, prefer it
        for the more intuitive (x, t) convention:

        >>> # Controller with (x, t) order - state is primary
        >>> def controller(x, t):  # Note: (x, t) order
        ...     K = np.array([[1.0, 2.0]])
        ...     return -K @ x
        >>>
        >>> result = sde_system.simulate(
        ...     x0=np.array([1.0, 0.0]),
        ...     controller=controller,  # Uses (x, t) signature
        ...     t_span=(0.0, 10.0),
        ...     dt=0.01
        ... )

        **Converting between conventions**:

        If you have a controller designed for simulate() and need to use integrate():

        >>> # Controller for simulate() - uses (x, t)
        >>> def my_controller(x, t):
        ...     return -K @ x
        >>>
        >>> # Wrap for integrate() - convert to (t, x)
        >>> result = integrator.integrate(
        ...     x0=x0,
        ...     u_func=lambda t, x: my_controller(x, t),  # Swap argument order
        ...     t_span=(0, 10)
        ... )

        **Comparing SDE with ODE integration**:

        >>> # Deterministic (ODE) - same result every time
        >>> ode_result = ode_integrator.integrate(x0, u_func, t_span)
        >>>
        >>> # Stochastic (SDE) - different result each time
        >>> sde_result1 = sde_integrator.integrate(x0, u_func, t_span)
        >>> sde_result2 = sde_integrator.integrate(x0, u_func, t_span)
        >>> # Trajectories will differ due to randomness

        See Also
        --------
        integrate_monte_carlo : Simulate multiple paths for statistical analysis
        get_trajectory_statistics : Compute statistics from Monte Carlo results
        set_seed : Set random seed for reproducibility
        step : Single SDE integration step with noise
        simulate : High-level simulation with (x, t) convention (if available)
        """

    # ========================================================================
    # Monte Carlo Simulation
    # ========================================================================

    def integrate_monte_carlo(
        self,
        x0: StateVector,
        u_func: Callable[[ScalarLike, StateVector], Optional[ControlVector]],
        t_span: TimeSpan,
        n_paths: int,
        t_eval: Optional[TimePoints] = None,
        store_paths: bool = True,
        parallel: bool = False,
    ) -> SDEIntegrationResult:
        """
        Integrate multiple SDE trajectories for Monte Carlo analysis.

        Simulates n_paths independent realizations of the SDE to estimate
        statistical properties (mean, variance, probability distributions).

        Parameters
        ----------
        x0 : ArrayLike
            Initial state (nx,)
        u_func : Callable
            Control policy (or None for autonomous)
        t_span : Tuple[float, float]
            Time interval
        n_paths : int
            Number of trajectories to simulate
        t_eval : Optional[ArrayLike]
            Evaluation times
        store_paths : bool
            If True, store all trajectories (memory intensive)
            If False, only compute statistics online
        parallel : bool
            If True, use parallel execution (if backend supports)

        Returns
        -------
        SDEIntegrationResult
            Result with shape (n_paths, T, nx) if store_paths=True
            Result with statistics only if store_paths=False

        Examples
        --------
        >>> # Monte Carlo with 1000 paths
        >>> result = integrator.integrate_monte_carlo(
        ...     x0=np.array([1.0, 0.0]),
        ...     u_func=lambda t, x: np.zeros(1),
        ...     t_span=(0.0, 10.0),
        ...     n_paths=1000
        ... )
        >>>
        >>> # Get statistics
        >>> stats = get_trajectory_statistics(result)
        >>> print(f"Mean at t=10: {stats['mean'][-1]}")
        >>> print(f"Std at t=10: {stats['std'][-1]}")
        >>>
        >>> # Confidence intervals
        >>> lower = stats['mean'] - 1.96 * stats['std']
        >>> upper = stats['mean'] + 1.96 * stats['std']
        """
        if store_paths:
            # Store all trajectories
            all_paths = []

            for i in range(n_paths):
                # Integrate single path
                result = self.integrate(x0, u_func, t_span, t_eval)
                all_paths.append(result["x"])

            # Stack trajectories
            if self.backend == "numpy":
                x_all = np.stack(all_paths, axis=0)  # (n_paths, T, nx)
            elif self.backend == "torch":
                import torch

                x_all = torch.stack(all_paths, dim=0)
            elif self.backend == "jax":
                import jax.numpy as jnp

                x_all = jnp.stack(all_paths, axis=0)

            mc_result: SDEIntegrationResult = {
                "t": result["t"],
                "x": x_all,
                "success": True,
                "message": f"Monte Carlo with {n_paths} paths completed",
                "nfev": self._stats["total_fev"],
                "nsteps": self._stats["total_steps"],
                "diffusion_evals": self._stats["diffusion_evals"],
                "n_paths": n_paths,
                "convergence_type": self.convergence_type.value,
                "solver": self.name,
                "sde_type": self.sde_type.value,
            }
            return mc_result
        # Online statistics (memory efficient)
        raise NotImplementedError(
            "Online statistics not yet implemented. Use store_paths=True.",
        )

    # ========================================================================
    # SDE-Specific Utilities
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
        >>> integrator.set_seed(42)
        >>> result1 = integrator.integrate(x0, u_func, t_span)
        >>> integrator.set_seed(42)
        >>> result2 = integrator.integrate(x0, u_func, t_span)
        >>> # result1 and result2 are identical
        """
        self.seed = seed
        self._rng = self._initialize_rng(self.backend, seed)

        # Also set global seed for backend consistency
        if self.backend == "numpy":
            np.random.seed(seed)
        elif self.backend == "torch":
            import torch

            torch.manual_seed(seed)
        # JAX uses explicit keys, handled in _initialize_rng

    def get_noise_info(self) -> Dict[str, Any]:
        """
        Get information about noise structure and optimizations.

        Returns
        -------
        Dict[str, Any]
            Noise structure information
        """
        return {
            "nw": self.nw,
            "is_additive": self._is_additive,
            "is_multiplicative": self._is_multiplicative,
            "is_diagonal": self._is_diagonal,
            "is_scalar": self._is_scalar,
            "cached_diffusion": self._cached_diffusion is not None,
            "noise_type": self.sde_system.get_noise_type().value,
        }

    def get_sde_stats(self) -> Dict[str, Any]:
        """
        Get SDE-specific integration statistics.

        Returns
        -------
        Dict[str, Any]
            Statistics including drift/diffusion evaluations
        """
        base_stats = self.get_stats()
        sde_stats = {
            "diffusion_evals": self._stats["diffusion_evals"],
            "noise_samples": self._stats["noise_samples"],
            "avg_diffusion_per_step": (
                self._stats["diffusion_evals"] / max(1, self._stats["total_steps"])
            ),
        }
        return {**base_stats, **sde_stats}

    def _apply_stratonovich_correction(
        self,
        x: StateVector,
        u: Optional[ControlVector],
        g: DiffusionMatrix,
        dt: ScalarLike,
    ) -> ArrayLike:
        """
        Apply Stratonovich correction term.

        Converts Stratonovich SDE to Itô form by adding drift correction:

            correction = 0.5 * Σ_j g_j · (∂g_j/∂x)

        where g_j is the j-th column of diffusion matrix G.

        This allows Stratonovich SDEs to be integrated using Itô methods
        by modifying the drift term.

        Parameters
        ----------
        x : ArrayLike
            Current state (nx,)
        u : Optional[ArrayLike]
            Control input (nu,) or None
        g : ArrayLike
            Diffusion matrix at current state (nx, nw)
        dt : float
            Time step (not used in correction, included for API consistency)

        Returns
        -------
        ArrayLike
            Correction term to add to drift (nx,)

        Notes
        -----
        For additive noise (g constant), correction is zero.
        For multiplicative noise, uses automatic differentiation when available.

        Implementation uses:
        - NumPy: Finite differences
        - PyTorch: Autograd
        - JAX: jax.jacobian

        Examples
        --------
        >>> # Geometric Brownian motion: dx = μx dt + σx dW (Stratonovich)
        >>> # Itô form: dx = (μ + 0.5*σ²)x dt + σx dW
        >>> correction = integrator._apply_stratonovich_correction(x, u, g, dt)
        >>> # correction = 0.5 * σ² * x for GBM
        """
        # For additive noise, correction is exactly zero
        if self._is_additive:
            return np.zeros_like(x)

        # Compute Stratonovich correction
        nx = len(x)
        nw = g.shape[1] if len(g.shape) > 1 else 1

        if self.backend == "numpy":
            return self._stratonovich_correction_numpy(x, u, g, nx, nw)
        if self.backend == "torch":
            return self._stratonovich_correction_torch(x, u, g, nx, nw)
        if self.backend == "jax":
            return self._stratonovich_correction_jax(x, u, g, nx, nw)
        raise ValueError(f"Unknown backend: {self.backend}")

    def _stratonovich_correction_numpy(
        self,
        x: StateVector,
        u: Optional[ControlVector],
        g: DiffusionMatrix,
        nx: int,
        nw: int,
    ) -> ArrayLike:
        """Compute Stratonovich correction using finite differences (NumPy)."""
        import numpy as np

        correction = np.zeros(nx)
        eps = 1e-7  # Finite difference step

        # For each noise dimension j
        for j in range(nw):
            g_j = g[:, j] if nw > 1 else g.flatten()

            # Compute Jacobian ∂g_j/∂x using finite differences
            # ∂g_j/∂x_i ≈ (g_j(x + eps*e_i) - g_j(x - eps*e_i)) / (2*eps)
            dg_j_dx = np.zeros((nx, nx))

            for i in range(nx):
                # Perturb state in i-th direction
                x_plus = x.copy()
                x_minus = x.copy()
                x_plus[i] += eps
                x_minus[i] -= eps

                # Evaluate diffusion at perturbed states
                g_plus = self._evaluate_diffusion(x_plus, u)
                g_minus = self._evaluate_diffusion(x_minus, u)

                # Extract j-th column
                g_j_plus = g_plus[:, j] if nw > 1 else g_plus.flatten()
                g_j_minus = g_minus[:, j] if nw > 1 else g_minus.flatten()

                # Finite difference approximation
                dg_j_dx[:, i] = (g_j_plus - g_j_minus) / (2 * eps)

            # Add contribution: 0.5 * g_j · (∂g_j/∂x)
            correction += 0.5 * dg_j_dx @ g_j

        return correction

    def _stratonovich_correction_torch(
        self,
        x: StateVector,
        u: Optional[ControlVector],
        g: DiffusionMatrix,
        nx: int,
        nw: int,
    ) -> ArrayLike:
        """Compute Stratonovich correction using autograd (PyTorch)."""
        import torch

        # Ensure tensors require grad
        x_torch = torch.as_tensor(x, dtype=torch.float64).requires_grad_(True)
        u_torch = torch.as_tensor(u, dtype=torch.float64) if u is not None else None

        correction = torch.zeros(nx, dtype=torch.float64)

        # For each noise dimension j
        for j in range(nw):
            # Define function to differentiate: g_j(x)
            def diffusion_column_j(x_var):
                g_val = self.sde_system.get_diffusion_matrix(x_var, u_torch, backend="torch")
                return g_val[:, j] if nw > 1 else g_val.flatten()

            # Compute g_j at current x
            g_j = diffusion_column_j(x_torch)

            # Compute Jacobian ∂g_j/∂x using autograd
            dg_j_dx = torch.zeros((nx, nx), dtype=torch.float64)
            for i in range(nx):
                # Compute ∂(g_j)_i/∂x
                grad_outputs = torch.zeros_like(g_j)
                grad_outputs[i] = 1.0

                # Clear previous gradients
                if x_torch.grad is not None:
                    x_torch.grad.zero_()

                # Compute gradient
                g_j_i = diffusion_column_j(x_torch)[i]
                g_j_i.backward(retain_graph=True)

                if x_torch.grad is not None:
                    dg_j_dx[i, :] = x_torch.grad.clone()

            # Add contribution: 0.5 * g_j · (∂g_j/∂x)
            correction += 0.5 * (dg_j_dx @ g_j.detach())

        return correction.detach().numpy()

    def _stratonovich_correction_jax(
        self,
        x: StateVector,
        u: Optional[ControlVector],
        g: DiffusionMatrix,
        nx: int,
        nw: int,
    ) -> ArrayLike:
        """Compute Stratonovich correction using jax.jacobian (JAX)."""
        import jax
        import jax.numpy as jnp

        x_jax = jnp.asarray(x)
        u_jax = jnp.asarray(u) if u is not None else None

        correction = jnp.zeros(nx)

        # For each noise dimension j
        for j in range(nw):
            # Define function for j-th column of diffusion
            def diffusion_column_j(x_var):
                g_val = self.sde_system.get_diffusion_matrix(x_var, u_jax, backend="jax")
                return g_val[:, j] if nw > 1 else g_val.flatten()

            # Compute g_j
            g_j = diffusion_column_j(x_jax)

            # Compute Jacobian using JAX
            jacobian_fn = jax.jacobian(diffusion_column_j)
            dg_j_dx = jacobian_fn(x_jax)  # Shape: (nx, nx)

            # Add contribution: 0.5 * (∂g_j/∂x) @ g_j
            correction += 0.5 * (dg_j_dx @ g_j)

        return np.asarray(correction)

    def reset_stats(self):
        """Reset all statistics including SDE-specific counters."""
        super().reset_stats()
        self._stats["diffusion_evals"] = 0
        self._stats["noise_samples"] = 0

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"{self.__class__.__name__}("
            f"dt={self.dt}, mode={self.step_mode.value}, "
            f"backend={self.backend}, sde_type={self.sde_type.value}, "
            f"convergence={self.convergence_type.value})"
        )

    def __str__(self) -> str:
        """Human-readable string."""
        noise_str = " (additive)" if self._is_additive else " (multiplicative)"
        return f"{self.name} (dt={self.dt:.4f}, {self.backend}, {self.sde_type.value}{noise_str})"
