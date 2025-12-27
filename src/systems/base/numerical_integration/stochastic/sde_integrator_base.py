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
supporting both ItΓ΄ and Stratonovich interpretations.

Mathematical Form
-----------------
Stochastic differential equations:

    dx = f(x, u, t)dt + g(x, u, t)dW

where:
    - f(x, u, t): Drift vector (nx Γ— 1) - deterministic dynamics
    - g(x, u, t): Diffusion matrix (nx Γ— nw) - stochastic intensity
    - dW: Brownian motion increments (nw independent Wiener processes)

Key Differences from ODE Integration
------------------------------------
1. **Random Noise**: Each step requires Brownian motion increments
2. **Weak vs Strong Convergence**: Different accuracy measures
3. **Noise Structure**: Additive, multiplicative, diagonal, scalar
4. **Multiple Trajectories**: Monte Carlo simulation support
5. **SDE Type**: ItΓ΄ vs Stratonovich interpretation

Architecture Consistency
-----------------------
- Inherits from IntegratorBase for consistency
- Adds SDE-specific methods (drift, diffusion, noise handling)
- Maintains same backend-agnostic interface
- Compatible with StochasticDynamicalSystem
"""

from abc import abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from src.systems.base.numerical_integration.integrator_base import (
    IntegratorBase,
    StepMode,
)
from src.systems.base.utils.stochastic.noise_analysis import SDEType

from src.types import ArrayLike

if TYPE_CHECKING:
    import jax.numpy as jnp
    import torch

    from src.systems.base.stochastic_dynamical_system import StochasticDynamicalSystem


class ConvergenceType(Enum):
    """
    Convergence criterion for SDE solvers.

    Attributes
    ----------
    STRONG : str
        Strong convergence - pathwise accuracy
        Error: E[|X(T) - X_h(T)|] = O(h^p)
        Best for: Single trajectory accuracy, control applications

    WEAK : str
        Weak convergence - moment accuracy
        Error: |E[f(X(T))] - E[f(X_h(T))]| = O(h^p)
        Best for: Statistical properties, Monte Carlo estimation
    """

    STRONG = "strong"
    WEAK = "weak"


class SDEIntegrationResult:
    """
    Container for SDE integration results.

    Extends IntegrationResult with SDE-specific information like
    multiple trajectories, noise samples, and convergence statistics.

    Attributes
    ----------
    t : ArrayLike
        Time points (T,)
    x : ArrayLike
        State trajectory (T, nx) or (n_paths, T, nx) for multiple paths
    success : bool
        Whether integration succeeded
    message : str
        Status message
    nfev : int
        Number of drift function evaluations
    nsteps : int
        Number of integration steps taken
    diffusion_evals : int
        Number of diffusion function evaluations
    noise_samples : Optional[ArrayLike]
        Brownian motion samples used (T, nw) or (n_paths, T, nw)
    n_paths : int
        Number of trajectories computed
    convergence_type : ConvergenceType
        Type of convergence (strong or weak)
    solver : str
        Solver method used
    sde_type : SDEType
        SDE interpretation (Ito or Stratonovich)
    """

    def __init__(
        self,
        t: ArrayLike,
        x: ArrayLike,
        success: bool = True,
        message: str = "SDE integration successful",
        nfev: int = 0,
        nsteps: int = 0,
        diffusion_evals: int = 0,
        noise_samples: Optional[ArrayLike] = None,
        n_paths: int = 1,
        convergence_type: ConvergenceType = ConvergenceType.STRONG,
        solver: Optional[str] = None,
        sde_type: SDEType = SDEType.ITO,
        **metadata,
    ):
        self.t = t
        self.x = x
        self.success = success
        self.message = message
        self.nfev = nfev
        self.nsteps = nsteps
        self.diffusion_evals = diffusion_evals
        self.noise_samples = noise_samples
        self.n_paths = n_paths
        self.convergence_type = convergence_type
        self.solver = solver
        self.sde_type = sde_type
        self.metadata = metadata

    def __repr__(self) -> str:
        return (
            f"SDEIntegrationResult("
            f"success={self.success}, "
            f"nsteps={self.nsteps}, "
            f"nfev={self.nfev}, "
            f"diffusion_evals={self.diffusion_evals}, "
            f"n_paths={self.n_paths})"
        )

    def get_statistics(self) -> Dict[str, Any]:
        """
        Compute trajectory statistics for Monte Carlo analysis.

        Returns
        -------
        Dict[str, Any]
            Statistics including mean, std, quantiles across paths
        """
        if self.n_paths == 1:
            return {
                "mean": self.x,
                "std": None,
                "n_paths": 1,
                "note": "Single trajectory - no statistics available",
            }

        # Multiple paths: compute statistics
        # x has shape (n_paths, T, nx)
        return {
            "mean": np.mean(self.x, axis=0),  # (T, nx)
            "std": np.std(self.x, axis=0),  # (T, nx)
            "min": np.min(self.x, axis=0),
            "max": np.max(self.x, axis=0),
            "median": np.median(self.x, axis=0),
            "q25": np.quantile(self.x, 0.25, axis=0),
            "q75": np.quantile(self.x, 0.75, axis=0),
            "n_paths": self.n_paths,
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
    >>>
    >>> # Multiple trajectories (Monte Carlo)
    >>> result = integrator.integrate_monte_carlo(
    ...     x0=np.array([1.0, 0.0]),
    ...     u_func=lambda t, x: np.zeros(1),
    ...     t_span=(0.0, 10.0),
    ...     n_paths=1000
    ... )
    >>> stats = result.get_statistics()
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
        sde_system: "StochasticDynamicalSystem",
        dt: Optional[float] = None,
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
                Cache noise samples for efficiency (default: True for additive)
            - noise_generator : Callable
                Custom noise generator (advanced)

        Raises
        ------
        ValueError
            If system is not stochastic
            If FIXED mode specified without dt
        TypeError
            If sde_system is not StochasticDynamicalSystem
        """
        # Validate system type
        from src.systems.base.stochastic_dynamical_system import StochasticDynamicalSystem

        if not isinstance(sde_system, StochasticDynamicalSystem):
            raise TypeError(
                f"sde_system must be StochasticDynamicalSystem, " f"got {type(sde_system).__name__}"
            )

        # Initialize parent (handles ODE functionality)
        super().__init__(sde_system, dt, step_mode, backend, **options)

        # SDE-specific attributes
        self.sde_system = sde_system
        self.nw = sde_system.nw  # Number of Wiener processes

        # SDE type (use system's if not specified)
        if sde_type is None:
            self.sde_type = sde_system.sde_type
        else:
            self.sde_type = sde_type

        self.convergence_type = convergence_type
        self.seed = seed

        # Initialize random state
        self._rng = self._initialize_rng(backend, seed)

        # SDE-specific statistics
        self._stats["diffusion_evals"] = 0
        self._stats["noise_samples"] = 0

        # Noise optimization flags
        self._is_additive = sde_system.is_additive_noise()
        self._is_multiplicative = sde_system.is_multiplicative_noise()
        self._is_diagonal = sde_system.is_diagonal_noise()
        self._is_scalar = sde_system.is_scalar_noise()

        # Cache constant diffusion for additive noise
        self._cached_diffusion = None
        if self._is_additive:
            try:
                # CRITICAL: Get constant noise in correct backend
                self._cached_diffusion = self.sde_system.get_constant_noise(
                    backend=self.backend  # Pass backend
                )
            except Exception as e:
                # If caching fails, will evaluate each time
                pass

    def _initialize_rng(self, backend: str, seed: Optional[int]):
        """
        Initialize random number generator for specified backend.

        Parameters
        ----------
        backend : str
            Backend name
        seed : Optional[int]
            Random seed

        Returns
        -------
        RNG object appropriate for backend
        """
        if backend == "numpy":
            # Create a new RandomState instance for reproducibility
            if seed is not None:
                return np.random.RandomState(seed)
            return np.random.RandomState()
        elif backend == "torch":
            import torch

            if seed is not None:
                torch.manual_seed(seed)
            # Return the torch module itself (has randn method)
            return torch
        elif backend == "jax":
            import jax

            if seed is not None:
                return jax.random.PRNGKey(seed)
            return jax.random.PRNGKey(0)
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def _cache_constant_diffusion(self):
        """
        Cache constant diffusion matrix for additive noise systems.

        For additive noise, g(x,u,t) is constant, so we compute once
        and reuse for significant performance gains.
        """
        if self._is_additive:
            # Evaluate at arbitrary point (result is constant)
            x_dummy = np.zeros(self.sde_system.nx)
            u_dummy = np.zeros(self.sde_system.nu) if self.sde_system.nu > 0 else None

            # CRITICAL: Pass backend parameter
            self._cached_diffusion = self.sde_system.get_constant_noise(backend=self.backend)

    # ========================================================================
    # SDE-Specific Evaluation Methods
    # ========================================================================

    def _evaluate_drift(self, x, u=None):
        """
        Evaluate drift term f(x, u).

        Parameters
        ----------
        x : ArrayLike
            Current state
        u : ArrayLike, optional
            Current control (None for autonomous systems)

        Returns
        -------
        ArrayLike
            Drift vector f(x, u), shape (nx,)
            Type matches self.backend

        Notes
        -----
        Ensures backend consistency by passing backend parameter.
        """
        # Defensive check
        if hasattr(x, "shape") and x.ndim > 1 and x.shape[0] == 0:
            raise RuntimeError(
                f"Internal error: SDE integrator received empty batch. "
                f"x.shape={x.shape}. This is a bug - please report."
            )
        # CRITICAL: Pass backend to ensure type consistency
        f = self.sde_system.drift(x, u, backend=self.backend)  # Add backend parameter
        self._stats["total_fev"] += 1
        return f

    def _evaluate_diffusion(self, x, u=None):
        """
        Evaluate diffusion term g(x, u).

        For additive noise, uses cached constant matrix.
        For multiplicative noise, evaluates at current state.

        Parameters
        ----------
        x : ArrayLike
            Current state
        u : ArrayLike, optional
            Current control (None for autonomous systems)

        Returns
        -------
        ArrayLike
            Diffusion matrix g(x, u), shape (nx, nw)
            Type matches self.backend

        Notes
        -----
        This method ensures backend consistency by explicitly passing
        backend parameter to system.diffusion().
        """
        # Defensive check
        if hasattr(x, "shape") and x.ndim > 1 and x.shape[0] == 0:
            raise RuntimeError(
                f"Internal error: SDE integrator received empty batch in diffusion. "
                f"x.shape={x.shape}. This is a bug - please report."
            )

        if self._is_additive and self._cached_diffusion is not None:
            # Use cached constant diffusion
            self._stats["diffusion_cache_hits"] = self._stats.get("diffusion_cache_hits", 0) + 1
            return self._cached_diffusion
        else:
            # Evaluate diffusion (multiplicative or not cached)
            # CRITICAL: Pass backend to ensure type consistency
            g = self.sde_system.diffusion(x, u, backend=self.backend)  # Add backend parameter
            self._stats["diffusion_evals"] += 1
            return g

    def _generate_noise(self, shape: Tuple[int, ...], dt: float) -> ArrayLike:
        """
        Generate Brownian motion increments: dW ~ N(0, dt).

        Parameters
        ----------
        shape : Tuple[int, ...]
            Shape of noise array (typically (nw,) or (batch, nw))
        dt : float
            Time step (variance = dt)

        Returns
        -------
        ArrayLike
            Brownian increments with correct backend type
        """
        self._stats["noise_samples"] += 1
        sqrt_dt = np.sqrt(dt)

        if self.backend == "numpy":
            return self._rng.randn(*shape) * sqrt_dt
        elif self.backend == "torch":
            import torch

            return torch.randn(*shape) * sqrt_dt
        elif self.backend == "jax":
            import jax

            # JAX requires explicit key management
            key, subkey = jax.random.split(self._rng)
            self._rng = key  # Update stored key
            return jax.random.normal(subkey, shape) * sqrt_dt

    def _apply_stratonovich_correction(
        self, x: ArrayLike, u: Optional[ArrayLike], g: ArrayLike, dt: float
    ) -> ArrayLike:
        """
        Apply Stratonovich correction term if using Stratonovich interpretation.

        The Stratonovich-to-Ito conversion adds a correction term:
            f_strat(x) = f_ito(x) + 0.5 * sum_j g_j * dg_j/dx

        Parameters
        ----------
        x : ArrayLike
            State
        u : Optional[ArrayLike]
            Control
        g : ArrayLike
            Diffusion matrix (nx, nw)
        dt : float
            Time step

        Returns
        -------
        ArrayLike
            Correction term to add to drift

        Notes
        -----
        This is only needed for Stratonovich SDEs. For Ito SDEs, returns zero.
        The correction requires computing dg/dx, which can be expensive.
        """
        if self.sde_type == SDEType.ITO:
            # No correction for Ito
            if self.backend == "numpy":
                return np.zeros_like(x)
            elif self.backend == "torch":
                import torch

                return torch.zeros_like(x)
            elif self.backend == "jax":
                import jax.numpy as jnp

                return jnp.zeros_like(x)

        # Stratonovich correction: 0.5 * sum_j g_j * dg_j/dx
        # This requires computing Jacobian of diffusion w.r.t. state
        # For now, return zero and document that Stratonovich requires
        # specialized implementation or automatic differentiation

        # TODO: Implement Stratonovich correction using autodiff
        # This requires:
        # 1. Compute dg/dx for each noise dimension
        # 2. Sum: 0.5 * sum_j g_j * (dg_j/dx)

        raise NotImplementedError(
            "Stratonovich correction not yet implemented. "
            "Use sde_type=SDEType.ITO or implement correction in subclass."
        )

    # ========================================================================
    # Abstract Methods (Must be Implemented by Subclasses)
    # ========================================================================

    @abstractmethod
    def step(
        self,
        x: ArrayLike,
        u: Optional[ArrayLike] = None,
        dt: Optional[float] = None,
        dW: Optional[ArrayLike] = None,
    ) -> ArrayLike:
        """
        Take one SDE integration step: x(t) β†' x(t + dt).

        Parameters
        ----------
        x : ArrayLike
            Current state (nx,) or (batch, nx)
        u : Optional[ArrayLike]
            Control input (nu,) or (batch, nu), or None for autonomous
        dt : Optional[float]
            Step size (uses self.dt if None)
        dW : Optional[ArrayLike]
            Brownian increments (nw,) or (batch, nw)
            If None, generated automatically

        Returns
        -------
        ArrayLike
            Next state x(t + dt)

        Notes
        -----
        Subclasses implement specific SDE schemes:
        - Euler-Maruyama (order 0.5 strong, 1.0 weak)
        - Milstein (order 1.0 strong)
        - Runge-Kutta SDE methods
        - etc.

        Examples
        --------
        >>> x_next = integrator.step(x, u, dt=0.01)
        >>> # With custom noise
        >>> dW = np.random.randn(nw) * np.sqrt(0.01)
        >>> x_next = integrator.step(x, u, dt=0.01, dW=dW)
        """
        pass

    @abstractmethod
    def integrate(
        self,
        x0: ArrayLike,
        u_func: Callable[[float, ArrayLike], Optional[ArrayLike]],
        t_span: Tuple[float, float],
        t_eval: Optional[ArrayLike] = None,
        dense_output: bool = False,
    ) -> SDEIntegrationResult:
        """
        Integrate SDE over time interval.

        Parameters
        ----------
        x0 : ArrayLike
            Initial state (nx,)
        u_func : Callable[[float, ArrayLike], Optional[ArrayLike]]
            Control policy: (t, x) β†' u (or None for autonomous)
        t_span : Tuple[float, float]
            Integration interval (t_start, t_end)
        t_eval : Optional[ArrayLike]
            Specific times at which to store solution
        dense_output : bool
            If True, return dense interpolated solution (if supported)

        Returns
        -------
        SDEIntegrationResult
            Integration result with trajectory and noise samples

        Examples
        --------
        >>> result = integrator.integrate(
        ...     x0=np.array([1.0, 0.0]),
        ...     u_func=lambda t, x: np.zeros(1),
        ...     t_span=(0.0, 10.0)
        ... )
        >>>
        >>> # Autonomous system
        >>> result = integrator.integrate(
        ...     x0=np.array([1.0, 0.0]),
        ...     u_func=lambda t, x: None,
        ...     t_span=(0.0, 10.0)
        ... )
        """
        pass

    # ========================================================================
    # Monte Carlo Simulation
    # ========================================================================

    def integrate_monte_carlo(
        self,
        x0: ArrayLike,
        u_func: Callable[[float, ArrayLike], Optional[ArrayLike]],
        t_span: Tuple[float, float],
        n_paths: int,
        t_eval: Optional[ArrayLike] = None,
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
        >>> stats = result.get_statistics()
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
                all_paths.append(result.x)

            # Stack trajectories
            if self.backend == "numpy":
                x_all = np.stack(all_paths, axis=0)  # (n_paths, T, nx)
            elif self.backend == "torch":
                import torch

                x_all = torch.stack(all_paths, dim=0)
            elif self.backend == "jax":
                import jax.numpy as jnp

                x_all = jnp.stack(all_paths, axis=0)

            return SDEIntegrationResult(
                t=result.t,
                x=x_all,
                success=True,
                message=f"Monte Carlo with {n_paths} paths completed",
                nfev=self._stats["total_fev"],
                nsteps=self._stats["total_steps"],
                diffusion_evals=self._stats["diffusion_evals"],
                n_paths=n_paths,
                convergence_type=self.convergence_type,
                solver=self.name,
                sde_type=self.sde_type,
            )
        else:
            # Online statistics (memory efficient)
            raise NotImplementedError(
                "Online statistics not yet implemented. Use store_paths=True."
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
        return (
            f"{self.name} (dt={self.dt:.4f}, {self.backend}, " f"{self.sde_type.value}{noise_str})"
        )
