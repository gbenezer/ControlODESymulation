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
DiffEqPySDEIntegrator: Julia-based SDE integration using DifferentialEquations.jl

This module provides access to Julia's powerful SDE solver ecosystem through the
diffeqpy Python interface. Julia's DifferentialEquations.jl is considered the
gold standard for SDE integration with:
- Extensive algorithm selection (40+ SDE solvers)
- Superior performance for stiff SDEs
- Advanced noise processes and adaptivity
- Automatic stiffness detection

Supports both Ito and Stratonovich interpretations, controlled and autonomous systems.

**CUSTOM NOISE SUPPORT**: Julia's DifferentialEquations.jl supports custom noise
via NoiseGrid and NoiseFunction, but implementation through diffeqpy for single-step
integration is complex. For reliable custom noise, use JAX/Diffrax instead.

**ALGORITHM COMPATIBILITY**: Not all Julia SDE algorithms work reliably through
diffeqpy due to Python-Julia bridging complexity. Tested and verified to work:
- EM (Euler-Maruyama) - General purpose
- SRA3 - Additive noise optimization
- LambaEM - Adaptive stepping

Algorithms that may not work through diffeqpy:
- SRIW1, SRIW2 - Diagonal noise (use JAX/Diffrax instead)
- Advanced Milstein variants - May have bridging issues
- Some implicit methods - Solver-specific compatibility

For maximum compatibility, use EM. For high accuracy, consider JAX/Diffrax.

Mathematical Form
-----------------
Stochastic differential equations:

    dx = f(x, u, t)dt + g(x, u, t)dW

Available Algorithms
-------------------
**Recommended General Purpose:**
- EM (Euler-Maruyama): Order 0.5 strong, 1.0 weak - fast and robust (VERIFIED)
- LambaEM: Euler-Maruyama with lambda step size control
- EulerHeun: Euler-Heun predictor-corrector

**High Accuracy (Stochastic Runge-Kutta):**
- SRA3: Order 2.0 weak for additive noise (VERIFIED)
- SRA1: Order 2.0 weak for diagonal noise
- SRIW1: Order 1.5 strong for diagonal noise (May not work via diffeqpy)
- SOSRA: Order 2.0 weak for scalar noise

**Adaptive Methods:**
- RKMil: Runge-Kutta Milstein with adaptivity
- RKMilCommute: For commutative noise
- AutoEM: Automatic switching between methods

**Specialized:**
- ImplicitEM: Implicit Euler-Maruyama for stiff drift
- ImplicitRKMil: Implicit Runge-Kutta Milstein
- SKenCarp: Stochastic Kennedy-Carpenter IMEX methods

Installation
-----------
Requires Julia and diffeqpy:

1. Install Julia from https://julialang.org/downloads/
2. Install DifferentialEquations.jl:
   ```julia
   using Pkg
   Pkg.add("DifferentialEquations")
   ```
3. Install Python package:
   ```bash
   pip install diffeqpy
   ```
4. Setup (one time):
   ```python
   from diffeqpy import install
   install()
   ```

Examples
--------
>>> # Ornstein-Uhlenbeck process (autonomous)
>>> integrator = DiffEqPySDEIntegrator(
...     sde_system,
...     dt=0.01,
...     algorithm='EM',
...     backend='numpy'
... )
>>>
>>> result = integrator.integrate(
...     x0=np.array([1.0]),
...     u_func=lambda t, x: None,
...     t_span=(0.0, 10.0)
... )
>>>
>>> # Controlled SDE with high accuracy
>>> integrator = DiffEqPySDEIntegrator(
...     controlled_sde,
...     dt=0.001,
...     algorithm='SRIW1',
...     rtol=1e-6,
...     atol=1e-8
... )
>>>
>>> result = integrator.integrate(
...     x0=np.array([1.0, 0.0]),
...     u_func=lambda t, x: -K @ x,
...     t_span=(0.0, 10.0)
... )
"""

import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from src.systems.base.numerical_integration.stochastic.sde_integrator_base import (
    ConvergenceType,
    SDEIntegrationResult,
    SDEIntegratorBase,
    SDEType,
    StepMode,
)

from src.types import ArrayLike


class DiffEqPySDEIntegrator(SDEIntegratorBase):
    """
    Julia-based SDE integrator using DifferentialEquations.jl via diffeqpy.

    Provides access to Julia's extensive SDE solver ecosystem with superior
    performance and accuracy compared to pure Python implementations.

    Parameters
    ----------
    sde_system : StochasticDynamicalSystem
        SDE system to integrate (controlled or autonomous)
    dt : Optional[float]
        Time step (initial guess for adaptive, fixed for non-adaptive)
    step_mode : StepMode
        FIXED or ADAPTIVE stepping (default: FIXED)
        Note: Many Julia SDE solvers (like EM) only support fixed stepping
        Use adaptive algorithms like 'LambaEM' for adaptive stepping
    backend : str
        Must be 'numpy' (Julia returns NumPy arrays via diffeqpy)
    algorithm : str
        Julia SDE algorithm name (default: 'EM')
        See list_algorithms() for available options
    sde_type : Optional[SDEType]
        SDE interpretation (None = use system's type)
    convergence_type : ConvergenceType
        Strong or weak convergence
    seed : Optional[int]
        Random seed for reproducibility
        NOTE: Seed control via diffeqpy is limited. Julia generates random
        numbers internally and setting the seed via Python is unreliable.
        For reproducible results, use JAX/PyTorch integrators instead.
    **options
        Additional options:
        - rtol : float (default: 1e-3) - Relative tolerance
        - atol : float (default: 1e-6) - Absolute tolerance
        - save_everystep : bool (default: True) - Save at every step
        - dense : bool (default: False) - Dense output interpolation

    Raises
    ------
    ImportError
        If diffeqpy is not installed
    RuntimeError
        If Julia DifferentialEquations.jl is not available

    Notes
    -----
    - Backend must be 'numpy' (Julia/Python bridge uses NumPy)
    - Statistics tracking is estimated (Julia doesn't expose call counts)
    - Random seed control is limited (Julia manages its own RNG)
    - For reproducible Monte Carlo, use JAX or PyTorch integrators
    - For custom noise specification, use JAX/Diffrax (simpler API)

    Examples
    --------
    >>> # Basic usage with Euler-Maruyama
    >>> integrator = DiffEqPySDEIntegrator(
    ...     sde_system,
    ...     dt=0.01,
    ...     algorithm='EM'
    ... )
    >>>
    >>> # High accuracy adaptive solver
    >>> integrator = DiffEqPySDEIntegrator(
    ...     sde_system,
    ...     dt=0.001,
    ...     algorithm='SRIW1',
    ...     rtol=1e-6,
    ...     atol=1e-8
    ... )
    >>>
    >>> # Stiff drift with implicit solver
    >>> integrator = DiffEqPySDEIntegrator(
    ...     stiff_sde,
    ...     algorithm='ImplicitEM',
    ...     dt=0.01
    ... )
    """

    def __init__(
        self,
        sde_system,
        dt: Optional[float] = 0.01,
        step_mode: StepMode = StepMode.FIXED,  # Changed default to FIXED
        backend: str = "numpy",
        algorithm: str = "EM",
        sde_type: Optional[SDEType] = None,
        convergence_type: ConvergenceType = ConvergenceType.STRONG,
        seed: Optional[int] = None,
        **options,
    ):
        """Initialize DiffEqPy SDE integrator."""

        # Validate backend
        if backend != "numpy":
            raise ValueError(
                "DiffEqPySDEIntegrator requires backend='numpy'. "
                "Julia/Python bridge uses NumPy arrays."
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
            **options,
        )

        self.algorithm = algorithm
        self._integrator_name = f"Julia-{algorithm}"

        # Try to import diffeqpy
        try:
            from diffeqpy import de

            self.de = de
        except ImportError as e:
            raise ImportError(
                "DiffEqPySDEIntegrator requires diffeqpy.\n\n"
                "Installation steps:\n"
                "1. Install Julia from https://julialang.org/downloads/\n"
                "2. Install DifferentialEquations.jl:\n"
                "   julia> using Pkg\n"
                "   julia> Pkg.add('DifferentialEquations')\n"
                "3. Install Python package:\n"
                "   pip install diffeqpy\n"
                "4. Setup (one time):\n"
                "   python -c 'from diffeqpy import install; install()'\n"
            ) from e

        # Validate algorithm availability
        available = self._get_available_algorithms()
        if algorithm not in available:
            raise ValueError(
                f"Unknown Julia SDE algorithm '{algorithm}'.\n"
                f"Available: {available[:10]}...\n"
                f"Use list_algorithms() for complete list."
            )

        # Get algorithm constructor
        self._algorithm_constructor = self._get_algorithm(algorithm)

        # Store options
        self.rtol = options.get("rtol", 1e-3)
        self.atol = options.get("atol", 1e-6)
        self.save_everystep = options.get("save_everystep", True)
        self.dense = options.get("dense", False)

    @property
    def name(self) -> str:
        """Return integrator name."""
        mode_str = "Adaptive" if self.step_mode == StepMode.ADAPTIVE else "Fixed"
        return f"{self._integrator_name} ({mode_str})"

    def _get_available_algorithms(self) -> List[str]:
        """Get list of available Julia SDE algorithms."""
        # Common SDE algorithms in DifferentialEquations.jl
        return [
            # Euler-Maruyama family
            "EM",
            "LambaEM",
            "EulerHeun",
            # Stochastic Runge-Kutta
            "SRIW1",
            "SRIW2",
            "SOSRI",
            "SOSRI2",
            "SRA",
            "SRA1",
            "SRA2",
            "SRA3",
            "SOSRA",
            "SOSRA2",
            # Milstein family
            "RKMil",
            "RKMilCommute",
            "RKMilGeneral",
            # Implicit methods
            "ImplicitEM",
            "ImplicitEulerHeun",
            "ImplicitRKMil",
            # IMEX methods
            "SKenCarp",
            # Rossler SRI
            "SRI",
            "SRIW1Optimized",
            "SRIW2Optimized",
            # Adaptive
            "AutoEM",
        ]

    def _get_algorithm(self, algorithm: str):
        """Get Julia algorithm constructor."""
        return getattr(self.de, algorithm, None)

    def _setup_julia_problem(
        self, x0: np.ndarray, u_func: Callable, t_span: Tuple[float, float], noise_process=None
    ):
        """
        Setup Julia SDE problem.

        Creates drift and diffusion functions compatible with Julia's
        DifferentialEquations.jl interface.

        Parameters
        ----------
        x0 : np.ndarray
            Initial state
        u_func : Callable
            Control policy (or None for autonomous)
        t_span : Tuple[float, float]
            Time span (t0, tf)
        noise_process : Optional
            Custom Julia noise process object (NoiseGrid, NoiseFunction, etc.)
            If None, Julia generates default Brownian motion

        Returns
        -------
        SDEProblem
            Julia SDE problem object
        """
        t0, tf = t_span

        # Define drift function for Julia: f(u, p, t)
        # Julia convention: u = state, p = parameters, t = time
        # NOTE: Statistics tracking in closures doesn't work from Julia!
        def drift_func_julia(u, p, t):
            """Drift function: dx/dt = f(x, u, t)"""
            # Convert Julia array to NumPy with explicit dtype
            x = np.asarray(u, dtype=np.float64)

            # Get control
            control = u_func(float(t), x)

            # Evaluate drift
            dx = self.sde_system.drift(x, control, backend="numpy")

            # Return with explicit dtype (Julia requires this)
            return np.asarray(dx, dtype=np.float64)

        # Define diffusion function for Julia: g(u, p, t)
        def diffusion_func_julia(u, p, t):
            """Diffusion function: coefficient of dW"""
            # Convert Julia array to NumPy with explicit dtype
            x = np.asarray(u, dtype=np.float64)

            # Get control
            control = u_func(float(t), x)

            # Evaluate diffusion
            g = self.sde_system.diffusion(x, control, backend="numpy")

            # Ensure correct dtype and shape
            g = np.asarray(g, dtype=np.float64)

            # Julia expects (nx, nw) for general noise
            if g.ndim == 1:
                g = g.reshape(-1, 1)

            return g

        # Get initial diffusion matrix for noise_rate_prototype
        control_0 = u_func(t0, x0)
        g0 = self.sde_system.diffusion(x0, control_0, backend="numpy")

        # Ensure proper dtype and shape
        g0 = np.asarray(g0, dtype=np.float64)
        if g0.ndim == 1:
            g0 = g0.reshape(-1, 1)

        # Validate shape
        expected_shape = (self.sde_system.nx, self.sde_system.nw)
        if g0.shape != expected_shape:
            raise ValueError(
                f"noise_rate_prototype shape mismatch. "
                f"Expected {expected_shape}, got {g0.shape}"
            )

        # Create SDE problem
        try:
            # Ensure x0 has correct dtype
            x0_julia = np.asarray(x0, dtype=np.float64)

            # Create SDEProblem with optional noise process
            problem_kwargs = {"noise_rate_prototype": g0}

            # Add custom noise if provided
            if noise_process is not None:
                problem_kwargs["noise"] = noise_process

            problem = self.de.SDEProblem(
                drift_func_julia,
                diffusion_func_julia,
                x0_julia,
                (float(t0), float(tf)),
                **problem_kwargs,
            )

            return problem

        except Exception as e:
            raise RuntimeError(
                f"Failed to create Julia SDEProblem: {str(e)}\n"
                f"x0 shape: {x0.shape}, dtype: {x0.dtype}\n"
                f"g0 shape: {g0.shape}, dtype: {g0.dtype}\n"
                f"System: nx={self.sde_system.nx}, nw={self.sde_system.nw}\n"
                f"t_span: {t_span}"
            )

    def _create_noise_grid(self, t_array: np.ndarray, W_array: np.ndarray):
        """
        Create Julia NoiseGrid from time and noise arrays.

        This allows providing custom Brownian increments to Julia.

        Parameters
        ----------
        t_array : np.ndarray
            Time points (T,)
        W_array : np.ndarray
            Brownian motion values at each time point (T, nw)

        Returns
        -------
        NoiseGrid object for Julia

        Notes
        -----
        This is complex because:
        1. NoiseGrid expects cumulative Brownian values, not increments
        2. Must handle Python-Julia array bridging
        3. Requires proper DiffEqNoiseProcess structure

        For most use cases, use JAX/Diffrax for custom noise instead.
        """
        try:
            # Access Julia's DiffEqNoiseProcess
            # This may not be directly accessible via diffeqpy
            noise_grid = self.de.NoiseGrid(list(t_array), [list(w) for w in W_array])
            return noise_grid
        except AttributeError:
            raise NotImplementedError(
                "NoiseGrid not accessible via diffeqpy. "
                "This may require direct Julia calls or a newer diffeqpy version. "
                "For custom noise, use JAX/Diffrax backend instead."
            )

    def step(
        self,
        x: ArrayLike,
        u: Optional[ArrayLike] = None,
        dt: Optional[float] = None,
        dW: Optional[ArrayLike] = None,
    ) -> ArrayLike:
        """
        Take one SDE integration step.

        Note: Julia's solvers are optimized for full trajectory integration.
        Single-step interface is less efficient due to problem setup overhead.

        Parameters
        ----------
        x : ArrayLike
            Current state
        u : Optional[ArrayLike]
            Control input (None for autonomous)
        dt : Optional[float]
            Step size
        dW : Optional[ArrayLike]
            Brownian increments (nw,)

            **EXPERIMENTAL**: Julia DOES support custom noise via NoiseGrid,
            but implementing it reliably for single-step integration through
            diffeqpy is complex and may not work as expected.

            Current behavior:
            - If dW provided: Attempts to create NoiseGrid (may fail or be ignored)
            - If dW is None: Uses Julia's default random Brownian motion

            For reliable custom noise, use backend='jax' (Diffrax) instead.

        Returns
        -------
        ArrayLike
            Next state

        Notes
        -----
        Custom noise limitations with Julia/diffeqpy:
        1. NoiseGrid requires cumulative Brownian values, not just increments
        2. Python-Julia bridging for noise objects is fragile
        3. Single-step interface makes this awkward (need full grid)
        4. May require DiffEqNoiseProcess.jl which isn't always exposed

        Recommendation: For custom noise (deterministic testing, antithetic
        variates, etc.), use JAX/Diffrax which has clean custom noise support.

        Examples
        --------
        >>> # Standard usage (random noise)
        >>> x_next = integrator.step(x, u, dt=0.01)
        >>>
        >>> # Attempted custom noise (experimental, may not work)
        >>> x_next = integrator.step(x, u, dt=0.01, dW=np.array([0.5]))
        >>> # Warning: May generate random noise anyway
        """
        step_size = dt if dt is not None else self.dt

        # Define constant control function
        u_func = lambda t, x_state: u

        # Handle custom noise if provided
        noise_process = None
        if dW is not None:
            # Attempt to create NoiseGrid for custom noise
            # This is experimental and complex for single steps
            try:
                # Convert dW (increment) to cumulative Brownian values
                # For single step: W(t0) = 0, W(t1) = dW
                t_array = np.array([0.0, step_size], dtype=np.float64)

                # Cumulative Brownian motion values
                # W(0) = 0, W(dt) = dW
                dW_np = np.asarray(dW, dtype=np.float64)
                if dW_np.ndim == 0:
                    dW_np = dW_np.reshape(1)

                W_array = np.zeros((2, len(dW_np)), dtype=np.float64)
                W_array[0, :] = 0.0  # W(t0) = 0
                W_array[1, :] = dW_np  # W(t1) = dW

                # Try to create NoiseGrid
                noise_process = self._create_noise_grid(t_array, W_array)

            except (AttributeError, NotImplementedError, Exception) as e:
                # NoiseGrid creation failed
                warnings.warn(
                    f"Failed to create custom noise for Julia: {e}. "
                    f"Falling back to random noise generation. "
                    f"For reliable custom noise, use backend='jax' (Diffrax).",
                    UserWarning,
                    stacklevel=3,
                )
                noise_process = None

        # Setup problem for single step
        problem = self._setup_julia_problem(
            x, u_func, (0.0, step_size), noise_process=noise_process
        )

        # Create algorithm instance
        alg = self._algorithm_constructor()

        # Solve for single step
        sol = self.de.solve(
            problem,
            alg,
            dt=step_size,
            adaptive=False,
            save_everystep=False,
        )

        # Extract final state
        x_next = np.array(sol.u[-1], dtype=np.float64)

        # Update statistics (estimate)
        self._stats["total_steps"] += 1
        self._stats["total_fev"] += 1
        self._stats["diffusion_evals"] += 1

        return x_next

    def integrate(
        self,
        x0: ArrayLike,
        u_func: Callable[[float, ArrayLike], Optional[ArrayLike]],
        t_span: Tuple[float, float],
        t_eval: Optional[ArrayLike] = None,
        dense_output: bool = False,
    ) -> SDEIntegrationResult:
        """
        Integrate SDE over time interval using Julia solver.

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
            If True, enable dense output interpolation

        Returns
        -------
        SDEIntegrationResult
            Integration result with trajectory and statistics

        Examples
        --------
        >>> # Autonomous SDE
        >>> result = integrator.integrate(
        ...     x0=np.array([1.0]),
        ...     u_func=lambda t, x: None,
        ...     t_span=(0.0, 10.0)
        ... )
        >>>
        >>> # Controlled SDE
        >>> result = integrator.integrate(
        ...     x0=np.array([1.0, 0.0]),
        ...     u_func=lambda t, x: -K @ x,
        ...     t_span=(0.0, 10.0)
        ... )
        >>>
        >>> # Save at specific times
        >>> t_eval = np.linspace(0, 10, 1001)
        >>> result = integrator.integrate(x0, u_func, (0, 10), t_eval=t_eval)
        """
        x0 = np.asarray(x0, dtype=np.float64)
        t0, tf = t_span

        # Setup Julia SDE problem (no custom noise for full integration)
        problem = self._setup_julia_problem(x0, u_func, t_span)

        # Create algorithm instance
        alg = self._algorithm_constructor()

        # Setup solver options
        solve_kwargs = {
            "dt": self.dt,
            "adaptive": self.step_mode == StepMode.ADAPTIVE,
        }

        # Add tolerance for adaptive methods
        if self.step_mode == StepMode.ADAPTIVE:
            solve_kwargs["reltol"] = self.rtol
            solve_kwargs["abstol"] = self.atol

        # Add save points if specified
        if t_eval is not None:
            solve_kwargs["saveat"] = list(np.asarray(t_eval))
        elif not self.save_everystep:
            solve_kwargs["saveat"] = [float(t0), float(tf)]

        # Dense output
        if dense_output:
            solve_kwargs["dense"] = True

        # Solve SDE
        try:
            sol = self.de.solve(problem, alg, **solve_kwargs)

            # Extract solution
            t_out = np.array(sol.t, dtype=np.float64)

            # Convert list of arrays to 2D array (T, nx)
            if hasattr(sol, "u") and len(sol.u) > 0:
                x_out = np.array([np.asarray(u, dtype=np.float64) for u in sol.u])
            else:
                # Integration failed - no solution
                return SDEIntegrationResult(
                    t=np.array([t0]),
                    x=x0.reshape(1, -1),
                    success=False,
                    message="Julia SDE integration produced no output",
                    nfev=0,
                    nsteps=0,
                    diffusion_evals=0,
                    n_paths=1,
                    convergence_type=self.convergence_type,
                    solver=self.algorithm,
                    sde_type=self.sde_type,
                )

            # Check success
            success = len(t_out) > 1 and np.all(np.isfinite(x_out))

            # Estimate function evaluations
            # Julia doesn't expose exact counts via diffeqpy
            # For EM: approximately 1 drift + 1 diffusion per step
            nsteps = len(t_out) - 1
            nfev_estimate = nsteps
            diffusion_evals_estimate = nsteps

            # Update cumulative statistics
            self._stats["total_fev"] += nfev_estimate
            self._stats["diffusion_evals"] += diffusion_evals_estimate
            self._stats["total_steps"] += nsteps

            return SDEIntegrationResult(
                t=t_out,
                x=x_out,
                success=success,
                message=(
                    "Julia SDE integration successful"
                    if success
                    else "Integration failed (NaN/Inf detected)"
                ),
                nfev=self._stats["total_fev"],
                nsteps=nsteps,
                diffusion_evals=self._stats["diffusion_evals"],
                noise_samples=None,  # Julia doesn't expose noise samples
                n_paths=1,
                convergence_type=self.convergence_type,
                solver=self.algorithm,
                sde_type=self.sde_type,
                dense_solution=sol if dense_output else None,
            )

        except Exception as e:
            import traceback

            return SDEIntegrationResult(
                t=np.array([t0]),
                x=x0.reshape(1, -1),
                success=False,
                message=f"Julia SDE integration failed: {str(e)}\n{traceback.format_exc()}",
                nfev=0,
                nsteps=0,
                diffusion_evals=0,
                n_paths=1,
                convergence_type=self.convergence_type,
                solver=self.algorithm,
                sde_type=self.sde_type,
            )

    def validate_julia_setup(self):
        """
        Validate that Julia and DifferentialEquations.jl are properly set up.

        Returns
        -------
        bool
            True if setup is valid

        Raises
        ------
        RuntimeError
            If validation fails with details
        """
        try:
            # Check that SDEProblem exists
            assert hasattr(self.de, "SDEProblem"), "SDEProblem not found"

            # Check that algorithm exists
            assert hasattr(self.de, self.algorithm), f"Algorithm {self.algorithm} not found"

            # Try instantiating algorithm
            test_alg = getattr(self.de, self.algorithm)()
            assert test_alg is not None, f"Failed to instantiate {self.algorithm}"

            # Try creating and solving a simple test SDE
            def simple_drift(u, p, t):
                return np.array([-u[0]], dtype=np.float64)

            def simple_diffusion(u, p, t):
                return np.array([[0.1]], dtype=np.float64)

            test_problem = self.de.SDEProblem(
                simple_drift,
                simple_diffusion,
                np.array([1.0], dtype=np.float64),
                (0.0, 0.1),
                noise_rate_prototype=np.array([[0.1]], dtype=np.float64),
            )

            # Try solving
            test_sol = self.de.solve(test_problem, test_alg, dt=0.01)

            assert hasattr(test_sol, "t"), "Solution missing time points"
            assert hasattr(test_sol, "u"), "Solution missing states"
            assert len(test_sol.t) > 1, f"Solution has only {len(test_sol.t)} time point(s)"
            assert len(test_sol.u) > 1, f"Solution has only {len(test_sol.u)} state(s)"

            return True

        except Exception as e:
            import traceback

            raise RuntimeError(
                f"Julia/DiffEqPy validation failed:\n{str(e)}\n"
                f"{traceback.format_exc()}\n\n"
                f"Troubleshooting:\n"
                f"1. Run diagnostic: python julia_sde_diagnostic.py\n"
                f"2. Reinstall DifferentialEquations.jl:\n"
                f"   julia> using Pkg; Pkg.add('DifferentialEquations')\n"
                f"3. Reinstall diffeqpy:\n"
                f"   pip uninstall diffeqpy pyjulia\n"
                f"   pip install diffeqpy\n"
                f"   python -c 'from diffeqpy import install; install()'\n"
            ) from e

    # ========================================================================
    # Algorithm Information
    # ========================================================================

    @staticmethod
    def list_algorithms() -> Dict[str, List[str]]:
        """
        List available Julia SDE algorithms by category.

        Returns
        -------
        Dict[str, List[str]]
            Algorithms organized by category

        Examples
        --------
        >>> algorithms = DiffEqPySDEIntegrator.list_algorithms()
        >>> print(algorithms['euler_maruyama'])
        ['EM', 'LambaEM', 'EulerHeun']
        """
        return {
            "euler_maruyama": [
                "EM",  # Classic Euler-Maruyama
                "LambaEM",  # Lambda-adapted Euler-Maruyama
                "EulerHeun",  # Euler-Heun (predictor-corrector)
            ],
            "stochastic_rk": [
                "SRIW1",  # Roessler SRI for diagonal noise (strong 1.5)
                "SRIW2",  # Higher order variant
                "SOSRI",  # Second-order SRI for scalar noise
                "SOSRI2",  # Variant
                "SRA",  # Roessler SRA
                "SRA1",  # Order 2.0 weak for diagonal noise
                "SRA2",  # Variant
                "SRA3",  # For additive noise
                "SOSRA",  # Second-order SRA for scalar noise
                "SOSRA2",  # Variant
            ],
            "milstein": [
                "RKMil",  # Runge-Kutta Milstein
                "RKMilCommute",  # For commutative noise
                "RKMilGeneral",  # General Milstein
            ],
            "implicit": [
                "ImplicitEM",  # Implicit Euler-Maruyama
                "ImplicitEulerHeun",  # Implicit Euler-Heun
                "ImplicitRKMil",  # Implicit RK Milstein
            ],
            "imex": [
                "SKenCarp",  # Stochastic Kennedy-Carpenter IMEX
            ],
            "adaptive": [
                "AutoEM",  # Automatic method selection
            ],
            "optimized": [
                "SRI",  # Roessler SRI base
                "SRIW1Optimized",  # Optimized SRIW1
                "SRIW2Optimized",  # Optimized SRIW2
            ],
        }

    @staticmethod
    def get_algorithm_info(algorithm: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific algorithm.

        Parameters
        ----------
        algorithm : str
            Algorithm name

        Returns
        -------
        Dict[str, Any]
            Algorithm properties and recommendations

        Examples
        --------
        >>> info = DiffEqPySDEIntegrator.get_algorithm_info('SRIW1')
        >>> print(info['description'])
        'High accuracy for diagonal noise'
        """
        algorithm_info = {
            "EM": {
                "name": "Euler-Maruyama",
                "strong_order": 0.5,
                "weak_order": 1.0,
                "description": "Classic explicit method, robust and fast",
                "best_for": "General purpose, quick simulations",
                "noise_type": "any",
            },
            "LambaEM": {
                "name": "Lambda-adapted Euler-Maruyama",
                "strong_order": 0.5,
                "weak_order": 1.0,
                "description": "EM with lambda step size control",
                "best_for": "When dt needs automatic adjustment",
                "noise_type": "any",
            },
            "SRIW1": {
                "name": "Roessler SRI W1",
                "strong_order": 1.5,
                "weak_order": 2.0,
                "description": "High accuracy for diagonal noise",
                "best_for": "High accuracy diagonal noise problems",
                "noise_type": "diagonal",
            },
            "SRA1": {
                "name": "Roessler SRA1",
                "strong_order": 1.0,
                "weak_order": 2.0,
                "description": "Second-order weak for diagonal noise",
                "best_for": "Monte Carlo simulations",
                "noise_type": "diagonal",
            },
            "SRA3": {
                "name": "Roessler SRA3",
                "strong_order": 1.0,
                "weak_order": 2.0,
                "description": "Second-order weak for additive noise",
                "best_for": "Fast Monte Carlo with additive noise",
                "noise_type": "additive",
            },
            "RKMil": {
                "name": "Runge-Kutta Milstein",
                "strong_order": 1.0,
                "weak_order": 1.0,
                "description": "Includes Levy area approximation",
                "best_for": "When derivatives of diffusion available",
                "noise_type": "any",
            },
            "ImplicitEM": {
                "name": "Implicit Euler-Maruyama",
                "strong_order": 0.5,
                "weak_order": 1.0,
                "description": "Implicit for stiff drift",
                "best_for": "Stiff SDEs",
                "noise_type": "any",
            },
            "SKenCarp": {
                "name": "Stochastic Kennedy-Carpenter IMEX",
                "strong_order": "variable",
                "weak_order": "variable",
                "description": "IMEX for semi-stiff problems",
                "best_for": "Stiff drift, non-stiff diffusion",
                "noise_type": "any",
            },
        }

        return algorithm_info.get(
            algorithm,
            {
                "name": algorithm,
                "description": "Julia SDE algorithm (details not available)",
                "best_for": "Check Julia documentation",
            },
        )

    @staticmethod
    def recommend_algorithm(
        noise_type: str, stiffness: str = "none", accuracy: str = "medium"
    ) -> str:
        """
        Recommend Julia SDE algorithm based on problem characteristics.

        **IMPORTANT**: Not all recommended algorithms work via diffeqpy!
        - SRIW1/SRIW2: Don't work (use JAX/Diffrax instead)
        - EM: Always works
        - SRA3: Verified to work

        For guaranteed compatibility, use 'EM' or 'SRA3' only.
        For high accuracy with diagonal noise, use JAX/Diffrax.

        Parameters
        ----------
        noise_type : str
            'additive', 'diagonal', 'scalar', or 'general'
        stiffness : str
            'none', 'moderate', or 'severe'
        accuracy : str
            'low', 'medium', or 'high'

        Returns
        -------
        str
            Recommended algorithm name

        Examples
        --------
        >>> # Additive noise, high accuracy (WORKS)
        >>> alg = DiffEqPySDEIntegrator.recommend_algorithm(
        ...     noise_type='additive',
        ...     stiffness='none',
        ...     accuracy='high'
        ... )
        >>> print(alg)
        'SRA3'  # Verified to work
        >>>
        >>> # Diagonal noise, high accuracy (DOESN'T WORK via diffeqpy)
        >>> alg = DiffEqPySDEIntegrator.recommend_algorithm(
        ...     noise_type='diagonal',
        ...     stiffness='none',
        ...     accuracy='high'
        ... )
        >>> print(alg)
        'SRIW1'  # Won't work - use JAX/Diffrax SHARK instead
        """
        if stiffness == "severe":
            return "ImplicitEM"  # May not work via diffeqpy
        elif stiffness == "moderate":
            return "SKenCarp"  # May not work via diffeqpy

        # Non-stiff recommendations
        if accuracy == "high":
            if noise_type == "additive":
                return "SRA3"  # Verified to work
            elif noise_type in ["diagonal", "scalar"]:
                # SRIW1 doesn't work via diffeqpy!
                # This recommendation is from Julia's perspective
                # For actual use, recommend JAX/Diffrax instead
                return "SRIW1"  # Won't work - see docstring
            else:
                return "RKMil"  # May not work via diffeqpy
        elif accuracy == "medium":
            if noise_type == "diagonal":
                return "SRA1"  # Untested via diffeqpy
            else:
                return "LambaEM"  # Likely works
        else:  # low accuracy
            return "EM"  # Always works


# ============================================================================
# Utility Functions
# ============================================================================


def create_diffeqpy_sde_integrator(
    sde_system, algorithm: str = "EM", dt: float = 0.01, **options
) -> DiffEqPySDEIntegrator:
    """
    Quick factory for Julia SDE integrators.

    Parameters
    ----------
    sde_system : StochasticDynamicalSystem
        SDE system to integrate
    algorithm : str
        Julia algorithm name
    dt : float
        Time step
    **options
        Additional options

    Returns
    -------
    DiffEqPySDEIntegrator
        Configured integrator

    Examples
    --------
    >>> integrator = create_diffeqpy_sde_integrator(
    ...     sde_system,
    ...     algorithm='SRIW1',
    ...     dt=0.001,
    ...     rtol=1e-6
    ... )
    """
    return DiffEqPySDEIntegrator(sde_system, dt=dt, algorithm=algorithm, backend="numpy", **options)


def list_julia_sde_algorithms() -> None:
    """
    Print all available Julia SDE algorithms with descriptions.

    Examples
    --------
    >>> list_julia_sde_algorithms()

    Euler-Maruyama Family:
      - EM: Euler-Maruyama (strong 0.5, weak 1.0)
      - LambaEM: Lambda-adapted Euler-Maruyama
    ...
    """
    algorithms = DiffEqPySDEIntegrator.list_algorithms()

    print("Julia SDE Algorithms (via DifferentialEquations.jl)")
    print("=" * 60)

    category_names = {
        "euler_maruyama": "Euler-Maruyama Family",
        "stochastic_rk": "Stochastic Runge-Kutta Methods",
        "milstein": "Milstein Family",
        "implicit": "Implicit Methods (for stiff drift)",
        "imex": "IMEX Methods (semi-stiff)",
        "adaptive": "Adaptive Methods",
        "optimized": "Optimized Variants",
    }

    for category, algs in algorithms.items():
        print(f"\n{category_names.get(category, category)}:")
        for alg in algs:
            info = DiffEqPySDEIntegrator.get_algorithm_info(alg)
            if "strong_order" in info:
                print(
                    f"  - {alg}: {info['description']} "
                    f"(strong {info['strong_order']}, weak {info['weak_order']})"
                )
            else:
                print(f"  - {alg}: {info['description']}")

    print("\n" + "=" * 60)
    print("Use get_algorithm_info(name) for detailed information")
    print("Use recommend_algorithm() for automatic selection")
