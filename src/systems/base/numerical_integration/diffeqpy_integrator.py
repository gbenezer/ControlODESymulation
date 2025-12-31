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
DiffEqPyIntegrator: Julia DifferentialEquations.jl ODE solver via diffeqpy.

Provides access to Julia's DifferentialEquations.jl ecosystem - the most
comprehensive and performant ODE solver suite available in any language.

Supports both controlled and autonomous systems (nu=0).

Features:
- 100+ ODE solvers (explicit, implicit, IMEX, stabilized, geometric)
- Automatic stiffness detection
- Event handling and callbacks
- Dense output and interpolation
- Exceptional performance for difficult ODEs

Requirements:
    Julia must be installed with DifferentialEquations.jl:
        julia> using Pkg
        julia> Pkg.add("DifferentialEquations")

    Python package:
        $ pip install diffeqpy

Notes
-----
This integrator only supports NumPy backend because Julia arrays are
converted to/from NumPy. For gradient-based workflows:
- Use TorchDiffEqIntegrator for PyTorch with autograd
- Use DiffraxIntegrator for JAX with JIT and autograd

DiffEqPyIntegrator is ideal for:
- Production simulations requiring highest reliability
- Very stiff or difficult ODEs where scipy struggles
- Problems where Julia's specialized solvers excel
- Non-gradient workflows where accuracy/performance matter most

DiffEqPyIntegrator: Julia DifferentialEquations.jl ODE solver via diffeqpy.


Known Limitations:
------------------
**Jacobian Autodiff Failures with Python ODEs**

Many Julia implicit/Rosenbrock methods fail when used with Python-defined ODE
functions due to autodiff errors in the Julia-Python bridge. The error message
is typically:

    "First call to automatic differentiation for the Jacobian"

**Methods Known to Fail:**
- Rosenbrock family: Rosenbrock23, Rosenbrock32, Rodas4, Rodas4P, Rodas5
- Implicit RK: RadauIIA5
- ESDIRK: TRBDF2, KenCarp3, KenCarp4, KenCarp5

**Methods That Work Reliably:**
- Non-stiff explicit: Tsit5, Vern6-9, DP5, DP8
- Stabilized explicit: ROCK2, ROCK4 (handle moderate stiffness)
- Auto-switching: May fail if it switches to a problematic method
- Geometric: SymplecticEuler, VelocityVerlet

**Recommended Workarounds for Stiff Systems:**
1. **Use scipy instead**: `scipy.BDF` or `scipy.Radau` work excellently for
   stiff systems in Python and don't have this limitation.

2. **Use ROCK methods**: For moderately stiff problems, ROCK2/ROCK4 are
   stabilized explicit methods that work with Python ODEs.

3. **Pure Julia**: If you need Julia's Rosenbrock methods, write your ODE
   function directly in Julia (not via Python/diffeqpy).

4. **Analytical Jacobian**: Provide the Jacobian analytically (advanced,
   not currently supported in this framework).

This is a fundamental limitation of the diffeqpy bridge, not a bug.

Examples
--------
>>> # High-accuracy non-stiff solver
>>> integrator = DiffEqPyIntegrator(
...     system,
...     backend='numpy',
...     algorithm='Vern9',
...     reltol=1e-12,
...     abstol=1e-14
... )
>>>
>>> # Stiff system solver
>>> integrator = DiffEqPyIntegrator(
...     system,
...     algorithm='Rosenbrock23'
... )
>>>
>>> # Auto-switching between non-stiff and stiff
>>> integrator = DiffEqPyIntegrator(
...     system,
...     algorithm='AutoTsit5(Rosenbrock23())'
... )
>>>
>>> # Autonomous system
>>> integrator = DiffEqPyIntegrator(autonomous_system, algorithm='Tsit5')
>>> result = integrator.integrate(
...     x0=np.array([1.0, 0.0]),
...     u_func=lambda t, x: None,
...     t_span=(0.0, 10.0)
... )
"""

import os
import re
import time
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from src.systems.base.numerical_integration.integrator_base import (
    IntegratorBase,
    StepMode,
)

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

from src.types.backends import Backend

if TYPE_CHECKING:
    from src.systems.base.core.continuous_system_base import ContinuousSystemBase


# Debug flag - set via environment variable
DEBUG_DIFFEQPY = os.environ.get("DEBUG_DIFFEQPY", "0") == "1"


class DiffEqPyIntegrator(IntegratorBase):
    """
    ODE integrator using Julia's DifferentialEquations.jl via diffeqpy.

    Provides access to 100+ ODE solvers from the Julia ecosystem, including:

    **Non-Stiff Solvers (Explicit Runge-Kutta):**
    - Tsit5: Tsitouras 5(4) - recommended default
    - Vern6, Vern7, Vern8, Vern9: Verner methods (high order)
    - DP5, DP8: Dormand-Prince
    - TanYam7, TsitPap8: High-order methods

    **Stiff Solvers (Implicit/Rosenbrock):**
    - Rosenbrock23, Rosenbrock32: Rosenbrock methods (recommended)
    - Rodas4, Rodas4P, Rodas5: High-accuracy Rosenbrock
    - TRBDF2, KenCarp3-5: ESDIRK methods
    - RadauIIA5: Implicit Runge-Kutta
    - QNDF, FBDF: BDF variants

    **Auto-Switching:**
    - AutoTsit5(Rosenbrock23()): Auto-detects stiffness
    - AutoVern7(Rodas5()): High-accuracy auto-switching

    **Stabilized (for moderately stiff):**
    - ROCK2, ROCK4: Stabilized explicit
    - ESERK4, ESERK5: Stabilized ERK

    **Geometric (structure-preserving):**
    - SymplecticEuler, VelocityVerlet: Symplectic integrators
    - McAte2-5: McClellan-Aitken methods

    Parameters
    ----------
    system : SymbolicDynamicalSystem
        Continuous-time system to integrate (controlled or autonomous)
    dt : Optional[float]
        Time step (initial guess for adaptive, fixed for ConstantStepSize)
    step_mode : StepMode
        FIXED or ADAPTIVE stepping mode
    backend : str
        Must be 'numpy' (Julia arrays convert to NumPy)
    algorithm : str, optional
        Julia solver algorithm. Default: 'Tsit5'
        See list_algorithms() for available options
    save_everystep : bool, optional
        Save every integration step (default: False for adaptive)
    dense : bool, optional
        Enable dense output for interpolation (default: False)
    callback : Optional[Callable]
        Julia callback for events, termination, etc.
    **options
        Additional solver options:
        - rtol: Relative tolerance (default: 1e-6)
        - atol: Absolute tolerance (default: 1e-8)
        - maxiters: Maximum iterations (default: 1e7)
        - dtmin: Minimum step size (adaptive only)
        - dtmax: Maximum step size (adaptive only)

    Attributes
    ----------
    de : module
        Julia DifferentialEquations module
    algorithm : str
        Algorithm name
    save_everystep : bool
        Whether to save all steps
    dense : bool
        Whether dense output is enabled

    Examples
    --------
    >>> # Controlled system - default high-quality solver
    >>> integrator = DiffEqPyIntegrator(system, backend='numpy')
    >>> result = integrator.integrate(x0, u_func, (0, 10))
    >>>
    >>> # Autonomous system
    >>> integrator = DiffEqPyIntegrator(autonomous_system)
    >>> result = integrator.integrate(
    ...     x0=np.array([1.0, 0.0]),
    ...     u_func=lambda t, x: None,
    ...     t_span=(0.0, 10.0)
    ... )
    >>>
    >>> # Very high accuracy
    >>> integrator = DiffEqPyIntegrator(
    ...     system,
    ...     algorithm='Vern9',
    ...     rtol=1e-12,
    ...     atol=1e-14
    ... )
    >>>
    >>> # Stiff system
    >>> integrator = DiffEqPyIntegrator(
    ...     system,
    ...     algorithm='Rodas5',
    ...     rtol=1e-8
    ... )
    >>>
    >>> # Fixed-step integration
    >>> integrator = DiffEqPyIntegrator(
    ...     system,
    ...     dt=0.01,
    ...     step_mode=StepMode.FIXED,
    ...     algorithm='Tsit5'
    ... )

    Notes
    -----
    - Julia must be installed with DifferentialEquations.jl
    - First call may be slow due to Julia JIT compilation
    - Subsequent calls are very fast (Julia's strength)
    - For difficult ODEs, this is often the best choice
    - Set DEBUG_DIFFEQPY=1 environment variable for debug output
    - Supports autonomous systems (nu=0) by passing u_func that returns None
    """

    def __init__(
        self,
        system: "ContinuousSystemBase",
        dt: Optional[ScalarLike] = None,
        step_mode: StepMode = StepMode.ADAPTIVE,
        backend: Backend = "numpy",
        algorithm: str = "Tsit5",
        save_everystep: bool = False,
        dense: bool = False,
        callback: Optional[Any] = None,
        **options,
    ):
        """
        Initialize DiffEqPy integrator.

        Parameters
        ----------
        system : SymbolicDynamicalSystem
            System to integrate (controlled or autonomous)
        dt : Optional[float]
            Time step (initial for adaptive, fixed for FIXED mode)
        step_mode : StepMode
            FIXED or ADAPTIVE
        backend : str
            Must be 'numpy'
        algorithm : str
            Julia algorithm name (default: 'Tsit5')
        save_everystep : bool
            Save every step (useful for visualization)
        dense : bool
            Enable dense output for interpolation
        callback : Optional
            Julia callback function
        **options
            Solver options (rtol, atol, etc.)
        """
        if backend != "numpy":
            raise ValueError(
                f"DiffEqPyIntegrator requires backend='numpy', got '{backend}'. "
                f"Julia arrays are automatically converted to NumPy."
            )

        super().__init__(system, dt, step_mode, backend, **options)

        self.algorithm = algorithm
        self.save_everystep = save_everystep
        self.dense = dense
        self.callback = callback
        self._integrator_name = f"DiffEqPy-{algorithm}"

        # Try to import diffeqpy
        try:
            from diffeqpy import de

            self.de = de
        except ImportError:
            raise ImportError(
                "diffeqpy is required for DiffEqPyIntegrator.\n\n"
                "Installation steps:\n"
                "1. Install Julia from https://julialang.org/downloads/\n"
                "2. Install DifferentialEquations.jl:\n"
                "   julia> using Pkg\n"
                "   julia> Pkg.add('DifferentialEquations')\n"
                "3. Install Python package:\n"
                "   pip install diffeqpy\n"
                "4. Run Julia setup from Python:\n"
                "   python -c 'from diffeqpy import install; install()'"
            )

        # Validate algorithm
        self._validate_algorithm()

    def _validate_algorithm(self):
        """
        Validate that the algorithm is supported.

        Raises
        ------
        ValueError
            If algorithm is not recognized
        """
        # Get all known algorithms from list_algorithms()
        all_algorithms = list_algorithms()
        known_algorithms = set()
        for category, algos in all_algorithms.items():
            known_algorithms.update(algos)

        # Check if algorithm is in known list
        if self.algorithm in known_algorithms:
            # For simple algorithms, validate they exist in Julia
            if "(" not in self.algorithm:
                try:
                    self._get_algorithm()
                except Exception as e:
                    raise ValueError(
                        f"Algorithm '{self.algorithm}' is listed as available but "
                        f"could not be loaded from Julia. Error: {e}"
                    )
        else:
            # Unknown algorithm - give helpful error
            raise ValueError(
                f"Unknown algorithm '{self.algorithm}'. "
                f"Use list_algorithms() to see supported options.\n\n"
                f"Supported categories:\n"
                f"  - nonstiff: {', '.join(all_algorithms['nonstiff'][:5])}...\n"
                f"  - stiff_rosenbrock: {', '.join(all_algorithms['stiff_rosenbrock'])}\n"
                f"  - auto_switching: {', '.join(all_algorithms['auto_switching'])}\n"
                f"  (Run list_algorithms() for complete list)"
            )

    @property
    def name(self) -> str:
        mode_str = "Fixed" if self.step_mode == StepMode.FIXED else "Adaptive"

        # USE DYNAMIC LOOKUP:
        all_algos = list_algorithms()

        if "Auto" in self.algorithm:
            type_str = " (Auto-Stiffness)"
        elif any(
            self.algorithm in all_algos.get(cat, [])
            for cat in ["stiff_rosenbrock", "stiff_esdirk", "stiff_implicit"]
        ):
            type_str = " (Stiff)"
        elif self.algorithm in all_algos.get("stabilized", []):
            type_str = " (Stabilized)"
        elif self.algorithm in all_algos.get("geometric", []):
            type_str = " (Geometric)"
        else:
            type_str = ""

        return f"{self._integrator_name} ({mode_str}){type_str}"

    def _get_algorithm(self):
        """
        Get Julia algorithm object.

        Handles both simple algorithms and complex ones like:
        - AutoTsit5(Rosenbrock23())
        - Composite algorithm specifications

        Returns
        -------
        Julia algorithm object
        """
        algo_str = self.algorithm

        # Handle auto-switching algorithms
        if "Auto" in algo_str or "(" in algo_str:
            try:
                # Evaluate as Julia expression using proper scoping
                # e.g., "AutoTsit5(Rosenbrock23())" -> self.de.AutoTsit5(self.de.Rosenbrock23())

                # Replace algorithm names with self.de.AlgorithmName
                def replace_algo(match):
                    algo_name = match.group(0)
                    # Don't replace 'self' or 'de'
                    if algo_name in ["self", "de"]:
                        return algo_name
                    return f"self.de.{algo_name}"

                # Find algorithm names (capital letter followed by letters/numbers)
                pattern = r"\b([A-Z][a-zA-Z0-9]*)\b"
                modified_str = re.sub(pattern, replace_algo, algo_str)

                return eval(modified_str)
            except Exception as e:
                raise ValueError(
                    f"Failed to create algorithm '{algo_str}'. " f"Check syntax. Error: {e}"
                )
        else:
            # Simple algorithm name
            try:
                algo_class = getattr(self.de, algo_str)
                return algo_class()
            except AttributeError:
                raise ValueError(
                    f"Algorithm '{algo_str}' not found in DifferentialEquations.jl. "
                    f"Use list_algorithms() to see available options."
                )

    def step(
        self, x: StateVector, u: Optional[ControlVector] = None, dt: Optional[ScalarLike] = None
    ) -> StateVector:
        """
        Take one integration step.

        For efficiency, uses integrate() internally with single step.

        Parameters
        ----------
        x : ArrayLike
            Current state (nx,)
        u : Optional[ArrayLike]
            Control input (nu,), or None for autonomous systems (assumed constant over step)
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

        x = np.asarray(x)

        # Handle autonomous systems - keep None as None
        if u is not None:
            u = np.asarray(u)

        # Use integrate for single step
        u_func = lambda t, x_cur: u  # May be None for autonomous

        result = self.integrate(
            x0=x, u_func=u_func, t_span=(0.0, step_size), t_eval=np.array([0.0, step_size])
        )

        return result["x"][-1]

    def integrate(
        self,
        x0: StateVector,
        u_func: Callable[[ScalarLike, StateVector], Optional[ControlVector]],
        t_span: TimeSpan,
        t_eval: Optional[TimePoints] = None,
        dense_output: bool = False,
    ) -> IntegrationResult:
        """
        Integrate ODE using Julia's DifferentialEquations.jl.

        Parameters
        ----------
        x0 : ArrayLike
            Initial state (nx,)
        u_func : Callable[[float, ArrayLike], Optional[ArrayLike]]
            Control policy (t, x) → u (or None for autonomous systems)
        t_span : Tuple[float, float]
            Time interval (t_start, t_end)
        t_eval : Optional[ArrayLike]
            Specific times at which to store solution
            If None:
            - FIXED mode: Uses uniform grid with dt
            - ADAPTIVE mode: Julia chooses points adaptively
        dense_output : bool
            Enable dense output for interpolation

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
            - sol: Julia solution object (if dense_output=True)
            - dense_output: True if dense output enabled

        Examples
        --------
        >>> # Controlled system - adaptive integration
        >>> result = integrator.integrate(
        ...     x0=np.array([1.0, 0.0]),
        ...     u_func=lambda t, x: -K @ x,
        ...     t_span=(0.0, 10.0)
        ... )
        >>>
        >>> # Autonomous system
        >>> result = integrator.integrate(
        ...     x0=np.array([1.0, 0.0]),
        ...     u_func=lambda t, x: None,
        ...     t_span=(0.0, 10.0)
        ... )
        >>>
        >>> # Fixed-step with specific evaluation times
        >>> t_eval = np.linspace(0, 10, 1001)
        >>> result = integrator.integrate(
        ...     x0, u_func, (0, 10),
        ...     t_eval=t_eval
        ... )
        >>>
        >>> # Dense output for interpolation
        >>> result = integrator.integrate(
        ...     x0, u_func, (0, 10),
        ...     dense_output=True
        ... )
        >>> # Interpolate at arbitrary time
        >>> x_at_5_5 = result["sol"](5.5)
        """
        start_time = time.time()

        t0, tf = t_span
        x0 = np.asarray(x0, dtype=np.float64)

        # Handle edge case
        if t0 == tf:
            result: IntegrationResult = {
                "t": np.array([t0]),
                "x": x0[None, :] if x0.ndim == 1 else x0,
                "success": True,
                "message": "Zero time span",
                "nfev": 0,
                "nsteps": 0,
                "integration_time": 0.0,
                "solver": self.name,
            }
            return result

        # Track function evaluations for this integration
        fev_count = [0]

        # Define ODE function for Julia using IN-PLACE formulation
        # Julia signature for in-place: f!(du, u, p, t) - modifies du in place
        def ode_func_inplace(du, u_val, p, t):
            """
            ODE function in Julia's IN-PLACE signature.

            Uses in-place formulation to avoid type stability issues.
            Modifies du in-place rather than returning a value.

            Parameters
            ----------
            du : array
                Output array to fill with derivatives (Julia array)
            u_val : array
                Current state (Julia array)
            p : any
                Parameters (unused, required by Julia)
            t : float
                Current time
            """
            # Convert from Julia arrays to NumPy
            x_np = np.asarray(u_val, dtype=np.float64)

            # Evaluate control policy
            u_control = u_func(float(t), x_np)

            # Handle autonomous systems - keep None as None
            # Do NOT convert None to array, as np.asarray(None) creates array(None, dtype=object)
            if u_control is not None:
                u_np = np.asarray(u_control, dtype=np.float64)
            else:
                u_np = None  # Keep as None for autonomous systems

            # Evaluate system dynamics (DynamicsEvaluator handles u_np=None correctly)
            dx = self.system(x_np, u_np, backend="numpy")

            # Track function evaluations
            fev_count[0] += 1

            # Convert to float64 array
            dx_array = np.asarray(dx, dtype=np.float64).flatten()

            # Fill du in-place (Julia's in-place convention)
            for i in range(len(dx_array)):
                du[i] = dx_array[i]

        # Prepare time span
        tspan = (float(t0), float(tf))

        # Prepare save points (saveat)
        if t_eval is not None:
            # User specified evaluation times
            saveat = list(np.asarray(t_eval))
        elif self.step_mode == StepMode.FIXED:
            # Fixed-step mode: create uniform grid
            if self.dt is None:
                raise ValueError("dt required for FIXED step mode")
            n_steps = int(np.ceil((tf - t0) / self.dt)) + 1
            saveat = list(np.linspace(t0, tf, n_steps))
        else:
            # Adaptive mode with no t_eval: let Julia choose
            saveat = []

        # Build solver options
        solve_kwargs = {
            "reltol": self.rtol,  # Julia uses 'reltol'
            "abstol": self.atol,  # Julia uses 'abstol'
            "maxiters": int(self.options.get("maxiters", 1e7)),
        }

        # Add saveat if specified
        if saveat:
            solve_kwargs["saveat"] = saveat

        # Control step size behavior
        if self.step_mode == StepMode.FIXED and self.dt is not None:
            # Fixed step size
            solve_kwargs["dt"] = self.dt
            solve_kwargs["adaptive"] = False
        else:
            # Adaptive stepping
            solve_kwargs["save_everystep"] = self.save_everystep
            if self.options.get("dtmin") is not None:
                solve_kwargs["dtmin"] = self.options["dtmin"]
            if self.options.get("dtmax") is not None:
                solve_kwargs["dtmax"] = self.options["dtmax"]

        # Dense output
        solve_kwargs["dense"] = dense_output or self.dense

        # Callback (if provided)
        if self.callback is not None:
            solve_kwargs["callback"] = self.callback

        # Set up ODE problem using IN-PLACE formulation
        prob = self.de.ODEProblem(ode_func_inplace, x0, tspan)

        # Get algorithm
        algorithm = self._get_algorithm()

        # Solve ODE
        try:
            sol = self.de.solve(prob, algorithm, **solve_kwargs)

            if DEBUG_DIFFEQPY:
                print(f"\n[DEBUG] Julia retcode: {sol.retcode}")
                print(f"[DEBUG] retcode type: {type(sol.retcode)}")
                print(f"[DEBUG] retcode str: '{str(sol.retcode)}'")
                print(f"[DEBUG] retcode repr: {repr(sol.retcode)}")
                print(f"[DEBUG] sol.t length: {len(sol.t)}")
                print(f"[DEBUG] sol.u length: {len(sol.u)}")

            # Extract solution
            # Julia returns solution object with attributes:
            # - sol.t: time points (Vector)
            # - sol.u: state vectors (Vector of Vectors)
            t_out = np.array(sol.t)

            # Convert list of state vectors to 2D array
            # sol.u is a list of vectors, need to stack them
            x_out = np.array([np.array(x_i) for x_i in sol.u])

            # Determine success
            # Julia retcode is a Symbol that gets converted to string by diffeqpy
            # Successful integration returns 'Success' (without the colon prefix)
            # Failed integrations return other codes like 'DtLessThanMin', 'MaxIters', etc.
            retcode_str = str(sol.retcode)

            # Check for successful integration
            # Note: diffeqpy converts Julia Symbol :Success to Python string 'Success'
            success = retcode_str == "Success"

            if DEBUG_DIFFEQPY:
                print(f"[DEBUG] Success check: '{retcode_str}' == 'Success' -> {success}")

            message = f"Integration {retcode_str}"

            # Update statistics
            nsteps = len(t_out) - 1
            nfev = fev_count[0]

            self._stats["total_steps"] += nsteps
            self._stats["total_fev"] += nfev

            elapsed = time.time() - start_time
            self._stats["total_time"] += elapsed

            # Create result dict with type annotation
            result: IntegrationResult = {
                "t": t_out,
                "x": x_out,
                "success": success,
                "message": message,
                "nfev": nfev,
                "nsteps": nsteps,
                "integration_time": elapsed,
                "solver": self.name,
            }
            
            # Add optional fields conditionally
            if dense_output or self.dense:
                result["sol"] = sol
                result["dense_output"] = True
            
            return result

        except Exception as e:
            elapsed = time.time() - start_time
            self._stats["total_time"] += elapsed

            if DEBUG_DIFFEQPY:
                print(f"\n[DEBUG] Integration exception: {type(e).__name__}")
                print(f"[DEBUG] Exception message: {str(e)}")
                import traceback

                traceback.print_exc()

            result: IntegrationResult = {
                "t": np.array([t0]),
                "x": x0[None, :] if x0.ndim == 1 else x0,
                "success": False,
                "message": f"Integration failed: {str(e)}",
                "nfev": fev_count[0],
                "nsteps": 0,
                "integration_time": elapsed,
                "solver": self.name,
            }
            return result

    def set_callback(self, callback):
        """
        Set Julia callback for events, termination, etc.

        Parameters
        ----------
        callback : Julia callback object
            See DifferentialEquations.jl documentation for callback types
        """
        self.callback = callback

    def __repr__(self) -> str:
        return (
            f"DiffEqPyIntegrator(algorithm='{self.algorithm}', "
            f"mode={self.step_mode.value}, "
            f"rtol={self.rtol:.1e}, atol={self.atol:.1e})"
        )


# ============================================================================
# Utility Functions
# ============================================================================


def list_algorithms() -> Dict[str, List[str]]:
    """
    List available Julia DifferentialEquations.jl algorithms.

    Returns
    -------
    Dict[str, List[str]]
        Dictionary mapping algorithm categories to lists of algorithm names

    Notes
    -----
     WARNING: Many stiff/implicit methods in 'stiff_rosenbrock',
    'stiff_esdirk', and 'stiff_implicit' categories FAIL when used with
    Python-defined ODE functions due to Jacobian autodiff limitations in
    the Julia-Python bridge.

    **Safe to use:** 'nonstiff', 'stabilized', 'geometric', 'low_order'
    **Problematic:** 'stiff_rosenbrock', 'stiff_esdirk', 'stiff_implicit'
    **Use scipy instead:** For stiff systems, use scipy.BDF or scipy.Radau

    Examples
    --------
    >>> algos = list_algorithms()
    >>> print(algos['nonstiff'])  # These work
    ['Tsit5', 'Vern7', 'Vern9', 'DP5', 'DP8', ...]
    >>>
    >>> print(algos['stiff_rosenbrock'])  # These fail with Python ODEs
    ['Rosenbrock23', 'Rosenbrock32', 'Rodas4', 'Rodas4P', 'Rodas5']
    >>>
    >>> print(algos['stabilized'])  # These work (moderate stiffness)
    ['ROCK2', 'ROCK4', 'ESERK4', 'ESERK5']
    """
    return {
        "nonstiff": [
            "Tsit5",  # Works - RECOMMENDED DEFAULT
            "Vern6",  # Works
            "Vern7",  # Works
            "Vern8",  # Works
            "Vern9",  # Works - very high accuracy
            "DP5",  # Works
            "DP8",  # Works
            "TanYam7",  # Works
            "TsitPap8",  # Works
        ],
        "stiff_rosenbrock": [
            # All fail with Python ODEs - use scipy.BDF instead
            "Rosenbrock23",  # FAILS: Jacobian autodiff error
            "Rosenbrock32",  # FAILS: Jacobian autodiff error
            "Rodas4",  # FAILS: Jacobian autodiff error
            "Rodas4P",  # FAILS: Jacobian autodiff error
            "Rodas5",  # FAILS: Jacobian autodiff error
        ],
        "stiff_esdirk": [
            # May fail with Python ODEs
            "TRBDF2",  # FAILS: Jacobian autodiff error
            "KenCarp3",  # FAILS: Jacobian autodiff error
            "KenCarp4",  # FAILS: Jacobian autodiff error
            "KenCarp5",  # FAILS: Jacobian autodiff error
        ],
        "stiff_implicit": [
            # May fail with Python ODEs
            "RadauIIA5",  # FAILS: Jacobian autodiff error
            "QNDF",  # Untested
            "FBDF",  # Untested
        ],
        "auto_switching": [
            #  May fail if switches to problematic method
            "AutoTsit5(Rosenbrock23())",  # May fail when switching to Rosenbrock
            "AutoVern7(Rodas5())",  # May fail when switching to Rodas
            "AutoVern8(Rodas5())",
            "AutoVern9(Rodas5())",
        ],
        "stabilized": [
            # These work - good for moderate stiffness
            "ROCK2",  # Works - stabilized explicit
            "ROCK4",  # Works - higher-order ROCK
            "ESERK4",  # Untested but should work
            "ESERK5",  # Untested but should work
        ],
        "geometric": [
            # These work - structure-preserving
            "SymplecticEuler",  # Works
            "VelocityVerlet",  # Works
            "VerletLeapfrog",  # Untested but should work
            "McAte2",  # Untested but should work
            "McAte4",  # Untested but should work
            "McAte5",  # Untested but should work
        ],
        "low_order": [
            # These work
            "Euler",  # Works (but use with small dt)
            "Midpoint",  # Works
            "Heun",  # Works
        ],
    }


def get_safe_algorithms() -> List[str]:
    """
    Get list of Julia algorithms that work reliably with Python ODEs.

    Returns algorithms that don't require Jacobian autodiff across the
    Julia-Python bridge.

    Returns
    -------
    List[str]
        Algorithm names that are safe to use

    Examples
    --------
    >>> safe = get_safe_algorithms()
    >>> print(safe[:5])
    ['Tsit5', 'Vern6', 'Vern7', 'Vern8', 'Vern9']
    """
    all_algos = list_algorithms()
    safe = []
    safe.extend(all_algos["nonstiff"])
    safe.extend(all_algos["stabilized"])
    safe.extend(all_algos["geometric"])
    safe.extend(all_algos["low_order"])
    return safe


def get_problematic_algorithms() -> List[str]:
    """
    Get list of Julia algorithms that fail with Python ODEs.

    These require Jacobian computation that fails across the Julia-Python bridge.

    Returns
    -------
    List[str]
        Algorithm names that are known to fail

    Examples
    --------
    >>> problematic = get_problematic_algorithms()
    >>> if 'Rosenbrock23' in problematic:
    ...     print("Use scipy.BDF instead")
    """
    all_algos = list_algorithms()
    problematic = []
    problematic.extend(all_algos["stiff_rosenbrock"])
    problematic.extend(all_algos["stiff_esdirk"])
    problematic.extend(all_algos["stiff_implicit"])
    return problematic


def print_algorithm_recommendations():
    """
    Print recommendations for algorithm selection.

    Examples
    --------
    >>> print_algorithm_recommendations()
    """
    print("=" * 70)
    print("Julia DifferentialEquations.jl Algorithm Recommendations")
    print("=" * 70)
    print()
    print("🎯 DEFAULT CHOICE:")
    print("  Tsit5")
    print("  - Excellent general-purpose solver")
    print("  - Good accuracy and performance")
    print("  - Works for most non-stiff problems")
    print()
    print("🔄 AUTO-SWITCHING (BEST FOR UNKNOWN STIFFNESS):")
    print("  AutoTsit5(Rosenbrock23())")
    print("  - Automatically detects stiffness")
    print("  - Switches between Tsit5 (non-stiff) and Rosenbrock23 (stiff)")
    print("  - Set-and-forget option")
    print()
    print("📈 HIGH ACCURACY (NON-STIFF):")
    print("  Vern9")
    print("  - 9th order accuracy")
    print("  - Best for very smooth problems requiring high precision")
    print("  - Orbital mechanics, celestial mechanics")
    print()
    print("🔥 STIFF SYSTEMS:")
    print("  Rosenbrock23 (RECOMMENDED)")
    print("  - Fast and reliable for moderately stiff problems")
    print("  Rodas5")
    print("  - High accuracy for very stiff problems")
    print("  - Chemical kinetics, circuit simulation")
    print()
    print("⚡ STABILIZED (MODERATELY STIFF):")
    print("  ROCK4")
    print("  - Good for problems between non-stiff and stiff")
    print("  - Larger stability region than explicit methods")
    print()
    print("🎨 GEOMETRIC (STRUCTURE-PRESERVING):")
    print("  VelocityVerlet")
    print("  - Symplectic integrator for Hamiltonian systems")
    print("  - Preserves energy in conservative systems")
    print("  - Molecular dynamics, celestial mechanics")
    print()
    print("=" * 70)


def create_diffeqpy_integrator(
    system: "ContinuousSystemBase",
    algorithm: str = "Tsit5",
    dt: Optional[float] = None,
    step_mode: StepMode = StepMode.ADAPTIVE,
    **options,
) -> DiffEqPyIntegrator:
    """
    Quick factory for DiffEqPy integrators.

    Parameters
    ----------
    system : SymbolicDynamicalSystem
        System to integrate (controlled or autonomous)
    algorithm : str
        Julia algorithm name (default: 'Tsit5')
    dt : Optional[float]
        Time step (initial for adaptive)
    step_mode : StepMode
        FIXED or ADAPTIVE
    **options
        Additional solver options (rtol, atol, etc.)

    Returns
    -------
    DiffEqPyIntegrator
        Configured integrator

    Examples
    --------
    >>> # Default high-quality solver
    >>> integrator = create_diffeqpy_integrator(system)
    >>>
    >>> # High-accuracy solver
    >>> integrator = create_diffeqpy_integrator(
    ...     system,
    ...     algorithm='Vern9',
    ...     rtol=1e-12,
    ...     atol=1e-14
    ... )
    >>>
    >>> # Auto-switching solver
    >>> integrator = create_diffeqpy_integrator(
    ...     system,
    ...     algorithm='AutoTsit5(Rosenbrock23())'
    ... )
    """
    return DiffEqPyIntegrator(
        system=system, dt=dt, step_mode=step_mode, backend="numpy", algorithm=algorithm, **options
    )


# ============================================================================
# Integration with Factory
# ============================================================================


def get_algorithm_info(algorithm: str) -> Dict[str, Any]:
    """
    Get information about a specific Julia algorithm.

    Parameters
    ----------
    algorithm : str
        Algorithm name

    Returns
    -------
    Dict[str, Any]
        Information about the algorithm

    Examples
    --------
    >>> info = get_algorithm_info('Tsit5')
    >>> print(info['description'])
    """
    algorithm_info = {
        "Tsit5": {
            "name": "Tsitouras 5(4)",
            "order": 5,
            "type": "Explicit Runge-Kutta",
            "description": "Excellent general-purpose solver with good efficiency",
            "best_for": "Most non-stiff problems",
            "stability": "Conditionally stable",
        },
        "Vern9": {
            "name": "Verner 9(8)",
            "order": 9,
            "type": "Explicit Runge-Kutta",
            "description": "Very high accuracy solver",
            "best_for": "High-precision requirements, smooth problems",
            "stability": "Conditionally stable",
        },
        "Rosenbrock23": {
            "name": "Rosenbrock 2/3",
            "order": 2,
            "type": "Rosenbrock (Stiff)",
            "description": "Efficient stiff solver with automatic Jacobian",
            "best_for": "Moderately stiff problems",
            "stability": "L-stable",
        },
        "Rodas5": {
            "name": "Rodas 5(4)",
            "order": 5,
            "type": "Rosenbrock (Stiff)",
            "description": "High-accuracy stiff solver",
            "best_for": "Very stiff problems requiring high accuracy",
            "stability": "L-stable",
        },
        "AutoTsit5(Rosenbrock23())": {
            "name": "Auto-switching Tsit5/Rosenbrock23",
            "order": "Variable (2-5)",
            "type": "Auto-switching",
            "description": "Automatically switches between non-stiff and stiff solvers",
            "best_for": "Unknown stiffness, set-and-forget",
            "stability": "Adaptive",
        },
        "ROCK4": {
            "name": "ROCK4",
            "order": 4,
            "type": "Stabilized Explicit",
            "description": "Stabilized explicit method for moderately stiff problems",
            "best_for": "Moderately stiff ODEs with large explicit part",
            "stability": "Extended stability",
        },
        "VelocityVerlet": {
            "name": "Velocity Verlet",
            "order": 2,
            "type": "Symplectic",
            "description": "Structure-preserving integrator for Hamiltonian systems",
            "best_for": "Conservative systems, molecular dynamics",
            "stability": "Symplectic (preserves energy)",
        },
    }

    return algorithm_info.get(
        algorithm,
        {
            "name": algorithm,
            "description": "No information available. Check Julia DifferentialEquations.jl documentation.",
        },
    )
