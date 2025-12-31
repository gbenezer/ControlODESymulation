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
Scipy Integrator - REFACTORED FOR TypedDict

Adaptive Integration using scipy.integrate.solve_ivp

Wraps scipy's professional-grade ODE solvers with adaptive time stepping,
error control, and automatic stiffness detection.

Supports both controlled and autonomous systems (nu=0).

Design Note
-----------
Refactored to use TypedDict-based IntegrationResult.
All functionality from the original version is preserved, with the key change
being the result type.

Changes from original:
- IntegrationResult imported from src.types.trajectories (not integrator_base)
- Results created as TypedDict with type annotation
- Optional fields added conditionally
- All other functionality preserved (events, autonomous systems, etc.)

Supported Methods:
- RK45: Explicit Runge-Kutta 5(4) - general purpose
- RK23: Explicit Runge-Kutta 3(2) - low accuracy/fast
- DOP853: Explicit Runge-Kutta 8 - high accuracy
- Radau: Implicit Runge-Kutta (Radau IIA) - stiff systems
- BDF: Backward Differentiation Formula - very stiff systems
- LSODA: Automatic stiffness detection and switching
"""

import time
from typing import TYPE_CHECKING, Callable, Optional, Tuple

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


class ScipyIntegrator(IntegratorBase):
    """
    Adaptive integrator using scipy.integrate.solve_ivp.

    Provides access to scipy's suite of professional ODE solvers with
    automatic step size control and error estimation.

    Key Features:
    - Automatic step size adaptation
    - Error control (rtol, atol)
    - Dense output (interpolated solution)
    - Event detection
    - Stiff system support
    - Supports both controlled and autonomous systems

    Available Methods:
    ------------------
    **Explicit (Non-Stiff):**
    - 'RK45': Dormand-Prince 5(4) [DEFAULT]
      * General-purpose, robust
      * Order 5 with 4th-order error estimate
      * Good for most problems

    - 'RK23': Bogacki-Shampine 3(2)
      * Lower accuracy, faster
      * Good for coarse simulations

    - 'DOP853': Dormand-Prince 8(5,3)
      * Very high accuracy
      * Good for precise orbit calculations

    **Implicit (Stiff):**
    - 'Radau': Implicit Runge-Kutta
      * Stiff-stable
      * Good for moderately stiff problems
      * 5th order accuracy

    - 'BDF': Backward Differentiation Formula
      * Very stiff systems
      * Used in circuit simulation, chemistry
      * Variable order (1-5)

    **Automatic:**
    - 'LSODA': Automatic stiffness detection
      * Switches between Adams (non-stiff) and BDF (stiff)
      * Best "set and forget" option
      * Used in MATLAB's ode15s

    Examples
    --------
    >>> # Controlled system - general-purpose adaptive integration
    >>> integrator = ScipyIntegrator(
    ...     system,
    ...     dt=0.01,  # Initial guess
    ...     method='RK45',
    ...     rtol=1e-6,
    ...     atol=1e-8
    ... )
    >>>
    >>> result = integrator.integrate(
    ...     x0=np.array([1.0, 0.0]),
    ...     u_func=lambda t, x: -K @ x,
    ...     t_span=(0.0, 10.0)
    ... )
    >>> print(f"Adaptive steps: {result['nsteps']}")
    >>> print(f"Success: {result['success']}")
    >>>
    >>> # Autonomous system
    >>> integrator = ScipyIntegrator(autonomous_system, method='RK45')
    >>> result = integrator.integrate(
    ...     x0=np.array([1.0, 0.0]),
    ...     u_func=lambda t, x: None,  # No control
    ...     t_span=(0.0, 10.0)
    ... )
    >>>
    >>> # Stiff system (automatic detection)
    >>> stiff_integrator = ScipyIntegrator(
    ...     stiff_system,
    ...     method='LSODA',  # Auto-detects stiffness
    ...     rtol=1e-8,
    ...     atol=1e-10
    ... )
    >>>
    >>> # Very stiff system (explicit method)
    >>> very_stiff_integrator = ScipyIntegrator(
    ...     chem_system,
    ...     method='BDF',  # Backward differentiation
    ...     rtol=1e-10
    ... )
    """

    def __init__(
        self,
        system: "ContinuousSystemBase",
        dt: Optional[ScalarLike] = 0.01,
        method: str = "RK45",
        backend: Backend = "numpy",
        **options,
    ):
        """
        Initialize scipy adaptive integrator.

        Parameters
        ----------
        system : SymbolicDynamicalSystem
            System to integrate (controlled or autonomous)
        dt : Optional[float]
            Initial time step guess (not used for adaptive, but kept for API consistency)
        method : str
            Solver method: 'RK45', 'RK23', 'DOP853', 'Radau', 'BDF', 'LSODA'
        backend : str
            Must be 'numpy' (scipy only supports NumPy)
        **options : dict
            Solver options:
            - rtol: Relative tolerance (default: 1e-6)
            - atol: Absolute tolerance (default: 1e-8)
            - max_step: Maximum step size (default: inf)
            - first_step: Initial step size (default: auto)
            - events: Event functions for detection

        Raises
        ------
        ValueError
            If backend is not 'numpy'
        ImportError
            If scipy is not installed
        """
        if backend != "numpy":
            raise ValueError(
                "ScipyIntegrator only supports NumPy backend. "
                "For PyTorch, use TorchDiffEqIntegrator. "
                "For JAX, use DiffraxIntegrator."
            )

        super().__init__(system, dt, StepMode.ADAPTIVE, backend, **options)

        # Validate method
        valid_methods = ["RK45", "RK23", "DOP853", "Radau", "BDF", "LSODA"]
        if method not in valid_methods:
            raise ValueError(f"Invalid method '{method}'. Choose from: {valid_methods}")

        self.method = method

        # Try to import scipy
        try:
            import scipy.integrate

            self._solve_ivp = scipy.integrate.solve_ivp
        except ImportError:
            raise ImportError(
                "scipy is required for ScipyIntegrator. " "Install with: pip install scipy"
            )

    def step(
        self, x: StateVector, u: Optional[ControlVector] = None, dt: Optional[ScalarLike] = None
    ) -> StateVector:
        """
        Take one integration step (uses integrate() internally).

        For adaptive integrators, this integrates from t=0 to t=dt
        using adaptive stepping internally, then returns the final state.

        Parameters
        ----------
        x : ArrayLike
            Current state
        u : Optional[ArrayLike]
            Control input (None for autonomous systems, assumed constant over step)
        dt : Optional[float]
            Step size (uses self.dt if None)

        Returns
        -------
        ArrayLike
            Next state

        Notes
        -----
        This is less efficient than integrate() for multiple steps
        because it reinitializes the solver each time. Use integrate()
        for trajectory generation.
        """
        dt = dt if dt is not None else self.dt

        # Use integrate for a single step
        u_func = lambda t, x_cur: u  # May be None for autonomous

        result = self.integrate(
            x0=x, u_func=u_func, t_span=(0.0, dt), t_eval=np.array([0.0, dt]), dense_output=False
        )

        # Return final state
        return result["x"][-1]

    def integrate(
        self,
        x0: StateVector,
        u_func: Callable[[ScalarLike, StateVector], Optional[ControlVector]],
        t_span: TimeSpan,
        t_eval: Optional[TimePoints] = None,
        dense_output: bool = False,
        events: Optional[Callable] = None,
    ) -> IntegrationResult:
        """
        Integrate using scipy.solve_ivp with adaptive stepping.

        Parameters
        ----------
        x0 : ArrayLike
            Initial state (nx,)
        u_func : Callable[[float, ArrayLike], Optional[ArrayLike]]
            Control policy (t, x) → u (or None for autonomous systems)
        t_span : Tuple[float, float]
            Integration interval (t_start, t_end)
        t_eval : Optional[ArrayLike]
            Specific times at which to store solution
            If None, solver chooses time points automatically
        dense_output : bool
            If True, compute continuous solution (allows interpolation)
        events : Optional[Callable]
            Event function for detection (e.g., impact, switching)

        Returns
        -------
        IntegrationResult
            TypedDict containing:
            - t: Time points (T,)
            - x: State trajectory (T, nx) - time-major ordering
            - success: Whether integration succeeded
            - message: Status message
            - nfev: Number of function evaluations
            - nsteps: Number of steps taken
            - integration_time: Computation time
            - solver: Integrator name
            - njev: Number of Jacobian evaluations (if available)
            - nlu: Number of LU decompositions (if available)
            - status: Solver status code (if available)
            - sol: Dense output object (if dense_output=True)
            - dense_output: True (if dense_output=True)

        Examples
        --------
        >>> # Controlled system - let solver choose time points
        >>> result = integrator.integrate(
        ...     x0=np.array([1.0, 0.0]),
        ...     u_func=lambda t, x: -K @ x,
        ...     t_span=(0.0, 10.0)
        ... )
        >>> # result["t"] has variable spacing (adaptive!)
        >>> print(f"Used {result['nsteps']} adaptive steps")
        >>>
        >>> # Autonomous system
        >>> result = integrator.integrate(
        ...     x0=np.array([1.0, 0.0]),
        ...     u_func=lambda t, x: None,
        ...     t_span=(0.0, 10.0)
        ... )
        >>> print(f"Autonomous: {result['success']}")
        >>>
        >>> # Evaluate at specific times
        >>> t_eval = np.linspace(0, 10, 1001)
        >>> result = integrator.integrate(
        ...     x0, u_func, (0, 10),
        ...     t_eval=t_eval
        ... )
        >>> # result["t"] matches t_eval
        >>>
        >>> # Dense output for interpolation
        >>> result = integrator.integrate(
        ...     x0, u_func, (0, 10),
        ...     dense_output=True
        ... )
        >>> x_at_5_5 = result["sol"](5.5)  # Interpolate at t=5.5
        >>>
        >>> # Event detection
        >>> def impact_event(t, x):
        ...     return x[1]  # Detect when velocity crosses zero
        >>> impact_event.terminal = True
        >>> result = integrator.integrate(
        ...     x0, u_func, (0, 10),
        ...     events=impact_event
        ... )
        """
        start_time = time.time()

        # Define ODE function for scipy
        def ode_func(t: float, x: np.ndarray) -> np.ndarray:
            """Dynamics function in scipy's signature: f(t, x) → dx/dt"""
            u = u_func(t, x)

            # Ensure state is NumPy array
            x_np = np.asarray(x) if not isinstance(x, np.ndarray) else x

            # Handle autonomous systems - keep None as None
            # Do NOT convert None to array, as np.asarray(None) creates array(None, dtype=object)
            if u is not None:
                u_np = np.asarray(u) if not isinstance(u, np.ndarray) else u
            else:
                u_np = None  # Keep as None for autonomous systems

            # Evaluate dynamics (DynamicsEvaluator handles u=None correctly)
            dx = self.system(x_np, u_np, backend="numpy")

            # Count function evaluation
            self._stats["total_fev"] += 1

            return dx

        # Call scipy.solve_ivp
        sol = self._solve_ivp(
            fun=ode_func,
            t_span=t_span,
            y0=x0,
            method=self.method,
            t_eval=t_eval,
            dense_output=dense_output,
            events=events,
            rtol=self.rtol,
            atol=self.atol,
            max_step=self.options.get("max_step", np.inf),
            first_step=self.options.get("first_step", None),
        )

        elapsed = time.time() - start_time
        self._stats["total_time"] += elapsed
        self._stats["total_steps"] += sol.nfev  # Approximate

        # Create TypedDict result
        result: IntegrationResult = {
            "t": sol.t,
            "x": sol.y.T,  # scipy returns (nx, T), we want (T, nx)
            "success": sol.success,
            "message": sol.message,
            "nfev": sol.nfev,
            "nsteps": sol.nfev,  # scipy doesn't track steps separately
            "integration_time": elapsed,
            "solver": self.name,
        }

        # Add optional fields (only if available)
        if hasattr(sol, "njev"):
            result["njev"] = sol.njev

        if hasattr(sol, "nlu"):
            result["nlu"] = sol.nlu

        if hasattr(sol, "status"):
            result["status"] = sol.status

        # Add dense output if requested
        if dense_output and hasattr(sol, "sol") and sol.sol is not None:
            result["sol"] = sol.sol
            result["dense_output"] = True

        return result

    @property
    def name(self) -> str:
        stiff_indicator = " (Stiff)" if self.method in ["Radau", "BDF"] else ""
        auto_indicator = " (Auto-Stiffness)" if self.method == "LSODA" else ""
        return f"scipy.{self.method}{stiff_indicator}{auto_indicator}"

    def __repr__(self) -> str:
        return (
            f"ScipyIntegrator(method='{self.method}', "
            f"rtol={self.rtol:.1e}, atol={self.atol:.1e})"
        )
