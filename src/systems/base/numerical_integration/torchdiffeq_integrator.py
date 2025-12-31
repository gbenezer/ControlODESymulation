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
TorchDiffEqIntegrator: PyTorch-based ODE integration using torchdiffeq library.

This module provides adaptive and fixed-step ODE integration with automatic
differentiation support through PyTorch's autograd.

Supports both controlled and autonomous systems (nu=0).

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

import torch
import torchdiffeq
from torch import Tensor

from src.systems.base.numerical_integration.integrator_base import (
    IntegratorBase,
    StepMode,
)

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
from src.types.backends import Backend

if TYPE_CHECKING:
    from src.systems.base.core.continuous_system_base import ContinuousSystemBase


class TorchDiffEqIntegrator(IntegratorBase):
    """
    PyTorch-based ODE integrator using the torchdiffeq library.

    Supports adaptive and fixed-step integration with various solvers
    and automatic differentiation through PyTorch's autograd.

    Parameters
    ----------
    system : SymbolicDynamicalSystem
        Continuous-time system to integrate (controlled or autonomous)
    dt : Optional[ScalarLike]
        Time step size
    step_mode : StepMode
        FIXED or ADAPTIVE stepping mode
    backend : str
        Must be 'torch' for this integrator
    method : str, optional
        Solver method. Options: 'dopri5', 'dopri8', 'adams', 'bosh3',
        'euler', 'midpoint', 'rk4', 'explicit_adams', 'implicit_adams'.
        Default: 'dopri5'
    adjoint : bool, optional
        Use adjoint method for memory-efficient backpropagation. Default: False
        Note: Adjoint method requires the ODE function to be an nn.Module.
        Only use adjoint=True for Neural ODE applications where the system
        is a neural network. For regular dynamical systems, use adjoint=False.
    **options
        Additional options including rtol, atol, max_steps

    Available Methods
    -----------------
    **Adaptive (Recommended):**
    - dopri5: Dormand-Prince 5(4) - general purpose [DEFAULT]
    - dopri8: Dormand-Prince 8 - high accuracy
    - bosh3: Bogacki-Shampine 3(2) - lower accuracy
    - adaptive_heun: Adaptive Heun method
    - fehlberg2: Fehlberg method

    **Fixed-Step:**
    - euler: Forward Euler (1st order)
    - midpoint: Midpoint method (2nd order)
    - rk4: Classic Runge-Kutta 4 (4th order)

    **Multistep:**
    - explicit_adams: Explicit Adams method
    - implicit_adams: Implicit Adams method
    - fixed_adams: Fixed-step Adams method

    Examples
    --------
    >>> import torch
    >>> from torchdiffeq_integrator import TorchDiffEqIntegrator
    >>>
    >>> # Regular controlled dynamical system (adjoint=False)
    >>> integrator = TorchDiffEqIntegrator(
    ...     system,
    ...     dt=0.01,
    ...     backend='torch',
    ...     method='dopri5',
    ...     adjoint=False  # Default for regular systems
    ... )
    >>>
    >>> x0 = torch.tensor([1.0, 0.0])
    >>> result = integrator.integrate(
    ...     x0,
    ...     lambda t, x: torch.zeros(1),
    ...     (0.0, 10.0)
    ... )
    >>> print(f"Success: {result['success']}")
    >>> print(f"Steps: {result['nsteps']}")
    >>>
    >>> # Autonomous system
    >>> integrator = TorchDiffEqIntegrator(autonomous_system, backend='torch')
    >>> result = integrator.integrate(
    ...     x0=torch.tensor([1.0, 0.0]),
    ...     u_func=lambda t, x: None,
    ...     t_span=(0.0, 10.0)
    ... )
    >>>
    >>> # Neural ODE (adjoint=True for memory efficiency)
    >>> class NeuralODE(torch.nn.Module):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.net = torch.nn.Sequential(
    ...             torch.nn.Linear(2, 50),
    ...             torch.nn.Tanh(),
    ...             torch.nn.Linear(50, 2)
    ...         )
    ...     def forward(self, t, x):
    ...         return self.net(x)
    >>>
    >>> neural_ode = NeuralODE()
    >>> integrator_neural = TorchDiffEqIntegrator(
    ...     neural_ode,
    ...     backend='torch',
    ...     method='dopri5',
    ...     adjoint=True  # Memory-efficient for neural networks
    ... )
    """

    def __init__(
        self,
        system: "ContinuousSystemBase",
        dt: Optional[ScalarLike] = None,
        step_mode: StepMode = StepMode.ADAPTIVE,
        backend: Backend = "torch",
        method: str = "dopri5",
        adjoint: bool = False,
        **options,
    ):
        """
        Initialize PyTorch-based integrator.

        Parameters
        ----------
        system : SymbolicDynamicalSystem
            System to integrate (controlled or autonomous)
        dt : Optional[ScalarLike]
            Time step size (initial guess for adaptive)
        step_mode : StepMode
            FIXED or ADAPTIVE stepping
        backend : str
            Must be 'torch'
        method : str
            Solver method (see class docstring)
        adjoint : bool
            Use adjoint method for backpropagation
        **options
            Additional solver options (rtol, atol, etc.)

        Raises
        ------
        ValueError
            If backend is not 'torch' or method is invalid
        """
        # Validate backend
        if backend != "torch":
            raise ValueError(f"TorchDiffEqIntegrator requires backend='torch', got '{backend}'")

        # Initialize base class
        super().__init__(system, dt, step_mode, backend, **options)

        self.method = method.lower()
        self.use_adjoint = adjoint
        self._integrator_name = f"torchdiffeq-{method}"

        # Available methods in torchdiffeq
        self.available_methods = [
            "dopri5",
            "dopri8",
            "bosh3",
            "fehlberg2",
            "adaptive_heun",
            "euler",
            "midpoint",
            "rk4",
            "explicit_adams",
            "implicit_adams",
            "fixed_adams",
            "scipy_solver",
        ]

        if self.method not in self.available_methods:
            raise ValueError(f"Unknown method '{method}'. Available: {self.available_methods}")

        # Select appropriate integration function
        # Note: adjoint requires nn.Module ODE functions
        if self.use_adjoint:
            self._odeint = torchdiffeq.odeint_adjoint
        else:
            self._odeint = torchdiffeq.odeint

    @property
    def name(self) -> str:
        """Return the name of the integrator."""
        mode_str = "Fixed Step" if self.step_mode == StepMode.FIXED else "Adaptive"
        adjoint_str = " (Adjoint)" if self.use_adjoint else ""
        return f"{self._integrator_name} ({mode_str}){adjoint_str}"

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
        ...     x=torch.tensor([1.0, 0.0]),
        ...     u=torch.tensor([0.5])
        ... )
        >>>
        >>> # Autonomous system
        >>> x_next = integrator.step(
        ...     x=torch.tensor([1.0, 0.0]),
        ...     u=None
        ... )
        """
        step_size = dt if dt is not None else self.dt

        if step_size is None:
            raise ValueError("Step size dt must be specified")

        # Convert to PyTorch tensors
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)

        # Handle autonomous systems - keep None as None
        # Do NOT convert None to tensor
        if u is not None and not isinstance(u, torch.Tensor):
            u = torch.tensor(u, dtype=torch.float32, device=x.device)

        # Define ODE function (pure function for autograd)
        def ode_func(t, state):
            return self.system(state, u, backend=self.backend)

        # Time points
        t = torch.tensor([0.0, step_size], dtype=x.dtype, device=x.device)

        # Integrate
        solution = self._odeint(
            ode_func,
            x,
            t,
            method=self.method,
            options={"step_size": step_size} if self.step_mode == StepMode.FIXED else {},
        )

        # Update stats
        self._stats["total_steps"] += 1
        self._stats["total_fev"] += 1  # Approximate

        return solution[-1]

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
        u_func : Callable[[ScalarLike, StateVector], Optional[ControlVector]]
            Control policy: (t, x) → u (or None for autonomous systems)
        t_span : TimeSpan
            Integration interval (t_start, t_end)
        t_eval : Optional[TimePoints]
            Specific times at which to store solution
        dense_output : bool
            If True, return dense interpolated solution (not supported by torchdiffeq)

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

        Examples
        --------
        >>> # Controlled system
        >>> result = integrator.integrate(
        ...     x0=torch.tensor([1.0, 0.0]),
        ...     u_func=lambda t, x: torch.tensor([0.5]),
        ...     t_span=(0.0, 10.0)
        ... )
        >>> print(f"Success: {result['success']}")
        >>> print(f"Final state: {result['x'][-1]}")
        >>>
        >>> # Autonomous system
        >>> result = integrator.integrate(
        ...     x0=torch.tensor([1.0, 0.0]),
        ...     u_func=lambda t, x: None,
        ...     t_span=(0.0, 10.0)
        ... )
        >>>
        >>> # Evaluate at specific times
        >>> t_eval = torch.linspace(0, 10, 1001)
        >>> result = integrator.integrate(x0, u_func, (0, 10), t_eval=t_eval)
        >>> assert result["t"].shape == (1001,)
        """
        t0, tf = t_span

        # Start timing
        start_time = time.perf_counter()

        # Convert to PyTorch tensor
        if not isinstance(x0, torch.Tensor):
            x0 = torch.tensor(x0, dtype=torch.float32)

        # Handle edge cases
        if t0 == tf:
            return IntegrationResult(
                t=torch.tensor([t0], dtype=x0.dtype, device=x0.device),
                x=x0.unsqueeze(0) if x0.ndim == 1 else x0,
                success=True,
                message="Zero time span",
                nfev=0,
                nsteps=0,
                integration_time=0.0,
                solver=self.name,
            )

        # Prepare time points
        fixed_step_methods = [
            "euler",
            "midpoint",
            "rk4",
            "explicit_adams",
            "implicit_adams",
            "fixed_adams",
        ]

        if t_eval is not None:
            if not isinstance(t_eval, torch.Tensor):
                t_eval = torch.tensor(t_eval, dtype=x0.dtype, device=x0.device)
            t_points = t_eval
        else:
            # For fixed-step methods, always generate full time grid
            if self.method in fixed_step_methods or self.step_mode == StepMode.FIXED:
                # Generate uniform grid with specified dt
                n_steps = max(2, int((tf - t0) / self.dt) + 1)
                t_points = torch.linspace(t0, tf, n_steps, dtype=x0.dtype, device=x0.device)
            else:
                # For adaptive methods, can use just endpoints
                t_points = torch.tensor([t0, tf], dtype=x0.dtype, device=x0.device)

        # Define ODE function (pure function for autograd)
        def ode_func(t, state):
            t_val = float(t.item()) if isinstance(t, torch.Tensor) else float(t)
            u = u_func(t_val, state)

            # Handle autonomous systems - keep None as None
            # Do NOT convert None to tensor
            if u is not None and not isinstance(u, torch.Tensor):
                u = torch.tensor(u, dtype=state.dtype, device=state.device)

            return self.system(state, u, backend=self.backend)

        # Set up solver options
        solve_options = {
            "method": self.method,
        }

        # Fixed-step methods require 'options' dict with step_size
        if self.method in fixed_step_methods:
            # These methods REQUIRE fixed step size
            if self.step_mode == StepMode.ADAPTIVE:
                # Force fixed stepping for these methods
                solve_options["options"] = {"step_size": self.dt}
            else:
                solve_options["options"] = {"step_size": self.dt}
        elif self.step_mode == StepMode.ADAPTIVE:
            # Adaptive methods use tolerances
            solve_options["rtol"] = self.rtol
            solve_options["atol"] = self.atol
        else:
            # Fixed step mode with adaptive method
            solve_options["options"] = {"step_size": self.dt}

        # Integrate
        try:
            y = self._odeint(ode_func, x0, t_points, **solve_options)

            # Check success (no NaN/Inf)
            success = torch.all(torch.isfinite(y)).item()

            # Update statistics (torchdiffeq doesn't expose detailed stats)
            nsteps = len(t_points) - 1
            nfev = nsteps * 4  # Approximate for RK4-like methods
            self._stats["total_steps"] += nsteps
            self._stats["total_fev"] += nfev

            # Compute integration time
            integration_time = time.perf_counter() - start_time
            self._stats["total_time"] += integration_time

            return IntegrationResult(
                t=t_points,
                x=y,
                success=success,
                message=(
                    "Integration successful" if success else "Integration failed (NaN/Inf detected)"
                ),
                nfev=nfev,
                nsteps=nsteps,
                integration_time=integration_time,
                solver=self.name,
            )

        except RuntimeError as e:
            # Compute integration time even on failure
            integration_time = time.perf_counter() - start_time
            self._stats["total_time"] += integration_time

            # Integration failed
            return IntegrationResult(
                t=torch.tensor([t0], dtype=x0.dtype, device=x0.device),
                x=x0.unsqueeze(0) if x0.ndim == 1 else x0,
                success=False,
                message=f"Integration failed: {str(e)}",
                nfev=0,
                nsteps=0,
                integration_time=integration_time,
                solver=self.name,
            )

    # ========================================================================
    # PyTorch-Specific Methods
    # ========================================================================

    def integrate_with_gradient(
        self,
        x0: StateVector,
        u_func: Callable[[ScalarLike, StateVector], Optional[ControlVector]],
        t_span: TimeSpan,
        loss_fn: Callable[[IntegrationResult], torch.Tensor],
        t_eval: Optional[TimePoints] = None,
    ):
        """
        Integrate and compute gradients w.r.t. initial conditions.

        Parameters
        ----------
        x0 : StateVector
            Initial state (requires_grad=True for gradients)
        u_func : Callable[[ScalarLike, StateVector], Optional[ControlVector]]
            Control policy (or None for autonomous)
        t_span : TimeSpan
            Time span (t_start, t_end)
        loss_fn : Callable[[IntegrationResult], torch.Tensor]
            Loss function taking IntegrationResult
        t_eval : Optional[TimePoints]
            Evaluation times

        Returns
        -------
        tuple
            (loss_value: float, gradient_wrt_x0: StateVector)

        Examples
        --------
        >>> # Define loss (e.g., final state error)
        >>> def loss_fn(result):
        ...     x_final = result["x"][-1]
        ...     x_target = torch.tensor([1.0, 0.0])
        ...     return torch.sum((x_final - x_target)**2)
        >>>
        >>> # Compute loss and gradient
        >>> x0 = torch.tensor([0.0, 0.0], requires_grad=True)
        >>> loss, grad = integrator.integrate_with_gradient(
        ...     x0=x0,
        ...     u_func=lambda t, x: torch.zeros(1),
        ...     t_span=(0.0, 10.0),
        ...     loss_fn=loss_fn
        ... )
        >>> print(f"Loss: {loss:.4f}")
        >>> print(f"Gradient: {grad}")
        """
        # Ensure x0 requires grad
        if not isinstance(x0, torch.Tensor):
            x0 = torch.tensor(x0, dtype=torch.float32)

        if not x0.requires_grad:
            x0 = x0.clone().detach().requires_grad_(True)

        # Integrate
        result = self.integrate(x0, u_func, t_span, t_eval)

        # Compute loss
        loss = loss_fn(result)

        # Compute gradient
        loss.backward()

        return loss.item(), x0.grad.clone()

    def enable_adjoint(self):
        """
        Enable adjoint method for memory-efficient backpropagation.

        Notes
        -----
        Adjoint method trades computation for memory - useful for Neural ODEs
        with many steps. Requires system to be an nn.Module.

        Examples
        --------
        >>> integrator.enable_adjoint()
        >>> assert integrator.use_adjoint == True
        """
        self.use_adjoint = True
        self._odeint = torchdiffeq.odeint_adjoint

    def disable_adjoint(self):
        """
        Disable adjoint method (use standard backpropagation).

        Examples
        --------
        >>> integrator.disable_adjoint()
        >>> assert integrator.use_adjoint == False
        """
        self.use_adjoint = False
        self._odeint = torchdiffeq.odeint

    def to_device(self, device: str):
        """
        Move system parameters to specified device (if applicable).

        Parameters
        ----------
        device : str
            Device identifier ('cpu', 'cuda', 'cuda:0', etc.)

        Notes
        -----
        This only works if the system is a PyTorch nn.Module.
        For regular dynamical systems, this is a no-op.

        Examples
        --------
        >>> # For Neural ODE systems
        >>> integrator.to_device('cuda:0')
        >>>
        >>> # For regular systems (no effect)
        >>> integrator.to_device('cpu')
        """
        if hasattr(self.system, "to"):
            self.system.to(device)

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"TorchDiffEqIntegrator("
            f"method='{self.method}', "
            f"mode={self.step_mode.value}, "
            f"adjoint={self.use_adjoint})"
        )
