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
"""

from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torchdiffeq
from torch import Tensor

from src.systems.base.numerical_integration.integrator_base import (
    IntegrationResult,
    IntegratorBase,
    StepMode,
)

from src.types import ArrayLike

class TorchDiffEqIntegrator(IntegratorBase):
    """
    PyTorch-based ODE integrator using the torchdiffeq library.

    Supports adaptive and fixed-step integration with various solvers
    and automatic differentiation through PyTorch's autograd.

    Parameters
    ----------
    system : SymbolicDynamicalSystem
        Continuous-time system to integrate (controlled or autonomous)
    dt : Optional[float]
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
        system,
        dt: Optional[float] = None,
        step_mode: StepMode = StepMode.ADAPTIVE,
        backend: str = "torch",
        method: str = "dopri5",
        adjoint: bool = False,  # Changed default to False
        **options,
    ):
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
        self, x: ArrayLike, u: Optional[ArrayLike] = None, dt: Optional[float] = None
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
        x0: ArrayLike,
        u_func: Callable[[float, ArrayLike], Optional[ArrayLike]],
        t_span: Tuple[float, float],
        t_eval: Optional[ArrayLike] = None,
        dense_output: bool = False,
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
        t_eval : Optional[ArrayLike]
            Specific times at which to store solution
        dense_output : bool
            If True, return dense interpolated solution (not supported)

        Returns
        -------
        IntegrationResult
            Object containing t, x, success, and metadata

        Examples
        --------
        >>> # Controlled system
        >>> result = integrator.integrate(
        ...     x0=torch.tensor([1.0, 0.0]),
        ...     u_func=lambda t, x: torch.tensor([0.5]),
        ...     t_span=(0.0, 10.0)
        ... )
        >>>
        >>> # Autonomous system
        >>> result = integrator.integrate(
        ...     x0=torch.tensor([1.0, 0.0]),
        ...     u_func=lambda t, x: None,
        ...     t_span=(0.0, 10.0)
        ... )
        """
        t0, tf = t_span

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
        fixed_step_methods = [
            "euler",
            "midpoint",
            "rk4",
            "explicit_adams",
            "implicit_adams",
            "fixed_adams",
        ]

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

            return IntegrationResult(
                t=t_points,
                x=y,
                success=success,
                message=(
                    "Integration successful" if success else "Integration failed (NaN/Inf detected)"
                ),
                nfev=nfev,
                nsteps=nsteps,
                method=self.method,
                adjoint=self.use_adjoint,
            )

        except RuntimeError as e:
            # Integration failed
            return IntegrationResult(
                t=torch.tensor([t0], dtype=x0.dtype, device=x0.device),
                x=x0.unsqueeze(0) if x0.ndim == 1 else x0,
                success=False,
                message=f"Integration failed: {str(e)}",
                nfev=0,
                nsteps=0,
            )

    # ========================================================================
    # PyTorch-Specific Methods
    # ========================================================================

    def integrate_with_gradient(
        self,
        x0: ArrayLike,
        u_func: Callable[[float, ArrayLike], Optional[ArrayLike]],
        t_span: Tuple[float, float],
        loss_fn: Callable[[IntegrationResult], torch.Tensor],
        t_eval: Optional[ArrayLike] = None,
    ):
        """
        Integrate and compute gradients w.r.t. initial conditions.

        Parameters
        ----------
        x0 : ArrayLike
            Initial state (requires_grad=True for gradients)
        u_func : Callable
            Control policy (or None for autonomous)
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
        """Enable adjoint method for memory-efficient backpropagation."""
        self.use_adjoint = True
        self._odeint = torchdiffeq.odeint_adjoint

    def disable_adjoint(self):
        """Disable adjoint method (use standard backpropagation)."""
        self.use_adjoint = False
        self._odeint = torchdiffeq.odeint

    def to_device(self, device: str):
        """
        Move integrator computations to specified device.

        Parameters
        ----------
        device : str
            Device ('cpu', 'cuda', 'cuda:0', etc.)

        Examples
        --------
        >>> integrator.to_device('cuda')
        >>> x0 = torch.tensor([1.0], device='cuda')
        >>> result = integrator.integrate(x0, u_func, (0, 10))
        """
        self.device = torch.device(device)
        # Note: Actual tensors must be moved by user
        # This just stores the device preference

    def vectorized_step(
        self, x_batch: ArrayLike, u_batch: Optional[ArrayLike] = None, dt: Optional[float] = None
    ):
        """
        Vectorized step over batch of states and controls.

        PyTorch naturally handles batched operations.

        Parameters
        ----------
        x_batch : ArrayLike
            Batch of states (batch, nx)
        u_batch : Optional[ArrayLike]
            Batch of controls (batch, nu), or None for autonomous systems
        dt : Optional[float]
            Step size

        Returns
        -------
        ArrayLike
            Batch of next states (batch, nx)
        """
        # PyTorch handles batching automatically
        # Just need to ensure proper shapes
        if not isinstance(x_batch, torch.Tensor):
            x_batch = torch.tensor(x_batch, dtype=torch.float32)

        # Handle autonomous systems
        if u_batch is not None and not isinstance(u_batch, torch.Tensor):
            u_batch = torch.tensor(u_batch, dtype=torch.float32)

        step_size = dt if dt is not None else self.dt

        # Vectorized ODE function
        def ode_func(t, state_batch):
            # Process each state with corresponding control
            if u_batch is not None:
                return torch.stack(
                    [
                        self.system(state_batch[i], u_batch[i], backend=self.backend)
                        for i in range(state_batch.shape[0])
                    ]
                )
            else:
                # Autonomous - no control
                return torch.stack(
                    [
                        self.system(state_batch[i], None, backend=self.backend)
                        for i in range(state_batch.shape[0])
                    ]
                )

        t = torch.tensor([0.0, step_size], dtype=x_batch.dtype, device=x_batch.device)

        solution = self._odeint(
            ode_func,
            x_batch,
            t,
            method=self.method,
        )

        return solution[-1]

    def set_options(self, **options):
        """Update solver options."""
        self.options.update(options)
        if "rtol" in options:
            self.rtol = options["rtol"]
        if "atol" in options:
            self.atol = options["atol"]
        if "adjoint" in options:
            self.use_adjoint = options["adjoint"]
            if self.use_adjoint:
                self._odeint = torchdiffeq.odeint_adjoint
            else:
                self._odeint = torchdiffeq.odeint

    def get_options(self) -> Dict[str, Any]:
        """Get current solver options."""
        return {
            "method": self.method,
            "step_mode": self.step_mode.value,
            "rtol": self.rtol,
            "atol": self.atol,
            "adjoint": self.use_adjoint,
            "dt": self.dt,
            **self.options,
        }


# ============================================================================
# Utility Functions
# ============================================================================


def create_torchdiffeq_integrator(
    system,
    method: str = "dopri5",
    dt: Optional[float] = 0.01,
    step_mode: StepMode = StepMode.ADAPTIVE,
    adjoint: bool = False,
    **options,
) -> TorchDiffEqIntegrator:
    """
    Quick factory for TorchDiffEq integrators.

    Parameters
    ----------
    system : SymbolicDynamicalSystem
        System to integrate (controlled or autonomous)
    method : str
        Solver method ('dopri5', 'rk4', 'euler', etc.)
    dt : Optional[float]
        Time step
    step_mode : StepMode
        FIXED or ADAPTIVE
    adjoint : bool
        Use adjoint method for backprop (default: False for regular systems)
    **options
        Additional solver options

    Returns
    -------
    TorchDiffEqIntegrator
        Configured integrator

    Examples
    --------
    >>> # Regular dynamical system
    >>> integrator = create_torchdiffeq_integrator(
    ...     system,
    ...     method='dopri5',
    ...     adjoint=False,
    ...     rtol=1e-7
    ... )
    >>>
    >>> # Autonomous system
    >>> integrator = create_torchdiffeq_integrator(autonomous_system)
    """
    return TorchDiffEqIntegrator(
        system=system,
        dt=dt,
        step_mode=step_mode,
        backend="torch",
        method=method,
        adjoint=adjoint,
        **options,
    )
