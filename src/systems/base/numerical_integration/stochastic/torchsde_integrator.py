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
TorchSDEIntegrator: PyTorch-based SDE integration using torchsde library.

This module provides GPU-accelerated SDE integration with automatic
differentiation support through PyTorch's autograd. Ideal for neural SDEs,
latent SDEs, and deep learning applications with stochastic dynamics.

Supports both Ito and Stratonovich interpretations, controlled and autonomous
systems, with excellent gradient support for training.

**CUSTOM NOISE LIMITATION**: TorchSDE does NOT support user-provided Brownian
increments. The library generates noise internally for efficiency and to enable
the adjoint method. For custom noise support (deterministic testing, antithetic
variates, etc.), use JAX/Diffrax instead.

Mathematical Form
-----------------
Stochastic differential equations:

    dx = f(x, u, t)dt + g(x, u, t)dW

Available Methods
----------------
**Recommended General Purpose:**
- euler: Euler-Maruyama (strong 0.5, weak 1.0) - fast and robust
- milstein: Milstein method (strong 1.0) - better accuracy
- srk: Stochastic Runge-Kutta - high accuracy

**For Neural SDEs:**
- euler with adjoint: Memory-efficient backprop
- midpoint: Better stability for neural networks

**Adaptive Methods:**
- reversible_heun: Adaptive with error control
- adaptive_heun: Variable time stepping

Key Features
-----------
- **GPU Acceleration**: Native CUDA support via PyTorch
- **Automatic Differentiation**: Full autograd support
- **Adjoint Method**: Memory-efficient backpropagation for neural SDEs
- **Latent SDEs**: Support for latent variable models
- **Batch Processing**: Easy batching with PyTorch tensors
- **Good Seed Reproducibility**: Reliable statistical reproducibility

Limitations
-----------
- **NO custom Brownian motion support** (architectural limitation)
- Cannot provide custom dW for deterministic testing
- Fewer specialized algorithms than Julia
- No implicit methods for stiff SDEs

For custom noise, use JAX/Diffrax. For implicit methods, use Julia/DiffEqPy.

Installation
-----------
Requires PyTorch and torchsde:

```bash
# CPU-only
pip install torch torchsde

# GPU (CUDA)
pip install torch torchsde --index-url https://download.pytorch.org/whl/cu118
```

Examples
--------
>>> # Basic usage (autonomous)
>>> integrator = TorchSDEIntegrator(
...     sde_system,
...     dt=0.01,
...     method='euler',
...     backend='torch'
... )
>>>
>>> result = integrator.integrate(
...     x0=torch.tensor([1.0]),
...     u_func=lambda t, x: None,
...     t_span=(0.0, 10.0)
... )
>>>
>>> # GPU acceleration
>>> integrator.to_device('cuda')
>>>
>>> # Neural SDE with adjoint
>>> integrator = TorchSDEIntegrator(
...     neural_sde,
...     dt=0.01,
...     method='euler',
...     adjoint=True  # Memory-efficient
... )
>>>
>>> # Reproducible via seed (statistical, not pathwise)
>>> integrator = TorchSDEIntegrator(
...     sde_system,
...     dt=0.01,
...     method='euler',
...     seed=42
... )
>>>
>>> # For custom noise, use JAX instead:
>>> # integrator = DiffraxSDEIntegrator(sde, backend='jax')
>>> # x_next = integrator.step(x, u, dW=custom_noise)
"""

import time
import warnings
from typing import Callable, Optional, TYPE_CHECKING

import torch
from torch import Tensor

from src.systems.base.numerical_integration.stochastic.sde_integrator_base import (
    SDEIntegratorBase,
    StepMode,
)

# Import from centralized type system
from src.types.core import (
    StateVector,
    ControlVector,
    ScalarLike,
)
from src.types.trajectories import (
    SDEIntegrationResult,
    TimeSpan,
    TimePoints,
)
from src.types.backends import (
    SDEType,
    ConvergenceType,
    NoiseType,
)

if TYPE_CHECKING:
    from src.systems.base.core.continuous_stochastic_system import ContinuousStochasticSystem


class TorchSDEIntegrator(SDEIntegratorBase):
    """
    PyTorch-based SDE integrator using the torchsde library.

    Provides GPU-accelerated SDE integration with automatic differentiation
    support. Ideal for neural SDEs and deep learning applications.

        Parameters
    ----------
    sde_system : StochasticDynamicalSystem
        SDE system to integrate (controlled or autonomous)
    dt : Optional[ScalarLike]
        Time step size
    step_mode : StepMode
        FIXED or ADAPTIVE stepping mode
    backend : str
        Must be 'torch' for this integrator
    method : str
        Integration method (default: 'euler')
        Options: 'euler', 'milstein', 'srk', 'midpoint', 'reversible_heun'
    sde_type : Optional[SDEType]
        SDE interpretation (None = use system's type)
    convergence_type : ConvergenceType
        Strong or weak convergence
    seed : Optional[int]
        Random seed for reproducibility
    adjoint : bool
        Use adjoint method for memory-efficient backpropagation
        Recommended for neural SDEs (default: False)
    noise_type : Optional[str]
        Noise type: 'diagonal', 'additive', 'scalar', 'general'
        Auto-detected from system if not specified
    **options
        Additional options:
        - rtol : float (default: 1e-3) - Relative tolerance
        - atol : float (default: 1e-6) - Absolute tolerance
        - dt_min : float - Minimum step size (adaptive only)

    Raises
    ------
    ValueError
        If backend is not 'torch'
    ImportError
        If PyTorch or torchsde not installed

    Notes
    -----
    - Backend must be 'torch' (torchsde is PyTorch-only)
    - Adjoint method recommended for neural SDEs to save memory
    - GPU acceleration via .to_device('cuda')
    - Excellent gradient support for training

    Examples
    --------
    >>> # Basic usage
    >>> integrator = TorchSDEIntegrator(
    ...     sde_system,
    ...     dt=0.01,
    ...     method='euler'
    ... )
    >>>
    >>> # High accuracy
    >>> integrator = TorchSDEIntegrator(
    ...     sde_system,
    ...     dt=0.001,
    ...     method='srk'
    ... )
    >>>
    >>> # Neural SDE with adjoint
    >>> integrator = TorchSDEIntegrator(
    ...     neural_sde,
    ...     dt=0.01,
    ...     method='euler',
    ...     adjoint=True
    ... )
    >>>
    >>> # GPU acceleration
    >>> integrator = TorchSDEIntegrator(
    ...     sde_system,
    ...     dt=0.01,
    ...     method='euler'
    ... )
    >>> integrator.to_device('cuda:0')

    **IMPORTANT LIMITATION**: TorchSDE does NOT support custom Brownian motion.
    The library generates noise internally and cannot accept user-provided dW values.
    This is an architectural design decision that enables:
    - Efficient GPU-based noise generation
    - Adjoint method for backpropagation
    - Optimized batched operations

    For custom noise needs (deterministic testing, quasi-Monte Carlo, antithetic
    variates), use JAX/Diffrax which has full custom noise support.

    All methods listed below are verified to work with TorchSDE.
    """

    def __init__(
        self,
        sde_system: "ContinuousStochasticSystem",
        dt: Optional[ScalarLike] = None,
        step_mode: StepMode = StepMode.FIXED,
        backend: str = "torch",
        method: str = "euler",
        sde_type: Optional[SDEType] = None,
        convergence_type: ConvergenceType = ConvergenceType.STRONG,
        seed: Optional[int] = None,
        adjoint: bool = False,
        noise_type: Optional[str] = None,
        **options,
    ):
        """Initialize TorchSDE integrator."""

        # Validate backend
        if backend != "torch":
            raise ValueError(f"TorchSDEIntegrator requires backend='torch', got '{backend}'")

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

        self.method = method.lower()
        self.use_adjoint = adjoint
        self._integrator_name = f"torchsde-{method}"

        # Try to import torchsde
        try:
            import torchsde

            self.torchsde = torchsde
        except ImportError as e:
            raise ImportError(
                "TorchSDEIntegrator requires torchsde.\n\n"
                "Installation:\n"
                "  pip install torchsde\n"
                "Or with CUDA:\n"
                "  pip install torch torchsde --index-url https://download.pytorch.org/whl/cu118"
            ) from e

        # Validate method
        available_methods = [
            "euler",
            "milstein",
            "srk",
            "midpoint",
            "reversible_heun",
            "adaptive_heun",
        ]
        if self.method not in available_methods:
            raise ValueError(f"Unknown method '{method}'. " f"Available: {available_methods}")

        # Validate method compatibility with SDE type
        self._validate_method_sde_compatibility()

        # Auto-detect noise type if not specified
        # Priority for torchsde compatibility:
        # 1. Check if truly scalar (nw=1 AND nx=1) -> 'scalar'
        # 2. Check if additive (constant) -> 'additive'
        # 3. Check if diagonal -> 'diagonal'
        # 4. Otherwise -> 'general'
        if noise_type is None:
            # For torchsde, 'scalar' means nw=1 AND nx=1
            # Otherwise, even with nw=1, we need full (batch, nx, nw) shape
            if self.sde_system.is_additive_noise():
                self.noise_type = "additive"
            elif self.sde_system.is_diagonal_noise():
                self.noise_type = "diagonal"
            else:
                self.noise_type = "general"
        else:
            self.noise_type = noise_type

        # Device management
        self.device = torch.device("cpu")

    def _validate_method_sde_compatibility(self):
        """Validate method compatibility with SDE type."""
        stratonovich_only = ["midpoint", "reversible_heun"]
        ito_only = ["milstein"]
        both = ["euler", "srk", "adaptive_heun"]

        if self.method in stratonovich_only and self.sde_type == SDEType.ITO:
            raise ValueError(
                f"Method '{self.method}' only supports Stratonovich SDEs, "
                f"but system SDE type is Ito.\n"
                f"Solutions:\n"
                f"  1. Use an Ito-compatible method: {ito_only + both}\n"
                f"  2. Change system to Stratonovich (set sde_type='stratonovich')\n"
                f"  3. Override integrator SDE type: sde_type=SDEType.STRATONOVICH"
            )

        if self.method in ito_only and self.sde_type == SDEType.STRATONOVICH:
            raise ValueError(
                f"Method '{self.method}' only supports Ito SDEs, "
                f"but system SDE type is Stratonovich.\n"
                f"Solutions:\n"
                f"  1. Use a Stratonovich-compatible method: {stratonovich_only + both}\n"
                f"  2. Change system to Ito (set sde_type='ito')\n"
                f"  3. Override integrator SDE type: sde_type=SDEType.ITO"
            )

    @property
    def name(self) -> str:
        """Return integrator name."""
        mode_str = "Fixed" if self.step_mode == StepMode.FIXED else "Adaptive"
        adjoint_str = " (Adjoint)" if self.use_adjoint else ""
        return f"{self._integrator_name} ({mode_str}){adjoint_str}"

    def _create_sde_wrapper(self, u_func: Callable):
        """Create torchsde-compatible SDE wrapper."""
        sde_system = self.sde_system
        backend = self.backend
        noise_type = self.noise_type
        sde_type = self.sde_type

        class SDEWrapper(torch.nn.Module):
            """Wrapper to make our SDE compatible with torchsde."""

            def __init__(self):
                super().__init__()
                self.noise_type = noise_type
                self.sde_type = "ito" if sde_type == SDEType.ITO else "stratonovich"

            def f(self, t, y):
                """Drift function."""
                if y.ndim == 1:
                    y = y.unsqueeze(0)

                batch_size = y.shape[0]

                # Process each batch element
                drifts = []
                for i in range(batch_size):
                    y_i = y[i]
                    u = u_func(float(t), y_i)
                    drift_i = sde_system.drift(y_i, u, backend=backend)
                    drifts.append(drift_i)

                return torch.stack(drifts, dim=0)

            def g(self, t, y):
                """Diffusion function."""
                if y.ndim == 1:
                    y = y.unsqueeze(0)

                batch_size = y.shape[0]

                # Process each batch element
                diffusions = []
                for i in range(batch_size):
                    y_i = y[i]
                    u = u_func(float(t), y_i)
                    g_i = sde_system.diffusion(y_i, u, backend=backend)

                    # Ensure proper shape: (nx, nw)
                    if g_i.ndim == 1:
                        g_i = g_i.unsqueeze(-1)

                    diffusions.append(g_i)

                return torch.stack(diffusions, dim=0)

        return SDEWrapper()

    def step(
        self,
        x: StateVector,
        u: Optional[ControlVector] = None,
        dt: Optional[ScalarLike] = None,
        dW: Optional[StateVector] = None,
    ) -> StateVector:
        """
        Take one SDE integration step.

        Handles both single and batched inputs automatically:
        - Input (nx,) → Output (nx,)
        - Input (batch, nx) → Output (batch, nx)

        Parameters
        ----------
        x : StateVector
            Current state (nx,) or (batch, nx)
        u : Optional[ControlVector]
            Control input (nu,) or (batch, nu), or None for autonomous
        dt : Optional[ScalarLike]
            Step size (uses self.dt if None)
        dW : Optional[StateVector]
            **NOT SUPPORTED** - TorchSDE does NOT support custom noise.
            This parameter is IGNORED. Use JAX/Diffrax for custom noise.

        Returns
        -------
        StateVector
            Next state x(t + dt)

        Examples
        --------
        >>> x_next = integrator.step(torch.tensor([1.0]), None)
        >>> # Custom noise ignored:
        >>> x_next = integrator.step(x, u, dW=torch.zeros(1))  # dW ignored!
        """
        if dW is not None:
            warnings.warn(
                "TorchSDE does NOT support custom noise. Parameter 'dW' will be IGNORED. "
                "For custom noise, use JAX/Diffrax: "
                "DiffraxSDEIntegrator(sde, backend='jax').step(x, u, dW=noise)",
                UserWarning,
                stacklevel=2,
            )

        step_size = dt if dt is not None else self.dt
        if step_size is None:
            raise ValueError("Step size dt must be specified")

        # Convert to PyTorch tensors
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)
        if u is not None and not isinstance(u, torch.Tensor):
            u = torch.tensor(u, dtype=torch.float32, device=self.device)

        squeeze_output = False

        # Normalize x to 2D: (batch, nx)
        if x.ndim == 1:
            x = x.unsqueeze(0)
            squeeze_output = True
        elif x.ndim == 2:
            # Already correct shape
            pass
        else:
            raise ValueError(f"x must be 1D (nx,) or 2D (batch, nx), got shape {x.shape}")

        batch_size = x.shape[0]

        # Normalize u to match batch size
        if u is not None:
            if u.ndim == 1:
                u = u.unsqueeze(0)
            if u.shape[0] == 1 and batch_size > 1:
                u = u.expand(batch_size, -1)
            elif u.shape[0] != batch_size:
                raise ValueError(f"Batch size mismatch: x has {batch_size}, u has {u.shape[0]}")

        # Define control function that handles batching
        def u_func_batched(t, x_state):
            return u if u is None else u

        sde = self._create_sde_wrapper(u_func_batched).to(self.device)
        ts = torch.tensor([0.0, step_size], dtype=x.dtype, device=self.device)

        ys = (self.torchsde.sdeint_adjoint if self.use_adjoint else self.torchsde.sdeint)(
            sde, x, ts, method=self.method, dt=step_size
        )

        x_next = ys[-1]
        if squeeze_output:
            x_next = x_next.squeeze(0)

        # Update statistics
        self._stats["total_steps"] += 1
        self._stats["total_fev"] += batch_size
        self._stats["diffusion_evals"] += batch_size

        return x_next

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

        Parameters
        ----------
        x0 : StateVector
            Initial state (nx,)
        u_func : Callable[[ScalarLike, StateVector], Optional[ControlVector]]
            Control policy: (t, x) → u or None
        t_span : TimeSpan
            Time interval (t_start, t_end)
        t_eval : Optional[TimePoints]
            Specific times to evaluate (uses automatic grid if None)
        dense_output : bool
            Not used (TorchSDE doesn't support dense output)

        Returns
        -------
        SDEIntegrationResult
            Integration result with trajectory and diagnostics
        """
        t0, tf = t_span
        t_start = time.perf_counter()

        # Convert to PyTorch tensor
        if not isinstance(x0, torch.Tensor):
            x0 = torch.tensor(x0, dtype=torch.float32, device=self.device)

        # Ensure x0 is on correct device
        x0 = x0.to(self.device)

        # Validate time span
        if tf <= t0:
            raise ValueError(f"End time must be greater than start time. Got t_span=({t0}, {tf})")

        # Prepare time points
        if t_eval is not None:
            ts = (
                torch.tensor(t_eval, dtype=x0.dtype, device=self.device)
                if not isinstance(t_eval, torch.Tensor)
                else t_eval.to(device=self.device, dtype=x0.dtype)
            )
        else:
            n_steps = (
                max(2, int((tf - t0) / self.dt) + 1) if self.step_mode == StepMode.FIXED else 100
            )
            ts = torch.linspace(t0, tf, n_steps, dtype=x0.dtype, device=self.device)

        sde = self._create_sde_wrapper(u_func).to(self.device)

        # Integrate using torchsde
        try:
            # torchsde expects batch dimension: (batch, nx)
            y0 = x0.unsqueeze(0) if x0.ndim == 1 else x0
            ys = (self.torchsde.sdeint_adjoint if self.use_adjoint else self.torchsde.sdeint)(
                sde, y0, ts, method=self.method, dt=self.dt
            )

            # Remove batch dimension if single trajectory
            if ys.shape[1] == 1:
                ys = ys.squeeze(1)

            # Check success
            success = torch.all(torch.isfinite(ys)).item()

            # Update statistics
            nsteps = len(ts) - 1
            integration_time = time.perf_counter() - t_start

            self._stats["total_steps"] += nsteps
            self._stats["total_fev"] += nsteps
            self._stats["diffusion_evals"] += nsteps

            return SDEIntegrationResult(
                t=ts,
                x=ys,
                success=success,
                message=(
                    "TorchSDE integration successful" if success else "Integration failed (NaN/Inf)"
                ),
                nfev=self._stats["total_fev"],
                nsteps=nsteps,
                diffusion_evals=self._stats["diffusion_evals"],
                noise_samples=None,
                n_paths=1,
                convergence_type=self.convergence_type.value,  # Convert enum to string
                solver=self.method,
                sde_type=self.sde_type.value,  # Convert enum to string
                integration_time=integration_time,
            )
        except Exception as e:
            import traceback

            integration_time = time.perf_counter() - t_start

            return SDEIntegrationResult(
                t=torch.tensor([t0], device=self.device),
                x=x0.unsqueeze(0) if x0.ndim == 1 else x0,
                success=False,
                message=f"TorchSDE integration failed: {str(e)}\n{traceback.format_exc()}",
                nfev=0,
                nsteps=0,
                diffusion_evals=0,
                n_paths=1,
                convergence_type=self.convergence_type.value,  # Convert enum to string
                solver=self.method,
                sde_type=self.sde_type.value,  # Convert enum to string
                integration_time=integration_time,
            )

    def integrate_with_gradient(
        self,
        x0: StateVector,
        u_func: Callable[[ScalarLike, StateVector], Optional[ControlVector]],
        t_span: TimeSpan,
        loss_fn: Callable,
        t_eval: Optional[TimePoints] = None,
    ):
        """
        Integrate and compute gradients.

        Parameters
        ----------
        x0 : StateVector
            Initial state (requires gradient)
        u_func : Callable
            Control policy
        t_span : TimeSpan
            Time interval
        loss_fn : Callable
            Loss function operating on integration result
        t_eval : Optional[TimePoints]
            Evaluation times

        Returns
        -------
        tuple
            (loss_value, gradient)
        """
        if not isinstance(x0, torch.Tensor):
            x0 = torch.tensor(x0, dtype=torch.float32, device=self.device)
        if not x0.requires_grad:
            x0 = x0.clone().detach().requires_grad_(True)

        # Integrate
        result = self.integrate(x0, u_func, t_span, t_eval)

        # Compute loss
        loss = loss_fn(result)

        # Compute gradient
        loss.backward()

        return loss.item(), x0.grad.clone()

    def to_device(self, device: str):
        """Move to device."""
        self.device = torch.device(device)

    def enable_adjoint(self):
        """Enable adjoint method."""
        self.use_adjoint = True

    def disable_adjoint(self):
        """Disable adjoint method."""
        self.use_adjoint = False

    @staticmethod
    def list_methods():
        """List available methods."""
        return {
            "basic": ["euler", "midpoint"],
            "high_accuracy": ["milstein", "srk"],
            "adaptive": ["reversible_heun", "adaptive_heun"],
        }

    @staticmethod
    def get_method_info(method: str):
        """Get method information."""
        info = {
            "euler": {
                "name": "Euler-Maruyama",
                "strong_order": 0.5,
                "weak_order": 1.0,
                "description": "Fast and robust, good for general use",
                "best_for": "Neural SDEs, quick simulations",
            },
            "milstein": {
                "name": "Milstein",
                "strong_order": 1.0,
                "weak_order": 1.0,
                "description": "Higher accuracy than Euler",
                "best_for": "When accuracy matters",
            },
            "srk": {
                "name": "Stochastic Runge-Kutta",
                "strong_order": 1.5,
                "weak_order": 1.0,
                "description": "High accuracy",
                "best_for": "Precision requirements",
            },
            "midpoint": {
                "name": "Midpoint",
                "strong_order": 0.5,
                "weak_order": 1.0,
                "description": "Better stability than Euler (Stratonovich only)",
                "best_for": "Neural networks with Stratonovich SDEs",
                "sde_type": "stratonovich",
            },
            "reversible_heun": {
                "name": "Reversible Heun",
                "strong_order": 0.5,
                "weak_order": 1.0,
                "description": "Adaptive with error control (Stratonovich only)",
                "best_for": "Adaptive stepping with Stratonovich",
                "sde_type": "stratonovich",
            },
            "adaptive_heun": {
                "name": "Adaptive Heun",
                "strong_order": 0.5,
                "weak_order": 1.0,
                "description": "Adaptive stepping (both Ito and Stratonovich)",
                "best_for": "When step size needs automatic adjustment",
            },
        }
        return info.get(method, {"name": method, "description": "torchsde method"})

    @staticmethod
    def recommend_method(use_case: str = "general", has_gpu: bool = False):
        """Recommend method based on use case."""
        if use_case == "neural_sde":
            return "euler"
        elif use_case == "high_accuracy":
            return "srk"
        elif use_case == "adaptive":
            return "reversible_heun"
        else:
            return "euler"

    def vectorized_step(
        self,
        x_batch: StateVector,
        u_batch: Optional[ControlVector] = None,
        dt: Optional[ScalarLike] = None,
    ) -> StateVector:
        """
        Vectorized step over batch.

        Parameters
        ----------
        x_batch : StateVector
            Batched states (batch, nx)
        u_batch : Optional[ControlVector]
            Batched controls (batch, nu)
        dt : Optional[ScalarLike]
            Step size

        Returns
        -------
        StateVector
            Next states (batch, nx)
        """
        if not isinstance(x_batch, torch.Tensor):
            x_batch = torch.tensor(x_batch, dtype=torch.float32, device=self.device)
        if u_batch is not None and not isinstance(u_batch, torch.Tensor):
            u_batch = torch.tensor(u_batch, dtype=torch.float32, device=self.device)

        step_size = dt if dt is not None else self.dt

        def u_func(t, x_state):
            if u_batch is None:
                return None
            batch_size = x_state.shape[0] if x_state.ndim > 1 else 1
            return (
                u_batch if batch_size == u_batch.shape[0] else u_batch[0:1].expand(batch_size, -1)
            )

        sde = self._create_sde_wrapper(u_func).to(self.device)
        ts = torch.tensor([0.0, step_size], dtype=x_batch.dtype, device=self.device)
        ys = self.torchsde.sdeint(sde, x_batch, ts, method=self.method, dt=step_size)

        return ys[-1]


def create_torchsde_integrator(
    sde_system: "ContinuousStochasticSystem", method="euler", dt=0.01, **options
):
    """Quick factory for TorchSDE integrators."""
    return TorchSDEIntegrator(sde_system, dt=dt, method=method, backend="torch", **options)


def list_torchsde_methods():
    """Print available methods."""
    methods = TorchSDEIntegrator.list_methods()
    print("TorchSDE Methods (PyTorch-based)")
    print("=" * 60)
    print("\nNOTE: TorchSDE does NOT support custom Brownian motion.")
    print("For custom noise (dW), use JAX/Diffrax instead.\n")

    for category, method_list in methods.items():
        print(f"\n{category.replace('_', ' ').title()}:")
        for method in method_list:
            info = TorchSDEIntegrator.get_method_info(method)
            if "strong_order" in info:
                print(
                    f"  - {method}: {info['description']} "
                    f"(strong {info['strong_order']}, weak {info['weak_order']})"
                )
            else:
                print(f"  - {method}: {info['description']}")

    print("\n" + "=" * 60)
