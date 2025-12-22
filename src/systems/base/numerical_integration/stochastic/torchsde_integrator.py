"""
TorchSDEIntegrator: PyTorch-based SDE integration using torchsde library.

This module provides GPU-accelerated SDE integration with automatic
differentiation support through PyTorch's autograd. Ideal for neural SDEs,
latent SDEs, and deep learning applications with stochastic dynamics.

Supports both Ito and Stratonovich interpretations, controlled and autonomous
systems, with excellent gradient support for training.

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
>>> integrator = TorchSDEIntegrator(
...     sde_system,
...     dt=0.01,
...     method='euler',
...     backend='torch'
... )
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
>>> # With gradients for training
>>> x0 = torch.tensor([1.0], requires_grad=True)
>>> result = integrator.integrate(x0, u_func, t_span)
>>> loss = result.x[-1].sum()
>>> loss.backward()
>>> print(x0.grad)
"""

from typing import Optional, Tuple, Callable, Dict, Any, List
import torch
from torch import Tensor

from src.systems.base.numerical_integration.stochastic.sde_integrator_base import (
    SDEIntegratorBase,
    SDEType,
    ConvergenceType,
    SDEIntegrationResult,
    StepMode,
    ArrayLike
)


class TorchSDEIntegrator(SDEIntegratorBase):
    """
    PyTorch-based SDE integrator using the torchsde library.
    
    Provides GPU-accelerated SDE integration with automatic differentiation
    support. Ideal for neural SDEs and deep learning applications.
    
    Parameters
    ----------
    sde_system : StochasticDynamicalSystem
        SDE system to integrate (controlled or autonomous)
    dt : Optional[float]
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
    noise_type : str
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
    """
    
    def __init__(
        self,
        sde_system,
        dt: Optional[float] = None,
        step_mode: StepMode = StepMode.FIXED,
        backend: str = 'torch',
        method: str = 'euler',
        sde_type: Optional[SDEType] = None,
        convergence_type: ConvergenceType = ConvergenceType.STRONG,
        seed: Optional[int] = None,
        adjoint: bool = False,
        noise_type: Optional[str] = None,
        **options
    ):
        """Initialize TorchSDE integrator."""
        
        # Validate backend
        if backend != 'torch':
            raise ValueError(
                f"TorchSDEIntegrator requires backend='torch', got '{backend}'"
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
            **options
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
        available_methods = ['euler', 'milstein', 'srk', 'midpoint', 
                            'reversible_heun', 'adaptive_heun']
        if self.method not in available_methods:
            raise ValueError(
                f"Unknown method '{method}'. "
                f"Available: {available_methods}"
            )
        
        # Auto-detect noise type if not specified
        if noise_type is None:
            if self.sde_system.is_scalar_noise():
                self.noise_type = 'scalar'
            elif self.sde_system.is_additive_noise():
                self.noise_type = 'additive'
            elif self.sde_system.is_diagonal_noise():
                self.noise_type = 'diagonal'
            else:
                self.noise_type = 'general'
        else:
            self.noise_type = noise_type
        
        # Device management
        self.device = torch.device('cpu')
    
    @property
    def name(self) -> str:
        """Return integrator name."""
        mode_str = "Fixed" if self.step_mode == StepMode.FIXED else "Adaptive"
        adjoint_str = " (Adjoint)" if self.use_adjoint else ""
        return f"{self._integrator_name} ({mode_str}){adjoint_str}"
    
    def _create_sde_wrapper(self, u_func):
        """
        Create torchsde-compatible SDE wrapper.
        
        torchsde expects an SDE class with f() and g() methods that handle
        batched inputs: y has shape (batch, nx).
        """
        sde_system = self.sde_system
        backend = self.backend
        noise_type = self.noise_type
        sde_type = self.sde_type
        
        class SDEWrapper(torch.nn.Module):
            """Wrapper to make our SDE compatible with torchsde."""
            
            def __init__(self):
                super().__init__()
                self.noise_type = noise_type
                self.sde_type = 'ito' if sde_type == SDEType.ITO else 'stratonovich'
            
            def f(self, t, y):
                """
                Drift function: f(t, y) -> dy/dt
                
                Parameters
                ----------
                t : float or Tensor
                    Time
                y : Tensor
                    State with shape (batch, nx)
                    
                Returns
                -------
                Tensor
                    Drift with shape (batch, nx)
                """
                # Handle batched input
                if y.ndim == 1:
                    y = y.unsqueeze(0)  # Add batch dimension
                
                batch_size = y.shape[0]
                
                # Process each batch element
                drifts = []
                for i in range(batch_size):
                    y_i = y[i]
                    u = u_func(float(t), y_i)
                    drift_i = sde_system.drift(y_i, u, backend=backend)
                    drifts.append(drift_i)
                
                # Stack results
                result = torch.stack(drifts, dim=0)
                
                return result
            
            def g(self, t, y):
                """
                Diffusion function: g(t, y) -> diffusion matrix
                
                Parameters
                ----------
                t : float or Tensor
                    Time
                y : Tensor
                    State with shape (batch, nx)
                    
                Returns
                -------
                Tensor
                    Diffusion with shape (batch, nx, nw) for general noise
                    Or (batch, nx) for diagonal/scalar noise
                """
                # Handle batched input
                if y.ndim == 1:
                    y = y.unsqueeze(0)
                
                batch_size = y.shape[0]
                
                # Process each batch element
                diffusions = []
                for i in range(batch_size):
                    y_i = y[i]
                    u = u_func(float(t), y_i)
                    g_i = sde_system.diffusion(y_i, u, backend=backend)
                    
                    # Ensure proper shape
                    if g_i.ndim == 1:
                        g_i = g_i.unsqueeze(-1)  # (nx,) -> (nx, 1)
                    
                    diffusions.append(g_i)
                
                # Stack: (batch, nx, nw)
                result = torch.stack(diffusions, dim=0)
                
                # Handle noise type specific formats
                if noise_type == 'scalar' and result.shape[2] == 1:
                    # Scalar noise: squeeze to (batch, nx)
                    result = result.squeeze(-1)
                elif noise_type == 'diagonal':
                    # Diagonal noise: torchsde might expect (batch, nx)
                    if result.shape[1] == result.shape[2]:  # Square matrix
                        # Extract diagonal
                        result = torch.stack([torch.diag(result[i]) for i in range(batch_size)])
                
                return result
        
        return SDEWrapper()
    
    def step(
        self,
        x: ArrayLike,
        u: Optional[ArrayLike] = None,
        dt: Optional[float] = None,
        dW: Optional[ArrayLike] = None
    ) -> ArrayLike:
        """
        Take one SDE integration step.
        
        Parameters
        ----------
        x : ArrayLike
            Current state (nx,) or (batch, nx)
        u : Optional[ArrayLike]
            Control input (nu,) or (batch, nu), or None for autonomous
        dt : Optional[float]
            Step size (uses self.dt if None)
        dW : Optional[ArrayLike]
            Brownian increments (not used - torchsde generates internally)
            
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
            x = torch.tensor(x, dtype=torch.float32, device=self.device)
        
        if u is not None and not isinstance(u, torch.Tensor):
            u = torch.tensor(u, dtype=torch.float32, device=self.device)
        
        # Define constant control function
        u_func = lambda t, x_state: u
        
        # Create SDE wrapper
        sde = self._create_sde_wrapper(u_func)
        sde = sde.to(self.device)
        
        # Time points
        ts = torch.tensor([0.0, step_size], dtype=x.dtype, device=self.device)
        
        # Integrate using torchsde
        if self.use_adjoint:
            ys = self.torchsde.sdeint_adjoint(
                sde, x.unsqueeze(0), ts, method=self.method, dt=step_size
            )
        else:
            ys = self.torchsde.sdeint(
                sde, x.unsqueeze(0), ts, method=self.method, dt=step_size
            )
        
        # Extract final state
        x_next = ys[-1, 0]
        
        # Update statistics
        self._stats['total_steps'] += 1
        self._stats['total_fev'] += 1
        self._stats['diffusion_evals'] += 1
        
        return x_next
    
    def integrate(
        self,
        x0: ArrayLike,
        u_func: Callable[[float, ArrayLike], Optional[ArrayLike]],
        t_span: Tuple[float, float],
        t_eval: Optional[ArrayLike] = None,
        dense_output: bool = False
    ) -> SDEIntegrationResult:
        """
        Integrate SDE over time interval.
        
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
            If True, enable dense output (not supported by torchsde)
            
        Returns
        -------
        SDEIntegrationResult
            Integration result with trajectory
            
        Examples
        --------
        >>> # Autonomous SDE
        >>> result = integrator.integrate(
        ...     x0=torch.tensor([1.0]),
        ...     u_func=lambda t, x: None,
        ...     t_span=(0.0, 10.0)
        ... )
        >>> 
        >>> # Controlled SDE
        >>> K = torch.tensor([[1.0, 2.0]])
        >>> result = integrator.integrate(
        ...     x0=torch.tensor([1.0, 0.0]),
        ...     u_func=lambda t, x: -K @ x,
        ...     t_span=(0.0, 10.0)
        ... )
        >>> 
        >>> # With gradient computation
        >>> x0 = torch.tensor([1.0], requires_grad=True)
        >>> result = integrator.integrate(x0, u_func, t_span)
        >>> loss = result.x[-1].sum()
        >>> loss.backward()
        >>> print(x0.grad)
        """
        t0, tf = t_span
        
        # Convert to PyTorch tensor
        if not isinstance(x0, torch.Tensor):
            x0 = torch.tensor(x0, dtype=torch.float32, device=self.device)
        
        # Ensure x0 is on correct device
        x0 = x0.to(self.device)
        
        # Validate time span
        if tf <= t0:
            raise ValueError(
                f"End time must be greater than start time. "
                f"Got t_span=({t0}, {tf})"
            )
        
        # Prepare time points
        if t_eval is not None:
            if not isinstance(t_eval, torch.Tensor):
                ts = torch.tensor(t_eval, dtype=x0.dtype, device=self.device)
            else:
                ts = t_eval.to(device=self.device, dtype=x0.dtype)
        else:
            # Generate uniform grid
            if self.step_mode == StepMode.FIXED:
                n_steps = max(2, int((tf - t0) / self.dt) + 1)
            else:
                n_steps = 100  # Default for adaptive
            ts = torch.linspace(t0, tf, n_steps, dtype=x0.dtype, device=self.device)
        
        # Create SDE wrapper
        sde = self._create_sde_wrapper(u_func)
        sde = sde.to(self.device)
        
        # Integrate using torchsde
        try:
            # torchsde expects batch dimension: (batch, nx)
            y0 = x0.unsqueeze(0) if x0.ndim == 1 else x0
            
            if self.use_adjoint:
                ys = self.torchsde.sdeint_adjoint(
                    sde, y0, ts, method=self.method, dt=self.dt
                )
            else:
                ys = self.torchsde.sdeint(
                    sde, y0, ts, method=self.method, dt=self.dt
                )
            
            # Remove batch dimension if single trajectory
            if ys.shape[1] == 1:
                ys = ys.squeeze(1)  # (T, nx)
            
            # Check success
            success = torch.all(torch.isfinite(ys)).item()
            
            # Update statistics
            nsteps = len(ts) - 1
            self._stats['total_steps'] += nsteps
            self._stats['total_fev'] += nsteps
            self._stats['diffusion_evals'] += nsteps
            
            return SDEIntegrationResult(
                t=ts,
                x=ys,
                success=success,
                message="TorchSDE integration successful" if success else "Integration failed (NaN/Inf detected)",
                nfev=self._stats['total_fev'],
                nsteps=nsteps,
                diffusion_evals=self._stats['diffusion_evals'],
                noise_samples=None,  # torchsde doesn't expose noise samples
                n_paths=1,
                convergence_type=self.convergence_type,
                solver=self.method,
                sde_type=self.sde_type,
            )
            
        except Exception as e:
            import traceback
            return SDEIntegrationResult(
                t=torch.tensor([t0], device=self.device),
                x=x0.unsqueeze(0) if x0.ndim == 1 else x0,
                success=False,
                message=f"TorchSDE integration failed: {str(e)}\n{traceback.format_exc()}",
                nfev=0,
                nsteps=0,
                diffusion_evals=0,
                n_paths=1,
                convergence_type=self.convergence_type,
                solver=self.method,
                sde_type=self.sde_type,
            )
    
    # ========================================================================
    # PyTorch-Specific Methods
    # ========================================================================
    
    def integrate_with_gradient(
        self,
        x0: ArrayLike,
        u_func: Callable[[float, ArrayLike], Optional[ArrayLike]],
        t_span: Tuple[float, float],
        loss_fn: Callable[[SDEIntegrationResult], torch.Tensor],
        t_eval: Optional[ArrayLike] = None,
    ):
        """
        Integrate and compute gradients w.r.t. initial conditions.
        
        Parameters
        ----------
        x0 : ArrayLike
            Initial state (requires_grad=True for gradients)
        u_func : Callable
            Control policy
        t_span : Tuple[float, float]
            Time span
        loss_fn : Callable
            Loss function taking IntegrationResult -> scalar tensor
        t_eval : Optional[ArrayLike]
            Evaluation times
            
        Returns
        -------
        tuple
            (loss_value, gradient_wrt_x0)
            
        Examples
        --------
        >>> x0 = torch.tensor([1.0], requires_grad=True)
        >>> 
        >>> def loss_fn(result):
        ...     return torch.sum(result.x[-1]**2)
        >>> 
        >>> loss, grad = integrator.integrate_with_gradient(
        ...     x0, u_func, t_span, loss_fn
        ... )
        """
        # Ensure x0 requires grad
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
        """
        Move integrator computations to specified device.
        
        Parameters
        ----------
        device : str
            Device ('cpu', 'cuda', 'cuda:0', etc.)
            
        Examples
        --------
        >>> integrator.to_device('cuda')
        >>> # All future integrations use GPU
        
        >>> integrator.to_device('cuda:1')
        >>> # Use specific GPU
        """
        self.device = torch.device(device)
    
    def enable_adjoint(self):
        """Enable adjoint method for memory-efficient backpropagation."""
        self.use_adjoint = True
    
    def disable_adjoint(self):
        """Disable adjoint method (use standard backpropagation)."""
        self.use_adjoint = False
    
    def vectorized_step(
        self,
        x_batch: ArrayLike,
        u_batch: Optional[ArrayLike] = None,
        dt: Optional[float] = None
    ):
        """
        Vectorized step over batch of states and controls.
        
        PyTorch naturally handles batched operations.
        
        Parameters
        ----------
        x_batch : ArrayLike
            Batch of states (batch, nx)
        u_batch : Optional[ArrayLike]
            Batch of controls (batch, nu), or None for autonomous
        dt : Optional[float]
            Step size
            
        Returns
        -------
        ArrayLike
            Batch of next states (batch, nx)
        """
        # Convert to tensors
        if not isinstance(x_batch, torch.Tensor):
            x_batch = torch.tensor(x_batch, dtype=torch.float32, device=self.device)
        
        if u_batch is not None and not isinstance(u_batch, torch.Tensor):
            u_batch = torch.tensor(u_batch, dtype=torch.float32, device=self.device)
        
        step_size = dt if dt is not None else self.dt
        
        # Define vectorized control function
        if u_batch is not None:
            def u_func(t, x_state):
                # Return corresponding control for each state
                batch_size = x_state.shape[0] if x_state.ndim > 1 else 1
                if batch_size == u_batch.shape[0]:
                    return u_batch
                else:
                    return u_batch[0:1].expand(batch_size, -1)
        else:
            u_func = lambda t, x_state: None
        
        # Create SDE wrapper
        sde = self._create_sde_wrapper(u_func)
        sde = sde.to(self.device)
        
        # Time points
        ts = torch.tensor([0.0, step_size], dtype=x_batch.dtype, device=self.device)
        
        # Integrate
        ys = self.torchsde.sdeint(sde, x_batch, ts, method=self.method, dt=step_size)
        
        return ys[-1]
    
    # ========================================================================
    # Algorithm Information
    # ========================================================================
    
    @staticmethod
    def list_methods() -> Dict[str, List[str]]:
        """
        List available torchsde methods by category.
        
        Returns
        -------
        Dict[str, List[str]]
            Methods organized by category
            
        Examples
        --------
        >>> methods = TorchSDEIntegrator.list_methods()
        >>> print(methods['basic'])
        ['euler', 'midpoint']
        """
        return {
            'basic': [
                'euler',        # Euler-Maruyama
                'midpoint',     # Midpoint method
            ],
            'high_accuracy': [
                'milstein',     # Milstein method
                'srk',          # Stochastic Runge-Kutta
            ],
            'adaptive': [
                'reversible_heun',  # Adaptive with reversibility
                'adaptive_heun',    # Adaptive Heun
            ],
        }
    
    @staticmethod
    def get_method_info(method: str) -> Dict[str, Any]:
        """
        Get information about a specific method.
        
        Parameters
        ----------
        method : str
            Method name
            
        Returns
        -------
        Dict[str, Any]
            Method properties
            
        Examples
        --------
        >>> info = TorchSDEIntegrator.get_method_info('euler')
        >>> print(info['description'])
        'Fast and robust, good for general use'
        """
        method_info = {
            'euler': {
                'name': 'Euler-Maruyama',
                'strong_order': 0.5,
                'weak_order': 1.0,
                'description': 'Fast and robust, good for general use',
                'best_for': 'Neural SDEs, quick simulations',
            },
            'milstein': {
                'name': 'Milstein',
                'strong_order': 1.0,
                'weak_order': 1.0,
                'description': 'Higher accuracy than Euler',
                'best_for': 'When accuracy matters',
            },
            'srk': {
                'name': 'Stochastic Runge-Kutta',
                'strong_order': 1.5,
                'weak_order': 1.0,
                'description': 'High accuracy',
                'best_for': 'Precision requirements',
            },
            'midpoint': {
                'name': 'Midpoint',
                'strong_order': 0.5,
                'weak_order': 1.0,
                'description': 'Better stability than Euler',
                'best_for': 'Neural networks',
            },
            'reversible_heun': {
                'name': 'Reversible Heun',
                'strong_order': 0.5,
                'weak_order': 1.0,
                'description': 'Adaptive with error control',
                'best_for': 'When adaptive stepping needed',
            },
        }
        
        return method_info.get(
            method,
            {
                'name': method,
                'description': 'torchsde method',
            }
        )
    
    @staticmethod
    def recommend_method(
        use_case: str = 'general',
        has_gpu: bool = False
    ) -> str:
        """
        Recommend method based on use case.
        
        Parameters
        ----------
        use_case : str
            'general', 'neural_sde', 'high_accuracy', 'adaptive'
        has_gpu : bool
            Whether GPU is available
            
        Returns
        -------
        str
            Recommended method name
            
        Examples
        --------
        >>> method = TorchSDEIntegrator.recommend_method('neural_sde')
        >>> print(method)
        'euler'
        """
        if use_case == 'neural_sde':
            return 'euler'  # Fast, works well with adjoint
        elif use_case == 'high_accuracy':
            return 'srk'
        elif use_case == 'adaptive':
            return 'reversible_heun'
        else:
            return 'euler'


# ============================================================================
# Utility Functions
# ============================================================================

def create_torchsde_integrator(
    sde_system,
    method: str = 'euler',
    dt: float = 0.01,
    **options
) -> TorchSDEIntegrator:
    """
    Quick factory for TorchSDE integrators.
    
    Parameters
    ----------
    sde_system : StochasticDynamicalSystem
        SDE system to integrate
    method : str
        Integration method
    dt : float
        Time step
    **options
        Additional options
        
    Returns
    -------
    TorchSDEIntegrator
        Configured integrator
        
    Examples
    --------
    >>> integrator = create_torchsde_integrator(
    ...     sde_system,
    ...     method='euler',
    ...     dt=0.01
    ... )
    """
    return TorchSDEIntegrator(
        sde_system,
        dt=dt,
        method=method,
        backend='torch',
        **options
    )


def list_torchsde_methods() -> None:
    """
    Print all available torchsde methods.
    
    Examples
    --------
    >>> list_torchsde_methods()
    
    TorchSDE Methods (PyTorch-based)
    =================================
    Basic Methods:
      - euler: Euler-Maruyama (strong 0.5, weak 1.0)
    ...
    """
    methods = TorchSDEIntegrator.list_methods()
    
    print("TorchSDE Methods (PyTorch-based)")
    print("=" * 60)
    
    category_names = {
        'basic': 'Basic Methods',
        'high_accuracy': 'High Accuracy Methods',
        'adaptive': 'Adaptive Methods',
    }
    
    for category, method_list in methods.items():
        print(f"\n{category_names.get(category, category)}:")
        for method in method_list:
            info = TorchSDEIntegrator.get_method_info(method)
            if 'strong_order' in info:
                print(f"  - {method}: {info['description']} "
                      f"(strong {info['strong_order']}, weak {info['weak_order']})")
            else:
                print(f"  - {method}: {info['description']}")
    
    print("\n" + "=" * 60)
    print("Use get_method_info(name) for details")
    print("Use recommend_method() for automatic selection")