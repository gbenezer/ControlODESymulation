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
Backend and Configuration Types

Defines types related to:
- Computational backends (NumPy, PyTorch, JAX)
- Device management (CPU, CUDA, MPS)
- Integration methods (RK45, Euler, etc.)
- Discretization methods (exact, tustin, etc.)
- SDE integration methods (Euler-Maruyama, Milstein, etc.)
- System configuration dictionaries
- Noise and stochastic types

These types standardize backend selection and algorithm configuration
across the entire framework.

Usage
-----
>>> from src.types.backends import Backend, Device, IntegrationMethod
>>>
>>> def integrate(
...     system,
...     backend: Backend = 'numpy',
...     method: IntegrationMethod = 'RK45'
... ):
...     # Type-safe backend and method selection
...     pass
"""

# Conditional imports
from typing import TYPE_CHECKING, Any, Dict, Literal, Optional

import numpy as np
from typing_extensions import TypedDict

if TYPE_CHECKING:
    import sympy as sp


# ============================================================================
# Backend Types
# ============================================================================

Backend = Literal["numpy", "torch", "jax"]
"""
Backend identifier for numerical computation.

Valid values:
- 'numpy': NumPy arrays (CPU-based, stable, universal)
- 'torch': PyTorch tensors (GPU support, autodiff, neural networks)
- 'jax': JAX arrays (JIT compilation, GPU/TPU, functional)

Backend Selection Guide:
- Production/Stability: 'numpy'
- GPU acceleration: 'torch' or 'jax'
- Gradient computation: 'torch' or 'jax'
- JIT compilation: 'jax'
- Neural networks: 'torch'
- Functional programming: 'jax'

Examples
--------
>>> backend: Backend = 'torch'
>>> system.set_default_backend('jax')
>>> 
>>> # Conditional on backend
>>> if backend == 'torch':
...     import torch
...     x = torch.tensor(x_np)
"""

Device = str
"""
Device identifier for hardware acceleration.

Common values:
- 'cpu': CPU computation
- 'cuda': NVIDIA GPU (any available)
- 'cuda:0', 'cuda:1', ...: Specific NVIDIA GPU
- 'mps': Apple Metal (M1/M2/M3 Macs)
- 'tpu': Google TPU (JAX only)

Examples
--------
>>> device: Device = 'cuda:0'
>>> system.set_preferred_device('cpu')
>>> 
>>> # PyTorch usage
>>> import torch
>>> tensor = torch.tensor(data, device='cuda')
>>> 
>>> # JAX usage (via jax.devices)
>>> import jax
>>> device = jax.devices('gpu')[0]
"""


class BackendConfig(TypedDict, total=False):
    """
    Backend configuration dictionary.

    Specifies backend, device, and precision settings.

    Attributes
    ----------
    backend : Backend
        Computational backend
    device : Optional[Device]
        Hardware device
    dtype : Optional[str]
        Data type ('float32', 'float64', etc.)

    Examples
    --------
    >>> config: BackendConfig = {
    ...     'backend': 'torch',
    ...     'device': 'cuda:0',
    ...     'dtype': 'float32'
    ... }
    >>>
    >>> # Minimal config
    >>> config_minimal: BackendConfig = {'backend': 'numpy'}
    """

    backend: Backend
    device: Optional[Device]
    dtype: Optional[str]


# ============================================================================
# Integration Method Types
# ============================================================================

IntegrationMethod = str
"""
Integration method for continuous-time systems (ODEs).
"""

DiscretizationMethod = str
"""
Discretization method for continuous → discrete transformation.
"""

SDEIntegrationMethod = str
"""
SDE integration method for stochastic differential equations.
"""

OptimizationMethod = str
"""
Optimization method for control/estimation problems.

"""


# ============================================================================
# Noise and Stochastic Types
# ============================================================================

NoiseType = Literal["additive", "multiplicative", "diagonal", "scalar", "general"]
"""
Noise structure classification for stochastic systems.

Categories:
- 'additive': g(x,u,t) = constant (state-independent)
  * Most efficient - can precompute
  * Example: dx = f(x)dt + σ*dW
  
- 'multiplicative': g(x,u,t) depends on state
  * State-dependent noise intensity
  * Example: dx = f(x)dt + σ*x*dW (Geometric Brownian Motion)
  
- 'diagonal': g(x,u,t) is diagonal matrix
  * Independent noise sources
  * Enables element-wise solvers
  
- 'scalar': Single noise source (nw=1)
  * Simplest stochastic case
  * One Wiener process
  
- 'general': Full coupling, no special structure
  * Most general, least efficient

Examples
--------
>>> noise_type: NoiseType = system.get_noise_type()
>>> 
>>> if noise_type == 'additive':
...     # Can optimize: precompute constant diffusion
...     G = system.get_constant_noise()
... elif noise_type == 'multiplicative':
...     # Must evaluate at each step
...     G = system.diffusion(x, u)
"""

SDEType = Literal["ito", "stratonovich"]
"""
SDE interpretation type.

Interpretations:
- 'ito': Itô interpretation (more common in control/finance)
  * dx = f(x)dt + g(x)dW
  * Martingale property
  * Simpler numerically
  
- 'stratonovich': Stratonovich interpretation (physics/engineering)
  * dx = f(x)dt + g(x)∘dW
  * Chain rule works normally
  * More intuitive for physical systems

Conversion:
  f_Stratonovich = f_Ito + 0.5 * g * (∂g/∂x)

For discrete systems: No distinction (both equivalent)

Examples
--------
>>> sde_type: SDEType = 'ito'
>>> system.sde_type = 'stratonovich'
>>> 
>>> # In define_system()
>>> self.sde_type = 'ito'
"""

ConvergenceType = Literal["strong", "weak"]
"""
SDE convergence type for numerical integration.

Types:
- 'strong': Pathwise/strong convergence
  * Individual sample paths converge
  * E[|X_numerical - X_true|] → 0
  * Needed for: Filtering, control synthesis, single trajectory accuracy
  * More expensive to achieve
  
- 'weak': Weak convergence
  * Distributions/moments converge
  * E[φ(X_numerical)] → E[φ(X_true)] for test functions φ
  * Needed for: Monte Carlo, statistics, ensemble behavior
  * Easier to achieve (higher order possible)

Order Comparison:
- Euler-Maruyama: Strong order 0.5, weak order 1.0
- Milstein: Strong order 1.0
- SRA1: Weak order 2.0

Examples
--------
>>> conv_type: ConvergenceType = 'strong'
>>> integrator = SDEIntegrator(system, convergence_type='weak')
>>> 
>>> # For single trajectory (e.g., control)
>>> conv: ConvergenceType = 'strong'
>>> 
>>> # For Monte Carlo (e.g., option pricing)
>>> conv: ConvergenceType = 'weak'
"""


# ============================================================================
# System Configuration Types
# ============================================================================


class SystemConfig(TypedDict, total=False):
    """
    Complete system configuration dictionary.

    Contains all system metadata and settings.

    Attributes
    ----------
    name : str
        System name (user-defined)
    class_name : str
        Python class name
    nx : int
        State dimension
    nu : int
        Control dimension
    ny : int
        Output dimension
    nw : int
        Noise dimension (stochastic only)
    is_discrete : bool
        Discrete-time vs continuous-time
    is_stochastic : bool
        Stochastic vs deterministic
    is_autonomous : bool
        Autonomous (nu=0) vs controlled
    backend : Backend
        Default computational backend
    device : Device
        Preferred hardware device
    parameters : Dict
        Symbolic parameter values

    Examples
    --------
    >>> config: SystemConfig = system.get_config_dict()
    >>> print(f"System: {config['class_name']}")
    >>> print(f"States: {config['nx']}, Controls: {config['nu']}")
    >>> print(f"Backend: {config['backend']}, Device: {config['device']}")
    >>>
    >>> # Create from scratch
    >>> config: SystemConfig = {
    ...     'name': 'Pendulum',
    ...     'class_name': 'InvertedPendulum',
    ...     'nx': 2,
    ...     'nu': 1,
    ...     'ny': 2,
    ...     'is_discrete': False,
    ...     'is_stochastic': False,
    ...     'backend': 'numpy',
    ... }
    """

    name: str
    class_name: str
    nx: int
    nu: int
    ny: int
    nw: int
    is_discrete: bool
    is_stochastic: bool
    is_autonomous: bool
    backend: Backend
    device: Device
    parameters: Dict[Any, float]


class IntegratorConfig(TypedDict, total=False):
    """
    Configuration for continuous-time integrators.

    Specifies integration method and tolerances.

    Attributes
    ----------
    method : IntegrationMethod
        Integration algorithm
    rtol : float
        Relative tolerance
    atol : float
        Absolute tolerance
    max_step : float
        Maximum allowed time step
    first_step : Optional[float]
        Initial step size (adaptive methods)
    vectorized : bool
        Whether dynamics function is vectorized
    dense_output : bool
        Compute dense output (for interpolation)

    Examples
    --------
    >>> # Standard configuration
    >>> config: IntegratorConfig = {
    ...     'method': 'RK45',
    ...     'rtol': 1e-6,
    ...     'atol': 1e-9,
    ...     'max_step': 0.1
    ... }
    >>>
    >>> # High accuracy configuration
    >>> config_accurate: IntegratorConfig = {
    ...     'method': 'DOP853',
    ...     'rtol': 1e-10,
    ...     'atol': 1e-12,
    ...     'dense_output': True
    ... }
    >>>
    >>> # Stiff system configuration
    >>> config_stiff: IntegratorConfig = {
    ...     'method': 'Radau',
    ...     'rtol': 1e-6,
    ...     'atol': 1e-8,
    ... }
    """

    method: IntegrationMethod
    rtol: float
    atol: float
    max_step: float
    first_step: Optional[float]
    vectorized: bool
    dense_output: bool


class DiscretizerConfig(TypedDict, total=False):
    """
    Configuration for system discretization.

    Specifies how continuous system is converted to discrete.

    Attributes
    ----------
    dt : float
        Time step (sampling period)
    method : DiscretizationMethod
        Discretization algorithm
    backend : Backend
        Backend for discrete system
    order : int
        Approximation order (for taylor-based methods)
    preserve_stability : bool
        Whether to preserve continuous-time stability

    Examples
    --------
    >>> # Standard configuration
    >>> config: DiscretizerConfig = {
    ...     'dt': 0.01,
    ...     'method': 'exact',
    ...     'backend': 'numpy'
    ... }
    >>>
    >>> # Digital control design
    >>> config_control: DiscretizerConfig = {
    ...     'dt': 0.01,
    ...     'method': 'zoh',
    ...     'preserve_stability': True
    ... }
    >>>
    >>> # Fast simulation
    >>> config_fast: DiscretizerConfig = {
    ...     'dt': 0.001,
    ...     'method': 'euler',
    ...     'backend': 'torch'
    ... }
    """

    dt: float
    method: DiscretizationMethod
    backend: Backend
    order: int
    preserve_stability: bool


class SDEIntegratorConfig(TypedDict, total=False):
    """
    Configuration for SDE integrators.

    Extends integrator config with stochastic-specific settings.

    Attributes
    ----------
    method : SDEIntegrationMethod
        SDE integration algorithm
    dt : float
        Time step (fixed-step methods)
    convergence_type : ConvergenceType
        Strong (pathwise) or weak (distributional)
    backend : Backend
        Computational backend
    seed : Optional[int]
        Random seed for reproducibility
    adaptive : bool
        Use adaptive stepping (if available)

    Examples
    --------
    >>> # Basic configuration
    >>> config: SDEIntegratorConfig = {
    ...     'method': 'euler',
    ...     'dt': 0.01,
    ...     'convergence_type': 'strong',
    ...     'backend': 'numpy'
    ... }
    >>>
    >>> # High-accuracy additive noise
    >>> config_accurate: SDEIntegratorConfig = {
    ...     'method': 'SRIW1',
    ...     'dt': 0.001,
    ...     'convergence_type': 'strong',
    ...     'backend': 'numpy',
    ...     'seed': 42
    ... }
    >>>
    >>> # Monte Carlo simulation
    >>> config_mc: SDEIntegratorConfig = {
    ...     'method': 'SEA',
    ...     'dt': 0.01,
    ...     'convergence_type': 'weak',
    ...     'backend': 'jax',
    ... }
    """

    method: SDEIntegrationMethod
    dt: float
    convergence_type: ConvergenceType
    backend: Backend
    seed: Optional[int]
    adaptive: bool


# ============================================================================
# Constants - Valid Values
# ============================================================================

VALID_BACKENDS = ("numpy", "torch", "jax")
"""
Tuple of valid backend names.

Use for validation:
>>> if backend not in VALID_BACKENDS:
...     raise ValueError(f"Invalid backend: {backend}")
"""

VALID_DEVICES = ("cpu", "cuda", "mps", "tpu")
"""
Common device identifiers.

Note: Actual availability depends on hardware and installed libraries.
- 'cpu': Always available
- 'cuda': Requires NVIDIA GPU + CUDA
- 'mps': Requires Apple Silicon (M1/M2/M3)
- 'tpu': Requires Google Cloud TPU + JAX

Examples
--------
>>> import torch
>>> if torch.cuda.is_available():
...     device = 'cuda'
... else:
...     device = 'cpu'
"""

DEFAULT_BACKEND: Backend = "numpy"
"""
Default backend if not specified.

NumPy is default because:
- Always available (core dependency)
- Stable and well-tested
- No GPU required
- Works everywhere

Examples
--------
>>> backend = DEFAULT_BACKEND  # 'numpy'
"""

DEFAULT_DEVICE: Device = "cpu"
"""
Default device if not specified.

CPU is default because:
- Always available
- No special hardware needed
- Predictable performance

Examples
--------
>>> device = DEFAULT_DEVICE  # 'cpu'
"""

DEFAULT_DTYPE = np.float64
"""
Default numerical precision.

Float64 (double precision) is default for:
- Numerical stability
- Control applications need precision
- Most scientific computing standard

Can override for:
- float32: Faster, less memory (neural networks, GPU)
- float16: Even faster (mixed precision training)

Examples
--------
>>> dtype = DEFAULT_DTYPE  # np.float64
>>> 
>>> # Override for GPU
>>> if backend == 'torch' and device == 'cuda':
...     dtype = torch.float32
"""

# ============================================================================
# Method Selection Utilities
# ============================================================================

def get_backend_default_method(backend: Backend, is_stochastic: bool = False) -> str:
    """
    Get default integration method for backend.

    Parameters
    ----------
    backend : Backend
        Computational backend
    is_stochastic : bool
        Whether system is stochastic

    Returns
    -------
    str
        Default method name for this backend

    Examples
    --------
    >>> get_backend_default_method('numpy', is_stochastic=False)
    'RK45'
    >>> get_backend_default_method('numpy', is_stochastic=True)
    'EM'
    >>> get_backend_default_method('torch', is_stochastic=True)
    'euler'
    """
    if is_stochastic:
        # SDE defaults by backend
        defaults = {
            "numpy": "EM",  # Julia Euler-Maruyama
            "torch": "euler",  # TorchSDE euler
            "jax": "Euler",  # Diffrax Euler
        }
    else:
        # ODE defaults by backend
        defaults = {
            "numpy": "RK45",  # SciPy adaptive RK
            "torch": "rk4",  # Fixed-step RK4
            "jax": "rk4",  # Fixed-step RK4
        }

    return defaults.get(backend, "euler")


def validate_backend(backend: str) -> Backend:
    """
    Validate and normalize backend string.

    Parameters
    ----------
    backend : str
        Backend name to validate

    Returns
    -------
    Backend
        Validated backend (typed)

    Raises
    ------
    ValueError
        If backend is not valid

    Examples
    --------
    >>> validate_backend('numpy')
    'numpy'
    >>> validate_backend('pytorch')  # ValueError
    """
    if backend not in VALID_BACKENDS:
        raise ValueError(f"Invalid backend '{backend}'. " f"Choose from: {VALID_BACKENDS}")
    return backend


def validate_device(device: str, backend: Backend) -> Device:
    """
    Validate device for given backend.

    Parameters
    ----------
    device : str
        Device identifier
    backend : Backend
        Backend being used

    Returns
    -------
    Device
        Validated device

    Raises
    ------
    ValueError
        If device incompatible with backend

    Examples
    --------
    >>> validate_device('cuda', 'torch')
    'cuda'
    >>> validate_device('cuda', 'numpy')  # ValueError - NumPy is CPU-only
    """
    # NumPy is CPU-only
    if backend == "numpy" and device not in ("cpu", "default"):
        raise ValueError(f"NumPy backend only supports CPU, got device='{device}'")

    # Basic validation (actual availability checked at runtime)
    if device.startswith("cuda"):
        if backend not in ("torch", "jax"):
            raise ValueError(f"CUDA device requires torch or jax backend, got '{backend}'")

    if device == "mps":
        if backend != "torch":
            raise ValueError(f"MPS device requires torch backend, got '{backend}'")

    return device


# ============================================================================
# Export All
# ============================================================================

__all__ = [
    # Backend types
    "Backend",
    "Device",
    "BackendConfig",
    # Method types
    "IntegrationMethod",
    "DiscretizationMethod",
    "SDEIntegrationMethod",
    "OptimizationMethod",
    # Noise and stochastic
    "NoiseType",
    "SDEType",
    "ConvergenceType",
    # Configuration
    "SystemConfig",
    "IntegratorConfig",
    "DiscretizerConfig",
    "SDEIntegratorConfig",
    # Constants
    "VALID_BACKENDS",
    "VALID_DEVICES",
    "DEFAULT_BACKEND",
    "DEFAULT_DEVICE",
    "DEFAULT_DTYPE",
    # Utilities
    "get_backend_default_method",
    "validate_backend",
    "validate_device",
]
