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
Utility Types and Functions

Defines utility types and helper functions for:
- Type guards (checking array backends)
- Type converters (between NumPy/PyTorch/JAX)
- Shape validators
- Dimension extractors
- Protocols (structural subtyping)
- Cache and metadata types
- Validation and performance metrics

These utilities enable type-safe operations across different backends
and provide runtime type checking and conversion capabilities.

Usage
-----
>>> from src.types.utilities import (
...     is_batched,
...     ensure_numpy,
...     get_backend,
...     ArrayConverter,
... )
>>>
>>> # Check if array is batched
>>> if is_batched(x):
...     batch_size = get_batch_size(x)
>>>
>>> # Convert to NumPy for analysis
>>> x_np = ensure_numpy(x)
>>>
>>> # Detect backend
>>> backend = get_backend(x)
"""

# Conditional imports for type checking
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Protocol, Tuple
from dataclasses import dataclass

import numpy as np
from typing_extensions import TypedDict

# Import from backends for Backend type
from src.types.backends import Backend

# Import from core for base types
from src.types.core import (
    ArrayLike,
    ControlVector,
    OutputVector,
    StateVector,
    SystemDimensions,
)

if TYPE_CHECKING:
    import jax.numpy as jnp
    import torch


# ============================================================================
# Protocol Definitions (Structural Subtyping)
# ============================================================================


class LinearizableProtocol(Protocol):
    """
    Protocol for systems that can be linearized.

    Any class implementing a linearize() method satisfies this protocol,
    enabling structural subtyping without explicit inheritance.

    Methods
    -------
    linearize(x_eq, u_eq, **kwargs)
        Compute linearization at equilibrium point

    Examples
    --------
    >>> def analyze_stability(system: LinearizableProtocol, x_eq, u_eq):
    ...     '''Works with any linearizable system.'''
    ...     result = system.linearize(x_eq, u_eq)
    ...     A = result[0]
    ...     eigenvalues = np.linalg.eigvals(A)
    ...     return np.all(np.real(eigenvalues) < 0)
    >>>
    >>> # Works with any system that has linearize()
    >>> is_stable = analyze_stability(my_system, x_eq, u_eq)
    """

    def linearize(
        self, x_eq: StateVector, u_eq: Optional[ControlVector] = None, **kwargs
    ) -> Any:  # Returns LinearizationResult
        """
        Compute linearization at equilibrium.

        Parameters
        ----------
        x_eq : StateVector
            Equilibrium state
        u_eq : Optional[ControlVector]
            Equilibrium control
        **kwargs
            Additional method-specific arguments

        Returns
        -------
        LinearizationResult
            (A, B) for deterministic or (A, B, G) for stochastic
        """
        ...


class SimulatableProtocol(Protocol):
    """
    Protocol for systems that can be simulated/stepped.

    Any class implementing a step() or integrate() method satisfies this.

    Methods
    -------
    step(x, u, **kwargs)
        Compute next state or derivative

    Examples
    --------
    >>> def run_simulation(system: SimulatableProtocol, x0, u_seq):
    ...     '''Works with any simulatable system.'''
    ...     x = x0
    ...     trajectory = [x]
    ...     for u in u_seq:
    ...         x = system.step(x, u)
    ...         trajectory.append(x)
    ...     return np.array(trajectory)
    >>>
    >>> # Works with discrete or integrated continuous systems
    >>> traj = run_simulation(my_system, x0, controls)
    """

    def step(self, x: StateVector, u: Optional[ControlVector] = None, **kwargs) -> StateVector:
        """
        Compute next state or derivative.

        Parameters
        ----------
        x : StateVector
            Current state
        u : Optional[ControlVector]
            Control input
        **kwargs
            Additional arguments (dt, noise, etc.)

        Returns
        -------
        StateVector
            Next state (discrete) or derivative (continuous)
        """
        ...


class StochasticProtocol(Protocol):
    """
    Protocol for stochastic systems.

    Any class with is_stochastic property and diffusion() method.

    Properties
    ----------
    is_stochastic : bool
        Whether system has stochastic dynamics

    Methods
    -------
    diffusion(x, u, **kwargs)
        Evaluate diffusion matrix

    Examples
    --------
    >>> def check_noise_structure(system: StochasticProtocol):
    ...     '''Analyze noise properties of any stochastic system.'''
    ...     if not system.is_stochastic:
    ...         return None
    ...
    ...     x_test = np.zeros(system.nx)
    ...     u_test = np.zeros(system.nu)
    ...     G = system.diffusion(x_test, u_test)
    ...
    ...     # Check if additive (constant)
    ...     G2 = system.diffusion(x_test + 1, u_test)
    ...     is_additive = np.allclose(G, G2)
    ...
    ...     return {'is_additive': is_additive, 'nw': G.shape[1]}
    """

    @property
    def is_stochastic(self) -> bool:
        """Whether system is stochastic."""
        ...

    def diffusion(self, x: StateVector, u: Optional[ControlVector] = None, **kwargs) -> ArrayLike:
        """
        Evaluate diffusion matrix.

        Parameters
        ----------
        x : StateVector
            Current state
        u : Optional[ControlVector]
            Control input
        **kwargs
            Additional arguments

        Returns
        -------
        ArrayLike
            Diffusion matrix (nx, nw)
        """
        ...


# ============================================================================
# Type Guard Functions
# ============================================================================


def is_batched(x: ArrayLike) -> bool:
    """
    Check if array is batched (has batch dimension).

    An array is considered batched if it has more than one dimension,
    where the first dimension is the batch dimension.

    Parameters
    ----------
    x : ArrayLike
        Array to check

    Returns
    -------
    bool
        True if x.ndim > 1 (has batch dimension)

    Examples
    --------
    >>> # Single state vector
    >>> x_single = np.array([1, 2, 3])
    >>> is_batched(x_single)
    False
    >>>
    >>> # Batched state vectors
    >>> x_batch = np.array([[1, 2, 3], [4, 5, 6]])
    >>> is_batched(x_batch)
    True
    >>>
    >>> # Trajectory (also batched)
    >>> x_traj = np.random.randn(100, 3)  # 100 time steps
    >>> is_batched(x_traj)
    True
    """
    if hasattr(x, "ndim"):
        return x.ndim > 1
    elif hasattr(x, "shape"):
        return len(x.shape) > 1
    return False


def get_batch_size(x: ArrayLike) -> Optional[int]:
    """
    Get batch size from batched array.

    Parameters
    ----------
    x : ArrayLike
        Array to check

    Returns
    -------
    Optional[int]
        Batch size (first dimension) if batched, None otherwise

    Examples
    --------
    >>> # Batched states
    >>> x_batch = np.random.randn(50, 3)
    >>> get_batch_size(x_batch)
    50
    >>>
    >>> # Single state
    >>> x_single = np.array([1, 2, 3])
    >>> get_batch_size(x_single)
    None
    >>>
    >>> # Trajectory
    >>> x_traj = np.random.randn(100, 5)
    >>> get_batch_size(x_traj)
    100
    """
    if is_batched(x):
        return x.shape[0]
    return None


def is_numpy(x: ArrayLike) -> bool:
    """
    Check if array is NumPy ndarray.

    Parameters
    ----------
    x : ArrayLike
        Array to check

    Returns
    -------
    bool
        True if x is np.ndarray

    Examples
    --------
    >>> x_np = np.array([1, 2, 3])
    >>> is_numpy(x_np)
    True
    >>>
    >>> import torch
    >>> x_torch = torch.tensor([1, 2, 3])
    >>> is_numpy(x_torch)
    False
    """
    return isinstance(x, np.ndarray)


def is_torch(x: ArrayLike) -> bool:
    """
    Check if array is PyTorch tensor.

    Parameters
    ----------
    x : ArrayLike
        Array to check

    Returns
    -------
    bool
        True if x is torch.Tensor

    Examples
    --------
    >>> import torch
    >>> x_torch = torch.tensor([1, 2, 3])
    >>> is_torch(x_torch)
    True
    >>>
    >>> x_np = np.array([1, 2, 3])
    >>> is_torch(x_np)
    False
    """
    try:
        import torch

        return isinstance(x, torch.Tensor)
    except ImportError:
        return False


def is_jax(x: ArrayLike) -> bool:
    """
    Check if array is JAX array.

    Parameters
    ----------
    x : ArrayLike
        Array to check

    Returns
    -------
    bool
        True if x is jax.numpy.ndarray

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> x_jax = jnp.array([1, 2, 3])
    >>> is_jax(x_jax)
    True
    >>>
    >>> x_np = np.array([1, 2, 3])
    >>> is_jax(x_np)
    False
    """
    try:
        import jax.numpy as jnp

        return isinstance(x, jnp.ndarray)
    except ImportError:
        return False


def get_backend(x: ArrayLike) -> Backend:
    """
    Detect backend from array type.

    Parameters
    ----------
    x : ArrayLike
        Array to check

    Returns
    -------
    Backend
        'numpy', 'torch', or 'jax'

    Raises
    ------
    TypeError
        If backend cannot be determined

    Examples
    --------
    >>> x_np = np.array([1, 2, 3])
    >>> get_backend(x_np)
    'numpy'
    >>>
    >>> import torch
    >>> x_torch = torch.tensor([1, 2, 3])
    >>> get_backend(x_torch)
    'torch'
    >>>
    >>> import jax.numpy as jnp
    >>> x_jax = jnp.array([1, 2, 3])
    >>> get_backend(x_jax)
    'jax'
    """
    if is_numpy(x):
        return "numpy"
    elif is_torch(x):
        return "torch"
    elif is_jax(x):
        return "jax"
    else:
        raise TypeError(f"Unknown backend for type {type(x)}")


# ============================================================================
# Type Conversion Functions
# ============================================================================


def ensure_numpy(x: ArrayLike) -> np.ndarray:
    """
    Convert to NumPy array regardless of backend.

    Handles conversion from PyTorch tensors and JAX arrays.

    Parameters
    ----------
    x : ArrayLike
        Array in any backend

    Returns
    -------
    np.ndarray
        NumPy array

    Examples
    --------
    >>> import torch
    >>> x_torch = torch.tensor([1.0, 2.0, 3.0])
    >>> x_np = ensure_numpy(x_torch)
    >>> type(x_np)
    <class 'numpy.ndarray'>
    >>>
    >>> # Already NumPy - no conversion
    >>> x_np2 = ensure_numpy(np.array([1, 2, 3]))
    >>> type(x_np2)
    <class 'numpy.ndarray'>
    """
    if isinstance(x, np.ndarray):
        return x

    # Try PyTorch conversion
    try:
        import torch

        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except ImportError:
        pass

    # Try JAX conversion
    try:
        import jax.numpy as jnp

        if isinstance(x, jnp.ndarray):
            return np.array(x)
    except ImportError:
        pass

    # Fallback to generic conversion
    return np.asarray(x)


def ensure_backend(x: ArrayLike, backend: Backend) -> ArrayLike:
    """
    Convert array to specified backend.

    Parameters
    ----------
    x : ArrayLike
        Input array
    backend : Backend
        Target backend ('numpy', 'torch', or 'jax')

    Returns
    -------
    ArrayLike
        Array in target backend

    Examples
    --------
    >>> x_np = np.array([1, 2, 3])
    >>>
    >>> # Convert to PyTorch
    >>> x_torch = ensure_backend(x_np, 'torch')
    >>> type(x_torch)
    <class 'torch.Tensor'>
    >>>
    >>> # Convert to JAX
    >>> x_jax = ensure_backend(x_np, 'jax')
    >>> type(x_jax)
    <class 'jaxlib.xla_extension.ArrayImpl'>
    """
    current_backend = get_backend(x)

    # Already correct backend
    if current_backend == backend:
        return x

    # Convert to NumPy first (common intermediate)
    x_np = ensure_numpy(x)

    # Convert to target backend
    if backend == "numpy":
        return x_np
    elif backend == "torch":
        import torch

        return torch.tensor(x_np)
    elif backend == "jax":
        import jax.numpy as jnp

        return jnp.array(x_np)
    else:
        raise ValueError(f"Unknown backend: {backend}")


# ============================================================================
# Shape Validation Functions
# ============================================================================


def check_state_shape(x: StateVector, nx: int, name: str = "state"):
    """
    Validate state vector shape.

    Checks that state has correct dimension, handles both single
    and batched states.

    Parameters
    ----------
    x : StateVector
        State to validate
    nx : int
        Expected state dimension
    name : str, optional
        Parameter name for error messages, by default "state"

    Raises
    ------
    ValueError
        If shape is incorrect

    Examples
    --------
    >>> x = np.array([1, 2, 3])
    >>> check_state_shape(x, nx=3, name="initial_state")  # OK
    >>>
    >>> # Batched state
    >>> x_batch = np.random.randn(10, 3)
    >>> check_state_shape(x_batch, nx=3)  # OK
    >>>
    >>> # Wrong dimension
    >>> x_wrong = np.array([1, 2])
    >>> check_state_shape(x_wrong, nx=3)  # ValueError
    """
    x_arr = ensure_numpy(x)

    if is_batched(x_arr):
        # Batched: (batch, nx)
        if x_arr.shape[1] != nx:
            raise ValueError(
                f"{name} has incorrect dimension. " f"Expected (..., {nx}), got shape {x_arr.shape}"
            )
    else:
        # Single: (nx,)
        if x_arr.shape[0] != nx:
            raise ValueError(
                f"{name} has incorrect dimension. " f"Expected ({nx},), got shape {x_arr.shape}"
            )


def check_control_shape(u: ControlVector, nu: int, name: str = "control"):
    """
    Validate control vector shape.

    Similar to check_state_shape but for control inputs.

    Parameters
    ----------
    u : ControlVector
        Control to validate
    nu : int
        Expected control dimension
    name : str, optional
        Parameter name for error messages, by default "control"

    Raises
    ------
    ValueError
        If shape is incorrect

    Examples
    --------
    >>> u = np.array([0.5])
    >>> check_control_shape(u, nu=1)  # OK
    >>>
    >>> # Batched control
    >>> u_batch = np.random.randn(10, 2)
    >>> check_control_shape(u_batch, nu=2)  # OK
    >>>
    >>> # Wrong dimension
    >>> u_wrong = np.array([0.5, 0.3])
    >>> check_control_shape(u_wrong, nu=1)  # ValueError
    """
    u_arr = ensure_numpy(u)

    if is_batched(u_arr):
        # Batched: (batch, nu)
        if u_arr.shape[1] != nu:
            raise ValueError(
                f"{name} has incorrect dimension. " f"Expected (..., {nu}), got shape {u_arr.shape}"
            )
    else:
        # Single: (nu,)
        if u_arr.shape[0] != nu:
            raise ValueError(
                f"{name} has incorrect dimension. " f"Expected ({nu},), got shape {u_arr.shape}"
            )


def get_array_shape(x: ArrayLike) -> Tuple[int, ...]:
    """
    Get shape of array regardless of backend.

    Parameters
    ----------
    x : ArrayLike
        Input array

    Returns
    -------
    Tuple[int, ...]
        Shape tuple

    Examples
    --------
    >>> x = np.random.randn(10, 3)
    >>> get_array_shape(x)
    (10, 3)
    >>>
    >>> import torch
    >>> x_torch = torch.randn(5, 2)
    >>> get_array_shape(x_torch)
    (5, 2)
    """
    if hasattr(x, "shape"):
        return tuple(x.shape)
    else:
        return ()


def extract_dimensions(
    x: Optional[StateVector] = None,
    u: Optional[ControlVector] = None,
    y: Optional[OutputVector] = None,
) -> SystemDimensions:
    """
    Extract system dimensions from example vectors.

    Parameters
    ----------
    x : Optional[StateVector]
        Example state vector
    u : Optional[ControlVector]
        Example control vector
    y : Optional[OutputVector]
        Example output vector

    Returns
    -------
    SystemDimensions
        Dimensions extracted from vectors

    Examples
    --------
    >>> x = np.array([1, 2, 3])
    >>> u = np.array([0.5])
    >>> y = np.array([1, 2, 3])
    >>>
    >>> dims = extract_dimensions(x, u, y)
    >>> print(dims)
    {'nx': 3, 'nu': 1, 'ny': 3, 'nw': 0}
    >>>
    >>> # Partial dimensions
    >>> dims_partial = extract_dimensions(x=x)
    >>> print(dims_partial)
    {'nx': 3, 'nu': 0, 'ny': 3, 'nw': 0}
    """
    nx = x.shape[-1] if x is not None else 0
    nu = u.shape[-1] if u is not None else 0
    ny = y.shape[-1] if y is not None else nx

    return SystemDimensions(nx=nx, nu=nu, ny=ny, nw=0)


# ============================================================================
# Array Converter Class
# ============================================================================


class ArrayConverter:
    """
    Utility class for array conversions between backends.

    Provides static methods for converting between NumPy, PyTorch, and JAX.

    Methods
    -------
    to_numpy(x)
        Convert to NumPy array
    to_torch(x, device=None)
        Convert to PyTorch tensor
    to_jax(x)
        Convert to JAX array
    convert(x, target_backend)
        Convert to specified backend

    Examples
    --------
    >>> converter = ArrayConverter()
    >>>
    >>> # NumPy to PyTorch
    >>> x_np = np.array([1, 2, 3])
    >>> x_torch = converter.to_torch(x_np)
    >>>
    >>> # PyTorch to JAX
    >>> x_jax = converter.to_jax(x_torch)
    >>>
    >>> # Generic conversion
    >>> x_converted = converter.convert(x_np, 'torch')
    """

    @staticmethod
    def to_numpy(x: ArrayLike) -> np.ndarray:
        """
        Convert to NumPy array.

        Parameters
        ----------
        x : ArrayLike
            Input array

        Returns
        -------
        np.ndarray
            NumPy array
        """
        return ensure_numpy(x)

    @staticmethod
    def to_torch(x: ArrayLike, device: Optional[str] = None) -> "torch.Tensor":
        """
        Convert to PyTorch tensor.

        Parameters
        ----------
        x : ArrayLike
            Input array
        device : Optional[str]
            Target device ('cpu', 'cuda', 'cuda:0', etc.)

        Returns
        -------
        torch.Tensor
            PyTorch tensor

        Examples
        --------
        >>> x_np = np.array([1, 2, 3])
        >>> x_torch = ArrayConverter.to_torch(x_np)
        >>>
        >>> # On GPU
        >>> x_cuda = ArrayConverter.to_torch(x_np, device='cuda')
        """
        import torch

        x_np = ensure_numpy(x)
        tensor = torch.tensor(x_np)
        if device is not None:
            tensor = tensor.to(device)
        return tensor

    @staticmethod
    def to_jax(x: ArrayLike) -> "jnp.ndarray":
        """
        Convert to JAX array.

        Parameters
        ----------
        x : ArrayLike
            Input array

        Returns
        -------
        jnp.ndarray
            JAX array

        Examples
        --------
        >>> x_np = np.array([1, 2, 3])
        >>> x_jax = ArrayConverter.to_jax(x_np)
        """
        import jax.numpy as jnp

        x_np = ensure_numpy(x)
        return jnp.array(x_np)

    @staticmethod
    def convert(x: ArrayLike, target_backend: Backend) -> ArrayLike:
        """
        Convert to specified backend.

        Parameters
        ----------
        x : ArrayLike
            Input array
        target_backend : Backend
            Target backend ('numpy', 'torch', or 'jax')

        Returns
        -------
        ArrayLike
            Array in target backend

        Examples
        --------
        >>> x_np = np.array([1, 2, 3])
        >>> x_torch = ArrayConverter.convert(x_np, 'torch')
        >>> x_jax = ArrayConverter.convert(x_torch, 'jax')
        """
        return ensure_backend(x, target_backend)


# ============================================================================
# Cache and Metadata Types
# ============================================================================

CacheKey = str
"""
Cache key for memoization.

String identifier for cached results, typically a hash or
composite of parameters.

Examples
--------
>>> key: CacheKey = "x_eq=[0.0,0.0]_u_eq=[0.0]_method=euler"
>>> cache[key] = result
>>> 
>>> # Using hash
>>> import hashlib
>>> key: CacheKey = hashlib.md5(str(params).encode()).hexdigest()
"""


class CacheStatistics(TypedDict):
    """
    Cache performance statistics.

    Tracks cache hits, misses, and efficiency metrics.

    Attributes
    ----------
    computes : int
        Number of actual computations performed
    cache_hits : int
        Number of times result retrieved from cache
    cache_misses : int
        Number of times computation was required
    total_requests : int
        Total requests (hits + misses)
    cache_size : int
        Number of items currently cached
    hit_rate : float
        Cache hit rate (hits / total_requests)

    Examples
    --------
    >>> stats: CacheStatistics = linearizer.get_cache_stats()
    >>> print(f"Hit rate: {stats['hit_rate']:.1%}")
    >>> print(f"Cache size: {stats['cache_size']}")
    >>>
    >>> # Monitor efficiency
    >>> if stats['hit_rate'] < 0.5:
    ...     print("Warning: Low cache efficiency")
    """

    computes: int
    cache_hits: int
    cache_misses: int
    total_requests: int
    cache_size: int
    hit_rate: float


Metadata = Dict[str, Any]
"""
General metadata dictionary.

Flexible dictionary for storing arbitrary metadata about
computations, systems, or results.

Examples
--------
>>> metadata: Metadata = {
...     'timestamp': '2025-01-01T00:00:00',
...     'version': '1.0.0',
...     'author': 'user',
...     'git_commit': 'abc123',
...     'parameters': {'dt': 0.01, 'method': 'RK45'}
... }
>>> 
>>> # Attach to results
>>> result_with_meta = {
...     'data': computation_result,
...     'metadata': metadata
... }
"""


# ============================================================================
# Validation and Performance Types
# ============================================================================


class ExecutionStats(TypedDict):
    """Execution statistics for tracking function performance.

    Tracks runtime performance of any callable component:
    - Function evaluation time
    - Call frequency
    - Average execution time

    This is distinct from PerformanceMetrics which measures
    control system performance (settling time, overshoot, etc.).
    """

    calls: int
    total_time: float
    avg_time: float


class ValidationResult(TypedDict, total=False):
    """
    System validation result.

    Result from validating system properties, configurations,
    or data integrity.

    Attributes
    ----------
    valid : bool
        Overall validation status
    errors : List[str]
        List of error messages
    warnings : List[str]
        List of warning messages
    checks_passed : int
        Number of validation checks passed
    checks_total : int
        Total number of validation checks

    Examples
    --------
    >>> result: ValidationResult = validator.validate(system)
    >>>
    >>> if result['valid']:
    ...     print("System is valid")
    ... else:
    ...     print("Errors found:")
    ...     for error in result['errors']:
    ...         print(f"  - {error}")
    >>>
    >>> # Check coverage
    >>> coverage = result['checks_passed'] / result['checks_total']
    >>> print(f"Validation coverage: {coverage:.1%}")
    """

    valid: bool
    errors: List[str]
    warnings: List[str]
    checks_passed: int
    checks_total: int


@dataclass(frozen=True)
class SymbolicValidationResult:
    """
    Container for symbolic system validation results.

    Attributes
    ----------
    is_valid : bool
        True if system passed all validation checks
    errors : List[str]
        List of validation errors (empty if valid)
    warnings : List[str]
        List of validation warnings (non-fatal issues)
    info : Dict
        Additional information about the validated system
    """

    is_valid: bool
    errors: List[str]
    warnings: List[str]
    info: Dict


class PerformanceMetrics(TypedDict, total=False):
    """
    Performance metrics for simulation/control.

    Standard performance metrics for analyzing system behavior.

    Attributes
    ----------
    settling_time : float
        Time to settle within tolerance
    rise_time : float
        10% to 90% rise time
    overshoot : float
        Maximum overshoot (%)
    steady_state_error : float
        Steady-state tracking error
    control_effort : float
        Integral of |u|²
    trajectory_cost : float
        Integral of cost function

    Examples
    --------
    >>> metrics: PerformanceMetrics = analyze_performance(trajectory)
    >>>
    >>> print(f"Settling time: {metrics['settling_time']:.2f}s")
    >>> print(f"Overshoot: {metrics['overshoot']:.1f}%")
    >>> print(f"Steady-state error: {metrics['steady_state_error']:.3f}")
    >>>
    >>> # Check performance requirements
    >>> if metrics['settling_time'] < 2.0 and metrics['overshoot'] < 10:
    ...     print("Performance requirements met")
    """

    settling_time: float
    rise_time: float
    overshoot: float
    steady_state_error: float
    control_effort: float
    trajectory_cost: float


# ============================================================================
# Export All
# ============================================================================

__all__ = [
    # Protocols
    "LinearizableProtocol",
    "SimulatableProtocol",
    "StochasticProtocol",
    # Type guards
    "is_batched",
    "get_batch_size",
    "is_numpy",
    "is_torch",
    "is_jax",
    "get_backend",
    # Converters
    "ensure_numpy",
    "ensure_backend",
    "ArrayConverter",
    # Validators
    "check_state_shape",
    "check_control_shape",
    "get_array_shape",
    "extract_dimensions",
    # Cache and metadata
    "CacheKey",
    "CacheStatistics",
    "Metadata",
    # Validation and performance
    "ExecutionStats",
    "ValidationResult",
    "SymbolicValidationResult",
    "PerformanceMetrics",
]
