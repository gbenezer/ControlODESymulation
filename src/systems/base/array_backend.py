"""
Backend-agnostic array operations.

This module provides a unified interface for array operations across
NumPy, PyTorch, JAX, and other array libraries.
"""

from abc import ABC, abstractmethod
from typing import TypeVar, Generic, List, Optional, Any
import numpy as np

ArrayType = TypeVar("ArrayType")


class ArrayBackend(ABC, Generic[ArrayType]):
    """Abstract interface for array operations"""

    @abstractmethod
    def zeros(self, shape: tuple, dtype=None) -> ArrayType:
        """Create array of zeros"""
        pass

    @abstractmethod
    def ones(self, shape: tuple, dtype=None) -> ArrayType:
        """Create array of ones"""
        pass

    @abstractmethod
    def eye(self, n: int, dtype=None) -> ArrayType:
        """Create identity matrix"""
        pass

    @abstractmethod
    def concatenate(self, arrays: List[ArrayType], axis: int) -> ArrayType:
        """Concatenate arrays along axis"""
        pass

    @abstractmethod
    def stack(self, arrays: List[ArrayType], axis: int) -> ArrayType:
        """Stack arrays along new axis"""
        pass

    @abstractmethod
    def expand_dims(self, x: ArrayType, axis: int) -> ArrayType:
        """Add dimension at axis"""
        pass

    @abstractmethod
    def squeeze(self, x: ArrayType, axis: Optional[int] = None) -> ArrayType:
        """Remove dimensions of size 1"""
        pass

    @abstractmethod
    def reshape(self, x: ArrayType, shape: tuple) -> ArrayType:
        """Reshape array"""
        pass

    @abstractmethod
    def to_numpy(self, x: ArrayType) -> np.ndarray:
        """Convert to numpy for serialization/interop"""
        pass

    @abstractmethod
    def from_numpy(self, x: np.ndarray) -> ArrayType:
        """Convert from numpy"""
        pass

    @abstractmethod
    def get_backend_name(self) -> str:
        """Get backend name (e.g., 'pytorch', 'numpy', 'jax')"""
        pass

    @abstractmethod
    def get_default_dtype(self):
        """Get default dtype for this backend"""
        pass

    def get_device(self) -> Optional[Any]:
        """
        Get device object (backend-specific).

        Returns:
            Device object or None if not applicable
        """
        return None

    def get_device_str(self) -> str:
        """
        Get device string.

        Returns:
            Device string ('cpu', 'cuda', 'gpu', etc.)
        """
        return "cpu"  # Default to CPU


class TorchArrayBackend(ArrayBackend):
    """
    PyTorch array operations with GPU support.

    Supports CPU, CUDA GPUs, and Apple Silicon (MPS).
    PyTorch arrays support automatic differentiation and can be
    moved between devices dynamically.

    Examples:
        >>> # CPU backend
        >>> backend = TorchArrayBackend(device='cpu')
        >>> arr = backend.ones((3, 3))
        >>>
        >>> # CUDA GPU
        >>> backend = TorchArrayBackend(device='cuda:0')
        >>> arr = backend.zeros((100, 100))  # On GPU
    """

    def __init__(self, device: str = "cpu", dtype=None):
        import torch

        self.torch = torch
        self.device_str = device
        self.device = torch.device(device)
        self.dtype = dtype or torch.float32

    def zeros(self, shape: tuple, dtype=None):
        return self.torch.zeros(shape, device=self.device, dtype=dtype or self.dtype)

    def ones(self, shape: tuple, dtype=None):
        return self.torch.ones(shape, device=self.device, dtype=dtype or self.dtype)

    def eye(self, n: int, dtype=None):
        return self.torch.eye(n, device=self.device, dtype=dtype or self.dtype)

    def concatenate(self, arrays, axis: int):
        return self.torch.cat(arrays, dim=axis)

    def stack(self, arrays, axis: int):
        return self.torch.stack(arrays, dim=axis)

    def expand_dims(self, x, axis: int):
        return x.unsqueeze(axis)

    def squeeze(self, x, axis: Optional[int] = None):
        if axis is None:
            return x.squeeze()
        return x.squeeze(axis)

    def reshape(self, x, shape: tuple):
        return x.reshape(shape)

    def to_numpy(self, x) -> np.ndarray:
        return x.detach().cpu().numpy()

    def from_numpy(self, x: np.ndarray):
        return self.torch.from_numpy(x).to(self.device, self.dtype)

    def get_backend_name(self) -> str:
        return "pytorch"

    def get_default_dtype(self):
        return self.dtype

    def get_device(self):
        """
        Get the PyTorch device object.

        Returns:
            torch.device object
        """
        return self.device

    def get_device_str(self) -> str:
        """
        Get the device string.

        Returns:
            Device string like 'cpu', 'cuda', 'cuda:0', 'mps'
        """
        return self.device_str


class NumpyArrayBackend(ArrayBackend):
    """
    NumPy array operations (CPU-only).

    NumPy is the foundational array library for Python scientific computing.
    Operations leverage optimized BLAS/LAPACK libraries for linear algebra.

    Examples:
        >>> # NumPy backend (always CPU)
        >>> backend = NumpyArrayBackend(dtype=np.float32)
        >>> arr = backend.ones((3, 3))
        >>>
        >>> # High precision
        >>> backend = NumpyArrayBackend(dtype=np.float64)
        >>> arr = backend.eye(10)
    """

    def __init__(self, dtype=np.float32):
        self.dtype = dtype
        self.device_str = "cpu"  # ADD: Always CPU for NumPy

    def zeros(self, shape: tuple, dtype=None):
        return np.zeros(shape, dtype=dtype or self.dtype)

    def ones(self, shape: tuple, dtype=None):
        return np.ones(shape, dtype=dtype or self.dtype)

    def eye(self, n: int, dtype=None):
        return np.eye(n, dtype=dtype or self.dtype)

    def concatenate(self, arrays, axis: int):
        return np.concatenate(arrays, axis=axis)

    def stack(self, arrays, axis: int):
        return np.stack(arrays, axis=axis)

    def expand_dims(self, x, axis: int):
        return np.expand_dims(x, axis=axis)

    def squeeze(self, x, axis: Optional[int] = None):
        if axis is None:
            return np.squeeze(x)
        return np.squeeze(x, axis=axis)

    def reshape(self, x, shape: tuple):
        return x.reshape(shape)

    def to_numpy(self, x) -> np.ndarray:
        return x

    def from_numpy(self, x: np.ndarray):
        return x.astype(self.dtype)

    def get_backend_name(self) -> str:
        return "numpy"

    def get_default_dtype(self):
        return self.dtype

    def get_device(self):
        """
        Get device object (always None for NumPy).

        Returns:
            None (NumPy doesn't have device objects)
        """
        return None

    def get_device_str(self) -> str:
        """
        Get device string (always 'cpu' for NumPy).

        Returns:
            Always 'cpu'
        """
        return "cpu"


class JAXArrayBackend(ArrayBackend):
    """
    JAX array operations with GPU support.

    Supports CPU, single GPU, and multi-GPU configurations.

    Examples:
        >>> # CPU backend
        >>> backend = JAXArrayBackend(device='cpu')
        >>>
        >>> # Default GPU
        >>> backend = JAXArrayBackend(device='gpu')
        >>>
        >>> # Specific GPU
        >>> backend = JAXArrayBackend(device='gpu:0')
    """

    def __init__(self, device: str = "cpu", dtype=None):
        """
        Initialize JAX array backend.

        Args:
            device: Device string - 'cpu', 'gpu', 'gpu:0', 'gpu:1', etc.
            dtype: JAX dtype (default: jnp.float32)

        Raises:
            ImportError: If JAX is not installed
            ValueError: If device string is invalid
            RuntimeError: If requested GPU is not available
        """
        try:
            import jax
            import jax.numpy as jnp

            # Store jax module reference (CRITICAL for tests!)
            self.jax = jax
            self.jnp = jnp

            # Store device string
            self.device_str = device

            # Parse and validate device
            if device == "cpu":
                self.device = jax.devices("cpu")[0]
            elif device == "gpu":
                # Default GPU (first available)
                gpu_devices = jax.devices("gpu")
                if len(gpu_devices) == 0:
                    raise RuntimeError("No GPU devices available")
                self.device = gpu_devices[0]
            elif device.startswith("gpu:"):
                # Specific GPU
                gpu_id = int(device.split(":")[1])
                gpu_devices = jax.devices("gpu")
                if gpu_id >= len(gpu_devices):
                    raise RuntimeError(
                        f"GPU {gpu_id} requested but only {len(gpu_devices)} available"
                    )
                self.device = gpu_devices[gpu_id]
            else:
                raise ValueError(
                    f"Invalid device string: {device}. " f"Use 'cpu', 'gpu', or 'gpu:N'"
                )

            self.dtype = dtype or jnp.float32

        except ImportError:
            raise ImportError(
                "JAX not installed. Install with:\n"
                "  CPU: pip install jax jaxlib\n"
                "  GPU: pip install jax[cuda12] (or cuda11)"
            )

    def _put_on_device(self, arr):
        """Helper to put array on correct device"""
        if self.device_str == "cpu":
            return arr  # Already on CPU
        else:
            return self.jax.device_put(arr, self.device)

    def zeros(self, shape: tuple, dtype=None):
        arr = self.jnp.zeros(shape, dtype=dtype or self.dtype)
        return self._put_on_device(arr)

    def ones(self, shape: tuple, dtype=None):
        arr = self.jnp.ones(shape, dtype=dtype or self.dtype)
        return self._put_on_device(arr)

    def eye(self, n: int, dtype=None):
        arr = self.jnp.eye(n, dtype=dtype or self.dtype)
        return self._put_on_device(arr)

    def concatenate(self, arrays, axis: int):
        """
        Concatenate arrays along axis.

        Note: JAX raises TypeError for shape mismatches (not ValueError)
        """
        return self.jnp.concatenate(arrays, axis=axis)

    def stack(self, arrays, axis: int):
        return self.jnp.stack(arrays, axis=axis)

    def expand_dims(self, x, axis: int):
        return self.jnp.expand_dims(x, axis=axis)

    def squeeze(self, x, axis=None):
        if axis is None:
            return self.jnp.squeeze(x)
        return self.jnp.squeeze(x, axis=axis)

    def reshape(self, x, shape: tuple):
        """
        Reshape array.

        Note: JAX raises TypeError for invalid shapes (not ValueError)
        """
        return self.jnp.reshape(x, shape)

    def to_numpy(self, x) -> np.ndarray:
        """
        Convert JAX array to NumPy.

        Works for both CPU and GPU arrays (automatically transfers from GPU).
        """
        return np.array(x)

    def from_numpy(self, x: np.ndarray):
        arr = self.jnp.array(x, dtype=self.dtype)
        return self._put_on_device(arr)

    def get_backend_name(self) -> str:
        return "jax"

    def get_default_dtype(self):
        return self.dtype

    def get_device(self):
        """
        Get the JAX device object.

        Returns:
            JAX Device object
        """
        return self.device

    def get_device_str(self) -> str:
        """
        Get the device string.

        Returns:
            Device string like 'cpu', 'gpu', 'gpu:0'
        """
        return self.device_str
