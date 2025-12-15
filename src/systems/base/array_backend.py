"""
Backend-agnostic array operations.

This module provides a unified interface for array operations across
NumPy, PyTorch, JAX, and other array libraries.
"""

from abc import ABC, abstractmethod
from typing import TypeVar, Generic, List, Optional
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


class TorchArrayBackend(ArrayBackend):
    """PyTorch implementation"""

    def __init__(self, device: str = "cpu", dtype=None):
        import torch

        self.torch = torch
        self.device = device
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


class NumpyArrayBackend(ArrayBackend):
    """NumPy implementation"""

    def __init__(self, dtype=np.float32):
        self.dtype = dtype

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


class JAXArrayBackend(ArrayBackend):
    """JAX implementation"""

    def __init__(self, device="cpu", dtype=None):
        import jax
        import jax.numpy as jnp

        self.jax = jax
        self.jnp = jnp
        self.device = device
        self.dtype = dtype or jnp.float32

    def zeros(self, shape, dtype=None):
        arr = self.jnp.zeros(shape, dtype=dtype or self.dtype)

        # Put on specified device
        if self.device != "cpu":
            arr = jax.device_put(arr, jax.devices(self.device)[0])

        return arr

    def ones(self, shape: tuple, dtype=None):
        return self.jnp.ones(shape, dtype=dtype or self.dtype)

    def eye(self, n: int, dtype=None):
        return self.jnp.eye(n, dtype=dtype or self.dtype)

    def concatenate(self, arrays, axis: int):
        return self.jnp.concatenate(arrays, axis=axis)

    def stack(self, arrays, axis: int):
        return self.jnp.stack(arrays, axis=axis)

    def expand_dims(self, x, axis: int):
        return self.jnp.expand_dims(x, axis=axis)

    def squeeze(self, x, axis: Optional[int] = None):
        if axis is None:
            return self.jnp.squeeze(x)
        return self.jnp.squeeze(x, axis=axis)

    def reshape(self, x, shape: tuple):
        return self.jnp.reshape(x, shape)

    def to_numpy(self, x) -> np.ndarray:
        return np.array(x)

    def from_numpy(self, x: np.ndarray):
        return self.jnp.array(x, dtype=self.dtype)

    def get_backend_name(self) -> str:
        return "jax"

    def get_default_dtype(self):
        return self.dtype
