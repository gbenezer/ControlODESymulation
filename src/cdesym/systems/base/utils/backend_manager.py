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
Backend Manager for Multi-Backend Array Handling

Handles:
- Backend detection from array types
- Array conversion between backends
- Backend availability checking
- Device management
- Default backend configuration

This class is completely standalone and can be reused by any class
that needs multi-backend array handling.
"""

from contextlib import contextmanager
from typing import TYPE_CHECKING, List, Optional

import numpy as np

from cdesym.types.backends import (
    DEFAULT_BACKEND,
    DEFAULT_DEVICE,
    Backend,
    BackendConfig,
    Device,
    validate_backend,
    validate_device,
)

# Import from centralized type system
from cdesym.types.core import ArrayLike
from cdesym.types.utilities import is_jax, is_numpy, is_torch

if TYPE_CHECKING:
    pass


class BackendManager:
    """
    Manages backend detection, conversion, and device placement.

    Supports NumPy, PyTorch, and JAX backends with automatic detection
    and conversion between them.

    Example:
        >>> mgr = BackendManager()
        >>> mgr.set_default('torch', device='cuda')
        >>>
        >>> # Auto-detect backend
        >>> x = torch.tensor([1.0])
        >>> backend = mgr.detect(x)  # Returns 'torch'
        >>>
        >>> # Convert between backends
        >>> x_jax = mgr.convert(x, 'jax')
        >>>
        >>> # Temporary backend switching
        >>> with mgr.use_backend('numpy'):
        ...     # Operations use NumPy
        ...     pass
    """

    def __init__(
        self,
        default_backend: Backend = DEFAULT_BACKEND,
        default_device: Device = DEFAULT_DEVICE,
    ):
        """
        Initialize backend manager.

        Args:
            default_backend: Default backend to use
            default_device: Default device for GPU backends
        """
        # Validate and store configuration
        self._default_backend: Backend = validate_backend(default_backend)
        self._preferred_device: Device = default_device

        # Detect available backends at initialization
        self._available_backends = self._detect_available_backends()

        # Validate default backend is available
        if default_backend not in self._available_backends:
            raise RuntimeError(
                f"Default backend '{default_backend}' is not available. "
                f"Available backends: {self._available_backends}",
            )

    # ========================================================================
    # Properties
    # ========================================================================

    @property
    def default_backend(self) -> Backend:
        """Get current default backend"""
        return self._default_backend

    @property
    def preferred_device(self) -> Device:
        """Get current preferred device"""
        return self._preferred_device

    @property
    def available_backends(self) -> List[Backend]:
        """Get list of available backends"""
        return self._available_backends.copy()

    # ========================================================================
    # Backend Detection
    # ========================================================================

    def _detect_available_backends(self) -> List[Backend]:
        """
        Detect which backends are available in the current environment.

        Returns:
            List of available backend names
        """
        available: List[Backend] = ["numpy"]  # NumPy is always available

        # Check PyTorch
        try:
            import torch

            available.append("torch")
        except ImportError:
            pass

        # Check JAX
        try:
            import jax

            available.append("jax")
        except ImportError:
            pass

        return available

    def detect(self, array: ArrayLike) -> Backend:
        """
        Detect backend from array type.

        Uses centralized type guards from utilities module for consistent
        backend detection across the framework.

        Args:
            array: Input array/tensor

        Returns:
            Backend identifier ('numpy', 'torch', or 'jax')

        Raises:
            TypeError: If array type is not recognized

        Example:
            >>> mgr = BackendManager()
            >>> x = np.array([1.0])
            >>> mgr.detect(x)  # Returns 'numpy'
            >>>
            >>> import torch
            >>> x_torch = torch.tensor([1.0])
            >>> mgr.detect(x_torch)  # Returns 'torch'
        """
        if is_torch(array):
            return "torch"
        if is_jax(array):
            return "jax"
        if is_numpy(array):
            return "numpy"
        raise TypeError(
            f"Unknown input type: {type(array)}. "
            f"Expected np.ndarray, torch.Tensor, or jax.numpy.ndarray",
        )

    def check_available(self, backend: Backend) -> bool:
        """
        Check if a backend is available.

        Args:
            backend: Backend name to check

        Returns:
            True if backend is available, False otherwise
        """
        return backend in self._available_backends

    def require_backend(self, backend: Backend):
        """
        Raise error if backend is not available.

        Args:
            backend: Backend name to check

        Raises:
            RuntimeError: If backend is not available

        Example:
            >>> mgr = BackendManager()
            >>> mgr.require_backend('torch')  # Raises if PyTorch not installed
        """
        if not self.check_available(backend):
            # Generate helpful error message
            if backend == "torch":
                msg = "PyTorch backend not available. Install with: pip install torch"
            elif backend == "jax":
                msg = "JAX backend not available. Install with: pip install jax jaxlib"
            else:
                msg = f"Backend '{backend}' not available"

            raise RuntimeError(msg)

    def ensure_type(self, arr: ArrayLike, backend: Optional[Backend] = None) -> ArrayLike:
        """
        Ensure array is in specified backend type.

        Converts array only if it's not already the correct type.
        Less aggressive than convert() - preserves existing type if compatible.

        Args:
            arr: Array to check/convert
            backend: Target backend (None = use default_backend)

        Returns:
            Array in correct backend type

        Example:
            >>> mgr = BackendManager()
            >>> x_np = np.array([1.0])
            >>> x_ensured = mgr.ensure_type(x_np, 'numpy')
            >>> assert x_ensured is x_np  # Same object (no conversion)
        """
        backend = backend or self.default_backend

        if backend == "numpy":
            if not isinstance(arr, np.ndarray):
                return np.asarray(arr)
            return arr

        if backend == "torch":
            import torch

            if not isinstance(arr, torch.Tensor):
                if isinstance(arr, np.ndarray):
                    dtype = torch.float64 if arr.dtype == np.float64 else torch.float32
                    return torch.tensor(arr, dtype=dtype, device=self.preferred_device)
                return torch.tensor(arr, dtype=torch.float64, device=self.preferred_device)
            # Ensure correct device
            return arr.to(self.preferred_device)

        if backend == "jax":
            import jax.numpy as jnp

            if not isinstance(arr, jnp.ndarray):
                return jnp.asarray(arr)
            return arr

    # ========================================================================
    # Array Conversion
    # ========================================================================

    def convert(
        self,
        array: ArrayLike,
        target_backend: Optional[Backend] = None,
        device: Optional[Device] = None,
    ) -> ArrayLike:
        """
        Convert array to target backend with optional device placement.
        
        This is the primary conversion method that combines backend conversion
        and device placement in a single operation. This method allows explicit device control.
        
        Args:
            array: Source array (numpy.ndarray, torch.Tensor, or jax.numpy.ndarray)
            target_backend: Target backend (None = use default_backend)
            device: Target device (None = use preferred_device)
        
        Returns:
            Array in target backend format on specified device
        
        Raises:
            RuntimeError: If target backend is not available
            ValueError: If device is incompatible with target backend
            TypeError: If array type is not recognized
        
        Example:
            >>> mgr = BackendManager()
            >>> x_np = np.array([1.0, 2.0, 3.0])
            >>> 
            >>> # Convert to PyTorch on CPU
            >>> x_torch = mgr.to_backend(x_np, 'torch', device='cpu')
            >>> 
            >>> # Convert to JAX (uses default device)
            >>> x_jax = mgr.to_backend(x_np, 'jax')
            >>> 
            >>> # Convert to default backend
            >>> x_default = mgr.to_backend(x_jax)  # Uses default_backend
        
        Note:
            - If target_backend is None, uses self.default_backend
            - If device is None, uses self.preferred_device
            - For numpy backend, device parameter is ignored (always CPU)
            - Performs no-op if array is already in target backend on correct device
        """
        
        import warnings
    
        # Resolve target backend
        target_backend = target_backend or self.default_backend
        target_backend = validate_backend(target_backend)
        
        # Resolve target device
        target_device = device or self.preferred_device
        target_device = validate_device(target_device, target_backend)
        
        # Check target backend is available
        self.require_backend(target_backend)
        
        # Detect source backend
        source_backend = self.detect(array)
        
        # Fast path: already correct backend and device
        if source_backend == target_backend:
            if target_backend == "numpy":
                return array  # NumPy always on CPU
            elif target_backend == "torch":
                import torch
                if isinstance(array, torch.Tensor):
                    # Check if already on correct device
                    current_device = str(array.device)
                    if current_device == target_device or (
                        target_device == "cpu" and current_device == "cpu"
                    ):
                        return array
                    # Move to target device
                    return array.to(target_device)
            elif target_backend == "jax":
                # JAX device handling is more complex, safer to recreate
                pass
        
        # Convert to NumPy as intermediate representation
        if source_backend == "numpy":
            array_np = array
        elif source_backend == "torch":
            array_np = array.detach().cpu().numpy()
        elif source_backend == "jax":
            array_np = np.array(array)
        else:
            raise TypeError(
                f"Unknown source array type: {type(array)}. "
                f"Expected np.ndarray, torch.Tensor, or jax.numpy.ndarray"
            )
        
        # Convert from NumPy to target backend
        if target_backend == "numpy":
            return array_np
        
        elif target_backend == "torch":
            import torch
            
            # Preserve dtype precision
            dtype = torch.float64 if array_np.dtype == np.float64 else torch.float32
            
            # Create tensor on target device
            tensor = torch.tensor(array_np, dtype=dtype, device=target_device)
            return tensor
        
        elif target_backend == "jax":
            import jax.numpy as jnp
            from jax import device_put, devices
            
            # Create JAX array
            array_jax = jnp.array(array_np)
            
            # Handle device placement for JAX
            if target_device != "cpu":
                try:
                    # Parse device string (e.g., 'gpu:0' -> device 0)
                    if target_device.startswith("gpu:"):
                        device_idx = int(target_device.split(":")[1])
                        jax_devices = devices("gpu")
                    elif target_device.startswith("cuda:"):
                        device_idx = int(target_device.split(":")[1])
                        jax_devices = devices("gpu")
                    elif target_device in ("gpu", "cuda"):
                        device_idx = 0
                        jax_devices = devices("gpu")
                    else:
                        # Unknown device format, stay on default
                        warnings.warn(
                            f"Unknown JAX device format '{target_device}'. "
                            f"Expected format: 'cpu', 'gpu', 'cuda', 'gpu:N', or 'cuda:N'. "
                            f"Array will be placed on default JAX device.",
                            UserWarning,
                            stacklevel=2,
                        )
                        return array_jax
                    
                    if device_idx < len(jax_devices):
                        array_jax = device_put(array_jax, jax_devices[device_idx])
                except (IndexError, RuntimeError, ValueError):
                    # If device placement fails, return on default device
                    pass
            
            return array_jax
        else:
            raise RuntimeError(f"Unhandled target backend: {target_backend}")

    # ========================================================================
    # Configuration
    # ========================================================================

    def set_default(
        self,
        backend: Backend,
        device: Optional[Device] = None,
    ) -> "BackendManager":
        """
        Set default backend and optionally device.

        Args:
            backend: Backend name
            device: Device name (if None, device is not changed)

        Returns:
            Self for method chaining

        Raises:
            ValueError: If backend name is invalid or device incompatible
            RuntimeError: If backend is not available
        """
        # Validate backend
        backend = validate_backend(backend)

        # Check availability
        self.require_backend(backend)

        # Set backend
        self._default_backend = backend

        # Validate current device against new backend
        # This ensures cuda/mps devices are only used with compatible backends
        validate_device(self._preferred_device, backend)

        # Set device if provided
        if device is not None:
            device = validate_device(device, backend)
            self._preferred_device = device

        return self

    def to_device(self, device: Device) -> "BackendManager":
        """
        Set preferred device for GPU-capable backends.

        Args:
            device: Device string ('cpu', 'cuda', 'cuda:0', 'gpu:0', etc.)

        Returns:
            Self for method chaining

        Note:
            Device is stored as a preference. Actual validation happens when
            backend is set or arrays are converted. This allows setting a
            preferred GPU device before switching to a GPU-capable backend.

        Example:
            >>> mgr = BackendManager()
            >>> mgr.to_device('cuda:0')  # Store preference
            >>> mgr.set_default('torch')  # Now validates cuda:0 for torch
        """
        # Just store the preference - validate when actually used
        self._preferred_device = device
        return self

    def reset(self):
        """
        Reset to default configuration (NumPy backend, CPU device).

        Example:
            >>> mgr = BackendManager()
            >>> mgr.set_default('torch', device='cuda')
            >>> mgr.reset()
            >>> mgr.default_backend  # Returns 'numpy'
            >>> mgr.preferred_device  # Returns 'cpu'
        """
        self._default_backend = DEFAULT_BACKEND
        self._preferred_device = DEFAULT_DEVICE

    # ========================================================================
    # Context Managers
    # ========================================================================

    @contextmanager
    def use_backend(
        self,
        backend: Backend,
        device: Optional[Device] = None,
    ):
        """
        Temporarily switch to a different backend and/or device.

        Args:
            backend: Temporary backend to use
            device: Optional temporary device

        Yields:
            Self with temporary configuration

        Example:
            >>> mgr = BackendManager(default_backend='numpy')
            >>> with mgr.use_backend('torch', device='cuda'):
            ...     # Code here uses torch backend on CUDA
            ...     x = mgr.convert(x_np, mgr.default_backend)
            >>> # Back to NumPy after context
            >>> mgr.default_backend  # Returns 'numpy'
        """
        # Save current state
        old_backend = self._default_backend
        old_device = self._preferred_device

        try:
            # Set temporary configuration
            self.set_default(backend, device)
            yield self
        finally:
            # Restore original state
            self._default_backend = old_backend
            self._preferred_device = old_device

    # ========================================================================
    # Information & Debugging
    # ========================================================================

    def get_info(self) -> BackendConfig:
        """
        Get backend configuration.

        Returns:
            Structured backend configuration

        Example:
            >>> mgr = BackendManager()
            >>> config = mgr.get_info()
            >>> print(config['backend'])
            'numpy'
        """
        config: BackendConfig = {
            "backend": self._default_backend,
            "device": self._preferred_device,
            "dtype": "float64",
        }
        return config

    # TODO: TypepdDict?
    def get_extended_info(self) -> dict:
        """
        Get extended backend information including versions.

        Returns:
            Dictionary with backend configuration and metadata

        Example:
            >>> mgr = BackendManager()
            >>> info = mgr.get_extended_info()
            >>> print(info['available_backends'])
            ['numpy', 'torch', 'jax']
        """
        info = {
            "default_backend": self._default_backend,
            "preferred_device": self._preferred_device,
            "available_backends": self.available_backends,
            "torch_available": self.check_available("torch"),
            "jax_available": self.check_available("jax"),
            "numpy_version": np.__version__,
        }

        if self.check_available("torch"):
            import torch

            info["torch_version"] = torch.__version__
        else:
            info["torch_version"] = None

        if self.check_available("jax"):
            import jax

            info["jax_version"] = jax.__version__
        else:
            info["jax_version"] = None

        return info

    def __repr__(self) -> str:
        """String representation for debugging"""
        return (
            f"BackendManager("
            f"default='{self._default_backend}', "
            f"device='{self._preferred_device}', "
            f"available={self.available_backends})"
        )

    def __str__(self) -> str:
        """Human-readable string"""
        return (
            f"BackendManager(default='{self._default_backend}', device='{self._preferred_device}')"
        )
