"""
Backend Manager for SymbolicDynamicalSystem

Handles:
- Backend detection from array types
- Array conversion between backends
- Backend availability checking
- Device management
- Default backend configuration

This class is completely standalone and can be reused by any class
that needs multi-backend array handling.
"""

from typing import Union, Optional, List, Dict
from contextlib import contextmanager
import numpy as np

# Type checking imports
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import torch
    import jax.numpy as jnp

# Type alias for backend-agnostic arrays
ArrayLike = Union[np.ndarray, "torch.Tensor", "jnp.ndarray"]


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
        >>>     # Operations use NumPy
        >>>     pass
    """
    
    def __init__(self, default_backend: str = 'numpy', default_device: str = 'cpu'):
        """
        Initialize backend manager.
        
        Args:
            default_backend: Default backend to use ('numpy', 'torch', 'jax')
            default_device: Default device for GPU backends ('cpu', 'cuda', etc.)
        """
        self._default_backend = default_backend
        self._preferred_device = default_device
        
        # Detect available backends at initialization
        self._available_backends = self._detect_available_backends()
        
        # Validate default backend is available
        if default_backend not in self._available_backends:
            raise RuntimeError(
                f"Default backend '{default_backend}' is not available. "
                f"Available backends: {self._available_backends}"
            )
    
    # ========================================================================
    # Properties
    # ========================================================================
    
    @property
    def default_backend(self) -> str:
        """Get current default backend"""
        return self._default_backend
    
    @property
    def preferred_device(self) -> str:
        """Get current preferred device"""
        return self._preferred_device
    
    @property
    def available_backends(self) -> List[str]:
        """Get list of available backends"""
        return self._available_backends.copy()
    
    # ========================================================================
    # Backend Detection
    # ========================================================================
    
    def _detect_available_backends(self) -> List[str]:
        """
        Detect which backends are available in the current environment.
        
        Returns:
            List of available backend names
        """
        available = ['numpy']  # NumPy is always available
        
        # Check PyTorch
        try:
            import torch
            available.append('torch')
        except ImportError:
            pass
        
        # Check JAX
        try:
            import jax
            available.append('jax')
        except ImportError:
            pass
        
        return available
    
    def detect(self, array: ArrayLike) -> str:
        """
        Detect backend from array type.
        
        Args:
            array: Input array/tensor
            
        Returns:
            Backend name ('numpy', 'torch', or 'jax')
            
        Raises:
            TypeError: If array type is not recognized
            
        Example:
            >>> mgr = BackendManager()
            >>> x = np.array([1.0])
            >>> mgr.detect(x)  # Returns 'numpy'
        """
        # Check PyTorch
        try:
            import torch
            if isinstance(array, torch.Tensor):
                return 'torch'
        except ImportError:
            pass
        
        # Check JAX
        try:
            import jax.numpy as jnp
            if isinstance(array, jnp.ndarray):
                return 'jax'
        except ImportError:
            pass
        
        # Check NumPy
        if isinstance(array, np.ndarray):
            return 'numpy'
        
        # Unknown type
        raise TypeError(
            f"Unknown input type: {type(array)}. "
            f"Expected np.ndarray, torch.Tensor, or jax.numpy.ndarray"
        )
    
    def check_available(self, backend: str) -> bool:
        """
        Check if a backend is available.
        
        Args:
            backend: Backend name to check
            
        Returns:
            True if backend is available, False otherwise
        """
        return backend in self._available_backends
    
    def require_backend(self, backend: str):
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
            if backend == 'torch':
                msg = (
                    "PyTorch backend not available. "
                    "Install with: pip install torch"
                )
            elif backend == 'jax':
                msg = (
                    "JAX backend not available. "
                    "Install with: pip install jax jaxlib"
                )
            else:
                msg = f"Backend '{backend}' not available"
            
            raise RuntimeError(msg)
    
    # ========================================================================
    # Backend Conversion
    # ========================================================================
    
    def convert(self, array: ArrayLike, target_backend: str) -> ArrayLike:
        """
        Convert array to target backend.
        
        Args:
            array: Input array/tensor
            target_backend: Target backend ('numpy', 'torch', 'jax')
            
        Returns:
            Array in target backend format
            
        Raises:
            RuntimeError: If target backend is not available
            ValueError: If target backend is invalid
            
        Example:
            >>> mgr = BackendManager()
            >>> x_np = np.array([1.0, 2.0])
            >>> x_torch = mgr.convert(x_np, 'torch')  # Returns torch.Tensor
        """
        # Validate target backend
        valid_backends = ['numpy', 'torch', 'jax']
        if target_backend not in valid_backends:
            raise ValueError(
                f"Invalid target backend '{target_backend}'. "
                f"Must be one of {valid_backends}"
            )
        
        # Check target backend is available
        self.require_backend(target_backend)
        
        # Detect source backend
        source_backend = self.detect(array)
        
        # If already correct backend, return as-is (no-op)
        if source_backend == target_backend:
            return array
        
        # Convert to NumPy as intermediate step (simplifies conversion logic)
        if source_backend == 'numpy':
            array_np = array
        elif source_backend == 'torch':
            array_np = array.detach().cpu().numpy()
        elif source_backend == 'jax':
            array_np = np.array(array)
        else:
            raise RuntimeError(f"Unknown source backend: {source_backend}")
        
        # Convert from NumPy to target backend
        if target_backend == 'numpy':
            return array_np
        elif target_backend == 'torch':
            import torch
            tensor = torch.tensor(array_np, dtype=torch.float32)
            
            # Move to preferred device if not CPU
            if self._preferred_device != 'cpu':
                tensor = tensor.to(self._preferred_device)
            
            return tensor
        elif target_backend == 'jax':
            import jax.numpy as jnp
            from jax import device_put, devices
            
            array_jax = jnp.array(array_np)
            
            # Move to preferred device if not CPU
            if self._preferred_device != 'cpu':
                try:
                    target_device = devices(self._preferred_device)[0]
                    array_jax = device_put(array_jax, target_device)
                except (IndexError, RuntimeError):
                    # Device not available, stay on default device
                    pass
            
            return array_jax
        else:
            raise RuntimeError(f"Unhandled target backend: {target_backend}")
    
    # ========================================================================
    # Configuration
    # ========================================================================
    
    def set_default(self, backend: str, device: Optional[str] = None) -> 'BackendManager':
        """
        Set default backend and optionally device.
        
        Args:
            backend: Backend name ('numpy', 'torch', 'jax')
            device: Device name ('cpu', 'cuda', 'cuda:0', 'gpu:0', etc.)
                   If None, device is not changed.
            
        Returns:
            Self for method chaining
            
        Raises:
            ValueError: If backend name is invalid
            RuntimeError: If backend is not available
            
        Example:
            >>> mgr = BackendManager()
            >>> mgr.set_default('torch', device='cuda')
            >>> mgr.default_backend  # Returns 'torch'
        """
        # Validate backend
        valid_backends = ['numpy', 'torch', 'jax']
        if backend not in valid_backends:
            raise ValueError(
                f"Invalid backend '{backend}'. "
                f"Must be one of {valid_backends}"
            )
        
        # Check availability
        self.require_backend(backend)
        
        # Set backend
        self._default_backend = backend
        
        # Set device if provided
        if device is not None:
            self.to_device(device)
        
        return self
    
    def to_device(self, device: str) -> 'BackendManager':
        """
        Set preferred device for GPU-capable backends.
        
        Args:
            device: Device string ('cpu', 'cuda', 'cuda:0', 'gpu:0', etc.)
            
        Returns:
            Self for method chaining
            
        Note:
            NumPy always uses CPU. Device setting only affects PyTorch and JAX.
            
        Example:
            >>> mgr = BackendManager()
            >>> mgr.to_device('cuda:0')
            >>> mgr.preferred_device  # Returns 'cuda:0'
        """
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
        self._default_backend = 'numpy'
        self._preferred_device = 'cpu'
    
    # ========================================================================
    # Context Managers
    # ========================================================================
    
    @contextmanager
    def use_backend(self, backend: str, device: Optional[str] = None):
        """
        Temporarily switch to a different backend and/or device.
        
        Args:
            backend: Temporary backend to use
            device: Optional temporary device
            
        Yields:
            Self with temporary configuration
            
        Example:
            >>> mgr = BackendManager(default_backend='numpy')
            >>> 
            >>> with mgr.use_backend('torch', device='cuda'):
            >>>     # Code here uses torch backend on CUDA
            >>>     x = mgr.convert(x_np, mgr.default_backend)
            >>> 
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
    
    def get_info(self) -> Dict[str, any]:
        """
        Get comprehensive information about backend status.
        
        Returns:
            Dictionary with backend configuration and availability
            
        Example:
            >>> mgr = BackendManager()
            >>> info = mgr.get_info()
            >>> print(info)
            {
                'default_backend': 'numpy',
                'preferred_device': 'cpu',
                'available_backends': ['numpy', 'torch', 'jax'],
                'torch_available': True,
                'jax_available': True,
                'numpy_version': '1.24.0',
                'torch_version': '2.0.0',
                'jax_version': '0.4.10'
            }
        """
        info = {
            'default_backend': self._default_backend,
            'preferred_device': self._preferred_device,
            'available_backends': self.available_backends,
            'torch_available': self.check_available('torch'),
            'jax_available': self.check_available('jax'),
        }
        
        # Add version information
        info['numpy_version'] = np.__version__
        
        if self.check_available('torch'):
            import torch
            info['torch_version'] = torch.__version__
        else:
            info['torch_version'] = None
        
        if self.check_available('jax'):
            import jax
            info['jax_version'] = jax.__version__
        else:
            info['jax_version'] = None
        
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
        return f"BackendManager(default='{self._default_backend}', device='{self._preferred_device}')"