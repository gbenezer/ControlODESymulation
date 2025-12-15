"""
Comprehensive tests for array backend implementations.

Tests verify that all backends (PyTorch, NumPy, JAX) implement the
ArrayBackend interface correctly and produce consistent results.

Run with:
    pytest tests/array_backend_test.py -v
    pytest tests/array_backend_test.py -v -k "torch"  # Only PyTorch tests
    pytest tests/array_backend_test.py -v -k "numpy"  # Only NumPy tests
    pytest tests/array_backend_test.py -v -k "jax"    # Only JAX tests
    pytest tests/array_backend_test.py -v -k "gpu"    # Only GPU tests
"""

import pytest
import numpy as np
from typing import List, Any

# Import array backends
from src.systems.base.array_backend import (
    ArrayBackend,
    TorchArrayBackend,
    NumpyArrayBackend,
)

# Import backend utilities
from src.systems.base.backend_utils import get_device_info

# Conditionally import optional backends
torch_available = False
jax_available = False
jax_gpu_available = False

try:
    import torch
    torch_available = True
except ImportError:
    pass

try:
    import jax
    import jax.numpy as jnp
    from src.systems.base.array_backend import JAXArrayBackend
    jax_available = True
    
    # Check for JAX GPU support
    try:
        devices = jax.devices('gpu')
        if len(devices) > 0:
            jax_gpu_available = True
    except RuntimeError:
        jax_gpu_available = False
except ImportError:
    pass


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def torch_backend():
    """PyTorch backend fixture"""
    if not torch_available:
        pytest.skip("PyTorch not available")
    return TorchArrayBackend(device='cpu', dtype=torch.float32)


@pytest.fixture
def torch_backend_gpu():
    """PyTorch GPU backend fixture"""
    if not torch_available:
        pytest.skip("PyTorch not available")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return TorchArrayBackend(device='cuda', dtype=torch.float32)


@pytest.fixture
def numpy_backend():
    """NumPy backend fixture"""
    return NumpyArrayBackend(dtype=np.float32)


@pytest.fixture
def jax_backend():
    """JAX backend fixture (CPU)"""
    if not jax_available:
        pytest.skip("JAX not available")
    return JAXArrayBackend(device='cpu')


@pytest.fixture
def jax_backend_gpu():
    """JAX GPU backend fixture"""
    if not jax_available:
        pytest.skip("JAX not available")
    if not jax_gpu_available:
        pytest.skip("JAX GPU not available")
    return JAXArrayBackend(device='gpu')


@pytest.fixture(params=['numpy', 'torch', 'jax'])
def any_backend(request):
    """Parametrized fixture that tests all available backends"""
    backend_name = request.param
    
    if backend_name == 'numpy':
        return NumpyArrayBackend(dtype=np.float32)
    elif backend_name == 'torch':
        if not torch_available:
            pytest.skip("PyTorch not available")
        return TorchArrayBackend(device='cpu', dtype=torch.float32)
    elif backend_name == 'jax':
        if not jax_available:
            pytest.skip("JAX not available")
        return JAXArrayBackend(device='cpu')


# ============================================================================
# Test: Basic Array Creation
# ============================================================================

class TestArrayCreation:
    """Test array creation operations"""
    
    def test_zeros(self, any_backend):
        """Test zeros() creates correct array"""
        arr = any_backend.zeros((3, 4))
        
        assert arr.shape == (3, 4)
        arr_np = any_backend.to_numpy(arr)
        assert np.allclose(arr_np, 0.0)
    
    def test_zeros_1d(self, any_backend):
        """Test zeros() with 1D shape"""
        arr = any_backend.zeros((5,))
        
        assert arr.shape == (5,)
        arr_np = any_backend.to_numpy(arr)
        assert np.allclose(arr_np, 0.0)
    
    def test_zeros_3d(self, any_backend):
        """Test zeros() with 3D shape"""
        arr = any_backend.zeros((2, 3, 4))
        
        assert arr.shape == (2, 3, 4)
        arr_np = any_backend.to_numpy(arr)
        assert np.allclose(arr_np, 0.0)
    
    def test_ones(self, any_backend):
        """Test ones() creates correct array"""
        arr = any_backend.ones((2, 3))
        
        assert arr.shape == (2, 3)
        arr_np = any_backend.to_numpy(arr)
        assert np.allclose(arr_np, 1.0)
    
    def test_eye(self, any_backend):
        """Test eye() creates identity matrix"""
        arr = any_backend.eye(4)
        
        assert arr.shape == (4, 4)
        arr_np = any_backend.to_numpy(arr)
        expected = np.eye(4)
        assert np.allclose(arr_np, expected)
    
    def test_eye_different_sizes(self, any_backend):
        """Test eye() with various sizes"""
        for n in [1, 2, 5, 10]:
            arr = any_backend.eye(n)
            assert arr.shape == (n, n)
            arr_np = any_backend.to_numpy(arr)
            assert np.allclose(arr_np, np.eye(n))


# ============================================================================
# Test: Array Manipulation
# ============================================================================

class TestArrayManipulation:
    """Test array manipulation operations"""
    
    def test_concatenate_axis0(self, any_backend):
        """Test concatenate along axis 0"""
        arr1 = any_backend.ones((2, 3))
        arr2 = any_backend.zeros((3, 3))
        
        result = any_backend.concatenate([arr1, arr2], axis=0)
        
        assert result.shape == (5, 3)
        result_np = any_backend.to_numpy(result)
        
        assert np.allclose(result_np[:2, :], 1.0)
        assert np.allclose(result_np[2:, :], 0.0)
    
    def test_concatenate_axis1(self, any_backend):
        """Test concatenate along axis 1"""
        arr1 = any_backend.ones((3, 2))
        arr2 = any_backend.zeros((3, 4))
        
        result = any_backend.concatenate([arr1, arr2], axis=1)
        
        assert result.shape == (3, 6)
        result_np = any_backend.to_numpy(result)
        
        assert np.allclose(result_np[:, :2], 1.0)
        assert np.allclose(result_np[:, 2:], 0.0)
    
    def test_concatenate_multiple_arrays(self, any_backend):
        """Test concatenate with more than 2 arrays"""
        arrays = [any_backend.ones((1, 3)) for _ in range(5)]
        
        result = any_backend.concatenate(arrays, axis=0)
        
        assert result.shape == (5, 3)
        result_np = any_backend.to_numpy(result)
        assert np.allclose(result_np, 1.0)
    
    def test_stack_axis0(self, any_backend):
        """Test stack along new axis 0"""
        arr1 = any_backend.ones((2, 3))
        arr2 = any_backend.zeros((2, 3))
        
        result = any_backend.stack([arr1, arr2], axis=0)
        
        assert result.shape == (2, 2, 3)
        result_np = any_backend.to_numpy(result)
        
        assert np.allclose(result_np[0], 1.0)
        assert np.allclose(result_np[1], 0.0)
    
    def test_stack_axis1(self, any_backend):
        """Test stack along new axis 1"""
        arr1 = any_backend.ones((2, 3))
        arr2 = any_backend.zeros((2, 3))
        
        result = any_backend.stack([arr1, arr2], axis=1)
        
        assert result.shape == (2, 2, 3)
        result_np = any_backend.to_numpy(result)
        
        assert np.allclose(result_np[:, 0, :], 1.0)
        assert np.allclose(result_np[:, 1, :], 0.0)
    
    def test_expand_dims(self, any_backend):
        """Test expand_dims adds dimension"""
        arr = any_backend.ones((2, 3))
        
        result0 = any_backend.expand_dims(arr, axis=0)
        assert result0.shape == (1, 2, 3)
        
        result1 = any_backend.expand_dims(arr, axis=1)
        assert result1.shape == (2, 1, 3)
        
        result2 = any_backend.expand_dims(arr, axis=2)
        assert result2.shape == (2, 3, 1)
    
    def test_expand_dims_1d(self, any_backend):
        """Test expand_dims on 1D array"""
        arr = any_backend.ones((5,))
        
        result0 = any_backend.expand_dims(arr, axis=0)
        assert result0.shape == (1, 5)
        
        result1 = any_backend.expand_dims(arr, axis=1)
        assert result1.shape == (5, 1)
    
    def test_squeeze_remove_single_dimensions(self, any_backend):
        """Test squeeze removes dimensions of size 1"""
        arr = any_backend.ones((1, 3, 1, 4, 1))
        
        result = any_backend.squeeze(arr)
        assert result.shape == (3, 4)
    
    def test_squeeze_specific_axis(self, any_backend):
        """Test squeeze on specific axis"""
        arr = any_backend.ones((1, 3, 1, 4))
        
        result = any_backend.squeeze(arr, axis=0)
        assert result.shape == (3, 1, 4)
        
        result = any_backend.squeeze(arr, axis=2)
        assert result.shape == (1, 3, 4)
    
    def test_reshape(self, any_backend):
        """Test reshape changes array shape"""
        arr = any_backend.ones((2, 3, 4))
        
        result = any_backend.reshape(arr, (6, 4))
        assert result.shape == (6, 4)
        
        result = any_backend.reshape(arr, (24,))
        assert result.shape == (24,)
        
        result = any_backend.reshape(arr, (4, 2, 3))
        assert result.shape == (4, 2, 3)
    
    def test_reshape_preserves_data(self, any_backend):
        """Test that reshape preserves array data"""
        arr_np = np.arange(12).reshape(3, 4).astype(np.float32)
        arr = any_backend.from_numpy(arr_np)
        
        result = any_backend.reshape(arr, (4, 3))
        result_np = any_backend.to_numpy(result)
        
        expected = arr_np.reshape(4, 3)
        assert np.allclose(result_np, expected)


# ============================================================================
# Test: Type Conversions
# ============================================================================

class TestTypeConversions:
    """Test conversions between backend types and NumPy"""
    
    def test_to_numpy_from_zeros(self, any_backend):
        """Test to_numpy on zeros array"""
        arr = any_backend.zeros((3, 4))
        arr_np = any_backend.to_numpy(arr)
        
        assert isinstance(arr_np, np.ndarray)
        assert arr_np.shape == (3, 4)
        assert np.allclose(arr_np, 0.0)
    
    def test_to_numpy_from_ones(self, any_backend):
        """Test to_numpy on ones array"""
        arr = any_backend.ones((2, 5))
        arr_np = any_backend.to_numpy(arr)
        
        assert isinstance(arr_np, np.ndarray)
        assert arr_np.shape == (2, 5)
        assert np.allclose(arr_np, 1.0)
    
    def test_from_numpy_simple(self, any_backend):
        """Test from_numpy creates correct array"""
        arr_np = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        
        arr = any_backend.from_numpy(arr_np)
        result_np = any_backend.to_numpy(arr)
        assert np.allclose(result_np, arr_np)
    
    def test_from_numpy_preserves_values(self, any_backend):
        """Test from_numpy preserves exact values"""
        arr_np = np.random.randn(10, 20).astype(np.float32)
        
        arr = any_backend.from_numpy(arr_np)
        result_np = any_backend.to_numpy(arr)
        
        assert np.allclose(result_np, arr_np, rtol=1e-6)
    
    def test_roundtrip_conversion(self, any_backend):
        """Test numpy -> backend -> numpy preserves data"""
        original = np.random.randn(5, 7).astype(np.float32)
        
        backend_arr = any_backend.from_numpy(original)
        result = any_backend.to_numpy(backend_arr)
        
        assert np.allclose(result, original, rtol=1e-6)
    
    def test_from_numpy_different_shapes(self, any_backend):
        """Test from_numpy with various shapes"""
        shapes = [(5,), (3, 4), (2, 3, 4), (1, 10, 1)]
        
        for shape in shapes:
            arr_np = np.random.randn(*shape).astype(np.float32)
            arr = any_backend.from_numpy(arr_np)
            result_np = any_backend.to_numpy(arr)
            
            assert result_np.shape == shape
            assert np.allclose(result_np, arr_np, rtol=1e-6)


# ============================================================================
# Test: Backend Properties (ALL BACKENDS EQUAL)
# ============================================================================

class TestBackendProperties:
    """Test backend identification and properties"""
    
    def test_get_backend_name_numpy(self, numpy_backend):
        """Test NumPy backend name"""
        assert numpy_backend.get_backend_name() == "numpy"
    
    def test_get_backend_name_torch(self, torch_backend):
        """Test PyTorch backend name"""
        assert torch_backend.get_backend_name() == "pytorch"
    
    def test_get_backend_name_jax(self, jax_backend):
        """Test JAX backend name"""
        assert jax_backend.get_backend_name() == "jax"
    
    def test_get_default_dtype(self, any_backend):
        """Test get_default_dtype returns valid dtype"""
        dtype = any_backend.get_default_dtype()
        assert dtype is not None
    
    def test_get_device_str_exists(self, any_backend):
        """Test get_device_str() method exists on ALL backends"""
        device_str = any_backend.get_device_str()
        assert isinstance(device_str, str)
        assert len(device_str) > 0
    
    def test_get_device_exists(self, any_backend):
        """Test get_device() method exists on ALL backends"""
        device = any_backend.get_device()
        # NumPy returns None (CPU-only), others return device objects
        # Both are valid - just checking method exists
        pass  # Method existence is the test
    
    def test_numpy_backend_dtype(self):
        """Test NumPy backend with different dtypes"""
        backend_f32 = NumpyArrayBackend(dtype=np.float32)
        backend_f64 = NumpyArrayBackend(dtype=np.float64)
        
        assert backend_f32.get_default_dtype() == np.float32
        assert backend_f64.get_default_dtype() == np.float64
    
    def test_numpy_backend_device_always_cpu(self):
        """Test NumPy backend always reports CPU"""
        backend = NumpyArrayBackend()
        
        assert backend.get_device_str() == 'cpu'
        assert backend.get_device() is None  # NumPy has no device objects
    
    @pytest.mark.skipif(not torch_available, reason="PyTorch not available")
    def test_torch_backend_dtype(self):
        """Test PyTorch backend with different dtypes"""
        backend_f32 = TorchArrayBackend(dtype=torch.float32)
        backend_f64 = TorchArrayBackend(dtype=torch.float64)
        
        assert backend_f32.get_default_dtype() == torch.float32
        assert backend_f64.get_default_dtype() == torch.float64
    
    @pytest.mark.skipif(not torch_available, reason="PyTorch not available")
    def test_torch_backend_device_cpu(self):
        """Test PyTorch backend device methods on CPU"""
        backend = TorchArrayBackend(device='cpu')
        
        assert backend.get_device_str() == 'cpu'
        device = backend.get_device()
        assert device.type == 'cpu'
    
    @pytest.mark.gpu
    @pytest.mark.skipif(not torch_available or not torch.cuda.is_available(),
                       reason="PyTorch CUDA not available")
    def test_torch_backend_device_cuda(self):
        """Test PyTorch backend device methods on CUDA"""
        backend = TorchArrayBackend(device='cuda')
        
        assert backend.get_device_str() == 'cuda'
        device = backend.get_device()
        assert device.type == 'cuda'
    
    @pytest.mark.skipif(not jax_available, reason="JAX not available")
    def test_jax_backend_device_cpu(self):
        """Test JAX backend device methods on CPU"""
        backend = JAXArrayBackend(device='cpu')
        
        assert backend.get_device_str() == 'cpu'
        device = backend.get_device()
        assert device is not None  # JAX has device objects even for CPU
    
    @pytest.mark.gpu
    @pytest.mark.skipif(not jax_gpu_available, reason="JAX GPU not available")
    def test_jax_backend_device_gpu(self):
        """Test JAX backend device methods on GPU"""
        backend = JAXArrayBackend(device='gpu')
        
        device_str = backend.get_device_str()
        assert device_str == 'gpu'
        
        device = backend.get_device()
        assert device is not None


# ============================================================================
# Test: PyTorch-Specific Features
# ============================================================================

@pytest.mark.skipif(not torch_available, reason="PyTorch not available")
class TestTorchBackendSpecific:
    """Test PyTorch-specific features"""
    
    def test_device_cpu(self):
        """Test PyTorch backend on CPU"""
        backend = TorchArrayBackend(device='cpu')
        arr = backend.zeros((3, 4))
        
        assert arr.device.type == 'cpu'
    
    @pytest.mark.gpu
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_device_cuda(self):
        """Test PyTorch backend on CUDA"""
        backend = TorchArrayBackend(device='cuda')
        arr = backend.zeros((3, 4))
        
        assert arr.device.type == 'cuda'
    
    def test_gradient_tracking(self):
        """Test that created arrays support gradients"""
        backend = TorchArrayBackend(device='cpu')
        arr = backend.ones((2, 3))
        
        arr.requires_grad_(True)
        assert arr.requires_grad
    
    def test_operations_preserve_device(self):
        """Test that operations preserve device"""
        backend = TorchArrayBackend(device='cpu')
        
        arr1 = backend.ones((2, 3))
        arr2 = backend.zeros((2, 3))
        
        result = backend.concatenate([arr1, arr2], axis=0)
        assert result.device.type == 'cpu'
    
    @pytest.mark.gpu
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_operations(self):
        """Test operations stay on GPU"""
        backend = TorchArrayBackend(device='cuda')
        
        arr1 = backend.ones((2, 3))
        arr2 = backend.zeros((2, 3))
        
        assert arr1.device.type == 'cuda'
        assert arr2.device.type == 'cuda'
        
        result = backend.stack([arr1, arr2], axis=0)
        assert result.device.type == 'cuda'
    
    @pytest.mark.gpu
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_to_cpu_conversion(self):
        """Test GPU arrays convert to CPU numpy correctly"""
        backend = TorchArrayBackend(device='cuda')
        arr = backend.ones((3, 4))
        
        arr_np = backend.to_numpy(arr)
        
        assert isinstance(arr_np, np.ndarray)
        assert np.allclose(arr_np, 1.0)


# ============================================================================
# Test: JAX-Specific Features
# ============================================================================

@pytest.mark.skipif(not jax_available, reason="JAX not available")
class TestJAXBackendSpecific:
    """Test JAX-specific features"""
    
    def test_device_cpu(self, jax_backend):
        """Test JAX backend on CPU"""
        arr = jax_backend.zeros((3, 4))
        
        # JAX arrays have device attributes
        assert hasattr(arr, 'devices') or hasattr(arr, 'device') or hasattr(arr, 'device_buffer')
    
    @pytest.mark.gpu
    @pytest.mark.skipif(not jax_gpu_available, reason="JAX GPU not available")
    def test_device_gpu(self, jax_backend_gpu):
        """Test JAX backend on GPU"""
        import jax
        
        arr = jax_backend_gpu.zeros((3, 4))
        
        # Use generic device info helper (backend equality!)
        device, device_str = get_device_info(arr, 'jax')
        assert 'gpu' in device_str.lower() or 'cuda' in device_str.lower()
    
    @pytest.mark.gpu
    @pytest.mark.skipif(not jax_gpu_available, reason="JAX GPU not available")
    def test_gpu_operations_stay_on_gpu(self, jax_backend_gpu):
        """Test that JAX GPU operations stay on GPU"""
        import jax
        
        arr1 = jax_backend_gpu.ones((2, 3))
        arr2 = jax_backend_gpu.zeros((2, 3))
        
        # Use generic helper
        dev1_obj, dev1_str = get_device_info(arr1, 'jax')
        dev2_obj, dev2_str = get_device_info(arr2, 'jax')
        
        assert 'gpu' in dev1_str.lower() or 'cuda' in dev1_str.lower()
        assert 'gpu' in dev2_str.lower() or 'cuda' in dev2_str.lower()
        
        # Operations should preserve device
        result = jax_backend_gpu.concatenate([arr1, arr2], axis=0)
        dev_result_obj, dev_result_str = get_device_info(result, 'jax')
        
        assert 'gpu' in dev_result_str.lower() or 'cuda' in dev_result_str.lower()
    
    @pytest.mark.gpu
    @pytest.mark.skipif(not jax_gpu_available, reason="JAX GPU not available")
    def test_gpu_to_numpy_conversion(self, jax_backend_gpu):
        """Test JAX GPU arrays convert to numpy correctly"""
        arr = jax_backend_gpu.ones((5, 7))
        
        arr_np = jax_backend_gpu.to_numpy(arr)
        
        assert isinstance(arr_np, np.ndarray)
        assert arr_np.shape == (5, 7)
        assert np.allclose(arr_np, 1.0)
    
    def test_jax_jit_compatibility(self, jax_backend):
        """Test that JAX arrays work with JIT compilation"""
        import jax
        
        arr = jax_backend.ones((3, 3))
        
        @jax.jit
        def simple_op(x):
            return x + 1.0
        
        result = simple_op(arr)
        result_np = jax_backend.to_numpy(result)
        
        assert np.allclose(result_np, 2.0)
    
    @pytest.mark.gpu
    @pytest.mark.skipif(not jax_gpu_available, reason="JAX GPU not available")
    def test_jax_gpu_jit_compilation(self, jax_backend_gpu):
        """Test JAX JIT compilation on GPU"""
        import jax
        import jax.numpy as jnp
        
        arr = jax_backend_gpu.ones((100, 100))
        
        @jax.jit
        def matrix_operation(x):
            return jnp.dot(x, x.T)
        
        # First call compiles
        result = matrix_operation(arr)
        
        # Second call uses compiled version
        result = matrix_operation(arr)
        
        # Use generic helper
        device, device_str = get_device_info(result, 'jax')
        assert 'gpu' in device_str.lower() or 'cuda' in device_str.lower()
    
    @pytest.mark.gpu
    @pytest.mark.skipif(not jax_gpu_available, reason="JAX GPU not available")
    def test_jax_multiple_gpus(self):
        """Test JAX with multiple GPUs (if available)"""
        import jax
        
        gpu_devices = jax.devices('gpu')
        
        if len(gpu_devices) > 1:
            backend_gpu0 = JAXArrayBackend(device='gpu:0')
            backend_gpu1 = JAXArrayBackend(device='gpu:1')
            
            arr0 = backend_gpu0.ones((5, 5))
            arr1 = backend_gpu1.ones((5, 5))
            
            # Use generic helper
            dev0, _ = get_device_info(arr0, 'jax')
            dev1, _ = get_device_info(arr1, 'jax')
            
            assert dev0 != dev1
        else:
            pytest.skip(f"Only {len(gpu_devices)} GPU(s) available, need 2+")


# ============================================================================
# Test: Cross-Backend Consistency
# ============================================================================

class TestCrossBackendConsistency:
    """Test that all backends produce consistent results"""
    
    @pytest.mark.skipif(not torch_available, reason="PyTorch not available")
    def test_numpy_torch_consistency(self):
        """Test NumPy and PyTorch backends give same results"""
        numpy_backend = NumpyArrayBackend(dtype=np.float32)
        torch_backend = TorchArrayBackend(device='cpu', dtype=torch.float32)
        
        arr_np = np.random.randn(5, 7).astype(np.float32)
        
        arr_numpy = numpy_backend.from_numpy(arr_np)
        arr_torch = torch_backend.from_numpy(arr_np)
        
        result_numpy = numpy_backend.reshape(arr_numpy, (7, 5))
        result_torch = torch_backend.reshape(arr_torch, (7, 5))
        
        result_numpy_np = numpy_backend.to_numpy(result_numpy)
        result_torch_np = torch_backend.to_numpy(result_torch)
        
        assert np.allclose(result_numpy_np, result_torch_np, rtol=1e-6)
    
    @pytest.mark.skipif(not (torch_available and jax_available), 
                       reason="Both PyTorch and JAX required")
    def test_torch_jax_consistency(self):
        """Test PyTorch and JAX backends give same results"""
        torch_backend = TorchArrayBackend(device='cpu', dtype=torch.float32)
        jax_backend = JAXArrayBackend(device='cpu')
        
        arr_np = np.random.randn(4, 6).astype(np.float32)
        
        arr_torch = torch_backend.from_numpy(arr_np)
        arr_jax = jax_backend.from_numpy(arr_np)
        
        result_torch = torch_backend.concatenate(
            [arr_torch, torch_backend.zeros((2, 6))], axis=0
        )
        result_jax = jax_backend.concatenate(
            [arr_jax, jax_backend.zeros((2, 6))], axis=0
        )
        
        result_torch_np = torch_backend.to_numpy(result_torch)
        result_jax_np = jax_backend.to_numpy(result_jax)
        
        assert np.allclose(result_torch_np, result_jax_np, rtol=1e-6)
    
    @pytest.mark.gpu
    @pytest.mark.skipif(not (torch_available and torch.cuda.is_available() and jax_gpu_available),
                       reason="Both PyTorch CUDA and JAX GPU required")
    def test_torch_gpu_jax_gpu_consistency(self):
        """Test PyTorch GPU and JAX GPU give same results"""
        torch_backend = TorchArrayBackend(device='cuda', dtype=torch.float32)
        jax_backend = JAXArrayBackend(device='gpu')
        
        arr_np = np.random.randn(10, 15).astype(np.float32)
        
        arr_torch = torch_backend.from_numpy(arr_np)
        arr_jax = jax_backend.from_numpy(arr_np)
        
        result_torch = torch_backend.concatenate(
            [arr_torch, torch_backend.ones((5, 15))], axis=0
        )
        result_jax = jax_backend.concatenate(
            [arr_jax, jax_backend.ones((5, 15))], axis=0
        )
        
        result_torch_np = torch_backend.to_numpy(result_torch)
        result_jax_np = jax_backend.to_numpy(result_jax)
        
        assert np.allclose(result_torch_np, result_jax_np, rtol=1e-6)
    
    @pytest.mark.parametrize("shape", [(5,), (3, 4), (2, 3, 4)])
    def test_all_backends_consistent_shapes(self, shape):
        """Test all backends handle various shapes identically"""
        backends = [NumpyArrayBackend(dtype=np.float32)]
        
        if torch_available:
            backends.append(TorchArrayBackend(device='cpu', dtype=torch.float32))
        
        if jax_available:
            backends.append(JAXArrayBackend(device='cpu'))
        
        if len(backends) < 2:
            pytest.skip("Need at least 2 backends")
        
        arr_np = np.random.randn(*shape).astype(np.float32)
        arrays = [backend.from_numpy(arr_np) for backend in backends]
        results = [backend.to_numpy(arr) for backend, arr in zip(backends, arrays)]
        
        for result in results[1:]:
            assert np.allclose(results[0], result, rtol=1e-6)


# ============================================================================
# Test: Edge Cases (ALL BACKENDS EQUAL)
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_empty_concatenate(self, any_backend):
        """Test concatenate with empty list"""
        with pytest.raises((ValueError, RuntimeError, TypeError)):
            any_backend.concatenate([], axis=0)
    
    def test_single_array_concatenate(self, any_backend):
        """Test concatenate with single array"""
        arr = any_backend.ones((3, 4))
        result = any_backend.concatenate([arr], axis=0)
        
        assert result.shape == (3, 4)
        result_np = any_backend.to_numpy(result)
        assert np.allclose(result_np, 1.0)
    
    def test_mismatched_shapes_concatenate(self, any_backend):
        """Test concatenate with incompatible shapes"""
        arr1 = any_backend.ones((2, 3))
        arr2 = any_backend.zeros((2, 4))
        
        # All backends raise some exception type - accept all equally
        # JAX: TypeError, PyTorch: RuntimeError, NumPy: ValueError
        with pytest.raises((ValueError, RuntimeError, TypeError)):
            any_backend.concatenate([arr1, arr2], axis=0)
    
    def test_squeeze_no_singleton_dims(self, any_backend):
        """Test squeeze on array without singleton dimensions"""
        arr = any_backend.ones((2, 3, 4))
        result = any_backend.squeeze(arr)
        
        assert result.shape == (2, 3, 4)
    
    def test_reshape_invalid_size(self, any_backend):
        """Test reshape with incompatible size"""
        arr = any_backend.ones((2, 3))
        
        # All backends raise some exception type - accept all equally
        # JAX: TypeError, PyTorch: RuntimeError, NumPy: ValueError
        with pytest.raises((ValueError, RuntimeError, TypeError)):
            any_backend.reshape(arr, (5, 2))
    
    def test_zero_size_array(self, any_backend):
        """Test creating zero-size arrays"""
        arr = any_backend.zeros((0, 5))
        assert arr.shape == (0, 5)
        
        arr = any_backend.ones((3, 0))
        assert arr.shape == (3, 0)


# ============================================================================
# Test: Performance (Optional, Informational)
# ============================================================================

class TestPerformance:
    """Performance comparison tests (informational only)"""
    
    @pytest.mark.slow
    def test_large_array_creation_speed(self, any_backend):
        """Compare large array creation speed (informational)"""
        import time
        
        start = time.time()
        arr = any_backend.zeros((1000, 1000))
        elapsed = time.time() - start
        
        print(f"\n{any_backend.get_backend_name()} zeros(1000, 1000): {elapsed:.4f}s")
        assert elapsed < 1.0
    
    @pytest.mark.slow
    @pytest.mark.gpu
    @pytest.mark.skipif(not torch_available or not torch.cuda.is_available(), 
                       reason="PyTorch CUDA not available")
    def test_torch_gpu_vs_cpu_creation(self):
        """Compare PyTorch GPU vs CPU creation (informational)"""
        import time
        
        cpu_backend = TorchArrayBackend(device='cpu')
        gpu_backend = TorchArrayBackend(device='cuda')
        
        start = time.time()
        arr_cpu = cpu_backend.zeros((5000, 5000))
        cpu_time = time.time() - start
        
        gpu_backend.zeros((100, 100))
        torch.cuda.synchronize()
        
        start = time.time()
        arr_gpu = gpu_backend.zeros((5000, 5000))
        torch.cuda.synchronize()
        gpu_time = time.time() - start
        
        print(f"\nPyTorch - CPU: {cpu_time:.4f}s, GPU: {gpu_time:.4f}s")
        print(f"Speedup: {cpu_time/gpu_time:.2f}x")
    
    @pytest.mark.slow
    @pytest.mark.gpu
    @pytest.mark.skipif(not jax_gpu_available, reason="JAX GPU not available")
    def test_jax_gpu_vs_cpu_creation(self):
        """Compare JAX GPU vs CPU creation (informational)"""
        import time
        import jax
        
        cpu_backend = JAXArrayBackend(device='cpu')
        gpu_backend = JAXArrayBackend(device='gpu')
        
        cpu_backend.zeros((100, 100))
        gpu_backend.zeros((100, 100))
        
        start = time.time()
        arr_cpu = cpu_backend.zeros((5000, 5000))
        cpu_time = time.time() - start
        
        start = time.time()
        arr_gpu = gpu_backend.zeros((5000, 5000))
        if hasattr(arr_gpu, 'block_until_ready'):
            arr_gpu.block_until_ready()
        gpu_time = time.time() - start
        
        print(f"\nJAX - CPU: {cpu_time:.4f}s, GPU: {gpu_time:.4f}s")
        print(f"Speedup: {cpu_time/gpu_time:.2f}x")


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])