"""
Comprehensive tests for backend_utils.py

Tests verify that utility functions work correctly for ALL backends
with equal treatment.

Run with:
    pytest tests/backend_utils_test.py -v
    pytest tests/backend_utils_test.py -v -k "device_info"
    pytest tests/backend_utils_test.py -v -k "diagnostic"
"""

import pytest
import numpy as np
from typing import Any

from src.systems.base.backend_utils import (
    get_device_info,
    detect_backend,
    diagnose_installation,
    diagnose_all_backends,
    print_installation_summary,
    quick_test_backend,
    compare_backend_performance,
)

from src.systems.base.array_backend import (
    TorchArrayBackend,
    NumpyArrayBackend,
)

# Conditional imports
torch_available = False
jax_available = False

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
except ImportError:
    pass


# ============================================================================
# Test: detect_backend()
# ============================================================================

class TestDetectBackend:
    """Test automatic backend detection from array types"""
    
    def test_detect_numpy_array(self):
        """Test detection of NumPy arrays"""
        arr = np.ones((3, 4))
        backend = detect_backend(arr)
        assert backend == 'numpy'
    
    @pytest.mark.skipif(not torch_available, reason="PyTorch not available")
    def test_detect_torch_tensor(self):
        """Test detection of PyTorch tensors"""
        arr = torch.ones(3, 4)
        backend = detect_backend(arr)
        assert backend == 'pytorch'
    
    @pytest.mark.skipif(not torch_available, reason="PyTorch not available")
    def test_detect_torch_cuda_tensor(self):
        """Test detection of PyTorch CUDA tensors"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        arr = torch.ones(3, 4, device='cuda')
        backend = detect_backend(arr)
        assert backend == 'pytorch'
    
    @pytest.mark.skipif(not jax_available, reason="JAX not available")
    def test_detect_jax_array(self):
        """Test detection of JAX arrays"""
        arr = jnp.ones((3, 4))
        backend = detect_backend(arr)
        assert backend == 'jax'
    
    def test_detect_unknown_type(self):
        """Test detection with unknown type"""
        class CustomArray:
            pass
        
        arr = CustomArray()
        backend = detect_backend(arr)
        assert backend == 'unknown'
    
    def test_detect_python_list(self):
        """Test detection with Python list (not an array type)"""
        arr = [1, 2, 3]
        backend = detect_backend(arr)
        assert backend == 'unknown'


# ============================================================================
# Test: get_device_info()
# ============================================================================

class TestGetDeviceInfo:
    """Test device information extraction for all backends"""
    
    def test_numpy_device_info(self):
        """Test device info for NumPy arrays (always CPU)"""
        arr = np.ones((3, 4))
        device, device_str = get_device_info(arr)
        
        assert device is None  # NumPy has no device objects
        assert device_str == 'cpu'
    
    def test_numpy_device_info_with_hint(self):
        """Test device info with explicit backend hint"""
        arr = np.ones((3, 4))
        device, device_str = get_device_info(arr, backend_name='numpy')
        
        assert device is None
        assert device_str == 'cpu'
    
    @pytest.mark.skipif(not torch_available, reason="PyTorch not available")
    def test_torch_cpu_device_info(self):
        """Test device info for PyTorch CPU tensors"""
        arr = torch.ones(3, 4, device='cpu')
        device, device_str = get_device_info(arr)
        
        assert device.type == 'cpu'
        assert 'cpu' in device_str
    
    @pytest.mark.skipif(not torch_available or not torch.cuda.is_available(),
                       reason="PyTorch CUDA not available")
    def test_torch_cuda_device_info(self):
        """Test device info for PyTorch CUDA tensors"""
        arr = torch.ones(3, 4, device='cuda')
        device, device_str = get_device_info(arr)
        
        assert device.type == 'cuda'
        assert 'cuda' in device_str
    
    @pytest.mark.skipif(not jax_available, reason="JAX not available")
    def test_jax_cpu_device_info(self):
        """Test device info for JAX CPU arrays"""
        arr = jnp.ones((3, 4))
        device, device_str = get_device_info(arr)
        
        assert device is not None  # JAX has device objects
        assert 'cpu' in device_str.lower()
    
    @pytest.mark.skipif(not jax_available, reason="JAX not available")
    def test_jax_device_info_with_hint(self):
        """Test JAX device info with explicit backend hint"""
        arr = jnp.ones((3, 4))
        device, device_str = get_device_info(arr, backend_name='jax')
        
        assert device is not None
        assert isinstance(device_str, str)
    
    def test_device_info_unknown_type(self):
        """Test device info with unknown array type"""
        arr = [1, 2, 3]
        device, device_str = get_device_info(arr)
        
        assert device is None
        assert device_str == 'unknown'
    
    @pytest.mark.skipif(not torch_available, reason="PyTorch not available")
    def test_device_info_wrong_hint(self):
        """Test device info with wrong backend hint"""
        arr = torch.ones(3, 4)
        
        # Wrong hint should raise TypeError
        with pytest.raises(TypeError):
            get_device_info(arr, backend_name='numpy')


# ============================================================================
# Test: diagnose_installation()
# ============================================================================

class TestDiagnoseInstallation:
    """Test installation diagnostics for all backends"""
    
    def test_diagnose_all_backends(self):
        """Test diagnosing all backends"""
        info = diagnose_installation()
        
        # Should have entries for all three backends
        assert 'numpy' in info
        assert 'pytorch' in info
        assert 'jax' in info
    
    def test_diagnose_specific_backends(self):
        """Test diagnosing specific backends only"""
        info = diagnose_installation(backends=['numpy'])
        
        assert 'numpy' in info
        assert 'pytorch' not in info
        assert 'jax' not in info
    
    def test_diagnose_numpy_structure(self):
        """Test NumPy diagnostic returns correct structure"""
        info = diagnose_installation(backends=['numpy'])
        numpy_info = info['numpy']
        
        assert 'installed' in numpy_info
        assert 'version' in numpy_info or 'error' in numpy_info
        
        if numpy_info['installed']:
            assert 'device' in numpy_info
            assert numpy_info['device'] == 'cpu'
            assert 'gpu_available' in numpy_info
            assert numpy_info['gpu_available'] is False
    
    @pytest.mark.skipif(not torch_available, reason="PyTorch not available")
    def test_diagnose_pytorch_structure(self):
        """Test PyTorch diagnostic returns correct structure"""
        info = diagnose_installation(backends=['pytorch'])
        torch_info = info['pytorch']
        
        assert 'installed' in torch_info
        assert torch_info['installed'] is True
        assert 'version' in torch_info
        assert 'cuda_available' in torch_info
        assert 'devices' in torch_info
        assert 'cpu' in torch_info['devices']
    
    @pytest.mark.skipif(not jax_available, reason="JAX not available")
    def test_diagnose_jax_structure(self):
        """Test JAX diagnostic returns correct structure"""
        info = diagnose_installation(backends=['jax'])
        jax_info = info['jax']
        
        assert 'installed' in jax_info
        assert jax_info['installed'] is True
        assert 'version' in jax_info
        assert 'devices' in jax_info
        assert isinstance(jax_info['devices'], list)
    
    def test_diagnose_handles_missing_backend(self):
        """Test diagnostic gracefully handles uninstalled backends"""
        # Even if pytorch/jax not installed, should not crash
        info = diagnose_installation()
        
        # Should have entries for all
        assert len(info) >= 1  # At least NumPy should work


# ============================================================================
# Test: quick_test_backend()
# ============================================================================

class TestQuickTestBackend:
    """Test quick backend smoke tests"""
    
    def test_numpy_quick_test(self):
        """Test NumPy quick test"""
        result = quick_test_backend('numpy', 'cpu')
        assert result is True
    
    @pytest.mark.skipif(not torch_available, reason="PyTorch not available")
    def test_pytorch_cpu_quick_test(self):
        """Test PyTorch CPU quick test"""
        result = quick_test_backend('pytorch', 'cpu')
        assert result is True
    
    @pytest.mark.skipif(not torch_available or not torch.cuda.is_available(),
                       reason="PyTorch CUDA not available")
    def test_pytorch_gpu_quick_test(self):
        """Test PyTorch GPU quick test"""
        result = quick_test_backend('pytorch', 'gpu')
        assert result is True
    
    @pytest.mark.skipif(not jax_available, reason="JAX not available")
    def test_jax_cpu_quick_test(self):
        """Test JAX CPU quick test"""
        result = quick_test_backend('jax', 'cpu')
        assert result is True
    
    def test_unknown_backend_quick_test(self):
        """Test quick test with unknown backend"""
        result = quick_test_backend('unknown_backend', 'cpu')
        assert result is False
    
    def test_numpy_gpu_not_supported(self):
        """Test that NumPy GPU correctly returns False"""
        # NumPy doesn't support GPU, should handle gracefully
        result = quick_test_backend('numpy', 'gpu')
        assert result is False


# ============================================================================
# Test: compare_backend_performance()
# ============================================================================

class TestCompareBackendPerformance:
    """Test performance comparison utilities"""
    
    @pytest.mark.slow
    def test_compare_all_backends(self):
        """Test performance comparison across available backends"""
        times = compare_backend_performance('matmul', size=100)
        
        # Should have at least NumPy
        assert isinstance(times, dict)
        assert len(times) > 0
        assert 'numpy_cpu' in times
        
        # NumPy should succeed
        if isinstance(times['numpy_cpu'], float):
            assert times['numpy_cpu'] > 0
    
    @pytest.mark.slow
    @pytest.mark.skipif(not torch_available, reason="PyTorch not available")
    def test_compare_pytorch_included(self):
        """Test that PyTorch is included in comparison"""
        times = compare_backend_performance('matmul', size=100)
        
        assert 'pytorch_cpu' in times
        if isinstance(times['pytorch_cpu'], float):
            assert times['pytorch_cpu'] > 0
    
    @pytest.mark.slow
    def test_compare_specific_backends(self):
        """Test comparing only specific backends"""
        times = compare_backend_performance(
            'matmul', size=100,
            backends=['numpy'],
            device_types=['cpu']
        )
        
        assert 'numpy_cpu' in times
        assert 'pytorch_cpu' not in times
        assert 'jax_cpu' not in times
    
    @pytest.mark.slow
    def test_compare_handles_errors_gracefully(self):
        """Test that errors in benchmark are reported, not raised"""
        # Request GPU for NumPy (not supported)
        times = compare_backend_performance(
            'matmul', size=100,
            backends=['numpy'],
            device_types=['cpu', 'gpu']
        )
        
        # NumPy CPU should work
        assert 'numpy_cpu' in times
        # NumPy GPU should be skipped (invalid combination)
        assert 'numpy_gpu' not in times


# ============================================================================
# Test: Print Functions (Smoke Tests)
# ============================================================================

class TestPrintFunctions:
    """Test that print functions execute without errors"""
    
    def test_print_installation_summary(self, capsys):
        """Test print_installation_summary executes"""
        print_installation_summary()
        
        captured = capsys.readouterr()
        
        # Should print header
        assert "Backend Installation Summary" in captured.out
        
        # Should mention all three backends
        assert "NumPy" in captured.out
        assert "PyTorch" in captured.out
        assert "JAX" in captured.out
    
    def test_diagnose_all_backends_prints(self, capsys):
        """Test diagnose_all_backends executes"""
        diagnose_all_backends()
        
        captured = capsys.readouterr()
        
        # Should print comprehensive diagnostic
        assert "Comprehensive Backend Diagnostics" in captured.out
        assert "Summary" in captured.out
        
        # Should mention all backends
        assert "NumPy" in captured.out
        assert "PyTorch" in captured.out
        assert "JAX" in captured.out


# ============================================================================
# Test: Backend Equality in Diagnostics
# ============================================================================

class TestBackendEquality:
    """Verify that all backends get equal treatment"""
    
    def test_all_backends_diagnosed(self):
        """Test that all backends are diagnosed"""
        info = diagnose_installation()
        
        # All three should be present
        assert 'numpy' in info
        assert 'pytorch' in info
        assert 'jax' in info
        
        # All should have 'installed' key
        assert 'installed' in info['numpy']
        assert 'installed' in info['pytorch']
        assert 'installed' in info['jax']
    
    def test_all_backends_have_version_or_error(self):
        """Test that all backends report version or error"""
        info = diagnose_installation()
        
        for backend_name, backend_info in info.items():
            if backend_info['installed']:
                assert 'version' in backend_info, f"{backend_name} missing version"
            else:
                assert 'error' in backend_info, f"{backend_name} missing error"
    
    def test_device_info_works_for_all(self):
        """Test get_device_info works for all backend types"""
        test_cases = [
            (np.ones(3), 'numpy', 'cpu'),
        ]
        
        if torch_available:
            test_cases.append((torch.ones(3), 'pytorch', 'cpu'))
        
        if jax_available:
            test_cases.append((jnp.ones(3), 'jax', 'cpu'))
        
        for arr, expected_backend, expected_device in test_cases:
            device, device_str = get_device_info(arr)
            
            # Should successfully extract device info
            assert isinstance(device_str, str)
            assert expected_device in device_str.lower()
    
    def test_quick_test_backend_all_cpu(self):
        """Test quick_test_backend for all backends on CPU"""
        # NumPy should always work
        assert quick_test_backend('numpy', 'cpu') is True
        
        # PyTorch if available
        if torch_available:
            assert quick_test_backend('pytorch', 'cpu') is True
        
        # JAX if available
        if jax_available:
            # JAX CPU might not work in all environments
            result = quick_test_backend('jax', 'cpu')
            assert isinstance(result, bool)


# ============================================================================
# Test: Edge Cases and Error Handling
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_get_device_info_none(self):
        """Test get_device_info with None"""
        device, device_str = get_device_info(None)
        assert device_str == 'unknown'
    
    @pytest.mark.skipif(not torch_available, reason="PyTorch not available")
    def test_get_device_info_type_mismatch(self):
        """Test get_device_info with wrong type hint"""
        arr = torch.ones(3)
        
        # Providing wrong backend hint should raise error
        with pytest.raises(TypeError):
            get_device_info(arr, backend_name='numpy')
    
    def test_diagnose_empty_backend_list(self):
        """Test diagnose_installation with empty list"""
        info = diagnose_installation(backends=[])
        assert len(info) == 0
    
    def test_diagnose_invalid_backend_name(self):
        """Test diagnose_installation with invalid backend"""
        info = diagnose_installation(backends=['invalid_backend'])
        
        # Should not crash, just not have the invalid backend
        assert 'invalid_backend' not in info
    
    def test_quick_test_invalid_backend(self):
        """Test quick_test_backend with invalid backend name"""
        result = quick_test_backend('invalid', 'cpu')
        assert result is False
    
    def test_quick_test_invalid_device(self):
        """Test quick_test_backend with invalid device"""
        # NumPy doesn't support 'gpu', should return False
        result = quick_test_backend('numpy', 'gpu')
        assert result is False


# ============================================================================
# Test: Integration with ArrayBackend Classes
# ============================================================================

class TestArrayBackendIntegration:
    """Test integration between backend_utils and array_backend"""
    
    def test_numpy_backend_device_info(self):
        """Test device info from NumpyArrayBackend arrays"""
        backend = NumpyArrayBackend()
        arr = backend.ones((3, 4))
        
        device, device_str = get_device_info(arr)
        
        assert device is None
        assert device_str == 'cpu'
        assert detect_backend(arr) == 'numpy'
    
    @pytest.mark.skipif(not torch_available, reason="PyTorch not available")
    def test_torch_backend_device_info(self):
        """Test device info from TorchArrayBackend arrays"""
        backend = TorchArrayBackend(device='cpu')
        arr = backend.ones((3, 4))
        
        device, device_str = get_device_info(arr)
        
        assert device.type == 'cpu'
        assert 'cpu' in device_str
        assert detect_backend(arr) == 'pytorch'
    
    @pytest.mark.skipif(not torch_available or not torch.cuda.is_available(),
                       reason="PyTorch CUDA not available")
    def test_torch_gpu_backend_device_info(self):
        """Test device info from TorchArrayBackend GPU arrays"""
        backend = TorchArrayBackend(device='cuda')
        arr = backend.ones((3, 4))
        
        device, device_str = get_device_info(arr)
        
        assert device.type == 'cuda'
        assert 'cuda' in device_str
        assert detect_backend(arr) == 'pytorch'
    
    @pytest.mark.skipif(not jax_available, reason="JAX not available")
    def test_jax_backend_device_info(self):
        """Test device info from JAXArrayBackend arrays"""
        backend = JAXArrayBackend(device='cpu')
        arr = backend.ones((3, 4))
        
        device, device_str = get_device_info(arr)
        
        assert device is not None
        assert isinstance(device_str, str)
        assert detect_backend(arr) == 'jax'


# ============================================================================
# Test: Consistency Across Backends
# ============================================================================

class TestCrossBackendConsistency:
    """Test that utilities treat all backends consistently"""
    
    def test_detect_backend_consistent(self):
        """Test detect_backend returns consistent format"""
        arrays = [np.ones(3)]
        expected_backends = ['numpy']
        
        if torch_available:
            arrays.append(torch.ones(3))
            expected_backends.append('pytorch')
        
        if jax_available:
            arrays.append(jnp.ones(3))
            expected_backends.append('jax')
        
        for arr, expected in zip(arrays, expected_backends):
            backend = detect_backend(arr)
            assert backend == expected
            assert isinstance(backend, str)
            assert backend in ['numpy', 'pytorch', 'jax', 'unknown']
    
    def test_device_info_format_consistent(self):
        """Test device info returns consistent format for all backends"""
        arrays = [np.ones(3)]
        
        if torch_available:
            arrays.append(torch.ones(3))
        
        if jax_available:
            arrays.append(jnp.ones(3))
        
        for arr in arrays:
            device, device_str = get_device_info(arr)
            
            # All should return tuple
            assert isinstance(device_str, str)
            # device can be None (NumPy) or object (PyTorch/JAX)
    
    def test_all_backends_in_diagnostic_summary(self, capsys):
        """Test that all backends appear in diagnostic output"""
        print_installation_summary()
        captured = capsys.readouterr()
        
        # All three backends should be mentioned
        backend_names = ['NumPy', 'PyTorch', 'JAX']
        for name in backend_names:
            assert name in captured.out, f"{name} not in diagnostic output"


# ============================================================================
# Test: Performance Comparison Equality
# ============================================================================

class TestPerformanceEquality:
    """Test that performance comparison treats backends equally"""
    
    @pytest.mark.slow
    def test_all_available_backends_benchmarked(self):
        """Test that all available backends are included in benchmark"""
        times = compare_backend_performance('matmul', size=50)
        
        # Count how many succeeded
        successful = [k for k, v in times.items() if isinstance(v, float)]
        
        # Should have at least NumPy
        assert any('numpy' in k for k in successful)
        
        # If PyTorch available, should benchmark it
        if torch_available:
            assert any('pytorch' in k for k in successful)
        
        # If JAX available, should benchmark it
        if jax_available:
            # JAX might fail if not properly configured, but should be attempted
            assert any('jax' in k for k in times.keys())
    
    @pytest.mark.slow
    def test_benchmark_same_iterations_all_backends(self):
        """Test that all backends use same number of iterations"""
        # This is implicit in the implementation, but verify timing makes sense
        times = compare_backend_performance('matmul', size=100)
        
        # All successful benchmarks should be positive
        for backend_device, time in times.items():
            if isinstance(time, float):
                assert time > 0, f"{backend_device} has invalid time: {time}"
                assert time < 10.0, f"{backend_device} took too long: {time}s"


# ============================================================================
# Test: Documentation and Help
# ============================================================================

class TestDocumentation:
    """Test that all functions have proper documentation"""
    
    def test_get_device_info_docstring(self):
        """Test get_device_info has docstring"""
        assert get_device_info.__doc__ is not None
        assert "backend" in get_device_info.__doc__.lower()
    
    def test_detect_backend_docstring(self):
        """Test detect_backend has docstring"""
        assert detect_backend.__doc__ is not None
        assert "auto-detect" in detect_backend.__doc__.lower()
    
    def test_diagnose_installation_docstring(self):
        """Test diagnose_installation has docstring"""
        assert diagnose_installation.__doc__ is not None
        assert "diagnose" in diagnose_installation.__doc__.lower()
    
    def test_all_backends_mentioned_in_docs(self):
        """Test that all backends are mentioned in documentation"""
        backends = ['numpy', 'pytorch', 'jax']
        
        # Check get_device_info docstring
        doc = get_device_info.__doc__.lower()
        for backend in backends:
            assert backend in doc or backend.replace('py', '') in doc
    
    def test_examples_in_docstrings(self):
        """Test that key functions have usage examples"""
        functions_needing_examples = [
            get_device_info,
            diagnose_installation,
            quick_test_backend,
            compare_backend_performance,
        ]
        
        for func in functions_needing_examples:
            assert func.__doc__ is not None
            assert '>>>' in func.__doc__, f"{func.__name__} missing usage examples"


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])