"""
Comprehensive unit tests for BackendManager

Tests cover:
1. Initialization and configuration
2. Backend detection
3. Backend conversion
4. Availability checking
5. Device management
6. Context managers
7. Information retrieval
"""

import pytest
import numpy as np

# Conditional imports for backends
torch_available = True
try:
    import torch
except ImportError:
    torch_available = False

jax_available = True
try:
    import jax
    import jax.numpy as jnp
except ImportError:
    jax_available = False

from src.systems.base.backend_manager import BackendManager


# ============================================================================
# Test Class 1: Initialization and Configuration
# ============================================================================


class TestInitializationAndConfiguration:
    """Test backend manager initialization and configuration"""
    
    def test_default_initialization(self):
        """Test default initialization"""
        mgr = BackendManager()
        
        assert mgr.default_backend == 'numpy'
        assert mgr.preferred_device == 'cpu'
        assert 'numpy' in mgr.available_backends
    
    def test_custom_initialization(self):
        """Test initialization with custom defaults"""
        if not torch_available:
            pytest.skip("PyTorch not installed")
        
        mgr = BackendManager(default_backend='torch', default_device='cuda')
        
        assert mgr.default_backend == 'torch'
        assert mgr.preferred_device == 'cuda'
    
    def test_invalid_default_backend(self):
        """Test error when default backend is unavailable"""
        # Try to set a backend that doesn't exist
        with pytest.raises(RuntimeError, match="not available"):
            BackendManager(default_backend='tensorflow')
    
    def test_available_backends_detected(self):
        """Test that available backends are detected correctly"""
        mgr = BackendManager()
        
        # NumPy should always be available
        assert 'numpy' in mgr.available_backends
        
        # Check torch and jax match actual availability
        assert ('torch' in mgr.available_backends) == torch_available
        assert ('jax' in mgr.available_backends) == jax_available
    
    def test_properties(self):
        """Test property access"""
        mgr = BackendManager()
        
        # Test properties are read-only (from user perspective)
        assert isinstance(mgr.default_backend, str)
        assert isinstance(mgr.preferred_device, str)
        assert isinstance(mgr.available_backends, list)
    
    def test_repr(self):
        """Test __repr__ output"""
        mgr = BackendManager()
        
        repr_str = repr(mgr)
        
        assert 'BackendManager' in repr_str
        assert 'numpy' in repr_str
        assert 'cpu' in repr_str
    
    def test_str(self):
        """Test __str__ output"""
        mgr = BackendManager()
        
        str_repr = str(mgr)
        
        assert 'BackendManager' in str_repr
        assert 'numpy' in str_repr


# ============================================================================
# Test Class 2: Backend Detection
# ============================================================================


class TestBackendDetection:
    """Test backend detection from array types"""
    
    def test_detect_numpy(self):
        """Test detecting NumPy arrays"""
        mgr = BackendManager()
        x = np.array([1.0, 2.0])
        
        backend = mgr.detect(x)
        
        assert backend == 'numpy'
    
    @pytest.mark.skipif(not torch_available, reason="PyTorch not installed")
    def test_detect_torch(self):
        """Test detecting PyTorch tensors"""
        mgr = BackendManager()
        x = torch.tensor([1.0, 2.0])
        
        backend = mgr.detect(x)
        
        assert backend == 'torch'
    
    @pytest.mark.skipif(not jax_available, reason="JAX not installed")
    def test_detect_jax(self):
        """Test detecting JAX arrays"""
        mgr = BackendManager()
        x = jnp.array([1.0, 2.0])
        
        backend = mgr.detect(x)
        
        assert backend == 'jax'
    
    def test_detect_invalid_type(self):
        """Test error on invalid array type"""
        mgr = BackendManager()
        
        with pytest.raises(TypeError, match="Unknown input type"):
            mgr.detect([1.0, 2.0])  # Python list
        
        with pytest.raises(TypeError, match="Unknown input type"):
            mgr.detect(1.0)  # Scalar
    
    def test_detect_various_shapes(self):
        """Test detection works for various array shapes"""
        mgr = BackendManager()
        
        # 1D
        assert mgr.detect(np.array([1.0])) == 'numpy'
        
        # 2D
        assert mgr.detect(np.array([[1.0, 2.0], [3.0, 4.0]])) == 'numpy'
        
        # Scalar array
        assert mgr.detect(np.array(1.0)) == 'numpy'


# ============================================================================
# Test Class 3: Backend Conversion
# ============================================================================


class TestBackendConversion:
    """Test array conversion between backends"""
    
    def test_numpy_to_numpy_noop(self):
        """Test NumPy to NumPy is no-op"""
        mgr = BackendManager()
        x = np.array([1.0, 2.0])
        
        x_converted = mgr.convert(x, 'numpy')
        
        assert x_converted is x  # Same object
        assert isinstance(x_converted, np.ndarray)
    
    @pytest.mark.skipif(not torch_available, reason="PyTorch not installed")
    def test_numpy_to_torch(self):
        """Test NumPy to PyTorch conversion"""
        mgr = BackendManager()
        x = np.array([1.0, 2.0])
        
        x_torch = mgr.convert(x, 'torch')
        
        assert isinstance(x_torch, torch.Tensor)
        assert torch.allclose(x_torch, torch.tensor([1.0, 2.0]))
    
    @pytest.mark.skipif(not torch_available, reason="PyTorch not installed")
    def test_torch_to_numpy(self):
        """Test PyTorch to NumPy conversion"""
        mgr = BackendManager()
        x = torch.tensor([1.0, 2.0])
        
        x_numpy = mgr.convert(x, 'numpy')
        
        assert isinstance(x_numpy, np.ndarray)
        assert np.allclose(x_numpy, np.array([1.0, 2.0]))
    
    @pytest.mark.skipif(not jax_available, reason="JAX not installed")
    def test_numpy_to_jax(self):
        """Test NumPy to JAX conversion"""
        mgr = BackendManager()
        x = np.array([1.0, 2.0])
        
        x_jax = mgr.convert(x, 'jax')
        
        assert isinstance(x_jax, jnp.ndarray)
        assert jnp.allclose(x_jax, jnp.array([1.0, 2.0]))
    
    @pytest.mark.skipif(not jax_available, reason="JAX not installed")
    def test_jax_to_numpy(self):
        """Test JAX to NumPy conversion"""
        mgr = BackendManager()
        x = jnp.array([1.0, 2.0])
        
        x_numpy = mgr.convert(x, 'numpy')  # Fixed: convert to 'numpy', not 'jax'
        
        assert isinstance(x_numpy, np.ndarray)
        assert np.allclose(x_numpy, np.array([1.0, 2.0]))
    
    @pytest.mark.skipif(not (torch_available and jax_available), 
                        reason="Both PyTorch and JAX required")
    def test_torch_to_jax(self):
        """Test PyTorch to JAX conversion (via NumPy)"""
        mgr = BackendManager()
        x = torch.tensor([1.0, 2.0])
        
        x_jax = mgr.convert(x, 'jax')
        
        assert isinstance(x_jax, jnp.ndarray)
        assert jnp.allclose(x_jax, jnp.array([1.0, 2.0]))
    
    def test_convert_invalid_backend(self):
        """Test error on invalid target backend"""
        mgr = BackendManager()
        x = np.array([1.0, 2.0])
        
        with pytest.raises(ValueError, match="Invalid target backend"):
            mgr.convert(x, 'tensorflow')
    
    def test_convert_unavailable_backend(self):
        """Test error when converting to unavailable backend"""
        mgr = BackendManager()
        x = np.array([1.0, 2.0])
        
        # Try to convert to a backend that's not installed
        if not torch_available:
            with pytest.raises(RuntimeError, match="PyTorch.*not available"):
                mgr.convert(x, 'torch')
        
        if not jax_available:
            with pytest.raises(RuntimeError, match="JAX.*not available"):
                mgr.convert(x, 'jax')
    
    @pytest.mark.skipif(not torch_available, reason="PyTorch not installed")
    def test_convert_preserves_values(self):
        """Test that conversion preserves numerical values"""
        mgr = BackendManager()
        
        # Test with various values
        x_np = np.array([1.5, -2.3, 0.0, 100.7])
        x_torch = mgr.convert(x_np, 'torch')
        x_back = mgr.convert(x_torch, 'numpy')
        
        assert np.allclose(x_np, x_back)
    
    @pytest.mark.skipif(not torch_available, reason="PyTorch not installed")
    def test_convert_with_device(self):
        """Test conversion respects device setting"""
        mgr = BackendManager()
        mgr.to_device('cpu')  # Explicitly set to CPU
        
        x = np.array([1.0, 2.0])
        x_torch = mgr.convert(x, 'torch')
        
        assert x_torch.device.type == 'cpu'


# ============================================================================
# Test Class 4: Availability Checking
# ============================================================================


class TestAvailabilityChecking:
    """Test backend availability checking"""
    
    def test_check_available_numpy(self):
        """Test NumPy is always available"""
        mgr = BackendManager()
        
        assert mgr.check_available('numpy') is True
    
    def test_check_available_torch(self):
        """Test torch availability matches import"""
        mgr = BackendManager()
        
        assert mgr.check_available('torch') == torch_available
    
    def test_check_available_jax(self):
        """Test JAX availability matches import"""
        mgr = BackendManager()
        
        assert mgr.check_available('jax') == jax_available
    
    def test_check_available_invalid(self):
        """Test invalid backend returns False"""
        mgr = BackendManager()
        
        assert mgr.check_available('tensorflow') is False
        assert mgr.check_available('invalid') is False
    
    def test_require_backend_numpy(self):
        """Test require_backend doesn't raise for NumPy"""
        mgr = BackendManager()
        
        # Should not raise
        mgr.require_backend('numpy')
    
    def test_require_backend_unavailable(self):
        """Test require_backend raises for unavailable backends"""
        mgr = BackendManager()
        
        if not torch_available:
            with pytest.raises(RuntimeError, match="PyTorch.*not available"):
                mgr.require_backend('torch')
        
        if not jax_available:
            with pytest.raises(RuntimeError, match="JAX.*not available"):
                mgr.require_backend('jax')
    
    @pytest.mark.skipif(not torch_available, reason="PyTorch not installed")
    def test_require_backend_available(self):
        """Test require_backend doesn't raise for available backends"""
        mgr = BackendManager()
        
        # Should not raise
        mgr.require_backend('torch')


# ============================================================================
# Test Class 5: Configuration Management
# ============================================================================


class TestConfigurationManagement:
    """Test backend and device configuration"""
    
    def test_set_default_backend(self):
        """Test setting default backend"""
        mgr = BackendManager()
        
        mgr.set_default('numpy')
        assert mgr.default_backend == 'numpy'
    
    @pytest.mark.skipif(not torch_available, reason="PyTorch not installed")
    def test_set_default_with_device(self):
        """Test setting default backend and device together"""
        mgr = BackendManager()
        
        result = mgr.set_default('torch', device='cuda')
        
        assert mgr.default_backend == 'torch'
        assert mgr.preferred_device == 'cuda'
        assert result is mgr  # Method chaining
    
    def test_set_default_invalid_backend(self):
        """Test error on invalid backend"""
        mgr = BackendManager()
        
        with pytest.raises(ValueError, match="Invalid backend"):
            mgr.set_default('tensorflow')
    
    def test_set_default_unavailable_backend(self):
        """Test error when setting unavailable backend"""
        mgr = BackendManager()
        
        if not torch_available:
            with pytest.raises(RuntimeError, match="not available"):
                mgr.set_default('torch')
    
    def test_to_device(self):
        """Test device setting"""
        mgr = BackendManager()
        
        result = mgr.to_device('cuda:0')
        
        assert mgr.preferred_device == 'cuda:0'
        assert result is mgr  # Method chaining
    
    def test_to_device_various_formats(self):
        """Test various device string formats"""
        mgr = BackendManager()
        
        # CPU
        mgr.to_device('cpu')
        assert mgr.preferred_device == 'cpu'
        
        # CUDA
        mgr.to_device('cuda')
        assert mgr.preferred_device == 'cuda'
        
        # CUDA with index
        mgr.to_device('cuda:1')
        assert mgr.preferred_device == 'cuda:1'
        
        # GPU (JAX style)
        mgr.to_device('gpu:0')
        assert mgr.preferred_device == 'gpu:0'
    
    def test_reset(self):
        """Test reset to defaults"""
        mgr = BackendManager()
        
        # Change configuration
        if torch_available:
            mgr.set_default('torch', device='cuda')
        
        # Reset
        mgr.reset()
        
        assert mgr.default_backend == 'numpy'
        assert mgr.preferred_device == 'cpu'
    
    def test_method_chaining(self):
        """Test that setter methods support chaining"""
        mgr = BackendManager()
        
        result = mgr.to_device('cuda').set_default('numpy')
        
        assert result is mgr
        assert mgr.default_backend == 'numpy'
        assert mgr.preferred_device == 'cuda'


# ============================================================================
# Test Class 6: Context Managers
# ============================================================================


class TestContextManagers:
    """Test temporary backend switching"""
    
    def test_use_backend_basic(self):
        """Test basic context manager usage"""
        if not torch_available:
            pytest.skip("PyTorch not installed")
        
        mgr = BackendManager(default_backend='numpy')
        
        assert mgr.default_backend == 'numpy'
        
        with mgr.use_backend('torch'):
            assert mgr.default_backend == 'torch'
        
        # Should restore after context
        assert mgr.default_backend == 'numpy'
    
    def test_use_backend_with_device(self):
        """Test context manager with device change"""
        if not torch_available:
            pytest.skip("PyTorch not installed")
        
        mgr = BackendManager(default_backend='numpy', default_device='cpu')
        
        with mgr.use_backend('torch', device='cuda'):
            assert mgr.default_backend == 'torch'
            assert mgr.preferred_device == 'cuda'
        
        # Should restore both
        assert mgr.default_backend == 'numpy'
        assert mgr.preferred_device == 'cpu'
    
    def test_use_backend_nested(self):
        """Test nested context managers"""
        if not torch_available:
            pytest.skip("PyTorch not installed")
        
        mgr = BackendManager(default_backend='numpy')
        
        with mgr.use_backend('torch'):
            assert mgr.default_backend == 'torch'
            
            # Nested context
            with mgr.use_backend('numpy'):
                assert mgr.default_backend == 'numpy'
            
            # Back to torch
            assert mgr.default_backend == 'torch'
        
        # Back to original
        assert mgr.default_backend == 'numpy'
    
    def test_use_backend_exception_safety(self):
        """Test context manager restores state even after exception"""
        if not torch_available:
            pytest.skip("PyTorch not installed")
        
        mgr = BackendManager(default_backend='numpy')
        
        try:
            with mgr.use_backend('torch'):
                assert mgr.default_backend == 'torch'
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        # Should still restore despite exception
        assert mgr.default_backend == 'numpy'


# ============================================================================
# Test Class 7: Information Retrieval
# ============================================================================


class TestInformationRetrieval:
    """Test getting information about backend status"""
    
    def test_get_info_basic(self):
        """Test basic info retrieval"""
        mgr = BackendManager()
        
        info = mgr.get_info()
        
        assert 'default_backend' in info
        assert 'preferred_device' in info
        assert 'available_backends' in info
        assert 'torch_available' in info
        assert 'jax_available' in info
        assert 'numpy_version' in info
    
    def test_get_info_content(self):
        """Test info content is correct"""
        mgr = BackendManager()
        
        info = mgr.get_info()
        
        assert info['default_backend'] == 'numpy'
        assert info['preferred_device'] == 'cpu'
        assert 'numpy' in info['available_backends']
        assert info['torch_available'] == torch_available
        assert info['jax_available'] == jax_available
        assert info['numpy_version'] is not None
    
    def test_get_info_versions(self):
        """Test version information is included"""
        mgr = BackendManager()
        
        info = mgr.get_info()
        
        # NumPy version should always be present
        assert info['numpy_version'] is not None
        
        # Torch/JAX versions should match availability
        if torch_available:
            assert info['torch_version'] is not None
        else:
            assert info['torch_version'] is None
        
        if jax_available:
            assert info['jax_version'] is not None
        else:
            assert info['jax_version'] is None
    
    def test_get_info_after_configuration(self):
        """Test info reflects configuration changes"""
        mgr = BackendManager()
        
        if torch_available:
            mgr.set_default('torch', device='cuda')
            info = mgr.get_info()
            
            assert info['default_backend'] == 'torch'
            assert info['preferred_device'] == 'cuda'


# ============================================================================
# Test Class 8: Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests combining multiple features"""
    
    @pytest.mark.skipif(not torch_available, reason="PyTorch not installed")
    def test_full_workflow(self):
        """Test complete workflow"""
        # Create manager
        mgr = BackendManager()
        
        # Create NumPy array
        x_np = np.array([1.0, 2.0, 3.0])
        
        # Convert to torch
        x_torch = mgr.convert(x_np, 'torch')
        assert isinstance(x_torch, torch.Tensor)
        
        # Detect backend
        assert mgr.detect(x_torch) == 'torch'
        
        # Convert back
        x_back = mgr.convert(x_torch, 'numpy')
        assert np.allclose(x_np, x_back)
        
        # Change default backend
        mgr.set_default('torch')
        assert mgr.default_backend == 'torch'
        
        # Use context manager
        with mgr.use_backend('numpy'):
            assert mgr.default_backend == 'numpy'
        
        assert mgr.default_backend == 'torch'
    
    def test_multi_backend_consistency(self):
        """Test that conversions maintain values across all backends"""
        mgr = BackendManager()
        
        original = np.array([1.5, -2.7, 0.0, 42.0])
        
        backends_to_test = ['numpy']
        if torch_available:
            backends_to_test.append('torch')
        if jax_available:
            backends_to_test.append('jax')
        
        # Test all pairwise conversions
        for backend in backends_to_test:
            converted = mgr.convert(original, backend)
            back_to_numpy = mgr.convert(converted, 'numpy')
            
            assert np.allclose(original, back_to_numpy), \
                f"Values changed during {backend} round-trip"


# ============================================================================
# Test Class 9: Edge Cases
# ============================================================================


class TestEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_empty_array(self):
        """Test with empty arrays"""
        mgr = BackendManager()
        
        x = np.array([])
        backend = mgr.detect(x)
        
        assert backend == 'numpy'
    
    @pytest.mark.skipif(not torch_available, reason="PyTorch not installed")
    def test_large_array(self):
        """Test with large arrays"""
        mgr = BackendManager()
        
        x = np.random.randn(1000, 1000)
        x_torch = mgr.convert(x, 'torch')
        
        assert isinstance(x_torch, torch.Tensor)
        assert x_torch.shape == (1000, 1000)
    
    def test_special_values(self):
        """Test with special float values"""
        mgr = BackendManager()
        
        # NaN, Inf, -Inf
        x = np.array([np.nan, np.inf, -np.inf, 0.0])
        
        # Should not raise
        backend = mgr.detect(x)
        assert backend == 'numpy'
        
        if torch_available:
            x_torch = mgr.convert(x, 'torch')
            assert torch.isnan(x_torch[0])
            assert torch.isinf(x_torch[1])


# ============================================================================
# Run Tests
# ============================================================================


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])