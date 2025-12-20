"""
Unit tests for IntegratorBase abstract interface.

Tests cover:
1. Enum definitions (StepMode)
2. IntegrationResult container
3. Base class initialization and validation
4. Statistics tracking
5. Common utilities
6. String representations
"""

import pytest
import numpy as np
from src.systems.base.numerical_integration.integrator_base import (
    IntegratorBase,
    StepMode,
    IntegrationResult
)


# ============================================================================
# Test Class 1: StepMode Enum
# ============================================================================

class TestStepMode:
    """Test StepMode enumeration"""
    
    def test_step_mode_values(self):
        """Test that StepMode has correct values"""
        assert StepMode.FIXED.value == "fixed"
        assert StepMode.ADAPTIVE.value == "adaptive"
    
    def test_step_mode_members(self):
        """Test that all expected members exist"""
        assert hasattr(StepMode, 'FIXED')
        assert hasattr(StepMode, 'ADAPTIVE')
    
    def test_step_mode_comparison(self):
        """Test StepMode comparison"""
        assert StepMode.FIXED == StepMode.FIXED
        assert StepMode.FIXED != StepMode.ADAPTIVE


# ============================================================================
# Test Class 2: IntegrationResult Container
# ============================================================================

class TestIntegrationResult:
    """Test IntegrationResult container class"""
    
    def test_basic_initialization(self):
        """Test basic IntegrationResult creation"""
        t = np.array([0.0, 0.1, 0.2])
        x = np.array([[1.0], [0.9], [0.8]])
        
        result = IntegrationResult(t=t, x=x)
        
        assert np.array_equal(result.t, t)
        assert np.array_equal(result.x, x)
        assert result.success is True
        assert result.message == "Integration successful"
    
    def test_initialization_with_metadata(self):
        """Test IntegrationResult with all parameters"""
        t = np.array([0.0, 1.0])
        x = np.array([[1.0], [0.5]])
        
        result = IntegrationResult(
            t=t,
            x=x,
            success=True,
            message="Completed",
            nfev=100,
            nsteps=50,
            solver_info="Additional data"
        )
        
        assert result.success is True
        assert result.message == "Completed"
        assert result.nfev == 100
        assert result.nsteps == 50
        assert result.metadata['solver_info'] == "Additional data"
    
    def test_failed_integration_result(self):
        """Test IntegrationResult for failed integration"""
        result = IntegrationResult(
            t=np.array([0.0]),
            x=np.array([[1.0]]),
            success=False,
            message="Integration failed: step size too small"
        )
        
        assert result.success is False
        assert "failed" in result.message.lower()
    
    def test_repr(self):
        """Test __repr__ output"""
        result = IntegrationResult(
            t=np.array([0.0, 1.0]),
            x=np.array([[1.0], [0.5]]),
            nfev=50,
            nsteps=25
        )
        
        repr_str = repr(result)
        
        assert 'IntegrationResult' in repr_str
        assert 'success=True' in repr_str
        assert 'nsteps=25' in repr_str
        assert 'nfev=50' in repr_str


# ============================================================================
# Test Class 3: IntegratorBase Abstract Interface
# ============================================================================

class TestIntegratorBaseInterface:
    """Test IntegratorBase abstract class behavior"""
    
    def test_cannot_instantiate_directly(self):
        """Test that IntegratorBase cannot be instantiated"""
        
        # Mock system
        class MockSystem:
            nx = 1
            nu = 1
            _initialized = True
            def __call__(self, x, u, backend='numpy'):
                return -x
        
        system = MockSystem()
        
        # Should raise TypeError because step() and integrate() are abstract
        with pytest.raises(TypeError, match="abstract"):
            IntegratorBase(system, dt=0.01)
    
    def test_subclass_must_implement_step(self):
        """Test that subclasses must implement step()"""
        
        class IncompleteIntegrator(IntegratorBase):
            # Missing step() implementation!
            def integrate(self, x0, u_func, t_span, t_eval=None, dense_output=False):
                pass
            
            @property
            def name(self):
                return "Incomplete"
        
        class MockSystem:
            nx = 1
            nu = 1
            _initialized = True
            def __call__(self, x, u, backend='numpy'):
                return -x
        
        # Should still raise because step() is abstract
        with pytest.raises(TypeError):
            IncompleteIntegrator(MockSystem(), dt=0.01)
    
    def test_subclass_must_implement_integrate(self):
        """Test that subclasses must implement integrate()"""
        
        class IncompleteIntegrator(IntegratorBase):
            def step(self, x, u, dt=None):
                return x  # Dummy
            # Missing integrate()!
            
            @property
            def name(self):
                return "Incomplete"
        
        class MockSystem:
            nx = 1
            nu = 1
            _initialized = True
            def __call__(self, x, u, backend='numpy'):
                return -x
        
        # Should raise because integrate() is abstract
        with pytest.raises(TypeError):
            IncompleteIntegrator(MockSystem(), dt=0.01)
    
    def test_subclass_must_implement_name(self):
        """Test that subclasses must implement name property"""
        
        class IncompleteIntegrator(IntegratorBase):
            def step(self, x, u, dt=None):
                return x
            
            def integrate(self, x0, u_func, t_span, t_eval=None, dense_output=False):
                pass
            # Missing name property!
        
        class MockSystem:
            nx = 1
            nu = 1
            _initialized = True
            def __call__(self, x, u, backend='numpy'):
                return -x
        
        # Should raise because name is abstract
        with pytest.raises(TypeError):
            IncompleteIntegrator(MockSystem(), dt=0.01)


# ============================================================================
# Test Class 4: Initialization and Validation
# ============================================================================

class TestInitializationValidation:
    """Test initialization and parameter validation"""
    
    def test_fixed_mode_requires_dt(self):
        """Test that FIXED mode requires dt parameter"""
        
        # Create a minimal concrete subclass for testing
        class MinimalIntegrator(IntegratorBase):
            def step(self, x, u, dt=None):
                return x
            def integrate(self, x0, u_func, t_span, t_eval=None, dense_output=False):
                return IntegrationResult(np.array([0]), np.array([[0]]))
            @property
            def name(self):
                return "Minimal"
        
        class MockSystem:
            nx = 1
            nu = 1
            _initialized = True
            def __call__(self, x, u, backend='numpy'):
                return -x
        
        system = MockSystem()
        
        # Should raise ValueError
        with pytest.raises(ValueError, match="dt.*required.*FIXED"):
            MinimalIntegrator(system, dt=None, step_mode=StepMode.FIXED)
    
    def test_adaptive_mode_default_dt(self):
        """Test that ADAPTIVE mode provides default dt if None"""
        
        class MinimalIntegrator(IntegratorBase):
            def step(self, x, u, dt=None):
                return x
            def integrate(self, x0, u_func, t_span, t_eval=None, dense_output=False):
                return IntegrationResult(np.array([0]), np.array([[0]]))
            @property
            def name(self):
                return "Minimal"
        
        class MockSystem:
            nx = 1
            nu = 1
            _initialized = True
            def __call__(self, x, u, backend='numpy'):
                return -x
        
        system = MockSystem()
        
        # Should not raise, provides default dt=0.01
        integrator = MinimalIntegrator(system, dt=None, step_mode=StepMode.ADAPTIVE)
        
        assert integrator.dt == 0.01  # Default value
    
    def test_invalid_backend_raises_error(self):
        """Test that invalid backend raises ValueError"""
        
        class MinimalIntegrator(IntegratorBase):
            def step(self, x, u, dt=None):
                return x
            def integrate(self, x0, u_func, t_span, t_eval=None, dense_output=False):
                return IntegrationResult(np.array([0]), np.array([[0]]))
            @property
            def name(self):
                return "Minimal"
        
        class MockSystem:
            nx = 1
            nu = 1
            _initialized = True
            def __call__(self, x, u, backend='numpy'):
                return -x
        
        system = MockSystem()
        
        with pytest.raises(ValueError, match="Invalid backend"):
            MinimalIntegrator(system, dt=0.01, backend='tensorflow')
    
    def test_options_stored(self):
        """Test that options are stored correctly"""
        
        class MinimalIntegrator(IntegratorBase):
            def step(self, x, u, dt=None):
                return x
            def integrate(self, x0, u_func, t_span, t_eval=None, dense_output=False):
                return IntegrationResult(np.array([0]), np.array([[0]]))
            @property
            def name(self):
                return "Minimal"
        
        class MockSystem:
            nx = 1
            nu = 1
            _initialized = True
            def __call__(self, x, u, backend='numpy'):
                return -x
        
        system = MockSystem()
        
        integrator = MinimalIntegrator(
            system, dt=0.01,
            rtol=1e-8,
            atol=1e-10,
            max_steps=5000,
            custom_option='test'
        )
        
        assert integrator.rtol == 1e-8
        assert integrator.atol == 1e-10
        assert integrator.max_steps == 5000
        assert integrator.options['custom_option'] == 'test'


# ============================================================================
# Test Class 5: Statistics Tracking
# ============================================================================

class TestStatisticsTracking:
    """Test statistics tracking in IntegratorBase"""
    
    def test_initial_stats(self):
        """Test that stats start at zero"""
        
        class MinimalIntegrator(IntegratorBase):
            def step(self, x, u, dt=None):
                return x
            def integrate(self, x0, u_func, t_span, t_eval=None, dense_output=False):
                return IntegrationResult(np.array([0]), np.array([[0]]))
            @property
            def name(self):
                return "Minimal"
        
        class MockSystem:
            nx = 1
            nu = 1
            _initialized = True
            def __call__(self, x, u, backend='numpy'):
                return -x
        
        integrator = MinimalIntegrator(MockSystem(), dt=0.01)
        
        stats = integrator.get_stats()
        
        assert stats['total_steps'] == 0
        assert stats['total_fev'] == 0
        assert stats['total_time'] == 0.0
        assert stats['avg_fev_per_step'] == 0.0
    
    def test_reset_stats(self):
        """Test resetting statistics"""
        
        class MinimalIntegrator(IntegratorBase):
            def step(self, x, u, dt=None):
                self._stats['total_steps'] += 1
                return x
            def integrate(self, x0, u_func, t_span, t_eval=None, dense_output=False):
                return IntegrationResult(np.array([0]), np.array([[0]]))
            @property
            def name(self):
                return "Minimal"
        
        class MockSystem:
            nx = 1
            nu = 1
            _initialized = True
            def __call__(self, x, u, backend='numpy'):
                return -x
        
        integrator = MinimalIntegrator(MockSystem(), dt=0.01)
        
        # Do some steps
        integrator.step(np.array([1.0]), np.array([0.0]))
        assert integrator.get_stats()['total_steps'] == 1
        
        # Reset
        integrator.reset_stats()
        stats = integrator.get_stats()
        
        assert stats['total_steps'] == 0
        assert stats['total_fev'] == 0
        assert stats['total_time'] == 0.0


# ============================================================================
# Test Class 6: String Representations
# ============================================================================

class TestStringRepresentations:
    """Test __repr__ and __str__ methods"""
    
    def test_repr(self):
        """Test __repr__ output"""
        
        class MinimalIntegrator(IntegratorBase):
            def step(self, x, u, dt=None):
                return x
            def integrate(self, x0, u_func, t_span, t_eval=None, dense_output=False):
                return IntegrationResult(np.array([0]), np.array([[0]]))
            @property
            def name(self):
                return "Minimal"
        
        class MockSystem:
            nx = 1
            nu = 1
            _initialized = True
            def __call__(self, x, u, backend='numpy'):
                return -x
        
        integrator = MinimalIntegrator(MockSystem(), dt=0.01, backend='numpy')
        
        repr_str = repr(integrator)
        
        assert 'MinimalIntegrator' in repr_str
        assert 'dt=0.01' in repr_str
        assert 'fixed' in repr_str
        assert 'numpy' in repr_str
    
    def test_str(self):
        """Test __str__ output"""
        
        class MinimalIntegrator(IntegratorBase):
            def step(self, x, u, dt=None):
                return x
            def integrate(self, x0, u_func, t_span, t_eval=None, dense_output=False):
                return IntegrationResult(np.array([0]), np.array([[0]]))
            @property
            def name(self):
                return "Minimal Test Integrator"
        
        class MockSystem:
            nx = 1
            nu = 1
            _initialized = True
            def __call__(self, x, u, backend='numpy'):
                return -x
        
        integrator = MinimalIntegrator(MockSystem(), dt=0.05, backend='numpy')
        
        str_repr = str(integrator)
        
        assert 'Minimal Test Integrator' in str_repr
        assert '0.05' in str_repr
        assert 'numpy' in str_repr


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])