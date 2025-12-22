"""
Unit Tests for Integrator Factory

Tests the factory class for creating numerical integrators, including:
- Backend-specific integrator creation
- Method validation and selection
- Use case-specific factory methods
- Error handling for invalid configurations
- Method listing and recommendation utilities
- Julia DiffEqPy integration support

Test Coverage
-------------
1. Basic integrator creation
2. Backend-method compatibility validation
3. Fixed-step vs adaptive selection
4. Default method selection
5. Use case-specific factories (production, optimization, neural ODE, julia, etc.)
6. Method listing and information retrieval
7. Recommendation system
8. Error handling and validation
9. Julia DiffEqPy support (with/without Julia installed)
10. Helper method validation (_is_julia_method, _is_fixed_step_method, etc.)
"""

import pytest
import numpy as np
from unittest.mock import Mock

# Conditional import for Julia
try:
    from diffeqpy import de
    JULIA_AVAILABLE = True
except ImportError:
    JULIA_AVAILABLE = False

# Import the factory
from src.systems.base.numerical_integration.integrator_factory import (
    IntegratorFactory,
    IntegratorType,
    create_integrator,
    auto_integrator,
)
from src.systems.base.numerical_integration.integrator_base import (
    IntegratorBase,
    StepMode
)


# ============================================================================
# Mock System Fixture
# ============================================================================

@pytest.fixture
def mock_system():
    """Create a mock SymbolicDynamicalSystem for testing."""
    system = Mock()
    system.nx = 2
    system.nu = 1
    system.ny = 2
    system.__call__ = Mock(return_value=np.array([1.0, 2.0]))
    system.forward = Mock(return_value=np.array([1.0, 2.0]))
    return system


# ============================================================================
# Test Class: Basic Creation
# ============================================================================

class TestBasicCreation:
    """Test basic integrator creation with IntegratorFactory.create()."""
    
    def test_create_default_numpy(self, mock_system):
        """Test creating integrator with default numpy backend."""
        integrator = IntegratorFactory.create(mock_system, backend='numpy')
        
        assert integrator is not None
        assert integrator.backend == 'numpy'
        assert integrator.system == mock_system
    
    def test_create_with_default_method(self, mock_system):
        """Test that default methods are selected correctly."""
        integrator = IntegratorFactory.create(mock_system, backend='numpy')
        assert hasattr(integrator, 'method')
    
    def test_create_with_specific_method(self, mock_system):
        """Test creating with specific method."""
        integrator = IntegratorFactory.create(
            mock_system,
            backend='numpy',
            method='RK45'
        )
        
        assert integrator.method == 'RK45'
    
    def test_create_fixed_step_requires_dt(self, mock_system):
        """Test that fixed-step methods require dt parameter."""
        with pytest.raises(ValueError, match="requires dt"):
            IntegratorFactory.create(
                mock_system,
                backend='numpy',
                method='rk4',
                step_mode=StepMode.FIXED
            )
    
    def test_create_fixed_step_with_dt(self, mock_system):
        """Test creating fixed-step integrator with dt."""
        integrator = IntegratorFactory.create(
            mock_system,
            backend='numpy',
            method='rk4',
            dt=0.01,
            step_mode=StepMode.FIXED
        )
        
        assert integrator.dt == 0.01
        assert integrator.step_mode == StepMode.FIXED
    
    def test_create_with_options(self, mock_system):
        """Test creating integrator with additional options."""
        integrator = IntegratorFactory.create(
            mock_system,
            backend='numpy',
            method='RK45',
            rtol=1e-9,
            atol=1e-11
        )
        
        assert integrator.rtol == 1e-9
        assert integrator.atol == 1e-11


# ============================================================================
# Test Class: Backend Validation
# ============================================================================

class TestBackendValidation:
    """Test backend validation and compatibility checks."""
    
    def test_invalid_backend_raises_error(self, mock_system):
        """Test that invalid backend name raises ValueError."""
        with pytest.raises(ValueError, match="Invalid backend"):
            IntegratorFactory.create(mock_system, backend='matlab')
    
    def test_valid_backends(self, mock_system):
        """Test all valid backends can be created."""
        # Test numpy (always available)
        integrator = IntegratorFactory.create(mock_system, backend='numpy')
        assert integrator.backend == 'numpy'
        
        # Test torch if available
        try:
            import torch
            integrator = IntegratorFactory.create(mock_system, backend='torch')
            assert integrator.backend == 'torch'
        except ImportError:
            pytest.skip("PyTorch not installed")
        
        # Test jax if available
        try:
            import jax
            integrator = IntegratorFactory.create(mock_system, backend='jax')
            assert integrator.backend == 'jax'
        except ImportError:
            pytest.skip("JAX not installed")


# ============================================================================
# Test Class: Method-Backend Compatibility
# ============================================================================

class TestMethodBackendCompatibility:
    """Test validation of method-backend compatibility."""
    
    def test_scipy_method_requires_numpy(self, mock_system):
        """Test scipy methods require numpy backend."""
        with pytest.raises(ValueError, match="requires backend"):
            IntegratorFactory.create(
                mock_system,
                backend='torch',
                method='RK45'  # Scipy-only method
            )
    
    def test_universal_methods_work_with_any_backend(self, mock_system):
        """Test that universal methods (euler, rk4) work with any backend."""
        universal_methods = ['euler', 'midpoint', 'rk4']
        
        for method in universal_methods:
            try:
                integrator = IntegratorFactory.create(
                    mock_system,
                    backend='numpy',
                    method=method,
                    dt=0.01,
                    step_mode=StepMode.FIXED
                )
                assert integrator is not None
            except Exception as e:
                pytest.fail(f"Universal method {method} failed: {e}")


# ============================================================================
# Test Class: NumPy Backend Creation
# ============================================================================

class TestNumpyBackendCreation:
    """Test creation of NumPy-based integrators."""
    
    @pytest.mark.parametrize("method", ['RK45', 'RK23', 'DOP853', 'Radau', 'BDF', 'LSODA'])
    def test_scipy_methods(self, mock_system, method):
        """Test all scipy methods can be created."""
        integrator = IntegratorFactory.create(
            mock_system,
            backend='numpy',
            method=method
        )
        
        assert integrator.method == method
        assert integrator.backend == 'numpy'
    
    @pytest.mark.parametrize("method", ['euler', 'midpoint', 'rk4'])
    def test_fixed_step_methods(self, mock_system, method):
        """Test fixed-step methods with numpy."""
        integrator = IntegratorFactory.create(
            mock_system,
            backend='numpy',
            method=method,
            dt=0.01,
            step_mode=StepMode.FIXED
        )
        
        assert integrator.dt == 0.01
        assert integrator.step_mode == StepMode.FIXED


# ============================================================================
# Test Class: Julia DiffEqPy Support
# ============================================================================

class TestJuliaSupport:
    """Test Julia DiffEqPy integrator support in factory."""
    
    @pytest.mark.skipif(not JULIA_AVAILABLE, reason="Julia/diffeqpy not installed")
    def test_create_julia_integrator(self, mock_system):
        """Test creating Julia integrator via factory."""
        integrator = IntegratorFactory.create(
            mock_system,
            backend='numpy',
            method='Tsit5'  # Capital = Julia
        )
        
        assert integrator is not None
        assert integrator.backend == 'numpy'
        assert integrator.algorithm == 'Tsit5'
    
    @pytest.mark.skipif(not JULIA_AVAILABLE, reason="Julia/diffeqpy not installed")
    def test_julia_method_detection(self):
        """Test _is_julia_method correctly identifies Julia algorithms."""
        # Julia methods (capital first letter, not all caps)
        assert IntegratorFactory._is_julia_method('Tsit5')
        assert IntegratorFactory._is_julia_method('Vern9')
        assert IntegratorFactory._is_julia_method('Rosenbrock23')
        
        # Auto-switching (contains parentheses)
        assert IntegratorFactory._is_julia_method('AutoTsit5(Rosenbrock23())')
        
        # NOT Julia methods
        assert not IntegratorFactory._is_julia_method('LSODA')  # All caps
        assert not IntegratorFactory._is_julia_method('RK45')   # All caps
        assert not IntegratorFactory._is_julia_method('dopri5') # Lowercase
        assert not IntegratorFactory._is_julia_method('tsit5')  # Lowercase (Diffrax)
    
    @pytest.mark.skipif(not JULIA_AVAILABLE, reason="Julia/diffeqpy not installed")
    @pytest.mark.parametrize("algorithm", ['Tsit5', 'Vern7', 'Vern9'])
    def test_julia_nonstiff_algorithms(self, mock_system, algorithm):
        """Test creating various Julia non-stiff algorithms."""
        integrator = IntegratorFactory.create(
            mock_system,
            backend='numpy',
            method=algorithm
        )
        
        assert integrator.algorithm == algorithm
    
    @pytest.mark.skipif(not JULIA_AVAILABLE, reason="Julia/diffeqpy not installed")
    def test_julia_auto_switching(self, mock_system):
        """Test Julia auto-switching algorithm."""
        integrator = IntegratorFactory.create(
            mock_system,
            backend='numpy',
            method='AutoTsit5(Rosenbrock23())'
        )
        
        assert integrator.algorithm == 'AutoTsit5(Rosenbrock23())'
        assert 'Auto' in integrator.name
    
    def test_julia_method_without_julia_raises_clear_error(self, mock_system):
        """Test clear error when Julia method used but Julia not installed."""
        if JULIA_AVAILABLE:
            pytest.skip("Julia is installed, can't test missing Julia error")
        
        with pytest.raises(ImportError, match="requires diffeqpy"):
            IntegratorFactory.create(
                mock_system,
                backend='numpy',
                method='Tsit5'
            )
    
    @pytest.mark.skipif(not JULIA_AVAILABLE, reason="Julia/diffeqpy not installed")
    def test_for_julia_factory_method(self, mock_system):
        """Test for_julia() factory method."""
        integrator = IntegratorFactory.for_julia(
            mock_system,
            algorithm='Vern9',
            rtol=1e-12
        )
        
        assert integrator.algorithm == 'Vern9'
        assert integrator.rtol == 1e-12
    
    @pytest.mark.skipif(not JULIA_AVAILABLE, reason="Julia/diffeqpy not installed")
    def test_for_production_with_julia(self, mock_system):
        """Test production factory with Julia option."""
        integrator = IntegratorFactory.for_production(
            mock_system,
            use_julia=True
        )
        
        assert integrator.algorithm == 'AutoTsit5(Rosenbrock23())'
        assert integrator.backend == 'numpy'
    
    def test_for_production_without_julia(self, mock_system):
        """Test production factory defaults to scipy."""
        integrator = IntegratorFactory.for_production(mock_system)
        
        assert integrator.method == 'LSODA'
        assert integrator.backend == 'numpy'
    
    def test_recommend_julia_use_case(self):
        """Test recommendation for Julia use case."""
        rec = IntegratorFactory.recommend('julia')
        
        assert rec['backend'] == 'numpy'
        assert rec['method'] == 'Tsit5'
        assert rec['step_mode'] == StepMode.ADAPTIVE


# ============================================================================
# Test Class: Helper Method Validation
# ============================================================================

class TestHelperMethods:
    """Test helper methods for method classification."""
    
    def test_is_fixed_step_method(self):
        """Test _is_fixed_step_method() detection."""
        # Fixed-step methods
        assert IntegratorFactory._is_fixed_step_method('euler')
        assert IntegratorFactory._is_fixed_step_method('midpoint')
        assert IntegratorFactory._is_fixed_step_method('rk4')
        
        # NOT fixed-step methods
        assert not IntegratorFactory._is_fixed_step_method('RK45')
        assert not IntegratorFactory._is_fixed_step_method('dopri5')
        assert not IntegratorFactory._is_fixed_step_method('Tsit5')
    
    def test_is_scipy_method(self):
        """Test _is_scipy_method() detection."""
        # Scipy methods
        assert IntegratorFactory._is_scipy_method('LSODA')
        assert IntegratorFactory._is_scipy_method('RK45')
        assert IntegratorFactory._is_scipy_method('BDF')
        assert IntegratorFactory._is_scipy_method('Radau')
        
        # NOT scipy methods
        assert not IntegratorFactory._is_scipy_method('Tsit5')  # Julia
        assert not IntegratorFactory._is_scipy_method('dopri5')  # Diffrax/Torch
        assert not IntegratorFactory._is_scipy_method('euler')  # Manual
    
    def test_is_julia_method_heuristic(self):
        """Test Julia method detection heuristic (when Julia not installed)."""
        # This should work even without Julia installed
        # Based on naming convention: Capital first, not all caps
        
        if JULIA_AVAILABLE:
            # Use actual algorithm list
            assert IntegratorFactory._is_julia_method('Tsit5')
            assert IntegratorFactory._is_julia_method('Vern9')
        else:
            # Fall back to heuristic
            # Capital first letter, not all uppercase = Julia
            assert IntegratorFactory._is_julia_method('Tsit5')
            assert IntegratorFactory._is_julia_method('Vern9')
            
        # Parentheses = auto-switching Julia
        assert IntegratorFactory._is_julia_method('AutoTsit5(Rosenbrock23())')
    
    def test_helper_methods_handle_edge_cases(self):
        """Test helper methods handle edge cases."""
        # Empty string
        assert not IntegratorFactory._is_julia_method('')
        assert not IntegratorFactory._is_fixed_step_method('')
        
        # None
        assert not IntegratorFactory._is_julia_method(None)


# ============================================================================
# Test Class: PyTorch Backend Creation
# ============================================================================

@pytest.mark.skipif(
    not __import__('importlib').util.find_spec('torch'),
    reason="PyTorch not installed"
)
class TestTorchBackendCreation:
    """Test creation of PyTorch-based integrators."""
    
    @pytest.mark.parametrize("method", ['euler', 'midpoint', 'rk4'])
    def test_fixed_step_methods_torch(self, mock_system, method):
        """Test fixed-step methods with torch backend."""
        integrator = IntegratorFactory.create(
            mock_system,
            backend='torch',
            method=method,
            dt=0.01,
            step_mode=StepMode.FIXED
        )
        
        assert integrator.backend == 'torch'
        assert integrator.dt == 0.01
    
    def test_torchdiffeq_method_available(self, mock_system):
        """Test that torchdiffeq integrator can be created."""
        integrator = IntegratorFactory.create(
            mock_system,
            backend='torch',
            method='adaptive_heun'  # TorchDiffEq-specific
        )
        
        assert integrator.backend == 'torch'


# ============================================================================
# Test Class: JAX Backend Creation
# ============================================================================

@pytest.mark.skipif(
    not __import__('importlib').util.find_spec('jax'),
    reason="JAX not installed"
)
class TestJaxBackendCreation:
    """Test creation of JAX-based integrators."""
    
    @pytest.mark.parametrize("method", ['tsit5', 'dopri5', 'dopri8'])
    def test_diffrax_explicit_methods(self, mock_system, method):
        """Test Diffrax explicit methods."""
        integrator = IntegratorFactory.create(
            mock_system,
            backend='jax',
            method=method
        )
        
        # DiffraxIntegrator stores method in 'solver_name'
        assert integrator.backend == 'jax'
        assert integrator.solver_name == method
        assert integrator is not None
    
    @pytest.mark.parametrize("method", ['heun', 'ralston', 'reversible_heun'])
    def test_diffrax_basic_methods(self, mock_system, method):
        """Test Diffrax methods that are NOT universal fixed-step methods."""
        integrator = IntegratorFactory.create(
            mock_system,
            backend='jax',
            method=method
        )
        
        assert integrator.backend == 'jax'
        assert integrator.solver_name == method
    
    def test_diffrax_reversible_methods(self, mock_system):
        """Test Diffrax special methods."""
        integrator = IntegratorFactory.create(
            mock_system,
            backend='jax',
            method='reversible_heun'
        )
        
        assert integrator.backend == 'jax'


# ============================================================================
# Test Class: Auto Selection
# ============================================================================

class TestAutoSelection:
    """Test automatic integrator selection."""
    
    def test_auto_defaults_to_available_backend(self, mock_system):
        """Test auto selection works with available backend."""
        integrator = IntegratorFactory.auto(mock_system)
        
        # Should create some integrator
        assert integrator is not None
        assert integrator.backend in ['numpy', 'torch', 'jax']
    
    @pytest.mark.skipif(
        not __import__('importlib').util.find_spec('jax'),
        reason="JAX not installed"
    )
    def test_auto_prefers_jax_when_available(self, mock_system):
        """Test auto selection prefers JAX if available."""
        integrator = IntegratorFactory.auto(mock_system)
        
        # If JAX is available, should use it
        assert integrator.backend == 'jax'
    
    def test_auto_respects_preference(self, mock_system):
        """Test auto selection respects backend preference."""
        integrator = IntegratorFactory.auto(
            mock_system,
            prefer_backend='numpy'
        )
        
        assert integrator.backend == 'numpy'


# ============================================================================
# Test Class: Use Case Specific Factories
# ============================================================================

class TestUseCaseFactories:
    """Test use case-specific factory methods."""
    
    def test_for_production(self, mock_system):
        """Test production integrator factory."""
        integrator = IntegratorFactory.for_production(mock_system)
        
        assert integrator.backend == 'numpy'
        assert integrator.method == 'LSODA'
        assert integrator.rtol <= 1e-8
        assert integrator.atol <= 1e-10
    
    def test_for_production_custom_tolerances(self, mock_system):
        """Test production factory with custom tolerances."""
        integrator = IntegratorFactory.for_production(
            mock_system,
            rtol=1e-12,
            atol=1e-14
        )
        
        assert integrator.rtol == 1e-12
        assert integrator.atol == 1e-14
    
    @pytest.mark.skipif(not JULIA_AVAILABLE, reason="Julia not installed")
    def test_for_production_with_julia_flag(self, mock_system):
        """Test production factory with use_julia=True."""
        integrator = IntegratorFactory.for_production(
            mock_system,
            use_julia=True
        )
        
        assert integrator.backend == 'numpy'
        assert integrator.algorithm == 'AutoTsit5(Rosenbrock23())'
    
    def test_for_production_julia_without_install_raises(self, mock_system):
        """Test error when use_julia=True but Julia not installed."""
        if JULIA_AVAILABLE:
            pytest.skip("Julia is installed")
        
        with pytest.raises(ImportError, match="use_julia=True requires diffeqpy"):
            IntegratorFactory.for_production(mock_system, use_julia=True)
    
    @pytest.mark.skipif(
        not __import__('importlib').util.find_spec('jax'),
        reason="JAX not installed"
    )
    def test_for_optimization_jax(self, mock_system):
        """Test optimization integrator prefers JAX."""
        integrator = IntegratorFactory.for_optimization(mock_system)
        
        assert integrator.backend == 'jax'
        assert integrator is not None
    
    @pytest.mark.skipif(
        not __import__('importlib').util.find_spec('torch'),
        reason="PyTorch not installed"
    )
    def test_for_optimization_torch(self, mock_system):
        """Test optimization integrator can use torch."""
        integrator = IntegratorFactory.for_optimization(
            mock_system,
            prefer_backend='torch'
        )
        
        assert integrator.backend == 'torch'
        assert integrator.method == 'dopri5'
    
    @pytest.mark.skipif(not JULIA_AVAILABLE, reason="Julia not installed")
    def test_for_julia_default(self, mock_system):
        """Test for_julia() with default settings."""
        integrator = IntegratorFactory.for_julia(mock_system)
        
        assert integrator.algorithm == 'Tsit5'
        assert integrator.backend == 'numpy'
    
    @pytest.mark.skipif(not JULIA_AVAILABLE, reason="Julia not installed")
    def test_for_julia_custom_algorithm(self, mock_system):
        """Test for_julia() with custom algorithm."""
        integrator = IntegratorFactory.for_julia(
            mock_system,
            algorithm='Vern9',
            rtol=1e-12,
            atol=1e-14
        )
        
        assert integrator.algorithm == 'Vern9'
        assert integrator.rtol == 1e-12
        assert integrator.atol == 1e-14
    
    def test_for_julia_without_install_raises(self, mock_system):
        """Test for_julia() raises error when Julia not installed."""
        if JULIA_AVAILABLE:
            pytest.skip("Julia is installed")
        
        with pytest.raises(ImportError, match="Julia integration requires diffeqpy"):
            IntegratorFactory.for_julia(mock_system)
    
    @pytest.mark.skipif(
        not __import__('importlib').util.find_spec('torch'),
        reason="PyTorch not installed"
    )
    def test_for_neural_ode(self, mock_system):
        """Test neural ODE integrator factory."""
        integrator = IntegratorFactory.for_neural_ode(mock_system)
        
        assert integrator.backend == 'torch'
        assert integrator.method == 'dopri5'
        assert integrator is not None
    
    def test_for_simple_simulation(self, mock_system):
        """Test simple simulation factory."""
        integrator = IntegratorFactory.for_simple_simulation(
            mock_system,
            dt=0.01
        )
        
        assert integrator.dt == 0.01
        assert integrator.step_mode == StepMode.FIXED
    
    def test_for_real_time(self, mock_system):
        """Test real-time integrator factory."""
        integrator = IntegratorFactory.for_real_time(
            mock_system,
            dt=0.01
        )
        
        assert integrator.dt == 0.01
        assert integrator.step_mode == StepMode.FIXED


# ============================================================================
# Test Class: Method Listing
# ============================================================================

class TestMethodListing:
    """Test method listing utilities."""
    
    def test_list_methods_all_backends(self):
        """Test listing methods for all backends."""
        methods = IntegratorFactory.list_methods()
        
        assert 'numpy' in methods
        assert 'torch' in methods
        assert 'jax' in methods
        
        # Check numpy methods (scipy + julia + manual)
        assert 'LSODA' in methods['numpy']
        assert 'RK45' in methods['numpy']
        assert 'Tsit5' in methods['numpy']  # Julia
        assert 'euler' in methods['numpy']  # Manual
        
        # Check torch methods
        assert 'dopri5' in methods['torch']
        
        # Check jax methods
        assert 'tsit5' in methods['jax']
    
    def test_list_methods_single_backend(self):
        """Test listing methods for specific backend."""
        methods = IntegratorFactory.list_methods('numpy')
        
        assert 'numpy' in methods
        assert len(methods) == 1
        assert 'LSODA' in methods['numpy']
        assert 'Tsit5' in methods['numpy']  # Julia included
    
    def test_list_methods_shows_julia_algorithms(self):
        """Test that Julia algorithms appear in numpy methods list."""
        methods = IntegratorFactory.list_methods('numpy')
        
        # Should include representative Julia algorithms
        numpy_methods = methods['numpy']
        assert 'Tsit5' in numpy_methods
        assert 'Vern9' in numpy_methods
        assert 'Rosenbrock23' in numpy_methods
        assert 'AutoTsit5(Rosenbrock23())' in numpy_methods
    
    def test_list_methods_shows_fixed_step(self):
        """Test that universal methods appear where supported."""
        methods = IntegratorFactory.list_methods()
        
        # NumPy and torch have rk4
        assert 'euler' in methods['numpy']
        assert 'rk4' in methods['numpy']
        assert 'rk4' in methods['torch']
        
        # JAX has euler
        assert 'euler' in methods['jax']
    
    def test_list_methods_numpy_count(self):
        """Test numpy has expected number of methods."""
        methods = IntegratorFactory.list_methods('numpy')
        
        # Should have 6 scipy + ~13 Julia + 3 manual = ~22 methods
        assert len(methods['numpy']) >= 15  # At least this many


# ============================================================================
# Test Class: Method Information
# ============================================================================

class TestMethodInformation:
    """Test method information retrieval."""
    
    def test_get_info_for_scipy_method(self):
        """Test getting information about scipy method."""
        info = IntegratorFactory.get_info('numpy', 'LSODA')
        
        assert 'name' in info
        assert 'order' in info
        assert 'description' in info
        assert 'best_for' in info
        assert 'library' in info
        assert info['name'] == 'LSODA'
        assert info['library'] == 'scipy'
    
    @pytest.mark.skipif(not JULIA_AVAILABLE, reason="Julia not installed")
    def test_get_info_for_julia_method(self):
        """Test getting information about Julia method (delegates to diffeqpy)."""
        info = IntegratorFactory.get_info('numpy', 'Tsit5')
        
        assert 'name' in info
        assert 'description' in info
        assert 'library' in info
        assert info['library'] == 'Julia DifferentialEquations.jl'
    
    @pytest.mark.skipif(not JULIA_AVAILABLE, reason="Julia not installed")
    def test_get_info_julia_algorithm_not_in_hardcoded_list(self):
        """Test get_info works for Julia algorithms not hardcoded in factory."""
        # Vern7 is NOT in the hardcoded method_info dict in get_info()
        # Should delegate to get_algorithm_info() from diffeqpy
        info = IntegratorFactory.get_info('numpy', 'Vern7')
        
        assert 'name' in info
        # Should have info from diffeqpy, not "No information available"
        assert info.get('description') != 'No information available'
    
    def test_get_info_julia_without_install(self):
        """Test get_info for Julia method when Julia not installed."""
        if JULIA_AVAILABLE:
            pytest.skip("Julia is installed")
        
        info = IntegratorFactory.get_info('numpy', 'Tsit5')
        
        # Should return generic info
        assert 'name' in info
        assert 'Julia' in info['name']
        assert 'not installed' in info.get('library', '')
    
    def test_get_info_for_fixed_step(self):
        """Test info for fixed-step methods."""
        info = IntegratorFactory.get_info('numpy', 'rk4')
        
        assert info['name'] == 'Classic Runge-Kutta 4'
        assert info['order'] == 4
        assert info['type'] == 'Fixed-step'
        assert info['library'] == 'Manual implementation'
    
    def test_get_info_for_diffrax(self):
        """Test info for Diffrax methods."""
        info = IntegratorFactory.get_info('jax', 'tsit5')
        
        assert info['name'] == 'Tsitouras 5(4)'
        assert info['order'] == 5
        assert info['type'] == 'Adaptive'
        assert info['library'] == 'Diffrax'
    
    def test_get_info_unknown_method(self):
        """Test info for unknown method returns minimal info."""
        info = IntegratorFactory.get_info('numpy', 'unknown_method_xyz')
        
        assert 'name' in info
        assert info['name'] == 'unknown_method_xyz'
        assert 'description' in info


# ============================================================================
# Test Class: Recommendation System
# ============================================================================

class TestRecommendationSystem:
    """Test integrator recommendation system."""
    
    def test_recommend_production(self):
        """Test recommendation for production use."""
        rec = IntegratorFactory.recommend('production')
        
        assert rec['backend'] == 'numpy'
        assert rec['method'] == 'LSODA'
        assert rec['step_mode'] == StepMode.ADAPTIVE
        assert 'reason' in rec
    
    def test_recommend_julia(self):
        """Test recommendation for Julia use case."""
        rec = IntegratorFactory.recommend('julia')
        
        assert rec['backend'] == 'numpy'
        assert rec['method'] == 'Tsit5'
        assert rec['step_mode'] == StepMode.ADAPTIVE
        assert 'reason' in rec
    
    def test_recommend_optimization(self):
        """Test recommendation for optimization."""
        rec = IntegratorFactory.recommend('optimization', has_jax=True)
        
        assert rec['backend'] == 'jax'
        assert rec['method'] == 'tsit5'
    
    def test_recommend_optimization_no_jax(self):
        """Test optimization recommendation without JAX."""
        rec = IntegratorFactory.recommend('optimization', has_jax=False, has_torch=True)
        
        assert rec['backend'] == 'torch'
        assert rec['method'] == 'dopri5'
    
    def test_recommend_neural_ode(self):
        """Test recommendation for neural ODE."""
        rec = IntegratorFactory.recommend('neural_ode')
        
        assert rec['backend'] == 'torch'
        assert rec['method'] == 'dopri5'
        assert rec['adjoint'] is True
    
    def test_recommend_prototype(self):
        """Test recommendation for prototyping."""
        rec = IntegratorFactory.recommend('prototype')
        
        assert rec['backend'] == 'numpy'
        assert rec['method'] == 'rk4'
        assert rec['step_mode'] == StepMode.FIXED
    
    def test_recommend_educational(self):
        """Test recommendation for education."""
        rec = IntegratorFactory.recommend('educational')
        
        assert rec['backend'] == 'numpy'
        assert rec['method'] == 'euler'
        assert rec['step_mode'] == StepMode.FIXED
    
    def test_recommend_real_time(self):
        """Test recommendation for real-time."""
        rec = IntegratorFactory.recommend('real_time')
        
        assert rec['backend'] == 'numpy'
        assert rec['method'] == 'rk4'
        assert rec['step_mode'] == StepMode.FIXED
    
    def test_recommend_with_gpu(self):
        """Test recommendation adjusts for GPU."""
        rec = IntegratorFactory.recommend(
            'optimization',
            has_jax=True,
            has_torch=True,
            has_gpu=True
        )
        
        # Should prefer GPU-capable backend
        assert rec['backend'] in ['torch', 'jax']
    
    def test_recommend_invalid_use_case(self):
        """Test invalid use case raises error."""
        with pytest.raises(ValueError, match="Unknown use case"):
            IntegratorFactory.recommend('invalid_use_case')


# ============================================================================
# Test Class: Convenience Functions
# ============================================================================

class TestConvenienceFunctions:
    """Test convenience wrapper functions."""
    
    def test_create_integrator_function(self, mock_system):
        """Test create_integrator convenience function."""
        integrator = create_integrator(mock_system, backend='numpy')
        
        assert integrator is not None
        assert integrator.backend == 'numpy'
    
    def test_create_integrator_with_method(self, mock_system):
        """Test create_integrator with method."""
        integrator = create_integrator(
            mock_system,
            backend='numpy',
            method='RK45'
        )
        
        assert integrator.method == 'RK45'
    
    @pytest.mark.skipif(not JULIA_AVAILABLE, reason="Julia not installed")
    def test_create_integrator_julia_method(self, mock_system):
        """Test create_integrator with Julia method."""
        integrator = create_integrator(
            mock_system,
            backend='numpy',
            method='Tsit5'
        )
        
        assert integrator.algorithm == 'Tsit5'
    
    def test_auto_integrator_function(self, mock_system):
        """Test auto_integrator convenience function."""
        integrator = auto_integrator(mock_system)
        
        assert integrator is not None


# ============================================================================
# Test Class: Error Handling
# ============================================================================

class TestErrorHandling:
    """Test error handling in factory."""
    
    def test_missing_pytorch_for_neural_ode(self, mock_system):
        """Test error when PyTorch not available for neural ODE."""
        try:
            import torch
            pytest.skip("PyTorch is installed")
        except ImportError:
            with pytest.raises(ImportError, match="PyTorch is required"):
                IntegratorFactory.for_neural_ode(mock_system)
    
    def test_method_backend_mismatch_clear_error(self, mock_system):
        """Test clear error message for method-backend mismatch."""
        with pytest.raises(ValueError) as exc_info:
            IntegratorFactory.create(
                mock_system,
                backend='numpy',
                method='dopri5'  # Available in torch and jax, not numpy scipy
            )
        
        error_msg = str(exc_info.value).lower()
        assert 'requires backend' in error_msg or 'backend in' in error_msg
    
    def test_fixed_step_without_dt_clear_error(self, mock_system):
        """Test clear error when dt missing for fixed-step."""
        with pytest.raises(ValueError) as exc_info:
            IntegratorFactory.create(
                mock_system,
                backend='numpy',
                method='rk4',
                step_mode=StepMode.FIXED
            )
        
        assert 'requires dt' in str(exc_info.value).lower()
    
    def test_julia_method_wrong_backend_error(self, mock_system):
        """Test error when Julia method used with wrong backend."""
        with pytest.raises(ValueError, match="requires backend"):
            IntegratorFactory.create(
                mock_system,
                backend='jax',
                method='Tsit5'  # Julia method, needs numpy
            )


# ============================================================================
# Test Class: Integration with Actual System
# ============================================================================

class TestIntegrationWithActualSystem:
    """Test factory with real system (if available)."""
    
    @pytest.mark.integration
    def test_create_and_verify_type(self, mock_system):
        """Test creating integrator returns correct type."""
        integrator = IntegratorFactory.create(
            mock_system,
            backend='numpy',
            method='RK45'
        )
        
        assert isinstance(integrator, IntegratorBase)
        assert integrator.backend == 'numpy'
        assert integrator.method == 'RK45'


# ============================================================================
# Test Class: Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_empty_options(self, mock_system):
        """Test creation with no additional options."""
        integrator = IntegratorFactory.create(mock_system)
        
        assert integrator is not None
    
    def test_many_options(self, mock_system):
        """Test creation with many options."""
        integrator = IntegratorFactory.create(
            mock_system,
            backend='numpy',
            method='RK45',
            rtol=1e-10,
            atol=1e-12,
            max_steps=50000,
            first_step=1e-6
        )
        
        assert integrator.rtol == 1e-10
    
    def test_none_method_uses_default(self, mock_system):
        """Test that method=None uses backend default."""
        integrator = IntegratorFactory.create(
            mock_system,
            backend='numpy',
            method=None
        )
        
        # Should default to LSODA for numpy
        assert integrator.method == 'LSODA'


# ============================================================================
# Test Class: IntegratorType Enum
# ============================================================================

class TestIntegratorTypeEnum:
    """Test IntegratorType enum."""
    
    def test_integrator_types_defined(self):
        """Test that all integrator types are defined."""
        assert hasattr(IntegratorType, 'PRODUCTION')
        assert hasattr(IntegratorType, 'OPTIMIZATION')
        assert hasattr(IntegratorType, 'NEURAL_ODE')
        assert hasattr(IntegratorType, 'JULIA')  # New!
        assert hasattr(IntegratorType, 'SIMPLE')
        assert hasattr(IntegratorType, 'EDUCATIONAL')
    
    def test_integrator_type_values(self):
        """Test integrator type enum values."""
        assert IntegratorType.PRODUCTION.value == 'production'
        assert IntegratorType.OPTIMIZATION.value == 'optimization'
        assert IntegratorType.NEURAL_ODE.value == 'neural_ode'
        assert IntegratorType.JULIA.value == 'julia'
        assert IntegratorType.SIMPLE.value == 'simple'
        assert IntegratorType.EDUCATIONAL.value == 'educational'


# ============================================================================
# Test Class: Method-to-Backend Mapping
# ============================================================================

class TestMethodToBackendMapping:
    """Test the _METHOD_TO_BACKEND mapping."""
    
    def test_mapping_contains_scipy_methods(self):
        """Test mapping includes all scipy methods."""
        scipy_methods = ['LSODA', 'RK45', 'RK23', 'DOP853', 'Radau', 'BDF']
        
        for method in scipy_methods:
            assert method in IntegratorFactory._METHOD_TO_BACKEND
            assert IntegratorFactory._METHOD_TO_BACKEND[method] == 'numpy'
    
    def test_mapping_contains_julia_methods(self):
        """Test mapping includes Julia methods."""
        julia_methods = ['Tsit5', 'Vern9', 'Rosenbrock23', 'Rodas5']
        
        for method in julia_methods:
            assert method in IntegratorFactory._METHOD_TO_BACKEND
            assert IntegratorFactory._METHOD_TO_BACKEND[method] == 'numpy'
    
    def test_mapping_contains_universal_methods(self):
        """Test mapping includes universal fixed-step methods."""
        universal_methods = ['euler', 'midpoint', 'rk4']
        
        for method in universal_methods:
            assert method in IntegratorFactory._METHOD_TO_BACKEND
            assert IntegratorFactory._METHOD_TO_BACKEND[method] == 'any'
    
    def test_mapping_contains_shared_methods(self):
        """Test mapping includes methods shared by multiple backends."""
        # dopri5 available in both torch and jax
        assert 'dopri5' in IntegratorFactory._METHOD_TO_BACKEND
        assert isinstance(IntegratorFactory._METHOD_TO_BACKEND['dopri5'], list)
        assert 'torch' in IntegratorFactory._METHOD_TO_BACKEND['dopri5']
        assert 'jax' in IntegratorFactory._METHOD_TO_BACKEND['dopri5']


# ============================================================================
# Test Class: Naming Convention
# ============================================================================

class TestNamingConvention:
    """Test naming convention for distinguishing integrator types."""
    
    def test_scipy_naming_all_caps(self):
        """Test scipy methods are all-caps or specific names."""
        scipy_methods = ['LSODA', 'RK45', 'BDF']
        
        for method in scipy_methods:
            # Should NOT be detected as Julia
            assert not (method[0].isupper() and not method.isupper())
    
    def test_julia_naming_capital_first(self):
        """Test Julia methods have capital first letter but not all caps."""
        julia_methods = ['Tsit5', 'Vern9', 'Rosenbrock23']
        
        for method in julia_methods:
            # Capital first, not all uppercase
            assert method[0].isupper()
            assert not method.isupper()
    
    def test_diffrax_naming_lowercase(self):
        """Test Diffrax methods are lowercase."""
        diffrax_methods = ['tsit5', 'dopri5', 'heun']
        
        for method in diffrax_methods:
            assert method.islower() or '_' in method
    
    def test_auto_switching_contains_parentheses(self):
        """Test auto-switching Julia algorithms contain parentheses."""
        auto_methods = [
            'AutoTsit5(Rosenbrock23())',
            'AutoVern7(Rodas5())'
        ]
        
        for method in auto_methods:
            assert '(' in method
            assert ')' in method


# ============================================================================
# Test Class: Julia-Specific Edge Cases
# ============================================================================

class TestJuliaEdgeCases:
    """Test edge cases specific to Julia integration."""
    
    @pytest.mark.skipif(not JULIA_AVAILABLE, reason="Julia not installed")
    def test_julia_with_custom_tolerances(self, mock_system):
        """Test Julia integrator with custom tolerances."""
        integrator = IntegratorFactory.create(
            mock_system,
            backend='numpy',
            method='Tsit5',
            rtol=1e-10,
            atol=1e-12
        )
        
        assert integrator.rtol == 1e-10
        assert integrator.atol == 1e-12
    
    @pytest.mark.skipif(not JULIA_AVAILABLE, reason="Julia not installed")
    def test_julia_fixed_step_mode(self, mock_system):
        """Test Julia integrator with fixed-step mode."""
        integrator = IntegratorFactory.create(
            mock_system,
            backend='numpy',
            method='Tsit5',
            dt=0.01,
            step_mode=StepMode.FIXED
        )
        
        assert integrator.step_mode == StepMode.FIXED
        assert integrator.dt == 0.01


# ============================================================================
# Test Class: Integrator Type Enum Usage
# ============================================================================

class TestIntegratorTypeEnumUsage:
    """Test IntegratorType enum in context."""
    
    def test_enum_members_match_recommend_keys(self):
        """Test that enum members match recommendation system keys."""
        # Get all use cases from recommend()
        # Should match IntegratorType enum values
        
        enum_values = {e.value for e in IntegratorType}
        
        # These should all be valid use cases
        for use_case in enum_values:
            try:
                rec = IntegratorFactory.recommend(use_case)
                assert rec is not None
            except ValueError:
                pytest.fail(f"IntegratorType.{use_case} not in recommend system")


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])