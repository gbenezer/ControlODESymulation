"""
Unit Tests for Discretizer
===========================

Tests discretization of continuous-time systems to discrete-time,
including multiple integration methods, linearization approaches,
and backend support.

Test Coverage:
- Basic discretization (step function)
- Multiple integration methods (euler, midpoint, rk4)
- Linearization methods (euler, exact, tustin)
- Autonomous systems (nu=0)
- Higher-order systems (order=2)
- Multi-backend support (numpy, torch, jax)
- Edge cases and error handling
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import warnings

# Import components to test
from src.systems.base.discretization.discretizer import Discretizer, DiscretizationMethod
from src.systems.base.numerical_integration.integrator_factory import IntegratorFactory
from src.systems.base.numerical_integration.integrator_base import StepMode

# Import test systems
from src.systems.builtin.linear_systems import (
    LinearSystem,
    AutonomousLinearSystem,
    LinearSystem2D
)
from src.systems.builtin.mechanical_systems import SymbolicPendulum


# ============================================================================
# Helper Functions (MUST BE DEFINED BEFORE USE IN DECORATORS)
# ============================================================================

def _torch_available():
    """Check if PyTorch is available."""
    try:
        import torch
        return True
    except ImportError:
        return False


def _jax_available():
    """Check if JAX is available."""
    try:
        import jax
        return True
    except ImportError:
        return False

# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def linear_system():
    """Linear system with equilibrium at origin."""
    system = LinearSystem(a=2.0, b=1.0)
    # Add equilibrium AFTER system creation
    system.add_equilibrium(
        'origin',
        x_eq=np.array([0.0]),
        u_eq=np.array([0.0]),
        verify=True
    )
    return system


@pytest.fixture
def autonomous_system():
    """Autonomous linear system: dx/dt = -a*x (no control)"""
    return AutonomousLinearSystem(a=2.0)


@pytest.fixture
def pendulum_system():
    """Pendulum with downward and upright equilibria."""
    system = SymbolicPendulum(
        m_val=1.0,
        l_val=0.5,
        beta_val=0.1,
        g_val=9.81
    )
    
    # Add equilibria after creation
    system.add_equilibrium(
        'downward',
        x_eq=np.array([0.0, 0.0]),
        u_eq=np.array([0.0]),
        verify=True
    )
    
    system.add_equilibrium(
        'upright',
        x_eq=np.array([np.pi, 0.0]),
        u_eq=np.array([0.0]),
        verify=True
    )
    
    return system


@pytest.fixture
def dt():
    """Standard time step for tests"""
    return 0.01


# ============================================================================
# Test: SymbolicPendulum Initialization
# ============================================================================


class TestPendulumInitialization:
    """Test SymbolicPendulum initialization."""

    def test_pendulum_creation_diagnostic(self):
        """Diagnostic test to see why SymbolicPendulum fails validation."""
        
        print("\n" + "="*70)
        print("DIAGNOSTIC: Creating SymbolicPendulum")
        print("="*70)
        
        try:
            system = SymbolicPendulum(
                m_val=1.0,
                l_val=0.5,
                beta_val=0.1,
                g_val=9.81
            )
            print("✓ System created successfully!")
            print(f"  nx = {system.nx}")
            print(f"  nu = {system.nu}")
            print(f"  ny = {system.ny}")
            print(f"  order = {system.order}")
            
        except Exception as e:
            print(f"✗ System creation FAILED!")
            print(f"  Error type: {type(e).__name__}")
            print(f"  Error message: {str(e)}")
            
            # Print full traceback
            import traceback
            print("\nFull traceback:")
            traceback.print_exc()
            
            # Re-raise to see in pytest output
            raise

    def test_pendulum_attributes_diagnostic(self):
        """Check if pendulum has all required attributes."""
        import sympy as sp
        
        print("\n" + "="*70)
        print("DIAGNOSTIC: Checking SymbolicPendulum Attributes")
        print("="*70)
        
        try:
            system = SymbolicPendulum(
                m_val=1.0,
                l_val=0.5,
                beta_val=0.1,
                g_val=9.81
            )
            
            # Check state_vars
            print(f"\nstate_vars: {system.state_vars}")
            print(f"  Type: {type(system.state_vars)}")
            print(f"  Length: {len(system.state_vars)}")
            
            # Check control_vars
            print(f"\ncontrol_vars: {system.control_vars}")
            print(f"  Type: {type(system.control_vars)}")
            print(f"  Length: {len(system.control_vars)}")
            
            # Check _f_sym
            print(f"\n_f_sym:")
            print(f"  Type: {type(system._f_sym)}")
            print(f"  Shape: {system._f_sym.shape}")
            print(f"  Content:\n{system._f_sym}")
            
            # Check _h_sym
            print(f"\n_h_sym:")
            print(f"  Type: {type(system._h_sym)}")
            if system._h_sym is not None:
                print(f"  Shape: {system._h_sym.shape}")
                print(f"  Content:\n{system._h_sym}")
            
            # Check parameters
            print(f"\nparameters: {system.parameters}")
            
            # Check order
            print(f"\norder: {system.order}")
            
            # Check output_vars
            print(f"\noutput_vars: {system.output_vars}")
            
            print("\n✓ All attributes present!")
            
        except Exception as e:
            print(f"\n✗ Attribute check FAILED!")
            import traceback
            traceback.print_exc()
            raise

    def test_pendulum_manual_validation(self):
        """Manually validate the pendulum system."""
        from src.systems.base.utils.symbolic_validator import SymbolicValidator
        import sympy as sp
        
        print("\n" + "="*70)
        print("DIAGNOSTIC: Manual Validation of SymbolicPendulum")
        print("="*70)
        
        # Create system (bypass validation temporarily)
        system = SymbolicPendulum.__new__(SymbolicPendulum)
        
        # Initialize containers manually
        system.state_vars = []
        system.control_vars = []
        system.output_vars = []
        system.parameters = {}
        system._f_sym = None
        system._h_sym = None
        system.order = 1
        system._initialized = False
        
        # Call define_system
        print("\nCalling define_system()...")
        system.define_system(m_val=1.0, l_val=0.5, beta_val=0.1, g_val=9.81)
        
        # Check what was set
        print(f"\nAfter define_system():")
        print(f"  state_vars: {system.state_vars}")
        print(f"  control_vars: {system.control_vars}")
        print(f"  output_vars: {system.output_vars}")
        print(f"  _f_sym.shape: {system._f_sym.shape if system._f_sym else None}")
        print(f"  _h_sym.shape: {system._h_sym.shape if system._h_sym else None}")
        print(f"  order: {system.order}")
        print(f"  parameters: {list(system.parameters.keys())}")
        
        # Now try validation
        print("\n" + "-"*70)
        print("Running SymbolicValidator...")
        print("-"*70)
        
        validator = SymbolicValidator(system)
        
        try:
            result = validator.validate(raise_on_error=True)
            print("✓ Validation PASSED!")
            print(f"  Result: {result}")
        except Exception as e:
            print(f"✗ Validation FAILED!")
            print(f"  Error: {e}")
            
            # Try to get more details
            try:
                result = validator.validate(raise_on_error=False)
                print(f"\nValidation result (non-raising):")
                print(f"  is_valid: {result.is_valid}")
                print(f"  errors: {result.errors}")
                print(f"  warnings: {result.warnings}")
            except:
                pass
            
            raise

    def test_pendulum_validation_rules(self):
        """Check specific validation rules that might be failing."""
        from src.systems.builtin.mechanical_systems import SymbolicPendulum
        import sympy as sp
        
        system = SymbolicPendulum.__new__(SymbolicPendulum)
        system.state_vars = []
        system.control_vars = []
        system.output_vars = []
        system.parameters = {}
        system._f_sym = None
        system._h_sym = None
        system.order = 1
        
        system.define_system(m_val=1.0, l_val=0.5, beta_val=0.1, g_val=9.81)
        
        print("\n" + "="*70)
        print("DIAGNOSTIC: Checking Validation Rules")
        print("="*70)
        
        # Rule 1: state_vars should be list of Symbols
        print("\n1. Checking state_vars...")
        print(f"   type(state_vars) = {type(system.state_vars)}")
        print(f"   len(state_vars) = {len(system.state_vars)}")
        for i, var in enumerate(system.state_vars):
            print(f"   state_vars[{i}] = {var}, type = {type(var)}")
            assert isinstance(var, sp.Symbol), f"state_vars[{i}] is not a Symbol!"
        
        # Rule 2: control_vars should be list of Symbols
        print("\n2. Checking control_vars...")
        print(f"   type(control_vars) = {type(system.control_vars)}")
        print(f"   len(control_vars) = {len(system.control_vars)}")
        for i, var in enumerate(system.control_vars):
            print(f"   control_vars[{i}] = {var}, type = {type(var)}")
            assert isinstance(var, sp.Symbol), f"control_vars[{i}] is not a Symbol!"
        
        # Rule 3: _f_sym should be Matrix
        print("\n3. Checking _f_sym...")
        print(f"   type(_f_sym) = {type(system._f_sym)}")
        print(f"   _f_sym.shape = {system._f_sym.shape}")
        assert isinstance(system._f_sym, sp.Matrix), "_f_sym is not a Matrix!"
        
        # Rule 4: Dimension consistency
        print("\n4. Checking dimension consistency...")
        nx = len(system.state_vars)
        nu = len(system.control_vars)
        print(f"   nx = {nx}")
        print(f"   nu = {nu}")
        print(f"   _f_sym rows = {system._f_sym.shape[0]}")
        print(f"   order = {system.order}")
        
        if system.order == 1:
            assert system._f_sym.shape[0] == nx, \
                f"For order=1, _f_sym must have {nx} rows, got {system._f_sym.shape[0]}"
        else:
            nq = nx // system.order
            assert system._f_sym.shape[0] == nq, \
                f"For order={system.order}, _f_sym must have {nq} rows, got {system._f_sym.shape[0]}"
        
        # Rule 5: Parameters should use Symbol keys
        print("\n5. Checking parameters...")
        print(f"   Number of parameters: {len(system.parameters)}")
        for key, val in system.parameters.items():
            print(f"   {key} = {val}, key type = {type(key)}")
            assert isinstance(key, sp.Symbol), f"Parameter key {key} is not a Symbol!"
        
        # Rule 6: Check _h_sym if present
        if system._h_sym is not None:
            print("\n6. Checking _h_sym...")
            print(f"   type(_h_sym) = {type(system._h_sym)}")
            print(f"   _h_sym.shape = {system._h_sym.shape}")
            print(f"   _h_sym:\n{system._h_sym}")
            
            # Check if _h_sym depends on control (it shouldn't)
            control_syms = set(system.control_vars)
            h_free_symbols = system._h_sym.free_symbols
            control_in_h = control_syms & h_free_symbols
            
            if control_in_h:
                print(f"   WARNING: _h_sym depends on control: {control_in_h}")
        
        print("\n" + "="*70)
        print("All validation rules passed!")
        print("="*70)


# ============================================================================
# Test: Initialization
# ============================================================================

class TestInitialization:
    """Test Discretizer initialization with various configurations."""
    
    def test_basic_initialization(self, linear_system, dt):
        """Test basic initialization with default parameters."""
        discretizer = Discretizer(linear_system, dt=dt)
        
        assert discretizer.dt == dt
        assert discretizer.nx == 1
        assert discretizer.nu == 1
        assert discretizer.backend == 'numpy'
        assert discretizer.method == 'rk4'  # Default
        assert discretizer.integrator is not None
    
    def test_initialization_with_method(self, linear_system, dt):
        """Test initialization with specific integration method."""
        for method in ['euler', 'midpoint', 'rk4']:
            discretizer = Discretizer(linear_system, dt=dt, method=method)
            assert discretizer.method == method
    
    def test_initialization_with_backend(self, linear_system, dt):
        """Test initialization with different backends."""
        for backend in ['numpy']:  # Add 'torch', 'jax' if available
            discretizer = Discretizer(linear_system, dt=dt, backend=backend)
            assert discretizer.backend == backend
    
    def test_initialization_with_custom_integrator(self, linear_system, dt):
        """Test initialization with custom integrator instance."""
        custom_integrator = IntegratorFactory.create(
            linear_system, backend='numpy', method='euler', dt=dt
        )
        
        discretizer = Discretizer(
            linear_system, dt=dt, integrator=custom_integrator
        )
        
        assert discretizer.integrator is custom_integrator
    
    def test_invalid_dt_raises_error(self, linear_system):
        """Test that invalid time steps raise ValueError."""
        with pytest.raises(ValueError, match="positive"):
            Discretizer(linear_system, dt=0.0)
        
        with pytest.raises(ValueError, match="positive"):
            Discretizer(linear_system, dt=-0.01)
    
    def test_uninitialized_system_raises_error(self):
        """Test that uninitialized system raises error."""
        from src.systems.base.symbolic_dynamical_system import SymbolicDynamicalSystem
        
        class UninitializedSystem(SymbolicDynamicalSystem):
            def define_system(self):
                pass  # Don't set required attributes
        
        system = UninitializedSystem.__new__(UninitializedSystem)
        system._initialized = False
        
        with pytest.raises(ValueError, match="not initialized"):
            Discretizer(system, dt=0.01)
    
    def test_dimensions_cached(self, pendulum_system, dt):
        """Test that system dimensions are cached correctly."""
        discretizer = Discretizer(pendulum_system, dt=dt)
        
        assert discretizer.nx == pendulum_system.nx
        assert discretizer.nu == pendulum_system.nu
        assert discretizer.ny == pendulum_system.ny
        assert discretizer.order == pendulum_system.order


# ============================================================================
# Test: Step Function (Discrete Dynamics)
# ============================================================================

class TestStepFunction:
    """Test discrete-time step function with various systems and methods."""
    
    def test_step_basic(self, linear_system, dt):
        """Test basic single step with linear system."""
        discretizer = Discretizer(linear_system, dt=dt, method='euler')
        
        x = np.array([1.0])
        u = np.array([0.0])
        
        x_next = discretizer.step(x, u)
        
        # Euler: x_next = x + dt * f(x, u) = x + dt * (-a*x + u)
        # With a=2, x=1, u=0: x_next = 1 + 0.01 * (-2) = 0.98
        expected = x + dt * (-2.0 * x + u)
        assert_allclose(x_next, expected, rtol=1e-10)
    
    def test_step_with_control(self, linear_system, dt):
        """Test step with non-zero control input."""
        discretizer = Discretizer(linear_system, dt=dt, method='euler')
        
        x = np.array([1.0])
        u = np.array([0.5])
        
        x_next = discretizer.step(x, u)
        
        # x_next = 1 + 0.01 * (-2*1 + 0.5) = 1 - 0.015 = 0.985
        expected = x + dt * (-2.0 * x + u)
        assert_allclose(x_next, expected, rtol=1e-10)
    
    def test_step_callable_interface(self, linear_system, dt):
        """Test that discretizer is callable."""
        discretizer = Discretizer(linear_system, dt=dt)
        
        x = np.array([1.0])
        u = np.array([0.0])
        
        x_next_call = discretizer(x, u)
        x_next_step = discretizer.step(x, u)
        
        assert_allclose(x_next_call, x_next_step)
    
    def test_step_autonomous_system(self, autonomous_system, dt):
        """Test step with autonomous system (nu=0)."""
        discretizer = Discretizer(autonomous_system, dt=dt, method='euler')
        
        x = np.array([1.0])
        
        # Should work with u=None
        x_next = discretizer.step(x, u=None)
        
        # x_next = 1 + 0.01 * (-2*1) = 0.98
        expected = x + dt * (-2.0 * x)
        assert_allclose(x_next, expected, rtol=1e-10)
    
    def test_step_batched(self, linear_system, dt):
        """Test batched step (multiple states at once)."""
        discretizer = Discretizer(linear_system, dt=dt, method='euler')
        
        x_batch = np.array([[1.0], [2.0], [3.0]])
        u_batch = np.array([[0.0], [0.5], [1.0]])
        
        x_next_batch = discretizer.step(x_batch, u_batch)
        
        # Should have shape (3, 1)
        assert x_next_batch.shape == (3, 1)
        
        # Verify each trajectory
        for i in range(3):
            expected = x_batch[i] + dt * (-2.0 * x_batch[i] + u_batch[i])
            assert_allclose(x_next_batch[i], expected, rtol=1e-10)
    
    def test_step_with_custom_dt(self, linear_system):
        """Test step with custom dt parameter."""
        discretizer = Discretizer(linear_system, dt=0.01, method='euler')
        
        x = np.array([1.0])
        u = np.array([0.0])
        
        # Use different dt for this step
        dt_custom = 0.02
        x_next = discretizer.step(x, u, dt=dt_custom)
        
        expected = x + dt_custom * (-2.0 * x + u)
        assert_allclose(x_next, expected, rtol=1e-10)
    
    def test_step_higher_order_system(self, pendulum_system, dt):
        """Test step with second-order system."""
        discretizer = Discretizer(pendulum_system, dt=dt, method='rk4')
        
        # Pendulum state: [theta, theta_dot]
        x = np.array([0.1, 0.0])  # Small angle, at rest
        u = np.array([0.0])
        
        x_next = discretizer.step(x, u)
        
        # Should have same shape
        assert x_next.shape == x.shape
        
        # Pendulum should oscillate (theta_dot changes)
        assert x_next[1] != 0.0  # Angular velocity changed


# ============================================================================
# Test: Integration Methods
# ============================================================================

class TestIntegrationMethods:
    """Test different integration methods produce correct results."""
    
    def test_euler_vs_analytical(self, linear_system):
        """Test Euler method against analytical solution."""
        dt = 0.01
        discretizer = Discretizer(linear_system, dt=dt, method='euler')
        
        x0 = np.array([1.0])
        u = np.array([0.0])
        
        # Simulate 100 steps
        x = x0.copy()
        for _ in range(100):
            x = discretizer.step(x, u)
        
        # Analytical solution: x(t) = x0 * exp(-a*t)
        # With a=2, t=1.0: x = 1.0 * exp(-2) ≈ 0.1353
        t_final = 100 * dt
        x_analytical = x0 * np.exp(-2.0 * t_final)
        
        # Euler has O(dt) error accumulation
        # For 100 steps with dt=0.01, expect ~1-2% error
        assert_allclose(x, x_analytical, rtol=0.02)  # Increased from 0.01 to 0.02
    
    def test_rk4_vs_analytical(self, linear_system):
        """Test RK4 method against analytical solution."""
        dt = 0.01
        discretizer = Discretizer(linear_system, dt=dt, method='rk4')
        
        x0 = np.array([1.0])
        u = np.array([0.0])
        
        # Simulate 100 steps
        x = x0.copy()
        for _ in range(100):
            x = discretizer.step(x, u)
        
        # Analytical solution
        t_final = 100 * dt
        x_analytical = x0 * np.exp(-2.0 * t_final)
        
        # RK4 has O(dt^4) error, so should be very accurate
        assert_allclose(x, x_analytical, rtol=1e-6)  # Much better accuracy
    
    def test_method_accuracy_comparison(self, linear_system):
        """Compare accuracy of different methods."""
        dt = 0.01
        x0 = np.array([1.0])
        u = np.array([0.0])
        
        methods = ['euler', 'midpoint', 'rk4']
        results = {}
        
        for method in methods:
            discretizer = Discretizer(linear_system, dt=dt, method=method)
            x = x0.copy()
            for _ in range(100):
                x = discretizer.step(x, u)
            results[method] = x
        
        # Get analytical solution
        t_final = 100 * dt
        x_analytical = x0 * np.exp(-2.0 * t_final)
        
        # Compute errors
        errors = {m: np.abs(results[m] - x_analytical)[0] for m in methods}
        
        # RK4 should be most accurate
        assert errors['rk4'] < errors['midpoint']
        assert errors['midpoint'] < errors['euler']
    
    def test_consistency_across_methods(self, linear_system, dt):
        """Test that all methods produce reasonable results."""
        x = np.array([1.0])
        u = np.array([0.5])
        
        methods = ['euler', 'midpoint', 'rk4']
        results = []
        
        for method in methods:
            discretizer = Discretizer(linear_system, dt=dt, method=method)
            x_next = discretizer.step(x, u)
            results.append(x_next)
        
        # All methods should give similar results for small dt
        # (within 1% of each other)
        for i in range(len(results) - 1):
            assert_allclose(results[i], results[i+1], rtol=0.01)


# ============================================================================
# Test: Linearization
# ============================================================================

class TestLinearization:
    """Test discrete-time linearization methods."""
    
    def test_linearize_euler_method(self, linear_system, dt):
        """Test Euler linearization method."""
        discretizer = Discretizer(linear_system, dt=dt)
        
        x_eq = np.array([0.0])
        u_eq = np.array([0.0])
        
        Ad, Bd = discretizer.linearize(x_eq, u_eq, method='euler')
        
        # For linear system: Ac = -a, Bc = 1
        # Ad = I + dt*Ac = 1 + dt*(-2) = 0.98
        # Bd = dt*Bc = dt*1 = 0.01
        assert_allclose(Ad, np.array([[0.98]]), rtol=1e-10)
        assert_allclose(Bd, np.array([[0.01]]), rtol=1e-10)
    
    def test_linearize_exact_method(self, linear_system, dt):
        """Test exact linearization using matrix exponential."""
        discretizer = Discretizer(linear_system, dt=dt)
        
        x_eq = np.array([0.0])
        u_eq = np.array([0.0])
        
        Ad, Bd = discretizer.linearize(x_eq, u_eq, method='exact')
        
        # For linear system: Ad = exp(Ac*dt) = exp(-2*0.01) ≈ 0.9802
        # This is exact for linear systems
        Ad_expected = np.exp(-2.0 * dt)
        
        assert_allclose(Ad, np.array([[Ad_expected]]), rtol=1e-6)
        
        # Bd should also be accurate
        assert Bd.shape == (1, 1)
    
    def test_linearize_tustin_method(self, linear_system, dt):
        """Test Tustin (bilinear) linearization."""
        discretizer = Discretizer(linear_system, dt=dt)
        
        x_eq = np.array([0.0])
        u_eq = np.array([0.0])
        
        Ad, Bd = discretizer.linearize(x_eq, u_eq, method='tustin')
        
        # Should produce valid matrices
        assert Ad.shape == (1, 1)
        assert Bd.shape == (1, 1)
        
        # Tustin should be stable for this system
        assert np.abs(Ad[0, 0]) < 1.0  # Discrete stability
    
    def test_linearize_autonomous_system(self, autonomous_system, dt):
        """Test linearization of autonomous system."""
        discretizer = Discretizer(autonomous_system, dt=dt)
        
        x_eq = np.array([0.0])
        
        Ad, Bd = discretizer.linearize(x_eq, u_eq=None, method='euler')
        
        # Ad should be correct
        assert_allclose(Ad, np.array([[1.0 - 2.0*dt]]), rtol=1e-10)
        
        # Bd should be empty (nx, 0) for autonomous system
        assert Bd.shape == (1, 0)
    
    def test_linearize_nonlinear_system(self, pendulum_system, dt):
        """Test linearization of nonlinear system."""
        discretizer = Discretizer(pendulum_system, dt=dt)
        
        # Linearize at downward equilibrium
        x_eq = np.array([0.0, 0.0])  # [theta, theta_dot]
        u_eq = np.array([0.0])
        
        Ad, Bd = discretizer.linearize(x_eq, u_eq, method='euler')
        
        # Should return correct dimensions
        assert Ad.shape == (2, 2)
        assert Bd.shape == (2, 1)
        
        # Discrete system should be stable at downward equilibrium
        eigenvalues = np.linalg.eigvals(Ad)
        assert np.all(np.abs(eigenvalues) < 1.0)
    
    def test_linearize_at_nonequilibrium(self, linear_system, dt):
        """Test linearization at non-equilibrium point."""
        discretizer = Discretizer(linear_system, dt=dt)
        
        # Linearize at non-zero state
        x = np.array([1.0])
        u = np.array([0.5])
        
        Ad, Bd = discretizer.linearize(x, u, method='euler')
        
        # For linear system, should be same as at equilibrium
        Ad_eq, Bd_eq = discretizer.linearize(
            np.array([0.0]), np.array([0.0]), method='euler'
        )
        
        assert_allclose(Ad, Ad_eq)
        assert_allclose(Bd, Bd_eq)
    
    def test_linearization_method_comparison(self, linear_system, dt):
        """Compare linearization methods for consistency."""
        discretizer = Discretizer(linear_system, dt=dt)
        
        x_eq = np.array([0.0])
        u_eq = np.array([0.0])
        
        methods = ['euler', 'exact', 'tustin']
        Ad_results = {}
        Bd_results = {}
        
        for method in methods:
            Ad, Bd = discretizer.linearize(x_eq, u_eq, method=method)
            Ad_results[method] = Ad
            Bd_results[method] = Bd
        
        # All methods should give similar results for small dt
        # Euler vs Exact should be close
        assert_allclose(Ad_results['euler'], Ad_results['exact'], rtol=0.01)
        
        # All should be stable
        for method in methods:
            assert np.abs(Ad_results[method][0, 0]) < 1.0
    
    def test_invalid_linearization_method(self, linear_system, dt):
        """Test that invalid method raises error."""
        discretizer = Discretizer(linear_system, dt=dt)
        
        with pytest.raises(ValueError, match="Unknown linearization method"):
            discretizer.linearize(
                np.array([0.0]), np.array([0.0]), method='invalid_method'
            )


# ============================================================================
# Test: Observation (Output Functions)
# ============================================================================

class TestObservation:
    """Test observation/output function delegation."""
    
    def test_h_output_function(self, linear_system, dt):
        """Test that h() delegates to continuous system."""
        discretizer = Discretizer(linear_system, dt=dt)
        
        x = np.array([1.0])
        y = discretizer.h(x)
        
        # Should match continuous system output
        y_continuous = linear_system.h(x)
        assert_allclose(y, y_continuous)
    
    def test_linearized_observation(self, linear_system, dt):
        """Test that linearized_observation delegates correctly."""
        discretizer = Discretizer(linear_system, dt=dt)
        
        x = np.array([0.0])
        C = discretizer.linearized_observation(x)
        
        # Should match continuous system
        C_continuous = linear_system.linearized_observation(x)
        assert_allclose(C, C_continuous)
    
    def test_observation_higher_order(self, pendulum_system, dt):
        """Test observation with higher-order system."""
        discretizer = Discretizer(pendulum_system, dt=dt)
        
        x = np.array([0.1, 0.2])
        y = discretizer.h(x)
        
        # Should have correct shape
        assert y.shape[0] == pendulum_system.ny


# ============================================================================
# Test: Utility Methods
# ============================================================================

class TestUtilityMethods:
    """Test utility methods like set_dt, get_info, etc."""
    
    def test_set_dt(self, linear_system):
        """Test changing time step."""
        discretizer = Discretizer(linear_system, dt=0.01, method='euler')
        
        # Change dt
        new_dt = 0.005
        discretizer.set_dt(new_dt)
        
        assert discretizer.dt == new_dt
        
        # Should still work
        x = np.array([1.0])
        u = np.array([0.0])
        x_next = discretizer.step(x, u)
        
        assert x_next.shape == x.shape
    
    def test_set_dt_invalid(self, linear_system, dt):
        """Test that invalid dt raises error."""
        discretizer = Discretizer(linear_system, dt=dt)
        
        with pytest.raises(ValueError, match="positive"):
            discretizer.set_dt(0.0)
        
        with pytest.raises(ValueError, match="positive"):
            discretizer.set_dt(-0.01)
    
    def test_get_info(self, linear_system, dt):
        """Test get_info returns correct information."""
        discretizer = Discretizer(linear_system, dt=dt, method='rk4')
        
        info = discretizer.get_info()
        
        assert info['dt'] == dt
        assert info['method'] == 'rk4'
        assert info['backend'] == 'numpy'
        assert info['dimensions']['nx'] == 1
        assert info['dimensions']['nu'] == 1
        assert info['is_autonomous'] == False
    
    def test_get_info_autonomous(self, autonomous_system, dt):
        """Test get_info for autonomous system."""
        discretizer = Discretizer(autonomous_system, dt=dt)
        
        info = discretizer.get_info()
        
        assert info['is_autonomous'] == True
        assert info['dimensions']['nu'] == 0
    
    def test_repr(self, linear_system, dt):
        """Test __repr__ returns valid string."""
        discretizer = Discretizer(linear_system, dt=dt, method='rk4')
        
        repr_str = repr(discretizer)
        
        assert 'Discretizer' in repr_str
        assert 'dt=0.01' in repr_str
        assert 'rk4' in repr_str
    
    def test_str(self, linear_system, dt):
        """Test __str__ returns readable string."""
        discretizer = Discretizer(linear_system, dt=dt, method='euler')
        
        str_repr = str(discretizer)
        
        assert 'Discretizer' in str_repr
        assert 'dt=0.0100' in str_repr or 'dt=0.01' in str_repr


# ============================================================================
# Test: Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_very_small_dt(self, linear_system):
        """Test with very small time step."""
        dt = 1e-6
        discretizer = Discretizer(linear_system, dt=dt, method='euler')
        
        x = np.array([1.0])
        u = np.array([0.0])
        
        x_next = discretizer.step(x, u)
        
        # Should barely change
        assert_allclose(x_next, x, rtol=1e-5)
    
    def test_zero_state(self, linear_system, dt):
        """Test with zero initial state."""
        discretizer = Discretizer(linear_system, dt=dt)
        
        x = np.array([0.0])
        u = np.array([0.0])
        
        x_next = discretizer.step(x, u)
        
        # Should stay at zero
        assert_allclose(x_next, np.array([0.0]))
    
    def test_equilibrium_point(self, linear_system, dt):
        """Test at equilibrium point."""
        discretizer = Discretizer(linear_system, dt=dt, method='euler')
        
        # For dx/dt = -a*x + u = 0 at equilibrium
        # x_eq = u_eq / a = 1.0 / 2.0 = 0.5
        x_eq = np.array([0.5])
        u_eq = np.array([1.0])
        
        x_next = discretizer.step(x_eq, u_eq)
        
        # Should stay close to equilibrium
        assert_allclose(x_next, x_eq, rtol=1e-2)
    
    def test_single_dimension_system(self, linear_system, dt):
        """Test with scalar (1D) system."""
        discretizer = Discretizer(linear_system, dt=dt)
        
        # Should handle scalar inputs
        x = np.array([1.0])
        u = np.array([0.5])
        
        x_next = discretizer.step(x, u)
        
        assert x_next.shape == (1,)
    
    def test_high_dimensional_system(self, dt):
        """Test with high-dimensional system."""
        # Create a simple high-dimensional linear system
        from src.systems.base.symbolic_dynamical_system import SymbolicDynamicalSystem
        import sympy as sp
        
        class HighDimSystem(SymbolicDynamicalSystem):
            def define_system(self):
                # 10D system
                x_vars = [sp.symbols(f'x{i}') for i in range(10)]
                u_var = sp.symbols('u')
                
                self.state_vars = x_vars
                self.control_vars = [u_var]
                
                # Simple dynamics: dx_i/dt = -x_i + u
                self._f_sym = sp.Matrix([[-x + u_var] for x in x_vars])
                self.parameters = {}
                self.order = 1
        
        system = HighDimSystem()
        discretizer = Discretizer(system, dt=dt)
        
        x = np.random.randn(10)
        u = np.array([0.5])
        
        x_next = discretizer.step(x, u)
        
        assert x_next.shape == (10,)


# ============================================================================
# Test: Backend Support (if torch/jax available)
# ============================================================================

class TestBackendSupport:
    """Test multi-backend support (requires torch/jax)."""
    
    @pytest.mark.skipif(
        not _torch_available(),
        reason="PyTorch not available"
    )
    def test_torch_backend(self, linear_system, dt):
        """Test PyTorch backend."""
        import torch
        
        discretizer = Discretizer(linear_system, dt=dt, backend='torch')
        
        x = torch.tensor([1.0])
        u = torch.tensor([0.0])
        
        x_next = discretizer.step(x, u)
        
        assert isinstance(x_next, torch.Tensor)
        assert x_next.shape == (1,)
    
    @pytest.mark.skipif(
        not _jax_available(),
        reason="JAX not available"
    )
    def test_jax_backend(self, linear_system, dt):
        """Test JAX backend."""
        import jax.numpy as jnp
        
        # Use 'midpoint' instead of 'rk4' for JAX (Diffrax doesn't have rk4)
        discretizer = Discretizer(linear_system, dt=dt, backend='jax', method='midpoint')
        
        x = jnp.array([1.0])
        u = jnp.array([0.0])
        
        x_next = discretizer.step(x, u)
        
        assert isinstance(x_next, jnp.ndarray)
        assert x_next.shape == (1,)
    
    @pytest.mark.skipif(
        not _torch_available(),
        reason="PyTorch not available"
    )
    def test_backend_consistency(self, linear_system, dt):
        """Test that different backends give same numerical results."""
        x_np = np.array([1.0])
        u_np = np.array([0.5])
        
        disc_np = Discretizer(linear_system, dt=dt, backend='numpy', method='rk4')
        x_next_np = disc_np.step(x_np, u_np)
        
        # Compare with torch
        import torch
        disc_torch = Discretizer(linear_system, dt=dt, backend='torch', method='rk4')
        x_torch = torch.tensor([1.0])
        u_torch = torch.tensor([0.5])
        x_next_torch = disc_torch.step(x_torch, u_torch)
        
        assert_allclose(x_next_np, x_next_torch.numpy(), rtol=1e-6)


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests combining multiple features."""
    
    def test_discretize_and_simulate_trajectory(self, linear_system):
        """Test discretization and trajectory simulation."""
        dt = 0.01
        T = 100
        
        discretizer = Discretizer(linear_system, dt=dt, method='rk4')
        
        # Initial state
        x = np.array([1.0])
        u = np.array([0.0])
        
        # Simulate trajectory
        trajectory = [x.copy()]
        for _ in range(T):
            x = discretizer.step(x, u)
            trajectory.append(x.copy())
        
        trajectory = np.array(trajectory)
        
        # Check shape
        assert trajectory.shape == (T + 1, 1)
        
        # Should decay exponentially
        assert trajectory[-1, 0] < trajectory[0, 0]
        
        # Compare with analytical solution
        t_final = T * dt
        x_analytical = np.exp(-2.0 * t_final)
        assert_allclose(trajectory[-1, 0], x_analytical, rtol=0.01)
    
    def test_linearize_and_dlqr(self, linear_system, dt):
        """Test linearization followed by discrete LQR design."""
        discretizer = Discretizer(linear_system, dt=dt)
        
        # Linearize at equilibrium
        x_eq = np.array([0.0])
        u_eq = np.array([0.0])
        
        Ad, Bd = discretizer.linearize(x_eq, u_eq, method='exact')
        
        # Design discrete LQR
        Q = np.array([[10.0]])
        R = np.array([[1.0]])
        
        import scipy.linalg
        S = scipy.linalg.solve_discrete_are(Ad, Bd, Q, R)
        K = -np.linalg.solve(R + Bd.T @ S @ Bd, Bd.T @ S @ Ad)
        
        # K should be reasonable
        assert K.shape == (1, 1)
        assert K[0, 0] < 0  # Negative feedback for stability
    
    def test_method_convergence_with_decreasing_dt(self, linear_system):
        """Test that solution converges as dt decreases."""
        x0 = np.array([1.0])
        u = np.array([0.0])
        t_final = 1.0
        
        dt_values = [0.1, 0.05, 0.01, 0.005]
        results = []
        
        for dt in dt_values:
            discretizer = Discretizer(linear_system, dt=dt, method='euler')
            
            x = x0.copy()
            num_steps = int(t_final / dt)
            
            for _ in range(num_steps):
                x = discretizer.step(x, u)
            
            results.append(x[0])
        
        # Analytical solution
        x_analytical = np.exp(-2.0 * t_final)
        
        # Errors should decrease
        errors = [abs(r - x_analytical) for r in results]
        
        # Each error should be smaller than the previous
        for i in range(len(errors) - 1):
            assert errors[i+1] < errors[i]


# ============================================================================
# Performance Tests (Optional)
# ============================================================================

class TestPerformance:
    """Performance benchmarks (optional, can be slow)."""
    
    @pytest.mark.slow
    def test_step_performance(self, linear_system, dt, benchmark):
        """Benchmark single step performance."""
        discretizer = Discretizer(linear_system, dt=dt, method='rk4')
        
        x = np.array([1.0])
        u = np.array([0.0])
        
        # Benchmark
        result = benchmark(discretizer.step, x, u)
        
        assert result.shape == (1,)
    
    @pytest.mark.slow
    def test_linearize_performance(self, pendulum_system, dt, benchmark):
        """Benchmark linearization performance."""
        discretizer = Discretizer(pendulum_system, dt=dt)
        
        x_eq = np.array([0.0, 0.0])
        u_eq = np.array([0.0])
        
        # Benchmark
        Ad, Bd = benchmark(discretizer.linearize, x_eq, u_eq, method='exact')
        
        assert Ad.shape == (2, 2)
        assert Bd.shape == (2, 1)


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])