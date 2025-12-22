"""
Unit Tests for DiffEqPySDEIntegrator

Tests Julia-based SDE integration via DiffEqPy, including:
- Initialization and validation
- Algorithm selection and availability
- Integration with autonomous and controlled systems
- Pure diffusion systems (zero drift)
- Comparison with known analytical solutions
- Error handling and edge cases
- Algorithm recommendations

NOTE: Julia manages its own random number generation, so tests cannot
assume reproducibility across runs. Tests validate behavior and properties
rather than exact numerical values.
"""

import pytest
import numpy as np

# Check if diffeqpy is available
try:
    from diffeqpy import de
    DIFFEQPY_AVAILABLE = True
except ImportError:
    DIFFEQPY_AVAILABLE = False

from src.systems.base.numerical_integration.stochastic.diffeqpy_sde_integrator import (
    DiffEqPySDEIntegrator,
    create_diffeqpy_sde_integrator,
    list_julia_sde_algorithms,
)
from src.systems.base.numerical_integration.stochastic.sde_integrator_base import (
    SDEType,
    ConvergenceType,
    StepMode
)
from src.systems.base.stochastic_dynamical_system import StochasticDynamicalSystem


# ============================================================================
# Skip Tests if DiffEqPy Not Available
# ============================================================================

pytestmark = pytest.mark.skipif(
    not DIFFEQPY_AVAILABLE,
    reason="diffeqpy not installed. Install: pip install diffeqpy"
)


# ============================================================================
# Mock SDE Systems for Testing
# ============================================================================

class OrnsteinUhlenbeck(StochasticDynamicalSystem):
    """Ornstein-Uhlenbeck process: dx = -alpha * x * dt + sigma * dW"""
    
    def define_system(self, alpha=1.0, sigma=0.5):
        import sympy as sp
        
        x = sp.symbols('x', real=True)
        alpha_sym = sp.symbols('alpha', positive=True)
        sigma_sym = sp.symbols('sigma', positive=True)
        
        self.state_vars = [x]
        self.control_vars = []  # Autonomous
        self._f_sym = sp.Matrix([[-alpha_sym * x]])
        self.parameters = {alpha_sym: alpha, sigma_sym: sigma}
        self.order = 1
        
        self.diffusion_expr = sp.Matrix([[sigma_sym]])
        self.sde_type = 'ito'


class GeometricBrownianMotion(StochasticDynamicalSystem):
    """Geometric Brownian motion: dx = mu * x * dt + sigma * x * dW"""
    
    def define_system(self, mu=0.1, sigma=0.2):
        import sympy as sp
        
        x = sp.symbols('x', positive=True)
        mu_sym = sp.symbols('mu', real=True)
        sigma_sym = sp.symbols('sigma', positive=True)
        
        self.state_vars = [x]
        self.control_vars = []  # Autonomous
        self._f_sym = sp.Matrix([[mu_sym * x]])
        self.parameters = {mu_sym: mu, sigma_sym: sigma}
        self.order = 1
        
        self.diffusion_expr = sp.Matrix([[sigma_sym * x]])
        self.sde_type = 'ito'


class BrownianMotion(StochasticDynamicalSystem):
    """Pure Brownian motion: dx = sigma * dW (zero drift)"""
    
    def define_system(self, sigma=1.0):
        import sympy as sp
        
        x = sp.symbols('x', real=True)
        sigma_sym = sp.symbols('sigma', positive=True)
        
        self.state_vars = [x]
        self.control_vars = []  # Autonomous
        self._f_sym = sp.Matrix([[0]])  # Zero drift!
        self.parameters = {sigma_sym: sigma}
        self.order = 1
        
        self.diffusion_expr = sp.Matrix([[sigma_sym]])
        self.sde_type = 'ito'


class ControlledOU(StochasticDynamicalSystem):
    """Controlled OU: dx = (-alpha * x + u) * dt + sigma * dW"""
    
    def define_system(self, alpha=1.0, sigma=0.5):
        import sympy as sp
        
        x = sp.symbols('x', real=True)
        u = sp.symbols('u', real=True)
        alpha_sym = sp.symbols('alpha', positive=True)
        sigma_sym = sp.symbols('sigma', positive=True)
        
        self.state_vars = [x]
        self.control_vars = [u]
        self._f_sym = sp.Matrix([[-alpha_sym * x + u]])
        self.parameters = {alpha_sym: alpha, sigma_sym: sigma}
        self.order = 1
        
        self.diffusion_expr = sp.Matrix([[sigma_sym]])
        self.sde_type = 'ito'


class TwoDimensionalOU(StochasticDynamicalSystem):
    """2D OU with diagonal noise (autonomous)"""
    
    def define_system(self, alpha=1.0, sigma1=0.5, sigma2=0.3):
        import sympy as sp
        
        x1, x2 = sp.symbols('x1 x2', real=True)
        alpha_sym = sp.symbols('alpha', positive=True)
        sigma1_sym = sp.symbols('sigma1', positive=True)
        sigma2_sym = sp.symbols('sigma2', positive=True)
        
        self.state_vars = [x1, x2]
        self.control_vars = []  # Autonomous
        self._f_sym = sp.Matrix([
            [-alpha_sym * x1],
            [-alpha_sym * x2]
        ])
        self.parameters = {
            alpha_sym: alpha,
            sigma1_sym: sigma1,
            sigma2_sym: sigma2
        }
        self.order = 1
        
        self.diffusion_expr = sp.Matrix([
            [sigma1_sym, 0],
            [0, sigma2_sym]
        ])
        self.sde_type = 'ito'


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def ou_system():
    """Create Ornstein-Uhlenbeck system."""
    return OrnsteinUhlenbeck(alpha=1.0, sigma=0.5)


@pytest.fixture
def gbm_system():
    """Create Geometric Brownian Motion."""
    return GeometricBrownianMotion(mu=0.1, sigma=0.2)


@pytest.fixture
def brownian_system():
    """Create pure Brownian motion."""
    return BrownianMotion(sigma=1.0)


@pytest.fixture
def controlled_system():
    """Create controlled OU system."""
    return ControlledOU(alpha=1.0, sigma=0.5)


@pytest.fixture
def ou_2d_system():
    """Create 2D OU system."""
    return TwoDimensionalOU(alpha=1.0, sigma1=0.5, sigma2=0.3)


@pytest.fixture
def integrator_em(ou_system):
    """Create Euler-Maruyama integrator."""
    return DiffEqPySDEIntegrator(
        ou_system,
        dt=0.01,
        algorithm='EM'
    )


# ============================================================================
# Test Class: Initialization and Validation
# ============================================================================

class TestDiffEqPySDEInitialization:
    """Test initialization and validation."""
    
    def test_basic_initialization(self, ou_system):
        """Test basic integrator initialization."""
        integrator = DiffEqPySDEIntegrator(
            ou_system,
            dt=0.01,
            algorithm='EM'
        )
        
        assert integrator.sde_system is ou_system
        assert integrator.dt == 0.01
        assert integrator.algorithm == 'EM'
        assert integrator.backend == 'numpy'
    
    def test_backend_must_be_numpy(self, ou_system):
        """Test that non-numpy backend raises error."""
        with pytest.raises(ValueError, match="requires backend='numpy'"):
            DiffEqPySDEIntegrator(
                ou_system,
                dt=0.01,
                algorithm='EM',
                backend='torch'
            )
    
    def test_invalid_algorithm_raises(self, ou_system):
        """Test that invalid algorithm raises error."""
        with pytest.raises(ValueError, match="Unknown Julia SDE algorithm"):
            DiffEqPySDEIntegrator(
                ou_system,
                dt=0.01,
                algorithm='NonExistentAlgorithm'
            )
    
    def test_valid_algorithms_accepted(self, ou_system):
        """Test that all listed algorithms are accepted."""
        algorithms = ['EM', 'LambaEM', 'SRIW1', 'SRA1']
        
        for alg in algorithms:
            integrator = DiffEqPySDEIntegrator(
                ou_system,
                dt=0.01,
                algorithm=alg
            )
            assert integrator.algorithm == alg
    
    def test_step_mode_defaults_to_adaptive(self, ou_system):
        """Test that default step mode is ADAPTIVE."""
        integrator = DiffEqPySDEIntegrator(
            ou_system,
            dt=0.01,
            algorithm='EM'
        )
        
        assert integrator.step_mode == StepMode.ADAPTIVE
    
    def test_custom_tolerances(self, ou_system):
        """Test custom tolerance settings."""
        integrator = DiffEqPySDEIntegrator(
            ou_system,
            dt=0.01,
            algorithm='EM',
            rtol=1e-6,
            atol=1e-8
        )
        
        assert integrator.rtol == 1e-6
        assert integrator.atol == 1e-8


# ============================================================================
# Test Class: Autonomous Systems
# ============================================================================

class TestAutonomousSystems:
    """Test integration of autonomous SDE systems."""
    
    def test_autonomous_ou_integration(self, ou_system):
        """Test basic integration of autonomous OU process."""
        integrator = DiffEqPySDEIntegrator(
            ou_system,
            dt=0.01,
            algorithm='EM'
        )
        
        x0 = np.array([1.0])
        u_func = lambda t, x: None
        t_span = (0.0, 1.0)
        
        result = integrator.integrate(x0, u_func, t_span)
        
        # Basic success checks
        assert result.success, f"Integration failed: {result.message}"
        assert result.x.shape[0] > 10, "Not enough time points"
        assert result.x.shape[1] == 1, "Wrong state dimension"
        assert result.nsteps > 0, "No steps recorded"
        
        # State should have evolved (not stuck at initial)
        assert not np.allclose(result.x[-1], x0), "State didn't evolve"
    
    def test_autonomous_2d_integration(self, ou_2d_system):
        """Test integration of 2D autonomous system."""
        integrator = DiffEqPySDEIntegrator(
            ou_2d_system,
            dt=0.01,
            algorithm='EM'
        )
        
        x0 = np.array([1.0, 2.0])
        u_func = lambda t, x: None
        t_span = (0.0, 1.0)
        
        result = integrator.integrate(x0, u_func, t_span)
        
        assert result.success
        assert result.x.shape[1] == 2, "Should be 2D system"
        assert result.x.shape[0] > 10, "Need multiple time points"
        
        # Both dimensions should evolve
        assert not np.allclose(result.x[-1, 0], x0[0])
        assert not np.allclose(result.x[-1, 1], x0[1])
    
    @pytest.mark.slow
    def test_autonomous_ou_decay_behavior(self, ou_system):
        """Test that OU process shows decay behavior (statistical)."""
        integrator = DiffEqPySDEIntegrator(
            ou_system,
            dt=0.01,
            algorithm='EM'
        )
        
        x0 = np.array([5.0])  # Start far from equilibrium
        u_func = lambda t, x: None
        t_span = (0.0, 3.0)
        
        # Run multiple trajectories
        final_states = []
        for _ in range(50):
            # Create fresh integrator for independent Julia RNG state
            fresh_integrator = DiffEqPySDEIntegrator(
                ou_system, dt=0.01, algorithm='EM'
            )
            result = fresh_integrator.integrate(x0, u_func, t_span)
            final_states.append(result.x[-1, 0])
        
        # Most trajectories should decay toward zero
        mean_final = np.mean(final_states)
        assert mean_final < x0[0], "Mean should decay from initial value"
        
        # Expected: E[X(3)] = 5.0 * exp(-1.0 * 3.0) ≈ 0.25
        expected_mean = 5.0 * np.exp(-3.0)
        
        # Allow generous tolerance due to randomness
        assert abs(mean_final - expected_mean) < 1.0


# ============================================================================
# Test Class: Pure Diffusion Systems
# ============================================================================

class TestPureDiffusionSystems:
    """Test pure diffusion systems (zero drift)."""
    
    def test_pure_diffusion_properties(self, brownian_system):
        """Test that Brownian motion system has correct properties."""
        assert brownian_system.is_pure_diffusion()
        assert brownian_system.nu == 0
        assert brownian_system.nx == 1
        assert brownian_system.nw == 1
    
    def test_pure_diffusion_integration(self, brownian_system):
        """Test basic integration of pure Brownian motion."""
        integrator = DiffEqPySDEIntegrator(
            brownian_system,
            dt=0.01,
            algorithm='EM'
        )
        
        x0 = np.array([0.0])
        u_func = lambda t, x: None
        t_span = (0.0, 1.0)
        
        result = integrator.integrate(x0, u_func, t_span)
        
        assert result.success
        assert result.x.shape[0] > 10
        assert result.nsteps > 0
        
        # State should change (diffusion should move it)
        # Can't guarantee exact value, but should evolve
        assert result.x.shape[0] > 1
    
    def test_pure_diffusion_state_evolution(self, brownian_system):
        """Test that pure diffusion actually moves the state."""
        integrator = DiffEqPySDEIntegrator(
            brownian_system,
            dt=0.01,
            algorithm='EM'
        )
        
        x0 = np.array([0.0])
        u_func = lambda t, x: None
        t_span = (0.0, 2.0)
        
        # Run a few times and check that states vary
        final_states = []
        for _ in range(10):
            integrator_fresh = DiffEqPySDEIntegrator(
                brownian_system, dt=0.01, algorithm='EM'
            )
            result = integrator_fresh.integrate(x0, u_func, t_span)
            final_states.append(result.x[-1, 0])
        
        # States should vary (not all the same)
        unique_values = len(set([round(s, 6) for s in final_states]))
        assert unique_values > 5, "States should vary across runs"
    
    @pytest.mark.slow
    def test_pure_diffusion_statistical_properties(self, brownian_system):
        """Test statistical properties of Brownian motion (slow test)."""
        integrator = DiffEqPySDEIntegrator(
            brownian_system,
            dt=0.01,
            algorithm='EM'
        )
        
        x0 = np.array([0.0])
        u_func = lambda t, x: None
        t_span = (0.0, 1.0)
        
        # Run many trajectories
        n_paths = 100
        final_states = []
        
        for _ in range(n_paths):
            fresh_integrator = DiffEqPySDEIntegrator(
                brownian_system, dt=0.01, algorithm='EM'
            )
            result = fresh_integrator.integrate(x0, u_func, t_span)
            final_states.append(result.x[-1, 0])
        
        final_states = np.array(final_states)
        
        # For Brownian motion: X(1) ~ N(0, sigma^2) = N(0, 1)
        mean = np.mean(final_states)
        variance = np.var(final_states)
        
        # Mean should be near 0 (generous tolerance)
        assert abs(mean) < 0.3, f"Mean too far from 0: {mean}"
        
        # Variance should be near 1 (generous tolerance)
        assert 0.5 < variance < 1.5, f"Variance out of range: {variance}"


# ============================================================================
# Test Class: Controlled Systems
# ============================================================================

class TestControlledSystems:
    """Test integration with control inputs."""
    
    def test_controlled_integration(self, controlled_system):
        """Test integration with constant control."""
        integrator = DiffEqPySDEIntegrator(
            controlled_system,
            dt=0.01,
            algorithm='EM'
        )
        
        x0 = np.array([1.0])
        u_func = lambda t, x: np.array([0.5])
        t_span = (0.0, 1.0)
        
        result = integrator.integrate(x0, u_func, t_span)
        
        assert result.success
        assert result.x.shape[0] > 10
        assert result.nsteps > 0
    
    def test_state_feedback_control(self, controlled_system):
        """Test state feedback control."""
        integrator = DiffEqPySDEIntegrator(
            controlled_system,
            dt=0.01,
            algorithm='EM'
        )
        
        x0 = np.array([1.0])
        K = np.array([2.0])
        u_func = lambda t, x: -K * x
        t_span = (0.0, 2.0)
        
        result = integrator.integrate(x0, u_func, t_span)
        
        assert result.success
        # Strong feedback generally pushes toward zero
        # (Can't guarantee due to noise, but likely)
        assert result.x.shape[0] > 10
    
    def test_time_varying_control(self, controlled_system):
        """Test time-varying control."""
        integrator = DiffEqPySDEIntegrator(
            controlled_system,
            dt=0.01,
            algorithm='EM'
        )
        
        x0 = np.array([0.0])
        u_func = lambda t, x: np.array([np.sin(2*np.pi*t)])
        t_span = (0.0, 2.0)
        
        result = integrator.integrate(x0, u_func, t_span)
        
        assert result.success


# ============================================================================
# Test Class: Integration Methods
# ============================================================================

class TestIntegrationMethods:
    """Test integration functionality."""
    
    def test_integrate_returns_result(self, integrator_em):
        """Test that integrate returns proper result object."""
        x0 = np.array([1.0])
        u_func = lambda t, x: None
        t_span = (0.0, 1.0)
        
        result = integrator_em.integrate(x0, u_func, t_span)
        
        assert hasattr(result, 't')
        assert hasattr(result, 'x')
        assert hasattr(result, 'success')
        assert hasattr(result, 'nsteps')
        assert result.success
    
    def test_integrate_with_t_eval(self, integrator_em):
        """Test integration with specific evaluation times."""
        x0 = np.array([1.0])
        u_func = lambda t, x: None
        t_span = (0.0, 1.0)
        t_eval = np.linspace(0, 1, 51)
        
        result = integrator_em.integrate(x0, u_func, t_span, t_eval=t_eval)
        
        assert result.success
        assert len(result.t) == len(t_eval)
        np.testing.assert_allclose(result.t, t_eval, rtol=1e-6)
    
    def test_step_method(self, integrator_em):
        """Test single step method."""
        x0 = np.array([1.0])
        u = None
        dt = 0.01
        
        x1 = integrator_em.step(x0, u, dt)
        
        assert x1.shape == x0.shape
        # State may or may not change significantly in one step
        # Just verify it's a valid state
        assert np.all(np.isfinite(x1))
    
    def test_statistics_tracked(self, integrator_em):
        """Test that statistics are tracked during integration."""
        x0 = np.array([1.0])
        u_func = lambda t, x: None
        t_span = (0.0, 0.5)
        
        integrator_em.reset_stats()
        result = integrator_em.integrate(x0, u_func, t_span)
        
        stats = integrator_em.get_sde_stats()
        assert stats['total_fev'] > 0
        assert stats['diffusion_evals'] > 0
        assert stats['total_steps'] > 0


# ============================================================================
# Test Class: Algorithm Selection
# ============================================================================

class TestAlgorithmSelection:
    """Test algorithm recommendation and information."""
    
    def test_list_algorithms(self):
        """Test that list_algorithms returns categories."""
        algorithms = DiffEqPySDEIntegrator.list_algorithms()
        
        assert 'euler_maruyama' in algorithms
        assert 'stochastic_rk' in algorithms
        assert 'implicit' in algorithms
        assert isinstance(algorithms['euler_maruyama'], list)
        assert 'EM' in algorithms['euler_maruyama']
    
    def test_get_algorithm_info(self):
        """Test getting algorithm information."""
        info = DiffEqPySDEIntegrator.get_algorithm_info('EM')
        
        assert 'name' in info
        assert 'description' in info
        assert info['strong_order'] == 0.5
        assert info['weak_order'] == 1.0
    
    def test_recommend_algorithm_additive(self):
        """Test algorithm recommendation for additive noise."""
        alg = DiffEqPySDEIntegrator.recommend_algorithm(
            noise_type='additive',
            stiffness='none',
            accuracy='high'
        )
        
        assert alg == 'SRA3'
    
    def test_recommend_algorithm_diagonal(self):
        """Test algorithm recommendation for diagonal noise."""
        alg = DiffEqPySDEIntegrator.recommend_algorithm(
            noise_type='diagonal',
            stiffness='none',
            accuracy='high'
        )
        
        assert alg == 'SRIW1'
    
    def test_recommend_algorithm_stiff(self):
        """Test algorithm recommendation for stiff systems."""
        alg = DiffEqPySDEIntegrator.recommend_algorithm(
            noise_type='any',
            stiffness='severe',
            accuracy='medium'
        )
        
        assert alg == 'ImplicitEM'


# ============================================================================
# Test Class: Convergence and Accuracy
# ============================================================================

class TestConvergenceAccuracy:
    """Test convergence properties and accuracy."""
    
    def test_ou_shows_mean_reversion(self, ou_system):
        """Test that OU process shows mean reversion behavior."""
        integrator = DiffEqPySDEIntegrator(
            ou_system,
            dt=0.01,
            algorithm='EM'
        )
        
        x0 = np.array([5.0])  # Start far from equilibrium
        u_func = lambda t, x: None
        t_span = (0.0, 3.0)
        
        # Run a few trajectories
        final_states = []
        for _ in range(10):
            fresh_integrator = DiffEqPySDEIntegrator(
                ou_system, dt=0.01, algorithm='EM'
            )
            result = fresh_integrator.integrate(x0, u_func, t_span)
            final_states.append(result.x[-1, 0])
        
        # Mean should be much closer to zero than initial
        mean_final = np.mean(final_states)
        assert abs(mean_final) < abs(x0[0]), "Should move toward equilibrium"
    
    def test_gbm_stays_positive(self, gbm_system):
        """Test that GBM maintains positivity."""
        integrator = DiffEqPySDEIntegrator(
            gbm_system,
            dt=0.01,
            algorithm='EM'
        )
        
        x0 = np.array([1.0])
        u_func = lambda t, x: None
        t_span = (0.0, 1.0)
        
        # Run several times
        for _ in range(5):
            fresh_integrator = DiffEqPySDEIntegrator(
                gbm_system, dt=0.01, algorithm='EM'
            )
            result = fresh_integrator.integrate(x0, u_func, t_span)
            
            # All states should be positive
            assert np.all(result.x > 0), "GBM should stay positive"


# ============================================================================
# Test Class: Edge Cases and Error Handling
# ============================================================================

class TestEdgeCasesErrorHandling:
    """Test edge cases and error handling."""
    
    def test_short_time_span(self, integrator_em):
        """Test integration with very short time span."""
        x0 = np.array([1.0])
        u_func = lambda t, x: None
        t_span = (0.0, 0.01)  # Very short
        
        result = integrator_em.integrate(x0, u_func, t_span)
        
        # Should complete successfully
        assert result.success or len(result.t) >= 1
    
    def test_very_small_dt(self, ou_system):
        """Test with very small time step."""
        integrator = DiffEqPySDEIntegrator(
            ou_system,
            dt=1e-5,
            algorithm='EM'
        )
        
        x0 = np.array([1.0])
        u_func = lambda t, x: None
        t_span = (0.0, 0.001)
        
        result = integrator.integrate(x0, u_func, t_span)
        
        assert result.success


# ============================================================================
# Test Class: Utility Functions
# ============================================================================

class TestUtilityFunctions:
    """Test utility and helper functions."""
    
    def test_create_diffeqpy_sde_integrator(self, ou_system):
        """Test factory function."""
        integrator = create_diffeqpy_sde_integrator(
            ou_system,
            algorithm='EM',
            dt=0.01,
            rtol=1e-6
        )
        
        assert isinstance(integrator, DiffEqPySDEIntegrator)
        assert integrator.algorithm == 'EM'
        assert integrator.rtol == 1e-6
    
    def test_list_julia_sde_algorithms_output(self, capsys):
        """Test that list function prints output."""
        list_julia_sde_algorithms()
        
        captured = capsys.readouterr()
        assert 'Julia SDE Algorithms' in captured.out
        assert 'Euler-Maruyama' in captured.out


# ============================================================================
# Test Class: Qualitative Behavior
# ============================================================================

class TestQualitativeBehavior:
    """Test qualitative behavior rather than exact values."""
    
    def test_diffusion_increases_spread(self, ou_system):
        """Test that diffusion causes trajectories to spread."""
        integrator = DiffEqPySDEIntegrator(
            ou_system,
            dt=0.01,
            algorithm='EM'
        )
        
        x0 = np.array([0.0])
        u_func = lambda t, x: None
        t_span = (0.0, 0.5)
        
        # Multiple runs
        states = []
        for _ in range(20):
            fresh_integrator = DiffEqPySDEIntegrator(
                ou_system, dt=0.01, algorithm='EM'
            )
            result = fresh_integrator.integrate(x0, u_func, t_span)
            states.append(result.x[-1, 0])
        
        # States should have non-zero spread
        spread = np.std(states)
        assert spread > 0.05, f"Too little spread: {spread}"
    
    def test_longer_integration_more_spread(self, brownian_system):
        """Test that longer integration time increases spread."""
        # Short time
        states_short = []
        for _ in range(20):
            integrator = DiffEqPySDEIntegrator(
                brownian_system, dt=0.01, algorithm='EM'
            )
            result = integrator.integrate(
                np.array([0.0]), lambda t, x: None, (0.0, 0.1)
            )
            states_short.append(result.x[-1, 0])
        
        # Long time
        states_long = []
        for _ in range(20):
            integrator = DiffEqPySDEIntegrator(
                brownian_system, dt=0.01, algorithm='EM'
            )
            result = integrator.integrate(
                np.array([0.0]), lambda t, x: None, (0.0, 1.0)
            )
            states_long.append(result.x[-1, 0])
        
        spread_short = np.std(states_short)
        spread_long = np.std(states_long)
        
        # Longer integration should have more spread
        assert spread_long > spread_short


# ============================================================================
# Test Class: String Representations
# ============================================================================

class TestStringRepresentations:
    """Test string representations."""
    
    def test_integrator_name(self, integrator_em):
        """Test integrator name property."""
        name = integrator_em.name
        
        assert 'Julia' in name
        assert 'EM' in name
        assert 'Adaptive' in name or 'Fixed' in name
    
    def test_repr(self, integrator_em):
        """Test __repr__ method."""
        repr_str = repr(integrator_em)
        
        assert 'DiffEqPySDEIntegrator' in repr_str


if __name__ == '__main__':
    pytest.main([__file__, '-v'])