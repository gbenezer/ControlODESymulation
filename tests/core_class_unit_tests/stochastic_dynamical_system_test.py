"""
Unit Tests for StochasticDynamicalSystem

Tests cover:
1. System initialization and validation (subclassing pattern)
2. Integration with parent SymbolicDynamicalSystem
3. Drift evaluation (delegated to parent)
4. Diffusion evaluation (via DiffusionHandler)
5. Noise characterization (automatic analysis)
6. Solver recommendations
7. Constant noise optimization
8. Compilation and caching
9. Backend support (NumPy, PyTorch, JAX)
10. String-based sde_type API
11. Information and diagnostics
12. Best practices validation

NOTE: Technical debt exists here because a lot of warnings are raised
due to the inherent fact that for stochastic systems, parameters that
define diffusion relations and are not used in the drift or observation
dynamics raise a warning about being unused
TODO: Fix above issue
"""

import pytest
import numpy as np
import sympy as sp
from typing import Dict

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

from src.systems.base.stochastic_dynamical_system import StochasticDynamicalSystem
from src.systems.base.utils.stochastic.noise_analysis import NoiseType, SDEType
from src.systems.base.utils.stochastic.sde_validator import ValidationError


# ============================================================================
# Test System Classes (Following Best Practices)
# ============================================================================


class OrnsteinUhlenbeck(StochasticDynamicalSystem):
    """Ornstein-Uhlenbeck process with mean reversion (additive noise)."""
    
    def define_system(self, alpha=1.0, sigma=0.5):
        """
        Define O-U process: dx = -α*x*dt + σ*dW
        
        Parameters
        ----------
        alpha : float
            Mean reversion rate
        sigma : float
            Noise intensity
        """
        x = sp.symbols('x', real=True)
        u = sp.symbols('u', real=True)
        alpha_sym = sp.symbols('alpha', positive=True)
        sigma_sym = sp.symbols('sigma', positive=True)
        
        # Drift
        self.state_vars = [x]
        self.control_vars = [u]
        self._f_sym = sp.Matrix([[-alpha_sym * x + u]])
        self.parameters = {alpha_sym: alpha, sigma_sym: sigma}
        self.order = 1
        
        # Diffusion (additive)
        self.diffusion_expr = sp.Matrix([[sigma_sym]])
        self.sde_type = 'ito'


class GeometricBrownianMotion(StochasticDynamicalSystem):
    """Geometric Brownian motion (multiplicative noise)."""
    
    def define_system(self, mu=0.1, sigma=0.2):
        """
        Define GBM: dx = μ*x*dt + σ*x*dW
        
        Parameters
        ----------
        mu : float
            Drift coefficient
        sigma : float
            Volatility
        """
        x = sp.symbols('x', positive=True)
        u = sp.symbols('u', real=True)
        mu_sym, sigma_sym = sp.symbols('mu sigma', positive=True)
        
        # Drift
        self.state_vars = [x]
        self.control_vars = [u]
        self._f_sym = sp.Matrix([[mu_sym * x + u]])
        self.parameters = {mu_sym: mu, sigma_sym: sigma}
        self.order = 1
        
        # Multiplicative diffusion
        self.diffusion_expr = sp.Matrix([[sigma_sym * x]])


class DiagonalNoiseSystem(StochasticDynamicalSystem):
    """3D system with diagonal (independent) noise sources."""
    
    def define_system(self, param1_val=-1.0, sigma1=0.1, sigma2=0.2, sigma3=0.3):
        """
        Define system with independent noise per state.
        
        Parameters
        ----------
        param1_val : float
            Drift parameter
        sigma1, sigma2, sigma3 : float
            Noise intensities for each state (multiplicative coefficients)
        """
        x1, x2, x3 = sp.symbols('x1 x2 x3', real=True)
        u = sp.symbols('u', real=True)
        param1_sym = sp.symbols('param1', real=True)
        sigma1_sym, sigma2_sym, sigma3_sym = sp.symbols('sigma1 sigma2 sigma3', positive=True)
        
        # Drift
        self.state_vars = [x1, x2, x3]
        self.control_vars = [u]
        self._f_sym = sp.Matrix([x2, x3, param1_sym * x1 + u])
        
        # CRITICAL: Include ALL symbolic parameters (drift + diffusion)
        self.parameters = {
            param1_sym: param1_val,
            sigma1_sym: sigma1,
            sigma2_sym: sigma2,
            sigma3_sym: sigma3
        }
        self.order = 1
        
        # Diagonal diffusion using symbolic parameters
        self.diffusion_expr = sp.Matrix([
            [sigma1_sym * x1, 0, 0],
            [0, sigma2_sym * x2, 0],
            [0, 0, sigma3_sym * x3]
        ])
        self.sde_type = 'ito'


class StratonovichSystem(StochasticDynamicalSystem):
    """System using Stratonovich interpretation."""
    
    def define_system(self, sigma_val=0.5):
        """
        Define Stratonovich SDE.
        
        Parameters
        ----------
        sigma_val : float
            Noise intensity
        """
        x = sp.symbols('x')
        u = sp.symbols('u')
        sigma_sym = sp.symbols('sigma', positive=True)  # Add positive assumption
        
        self.state_vars = [x]
        self.control_vars = [u]
        self._f_sym = sp.Matrix([[-x + u]])
        
        # Use symbolic parameter, not hardcoded
        self.parameters = {sigma_sym: sigma_val}
        self.order = 1
        
        # Use symbolic parameter in diffusion
        self.diffusion_expr = sp.Matrix([[sigma_sym]])
        self.sde_type = 'stratonovich'  # String-based API

class AutonomousOrnsteinUhlenbeck(StochasticDynamicalSystem):
    """Autonomous Ornstein-Uhlenbeck process (no control input)."""
    
    def define_system(self, alpha=1.0, sigma=0.5):
        """
        Define autonomous O-U: dx = -α*x*dt + σ*dW
        
        Parameters
        ----------
        alpha : float
            Mean reversion rate
        sigma : float
            Noise intensity
        """
        x = sp.symbols('x', real=True)
        alpha_sym = sp.symbols('alpha', positive=True)
        sigma_sym = sp.symbols('sigma', positive=True)
        
        # Drift (no control)
        self.state_vars = [x]
        self.control_vars = []  # AUTONOMOUS
        self._f_sym = sp.Matrix([[-alpha_sym * x]])
        self.parameters = {alpha_sym: alpha, sigma_sym: sigma}
        self.order = 1
        
        # Diffusion (additive)
        self.diffusion_expr = sp.Matrix([[sigma_sym]])
        self.sde_type = 'ito'


class AutonomousGeometricBrownianMotion(StochasticDynamicalSystem):
    """Autonomous Geometric Brownian motion (no control)."""
    
    def define_system(self, mu=0.1, sigma=0.2):
        """
        Define autonomous GBM: dx = μ*x*dt + σ*x*dW
        
        Parameters
        ----------
        mu : float
            Drift coefficient
        sigma : float
            Volatility
        """
        x = sp.symbols('x', positive=True)
        mu_sym, sigma_sym = sp.symbols('mu sigma', positive=True)
        
        # Drift (no control)
        self.state_vars = [x]
        self.control_vars = []  # AUTONOMOUS
        self._f_sym = sp.Matrix([[mu_sym * x]])
        self.parameters = {mu_sym: mu, sigma_sym: sigma}
        self.order = 1
        
        # Multiplicative diffusion
        self.diffusion_expr = sp.Matrix([[sigma_sym * x]])
        self.sde_type = 'ito'


class Autonomous2DBrownianMotion(StochasticDynamicalSystem):
    """2D autonomous Brownian motion with independent noise (diagonal AND additive)."""
    
    def define_system(self, sigma1=0.5, sigma2=0.3):
        """
        Define 2D Brownian motion: dx = σ*dW
        
        This has BOTH diagonal AND additive noise (constant diagonal matrix).
        Additive takes priority in classification.
        
        Parameters
        ----------
        sigma1, sigma2 : float
            Noise intensities for each dimension
        """
        x1, x2 = sp.symbols('x1 x2', real=True)
        sigma1_sym, sigma2_sym = sp.symbols('sigma1 sigma2', positive=True)
        
        # Pure diffusion (no drift)
        self.state_vars = [x1, x2]
        self.control_vars = []  # AUTONOMOUS
        self._f_sym = sp.Matrix([[0], [0]])
        self.parameters = {sigma1_sym: sigma1, sigma2_sym: sigma2}
        self.order = 1
        
        # Diagonal AND additive diffusion (constant diagonal matrix)
        self.diffusion_expr = sp.Matrix([
            [sigma1_sym, 0],
            [0, sigma2_sym]
        ])
        self.sde_type = 'ito'


class AutonomousDiagonalMultiplicative(StochasticDynamicalSystem):
    """2D autonomous system with diagonal multiplicative noise (NOT additive)."""
    
    def define_system(self, sigma1=0.5, sigma2=0.3):
        """
        Define system with diagonal but state-dependent noise.
        
        This is DIAGONAL but NOT additive (state-dependent).
        
        Parameters
        ----------
        sigma1, sigma2 : float
            Noise intensity coefficients
        """
        x1, x2 = sp.symbols('x1 x2', real=True)
        sigma1_sym, sigma2_sym = sp.symbols('sigma1 sigma2', positive=True)
        
        # Simple autonomous drift
        self.state_vars = [x1, x2]
        self.control_vars = []  # AUTONOMOUS
        self._f_sym = sp.Matrix([[-x1], [-x2]])
        self.parameters = {sigma1_sym: sigma1, sigma2_sym: sigma2}
        self.order = 1
        
        # Diagonal BUT multiplicative diffusion (state-dependent diagonal)
        self.diffusion_expr = sp.Matrix([
            [sigma1_sym * x1, 0],
            [0, sigma2_sym * x2]
        ])
        self.sde_type = 'ito'


class ControlledDiagonalMultiplicative(StochasticDynamicalSystem):
    """2D controlled system with diagonal multiplicative noise (NOT additive)."""
    
    def define_system(self, sigma1=0.5, sigma2=0.3):
        """
        Define controlled system with diagonal but state-dependent noise.
        
        This is DIAGONAL but NOT additive (state-dependent).
        
        Parameters
        ----------
        sigma1, sigma2 : float
            Noise intensity coefficients
        """
        x1, x2 = sp.symbols('x1 x2', real=True)
        u = sp.symbols('u', real=True)
        sigma1_sym, sigma2_sym = sp.symbols('sigma1 sigma2', positive=True)
        
        # Controlled drift
        self.state_vars = [x1, x2]
        self.control_vars = [u]
        self._f_sym = sp.Matrix([[-x1 + u], [-x2]])
        self.parameters = {sigma1_sym: sigma1, sigma2_sym: sigma2}
        self.order = 1
        
        # Diagonal BUT multiplicative diffusion (state-dependent diagonal)
        self.diffusion_expr = sp.Matrix([
            [sigma1_sym * x1, 0],
            [0, sigma2_sym * x2]
        ])
        self.sde_type = 'ito'

# ============================================================================
# Test Initialization and Validation
# ============================================================================


class TestInitializationAndValidation:
    """Test system initialization and validation."""
    
    def test_successful_initialization(self):
        """Test that valid system initializes successfully."""
        system = OrnsteinUhlenbeck(alpha=2.0, sigma=0.3)
        
        assert system._initialized is True
        assert system.nx == 1
        assert system.nu == 1
        assert system.nw == 1
        assert system.is_stochastic is True
    
    def test_subclassing_pattern(self):
        """Test that define_system is called automatically."""
        system = OrnsteinUhlenbeck()
        
        # Should have attributes from define_system
        assert len(system.state_vars) == 1
        assert len(system.control_vars) == 1
        assert system._f_sym is not None
        assert system.diffusion_expr is not None
    
    def test_missing_diffusion_expr_error(self):
        """Test error when diffusion_expr not set in define_system."""
        
        class BadSystem(StochasticDynamicalSystem):
            def define_system(self):
                x, u = sp.symbols('x u')
                self.state_vars = [x]
                self.control_vars = [u]
                self._f_sym = sp.Matrix([[-x + u]])
                self.parameters = {}
                self.order = 1
                # Forgot to set self.diffusion_expr!
        
        with pytest.raises(ValueError, match="must set self.diffusion_expr"):
            BadSystem()
    
    def test_sde_type_default_ito(self):
        """Test default SDE type is Itô."""
        system = OrnsteinUhlenbeck()
        
        assert system.sde_type == SDEType.ITO
    
    def test_sde_type_string_ito(self):
        """Test sde_type accepts 'ito' string."""
        system = OrnsteinUhlenbeck()
        
        # System uses 'ito' string in define_system
        assert system.sde_type == SDEType.ITO
    
    def test_sde_type_string_stratonovich(self):
        """Test sde_type accepts 'stratonovich' string."""
        system = StratonovichSystem()
        
        # System uses 'stratonovich' string
        assert system.sde_type == SDEType.STRATONOVICH
    
    def test_sde_type_enum_still_works(self):
        """Test sde_type still accepts SDEType enum."""
        
        class EnumSystem(StochasticDynamicalSystem):
            def define_system(self):
                x, u = sp.symbols('x u')
                self.state_vars = [x]
                self.control_vars = [u]
                self._f_sym = sp.Matrix([[-x]])
                self.parameters = {}
                self.order = 1
                self.diffusion_expr = sp.Matrix([[0.5]])
                self.sde_type = SDEType.STRATONOVICH  # Enum
        
        system = EnumSystem()
        assert system.sde_type == SDEType.STRATONOVICH
    
    def test_sde_type_invalid_string(self):
        """Test error on invalid sde_type string."""
        
        class BadTypeSystem(StochasticDynamicalSystem):
            def define_system(self):
                x, u = sp.symbols('x u')
                self.state_vars = [x]
                self.control_vars = [u]
                self._f_sym = sp.Matrix([[-x]])
                self.parameters = {}
                self.order = 1
                self.diffusion_expr = sp.Matrix([[0.5]])
                self.sde_type = 'invalid'  # Bad value
        
        with pytest.raises(ValueError, match="Invalid sde_type"):
            BadTypeSystem()
    
    def test_validation_dimension_mismatch(self):
        """Test validation fails with dimension mismatch."""
        
        class BadDimSystem(StochasticDynamicalSystem):
            def define_system(self):
                x1, x2 = sp.symbols('x1 x2')
                u = sp.symbols('u')
                
                self.state_vars = [x1, x2]  # 2 states
                self.control_vars = [u]
                self._f_sym = sp.Matrix([[x2], [-x1]])
                self.parameters = {}
                self.order = 1
                
                # 3D diffusion for 2D system - MISMATCH!
                self.diffusion_expr = sp.Matrix([[0.1], [0.2], [0.3]])
        
        with pytest.raises(ValidationError):
            BadDimSystem()
    
    def test_validation_undefined_symbol_diffusion(self):
        """Test validation fails with undefined symbol in diffusion."""
        
        class UndefinedSymbolSystem(StochasticDynamicalSystem):
            def define_system(self):
                x, u = sp.symbols('x u')
                mystery = sp.symbols('mystery')  # Not in parameters!
                
                self.state_vars = [x]
                self.control_vars = [u]
                self._f_sym = sp.Matrix([[-x + u]])
                self.parameters = {}
                self.order = 1
                
                self.diffusion_expr = sp.Matrix([[mystery * x]])  # Undefined!
        
        with pytest.raises(ValidationError):
            UndefinedSymbolSystem()


# ============================================================================
# Test Noise Characterization
# ============================================================================


class TestNoiseCharacterization:
    """Test automatic noise structure analysis."""
    
    def test_additive_noise_detection(self):
        """Test detection of additive noise."""
        system = OrnsteinUhlenbeck(alpha=1.0, sigma=0.5)
        
        assert system.is_additive_noise()
        assert not system.is_multiplicative_noise()
        assert system.get_noise_type() == NoiseType.ADDITIVE
    
    def test_multiplicative_noise_detection(self):
        """Test detection of multiplicative noise."""
        system = GeometricBrownianMotion(mu=0.1, sigma=0.2)
        
        assert not system.is_additive_noise()
        assert system.is_multiplicative_noise()
        assert system.is_scalar_noise()  # nw=1, scalar takes priority
    
    def test_diagonal_noise_detection(self):
        """Test detection of diagonal noise."""
        system = DiagonalNoiseSystem()
        
        assert system.is_diagonal_noise()
        assert system.get_noise_type() == NoiseType.DIAGONAL
    
    def test_scalar_noise_detection(self):
        """Test detection of scalar noise."""
        system = OrnsteinUhlenbeck()
        
        assert system.is_scalar_noise()
        assert system.nw == 1
    
    def test_noise_dependencies(self):
        """Test detection of noise dependencies."""
        system = GeometricBrownianMotion()
        
        assert system.depends_on_state()
        assert not system.depends_on_control()
        assert not system.depends_on_time()


# ============================================================================
# Test Drift Evaluation (Delegated to Parent)
# ============================================================================


class TestDriftEvaluation:
    """Test drift evaluation (reuses parent class)."""
    
    def test_drift_numpy(self):
        """Test drift evaluation with NumPy."""
        system = OrnsteinUhlenbeck(alpha=1.0, sigma=0.5)
        
        x = np.array([2.0])
        u = np.array([0.5])
        
        f = system.drift(x, u, backend='numpy')
        
        assert isinstance(f, np.ndarray)
        # f = -alpha*x + u = -1.0*2.0 + 0.5 = -1.5
        np.testing.assert_almost_equal(f, np.array([-1.5]))
    
    def test_drift_callable_interface(self):
        """Test that __call__ evaluates drift."""
        system = OrnsteinUhlenbeck(alpha=1.0, sigma=0.5)
        
        x = np.array([2.0])
        u = np.array([0.5])
        
        # __call__ should evaluate drift
        f1 = system(x, u)
        f2 = system.drift(x, u)
        
        np.testing.assert_array_almost_equal(f1, f2)
    
    @pytest.mark.skipif(not torch_available, reason="PyTorch not installed")
    def test_drift_torch(self):
        """Test drift evaluation with PyTorch."""
        system = OrnsteinUhlenbeck(alpha=1.0, sigma=0.5)
        
        x = torch.tensor([2.0])
        u = torch.tensor([0.5])
        
        f = system.drift(x, u, backend='torch')
        
        assert isinstance(f, torch.Tensor)
        torch.testing.assert_close(f, torch.tensor([-1.5]))
    
    @pytest.mark.skipif(not jax_available, reason="JAX not installed")
    def test_drift_jax(self):
        """Test drift evaluation with JAX."""
        system = OrnsteinUhlenbeck(alpha=1.0, sigma=0.5)
        
        x = jnp.array([2.0])
        u = jnp.array([0.5])
        
        f = system.drift(x, u, backend='jax')
        
        assert isinstance(f, jnp.ndarray)
        np.testing.assert_almost_equal(f, np.array([-1.5]))


# ============================================================================
# Test Diffusion Evaluation
# ============================================================================


class TestDiffusionEvaluation:
    """Test diffusion evaluation via DiffusionHandler."""
    
    def test_diffusion_additive_numpy(self):
        """Test diffusion evaluation for additive noise."""
        system = OrnsteinUhlenbeck(alpha=1.0, sigma=0.5)
        
        x = np.array([2.0])
        u = np.array([0.5])
        
        g = system.diffusion(x, u, backend='numpy')
        
        assert isinstance(g, np.ndarray)
        assert g.shape == (1, 1)
        np.testing.assert_almost_equal(g, np.array([[0.5]]))
    
    def test_diffusion_multiplicative_numpy(self):
        """Test diffusion evaluation for multiplicative noise."""
        system = GeometricBrownianMotion(mu=0.1, sigma=0.2)
        
        x = np.array([2.0])
        u = np.array([0.0])
        
        g = system.diffusion(x, u, backend='numpy')
        
        assert g.shape == (1, 1)
        # g = sigma*x = 0.2*2.0 = 0.4
        np.testing.assert_almost_equal(g, np.array([[0.4]]))
    
    def test_diffusion_different_states(self):
        """Test that multiplicative diffusion varies with state."""
        system = GeometricBrownianMotion(mu=0.1, sigma=0.2)
        
        x1 = np.array([1.0])
        x2 = np.array([2.0])
        u = np.array([0.0])
        
        g1 = system.diffusion(x1, u)
        g2 = system.diffusion(x2, u)
        
        # Should be different (state-dependent)
        assert not np.allclose(g1, g2)
        assert g2[0, 0] == 2 * g1[0, 0]  # Linear in x
    
    def test_diffusion_diagonal_numpy(self):
        """Test diffusion evaluation for diagonal noise."""
        system = DiagonalNoiseSystem()
        
        x = np.array([1.0, 2.0, 3.0])
        u = np.array([0.0])
        
        g = system.diffusion(x, u, backend='numpy')
        
        expected = np.array([
            [0.1, 0, 0],
            [0, 0.4, 0],
            [0, 0, 0.9]
        ])
        
        assert g.shape == (3, 3)
        np.testing.assert_almost_equal(g, expected)
    
    @pytest.mark.skipif(not torch_available, reason="PyTorch not installed")
    def test_diffusion_torch(self):
        """Test diffusion evaluation with PyTorch."""
        system = OrnsteinUhlenbeck(alpha=1.0, sigma=0.5)
        
        x = torch.tensor([2.0])
        u = torch.tensor([0.5])
        
        g = system.diffusion(x, u, backend='torch')
        
        assert isinstance(g, torch.Tensor)
        assert g.shape == (1, 1)
        torch.testing.assert_close(g, torch.tensor([[0.5]]))
    
    @pytest.mark.skipif(not jax_available, reason="JAX not installed")
    def test_diffusion_jax(self):
        """Test diffusion evaluation with JAX."""
        system = OrnsteinUhlenbeck(alpha=1.0, sigma=0.5)
        
        x = jnp.array([2.0])
        u = jnp.array([0.5])
        
        g = system.diffusion(x, u, backend='jax')
        
        assert isinstance(g, jnp.ndarray)
        assert g.shape == (1, 1)


# ============================================================================
# Test Constant Noise Optimization
# ============================================================================


class TestConstantNoiseOptimization:
    """Test constant noise optimization for additive noise."""
    
    def test_get_constant_noise_additive(self):
        """Test getting constant noise matrix."""
        system = OrnsteinUhlenbeck(alpha=1.0, sigma=0.3)
        
        G = system.get_constant_noise('numpy')
        
        assert isinstance(G, np.ndarray)
        assert G.shape == (1, 1)
        np.testing.assert_almost_equal(G, np.array([[0.3]]))
    
    def test_constant_noise_matches_evaluation(self):
        """Test that constant noise matches diffusion evaluation."""
        system = OrnsteinUhlenbeck(alpha=1.0, sigma=0.5)
        
        G = system.get_constant_noise('numpy')
        
        # Evaluate at arbitrary point
        g = system.diffusion(np.array([100.0]), np.array([50.0]))
        
        # Should be identical (additive)
        np.testing.assert_array_almost_equal(g, G)
    
    def test_get_constant_noise_error_multiplicative(self):
        """Test error when requesting constant noise for multiplicative noise."""
        system = GeometricBrownianMotion(mu=0.1, sigma=0.2)
        
        with pytest.raises(ValueError, match="only valid for additive noise"):
            system.get_constant_noise('numpy')
    
    def test_can_optimize_for_additive(self):
        """Test checking optimization availability."""
        system_additive = OrnsteinUhlenbeck()
        system_multiplicative = GeometricBrownianMotion()
        
        assert system_additive.can_optimize_for_additive()
        assert not system_multiplicative.can_optimize_for_additive()


# ============================================================================
# Test Solver Recommendations
# ============================================================================


class TestSolverRecommendations:
    """Test automatic solver recommendations."""
    
    def test_recommend_solvers_additive_jax(self):
        """Test solver recommendations for additive noise (JAX)."""
        system = OrnsteinUhlenbeck()
        
        solvers = system.recommend_solvers('jax')
        
        assert len(solvers) > 0
        assert 'sea' in solvers or 'shark' in solvers or 'sra1' in solvers
    
    def test_recommend_solvers_multiplicative_torch(self):
        """Test solver recommendations for multiplicative noise (PyTorch)."""
        system = GeometricBrownianMotion()
        
        solvers = system.recommend_solvers('torch')
        
        assert len(solvers) > 0
    
    def test_recommend_solvers_diagonal_jax(self):
        """Test solver recommendations for diagonal noise."""
        system = DiagonalNoiseSystem()
        
        solvers = system.recommend_solvers('jax')
        
        assert len(solvers) > 0
        assert 'spark' in solvers or 'euler_heun' in solvers
    
    def test_recommend_solvers_all_backends(self):
        """Test solver recommendations for all backends."""
        system = OrnsteinUhlenbeck()
        
        jax_solvers = system.recommend_solvers('jax')
        torch_solvers = system.recommend_solvers('torch')
        julia_solvers = system.recommend_solvers('numpy')
        
        assert len(jax_solvers) > 0
        assert len(torch_solvers) > 0
        assert len(julia_solvers) > 0


# ============================================================================
# Test Optimization Opportunities
# ============================================================================


class TestOptimizationOpportunities:
    """Test optimization opportunity detection."""
    
    def test_optimization_flags_additive(self):
        """Test optimization flags for additive noise."""
        system = OrnsteinUhlenbeck()
        
        opts = system.get_optimization_opportunities()
        
        assert opts['precompute_diffusion']
        assert opts['cache_diffusion']
        assert opts['vectorize_easily']
    
    def test_optimization_flags_multiplicative(self):
        """Test optimization flags for multiplicative noise."""
        system = GeometricBrownianMotion()
        
        opts = system.get_optimization_opportunities()
        
        assert not opts['precompute_diffusion']
        assert not opts['cache_diffusion']
    
    def test_optimization_flags_diagonal(self):
        """Test optimization flags for diagonal noise."""
        system = DiagonalNoiseSystem()
        
        opts = system.get_optimization_opportunities()
        
        assert opts['use_diagonal_solver']
        assert opts['vectorize_easily']


# ============================================================================
# Test Compilation
# ============================================================================


class TestCompilation:
    """Test code generation and compilation."""
    
    def test_compile_diffusion_numpy(self):
        """Test compiling diffusion for NumPy."""
        system = OrnsteinUhlenbeck()
        
        timings = system.compile_diffusion(backends=['numpy'], verbose=False)
        
        assert 'numpy' in timings
        assert timings['numpy'] is not None
        assert isinstance(timings['numpy'], float)
        assert timings['numpy'] > 0
    
    def test_compile_all_backends(self):
        """Test compiling both drift and diffusion."""
        system = OrnsteinUhlenbeck()
        
        timings = system.compile_all(backends=['numpy'], verbose=False)
        
        assert 'numpy' in timings
        assert 'drift' in timings['numpy']
        assert 'diffusion' in timings['numpy']
        assert timings['numpy']['drift'] is not None
        assert timings['numpy']['diffusion'] is not None
    
    def test_reset_diffusion_cache(self):
        """Test resetting diffusion cache."""
        system = OrnsteinUhlenbeck()
        
        # Compile
        system.compile_diffusion(backends=['numpy'])
        assert system.diffusion_handler.is_compiled('numpy')
        
        # Reset
        system.reset_diffusion_cache(['numpy'])
        assert not system.diffusion_handler.is_compiled('numpy')
    
    def test_reset_all_caches(self):
        """Test resetting both drift and diffusion caches."""
        system = OrnsteinUhlenbeck()
        
        # Compile both
        system.compile_all(backends=['numpy'])
        
        # Reset all
        system.reset_all_caches(['numpy'])
        
        # Both should be cleared
        assert not system.diffusion_handler.is_compiled('numpy')


# ============================================================================
# Test Information and Diagnostics
# ============================================================================


class TestInformation:
    """Test information retrieval and diagnostics."""
    
    def test_get_info_structure(self):
        """Test structure of info dictionary."""
        system = OrnsteinUhlenbeck()
        
        info = system.get_info()
        
        assert 'system_type' in info
        assert 'is_stochastic' in info
        assert 'sde_type' in info
        assert 'dimensions' in info
        assert 'noise' in info
        assert 'recommended_solvers' in info
        assert 'optimization_opportunities' in info
    
    def test_get_info_dimensions(self):
        """Test dimension information."""
        system = DiagonalNoiseSystem()
        
        info = system.get_info()
        
        assert info['dimensions']['nx'] == 3
        assert info['dimensions']['nu'] == 1
        assert info['dimensions']['nw'] == 3
    
    def test_get_info_noise_characteristics(self):
        """Test noise characteristics in info."""
        system = OrnsteinUhlenbeck()
        
        info = system.get_info()
        
        assert info['noise']['type'] == 'additive'
        assert info['noise']['is_additive'] is True
        assert info['noise']['depends_on']['state'] is False
    
    def test_get_info_recommended_solvers(self):
        """Test solver recommendations in info."""
        system = OrnsteinUhlenbeck()
        
        info = system.get_info()
        
        assert 'jax' in info['recommended_solvers']
        assert 'torch' in info['recommended_solvers']
        assert 'julia' in info['recommended_solvers']
        assert len(info['recommended_solvers']['jax']) > 0
    
    def test_print_sde_info(self, capsys):
        """Test formatted SDE info printing."""
        system = OrnsteinUhlenbeck()
        
        system.print_sde_info()
        
        captured = capsys.readouterr()
        assert 'Stochastic Dynamical System' in captured.out
        assert 'OrnsteinUhlenbeck' in captured.out
        assert 'Noise Type:' in captured.out
        assert 'Recommended Solvers' in captured.out
        assert 'Optimization Opportunities' in captured.out


# ============================================================================
# Test String Representations
# ============================================================================


class TestStringRepresentations:
    """Test __repr__ and __str__ methods."""
    
    def test_repr(self):
        """Test __repr__ output."""
        system = OrnsteinUhlenbeck()
        
        repr_str = repr(system)
        
        assert 'OrnsteinUhlenbeck' in repr_str
        assert 'nx=1' in repr_str
        assert 'nw=1' in repr_str
        assert 'additive' in repr_str
        assert 'ito' in repr_str
    
    def test_str(self):
        """Test __str__ output."""
        system = DiagonalNoiseSystem()
        
        str_repr = str(system)
        
        assert 'DiagonalNoiseSystem' in str_repr
        assert '3 states' in str_repr
        assert '3 noise sources' in str_repr
        assert 'diagonal' in str_repr
    
    def test_str_singular_vs_plural(self):
        """Test proper singular/plural in __str__."""
        system_1d = OrnsteinUhlenbeck()
        system_3d = DiagonalNoiseSystem()
        
        str_1d = str(system_1d)
        str_3d = str(system_3d)
        
        assert '1 state,' in str_1d  # Singular
        assert '3 states' in str_3d  # Plural


# ============================================================================
# Test Conversion Utilities
# ============================================================================


class TestConversionUtilities:
    """Test conversion to deterministic system."""
    
    def test_to_deterministic(self):
        """Test extracting deterministic (drift-only) system."""
        system = OrnsteinUhlenbeck(alpha=2.0, sigma=0.5)
        
        det_system = system.to_deterministic()
        
        # Should have same drift
        assert det_system.nx == system.nx
        assert det_system.nu == system.nu
        
        # Evaluate at same point
        x = np.array([2.0])
        u = np.array([0.5])
        
        f_stochastic = system.drift(x, u)
        f_deterministic = det_system(x, u)
        
        np.testing.assert_array_almost_equal(f_stochastic, f_deterministic)


# ============================================================================
# Test Integration with Parent Class
# ============================================================================


class TestParentIntegration:
    """Test integration with SymbolicDynamicalSystem parent."""
    
    def test_inherits_backend_methods(self):
        """Test that backend methods are inherited."""
        system = OrnsteinUhlenbeck()
        
        # Should have parent's backend methods
        assert hasattr(system, 'set_default_backend')
        assert hasattr(system, 'to_device')
        assert hasattr(system, 'get_backend_info')
        assert hasattr(system, 'compile')
    
    def test_set_default_backend(self):
        """Test setting default backend (inherited)."""
        system = OrnsteinUhlenbeck()
        
        system.set_default_backend('numpy')
        assert system._default_backend == 'numpy'
    
    def test_equilibrium_management(self):
        """Test equilibrium management (inherited from parent)."""
        system = OrnsteinUhlenbeck(alpha=1.0, sigma=0.5)
        
        # Should have equilibria handler
        assert hasattr(system, 'equilibria')
        
        # Add equilibrium (x=0, u=0 is equilibrium for O-U)
        system.add_equilibrium(
            'origin',
            x_eq=np.array([0.0]),
            u_eq=np.array([0.0]),
            verify=True
        )
        
        x_eq = system.equilibria.get_x('origin')
        assert x_eq is not None
        np.testing.assert_almost_equal(x_eq, np.array([0.0]))
    
    def test_parent_forward_method_works(self):
        """Test that parent's forward() method still works."""
        system = OrnsteinUhlenbeck(alpha=1.0, sigma=0.5)
        
        x = np.array([1.0])
        u = np.array([0.0])
        
        # Parent's forward method
        f1 = system.forward(x, u)
        
        # Our drift method
        f2 = system.drift(x, u)
        
        np.testing.assert_array_almost_equal(f1, f2)


# ============================================================================
# Test Parameter Usage Best Practices
# ============================================================================


class TestParameterBestPractices:
    """Test that systems follow parameter best practices."""
    
    def test_all_constructor_params_used(self):
        """Test that all constructor parameters are used in dynamics."""
        system = OrnsteinUhlenbeck(alpha=2.0, sigma=0.3)
        
        # Check that parameters are in the symbolic expressions
        drift_symbols = system._f_sym.free_symbols
        diffusion_symbols = system.diffusion_expr.free_symbols
        
        param_symbols = set(system.parameters.keys())
        used_symbols = drift_symbols | diffusion_symbols
        
        # All parameters should be used
        assert param_symbols.issubset(used_symbols)
    
    def test_no_hardcoded_values_in_drift(self):
        """Test that drift uses symbolic parameters, not hardcoded values."""
        system = OrnsteinUhlenbeck(alpha=2.0, sigma=0.3)
        
        # Parameters dict should not be empty
        assert len(system.parameters) > 0
        
        # Check that alpha_sym and sigma_sym are in parameters
        param_names = {str(k) for k in system.parameters.keys()}
        assert 'alpha' in param_names
        assert 'sigma' in param_names
    
    def test_no_hardcoded_values_in_diffusion(self):
        """Test that diffusion uses symbolic parameters."""
        system = GeometricBrownianMotion(mu=0.1, sigma=0.2)
        
        # sigma should be a parameter symbol
        diffusion_symbols = system.diffusion_expr.free_symbols
        param_symbols = set(system.parameters.keys())
        
        # At least one parameter should appear in diffusion
        assert len(diffusion_symbols & param_symbols) > 0


# ============================================================================
# Test Edge Cases
# ============================================================================


class TestEdgeCases:
    """Test edge cases and special scenarios."""
    
    def test_single_state_single_noise(self):
        """Test minimal 1D system."""
        system = OrnsteinUhlenbeck()
        
        assert system.nx == 1
        assert system.nw == 1
        
        # Evaluate
        f = system.drift(np.array([1.0]), np.array([0.0]))
        g = system.diffusion(np.array([1.0]), np.array([0.0]))
        
        assert f is not None
        assert g is not None
    
    def test_rectangular_diffusion(self):
        """Test system with more states than noise sources."""
        
        class RectangularSystem(StochasticDynamicalSystem):
            def define_system(self):
                x1, x2, x3 = sp.symbols('x1 x2 x3')
                u = sp.symbols('u')
                
                self.state_vars = [x1, x2, x3]
                self.control_vars = [u]
                self._f_sym = sp.Matrix([[x2], [x3], [-x1 + u]])
                self.parameters = {}
                self.order = 1
                
                # (3, 2) diffusion matrix
                self.diffusion_expr = sp.Matrix([
                    [0.1, 0.0],
                    [0.0, 0.2],
                    [0.1, 0.1]
                ])
        
        system = RectangularSystem()
        
        assert system.nx == 3
        assert system.nw == 2
        
        g = system.diffusion(np.array([1.0, 2.0, 3.0]), np.array([0.0]))
        assert g.shape == (3, 2)
    
    def test_parameter_variation(self):
        """Test creating systems with different parameter values."""
        system1 = OrnsteinUhlenbeck(alpha=1.0, sigma=0.5)
        system2 = OrnsteinUhlenbeck(alpha=2.0, sigma=0.3)
        
        # Different parameter values
        assert system1.parameters != system2.parameters
        
        # Different numerical behavior
        x = np.array([1.0])
        u = np.array([0.0])
        
        f1 = system1.drift(x, u)
        f2 = system2.drift(x, u)
        
        assert not np.allclose(f1, f2)


# ============================================================================
# Test Integration Workflows
# ============================================================================


class TestIntegrationWorkflows:
    """Integration tests combining multiple features."""
    
    def test_full_workflow_additive(self):
        """Test complete workflow for additive noise system."""
        system = OrnsteinUhlenbeck(alpha=2.0, sigma=0.3)
        
        # 1. Check noise type
        assert system.is_additive_noise()
        assert system.get_noise_type() == NoiseType.ADDITIVE
        
        # 2. Get recommendations
        solvers = system.recommend_solvers('jax')
        assert len(solvers) > 0
        
        # 3. Optimize for additive noise
        assert system.can_optimize_for_additive()
        G = system.get_constant_noise('numpy')
        assert G.shape == (1, 1)
        
        # 4. Evaluate drift and diffusion
        x = np.array([2.0])
        u = np.array([0.5])
        
        f = system.drift(x, u)
        g = system.diffusion(x, u)
        
        # For additive, diffusion should match constant
        np.testing.assert_array_almost_equal(g, G)
        
        # 5. Get comprehensive info
        info = system.get_info()
        assert info['noise']['is_additive']
        assert info['system_type'] == 'StochasticDynamicalSystem'
        
        # 6. Compile for performance
        timings = system.compile_all(backends=['numpy'], verbose=False)
        assert timings['numpy']['drift'] is not None
        assert timings['numpy']['diffusion'] is not None
    
    def test_full_workflow_multiplicative(self):
        """Test complete workflow for multiplicative noise system."""
        system = GeometricBrownianMotion(mu=0.1, sigma=0.2)
        
        # 1. Check noise type
        assert system.is_multiplicative_noise()
        assert system.depends_on_state()
        
        # 2. Cannot optimize for additive
        assert not system.can_optimize_for_additive()
        
        with pytest.raises(ValueError):
            system.get_constant_noise('numpy')
        
        # 3. Get recommendations
        solvers = system.recommend_solvers('torch')
        assert len(solvers) > 0
        
        # 4. Evaluate at different states
        x1 = np.array([1.0])
        x2 = np.array([2.0])
        u = np.array([0.0])
        
        g1 = system.diffusion(x1, u)
        g2 = system.diffusion(x2, u)
        
        # Should be different (state-dependent)
        assert not np.allclose(g1, g2)
        
        # 5. Compile and check
        timings = system.compile_all(backends=['numpy'], verbose=False)
        assert timings['numpy']['diffusion'] is not None
        
        # 6. Get info
        info = system.get_info()
        assert info['noise']['is_multiplicative']
        assert len(info['optimization_opportunities']) > 0
    
    def test_simulation_components(self):
        """Test that system has components needed for simulation."""
        system = OrnsteinUhlenbeck(alpha=1.0, sigma=0.5)
        
        # Should have diffusion handler
        assert system.diffusion_handler is not None
        
        # Should have noise characteristics
        assert system.noise_characteristics is not None
        
        # Should be able to evaluate both terms
        x = np.array([1.0])
        u = np.array([0.0])
        
        f = system.drift(x, u)
        g = system.diffusion(x, u)
        
        assert f.shape == (1,)
        assert g.shape == (1, 1)


# ============================================================================
# Test Best Practices Examples
# ============================================================================


class TestBestPracticesExamples:
    """Test that example systems follow best practices."""
    
    def test_ornstein_uhlenbeck_best_practices(self):
        """Test O-U example follows all best practices."""
        system = OrnsteinUhlenbeck(alpha=2.0, sigma=0.3)
        
        # ✅ All constructor params used
        assert len(system.parameters) == 2
        
        # ✅ Parameters dict not empty
        assert system.parameters != {}
        
        # ✅ Uses symbolic parameters (not hardcoded)
        param_names = {str(k) for k in system.parameters.keys()}
        assert 'alpha' in param_names
        assert 'sigma' in param_names
        
        # ✅ Diffusion uses symbolic parameter
        diffusion_symbols = system.diffusion_expr.free_symbols
        assert len(diffusion_symbols & set(system.parameters.keys())) > 0
    
    def test_gbm_best_practices(self):
        """Test GBM example follows best practices."""
        system = GeometricBrownianMotion(mu=0.15, sigma=0.25)
        
        # ✅ All parameters used
        assert len(system.parameters) == 2
        
        # ✅ Both drift and diffusion use parameters
        drift_symbols = system._f_sym.free_symbols
        diffusion_symbols = system.diffusion_expr.free_symbols
        param_symbols = set(system.parameters.keys())
        
        # mu used in drift
        assert len(drift_symbols & param_symbols) > 0
        # sigma used in diffusion
        assert len(diffusion_symbols & param_symbols) > 0


# ============================================================================
# Test Invalid Systems (Anti-Patterns)
# ============================================================================


class TestInvalidSystems:
    """Test that anti-patterns are caught."""
    
    def test_unused_parameter_warning(self):
        """Test warning for unused parameters in constructor."""
        
        class UnusedParamSystem(StochasticDynamicalSystem):
            def define_system(self, alpha=1.0, unused=99.0):  # unused not used!
                x, u = sp.symbols('x u')
                alpha_sym = sp.symbols('alpha', positive=True)
                
                self.state_vars = [x]
                self.control_vars = [u]
                self._f_sym = sp.Matrix([[-alpha_sym * x + u]])
                self.parameters = {alpha_sym: alpha}  # unused not in dict!
                self.order = 1
                
                self.diffusion_expr = sp.Matrix([[0.5]])  # Hardcoded!
        
        # This should create a warning (if we add that check)
        # For now, it just works but is bad practice
        system = UnusedParamSystem(alpha=1.0, unused=999.0)
        assert system is not None  # Works but not ideal
    
    def test_empty_parameters_dict(self):
        """Test system with empty parameters (anti-pattern but valid)."""
        
        class EmptyParamsSystem(StochasticDynamicalSystem):
            def define_system(self):
                x, u = sp.symbols('x u')
                
                self.state_vars = [x]
                self.control_vars = [u]
                self._f_sym = sp.Matrix([[-x + u]])  # Implicit -1 coefficient
                self.parameters = {}  # Empty!
                self.order = 1
                
                self.diffusion_expr = sp.Matrix([[0.5]])  # Hardcoded!
        
        # Valid but not best practice
        system = EmptyParamsSystem()
        assert system.parameters == {}


# ============================================================================
# Test Class: Autonomous SDE Initialization
# ============================================================================


class TestAutonomousInitialization:
    """Test autonomous SDE initialization and validation."""
    
    def test_autonomous_ou_initialization(self):
        """Test autonomous O-U process initializes successfully."""
        system = AutonomousOrnsteinUhlenbeck(alpha=2.0, sigma=0.3)
        
        assert system._initialized is True
        assert system.nx == 1
        assert system.nu == 0  # Autonomous
        assert system.nw == 1
        assert len(system.control_vars) == 0
    
    def test_autonomous_gbm_initialization(self):
        """Test autonomous GBM initializes successfully."""
        system = AutonomousGeometricBrownianMotion(mu=0.1, sigma=0.2)
        
        assert system._initialized is True
        assert system.nu == 0
        assert system.is_stochastic is True
    
    def test_autonomous_2d_initialization(self):
        """Test 2D autonomous SDE initialization."""
        system = Autonomous2DBrownianMotion(sigma1=0.5, sigma2=0.3)
        
        assert system.nx == 2
        assert system.nu == 0
        assert system.nw == 2
        assert system.is_diagonal_noise()
    
    def test_autonomous_validation_passes(self):
        """Test that autonomous SDEs pass validation."""
        # Should not raise any errors
        system = AutonomousOrnsteinUhlenbeck(alpha=2.0, sigma=0.3)
        assert system._initialized is True


# ============================================================================
# Test Class: Autonomous SDE Drift Evaluation
# ============================================================================


class TestAutonomousDriftEvaluation:
    """Test drift evaluation for autonomous SDEs."""
    
    def test_drift_autonomous_no_u_arg(self):
        """Test drift evaluation without u argument."""
        system = AutonomousOrnsteinUhlenbeck(alpha=2.0, sigma=0.3)
        
        x = np.array([1.0])
        
        # Call without u
        f = system.drift(x)
        
        assert isinstance(f, np.ndarray)
        # f = -2*x = -2.0
        np.testing.assert_almost_equal(f, np.array([-2.0]))
    
    def test_drift_autonomous_u_none(self):
        """Test drift evaluation with u=None."""
        system = AutonomousOrnsteinUhlenbeck(alpha=2.0, sigma=0.3)
        
        x = np.array([1.0])
        
        # Explicitly pass u=None
        f = system.drift(x, u=None)
        
        np.testing.assert_almost_equal(f, np.array([-2.0]))
    
    def test_drift_autonomous_2d(self):
        """Test drift for 2D autonomous system."""
        system = Autonomous2DBrownianMotion(sigma1=0.5, sigma2=0.3)
        
        x = np.array([1.0, 2.0])
        
        f = system.drift(x)
        
        # Pure diffusion - drift is zero
        assert f.shape == (2,)
        np.testing.assert_array_almost_equal(f, np.array([0.0, 0.0]))
    
    @pytest.mark.skipif(not torch_available, reason="PyTorch not installed")
    def test_drift_autonomous_torch(self):
        """Test autonomous drift with PyTorch."""
        system = AutonomousOrnsteinUhlenbeck(alpha=2.0, sigma=0.3)
        
        x = torch.tensor([1.0])
        
        f = system.drift(x)
        
        assert isinstance(f, torch.Tensor)
        torch.testing.assert_close(f, torch.tensor([-2.0]))
    
    @pytest.mark.skipif(not jax_available, reason="JAX not installed")
    def test_drift_autonomous_jax(self):
        """Test autonomous drift with JAX."""
        system = AutonomousOrnsteinUhlenbeck(alpha=2.0, sigma=0.3)
        
        x = jnp.array([1.0])
        
        f = system.drift(x)
        
        assert isinstance(f, jnp.ndarray)
        np.testing.assert_almost_equal(f, np.array([-2.0]))
    
    def test_autonomous_callable_interface(self):
        """Test __call__ works for autonomous systems."""
        system = AutonomousOrnsteinUhlenbeck(alpha=2.0, sigma=0.3)
        
        x = np.array([1.0])
        
        # Both should work and give same result
        f1 = system(x)
        f2 = system.drift(x)
        
        np.testing.assert_array_almost_equal(f1, f2)
    
    def test_autonomous_rejects_control(self):
        """Test that autonomous SDE rejects control input."""
        system = AutonomousOrnsteinUhlenbeck(alpha=2.0, sigma=0.3)
        
        x = np.array([1.0])
        u = np.array([0.5])  # Should be rejected
        
        with pytest.raises(ValueError, match="does not accept control input|requires control input"):
            system.drift(x, u)


# ============================================================================
# Test Class: Autonomous SDE Diffusion Evaluation
# ============================================================================


class TestAutonomousDiffusionEvaluation:
    """Test diffusion evaluation for autonomous SDEs."""
    
    def test_diffusion_autonomous_additive(self):
        """Test diffusion for autonomous additive noise."""
        system = AutonomousOrnsteinUhlenbeck(alpha=2.0, sigma=0.3)
        
        x = np.array([1.0])
        
        # Call without u
        g = system.diffusion(x)
        
        assert isinstance(g, np.ndarray)
        assert g.shape == (1, 1)
        np.testing.assert_almost_equal(g, np.array([[0.3]]))
    
    def test_diffusion_autonomous_multiplicative(self):
        """Test diffusion for autonomous multiplicative noise."""
        system = AutonomousGeometricBrownianMotion(mu=0.1, sigma=0.2)
        
        x = np.array([2.0])
        
        g = system.diffusion(x)
        
        # g = sigma*x = 0.2*2.0 = 0.4
        assert g.shape == (1, 1)
        np.testing.assert_almost_equal(g, np.array([[0.4]]))
    
    def test_diffusion_autonomous_u_none(self):
        """Test diffusion with explicit u=None."""
        system = AutonomousOrnsteinUhlenbeck(alpha=2.0, sigma=0.3)
        
        x = np.array([1.0])
        
        g = system.diffusion(x, u=None)
        
        np.testing.assert_almost_equal(g, np.array([[0.3]]))
    
    def test_diffusion_autonomous_2d(self):
        """Test diffusion for 2D autonomous system."""
        system = Autonomous2DBrownianMotion(sigma1=0.5, sigma2=0.3)
        
        x = np.array([1.0, 2.0])
        
        g = system.diffusion(x)
        
        expected = np.array([
            [0.5, 0],
            [0, 0.3]
        ])
        
        assert g.shape == (2, 2)
        np.testing.assert_array_almost_equal(g, expected)
    
    @pytest.mark.skipif(not torch_available, reason="PyTorch not installed")
    def test_diffusion_autonomous_torch(self):
        """Test autonomous diffusion with PyTorch."""
        system = AutonomousOrnsteinUhlenbeck(alpha=2.0, sigma=0.3)
        
        x = torch.tensor([1.0])
        
        g = system.diffusion(x, backend='torch')
        
        assert isinstance(g, torch.Tensor)
        assert g.shape == (1, 1)
        torch.testing.assert_close(g, torch.tensor([[0.3]]))
    
    @pytest.mark.skipif(not jax_available, reason="JAX not installed")
    def test_diffusion_autonomous_jax(self):
        """Test autonomous diffusion with JAX."""
        system = AutonomousOrnsteinUhlenbeck(alpha=2.0, sigma=0.3)
        
        x = jnp.array([1.0])
        
        g = system.diffusion(x, backend='jax')
        
        assert isinstance(g, jnp.ndarray)
        assert g.shape == (1, 1)
    
    def test_autonomous_diffusion_rejects_control(self):
        """Test that autonomous diffusion rejects control input."""
        system = AutonomousOrnsteinUhlenbeck(alpha=2.0, sigma=0.3)
        
        x = np.array([1.0])
        u = np.array([0.5])  # Should be rejected
        
        # The updated diffusion() method checks nu and raises error
        with pytest.raises(ValueError, match="Autonomous system cannot take control input"):
            system.diffusion(x, u)


# ============================================================================
# Test Class: Autonomous SDE Noise Characterization
# ============================================================================


class TestAutonomousNoiseCharacterization:
    """Test noise characterization for autonomous SDEs."""
    
    def test_autonomous_additive_noise(self):
        """Test autonomous system with additive noise."""
        system = AutonomousOrnsteinUhlenbeck(alpha=2.0, sigma=0.3)
        
        assert system.is_additive_noise()
        assert not system.is_multiplicative_noise()
        assert system.get_noise_type() == NoiseType.ADDITIVE
    
    def test_autonomous_multiplicative_noise(self):
        """Test autonomous system with multiplicative noise."""
        system = AutonomousGeometricBrownianMotion(mu=0.1, sigma=0.2)
        
        assert system.is_multiplicative_noise()
        assert not system.is_additive_noise()
        assert system.is_scalar_noise()
    
    def test_autonomous_diagonal_additive(self):
        """Test autonomous system with diagonal additive noise."""
        system = Autonomous2DBrownianMotion(sigma1=0.5, sigma2=0.3)
        
        # This system is BOTH diagonal AND additive (constant diagonal matrix)
        assert system.is_diagonal_noise()
        assert system.is_additive_noise()
        
        # Additive classification takes priority (more specific optimization)
        noise_type = system.get_noise_type()
        assert noise_type == NoiseType.ADDITIVE
    
    def test_autonomous_diagonal_multiplicative(self):
        """Test autonomous system with diagonal multiplicative noise."""
        system = AutonomousDiagonalMultiplicative(sigma1=0.5, sigma2=0.3)
        
        # This system is diagonal but NOT additive (state-dependent)
        assert system.is_diagonal_noise()
        assert not system.is_additive_noise()
        assert system.is_multiplicative_noise()
        
        # Diagonal classification should be used (not additive)
        noise_type = system.get_noise_type()
        assert noise_type == NoiseType.DIAGONAL
    
    def test_autonomous_noise_dependencies(self):
        """Test noise dependencies for autonomous systems."""
        system_additive = AutonomousOrnsteinUhlenbeck()
        system_multiplicative = AutonomousGeometricBrownianMotion()
        
        # Additive: no dependencies
        assert not system_additive.depends_on_state()
        assert not system_additive.depends_on_control()
        assert not system_additive.depends_on_time()
        
        # Multiplicative: state dependency only
        assert system_multiplicative.depends_on_state()
        assert not system_multiplicative.depends_on_control()
        assert not system_multiplicative.depends_on_time()


# ============================================================================
# Test Class: Autonomous SDE Constant Noise Optimization
# ============================================================================


class TestAutonomousConstantNoise:
    """Test constant noise optimization for autonomous additive SDEs."""
    
    def test_get_constant_noise_autonomous(self):
        """Test getting constant noise for autonomous system."""
        system = AutonomousOrnsteinUhlenbeck(alpha=1.0, sigma=0.3)
        
        G = system.get_constant_noise('numpy')
        
        assert isinstance(G, np.ndarray)
        assert G.shape == (1, 1)
        np.testing.assert_almost_equal(G, np.array([[0.3]]))
    
    def test_autonomous_constant_noise_matches_evaluation(self):
        """Test constant noise matches diffusion evaluation for autonomous."""
        system = AutonomousOrnsteinUhlenbeck(alpha=1.0, sigma=0.5)
        
        G = system.get_constant_noise('numpy')
        
        # Evaluate at arbitrary points (should all be same for additive)
        g1 = system.diffusion(np.array([1.0]))
        g2 = system.diffusion(np.array([10.0]))
        g3 = system.diffusion(np.array([-5.0]))
        
        np.testing.assert_array_almost_equal(g1, G)
        np.testing.assert_array_almost_equal(g2, G)
        np.testing.assert_array_almost_equal(g3, G)
    
    def test_autonomous_can_optimize_for_additive(self):
        """Test optimization check for autonomous systems."""
        system_additive = AutonomousOrnsteinUhlenbeck()
        system_multiplicative = AutonomousGeometricBrownianMotion()
        
        assert system_additive.can_optimize_for_additive()
        assert not system_multiplicative.can_optimize_for_additive()


# ============================================================================
# Test Class: Autonomous SDE Solver Recommendations
# ============================================================================


class TestAutonomousSolverRecommendations:
    """Test solver recommendations for autonomous SDEs."""
    
    def test_autonomous_additive_recommendations(self):
        """Test solver recommendations for autonomous additive noise."""
        system = AutonomousOrnsteinUhlenbeck()
        
        solvers = system.recommend_solvers('jax')
        
        assert len(solvers) > 0
        # Should recommend additive-specialized solvers
        assert any(s in solvers for s in ['sea', 'shark', 'sra1'])
    
    def test_autonomous_multiplicative_recommendations(self):
        """Test solver recommendations for autonomous multiplicative noise."""
        system = AutonomousGeometricBrownianMotion()
        
        solvers = system.recommend_solvers('jax')
        
        assert len(solvers) > 0
    
    def test_autonomous_diagonal_additive_recommendations(self):
        """Test solver recommendations for autonomous diagonal additive noise."""
        system = Autonomous2DBrownianMotion()
        
        solvers = system.recommend_solvers('jax')
        
        assert len(solvers) > 0
        # System is ADDITIVE (constant) with diagonal structure
        # Additive takes priority, so we get additive-specialized solvers
        assert any(s in solvers for s in ['sea', 'shark', 'sra1'])
    
    def test_autonomous_diagonal_multiplicative_recommendations(self):
        """Test solver recommendations for diagonal multiplicative noise."""
        system = AutonomousDiagonalMultiplicative()
        
        solvers = system.recommend_solvers('jax')
        
        assert len(solvers) > 0
        # System is DIAGONAL (not additive), should get diagonal solvers
        assert 'spark' in solvers or 'euler_heun' in solvers
    
    def test_controlled_diagonal_multiplicative_recommendations(self):
        """Test solver recommendations for controlled diagonal multiplicative."""
        system = ControlledDiagonalMultiplicative()
        
        solvers = system.recommend_solvers('jax')
        
        assert len(solvers) > 0
        # Should recommend diagonal-optimized solvers
        assert 'spark' in solvers or 'euler_heun' in solvers


# ============================================================================
# Test Class: Autonomous SDE Information
# ============================================================================


class TestAutonomousInformation:
    """Test information retrieval for autonomous SDEs."""
    
    def test_get_info_autonomous(self):
        """Test get_info for autonomous system."""
        system = AutonomousOrnsteinUhlenbeck(alpha=2.0, sigma=0.3)
        
        info = system.get_info()
        
        assert info['dimensions']['nu'] == 0
        assert info['is_stochastic'] is True
        assert info['noise']['depends_on']['control'] is False
    
    def test_print_sde_info_autonomous(self, capsys):
        """Test formatted info printing for autonomous SDE."""
        system = AutonomousOrnsteinUhlenbeck(alpha=2.0, sigma=0.3)
        
        system.print_sde_info()
        
        captured = capsys.readouterr()
        assert 'AutonomousOrnsteinUhlenbeck' in captured.out
        assert 'nu=0' in captured.out
        assert 'additive' in captured.out
    
    def test_repr_autonomous(self):
        """Test __repr__ for autonomous system."""
        system = AutonomousOrnsteinUhlenbeck()
        
        repr_str = repr(system)
        
        assert 'AutonomousOrnsteinUhlenbeck' in repr_str
        assert 'nu=0' in repr_str
        assert 'additive' in repr_str
    
    def test_str_autonomous(self):
        """Test __str__ for autonomous system."""
        system = AutonomousOrnsteinUhlenbeck()
        
        str_repr = str(system)
        
        assert 'AutonomousOrnsteinUhlenbeck' in str_repr
        assert '0 control' in str_repr  # Singular for nu=0


# ============================================================================
# Test Class: Autonomous SDE Integration Workflows
# ============================================================================


class TestAutonomousIntegrationWorkflows:
    """Integration tests for autonomous SDEs."""
    
    def test_full_workflow_autonomous_additive(self):
        """Test complete workflow for autonomous additive noise."""
        system = AutonomousOrnsteinUhlenbeck(alpha=2.0, sigma=0.3)
        
        # 1. Check noise type
        assert system.is_additive_noise()
        assert system.get_noise_type() == NoiseType.ADDITIVE
        
        # 2. Get recommendations
        solvers = system.recommend_solvers('jax')
        assert len(solvers) > 0
        
        # 3. Optimize for additive noise
        assert system.can_optimize_for_additive()
        G = system.get_constant_noise('numpy')
        assert G.shape == (1, 1)
        
        # 4. Evaluate drift and diffusion (no control)
        x = np.array([2.0])
        
        f = system.drift(x)
        g = system.diffusion(x)
        
        # For additive, diffusion should match constant
        np.testing.assert_array_almost_equal(g, G)
        
        # 5. Get comprehensive info
        info = system.get_info()
        assert info['noise']['is_additive']
        assert info['dimensions']['nu'] == 0
    
    def test_full_workflow_autonomous_multiplicative(self):
        """Test complete workflow for autonomous multiplicative noise."""
        system = AutonomousGeometricBrownianMotion(mu=0.1, sigma=0.2)
        
        # 1. Check noise type
        assert system.is_multiplicative_noise()
        assert system.depends_on_state()
        
        # 2. Cannot optimize for additive
        assert not system.can_optimize_for_additive()
        
        # 3. Get recommendations
        solvers = system.recommend_solvers('jax')
        assert len(solvers) > 0
        
        # 4. Evaluate at different states (no control)
        x1 = np.array([1.0])
        x2 = np.array([2.0])
        
        g1 = system.diffusion(x1)
        g2 = system.diffusion(x2)
        
        # Should be different (state-dependent)
        assert not np.allclose(g1, g2)
        assert g2[0, 0] == 2 * g1[0, 0]  # Linear in x
        
        # 5. Get info
        info = system.get_info()
        assert info['noise']['is_multiplicative']
        assert info['dimensions']['nu'] == 0
    
    def test_autonomous_simulation_components(self):
        """Test autonomous system has all simulation components."""
        system = AutonomousOrnsteinUhlenbeck(alpha=1.0, sigma=0.5)
        
        # Should have diffusion handler
        assert system.diffusion_handler is not None
        
        # Should have noise characteristics
        assert system.noise_characteristics is not None
        
        # Should be able to evaluate both terms without control
        x = np.array([1.0])
        
        f = system.drift(x)
        g = system.diffusion(x)
        
        assert f.shape == (1,)
        assert g.shape == (1, 1)
        
        # Simulate one step
        dt = 0.01
        dW = np.random.randn(1) * np.sqrt(dt)
        dx = f * dt + g @ dW
        
        assert dx.shape == (1,)
    
    def test_autonomous_to_deterministic(self):
        """Test converting autonomous SDE to deterministic."""
        system = AutonomousOrnsteinUhlenbeck(alpha=2.0, sigma=0.5)
        
        det_system = system.to_deterministic()
        
        # Should have same drift, no control
        assert det_system.nx == system.nx
        assert det_system.nu == system.nu
        assert det_system.nu == 0  # Still autonomous
        
        # Evaluate at same point
        x = np.array([2.0])
        
        f_stochastic = system.drift(x)
        f_deterministic = det_system(x)  # No u needed
        
        np.testing.assert_array_almost_equal(f_stochastic, f_deterministic)


# ============================================================================
# Test Class: Controlled vs Autonomous SDEs
# ============================================================================


class TestControlledVsAutonomousSDEs:
    """Test differences between controlled and autonomous SDEs."""
    
    def test_controlled_has_control(self):
        """Test controlled SDE has control variables."""
        system = OrnsteinUhlenbeck()
        
        assert system.nu > 0
        assert len(system.control_vars) > 0
    
    def test_autonomous_has_no_control(self):
        """Test autonomous SDE has no control variables."""
        system = AutonomousOrnsteinUhlenbeck()
        
        assert system.nu == 0
        assert len(system.control_vars) == 0
    
    def test_controlled_requires_u_drift(self):
        """Test controlled system requires u for drift."""
        system = OrnsteinUhlenbeck()
        
        x = np.array([1.0])
        
        with pytest.raises(ValueError):
            system.drift(x)  # Missing u
    
    def test_autonomous_does_not_require_u_drift(self):
        """Test autonomous system doesn't require u for drift."""
        system = AutonomousOrnsteinUhlenbeck()
        
        x = np.array([1.0])
        
        # Should work without u
        f = system.drift(x)
        assert isinstance(f, np.ndarray)
    
    def test_controlled_requires_u_diffusion(self):
        """Test controlled system requires u for diffusion."""
        system = OrnsteinUhlenbeck()
        
        x = np.array([1.0])
        
        with pytest.raises(ValueError):
            system.diffusion(x)  # Missing u
    
    def test_autonomous_does_not_require_u_diffusion(self):
        """Test autonomous system doesn't require u for diffusion."""
        system = AutonomousOrnsteinUhlenbeck()
        
        x = np.array([1.0])
        
        # Should work without u
        g = system.diffusion(x)
        assert isinstance(g, np.ndarray)


# ============================================================================
# Test Class: Autonomous SDE Best Practices
# ============================================================================


class TestAutonomousBestPractices:
    """Test that autonomous SDE examples follow best practices."""
    
    def test_autonomous_ou_best_practices(self):
        """Test autonomous O-U follows best practices."""
        system = AutonomousOrnsteinUhlenbeck(alpha=2.0, sigma=0.3)
        
        # ✅ All constructor params used
        assert len(system.parameters) == 2
        
        # ✅ Parameters dict not empty
        assert system.parameters != {}
        
        # ✅ Uses symbolic parameters
        param_names = {str(k) for k in system.parameters.keys()}
        assert 'alpha' in param_names
        assert 'sigma' in param_names
        
        # ✅ Both drift and diffusion use parameters
        drift_symbols = system._f_sym.free_symbols
        diffusion_symbols = system.diffusion_expr.free_symbols
        param_symbols = set(system.parameters.keys())
        
        assert len(drift_symbols & param_symbols) > 0
        assert len(diffusion_symbols & param_symbols) > 0
    
    def test_autonomous_gbm_best_practices(self):
        """Test autonomous GBM follows best practices."""
        system = AutonomousGeometricBrownianMotion(mu=0.15, sigma=0.25)
        
        # ✅ All parameters used
        assert len(system.parameters) == 2
        
        # ✅ Both mu and sigma are parameters
        param_names = {str(k) for k in system.parameters.keys()}
        assert 'mu' in param_names
        assert 'sigma' in param_names


# ============================================================================
# Test Class: Autonomous SDE Compilation
# ============================================================================


class TestAutonomousCompilation:
    """Test compilation for autonomous SDEs."""
    
    def test_compile_autonomous_drift_and_diffusion(self):
        """Test compiling autonomous SDE."""
        system = AutonomousOrnsteinUhlenbeck()
        
        timings = system.compile_all(backends=['numpy'], verbose=False)
        
        assert 'numpy' in timings
        assert 'drift' in timings['numpy']
        assert 'diffusion' in timings['numpy']
        assert timings['numpy']['drift'] is not None
        assert timings['numpy']['diffusion'] is not None
    
    def test_compile_autonomous_diffusion_only(self):
        """Test compiling only diffusion for autonomous system."""
        system = AutonomousOrnsteinUhlenbeck()
        
        timings = system.compile_diffusion(backends=['numpy'], verbose=False)
        
        assert 'numpy' in timings
        assert timings['numpy'] is not None


# ============================================================================
# Test Class: Diagonal Noise Classification
# ============================================================================

class TestDiagonalNoiseClassification:
    """Test diagonal noise classification edge cases."""
    
    def test_diagonal_additive_classification(self):
        """Test that constant diagonal is classified as ADDITIVE."""
        system = Autonomous2DBrownianMotion(sigma1=0.5, sigma2=0.3)
        
        # Constant diagonal matrix is both diagonal and additive
        assert system.is_diagonal_noise()
        assert system.is_additive_noise()
        
        # Additive takes priority (more specific)
        assert system.get_noise_type() == NoiseType.ADDITIVE
    
    def test_diagonal_multiplicative_autonomous_classification(self):
        """Test that state-dependent diagonal is classified as DIAGONAL."""
        system = AutonomousDiagonalMultiplicative(sigma1=0.5, sigma2=0.3)
        
        # State-dependent diagonal is diagonal but not additive
        assert system.is_diagonal_noise()
        assert not system.is_additive_noise()
        
        # Diagonal classification used
        assert system.get_noise_type() == NoiseType.DIAGONAL
    
    def test_diagonal_multiplicative_controlled_classification(self):
        """Test that controlled diagonal multiplicative is DIAGONAL."""
        system = ControlledDiagonalMultiplicative(sigma1=0.5, sigma2=0.3)
        
        # State-dependent diagonal is diagonal but not additive
        assert system.is_diagonal_noise()
        assert not system.is_additive_noise()
        
        # Diagonal classification used
        assert system.get_noise_type() == NoiseType.DIAGONAL
    
    def test_diagonal_additive_gets_additive_optimizations(self):
        """Test that diagonal additive can use additive optimizations."""
        system = Autonomous2DBrownianMotion(sigma1=0.5, sigma2=0.3)
        
        # Can use additive optimization (precompute)
        assert system.can_optimize_for_additive()
        
        G = system.get_constant_noise('numpy')
        assert G.shape == (2, 2)
        
        # Verify it's diagonal
        assert G[0, 1] == 0.0
        assert G[1, 0] == 0.0
    
    def test_diagonal_multiplicative_cannot_precompute(self):
        """Test that diagonal multiplicative cannot precompute."""
        system = AutonomousDiagonalMultiplicative(sigma1=0.5, sigma2=0.3)
        
        # Cannot use additive optimization
        assert not system.can_optimize_for_additive()
        
        # But can use diagonal optimization
        opts = system.get_optimization_opportunities()
        assert opts['use_diagonal_solver']

# ============================================================================
# Test Class: New Equilibrium API Methods for SDEs
# ============================================================================

class TestSDEEquilibriumAPIMethods:
    """Test equilibrium convenience methods on StochasticDynamicalSystem"""

    def test_set_default_equilibrium_sde(self):
        """Test set_default_equilibrium for SDE systems"""
        system = OrnsteinUhlenbeck(alpha=1.0, sigma=0.5)
        
        # For O-U, equilibrium at x=0, u=0
        system.add_equilibrium("zero", np.array([0.0]), np.array([0.0]), verify=True)
        
        result = system.set_default_equilibrium("zero")
        
        assert system.equilibria._default == "zero"
        assert result is system  # Returns self for chaining

    def test_set_default_equilibrium_autonomous_sde(self):
        """Test set_default_equilibrium for autonomous SDE"""
        system = AutonomousOrnsteinUhlenbeck(alpha=2.0, sigma=0.3)
        
        system.add_equilibrium("zero", np.array([0.0]), np.array([]), verify=True)
        
        result = system.set_default_equilibrium("zero")
        
        assert system.equilibria._default == "zero"
        assert result is system

    def test_get_equilibrium_sde_default_backend(self):
        """Test get_equilibrium uses system default backend for SDE"""
        system = OrnsteinUhlenbeck()
        system.add_equilibrium("test", np.array([0.0]), np.array([0.0]), verify=False)
        
        x_eq, u_eq = system.get_equilibrium("test")
        
        assert isinstance(x_eq, np.ndarray)
        assert isinstance(u_eq, np.ndarray)

    @pytest.mark.skipif(not torch_available, reason="PyTorch not installed")
    def test_get_equilibrium_sde_explicit_backend(self):
        """Test get_equilibrium with explicit backend for SDE"""
        system = OrnsteinUhlenbeck()
        system.add_equilibrium("test", np.array([0.0]), np.array([0.0]), verify=False)
        
        x_eq, u_eq = system.get_equilibrium("test", backend="torch")
        
        assert isinstance(x_eq, torch.Tensor)
        assert isinstance(u_eq, torch.Tensor)

    def test_list_equilibria_sde(self):
        """Test list_equilibria convenience method for SDE"""
        system = OrnsteinUhlenbeck()
        system.add_equilibrium("eq1", np.array([0.0]), np.array([0.0]), verify=False)
        system.add_equilibrium("eq2", np.array([0.5]), np.array([0.5]), verify=False)
        
        names = system.list_equilibria()
        
        assert len(names) == 3  # origin + eq1 + eq2
        assert all(name in names for name in ["origin", "eq1", "eq2"])

    def test_remove_equilibrium_sde(self):
        """Test remove_equilibrium for SDE"""
        system = OrnsteinUhlenbeck()
        system.add_equilibrium("temp", np.array([0.0]), np.array([0.0]), verify=False)
        
        system.remove_equilibrium("temp")
        
        assert "temp" not in system.list_equilibria()

    def test_cannot_remove_origin_sde(self):
        """Test that origin cannot be removed from SDE"""
        system = OrnsteinUhlenbeck()
        
        with pytest.raises(ValueError, match="Cannot remove origin"):
            system.remove_equilibrium("origin")

    def test_get_equilibrium_metadata_sde(self):
        """Test get_equilibrium_metadata for SDE"""
        system = OrnsteinUhlenbeck()
        system.add_equilibrium(
            "test",
            np.array([0.0]),
            np.array([0.0]),
            verify=False,
            stability="stable",
            description="Zero equilibrium"
        )
        
        meta = system.get_equilibrium_metadata("test")
        
        assert meta["stability"] == "stable"
        assert meta["description"] == "Zero equilibrium"

    def test_autonomous_sde_equilibrium_empty_control(self):
        """Test equilibrium with empty control for autonomous SDE"""
        system = AutonomousOrnsteinUhlenbeck(alpha=2.0, sigma=0.3)
        
        system.add_equilibrium(
            "zero",
            np.array([0.0]),
            np.array([]),  # Empty control
            verify=True
        )
        
        x_eq, u_eq = system.get_equilibrium("zero")
        assert x_eq.shape == (1,)
        assert u_eq.shape == (0,)


# ============================================================================
# Test Class: Linearization with Equilibrium Names for SDEs
# ============================================================================


class TestSDELinearizationWithEquilibriumNames:
    """Test linearization methods accepting equilibrium names for SDEs"""

    def test_linearized_dynamics_with_name_sde(self):
        """Test linearized_dynamics accepts equilibrium name for SDE"""
        system = OrnsteinUhlenbeck(alpha=2.0, sigma=0.5)
        
        # For O-U: equilibrium at origin
        system.add_equilibrium("zero", np.array([0.0]), np.array([0.0]), verify=True)
        
        A, B = system.linearized_dynamics("zero")
        
        assert A.shape == (1, 1)
        assert B.shape == (1, 1)
        # A = -alpha = -2.0, B = 1.0
        np.testing.assert_almost_equal(A, np.array([[-2.0]]))
        np.testing.assert_almost_equal(B, np.array([[1.0]]))

    def test_linearized_dynamics_symbolic_with_name_sde(self):
        """Test linearized_dynamics_symbolic accepts equilibrium name for SDE"""
        system = OrnsteinUhlenbeck(alpha=2.0, sigma=0.5)
        system.add_equilibrium("zero", np.array([0.0]), np.array([0.0]), verify=True)
        
        A_sym, B_sym = system.linearized_dynamics_symbolic("zero")
        
        assert isinstance(A_sym, sp.Matrix)
        assert isinstance(B_sym, sp.Matrix)
        
        # Convert and check
        A_np = np.array(A_sym, dtype=float)
        np.testing.assert_almost_equal(A_np, np.array([[-2.0]]))

    def test_linearized_dynamics_autonomous_sde_with_name(self):
        """Test linearization with name for autonomous SDE"""
        system = AutonomousOrnsteinUhlenbeck(alpha=2.0, sigma=0.3)
        system.add_equilibrium("zero", np.array([0.0]), np.array([]), verify=True)
        
        A, B = system.linearized_dynamics("zero")
        
        assert A.shape == (1, 1)
        assert B.shape == (1, 0)  # Empty B for autonomous


# ============================================================================
# Test Class: SDE Config Dict Updates
# ============================================================================


class TestSDEConfigDictUpdates:
    """Test get_config_dict includes equilibria information for SDEs"""

    def test_sde_config_dict_includes_equilibria_list(self):
        """Test that SDE config dict includes list of equilibria names"""
        system = OrnsteinUhlenbeck()
        system.add_equilibrium("eq1", np.array([0.0]), np.array([0.0]), verify=False)
        
        config = system.get_config_dict()
        
        assert "equilibria" in config
        assert "eq1" in config["equilibria"]
        assert "origin" in config["equilibria"]

    def test_sde_config_dict_includes_default_equilibrium(self):
        """Test that SDE config dict includes default equilibrium name"""
        system = OrnsteinUhlenbeck()
        
        config = system.get_config_dict()
        
        assert "default_equilibrium" in config
        assert config["default_equilibrium"] == "origin"

    def test_sde_config_dict_includes_sde_info(self):
        """Test that SDE config dict includes stochastic-specific info"""
        system = OrnsteinUhlenbeck(alpha=2.0, sigma=0.5)
        
        config = system.get_config_dict()
        
        # Should have base info
        assert "class_name" in config
        assert "nx" in config
        assert "nu" in config
        
        # Should be marked as stochastic
        assert config["class_name"] == "OrnsteinUhlenbeck"


# ============================================================================
# Test Class: SDE Equilibrium Handler Dimension Updates
# ============================================================================


class TestSDEEquilibriumHandlerDimensionUpdates:
    """Test that equilibrium handler dimensions update correctly for SDEs"""

    def test_sde_equilibrium_handler_dimensions_updated(self):
        """Test equilibrium handler gets correct dimensions after SDE init"""
        system = OrnsteinUhlenbeck()
        
        assert system.equilibria.nx == 1
        assert system.equilibria.nu == 1

    def test_autonomous_sde_equilibrium_handler_dimensions(self):
        """Test equilibrium handler dimensions for autonomous SDE"""
        system = AutonomousOrnsteinUhlenbeck()
        
        assert system.equilibria.nx == 1
        assert system.equilibria.nu == 0

    def test_sde_equilibrium_handler_origin_correct_shape(self):
        """Test that origin has correct shape after SDE dimension update"""
        system = OrnsteinUhlenbeck()
        
        x_eq = system.equilibria.get_x("origin")
        u_eq = system.equilibria.get_u("origin")
        
        assert x_eq.shape == (1,)
        assert u_eq.shape == (1,)
        np.testing.assert_array_almost_equal(x_eq, np.zeros(1))
        np.testing.assert_array_almost_equal(u_eq, np.zeros(1))

    def test_autonomous_sde_origin_empty_control(self):
        """Test that autonomous SDE origin has empty control"""
        system = AutonomousOrnsteinUhlenbeck()
        
        u_eq = system.equilibria.get_u("origin")
        
        assert u_eq.shape == (0,)

    def test_diagonal_sde_equilibrium_dimensions(self):
        """Test equilibrium handler dimensions for multi-dimensional SDE"""
        system = DiagonalNoiseSystem()
        
        assert system.equilibria.nx == 3
        assert system.equilibria.nu == 1
        
        x_eq = system.equilibria.get_x("origin")
        u_eq = system.equilibria.get_u("origin")
        
        assert x_eq.shape == (3,)
        assert u_eq.shape == (1,)


# ============================================================================
# Test Class: SDE Equilibrium Verification Integration
# ============================================================================


class TestSDEEquilibriumVerification:
    """Test equilibrium verification for stochastic systems"""

    def test_sde_equilibrium_verification_considers_drift_only(self):
        """Test that equilibrium verification uses drift only (not diffusion)"""
        system = OrnsteinUhlenbeck(alpha=1.0, sigma=0.5)
        
        # x=0, u=0 is equilibrium for drift (diffusion doesn't affect this)
        system.add_equilibrium(
            "zero",
            np.array([0.0]),
            np.array([0.0]),
            verify=True,
            tol=1e-10
        )
        
        assert "zero" in system.list_equilibria()
        meta = system.get_equilibrium_metadata("zero")
        assert meta.get("verified") is True

    def test_sde_invalid_equilibrium_warns(self):
        """Test warning for invalid equilibrium in SDE"""
        system = OrnsteinUhlenbeck(alpha=2.0, sigma=0.5)
        
        # x=1, u=0 is NOT equilibrium: -2*1 + 0 = -2 ≠ 0
        with pytest.warns(UserWarning, match="may not be valid"):
            system.add_equilibrium(
                "invalid",
                np.array([1.0]),
                np.array([0.0]),
                verify=True
            )

    def test_autonomous_sde_equilibrium_verification(self):
        """Test equilibrium verification for autonomous SDE"""
        system = AutonomousOrnsteinUhlenbeck(alpha=2.0, sigma=0.3)
        
        # x=0 is equilibrium: -2*0 = 0
        system.add_equilibrium(
            "zero",
            np.array([0.0]),
            np.array([]),
            verify=True,
            tol=1e-10
        )
        
        meta = system.get_equilibrium_metadata("zero")
        assert meta.get("verified") is True


# ============================================================================
# Test Class: SDE Linearization at Equilibria
# ============================================================================


class TestSDELinearizationAtEquilibria:
    """Test linearizing SDEs at stored equilibria"""

    def test_linearize_at_stored_equilibrium(self):
        """Test linearizing SDE at a stored equilibrium point"""
        system = OrnsteinUhlenbeck(alpha=2.0, sigma=0.5)
        system.add_equilibrium("zero", np.array([0.0]), np.array([0.0]), verify=True)
        
        # Linearize using equilibrium name
        A, B = system.linearized_dynamics("zero")
        
        # Linearize using values directly
        x_eq, u_eq = system.get_equilibrium("zero")
        A_direct, B_direct = system.linearized_dynamics(x_eq, u_eq)
        
        # Should be identical
        np.testing.assert_array_almost_equal(A, A_direct)
        np.testing.assert_array_almost_equal(B, B_direct)

    def test_linearize_gbm_at_equilibrium(self):
        """Test linearizing GBM (multiplicative noise) at equilibrium"""
        system = GeometricBrownianMotion(mu=0.0, sigma=0.2)
        
        # For GBM with mu=0, x=0 is technically an equilibrium (though unstable)
        system.add_equilibrium("unstable", np.array([0.0]), np.array([0.0]), verify=False)
        
        A, B = system.linearized_dynamics("unstable")
        
        # Should get valid Jacobians
        assert A.shape == (1, 1)
        assert B.shape == (1, 1)

    def test_autonomous_sde_linearize_at_equilibrium(self):
        """Test linearizing autonomous SDE at equilibrium"""
        system = AutonomousOrnsteinUhlenbeck(alpha=2.0, sigma=0.3)
        system.add_equilibrium("zero", np.array([0.0]), np.array([]), verify=True)
        
        A, B = system.linearized_dynamics("zero")
        
        assert A.shape == (1, 1)
        assert B.shape == (1, 0)
        np.testing.assert_almost_equal(A, np.array([[-2.0]]))


# ============================================================================
# Test Class: SDE Integration with Equilibria
# ============================================================================


class TestSDEIntegrationWithEquilibria:
    """Test complete workflows combining SDEs and equilibria"""

    def test_sde_full_workflow_with_equilibrium(self):
        """Test complete SDE workflow using equilibrium"""
        system = OrnsteinUhlenbeck(alpha=1.0, sigma=0.3)
        
        # 1. Add and verify equilibrium
        system.add_equilibrium("zero", np.array([0.0]), np.array([0.0]), verify=True)
        
        # 2. Set as default
        system.set_default_equilibrium("zero")
        
        # 3. Get equilibrium in different backends
        x_eq_np, u_eq_np = system.get_equilibrium("zero", backend="numpy")
        assert isinstance(x_eq_np, np.ndarray)
        
        # 4. Evaluate drift at equilibrium (should be ~0)
        f = system.drift(x_eq_np, u_eq_np)
        assert np.abs(f).max() < 1e-10
        
        # 5. Evaluate diffusion at equilibrium (state-independent for O-U)
        g = system.diffusion(x_eq_np, u_eq_np)
        assert g.shape == (1, 1)
        
        # 6. Linearize at equilibrium
        A, B = system.linearized_dynamics("zero")
        assert A.shape == (1, 1)
        assert B.shape == (1, 1)
        
        # 7. Check it's additive noise
        assert system.can_optimize_for_additive()
        G = system.get_constant_noise("numpy")
        np.testing.assert_array_almost_equal(g, G)

    def test_autonomous_sde_full_workflow_with_equilibrium(self):
        """Test complete autonomous SDE workflow with equilibrium"""
        system = AutonomousOrnsteinUhlenbeck(alpha=2.0, sigma=0.3)
        
        # 1. Add equilibrium (no control)
        system.add_equilibrium("zero", np.array([0.0]), np.array([]), verify=True)
        
        # 2. Evaluate at equilibrium
        x_eq = system.equilibria.get_x("zero")
        f = system.drift(x_eq)
        assert np.abs(f).max() < 1e-10
        
        # 3. Diffusion at equilibrium
        g = system.diffusion(x_eq)
        assert g.shape == (1, 1)
        
        # 4. Linearize using name
        A, B = system.linearized_dynamics("zero")
        assert B.shape == (1, 0)  # Empty for autonomous

# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])