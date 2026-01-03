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
Unit Tests for SDEIntegratorBase
================================

Tests the abstract base class for SDE integrators, including:
1. Initialization and validation
2. Random number generation
3. Drift and diffusion evaluation
4. Integration methods
5. Monte Carlo simulation
6. Noise information and statistics
7. SDEIntegrationResult container
8. String representations
9. Error handling
10. Autonomous system integration
11. Pure diffusion systems
12. Equilibrium-based integration
"""

import numpy as np
import pytest

from src.systems.base.core.continuous_stochastic_system import StochasticDynamicalSystem
from src.systems.base.numerical_integration.stochastic.sde_integrator_base import (
    ConvergenceType,
    SDEIntegrationResult,
    SDEIntegratorBase,
    StepMode,
    get_trajectory_statistics,
)
from src.types.backends import SDEType

# ============================================================================
# Helper Functions
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
        import jax.numpy as jnp

        return True
    except ImportError:
        return False


# ============================================================================
# Mock SDE Systems
# ============================================================================


class MockSDESystem(StochasticDynamicalSystem):
    """
    Simple mock SDE system (Ornstein-Uhlenbeck process).

    dx = -alpha * x * dt + sigma * dW

    Properties:
    - 1D state (nx=1), no control (nu=0), 1D noise (nw=1)
    - Additive noise
    """

    def define_system(self, alpha=1.0, sigma=0.5):
        import sympy as sp

        x = sp.symbols("x", real=True)
        alpha_sym = sp.symbols("alpha", positive=True)
        sigma_sym = sp.symbols("sigma", positive=True)

        self.state_vars = [x]
        self.control_vars = []
        self._f_sym = sp.Matrix([[-alpha_sym * x]])
        self.parameters = {alpha_sym: alpha, sigma_sym: sigma}
        self.order = 1

        self.diffusion_expr = sp.Matrix([[sigma_sym]])
        self.sde_type = "ito"

    def get_sde_type(self):
        """Return SDE type as enum."""
        return SDEType.ITO

    def get_diffusion_matrix(self, x, u=None, backend="numpy"):
        """Return diffusion matrix."""
        import numpy as np

        # Additive noise: constant diffusion
        sigma = list(self.parameters.values())[1]  # Second parameter is sigma
        return np.array([[sigma]])


class MockSDESystemMultiplicative(StochasticDynamicalSystem):
    """
    Mock SDE with multiplicative noise (Geometric Brownian Motion).

    dx = mu * x * dt + sigma * x * dW
    """

    def define_system(self, mu=0.1, sigma=0.2):
        import sympy as sp

        x = sp.symbols("x", positive=True)
        mu_sym = sp.symbols("mu", real=True)
        sigma_sym = sp.symbols("sigma", positive=True)

        self.state_vars = [x]
        self.control_vars = []
        self._f_sym = sp.Matrix([[mu_sym * x]])
        self.parameters = {mu_sym: mu, sigma_sym: sigma}
        self.order = 1

        self.diffusion_expr = sp.Matrix([[sigma_sym * x]])
        self.sde_type = "ito"

    def get_sde_type(self):
        """Return SDE type as enum."""
        return SDEType.ITO

    def get_diffusion_matrix(self, x, u=None, backend="numpy"):
        """Return diffusion matrix (multiplicative)."""
        import numpy as np

        sigma = list(self.parameters.values())[1]  # Second parameter is sigma
        x_val = np.atleast_1d(x)[0]
        return np.array([[sigma * x_val]])


class MockSDESystemControlled(StochasticDynamicalSystem):
    """
    Controlled Ornstein-Uhlenbeck process.

    dx = (-alpha * x + u) * dt + sigma * dW
    """

    def define_system(self, alpha=1.0, sigma=0.5):
        import sympy as sp

        x = sp.symbols("x", real=True)
        u = sp.symbols("u", real=True)
        alpha_sym = sp.symbols("alpha", positive=True)
        sigma_sym = sp.symbols("sigma", positive=True)

        self.state_vars = [x]
        self.control_vars = [u]
        self._f_sym = sp.Matrix([[-alpha_sym * x + u]])
        self.parameters = {alpha_sym: alpha, sigma_sym: sigma}
        self.order = 1

        self.diffusion_expr = sp.Matrix([[sigma_sym]])
        self.sde_type = "ito"

    def get_sde_type(self):
        """Return SDE type as enum."""
        return SDEType.ITO

    def get_diffusion_matrix(self, x, u=None, backend="numpy"):
        """Return diffusion matrix."""
        import numpy as np

        sigma = list(self.parameters.values())[1]  # Second parameter is sigma
        return np.array([[sigma]])


class MockSDESystemPureDiffusion(StochasticDynamicalSystem):
    """
    Pure diffusion system (Brownian motion with zero drift).

    dx = 0 * dt + sigma * dW
    """

    def define_system(self, sigma=1.0):
        import sympy as sp

        x = sp.symbols("x", real=True)
        sigma_sym = sp.symbols("sigma", positive=True)

        self.state_vars = [x]
        self.control_vars = []
        self._f_sym = sp.Matrix([[0]])
        self.parameters = {sigma_sym: sigma}
        self.order = 1

        self.diffusion_expr = sp.Matrix([[sigma_sym]])
        self.sde_type = "ito"

    def get_sde_type(self):
        """Return SDE type as enum."""
        return SDEType.ITO

    def get_diffusion_matrix(self, x, u=None, backend="numpy"):
        """Return diffusion matrix."""
        import numpy as np

        sigma = list(self.parameters.values())[0]  # First parameter is sigma
        return np.array([[sigma]])


class MockSDESystem2D(StochasticDynamicalSystem):
    """
    2D autonomous SDE system with diagonal noise.

    dx1 = -alpha * x1 * dt + sigma1 * dW1
    dx2 = -alpha * x2 * dt + sigma2 * dW2
    """

    def define_system(self, alpha=1.0, sigma1=0.5, sigma2=0.3):
        import sympy as sp

        x1, x2 = sp.symbols("x1 x2", real=True)
        alpha_sym = sp.symbols("alpha", positive=True)
        sigma1_sym = sp.symbols("sigma1", positive=True)
        sigma2_sym = sp.symbols("sigma2", positive=True)

        self.state_vars = [x1, x2]
        self.control_vars = []
        self._f_sym = sp.Matrix([[-alpha_sym * x1], [-alpha_sym * x2]])
        self.parameters = {alpha_sym: alpha, sigma1_sym: sigma1, sigma2_sym: sigma2}
        self.order = 1

        self.diffusion_expr = sp.Matrix([[sigma1_sym, 0], [0, sigma2_sym]])
        self.sde_type = "ito"

    def get_sde_type(self):
        """Return SDE type as enum."""
        return SDEType.ITO

    def get_diffusion_matrix(self, x, u=None, backend="numpy"):
        """Return diffusion matrix (2x2 diagonal)."""
        import numpy as np

        params = list(self.parameters.values())
        sigma1 = params[1]  # Second parameter
        sigma2 = params[2]  # Third parameter
        return np.array([[sigma1, 0], [0, sigma2]])


# ============================================================================
# Concrete Test Integrator
# ============================================================================


class ConcreteSDEIntegrator(SDEIntegratorBase):
    """Minimal Euler-Maruyama implementation for testing."""

    @property
    def name(self) -> str:
        return "Test Euler-Maruyama"

    def step(self, x, u=None, dt=None, dW=None):
        """Simple Euler-Maruyama step with Stratonovich correction if needed."""
        dt = dt if dt is not None else self.dt

        if dW is None:
            dW = self._generate_noise((self.nw,))

        f = self._evaluate_drift(x, u)
        g = self._evaluate_diffusion(x, u)

        # Apply Stratonovich correction if using Stratonovich interpretation
        if self.sde_type == SDEType.STRATONOVICH:
            stratonovich_correction = self._apply_stratonovich_correction(x, u, g, dt)
            f = f + stratonovich_correction

        if self.backend == "numpy":
            x_next = x + f * dt + g @ dW
        elif self.backend == "torch":
            import torch

            x_next = x + f * dt + torch.matmul(g, dW)
        elif self.backend == "jax":
            import jax.numpy as jnp

            x_next = x + f * dt + jnp.dot(g, dW)

        return x_next

    def integrate(self, x0, u_func, t_span, t_eval=None, dense_output=False):
        """Simple integration loop."""
        t0, tf = t_span

        if t_eval is None:
            n_steps = int((tf - t0) / self.dt)
            if self.backend == "numpy":
                t_eval = np.linspace(t0, tf, n_steps + 1)
            elif self.backend == "torch":
                import torch

                t_eval = torch.linspace(t0, tf, n_steps + 1)
            elif self.backend == "jax":
                import jax.numpy as jnp

                t_eval = jnp.linspace(t0, tf, n_steps + 1)

        trajectory = [x0]
        noise_samples = []
        x = x0

        for i in range(len(t_eval) - 1):
            t = float(t_eval[i])
            dt_step = float(t_eval[i + 1] - t_eval[i])

            u = u_func(t, x)
            dW = self._generate_noise((self.nw,))

            x = self.step(x, u, dt_step, dW)
            trajectory.append(x)
            noise_samples.append(dW)

            self._stats["total_steps"] += 1

        if self.backend == "numpy":
            x_traj = np.stack(trajectory)
            noise_traj = np.stack(noise_samples)
        elif self.backend == "torch":
            import torch

            x_traj = torch.stack(trajectory)
            noise_traj = torch.stack(noise_samples)
        elif self.backend == "jax":
            import jax.numpy as jnp

            x_traj = jnp.stack(trajectory)
            noise_traj = jnp.stack(noise_samples)

        result: SDEIntegrationResult = {
            "t": t_eval,
            "x": x_traj,
            "success": True,
            "nfev": self._stats["total_fev"],
            "nsteps": self._stats["total_steps"],
            "diffusion_evals": self._stats["diffusion_evals"],
            "noise_samples": noise_traj,
            "n_paths": 1,
            "convergence_type": self.convergence_type.value,
            "solver": self.name,
            "sde_type": self.sde_type.value,
        }
        return result


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def mock_sde_system():
    """Create mock autonomous SDE system (additive noise)."""
    return MockSDESystem(alpha=1.0, sigma=0.5)


@pytest.fixture
def mock_sde_multiplicative():
    """Create mock SDE with multiplicative noise."""
    return MockSDESystemMultiplicative(mu=0.1, sigma=0.2)


@pytest.fixture
def mock_sde_controlled():
    """Create mock controlled SDE system."""
    return MockSDESystemControlled(alpha=1.0, sigma=0.5)


@pytest.fixture
def mock_sde_pure_diffusion():
    """Create mock pure diffusion system (zero drift)."""
    return MockSDESystemPureDiffusion(sigma=1.0)


@pytest.fixture
def mock_sde_2d():
    """Create mock 2D autonomous SDE system."""
    return MockSDESystem2D(alpha=1.0, sigma1=0.5, sigma2=0.3)


@pytest.fixture
def integrator_numpy(mock_sde_system):
    """Create integrator with NumPy backend."""
    return ConcreteSDEIntegrator(mock_sde_system, dt=0.01, backend="numpy", seed=42)


@pytest.fixture
def integrator_additive(mock_sde_system):
    """Create integrator with additive noise system."""
    return ConcreteSDEIntegrator(mock_sde_system, dt=0.01, backend="numpy", seed=42)


@pytest.fixture
def integrator_multiplicative(mock_sde_multiplicative):
    """Create integrator with multiplicative noise system."""
    return ConcreteSDEIntegrator(mock_sde_multiplicative, dt=0.01, backend="numpy", seed=42)


@pytest.fixture
def integrator_pure_diffusion(mock_sde_pure_diffusion):
    """Create integrator with pure diffusion system."""
    return ConcreteSDEIntegrator(mock_sde_pure_diffusion, dt=0.01, backend="numpy", seed=42)


@pytest.fixture
def integrator_2d(mock_sde_2d):
    """Create integrator with 2D autonomous system."""
    return ConcreteSDEIntegrator(mock_sde_2d, dt=0.01, backend="numpy", seed=42)


# ============================================================================
# Test Class: Initialization and Validation
# ============================================================================


class TestSDEIntegratorInitialization:
    """Test initialization and validation of SDEIntegratorBase."""

    def test_initialization_basic(self, mock_sde_system):
        """Test basic initialization with valid SDE system."""
        integrator = ConcreteSDEIntegrator(mock_sde_system, dt=0.01, backend="numpy")

        assert integrator.sde_system is mock_sde_system
        assert integrator.dt == 0.01
        assert integrator.backend == "numpy"
        assert integrator.nw == 1
        assert integrator.step_mode == StepMode.FIXED

    def test_initialization_invalid_system_type(self):
        """Test that non-SDE systems raise TypeError."""
        from src.systems.base.core.continuous_symbolic_system import SymbolicDynamicalSystem

        class MockODESystem(SymbolicDynamicalSystem):
            def define_system(self):
                import sympy as sp

                x = sp.symbols("x")
                self.state_vars = [x]
                self.control_vars = []
                self._f_sym = sp.Matrix([[-x]])
                self.parameters = {}
                self.order = 1

        ode_system = MockODESystem()

        with pytest.raises(TypeError, match="StochasticDynamicalSystem"):
            ConcreteSDEIntegrator(ode_system, dt=0.01, backend="numpy")

    def test_initialization_fixed_mode_requires_dt(self, mock_sde_system):
        """Test that FIXED mode requires dt to be specified."""
        with pytest.raises(ValueError, match="Time step dt is required"):
            ConcreteSDEIntegrator(
                mock_sde_system,
                dt=None,
                step_mode=StepMode.FIXED,
                backend="numpy",
            )

    def test_sde_type_inheritance(self, mock_sde_system):
        """Test that integrator inherits SDE type from system."""
        integrator = ConcreteSDEIntegrator(mock_sde_system, dt=0.01, backend="numpy")

        assert integrator.sde_type == SDEType.ITO
        assert integrator.sde_type.value == mock_sde_system.sde_type.value

    def test_sde_type_override(self, mock_sde_system):
        """Test that SDE type can be overridden."""
        integrator = ConcreteSDEIntegrator(
            mock_sde_system,
            dt=0.01,
            backend="numpy",
            sde_type=SDEType.STRATONOVICH,
        )

        assert integrator.sde_type == SDEType.STRATONOVICH

    def test_convergence_type_default(self, mock_sde_system):
        """Test default convergence type is STRONG."""
        integrator = ConcreteSDEIntegrator(mock_sde_system, dt=0.01, backend="numpy")

        assert integrator.convergence_type == ConvergenceType.STRONG

    def test_convergence_type_custom(self, mock_sde_system):
        """Test custom convergence type."""
        integrator = ConcreteSDEIntegrator(
            mock_sde_system,
            dt=0.01,
            backend="numpy",
            convergence_type=ConvergenceType.WEAK,
        )

        assert integrator.convergence_type == ConvergenceType.WEAK

    def test_noise_structure_detection(self, integrator_additive, integrator_multiplicative):
        """Test automatic detection of noise structure."""
        assert integrator_additive._is_additive is True
        assert integrator_additive._is_multiplicative is False

        assert integrator_multiplicative._is_additive is False
        assert integrator_multiplicative._is_multiplicative is True

    def test_statistics_initialization(self, integrator_numpy):
        """Test that SDE-specific statistics are initialized."""
        stats = integrator_numpy._stats

        assert "diffusion_evals" in stats
        assert "noise_samples" in stats
        assert stats["diffusion_evals"] == 0
        assert stats["noise_samples"] == 0


# ============================================================================
# Test Class: Random Number Generation
# ============================================================================


class TestRandomNumberGeneration:
    """Test random number generation across backends."""

    def test_seed_reproducibility_numpy(self, mock_sde_system):
        """Test that same seed produces same results (NumPy)."""
        integrator1 = ConcreteSDEIntegrator(mock_sde_system, dt=0.01, backend="numpy", seed=42)
        integrator2 = ConcreteSDEIntegrator(mock_sde_system, dt=0.01, backend="numpy", seed=42)

        noise1 = integrator1._generate_noise((5,), 0.01)
        noise2 = integrator2._generate_noise((5,), 0.01)

        np.testing.assert_array_almost_equal(noise1, noise2)

    def test_seed_different_results(self, mock_sde_system):
        """Test that different seeds produce different results."""
        integrator1 = ConcreteSDEIntegrator(mock_sde_system, dt=0.01, backend="numpy", seed=42)
        integrator2 = ConcreteSDEIntegrator(mock_sde_system, dt=0.01, backend="numpy", seed=123)

        noise1 = integrator1._generate_noise((100,), 0.01)
        noise2 = integrator2._generate_noise((100,), 0.01)

        assert not np.allclose(noise1, noise2)

    def test_noise_scaling_with_dt(self, integrator_numpy):
        """Test that noise scales correctly with sqrt(dt)."""
        dt1 = 0.01
        dt2 = 0.04

        n_samples = 10000
        noise1 = np.array(
            [integrator_numpy._generate_noise((1,), dt1)[0] for _ in range(n_samples)],
        )

        integrator_numpy.set_seed(42)
        noise2 = np.array(
            [integrator_numpy._generate_noise((1,), dt2)[0] for _ in range(n_samples)],
        )

        var1 = np.var(noise1)
        var2 = np.var(noise2)

        ratio = var2 / var1
        assert 3.5 < ratio < 4.5

    def test_noise_shape(self, integrator_numpy):
        """Test that generated noise has correct shape."""
        noise = integrator_numpy._generate_noise((5,), 0.01)
        assert noise.shape == (5,)

        noise = integrator_numpy._generate_noise((10, 3), 0.01)
        assert noise.shape == (10, 3)

    def test_set_seed_method(self, integrator_numpy):
        """Test set_seed() method."""
        noise1 = integrator_numpy._generate_noise((5,), 0.01)

        integrator_numpy.set_seed(42)
        noise2 = integrator_numpy._generate_noise((5,), 0.01)

        np.testing.assert_array_almost_equal(noise1, noise2)

    @pytest.mark.skipif(not _torch_available(), reason="PyTorch not available")
    def test_noise_generation_torch(self, mock_sde_system):
        """Test noise generation with PyTorch backend."""
        import torch

        integrator = ConcreteSDEIntegrator(mock_sde_system, dt=0.01, backend="torch", seed=42)

        noise = integrator._generate_noise((5,), 0.01)

        assert isinstance(noise, torch.Tensor)
        assert noise.shape == (5,)


# ============================================================================
# Test Class: Drift and Diffusion Evaluation
# ============================================================================


class TestDriftDiffusionEvaluation:
    """Test drift and diffusion evaluation methods."""

    def test_evaluate_drift_autonomous(self, integrator_numpy):
        """Test drift evaluation for autonomous system."""
        x = np.array([1.0])

        drift = integrator_numpy._evaluate_drift(x, u=None)

        assert drift.shape == (1,)
        np.testing.assert_almost_equal(drift[0], -1.0)

    def test_evaluate_drift_controlled(self, mock_sde_controlled):
        """Test drift evaluation for controlled system."""
        integrator = ConcreteSDEIntegrator(mock_sde_controlled, dt=0.01, backend="numpy")

        x = np.array([1.0])
        u = np.array([0.5])

        drift = integrator._evaluate_drift(x, u)

        np.testing.assert_almost_equal(drift[0], -0.5)

    def test_evaluate_drift_statistics(self, integrator_numpy):
        """Test that drift evaluations are counted."""
        x = np.array([1.0])

        initial_count = integrator_numpy._stats["total_fev"]
        integrator_numpy._evaluate_drift(x, None)

        assert integrator_numpy._stats["total_fev"] == initial_count + 1

    def test_evaluate_diffusion_additive(self, integrator_additive):
        """Test diffusion evaluation for additive noise."""
        x = np.array([1.0])

        diffusion = integrator_additive._evaluate_diffusion(x, None)

        assert diffusion.shape == (1, 1)
        np.testing.assert_almost_equal(diffusion[0, 0], 0.5)

    def test_evaluate_diffusion_multiplicative(self, integrator_multiplicative):
        """Test diffusion evaluation for multiplicative noise."""
        x = np.array([2.0])

        diffusion = integrator_multiplicative._evaluate_diffusion(x, None)

        assert diffusion.shape == (1, 1)
        np.testing.assert_almost_equal(diffusion[0, 0], 0.4)

    def test_diffusion_caching_additive(self, integrator_additive):
        """Test that additive diffusion is cached."""
        assert integrator_additive._cached_diffusion is not None

        x = np.array([1.0])
        initial_evals = integrator_additive._stats["diffusion_evals"]

        diff1 = integrator_additive._evaluate_diffusion(x, None)
        diff2 = integrator_additive._evaluate_diffusion(x, None)

        assert integrator_additive._stats["diffusion_evals"] == initial_evals
        np.testing.assert_array_equal(diff1, diff2)

    def test_diffusion_no_caching_multiplicative(self, integrator_multiplicative):
        """Test that multiplicative diffusion is NOT cached."""
        assert integrator_multiplicative._cached_diffusion is None

        x = np.array([1.0])
        initial_evals = integrator_multiplicative._stats["diffusion_evals"]
        integrator_multiplicative._evaluate_diffusion(x, None)

        assert integrator_multiplicative._stats["diffusion_evals"] == initial_evals + 1


# ============================================================================
# Test Class: Integration
# ============================================================================


class TestIntegration:
    """Test integration methods."""

    def test_single_step(self, integrator_numpy):
        """Test single integration step."""
        x0 = np.array([1.0])
        u = None
        dt = 0.01

        x1 = integrator_numpy.step(x0, u, dt)

        assert x1.shape == x0.shape
        assert not np.array_equal(x0, x1)

    def test_step_with_custom_noise(self, integrator_numpy):
        """Test step with user-provided noise."""
        x0 = np.array([1.0])
        dW = np.array([0.1])

        x1 = integrator_numpy.step(x0, u=None, dt=0.01, dW=dW)

        assert x1.shape == (1,)

    def test_integrate_trajectory(self, integrator_numpy):
        """Test full trajectory integration."""
        x0 = np.array([1.0])
        u_func = lambda t, x: None
        t_span = (0.0, 1.0)

        result = integrator_numpy.integrate(x0, u_func, t_span)

        assert result["success"]
        assert result["t"].shape[0] > 1
        assert result["x"].shape[0] == result["t"].shape[0]
        assert result["x"].shape[1] == 1

    def test_integrate_with_control(self, mock_sde_controlled):
        """Test integration with control input."""
        integrator = ConcreteSDEIntegrator(mock_sde_controlled, dt=0.01, backend="numpy", seed=42)

        x0 = np.array([1.0])
        u_func = lambda t, x: np.array([0.5])
        t_span = (0.0, 0.1)

        result = integrator.integrate(x0, u_func, t_span)

        assert result["success"]
        assert result["x"].shape[0] > 1

    def test_integrate_statistics(self, integrator_numpy):
        """Test that integration updates statistics."""
        x0 = np.array([1.0])
        u_func = lambda t, x: None
        t_span = (0.0, 0.1)

        result = integrator_numpy.integrate(x0, u_func, t_span)

        assert result["nfev"] > 0
        assert result["nsteps"] > 0
        assert result["diffusion_evals"] >= 0

    def test_integrate_returns_noise_samples(self, integrator_numpy):
        """Test that integration returns noise samples."""
        x0 = np.array([1.0])
        u_func = lambda t, x: None
        t_span = (0.0, 0.1)

        result = integrator_numpy.integrate(x0, u_func, t_span)

        assert result["noise_samples"] is not None
        assert result["noise_samples"].shape[0] > 0


# ============================================================================
# Test Class: Monte Carlo Simulation
# ============================================================================


class TestMonteCarloSimulation:
    """Test Monte Carlo trajectory simulation."""

    def test_monte_carlo_basic(self, integrator_numpy):
        """Test basic Monte Carlo simulation."""
        x0 = np.array([1.0])
        u_func = lambda t, x: None
        t_span = (0.0, 0.5)
        n_paths = 10

        result = integrator_numpy.integrate_monte_carlo(
            x0,
            u_func,
            t_span,
            n_paths,
            store_paths=True,
        )

        assert result["success"]
        assert result["n_paths"] == n_paths
        assert result["x"].shape[0] == n_paths

    def test_monte_carlo_statistics(self, integrator_numpy):
        """Test Monte Carlo statistics computation."""
        x0 = np.array([1.0])
        u_func = lambda t, x: None
        t_span = (0.0, 0.5)
        n_paths = 100

        result = integrator_numpy.integrate_monte_carlo(x0, u_func, t_span, n_paths)

        stats = get_trajectory_statistics(result)

        assert "mean" in stats
        assert "std" in stats
        assert stats["mean"].shape[0] > 1
        assert stats["n_paths"] == n_paths

    def test_monte_carlo_variance_increases(self, integrator_numpy):
        """Test that variance increases over time (as expected for diffusion)."""
        x0 = np.array([0.0])
        u_func = lambda t, x: None
        t_span = (0.0, 1.0)
        n_paths = 500

        result = integrator_numpy.integrate_monte_carlo(x0, u_func, t_span, n_paths)

        stats = get_trajectory_statistics(result)

        var_start = stats["std"][1] ** 2
        var_end = stats["std"][-1] ** 2

        assert var_end > var_start


# ============================================================================
# Test Class: Noise Information and Statistics
# ============================================================================


class TestNoiseInformation:
    """Test noise structure queries and statistics."""

    def test_get_noise_info(self, integrator_additive):
        """Test get_noise_info() method."""
        info = integrator_additive.get_noise_info()

        assert "nw" in info
        assert "is_additive" in info
        assert "is_diagonal" in info
        assert "noise_type" in info
        assert info["nw"] == 1
        assert info["is_additive"] is True

    def test_get_sde_stats(self, integrator_numpy):
        """Test get_sde_stats() method."""
        x0 = np.array([1.0])
        u_func = lambda t, x: None
        integrator_numpy.integrate(x0, u_func, (0.0, 0.1))

        stats = integrator_numpy.get_sde_stats()

        assert "diffusion_evals" in stats
        assert "noise_samples" in stats
        assert "avg_diffusion_per_step" in stats
        assert stats["diffusion_evals"] >= 0

    def test_reset_stats_includes_sde(self, integrator_numpy):
        """Test that reset_stats() clears SDE-specific counters."""
        integrator_numpy._stats["diffusion_evals"] = 10
        integrator_numpy._stats["noise_samples"] = 20

        integrator_numpy.reset_stats()

        assert integrator_numpy._stats["diffusion_evals"] == 0
        assert integrator_numpy._stats["noise_samples"] == 0


# ============================================================================
# Test Class: SDEIntegrationResult
# ============================================================================


class TestSDEIntegrationResult:
    """Test SDEIntegrationResult container."""

    def test_result_initialization(self):
        """Test basic result initialization."""
        t = np.linspace(0, 1, 11)
        x = np.random.randn(11, 2)

        result: SDEIntegrationResult = {
            "t": t,
            "x": x,
            "success": True,
            "nsteps": 10,
            "nfev": 10,
            "solver": "Test",
            "n_paths": 1,
        }

        assert result["success"]
        assert result["nsteps"] == 10
        assert result["n_paths"] == 1

    def test_result_statistics_single_path(self):
        """Test statistics for single trajectory."""
        t = np.linspace(0, 1, 11)
        x = np.random.randn(11, 2)

        result: SDEIntegrationResult = {
            "t": t,
            "x": x,
            "n_paths": 1,
            "success": True,
            "nfev": 10,
            "nsteps": 10,
            "solver": "Test",
        }
        stats = get_trajectory_statistics(result)

        assert stats["n_paths"] == 1
        assert "note" in stats
        np.testing.assert_array_equal(stats["mean"], x)

    def test_result_statistics_multiple_paths(self):
        """Test statistics for multiple trajectories."""
        t = np.linspace(0, 1, 11)
        x = np.random.randn(100, 11, 2)

        result: SDEIntegrationResult = {
            "t": t,
            "x": x,
            "n_paths": 100,
            "success": True,
            "nfev": 100,
            "nsteps": 10,
            "solver": "Test",
        }
        stats = get_trajectory_statistics(result)

        assert stats["n_paths"] == 100
        assert stats["mean"].shape == (11, 2)
        assert stats["std"].shape == (11, 2)
        assert "median" in stats
        assert "q25" in stats
        assert "q75" in stats


# ============================================================================
# Test Class: String Representations
# ============================================================================


class TestStringRepresentations:
    """Test __repr__ and __str__ methods."""

    def test_integrator_repr(self, integrator_numpy):
        """Test integrator __repr__."""
        repr_str = repr(integrator_numpy)

        assert "ConcreteSDEIntegrator" in repr_str
        assert "dt=0.01" in repr_str
        assert "numpy" in repr_str
        assert "ito" in repr_str

    def test_integrator_str(self, integrator_numpy):
        """Test integrator __str__."""
        str_str = str(integrator_numpy)

        assert "Test Euler-Maruyama" in str_str
        assert "0.0100" in str_str or "0.01" in str_str
        assert "numpy" in str_str

    def test_result_repr(self):
        """Test SDEIntegrationResult as TypedDict."""
        result: SDEIntegrationResult = {
            "t": np.array([0, 1]),
            "x": np.array([[1], [2]]),
            "nsteps": 1,
            "nfev": 5,
            "diffusion_evals": 3,
            "n_paths": 10,
            "success": True,
            "solver": "Test",
        }

        # TypedDict is a dict
        assert isinstance(result, dict)
        assert "t" in result
        assert "x" in result
        assert result["nsteps"] == 1
        assert result["n_paths"] == 10


# ============================================================================
# Test Class: Error Handling
# ============================================================================


class TestErrorHandling:
    """Test error handling and validation."""

    def test_stratonovich_correction_additive_is_zero(self, mock_sde_system):
        """Test that Stratonovich correction is zero for additive noise."""
        integrator = ConcreteSDEIntegrator(
            mock_sde_system,
            dt=0.01,
            backend="numpy",
            sde_type=SDEType.STRATONOVICH,
        )

        x = np.array([1.0])
        g = np.array([[0.5]])  # Additive (constant)

        correction = integrator._apply_stratonovich_correction(x, None, g, 0.01)

        # Additive noise has zero correction
        np.testing.assert_array_almost_equal(correction, np.zeros(1))

    def test_stratonovich_correction_multiplicative(self, mock_sde_multiplicative):
        """Test Stratonovich correction for multiplicative noise (GBM)."""
        # Geometric Brownian Motion: dx = μx dt + σx dW (Stratonovich)
        # Itô form: dx = (μ + 0.5*σ²)x dt + σx dW
        # So correction should be 0.5*σ²*x

        integrator = ConcreteSDEIntegrator(
            mock_sde_multiplicative,
            dt=0.01,
            backend="numpy",
            sde_type=SDEType.STRATONOVICH,
        )

        x = np.array([2.0])  # State value
        sigma = 0.2  # From MockSDESystemMultiplicative default

        # Get diffusion matrix
        g = integrator._evaluate_diffusion(x, None)

        # Compute correction
        correction = integrator._apply_stratonovich_correction(x, None, g, 0.01)

        # Expected: 0.5 * σ² * x
        expected = 0.5 * sigma**2 * x

        np.testing.assert_array_almost_equal(correction, expected, decimal=5)

    def test_invalid_backend_raises(self, mock_sde_system):
        """Test that invalid backend raises ValueError."""
        with pytest.raises(ValueError, match="Invalid backend"):
            ConcreteSDEIntegrator(mock_sde_system, dt=0.01, backend="invalid_backend")


# ============================================================================
# Test Class: Autonomous System Integration
# ============================================================================


class TestAutonomousSystems:
    """Test autonomous stochastic systems (nu=0)."""

    def test_autonomous_system_properties(self, mock_sde_system):
        """Test that autonomous system has correct properties."""
        assert mock_sde_system.nu == 0
        assert mock_sde_system.nx == 1
        assert mock_sde_system.nw == 1

    def test_autonomous_drift_evaluation(self, integrator_numpy):
        """Test drift evaluation with u=None for autonomous system."""
        x = np.array([1.0])

        drift = integrator_numpy._evaluate_drift(x, u=None)

        assert drift.shape == (1,)
        np.testing.assert_almost_equal(drift[0], -1.0)

    def test_autonomous_diffusion_evaluation(self, integrator_numpy):
        """Test diffusion evaluation with u=None for autonomous system."""
        x = np.array([1.0])

        diffusion = integrator_numpy._evaluate_diffusion(x, u=None)

        assert diffusion.shape == (1, 1)
        np.testing.assert_almost_equal(diffusion[0, 0], 0.5)

    def test_autonomous_single_step(self, integrator_numpy):
        """Test single step for autonomous system."""
        x0 = np.array([1.0])

        x1 = integrator_numpy.step(x0, u=None, dt=0.01)

        assert x1.shape == x0.shape
        assert not np.array_equal(x0, x1)

    def test_autonomous_integrate(self, integrator_numpy):
        """Test full integration for autonomous system."""
        x0 = np.array([1.0])
        u_func = lambda t, x: None
        t_span = (0.0, 1.0)

        result = integrator_numpy.integrate(x0, u_func, t_span)

        assert result["success"]
        assert result["x"].shape[0] > 1
        assert result["x"].shape[1] == 1

    def test_autonomous_monte_carlo(self, integrator_numpy):
        """Test Monte Carlo for autonomous system."""
        x0 = np.array([1.0])
        u_func = lambda t, x: None
        t_span = (0.0, 0.5)
        n_paths = 50

        result = integrator_numpy.integrate_monte_carlo(x0, u_func, t_span, n_paths)

        assert result["success"]
        assert result["n_paths"] == n_paths
        assert result["x"].shape[0] == n_paths

    def test_2d_autonomous_system(self, integrator_2d):
        """Test 2D autonomous system integration."""
        x0 = np.array([1.0, 2.0])
        u_func = lambda t, x: None
        t_span = (0.0, 0.5)

        result = integrator_2d.integrate(x0, u_func, t_span)

        assert result["success"]
        assert result["x"].shape[1] == 2
        assert result["x"].shape[0] > 1

    def test_2d_autonomous_noise_independence(self, integrator_2d):
        """Test that 2D noise sources are independent."""
        x0 = np.array([0.0, 0.0])
        u_func = lambda t, x: None
        t_span = (0.0, 2.0)
        n_paths = 500

        result = integrator_2d.integrate_monte_carlo(x0, u_func, t_span, n_paths)

        x1_final = result["x"][:, -1, 0]
        x2_final = result["x"][:, -1, 1]

        correlation = np.corrcoef(x1_final, x2_final)[0, 1]

        assert abs(correlation) < 0.15


# ============================================================================
# Test Class: Pure Diffusion Systems
# ============================================================================


class TestPureDiffusionSystems:
    """Test pure diffusion systems (zero drift)."""

    def test_pure_diffusion_properties(self, mock_sde_pure_diffusion):
        """Test that pure diffusion system has correct properties."""
        assert mock_sde_pure_diffusion.is_pure_diffusion()
        assert mock_sde_pure_diffusion.nu == 0
        assert mock_sde_pure_diffusion.nx == 1
        assert mock_sde_pure_diffusion.nw == 1

    def test_pure_diffusion_zero_drift(self, integrator_pure_diffusion):
        """Test that drift is exactly zero."""
        x = np.array([1.0])

        drift = integrator_pure_diffusion._evaluate_drift(x, None)

        np.testing.assert_array_equal(drift, np.array([0.0]))

    def test_pure_diffusion_nonzero_diffusion(self, integrator_pure_diffusion):
        """Test that diffusion is non-zero."""
        x = np.array([1.0])

        diffusion = integrator_pure_diffusion._evaluate_diffusion(x, None)

        assert diffusion.shape == (1, 1)
        np.testing.assert_almost_equal(diffusion[0, 0], 1.0)

    def test_pure_diffusion_step(self, integrator_pure_diffusion):
        """Test single step for pure diffusion."""
        x0 = np.array([0.0])
        dt = 0.01

        x1 = integrator_pure_diffusion.step(x0, u=None, dt=dt)

        assert x1.shape == x0.shape

    def test_pure_diffusion_integrate(self, integrator_pure_diffusion):
        """Test integration of pure diffusion process."""
        x0 = np.array([0.0])
        u_func = lambda t, x: None
        t_span = (0.0, 1.0)

        result = integrator_pure_diffusion.integrate(x0, u_func, t_span)

        assert result["success"]
        assert result["x"].shape[0] > 1

    def test_pure_diffusion_zero_mean(self, integrator_pure_diffusion):
        """Test that pure diffusion starting at zero has zero mean."""
        x0 = np.array([0.0])
        u_func = lambda t, x: None
        t_span = (0.0, 1.0)
        n_paths = 1000

        result = integrator_pure_diffusion.integrate_monte_carlo(x0, u_func, t_span, n_paths)

        stats = get_trajectory_statistics(result)
        final_mean = stats["mean"][-1, 0]

        assert abs(final_mean) < 0.1

    def test_pure_diffusion_variance_linear_growth(self, integrator_pure_diffusion):
        """Test that variance grows linearly with time for pure diffusion."""
        x0 = np.array([0.0])
        u_func = lambda t, x: None
        t_span = (0.0, 2.0)
        n_paths = 1000

        result = integrator_pure_diffusion.integrate_monte_carlo(x0, u_func, t_span, n_paths)

        stats = get_trajectory_statistics(result)

        t_array = result["t"]
        var_array = stats["std"][:, 0] ** 2

        idx_half = len(t_array) // 4
        idx_three_half = 3 * len(t_array) // 4

        t_half = float(t_array[idx_half])
        t_three_half = float(t_array[idx_three_half])

        var_half = var_array[idx_half]
        var_three_half = var_array[idx_three_half]

        time_ratio = t_three_half / t_half
        var_ratio = var_three_half / var_half

        assert abs(var_ratio - time_ratio) / time_ratio < 0.2


# ============================================================================
# Test Class: Equilibrium-Based Integration
# ============================================================================


class TestSDEEquilibriumIntegration:
    """Test SDE integration starting from named equilibria."""

    def test_integrate_from_origin_equilibrium(self, mock_sde_controlled):
        """Test integration starting from origin equilibrium."""
        x_eq, u_eq = mock_sde_controlled.get_equilibrium("origin")

        assert np.allclose(x_eq, np.zeros(1))
        assert np.allclose(u_eq, np.zeros(1))

        integrator = ConcreteSDEIntegrator(mock_sde_controlled, dt=0.01, backend="numpy", seed=42)

        u_func = lambda t, x: u_eq
        result = integrator.integrate(x_eq, u_func, t_span=(0, 1))

        assert result["success"]
        assert result["x"].shape[0] > 1

    def test_integrate_from_custom_equilibrium(self, mock_sde_controlled):
        """Test integration from non-origin equilibrium."""
        mock_sde_controlled.add_equilibrium(
            "custom",
            x_eq=np.array([1.0]),
            u_eq=np.array([1.0]),
            verify=True,
            tol=1e-10,
        )

        x_eq, u_eq = mock_sde_controlled.get_equilibrium("custom")

        integrator = ConcreteSDEIntegrator(mock_sde_controlled, dt=0.01, backend="numpy", seed=42)

        u_func = lambda t, x: u_eq
        result = integrator.integrate(x_eq, u_func, t_span=(0, 1))

        assert result["success"]


# ============================================================================
# Test Class: Integration End-to-End
# ============================================================================


class TestIntegrationEndToEnd:
    """End-to-end integration tests."""

    def test_full_simulation_workflow(self, integrator_numpy):
        """Test complete simulation workflow."""
        x0 = np.array([1.0])
        u_func = lambda t, x: None
        t_span = (0.0, 1.0)

        integrator_numpy.reset_stats()

        result = integrator_numpy.integrate(x0, u_func, t_span)

        assert result["success"]
        assert result["x"].shape[0] > 10

        stats = integrator_numpy.get_sde_stats()
        assert stats["total_steps"] > 0
        assert stats["total_fev"] > 0

    def test_reproducibility_with_seed(self, mock_sde_system):
        """Test that same seed gives reproducible results."""
        x0 = np.array([1.0])
        u_func = lambda t, x: None
        t_span = (0.0, 0.5)

        integrator1 = ConcreteSDEIntegrator(mock_sde_system, dt=0.01, backend="numpy", seed=42)
        result1 = integrator1.integrate(x0, u_func, t_span)

        integrator2 = ConcreteSDEIntegrator(mock_sde_system, dt=0.01, backend="numpy", seed=42)
        result2 = integrator2.integrate(x0, u_func, t_span)

        np.testing.assert_array_almost_equal(result1["x"], result2["x"])

    def test_monte_carlo_convergence(self, integrator_numpy):
        """Test that Monte Carlo estimates converge with more paths."""
        x0 = np.array([0.0])
        u_func = lambda t, x: None
        t_span = (0.0, 1.0)

        result_small = integrator_numpy.integrate_monte_carlo(x0, u_func, t_span, n_paths=50)
        stats_small = get_trajectory_statistics(result_small)

        integrator_numpy.set_seed(42)
        result_large = integrator_numpy.integrate_monte_carlo(x0, u_func, t_span, n_paths=500)
        stats_large = get_trajectory_statistics(result_large)

        se_small = stats_small["std"][-1] / np.sqrt(50)
        se_large = stats_large["std"][-1] / np.sqrt(500)

        assert se_large < se_small


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
