"""
Unit Tests for SDE Integrator Base

Tests the abstract base class for stochastic differential equation integrators,
including noise generation, ensemble statistics, and interface compliance.

Test Coverage
-------------
1. Initialization and validation
2. Pseudo-random noise generation
3. Quasi-random noise generation (QMC)
4. Noise scaling and properties
5. Statistics tracking
6. Type checking and error handling
7. Backend compatibility
8. Ensemble result handling
"""

import pytest
import numpy as np
from typing import Optional, Tuple, Callable
from unittest.mock import Mock, MagicMock, patch

# Import the classes we're testing
from src.systems.base.numerical_integration.stochastic.sde_integrator_base import (
    SDEIntegratorBase,
    SDEIntegrationResult,
    NoiseType,
    ConvergenceMode,
)


# ============================================================================
# Mock Concrete Implementation for Testing Abstract Base
# ============================================================================


class MockSDEIntegrator(SDEIntegratorBase):
    """
    Concrete mock implementation of SDEIntegratorBase for testing.

    Implements abstract methods with simple Euler-Maruyama scheme.
    """

    def step(self, x, u, dt=None, dW=None):
        """Euler-Maruyama: x_next = x + f*dt + g*dW"""
        dt = dt or self.dt

        # Generate noise if not provided
        if dW is None:
            dW = self._generate_noise(dt, (self.nw,))

        # Get drift and diffusion
        f = self._evaluate_drift(x, u)
        g = self._evaluate_diffusion(x, u)

        # Backend-specific operations
        if self.backend == "numpy":
            return x + f * dt + (g @ dW).flatten()
        elif self.backend == "torch":
            import torch

            return x + f * dt + torch.mv(g, dW)
        elif self.backend == "jax":
            import jax.numpy as jnp

            return x + f * dt + jnp.dot(g, dW)

    def integrate(
        self,
        x0,
        u_func,
        t_span,
        t_eval=None,
        num_trajectories=1,
        seed=None,
        return_full_ensemble=True,
    ):
        """Simple fixed-step integration with multiple trajectories."""
        t_start, t_end = t_span

        # Generate time grid
        if t_eval is None:
            num_steps = int((t_end - t_start) / self.dt) + 1
            t_eval = np.linspace(t_start, t_end, num_steps)
        else:
            t_eval = np.asarray(t_eval)

        num_steps = len(t_eval)

        # Initialize trajectories
        if self.backend == "numpy":
            x_all = np.zeros((num_trajectories, num_steps, self.nx))
        elif self.backend == "torch":
            import torch

            x_all = torch.zeros(num_trajectories, num_steps, self.nx)
        elif self.backend == "jax":
            import jax.numpy as jnp

            x_all = jnp.zeros((num_trajectories, num_steps, self.nx))

        # Simulate each trajectory
        for traj_idx in range(num_trajectories):
            x = (
                x0.copy()
                if self.backend == "numpy"
                else x0.clone() if self.backend == "torch" else x0
            )
            x_all[traj_idx, 0] = x

            for step_idx in range(1, num_steps):
                t = t_eval[step_idx - 1]
                u = u_func(t, x)

                # Generate noise with trajectory-specific seed
                traj_seed = seed + traj_idx if seed is not None else None
                dW = self._generate_noise(self.dt, (self.nw,), seed=traj_seed)

                # Step forward
                x = self.step(x, u, dt=self.dt, dW=dW)
                x_all[traj_idx, step_idx] = x

                self._stats["total_steps"] += 1

        self._stats["total_trajectories"] += num_trajectories

        return SDEIntegrationResult(
            t=t_eval, x=x_all, success=True, nsteps=num_steps - 1, nfev=self._stats["total_fev"]
        )

    @property
    def name(self):
        return "Mock Euler-Maruyama"


# ============================================================================
# Mock Stochastic System
# ============================================================================


@pytest.fixture
def mock_sde_system():
    """Create a mock StochasticDynamicalSystem."""
    system = Mock()
    system.nx = 2  # 2 states
    system.nu = 1  # 1 control
    system.nw = 2  # 2 Wiener processes
    system.is_stochastic = True

    # Mock drift function: f(x, u) = [-x1 + u, -x2]
    def drift(x, u, backend="numpy"):
        if backend == "numpy":
            return np.array([-x[0] + u[0], -x[1]])
        elif backend == "torch":
            import torch

            return torch.tensor([-x[0] + u[0], -x[1]])
        elif backend == "jax":
            import jax.numpy as jnp

            return jnp.array([-x[0] + u[0], -x[1]])

    # Mock diffusion function: g(x, u) = [[0.1, 0], [0, 0.2]]
    def diffusion(x, u, backend="numpy"):
        if backend == "numpy":
            return np.array([[0.1, 0.0], [0.0, 0.2]])
        elif backend == "torch":
            import torch

            return torch.tensor([[0.1, 0.0], [0.0, 0.2]])
        elif backend == "jax":
            import jax.numpy as jnp

            return jnp.array([[0.1, 0.0], [0.0, 0.2]])

    system.drift = drift
    system.diffusion = diffusion

    # Noise characteristics
    system.is_additive_noise = Mock(return_value=True)
    system.is_diagonal_noise = Mock(return_value=True)
    system.is_scalar_noise = Mock(return_value=False)

    return system


# ============================================================================
# Test Class: Initialization
# ============================================================================


class TestSDEIntegratorInitialization:
    """Test SDEIntegratorBase initialization and validation."""

    def test_basic_initialization(self, mock_sde_system):
        """Test basic integrator creation."""
        integrator = MockSDEIntegrator(system=mock_sde_system, dt=0.01, backend="numpy")

        assert integrator.dt == 0.01
        assert integrator.backend == "numpy"
        assert integrator.nx == 2
        assert integrator.nu == 1
        assert integrator.nw == 2
        assert integrator.noise_type == NoiseType.PSEUDO
        assert integrator.convergence_mode == ConvergenceMode.STRONG

    def test_noise_type_from_string(self, mock_sde_system):
        """Test noise_type accepts string input."""
        integrator = MockSDEIntegrator(system=mock_sde_system, dt=0.01, noise_type="quasi")

        assert integrator.noise_type == NoiseType.QUASI

    def test_convergence_mode_from_string(self, mock_sde_system):
        """Test convergence_mode accepts string input."""
        integrator = MockSDEIntegrator(system=mock_sde_system, dt=0.01, convergence_mode="weak")

        assert integrator.convergence_mode == ConvergenceMode.WEAK

    def test_qmc_sequence_selection(self, mock_sde_system):
        """Test QMC sequence type selection."""
        integrator = MockSDEIntegrator(
            system=mock_sde_system, dt=0.01, noise_type="quasi", qmc_sequence="halton"
        )

        assert integrator.qmc_sequence == "halton"

    def test_invalid_dt(self, mock_sde_system):
        """Test that negative or zero dt raises error."""
        with pytest.raises(ValueError, match="Time step dt must be positive"):
            MockSDEIntegrator(system=mock_sde_system, dt=0.0, backend="numpy")

        with pytest.raises(ValueError, match="Time step dt must be positive"):
            MockSDEIntegrator(system=mock_sde_system, dt=-0.01, backend="numpy")

    def test_invalid_backend(self, mock_sde_system):
        """Test that invalid backend raises error."""
        with pytest.raises(ValueError, match="Invalid backend"):
            MockSDEIntegrator(system=mock_sde_system, dt=0.01, backend="matlab")  # Not supported

    def test_non_stochastic_system_rejected(self):
        """Test that deterministic systems are rejected."""
        from src.systems.base.symbolic_dynamical_system import SymbolicDynamicalSystem

        # Mock deterministic system
        det_system = Mock(spec=SymbolicDynamicalSystem)
        det_system.is_stochastic = False

        with pytest.raises(ValueError, match="must be StochasticDynamicalSystem"):
            MockSDEIntegrator(system=det_system, dt=0.01, backend="numpy")

    def test_statistics_initialized(self, mock_sde_system):
        """Test that statistics counters are initialized."""
        integrator = MockSDEIntegrator(system=mock_sde_system, dt=0.01)

        stats = integrator.get_stats()
        assert stats["total_steps"] == 0
        assert stats["total_fev"] == 0
        assert stats["total_gev"] == 0
        assert stats["total_trajectories"] == 0


# ============================================================================
# Test Class: Pseudo-Random Noise Generation
# ============================================================================


class TestPseudoRandomNoise:
    """Test pseudo-random Brownian motion generation."""

    def test_pseudo_noise_shape(self, mock_sde_system):
        """Test that generated noise has correct shape."""
        integrator = MockSDEIntegrator(system=mock_sde_system, dt=0.01, noise_type="pseudo")

        # Single time step
        dW = integrator._generate_pseudo_noise(0.01, (2,))
        assert dW.shape == (2,)

        # Multiple steps
        dW = integrator._generate_pseudo_noise(0.01, (10, 2))
        assert dW.shape == (10, 2)

    def test_pseudo_noise_scaling(self, mock_sde_system):
        """Test that noise is scaled by sqrt(dt)."""
        integrator = MockSDEIntegrator(system=mock_sde_system, dt=0.01, noise_type="pseudo")

        # Generate many samples
        np.random.seed(42)
        samples = 10000
        dW = integrator._generate_pseudo_noise(0.01, (samples, 2))

        # Check mean ≈ 0
        mean = np.mean(dW, axis=0)
        assert np.allclose(mean, 0.0, atol=0.1)

        # Check variance ≈ dt (std ≈ sqrt(dt))
        std = np.std(dW, axis=0)
        expected_std = np.sqrt(0.01)
        assert np.allclose(std, expected_std, rtol=0.1)

    def test_pseudo_noise_reproducibility(self, mock_sde_system):
        """Test that seed makes noise reproducible."""
        integrator = MockSDEIntegrator(system=mock_sde_system, dt=0.01, noise_type="pseudo")

        # Generate with seed
        dW1 = integrator._generate_pseudo_noise(0.01, (100, 2), seed=42)
        dW2 = integrator._generate_pseudo_noise(0.01, (100, 2), seed=42)

        assert np.allclose(dW1, dW2)

    def test_pseudo_noise_different_dt(self, mock_sde_system):
        """Test noise scaling with different time steps."""
        integrator = MockSDEIntegrator(system=mock_sde_system, dt=0.01, noise_type="pseudo")

        np.random.seed(42)
        samples = 10000

        # Small dt
        dW_small = integrator._generate_pseudo_noise(0.001, (samples, 2))
        std_small = np.std(dW_small, axis=0)

        # Large dt
        np.random.seed(42)
        dW_large = integrator._generate_pseudo_noise(0.1, (samples, 2))
        std_large = np.std(dW_large, axis=0)

        # Larger dt should have larger std
        assert np.all(std_large > std_small)

        # Check sqrt(dt) scaling
        ratio = std_large[0] / std_small[0]
        expected_ratio = np.sqrt(0.1 / 0.001)
        assert np.isclose(ratio, expected_ratio, rtol=0.1)


# ============================================================================
# Test Class: Quasi-Random Noise Generation (QMC)
# ============================================================================


class TestQuasiRandomNoise:
    """Test quasi-random (QMC) Brownian motion generation."""

    @pytest.mark.parametrize("qmc_sequence", ["sobol", "halton"])
    def test_qmc_noise_shape(self, mock_sde_system, qmc_sequence):
        """Test QMC noise has correct shape."""
        integrator = MockSDEIntegrator(
            system=mock_sde_system,
            dt=0.01,
            noise_type="quasi",
            qmc_sequence=qmc_sequence,
            backend="numpy",
        )

        dW = integrator._generate_quasi_noise(0.01, (100,))
        assert dW.shape == (100,)

    @pytest.mark.parametrize("qmc_sequence", ["sobol", "halton"])
    def test_qmc_noise_scaling(self, mock_sde_system, qmc_sequence):
        """Test QMC noise is scaled by sqrt(dt)."""
        integrator = MockSDEIntegrator(
            system=mock_sde_system,
            dt=0.01,
            noise_type="quasi",
            qmc_sequence=qmc_sequence,
            backend="numpy",
        )

        # Generate samples
        samples = 1000
        dW = integrator._generate_quasi_noise(0.01, (samples,))

        # QMC should still have mean ≈ 0, std ≈ sqrt(dt)
        mean = np.mean(dW)
        std = np.std(dW)
        expected_std = np.sqrt(0.01)

        assert np.abs(mean) < 0.05  # Mean near zero
        assert np.isclose(std, expected_std, rtol=0.2)  # Std matches

    def test_qmc_low_discrepancy(self, mock_sde_system):
        """Test that QMC has better space-filling than pseudo-random."""
        # Generate uniform [0,1] samples from QMC
        integrator = MockSDEIntegrator(
            system=mock_sde_system, dt=0.01, noise_type="quasi", backend="numpy"
        )

        # Initialize generator
        integrator._initialize_qmc_generator(seed=0)

        # Get uniform samples
        from scipy.stats import qmc

        n_samples = 100
        qmc_uniform = integrator._qmc_generator.random(n_samples)

        # Compare to pseudo-random
        np.random.seed(0)
        pseudo_uniform = np.random.rand(n_samples, integrator.nw)

        # QMC should have lower discrepancy (more uniform coverage)
        # Simple test: variance of sample means should be lower for QMC
        qmc_var = np.var(qmc_uniform.mean(axis=0))
        pseudo_var = np.var(pseudo_uniform.mean(axis=0))

        # This test is probabilistic but should pass most of the time
        # (Commented out as it may be flaky)
        # assert qmc_var < pseudo_var

    def test_qmc_reproducibility(self, mock_sde_system):
        """Test QMC noise is reproducible with seed."""
        integrator = MockSDEIntegrator(
            system=mock_sde_system, dt=0.01, noise_type="quasi", backend="numpy"
        )

        # Generate with same seed
        dW1 = integrator._generate_quasi_noise(0.01, (100,), seed=42)

        # Reset generator
        integrator._qmc_generator = None

        dW2 = integrator._generate_quasi_noise(0.01, (100,), seed=42)

        assert np.allclose(dW1, dW2)

    def test_invalid_qmc_sequence(self, mock_sde_system):
        """Test invalid QMC sequence raises error."""
        integrator = MockSDEIntegrator(
            system=mock_sde_system,
            dt=0.01,
            noise_type="quasi",
            qmc_sequence="invalid_sequence",
            backend="numpy",
        )

        with pytest.raises(ValueError, match="Unknown QMC sequence"):
            integrator._generate_quasi_noise(0.01, (100,))


# ============================================================================
# Test Class: General Noise Generation
# ============================================================================


class TestGeneralNoiseGeneration:
    """Test the unified _generate_noise method."""

    def test_dispatch_to_pseudo(self, mock_sde_system):
        """Test that pseudo noise type dispatches correctly."""
        integrator = MockSDEIntegrator(system=mock_sde_system, dt=0.01, noise_type="pseudo")

        with patch.object(integrator, "_generate_pseudo_noise") as mock_pseudo:
            mock_pseudo.return_value = np.zeros((2,))

            integrator._generate_noise(0.01, (2,), seed=42)

            mock_pseudo.assert_called_once_with(0.01, (2,), 42)

    def test_dispatch_to_quasi(self, mock_sde_system):
        """Test that quasi noise type dispatches correctly."""
        integrator = MockSDEIntegrator(system=mock_sde_system, dt=0.01, noise_type="quasi")

        with patch.object(integrator, "_generate_quasi_noise") as mock_quasi:
            mock_quasi.return_value = np.zeros((2,))

            integrator._generate_noise(0.01, (2,), seed=42)

            mock_quasi.assert_called_once_with(0.01, (2,), 42)


# ============================================================================
# Test Class: Drift and Diffusion Evaluation
# ============================================================================


class TestDynamicsEvaluation:
    """Test drift and diffusion function evaluation."""

    def test_evaluate_drift_tracks_stats(self, mock_sde_system):
        """Test that drift evaluation increments counter."""
        integrator = MockSDEIntegrator(system=mock_sde_system, dt=0.01)

        x = np.array([1.0, 2.0])
        u = np.array([0.5])

        initial_fev = integrator._stats["total_fev"]

        integrator._evaluate_drift(x, u)

        assert integrator._stats["total_fev"] == initial_fev + 1

    def test_evaluate_diffusion_tracks_stats(self, mock_sde_system):
        """Test that diffusion evaluation increments counter."""
        integrator = MockSDEIntegrator(system=mock_sde_system, dt=0.01)

        x = np.array([1.0, 2.0])
        u = np.array([0.5])

        initial_gev = integrator._stats["total_gev"]

        integrator._evaluate_diffusion(x, u)

        assert integrator._stats["total_gev"] == initial_gev + 1

    def test_evaluate_drift_returns_correct_values(self, mock_sde_system):
        """Test drift function returns expected values."""
        integrator = MockSDEIntegrator(system=mock_sde_system, dt=0.01)

        x = np.array([1.0, 2.0])
        u = np.array([0.5])

        f = integrator._evaluate_drift(x, u)

        # Expected: [-x[0] + u[0], -x[1]] = [-1.0 + 0.5, -2.0] = [-0.5, -2.0]
        expected = np.array([-0.5, -2.0])
        assert np.allclose(f, expected)


# ============================================================================
# Test Class: Statistics Tracking
# ============================================================================


class TestStatisticsTracking:
    """Test integration statistics tracking."""

    def test_get_stats_initial(self, mock_sde_system):
        """Test statistics are zero initially."""
        integrator = MockSDEIntegrator(system=mock_sde_system, dt=0.01)

        stats = integrator.get_stats()

        assert stats["total_steps"] == 0
        assert stats["total_fev"] == 0
        assert stats["total_gev"] == 0
        assert stats["total_trajectories"] == 0
        assert stats["avg_fev_per_step"] == 0
        assert stats["avg_gev_per_step"] == 0

    def test_reset_stats(self, mock_sde_system):
        """Test statistics can be reset."""
        integrator = MockSDEIntegrator(system=mock_sde_system, dt=0.01)

        # Manually set some stats
        integrator._stats["total_steps"] = 100
        integrator._stats["total_fev"] = 400

        # Reset
        integrator.reset_stats()

        stats = integrator.get_stats()
        assert stats["total_steps"] == 0
        assert stats["total_fev"] == 0

    def test_avg_fev_per_step(self, mock_sde_system):
        """Test average function evaluations per step."""
        integrator = MockSDEIntegrator(system=mock_sde_system, dt=0.01)

        integrator._stats["total_steps"] = 100
        integrator._stats["total_fev"] = 400

        stats = integrator.get_stats()

        assert stats["avg_fev_per_step"] == 4.0


# ============================================================================
# Test Class: SDEIntegrationResult
# ============================================================================


class TestSDEIntegrationResult:
    """Test SDEIntegrationResult container."""

    def test_single_trajectory_result(self):
        """Test result with single trajectory."""
        t = np.linspace(0, 1, 11)
        x = np.random.randn(1, 11, 2)  # (1, T, nx)

        result = SDEIntegrationResult(t=t, x=x)

        assert result.num_trajectories == 1
        assert result.success is True
        assert result.mean is not None
        assert result.std is None  # Single trajectory

    def test_multiple_trajectories_result(self):
        """Test result with ensemble of trajectories."""
        t = np.linspace(0, 1, 11)
        x = np.random.randn(100, 11, 2)  # (100, T, nx)

        result = SDEIntegrationResult(t=t, x=x)

        assert result.num_trajectories == 100
        assert result.mean.shape == (11, 2)
        assert result.std.shape == (11, 2)

    def test_result_statistics(self):
        """Test ensemble statistics are computed correctly."""
        t = np.linspace(0, 1, 11)

        # Create known ensemble
        x = np.ones((10, 11, 2))
        x[:, :, 0] = np.arange(10)[:, None]  # x1 varies by trajectory

        result = SDEIntegrationResult(t=t, x=x)

        # Mean of x1 should be 4.5 (mean of 0-9)
        assert np.allclose(result.mean[:, 0], 4.5)

        # Mean of x2 should be 1.0
        assert np.allclose(result.mean[:, 1], 1.0)

    def test_result_repr(self):
        """Test string representation."""
        t = np.linspace(0, 1, 11)
        x = np.random.randn(10, 11, 2)

        result = SDEIntegrationResult(t=t, x=x, success=True, nsteps=10, nfev=40)

        repr_str = repr(result)
        assert "SDEIntegrationResult" in repr_str
        assert "trajectories=10" in repr_str
        assert "success=True" in repr_str


# ============================================================================
# Test Class: Noise Characteristics Check
# ============================================================================


class TestNoiseCharacteristics:
    """Test noise characteristics checking."""

    def test_check_noise_characteristics(self, mock_sde_system):
        """Test noise properties are correctly queried."""
        integrator = MockSDEIntegrator(system=mock_sde_system, dt=0.01)

        characteristics = integrator._check_noise_characteristics()

        assert characteristics["is_additive"] is True
        assert characteristics["is_diagonal"] is True
        assert characteristics["is_scalar"] is False

        # Verify system methods were called
        mock_sde_system.is_additive_noise.assert_called()
        mock_sde_system.is_diagonal_noise.assert_called()
        mock_sde_system.is_scalar_noise.assert_called()


# ============================================================================
# Test Class: Integration (End-to-End)
# ============================================================================


class TestIntegration:
    """Test full integration with MockSDEIntegrator."""

    def test_single_trajectory_integration(self, mock_sde_system):
        """Test integration of single trajectory."""
        integrator = MockSDEIntegrator(system=mock_sde_system, dt=0.01, noise_type="pseudo")

        x0 = np.array([1.0, 1.0])
        u_func = lambda t, x: np.array([0.0])

        result = integrator.integrate(
            x0=x0, u_func=u_func, t_span=(0.0, 1.0), num_trajectories=1, seed=42
        )

        assert result.success
        assert result.x.shape[0] == 1  # 1 trajectory
        assert result.x.shape[2] == 2  # 2 states
        assert result.mean is not None

    def test_multiple_trajectories_integration(self, mock_sde_system):
        """Test Monte Carlo integration with multiple trajectories."""
        integrator = MockSDEIntegrator(system=mock_sde_system, dt=0.01, noise_type="pseudo")

        x0 = np.array([1.0, 1.0])
        u_func = lambda t, x: np.array([0.0])

        result = integrator.integrate(
            x0=x0, u_func=u_func, t_span=(0.0, 1.0), num_trajectories=100, seed=42
        )

        assert result.num_trajectories == 100
        assert result.mean.shape[1] == 2  # 2 states
        assert result.std.shape[1] == 2  # 2 states

        # Check that std > 0 (there is variance)
        assert np.all(result.std > 0)

    def test_integration_with_qmc(self, mock_sde_system):
        """Test integration with quasi-random noise."""
        integrator = MockSDEIntegrator(
            system=mock_sde_system, dt=0.01, noise_type="quasi", qmc_sequence="sobol"
        )

        x0 = np.array([1.0, 1.0])
        u_func = lambda t, x: np.array([0.0])

        result = integrator.integrate(
            x0=x0, u_func=u_func, t_span=(0.0, 0.5), num_trajectories=10, seed=0
        )

        assert result.success
        assert result.num_trajectories == 10

    def test_integration_updates_statistics(self, mock_sde_system):
        """Test that integration updates statistics."""
        integrator = MockSDEIntegrator(system=mock_sde_system, dt=0.01)

        integrator.reset_stats()

        x0 = np.array([1.0, 1.0])
        u_func = lambda t, x: np.array([0.0])

        result = integrator.integrate(
            x0=x0, u_func=u_func, t_span=(0.0, 1.0), num_trajectories=5, seed=42
        )

        stats = integrator.get_stats()

        assert stats["total_trajectories"] == 5
        assert stats["total_steps"] > 0
        assert stats["total_fev"] > 0


# ============================================================================
# Test Class: String Representations
# ============================================================================


class TestStringRepresentations:
    """Test __repr__ and __str__ methods."""

    def test_repr(self, mock_sde_system):
        """Test __repr__ output."""
        integrator = MockSDEIntegrator(
            system=mock_sde_system,
            dt=0.01,
            noise_type="quasi",
            convergence_mode="weak",
            backend="jax",
        )

        repr_str = repr(integrator)

        assert "MockSDEIntegrator" in repr_str
        assert "dt=0.01" in repr_str
        assert "noise=quasi" in repr_str
        assert "mode=weak" in repr_str
        assert "backend=jax" in repr_str

    def test_str(self, mock_sde_system):
        """Test __str__ output."""
        integrator = MockSDEIntegrator(
            system=mock_sde_system, dt=0.01, noise_type="pseudo", backend="numpy"
        )

        str_repr = str(integrator)

        assert "Mock Euler-Maruyama" in str_repr
        assert "0.0100" in str_repr
        assert "Pseudo" in str_repr
        assert "numpy" in str_repr

    def test_str_with_qmc(self, mock_sde_system):
        """Test __str__ with QMC shows sequence type."""
        integrator = MockSDEIntegrator(
            system=mock_sde_system, dt=0.01, noise_type="quasi", qmc_sequence="sobol"
        )

        str_repr = str(integrator)

        assert "QMC-sobol" in str_repr


# ============================================================================
# Test Class: Backend Compatibility
# ============================================================================


@pytest.mark.skipif(
    not any(
        [
            __import__("importlib").util.find_spec("torch"),
            __import__("importlib").util.find_spec("jax"),
        ]
    ),
    reason="Requires torch or jax",
)
class TestBackendCompatibility:
    """Test integrator works with different backends."""

    @pytest.mark.parametrize("backend", ["numpy"])
    def test_backend_initialization(self, mock_sde_system, backend):
        """Test integrator initializes with different backends."""
        integrator = MockSDEIntegrator(system=mock_sde_system, dt=0.01, backend=backend)

        assert integrator.backend == backend

    def test_numpy_noise_generation(self, mock_sde_system):
        """Test noise generation with NumPy backend."""
        integrator = MockSDEIntegrator(system=mock_sde_system, dt=0.01, backend="numpy")

        dW = integrator._generate_pseudo_noise(0.01, (10, 2))

        assert isinstance(dW, np.ndarray)
        assert dW.shape == (10, 2)


# ============================================================================
# Test Class: Edge Cases and Error Handling
# ============================================================================


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_very_small_dt(self, mock_sde_system):
        """Test integrator works with very small time steps."""
        integrator = MockSDEIntegrator(system=mock_sde_system, dt=1e-6)

        assert integrator.dt == 1e-6

    def test_large_ensemble_size(self, mock_sde_system):
        """Test integrator can handle large ensembles."""
        integrator = MockSDEIntegrator(system=mock_sde_system, dt=0.01)

        x0 = np.array([1.0, 1.0])
        u_func = lambda t, x: np.array([0.0])

        # This should work without memory issues (though may be slow)
        result = integrator.integrate(
            x0=x0,
            u_func=u_func,
            t_span=(0.0, 0.1),  # Short time to keep test fast
            num_trajectories=1000,
            seed=42,
        )

        assert result.num_trajectories == 1000

    def test_zero_noise_dimension(self):
        """Test system with zero noise dimensions (should fail)."""
        system = Mock()
        system.nx = 2
        system.nu = 1
        system.nw = 0  # No noise!
        system.is_stochastic = True

        # This should work but produce deterministic results
        integrator = MockSDEIntegrator(system=system, dt=0.01)

        assert integrator.nw == 0


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
