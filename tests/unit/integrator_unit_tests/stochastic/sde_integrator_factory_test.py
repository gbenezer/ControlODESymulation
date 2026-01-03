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
Unit tests for SDEIntegratorFactory

Tests factory creation, method routing, use-case helpers, and integration
for controlled systems, autonomous systems, and pure diffusion processes.

Design Note
-----------
This test suite uses types from the centralized type system to ensure
consistency with the refactored factory implementation:
- SDEType, ConvergenceType, NoiseType from src.types.backends
- SDEIntegrationResult from src.types.trajectories
- Backend type for backend specifications
- ScalarLike for time and parameter values

This ensures the tests validate the correct type usage patterns.

Run with:
    pytest test_sde_integrator_factory.py -v
"""

from unittest.mock import patch

import numpy as np
import pytest

from src.systems.base.core.continuous_stochastic_system import StochasticDynamicalSystem

# Import base classes and utilities
from src.systems.base.numerical_integration.integrator_base import StepMode
from src.systems.base.numerical_integration.stochastic.sde_integrator_base import (
    SDEIntegratorBase,
)

# Import factory and base classes
from src.systems.base.numerical_integration.stochastic.sde_integrator_factory import (
    SDEIntegratorFactory,
    auto_sde_integrator,
    create_sde_integrator,
)

# Import from centralized type system
from src.types.backends import (
    Backend,
    ConvergenceType,
    NoiseType,
    SDEType,
)
from src.types.core import ControlVector, ScalarLike, StateVector
from src.types.trajectories import SDEIntegrationResult

# ============================================================================
# Mock SDE Systems for Testing
# ============================================================================


class MockSDESystem(StochasticDynamicalSystem):
    """Mock stochastic dynamical system for testing.

    Uses object.__setattr__() to bypass read-only properties.
    Follows centralized type system for all type annotations.
    """

    def __init__(self, nx: int = 2, nu: int = 1, nw: int = 2, noise_type: str = "general"):
        # Don't call super().__init__() to avoid complex initialization
        # Instead, set attributes directly using object.__setattr__() to bypass properties
        object.__setattr__(self, "_nx", nx)
        object.__setattr__(self, "_nu", nu)
        object.__setattr__(self, "_nw", nw)
        object.__setattr__(self, "_sde_type", SDEType.ITO)
        object.__setattr__(self, "_noise_type", noise_type)
        object.__setattr__(self, "_system_name", "MockSDESystem")

        # Initialize required internal state
        object.__setattr__(self, "_backend", "numpy")

    @property
    def nx(self) -> int:
        return self._nx

    @property
    def nu(self) -> int:
        return self._nu

    @property
    def nw(self) -> int:
        return self._nw

    @property
    def sde_type(self) -> SDEType:
        """SDE interpretation type (Ito or Stratonovich)."""
        return self._sde_type

    @property
    def backend(self) -> Backend:
        """Default backend for this system."""
        return self._backend

    @property
    def is_stochastic(self) -> bool:
        """Whether this is a stochastic system."""
        return True

    @property
    def is_continuous(self) -> bool:
        """Whether this is a continuous-time system."""
        return True

    @property
    def name(self) -> str:
        """System name."""
        return self._system_name

    def define_system(self):
        """Required abstract method - mock implementation."""

    def drift(
        self, x: StateVector, u: ControlVector, t: ScalarLike = 0.0, backend: Backend = "numpy",
    ) -> StateVector:
        """Mock drift function: f(x, u) = -x + u

        Note: Added t parameter to match StochasticDynamicalSystem.__call__() signature.
        """
        if u is None:
            return -x  # Autonomous
        if backend == "numpy":
            return -x + np.asarray(u)[: self.nx]
        if backend == "torch":
            import torch

            return -x + torch.as_tensor(u)[: self.nx]
        if backend == "jax":
            import jax.numpy as jnp

            return -x + jnp.asarray(u)[: self.nx]

    def diffusion(self, x: StateVector, u: ControlVector, backend: Backend = "numpy"):
        """Mock diffusion function: g(x, u) = constant or multiplicative"""
        if self._noise_type == "additive":
            # Constant diffusion
            if backend == "numpy":
                return 0.1 * np.ones((self.nx, self.nw))
            if backend == "torch":
                import torch

                return 0.1 * torch.ones((self.nx, self.nw))
            if backend == "jax":
                import jax.numpy as jnp

                return 0.1 * jnp.ones((self.nx, self.nw))
        # State-dependent diffusion
        elif backend == "numpy":
            return 0.1 * np.diag(np.abs(x))[:, : self.nw]
        elif backend == "torch":
            import torch

            return 0.1 * torch.diag(torch.abs(x))[:, : self.nw]
        elif backend == "jax":
            import jax.numpy as jnp

            return 0.1 * jnp.diag(jnp.abs(x))[:, : self.nw]

    def is_additive_noise(self) -> bool:
        return self._noise_type == "additive"

    def is_diagonal_noise(self) -> bool:
        return self._noise_type == "diagonal"

    def is_multiplicative_noise(self) -> bool:
        return self._noise_type != "additive"

    def is_scalar_noise(self) -> bool:
        return self.nw == 1 and self.nx == 1

    def get_noise_type(self) -> NoiseType:
        """Return noise type using centralized NoiseType enum."""
        if self._noise_type == "additive":
            return NoiseType.ADDITIVE
        if self._noise_type == "diagonal":
            return NoiseType.DIAGONAL
        return NoiseType.GENERAL

    def get_constant_noise(self, backend: Backend = "numpy"):
        if self._noise_type == "additive":
            return self.diffusion(np.zeros(self.nx), None, backend)
        return None

    def get_sde_type(self) -> SDEType:
        """Return SDE interpretation type (Ito or Stratonovich)."""
        return self._sde_type

    def get_drift_matrix(
        self, x: StateVector, u: ControlVector, backend: Backend = "numpy",
    ) -> StateVector:
        """Alias for drift() - some integrators may use this name."""
        return self.drift(x, u, backend)

    def get_diffusion_matrix(self, x: StateVector, u: ControlVector, backend: Backend = "numpy"):
        """Alias for diffusion() - some integrators may use this name."""
        return self.diffusion(x, u, backend)


class MockAutonomousSDESystem(MockSDESystem):
    """Mock autonomous SDE system (nu=0)."""

    def __init__(self, nx: int = 2, nw: int = 2, noise_type: str = "general"):
        super().__init__(nx=nx, nu=0, nw=nw, noise_type=noise_type)

    def drift(
        self, x: StateVector, u: ControlVector, t: ScalarLike = 0.0, backend: Backend = "numpy",
    ) -> StateVector:
        """Autonomous drift: f(x) = -x

        Note: Added t parameter to match StochasticDynamicalSystem.__call__() signature.
        """
        return -x

    def diffusion(self, x: StateVector, u: ControlVector, backend: Backend = "numpy"):
        """Autonomous diffusion"""
        return super().diffusion(x, None, backend)


class MockPureDiffusionSystem(MockSDESystem):
    """Pure diffusion process (zero drift)."""

    def __init__(self, nx: int = 2, nw: int = 2, noise_type: str = "additive"):
        super().__init__(nx=nx, nu=0, nw=nw, noise_type=noise_type)

    def drift(
        self, x: StateVector, u: ControlVector, t: ScalarLike = 0.0, backend: Backend = "numpy",
    ) -> StateVector:
        """Zero drift

        Note: Added t parameter to match StochasticDynamicalSystem.__call__() signature.
        """
        if backend == "numpy":
            return np.zeros_like(x)
        if backend == "torch":
            import torch

            return torch.zeros_like(x)
        if backend == "jax":
            import jax.numpy as jnp

            return jnp.zeros_like(x)


# ============================================================================
# Mock Integrators
# ============================================================================


class MockSDEIntegrator(SDEIntegratorBase):
    """Mock SDE integrator for testing factory.

    Uses centralized types for all type annotations.
    """

    def __init__(
        self,
        sde_system: StochasticDynamicalSystem,
        dt: ScalarLike = 0.01,
        backend: Backend = "numpy",
        method: str = "mock",
        **options,
    ):
        super().__init__(sde_system, dt=dt, step_mode=StepMode.FIXED, backend=backend, **options)
        self.method = method
        self._name = f"Mock-{method}"

    @property
    def name(self) -> str:
        return self._name

    def step(
        self,
        x: StateVector,
        u: ControlVector = None,
        dt: ScalarLike = None,
        dW=None,
    ) -> StateVector:
        """Mock step - just return x + small change"""
        dt = dt if dt is not None else self.dt
        # Use parent class method to evaluate drift (handles statistics tracking)
        dx = self._evaluate_drift(x, u)
        return x + dt * dx

    def integrate(
        self,
        x0: StateVector,
        u_func,
        t_span,
        t_eval=None,
        dense_output: bool = False,
    ) -> SDEIntegrationResult:
        """Mock integration"""
        t0, tf = t_span

        # Create simple trajectory
        if t_eval is None:
            n_steps = max(2, int((tf - t0) / self.dt) + 1)
            t_points = np.linspace(t0, tf, n_steps)
        else:
            t_points = np.asarray(t_eval)
            n_steps = len(t_points)

        # Simple trajectory: exponential decay
        x_traj = np.zeros((n_steps, len(x0)))
        x = np.asarray(x0)
        x_traj[0] = x

        for i in range(1, n_steps):
            t = t_points[i]
            u = u_func(t, x) if u_func else None
            x = self.step(x, u, self.dt)
            x_traj[i] = x

        # Return SDEIntegrationResult TypedDict
        return SDEIntegrationResult(
            t=t_points,
            x=x_traj,
            success=True,
            message="Mock integration completed",
            nfev=n_steps,
            nsteps=n_steps,
            solver=self.name,
            integration_time=0.001,
            diffusion_evals=n_steps,
            noise_samples=np.zeros((n_steps - 1, self.sde_system.nw)),
            n_paths=1,
            convergence_type="strong",
            sde_type="ito",
        )


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def controlled_sde_system():
    """Controlled SDE system (nu > 0)."""
    return MockSDESystem(nx=2, nu=1, nw=2, noise_type="general")


@pytest.fixture
def autonomous_sde_system():
    """Autonomous SDE system (nu = 0)."""
    return MockAutonomousSDESystem(nx=2, nw=2, noise_type="additive")


@pytest.fixture
def pure_diffusion_system():
    """Pure diffusion process (zero drift)."""
    return MockPureDiffusionSystem(nx=2, nw=2, noise_type="additive")


@pytest.fixture
def additive_noise_system():
    """System with additive noise."""
    return MockSDESystem(nx=2, nu=1, nw=2, noise_type="additive")


@pytest.fixture
def diagonal_noise_system():
    """System with diagonal noise."""
    return MockSDESystem(nx=2, nu=1, nw=2, noise_type="diagonal")


# ============================================================================
# Test Basic Factory Creation
# ============================================================================


class TestFactoryBasics:
    """Test basic factory creation and backend selection."""

    def test_create_numpy_default(self, controlled_sde_system):
        """Test creation with default NumPy backend."""
        mock_integrator = MockSDEIntegrator(controlled_sde_system)

        with patch(
            "src.systems.base.numerical_integration.stochastic.diffeqpy_sde_integrator.DiffEqPySDEIntegrator",
            return_value=mock_integrator,
        ) as mock_class:
            integrator = SDEIntegratorFactory.create(controlled_sde_system)

            assert isinstance(integrator, SDEIntegratorBase)
            mock_class.assert_called_once()
            call_kwargs = mock_class.call_args[1]
            assert call_kwargs["backend"] == "numpy"

    def test_create_numpy_explicit_method(self, controlled_sde_system):
        """Test creation with explicit Julia method."""
        mock_integrator = MockSDEIntegrator(controlled_sde_system)

        with patch(
            "src.systems.base.numerical_integration.stochastic.diffeqpy_sde_integrator.DiffEqPySDEIntegrator",
            return_value=mock_integrator,
        ) as mock_class:
            integrator = SDEIntegratorFactory.create(
                controlled_sde_system, backend="numpy", method="SRIW1",
            )

            call_kwargs = mock_class.call_args[1]
            assert call_kwargs["algorithm"] == "SRIW1"

    def test_create_torch_backend(self, controlled_sde_system):
        """Test creation with PyTorch backend."""
        mock_integrator = MockSDEIntegrator(controlled_sde_system, backend="torch")

        with patch(
            "src.systems.base.numerical_integration.stochastic.torchsde_integrator.TorchSDEIntegrator",
            return_value=mock_integrator,
        ) as mock_class:
            integrator = SDEIntegratorFactory.create(controlled_sde_system, backend="torch")

            assert isinstance(integrator, SDEIntegratorBase)
            mock_class.assert_called_once()
            call_kwargs = mock_class.call_args[1]
            assert call_kwargs["backend"] == "torch"

    def test_create_jax_backend(self, controlled_sde_system):
        """Test creation with JAX backend."""
        mock_integrator = MockSDEIntegrator(controlled_sde_system, backend="jax")

        with patch(
            "src.systems.base.numerical_integration.stochastic.diffrax_sde_integrator.DiffraxSDEIntegrator",
            return_value=mock_integrator,
        ) as mock_class:
            integrator = SDEIntegratorFactory.create(controlled_sde_system, backend="jax")

            assert isinstance(integrator, SDEIntegratorBase)
            mock_class.assert_called_once()
            call_kwargs = mock_class.call_args[1]
            assert call_kwargs["backend"] == "jax"

    def test_invalid_backend_raises(self, controlled_sde_system):
        """Test that invalid backend raises ValueError."""
        with pytest.raises(ValueError, match="Invalid backend"):
            SDEIntegratorFactory.create(controlled_sde_system, backend="invalid")


# ============================================================================
# Test Method Routing
# ============================================================================


class TestMethodRouting:
    """Test that methods are routed to correct backends."""

    def test_julia_method_requires_numpy(self, controlled_sde_system):
        """Test Julia methods require NumPy backend."""
        with pytest.raises(ValueError, match="requires backend 'numpy'"):
            SDEIntegratorFactory.create(controlled_sde_system, backend="jax", method="SRIW1")

    def test_torchsde_method_requires_torch(self, controlled_sde_system):
        """Test TorchSDE methods require torch backend."""
        with pytest.raises(ValueError, match="requires backend 'torch'"):
            SDEIntegratorFactory.create(controlled_sde_system, backend="numpy", method="euler")

    def test_diffrax_method_requires_jax(self, controlled_sde_system):
        """Test Diffrax methods require JAX backend."""
        with pytest.raises(ValueError, match="requires backend 'jax'"):
            SDEIntegratorFactory.create(controlled_sde_system, backend="numpy", method="SEA")

    def test_correct_backend_method_combination(self, controlled_sde_system):
        """Test correct backend-method combinations work."""
        mock_integrator = MockSDEIntegrator(controlled_sde_system)

        # NumPy + Julia method
        with patch(
            "src.systems.base.numerical_integration.stochastic.diffeqpy_sde_integrator.DiffEqPySDEIntegrator",
            return_value=mock_integrator,
        ):
            integrator = SDEIntegratorFactory.create(
                controlled_sde_system, backend="numpy", method="EM",
            )
            assert isinstance(integrator, SDEIntegratorBase)

        # Torch + TorchSDE method
        with patch(
            "src.systems.base.numerical_integration.stochastic.torchsde_integrator.TorchSDEIntegrator",
            return_value=mock_integrator,
        ):
            integrator = SDEIntegratorFactory.create(
                controlled_sde_system, backend="torch", method="euler",
            )
            assert isinstance(integrator, SDEIntegratorBase)

        # JAX + Diffrax method
        with patch(
            "src.systems.base.numerical_integration.stochastic.diffrax_sde_integrator.DiffraxSDEIntegrator",
            return_value=mock_integrator,
        ):
            integrator = SDEIntegratorFactory.create(
                controlled_sde_system, backend="jax", method="Euler",
            )
            assert isinstance(integrator, SDEIntegratorBase)


# ============================================================================
# Test Use Case Helpers
# ============================================================================


class TestUseCaseHelpers:
    """Test specialized factory methods for different use cases."""

    def test_auto_selection(self, controlled_sde_system):
        """Test auto() tries backends in order."""
        mock_integrator = MockSDEIntegrator(controlled_sde_system)

        # Mock all backends as unavailable except numpy
        with patch(
            "src.systems.base.numerical_integration.stochastic.diffeqpy_sde_integrator.DiffEqPySDEIntegrator",
            return_value=mock_integrator,
        ):
            integrator = SDEIntegratorFactory.auto(controlled_sde_system)
            assert isinstance(integrator, SDEIntegratorBase)

    def test_for_optimization(self, controlled_sde_system):
        """Test for_optimization() prefers JAX."""
        mock_integrator = MockSDEIntegrator(controlled_sde_system, backend="jax")

        with patch(
            "src.systems.base.numerical_integration.stochastic.diffrax_sde_integrator.DiffraxSDEIntegrator",
            return_value=mock_integrator,
        ):
            integrator = SDEIntegratorFactory.for_optimization(controlled_sde_system)
            assert isinstance(integrator, SDEIntegratorBase)

    def test_for_neural_sde(self, controlled_sde_system):
        """Test for_neural_sde() uses TorchSDE with adjoint."""
        mock_integrator = MockSDEIntegrator(controlled_sde_system, backend="torch")

        with patch(
            "src.systems.base.numerical_integration.stochastic.torchsde_integrator.TorchSDEIntegrator",
            return_value=mock_integrator,
        ) as mock_class:
            integrator = SDEIntegratorFactory.for_neural_sde(controlled_sde_system)

            call_kwargs = mock_class.call_args[1]
            assert call_kwargs["backend"] == "torch"
            assert call_kwargs["method"] == "euler"
            assert call_kwargs["adjoint"] is True

    def test_for_julia(self, controlled_sde_system):
        """Test for_julia() uses DiffEqPy."""
        mock_integrator = MockSDEIntegrator(controlled_sde_system)

        with patch(
            "src.systems.base.numerical_integration.stochastic.diffeqpy_sde_integrator.DiffEqPySDEIntegrator",
            return_value=mock_integrator,
        ) as mock_class:
            integrator = SDEIntegratorFactory.for_julia(controlled_sde_system, algorithm="SRIW1")

            call_kwargs = mock_class.call_args[1]
            assert call_kwargs["backend"] == "numpy"
            assert call_kwargs["algorithm"] == "SRIW1"

    def test_for_monte_carlo(self, additive_noise_system):
        """Test for_monte_carlo() uses weak convergence methods."""
        mock_integrator = MockSDEIntegrator(additive_noise_system)

        with patch(
            "src.systems.base.numerical_integration.stochastic.diffeqpy_sde_integrator.DiffEqPySDEIntegrator",
            return_value=mock_integrator,
        ) as mock_class:
            integrator = SDEIntegratorFactory.for_monte_carlo(
                additive_noise_system, noise_type="additive",
            )

            call_kwargs = mock_class.call_args[1]
            assert call_kwargs["algorithm"] == "SRA3"  # Optimized for additive noise
            assert call_kwargs["convergence_type"] == ConvergenceType.WEAK


# ============================================================================
# Test Autonomous Systems
# ============================================================================


class TestAutonomousSystems:
    """Test factory with autonomous systems (nu=0)."""

    def test_create_autonomous_numpy(self, autonomous_sde_system):
        """Test creation with autonomous system."""
        mock_integrator = MockSDEIntegrator(autonomous_sde_system)

        with patch(
            "src.systems.base.numerical_integration.stochastic.diffeqpy_sde_integrator.DiffEqPySDEIntegrator",
            return_value=mock_integrator,
        ):
            integrator = SDEIntegratorFactory.create(autonomous_sde_system)
            assert isinstance(integrator, SDEIntegratorBase)

    def test_auto_autonomous(self, autonomous_sde_system):
        """Test auto() with autonomous system."""
        mock_integrator = MockSDEIntegrator(autonomous_sde_system)

        with patch(
            "src.systems.base.numerical_integration.stochastic.diffeqpy_sde_integrator.DiffEqPySDEIntegrator",
            return_value=mock_integrator,
        ):
            integrator = SDEIntegratorFactory.auto(autonomous_sde_system)
            assert isinstance(integrator, SDEIntegratorBase)


# ============================================================================
# Test Noise Type Specialization
# ============================================================================


class TestNoiseTypeSpecialization:
    """Test factory recommendations based on noise structure."""

    def test_recommend_additive_noise(self):
        """Test recommendation for additive noise uses SRA3."""
        rec = SDEIntegratorFactory.recommend("monte_carlo", noise_type="additive")

        assert rec["backend"] == "numpy"
        assert rec["method"] == "SRA3"
        assert rec["convergence_type"] == ConvergenceType.WEAK

    def test_recommend_diagonal_noise(self):
        """Test recommendation for diagonal noise uses SRA1."""
        rec = SDEIntegratorFactory.recommend("monte_carlo", noise_type="diagonal")

        assert rec["backend"] == "numpy"
        assert rec["method"] == "SRA1"

    def test_recommend_general_noise(self):
        """Test recommendation for general noise uses EM."""
        rec = SDEIntegratorFactory.recommend("simple", noise_type="general")

        assert rec["backend"] == "numpy"
        assert rec["method"] == "EM"


# ============================================================================
# Test List Methods
# ============================================================================


class TestListMethods:
    """Test method listing functionality."""

    def test_list_all_methods(self):
        """Test listing all available methods."""
        methods = SDEIntegratorFactory.list_methods()

        assert "numpy" in methods
        assert "torch" in methods
        assert "jax" in methods

        # Check Julia methods
        assert "EM" in methods["numpy"]
        assert "SRIW1" in methods["numpy"]

        # Check TorchSDE methods
        assert "euler" in methods["torch"]
        assert "milstein" in methods["torch"]

        # Check Diffrax methods
        assert "Euler" in methods["jax"]
        assert "SEA" in methods["jax"]

    def test_list_backend_specific_methods(self):
        """Test listing methods for specific backend."""
        numpy_methods = SDEIntegratorFactory.list_methods(backend="numpy")

        assert "numpy" in numpy_methods
        assert "torch" not in numpy_methods
        assert "jax" not in numpy_methods

        assert "EM" in numpy_methods["numpy"]


# ============================================================================
# Test Information Queries
# ============================================================================


class TestInformationQueries:
    """Test get_info() and recommendation functions."""

    def test_get_info_numpy_method(self):
        """Test get_info() for Julia method."""
        info = SDEIntegratorFactory.get_info("numpy", "SRIW1")

        assert "name" in info
        assert "description" in info

    def test_get_info_torch_method(self):
        """Test get_info() for TorchSDE method."""
        info = SDEIntegratorFactory.get_info("torch", "euler")

        assert "name" in info
        assert "description" in info

    def test_get_info_jax_method(self):
        """Test get_info() for Diffrax method."""
        info = SDEIntegratorFactory.get_info("jax", "Euler")

        assert "name" in info
        assert "description" in info


# ============================================================================
# Test Convenience Functions
# ============================================================================


class TestConvenienceFunctions:
    """Test convenience wrapper functions."""

    def test_create_sde_integrator(self, controlled_sde_system):
        """Test create_sde_integrator() convenience function."""
        mock_integrator = MockSDEIntegrator(controlled_sde_system)

        with patch(
            "src.systems.base.numerical_integration.stochastic.diffeqpy_sde_integrator.DiffEqPySDEIntegrator",
            return_value=mock_integrator,
        ) as mock_class:
            integrator = create_sde_integrator(controlled_sde_system)

            assert isinstance(integrator, SDEIntegratorBase)
            mock_class.assert_called_once()

    def test_auto_sde_integrator(self, controlled_sde_system):
        """Test auto_sde_integrator() convenience function."""
        mock_integrator = MockSDEIntegrator(controlled_sde_system)

        with patch(
            "src.systems.base.numerical_integration.stochastic.diffeqpy_sde_integrator.DiffEqPySDEIntegrator",
            return_value=mock_integrator,
        ):
            # Make jax and torch unavailable for predictable behavior
            import sys

            jax_backup = sys.modules.get("jax")
            torch_backup = sys.modules.get("torch")

            if "jax" in sys.modules:
                del sys.modules["jax"]
            if "torch" in sys.modules:
                del sys.modules["torch"]

            try:
                # Use patch.dict to set them to None
                with patch.dict("sys.modules", {"jax": None, "torch": None}, clear=False):
                    integrator = auto_sde_integrator(controlled_sde_system, seed=42)
                    assert isinstance(integrator, SDEIntegratorBase)
            finally:
                if jax_backup:
                    sys.modules["jax"] = jax_backup
                if torch_backup:
                    sys.modules["torch"] = torch_backup


# ============================================================================
# Test Options Propagation
# ============================================================================


class TestOptionsPropagation:
    """Test that options are correctly propagated to integrators."""

    def test_dt_propagation(self, controlled_sde_system):
        """Test dt option is propagated."""
        mock_integrator = MockSDEIntegrator(controlled_sde_system)

        with patch(
            "src.systems.base.numerical_integration.stochastic.diffeqpy_sde_integrator.DiffEqPySDEIntegrator",
            return_value=mock_integrator,
        ) as mock_class:
            integrator = SDEIntegratorFactory.create(
                controlled_sde_system, backend="numpy", dt=0.001,
            )

            call_kwargs = mock_class.call_args[1]
            assert call_kwargs["dt"] == 0.001

    def test_seed_propagation(self, controlled_sde_system):
        """Test seed option is propagated."""
        mock_integrator = MockSDEIntegrator(controlled_sde_system)

        with patch(
            "src.systems.base.numerical_integration.stochastic.diffeqpy_sde_integrator.DiffEqPySDEIntegrator",
            return_value=mock_integrator,
        ) as mock_class:
            integrator = SDEIntegratorFactory.create(
                controlled_sde_system, backend="numpy", seed=12345,
            )

            call_kwargs = mock_class.call_args[1]
            assert call_kwargs["seed"] == 12345

    def test_convergence_type_propagation(self, controlled_sde_system):
        """Test convergence_type option is propagated."""
        mock_integrator = MockSDEIntegrator(controlled_sde_system)

        with patch(
            "src.systems.base.numerical_integration.stochastic.diffeqpy_sde_integrator.DiffEqPySDEIntegrator",
            return_value=mock_integrator,
        ) as mock_class:
            integrator = SDEIntegratorFactory.create(
                controlled_sde_system, backend="numpy", convergence_type=ConvergenceType.WEAK,
            )

            call_kwargs = mock_class.call_args[1]
            assert call_kwargs["convergence_type"] == ConvergenceType.WEAK

    def test_custom_options_propagation(self, controlled_sde_system):
        """Test custom options are propagated."""
        mock_integrator = MockSDEIntegrator(controlled_sde_system)

        with patch(
            "src.systems.base.numerical_integration.stochastic.diffeqpy_sde_integrator.DiffEqPySDEIntegrator",
            return_value=mock_integrator,
        ) as mock_class:
            integrator = SDEIntegratorFactory.create(
                controlled_sde_system, backend="numpy", rtol=1e-6, atol=1e-8, custom_option="value",
            )

            call_kwargs = mock_class.call_args[1]
            assert call_kwargs["rtol"] == 1e-6
            assert call_kwargs["atol"] == 1e-8
            assert call_kwargs["custom_option"] == "value"


# ============================================================================
# Integration Tests (End-to-End)
# ============================================================================


class TestEndToEndIntegration:
    """End-to-end integration tests with real mock integrators."""

    def test_controlled_system_full_workflow(self, controlled_sde_system):
        """Test full workflow with controlled system."""
        integrator = MockSDEIntegrator(controlled_sde_system, backend="numpy")

        with patch(
            "src.systems.base.numerical_integration.stochastic.diffeqpy_sde_integrator.DiffEqPySDEIntegrator",
            return_value=integrator,
        ):
            # Create via factory
            integrator = SDEIntegratorFactory.create(
                controlled_sde_system, backend="numpy", method="EM", dt=0.01, seed=42,
            )

            # Set up control
            K = np.array([[1.0, 0.5]])
            u_func = lambda t, x: -K @ x

            # Initial condition
            x0 = np.array([1.0, 0.5])

            # Integrate
            result = integrator.integrate(x0=x0, u_func=u_func, t_span=(0.0, 2.0))

            # Verify result
            assert result["success"]
            assert len(result["t"]) > 10
            assert result["x"].shape == (len(result["t"]), controlled_sde_system.nx)
            assert result["nsteps"] > 0
            assert result["n_paths"] == 1

    def test_autonomous_system_full_workflow(self, autonomous_sde_system):
        """Test full workflow with autonomous system."""
        integrator = MockSDEIntegrator(autonomous_sde_system, backend="numpy")

        with patch(
            "src.systems.base.numerical_integration.stochastic.diffeqpy_sde_integrator.DiffEqPySDEIntegrator",
            return_value=integrator,
        ):
            # Create via factory
            integrator = SDEIntegratorFactory.auto(autonomous_sde_system)

            # No control
            u_func = lambda t, x: None

            # Initial condition
            x0 = np.array([1.0, 0.5])

            # Integrate
            result = integrator.integrate(x0=x0, u_func=u_func, t_span=(0.0, 2.0))

            # Verify result
            assert result["success"]
            assert len(result["t"]) > 10
            assert result["x"].shape == (len(result["t"]), autonomous_sde_system.nx)

    def test_pure_diffusion_full_workflow(self, pure_diffusion_system):
        """Test full workflow with pure diffusion process."""
        integrator = MockSDEIntegrator(pure_diffusion_system, backend="numpy")

        with patch(
            "src.systems.base.numerical_integration.stochastic.diffeqpy_sde_integrator.DiffEqPySDEIntegrator",
            return_value=integrator,
        ):
            # Create via factory
            integrator = SDEIntegratorFactory.create(
                pure_diffusion_system, backend="numpy", dt=0.01, seed=42,
            )

            # No control
            u_func = lambda t, x: None

            # Initial condition
            x0 = np.array([0.0, 0.0])

            # Integrate
            result = integrator.integrate(x0=x0, u_func=u_func, t_span=(0.0, 1.0))

            # Verify result
            assert result["success"]
            assert len(result["t"]) > 10
            # Pure diffusion with zero drift - state changes only from noise
            # Mock doesn't add noise, so state should remain at zero


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
