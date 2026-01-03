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
Unit Tests: Control Framework Integration in DiscreteSystemBase

Tests the control framework integration properties:
- system.control (ControlSynthesis)
- system.analysis (SystemAnalysis)

These tests verify:
1. Lazy instantiation of control utilities
2. Backend consistency propagation
3. Singleton behavior (same instance returned)
4. Integration with actual control functions
5. Error handling and edge cases
6. Discrete-time specific functionality (system_type='discrete')

Test Structure
--------------
- TestControlProperty: Tests for system.control property
- TestAnalysisProperty: Tests for system.analysis property
- TestControlIntegration: Integration tests with real control algorithms
- TestAnalysisIntegration: Integration tests with real analysis algorithms
- TestBackendConsistency: Backend propagation tests
- TestAllBackends: Comprehensive backend tests (NumPy, PyTorch, JAX)
- TestEdgeCases: Edge cases and error handling
"""

import unittest
from unittest.mock import Mock

import numpy as np

from src.control.control_synthesis import ControlSynthesis
from src.control.system_analysis import SystemAnalysis

# Import the base class we're testing
from src.systems.base.core.discrete_system_base import DiscreteSystemBase
from src.types.backends import Backend

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

# ============================================================================
# Mock System for Testing
# ============================================================================


class MockDiscreteSystem(DiscreteSystemBase):
    """
    Minimal concrete implementation for testing control integration.

    Provides a simple 2D linear discrete system for testing:
        x[k+1] = Ad·x[k] + Bd·u[k]
        Ad = [[0.9, 0.1], [0, 0.85]]  (stable discrete system)
        Bd = [[0], [0.1]]
    """

    def __init__(self, dt: float = 0.1, backend: Backend = "numpy"):
        """Initialize with configurable backend and time step."""
        self._dt = dt
        self.backend_obj = Mock()
        self.backend_obj.default_backend = backend
        self.nx = 2
        self.nu = 1
        self.ny = 2

        # System matrices (stable discrete system: |λ| < 1)
        self.Ad = np.array([[0.9, 0.1], [0, 0.85]], dtype=np.float64)
        self.Bd = np.array([[0], [0.1]], dtype=np.float64)
        self.C = np.eye(2)

    @property
    def dt(self) -> float:
        """Return sampling period."""
        return self._dt

    @property
    def backend(self):
        """Mock backend property."""
        return self.backend_obj

    def step(self, x, u=None, k=0):
        """Simple linear discrete dynamics: x[k+1] = Ad·x[k] + Bd·u[k]"""
        u = u if u is not None else np.zeros(self.nu)
        return self.Ad @ x + self.Bd @ u

    def simulate(self, x0, u_sequence=None, n_steps=100, **kwargs):
        """Mock simulation (not tested here)."""
        raise NotImplementedError("Simulation not needed for control tests")

    def linearize(self, x_eq, u_eq=None):
        """Return fixed Ad, Bd matrices (already linear)."""
        return (self.Ad, self.Bd)


# ============================================================================
# Test: system.control Property
# ============================================================================


class TestControlProperty(unittest.TestCase):
    """Test the system.control property and ControlSynthesis integration."""

    def setUp(self):
        """Create fresh mock system for each test."""
        self.system = MockDiscreteSystem(dt=0.1, backend="numpy")

    def test_control_property_exists(self):
        """Test that control property exists and is accessible."""
        self.assertTrue(hasattr(self.system, "control"))
        control = self.system.control
        self.assertIsNotNone(control)

    def test_control_returns_control_synthesis(self):
        """Test that control property returns ControlSynthesis instance."""
        control = self.system.control
        self.assertIsInstance(control, ControlSynthesis)

    def test_control_lazy_instantiation(self):
        """Test that ControlSynthesis is not created until first access."""
        # Before access, _control_synthesis should not exist or be None
        self.assertFalse(
            hasattr(self.system, "_control_synthesis")
            and self.system._control_synthesis is not None,
        )

        # After access, should exist
        _ = self.system.control
        self.assertTrue(hasattr(self.system, "_control_synthesis"))
        self.assertIsNotNone(self.system._control_synthesis)

    def test_control_singleton_behavior(self):
        """Test that multiple accesses return the same instance."""
        control1 = self.system.control
        control2 = self.system.control
        self.assertIs(control1, control2, "Multiple accesses should return same instance")

    def test_control_backend_propagation(self):
        """Test that backend is correctly propagated to ControlSynthesis."""
        # Test with numpy backend
        system_np = MockDiscreteSystem(dt=0.1, backend="numpy")
        control_np = system_np.control
        self.assertEqual(control_np.backend, "numpy")

        # Test with torch backend
        system_torch = MockDiscreteSystem(dt=0.1, backend="torch")
        control_torch = system_torch.control
        self.assertEqual(control_torch.backend, "torch")

        # Test with jax backend
        system_jax = MockDiscreteSystem(dt=0.1, backend="jax")
        control_jax = system_jax.control
        self.assertEqual(control_jax.backend, "jax")

    def test_control_with_no_backend_attribute(self):
        """Test control property when system has no backend attribute."""

        # Create system without backend attribute
        class MinimalSystem(MockDiscreteSystem):
            def __init__(self):
                super().__init__()
                delattr(self, "backend_obj")

        system = MinimalSystem()
        control = system.control

        # Should default to numpy
        self.assertEqual(control.backend, "numpy")

    def test_control_with_invalid_backend_attribute(self):
        """Test control property when backend attribute is malformed."""
        system = MockDiscreteSystem()
        # Give backend_obj without default_backend attribute
        system.backend_obj = Mock(spec=[])  # Empty spec, no attributes

        control = system.control
        # Should default to numpy
        self.assertEqual(control.backend, "numpy")

    def test_control_has_required_methods(self):
        """Test that ControlSynthesis has all required methods."""
        control = self.system.control

        # Check for required methods
        required_methods = ["design_lqr", "design_kalman", "design_lqg"]
        for method_name in required_methods:
            self.assertTrue(
                hasattr(control, method_name), f"ControlSynthesis should have {method_name} method",
            )
            self.assertTrue(
                callable(getattr(control, method_name)), f"{method_name} should be callable",
            )


# ============================================================================
# Test: system.analysis Property
# ============================================================================


class TestAnalysisProperty(unittest.TestCase):
    """Test the system.analysis property and SystemAnalysis integration."""

    def setUp(self):
        """Create fresh mock system for each test."""
        self.system = MockDiscreteSystem(dt=0.1, backend="numpy")

    def test_analysis_property_exists(self):
        """Test that analysis property exists and is accessible."""
        self.assertTrue(hasattr(self.system, "analysis"))
        analysis = self.system.analysis
        self.assertIsNotNone(analysis)

    def test_analysis_returns_system_analysis(self):
        """Test that analysis property returns SystemAnalysis instance."""
        analysis = self.system.analysis
        self.assertIsInstance(analysis, SystemAnalysis)

    def test_analysis_lazy_instantiation(self):
        """Test that SystemAnalysis is not created until first access."""
        # Before access, _system_analysis should not exist or be None
        self.assertFalse(
            hasattr(self.system, "_system_analysis") and self.system._system_analysis is not None,
        )

        # After access, should exist
        _ = self.system.analysis
        self.assertTrue(hasattr(self.system, "_system_analysis"))
        self.assertIsNotNone(self.system._system_analysis)

    def test_analysis_singleton_behavior(self):
        """Test that multiple accesses return the same instance."""
        analysis1 = self.system.analysis
        analysis2 = self.system.analysis
        self.assertIs(analysis1, analysis2, "Multiple accesses should return same instance")

    def test_analysis_backend_propagation(self):
        """Test that backend is correctly propagated to SystemAnalysis."""
        # Test with numpy backend
        system_np = MockDiscreteSystem(dt=0.1, backend="numpy")
        analysis_np = system_np.analysis
        self.assertEqual(analysis_np.backend, "numpy")

        # Test with torch backend
        system_torch = MockDiscreteSystem(dt=0.1, backend="torch")
        analysis_torch = system_torch.analysis
        self.assertEqual(analysis_torch.backend, "torch")

        # Test with jax backend
        system_jax = MockDiscreteSystem(dt=0.1, backend="jax")
        analysis_jax = system_jax.analysis
        self.assertEqual(analysis_jax.backend, "jax")

    def test_analysis_with_no_backend_attribute(self):
        """Test analysis property when system has no backend attribute."""

        # Create system without backend attribute
        class MinimalSystem(MockDiscreteSystem):
            def __init__(self):
                super().__init__()
                delattr(self, "backend_obj")

        system = MinimalSystem()
        analysis = system.analysis

        # Should default to numpy
        self.assertEqual(analysis.backend, "numpy")

    def test_analysis_with_invalid_backend_attribute(self):
        """Test analysis property when backend attribute is malformed."""
        system = MockDiscreteSystem()
        # Give backend_obj without default_backend attribute
        system.backend_obj = Mock(spec=[])  # Empty spec, no attributes

        analysis = system.analysis
        # Should default to numpy
        self.assertEqual(analysis.backend, "numpy")

    def test_analysis_has_required_methods(self):
        """Test that SystemAnalysis has all required methods."""
        analysis = self.system.analysis

        # Check for required methods
        required_methods = [
            "stability",
            "controllability",
            "observability",
            "analyze_linearization",
        ]
        for method_name in required_methods:
            self.assertTrue(
                hasattr(analysis, method_name), f"SystemAnalysis should have {method_name} method",
            )
            self.assertTrue(
                callable(getattr(analysis, method_name)), f"{method_name} should be callable",
            )


# ============================================================================
# Integration Tests: Control Synthesis
# ============================================================================


class TestControlIntegration(unittest.TestCase):
    """Integration tests for system.control with actual control algorithms."""

    def setUp(self):
        """Create system and define control design parameters."""
        self.system = MockDiscreteSystem(dt=0.1, backend="numpy")

        # Get linearization
        self.x_eq = np.zeros(2)
        self.u_eq = np.zeros(1)
        self.Ad, self.Bd = self.system.linearize(self.x_eq, self.u_eq)

        # Define cost matrices for LQR
        self.Q = np.diag([10.0, 1.0])  # Penalize first state more
        self.R = np.array([[0.1]])

        # Define noise covariances for Kalman
        self.C = np.eye(2)  # Full state measurement
        self.Q_proc = 0.01 * np.eye(2)
        self.R_meas = 0.1 * np.eye(2)

    def test_design_lqr_discrete(self):
        """Test LQR design via system.control for discrete-time system."""
        result = self.system.control.design_lqr(
            self.Ad, self.Bd, self.Q, self.R, system_type="discrete",
        )

        # Check result structure
        self.assertIn("gain", result)
        self.assertIn("cost_to_go", result)
        self.assertIn("closed_loop_eigenvalues", result)
        self.assertIn("stability_margin", result)

        # Check gain shape
        K = result["gain"]
        self.assertEqual(K.shape, (self.system.nu, self.system.nx))

        # Check stability (discrete: stability_margin > 0 means max|λ| < 1)
        self.assertGreater(result["stability_margin"], 0, "LQR should stabilize the system")

        # Verify closed-loop eigenvalues are inside unit circle
        eigenvalues = result["closed_loop_eigenvalues"]
        max_magnitude = np.max(np.abs(eigenvalues))
        self.assertLess(max_magnitude, 1.0, "Discrete LQR eigenvalues should be inside unit circle")

    def test_design_lqr_with_cross_coupling(self):
        """Test LQR design with cross-coupling term N."""
        N = np.array([[0.5], [0.1]])

        result = self.system.control.design_lqr(
            self.Ad, self.Bd, self.Q, self.R, N=N, system_type="discrete",
        )

        self.assertIn("gain", result)
        self.assertGreater(result["stability_margin"], 0)

    def test_design_kalman_discrete(self):
        """Test Kalman filter design via system.control."""
        result = self.system.control.design_kalman(
            self.Ad, self.C, self.Q_proc, self.R_meas, system_type="discrete",
        )

        # Check result structure
        self.assertIn("gain", result)
        self.assertIn("error_covariance", result)
        self.assertIn("innovation_covariance", result)
        self.assertIn("observer_eigenvalues", result)

        # Check gain shape
        L = result["gain"]
        self.assertEqual(L.shape, (self.system.nx, self.C.shape[0]))

        # Verify observer eigenvalues are inside unit circle
        eigenvalues = result["observer_eigenvalues"]
        max_magnitude = np.max(np.abs(eigenvalues))
        self.assertLess(max_magnitude, 1.0, "Observer eigenvalues should be inside unit circle")

    def test_design_lqg_discrete(self):
        """Test LQG design via system.control."""
        result = self.system.control.design_lqg(
            self.Ad,
            self.Bd,
            self.C,
            self.Q,
            self.R,
            self.Q_proc,
            self.R_meas,
            system_type="discrete",
        )

        # Check result structure
        self.assertIn("controller_gain", result)
        self.assertIn("estimator_gain", result)
        self.assertIn("controller_riccati", result)
        self.assertIn("estimator_covariance", result)
        self.assertIn("closed_loop_eigenvalues", result)
        self.assertIn("observer_eigenvalues", result)

        # Check gain shapes
        K = result["controller_gain"]
        L = result["estimator_gain"]
        self.assertEqual(K.shape, (self.system.nu, self.system.nx))
        self.assertEqual(L.shape, (self.system.nx, self.C.shape[0]))

        # Verify both controller and observer are stable
        ctrl_eigs = result["closed_loop_eigenvalues"]
        obs_eigs = result["observer_eigenvalues"]

        self.assertLess(
            np.max(np.abs(ctrl_eigs)), 1.0, "Controller eigenvalues should be inside unit circle",
        )
        self.assertLess(
            np.max(np.abs(obs_eigs)), 1.0, "Observer eigenvalues should be inside unit circle",
        )

    def test_control_backend_numpy(self):
        """Test that control synthesis works with numpy backend."""
        system = MockDiscreteSystem(dt=0.1, backend="numpy")
        result = system.control.design_lqr(self.Ad, self.Bd, self.Q, self.R, system_type="discrete")

        # Result should be numpy arrays
        self.assertIsInstance(result["gain"], np.ndarray)

    @unittest.skipUnless(torch_available, "PyTorch not installed")
    def test_control_backend_torch(self):
        """Test that control synthesis works with torch backend."""
        import torch

        system = MockDiscreteSystem(dt=0.1, backend="torch")
        Ad_torch = torch.from_numpy(self.Ad)
        Bd_torch = torch.from_numpy(self.Bd)
        Q_torch = torch.from_numpy(self.Q)
        R_torch = torch.from_numpy(self.R)

        result = system.control.design_lqr(
            Ad_torch, Bd_torch, Q_torch, R_torch, system_type="discrete",
        )

        # Result should be torch tensors
        self.assertIsInstance(result["gain"], torch.Tensor)

    @unittest.skipUnless(jax_available, "JAX not installed")
    def test_control_backend_jax(self):
        """Test that control synthesis works with jax backend."""
        import jax.numpy as jnp
        from jax import config

        # Enable float64 for JAX
        config.update("jax_enable_x64", True)

        system = MockDiscreteSystem(dt=0.1, backend="jax")
        Ad_jax = jnp.array(self.Ad)
        Bd_jax = jnp.array(self.Bd)
        Q_jax = jnp.array(self.Q)
        R_jax = jnp.array(self.R)

        result = system.control.design_lqr(Ad_jax, Bd_jax, Q_jax, R_jax, system_type="discrete")

        # Result should be jax arrays
        self.assertTrue(hasattr(result["gain"], "__array__"))


# ============================================================================
# Integration Tests: System Analysis
# ============================================================================


class TestAnalysisIntegration(unittest.TestCase):
    """Integration tests for system.analysis with actual analysis algorithms."""

    def setUp(self):
        """Create system and get linearization."""
        self.system = MockDiscreteSystem(dt=0.1, backend="numpy")

        self.x_eq = np.zeros(2)
        self.u_eq = np.zeros(1)
        self.Ad, self.Bd = self.system.linearize(self.x_eq, self.u_eq)
        self.C = np.array([[1, 0]])  # Partial measurement

    def test_stability_analysis_discrete(self):
        """Test stability analysis via system.analysis."""
        result = self.system.analysis.stability(self.Ad, system_type="discrete")

        # Check result structure
        self.assertIn("eigenvalues", result)
        self.assertIn("magnitudes", result)
        self.assertIn("max_magnitude", result)
        self.assertIn("spectral_radius", result)
        self.assertIn("is_stable", result)
        self.assertIn("is_marginally_stable", result)
        self.assertIn("is_unstable", result)

        # System should be stable (eigenvalues: 0.9, 0.85)
        self.assertTrue(result["is_stable"], "Test system should be stable")
        self.assertFalse(result["is_unstable"])

        # Check eigenvalues are inside unit circle
        eigenvalues = result["eigenvalues"]
        max_magnitude = np.max(np.abs(eigenvalues))
        self.assertLess(max_magnitude, 1.0)

        # Verify spectral radius
        self.assertAlmostEqual(result["spectral_radius"], max_magnitude)

    def test_controllability_analysis(self):
        """Test controllability analysis via system.analysis."""
        result = self.system.analysis.controllability(self.Ad, self.Bd)

        # Check result structure
        self.assertIn("controllability_matrix", result)
        self.assertIn("rank", result)
        self.assertIn("is_controllable", result)

        # System should be controllable
        self.assertTrue(result["is_controllable"], "Test system should be controllable")
        self.assertEqual(
            result["rank"], self.system.nx, "Controllability matrix should have full rank",
        )

    def test_observability_analysis(self):
        """Test observability analysis via system.analysis."""
        result = self.system.analysis.observability(self.Ad, self.C)

        # Check result structure
        self.assertIn("observability_matrix", result)
        self.assertIn("rank", result)
        self.assertIn("is_observable", result)

        # System should be observable (measure position, can infer velocity)
        self.assertTrue(result["is_observable"], "Test system should be observable")
        self.assertEqual(
            result["rank"], self.system.nx, "Observability matrix should have full rank",
        )

    def test_analyze_linearization_comprehensive(self):
        """Test comprehensive linearization analysis."""
        result = self.system.analysis.analyze_linearization(
            self.Ad, self.Bd, self.C, system_type="discrete",
        )

        # Check result structure
        self.assertIn("stability", result)
        self.assertIn("controllability", result)
        self.assertIn("observability", result)
        self.assertIn("summary", result)

        # Check summary flags
        summary = result["summary"]
        self.assertIn("is_stable", summary)
        self.assertIn("is_controllable", summary)
        self.assertIn("is_observable", summary)
        self.assertIn("ready_for_lqr", summary)
        self.assertIn("ready_for_kalman", summary)
        self.assertIn("ready_for_lqg", summary)

        # System should pass all checks
        self.assertTrue(summary["is_stable"])
        self.assertTrue(summary["is_controllable"])
        self.assertTrue(summary["is_observable"])
        self.assertTrue(summary["ready_for_lqr"])
        self.assertTrue(summary["ready_for_kalman"])
        self.assertTrue(summary["ready_for_lqg"])

    def test_uncontrollable_system(self):
        """Test analysis detects uncontrollable discrete system."""
        # Create truly uncontrollable system
        Ad_unc = np.array([[0.9, 0.1], [0, 0.9]])  # Jordan block
        Bd_unc = np.array([[0], [0]])  # Zero input → uncontrollable

        result = self.system.analysis.controllability(Ad_unc, Bd_unc)

        self.assertFalse(result["is_controllable"])
        self.assertLess(result["rank"], 2)

    def test_partially_controllable_system(self):
        """Test discrete system with one controllable and one uncontrollable mode."""
        # Diagonal system with one zero input
        Ad_partial = np.array([[0.9, 0], [0, 0.85]])
        Bd_partial = np.array([[1], [0]])  # Can only control first state

        result = self.system.analysis.controllability(Ad_partial, Bd_partial)

        # Controllability matrix: [Bd, Ad·Bd] = [[1, 0.9], [0, 0]]
        # Has rank 1, not full rank
        self.assertFalse(result["is_controllable"])
        self.assertEqual(result["rank"], 1)

    def test_unobservable_system(self):
        """Test analysis detects unobservable discrete system."""
        # Create truly unobservable system
        Ad_unobs = np.array([[0.9, 0], [0, 0.9]])  # Diagonal
        C_unobs = np.array([[0, 0]])  # Zero measurement → unobservable

        result = self.system.analysis.observability(Ad_unobs, C_unobs)

        self.assertFalse(result["is_observable"])
        self.assertLess(result["rank"], 2)

    def test_partially_observable_system(self):
        """Test discrete system with one observable and one unobservable mode."""
        # Diagonal system with measurement of only one state
        Ad_partial = np.array([[0.9, 0], [0, 0.85]])
        C_partial = np.array([[1, 0]])  # Can only observe first state

        result = self.system.analysis.observability(Ad_partial, C_partial)

        # Observability matrix: [C; C·Ad] = [[1, 0], [0.9, 0]]
        # Has rank 1, not full rank
        self.assertFalse(result["is_observable"])
        self.assertEqual(result["rank"], 1)

    def test_unstable_discrete_system(self):
        """Test stability analysis detects unstable discrete system."""
        # Create unstable discrete system (eigenvalue > 1)
        Ad_unstable = np.array([[1.1, 0.1], [0, 1.05]])  # Eigenvalues > 1

        result = self.system.analysis.stability(Ad_unstable, system_type="discrete")

        self.assertFalse(result["is_stable"])
        self.assertTrue(result["is_unstable"])
        self.assertGreater(result["spectral_radius"], 1.0)

    def test_marginally_stable_discrete_system(self):
        """Test detection of marginally stable discrete system."""
        # System with eigenvalue on unit circle
        Ad_marginal = np.array([[0, 1], [-1, 0]])  # Eigenvalues: ±j (|λ| = 1)

        result = self.system.analysis.stability(
            Ad_marginal, system_type="discrete", tolerance=1e-10,
        )

        self.assertTrue(result["is_marginally_stable"])
        self.assertAlmostEqual(result["spectral_radius"], 1.0, places=10)


# ============================================================================
# Backend Consistency Tests
# ============================================================================


class TestBackendConsistency(unittest.TestCase):
    """Test backend propagation and consistency across properties."""

    def test_control_and_analysis_same_backend(self):
        """Test that control and analysis use same backend as system."""
        for backend in ["numpy", "torch", "jax"]:
            with self.subTest(backend=backend):
                system = MockDiscreteSystem(dt=0.1, backend=backend)

                control = system.control
                analysis = system.analysis

                self.assertEqual(control.backend, backend)
                self.assertEqual(analysis.backend, backend)

    def test_backend_consistency_after_multiple_accesses(self):
        """Test backend remains consistent after multiple property accesses."""
        system = MockDiscreteSystem(dt=0.1, backend="numpy")

        # Access multiple times
        control1 = system.control
        analysis1 = system.analysis
        control2 = system.control
        analysis2 = system.analysis

        # All should have same backend
        self.assertEqual(control1.backend, "numpy")
        self.assertEqual(analysis1.backend, "numpy")
        self.assertEqual(control2.backend, "numpy")
        self.assertEqual(analysis2.backend, "numpy")

        # Should be same instances
        self.assertIs(control1, control2)
        self.assertIs(analysis1, analysis2)

    def test_backend_fallback_to_numpy(self):
        """Test that backend falls back to numpy if not specified."""

        # Create system without backend
        class NoBackendSystem(MockDiscreteSystem):
            def __init__(self):
                super().__init__()
                delattr(self, "backend_obj")

        system = NoBackendSystem()

        # Both should default to numpy
        self.assertEqual(system.control.backend, "numpy")
        self.assertEqual(system.analysis.backend, "numpy")


# ============================================================================
# Comprehensive Backend Tests (NumPy, PyTorch, JAX)
# ============================================================================


class TestAllBackends(unittest.TestCase):
    """Comprehensive tests for all three backends: NumPy, PyTorch, JAX."""

    def setUp(self):
        """Set up test matrices."""
        # Stable discrete system
        self.Ad_np = np.array([[0.9, 0.1], [0, 0.85]], dtype=np.float64)
        self.Bd_np = np.array([[0], [0.1]], dtype=np.float64)
        self.C_np = np.eye(2)

        # LQR weights
        self.Q_np = np.diag([10.0, 1.0])
        self.R_np = np.array([[0.1]])

        # Kalman noise covariances
        self.Q_proc_np = 0.01 * np.eye(2)
        self.R_meas_np = 0.1 * np.eye(2)

    def test_numpy_backend_lqr(self):
        """Test LQR with NumPy backend."""
        system = MockDiscreteSystem(dt=0.1, backend="numpy")

        result = system.control.design_lqr(
            self.Ad_np, self.Bd_np, self.Q_np, self.R_np, system_type="discrete",
        )

        # Verify result types
        self.assertIsInstance(result["gain"], np.ndarray)
        self.assertIsInstance(result["cost_to_go"], np.ndarray)
        self.assertIsInstance(result["closed_loop_eigenvalues"], np.ndarray)

        # Verify stability
        self.assertGreater(result["stability_margin"], 0)
        self.assertLess(np.max(np.abs(result["closed_loop_eigenvalues"])), 1.0)

    def test_numpy_backend_kalman(self):
        """Test Kalman filter with NumPy backend."""
        system = MockDiscreteSystem(dt=0.1, backend="numpy")

        result = system.control.design_kalman(
            self.Ad_np, self.C_np, self.Q_proc_np, self.R_meas_np, system_type="discrete",
        )

        # Verify result types
        self.assertIsInstance(result["gain"], np.ndarray)
        self.assertIsInstance(result["error_covariance"], np.ndarray)
        self.assertIsInstance(result["observer_eigenvalues"], np.ndarray)

        # Verify observer stability
        self.assertLess(np.max(np.abs(result["observer_eigenvalues"])), 1.0)

    def test_numpy_backend_analysis(self):
        """Test system analysis with NumPy backend."""
        system = MockDiscreteSystem(dt=0.1, backend="numpy")

        # Stability
        stability = system.analysis.stability(self.Ad_np, system_type="discrete")
        self.assertTrue(stability["is_stable"])
        self.assertIsInstance(stability["eigenvalues"], np.ndarray)

        # Controllability
        ctrl = system.analysis.controllability(self.Ad_np, self.Bd_np)
        self.assertTrue(ctrl["is_controllable"])
        self.assertIsInstance(ctrl["controllability_matrix"], np.ndarray)

        # Observability
        obs = system.analysis.observability(self.Ad_np, self.C_np)
        self.assertTrue(obs["is_observable"])
        self.assertIsInstance(obs["observability_matrix"], np.ndarray)

    @unittest.skipUnless(torch_available, "PyTorch not installed")
    def test_pytorch_backend_lqr(self):
        """Test LQR with PyTorch backend."""
        import torch

        system = MockDiscreteSystem(dt=0.1, backend="torch")

        # Convert to torch tensors
        Ad_torch = torch.from_numpy(self.Ad_np)
        Bd_torch = torch.from_numpy(self.Bd_np)
        Q_torch = torch.from_numpy(self.Q_np)
        R_torch = torch.from_numpy(self.R_np)

        result = system.control.design_lqr(
            Ad_torch, Bd_torch, Q_torch, R_torch, system_type="discrete",
        )

        # Verify result types (should be torch tensors)
        self.assertIsInstance(result["gain"], torch.Tensor)
        self.assertIsInstance(result["cost_to_go"], torch.Tensor)
        self.assertIsInstance(result["closed_loop_eigenvalues"], torch.Tensor)

        # Verify stability
        self.assertGreater(result["stability_margin"], 0)
        eigenvalues_np = result["closed_loop_eigenvalues"].numpy()
        self.assertLess(np.max(np.abs(eigenvalues_np)), 1.0)

    @unittest.skipUnless(torch_available, "PyTorch not installed")
    def test_pytorch_backend_kalman(self):
        """Test Kalman filter with PyTorch backend."""
        import torch

        system = MockDiscreteSystem(dt=0.1, backend="torch")

        # Convert to torch tensors
        Ad_torch = torch.from_numpy(self.Ad_np)
        C_torch = torch.from_numpy(self.C_np)
        Q_proc_torch = torch.from_numpy(self.Q_proc_np)
        R_meas_torch = torch.from_numpy(self.R_meas_np)

        result = system.control.design_kalman(
            Ad_torch, C_torch, Q_proc_torch, R_meas_torch, system_type="discrete",
        )

        # Verify result types
        self.assertIsInstance(result["gain"], torch.Tensor)
        self.assertIsInstance(result["error_covariance"], torch.Tensor)
        self.assertIsInstance(result["observer_eigenvalues"], torch.Tensor)

        # Verify observer stability
        eigenvalues_np = result["observer_eigenvalues"].numpy()
        self.assertLess(np.max(np.abs(eigenvalues_np)), 1.0)

    @unittest.skipUnless(torch_available, "PyTorch not installed")
    def test_pytorch_backend_lqg(self):
        """Test LQG with PyTorch backend."""
        import torch

        system = MockDiscreteSystem(dt=0.1, backend="torch")

        # Convert to torch tensors
        Ad_torch = torch.from_numpy(self.Ad_np)
        Bd_torch = torch.from_numpy(self.Bd_np)
        C_torch = torch.from_numpy(self.C_np)
        Q_torch = torch.from_numpy(self.Q_np)
        R_torch = torch.from_numpy(self.R_np)
        Q_proc_torch = torch.from_numpy(self.Q_proc_np)
        R_meas_torch = torch.from_numpy(self.R_meas_np)

        result = system.control.design_lqg(
            Ad_torch,
            Bd_torch,
            C_torch,
            Q_torch,
            R_torch,
            Q_proc_torch,
            R_meas_torch,
            system_type="discrete",
        )

        # Verify result types
        self.assertIsInstance(result["controller_gain"], torch.Tensor)
        self.assertIsInstance(result["estimator_gain"], torch.Tensor)

        # Verify both components are stable
        ctrl_eigs_np = result["closed_loop_eigenvalues"].numpy()
        obs_eigs_np = result["observer_eigenvalues"].numpy()
        self.assertLess(np.max(np.abs(ctrl_eigs_np)), 1.0)
        self.assertLess(np.max(np.abs(obs_eigs_np)), 1.0)

    @unittest.skipUnless(torch_available, "PyTorch not installed")
    def test_pytorch_backend_analysis(self):
        """Test system analysis with PyTorch backend."""
        import torch

        system = MockDiscreteSystem(dt=0.1, backend="torch")

        Ad_torch = torch.from_numpy(self.Ad_np)
        Bd_torch = torch.from_numpy(self.Bd_np)
        C_torch = torch.from_numpy(self.C_np)

        # Stability
        stability = system.analysis.stability(Ad_torch, system_type="discrete")
        self.assertTrue(stability["is_stable"])

        # Controllability
        ctrl = system.analysis.controllability(Ad_torch, Bd_torch)
        self.assertTrue(ctrl["is_controllable"])

        # Observability
        obs = system.analysis.observability(Ad_torch, C_torch)
        self.assertTrue(obs["is_observable"])

    @unittest.skipUnless(jax_available, "JAX not installed")
    def test_jax_backend_lqr(self):
        """Test LQR with JAX backend."""
        import jax.numpy as jnp
        from jax import config

        # Enable float64 for JAX to match NumPy/PyTorch precision
        config.update("jax_enable_x64", True)

        system = MockDiscreteSystem(dt=0.1, backend="jax")

        # Convert to JAX arrays
        Ad_jax = jnp.array(self.Ad_np)
        Bd_jax = jnp.array(self.Bd_np)
        Q_jax = jnp.array(self.Q_np)
        R_jax = jnp.array(self.R_np)

        result = system.control.design_lqr(Ad_jax, Bd_jax, Q_jax, R_jax, system_type="discrete")

        # Verify result types (should be JAX arrays)
        self.assertTrue(hasattr(result["gain"], "__array__"))
        self.assertTrue(hasattr(result["cost_to_go"], "__array__"))
        self.assertTrue(hasattr(result["closed_loop_eigenvalues"], "__array__"))

        # Verify stability
        self.assertGreater(result["stability_margin"], 0)
        eigenvalues_np = np.array(result["closed_loop_eigenvalues"])
        self.assertLess(np.max(np.abs(eigenvalues_np)), 1.0)

    @unittest.skipUnless(jax_available, "JAX not installed")
    def test_jax_backend_kalman(self):
        """Test Kalman filter with JAX backend."""
        import jax.numpy as jnp
        from jax import config

        # Enable float64 for JAX to match NumPy/PyTorch precision
        config.update("jax_enable_x64", True)

        system = MockDiscreteSystem(dt=0.1, backend="jax")

        # Convert to JAX arrays
        Ad_jax = jnp.array(self.Ad_np)
        C_jax = jnp.array(self.C_np)
        Q_proc_jax = jnp.array(self.Q_proc_np)
        R_meas_jax = jnp.array(self.R_meas_np)

        result = system.control.design_kalman(
            Ad_jax, C_jax, Q_proc_jax, R_meas_jax, system_type="discrete",
        )

        # Verify result types
        self.assertTrue(hasattr(result["gain"], "__array__"))
        self.assertTrue(hasattr(result["error_covariance"], "__array__"))
        self.assertTrue(hasattr(result["observer_eigenvalues"], "__array__"))

        # Verify observer stability
        eigenvalues_np = np.array(result["observer_eigenvalues"])
        self.assertLess(np.max(np.abs(eigenvalues_np)), 1.0)

    @unittest.skipUnless(jax_available, "JAX not installed")
    def test_jax_backend_lqg(self):
        """Test LQG with JAX backend."""
        import jax.numpy as jnp
        from jax import config

        # Enable float64 for JAX to match NumPy/PyTorch precision
        config.update("jax_enable_x64", True)

        system = MockDiscreteSystem(dt=0.1, backend="jax")

        # Convert to JAX arrays
        Ad_jax = jnp.array(self.Ad_np)
        Bd_jax = jnp.array(self.Bd_np)
        C_jax = jnp.array(self.C_np)
        Q_jax = jnp.array(self.Q_np)
        R_jax = jnp.array(self.R_np)
        Q_proc_jax = jnp.array(self.Q_proc_np)
        R_meas_jax = jnp.array(self.R_meas_np)

        result = system.control.design_lqg(
            Ad_jax, Bd_jax, C_jax, Q_jax, R_jax, Q_proc_jax, R_meas_jax, system_type="discrete",
        )

        # Verify result types
        self.assertTrue(hasattr(result["controller_gain"], "__array__"))
        self.assertTrue(hasattr(result["estimator_gain"], "__array__"))

        # Verify both components are stable
        ctrl_eigs_np = np.array(result["closed_loop_eigenvalues"])
        obs_eigs_np = np.array(result["observer_eigenvalues"])
        self.assertLess(np.max(np.abs(ctrl_eigs_np)), 1.0)
        self.assertLess(np.max(np.abs(obs_eigs_np)), 1.0)

    @unittest.skipUnless(jax_available, "JAX not installed")
    def test_jax_backend_analysis(self):
        """Test system analysis with JAX backend."""
        import jax.numpy as jnp
        from jax import config

        # Enable float64 for JAX to match NumPy/PyTorch precision
        config.update("jax_enable_x64", True)

        system = MockDiscreteSystem(dt=0.1, backend="jax")

        Ad_jax = jnp.array(self.Ad_np)
        Bd_jax = jnp.array(self.Bd_np)
        C_jax = jnp.array(self.C_np)

        # Stability
        stability = system.analysis.stability(Ad_jax, system_type="discrete")
        self.assertTrue(stability["is_stable"])

        # Controllability
        ctrl = system.analysis.controllability(Ad_jax, Bd_jax)
        self.assertTrue(ctrl["is_controllable"])

        # Observability
        obs = system.analysis.observability(Ad_jax, C_jax)
        self.assertTrue(obs["is_observable"])

    def test_backend_result_equivalence(self):
        """Test that all backends produce equivalent results (within tolerance)."""
        backends_to_test = ["numpy"]

        # Add optional backends if available
        if torch_available:
            backends_to_test.append("torch")
        if jax_available:
            backends_to_test.append("jax")

        # Enable float64 for JAX if available
        if "jax" in backends_to_test:
            from jax import config

            config.update("jax_enable_x64", True)

        results = {}

        for backend in backends_to_test:
            with self.subTest(backend=backend):
                system = MockDiscreteSystem(dt=0.1, backend=backend)

                # Convert inputs to appropriate backend
                if backend == "numpy":
                    Ad, Bd = self.Ad_np, self.Bd_np
                    Q, R = self.Q_np, self.R_np
                elif backend == "torch":
                    import torch

                    Ad = torch.from_numpy(self.Ad_np)
                    Bd = torch.from_numpy(self.Bd_np)
                    Q = torch.from_numpy(self.Q_np)
                    R = torch.from_numpy(self.R_np)
                elif backend == "jax":
                    import jax.numpy as jnp

                    Ad = jnp.array(self.Ad_np)
                    Bd = jnp.array(self.Bd_np)
                    Q = jnp.array(self.Q_np)
                    R = jnp.array(self.R_np)

                # Run LQR
                result = system.control.design_lqr(Ad, Bd, Q, R, system_type="discrete")

                # Convert to numpy for comparison
                gain_np = np.asarray(result["gain"])
                results[backend] = gain_np

        # Compare results across backends (should be nearly identical)
        if len(results) > 1:
            ref_gain = results["numpy"]
            for backend, gain in results.items():
                if backend != "numpy":
                    np.testing.assert_allclose(
                        gain,
                        ref_gain,
                        rtol=1e-10,
                        atol=1e-12,
                        err_msg=f"{backend} backend produces different results than NumPy",
                    )


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""

    def test_control_with_existing_control_synthesis(self):
        """Test control property when _control_synthesis already exists."""
        system = MockDiscreteSystem()

        # Manually set _control_synthesis
        existing_control = ControlSynthesis(backend="numpy")
        system._control_synthesis = existing_control

        # Property should return existing instance
        control = system.control
        self.assertIs(control, existing_control)

    def test_analysis_with_existing_system_analysis(self):
        """Test analysis property when _system_analysis already exists."""
        system = MockDiscreteSystem()

        # Manually set _system_analysis
        existing_analysis = SystemAnalysis(backend="numpy")
        system._system_analysis = existing_analysis

        # Property should return existing instance
        analysis = system.analysis
        self.assertIs(analysis, existing_analysis)

    def test_control_property_thread_safety(self):
        """Test that control property handles concurrent access gracefully."""
        import threading

        system = MockDiscreteSystem()
        controls = []

        def access_control():
            controls.append(system.control)

        # Create multiple threads accessing control property
        threads = [threading.Thread(target=access_control) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All should be same instance
        first_control = controls[0]
        for control in controls[1:]:
            self.assertIs(control, first_control)

    def test_analysis_property_thread_safety(self):
        """Test that analysis property handles concurrent access gracefully."""
        import threading

        system = MockDiscreteSystem()
        analyses = []

        def access_analysis():
            analyses.append(system.analysis)

        # Create multiple threads accessing analysis property
        threads = [threading.Thread(target=access_analysis) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All should be same instance
        first_analysis = analyses[0]
        for analysis in analyses[1:]:
            self.assertIs(analysis, first_analysis)

    def test_discrete_system_type_consistency(self):
        """Test that discrete systems use system_type='discrete' consistently."""
        system = MockDiscreteSystem()
        Ad, Bd = system.linearize(np.zeros(2), np.zeros(1))
        Q = np.diag([10.0, 1.0])
        R = np.array([[0.1]])

        # Using system_type='discrete' should work
        result = system.control.design_lqr(Ad, Bd, Q, R, system_type="discrete")
        self.assertGreater(result["stability_margin"], 0)

        # Verify closed-loop is stable (discrete: |λ| < 1)
        eigenvalues = result["closed_loop_eigenvalues"]
        self.assertLess(np.max(np.abs(eigenvalues)), 1.0)


# ============================================================================
# Helper Functions
# ============================================================================


def _check_package_available(package_name):
    """Check if a package is available for import."""
    try:
        __import__(package_name)
        return True
    except ImportError:
        return False


# ============================================================================
# Test Suite Configuration
# ============================================================================


def suite():
    """Create test suite for discrete control framework integration."""
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestControlProperty))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestAnalysisProperty))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestControlIntegration))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestAnalysisIntegration))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestBackendConsistency))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestAllBackends))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestEdgeCases))

    return suite


if __name__ == "__main__":
    # Run with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())
