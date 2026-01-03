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
Unit Tests: Control Framework Integration in ContinuousSystemBase

Tests the control framework integration properties:
- system.control (ControlSynthesis)
- system.analysis (SystemAnalysis)

These tests verify:
1. Lazy instantiation of control utilities
2. Backend consistency propagation
3. Singleton behavior (same instance returned)
4. Integration with actual control functions
5. Error handling and edge cases

Test Structure
--------------
- TestControlProperty: Tests for system.control property
- TestAnalysisProperty: Tests for system.analysis property
- TestControlIntegration: Integration tests with real control algorithms
- TestAnalysisIntegration: Integration tests with real analysis algorithms
- TestBackendConsistency: Backend propagation tests
"""

import unittest
from unittest.mock import Mock

import numpy as np

from src.control.control_synthesis import ControlSynthesis
from src.control.system_analysis import SystemAnalysis

# Import the base class we're testing
from src.systems.base.core.continuous_system_base import ContinuousSystemBase
from src.types.backends import Backend

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


class MockContinuousSystem(ContinuousSystemBase):
    """
    Minimal concrete implementation for testing control integration.

    Provides a simple 2D linear system for testing:
        dx/dt = Ax + Bu
        A = [[0, 1], [-2, -3]]  (stable continuous system)
        B = [[0], [1]]
    """

    def __init__(self, backend: Backend = "numpy"):
        """Initialize with configurable backend."""
        self.backend_obj = Mock()
        self.backend_obj.default_backend = backend
        self.nx = 2
        self.nu = 1
        self.ny = 2

        # System matrices (stable)
        self.A = np.array([[0, 1], [-2, -3]], dtype=np.float64)
        self.B = np.array([[0], [1]], dtype=np.float64)
        self.C = np.eye(2)

    @property
    def backend(self):
        """Mock backend property."""
        return self.backend_obj

    def __call__(self, x, u=None, t=0.0, backend=None):
        """Simple linear dynamics: dx/dt = Ax + Bu"""
        u = u if u is not None else np.zeros(self.nu)
        return self.A @ x + self.B @ u

    def integrate(self, x0, u=None, t_span=(0.0, 10.0), method="RK45", **kwargs):
        """Mock integration (not tested here)"""
        raise NotImplementedError("Integration not needed for control tests")

    def linearize(self, x_eq, u_eq=None):
        """Return fixed A, B matrices (already linear)"""
        return (self.A, self.B)


# ============================================================================
# Test: system.control Property
# ============================================================================


class TestControlProperty(unittest.TestCase):
    """Test the system.control property and ControlSynthesis integration."""

    def setUp(self):
        """Create fresh mock system for each test."""
        self.system = MockContinuousSystem(backend="numpy")

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
        system_np = MockContinuousSystem(backend="numpy")
        control_np = system_np.control
        self.assertEqual(control_np.backend, "numpy")

        # Test with torch backend
        system_torch = MockContinuousSystem(backend="torch")
        control_torch = system_torch.control
        self.assertEqual(control_torch.backend, "torch")

        # Test with jax backend
        system_jax = MockContinuousSystem(backend="jax")
        control_jax = system_jax.control
        self.assertEqual(control_jax.backend, "jax")

    def test_control_with_no_backend_attribute(self):
        """Test control property when system has no backend attribute."""

        # Create system without backend attribute
        class MinimalSystem(MockContinuousSystem):
            def __init__(self):
                super().__init__()
                delattr(self, "backend_obj")

        system = MinimalSystem()
        control = system.control

        # Should default to numpy
        self.assertEqual(control.backend, "numpy")

    def test_control_with_invalid_backend_attribute(self):
        """Test control property when backend attribute is malformed."""
        system = MockContinuousSystem()
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
        self.system = MockContinuousSystem(backend="numpy")

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
        system_np = MockContinuousSystem(backend="numpy")
        analysis_np = system_np.analysis
        self.assertEqual(analysis_np.backend, "numpy")

        # Test with torch backend
        system_torch = MockContinuousSystem(backend="torch")
        analysis_torch = system_torch.analysis
        self.assertEqual(analysis_torch.backend, "torch")

        # Test with jax backend
        system_jax = MockContinuousSystem(backend="jax")
        analysis_jax = system_jax.analysis
        self.assertEqual(analysis_jax.backend, "jax")

    def test_analysis_with_no_backend_attribute(self):
        """Test analysis property when system has no backend attribute."""

        # Create system without backend attribute
        class MinimalSystem(MockContinuousSystem):
            def __init__(self):
                super().__init__()
                delattr(self, "backend_obj")

        system = MinimalSystem()
        analysis = system.analysis

        # Should default to numpy
        self.assertEqual(analysis.backend, "numpy")

    def test_analysis_with_invalid_backend_attribute(self):
        """Test analysis property when backend attribute is malformed."""
        system = MockContinuousSystem()
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
        self.system = MockContinuousSystem(backend="numpy")

        # Get linearization
        self.x_eq = np.zeros(2)
        self.u_eq = np.zeros(1)
        self.A, self.B = self.system.linearize(self.x_eq, self.u_eq)

        # Define cost matrices for LQR
        self.Q = np.diag([10.0, 1.0])  # Penalize position more
        self.R = np.array([[0.1]])

        # Define noise covariances for Kalman
        self.C = np.eye(2)  # Full state measurement
        self.Q_proc = 0.01 * np.eye(2)
        self.R_meas = 0.1 * np.eye(2)

    def test_design_lqr_continuous(self):
        """Test LQR design via system.control for continuous-time system."""
        result = self.system.control.design_lqr(
            self.A, self.B, self.Q, self.R, system_type="continuous",
        )

        # Check result structure
        self.assertIn("gain", result)
        self.assertIn("cost_to_go", result)
        self.assertIn("closed_loop_eigenvalues", result)
        self.assertIn("stability_margin", result)

        # Check gain shape
        K = result["gain"]
        self.assertEqual(K.shape, (self.system.nu, self.system.nx))

        # Check stability
        self.assertGreater(result["stability_margin"], 0, "LQR should stabilize the system")

        # Verify closed-loop eigenvalues are in left half-plane
        eigenvalues = result["closed_loop_eigenvalues"]
        max_real = np.max(np.real(eigenvalues))
        self.assertLess(max_real, 0, "Continuous LQR eigenvalues should have negative real parts")

    def test_design_lqr_with_cross_coupling(self):
        """Test LQR design with cross-coupling term N."""
        N = np.array([[0.5], [0.1]])

        result = self.system.control.design_lqr(
            self.A, self.B, self.Q, self.R, N=N, system_type="continuous",
        )

        self.assertIn("gain", result)
        self.assertGreater(result["stability_margin"], 0)

    def test_design_kalman_continuous(self):
        """Test Kalman filter design via system.control."""
        result = self.system.control.design_kalman(
            self.A, self.C, self.Q_proc, self.R_meas, system_type="continuous",
        )

        # Check result structure
        self.assertIn("gain", result)
        self.assertIn("error_covariance", result)
        self.assertIn("innovation_covariance", result)
        self.assertIn("observer_eigenvalues", result)

        # Check gain shape
        L = result["gain"]
        self.assertEqual(L.shape, (self.system.nx, self.C.shape[0]))

        # Verify observer eigenvalues are in left half-plane
        eigenvalues = result["observer_eigenvalues"]
        max_real = np.max(np.real(eigenvalues))
        self.assertLess(max_real, 0, "Observer eigenvalues should have negative real parts")

    def test_design_lqg_continuous(self):
        """Test LQG design via system.control."""
        result = self.system.control.design_lqg(
            self.A,
            self.B,
            self.C,
            self.Q,
            self.R,
            self.Q_proc,
            self.R_meas,
            system_type="continuous",
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

        self.assertLess(np.max(np.real(ctrl_eigs)), 0, "Controller eigenvalues should be stable")
        self.assertLess(np.max(np.real(obs_eigs)), 0, "Observer eigenvalues should be stable")

    def test_control_backend_numpy(self):
        """Test that control synthesis works with numpy backend."""
        system = MockContinuousSystem(backend="numpy")
        result = system.control.design_lqr(self.A, self.B, self.Q, self.R, system_type="continuous")

        # Result should be numpy arrays
        self.assertIsInstance(result["gain"], np.ndarray)

    @unittest.skipIf(not torch_available, "Requires PyTorch installation")
    def test_control_backend_torch(self):
        """Test that control synthesis works with torch backend."""
        import torch

        system = MockContinuousSystem(backend="torch")
        A_torch = torch.from_numpy(self.A)
        B_torch = torch.from_numpy(self.B)
        Q_torch = torch.from_numpy(self.Q)
        R_torch = torch.from_numpy(self.R)

        result = system.control.design_lqr(
            A_torch, B_torch, Q_torch, R_torch, system_type="continuous",
        )

        # Result should be torch tensors
        self.assertIsInstance(result["gain"], torch.Tensor)

    @unittest.skipIf(not jax_available, "Requires JAX installation")
    def test_control_backend_jax(self):
        """Test that control synthesis works with jax backend."""
        import jax.numpy as jnp

        system = MockContinuousSystem(backend="jax")
        A_jax = jnp.array(self.A)
        B_jax = jnp.array(self.B)
        Q_jax = jnp.array(self.Q)
        R_jax = jnp.array(self.R)

        result = system.control.design_lqr(A_jax, B_jax, Q_jax, R_jax, system_type="continuous")

        # Result should be jax arrays
        self.assertTrue(hasattr(result["gain"], "__array__"))  # JAX array protocol


# ============================================================================
# Integration Tests: System Analysis
# ============================================================================


class TestAnalysisIntegration(unittest.TestCase):
    """Integration tests for system.analysis with actual analysis algorithms."""

    def setUp(self):
        """Create system and get linearization."""
        self.system = MockContinuousSystem(backend="numpy")

        self.x_eq = np.zeros(2)
        self.u_eq = np.zeros(1)
        self.A, self.B = self.system.linearize(self.x_eq, self.u_eq)
        self.C = np.array([[1, 0]])  # Partial measurement

    def test_stability_analysis_continuous(self):
        """Test stability analysis via system.analysis."""
        result = self.system.analysis.stability(self.A, system_type="continuous")

        # Check result structure
        self.assertIn("eigenvalues", result)
        self.assertIn("magnitudes", result)
        self.assertIn("max_magnitude", result)
        self.assertIn("spectral_radius", result)
        self.assertIn("is_stable", result)
        self.assertIn("is_marginally_stable", result)
        self.assertIn("is_unstable", result)

        # System should be stable (eigenvalues: -1, -2)
        self.assertTrue(result["is_stable"], "Test system should be stable")
        self.assertFalse(result["is_unstable"])

        # Check eigenvalues are in left half-plane
        eigenvalues = result["eigenvalues"]
        max_real = np.max(np.real(eigenvalues))
        self.assertLess(max_real, 0)

    def test_controllability_analysis(self):
        """Test controllability analysis via system.analysis."""
        result = self.system.analysis.controllability(self.A, self.B)

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
        result = self.system.analysis.observability(self.A, self.C)

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
            self.A, self.B, self.C, system_type="continuous",
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
        """Test analysis detects uncontrollable system."""
        # Create truly uncontrollable system
        # A with repeated eigenvalue, B not in span of all eigenvectors
        A_unc = np.array([[1, 1], [0, 1]])  # Jordan block (repeated eigenvalue λ=1)
        B_unc = np.array([[0], [0]])  # Zero input → uncontrollable

        result = self.system.analysis.controllability(A_unc, B_unc)

        self.assertFalse(result["is_controllable"])
        self.assertLess(result["rank"], 2)

    def test_partially_controllable_system(self):
        """Test system with one controllable and one uncontrollable mode."""
        # Diagonal system with one zero input
        A_partial = np.array([[1, 0], [0, 2]])
        B_partial = np.array([[1], [0]])  # Can only control first state

        result = self.system.analysis.controllability(A_partial, B_partial)

        # Controllability matrix: [B, AB] = [[1, 1], [0, 0]]
        # Has rank 1, not full rank
        self.assertFalse(result["is_controllable"])
        self.assertEqual(result["rank"], 1)

    def test_unobservable_system(self):
        """Test analysis detects unobservable system."""
        # Create truly unobservable system
        # Diagonal A with measurement that can't distinguish states
        A_unobs = np.array([[1, 0], [0, 1]])  # Two independent modes
        C_unobs = np.array([[0, 0]])  # Zero measurement → unobservable

        result = self.system.analysis.observability(A_unobs, C_unobs)

        self.assertFalse(result["is_observable"])
        self.assertLess(result["rank"], 2)

    def test_partially_observable_system(self):
        """Test system with one observable and one unobservable mode."""
        # Diagonal system with measurement of only one state
        A_partial = np.array([[1, 0], [0, 2]])
        C_partial = np.array([[1, 0]])  # Can only observe first state

        result = self.system.analysis.observability(A_partial, C_partial)

        # Observability matrix: [C; CA] = [[1, 0], [1, 0]]
        # Has rank 1, not full rank
        self.assertFalse(result["is_observable"])
        self.assertEqual(result["rank"], 1)

    def test_unstable_system(self):
        """Test stability analysis detects unstable system."""
        # Create unstable system
        A_unstable = np.array([[1, 1], [0, 1]])  # Eigenvalue at +1

        result = self.system.analysis.stability(A_unstable, system_type="continuous")

        self.assertFalse(result["is_stable"])
        self.assertTrue(result["is_unstable"])


# ============================================================================
# Backend Consistency Tests
# ============================================================================


class TestBackendConsistency(unittest.TestCase):
    """Test backend propagation and consistency across properties."""

    def test_control_and_analysis_same_backend(self):
        """Test that control and analysis use same backend as system."""
        for backend in ["numpy", "torch", "jax"]:
            with self.subTest(backend=backend):
                system = MockContinuousSystem(backend=backend)

                control = system.control
                analysis = system.analysis

                self.assertEqual(control.backend, backend)
                self.assertEqual(analysis.backend, backend)

    def test_backend_consistency_after_multiple_accesses(self):
        """Test backend remains consistent after multiple property accesses."""
        system = MockContinuousSystem(backend="numpy")

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
        class NoBackendSystem(MockContinuousSystem):
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
        # Stable continuous system
        self.A_np = np.array([[0, 1], [-2, -3]], dtype=np.float64)
        self.B_np = np.array([[0], [1]], dtype=np.float64)
        self.C_np = np.eye(2)

        # LQR weights
        self.Q_np = np.diag([10.0, 1.0])
        self.R_np = np.array([[0.1]])

        # Kalman noise covariances
        self.Q_proc_np = 0.01 * np.eye(2)
        self.R_meas_np = 0.1 * np.eye(2)

    def test_numpy_backend_lqr(self):
        """Test LQR with NumPy backend."""
        system = MockContinuousSystem(backend="numpy")

        result = system.control.design_lqr(
            self.A_np, self.B_np, self.Q_np, self.R_np, system_type="continuous",
        )

        # Verify result types
        self.assertIsInstance(result["gain"], np.ndarray)
        self.assertIsInstance(result["cost_to_go"], np.ndarray)
        self.assertIsInstance(result["closed_loop_eigenvalues"], np.ndarray)

        # Verify stability
        self.assertGreater(result["stability_margin"], 0)
        self.assertLess(np.max(np.real(result["closed_loop_eigenvalues"])), 0)

    def test_numpy_backend_kalman(self):
        """Test Kalman filter with NumPy backend."""
        system = MockContinuousSystem(backend="numpy")

        result = system.control.design_kalman(
            self.A_np, self.C_np, self.Q_proc_np, self.R_meas_np, system_type="continuous",
        )

        # Verify result types
        self.assertIsInstance(result["gain"], np.ndarray)
        self.assertIsInstance(result["error_covariance"], np.ndarray)
        self.assertIsInstance(result["observer_eigenvalues"], np.ndarray)

        # Verify observer stability
        self.assertLess(np.max(np.real(result["observer_eigenvalues"])), 0)

    def test_numpy_backend_analysis(self):
        """Test system analysis with NumPy backend."""
        system = MockContinuousSystem(backend="numpy")

        # Stability
        stability = system.analysis.stability(self.A_np, system_type="continuous")
        self.assertTrue(stability["is_stable"])
        self.assertIsInstance(stability["eigenvalues"], np.ndarray)

        # Controllability
        ctrl = system.analysis.controllability(self.A_np, self.B_np)
        self.assertTrue(ctrl["is_controllable"])
        self.assertIsInstance(ctrl["controllability_matrix"], np.ndarray)

        # Observability
        obs = system.analysis.observability(self.A_np, self.C_np)
        self.assertTrue(obs["is_observable"])
        self.assertIsInstance(obs["observability_matrix"], np.ndarray)

    @unittest.skipUnless(torch_available, "PyTorch not installed")
    def test_pytorch_backend_lqr(self):
        """Test LQR with PyTorch backend."""
        import torch

        system = MockContinuousSystem(backend="torch")

        # Convert to torch tensors
        A_torch = torch.from_numpy(self.A_np)
        B_torch = torch.from_numpy(self.B_np)
        Q_torch = torch.from_numpy(self.Q_np)
        R_torch = torch.from_numpy(self.R_np)

        result = system.control.design_lqr(
            A_torch, B_torch, Q_torch, R_torch, system_type="continuous",
        )

        # Verify result types (should be torch tensors)
        self.assertIsInstance(result["gain"], torch.Tensor)
        self.assertIsInstance(result["cost_to_go"], torch.Tensor)
        self.assertIsInstance(result["closed_loop_eigenvalues"], torch.Tensor)

        # Verify stability
        self.assertGreater(result["stability_margin"], 0)
        eigenvalues_np = result["closed_loop_eigenvalues"].numpy()
        self.assertLess(np.max(np.real(eigenvalues_np)), 0)

    @unittest.skipUnless(torch_available, "PyTorch not installed")
    def test_pytorch_backend_kalman(self):
        """Test Kalman filter with PyTorch backend."""
        import torch

        system = MockContinuousSystem(backend="torch")

        # Convert to torch tensors
        A_torch = torch.from_numpy(self.A_np)
        C_torch = torch.from_numpy(self.C_np)
        Q_proc_torch = torch.from_numpy(self.Q_proc_np)
        R_meas_torch = torch.from_numpy(self.R_meas_np)

        result = system.control.design_kalman(
            A_torch, C_torch, Q_proc_torch, R_meas_torch, system_type="continuous",
        )

        # Verify result types
        self.assertIsInstance(result["gain"], torch.Tensor)
        self.assertIsInstance(result["error_covariance"], torch.Tensor)
        self.assertIsInstance(result["observer_eigenvalues"], torch.Tensor)

        # Verify observer stability
        eigenvalues_np = result["observer_eigenvalues"].numpy()
        self.assertLess(np.max(np.real(eigenvalues_np)), 0)

    @unittest.skipUnless(torch_available, "PyTorch not installed")
    def test_pytorch_backend_lqg(self):
        """Test LQG with PyTorch backend."""
        import torch

        system = MockContinuousSystem(backend="torch")

        # Convert to torch tensors
        A_torch = torch.from_numpy(self.A_np)
        B_torch = torch.from_numpy(self.B_np)
        C_torch = torch.from_numpy(self.C_np)
        Q_torch = torch.from_numpy(self.Q_np)
        R_torch = torch.from_numpy(self.R_np)
        Q_proc_torch = torch.from_numpy(self.Q_proc_np)
        R_meas_torch = torch.from_numpy(self.R_meas_np)

        result = system.control.design_lqg(
            A_torch,
            B_torch,
            C_torch,
            Q_torch,
            R_torch,
            Q_proc_torch,
            R_meas_torch,
            system_type="continuous",
        )

        # Verify result types
        self.assertIsInstance(result["controller_gain"], torch.Tensor)
        self.assertIsInstance(result["estimator_gain"], torch.Tensor)

        # Verify both components are stable
        ctrl_eigs_np = result["closed_loop_eigenvalues"].numpy()
        obs_eigs_np = result["observer_eigenvalues"].numpy()
        self.assertLess(np.max(np.real(ctrl_eigs_np)), 0)
        self.assertLess(np.max(np.real(obs_eigs_np)), 0)

    @unittest.skipUnless(torch_available, "PyTorch not installed")
    def test_pytorch_backend_analysis(self):
        """Test system analysis with PyTorch backend."""
        import torch

        system = MockContinuousSystem(backend="torch")

        A_torch = torch.from_numpy(self.A_np)
        B_torch = torch.from_numpy(self.B_np)
        C_torch = torch.from_numpy(self.C_np)

        # Stability
        stability = system.analysis.stability(A_torch, system_type="continuous")
        self.assertTrue(stability["is_stable"])
        # Note: stability returns numpy arrays even with torch input (internal conversion)

        # Controllability
        ctrl = system.analysis.controllability(A_torch, B_torch)
        self.assertTrue(ctrl["is_controllable"])

        # Observability
        obs = system.analysis.observability(A_torch, C_torch)
        self.assertTrue(obs["is_observable"])

    @unittest.skipUnless(jax_available, "JAX not installed")
    def test_jax_backend_lqr(self):
        """Test LQR with JAX backend."""
        import jax.numpy as jnp
        from jax import config

        # Enable float64 for JAX to match NumPy/PyTorch precision
        config.update("jax_enable_x64", True)

        system = MockContinuousSystem(backend="jax")

        # Convert to JAX arrays
        A_jax = jnp.array(self.A_np)
        B_jax = jnp.array(self.B_np)
        Q_jax = jnp.array(self.Q_np)
        R_jax = jnp.array(self.R_np)

        result = system.control.design_lqr(A_jax, B_jax, Q_jax, R_jax, system_type="continuous")

        # Verify result types (should be JAX arrays)
        # JAX arrays have __array__ protocol
        self.assertTrue(hasattr(result["gain"], "__array__"))
        self.assertTrue(hasattr(result["cost_to_go"], "__array__"))
        self.assertTrue(hasattr(result["closed_loop_eigenvalues"], "__array__"))

        # Verify stability
        self.assertGreater(result["stability_margin"], 0)
        eigenvalues_np = np.array(result["closed_loop_eigenvalues"])
        self.assertLess(np.max(np.real(eigenvalues_np)), 0)

    @unittest.skipUnless(jax_available, "JAX not installed")
    def test_jax_backend_kalman(self):
        """Test Kalman filter with JAX backend."""
        import jax.numpy as jnp
        from jax import config

        # Enable float64 for JAX to match NumPy/PyTorch precision
        config.update("jax_enable_x64", True)

        system = MockContinuousSystem(backend="jax")

        # Convert to JAX arrays
        A_jax = jnp.array(self.A_np)
        C_jax = jnp.array(self.C_np)
        Q_proc_jax = jnp.array(self.Q_proc_np)
        R_meas_jax = jnp.array(self.R_meas_np)

        result = system.control.design_kalman(
            A_jax, C_jax, Q_proc_jax, R_meas_jax, system_type="continuous",
        )

        # Verify result types
        self.assertTrue(hasattr(result["gain"], "__array__"))
        self.assertTrue(hasattr(result["error_covariance"], "__array__"))
        self.assertTrue(hasattr(result["observer_eigenvalues"], "__array__"))

        # Verify observer stability
        eigenvalues_np = np.array(result["observer_eigenvalues"])
        self.assertLess(np.max(np.real(eigenvalues_np)), 0)

    @unittest.skipUnless(jax_available, "JAX not installed")
    def test_jax_backend_lqg(self):
        """Test LQG with JAX backend."""
        import jax.numpy as jnp
        from jax import config

        # Enable float64 for JAX to match NumPy/PyTorch precision
        config.update("jax_enable_x64", True)

        system = MockContinuousSystem(backend="jax")

        # Convert to JAX arrays
        A_jax = jnp.array(self.A_np)
        B_jax = jnp.array(self.B_np)
        C_jax = jnp.array(self.C_np)
        Q_jax = jnp.array(self.Q_np)
        R_jax = jnp.array(self.R_np)
        Q_proc_jax = jnp.array(self.Q_proc_np)
        R_meas_jax = jnp.array(self.R_meas_np)

        result = system.control.design_lqg(
            A_jax, B_jax, C_jax, Q_jax, R_jax, Q_proc_jax, R_meas_jax, system_type="continuous",
        )

        # Verify result types
        self.assertTrue(hasattr(result["controller_gain"], "__array__"))
        self.assertTrue(hasattr(result["estimator_gain"], "__array__"))

        # Verify both components are stable
        ctrl_eigs_np = np.array(result["closed_loop_eigenvalues"])
        obs_eigs_np = np.array(result["observer_eigenvalues"])
        self.assertLess(np.max(np.real(ctrl_eigs_np)), 0)
        self.assertLess(np.max(np.real(obs_eigs_np)), 0)

    @unittest.skipUnless(jax_available, "JAX not installed")
    def test_jax_backend_analysis(self):
        """Test system analysis with JAX backend."""
        import jax.numpy as jnp
        from jax import config

        # Enable float64 for JAX to match NumPy/PyTorch precision
        config.update("jax_enable_x64", True)

        system = MockContinuousSystem(backend="jax")

        A_jax = jnp.array(self.A_np)
        B_jax = jnp.array(self.B_np)
        C_jax = jnp.array(self.C_np)

        # Stability
        stability = system.analysis.stability(A_jax, system_type="continuous")
        self.assertTrue(stability["is_stable"])

        # Controllability
        ctrl = system.analysis.controllability(A_jax, B_jax)
        self.assertTrue(ctrl["is_controllable"])

        # Observability
        obs = system.analysis.observability(A_jax, C_jax)
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
                system = MockContinuousSystem(backend=backend)

                # Convert inputs to appropriate backend
                if backend == "numpy":
                    A, B = self.A_np, self.B_np
                    Q, R = self.Q_np, self.R_np
                elif backend == "torch":
                    import torch

                    A = torch.from_numpy(self.A_np)
                    B = torch.from_numpy(self.B_np)
                    Q = torch.from_numpy(self.Q_np)
                    R = torch.from_numpy(self.R_np)
                elif backend == "jax":
                    import jax.numpy as jnp

                    A = jnp.array(self.A_np)
                    B = jnp.array(self.B_np)
                    Q = jnp.array(self.Q_np)
                    R = jnp.array(self.R_np)

                # Run LQR
                result = system.control.design_lqr(A, B, Q, R, system_type="continuous")

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
# Edge Cases and Error Handling
# ============================================================================


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""

    def test_control_with_existing_control_synthesis(self):
        """Test control property when _control_synthesis already exists."""
        system = MockContinuousSystem()

        # Manually set _control_synthesis
        existing_control = ControlSynthesis(backend="numpy")
        system._control_synthesis = existing_control

        # Property should return existing instance
        control = system.control
        self.assertIs(control, existing_control)

    def test_analysis_with_existing_system_analysis(self):
        """Test analysis property when _system_analysis already exists."""
        system = MockContinuousSystem()

        # Manually set _system_analysis
        existing_analysis = SystemAnalysis(backend="numpy")
        system._system_analysis = existing_analysis

        # Property should return existing instance
        analysis = system.analysis
        self.assertIs(analysis, existing_analysis)

    def test_control_property_thread_safety(self):
        """Test that control property handles concurrent access gracefully."""
        import threading

        system = MockContinuousSystem()
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

        system = MockContinuousSystem()
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


# ============================================================================
# Test Suite Configuration
# ============================================================================


def suite():
    """Create test suite for control framework integration."""
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
