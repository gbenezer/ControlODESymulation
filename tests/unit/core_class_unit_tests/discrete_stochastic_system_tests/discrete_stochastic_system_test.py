# Location: tests/systems/test_discrete_stochastic_system.py

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
Comprehensive Unit Tests for DiscreteStochasticSystem
======================================================

Test suite covering:
1. System initialization and validation
2. Deterministic and stochastic evaluation
3. Noise type detection and classification
4. Linearization with diffusion
5. Monte Carlo simulation
6. Backend compatibility (NumPy, PyTorch, JAX)
7. Batched operations
8. Edge cases and error handling
9. Performance and optimization
10. Integration with existing framework

Authors
-------
Gil Benezer

License
-------
AGPL-3.0
"""

import unittest
from typing import Tuple

import numpy as np
import sympy as sp

from src.systems.base.core.discrete_stochastic_system import DiscreteStochasticSystem
from src.systems.base.utils.stochastic.sde_validator import ValidationError
from src.systems.base.utils.stochastic.noise_analysis import NoiseType, SDEType


# ============================================================================
# Test System Definitions
# ============================================================================


class DiscreteOU(DiscreteStochasticSystem):
    """Discrete-time Ornstein-Uhlenbeck process (AR(1) with additive noise)."""

    def define_system(self, alpha=1.0, sigma=0.5, dt=0.1):
        x = sp.symbols("x", real=True)
        u = sp.symbols("u", real=True)
        alpha_sym = sp.symbols("alpha", positive=True)
        sigma_sym = sp.symbols("sigma", positive=True)
        dt_sym = sp.symbols("dt", positive=True)

        # Deterministic: x[k+1] = (1 - α*dt)*x[k] + u[k]
        self.state_vars = [x]
        self.control_vars = [u]
        self._f_sym = sp.Matrix([(1 - alpha_sym * dt_sym) * x + u])
        self.parameters = {alpha_sym: alpha, sigma_sym: sigma, dt_sym: dt}
        self._dt = dt
        self.order = 1

        # Stochastic: additive noise
        self.diffusion_expr = sp.Matrix([[sigma_sym]])
        self.sde_type = "ito"


class GeometricRandomWalk(DiscreteStochasticSystem):
    """Discrete-time geometric random walk (multiplicative noise)."""

    def define_system(self, mu=0.1, sigma=0.2, dt=1.0):
        x = sp.symbols("x", positive=True)
        u = sp.symbols("u", real=True)
        mu_sym = sp.symbols("mu", real=True)
        sigma_sym = sp.symbols("sigma", positive=True)

        # Deterministic: x[k+1] = (1 + μ)*x[k] + u[k]
        self.state_vars = [x]
        self.control_vars = [u]
        self._f_sym = sp.Matrix([(1 + mu_sym) * x + u])
        self.parameters = {mu_sym: mu, sigma_sym: sigma}
        self._dt = dt
        self.order = 1

        # Stochastic: multiplicative noise
        self.diffusion_expr = sp.Matrix([[sigma_sym * x]])
        self.sde_type = "ito"


class MultiDimensionalStochastic(DiscreteStochasticSystem):
    """2D system with coupled noise."""

    def define_system(self, a=0.9, b=0.1, sigma1=0.3, sigma2=0.2, dt=0.1):
        x1, x2 = sp.symbols("x1 x2", real=True)
        u = sp.symbols("u", real=True)
        a_sym, b_sym = sp.symbols("a b", real=True)
        sigma1_sym, sigma2_sym = sp.symbols("sigma1 sigma2", positive=True)

        # Deterministic: linear coupling
        self.state_vars = [x1, x2]
        self.control_vars = [u]
        self._f_sym = sp.Matrix([a_sym * x1 + b_sym * x2, -b_sym * x1 + a_sym * x2 + u])
        self.parameters = {a_sym: a, b_sym: b, sigma1_sym: sigma1, sigma2_sym: sigma2}
        self._dt = dt
        self.order = 1

        # Stochastic: diagonal noise
        self.diffusion_expr = sp.Matrix([[sigma1_sym, 0], [0, sigma2_sym]])
        self.sde_type = "ito"


class AutonomousStochastic(DiscreteStochasticSystem):
    """Autonomous system (no control input)."""

    def define_system(self, alpha=0.95, sigma=0.1, dt=0.1):
        x = sp.symbols("x", real=True)
        alpha_sym = sp.symbols("alpha", positive=True)
        sigma_sym = sp.symbols("sigma", positive=True)

        # Autonomous: no control
        self.state_vars = [x]
        self.control_vars = []  # No control!
        self._f_sym = sp.Matrix([alpha_sym * x])
        self.parameters = {alpha_sym: alpha, sigma_sym: sigma}
        self._dt = dt
        self.order = 1

        # Additive noise
        self.diffusion_expr = sp.Matrix([[sigma_sym]])
        self.sde_type = "ito"


class InvalidDiffusionSystem(DiscreteStochasticSystem):
    """System that doesn't set diffusion_expr (should fail)."""

    def define_system(self, dt=0.1):
        x = sp.symbols("x")
        self.state_vars = [x]
        self.control_vars = []
        self._f_sym = sp.Matrix([0.9 * x])
        self.parameters = {}
        self._dt = dt
        self.order = 1
        # Missing: self.diffusion_expr!


class WrongDimensionDiffusion(DiscreteStochasticSystem):
    """System with diffusion dimension mismatch."""

    def define_system(self, dt=0.1):
        x1, x2 = sp.symbols("x1 x2")
        sigma = sp.symbols("sigma", positive=True)

        self.state_vars = [x1, x2]  # nx=2
        self.control_vars = []
        self._f_sym = sp.Matrix([0.9 * x1, 0.9 * x2])
        self.parameters = {sigma: 0.3}
        self._dt = dt
        self.order = 1

        # Wrong: diffusion has 1 row but should have 2
        self.diffusion_expr = sp.Matrix([[sigma]])  # Should be (2, nw)!
        self.sde_type = "ito"


class MultiplicativeNoise2D(DiscreteStochasticSystem):
    """
    2D system with state-dependent noise (2 noise sources).

    This has TRUE multiplicative noise because:
    - nw = 2 (not scalar)
    - g depends on state
    - Not additive (not constant)
    """

    def define_system(self, a=0.9, sigma1=0.2, sigma2=0.15, dt=0.1):
        x1, x2 = sp.symbols("x1 x2", real=True)
        u = sp.symbols("u", real=True)
        a_sym = sp.symbols("a", real=True)
        sigma1_sym, sigma2_sym = sp.symbols("sigma1 sigma2", positive=True)

        # Deterministic: simple linear
        self.state_vars = [x1, x2]
        self.control_vars = [u]
        self._f_sym = sp.Matrix([a_sym * x1 + u, a_sym * x2])
        self.parameters = {a_sym: a, sigma1_sym: sigma1, sigma2_sym: sigma2}
        self._dt = dt
        self.order = 1

        # MULTIPLICATIVE: noise depends on state, nw=2
        self.diffusion_expr = sp.Matrix(
            [
                [sigma1_sym * x1, 0],  # First noise scales with x1
                [0, sigma2_sym * x2],  # Second noise scales with x2
            ]
        )
        self.sde_type = "ito"


class FullyMultiplicativeNoise(DiscreteStochasticSystem):
    """
    System where ALL noise sources are state-dependent.

    g(x) = [σ1*x1  σ2*x1]
           [σ3*x2  σ4*x2]

    Both noise sources affect both states, all scaled by state.
    """

    def define_system(self, a=0.95, sigma1=0.2, sigma2=0.15, sigma3=0.1, sigma4=0.12, dt=0.1):
        x1, x2 = sp.symbols("x1 x2", real=True)
        a_sym = sp.symbols("a", real=True)
        s1, s2, s3, s4 = sp.symbols("sigma1 sigma2 sigma3 sigma4", positive=True)

        self.state_vars = [x1, x2]
        self.control_vars = []  # Autonomous for simplicity
        self._f_sym = sp.Matrix([a_sym * x1, a_sym * x2])
        self.parameters = {a_sym: a, s1: sigma1, s2: sigma2, s3: sigma3, s4: sigma4}
        self._dt = dt
        self.order = 1

        # Fully coupled multiplicative noise
        self.diffusion_expr = sp.Matrix(
            [
                [s1 * x1, s2 * x1],  # Both noise sources scale with x1
                [s3 * x2, s4 * x2],  # Both noise sources scale with x2
            ]
        )
        self.sde_type = "ito"


class StateAndControlDependentNoise(DiscreteStochasticSystem):
    """
    Noise depends on both state AND control.

    g(x, u) = [σ1*x1*u]
              [σ2*x2  ]

    First noise source depends on both x1 and u.
    Second depends only on x2.
    """

    def define_system(self, a=0.9, sigma1=0.2, sigma2=0.15, dt=0.1):
        x1, x2 = sp.symbols("x1 x2", real=True)
        u = sp.symbols("u", real=True)
        a_sym = sp.symbols("a", real=True)
        sigma1_sym, sigma2_sym = sp.symbols("sigma1 sigma2", positive=True)

        self.state_vars = [x1, x2]
        self.control_vars = [u]
        self._f_sym = sp.Matrix([a_sym * x1 + u, a_sym * x2])
        self.parameters = {a_sym: a, sigma1_sym: sigma1, sigma2_sym: sigma2}
        self._dt = dt
        self.order = 1

        # Mixed: first depends on x1*u, second on x2
        self.diffusion_expr = sp.Matrix(
            [
                [sigma1_sym * x1 * u],  # Depends on state AND control
                [sigma2_sym * x2],  # Depends only on state
            ]
        )
        self.sde_type = "ito"


class ThreeNoiseSourcesMultiplicative(DiscreteStochasticSystem):
    """
    2D system with 3 independent noise sources, all multiplicative.

    More noise sources than states (nw > nx).
    """

    def define_system(self, a=0.9, s1=0.2, s2=0.15, s3=0.1, dt=0.1):
        x1, x2 = sp.symbols("x1 x2", real=True)
        a_sym = sp.symbols("a", real=True)
        sigma1, sigma2, sigma3 = sp.symbols("sigma1 sigma2 sigma3", positive=True)

        self.state_vars = [x1, x2]
        self.control_vars = []
        self._f_sym = sp.Matrix([a_sym * x1, a_sym * x2])
        self.parameters = {a_sym: a, sigma1: s1, sigma2: s2, sigma3: s3}
        self._dt = dt
        self.order = 1

        # 3 noise sources (nw=3 > nx=2)
        self.diffusion_expr = sp.Matrix(
            [[sigma1 * x1, sigma2 * x1, 0], [0, sigma3 * x2, sigma3 * x2]]
        )
        self.sde_type = "ito"


# ============================================================================
# Test Suite
# ============================================================================


class TestDiscreteStochasticSystemInitialization(unittest.TestCase):
    """Test system initialization and validation."""

    def test_basic_initialization(self):
        """Test basic system creation."""
        system = DiscreteOU(alpha=2.0, sigma=0.3, dt=0.1)

        # Check dimensions
        self.assertEqual(system.nx, 1)
        self.assertEqual(system.nu, 1)
        self.assertEqual(system.nw, 1)
        self.assertEqual(system.dt, 0.1)

        # Check stochastic flag
        self.assertTrue(system.is_stochastic)

        # Check components exist
        self.assertIsNotNone(system.diffusion_handler)
        self.assertIsNotNone(system.noise_characteristics)

    def test_multidimensional_initialization(self):
        """Test 2D system with diagonal noise."""
        system = MultiDimensionalStochastic(dt=0.05)

        self.assertEqual(system.nx, 2)
        self.assertEqual(system.nu, 1)
        self.assertEqual(system.nw, 2)
        self.assertEqual(system.dt, 0.05)

    def test_autonomous_initialization(self):
        """Test autonomous system (nu=0)."""
        system = AutonomousStochastic(dt=0.1)

        self.assertEqual(system.nx, 1)
        self.assertEqual(system.nu, 0)  # Autonomous!
        self.assertEqual(system.nw, 1)

    def test_missing_diffusion_expr(self):
        """Test that missing diffusion_expr raises error."""
        with self.assertRaises(ValueError) as cm:
            system = InvalidDiffusionSystem(dt=0.1)

        self.assertIn("must set self.diffusion_expr", str(cm.exception))

    def test_wrong_dimension_diffusion(self):
        """Test that dimension mismatch in diffusion raises error."""
        with self.assertRaises(ValidationError) as cm:
            system = WrongDimensionDiffusion(dt=0.1)

        # Should catch dimension mismatch during validation
        self.assertIn("dimension", str(cm.exception).lower())

    def test_sde_type_normalization(self):
        """Test that sde_type string is normalized to enum."""
        system = DiscreteOU(dt=0.1)

        # Should be converted to enum
        self.assertIsInstance(system.sde_type, SDEType)
        self.assertEqual(system.sde_type, SDEType.ITO)


class TestNoiseCharacterization(unittest.TestCase):
    """Test automatic noise type detection."""

    def test_additive_noise_detection(self):
        """Test additive noise is correctly detected."""
        system = DiscreteOU(sigma=0.3, dt=0.1)

        # Noise characteristics
        self.assertTrue(system.is_additive_noise())
        self.assertFalse(system.is_multiplicative_noise())
        self.assertEqual(system.get_noise_type(), NoiseType.ADDITIVE)

        # Dependencies
        self.assertFalse(system.depends_on_state())
        self.assertFalse(system.depends_on_control())
        self.assertFalse(system.depends_on_time())

    def test_multiplicative_noise_detection_scalar_override(self):
        system = GeometricRandomWalk(mu=0.1, sigma=0.2, dt=1.0)

        # It IS multiplicative in nature
        self.assertTrue(system.depends_on_state())
        # But classified as SCALAR because nw=1
        self.assertTrue(system.is_scalar_noise())

    def test_multiplicative_noise_detection(self):
        """Test multiplicative noise is correctly detected."""
        system = GeometricRandomWalk(mu=0.1, sigma=0.2, dt=1.0)

        # Noise characteristics
        self.assertFalse(system.is_additive_noise())
        self.assertTrue(system.is_multiplicative_noise())
        # While the noise is multiplicative, scalar is higher up in hierarchy
        self.assertEqual(system.get_noise_type(), NoiseType.SCALAR)

        # Dependencies
        self.assertTrue(system.depends_on_state())
        self.assertFalse(system.depends_on_control())

    def test_diagonal_noise_detection(self):
        """Test diagonal noise structure."""
        system = MultiDimensionalStochastic(dt=0.1)

        # Should detect diagonal structure
        self.assertTrue(system.is_diagonal_noise())

    def test_scalar_noise_detection(self):
        """Test scalar noise (nw=1)."""
        system = DiscreteOU(dt=0.1)

        self.assertTrue(system.is_scalar_noise())
        self.assertEqual(system.nw, 1)


class TestDeterministicEvaluation(unittest.TestCase):
    """Test deterministic part evaluation."""

    def setUp(self):
        """Set up test systems."""
        self.system = DiscreteOU(alpha=2.0, sigma=0.3, dt=0.1)

    def test_call_single_state(self):
        """Test __call__ with single state."""
        x_k = np.array([1.0])
        u_k = np.array([0.0])

        f = self.system(x_k, u_k)

        # Expected: (1 - 2.0*0.1)*1.0 + 0.0 = 0.8
        self.assertAlmostEqual(f[0], 0.8, places=10)

    def test_call_batched(self):
        """Test __call__ with batched inputs."""
        x_batch = np.array([[1.0], [2.0], [3.0]])
        u_batch = np.array([[0.0], [0.5], [1.0]])

        f = self.system(x_batch, u_batch)

        # Check shape
        self.assertEqual(f.shape, (3, 1))

        # Check values
        # f[0] = 0.8*1.0 + 0.0 = 0.8
        # f[1] = 0.8*2.0 + 0.5 = 2.1
        # f[2] = 0.8*3.0 + 1.0 = 3.4
        np.testing.assert_allclose(f[:, 0], [0.8, 2.1, 3.4], rtol=1e-10)

    def test_call_autonomous(self):
        """Test autonomous system (nu=0)."""
        system = AutonomousStochastic(alpha=0.95, dt=0.1)
        x_k = np.array([1.0])

        # Should work without u
        f = system(x_k)

        self.assertAlmostEqual(f[0], 0.95, places=10)


class TestDiffusionEvaluation(unittest.TestCase):
    """Test stochastic part (diffusion) evaluation."""

    def setUp(self):
        """Set up test systems."""
        self.additive_system = DiscreteOU(sigma=0.3, dt=0.1)
        self.multiplicative_system = GeometricRandomWalk(sigma=0.2, dt=1.0)

    def test_diffusion_additive(self):
        """Test diffusion evaluation for additive noise."""
        x_k = np.array([1.0])
        u_k = np.array([0.0])

        g = self.additive_system.diffusion(x_k, u_k)

        # Should be constant
        self.assertEqual(g.shape, (1, 1))
        self.assertAlmostEqual(g[0, 0], 0.3, places=10)

    def test_diffusion_multiplicative(self):
        """Test diffusion evaluation for multiplicative noise."""
        x_k = np.array([2.0])
        u_k = np.array([0.0])

        g = self.multiplicative_system.diffusion(x_k, u_k)

        # Should be state-dependent: σ*x = 0.2*2.0 = 0.4
        self.assertEqual(g.shape, (1, 1))
        self.assertAlmostEqual(g[0, 0], 0.4, places=10)

    def test_diffusion_batched_additive(self):
        """Test batched diffusion evaluation for additive noise."""
        x_batch = np.array([[1.0], [2.0], [3.0]])
        u_batch = np.array([[0.0], [0.0], [0.0]])

        g = self.additive_system.diffusion(x_batch, u_batch)

        # Additive: should return (nx, nw) constant
        self.assertEqual(g.shape, (1, 1))
        self.assertAlmostEqual(g[0, 0], 0.3, places=10)

    def test_diffusion_batched_multiplicative(self):
        """Test batched diffusion for multiplicative noise."""
        x_batch = np.array([[1.0], [2.0], [3.0]])
        u_batch = np.array([[0.0], [0.0], [0.0]])

        g = self.multiplicative_system.diffusion(x_batch, u_batch)

        # Multiplicative: should return (batch, nx, nw)
        self.assertEqual(g.shape, (3, 1, 1))

        # Check values: σ*x for each sample
        np.testing.assert_allclose(
            g[:, 0, 0], np.array([0.2, 0.4, 0.6]), rtol=1e-10  # 0.2*[1, 2, 3]
        )


class TestStochasticStep(unittest.TestCase):
    """Test full stochastic step evaluation."""

    def setUp(self):
        """Set up test systems."""
        self.system = DiscreteOU(alpha=2.0, sigma=0.3, dt=0.1)

    def test_step_stochastic_with_noise(self):
        """Test stochastic step with provided noise."""
        x_k = np.array([1.0])
        u_k = np.array([0.0])
        w_k = np.array([1.0])  # Unit noise

        x_next = self.system.step_stochastic(x_k, u_k, w_k)

        # Expected: f + g*w = 0.8 + 0.3*1.0 = 1.1
        self.assertAlmostEqual(x_next[0], 1.1, places=10)

    def test_step_stochastic_zero_noise(self):
        """Test that zero noise gives deterministic result."""
        x_k = np.array([1.0])
        u_k = np.array([0.0])
        w_k = np.zeros(1)

        x_next = self.system.step_stochastic(x_k, u_k, w_k)
        f = self.system(x_k, u_k)

        # Should match deterministic part
        np.testing.assert_allclose(x_next, f, rtol=1e-10)

    def test_step_stochastic_auto_noise_generation(self):
        """Test automatic noise generation."""
        np.random.seed(42)
        x_k = np.array([1.0])
        u_k = np.array([0.0])

        x_next = self.system.step_stochastic(x_k, u_k)  # w=None

        # Should generate noise internally
        self.assertEqual(x_next.shape, (1,))
        # Result should be stochastic (not deterministic)
        f = self.system(x_k, u_k)
        self.assertNotEqual(x_next[0], f[0])


class TestLinearization(unittest.TestCase):
    """Test linearization with diffusion."""

    def setUp(self):
        """Set up test system."""
        self.system = DiscreteOU(alpha=2.0, sigma=0.3, dt=0.1)

    def test_linearization_returns_three_matrices(self):
        """Test that stochastic linearization returns (Ad, Bd, Gd)."""
        x_eq = np.zeros(1)
        u_eq = np.zeros(1)

        result = self.system.linearize(x_eq, u_eq)

        # Should return 3-tuple
        self.assertEqual(len(result), 3)

        Ad, Bd, Gd = result

        # Check shapes
        self.assertEqual(Ad.shape, (1, 1))
        self.assertEqual(Bd.shape, (1, 1))
        self.assertEqual(Gd.shape, (1, 1))

    def test_linearization_values_additive(self):
        """Test linearization values for additive noise."""
        x_eq = np.zeros(1)
        u_eq = np.zeros(1)

        Ad, Bd, Gd = self.system.linearize(x_eq, u_eq)

        # Ad = ∂f/∂x = (1 - α*dt) = 1 - 2.0*0.1 = 0.8
        self.assertAlmostEqual(Ad[0, 0], 0.8, places=10)

        # Bd = ∂f/∂u = 1.0
        self.assertAlmostEqual(Bd[0, 0], 1.0, places=10)

        # Gd = g(x_eq, u_eq) = σ = 0.3
        self.assertAlmostEqual(Gd[0, 0], 0.3, places=10)


class TestStochasticSimulation(unittest.TestCase):
    """Test simulate_stochastic method."""

    def setUp(self):
        """Set up test system."""
        self.system = DiscreteOU(alpha=2.0, sigma=0.3, dt=0.1)

    def test_simulate_single_path(self):
        """Test single path simulation."""
        x0 = np.array([1.0])
        n_steps = 100

        result = self.system.simulate_stochastic(x0=x0, u_sequence=None, n_steps=n_steps, seed=42)

        # Check result structure
        self.assertIn("states", result)
        self.assertIn("time_steps", result)
        self.assertIn("dt", result)
        self.assertIn("metadata", result)

        # Check shapes
        self.assertEqual(result["states"].shape, (n_steps + 1, 1))
        self.assertEqual(len(result["time_steps"]), n_steps + 1)

        # Check metadata
        self.assertEqual(result["metadata"]["n_paths"], 1)
        self.assertEqual(result["metadata"]["seed"], 42)

        # Check initial condition
        np.testing.assert_allclose(result["states"][0, :], x0, rtol=1e-10)

    def test_simulate_monte_carlo(self):
        """Test Monte Carlo simulation with multiple paths."""
        x0 = np.array([1.0])
        n_steps = 100
        n_paths = 50

        result = self.system.simulate_stochastic(
            x0=x0, u_sequence=None, n_steps=n_steps, n_paths=n_paths, seed=42
        )

        # Check shape: (n_paths, n_steps+1, nx)
        self.assertEqual(result["states"].shape, (n_paths, n_steps + 1, 1))

        # Check metadata
        self.assertEqual(result["metadata"]["n_paths"], n_paths)

        # Check all paths start at x0
        for path in range(n_paths):
            np.testing.assert_allclose(result["states"][path, 0, :], x0, rtol=1e-10)

    def test_simulate_reproducibility(self):
        """Test that same seed gives same results."""
        x0 = np.array([1.0])
        n_steps = 100

        result1 = self.system.simulate_stochastic(x0, None, n_steps, seed=42)
        result2 = self.system.simulate_stochastic(x0, None, n_steps, seed=42)

        # Should be identical
        np.testing.assert_allclose(result1["states"], result2["states"], rtol=1e-10)


class TestConstantNoiseOptimization(unittest.TestCase):
    """Test constant noise precomputation for additive systems."""

    def test_get_constant_noise_additive(self):
        """Test getting constant noise matrix."""
        system = DiscreteOU(sigma=0.3, dt=0.1)

        G = system.get_constant_noise(backend="numpy")

        self.assertEqual(G.shape, (1, 1))
        self.assertAlmostEqual(G[0, 0], 0.3, places=10)

    def test_get_constant_noise_multiplicative_fails(self):
        """Test that multiplicative noise can't use constant optimization."""
        system = GeometricRandomWalk(sigma=0.2, dt=1.0)

        with self.assertRaises(ValueError):
            G = system.get_constant_noise()


class TestBackendCompatibility(unittest.TestCase):
    """Test multi-backend support."""

    def setUp(self):
        """Set up test system."""
        self.system = DiscreteOU(alpha=2.0, sigma=0.3, dt=0.1)
        self.x = np.array([1.0])
        self.u = np.array([0.0])

    def test_numpy_backend(self):
        """Test NumPy backend."""
        f = self.system(self.x, self.u, backend="numpy")
        g = self.system.diffusion(self.x, self.u, backend="numpy")

        self.assertIsInstance(f, np.ndarray)
        self.assertIsInstance(g, np.ndarray)


class TestPrintingAndInfo(unittest.TestCase):
    """Test printing and information methods."""

    def setUp(self):
        """Set up test system."""
        self.system = DiscreteOU(alpha=2.0, sigma=0.3, dt=0.1)

    def test_print_equations(self):
        """Test that print_equations doesn't crash."""
        # Should not raise
        self.system.print_equations(simplify=True)
        self.system.print_equations(simplify=False)

    def test_print_stochastic_info(self):
        """Test that print_stochastic_info doesn't crash."""
        # Should not raise
        self.system.print_stochastic_info()

    def test_get_info_structure(self):
        """Test get_info returns correct structure."""
        info = self.system.get_info()

        # Check required keys
        self.assertIn("system_type", info)
        self.assertIn("is_discrete", info)
        self.assertIn("is_stochastic", info)
        self.assertIn("dimensions", info)
        self.assertIn("noise", info)

        # Check values
        self.assertEqual(info["system_type"], "DiscreteStochasticSystem")
        self.assertTrue(info["is_discrete"])
        self.assertTrue(info["is_stochastic"])


class TestStatisticalProperties(unittest.TestCase):
    """Test statistical properties of simulations."""

    def test_variance_accumulation(self):
        """Test that variance accumulates correctly for additive noise."""
        system = DiscreteOU(alpha=0.0, sigma=0.3, dt=0.1)  # Pure random walk
        x0 = np.zeros(1)
        n_steps = 100
        n_paths = 500

        result = system.simulate_stochastic(x0, None, n_steps, n_paths, seed=42)

        # Compute variance at each time step
        variance_traj = result["states"].var(axis=0)[:, 0]

        # For random walk: Var[x[k]] = k * σ²
        expected_variance = np.arange(n_steps + 1) * 0.3**2

        # Check at specific points (allow 30% error due to finite samples)
        for k in [10, 50, 100]:
            self.assertAlmostEqual(variance_traj[k] / expected_variance[k], 1.0, delta=0.3)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""

    def test_zero_noise(self):
        """Test system with zero noise."""
        system = DiscreteOU(alpha=2.0, sigma=0.0, dt=0.1)  # σ=0!

        x_k = np.array([1.0])
        u_k = np.array([0.0])

        # Diffusion should be zero
        g = system.diffusion(x_k, u_k)
        np.testing.assert_allclose(g, np.zeros((1, 1)), rtol=1e-10)

    def test_very_small_dt(self):
        """Test system with very small time step."""
        system = DiscreteOU(alpha=2.0, sigma=0.3, dt=1e-6)

        self.assertEqual(system.dt, 1e-6)

        # Should still work
        x_k = np.array([1.0])
        u_k = np.array([0.0])
        f = system(x_k, u_k)
        g = system.diffusion(x_k, u_k)

        # f ≈ (1 - α*dt)*x = (1 - 2e-6)*1 ≈ 1.0
        self.assertAlmostEqual(f[0], 1.0 - 2e-6, places=8)


# ============================================================================
# Test Suite for Multiplicative Noise Detection
# ============================================================================


class TestMultiplicativeNoiseDetection(unittest.TestCase):
    """
    Test that multiplicative noise is correctly detected.

    Key principle: For MULTIPLICATIVE classification (not SCALAR):
    - Need nw > 1 (multiple noise sources), OR
    - System must be multiplicative with nw=1 but not classifiable as SCALAR

    The NoiseCharacterizer hierarchy is:
    1. SCALAR (if nw == 1)
    2. ADDITIVE (if constant)
    3. DIAGONAL (if diagonal matrix)
    4. MULTIPLICATIVE (if state-dependent)
    5. GENERAL (fallback)
    """

    def test_multiplicative_2d_detected_diagonal_override(self):
        """Test 2D multiplicative noise with 2 INDEPENDENT noise sources."""
        system = MultiplicativeNoise2D(sigma1=0.2, sigma2=0.15, dt=0.1)

        # Should be classified as MULTIPLICATIVE
        self.assertTrue(system.is_multiplicative_noise())
        self.assertFalse(system.is_additive_noise())
        self.assertFalse(system.is_scalar_noise())  # nw=2, not scalar
        self.assertEqual(system.get_noise_type(), NoiseType.DIAGONAL)

        # Dependencies
        self.assertTrue(system.depends_on_state())
        self.assertFalse(system.depends_on_control())

        # Dimensions
        self.assertEqual(system.nw, 2)
        self.assertEqual(system.nx, 2)

    def test_fully_multiplicative_detected(self):
        """Test fully coupled multiplicative noise."""
        system = FullyMultiplicativeNoise(sigma1=0.2, sigma2=0.15, sigma3=0.1, sigma4=0.12, dt=0.1)

        # Should be MULTIPLICATIVE (nw=2, state-dependent)
        self.assertTrue(system.is_multiplicative_noise())
        self.assertEqual(system.get_noise_type(), NoiseType.MULTIPLICATIVE)

        # All noise depends on state
        self.assertTrue(system.depends_on_state())
        self.assertFalse(system.depends_on_control())

        # Not diagonal (off-diagonal terms exist)
        self.assertFalse(system.is_diagonal_noise())

    def test_state_and_control_dependent(self):
        """Test noise depending on both state and control."""
        system = StateAndControlDependentNoise(sigma1=0.2, sigma2=0.15, dt=0.1)

        # Should be MULTIPLICATIVE (depends on x and u)
        # Note: might be SCALAR if nw=1, let's check
        self.assertEqual(system.nw, 1)

        # With nw=1, classified as SCALAR even though multiplicative
        self.assertTrue(system.is_scalar_noise())

        # But dependencies should still be detected
        self.assertTrue(system.depends_on_state())
        self.assertTrue(system.depends_on_control())  # ✓ This is key!

    def test_three_noise_sources_multiplicative(self):
        """Test system with more noise sources than states."""
        system = ThreeNoiseSourcesMultiplicative(s1=0.2, s2=0.15, s3=0.1, dt=0.1)

        # nw=3 > nx=2
        self.assertEqual(system.nw, 3)
        self.assertEqual(system.nx, 2)

        # Should be MULTIPLICATIVE (nw > 1, state-dependent)
        self.assertTrue(system.is_multiplicative_noise())
        self.assertFalse(system.is_scalar_noise())  # nw=3, not scalar
        self.assertEqual(system.get_noise_type(), NoiseType.MULTIPLICATIVE)

        self.assertTrue(system.depends_on_state())

    def test_multiplicative_diffusion_evaluation(self):
        """Test that multiplicative diffusion evaluates correctly."""
        system = MultiplicativeNoise2D(sigma1=0.2, sigma2=0.15, dt=0.1)

        # Evaluate at specific state
        x = np.array([2.0, 3.0])
        u = np.array([0.0])

        g = system.diffusion(x, u)

        # Expected: [[σ1*x1, 0], [0, σ2*x2]]
        # = [[0.2*2.0, 0], [0, 0.15*3.0]]
        # = [[0.4, 0], [0, 0.45]]
        expected_g = np.array([[0.4, 0.0], [0.0, 0.45]])

        np.testing.assert_allclose(g, expected_g, rtol=1e-10)

    # TODO: fix test or class
    def test_multiplicative_batched_evaluation(self):
        """Test batched multiplicative diffusion."""
        system = MultiplicativeNoise2D(sigma1=0.2, sigma2=0.15, dt=0.1)

        x_batch = np.array([[1.0, 2.0], [3.0, 4.0]])
        u_batch = np.array([[0.0], [0.0]])

        print(x_batch)
        print(u_batch)
        print(system)
        system.print_equations(simplify=True)
        print(system.diffusion_expr)
        print(system.diffusion_handler)
        print(system.diffusion)
        system.compile_diffusion(backends=["numpy"])

        g = system.diffusion(x_batch, u_batch)
        print(g.shape)
        print(g[0])
        print(g[1])

        # Should return (batch, nx, nw) = (2, 2, 2)
        self.assertEqual(g.shape, (2, 2, 2))

        # First sample: x=[1, 2]
        # g[0] = [[0.2*1, 0], [0, 0.15*2]] = [[0.2, 0], [0, 0.3]]
        np.testing.assert_allclose(g[0], np.array([[0.2, 0.0], [0.0, 0.3]]), rtol=1e-10)

        # Second sample: x=[3, 4]
        # g[1] = [[0.2*3, 0], [0, 0.15*4]] = [[0.6, 0], [0, 0.6]]
        np.testing.assert_allclose(g[1], np.array([[0.6, 0.0], [0.0, 0.6]]), rtol=1e-10)

    def test_multiplicative_stochastic_step(self):
        """Test stochastic step with multiplicative noise."""
        system = MultiplicativeNoise2D(a=0.9, sigma1=0.2, sigma2=0.15, dt=0.1)

        x_k = np.array([1.0, 2.0])
        u_k = np.array([0.0])
        w_k = np.array([1.0, 1.0])  # Unit noise in both directions

        x_next = system.step_stochastic(x_k, u_k, w_k)

        # Deterministic: f = [0.9*1.0, 0.9*2.0] = [0.9, 1.8]
        # Stochastic: g*w = [[0.2*1, 0], [0, 0.15*2]] @ [1, 1]
        #                 = [[0.2*1], [0.15*2*1]]
        #                 = [[0.2], [0.3]]
        # Total: x[k+1] = [0.9 + 0.2, 1.8 + 0.3] = [1.1, 2.1]

        expected = np.array([1.1, 2.1])
        np.testing.assert_allclose(x_next, expected, rtol=1e-10)

    def test_multiplicative_no_optimization(self):
        """Test that multiplicative systems can't precompute noise."""
        system = MultiplicativeNoise2D(dt=0.1)

        # Cannot optimize for constant noise
        self.assertFalse(system.can_optimize_for_additive())

        # Should raise error when trying to get constant noise
        with self.assertRaises(ValueError):
            G = system.get_constant_noise()

        # Optimization opportunities
        opts = system.get_optimization_opportunities()
        self.assertFalse(opts["precompute_diffusion"])


class TestScalarVsMultiplicativeClassification(unittest.TestCase):
    """
    Test the distinction between SCALAR and MULTIPLICATIVE classification.

    Key insight from NoiseCharacterizer:
    - SCALAR is prioritized when nw == 1
    - MULTIPLICATIVE requires nw > 1 OR special conditions
    """

    def test_scalar_multiplicative_system(self):
        """
        Test system with nw=1 and state-dependent noise.

        This is multiplicative in NATURE but classified as SCALAR.
        """

        class ScalarMultiplicative(DiscreteStochasticSystem):
            def define_system(self, sigma=0.2, dt=0.1):
                x = sp.symbols("x", real=True)
                sigma_sym = sp.symbols("sigma", positive=True)

                self.state_vars = [x]
                self.control_vars = []
                self._f_sym = sp.Matrix([0.9 * x])
                self.parameters = {sigma_sym: sigma}
                self._dt = dt
                self.order = 1

                # Multiplicative in nature: g = σ*x
                # But nw=1, so classified as SCALAR
                self.diffusion_expr = sp.Matrix([[sigma_sym * x]])
                self.sde_type = "ito"

        system = ScalarMultiplicative(sigma=0.2, dt=0.1)

        # Classified as SCALAR (nw=1 takes priority)
        self.assertTrue(system.is_scalar_noise())
        self.assertEqual(system.get_noise_type(), NoiseType.SCALAR)

        # But still state-dependent!
        self.assertTrue(system.depends_on_state())

        # Is also classified as multiplicative (however SCALAR takes precedence)
        self.assertTrue(system.is_multiplicative_noise())

    def test_true_multiplicative_requires_multiple_noise(self):
        """
        Test that TRUE multiplicative classification requires nw > 1.
        """

        # nw=1: Classified as SCALAR (even if state-dependent)
        class SingleNoise(DiscreteStochasticSystem):
            def define_system(self, dt=0.1):
                x = sp.symbols("x")
                sigma = sp.symbols("sigma", positive=True)

                self.state_vars = [x]
                self.control_vars = []
                self._f_sym = sp.Matrix([0.9 * x])
                self.parameters = {sigma: 0.2}
                self._dt = dt
                self.order = 1

                self.diffusion_expr = sp.Matrix([[sigma * x]])  # nw=1
                self.sde_type = "ito"

        # nw=2: Classified as MULTIPLICATIVE
        class DoubleNoise(DiscreteStochasticSystem):
            def define_system(self, dt=0.1):
                x = sp.symbols("x")
                s1, s2 = sp.symbols("sigma1 sigma2", positive=True)

                self.state_vars = [x]
                self.control_vars = []
                self._f_sym = sp.Matrix([0.9 * x])
                self.parameters = {s1: 0.2, s2: 0.15}
                self._dt = dt
                self.order = 1

                self.diffusion_expr = sp.Matrix([[s1 * x, s2 * x]])  # nw=2
                self.sde_type = "ito"

        single = SingleNoise(dt=0.1)
        double = DoubleNoise(dt=0.1)

        # Single noise: SCALAR and also MULTIPLICATIVE
        self.assertTrue(single.is_scalar_noise())
        self.assertTrue(single.is_multiplicative_noise())

        # Double noise: MULTIPLICATIVE
        self.assertFalse(double.is_scalar_noise())
        self.assertTrue(double.is_multiplicative_noise())
        self.assertEqual(double.get_noise_type(), NoiseType.MULTIPLICATIVE)


class TestMultiplicativeNoiseNumericalBehavior(unittest.TestCase):
    """Test numerical behavior of multiplicative noise systems."""

    def test_noise_grows_with_state(self):
        """Test that multiplicative noise intensity grows with state magnitude."""
        system = MultiplicativeNoise2D(sigma1=0.2, sigma2=0.15, dt=0.1)

        # Small state
        x_small = np.array([0.1, 0.1])
        g_small = system.diffusion(x_small, np.array([0.0]))

        # Large state
        x_large = np.array([10.0, 10.0])
        g_large = system.diffusion(x_large, np.array([0.0]))

        # Noise should be larger for larger state
        noise_small = np.linalg.norm(g_small)
        noise_large = np.linalg.norm(g_large)

        self.assertGreater(noise_large, noise_small)

        # Should scale linearly with state (for linear multiplicative)
        ratio = noise_large / noise_small
        expected_ratio = 10.0 / 0.1  # 100x
        self.assertAlmostEqual(ratio, expected_ratio, places=1)

    def test_variance_grows_faster_than_additive(self):
        """
        Test that multiplicative noise causes faster variance growth.

        For multiplicative: Var[x[k]] grows exponentially
        For additive: Var[x[k]] grows linearly
        """
        # Additive noise system
        additive = DiscreteOU(alpha=0.0, sigma=0.3, dt=0.1)

        # Multiplicative noise system (scalar classification, but state-dependent)
        class SimpleMultiplicative(DiscreteStochasticSystem):
            def define_system(self, sigma=0.3, dt=0.1):
                x = sp.symbols("x")
                sigma_sym = sp.symbols("sigma", positive=True)

                self.state_vars = [x]
                self.control_vars = []
                # No drift (pure noise evolution)
                self._f_sym = sp.Matrix([x])  # x[k+1] = x[k] + noise
                self.parameters = {sigma_sym: sigma}
                self._dt = dt
                self.order = 1

                # Multiplicative: g = σ*x
                self.diffusion_expr = sp.Matrix([[sigma_sym * x]])
                self.sde_type = "ito"

        multiplicative = SimpleMultiplicative(sigma=0.3, dt=0.1)

        x0 = np.array([1.0])
        n_steps = 50
        n_paths = 200

        # Simulate both
        result_add = additive.simulate_stochastic(x0, None, n_steps, n_paths, seed=42)
        result_mult = multiplicative.simulate_stochastic(x0, None, n_steps, n_paths, seed=43)

        # Compute variance trajectories
        var_add = result_add["states"].var(axis=0)[:, 0]
        var_mult = result_mult["states"].var(axis=0)[:, 0]

        # At late times, multiplicative should have higher variance
        # (multiplicative noise amplifies with state)
        self.assertGreater(var_mult[-1], var_add[-1])

    def test_linearization_captures_state_dependence(self):
        """Test that linearization at different points gives different Gd."""
        system = MultiplicativeNoise2D(sigma1=0.2, sigma2=0.15, dt=0.1)

        # Linearize at two different equilibria
        x_eq1 = np.array([1.0, 1.0])
        x_eq2 = np.array([5.0, 5.0])
        u_eq = np.array([0.0])

        Ad1, Bd1, Gd1 = system.linearize(x_eq1, u_eq)
        Ad2, Bd2, Gd2 = system.linearize(x_eq2, u_eq)

        # Ad and Bd should be same (linear deterministic part)
        np.testing.assert_allclose(Ad1, Ad2, rtol=1e-10)
        np.testing.assert_allclose(Bd1, Bd2, rtol=1e-10)

        # Gd should be DIFFERENT (multiplicative noise)
        # Gd1 = [[0.2*1, 0], [0, 0.15*1]]
        # Gd2 = [[0.2*5, 0], [0, 0.15*5]]
        expected_Gd1 = np.array([[0.2, 0.0], [0.0, 0.15]])
        expected_Gd2 = np.array([[1.0, 0.0], [0.0, 0.75]])

        np.testing.assert_allclose(Gd1, expected_Gd1, rtol=1e-10)
        np.testing.assert_allclose(Gd2, expected_Gd2, rtol=1e-10)

        # Gd2 should be 5x larger
        np.testing.assert_allclose(Gd2, 5.0 * Gd1, rtol=1e-10)


class TestMultiplicativeNoiseSimulation(unittest.TestCase):
    """Test simulation with multiplicative noise."""

    def test_simulation_with_multiplicative_noise(self):
        """Test that simulation works with multiplicative noise."""
        system = MultiplicativeNoise2D(a=0.95, sigma1=0.1, sigma2=0.08, dt=0.1)

        x0 = np.array([1.0, 1.0])
        n_steps = 100

        result = system.simulate_stochastic(x0=x0, u_sequence=None, n_steps=n_steps, seed=42)

        # Should complete successfully
        self.assertTrue(result["success"])
        self.assertEqual(result["states"].shape, (n_steps + 1, 2))

    def test_monte_carlo_with_multiplicative(self):
        """Test Monte Carlo with multiplicative noise."""
        system = MultiplicativeNoise2D(sigma1=0.1, sigma2=0.08, dt=0.1)

        x0 = np.array([1.0, 1.0])
        n_steps = 50
        n_paths = 100

        result = system.simulate_stochastic(
            x0=x0, u_sequence=None, n_steps=n_steps, n_paths=n_paths, seed=42
        )

        self.assertEqual(result["states"].shape, (n_paths, n_steps + 1, 2))
        self.assertEqual(result["metadata"]["n_paths"], n_paths)


class TestNoiseTypeHierarchy(unittest.TestCase):
    """
    Test understanding of noise type classification hierarchy.

    NoiseCharacterizer priority (from code inspection):
    1. ADDITIVE (constant, no dependencies) - takes absolute priority
    2. SCALAR (nw == 1)
    3. DIAGONAL (diagonal matrix structure)
    4. MULTIPLICATIVE (state-dependent, not diagonal)
    5. GENERAL (fallback)

    NOTE: hierarchy was updated to allow noise types to have
    multiple attributes true at the same time; for optimization
    the main categorization follows the hierarchy.
    """

    # TODO: fix test to actually cover the whole hierarchy
    def test_classification_hierarchy(self):
        """Test classification follows expected hierarchy."""

        # Case 1: nw=1, constant → ADDITIVE and SCALAR; ADDITIVE takes precedence
        class Case1(DiscreteStochasticSystem):
            def define_system(self, dt=0.1):
                x = sp.symbols("x")
                self.state_vars = [x]
                self.control_vars = []
                self._f_sym = sp.Matrix([0.9 * x])
                self.parameters = {sp.symbols("sigma"): 0.3}
                self._dt = dt
                self.order = 1
                self.diffusion_expr = sp.Matrix([[0.3]])  # Constant, nw=1
                self.sde_type = "ito"

        c1 = Case1(dt=0.1)
        self.assertEqual(c1.get_noise_type(), NoiseType.ADDITIVE)  # ADDITIVE and SCALAR

        # Case 2: nw=2, constant → ADDITIVE
        class Case2(DiscreteStochasticSystem):
            def define_system(self, dt=0.1):
                x = sp.symbols("x")
                s1, s2 = sp.symbols("sigma1 sigma2", positive=True)
                self.state_vars = [x]
                self.control_vars = []
                self._f_sym = sp.Matrix([0.9 * x])
                self.parameters = {s1: 0.3, s2: 0.2}
                self._dt = dt
                self.order = 1
                self.diffusion_expr = sp.Matrix([[0.3, 0.2]])  # Constant, nw=2
                self.sde_type = "ito"

        c2 = Case2(dt=0.1)
        self.assertEqual(c2.get_noise_type(), NoiseType.ADDITIVE)

        # Case 3: nw=2, diagonal → DIAGONAL (takes priority over MULTIPLICATIVE)
        # Also ADDITIVE, so it also is classified as ADDITIVE
        c3 = MultiDimensionalStochastic(dt=0.1)
        self.assertEqual(c3.get_noise_type(), NoiseType.ADDITIVE)

        # Case 4: nw=2, state-dependent, non-diagonal → MULTIPLICATIVE
        c4 = FullyMultiplicativeNoise(dt=0.1)
        self.assertEqual(c4.get_noise_type(), NoiseType.MULTIPLICATIVE)


# ============================================================================
# Test Runner
# ============================================================================


def run_tests(verbosity=2):
    """
    Run all tests with specified verbosity.

    Parameters
    ----------
    verbosity : int
        Verbosity level (0=quiet, 1=normal, 2=verbose)

    Returns
    -------
    unittest.TestResult
        Test results
    """
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    test_classes = [
        TestDiscreteStochasticSystemInitialization,
        TestNoiseCharacterization,
        TestDeterministicEvaluation,
        TestDiffusionEvaluation,
        TestStochasticStep,
        TestLinearization,
        TestStochasticSimulation,
        TestConstantNoiseOptimization,
        TestBackendCompatibility,
        TestPrintingAndInfo,
        TestStatisticalProperties,
        TestEdgeCases,
        TestMultiplicativeNoiseDetection,
        TestScalarVsMultiplicativeClassification,
        TestMultiplicativeNoiseNumericalBehavior,
        TestMultiplicativeNoiseSimulation,
        TestNoiseTypeHierarchy,
    ]

    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    runner = unittest.TextTestRunner(verbosity=verbosity)
    return runner.run(suite)


if __name__ == "__main__":
    # Run with verbose output
    result = run_tests(verbosity=2)

    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")

    if result.wasSuccessful():
        print("\n✓ ALL TESTS PASSED!")
    else:
        print("\n✗ SOME TESTS FAILED")

    print("=" * 70)
