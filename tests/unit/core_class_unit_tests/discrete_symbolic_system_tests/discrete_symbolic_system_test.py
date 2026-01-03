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
Unit Tests for DiscreteSymbolicSystem
======================================

Comprehensive test suite for the refactored DiscreteSymbolicSystem class.

Test Coverage
-------------
- Multiple inheritance architecture
- DiscreteSystemBase interface implementation
- State transition (step) evaluation
- Multi-step simulation with various control formats
- Linearization (discrete-time Ad, Bd matrices)
- Equilibrium verification (fixed point condition)
- Sampling period (dt) management
- Performance tracking
- Time-major convention consistency
- Autonomous discrete systems (nu=0)
- Backward compatibility

Test Organization
-----------------
- TestMultipleInheritance: Architecture validation
- TestDiscreteInterface: DiscreteSystemBase methods
- TestSamplingPeriod: dt property and validation
- TestStateTransition: step() evaluation
- TestSimulation: Multi-step simulation
- TestControlSequenceHandling: Flexible control inputs
- TestLinearization: Discrete Jacobian computation
- TestEquilibriumVerification: Fixed point checking
- TestPerformanceTracking: Statistics
- TestAutonomousSystems: nu=0 systems
- TestOutputFunctions: y[k] evaluation
- TestBackwardCompatibility: Aliases
- TestEdgeCases: Boundary conditions

Usage
-----
pytest test_discrete_symbolic_system.py -v
pytest test_discrete_symbolic_system.py::TestSimulation -v
pytest test_discrete_symbolic_system.py --cov
"""


import numpy as np
import pytest
import sympy as sp

from src.systems.base.core.discrete_symbolic_system import (
    DiscreteDynamicalSystem,
    DiscreteSymbolicSystem,
)
from src.systems.base.core.discrete_system_base import DiscreteSystemBase
from src.systems.base.core.symbolic_system_base import SymbolicSystemBase

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
# Test System Implementations
# ============================================================================


class DiscreteLinear(DiscreteSymbolicSystem):
    """Simple linear discrete system: x[k+1] = a*x[k] + b*u[k]"""

    def define_system(self, a=0.9, b=0.1, dt=0.01):
        x = sp.symbols("x", real=True)
        u = sp.symbols("u", real=True)
        a_sym, b_sym = sp.symbols("a b", real=True)

        self.state_vars = [x]
        self.control_vars = [u]
        self._f_sym = sp.Matrix([a_sym * x + b_sym * u])
        self.parameters = {a_sym: a, b_sym: b}
        self._dt = dt
        self.order = 1


class DiscreteAutonomous(DiscreteSymbolicSystem):
    """Autonomous discrete: x[k+1] = a*x[k]"""

    def define_system(self, a=0.95, dt=0.1):
        x = sp.symbols("x", real=True)
        a_sym = sp.symbols("a", real=True)

        self.state_vars = [x]
        self.control_vars = []  # No control
        self._f_sym = sp.Matrix([a_sym * x])
        self.parameters = {a_sym: a}
        self._dt = dt
        self.order = 1


class DiscreteTwoState(DiscreteSymbolicSystem):
    """Two-state discrete system"""

    def define_system(self, a11=0.9, a12=0.1, a21=-0.1, a22=0.8, b1=0.1, b2=0.05, dt=0.01):
        x1, x2 = sp.symbols("x1 x2", real=True)
        u = sp.symbols("u", real=True)

        a11_sym, a12_sym, a21_sym, a22_sym = sp.symbols("a11 a12 a21 a22", real=True)
        b1_sym, b2_sym = sp.symbols("b1 b2", real=True)

        self.state_vars = [x1, x2]
        self.control_vars = [u]
        self._f_sym = sp.Matrix(
            [a11_sym * x1 + a12_sym * x2 + b1_sym * u, a21_sym * x1 + a22_sym * x2 + b2_sym * u],
        )
        self.parameters = {
            a11_sym: a11,
            a12_sym: a12,
            a21_sym: a21,
            a22_sym: a22,
            b1_sym: b1,
            b2_sym: b2,
        }
        self._dt = dt
        self.order = 1


class DiscreteLogisticMap(DiscreteSymbolicSystem):
    """Nonlinear: x[k+1] = r*x[k]*(1-x[k])"""

    def define_system(self, r=3.5, dt=1.0):
        x = sp.symbols("x", real=True, positive=True)
        r_sym = sp.symbols("r", positive=True)

        self.state_vars = [x]
        self.control_vars = []
        self._f_sym = sp.Matrix([r_sym * x * (1 - x)])
        self.parameters = {r_sym: r}
        self._dt = dt
        self.order = 1


class DiscreteWithOutput(DiscreteSymbolicSystem):
    """System with custom output"""

    def define_system(self, dt=0.01):
        x, v = sp.symbols("x v", real=True)
        u = sp.symbols("u", real=True)

        self.state_vars = [x, v]
        self.control_vars = [u]
        self._f_sym = sp.Matrix([0.9 * x + 0.1 * v, v + 0.05 * u])
        self._h_sym = sp.Matrix([x, x**2 + v**2])
        self.parameters = {}
        self._dt = dt
        self.order = 1


# ============================================================================
# Test: Multiple Inheritance
# ============================================================================


class TestMultipleInheritance:
    """Validate multiple inheritance architecture."""

    def test_inherits_from_both_bases(self):
        """System inherits from both bases."""
        system = DiscreteLinear()

        assert isinstance(system, SymbolicSystemBase)
        assert isinstance(system, DiscreteSystemBase)
        assert isinstance(system, DiscreteSymbolicSystem)

    def test_mro_order(self):
        """Method Resolution Order is correct."""
        mro = DiscreteSymbolicSystem.__mro__

        assert DiscreteSymbolicSystem in mro
        assert SymbolicSystemBase in mro
        assert DiscreteSystemBase in mro

        symbolic_idx = mro.index(SymbolicSystemBase)
        discrete_idx = mro.index(DiscreteSystemBase)
        assert symbolic_idx < discrete_idx

    def test_has_symbolic_methods(self):
        """Has SymbolicSystemBase methods."""
        system = DiscreteLinear()

        assert hasattr(system, "substitute_parameters")
        assert hasattr(system, "compile")
        assert hasattr(system, "add_equilibrium")
        assert hasattr(system, "get_config_dict")

    def test_has_discrete_methods(self):
        """Has DiscreteSystemBase methods."""
        system = DiscreteLinear()

        assert hasattr(system, "step")
        assert hasattr(system, "simulate")
        assert hasattr(system, "linearize")
        assert hasattr(system, "rollout")

    def test_has_discrete_properties(self):
        """Has discrete-time properties."""
        system = DiscreteLinear()

        assert system.is_discrete
        assert not system.is_continuous
        assert hasattr(system, "dt")


# ============================================================================
# Test: DiscreteSystemBase Interface
# ============================================================================


class TestDiscreteInterface:
    """Test DiscreteSystemBase interface implementation."""

    def test_step_evaluates_next_state(self):
        """step() computes x[k+1] = f(x[k], u[k])."""
        system = DiscreteLinear(a=0.9, b=0.1)

        x_k = np.array([1.0])
        u_k = np.array([0.5])

        x_next = system.step(x_k, u_k)

        assert x_next.shape == (1,)
        expected = 0.9 * 1.0 + 0.1 * 0.5
        np.testing.assert_allclose(x_next, [expected])

    def test_step_with_time_index(self):
        """step() accepts k parameter."""
        system = DiscreteLinear()

        x = np.array([1.0])
        u = np.array([0.0])

        # Should not raise even though k is ignored
        x_next = system.step(x, u, k=5)
        assert x_next.shape == (1,)

    def test_simulate_returns_discrete_result(self):
        """simulate() returns DiscreteSimulationResult."""
        system = DiscreteLinear()

        result = system.simulate(x0=np.array([1.0]), u_sequence=None, n_steps=10)

        # Check structure
        assert "states" in result
        assert "time_steps" in result
        assert "dt" in result
        assert "metadata" in result

    def test_linearize_returns_discrete_matrices(self):
        """linearize() returns (Ad, Bd)."""
        system = DiscreteLinear(a=0.8, b=0.2)

        Ad, Bd = system.linearize(np.zeros(1), np.zeros(1))

        assert Ad.shape == (1, 1)
        assert Bd.shape == (1, 1)
        np.testing.assert_allclose(Ad, [[0.8]])
        np.testing.assert_allclose(Bd, [[0.2]])

    def test_rollout_with_policy(self):
        """rollout() works with state feedback."""
        system = DiscreteLinear(a=0.9, b=0.1)

        def policy(x, k):
            return -0.5 * x

        result = system.rollout(x0=np.array([1.0]), policy=policy, n_steps=10)

        # Time-major: (T, nx)
        assert result["states"].shape == (11, 1)  # (n_steps+1, nx)
        assert result["controls"] is not None


# ============================================================================
# Test: Sampling Period (dt)
# ============================================================================


class TestSamplingPeriod:
    """Test dt property and validation."""

    def test_dt_property_returns_value(self):
        """dt property returns sampling period."""
        system = DiscreteLinear(dt=0.05)

        assert system.dt == 0.05

    def test_dt_required_in_define_system(self):
        """define_system() must set _dt."""

        class MissingDt(DiscreteSymbolicSystem):
            def define_system(self):
                x = sp.symbols("x")
                self.state_vars = [x]
                self.control_vars = []
                self._f_sym = sp.Matrix([0.9 * x])
                self.parameters = {}
                self.order = 1
                # Missing: self._dt = ...

        with pytest.raises(ValueError, match="must define self._dt"):
            MissingDt()

    def test_various_dt_values(self):
        """System works with various dt values."""
        dt_values = [0.001, 0.01, 0.1, 1.0]

        for dt in dt_values:
            system = DiscreteLinear(dt=dt)
            assert system.dt == dt

    def test_sampling_frequency_property(self):
        """Can compute sampling frequency."""
        system = DiscreteLinear(dt=0.01)

        freq = system.sampling_frequency

        assert freq == 1.0 / 0.01  # 100 Hz


# ============================================================================
# Test: State Transition
# ============================================================================


class TestStateTransition:
    """Test step() method."""

    def test_step_single_state(self):
        """Single step evaluation."""
        system = DiscreteLinear(a=0.95, b=0.05)

        x = np.array([1.0])
        u = np.array([0.5])

        x_next = system.step(x, u)

        expected = 0.95 * 1.0 + 0.05 * 0.5
        np.testing.assert_allclose(x_next, [expected])

    def test_step_autonomous(self):
        """Step without control."""
        system = DiscreteAutonomous(a=0.9)

        x = np.array([1.0])
        x_next = system.step(x)  # u=None

        assert x_next.shape == (1,)
        np.testing.assert_allclose(x_next, [0.9])

    def test_step_multi_state(self):
        """Step with multiple states."""
        system = DiscreteTwoState()

        x = np.array([1.0, 0.0])
        u = np.array([0.0])

        x_next = system.step(x, u)

        assert x_next.shape == (2,)

    def test_step_nonlinear(self):
        """Step with nonlinear dynamics."""
        system = DiscreteLogisticMap(r=3.0)

        x = np.array([0.5])
        x_next = system.step(x)

        # x[k+1] = 3*0.5*(1-0.5) = 3*0.5*0.5 = 0.75
        np.testing.assert_allclose(x_next, [0.75])

    def test_repeated_stepping(self):
        """Can step repeatedly."""
        system = DiscreteLinear(a=0.9, b=0.0)

        x = np.array([1.0])
        u = np.array([0.0])  # Provide zero control, not None
        for k in range(10):
            x = system.step(x, u, k=k)

        assert x[0] < 0.5


# ============================================================================
# Test: Simulation
# ============================================================================


class TestSimulation:
    """Test multi-step simulation."""

    def test_simulate_with_none_control(self):
        """Simulate with u=None (zero control)."""
        system = DiscreteLinear(a=0.9, b=0.1)

        result = system.simulate(x0=np.array([1.0]), u_sequence=None, n_steps=10)

        assert result["states"].shape == (11, 1)  # Time-major: (n_steps+1, nx)
        assert result["time_steps"].shape == (11,)
        assert result["dt"] == 0.01

    def test_simulate_with_constant_control(self):
        """Simulate with constant control."""
        system = DiscreteLinear(a=0.9, b=0.1)

        result = system.simulate(x0=np.array([1.0]), u_sequence=np.array([0.5]), n_steps=10)

        assert result["controls"] is not None
        assert result["controls"].shape == (1, 10)  # (nu, n_steps)

    def test_simulate_with_precomputed_sequence_time_major(self):
        """Simulate with pre-computed (n_steps, nu) sequence."""
        system = DiscreteLinear()

        u_seq = np.array([[0.1], [0.2], [0.3], [0.4], [0.5]])  # (5, 1)

        result = system.simulate(x0=np.array([1.0]), u_sequence=u_seq, n_steps=5)

        assert result["states"].shape == (6, 1)  # (n_steps+1, nx)

    def test_simulate_with_precomputed_sequence_state_major(self):
        """Simulate with pre-computed (nu, n_steps) sequence."""
        system = DiscreteLinear()

        u_seq = np.array([[0.1, 0.2, 0.3, 0.4, 0.5]])  # (1, 5)

        result = system.simulate(x0=np.array([1.0]), u_sequence=u_seq, n_steps=5)

        assert result["states"].shape == (6, 1)

    def test_simulate_with_time_indexed_function(self):
        """Simulate with u(k) function."""
        system = DiscreteLinear()

        def u_func(k):
            return np.array([0.1 * k])

        result = system.simulate(x0=np.array([1.0]), u_sequence=u_func, n_steps=10)

        assert result["success"]

    def test_simulate_with_state_feedback(self):
        """Simulate with u(k, x) state feedback."""
        system = DiscreteLinear()

        def policy(k, x):
            return -0.5 * x

        result = system.simulate(x0=np.array([1.0]), u_sequence=policy, n_steps=10)

        assert result["success"]

    def test_simulate_with_list_control(self):
        """Simulate with list of controls."""
        system = DiscreteLinear()

        u_list = [np.array([0.1]), np.array([0.2]), np.array([0.3])]

        result = system.simulate(x0=np.array([1.0]), u_sequence=u_list, n_steps=3)

        assert result["success"]

    def test_simulate_convergence(self):
        """Stable discrete system converges."""
        system = DiscreteLinear(a=0.8, b=0.0)  # Stable

        result = system.simulate(x0=np.array([1.0]), u_sequence=None, n_steps=50)

        # Should converge to zero
        assert abs(result["states"][-1, 0]) < 0.01

    def test_simulate_autonomous(self):
        """Simulate autonomous system."""
        system = DiscreteAutonomous(a=0.95)

        result = system.simulate(x0=np.array([1.0]), u_sequence=None, n_steps=20)

        assert result["success"]
        # x[k] = 0.95^k
        expected_final = 0.95**20
        np.testing.assert_allclose(result["states"][-1, 0], expected_final, rtol=1e-6)


# ============================================================================
# Test: Control Sequence Handling
# ============================================================================


class TestControlSequenceHandling:
    """Test _prepare_control_sequence()."""

    def test_none_autonomous(self):
        """None for autonomous system."""
        system = DiscreteAutonomous()

        u_func = system._prepare_control_sequence(None, 10)

        result = u_func(0, np.array([1.0]))
        assert result is None

    def test_none_noautonomous_zeros(self):
        """None for non-autonomous gives zeros."""
        system = DiscreteLinear()

        u_func = system._prepare_control_sequence(None, 10)

        result = u_func(np.array([1.0]), 0)  # (x, k)
        np.testing.assert_array_equal(result, np.zeros(1))

    def test_constant_array(self):
        """Constant control array."""
        system = DiscreteLinear()

        u_const = np.array([0.5])
        u_func = system._prepare_control_sequence(u_const, 10)

        # Use (x, k) order
        assert np.allclose(u_func(np.zeros(1), 0), u_const)
        assert np.allclose(u_func(np.zeros(1), 5), u_const)

    def test_time_indexed_callable(self):
        """Time-indexed u(k)."""
        system = DiscreteLinear()

        def u_indexed(k):
            return np.array([k * 0.1])

        u_func = system._prepare_control_sequence(u_indexed, 10)

        # Correct order: (x, k) not (k, x)
        assert np.allclose(u_func(np.zeros(1), 5), [0.5])

    def test_state_feedback_xk_order(self):  # ✅ Renamed
        """State feedback u(x, k)."""
        system = DiscreteLinear()

        def policy(x, k):  # ✅ Correct order
            return -0.5 * x

        u_func = system._prepare_control_sequence(policy, 10)

        result = u_func(np.array([2.0]), 0)  # (x, k)
        np.testing.assert_allclose(result, [-1.0])

    def test_invalid_control_type_raises(self):
        """Invalid control type raises TypeError."""
        system = DiscreteLinear()

        with pytest.raises(TypeError, match="Invalid control sequence type"):
            system._prepare_control_sequence("invalid", 10)

    def test_wrong_dimension_constant_raises(self):
        """Constant control with wrong dimension raises."""
        system = DiscreteLinear()  # nu=1

        u_wrong = np.array([0.1, 0.2])  # Wrong dimension

        with pytest.raises(ValueError, match="dimension mismatch"):
            system._prepare_control_sequence(u_wrong, 10)


# ============================================================================
# Test: Linearization
# ============================================================================


class TestLinearization:
    """Test discrete linearization."""

    def test_linearize_at_origin(self):
        """Linearize at origin."""
        system = DiscreteLinear(a=0.85, b=0.15)

        Ad, Bd = system.linearize(np.zeros(1), np.zeros(1))

        assert Ad.shape == (1, 1)
        assert Bd.shape == (1, 1)
        np.testing.assert_allclose(Ad, [[0.85]])
        np.testing.assert_allclose(Bd, [[0.15]])

    def test_linearize_multistate(self):
        """Linearize multi-state system."""
        system = DiscreteTwoState()

        Ad, Bd = system.linearize(np.zeros(2), np.zeros(1))

        assert Ad.shape == (2, 2)
        assert Bd.shape == (2, 1)

    def test_linearize_autonomous(self):
        """Linearize autonomous system."""
        system = DiscreteAutonomous(a=0.95)

        Ad, Bd = system.linearize(np.zeros(1), np.array([]))

        assert Ad.shape == (1, 1)
        assert Bd.shape == (1, 0)  # Empty
        np.testing.assert_allclose(Ad, [[0.95]])

    def test_linearize_nonlinear(self):
        """Linearize nonlinear system."""
        system = DiscreteLogisticMap(r=3.0)

        # Linearize at x=0.5
        x_eq = np.array([0.5])
        Ad, Bd = system.linearize(x_eq, np.array([]))

        assert Ad.shape == (1, 1)
        # Ad = d(r*x*(1-x))/dx at x=0.5 = r*(1-2x) = 3*(1-1) = 0
        np.testing.assert_allclose(Ad, [[0.0]], atol=1e-10)

    def test_linearized_dynamics_alias(self):
        """linearized_dynamics() is alias."""
        system = DiscreteLinear()

        A1, B1 = system.linearize(np.zeros(1), np.zeros(1))
        A2, B2 = system.linearized_dynamics(np.zeros(1), np.zeros(1))

        np.testing.assert_array_equal(A1, A2)
        np.testing.assert_array_equal(B1, B2)

    def test_linearized_dynamics_symbolic(self):
        """Symbolic linearization."""
        system = DiscreteLinear(a=0.9, b=0.1)

        Ad_sym, Bd_sym = system.linearized_dynamics_symbolic()

        assert isinstance(Ad_sym, sp.Matrix)
        assert float(Ad_sym[0, 0]) == 0.9
        assert float(Bd_sym[0, 0]) == 0.1

    def test_discrete_stability_check(self):
        """Can check discrete stability from linearization."""
        system = DiscreteLinear(a=0.8)  # Stable

        Ad, Bd = system.linearize(np.zeros(1), np.zeros(1))

        eigenvalues = np.linalg.eigvals(Ad)
        is_stable = np.all(np.abs(eigenvalues) < 1.0)

        assert is_stable


# ============================================================================
# Test: Equilibrium Verification
# ============================================================================


class TestEquilibriumVerification:
    """Test discrete equilibrium (fixed point) verification."""

    def test_verify_equilibrium_hook_implemented(self):
        """_verify_equilibrium_numpy is implemented."""
        system = DiscreteLinear()

        assert hasattr(system, "_verify_equilibrium_numpy")
        assert callable(system._verify_equilibrium_numpy)

    def test_verify_valid_fixed_point(self):
        """Valid fixed point returns True."""
        system = DiscreteLinear(a=0.9, b=0.1)

        # For x[k+1] = 0.9*x + 0.1*u, fixed point when x = 0.9*x + 0.1*u
        # => 0.1*x = 0.1*u => x = u
        x_eq = np.array([1.0])
        u_eq = np.array([1.0])

        is_valid = system._verify_equilibrium_numpy(x_eq, u_eq, tol=1e-6)

        assert is_valid

    def test_verify_invalid_fixed_point(self):
        """Invalid fixed point returns False."""
        system = DiscreteLinear(a=0.9, b=0.1)

        # NOT fixed point: x[k+1] = 0.9*2 + 0.1*0 = 1.8 ≠ 2
        x_eq = np.array([2.0])
        u_eq = np.array([0.0])

        is_valid = system._verify_equilibrium_numpy(x_eq, u_eq, tol=1e-6)

        assert not is_valid

    def test_add_equilibrium_with_verification(self):
        """Add equilibrium with verification."""
        system = DiscreteLinear(a=0.9, b=0.1)

        x_eq = np.array([1.0])
        u_eq = np.array([1.0])

        system.add_equilibrium("test", x_eq, u_eq, verify=True)

        assert "test" in system.list_equilibria()

    def test_add_invalid_equilibrium_warns(self):
        """Adding invalid equilibrium warns."""
        system = DiscreteLinear(a=0.9, b=0.1)

        x_eq = np.array([2.0])
        u_eq = np.array([0.0])  # Wrong!

        with pytest.warns(UserWarning, match="failed verification"):
            system.add_equilibrium("invalid", x_eq, u_eq, verify=True, tol=1e-6)

    def test_origin_is_fixed_point_with_zero_control(self):
        """Origin is fixed point for zero control."""
        system = DiscreteLinear(a=0.9, b=0.1)

        # x[k+1] = 0.9*0 + 0.1*0 = 0 ✓
        is_valid = system._verify_equilibrium_numpy(np.zeros(1), np.zeros(1), tol=1e-6)

        assert is_valid


# ============================================================================
# Test: Performance Tracking
# ============================================================================


class TestPerformanceTracking:
    """Test performance statistics."""

    def test_get_performance_stats_structure(self):
        """Performance stats have correct structure."""
        system = DiscreteLinear()

        stats = system.get_performance_stats()

        assert "forward_calls" in stats
        assert "forward_time" in stats
        assert "linearization_calls" in stats

    def test_stats_track_step_calls(self):
        """Stats track step() calls."""
        system = DiscreteLinear()
        system.reset_performance_stats()

        x = np.array([1.0])
        for _ in range(10):
            x = system.step(x, np.array([0.0]))

        stats = system.get_performance_stats()
        assert stats["forward_calls"] >= 10

    def test_reset_stats(self):
        """Can reset statistics."""
        system = DiscreteLinear()

        # Generate some calls
        for _ in range(5):
            system.step(np.array([1.0]), np.array([0.0]))

        system.reset_performance_stats()

        stats = system.get_performance_stats()
        assert stats["forward_calls"] == 0


# ============================================================================
# Test: Autonomous Systems
# ============================================================================


class TestAutonomousSystems:
    """Test discrete autonomous systems."""

    def test_autonomous_nu_zero(self):
        """Autonomous system has nu=0."""
        system = DiscreteAutonomous()

        assert system.nu == 0
        assert len(system.control_vars) == 0

    def test_autonomous_step_no_control(self):
        """Can step without control."""
        system = DiscreteAutonomous(a=0.9)

        x = np.array([1.0])
        x_next = system.step(x)

        np.testing.assert_allclose(x_next, [0.9])

    def test_autonomous_simulate(self):
        """Simulate autonomous system."""
        system = DiscreteAutonomous(a=0.95)

        result = system.simulate(x0=np.array([1.0]), u_sequence=None, n_steps=10)

        assert result["success"]
        # x[k] = a^k * x0
        expected = 0.95**10 * 1.0
        np.testing.assert_allclose(result["states"][-1, 0], expected, rtol=1e-6)

    def test_autonomous_linearize(self):
        """Linearize autonomous system."""
        system = DiscreteAutonomous(a=0.8)

        Ad, Bd = system.linearize(np.zeros(1), np.array([]))

        assert Ad.shape == (1, 1)
        assert Bd.shape == (1, 0)


# ============================================================================
# Test: Output Functions
# ============================================================================


class TestOutputFunctions:
    """Test output evaluation."""

    def test_h_identity_output(self):
        """Identity output."""
        system = DiscreteLinear()

        x = np.array([1.5])
        y = system.h(x)

        np.testing.assert_array_equal(y, x)

    def test_h_custom_output(self):
        """Custom output function."""
        system = DiscreteWithOutput()

        x = np.array([1.0, 2.0])
        y = system.h(x)

        # y = [x, x^2 + v^2] = [1, 1+4] = [1, 5]
        expected = np.array([1.0, 5.0])
        np.testing.assert_allclose(y, expected)

    def test_linearized_observation(self):
        """Observation linearization."""
        system = DiscreteWithOutput()

        x = np.array([1.0, 2.0])
        C = system.linearized_observation(x)

        assert C.shape == (2, 2)
        # C = [[1, 0], [2*x, 2*v]]
        expected = np.array([[1.0, 0.0], [2.0, 4.0]])
        np.testing.assert_allclose(C, expected)


# ============================================================================
# Test: Time-Major Convention Consistency
# ============================================================================


class TestTimeMajorConvention:
    """Test consistent time-major (T, nx) convention."""

    def test_simulate_returns_time_major(self):
        """simulate() returns (n_steps+1, nx) shape."""
        system = DiscreteTwoState()

        result = system.simulate(x0=np.array([1.0, 0.0]), u_sequence=None, n_steps=20)

        # Time-major: (T, nx)
        assert result["states"].shape == (21, 2)  # (n_steps+1, nx)

    def test_time_major_indexing(self):
        """Time-major indexing pattern."""
        system = DiscreteLinear()

        result = system.simulate(x0=np.array([1.0]), u_sequence=None, n_steps=10)

        # Access state at time k
        x_at_k5 = result["states"][5, :]  # (nx,)
        assert x_at_k5.shape == (1,)

        # Access state component over time
        x_component = result["states"][:, 0]  # (T,)
        assert x_component.shape == (11,)

    def test_controls_shape(self):
        """Control sequence has correct shape."""
        system = DiscreteLinear()

        result = system.simulate(x0=np.array([1.0]), u_sequence=np.array([0.5]), n_steps=10)

        # Controls: (nu, n_steps)
        assert result["controls"].shape == (1, 10)


# ============================================================================
# Test: Print Equations
# ============================================================================


class TestPrintEquations:
    """Test print_equations() output."""

    def test_print_shows_discrete_notation(self, capsys):
        """Uses x[k+1] notation."""
        system = DiscreteLinear()

        system.print_equations()

        captured = capsys.readouterr()
        assert "Discrete-Time" in captured.out
        assert "[k+1]" in captured.out

    def test_print_shows_dt(self, capsys):
        """Displays sampling period."""
        system = DiscreteLinear(dt=0.05)

        system.print_equations()

        captured = capsys.readouterr()
        assert "0.05" in captured.out


# ============================================================================
# Test: Backward Compatibility
# ============================================================================


class TestBackwardCompatibility:
    """Test migration alias."""

    def test_discrete_dynamical_system_alias(self):
        """DiscreteDynamicalSystem is alias."""
        assert DiscreteDynamicalSystem is DiscreteSymbolicSystem

    def test_alias_works(self):
        """Can use alias name."""

        class SystemWithAlias(DiscreteDynamicalSystem):
            def define_system(self, dt=0.1):
                x = sp.symbols("x")
                self.state_vars = [x]
                self.control_vars = []
                self._f_sym = sp.Matrix([0.9 * x])
                self.parameters = {}
                self._dt = dt
                self.order = 1

        system = SystemWithAlias()

        assert system._initialized
        assert isinstance(system, DiscreteSymbolicSystem)


# ============================================================================
# Test: Edge Cases
# ============================================================================


class TestEdgeCases:
    """Test boundary conditions."""

    def test_single_step_simulation(self):
        """Simulate for single step."""
        system = DiscreteLinear()

        result = system.simulate(x0=np.array([1.0]), u_sequence=None, n_steps=1)

        assert result["states"].shape == (2, 1)  # x[0] and x[1]

    def test_zero_steps_simulation(self):
        """Simulate for zero steps."""
        system = DiscreteLinear()

        result = system.simulate(x0=np.array([1.0]), u_sequence=None, n_steps=0)

        # Should just return initial state
        assert result["states"].shape == (1, 1)
        np.testing.assert_array_equal(result["states"][0, :], [1.0])

    def test_long_simulation(self):
        """Long simulation."""
        system = DiscreteLinear(a=0.99)

        result = system.simulate(x0=np.array([1.0]), u_sequence=None, n_steps=1000)

        assert result["states"].shape == (1001, 1)

    def test_unstable_system_diverges(self):
        """Unstable system diverges."""
        system = DiscreteLinear(a=1.1, b=0.0)  # Unstable

        result = system.simulate(x0=np.array([1.0]), u_sequence=None, n_steps=20)

        # Should grow
        assert result["states"][-1, 0] > result["states"][0, 0]


# ============================================================================
# Test: Comparison Between Continuous and Discrete
# ============================================================================


class TestContinuousVsDiscrete:
    """Compare conventions between continuous and discrete."""

    def test_both_use_time_major(self):
        """Both continuous and discrete use time-major convention."""
        from src.systems.base.core.continuous_symbolic_system import ContinuousSymbolicSystem

        # Continuous system
        class SimpleContinuous(ContinuousSymbolicSystem):
            def define_system(self):
                x = sp.symbols("x")
                self.state_vars = [x]
                self.control_vars = []
                self._f_sym = sp.Matrix([-x])
                self.parameters = {}
                self.order = 1

        cont = SimpleContinuous()

        # Discrete system
        disc = DiscreteLinear(dt=0.01)

        # Integrate continuous
        result_cont = cont.integrate(
            x0=np.array([1.0]),
            u=None,
            t_span=(0.0, 0.1),
            method="rk4",
            dt=0.01,
        )

        # Simulate discrete
        result_disc = disc.simulate(x0=np.array([1.0]), u_sequence=None, n_steps=10)

        # Both should be time-major: (T, nx)
        assert result_cont["x"].shape[0] == len(result_cont["t"])  # (T, nx)
        assert result_disc["states"].shape[0] == len(result_disc["time_steps"])  # (T, nx)


# ============================================================================
# Parametrized Tests
# ============================================================================


class TestParametrizedScenarios:
    """Parametrized tests."""

    @pytest.mark.parametrize("a_val", [0.5, 0.8, 0.95, 0.99])
    def test_various_stability_margins(self, a_val):
        """Test systems with different stability margins."""
        system = DiscreteLinear(a=a_val, b=0.0)

        Ad, Bd = system.linearize(np.zeros(1), np.zeros(1))

        eigenvalues = np.linalg.eigvals(Ad)
        is_stable = np.all(np.abs(eigenvalues) < 1.0)

        assert is_stable  # All test values are stable

    @pytest.mark.parametrize("dt_val", [0.001, 0.01, 0.1, 1.0])
    def test_various_sampling_periods(self, dt_val):
        """Test various dt values."""
        system = DiscreteLinear(dt=dt_val)

        assert system.dt == dt_val

    @pytest.mark.parametrize("n_steps", [1, 10, 100, 1000])
    def test_various_simulation_lengths(self, n_steps):
        """Test different simulation lengths."""
        system = DiscreteLinear()

        result = system.simulate(x0=np.array([1.0]), u_sequence=None, n_steps=n_steps)

        assert result["states"].shape == (n_steps + 1, 1)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def discrete_linear():
    """Linear discrete system fixture."""
    return DiscreteLinear(a=0.9, b=0.1, dt=0.01)


@pytest.fixture
def discrete_autonomous():
    """Autonomous discrete system fixture."""
    return DiscreteAutonomous(a=0.95, dt=0.1)


@pytest.fixture
def discrete_twostate():
    """Two-state discrete system fixture."""
    return DiscreteTwoState()


@pytest.fixture
def discrete_nonlinear():
    """Nonlinear discrete system fixture."""
    return DiscreteLogisticMap(r=3.5)


# ============================================================================
# Tests Using Fixtures
# ============================================================================


class TestUsingFixtures:
    """Tests using fixtures."""

    def test_linear_fixture(self, discrete_linear):
        """Linear fixture works."""
        assert discrete_linear.nx == 1
        assert discrete_linear.nu == 1
        assert discrete_linear.dt == 0.01

    def test_autonomous_fixture(self, discrete_autonomous):
        """Autonomous fixture works."""
        assert discrete_autonomous.nu == 0

    def test_step_with_fixture(self, discrete_linear):
        """Can step with fixture."""
        x_next = discrete_linear.step(np.array([1.0]), np.array([0.0]))
        assert x_next.shape == (1,)

    def test_simulate_with_fixture(self, discrete_linear):
        """Can simulate with fixture."""
        result = discrete_linear.simulate(x0=np.array([1.0]), u_sequence=None, n_steps=10)
        assert result["success"]


# ============================================================================
# End-to-End Workflows
# ============================================================================


class TestEndToEndWorkflows:
    """Test complete workflows."""

    def test_complete_workflow(self):
        """Complete discrete system workflow."""
        # 1. Create
        system = DiscreteTwoState(dt=0.01)

        # 2. Configure
        system.set_default_backend("numpy")

        # 3. Simulate
        result = system.simulate(x0=np.array([1.0, 0.0]), u_sequence=np.array([0.0]), n_steps=50)

        # 4. Linearize
        Ad, Bd = system.linearize(np.zeros(2), np.zeros(1))

        # 5. Check stability
        eigenvalues = np.linalg.eigvals(Ad)
        is_stable = np.all(np.abs(eigenvalues) < 1.0)

        assert result["success"]
        assert Ad.shape == (2, 2)

    def test_feedback_control_workflow(self):
        """Discrete feedback control."""
        system = DiscreteLinear(a=0.95, b=0.1)

        Ad, Bd = system.linearize(np.zeros(1), np.zeros(1))
        K = np.array([[0.5]])

        def policy(x, k):  # Correct order: (x, k)
            return -K @ x

        result = system.rollout(x0=np.array([1.0]), policy=policy, n_steps=50)

        assert abs(result["states"][-1, 0]) < 0.1

    def test_autonomous_convergence(self):
        """Autonomous system convergence."""
        system = DiscreteAutonomous(a=0.9)

        result = system.simulate(x0=np.array([1.0]), u_sequence=None, n_steps=100)

        # Should converge to zero
        assert abs(result["states"][-1, 0]) < 0.01


# ============================================================================
# Test: Error Handling
# ============================================================================


class TestErrorHandling:
    """Test error conditions."""

    def test_missing_dt_raises(self):
        """Missing dt in define_system raises."""

        class NoDt(DiscreteSymbolicSystem):
            def define_system(self):
                x = sp.symbols("x")
                self.state_vars = [x]
                self.control_vars = []
                self._f_sym = sp.Matrix([0.9 * x])
                self.parameters = {}
                # Missing: self._dt

        with pytest.raises(ValueError, match="must define self._dt"):
            NoDt()

    def test_invalid_control_sequence_type(self):
        """Invalid control sequence type raises."""
        system = DiscreteLinear()

        with pytest.raises(TypeError):
            system._prepare_control_sequence("invalid", 10)

    def test_dimension_mismatch_in_step(self):
        """Dimension mismatch caught in step()."""
        system = DiscreteLinear()

        x = np.array([1.0])
        u_wrong = np.array([0.0, 0.0])  # Wrong dimension

        with pytest.raises((ValueError, RuntimeError)):
            system.step(x, u_wrong)


# ============================================================================
# Run Tests
# ============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
