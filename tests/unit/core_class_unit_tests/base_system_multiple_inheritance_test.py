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
Multiple Inheritance Architecture Validation Tests
==================================================

These tests validate that the multiple inheritance architecture works
correctly BEFORE implementing the full ContinuousDynamicalSystem and
DiscreteDynamicalSystem classes.

Critical validation points:
- Method Resolution Order (MRO) is correct
- super().__init__() chains properly
- Abstract methods from both bases are enforced
- No diamond inheritance problems
- Interface contracts are satisfied

Run these tests FIRST before implementing concrete systems!

Usage
-----
pytest tests/test_multiple_inheritance_architecture.py -v
"""

import numpy as np
import pytest
import sympy as sp
from scipy.integrate import solve_ivp

from src.systems.base.core.symbolic_system_base import SymbolicSystemBase
from src.systems.base.core.continuous_system_base import ContinuousSystemBase
from src.systems.base.core.discrete_system_base import DiscreteSystemBase
from src.systems.base.utils.symbolic_validator import ValidationError

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
# Mock Continuous System (Tests Multiple Inheritance)
# ============================================================================


class MockContinuousSystem(SymbolicSystemBase, ContinuousSystemBase):
    """
    Minimal continuous system for testing multiple inheritance.

    Tests that:
    - SymbolicSystemBase and ContinuousSystemBase can be combined
    - MRO resolves correctly
    - All abstract methods are implemented
    - Integration works
    """

    def define_system(self, a: float = 1.0):
        """Define simple continuous system: dx/dt = -a*x + u"""
        x = sp.symbols("x", real=True)
        u = sp.symbols("u", real=True)
        a_sym = sp.symbols("a", positive=True)

        self.state_vars = [x]
        self.control_vars = [u]
        self._f_sym = sp.Matrix([-a_sym * x + u])
        self.parameters = {a_sym: a}
        self.order = 1

    def print_equations(self, simplify: bool = True):
        """Print with continuous notation."""
        print("=" * 70)
        print(f"{self.__class__.__name__} (Continuous)")
        print("=" * 70)
        print(f"Dynamics: dx/dt = f(x, u)")
        for var, expr in zip(self.state_vars, self._f_sym):
            expr_sub = self.substitute_parameters(expr)
            if simplify:
                expr_sub = sp.simplify(expr_sub)
            print(f"  d{var}/dt = {expr_sub}")
        print("=" * 70)

    # Implement ContinuousSystemBase interface
    def __call__(self, x, u=None, t=0.0):
        """Evaluate dx/dt = f(x, u)."""
        u = u if u is not None else np.zeros(self.nu)
        a_val = list(self.parameters.values())[0]
        return -a_val * x + u

    def integrate(self, x0, u=None, t_span=(0.0, 10.0), method="RK45", **kwargs):
        """Integrate using scipy."""
        # Prepare control function
        if u is None:
            u_func = lambda t, x: np.zeros(self.nu) if self.nu > 0 else None
        elif callable(u):
            u_func = u
        else:
            u_func = lambda t, x: u

        # Define ODE RHS
        def rhs(t, x):
            u_val = u_func(t, x)
            return self(x, u_val, t)

        # Solve
        result = solve_ivp(rhs, t_span, x0, method=method, **kwargs)

        return {
            "t": result.t,
            "y": result.y,
            "success": result.success,
            "message": result.message,
            "nfev": result.nfev,
        }

    def linearize(self, x_eq, u_eq=None):
        """Continuous linearization: (A, B)."""
        a_val = list(self.parameters.values())[0]
        A = np.array([[-a_val]])
        B = np.array([[1.0]])
        return (A, B)

    def _verify_equilibrium_numpy(self, x_eq, u_eq, tol):
        """Continuous: f(x_eq, u_eq) ≈ 0"""
        dx = self(x_eq, u_eq, t=0.0)
        return np.linalg.norm(dx) < tol


# ============================================================================
# Mock Discrete System (Tests Multiple Inheritance)
# ============================================================================


class MockDiscreteSystem(SymbolicSystemBase, DiscreteSystemBase):
    """
    Minimal discrete system for testing multiple inheritance.

    Tests that:
    - SymbolicSystemBase and DiscreteSystemBase can be combined
    - MRO resolves correctly
    - All abstract methods are implemented
    - Stepping works
    """

    def define_system(self, a: float = 0.9, dt: float = 0.1):
        """Define simple discrete system: x[k+1] = a*x[k] + b*u[k]"""
        x = sp.symbols("x", real=True)
        u = sp.symbols("u", real=True)
        a_sym = sp.symbols("a", real=True)
        b_sym = sp.symbols("b", real=True)

        self.state_vars = [x]
        self.control_vars = [u]
        self._f_sym = sp.Matrix([a_sym * x + b_sym * u])
        self.parameters = {a_sym: a, b_sym: 0.1}
        self.order = 1
        self._dt = dt  # Discrete systems must set dt!

    def print_equations(self, simplify: bool = True):
        """Print with discrete notation."""
        print("=" * 70)
        print(f"{self.__class__.__name__} (Discrete, dt={self.dt})")
        print("=" * 70)
        print(f"Dynamics: x[k+1] = f(x[k], u[k])")
        for var, expr in zip(self.state_vars, self._f_sym):
            expr_sub = self.substitute_parameters(expr)
            if simplify:
                expr_sub = sp.simplify(expr_sub)
            print(f"  {var}[k+1] = {expr_sub}")
        print("=" * 70)

    # Implement DiscreteSystemBase interface
    @property
    def dt(self) -> float:
        """Sampling period."""
        return self._dt

    def step(self, x, u=None, k=0):
        """Compute x[k+1] = a*x[k] + b*u[k]."""
        u = u if u is not None else np.zeros(self.nu)
        param_values = list(self.parameters.values())
        a, b = param_values[0], param_values[1]
        return a * x + b * u

    def simulate(self, x0, u_sequence=None, n_steps=100, **kwargs):
        """Multi-step simulation with TIME-MAJOR ordering."""
        states = np.zeros((n_steps + 1, self.nx))  # TIME-MAJOR: (n_steps+1, nx)
        states[0, :] = x0

        # Prepare control function
        if u_sequence is None:
            u_func = lambda k: np.zeros(self.nu) if self.nu > 0 else None
        elif callable(u_sequence):
            u_func = u_sequence
        elif isinstance(u_sequence, np.ndarray) and u_sequence.ndim == 1:
            u_func = lambda k: u_sequence
        else:
            u_func = lambda k: u_sequence[k] if k < len(u_sequence) else u_sequence[-1]

        # Simulate
        controls = []
        for k in range(n_steps):
            u = u_func(k)
            controls.append(u)
            states[k + 1, :] = self.step(states[k, :], u, k)

        controls_array = np.array(controls) if controls[0] is not None else None  # (n_steps, nu)

        return {
            "states": states,  # (n_steps+1, nx)
            "controls": controls_array,  # (n_steps, nu)
            "time_steps": np.arange(n_steps + 1),
            "dt": self.dt,
            "metadata": {"method": "discrete_step", "success": True, **kwargs},
        }

    def linearize(self, x_eq, u_eq=None):
        """Discrete linearization: (Ad, Bd)."""
        param_values = list(self.parameters.values())
        a, b = param_values[0], param_values[1]
        Ad = np.array([[a]])
        Bd = np.array([[b]])
        return (Ad, Bd)

    def _verify_equilibrium_numpy(self, x_eq, u_eq, tol):
        """Discrete: f(x_eq, u_eq) ≈ x_eq"""
        x_next = self.step(x_eq, u_eq, k=0)
        return np.linalg.norm(x_next - x_eq) < tol


# ============================================================================
# Test: MRO and Initialization
# ============================================================================


class TestMultipleInheritanceMRO:
    """Test Method Resolution Order for multiple inheritance."""

    def test_continuous_mro_order(self):
        """MRO for continuous system is correct."""
        mro = MockContinuousSystem.__mro__

        # Should be: MockContinuousSystem -> SymbolicSystemBase -> ContinuousSystemBase -> ABC -> object
        assert mro[0] == MockContinuousSystem
        assert SymbolicSystemBase in mro
        assert ContinuousSystemBase in mro

        # SymbolicSystemBase should come before ContinuousSystemBase
        symbolic_idx = mro.index(SymbolicSystemBase)
        continuous_idx = mro.index(ContinuousSystemBase)
        assert symbolic_idx < continuous_idx

    def test_discrete_mro_order(self):
        """MRO for discrete system is correct."""
        mro = MockDiscreteSystem.__mro__

        assert mro[0] == MockDiscreteSystem
        assert SymbolicSystemBase in mro
        assert DiscreteSystemBase in mro

        # SymbolicSystemBase should come before DiscreteSystemBase
        symbolic_idx = mro.index(SymbolicSystemBase)
        discrete_idx = mro.index(DiscreteSystemBase)
        assert symbolic_idx < discrete_idx

    def test_continuous_initialization_sequence(self):
        """Continuous system initializes both bases."""
        system = MockContinuousSystem(a=2.0)

        # SymbolicSystemBase initialization
        assert system._initialized
        assert system.state_vars
        assert system._code_gen is not None

        # ContinuousSystemBase properties
        assert system.is_continuous
        assert not system.is_discrete

    def test_discrete_initialization_sequence(self):
        """Discrete system initializes both bases."""
        system = MockDiscreteSystem(a=0.95, dt=0.01)

        # SymbolicSystemBase initialization
        assert system._initialized
        assert system.state_vars
        assert system._code_gen is not None

        # DiscreteSystemBase properties
        assert system.is_discrete
        assert not system.is_continuous
        assert system.dt == 0.01


# ============================================================================
# Test: Interface Implementation
# ============================================================================


class TestInterfaceImplementation:
    """Test that both interfaces are properly implemented."""

    def test_continuous_has_symbolic_methods(self):
        """Continuous system has SymbolicSystemBase methods."""
        system = MockContinuousSystem()

        # Symbolic methods
        assert hasattr(system, "substitute_parameters")
        assert hasattr(system, "compile")
        assert hasattr(system, "reset_caches")
        assert hasattr(system, "add_equilibrium")
        assert hasattr(system, "get_config_dict")

        # Properties
        assert hasattr(system, "nx")
        assert hasattr(system, "nu")
        assert hasattr(system, "ny")

    def test_continuous_has_continuous_methods(self):
        """Continuous system has ContinuousSystemBase methods."""
        system = MockContinuousSystem()

        # Continuous methods
        assert callable(system)  # __call__
        assert hasattr(system, "integrate")
        assert hasattr(system, "linearize")
        assert hasattr(system, "simulate")

        # Properties
        assert system.is_continuous
        assert not system.is_discrete

    def test_discrete_has_symbolic_methods(self):
        """Discrete system has SymbolicSystemBase methods."""
        system = MockDiscreteSystem()

        # Symbolic methods
        assert hasattr(system, "substitute_parameters")
        assert hasattr(system, "compile")
        assert hasattr(system, "add_equilibrium")

        # Properties
        assert hasattr(system, "nx")
        assert hasattr(system, "nu")

    def test_discrete_has_discrete_methods(self):
        """Discrete system has DiscreteSystemBase methods."""
        system = MockDiscreteSystem()

        # Discrete methods
        assert hasattr(system, "step")
        assert hasattr(system, "simulate")
        assert hasattr(system, "linearize")
        assert hasattr(system, "rollout")

        # Properties
        assert hasattr(system, "dt")
        assert system.is_discrete
        assert not system.is_continuous


# ============================================================================
# Test: Functional Validation
# ============================================================================


class TestFunctionalValidation:
    """Test that systems actually work end-to-end."""

    def test_continuous_dynamics_evaluation(self):
        """Continuous system can evaluate dynamics."""
        system = MockContinuousSystem(a=1.0)

        x = np.array([1.0])
        u = np.array([0.5])

        dx = system(x, u)

        assert dx.shape == (1,)
        expected = -1.0 * 1.0 + 0.5  # -a*x + u
        np.testing.assert_allclose(dx, [expected])

    def test_continuous_integration(self):
        """Continuous system can integrate."""
        system = MockContinuousSystem(a=1.0)

        x0 = np.array([1.0])
        result = system.integrate(x0, u=None, t_span=(0.0, 1.0))

        assert result["success"]
        assert result["t"].shape[0] > 0
        assert result["y"].shape == (1, result["t"].shape[0])

        # System should decay toward zero
        assert np.abs(result["y"][0, -1]) < np.abs(x0[0])

    def test_continuous_linearization(self):
        """Continuous system can linearize."""
        system = MockContinuousSystem(a=2.0)

        x_eq = np.array([0.0])
        u_eq = np.array([0.0])

        A, B = system.linearize(x_eq, u_eq)

        assert A.shape == (1, 1)
        assert B.shape == (1, 1)
        np.testing.assert_allclose(A, [[-2.0]])
        np.testing.assert_allclose(B, [[1.0]])

    def test_discrete_step(self):
        """Discrete system can step."""
        system = MockDiscreteSystem(a=0.9, dt=0.1)

        x = np.array([1.0])
        u = np.array([0.5])

        x_next = system.step(x, u)

        assert x_next.shape == (1,)
        expected = 0.9 * 1.0 + 0.1 * 0.5  # a*x + b*u
        np.testing.assert_allclose(x_next, [expected])

    def test_discrete_simulate(self):
        """Discrete system can simulate."""
        system = MockDiscreteSystem(a=0.9, dt=0.1)

        x0 = np.array([1.0])
        result = system.simulate(x0, u_sequence=None, n_steps=10)

        # TIME-MAJOR: (n_steps+1, nx)
        assert result["states"].shape == (11, 1)  # Not (1, 11)
        assert result["time_steps"].shape == (11,)
        assert result["dt"] == 0.1

        # System should decay toward zero
        assert np.abs(result["states"][-1, 0]) < np.abs(x0[0])

    def test_discrete_linearization(self):
        """Discrete system can linearize."""
        system = MockDiscreteSystem(a=0.95, dt=0.1)

        x_eq = np.array([0.0])
        u_eq = np.array([0.0])

        Ad, Bd = system.linearize(x_eq, u_eq)

        assert Ad.shape == (1, 1)
        assert Bd.shape == (1, 1)
        np.testing.assert_allclose(Ad, [[0.95]])
        np.testing.assert_allclose(Bd, [[0.1]])

    def test_discrete_rollout(self):
        """Discrete system can rollout with policy."""
        system = MockDiscreteSystem(a=0.9, dt=0.1)

        def policy(x, k):
            return -0.5 * x  # Simple feedback

        x0 = np.array([1.0])
        result = system.rollout(x0, policy, n_steps=10)

        # TIME-MAJOR: (n_steps+1, nx)
        assert result["states"].shape == (11, 1)  # Not (1, 11)
        assert result["controls"] is not None


# ============================================================================
# Test: Equilibrium Verification Works
# ============================================================================


class TestEquilibriumVerification:
    """Test that equilibrium verification works with concrete implementations."""

    def test_continuous_equilibrium_verification_success(self):
        """Continuous system verifies valid equilibrium."""
        system = MockContinuousSystem(a=1.0)

        # For dx/dt = -x + u, equilibrium at x=1, u=1
        x_eq = np.array([1.0])
        u_eq = np.array([1.0])

        # Should verify successfully
        system.add_equilibrium("test", x_eq, u_eq, verify=True, tol=1e-6)

        assert "test" in system.list_equilibria()

    def test_continuous_equilibrium_verification_failure(self):
        """Continuous system warns on invalid equilibrium."""
        system = MockContinuousSystem(a=1.0)

        # NOT an equilibrium: dx/dt = -0 + 0 = 0 ✓, but we'll use wrong point
        x_eq = np.array([1.0])
        u_eq = np.array([0.0])  # Wrong! Should be u=1

        # Should warn but still add
        with pytest.warns(UserWarning, match="verification"):
            system.add_equilibrium("test", x_eq, u_eq, verify=True, tol=1e-10)

    def test_discrete_equilibrium_verification_success(self):
        """Discrete system verifies valid equilibrium."""
        system = MockDiscreteSystem(a=0.9, dt=0.1)

        # For x[k+1] = 0.9*x + 0.1*u, equilibrium when x = 0.9*x + 0.1*u
        # => 0.1*x = 0.1*u => x = u
        x_eq = np.array([1.0])
        u_eq = np.array([1.0])

        system.add_equilibrium("test", x_eq, u_eq, verify=True, tol=1e-6)

        assert "test" in system.list_equilibria()

    def test_verification_hook_not_implemented_warns(self):
        """Base class warns if verification not implemented."""

        # Create system that doesn't override _verify_equilibrium_numpy
        class NoVerifySystem(SymbolicSystemBase):
            def define_system(self):
                x = sp.symbols("x")
                self.state_vars = [x]
                self.control_vars = []
                self._f_sym = sp.Matrix([0])
                self.parameters = {}
                self.order = 1

            def print_equations(self, simplify=True):
                pass

            # Note: Does NOT implement _verify_equilibrium_numpy

        system = NoVerifySystem()

        with pytest.warns(UserWarning, match="does not implement"):
            system.add_equilibrium("test", np.array([0.0]), np.array([]), verify=True)


# ============================================================================
# Test: Symbolic Properties Work with Time-Domain
# ============================================================================


class TestSymbolicPropertiesWithTimeDomain:
    """Test symbolic properties work correctly with time-domain interfaces."""

    def test_continuous_substitute_parameters(self):
        """Parameter substitution works in continuous system."""
        system = MockContinuousSystem(a=3.0)

        a_sym = list(system.parameters.keys())[0]
        x = sp.symbols("x")
        expr = a_sym * x

        result = system.substitute_parameters(expr)
        assert result == 3.0 * x

    def test_discrete_substitute_parameters(self):
        """Parameter substitution works in discrete system."""
        system = MockDiscreteSystem(a=0.8, dt=0.1)

        result = system.substitute_parameters(system._f_sym)

        # Should have no parameter symbols
        for param_sym in system.parameters.keys():
            assert param_sym not in result.free_symbols

    def test_continuous_dimensions(self):
        """Continuous system has correct dimensions."""
        system = MockContinuousSystem()

        assert system.nx == 1
        assert system.nu == 1
        assert system.ny == 1
        assert system.nq == 1

    def test_discrete_dimensions(self):
        """Discrete system has correct dimensions."""
        system = MockDiscreteSystem()

        assert system.nx == 1
        assert system.nu == 1
        assert system.ny == 1
        assert system.dt == 0.1


# ============================================================================
# Test: No Diamond Inheritance Problems
# ============================================================================


class TestNoDiamondProblems:
    """Verify no diamond inheritance issues."""

    def test_no_duplicate_methods(self):
        """No methods defined in multiple bases (except object)."""
        continuous_methods = set(dir(ContinuousSystemBase))
        symbolic_methods = set(dir(SymbolicSystemBase))

        # Should have no overlap except object methods
        overlap = continuous_methods & symbolic_methods
        overlap = {m for m in overlap if not m.startswith("_")}

        # These are expected overlaps from object
        allowed_overlap = {"__class__", "__init__", "__repr__", "__str__"}
        unexpected = overlap - allowed_overlap

        assert len(unexpected) == 0, f"Unexpected method overlap: {unexpected}"

    def test_super_init_called_once(self):
        """super().__init__() in SymbolicSystemBase doesn't cause double initialization."""
        # This is hard to test directly, but we can verify no errors occur
        system = MockContinuousSystem()

        # Should initialize successfully without errors
        assert system._initialized

    def test_cooperative_inheritance_works(self):
        """Cooperative inheritance chains work correctly."""
        # Create system - if cooperative inheritance broken, this would fail
        system = MockContinuousSystem(a=1.5)

        # Both bases should be properly initialized
        assert system._initialized  # From SymbolicSystemBase
        assert hasattr(system, "is_continuous")  # From ContinuousSystemBase
        assert system.is_continuous


# ============================================================================
# Test: Backend Operations with Both Interfaces
# ============================================================================


class TestBackendOperationsMultipleInheritance:
    """Test backend operations work with multiple inheritance."""

    def test_continuous_backend_switching(self):
        """Continuous system can switch backends."""
        system = MockContinuousSystem()

        system.set_default_backend("numpy")
        assert system._default_backend == "numpy"

        # Can still evaluate dynamics
        dx = system(np.array([1.0]), np.array([0.0]))
        assert dx.shape == (1,)

    def test_discrete_backend_switching(self):
        """Discrete system can switch backends."""
        system = MockDiscreteSystem()

        system.set_default_backend("numpy")
        assert system._default_backend == "numpy"

        # Can still step
        x_next = system.step(np.array([1.0]), np.array([0.0]))
        assert x_next.shape == (1,)


# ============================================================================
# Test: Print Equations with Correct Notation
# ============================================================================


class TestPrintEquationsNotation:
    """Test print_equations uses correct notation for time domain."""

    def test_continuous_uses_derivative_notation(self, capsys):
        """Continuous system uses dx/dt notation."""
        system = MockContinuousSystem()

        system.print_equations()

        captured = capsys.readouterr()
        assert "dx/dt" in captured.out or "d" in captured.out

    def test_discrete_uses_index_notation(self, capsys):
        """Discrete system uses x[k+1] notation."""
        system = MockDiscreteSystem()

        system.print_equations()

        captured = capsys.readouterr()
        assert "[k+1]" in captured.out or "k+1" in captured.out


# ============================================================================
# Test: Stability Analysis Example
# ============================================================================


class TestStabilityAnalysisExample:
    """Example showing how to use both interfaces for stability analysis."""

    def test_continuous_stability_from_linearization(self):
        """Continuous system stability via eigenvalues."""
        system = MockContinuousSystem(a=2.0)

        # Linearize at origin
        A, B = system.linearize(np.zeros(1), np.zeros(1))

        # Check continuous stability: Re(λ) < 0
        eigenvalues = np.linalg.eigvals(A)
        is_stable = np.all(np.real(eigenvalues) < 0)

        assert is_stable  # a=2 > 0, so A = -2 is stable

    def test_discrete_stability_from_linearization(self):
        """Discrete system stability via eigenvalues."""
        system = MockDiscreteSystem(a=0.8, dt=0.1)

        # Linearize at origin
        Ad, Bd = system.linearize(np.zeros(1), np.zeros(1))

        # Check discrete stability: |λ| < 1
        eigenvalues = np.linalg.eigvals(Ad)
        is_stable = np.all(np.abs(eigenvalues) < 1.0)

        assert is_stable  # a=0.8 < 1, so stable

    def test_integration_matches_step_for_small_dt(self):
        """Continuous integration ≈ discrete stepping for small dt."""
        # This tests that discretization is consistent

        # Continuous system
        cont = MockContinuousSystem(a=1.0)

        # Integrate for one step
        dt = 0.01
        x0 = np.array([1.0])
        u = np.array([0.0])

        result = cont.integrate(x0, u, t_span=(0.0, dt), method="RK45")
        x_final_cont = result["y"][:, -1]

        # Discrete system (Euler approximation)
        disc = MockDiscreteSystem(a=1.0 - dt * 1.0, dt=dt)  # Euler: Ad ≈ I + dt*Ac
        x_final_disc = disc.step(x0, u)

        # Should be close (not exact due to RK45 vs Euler)
        np.testing.assert_allclose(x_final_cont, x_final_disc, rtol=0.1)


# ============================================================================
# Test: Complete Workflow Examples
# ============================================================================


class TestCompleteWorkflows:
    """End-to-end workflow tests."""

    def test_continuous_workflow_simulate_and_analyze(self):
        """Complete continuous workflow."""
        # 1. Create system
        system = MockContinuousSystem(a=1.0)

        # 2. Configure
        system.set_default_backend("numpy")

        # 3. Add equilibrium
        system.add_equilibrium("origin", np.zeros(1), np.zeros(1), verify=True)

        # 4. Integrate
        result = system.integrate(np.array([1.0]), None, (0, 2))

        # 5. Linearize
        A, B = system.linearize(np.zeros(1), np.zeros(1))

        # 6. Check stability
        is_stable = np.all(np.real(np.linalg.eigvals(A)) < 0)

        assert result["success"]
        assert is_stable

    def test_discrete_workflow_simulate_and_control(self):
        """Complete discrete workflow."""
        # 1. Create system
        system = MockDiscreteSystem(a=0.95, dt=0.01)

        # 2. Design controller (LQR-like)
        Ad, Bd = system.linearize(np.zeros(1), np.zeros(1))
        K = np.array([[0.5]])  # Simple gain

        # 3. Simulate with feedback
        def policy(x, k):
            return -K @ x

        result = system.rollout(np.array([1.0]), policy, n_steps=50)

        # 4. Verify convergence (TIME-MAJOR)
        final_state = result["states"][-1, :]  # Not [:, -1]
        assert np.abs(final_state[0]) < 0.1  # Should converge toward zero


# ============================================================================
# Test: Type Checking Compatibility
# ============================================================================


class TestTypeCheckingCompatibility:
    """Test that systems work with type annotations."""

    def test_continuous_type_annotations(self):
        """Continuous system satisfies type hints."""
        system: ContinuousSystemBase = MockContinuousSystem()

        # Should satisfy ContinuousSystemBase interface
        assert isinstance(system, ContinuousSystemBase)
        assert isinstance(system, SymbolicSystemBase)

    def test_discrete_type_annotations(self):
        """Discrete system satisfies type hints."""
        system: DiscreteSystemBase = MockDiscreteSystem()

        # Should satisfy DiscreteSystemBase interface
        assert isinstance(system, DiscreteSystemBase)
        assert isinstance(system, SymbolicSystemBase)

    def test_polymorphic_function_works(self):
        """Polymorphic functions work with both system types."""
        from src.types.utilities import LinearizableProtocol

        def analyze_stability(system: LinearizableProtocol):
            """Works with any linearizable system."""
            x_eq = np.zeros(system.nx)
            u_eq = np.zeros(system.nu)
            result = system.linearize(x_eq, u_eq)
            A = result[0]
            return np.linalg.eigvals(A)

        # Test with both
        cont = MockContinuousSystem()
        disc = MockDiscreteSystem()

        eigs_cont = analyze_stability(cont)
        eigs_disc = analyze_stability(disc)

        assert eigs_cont.shape[0] == 1
        assert eigs_disc.shape[0] == 1


# ============================================================================
# Run Tests
# ============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
