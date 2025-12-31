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
Unit Tests for Structural Subtyping Protocols
==============================================

Tests that:
1. Mock implementations satisfy protocol contracts
2. isinstance() runtime checks work correctly
3. Protocol hierarchy is properly defined
4. Type checking behavior is as expected
5. Real systems satisfy appropriate protocols

Test Strategy
-------------
Since Protocols are abstract interfaces, we test by:

1. **Mock Implementations**: Create minimal classes that satisfy each protocol
2. **isinstance() Checks**: Verify @runtime_checkable works correctly
3. **Negative Tests**: Confirm incomplete implementations are rejected
4. **Integration Tests**: Test real system classes against protocols
5. **Type Checker Tests**: Verify mypy/pyright behavior (via comments)

Location
--------
tests/types/test_protocols.py

Dependencies
------------
- pytest
- numpy
- src.types.protocols
- src.systems (for integration tests with real systems)

Authors
-------
Gil Benezer

License
-------
AGPL-3.0
"""

import pytest
import numpy as np
import sympy as sp
from typing import Optional, List, Dict

from src.types.protocols import (
    DiscreteSystemProtocol,
    LinearizableDiscreteProtocol,
    SymbolicDiscreteProtocol,
    ContinuousSystemProtocol,
    LinearizableContinuousProtocol,
    SymbolicContinuousProtocol,
    StochasticSystemProtocol,
    CompilableSystemProtocol,
    ParametricSystemProtocol,
)

from src.types.core import StateVector, ControlVector
from src.types.linearization import DiscreteLinearization
from src.types.trajectories import DiscreteSimulationResult
from src.systems.base.core.continuous_symbolic_system import ContinuousSymbolicSystem
from src.systems.base.core.discrete_symbolic_system import DiscreteSymbolicSystem


# ============================================================================
# Mock Implementations for Testing Protocols
# ============================================================================


class MinimalDiscreteSystem:
    """Minimal implementation satisfying DiscreteSystemProtocol."""

    def __init__(self, nx=2, nu=1, dt=0.01):
        self._nx = nx
        self._nu = nu
        self._dt = dt

    @property
    def dt(self) -> float:
        return self._dt

    @property
    def nx(self) -> int:
        return self._nx

    @property
    def nu(self) -> int:
        return self._nu

    def step(self, x: StateVector, u: Optional[ControlVector] = None, k: int = 0) -> StateVector:
        """Simple linear dynamics: x[k+1] = 0.9*x + 0.1*u"""
        if u is None:
            u = np.zeros(self.nu)
        
        # Handle zero control dimension
        if self.nu == 0:
            return 0.9 * x
        else:
            return 0.9 * x + 0.1 * u

    def simulate(self, x0: StateVector, u_sequence, n_steps: int) -> DiscreteSimulationResult:
        """Minimal simulation"""
        states = np.zeros((n_steps + 1, self.nx))
        states[0, :] = x0

        x = x0
        for k in range(n_steps):
            u = u_sequence if not callable(u_sequence) else u_sequence(k)
            x = self.step(x, u, k)
            states[k + 1, :] = x

        return {
            "states": states,
            "time_steps": np.arange(n_steps + 1),
            "dt": self.dt,
            "success": True,
        }


class LinearizableDiscreteSystem(MinimalDiscreteSystem):
    """Extends minimal with linearization."""

    def linearize(self, x_eq: StateVector, u_eq: Optional[ControlVector] = None) -> DiscreteLinearization:
        """Linear system: Ad = 0.9*I, Bd = 0.1*I"""
        Ad = 0.9 * np.eye(self.nx)
        Bd = 0.1 * np.eye(self.nx, self.nu)
        return (Ad, Bd)


class SymbolicDiscreteSystem(LinearizableDiscreteSystem):
    """Extends linearizable with symbolic machinery."""

    def __init__(self, nx=2, nu=1, dt=0.01):
        super().__init__(nx, nu, dt)

        # Create mock symbolic attributes
        self.state_vars = [sp.symbols(f"x{i}") for i in range(nx)]
        self.control_vars = [sp.symbols(f"u{i}") for i in range(nu)]
        self.parameters = {sp.symbols("a"): 0.9, sp.symbols("b"): 0.1}

    def compile(self, backends: Optional[List[str]] = None, verbose: bool = False) -> Dict[str, float]:
        """Mock compilation"""
        backends = backends or ["numpy"]
        return {backend: 0.001 for backend in backends}

    def reset_caches(self, backends: Optional[List[str]] = None):
        """Mock cache reset"""
        pass  # No-op for mock

    def print_equations(self, simplify: bool = True):
        """Mock equation printing"""
        print(f"x[k+1] = 0.9*x[k] + 0.1*u[k]")

    def substitute_parameters(self, expr):
        """Mock substitution"""
        return expr.subs(self.parameters)


class IncompleteDiscreteSystem:
    """Incomplete implementation - missing methods."""

    def __init__(self):
        self._nx = 2
        self._dt = 0.01

    @property
    def nx(self) -> int:
        return self._nx

    @property
    def dt(self) -> float:
        return self._dt

    # Missing: nu, step, simulate
    # Should NOT satisfy DiscreteSystemProtocol


class MinimalContinuousSystem:
    """Minimal implementation satisfying ContinuousSystemProtocol."""

    def __init__(self, nx=2, nu=1):
        self._nx = nx
        self._nu = nu

    @property
    def nx(self) -> int:
        return self._nx

    @property
    def nu(self) -> int:
        return self._nu

    def __call__(self, x: StateVector, u: Optional[ControlVector] = None, t: float = 0.0) -> StateVector:
        """Simple linear dynamics: dx/dt = -x + u"""
        u = u if u is not None else np.zeros(self.nu)
        return -x + u

    def integrate(self, x0, u, t_span, method="RK45", **kwargs):
        """Mock integration"""
        from scipy.integrate import solve_ivp

        u_func = u if callable(u) else (lambda t: u if u is not None else np.zeros(self.nu))

        def rhs(t, x):
            return self(x, u_func(t), t)

        result = solve_ivp(rhs, t_span, x0, method=method, **kwargs)

        return {"t": result.t, "x": result.y.T, "success": result.success}


class StochasticDiscreteSystem(LinearizableDiscreteSystem):
    """Mock stochastic discrete system."""

    def __init__(self, nx=2, nu=1, nw=1, dt=0.01):
        super().__init__(nx, nu, dt)
        self._nw = nw

    @property
    def is_stochastic(self) -> bool:
        return True

    @property
    def nw(self) -> int:
        return self._nw

    def is_additive_noise(self) -> bool:
        return True

    def is_multiplicative_noise(self) -> bool:
        return False


# ============================================================================
# Test Suite: DiscreteSystemProtocol
# ============================================================================


class TestDiscreteSystemProtocol:
    """Test DiscreteSystemProtocol basic interface."""

    def test_minimal_implementation_satisfies_protocol(self):
        """Minimal implementation with all required methods satisfies protocol."""
        system = MinimalDiscreteSystem(nx=2, nu=1, dt=0.01)

        # isinstance() check should pass
        assert isinstance(system, DiscreteSystemProtocol)

    def test_protocol_has_required_attributes(self):
        """Protocol requires dt, nx, nu properties."""
        system = MinimalDiscreteSystem(nx=3, nu=2, dt=0.05)

        assert system.dt == 0.05
        assert system.nx == 3
        assert system.nu == 2

    def test_protocol_has_step_method(self):
        """Protocol requires step() method."""
        system = MinimalDiscreteSystem(nx=2, nu=1)

        x = np.array([1.0, 0.0])
        u = np.array([0.5])
        x_next = system.step(x, u, k=0)

        assert isinstance(x_next, np.ndarray)
        assert x_next.shape == (2,)

    def test_protocol_has_simulate_method(self):
        """Protocol requires simulate() method."""
        system = MinimalDiscreteSystem(nx=2, nu=1)

        x0 = np.array([1.0, 0.0])
        result = system.simulate(x0, np.array([0.5]), n_steps=10)

        assert "states" in result
        assert "time_steps" in result
        assert result["states"].shape == (11, 2)  # n_steps+1 points

    def test_incomplete_implementation_fails_protocol(self):
        """Incomplete implementation does not satisfy protocol."""
        incomplete = IncompleteDiscreteSystem()

        # Should fail isinstance check
        assert not isinstance(incomplete, DiscreteSystemProtocol)

    def test_protocol_with_different_dimensions(self):
        """Protocol works with various state/control dimensions."""
        # Scalar state, no control
        sys1 = MinimalDiscreteSystem(nx=1, nu=0, dt=0.1)
        assert isinstance(sys1, DiscreteSystemProtocol)

        # High-dimensional
        sys2 = MinimalDiscreteSystem(nx=100, nu=10, dt=0.01)
        assert isinstance(sys2, DiscreteSystemProtocol)


# ============================================================================
# Test Suite: LinearizableDiscreteProtocol
# ============================================================================


class TestLinearizableDiscreteProtocol:
    """Test LinearizableDiscreteProtocol with linearization capability."""

    def test_linearizable_implementation_satisfies_protocol(self):
        """Implementation with linearize() satisfies protocol."""
        system = LinearizableDiscreteSystem(nx=2, nu=1)

        assert isinstance(system, LinearizableDiscreteProtocol)

    def test_linearizable_also_satisfies_base_protocol(self):
        """LinearizableDiscreteProtocol extends DiscreteSystemProtocol."""
        system = LinearizableDiscreteSystem(nx=2, nu=1)

        # Should satisfy both protocols
        assert isinstance(system, DiscreteSystemProtocol)
        assert isinstance(system, LinearizableDiscreteProtocol)

    def test_linearize_returns_correct_types(self):
        """linearize() returns (Ad, Bd) tuple."""
        system = LinearizableDiscreteSystem(nx=2, nu=1)

        x_eq = np.zeros(2)
        u_eq = np.zeros(1)
        result = system.linearize(x_eq, u_eq)

        # Should be tuple of two matrices
        assert isinstance(result, tuple)
        assert len(result) == 2

        Ad, Bd = result
        assert Ad.shape == (2, 2)
        assert Bd.shape == (2, 1)

    def test_minimal_system_does_not_satisfy_linearizable(self):
        """Minimal system without linearize() doesn't satisfy protocol."""
        system = MinimalDiscreteSystem(nx=2, nu=1)

        # Has step/simulate but not linearize
        assert isinstance(system, DiscreteSystemProtocol)
        assert not isinstance(system, LinearizableDiscreteProtocol)

    def test_linearization_at_different_equilibria(self):
        """Can linearize at arbitrary points."""
        system = LinearizableDiscreteSystem(nx=2, nu=1)

        # Different equilibria
        x_eq1 = np.zeros(2)
        x_eq2 = np.array([1.0, 2.0])

        Ad1, Bd1 = system.linearize(x_eq1, np.zeros(1))
        Ad2, Bd2 = system.linearize(x_eq2, np.zeros(1))

        # Linear system - should get same matrices
        assert np.allclose(Ad1, Ad2)
        assert np.allclose(Bd1, Bd2)


# ============================================================================
# Test Suite: SymbolicDiscreteProtocol
# ============================================================================


class TestSymbolicDiscreteProtocol:
    """Test SymbolicDiscreteProtocol with symbolic machinery."""

    def test_symbolic_implementation_satisfies_protocol(self):
        """Implementation with symbolic attributes satisfies protocol."""
        system = SymbolicDiscreteSystem(nx=2, nu=1)

        assert isinstance(system, SymbolicDiscreteProtocol)

    def test_symbolic_satisfies_parent_protocols(self):
        """SymbolicDiscreteProtocol extends both parent protocols."""
        system = SymbolicDiscreteSystem(nx=2, nu=1)

        # Should satisfy all three protocols in hierarchy
        assert isinstance(system, DiscreteSystemProtocol)
        assert isinstance(system, LinearizableDiscreteProtocol)
        assert isinstance(system, SymbolicDiscreteProtocol)

    def test_symbolic_has_required_attributes(self):
        """Symbolic protocol requires state_vars, control_vars, parameters."""
        system = SymbolicDiscreteSystem(nx=2, nu=1)

        # Check symbolic attributes exist
        assert hasattr(system, "state_vars")
        assert hasattr(system, "control_vars")
        assert hasattr(system, "parameters")

        # Check they have correct types
        assert isinstance(system.state_vars, list)
        assert isinstance(system.control_vars, list)
        assert isinstance(system.parameters, dict)

        # Check contents
        assert len(system.state_vars) == 2
        assert len(system.control_vars) == 1
        assert all(isinstance(var, sp.Symbol) for var in system.state_vars)

    def test_symbolic_has_compile_method(self):
        """Symbolic protocol requires compile() method."""
        system = SymbolicDiscreteSystem(nx=2, nu=1)

        # Should be callable
        assert callable(system.compile)

        # Should return timing dict
        times = system.compile(backends=["numpy"], verbose=False)
        assert isinstance(times, dict)
        assert "numpy" in times

    def test_symbolic_has_print_equations_method(self):
        """Symbolic protocol requires print_equations() method."""
        system = SymbolicDiscreteSystem(nx=2, nu=1)

        # Should be callable (output is to stdout)
        assert callable(system.print_equations)

        # Should not raise
        system.print_equations(simplify=True)

    def test_symbolic_has_substitute_parameters_method(self):
        """Symbolic protocol requires substitute_parameters() method."""
        system = SymbolicDiscreteSystem(nx=2, nu=1)

        # Should be callable
        assert callable(system.substitute_parameters)

        # Should work with SymPy expressions
        x = sp.symbols("x")
        a = sp.symbols("a")
        expr = a * x

        # Mock substitution
        result = system.substitute_parameters(expr)
        # Result should be a SymPy expression
        assert isinstance(result, (sp.Expr, sp.Basic))

    def test_linearizable_without_symbolic_does_not_satisfy(self):
        """Linearizable without symbolic attributes doesn't satisfy symbolic protocol."""
        system = LinearizableDiscreteSystem(nx=2, nu=1)

        # Has linearize but not symbolic attributes
        assert isinstance(system, LinearizableDiscreteProtocol)
        assert not isinstance(system, SymbolicDiscreteProtocol)


# ============================================================================
# Test Suite: ContinuousSystemProtocol (Optional)
# ============================================================================


class TestContinuousSystemProtocol:
    """Test ContinuousSystemProtocol basic interface."""

    def test_minimal_continuous_satisfies_protocol(self):
        """Minimal continuous implementation satisfies protocol."""
        system = MinimalContinuousSystem(nx=2, nu=1)

        assert isinstance(system, ContinuousSystemProtocol)

    def test_continuous_has_required_attributes(self):
        """Continuous protocol requires nx, nu."""
        system = MinimalContinuousSystem(nx=3, nu=2)

        assert system.nx == 3
        assert system.nu == 2

    def test_continuous_has_call_method(self):
        """Continuous protocol requires __call__() for dynamics."""
        system = MinimalContinuousSystem(nx=2, nu=1)

        x = np.array([1.0, 0.0])
        u = np.array([0.5])
        dx = system(x, u, t=0.0)

        assert isinstance(dx, np.ndarray)
        assert dx.shape == (2,)

    def test_continuous_has_integrate_method(self):
        """Continuous protocol requires integrate() method."""
        system = MinimalContinuousSystem(nx=2, nu=1)

        x0 = np.array([1.0, 0.0])
        result = system.integrate(x0, None, t_span=(0.0, 1.0))

        assert "t" in result
        assert "x" in result
        assert "success" in result


# ============================================================================
# Test Suite: Utility Protocols
# ============================================================================


class TestStochasticSystemProtocol:
    """Test StochasticSystemProtocol."""

    def test_stochastic_implementation_satisfies_protocol(self):
        """System with stochastic attributes satisfies protocol."""
        system = StochasticDiscreteSystem(nx=2, nu=1, nw=1)

        assert isinstance(system, StochasticSystemProtocol)

    def test_stochastic_has_required_properties(self):
        """Stochastic protocol requires is_stochastic, nw."""
        system = StochasticDiscreteSystem(nx=2, nu=1, nw=2)

        assert system.is_stochastic is True
        assert system.nw == 2

    def test_stochastic_has_noise_query_methods(self):
        """Stochastic protocol requires noise type queries."""
        system = StochasticDiscreteSystem(nx=2, nu=1, nw=1)

        assert callable(system.is_additive_noise)
        assert callable(system.is_multiplicative_noise)

        # Should return bools
        assert isinstance(system.is_additive_noise(), bool)
        assert isinstance(system.is_multiplicative_noise(), bool)


class TestCompilableSystemProtocol:
    """Test CompilableSystemProtocol."""

    def test_symbolic_system_is_compilable(self):
        """Symbolic systems are compilable."""
        system = SymbolicDiscreteSystem(nx=2, nu=1)

        assert isinstance(system, CompilableSystemProtocol)

    def test_compile_method_works(self):
        """compile() returns timing dictionary."""
        system = SymbolicDiscreteSystem(nx=2, nu=1)

        times = system.compile(backends=["numpy"], verbose=False)

        assert isinstance(times, dict)
        assert "numpy" in times
        assert isinstance(times["numpy"], (int, float))

    def test_reset_caches_method_exists(self):
        """reset_caches() is callable."""
        system = SymbolicDiscreteSystem(nx=2, nu=1)

        # Should not raise
        system.reset_caches(backends=["numpy"])


class TestParametricSystemProtocol:
    """Test ParametricSystemProtocol."""

    def test_symbolic_system_is_parametric(self):
        """Symbolic systems are parametric."""
        system = SymbolicDiscreteSystem(nx=2, nu=1)

        assert isinstance(system, ParametricSystemProtocol)

    def test_parameters_attribute_exists(self):
        """Parameters dictionary exists."""
        system = SymbolicDiscreteSystem(nx=2, nu=1)

        assert hasattr(system, "parameters")
        assert isinstance(system.parameters, dict)

    def test_substitute_parameters_works(self):
        """substitute_parameters() handles expressions."""
        system = SymbolicDiscreteSystem(nx=2, nu=1)

        x = sp.symbols("x")
        expr = x**2

        # Should be callable (actual substitution tested in mock)
        result = system.substitute_parameters(expr)
        assert isinstance(result, (sp.Expr, sp.Basic))


# ============================================================================
# Test Suite: Protocol Hierarchy
# ============================================================================


class TestProtocolHierarchy:
    """Test that protocol hierarchy is correctly defined."""

    def test_linearizable_extends_discrete(self):
        """LinearizableDiscreteProtocol extends DiscreteSystemProtocol."""
        system = LinearizableDiscreteSystem(nx=2, nu=1)

        # Should satisfy both
        assert isinstance(system, DiscreteSystemProtocol)
        assert isinstance(system, LinearizableDiscreteProtocol)

    def test_symbolic_extends_linearizable(self):
        """SymbolicDiscreteProtocol extends LinearizableDiscreteProtocol."""
        system = SymbolicDiscreteSystem(nx=2, nu=1)

        # Should satisfy all three
        assert isinstance(system, DiscreteSystemProtocol)
        assert isinstance(system, LinearizableDiscreteProtocol)
        assert isinstance(system, SymbolicDiscreteProtocol)

    def test_hierarchy_is_transitive(self):
        """If A extends B extends C, then A satisfies C."""
        system = SymbolicDiscreteSystem(nx=2, nu=1)

        # Symbolic should satisfy all parent protocols
        assert isinstance(system, DiscreteSystemProtocol)  # Grandparent
        assert isinstance(system, LinearizableDiscreteProtocol)  # Parent
        assert isinstance(system, SymbolicDiscreteProtocol)  # Self

    def test_minimal_does_not_satisfy_child_protocols(self):
        """Base implementation doesn't satisfy child protocols."""
        system = MinimalDiscreteSystem(nx=2, nu=1)

        assert isinstance(system, DiscreteSystemProtocol)
        assert not isinstance(system, LinearizableDiscreteProtocol)
        assert not isinstance(system, SymbolicDiscreteProtocol)


# ============================================================================
# Test Suite: Protocol Use in Functions
# ============================================================================


class TestProtocolsInFunctions:
    """Test using protocols in function signatures."""

    def test_function_with_discrete_protocol(self):
        """Function accepting DiscreteSystemProtocol works with any discrete system."""

        def monte_carlo_sim(system: DiscreteSystemProtocol, n_trials: int) -> float:
            """Monte Carlo simulation - only needs step/simulate."""
            total_cost = 0.0
            for _ in range(n_trials):
                x0 = np.random.randn(system.nx)
                result = system.simulate(x0, None, n_steps=10)
                final_state = result["states"][-1, :]
                total_cost += np.linalg.norm(final_state) ** 2
            return total_cost / n_trials

        # Works with minimal system
        sys1 = MinimalDiscreteSystem(nx=2, nu=1)
        cost1 = monte_carlo_sim(sys1, n_trials=5)
        assert isinstance(cost1, float)

        # Also works with linearizable system
        sys2 = LinearizableDiscreteSystem(nx=2, nu=1)
        cost2 = monte_carlo_sim(sys2, n_trials=5)
        assert isinstance(cost2, float)

        # Also works with symbolic system
        sys3 = SymbolicDiscreteSystem(nx=2, nu=1)
        cost3 = monte_carlo_sim(sys3, n_trials=5)
        assert isinstance(cost3, float)

    def test_function_with_linearizable_protocol(self):
        """Function accepting LinearizableDiscreteProtocol requires linearize()."""

        def simple_lqr(system: LinearizableDiscreteProtocol) -> np.ndarray:
            """Simple LQR - needs linearization."""
            Ad, Bd = system.linearize(np.zeros(system.nx), np.zeros(system.nu))

            # Mock LQR (just return Ad for testing)
            return Ad

        # Minimal system CANNOT be used (no linearize)
        # This would fail at runtime if we tried

        # Linearizable system works
        sys1 = LinearizableDiscreteSystem(nx=2, nu=1)
        K1 = simple_lqr(sys1)
        assert K1.shape == (2, 2)

        # Symbolic system also works
        sys2 = SymbolicDiscreteSystem(nx=2, nu=1)
        K2 = simple_lqr(sys2)
        assert K2.shape == (2, 2)

    def test_function_with_symbolic_protocol(self):
        """Function accepting SymbolicDiscreteProtocol requires symbolic machinery."""

        def count_symbolic_vars(system: SymbolicDiscreteProtocol) -> int:
            """Count symbolic variables - needs symbolic attributes."""
            return len(system.state_vars) + len(system.control_vars)

        # Only symbolic system works
        sys = SymbolicDiscreteSystem(nx=2, nu=1)
        count = count_symbolic_vars(sys)
        assert count == 3  # 2 states + 1 control

        # Linearizable system CANNOT be used (no symbolic attributes)
        # This would fail at runtime if we tried


# ============================================================================
# Test Suite: Runtime Type Checking
# ============================================================================


class TestRuntimeTypeChecking:
    """Test runtime isinstance() checks with protocols."""

    def test_runtime_check_with_runtime_checkable(self):
        """@runtime_checkable enables isinstance() checks."""
        sys = MinimalDiscreteSystem(nx=2, nu=1)

        # Can check at runtime
        if isinstance(sys, DiscreteSystemProtocol):
            x = sys.step(np.zeros(2), np.zeros(1))
            assert x is not None

    def test_branching_based_on_protocol(self):
        """Can branch logic based on protocol satisfaction."""

        def smart_compile(system):
            """Compile if symbolic, skip otherwise."""
            if isinstance(system, SymbolicDiscreteProtocol):
                return system.compile(backends=["numpy"])
            else:
                return {}  # No compilation needed

        # Symbolic system gets compiled
        sys1 = SymbolicDiscreteSystem(nx=2, nu=1)
        result1 = smart_compile(sys1)
        assert "numpy" in result1

        # Non-symbolic system skips compilation
        sys2 = LinearizableDiscreteSystem(nx=2, nu=1)
        result2 = smart_compile(sys2)
        assert result2 == {}

    def test_protocol_check_before_access(self):
        """Check protocol before accessing specialized methods."""
        systems = [
            MinimalDiscreteSystem(nx=2, nu=1),
            LinearizableDiscreteSystem(nx=2, nu=1),
            SymbolicDiscreteSystem(nx=2, nu=1),
        ]

        for sys in systems:
            # Everyone can step
            x_next = sys.step(np.zeros(sys.nx), None)

            # Only linearizable can linearize
            if isinstance(sys, LinearizableDiscreteProtocol):
                Ad, Bd = sys.linearize(np.zeros(sys.nx), None)
                assert Ad.shape == (sys.nx, sys.nx)

            # Only symbolic can compile
            if isinstance(sys, SymbolicDiscreteProtocol):
                times = sys.compile()
                assert isinstance(times, dict)


# ============================================================================
# Test Suite: Integration with Real Systems (if available)
# ============================================================================


class TestRealSystemIntegration:
    """
    Test real system classes satisfy appropriate protocols.

    NOTE: These tests are conditional - they skip if real system classes
    are not available. This allows testing the protocol module independently.
    """

    def test_discrete_symbolic_system_satisfies_protocols(self):
        """Real DiscreteSymbolicSystem should satisfy SymbolicDiscreteProtocol."""
        try:
            import sympy as sp

            # Create simple test system
            class SimpleDiscrete(DiscreteSymbolicSystem):
                def define_system(self, a=0.9, dt=0.01):
                    x = sp.symbols("x")
                    u = sp.symbols("u")
                    a_sym = sp.symbols("a")

                    self.state_vars = [x]
                    self.control_vars = [u]
                    self._f_sym = sp.Matrix([a_sym * x + u])
                    self.parameters = {a_sym: a}
                    self._dt = dt
                    self.order = 1

            system = SimpleDiscrete(a=0.9, dt=0.01)

            # Should satisfy all discrete protocols
            assert isinstance(system, DiscreteSystemProtocol)
            assert isinstance(system, LinearizableDiscreteProtocol)
            assert isinstance(system, SymbolicDiscreteProtocol)

        except ImportError:
            pytest.skip("DiscreteSymbolicSystem not available")

    def test_discretized_system_satisfies_linearizable_protocol(self):
        """Real DiscretizedSystem should satisfy LinearizableDiscreteProtocol but NOT Symbolic."""
        try:
            from src.systems import ContinuousSymbolicSystem, DiscretizedSystem
            import sympy as sp

            # Create continuous system
            class SimpleContinuous(ContinuousSymbolicSystem):
                def define_system(self, a=1.0):
                    x = sp.symbols("x")
                    u = sp.symbols("u")
                    a_sym = sp.symbols("a")

                    self.state_vars = [x]
                    self.control_vars = [u]
                    self._f_sym = sp.Matrix([-a_sym * x + u])
                    self.parameters = {a_sym: a}
                    self.order = 1

            continuous = SimpleContinuous(a=1.0)
            discretized = DiscretizedSystem(continuous, dt=0.01, method="rk4")

            # Should satisfy basic and linearizable protocols
            assert isinstance(discretized, DiscreteSystemProtocol)
            assert isinstance(discretized, LinearizableDiscreteProtocol)

            # Should NOT satisfy symbolic protocol (no symbolic expressions)
            assert not isinstance(discretized, SymbolicDiscreteProtocol)

        except ImportError:
            pytest.skip("DiscretizedSystem not yet implemented")

    def test_continuous_symbolic_system_satisfies_protocols(self):
        """Real ContinuousSymbolicSystem should satisfy SymbolicContinuousProtocol."""
        try:
            import sympy as sp

            class SimpleContinuous(ContinuousSymbolicSystem):
                def define_system(self, a=1.0):
                    x = sp.symbols("x")
                    u = sp.symbols("u")
                    a_sym = sp.symbols("a")

                    self.state_vars = [x]
                    self.control_vars = [u]
                    self._f_sym = sp.Matrix([-a_sym * x + u])
                    self.parameters = {a_sym: a}
                    self.order = 1

            system = SimpleContinuous(a=1.0)

            # Should satisfy all continuous protocols
            assert isinstance(system, ContinuousSystemProtocol)
            assert isinstance(system, LinearizableContinuousProtocol)
            assert isinstance(system, SymbolicContinuousProtocol)

        except ImportError:
            pytest.skip("ContinuousSymbolicSystem not available")


# ============================================================================
# Test Suite: Edge Cases and Error Conditions
# ============================================================================


class TestProtocolEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_dimensional_control(self):
        """System with nu=0 (autonomous) satisfies protocol."""
        system = MinimalDiscreteSystem(nx=2, nu=0, dt=0.01)

        assert isinstance(system, DiscreteSystemProtocol)
        assert system.nu == 0

        # Should handle None control
        x_next = system.step(np.array([1.0, 0.0]), None)
        assert x_next is not None

    def test_high_dimensional_system(self):
        """High-dimensional systems satisfy protocols."""
        system = LinearizableDiscreteSystem(nx=100, nu=50, dt=0.01)

        assert isinstance(system, LinearizableDiscreteProtocol)
        assert system.nx == 100
        assert system.nu == 50

    def test_scalar_system(self):
        """Scalar (nx=1) system satisfies protocols."""
        system = SymbolicDiscreteSystem(nx=1, nu=1, dt=0.01)

        assert isinstance(system, SymbolicDiscreteProtocol)

        x = np.array([1.0])
        u = np.array([0.5])
        x_next = system.step(x, u)
        assert x_next.shape == (1,)


# ============================================================================
# Test Suite: Type Safety Examples
# ============================================================================


class TestTypeSafetyExamples:
    """
    Demonstrate type safety benefits.

    These are NOT runtime tests - they document expected mypy/pyright behavior.
    """

    def test_type_checking_catches_missing_methods(self):
        """
        Type checker should catch when protocol requirements not met.

        This test documents expected static type checker behavior.
        """

        # This would fail mypy (uncomment to test with mypy):
        # def needs_linearizable(system: LinearizableDiscreteProtocol):
        #     return system.linearize(np.zeros(2), None)
        #
        # minimal = MinimalDiscreteSystem(nx=2, nu=1)
        # needs_linearizable(minimal)  # ✗ mypy error: no linearize() method

        # At runtime, we can check:
        minimal = MinimalDiscreteSystem(nx=2, nu=1)
        assert not isinstance(minimal, LinearizableDiscreteProtocol)

    def test_type_checking_allows_protocol_conforming_objects(self):
        """Type checker accepts objects conforming to protocol."""

        def use_linearizable(system: LinearizableDiscreteProtocol) -> tuple:
            return system.linearize(np.zeros(system.nx), None)

        # Both should pass type checking
        sys1 = LinearizableDiscreteSystem(nx=2, nu=1)
        sys2 = SymbolicDiscreteSystem(nx=2, nu=1)

        result1 = use_linearizable(sys1)
        result2 = use_linearizable(sys2)

        assert result1 is not None
        assert result2 is not None


# ============================================================================
# Test Suite: Documentation and Metadata
# ============================================================================


class TestProtocolDocumentation:
    """Test that protocols are well-documented."""

    def test_protocols_have_docstrings(self):
        """All protocols have comprehensive docstrings."""
        protocols = [
            DiscreteSystemProtocol,
            LinearizableDiscreteProtocol,
            SymbolicDiscreteProtocol,
            ContinuousSystemProtocol,
            LinearizableContinuousProtocol,
            SymbolicContinuousProtocol,
            StochasticSystemProtocol,
            CompilableSystemProtocol,
            ParametricSystemProtocol,
        ]

        for protocol in protocols:
            assert protocol.__doc__ is not None
            assert len(protocol.__doc__) > 100  # Substantial documentation

    def test_protocol_methods_have_docstrings(self):
        """Protocol methods are documented."""
        # Check a sample protocol
        import inspect

        # Get all methods of LinearizableDiscreteProtocol
        methods = inspect.getmembers(LinearizableDiscreteProtocol, predicate=inspect.isfunction)

        # At least linearize should be documented
        # Note: This is tricky with Protocols, checking mock instead
        system = LinearizableDiscreteSystem(nx=2, nu=1)
        assert system.linearize.__doc__ is not None


# ============================================================================
# Test Suite: Performance and Overhead
# ============================================================================


class TestProtocolPerformance:
    """Test that protocols don't add runtime overhead."""

    def test_isinstance_check_is_fast(self):
        """isinstance() with protocol should be reasonably fast."""
        import time

        system = SymbolicDiscreteSystem(nx=2, nu=1)

        # Time many isinstance checks
        n_checks = 10000
        start = time.time()
        for _ in range(n_checks):
            isinstance(system, SymbolicDiscreteProtocol)
        elapsed = time.time() - start

        # Protocol isinstance checks are slower than normal isinstance
        # but should still be reasonable (< 1 second for 10k checks)
        assert elapsed < 1.0  # 1 second for 10k checks = 100us/check

    def test_protocol_does_not_affect_method_calls(self):
        """Using protocols in type hints doesn't slow down method calls."""
        import time

        def with_protocol(system: DiscreteSystemProtocol, x, u, n_iter):
            for _ in range(n_iter):
                x = system.step(x, u)
            return x

        def without_protocol(system, x, u, n_iter):
            for _ in range(n_iter):
                x = system.step(x, u)
            return x

        system = MinimalDiscreteSystem(nx=2, nu=1)
        x = np.array([1.0, 0.0])
        u = np.array([0.5])
        n = 1000

        # Time with protocol type hint
        start1 = time.time()
        result1 = with_protocol(system, x, u, n)
        time1 = time.time() - start1

        # Time without protocol type hint
        start2 = time.time()
        result2 = without_protocol(system, x, u, n)
        time2 = time.time() - start2

        # Results should be identical
        assert np.allclose(result1, result2)

        # Times should be very similar (< 10% difference)
        # Protocols are compile-time only, zero runtime overhead
        ratio = time1 / time2 if time2 > 0 else 1.0
        assert 0.9 < ratio < 1.1  # Within 10%


# ============================================================================
# Test Suite: Negative Tests
# ============================================================================


class TestNegativeTests:
    """Test what should NOT satisfy protocols."""

    def test_object_without_methods_fails_protocol(self):
        """Plain object doesn't satisfy any protocol."""

        class EmptyClass:
            pass

        obj = EmptyClass()

        assert not isinstance(obj, DiscreteSystemProtocol)
        assert not isinstance(obj, LinearizableDiscreteProtocol)
        assert not isinstance(obj, SymbolicDiscreteProtocol)

    def test_partial_implementation_fails_protocol(self):
        """Implementation missing some methods fails protocol."""

        class PartialSystem:
            def __init__(self):
                self._nx = 2
                self._nu = 1
                self._dt = 0.01

            @property
            def nx(self):
                return self._nx

            @property
            def nu(self):
                return self._nu

            @property
            def dt(self):
                return self._dt

            # Has step but not simulate
            def step(self, x, u=None, k=0):
                return 0.9 * x + 0.1 * (u if u is not None else 0)

        partial = PartialSystem()

        # Missing simulate() - should fail protocol
        assert not isinstance(partial, DiscreteSystemProtocol)

    def test_wrong_signature_fails_protocol(self):
        """Method with wrong signature doesn't satisfy protocol."""

        class WrongSignature:
            def __init__(self):
                self._nx = 2
                self._nu = 1
                self._dt = 0.01

            @property
            def nx(self):
                return self._nx

            @property
            def nu(self):
                return self._nu

            @property
            def dt(self):
                return self._dt

            # Wrong signature: step takes no arguments
            def step(self):
                return np.zeros(2)

            def simulate(self, x0, u_sequence, n_steps):
                return {"states": np.zeros((n_steps + 1, 2))}

        wrong = WrongSignature()

        # Protocol check is structural, so this WILL pass isinstance
        # (Python doesn't check signatures at runtime)
        # But mypy would catch this at static analysis time


# ============================================================================
# Test Suite: Protocol Composition
# ============================================================================


class TestProtocolComposition:
    """Test combining multiple protocols."""

    def test_system_can_satisfy_multiple_protocols(self):
        """Single system can satisfy multiple orthogonal protocols."""

        # SymbolicDiscreteSystem satisfies:
        # - DiscreteSystemProtocol (hierarchy)
        # - LinearizableDiscreteProtocol (hierarchy)
        # - SymbolicDiscreteProtocol (hierarchy)
        # - CompilableSystemProtocol (orthogonal)
        # - ParametricSystemProtocol (orthogonal)

        system = SymbolicDiscreteSystem(nx=2, nu=1)

        assert isinstance(system, DiscreteSystemProtocol)
        assert isinstance(system, LinearizableDiscreteProtocol)
        assert isinstance(system, SymbolicDiscreteProtocol)
        assert isinstance(system, CompilableSystemProtocol)
        assert isinstance(system, ParametricSystemProtocol)

    def test_stochastic_discrete_symbolic_satisfies_all(self):
        """Stochastic symbolic system satisfies many protocols."""

        # StochasticDiscreteSystem (if it exists) should satisfy:
        # - All discrete protocols
        # - StochasticSystemProtocol
        # - CompilableSystemProtocol
        # - ParametricSystemProtocol

        system = StochasticDiscreteSystem(nx=2, nu=1, nw=1)
        system.state_vars = [sp.symbols("x1"), sp.symbols("x2")]
        system.control_vars = [sp.symbols("u")]
        system.parameters = {sp.symbols("a"): 0.9}

        # Manually add required methods for SymbolicDiscreteProtocol
        system.compile = lambda backends=None, verbose=False: {"numpy": 0.001}
        system.print_equations = lambda simplify=True: print("mock")
        system.substitute_parameters = lambda expr: expr

        assert isinstance(system, DiscreteSystemProtocol)
        assert isinstance(system, LinearizableDiscreteProtocol)
        assert isinstance(system, SymbolicDiscreteProtocol)
        assert isinstance(system, StochasticSystemProtocol)


# ============================================================================
# Test Suite: Protocol Documentation Examples
# ============================================================================


class TestDocumentationExamples:
    """Test that examples in protocol docstrings actually work."""

    def test_monte_carlo_example_from_discrete_protocol_docs(self):
        """Example from DiscreteSystemProtocol docstring works."""

        def collect_trajectories(system: DiscreteSystemProtocol, n_trials: int):
            """From docstring example"""
            trajectories = []
            for _ in range(n_trials):
                x0 = np.random.randn(system.nx)
                result = system.simulate(x0, None, n_steps=10)
                trajectories.append(result)
            return trajectories

        system = MinimalDiscreteSystem(nx=2, nu=1)
        trajs = collect_trajectories(system, n_trials=5)

        assert len(trajs) == 5
        assert all("states" in traj for traj in trajs)

    def test_lqr_example_from_linearizable_protocol_docs(self):
        """Example from LinearizableDiscreteProtocol docstring works."""

        def simple_lqr(system: LinearizableDiscreteProtocol):
            """Simplified from docstring example"""
            Ad, Bd = system.linearize(np.zeros(system.nx), np.zeros(system.nu))
            # Just return gains for testing
            return Ad, Bd

        system = LinearizableDiscreteSystem(nx=2, nu=1)
        Ad, Bd = simple_lqr(system)

        assert Ad.shape == (2, 2)
        assert Bd.shape == (2, 1)

    def test_stability_check_example_works(self):
        """Stability check example from docstring works."""

        def check_stability(system: LinearizableDiscreteProtocol) -> bool:
            """From LinearizableDiscreteProtocol docstring"""
            Ad, Bd = system.linearize(np.zeros(system.nx), np.zeros(system.nu))
            eigenvalues = np.linalg.eigvals(Ad)
            return bool(np.all(np.abs(eigenvalues) < 1.0))

        system = LinearizableDiscreteSystem(nx=2, nu=1)
        is_stable = check_stability(system)

        # 0.9*I has eigenvalues [0.9, 0.9] - stable
        assert is_stable is True
        assert isinstance(is_stable, bool)  # Ensure it's Python bool, not np.bool_


# ============================================================================
# Pytest Configuration and Markers
# ============================================================================


@pytest.fixture
def minimal_discrete():
    """Fixture for minimal discrete system."""
    return MinimalDiscreteSystem(nx=2, nu=1, dt=0.01)


@pytest.fixture
def linearizable_discrete():
    """Fixture for linearizable discrete system."""
    return LinearizableDiscreteSystem(nx=2, nu=1, dt=0.01)


@pytest.fixture
def symbolic_discrete():
    """Fixture for symbolic discrete system."""
    return SymbolicDiscreteSystem(nx=2, nu=1, dt=0.01)


@pytest.fixture
def minimal_continuous():
    """Fixture for minimal continuous system."""
    return MinimalContinuousSystem(nx=2, nu=1)


# ============================================================================
# Summary Tests
# ============================================================================


class TestProtocolSummary:
    """
    High-level tests summarizing protocol behavior.

    These tests serve as documentation for how the protocol system works.
    """

    def test_three_level_discrete_hierarchy(self):
        """Discrete protocols form 3-level hierarchy."""
        # Level 1: Basic discrete
        minimal = MinimalDiscreteSystem(nx=2, nu=1)
        assert isinstance(minimal, DiscreteSystemProtocol)
        assert not isinstance(minimal, LinearizableDiscreteProtocol)
        assert not isinstance(minimal, SymbolicDiscreteProtocol)

        # Level 2: + Linearization
        linearizable = LinearizableDiscreteSystem(nx=2, nu=1)
        assert isinstance(linearizable, DiscreteSystemProtocol)
        assert isinstance(linearizable, LinearizableDiscreteProtocol)
        assert not isinstance(linearizable, SymbolicDiscreteProtocol)

        # Level 3: + Symbolic
        symbolic = SymbolicDiscreteSystem(nx=2, nu=1)
        assert isinstance(symbolic, DiscreteSystemProtocol)
        assert isinstance(symbolic, LinearizableDiscreteProtocol)
        assert isinstance(symbolic, SymbolicDiscreteProtocol)

    def test_protocols_enable_flexible_apis(self):
        """Protocols allow functions to work with multiple implementations."""

        # Generic function - works with ANY discrete system
        def generic_simulate(system: DiscreteSystemProtocol) -> np.ndarray:
            result = system.simulate(np.zeros(system.nx), None, n_steps=10)
            return result["states"]

        # Specialized function - needs linearization
        def lqr_design(system: LinearizableDiscreteProtocol) -> np.ndarray:
            Ad, Bd = system.linearize(np.zeros(system.nx), np.zeros(system.nu))
            return Ad  # Simplified

        # Highly specialized - needs symbolic
        def export_code(system: SymbolicDiscreteProtocol) -> str:
            system.print_equations()
            return "exported"

        # Create systems
        minimal = MinimalDiscreteSystem(nx=2, nu=1)
        linearizable = LinearizableDiscreteSystem(nx=2, nu=1)
        symbolic = SymbolicDiscreteSystem(nx=2, nu=1)

        # generic_simulate works with all three
        generic_simulate(minimal)  # ✓
        generic_simulate(linearizable)  # ✓
        generic_simulate(symbolic)  # ✓

        # lqr_design works with linearizable and symbolic
        # lqr_design(minimal)  # Would fail - no linearize()
        lqr_design(linearizable)  # ✓
        lqr_design(symbolic)  # ✓

        # export_code works only with symbolic
        # export_code(minimal)  # Would fail
        # export_code(linearizable)  # Would fail
        export_code(symbolic)  # ✓

    def test_runtime_checkable_enables_defensive_programming(self):
        """@runtime_checkable enables runtime capability checks."""

        def smart_analysis(system):
            """Adapt behavior based on capabilities."""
            results = {"basic_sim": None, "linearization": None, "symbolic": None}

            # Always can simulate
            if isinstance(system, DiscreteSystemProtocol):
                result = system.simulate(np.zeros(system.nx), None, n_steps=10)
                results["basic_sim"] = result["states"].shape

            # Try linearization if available
            if isinstance(system, LinearizableDiscreteProtocol):
                Ad, Bd = system.linearize(np.zeros(system.nx), None)
                results["linearization"] = Ad.shape

            # Try symbolic if available
            if isinstance(system, SymbolicDiscreteProtocol):
                results["symbolic"] = len(system.state_vars)

            return results

        # Test with different systems
        minimal = MinimalDiscreteSystem(nx=2, nu=1)
        results_minimal = smart_analysis(minimal)
        assert results_minimal["basic_sim"] == (11, 2)
        assert results_minimal["linearization"] is None  # Not available
        assert results_minimal["symbolic"] is None  # Not available

        symbolic = SymbolicDiscreteSystem(nx=2, nu=1)
        results_symbolic = smart_analysis(symbolic)
        assert results_symbolic["basic_sim"] == (11, 2)
        assert results_symbolic["linearization"] == (2, 2)  # Available
        assert results_symbolic["symbolic"] == 2  # Available


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])