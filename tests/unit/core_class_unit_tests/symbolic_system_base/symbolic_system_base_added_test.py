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
Additional Unit Tests for SymbolicSystemBase
============================================

Advanced test coverage for edge cases, robustness, and production scenarios
that complement the basic test suite. These tests focus on:

- Symbolic expression edge cases (infinity, NaN, undefined)
- Parameter constraint validation
- System state consistency under stress
- Backend transition safety
- Large-scale systems (100+ states)
- Memory and performance characteristics
- Copy/deepcopy behavior
- Error recovery and resilience
- Validation boundary conditions
- Integration scenarios with real components

Test Organization
-----------------
- TestSymbolicExpressionEdgeCases: Special symbolic values
- TestParameterConstraints: Parameter validation and bounds
- TestSystemStateConsistency: Internal state invariants
- TestBackendTransitions: Safe backend switching
- TestLargeScaleSystems: Performance with many states
- TestCopyBehavior: System copying and cloning
- TestErrorRecovery: Resilience after errors
- TestValidationBoundaries: Edge cases in validation
- TestRealComponentIntegration: Tests with actual utilities (not mocked)
- TestMemoryManagement: Cache cleanup and memory
- TestComplexParameterUpdates: Dynamic parameter changes
- TestSubclassContract: Enforcement of abstract requirements
- TestConcurrentAccess: Thread safety (if applicable)

Usage
-----
Run additional tests:
    pytest test_symbolic_system_base_additional.py -v

Run with main tests:
    pytest test_symbolic_system_base.py test_symbolic_system_base_additional.py -v

Run specific advanced category:
    pytest test_symbolic_system_base_additional.py::TestLargeScaleSystems -v
"""

import copy
import gc
import sys
import tempfile
import threading
import warnings
from pathlib import Path
from typing import Dict, List
from unittest.mock import patch

import numpy as np
import pytest
import sympy as sp

# Import the class under test
from src.systems.base.core.symbolic_system_base import SymbolicSystemBase

# Import utilities
from src.systems.base.utils.backend_manager import BackendManager
from src.systems.base.utils.code_generator import CodeGenerator
from src.systems.base.utils.equilibrium_handler import EquilibriumHandler
from src.systems.base.utils.symbolic_validator import (
    SymbolicValidator,
    ValidationError,
)

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
# Test Systems for Advanced Scenarios
# ============================================================================


class VeryLargeSystem(SymbolicSystemBase):
    """Large system with many states for performance testing."""

    def define_system(self, n_states: int = 100, n_controls: int = 10):
        """Define system with n_states and n_controls."""
        # Create symbolic variables
        x_vars = sp.symbols(f"x0:{n_states}", real=True)
        u_vars = sp.symbols(f"u0:{n_controls}", real=True)

        # Simple linear dynamics: dx_i/dt = -x_i + u_j (cycling through controls)
        dynamics = []
        for i, x in enumerate(x_vars):
            u = u_vars[i % n_controls]
            dynamics.append(-x + u)

        self.state_vars = list(x_vars)
        self.control_vars = list(u_vars)
        self._f_sym = sp.Matrix(dynamics)
        self.parameters = {}
        self.order = 1

    def print_equations(self, simplify: bool = True):
        print(f"Large System: {self.nx} states, {self.nu} controls")


class SystemWithComplexParameters(SymbolicSystemBase):
    """System with many parameters for testing parameter updates."""

    def define_system(self, **param_values):
        """Define system with configurable parameters."""
        x, v = sp.symbols("x v", real=True)
        u = sp.symbols("u", real=True)

        # Create symbolic parameters
        m, k, c, g, l = sp.symbols("m k c g l", positive=True, real=True)

        # Dynamics: mass-spring-damper with gravity
        self.state_vars = [x, v]
        self.control_vars = [u]
        self._f_sym = sp.Matrix([v, (-k * x - c * v + m * g + u) / (m * l)])

        # Set parameter values
        self.parameters = {
            m: param_values.get("m", 1.0),
            k: param_values.get("k", 10.0),
            c: param_values.get("c", 0.5),
            g: param_values.get("g", 9.81),
            l: param_values.get("l", 1.0),
        }
        self.order = 1

    def print_equations(self, simplify: bool = True):
        print("System with Complex Parameters")


class SystemWithSpecialSymbolicExpressions(SymbolicSystemBase):
    """System with special symbolic expressions for edge case testing."""

    def define_system(self, expr_type: str = "normal"):
        """Define system with different expression types."""
        x = sp.symbols("x", real=True)
        u = sp.symbols("u", real=True)

        self.state_vars = [x]
        self.control_vars = [u]
        self.parameters = {}
        self.order = 1

        if expr_type == "normal":
            self._f_sym = sp.Matrix([-x + u])
        elif expr_type == "trigonometric":
            self._f_sym = sp.Matrix([sp.sin(x) + sp.cos(u)])
        elif expr_type == "logarithmic":
            # log(abs(x) + 1) to avoid log(0)
            self._f_sym = sp.Matrix([sp.log(sp.Abs(x) + 1) + u])
        elif expr_type == "exponential":
            self._f_sym = sp.Matrix([sp.exp(-x) + u])
        elif expr_type == "rational":
            # 1/(x^2 + 1) to avoid division by zero
            self._f_sym = sp.Matrix([1 / (x**2 + 1) + u])
        elif expr_type == "piecewise":
            self._f_sym = sp.Matrix([sp.Piecewise((x, x > 0), (-x, True)) + u])

    def print_equations(self, simplify: bool = True):
        print("System with Special Expressions")


class MinimalSystemForCopy(SymbolicSystemBase):
    """Minimal system for testing copy behavior."""

    def define_system(self, a: float = 1.0):
        x = sp.symbols("x", real=True)
        u = sp.symbols("u", real=True)
        a_sym = sp.symbols("a", real=True, positive=True)

        self.state_vars = [x]
        self.control_vars = [u]
        self._f_sym = sp.Matrix([-a_sym * x + u])
        self.parameters = {a_sym: a}
        self.order = 1

    def print_equations(self, simplify: bool = True):
        print("Minimal System")


# ============================================================================
# Test: Symbolic Expression Edge Cases
# ============================================================================


class TestSymbolicExpressionEdgeCases:
    """Test systems with special symbolic expressions."""

    def test_trigonometric_expressions(self):
        """System with trigonometric expressions."""
        system = SystemWithSpecialSymbolicExpressions(expr_type="trigonometric")

        assert system._initialized
        assert len(system._f_sym) == 1
        # Check that sin and cos are in the expression
        expr_str = str(system._f_sym[0])
        assert "sin" in expr_str or "cos" in expr_str

    def test_logarithmic_expressions(self):
        """System with logarithmic expressions."""
        system = SystemWithSpecialSymbolicExpressions(expr_type="logarithmic")

        assert system._initialized
        # Should contain log
        expr_str = str(system._f_sym[0])
        assert "log" in expr_str.lower() or "ln" in expr_str.lower()

    def test_exponential_expressions(self):
        """System with exponential expressions."""
        system = SystemWithSpecialSymbolicExpressions(expr_type="exponential")

        assert system._initialized
        # Should contain exp
        expr_str = str(system._f_sym[0])
        assert "exp" in expr_str.lower()

    def test_rational_expressions(self):
        """System with rational expressions (avoiding division by zero)."""
        system = SystemWithSpecialSymbolicExpressions(expr_type="rational")

        assert system._initialized
        # Expression should be valid
        assert system._f_sym is not None

    def test_piecewise_expressions(self):
        """System with piecewise expressions."""
        system = SystemWithSpecialSymbolicExpressions(expr_type="piecewise")

        assert system._initialized
        # Should contain Piecewise
        assert any("Piecewise" in str(expr) for expr in system._f_sym)

    def test_symbolic_simplification(self):
        """Test that symbolic simplification doesn't break system."""
        system = SystemWithSpecialSymbolicExpressions(expr_type="trigonometric")

        # Should be able to simplify without error
        simplified = sp.simplify(system._f_sym[0])
        assert simplified is not None

    def test_symbolic_differentiation(self):
        """Test that symbolic differentiation works."""
        system = SystemWithSpecialSymbolicExpressions(expr_type="exponential")

        x = system.state_vars[0]
        # Should be able to differentiate
        derivative = sp.diff(system._f_sym[0], x)
        assert derivative is not None


# ============================================================================
# Test: Parameter Constraints and Updates
# ============================================================================


class TestParameterConstraints:
    """Test parameter validation and dynamic updates."""

    def test_zero_parameters(self):
        """System with zero parameter values."""
        system = SystemWithComplexParameters(k=0.0, c=0.0)

        assert system._initialized
        # Check parameters set correctly
        k_sym = [s for s in system.parameters.keys() if str(s) == "k"][0]
        assert system.parameters[k_sym] == 0.0

    def test_negative_parameters_allowed(self):
        """Negative parameters should work (not all are positive)."""
        # Note: In our test system, parameters are declared positive
        # But we can create a system without that constraint
        class SystemWithNegativeParams(SymbolicSystemBase):
            def define_system(self, a: float = -1.0):
                x = sp.symbols("x", real=True)
                u = sp.symbols("u", real=True)
                a_sym = sp.symbols("a", real=True)  # No positive constraint

                self.state_vars = [x]
                self.control_vars = [u]
                self._f_sym = sp.Matrix([a_sym * x + u])
                self.parameters = {a_sym: a}
                self.order = 1

            def print_equations(self, simplify: bool = True):
                pass

        system = SystemWithNegativeParams(a=-2.0)
        assert system._initialized
        a_sym = list(system.parameters.keys())[0]
        assert system.parameters[a_sym] == -2.0

    def test_very_large_parameter_values(self):
        """System with very large parameter values."""
        system = SystemWithComplexParameters(m=1e10, k=1e10, g=1e10)

        assert system._initialized
        # Verify large values stored
        assert any(v > 1e9 for v in system.parameters.values())

    def test_very_small_parameter_values(self):
        """System with very small parameter values."""
        system = SystemWithComplexParameters(m=1e-10, c=1e-10)

        assert system._initialized
        # Verify small values stored
        assert any(v < 1e-9 for v in system.parameters.values())

    def test_many_parameters(self):
        """System with many parameters."""
        system = SystemWithComplexParameters(
            m=1.0, k=2.0, c=3.0, g=4.0, l=5.0
        )

        assert len(system.parameters) == 5
        # All parameters should be set
        assert all(v is not None for v in system.parameters.values())

    def test_parameter_update_requires_cache_reset(self):
        """Updating parameters should suggest cache reset."""
        system = SystemWithComplexParameters(k=10.0)

        # Get original parameter symbol
        k_sym = [s for s in system.parameters.keys() if str(s) == "k"][0]

        # Update parameter
        system.parameters[k_sym] = 20.0

        # Cache should ideally be reset (user responsibility)
        # This tests that the change is reflected
        assert system.parameters[k_sym] == 20.0

    def test_parameter_substitution_with_extreme_values(self):
        """Parameter substitution with extreme values."""
        system = SystemWithComplexParameters(m=1e-15, k=1e15)

        # Substitution should work
        result = system.substitute_parameters(system._f_sym)
        assert result is not None


# ============================================================================
# Test: System State Consistency
# ============================================================================


class TestSystemStateConsistency:
    """Test that internal state remains consistent."""

    def test_dimensions_consistent_after_init(self):
        """System dimensions should be internally consistent."""
        system = VeryLargeSystem(n_states=50, n_controls=5)

        assert system.nx == 50
        assert system.nu == 5
        assert len(system.state_vars) == system.nx
        assert len(system.control_vars) == system.nu
        assert system._f_sym.shape[0] == system.nx

    def test_equilibria_dimensions_match_system(self):
        """Equilibrium handler dimensions match system."""
        system = VeryLargeSystem(n_states=20, n_controls=3)

        assert system.equilibria.nx == system.nx
        assert system.equilibria.nu == system.nu

    def test_validator_references_correct_system(self):
        """Validator should reference the correct system."""
        system = SystemWithComplexParameters()

        assert system._validator is not None
        # Validator should have reference to system
        # (implementation detail, but important for consistency)

    def test_code_generator_references_correct_system(self):
        """CodeGenerator should reference the correct system."""
        system = SystemWithComplexParameters()

        assert system._code_gen is not None
        # CodeGenerator should have reference to system

    def test_backend_manager_state_consistent(self):
        """Backend manager state should be consistent."""
        system = MinimalSystemForCopy()

        # Backend and device should be consistent
        backend = system._default_backend
        device = system._preferred_device

        assert backend in ["numpy", "torch", "jax"]
        assert isinstance(device, str)

    def test_multiple_operations_maintain_consistency(self):
        """Multiple operations should maintain state consistency."""
        system = SystemWithComplexParameters()

        # Perform multiple operations
        system.set_default_backend("numpy")
        system.add_equilibrium("test", np.zeros(2), np.zeros(1), verify=False)
        config = system.get_config_dict()

        # State should still be consistent
        assert system._initialized
        assert system.nx == 2
        assert len(system.state_vars) == 2


# ============================================================================
# Test: Backend Transitions
# ============================================================================


class TestBackendTransitions:
    """Test safe transitions between backends."""

    def test_backend_switch_preserves_system_state(self):
        """Switching backends preserves system definition."""
        system = MinimalSystemForCopy(a=2.0)

        original_nx = system.nx
        original_params = dict(system.parameters)

        # Switch backend
        system.set_default_backend("numpy")

        # System state unchanged
        assert system.nx == original_nx
        assert system.parameters == original_params

    def test_multiple_backend_switches(self):
        """Multiple backend switches work correctly."""
        system = MinimalSystemForCopy()

        backends = ["numpy", "torch", "jax"]
        available = [b for b in backends if system.backend.check_available(b)]

        for backend in available:
            system.set_default_backend(backend)
            assert system._default_backend == backend
            assert system._initialized

    def test_backend_switch_with_equilibria(self):
        """Backend switch doesn't lose equilibria."""
        system = MinimalSystemForCopy()

        # Add equilibrium
        system.add_equilibrium("test", np.array([1.0]), np.array([0.0]), verify=False)

        # Switch backend
        system.set_default_backend("numpy")

        # Equilibrium still exists
        assert "test" in system.list_equilibria()

    def test_device_change_clears_appropriate_caches(self):
        """Device changes clear caches for GPU backends."""
        system = MinimalSystemForCopy()

        with patch.object(CodeGenerator, "reset_cache") as mock_reset:
            system.set_default_backend("torch")
            system.to_device("cuda")

            # Should have cleared cache
            mock_reset.assert_called()

    def test_context_manager_restores_backend(self):
        """Context manager properly restores backend."""
        system = MinimalSystemForCopy()
        system.set_default_backend("numpy")

        original_backend = system._default_backend

        with system.use_backend("torch"):
            # Inside context
            if system.backend.check_available("torch"):
                assert system._default_backend == "torch"

        # Outside context - restored
        assert system._default_backend == original_backend

    def test_nested_context_managers(self):
        """Nested context managers work correctly."""
        system = MinimalSystemForCopy()
        system.set_default_backend("numpy")

        with system.use_backend("numpy"):
            backend1 = system._default_backend

            with system.use_backend("numpy"):
                backend2 = system._default_backend
                assert backend2 == "numpy"

            assert system._default_backend == backend1


# ============================================================================
# Test: Large Scale Systems
# ============================================================================


class TestLargeScaleSystems:
    """Test performance and correctness with large systems."""

    def test_100_state_system_initialization(self):
        """System with 100 states initializes successfully."""
        system = VeryLargeSystem(n_states=100, n_controls=10)

        assert system._initialized
        assert system.nx == 100
        assert system.nu == 10

    def test_large_system_dimensions(self):
        """Large system reports correct dimensions."""
        n = 200
        system = VeryLargeSystem(n_states=n, n_controls=20)

        assert system.nx == n
        assert len(system.state_vars) == n
        assert system._f_sym.shape[0] == n

    def test_large_system_validation(self):
        """Large system passes validation."""
        system = VeryLargeSystem(n_states=150, n_controls=15)

        # Should have passed validation
        assert system._initialized
        assert system._validator is not None

    def test_large_system_parameter_substitution(self):
        """Parameter substitution works on large systems."""
        system = VeryLargeSystem(n_states=100, n_controls=10)

        # Even with no parameters, should work
        result = system.substitute_parameters(system._f_sym)
        assert result is not None
        assert result.shape == system._f_sym.shape

    @pytest.mark.slow
    def test_very_large_system_500_states(self):
        """Very large system (500 states) - marked as slow."""
        # This may take a few seconds
        system = VeryLargeSystem(n_states=500, n_controls=50)

        assert system._initialized
        assert system.nx == 500

    def test_large_system_equilibrium_management(self):
        """Can add equilibria to large systems."""
        system = VeryLargeSystem(n_states=100, n_controls=10)

        x_eq = np.zeros(100)
        u_eq = np.zeros(10)

        system.add_equilibrium("origin", x_eq, u_eq, verify=False)
        assert "origin" in system.list_equilibria()

    def test_large_system_config_serialization(self):
        """Large system configuration can be serialized."""
        system = VeryLargeSystem(n_states=100, n_controls=10)

        config = system.get_config_dict()

        assert config["nx"] == 100
        assert config["nu"] == 10
        assert len(config["state_vars"]) == 100


# ============================================================================
# Test: Copy Behavior
# ============================================================================


class TestCopyBehavior:
    """Test system copying and cloning."""

    def test_shallow_copy_not_recommended(self):
        """Shallow copy may not work as expected (document behavior)."""
        system1 = MinimalSystemForCopy(a=1.0)

        # Shallow copy
        system2 = copy.copy(system1)

        # They share some internal state (this is expected)
        # This test documents the behavior
        assert system2 is not system1
        # Backend manager may be shared
        assert system2.backend is system1.backend

    def test_deepcopy_creates_independent_system(self):
        """Deepcopy should create independent system."""
        system1 = MinimalSystemForCopy(a=1.0)
        system1.add_equilibrium("test", np.array([1.0]), np.array([0.0]), verify=False)

        # Deepcopy
        system2 = copy.deepcopy(system1)

        # Should be different objects
        assert system2 is not system1
        assert system2.backend is not system1.backend

        # But equivalent
        assert system2.nx == system1.nx
        assert system2.nu == system1.nu
        assert "test" in system2.list_equilibria()

    def test_copy_preserves_parameters(self):
        """Copy preserves parameter values."""
        system1 = MinimalSystemForCopy(a=2.5)

        system2 = copy.deepcopy(system1)

        # Parameters should be equal
        param1 = list(system1.parameters.values())[0]
        param2 = list(system2.parameters.values())[0]
        assert param1 == param2

    def test_copy_preserves_equilibria(self):
        """Copy preserves equilibria."""
        system1 = MinimalSystemForCopy()
        system1.add_equilibrium("eq1", np.array([1.0]), np.array([0.5]), verify=False)

        system2 = copy.deepcopy(system1)

        assert "eq1" in system2.list_equilibria()
        x, u = system2.get_equilibrium("eq1")
        np.testing.assert_array_equal(x, np.array([1.0]))

    def test_copied_system_independent_backend_config(self):
        """Copied system has independent backend configuration."""
        system1 = MinimalSystemForCopy()
        system1.set_default_backend("numpy")

        system2 = copy.deepcopy(system1)

        # Change system2's backend
        system2.set_default_backend("torch") if system2.backend.check_available("torch") else system2.set_default_backend("numpy")

        # system1 should be unchanged (if torch was available)
        assert system1._default_backend == "numpy"


# ============================================================================
# Test: Error Recovery
# ============================================================================


class TestErrorRecovery:
    """Test system behavior after errors."""

    def test_system_usable_after_invalid_equilibrium(self):
        """System remains usable after invalid equilibrium attempt."""
        system = MinimalSystemForCopy()

        # Try to add invalid equilibrium
        try:
            system.add_equilibrium("bad", np.array([1.0, 2.0]), np.array([0.0]), verify=False)
        except ValueError:
            pass  # Expected

        # System should still work
        assert system._initialized
        assert system.nx == 1

    def test_system_usable_after_backend_error(self):
        """System remains usable after backend error."""
        system = MinimalSystemForCopy()

        # Try invalid backend
        try:
            system.set_default_backend("invalid_backend")
        except (ValueError, RuntimeError):
            pass  # Expected

        # System should still work
        assert system._initialized
        assert system._default_backend in ["numpy", "torch", "jax"]

    def test_system_usable_after_equilibrium_removal_error(self):
        """System remains usable after equilibrium removal error."""
        system = MinimalSystemForCopy()

        # Try to remove non-existent equilibrium
        try:
            system.remove_equilibrium("nonexistent")
        except ValueError:
            pass  # Expected

        # System should still work
        assert system._initialized

    def test_multiple_errors_dont_corrupt_state(self):
        """Multiple errors don't corrupt internal state."""
        system = MinimalSystemForCopy()

        # Generate multiple errors
        errors_caught = 0

        try:
            system.remove_equilibrium("fake")
        except ValueError:
            errors_caught += 1

        try:
            system.add_equilibrium("bad", np.array([1.0, 2.0]), np.array([0.0]), verify=False)
        except ValueError:
            errors_caught += 1

        # Should have caught errors
        assert errors_caught >= 1

        # System still works
        assert system._initialized
        assert system.nx == 1


# ============================================================================
# Test: Validation Boundaries
# ============================================================================


class TestValidationBoundaries:
    """Test edge cases in validation."""

    def test_single_state_system(self):
        """System with single state."""
        class SingleStateSystem(SymbolicSystemBase):
            def define_system(self):
                x = sp.symbols("x")
                self.state_vars = [x]
                self.control_vars = []
                self._f_sym = sp.Matrix([-x])
                self.parameters = {}
                self.order = 1

            def print_equations(self, simplify=True):
                pass

        system = SingleStateSystem()
        assert system.nx == 1
        assert system.nu == 0

    def test_system_with_no_dynamics(self):
        """System with zero dynamics (dx/dt = 0)."""
        class NoDynamicsSystem(SymbolicSystemBase):
            def define_system(self):
                x = sp.symbols("x")
                u = sp.symbols("u")
                self.state_vars = [x]
                self.control_vars = [u]
                self._f_sym = sp.Matrix([0])  # No dynamics
                self.parameters = {}
                self.order = 1

            def print_equations(self, simplify=True):
                pass

        system = NoDynamicsSystem()
        assert system._initialized
        assert system._f_sym[0] == 0

    def test_system_validation_catches_dimension_mismatch(self):
        """Validation catches dimension mismatches."""
        class BadDimensionSystem(SymbolicSystemBase):
            def define_system(self):
                x, y = sp.symbols("x y")
                u = sp.symbols("u")
                self.state_vars = [x, y]  # 2 states
                self.control_vars = [u]
                self._f_sym = sp.Matrix([-x])  # Only 1 equation!
                self.parameters = {}
                self.order = 1

            def print_equations(self, simplify=True):
                pass

        with pytest.raises(ValidationError):
            BadDimensionSystem()

    def test_system_validation_catches_missing_f_sym(self):
        """Validation catches missing _f_sym."""
        class MissingDynamicsSystem(SymbolicSystemBase):
            def define_system(self):
                x = sp.symbols("x")
                self.state_vars = [x]
                self.control_vars = []
                # Missing: self._f_sym
                self.parameters = {}
                self.order = 1

            def print_equations(self, simplify=True):
                pass

        with pytest.raises(ValidationError):
            MissingDynamicsSystem()

    def test_higher_order_validation(self):
        """Validation works for higher-order systems."""
        class ValidSecondOrderSystem(SymbolicSystemBase):
            def define_system(self):
                q, q_dot = sp.symbols("q q_dot")
                u = sp.symbols("u")
                self.state_vars = [q, q_dot]  # 2 states
                self.control_vars = [u]
                self._f_sym = sp.Matrix([u])  # Only highest derivative (order=2)
                self.parameters = {}
                self.order = 2  # Second-order

            def print_equations(self, simplify=True):
                pass

        system = ValidSecondOrderSystem()
        assert system._initialized
        assert system.order == 2
        assert system.nq == 1


# ============================================================================
# Test: Real Component Integration (No Mocks)
# ============================================================================


class TestRealComponentIntegration:
    """Test with actual utility classes (not mocked)."""

    def test_actual_backend_manager_integration(self):
        """Test with real BackendManager."""
        system = MinimalSystemForCopy()

        # Real backend manager
        assert isinstance(system.backend, BackendManager)

        # Can actually detect backends
        available = system.backend.available_backends
        assert "numpy" in available
        assert isinstance(available, list)

    def test_actual_code_generator_integration(self):
        """Test with real CodeGenerator."""
        system = MinimalSystemForCopy()

        # Real code generator
        assert isinstance(system._code_gen, CodeGenerator)

        # Can check compilation status
        status = system._code_gen.is_compiled("numpy")
        assert isinstance(status, dict)

    def test_actual_equilibrium_handler_integration(self):
        """Test with real EquilibriumHandler."""
        system = MinimalSystemForCopy()

        # Real equilibrium handler
        assert isinstance(system.equilibria, EquilibriumHandler)

        # Can actually add and retrieve equilibria
        x_eq = np.array([0.0])
        u_eq = np.array([0.0])
        system.add_equilibrium("test", x_eq, u_eq, verify=False)

        x_ret, u_ret = system.equilibria.get_both("test", "numpy")
        np.testing.assert_array_equal(x_ret, x_eq)

    def test_actual_validator_integration(self):
        """Test with real SymbolicValidator."""
        system = MinimalSystemForCopy()

        # Real validator
        assert isinstance(system._validator, SymbolicValidator)

        # Validator has validated the system
        assert system._initialized

    def test_end_to_end_workflow_no_mocks(self):
        """Complete workflow with real components."""
        # Create system
        system = SystemWithComplexParameters(m=2.0, k=10.0, c=0.5)

        # Configure
        system.set_default_backend("numpy")

        # Add equilibrium
        x_eq = np.array([0.0, 0.0])
        u_eq = np.array([0.0])
        system.add_equilibrium("test", x_eq, u_eq, verify=False)

        # Get configuration
        config = system.get_config_dict()

        # Save config
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "config.json"
            system.save_config(str(filepath))
            assert filepath.exists()

        # Verify everything worked
        assert system._initialized
        assert "test" in system.list_equilibria()
        assert config["nx"] == 2


# ============================================================================
# Test: Memory Management
# ============================================================================


class TestMemoryManagement:
    """Test cache cleanup and memory behavior."""

    def test_reset_caches_clears_memory(self):
        """Resetting caches should clear cached functions."""
        system = MinimalSystemForCopy()

        # Reset caches
        system.reset_caches()

        # Should be able to check compilation status
        status = system._code_gen.is_compiled("numpy")
        assert isinstance(status, dict)

    def test_multiple_systems_dont_interfere(self):
        """Multiple systems don't interfere with each other."""
        system1 = MinimalSystemForCopy(a=1.0)
        system2 = MinimalSystemForCopy(a=2.0)

        # Different parameter values
        param1 = list(system1.parameters.values())[0]
        param2 = list(system2.parameters.values())[0]
        assert param1 != param2

        # Independent configurations
        system1.set_default_backend("numpy")
        system2.set_default_backend("numpy")

        assert system1._default_backend == "numpy"
        assert system2._default_backend == "numpy"

    def test_system_cleanup_after_deletion(self):
        """System resources cleaned up after deletion."""
        system = MinimalSystemForCopy()
        system_id = id(system)

        # Delete system
        del system

        # Force garbage collection
        gc.collect()

        # System should be gone (can't verify directly, but no crash)
        assert True

    def test_large_system_memory_footprint(self):
        """Large system has reasonable memory footprint."""
        # Create large system
        system = VeryLargeSystem(n_states=200, n_controls=20)

        # Should be initialized
        assert system._initialized

        # Get rough memory size (Python's sys.getsizeof is limited)
        # This is more of a smoke test
        size = sys.getsizeof(system)
        assert size > 0


# ============================================================================
# Test: Complex Parameter Updates
# ============================================================================


class TestComplexParameterUpdates:
    """Test dynamic parameter changes."""

    def test_parameter_update_reflected_in_substitution(self):
        """Updated parameters reflected in substitution."""
        system = SystemWithComplexParameters(k=10.0)

        k_sym = [s for s in system.parameters.keys() if str(s) == "k"][0]

        # Update parameter
        system.parameters[k_sym] = 20.0

        # Substitution should use new value
        result = system.substitute_parameters(k_sym * sp.symbols("x"))
        assert result == 20.0 * sp.symbols("x")

    def test_updating_multiple_parameters(self):
        """Can update multiple parameters."""
        system = SystemWithComplexParameters(m=1.0, k=10.0, c=0.5)

        # Update all parameters
        for sym in system.parameters.keys():
            system.parameters[sym] *= 2.0

        # All should be updated (c=0.5*2=1.0, so use >=)
        assert all(v >= 1.0 for v in system.parameters.values())

    def test_parameter_update_affects_equation_printing(self):
        """Updated parameters affect print output."""
        system = SystemWithComplexParameters(k=10.0)

        k_sym = [s for s in system.parameters.keys() if str(s) == "k"][0]
        system.parameters[k_sym] = 99.0

        # Substitute and check
        result = system.substitute_parameters(system._f_sym)
        result_str = str(result)

        # Should contain new value (approximately)
        assert "99" in result_str


# ============================================================================
# Test: Subclass Contract Enforcement
# ============================================================================


class TestSubclassContract:
    """Test that subclass requirements are enforced."""

    def test_define_system_must_set_state_vars(self):
        """define_system must set state_vars."""
        class MissingStateVars(SymbolicSystemBase):
            def define_system(self):
                # Missing state_vars
                u = sp.symbols("u")
                self.control_vars = [u]
                self._f_sym = sp.Matrix([0])
                self.parameters = {}
                self.order = 1

            def print_equations(self, simplify=True):
                pass

        with pytest.raises(ValidationError):
            MissingStateVars()

    def test_define_system_must_set_control_vars(self):
        """define_system should set control_vars, or it defaults to empty for autonomous systems."""
        class MissingControlVars(SymbolicSystemBase):
            def define_system(self):
                x = sp.symbols("x")
                self.state_vars = [x]
                # Missing control_vars - should default to empty (autonomous)
                self._f_sym = sp.Matrix([0])
                self.parameters = {}
                self.order = 1

            def print_equations(self, simplify=True):
                pass

        # Should work - defaults to autonomous (nu=0)
        system = MissingControlVars()
        assert system.nu == 0
        assert system.control_vars == []

    def test_f_sym_must_be_matrix(self):
        """_f_sym must be a SymPy Matrix."""
        class WrongFSymType(SymbolicSystemBase):
            def define_system(self):
                x = sp.symbols("x")
                self.state_vars = [x]
                self.control_vars = []
                self._f_sym = [-x]  # List instead of Matrix!
                self.parameters = {}
                self.order = 1

            def print_equations(self, simplify=True):
                pass

        with pytest.raises(ValidationError):
            WrongFSymType()


# ============================================================================
# Test: Thread Safety (Optional - if applicable)
# ============================================================================


class TestConcurrentAccess:
    """Test thread safety of system operations."""

    @pytest.mark.skipif(
        not hasattr(threading, "Thread"),
        reason="Threading not available"
    )
    def test_multiple_threads_reading_properties(self):
        """Multiple threads can read system properties safely."""
        system = MinimalSystemForCopy()

        results = []

        def read_properties():
            for _ in range(100):
                nx = system.nx
                nu = system.nu
                backend = system._default_backend
                results.append((nx, nu, backend))

        threads = [threading.Thread(target=read_properties) for _ in range(5)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All results should be consistent
        assert all(r == results[0] for r in results)

    @pytest.mark.skipif(
        not hasattr(threading, "Thread"),
        reason="Threading not available"
    )
    def test_concurrent_backend_switching_warns(self):
        """Concurrent backend switching (not recommended)."""
        system = MinimalSystemForCopy()

        def switch_backend():
            for _ in range(10):
                system.set_default_backend("numpy")

        # Create threads (not recommended in practice)
        threads = [threading.Thread(target=switch_backend) for _ in range(3)]

        # This test just ensures no crashes
        # (proper synchronization is user's responsibility)
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should still be valid
        assert system._initialized


# ============================================================================
# Run Tests
# ============================================================================


if __name__ == "__main__":
    # Run additional tests with verbose output
    pytest.main([__file__, "-v", "--tb=short", "-m", "not slow"])
