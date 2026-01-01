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
Unit Tests for SymbolicSystemBase
==================================

Comprehensive test suite for the SymbolicSystemBase abstract base class,
covering all aspects of symbolic system functionality:

- Initialization and validation
- Abstract method enforcement
- System dimensions (nx, nu, ny, nq)
- Backend configuration and management
- Code generation and compilation
- Performance tracking
- Symbolic utilities (parameter substitution)
- Equilibrium management
- Configuration persistence
- Error handling and edge cases

Test Organization
-----------------
Tests are organized into logical groups using pytest classes:

- TestAbstractEnforcement: Verify abstract methods must be implemented
- TestInitializationAndValidation: System setup and validation
- TestSystemDimensions: Properties (nx, nu, ny, nq)
- TestBackendConfiguration: Backend and device management
- TestCodeGeneration: Compilation and caching
- TestSymbolicUtilities: Parameter substitution
- TestEquilibriumManagement: Equilibrium point handling
- TestConfigurationPersistence: Save/load configuration
- TestPerformanceTracking: Statistics and timing
- TestEdgeCases: Boundary conditions and error cases
- TestHigherOrderSystems: Second and higher-order systems

Usage
-----
Run all tests:
    pytest test_symbolic_system_base.py -v

Run specific test class:
    pytest test_symbolic_system_base.py::TestSystemDimensions -v

Run with coverage:
    pytest test_symbolic_system_base.py --cov=src.systems.base.symbolic_system_base --cov-report=html

Notes
-----
- Uses concrete test implementations (MinimalSystem, LinearSystem, etc.)
- Mocks external dependencies where appropriate
- Tests both success and failure paths
- Includes parametrized tests for thorough coverage
"""

import json
import tempfile
from pathlib import Path
from typing import Dict, List, Optional
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
import sympy as sp

# Import the class under test
from src.systems.base.core.symbolic_system_base import SymbolicSystemBase

# Import required utilities and exceptions
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
# Concrete Test Implementations (for testing abstract base)
# ============================================================================


class MinimalSystem(SymbolicSystemBase):
    """
    Minimal concrete implementation for testing base class.
    
    Implements only the required abstract methods with simplest possible logic.
    """

    def define_system(self, a: float = 1.0):
        """Define a simple first-order linear system."""
        x = sp.symbols("x", real=True)
        u = sp.symbols("u", real=True)
        a_sym = sp.symbols("a", real=True, positive=True)

        self.state_vars = [x]
        self.control_vars = [u]
        self._f_sym = sp.Matrix([-a_sym * x + u])
        self.parameters = {a_sym: a}
        self.order = 1

    def print_equations(self, simplify: bool = True):
        """Print with generic notation."""
        print("Minimal System Equations")
        print(f"State: {self.state_vars}")
        print(f"Control: {self.control_vars}")
        for var, expr in zip(self.state_vars, self._f_sym):
            expr_sub = self.substitute_parameters(expr)
            if simplify:
                expr_sub = sp.simplify(expr_sub)
            print(f"  f({var}) = {expr_sub}")


class LinearSystem(SymbolicSystemBase):
    """
    Linear system for testing: dx/dt = A*x + B*u
    """

    def define_system(self, A: np.ndarray = None, B: np.ndarray = None):
        """Define linear system from matrices."""
        if A is None:
            A = np.array([[-1.0, 0.0], [0.0, -2.0]])
        if B is None:
            B = np.array([[1.0], [0.0]])

        nx, nu = A.shape[0], B.shape[1]

        # Create symbolic variables
        x_vars = sp.symbols(f"x0:{nx}", real=True)
        u_vars = sp.symbols(f"u0:{nu}", real=True)

        # Create symbolic matrices
        A_sym = sp.Matrix(A)
        B_sym = sp.Matrix(B)
        x_vec = sp.Matrix(x_vars)
        u_vec = sp.Matrix(u_vars)

        self.state_vars = list(x_vars)
        self.control_vars = list(u_vars)
        self._f_sym = A_sym * x_vec + B_sym * u_vec
        self.parameters = {}
        self.order = 1

    def print_equations(self, simplify: bool = True):
        """Print linear system equations."""
        print("Linear System: f(x,u) = A*x + B*u")
        for i, expr in enumerate(self._f_sym):
            print(f"  f[{i}] = {expr}")


class SecondOrderSystem(SymbolicSystemBase):
    """
    Second-order system for testing: q̈ = -k*q - c*q̇ + u/m
    """

    def define_system(self, m: float = 1.0, k: float = 10.0, c: float = 0.5):
        """Define second-order oscillator."""
        q, q_dot = sp.symbols("q q_dot", real=True)
        u = sp.symbols("u", real=True)
        m_sym, k_sym, c_sym = sp.symbols("m k c", positive=True)

        # Higher-order form: only return acceleration
        q_ddot = (-k_sym * q - c_sym * q_dot + u) / m_sym

        self.state_vars = [q, q_dot]
        self.control_vars = [u]
        self._f_sym = sp.Matrix([q_ddot])
        self.parameters = {m_sym: m, k_sym: k, c_sym: c}
        self.order = 2

    def print_equations(self, simplify: bool = True):
        """Print second-order equations."""
        print("Second-Order System: q̈ = f(q, q̇, u)")
        expr = self._f_sym[0]
        expr_sub = self.substitute_parameters(expr)
        if simplify:
            expr_sub = sp.simplify(expr_sub)
        print(f"  q̈ = {expr_sub}")


class SystemWithOutput(SymbolicSystemBase):
    """
    System with custom output function for testing.
    """

    def define_system(self):
        """Define system with output y = [x, x^2]."""
        x = sp.symbols("x", real=True)
        u = sp.symbols("u", real=True)

        self.state_vars = [x]
        self.control_vars = [u]
        self._f_sym = sp.Matrix([-x + u])
        self._h_sym = sp.Matrix([x, x**2])  # Custom output
        self.parameters = {}
        self.order = 1

    def print_equations(self, simplify: bool = True):
        """Print with output."""
        print("System with Output")
        print(f"f(x,u) = {self._f_sym[0]}")
        print(f"h(x) = {self._h_sym}")


class AutonomousSystem(SymbolicSystemBase):
    """
    Autonomous system (no control input) for testing.
    """

    def define_system(self, alpha: float = 1.0):
        """Define autonomous system: dx/dt = -alpha * x."""
        x = sp.symbols("x", real=True)
        alpha_sym = sp.symbols("alpha", positive=True)

        self.state_vars = [x]
        self.control_vars = []  # No control
        self._f_sym = sp.Matrix([-alpha_sym * x])
        self.parameters = {alpha_sym: alpha}
        self.order = 1

    def print_equations(self, simplify: bool = True):
        """Print autonomous system."""
        print("Autonomous System")
        print(f"dx/dt = {self.substitute_parameters(self._f_sym[0])}")


# ============================================================================
# Test: Abstract Method Enforcement
# ============================================================================


class TestAbstractEnforcement:
    """Test that abstract methods must be implemented."""

    def test_cannot_instantiate_base_class(self):
        """Cannot instantiate SymbolicSystemBase directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            SymbolicSystemBase()

    def test_must_implement_define_system(self):
        """Subclass must implement define_system()."""

        class IncompleteSystem1(SymbolicSystemBase):
            # Missing define_system
            def print_equations(self, simplify=True):
                pass

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteSystem1()

    def test_must_implement_print_equations(self):
        """Subclass must implement print_equations()."""

        class IncompleteSystem2(SymbolicSystemBase):
            def define_system(self):
                pass
            # Missing print_equations

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteSystem2()

    def test_concrete_implementation_works(self):
        """Concrete implementation with both methods works."""
        system = MinimalSystem(a=2.0)
        assert system is not None
        assert system._initialized


# ============================================================================
# Test: Initialization and Validation
# ============================================================================


class TestInitializationAndValidation:
    """Test system initialization and validation."""

    def test_successful_initialization(self):
        """System initializes successfully with valid definition."""
        system = MinimalSystem(a=1.5)
        
        assert system._initialized
        assert len(system.state_vars) == 1
        assert len(system.control_vars) == 1
        assert system._f_sym is not None
        assert len(system.parameters) == 1

    def test_initialization_sequence(self):
        """Initialization follows correct sequence."""
        system = MinimalSystem()
        
        # Containers initialized
        assert isinstance(system.state_vars, list)
        assert isinstance(system.control_vars, list)
        assert isinstance(system.parameters, dict)
        
        # Components initialized
        assert isinstance(system.backend, BackendManager)
        assert isinstance(system._validator, SymbolicValidator)
        assert isinstance(system.equilibria, EquilibriumHandler)
        assert isinstance(system._code_gen, CodeGenerator)

    def test_validation_on_init(self):
        """Validation runs during initialization."""
        
        class InvalidSystem(SymbolicSystemBase):
            def define_system(self):
                # Invalid: missing state_vars
                self.control_vars = []
                self._f_sym = sp.Matrix([0])
                self.parameters = {}
                self.order = 1
            
            def print_equations(self, simplify=True):
                pass
        
        with pytest.raises(ValidationError):
            InvalidSystem()

    def test_arguments_passed_to_define_system(self):
        """Arguments are correctly passed to define_system()."""
        a_value = 3.14
        system = MinimalSystem(a=a_value)
        
        # Check parameter was set
        a_sym = list(system.parameters.keys())[0]
        assert system.parameters[a_sym] == a_value

    def test_equilibria_dimensions_updated(self):
        """Equilibria handler dimensions updated after validation."""
        system = LinearSystem()
        
        assert system.equilibria.nx == system.nx
        assert system.equilibria.nu == system.nu

    def test_kwargs_passed_correctly(self):
        """Keyword arguments passed to define_system()."""
        system = SecondOrderSystem(m=2.0, k=5.0, c=0.3)
        
        # Verify parameters were set
        param_values = list(system.parameters.values())
        assert 2.0 in param_values
        assert 5.0 in param_values
        assert 0.3 in param_values


# ============================================================================
# Test: System Dimensions
# ============================================================================


class TestSystemDimensions:
    """Test dimension properties (nx, nu, ny, nq)."""

    def test_nx_property(self):
        """nx returns correct number of states."""
        system1 = MinimalSystem()  # 1 state
        assert system1.nx == 1
        
        system2 = LinearSystem()  # 2 states
        assert system2.nx == 2

    def test_nu_property(self):
        """nu returns correct number of controls."""
        system1 = MinimalSystem()  # 1 control
        assert system1.nu == 1
        
        system2 = AutonomousSystem()  # 0 controls
        assert system2.nu == 0

    def test_ny_property_identity_output(self):
        """ny equals nx for identity output."""
        system = MinimalSystem()
        assert system._h_sym is None
        assert system.ny == system.nx

    def test_ny_property_custom_output(self):
        """ny reflects custom output dimension."""
        system = SystemWithOutput()
        assert system._h_sym is not None
        assert system.ny == 2  # Custom output has 2 elements

    def test_ny_property_with_output_vars(self):
        """ny uses output_vars if defined."""
        system = MinimalSystem()
        system.output_vars = [sp.symbols("y1"), sp.symbols("y2"), sp.symbols("y3")]
        assert system.ny == 3

    def test_nq_property_first_order(self):
        """nq equals nx for first-order systems."""
        system = MinimalSystem()
        assert system.order == 1
        assert system.nq == system.nx

    def test_nq_property_second_order(self):
        """nq equals nx/2 for second-order systems."""
        system = SecondOrderSystem()
        assert system.order == 2
        assert system.nx == 2
        assert system.nq == 1

    def test_nq_property_higher_order(self):
        """nq computed correctly for higher-order systems."""
        system = LinearSystem()
        system.order = 4
        system.state_vars = sp.symbols("x0:8")  # 8 states
        assert system.nx == 8
        assert system.nq == 2  # 8 / 4


# ============================================================================
# Test: Backend Configuration
# ============================================================================


class TestBackendConfiguration:
    """Test backend and device configuration."""

    def test_default_backend_numpy(self):
        """Default backend is numpy."""
        system = MinimalSystem()
        assert system._default_backend == "numpy"

    def test_set_default_backend(self):
        """Can set default backend."""
        system = MinimalSystem()
        
        system.set_default_backend("torch")
        assert system._default_backend == "torch"

    def test_set_default_backend_with_device(self):
        """Can set backend and device together."""
        system = MinimalSystem()
        
        result = system.set_default_backend("torch", device="cpu")
        assert result is system  # Returns self for chaining
        assert system._default_backend == "torch"
        assert system._preferred_device == "cpu"

    def test_backend_property_setter(self):
        """_default_backend property setter works."""
        system = MinimalSystem()
        
        system._default_backend = "jax"
        assert system._default_backend == "jax"

    def test_device_property_getter_setter(self):
        """_preferred_device property works."""
        system = MinimalSystem()
        
        assert system._preferred_device == "cpu"  # Default
        system._preferred_device = "cuda:0"
        assert system._preferred_device == "cuda:0"

    def test_to_device_returns_self(self):
        """to_device() returns self for chaining."""
        system = MinimalSystem()
        
        result = system.to_device("cpu")
        assert result is system

    @patch.object(CodeGenerator, "reset_cache")
    def test_to_device_clears_cache_for_gpu_backends(self, mock_reset):
        """to_device() clears cache for torch/jax backends."""
        system = MinimalSystem()
        system.set_default_backend("torch")
        
        system.to_device("cuda")
        mock_reset.assert_called_once()

    def test_use_backend_context_manager(self):
        """use_backend() temporarily changes backend."""
        system = MinimalSystem()
        system.set_default_backend("numpy")
        
        with system.use_backend("torch") as sys:
            assert sys is system
            assert system._default_backend == "torch"
        
        assert system._default_backend == "numpy"

    def test_use_backend_with_device(self):
        """use_backend() can temporarily change device."""
        system = MinimalSystem()
        original_device = system._preferred_device
        
        with system.use_backend("torch", device="cuda"):
            assert system._preferred_device == "cuda"
        
        assert system._preferred_device == original_device

    def test_get_backend_info(self):
        """get_backend_info() returns complete information."""
        system = MinimalSystem()
        
        info = system.get_backend_info()
        
        assert "default_backend" in info
        assert "preferred_device" in info
        assert "available_backends" in info
        assert "compiled_backends" in info
        assert "initialized" in info
        assert info["initialized"] is True

    def test_backend_info_includes_numpy(self):
        """Backend info always includes numpy."""
        system = MinimalSystem()
        info = system.get_backend_info()
        
        assert "numpy" in info["available_backends"]


# ============================================================================
# Test: Code Generation
# ============================================================================


class TestCodeGeneration:
    """Test code generation and compilation."""

    def test_code_generator_initialized(self):
        """CodeGenerator is initialized after validation."""
        system = MinimalSystem()
        assert system._code_gen is not None
        assert isinstance(system._code_gen, CodeGenerator)

    @patch.object(CodeGenerator, "compile_all")
    def test_compile_delegates_to_code_generator(self, mock_compile):
        """compile() delegates to CodeGenerator.compile_all()."""
        mock_compile.return_value = {"numpy": 0.1}
        system = MinimalSystem()
        
        result = system.compile(backends=["numpy"], verbose=True)
        
        mock_compile.assert_called_once_with(backends=["numpy"], verbose=True)
        assert result == {"numpy": 0.1}

    @patch.object(CodeGenerator, "compile_all")
    def test_compile_returns_timing_dict(self, mock_compile):
        """compile() returns compilation times."""
        mock_compile.return_value = {"numpy": 0.1, "torch": 0.2}
        system = MinimalSystem()
        
        times = system.compile()
        
        assert isinstance(times, dict)
        assert "numpy" in times
        assert all(isinstance(v, (int, float)) for v in times.values())

    @patch.object(CodeGenerator, "reset_cache")
    def test_reset_caches_all_backends(self, mock_reset):
        """reset_caches() with no args resets all backends."""
        system = MinimalSystem()
        
        system.reset_caches()
        
        mock_reset.assert_called_once_with(None)

    @patch.object(CodeGenerator, "reset_cache")
    def test_reset_caches_specific_backends(self, mock_reset):
        """reset_caches() can target specific backends."""
        system = MinimalSystem()
        
        system.reset_caches(["torch", "jax"])
        
        mock_reset.assert_called_once_with(["torch", "jax"])

    @patch.object(CodeGenerator, "reset_cache")
    def test_clear_backend_cache(self, mock_reset):
        """_clear_backend_cache() is internal method."""
        system = MinimalSystem()
        
        system._clear_backend_cache("torch")
        
        mock_reset.assert_called_once_with(["torch"])


# ============================================================================
# Test: Performance Tracking
# ============================================================================


class TestPerformanceTracking:
    """Test performance statistics."""

    def test_get_performance_stats_returns_dict(self):
        """get_performance_stats() returns dictionary."""
        system = MinimalSystem()
        
        stats = system.get_performance_stats()
        
        assert isinstance(stats, dict)

    def test_performance_stats_has_required_keys(self):
        """Performance stats have expected keys."""
        system = MinimalSystem()
        
        stats = system.get_performance_stats()
        
        assert "forward_calls" in stats
        assert "forward_time" in stats
        assert "avg_forward_time" in stats

    def test_performance_stats_initialized_to_zero(self):
        """Performance stats start at zero."""
        system = MinimalSystem()
        
        stats = system.get_performance_stats()
        
        assert stats["forward_calls"] == 0
        assert stats["forward_time"] == 0.0
        assert stats["avg_forward_time"] == 0.0

    def test_reset_performance_stats(self):
        """reset_performance_stats() can be called."""
        system = MinimalSystem()
        
        # Should not raise
        system.reset_performance_stats()
        
        # Stats should still be accessible
        stats = system.get_performance_stats()
        assert isinstance(stats, dict)


# ============================================================================
# Test: Symbolic Utilities
# ============================================================================


class TestSymbolicUtilities:
    """Test symbolic manipulation utilities."""

    def test_substitute_parameters_simple(self):
        """substitute_parameters() works with simple expression."""
        system = MinimalSystem(a=2.0)
        
        a_sym = list(system.parameters.keys())[0]
        x = sp.symbols("x")
        expr = a_sym * x
        
        result = system.substitute_parameters(expr)
        
        expected = 2.0 * x
        assert result == expected

    def test_substitute_parameters_matrix(self):
        """substitute_parameters() works with matrices."""
        system = SecondOrderSystem(m=2.0, k=10.0, c=0.5)
        
        result = system.substitute_parameters(system._f_sym)
        
        assert isinstance(result, sp.Matrix)
        # Check that parameters were substituted (no parameter symbols remain)
        for param_sym in system.parameters.keys():
            assert param_sym not in result.free_symbols

    def test_substitute_parameters_multiple(self):
        """substitute_parameters() handles multiple parameters."""
        system = SecondOrderSystem(m=1.5, k=8.0, c=0.3)
        
        # Create expression with all parameters
        m, k, c = system.parameters.keys()
        q, q_dot = sp.symbols("q q_dot")
        expr = m * q + k * q_dot - c
        
        result = system.substitute_parameters(expr)
        
        # Check all parameters substituted
        assert m not in result.free_symbols
        assert k not in result.free_symbols
        assert c not in result.free_symbols
        
        # Check values
        expected = 1.5 * q + 8.0 * q_dot - 0.3
        assert result == expected

    def test_substitute_parameters_no_parameters(self):
        """substitute_parameters() works with no parameters."""
        system = LinearSystem()  # No parameters
        
        x = sp.symbols("x")
        expr = x**2 + x
        
        result = system.substitute_parameters(expr)
        
        assert result == expr  # Unchanged


# ============================================================================
# Test: Equilibrium Management
# ============================================================================


class TestEquilibriumManagement:
    """Test equilibrium point management."""

    def test_equilibria_handler_initialized(self):
        """EquilibriumHandler is initialized."""
        system = MinimalSystem()
        assert system.equilibria is not None
        assert isinstance(system.equilibria, EquilibriumHandler)

    def test_add_equilibrium_basic(self):
        """Can add equilibrium point."""
        system = MinimalSystem()
        
        x_eq = np.array([0.0])
        u_eq = np.array([0.0])
        
        system.add_equilibrium("test", x_eq, u_eq, verify=False)
        
        assert "test" in system.list_equilibria()

    def test_add_equilibrium_with_metadata(self):
        """Can add equilibrium with metadata."""
        system = MinimalSystem()
        
        x_eq = np.array([0.0])
        u_eq = np.array([0.0])
        
        system.add_equilibrium(
            "stable_point",
            x_eq,
            u_eq,
            verify=False,
            stability="stable",
            notes="Test equilibrium"
        )
        
        metadata = system.get_equilibrium_metadata("stable_point")
        assert metadata["stability"] == "stable"
        assert metadata["notes"] == "Test equilibrium"

    def test_get_equilibrium_default_backend(self):
        """get_equilibrium() uses default backend."""
        system = MinimalSystem()
        
        x_eq = np.array([1.0])
        u_eq = np.array([0.5])
        system.add_equilibrium("test", x_eq, u_eq, verify=False)
        
        x_ret, u_ret = system.get_equilibrium("test")
        
        assert isinstance(x_ret, np.ndarray)
        assert isinstance(u_ret, np.ndarray)
        np.testing.assert_array_equal(x_ret, x_eq)
        np.testing.assert_array_equal(u_ret, u_eq)

    def test_get_equilibrium_specific_backend(self):
        """get_equilibrium() respects backend argument."""
        system = MinimalSystem()
        
        x_eq = np.array([1.0])
        u_eq = np.array([0.5])
        system.add_equilibrium("test", x_eq, u_eq, verify=False)
        
        x_ret, u_ret = system.get_equilibrium("test", backend="numpy")
        
        assert isinstance(x_ret, np.ndarray)

    def test_list_equilibria(self):
        """list_equilibria() returns all equilibrium names."""
        system = MinimalSystem()
        
        system.add_equilibrium("eq1", np.array([0.0]), np.array([0.0]), verify=False)
        system.add_equilibrium("eq2", np.array([1.0]), np.array([0.0]), verify=False)
        
        names = system.list_equilibria()
        
        assert isinstance(names, list)
        assert "eq1" in names
        assert "eq2" in names

    def test_set_default_equilibrium(self):
        """set_default_equilibrium() sets default."""
        system = MinimalSystem()
        
        system.add_equilibrium("custom", np.array([1.0]), np.array([0.0]), verify=False)
        
        result = system.set_default_equilibrium("custom")
        
        assert result is system  # Returns self for chaining
        assert system.equilibria._default == "custom"

    def test_remove_equilibrium(self):
        """remove_equilibrium() removes equilibrium."""
        system = MinimalSystem()
        
        system.add_equilibrium("temp", np.array([0.0]), np.array([0.0]), verify=False)
        assert "temp" in system.list_equilibria()
        
        system.remove_equilibrium("temp")
        
        assert "temp" not in system.list_equilibria()

    def test_cannot_remove_origin(self):
        """Cannot remove 'origin' equilibrium."""
        system = MinimalSystem()
        
        with pytest.raises(ValueError, match="Cannot remove origin"):
            system.remove_equilibrium("origin")

    def test_remove_nonexistent_equilibrium_raises(self):
        """Removing nonexistent equilibrium raises error."""
        system = MinimalSystem()
        
        with pytest.raises(ValueError, match="Unknown equilibrium"):
            system.remove_equilibrium("nonexistent")

    def test_remove_default_resets_to_origin(self):
        """Removing default equilibrium resets to origin."""
        system = MinimalSystem()
        
        system.add_equilibrium("custom", np.array([0.0]), np.array([0.0]), verify=False)
        system.set_default_equilibrium("custom")
        
        system.remove_equilibrium("custom")
        
        assert system.equilibria._default == "origin"


# ============================================================================
# Test: Configuration Persistence
# ============================================================================


class TestConfigurationPersistence:
    """Test configuration save/load functionality."""

    def test_get_config_dict_structure(self):
        """get_config_dict() returns proper structure."""
        system = MinimalSystem(a=1.5)
        
        config = system.get_config_dict()
        
        assert isinstance(config, dict)
        assert "class_name" in config
        assert "state_vars" in config
        assert "control_vars" in config
        assert "output_vars" in config
        assert "parameters" in config
        assert "order" in config
        assert "nx" in config
        assert "nu" in config
        assert "ny" in config
        assert "backend" in config
        assert "device" in config

    def test_get_config_dict_values(self):
        """get_config_dict() contains correct values."""
        system = MinimalSystem(a=2.5)
        
        config = system.get_config_dict()
        
        assert config["class_name"] == "MinimalSystem"
        assert config["nx"] == 1
        assert config["nu"] == 1
        assert config["order"] == 1
        assert config["backend"] == "numpy"

    def test_get_config_dict_parameters_converted(self):
        """get_config_dict() converts parameter symbols to strings."""
        system = SecondOrderSystem(m=2.0, k=10.0, c=0.5)
        
        config = system.get_config_dict()
        params = config["parameters"]
        
        # Parameters should be string keys (for JSON serialization)
        assert all(isinstance(k, str) for k in params.keys())
        assert all(isinstance(v, (int, float)) for v in params.values())

    def test_save_config_creates_file(self):
        """save_config() creates JSON file."""
        system = MinimalSystem()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "config.json"
            
            system.save_config(str(filepath))
            
            assert filepath.exists()

    def test_save_config_valid_json(self):
        """save_config() creates valid JSON."""
        system = MinimalSystem(a=3.14)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "config.json"
            
            system.save_config(str(filepath))
            
            # Load and verify
            with open(filepath, "r") as f:
                config = json.load(f)
            
            assert config["class_name"] == "MinimalSystem"
            assert config["nx"] == 1

    def test_save_config_includes_all_data(self):
        """save_config() includes complete configuration."""
        system = SecondOrderSystem(m=1.5, k=8.0, c=0.3)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "config.json"
            
            system.save_config(str(filepath))
            
            with open(filepath, "r") as f:
                config = json.load(f)
            
            assert "m" in config["parameters"]
            assert "k" in config["parameters"]
            assert "c" in config["parameters"]
            assert config["order"] == 2

    def test_config_roundtrip_consistency(self):
        """Config can be saved and loaded with consistent data."""
        system1 = LinearSystem()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "config.json"
            
            # Save
            system1.save_config(str(filepath))
            
            # Load
            with open(filepath, "r") as f:
                config = json.load(f)
            
            # Verify consistency
            assert config["nx"] == system1.nx
            assert config["nu"] == system1.nu
            assert config["ny"] == system1.ny
            assert config["order"] == system1.order


# ============================================================================
# Test: String Representations
# ============================================================================


class TestStringRepresentations:
    """Test __str__ and __repr__ methods."""

    def test_repr_includes_class_name(self):
        """__repr__ includes class name."""
        system = MinimalSystem()
        
        repr_str = repr(system)
        
        assert "MinimalSystem" in repr_str

    def test_repr_includes_dimensions(self):
        """__repr__ includes dimensions."""
        system = LinearSystem()
        
        repr_str = repr(system)
        
        assert "nx=2" in repr_str
        assert "nu=1" in repr_str
        assert "ny=2" in repr_str

    def test_repr_includes_order(self):
        """__repr__ includes system order."""
        system = SecondOrderSystem()
        
        repr_str = repr(system)
        
        assert "order=2" in repr_str

    def test_repr_includes_backend_info(self):
        """__repr__ includes backend and device."""
        system = MinimalSystem()
        
        repr_str = repr(system)
        
        assert "backend=numpy" in repr_str
        assert "device=cpu" in repr_str

    def test_str_concise_format(self):
        """__str__ provides concise format."""
        system = MinimalSystem()
        
        str_repr = str(system)
        
        assert "MinimalSystem" in str_repr
        assert "nx=1" in str_repr
        assert "nu=1" in str_repr

    def test_str_includes_equilibria_count(self):
        """__str__ includes equilibria count if > 1."""
        system = MinimalSystem()
        
        # Add extra equilibrium
        system.add_equilibrium("test", np.array([1.0]), np.array([0.0]), verify=False)
        
        str_repr = str(system)
        
        assert "equilibria" in str_repr.lower()


# ============================================================================
# Test: Edge Cases and Error Handling
# ============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_control_autonomous_system(self):
        """Autonomous system (nu=0) works correctly."""
        system = AutonomousSystem(alpha=1.0)
        
        assert system.nu == 0
        assert len(system.control_vars) == 0

    def test_system_with_no_parameters(self):
        """System with no parameters works."""
        system = LinearSystem()
        
        assert len(system.parameters) == 0
        assert system._initialized

    def test_large_system_dimensions(self):
        """System with many states works."""
        # Create 10-state system
        A = -np.eye(10)
        B = np.ones((10, 3))
        
        system = LinearSystem(A=A, B=B)
        
        assert system.nx == 10
        assert system.nu == 3

    def test_parameter_substitution_with_no_parameters(self):
        """Parameter substitution works with empty parameters dict."""
        system = LinearSystem()  # No parameters
        
        x = sp.symbols("x")
        expr = x**2
        
        result = system.substitute_parameters(expr)
        assert result == expr

    def test_equilibrium_dimension_mismatch_handled(self):
        """EquilibriumHandler validates dimensions (via handler)."""
        system = MinimalSystem()
        
        # This should be caught by EquilibriumHandler
        x_wrong = np.array([0.0, 1.0])  # Wrong dimension
        u_eq = np.array([0.0])
        
        with pytest.raises(ValueError):
            system.add_equilibrium("wrong", x_wrong, u_eq, verify=False)

    def test_backend_configuration_chaining(self):
        """Backend methods support chaining."""
        system = MinimalSystem()
        
        result = (
            system
            .set_default_backend("numpy")
            .to_device("cpu")
            .set_default_equilibrium("origin")
        )
        
        assert result is system

    def test_multiple_initialization_not_possible(self):
        """System can only be initialized once (via __init__)."""
        system = MinimalSystem()
        
        # System should be initialized
        assert system._initialized
        
        # Calling define_system again would require manual re-validation
        # which is not supported (users shouldn't do this)
        # Just verify the system works normally
        assert system.nx == 1
        assert system.nu == 1

    def test_validation_error_provides_context(self):
        """ValidationError includes system class name."""
        
        class BrokenSystem(SymbolicSystemBase):
            def define_system(self):
                # Deliberately broken
                pass
            
            def print_equations(self, simplify=True):
                pass
        
        with pytest.raises(ValidationError) as exc_info:
            BrokenSystem()
        
        assert "BrokenSystem" in str(exc_info.value)


# ============================================================================
# Test: Higher-Order Systems
# ============================================================================


class TestHigherOrderSystems:
    """Test second and higher-order system handling."""

    def test_second_order_system_dimensions(self):
        """Second-order system has correct dimensions."""
        system = SecondOrderSystem()
        
        assert system.order == 2
        assert system.nx == 2  # [q, q_dot]
        assert system.nq == 1  # One generalized coordinate
        assert len(system._f_sym) == 1  # Only highest derivative

    def test_second_order_nq_calculation(self):
        """nq correctly computed for second-order system."""
        system = SecondOrderSystem()
        
        assert system.nq == system.nx // system.order
        assert system.nq == 1

    def test_third_order_system(self):
        """Can create third-order system."""
        
        class ThirdOrderSystem(SymbolicSystemBase):
            def define_system(self):
                q, q_dot, q_ddot = sp.symbols("q q_dot q_ddot", real=True)
                u = sp.symbols("u", real=True)
                
                self.state_vars = [q, q_dot, q_ddot]
                self.control_vars = [u]
                self._f_sym = sp.Matrix([u])  # Only q_dddot
                self.parameters = {}
                self.order = 3
            
            def print_equations(self, simplify=True):
                print("Third-order system")
        
        system = ThirdOrderSystem()
        
        assert system.order == 3
        assert system.nx == 3
        assert system.nq == 1

    def test_multiple_generalized_coordinates(self):
        """Second-order system with multiple coordinates."""
        
        class MultiCoordinateSystem(SymbolicSystemBase):
            def define_system(self):
                # Two coordinates: q1, q2
                q1, q1_dot, q2, q2_dot = sp.symbols("q1 q1_dot q2 q2_dot", real=True)
                u1, u2 = sp.symbols("u1 u2", real=True)
                
                self.state_vars = [q1, q1_dot, q2, q2_dot]
                self.control_vars = [u1, u2]
                self._f_sym = sp.Matrix([u1, u2])  # Both accelerations
                self.parameters = {}
                self.order = 2
            
            def print_equations(self, simplify=True):
                print("Multi-coordinate second-order")
        
        system = MultiCoordinateSystem()
        
        assert system.nx == 4
        assert system.nq == 2
        assert system.order == 2

    def test_first_order_state_space_equivalent(self):
        """First-order state-space form works correctly."""
        
        class FirstOrderEquivalent(SymbolicSystemBase):
            def define_system(self):
                q, q_dot = sp.symbols("q q_dot", real=True)
                u = sp.symbols("u", real=True)
                k, c = sp.symbols("k c", positive=True)
                
                # First-order form: return ALL derivatives
                self.state_vars = [q, q_dot]
                self.control_vars = [u]
                self._f_sym = sp.Matrix([
                    q_dot,
                    -k*q - c*q_dot + u
                ])
                self.parameters = {k: 10.0, c: 0.5}
                self.order = 1  # First-order state-space
            
            def print_equations(self, simplify=True):
                print("First-order state-space")
        
        system = FirstOrderEquivalent()
        
        assert system.order == 1
        assert system.nx == 2
        assert system.nq == 2  # Same as nx for order=1
        assert len(system._f_sym) == 2  # All derivatives


# ============================================================================
# Test: Print Equations
# ============================================================================


class TestPrintEquations:
    """Test print_equations() implementations."""

    def test_print_equations_callable(self, capsys):
        """print_equations() can be called."""
        system = MinimalSystem()
        
        system.print_equations()
        
        captured = capsys.readouterr()
        assert len(captured.out) > 0

    def test_print_equations_shows_system_info(self, capsys):
        """print_equations() displays system information."""
        system = MinimalSystem()
        
        system.print_equations()
        
        captured = capsys.readouterr()
        assert "Minimal System" in captured.out

    def test_print_equations_simplify_parameter(self, capsys):
        """print_equations() respects simplify parameter."""
        system = SecondOrderSystem()
        
        # Should not raise
        system.print_equations(simplify=True)
        system.print_equations(simplify=False)

    def test_print_equations_shows_dynamics(self, capsys):
        """print_equations() displays dynamics."""
        system = MinimalSystem()
        
        system.print_equations()
        
        captured = capsys.readouterr()
        # Should show state and control variables
        assert "x" in captured.out.lower() or "state" in captured.out.lower()


# ============================================================================
# Parametrized Tests
# ============================================================================


class TestParametrizedScenarios:
    """Parametrized tests for multiple scenarios."""

    @pytest.mark.parametrize("order,nx,expected_nq", [
        (1, 2, 2),   # First-order: nq = nx
        (2, 2, 1),   # Second-order: nq = nx/2
        (2, 4, 2),   # Second-order: nq = nx/2
        (3, 6, 2),   # Third-order: nq = nx/3
        (4, 8, 2),   # Fourth-order: nq = nx/4
    ])
    def test_nq_calculation_parametrized(self, order, nx, expected_nq):
        """Test nq calculation for various orders."""
        system = LinearSystem()
        system.order = order
        system.state_vars = sp.symbols(f"x0:{nx}")
        
        assert system.nq == expected_nq

    @pytest.mark.parametrize("backend", ["numpy", "torch", "jax"])
    def test_backend_switching(self, backend):
        """Test switching to different backends."""
        system = MinimalSystem()
        
        # Should not raise
        system.set_default_backend(backend)
        assert system._default_backend == backend

    @pytest.mark.parametrize("nx,nu,ny", [
        (1, 1, 1),    # SISO
        (2, 2, 2),    # 2x2 MIMO
        (3, 2, 1),    # 3 states, 2 controls, 1 output
        (10, 5, 3),   # Large system
    ])
    def test_various_system_dimensions(self, nx, nu, ny):
        """Test systems with various dimensions."""
        # Create appropriate A and B matrices
        A = -np.eye(nx)
        B = np.ones((nx, nu))
        
        system = LinearSystem(A=A, B=B)
        
        assert system.nx == nx
        assert system.nu == nu


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests combining multiple features."""

    def test_complete_workflow(self):
        """Test complete workflow: create, configure, use."""
        # Create system
        system = SecondOrderSystem(m=2.0, k=10.0, c=0.5)
        
        # Configure backend
        system.set_default_backend("numpy")
        
        # Add equilibrium
        x_eq = np.array([0.0, 0.0])
        u_eq = np.array([0.0])
        system.add_equilibrium("origin", x_eq, u_eq, verify=False)
        
        # Get configuration
        config = system.get_config_dict()
        
        # Verify everything works
        assert system._initialized
        assert len(system.parameters) == 3
        assert "origin" in system.list_equilibria()
        assert config["nx"] == 2

    def test_multiple_equilibria_workflow(self):
        """Test managing multiple equilibria."""
        system = MinimalSystem()
        
        # Add multiple equilibria
        system.add_equilibrium("eq1", np.array([0.0]), np.array([0.0]), verify=False)
        system.add_equilibrium("eq2", np.array([1.0]), np.array([0.5]), verify=False)
        system.add_equilibrium("eq3", np.array([-1.0]), np.array([-0.5]), verify=False)
        
        # List all
        names = system.list_equilibria()
        assert len(names) >= 3
        
        # Set default
        system.set_default_equilibrium("eq2")
        
        # Get default
        x, u = system.get_equilibrium()
        np.testing.assert_array_equal(x, np.array([1.0]))

    def test_backend_and_compilation_workflow(self):
        """Test backend configuration and compilation."""
        system = LinearSystem()
        
        # Configure backend
        system.set_default_backend("numpy")
        info_before = system.get_backend_info()
        
        # Compile
        with patch.object(CodeGenerator, "compile_all", return_value={"numpy": 0.1}):
            times = system.compile(backends=["numpy"])
        
        # Verify
        assert "numpy" in info_before["available_backends"]

    def test_configuration_persistence_workflow(self):
        """Test saving and loading configuration."""
        system = SecondOrderSystem(m=1.5, k=8.0, c=0.3)
        
        # Add some configuration
        system.set_default_backend("torch")
        system.add_equilibrium("test", np.array([0.0, 0.0]), np.array([0.0]), verify=False)
        
        # Save
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "system_config.json"
            system.save_config(str(filepath))
            
            # Verify file
            assert filepath.exists()
            
            # Load and check
            with open(filepath, "r") as f:
                config = json.load(f)
            
            assert config["backend"] == "torch"
            assert "m" in config["parameters"]


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def minimal_system():
    """Fixture providing a minimal test system."""
    return MinimalSystem(a=1.0)


@pytest.fixture
def linear_system():
    """Fixture providing a linear test system."""
    return LinearSystem()


@pytest.fixture
def second_order_system():
    """Fixture providing a second-order test system."""
    return SecondOrderSystem(m=1.0, k=10.0, c=0.5)


@pytest.fixture
def system_with_output():
    """Fixture providing a system with custom output."""
    return SystemWithOutput()


@pytest.fixture
def autonomous_system():
    """Fixture providing an autonomous system."""
    return AutonomousSystem(alpha=1.0)


# ============================================================================
# Test Using Fixtures
# ============================================================================


class TestUsingFixtures:
    """Tests using pytest fixtures."""

    def test_minimal_system_fixture(self, minimal_system):
        """Test minimal system fixture."""
        assert minimal_system.nx == 1
        assert minimal_system.nu == 1

    def test_linear_system_fixture(self, linear_system):
        """Test linear system fixture."""
        assert linear_system.nx == 2
        assert linear_system.nu == 1

    def test_second_order_fixture(self, second_order_system):
        """Test second-order system fixture."""
        assert second_order_system.order == 2
        assert second_order_system.nq == 1

    def test_output_system_fixture(self, system_with_output):
        """Test system with output fixture."""
        assert system_with_output.ny == 2

    def test_autonomous_fixture(self, autonomous_system):
        """Test autonomous system fixture."""
        assert autonomous_system.nu == 0


# ============================================================================
# Run Tests
# ============================================================================


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])
