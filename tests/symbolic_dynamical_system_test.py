"""
Comprehensive unit tests for SymbolicDynamicalSystem

Tests cover:
1. System initialization and validation
2. Backend detection and conversion
3. Forward dynamics (all backends)
4. Linearization (all backends)
5. Output functions (all backends)
6. Equilibrium handling
7. Backend switching and device management
8. Code generation and caching
9. Performance monitoring
10. Configuration save/load
"""

import pytest
import numpy as np
import sympy as sp
from typing import Dict, List, Tuple

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

from src.systems.base.symbolic_dynamical_system import SymbolicDynamicalSystem


# ============================================================================
# Test Fixtures - Simple Systems
# ============================================================================


class SimpleFirstOrderSystem(SymbolicDynamicalSystem):
    """Simple linear system: dx/dt = -x + u"""

    def define_system(self, a=1.0):
        a_sym = sp.symbols("a", real=True, positive=True)
        x = sp.symbols("x", real=True)
        u = sp.symbols("u", real=True)

        dx = -a_sym * x + u

        self.state_vars = [x]
        self.control_vars = [u]
        self._f_sym = sp.Matrix([dx])
        self.parameters = {a_sym: a}
        self.order = 1


class SimpleSecondOrderSystem(SymbolicDynamicalSystem):
    """Simple harmonic oscillator: q̈ = -k*q - c*q̇ + u"""

    def define_system(self, k=10.0, c=0.5):
        k_sym, c_sym = sp.symbols("k c", real=True, positive=True)
        q, q_dot = sp.symbols("q q_dot", real=True)
        u = sp.symbols("u", real=True)

        q_ddot = -k_sym * q - c_sym * q_dot + u

        self.state_vars = [q, q_dot]
        self.control_vars = [u]
        self._f_sym = sp.Matrix([q_ddot])
        self.parameters = {k_sym: k, c_sym: c}
        self.order = 2


class CustomOutputSystem(SymbolicDynamicalSystem):
    """System with custom output: y = [x1, x1^2 + x2^2]"""

    def define_system(self):
        x1, x2 = sp.symbols("x1 x2", real=True)
        u = sp.symbols("u", real=True)

        self.state_vars = [x1, x2]
        self.control_vars = [u]
        self._f_sym = sp.Matrix([x2, -x1])
        self._h_sym = sp.Matrix([x1, x1**2 + x2**2])
        self.parameters = {}
        self.order = 1


# ============================================================================
# Test Class 1: Initialization and Validation
# ============================================================================


class TestInitializationAndValidation:
    """Test system initialization and validation"""

    def test_successful_initialization(self):
        """Test that valid system initializes successfully"""
        system = SimpleFirstOrderSystem(a=2.0)

        assert system._initialized is True
        assert system.nx == 1
        assert system.nu == 1
        assert system.order == 1
        assert system._default_backend == "numpy"

    def test_template_method_pattern(self):
        """Test that define_system is called automatically"""
        system = SimpleFirstOrderSystem()

        # Should have state vars from define_system
        assert len(system.state_vars) == 1
        assert len(system.control_vars) == 1
        assert system._f_sym is not None

    def test_validation_empty_state_vars(self):
        """Test validation fails with empty state_vars"""

        class BadSystem(SymbolicDynamicalSystem):
            def define_system(self):
                self.state_vars = []  # Empty!
                self.control_vars = [sp.symbols("u")]
                self._f_sym = sp.Matrix([sp.symbols("u")])
                self.parameters = {}

        with pytest.raises(ValueError, match="state_vars is empty"):
            BadSystem()

    def test_validation_empty_control_vars(self):
        """Test validation fails with empty control_vars"""

        class BadSystem(SymbolicDynamicalSystem):
            def define_system(self):
                self.state_vars = [sp.symbols("x")]
                self.control_vars = []  # Empty!
                self._f_sym = sp.Matrix([sp.symbols("x")])
                self.parameters = {}

        with pytest.raises(ValueError, match="control_vars is empty"):
            BadSystem()

    def test_validation_missing_f_sym(self):
        """Test validation fails with missing _f_sym"""

        class BadSystem(SymbolicDynamicalSystem):
            def define_system(self):
                self.state_vars = [sp.symbols("x")]
                self.control_vars = [sp.symbols("u")]
                self._f_sym = None  # Not defined!
                self.parameters = {}

        with pytest.raises(ValueError, match="_f_sym is not defined"):
            BadSystem()

    def test_validation_wrong_parameter_keys(self):
        """Test validation fails when using string parameter keys"""

        class BadSystem(SymbolicDynamicalSystem):
            def define_system(self):
                x = sp.symbols("x")
                u = sp.symbols("u")

                self.state_vars = [x]
                self.control_vars = [u]
                self._f_sym = sp.Matrix([-x + u])
                self.parameters = {"a": 1.0}  # String key! Should be Symbol

        with pytest.raises(ValueError, match="Parameter key"):
            BadSystem()

    def test_validation_non_symbol_state_vars(self):
        """Test validation fails when state_vars contains non-Symbols"""

        class BadSystem(SymbolicDynamicalSystem):
            def define_system(self):
                self.state_vars = ["x", "y"]  # Strings! Should be Symbols
                self.control_vars = [sp.symbols("u")]
                self._f_sym = sp.Matrix([0, 0])
                self.parameters = {}

        with pytest.raises(ValueError, match="not a SymPy Symbol"):
            BadSystem()

    def test_validation_f_sym_not_matrix(self):
        """Test validation fails when _f_sym is not a Matrix"""

        class BadSystem(SymbolicDynamicalSystem):
            def define_system(self):
                x = sp.symbols("x")
                u = sp.symbols("u")

                self.state_vars = [x]
                self.control_vars = [u]
                self._f_sym = [x + u]  # List! Should be Matrix
                self.parameters = {}

        with pytest.raises(ValueError, match="_f_sym must be sp.Matrix"):
            BadSystem()

    def test_validation_dimension_mismatch(self):
        """Test validation fails when _f_sym has wrong dimensions"""

        class BadSystem(SymbolicDynamicalSystem):
            def define_system(self):
                x, y = sp.symbols("x y")
                u = sp.symbols("u")

                self.state_vars = [x, y]
                self.control_vars = [u]
                self._f_sym = sp.Matrix([x])  # Only 1 row, need 2!
                self.parameters = {}
                self.order = 1

        with pytest.raises(ValueError, match="rows but expected"):
            BadSystem()

    def test_validation_order_dimension_mismatch(self):
        """Test validation fails when nx not divisible by order"""

        class BadSystem(SymbolicDynamicalSystem):
            def define_system(self):
                x1, x2, x3 = sp.symbols("x1 x2 x3")
                u = sp.symbols("u")

                self.state_vars = [x1, x2, x3]  # 3 states
                self.control_vars = [u]
                self._f_sym = sp.Matrix([0])  # Placeholder
                self.parameters = {}
                self.order = 2  # But 3 is not divisible by 2!

        with pytest.raises(ValueError, match="must be divisible by order"):
            BadSystem()

    def test_validation_undefined_symbols(self):
        """Test validation fails with undefined symbols in _f_sym"""

        class BadSystem(SymbolicDynamicalSystem):
            def define_system(self):
                x = sp.symbols("x")
                u = sp.symbols("u")
                mystery = sp.symbols("mystery")  # Not declared anywhere!

                self.state_vars = [x]
                self.control_vars = [u]
                self._f_sym = sp.Matrix([x + mystery])  # Uses undefined symbol
                self.parameters = {}

        with pytest.raises(ValueError, match="undefined symbols"):
            BadSystem()

    def test_validation_control_in_output(self):
        """Test validation fails when output depends on control"""

        class BadSystem(SymbolicDynamicalSystem):
            def define_system(self):
                x = sp.symbols("x")
                u = sp.symbols("u")

                self.state_vars = [x]
                self.control_vars = [u]
                self._f_sym = sp.Matrix([x + u])
                self._h_sym = sp.Matrix([x + u])  # Output depends on control!
                self.parameters = {}

        with pytest.raises(ValueError, match="Output.*should only depend on states"):
            BadSystem()

    def test_validation_duplicate_variables(self):
        """Test validation fails with duplicate variable names"""

        class BadSystem(SymbolicDynamicalSystem):
            def define_system(self):
                x1 = sp.symbols("x")
                x2 = sp.symbols("x")  # Same name!
                u = sp.symbols("u")

                self.state_vars = [x1, x2]
                self.control_vars = [u]
                self._f_sym = sp.Matrix([0, 0])
                self.parameters = {}

        with pytest.raises(ValueError, match="Duplicate variable names"):
            BadSystem()

    def test_validation_non_finite_parameter(self):
        """Test validation fails with NaN/Inf parameter"""

        class BadSystem(SymbolicDynamicalSystem):
            def define_system(self):
                a = sp.symbols("a")
                x = sp.symbols("x")
                u = sp.symbols("u")

                self.state_vars = [x]
                self.control_vars = [u]
                self._f_sym = sp.Matrix([a * x])
                self.parameters = {a: np.inf}  # Infinite!

        with pytest.raises(ValueError, match="non-finite value"):
            BadSystem()

    def test_validation_negative_mass(self):
        """Test validation fails with negative mass"""

        class BadSystem(SymbolicDynamicalSystem):
            def define_system(self):
                m = sp.symbols("m")
                x = sp.symbols("x")
                u = sp.symbols("u")

                self.state_vars = [x]
                self.control_vars = [u]
                self._f_sym = sp.Matrix([u / m])
                self.parameters = {m: -1.0}  # Negative mass!

        with pytest.raises(ValueError, match="should be positive"):
            BadSystem()

    def test_validation_warnings(self):
        """Test that warnings are issued for unusual configurations"""

        class WeirdSystem(SymbolicDynamicalSystem):
            def define_system(self):
                x = sp.symbols("x")
                u = sp.symbols("u")
                a = sp.symbols("a")

                self.state_vars = [x]
                self.control_vars = [u]
                self._f_sym = sp.Matrix([u])  # Doesn't depend on x!
                self.parameters = {a: 1.0}  # Unused parameter!

        with pytest.warns(UserWarning):
            WeirdSystem()


# ============================================================================
# Test Class 2: Backend Detection and Conversion
# ============================================================================


class TestBackendHandling:
    """Test backend detection, conversion, and availability checking"""

    def test_detect_numpy_backend(self):
        """Test detecting NumPy arrays"""
        system = SimpleFirstOrderSystem()
        x = np.array([1.0])

        backend = system._detect_backend(x)
        assert backend == "numpy"

    @pytest.mark.skipif(not torch_available, reason="PyTorch not installed")
    def test_detect_torch_backend(self):
        """Test detecting PyTorch tensors"""
        system = SimpleFirstOrderSystem()
        x = torch.tensor([1.0])

        backend = system._detect_backend(x)
        assert backend == "torch"

    @pytest.mark.skipif(not jax_available, reason="JAX not installed")
    def test_detect_jax_backend(self):
        """Test detecting JAX arrays"""
        system = SimpleFirstOrderSystem()
        x = jnp.array([1.0])

        backend = system._detect_backend(x)
        assert backend == "jax"

    def test_detect_unknown_type(self):
        """Test error on unknown array type"""
        system = SimpleFirstOrderSystem()

        with pytest.raises(TypeError, match="Unknown input type"):
            system._detect_backend([1.0])  # Python list

    def test_convert_numpy_to_numpy(self):
        """Test numpy -> numpy is no-op"""
        system = SimpleFirstOrderSystem()
        x = np.array([1.0, 2.0])

        x_converted = system._convert_to_backend(x, "numpy")

        assert x_converted is x  # Same object
        assert isinstance(x_converted, np.ndarray)

    @pytest.mark.skipif(not torch_available, reason="PyTorch not installed")
    def test_convert_numpy_to_torch(self):
        """Test numpy -> torch conversion"""
        system = SimpleFirstOrderSystem()
        x = np.array([1.0, 2.0])

        x_torch = system._convert_to_backend(x, "torch")

        assert isinstance(x_torch, torch.Tensor)
        assert torch.allclose(x_torch, torch.tensor([1.0, 2.0]))

    @pytest.mark.skipif(not torch_available, reason="PyTorch not installed")
    def test_convert_torch_to_numpy(self):
        """Test torch -> numpy conversion"""
        system = SimpleFirstOrderSystem()
        x = torch.tensor([1.0, 2.0])

        x_numpy = system._convert_to_backend(x, "numpy")

        assert isinstance(x_numpy, np.ndarray)
        assert np.allclose(x_numpy, np.array([1.0, 2.0]))

    @pytest.mark.skipif(not jax_available, reason="JAX not installed")
    def test_convert_numpy_to_jax(self):
        """Test numpy -> jax conversion"""
        system = SimpleFirstOrderSystem()
        x = np.array([1.0, 2.0])

        x_jax = system._convert_to_backend(x, "jax")

        assert isinstance(x_jax, jnp.ndarray)
        assert jnp.allclose(x_jax, jnp.array([1.0, 2.0]))

    def test_check_backend_available_numpy(self):
        """Test NumPy is always available"""
        system = SimpleFirstOrderSystem()

        # Should not raise
        system._check_backend_available("numpy")

    def test_check_backend_unavailable(self):
        """Test error when backend not installed"""
        system = SimpleFirstOrderSystem()

        # This will fail unless you have both torch and jax
        # Pick the one you don't have installed
        if not torch_available:
            with pytest.raises(RuntimeError, match="PyTorch.*not available"):
                system._check_backend_available("torch")

        if not jax_available:
            with pytest.raises(RuntimeError, match="JAX.*not available"):
                system._check_backend_available("jax")

    def test_set_default_backend(self):
        """Test setting default backend"""
        system = SimpleFirstOrderSystem()

        system.set_default_backend("numpy")
        assert system._default_backend == "numpy"

        system.set_default_backend("numpy", device="cpu")
        assert system._preferred_device == "cpu"

    def test_set_invalid_backend(self):
        """Test error on invalid backend name"""
        system = SimpleFirstOrderSystem()

        with pytest.raises(ValueError, match="Invalid backend"):
            system.set_default_backend("tensorflow")


# ============================================================================
# Test Class 3: Forward Dynamics
# ============================================================================


class TestForwardDynamics:
    """Test forward dynamics evaluation across backends"""

    def test_forward_numpy(self):
        """Test forward dynamics with NumPy"""
        system = SimpleFirstOrderSystem(a=2.0)

        x = np.array([1.0])
        u = np.array([0.5])

        dx = system(x, u)

        assert isinstance(dx, np.ndarray)
        assert dx.shape == (1,)
        # dx = -2*x + u = -2*1 + 0.5 = -1.5
        assert np.allclose(dx, np.array([-1.5]))

    @pytest.mark.skipif(not torch_available, reason="PyTorch not installed")
    def test_forward_torch(self):
        """Test forward dynamics with PyTorch"""
        system = SimpleFirstOrderSystem(a=2.0)

        x = torch.tensor([1.0])
        u = torch.tensor([0.5])

        dx = system(x, u)

        assert isinstance(dx, torch.Tensor)
        assert dx.shape == (1,)
        assert torch.allclose(dx, torch.tensor([-1.5]))

    @pytest.mark.skipif(not jax_available, reason="JAX not installed")
    def test_forward_jax(self):
        """Test forward dynamics with JAX"""
        system = SimpleFirstOrderSystem(a=2.0)

        x = jnp.array([1.0])
        u = jnp.array([0.5])

        dx = system(x, u)

        assert isinstance(dx, jnp.ndarray)
        assert jnp.allclose(dx, jnp.array([-1.5]))

    def test_forward_batched_numpy(self):
        """Test batched forward dynamics"""
        system = SimpleFirstOrderSystem(a=2.0)

        x = np.array([[1.0], [2.0], [3.0]])  # Batch of 3
        u = np.array([[0.5], [0.5], [0.5]])

        dx = system(x, u)

        assert dx.shape == (3, 1)
        expected = np.array([[-1.5], [-3.5], [-5.5]])
        assert np.allclose(dx, expected)

    def test_forward_backend_override_numpy_to_torch(self):
        """Test forcing PyTorch backend with NumPy input"""
        if not torch_available:
            pytest.skip("PyTorch not installed")

        system = SimpleFirstOrderSystem(a=2.0)

        x = np.array([1.0])
        u = np.array([0.5])

        dx = system(x, u, backend="torch")

        assert isinstance(dx, torch.Tensor)
        assert torch.allclose(dx, torch.tensor([-1.5]))

    def test_forward_with_default_backend(self):
        """Test using configured default backend"""
        if not torch_available:
            pytest.skip("PyTorch not installed")

        system = SimpleFirstOrderSystem(a=2.0)
        system.set_default_backend("torch")

        x = np.array([1.0])
        u = np.array([0.5])

        dx = system(x, u, backend="default")

        assert isinstance(dx, torch.Tensor)

    def test_forward_second_order_system(self):
        """Test forward dynamics for second-order system"""
        system = SimpleSecondOrderSystem(k=10.0, c=0.5)

        x = np.array([0.1, 0.0])  # [q, q_dot]
        u = np.array([0.0])

        dx = system(x, u)

        assert dx.shape == (1,)
        # dx = q_ddot = -10*0.1 - 0.5*0 + 0 = -1.0
        assert np.allclose(dx, np.array([-1.0]))

    def test_callable_interface(self):
        """Test that __call__ works"""
        system = SimpleFirstOrderSystem(a=1.0)

        x = np.array([1.0])
        u = np.array([0.0])

        # These should be equivalent
        dx1 = system(x, u)
        dx2 = system.forward(x, u)

        assert np.allclose(dx1, dx2)


# ============================================================================
# Test Class 4: Linearization
# ============================================================================


class TestLinearization:
    """Test linearized dynamics computation"""

    def test_linearized_dynamics_numpy(self):
        """Test linearization with NumPy"""
        system = SimpleFirstOrderSystem(a=2.0)

        x = np.array([1.0])
        u = np.array([0.0])

        A, B = system.linearized_dynamics(x, u)

        assert isinstance(A, np.ndarray)
        assert isinstance(B, np.ndarray)
        assert A.shape == (1, 1)
        assert B.shape == (1, 1)

        # dδ/dx = -a = -2
        assert np.allclose(A, np.array([[-2.0]]))
        # dδ/du = 1
        assert np.allclose(B, np.array([[1.0]]))

    @pytest.mark.skipif(not torch_available, reason="PyTorch not installed")
    def test_linearized_dynamics_torch(self):
        """Test linearization with PyTorch"""
        system = SimpleFirstOrderSystem(a=2.0)

        x = torch.tensor([1.0])
        u = torch.tensor([0.0])

        A, B = system.linearized_dynamics(x, u)

        assert isinstance(A, torch.Tensor)
        assert isinstance(B, torch.Tensor)
        assert torch.allclose(A, torch.tensor([[-2.0]]))
        assert torch.allclose(B, torch.tensor([[1.0]]))

    def test_linearized_dynamics_second_order(self):
        """Test linearization of second-order system"""
        system = SimpleSecondOrderSystem(k=10.0, c=0.5)

        x = np.array([0.0, 0.0])
        u = np.array([0.0])

        A, B = system.linearized_dynamics(x, u)

        # State-space form for q̈ = -k*q - c*q̇ + u:
        # dx/dt = [q̇; q̈] = [0 1; -k -c][q; q̇] + [0; 1]u
        expected_A = np.array([[0, 1], [-10, -0.5]])
        expected_B = np.array([[0], [1]])

        assert A.shape == (2, 2)
        assert B.shape == (2, 1)
        assert np.allclose(A, expected_A)
        assert np.allclose(B, expected_B)

    def test_linearized_dynamics_symbolic(self):
        """Test symbolic linearization"""
        system = SimpleFirstOrderSystem(a=2.0)

        x_eq = sp.Matrix([0])
        u_eq = sp.Matrix([0])

        A_sym, B_sym = system.linearized_dynamics_symbolic(x_eq, u_eq)

        assert isinstance(A_sym, sp.Matrix)
        assert isinstance(B_sym, sp.Matrix)

        # Convert to numpy for checking
        A_np = np.array(A_sym, dtype=float)
        B_np = np.array(B_sym, dtype=float)

        assert np.allclose(A_np, np.array([[-2.0]]))
        assert np.allclose(B_np, np.array([[1.0]]))


# ============================================================================
# Test Class 5: Output Functions
# ============================================================================


class TestOutputFunctions:
    """Test output equation evaluation"""

    def test_default_output_numpy(self):
        """Test default output (identity) with NumPy"""
        system = SimpleFirstOrderSystem()

        x = np.array([1.0])
        y = system.h(x)

        assert isinstance(y, np.ndarray)
        assert np.allclose(y, x)

    def test_custom_output_numpy(self):
        """Test custom output function"""
        system = CustomOutputSystem()

        x = np.array([1.0, 2.0])
        y = system.h(x)

        # y = [x1, x1^2 + x2^2] = [1, 5]
        assert y.shape == (2,)
        assert np.allclose(y, np.array([1.0, 5.0]))

    @pytest.mark.skipif(not torch_available, reason="PyTorch not installed")
    def test_custom_output_torch(self):
        """Test custom output with PyTorch"""
        system = CustomOutputSystem()

        x = torch.tensor([1.0, 2.0])
        y = system.h(x)

        assert isinstance(y, torch.Tensor)
        assert torch.allclose(y, torch.tensor([1.0, 5.0]))

    def test_linearized_observation_numpy(self):
        """Test linearized observation matrix"""
        system = CustomOutputSystem()

        x = np.array([1.0, 2.0])
        C = system.linearized_observation(x)

        # C = dh/dx = [[1, 0], [2*x1, 2*x2]] = [[1, 0], [2, 4]]
        expected_C = np.array([[1.0, 0.0], [2.0, 4.0]])

        assert C.shape == (2, 2)
        assert np.allclose(C, expected_C)

    def test_output_backend_override(self):
        """Test forcing backend for output evaluation"""
        if not torch_available:
            pytest.skip("PyTorch not installed")

        system = CustomOutputSystem()

        x = np.array([1.0, 2.0])
        y = system.h(x, backend="torch")

        assert isinstance(y, torch.Tensor)


# ============================================================================
# Test Class 6: Equilibrium Handling
# ============================================================================


class TestEquilibriumHandling:
    """Test equilibrium management"""

    def test_default_origin_equilibrium(self):
        """Test that origin equilibrium exists by default"""
        system = SimpleFirstOrderSystem()

        equilibria = system.equilibria.list_names()

        assert "origin" in equilibria
        assert len(equilibria) == 1

    def test_get_origin_equilibrium(self):
        """Test getting origin equilibrium"""
        system = SimpleFirstOrderSystem()

        x_eq = system.equilibria.get_x("origin")
        u_eq = system.equilibria.get_u("origin")

        assert isinstance(x_eq, np.ndarray)
        assert isinstance(u_eq, np.ndarray)
        assert np.allclose(x_eq, np.array([0.0]))
        assert np.allclose(u_eq, np.array([0.0]))

    def test_add_equilibrium(self):
        """Test adding custom equilibrium"""
        system = SimpleFirstOrderSystem(a=2.0)

        # For dx = -2x + u, equilibrium at x=1 requires u=2
        system.add_equilibrium(
            "custom",
            x_eq=np.array([1.0]),
            u_eq=np.array([2.0]),
            verify=False,  # Skip verification for this test
        )

        x_eq = system.equilibria.get_x("custom")
        assert np.allclose(x_eq, np.array([1.0]))

    def test_equilibrium_verification_valid(self):
        """Test verification accepts valid equilibrium"""
        system = SimpleFirstOrderSystem(a=2.0)

        system.add_equilibrium(
            "valid", x_eq=np.array([1.0]), u_eq=np.array([2.0]), verify=True, tol=1e-6
        )

        # If we got here, no exception was raised - test passes
        assert "valid" in system.equilibria.list_names()

    def test_equilibrium_verification_invalid(self):
        """Test verification warns on invalid equilibrium"""
        system = SimpleFirstOrderSystem(a=2.0)

        # This is NOT an equilibrium: -2*1 + 0 = -2 ≠ 0
        with pytest.warns(UserWarning, match="may not be valid"):
            system.add_equilibrium(
                "invalid", x_eq=np.array([1.0]), u_eq=np.array([0.0]), verify=True  # Wrong control
            )

    def test_set_default_equilibrium(self):
        """Test changing default equilibrium"""
        system = SimpleFirstOrderSystem()

        system.add_equilibrium("custom", np.array([1.0]), np.array([0.0]), verify=False)
        system.equilibria.set_default("custom")

        assert system.equilibria._default == "custom"

    def test_get_equilibrium_with_backend(self):
        """Test getting equilibrium in different backends"""
        system = SimpleFirstOrderSystem()

        system.add_equilibrium("test", np.array([1.0]), np.array([0.5]), verify=False)

        # NumPy
        x_np = system.equilibria.get_x("test", backend="numpy")
        assert isinstance(x_np, np.ndarray)

        if torch_available:
            # PyTorch
            x_torch = system.equilibria.get_x("test", backend="torch")
            assert isinstance(x_torch, torch.Tensor)

        if jax_available:
            # JAX
            x_jax = system.equilibria.get_x("test", backend="jax")
            assert isinstance(x_jax, jnp.ndarray)

    def test_equilibrium_dimension_validation(self):
        """Test that wrong-sized equilibria are rejected"""
        system = SimpleFirstOrderSystem()

        with pytest.raises(ValueError, match="must have shape"):
            system.add_equilibrium(
                "bad",
                x_eq=np.array([1.0, 2.0]),  # 2 states, but system has 1!
                u_eq=np.array([0.0]),
            )


# ============================================================================
# Test Class 7: Code Generation and Caching
# ============================================================================


class TestCodeGeneration:
    """Test code generation and function caching"""

    def test_lazy_generation_numpy(self):
        """Test that functions are generated lazily"""
        system = SimpleFirstOrderSystem()

        assert system._code_gen.get_dynamics("numpy") is None  # Not generated yet

        x = np.array([1.0])
        u = np.array([0.0])

        dx = system(x, u)  # First call generates function

        assert system._code_gen.get_dynamics("numpy") is not None  # Now cached

    def test_compile_all_backends(self):
        """Test compiling for all available backends"""
        system = SimpleFirstOrderSystem()

        timings = system.compile(verbose=False)

        assert "numpy" in timings
        assert timings["numpy"] is not None
        assert system._code_gen.get_dynamics("numpy") is not None

    def test_compile_specific_backend(self):
        """Test compiling for specific backend"""
        system = SimpleFirstOrderSystem()

        timings = system.compile(backends=["numpy"], verbose=False)

        assert "numpy" in timings
        assert "torch" not in timings
        assert "jax" not in timings

    def test_compile_verbose(self, capsys):
        """Test verbose compilation output"""
        system = SimpleFirstOrderSystem()

        system.compile(backends=["numpy"], verbose=True)

        captured = capsys.readouterr()
        assert "Compiling numpy" in captured.out

    def test_function_reuse(self):
        """Test that cached functions are reused"""
        system = SimpleFirstOrderSystem()

        # First call
        x = np.array([1.0])
        u = np.array([0.0])
        system(x, u)

        func1 = system._code_gen.get_dynamics("numpy")

        # Second call
        system(x, u)

        func2 = system._code_gen.get_dynamics("numpy")

        assert func1 is func2  # Same function object

    def test_reset_caches(self):
        """Test cache clearing"""
        system = SimpleFirstOrderSystem()

        # Generate function
        system.compile(backends=["numpy"])
        assert system._code_gen.get_dynamics("numpy") is not None

        # Clear cache
        system.reset_caches(["numpy"])
        assert system._code_gen.get_dynamics("numpy") is None


# ============================================================================
# Test Class 8: Backend Switching
# ============================================================================


class TestBackendSwitching:
    """Test dynamic backend switching"""

    def test_to_device(self):
        """Test device setting"""
        system = SimpleFirstOrderSystem()

        result = system.to_device("cuda")

        assert system._preferred_device == "cuda"
        assert result is system  # Returns self for chaining

    @pytest.mark.skipif(not torch_available, reason="PyTorch not installed")
    def test_device_clears_cache(self):
        """Test that changing device clears backend cache"""
        system = SimpleFirstOrderSystem()
        system.set_default_backend("torch")

        # Generate function
        system.compile(backends=["torch"])
        assert system._code_gen.get_dynamics("torch") is not None

        # Change device
        system.to_device("cuda")

        # Cache should be cleared
        assert system._code_gen.get_dynamics("torch") is None

    def test_use_backend_context_manager(self):
        """Test temporary backend switching"""
        if not torch_available:
            pytest.skip("PyTorch not installed")

        system = SimpleFirstOrderSystem()
        system.set_default_backend("numpy")

        assert system._default_backend == "numpy"

        # Temporarily switch to torch
        with system.use_backend("torch"):
            assert system._default_backend == "torch"

        # Back to numpy
        assert system._default_backend == "numpy"

    def test_clone_same_backend(self):
        """Test cloning system"""
        system = SimpleFirstOrderSystem(a=2.0)

        cloned = system.clone()

        assert cloned is not system
        assert cloned._default_backend == system._default_backend
        assert len(cloned.state_vars) == len(system.state_vars)

    def test_clone_different_backend(self):
        """Test cloning with backend change"""
        if not torch_available:
            pytest.skip("PyTorch not installed")

        system = SimpleFirstOrderSystem()
        system.set_default_backend("numpy")

        cloned = system.clone(backend="torch")

        assert system._default_backend == "numpy"
        assert cloned._default_backend == "torch"

    def test_get_backend_info(self):
        """Test getting backend information"""
        system = SimpleFirstOrderSystem()

        info = system.get_backend_info()

        assert "default_backend" in info
        assert "available_backends" in info
        assert "compiled_backends" in info
        assert "numpy" in info["available_backends"]
        assert info["default_backend"] == "numpy"


# ============================================================================
# Test Class 9: Jacobian Verification
# ============================================================================


class TestJacobianVerification:
    """Test Jacobian verification against autodiff"""

    @pytest.mark.skipif(not torch_available, reason="PyTorch not installed")
    def test_verify_jacobians_torch(self):
        """Test Jacobian verification with PyTorch autodiff"""
        system = SimpleFirstOrderSystem(a=2.0)

        x = torch.tensor([1.0])
        u = torch.tensor([0.0])

        results = system.verify_jacobians(x, u, backend="torch", tol=1e-4)

        assert results["A_match"] is True
        assert results["B_match"] is True
        assert results["A_error"] < 1e-4
        assert results["B_error"] < 1e-4

    @pytest.mark.skipif(not jax_available, reason="JAX not installed")
    def test_verify_jacobians_jax(self):
        """Test Jacobian verification with JAX autodiff"""
        system = SimpleFirstOrderSystem(a=2.0)

        x = jnp.array([1.0])
        u = jnp.array([0.0])

        results = system.verify_jacobians(x, u, backend="jax", tol=1e-4)

        assert results["A_match"] is True
        assert results["B_match"] is True

    def test_verify_jacobians_numpy_fails(self):
        """Test that NumPy backend is rejected for verification"""
        system = SimpleFirstOrderSystem()

        x = np.array([1.0])
        u = np.array([0.0])

        with pytest.raises(ValueError, match="requires autodiff backend"):
            system.verify_jacobians(x, u, backend="numpy")

    @pytest.mark.skipif(not torch_available, reason="PyTorch not installed")
    def test_verify_second_order_jacobians(self):
        """Test Jacobian verification for second-order system"""
        system = SimpleSecondOrderSystem(k=10.0, c=0.5)

        x = torch.tensor([0.1, 0.0])
        u = torch.tensor([0.0])

        results = system.verify_jacobians(x, u, backend="torch", tol=1e-3)

        assert results["A_match"] is True
        assert results["B_match"] is True


# ============================================================================
# Test Class 10: Performance Monitoring
# ============================================================================


class TestPerformanceMonitoring:
    """Test performance statistics tracking"""

    def test_forward_call_counting(self):
        """Test that forward calls are counted"""
        system = SimpleFirstOrderSystem()
        
        x = np.array([1.0])
        u = np.array([0.0])
        
        # Get initial calls from dynamics evaluator
        initial_calls = system._dynamics.get_stats()['calls']
        
        system(x, u)
        system(x, u)
        system(x, u)
        
        # Check calls from dynamics evaluator
        assert system._dynamics.get_stats()['calls'] == initial_calls + 3

    def test_get_performance_stats(self):
        """Test getting performance statistics"""
        system = SimpleFirstOrderSystem()

        x = np.array([1.0])
        u = np.array([0.0])

        system(x, u)

        stats = system.get_performance_stats()

        assert "forward_calls" in stats
        assert "avg_forward_time" in stats
        assert stats["forward_calls"] >= 1

    def test_reset_performance_stats(self):
        """Test resetting performance counters"""
        system = SimpleFirstOrderSystem()
        
        x = np.array([1.0])
        u = np.array([0.0])
        
        system(x, u)
        assert system._dynamics.get_stats()['calls'] > 0
        
        system.reset_performance_stats()
        assert system._dynamics.get_stats()['calls'] == 0
        assert system._dynamics.get_stats()['total_time'] == 0.0


# ============================================================================
# Test Class 11: Configuration Save/Load
# ============================================================================


class TestConfiguration:
    """Test configuration persistence"""

    def test_get_config_dict(self):
        """Test getting configuration as dictionary"""
        system = SimpleFirstOrderSystem(a=2.0)

        config = system.get_config_dict()

        assert config["class_name"] == "SimpleFirstOrderSystem"
        assert config["nx"] == 1
        assert config["nu"] == 1
        assert config["order"] == 1
        assert config["default_backend"] == "numpy"
        assert "a" in config["parameters"]

    def test_save_config_json(self, tmp_path):
        """Test saving configuration to JSON"""
        system = SimpleFirstOrderSystem(a=2.0)

        config_file = tmp_path / "system_config.json"
        system.save_config(str(config_file))

        assert config_file.exists()

        # Load and verify
        import json

        with open(config_file) as f:
            loaded = json.load(f)

        assert loaded["class_name"] == "SimpleFirstOrderSystem"
        assert loaded["default_backend"] == "numpy"

    def test_save_config_with_equilibria(self, tmp_path):
        """Test that equilibria are saved"""
        system = SimpleFirstOrderSystem()
        system.add_equilibrium("test", np.array([1.0]), np.array([0.5]), verify=False)

        config_file = tmp_path / "config.json"
        system.save_config(str(config_file))

        import json

        with open(config_file) as f:
            config = json.load(f)

        assert "equilibria" in config
        assert "test" in config["equilibria"]
        assert config["equilibria"]["test"]["x"] == [1.0]


# ============================================================================
# Test Class 12: Utility Methods
# ============================================================================


class TestUtilityMethods:
    """Test utility and helper methods"""

    def test_repr(self):
        """Test __repr__ output"""
        system = SimpleFirstOrderSystem()

        repr_str = repr(system)

        assert "SimpleFirstOrderSystem" in repr_str
        assert "nx=1" in repr_str
        assert "nu=1" in repr_str
        assert "backend=numpy" in repr_str

    def test_str(self):
        """Test __str__ output"""
        system = SimpleFirstOrderSystem()

        str_repr = str(system)

        assert "SimpleFirstOrderSystem" in str_repr
        assert "nx=1" in str_repr

    def test_str_with_multiple_equilibria(self):
        """Test __str__ shows equilibria count"""
        system = SimpleFirstOrderSystem()
        system.add_equilibrium("eq1", np.array([1.0]), np.array([0.0]), verify=False)
        system.add_equilibrium("eq2", np.array([2.0]), np.array([0.0]), verify=False)

        str_repr = str(system)

        assert "equilibria" in str_repr

    def test_print_equations(self, capsys):
        """Test equation printing"""
        system = SimpleFirstOrderSystem(a=2.0)

        system.print_equations(simplify=True)

        captured = capsys.readouterr()
        assert "SimpleFirstOrderSystem" in captured.out
        assert "State Variables" in captured.out
        assert "Dynamics" in captured.out

    def test_print_equations_with_output(self, capsys):
        """Test printing system with custom output"""
        system = CustomOutputSystem()

        system.print_equations(simplify=True)

        captured = capsys.readouterr()
        assert "Output" in captured.out


# ============================================================================
# Test Class 13: Edge Cases and Error Handling
# ============================================================================


class TestEdgeCases:
    """Test edge cases and error conditions"""

    def test_zero_dimensional_input_error(self):
        """Test error on scalar input"""
        system = SimpleFirstOrderSystem()

        with pytest.raises(ValueError, match="at least 1D"):
            system(np.array(1.0), np.array(0.0))  # 0D arrays

    def test_wrong_state_dimension(self):
        """Test error on wrong state dimension"""
        system = SimpleFirstOrderSystem()

        x = np.array([1.0, 2.0])  # 2D but system is 1D!
        u = np.array([0.0])

        with pytest.raises(ValueError, match="Expected state dimension"):
            system(x, u)

    def test_wrong_control_dimension(self):
        """Test error on wrong control dimension"""
        system = SimpleFirstOrderSystem()

        x = np.array([1.0])
        u = np.array([0.0, 0.0])  # 2D but system has 1 control!

        with pytest.raises(ValueError, match="Expected control dimension"):
            system(x, u)

    def test_warmup_success(self, capsys):
        """Test backend warmup"""
        system = SimpleFirstOrderSystem()

        success = system.warmup(backend="numpy")

        captured = capsys.readouterr()
        assert "Warming up" in captured.out
        assert success is True

    def test_warmup_with_test_point(self, capsys):
        """Test warmup with custom test point"""
        system = SimpleFirstOrderSystem()

        x_test = np.array([1.0])
        u_test = np.array([0.5])

        success = system.warmup(backend="numpy", test_point=(x_test, u_test))

        assert success is True


# ============================================================================
# Test Class 14: Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests combining multiple features"""

    def test_full_workflow_numpy(self):
        """Test complete workflow with NumPy backend"""
        # Create system
        system = SimpleFirstOrderSystem(a=2.0)

        # Add custom equilibrium
        system.add_equilibrium("eq1", np.array([1.0]), np.array([2.0]), verify=True)

        # Get equilibrium
        x_eq = system.equilibria.get_x("eq1")
        u_eq = system.equilibria.get_u("eq1")

        # Evaluate dynamics at equilibrium
        dx = system(x_eq, u_eq)

        # Should be near zero (at equilibrium)
        assert np.abs(dx).max() < 1e-10

        # Linearize at equilibrium
        A, B = system.linearized_dynamics(x_eq, u_eq)

        assert A.shape == (1, 1)
        assert B.shape == (1, 1)

    @pytest.mark.skipif(not torch_available, reason="PyTorch not installed")
    def test_full_workflow_torch(self):
        """Test complete workflow with PyTorch backend"""
        system = SimpleFirstOrderSystem(a=2.0)
        system.set_default_backend("torch")

        # Compile
        system.compile(backends=["torch"])

        # Evaluate
        x = torch.tensor([1.0])
        u = torch.tensor([0.0])
        dx = system(x, u, backend="default")

        assert isinstance(dx, torch.Tensor)

        # Verify gradients work
        x.requires_grad_(True)
        dx = system(x, u)
        dx.backward()

        assert x.grad is not None

    def test_multi_backend_consistency(self):
        """Test that all backends give same results"""
        system = SimpleFirstOrderSystem(a=2.0)

        x_np = np.array([1.0])
        u_np = np.array([0.5])

        # NumPy result
        dx_np = system(x_np, u_np)

        backends_to_test = ["numpy"]
        if torch_available:
            backends_to_test.append("torch")
        if jax_available:
            backends_to_test.append("jax")

        # All backends should give same numerical result
        for backend in backends_to_test:
            dx = system(x_np, u_np, backend=backend)
            dx_val = np.array(dx) if not isinstance(dx, np.ndarray) else dx

            assert np.allclose(dx_val, dx_np), f"{backend} doesn't match NumPy"

    def test_second_order_complete_workflow(self):
        """Test second-order system end-to-end"""
        system = SimpleSecondOrderSystem(k=10.0, c=0.5)

        # State: [position, velocity]
        x = np.array([0.1, 0.0])
        u = np.array([0.0])

        # Dynamics (acceleration)
        dx = system(x, u)
        assert dx.shape == (1,)

        # Linearization (state-space form)
        A, B = system.linearized_dynamics(x, u)
        assert A.shape == (2, 2)
        assert B.shape == (2, 1)

        # Verify state-space structure
        assert np.allclose(A[0, 1], 1.0)  # dq/dt = q̇
        assert np.allclose(B[0, 0], 0.0)  # Control doesn't affect position directly


# ============================================================================
# Test Class 15: Property Tests
# ============================================================================


class TestProperties:
    """Test system properties"""

    def test_nx_property(self):
        """Test nx property"""
        system = SimpleFirstOrderSystem()
        assert system.nx == 1

        system2 = SimpleSecondOrderSystem()
        assert system2.nx == 2

    def test_nu_property(self):
        """Test nu property"""
        system = SimpleFirstOrderSystem()
        assert system.nu == 1

    def test_ny_property_default(self):
        """Test ny defaults to nx"""
        system = SimpleFirstOrderSystem()
        assert system.ny == system.nx

    def test_ny_property_custom(self):
        """Test ny with custom output"""
        system = CustomOutputSystem()
        assert system.ny == 2

    def test_nq_property_first_order(self):
        """Test nq for first-order system"""
        system = SimpleFirstOrderSystem()
        assert system.nq == system.nx

    def test_nq_property_second_order(self):
        """Test nq for second-order system"""
        system = SimpleSecondOrderSystem()
        assert system.nq == 1  # nx=2, order=2, so nq=1


# ============================================================================
# Run Tests
# ============================================================================


if __name__ == "__main__":
    # Run with pytest
    # pytest symbolic_dynamical_system_test.py -v

    # Or run specific test class
    # pytest symbolic_dynamical_system_test.py::TestInitializationAndValidation -v

    # Or with coverage
    # pytest symbolic_dynamical_system_test.py --cov=src.systems.base.symbolic_dynamical_system

    pytest.main([__file__, "-v", "--tb=short"])
