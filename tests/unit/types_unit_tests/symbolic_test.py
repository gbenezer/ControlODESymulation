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
Unit Tests for Symbolic Types

Tests type definitions, usage patterns, and integration with SymPy
for the symbolic mathematics module.
"""

import numpy as np
import sympy as sp

from src.types.symbolic import (
    ParameterDict,
    SubstitutionDict,
    SymbolDict,
    SymbolicDiffusionMatrix,
    SymbolicExpression,
    SymbolicGradient,
    SymbolicHessian,
    SymbolicJacobian,
    SymbolicMatrix,
    SymbolicOutputEquations,
    SymbolicStateEquations,
    SymbolicSymbol,
)


class TestBasicSymbolicTypes:
    """Test basic symbolic type definitions."""

    def test_symbolic_expression_creation(self):
        """Test creating SymbolicExpression instances."""
        x, y = sp.symbols("x y")
        expr: SymbolicExpression = x**2 + sp.sin(y)

        assert isinstance(expr, sp.Expr)
        assert isinstance(expr, SymbolicExpression)

    def test_symbolic_expression_operations(self):
        """Test operations on SymbolicExpression."""
        x, y = sp.symbols("x y")
        expr: SymbolicExpression = x**2 + y

        # Differentiation
        dexpr_dx = sp.diff(expr, x)
        assert dexpr_dx == 2 * x

        # Evaluation
        result = expr.subs({x: 1, y: 0})
        assert result == 1

        # Simplification
        complex_expr: SymbolicExpression = sp.sin(x) ** 2 + sp.cos(x) ** 2
        simplified = sp.simplify(complex_expr)
        assert simplified == 1

    def test_symbolic_matrix_creation(self):
        """Test creating SymbolicMatrix instances."""
        x, y = sp.symbols("x y")
        mat: SymbolicMatrix = sp.Matrix([x, y, x + y])

        assert isinstance(mat, sp.Matrix)
        assert isinstance(mat, SymbolicMatrix)
        assert mat.shape == (3, 1)

    def test_symbolic_matrix_operations(self):
        """Test operations on SymbolicMatrix."""
        x, y = sp.symbols("x y")
        mat: SymbolicMatrix = sp.Matrix([x, y])

        # Element access
        assert mat[0] == x
        assert mat[1] == y

        # Jacobian computation
        jac = mat.jacobian([x, y])
        assert jac.shape == (2, 2)
        assert jac == sp.Matrix([[1, 0], [0, 1]])

    def test_symbolic_symbol_creation(self):
        """Test creating SymbolicSymbol instances."""
        x: SymbolicSymbol = sp.Symbol("x", real=True)
        m: SymbolicSymbol = sp.Symbol("m", positive=True)
        z: SymbolicSymbol = sp.Symbol("z", complex=True)

        assert isinstance(x, sp.Symbol)
        assert isinstance(m, sp.Symbol)
        assert isinstance(z, sp.Symbol)

    def test_symbolic_symbol_assumptions(self):
        """Test symbol assumptions."""
        x: SymbolicSymbol = sp.Symbol("x", real=True)
        m: SymbolicSymbol = sp.Symbol("m", positive=True)

        assert x.is_real is True
        assert m.is_positive is True
        assert m.is_real is True  # positive implies real

    def test_symbol_dict_creation(self):
        """Test creating and using SymbolDict."""
        symbols: SymbolDict = {"x": sp.Symbol("x"), "y": sp.Symbol("y"), "z": sp.Symbol("z")}

        assert "x" in symbols
        assert "y" in symbols
        assert isinstance(symbols["x"], sp.Symbol)
        assert len(symbols) == 3

    def test_symbol_dict_usage(self):
        """Test practical usage of SymbolDict."""
        state_vars: SymbolDict = {
            "position": sp.Symbol("x", real=True),
            "velocity": sp.Symbol("v", real=True),
        }

        control_vars: SymbolDict = {"force": sp.Symbol("u", real=True)}

        # Build expression using symbols from dicts
        x = state_vars["position"]
        v = state_vars["velocity"]
        u = control_vars["force"]

        expr = x + v + u
        assert isinstance(expr, sp.Expr)


class TestSystemEquationTypes:
    """Test system equation type definitions."""

    def test_symbolic_state_equations_simple(self):
        """Test simple linear state equations."""
        x, v = sp.symbols("x v", real=True)
        u = sp.symbols("u", real=True)
        k, m = sp.symbols("k m", positive=True)

        f: SymbolicStateEquations = sp.Matrix([v, -k / m * x + u / m])

        assert isinstance(f, sp.Matrix)
        assert isinstance(f, SymbolicStateEquations)
        assert f.shape == (2, 1)
        assert f[0] == v
        assert f[1] == -k / m * x + u / m

    def test_symbolic_state_equations_nonlinear(self):
        """Test nonlinear state equations (pendulum)."""
        theta, omega = sp.symbols("theta omega", real=True)
        u = sp.symbols("u", real=True)
        m, l, g, b = sp.symbols("m l g b", positive=True)

        f: SymbolicStateEquations = sp.Matrix(
            [omega, -(g / l) * sp.sin(theta) - (b / m) * omega + u / (m * l**2)],
        )

        assert f.shape == (2, 1)
        assert f[0] == omega
        # Check structure of second equation
        assert sp.sin(theta) in f[1].atoms(sp.sin)

    def test_symbolic_state_equations_jacobian(self):
        """Test computing Jacobian of state equations."""
        x, y = sp.symbols("x y")
        f: SymbolicStateEquations = sp.Matrix([y, -x])

        # Compute Jacobian (A matrix for linearization)
        A = f.jacobian([x, y])
        assert A.shape == (2, 2)
        assert sp.Matrix([[0, 1], [-1, 0]]) == A

    def test_symbolic_output_equations_partial(self):
        """Test partial state observation."""
        theta, omega = sp.symbols("theta omega")

        # Measure position only
        h: SymbolicOutputEquations = sp.Matrix([theta])

        assert isinstance(h, sp.Matrix)
        assert isinstance(h, SymbolicOutputEquations)
        assert h.shape == (1, 1)
        assert h[0] == theta

    def test_symbolic_output_equations_full(self):
        """Test full state observation."""
        theta, omega = sp.symbols("theta omega")

        # Measure both position and velocity
        h: SymbolicOutputEquations = sp.Matrix([theta, omega])

        assert h.shape == (2, 1)
        assert h[0] == theta
        assert h[1] == omega

    def test_symbolic_output_equations_nonlinear(self):
        """Test nonlinear output equations."""
        x, y = sp.symbols("x y")

        # Polar coordinates from Cartesian
        h: SymbolicOutputEquations = sp.Matrix(
            [sp.sqrt(x**2 + y**2), sp.atan2(y, x)],  # Distance  # Angle
        )

        assert h.shape == (2, 1)
        assert sp.sqrt(x**2 + y**2) == h[0]

    def test_symbolic_diffusion_matrix_additive(self):
        """Test additive (constant) noise."""
        g: SymbolicDiffusionMatrix = sp.Matrix([[0.1], [0.2]])

        assert isinstance(g, sp.Matrix)
        assert isinstance(g, SymbolicDiffusionMatrix)
        assert g.shape == (2, 1)
        assert g[0] == 0.1
        assert g[1] == 0.2

    def test_symbolic_diffusion_matrix_multiplicative(self):
        """Test multiplicative (state-dependent) noise."""
        x, v = sp.symbols("x v")
        sigma = sp.Symbol("sigma", positive=True)

        # Noise proportional to velocity
        g: SymbolicDiffusionMatrix = sp.Matrix([[0], [sigma * v]])

        assert g.shape == (2, 1)
        assert g[0] == 0
        assert g[1] == sigma * v

    def test_symbolic_diffusion_matrix_multiple_sources(self):
        """Test multiple independent noise sources."""
        g: SymbolicDiffusionMatrix = sp.Matrix([[0.1, 0.0], [0.0, 0.2]])

        assert g.shape == (2, 2)
        assert g[0, 0] == 0.1
        assert g[1, 1] == 0.2
        assert g[0, 1] == 0.0
        assert g[1, 0] == 0.0


class TestParameterTypes:
    """Test parameter and substitution type definitions."""

    def test_parameter_dict_creation(self):
        """Test creating ParameterDict."""
        m, k, b = sp.symbols("m k b", positive=True)

        params: ParameterDict = {m: 1.5, k: 10.0, b: 0.5}

        assert isinstance(params, dict)
        assert params[m] == 1.5
        assert params[k] == 10.0
        assert params[b] == 0.5

    def test_parameter_dict_substitution(self):
        """Test using ParameterDict for substitution."""
        x = sp.Symbol("x")
        m, k, b = sp.symbols("m k b", positive=True)

        expr = m * x + k * x**2 + b
        params: ParameterDict = {m: 1.5, k: 10.0, b: 0.5}

        expr_numeric = expr.subs(params)

        # Parameters should be substituted
        result = expr_numeric.subs({x: 2.0})
        expected = 1.5 * 2.0 + 10.0 * 4.0 + 0.5  # 43.5
        assert float(result) == expected

    def test_substitution_dict_numerical(self):
        """Test SubstitutionDict with numerical values."""
        x, y = sp.symbols("x y")

        subs: SubstitutionDict = {x: 1.0, y: 2.0}

        expr = x + y
        result = expr.subs(subs)
        assert result == 3.0

    def test_substitution_dict_symbolic(self):
        """Test SubstitutionDict with symbolic values."""
        x, y, a, b = sp.symbols("x y a b")

        # Change of variables
        subs: SubstitutionDict = {x: a + b, y: a - b}

        expr = x * y
        result = expr.subs(subs)
        expected = (a + b) * (a - b)

        # Expand and compare
        assert sp.expand(result) == sp.expand(expected)
        assert sp.expand(result) == a**2 - b**2

    def test_substitution_dict_mixed(self):
        """Test SubstitutionDict with mixed values."""
        x, y, z = sp.symbols("x y z")

        subs: SubstitutionDict = {x: 1.0, y: z**2}  # Numerical  # Symbolic

        expr = x + y
        result = expr.subs(subs)
        assert result == 1.0 + z**2


class TestDerivativeTypes:
    """Test symbolic derivative type definitions."""

    def test_symbolic_jacobian_simple(self):
        """Test computing Jacobian for simple function."""
        x, y = sp.symbols("x y")
        f = sp.Matrix([x**2 + y, x * y])

        J: SymbolicJacobian = f.jacobian([x, y])

        assert isinstance(J, sp.Matrix)
        assert isinstance(J, SymbolicJacobian)
        assert J.shape == (2, 2)
        assert J[0, 0] == 2 * x
        assert J[0, 1] == 1
        assert J[1, 0] == y
        assert J[1, 1] == x

    def test_symbolic_jacobian_evaluation(self):
        """Test evaluating Jacobian at a point."""
        x, y = sp.symbols("x y")
        f = sp.Matrix([x**2, x * y])

        J: SymbolicJacobian = f.jacobian([x, y])
        J_eval = J.subs({x: 2.0, y: 3.0})

        assert J_eval[0, 0] == 4.0  # 2*2
        assert J_eval[0, 1] == 0
        assert J_eval[1, 0] == 3.0  # y
        assert J_eval[1, 1] == 2.0  # x

    def test_symbolic_jacobian_linearization(self):
        """Test Jacobian for system linearization."""
        theta, omega = sp.symbols("theta omega", real=True)
        g, l = sp.symbols("g l", positive=True)

        f = sp.Matrix([omega, -(g / l) * sp.sin(theta)])

        # State Jacobian (A matrix)
        A: SymbolicJacobian = f.jacobian([theta, omega])

        assert A.shape == (2, 2)
        assert A[0, 0] == 0
        assert A[0, 1] == 1
        assert A[1, 1] == 0
        # A[1,0] should be derivative of -(g/l)*sin(theta)
        assert A[1, 0] == -(g / l) * sp.cos(theta)

    def test_symbolic_gradient_simple(self):
        """Test computing gradient for scalar function."""
        x, y = sp.symbols("x y")
        f = x**2 + y**2

        grad: SymbolicGradient = sp.Matrix([sp.diff(f, x), sp.diff(f, y)])

        assert isinstance(grad, sp.Matrix)
        assert isinstance(grad, SymbolicGradient)
        assert grad.shape == (2, 1)
        assert grad[0] == 2 * x
        assert grad[1] == 2 * y

    def test_symbolic_gradient_lyapunov(self):
        """Test gradient for Lyapunov function."""
        x, y = sp.symbols("x y")
        V = x**2 + 2 * y**2  # Quadratic Lyapunov function

        grad_V: SymbolicGradient = sp.Matrix([sp.diff(V, x), sp.diff(V, y)])

        assert grad_V[0] == 2 * x
        assert grad_V[1] == 4 * y

        # Compute Lie derivative: V̇ = ∇V · f
        f = sp.Matrix([y, -x])  # System dynamics
        V_dot = grad_V.T @ f

        # Should be 2*x*y + 4*y*(-x) = -2*x*y
        assert sp.simplify(V_dot[0]) == -2 * x * y

    def test_symbolic_hessian_simple(self):
        """Test computing Hessian for scalar function."""
        x, y = sp.symbols("x y")
        f = x**2 + x * y + y**2

        # Compute via gradient Jacobian
        grad = sp.Matrix([sp.diff(f, x), sp.diff(f, y)])
        H: SymbolicHessian = grad.jacobian([x, y])

        assert isinstance(H, sp.Matrix)
        assert isinstance(H, SymbolicHessian)
        assert H.shape == (2, 2)
        assert H[0, 0] == 2
        assert H[0, 1] == 1
        assert H[1, 0] == 1
        assert H[1, 1] == 2

    def test_symbolic_hessian_symmetry(self):
        """Test that Hessian is symmetric."""
        x, y = sp.symbols("x y")
        f = x**3 + x * y**2 + y**3

        grad = sp.Matrix([sp.diff(f, x), sp.diff(f, y)])
        H: SymbolicHessian = grad.jacobian([x, y])

        # Hessian should be symmetric
        assert H[0, 1] == H[1, 0]

    def test_symbolic_hessian_eigenvalues(self):
        """Test computing eigenvalues of Hessian."""
        x, y = sp.symbols("x y")
        f = x**2 + y**2

        grad = sp.Matrix([sp.diff(f, x), sp.diff(f, y)])
        H: SymbolicHessian = grad.jacobian([x, y])

        # Hessian should be [[2, 0], [0, 2]]
        eigenvals = H.eigenvals()
        # Both eigenvalues should be 2
        assert 2 in eigenvals


class TestCompilation:
    """Test that symbolic types can be compiled to callables."""

    def test_lambdify_state_equations(self):
        """Test compiling state equations to NumPy function."""
        x, v = sp.symbols("x v")
        u = sp.symbols("u")
        m, k = sp.symbols("m k", positive=True)

        f: SymbolicStateEquations = sp.Matrix([v, -k / m * x + u / m])

        # Compile to NumPy - use separate args instead of lists
        f_func = sp.lambdify((x, v, u, m, k), f, "numpy")

        # Test with numerical values
        x_val = 1.0
        v_val = 0.0
        u_val = 0.0
        m_val = 1.0
        k_val = 10.0

        result = f_func(x_val, v_val, u_val, m_val, k_val)

        # Check result - convert to numpy array
        result_array = np.array(result).astype(float)

        # Check values
        assert result_array[0] == 0.0  # v = 0
        assert result_array[1] == -10.0  # -k/m * x = -10*1

    def test_lambdify_output_equations(self):
        """Test compiling output equations."""
        x, y = sp.symbols("x y")
        h: SymbolicOutputEquations = sp.Matrix([sp.sqrt(x**2 + y**2)])

        h_func = sp.lambdify((x, y), h, "numpy")

        result = h_func(3.0, 4.0)
        result_array = np.array(result).astype(float).flatten()

        assert np.isclose(result_array[0], 5.0)

    def test_lambdify_with_parameters(self):
        """Test lambdify with parameter substitution."""
        x = sp.Symbol("x")
        m, k = sp.symbols("m k", positive=True)

        expr = -k / m * x

        # First substitute parameters
        params: ParameterDict = {m: 2.0, k: 10.0}
        expr_with_params = expr.subs(params)

        # Then compile
        f = sp.lambdify(x, expr_with_params, "numpy")

        result = f(1.0)
        assert result == -5.0  # -10/2 * 1


class TestDocumentationExamples:
    """Test that all documentation examples work."""

    def test_module_docstring_example(self):
        """Test basic usage example from module docstring."""
        x, u = sp.symbols("x u", real=True)
        m, k = sp.symbols("m k", positive=True)

        # Define equation
        f: SymbolicExpression = -k / m * x + u / m

        # Create parameter dict
        params: ParameterDict = {m: 1.0, k: 10.0}

        # Evaluate
        result = f.subs(params).subs({x: 1.0, u: 0.0})
        assert float(result) == -10.0

    def test_state_equations_example(self):
        """Test state equations example from docstring."""
        theta, omega = sp.symbols("theta omega", real=True)
        u = sp.symbols("u", real=True)
        m, l, g, b = sp.symbols("m l g b", positive=True)

        f: SymbolicStateEquations = sp.Matrix(
            [omega, -(g / l) * sp.sin(theta) - (b / m) * omega + u / (m * l**2)],
        )

        assert f.shape == (2, 1)
        assert f[0] == omega

    def test_jacobian_example(self):
        """Test Jacobian example from docstring."""
        x, y = sp.symbols("x y")
        f = sp.Matrix([x**2 + y, x * y])

        J: SymbolicJacobian = f.jacobian([x, y])

        assert J.shape == (2, 2)
        assert J[0, 0] == 2 * x
        assert J[1, 1] == x


class TestTypeIdentity:
    """Test that type aliases work correctly."""

    def test_types_are_sympy_classes(self):
        """Verify types are actually SymPy classes."""
        assert SymbolicExpression is sp.Expr
        assert SymbolicMatrix is sp.Matrix
        assert SymbolicSymbol is sp.Symbol

    def test_isinstance_checks(self):
        """Test isinstance checks work with type aliases."""
        x = sp.Symbol("x")
        expr = x**2
        mat = sp.Matrix([x, x**2])

        assert isinstance(x, SymbolicSymbol)
        assert isinstance(expr, SymbolicExpression)
        assert isinstance(mat, SymbolicMatrix)


class TestPracticalUseCases:
    """Test realistic use cases for symbolic types."""

    def test_pendulum_system_definition(self):
        """Test defining complete pendulum system."""
        # Define symbols
        theta, omega = sp.symbols("theta omega", real=True)
        u = sp.symbols("u", real=True)
        m, l, g, b = sp.symbols("m l g b", positive=True)

        # State equations
        f: SymbolicStateEquations = sp.Matrix(
            [omega, -(g / l) * sp.sin(theta) - (b / (m * l**2)) * omega + u / (m * l**2)],
        )

        # Output (measure angle only)
        h: SymbolicOutputEquations = sp.Matrix([theta])

        # Parameters
        params: ParameterDict = {m: 0.15, l: 0.5, g: 9.81, b: 0.1}  # kg  # m  # m/s²  # damping

        # Linearize at upright equilibrium
        eq_point = {theta: sp.pi, omega: 0, u: 0}
        A = f.jacobian([theta, omega]).subs(eq_point).subs(params)
        B = f.jacobian([u]).subs(eq_point).subs(params)
        C = h.jacobian([theta, omega]).subs(eq_point).subs(params)

        assert A.shape == (2, 2)
        assert B.shape == (2, 1)
        assert C.shape == (1, 2)

    def test_stochastic_system_definition(self):
        """Test defining system with stochastic dynamics."""
        x, v = sp.symbols("x v", real=True)
        u = sp.symbols("u", real=True)
        m, k, sigma = sp.symbols("m k sigma", positive=True)

        # Deterministic drift
        f: SymbolicStateEquations = sp.Matrix([v, -k / m * x + u / m])

        # Stochastic diffusion (additive noise)
        g: SymbolicDiffusionMatrix = sp.Matrix([[0], [sigma]])

        assert f.shape == (2, 1)
        assert g.shape == (2, 1)

        # Compute linearization
        A = f.jacobian([x, v])
        B = f.jacobian([u])
        G = g  # Additive noise, no linearization needed

        assert A.shape == (2, 2)
        assert B.shape == (2, 1)
        assert G.shape == (2, 1)
