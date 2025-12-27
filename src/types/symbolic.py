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
Symbolic Types

Defines types for symbolic mathematics using SymPy:
- Symbolic expressions and matrices
- Symbol dictionaries
- State and output equations
- Parameter and substitution dictionaries
- Symbolic derivatives (Jacobians, gradients, Hessians)

These types enable the framework's core "write once, run anywhere"
paradigm by allowing symbolic system specification that can be
automatically compiled to NumPy, PyTorch, or JAX.

Mathematical Context
-------------------
Symbolic systems are defined using SymPy:

State equations:
    dx/dt = f_sym(x, u, params)

Output equations:
    y = h_sym(x, u)

Diffusion (stochastic):
    dW scaled by g_sym(x, u)

These symbolic forms are:
1. Stored for documentation/analysis
2. Differentiated to compute Jacobians
3. Compiled to callable functions via lambdify
4. Executed on any backend (NumPy/PyTorch/JAX)

Usage
-----
>>> from src.types.symbolic import (
...     SymbolicExpression,
...     SymbolicMatrix,
...     ParameterDict,
... )
>>>
>>> import sympy as sp
>>>
>>> # Define symbols
>>> x, u = sp.symbols('x u', real=True)
>>> m, k = sp.symbols('m k', positive=True)
>>>
>>> # Define equation
>>> f: SymbolicExpression = -k/m * x + u/m
>>>
>>> # Create parameter dict
>>> params: ParameterDict = {m: 1.0, k: 10.0}
>>>
>>> # Evaluate
>>> result = f.subs(params).subs({x: 1.0, u: 0.0})
>>> print(result)  # -10.0

Design Philosophy
----------------
These are TYPE DEFINITIONS only - no implementation logic.
They provide:
- Clear semantic naming (SymbolicStateEquations vs sp.Matrix)
- Self-documenting code (intent is clear from type name)
- Consistent vocabulary across the framework
- Foundation for type-safe symbolic system specification
"""

from typing import Dict, List, Union

import sympy as sp

# Note: No imports from other type modules needed
# This module is foundational and self-contained


# ============================================================================
# Basic Symbolic Types
# ============================================================================

SymbolicExpression = sp.Expr
"""
Single symbolic expression.

Any SymPy expression representing a mathematical formula.

Examples
--------
>>> import sympy as sp
>>> x, u = sp.symbols('x u')
>>> expr: SymbolicExpression = x**2 + sp.sin(u)
>>> 
>>> # Differentiate
>>> dexpr_dx = sp.diff(expr, x)  # 2*x
>>> 
>>> # Evaluate
>>> result = expr.subs({x: 1.0, u: 0.0})  # 1.0
>>> 
>>> # Simplify
>>> simplified: SymbolicExpression = sp.simplify(expr)
"""

SymbolicMatrix = sp.Matrix
"""
Matrix of symbolic expressions.

Used for vector-valued functions like state equations.

Shape: Can be any (m, n), commonly (n, 1) for column vectors

Examples
--------
>>> import sympy as sp
>>> x1, x2 = sp.symbols('x1 x2')
>>> u = sp.symbols('u')
>>> 
>>> # Dynamics: dx/dt = f(x, u)
>>> f: SymbolicMatrix = sp.Matrix([
...     x2,
...     -x1 + u
... ])
>>> 
>>> # Jacobian
>>> A = f.jacobian([x1, x2])  # 2x2 matrix
>>> print(A)
>>> # Matrix([[0, 1], [-1, 0]])
>>> 
>>> # Access elements
>>> f[0]  # x2
>>> f[1]  # -x1 + u
"""

SymbolicSymbol = sp.Symbol
"""
Single symbolic variable.

Used to represent state variables, control inputs, parameters, etc.

Examples
--------
>>> import sympy as sp
>>> 
>>> # State variable
>>> x: SymbolicSymbol = sp.Symbol('x', real=True)
>>> 
>>> # Angle (real-valued)
>>> theta: SymbolicSymbol = sp.Symbol('theta', real=True)
>>> 
>>> # Mass (positive)
>>> m: SymbolicSymbol = sp.Symbol('m', positive=True)
>>> 
>>> # Complex variable
>>> z: SymbolicSymbol = sp.Symbol('z', complex=True)
>>> 
>>> # Create multiple symbols at once
>>> x, y, z = sp.symbols('x y z', real=True)
"""

SymbolDict = Dict[str, sp.Symbol]
"""
Dictionary mapping variable names to symbols.

Keys are string names, values are SymPy symbols.
Useful for organizing symbols by category (states, controls, params).

Examples
--------
>>> import sympy as sp
>>> 
>>> # State variables
>>> state_vars: SymbolDict = {
...     'position': sp.Symbol('x', real=True),
...     'velocity': sp.Symbol('v', real=True)
... }
>>> 
>>> # Control inputs
>>> control_vars: SymbolDict = {
...     'force': sp.Symbol('u', real=True)
... }
>>> 
>>> # Parameters
>>> params: SymbolDict = {
...     'mass': sp.Symbol('m', positive=True),
...     'damping': sp.Symbol('b', positive=True)
... }
>>> 
>>> # Access
>>> x = state_vars['position']
>>> m = params['mass']
"""

SymbolicExpressionInput = Union[sp.Expr, List[sp.Expr], sp.Matrix]
"""
Flexible symbolic expression input.

Accepts any of:
- Single expression (sp.Expr)
- List of expressions (list[sp.Expr]) 
- Matrix of expressions (sp.Matrix)

This is the typical input type for code generation and compilation
functions that need to handle various symbolic forms.

Examples
--------
>>> import sympy as sp
>>> 
>>> # Scalar
>>> expr1: SymbolicExpressionInput = x**2 + y
>>> 
>>> # Vector (list)
>>> expr2: SymbolicExpressionInput = [x, y, x*y]
>>> 
>>> # Vector (matrix)
>>> expr3: SymbolicExpressionInput = sp.Matrix([x, y, x*y])
>>> 
>>> # All three can be passed to code generation
>>> from codegen_utils import generate_function
>>> f1 = generate_function(expr1, [x, y])
>>> f2 = generate_function(expr2, [x, y])
>>> f3 = generate_function(expr3, [x, y])
"""

# ============================================================================
# System Equation Types
# ============================================================================

SymbolicStateEquations = sp.Matrix
"""
Symbolic state equations: f(x, u, params).

Column vector representing dx/dt (continuous) or x[k+1] (discrete).
Each row is one state derivative or next-state equation.

Shape: (nx, 1)

Continuous systems:
    dx/dt = f(x, u) where each element is dxᵢ/dt

Discrete systems:
    x[k+1] = f(x[k], u[k]) where each element is xᵢ[k+1]

Examples
--------
>>> import sympy as sp
>>> 
>>> # Simple pendulum
>>> theta, omega = sp.symbols('theta omega', real=True)
>>> u = sp.symbols('u', real=True)
>>> m, l, g, b = sp.symbols('m l g b', positive=True)
>>> 
>>> # Continuous dynamics: [dθ/dt, dω/dt]
>>> f: SymbolicStateEquations = sp.Matrix([
...     omega,
...     -(g/l)*sp.sin(theta) - (b/m)*omega + u/(m*l**2)
... ])
>>> 
>>> # Compile to callable function
>>> from sympy import lambdify
>>> f_func = lambdify(
...     ([theta, omega], u, [m, l, g, b]),
...     f,
...     'numpy'
... )
>>> 
>>> # Linear system
>>> x1, x2, u = sp.symbols('x1 x2 u')
>>> A11, A12, A21, A22 = sp.symbols('A11 A12 A21 A22')
>>> B1, B2 = sp.symbols('B1 B2')
>>> 
>>> f_linear: SymbolicStateEquations = sp.Matrix([
...     A11*x1 + A12*x2 + B1*u,
...     A21*x1 + A22*x2 + B2*u
... ])
"""

SymbolicOutputEquations = sp.Matrix
"""
Symbolic output equations: h(x, u).

Column vector representing measurement/output.
Each row is one output equation.

Shape: (ny, 1)

Output equation:
    y = h(x, u) where each element is yᵢ

Examples
--------
>>> import sympy as sp
>>> 
>>> # Measure position only (partial observation)
>>> theta, omega = sp.symbols('theta omega')
>>> h: SymbolicOutputEquations = sp.Matrix([theta])
>>> 
>>> # Measure position and velocity (full state observation)
>>> h_full: SymbolicOutputEquations = sp.Matrix([theta, omega])
>>> 
>>> # Nonlinear measurement
>>> x, y = sp.symbols('x y')
>>> h_nonlinear: SymbolicOutputEquations = sp.Matrix([
...     sp.sqrt(x**2 + y**2),  # Distance from origin
...     sp.atan2(y, x)          # Angle
... ])
>>> 
>>> # Output depends on control (feedthrough)
>>> u = sp.symbols('u')
>>> h_feedthrough: SymbolicOutputEquations = sp.Matrix([
...     x + 0.5*u  # Direct feedthrough term
... ])
"""

SymbolicDiffusionMatrix = sp.Matrix
"""
Symbolic diffusion matrix: g(x, u).

Matrix scaling Brownian motion in stochastic differential equations (SDEs).
Each column corresponds to one noise source.

Shape: (nx, nw)

Continuous SDE:
    dx = f(x,u)dt + g(x,u)dW
    where dW is nw-dimensional Brownian motion

Discrete stochastic:
    x[k+1] = f(x,u) + g(x,u)w[k]
    where w[k] ~ N(0, I) is standard normal noise

Examples
--------
>>> import sympy as sp
>>> 
>>> # Additive noise (constant, state-independent)
>>> g_additive: SymbolicDiffusionMatrix = sp.Matrix([
...     [0.1],
...     [0.2]
... ])
>>> 
>>> # Multiplicative noise (state-dependent)
>>> x, v = sp.symbols('x v')
>>> sigma = sp.Symbol('sigma', positive=True)
>>> g_multiplicative: SymbolicDiffusionMatrix = sp.Matrix([
...     [0],
...     [sigma * v]  # Noise proportional to velocity
... ])
>>> 
>>> # Multiple noise sources
>>> g_multi: SymbolicDiffusionMatrix = sp.Matrix([
...     [0.1, 0.0],
...     [0.0, 0.2]
... ])  # Independent noise on each state
>>> 
>>> # Control-dependent noise
>>> u = sp.symbols('u')
>>> g_control: SymbolicDiffusionMatrix = sp.Matrix([
...     [0],
...     [sigma * sp.sqrt(sp.Abs(u))]  # Noise from control effort
... ])
"""


# ============================================================================
# Parameter and Substitution Types
# ============================================================================

ParameterDict = Dict[sp.Symbol, float]
"""
Dictionary mapping parameter symbols to numerical values.

Used for evaluating symbolic expressions with specific parameter values.
Keys are SymPy symbols, values are float constants.

Examples
--------
>>> import sympy as sp
>>> 
>>> # Define parameters
>>> m, k, b = sp.symbols('m k b', positive=True)
>>> params: ParameterDict = {
...     m: 1.5,   # Mass (kg)
...     k: 10.0,  # Spring constant (N/m)
...     b: 0.5    # Damping coefficient (N⋅s/m)
... }
>>> 
>>> # Substitute into expression
>>> x = sp.Symbol('x')
>>> expr = m*x + k*x**2 + b
>>> expr_numeric = expr.subs(params)
>>> print(expr_numeric)  # 1.5*x + 10.0*x**2 + 0.5
>>> 
>>> # Evaluate with state value
>>> result = expr_numeric.subs({x: 2.0})
>>> print(result)  # 43.5
>>> 
>>> # Common pattern: separate params from state/control
>>> theta, omega, u = sp.symbols('theta omega u')
>>> g, l = sp.symbols('g l', positive=True)
>>> 
>>> f = sp.Matrix([omega, -(g/l)*sp.sin(theta) + u/(m*l**2)])
>>> params: ParameterDict = {g: 9.81, l: 0.5, m: 0.15}
>>> f_with_params = f.subs(params)
"""

SubstitutionDict = Dict[sp.Symbol, Union[float, sp.Expr]]
"""
Dictionary for symbolic substitutions.

Can substitute symbols with numerical values OR other symbolic expressions.
More general than ParameterDict.

Keys are SymPy symbols, values are floats or SymPy expressions.

Examples
--------
>>> import sympy as sp
>>> 
>>> # Numerical substitution
>>> x, y = sp.symbols('x y')
>>> subs1: SubstitutionDict = {x: 1.0, y: 2.0}
>>> expr = x + y
>>> result = expr.subs(subs1)  # 3.0
>>> 
>>> # Symbolic substitution (change of variables)
>>> a, b = sp.symbols('a b')
>>> subs2: SubstitutionDict = {
...     x: a + b,
...     y: a - b
... }
>>> expr = x * y
>>> result = expr.subs(subs2)  # (a + b)*(a - b) = a² - b²
>>> 
>>> # Mixed substitution
>>> z = sp.symbols('z')
>>> subs3: SubstitutionDict = {
...     x: 1.0,      # Numerical
...     y: z**2      # Symbolic
... }
>>> 
>>> # Common use: equilibrium linearization
>>> x_eq, u_eq = 0.0, 0.0  # Equilibrium point
>>> theta, omega, u = sp.symbols('theta omega u')
>>> subs_eq: SubstitutionDict = {
...     theta: x_eq,
...     omega: 0.0,
...     u: u_eq
... }
"""


# ============================================================================
# Symbolic Derivative Types
# ============================================================================

SymbolicJacobian = sp.Matrix
"""
Jacobian matrix: ∂f/∂x.

Matrix of partial derivatives of a vector function.

Shape: 
- For f: ℝⁿ → ℝᵐ, Jacobian ∂f/∂x: (m, n)
- Element (i,j) is ∂fᵢ/∂xⱼ

Usage:
- Linearization: A = ∂f/∂x at (x_eq, u_eq)
- Sensitivity analysis: how outputs change with inputs
- Optimization: gradient information

Examples
--------
>>> import sympy as sp
>>> 
>>> # Nonlinear system
>>> x, y = sp.symbols('x y')
>>> f = sp.Matrix([x**2 + y, x*y])
>>> 
>>> J: SymbolicJacobian = f.jacobian([x, y])
>>> print(J)
>>> # Matrix([[2*x, 1],
>>> #         [y,   x]])
>>> 
>>> # Evaluate at a point
>>> J_at_point = J.subs({x: 1.0, y: 2.0})
>>> # Matrix([[2.0, 1.0],
>>> #         [2.0, 1.0]])
>>> 
>>> # For system dynamics linearization
>>> theta, omega = sp.symbols('theta omega', real=True)
>>> g, l = sp.symbols('g l', positive=True)
>>> 
>>> f_dynamics = sp.Matrix([
...     omega,
...     -(g/l)*sp.sin(theta)
... ])
>>> 
>>> # State Jacobian (A matrix)
>>> A: SymbolicJacobian = f_dynamics.jacobian([theta, omega])
>>> # Matrix([[0,              1],
>>> #         [-(g/l)*cos(theta), 0]])
>>> 
>>> # Evaluate at upright equilibrium (theta=π, omega=0)
>>> A_upright = A.subs({theta: sp.pi, omega: 0})
>>> # Matrix([[0, 1],
>>> #         [g/l, 0]])  # Unstable!
"""

SymbolicGradient = sp.Matrix
"""
Gradient vector: ∇f.

Column vector of partial derivatives for a scalar function.

Shape:
- For f: ℝⁿ → ℝ, gradient ∇f: (n, 1)
- Element i is ∂f/∂xᵢ

Usage:
- Optimization: direction of steepest ascent
- Lyapunov functions: V̇ = ∇V · f
- Cost function derivatives

Examples
--------
>>> import sympy as sp
>>> 
>>> # Scalar function
>>> x, y = sp.symbols('x y')
>>> f = x**2 + y**2  # Distance squared from origin
>>> 
>>> grad: SymbolicGradient = sp.Matrix([
...     sp.diff(f, x),
...     sp.diff(f, y)
... ])
>>> print(grad)
>>> # Matrix([[2*x],
>>> #         [2*y]])
>>> 
>>> # Lyapunov function gradient
>>> V = x**2 + 2*y**2  # Quadratic Lyapunov function
>>> grad_V: SymbolicGradient = sp.Matrix([
...     sp.diff(V, x),  # 2*x
...     sp.diff(V, y)   # 4*y
... ])
>>> 
>>> # Compute Lie derivative: V̇ = ∇V · f
>>> f_system = sp.Matrix([y, -x])  # System dynamics
>>> V_dot = grad_V.T @ f_system
>>> # Result: 2*x*y + 4*y*(-x) = 2*x*y - 4*x*y = -2*x*y
>>> 
>>> # Cost function for optimal control
>>> x1, x2, u = sp.symbols('x1 x2 u')
>>> J = x1**2 + x2**2 + 0.1*u**2
>>> grad_J_x = sp.Matrix([sp.diff(J, x1), sp.diff(J, x2)])
>>> grad_J_u = sp.diff(J, u)
"""

SymbolicHessian = sp.Matrix
"""
Hessian matrix: ∇²f.

Matrix of second-order partial derivatives for a scalar function.

Shape:
- For f: ℝⁿ → ℝ, Hessian ∇²f: (n, n)
- Element (i,j) is ∂²f/∂xᵢ∂xⱼ
- Symmetric if f is twice continuously differentiable

Usage:
- Optimization: local curvature, detect minima/maxima
- Stability analysis: local behavior near equilibria
- Control: Riccati equation, cost-to-go

Examples
--------
>>> import sympy as sp
>>> 
>>> # Quadratic function
>>> x, y = sp.symbols('x y')
>>> f = x**2 + x*y + y**2
>>> 
>>> # Compute Hessian via gradient Jacobian
>>> grad = sp.Matrix([sp.diff(f, x), sp.diff(f, y)])
>>> H: SymbolicHessian = grad.jacobian([x, y])
>>> print(H)
>>> # Matrix([[2, 1],
>>> #         [1, 2]])
>>> 
>>> # Check positive definiteness (for minima)
>>> eigenvals = H.eigenvals()
>>> # Both eigenvalues positive → f has minimum
>>> 
>>> # Lyapunov function Hessian
>>> V = x**2 + 2*x*y + 3*y**2
>>> grad_V = sp.Matrix([sp.diff(V, x), sp.diff(V, y)])
>>> H_V: SymbolicHessian = grad_V.jacobian([x, y])
>>> # Matrix([[2, 2],
>>> #         [2, 6]])
>>> 
>>> # For control: Riccati equation involves Hessian of cost
>>> # Cost: J = x'Qx + u'Ru
>>> # Hessian w.r.t. x is 2Q
>>> 
>>> # Newton's method for optimization uses Hessian
>>> # x_next = x_current - H⁻¹(x_current) @ grad(x_current)
"""


# ============================================================================
# Export All
# ============================================================================

__all__ = [
    # Basic types
    "SymbolicExpression",
    "SymbolicMatrix",
    "SymbolicSymbol",
    "SymbolDict",
    "SymbolicExpressionInput",
    # System equations
    "SymbolicStateEquations",
    "SymbolicOutputEquations",
    "SymbolicDiffusionMatrix",
    # Parameters
    "ParameterDict",
    "SubstitutionDict",
    # Derivatives
    "SymbolicJacobian",
    "SymbolicGradient",
    "SymbolicHessian",
]
