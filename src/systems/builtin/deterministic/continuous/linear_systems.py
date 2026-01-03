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
Linear Systems - Educational and Testing Examples
==================================================

Provides simple linear dynamical systems for testing, verification,
and educational purposes.

Systems included:
- LinearSystem: First-order scalar system (controlled)
- AutonomousLinearSystem: First-order scalar system (no control)
- LinearSystem2D: Second-order system (controlled)

All systems have analytical solutions for verification.
"""


import sympy as sp

from src.systems.base.symbolic_dynamical_system import SymbolicDynamicalSystem


class LinearSystem(SymbolicDynamicalSystem):
    """
    First-order linear system: dx/dt = -a*x + b*u

    Canonical example for testing and learning.

    Parameters
    ----------
    a : float, default=1.0
        State coefficient (positive = stable)
    b : float, default=1.0
        Control gain

    Examples
    --------
    >>> system = LinearSystem(a=2.0, b=1.0)
    >>> x = np.array([1.0])
    >>> u = np.array([0.5])
    >>> dx = system(x, u)  # -2*1 + 1*0.5 = -1.5
    """

    def define_system(self, a: float = 1.0, b: float = 1.0):
        """Define linear system dynamics."""
        # Symbolic variables
        x = sp.symbols("x", real=True)
        u = sp.symbols("u", real=True)
        a_sym = sp.symbols("a", real=True)
        b_sym = sp.symbols("b", real=True)

        # System definition
        self.state_vars = [x]
        self.control_vars = [u]
        self._f_sym = sp.Matrix([[-a_sym * x + b_sym * u]])
        self.parameters = {a_sym: a, b_sym: b}
        self.order = 1

        # NOTE: Don't add equilibrium in define_system()!
        # The EquilibriumHandler is not initialized yet during define_system()
        # Users should add equilibria AFTER system creation if needed


class AutonomousLinearSystem(SymbolicDynamicalSystem):
    """
    Autonomous first-order linear system: dx/dt = -a*x

    No control input (nu=0).

    Parameters
    ----------
    a : float, default=1.0
        Decay rate (positive = stable)

    Examples
    --------
    >>> system = AutonomousLinearSystem(a=1.0)
    >>> system.nu
    0
    >>> dx = system(np.array([1.0]), u=None)  # -1*1 = -1
    """

    def define_system(self, a: float = 1.0):
        """Define autonomous linear dynamics."""
        # Symbolic variables
        x = sp.symbols("x", real=True)
        a_sym = sp.symbols("a", real=True, positive=True)

        # System definition
        self.state_vars = [x]
        self.control_vars = []  # ← Empty for autonomous
        self._f_sym = sp.Matrix([[-a_sym * x]])
        self.parameters = {a_sym: a}
        self.order = 1


class LinearSystem2D(SymbolicDynamicalSystem):
    """
    Two-dimensional linear system.

    Dynamics:
        dx1/dt = a11*x1 + a12*x2 + b1*u
        dx2/dt = a21*x1 + a22*x2 + b2*u

    Parameters
    ----------
    a11, a12, a21, a22 : float
        State matrix coefficients
    b1, b2 : float
        Control gain coefficients

    Examples
    --------
    >>> # Stable spiral
    >>> system = LinearSystem2D(
    ...     a11=-1, a12=2, a21=-2, a22=-1, b1=1, b2=0
    ... )
    >>> eigenvalues = np.linalg.eigvals(np.array([[-1, 2], [-2, -1]]))
    >>> # λ = -1 ± 2j (stable spiral)
    """

    def define_system(
        self,
        a11: float = -1.0,
        a12: float = 0.0,
        a21: float = 0.0,
        a22: float = -1.0,
        b1: float = 1.0,
        b2: float = 0.0,
    ):
        """Define 2D linear dynamics."""
        # State and control
        x1, x2 = sp.symbols("x1 x2", real=True)
        u = sp.symbols("u", real=True)

        # Parameters
        a11_sym = sp.symbols("a11", real=True)
        a12_sym = sp.symbols("a12", real=True)
        a21_sym = sp.symbols("a21", real=True)
        a22_sym = sp.symbols("a22", real=True)
        b1_sym = sp.symbols("b1", real=True)
        b2_sym = sp.symbols("b2", real=True)

        self.state_vars = [x1, x2]
        self.control_vars = [u]

        # Dynamics
        self._f_sym = sp.Matrix(
            [
                [a11_sym * x1 + a12_sym * x2 + b1_sym * u],
                [a21_sym * x1 + a22_sym * x2 + b2_sym * u],
            ],
        )

        self.parameters = {
            a11_sym: a11,
            a12_sym: a12,
            a21_sym: a21,
            a22_sym: a22,
            b1_sym: b1,
            b2_sym: b2,
        }
        self.order = 1


# ============================================================================
# Convenience Factory Functions
# ============================================================================


def create_stable_system(time_constant: float = 1.0) -> LinearSystem:
    """
    Create stable linear system with specified time constant.

    Parameters
    ----------
    time_constant : float
        Time constant τ in seconds (a = 1/τ)

    Returns
    -------
    LinearSystem
        System with decay rate a = 1/τ

    Examples
    --------
    >>> # Fast system (settles in ~2 seconds)
    >>> fast = create_stable_system(time_constant=0.5)
    >>> # a = 1/0.5 = 2.0
    """
    a = 1.0 / time_constant
    return LinearSystem(a=a, b=1.0)


def create_critically_damped_2d() -> LinearSystem2D:
    """
    Create critically damped 2D system (no oscillation).

    Returns
    -------
    LinearSystem2D
        System with repeated real eigenvalues
    """
    return LinearSystem2D(a11=-2.0, a12=1.0, a21=0.0, a22=-2.0, b1=1.0, b2=1.0)


def create_oscillatory_2d(omega: float = 2.0, damping: float = 0.1) -> LinearSystem2D:
    """
    Create oscillatory 2D system (damped spiral).

    Parameters
    ----------
    omega : float
        Natural frequency (rad/s)
    damping : float
        Damping coefficient

    Returns
    -------
    LinearSystem2D
        System with complex eigenvalues λ = -damping ± j*omega
    """
    return LinearSystem2D(a11=-damping, a12=omega, a21=-omega, a22=-damping, b1=1.0, b2=0.0)
