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
Discrete-time Symbolic Dynamical System

Represents difference equations: x[k+1] = f(x[k], u[k])
"""

from typing import List, Optional, Dict, Any
import sympy as sp
import numpy as np

from src.systems.base.symbolic_dynamical_system import SymbolicDynamicalSystem


class DiscreteSymbolicSystem(SymbolicDynamicalSystem):
    """
    Symbolic discrete-time dynamical system.
    
    Represents difference equations of the form:
        x[k+1] = f(x[k], u[k])
        y[k] = h(x[k])
    
    where:
        - x[k] is the state at discrete time k
        - u[k] is the control input
        - y[k] is the output
    
    This class inherits from SymbolicDynamicalSystem but interprets
    state_equations as difference equations (x[k+1]) rather than
    differential equations (dx/dt). Most functionality is reused from
    the parent class with minimal semantic differences.
    
    Key Differences from Continuous Systems
    ---------------------------------------
    - **Dynamics Interpretation**: 
      - Continuous: _f_sym represents dx/dt (rate of change)
      - Discrete: _f_sym represents x[k+1] (next state directly)
    
    - **Time Semantics**:
      - Continuous: Time t ∈ ℝ (real-valued)
      - Discrete: Time k ∈ ℤ (integer-valued)
    
    - **Integration**:
      - Continuous: Requires numerical integration (Euler, RK4, etc.)
      - Discrete: Direct evaluation (no integration needed)
    
    - **Linearization**:
      - Continuous: A = ∂f/∂x gives continuous-time A matrix
      - Discrete: A = ∂f/∂x gives discrete-time state transition matrix
    
    Implementation Strategy
    ----------------------
    This class uses the **minimal inheritance approach** - it inherits
    from SymbolicDynamicalSystem and reuses almost all functionality
    with just a flag to indicate discrete-time interpretation.
    
    The parent class's methods work correctly because:
    - Symbolic manipulation is the same (Jacobians, substitution, etc.)
    - Code generation is the same (lambdify works for both)
    - Backend management is the same (NumPy/PyTorch/JAX)
    - The only difference is semantic interpretation
    
    Future Refactoring Path
    -----------------------
    If this minimal approach becomes limiting, consider refactoring to:
    
    ```python
    class SymbolicSystem(ABC):
        # Common symbolic machinery:
        # - Variable management
        # - Parameter handling
        # - Code generation
        # - Backend management
        # - Validation
        pass
    
    class ContinuousSymbolicSystem(SymbolicSystem):
        # _f_sym = dx/dt (differential equations)
        # Requires integrator for simulation
        pass
    
    class DiscreteSymbolicSystem(SymbolicSystem):
        # _f_sym = x[k+1] (difference equations)
        # Direct evaluation (no integration)
        pass
    ```
    
    This would cleanly separate concerns but requires significant
    refactoring. The current approach minimizes code duplication
    while we validate the design.
    
    Parameters
    ----------
    *args : tuple
        Positional arguments passed to define_system()
    **kwargs : dict
        Keyword arguments passed to define_system()
    
    Attributes (Same as Parent)
    --------------------------
    state_vars : List[sp.Symbol]
        State variables x[k]
    control_vars : List[sp.Symbol]
        Control input variables u[k]
    output_vars : List[sp.Symbol]
        Output variables y[k]
    _f_sym : sp.Matrix
        Symbolic difference equations: x[k+1] = f(x[k], u[k])
    _h_sym : Optional[sp.Matrix]
        Symbolic output: y[k] = h(x[k])
    parameters : Dict[sp.Symbol, float]
        System parameters
    order : int
        System order (usually 1 for discrete systems)
    
    Attributes (Discrete-Specific)
    ------------------------------
    _is_discrete : bool
        Flag indicating discrete-time system (always True)
    
    Examples
    --------
    Linear discrete-time system:
    >>> class DiscreteLinearSystem(DiscreteSymbolicSystem):
    ...     '''Discrete-time linear system: x[k+1] = A*x[k] + B*u[k]'''
    ...     
    ...     def define_system(self, a11=0.9, a12=0.1, a21=-0.1, a22=0.8, b=1.0):
    ...         x1, x2 = sp.symbols('x1 x2', real=True)
    ...         u = sp.symbols('u', real=True)
    ...         
    ...         a11_sym, a12_sym = sp.symbols('a11 a12', real=True)
    ...         a21_sym, a22_sym = sp.symbols('a21 a22', real=True)
    ...         b_sym = sp.symbols('b', real=True)
    ...         
    ...         # Define difference equations: x[k+1] = A*x[k] + B*u[k]
    ...         self.state_vars = [x1, x2]
    ...         self.control_vars = [u]
    ...         self._f_sym = sp.Matrix([
    ...             a11_sym*x1 + a12_sym*x2,        # x1[k+1]
    ...             a21_sym*x1 + a22_sym*x2 + b_sym*u  # x2[k+1]
    ...         ])
    ...         self.parameters = {
    ...             a11_sym: a11, a12_sym: a12,
    ...             a21_sym: a21, a22_sym: a22,
    ...             b_sym: b
    ...         }
    ...         self.order = 1
    
    >>> system = DiscreteLinearSystem()
    >>> 
    >>> # Evaluate next state directly (no integration needed)
    >>> x_k = np.array([1.0, 0.0])
    >>> u_k = np.array([0.0])
    >>> x_next = system(x_k, u_k)  # Returns x[k+1]
    >>> 
    >>> # Linearization gives discrete-time A, B matrices
    >>> A, B = system.linearized_dynamics(x_k, u_k)
    >>> # A is the discrete state transition matrix
    
    Nonlinear discrete system:
    >>> class DiscreteLogisticMap(DiscreteSymbolicSystem):
    ...     '''Logistic map: x[k+1] = r*x[k]*(1 - x[k])'''
    ...     
    ...     def define_system(self, r=3.5):
    ...         x = sp.symbols('x', real=True, positive=True)
    ...         r_sym = sp.symbols('r', positive=True)
    ...         
    ...         # Autonomous discrete system (no control)
    ...         self.state_vars = [x]
    ...         self.control_vars = []
    ...         self._f_sym = sp.Matrix([r_sym * x * (1 - x)])
    ...         self.parameters = {r_sym: r}
    ...         self.order = 1
    
    >>> system = DiscreteLogisticMap(r=3.8)
    >>> x_k = np.array([0.5])
    >>> x_next = system(x_k)  # Autonomous, no control needed
    
    Discrete-time control system:
    >>> class DigitalController(DiscreteSymbolicSystem):
    ...     '''Digital PID-like controller with sample-and-hold'''
    ...     
    ...     def define_system(self, kp=1.0, ki=0.1, dt=0.01):
    ...         # States: [error_integral, previous_error]
    ...         ei, e_prev = sp.symbols('e_i e_prev', real=True)
    ...         # Control: current error
    ...         e = sp.symbols('e', real=True)
    ...         
    ...         kp_sym, ki_sym, dt_sym = sp.symbols('k_p k_i dt', positive=True)
    ...         
    ...         self.state_vars = [ei, e_prev]
    ...         self.control_vars = [e]
    ...         self._f_sym = sp.Matrix([
    ...             ei + dt_sym * e,  # Integrate error
    ...             e                 # Store current error
    ...         ])
    ...         self.parameters = {kp_sym: kp, ki_sym: ki, dt_sym: dt}
    ...         self.order = 1
    ...         
    ...         # Output: PID control signal
    ...         self._h_sym = sp.Matrix([
    ...             kp_sym * e + ki_sym * ei  # Proportional + Integral
    ...         ])
    
    Notes
    -----
    - All parent class methods (forward, linearized_dynamics, etc.) work correctly
    - The `order` attribute is typically 1 for discrete systems (first-order difference)
    - Higher-order discrete systems can be represented but are less common
    - Use `DiscreteTimeSystem` wrapper for simulation with numerical integrators
    
    See Also
    --------
    SymbolicDynamicalSystem : Parent class for continuous systems
    DiscreteTimeSystem : Wrapper for numerical simulation
    StochasticDiscreteSystem : Future extension for stochastic difference equations
    """
    
    def __init__(self, *args, **kwargs):
        """
        Initialize discrete-time symbolic system.
        
        Follows the same template method pattern as parent:
        1. Set discrete-time flag
        2. Call parent __init__ (which validates and initializes)
        
        The parent's validation and initialization work correctly
        because discrete systems use the same symbolic machinery.
        """
        # Mark as discrete-time system
        self._is_discrete = True
        
        # Call parent initialization (handles everything else)
        super().__init__(*args, **kwargs)
    
    def __repr__(self) -> str:
        """
        Detailed string representation.
        
        Returns
        -------
        str
            Representation indicating discrete-time system
        
        Examples
        --------
        >>> repr(system)
        'DiscreteLinearSystem(nx=2, nu=1, ny=2, discrete=True, backend=numpy)'
        """
        return (
            f"{self.__class__.__name__}("
            f"nx={self.nx}, nu={self.nu}, ny={self.ny}, "
            f"discrete=True, order={self.order}, "
            f"backend={self._default_backend}, device={self._preferred_device})"
        )
    
    def __str__(self) -> str:
        """
        Human-readable string representation.
        
        Returns
        -------
        str
            Concise representation
        
        Examples
        --------
        >>> str(system)
        'DiscreteLinearSystem(nx=2, nu=1, discrete-time)'
        """
        equilibria_str = (
            f", {len(self.equilibria.list_names())} equilibria"
            if len(self.equilibria.list_names()) > 1
            else ""
        )
        return (
            f"{self.__class__.__name__}(nx={self.nx}, nu={self.nu}, "
            f"discrete-time{equilibria_str})"
        )
    
    def forward(self, x_k, u_k=None, backend=None):
        """
        Evaluate discrete-time dynamics: x[k+1] = f(x[k], u[k]).
        
        Computes the next state directly (no integration needed).
        
        Parameters
        ----------
        x_k : ArrayLike
            Current state x[k]
        u_k : Optional[ArrayLike]
            Current control u[k] (None for autonomous systems)
        backend : Optional[str]
            Backend selection
        
        Returns
        -------
        ArrayLike
            Next state x[k+1]
        
        Examples
        --------
        >>> x_k = np.array([1.0, 0.0])
        >>> u_k = np.array([0.5])
        >>> x_next = system.forward(x_k, u_k)
        >>> # x_next is x[k+1] directly, no integration
        
        Notes
        -----
        Unlike continuous systems where forward() returns dx/dt,
        this returns the next state x[k+1] directly. This is the
        key semantic difference between continuous and discrete systems.
        
        The implementation delegates to parent's forward() which
        evaluates _f_sym. The difference is only in interpretation:
        - Continuous: _f_sym = dx/dt (needs integration to get next x)
        - Discrete: _f_sym = x[k+1] (already the next state)
        """
        return super().forward(x_k, u_k, backend)
    
    def print_equations(self, simplify: bool = True):
        """
        Print symbolic equations in human-readable format.
        
        Overrides parent to use discrete-time notation (k, k+1)
        instead of continuous-time notation (t, dx/dt).
        
        Parameters
        ----------
        simplify : bool
            If True, simplify expressions before printing
        
        Examples
        --------
        >>> system.print_equations()
        ======================================================================
        DiscreteLinearSystem (Discrete-Time)
        ======================================================================
        State Variables: [x1, x2]
        Control Variables: [u]
        System Order: 1
        Dimensions: nx=2, nu=1, ny=2
        
        Dynamics: x[k+1] = f(x[k], u[k])
          x1[k+1] = 0.9*x1[k] + 0.1*x2[k]
          x2[k+1] = -0.1*x1[k] + 0.8*x2[k] + u[k]
        ======================================================================
        """
        print("=" * 70)
        print(f"{self.__class__.__name__} (Discrete-Time)")
        print("=" * 70)
        print(f"State Variables: {self.state_vars}")
        print(f"Control Variables: {self.control_vars}")
        print(f"System Order: {self.order}")
        print(f"Dimensions: nx={self.nx}, nu={self.nu}, ny={self.ny}")

        print("\nDynamics: x[k+1] = f(x[k], u[k])")
        for i, (var, expr) in enumerate(zip(self.state_vars, self._f_sym)):
            expr_sub = self.substitute_parameters(expr)
            if simplify:
                expr_sub = sp.simplify(expr_sub)
            print(f"  {var}[k+1] = {expr_sub}")

        if self._h_sym is not None:
            print("\nOutput: y[k] = h(x[k])")
            for i, expr in enumerate(self._h_sym):
                expr_sub = self.substitute_parameters(expr)
                if simplify:
                    expr_sub = sp.simplify(expr_sub)
                print(f"  y[{i}][k] = {expr_sub}")

        print("=" * 70)
    
    def get_config_dict(self) -> Dict:
        """
        Get system configuration including discrete-time flag.
        
        Extends parent's config with discrete-time indicator.
        
        Returns
        -------
        dict
            Configuration dictionary
        
        Examples
        --------
        >>> config = system.get_config_dict()
        >>> config['is_discrete']
        True
        """
        config = super().get_config_dict()
        config['is_discrete'] = True
        return config