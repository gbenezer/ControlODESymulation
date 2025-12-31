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
Structural Subtyping Protocols for ControlDESymulation
======================================================

This module defines Protocol classes that enable structural subtyping (duck typing
with type checking) across different system implementations. Protocols allow
multiple concrete classes to satisfy the same interface without sharing inheritance,
enabling flexible composition-based architectures.

Key Benefits
------------
1. **Flexibility**: DiscretizedSystem (wrapper) and DiscreteSymbolicSystem (symbolic)
   can both be used in the same algorithms

2. **Type Safety**: Static type checkers (mypy, pyright) catch when symbolic machinery
   is required vs optional

3. **Future Proof**: Easy to add new system types (neural, data-driven) without
   modifying existing class hierarchies

4. **Clear Contracts**: Protocols document expected interfaces explicitly

5. **No Runtime Overhead**: Protocols are purely for static type checking

Naming Convention
-----------------
All protocols use "Protocol" suffix to distinguish from concrete classes:

**Protocols** (interfaces):
- DiscreteSystemProtocol
- LinearizableDiscreteProtocol
- SymbolicDiscreteProtocol

**Concrete Classes** (implementations):
- DiscreteSymbolicSystem
- DiscretizedSystem
- DiscreteStochasticSystem

This naming makes it crystal clear:
- Import from `src.types.protocols` → interface/contract
- Import from `src.systems` → concrete implementation

Protocol Hierarchy
------------------
```
DiscreteSystemProtocol
    ↓ extends (adds linearize)
LinearizableDiscreteProtocol  
    ↓ extends (adds symbolic machinery)
SymbolicDiscreteProtocol


ContinuousSystemProtocol (optional)
    ↓ extends
LinearizableContinuousProtocol
    ↓ extends  
SymbolicContinuousProtocol
```

Usage Examples
--------------
**Example 1: LQR Design (most common)**

>>> from src.types.protocols import LinearizableDiscreteProtocol
>>> from src.types.control_classical import LQRResult
>>> 
>>> def discrete_lqr(
...     system: LinearizableDiscreteProtocol,
...     Q: np.ndarray,
...     R: np.ndarray
... ) -> LQRResult:
...     '''Design LQR for any linearizable discrete system.'''
...     Ad, Bd = system.linearize(np.zeros(system.nx), np.zeros(system.nu))
...     # ... solve discrete-time Riccati equation
...     return {"K": K, "P": P, ...}
>>> 
>>> # Both work:
>>> lqr1 = discrete_lqr(DiscreteSymbolicSystem(...), Q, R)  # ✓
>>> lqr2 = discrete_lqr(DiscretizedSystem(...), Q, R)       # ✓

**Example 2: Equation Export (requires symbolic)**

>>> from src.types.protocols import SymbolicDiscreteProtocol
>>> 
>>> def export_to_latex(system: SymbolicDiscreteProtocol) -> str:
...     '''Export equations - requires symbolic expressions.'''
...     latex = []
...     for var, expr in zip(system.state_vars, system._f_sym):
...         latex.append(f"{sp.latex(var)}_{k+1} = {sp.latex(expr)}")
...     return "\\n".join(latex)
>>> 
>>> # Type checker catches misuse:
>>> export_to_latex(DiscreteSymbolicSystem(...))  # ✓ OK
>>> export_to_latex(DiscretizedSystem(...))       # ✗ mypy error - good!

**Example 3: Generic Simulation**

>>> from src.types.protocols import DiscreteSystemProtocol
>>> 
>>> def monte_carlo(system: DiscreteSystemProtocol, n_trials: int):
...     '''Works with ANY discrete system.'''
...     results = []
...     for _ in range(n_trials):
...         x0 = np.random.randn(system.nx)
...         result = system.simulate(x0, None, n_steps=100)
...         results.append(result)
...     return results

**Example 4: Runtime Type Checking (use sparingly)**

>>> from src.types.protocols import SymbolicDiscreteProtocol
>>> 
>>> def smart_analysis(system: LinearizableDiscreteProtocol):
...     '''Adapt behavior based on capabilities.'''
...     Ad, Bd = system.linearize(x_eq, u_eq)
...     
...     # Check if symbolic machinery available
...     if isinstance(system, SymbolicDiscreteProtocol):
...         system.compile(backends=['jax'])  # Use code generation
...         system.print_equations()
...     else:
...         print("Numerical system - using direct evaluation")

When to Use Each Protocol
-------------------------
**DiscreteSystemProtocol**:
- Monte Carlo simulation
- Trajectory collection
- Reinforcement learning rollouts
- Any algorithm that just needs step()/simulate()

**LinearizableDiscreteProtocol** (MOST COMMON):
- LQR design
- MPC with linearization
- Pole placement
- Stability analysis
- Observer design
- Most control algorithms

**SymbolicDiscreteProtocol**:
- Equation export (LaTeX, Markdown)
- Code generation (C, CUDA, etc.)
- Symbolic manipulation
- Documentation generation
- Parameter sensitivity (symbolic)

**ContinuousSystemProtocol** (add only if needed):
- Trajectory optimization
- Safety verification
- Reachability analysis
- Discretization utilities

Authors
-------
Gil Benezer

License
-------
AGPL-3.0

See Also
--------
- typing.Protocol: Python's structural subtyping
- PEP 544: Protocol specification
- src.types.core: Core array and vector types
- src.types.linearization: Linearization result types
"""

from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, Union, runtime_checkable

# Type imports
from src.types.core import ControlVector, DiscreteControlInput, OutputVector, StateVector
from src.types.backends import Backend
from src.types.linearization import DiscreteLinearization, LinearizationResult
from src.types.trajectories import DiscreteSimulationResult, IntegrationResult, TimeSpan


# ============================================================================
# Discrete System Protocols
# ============================================================================


@runtime_checkable
class DiscreteSystemProtocol(Protocol):
    """
    Minimal interface for discrete-time dynamical systems.

    This protocol defines the basic contract that all discrete-time systems must
    satisfy, regardless of their implementation (symbolic, numerical, learned, etc.).

    Any object implementing this protocol can be:
    - Simulated forward in time
    - Used in trajectory-based algorithms
    - Rolled out with policies
    - Used in reinforcement learning

    Implementations
    ---------------
    Concrete classes that satisfy this protocol:
    - DiscreteSymbolicSystem: Symbolic discrete-time system
    - DiscreteStochasticSystem: Stochastic discrete-time system
    - DiscretizedSystem: Numerical discretization of continuous system
    - NeuralDiscreteSystem: Neural network dynamics (future)
    - DataDrivenDiscreteSystem: Learned from data (future)

    Required Attributes
    -------------------
    dt : float
        Sampling period in seconds
    nx : int
        Number of state variables
    nu : int
        Number of control inputs

    Required Methods
    ----------------
    step(x, u, k) -> x_next
        Single time step update: x[k+1] = f(x[k], u[k])
    simulate(x0, u_sequence, n_steps) -> result
        Multi-step simulation

    Use Cases
    ---------
    - Monte Carlo simulation
    - Trajectory collection for learning
    - Rollout with exploration policies
    - Basic dynamics analysis

    Examples
    --------
    Function accepting any discrete system:

    >>> def collect_trajectories(
    ...     system: DiscreteSystemProtocol,
    ...     n_trials: int = 100
    ... ) -> List[DiscreteSimulationResult]:
    ...     '''Collect random trajectories from any discrete system.'''
    ...     trajectories = []
    ...     for _ in range(n_trials):
    ...         x0 = np.random.randn(system.nx)
    ...         u_seq = np.random.randn(100, system.nu)
    ...         result = system.simulate(x0, u_seq, n_steps=100)
    ...         trajectories.append(result)
    ...     return trajectories
    >>>
    >>> # Works with any discrete system:
    >>> trajs1 = collect_trajectories(DiscreteSymbolicSystem(...))  # ✓
    >>> trajs2 = collect_trajectories(DiscretizedSystem(...))       # ✓

    Reinforcement learning rollout:

    >>> def evaluate_policy(
    ...     system: DiscreteSystemProtocol,
    ...     policy: Callable,
    ...     n_episodes: int
    ... ) -> float:
    ...     '''Evaluate policy on system.'''
    ...     total_reward = 0.0
    ...     for _ in range(n_episodes):
    ...         x = np.random.randn(system.nx)
    ...         for k in range(100):
    ...             u = policy(x, k)
    ...             x = system.step(x, u, k)
    ...             total_reward += reward_function(x, u)
    ...     return total_reward / n_episodes

    Type checking example:

    >>> def bad_function(system: DiscreteSystemProtocol):
    ...     system.linearize(...)  # ✗ mypy error: not in protocol!
    >>>
    >>> def good_function(system: LinearizableDiscreteProtocol):
    ...     system.linearize(...)  # ✓ OK: protocol includes this

    Notes
    -----
    The @runtime_checkable decorator allows isinstance() checks, but this
    should be used sparingly. Prefer static type checking at development time.
    """

    @property
    def dt(self) -> float:
        """
        Sampling period in seconds.

        Returns
        -------
        float
            Time step between consecutive states (t[k+1] - t[k])
        """
        ...

    @property
    def nx(self) -> int:
        """
        Number of state variables.

        Returns
        -------
        int
            Dimension of state vector x
        """
        ...

    @property
    def nu(self) -> int:
        """
        Number of control inputs.

        Returns
        -------
        int
            Dimension of control vector u
        """
        ...

    def step(
        self, x: StateVector, u: Optional[ControlVector] = None, k: int = 0
    ) -> StateVector:
        """
        Compute next state: x[k+1] = f(x[k], u[k]).

        Parameters
        ----------
        x : StateVector
            Current state (nx,)
        u : Optional[ControlVector]
            Control input (nu,), None for autonomous/zero control
        k : int
            Time step index (for time-varying systems)

        Returns
        -------
        StateVector
            Next state x[k+1]
        """
        ...

    def simulate(
        self, x0: StateVector, u_sequence: DiscreteControlInput, n_steps: int
    ) -> DiscreteSimulationResult:
        """
        Simulate system for multiple steps.

        Parameters
        ----------
        x0 : StateVector
            Initial state (nx,)
        u_sequence : DiscreteControlInput
            Control sequence (various formats supported)
        n_steps : int
            Number of steps to simulate

        Returns
        -------
        DiscreteSimulationResult
            Trajectory data including states, controls, time indices
        """
        ...


@runtime_checkable
class LinearizableDiscreteProtocol(DiscreteSystemProtocol, Protocol):
    """
    Discrete system with linearization capability.

    Extends DiscreteSystemProtocol with Jacobian computation, enabling
    linear control design algorithms (LQR, MPC, pole placement).

    The linearization provides discrete-time Jacobian matrices:
        δx[k+1] = Ad·δx[k] + Bd·δu[k]

    where:
        Ad = ∂f/∂x evaluated at (x_eq, u_eq)
        Bd = ∂f/∂u evaluated at (x_eq, u_eq)

    Implementations
    ---------------
    Concrete classes that satisfy this protocol:
    - DiscreteSymbolicSystem: Symbolic Jacobians via automatic differentiation
    - DiscreteStochasticSystem: Symbolic Jacobians + diffusion matrix
    - DiscretizedSystem: Wraps continuous linearization then discretizes
    - LinearizedDiscreteSystem: Explicitly provided A, B matrices

    Required Methods (in addition to DiscreteSystemProtocol)
    --------------------------------------------------------
    linearize(x_eq, u_eq) -> (Ad, Bd)
        Compute discrete Jacobian matrices

    Use Cases
    ---------
    - LQR controller design
    - Model Predictive Control (MPC) with linearization
    - Pole placement
    - Discrete Kalman filter design
    - Stability analysis (eigenvalue-based)
    - Controllability/observability analysis
    - Most modern control algorithms

    Examples
    --------
    LQR design function:

    >>> from scipy.linalg import solve_discrete_are
    >>> 
    >>> def design_lqr(
    ...     system: LinearizableDiscreteProtocol,
    ...     Q: np.ndarray,
    ...     R: np.ndarray,
    ...     x_eq: Optional[StateVector] = None,
    ...     u_eq: Optional[ControlVector] = None
    ... ) -> LQRResult:
    ...     '''Design LQR controller for any linearizable discrete system.'''
    ...     # Default to origin
    ...     if x_eq is None:
    ...         x_eq = np.zeros(system.nx)
    ...         u_eq = np.zeros(system.nu)
    ...     
    ...     # Get linearization
    ...     Ad, Bd = system.linearize(x_eq, u_eq)
    ...     
    ...     # Solve discrete-time algebraic Riccati equation
    ...     P = solve_discrete_are(Ad, Bd, Q, R)
    ...     K = np.linalg.inv(R + Bd.T @ P @ Bd) @ (Bd.T @ P @ Ad)
    ...     
    ...     # Check closed-loop stability
    ...     A_cl = Ad - Bd @ K
    ...     eigenvalues = np.linalg.eigvals(A_cl)
    ...     
    ...     return {
    ...         "K": K,
    ...         "P": P,
    ...         "eigenvalues": eigenvalues,
    ...         "cost": x_eq.T @ P @ x_eq
    ...     }
    >>> 
    >>> # Works with DiscreteSymbolicSystem
    >>> symbolic_sys = DiscreteOscillator(dt=0.01)
    >>> result1 = design_lqr(symbolic_sys, Q, R)
    >>> 
    >>> # Also works with DiscretizedSystem
    >>> continuous = Pendulum(m=1.0, l=0.5)
    >>> discretized = DiscretizedSystem(continuous, dt=0.01)
    >>> result2 = design_lqr(discretized, Q, R)  # ✓ Same function!

    Stability analysis:

    >>> def check_stability(system: LinearizableDiscreteProtocol) -> bool:
    ...     '''Check if discrete system is stable at origin.'''
    ...     Ad, Bd = system.linearize(
    ...         np.zeros(system.nx),
    ...         np.zeros(system.nu)
    ...     )
    ...     eigenvalues = np.linalg.eigvals(Ad)
    ...     return np.all(np.abs(eigenvalues) < 1.0)
    >>> 
    >>> is_stable = check_stability(any_discrete_system)  # Works with any!

    MPC with linearization:

    >>> def mpc_step(
    ...     system: LinearizableDiscreteProtocol,
    ...     x_current: StateVector,
    ...     x_ref: StateVector,
    ...     horizon: int = 10
    ... ) -> ControlVector:
    ...     '''MPC using linearization around reference.'''
    ...     # Linearize around reference
    ...     Ad, Bd = system.linearize(x_ref, np.zeros(system.nu))
    ...     
    ...     # Solve QP for optimal control
    ...     # ... MPC formulation ...
    ...     return u_optimal

    Notes
    -----
    The linearization is typically valid only for small deviations from
    the equilibrium point: δx = x - x_eq, δu = u - u_eq.

    For nonlinear systems, linearization provides a local approximation
    useful for controller design, but may not capture global behavior.

    The @runtime_checkable decorator enables isinstance() checks at runtime,
    though this should be used sparingly in favor of static type checking.
    """

    def linearize(
        self, x_eq: StateVector, u_eq: Optional[ControlVector] = None
    ) -> DiscreteLinearization:
        """
        Compute discrete-time linearization: Ad = ∂f/∂x, Bd = ∂f/∂u.

        Parameters
        ----------
        x_eq : StateVector
            Equilibrium state (nx,)
        u_eq : Optional[ControlVector]
            Equilibrium control (nu,), None = zero control

        Returns
        -------
        DiscreteLinearization
            Tuple (Ad, Bd) of Jacobian matrices:
            - Ad: State transition matrix (nx, nx)
            - Bd: Control input matrix (nx, nu)
        """
        ...


@runtime_checkable
class SymbolicDiscreteProtocol(LinearizableDiscreteProtocol, Protocol):
    """
    Discrete system with symbolic machinery.

    Extends LinearizableDiscreteProtocol with access to underlying symbolic
    expressions and code generation capabilities. This enables:
    - Symbolic code generation (C, CUDA, Julia)
    - Equation export (LaTeX, Markdown)
    - Symbolic parameter manipulation
    - Documentation generation

    Only systems with explicit symbolic representations satisfy this protocol.
    Numerical systems (DiscretizedSystem, neural networks) do NOT.

    Implementations
    ---------------
    Classes that satisfy this protocol:
    - DiscreteSymbolicSystem: Symbolic discrete dynamics
    - DiscreteStochasticSystem: Symbolic stochastic discrete dynamics

    Classes that DO NOT satisfy this protocol:
    - DiscretizedSystem: Purely numerical (no symbolic expressions)
    - NeuralDiscreteSystem: Learned dynamics (future)
    - DataDrivenDiscreteSystem: Identified from data (future)

    Required Attributes (in addition to LinearizableDiscreteProtocol)
    ------------------------------------------------------------------
    state_vars : list
        List of SymPy Symbol objects for state variables
    control_vars : list
        List of SymPy Symbol objects for control variables
    parameters : dict
        Dictionary mapping SymPy Symbols to numerical values

    Required Methods (in addition to LinearizableDiscreteProtocol)
    ---------------------------------------------------------------
    compile(backends, verbose) -> timing_dict
        Pre-compile symbolic expressions to numerical functions
    print_equations(simplify)
        Print symbolic equations in human-readable format
    substitute_parameters(expr) -> expr
        Substitute numerical parameter values into symbolic expression

    Use Cases
    ---------
    - **Code Generation**: Export to C, CUDA, Julia for real-time systems
    - **Documentation**: Generate LaTeX equations for papers/reports
    - **Symbolic Analysis**: Parameter sensitivity, symbolic stability
    - **Teaching**: Display equations for educational purposes
    - **Verification**: Symbolic checking of system properties

    Examples
    --------
    LaTeX equation export:

    >>> import sympy as sp
    >>> 
    >>> def export_equations_latex(
    ...     system: SymbolicDiscreteProtocol,
    ...     simplify: bool = True
    ... ) -> str:
    ...     '''Export system equations to LaTeX.'''
    ...     latex_lines = []
    ...     latex_lines.append(r"\\begin{align}")
    ...     
    ...     for var, expr in zip(system.state_vars, system._f_sym):
    ...         expr_sub = system.substitute_parameters(expr)
    ...         if simplify:
    ...             expr_sub = sp.simplify(expr_sub)
    ...         
    ...         var_latex = sp.latex(var)
    ...         expr_latex = sp.latex(expr_sub)
    ...         latex_lines.append(f"{var_latex}_{{k+1}} &= {expr_latex} \\\\\\\\")
    ...     
    ...     latex_lines.append(r"\\end{align}")
    ...     return "\\n".join(latex_lines)
    >>> 
    >>> # Only works with symbolic systems:
    >>> latex = export_equations_latex(DiscreteSymbolicSystem(...))  # ✓
    >>> latex = export_equations_latex(DiscretizedSystem(...))       # ✗ Type error

    C code generation:

    >>> def generate_c_code(system: SymbolicDiscreteProtocol) -> str:
    ...     '''Generate C code for embedded systems.'''
    ...     c_code = []
    ...     c_code.append("// Discrete dynamics: x[k+1] = f(x[k], u[k])")
    ...     c_code.append("void dynamics(double* x, double* u, double* x_next) {")
    ...     
    ...     for i, expr in enumerate(system._f_sym):
    ...         expr_sub = system.substitute_parameters(expr)
    ...         # Convert to C expression
    ...         c_expr = sp.ccode(expr_sub)
    ...         c_code.append(f"    x_next[{i}] = {c_expr};")
    ...     
    ...     c_code.append("}")
    ...     return "\\n".join(c_code)

    Symbolic parameter study:

    >>> def parameter_sensitivity(
    ...     system: SymbolicDiscreteProtocol,
    ...     param_name: str
    ... ) -> sp.Matrix:
    ...     '''Compute symbolic sensitivity to parameter.'''
    ...     # Find parameter symbol
    ...     param_sym = None
    ...     for sym, val in system.parameters.items():
    ...         if str(sym) == param_name:
    ...             param_sym = sym
    ...             break
    ...     
    ...     if param_sym is None:
    ...         raise ValueError(f"Parameter '{param_name}' not found")
    ...     
    ...     # Compute symbolic derivative: ∂f/∂param
    ...     sensitivity = sp.Matrix([
    ...         sp.diff(expr, param_sym) for expr in system._f_sym
    ...     ])
    ...     
    ...     return sensitivity

    Runtime capability checking:

    >>> def smart_compile(system: LinearizableDiscreteProtocol):
    ...     '''Compile if symbolic, otherwise skip.'''
    ...     if isinstance(system, SymbolicDiscreteProtocol):
    ...         print("Symbolic system - pre-compiling...")
    ...         system.compile(backends=['jax'], verbose=True)
    ...     else:
    ...         print("Numerical system - no compilation needed")

    Notes
    -----
    This is the MOST RESTRICTIVE protocol - only use when you genuinely
    need symbolic machinery. For most control algorithms, use
    LinearizableDiscreteProtocol instead.

    The symbolic attributes (state_vars, control_vars, parameters) use
    SymPy Symbol objects, not strings. Access the symbolic matrix _f_sym
    directly for advanced symbolic manipulation.
    """

    # Symbolic attributes (in addition to nx, nu, dt from parent protocol)
    state_vars: List  # List[sp.Symbol] - SymPy state variable symbols
    control_vars: List  # List[sp.Symbol] - SymPy control variable symbols
    parameters: Dict  # Dict[sp.Symbol, float] - Parameter values

    def compile(
        self, backends: Optional[List[str]] = None, verbose: bool = False
    ) -> Dict[str, float]:
        """
        Pre-compile symbolic expressions to numerical functions.

        Parameters
        ----------
        backends : Optional[List[str]]
            Target backends ('numpy', 'torch', 'jax'), None = all available
        verbose : bool
            Print compilation progress

        Returns
        -------
        Dict[str, float]
            Compilation times per backend (seconds)
        """
        ...

    def print_equations(self, simplify: bool = True):
        """
        Print symbolic equations in human-readable format.

        Parameters
        ----------
        simplify : bool
            Simplify expressions before printing
        """
        ...

    def substitute_parameters(self, expr) -> Any:
        """
        Substitute numerical parameter values into symbolic expression.

        Parameters
        ----------
        expr : Union[sp.Expr, sp.Matrix]
            Symbolic expression

        Returns
        -------
        Union[sp.Expr, sp.Matrix]
            Expression with parameters substituted
        """
        ...


# ============================================================================
# Continuous System Protocols (Optional - Add Only If Needed)
# ============================================================================


@runtime_checkable
class ContinuousSystemProtocol(Protocol):
    """
    Minimal interface for continuous-time dynamical systems.

    **OPTIONAL**: Only add this if you have functions that accept arbitrary
    continuous systems. Most control design happens in discrete time, so you
    may not need these protocols.

    Add continuous protocols if you have:
    - Trajectory optimization for continuous systems
    - Continuous-time verification algorithms
    - Reachability analysis
    - Generic continuous control design

    If you're unsure, skip this and add later if needed.

    Implementations
    ---------------
    - ContinuousSymbolicSystem
    - ContinuousStochasticSystem
    - NeuralODE (future)

    Use Cases
    ---------
    - Trajectory optimization
    - Safety verification
    - Reachability analysis
    - Discretization utilities

    Examples
    --------
    >>> def discretize_any(
    ...     system: ContinuousSystemProtocol,
    ...     dt: float
    ... ) -> DiscretizedSystem:
    ...     '''Discretize any continuous system.'''
    ...     return DiscretizedSystem(system, dt, method='rk4')
    >>> 
    >>> # Works with any continuous system:
    >>> discrete1 = discretize_any(ContinuousSymbolicSystem(...), dt=0.01)
    >>> discrete2 = discretize_any(ContinuousStochasticSystem(...), dt=0.01)

    Notes
    -----
    Consider whether you actually need this before adding it. Most discrete
    control algorithms don't need to accept arbitrary continuous systems.
    """

    @property
    def nx(self) -> int:
        """Number of state variables"""
        ...

    @property
    def nu(self) -> int:
        """Number of control inputs"""
        ...

    def __call__(
        self, x: StateVector, u: Optional[ControlVector] = None, t: float = 0.0
    ) -> StateVector:
        """
        Evaluate continuous dynamics: dx/dt = f(x, u, t).

        Parameters
        ----------
        x : StateVector
            Current state
        u : Optional[ControlVector]
            Control input
        t : float
            Current time

        Returns
        -------
        StateVector
            State derivative dx/dt
        """
        ...

    def integrate(
        self, x0: StateVector, u, t_span: TimeSpan, method: str = "RK45", **kwargs
    ) -> IntegrationResult:
        """
        Numerically integrate continuous system.

        Parameters
        ----------
        x0 : StateVector
            Initial state
        u : Union[ControlVector, Callable, None]
            Control input
        t_span : TimeSpan
            Integration interval (t_start, t_end)
        method : str
            Integration method
        **kwargs
            Additional integrator options

        Returns
        -------
        IntegrationResult
            Integration result with trajectory data
        """
        ...


@runtime_checkable
class LinearizableContinuousProtocol(ContinuousSystemProtocol, Protocol):
    """
    Continuous system with linearization capability.

    **OPTIONAL**: Only add if you have continuous-time control design algorithms.

    Implementations
    ---------------
    - ContinuousSymbolicSystem
    - ContinuousStochasticSystem (returns A, B, G)

    Use Cases
    ---------
    - Continuous LQR/LQG design
    - H2/H-infinity control
    - Continuous Kalman filter
    - Lyapunov stability analysis

    Examples
    --------
    >>> def continuous_lqr(
    ...     system: LinearizableContinuousProtocol,
    ...     Q: np.ndarray,
    ...     R: np.ndarray
    ... ):
    ...     '''Design continuous-time LQR.'''
    ...     A, B = system.linearize(np.zeros(system.nx), np.zeros(system.nu))
    ...     from scipy.linalg import solve_continuous_are
    ...     P = solve_continuous_are(A, B, Q, R)
    ...     K = np.linalg.inv(R) @ B.T @ P
    ...     return {"K": K, "P": P}
    """

    def linearize(
        self, x_eq: StateVector, u_eq: Optional[ControlVector] = None
    ) -> LinearizationResult:
        """
        Compute continuous-time linearization: A = ∂f/∂x, B = ∂f/∂u.

        Parameters
        ----------
        x_eq : StateVector
            Equilibrium state
        u_eq : Optional[ControlVector]
            Equilibrium control

        Returns
        -------
        LinearizationResult
            Tuple (A, B) or (A, B, G) for stochastic systems
        """
        ...


@runtime_checkable
class SymbolicContinuousProtocol(LinearizableContinuousProtocol, Protocol):
    """
    Continuous system with symbolic machinery.

    **OPTIONAL**: Only add if you have algorithms that need symbolic
    continuous systems specifically.

    Implementations
    ---------------
    - ContinuousSymbolicSystem
    - ContinuousStochasticSystem

    NOT implemented by:
    - NeuralODE (future)
    - Data-driven continuous systems (future)

    Use Cases
    ---------
    - Symbolic code generation for continuous systems
    - Equation export for continuous dynamics
    - Symbolic verification

    Examples
    --------
    >>> def export_ode_to_matlab(system: SymbolicContinuousProtocol) -> str:
    ...     '''Generate MATLAB ODE function.'''
    ...     matlab_code = []
    ...     matlab_code.append("function dxdt = dynamics(t, x, u)")
    ...     
    ...     for i, expr in enumerate(system._f_sym):
    ...         expr_sub = system.substitute_parameters(expr)
    ...         matlab_expr = sp.octave_code(expr_sub)  # SymPy → MATLAB
    ...         matlab_code.append(f"    dxdt({i+1}) = {matlab_expr};")
    ...     
    ...     matlab_code.append("end")
    ...     return "\\n".join(matlab_code)
    """

    # Symbolic attributes
    state_vars: List  # List[sp.Symbol]
    control_vars: List  # List[sp.Symbol]
    parameters: Dict  # Dict[sp.Symbol, float]

    def compile(
        self, backends: Optional[List[str]] = None, verbose: bool = False
    ) -> Dict[str, float]:
        """Pre-compile symbolic expressions"""
        ...

    def print_equations(self, simplify: bool = True):
        """Print symbolic equations"""
        ...

    def substitute_parameters(self, expr) -> Any:
        """Substitute parameter values"""
        ...


# ============================================================================
# Utility Protocols (Domain-Specific Capabilities)
# ============================================================================


@runtime_checkable
class StochasticSystemProtocol(Protocol):
    """
    System with stochastic dynamics (continuous or discrete).

    This protocol can be combined with discrete or continuous protocols
    to indicate stochastic capabilities.

    Implementations
    ---------------
    - ContinuousStochasticSystem
    - DiscreteStochasticSystem

    Examples
    --------
    >>> def estimate_noise_covariance(
    ...     system: StochasticSystemProtocol,
    ...     n_samples: int = 1000
    ... ) -> np.ndarray:
    ...     '''Estimate noise covariance from simulation.'''
    ...     if system.is_additive_noise():
    ...         # Constant noise - compute once
    ...         G = system.get_constant_noise()
    ...         return G @ G.T
    ...     else:
    ...         # State-dependent - estimate via Monte Carlo
    ...         # ...
    ...         pass
    """

    @property
    def is_stochastic(self) -> bool:
        """True for stochastic systems"""
        ...

    @property
    def nw(self) -> int:
        """Number of noise sources"""
        ...

    def is_additive_noise(self) -> bool:
        """True if noise is state-independent"""
        ...

    def is_multiplicative_noise(self) -> bool:
        """True if noise depends on state"""
        ...


@runtime_checkable
class CompilableSystemProtocol(Protocol):
    """
    System with code compilation capabilities.

    Satisfied by any system that can pre-compile its functions for
    performance optimization.

    Implementations
    ---------------
    - Any symbolic system (continuous or discrete)
    - Systems with JIT compilation

    Examples
    --------
    >>> def warmup_system(system: CompilableSystemProtocol):
    ...     '''Pre-compile for all available backends.'''
    ...     print("Compiling system...")
    ...     times = system.compile(backends=['numpy', 'torch', 'jax'], verbose=True)
    ...     print(f"Compilation times: {times}")
    """

    def compile(
        self, backends: Optional[List[str]] = None, verbose: bool = False
    ) -> Dict[str, float]:
        """Pre-compile functions"""
        ...

    def reset_caches(self, backends: Optional[List[str]] = None):
        """Clear compiled function cache"""
        ...


@runtime_checkable  
class ParametricSystemProtocol(Protocol):
    """
    System with modifiable parameters.

    Satisfied by symbolic systems that have parameter dictionaries.

    Examples
    --------
    >>> def parameter_sweep(
    ...     system: ParametricSystemProtocol,
    ...     param_name: str,
    ...     param_values: List[float]
    ... ):
    ...     '''Sweep parameter and analyze behavior.'''
    ...     results = []
    ...     for val in param_values:
    ...         # Find parameter symbol
    ...         for sym in system.parameters.keys():
    ...             if str(sym) == param_name:
    ...                 system.parameters[sym] = val
    ...                 break
    ...         
    ...         # Recompile and analyze
    ...         system.reset_caches()
    ...         result = analyze_stability(system)
    ...         results.append(result)
    ...     
    ...     return results
    """

    parameters: Dict  # Dict[sp.Symbol, float]

    def substitute_parameters(self, expr) -> Any:
        """Substitute parameter values"""
        ...


# ============================================================================
# Export All Protocols
# ============================================================================

__all__ = [
    # Discrete system protocols (CORE - always needed)
    "DiscreteSystemProtocol",
    "LinearizableDiscreteProtocol",
    "SymbolicDiscreteProtocol",
    
    # Continuous system protocols (OPTIONAL - add if needed)
    "ContinuousSystemProtocol",
    "LinearizableContinuousProtocol",
    "SymbolicContinuousProtocol",
    
    # Utility protocols (domain-specific capabilities)
    "StochasticSystemProtocol",
    "CompilableSystemProtocol",
    "ParametricSystemProtocol",
]