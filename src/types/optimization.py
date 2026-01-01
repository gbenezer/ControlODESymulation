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
Optimization Types

Result types for optimization algorithms used in control and estimation:
- General nonlinear optimization
- Constrained optimization
- Trajectory optimization
- Parameter optimization

These types provide structured return values from optimization solvers,
enabling consistent interfaces across scipy.optimize, cvxpy, casadi, etc.

Mathematical Background
----------------------
Nonlinear Programming (NLP):
    minimize    f(x)
    subject to  g(x) ≤ 0  (inequality constraints)
                h(x) = 0  (equality constraints)
                lb ≤ x ≤ ub  (bounds)

    Algorithms: SLSQP, IPOPT, SQP, interior-point

    KKT Conditions (optimality):
        ∇f(x*) + Σλᵢ∇gᵢ(x*) + Σμⱼ∇hⱼ(x*) = 0
        λᵢgᵢ(x*) = 0  (complementarity)
        λᵢ ≥ 0, gᵢ(x*) ≤ 0

Trajectory Optimization:
    minimize    Σₖ L(x[k], u[k]) + Φ(x[N])
    subject to  x[k+1] = f(x[k], u[k])
                g(x[k], u[k]) ≤ 0

    Methods: Direct transcription, shooting, collocation
    Solvers: IPOPT, SNOPT, CasADi

Convex Optimization:
    minimize    f(x)  (convex f)
    subject to  Ax ≤ b  (linear inequalities)
                Cx = d  (linear equalities)

    Special cases:
    - QP: f(x) = ½x'Qx + c'x
    - SOCP: Second-order cone constraints
    - SDP: Semidefinite constraints

    Advantages: Global optimum guaranteed, polynomial-time

Usage
-----
>>> from src.types.optimization import (
...     OptimizationResult,
...     TrajectoryOptimizationResult,
...     ConvexOptimizationResult,
... )
>>>
>>> # Nonlinear optimization
>>> from scipy.optimize import minimize
>>> result: OptimizationResult = minimize(
...     fun=cost_function,
...     x0=x_init,
...     method='SLSQP',
...     constraints=constraints
... )
>>> x_opt = result['x']
>>> print(f"Success: {result['success']}, Cost: {result['fun']:.3f}")
>>>
>>> # Trajectory optimization
>>> traj_result: TrajectoryOptimizationResult = solve_ocp(
...     system, x0, xf, horizon=100
... )
>>> x_traj = traj_result['state_trajectory']
>>> u_traj = traj_result['control_trajectory']
"""

from typing import Dict, Optional

from typing_extensions import TypedDict

from .core import (
    ArrayLike,
)
from .trajectories import (
    ControlSequence,
    StateTrajectory,
)

# ============================================================================
# General Optimization Results
# ============================================================================


class OptimizationBounds(TypedDict):
    """
    Optimization variable bounds.

    Specifies box constraints: lower ≤ x ≤ upper

    Fields
    ------
    lower : ArrayLike
        Lower bounds on variables (n,)
    upper : ArrayLike
        Upper bounds on variables (n,)

    Examples
    --------
    >>> # Bound state variables
    >>> bounds: OptimizationBounds = {
    ...     'lower': np.array([-10.0, -5.0, 0.0]),
    ...     'upper': np.array([10.0, 5.0, 1.0]),
    ... }
    >>>
    >>> # Use with scipy.optimize
    >>> from scipy.optimize import Bounds
    >>> scipy_bounds = Bounds(bounds['lower'], bounds['upper'])
    """

    lower: ArrayLike
    upper: ArrayLike


class OptimizationResult(TypedDict, total=False):
    """
    General nonlinear optimization result.

    Compatible with scipy.optimize return format.

    Fields
    ------
    x : ArrayLike
        Optimal solution x* (n,)
    fun : float
        Optimal objective value f(x*)
    success : bool
        Whether optimization converged successfully
    message : str
        Solver status message
    nit : int
        Number of iterations performed
    nfev : int
        Number of objective function evaluations
    njev : int
        Number of Jacobian evaluations (if applicable)

    Examples
    --------
    >>> # Unconstrained optimization
    >>> def rosenbrock(x):
    ...     return (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2
    >>>
    >>> from scipy.optimize import minimize
    >>> result: OptimizationResult = minimize(
    ...     rosenbrock,
    ...     x0=np.array([0.0, 0.0]),
    ...     method='BFGS'
    ... )
    >>>
    >>> if result['success']:
    ...     print(f"Optimum: {result['x']}")  # [1, 1]
    ...     print(f"Cost: {result['fun']}")   # ~0
    ...     print(f"Iterations: {result['nit']}")
    >>>
    >>> # Constrained optimization
    >>> def objective(x):
    ...     return x[0]**2 + x[1]**2
    >>>
    >>> def constraint(x):
    ...     return x[0] + x[1] - 1  # x[0] + x[1] >= 1
    >>>
    >>> from scipy.optimize import NonlinearConstraint
    >>> result: OptimizationResult = minimize(
    ...     objective,
    ...     x0=np.array([1.0, 1.0]),
    ...     method='SLSQP',
    ...     constraints=NonlinearConstraint(constraint, 0, np.inf)
    ... )
    """

    x: ArrayLike
    fun: float
    success: bool
    message: str
    nit: int
    nfev: int
    njev: int


class ConstrainedOptimizationResult(TypedDict, total=False):
    """
    Constrained optimization result with dual variables.

    Includes Lagrange multipliers for sensitivity analysis.

    Fields
    ------
    x : ArrayLike
        Optimal primal solution x* (n,)
    fun : float
        Optimal objective f(x*)
    success : bool
        Convergence success
    message : str
        Solver message
    nit : int
        Number of iterations
    nfev : int
        Function evaluations
    njev : int
        Jacobian evaluations
    lagrange_multipliers : Dict[str, ArrayLike]
        Dual variables λ for constraints
    constraint_violations : ArrayLike
        Constraint residuals (should be ≈0)
    kkt_residual : float
        KKT optimality condition residual

    Examples
    --------
    >>> # Optimization with inequality constraints
    >>> result: ConstrainedOptimizationResult = solve_nlp(
    ...     objective=lambda x: x[0]**2 + x[1]**2,
    ...     constraints={'ineq': lambda x: x[0] + x[1] - 1},
    ...     x0=np.array([1.0, 1.0])
    ... )
    >>>
    >>> x_opt = result['x']
    >>>
    >>> # Check KKT conditions
    >>> if result['kkt_residual'] < 1e-6:
    ...     print("KKT conditions satisfied")
    >>>
    >>> # Sensitivity analysis via Lagrange multipliers
    >>> lambda_ineq = result['lagrange_multipliers']['ineq']
    >>> print(f"Shadow price: {lambda_ineq}")
    >>>
    >>> # Verify constraints satisfied
    >>> violations = result['constraint_violations']
    >>> if np.max(np.abs(violations)) < 1e-6:
    ...     print("All constraints satisfied")
    """

    x: ArrayLike
    fun: float
    success: bool
    message: str
    nit: int
    nfev: int
    njev: int
    lagrange_multipliers: Dict[str, ArrayLike]
    constraint_violations: ArrayLike
    kkt_residual: float


# ============================================================================
# Trajectory Optimization
# ============================================================================


class TrajectoryOptimizationResult(TypedDict, total=False):
    """
    Trajectory optimization result.

    Result from optimal control problem (OCP) solved via
    direct transcription, shooting, or collocation.

    Fields
    ------
    state_trajectory : StateTrajectory
        Optimal state trajectory x*(t) (N+1, nx)
    control_trajectory : ControlSequence
        Optimal control trajectory u*(t) (N, nu)
    cost : float
        Total cost J = Σ L(x,u) + Φ(x[N])
    success : bool
        Whether optimization converged
    message : str
        Solver message
    solve_time : float
        Computation time in seconds
    iterations : int
        Number of optimization iterations
    constraint_violations : Optional[ArrayLike]
        Dynamics and path constraint violations

    Examples
    --------
    >>> # Minimum-time problem
    >>> def running_cost(x, u):
    ...     return 1.0  # Time-optimal
    >>>
    >>> def terminal_cost(x):
    ...     return 0.0
    >>>
    >>> result: TrajectoryOptimizationResult = solve_ocp(
    ...     system=pendulum,
    ...     x0=np.array([np.pi, 0]),     # Hanging down
    ...     xf=np.array([0, 0]),         # Upright
    ...     running_cost=running_cost,
    ...     terminal_cost=terminal_cost,
    ...     horizon=100,
    ...     dt=0.05
    ... )
    >>>
    >>> if result['success']:
    ...     x_traj = result['state_trajectory']
    ...     u_traj = result['control_trajectory']
    ...
    ...     import matplotlib.pyplot as plt
    ...     plt.plot(x_traj[:, 0], label='theta')
    ...     plt.plot(u_traj[:, 0], label='torque')
    ...     plt.legend()
    ...
    ...     print(f"Minimum time: {result['cost']:.3f} seconds")
    >>>
    >>> # Check dynamics constraints
    >>> if 'constraint_violations' in result:
    ...     max_viol = np.max(np.abs(result['constraint_violations']))
    ...     print(f"Max constraint violation: {max_viol:.2e}")
    """

    state_trajectory: StateTrajectory
    control_trajectory: ControlSequence
    cost: float
    success: bool
    message: str
    solve_time: float
    iterations: int
    constraint_violations: Optional[ArrayLike]


# ============================================================================
# Convex Optimization
# ============================================================================


class ConvexOptimizationResult(TypedDict, total=False):
    """
    Convex optimization result (QP, SOCP, SDP).

    For problems solved via CVX, CVXPY, MOSEK, etc.

    Fields
    ------
    x : ArrayLike
        Optimal primal solution (n,)
    objective_value : float
        Optimal objective
    success : bool
        Problem status (optimal/feasible)
    solver : str
        Solver used ('ECOS', 'MOSEK', 'SCS', etc.)
    solve_time : float
        Solve time in seconds
    dual_variables : Optional[Dict[str, ArrayLike]]
        Dual variables for constraints

    Examples
    --------
    >>> # Quadratic Program (QP)
    >>> import cvxpy as cp
    >>>
    >>> x = cp.Variable(2)
    >>> Q = np.eye(2)
    >>> objective = cp.Minimize(cp.quad_form(x, Q))
    >>> constraints = [x[0] + x[1] >= 1]
    >>>
    >>> problem = cp.Problem(objective, constraints)
    >>> problem.solve()
    >>>
    >>> result: ConvexOptimizationResult = {
    ...     'x': x.value,
    ...     'objective_value': problem.value,
    ...     'success': problem.status == 'optimal',
    ...     'solver': problem.solver_stats.solver_name,
    ...     'solve_time': problem.solver_stats.solve_time,
    ... }
    >>>
    >>> if result['success']:
    ...     print(f"Optimal x: {result['x']}")
    ...     print(f"Optimal cost: {result['objective_value']:.3f}")
    >>>
    >>> # Semidefinite Program (SDP)
    >>> P = cp.Variable((3, 3), symmetric=True)
    >>> objective = cp.Minimize(cp.trace(P))
    >>> constraints = [
    ...     P >> 0,  # Positive semidefinite
    ...     P[0, 0] == 1
    ... ]
    >>> problem = cp.Problem(objective, constraints)
    >>> problem.solve()
    """

    x: ArrayLike
    objective_value: float
    success: bool
    solver: str
    solve_time: float
    dual_variables: Optional[Dict[str, ArrayLike]]


# ============================================================================
# Parameter Optimization
# ============================================================================


class ParameterOptimizationResult(TypedDict, total=False):
    """
    Parameter optimization result.

    For optimizing model parameters, controller gains, etc.

    Fields
    ------
    parameters : ArrayLike
        Optimal parameters θ* (nθ,)
    cost : float
        Final cost (e.g., prediction error)
    success : bool
        Optimization converged
    iterations : int
        Number of iterations
    gradient_norm : float
        Final gradient norm (should be ≈0 at optimum)
    hessian : Optional[ArrayLike]
        Hessian at optimum (for uncertainty quantification)

    Examples
    --------
    >>> # Fit system parameters to data
    >>> def prediction_error(theta, u_data, y_data):
    ...     y_pred = simulate_system(theta, u_data)
    ...     return np.mean((y_data - y_pred)**2)
    >>>
    >>> result: ParameterOptimizationResult = optimize_parameters(
    ...     objective=lambda theta: prediction_error(theta, u_data, y_data),
    ...     theta0=np.array([1.0, 1.0, 0.5]),
    ...     method='L-BFGS-B'
    ... )
    >>>
    >>> if result['success']:
    ...     theta_opt = result['parameters']
    ...     print(f"Optimal parameters: {theta_opt}")
    ...     print(f"Prediction error: {result['cost']:.3e}")
    ...
    ...     # Check convergence
    ...     if result['gradient_norm'] < 1e-6:
    ...         print("Gradient converged to zero")
    >>>
    >>> # Parameter uncertainty from Hessian
    >>> if 'hessian' in result:
    ...     H = result['hessian']
    ...     param_covariance = np.linalg.inv(H)
    ...     param_std = np.sqrt(np.diag(param_covariance))
    ...     print(f"Parameter std: {param_std}")
    """

    parameters: ArrayLike
    cost: float
    success: bool
    iterations: int
    gradient_norm: float
    hessian: Optional[ArrayLike]


# ============================================================================
# Export All
# ============================================================================

__all__ = [
    # Basic optimization
    "OptimizationBounds",
    "OptimizationResult",
    "ConstrainedOptimizationResult",
    # Trajectory optimization
    "TrajectoryOptimizationResult",
    # Specialized optimization
    "ConvexOptimizationResult",
    "ParameterOptimizationResult",
]
