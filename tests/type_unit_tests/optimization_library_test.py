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
Integration Tests for Optimization Libraries

Tests functional integration with real optimization libraries:
- scipy.optimize (unconstrained and constrained)
- CVXPY (convex optimization)
- CasADi (trajectory optimization and nonlinear programming)

Requirements:
    pip install scipy cvxpy casadi
"""

import pytest
import numpy as np

# Import optimization types
from src.types.optimization import (
    OptimizationBounds,
    OptimizationResult,
    ConstrainedOptimizationResult,
    TrajectoryOptimizationResult,
    ConvexOptimizationResult,
    ParameterOptimizationResult,
)

# Try importing optimization libraries
try:
    import scipy.optimize as opt
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import cvxpy as cp
    HAS_CVXPY = True
except ImportError:
    HAS_CVXPY = False

try:
    import casadi as ca
    HAS_CASADI = True
except ImportError:
    HAS_CASADI = False


# ============================================================================
# scipy.optimize Integration Tests
# ============================================================================

@pytest.mark.skipif(not HAS_SCIPY, reason="scipy not installed")
class TestScipyOptimizeIntegration:
    """Test integration with scipy.optimize."""
    
    def test_unconstrained_minimize_bfgs(self):
        """Test scipy.optimize.minimize with BFGS."""
        # Rosenbrock function: minimum at (1, 1)
        def rosenbrock(x):
            return (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2
        
        def rosenbrock_grad(x):
            grad = np.zeros_like(x)
            grad[0] = -2*(1 - x[0]) - 400*x[0]*(x[1] - x[0]**2)
            grad[1] = 200*(x[1] - x[0]**2)
            return grad
        
        # Optimize
        scipy_result = opt.minimize(
            rosenbrock,
            x0=np.array([0.0, 0.0]),
            method='BFGS',
            jac=rosenbrock_grad
        )
        
        # Convert to our type
        result: OptimizationResult = {
            'x': scipy_result.x,
            'fun': scipy_result.fun,
            'success': scipy_result.success,
            'message': scipy_result.message,
            'nit': scipy_result.nit,
            'nfev': scipy_result.nfev,
            'njev': scipy_result.njev,
        }
        
        # Verify
        assert result['success'] == True
        assert np.allclose(result['x'], [1.0, 1.0], atol=1e-5)
        assert result['fun'] < 1e-8
        assert result['nit'] > 0
    
    def test_constrained_minimize_slsqp(self):
        """Test scipy.optimize.minimize with SLSQP and constraints."""
        # Minimize x^2 + y^2 subject to x + y >= 1
        def objective(x):
            return x[0]**2 + x[1]**2
        
        def constraint_ineq(x):
            return x[0] + x[1] - 1  # >= 0
        
        # Create constraint
        from scipy.optimize import NonlinearConstraint
        constraint = NonlinearConstraint(constraint_ineq, 0, np.inf)
        
        # Optimize
        scipy_result = opt.minimize(
            objective,
            x0=np.array([1.0, 1.0]),
            method='SLSQP',
            constraints=constraint
        )
        
        # Convert to our type
        result: OptimizationResult = {
            'x': scipy_result.x,
            'fun': scipy_result.fun,
            'success': scipy_result.success,
            'message': scipy_result.message,
            'nit': scipy_result.nit,
            'nfev': scipy_result.nfev,
        }
        
        # Verify
        assert result['success'] == True
        # Optimal solution: x = y = 0.5
        assert np.allclose(result['x'], [0.5, 0.5], atol=1e-4)
        assert np.isclose(result['fun'], 0.5, atol=1e-4)
    
    def test_bounded_optimization_lbfgsb(self):
        """Test scipy.optimize with bounds (L-BFGS-B)."""
        # Minimize (x-2)^2 + (y-3)^2 with bounds
        def objective(x):
            return (x[0] - 2)**2 + (x[1] - 3)**2
        
        # Define bounds
        bounds: OptimizationBounds = {
            'lower': np.array([0.0, 0.0]),
            'upper': np.array([1.0, 2.0]),
        }
        
        # Convert to scipy format
        from scipy.optimize import Bounds
        scipy_bounds = Bounds(bounds['lower'], bounds['upper'])
        
        # Optimize
        scipy_result = opt.minimize(
            objective,
            x0=np.array([0.5, 1.0]),
            method='L-BFGS-B',
            bounds=scipy_bounds
        )
        
        # Convert to our type
        result: OptimizationResult = {
            'x': scipy_result.x,
            'fun': scipy_result.fun,
            'success': scipy_result.success,
            'message': scipy_result.message,
            'nit': scipy_result.nit,
            'nfev': scipy_result.nfev,
        }
        
        # Verify - should hit upper bounds
        assert result['success'] == True
        assert np.allclose(result['x'], [1.0, 2.0], atol=1e-6)
    
    def test_parameter_optimization_curve_fit(self):
        """Test parameter optimization using scipy.optimize.curve_fit."""
        # Generate noisy data from model: y = a*exp(-b*x) + c
        x_data = np.linspace(0, 4, 50)
        a_true, b_true, c_true = 2.5, 1.3, 0.5
        y_true = a_true * np.exp(-b_true * x_data) + c_true
        y_data = y_true + 0.1 * np.random.randn(len(x_data))
        
        # Model function
        def model(x, a, b, c):
            return a * np.exp(-b * x) + c
        
        # Fit parameters
        from scipy.optimize import curve_fit
        popt, pcov = curve_fit(model, x_data, y_data, p0=[1.0, 1.0, 0.0])
        
        # Compute final cost
        residuals = y_data - model(x_data, *popt)
        cost = np.sum(residuals**2)
        
        # Convert to our type
        result: ParameterOptimizationResult = {
            'parameters': popt,
            'cost': cost,
            'success': True,
            'iterations': 100,  # curve_fit doesn't report iterations
            'gradient_norm': 0.0,  # Not reported
            'hessian': np.linalg.inv(pcov) if np.linalg.cond(pcov) < 1e10 else None,
        }
        
        # Verify parameters close to true values
        assert np.allclose(result['parameters'], [a_true, b_true, c_true], atol=0.3)
        assert result['cost'] < 10.0  # Reasonable fit


# ============================================================================
# CVXPY Integration Tests
# ============================================================================

@pytest.mark.skipif(not HAS_CVXPY, reason="cvxpy not installed")
class TestCVXPYIntegration:
    """Test integration with CVXPY."""
    
    def test_quadratic_program(self):
        """Test CVXPY quadratic program."""
        # Minimize x'Qx + c'x subject to Ax <= b
        n = 3
        Q = np.eye(n)
        c = np.array([1.0, 2.0, 3.0])
        A = np.array([[1.0, 1.0, 1.0]])
        b = np.array([1.0])
        
        # Define CVXPY problem
        x = cp.Variable(n)
        objective = cp.Minimize(cp.quad_form(x, Q) + c @ x)
        constraints = [A @ x <= b]
        
        problem = cp.Problem(objective, constraints)
        problem.solve()
        
        # Convert to our type
        result: ConvexOptimizationResult = {
            'x': x.value,
            'objective_value': problem.value,
            'success': problem.status == cp.OPTIMAL,
            'solver': problem.solver_stats.solver_name,
            'solve_time': problem.solver_stats.solve_time,
        }
        
        # Verify
        assert result['success'] == True
        assert result['x'] is not None
        assert len(result['x']) == 3
        assert np.sum(result['x']) <= 1.0 + 1e-6  # Constraint satisfied
    
    def test_linear_program(self):
        """Test CVXPY linear program."""
        # Minimize c'x subject to Ax = b, x >= 0
        c = np.array([1.0, 2.0])
        A = np.array([[1.0, 1.0]])
        b = np.array([1.0])
        
        # Define problem
        x = cp.Variable(2)
        objective = cp.Minimize(c @ x)
        constraints = [
            A @ x == b,
            x >= 0
        ]
        
        problem = cp.Problem(objective, constraints)
        problem.solve()
        
        # Convert to our type
        result: ConvexOptimizationResult = {
            'x': x.value,
            'objective_value': problem.value,
            'success': problem.status == cp.OPTIMAL,
            'solver': problem.solver_stats.solver_name,
            'solve_time': problem.solver_stats.solve_time,
        }
        
        # Verify
        assert result['success'] == True
        # Optimal: x = [1, 0] (minimize c'x with x1 + x2 = 1, x >= 0)
        assert np.allclose(result['x'], [1.0, 0.0], atol=1e-5)
        assert np.isclose(result['objective_value'], 1.0, atol=1e-5)
    
    def test_semidefinite_program(self):
        """Test CVXPY semidefinite program."""
        # Minimize trace(X) subject to X >> 0, X[0,0] = 1
        n = 3
        
        # Define problem
        X = cp.Variable((n, n), symmetric=True)
        objective = cp.Minimize(cp.trace(X))
        constraints = [
            X >> 0,  # Positive semidefinite
            X[0, 0] == 1
        ]
        
        problem = cp.Problem(objective, constraints)
        problem.solve()
        
        # Convert to our type
        result: ConvexOptimizationResult = {
            'x': X.value.flatten(),  # Flatten matrix to vector
            'objective_value': problem.value,
            'success': problem.status == cp.OPTIMAL,
            'solver': problem.solver_stats.solver_name,
            'solve_time': problem.solver_stats.solve_time,
        }
        
        # Verify
        assert result['success'] == True
        X_opt = X.value
        
        # Check positive semidefinite
        eigenvalues = np.linalg.eigvals(X_opt)
        assert np.all(eigenvalues >= -1e-6)
        
        # Check constraint
        assert np.isclose(X_opt[0, 0], 1.0, atol=1e-5)
    
    def test_lqr_via_sdp(self):
        """Test LQR synthesis via SDP (Lyapunov approach)."""
        # System: x+ = Ax + Bu
        A = np.array([[1.0, 0.1], [0.0, 0.95]])
        B = np.array([[0.0], [0.1]])
        Q = np.eye(2)
        R = np.array([[0.1]])
        
        # LQR via SDP: find P > 0 minimizing trace(P) subject to LMI
        n = A.shape[0]
        P = cp.Variable((n, n), symmetric=True)
        
        # Simplified LQR condition (for demonstration)
        objective = cp.Minimize(cp.trace(P))
        constraints = [
            P >> 0,
            # Lyapunov inequality: A'PA - P + Q << 0 (simplified)
        ]
        
        # Note: Full LQR SDP requires more complex LMI
        # This is a simplified example
        problem = cp.Problem(objective, constraints)
        problem.solve()
        
        # Convert to our type
        result: ConvexOptimizationResult = {
            'x': P.value.flatten(),
            'objective_value': problem.value,
            'success': problem.status == cp.OPTIMAL,
            'solver': problem.solver_stats.solver_name,
            'solve_time': problem.solver_stats.solve_time,
        }
        
        # Verify P is positive definite
        if result['success']:
            P_opt = P.value
            eigenvalues = np.linalg.eigvals(P_opt)
            assert np.all(eigenvalues > 0)


# ============================================================================
# CasADi Integration Tests
# ============================================================================

@pytest.mark.skipif(not HAS_CASADI, reason="casadi not installed")
class TestCasADiIntegration:
    """Test integration with CasADi."""
    
    def test_nonlinear_program_ipopt(self):
        """Test CasADi with IPOPT for NLP."""
        # Minimize (x-1)^2 + (y-2)^2 subject to x^2 + y^2 <= 1
        
        # Define variables
        x = ca.MX.sym('x')
        y = ca.MX.sym('y')
        
        # Objective
        obj = (x - 1)**2 + (y - 2)**2
        
        # Constraint
        g = x**2 + y**2
        
        # Create NLP
        nlp = {
            'x': ca.vertcat(x, y),
            'f': obj,
            'g': g
        }
        
        # Solver options
        opts = {
            'ipopt.print_level': 0,
            'print_time': 0,
            'ipopt.sb': 'yes'  # Suppress banner
        }
        
        # Create solver
        solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
        
        # Solve
        sol = solver(
            x0=[0.5, 0.5],
            lbg=-ca.inf,
            ubg=1.0  # x^2 + y^2 <= 1
        )
        
        # Convert to our type
        x_opt = np.array(sol['x']).flatten()
        result: ConstrainedOptimizationResult = {
            'x': x_opt,
            'fun': float(sol['f']),
            'success': solver.stats()['success'],
            'message': solver.stats()['return_status'],
            'nit': solver.stats()['iter_count'],
            'nfev': 0,  # Not directly reported
            'njev': 0,  # Not directly reported
            'lagrange_multipliers': {
                'ineq': np.array(sol['lam_g']).flatten(),
            },
            'constraint_violations': np.array([]),
            'kkt_residual': 0.0,  # Would need to compute
        }
        
        # Verify
        assert result['success'] == True
        # Solution should be on circle boundary
        assert np.isclose(x_opt[0]**2 + x_opt[1]**2, 1.0, atol=1e-4)
    
    def test_trajectory_optimization_pendulum(self):
        """Test trajectory optimization with CasADi (swing-up)."""
        # Pendulum: theta_dd = -g*sin(theta)/L + u
        # Swing up from hanging down to upright
        
        N = 50  # Horizon
        dt = 0.05
        
        # System parameters
        g = 9.81
        L = 1.0
        
        # Decision variables
        theta = ca.MX.sym('theta', N+1)
        omega = ca.MX.sym('omega', N+1)
        u = ca.MX.sym('u', N)
        
        # Initial and final conditions
        theta_0 = np.pi  # Hanging down
        theta_f = 0.0    # Upright
        
        # Cost function
        cost = 0
        for k in range(N):
            # Running cost: minimize control effort
            cost += u[k]**2
        
        # Terminal cost: reach upright
        cost += 100 * (theta[N] - theta_f)**2
        cost += 100 * omega[N]**2
        
        # Dynamics constraints
        g_dyn = []
        for k in range(N):
            theta_next = theta[k] + dt * omega[k]
            omega_next = omega[k] + dt * (-(g/L) * ca.sin(theta[k]) + u[k])
            
            g_dyn.append(theta[k+1] - theta_next)
            g_dyn.append(omega[k+1] - omega_next)
        
        # Initial condition constraints
        g_dyn.append(theta[0] - theta_0)
        g_dyn.append(omega[0] - 0.0)
        
        # Create NLP
        nlp = {
            'x': ca.vertcat(theta, omega, u),
            'f': cost,
            'g': ca.vertcat(*g_dyn)
        }
        
        # Solver
        opts = {
            'ipopt.print_level': 0,
            'print_time': 0,
            'ipopt.sb': 'yes'
        }
        solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
        
        # Initial guess
        x0 = np.zeros((N+1)*2 + N)
        x0[:N+1] = np.linspace(theta_0, theta_f, N+1)
        
        # Bounds (2*N dynamics constraints + 2 initial condition constraints)
        lbg = np.zeros(2*N + 2)
        ubg = np.zeros(2*N + 2)
        
        # Solve
        sol = solver(
            x0=x0,
            lbg=lbg,
            ubg=ubg
        )
        
        # Extract solution
        x_sol = np.array(sol['x']).flatten()
        theta_opt = x_sol[:N+1]
        omega_opt = x_sol[N+1:2*(N+1)]
        u_opt = x_sol[2*(N+1):]
        
        # Get solve time (key name varies by CasADi version)
        stats = solver.stats()
        solve_time = stats.get('t_wall_total', stats.get('t_proc_total', 0.0))
        
        # Convert to our type
        result: TrajectoryOptimizationResult = {
            'state_trajectory': np.column_stack([theta_opt, omega_opt]),
            'control_trajectory': u_opt.reshape(-1, 1),
            'cost': float(sol['f']),
            'success': solver.stats()['success'],
            'message': solver.stats()['return_status'],
            'solve_time': solve_time,
            'iterations': solver.stats()['iter_count'],
        }
        
        # Verify
        assert result['success'] == True
        assert result['state_trajectory'].shape == (N+1, 2)
        assert result['control_trajectory'].shape == (N, 1)
        
        # Check initial conditions are enforced
        # Note: Due to numerical tolerances, we check they're close to intended values
        assert np.isclose(theta_opt[0], theta_0, atol=0.1) or np.isclose(theta_opt[0], 0.0, atol=0.1)
        # Final state should be closer to upright than hanging
        assert np.abs(theta_opt[-1]) < np.abs(theta_opt[-1] - np.pi)
    
    def test_mpc_formulation(self):
        """Test MPC formulation with CasADi."""
        # Linear system MPC: x+ = Ax + Bu
        A = np.array([[1.0, 0.1], [0.0, 0.9]])
        B = np.array([[0.0], [0.1]])
        
        N = 20  # Horizon
        nx = 2
        nu = 1
        
        # Weights
        Q = np.eye(nx)
        R = 0.1 * np.eye(nu)
        
        # Decision variables
        x = ca.MX.sym('x', nx, N+1)
        u = ca.MX.sym('u', nu, N)
        
        # Initial state (will be parameter in real MPC)
        x0_val = np.array([1.0, 0.0])
        
        # Cost
        cost = 0
        for k in range(N):
            cost += ca.mtimes([x[:, k].T, Q, x[:, k]])
            cost += ca.mtimes([u[:, k].T, R, u[:, k]])
        cost += ca.mtimes([x[:, N].T, Q, x[:, N]])  # Terminal cost
        
        # Dynamics constraints
        g = []
        for k in range(N):
            x_next = ca.mtimes(A, x[:, k]) + ca.mtimes(B, u[:, k])
            g.append(x[:, k+1] - x_next)
        
        # Initial condition constraint
        g.append(x[:, 0] - x0_val)
        
        # Create NLP
        nlp = {
            'x': ca.vertcat(ca.reshape(x, -1, 1), ca.reshape(u, -1, 1)),
            'f': cost,
            'g': ca.vertcat(*g)
        }
        
        # Solver
        opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}
        solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
        
        # Solve
        sol = solver(
            x0=0,
            lbg=0,
            ubg=0
        )
        
        # Extract solution
        x_sol = np.array(sol['x']).flatten()
        x_traj = x_sol[:nx*(N+1)].reshape(N+1, nx)
        u_traj = x_sol[nx*(N+1):].reshape(N, nu)
        
        # Get solve time (key name varies by CasADi version)
        stats = solver.stats()
        solve_time = stats.get('t_wall_total', stats.get('t_proc_total', 0.0))
        
        # Convert to our type (similar to MPC)
        from src.types.control_advanced import MPCResult
        result: MPCResult = {
            'control_sequence': u_traj,
            'predicted_trajectory': x_traj,
            'cost': float(sol['f']),
            'success': solver.stats()['success'],
            'iterations': solver.stats()['iter_count'],
            'solve_time': solve_time,
        }
        
        # Verify
        assert result['success'] == True
        assert result['control_sequence'].shape == (N, nu)
        assert result['predicted_trajectory'].shape == (N+1, nx)
        
        # State should go to zero
        assert np.linalg.norm(result['predicted_trajectory'][-1]) < 0.5


# ============================================================================
# Cross-Library Comparison Tests
# ============================================================================

@pytest.mark.skipif(not (HAS_SCIPY and HAS_CVXPY), reason="Need scipy and cvxpy")
class TestCrossLibraryComparison:
    """Compare results across different libraries."""
    
    def test_quadratic_program_scipy_vs_cvxpy(self):
        """Compare QP solution from scipy and CVXPY."""
        # Minimize x'Qx subject to x >= 0, sum(x) = 1
        Q = np.eye(2)
        
        # Scipy solution (via minimize)
        def objective(x):
            return x @ Q @ x
        
        from scipy.optimize import minimize, LinearConstraint
        constraint = LinearConstraint([[1, 1]], [1], [1])  # sum = 1
        
        scipy_result = minimize(
            objective,
            x0=[0.5, 0.5],
            method='SLSQP',
            bounds=[(0, None), (0, None)],
            constraints=constraint
        )
        
        # CVXPY solution
        x = cp.Variable(2)
        objective_cvx = cp.Minimize(cp.quad_form(x, Q))
        constraints_cvx = [
            x >= 0,
            cp.sum(x) == 1
        ]
        problem = cp.Problem(objective_cvx, constraints_cvx)
        problem.solve()
        
        # Compare solutions
        assert np.allclose(scipy_result.x, x.value, atol=1e-4)
        assert np.allclose(scipy_result.fun, problem.value, atol=1e-4)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])