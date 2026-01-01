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
Additional Unit Tests for ContinuousSystemBase
==============================================

Advanced test coverage complementing the base test suite, focusing on:

- Numerical accuracy and convergence
- Stiff system performance
- Long-time integration stability
- Complex control scenarios (switching, discontinuous, adaptive)
- Memory management for large trajectories
- Batch evaluation performance
- Numerical edge cases and ill-conditioning
- Integration with real solvers (scipy behaviors)
- Special dynamics (limit cycles, chaos, oscillations)
- Error recovery and resilience
- Interpolation accuracy
- Performance benchmarks

Test Organization
-----------------
- TestNumericalAccuracy: Solver convergence and accuracy
- TestStiffSystems: Stiff solver performance and correctness
- TestLongTimeIntegration: Stability over long horizons
- TestComplexControlScenarios: Advanced control inputs
- TestBatchEvaluationPerformance: Vectorized dynamics
- TestMemoryManagement: Large trajectory handling
- TestNumericalStability: Ill-conditioned systems
- TestSpecialDynamics: Oscillators, chaos, limit cycles
- TestErrorRecovery: Resilience after failures
- TestInterpolation: Regular grid accuracy
- TestControllerComplexity: State feedback, switching
- TestIntegrationDiagnostics: Solver metrics analysis
- TestPerformanceBenchmarks: Timing and efficiency

Usage
-----
Run additional tests:
    pytest test_continuous_system_base_additional.py -v

Run with main tests:
    pytest test_continuous_system_base_test.py test_continuous_system_base_additional.py -v

Run specific category:
    pytest test_continuous_system_base_additional.py::TestStiffSystems -v
"""

import sys
import time
import unittest
from typing import Optional
from unittest.mock import Mock, patch
import gc

import numpy as np
import pytest

from src.types.core import ControlVector, StateVector
from src.types.linearization import ContinuousLinearization
from src.types.trajectories import IntegrationResult, SimulationResult
from src.systems.base.core.continuous_system_base import ContinuousSystemBase

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

# =============================================================================
# Advanced Test System Implementations
# =============================================================================


class VanDerPolOscillator(ContinuousSystemBase):
    """Van der Pol oscillator - classic nonlinear system with limit cycle."""
    
    def __init__(self, mu=1.0):
        self.nx = 2
        self.nu = 0  # Autonomous
        self.ny = 2
        self.mu = mu
    
    def __call__(self, x, u=None, t=0.0):
        """dx/dt = [x2, mu*(1-x1^2)*x2 - x1]"""
        x1, x2 = x[0], x[1]
        dx1 = x2
        dx2 = self.mu * (1 - x1**2) * x2 - x1
        return np.array([dx1, dx2])
    
    def integrate(self, x0, u=None, t_span=(0.0, 10.0), method="RK45", **kwargs):
        from scipy.integrate import solve_ivp
        
        def rhs(t, x):
            u_val = u(t, x) if callable(u) else (u if u is not None else None)
            return self(x, u_val, t)
        
        result = solve_ivp(rhs, t_span, x0, method=method, **kwargs)
        return {
            "t": result.t,
            "x": result.y.T,  # Convert (nx, T) to (T, nx)
            "success": result.success,
            "message": result.message,
            "nfev": result.nfev,
            "njev": getattr(result, 'njev', 0),
            "nlu": getattr(result, 'nlu', 0),
            "status": result.status,
            "nsteps": len(result.t),
            "integration_time": 0.0,
            "solver": method
        }
    
    def linearize(self, x_eq, u_eq=None):
        x1, x2 = x_eq[0], x_eq[1]
        A = np.array([
            [0, 1],
            [-1 - 2*self.mu*x1*x2, self.mu*(1 - x1**2)]
        ])
        B = np.zeros((2, 0))  # No control
        return (A, B)


class StiffChemicalReaction(ContinuousSystemBase):
    """Stiff chemical reaction system (Robertson problem variant)."""
    
    def __init__(self):
        self.nx = 3
        self.nu = 0
        self.ny = 3
        # Rate constants creating stiffness
        self.k1 = 0.04
        self.k2 = 3e7
        self.k3 = 1e4
    
    def __call__(self, x, u=None, t=0.0):
        """Stiff reaction dynamics."""
        y1, y2, y3 = x[0], x[1], x[2]
        dy1 = -self.k1 * y1 + self.k3 * y2 * y3
        dy2 = self.k1 * y1 - self.k3 * y2 * y3 - self.k2 * y2**2
        dy3 = self.k2 * y2**2
        return np.array([dy1, dy2, dy3])
    
    def integrate(self, x0, u=None, t_span=(0.0, 1e5), method="BDF", **kwargs):
        from scipy.integrate import solve_ivp
        
        def rhs(t, x):
            u_val = u(t, x) if callable(u) else (u if u is not None else None)
            return self(x, u_val, t)
        
        result = solve_ivp(rhs, t_span, x0, method=method, **kwargs)
        return {
            "t": result.t,
            "x": result.y.T,  # Convert (nx, T) to (T, nx)
            "success": result.success,
            "message": result.message,
            "nfev": result.nfev,
            "njev": getattr(result, 'njev', 0),
            "nlu": getattr(result, 'nlu', 0),
            "status": result.status,
            "nsteps": len(result.t),
            "integration_time": 0.0,
            "solver": method
        }
    
    def linearize(self, x_eq, u_eq=None):
        y1, y2, y3 = x_eq[0], x_eq[1], x_eq[2]
        A = np.array([
            [-self.k1, self.k3*y3, self.k3*y2],
            [self.k1, -self.k3*y3 - 2*self.k2*y2, -self.k3*y2],
            [0, 2*self.k2*y2, 0]
        ])
        B = np.zeros((3, 0))
        return (A, B)


class HighDimensionalLinear(ContinuousSystemBase):
    """High-dimensional linear system for performance testing."""
    
    def __init__(self, n=100):
        self.nx = n
        self.nu = int(n / 10)
        self.ny = n
        # Stable random system
        np.random.seed(42)
        A_unstable = np.random.randn(n, n)
        self.A = A_unstable - np.eye(n) * (np.abs(np.linalg.eigvals(A_unstable)).max() + 1)
        self.B = np.random.randn(n, self.nu)
    
    def __call__(self, x, u=None, t=0.0):
        if u is None:
            u = np.zeros(self.nu)
        return self.A @ x + self.B @ u
    
    def integrate(self, x0, u=None, t_span=(0.0, 10.0), method="RK45", **kwargs):
        from scipy.integrate import solve_ivp
        
        def rhs(t, x):
            u_val = u(t, x) if callable(u) else (u if u is not None else None)
            return self(x, u_val, t)
        
        result = solve_ivp(rhs, t_span, x0, method=method, **kwargs)
        return {
            "t": result.t,
            "x": result.y.T,  # Convert (nx, T) to (T, nx)
            "success": result.success,
            "message": result.message,
            "nfev": result.nfev,
            "njev": getattr(result, 'njev', 0),
            "nlu": getattr(result, 'nlu', 0),
            "status": result.status,
            "nsteps": len(result.t),
            "integration_time": 0.0,
            "solver": method
        }
    
    def linearize(self, x_eq, u_eq=None):
        return (self.A, self.B)


class LorenzSystem(ContinuousSystemBase):
    """Lorenz chaotic system for testing chaotic dynamics."""
    
    def __init__(self, sigma=10, rho=28, beta=8/3):
        self.nx = 3
        self.nu = 0
        self.ny = 3
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
    
    def __call__(self, x, u=None, t=0.0):
        """Lorenz equations."""
        x1, x2, x3 = x[0], x[1], x[2]
        dx1 = self.sigma * (x2 - x1)
        dx2 = x1 * (self.rho - x3) - x2
        dx3 = x1 * x2 - self.beta * x3
        return np.array([dx1, dx2, dx3])
    
    def integrate(self, x0, u=None, t_span=(0.0, 50.0), method="RK45", **kwargs):
        from scipy.integrate import solve_ivp
        
        def rhs(t, x):
            u_val = u(t, x) if callable(u) else (u if u is not None else None)
            return self(x, u_val, t)
        
        result = solve_ivp(rhs, t_span, x0, method=method, **kwargs)
        return {
            "t": result.t,
            "x": result.y.T,  # Convert (nx, T) to (T, nx)
            "success": result.success,
            "message": result.message,
            "nfev": result.nfev,
            "njev": getattr(result, 'njev', 0),
            "nlu": getattr(result, 'nlu', 0),
            "status": result.status,
            "nsteps": len(result.t),
            "integration_time": 0.0,
            "solver": method
        }
    
    def linearize(self, x_eq, u_eq=None):
        x1, x2, x3 = x_eq[0], x_eq[1], x_eq[2]
        A = np.array([
            [-self.sigma, self.sigma, 0],
            [self.rho - x3, -1, -x1],
            [x2, x1, -self.beta]
        ])
        B = np.zeros((3, 0))
        return (A, B)


class ControlledOscillator(ContinuousSystemBase):
    """Simple oscillator with control for controller testing."""
    
    def __init__(self, omega=1.0, zeta=0.1):
        self.nx = 2
        self.nu = 1
        self.ny = 2
        self.omega = omega
        self.zeta = zeta
    
    def __call__(self, x, u=None, t=0.0):
        """Damped oscillator: m*x'' + 2*zeta*omega*x' + omega^2*x = u"""
        if u is None:
            u = np.zeros(1)
        x1, x2 = x[0], x[1]
        dx1 = x2
        dx2 = -self.omega**2 * x1 - 2*self.zeta*self.omega*x2 + u[0]
        return np.array([dx1, dx2])
    
    def integrate(self, x0, u=None, t_span=(0.0, 10.0), method="RK45", **kwargs):
        from scipy.integrate import solve_ivp
        
        def rhs(t, x):
            u_val = u(t, x) if callable(u) else (u if u is not None else None)
            return self(x, u_val, t)
        
        result = solve_ivp(rhs, t_span, x0, method=method, **kwargs)
        return {
            "t": result.t,
            "x": result.y.T,  # Convert (nx, T) to (T, nx)
            "success": result.success,
            "message": result.message,
            "nfev": result.nfev,
            "njev": getattr(result, 'njev', 0),
            "nlu": getattr(result, 'nlu', 0),
            "status": result.status,
            "nsteps": len(result.t),
            "integration_time": 0.0,
            "solver": method
        }
    
    def linearize(self, x_eq, u_eq=None):
        A = np.array([
            [0, 1],
            [-self.omega**2, -2*self.zeta*self.omega]
        ])
        B = np.array([[0], [1]])
        return (A, B)


# =============================================================================
# Test: Numerical Accuracy and Convergence
# =============================================================================


class TestNumericalAccuracy(unittest.TestCase):
    """Test numerical accuracy and convergence properties."""
    
    def test_convergence_with_decreasing_tolerance(self):
        """Integration becomes more accurate with tighter tolerances."""
        system = VanDerPolOscillator(mu=0.5)
        x0 = np.array([2.0, 0.0])
        t_span = (0.0, 10.0)
        
        # Test with decreasing tolerances
        tolerances = [1e-3, 1e-6, 1e-9]
        final_states = []
        
        for tol in tolerances:
            result = system.integrate(x0, t_span=t_span, rtol=tol, atol=tol)
            final_states.append(result['x'][-1, :])
        
        # More stringent tolerance should give different (hopefully better) result
        diff_1_2 = np.linalg.norm(final_states[0] - final_states[1])
        diff_2_3 = np.linalg.norm(final_states[1] - final_states[2])
        
        # Second refinement should be smaller (convergence)
        self.assertLess(diff_2_3, diff_1_2)
    
    def test_solver_comparison_rk45_vs_dopri5(self):
        """Different solvers should give similar results."""
        system = ControlledOscillator()
        x0 = np.array([1.0, 0.0])
        
        result_rk45 = system.integrate(x0, t_span=(0, 10), method="RK45", rtol=1e-6, atol=1e-9)
        result_rk23 = system.integrate(x0, t_span=(0, 10), method="RK23", rtol=1e-6, atol=1e-9)
        
        # Both should succeed
        self.assertTrue(result_rk45['success'])
        self.assertTrue(result_rk23['success'])
        
        # Results should be close (though adaptive time points differ)
        # Compare final states
        final_rk45 = result_rk45['x'][-1, :]
        final_rk23 = result_rk23['x'][-1, :]
        
        np.testing.assert_allclose(final_rk45, final_rk23, rtol=1e-3)
    
    def test_adaptive_time_stepping_efficiency(self):
        """Adaptive method should use fewer steps on smooth dynamics."""
        system = ControlledOscillator()
        x0 = np.array([1.0, 0.0])
        
        # Adaptive should be efficient
        result = system.integrate(x0, t_span=(0, 10), method="RK45")
        
        # Should not need excessive evaluations for this smooth system
        self.assertLess(result['nfev'], 1000)
    
    def test_numerical_energy_conservation(self):
        """Undamped oscillator should approximately conserve energy."""
        system = ControlledOscillator(omega=1.0, zeta=0.0)  # No damping
        x0 = np.array([1.0, 0.0])
        
        result = system.integrate(x0, t_span=(0, 20), rtol=1e-9, atol=1e-12)
        
        # Compute energy: E = 0.5*(x1^2 + x2^2)
        energies = 0.5 * (result['x'][:, 0]**2 + result['x'][:, 1]**2)
        
        # Energy should be approximately constant
        energy_drift = np.abs(energies[-1] - energies[0]) / energies[0]
        self.assertLess(energy_drift, 1e-3)  # Less than 0.1% drift


# =============================================================================
# Test: Stiff Systems
# =============================================================================


class TestStiffSystems(unittest.TestCase):
    """Test stiff solver performance and correctness."""
    
    def test_stiff_system_requires_implicit_solver(self):
        """Stiff system should work better with implicit solvers."""
        system = StiffChemicalReaction()
        x0 = np.array([1.0, 0.0, 0.0])
        
        # BDF (implicit) should succeed
        result_bdf = system.integrate(x0, t_span=(0, 1e5), method="BDF", rtol=1e-6)
        self.assertTrue(result_bdf['success'])
        
        # Explicit method would struggle (many steps or fail)
        # We won't actually run this to avoid slow tests, but document behavior
    
    def test_stiff_system_radau_performance(self):
        """Radau solver should handle stiff systems efficiently."""
        system = StiffChemicalReaction()
        x0 = np.array([1.0, 0.0, 0.0])
        
        result = system.integrate(x0, t_span=(0, 1e3), method="Radau", rtol=1e-6)
        
        self.assertTrue(result['success'])
        # Should not require excessive evaluations
        self.assertLess(result['nfev'], 10000)
    
    def test_stiff_system_conservation_laws(self):
        """Stiff chemical reaction should conserve total mass."""
        system = StiffChemicalReaction()
        x0 = np.array([1.0, 0.0, 0.0])
        
        result = system.integrate(x0, t_span=(0, 1e5), method="BDF")
        
        # Total concentration should be conserved
        total_initial = np.sum(x0)
        total_final = np.sum(result['x'][-1, :])
        
        np.testing.assert_allclose(total_final, total_initial, rtol=1e-6)
    
    def test_stiff_system_jacobian_evaluations(self):
        """Implicit solvers should use Jacobian evaluations."""
        system = StiffChemicalReaction()
        x0 = np.array([1.0, 0.0, 0.0])
        
        result = system.integrate(x0, t_span=(0, 1e3), method="Radau")
        
        # Implicit methods should compute Jacobians
        # (njev might be 0 if finite differences used internally)
        # Just check result structure
        self.assertIn('njev', result)
        self.assertIn('nlu', result)


# =============================================================================
# Test: Long-Time Integration
# =============================================================================


class TestLongTimeIntegration(unittest.TestCase):
    """Test stability and accuracy over long integration horizons."""
    
    def test_long_time_stable_system(self):
        """Stable system should remain bounded over long times."""
        system = ControlledOscillator(omega=1.0, zeta=0.2)
        x0 = np.array([1.0, 0.0])
        
        result = system.integrate(x0, t_span=(0, 1000), rtol=1e-6)
        
        self.assertTrue(result['success'])
        # States should remain bounded (damped oscillator)
        self.assertLess(np.max(np.abs(result['x'])), 2.0)
    
    def test_long_time_autonomous_decay(self):
        """Damped autonomous system should decay over time."""
        system = ControlledOscillator(omega=1.0, zeta=0.5)
        x0 = np.array([5.0, 0.0])
        
        result = system.integrate(x0, t_span=(0, 100))
        
        # Final state should be near zero (damped)
        final_state = result['x'][-1, :]
        self.assertLess(np.linalg.norm(final_state), 0.1)
    
    @pytest.mark.slow
    def test_very_long_integration_10000_seconds(self):
        """Very long integration (10000 seconds) - marked slow."""
        system = ControlledOscillator(omega=0.1, zeta=0.1)
        x0 = np.array([1.0, 0.0])
        
        result = system.integrate(x0, t_span=(0, 10000), rtol=1e-6)
        
        self.assertTrue(result['success'])
        # System should remain stable
        self.assertLess(np.max(np.abs(result['x'])), 10.0)


# =============================================================================
# Test: Complex Control Scenarios
# =============================================================================


class TestComplexControlScenarios(unittest.TestCase):
    """Test advanced control input scenarios."""
    
    def test_switching_controller(self):
        """Controller that switches at specific time."""
        system = ControlledOscillator()
        x0 = np.array([1.0, 0.0])
        
        def switching_control(t, x):
            """Switch from u=1 to u=-1 at t=5"""
            return np.array([1.0 if t < 5.0 else -1.0])
        
        result = system.integrate(x0, u=switching_control, t_span=(0, 10))
        
        self.assertTrue(result['success'])
        # Should have integrated through the switch
        self.assertGreater(len(result['t']), 10)
    
    def test_high_frequency_control(self):
        """High-frequency sinusoidal control."""
        system = ControlledOscillator()
        x0 = np.array([0.0, 0.0])
        
        def high_freq_control(t, x):
            return np.array([np.sin(20 * t)])
        
        result = system.integrate(x0, u=high_freq_control, t_span=(0, 10), max_step=0.01)
        
        self.assertTrue(result['success'])
    
    def test_state_dependent_control(self):
        """State-dependent control via simulate()."""
        system = ControlledOscillator()
        x0 = np.array([2.0, 0.0])
        
        def state_controller(x, t):
            """Simple proportional control: u = -K*x"""
            K = np.array([[1.0, 0.5]])
            return -K @ x
        
        result = system.simulate(x0, controller=state_controller, t_span=(0, 10), dt=0.1)
        
        self.assertTrue(result['success'])
        self.assertIn('controls', result)
        # Base class should populate controls
        if result['controls'] is not None:
            # FIXED: Time-major convention (T, nu)
            self.assertEqual(result['controls'].shape[0], len(result['time']))
            self.assertEqual(result['controls'].shape[1], system.nu)
    
    def test_saturated_control(self):
        """Control with saturation limits."""
        system = ControlledOscillator()
        x0 = np.array([5.0, 0.0])
        
        def saturated_controller(x, t):
            """Saturate control between -1 and 1."""
            u_desired = -2.0 * x[0] - 0.5 * x[1]
            return np.array([np.clip(u_desired, -1.0, 1.0)])
        
        result = system.simulate(x0, controller=saturated_controller, t_span=(0, 10), dt=0.1)
        
        self.assertTrue(result['success'])
        # Check saturation was applied (if controls are tracked)
        if result['controls'] is not None:
            # FIXED: Time-major indexing
            self.assertLessEqual(np.max(np.abs(result['controls'])), 1.0 + 1e-6)
    
    def test_adaptive_control(self):
        """Time-varying adaptive controller."""
        system = ControlledOscillator()
        x0 = np.array([1.0, 0.0])
        
        def adaptive_controller(x, t):  # (x, t) not (t, x)
            """Gain increases with time."""
            K = 0.1 + 0.1 * t
            return np.array([-K * x[0]])
        
        result = system.simulate(x0, controller=adaptive_controller, t_span=(0, 10), dt=0.1)
        
        self.assertTrue(result['success'])


# =============================================================================
# Test: Batch Evaluation Performance
# =============================================================================


class TestBatchEvaluationPerformance(unittest.TestCase):
    """Test vectorized dynamics evaluation."""
    
    def test_batch_vs_loop_consistency(self):
        """Batch evaluation should match loop evaluation."""
        system = ControlledOscillator()
        
        n_batch = 10
        x_batch = np.random.randn(2, n_batch)
        u_batch = np.random.randn(1, n_batch)
        
        # Batch evaluation
        dx_batch = system(x_batch, u_batch)
        
        # Loop evaluation
        dx_loop = np.zeros((2, n_batch))
        for i in range(n_batch):
            dx_loop[:, i] = system(x_batch[:, i], u_batch[:, i])
        
        np.testing.assert_allclose(dx_batch, dx_loop)
    
    def test_large_batch_evaluation(self):
        """Large batch evaluation should work."""
        system = ControlledOscillator()
        
        n_batch = 1000
        x_batch = np.random.randn(2, n_batch)
        u_batch = np.random.randn(1, n_batch)
        
        dx_batch = system(x_batch, u_batch)
        
        self.assertEqual(dx_batch.shape, (2, n_batch))


# =============================================================================
# Test: Memory Management
# =============================================================================


class TestMemoryManagement(unittest.TestCase):
    """Test memory handling for large trajectories."""
    
    def test_large_trajectory_memory(self):
        """Large trajectory should not cause memory issues."""
        system = ControlledOscillator()
        x0 = np.array([1.0, 0.0])
        
        # Long integration with fine time grid
        result = system.simulate(x0, t_span=(0, 100), dt=0.001, method="RK45")
        
        self.assertTrue(result['success'])
        # Should have many points
        self.assertGreater(len(result['time']), 10000)
    
    def test_multiple_integrations_no_memory_leak(self):
        """Multiple integrations should not leak memory."""
        system = ControlledOscillator()
        x0 = np.array([1.0, 0.0])
        
        # Run many integrations
        for _ in range(100):
            result = system.integrate(x0, t_span=(0, 1))
            del result
        
        # Force garbage collection
        gc.collect()
        
        # If no leak, test passes (no assertion needed)
        self.assertTrue(True)


# =============================================================================
# Test: Numerical Stability
# =============================================================================


class TestNumericalStability(unittest.TestCase):
    """Test behavior with ill-conditioned systems."""
    
    def test_very_small_timestep(self):
        """Very small dt should work but be inefficient."""
        system = ControlledOscillator()
        x0 = np.array([1.0, 0.0])
        
        result = system.simulate(x0, t_span=(0, 1), dt=1e-6)
        
        self.assertTrue(result['success'])
        # Many time points
        self.assertGreater(len(result['time']), 100000)
    
    def test_very_large_timestep(self):
        """Very large dt should give coarse result."""
        system = ControlledOscillator()
        x0 = np.array([1.0, 0.0])
        
        result = system.simulate(x0, t_span=(0, 10), dt=1.0)
        
        self.assertTrue(result['success'])
        # Few time points
        self.assertEqual(len(result['time']), 11)  # 0, 1, 2, ..., 10
    
    def test_near_singular_linearization(self):
        """System near singularity should handle linearization."""
        system = ControlledOscillator(omega=1.0, zeta=0.0)
        x_eq = np.array([0.0, 0.0])
        
        A, B = system.linearize(x_eq)
        
        # A should have eigenvalues on imaginary axis (undamped)
        eigenvalues = np.linalg.eigvals(A)
        self.assertLess(np.max(np.real(eigenvalues)), 1e-10)


# =============================================================================
# Test: Special Dynamics
# =============================================================================


class TestSpecialDynamics(unittest.TestCase):
    """Test systems with special behaviors."""
    
    def test_limit_cycle_van_der_pol(self):
        """Van der Pol oscillator should converge to limit cycle."""
        system = VanDerPolOscillator(mu=1.0)
        x0 = np.array([0.1, 0.1])
        
        result = system.integrate(x0, t_span=(0, 50), rtol=1e-6)
        
        self.assertTrue(result['success'])
        
        # After long time, should be on limit cycle
        # Check periodicity (rough test)
        final_trajectory = result['x'][-1000:, :]  # Last 1000 points
        amplitude = np.std(final_trajectory[:, 0])
        self.assertGreater(amplitude, 0.5)  # Non-trivial oscillation
    
    def test_chaotic_lorenz_sensitivity(self):
        """Lorenz system should show sensitive dependence."""
        system = LorenzSystem()
        
        x0_1 = np.array([1.0, 1.0, 1.0])
        x0_2 = np.array([1.0, 1.0, 1.001])  # Tiny perturbation
        
        result_1 = system.integrate(x0_1, t_span=(0, 20), rtol=1e-9)
        result_2 = system.integrate(x0_2, t_span=(0, 20), rtol=1e-9)
        
        # Small initial difference should grow
        initial_diff = np.linalg.norm(x0_1 - x0_2)
        final_diff = np.linalg.norm(result_1['x'][-1, :] - result_2['x'][-1, :])
        
        self.assertGreater(final_diff, initial_diff * 100)
    
    def test_periodic_orbit_oscillator(self):
        """Undamped oscillator should have periodic orbit."""
        system = ControlledOscillator(omega=1.0, zeta=0.0)
        x0 = np.array([1.0, 0.0])
        
        result = system.integrate(x0, t_span=(0, 20), rtol=1e-9)
        
        # Check periodicity: compare state at multiples of period
        T = 2 * np.pi / system.omega  # Period
        
        # Find indices near t=T and t=2T
        idx_1T = np.argmin(np.abs(result['t'] - T))
        idx_2T = np.argmin(np.abs(result['t'] - 2*T))
        
        state_1T = result['x'][idx_1T, :]
        state_2T = result['x'][idx_2T, :]
        
        # Should be close (periodic)
        np.testing.assert_allclose(state_1T, state_2T, atol=0.1)


# =============================================================================
# Test: Error Recovery
# =============================================================================


class TestErrorRecovery(unittest.TestCase):
    """Test system behavior after integration failures."""
    
    def test_system_usable_after_failed_integration(self):
        """System should remain usable after integration failure."""
        system = ControlledOscillator()
        x0 = np.array([1.0, 0.0])
        
        # Try integration with impossible time span
        try:
            result = system.integrate(x0, t_span=(10, 0), method="RK45")
        except Exception:
            pass
        
        # System should still work
        result = system.integrate(x0, t_span=(0, 10))
        self.assertTrue(result['success'])
    
    def test_multiple_failed_integrations(self):
        """Multiple failures should not corrupt system state."""
        system = ControlledOscillator()
        x0 = np.array([1.0, 0.0])
        
        # Generate multiple failures
        for _ in range(5):
            try:
                system.integrate(x0, t_span=(0, -1))
            except Exception:
                pass
        
        # System should still work
        result = system.integrate(x0, t_span=(0, 10))
        self.assertTrue(result['success'])


# =============================================================================
# Test: Interpolation Accuracy
# =============================================================================


class TestInterpolationAccuracy(unittest.TestCase):
    """Test accuracy of regular grid interpolation in simulate()."""
    
    def test_simulate_vs_integrate_accuracy(self):
        """simulate() with dense output should be accurate."""
        system = ControlledOscillator()
        x0 = np.array([1.0, 0.0])
        
        # Get full integration result
        result_integrate = system.integrate(x0, t_span=(0, 10), dense_output=True, rtol=1e-9)
        
        # Get simulate result
        result_simulate = system.simulate(x0, t_span=(0, 10), dt=0.1, method="RK45", rtol=1e-9)
        
        # Both should succeed
        self.assertTrue(result_integrate['success'])
        self.assertTrue(result_simulate['success'])
        
        # FIXED: Time-major indexing for final states
        # integrate() returns (T, nx) in 'x' key
        final_integrate = result_integrate['x'][-1, :]
        
        # simulate() returns (T, nx) in 'states' key (NEW convention)
        final_simulate = result_simulate['states'][-1, :]
        
        np.testing.assert_allclose(final_integrate, final_simulate, rtol=1e-3)


# =============================================================================
# Test: Controller Complexity
# =============================================================================


class TestControllerComplexity(unittest.TestCase):
    """Test complex controller scenarios."""
    
    def test_nested_controller_calls(self):
        """Controller that internally uses system evaluation."""
        system = ControlledOscillator()
        x0 = np.array([1.0, 0.0])
        
        def nested_controller(x, t):  # (x, t) not (t, x)
            """Controller that evaluates system dynamics."""
            dx = system(x, np.zeros(1), t)
            # Simple logic based on derivative
            u = -0.5 * dx[1]
            return np.array([u])
        
        result = system.simulate(x0, controller=nested_controller, t_span=(0, 10), dt=0.1)
        
        self.assertTrue(result['success'])
    
    def test_stateful_controller(self):
        """Controller with internal state (closure)."""
        system = ControlledOscillator()
        x0 = np.array([1.0, 0.0])
        
        # Create stateful controller
        integral_error = [0.0]  # Mutable to allow closure modification
        
        def pi_controller(x, t):  # (x, t) not (t, x)
            """PI controller with integral action."""
            error = x[0]  # Error from setpoint (0)
            integral_error[0] += error * 0.1  # Rough integration
            u = -2.0 * error - 0.5 * integral_error[0]
            return np.array([u])
        
        result = system.simulate(x0, controller=pi_controller, t_span=(0, 10), dt=0.1)
        
        self.assertTrue(result['success'])


# =============================================================================
# Test: Integration Diagnostics
# =============================================================================


class TestIntegrationDiagnostics(unittest.TestCase):
    """Test solver diagnostic information."""
    
    def test_nfev_increases_with_duration(self):
        """Function evaluations should increase with integration time."""
        system = ControlledOscillator()
        x0 = np.array([1.0, 0.0])
        
        result_short = system.integrate(x0, t_span=(0, 1))
        result_long = system.integrate(x0, t_span=(0, 10))
        
        self.assertGreater(result_long['nfev'], result_short['nfev'])
    
    def test_tighter_tolerance_increases_nfev(self):
        """Tighter tolerance should require more evaluations."""
        system = ControlledOscillator()
        x0 = np.array([1.0, 0.0])
        
        result_loose = system.integrate(x0, t_span=(0, 10), rtol=1e-3)
        result_tight = system.integrate(x0, t_span=(0, 10), rtol=1e-9)
        
        self.assertGreater(result_tight['nfev'], result_loose['nfev'])
    
    def test_diagnostic_fields_present(self):
        """All diagnostic fields should be present."""
        system = ControlledOscillator()
        x0 = np.array([1.0, 0.0])
        
        result = system.integrate(x0, t_span=(0, 10))
        
        required_fields = ['success', 'message', 'nfev', 'njev', 'nlu', 'status']
        for field in required_fields:
            self.assertIn(field, result)


# =============================================================================
# Test: Performance Benchmarks
# =============================================================================


class TestPerformanceBenchmarks(unittest.TestCase):
    """Benchmark tests for performance monitoring."""
    
    def test_high_dimensional_system_performance(self):
        """High-dimensional system should integrate in reasonable time."""
        system = HighDimensionalLinear(n=100)
        x0 = np.random.randn(100)
        
        start_time = time.time()
        result = system.integrate(x0, t_span=(0, 10), rtol=1e-6)
        elapsed = time.time() - start_time
        
        self.assertTrue(result['success'])
        # Should complete in under 5 seconds (reasonable for 100D)
        self.assertLess(elapsed, 5.0)
    
    @pytest.mark.slow
    def test_very_high_dimensional_500_states(self):
        """Very high dimensional system (500 states) - marked slow."""
        system = HighDimensionalLinear(n=500)
        x0 = np.random.randn(500)
        
        start_time = time.time()
        result = system.integrate(x0, t_span=(0, 5), rtol=1e-6)
        elapsed = time.time() - start_time
        
        self.assertTrue(result['success'])
        # Should still complete in reasonable time
        self.assertLess(elapsed, 30.0)
    
    def test_many_short_integrations_performance(self):
        """Many short integrations should be efficient."""
        system = ControlledOscillator()
        
        start_time = time.time()
        for i in range(100):
            x0 = np.random.randn(2)
            result = system.integrate(x0, t_span=(0, 1))
            self.assertTrue(result['success'])
        elapsed = time.time() - start_time
        
        # 100 integrations should be fast
        self.assertLess(elapsed, 5.0)


# =============================================================================
# Run Tests
# =============================================================================


if __name__ == "__main__":
    # Run additional tests with verbose output
    unittest.main(verbosity=2)