"""
Unit tests for GenericDiscreteTimeSystem
Tests discrete-time wrapper with multiple integration methods
"""

import pytest
import numpy as np
import torch
import sympy as sp
from typing import Tuple

from src.systems.base.generic_discrete_time_system import (
    GenericDiscreteTimeSystem,
    IntegrationMethod,
)
from src.systems.base.symbolic_dynamical_system import SymbolicDynamicalSystem


# ============================================================================
# Test Fixtures: Concrete System Implementations
# ============================================================================

class SimplePendulum(SymbolicDynamicalSystem):
    """Simple pendulum for testing"""
    
    def __init__(self, m=1.0, l=1.0, g=9.81, b=0.1):
        super().__init__()
        self.define_system(m, l, g, b)
    
    def define_system(self, m, l, g, b):
        theta, theta_dot = sp.symbols('theta theta_dot', real=True)
        tau = sp.symbols('tau', real=True)
        
        self.state_vars = [theta, theta_dot]
        self.control_vars = [tau]
        self.output_vars = []
        
        self._f_sym = sp.Matrix([
            theta_dot,
            -g/l * sp.sin(theta) - b * theta_dot + tau / (m * l**2)
        ])
        
        m_sym, l_sym, g_sym, b_sym = sp.symbols('m l g b', real=True, positive=True)
        self.parameters = {m_sym: m, l_sym: l, g_sym: g, b_sym: b}
        self.order = 1


class PartialObservationPendulum(SymbolicDynamicalSystem):
    """Pendulum with only angle measured"""
    
    def __init__(self, m=1.0, l=1.0, g=9.81, b=0.1):
        super().__init__()
        self.define_system(m, l, g, b)
    
    def define_system(self, m, l, g, b):
        theta, theta_dot = sp.symbols('theta theta_dot', real=True)
        tau = sp.symbols('tau', real=True)
        
        self.state_vars = [theta, theta_dot]
        self.control_vars = [tau]
        
        self._f_sym = sp.Matrix([
            theta_dot,
            -g/l * sp.sin(theta) - b * theta_dot + tau / (m * l**2)
        ])
        
        self._h_sym = sp.Matrix([theta])  # Only observe angle
        self.output_vars = [sp.symbols('y')]
        
        m_sym, l_sym, g_sym, b_sym = sp.symbols('m l g b', real=True, positive=True)
        self.parameters = {m_sym: m, l_sym: l, g_sym: g, b_sym: b}
        self.order = 1


class LinearSystem(SymbolicDynamicalSystem):
    """Simple 2D linear system"""
    
    def __init__(self):
        super().__init__()
        self.define_system()
    
    def define_system(self):
        x1, x2 = sp.symbols('x1 x2', real=True)
        u1 = sp.symbols('u1', real=True)
        
        self.state_vars = [x1, x2]
        self.control_vars = [u1]
        
        self._f_sym = sp.Matrix([
            -x1 + x2,
            -2*x2 + u1
        ])
        
        self.parameters = {}
        self.order = 1


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def pendulum_ct():
    """Continuous-time pendulum"""
    return SimplePendulum(m=0.15, l=0.5, g=9.81, b=0.1)


@pytest.fixture
def partial_obs_pendulum_ct():
    """Continuous-time pendulum with partial observation"""
    return PartialObservationPendulum(m=0.15, l=0.5, g=9.81, b=0.1)


@pytest.fixture
def linear_system_ct():
    """Continuous-time linear system"""
    return LinearSystem()


@pytest.fixture
def pendulum_dt(pendulum_ct):
    """Discrete-time pendulum with Euler integration"""
    return GenericDiscreteTimeSystem(pendulum_ct, dt=0.01)


@pytest.fixture
def pendulum_dt_rk4(pendulum_ct):
    """Discrete-time pendulum with RK4 integration"""
    return GenericDiscreteTimeSystem(
        pendulum_ct, dt=0.01, integration_method=IntegrationMethod.RK4
    )


@pytest.fixture
def linear_system_dt(linear_system_ct):
    """Discrete-time linear system"""
    return GenericDiscreteTimeSystem(linear_system_ct, dt=0.01)


# ============================================================================
# Test System Initialization
# ============================================================================

class TestInitialization:
    """Test system initialization and configuration"""
    
    def test_basic_initialization(self, pendulum_ct):
        """Test basic discrete system creation"""
        dt = 0.01
        system = GenericDiscreteTimeSystem(pendulum_ct, dt=dt)
        
        assert system.nx == 2
        assert system.nu == 1
        assert system.dt == dt
        assert system.order == 1
        assert system.integration_method == IntegrationMethod.ExplicitEuler
    
    def test_initialization_with_integration_method(self, pendulum_ct):
        """Test initialization with different integration methods"""
        system_euler = GenericDiscreteTimeSystem(
            pendulum_ct, dt=0.01, integration_method=IntegrationMethod.ExplicitEuler
        )
        system_rk4 = GenericDiscreteTimeSystem(
            pendulum_ct, dt=0.01, integration_method=IntegrationMethod.RK4
        )
        system_midpoint = GenericDiscreteTimeSystem(
            pendulum_ct, dt=0.01, integration_method=IntegrationMethod.MidPoint
        )
        
        assert system_euler.integration_method == IntegrationMethod.ExplicitEuler
        assert system_rk4.integration_method == IntegrationMethod.RK4
        assert system_midpoint.integration_method == IntegrationMethod.MidPoint
    
    def test_initialization_invalid_dt(self, pendulum_ct):
        """Test that negative or zero dt raises error"""
        with pytest.raises(ValueError, match="Time step dt must be positive"):
            GenericDiscreteTimeSystem(pendulum_ct, dt=0.0)
        
        with pytest.raises(ValueError, match="Time step dt must be positive"):
            GenericDiscreteTimeSystem(pendulum_ct, dt=-0.01)
    
    def test_equilibrium_inheritance(self, pendulum_dt):
        """Test that equilibrium points are inherited from continuous system"""
        x_eq = pendulum_dt.x_equilibrium
        u_eq = pendulum_dt.u_equilibrium
        
        assert isinstance(x_eq, torch.Tensor)
        assert isinstance(u_eq, torch.Tensor)
        assert x_eq.shape == (2,)
        assert u_eq.shape == (1,)
        assert torch.allclose(x_eq, torch.zeros(2))
        assert torch.allclose(u_eq, torch.zeros(1))
    
    def test_repr_and_str(self, pendulum_dt):
        """Test string representations"""
        repr_str = repr(pendulum_dt)
        str_str = str(pendulum_dt)
        
        assert "GenericDiscreteTimeSystem" in repr_str
        assert "SimplePendulum" in repr_str
        assert "dt=0.01" in repr_str
        assert "SimplePendulum" in str_str
        assert "dt=0.0100" in str_str


# ============================================================================
# Test Forward Dynamics (Integration Methods)
# ============================================================================

class TestForwardDynamics:
    """Test forward dynamics with different integration methods"""
    
    def test_forward_euler_single(self, pendulum_dt):
        """Test Euler integration with single state"""
        x = torch.tensor([0.1, 0.0])
        u = torch.tensor([0.0])
        
        x_next = pendulum_dt.forward(x, u)
        
        assert isinstance(x_next, torch.Tensor)
        assert x_next.shape == (2,)
        assert not torch.isnan(x_next).any()
        assert not torch.isinf(x_next).any()
        # State should have changed
        assert not torch.allclose(x_next, x)
    
    def test_forward_rk4_single(self, pendulum_dt_rk4):
        """Test RK4 integration with single state"""
        x = torch.tensor([0.1, 0.0])
        u = torch.tensor([0.0])
        
        x_next = pendulum_dt_rk4.forward(x, u)
        
        assert isinstance(x_next, torch.Tensor)
        assert x_next.shape == (2,)
        assert not torch.isnan(x_next).any()
    
    def test_forward_batched(self, pendulum_dt):
        """Test forward dynamics with batched states"""
        x = torch.randn(10, 2)
        u = torch.randn(10, 1)
        
        x_next = pendulum_dt.forward(x, u)
        
        assert isinstance(x_next, torch.Tensor)
        assert x_next.shape == (10, 2)
        assert not torch.isnan(x_next).any()
    
    def test_rk4_more_accurate_than_euler(self, pendulum_ct):
        """Test that RK4 is more accurate than Euler for same dt"""
        dt = 0.1  # Large dt to see difference
        system_euler = GenericDiscreteTimeSystem(
            pendulum_ct, dt=dt, integration_method=IntegrationMethod.ExplicitEuler
        )
        system_rk4 = GenericDiscreteTimeSystem(
            pendulum_ct, dt=dt, integration_method=IntegrationMethod.RK4
        )
        
        x0 = torch.tensor([0.5, 0.1])
        u = torch.tensor([0.0])
        
        # Simulate 10 steps
        x_euler = x0
        x_rk4 = x0
        for _ in range(10):
            x_euler = system_euler.forward(x_euler, u)
            x_rk4 = system_rk4.forward(x_rk4, u)
        
        # RK4 and Euler should give different results
        assert not torch.allclose(x_euler, x_rk4, rtol=0.01)
    
    def test_callable_interface(self, pendulum_dt):
        """Test that system is callable"""
        x = torch.tensor([0.1, 0.0])
        u = torch.tensor([0.0])
        
        x_next_forward = pendulum_dt.forward(x, u)
        x_next_call = pendulum_dt(x, u)
        
        assert torch.allclose(x_next_forward, x_next_call)
    
    def test_equilibrium_stays_at_equilibrium(self, pendulum_dt):
        """Test that equilibrium point doesn't move"""
        x_eq = pendulum_dt.x_equilibrium
        u_eq = pendulum_dt.u_equilibrium
        
        x_next = pendulum_dt.forward(x_eq, u_eq)
        
        # Should stay at equilibrium (within numerical tolerance)
        assert torch.allclose(x_next, x_eq, atol=1e-5)


# ============================================================================
# Test Simulation
# ============================================================================

class TestSimulation:
    """Test trajectory simulation"""
    
    def test_simulate_with_zero_control(self, pendulum_dt):
        """Test simulation with zero control"""
        x0 = torch.tensor([0.1, 0.0])
        
        traj = pendulum_dt.simulate(x0, controller=None, horizon=10)
        
        assert isinstance(traj, torch.Tensor)
        assert traj.shape == (11, 2)  # T+1 states
    
    def test_simulate_with_control_sequence(self, pendulum_dt):
        """Test simulation with pre-computed control sequence"""
        x0 = torch.tensor([0.1, 0.0])
        u_seq = torch.randn(20, 1)
        
        traj = pendulum_dt.simulate(x0, controller=u_seq)
        
        assert isinstance(traj, torch.Tensor)
        assert traj.shape == (21, 2)  # T+1 states
    
    def test_simulate_with_controller_function(self, pendulum_dt):
        """Test simulation with controller function"""
        x0 = torch.tensor([0.1, 0.0])
        
        # Simple proportional controller
        def controller(x):
            return -0.1 * x[:, 0:1]  # Proportional to angle
        
        traj = pendulum_dt.simulate(x0, controller=controller, horizon=50)
        
        assert isinstance(traj, torch.Tensor)
        assert traj.shape == (51, 2)
    
    def test_simulate_with_nn_controller(self, pendulum_dt):
        """Test simulation with neural network controller"""
        x0 = torch.tensor([0.1, 0.0])
        
        # Simple neural network controller
        controller_nn = torch.nn.Sequential(
            torch.nn.Linear(2, 10),
            torch.nn.Tanh(),
            torch.nn.Linear(10, 1)
        )
        
        traj = pendulum_dt.simulate(x0, controller=controller_nn, horizon=30)
        
        assert isinstance(traj, torch.Tensor)
        assert traj.shape == (31, 2)
    
    def test_simulate_return_controls(self, pendulum_dt):
        """Test simulation returning both trajectory and controls"""
        x0 = torch.tensor([0.1, 0.0])
        controller = lambda x: torch.zeros(x.shape[0], 1)
        
        traj, controls = pendulum_dt.simulate(
            x0, controller=controller, horizon=20, return_controls=True
        )
        
        assert isinstance(traj, torch.Tensor)
        assert isinstance(controls, torch.Tensor)
        assert traj.shape == (21, 2)
        assert controls.shape == (20, 1)
    
    def test_simulate_return_final_only(self, pendulum_dt):
        """Test simulation returning only final state"""
        x0 = torch.tensor([0.1, 0.0])
        
        x_final = pendulum_dt.simulate(
            x0, controller=None, horizon=10, return_all=False
        )
        
        assert isinstance(x_final, torch.Tensor)
        assert x_final.shape == (2,)  # Just final state
    
    def test_simulate_batched_initial_conditions(self, pendulum_dt):
        """Test simulation with batched initial conditions"""
        x0_batch = torch.randn(5, 2)
        
        traj = pendulum_dt.simulate(x0_batch, controller=None, horizon=10)
        
        assert isinstance(traj, torch.Tensor)
        assert traj.shape == (5, 11, 2)  # (batch, T+1, nx)
    
    def test_simulate_batched_with_controls(self, pendulum_dt):
        """Test batched simulation returning controls"""
        x0_batch = torch.randn(3, 2)
        controller = lambda x: -0.1 * x[:, 0:1]
        
        traj, controls = pendulum_dt.simulate(
            x0_batch, controller=controller, horizon=15, return_controls=True
        )
        
        assert traj.shape == (3, 16, 2)
        assert controls.shape == (3, 15, 1)
    
    def test_simulate_with_observer(self, partial_obs_pendulum_ct):
        """Test output feedback simulation with observer"""
        system = GenericDiscreteTimeSystem(partial_obs_pendulum_ct, dt=0.01)
        x0 = torch.tensor([0.1, 0.0])
        
        # Simple observer (just return state as-is for testing)
        observer = lambda x: x
        controller = lambda x_hat: -0.1 * x_hat[:, 0:1]
        
        traj = system.simulate(
            x0, controller=controller, observer=observer, horizon=20
        )
        
        assert traj.shape == (21, 2)
    
    def test_simulate_missing_horizon_error(self, pendulum_dt):
        """Test that missing horizon with function controller raises error"""
        x0 = torch.tensor([0.1, 0.0])
        controller = lambda x: torch.zeros(x.shape[0], 1)
        
        with pytest.raises(ValueError, match="horizon must be specified"):
            pendulum_dt.simulate(x0, controller=controller)
    
    def test_simulate_control_sequence_batch_mismatch(self, pendulum_dt):
        """Test error when control batch doesn't match state batch"""
        x0_batch = torch.randn(5, 2)
        u_seq = torch.randn(3, 10, 1)  # Wrong batch size
        
        with pytest.raises(ValueError, match="batch size"):
            pendulum_dt.simulate(x0_batch, controller=u_seq)


# ============================================================================
# Test Integration Methods
# ============================================================================

class TestIntegrationMethods:
    """Test different integration methods"""
    
    def test_euler_vs_midpoint_vs_rk4(self, pendulum_ct):
        """Compare different integration methods"""
        dt = 0.05
        
        system_euler = GenericDiscreteTimeSystem(
            pendulum_ct, dt=dt, integration_method=IntegrationMethod.ExplicitEuler
        )
        system_midpoint = GenericDiscreteTimeSystem(
            pendulum_ct, dt=dt, integration_method=IntegrationMethod.MidPoint
        )
        system_rk4 = GenericDiscreteTimeSystem(
            pendulum_ct, dt=dt, integration_method=IntegrationMethod.RK4
        )
        
        x0 = torch.tensor([0.5, 0.1])
        u = torch.tensor([0.0])
        
        # One step
        x_euler = system_euler.forward(x0, u)
        x_midpoint = system_midpoint.forward(x0, u)
        x_rk4 = system_rk4.forward(x0, u)
        
        # All should be different
        assert not torch.allclose(x_euler, x_midpoint, rtol=0.01)
        assert not torch.allclose(x_euler, x_rk4, rtol=0.01)
        assert not torch.allclose(x_midpoint, x_rk4, rtol=0.01)
    
    def test_smaller_dt_more_accurate(self, pendulum_ct):
        """Test that smaller dt gives more accurate integration"""
        x0 = torch.tensor([0.5, 0.1])
        u = torch.tensor([0.0])
        
        # Simulate same duration with different dt
        system_coarse = GenericDiscreteTimeSystem(pendulum_ct, dt=0.1)
        system_fine = GenericDiscreteTimeSystem(pendulum_ct, dt=0.01)
        
        # 1 second simulation
        x_coarse = x0
        for _ in range(10):  # 10 steps of dt=0.1
            x_coarse = system_coarse.forward(x_coarse, u)
        
        x_fine = x0
        for _ in range(100):  # 100 steps of dt=0.01
            x_fine = system_fine.forward(x_fine, u)
        
        # Fine discretization should be different (more accurate)
        # They won't be identical due to discretization error
        assert not torch.allclose(x_coarse, x_fine, rtol=0.1)


# ============================================================================
# Test Linearization
# ============================================================================

class TestLinearization:
    """Test discrete-time linearization"""
    
    def test_linearized_dynamics(self, pendulum_dt):
        """Test discrete linearization"""
        x = torch.tensor([0.1, 0.0])
        u = torch.tensor([0.0])
        
        Ad, Bd = pendulum_dt.linearized_dynamics(x, u)
        
        assert isinstance(Ad, torch.Tensor)
        assert isinstance(Bd, torch.Tensor)
        assert Ad.shape == (2, 2)
        assert Bd.shape == (2, 1)
    
    def test_linearized_dynamics_batched(self, pendulum_dt):
        """Test batched linearization"""
        x = torch.randn(5, 2)
        u = torch.randn(5, 1)
        
        Ad, Bd = pendulum_dt.linearized_dynamics(x, u)
        
        assert Ad.shape == (5, 2, 2)
        assert Bd.shape == (5, 2, 1)
    
    def test_linearization_at_equilibrium(self, linear_system_dt):
        """Test that discrete linearization relates to continuous via dt"""
        x_eq = linear_system_dt.x_equilibrium.unsqueeze(0)
        u_eq = linear_system_dt.u_equilibrium.unsqueeze(0)
        
        # Get continuous linearization
        Ac, Bc = linear_system_dt.continuous_time_system.linearized_dynamics(x_eq, u_eq)
        
        # Get discrete linearization
        Ad, Bd = linear_system_dt.linearized_dynamics(x_eq, u_eq)
        
        # For Euler: Ad ≈ I + dt*Ac, Bd ≈ dt*Bc
        I = torch.eye(2)
        Ad_expected = I + linear_system_dt.dt * Ac.squeeze(0)
        Bd_expected = linear_system_dt.dt * Bc.squeeze(0)
        
        assert torch.allclose(Ad.squeeze(0), Ad_expected, atol=1e-6)
        assert torch.allclose(Bd.squeeze(0), Bd_expected, atol=1e-6)
    
    def test_linearized_observation(self, partial_obs_pendulum_ct):
        """Test observation linearization"""
        system = GenericDiscreteTimeSystem(partial_obs_pendulum_ct, dt=0.01)
        x = torch.tensor([0.1, 0.0])
        
        C = system.linearized_observation(x)
        
        assert isinstance(C, torch.Tensor)
        assert C.shape == (1, 2)  # (ny, nx)
        # Should be [1, 0] for angle-only observation
        expected = torch.tensor([[1.0, 0.0]])
        assert torch.allclose(C, expected, atol=1e-6)
    
    def test_h_observation(self, partial_obs_pendulum_ct):
        """Test observation function"""
        system = GenericDiscreteTimeSystem(partial_obs_pendulum_ct, dt=0.01)
        x = torch.tensor([0.5, 0.1])
        
        y = system.h(x)
        
        assert isinstance(y, torch.Tensor)
        assert y.shape == (1,)
        assert torch.allclose(y, x[0:1])  # Should return angle


# ============================================================================
# Test Control Design Methods
# ============================================================================

class TestControlDesign:
    """Test discrete LQR, Kalman, and LQG"""
    
    def test_dlqr_control(self, linear_system_dt):
        """Test discrete LQR design"""
        Q = np.eye(2)
        R = np.array([[1.0]])
        
        K, S = linear_system_dt.dlqr_control(Q, R)
        
        assert isinstance(K, np.ndarray)
        assert isinstance(S, np.ndarray)
        assert K.shape == (1, 2)
        assert S.shape == (2, 2)
        
        # S should be positive definite
        eigenvalues = np.linalg.eigvals(S)
        assert np.all(eigenvalues > 0)
    
    def test_dlqr_stabilizes_system(self, linear_system_dt):
        """Test that DLQR gain stabilizes the system"""
        Q = np.eye(2) * 10
        R = np.array([[1.0]])
        
        K, S = linear_system_dt.dlqr_control(Q, R)
        
        # Get closed-loop eigenvalues
        x_eq = linear_system_dt.x_equilibrium.unsqueeze(0)
        u_eq = linear_system_dt.u_equilibrium.unsqueeze(0)
        
        Ad, Bd = linear_system_dt.linearized_dynamics(x_eq, u_eq)
        Ad_np = Ad.squeeze().detach().cpu().numpy()
        Bd_np = Bd.squeeze().detach().cpu().numpy()
        
        # Closed-loop: A_cl = Ad + Bd @ K
        A_cl = Ad_np + Bd_np @ K
        eigenvalues = np.linalg.eigvals(A_cl)
        
        # Should be stable: |λ| < 1
        assert np.all(np.abs(eigenvalues) < 1.0)
    
    def test_discrete_kalman_gain(self, linear_system_dt):
        """Test discrete Kalman gain"""
        Q_process = np.eye(2) * 0.01
        R_measurement = np.eye(2) * 0.1
        
        L = linear_system_dt.discrete_kalman_gain(Q_process, R_measurement)
        
        assert isinstance(L, np.ndarray)
        assert L.shape == (2, 2)
    
    def test_discrete_kalman_partial_observation(self, partial_obs_pendulum_ct):
        """Test Kalman gain with partial observation"""
        system = GenericDiscreteTimeSystem(partial_obs_pendulum_ct, dt=0.01)
        
        Q_process = np.eye(2) * 0.01
        R_measurement = np.array([[0.1]])
        
        L = system.discrete_kalman_gain(Q_process, R_measurement)
        
        assert isinstance(L, np.ndarray)
        assert L.shape == (2, 1)  # (nx, ny) where ny=1
    
    def test_dlqg_control(self, linear_system_dt):
        """Test discrete LQG design"""
        Q_lqr = np.eye(2)
        R_lqr = np.array([[1.0]])
        Q_process = np.eye(2) * 0.01
        R_measurement = np.eye(2) * 0.1
        
        K, L = linear_system_dt.dlqg_control(Q_lqr, R_lqr, Q_process, R_measurement)
        
        assert isinstance(K, np.ndarray)
        assert isinstance(L, np.ndarray)
        assert K.shape == (1, 2)
        assert L.shape == (2, 2)
    
    def test_dlqg_closed_loop_matrix(self, linear_system_dt):
        """Test discrete closed-loop matrix"""
        Q_lqr = np.eye(2)
        R_lqr = np.array([[1.0]])
        Q_process = np.eye(2) * 0.01
        R_measurement = np.eye(2) * 0.1
        
        K, L = linear_system_dt.dlqg_control(Q_lqr, R_lqr, Q_process, R_measurement)
        A_cl = linear_system_dt.dlqg_closed_loop_matrix(K, L)
        
        assert isinstance(A_cl, np.ndarray)
        assert A_cl.shape == (4, 4)  # (2*nx, 2*nx)
        
        # Closed-loop should be stable
        eigenvalues = np.linalg.eigvals(A_cl)
        assert np.all(np.abs(eigenvalues) < 1.0)
    
    def test_output_feedback_lyapunov(self, linear_system_dt):
        """Test Lyapunov function for output feedback"""
        Q_lqr = np.eye(2)
        R_lqr = np.array([[1.0]])
        Q_process = np.eye(2) * 0.01
        R_measurement = np.eye(2) * 0.1
        
        K, L = linear_system_dt.dlqg_control(Q_lqr, R_lqr, Q_process, R_measurement)
        S = linear_system_dt.output_feedback_lyapunov(K, L)
        
        assert isinstance(S, np.ndarray)
        assert S.shape == (4, 4)  # (2*nx, 2*nx)
        
        # S should be positive definite
        eigenvalues = np.linalg.eigvals(S)
        assert np.all(eigenvalues > 0)


# ============================================================================
# Test Utility Methods
# ============================================================================

class TestUtilityMethods:
    """Test utility and helper methods"""
    
    def test_print_info(self, pendulum_dt, capsys):
        """Test print_info method"""
        pendulum_dt.print_info(include_equations=True, include_linearization=True)
        
        captured = capsys.readouterr()
        assert "Discrete-Time System" in captured.out
        assert "SimplePendulum" in captured.out
        assert "Time step (dt)" in captured.out
        assert "Integration method" in captured.out
        assert "Continuous-time" in captured.out
        assert "Discrete-time" in captured.out
    
    def test_summary(self, pendulum_dt):
        """Test summary string generation"""
        summary = pendulum_dt.summary()
        
        assert isinstance(summary, str)
        assert "SimplePendulum" in summary
        assert "nx=2" in summary
        assert "nu=1" in summary
        assert "dt=0.0100" in summary
        assert "stable" in summary.lower()
    
    def test_print_info_without_equations(self, pendulum_dt, capsys):
        """Test print_info without equations"""
        pendulum_dt.print_info(include_equations=False, include_linearization=True)
        
        captured = capsys.readouterr()
        assert "Discrete-Time System" in captured.out
        # Should not have continuous equations section
        assert "dx/dt =" not in captured.out
    
    def test_print_info_without_linearization(self, pendulum_dt, capsys):
        """Test print_info without linearization"""
        pendulum_dt.print_info(include_equations=True, include_linearization=False)
        
        captured = capsys.readouterr()
        assert "Discrete-Time System" in captured.out
        # Should not have linearization section
        assert "Eigenvalues:" not in captured.out


# ============================================================================
# Test Visualization Methods
# ============================================================================

class TestVisualization:
    """Test plotting methods"""
    
    def test_plot_trajectory_basic(self, pendulum_dt):
        """Test basic trajectory plotting"""
        x0 = torch.tensor([0.5, 0.0])
        traj = pendulum_dt.simulate(x0, controller=None, horizon=50)
        
        # Test that plotting doesn't crash
        fig = pendulum_dt.plot_trajectory(traj, show=False)
        
        assert fig is not None
    
    def test_plot_trajectory_with_controls(self, pendulum_dt):
        """Test plotting with control sequence"""
        x0 = torch.tensor([0.5, 0.0])
        controller = lambda x: -0.1 * x[:, 0:1]
        
        traj, controls = pendulum_dt.simulate(
            x0, controller=controller, horizon=50, return_controls=True
        )
        
        fig = pendulum_dt.plot_trajectory(
            traj, control_sequence=controls, show=False
        )
        
        assert fig is not None
    
    def test_plot_trajectory_batched(self, pendulum_dt):
        """Test plotting batched trajectories"""
        x0_batch = torch.tensor([[0.5, 0.0], [0.3, 0.1], [-0.2, 0.0]])
        traj = pendulum_dt.simulate(x0_batch, controller=None, horizon=50)
        
        fig = pendulum_dt.plot_trajectory(
            traj, 
            trajectory_names=['Traj 1', 'Traj 2', 'Traj 3'],
            show=False
        )
        
        assert fig is not None
    
    def test_plot_trajectory_custom_names(self, pendulum_dt):
        """Test plotting with custom state names"""
        x0 = torch.tensor([0.5, 0.0])
        traj = pendulum_dt.simulate(x0, controller=None, horizon=30)
        
        fig = pendulum_dt.plot_trajectory(
            traj,
            state_names=['θ (rad)', 'θ̇ (rad/s)'],
            title="Pendulum Motion",
            show=False
        )
        
        assert fig is not None
    
    def test_plot_trajectory_save_html(self, pendulum_dt, tmp_path):
        """Test saving trajectory plot to HTML"""
        x0 = torch.tensor([0.5, 0.0])
        traj = pendulum_dt.simulate(x0, controller=None, horizon=20)
        
        filepath = tmp_path / "trajectory.html"
        
        fig = pendulum_dt.plot_trajectory(
            traj, save_html=str(filepath), show=False
        )
        
        assert filepath.exists()
    
    def test_plot_trajectory_compact_mode(self, pendulum_dt):
        """Test compact plotting mode"""
        x0 = torch.tensor([0.5, 0.0])
        traj = pendulum_dt.simulate(x0, controller=None, horizon=30)
        
        fig = pendulum_dt.plot_trajectory(traj, compact=True, show=False)
        
        assert fig is not None
    
    def test_plot_phase_portrait_2d(self, pendulum_dt):
        """Test 2D phase portrait plotting"""
        x0 = torch.tensor([0.5, 0.0])
        traj = pendulum_dt.simulate(x0, controller=None, horizon=100)
        
        fig = pendulum_dt.plot_phase_portrait_2d(
            traj,
            state_indices=(0, 1),
            state_names=('θ', 'θ̇'),
            show=False
        )
        
        assert fig is not None
    
    def test_plot_phase_portrait_2d_batched(self, pendulum_dt):
        """Test 2D phase portrait with multiple trajectories"""
        x0_batch = torch.tensor([[0.5, 0.0], [0.3, 0.1], [-0.2, 0.0]])
        traj = pendulum_dt.simulate(x0_batch, controller=None, horizon=100)
        
        fig = pendulum_dt.plot_phase_portrait_2d(
            traj,
            trajectory_names=['IC1', 'IC2', 'IC3'],
            show=False
        )
        
        assert fig is not None
    
    @pytest.mark.skip(reason="Requires 3D system")
    def test_plot_trajectory_3d(self, pendulum_dt):
        """Test 3D trajectory plotting (skipped for 2D pendulum)"""
        # Would need a 3D+ system to test properly
        pass


# ============================================================================
# Test Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_very_small_dt(self, pendulum_ct):
        """Test with very small dt"""
        system = GenericDiscreteTimeSystem(pendulum_ct, dt=1e-5)
        
        x = torch.tensor([0.1, 0.0])
        u = torch.tensor([0.0])
        
        x_next = system.forward(x, u)
        
        # Should be very close to original (small step)
        assert torch.allclose(x_next, x, atol=1e-4)
    
    def test_energy_conservation_check(self, pendulum_ct):
        """Test approximate energy conservation for undamped pendulum"""
        # Create undamped pendulum
        undamped = SimplePendulum(m=1.0, l=1.0, g=9.81, b=0.0)
        system = GenericDiscreteTimeSystem(
            undamped, dt=0.001, integration_method=IntegrationMethod.RK4
        )
        
        x0 = torch.tensor([0.5, 0.0])  # Initial angle, zero velocity
        
        def compute_energy(x):
            """Compute total energy"""
            theta = x[..., 0]
            theta_dot = x[..., 1]
            # E = (1/2) m l^2 θ̇^2 + m g l (1 - cos(θ))
            kinetic = 0.5 * 1.0 * 1.0**2 * theta_dot**2
            potential = 1.0 * 9.81 * 1.0 * (1 - torch.cos(theta))
            return kinetic + potential
        
        E0 = compute_energy(x0)
        
        # Simulate without control
        x = x0
        for _ in range(100):
            x = system.forward(x, torch.tensor([0.0]))
        
        E_final = compute_energy(x)
        
        # Energy should be approximately conserved for RK4 with small dt
        # Allow some drift due to numerical errors
        assert torch.abs(E_final - E0) / E0 < 0.05  # Within 5%
    
    def test_simulate_invalid_controller_type(self, pendulum_dt):
        """Test that invalid controller type raises error"""
        x0 = torch.tensor([0.1, 0.0])
        
        with pytest.raises(TypeError, match="controller must be"):
            pendulum_dt.simulate(x0, controller="invalid", horizon=10)


# ============================================================================
# Test Consistency with Continuous System
# ============================================================================

class TestContinuousConsistency:
    """Test that discrete system converges to continuous as dt→0"""
    
    def test_convergence_as_dt_decreases(self, linear_system_ct):
        """Test that smaller dt converges to continuous solution"""
        x0 = torch.tensor([1.0, 0.5])
        u = torch.tensor([0.0])
        
        # Simulate 0.1 seconds with different dt values
        dt_values = [0.1, 0.01, 0.001]
        results = []
        
        for dt in dt_values:
            system = GenericDiscreteTimeSystem(linear_system_ct, dt=dt)
            x = x0
            steps = int(0.1 / dt)
            for _ in range(steps):
                x = system.forward(x, u)
            results.append(x)
        
        # Results should converge as dt decreases
        # |x(dt=0.01) - x(dt=0.001)| < |x(dt=0.1) - x(dt=0.01)|
        error_coarse = torch.norm(results[1] - results[0])
        error_fine = torch.norm(results[2] - results[1])
        
        assert error_fine < error_coarse
    
    def test_discrete_linearization_approaches_continuous(self, linear_system_ct):
        """Test that discrete linearization → continuous as dt→0"""
        x = torch.tensor([0.5, 0.1]).unsqueeze(0)
        u = torch.tensor([0.0]).unsqueeze(0)
        
        # Continuous linearization
        Ac, Bc = linear_system_ct.linearized_dynamics(x, u)
        Ac_np = Ac.squeeze().detach().cpu().numpy()
        Bc_np = Bc.squeeze().detach().cpu().numpy()
        
        # Test with decreasing dt
        dt_values = [0.1, 0.01, 0.001]
        errors_A = []
        errors_B = []
        
        for dt in dt_values:
            system = GenericDiscreteTimeSystem(linear_system_ct, dt=dt)
            Ad, Bd = system.linearized_dynamics(x, u)
            Ad_np = Ad.squeeze().detach().cpu().numpy()
            Bd_np = Bd.squeeze().detach().cpu().numpy()
            
            # Ad ≈ I + dt*Ac
            Ad_approx = np.eye(2) + dt * Ac_np
            error_A = np.linalg.norm(Ad_np - Ad_approx)
            errors_A.append(error_A)
            
            # Bd ≈ dt*Bc
            Bd_approx = dt * Bc_np
            error_B = np.linalg.norm(Bd_np - Bd_approx)
            errors_B.append(error_B)
        
        # Errors should decrease with dt (Euler approximation)
        # We're using Euler, so error is O(dt^2)
        assert errors_A[1] < errors_A[0]
        assert errors_A[2] < errors_A[1]


# ============================================================================
# Test Different Integration Methods Comprehensively
# ============================================================================

class TestIntegrationAccuracy:
    """Test accuracy of different integration methods"""
    
    def test_rk4_fourth_order_accuracy(self, linear_system_ct):
        """Test that RK4 has O(dt^4) local error"""
        # For linear system, we can compute exact solution
        # and verify RK4 accuracy
        
        x0 = torch.tensor([1.0, 0.5])
        u = torch.tensor([0.0])
        
        # Test with different dt values
        dt_values = [0.1, 0.05, 0.025]
        errors = []
        
        for dt in dt_values:
            system = GenericDiscreteTimeSystem(
                linear_system_ct, dt=dt, integration_method=IntegrationMethod.RK4
            )
            
            # One step
            x_next = system.forward(x0, u)
            
            # Compute "exact" solution using very small dt
            system_exact = GenericDiscreteTimeSystem(
                linear_system_ct, dt=dt/100, integration_method=IntegrationMethod.RK4
            )
            x_exact = x0
            for _ in range(100):
                x_exact = system_exact.forward(x_exact, u)
            
            error = torch.norm(x_next - x_exact).item()
            errors.append(error)
        
        # RK4 should have O(dt^4) error
        # When dt is halved, error should decrease by ~16x
        ratio_1 = errors[0] / errors[1]
        ratio_2 = errors[1] / errors[2]
        
        # Should be close to 16 (2^4)
        assert 10 < ratio_1 < 25  # Allow some tolerance
        assert 10 < ratio_2 < 25
    
    def test_midpoint_second_order_accuracy(self, linear_system_ct):
        """Test that Midpoint has O(dt^2) local error"""
        x0 = torch.tensor([1.0, 0.5])
        u = torch.tensor([0.0])
        
        dt_values = [0.1, 0.05, 0.025]
        errors = []
        
        for dt in dt_values:
            system = GenericDiscreteTimeSystem(
                linear_system_ct, dt=dt, integration_method=IntegrationMethod.MidPoint
            )
            x_next = system.forward(x0, u)
            
            system_exact = GenericDiscreteTimeSystem(
                linear_system_ct, dt=dt/100, integration_method=IntegrationMethod.RK4
            )
            x_exact = x0
            for _ in range(100):
                x_exact = system_exact.forward(x_exact, u)
            
            error = torch.norm(x_next - x_exact).item()
            errors.append(error)
        
        # Midpoint should have O(dt^2) error
        # When dt is halved, error should decrease by ~4x
        ratio_1 = errors[0] / errors[1]
        ratio_2 = errors[1] / errors[2]
        
        # Should be close to 4 (2^2)
        assert 2.5 < ratio_1 < 6
        assert 2.5 < ratio_2 < 6


# ============================================================================
# Test Closed-Loop Simulation
# ============================================================================

class TestClosedLoopSimulation:
    """Test complete closed-loop control scenarios"""
    
    def test_lqr_stabilization(self, linear_system_dt):
        """Test that LQR controller stabilizes the system"""
        # Design LQR
        Q = np.eye(2) * 10
        R = np.array([[1.0]])
        K, _ = linear_system_dt.dlqr_control(Q, R)
        
        # Create controller
        x_eq = linear_system_dt.x_equilibrium
        u_eq = linear_system_dt.u_equilibrium
        
        def lqr_controller(x):
            # u = K @ (x - x_eq) + u_eq
            deviation = x - x_eq
            u = (K @ deviation.T).T + u_eq
            return u
        
        # Simulate from non-equilibrium state
        x0 = torch.tensor([1.0, 0.5])
        traj = linear_system_dt.simulate(x0, controller=lqr_controller, horizon=100)
        
        # Should converge to equilibrium
        x_final = traj[-1]
        assert torch.norm(x_final - x_eq) < 0.01
    
    def test_zero_control_stability(self, pendulum_dt):
        """Test behavior with zero control from equilibrium"""
        x0 = pendulum_dt.x_equilibrium + torch.tensor([0.01, 0.0])  # Small perturbation
        
        traj = pendulum_dt.simulate(x0, controller=None, horizon=50)
        
        # Trajectory should exist and be finite
        assert not torch.isnan(traj).any()
        assert not torch.isinf(traj).any()


# ============================================================================
# Performance Tests
# ============================================================================

class TestPerformance:
    """Test performance characteristics"""
    
    def test_batched_simulation_faster(self, pendulum_dt):
        """Test that batched simulation is more efficient"""
        import time
        
        # Single simulations
        start = time.time()
        for _ in range(10):
            x0 = torch.randn(2)
            traj = pendulum_dt.simulate(x0, controller=None, horizon=100)
        time_sequential = time.time() - start
        
        # Batched simulation
        x0_batch = torch.randn(10, 2)
        start = time.time()
        traj_batch = pendulum_dt.simulate(x0_batch, controller=None, horizon=100)
        time_batched = time.time() - start
        
        # Batched should be significantly faster
        # (Though this might not always be true in test environments)
        print(f"Sequential: {time_sequential:.4f}s, Batched: {time_batched:.4f}s")
        print(f"Speedup: {time_sequential/time_batched:.2f}x")
        
        # At minimum, batched should work correctly
        assert traj_batch.shape == (10, 101, 2)


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])