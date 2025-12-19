from typing import Optional, Tuple, Union
import numpy as np
import scipy.linalg
from src.systems.base.symbolic_dynamical_system import SymbolicDynamicalSystem
from src.systems.base.generic_discrete_time_system import GenericDiscreteTimeSystem

class ControlDesigner:
    """
    Unified control design interface for both continuous and discrete systems.
    
    Handles:
    - LQR (Linear Quadratic Regulator)
    - Kalman filtering
    - LQG (combined controller + observer)
    
    Works with both SymbolicDynamicalSystem and GenericDiscreteTimeSystem.
    """
    
    def __init__(self, system):
        """
        Initialize control designer.
        
        Args:
            system: Either SymbolicDynamicalSystem or GenericDiscreteTimeSystem
        """
        
        self.system = system
        
        # Detect system type
        if isinstance(system, GenericDiscreteTimeSystem):
            self.is_discrete = True
            self.nx = system.nx
            self.nu = system.nu
            self.ny = system.continuous_time_system.ny
        elif isinstance(system, SymbolicDynamicalSystem):
            self.is_discrete = False
            self.nx = system.nx
            self.nu = system.nu
            self.ny = system.ny
        else:
            raise TypeError(f"Unknown system type: {type(system)}")
    
    def lqr(self, Q: np.ndarray, R: np.ndarray,
            equilibrium: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Design LQR controller (automatically handles continuous vs discrete).
        
        Args:
            Q: State cost matrix (nx, nx)
            R: Control cost matrix (nu, nu) or scalar
            equilibrium: Named equilibrium (uses default if None)
        
        Returns:
            K: Control gain (nu, nx)
            S: Riccati solution (nx, nx)
        """
        # Get equilibrium point
        x_eq, u_eq = self._get_equilibrium(equilibrium)
        
        # Get linearization
        A, B = self._linearize(x_eq, u_eq)
        
        # Ensure proper dimensions
        B = self._ensure_2d(B)
        R = self._ensure_2d_cost(R, self.nu)
        
        # Solve appropriate Riccati equation
        if self.is_discrete:
            S = scipy.linalg.solve_discrete_are(A, B, Q, R)
            K = -np.linalg.solve(R + B.T @ S @ B, B.T @ S @ A)
        else:
            S = scipy.linalg.solve_continuous_are(A, B, Q, R)
            K = -np.linalg.solve(R, B.T @ S)
        
        return K, S
    
    def kalman(self, Q_process: Optional[np.ndarray] = None,
               R_measurement: Optional[np.ndarray] = None,
               equilibrium: Optional[str] = None) -> np.ndarray:
        """
        Design Kalman filter gain (automatically handles continuous vs discrete).
        
        Args:
            Q_process: Process noise covariance (nx, nx)
            R_measurement: Measurement noise covariance (ny, ny) or scalar
            equilibrium: Named equilibrium
        
        Returns:
            L: Kalman gain (nx, ny)
        """
        # Defaults
        if Q_process is None:
            Q_process = np.eye(self.nx) * 1e-3
        if R_measurement is None:
            R_measurement = np.eye(self.ny) * 1e-3
        
        # Get equilibrium and linearization
        x_eq, u_eq = self._get_equilibrium(equilibrium)
        A, _ = self._linearize(x_eq, u_eq)
        C = self._get_observation_matrix(x_eq)
        
        # Ensure proper dimensions
        C = self._ensure_2d(C)
        R_measurement = self._ensure_2d_cost(R_measurement, self.ny)
        
        # Solve dual Riccati equation
        if self.is_discrete:
            P = scipy.linalg.solve_discrete_are(A.T, C.T, Q_process, R_measurement)
            L = P @ C.T @ np.linalg.inv(C @ P @ C.T + R_measurement)
        else:
            P = scipy.linalg.solve_continuous_are(A.T, C.T, Q_process, R_measurement)
            L = P @ C.T @ np.linalg.inv(R_measurement)
        
        return L
    
    def lqg(self, Q_lqr: np.ndarray, R_lqr: np.ndarray,
            Q_process: Optional[np.ndarray] = None,
            R_measurement: Optional[np.ndarray] = None,
            equilibrium: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Design LQG controller (LQR + Kalman).
        
        Returns:
            K: Control gain (nu, nx)
            L: Observer gain (nx, ny)
        """
        K, _ = self.lqr(Q_lqr, R_lqr, equilibrium)
        L = self.kalman(Q_process, R_measurement, equilibrium)
        return K, L
    
    def closed_loop_matrix(self, K: np.ndarray, L: np.ndarray,
                          equilibrium: Optional[str] = None) -> np.ndarray:
        """
        Compute closed-loop system matrix for LQG control.
        
        Returns:
            A_cl: Closed-loop matrix (2*nx, 2*nx)
        """
        x_eq, u_eq = self._get_equilibrium(equilibrium)
        A, B = self._linearize(x_eq, u_eq)
        C = self._get_observation_matrix(x_eq)
        
        # Ensure proper shapes
        B = self._ensure_2d(B)
        C = self._ensure_2d(C)
        K = self._ensure_2d(K, shape=(self.nu, self.nx))
        L = self._ensure_2d(L, shape=(self.nx, self.ny))
        
        # Closed-loop: [x, x̂]
        A_cl = np.vstack([
            np.hstack([A + B @ K, -B @ K]),
            np.hstack([L @ C, A + B @ K - L @ C])
        ])
        
        # Clean up near-zero
        A_cl[np.abs(A_cl) <= 1e-6] = 0
        
        return A_cl
    
    # Helper methods
    
    def _get_equilibrium(self, name: Optional[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Get equilibrium from system in NumPy format"""
        if self.is_discrete:
            # Discrete system wraps continuous
            handler = self.system.continuous_time_system.equilibria
        else:
            handler = self.system.equilibria
        
        x_eq = handler.get_x(name, backend='numpy')
        u_eq = handler.get_u(name, backend='numpy')
        return x_eq, u_eq
    
    def _linearize(self, x_eq: np.ndarray, u_eq: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Get linearization at equilibrium"""
        import torch
        
        # Convert to torch for linearization
        x_torch = torch.tensor(x_eq, dtype=torch.float32).unsqueeze(0)
        u_torch = torch.tensor(u_eq, dtype=torch.float32).unsqueeze(0)
        
        if self.is_discrete:
            A, B = self.system.linearized_dynamics(x_torch, u_torch)
        else:
            A, B = self.system.linearized_dynamics(x_torch, u_torch)
        
        # Convert back to NumPy
        A = A.squeeze().detach().cpu().numpy()
        B = B.squeeze().detach().cpu().numpy()
        
        return A, B
    
    def _get_observation_matrix(self, x_eq: np.ndarray) -> np.ndarray:
        """Get observation matrix at equilibrium"""
        import torch
        
        x_torch = torch.tensor(x_eq, dtype=torch.float32).unsqueeze(0)
        
        if self.is_discrete:
            C = self.system.continuous_time_system.linearized_observation(x_torch)
        else:
            C = self.system.linearized_observation(x_torch)
        
        return C.squeeze().detach().cpu().numpy()
    
    @staticmethod
    def _ensure_2d(arr: np.ndarray, shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """Ensure array is 2D with optional shape validation"""
        if arr.ndim == 1:
            if shape is not None:
                arr = arr.reshape(shape)
            else:
                arr = arr.reshape(-1, 1)
        return arr
    
    @staticmethod
    def _ensure_2d_cost(cost, dim: int) -> np.ndarray:
        """Ensure cost matrix is 2D"""
        if isinstance(cost, (int, float)):
            return np.array([[cost]])
        elif cost.ndim == 1:
            return np.diag(cost)
        return cost