import numpy as np
import scipy
import torch
import torch.nn as nn
from typing import List, Tuple, Union, Optional, Callable
from enum import Enum
import control
import warnings
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from src.systems.base.symbolic_dynamical_system import SymbolicDynamicalSystem


class IntegrationMethod(Enum):
    """Available numerical integration methods"""

    ExplicitEuler = 1
    MidPoint = 2
    RK4 = 3


class GenericDiscreteTimeSystem(nn.Module):
    """
    Generic discrete-time system for arbitrary order continuous systems.

    Automatically handles first-order, second-order, and higher-order systems
    using various numerical integration methods.

    Attributes:
        continuous_time_system: The underlying continuous-time system
        dt: Integration time step
        order: System order (inherited from continuous system)
        integration_method: Method for integrating derivatives
        position_integration: Method for integrating positions (order > 1)
    """

    def __init__(
        self,
        continuous_time_system: SymbolicDynamicalSystem,
        dt: float,
        integration_method: IntegrationMethod = IntegrationMethod.ExplicitEuler,
        position_integration: Optional[IntegrationMethod] = None,
    ):
        """
        Initialize discrete-time system wrapper

        Args:
            continuous_time_system: Symbolic dynamical system
            dt: Time step for discretization
            integration_method: Method for velocity/derivative integration
            position_integration: Method for position integration (order > 1)
        """
        super().__init__()

        # Validate continuous system
        if (
            not hasattr(continuous_time_system, "_initialized")
            or not continuous_time_system._initialized
        ):
            continuous_time_system._validate_system()

        self.continuous_time_system = continuous_time_system
        self.nx = continuous_time_system.nx
        self.nu = continuous_time_system.nu
        self.dt = float(dt)
        self.order = continuous_time_system.order
        self.integration_method = integration_method
        self.position_integration = position_integration or integration_method
        self.Ix = torch.eye(self.nx)

        if dt <= 0:
            raise ValueError(f"Time step dt must be positive, got {dt}")

    def forward(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """

        Compute next state: x[k+1] = discrete_dynamics(x, u[k])

        **CRITICAL DISTINCTION**: This method returns the NEXT STATE x[k+1], NOT
        the derivative dx/dt.

        Args:
            x: Current state
            u: Control input

        Returns:
            Next state after one time step (same shape as input)
        """
        if self.order == 1:
            return self._integrate_first_order(x, u)
        elif self.order == 2:
            return self._integrate_second_order(x, u)
        else:
            return self._integrate_arbitrary_order(x, u)

    def __call__(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Make the system callable like a function"""
        return self.forward(x, u)

    def simulate(
        self,
        x0: torch.Tensor,
        controller: Optional[Union[torch.Tensor, Callable, torch.nn.Module]] = None,
        horizon: Optional[int] = None,
        return_controls: bool = False,
        return_all: bool = True,
        observer: Optional[Callable] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Simulate trajectory from initial state.

        Args:
            x0: Initial state (nx,) or (batch, nx)
            controller:
                - torch.Tensor: Control sequence (T, nu) - horizon inferred from T
                - Callable/nn.Module: Controller π(x) - horizon MUST be specified
                - None: Zero control - horizon MUST be specified
            horizon: Number of timesteps (required for controller functions)
            return_controls: If True, return (trajectory, controls)
            return_all: If True, return all states; if False, only final state
            observer: Optional observer x̂ = obs(x) for output feedback

        Returns:
            If return_controls=False:
                Trajectory: (T+1, nx) or (batch, T+1, nx) if return_all=True
                        (nx,) or (batch, nx) if return_all=False
            If return_controls=True:
                (Trajectory, Controls): Both same batch format

        Examples:
            >>> # 1. Pre-computed control sequence
            >>> u_seq = torch.zeros(100, 1)
            >>> traj = system.simulate(x0, controller=u_seq)

            >>> # 2. Neural network controller
            >>> controller_nn = NeuralNetworkController(...)
            >>> traj = system.simulate(x0, controller=controller_nn, horizon=100)

            >>> # 3. Lambda function controller
            >>> lqr_controller = lambda x: K @ (x - x_eq).T
            >>> traj = system.simulate(x0, controller=lqr_controller, horizon=100)

            >>> # 4. Output feedback with observer
            >>> traj = system.simulate(x0, controller=controller_nn,
            ...                        observer=observer_nn, horizon=100)

            >>> # 5. Return both trajectory and controls
            >>> traj, controls = system.simulate(x0, controller=controller_nn,
            ...                                  horizon=100, return_controls=True)
        """

        # Determine type
        is_control_sequence = isinstance(controller, torch.Tensor)

        if is_control_sequence:
            # Can infer T from sequence
            u_sequence = controller
            if len(u_sequence.shape) == 2:
                u_sequence = u_sequence.unsqueeze(0)
            T = u_sequence.shape[1]

            if horizon is not None and horizon != T:
                warnings.warn(
                    f"horizon={horizon} specified but control sequence has length {T}. "
                    f"Using sequence length T={T}."
                )

        else:
            # Controller function or None - MUST have horizon
            if horizon is None:
                raise ValueError(
                    "horizon must be specified when controller is a function or None.\n"
                    "Usage:\n"
                    "  - Control sequence: simulate(x0, controller=u_seq)  # horizon inferred\n"
                    "  - Controller func:  simulate(x0, controller=π, horizon=100)\n"
                    "  - Zero control:     simulate(x0, controller=None, horizon=100)"
                )
            T = horizon

            if controller is None:
                controller_func = lambda x: torch.zeros(
                    x.shape[0], self.nu, device=x.device
                )
            else:
                controller_func = controller

        # Handle dimensionality
        if len(x0.shape) == 1:
            x0 = x0.unsqueeze(0)
            squeeze_batch = True
        else:
            squeeze_batch = False

        batch_size = x0.shape[0]

        # Determine if controller is a sequence or a function
        is_control_sequence = isinstance(controller, torch.Tensor)
        is_controller_function = callable(controller) or isinstance(
            controller, torch.nn.Module
        )

        if controller is None:
            # Zero control
            if horizon is None:
                raise ValueError("horizon must be specified when controller is None")
            T = horizon
            controller_func = lambda x: torch.zeros(
                x.shape[0], self.nu, device=x.device, dtype=x.dtype
            )
            is_controller_function = True
        elif is_control_sequence:
            # Pre-computed control sequence
            u_sequence = controller

            # Handle batch dimensions
            if len(u_sequence.shape) == 2:
                u_sequence = u_sequence.unsqueeze(0)

            # Expand to match batch size if needed
            if u_sequence.shape[0] == 1 and batch_size > 1:
                u_sequence = u_sequence.expand(batch_size, -1, -1)

            if u_sequence.shape[0] != batch_size:
                raise ValueError(
                    f"Control sequence batch size {u_sequence.shape[0]} "
                    f"doesn't match state batch size {batch_size}"
                )

            T = u_sequence.shape[1]

        elif is_controller_function:
            # Controller is a function or neural network
            if horizon is None:
                raise ValueError(
                    "horizon must be specified when controller is a callable"
                )
            T = horizon
            controller_func = controller
        else:
            raise TypeError(
                f"controller must be torch.Tensor, callable, nn.Module, or None. "
                f"Got {type(controller)}"
            )

        # Initialize storage
        if return_all:
            trajectory = [x0]
        if return_controls:
            controls = []

        x = x0

        # Simulation loop
        for t in range(T):
            if is_control_sequence:
                # Use pre-computed control
                u = u_sequence[:, t, :]
            else:
                # Compute control from current state
                if observer is not None:
                    # Output feedback: u = π(x̂) where x̂ = obs(x)
                    with torch.no_grad():
                        x_hat = observer(x)
                    u = controller_func(x_hat)
                else:
                    # State feedback: u = π(x)
                    u = controller_func(x)

                # Ensure proper shape
                if len(u.shape) == 1:
                    u = u.unsqueeze(0)
                elif len(u.shape) == 3:
                    u = u.squeeze(1)

            # Store control if requested
            if return_controls:
                controls.append(u)

            # Step forward
            x = self.forward(x, u)

            if return_all:
                trajectory.append(x)

        # Format outputs
        if return_all:
            result = torch.stack(trajectory, dim=1)  # (batch, T+1, nx)
            if squeeze_batch:
                result = result.squeeze(0)
        else:
            result = x.squeeze(0) if squeeze_batch else x

        if return_controls:
            controls_tensor = torch.stack(controls, dim=1)  # (batch, T, nu)
            if squeeze_batch:
                controls_tensor = controls_tensor.squeeze(0)
            return result, controls_tensor
        else:
            return result

    def _integrate_first_order(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Integrate first-order system: dx/dt = f(x, u)"""

        # Handle 1D vs 2D input
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
            u = u.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        xdot = self.continuous_time_system.forward(x, u)

        if self.integration_method == IntegrationMethod.ExplicitEuler:
            x_next = x + xdot * self.dt
        elif self.integration_method == IntegrationMethod.MidPoint:
            k1 = xdot
            x_mid = x + 0.5 * self.dt * k1
            k2 = self.continuous_time_system.forward(x_mid, u)
            x_next = x + self.dt * k2
        elif self.integration_method == IntegrationMethod.RK4:
            k1 = xdot
            k2 = self.continuous_time_system.forward(x + 0.5 * self.dt * k1, u)
            k3 = self.continuous_time_system.forward(x + 0.5 * self.dt * k2, u)
            k4 = self.continuous_time_system.forward(x + self.dt * k3, u)
            x_next = x + (self.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        else:
            raise NotImplementedError(
                f"Integration method {self.integration_method} not implemented"
            )

        return x_next

    def _integrate_second_order(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Integrate second-order system: x = [q, qdot], qddot = f(x, u)
        """

        # Handle 1D vs 2D input
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
            u = u.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        nq = self.continuous_time_system.nq
        q = x[:, :nq]
        qdot = x[:, nq:]

        # Compute acceleration at current state
        qddot = self.continuous_time_system.forward(x, u)

        # Validate acceleration shape
        if qddot.shape[1] != nq:
            if qddot.shape[1] == 1 and nq == 1:
                pass  # Correct
            else:
                raise ValueError(f"Expected qddot shape (*, {nq}), got {qddot.shape}")

        if self.integration_method == IntegrationMethod.ExplicitEuler:
            qdot_next = qdot + qddot * self.dt

        elif self.integration_method == IntegrationMethod.MidPoint:
            qdot_mid = qdot + 0.5 * self.dt * qddot
            x_mid = torch.cat([q, qdot_mid], dim=1)
            qddot_mid = self.continuous_time_system.forward(x_mid, u)
            qdot_next = qdot + self.dt * qddot_mid

        elif self.integration_method == IntegrationMethod.RK4:
            k1_vel = qddot
            qdot_stage2 = qdot + 0.5 * self.dt * k1_vel
            x_stage2 = torch.cat([q, qdot_stage2], dim=1)
            k2_vel = self.continuous_time_system.forward(x_stage2, u)
            qdot_stage3 = qdot + 0.5 * self.dt * k2_vel
            x_stage3 = torch.cat([q, qdot_stage3], dim=1)
            k3_vel = self.continuous_time_system.forward(x_stage3, u)
            qdot_stage4 = qdot + self.dt * k3_vel
            x_stage4 = torch.cat([q, qdot_stage4], dim=1)
            k4_vel = self.continuous_time_system.forward(x_stage4, u)

            # Combine stages
            qdot_next = qdot + (self.dt / 6.0) * (
                k1_vel + 2 * k2_vel + 2 * k3_vel + k4_vel
            )

        else:
            raise NotImplementedError(
                f"Integration method {self.integration_method} not implemented for 2nd order"
            )

        if self.position_integration == IntegrationMethod.ExplicitEuler:
            q_next = q + qdot * self.dt

        elif self.position_integration == IntegrationMethod.MidPoint:
            q_next = q + (qdot_next + qdot) / 2 * self.dt

        elif self.position_integration == IntegrationMethod.RK4:

            if self.integration_method == IntegrationMethod.RK4:
                k1_pos = qdot
                k2_pos = qdot_stage2
                k3_pos = qdot_stage3
                k4_pos = qdot_next  # Final velocity

                q_next = q + (self.dt / 6.0) * (
                    k1_pos + 2 * k2_pos + 2 * k3_pos + k4_pos
                )
            else:
                q_next = q + (qdot_next + qdot) / 2 * self.dt

        else:
            raise NotImplementedError(
                f"Position integration {self.position_integration} not implemented"
            )

        result = torch.cat([q_next, qdot_next], dim=1)

        # Squeeze back to 1D if input was 1D
        if squeeze_output:
            result = result.squeeze(0)

        return result

    def _integrate_arbitrary_order(
        self, x: torch.Tensor, u: torch.Tensor
    ) -> torch.Tensor:
        """
        Integrate arbitrary order system: x = [q, q', ..., q^(n-1)], q^(n) = f(x, u)
        """

        # Handle 1D vs 2D input
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
            u = u.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        order = self.order
        nq = self.nx // order

        # Split state into derivative levels
        derivatives = [x[:, i * nq : (i + 1) * nq] for i in range(order)]
        highest_deriv = self.continuous_time_system.forward(x, u)

        derivatives_next = []

        if self.integration_method == IntegrationMethod.ExplicitEuler:
            for i in range(order - 1):
                derivatives_next.append(derivatives[i] + derivatives[i + 1] * self.dt)
            derivatives_next.append(derivatives[-1] + highest_deriv * self.dt)

        elif self.integration_method == IntegrationMethod.MidPoint:
            for i in range(order - 1):
                derivatives_next.append(derivatives[i] + self.dt * derivatives[i + 1])

            q_highest_mid = derivatives[-1] + 0.5 * self.dt * highest_deriv
            x_mid = torch.cat(derivatives[:-1] + [q_highest_mid], dim=1)
            highest_deriv_mid = self.continuous_time_system.forward(x_mid, u)
            derivatives_next.append(derivatives[-1] + self.dt * highest_deriv_mid)

        elif self.integration_method == IntegrationMethod.RK4:
            # Stage 1
            k1_derivs = derivatives[1:] + [highest_deriv]

            # Stage 2
            x_stage2 = [
                derivatives[i] + 0.5 * self.dt * k1_derivs[i] for i in range(order)
            ]
            x_mid_2 = torch.cat(x_stage2, dim=1)
            highest_deriv_2 = self.continuous_time_system.forward(x_mid_2, u)
            k2_derivs = x_stage2[1:] + [highest_deriv_2]

            # Stage 3
            x_stage3 = [
                derivatives[i] + 0.5 * self.dt * k2_derivs[i] for i in range(order)
            ]
            x_mid_3 = torch.cat(x_stage3, dim=1)
            highest_deriv_3 = self.continuous_time_system.forward(x_mid_3, u)
            k3_derivs = x_stage3[1:] + [highest_deriv_3]

            # Stage 4
            x_stage4 = [derivatives[i] + self.dt * k3_derivs[i] for i in range(order)]
            x_end = torch.cat(x_stage4, dim=1)
            highest_deriv_4 = self.continuous_time_system.forward(x_end, u)
            k4_derivs = x_stage4[1:] + [highest_deriv_4]

            # Combine
            for i in range(order):
                weighted = (
                    k1_derivs[i] + 2 * k2_derivs[i] + 2 * k3_derivs[i] + k4_derivs[i]
                ) / 6.0
                derivatives_next.append(derivatives[i] + self.dt * weighted)

        else:
            raise NotImplementedError(
                f"Integration method {self.integration_method} not implemented for order {order}"
            )

        result = torch.cat(derivatives_next, dim=1)

        # Squeeze back to 1D if input was 1D
        if squeeze_output:
            result = result.squeeze(0)

        return result

    def linearized_dynamics(
        self, x: torch.Tensor, u: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute linearized discrete dynamics using Euler approximation

        Returns:
            (Ad, Bd): Discrete-time linearized dynamics
        """
        Ac, Bc = self.continuous_time_system.linearized_dynamics(x, u)
        Ad = self.dt * Ac + self.Ix.to(x.device)
        Bd = self.dt * Bc
        return Ad, Bd

    def linearized_observation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute linearized observation matrix C = dh/dx

        For discrete-time systems, the observation is the same as continuous-time
        since h(x) doesn't depend on the discretization. The observation is with respect to
        state x and not time t.

        Args:
            x: State tensor (batch, nx) or (nx,)

        Returns:
            C: Observation Jacobian (batch, ny, nx) or (ny, nx)
        """
        return self.continuous_time_system.linearized_observation(x)

    def h(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate output equation: y = h(x)

        For discrete-time systems, the observation is the same as continuous-time
        since h(x) doesn't depend on the discretization. The observation is with respect to
        state x and not time t.

        Args:
            x: State tensor

        Returns:
            Output tensor
        """
        return self.continuous_time_system.h(x)

    @property
    def x_equilibrium(self) -> torch.Tensor:
        return self.continuous_time_system.x_equilibrium

    @property
    def u_equilibrium(self) -> torch.Tensor:
        return self.continuous_time_system.u_equilibrium

    def __repr__(self) -> str:
        return (
            f"GenericDiscreteTimeSystem({self.continuous_time_system.__class__.__name__}, "
            f"dt={self.dt}, method={self.integration_method.name})"
        )

    def __str__(self) -> str:
        """Human-readable string representation"""
        return (
            f"Discrete {self.continuous_time_system.__class__.__name__} "
            f"(dt={self.dt:.4f}, {self.integration_method.name})"
        )

    def dlqr_control(
        self,
        Q: np.ndarray,
        R: np.ndarray,
        x_eq: Optional[torch.Tensor] = None,
        u_eq: Optional[torch.Tensor] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute discrete-time LQR control gain.

        **IMPORTANT**: This method linearizes the nonlinear discrete-time system
        around the equilibrium point and computes the optimal gain for the linearized
        system. The resulting controller is:
        - Globally optimal for linear discrete-time systems
        - Locally optimal near equilibrium for nonlinear systems
        - Performance degrades as state moves away from equilibrium

        Theory:
        ------
        Solves the discrete-time algebraic Riccati equation (DARE):
            S = Q + A^T S A - A^T S B (R + B^T S B)^{-1} B^T S A

        The optimal gain is:
            K = -(R + B^T S B)^{-1} B^T S A

        Control law:
            u[k] = K @ (x[k] - x_eq) + u_eq

        Cost function minimized (for linearized system):
            J = Σ[k=0,∞] [(x[k]-x_eq)^T Q (x[k]-x_eq) + (u[k]-u_eq)^T R (u[k]-u_eq)]

        Args:
            Q: State cost matrix (nx, nx). Must be positive semi-definite.
            R: Control cost matrix (nu, nu) or scalar. Must be positive definite.
            x_eq: Equilibrium state (uses self.x_equilibrium if None)
            u_eq: Equilibrium control (uses self.u_equilibrium if None)

        Returns:
            K: Discrete control gain matrix (nu, nx)
            S: Solution to discrete-time Riccati equation (nx, nx)

        Raises:
            ValueError: If matrix dimensions are incompatible
            LinAlgError: If DARE has no stabilizing solution

        Example:
            >>> # Create discrete-time system
            >>> pendulum_ct = SymbolicPendulum(m=0.15, l=0.5, beta=0.1, g=9.81)
            >>> pendulum_dt = GenericDiscreteTimeSystem(pendulum_ct, dt=0.01)
            >>>
            >>> # Design discrete LQR
            >>> Q = np.diag([10.0, 1.0])
            >>> R = np.array([[0.1]])
            >>> K, S = pendulum_dt.dlqr_control(Q, R)
            >>>
            >>> # Simulate
            >>> controller = lambda x: K @ (x - pendulum_dt.x_equilibrium) + pendulum_dt.u_equilibrium
            >>> trajectory = pendulum_dt.simulate(x0, controller, horizon=1000)

        Notes:
            - Linearization uses the discretization method specified in __init__
            - For second-order systems, the full discrete state-space form is used
            - Closed-loop eigenvalues should satisfy |λ| < 1 for stability
            - Discrete LQR often performs better than discretized continuous LQR

        See Also:
            lqr_control: Continuous-time version
            discrete_kalman_gain: Discrete observer design
            dlqg_control: Combined controller and observer
        """
        if x_eq is None:
            x_eq = self.x_equilibrium
        if u_eq is None:
            u_eq = self.u_equilibrium

        # Ensure proper shape
        if len(x_eq.shape) == 1:
            x_eq = x_eq.unsqueeze(0)
        if len(u_eq.shape) == 1:
            u_eq = u_eq.unsqueeze(0)

        # Get discrete linearized dynamics at equilibrium
        Ad, Bd = self.linearized_dynamics(x_eq, u_eq)
        Ad = Ad.squeeze().detach().cpu().numpy()
        Bd = Bd.squeeze().detach().cpu().numpy()

        # Ensure Bd is 2D
        if Bd.ndim == 1:
            Bd = Bd.reshape(-1, 1)

        # Ensure R is 2D
        if isinstance(R, (int, float)):
            R = np.array([[R]])
        elif R.ndim == 1:
            R = np.diag(R)

        # Solve discrete-time algebraic Riccati equation
        S = scipy.linalg.solve_discrete_are(Ad, Bd, Q, R)

        # Compute optimal gain
        K = -np.linalg.solve(R + Bd.T @ S @ Bd, Bd.T @ S @ Ad)

        return K, S

    def discrete_kalman_gain(
        self,
        Q_process: Optional[np.ndarray] = None,
        R_measurement: Optional[np.ndarray] = None,
        x_eq: Optional[torch.Tensor] = None,
    ) -> np.ndarray:
        """
        Compute discrete-time Kalman filter gain.

        **IMPORTANT**: This method linearizes the nonlinear discrete-time system
        and computes the optimal observer gain for the linearized system. For
        nonlinear systems, this provides local optimality near equilibrium.

        Theory:
        ------
        Solves the discrete-time dual Riccati equation:
            P = Q + A P A^T - A P C^T (C P C^T + R)^{-1} C P A^T

        The optimal gain is:
            L = P C^T (C P C^T + R)^{-1}

        Observer update:
            x̂[k|k-1] = f(x̂[k-1|k-1], u[k-1])    [Predict]
            x̂[k|k] = x̂[k|k-1] + L(y[k] - h(x̂[k|k-1]))    [Update]

        For linearized system:
            x̂[k+1|k] = A x̂[k|k] + B u[k]
            x̂[k|k] = x̂[k|k-1] + L(y[k] - C x̂[k|k-1])

        Args:
            Q_process: Process noise covariance (nx, nx). Default: 0.001 * I
            R_measurement: Measurement noise covariance (ny, ny) or scalar. Default: 0.001 * I
            x_eq: Equilibrium state for linearization (uses self.x_equilibrium if None)

        Returns:
            L: Kalman gain matrix (nx, ny)

        Example:
            >>> # Create discrete system
            >>> pendulum_ct = SymbolicPendulum(m=0.15, l=0.5, beta=0.1, g=9.81)
            >>> pendulum_dt = GenericDiscreteTimeSystem(pendulum_ct, dt=0.01)
            >>>
            >>> # Design Kalman filter
            >>> Q_process = np.diag([0.001, 0.01])
            >>> R_measurement = np.array([[0.1]])
            >>> L = pendulum_dt.discrete_kalman_gain(Q_process, R_measurement)
            >>>
            >>> # Use in simulation
            >>> observer = LinearObserver(pendulum_dt, L)
            >>> for k in range(steps):
            >>>     observer.update(u[k], y_measured[k], dt=pendulum_dt.dt)
            >>>     x_estimate = observer.x_hat

        Notes:
            - This is the steady-state Kalman gain (infinite horizon)
            - For time-varying gain, implement a Kalman filter loop
            - The gain balances prediction uncertainty and measurement noise
            - Linearization matches the integration method used for dynamics

        See Also:
            dlqr_control: Discrete controller design
            dlqg_control: Combined controller and observer
            ExtendedKalmanFilter: Nonlinear filtering
        """
        if Q_process is None:
            Q_process = np.eye(self.nx) * 1e-3
        if R_measurement is None:
            R_measurement = np.eye(self.continuous_time_system.ny) * 1e-3
        if x_eq is None:
            x_eq = self.x_equilibrium

        # Ensure proper shape
        if len(x_eq.shape) == 1:
            x_eq = x_eq.unsqueeze(0)

        # Get discrete linearized dynamics
        u_eq = self.u_equilibrium
        if len(u_eq.shape) == 1:
            u_eq = u_eq.unsqueeze(0)

        Ad, _ = self.linearized_dynamics(x_eq, u_eq)
        Ad = Ad.squeeze().detach().cpu().numpy()

        C = self.continuous_time_system.linearized_observation(x_eq)
        C = C.squeeze().detach().cpu().numpy()

        # Ensure C is 2D
        if C.ndim == 1:
            C = C.reshape(1, -1)

        # Ensure R_measurement is 2D
        if isinstance(R_measurement, (int, float)):
            R_measurement = np.array([[R_measurement]])
        elif R_measurement.ndim == 1:
            R_measurement = np.diag(R_measurement)

        # Solve discrete-time algebraic Riccati equation (dual problem)
        P = scipy.linalg.solve_discrete_are(Ad.T, C.T, Q_process, R_measurement)

        # Compute Kalman gain
        L = P @ C.T @ np.linalg.inv(C @ P @ C.T + R_measurement)

        return L

    def dlqg_control(
        self,
        Q_lqr: np.ndarray,
        R_lqr: np.ndarray,
        Q_process: Optional[np.ndarray] = None,
        R_measurement: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute discrete-time LQG controller (LQR + Kalman filter).

        **IMPORTANT**: Designs output feedback control for the linearized discrete-time
        system. The separation principle applies to linear systems but not globally
        to nonlinear systems.

        Theory:
        ------
        Discrete-time LQG control:
            x̂[k|k-1] = f(x̂[k-1|k-1], u[k-1])           [Predict]
            x̂[k|k] = x̂[k|k-1] + L(y[k] - h(x̂[k|k-1]))  [Correct]
            u[k] = K @ (x̂[k|k] - x_eq) + u_eq          [Control]

        Closed-loop eigenvalues (for linear system):
            eig(A_cl) = {eig(A + BK)} ∪ {eig(A - LC)}

        Args:
            Q_lqr: State cost for LQR (nx, nx)
            R_lqr: Control cost for LQR (nu, nu) or scalar
            Q_process: Process noise covariance (nx, nx). Default: 0.001 * I
            R_measurement: Measurement noise covariance (ny, ny) or scalar. Default: 0.001 * I

        Returns:
            K: Discrete LQR control gain (nu, nx)
            L: Discrete Kalman gain (nx, ny)

        Example:
            >>> # Create discrete system
            >>> quad_ct = SymbolicQuadrotor2D()
            >>> quad_dt = GenericDiscreteTimeSystem(quad_ct, dt=0.01,
            ...                                     integration_method=IntegrationMethod.RK4)
            >>>
            >>> # Design LQG controller
            >>> Q_lqr = np.diag([10, 10, 5, 1, 1, 1])  # Position > velocity
            >>> R_lqr = np.eye(2) * 0.1
            >>> Q_process = np.eye(6) * 0.01
            >>> R_measurement = np.eye(3) * 0.1  # Measure [x, y, theta]
            >>>
            >>> K, L = quad_dt.dlqg_control(Q_lqr, R_lqr, Q_process, R_measurement)
            >>>
            >>> # Simulate with observer feedback
            >>> controller = LinearController(K, quad_dt.x_equilibrium, quad_dt.u_equilibrium)
            >>> observer = LinearObserver(quad_dt, L)
            >>>
            >>> x = x0
            >>> for k in range(1000):
            >>>     y = quad_dt.h(x) + measurement_noise()
            >>>     observer.update(u, y, dt=quad_dt.dt)
            >>>     u = controller(observer.x_hat)
            >>>     x = quad_dt(x, u)

        Notes:
            - Separation principle: design K and L independently (for linear systems)
            - Discrete-time implementation more natural for digital control
            - Observer and controller run at the same rate (dt)
            - For different rates, use multirate control techniques

        See Also:
            dlqr_control: Controller only
            discrete_kalman_gain: Observer only
            dlqg_closed_loop_matrix: Closed-loop analysis
        """
        K, _ = self.dlqr_control(Q_lqr, R_lqr)
        L = self.discrete_kalman_gain(Q_process, R_measurement)
        return K, L

    def dlqg_closed_loop_matrix(self, K: np.ndarray, L: np.ndarray) -> np.ndarray:
        """
        Compute closed-loop discrete system matrix for LQG control.

        Returns the linearized dynamics of the augmented discrete-time system [x, x̂].

        Theory:
        ------
        Closed-loop discrete dynamics:
            [x[k+1]]  = [A + BK       -BK    ] [x[k]]
            [x̂[k+1]]  = [LC           A+BK-LC] [x̂[k]]

        Eigenvalues (for linear systems):
            eig(A_cl) = {eig(A + BK)} ∪ {eig(A - LC)}

        Stability condition:
            All eigenvalues must satisfy |λ| < 1

        Args:
            K: Discrete LQR gain (nu, nx) from dlqr_control()
            L: Discrete Kalman gain (nx, ny) from discrete_kalman_gain()

        Returns:
            A_cl: Closed-loop discrete system matrix (2*nx, 2*nx)

        Example:
            >>> # Design and analyze discrete LQG
            >>> K, L = system.dlqg_control(Q_lqr, R_lqr, Q_process, R_measurement)
            >>> A_cl = system.dlqg_closed_loop_matrix(K, L)
            >>>
            >>> # Check discrete stability
            >>> eigenvalues = np.linalg.eigvals(A_cl)
            >>> is_stable = np.all(np.abs(eigenvalues) < 1.0)
            >>> print(f"Closed-loop stable: {is_stable}")
            >>> print(f"Spectral radius: {np.max(np.abs(eigenvalues)):.4f}")
            >>>
            >>> # Visualize eigenvalues on unit circle
            >>> import matplotlib.pyplot as plt
            >>> theta = np.linspace(0, 2*np.pi, 100)
            >>> plt.plot(np.cos(theta), np.sin(theta), 'k--', label='Unit circle')
            >>> plt.plot(eigenvalues.real, eigenvalues.imag, 'rx', label='Poles')
            >>> plt.axis('equal')
            >>> plt.legend()
            >>> plt.show()

        Notes:
            - Eigenvalues inside unit circle → stable
            - Eigenvalues on unit circle → marginally stable
            - Eigenvalues outside unit circle → unstable
            - Small entries (< 1e-6) are zeroed for cleanliness

        See Also:
            dlqg_control: Design the gains
            output_feedback_lyapunov: Lyapunov stability analysis
        """
        x_eq = self.x_equilibrium.unsqueeze(0)
        u_eq = self.u_equilibrium.unsqueeze(0)

        Ad, Bd = self.linearized_dynamics(x_eq, u_eq)
        Ad = Ad.squeeze().detach().cpu().numpy()
        Bd = Bd.squeeze().detach().cpu().numpy()

        # Ensure Bd is 2D (nx, nu)
        if Bd.ndim == 1:
            Bd = Bd.reshape(-1, 1)

        C = self.continuous_time_system.linearized_observation(x_eq)
        C = C.squeeze().detach().cpu().numpy()

        # Ensure C is 2D (ny, nx)
        if C.ndim == 1:
            C = C.reshape(1, -1)

        # Ensure K is 2D (nu, nx)
        if K.ndim == 1:
            K = K.reshape(1, -1)

        # Ensure L is 2D (nx, ny)
        if L.ndim == 1:
            L = L.reshape(-1, 1)

        # Closed-loop discrete system: [x[k], x̂[k]]
        # x[k+1] = Ad @ x[k] + Bd @ K @ x̂[k]
        # x̂[k+1] = Ad @ x̂[k] + Bd @ K @ x̂[k] + L @ (C @ x[k] - C @ x̂[k])
        #         = (Ad + Bd @ K - L @ C) @ x̂[k] + L @ C @ x[k]
        A_cl = np.vstack(
            [
                np.hstack([Ad + Bd @ K, -Bd @ K]),  # x[k+1]
                np.hstack([L @ C, Ad + Bd @ K - L @ C]),  # x̂[k+1]
            ]
        )

        # Clean up near-zero entries
        A_cl[np.abs(A_cl) <= 1e-6] = 0

        return A_cl

    def output_feedback_lyapunov(self, K: np.ndarray, L: np.ndarray) -> np.ndarray:
        """
        Solve discrete-time Lyapunov equation for output feedback system.

        **IMPORTANT**: This solves the Lyapunov equation for the LINEARIZED
        closed-loop system around equilibrium. The resulting Lyapunov function
        V(z) = z^T S z proves:
        - Global asymptotic stability for LINEAR systems
        - LOCAL asymptotic stability for NONLINEAR systems (near equilibrium only)

        Finds positive definite matrix S satisfying:
            A_cl^T S A_cl - S + I = 0

        where A_cl is the LINEARIZED closed-loop system matrix.

        Args:
            K: Control gain (nu, nx)
            L: Observer gain (nx, ny)

        Returns:
            S: Solution to discrete Lyapunov equation (2*nx, 2*nx)
            Positive definite if linearized closed-loop is stable

        Example:
            >>> K, L = system.dlqg_control(Q_lqr, R_lqr, Q_process, R_measurement)
            >>> S = system.output_feedback_lyapunov(K, L)
            >>>
            >>> # Verify LOCAL stability
            >>> eigenvalues = np.linalg.eigvals(S)
            >>> is_positive_definite = np.all(eigenvalues > 0)
            >>> print(f"Linearized system locally stable: {is_positive_definite}")
            >>>
            >>> # WARNING: This doesn't estimate region of attraction!
            >>> # For nonlinear system, stability only guaranteed near equilibrium
            >>> def lyapunov_function(z):
            >>>     '''V(z) proves local stability, not global'''
            >>>     return z.T @ S @ z
            >>>
            >>> # To estimate region: sample many initial conditions
            >>> # and check which ones converge (Monte Carlo approach)

        Notes:
            - Positive definite S → linearized closed-loop is asymptotically stable
            - For nonlinear systems: only proves local stability
            - The linearization is computed at (x_eq, u_eq)
            - Does NOT provide region of attraction estimate

        See Also:
            dlqg_closed_loop_matrix: Get the linearized closed-loop matrix A_cl
            dlqg_control: Design the gains
        """

        A_cl = self.dlqg_closed_loop_matrix(K, L)
        S = control.dlyap(A_cl, np.eye(2 * self.nx))

        return S

    def print_info(
        self, include_equations: bool = True, include_linearization: bool = True
    ):
        """
        Print comprehensive information about the discrete-time system

        Args:
            include_equations: Whether to print symbolic equations
            include_linearization: Whether to print linearization at equilibrium
        """
        print("=" * 70)
        print(f"Discrete-Time System: {self.continuous_time_system.__class__.__name__}")
        print("=" * 70)

        # Basic info
        print(f"\nDiscretization:")
        print(f"  Time step (dt):        {self.dt}")
        print(f"  Integration method:    {self.integration_method.name}")
        if self.order > 1:
            print(f"  Position integration:  {self.position_integration.name}")

        print(f"\nDimensions:")
        print(f"  State dimension (nx):    {self.nx}")
        print(f"  Control dimension (nu):  {self.nu}")
        print(f"  Output dimension (ny):   {self.continuous_time_system.ny}")
        print(f"  System order:            {self.order}")
        if self.order > 1:
            print(f"  Generalized coords (nq): {self.continuous_time_system.nq}")

        print(f"\nEquilibrium:")
        x_eq = self.x_equilibrium.detach().cpu().numpy()
        u_eq = self.u_equilibrium.detach().cpu().numpy()
        print(f"  x_eq = {x_eq}")
        print(f"  u_eq = {u_eq}")

        # Symbolic equations from continuous system
        if include_equations:
            print("\n" + "-" * 70)
            print("Continuous-Time Dynamics (before discretization):")
            print("-" * 70)
            self.continuous_time_system.print_equations(simplify=True)

        # Linearization at equilibrium
        if include_linearization:
            print("\n" + "-" * 70)
            print("Linearization at Equilibrium:")
            print("-" * 70)

            # Continuous-time linearization
            Ac, Bc = self.continuous_time_system.linearized_dynamics(
                self.x_equilibrium.unsqueeze(0), self.u_equilibrium.unsqueeze(0)
            )
            Ac_np = Ac.squeeze().detach().cpu().numpy()
            Bc_np = Bc.squeeze().detach().cpu().numpy()

            print("Continuous-time (Ac, Bc):")
            print(f"  Ac =\n{Ac_np}")
            print(f"  Bc =\n{Bc_np}")

            # Discrete-time linearization
            Ad, Bd = self.linearized_dynamics(
                self.x_equilibrium.unsqueeze(0), self.u_equilibrium.unsqueeze(0)
            )
            Ad_np = Ad.squeeze().detach().cpu().numpy()
            Bd_np = Bd.squeeze().detach().cpu().numpy()

            print(f"\nDiscrete-time (Ad, Bd) with dt={self.dt}:")
            print(f"  Ad =\n{Ad_np}")
            print(f"  Bd =\n{Bd_np}")

            # Eigenvalues
            eigs_c = np.linalg.eigvals(Ac_np)
            eigs_d = np.linalg.eigvals(Ad_np)

            print(f"\nEigenvalues:")
            print(f"  Continuous: {eigs_c}")
            print(f"  Discrete:   {eigs_d}")
            print(f"\nStability:")
            print(f"  Continuous stable? {np.all(np.real(eigs_c) < 0)}")
            print(f"  Discrete stable?   {np.all(np.abs(eigs_d) < 1)}")

            # Observation matrix
            C = self.linearized_observation(self.x_equilibrium.unsqueeze(0))
            C_np = C.squeeze().detach().cpu().numpy()
            print(f"\nObservation matrix C:")
            print(f"  C =\n{C_np}")

        print("=" * 70)

    def summary(self) -> str:
        """
        Get a brief summary string

        Returns:
            Summary string with key system info
        """
        ct_stable = self.continuous_time_system.is_stable_equilibrium(
            discrete_time=False
        )

        # Check discrete stability
        eigs_d = np.linalg.eigvals(
            self.linearized_dynamics(
                self.x_equilibrium.unsqueeze(0), self.u_equilibrium.unsqueeze(0)
            )[0]
            .squeeze()
            .detach()
            .cpu()
            .numpy()
        )
        dt_stable = bool(np.all(np.abs(eigs_d) < 1))

        summary_str = (
            f"{self.continuous_time_system.__class__.__name__} "
            f"(nx={self.nx}, nu={self.nu}, ny={self.continuous_time_system.ny}, "
            f"order={self.order}, dt={self.dt:.4f}, {self.integration_method.name})\n"
            f"  Continuous stable: {ct_stable}, Discrete stable: {dt_stable}"
        )
        return summary_str

    def plot_trajectory(
        self,
        trajectory: torch.Tensor,
        state_names: Optional[List[str]] = None,
        control_sequence: Optional[torch.Tensor] = None,
        control_names: Optional[List[str]] = None,
        title: Optional[str] = None,
        trajectory_names: Optional[List[str]] = None,
        colorway: Union[str, List[str]] = "Plotly",
        compact: bool = False,
        aspect_ratio: float = 1.5,
        max_height: Optional[int] = None,
        max_width: Optional[int] = None,
        save_html: Optional[str] = None,
        show: bool = True,
    ):
        """
        Plot trajectory using Plotly (interactive visualization) with adaptive sizing

        Args:
            trajectory: State trajectory (T, nx) or (batch, T, nx)
            state_names: Names for state variables (uses x0, x1, ... if None)
            control_sequence: Optional control inputs (T, nu) or (batch, T, nu)
            control_names: Names for control variables (uses u0, u1, ... if None)
            title: Plot title
            trajectory_names: Optional names for each trajectory (uses "Trajectory 1", etc. if None)
            colorway: Plotly color sequence name or list of colors. Options:
                    'Plotly' (default), 'D3', 'G10', 'T10', 'Alphabet',
                    'Dark24', 'Light24', 'Set1', 'Pastel', 'Vivid',
                    or a custom list of color strings
            compact: If True, use smaller subplots for many variables (default: False)
            aspect_ratio: Target width:height ratio per subplot (default: 1.5)
            max_height: Maximum figure height in pixels (default: None = auto)
            max_width: Maximum figure width in pixels (default: None = auto)
            save_html: If provided, save interactive plot to this HTML file
            show: If True, display the plot
        """

        # Handle batched trajectories
        if len(trajectory.shape) == 3:
            batch_size = trajectory.shape[0]
            print(f"Plotting {batch_size} trajectories...")
        else:
            trajectory = trajectory.unsqueeze(0)
            batch_size = 1
            if control_sequence is not None:
                control_sequence = control_sequence.unsqueeze(0)

        # Convert to numpy
        traj_np = trajectory.detach().cpu().numpy()

        # Get color sequence
        if isinstance(colorway, str):
            # Map string names to Plotly color sequences
            color_sequences = {
                "Plotly": px.colors.qualitative.Plotly,
                "D3": px.colors.qualitative.D3,
                "G10": px.colors.qualitative.G10,
                "T10": px.colors.qualitative.T10,
                "Alphabet": px.colors.qualitative.Alphabet,
                "Dark24": px.colors.qualitative.Dark24,
                "Light24": px.colors.qualitative.Light24,
                "Set1": px.colors.qualitative.Set1,
                "Pastel": px.colors.qualitative.Pastel,
                "Vivid": px.colors.qualitative.Vivid,
            }
            colors = color_sequences.get(colorway, px.colors.qualitative.Plotly)
        else:
            colors = colorway

        # Determine subplot layout
        has_control = control_sequence is not None
        num_plots = self.nx + (self.nu if has_control else 0)

        # Adaptive subplot layout calculation
        def calculate_subplot_layout(n):
            """Calculate optimal rows/cols based on number of plots"""
            if n == 1:
                return 1, 1
            elif n == 2:
                return 1, 2
            elif n == 3:
                return 1, 3
            elif n == 4:
                return 2, 2
            elif n <= 6:
                return 2, 3
            elif n <= 9:
                return 3, 3
            elif n <= 12:
                return 3, 4
            else:
                # For many plots, prefer more columns than rows (easier to scroll vertically)
                cols = int(np.ceil(np.sqrt(n * 1.5)))
                rows = int(np.ceil(n / cols))
                return rows, cols

        rows, cols = calculate_subplot_layout(num_plots)

        # Adaptive figure dimensions
        def calculate_figure_dimensions(rows, cols, compact_mode, aspect):
            """Calculate optimal figure width and height"""
            if compact_mode:
                base_subplot_height = 200
                base_subplot_width = base_subplot_height * aspect
            else:
                # Scale based on layout
                if rows == 1:
                    base_subplot_height = 400  # Taller for single row
                elif rows == 2:
                    base_subplot_height = 350
                elif rows == 3:
                    base_subplot_height = 300
                else:
                    base_subplot_height = 280

                base_subplot_width = base_subplot_height * aspect

            # Calculate total dimensions
            height = int(rows * base_subplot_height)
            width = int(cols * base_subplot_width)

            # Apply caps
            if max_height is not None:
                height = min(height, max_height)
            else:
                # Default max based on typical screen heights
                height = min(height, 1400)

            if max_width is not None:
                width = min(width, max_width)
            else:
                # Default max based on typical screen widths
                width = min(width, 1800)

            return width, height

        fig_width, fig_height = calculate_figure_dimensions(
            rows, cols, compact, aspect_ratio
        )

        # Adaptive spacing based on layout
        if rows > 3:
            vertical_spacing = 0.15
        else:
            vertical_spacing = 0.12

        if cols > 3:
            horizontal_spacing = 0.08
        else:
            horizontal_spacing = 0.10

        # State names
        if state_names is None:
            state_names = [f"x{i}" for i in range(self.nx)]

        subplot_titles = state_names.copy()
        if has_control and control_names == None:
            control_names = [f"u{i}" for i in range(self.nu)]
            subplot_titles.extend(control_names)
        elif has_control:
            subplot_titles.extend(control_names)

        fig = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=subplot_titles,
            vertical_spacing=vertical_spacing,
            horizontal_spacing=horizontal_spacing,
        )

        # Time axis
        T = traj_np.shape[1]
        time_steps = np.arange(T) * self.dt

        # Adaptive font sizes based on compact mode and number of plots
        if compact:
            title_font_size = 12 if num_plots > 12 else 14
            axis_font_size = 10 if num_plots > 12 else 11
            tick_font_size = 9 if num_plots > 12 else 10
        else:
            title_font_size = 14
            axis_font_size = 12
            tick_font_size = 10

        # Plot states
        for i in range(self.nx):
            row = i // cols + 1
            col = i % cols + 1

            for b in range(batch_size):
                # Use same color for same trajectory across all subplots
                color = colors[b % len(colors)]

                # Create trajectory name
                if batch_size > 1:
                    if trajectory_names is not None:
                        traj_name = trajectory_names[b]
                    else:
                        traj_name = f"Trajectory {b+1}"
                else:
                    traj_name = state_names[i]

                fig.add_trace(
                    go.Scatter(
                        x=time_steps,
                        y=traj_np[b, :, i],
                        mode="lines",
                        name=traj_name,
                        showlegend=(i == 0),  # Only show legend in first subplot
                        legendgroup=f"traj_{b}",  # Group for linked legend behavior
                        line=dict(width=2, color=color),
                    ),
                    row=row,
                    col=col,
                )

            fig.update_xaxes(
                title_text="Time (s)",
                row=row,
                col=col,
                title_font=dict(size=axis_font_size),
                tickfont=dict(size=tick_font_size),
            )
            fig.update_yaxes(
                title_text=state_names[i],
                row=row,
                col=col,
                title_font=dict(size=axis_font_size),
                tickfont=dict(size=tick_font_size),
            )

        # Plot controls
        if has_control:
            control_np = control_sequence.detach().cpu().numpy()
            control_time = np.arange(control_np.shape[1]) * self.dt

            for i in range(self.nu):
                plot_idx = self.nx + i
                row = plot_idx // cols + 1
                col = plot_idx % cols + 1

                for b in range(batch_size):
                    color = colors[b % len(colors)]

                    if batch_size > 1:
                        if trajectory_names is not None:
                            traj_name = trajectory_names[b]
                        else:
                            traj_name = f"Trajectory {b+1}"
                    else:
                        traj_name = f"u{i}"

                    fig.add_trace(
                        go.Scatter(
                            x=control_time,
                            y=control_np[b, :, i],
                            mode="lines",
                            name=traj_name,
                            showlegend=False,  # Controls use same legend as states
                            legendgroup=f"traj_{b}",
                            line=dict(width=2, dash="dash", color=color),
                        ),
                        row=row,
                        col=col,
                    )

                fig.update_xaxes(
                    title_text="Time (s)",
                    row=row,
                    col=col,
                    title_font=dict(size=axis_font_size),
                    tickfont=dict(size=tick_font_size),
                )
                fig.update_yaxes(
                    title_text=control_names[i],
                    row=row,
                    col=col,
                    title_font=dict(size=axis_font_size),
                    tickfont=dict(size=tick_font_size),
                )

        # Update layout
        if title is None:
            title = f"{self.continuous_time_system.__class__.__name__} Trajectory"

        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=title_font_size + 4),
            ),
            width=fig_width,
            height=fig_height,
            showlegend=True,
            hovermode="x unified",
            font=dict(size=tick_font_size),
        )

        # Update subplot title font sizes
        for annotation in fig.layout.annotations:
            annotation.font.size = title_font_size

        # Save if requested
        if save_html:
            fig.write_html(save_html)
            print(
                f"Interactive plot saved to {save_html} (size: {fig_width}x{fig_height}px)"
            )

        # Show if requested
        if show:
            fig.show()

        return fig

    def plot_trajectory_3d(
        self,
        trajectory: torch.Tensor,
        state_indices: Tuple[int, int, int] = (0, 1, 2),
        state_names: Optional[Tuple[str, str, str]] = None,
        title: Optional[str] = None,
        trajectory_names: Optional[List[str]] = None,
        colorway: Union[str, List[str]] = "Plotly",
        save_html: Optional[str] = None,
        show: bool = True,
        show_markers: bool = True,
        marker_size: int = 2,
        line_width: int = 3,
    ):
        """
        Plot 3D trajectory visualization with time-colored paths.

        Args:
            trajectory: State trajectory (T, nx) or (batch, T, nx)
            state_indices: Which three states to plot (default: first three)
            state_names: Names for the three states
            title: Plot title
            trajectory_names: Optional names for each trajectory
            colorway: Plotly color sequence name or list of colors
            save_html: If provided, save to this HTML file
            show: If True, display the plot
            show_markers: If True, show markers along trajectory
            marker_size: Size of markers (default: 2)
            line_width: Width of trajectory lines (default: 3)

        Example:
            >>> # Single trajectory with time coloring
            >>> system.plot_trajectory_3d(traj, state_indices=(0, 1, 2),
            ...                          state_names=('x', 'y', 'z'))
            >>>
            >>> # Multiple trajectories from different initial conditions
            >>> trajs = torch.stack([traj1, traj2, traj3])
            >>> system.plot_trajectory_3d(trajs, trajectory_names=['IC1', 'IC2', 'IC3'])
        """

        # Handle batched trajectories
        if len(trajectory.shape) == 3:
            batch_size = trajectory.shape[0]
        else:
            trajectory = trajectory.unsqueeze(0)
            batch_size = 1

        traj_np = trajectory.detach().cpu().numpy()

        # Get color sequence
        if isinstance(colorway, str):
            color_sequences = {
                "Plotly": px.colors.qualitative.Plotly,
                "D3": px.colors.qualitative.D3,
                "G10": px.colors.qualitative.G10,
                "T10": px.colors.qualitative.T10,
                "Alphabet": px.colors.qualitative.Alphabet,
                "Dark24": px.colors.qualitative.Dark24,
                "Light24": px.colors.qualitative.Light24,
                "Set1": px.colors.qualitative.Set1,
                "Pastel": px.colors.qualitative.Pastel,
                "Vivid": px.colors.qualitative.Vivid,
            }
            colors = color_sequences.get(colorway, px.colors.qualitative.Plotly)
        else:
            colors = colorway

        idx0, idx1, idx2 = state_indices
        if state_names is None:
            state_names = (f"x{idx0}", f"x{idx1}", f"x{idx2}")

        fig = go.Figure()

        # Time axis for color gradient
        T = traj_np.shape[1]
        time_steps = np.arange(T) * self.dt

        # Different colormaps for each trajectory
        colormaps = [
            "Viridis",
            "Plasma",
            "Inferno",
            "Magma",
            "Cividis",
            "Turbo",
            "Blues",
            "Greens",
            "Reds",
            "Purples",
        ]

        # Plot each trajectory
        for b in range(batch_size):
            color = colors[b % len(colors)]
            colormap = colormaps[b % len(colormaps)]

            if trajectory_names is not None:
                traj_name = trajectory_names[b]
            else:
                traj_name = f"Trajectory {b+1}" if batch_size > 1 else "Trajectory"

            # Main trajectory line with time-based color gradient
            # Use different colormap for each trajectory to distinguish them
            mode = "lines+markers" if show_markers else "lines"

            # For single trajectory, use time coloring with colorbar
            # For multiple trajectories, use solid colors to distinguish
            if batch_size == 1:
                line_config = dict(
                    width=line_width,
                    color=time_steps,
                    colorscale=colormap,
                    showscale=True,
                    colorbar=dict(
                        title="Time (s)",
                        x=1.02,
                        xanchor="left",
                        len=0.75,
                        y=0.5,
                        yanchor="middle",
                    ),
                )
                marker_config = (
                    dict(
                        size=marker_size,
                        color=time_steps,
                        colorscale=colormap,
                        showscale=False,
                    )
                    if show_markers
                    else None
                )
            else:
                # Multiple trajectories: use solid colors for clarity
                line_config = dict(width=line_width, color=color)
                marker_config = (
                    dict(size=marker_size, color=color) if show_markers else None
                )

            fig.add_trace(
                go.Scatter3d(
                    x=traj_np[b, :, idx0],
                    y=traj_np[b, :, idx1],
                    z=traj_np[b, :, idx2],
                    mode=mode,
                    name=traj_name,
                    line=line_config,
                    marker=marker_config,
                    legendgroup=f"traj_{b}",
                    hovertemplate=f"<b>{traj_name}</b><br>"
                    f"{state_names[0]}: %{{x:.4f}}<br>"
                    f"{state_names[1]}: %{{y:.4f}}<br>"
                    f"{state_names[2]}: %{{z:.4f}}<br>"
                    f"Time: %{{text:.3f}}s<extra></extra>",
                    text=time_steps,
                )
            )

        # Add start markers
        for b in range(batch_size):
            fig.add_trace(
                go.Scatter3d(
                    x=[traj_np[b, 0, idx0]],
                    y=[traj_np[b, 0, idx1]],
                    z=[traj_np[b, 0, idx2]],
                    mode="markers",
                    name="Start" if b == 0 else None,
                    marker=dict(size=8, color="green", symbol="diamond"),
                    showlegend=(b == 0),
                    legendgroup="markers",
                    hovertemplate="<b>Start</b><extra></extra>",
                )
            )

        # Add end markers
        for b in range(batch_size):
            fig.add_trace(
                go.Scatter3d(
                    x=[traj_np[b, -1, idx0]],
                    y=[traj_np[b, -1, idx1]],
                    z=[traj_np[b, -1, idx2]],
                    mode="markers",
                    name="End" if b == 0 else None,
                    marker=dict(size=8, color="red", symbol="x"),
                    showlegend=(b == 0),
                    legendgroup="markers",
                    hovertemplate="<b>End</b><extra></extra>",
                )
            )

        # Add equilibrium marker if system has at least 3 states
        if self.nx >= 3:
            x_eq = self.continuous_time_system.x_equilibrium.detach().cpu().numpy()
            fig.add_trace(
                go.Scatter3d(
                    x=[x_eq[idx0]],
                    y=[x_eq[idx1]],
                    z=[x_eq[idx2]],
                    mode="markers",
                    name="Equilibrium",
                    marker=dict(size=10, color="black", symbol="square"),
                    legendgroup="markers",
                    hovertemplate="<b>Equilibrium</b><extra></extra>",
                )
            )

        if title is None:
            title = f"{self.continuous_time_system.__class__.__name__} 3D Trajectory"

        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title=state_names[0],
                yaxis_title=state_names[1],
                zaxis_title=state_names[2],
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.3)),
            ),
            hovermode="closest",
            width=1000,
            height=700,
            legend=dict(
                x=0.02,
                y=0.98,
                xanchor="left",
                yanchor="top",
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="rgba(0, 0, 0, 0.3)",
                borderwidth=1,
            ),
        )

        if save_html:
            fig.write_html(save_html)
            print(f"3D trajectory saved to {save_html}")

        if show:
            fig.show()

        return fig

    def plot_phase_portrait_2d(
        self,
        trajectory: torch.Tensor,
        state_indices: Tuple[int, int] = (0, 1),
        state_names: Optional[Tuple[str, str]] = None,
        title: Optional[str] = None,
        trajectory_names: Optional[List[str]] = None,
        colorway: Union[str, List[str]] = "Plotly",
        save_html: Optional[str] = None,
        show: bool = True,
    ):
        """
        Plot 2D phase portrait

        Args:
            trajectory: State trajectory (T, nx) or (batch, T, nx)
            state_indices: Which two states to plot (default: first two)
            state_names: Names for the two states
            title: Plot title
            trajectory_names: Optional names for each trajectory (uses "Trajectory 1", etc. if None)
            colorway: Plotly color sequence name or list of colors
            save_html: If provided, save to this HTML file
            show: If True, display the plot
        """

        # Handle batched trajectories
        if len(trajectory.shape) == 3:
            batch_size = trajectory.shape[0]
        else:
            trajectory = trajectory.unsqueeze(0)
            batch_size = 1

        traj_np = trajectory.detach().cpu().numpy()

        # Get color sequence
        if isinstance(colorway, str):
            color_sequences = {
                "Plotly": px.colors.qualitative.Plotly,
                "D3": px.colors.qualitative.D3,
                "G10": px.colors.qualitative.G10,
                "T10": px.colors.qualitative.T10,
                "Alphabet": px.colors.qualitative.Alphabet,
                "Dark24": px.colors.qualitative.Dark24,
                "Light24": px.colors.qualitative.Light24,
                "Set1": px.colors.qualitative.Set1,
                "Pastel": px.colors.qualitative.Pastel,
                "Vivid": px.colors.qualitative.Vivid,
            }
            colors = color_sequences.get(colorway, px.colors.qualitative.Plotly)
        else:
            colors = colorway

        idx0, idx1 = state_indices
        if state_names is None:
            state_names = (f"x{idx0}", f"x{idx1}")

        fig = go.Figure()

        # First: Add all trajectory lines
        for b in range(batch_size):
            color = colors[b % len(colors)]

            if trajectory_names is not None:
                traj_name = trajectory_names[b]
            else:
                traj_name = f"Trajectory {b+1}" if batch_size > 1 else "Trajectory"

            fig.add_trace(
                go.Scatter(
                    x=traj_np[b, :, idx0],
                    y=traj_np[b, :, idx1],
                    mode="lines+markers",
                    name=traj_name,
                    line=dict(width=2, color=color),
                    marker=dict(size=4, color=color),
                    legendgroup=f"traj_{b}",
                )
            )

        # Second: Add start/end markers for all trajectories
        for b in range(batch_size):
            fig.add_trace(
                go.Scatter(
                    x=[traj_np[b, 0, idx0]],
                    y=[traj_np[b, 0, idx1]],
                    mode="markers",
                    name="Start" if b == 0 else None,
                    marker=dict(size=12, color="green", symbol="star"),
                    showlegend=(b == 0),
                    legendgroup="markers",
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=[traj_np[b, -1, idx0]],
                    y=[traj_np[b, -1, idx1]],
                    mode="markers",
                    name="End" if b == 0 else None,
                    marker=dict(size=12, color="red", symbol="x"),
                    showlegend=(b == 0),
                    legendgroup="markers",
                )
            )

        # Finally: Add equilibrium marker
        if self.nx >= 2:
            x_eq = self.continuous_time_system.x_equilibrium.detach().cpu().numpy()
            fig.add_trace(
                go.Scatter(
                    x=[x_eq[idx0]],
                    y=[x_eq[idx1]],
                    mode="markers",
                    name="Equilibrium",
                    marker=dict(size=15, color="black", symbol="diamond"),
                    legendgroup="markers",
                )
            )

        if title is None:
            title = f"{self.continuous_time_system.__class__.__name__} Phase Portrait"

        fig.update_layout(
            title=title,
            xaxis_title=state_names[0],
            yaxis_title=state_names[1],
            hovermode="closest",
            width=800,
            height=600,
        )

        if save_html:
            fig.write_html(save_html)
            print(f"Phase portrait saved to {save_html}")

        if show:
            fig.show()

        return fig

    def plot_phase_portrait_3d(
        self,
        trajectory: torch.Tensor,
        state_indices: Tuple[int, int, int] = (0, 1, 2),
        state_names: Optional[Tuple[str, str, str]] = None,
        title: Optional[str] = None,
        trajectory_names: Optional[List[str]] = None,
        colorway: Union[str, List[str]] = "Plotly",
        save_html: Optional[str] = None,
        show: bool = True,
        show_time_markers: bool = False,
        marker_interval: int = 10,
    ):
        """
        Plot 3D phase portrait (state space visualization without time coloring).

        Unlike plot_trajectory_3d which uses time-based color gradients, this function
        uses solid colors per trajectory for clearer distinction between multiple paths.

        Args:
            trajectory: State trajectory (T, nx) or (batch, T, nx)
            state_indices: Which three states to plot (default: first three)
            state_names: Names for the three states
            title: Plot title
            trajectory_names: Optional names for each trajectory
            colorway: Plotly color sequence name or list of colors
            save_html: If provided, save to this HTML file
            show: If True, display the plot
            show_time_markers: If True, add periodic markers showing time progression
            marker_interval: Interval for time markers (e.g., every 10 steps)

        Example:
            >>> # Compare multiple trajectories in phase space
            >>> trajs = torch.stack([traj_ic1, traj_ic2, traj_ic3])
            >>> system.plot_phase_portrait_3d(trajs,
            ...     state_indices=(0, 2, 4),
            ...     state_names=('x', 'theta', 'v_x'),
            ...     trajectory_names=['Stable', 'Limit Cycle', 'Divergent'])
        """

        # Handle batched trajectories
        if len(trajectory.shape) == 3:
            batch_size = trajectory.shape[0]
        else:
            trajectory = trajectory.unsqueeze(0)
            batch_size = 1

        traj_np = trajectory.detach().cpu().numpy()

        # Get color sequence
        if isinstance(colorway, str):
            color_sequences = {
                "Plotly": px.colors.qualitative.Plotly,
                "D3": px.colors.qualitative.D3,
                "G10": px.colors.qualitative.G10,
                "T10": px.colors.qualitative.T10,
                "Alphabet": px.colors.qualitative.Alphabet,
                "Dark24": px.colors.qualitative.Dark24,
                "Light24": px.colors.qualitative.Light24,
                "Set1": px.colors.qualitative.Set1,
                "Pastel": px.colors.qualitative.Pastel,
                "Vivid": px.colors.qualitative.Vivid,
            }
            colors = color_sequences.get(colorway, px.colors.qualitative.Plotly)
        else:
            colors = colorway

        idx0, idx1, idx2 = state_indices
        if state_names is None:
            state_names = (f"x{idx0}", f"x{idx1}", f"x{idx2}")

        fig = go.Figure()

        T = traj_np.shape[1]
        time_steps = np.arange(T) * self.dt

        # Plot each trajectory
        for b in range(batch_size):
            color = colors[b % len(colors)]

            if trajectory_names is not None:
                traj_name = trajectory_names[b]
            else:
                traj_name = f"Trajectory {b+1}" if batch_size > 1 else "Trajectory"

            # Main trajectory line (solid color)
            fig.add_trace(
                go.Scatter3d(
                    x=traj_np[b, :, idx0],
                    y=traj_np[b, :, idx1],
                    z=traj_np[b, :, idx2],
                    mode="lines",
                    name=traj_name,
                    line=dict(width=3, color=color),
                    legendgroup=f"traj_{b}",
                    hovertemplate=f"<b>{traj_name}</b><br>"
                    f"{state_names[0]}: %{{x:.4f}}<br>"
                    f"{state_names[1]}: %{{y:.4f}}<br>"
                    f"{state_names[2]}: %{{z:.4f}}<br>"
                    f"Time: %{{text:.3f}}s<extra></extra>",
                    text=time_steps,
                )
            )

            # Add periodic time markers if requested
            if show_time_markers:
                marker_indices = np.arange(0, T, marker_interval)
                fig.add_trace(
                    go.Scatter3d(
                        x=traj_np[b, marker_indices, idx0],
                        y=traj_np[b, marker_indices, idx1],
                        z=traj_np[b, marker_indices, idx2],
                        mode="markers",
                        name=f"{traj_name} - Time Markers" if batch_size == 1 else None,
                        marker=dict(size=3, color=color, opacity=0.5),
                        showlegend=False,
                        legendgroup=f"traj_{b}",
                        hovertemplate=f"t=%{{text:.3f}}s<extra></extra>",
                        text=time_steps[marker_indices],
                    )
                )

        # Add start markers
        for b in range(batch_size):
            fig.add_trace(
                go.Scatter3d(
                    x=[traj_np[b, 0, idx0]],
                    y=[traj_np[b, 0, idx1]],
                    z=[traj_np[b, 0, idx2]],
                    mode="markers",
                    name="Start" if b == 0 else None,
                    marker=dict(size=10, color="green", symbol="diamond"),
                    showlegend=(b == 0),
                    legendgroup="markers",
                    hovertemplate="<b>Start</b><extra></extra>",
                )
            )

        # Add end markers
        for b in range(batch_size):
            fig.add_trace(
                go.Scatter3d(
                    x=[traj_np[b, -1, idx0]],
                    y=[traj_np[b, -1, idx1]],
                    z=[traj_np[b, -1, idx2]],
                    mode="markers",
                    name="End" if b == 0 else None,
                    marker=dict(size=10, color="red", symbol="x"),
                    showlegend=(b == 0),
                    legendgroup="markers",
                    hovertemplate="<b>End</b><extra></extra>",
                )
            )

        # Add equilibrium marker
        if self.nx >= 3:
            x_eq = self.continuous_time_system.x_equilibrium.detach().cpu().numpy()
            fig.add_trace(
                go.Scatter3d(
                    x=[x_eq[idx0]],
                    y=[x_eq[idx1]],
                    z=[x_eq[idx2]],
                    mode="markers",
                    name="Equilibrium",
                    marker=dict(size=12, color="black", symbol="square"),
                    legendgroup="markers",
                    hovertemplate="<b>Equilibrium</b><extra></extra>",
                )
            )

        if title is None:
            title = (
                f"{self.continuous_time_system.__class__.__name__} 3D Phase Portrait"
            )

        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title=state_names[0],
                yaxis_title=state_names[1],
                zaxis_title=state_names[2],
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.3)),
            ),
            hovermode="closest",
            width=900,
            height=700,
        )

        if save_html:
            fig.write_html(save_html)
            print(f"3D phase portrait saved to {save_html}")

        if show:
            fig.show()

        return fig
