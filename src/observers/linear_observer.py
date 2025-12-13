import numpy as np
import torch
from typing import Optional

class LinearObserver:
    """
    Linear state observer with constant gain.

    Implements the observer:
        d x̂/dt = f(x̂, u) + L(y - h(x̂))                         [Continuous-time]
        x̂[k+1] = f(x̂[k], u[k]) + L(y[k+1] - h(f(x̂[k], u[k])))  [Discrete-time]

    where:
    - x̂ is the state estimate
    - L is the observer gain matrix
    - y is the measurement
    - h(x̂) is the predicted measurement

    Attributes:
        system: SymbolicDynamicalSystem or GenericDiscreteTimeSystem
        L: Observer gain matrix (nx, ny)
        x_hat: Current state estimate (nx,)

    Example - Continuous-Time Observer:
        >>> # Design Kalman gain
        >>> system = SymbolicPendulum(m=1.0, l=0.5)
        >>> Q_process = np.diag([0.001, 0.01])
        >>> R_measurement = np.array([[0.1]])
        >>> L = system.kalman_gain(Q_process, R_measurement)
        >>>
        >>> # Create observer
        >>> observer = LinearObserver(system, L)
        >>> observer.reset(x0=torch.zeros(2))
        >>>
        >>> # Observer loop
        >>> dt = 0.01
        >>> for t in range(num_steps):
        >>>     observer.update(u[t], y_measured[t], dt)
        >>>     x_estimate = observer.x_hat

    Example - Discrete-Time Observer:
        >>> # Create discrete system
        >>> system_ct = SymbolicQuadrotor2D()
        >>> system_dt = GenericDiscreteTimeSystem(system_ct, dt=0.01)
        >>>
        >>> # Design discrete Kalman gain
        >>> L = system_dt.discrete_kalman_gain(Q_process, R_measurement)
        >>>
        >>> # Create observer
        >>> observer = LinearObserver(system_dt, L)
        >>>
        >>> # Observer loop (no dt needed for discrete)
        >>> for k in range(num_steps):
        >>>     observer.update(u[k], y[k], dt=None)  # dt ignored for discrete
        >>>     x_estimate = observer.x_hat

    Example - Output Feedback Control:
        >>> # Design LQG controller
        >>> K, L = system.lqg_control(Q_lqr, R_lqr, Q_process, R_measurement)
        >>>
        >>> # Create controller and observer
        >>> controller = LinearController(K, system.x_equilibrium, system.u_equilibrium)
        >>> observer = LinearObserver(system, L)
        >>>
        >>> # Closed-loop with output feedback
        >>> x_true = x0
        >>> observer.reset(x0=system.x_equilibrium)  # Start from equilibrium guess
        >>>
        >>> for t in range(num_steps):
        >>>     # Measure (with noise)
        >>>     y = system.h(x_true) + torch.randn(system.ny) * 0.1
        >>>
        >>>     # Update observer
        >>>     observer.update(u, y, dt)
        >>>
        >>>     # Compute control based on estimate
        >>>     u = controller(observer.x_hat)
        >>>
        >>>     # Update true system
        >>>     x_true = system(x_true, u) if discrete else integrate(x_true, u, dt)

    Notes:
        - Gain L is constant and designed at equilibrium
        - For nonlinear systems, only accurate near equilibrium
        - No covariance tracking
        - Can be used with both continuous and discrete systems
        - Lower computational cost than EKF

    See Also:
        kalman_gain: Design L for continuous systems
        discrete_kalman_gain: Design L for discrete systems
        ExtendedKalmanFilter: Adaptive nonlinear observer
        LinearController: State feedback controller
        lqg_control: Combined controller and observer design
    """

    def __init__(self, system, L: np.ndarray):
        """
        Initialize linear observer with constant gain.

        Args:
            system: SymbolicDynamicalSystem or GenericDiscreteTimeSystem
            L: Observer gain matrix (nx, ny). Typically from kalman_gain()
               or discrete_kalman_gain().

        Example:
            >>> L = system.kalman_gain(Q_process, R_measurement)
            >>> observer = LinearObserver(system, L)
        """
        self.system = system
        self.L = torch.tensor(L, dtype=torch.float32)
        self.x_hat = system.x_equilibrium.clone()

    def update(self, u: torch.Tensor, y: torch.Tensor, dt: float):
        """
        Update observer state estimate.

        Continuous-time:
            d x̂/dt = f(x̂, u) + L(y - h(x̂))
            x̂_new ≈ x̂_old + dt * [f(x̂, u) + L(y - h(x̂))]

        Discrete-time:
            x̂_pred = f(x̂, u)
            x̂_new = x̂_pred + L(y - h(x̂_pred))

        Args:
            u: Control input (nu,)
            y: Measurement (ny,)
            dt: Time step (used for continuous systems, ignored for discrete)

        Example:
            >>> u = torch.tensor([1.0])
            >>> y_measured = torch.tensor([0.15])  # Noisy angle measurement
            >>> observer.update(u, y_measured, dt=0.01)
            >>> print(f"Estimate: {observer.x_hat}")

        Notes:
            - For continuous systems: uses Euler integration
            - For discrete systems: dt parameter is ignored
            - Innovation = y - h(x̂_pred) is the measurement residual
            - Large innovation suggests either bad estimate or bad measurement
            - Gain L determines how much to trust innovation vs model
        """
        # Predict
        with torch.no_grad():
            if hasattr(self.system, "continuous_time_system"):
                # Discrete system
                x_pred = self.system(self.x_hat.unsqueeze(0), u.unsqueeze(0)).squeeze(0)
                y_pred = self.system.continuous_time_system.h(
                    x_pred.unsqueeze(0)
                ).squeeze(0)
            else:
                # Continuous system
                dx = self.system.forward(
                    self.x_hat.unsqueeze(0), u.unsqueeze(0)
                ).squeeze(0)
                x_pred = self.x_hat + dx * dt
                y_pred = self.system.h(x_pred.unsqueeze(0)).squeeze(0)

            # Correct
            innovation = y - y_pred
            self.x_hat = x_pred + (self.L @ innovation.unsqueeze(-1)).squeeze(-1)

    def reset(self, x0: Optional[torch.Tensor] = None):
        """
        Reset observer to initial state estimate.

        Args:
            x0: Initial state estimate (nx,). Uses equilibrium if None.

        Example:
            >>> # Reset to known initial condition
            >>> observer.reset(x0=torch.tensor([0.1, 0.0]))
            >>>
            >>> # Reset to equilibrium
            >>> observer.reset()  # Uses system.x_equilibrium

        Notes:
            - Called automatically in __init__()
            - Unlike EKF, no covariance to reset
            - Start observer from best available estimate
        """
        if x0 is not None:
            self.x_hat = x0.clone()
        else:
            self.x_hat = self.system.x_equilibrium.clone()

    def to(self, device):
        """
        Move observer to specified device (CPU/GPU).

        Args:
            device: torch.device or string ('cpu', 'cuda')

        Returns:
            Self for chaining
        """
        self.L = self.L.to(device)
        self.x_hat = self.x_hat.to(device)
        return self