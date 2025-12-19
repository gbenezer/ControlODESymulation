import numpy as np
import torch

class LinearController:
    """
    Linear state feedback controller with equilibrium offset.

    Implements the control law:
        u(x) = K @ (x - x_eq) + u_eq

    where:
    - K is the control gain matrix
    - x_eq is the equilibrium/reference state
    - u_eq is the equilibrium/feedforward control

    Valid Region:
    ------------
    - **Linear systems**: Globally valid
    - **Nonlinear systems**: Valid near equilibrium where linearization holds
    - Performance degrades as ||x - x_eq|| increases

    Attributes:
        K: Control gain matrix (nu, nx)
        x_eq: Equilibrium/reference state (nx,)
        u_eq: Equilibrium/feedforward control (nu,)

    Example - LQR Control:
        >>> # Design LQR controller
        >>> system = SymbolicPendulum(m=1.0, l=0.5)
        >>> Q = np.diag([10.0, 1.0])
        >>> R = np.array([[0.1]])
        >>> K, S = system.lqr_control(Q, R)
        >>>
        >>> # Create controller
        >>> controller = LinearController(K, system.x_equilibrium, system.u_equilibrium)
        >>>
        >>> # Use in simulation
        >>> x = torch.tensor([0.1, 0.0])  # Small deviation from equilibrium
        >>> u = controller(x)
        >>> print(f"Control: {u}")

    Example - Tracking Reference:
        >>> # Track different equilibrium
        >>> x_ref = torch.tensor([0.5, 0.0])  # Desired position
        >>> u_ref = compute_feedforward(x_ref)
        >>>
        >>> # Controller drives x → x_ref
        >>> tracking_controller = LinearController(K, x_ref, u_ref)
        >>>
        >>> # In control loop
        >>> for t in range(steps):
        >>>     u = tracking_controller(x[t])
        >>>     x[t+1] = system(x[t], u)

    Example - Output Feedback (with Observer):
        >>> # Design LQG
        >>> K, L = system.lqg_control(Q_lqr, R_lqr, Q_process, R_measurement)
        >>>
        >>> # Controller uses state estimate, not true state
        >>> controller = LinearController(K, system.x_equilibrium, system.u_equilibrium)
        >>> observer = LinearObserver(system, L)
        >>>
        >>> for t in range(steps):
        >>>     y = measure(x_true[t])
        >>>     observer.update(u, y, dt)
        >>>     u = controller(observer.x_hat)  # Use estimate!
        >>>     x_true[t+1] = system(x_true[t], u)

    Example - Gain Scheduling:
        >>> # Different gains for different regions
        >>> K_upright = system.lqr_control(Q1, R)[0]  # Near upright
        >>> K_hanging = system.lqr_control(Q2, R)[0]  # Near hanging
        >>>
        >>> controller_upright = LinearController(K_upright, x_eq_up, u_eq_up)
        >>> controller_hanging = LinearController(K_hanging, x_eq_down, u_eq_down)
        >>>
        >>> # Switch based on state
        >>> def adaptive_control(x):
        >>>     if abs(x[0]) < np.pi/4:  # Near upright
        >>>         return controller_upright(x)
        >>>     else:
        >>>         return controller_hanging(x)

    Notes:
        - The feedforward term u_eq ensures u=u_eq at x=x_eq
        - For tracking time-varying references, update x_eq and u_eq online
        - Can be used with state estimation (observer-based control)
        - Handles both 1D and batched inputs automatically

    See Also:
        lqr_control: Design optimal K matrix
        LinearObserver: State estimation for output feedback
        lqg_control: Combined controller and observer design
    """

    def __init__(self, K: np.ndarray, x_eq: torch.Tensor, u_eq: torch.Tensor):
        """
        Initialize linear state feedback controller.

        Args:
            K: Control gain matrix (nu, nx).
            x_eq: Equilibrium/reference state (nx,)
            u_eq: Equilibrium/feedforward control (nu,)

        Example:
            >>> K = np.array([[-12.5, -5.0]])  # SISO system
            >>> x_eq = torch.zeros(2)
            >>> u_eq = torch.zeros(1)
            >>> controller = LinearController(K, x_eq, u_eq)
        """
        self.K = torch.tensor(K, dtype=torch.float32)
        self.x_eq = x_eq
        self.u_eq = u_eq

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute control input: u = K @ (x - x_eq) + u_eq

        Args:
            x: Current state (nx,) or (batch, nx)

        Returns:
            u: Control input (nu,) or (batch, nu)

        Example:
            >>> x = torch.tensor([0.1, 0.05])
            >>> u = controller(x)
            >>> print(f"Control: {u}")
            >>>
            >>> # Batched computation
            >>> x_batch = torch.randn(100, 2)  # 100 states
            >>> u_batch = controller(x_batch)  # 100 controls

        Notes:
            - Automatically handles 1D or 2D inputs
            - Returns same batch structure as input
            - All operations differentiable (can be used in learning)
        """
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        u = self.u_eq + (self.K @ (x - self.x_eq).T).T

        if squeeze:
            u = u.squeeze(0)

        return u

    def to(self, device):
        """
        Move controller to specified device (CPU/GPU).

        Args:
            device: torch.device or string ('cpu', 'cuda')

        Returns:
            Self for chaining
        """
        self.K = self.K.to(device)
        self.x_eq = self.x_eq.to(device)
        self.u_eq = self.u_eq.to(device)
        return self