# src/systems/base/generic_discrete_time_system.py

class GenericDiscreteTimeSystem:  # No longer nn.Module
    """
    Discrete-time wrapper for continuous systems.
    
    Core responsibilities:
    - Numerical integration (Euler, RK4, etc.)
    - Simulation
    """
    
    def __init__(self, continuous_time_system, dt, 
                 integration_method=IntegrationMethod.ExplicitEuler,
                 position_integration=None):
        """Initialize discrete-time system wrapper"""
        
        self.continuous_time_system = continuous_time_system
        self.nx = continuous_time_system.nx
        self.nu = continuous_time_system.nu
        self.dt = float(dt)
        self.order = continuous_time_system.order
        self.integration_method = integration_method
        self.position_integration = position_integration or integration_method
        
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt}")
    
    def forward(self, x, u):
        """Compute x[k+1] = f_discrete(x[k], u[k])"""
        if self.order == 1:
            return self._integrate_first_order(x, u)
        elif self.order == 2:
            return self._integrate_second_order(x, u)
        else:
            return self._integrate_arbitrary_order(x, u)
    
    def __call__(self, x, u):
        """Make callable"""
        return self.forward(x, u)
    
    def simulate(self, x0, controller=None, horizon=None, 
                return_controls=False, return_all=True, observer=None):
        """Simulate trajectory"""
        # ... existing implementation ...
    
    def linearized_dynamics(self, x, u):
        """Linearized discrete dynamics"""
        Ac, Bc = self.continuous_time_system.linearized_dynamics(x, u)
        # Euler approximation for discrete-time
        Ad = self.dt * Ac + torch.eye(self.nx).to(x.device)
        Bd = self.dt * Bc
        return Ad, Bd
    
    # Integration methods (keep these)
    def _integrate_first_order(self, x, u):
        # ... existing ...
    
    def _integrate_second_order(self, x, u):
        # ... existing ...
    
    def _integrate_arbitrary_order(self, x, u):
        # ... existing ...
    
    # REMOVED: dlqr_control, discrete_kalman_gain, dlqg_control
    # These move to ControlDesigner
    
    # Convenience: Quick access to control design
    @property
    def control(self):
        """Get control designer for this system"""
        if not hasattr(self, '_control_designer'):
            from src.control import ControlDesigner
            self._control_designer = ControlDesigner(self)
        return self._control_designer
    
    # KEEP visualization methods (or move to separate Visualizer class)
    def plot_trajectory(self, ...):
        # ... existing ...