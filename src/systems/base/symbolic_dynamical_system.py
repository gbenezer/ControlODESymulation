# src/systems/base/symbolic_dynamical_system.py

class SymbolicDynamicalSystem(ABC):  # No nn.Module (now optional)
    """
    Symbolic dynamical system with multi-backend execution.
    
    Core responsibilities:
    - Symbolic definition
    - Code generation  
    - Multi-backend dispatch
    """
    
    def __init__(self, *args, **kwargs):
        # Symbolic definition
        self.state_vars: List[sp.Symbol] = []
        self.control_vars: List[sp.Symbol] = []
        self.output_vars: List[sp.Symbol] = []
        self.parameters: Dict[sp.Symbol, float] = {}
        self._f_sym: Optional[sp.Matrix] = None
        self._h_sym: Optional[sp.Matrix] = None
        self.order: int = 1
        
        # Backend configuration
        self._default_backend = "numpy"
        self._preferred_device = "cpu"
        
        # Cached functions
        self._f_numpy: Optional[Callable] = None
        self._f_torch: Optional[Callable] = None
        self._f_jax: Optional[Callable] = None
        self._h_numpy: Optional[Callable] = None
        self._h_torch: Optional[Callable] = None
        self._h_jax: Optional[Callable] = None
        
        # Cached Jacobians
        self._A_sym_cached: Optional[sp.Matrix] = None
        self._B_sym_cached: Optional[sp.Matrix] = None
        self._C_sym_cached: Optional[sp.Matrix] = None
        
        self._initialized: bool = False
        
        # Call template method
        self.define_system(*args, **kwargs)
        self._validate_system()
        
        # COMPOSITION: Delegate equilibrium management
        from src.systems.equilibrium import EquilibriumHandler
        self.equilibria = EquilibriumHandler(self.nx, self.nu)
        
        # COMPOSITION: Delegate performance monitoring (optional)
        from src.systems.monitoring import PerformanceMonitor
        self.performance = PerformanceMonitor()
    
    @abstractmethod
    def define_system(self, *args, **kwargs):
        """Define symbolic system (implemented by subclasses)"""
        pass
    
    # Core methods (keep these)
    def forward(self, x, u):
        """Multi-backend dynamics evaluation"""
        # ... existing implementation ...
    
    def linearized_dynamics(self, x, u):
        """Multi-backend linearization"""
        # ... existing implementation ...
    
    def linearized_observation(self, x):
        """Multi-backend observation linearization"""
        # ... existing implementation ...
    
    def h(self, x):
        """Multi-backend output evaluation"""
        # ... existing implementation ...
    
    # Code generation methods (keep these)
    def generate_numpy_function(self):
        # ... existing ...
    
    def generate_torch_function(self):
        # ... existing ...
    
    def generate_jax_function(self):
        # ... existing ...
    
    # Symbolic methods (keep these)
    def substitute_parameters(self, expr):
        # ... existing ...
    
    def linearized_dynamics_symbolic(self, x_eq, u_eq):
        # ... existing ...
    
    # Utility methods (keep these)
    def print_equations(self, simplify=True):
        # ... existing ...
    
    def verify_jacobians(self, x, u, tol=1e-3):
        # ... existing ...
    
    # REMOVED: LQR, Kalman, LQG methods
    # These move to ControlDesigner
    
    # REMOVED: Performance stats methods
    # These move to PerformanceMonitor
    
    # Convenience: Quick access to control design
    @property
    def control(self):
        """Get control designer for this system"""
        if not hasattr(self, '_control_designer'):
            from src.control import ControlDesigner
            self._control_designer = ControlDesigner(self)
        return self._control_designer