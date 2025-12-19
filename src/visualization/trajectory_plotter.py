# src/visualization/trajectory_plotter.py

class TrajectoryPlotter:
    """
    Handles all trajectory visualization for dynamical systems.
    
    Separates plotting logic from system dynamics.
    """
    
    def __init__(self, system):
        """
        Args:
            system: GenericDiscreteTimeSystem or SymbolicDynamicalSystem
        """
        self.system = system
    
    def plot_trajectory(self, trajectory, state_names=None, 
                       control_sequence=None, **kwargs):
        """Plot time-series trajectory"""
        # ... move from GenericDiscreteTimeSystem ...
    
    def plot_trajectory_3d(self, trajectory, state_indices=(0,1,2), **kwargs):
        """Plot 3D trajectory"""
        # ... move from GenericDiscreteTimeSystem ...
    
    def plot_phase_portrait_2d(self, trajectory, **kwargs):
        """Plot 2D phase portrait"""
        # ... move from GenericDiscreteTimeSystem ...

# Convenience in GenericDiscreteTimeSystem
class GenericDiscreteTimeSystem:
    @property
    def plot(self):
        """Get plotter for this system"""
        if not hasattr(self, '_plotter'):
            from src.visualization import TrajectoryPlotter
            self._plotter = TrajectoryPlotter(self)
        return self._plotter