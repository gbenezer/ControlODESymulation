"""
Discrete-time wrapper for continuous symbolic dynamical systems.

Provides discrete-time simulation, linearization, and control design
by composing specialized components.

Architecture:
- Discretizer: Handles continuous → discrete conversion
- DiscreteSimulator: Trajectory simulation
- DiscreteLinearization: Linearization caching

Examples
--------
>>> # Create discrete-time system
>>> pendulum = SymbolicPendulum(m=1.0, l=0.5, g=9.81)
>>> dt_system = DiscreteTimeSystem(pendulum, dt=0.01, method='rk4')
>>> 
>>> # Single step
>>> x_next = dt_system.step(x, u)
>>> 
>>> # Simulate trajectory
>>> traj = dt_system.simulate(x0, controller=lambda x: -K @ x, horizon=100)
>>> 
>>> # Linearize
>>> Ad, Bd = dt_system.linearized_dynamics(x_eq, u_eq)
"""