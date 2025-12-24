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