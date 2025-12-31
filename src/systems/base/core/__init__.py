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
Core System Base Classes (Layer 1)
==================================

This module provides the abstract base classes that define the fundamental
interfaces for all dynamical systems in the ControlDESymulation framework.

Architecture Overview
--------------------
The core module implements Layer 1 of the four-layer system architecture:

Layer 1 (Abstract Interfaces) - YOU ARE HERE:
    - ContinuousSystemBase: Interface for continuous-time systems
    - DiscreteSystemBase: Interface for discrete-time systems

Layer 2 (Symbolic Implementations):
    - ContinuousSymbolicSystem(ContinuousSystemBase)
    - ContinuousStochasticSystem(ContinuousSymbolicSystem)

Layer 3 (Discrete Implementations):
    - DiscreteSymbolicSystem(DiscreteSystemBase)
    - DiscreteStochasticSystem(DiscreteSymbolicSystem, DiscreteSystemBase)

Layer 4 (Bridges):
    - DiscreteTimeWrapper
    - Discretizer

Key Design Principles
--------------------
1. **Clear Separation**: Continuous and discrete domains are completely separated
2. **Type Safety**: Full type hints using src/types throughout
3. **Minimal Interface**: Only essential methods are abstract
4. **Semantic Clarity**: Method names reflect intent (integrate vs simulate)
5. **Multi-Backend**: Works with NumPy, PyTorch, JAX

Type System Integration
----------------------
The base classes integrate with the comprehensive type system:

Result Types (from src/types/trajectories):
    - IntegrationResult: Low-level ODE solver output (adaptive time points)
    - SimulationResult: High-level continuous trajectory (regular time grid)
    - DiscreteSimulationResult: Discrete-time trajectory (integer time steps)

Linearization Types (from src/types/linearization):
    - Tuples: (A, B) for deterministic, (A, B, G) for stochastic
    - ContinuousLinearization: Alias for continuous systems
    - DiscreteLinearization: Alias for discrete systems

Core Types (from src/types/core):
    - StateVector, ControlVector, OutputVector
    - StateMatrix, InputMatrix, DiffusionMatrix
    - Backend-agnostic ArrayLike

Method Interfaces
----------------
ContinuousSystemBase:
    __call__(x, u, t) → StateVector
        Evaluate dynamics: dx/dt = f(x, u, t)
    
    integrate(x0, u, t_span, method) → IntegrationResult
        Low-level ODE solver with diagnostics (adaptive steps)
    
    simulate(x0, controller, t_span, dt) → SimulationResult
        High-level trajectory on regular time grid
    
    linearize(x_eq, u_eq) → (A, B) or (A, B, G)
        Compute Jacobian matrices at equilibrium

DiscreteSystemBase:
    dt (property) → float
        Sampling period / time step
    
    step(x, u, k) → StateVector
        Single time step: x[k+1] = f(x[k], u[k], k)
    
    simulate(x0, u_sequence, n_steps) → DiscreteSimulationResult
        Multi-step open/closed-loop simulation
    
    rollout(x0, policy, n_steps) → DiscreteSimulationResult
        Closed-loop with state feedback policy
    
    linearize(x_eq, u_eq) → (Ad, Bd) or (Ad, Bd, Gd)
        Compute discrete Jacobian matrices

Usage Examples
-------------
>>> # Define a custom continuous system
>>> class MyODE(ContinuousSystemBase):
...     def __init__(self):
...         self.nx = 2
...         self.nu = 1
...     
...     def __call__(self, x, u=None, t=0.0):
...         return -x + (u if u is not None else 0.0)
...     
...     def integrate(self, x0, u, t_span, method="RK45", **kwargs):
...         # Use scipy.integrate.solve_ivp
...         result = solve_ivp(...)
...         return {
...             "t": result.t,
...             "y": result.y,
...             "success": result.success,
...             "nfev": result.nfev
...         }
...     
...     def linearize(self, x_eq, u_eq):
...         A = -np.eye(self.nx)
...         B = np.eye(self.nx, self.nu)
...         return (A, B)

>>> # Use the system
>>> system = MyODE()
>>> x0 = np.array([1.0, 0.0])
>>> 
>>> # Low-level: Access solver diagnostics
>>> int_result = system.integrate(x0, u=None, t_span=(0, 10))
>>> print(f"Solver used {int_result['nfev']} evaluations")
>>> 
>>> # High-level: Get clean trajectory
>>> sim_result = system.simulate(x0, controller=None, t_span=(0, 10), dt=0.01)
>>> plt.plot(sim_result["time"], sim_result["states"][0, :])

>>> # Define a custom discrete system
>>> class MyDiscreteSystem(DiscreteSystemBase):
...     def __init__(self, dt=0.1):
...         self._dt = dt
...         self.nx = 2
...         self.nu = 1
...     
...     @property
...     def dt(self):
...         return self._dt
...     
...     def step(self, x, u=None, k=0):
...         u = u if u is not None else np.zeros(self.nu)
...         return 0.9 * x + 0.1 * u
...     
...     def simulate(self, x0, u_sequence, n_steps):
...         # Implement multi-step simulation
...         ...
...     
...     def linearize(self, x_eq, u_eq):
...         Ad = 0.9 * np.eye(self.nx)
...         Bd = 0.1 * np.eye(self.nx, self.nu)
...         return (Ad, Bd)

>>> # Use the discrete system
>>> discrete_sys = MyDiscreteSystem(dt=0.1)
>>> 
>>> # Open-loop simulation
>>> result = discrete_sys.simulate(x0, u_sequence=u_seq, n_steps=100)
>>> 
>>> # Closed-loop with state feedback
>>> K = np.array([[-1.0, -2.0]])
>>> def policy(x, k):
...     return -K @ x
>>> result = discrete_sys.rollout(x0, policy, n_steps=100)

Integration with Phase 2
------------------------
After Phase 1, the existing symbolic system classes will be refactored:

Phase 2.1: SymbolicDynamicalSystem → ContinuousSymbolicSystem
    - Will inherit from ContinuousSystemBase
    - Implements abstract methods using existing functionality
    - Adds backward compatibility alias

Phase 2.2: StochasticDynamicalSystem → ContinuousStochasticSystem
    - Will inherit from ContinuousSymbolicSystem
    - Extends linearize() to return (A, B, G) tuple
    - Adds backward compatibility alias

Phase 3: Discrete systems will be updated similarly
    - DiscreteSymbolicSystem inherits from DiscreteSystemBase
    - DiscreteStochasticSystem uses multiple inheritance

Migration Notes
--------------
For users of the library:
- Old class names will remain as aliases for 2+ versions
- New code should use ContinuousSymbolicSystem, ContinuousStochasticSystem
- linearize() returns tuples (always has in this codebase)
- simulate() returns dicts (TypedDict annotations, not instances)

For developers:
- Follow the abstract base class contracts
- Use type hints from src/types throughout
- Return IntegrationResult from integrate(), SimulationResult from simulate()
- Return tuples from linearize(), not dicts or objects

See Also
--------
- src/types/trajectories: Result type definitions
- src/types/linearization: Linearization type definitions
- src/types/core: Core vector and matrix types
- tests/unit/core_class_unit_tests/: Unit tests for base classes

References
---------
Phase 1 Implementation:
    Date: December 26, 2025
    Conversation: Extended design discussion on type system semantics
    Key Decisions:
        - Three-tier result types (Integration/Simulation/DiscreteSim)
        - integrate() vs simulate() semantic distinction
        - Tuples for linearization (not dicts or dataclasses)
        - TypedDict returns plain dicts (not instances)

Authors
-------
Gil Benezer

License
-------
GNU Affero General Public License v3.0
"""

# Import base classes
from .symbolic_system_base import SymbolicSystemBase
from .continuous_system_base import ContinuousSystemBase
from .discrete_system_base import DiscreteSystemBase
from .continuous_symbolic_system import ContinuousSymbolicSystem
from .discrete_symbolic_system import DiscreteSymbolicSystem

# Export public API
__all__ = [
    # Layer 1: Abstract base classes
    "SymbolicSystemBase",
    "ContinuousSystemBase",
    "DiscreteSystemBase",
    # Layer 2: Deterministic Systems
    "ContinuousSymbolicSystem",
    "DiscreteSymbolicSystem",
    # Layer 3: Stochastic Systems
    # Layer 4: Discretized Systems
]

# Version tracking for this module
__version__ = "2.0.0"
__phase__ = "Phase 2: Deterministic System Classes"
__last_updated__ = "2025-12-30"

# Module-level documentation
__doc_title__ = "Core System Classes"
__doc_summary__ = "Interfaces for continuous and discrete dynamical systems"