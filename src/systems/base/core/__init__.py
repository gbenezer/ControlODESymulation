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
Core System Classes
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
    - DiscretizedSystem

Key Design Principles
--------------------
1. **Clear Separation**: Continuous and discrete domains are completely separated
2. **Type Safety**: Full type hints using src/types throughout
3. **Minimal Interface**: Only essential methods are abstract
4. **Semantic Clarity**: Method names reflect intent (integrate vs simulate)
5. **Multi-Backend**: Works with NumPy, PyTorch, JAX

Method Interfaces
----------------

Layer 1 - Abstract Base Classes:

    SymbolicSystemBase:
        define_system(*args, **kwargs)
            Define symbolic dynamics (override in subclass)
        __repr__ and __str__
            Define string representations for debugging
        print_equations
            Display symbolic equations
        compile
            Pre-compile dynamics functions for specified backends
        reset_caches
            Reset cached compiled functions for specified backends
        nx, nu, ny, nq (properties)
            State, control, output, and noise dimensions
        set_default_backend
            Set default computation backend
        to_device
            Switches the device the system is on (PyTorch/JAX only)
        get_backend_info
            Get info on backend configuration
        use_backend
            Temporarily use a different backend without changing default
        get_performance_stats
            Get performance statistics for system operations
        reset_performance_stats
            Reset all performance counters to zero
        substitute_parameters
            Substitute numerical parameter values into symbolic expression
        setup_equilibria
            Optional hook to add equilibria after system initialization (override in subclass)
        add_equilibrium(name, x_eq, u_eq, ...)
            Register named equilibrium point
        set_default_equilibrium
            Set default equilibrium for get operations without name
        get_equilibrium
            Get equilibrium state and control in specified backend
        list_equilibria
            List all equilibrium names
        get_equilibrium_metadata
            Get metadata for equilibrium
        remove_equilibrium
            Remove an equilibrium point
        get_config_dict
            Get system configuration as dictionary
        save_config
            Save system configuration to JSON file

    ContinuousSystemBase:
        __call__(x, u, t) → StateVector
            Evaluate dynamics: dx/dt = f(x, u, t)
        integrate(x0, u, t_span, method) → IntegrationResult
            Low-level ODE solver with diagnostics (adaptive steps)
        simulate(x0, controller, t_span, dt) → SimulationResult
            High-level trajectory on regular time grid
        rollout(x0, policy, t_span, dt) → SimulationResult
            Closed-loop simulation with state feedback policy
        linearize(x_eq, u_eq) → (A, B)
            Compute Jacobian matrices at equilibrium
        control, analysis, plotter, phase_plotter, control_plotter (properties)
            Access control synthesis, analysis, and plotting interfaces
        plot
            alias for plotter.plot_trajectory
        is_continuous, is_discrete, is_stochastic, is_time_varying (properties)
            Boolean properties for downstream applications based on system characteristics

    DiscreteSystemBase:
        dt (property) → float
            Sampling period / time step
        sampling_frequency (property)
            Get the sampling frequency in Hertz
        step(x, u, k) → StateVector
            Single time step: x[k+1] = f(x[k], u[k], k)
        simulate(x0, u_sequence, n_steps) → DiscreteSimulationResult
            Multi-step open/closed-loop simulation
        rollout(x0, policy, n_steps) → DiscreteSimulationResult
            Closed-loop with state feedback policy
        linearize(x_eq, u_eq) → (Ad, Bd)
            Compute discrete Jacobian matrices
        control, analysis, plotter, phase_plotter, control_plotter (properties)
            Access control synthesis, analysis, and plotting interfaces
        plot
            alias for plotter.plot_trajectory
        is_continuous, is_discrete, is_stochastic, is_time_varying (properties)
            Boolean properties for downstream applications based on system characteristics

Layer 2 - Deterministic Symbolic Systems:

    ContinuousSymbolicSystem(SymbolicSystemBase, ContinuousSystemBase):
        forward(x, u, t, backend) → StateVector
            Evaluate dynamics with explicit backend selection
        linearized_dynamics_symbolic
            Compute symbolic linearization: A = ∂f/∂x, B = ∂f/∂u.
        linearized_dynamics(x, u, A, B, backend) → StateVector
            Evaluate linearized dynamics: dx/dt = A(x - x_eq) + B(u - u_eq)
        verify_jacobians
            Verify symbolic Jacobians against automatic differentiation.
        h(x, backend) → OutputVector
            Evaluate output function y = h(x)
        linearized_observation_symbolic
            Compute symbolic observation Jacobian: C = ∂h/∂x
        linearized_observation(x, x_eq, C, backend) → OutputVector
            Evaluate linearized output: y = C(x - x_eq)
        warmup(backends, n_calls)
            Pre-compile and warm up JIT caches

    DiscreteSymbolicSystem(SymbolicSystemBase, DiscreteSystemBase):
        forward(x, u, k, backend) → StateVector
            Evaluate dynamics with explicit backend selection
        linearized_dynamics_symbolic
            Compute symbolic discrete linearization
        linearized_dynamics(x, u, Ad, Bd, backend) → StateVector
            Evaluate linearized dynamics: x[k+1] = Ad(x - x_eq) + Bd(u - u_eq)
        h(x, backend) → OutputVector
            Evaluate output function y = h(x)
        linearized_observation_symbolic
            Compute symbolic observation Jacobian: C = ∂h/∂x
        linearized_observation(x, x_eq, C, backend) → OutputVector
            Evaluate linearized output: y = C(x - x_eq)
        warmup(backends, n_calls)
            Pre-compile and warm up JIT caches

Layer 3 - Stochastic Systems:

    ContinuousStochasticSystem(ContinuousSymbolicSystem):
        drift(x, u, t, backend) → StateVector
            Evaluate deterministic drift: f(x, u, t)
        diffusion(x, u, t, backend) → DiffusionMatrix
            Evaluate noise diffusion: G(x, u, t)
        linearize(x_eq, u_eq) → (A, B, G)
            Returns Jacobians plus diffusion matrix
        recommend_solvers
            Recommend efficient SDE solvers based on noise structure
        is_additive_noise, is_multiplicative_noise, is_diagonal_noise, is_scalar_noise (properties)
            Noise structure classification
        is_pure_diffusion
            Whether or not the system is a pure diffusion (no drift) system
        get_noise_type() → NoiseType
            Get noise type (additive, multiplicative, etc.)
        get_sde_type() → SDEType
            Get SDE type (Ito, Stratonovich)
        get_diffusion_matrix
            Get the diffusion matrix
        depends_on_state, depends_on_control, depends_on_time
            Check if diffusion depends on particular variables
        compile_diffusion, compile_all, reset_diffusion_cache, reset_all_caches
            Companion methods to pre-compile diffusion functions or both functions and reset them

    DiscreteStochasticSystem(DiscreteSymbolicSystem):
        diffusion(x, u, k, backend) → DiffusionMatrix
            Evaluate noise diffusion: G(x, u, k)
        step_stochastic(x, u, k, noise, backend) → StateVector
            Single stochastic step with explicit noise
        simulate_stochastic(x0, u_sequence, n_steps, ...) → DiscreteSimulationResult
            Monte Carlo simulation with noise realizations
        linearize(x_eq, u_eq) → (Ad, Bd, Gd)
            Returns discrete Jacobians plus diffusion matrix
        is_additive_noise, is_multiplicative_noise, is_diagonal_noise, is_scalar_noise (properties)
            Noise structure classification
        is_pure_diffusion
            Whether or not the system is a pure diffusion (no drift) system
        depends_on_state, depends_on_control, depends_on_time
            Check if diffusion depends on particular variables
        compile_diffusion, compile_all, reset_diffusion_cache, reset_all_caches
            Companion methods to pre-compile diffusion functions or both functions and reset them

Layer 4 - Discretized Systems (Bridge):

    DiscretizedSystem(DiscreteSystemBase):
        dt, mode (properties)
            Sampling period and discretization mode
        get_available_methods
            Get available integration methods for a backend.
        step(x, u, k) → StateVector
            Discretized step using configured method
        simulate(x0, u_sequence, n_steps) → DiscreteSimulationResult
            Simulate using fixed or dense output mode
        simulate_stochastic(x0, u_sequence, n_steps, ...) → DiscreteSimulationResult
            Stochastic simulation (for wrapped stochastic systems)
        linearize(x_eq, u_eq) → (Ad, Bd) or (Ad, Bd, Gd)
            Discrete linearization of continuous system
        compare_modes(x0, u_sequence, n_steps) → dict
            Compare FIXED vs DENSE discretization accuracy
        change_method(new_method) → DiscretizedSystem
            Return copy with different integration method
        get_info() → dict
            Get discretization configuration details

    Helper Functions:
        discretize(system, dt, method, mode) → DiscretizedSystem
            Create discretized wrapper for continuous system
        discretize_batch(systems, dt, method) → List[DiscretizedSystem]
            Batch discretization of multiple systems
        analyze_discretization_error(system, dt_values, ...) → dict
            Analyze discretization error across time steps
        recommend_dt(system, x0, u, ...) → float
            Recommend appropriate time step
        compute_discretization_quality(system, dt, ...) → dict
            Compute quality metrics for given dt

Authors
-------
Gil Benezer

License
-------
GNU Affero General Public License v3.0
"""

# Layer 3: Stochastic systems
from .continuous_stochastic_system import ContinuousStochasticSystem, StochasticDynamicalSystem

# Layer 2: Deterministic symbolic systems
from .continuous_symbolic_system import (
    ContinuousDynamicalSystem,
    ContinuousSymbolicSystem,
    SymbolicDynamicalSystem,
)

# Layer 1: Abstract base classes
from .continuous_system_base import ContinuousSystemBase
from .discrete_stochastic_system import DiscreteStochasticSystem
from .discrete_symbolic_system import DiscreteDynamicalSystem, DiscreteSymbolicSystem
from .discrete_system_base import DiscreteSystemBase

# Layer 4: Discretized systems (bridge)
from .discretized_system import (
    DiscretizationMode,
    DiscretizedSystem,
    analyze_discretization_error,
    compute_discretization_quality,
    discretize,
    discretize_batch,
    recommend_dt,
)
from .symbolic_system_base import SymbolicSystemBase

# Export public API
__all__ = [
    # Layer 1: Abstract base classes
    "SymbolicSystemBase",
    "ContinuousSystemBase",
    "DiscreteSystemBase",
    # Layer 2: Deterministic symbolic systems and aliases
    "ContinuousSymbolicSystem",
    "ContinuousDynamicalSystem",
    "SymbolicDynamicalSystem",
    "DiscreteDynamicalSystem",
    "DiscreteSymbolicSystem",
    # Layer 3: Stochastic systems and aliases
    "ContinuousStochasticSystem",
    "StochasticDynamicalSystem",
    "DiscreteStochasticSystem",
    # Layer 4: Discretized systems
    "DiscretizationMode",
    "DiscretizedSystem",
    "discretize",
    "discretize_batch",
    "analyze_discretization_error",
    "recommend_dt",
    "compute_discretization_quality",
]

# Version tracking for this module
__version__ = "1.0.0"
__phase__ = "Phase 4: Complete Core System Classes"
__last_updated__ = "2026-01-03"

# Module-level documentation
__doc_title__ = "Core System Classes"
__doc_summary__ = "Interfaces for continuous and discrete dynamical systems"
