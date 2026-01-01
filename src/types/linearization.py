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
Linearization Types

Defines types for system linearization results and related operations:
- Jacobian matrices (A, B, C, D, G)
- Linearization result tuples
- Deterministic vs stochastic linearization
- Continuous vs discrete linearization

Linearization provides local linear approximation of nonlinear systems,
enabling:
- Local stability analysis (eigenvalues)
- Linear controller design (LQR, pole placement)
- Observer design (Kalman filter)
- Small-signal analysis

Mathematical Forms
------------------
Continuous Deterministic:
    dx/dt = f(x, u) ≈ Ac*δx + Bc*δu
    y = h(x) ≈ Cc*δx + Dc*δu

Continuous Stochastic:
    dx = f(x,u)dt + g(x,u)dW ≈ (Ac*δx + Bc*δu)dt + Gc*dW

Discrete Deterministic:
    x[k+1] = f(x,u) ≈ Ad*δx + Bd*δu
    y[k] = h(x) ≈ Cd*δx + Dd*δu

Discrete Stochastic:
    x[k+1] = f(x,u) + g(x,u)*w ≈ Ad*δx + Bd*δu + Gd*w

Where:
    δx = x - x_eq (state deviation)
    δu = u - u_eq (control deviation)
    Ac/Ad = ∂f/∂x (state Jacobian)
    Bc/Bd = ∂f/∂u (control Jacobian)
    Cc/Cd = ∂h/∂x (output Jacobian)
    Gc/Gd = ∂g/∂x or g(x_eq) (diffusion)

Usage
-----
>>> from src.types.linearization import (
...     LinearizationResult,
...     DeterministicLinearization,
...     StochasticLinearization,
... )
>>>
>>> # Deterministic
>>> result: DeterministicLinearization = system.linearize(x_eq, u_eq)
>>> Ac, Bc = result
>>>
>>> # Stochastic
>>> result: StochasticLinearization = sde_system.linearize(x_eq, u_eq)
>>> Ac, Bc, Gc = result
>>>
>>> # Polymorphic (handles both)
>>> result: LinearizationResult = system.linearize(x_eq, u_eq)
>>> A, B = result[0], result[1]
>>> if len(result) == 3:
...     G = result[2]  # Stochastic
"""

from typing import Tuple, Union

from .core import (
    DiffusionMatrix,
    FeedthroughMatrix,
    InputMatrix,
    OutputMatrix,
    StateMatrix,
)

# ============================================================================
# Linearization Result Types
# ============================================================================

DeterministicLinearization = Tuple[StateMatrix, InputMatrix]
"""
Linearization result for deterministic systems: (A, B).

Returns Jacobian matrices of dynamics with respect to state and control.

Continuous systems:
    dx/dt = f(x, u) linearized to: dx/dt ≈ Ac*δx + Bc*δu
    Returns: (Ac, Bc) where
        Ac = ∂f/∂x at (x_eq, u_eq)  (nx, nx)
        Bc = ∂f/∂u at (x_eq, u_eq)  (nx, nu)

Discrete systems:
    x[k+1] = f(x, u) linearized to: x[k+1] ≈ Ad*(x[k]-x_eq) + Bd*(u[k]-u_eq) + x_eq
    Returns: (Ad, Bd) where
        Ad = ∂f/∂x at (x_eq, u_eq)  (nx, nx)
        Bd = ∂f/∂u at (x_eq, u_eq)  (nx, nu)

Usage:
    Ac, Bc = continuous_system.linearize(x_eq, u_eq)
    Ad, Bd = discrete_system.linearize(x_eq, u_eq)

Examples
--------
>>> # Continuous system linearization
>>> Ac, Bc = continuous_system.linearize(
...     x_eq=np.zeros(2),
...     u_eq=np.zeros(1)
... )
>>> print(Ac.shape)  # (2, 2)
>>> print(Bc.shape)  # (2, 1)
>>> 
>>> # Use for LQR design
>>> from scipy.linalg import solve_continuous_are
>>> P = solve_continuous_are(Ac, Bc, Q, R)
>>> K = np.linalg.inv(R) @ Bc.T @ P
>>> 
>>> # Discrete system linearization
>>> Ad, Bd = discrete_system.linearize(
...     x_eq=np.zeros(2),
...     u_eq=np.zeros(1)
... )
>>> 
>>> # Check stability
>>> eigenvalues = np.linalg.eigvals(Ad)
>>> is_stable = np.all(np.abs(eigenvalues) < 1.0)
"""

StochasticLinearization = Tuple[StateMatrix, InputMatrix, DiffusionMatrix]
"""
Linearization result for stochastic systems: (A, B, G).

Returns Jacobians of drift and diffusion terms.

Continuous SDE:
    dx = f(x,u)dt + g(x,u)dW linearized to:
    dx ≈ (Ac*δx + Bc*δu)dt + Gc*dW
    
    Returns: (Ac, Bc, Gc) where
        Ac = ∂f/∂x at (x_eq, u_eq)  (nx, nx)
        Bc = ∂f/∂u at (x_eq, u_eq)  (nx, nu)
        Gc = ∂g/∂x at (x_eq, u_eq) or g(x_eq) if additive  (nx, nw)

Discrete stochastic:
    x[k+1] = f(x,u) + g(x,u)*w[k] linearized to:
    x[k+1] ≈ Ad*δx + Bd*δu + Gd*w[k]
    
    Returns: (Ad, Bd, Gd) where
        Ad = ∂f/∂x  (nx, nx)
        Bd = ∂f/∂u  (nx, nu)
        Gd = g(x_eq) or ∂g/∂x  (nx, nw)

Usage:
    Ac, Bc, Gc = sde_system.linearize(x_eq, u_eq)
    Ad, Bd, Gd = discrete_stochastic.linearize(x_eq, u_eq)

Examples
--------
>>> # Continuous SDE linearization
>>> Ac, Bc, Gc = sde_system.linearize(
...     x_eq=np.zeros(2),
...     u_eq=np.zeros(1)
... )
>>> print(Ac.shape)  # (2, 2)
>>> print(Bc.shape)  # (2, 1)
>>> print(Gc.shape)  # (2, nw)
>>> 
>>> # Process noise covariance
>>> Q = Gc @ Gc.T
>>> 
>>> # Discrete stochastic linearization
>>> Ad, Bd, Gd = discrete_stochastic.linearize(
...     x_eq=np.zeros(2),
...     u_eq=np.zeros(1)
... )
>>> 
>>> # Design LQG controller
>>> # 1. LQR for control
>>> P_control = solve_discrete_are(Ad, Bd, Q_cost, R_cost)
>>> K_lqr = np.linalg.inv(R_cost + Bd.T @ P_control @ Bd) @ (Bd.T @ P_control @ Ad)
>>> 
>>> # 2. Kalman for estimation
>>> Q_noise = Gd @ Gd.T
>>> P_estimate = solve_discrete_are(Ad.T, Cd.T, Q_noise, R_noise)
>>> L_kalman = Ad @ P_estimate @ Cd.T @ np.linalg.inv(Cd @ P_estimate @ Cd.T + R_noise)
"""

LinearizationResult = Union[DeterministicLinearization, StochasticLinearization]
"""
Flexible linearization result type.

Can be either:
- (A, B) for deterministic systems
- (A, B, G) for stochastic systems

This union type enables polymorphic code that handles both deterministic
and stochastic systems with a single function signature.

Polymorphic Unpacking Pattern:
    result = system.linearize(x_eq, u_eq)
    A, B = result[0], result[1]  # Always available
    if len(result) == 3:
        G = result[2]  # Stochastic only

Examples
--------
>>> # Polymorphic function
>>> def analyze_linearization(
...     system,
...     x_eq: StateVector,
...     u_eq: ControlVector
... ) -> dict:
...     '''Works with deterministic AND stochastic systems.'''
...     result: LinearizationResult = system.linearize(x_eq, u_eq)
...     
...     A = result[0]  # State matrix (always present)
...     B = result[1]  # Control matrix (always present)
...     
...     info = {
...         'eigenvalues': np.linalg.eigvals(A),
...         'is_stochastic': len(result) == 3
...     }
...     
...     if len(result) == 3:
...         G = result[2]
...         info['process_noise_cov'] = G @ G.T
...     
...     return info
>>> 
>>> # Use with deterministic
>>> result_det = analyze_linearization(ode_system, x_eq, u_eq)
>>> 
>>> # Use with stochastic
>>> result_stoch = analyze_linearization(sde_system, x_eq, u_eq)
"""

ObservationLinearization = Tuple[OutputMatrix, FeedthroughMatrix]
"""
Observation/output linearization: (C, D).

Linearizes output equation y = h(x, u).

Result:
    y ≈ C*δx + D*δu + y_eq
    
    where:
        C = ∂h/∂x at (x_eq, u_eq)  (ny, nx)
        D = ∂h/∂u at (x_eq, u_eq)  (ny, nu)
        y_eq = h(x_eq, u_eq)

Typically D = 0 (no direct feedthrough).

Examples
--------
>>> # Linearize output
>>> C, D = system.linearized_observation(x_eq, u_eq)
>>> print(C.shape)  # (ny, nx)
>>> print(D.shape)  # (ny, nu)
>>> 
>>> # For Kalman filter
>>> # Prediction: ŷ = C @ x̂_pred
>>> # Innovation: v = y - ŷ
>>> # Update: x̂ = x̂_pred + L @ v
>>> 
>>> # Full state observation (common case)
>>> C_full = np.eye(nx)  # y = x
>>> D_full = np.zeros((nx, nu))  # No feedthrough
"""

ContinuousLinearization = DeterministicLinearization
"""
Alias for continuous-time deterministic linearization.

Same as DeterministicLinearization but emphasizes continuous-time context.
Returns (Ac, Bc) where c indicates continuous.

Examples
--------
>>> Ac, Bc = continuous_system.linearize(x_eq, u_eq)
>>> # Continuous-time stability: Re(λ) < 0
>>> eigenvalues = np.linalg.eigvals(Ac)
>>> is_stable = np.all(np.real(eigenvalues) < 0)
"""

DiscreteLinearization = DeterministicLinearization
"""
Alias for discrete-time deterministic linearization.

Same as DeterministicLinearization but emphasizes discrete-time context.
Returns (Ad, Bd) where d indicates discrete.

Examples
--------
>>> Ad, Bd = discrete_system.linearize(x_eq, u_eq)
>>> # Discrete-time stability: |λ| < 1
>>> eigenvalues = np.linalg.eigvals(Ad)
>>> is_stable = np.all(np.abs(eigenvalues) < 1.0)
"""

ContinuousStochasticLinearization = StochasticLinearization
"""
Alias for continuous-time stochastic linearization.

Returns (Ac, Bc, Gc) for continuous SDEs.

Examples
--------
>>> Ac, Bc, Gc = sde_system.linearize(x_eq, u_eq)
>>> # Drift stability
>>> drift_stable = np.all(np.real(np.linalg.eigvals(Ac)) < 0)
"""

DiscreteStochasticLinearization = StochasticLinearization
"""
Alias for discrete-time stochastic linearization.

Returns (Ad, Bd, Gd) for discrete stochastic systems.

Examples
--------
>>> Ad, Bd, Gd = discrete_stochastic.linearize(x_eq, u_eq)
>>> # Mean-square stability (drift only)
>>> ms_stable = np.all(np.abs(np.linalg.eigvals(Ad)) < 1.0)
"""


# ============================================================================
# Linearization with Output
# ============================================================================

FullLinearization = Tuple[StateMatrix, InputMatrix, OutputMatrix, FeedthroughMatrix]
"""
Complete linearization including output: (A, B, C, D).

Linearizes both dynamics and output equations.

State space form:
    Continuous: dx/dt = Ac*x + Bc*u, y = Cc*x + Dc*u
    Discrete:   x[k+1] = Ad*x + Bd*u, y = Cd*x + Dd*u

Typically:
    - C is identity for full state observation
    - D is zero for no direct feedthrough

Examples
--------
>>> A, B, C, D = system.full_linearization(x_eq, u_eq)
>>> 
>>> # State space model
>>> ss_continuous = scipy.signal.StateSpace(A, B, C, D)
>>> ss_discrete = scipy.signal.dlti(A, B, C, D, dt=0.01)
>>> 
>>> # Transfer function
>>> G_s = C @ np.linalg.inv(s*I - A) @ B + D
"""

FullStochasticLinearization = Tuple[
    StateMatrix, InputMatrix, DiffusionMatrix, OutputMatrix, FeedthroughMatrix,
]
"""
Complete stochastic linearization: (A, B, G, C, D).

Includes both dynamics and output linearization.

Examples
--------
>>> A, B, G, C, D = sde_system.full_linearization(x_eq, u_eq)
>>> 
>>> # LQG design
>>> K_lqr = design_lqr(A, B, Q_control, R_control)
>>> L_kalman = design_kalman(A, C, G@G.T, R_meas)
"""


# ============================================================================
# Jacobian-Specific Types
# ============================================================================

StateJacobian = StateMatrix
"""
State Jacobian ∂f/∂x.

Matrix of partial derivatives of dynamics with respect to state.

Continuous: Ac = ∂f/∂x where f is drift
Discrete: Ad = ∂f/∂x where f is next-state function

Shape: (nx, nx)

Examples
--------
>>> Ac: StateJacobian = system.state_jacobian(x_eq, u_eq)
>>> # For pendulum: Ac = [[0, 1], [-g/L*cos(θ), -b]]
"""

ControlJacobian = InputMatrix
"""
Control Jacobian ∂f/∂u.

Matrix of partial derivatives of dynamics with respect to control.

Shape: (nx, nu)

Examples
--------
>>> Bc: ControlJacobian = system.control_jacobian(x_eq, u_eq)
>>> # Often constant: Bc = [[0], [1/m]]
"""

OutputJacobian = OutputMatrix
"""
Output Jacobian ∂h/∂x.

Matrix of partial derivatives of output with respect to state.

Shape: (ny, nx)

Examples
--------
>>> Cc: OutputJacobian = system.output_jacobian(x_eq)
>>> # Position measurement: Cc = [[1, 0]] (measure position, not velocity)
"""

DiffusionJacobian = DiffusionMatrix
"""
Diffusion Jacobian ∂g/∂x (stochastic systems).

For multiplicative noise: g depends on x, so linearize.
For additive noise: g is constant, Jacobian is just g itself.

Shape: (nx, nw)

Examples
--------
>>> Gc: DiffusionJacobian = sde_system.diffusion_jacobian(x_eq, u_eq)
>>> 
>>> # Additive noise (constant)
>>> Gc_additive = 0.1 * np.eye(nx)
>>> 
>>> # Multiplicative noise (state-dependent)
>>> # For dx = f(x)dt + σ*x*dW:
>>> Gc_multiplicative = σ * x_eq  # ∂(σ*x)/∂x evaluated at x_eq
"""


# ============================================================================
# Linearization Cache Types
# ============================================================================

LinearizationCacheKey = str
"""
Cache key for linearization results.

Typically includes equilibrium point and method.

Format: "x_eq=<hash>_u_eq=<hash>_method=<name>"

Examples
--------
>>> key: LinearizationCacheKey = "x_eq=abc123_u_eq=def456_method=euler"
>>> cache[key] = (Ad, Bd)
"""


# ============================================================================
# Export All
# ============================================================================

__all__ = [
    # Main result types
    "DeterministicLinearization",
    "StochasticLinearization",
    "LinearizationResult",
    "ObservationLinearization",
    # Time-domain specific aliases
    "ContinuousLinearization",
    "DiscreteLinearization",
    "ContinuousStochasticLinearization",
    "DiscreteStochasticLinearization",
    # Full linearization
    "FullLinearization",
    "FullStochasticLinearization",
    # Jacobian-specific
    "StateJacobian",
    "ControlJacobian",
    "OutputJacobian",
    "DiffusionJacobian",
    # Cache
    "LinearizationCacheKey",
]
