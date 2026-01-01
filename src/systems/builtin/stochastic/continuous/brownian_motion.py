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
Brownian Motion - Pure Diffusion Stochastic System
===================================================

Standard Brownian motion (Wiener process) is the fundamental stochastic
process underlying all SDEs. It represents pure random walk with no drift.

Mathematical Form
-----------------
Continuous-time SDE:
    dx = σ*dW

where:
    - σ > 0: Diffusion coefficient (volatility)
    - W(t): Standard Wiener process
    - No drift term (f = 0)

Multi-dimensional version:
    dx₁ = σ₁*dW₁
    dx₂ = σ₂*dW₂
    ...

Properties
----------
- **Zero Drift**: No deterministic evolution (f = 0)
- **Additive Noise**: σ is constant (state-independent)
- **Gaussian**: X(t) ~ N(x₀, σ²*t) for scalar case
- **Independent Increments**: X(t+s) - X(t) independent of X(t)
- **Continuous Paths**: Sample paths are continuous
- **Nowhere Differentiable**: Paths are fractal (not smooth)
- **Self-Similar**: Statistical properties scale with time
- **Markov Property**: Future independent of past given present

Statistical Properties
----------------------
For initial condition x(0) = x₀:

Mean:
    E[X(t)] = x₀ (constant - no drift)

Variance:
    Var[X(t)] = σ²*t (linear growth)

Standard Deviation:
    Std[X(t)] = σ*sqrt(t)

Covariance:
    Cov[X(s), X(t)] = σ²*min(s,t)

Distribution:
    X(t) ~ N(x₀, σ²*t)

Increments:
    X(t+Δt) - X(t) ~ N(0, σ²*Δt)

Applications
------------
- **Physics**: Particle diffusion, random walk
- **Finance**: Building block for option pricing
- **Signal Processing**: White noise integration
- **Biology**: Random molecular motion
- **Mathematics**: Foundation of stochastic calculus

Examples
--------
>>> # Standard Brownian motion
>>> bm = BrownianMotion(sigma=1.0)
>>>
>>> # Check properties
>>> bm.is_additive_noise()
True
>>> bm.is_pure_diffusion()
True
>>>
>>> # Evaluate (drift is zero)
>>> x = np.array([5.0])
>>> f = bm.drift(x)  # Returns [0.0]
>>> g = bm.diffusion(x)  # Returns [[1.0]]
>>>
>>> # Precompute constant noise
>>> G = bm.get_constant_noise('numpy')  # [[1.0]]
>>>
>>> # Half volatility
>>> bm_half = BrownianMotion(sigma=0.5)
>>> # Variance at t=1: 0.25 (vs 1.0 for standard)
>>>
>>> # 2D Brownian motion
>>> bm_2d = BrownianMotion2D(sigma1=1.0, sigma2=0.5)
>>> # Independent Brownian motions in each dimension
"""

from typing import Optional

import numpy as np
import sympy as sp

from src.systems.base.stochastic_dynamical_system import StochasticDynamicalSystem


class BrownianMotion(StochasticDynamicalSystem):
    """
    Standard Brownian motion (Wiener process) - pure diffusion.

    Stochastic differential equation:
        dx = σ*dW

    No drift term, no control. Pure random walk.

    Parameters
    ----------
    sigma : float, default=1.0
        Diffusion coefficient (volatility)
        Standard Brownian motion: σ = 1
        Variance at time t: Var[X(t)] = σ²*t

    Attributes
    ----------
    nx : int
        Always 1 (scalar state)
    nu : int
        0 (autonomous - no control)
    nw : int
        1 (single Wiener process)

    Examples
    --------
    >>> # Standard Brownian motion (σ=1)
    >>> bm = BrownianMotion(sigma=1.0)
    >>>
    >>> # Scaled Brownian motion
    >>> bm_scaled = BrownianMotion(sigma=0.5)
    >>> # Variance grows slower: Var[X(t)] = 0.25*t
    >>>
    >>> # Evaluate (no state/control dependence)
    >>> x = np.array([100.0])  # Value doesn't matter
    >>> f = bm.drift(x)  # [0.0]
    >>> g = bm.diffusion(x)  # [[1.0]]
    """

    def define_system(self, sigma: float = 1.0):
        """
        Define standard Brownian motion.

        Parameters
        ----------
        sigma : float
            Diffusion coefficient (must be positive)

        Notes
        -----
        **Special Properties:**
        - This is a pure diffusion process (zero drift)
        - It is autonomous (no control input)
        - The diffusion is additive (constant)
        - The state can take any real value

        **Standard Brownian Motion:**
        When σ = 1, this is the standard Wiener process W(t).
        All other Brownian motions can be written as σ*W(t).

        **Discretization:**
        Discrete version: X[k+1] = X[k] + σ*sqrt(dt)*w[k]
        where w[k] ~ N(0,1)
        """
        if sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}")

        # Define symbolic variables
        x = sp.symbols("x", real=True)
        sigma_sym = sp.symbols("sigma", positive=True)

        # System definition
        self.state_vars = [x]
        self.control_vars = []  # Autonomous (no control)

        # Drift: f(x) = 0 (zero drift!)
        self._f_sym = sp.Matrix([[0]])

        self.parameters = {sigma_sym: sigma}
        self.order = 1

        # Diffusion: g(x) = σ (constant - additive noise)
        self.diffusion_expr = sp.Matrix([[sigma_sym]])
        self.sde_type = "ito"


class BrownianMotion2D(StochasticDynamicalSystem):
    """
    Two-dimensional Brownian motion with independent components.

    Stochastic differential equation:
        dx₁ = σ₁*dW₁
        dx₂ = σ₂*dW₂

    where W₁ and W₂ are independent Wiener processes.

    Parameters
    ----------
    sigma1 : float, default=1.0
        Diffusion coefficient for first dimension
    sigma2 : float, default=1.0
        Diffusion coefficient for second dimension

    Examples
    --------
    >>> # Isotropic 2D Brownian motion
    >>> bm_2d = BrownianMotion2D(sigma1=1.0, sigma2=1.0)
    >>>
    >>> # Anisotropic (different rates in each direction)
    >>> bm_2d_aniso = BrownianMotion2D(sigma1=1.0, sigma2=0.5)
    >>> # Diffuses twice as fast in x₁ direction
    """

    def define_system(self, sigma1: float = 1.0, sigma2: float = 1.0):
        """Define 2D Brownian motion."""
        if sigma1 <= 0 or sigma2 <= 0:
            raise ValueError(f"sigma values must be positive")

        # Define symbolic variables
        x1, x2 = sp.symbols("x1 x2", real=True)
        sigma1_sym = sp.symbols("sigma1", positive=True)
        sigma2_sym = sp.symbols("sigma2", positive=True)

        # System definition
        self.state_vars = [x1, x2]
        self.control_vars = []  # Autonomous

        # Zero drift
        self._f_sym = sp.Matrix([[0], [0]])

        self.parameters = {sigma1_sym: sigma1, sigma2_sym: sigma2}
        self.order = 1

        # Diagonal diffusion (independent noise sources)
        self.diffusion_expr = sp.Matrix([[sigma1_sym, 0], [0, sigma2_sym]])
        self.sde_type = "ito"


class BrownianBridge(StochasticDynamicalSystem):
    """
    Brownian bridge - Brownian motion conditioned on endpoints.

    A Brownian bridge is a Brownian motion that starts at a and
    ends at b at time T. Useful for interpolation and finance.

    SDE form (non-autonomous, time-dependent):
        dx = -(x-b)/(T-t)*dt + σ*dW

    Note: This requires time-dependent drift, which needs special handling.
    For now, this is a placeholder/future extension.
    """

    def define_system(self, sigma: float = 1.0, T: float = 1.0, b: float = 0.0):
        """
        Define Brownian bridge.

        Note: Full implementation requires time-dependent drift.
        Current implementation is simplified.
        """
        import warnings

        warnings.warn(
            "BrownianBridge with time-dependent drift not fully supported yet. "
            "Use BrownianMotion for now.",
            UserWarning,
        )

        # Simplified version (treat as standard Brownian for now)
        x = sp.symbols("x", real=True)
        sigma_sym = sp.symbols("sigma", positive=True)

        self.state_vars = [x]
        self.control_vars = []
        self._f_sym = sp.Matrix([[0]])  # Simplified - should be time-dependent
        self.parameters = {sigma_sym: sigma}
        self.order = 1

        self.diffusion_expr = sp.Matrix([[sigma_sym]])
        self.sde_type = "ito"


# ============================================================================
# Convenience Factory Functions
# ============================================================================


def create_standard_brownian_motion() -> BrownianMotion:
    """
    Create standard Wiener process W(t).

    Properties: W(0) = 0, W(t) ~ N(0, t)

    Returns
    -------
    BrownianMotion
        Standard Brownian motion with σ = 1

    Examples
    --------
    >>> W = create_standard_brownian_motion()
    >>> # W(t) ~ N(0, t)
    """
    return BrownianMotion(sigma=1.0)


def create_scaled_brownian_motion(scale: float) -> BrownianMotion:
    """
    Create scaled Brownian motion: σ*W(t).

    Parameters
    ----------
    scale : float
        Scaling factor σ

    Returns
    -------
    BrownianMotion
        Scaled Brownian motion

    Examples
    --------
    >>> # Half-scale Brownian
    >>> bm_half = create_scaled_brownian_motion(0.5)
    >>> # Variance at t=1: 0.25
    """
    return BrownianMotion(sigma=scale)
