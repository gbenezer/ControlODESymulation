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
Ornstein-Uhlenbeck Process - Mean-Reverting Stochastic System
==============================================================

The Ornstein-Uhlenbeck (OU) process is a fundamental stochastic differential
equation exhibiting mean reversion. It's widely used in physics, finance,
and biology.

Mathematical Form
-----------------
Continuous-time SDE:
    dx = -α(x - μ)*dt + σ*dW

or in the centered form (μ=0):
    dx = -α*x*dt + σ*dW

where:
    - α > 0: Mean reversion rate (how fast it returns to mean)
    - σ > 0: Volatility (noise intensity)
    - μ: Long-term mean (set to 0 in this implementation)
    - W(t): Standard Wiener process (Brownian motion)

With control input u:
    dx = (-α*x + u)*dt + σ*dW

Properties
----------
- **Mean Reversion**: Process is pulled toward mean (0) at rate α
- **Additive Noise**: σ is constant (state-independent)
- **Stationary Distribution**: X ~ N(0, σ²/(2α)) as t → ∞
- **Autocorrelation**: Cov[X(t), X(t+s)] = (σ²/2α)*exp(-α*s)
- **Ergodic**: Time averages equal ensemble averages

Analytical Solution
-------------------
For initial condition x(0) = x₀ and constant control u:
    x(t) = x₀*exp(-αt) + (u/α)*(1 - exp(-αt)) + ∫₀ᵗ σ*exp(-α(t-s))*dW(s)

Mean:
    E[x(t)] = x₀*exp(-αt) + (u/α)*(1 - exp(-αt))

Variance:
    Var[x(t)] = (σ²/2α)*(1 - exp(-2αt))

As t → ∞:
    E[x(∞)] = u/α
    Var[x(∞)] = σ²/(2α)

Applications
------------
- **Physics**: Velocity of particle in viscous fluid (Langevin equation)
- **Finance**: Interest rate models (Vasicek model)
- **Neuroscience**: Neural membrane potential fluctuations
- **Control**: Benchmark for stochastic control algorithms

Examples
--------
>>> # Basic OU process
>>> system = OrnsteinUhlenbeck(alpha=2.0, sigma=0.5)
>>>
>>> # Check noise type
>>> system.is_additive_noise()
True
>>>
>>> # Evaluate drift and diffusion
>>> x = np.array([1.0])
>>> u = np.array([0.0])
>>> f = system.drift(x, u)  # Drift: -2*1 = -2
>>> g = system.diffusion(x, u)  # Diffusion: 0.5 (constant)
>>>
>>> # For additive noise, precompute constant matrix
>>> G = system.get_constant_noise('numpy')  # [[0.5]]
>>>
>>> # Get solver recommendations
>>> system.recommend_solvers('jax')
['sea', 'shark', 'sra1']  # Specialized for additive noise
>>>
>>> # Fast mean reversion (α large)
>>> fast_ou = OrnsteinUhlenbeck(alpha=10.0, sigma=0.5)
>>> # Returns to mean quickly, small variance σ²/(2α) = 0.0125
>>>
>>> # Slow mean reversion (α small)
>>> slow_ou = OrnsteinUhlenbeck(alpha=0.5, sigma=0.5)
>>> # Returns to mean slowly, large variance σ²/(2α) = 0.25
>>>
>>> # High volatility
>>> volatile_ou = OrnsteinUhlenbeck(alpha=2.0, sigma=2.0)
>>> # Fast fluctuations around mean
"""

import numpy as np
import sympy as sp

from src.systems.base.stochastic_dynamical_system import StochasticDynamicalSystem


class OrnsteinUhlenbeck(StochasticDynamicalSystem):
    """
    Ornstein-Uhlenbeck process with mean reversion and additive noise.

    Stochastic differential equation:
        dx = (-α*x + u)*dt + σ*dW

    where:
        - x ∈ ℝ: State (position, price, etc.)
        - u ∈ ℝ: Control input (forcing term)
        - α > 0: Mean reversion rate
        - σ > 0: Volatility (constant)
        - W(t): Standard Wiener process

    Parameters
    ----------
    alpha : float, default=1.0
        Mean reversion rate (larger = faster return to mean)
        Time constant: τ = 1/α
    sigma : float, default=1.0
        Volatility (noise intensity)
        Stationary std: σ/sqrt(2α)

    Attributes
    ----------
    nx : int
        Always 1 (scalar state)
    nu : int
        1 (controlled system)
    nw : int
        1 (single Wiener process)

    Examples
    --------
    >>> # Standard OU process
    >>> system = OrnsteinUhlenbeck(alpha=2.0, sigma=0.5)
    >>>
    >>> # Fast mean reversion
    >>> fast = OrnsteinUhlenbeck(alpha=5.0, sigma=0.5)
    >>> # τ = 0.2s, settles in ~1s
    >>>
    >>> # High volatility
    >>> volatile = OrnsteinUhlenbeck(alpha=1.0, sigma=2.0)
    >>> # Large fluctuations around mean
    """

    def define_system(self, alpha: float = 1.0, sigma: float = 1.0):
        """
        Define Ornstein-Uhlenbeck process.

        Parameters
        ----------
        alpha : float
            Mean reversion rate (must be positive for stability)
        sigma : float
            Volatility (must be positive)

        Notes
        -----
        **Stability:**
        - α > 0: Stable (mean-reverting)
        - α = 0: Brownian motion (no mean reversion)
        - α < 0: Unstable (explosive)

        **Time Scale:**
        - Time constant: τ = 1/α
        - Half-life: t_half = ln(2)/α ≈ 0.693/α
        - Settles to mean in ~4τ seconds

        **Stationary Statistics:**
        - Mean: E[X(∞)] = u/α
        - Variance: Var[X(∞)] = σ²/(2α)
        - Std: σ/sqrt(2α)
        """
        # Validate parameters
        if alpha <= 0:
            import warnings

            warnings.warn(
                f"alpha={alpha} ≤ 0 leads to unstable/non-reverting process. "
                f"Use alpha > 0 for mean reversion.",
                UserWarning,
            )

        if sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}")

        # Define symbolic variables
        x = sp.symbols("x", real=True)
        u = sp.symbols("u", real=True)

        # Define symbolic parameters
        alpha_sym = sp.symbols("alpha", positive=True)
        sigma_sym = sp.symbols("sigma", positive=True)

        # System definition
        self.state_vars = [x]
        self.control_vars = [u]

        # Drift: f(x, u) = -α*x + u
        self._f_sym = sp.Matrix([[-alpha_sym * x + u]])

        self.parameters = {alpha_sym: alpha, sigma_sym: sigma}
        self.order = 1

        # Diffusion: g(x, u) = σ (constant - additive noise)
        self.diffusion_expr = sp.Matrix([[sigma_sym]])
        self.sde_type = "ito"

    def get_stationary_std(self) -> float:
        """
        Get theoretical stationary standard deviation.

        Returns
        -------
        float
            Stationary std: σ/sqrt(2α)

        Examples
        --------
        >>> system = OrnsteinUhlenbeck(alpha=2.0, sigma=0.5)
        >>> system.get_stationary_std()
        0.25
        """
        # Extract parameter values
        alpha = None
        sigma = None

        for key, val in self.parameters.items():
            if str(key) == "alpha":
                alpha = val
            elif str(key) == "sigma":
                sigma = val

        return sigma / np.sqrt(2.0 * alpha)

    def get_time_constant(self) -> float:
        """
        Get mean reversion time constant τ = 1/α.

        Returns
        -------
        float
            Time constant in seconds

        Examples
        --------
        >>> system = OrnsteinUhlenbeck(alpha=2.0, sigma=0.5)
        >>> system.get_time_constant()
        0.5
        """
        for key, val in self.parameters.items():
            if str(key) == "alpha":
                return 1.0 / val

        raise RuntimeError("alpha parameter not found")


# ============================================================================
# Convenience Factory Functions
# ============================================================================


def create_ou_process(time_constant: float = 1.0, volatility: float = 1.0) -> OrnsteinUhlenbeck:
    """
    Create OU process with specified time constant and volatility.

    Parameters
    ----------
    time_constant : float
        Time constant τ in seconds (α = 1/τ)
    volatility : float
        Noise intensity σ

    Returns
    -------
    OrnsteinUhlenbeck
        OU process with α = 1/τ

    Examples
    --------
    >>> # Fast mean reversion (settles in ~2 seconds)
    >>> fast_ou = create_ou_process(time_constant=0.5, volatility=0.5)
    >>> # α = 2.0, τ = 0.5s
    """
    alpha = 1.0 / time_constant
    return OrnsteinUhlenbeck(alpha=alpha, sigma=volatility)


def create_vasicek_model(
    mean_reversion: float = 0.5, long_term_rate: float = 0.05, volatility: float = 0.01
) -> OrnsteinUhlenbeck:
    """
    Create Vasicek interest rate model (special case of OU).

    In finance: dr = κ(θ - r)*dt + σ*dW

    Parameters
    ----------
    mean_reversion : float
        Mean reversion speed κ
    long_term_rate : float
        Long-term mean interest rate θ
    volatility : float
        Interest rate volatility σ

    Returns
    -------
    OrnsteinUhlenbeck
        OU process configured for interest rate modeling

    Examples
    --------
    >>> # Vasicek model: mean=5%, volatility=1%, fast reversion
    >>> vasicek = create_vasicek_model(
    ...     mean_reversion=0.5,
    ...     long_term_rate=0.05,
    ...     volatility=0.01
    ... )

    Notes
    -----
    This returns the centered form. To match Vasicek exactly,
    add offset in control: u = κ*θ
    """
    return OrnsteinUhlenbeck(alpha=mean_reversion, sigma=volatility)
