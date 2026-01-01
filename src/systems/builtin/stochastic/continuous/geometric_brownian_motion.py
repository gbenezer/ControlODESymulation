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
Geometric Brownian Motion - Multiplicative Noise Stochastic System
===================================================================

Geometric Brownian Motion (GBM) is the canonical model for processes that
exhibit exponential growth with proportional noise. It's the foundation of
the Black-Scholes option pricing model.

Mathematical Form
-----------------
Continuous-time SDE:
    dx = μ*x*dt + σ*x*dW

where:
    - μ: Drift coefficient (expected return rate)
    - σ > 0: Volatility (proportional to state)
    - W(t): Standard Wiener process (Brownian motion)
    - x > 0: State (typically a price or population)

With control input u:
    dx = (μ*x + u)*dt + σ*x*dW

Properties
----------
- **Multiplicative Noise**: σ*x (state-dependent)
- **Log-Normal Distribution**: X(t) is log-normally distributed
- **Positive States**: If x(0) > 0, then x(t) > 0 for all t
- **Exponential Growth**: E[X(t)] = x₀*exp(μt) for u=0
- **Non-Stationary**: Variance grows without bound

Analytical Solution
-------------------
For constant u=0 and initial condition x(0) = x₀:
    x(t) = x₀ * exp((μ - σ²/2)*t + σ*W(t))

The solution is log-normal:
    ln(x(t)) ~ N(ln(x₀) + (μ - σ²/2)*t, σ²*t)

Mean and Variance:
    E[x(t)] = x₀*exp(μ*t)
    Var[x(t)] = x₀²*exp(2μt)*(exp(σ²t) - 1)

As t → ∞:
    E[x(t)] → ∞ (for μ ≥ 0)
    Var[x(t)] → ∞

Itô vs Stratonovich
-------------------
This implementation uses Itô calculus. The Stratonovich equivalent would be:
    dx = μ*x*dt + σ*x∘dW

where ∘ denotes Stratonovich product. The drift differs by +σ²*x/2.

Applications
------------
- **Finance**: Stock prices, currency exchange rates
- **Biology**: Population dynamics with environmental stochasticity
- **Physics**: Multiplicative noise processes
- **Economics**: GDP growth models

Examples
--------
>>> # Standard GBM (finance)
>>> stock = GeometricBrownianMotion(mu=0.05, sigma=0.2)
>>> # 5% annual drift, 20% annual volatility (typical stock)
>>>
>>> # Check noise type
>>> stock.is_multiplicative_noise()
True
>>> stock.is_additive_noise()
False
>>>
>>> # Evaluate at x=100 (stock price)
>>> x = np.array([100.0])
>>> u = np.array([0.0])
>>> f = stock.drift(x, u)  # μ*x = 0.05*100 = 5
>>> g = stock.diffusion(x, u)  # σ*x = 0.2*100 = 20
>>>
>>> # Recommended solvers (multiplicative noise)
>>> stock.recommend_solvers('torch')
['euler', 'milstein', 'srk']
>>>
>>> # Population growth with noise
>>> population = GeometricBrownianMotion(mu=0.02, sigma=0.1)
>>> # 2% growth rate, 10% environmental stochasticity
>>>
>>> # Analytical expected value
>>> E_1yr = stock.get_expected_value(x0=100, t=1.0)
>>> print(f"Expected price after 1 year: ${E_1yr:.2f}")
Expected price after 1 year: $105.13
"""

import numpy as np
import sympy as sp

from src.systems.base.stochastic_dynamical_system import StochasticDynamicalSystem


class GeometricBrownianMotion(StochasticDynamicalSystem):
    """
    Geometric Brownian motion with multiplicative noise.

    Stochastic differential equation:
        dx = (μ*x + u)*dt + σ*x*dW

    where:
        - x ∈ ℝ₊: State (price, population, etc.)
        - u ∈ ℝ: Control input
        - μ: Drift rate (expected growth)
        - σ: Volatility (proportional to state)
        - W(t): Standard Wiener process

    Parameters
    ----------
    mu : float, default=0.1
        Drift coefficient (expected growth rate)
        Can be positive (growth) or negative (decay)
    sigma : float, default=0.2
        Volatility (must be positive)
        Typical stocks: 0.15-0.30 (15%-30% annual vol)

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
    >>> # Stock price model
    >>> stock = GeometricBrownianMotion(mu=0.05, sigma=0.2)
    >>>
    >>> # Evaluate at price $100
    >>> x = np.array([100.0])
    >>> u = np.array([0.0])
    >>> drift = stock.drift(x, u)  # 5 $/year
    >>> diffusion = stock.diffusion(x, u)  # 20 $/√year
    >>>
    >>> # Population model
    >>> pop = GeometricBrownianMotion(mu=0.02, sigma=0.1)
    >>>
    >>> # Decaying process
    >>> decay = GeometricBrownianMotion(mu=-0.1, sigma=0.05)
    """

    def define_system(self, mu: float = 0.1, sigma: float = 0.2):
        """
        Define geometric Brownian motion.

        Parameters
        ----------
        mu : float
            Drift coefficient (growth rate)
        sigma : float
            Volatility (must be positive)

        Notes
        -----
        **Parameter Interpretation:**

        μ (drift):
        - μ > 0: Expected growth (stock appreciation)
        - μ = 0: Pure diffusion with multiplicative noise
        - μ < 0: Expected decay (radioactive-like with noise)

        σ (volatility):
        - Small σ (< 0.1): Low uncertainty, smooth paths
        - Medium σ (0.1-0.3): Typical for stocks
        - Large σ (> 0.5): High uncertainty, wild fluctuations

        **State Positivity:**
        If x(0) > 0, then x(t) > 0 for all t (almost surely).
        Do not use with x(0) ≤ 0.
        """
        if sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}")

        # Define symbolic variables
        x = sp.symbols("x", positive=True)  # Positive state
        u = sp.symbols("u", real=True)

        # Define symbolic parameters
        mu_sym = sp.symbols("mu", real=True)
        sigma_sym = sp.symbols("sigma", positive=True)

        # System definition
        self.state_vars = [x]
        self.control_vars = [u]

        # Drift: f(x, u) = μ*x + u
        self._f_sym = sp.Matrix([[mu_sym * x + u]])

        self.parameters = {mu_sym: mu, sigma_sym: sigma}
        self.order = 1

        # Diffusion: g(x, u) = σ*x (multiplicative!)
        self.diffusion_expr = sp.Matrix([[sigma_sym * x]])
        self.sde_type = "ito"

    def get_expected_value(self, x0: float, t: float, u: float = 0.0) -> float:
        """
        Get analytical expected value at time t.

        For u=0: E[X(t)] = x₀*exp(μ*t)

        Parameters
        ----------
        x0 : float
            Initial state (must be positive)
        t : float
            Time (must be non-negative)
        u : float
            Control (assumed constant)

        Returns
        -------
        float
            Expected value E[X(t)]

        Examples
        --------
        >>> gbm = GeometricBrownianMotion(mu=0.05, sigma=0.2)
        >>> E_1yr = gbm.get_expected_value(x0=100, t=1.0)
        >>> print(f"Expected: ${E_1yr:.2f}")
        Expected: $105.13
        """
        if x0 <= 0:
            raise ValueError(f"Initial state must be positive, got {x0}")
        if t < 0:
            raise ValueError(f"Time must be non-negative, got {t}")

        # Extract mu
        mu = None
        for key, val in self.parameters.items():
            if str(key) == "mu":
                mu = val
                break

        if u == 0:
            # Simple exponential growth
            return x0 * np.exp(mu * t)
        else:
            # With constant control (approximate)
            # This is approximate; exact solution with u is more complex
            return x0 * np.exp(mu * t) + u * t * np.exp(mu * t)

    def get_variance(self, x0: float, t: float) -> float:
        """
        Get analytical variance at time t (for u=0).

        Var[X(t)] = x₀²*exp(2μt)*(exp(σ²t) - 1)

        Parameters
        ----------
        x0 : float
            Initial state (must be positive)
        t : float
            Time (must be non-negative)

        Returns
        -------
        float
            Variance Var[X(t)]

        Examples
        --------
        >>> gbm = GeometricBrownianMotion(mu=0.05, sigma=0.2)
        >>> var_1yr = gbm.get_variance(x0=100, t=1.0)
        >>> std_1yr = np.sqrt(var_1yr)
        >>> print(f"Std dev: ${std_1yr:.2f}")
        Std dev: $22.36
        """
        if x0 <= 0:
            raise ValueError(f"Initial state must be positive, got {x0}")
        if t < 0:
            raise ValueError(f"Time must be non-negative, got {t}")

        # Extract parameters
        mu = None
        sigma = None
        for key, val in self.parameters.items():
            if str(key) == "mu":
                mu = val
            elif str(key) == "sigma":
                sigma = val

        return x0**2 * np.exp(2 * mu * t) * (np.exp(sigma**2 * t) - 1)


# ============================================================================
# Specialized GBM Variants
# ============================================================================


class BrownianMotionWithDrift(GeometricBrownianMotion):
    """
    Alias for GBM - sometimes called Brownian motion with drift.

    Mathematically identical to GeometricBrownianMotion.
    """

    pass


def create_stock_price_model(
    expected_return: float = 0.07, annual_volatility: float = 0.20
) -> GeometricBrownianMotion:
    """
    Create GBM model for stock price dynamics.

    Parameters
    ----------
    expected_return : float
        Expected annual return (e.g., 0.07 = 7%)
    annual_volatility : float
        Annual volatility (e.g., 0.20 = 20%)

    Returns
    -------
    GeometricBrownianMotion
        Stock price model

    Examples
    --------
    >>> # S&P 500 typical statistics
    >>> sp500 = create_stock_price_model(
    ...     expected_return=0.10,
    ...     annual_volatility=0.18
    ... )
    """
    return GeometricBrownianMotion(mu=expected_return, sigma=annual_volatility)
