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
Discrete GARCH - Time-Varying Volatility Models (Nobel Prize 2003)
===================================================================

This module provides GARCH (Generalized Autoregressive Conditional 
Heteroskedasticity) models, the fundamental framework for modeling time-varying
volatility in financial markets. GARCH models serve as:

- The industry standard for volatility modeling (Nobel Prize 2003, Robert Engle)
- The foundation for risk management (VaR, CVaR calculations)
- A benchmark for option pricing with stochastic volatility
- An illustration of conditional variance dynamics (heteroskedasticity)
- The canonical model for volatility clustering and persistence

GARCH extends constant-variance models by allowing volatility to change over
time in response to past shocks and past volatility levels, capturing the
fundamental stylized facts of financial returns:
1. **Volatility clustering:** Large changes followed by large changes
2. **Persistence:** High volatility persists (autocorrelated)
3. **Mean reversion:** Volatility returns to long-term average
4. **Heavy tails:** More extreme events than normal distribution

The GARCH(1,1) model is the workhorse: Simple, parsimonious (3 parameters),
yet captures empirical features better than complex alternatives.

Historical Context
------------------

**ARCH Revolution (Engle 1982):**

Robert Engle introduced ARCH (Autoregressive Conditional Heteroskedasticity):
    
    r[k] = σ[k]·w[k]
    σ²[k] = α₀ + α₁·r²[k-1]

**Key Insight:** Variance depends on past squared returns.

**Problem:** Needed many lags (high-order ARCH) to capture persistence.

**GARCH Extension (Bollerslev 1986):**

Generalized ARCH by adding lagged variance:
    
    r[k] = σ[k]·w[k]
    σ²[k] = α₀ + α₁·r²[k-1] + β₁·σ²[k-1]

**Breakthrough:** Just 3 parameters capture volatility dynamics!

**Nobel Prize (2003):**
Robert Engle awarded Nobel Prize for ARCH/GARCH methods.

**Impact:**
- Every financial institution uses GARCH
- Risk management standard (VaR models)
- Option pricing (implied vs realized vol)
- Portfolio optimization (time-varying covariance)

Mathematical Formulation
------------------------

**GARCH(1,1) Model:**

Return equation:
    r[k] = μ + σ[k]·w[k]

Variance equation:
    σ²[k] = ω + α·r²[k-1] + β·σ²[k-1]
          = ω + α·ε²[k-1] + β·σ²[k-1]

where:
    - r[k]: Return (observable)
    - ε[k] = r[k] - μ: Innovation (centered return)
    - σ²[k]: Conditional variance (time-varying, latent)
    - w[k] ~ N(0,1): Standardized innovation (iid)
    - ω > 0: Constant term (long-run variance)
    - α ≥ 0: ARCH effect (shock persistence)
    - β ≥ 0: GARCH effect (volatility persistence)

**Key Feature:**
Variance σ²[k] is **not constant** - it varies based on:
1. Past squared shocks: α·ε²[k-1] (ARCH component)
2. Past variance: β·σ²[k-1] (GARCH component)

**Constraints:**

For well-defined model:
1. ω > 0: Ensures positive variance
2. α ≥ 0, β ≥ 0: Non-negativity
3. α + β < 1: Stationarity (covariance stationarity)

**Unconditional Variance:**
    Var[r] = ω/(1 - α - β)

Exists only if α + β < 1.

**Volatility Persistence:**
Measured by α + β:
- Near 1: Highly persistent (shocks last long)
- Far from 1: Quick mean reversion
- Typical financial data: α + β ≈ 0.95-0.99 (very persistent!)

**Half-Life:**
Time for variance shock to decay by 50%:
    h_{1/2} ≈ ln(0.5)/ln(α + β)

For α + β = 0.95: h_{1/2} ≈ 13.5 days

Stylized Facts of Financial Returns
------------------------------------

**Empirical Observations:**

1. **Returns:**
   - Mean ≈ 0 (for short horizons)
   - Unpredictable (weak autocorrelation)
   - Heavy tails (excess kurtosis)

2. **Volatility:**
   - Time-varying (not constant!)
   - Clustered (high vol → high vol)
   - Persistent (autocorrelated)
   - Mean-reverting (doesn't explode)

3. **Leverage Effect:**
   - Negative returns → higher future volatility
   - Asymmetry (bad news > good news)
   - Not captured by basic GARCH (need EGARCH, GJR)

**GARCH Captures:**
✓ Volatility clustering (via α term)
✓ Persistence (via β term)
✓ Mean reversion (via α + β < 1)
✓ Heavy tails (conditional t-distribution possible)

**GARCH Doesn't Capture:**
✗ Leverage effect (symmetric response to ±shocks)
✗ Long memory (if true fractional integration)

ARCH vs GARCH
-------------

**ARCH(q):**
    σ²[k] = ω + Σα_i·ε²[k-i]

Needs many lags (large q) to capture persistence.
- ARCH(1): Only immediate past shock
- ARCH(5): Five lags (5 parameters)

**GARCH(1,1):**
    σ²[k] = ω + α·ε²[k-1] + β·σ²[k-1]

Captures same persistence with fewer parameters (3 total).
- More parsimonious
- Better forecasting
- Standard in practice

**GARCH as ARCH(∞):**

GARCH(1,1) equivalent to ARCH(∞):
    σ²[k] = ω/(1-β) + α·Σβ^i·ε²[k-1-i]

Infinite memory with geometric decay (via β^i).

Applications
------------

**1. Risk Management:**

**Value-at-Risk (VaR):**
Estimate: P(loss > VaR) = 0.05

Using GARCH:
1. Forecast σ²[k+1] from GARCH
2. VaR = μ + z_α·σ[k+1]
3. Updates daily (time-varying risk)

**Conditional VaR (CVaR):**
Expected loss given exceedance.

**2. Option Pricing:**

**Implied vs Realized Volatility:**
- Implied: From Black-Scholes (market's forecast)
- Realized: From GARCH (historical model)
- Compare: Volatility risk premium

**GARCH Option Pricing:**
Monte Carlo with GARCH volatility path.

**3. Portfolio Optimization:**

**Time-Varying Covariance:**
Multivariate GARCH (MGARCH, DCC):
- Correlations change over time
- Portfolio weights adjust dynamically
- Risk parity with GARCH

**4. Trading Strategies:**

**Volatility Trading:**
- Sell options when GARCH σ high (overpriced)
- Buy when σ low (underpriced)

**Risk Targeting:**
Scale positions inversely with σ[k].

**5. Regulatory:**

**Basel III:**
Banks required to estimate VaR.
GARCH standard for internal models.

**Stress Testing:**
Forecast tail events using GARCH.

**6. Macroeconomics:**

**Uncertainty Shocks:**
Macro volatility (GDP, inflation uncertainty).

**Policy Uncertainty:**
Time-varying policy impact.

Volatility Forecasting
----------------------

**One-Step-Ahead:**
    σ²[k+1|k] = ω + α·ε²[k] + β·σ²[k]

Direct from GARCH equation.

**Multi-Step-Ahead:**
    σ²[k+h|k] = ω·Σ(α+β)^i + (α+β)^h·σ²[k]

For large h:
    σ²[k+h|k] → ω/(1-α-β) (unconditional variance)

**Mean Reversion:**
Forecasts revert to long-run average.
Speed depends on α + β.

Estimation
----------

**Maximum Likelihood:**

Log-likelihood (Gaussian):
    ℓ = -Σ[0.5·ln(2π) + 0.5·ln(σ²[k]) + 0.5·ε²[k]/σ²[k]]

Maximize numerically (nonlinear optimization).

**Steps:**
1. Initialize: σ²[0], ε[0]
2. For k=1,...,N:
   - Compute σ²[k] from GARCH equation
   - Compute ε[k] = r[k] - μ
   - Add to log-likelihood
3. Optimize over (ω, α, β, μ)

**Constraints:**
- ω > 0
- α ≥ 0, β ≥ 0
- α + β < 1 (usually enforced)

**Quasi-Maximum Likelihood (QML):**
Gaussian likelihood even if returns not Gaussian.
Robust to misspecification (White standard errors).

**Validation:**

1. **Standardized Residuals:**
   z[k] = ε[k]/σ[k]
   
   Should be iid N(0,1).

2. **Tests:**
   - Ljung-Box on z²[k] (no remaining ARCH)
   - Normality (if assumed)
   - Sign bias test (asymmetry check)

Common Pitfalls
---------------

1. **Constraint Violations:**
   - α + β ≥ 1: Non-stationary (IGARCH)
   - Can happen in estimation
   - Forecasts explode

2. **Overfitting:**
   - GARCH(2,2), GARCH(3,3) overfit
   - GARCH(1,1) usually sufficient
   - Use information criteria

3. **Symmetric Response:**
   - GARCH treats +/- shocks equally
   - Real markets: Leverage effect (- worse than +)
   - Use EGARCH, GJR-GARCH if asymmetry present

4. **Normality Assumption:**
   - Returns often fat-tailed
   - Use Student-t innovations
   - Or non-parametric methods

5. **Initialization:**
   - Need σ²[0] for recursion
   - Common: Unconditional variance or sample variance
   - Or treat as parameter (estimate)

6. **Numerical Issues:**
   - σ²[k] can become negative if constraints violated
   - σ²[k] → 0: Division issues
   - Need bounds in optimization

Extensions
----------

**1. EGARCH (Exponential GARCH):**
    ln(σ²[k]) = ω + α·|z[k-1]| + γ·z[k-1] + β·ln(σ²[k-1])

Asymmetry via γ (leverage effect).

**2. GJR-GARCH:**
    σ²[k] = ω + α·ε²[k-1] + γ·I_{ε<0}·ε²[k-1] + β·σ²[k-1]

Negative shocks have extra impact γ.

**3. IGARCH (Integrated):**
    α + β = 1 (unit root in variance)

Non-stationary but widely used.

**4. Component GARCH:**
    σ²[k] = permanent[k] + transitory[k]

Separates short and long-run components.

**5. Multivariate:**
- VECH, BEKK, DCC, CCC
- Time-varying correlations
- Portfolio applications

**6. GARCH-M (in Mean):**
    r[k] = μ + λ·σ²[k] + σ[k]·w[k]

Risk premium λ·σ².

**Impact:**
GARCH demonstrated:
- Volatility is predictable (even if returns aren't)
- Time-varying risk is the norm (not constant)
- Simple models can capture complex patterns
- Financial econometrics can be practical
"""

import numpy as np
import sympy as sp
from typing import Optional

from src.systems.base.core.discrete_stochastic_system import DiscreteStochasticSystem


class DiscreteGARCH11(DiscreteStochasticSystem):
    """
    GARCH(1,1) - time-varying volatility model (Nobel Prize 2003).

    The industry standard for modeling financial volatility, capturing
    volatility clustering, persistence, and mean reversion with just
    three parameters. This represents one of the most important models
    in financial econometrics.

    Model Equations
    ---------------
    Return process:
        r[k] = μ + ε[k]
        ε[k] = σ[k]·w[k]

    Variance process (GARCH):
        σ²[k] = ω + α·ε²[k-1] + β·σ²[k-1]

    where:
        - r[k]: Return (observable)
        - μ: Mean return (typically ≈ 0 for daily)
        - ε[k]: Innovation (zero-mean shock)
        - σ²[k]: Conditional variance (time-varying, latent)
        - w[k] ~ N(0,1): Standardized innovation (iid)
        - ω > 0: Constant term
        - α ≥ 0: ARCH coefficient (shock effect)
        - β ≥ 0: GARCH coefficient (persistence)

    **State-Space Form:**

    Augmented state: Z = [r, σ²] (return and variance)

    Dynamics (nonlinear!):
        r[k] = μ + σ[k]·w[k]
        σ²[k+1] = ω + α·(r[k] - μ)² + β·σ²[k]

    Physical Interpretation
    -----------------------
    **What GARCH Models:**

    Financial markets exhibit **volatility clustering**:
    - Large price movements followed by large movements
    - Calm periods followed by calm periods
    - "Volatility begets volatility"

    **Why This Happens:**

    1. **Information Flow:**
       - News arrives in clusters
       - Uncertainty propagates

    2. **Market Microstructure:**
       - Feedback: Vol → wider spreads → more vol
       - Herding behavior

    3. **Risk Preferences:**
       - Risk-off → sell → higher vol → more risk-off
       - Feedback loops

    **GARCH Equation Interpretation:**

    σ²[k] = ω + α·ε²[k-1] + β·σ²[k-1]

    Components:
    1. **ω:** Baseline variance (floor)
    2. **α·ε²[k-1]:** Recent shock (ARCH effect)
       - Large |ε[k-1]| → high σ²[k]
       - "News impact curve"
    3. **β·σ²[k-1]:** Past variance (persistence)
       - High σ²[k-1] → high σ²[k]
       - Volatility clustering

    Key Features
    ------------
    **Time-Varying Variance:**
    Unlike constant-variance models, σ²[k] changes every period.

    **Volatility Clustering:**
    High volatility tends to persist (α + β near 1).

    **Mean Reversion:**
    Variance reverts to ω/(1-α-β) over time.

    **Conditional vs Unconditional:**
    - Conditional: Var[r[k]|past] = σ²[k] (time-varying)
    - Unconditional: Var[r[k]] = ω/(1-α-β) (constant)

    **Heavy Tails:**
    Even with Gaussian w[k], unconditional distribution has excess kurtosis.

    **Predictable Volatility:**
    Can forecast σ²[k+h] from past (even if returns unpredictable).

    Mathematical Properties
    -----------------------
    **Stationarity Condition:**
        α + β < 1

    If α + β = 1: IGARCH (integrated, non-stationary variance)
    If α + β > 1: Explosive (unstable)

    **Unconditional Moments:**

    Mean: E[r] = μ

    Variance: Var[r] = ω/(1 - α - β)

    Kurtosis: Excess kurtosis > 0 (heavy tails)
        Kurt = 3·(1 - (α+β)²)/(1 - (α+β)² - 2α²)

    **Autocorrelation:**
    - Returns r[k]: ρ(h) ≈ 0 (unpredictable)
    - Squared returns ε²[k]: ρ(h) = (α+β)^h (geometric decay)

    **Persistence:**
    Measured by α + β (sum of ARCH and GARCH):
    - α + β ≈ 0.99: Very persistent (typical stocks)
    - α + β ≈ 0.95: Persistent
    - α + β ≈ 0.80: Moderate persistence

    **Volatility Forecast:**
    h-step ahead:
        σ²[k+h|k] = σ̄² + (α+β)^h·(σ²[k] - σ̄²)

    where σ̄² = ω/(1-α-β) is long-run variance.

    Exponential decay to long-run average.

    Physical Interpretation
    -----------------------
    **Parameter ω (Baseline):**
    - Units: [return]²
    - Long-run variance: σ̄² = ω/(1-α-β)
    - Typical: Relates to annual volatility

    **Parameter α (ARCH Effect):**
    - Dimensionless
    - Weight on recent shock
    - Typical: 0.05-0.15
    - Higher α: More reactive to news

    **Parameter β (GARCH Effect):**
    - Dimensionless
    - Weight on past variance
    - Typical: 0.80-0.92
    - Higher β: More persistent

    **Sum α + β (Total Persistence):**
    - Typical: 0.95-0.99 (very high!)
    - Near 1: "Volatility never dies"
    - Financial data: Usually >0.90

    **Typical Values (Daily Stock Returns):**
    - ω ≈ 0.000001-0.00001 (depends on scaling)
    - α ≈ 0.05-0.10
    - β ≈ 0.85-0.92
    - α + β ≈ 0.95-0.98

    State Space
    -----------
    Augmented state: Z = [r, σ²] ∈ ℝ²
        - r: Observable return
        - σ²: Latent conditional variance

    Observable: r ∈ ℝ
        - What we actually measure (price changes)

    Latent: σ² ∈ ℝ₊
        - Unobserved (must be estimated/filtered)

    Control: u (typically none for GARCH)
        - Could add for interventions

    Parameters
    ----------
    omega : float, default=0.00001
        Constant term (must be positive)
        - Sets baseline variance
        - Typical: 1e-6 to 1e-4 (daily returns)

    alpha : float, default=0.1
        ARCH coefficient (must be non-negative)
        - News impact (shock sensitivity)
        - Typical: 0.05-0.15

    beta : float, default=0.85
        GARCH coefficient (must be non-negative)
        - Persistence (volatility memory)
        - Typical: 0.80-0.92

    mu : float, default=0.0
        Mean return
        - Often ≈ 0 for daily returns
        - Can estimate or fix

    dt : float, default=1.0
        Sampling period (1 = daily typical)

    Stochastic Properties
    ---------------------
    - System Type: NONLINEAR (σ² equation)
    - Conditional: Variance time-varying
    - Stationary: If α + β < 1
    - Heavy Tails: Yes (excess kurtosis)
    - Volatility Clustering: Yes (by design)
    - Leverage Effect: No (symmetric)

    Applications
    ------------
    **1. Risk Management:**
    - VaR calculation
    - Expected shortfall (CVaR)
    - Stress testing

    **2. Option Pricing:**
    - Volatility forecasting
    - Monte Carlo with GARCH
    - Implied vs realized vol

    **3. Portfolio:**
    - Time-varying covariance (MGARCH)
    - Dynamic hedging
    - Risk parity

    **4. Trading:**
    - Volatility strategies
    - Position sizing (risk targeting)
    - Options trading signals

    **5. Regulation:**
    - Basel capital requirements
    - Internal VaR models
    - Stress testing

    Numerical Simulation
    --------------------
    **Recursion:**

    Initialize: σ²[0] = ω/(1-α-β)

    For k = 0, 1, 2, ...:
        1. Draw w[k] ~ N(0,1)
        2. Compute r[k] = μ + σ[k]·w[k]
        3. Update σ²[k+1] = ω + α·(r[k]-μ)² + β·σ²[k]

    **State-Space:**
    State Z = [r, σ²]
    - r observable
    - σ² latent (filtered in practice)

    Comparison with Constant Variance
    ----------------------------------
    **Constant Variance:**
        r[k] = μ + σ·w[k]

    Problems:
    - Misses volatility clustering
    - Underestimates tail risk
    - Poor VaR forecasts

    **GARCH:**
        r[k] = μ + σ[k]·w[k]
        σ²[k] = f(past)

    Advantages:
    - Captures clustering
    - Better tail risk
    - Accurate VaR

    Limitations
    -----------
    - Symmetric (leverage effect ignored)
    - Gaussian (can extend to Student-t)
    - Low frequency (daily typically, not intraday)
    - Univariate (for MGARCH extension)

    Extensions
    ----------
    - EGARCH: Asymmetry
    - GJR-GARCH: Threshold effects
    - FIGARCH: Long memory
    - DCC-GARCH: Dynamic conditional correlation
    - GARCH-M: Risk premium in mean
    """

    def define_system(
        self,
        omega: float = 0.00001,
        alpha: float = 0.1,
        beta: float = 0.85,
        mu: float = 0.0,
        dt: float = 1.0,
    ):
        """
        Define GARCH(1,1) dynamics.

        Parameters
        ----------
        omega : float, default=0.00001
            Constant term in variance equation (must be positive)
            - Sets baseline variance
            - Typical daily: 1e-6 to 1e-4

        alpha : float, default=0.1
            ARCH coefficient (must be non-negative)
            - Weight on past squared shock
            - Typical: 0.05-0.15
            - Higher: More reactive to news

        beta : float, default=0.85
            GARCH coefficient (must be non-negative)
            - Weight on past variance
            - Typical: 0.80-0.92
            - Higher: More persistent

        mu : float, default=0.0
            Mean return
            - Typically ≈ 0 for daily
            - Can be non-zero for longer horizons

        dt : float, default=1.0
            Sampling period (1 = daily typical)

        Raises
        ------
        ValueError
            If ω ≤ 0, α < 0, β < 0, or α + β ≥ 1

        Notes
        -----
        **Parameter Constraints:**

        1. **Positivity:**
           - ω > 0: Ensures σ²[k] > 0
           - α ≥ 0, β ≥ 0: Non-negativity

        2. **Stationarity:**
           - α + β < 1: Variance mean-reverting
           - α + β = 1: IGARCH (non-stationary)
           - α + β > 1: Explosive (invalid)

        **Long-Run Variance:**
            σ̄² = ω/(1 - α - β)

        Must be finite (requires α + β < 1).

        **Persistence:**
            α + β

        Typical values:
        - Stocks: 0.95-0.99 (very persistent!)
        - Bonds: 0.90-0.95
        - FX: 0.93-0.97

        **Half-Life:**
        Time for variance shock to decay by 50%:
            h_{1/2} ≈ ln(0.5)/ln(α+β)

        Example: α + β = 0.95 → h ≈ 13.5 days

        **Typical Configurations:**

        **S&P 500 (daily):**
        - ω ≈ 0.000001
        - α ≈ 0.10
        - β ≈ 0.88
        - α + β = 0.98 (very persistent)

        **Emerging Market (more volatile):**
        - ω ≈ 0.00005
        - α ≈ 0.15
        - β ≈ 0.80
        - α + β = 0.95

        **Low Volatility Stock:**
        - ω ≈ 0.0000005
        - α ≈ 0.05
        - β ≈ 0.92
        - α + β = 0.97

        **Initialization:**

        For σ²[0], common choices:
        1. Unconditional: ω/(1-α-β)
        2. Sample variance from data
        3. Estimated as parameter

        **State-Space Structure:**

        This is NONLINEAR state-space:
        - r[k] depends on σ[k] (multiplicatively)
        - σ²[k+1] depends on ε²[k] (quadratically)

        Not amenable to standard Kalman filter.
        Need extended/unscented/particle filter.

        Examples
        --------
        >>> # Standard stock GARCH
        >>> stock = DiscreteGARCH11(
        ...     omega=0.000002,
        ...     alpha=0.08,
        ...     beta=0.90,
        ...     mu=0.0005
        ... )
        >>> 
        >>> # Check constraints
        >>> persistence = 0.08 + 0.90
        >>> print(f"Stationary: {persistence < 1}")
        >>> 
        >>> # Long-run volatility
        >>> sigma_lr = np.sqrt(0.000002 / (1 - persistence))
        >>> print(f"Long-run daily vol: {sigma_lr:.4f}")
        >>> annual_vol = sigma_lr * np.sqrt(252)
        >>> print(f"Annualized: {annual_vol:.2%}")
        """
        # Validate parameters
        if omega <= 0:
            raise ValueError(f"omega must be positive, got {omega}")
        if alpha < 0:
            raise ValueError(f"alpha must be non-negative, got {alpha}")
        if beta < 0:
            raise ValueError(f"beta must be non-negative, got {beta}")

        # Check stationarity
        persistence = alpha + beta
        if persistence >= 1.0:
            if np.isclose(persistence, 1.0, atol=1e-6):
                import warnings
                warnings.warn(
                    f"α + β = {persistence:.6f} ≈ 1: IGARCH (integrated GARCH). "
                    "Variance is non-stationary (no unconditional variance). "
                    "Common in practice but theoretically problematic.",
                    UserWarning
                )
            else:
                raise ValueError(
                    f"α + β = {persistence:.6f} ≥ 1: Explosive variance. "
                    "For stationary GARCH, require α + β < 1."
                )

        # Store parameters
        self.omega = omega
        self.alpha = alpha
        self.beta = beta
        self.mu = mu

        # Augmented state: [r, σ²]
        r, sigma_sq = sp.symbols("r sigma_sq", real=True)
        # Note: σ² should be positive but we use real=True for broader compatibility

        # Parameters as symbols
        omega_sym = sp.symbols("omega", positive=True)
        alpha_sym = sp.symbols("alpha", nonnegative=True)
        beta_sym = sp.symbols("beta", nonnegative=True)
        mu_sym = sp.symbols("mu", real=True)

        self.state_vars = [r, sigma_sq]
        self.control_vars = []  # GARCH typically autonomous

        # DETERMINISTIC PART (conditional mean and variance update)
        # This represents E[Z[k+1]|Z[k]] before adding innovation
        
        # Return: E[r[k+1]|past] = μ (return unpredictable given variance known)
        r_next = mu_sym

        # Variance: σ²[k+1] = ω + α·(r[k]-μ)² + β·σ²[k]
        epsilon_sq = (r - mu_sym)**2
        sigma_sq_next = omega_sym + alpha_sym * epsilon_sq + beta_sym * sigma_sq

        self._f_sym = sp.Matrix([r_next, sigma_sq_next])

        self.parameters = {
            omega_sym: omega,
            alpha_sym: alpha,
            beta_sym: beta,
            mu_sym: mu,
        }
        self.order = 1
        self._dt = dt

        # STOCHASTIC PART
        # Innovation structure:
        # - Return gets σ[k]·w[k] (conditional on variance)
        # - Variance gets no direct noise (deterministic given ε²)
        
        # For state-space form, this is tricky because return innovation
        # depends on current σ[k]. We use simplified additive form.
        
        # Simplified: Treat as additive noise scaled by √σ²
        # More accurate: Would need multiplicative noise
        self.diffusion_expr = sp.Matrix([
            [sp.sqrt(sigma_sq)],  # Return innovation scaled by volatility
            [0]                    # Variance has no direct noise
        ])

        self.sde_type = "ito"

        # Output: Return (what's observed)
        self._h_sym = sp.Matrix([r])

    def get_unconditional_variance(self) -> float:
        """
        Get long-run (unconditional) variance σ̄² = ω/(1-α-β).

        Returns
        -------
        float
            Unconditional variance

        Raises
        ------
        ValueError
            If α + β ≥ 1 (non-stationary)

        Examples
        --------
        >>> garch = DiscreteGARCH11(omega=0.00001, alpha=0.1, beta=0.85)
        >>> var_uncond = garch.get_unconditional_variance()
        >>> vol_uncond = np.sqrt(var_uncond)
        >>> print(f"Long-run daily vol: {vol_uncond:.4f}")
        """
        if self.alpha + self.beta >= 1:
            raise ValueError(
                f"Unconditional variance undefined for α+β = {self.alpha + self.beta} ≥ 1"
            )

        return self.omega / (1 - self.alpha - self.beta)

    def get_persistence(self) -> float:
        """Get volatility persistence α + β."""
        return self.alpha + self.beta

    def get_half_life(self) -> float:
        """
        Get volatility shock half-life (periods to decay by 50%).

        Returns
        -------
        float
            Half-life [periods]

        Examples
        --------
        >>> garch = DiscreteGARCH11(alpha=0.1, beta=0.85)
        >>> hl = garch.get_half_life()
        >>> print(f"Volatility half-life: {hl:.1f} days")
        """
        persistence = self.get_persistence()
        if persistence <= 0 or persistence >= 1:
            return np.inf
        return np.log(0.5) / np.log(persistence)

    def forecast_variance(
        self,
        current_variance: float,
        horizon: int = 10,
    ) -> np.ndarray:
        """
        Forecast variance h steps ahead.

        σ²[k+h|k] = σ̄² + (α+β)^h·(σ²[k] - σ̄²)

        Parameters
        ----------
        current_variance : float
            Current conditional variance σ²[k]
        horizon : int, default=10
            Forecast horizon

        Returns
        -------
        np.ndarray
            Variance forecasts for h=1,...,horizon

        Examples
        --------
        >>> garch = DiscreteGARCH11(omega=0.00001, alpha=0.1, beta=0.85)
        >>> forecasts = garch.forecast_variance(current_variance=0.0001, horizon=20)
        >>> print(f"1-day: {np.sqrt(forecasts[0]):.4f}")
        >>> print(f"20-day: {np.sqrt(forecasts[-1]):.4f}")
        """
        sigma_bar_sq = self.get_unconditional_variance()
        persistence = self.get_persistence()

        forecasts = np.zeros(horizon)
        for h in range(horizon):
            forecasts[h] = sigma_bar_sq + persistence**(h+1) * (current_variance - sigma_bar_sq)

        return forecasts


# Convenience functions
def create_equity_garch(
    asset_type: str = 'large_cap',
) -> DiscreteGARCH11:
    """
    Create GARCH model for equity returns.

    Parameters
    ----------
    asset_type : str, default='large_cap'
        'large_cap', 'small_cap', or 'emerging'

    Returns
    -------
    DiscreteGARCH11

    Examples
    --------
    >>> # S&P 500 stock
    >>> large_cap = create_equity_garch('large_cap')
    >>> 
    >>> # Small cap (higher vol, less persistent)
    >>> small_cap = create_equity_garch('small_cap')
    """
    presets = {
        'large_cap': {'omega': 0.000001, 'alpha': 0.08, 'beta': 0.90},
        'small_cap': {'omega': 0.000005, 'alpha': 0.12, 'beta': 0.85},
        'emerging': {'omega': 0.00002, 'alpha': 0.15, 'beta': 0.80},
    }

    params = presets.get(asset_type, presets['large_cap'])
    return DiscreteGARCH11(**params, mu=0.0005, dt=1.0)


def create_fx_garch(
    pair: str = 'major',
) -> DiscreteGARCH11:
    """
    Create GARCH for FX returns.

    Parameters
    ----------
    pair : str, default='major'
        'major' (EUR/USD) or 'emerging' (USD/TRY)

    Returns
    -------
    DiscreteGARCH11

    Examples
    --------
    >>> # Major pair (low vol)
    >>> eurusd = create_fx_garch('major')
    """
    presets = {
        'major': {'omega': 0.0000005, 'alpha': 0.05, 'beta': 0.93},
        'emerging': {'omega': 0.00001, 'alpha': 0.10, 'beta': 0.88},
    }

    params = presets.get(pair, presets['major'])
    return DiscreteGARCH11(**params, mu=0.0, dt=1.0)