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
Discrete AR(1) Process - First-Order Autoregressive System
===========================================================

This module provides implementations of the discrete-time AR(1) (Autoregressive
order 1) process, the fundamental building block of time series analysis. The
AR(1) process is distinguished by:

- The simplest non-trivial autoregressive model
- Discrete-time analog of the Ornstein-Uhlenbeck process
- Foundation for ARMA, ARIMA, and state-space models
- Widely used in econometrics, signal processing, and control
- Exact analytical solution for all moments and distributions

The AR(1) process represents the discrete-time limit of many continuous
stochastic processes and is the canonical model for studying persistence,
mean reversion, and autocorrelation in time series data.

Mathematical Background
-----------------------
The AR(1) is the simplest discrete stochastic model exhibiting:
- Memory: Current value depends on past value
- Stochastic dynamics: Random innovations each period
- Markov property: Only immediate past matters

**Physical Interpretation:**
AR(1) naturally arises in discrete-time systems with:
- Partial persistence from period to period
- Random shocks each period
- First-order dynamics (no higher-order memory)

Examples:
- Economic variables: GDP growth, inflation (period = quarter/year)
- Financial returns: Stock returns with momentum (period = day)
- Signal processing: Filtered noise (period = sample)
- Control: Discretized continuous systems (period = sampling interval)

Mathematical Formulation
------------------------

**Standard Form:**
    X[k+1] = φ·X[k] + w[k]

where:
    - X[k]: State at discrete time k
    - φ: Autoregressive coefficient (-∞ < φ < ∞)
    - w[k]: White noise innovation ~ N(0, σ²)
    - Innovations uncorrelated: E[w[i]·w[j]] = 0 for i ≠ j

**With Control Input:**
    X[k+1] = φ·X[k] + u[k] + σ·w[k]

The control u[k] provides external forcing or policy intervention.

**Centered vs. Non-Centered:**
Centered (zero mean):
    X[k+1] = φ·X[k] + σ·w[k]

Non-centered (mean μ):
    X[k+1] = c + φ·X[k] + σ·w[k]
    
where c = μ·(1-φ) ensures E[X] = μ.

**Equivalent Forms:**
Mean deviation form:
    (X[k+1] - μ) = φ·(X[k] - μ) + σ·w[k]

This shows φ controls persistence of deviations from mean.

Relationship to Ornstein-Uhlenbeck
-----------------------------------
The AR(1) is the discrete-time analog of the continuous Ornstein-Uhlenbeck
process. For small sampling interval Δt:

**OU Process:**
    dX = -α·X·dt + σ_c·dW

**Exact Discretization:**
    X[k+1] = e^(-α·Δt)·X[k] + √((σ_c²/2α)·(1-e^(-2α·Δt)))·w[k]

**Correspondence:**
    φ = e^(-α·Δt)
    σ² = (σ_c²/2α)·(1-e^(-2α·Δt))

**Limits:**
- Small Δt: φ ≈ 1 - α·Δt (near unit root)
- Large α·Δt: φ → 0 (white noise)

Analytical Properties
---------------------

**Exact Solution:**
For X[0] = X₀ and u[k] = u (constant):

    X[k] = φ^k·X₀ + u·(1-φ^k)/(1-φ) + Σⱼ₌₀^(k-1) φ^(k-1-j)·σ·w[j]

**Moments (u = 0):**

Mean:
    E[X[k]] = φ^k·X₀

Exponentially decays to zero (if |φ| < 1).

Variance:
    Var[X[k]] = σ²·(1-φ^(2k))/(1-φ²)

Increases from 0 to stationary value σ²/(1-φ²).

Autocovariance (lag h):
    γ(h) = φ^h·σ²/(1-φ²)

Exponentially decaying with lag.

**Asymptotic Behavior (k → ∞):**

For |φ| < 1 (Stationary):
    E[X[∞]] = u/(1-φ)
    Var[X[∞]] = σ²/(1-φ²)
    Distribution: X[∞] ~ N(u/(1-φ), σ²/(1-φ²))

For |φ| ≥ 1 (Non-Stationary):
    Variance → ∞ (unit root or explosive)

**Autocorrelation Function:**
    ρ(h) = φ^h

Key insight: Autocorrelation decays geometrically at rate φ.
- φ = 0.9: Slow decay (long memory)
- φ = 0.5: Moderate decay
- φ = 0.1: Fast decay (short memory)

Key Properties
--------------

**1. Stationarity:**
Process is (covariance) stationary if and only if |φ| < 1.

**Stationary:**
- |φ| < 1: Mean and variance converge to constants
- Autocorrelation decays to zero
- Ergodic (time averages = ensemble averages)

**Non-Stationary:**
- φ = 1: Unit root (random walk), variance grows linearly
- |φ| > 1: Explosive, variance grows exponentially
- φ = -1: Oscillates between ±∞

**2. Mean Reversion:**
For |φ| < 1, process reverts to mean with:
- Half-life: h = ln(0.5)/ln(φ) periods
- Time constant: τ = -1/ln(φ) periods

Smaller φ → faster reversion
Larger φ (near 1) → slower reversion

**3. Persistence:**
Measured by φ:
- φ near 1: High persistence (shocks last long time)
- φ near 0: Low persistence (shocks die out quickly)
- φ < 0: Negative autocorrelation (oscillatory)

**4. Markov Property:**
X[k+1] depends only on X[k], not on X[k-1], X[k-2], ...
This is the defining property of AR(1).

**5. Gaussian Process:**
If w[k] ~ N(0,σ²), then X[k] ~ N(μ[k], σ²[k]) for all k.
Joint distribution of (X[k₁], ..., X[kₙ]) is multivariate normal.

**6. Additive Noise:**
Innovation w[k] independent of state.
Simplifies analysis compared to multiplicative noise.

**7. Linear Dynamics:**
Superposition principle applies:
- Solution is sum of homogeneous + particular solutions
- Can use linear control theory

Physical and Statistical Interpretation
----------------------------------------

**Autoregressive Coefficient φ:**
- Dimension: Dimensionless (ratio)
- Interpretation: Fraction of current value persisting to next period
- Memory parameter: Larger φ → longer memory

**Stability Regions:**
- 0 < φ < 1: Stable, positive persistence (typical)
- φ = 1: Unit root (random walk boundary)
- φ > 1: Unstable, explosive
- -1 < φ < 0: Stable, alternating (rare)
- φ = -1: Unstable, perfect alternation
- φ < -1: Unstable, explosive alternation

**Innovation Variance σ²:**
- Dimension: [state]²
- Interpretation: Variance of random shock each period
- Sets scale of fluctuations

**Stationary Variance:**
    Var[X[∞]] = σ²/(1-φ²)

Increases with:
- Larger σ² (more noise)
- Larger φ (more persistence)

**Signal-to-Noise Ratio:**
    SNR = φ²/(1-φ²) = φ²·Var[X]/σ²

High SNR: Process dominated by persistence
Low SNR: Process dominated by noise

**Unconditional vs. Conditional Variance:**
- Unconditional: Var[X[k]] (stationary variance)
- Conditional: Var[X[k+1]|X[k]] = σ² (innovation variance)

Connection to Continuous Time
------------------------------
AR(1) arises from discretizing continuous processes with sampling period Δt.

**From OU Process:**
OU: dX = -α·X·dt + σ_c·dW
AR(1): φ = e^(-α·Δt)

**Recover Continuous Parameters:**
    α = -ln(φ)/Δt
    σ_c ≈ σ/√Δt (approximate for small Δt)

**Examples:**
- Δt = 0.1, φ = 0.9 → α ≈ 1.05
- Δt = 1.0, φ = 0.5 → α ≈ 0.69
- Δt = 0.01, φ = 0.99 → α ≈ 1.01

Applications
------------

**1. Econometrics:**

**GDP Growth:**
Quarterly GDP growth exhibits AR(1) behavior:
    g[t] = φ·g[t-1] + ε[t]

Typical φ ≈ 0.3-0.5 (moderate persistence).

**Inflation:**
Monthly/quarterly inflation shows persistence:
Typical φ ≈ 0.7-0.9 (high persistence).

**Interest Rates:**
Short-term rates discretized from continuous models.

**Unemployment:**
Job market dynamics with memory.

**2. Financial Time Series:**

**Asset Returns:**
Daily stock returns may show small AR(1):
    r[t] = φ·r[t-1] + ε[t]

Typical φ ≈ 0-0.1 (weak persistence).

**Volatility:**
Log-volatility often AR(1) in GARCH models.

**Exchange Rates:**
Currency returns discretized.

**3. Signal Processing:**

**Colored Noise:**
AR(1) generates colored noise from white noise.
Power spectral density:
    S(f) ∝ σ²/|1 - φ·e^(-2πif·Δt)|²

**Filtering:**
Digital filter with one pole at z = φ.

**Prediction:**
One-step-ahead predictor: X̂[k+1|k] = φ·X[k]

**4. Control Systems:**

**Disturbance Modeling:**
Model disturbances as AR(1) for control design.

**State Estimation:**
Kalman filter with AR(1) process noise.

**Model Predictive Control:**
Predict future states using AR(1) model.

**5. Climate Science:**

**Temperature Anomalies:**
Monthly temperature deviations.

**El Niño Index:**
ENSO index shows AR(1) behavior.

**6. Biology:**

**Population Counts:**
Annual census data with carryover.

**Gene Expression:**
Discrete-time measurements.

Numerical Simulation
--------------------

**Direct Simulation:**
    X[k+1] = φ·X[k] + u + σ·Z[k]

where Z[k] ~ N(0,1) is standard normal.

This is **exact** - no approximation involved.

**Algorithm:**
```python
X = np.zeros(N+1)
X[0] = X0
for k in range(N):
    X[k+1] = phi * X[k] + u + sigma * np.random.randn()
```

**Vectorized:**
```python
Z = np.random.randn(N)
X = signal.lfilter([sigma], [1, -phi], Z, zi=[X0*phi])[0]
```

**Efficiency:**
- No time-stepping error
- No stability constraints
- Fast simulation
- Can use FFT for long sequences

Statistical Analysis
--------------------

**Parameter Estimation:**

**Least Squares (OLS):**
Given observations X[0], X[1], ..., X[N]:

    φ̂ = Σ X[k]·X[k+1] / Σ X[k]²
    σ̂² = (1/N)·Σ (X[k+1] - φ̂·X[k])²

**Maximum Likelihood:**
Same as OLS for Gaussian innovations.

**Yule-Walker:**
From autocorrelations:
    φ̂ = ρ̂(1)
    σ̂² = γ̂(0)·(1 - ρ̂(1)²)

where ρ̂(h) is sample autocorrelation at lag h.

**Bias Correction:**
OLS estimator biased in small samples. Corrected:
    φ̂_unbiased ≈ φ̂ + (1+3φ̂)/N

**Hypothesis Testing:**

1. **Unit Root Test:**
   - H₀: φ = 1 (random walk)
   - H₁: |φ| < 1 (stationary)
   - Use Dickey-Fuller test

2. **Stationarity:**
   - Test |φ̂| < 1
   - Confidence interval for φ

3. **White Noise Test:**
   - H₀: φ = 0 (white noise)
   - t-test: t = φ̂/SE(φ̂)

4. **Model Adequacy:**
   - Check residuals for autocorrelation (Ljung-Box)
   - Check normality (Jarque-Bera)

**Model Selection:**
- AIC: penalizes complexity
- BIC: stronger penalty
- Compare AR(1) vs AR(2), ARMA(1,1)

Comparison with Other Models
-----------------------------

**vs. White Noise:**
- WN: φ = 0 (no persistence)
- AR(1): φ ≠ 0 (memory)
- AR(1) reduces to WN when φ = 0

**vs. Random Walk:**
- RW: φ = 1 (unit root)
- AR(1): |φ| < 1 (stationary)
- AR(1) → RW as φ → 1

**vs. AR(p) (Higher Order):**
- AR(p): Depends on p past values
- AR(1): Only immediate past
- AR(1) is special case of AR(p)

**vs. MA(1) (Moving Average):**
- MA(1): X[k] = w[k] + θ·w[k-1]
- AR(1): Depends on X[k-1], not w[k-1]
- AR(1) has infinite MA representation

**vs. ARMA(1,1):**
- ARMA: Combines AR and MA
- AR(1): Pure autoregressive
- More flexible but more parameters

**vs. Ornstein-Uhlenbeck:**
- OU: Continuous time
- AR(1): Discrete time
- AR(1) is discretization of OU

Extensions and Generalizations
-------------------------------

**1. Higher-Order AR:**
    AR(p): X[k] = Σφᵢ·X[k-i] + w[k]

**2. ARMA Models:**
    ARMA(p,q): Adds moving average terms

**3. ARIMA:**
    Integrated AR: Differences for non-stationary data

**4. GARCH:**
    Autoregressive conditional heteroskedasticity (time-varying variance)

**5. Multivariate:**
    VAR(1): Vector AR with multiple variables

**6. Non-Linear:**
    TAR: Threshold AR (regime switching)
    SETAR: Self-exciting threshold AR

**7. State-Space Form:**
Can write as:
    X[k+1] = F·X[k] + w[k]  (state equation)
    Y[k] = H·X[k] + v[k]    (observation equation)

Limitations
-----------

**1. Linearity:**
Real systems often non-linear.
**Solution:** TAR, neural networks, regime-switching.

**2. Constant Parameters:**
φ, σ² may vary over time.
**Solution:** Time-varying coefficient models, rolling estimation.

**3. Gaussian Assumption:**
Real innovations may have fat tails.
**Solution:** Robust estimation, Student-t innovations.

**4. No Seasonality:**
No built-in seasonal patterns.
**Solution:** Seasonal AR (SAR), seasonal differencing.

**5. Short Memory:**
Only immediate past matters.
**Solution:** Higher-order AR(p), long-memory models.

Common Pitfalls
---------------

1. **Spurious Regression:**
   - Regressing non-stationary on non-stationary
   - Can find correlation when none exists
   - Always test for stationarity first

2. **Overlooking Unit Root:**
   - φ = 1 requires different treatment
   - Standard inference breaks down
   - Use unit root tests

3. **Small Sample Bias:**
   - φ̂ biased downward in small samples
   - Use bias correction or simulation

4. **Incorrect Lag Length:**
   - Using AR(1) when AR(2) needed
   - Use information criteria (AIC, BIC)

5. **Ignoring Structural Breaks:**
   - Parameters change at breakpoints
   - Can appear as unit root
   - Test for breaks

Testing and Validation
-----------------------

**Diagnostic Checks:**

1. **Residual Analysis:**
   - ε[k] = X[k] - φ·X[k-1] should be white noise
   - Plot ACF/PACF of residuals
   - Should show no structure

2. **Portmanteau Tests:**
   - Ljung-Box test for joint autocorrelation
   - Should not reject for residuals

3. **Normality:**
   - Q-Q plot of residuals
   - Jarque-Bera test

4. **Parameter Stability:**
   - Recursive estimation
   - CUSUM test

5. **Out-of-Sample Validation:**
   - Forecast accuracy (RMSE, MAE)
   - Compare to benchmarks (random walk, mean)
"""

import numpy as np
import sympy as sp

from src.systems.base.core.discrete_stochastic_system import DiscreteStochasticSystem


class DiscreteAR1(DiscreteStochasticSystem):
    """
    First-order autoregressive process with additive noise.

    The AR(1) model is the fundamental discrete-time stochastic process
    exhibiting memory and mean reversion. It is the discrete analog of
    the Ornstein-Uhlenbeck process and the building block for ARMA and
    ARIMA models.

    Difference Equation
    -------------------
    Standard form:
        X[k+1] = φ·X[k] + σ·w[k]

    With control:
        X[k+1] = φ·X[k] + u[k] + σ·w[k]

    where:
        - X[k]: State at time k
        - φ: Autoregressive coefficient (persistence parameter)
        - u[k]: Control input (optional)
        - σ: Innovation standard deviation
        - w[k] ~ N(0,1): Standard normal white noise

    Key Features
    ------------
    **Persistence:**
    Parameter φ controls memory:
    - φ near 1: High persistence (long memory)
    - φ near 0: Low persistence (short memory)
    - φ = 0: White noise (no memory)

    **Stationarity:**
    Process stationary if and only if |φ| < 1:
    - |φ| < 1: Stable, mean-reverting
    - φ = 1: Unit root (random walk)
    - |φ| > 1: Explosive

    **Additive Noise:**
    Innovation w[k] independent of state.
    Constant variance σ².

    **Markov Property:**
    Future depends only on present, not past history.

    Mathematical Properties
    -----------------------
    **Stationary Distribution (|φ| < 1):**
    Mean: E[X[∞]] = u/(1-φ)
    Variance: Var[X[∞]] = σ²/(1-φ²)
    Distribution: N(u/(1-φ), σ²/(1-φ²))

    **Autocorrelation:**
    ρ(h) = φ^h

    Geometric decay with lag h.

    **Mean Reversion:**
    Half-life: h = ln(0.5)/ln(φ) periods
    Time constant: τ = -1/ln(φ) periods

    Physical Interpretation
    -----------------------
    **Autoregressive Coefficient φ:**
    - Dimensionless (ratio)
    - Fraction of value persisting to next period
    - Typical range: 0 to 0.95

    **Interpretation by Value:**
    - φ = 0.9: High persistence, slow decay
    - φ = 0.5: Moderate persistence
    - φ = 0.1: Low persistence, fast decay
    - φ = 1: Random walk (non-stationary)

    **Innovation Variance σ²:**
    - Units: [state]²
    - Shock variance each period
    - Stationary variance: σ²/(1-φ²)

    State Space
    -----------
    State: x ∈ ℝ
        - Unbounded (can take any real value)
        - Equilibrium: u/(1-φ) (for |φ| < 1)

    Control: u ∈ ℝ (optional)
        - Shifts equilibrium
        - External forcing

    Parameters
    ----------
    phi : float, default=0.9
        Autoregressive coefficient
        - |φ| < 1: Stationary (typical)
        - φ = 1: Unit root (special case)
        - |φ| > 1: Explosive (avoid)
        - Typical: 0.5 to 0.95

    sigma : float, default=0.1
        Innovation standard deviation (must be positive)
        - Controls noise magnitude
        - Stationary std: σ/√(1-φ²)

    dt : float, default=1.0
        Sampling period (time between observations)
        - Sets time units
        - Needed for discrete system

    Stochastic Properties
    ---------------------
    - Noise Type: ADDITIVE
    - Innovation: w[k] ~ N(0,1) iid
    - Markov: Memoryless given current state
    - Stationary: If |φ| < 1
    - Ergodic: If |φ| < 1

    Applications
    ------------
    **1. Econometrics:**
    - GDP growth (quarterly data)
    - Inflation rates (monthly data)
    - Interest rates (daily/weekly)
    - Unemployment rates

    **2. Finance:**
    - Asset returns (daily)
    - Volatility models (log-volatility)
    - Exchange rates

    **3. Signal Processing:**
    - Colored noise generation
    - Digital filtering (one-pole filter)
    - Prediction algorithms

    **4. Control Systems:**
    - Disturbance models
    - State estimation (Kalman filter)
    - Model predictive control

    **5. Time Series:**
    - Foundation for ARMA, ARIMA
    - Benchmark for forecasting

    Numerical Simulation
    --------------------
    **Exact Sampling:**
        X[k+1] = φ·X[k] + u + σ·Z[k]

    where Z[k] ~ N(0,1).

    This is exact (no discretization needed).

    **Vectorized:**
    Can use scipy.signal.lfilter for efficient generation.

    Statistical Analysis
    --------------------
    **Parameter Estimation:**
    - OLS: φ̂ = Σ X[k]·X[k+1] / Σ X[k]²
    - MLE: Same as OLS for Gaussian
    - Yule-Walker: From autocorrelations

    **Model Validation:**
    - Unit root test (Dickey-Fuller)
    - Residual diagnostics (Ljung-Box)
    - Information criteria (AIC, BIC)

    Comparison with Other Models
    -----------------------------
    **vs. White Noise:**
    - WN: φ = 0 (no persistence)
    - AR(1): φ ≠ 0 (memory)

    **vs. Random Walk:**
    - RW: φ = 1 (unit root)
    - AR(1): |φ| < 1 (stationary)

    **vs. Ornstein-Uhlenbeck:**
    - OU: Continuous time
    - AR(1): Discrete time
    - Connection: φ = e^(-α·Δt)

    Limitations
    -----------
    - Linear dynamics only
    - Constant parameters
    - Gaussian innovations
    - Short memory (one lag)

    **Extensions:**
    - AR(p): Higher-order lags
    - ARMA: Add moving average
    - GARCH: Time-varying variance
    - TAR: Threshold/regime-switching

    See Also
    --------
    OrnsteinUhlenbeck : Continuous-time analog
    DiscreteRandomWalk : Unit root case (φ=1)
    DiscreteWhiteNoise : No persistence (φ=0)
    """

    def define_system(self, phi: float = 0.9, sigma: float = 0.1, dt: float = 1.0):
        """
        Define AR(1) process dynamics.

        Sets up the difference equation:
            X[k+1] = φ·X[k] + u[k] + σ·w[k]

        Parameters
        ----------
        phi : float, default=0.9
            Autoregressive coefficient
            - |φ| < 1: Stationary (mean-reverting)
            - φ = 1: Unit root (random walk)
            - |φ| > 1: Explosive (unstable)
            - Typical: 0.5 to 0.95

        sigma : float, default=0.1
            Innovation standard deviation (must be positive)
            - Controls shock magnitude
            - Stationary std: σ/√(1-φ²)

        dt : float, default=1.0
            Sampling period [time units]
            - Required for discrete system
            - Sets time scale

        Raises
        ------
        ValueError
            If sigma ≤ 0
        UserWarning
            If |phi| ≥ 1 (non-stationary)

        Notes
        -----
        **Stationarity Condition:**
        Process is stationary if and only if |φ| < 1.

        **Critical Cases:**
        - φ = 1: Unit root (random walk)
          * Non-stationary
          * Variance grows linearly: Var[X[k]] = k·σ²
          * Requires different statistical treatment

        - φ = -1: Perfect negative autocorrelation
          * Alternates between extremes
          * Non-stationary

        - |φ| > 1: Explosive
          * Variance grows exponentially
          * Diverges to ±∞

        **Stationary Properties:**
        For |φ| < 1:
        - Mean: μ = u/(1-φ)
        - Variance: γ(0) = σ²/(1-φ²)
        - Autocorrelation: ρ(h) = φ^h
        - Half-life: ln(0.5)/ln(φ) periods

        **Relationship to Continuous Time:**
        If discretizing OU process with parameter α:
            φ = exp(-α·dt)
            σ ≈ σ_continuous·√(2α·dt) (approximate)

        **Parameter Selection:**
        - High persistence (φ ≈ 0.9): Financial returns, macro data
        - Moderate (φ ≈ 0.5): GDP growth, some commodities
        - Low (φ ≈ 0.1): Nearly white noise

        **Innovation Variance:**
        Total variance decomposes:
        - Stationary variance: σ²/(1-φ²)
        - Innovation variance: σ²
        - Ratio: 1/(1-φ²) ≥ 1
        """
        # Validate stationarity
        if abs(phi) >= 1:
            import warnings
            if phi == 1:
                warnings.warn(
                    "phi = 1 creates unit root (random walk). "
                    "Process is non-stationary with linearly growing variance. "
                    "Consider using DiscreteRandomWalk for clarity.",
                    UserWarning,
                )
            else:
                warnings.warn(
                    f"|phi| = {abs(phi)} >= 1 creates non-stationary process. "
                    f"For |phi| > 1, process is explosive (unstable). "
                    f"For stationary AR(1), use |phi| < 1.",
                    UserWarning,
                )

        if sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}")

        # Define symbolic variables
        x = sp.symbols("x", real=True)
        u = sp.symbols("u", real=True)
        phi_sym = sp.symbols("phi", real=True)
        sigma_sym = sp.symbols("sigma", positive=True)

        # System definition
        self.state_vars = [x]
        self.control_vars = [u]

        # AR(1) dynamics: X[k+1] = φ·X[k] + u[k]
        self._f_sym = sp.Matrix([[phi_sym * x + u]])

        # Additive noise: w[k] ~ N(0,1)
        self.diffusion_expr = sp.Matrix([[sigma_sym]])

        self.parameters = {phi_sym: phi, sigma_sym: sigma}
        self.order = 1
        self._dt = dt  # Required for discrete system
        self.sde_type = "ito"  # Discrete analog

    def get_stationary_variance(self) -> float:
        """
        Get theoretical stationary variance σ²/(1-φ²).

        Only valid for |φ| < 1.

        Returns
        -------
        float
            Stationary variance

        Raises
        ------
        ValueError
            If |φ| ≥ 1 (non-stationary)
        """
        phi = next(v for k, v in self.parameters.items() if str(k) == "phi")
        sigma = next(v for k, v in self.parameters.items() if str(k) == "sigma")

        if abs(phi) >= 1:
            raise ValueError(
                f"Stationary variance undefined for |phi| = {abs(phi)} >= 1. "
                "Process is non-stationary."
            )

        return sigma**2 / (1 - phi**2)

    def get_half_life(self) -> float:
        """
        Get half-life: number of periods to reduce deviation by 50%.

        Formula: h = ln(0.5) / ln(φ)

        Only meaningful for 0 < φ < 1.

        Returns
        -------
        float
            Half-life [periods]
        """
        phi = next(v for k, v in self.parameters.items() if str(k) == "phi")

        if phi <= 0 or phi >= 1:
            raise ValueError(
                f"Half-life only meaningful for 0 < phi < 1, got phi = {phi}"
            )

        return np.log(0.5) / np.log(phi)