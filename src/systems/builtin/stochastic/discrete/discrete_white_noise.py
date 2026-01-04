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
Discrete White Noise - Pure Random Process
===========================================

This module provides implementations of discrete white noise, the fundamental
building block of all stochastic processes. White noise represents pure
randomness with no memory or structure, serving as:

- The simplest non-trivial stochastic process
- The innovation sequence for all ARMA/ARIMA models
- The baseline for testing signal detection algorithms
- The null hypothesis for time series independence tests
- The idealization of measurement noise and random disturbances

White noise is called "white" by analogy with white light, which contains
all frequencies in equal proportion. Similarly, white noise has flat power
spectral density across all frequencies.

Mathematical Background
-----------------------
White noise is the discrete-time analog of the Dirac delta function in
continuous time. It represents the derivative (in a distributional sense)
of Brownian motion.

**Physical Origins:**
White noise arises from:
- Thermal fluctuations (Johnson-Nyquist noise)
- Quantum fluctuations (shot noise)
- Chaotic dynamics in high-dimensional systems
- Aggregation of many independent small effects (Central Limit Theorem)

Mathematical Definition
-----------------------

**Standard Form:**
    X[k] = σ·w[k]

where w[k] ~ N(0,1) are iid standard normal random variables.

**Key Properties:**
1. **Zero Mean:** E[X[k]] = 0 for all k
2. **Constant Variance:** Var[X[k]] = σ² for all k
3. **Independence:** X[k] ⊥ X[j] for k ≠ j
4. **Stationarity:** Statistical properties constant over time
5. **Ergodicity:** Time averages equal ensemble averages

**Equivalent Characterizations:**

1. **Via Moments:**
   E[X[k]] = 0
   E[X[k]·X[j]] = σ²·δ_{kj} (Kronecker delta)

2. **Via Autocorrelation:**
   γ(h) = σ²·δ_h  (zero except at lag 0)

3. **Via Power Spectrum:**
   S(f) = σ²  (flat across all frequencies)

4. **Via Innovations:**
   Cannot be predicted from past values:
   E[X[k] | X[k-1], X[k-2], ...] = 0

Relationship to Other Processes
--------------------------------

**White Noise as Foundation:**
All stationary processes can be written as filtered white noise
(Wold decomposition):
    X[k] = μ + Σ ψ_j·w[k-j]

Special cases:
- MA(q): Finite filter
- AR(p): Infinite filter (recursive)
- ARMA(p,q): Rational filter

**Limiting Cases:**
- AR(1) with φ = 0 → White noise
- Random walk differences → White noise
- Brownian motion increments → White noise (continuous analog)

**Integration:**
Cumulative sum of white noise:
    Y[k] = Σ_{j=0}^k X[j]

produces random walk (Brownian motion analog).

Key Properties
--------------

**1. Memorylessness:**
No autocorrelation: Cov[X[k], X[j]] = 0 for k ≠ j

This is the defining property - knowing past values provides
no information about future values.

**2. Unpredictability:**
Best predictor of X[k+1] given past: E[X[k+1] | past] = 0

Mean squared prediction error: E[(X[k+1] - X̂[k+1])²] = σ²

Cannot be reduced by using past information.

**3. Stationarity:**
Strictly stationary: Joint distribution invariant to time shifts
Weakly stationary: Mean and autocovariance time-invariant

**4. Ergodicity:**
Sample mean converges to population mean:
    (1/N)·Σ X[k] → 0 as N → ∞

Sample variance converges to σ².

**5. Gaussian (if Assumed):**
If w[k] ~ N(0,1), then:
- X[k] ~ N(0, σ²)
- Any linear combination is Gaussian
- Uncorrelated implies independent

**6. Flat Power Spectrum:**
Power spectral density:
    S(f) = σ² for all f ∈ [-1/(2Δt), 1/(2Δt)]

Equal power at all frequencies (hence "white").

**7. Infinite Bandwidth:**
Contains all frequencies up to Nyquist limit.

**8. Maximum Entropy:**
Among all processes with variance σ², white Gaussian noise
has maximum entropy: H = (1/2)·ln(2πe·σ²)

Physical and Statistical Interpretation
----------------------------------------

**Standard Deviation σ:**
- Dimension: [state]
- Interpretation: Scale of fluctuations
- Typical range: Application-dependent

**Examples:**
- Measurement noise: σ = instrument precision
- Financial returns: σ = daily volatility
- Communication: σ = channel noise level

**Signal-to-Noise Ratio:**
For signal S[k] corrupted by noise:
    Y[k] = S[k] + X[k]

SNR = Var[S] / σ²

High SNR: Signal dominates, easy detection
Low SNR: Noise dominates, difficult detection

**Bandwidth:**
White noise has infinite bandwidth (all frequencies present).
Real systems have finite bandwidth (band-limited noise).

**Thermal Noise:**
Johnson-Nyquist noise in resistor:
    σ² = 4·k_B·T·R·Δf

where:
- k_B: Boltzmann constant
- T: Temperature
- R: Resistance
- Δf: Bandwidth

Applications
------------

**1. Signal Processing:**

**Noise Modeling:**
Background noise in measurements:
    Observation = Signal + White noise

**Filter Testing:**
Test filter performance with known input spectrum.

**System Identification:**
Excite system with white noise to estimate transfer function.

**Detection Theory:**
Optimal detector design assumes white Gaussian noise.

**2. Communications:**

**Channel Noise:**
AWGN (Additive White Gaussian Noise) channel:
    Received = Transmitted + Noise

Foundation of Shannon's channel capacity.

**Coding Theory:**
Design error-correcting codes assuming white noise.

**3. Control Systems:**

**Process Noise:**
Stochastic disturbances in state-space models:
    x[k+1] = A·x[k] + B·u[k] + w[k]

**Kalman Filter:**
Assumes white noise for process and measurement:
    x[k+1] = F·x[k] + w[k]
    y[k] = H·x[k] + v[k]

**4. Econometrics & Finance:**

**Innovation Sequence:**
Unpredictable component of time series (ARMA residuals).

**Efficient Markets:**
If markets efficient, returns should be white noise
(all information already in price).

**Risk Models:**
Baseline for volatility estimation and stress testing.

**5. Simulation & Monte Carlo:**

**Random Number Generation:**
Generate white noise as basis for other distributions.

**Bootstrap Methods:**
Resample residuals (assumed white noise).

**6. Testing:**

**Benchmark:**
Test signal detection against white noise null hypothesis.

**Portmanteau Tests:**
Ljung-Box test checks if residuals are white noise.

**Randomness Tests:**
Runs test, serial correlation test.

Statistical Analysis
--------------------

**Sample Statistics:**
Given observations X[0], X[1], ..., X[N-1]:

Sample Mean:
    μ̂ = (1/N)·Σ X[k]
    E[μ̂] = 0
    Var[μ̂] = σ²/N

Sample Variance:
    σ̂² = (1/N)·Σ (X[k] - μ̂)²
    E[σ̂²] ≈ σ² (slightly biased for finite N)

Sample Autocorrelation (lag h):
    ρ̂(h) = (Σ X[k]·X[k+h]) / (Σ X[k]²)
    E[ρ̂(h)] ≈ 0 for h ≠ 0
    Var[ρ̂(h)] ≈ 1/N

**Hypothesis Testing:**

1. **Test for White Noise:**
   - H₀: Process is white noise
   - Use Ljung-Box Q-statistic
   - Tests joint significance of autocorrelations

2. **Test for Independence:**
   - Runs test
   - Turning points test
   - Serial correlation test

3. **Test for Normality:**
   - Jarque-Bera test
   - Shapiro-Wilk test
   - Q-Q plot

**Confidence Intervals:**
For autocorrelation under white noise null:
    ρ̂(h) ± 1.96/√N  (95% CI)

Values outside indicate non-white noise.

Numerical Simulation
--------------------

**Direct Generation:**
    X[k] = σ·randn()

where randn() generates N(0,1).

**Efficient Implementation:**
Vectorized generation of N samples:
    X = σ * np.random.randn(N)

**Quality Checks:**
1. Mean ≈ 0: |μ̂| < 3σ/√N
2. Variance ≈ σ²: Check sample variance
3. Autocorrelation ≈ 0: |ρ̂(h)| < 1.96/√N for h > 0
4. Normality: Q-Q plot, formal tests

**Random Number Generator:**
Use high-quality RNG:
- Mersenne Twister (default in NumPy)
- PCG (better statistical properties)
- Cryptographic RNG (if security needed)
Comparison with Other Processes
--------------------------------

**vs. Colored Noise:**
- White noise: Flat spectrum
- Colored noise: Non-flat spectrum (e.g., pink, brown)
- White is special case (no filtering)

**vs. Random Walk:**
- Random walk: X[k] = X[k-1] + w[k] (integrated white noise)
- White noise: X[k] = w[k] (no integration)
- Random walk has memory, white noise doesn't

**vs. AR(1):**
- AR(1): X[k] = φ·X[k-1] + w[k] (filtered white noise)
- White noise: AR(1) with φ = 0
- White noise is AR(1) limit

**vs. Brownian Motion:**
- Brownian motion: Continuous-time cumulative white noise
- White noise: Discrete-time, memoryless
- dW/dt is white noise (in distribution sense)

**vs. Poisson Process:**
- Poisson: Discrete events, integer-valued
- White noise: Continuous-valued, Gaussian
- Different probabilistic structure

Extensions and Generalizations
-------------------------------

**1. Colored Noise:**
Filtered white noise with specific spectrum:
- Pink noise: 1/f spectrum (natural phenomena)
- Brown noise: 1/f² spectrum (integrated white)
- Blue noise: f spectrum (rare)

**2. Non-Gaussian White Noise:**
Independent but not Gaussian:
- Uniform white noise
- Laplace white noise (heavier tails)
- Student-t white noise (fat tails)

**3. Conditional Heteroskedastic:**
Variance depends on past (ARCH/GARCH):
    X[k] = σ[k]·w[k]
where σ²[k] = f(X[k-1], ..., σ²[k-1], ...)

**4. Multivariate:**
Vector white noise:
    X[k] ~ N(0, Σ)
with covariance matrix Σ.

**5. Complex White Noise:**
For communication systems:
    Z[k] = X[k] + i·Y[k]
where X[k], Y[k] are independent white noise.

**6. Band-Limited:**
White noise filtered to specific frequency band.

Limitations
-----------

**1. Idealization:**
Real noise always has finite bandwidth (not truly white).
True white noise has infinite power.

**2. Gaussian Assumption:**
Real noise may have:
- Fat tails (outliers more common)
- Skewness
- Time-varying variance

**3. Independence:**
Real sequences may have weak dependence not detected by
standard tests (long-range dependence).

**4. Stationarity:**
Real noise may have:
- Trends
- Structural breaks
- Time-varying statistics

Common Pitfalls
---------------

1. **Confusing White Noise with IID:**
   - White noise: E[X[k]·X[j]] = 0 for k ≠ j
   - IID: Independent and identically distributed
   - IID ⟹ white noise
   - White noise ⟹ IID only if Gaussian

2. **Assuming Gaussianity:**
   - White noise need not be Gaussian
   - Many tests assume Gaussian
   - Check normality separately

3. **Ignoring Finite Sample Effects:**
   - Sample autocorrelations not exactly zero
   - Use confidence bands: ±1.96/√N

4. **Over-Testing:**
   - With 20 lags, expect 1 to exceed 95% CI by chance
   - Use joint tests (Ljung-Box) or adjust significance

5. **Treating Filtered Noise as White:**
   - After filtering, noise is colored
   - Need to account for filter in analysis

Testing and Validation
-----------------------

**Statistical Tests:**

1. **Ljung-Box Q-Test:**
   Tests joint significance of first m autocorrelations:
   Q = N(N+2)·Σ_{h=1}^m ρ̂²(h)/(N-h)
   
   Under H₀ (white noise): Q ~ χ²(m)

2. **Durbin-Watson:**
   Tests first-order autocorrelation:
   DW = Σ(X[k] - X[k-1])² / Σ X[k]²
   
   DW ≈ 2 indicates white noise

3. **Runs Test:**
   Tests randomness via number of runs above/below mean.

4. **Spectral Test:**
   Periodogram should be approximately flat.

**Visual Diagnostics:**
- Time series plot: Should look random
- ACF plot: Should be zero except lag 0
- Periodogram: Should be flat
- Q-Q plot: Should be linear (if Gaussian)

**Monte Carlo Validation:**
Generate N sequences, check:
- Fraction of ρ̂(h) in confidence band ≈ 95%
- Distribution of Q-statistic matches χ²
- Power spectrum approximately flat
"""

import numpy as np
import sympy as sp

from src.systems.base.core.discrete_stochastic_system import DiscreteStochasticSystem


class DiscreteWhiteNoise(DiscreteStochasticSystem):
    """
    Pure white noise process - memoryless random sequence.

    The fundamental building block of all stochastic processes, representing
    pure randomness with no memory or structure. White noise is the innovation
    sequence underlying ARMA models and the limiting case of AR(1) with φ = 0.

    Difference Equation
    -------------------
    X[k] = σ·w[k]

    where w[k] ~ N(0,1) are iid standard normal random variables.

    **Alternative View:**
    X[k+1] = 0·X[k] + σ·w[k]

    This shows white noise as AR(0) - no dependence on past.

    Key Features
    ------------
    **Memorylessness:**
    Zero autocorrelation: Cov[X[k], X[j]] = 0 for k ≠ j

    Each observation independent of all others.

    **Unpredictability:**
    Best prediction: E[X[k+1] | past] = 0
    Prediction error: MSE = σ²

    Cannot be improved by using past information.

    **Stationarity:**
    Strictly stationary - distribution invariant to time shifts.

    **Flat Spectrum:**
    Power spectral density: S(f) = σ² (constant across frequencies)

    Hence "white" - equal power at all frequencies.

    Mathematical Properties
    -----------------------
    **Moments:**
    Mean: E[X[k]] = 0 for all k
    Variance: Var[X[k]] = σ² for all k
    Higher moments: Standard Gaussian if w ~ N(0,1)

    **Autocorrelation:**
    γ(h) = σ²·δ_h where δ_h is Kronecker delta:
        γ(0) = σ²
        γ(h) = 0 for h ≠ 0

    **Power Spectrum:**
    S(f) = σ² for all f ∈ [-f_s/2, f_s/2]

    where f_s = 1/Δt is sampling frequency.

    Physical Interpretation
    -----------------------
    **Standard Deviation σ:**
    - Units: [state]
    - Scale of random fluctuations
    - Examples:
      * Measurement noise: instrument precision
      * Thermal noise: √(4·k_B·T·R·Δf)
      * Financial: daily return volatility

    **No Parameters Besides σ:**
    White noise is completely characterized by variance σ².
    No memory, no time constants, no structure.

    State Space
    -----------
    State: x ∈ ℝ (unbounded)
        - No persistence between samples
        - Each value independent
        - Gaussian: X[k] ~ N(0, σ²)

    Control: None (autonomous)
        - Pure random process
        - Cannot be controlled or predicted

    Parameters
    ----------
    sigma : float, default=1.0
        Standard deviation of white noise
        - Must be positive
        - σ = 1: Standard white noise
        - Variance: σ²

    dt : float, default=1.0
        Sampling period [time units]
        - Required for discrete system
        - Sets time scale

    Stochastic Properties
    ---------------------
    - Type: Pure random (no deterministic component)
    - Innovation: w[k] ~ N(0,1) iid
    - Memory: None (memoryless)
    - Stationary: Yes (strictly stationary)
    - Ergodic: Yes
    - Predictable: No

    Applications
    ------------
    **1. Signal Processing:**
    - Noise modeling in measurements
    - Filter testing and validation
    - System identification (input signal)
    - Detection theory (null hypothesis)

    **2. Communications:**
    - AWGN channel model
    - Error analysis
    - Channel capacity calculations

    **3. Time Series:**
    - Innovation sequence for ARMA models
    - Residual diagnostics (should be white)
    - Benchmark for forecasting performance

    **4. Control Systems:**
    - Process noise in Kalman filter
    - Disturbance modeling
    - Robustness analysis

    **5. Finance:**
    - Efficient market hypothesis (returns should be white)
    - Monte Carlo simulation
    - Risk modeling baseline

    **6. Testing:**
    - Null hypothesis for independence tests
    - Benchmark for signal detection
    - Residual analysis

    Numerical Simulation
    --------------------
    **Direct Generation:**
        X[k] = σ·randn()

    Simple, exact, no approximation.

    **Quality Checks:**
    - Mean ≈ 0
    - Variance ≈ σ²
    - Autocorrelation ≈ 0 for h > 0
    - Within confidence bands: ±1.96/√N

    Statistical Analysis
    --------------------
    **Tests for White Noise:**
    - Ljung-Box Q-test (joint autocorrelation)
    - Durbin-Watson (first-order autocorrelation)
    - Runs test (randomness)
    - Portmanteau test

    **Diagnostics:**
    - ACF plot: Only lag 0 significant
    - Periodogram: Approximately flat
    - Q-Q plot: Linear (if Gaussian)

    Comparison with Other Processes
    --------------------------------
    **vs. Random Walk:**
    - RW: Cumulative sum of white noise
    - WN: Memoryless, stationary

    **vs. AR(1):**
    - AR(1): X[k] = φ·X[k-1] + w[k]
    - WN: AR(1) with φ = 0

    **vs. Brownian Motion:**
    - BM: Continuous-time integral of white noise
    - WN: Discrete-time, no integration

    Limitations
    -----------
    - Idealization (infinite bandwidth)
    - Real noise often band-limited
    - May have weak temporal dependence
    - Gaussian assumption may not hold

    See Also
    --------
    DiscreteAR1 : White noise is AR(1) with φ=0
    DiscreteRandomWalk : Cumulative sum of white noise
    BrownianMotion : Continuous-time analog
    """

    def define_system(self, sigma: float = 1.0, dt: float = 1.0):
        """
        Define white noise process.

        Sets up pure random sequence:
            X[k] = σ·w[k]

        where w[k] ~ N(0,1) are iid.

        Parameters
        ----------
        sigma : float, default=1.0
            Standard deviation (must be positive)
            - σ = 1: Standard white noise
            - σ² = variance
            - Typical: 0.1 to 10.0 depending on application

        dt : float, default=1.0
            Sampling period [time units]
            - Required for discrete system
            - Sets time scale
            - Does not affect statistics (memoryless)

        Raises
        ------
        ValueError
            If sigma ≤ 0

        Notes
        -----
        **Complete Characterization:**
        White noise is completely specified by variance σ².
        No other parameters needed - no memory, no time constants.

        **Independence:**
        Each sample is independent:
            P(X[k] | X[k-1], ...) = P(X[k])

        This is the strongest form of memorylessness.

        **Gaussian White Noise:**
        If w[k] ~ N(0,1), then X[k] ~ N(0, σ²).
        Uncorrelated + Gaussian ⟹ Independent

        **Non-Gaussian Extensions:**
        Can use other distributions:
        - Uniform: w[k] ~ U(-√3, √3) (variance 1)
        - Laplace: Heavier tails
        - Student-t: Fat tails

        **Sampling Period dt:**
        Unlike AR(1) or OU, dt doesn't affect white noise statistics.
        Samples at any interval are independent with same variance.

        **Power Spectral Density:**
        S(f) = σ² for |f| ≤ f_Nyquist = 1/(2·dt)

        Flat spectrum - equal power at all frequencies.

        **Use Cases:**
        - Testing: Null hypothesis for independence
        - Simulation: Generate innovations for ARMA
        - Benchmarking: Compare signal detection performance
        - Modeling: Measurement noise, residuals

        **Relationship to AR(1):**
        White noise is AR(1) with φ = 0:
            X[k+1] = 0·X[k] + σ·w[k]

        **Wold Decomposition:**
        Any stationary process can be written as:
            X[k] = μ + Σ ψ_j·w[k-j]
        where w[k] is white noise.
        """
        if sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}")

        # Define symbolic variables
        x = sp.symbols("x", real=True)
        sigma_sym = sp.symbols("sigma", positive=True)

        # System definition
        self.state_vars = [x]
        self.control_vars = []  # Autonomous (no control)

        # No deterministic evolution (zero drift)
        # X[k+1] = 0·X[k] + σ·w[k]
        self._f_sym = sp.Matrix([[0]])

        # Pure noise (additive)
        self.diffusion_expr = sp.Matrix([[sigma_sym]])

        self.parameters = {sigma_sym: sigma}
        self.order = 1
        self._dt = dt  # Required for discrete system
        self.sde_type = "ito"  # Discrete analog

    def get_variance(self) -> float:
        """
        Get theoretical variance σ².

        For white noise, variance is simply σ².

        Returns
        -------
        float
            Variance

        Notes
        -----
        Unlike AR(1) or OU, white noise variance doesn't depend
        on any time constants - it's just σ².
        """
        sigma = next(v for k, v in self.parameters.items() if str(k) == "sigma")
        return sigma**2

    def get_standard_deviation(self) -> float:
        """
        Get theoretical standard deviation σ.

        Returns
        -------
        float
            Standard deviation

        Examples
        --------
        >>> wn = DiscreteWhiteNoise(sigma=1.5, dt=1.0)
        >>> std = wn.get_standard_deviation()
        >>> print(f"Std: {std:.3f}")  # 1.5
        """
        return next(v for k, v in self.parameters.items() if str(k) == "sigma")

    def get_autocorrelation(self, lag: int) -> float:
        """
        Get theoretical autocorrelation at given lag.

        For white noise:
            ρ(0) = 1
            ρ(h) = 0 for h ≠ 0

        Parameters
        ----------
        lag : int
            Lag h (non-negative)

        Returns
        -------
        float
            Autocorrelation: 1 if lag=0, else 0

        Examples
        --------
        >>> wn = DiscreteWhiteNoise(sigma=1.0, dt=1.0)
        >>> print(f"ρ(0) = {wn.get_autocorrelation(0)}")  # 1.0
        >>> print(f"ρ(1) = {wn.get_autocorrelation(1)}")  # 0.0
        >>> print(f"ρ(10) = {wn.get_autocorrelation(10)}")  # 0.0
        """
        return 1.0 if lag == 0 else 0.0


# ============================================================================
# Convenience Factory Functions
# ============================================================================


def create_standard_white_noise(dt: float = 1.0) -> DiscreteWhiteNoise:
    """
    Create standard white noise with unit variance.

    Standard white noise: X[k] ~ N(0, 1)

    Parameters
    ----------
    dt : float, default=1.0
        Sampling period

    Returns
    -------
    DiscreteWhiteNoise
        Unit variance white noise

    Examples
    --------
    >>> wn = create_standard_white_noise(dt=1.0)
    >>> print("Mean: 0, Variance: 1")
    """
    return DiscreteWhiteNoise(sigma=1.0, dt=dt)


def create_measurement_noise(precision: float, dt: float = 1.0) -> DiscreteWhiteNoise:
    """
    Create white noise model for measurement error.

    For sensor with precision (standard deviation) p,
    measurement error modeled as white noise with σ = p.

    Parameters
    ----------
    precision : float
        Measurement precision (standard deviation)
        - Example: precision = 0.01 for 1% sensor

    dt : float, default=1.0
        Sampling period

    Returns
    -------
    DiscreteWhiteNoise
        Measurement noise model

    Examples
    --------
    >>> # 1 cm precision for position sensor
    >>> pos_noise = create_measurement_noise(precision=0.01, dt=0.1)
    >>> 
    >>> # 0.1 degree precision for angle sensor
    >>> angle_noise = create_measurement_noise(
    ...     precision=0.1*np.pi/180,  # Convert to radians
    ...     dt=0.01
    ... )
    """
    return DiscreteWhiteNoise(sigma=precision, dt=dt)


def create_thermal_noise(
    temperature: float,
    resistance: float,
    bandwidth: float,
    dt: float = 1.0
) -> DiscreteWhiteNoise:
    """
    Create Johnson-Nyquist thermal noise model.

    Thermal noise in resistor:
        V_rms = √(4·k_B·T·R·Δf)

    Parameters
    ----------
    temperature : float
        Temperature [K]
    resistance : float
        Resistance [Ω]
    bandwidth : float
        Bandwidth [Hz]
    dt : float, default=1.0
        Sampling period [s]

    Returns
    -------
    DiscreteWhiteNoise
        Thermal noise model

    Notes
    -----
    Boltzmann constant: k_B = 1.380649e-23 J/K

    Examples
    --------
    >>> # Room temperature (300K), 1kΩ resistor, 1MHz bandwidth
    >>> thermal = create_thermal_noise(
    ...     temperature=300,
    ...     resistance=1000,
    ...     bandwidth=1e6,
    ...     dt=1e-6
    ... )
    """
    k_B = 1.380649e-23  # Boltzmann constant [J/K]
    sigma = np.sqrt(4 * k_B * temperature * resistance * bandwidth)
    return DiscreteWhiteNoise(sigma=sigma, dt=dt)