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
Cox-Ingersoll-Ross Process - Positive Mean-Reverting Stochastic System
=======================================================================

This module provides implementations of the Cox-Ingersoll-Ross (CIR) process,
a fundamental mean-reverting stochastic differential equation that ensures
state positivity. The CIR process is distinguished by:

- The canonical model for positive-valued mean-reverting processes
- Foundation of modern interest rate modeling (CIR term structure model, 1985)
- Solution to the negativity problem in the Vasicek model
- Explicit stationary distribution (non-central chi-squared scaled)
- Analytical bond pricing formulas in finance
- Connection to Bessel processes and squared Ornstein-Uhlenbeck

The CIR process represents the continuous-time evolution of quantities that:
1. Must remain positive (interest rates, volatility, intensity)
2. Exhibit mean reversion (return to long-term average)
3. Have volatility proportional to square root of level

Mathematical Background
-----------------------
The Cox-Ingersoll-Ross model was introduced in 1985 by John Cox, Jonathan
Ingersoll, and Stephen Ross in their seminal paper "A Theory of the Term
Structure of Interest Rates." It solved a fundamental problem:

**The Vasicek Problem:**
Vasicek (1977): dr = κ(θ - r)·dt + σ·dW
- Mean-reverting ✓
- Tractable ✓
- Can be negative ✗ (interest rates should be positive)

**The CIR Solution:**
    dr = κ(θ - r)·dt + σ·√r·dW
    
Key innovation: Multiplicative noise σ·√r ensures:
- Positivity: r(t) ≥ 0 for all t (if Feller condition holds)
- Larger noise at higher levels (realistic for rates)
- Noise vanishes as r → 0 (boundary unattainable)

**Physical Interpretation:**
CIR naturally arises in:
- Financial: Interest rates, credit spreads, stochastic volatility (Heston)
- Physics: Squared Bessel process (radial Brownian motion)
- Biology: Population dynamics with demographic stochasticity
- Queueing: Queue length with state-dependent service

Mathematical Formulation
------------------------

**Standard Form (Centered):**
    dx = -κ·x·dt + σ·√x·dW

Mean-reverting to zero with multiplicative noise.

**General Form (Non-Centered):**
    dr = κ·(θ - r)·dt + σ·√r·dW

where:
    - r(t): State (interest rate, volatility, intensity) ≥ 0
    - κ > 0: Mean reversion speed (dimension: 1/time)
    - θ > 0: Long-term mean (equilibrium level)
    - σ > 0: Volatility (noise intensity, dimension: 1/√time)
    - √r: Square-root diffusion (ensures positivity)
    - W(t): Standard Wiener process

**With Control:**
    dr = κ·(θ - r) + u)·dt + σ·√r·dW

Control u shifts the target equilibrium.

**Key Feature: Square-Root Diffusion**

The √r term is crucial:
- High r: Large noise (volatility increases with level)
- Low r: Small noise (volatility decreases near zero)
- r = 0: Zero noise (boundary is "sticky" but unattainable)

This creates a natural barrier at zero.

Feller Condition and Positivity
--------------------------------

**Feller Condition:**

If **2κθ ≥ σ²**, then r(t) > 0 for all t (strictly positive).

If **2κθ < σ²**, then r(t) can reach zero and stay there (absorbing boundary).

**Practical Implications:**

Good regime (2κθ ≥ σ²):
- Interest rate stays positive
- Well-defined for all time
- No special treatment needed

Bad regime (2κθ < σ²):
- Can hit zero and get stuck
- Need boundary conditions
- Problematic for interest rate modeling

**Parameter Selection:**
Always ensure 2κθ ≥ σ² for financial applications.

Example: θ = 0.05, κ = 0.5
    → Need σ² ≤ 2·0.5·0.05 = 0.05
    → σ ≤ 0.224

Analytical Properties
---------------------

**Exact Solution:**
No closed-form solution in general, but transition density is known.

**Moments (Starting at r₀):**

Mean:
    E[r(t)] = r₀·e^(-κt) + θ·(1 - e^(-κt))

Exponentially approaches long-term mean θ.

Variance:
    Var[r(t)] = r₀·(σ²/κ)·(e^(-κt) - e^(-2κt)) + θ·(σ²/2κ)·(1 - e^(-κt))²

**Asymptotic (t → ∞):**

Mean: E[r(∞)] = θ
Variance: Var[r(∞)] = θ·σ²/(2κ)

**Stationary Distribution:**

If Feller condition holds, r(∞) follows a **Gamma distribution**:
    r(∞) ~ Gamma(α, β)

where:
    α = 2κθ/σ²  (shape parameter)
    β = 2κ/σ²   (rate parameter)

Mean: E[r] = α/β = θ
Variance: Var[r] = α/β² = θ·σ²/(2κ)

Probability density:
    p(r) = (β^α/Γ(α))·r^(α-1)·exp(-β·r)

**Transition Density:**

The conditional density p(r_t | r_s) is a non-central chi-squared
distribution (scaled):
    2c·r_t | r_s ~ χ²(d, λ)

where:
- d = 4κθ/σ² (degrees of freedom)
- λ = 2c·r_s·e^(-κ(t-s)) (non-centrality parameter)  
- c = 2κ/(σ²(1 - e^(-κ(t-s))))

This allows exact sampling (no discretization error)!

**Autocorrelation:**

For stationary process:
    Cov[r(t), r(t+τ)] = (θ·σ²/2κ)·e^(-κ·τ)

Exponential decay (same as OU process).

Key Properties
--------------

**1. Positivity:**
r(t) ≥ 0 for all t (if Feller condition satisfied).

**2. Mean Reversion:**
Drift -κ(r - θ) pulls toward long-term mean θ.

**3. Multiplicative Noise:**
Diffusion σ·√r scales with state:
- Higher r → higher volatility (realistic)
- Lower r → lower volatility
- r = 0 → zero volatility (boundary inaccessible)

**4. Stationary Distribution:**
Gamma distribution (explicit, known moments).

**5. Exact Sampling:**
Transition density known → can sample exactly.

**6. Affine Structure:**
E[r(t)|r(s)] and Var[r(t)|r(s)] are affine in r(s).
Enables analytical bond pricing.

**7. Markov Property:**
Future independent of past given present.

**8. Connection to Bessel Process:**
√r follows a Bessel process (radial Brownian motion).

Physical and Financial Interpretation
--------------------------------------

**Mean Reversion Rate κ:**
- Dimension: [1/time]
- Controls speed of return to θ
- Time constant: τ = 1/κ
- Half-life: t₁/₂ = ln(2)/κ

**Examples:**
- κ = 0.1: Slow reversion (τ = 10 years for interest rates)
- κ = 0.5: Moderate (τ = 2 years)
- κ = 2.0: Fast (τ = 6 months)

**Long-Term Mean θ:**
- Dimension: Same as state [%/100 for rates]
- Equilibrium level
- Economic interpretation: long-run average interest rate

**Examples:**
- θ = 0.05: 5% long-term rate
- θ = 0.03: 3% (low rate environment)

**Volatility σ:**
- Dimension: [1/√time]
- Controls noise intensity
- Stationary std: √(θ·σ²/2κ)

**Examples:**
- σ = 0.1: Low volatility (10%)
- σ = 0.2: Moderate (typical)

**Square-Root Diffusion:**

The √r term creates realistic behavior:
- High rates: More volatile (uncertainty proportional to level)
- Low rates: Less volatile (near-zero rates stabilize)
- Zero rate: No volatility (absorbing if Feller violated)

**Economic Intuition:**
When rates are high (r = 10%), 1% absolute change is plausible.
When rates are low (r = 1%), 1% absolute change is implausible.
The √r scaling captures this naturally.

Applications
------------

**1. Mathematical Finance:**

**Interest Rate Modeling (Original Application):**
    dr = κ·(θ - r)·dt + σ·√r·dW

CIR model for short-term interest rate:
- Positive rates (realistic)
- Mean reversion (rates don't wander arbitrarily)
- Analytical bond pricing (closed-form formulas)

**Bond Pricing:**
Zero-coupon bond price: P(t,T) = A(t,T)·exp(-B(t,T)·r(t))

where A, B have closed-form expressions (affine structure).

**Term Structure:**
Yield curve: R(t,T) = -ln(P(t,T))/(T-t)

Captured by CIR model with analytical formulas.

**Stochastic Volatility (Heston Model):**
    dS = μ·S·dt + √V·S·dW_S
    dV = κ·(θ - V)·dt + σ·√V·dW_V

Variance V follows CIR process → ensures V ≥ 0.

**Credit Spreads:**
Intensity of default follows CIR (positive, mean-reverting).

**2. Econometrics:**
- Commodity prices (mean-reverting with positive constraint)
- Inflation rates (non-negative)
- Unemployment rates (bounded below)

**3. Physics:**

**Squared Bessel Process:**
If X is Bessel process: X² follows CIR.
Application: Radial Brownian motion in higher dimensions.

**4. Biology:**

**Population Dynamics:**
Population size with demographic stochasticity:
    dN = r·N·(1 - N/K)·dt + σ·√N·dW

Rescaled, becomes CIR-like. Ensures N ≥ 0.

**5. Queueing Theory:**

Queue length with state-dependent arrival/service.

Comparison with Vasicek (OU)
-----------------------------

**Vasicek:**
    dr = κ·(θ - r)·dt + σ·dW

**CIR:**
    dr = κ·(θ - r)·dt + σ·√r·dW

**Key Differences:**

| Feature | Vasicek | CIR |
|---------|---------|-----|
| Noise | Additive (constant) | Multiplicative (√r) |
| Positivity | Can be negative | Always positive* |
| Distribution | Gaussian | Gamma (stationary) |
| Volatility | Constant | Increases with level |
| Complexity | Simple | Moderate |

*If Feller condition holds: 2κθ ≥ σ²

**When to Use Each:**

Vasicek:
- Spreads (can be negative)
- Simplicity preferred
- Linear analysis needed

CIR:
- Interest rates (must be positive)
- Volatility (Heston model)
- Realistic volatility scaling

Numerical Simulation
--------------------

**Challenge: Square-Root Diffusion**

Standard Euler-Maruyama:
    r[k+1] = r[k] + κ·(θ - r[k])·Δt + σ·√(r[k])·√Δt·Z[k]

**Problem:** Can produce r[k+1] < 0 if Z[k] is large negative!

**Solutions:**

1. **Reflect at Zero:**
   If r[k+1] < 0: set r[k+1] = |r[k+1]|
   Simple but biased.

2. **Absorb at Zero:**
   If r[k+1] < 0: set r[k+1] = 0
   Can get stuck at zero (if Feller violated).

3. **Implicit Method:**
   Solve: r[k+1] = r[k] + κ·(θ - r[k+1])·Δt + σ·√(r[k])·√Δt·Z[k]
   Always positive if Feller holds.

4. **Exact Sampling (Best):**
   Use non-central chi-squared distribution directly.
   No discretization error!

**Milstein Scheme:**
    r[k+1] = r[k] + κ·(θ - r[k])·Δt + σ·√(r[k])·√Δt·Z 
             + (σ²/4)·(Z² - 1)·Δt

Better accuracy, still can go negative.

**Recommended:**
- Use exact sampling when available (transition density known)
- Otherwise: Implicit Euler-Maruyama with small Δt
- Ensure Feller condition: 2κθ ≥ σ²

**Convergence:**
- Weak order: O(Δt) for Euler-Maruyama
- Strong order: O(√Δt)
- Exact sampling: No error

Extensions and Generalizations
-------------------------------

**1. Multi-Factor CIR:**
Multiple correlated CIR processes:
    dr_i = κ_i·(θ_i - r_i)·dt + σ_i·√r_i·dW_i

with correlated Brownian motions.

**2. Affine Jump-Diffusion:**
Add jumps:
    dr = κ·(θ - r)·dt + σ·√r·dW + dJ

where J is compound Poisson (sudden rate changes).

**3. Time-Varying Parameters:**
    dr = κ(t)·(θ(t) - r)·dt + σ(t)·√r·dW

Captures changing market conditions.

**4. Regime-Switching CIR:**
Parameters switch between regimes (Markov chain).

**5. Stochastic Volatility:**
σ itself follows stochastic process (double stochastic).

**6. CIR with Jumps to Default:**
Credit intensity with default events.

**7. Wright-Fisher Diffusion:**
Allele frequency in population genetics (CIR with specific parameters).

Limitations
-----------

**1. Square-Root Boundary:**
Near r = 0, √r creates numerical issues:
- Need small Δt for accurate simulation
- Stochastic calculus different at boundary

**2. Non-Gaussian:**
Stationary distribution is Gamma, not Gaussian:
- Skewed (right tail)
- Linear methods (Kalman) suboptimal
- Need nonlinear filtering

**3. Feller Condition:**
Must satisfy 2κθ ≥ σ² for strict positivity:
- Restricts parameter space
- May not hold for all calibrated parameters

**4. Affine Limitation:**
Term structure shape limited by affine structure.
Cannot match all empirical yield curve features.

**5. Mean Reversion Strength:**
Single-factor CIR: Mean reversion same at all maturities.
Reality: Short rates more mean-reverting than long rates.

Common Pitfalls
---------------

1. **Violating Feller Condition:**
   - Using 2κθ < σ² → absorbing at zero
   - Check parameters: 2·κ·θ ≥ σ²

2. **Negative Rates from Euler:**
   - Standard Euler-Maruyama can give r < 0
   - Use implicit scheme or exact sampling

3. **Wrong Units:**
   - θ in decimal (0.05 = 5% rate)
   - σ in 1/√time units
   - Time in years (typically)

4. **Ignoring Boundary:**
   - r = 0 requires special treatment
   - Numerical schemes must respect positivity

5. **Comparing with Vasicek:**
   - CIR is NOT just Vasicek with constraint
   - Fundamentally different dynamics (multiplicative noise)

6. **Parameter Estimation:**
   - MLE more complex than Vasicek (non-Gaussian)
   - Need specialized methods (exact transition density)

**Impact:**
CIR model demonstrated that:
- Positivity constraint can be enforced naturally (via √r diffusion)
- Analytical tractability preserved (affine structure)
- Realistic features achievable (volatility scaling)

Testing and Validation
-----------------------

**Statistical Tests:**

1. **Positivity Test:**
   - Verify r(t) ≥ 0 for all simulated paths
   - No negative values should occur

2. **Feller Condition:**
   - Check 2κθ ≥ σ² numerically
   - Compare theoretical vs sample positivity

3. **Stationary Distribution:**
   - Long-time histogram should match Gamma(2κθ/σ², 2κ/σ²)
   - Kolmogorov-Smirnov test
   - Q-Q plot against Gamma

4. **Mean Reversion:**
   - From r₀ ≠ θ, should approach θ
   - Time scale: τ = 1/κ

5. **Autocorrelation:**
   - Should decay as exp(-κ·τ)
   - Same as OU process

6. **Transition Density:**
   - Exact sampling: Use non-central χ² formula
   - Compare with numerical scheme

**Parameter Estimation:**

Maximum likelihood using transition density:
- More complex than Vasicek (non-Gaussian)
- Exact transition density known (non-central χ²)
- Numerical optimization required

"""

import sympy as sp

from src.systems.base.core.continuous_stochastic_system import ContinuousStochasticSystem


class CoxIngersollRoss(ContinuousStochasticSystem):
    """
    Cox-Ingersoll-Ross process with mean reversion and square-root diffusion.

    The fundamental model for positive-valued mean-reverting processes,
    combining mean reversion (like OU) with multiplicative noise (like GBM)
    to ensure positivity while maintaining stationarity.

    Stochastic Differential Equation
    ---------------------------------
    General form:
        dr = κ·(θ - r)·dt + σ·√r·dW

    With control:
        dr = (κ·(θ - r) + u)·dt + σ·√r·dW

    where:
        - r(t) ∈ ℝ₊: State (interest rate, volatility, intensity)
        - κ > 0: Mean reversion speed [1/time]
        - θ > 0: Long-term mean (equilibrium level)
        - σ > 0: Volatility [1/√time]
        - u ∈ ℝ: Control input (shifts equilibrium)
        - W(t): Standard Wiener process
        - √r: Square-root diffusion (ensures positivity)

    Key Features
    ------------
    **Positivity:**
    r(t) ≥ 0 for all t if **Feller condition** holds:
        2κθ ≥ σ²

    This ensures boundary r=0 is unattainable.

    **Mean Reversion:**
    Drift κ·(θ - r) pulls toward long-term mean θ:
    - Above θ: Negative drift (pulls down)
    - Below θ: Positive drift (pulls up)

    **Square-Root Diffusion:**
    Noise σ·√r scales with square root of state:
    - High r: Large volatility
    - Low r: Small volatility
    - r = 0: Zero volatility (boundary sticky)

    **Stationary Distribution:**
    r(∞) ~ Gamma(2κθ/σ², 2κ/σ²)

    Mathematical Properties
    -----------------------
    **Feller Condition:**
        2κθ ≥ σ² → r(t) > 0 strictly
        2κθ < σ² → r(t) can hit and stay at 0

    **Moments:**
    Mean: E[r(t)] = r₀·e^(-κt) + θ·(1 - e^(-κt))
    Variance: Var[r(t)] = r₀·(σ²/κ)·(e^(-κt) - e^(-2κt)) + θ·(σ²/2κ)·(1 - e^(-κt))²

    **Asymptotic:**
    Mean: E[r(∞)] = θ
    Variance: Var[r(∞)] = θ·σ²/(2κ)
    Distribution: Gamma(2κθ/σ², 2κ/σ²)

    **Autocorrelation:**
    Cov[r(t), r(t+τ)] = (θ·σ²/2κ)·e^(-κ·τ)

    Physical Interpretation
    -----------------------
    **Mean Reversion Speed κ:**
    - Time constant: τ = 1/κ
    - Half-life: ln(2)/κ
    - Typical: 0.1-2.0 (annual)

    **Long-Term Mean θ:**
    - Equilibrium level
    - Economic: Long-run average rate
    - Typical: 0.02-0.08 (2%-8%)

    **Volatility σ:**
    - Noise intensity
    - Stationary std: √(θ·σ²/2κ)
    - Typical: 0.05-0.20

    **Feller Condition:**
        2κθ ≥ σ²

    Ensures positivity. Always verify!

    State Space
    -----------
    State: r ∈ ℝ₊ = [0, ∞)
        - Non-negative (strictly positive if Feller holds)
        - Equilibrium: θ (long-term mean)
        - Stationary distribution: Gamma

    Control: u ∈ ℝ (optional)
        - Shifts equilibrium to (θ + u/κ)

    Parameters
    ----------
    kappa : float, default=0.5
        Mean reversion speed (must be positive)
        - Typical: 0.1-2.0 (annual)
        - Time constant: 1/κ

    theta : float, default=0.05
        Long-term mean (must be positive)
        - Typical rates: 0.02-0.08 (2%-8%)
        - Economic equilibrium

    sigma : float, default=0.1
        Volatility (must be positive)
        - Typical: 0.05-0.20
        - Check Feller: 2κθ ≥ σ²

    Stochastic Properties
    ---------------------
    - Noise Type: MULTIPLICATIVE (√r diffusion)
    - SDE Type: Itô (standard)
    - Noise Dimension: nw = 1
    - Stationary: Yes (Gamma distribution)
    - Ergodic: Yes (if Feller holds)
    - Positive: Yes (if Feller holds)

    Applications
    ------------
    **1. Interest Rate Modeling:**
    - Short-term rate dynamics
    - Bond pricing (analytical formulas)
    - Term structure modeling

    **2. Stochastic Volatility:**
    - Heston model (variance process)
    - SABR model component
    - Realized volatility

    **3. Credit Risk:**
    - Default intensity (positive, mean-reverting)
    - Spread dynamics

    **4. Commodity Prices:**
    - Mean-reverting prices (ensures positive)

    Numerical Simulation
    --------------------
    **Euler-Maruyama (Simple):**
        r[k+1] = r[k] + κ·(θ - r[k])·Δt + σ·√r[k]·√Δt·Z[k]

    Can go negative! Use reflection or absorption.

    **Exact Sampling (Preferred):**
    Use non-central chi-squared transition density.
    No discretization error, always positive.

    **Validation:**
    - Check positivity: All r[k] ≥ 0
    - Verify Feller: 2κθ ≥ σ²
    - Test stationary distribution (Gamma)

    Comparison with Other Processes
    --------------------------------
    **vs. Ornstein-Uhlenbeck:**
    - OU: Additive noise, can be negative
    - CIR: Multiplicative noise, always positive

    **vs. Geometric Brownian Motion:**
    - GBM: No mean reversion, log-normal
    - CIR: Mean reversion, Gamma (stationary)

    **vs. Vasicek:**
    - Vasicek: Additive σ
    - CIR: Multiplicative σ·√r

    Limitations
    -----------
    - Feller condition restricts parameters
    - Non-Gaussian (complicates filtering)
    - Square-root creates numerical challenges
    - Single-factor (term structure limitations)

    Extensions
    ----------
    - Multi-factor CIR (correlated processes)
    - Jump-CIR (Poisson jumps)
    - Regime-switching CIR
    - Time-varying parameters
    """

    def define_system(
        self,
        kappa: float = 0.5,
        theta: float = 0.05,
        sigma: float = 0.1,
    ):
        """
        Define Cox-Ingersoll-Ross process dynamics.

        Sets up the SDE:
            dr = κ·(θ - r)·dt + σ·√r·dW

        Parameters
        ----------
        kappa : float, default=0.5
            Mean reversion speed (must be positive)
            - Controls speed of return to θ
            - Time constant: τ = 1/κ
            - Typical: 0.1-2.0 (annual for rates)

        theta : float, default=0.05
            Long-term mean (must be positive)
            - Equilibrium level
            - Typical rates: 0.02-0.08 (2%-8%)
            - As decimal, not percentage

        sigma : float, default=0.1
            Volatility (must be positive)
            - Noise intensity
            - Typical: 0.05-0.20
            - Check Feller: 2κθ ≥ σ²

        Raises
        ------
        ValueError
            If any parameter is non-positive
        UserWarning
            If Feller condition violated (2κθ < σ²)

        Notes
        -----
        **Feller Condition:**

        Critical requirement: 2κθ ≥ σ²

        If satisfied:
        - r(t) > 0 strictly for all t
        - Boundary r=0 unattainable
        - Well-defined for all time

        If violated:
        - r(t) can hit zero and stay there
        - Absorbing boundary
        - Problematic for interest rates

        **Parameter Relationships:**

        Stationary statistics:
        - Mean: θ
        - Variance: θ·σ²/(2κ)
        - Std: √(θ·σ²/2κ)

        Time scales:
        - Mean reversion time: 1/κ
        - Half-life: ln(2)/κ

        **Design Guidelines:**

        To achieve target mean μ and std s:
        1. Set θ = μ
        2. Set σ² = 2κ·s²/μ
        3. Choose κ for desired time scale
        4. Verify Feller: 2κμ ≥ 2κs²/μ → μ² ≥ s²

        **Typical Combinations:**

        Conservative (low vol):
        - κ = 0.5, θ = 0.05, σ = 0.05
        - Feller: 0.05 ≥ 0.0025 ✓
        - σ_stat = 0.016 (1.6%)

        Standard:
        - κ = 0.5, θ = 0.05, σ = 0.1
        - Feller: 0.05 ≥ 0.01 ✓
        - σ_stat = 0.032 (3.2%)

        High volatility:
        - κ = 0.5, θ = 0.05, σ = 0.2
        - Feller: 0.05 < 0.04 ✗ (violates!)
        - Boundary attainable
        """
        # Validate parameters
        if kappa <= 0:
            raise ValueError(f"kappa must be positive, got {kappa}")
        if theta <= 0:
            raise ValueError(f"theta must be positive, got {theta}")
        if sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}")

        # Check Feller condition
        feller_lhs = 2 * kappa * theta
        feller_rhs = sigma**2
        
        if feller_lhs < feller_rhs:
            import warnings
            warnings.warn(
                f"Feller condition violated: 2κθ = {feller_lhs:.6f} < σ² = {feller_rhs:.6f}. "
                f"Process can reach r=0 and become absorbed. "
                f"For strict positivity, ensure 2κθ ≥ σ². "
                f"Current ratio: {feller_lhs/feller_rhs:.3f} (need ≥ 1.0)",
                UserWarning
            )

        # Define symbolic variables
        r = sp.symbols("r", real=True, positive=True)
        u = sp.symbols("u", real=True)

        # Symbolic parameters
        kappa_sym = sp.symbols("kappa", positive=True)
        theta_sym = sp.symbols("theta", positive=True)
        sigma_sym = sp.symbols("sigma", positive=True)

        # System definition
        self.state_vars = [r]
        self.control_vars = [u]

        # Drift: f(r, u) = κ·(θ - r) + u
        self._f_sym = sp.Matrix([[kappa_sym * (theta_sym - r) + u]])

        self.parameters = {
            kappa_sym: kappa, 
            theta_sym: theta, 
            sigma_sym: sigma
        }
        self.order = 1

        # Diffusion: g(r, u) = σ·√r (multiplicative!)
        # This is the key feature ensuring positivity
        self.diffusion_expr = sp.Matrix([[sigma_sym * sp.sqrt(r)]])

        # Itô SDE
        self.sde_type = "ito"

    def get_feller_parameter(self) -> float:
        """
        Get Feller parameter ν = 2κθ/σ².

        If ν ≥ 1: Feller condition satisfied (strictly positive)
        If ν < 1: Feller condition violated (can hit zero)

        Returns
        -------
        float
            Feller parameter ν

        Notes
        -----
        Physical interpretation:
        - ν: Degrees of freedom in stationary chi-squared
        - ν >> 1: Far from boundary, nearly Gaussian
        - ν ≈ 1: Near boundary, skewed
        - ν < 1: Boundary attainable

        Examples
        --------
        >>> cir = CoxIngersollRoss(kappa=0.5, theta=0.05, sigma=0.1)
        >>> nu = cir.get_feller_parameter()
        >>> print(f"Feller parameter: {nu:.2f}")
        >>> print(f"Strictly positive: {nu >= 1}")
        """
        kappa = self.parameters[sp.symbols('kappa')]
        theta = self.parameters[sp.symbols('theta')]
        sigma = self.parameters[sp.symbols('sigma')]
        
        return 2 * kappa * theta / sigma**2

    def get_stationary_mean(self) -> float:
        """
        Get stationary mean E[r(∞)] = θ.

        Returns
        -------
        float
            Stationary mean

        Examples
        --------
        >>> cir = CoxIngersollRoss(theta=0.05)
        >>> mean = cir.get_stationary_mean()
        >>> print(f"Long-term mean: {mean:.4f}")  # 0.05
        """
        return self.parameters[sp.symbols('theta')]

    def get_stationary_variance(self) -> float:
        """
        Get stationary variance Var[r(∞)] = θ·σ²/(2κ).

        Returns
        -------
        float
            Stationary variance

        Examples
        --------
        >>> cir = CoxIngersollRoss(kappa=0.5, theta=0.05, sigma=0.1)
        >>> var = cir.get_stationary_variance()
        >>> std = np.sqrt(var)
        >>> print(f"Stationary std: {std:.4f}")
        """
        kappa = self.parameters[sp.symbols('kappa')]
        theta = self.parameters[sp.symbols('theta')]
        sigma = self.parameters[sp.symbols('sigma')]
        
        return theta * sigma**2 / (2 * kappa)

    def get_time_constant(self) -> float:
        """
        Get mean reversion time constant τ = 1/κ.

        Returns
        -------
        float
            Time constant

        Examples
        --------
        >>> cir = CoxIngersollRoss(kappa=0.5)
        >>> tau = cir.get_time_constant()
        >>> print(f"Time constant: {tau:.2f} years")  # 2.0
        """
        kappa = self.parameters[sp.symbols('kappa')]
        return 1.0 / kappa


# Convenience functions
def create_interest_rate_cir(
    long_term_rate: float = 0.05,
    mean_reversion: float = 0.5,
    volatility: float = 0.1,
) -> CoxIngersollRoss:
    """
    Create CIR model for interest rate dynamics.

    Parameters
    ----------
    long_term_rate : float, default=0.05
        Long-term mean rate θ (as decimal: 0.05 = 5%)
    mean_reversion : float, default=0.5
        Mean reversion speed κ [1/year]
    volatility : float, default=0.1
        Volatility σ [1/√year]

    Returns
    -------
    CoxIngersollRoss

    Notes
    -----
    Automatically checks Feller condition and warns if violated.

    Examples
    --------
    >>> # Standard configuration
    >>> r_model = create_interest_rate_cir(
    ...     long_term_rate=0.05,
    ...     mean_reversion=0.5,
    ...     volatility=0.1
    ... )
    """
    return CoxIngersollRoss(kappa=mean_reversion, theta=long_term_rate, sigma=volatility)


def create_variance_process(
    long_term_var: float = 0.04,
    mean_reversion: float = 2.0,
    vol_of_vol: float = 0.3,
) -> CoxIngersollRoss:
    """
    Create CIR process for stochastic volatility (Heston model).

    In Heston model, variance V follows CIR:
        dV = κ·(θ - V)·dt + σ·√V·dW

    Parameters
    ----------
    long_term_var : float, default=0.04
        Long-term variance θ (e.g., 0.04 = 20% vol)
    mean_reversion : float, default=2.0
        Mean reversion speed κ
    vol_of_vol : float, default=0.3
        Volatility of variance σ

    Returns
    -------
    CoxIngersollRoss

    Notes
    -----
    Heston model uses CIR for variance to ensure V ≥ 0.

    Examples
    --------
    >>> # Heston variance process
    >>> V_process = create_variance_process(
    ...     long_term_var=0.04,  # 20% long-term vol
    ...     mean_reversion=2.0,
    ...     vol_of_vol=0.3
    ... )
    """
    return CoxIngersollRoss(kappa=mean_reversion, theta=long_term_var, sigma=vol_of_vol)