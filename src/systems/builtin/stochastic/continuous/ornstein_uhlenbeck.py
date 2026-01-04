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

This module provides implementations of the Ornstein-Uhlenbeck (OU) process,
the fundamental mean-reverting stochastic differential equation. The OU process
is distinguished by:

- The canonical example of mean-reverting stochastic processes
- Foundation for many models in physics, finance, and biology
- The stochastic analog of linear damped dynamics
- Unique stationary Gaussian process with exponential correlation
- Explicit analytical solution in closed form

The OU process represents the continuous-time limit of discrete AR(1)
autoregressive processes and serves as the building block for more
complex mean-reverting models.

Mathematical Background
-----------------------
The Ornstein-Uhlenbeck process was introduced in 1930 by Leonard Ornstein
and George Uhlenbeck as a model for the velocity of a massive Brownian
particle undergoing friction. It solved a fundamental problem with
Einstein's original Brownian motion model: velocities must be bounded
due to finite energy.

**Physical Motivation:**
Consider a particle of mass m in a viscous fluid:
- Friction force: -γ·v (opposes motion)
- Random thermal force: η(t) (white noise)
- Newton's law: m·dv = -γ·v·dt + η(t)·dt

Dividing by m and setting α = γ/m, σ = √(2kᵦT/m):
    dv = -α·v·dt + σ·dW

This is the OU process. The particle's velocity fluctuates around zero
(thermal equilibrium) with characteristic time scale 1/α.

Mathematical Formulation
------------------------

**Standard Form (Centered):**
    dX = -α·X·dt + σ·dW

where:
    - X(t): State variable (velocity, rate, deviation from mean)
    - α > 0: Mean reversion rate (dimension: 1/time)
    - σ > 0: Volatility (noise intensity, dimension: state/√time)
    - W(t): Standard Wiener process
    - dW ~ N(0, dt)

**General Form (Non-Centered):**
    dX = α·(μ - X)·dt + σ·dW
    
Equivalent to: dX = -α·X·dt + α·μ·dt + σ·dW

This reverts to long-term mean μ rather than zero.

**With Control Input:**
    dX = (-α·X + u)·dt + σ·dW

The control u provides external forcing (can be time-varying or
state-dependent).

**Key Distinction from Brownian Motion:**
- Brownian Motion: dX = σ·dW (no drift, no reversion)
- OU Process: dX = -α·X·dt + σ·dW (drift opposes displacement)

The drift term -α·X creates a restoring force toward the origin,
fundamentally changing the long-term behavior.

Analytical Solution
-------------------
The OU process is one of the few SDEs with an explicit closed-form solution.

**Exact Solution:**
For X(0) = X₀ and constant control u:

    X(t) = X₀·e^(-α·t) + (u/α)·(1 - e^(-α·t)) + ∫₀ᵗ σ·e^(-α·(t-s))·dW(s)

The stochastic integral is Gaussian with zero mean.

**Interpretation:**
1. X₀·e^(-α·t): Initial condition exponentially decays
2. (u/α)·(1 - e^(-α·t)): Deterministic approach to u/α
3. Stochastic integral: Accumulated noise, filtered by reversion

**Moments (u = 0):**

Mean:
    E[X(t)] = X₀·e^(-α·t)

Exponentially decays to zero with time constant τ = 1/α.

Variance:
    Var[X(t)] = (σ²/2α)·(1 - e^(-2α·t))

Increases from 0 to stationary value σ²/(2α).

Covariance (s < t):
    Cov[X(s), X(t)] = (σ²/2α)·e^(-α·|t-s|)·(1 - e^(-2α·s))

For stationary process (s,t → ∞):
    Cov[X(s), X(t)] = (σ²/2α)·e^(-α·|t-s|)

**Asymptotic Behavior (t → ∞):**

Mean:
    E[X(∞)] = u/α

Variance (Stationary):
    Var[X(∞)] = σ²/(2α)

Standard Deviation:
    Std[X(∞)] = σ/√(2α)

Distribution:
    X(∞) ~ N(u/α, σ²/(2α))

The process reaches a **stationary distribution** - unique among
common SDEs

Key Properties
--------------

**1. Mean Reversion:**
The defining feature. Drift -α·X pulls process toward mean:
- Above mean (X > 0): Negative drift pushes down
- Below mean (X < 0): Positive drift pushes up
- At mean (X = 0): No drift (but still diffuses)

**2. Stationarity:**
Unique stationary Gaussian distribution:
    π(x) = N(0, σ²/(2α))

This is the **equilibrium distribution** - probability density
doesn't change with time for t → ∞.

**3. Ergodicity:**
Time averages equal ensemble averages:
    lim_{T→∞} (1/T)∫₀ᵀ f(X(t))dt = E[f(X)]

Can estimate statistics from single long trajectory.

**4. Markov Property:**
Future independent of past given present:
    P(X(t)|X(s), s ≤ u) = P(X(t)|X(u)) for t > u

**5. Gaussian Process:**
For any finite set of times {t₁, ..., tₙ}:
    (X(t₁), ..., X(tₙ)) ~ Multivariate Normal

Completely characterized by mean and covariance functions.

**6. Exponential Correlation:**
Autocorrelation function:
    ρ(τ) = Cov[X(t), X(t+τ)]/Var[X] = e^(-α·τ)

Correlation decays exponentially with lag τ.
- τ = 0: ρ = 1 (perfect correlation)
- τ = 1/α: ρ = e^(-1) ≈ 0.37
- τ = 5/α: ρ ≈ 0.007 (essentially uncorrelated)

**7. Additive Noise:**
Diffusion σ is constant (state-independent).
- Simplifies analysis and simulation
- Noise intensity same everywhere in state space
- Contrasts with multiplicative noise (GBM)

**8. Continuity:**
Sample paths are continuous but nowhere differentiable (like
Brownian motion).

**9. Gaussianity:**
X(t) is Gaussian for all t (inherited from Brownian motion).

Physical and Mathematical Interpretation
-----------------------------------------

**Mean Reversion Rate α:**
- Dimension: [1/time]
- Physical: Friction coefficient / mass
- Financial: Speed of adjustment to equilibrium
- Controls time scale of dynamics

**Time Constant τ = 1/α:**
- Average time to revert to mean
- After time τ: Deviation reduced by factor e ≈ 0.37
- After time 5τ: ~99% reverted

**Half-Life t₁/₂ = ln(2)/α ≈ 0.693/α:**
- Time to reduce deviation by 50%
- More intuitive than time constant
- Commonly used in finance

**Relaxation Time:**
Time to reach stationarity: t_relax ≈ 3-5τ

**Examples:**
- α = 0.1: Slow reversion, τ = 10s
- α = 1.0: Moderate reversion, τ = 1s
- α = 10.0: Fast reversion, τ = 0.1s

**Volatility σ:**
- Dimension: [state]/√[time]
- Physical: Noise intensity from thermal fluctuations
- Financial: Instantaneous standard deviation of changes
- Not the stationary standard deviation

**Stationary Standard Deviation:**
    σ_stat = σ/√(2α)

Balance between noise injection (σ) and dissipation (α):
- Large σ: More noise → larger fluctuations
- Large α: Faster reversion → smaller fluctuations

**Ratio α/σ:**
Signal-to-noise ratio:
- Large α/σ: Strong mean reversion, tight around mean
- Small α/σ: Weak mean reversion, large excursions

**Energy Interpretation:**
In physical systems:
    σ² = 2α·kᵦ·T/m

This is the **fluctuation-dissipation theorem**: noise and friction
are related through temperature.

Connection to Discrete AR(1)
-----------------------------
The OU process is the continuous-time limit of discrete AR(1):
    X[k+1] = φ·X[k] + ε[k]

where ε ~ N(0, σ_ε²).

**Correspondence:**
For small Δt:
    φ = e^(-α·Δt) ≈ 1 - α·Δt
    σ_ε² = σ²·Δt

Exact discretization:
    X[k+1] = e^(-α·Δt)·X[k] + √((σ²/2α)·(1 - e^(-2α·Δt)))·Z[k]

where Z ~ N(0,1).

Applications
------------

**1. Physics:**

**Langevin Equation (Particle Velocity):**
Velocity of massive particle in viscous fluid:
    dv = -γ·v·dt + √(2D)·dW

Where γ is friction, D is diffusion coefficient.

**Thermal Equilibrium:**
At equilibrium: E[v²] = kᵦT/m (equipartition theorem)
Implies: σ² = 2γ·kᵦT/m (fluctuation-dissipation)

**Applications:**
- Brownian motion (velocity, not position)
- Colloidal particle dynamics
- Molecular diffusion in traps
- Optical tweezers experiments
- Single-molecule biophysics

**2. Mathematical Finance:**

**Vasicek Interest Rate Model (1977):**
    dr = κ·(θ - r)·dt + σ·dW

Where:
- r(t): Short-term interest rate
- θ: Long-term mean rate
- κ: Mean reversion speed
- σ: Interest rate volatility

**Advantages:**
- Analytical bond prices
- Tractable option formulas
- Mean reversion captures rate dynamics

**Limitations:**
- Can become negative (unrealistic)
- Constant volatility (unrealistic)
- Led to extensions (CIR, Hull-White)

**Commodity Prices:**
Many commodities (oil, gas, metals) exhibit mean reversion:
- High prices → increased production → price drop
- Low prices → decreased supply → price rise

Model: dS = κ·(μ - ln(S))·S·dt + σ·S·dW
(Geometric OU in log-space)

**Pairs Trading:**
Spread between correlated assets often mean-reverts:
    Spread = Stock_A - β·Stock_B

**Credit Spreads:**
Corporate bond spreads over treasuries mean-revert.

**3. Neuroscience:**

**Neural Membrane Potential:**
Voltage across neuron membrane between spikes:
    dV = -(V - V_rest)/τ_m·dt + σ·dW

Where:
- V: Membrane potential
- V_rest: Resting potential
- τ_m: Membrane time constant (~10-20 ms)
- σ: Synaptic noise

**Leaky Integrate-and-Fire Model:**
OU process until threshold, then spike and reset.

**Synaptic Conductances:**
Time-varying conductances often modeled as OU.

**4. Biology:**

**Population Fluctuations:**
Population around carrying capacity:
    dN = -α·(N - K)·dt + σ·dW

Where K is carrying capacity.

**Gene Expression:**
Protein levels with stochastic production/degradation.

**Ecological Dynamics:**
Predator-prey systems with environmental noise.

**5. Climate Science:**

**Temperature Anomalies:**
Deviations from long-term average exhibit mean reversion.

**El Niño/La Niña:**
Ocean temperature fluctuations modeled as OU.

**6. Engineering:**

**Control Systems:**
Benchmark for stochastic optimal control:
- LQG (Linear-Quadratic-Gaussian) problem
- Kalman filtering applications
- Stochastic stability analysis

**Signal Processing:**
- Colored noise generation
- Time series modeling
- Autoregressive processes

**Communication Channels:**
Channel state fluctuations.

Numerical Simulation
--------------------

**Euler-Maruyama Discretization:**
    X[k+1] = X[k] + (-α·X[k] + u)·Δt + σ·√Δt·Z[k]
            = (1 - α·Δt)·X[k] + u·Δt + σ·√Δt·Z[k]

where Z ~ N(0,1).

**Convergence:**
- Weak order: O(Δt)
- Strong order: O(√Δt)

**Stability:**
Requires α·Δt < 1 for numerical stability.
Otherwise can overshoot and oscillate.

**Exact Discretization (Preferred):**
    X[k+1] = e^(-α·Δt)·X[k] + (u/α)·(1 - e^(-α·Δt)) 
             + √((σ²/2α)·(1 - e^(-2α·Δt)))·Z[k]

**Advantages:**
- Exact (no discretization error)
- Unconditionally stable
- Preserves stationary distribution
- Matches autocorrelation exactly

**Implementation:**
For additive noise, framework can use specialized solvers:
- No state dependence in diffusion
- Can precompute diffusion matrix
- More efficient integration

**Recommended Methods:**
- 'euler-maruyama': Simple, fast for small Δt
- 'milstein': Higher order (but same as Euler for additive noise)
- Exact scheme: Best choice when available

Statistical Analysis
--------------------

**Parameter Estimation:**
Given observations X₀, X₁, ..., X_n at times t₀, t₁, ..., t_n:

**Maximum Likelihood (Discrete Sampling):**
For equally spaced observations with Δt:

α̂ = -ln(∑X_i·X_{i-1} / ∑X_i²) / Δt

σ̂² = (2α̂/n)·∑(X_{i+1} - e^(-α̂·Δt)·X_i)² / (1 - e^(-2α̂·Δt))

**Method of Moments:**
Sample mean: m̄ → E[X] = 0 (check consistency)
Sample variance: s² → σ²/(2α)
Sample autocorrelation: ρ̂(Δt) → e^(-α·Δt)

From autocorrelation: α̂ = -ln(ρ̂(Δt)) / Δt

**Hypothesis Testing:**

1. **Mean Reversion Test:**
   - H₀: α = 0 (Brownian motion)
   - H₁: α > 0 (mean reversion)
   - Use unit root tests (ADF, PP)

2. **Stationarity Test:**
   - H₀: Process is stationary
   - Use KPSS test

3. **Gaussianity:**
   - Test residuals for normality
   - Jarque-Bera, Shapiro-Wilk

4. **Autocorrelation:**
   - Should decay exponentially
   - Plot log(ρ̂(τ)) vs τ (should be linear)

**Model Validation:**
- Compare theoretical and sample ACF
- Check residual independence
- Verify stationary variance: s² ≈ σ²/(2α)
- Test for parameter constancy over time

Comparison with Other Processes
--------------------------------

**vs. Brownian Motion:**
- BM: No drift, grows indefinitely (non-stationary)
- OU: Mean-reverting, bounded variance (stationary)
- BM is OU with α = 0

**vs. Geometric Brownian Motion:**
- GBM: Multiplicative noise, log-normal, for prices
- OU: Additive noise, Gaussian, for rates/deviations
- GBM non-stationary, OU stationary

**vs. Cox-Ingersoll-Ross (CIR):**
- CIR: dX = κ·(θ-X)·dt + σ·√X·dW (multiplicative noise)
- OU: dX = α·(μ-X)·dt + σ·dW (additive noise)
- CIR ensures X > 0, OU can be negative
- CIR for interest rates (positive), OU for spreads (can be negative)

**vs. Vasicek (Extended OU):**
- Vasicek: OU with non-zero mean μ
- Both are Gaussian, mean-reverting
- Vasicek = OU shifted and scaled

**vs. AR(1) Process:**
- AR(1): Discrete-time analog
- OU: Continuous-time limit of AR(1)
- Similar properties (stationarity, autocorrelation)

Extensions and Generalizations
-------------------------------

**1. Multivariate OU:**
    dX = A·X·dt + Σ·dW

Where A is stability matrix, Σ is diffusion matrix.
Applications: Multiple correlated rates, portfolio dynamics.

**2. Geometric OU:**
    dX = α·(μ - ln(X))·X·dt + σ·X·dW

Mean reversion in log-space, ensures X > 0.

**3. Non-Linear Mean Reversion:**
    dX = f(X)·dt + σ·dW

Where f(X) is non-linear restoring force.

**4. Time-Varying Parameters:**
    dX = -α(t)·X·dt + σ(t)·dW

Captures changing market conditions.

**5. Jump-OU:**
    dX = -α·X·dt + σ·dW + dJ

Adds discontinuous jumps to continuous process.

**6. Fractional OU:**
Replace Brownian motion with fractional Brownian motion (long memory).

Limitations
-----------

**1. Can Be Negative:**
OU process ranges over all real numbers. Problematic for:
- Interest rates (should be ≥ 0)
- Prices (should be > 0)
- Populations (should be ≥ 0)

**Solution:** Use CIR or geometric OU for positive quantities.

**2. Constant Volatility:**
σ independent of state. Reality:
- Interest rate volatility increases with rate level
- Spread volatility changes with spread

**Solution:** Use state-dependent diffusion.

**3. Linear Mean Reversion:**
Drift -α·X assumes linear restoring force. Reality:
- May have non-linear reversion
- Asymmetric reversion (faster from extremes)

**4. Gaussian Distribution:**
Real data often shows:
- Fat tails (jumps, regime changes)
- Skewness
- Time-varying moments

**5. Stationarity Assumption:**
Markets change:
- Mean μ drifts over time
- α, σ change with market regime

Common Pitfalls
---------------

1. **Confusing σ with Stationary Std:**
   - σ: Noise intensity parameter
   - σ/√(2α): Stationary standard deviation
   - σ_stat < σ (reduced by mean reversion)

2. **Ignoring α·Δt Constraint:**
   - Euler-Maruyama unstable if α·Δt > 1
   - Use smaller Δt or exact discretization

3. **Wrong Time Scale:**
   - Must match α units with time units
   - α in years⁻¹ requires time in years

4. **Forgetting Initial Transient:**
   - Takes time 3-5τ to reach stationarity
   - Don't use initial data for estimating stationary stats

5. **Assuming Independence:**
   - OU has exponential autocorrelation
   - Not white noise!
   - Observations correlated over time scale 1/α

6. **Negative Rate Problem:**
   - OU can go negative
   - Vasicek model predicts negative interest rates
   - Use CIR or other models for strictly positive quantities

**Physical Significance:**
The OU process resolved a fundamental problem: Einstein's Brownian
motion for position implies infinite kinetic energy. The OU process
for velocity respects finite energy (equipartition theorem) while
maintaining stochastic behavior.

**Mathematical Significance:**
First example showing that:
- Stationary processes need not be trivial
- Gaussian processes can have non-trivial correlation
- Mean reversion creates equilibrium distribution

Testing and Validation
-----------------------

**Unit Tests for OU Process:**

1. **Mean Reversion:**
   - Start at x₀ ≠ 0
   - Verify E[X(t)] → 0 as t → ∞
   - Check exponential decay rate

2. **Stationary Variance:**
   - Long simulation (t >> 1/α)
   - Sample variance should equal σ²/(2α)

3. **Autocorrelation:**
   - Compute ACF from simulation
   - Should match e^(-α·τ)

4. **Gaussianity:**
   - Histogram should be normal
   - Q-Q plot should be linear

5. **Exact Solution:**
   - Compare numerical to analytical moments

"""

import numpy as np
import sympy as sp

from src.systems.base.core.continuous_stochastic_system import ContinuousStochasticSystem


class OrnsteinUhlenbeck(ContinuousStochasticSystem):
    """
    Ornstein-Uhlenbeck process with mean reversion and additive noise.

    The fundamental mean-reverting stochastic process, combining deterministic
    relaxation toward an equilibrium with random fluctuations. This is the
    stochastic analog of a damped harmonic oscillator and the continuous-time
    limit of AR(1) processes.

    Stochastic Differential Equation
    ---------------------------------
    Centered form (mean = 0):
        dX = -α·X·dt + σ·dW

    With control:
        dX = (-α·X + u)·dt + σ·dW

    where:
        - X(t) ∈ ℝ: State (velocity, rate, deviation)
        - α > 0: Mean reversion rate (1/time)
        - σ > 0: Volatility (noise intensity, state/√time)
        - u ∈ ℝ: External forcing/control
        - W(t): Standard Wiener process

    Key Features
    ------------
    **Mean Reversion:**
    Drift term -α·X creates restoring force toward zero:
    - Above zero: Negative drift pushes down
    - Below zero: Positive drift pushes up
    - Strength proportional to displacement

    **Stationarity:**
    Unique stationary distribution:
        X(∞) ~ N(u/α, σ²/(2α))

    Unlike Brownian motion or GBM, variance bounded.

    **Additive Noise:**
    Diffusion σ constant (state-independent).
    Simplifies analysis and simulation.

    **Ergodicity:**
    Time averages equal ensemble averages.
    Can estimate statistics from single long trajectory.

    Mathematical Properties
    -----------------------
    **Exact Solution (u = 0, X(0) = X₀):**
        X(t) = X₀·e^(-α·t) + ∫₀ᵗ σ·e^(-α·(t-s))·dW(s)

    **Moments:**
    Mean:
        E[X(t)] = X₀·e^(-α·t) + (u/α)·(1 - e^(-α·t))

    Variance:
        Var[X(t)] = (σ²/2α)·(1 - e^(-2α·t))

    **Asymptotic (t → ∞):**
    Mean: E[X(∞)] = u/α
    Variance: Var[X(∞)] = σ²/(2α)
    Distribution: N(u/α, σ²/(2α))

    **Autocorrelation:**
    For stationary process:
        Cov[X(t), X(t+τ)] = (σ²/2α)·e^(-α·τ)

    Exponential decay with rate α.

    Physical Interpretation
    -----------------------
    **Mean Reversion Rate α:**
    - Controls speed of return to equilibrium
    - Units: [1/time]
    - Time constant: τ = 1/α
    - Half-life: t₁/₂ = ln(2)/α ≈ 0.693/α

    **Examples:**
    - α = 0.1: Slow reversion, τ = 10s
    - α = 1.0: Moderate, τ = 1s  
    - α = 10.0: Fast, τ = 0.1s

    **Volatility σ:**
    - Instantaneous noise intensity
    - Units: [state]/√[time]
    - Stationary std: σ/√(2α)

    **Ratio σ/α:**
    Effective noise level:
    - Large: Weak reversion, large fluctuations
    - Small: Strong reversion, tight around mean

    **Stationary Standard Deviation:**
        σ_stat = σ/√(2α)

    Balance between noise injection (σ) and dissipation (α).

    State Space
    -----------
    State: x ∈ ℝ (unbounded)
        - Can take any real value
        - Equilibrium at x = u/α
        - Fluctuates around equilibrium

    Control: u ∈ ℝ (optional)
        - Shifts equilibrium to u/α
        - Examples: External force, policy intervention

    Parameters
    ----------
    alpha : float, default=1.0
        Mean reversion rate (must be positive for stability)
        - Larger α: Faster reversion, smaller steady-state variance
        - Time constant τ = 1/α
        - Typical range: 0.1 to 10.0

    sigma : float, default=1.0
        Volatility (must be positive)
        - Controls noise intensity
        - Stationary std: σ/√(2α)
        - Typical: 0.1 to 2.0

    Stochastic Properties
    ---------------------
    - Noise Type: ADDITIVE
    - Diffusion: g(x) = σ (constant, state-independent)
    - SDE Type: Itô (standard)
    - Noise Dimension: nw = 1
    - Stationary: Yes (unique equilibrium distribution)
    - Ergodic: Yes (time averages = ensemble averages)

    Applications
    ------------
    **1. Physics:**
    - Langevin equation (particle velocity in fluid)
    - Thermal equilibrium (velocity distribution)
    - Optical tweezers (trapped particle)
    - Molecular dynamics

    **2. Finance:**
    - Vasicek interest rate model
    - Commodity price spreads
    - Pairs trading (spread between correlated assets)
    - Credit spreads

    **3. Neuroscience:**
    - Neural membrane potential between spikes
    - Leaky integrate-and-fire neurons
    - Synaptic conductances

    **4. Biology:**
    - Population fluctuations around carrying capacity
    - Gene expression dynamics
    - Ecological systems with noise

    **5. Control & Signal Processing:**
    - Colored noise generation
    - Autoregressive processes (continuous AR(1))
    - Kalman filtering benchmark

    Numerical Simulation
    --------------------
    **Euler-Maruyama:**
        X[k+1] = (1 - α·Δt)·X[k] + u·Δt + σ·√Δt·Z[k]

    Requires α·Δt < 1 for stability.

    **Exact Discretization (Preferred):**
        X[k+1] = e^(-α·Δt)·X[k] + (u/α)·(1-e^(-α·Δt)) 
                 + σ_eff·Z[k]

    where σ_eff = √((σ²/2α)·(1-e^(-2α·Δt)))

    **Advantages:**
    - Exact (no discretization error)
    - Unconditionally stable
    - Preserves stationary distribution

    Statistical Analysis
    --------------------
    **Parameter Estimation:**
    From discrete observations, estimate:
        α̂ = -ln(autocorr(Δt)) / Δt
        σ̂ = sample_std · √(2α̂)

    **Model Validation:**
    - Check exponential autocorrelation
    - Verify Gaussian residuals
    - Test stationarity (KPSS)
    - Test mean reversion (ADF unit root)

    Comparison with Other Processes
    --------------------------------
    **vs. Brownian Motion:**
    - BM: No reversion (non-stationary)
    - OU: Mean-reverting (stationary)

    **vs. Geometric Brownian Motion:**
    - GBM: Multiplicative noise, for prices
    - OU: Additive noise, for rates/deviations

    **vs. CIR Process:**
    - CIR: Multiplicative noise √X (stays positive)
    - OU: Additive noise (can be negative)

    Limitations
    -----------
    - Can be negative (problem for rates/prices)
    - Constant volatility (unrealistic for some applications)
    - Linear mean reversion (may be non-linear in reality)
    - Gaussian (real data may have fat tails)

    **Solutions:**
    - CIR for positive quantities
    - State-dependent diffusion
    - Non-linear drift functions
    - Jump extensions

    See Also
    --------
    BrownianMotion : No mean reversion (α=0 limit)
    GeometricBrownianMotion : Multiplicative noise
    CoxIngersollRoss : Mean-reverting with √X diffusion
    """

    def define_system(self, alpha: float = 1.0, sigma: float = 1.0):
        """
        Define Ornstein-Uhlenbeck process dynamics.

        Sets up the stochastic differential equation:
            dX = (-α·X + u)·dt + σ·dW

        with mean reversion and additive noise.

        Parameters
        ----------
        alpha : float, default=1.0
            Mean reversion rate (should be positive)
            - α > 0: Stable (mean-reverting)
            - α = 0: Brownian motion (no reversion)
            - α < 0: Unstable (explosive)

        sigma : float, default=1.0
            Volatility (must be positive)
            - Controls noise intensity
            - Stationary std: σ/√(2α)

        Raises
        ------
        ValueError
            If sigma ≤ 0
        UserWarning
            If alpha ≤ 0 (unstable/non-reverting)

        Notes
        -----
        **Stability Condition:**
        Require α > 0 for mean reversion and stationarity.
        - α > 0: Process stable, reverts to mean
        - α = 0: Becomes Brownian motion
        - α < 0: Unstable, diverges exponentially

        **Time Scales:**
        - Time constant: τ = 1/α
        - Half-life: t₁/₂ = ln(2)/α ≈ 0.693/α
        - Settling time: t_settle ≈ 5/α (1% of initial deviation)

        After time τ: Deviation reduced by factor e ≈ 0.368
        After time 5τ: Deviation reduced by factor e⁻⁵ ≈ 0.007 (~99% reverted)

        **Stationary Statistics:**
        For u = 0:
        - Mean: E[X(∞)] = 0
        - Variance: Var[X(∞)] = σ²/(2α)
        - Std: σ/√(2α)

        For u ≠ 0:
        - Mean: E[X(∞)] = u/α
        - Variance: σ²/(2α) (same)

        **Parameter Selection:**
        - Fast reversion: α = 5-10 (τ = 0.1-0.2s)
        - Moderate: α = 1-2 (τ = 0.5-1s)
        - Slow: α = 0.1-0.5 (τ = 2-10s)

        **Noise Level:**
        Effective noise (stationary std): σ/√(2α)
        - To achieve desired std s: set σ = s·√(2α)

        **Fluctuation-Dissipation:**
        In physical systems at temperature T:
            σ² = 2α·kᵦ·T/m
        Balance between thermal noise and friction.
        """
        # Validate parameters
        if alpha <= 0:
            import warnings
            warnings.warn(
                f"alpha={alpha} ≤ 0 leads to unstable/non-reverting process. "
                f"Use alpha > 0 for mean reversion. "
                f"alpha = 0 gives Brownian motion, alpha < 0 is explosive.",
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

        # Drift: f(x, u) = -α·x + u
        # Mean-reverting drift toward u/α
        self._f_sym = sp.Matrix([[-alpha_sym * x + u]])

        self.parameters = {alpha_sym: alpha, sigma_sym: sigma}
        self.order = 1

        # Diffusion: g(x, u) = σ (constant - additive noise)
        self.diffusion_expr = sp.Matrix([[sigma_sym]])
        self.sde_type = "ito"

    def get_stationary_std(self) -> float:
        """
        Compute theoretical stationary standard deviation.

        For OU process, stationary distribution is:
            X(∞) ~ N(u/α, σ²/(2α))

        Returns
        -------
        float
            Stationary standard deviation: σ/√(2α)

        Notes
        -----
        **Interpretation:**
        This is the long-term standard deviation around the mean,
        reached after transient dies out (t >> 1/α).

        **Relationship to Parameters:**
        - Increases with σ (more noise)
        - Decreases with α (stronger reversion)
        - Ratio σ/α sets scale

        **Comparison with σ:**
        - σ: Instantaneous noise intensity
        - σ/√(2α): Accumulated noise effect
        - Always: σ_stat < σ (for α > 0.5)

        **Design:**
        To achieve target std s:
            σ = s · √(2α)
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

        The time constant is the characteristic time scale for
        the process to revert to its mean. After time τ, the
        deviation from mean is reduced by factor e ≈ 0.368.

        Returns
        -------
        float
            Time constant [time units]

        Notes
        -----
        **Physical Meaning:**
        - Time to reduce deviation by ~63%
        - Reciprocal of mean reversion rate
        - Sets time scale of dynamics

        **Related Quantities:**
        - Half-life: t₁/₂ = ln(2)·τ ≈ 0.693·τ
        - 99% settling: t_99 ≈ 5·τ
        - 1% settling: t_01 ≈ 0.01·τ

        **Rule of Thumb:**
        Process reaches stationarity after 3-5 time constants.
        - t < τ: Still in transient
        - t ≈ 5τ: Essentially stationary

        **Examples:**
        - τ = 0.1s: Fast dynamics (α = 10)
        - τ = 1.0s: Moderate (α = 1)
        - τ = 10s: Slow (α = 0.1)
        """
        for key, val in self.parameters.items():
            if str(key) == "alpha":
                return 1.0 / val

        raise RuntimeError("alpha parameter not found")

    def get_half_life(self) -> float:
        """
        Get half-life t₁/₂ = ln(2)/α ≈ 0.693/α.

        Time for deviation from mean to reduce by 50%.
        More intuitive than time constant for some applications.

        Returns
        -------
        float
            Half-life [time units]
        """
        return np.log(2.0) / self.get_time_constant()


# ============================================================================
# Convenience Factory Functions
# ============================================================================


def create_ou_process(
    time_constant: float = 1.0, 
    volatility: float = 1.0
) -> OrnsteinUhlenbeck:
    """
    Create OU process with specified time constant and volatility.

    More intuitive parameterization using time constant τ instead
    of mean reversion rate α = 1/τ.

    Parameters
    ----------
    time_constant : float, default=1.0
        Time constant τ [time units]
        - Mean reversion rate: α = 1/τ
        - Typical: 0.1 to 10 seconds
        
    volatility : float, default=1.0
        Noise intensity σ
        - Stationary std: σ/√(2/τ)

    Returns
    -------
    OrnsteinUhlenbeck
        OU process with α = 1/τ, σ = volatility

    Notes
    -----
    **Time Constant Interpretation:**
    - τ = 0.1s: Fast reversion (settles in ~0.5s)
    - τ = 1.0s: Moderate (settles in ~5s)
    - τ = 10s: Slow (settles in ~50s)

    **Design Pattern:**
    Choose τ based on desired response time, then
    choose σ based on desired fluctuation magnitude.
    """
    alpha = 1.0 / time_constant
    return OrnsteinUhlenbeck(alpha=alpha, sigma=volatility)


def create_vasicek_model(
    mean_reversion: float = 0.5,
    long_term_rate: float = 0.05,
    volatility: float = 0.01,
) -> OrnsteinUhlenbeck:
    """
    Create Vasicek interest rate model.

    The Vasicek model (1977) is an OU process for short-term
    interest rates with mean reversion to a long-term level.

    Mathematical Form:
        dr = κ·(θ - r)·dt + σ·dW

    Equivalent to:
        dr = -κ·r·dt + κ·θ·dt + σ·dW
        
    which is OU with control u = κ·θ.

    Parameters
    ----------
    mean_reversion : float, default=0.5
        Mean reversion speed κ (1/year)
        - Typical: 0.1 to 2.0
        - Higher: Faster reversion to long-term rate
        
    long_term_rate : float, default=0.05
        Long-term mean interest rate θ
        - As decimal: 0.05 = 5% annual rate
        - Typical: 0.02 to 0.08
        
    volatility : float, default=0.01
        Interest rate volatility σ (1/√year)
        - Typical: 0.005 to 0.02 (0.5% to 2%)

    Returns
    -------
    OrnsteinUhlenbeck
        Vasicek model (OU process with parameters κ, σ)

    Notes
    -----
    **Historical Context:**
    Introduced by Oldřich Vašíček in 1977, this was the first
    equilibrium model of the term structure. Revolutionary for:
    - Analytical bond pricing formulas
    - Mean reversion (rates don't wander arbitrarily)
    - Stochastic calculus in finance

    **Advantages:**
    - Tractable analytical formulas
    - Mean reversion captures rate behavior
    - Simple to estimate from data

    **Limitations:**
    - Can produce negative rates (problematic)
    - Constant volatility (unrealistic)
    - Normal distribution (fat tails in reality)

    **Modern Usage:**
    Still used for:
    - Teaching and intuition
    - Baseline comparisons
    - Simple scenarios

    Superseded by:
    - CIR model (positive rates)
    - Hull-White (time-varying parameters)
    - LIBOR market models

    **Implementation Note:**
    This returns centered OU process. To match Vasicek exactly
    with long-term mean θ, set control:
        u = κ·θ

    Then equilibrium rate is u/κ = θ.

    **Stationary Distribution:**
        r(∞) ~ N(θ, σ²/(2κ))

    **Bond Pricing:**
    Zero-coupon bond price:
        P(t,T) = A(t,T)·exp(-B(t,T)·r(t))
    
    where B(t,T) and A(t,T) have closed-form expressions.
    See Also
    --------
    OrnsteinUhlenbeck : Base class
    CoxIngersollRoss : Positive-rates alternative
    """
    # Note: Returns centered OU. User must apply control u = κ·θ for correct mean.
    return OrnsteinUhlenbeck(alpha=mean_reversion, sigma=volatility)


"""
Multivariate Ornstein-Uhlenbeck Process - Coupled Mean-Reverting System
========================================================================

This module provides the multivariate (vector) Ornstein-Uhlenbeck process,
a fundamental linear stochastic system with coupling. The multivariate OU
process serves as:

- The canonical model for coupled mean-reverting processes
- Foundation for multi-asset portfolio dynamics and yield curve modeling
- A benchmark for testing multivariate Kalman filtering algorithms
- The continuous-time limit of Vector AutoRegressive (VAR) models
- A model for coupled Langevin equations in statistical physics

The multivariate OU process generalizes the scalar OU by allowing:
1. **Coupling:** Components influence each other's dynamics
2. **Correlation:** Noise sources can be correlated
3. **Asymmetric reversion:** Different time scales and equilibria
4. **Rich dynamics:** Oscillations, exponential modes, complex eigenvalues

This enables modeling of realistic systems where variables interact,
such as interest rates at different maturities, correlated asset prices,
or coupled physical oscillators.

Mathematical Background
-----------------------

**Scalar OU Process:**
    dx = -α·x·dt + σ·dW

Single variable, simple exponential decay to zero.

**Multivariate Extension:**
    dX = A·X·dt + Σ·dW

where:
- X ∈ ℝⁿ: State vector (n variables)
- A ∈ ℝⁿˣⁿ: Drift matrix (coupling + mean reversion)
- Σ ∈ ℝⁿˣᵐ: Diffusion matrix (noise intensities)
- W ∈ ℝᵐ: Vector Wiener process (m noise sources)

**Key Generalizations:**

1. **Coupling via A:**
   Off-diagonal elements: A_ij ≠ 0 for i ≠ j
   - X_i influences dX_j/dt
   - Creates interaction between components

2. **Correlated Noise:**
   If m < n or Σ non-diagonal:
   - Noise sources correlated
   - Common shocks affect multiple variables

3. **Stability Matrix A:**
   Eigenvalues determine behavior:
   - All Re(λ) < 0: Stable (stationary distribution exists)
   - Any Re(λ) > 0: Unstable (diverges)
   - Complex λ: Oscillatory decay

**With Control:**
    dX = (A·X + B·u)·dt + Σ·dW

Adds control input u ∈ ℝᵖ via control matrix B ∈ ℝⁿˣᵖ.

**Non-Centered Form:**
    dX = A·(μ - X)·dt + Σ·dW

Reverts to mean vector μ instead of zero.

Analytical Solution
-------------------

**Exact Solution:**

For X(0) = X₀ and constant control u:
    X(t) = e^(At)·X₀ + ∫₀ᵗ e^(A(t-s))·B·u·ds + ∫₀ᵗ e^(A(t-s))·Σ·dW(s)

**Moments:**

Mean:
    E[X(t)] = e^(At)·X₀ + A^(-1)·(e^(At) - I)·B·u

For stable A (eigenvalues in LHP):
    E[X(∞)] = -A^(-1)·B·u

Variance (Lyapunov equation):
    Σ_X(t) satisfies: dΣ_X/dt = A·Σ_X + Σ_X·Aᵀ + Σ·Qₙ·Σᵀ

where Qₙ is noise covariance (often identity).

**Stationary Covariance (t → ∞):**
    A·Σ_∞ + Σ_∞·Aᵀ + Σ·Qₙ·Σᵀ = 0

This is a **matrix Lyapunov equation** - linear in Σ_∞.

Solved via: Σ_∞ = ∫₀^∞ e^(As)·Σ·Qₙ·Σᵀ·e^(Aᵀs)·ds

**Stationary Distribution:**
    X(∞) ~ N(-A^(-1)·B·u, Σ_∞)

Multivariate Gaussian with explicit mean and covariance.

Key Properties
--------------

**1. Linearity:**
Linear drift A·X enables exact analytical solutions.

**2. Stationarity:**
If all eigenvalues of A have negative real parts, process is stationary
with Gaussian equilibrium distribution.

**3. Gaussian Process:**
X(t) is multivariate Gaussian for all t (linearity preserves Gaussianity).

**4. Markov Property:**
Future independent of past given present (memoryless).

**5. Affine Structure:**
Mean and covariance affine in initial conditions and control.

**6. Exponential Correlation:**
Autocorrelation decays exponentially (eigenvalue-dependent rates).

**7. Coupling:**
Off-diagonal A creates interaction:
- Positive A_ij: X_i increases when X_j positive (co-movement)
- Negative A_ij: X_i decreases when X_j positive (opposition)

**8. Noise Correlation:**
Diffusion Σ·Σᵀ determines instantaneous correlations:
- Diagonal: Independent noise sources
- Non-diagonal: Correlated disturbances

Physical Interpretation
-----------------------

**Drift Matrix A:**

Diagonal elements A_ii < 0:
- Mean reversion rate for X_i
- Faster reversion: More negative A_ii
- Time constant: τ_i ≈ -1/A_ii

Off-diagonal elements A_ij (i ≠ j):
- Coupling strength from X_j to X_i
- Positive: Reinforcing (co-movement)
- Negative: Opposing (anti-correlation)

**Diffusion Matrix Σ:**

Each row: Σ_i,· determines noise for dX_i
- Diagonal Σ: Independent noise per component
- Non-diagonal: Common factors driving multiple components

**Noise Covariance:**
    Q = Σ·Σᵀ

Instantaneous covariance of noise increments.

**Stationary Covariance Σ_∞:**

Encodes long-term correlations:
- (Σ_∞)_ii: Variance of X_i
- (Σ_∞)_ij: Covariance between X_i and X_j
- Correlation: ρ_ij = (Σ_∞)_ij / √((Σ_∞)_ii·(Σ_∞)_jj)

Applications
------------

**1. Financial Economics:**

**Multi-Factor Interest Rate Models:**
    dr_i = κ_i·(θ_i - r_i)·dt + Σ σ_ij·dW_j

Multiple rates (short, medium, long) coupled:
- Short rate mean-reverts faster (large κ)
- Long rate slower (small κ)
- Correlation via common factors

**Yield Curve Dynamics:**
Entire yield curve as vector OU process.

**Portfolio Dynamics:**
Multiple correlated assets:
    dS_i/S_i = μ_i·dt + Σ σ_ij·dW_j

Log-prices follow multivariate OU.

**Pairs Trading:**
Spread between two assets:
    S_spread = S_1 - β·S_2

Often modeled as scalar OU, but better as 2D OU.

**FX Rates:**
Exchange rate triangles (consistency constraints).

**2. Physics:**

**Coupled Langevin Equations:**
Multiple particles with coupling:
    dv_i = -γ·v_i·dt + Σ K_ij·(x_j - x_i)·dt + σ·dW_i

Spring coupling + individual noise.

**Velocity Distribution:**
Maxwell-Boltzmann for each component, correlated.

**Coupled Harmonic Oscillators:**
Normal modes as independent OU processes.

**3. Neuroscience:**

**Neural Population:**
Multiple neurons with synaptic coupling:
    dV_i = -(V_i - V_rest)/τ_m·dt + Σ w_ij·s_j·dt + σ·dW_i

Connectivity matrix w_ij creates correlations.

**4. Climate Science:**

**Multi-Region Temperature:**
Temperatures in different regions coupled via heat transport.

**5. Ecology:**

**Multi-Species Dynamics:**
Linearized Lotka-Volterra around equilibrium with noise.

**6. Control Systems:**

**Multi-Variable LQG:**
Coupled subsystems with noise:
- MIMO control (multiple inputs, multiple outputs)
- Cross-coupling in dynamics
- Optimal coordinated control

Connection to VAR Models
-------------------------

**Vector Autoregressive VAR(1):**

Discrete-time multivariate AR:
    X[k+1] = Φ·X[k] + ε[k]

where ε[k] ~ N(0, Σ_ε).

**Relationship to Multivariate OU:**

Exact discretization of multivariate OU with sampling Δt:
    Φ = exp(A·Δt)
    Σ_ε = ∫₀^Δt exp(A·s)·Σ·Qₙ·Σᵀ·exp(Aᵀ·s)·ds

**Conversion:**

From continuous to discrete:
- Eigenvalues: λ_discrete = exp(λ_continuous·Δt)
- Stability: Re(λ_cont) < 0 ↔ |λ_disc| < 1

From discrete to continuous (approximate):
    A ≈ (Φ - I)/Δt for small Δt

**Use Cases:**
- Continuous OU: Theoretical analysis, high-frequency data
- Discrete VAR: Econometric estimation, prediction

Eigenvalue Analysis
-------------------

**Stability:**

All eigenvalues must have Re(λ) < 0 for stability.

**Eigenvalue Types:**

1. **Real Negative:** Exponential decay (no oscillation)
   - τ_i = -1/λ_i: Time constant
   - Faster decay: More negative λ

2. **Complex Conjugate Pair:** α ± iβ with α < 0
   - Damped oscillations
   - Decay rate: -α
   - Frequency: β
   - Period: 2π/β

**Mode Decomposition:**

Via eigendecomposition: A = V·Λ·V^(-1)
- V: Eigenvectors (modes)
- Λ: Eigenvalues (rates)

Each mode evolves independently:
    Y = V^(-1)·X (transformed coordinates)
    dY_i = λ_i·Y_i·dt + noise

**Physical Interpretation:**
- Fast modes: Quickly equilibrate
- Slow modes: Dominate long-term dynamics
- Oscillatory modes: Create cycles

Numerical Simulation
--------------------

**Euler-Maruyama:**
    X[k+1] = X[k] + A·X[k]·Δt + Σ·√Δt·Z[k]

where Z[k] ~ N(0, I_m).

**Exact Discretization:**
    X[k+1] = Φ·X[k] + w[k]

where:
    Φ = exp(A·Δt) (matrix exponential)
    w ~ N(0, Q_discrete) with Q computed from Lyapunov

**Recommended:**
For moderate n (≤ 10): Exact discretization
For large n: Euler-Maruyama with small Δt

Common Configurations
---------------------

**1. Diagonal A (Independent OU):**
    A = diag(-α_1, -α_2, ..., -α_n)

Each component independent OU with own reversion rate.

**2. Block Diagonal:**
Groups of coupled variables, no cross-group coupling.

**3. Circulant:**
Translation-invariant coupling (periodic boundary).

**4. Tridiagonal:**
Nearest-neighbor coupling (spatial discretization).

**5. Full (Dense):**
All components coupled (most general).

Limitations
-----------
- Linearity (no nonlinear interactions)
- Constant parameters (no time-variation)
- Gaussian (no heavy tails)
- Additive noise only
- Stationary dynamics only

Extensions
----------
- Nonlinear drift: dX = f(X)·dt + Σ·dW
- Time-varying: A(t), Σ(t)
- Regime-switching: Parameters switch between states
- Jump component: Add Poisson jumps
- Infinite-dimensional: Stochastic PDEs

"""

import numpy as np
import sympy as sp
from typing import Optional, Union

from src.systems.base.core.continuous_stochastic_system import ContinuousStochasticSystem


class MultivariateOrnsteinUhlenbeck(ContinuousStochasticSystem):
    """
    Multivariate Ornstein-Uhlenbeck process with coupling and correlated noise.

    Generalizes scalar OU to vector-valued process with interaction between
    components and correlated noise sources. This is the fundamental linear
    stochastic system for modeling coupled mean-reverting processes.

    Stochastic Differential Equation
    ---------------------------------
    Vector SDE:
        dX = A·X·dt + Σ·dW

    With control:
        dX = (A·X + B·u)·dt + Σ·dW

    where:
        - X ∈ ℝⁿ: State vector
        - A ∈ ℝⁿˣⁿ: Drift matrix (coupling + mean reversion)
        - B ∈ ℝⁿˣᵖ: Control matrix
        - u ∈ ℝᵖ: Control input
        - Σ ∈ ℝⁿˣᵐ: Diffusion matrix
        - W ∈ ℝᵐ: Vector Wiener process (m noise sources)
        - dW ~ N(0, I_m·dt): Independent increments

    **Matrix Dimensions:**
    - n states
    - m noise sources (typically m ≤ n)
    - p control inputs

    Physical Interpretation
    -----------------------
    **Drift Matrix A:**

    Diagonal elements A_ii:
    - Mean reversion for X_i
    - Must be negative for stability: A_ii < 0
    - Time constant: τ_i = -1/A_ii

    Off-diagonal elements A_ij (i ≠ j):
    - Coupling from X_j to dX_i/dt
    - Positive: X_j increases → X_i increases (co-movement)
    - Negative: X_j increases → X_i decreases (opposition)
    - Zero: No direct coupling

    **Example (2D):**
        A = [-α₁   γ  ]
            [ γ   -α₂ ]

    - Own reversion: -α₁, -α₂ (diagonal)
    - Cross-coupling: γ (off-diagonal)
    - Symmetric: Bidirectional coupling

    **Diffusion Matrix Σ:**

    Each row Σ_i,· determines noise for X_i:
    - If m = n and Σ diagonal: Independent noise per component
    - If m < n: Common factors (dimension reduction)
    - Non-diagonal: Correlated noise sources

    **Noise Covariance:**
        Q = Σ·Σᵀ ∈ ℝⁿˣⁿ

    Instantaneous noise covariance:
    - Q_ii: Variance rate for X_i
    - Q_ij: Covariance rate between X_i and X_j

    Key Features
    ------------
    **Coupling:**
    Components interact via drift matrix A.
    Creates rich dynamics impossible in independent processes.

    **Correlation:**
    Noise sources can be correlated via Σ.
    Models common shocks affecting multiple variables.

    **Stability:**
    Stable if all eigenvalues of A have Re(λ) < 0.

    **Stationary Distribution:**
    X(∞) ~ N(-A^(-1)·B·u, Σ_∞)

    where Σ_∞ from Lyapunov equation.

    **Gaussian:**
    Linear dynamics preserve multivariate Gaussianity.

    **Dimension Reduction:**
    If m < n, dynamics driven by m < n factors.
    Used in factor models (finance).

    Mathematical Properties
    -----------------------
    **Stationary Covariance:**

    Lyapunov equation:
        A·Σ_∞ + Σ_∞·Aᵀ + Q = 0

    where Q = Σ·Σᵀ.

    Solutions:
    - Direct: Analytical (small n)
    - Numerical: scipy.linalg.solve_continuous_lyapunov
    - Eigenvalue: Via eigendecomposition of A

    **Autocorrelation:**
        Cov[X(t), X(t+τ)] = Σ_∞·exp(Aᵀτ)

    Matrix exponential decay.

    **Eigenvalue Decomposition:**
        A = V·Λ·V^(-1)

    Solution in eigenbasis:
        Y = V^(-1)·X (modal coordinates)
        Y_i(t) = Y_i(0)·exp(λ_i·t) + noise

    Each mode independent OU with rate λ_i.

    **Cross-Correlation Structure:**

    For components i, j:
        Corr[X_i(t), X_j(t+τ)] = (Σ_∞)_ij·exp(dominant eigenvalue·τ)

    Determined by slowest (least negative) eigenvalue.

    Physical Interpretation
    -----------------------
    **Two-Asset Example:**

    Consider two correlated assets:
        dX₁ = -α₁·X₁·dt + γ·X₂·dt + σ₁·dW₁
        dX₂ = γ·X₁·dt - α₂·X₂·dt + σ₂·dW₂

    Interpretation:
    - -α_i·X_i: Own mean reversion
    - γ·X_j: Spillover from other asset
    - σ_i·dW_i: Idiosyncratic shocks

    **Yield Curve Example:**

    Interest rates at different maturities:
        dr_short = -κ_s·r_short·dt + ...
        dr_long = -κ_l·r_long·dt + coupling·dr_short·dt + ...

    Short rate affects long rate (term structure).

    State Space
    -----------
    State: X ∈ ℝⁿ
        - Vector of coupled variables
        - Unbounded (Gaussian)
        - Equilibrium at -A^(-1)·B·u

    Control: u ∈ ℝᵖ (optional)
        - Vector control input
        - Can be state feedback: u = u(X)

    Noise: W ∈ ℝᵐ
        - m independent Wiener processes
        - m ≤ n typically (factor structure)

    Parameters
    ----------
    A : np.ndarray or list, shape (n, n)
        Drift matrix (must have eigenvalues with Re(λ) < 0 for stability)
        - Diagonal: Own mean reversion
        - Off-diagonal: Coupling

    Sigma : np.ndarray or list, shape (n, m)
        Diffusion matrix
        - Each row: Noise coefficients for one state
        - Can be square (m=n) or rectangular (m<n)

    B : Optional[np.ndarray], shape (n, p)
        Control matrix (if system is controlled)

    Stochastic Properties
    ---------------------
    - System Type: LINEAR
    - Noise Type: ADDITIVE (constant Σ)
    - SDE Type: Itô
    - Noise Dimension: nw = m
    - Stationary: Yes (if A stable)
    - Ergodic: Yes (if A stable)
    - Gaussian: Yes (always)

    Applications
    ------------
    **1. Finance:**
    - Multi-factor interest rate models
    - Correlated asset dynamics
    - Yield curve modeling
    - Portfolio optimization

    **2. Physics:**
    - Coupled Langevin equations
    - Multiple particles in potential
    - Collective modes

    **3. Economics:**
    - Multi-country VAR models
    - Spillover effects
    - Policy transmission

    **4. Neuroscience:**
    - Neural population dynamics
    - Synaptic coupling
    - Network oscillations

    **5. Climate:**
    - Multi-region temperatures
    - Heat transport coupling

    **6. Control:**
    - Multi-variable LQG
    - Coordinated control
    - Decentralized vs centralized

    Numerical Integration
    ---------------------
    **Euler-Maruyama:**
        X[k+1] = X[k] + A·X[k]·Δt + Σ·√Δt·Z[k]

    **Exact Discretization:**
        X[k+1] = Φ·X[k] + w[k]

    where Φ = exp(A·Δt) (matrix exponential).

    **Recommended:** Exact for moderate n (fast matrix exponential).

    Eigenvalue Analysis
    -------------------
    **Stability Check:**
        eigenvalues, eigenvectors = np.linalg.eig(A)
        stable = np.all(np.real(eigenvalues) < 0)

    **Time Scales:**
        time_constants = -1 / np.real(eigenvalues)

    **Oscillatory Modes:**
        frequencies = np.imag(eigenvalues) / (2*np.pi)

    Comparison with Scalar OU
    --------------------------
    **Scalar OU:**
    - 1D, no coupling
    - Single time constant
    - Exponential correlation

    **Multivariate OU:**
    - nD, with coupling
    - Multiple time scales
    - Complex correlation structure

    **When Coupling Matters:**
    - Asset correlations (portfolio risk)
    - Spillover effects (contagion)
    - Coordinated control (multi-agent)

    Limitations
    -----------
    - Linear dynamics only
    - Constant A, Σ
    - Gaussian distribution
    - No jumps
    - Stationary assumptions

    Extensions
    ----------
    - Nonlinear: dX = f(X)·dt + Σ·dW
    - Time-varying: A(t), Σ(t)
    - Regime-switching: Parameters switch
    - Infinite-dimensional: SPDE


    See Also
    --------
    OrnsteinUhlenbeck : Scalar version
    VectorAutoRegressive : Discrete-time analog
    """

    def define_system(
        self,
        A: Union[np.ndarray, list],
        Sigma: Union[np.ndarray, list],
        B: Optional[Union[np.ndarray, list]] = None,
    ):
        """
        Define multivariate OU process dynamics.

        Parameters
        ----------
        A : np.ndarray or list, shape (n, n)
            Drift matrix
            - Diagonal: Mean reversion rates (must be negative)
            - Off-diagonal: Coupling strengths
            - Must be stable: All Re(λ) < 0

        Sigma : np.ndarray or list, shape (n, m)
            Diffusion matrix
            - n states, m noise sources
            - Can be square (m=n) or rectangular (m<n)
            - Q = Σ·Σᵀ is noise covariance

        B : Optional[np.ndarray], shape (n, p)
            Control matrix (if controlled system)
            - p control inputs
            - If None, system is autonomous

        Raises
        ------
        ValueError
            If A is not square, or dimensions incompatible

        UserWarning
            If A has eigenvalues with Re(λ) ≥ 0 (unstable)

        Notes
        -----
        **Stability Requirement:**

        For stationary distribution, A must be stable:
            All eigenvalues: Re(λ) < 0

        Check: eigenvalues, _ = np.linalg.eig(A)

        **Coupling Structure:**

        A encodes interactions:
        - Diagonal elements: Self-dynamics (must be negative)
        - Off-diagonal: Cross-effects
        - Symmetric A: Detailed balance (reversible)
        - Asymmetric A: Non-reversible (cycles possible)

        **Noise Structure:**

        Common configurations:
        1. **Independent (m=n, Σ diagonal):**
           Each component has own noise source

        2. **Factor model (m<n):**
           States driven by fewer factors
           Example: n=10 rates, m=3 factors (level, slope, curvature)

        3. **Correlated (Σ non-diagonal):**
           Common shocks affect multiple components

        **Stationary Covariance:**

        Solve Lyapunov equation:
            A·Σ_∞ + Σ_∞·Aᵀ + Σ·Σᵀ = 0

        Gives long-term covariance structure.

        **Dimension Guidelines:**
        - Small (n ≤ 5): Fully coupled, dense A
        - Medium (n = 5-20): Sparse A (limited coupling)
        - Large (n > 20): Factor structure (m << n)

        Examples
        --------
        >>> # 2D coupled OU
        >>> A = [[-1.0, 0.2],    # X₂ pulls on X₁
        ...      [0.3, -2.0]]    # X₁ pulls on X₂
        >>> Sigma = [[0.5, 0.0],
        ...          [0.0, 1.0]]
        >>> 
        >>> mou = MultivariateOrnsteinUhlenbeck(A=A, Sigma=Sigma)
        >>> 
        >>> # Check stability
        >>> eigenvalues = np.linalg.eigvals(np.array(A))
        >>> print(f"Eigenvalues: {eigenvalues}")
        >>> print(f"Stable: {np.all(np.real(eigenvalues) < 0)}")
        """
        # Convert to numpy arrays
        A = np.array(A, dtype=float)
        Sigma = np.array(Sigma, dtype=float)
        
        # Validate dimensions
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError(f"A must be square, got shape {A.shape}")
        
        n = A.shape[0]  # Number of states
        
        if Sigma.ndim != 2:
            raise ValueError(f"Sigma must be 2D, got shape {Sigma.shape}")
        if Sigma.shape[0] != n:
            raise ValueError(
                f"Sigma must have {n} rows to match A, got {Sigma.shape[0]}"
            )
        
        m = Sigma.shape[1]  # Number of noise sources
        
        # Check stability
        eigenvalues = np.linalg.eigvals(A)
        if not np.all(np.real(eigenvalues) < 0):
            import warnings
            warnings.warn(
                f"Drift matrix A has eigenvalues with Re(λ) ≥ 0: {eigenvalues}. "
                f"System is unstable - no stationary distribution exists. "
                f"For stable multivariate OU, ensure all Re(λ) < 0.",
                UserWarning
            )

        # Process control matrix
        if B is not None:
            B = np.array(B, dtype=float)
            if B.shape[0] != n:
                raise ValueError(f"B must have {n} rows, got {B.shape[0]}")
            p = B.shape[1]  # Number of controls
        else:
            B = np.zeros((n, 1))  # Dummy for autonomous
            p = 0

        # Store matrices for later use
        self._A_matrix = A
        self._Sigma_matrix = Sigma
        self._B_matrix = B if B is not None else None
        self._n_states = n
        self._n_noise = m
        self._n_controls = p

        # Create symbolic variables
        state_vars = [sp.symbols(f"x{i}", real=True) for i in range(n)]
        
        if p > 0:
            control_vars = [sp.symbols(f"u{i}", real=True) for i in range(p)]
        else:
            control_vars = []

        # Create symbolic parameter matrices
        # For simplicity, we'll use numerical values directly in symbolic form
        # This avoids creating n² + n·m symbolic parameters
        
        # Build drift symbolically: A·X (+ B·u if controlled)
        X_vec = sp.Matrix(state_vars)
        A_sym = sp.Matrix(A)
        
        if p > 0:
            u_vec = sp.Matrix(control_vars)
            B_sym = sp.Matrix(B)
            drift = A_sym * X_vec + B_sym * u_vec
        else:
            drift = A_sym * X_vec

        # Build diffusion: Σ (constant matrix)
        Sigma_sym = sp.Matrix(Sigma)

        # System definition
        self.state_vars = state_vars
        self.control_vars = control_vars
        self._f_sym = drift
        self.diffusion_expr = Sigma_sym
        
        # Parameters (matrices stored as arrays, not individual symbols)
        self.parameters = {}  # Matrices already in symbolic expressions
        self.order = 1
        self.sde_type = "ito"

        # Output: Full state
        self._h_sym = X_vec

    def get_drift_matrix(self) -> np.ndarray:
        """
        Get drift matrix A.

        Returns
        -------
        np.ndarray
            Drift matrix (n, n)
        """
        return self._A_matrix

    def get_diffusion_matrix(self) -> np.ndarray:
        """
        Get diffusion matrix Σ.

        Returns
        -------
        np.ndarray
            Diffusion matrix (n, m)
        """
        return self._Sigma_matrix

    def get_noise_covariance(self) -> np.ndarray:
        """
        Get instantaneous noise covariance Q = Σ·Σᵀ.

        Returns
        -------
        np.ndarray
            Noise covariance (n, n)

        Examples
        --------
        >>> mou = MultivariateOrnsteinUhlenbeck(
        ...     A=[[-1, 0], [0, -2]],
        ...     Sigma=[[1, 0], [0, 2]]
        ... )
        >>> Q = mou.get_noise_covariance()
        >>> print(f"Noise covariance:\\n{Q}")
        """
        return self._Sigma_matrix @ self._Sigma_matrix.T

    def get_stationary_covariance(self) -> np.ndarray:
        """
        Compute stationary covariance Σ_∞.

        Solves Lyapunov equation: A·Σ_∞ + Σ_∞·Aᵀ + Q = 0

        Returns
        -------
        np.ndarray
            Stationary covariance (n, n)

        Raises
        ------
        ValueError
            If A is unstable (no stationary distribution)

        Examples
        --------
        >>> mou = MultivariateOrnsteinUhlenbeck(
        ...     A=[[-1, 0], [0, -2]],
        ...     Sigma=[[1, 0], [0, 1]]
        ... )
        >>> Sigma_inf = mou.get_stationary_covariance()
        >>> print(f"Stationary covariance:\\n{Sigma_inf}")
        """
        from scipy.linalg import solve_continuous_lyapunov
        
        # Check stability
        eigenvalues = np.linalg.eigvals(self._A_matrix)
        if not np.all(np.real(eigenvalues) < 0):
            raise ValueError(
                "Cannot compute stationary covariance for unstable system. "
                f"Eigenvalues: {eigenvalues}"
            )
        
        Q = self.get_noise_covariance()
        
        # Solve: A·Σ + Σ·Aᵀ + Q = 0
        # scipy solves: A·X + X·Aᵀ = -Q
        Sigma_inf = solve_continuous_lyapunov(self._A_matrix, -Q)
        
        return Sigma_inf

    def get_correlation_matrix(self) -> np.ndarray:
        """
        Get stationary correlation matrix.

        Returns
        -------
        np.ndarray
            Correlation matrix (n, n) with 1s on diagonal

        Examples
        --------
        >>> mou = MultivariateOrnsteinUhlenbeck(
        ...     A=[[-1, 0.5], [0.5, -1]],
        ...     Sigma=[[1, 0], [0, 1]]
        ... )
        >>> rho = mou.get_correlation_matrix()
        >>> print(f"Correlation:\\n{rho}")
        """
        Sigma_inf = self.get_stationary_covariance()
        std = np.sqrt(np.diag(Sigma_inf))
        
        return Sigma_inf / np.outer(std, std)

    def get_eigenvalues(self) -> np.ndarray:
        """
        Get eigenvalues of drift matrix A.

        Returns
        -------
        np.ndarray
            Eigenvalues (may be complex)

        Notes
        -----
        Eigenvalues determine:
        - Stability: All Re(λ) < 0 required
        - Time scales: τ_i = -1/Re(λ_i)
        - Oscillations: Im(λ) ≠ 0 → periodic components

        Examples
        --------
        >>> mou = MultivariateOrnsteinUhlenbeck(
        ...     A=[[-1, 2], [-2, -1]],
        ...     Sigma=[[1, 0], [0, 1]]
        ... )
        >>> eigs = mou.get_eigenvalues()
        >>> print(f"Eigenvalues: {eigs}")
        >>> if np.any(np.imag(eigs) != 0):
        ...     print("System has oscillatory modes")
        """
        return np.linalg.eigvals(self._A_matrix)

    def get_time_constants(self) -> np.ndarray:
        """
        Get time constants from eigenvalues: τ_i = -1/Re(λ_i).

        Returns
        -------
        np.ndarray
            Time constants [s]

        Examples
        --------
        >>> mou = MultivariateOrnsteinUhlenbeck(
        ...     A=[[-1, 0], [0, -5]],
        ...     Sigma=[[1, 0], [0, 1]]
        ... )
        >>> tau = mou.get_time_constants()
        >>> print(f"Time constants: {tau}")  # [1.0, 0.2]
        """
        eigenvalues = self.get_eigenvalues()
        return -1.0 / np.real(eigenvalues)


# Convenience functions
def create_two_asset_model(
    alpha1: float = 1.0,
    alpha2: float = 1.0,
    coupling: float = 0.5,
    sigma1: float = 0.5,
    sigma2: float = 0.5,
    correlation: float = 0.0,
) -> MultivariateOrnsteinUhlenbeck:
    """
    Create 2D OU for two coupled assets.

    Parameters
    ----------
    alpha1, alpha2 : float
        Mean reversion rates for each asset
    coupling : float
        Cross-coupling strength
    sigma1, sigma2 : float
        Noise intensities
    correlation : float
        Noise correlation (-1 to 1)

    Returns
    -------
    MultivariateOrnsteinUhlenbeck

    Examples
    --------
    >>> # Weakly coupled, uncorrelated noise
    >>> assets = create_two_asset_model(
    ...     alpha1=1.0,
    ...     alpha2=2.0,
    ...     coupling=0.2,
    ...     correlation=0.0
    ... )
    """
    A = np.array([[-alpha1, coupling],
                  [coupling, -alpha2]])
    
    # Construct Σ from marginal sigmas and correlation
    if correlation == 0:
        Sigma = np.array([[sigma1, 0],
                          [0, sigma2]])
    else:
        # Cholesky-like decomposition to get correlation
        Sigma = np.array([[sigma1, 0],
                          [sigma2*correlation, sigma2*np.sqrt(1-correlation**2)]])
    
    return MultivariateOrnsteinUhlenbeck(A=A, Sigma=Sigma)


def create_yield_curve_model(
    n_factors: int = 3,
    decay_rates: Optional[list] = None,
) -> MultivariateOrnsteinUhlenbeck:
    """
    Create multi-factor yield curve model.

    Standard: 3 factors (level, slope, curvature).

    Parameters
    ----------
    n_factors : int, default=3
        Number of factors
    decay_rates : Optional[list]
        Decay rates for each factor (if None, use defaults)

    Returns
    -------
    MultivariateOrnsteinUhlenbeck

    Notes
    -----
    Standard configuration:
    - Factor 1 (level): Slow decay (persistent)
    - Factor 2 (slope): Medium decay
    - Factor 3 (curvature): Fast decay

    Examples
    --------
    >>> # Three-factor yield curve
    >>> yield_model = create_yield_curve_model(n_factors=3)
    """
    if decay_rates is None:
        # Default: Decreasing time constants
        decay_rates = [0.1, 0.5, 2.0][:n_factors]
    
    A = -np.diag(decay_rates)
    Sigma = np.eye(n_factors)
    
    return MultivariateOrnsteinUhlenbeck(A=A, Sigma=Sigma)


def create_oscillatory_ou(
    damping: float = 1.0,
    frequency: float = 2.0,
    noise_level: float = 0.5,
) -> MultivariateOrnsteinUhlenbeck:
    """
    Create 2D OU with oscillatory modes (complex eigenvalues).

    Parameters
    ----------
    damping : float
        Decay rate (negative real part)
    frequency : float
        Oscillation frequency (imaginary part)
    noise_level : float
        Noise intensity

    Returns
    -------
    MultivariateOrnsteinUhlenbeck

    Notes
    -----
    Creates A with eigenvalues: -damping ± i·frequency

    Examples
    --------
    >>> # Damped oscillator with noise
    >>> osc = create_oscillatory_ou(
    ...     damping=0.5,
    ...     frequency=3.0,
    ...     noise_level=1.0
    ... )
    """
    # A with eigenvalues -damping ± i·frequency
    A = np.array([[-damping, frequency],
                  [-frequency, -damping]])
    
    Sigma = noise_level * np.eye(2)
    
    return MultivariateOrnsteinUhlenbeck(A=A, Sigma=Sigma)