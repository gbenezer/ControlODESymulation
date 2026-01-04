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
Brownian Motion - Pure Diffusion Stochastic Systems
===================================================

This module provides implementations of Brownian motion (Wiener process) and
related stochastic processes. These are the fundamental building blocks of
stochastic differential equations and serve as:

- The canonical example of continuous-time stochastic processes
- The foundation for all Itô and Stratonovich SDEs
- A model for diffusion processes in physics, finance, and biology
- The continuous-time limit of random walks
- A benchmark for testing SDE integration methods

Brownian motion represents pure random diffusion with no deterministic drift,
making it the stochastic analog of a zero dynamics system in deterministic
control theory.

Mathematical Background
-----------------------
Standard Brownian motion W(t) (also called the Wiener process) is characterized
by four fundamental properties:

1. **W(0) = 0**: Starts at the origin (with probability 1)
2. **Independent Increments**: W(t) - W(s) is independent of W(r) for r ≤ s < t
3. **Gaussian Increments**: W(t) - W(s) ~ N(0, t-s) for t > s
4. **Continuous Paths**: W(t) is continuous in t (but nowhere differentiable)

These properties make Brownian motion:
- A Markov process (future independent of past given present)
- A martingale (expected future value equals current value)
- Self-similar (statistical properties scale with √t)
- The limit of discrete random walks as step size → 0

Physical Interpretation
-----------------------

**Physical Systems Exhibiting Brownian Motion:**

1. **Molecular Diffusion**: 
   - Pollen grains in fluid (original observation)
   - Smoke particles in air
   - Molecules in solution
   - Ions in electrolyte

2. **Thermal Noise**:
   - Johnson-Nyquist noise in resistors
   - Shot noise in electronic devices
   - Quantum fluctuations at nanoscale

3. **Financial Markets**:
   - Stock price fluctuations (geometric Brownian motion)
   - Interest rate dynamics
   - Currency exchange rates
   - Commodity prices

4. **Biology**:
   - Random molecular motion in cells
   - Protein folding trajectories
   - Gene expression stochasticity
   - Neural membrane potential fluctuations

5. **Environmental Processes**:
   - Pollutant dispersion
   - Heat diffusion with fluctuations
   - Population dynamics with demographic noise

Mathematical Properties
-----------------------

**Moments and Distribution:**
For X(t) starting at X(0) = x₀ with diffusion coefficient σ:

- Mean: E[X(t)] = x₀ (constant - no drift)
- Variance: Var[X(t)] = σ²·t (grows linearly with time)
- Standard Deviation: Std[X(t)] = σ·√t (grows with square root of time)
- Distribution: X(t) ~ N(x₀, σ²·t)

**Covariance Structure:**
- Cov[X(s), X(t)] = σ²·min(s,t)
- Correlation: ρ(s,t) = √(min(s,t)/max(s,t))
- This creates the characteristic "triangular" covariance matrix

**Path Properties:**
- Continuous everywhere but differentiable nowhere
- Hölder continuous with exponent α < 1/2
- Unbounded variation on any finite interval
- Quadratic variation on [0,T] equals σ²·T (fundamental to Itô calculus)
- Hits every real value infinitely often (recurrence)
- Returns to origin infinitely often (recurrence in 1D, transience in 3D+)

**Scaling Properties (Self-Similarity):**
For any c > 0:
    {W(c²·t) : t ≥ 0} has the same distribution as {c·W(t) : t ≥ 0}

This means Brownian motion "looks the same" at all time scales.

**First Passage Time:**
The time τ_a to first reach level a > 0 has distribution:
    P(τ_a ≤ t) = 2·Φ(-a/√t) where Φ is standard normal CDF
    E[τ_a] = ∞ (infinite expected hitting time!)

Connection to Stochastic Calculus
----------------------------------

**Itô's Lemma:**
The fundamental theorem of stochastic calculus. For a smooth function f(X(t)):
    df = (σ²/2)·f''(X)·dt + σ·f'(X)·dW

The σ²/2·f'' term (Itô correction) is unique to stochastic calculus and
arises from the quadratic variation of Brownian motion.

**Martingale Property:**
W(t) is a martingale: E[W(t)|W(s)] = W(s) for s < t
This makes Brownian motion "fair" - no expected gain or loss.

**Applications in SDE Theory:**
General SDEs have the form:
    dX = f(X,t)·dt + g(X,t)·dW

Where:
- f(X,t): Drift (deterministic tendency)
- g(X,t): Diffusion (stochastic volatility)
- dW: Brownian motion increment

Brownian motion is the special case f ≡ 0, g = constant.

Relationship to Heat Equation
------------------------------
The probability density p(x,t) of Brownian motion satisfies the heat equation:
    ∂p/∂t = (σ²/2)·∂²p/∂x²

With initial condition p(x,0) = δ(x - x₀), giving:
    p(x,t) = 1/√(2πσ²t) · exp(-(x-x₀)²/(2σ²t))

This is the Gaussian distribution N(x₀, σ²t).

**Physical Interpretation:**
- Diffusion of heat/particles obeys same equation
- Diffusion coefficient D = σ²/2
- Fick's second law of diffusion
- Einstein relation: D = kᵦT/γ (temperature/friction)

Numerical Simulation
--------------------

**Exact Sampling:**
For Brownian motion on interval [0,T] with N steps:
    dt = T/N
    W[k+1] = W[k] + √dt · Z[k]  where Z[k] ~ N(0,1)

This is exact for Brownian motion (no discretization error). For pure Brownian motion, 
Euler-Maruyama is exact.

Extensions and Generalizations
-------------------------------

**Geometric Brownian Motion:**
    dX = μ·X·dt + σ·X·dW
Used in Black-Scholes model, ensures X > 0.

**Ornstein-Uhlenbeck Process:**
    dX = -α·X·dt + σ·dW
Mean-reverting process, stationary distribution.

**Fractional Brownian Motion:**
Self-similar with Hurst parameter H ≠ 1/2, non-Markovian.

**Multidimensional Brownian Motion:**
Independent components, or correlated via covariance matrix.

**Brownian Bridge:**
Conditioned to return to specific value at final time.

**Reflected Brownian Motion:**
Bounces off boundary, used in queueing theory.

**Absorbed Brownian Motion:**
Stops upon hitting boundary, used in barrier options.

Characteristic Scales
---------------------

**Diffusion Length:**
The typical distance traveled in time t:
    ℓ(t) ~ σ·√t

This √t scaling is characteristic of diffusion processes.

**Diffusion Time:**
Time to diffuse distance d:
    τ(d) ~ (d/σ)²

This quadratic relationship makes diffusion slow at large distances.

**Examples:**
- Protein diffusion in water (σ ~ 10⁻⁶ m/s):
  - 1 nm in ~0.01 ms
  - 1 μm in ~1 s  
  - 1 mm in ~10⁶ s (~11 days!)

Validation and Testing
----------------------
Standard tests for Brownian motion simulation:

1. **Mean Test**: E[W(T)] ≈ 0
2. **Variance Test**: Var[W(T)] ≈ σ²·T
3. **Normality Test**: Kolmogorov-Smirnov, Shapiro-Wilk
4. **Independence Test**: Autocorrelation of increments ≈ 0
5. **Quadratic Variation**: Σ(ΔW)² ≈ σ²·T
6. **Path Continuity**: Max|ΔW| → 0 as Δt → 0

Common Pitfalls
---------------

1. **Confusing Brownian Motion with White Noise:**
   - White noise ξ(t) is the formal derivative: dW/dt
   - White noise is NOT a function (distribution/generalized function)
   - W(t) is the integral of white noise

2. **Wrong Scaling:**
   - Increment scales as √dt, not dt
   - Forgetting √dt leads to incorrect variance

3. **Non-Differentiability:**
   - Cannot write dW/dt in classical sense
   - Must use differential notation dW

4. **Quadratic Variation:**
   - (dW)² = dt (Itô calculus)
   - NOT zero as in classical calculus
   - Source of Itô correction term

5. **Time-Reversal:**
   - W(T-t) is NOT Brownian motion (it's not adapted)
   - Must be careful with time direction

6. **Markov Property:**
   - Valid for Brownian motion
   - NOT valid for fractional Brownian motion

7. **Boundary Behavior:**
   - Standard Brownian motion is recurrent in 1D, 2D
   - Transient in 3D+ (escapes to infinity)

The theory unifies physics (diffusion), mathematics (probability), 
finance (option pricing), and biology (molecular dynamics).

"""

import sympy as sp

from src.systems.base.core.continuous_stochastic_system import ContinuousStochasticSystem


class BrownianMotion(ContinuousStochasticSystem):
    """
    Standard Brownian motion (Wiener process) - pure diffusion process.

    This is the fundamental stochastic process underlying all stochastic
    differential equations. It represents pure random walk with no
    deterministic drift, making it the stochastic analog of a zero
    dynamics system.

    Stochastic Differential Equation
    ---------------------------------
    Continuous-time SDE:
        dX = σ·dW

    where:
        - X(t): State (position) at time t
        - σ > 0: Diffusion coefficient (volatility/noise intensity)
        - W(t): Standard Wiener process
        - dW: Brownian motion increment (infinitesimal random variable)
        - No drift term: f(X) = 0
        - Additive noise: σ is constant (state-independent)

    **Key Distinction from Other SDEs:**
    Unlike most SDEs, Brownian motion has:
    - Zero drift (f = 0): No deterministic tendency
    - Constant diffusion (g = σ): Noise intensity independent of state
    - This makes it the "simplest" non-trivial stochastic process

    Mathematical Properties
    -----------------------
    **Distribution:**
    Starting at X(0) = x₀, the solution is:
        X(t) ~ N(x₀, σ²·t)

    **Moments:**
    - Mean: E[X(t)] = x₀ (constant for all time)
    - Variance: Var[X(t)] = σ²·t (grows linearly)
    - Standard Deviation: Std[X(t)] = σ·√t
    - Skewness: 0 (symmetric)
    - Kurtosis: 3 (Gaussian)

    **Path Properties:**
    - Continuous everywhere
    - Differentiable nowhere (fractal)
    - Hölder continuous: |X(t) - X(s)| ≤ C·|t-s|^α for α < 1/2
    - Unbounded variation: paths are infinitely "wiggly"
    - Quadratic variation: ∫(dX)² = σ²·t (fundamental to Itô calculus)

    **Markov Property:**
    The future evolution depends only on current state, not history:
        P(X(t) | X(s), s ≤ u) = P(X(t) | X(u)) for t > u

    **Martingale Property:**
    Expected future value equals current value:
        E[X(t) | X(s)] = X(s) for t > s

    This makes Brownian motion "fair" - no expected gain or loss.

    **Self-Similarity:**
    For any c > 0:
        {σ·W(c²·t) : t ≥ 0} has the same distribution as {c·σ·W(t) : t ≥ 0}

    Physical Interpretation
    -----------------------
    **Diffusion Coefficient σ:**
    Controls the "speed" of diffusion:
    - Larger σ: Faster diffusion, more volatile paths
    - Smaller σ: Slower diffusion, smoother paths
    - Diffusion length scale: ℓ ~ σ·√t

    **Relationship to Temperature:**
    In physical systems:
        σ = √(2·D) where D = kᵦT/γ
    - D: Diffusion coefficient
    - kᵦ: Boltzmann constant
    - T: Temperature
    - γ: Friction coefficient

    Higher temperature → larger σ → faster diffusion

    Parameters
    ----------
    sigma : float, default=1.0
        Diffusion coefficient (volatility) [units depend on application]
        - Must be positive: σ > 0
        - Controls noise intensity
        - Units: [state]/√[time]
        - Standard Brownian motion: σ = 1

        **Physical Meaning:**
        - σ² = 2D where D is diffusion coefficient
        - Variance growth rate: dVar/dt = σ²
        - RMS displacement in unit time: √(σ²·1) = σ

    State Space
    -----------
    State: x ∈ ℝ (unbounded)
        - Can take any real value: -∞ < x < +∞
        - No equilibria (system is non-stationary)
        - No attractors or repellers
        - All points equally likely in long-time limit (if unbounded)

    Control: None (autonomous system)
        - nu = 0: No control input
        - Purely noise-driven
        - Cannot be "steered" by external input

    **Boundary Conditions:**
    For physical applications, boundaries may be imposed:
    - Absorbing: Process stops upon hitting boundary
    - Reflecting: Process bounces back from boundary
    - Periodic: Wraps around (equivalent to circle topology)

    Stochastic Properties
    ---------------------
    **Noise Type:** ADDITIVE
    - Diffusion matrix g = σ (constant)
    - Does not depend on state x
    - Simplest form of stochastic dynamics

    **SDE Type:** Itô (standard interpretation)
    - Can also use Stratonovich (equivalent for additive noise)
    - No Itô correction needed (f independent of x)

    **Noise Dimension:** 
    - nw = 1: Single Wiener process drives the system
    - Scalar noise in scalar system

    Simulation Methods
    ------------------
    **Exact Discrete-Time Solution:**
    For time step Δt:
        X[k+1] = X[k] + σ·√(Δt)·Z[k]
    where Z[k] ~ N(0,1) is standard normal.

    This is EXACT - no discretization error for Brownian motion!

    Statistical Analysis
    --------------------
    **Hypothesis Testing:**
    To verify simulated paths are truly Brownian:

    1. **Mean Test**: E[X(T) - X(0)] = 0
       - Sample mean should be ≈0
       - Standard error: σ/√(n_samples)

    2. **Variance Test**: Var[X(T) - X(0)] = σ²·T
       - Sample variance should grow linearly with T

    3. **Normality Test**: X(T) - X(0) ~ N(0, σ²·T)
       - Use Shapiro-Wilk or Kolmogorov-Smirnov test
       - Should not reject normality

    4. **Independence Test**: Increments uncorrelated
       - Autocorrelation of ΔX should be zero
       - Ljung-Box test for white noise

    5. **Quadratic Variation Test**:
       - Σ(ΔX)² should approach σ²·T as Δt → 0
       - Fundamental property distinguishing from smooth functions

    Applications
    ------------
    **1. Physics:**
    - Particle diffusion in fluids
    - Thermal noise in electronic circuits
    - Quantum fluctuations (simplified model)
    - Langevin equation foundation

    **2. Finance:**
    - Building block for stock price models
    - Interest rate dynamics (Vasicek, Hull-White)
    - Option pricing (Black-Scholes foundation)
    - Risk-free asset path (money market account noise)

    **3. Biology:**
    - Molecular diffusion in cells
    - Random walk of bacteria (before chemotaxis)
    - Genetic drift in population genetics
    - Neural membrane potential fluctuations

    **4. Signal Processing:**
    - White noise integration
    - Random signal generation
    - Filter design and testing
    - Noise modeling

    **5. Mathematics:**
    - Foundation of stochastic calculus
    - Benchmark for SDE solvers
    - Example of martingales
    - Study of stochastic processes

    **6. Machine Learning:**
    - Stochastic gradient descent noise
    - Diffusion models (score-based generative models)
    - Langevin dynamics sampling
    - Variational inference with reparametrization

    Comparison with Other Processes
    --------------------------------
    **vs. Geometric Brownian Motion:**
    - GBM: dX = μ·X·dt + σ·X·dW (multiplicative noise)
    - BM: dX = σ·dW (additive noise)
    - GBM stays positive, BM can be negative
    - GBM for stock prices, BM for price returns

    **vs. Ornstein-Uhlenbeck:**
    - OU: dX = -α·X·dt + σ·dW (mean-reverting)
    - BM: dX = σ·dW (no mean reversion)
    - OU has stationary distribution, BM does not
    - OU for interest rates, temperatures

    **vs. Random Walk:**
    - Random walk: discrete time, discrete space
    - Brownian motion: continuous time, continuous space
    - BM is limit of random walk as Δt, Δx → 0 with Δx² ~ Δt

    **vs. Lévy Process:**
    - Lévy: Can have jumps (Poisson, stable processes)
    - Brownian: Continuous paths only
    - Brownian is special case of Lévy (Gaussian Lévy)

    Theoretical Importance
    ----------------------
    **Why Brownian Motion is Fundamental:**

    1. **Donsker's Theorem (Functional CLT):**
       Random walk → Brownian motion as step size → 0
       Makes BM the universal limit of random walks

    2. **Lévy Characterization:**
       Any continuous martingale with quadratic variation t
       must be Brownian motion (up to time change)

    3. **Feynman-Kac Formula:**
       Connects SDEs to PDEs via expectation
       Solution to heat equation = expected value over Brownian paths

    4. **Girsanov Theorem:**
       Change of measure for SDEs
       Converts drift to diffusion via change of probability measure

    5. **Reflection Principle:**
       P(max X(t) > a) = 2·P(X(T) > a)
       Used in barrier option pricing

    Limitations and Extensions
    --------------------------
    **Standard Brownian Motion Limitations:**
    - No memory (Markov property may be unrealistic)
    - Gaussian increments (real data often heavy-tailed)
    - Continuous paths (no jumps)
    - Constant volatility (time-varying in reality)
    - Linear variance growth (subdiffusion/superdiffusion in complex media)

    **Extensions to Address Limitations:**
    - Fractional Brownian motion: Long-range dependence (Hurst parameter)
    - Lévy processes: Jumps and heavy tails
    - Stochastic volatility: Time-varying σ(t)
    - Rough volatility: σ follows rough process
    - Anomalous diffusion: Variance ~ t^α, α ≠ 1

    See Also
    --------
    BrownianMotion2D : Two-dimensional independent Brownian motions
    BrownianBridge : Brownian motion conditioned on endpoints
    GeometricBrownianMotion : Multiplicative noise (stock prices)
    OrnsteinUhlenbeck : Mean-reverting Brownian motion
    """

    def define_system(self, sigma: float = 1.0):
        """
        Define standard Brownian motion dynamics.

        This sets up the stochastic differential equation:
            dX = σ·dW

        with zero drift and constant diffusion.

        Parameters
        ----------
        sigma : float, default=1.0
            Diffusion coefficient (must be positive)
            - Standard Brownian motion: σ = 1
            - Units: [state]/√[time]
            - Controls noise intensity
            - Variance growth rate: σ²

        Raises
        ------
        ValueError
            If sigma ≤ 0 (diffusion coefficient must be positive)

        Notes
        -----
        **System Structure:**
        This is a minimal stochastic system with:
        - Zero drift: f(x) = 0
        - Constant diffusion: g(x) = σ
        - Single noise source: nw = 1
        - No control input: nu = 0
        - First-order dynamics: order = 1

        **Special Properties:**
        - Pure diffusion process (no drift component)
        - Autonomous (no time dependence)
        - Additive noise (σ independent of state)
        - Unbounded state space (x ∈ ℝ)
        - No equilibrium points (non-stationary)

        **Standard vs. Scaled Brownian Motion:**
        When σ = 1, this is the standard Wiener process W(t).
        For σ ≠ 1, we have scaled Brownian motion: X(t) = σ·W(t)

        All Brownian motions can be written as:
            X(t) = x₀ + σ·W(t)
        where W(t) is standard Brownian motion.

        **Discretization:**
        The discrete-time equivalent is:
            X[k+1] = X[k] + σ·√(dt)·w[k]
        where w[k] ~ N(0,1) and dt is the time step.

        This is exact (no discretization error) for Brownian motion.

        **Comparison with Other SDEs:**
        Unlike most SDEs:
        - No drift means E[X(t)] = x₀ for all t (constant mean)
        - Constant diffusion means integration is particularly simple
        - Euler-Maruyama method is exact (no approximation error)
        - No stability concerns regardless of time step size
        """
        if sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}")

        # Define symbolic variables
        x = sp.symbols("x", real=True)
        sigma_sym = sp.symbols("sigma", positive=True)

        # System definition
        self.state_vars = [x]
        self.control_vars = []  # Autonomous (no control)

        # Drift: f(x) = 0 (zero drift - key property of Brownian motion)
        self._f_sym = sp.Matrix([[0]])

        self.parameters = {sigma_sym: sigma}
        self.order = 1

        # Diffusion: g(x) = σ (constant - additive noise)
        self.diffusion_expr = sp.Matrix([[sigma_sym]])
        self.sde_type = "ito"


class BrownianMotion2D(ContinuousStochasticSystem):
    """
    Two-dimensional Brownian motion with independent components.

    This represents two independent scalar Brownian motions evolving
    simultaneously. It models diffusion in a two-dimensional space
    with possibly different diffusion rates in each direction.

    Stochastic Differential Equation
    ---------------------------------
    System of two SDEs:
        dX₁ = σ₁·dW₁
        dX₂ = σ₂·dW₂

    where:
        - X₁, X₂: Position coordinates in 2D space
        - σ₁, σ₂: Diffusion coefficients for each dimension
        - W₁, W₂: Independent standard Wiener processes
        - dW₁, dW₂: Independent Brownian motion increments

    **Key Features:**
    - Two independent noise sources (nw = 2)
    - No coupling between dimensions
    - Diagonal diffusion matrix: G = diag(σ₁, σ₂)
    - Can be anisotropic (σ₁ ≠ σ₂)

    Mathematical Properties
    -----------------------
    **Joint Distribution:**
    The state vector [X₁(t), X₂(t)] starting at [x₁₀, x₂₀] has distribution:
        [X₁(t)]     [x₁₀]     [σ₁²·t    0   ]
        [X₂(t)] ~ N([x₂₀]  ,  [  0   σ₂²·t])

    **Independence:**
    X₁(t) and X₂(t) are independent for all t:
        Cov[X₁(t), X₂(t)] = 0

    **Radial Distance:**
    The distance from origin R(t) = √(X₁² + X₂²) follows:
        E[R²(t)] = r₀² + (σ₁² + σ₂²)·t
    where r₀² = x₁₀² + x₂₀².

    **First Passage Time:**
    For hitting a circle of radius a:
        E[τ_a] = a²/(2D) where D = (σ₁² + σ₂²)/2
    (For σ₁ = σ₂ = σ, this gives E[τ_a] = a²/(2σ²))

    Physical Interpretation
    -----------------------
    **Isotropic vs. Anisotropic Diffusion:**

    1. **Isotropic (σ₁ = σ₂):**
       - Diffuses equally in all directions
       - Circular symmetry
       - Examples: Particle in homogeneous fluid, thermal diffusion

    2. **Anisotropic (σ₁ ≠ σ₂):**
       - Different diffusion rates in x and y
       - Elliptical symmetry
       - Examples: Diffusion in crystals, flow in porous media

    **Applications:**
    - Particle tracking in 2D microscopy
    - Random walk on plane (e.g., animal foraging)
    - 2D heat diffusion with noise
    - Currency pair exchange rate fluctuations
    - Spatial point processes

    Parameters
    ----------
    sigma1 : float, default=1.0
        Diffusion coefficient for first dimension
        - Must be positive: σ₁ > 0
        - Controls diffusion rate in X₁ direction
        - Variance growth: Var[X₁(t)] = σ₁²·t

    sigma2 : float, default=1.0
        Diffusion coefficient for second dimension
        - Must be positive: σ₂ > 0
        - Controls diffusion rate in X₂ direction
        - Variance growth: Var[X₂(t)] = σ₂²·t

    State Space
    -----------
    State: x = [x₁, x₂] ∈ ℝ² (unbounded 2D space)
        - First component: x₁ (arbitrary real number)
        - Second component: x₂ (arbitrary real number)
        - Joint distribution is bivariate normal
        - No boundaries (can go anywhere in plane)

    Control: None (autonomous)
        - nu = 0: No control inputs
        - Purely noise-driven motion

    Stochastic Properties
    ---------------------
    **Noise Structure:**
    - Type: ADDITIVE (diffusion matrix constant)
    - Dimension: nw = 2 (two independent noise sources)
    - Coupling: DIAGONAL (no cross-terms)
    - Independence: W₁ ⊥ W₂ (orthogonal noise sources)

    **Diffusion Matrix:**
        G(x) = [σ₁  0 ]
               [0   σ₂]

    This diagonal structure means:
    - Each dimension has its own noise source
    - No correlation between dimensions
    - Can be simulated independently

    **Covariance Structure:**
    For times s < t:
        Cov[X₁(s), X₁(t)] = σ₁²·s
        Cov[X₂(s), X₂(t)] = σ₂²·s
        Cov[X₁(s), X₂(t)] = 0 (independent)

    Statistical Analysis
    --------------------
    **Hypothesis Tests:**

    1. **Independence Test:**
       - H₀: X₁ and X₂ are independent
       - Test: Correlation should be ≈0
       - Method: Pearson correlation test

    2. **Isotropy Test:**
       - H₀: σ₁ = σ₂ (isotropic diffusion)
       - Test: Var[X₁] = Var[X₂]
       - Method: F-test for equal variances

    3. **Normality Test:**
       - H₀: (X₁, X₂) follows bivariate normal
       - Test: Mardia's test, Henze-Zirkler test
       - Alternative: Q-Q plot for each dimension

    4. **Radial Distribution:**
       - Distance R = √(X₁² + X₂²)
       - For isotropic case: R²/t ~ χ²(2) scaled
       - Test: Goodness-of-fit to chi-squared

    Simulation Considerations
    -------------------------
    **Independent Sampling:**
    Each dimension can be sampled independently:
        X₁[k+1] = X₁[k] + σ₁·√(Δt)·Z₁[k]
        X₂[k+1] = X₂[k] + σ₂·√(Δt)·Z₂[k]
    where Z₁[k], Z₂[k] ~ N(0,1) are independent.

    Applications
    ------------
    **1. Particle Tracking:**
    - Track particles in microscopy
    - Determine diffusion coefficients
    - Identify anisotropy in medium
    - Classify motion types (confined, free, directed)

    **2. Animal Movement:**
    - Random walk models of foraging
    - Home range estimation
    - Habitat selection analysis
    - Dispersal modeling

    **3. Financial Markets:**
    - Correlated asset pairs (when extended to non-zero correlation)
    - Exchange rate dynamics
    - Portfolio diffusion
    - Risk analysis

    **4. Physics:**
    - 2D diffusion in membranes
    - Molecular motion on surfaces
    - Colloidal particle dynamics
    - Thermal fluctuations

    **5. Spatial Statistics:**
    - Point process backgrounds
    - Spatial inhomogeneity detection
    - Landscape ecology
    - Epidemiology (disease spread)

    Extensions
    ----------
    **Correlated Brownian Motion:**
    For correlated noise:
        G = [σ₁        ρ·σ₁·σ₂]
            [ρ·σ₁·σ₂       σ₂  ]
    where ρ is correlation coefficient (-1 ≤ ρ ≤ 1).

    **Reflected Brownian Motion:**
    Bounce off rectangular boundary:
        [0, L₁] × [0, L₂]

    **Drift Addition:**
    Add deterministic drift:
        dX₁ = μ₁·dt + σ₁·dW₁
        dX₂ = μ₂·dt + σ₂·dW₂

    **Time-Varying Diffusion:**
    Allow σ₁(t), σ₂(t) to vary with time.

    See Also
    --------
    BrownianMotion : One-dimensional version
    """

    def define_system(self, sigma1: float = 1.0, sigma2: float = 1.0):
        """
        Define 2D Brownian motion dynamics.

        Sets up the system:
            dX₁ = σ₁·dW₁
            dX₂ = σ₂·dW₂

        with two independent noise sources.

        Parameters
        ----------
        sigma1 : float, default=1.0
            Diffusion coefficient for first dimension
            - Must be positive: σ₁ > 0
            - Controls X₁ diffusion rate

        sigma2 : float, default=1.0
            Diffusion coefficient for second dimension
            - Must be positive: σ₂ > 0
            - Controls X₂ diffusion rate

        Raises
        ------
        ValueError
            If either sigma value is non-positive

        Notes
        -----
        **Diffusion Matrix Structure:**
        The diffusion matrix is diagonal:
            G = diag(σ₁, σ₂)

        This means:
        - Independent noise sources for each dimension
        - No correlation between X₁ and X₂ dynamics
        - Can simulate each dimension separately

        **Isotropy Condition:**
        The system is isotropic if and only if σ₁ = σ₂.
        - Isotropic: Diffusion looks the same in all directions
        - Anisotropic: Preferred directions of diffusion

        **Effective Diffusion Coefficient:**
        For radial distance R = √(X₁² + X₂²):
            D_eff = (σ₁² + σ₂²)/2

        **Variance Ellipse:**
        At time t, the state variance forms an ellipse:
        - Semi-major axis: max(σ₁, σ₂)·√t
        - Semi-minor axis: min(σ₁, σ₂)·√t
        - Aligned with coordinate axes (uncorrelated)
        """
        if sigma1 <= 0 or sigma2 <= 0:
            raise ValueError("sigma values must be positive")

        # Define symbolic variables
        x1, x2 = sp.symbols("x1 x2", real=True)
        sigma1_sym = sp.symbols("sigma1", positive=True)
        sigma2_sym = sp.symbols("sigma2", positive=True)

        # System definition
        self.state_vars = [x1, x2]
        self.control_vars = []  # Autonomous

        # Zero drift in both dimensions
        self._f_sym = sp.Matrix([[0], [0]])

        self.parameters = {sigma1_sym: sigma1, sigma2_sym: sigma2}
        self.order = 1

        # Diagonal diffusion (independent noise sources)
        self.diffusion_expr = sp.Matrix([[sigma1_sym, 0], [0, sigma2_sym]])
        self.sde_type = "ito"


# class BrownianBridge(ContinuousStochasticSystem):
#     """
#     Brownian bridge - Brownian motion conditioned on endpoints.

#     A Brownian bridge is a Brownian motion that is constrained to
#     start at one value and end at another value at a specified time.
#     It represents a "bridge" between two points in state space.

#     Mathematical Definition
#     -----------------------
#     A Brownian bridge from a to b on interval [0, T] satisfies:
#         - X(0) = a (initial condition)
#         - X(T) = b (terminal condition)
#         - Continuous sample paths
#         - Conditioned Gaussian process

#     **Relation to Standard Brownian Motion:**
#     If W(t) is standard Brownian motion:
#         B(t) = W(t) - (t/T)·W(T)
#     is a Brownian bridge from 0 to 0 on [0,T].

#     More generally, a bridge from a to b:
#         X(t) = a + (t/T)·(b-a) + σ·B(t)

#     SDE Formulation
#     ---------------
#     Time-dependent SDE:
#         dX = -(X - b)/(T - t)·dt + σ·dW

#     This drift term pushes the process toward the endpoint b,
#     with strength increasing as t → T.

#     **Limitations:**
#     - Drift term singular at t = T
#     - Requires time-dependent coefficients
#     - Not fully supported in current framework
#     - Future extension planned

#     Properties
#     ----------
#     **Distribution:**
#     At intermediate time t ∈ (0, T):
#         X(t) ~ N(a + (t/T)·(b-a), σ²·t·(T-t)/T)

#     Mean linearly interpolates from a to b:
#         E[X(t)] = a + (t/T)·(b-a)

#     Variance is parabolic (maximal at t = T/2):
#         Var[X(t)] = σ²·t·(1 - t/T)

#     **Path Properties:**
#     - Starts at a: X(0) = a
#     - Ends at b: X(T) = b
#     - Mean path is linear: μ(t) = a + (t/T)·(b-a)
#     - Variance zero at endpoints, maximal at center
#     - More constrained than free Brownian motion

#     Applications
#     ------------
#     **1. Statistics:**
#     - Missing data interpolation
#     - Conditional simulation
#     - Bayesian inference
#     - Monte Carlo variance reduction

#     **2. Finance:**
#     - Interest rate models with terminal condition
#     - Barrier options (path-dependent)
#     - Yield curve modeling
#     - Credit spread interpolation

#     **3. Physics:**
#     - Polymer configurations (end-to-end distance)
#     - Quantum tunneling paths
#     - Reaction coordinates
#     - Fluid particle trajectories

#     **4. Image Processing:**
#     - Texture synthesis
#     - Image inpainting
#     - Stochastic interpolation
#     - Noise generation

#     Simulation Methods
#     ------------------
#     **1. Sequential Conditioning:**
#     - Sample X(t₁), condition on X(T) = b
#     - Sample X(t₂) | X(t₁), X(T)
#     - Continue recursively
#     - Exact but computationally expensive

#     **2. Brownian Bridge Construction:**
#     - Sample standard Brownian motion W(t)
#     - Transform: X(t) = a + (t/T)·(b-a) + σ·[W(t) - (t/T)·W(T)]
#     - Simple and exact
#     - Requires full path of W

#     **3. SDE Integration:**
#     - Integrate time-dependent SDE
#     - Requires specialized SDE solver
#     - Can handle general σ(t)
#     - Current implementation: simplified version

#     Current Implementation Status
#     -----------------------------
#     **Limitations:**
#     - Time-dependent drift not fully supported
#     - Simplified to standard Brownian motion
#     - Warning issued on instantiation
#     - Full implementation planned for future release

#     **Workaround:**
#     Use post-processing transformation:
#     1. Simulate standard Brownian motion
#     2. Apply conditioning transformation
#     3. Adjust for endpoints a and b

#     Parameters
#     ----------
#     sigma : float, default=1.0
#         Diffusion coefficient
#         - Controls path variability
#         - Larger σ: More deviation from linear path

#     T : float, default=1.0
#         Terminal time (endpoint time)
#         - Length of time interval
#         - Affects variance profile

#     b : float, default=0.0
#         Terminal value (endpoint value)
#         - Target value at time T
#         - Combined with initial condition determines drift

#     Notes
#     -----
#     **Future Extensions:**
#     - Time-dependent drift implementation
#     - Support for general terminal conditions
#     - Efficient sampling algorithms
#     - Multidimensional bridges
#     - Stochastic differential bridges

#     **Related Processes:**
#     - **Bessel Bridge:** Radial bridge process
#     - **Excursion:** Bridge from 0 to 0 staying positive
#     - **Meander:** Bridge from 0 staying positive
#     - **Pinned Brownian Motion:** Bridge with multiple constraints

#     See Also
#     --------
#     BrownianMotion : Unconstrained Brownian motion
#     OrnsteinUhlenbeck : Mean-reverting process (similar drift structure)

#     References
#     ----------
#     .. [1] Revuz, D. & Yor, M. (1999). "Continuous Martingales and Brownian Motion"
#     .. [2] Karatzas, I. & Shreve, S. (1991). "Brownian Motion and Stochastic Calculus"
#     .. [3] Glasserman, P. (2003). "Monte Carlo Methods in Financial Engineering"
#     """

#     def define_system(self, sigma: float = 1.0, T: float = 1.0, b: float = 0.0):
#         """
#         Define Brownian bridge dynamics.

#         WARNING: Full implementation with time-dependent drift is not
#         yet supported. This creates a simplified version that behaves
#         as standard Brownian motion.

#         Parameters
#         ----------
#         sigma : float, default=1.0
#             Diffusion coefficient
            
#         T : float, default=1.0
#             Terminal time (endpoint)
            
#         b : float, default=0.0
#             Terminal value (target endpoint)

#         Notes
#         -----
#         **Current Limitations:**
#         The time-dependent drift term:
#             f(x,t) = -(x-b)/(T-t)
        
#         requires special handling not yet implemented in the framework.
        
#         **Planned Features:**
#         - Time-dependent coefficient support
#         - Automatic endpoint conditioning
#         - Efficient bridge sampling
#         - Multidimensional extension

#         **Temporary Workaround:**
#         Use BrownianMotion for now, then post-process using:
#             X_bridge(t) = X_BM(t) - (t/T)·X_BM(T) + (t/T)·b

#         Warnings
#         --------
#         UserWarning
#             Issued on instantiation to inform about limitation
#         """
#         import warnings

#         warnings.warn(
#             "BrownianBridge with time-dependent drift not fully supported yet. "
#             "Use BrownianMotion for now and apply conditioning post-processing.",
#             UserWarning,
#         )

#         # Simplified version (treat as standard Brownian for now)
#         x = sp.symbols("x", real=True)
#         sigma_sym = sp.symbols("sigma", positive=True)

#         self.state_vars = [x]
#         self.control_vars = []
#         self._f_sym = sp.Matrix([[0]])  # Simplified - should be time-dependent
#         self.parameters = {sigma_sym: sigma}
#         self.order = 1

#         self.diffusion_expr = sp.Matrix([[sigma_sym]])
#         self.sde_type = "ito"


# ============================================================================
# Convenience Factory Functions
# ============================================================================


def create_standard_brownian_motion() -> BrownianMotion:
    """
    Create standard Wiener process W(t).

    The standard Wiener process is the fundamental building block
    of stochastic calculus, characterized by:
    - W(0) = 0 (starts at origin)
    - W(t) ~ N(0, t) (Gaussian with variance t)
    - Independent increments
    - Continuous sample paths

    Returns
    -------
    BrownianMotion
        Standard Brownian motion with σ = 1

    Notes
    -----
    **Standard Brownian Motion Properties:**
    - Unit variance per unit time: Var[W(t)] = t
    - RMS displacement: √t at time t
    - Mean square displacement: <W²(t)> = t
    - This is the reference against which all other
      Brownian motions are scaled

    **Historical Note:**
    This is the mathematical abstraction that Norbert Wiener
    rigorously constructed in 1923, giving Brownian motion
    its alternative name: the Wiener process.
    """
    return BrownianMotion(sigma=1.0)


def create_scaled_brownian_motion(scale: float) -> BrownianMotion:
    """
    Create scaled Brownian motion: σ·W(t).

    Any Brownian motion with diffusion coefficient σ can be
    written as a scaled version of standard Brownian motion:
        X(t) = σ·W(t)
    where W(t) is standard Brownian motion.

    Parameters
    ----------
    scale : float
        Scaling factor σ (must be positive)
        - Controls volatility/noise intensity
        - Variance growth rate: σ²
        - RMS displacement at t=1: σ

    Returns
    -------
    BrownianMotion
        Scaled Brownian motion with specified σ

    Notes
    -----
    **Scaling Properties:**
    - Mean: E[X(t)] = 0 (unchanged by scaling)
    - Variance: Var[X(t)] = σ²·t
    - Distribution: X(t) ~ N(0, σ²·t)
    - Sample paths: Vertical stretching by factor σ

    **Physical Interpretation:**
    - Larger scale: Faster diffusion, more volatile
    - Smaller scale: Slower diffusion, smoother paths
    - Scale relates to temperature in physical systems

    **Common Scales:**
    - σ = 0.1: Low volatility (10% annual in finance)
    - σ = 0.5: Moderate volatility
    - σ = 1.0: Standard Brownian motion
    - σ = 2.0: High volatility

    Raises
    ------
    ValueError
        If scale ≤ 0 (from BrownianMotion constructor)
    """
    return BrownianMotion(sigma=scale)