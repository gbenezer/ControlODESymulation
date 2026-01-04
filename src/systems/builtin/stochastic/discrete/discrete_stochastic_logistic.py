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
Discrete Stochastic Logistic Map - Chaos and Noise in Population Dynamics
==========================================================================

This module provides the discrete-time stochastic logistic map, combining
deterministic chaos with random perturbations to create one of the most
studied systems in nonlinear dynamics. The stochastic logistic map serves as:

- The paradigmatic example of chaos with noise (route to stochasticity)
- A fundamental model for population dynamics under environmental uncertainty
- A benchmark for distinguishing deterministic chaos from random fluctuations
- An illustration of noise-induced transitions between attractors
- A test case for nonlinear time series analysis and prediction

The logistic map demonstrates the full spectrum of dynamical behaviors:
fixed points, periodic orbits, chaos, and how noise modifies each regime.
This makes it essential for understanding the interplay between deterministic
nonlinearity and stochastic perturbations.

Physical Context
----------------

**Ecological Origins:**

The logistic map was introduced by Robert May (1976) to model discrete-time
population growth with density-dependent regulation:

- Generation-to-generation dynamics (annual breeding cycles)
- Carrying capacity limits (resource constraints)
- Overcompensation at high density (competition, predation)
- Environmental stochasticity (weather, resource variability)

**Examples of Discrete-Time Populations:**

1. **Insect Populations:**
   - Annual life cycles (one generation per year)
   - Synchronized breeding seasons
   - Larvae-adult transitions
   - Strong density dependence (limited resources)
   - Environmental variation (temperature, rainfall)

2. **Fish Populations:**
   - Discrete breeding seasons (spawning)
   - Year-class structure
   - Recruitment variability
   - Fishing pressure (harvesting as control)
   - Oceanographic conditions (noise)

3. **Plant Populations:**
   - Seed bank dynamics
   - Annual germination cycles
   - Density-dependent mortality
   - Climate variability (frost, drought)

4. **Economic Systems:**
   - Market cycles with feedback
   - Inventory dynamics
   - Boom-bust patterns
   - Economic shocks (noise)

**Why Discrete-Time Model is Essential:**

Many biological systems have non-overlapping generations:
- Insects with distinct life stages
- Annual plants
- Fish with synchronized spawning
- Seasonal breeders

Continuous models miss discrete-time structure and can give qualitatively
wrong predictions (e.g., missing period-doubling cascades).

Mathematical Background
-----------------------

**Deterministic Logistic Map:**

    x[k+1] = r·x[k]·(1 - x[k])

where:
- x[k] ∈ [0,1]: Population fraction (normalized by carrying capacity)
- r ∈ [0,4]: Growth parameter (intrinsic growth rate)
- (1 - x[k]): Density-dependent regulation

**Stochastic Extension:**

Add multiplicative or additive noise to capture environmental uncertainty:

**Additive Noise (This Implementation):**
    x[k+1] = r·x[k]·(1 - x[k]) + σ·w[k]

**Multiplicative Noise (Alternative):**
    x[k+1] = (r + σ·w[k])·x[k]·(1 - x[k])

**Environmental Stochasticity:**
    x[k+1] = r[k]·x[k]·(1 - x[k])
    
where r[k] = r̄ + σ·w[k] (random growth rate each generation).

Bifurcation Diagram and Parameter Regimes
------------------------------------------

The deterministic logistic map exhibits distinct dynamical regimes as r varies:

**r < 1:** Extinction
- x* = 0 stable
- Population dies out
- Subcritical growth

**1 < r < 3:** Stable Fixed Point
- x* = (r-1)/r stable
- Monotonic convergence
- Regulated population

**r = 3:** First Bifurcation
- Period-doubling begins
- Pitchfork bifurcation
- Onset of oscillations

**3 < r < 1+√6 ≈ 3.45:** Period-2 Cycle
- x oscillates between two values
- Alternating generations (high-low)
- Biennial dynamics

**r ≈ 3.45 - 3.54:** Period-Doubling Cascade
- Period 4, 8, 16, 32, ...
- Feigenbaum universality (δ ≈ 4.669)
- Rapid approach to chaos

**3.57 < r < 4:** Chaotic Regime
- Sensitive dependence on initial conditions
- Positive Lyapunov exponent
- Unpredictable long-term behavior
- Windows of periodic orbits (e.g., period-3 at r ≈ 3.83)

**r = 4:** Fully Developed Chaos
- Analytical solution via tent map
- Invariant density: ρ(x) = 1/(π·√(x·(1-x)))
- Maximum Lyapunov exponent: λ = ln(2)

**r > 4:** Escape
- Iterates leave [0,1] interval
- Population goes negative (unphysical)
- Mathematical instability

Effect of Noise on Dynamics
----------------------------

**Small Noise (σ << |r-r_c|):**

Near fixed points or periodic orbits:
- Fluctuations around attractor
- Gaussian-like distribution
- Correlation time from linearization
- Predictable short-term behavior

**Moderate Noise:**

Near bifurcation points (r ≈ r_c):
- Noise-induced transitions between attractors
- Critical fluctuations enhanced
- Power-law distributions (near criticality)
- Residence time statistics

**Large Noise (σ ~ 1):**

Chaos-like regime:
- Noise dominates deterministic dynamics
- Loss of deterministic structure
- Effective Lyapunov exponent increases
- Harder to distinguish from pure chaos

**Stochastic Bifurcations:**

As σ increases at fixed r:
- P-bifurcation: Stationary distribution changes shape
- D-bifurcation: Lyapunov exponent sign change
- Different from deterministic bifurcation diagram!

**Noise-Induced Chaos:**

Below chaotic threshold (3 < r < 3.57):
- Deterministic: Periodic
- With noise: Appears chaotic
- Positive effective Lyapunov exponent
- Challenge: Distinguish from deterministic chaos

Lyapunov Exponent with Noise
-----------------------------

**Deterministic Lyapunov Exponent:**

    λ_det = lim(N→∞) (1/N)·Σ ln|r·(1 - 2·x[k])|

- λ > 0: Chaos (sensitive dependence)
- λ = 0: Critical (neutral stability)
- λ < 0: Stable (convergence to attractor)

**With Additive Noise:**

Effective Lyapunov exponent combines deterministic divergence and noise:

    λ_eff ≈ λ_det + σ²·C

where C depends on trajectory. Noise generally increases λ_eff.

**With Multiplicative Noise:**

More complex - noise can stabilize or destabilize depending on structure.

**Practical Implication:**

For time series analysis:
- Estimate λ_eff from data
- If λ_eff > 0 but σ large: Could be noise-induced
- Need methods to separate chaos from noise (surrogate data tests)

Invariant Density Under Noise
------------------------------

**Deterministic (r = 4):**

Exact invariant density:
    ρ(x) = 1/(π·√(x·(1-x)))

U-shaped: Spends most time near boundaries.

**With Noise:**

Fokker-Planck equation (continuous-time approximation):
    ∂ρ/∂t = -∂[f(x)·ρ]/∂x + (σ²/2)·∂²ρ/∂x²

where f(x) = r·x·(1-x) - x (deterministic flow).

**Stationary Distribution:**

Solving Fokker-Planck:
- Smooths out delta functions (periodic orbits)
- Fills in gaps (chaotic repellers)
- Alters shape (noise-dependent)
- Must satisfy boundary conditions at 0 and 1

**Numerical Approach:**

Run long simulation, compute histogram:
- Bins: Partition [0,1]
- Count visits to each bin
- Normalize to get probability density

Applications
------------

**1. Population Ecology:**

**Pest Management:**
- Discrete breeding (insect generations)
- Control via r (harvest rate, pesticides)
- Noise from weather, natural enemies
- Goal: Maintain low population despite variability

**Fisheries:**
- Annual recruitment (spawning success)
- Fishing quota as control
- Environmental regime shifts (oceanographic)
- Avoid collapse via adaptive management

**Conservation:**
- Small populations (demographic stochasticity)
- Habitat fragmentation effects
- Climate variability
- Extinction risk assessment

**2. Time Series Analysis:**

**Distinguishing Chaos from Noise:**
- Estimate Lyapunov exponent
- Correlation dimension
- Recurrence plots
- Surrogate data tests

**Forecasting:**
- Short-term: Use nonlinear models
- Long-term: Probabilistic (chaos limit)
- Ensemble methods with noise
- Adaptive to regime shifts

**3. Dynamical Systems Theory:**

**Routes to Chaos:**
- Period-doubling (Feigenbaum)
- Intermittency (near tangent bifurcation)
- Quasi-periodicity (torus breakdown)
- Crisis (attractor collision)

**Universality:**
- Feigenbaum constants apply to broad class
- Scaling near onset of chaos
- Renormalization group theory

**4. Cryptography:**

**Pseudorandom Number Generation:**
- Chaotic maps as RNG
- Sensitivity to initial conditions
- Noise improves randomness
- Fast computation

**Secure Communication:**
- Chaos synchronization
- Message masking
- Robustness to noise

**5. Economics:**

**Market Dynamics:**
- Boom-bust cycles (period-doubling)
- Speculative bubbles (chaos)
- Policy interventions (control)
- Economic shocks (noise)

**6. Engineering:**

**Digital Systems:**
- Nonlinear oscillators
- Phase-locked loops
- Switching circuits
- Noise margins

Prediction and Control
----------------------

**Short-Term Prediction:**

For low noise and non-chaotic r:
- One-step ahead: x̂[k+1] = r·x[k]·(1 - x[k])
- Works well if λ_eff < 0
- Forecast horizon: ~1/|λ_eff| generations

**Chaos Limit:**

For λ_eff > 0:
- Prediction horizon: ~1/λ_eff generations
- Exponential error growth: ε(t) ~ ε₀·exp(λ_eff·t)
- Long-term: Use ensemble forecasts

**Control Strategies:**

**Parameter Control:**
Adjust r to maintain stability:
- Target: Keep r < 3 (fixed point regime)
- Harvest control: r_eff = r - h (h = harvest rate)
- Avoid chaotic regime (3.57 < r < 4)

**State Feedback:**
OGY (Ott-Grebogi-Yorke) method:
- Stabilize unstable periodic orbits in chaos
- Small perturbations to parameters
- Requires precise state knowledge

**Threshold Control:**
If x > x_threshold: Reduce r or add mortality
- Simple, robust to noise
- Prevents extreme fluctuations

**Challenges with Noise:**
- Observation uncertainty (can't measure x exactly)
- Control noise (can't set r perfectly)
- State estimation required (filtering)

State Estimation Challenge
---------------------------

**Problem:**

Cannot observe x[k] perfectly due to:
- Sampling error (finite population)
- Measurement noise (survey accuracy)
- Partial observability (not all individuals counted)

**Observation Model:**

    y[k] = x[k] + v[k]

where v[k] ~ N(0, R) is measurement noise.

**Filtering Challenge:**

Use Extended Kalman Filter (EKF):

**Prediction:**
    x̂[k+1|k] = r·x̂[k|k]·(1 - x̂[k|k])
    F[k] = r·(1 - 2·x̂[k|k])  (Jacobian)
    P[k+1|k] = F[k]·P[k|k]·F[k]ᵀ + Q

**Update:**
    K[k] = P[k|k-1]·(P[k|k-1] + R)⁻¹
    x̂[k|k] = x̂[k|k-1] + K[k]·(y[k] - x̂[k|k-1])
    P[k|k] = (1 - K[k])·P[k|k-1]

**Issues:**
- EKF assumes local linearity (fails in chaos)
- Particle filter better for strong nonlinearity
- Need good process noise Q estimate

Common Pitfalls
---------------

1. **Boundary Violations:**
   - Logistic map defined on [0,1]
   - Noise can push x < 0 or x > 1 (unphysical)
   - Solution: Truncate or reflect at boundaries
   - Or use multiplicative noise (keeps in domain)

2. **Wrong Noise Model:**
   - Additive vs multiplicative matters
   - Environmental: Affects r (multiplicative)
   - Demographic: Affects population directly (additive)
   - Choice affects long-term statistics

3. **Ignoring Discreteness:**
   - For small N, binomial sampling matters
   - x[k] = n[k]/K (where n[k] ∈ {0,1,...,K})
   - Continuous approximation breaks down for N < 100
   - Need individual-based models

4. **Linearization in Chaos:**
   - EKF fails in chaotic regime (diverges)
   - Particle filter or ensemble Kalman filter needed
   - Or restrict to stable regimes only

5. **Parameter Misidentification:**
   - Estimating r from noisy data difficult
   - Noise can mask deterministic structure
   - Need long time series (N > 1000)
   - Use maximum likelihood or Bayesian methods

6. **Confusing Chaos with Noise:**
   - Both create irregular fluctuations
   - Need surrogate data tests
   - Phase space reconstruction
   - Lyapunov exponent estimation

Testing and Validation
-----------------------

**Deterministic Structure:**

1. **Return Map:**
   Plot x[k+1] vs x[k]
   - Should trace out parabola
   - Noise adds scatter around curve

2. **Bifurcation Diagram:**
   Plot long-term x values vs r
   - Period-doubling cascade visible
   - Chaotic bands
   - Periodic windows

3. **Lyapunov Exponent:**
   Compute from time series:
   - λ < 0: Stable
   - λ ≈ 0: Near bifurcation
   - λ > 0: Chaos

**Stochastic Effects:**

1. **Stationary Distribution:**
   Compare histogram with theory
   - Fixed point: Delta function → Gaussian
   - Period-2: Two deltas → Two Gaussians
   - Chaos: Smooth out invariant density

2. **Noise-Induced Transitions:**
   Monitor transitions between coexisting attractors
   - Measure residence times
   - Estimate transition rates
   - Compare with Kramers' theory

3. **Surrogate Data:**
   Test null hypothesis: "Data is linear stochastic process"
   - Generate surrogate data (same spectrum, different phase)
   - Compute nonlinear statistic (e.g., correlation dimension)
   - Compare with original: If different → nonlinear determinism
"""

import numpy as np
import sympy as sp
from typing import Optional, Tuple, List

from src.systems.base.core.discrete_stochastic_system import DiscreteStochasticSystem


class DiscreteStochasticLogisticMap(DiscreteStochasticSystem):
    """
    Discrete-time stochastic logistic map - chaos meets noise.

    The paradigmatic example of deterministic chaos with environmental
    stochasticity, demonstrating the full range of nonlinear phenomena:
    fixed points, periodic orbits, period-doubling cascades, chaos, and
    noise-induced transitions.

    Stochastic Difference Equation
    -------------------------------
    Additive noise formulation:
        x[k+1] = r·x[k]·(1 - x[k]) + σ·w[k]

    where:
        - x[k] ∈ [0,1]: Population fraction (normalized)
        - r ∈ [0,4]: Growth parameter (bifurcation parameter)
        - σ ≥ 0: Noise intensity (environmental stochasticity)
        - w[k] ~ N(0,1): Standard Gaussian white noise

    **Deterministic Part:**
        f(x) = r·x·(1 - x)

    Classic parabola with single maximum at x = 0.5.

    **Stochastic Part:**
        g(x) = σ (constant, additive)

    Environmental noise independent of population size.

    Physical Interpretation
    -----------------------
    **Ecological Meaning:**

    x[k]: Fraction of carrying capacity
    - x = 0: Extinction
    - x = 1: At carrying capacity K
    - x = 0.5: Half capacity (maximum growth rate)

    r: Intrinsic growth rate per generation
    - r < 1: Extinction (death rate > birth rate)
    - 1 < r < 3: Regulated (stable equilibrium)
    - 3 < r < 3.57: Oscillations (periodic)
    - 3.57 < r < 4: Chaos (unpredictable)

    σ: Environmental variability
    - Weather fluctuations
    - Resource availability changes
    - Predator abundance variation
    - Typical: σ ~ 0.01-0.1 for small noise

    **Nonlinear Regulation:**

    Term (1 - x[k]) creates negative feedback:
    - x small: Growth approximately r·x (exponential)
    - x near 1: Growth slows (carrying capacity)
    - x > 1: Negative growth (overcompensation)

    Dynamical Regimes
    ------------------
    **Fixed Point Regime (1 < r < 3):**
    
    Deterministic:
    - x* = (r-1)/r stable
    - Monotonic convergence
    
    With noise:
    - Fluctuations around x*
    - Approximately Gaussian
    - Standard deviation ~ σ/√(1-λ²) where λ = |2-r|

    **Period-Doubling Regime (3 < r < 3.57):**
    
    Deterministic:
    - Period-2, 4, 8, ... orbits
    - Feigenbaum cascade
    
    With noise:
    - Cycles blur into bands
    - Noise-induced transitions between branches
    - Bimodal distributions

    **Chaotic Regime (3.57 < r < 4):**
    
    Deterministic:
    - Sensitive dependence on initial conditions
    - Positive Lyapunov exponent
    - Bounded but unpredictable
    
    With noise:
    - Enhanced unpredictability
    - Noise can fill in gaps (chaotic repellers)
    - Smooth invariant density

    **Critical Point (r = 4):**
    
    Deterministic:
    - Fully developed chaos
    - Analytical solution exists
    - Maximum entropy
    
    With noise:
    - Prevents boundary accumulation
    - Regularizes singularities

    Key Properties
    --------------
    **Nonlinearity:**
    Quadratic nonlinearity creates:
    - Multiple equilibria (up to 2)
    - Bifurcations (parameter-dependent)
    - Chaos (sensitivity to initial conditions)

    **Boundedness:**
    For r ≤ 4 and σ small:
    - Deterministic: x ∈ [0,1] preserved
    - With noise: Can exceed bounds (unphysical)
    - Solution: Truncate or use multiplicative noise

    **Sensitive Dependence:**
    In chaotic regime (λ > 0):
    - Prediction horizon: ~1/λ generations
    - Exponential error growth
    - Long-term: Ensemble forecasts only

    **Noise Effects:**
    - Smooth attractors
    - Induce transitions
    - Enhance effective Lyapunov exponent
    - Can create or destroy chaos

    Mathematical Properties
    -----------------------
    **Fixed Points (Deterministic):**

    x₁* = 0 (extinction)
    - Stable for r < 1
    - Unstable for r > 1

    x₂* = (r-1)/r (regulated)
    - Exists for r > 1
    - Stable for 1 < r < 3
    - Unstable for r > 3 (bifurcation)

    **Stability (Linearization):**

    At x*: f'(x*) = r·(1 - 2·x*)
    - |f'| < 1: Stable
    - |f'| > 1: Unstable
    - f' = -1: Bifurcation

    At x₂*: f'(x₂*) = 2 - r
    - Stable: 1 < r < 3
    - Period-doubling: r = 3

    **Lyapunov Exponent:**

    λ = lim(N→∞) (1/N)·Σ ln|r·(1 - 2·x[k])|

    - λ < 0: Stable (r < 3)
    - λ = 0: Critical (r = 3)
    - λ > 0: Chaos (r > 3.57)
    - λ_max = ln(2) at r = 4

    **Invariant Density (r = 4):**

    ρ(x) = 1/(π·√(x·(1-x)))

    U-shaped: Maximum at boundaries, minimum at x = 0.5.

    State Space
    -----------
    State: x ∈ [0,1] (bounded interval)
        - Physical constraint: population fraction
        - Noise can violate bounds (handle carefully)

    Control: None in basic model
        - Can add control: x[k+1] = r·x[k]·(1-x[k]) + u[k]
        - Harvest control: u < 0 (remove individuals)

    Noise: w[k] ~ N(0,1)
        - Additive (constant intensity)
        - Environmental stochasticity

    Parameters
    ----------
    r : float, default=3.8
        Growth parameter (bifurcation parameter)
        - r < 1: Extinction
        - 1 < r < 3: Stable fixed point
        - 3 < r < 3.57: Periodic orbits
        - 3.57 < r < 4: Chaos
        - r = 3.8: Generic chaotic (default)
        - r > 4: Escape from [0,1]

    sigma : float, default=0.05
        Noise intensity (environmental stochasticity)
        - σ = 0: Deterministic
        - σ ~ 0.01-0.05: Small noise (perturbative)
        - σ ~ 0.1-0.2: Moderate noise
        - σ > 0.2: Large noise (dominates)
        - Typical: σ ~ 0.05 for biological systems

    x0 : float, default=0.5
        Initial condition
        - Typically x0 = 0.5 (maximum growth rate)
        - Chaos: Any x0 ∈ (0,1) gives same long-term statistics

    dt : float, default=1.0
        Generation time (time units per iteration)
        - Typically dt = 1 generation

    enforce_bounds : bool, default=True
        Whether to enforce x ∈ [0,1]
        - True: Truncate to [0,1] (prevents unphysical values)
        - False: Allow escape (for mathematical studies)

    Stochastic Properties
    ---------------------
    - System Type: NONLINEAR (quadratic)
    - Noise Type: ADDITIVE (constant intensity)
    - Chaotic: Depends on r (λ(r))
    - Stationary: Yes (for r < 4, bounded noise)
    - Ergodic: Yes (in chaotic regime)
    - Markov: Yes (one-step memory)

    Applications
    ------------
    **1. Population Ecology:**
    - Insect outbreak dynamics
    - Fish stock assessment
    - Pest management
    - Conservation biology

    **2. Time Series Analysis:**
    - Chaos detection
    - Nonlinear prediction
    - Lyapunov exponent estimation
    - Surrogate data testing

    **3. Dynamical Systems:**
    - Bifurcation theory
    - Routes to chaos
    - Universality classes
    - Renormalization group

    **4. Cryptography:**
    - Pseudorandom number generation
    - Secure communication
    - Chaos-based encryption

    **5. Control Theory:**
    - OGY method (chaos control)
    - Stabilizing unstable orbits
    - Adaptive control under uncertainty

    Numerical Simulation
    --------------------
    **Direct Iteration:**
        x[k+1] = r·x[k]·(1 - x[k]) + σ·randn()

    Exact (no discretization error).

    **Boundary Handling:**
    If enforce_bounds:
        x[k+1] = clip(x[k+1], 0, 1)

    **Long-Term Statistics:**
    - Run N >> 1000 iterations
    - Discard initial transient (burn-in)
    - Compute histogram (invariant density)
    - Calculate Lyapunov exponent

    Comparison with Other Models
    -----------------------------
    **vs. Ricker Map:**
    - Ricker: x[k+1] = x[k]·exp(r·(1 - x[k]))
    - Logistic: Polynomial, simpler
    - Both show period-doubling

    **vs. Tent Map:**
    - Tent: Piecewise linear, chaotic
    - Logistic: Smooth, more realistic
    - Topologically conjugate at r = 4

    **vs. Continuous Logistic:**
    - Continuous: dx/dt = r·x·(1 - x)
    - Discrete: Captures generation structure
    - Discrete: Richer dynamics (chaos possible)

    Limitations
    -----------
    - Bounded state space (x ∈ [0,1])
    - Single variable (scalar)
    - Additive noise (multiplicative more realistic)
    - Noise can cause boundary violations
    - No age/stage structure

    Extensions
    ----------
    - Multiplicative noise: σ·x·(1-x)·w[k]
    - Two-dimensional: Coupled logistic maps
    - Spatially extended: Lattice of coupled maps
    - Discrete-time: Ricker, Beverton-Holt
    - Continuous-time: Logistic ODE

    See Also
    --------
    DiscreteAR1 : Linear analog (no chaos)
    DiscreteRandomWalk : No regulation
    DiscreteStochasticPendulum : Continuous state space nonlinearity
    """

    def define_system(
        self,
        r: float = 3.8,
        sigma: float = 0.05,
        x0: float = 0.5,
        dt: float = 1.0,
        enforce_bounds: bool = True,
    ):
        """
        Define stochastic logistic map dynamics.

        Parameters
        ----------
        r : float, default=3.8
            Growth parameter (must be in [0,4] for boundedness)
            - r < 1: Extinction regime
            - 1 < r < 3: Fixed point regime
            - r = 3: First bifurcation
            - 3 < r < 3.57: Periodic regime
            - 3.57 < r < 4: Chaotic regime
            - r = 3.8: Generic chaos (default)

        sigma : float, default=0.05
            Noise intensity
            - σ = 0: Deterministic
            - σ ~ 0.01-0.05: Small perturbations
            - σ ~ 0.1: Moderate noise
            - σ > 0.2: Large noise

        x0 : float, default=0.5
            Initial condition (must be in [0,1])

        dt : float, default=1.0
            Generation time

        enforce_bounds : bool, default=True
            Whether to clip x to [0,1] each step

        Raises
        ------
        ValueError
            If r < 0, r > 4, or x0 not in [0,1]

        UserWarning
            If r > 4 (escape from domain)
            If r < 1 (extinction regime)

        Notes
        -----
        **Parameter Selection:**

        For exploring dynamics:
        - Fixed point: r = 2.5, σ = 0.01
        - Period-2: r = 3.2, σ = 0.02
        - Chaos onset: r = 3.57, σ = 0.03
        - Full chaos: r = 3.8-4.0, σ = 0.05

        **Noise Scaling:**

        Unlike continuous models, σ doesn't scale with √dt
        because discrete-time noise is per-generation.

        **Boundary Enforcement:**

        If enforce_bounds=True:
        - Prevents x < 0 (extinction)
        - Prevents x > 1 (exceeding capacity)
        - Reflects or truncates at boundaries

        If False:
        - Can escape [0,1] (unphysical)
        - Useful for studying escape rates
        - Most simulations should use True

        **Lyapunov Exponent:**

        Estimate for deterministic part:
        - r = 2.5: λ ≈ -0.69 (stable)
        - r = 3.0: λ ≈ 0 (critical)
        - r = 3.5: λ ≈ 0.2 (periodic)
        - r = 3.8: λ ≈ 0.5 (chaotic)
        - r = 4.0: λ = ln(2) ≈ 0.693 (maximum)

        With noise, effective λ increases.

        **Period-Doubling Points:**

        Bifurcation cascade:
        - r₁ = 3.0 (period 2)
        - r₂ ≈ 3.449 (period 4)
        - r₃ ≈ 3.544 (period 8)
        - r∞ ≈ 3.5699 (chaos onset)

        Feigenbaum ratio: δ = (rₙ₊₁ - rₙ)/(rₙ₊₂ - rₙ₊₁) → 4.669

        **Chaos Windows:**

        Periodic windows within chaos:
        - r ≈ 3.83: Period-3 window
        - r ≈ 3.74: Period-5 window
        - Many others (dense set)
        """
        if r < 0:
            raise ValueError(f"r must be non-negative, got {r}")
        if r > 4:
            import warnings
            warnings.warn(
                f"r = {r} > 4: Iterates will escape [0,1]. "
                f"Bounded dynamics require r ≤ 4.",
                UserWarning
            )
        if r < 1:
            import warnings
            warnings.warn(
                f"r = {r} < 1: Extinction regime. "
                f"Population will converge to x = 0.",
                UserWarning
            )
        if x0 < 0 or x0 > 1:
            raise ValueError(f"Initial condition x0 must be in [0,1], got {x0}")
        if sigma < 0:
            raise ValueError(f"sigma must be non-negative, got {sigma}")

        # Store parameters
        self.r_param = r
        self.sigma_param = sigma
        self.x0_param = x0
        self.enforce_bounds = enforce_bounds

        # State variable
        x = sp.symbols("x", real=True)

        # Parameters
        r_sym = sp.symbols("r", real=True, positive=True)
        sigma_sym = sp.symbols("sigma", real=True, nonnegative=True)

        self.state_vars = [x]
        self.control_vars = []  # No control in basic model

        # DETERMINISTIC PART: f(x) = r·x·(1-x)
        f_logistic = r_sym * x * (1 - x)
        self._f_sym = sp.Matrix([f_logistic])

        # STOCHASTIC PART: Additive noise
        self.diffusion_expr = sp.Matrix([[sigma_sym]])

        self.parameters = {
            r_sym: r,
            sigma_sym: sigma,
        }
        self.order = 1
        self._dt = dt
        self.sde_type = "ito"

        # Output: State itself
        self._h_sym = sp.Matrix([x])

    def setup_equilibria(self):
        """
        Set up equilibrium points (deterministic part).

        Logistic map has up to two fixed points:
        1. x* = 0 (extinction)
        2. x* = (r-1)/r (regulated population)
        """
        r = self.r_param

        # Extinction equilibrium
        self.add_equilibrium(
            "extinction",
            x_eq=np.array([0.0]),
            u_eq=np.array([]),
            verify=True,
            stability="stable" if r < 1 else "unstable",
            notes=f"Extinction equilibrium. Stable for r < 1, unstable for r > 1."
        )

        # Regulated equilibrium (exists for r > 1)
        if r > 1:
            x_star = (r - 1) / r
            
            # Check stability: |f'(x*)| = |2 - r|
            stability = "stable" if abs(2 - r) < 1 else "unstable"
            if abs(abs(2-r) - 1) < 0.01:
                stability = "critical"

            notes = f"Regulated equilibrium at x* = {x_star:.3f}. "
            if r < 3:
                notes += "Stable fixed point regime."
            elif r == 3:
                notes += "Bifurcation point (period-doubling)."
            else:
                notes += "Unstable (periodic or chaotic regime)."

            self.add_equilibrium(
                "regulated",
                x_eq=np.array([x_star]),
                u_eq=np.array([]),
                verify=True,
                stability=stability,
                notes=notes
            )
            self.set_default_equilibrium("regulated")
        else:
            self.set_default_equilibrium("extinction")

    def compute_lyapunov_exponent(
        self,
        x0: Optional[float] = None,
        n_iterations: int = 10000,
        transient: int = 1000,
    ) -> float:
        """
        Estimate Lyapunov exponent from trajectory.

        λ = lim(N→∞) (1/N)·Σ ln|f'(x[k])|

        where f'(x) = r·(1 - 2·x).

        Parameters
        ----------
        x0 : Optional[float]
            Initial condition (uses self.x0_param if None)
        n_iterations : int
            Number of iterations for estimation
        transient : int
            Number of initial iterations to discard

        Returns
        -------
        float
            Estimated Lyapunov exponent

        Notes
        -----
        - λ < 0: Stable (convergence)
        - λ ≈ 0: Critical (near bifurcation)
        - λ > 0: Chaos (sensitive dependence)

        This computes the deterministic Lyapunov exponent
        (noise-free). With noise, use time series methods.

        Examples
        --------
        >>> logistic = DiscreteStochasticLogisticMap(r=3.8, sigma=0.0)
        >>> lambda_est = logistic.compute_lyapunov_exponent()
        >>> print(f"λ ≈ {lambda_est:.3f}")  # Should be ~0.5
        """
        if x0 is None:
            x0 = self.x0_param

        r = self.r_param
        x = x0

        lyap_sum = 0.0

        for i in range(transient + n_iterations):
            # Derivative: f'(x) = r·(1 - 2·x)
            derivative = r * (1 - 2*x)

            # Accumulate log of absolute derivative
            if i >= transient and abs(derivative) > 1e-10:
                lyap_sum += np.log(abs(derivative))

            # Update (deterministic)
            x = r * x * (1 - x)

            # Enforce bounds if requested
            if self.enforce_bounds:
                x = np.clip(x, 0, 1)

        return lyap_sum / n_iterations

    def compute_bifurcation_diagram(
        self,
        r_range: Tuple[float, float] = (2.5, 4.0),
        n_r_values: int = 1000,
        transient: int = 500,
        n_plot_points: int = 100,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute bifurcation diagram (deterministic).

        For each r value, iterate map and record long-term behavior.

        Parameters
        ----------
        r_range : Tuple[float, float]
            Range of r values to explore
        n_r_values : int
            Number of r values to sample
        transient : int
            Iterations to discard (burn-in)
        n_plot_points : int
            Number of points to keep per r value

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (r_values, x_values) for plotting

        Examples
        --------
        >>> logistic = DiscreteStochasticLogisticMap(r=3.0, sigma=0.0)
        >>> r_vals, x_vals = logistic.compute_bifurcation_diagram()
        >>> plt.plot(r_vals, x_vals, ',k', alpha=0.5, markersize=0.5)
        >>> plt.xlabel('r')
        >>> plt.ylabel('x')
        >>> plt.title('Bifurcation Diagram')
        """
        r_values = np.linspace(r_range[0], r_range[1], n_r_values)
        x_all = []
        r_all = []

        x0 = self.x0_param

        for r in r_values:
            x = x0

            # Transient
            for _ in range(transient):
                x = r * x * (1 - x)
                if self.enforce_bounds:
                    x = np.clip(x, 0, 1)

            # Collect points
            x_points = []
            for _ in range(n_plot_points):
                x = r * x * (1 - x)
                if self.enforce_bounds:
                    x = np.clip(x, 0, 1)
                x_points.append(x)

            # Store
            x_all.extend(x_points)
            r_all.extend([r] * n_plot_points)

        return np.array(r_all), np.array(x_all)

    def estimate_invariant_density(
        self,
        n_iterations: int = 100000,
        transient: int = 1000,
        n_bins: int = 100,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate invariant density from long trajectory.

        Parameters
        ----------
        n_iterations : int
            Number of iterations to collect
        transient : int
            Burn-in period
        n_bins : int
            Number of histogram bins

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (bin_centers, density)

        Examples
        --------
        >>> logistic = DiscreteStochasticLogisticMap(r=4.0, sigma=0.0)
        >>> x_bins, density = logistic.estimate_invariant_density()
        >>> plt.plot(x_bins, density)
        >>> plt.xlabel('x')
        >>> plt.ylabel('ρ(x)')
        >>> # Compare with exact: ρ(x) = 1/(π·√(x·(1-x)))
        """
        x0 = self.x0_param
        r = self.r_param
        sigma = self.sigma_param

        x_trajectory = []
        x = x0

        # Run simulation
        for i in range(transient + n_iterations):
            # Update
            x_new = r * x * (1 - x) + sigma * np.random.randn()

            if self.enforce_bounds:
                x_new = np.clip(x_new, 0, 1)

            x = x_new

            # Collect after transient
            if i >= transient:
                x_trajectory.append(x)

        # Compute histogram
        density, bin_edges = np.histogram(
            x_trajectory,
            bins=n_bins,
            range=(0, 1),
            density=True
        )
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        return bin_centers, density

    def theoretical_invariant_density(
        self,
        x_values: np.ndarray,
    ) -> np.ndarray:
        """
        Compute theoretical invariant density for r=4 (deterministic).

        ρ(x) = 1/(π·√(x·(1-x)))

        Parameters
        ----------
        x_values : np.ndarray
            Points at which to evaluate density

        Returns
        -------
        np.ndarray
            Density values

        Notes
        -----
        Only valid for r = 4 and σ = 0.
        U-shaped distribution.

        Examples
        --------
        >>> x = np.linspace(0.01, 0.99, 100)
        >>> logistic = DiscreteStochasticLogisticMap(r=4.0, sigma=0.0)
        >>> rho_theory = logistic.theoretical_invariant_density(x)
        """
        if self.r_param != 4.0:
            import warnings
            warnings.warn(
                f"Theoretical density only exact for r=4, got r={self.r_param}",
                UserWarning
            )

        # Avoid division by zero at boundaries
        x_safe = np.clip(x_values, 1e-10, 1 - 1e-10)
        return 1.0 / (np.pi * np.sqrt(x_safe * (1 - x_safe)))


# Convenience functions
def create_fixed_point_regime(
    noise_level: str = 'small'
) -> DiscreteStochasticLogisticMap:
    """
    Create logistic map in stable fixed point regime.

    Parameters
    ----------
    noise_level : str
        'small', 'medium', or 'large'

    Returns
    -------
    DiscreteStochasticLogisticMap

    Examples
    --------
    >>> # Stable regime with small noise
    >>> stable = create_fixed_point_regime('small')
    """
    noise_map = {'small': 0.01, 'medium': 0.05, 'large': 0.1}
    sigma = noise_map.get(noise_level, 0.01)

    return DiscreteStochasticLogisticMap(
        r=2.8,  # Stable fixed point
        sigma=sigma,
        x0=0.5,
        dt=1.0
    )


def create_chaotic_regime(
    noise_level: str = 'small'
) -> DiscreteStochasticLogisticMap:
    """
    Create logistic map in chaotic regime.

    Parameters
    ----------
    noise_level : str
        'small', 'medium', or 'large'

    Returns
    -------
    DiscreteStochasticLogisticMap

    Examples
    --------
    >>> # Full chaos with small noise
    >>> chaos = create_chaotic_regime('small')
    """
    noise_map = {'small': 0.02, 'medium': 0.05, 'large': 0.1}
    sigma = noise_map.get(noise_level, 0.02)

    return DiscreteStochasticLogisticMap(
        r=3.8,  # Generic chaotic
        sigma=sigma,
        x0=0.5,
        dt=1.0
    )


def create_edge_of_chaos(
    noise_level: str = 'small'
) -> DiscreteStochasticLogisticMap:
    """
    Create logistic map at onset of chaos.

    Parameters
    ----------
    noise_level : str
        'small', 'medium', or 'large'

    Returns
    -------
    DiscreteStochasticLogisticMap

    Examples
    --------
    >>> # Critical point with minimal noise
    >>> critical = create_edge_of_chaos('small')
    """
    noise_map = {'small': 0.005, 'medium': 0.02, 'large': 0.05}
    sigma = noise_map.get(noise_level, 0.005)

    return DiscreteStochasticLogisticMap(
        r=3.5699,  # Feigenbaum point (chaos onset)
        sigma=sigma,
        x0=0.5,
        dt=1.0
    )