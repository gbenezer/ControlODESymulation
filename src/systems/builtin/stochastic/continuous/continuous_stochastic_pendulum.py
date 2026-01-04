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
Stochastic Pendulum - Nonlinear Oscillator with Random Forcing
===============================================================

This module provides the stochastic pendulum, a fundamental nonlinear system
exhibiting rich stochastic phenomena. The stochastic pendulum serves as:

- The canonical example of noise-induced transitions over an energy barrier
- A benchmark for understanding stochastic resonance and coherence resonance
- An illustration of Kramers' escape rate theory (chemical kinetics foundation)
- A model for noise-induced synchronization and phase locking
- The simplest system demonstrating interplay between nonlinearity and randomness

The stochastic pendulum extends the classic deterministic pendulum by adding
random forcing, creating phenomena impossible in deterministic systems:
- Noise can help the pendulum overcome the barrier (activation)
- Optimal noise level maximizes response to periodic forcing (stochastic resonance)
- Noise creates coherent oscillations even without periodic forcing (coherence resonance)

Physical Context
----------------

**Classical Pendulum with Random Forcing:**

A simple pendulum subject to:
- Gravity: Creates restoring torque -sin(θ)
- Damping: Friction proportional to velocity -b·ω
- Control: External applied torque u
- Random forcing: Environmental disturbances η(t)

Equation of motion:
    I·θ̈ = -m·g·L·sin(θ) - b·θ̇ + u + η(t)

For unit parameters (normalized): θ̈ = -sin(θ) - b·θ̇ + u + w(t)

**Physical Systems:**

1. **Mechanical Pendulum:**
   - Simple pendulum with random air currents
   - Torsion pendulum with thermal fluctuations
   - Inverted pendulum with ground vibrations

2. **Josephson Junction:**
   - Superconducting quantum device
   - Phase difference θ behaves like pendulum
   - Thermal noise from resistive element
   - Critical for quantum computing, SQUIDs

3. **Phase-Locked Loops (PLL):**
   - Electronic oscillator synchronization
   - Phase error θ
   - Noise from components
   - Used in communications, frequency synthesis

4. **Coupled Oscillators:**
   - Biological rhythms (circadian clocks)
   - Neural oscillations (brain rhythms)
   - Power grid synchronization
   - Noise from environment or intrinsic sources

5. **Brownian Motor:**
   - Ratchet potential (asymmetric)
   - Thermal fluctuations
   - Directed motion from noise + asymmetry

**Why Pendulum is Special:**

Unlike linear systems, pendulum has:
- **Bistability:** Two stable equilibria (θ = 0, ±2π, ±4π, ...)
- **Energy barrier:** Separating upright and inverted
- **Nonlinearity:** sin(θ) creates rich dynamics
- **Multiple scales:** Fast oscillations, slow rotations

Mathematical Formulation
------------------------

**Stochastic Differential Equations:**

State-space form (θ: angle, ω: angular velocity):
    dθ = ω·dt
    dω = (-(g/L)·sin(θ) - b·ω + u)·dt + σ·dW

where:
    - θ ∈ S¹ or ℝ: Angle [rad] (periodic or unwrapped)
    - ω ∈ ℝ: Angular velocity [rad/s]
    - g/L: Gravitational constant (normalized: often 1)
    - b ≥ 0: Damping coefficient [1/s]
    - u: Applied torque [rad/s²]
    - σ: Noise intensity [rad/(s²·√s)]
    - W(t): Standard Wiener process

**Normalized Form (g/L = 1):**
    dθ = ω·dt
    dω = (-sin(θ) - b·ω + u)·dt + σ·dW

**Energy:**
    E = (1/2)·ω² - cos(θ)

Kinetic + potential. Noise continuously injects/removes energy.

**Regimes:**

1. **Underdamped (b small):** Oscillations
2. **Critically damped (b = 2):** Fast settling
3. **Overdamped (b large):** Slow creeping

Stochastic Phenomena
--------------------

**1. Noise-Induced Transitions (Kramers Problem):**

**Setup:** Pendulum at stable equilibrium θ = 0 (hanging down).
Energy barrier ΔE = 2 separates from inverted position θ = π.

**Question:** How long until noise causes pendulum to flip over barrier?

**Kramers' Escape Rate:**
    k_escape ≈ (ω_well·ω_barrier)/(2π·b)·exp(-ΔE/(σ²))

where:
- ω_well = √(g/L): Frequency at bottom (stable)
- ω_barrier = √(-g/L) = i·ω_well: "Frequency" at top (imaginary!)
- ΔE = 2: Energy barrier height

**Mean First Passage Time:**
    τ_escape = 1/k_escape ≈ (2π·b/ω²)·exp(ΔE/σ²)

**Exponential Dependence:**
- Small noise (σ = 0.1): τ ~ exp(200) ~ 10^87 seconds (never!)
- Medium noise (σ = 0.5): τ ~ exp(8) ~ 3000 seconds
- Large noise (σ = 1.0): τ ~ exp(2) ~ 7 seconds

Tiny change in σ causes enormous change in escape time!

**2. Stochastic Resonance:**

**Setup:** Pendulum driven by weak periodic force + noise.
    u(t) = A·cos(ω_drive·t)

**Phenomenon:** At optimal noise level σ_opt, response is MAXIMUM.

**Mechanism:**
- No noise: Signal too weak to overcome barrier
- Optimal noise: Noise helps escape, synchronized with signal
- Too much noise: Random escapes, no synchronization

**Signal-to-Noise Ratio (SNR):**
    SNR(σ) is non-monotonic - peaks at σ_opt!

**Applications:**
- Sensory biology: Neurons detect weak signals via noise
- Climate: Ice age transitions enhanced by noise
- Electronics: Signal detection in noisy channels

**3. Coherence Resonance:**

**Setup:** No periodic forcing (u = 0), only noise.

**Phenomenon:** At optimal σ, pendulum oscillates most coherently.

**Mechanism:**
- Noise causes random barrier crossings
- Crossings create quasi-periodic oscillations
- Regularity measured by autocorrelation time
- Maximum coherence at intermediate σ

**Measure:** Coherence factor = mean period / std(period)

Peaks at σ_opt, not at σ = 0 or σ → ∞!

**4. Noise-Induced Phase Locking:**

Multiple stochastic pendulums can synchronize via noise.
Common noise creates correlations → synchronization.

**5. Stochastic Bifurcations:**

As noise σ increases:
- P-bifurcation: Stationary distribution changes shape
- D-bifurcation: Stability changes (Lyapunov exponent)

Equilibria and Bistability
---------------------------

**Deterministic Equilibria:**

1. **Downward (Stable):** θ = 0, 2π, 4π, ... (mod 2π)
   - Minimum energy
   - Stable in potential well
   - Small oscillations: ω_0 = √(g/L)

2. **Upward (Unstable):** θ = π, 3π, 5π, ... (mod 2π)
   - Maximum energy  
   - Unstable saddle
   - Energy barrier: ΔE = 2 (normalized)

**With Noise:**

**Stochastic Equilibria:**
Not fixed points but probability distributions.

**Stationary Distribution:**

For damped pendulum (b > 0) with noise, stationary density:
    p_∞(θ, ω) ∝ exp(-E(θ,ω)/(σ²/b))

where E(θ,ω) = (1/2)·ω² - cos(θ) is energy.

**Features:**
- Peaks at stable equilibria (θ = 0, ±2π, ...)
- Valleys at unstable equilibria (θ = ±π, ±3π, ...)
- Width depends on σ/b ratio

**Effective Temperature:**
    T_eff = σ²/b

Analogy with statistical mechanics:
- Low T_eff: Concentrated near minima
- High T_eff: Spread over all θ

**Metastable States:**

At low noise, system stays near stable equilibria for long time,
then occasionally transitions (activation over barrier).

Applications
------------

**1. Physics & Chemistry:**

**Kramers' Escape Rate:**
Foundation of chemical reaction rate theory:
- Reactant well → transition state → product well
- Activation energy ≡ barrier height
- Temperature ≡ noise intensity
- Arrhenius rate ≡ Kramers' rate

**Brownian Motion in Periodic Potential:**
Particle in sinusoidal potential with thermal noise.

**2. Electronics:**

**Josephson Junction:**
- Phase difference in superconductor
- Noise from thermal fluctuations
- Critical current, switching dynamics

**Phase-Locked Loop:**
- Phase tracking with noise
- Cycle slipping (barrier crossing)
- Frequency synthesis

**3. Biology:**

**Neuron Models:**
- Phase oscillator with noise
- Spike timing variability
- Stochastic resonance in neurons

**Circadian Rhythms:**
- Biological clock with environmental noise
- Entrainment to light-dark cycles
- Phase coherence

**4. Geophysics:**

**Climate Dynamics:**
- Ice age transitions
- Noise-enhanced periodic forcing
- Stochastic resonance in paleoclimate

**5. Control Systems:**

**Nonlinear Control Benchmark:**
- Swing-up control with noise
- Stabilization at inverted position
- Robustness to disturbances

**6. Synchronization:**

**Coupled Oscillators:**
- Power grid stability
- Neuronal synchronization
- Laser arrays

Numerical Integration
---------------------

**Challenges:**

1. **Angle Wrapping:**
   - θ can exceed ±π (multiple rotations)
   - Unwrapped: Track total angle
   - Wrapped: Use mod 2π (periodic boundary)

2. **Nonlinearity:**
   - sin(θ) requires careful integration
   - Small dt near barrier crossing

3. **Multiple Time Scales:**
   - Fast oscillations: Period ~ 2π/√(g/L)
   - Slow transitions: τ_escape ~ exp(ΔE/σ²)

**Recommended Methods:**

1. **Euler-Maruyama:**
   - Simple, robust
   - dt ~ 0.01 s typical
   - Check convergence

2. **Milstein:**
   - Higher order
   - Requires derivatives (g' = 0 for additive noise → same as Euler)

3. **Stochastic Runge-Kutta:**
   - Better for very accurate trajectories
   - More computational cost

**For Pendulum:**
Euler-Maruyama with dt = 0.01-0.1 s usually sufficient.

Stochastic Resonance Example
-----------------------------

**Setup:**
- Weak periodic forcing: u = A·cos(ω·t), A small
- Noise: σ
- Bistable potential: ±1 wells

**Measure:** Signal-to-Noise Ratio vs σ

**Result:** SNR(σ) peaks at σ_opt!

**Why?**
- No noise: Cannot escape well, no response
- Optimal noise: Escapes synchronized with forcing
- Too much noise: Random escapes, no correlation

**Signature:**
Autocorrelation shows periodic component at ω.

Comparison with Deterministic
------------------------------

**Deterministic Pendulum:**
- Fixed trajectories
- Periodic orbits (oscillations)
- Separatrix divides oscillations from rotations
- Predictable (no randomness)

**Stochastic Pendulum:**
- Random trajectories
- Quasi-periodic (irregular oscillations)
- Noisy separatrix (probabilistic barrier)
- Transitions possible (activation)

**Critical Differences:**

1. **Barrier Crossing:**
   - Deterministic: Need E > ΔE (sufficient energy)
   - Stochastic: Possible at any E (rare events)

2. **Oscillation Period:**
   - Deterministic: Fixed by amplitude
   - Stochastic: Random (jitter)

3. **Long-Time Behavior:**
   - Deterministic: Decays to equilibrium (with damping)
   - Stochastic: Fluctuates forever (noise maintains activity)

Extensions
----------

**1. Multiplicative Noise:**
    dω = (-sin(θ) - b·ω + u)·dt + σ·ω·dW

Noise scales with velocity (rare in mechanics).

**2. Position and Velocity Noise:**
    dθ = ω·dt + σ_θ·dW_θ
    dω = (...)·dt + σ_ω·dW_ω

Two independent noise sources.

**3. Colored Noise:**
    dω = (...)·dt + η·dt
    dη = -α·η·dt + σ·dW

Noise has memory (Ornstein-Uhlenbeck driving).

**4. Parametric Noise:**
    dω = (-sin(θ) - b·ω + u)·dt + σ·sin(θ)·dW

Noise in potential (not force).

**5. Double Pendulum:**
Two coupled pendulums with noise - chaotic + random.

**6. Elastic Pendulum:**
Variable length with noise - more degrees of freedom.

Common Pitfalls
---------------

1. **Angle Wrapping:**
   - Forgetting to handle θ > 2π
   - Use unwrapped for counting rotations
   - Use wrapped for phase analysis

2. **Energy Conservation:**
   - Stochastic system doesn't conserve energy
   - Noise injects/removes energy randomly
   - Energy drifts over time

3. **Linearization Limits:**
   - Small-angle sin(θ) ≈ θ only for |θ| << 1
   - Kalman filter (linearized) fails for large swings
   - Need Extended/Unscented Kalman or particle filter

4. **Stiffness:**
   - Fast oscillations (ω_0 large) + slow barrier crossing
   - Multiple time scales
   - May need smaller dt or stiff solvers

5. **Ergodicity:**
   - With damping: Ergodic (long trajectory → stationary distribution)
   - Without damping: Not ergodic (energy preserved on average)

6. **Escape Rate:**
   - Exponentially sensitive to σ
   - Small changes in noise → huge changes in escape time
   - Difficult to estimate from simulation (rare events)

**Impact:**
Stochastic pendulum demonstrated that:
- Noise is not always detrimental (can be beneficial)
- Optimal noise level exists (not zero, not infinite)
- Nonlinearity + randomness creates new phenomena

Testing and Validation
-----------------------

**Statistical Tests:**

1. **Kramers Rate:**
   - Monte Carlo: Count escapes, estimate rate
   - Compare with theoretical exp(-ΔE/σ²)
   - Need many runs for rare events

2. **Stationary Distribution:**
   - Long simulation → histogram
   - Should match exp(-E/(σ²/b))
   - Chi-squared goodness-of-fit

3. **Autocorrelation:**
   - Measure correlation time
   - Coherence resonance: Maximum at σ_opt

4. **Energy Distribution:**
   - Should match Boltzmann-like distribution
   - Effective temperature T_eff = σ²/b

5. **Stochastic Resonance:**
   - Add periodic forcing
   - Measure SNR vs σ
   - Should peak (non-monotonic)

"""

import numpy as np
import sympy as sp

from src.systems.base.core.continuous_stochastic_system import ContinuousStochasticSystem


class ContinuousStochasticPendulum(ContinuousStochasticSystem):
    """
    Stochastic pendulum with random forcing - canonical nonlinear stochastic system.

    Combines the nonlinear dynamics of a simple pendulum with continuous random
    forcing, creating a fundamental model for studying noise-induced phenomena,
    Kramers' escape theory, stochastic resonance, and nonlinear stochastic control.

    Stochastic Differential Equations
    ----------------------------------
    State-space form:
        dθ = ω·dt
        dω = (-(g/L)·sin(θ) - b·ω + u)·dt + σ·dW

    Normalized form (g/L = 1):
        dθ = ω·dt
        dω = (-sin(θ) - b·ω + u)·dt + σ·dW

    where:
        - θ: Angle from downward vertical [rad]
        - ω: Angular velocity [rad/s]
        - g/L: Gravitational constant (normalized to 1)
        - b: Damping coefficient [1/s]
        - u: Applied torque [rad/s²]
        - σ: Noise intensity [rad/(s²·√s)]
        - W(t): Standard Wiener process

    Physical Interpretation
    -----------------------
    **Equation of Motion:**

    Newton's rotational law: I·θ̈ = τ_total

    For unit moment of inertia (I=1):
        θ̈ = τ_gravity + τ_damping + τ_control + τ_noise
        θ̈ = -(g/L)·sin(θ) - b·θ̇ + u + σ·ξ(t)

    where ξ(t) is white noise (formal derivative of W).

    **Torque Components:**
    1. Gravity: -(g/L)·sin(θ) - Restoring torque toward θ=0
    2. Damping: -b·ω - Dissipative (energy removal)
    3. Control: u - Applied torque
    4. Noise: σ·dW/dt - Random disturbances

    **Energy:**
        E = (1/2)·ω² + (g/L)·(1 - cos(θ))

    Kinetic + potential (normalized).

    **Noise Sources:**
    - Mechanical: Air currents, vibrations
    - Thermal: Brownian motion at pivot
    - Model uncertainty: Unmodeled dynamics

    Key Features
    ------------
    **Nonlinearity:**
    sin(θ) creates fundamentally different behavior from linear oscillator.

    **Bistability:**
    Two potential wells (θ = 0, ±2π, ...).
    Noise causes transitions.

    **Energy Barrier:**
    ΔE = 2·g/L separates downward from inverted.

    **Additive Noise:**
    Constant σ (state-independent).
    Enters velocity equation (physical forces).

    **Damping:**
    b > 0 required for stationary distribution.
    Without damping: Energy fluctuates, no equilibrium.

    Mathematical Properties
    -----------------------
    **Small Angle Approximation (|θ| << 1):**

    sin(θ) ≈ θ gives linearized SDE:
        dθ = ω·dt
        dω = (-(g/L)·θ - b·ω + u)·dt + σ·dW

    This is linear (harmonic oscillator with noise).

    **Stationary Distribution (b > 0, u = 0):**

    For damped pendulum:
        p_∞(θ, ω) ∝ exp(-E(θ,ω)/(σ²/b))

    where E = (1/2)·ω² - (g/L)·cos(θ).

    **Effective Temperature:**
        T_eff = σ²/b

    Ratio of noise injection to dissipation.

    **Kramers' Escape Rate:**

    From θ = 0 over barrier to θ = π:
        k_escape ≈ (ω_well²/2πb)·exp(-ΔE·b/σ²)

    Exponentially small for σ << √(ΔE·b).

    Physical Interpretation
    -----------------------
    **Damping b:**
    - Units: [1/s]
    - Controls energy dissipation
    - Time constant: τ = 1/b
    - Quality factor: Q = (g/L)^(1/2)/b

    **Examples:**
    - b = 0.1: Underdamped (Q = 10)
    - b = 1.0: Moderate damping
    - b = 10.0: Overdamped (creeping motion)

    **Noise Intensity σ:**
    - Units: [rad/(s²·√s)]
    - Controls random torque magnitude
    - Effective temperature: σ²/b

    **Examples:**
    - σ = 0.1: Small noise (rare escapes)
    - σ = 0.5: Moderate (occasional escapes)
    - σ = 1.0: Large (frequent escapes)

    **Ratio σ/√b:**
    Dimensionless noise level:
    - Small: Deep potential wells, rare transitions
    - Large: Shallow effective wells, frequent transitions

    State Space
    -----------
    State: X = [θ, ω]
        - θ ∈ ℝ or S¹: Angle (unwrapped or periodic)
        - ω ∈ ℝ: Angular velocity

    Control: u ∈ ℝ
        - Applied torque

    Noise: w ∈ ℝ
        - Single Wiener process
        - Enters velocity equation

    Parameters
    ----------
    g : float, default=9.81
        Gravitational acceleration [m/s²]

    L : float, default=1.0
        Pendulum length [m]

    b : float, default=0.5
        Damping coefficient [1/s]
        - b > 0 for stationary distribution
        - Typical: 0.1-10.0

    sigma : float, default=0.5
        Noise intensity [rad/(s²·√s)]
        - Controls random torque
        - Typical: 0.1-2.0

    m : float, default=1.0
        Mass [kg] (optional, for dimensional analysis)

    Stochastic Properties
    ---------------------
    - System Type: NONLINEAR
    - Noise Type: ADDITIVE
    - SDE Type: Itô
    - Noise Dimension: nw = 1
    - Stationary: Yes (if b > 0)
    - Ergodic: Yes (if b > 0)
    - Bistable: Yes (multiple wells)

    Applications
    ------------
    **1. Kramers' Escape Theory:**
    - Noise-activated barrier crossing
    - Chemical reaction rates
    - Exponential escape time

    **2. Stochastic Resonance:**
    - Weak periodic signal enhanced by noise
    - Optimal noise level exists
    - Biology, climate, electronics

    **3. Coherence Resonance:**
    - Noise creates coherent oscillations
    - No periodic forcing needed
    - Maximum regularity at optimal σ

    **4. Nonlinear Control:**
    - Swing-up with noise
    - Stabilization at inverted
    - Robust control design

    **5. Synchronization:**
    - Coupled oscillators
    - Common noise induces correlation
    - Phase locking

    Numerical Integration
    ---------------------
    **Recommended:**
    - Euler-Maruyama: dt = 0.01-0.1 s
    - Check angle wrapping (if needed)
    - Monitor energy (should fluctuate)

    **Convergence:**
    - Weak: O(dt) for moments
    - Strong: O(√dt) for paths

    Kramers Escape Analysis
    ------------------------
    **Escape Time:**
    Mean time to cross barrier:
        τ ~ exp(ΔE·b/σ²)

    Exponentially sensitive to:
    - Barrier height ΔE
    - Damping b (appears in exponent!)
    - Noise σ² (inverse in exponent)

    Comparison with Linear Oscillator
    ----------------------------------
    **Linear (Harmonic):**
    - sin(θ) ≈ θ
    - Gaussian stationary distribution
    - No bistability

    **Nonlinear (Pendulum):**
    - Full sin(θ)
    - Non-Gaussian stationary distribution
    - Bistable (multiple wells)

    Limitations
    -----------
    - 1D angle (no spatial motion)
    - Additive noise only
    - Constant parameters
    - No joint flexibility
    - Rigid body assumption

    Extensions
    ----------
    - Double pendulum (chaos + noise)
    - Elastic pendulum (variable length)
    - Spherical pendulum (2D angle)
    - Coupled pendulums (arrays)
    - Parametric noise (variable g or L)

    See Also
    --------
    Pendulum : Deterministic version
    StochasticDoubleIntegrator : Linear analog
    """

    def define_system(
        self,
        g: float = 9.81,
        L: float = 1.0,
        b: float = 0.5,
        sigma: float = 0.5,
        m: float = 1.0,
    ):
        """
        Define stochastic pendulum dynamics.

        Parameters
        ----------
        g : float, default=9.81
            Gravitational acceleration [m/s²]

        L : float, default=1.0
            Pendulum length [m]

        b : float, default=0.5
            Damping coefficient [1/s]
            - b > 0 required for stationary distribution
            - Typical: 0.1-10.0
            - Quality factor: Q = √(g/L)/b

        sigma : float, default=0.5
            Noise intensity [rad/(s²·√s)]
            - Controls random torque magnitude
            - Typical: 0.1-2.0
            - Effective temperature: T_eff = σ²/b

        m : float, default=1.0
            Mass [kg] (optional, for dimensional analysis)

        Notes
        -----
        **Natural Frequency:**
            ω_0 = √(g/L)

        For g=9.81, L=1: ω_0 ≈ 3.13 rad/s, Period ≈ 2 s

        **Damping Regimes:**
        
        Damping ratio: ζ = b/(2·ω_0)
        - ζ < 1: Underdamped (oscillatory decay)
        - ζ = 1: Critically damped (fastest non-oscillatory)
        - ζ > 1: Overdamped (slow exponential decay)

        **Effective Temperature:**
            T_eff = σ²/b

        Analogy with thermodynamics:
        - Noise injects energy: ~ σ²
        - Damping removes energy: ~ b
        - Equilibrium: Balance at T_eff

        **Energy Barrier:**
        From stable (θ=0) to unstable (θ=π):
            ΔE = 2·g/L

        For g=9.81, L=1: ΔE = 19.62 J/kg

        **Kramers Time:**
        Mean escape time:
            τ_escape ~ (b/ω_0²)·exp(ΔE·b/σ²)

        Example: b=0.5, σ=0.5, ΔE≈20
            → τ ~ 0.05·exp(20) ~ 2.4×10⁸ seconds!

        **Noise Level Guidelines:**

        Low noise (σ < 0.3):
        - Rare escapes (τ > hours)
        - Nearly deterministic
        - Small fluctuations around equilibrium

        Moderate noise (σ = 0.3-1.0):
        - Occasional escapes (τ ~ minutes to hours)
        - Visible stochastic effects
        - Coherence resonance regime

        High noise (σ > 1.0):
        - Frequent escapes (τ ~ seconds)
        - Dominated by randomness
        - Large fluctuations

        **Stochastic Resonance Regime:**
        For periodic forcing u = A·cos(ω·t):
        - Optimal σ ≈ √(ΔE·b) ≈ √(2·g/L·b)
        - Maximizes response to weak signal
        """
        if g <= 0:
            raise ValueError(f"g must be positive, got {g}")
        if L <= 0:
            raise ValueError(f"L must be positive, got {L}")
        if b < 0:
            raise ValueError(f"b must be non-negative, got {b}")
        if b == 0:
            import warnings
            warnings.warn(
                "b = 0 (no damping) means no stationary distribution. "
                "Energy will fluctuate randomly. For equilibrium, use b > 0.",
                UserWarning
            )
        if sigma < 0:
            raise ValueError(f"sigma must be non-negative, got {sigma}")
        if m <= 0:
            raise ValueError(f"m must be positive, got {m}")

        # State variables
        theta, omega = sp.symbols("theta omega", real=True)
        u = sp.symbols("u", real=True)

        # Parameters
        g_sym, L_sym, b_sym, m_sym = sp.symbols("g L b m", positive=True)
        sigma_sym = sp.symbols("sigma", real=True, nonnegative=True)

        self.state_vars = [theta, omega]
        self.control_vars = [u]

        # DRIFT (Deterministic pendulum dynamics)
        # dθ/dt = ω
        # dω/dt = -(g/L)·sin(θ) - b·ω + u
        self._f_sym = sp.Matrix([
            omega,
            -(g_sym / L_sym) * sp.sin(theta) - b_sym * omega + u
        ])

        self.parameters = {
            g_sym: g,
            L_sym: L,
            b_sym: b,
            sigma_sym: sigma,
            m_sym: m,
        }
        self.order = 1

        # DIFFUSION (Random forcing)
        # Noise on velocity equation only (physical forces)
        self.diffusion_expr = sp.Matrix([
            [0],
            [sigma_sym]
        ])

        # Itô SDE
        self.sde_type = "ito"

        # Output: Typically measure angle
        self._h_sym = sp.Matrix([theta])

    def setup_equilibria(self):
        """
        Set up equilibrium points (deterministic part).

        Pendulum has periodic equilibria at θ = n·π.
        """
        # Downward (stable)
        self.add_equilibrium(
            "downward",
            x_eq=np.array([0.0, 0.0]),
            u_eq=np.array([0.0]),
            verify=True,
            stability="stable",
            notes="Stable equilibrium (hanging down). With noise, stationary "
                  "distribution peaks here. Occasionally escapes over barrier."
        )

        # Upward (unstable)
        self.add_equilibrium(
            "upward",
            x_eq=np.array([np.pi, 0.0]),
            u_eq=np.array([0.0]),
            verify=True,
            stability="unstable",
            notes="Unstable equilibrium (inverted). Energy barrier separating "
                  "from stable equilibrium. Noise causes escapes."
        )

        self.set_default_equilibrium("downward")

    def compute_energy(self, theta: float, omega: float) -> float:
        """
        Compute mechanical energy.

        E = (1/2)·ω² + (g/L)·(1 - cos(θ))

        Parameters
        ----------
        theta : float
            Angle [rad]
        omega : float
            Angular velocity [rad/s]

        Returns
        -------
        float
            Energy [J/kg] (or normalized)

        Notes
        -----
        Energy fluctuates randomly due to noise.
        With damping: Mean energy dissipates to T_eff = σ²/b.

        Examples
        --------
        >>> pend = ContinuousStochasticPendulum()
        >>> E = pend.compute_energy(theta=0.5, omega=1.0)
        >>> print(f"Energy: {E:.3f}")
        """
        g = self.parameters[sp.symbols('g')]
        L = self.parameters[sp.symbols('L')]
        
        KE = 0.5 * omega**2
        PE = (g / L) * (1 - np.cos(theta))
        
        return KE + PE

    def get_natural_frequency(self) -> float:
        """
        Get natural frequency ω_0 = √(g/L).

        Returns
        -------
        float
            Natural frequency [rad/s]

        Examples
        --------
        >>> pend = ContinuousStochasticPendulum(g=9.81, L=1.0)
        >>> omega_0 = pend.get_natural_frequency()
        >>> period = 2 * np.pi / omega_0
        >>> print(f"Period: {period:.2f} s")
        """
        g = self.parameters[sp.symbols('g')]
        L = self.parameters[sp.symbols('L')]
        return np.sqrt(g / L)

    def get_effective_temperature(self) -> float:
        """
        Get effective temperature T_eff = σ²/b.

        Ratio of noise injection to dissipation.

        Returns
        -------
        float
            Effective temperature

        Notes
        -----
        Analogy with thermodynamics:
        - Higher T_eff: More energetic fluctuations
        - Lower T_eff: Smaller fluctuations
        - Controls stationary distribution width

        Examples
        --------
        >>> pend = ContinuousStochasticPendulum(b=0.5, sigma=0.5)
        >>> T_eff = pend.get_effective_temperature()
        >>> print(f"T_eff: {T_eff:.3f}")
        """
        b = self.parameters[sp.symbols('b')]
        sigma = self.parameters[sp.symbols('sigma')]
        
        if b == 0:
            return np.inf
        
        return sigma**2 / b

    def estimate_kramers_escape_time(self) -> float:
        """
        Estimate mean escape time using Kramers formula.

        τ ~ (2πb/ω_0²)·exp(ΔE·b/σ²)

        Returns
        -------
        float
            Estimated mean escape time [s]

        Notes
        -----
        This is approximate, valid for:
        - Small noise: σ² << ΔE·b
        - Moderate damping: b ~ ω_0

        For accurate times, use Monte Carlo simulation.

        Examples
        --------
        >>> pend = ContinuousStochasticPendulum(b=0.5, sigma=0.5)
        >>> tau = pend.estimate_kramers_escape_time()
        >>> print(f"Mean escape time: {tau:.2e} s")
        """
        g = self.parameters[sp.symbols('g')]
        L = self.parameters[sp.symbols('L')]
        b = self.parameters[sp.symbols('b')]
        sigma = self.parameters[sp.symbols('sigma')]
        
        omega_0 = np.sqrt(g / L)
        Delta_E = 2 * g / L  # Barrier height
        
        # Kramers formula
        prefactor = 2 * np.pi * b / omega_0**2
        exponential = np.exp(Delta_E * b / sigma**2)
        
        return prefactor * exponential


# Convenience functions
def create_unit_pendulum(
    damping: float = 0.5,
    noise_level: float = 0.5,
) -> ContinuousStochasticPendulum:
    """
    Create unit length pendulum with Earth gravity.

    Parameters
    ----------
    damping : float, default=0.5
        Damping coefficient b [1/s]
    noise_level : float, default=0.5
        Noise intensity σ [rad/(s²·√s)]

    Returns
    -------
    ContinuousStochasticPendulum

    Examples
    --------
    >>> # Standard unit pendulum
    >>> pend = create_unit_pendulum(damping=0.5, noise_level=0.5)
    """
    return ContinuousStochasticPendulum(
        g=9.81,
        L=1.0,
        b=damping,
        sigma=noise_level,
        m=1.0
    )


def create_josephson_junction(
    critical_current: float = 1.0,
    capacitance: float = 1.0,
    resistance: float = 10.0,
    temperature: float = 4.2,
) -> ContinuousStochasticPendulum:
    """
    Create Josephson junction model (pendulum analog).

    Resistively-shunted junction (RSJ) model with thermal noise.

    Parameters
    ----------
    critical_current : float
        I_c [A]
    capacitance : float
        C [F]
    resistance : float
        R [Ω]
    temperature : float
        T [K]

    Returns
    -------
    ContinuousStochasticPendulum

    Notes
    -----
    Josephson junction phase difference φ obeys:
        C·φ̈ + (1/R)·φ̇ + I_c·sin(φ) = I + i_noise

    This is a driven stochastic pendulum!

    Thermal noise intensity:
        σ² = 2·k_B·T/R

    Examples
    --------
    >>> # Josephson junction at liquid helium temperature
    >>> junction = create_josephson_junction(
    ...     critical_current=1.0,
    ...     resistance=10.0,
    ...     temperature=4.2  # Liquid He
    ... )
    """
    # Josephson plasma frequency
    omega_p = np.sqrt(critical_current / capacitance)
    
    # Damping from resistance
    b = 1.0 / (resistance * capacitance)
    
    # Thermal noise (Johnson-Nyquist)
    k_B = 1.380649e-23  # Boltzmann constant
    sigma = np.sqrt(2 * k_B * temperature / resistance)
    
    # Map to pendulum
    # g/L → ω_p²
    g_eff = omega_p**2
    L_eff = 1.0
    
    return ContinuousStochasticPendulum(
        g=g_eff,
        L=L_eff,
        b=b,
        sigma=sigma,
        m=1.0
    )


def create_stochastic_resonance_pendulum(
    barrier_height: float = 2.0,
    damping: float = 0.5,
) -> ContinuousStochasticPendulum:
    """
    Create pendulum optimized for stochastic resonance demonstration.

    Parameters
    ----------
    barrier_height : float, default=2.0
        Energy barrier ΔE
    damping : float, default=0.5
        Damping coefficient b

    Returns
    -------
    ContinuousStochasticPendulum

    Notes
    -----
    Optimal noise for stochastic resonance:
        σ_opt ≈ √(ΔE·b/2)

    For ΔE=2, b=0.5: σ_opt ≈ 0.71

    Examples
    --------
    >>> # Setup for stochastic resonance
    >>> pend = create_stochastic_resonance_pendulum(
    ...     barrier_height=2.0,
    ...     damping=0.5
    ... )
    >>> 
    >>> # Optimal noise
    >>> sigma_opt = np.sqrt(2.0 * 0.5 / 2)
    >>> print(f"Optimal σ ≈ {sigma_opt:.2f}")
    """
    # Choose g/L such that ΔE = 2·g/L matches desired barrier
    g_L = barrier_height / 2.0
    
    # Optimal noise for stochastic resonance
    sigma_opt = np.sqrt(barrier_height * damping / 2)
    
    return ContinuousStochasticPendulum(
        g=g_L,
        L=1.0,
        b=damping,
        sigma=sigma_opt,
        m=1.0
    )