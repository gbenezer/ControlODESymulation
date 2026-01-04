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
Discrete-time logistic map (chaotic population dynamics).

This module provides the classic logistic map, one of the most studied
discrete dynamical systems in mathematics. Despite its simple form, it
exhibits extraordinarily rich behavior including:
- Fixed points and period-doubling bifurcations
- Chaotic dynamics and strange attractors
- Sensitive dependence on initial conditions
- Universal scaling laws (Feigenbaum constants)
- Windows of periodic behavior within chaos

The logistic map serves as:
- A canonical example of deterministic chaos
- A model for population dynamics with limited resources
- A testbed for understanding nonlinear dynamics
- An illustration of the period-doubling route to chaos
- A pedagogical tool for dynamical systems theory

Originally proposed by Robert May (1976) to model biological populations,
the logistic map has become foundational in chaos theory, appearing in
contexts from ecology to cryptography to neural networks.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import sympy as sp
from scipy.optimize import fsolve

from src.systems.base.core.discrete_symbolic_system import DiscreteSymbolicSystem


class LogisticMap(DiscreteSymbolicSystem):
    """
    The logistic map: x[k+1] = r·x[k]·(1 - x[k])

    Physical System:
    ---------------
    The logistic map was introduced by Robert May in 1976 as a discrete-time
    model for population dynamics with limited resources. It represents the
    simplest nonlinear difference equation exhibiting complex behavior.

    **Biological Interpretation:**
    Consider a population with:
    - x[k]: Population at generation k (normalized: 0 ≤ x ≤ 1)
      * x = 0: Extinction
      * x = 1: Maximum carrying capacity
      * x = 0.5: Half of carrying capacity

    - r: Growth rate parameter (0 ≤ r ≤ 4)
      * Controls reproduction rate
      * Higher r → faster growth
      * Too high r → instability, chaos

    The dynamics capture:
    1. **Reproduction**: r·x[k] term (proportional to population)
    2. **Competition**: -r·x[k]² term (limited resources)
    3. **Net growth**: Balance between birth and death

    **Discrete Time Steps:**
    Unlike continuous models (differential equations), the logistic map
    assumes discrete generations with no overlap. This is appropriate for:
    - Insects with distinct seasonal generations
    - Annual plants
    - Laboratory populations with controlled breeding
    - Economic cycles with discrete time periods
    - Digital sampling of continuous processes

    **The Remarkable Complexity:**
    Despite having only ONE parameter (r) and ONE state variable (x),
    the logistic map exhibits virtually all known behaviors of dynamical
    systems:
    - Fixed points (equilibria)
    - Periodic orbits (period-2, 4, 8, 16, ...)
    - Period-doubling cascades
    - Deterministic chaos
    - Intermittency
    - Crises and sudden changes
    - Strange attractors

    This makes it a "Rosetta Stone" for understanding nonlinear dynamics.

    State Space:
    -----------
    State: x[k] ∈ [0, 1]
        Population fraction:
        - x[k] = 0: Extinction
        - x[k] = 1: Maximum capacity
        - 0 < x[k] < 1: Viable population

        **Physical meaning:**
        If N_max is carrying capacity, actual population is:
            N[k] = x[k]·N_max

    Control: u[k] (optional, for controlled logistic map)
        - Can represent:
          * Harvesting/culling: u < 0
          * Stocking/immigration: u > 0
          * Environmental intervention
        - Standard logistic map: u = 0 (autonomous)

    Output: y[k] = x[k]
        - Direct observation of population fraction
        - In practice, population counts with measurement noise

    Dynamics:
    --------
    The defining equation is deceptively simple:

        x[k+1] = r·x[k]·(1 - x[k])

    **Mathematical Properties:**

    1. **Quadratic map:**
       - Second-order polynomial
       - Single hump (inverted parabola)
       - Maximum at x = 0.5: f(0.5) = r/4

    2. **Bounded dynamics:**
       - If x[0] ∈ [0, 1], then x[k] ∈ [0, 1] for all k (when r ≤ 4)
       - Invariant interval: [0, 1]
       - Escape possible if r > 4

    3. **Two fixed points:**
       Setting x[k+1] = x[k]:
           x* = r·x*·(1 - x*)
           
       Solutions:
           x₁* = 0  (extinction)
           x₂* = 1 - 1/r  (nontrivial equilibrium)

    4. **Stability via eigenvalue:**
       The Jacobian is just a scalar:
           λ(x*) = df/dx|_{x*} = r·(1 - 2x*)

       Fixed point is stable if |λ| < 1.

    Parameters:
    ----------
    r : float, default=3.5
        Growth rate parameter (must satisfy 0 < r ≤ 4 for bounded dynamics)

        **Parameter Regimes:**

        **0 < r < 1:**
        - Extinction regime
        - All initial conditions → 0
        - Both fixed points unstable or non-existent
        - Not biologically realistic (population dies out)

        **1 ≤ r < 3:**
        - Stable fixed point regime
        - Population converges to x* = 1 - 1/r
        - Monotonic approach (no oscillations)
        - λ = r·(1 - 2x*) = 2 - r
        - |λ| < 1 requires r < 3

        **3 ≤ r < 1 + √6 ≈ 3.449:**
        - Period-2 oscillations
        - Fixed point becomes unstable (|λ| > 1)
        - Population oscillates between two values
        - First period-doubling bifurcation at r = 3

        **3.449 < r < 3.544:**
        - Period-doubling cascade
        - Period-4, 8, 16, 32, ... orbits
        - Feigenbaum cascade to chaos
        - Each bifurcation occurs at specific r values
        - Spacing between bifurcations decreases geometrically

        **3.544 < r < 3.569:**
        - Onset of chaos
        - Aperiodic, bounded, deterministic behavior
        - Sensitive dependence on initial conditions
        - Strange attractor forms
        - Lyapunov exponent becomes positive

        **3.569 < r < 4:**
        - Fully developed chaos with periodic windows
        - Period-3 window at r ≈ 3.83 (remarkable!)
        - Li-Yorke theorem: period-3 implies chaos
        - Infinitely many periodic windows
        - Fractal structure in bifurcation diagram

        **r = 4:**
        - Special case: fully chaotic
        - Exact solution possible (analytically)
        - Probability density: ρ(x) = 1/(π√(x(1-x)))
        - Maximum Lyapunov exponent: λ = ln(2)

        **r > 4:**
        - Escape regime
        - Trajectories can leave [0, 1]
        - Eventually diverge to -∞
        - Not physically meaningful for population model

    dt : float, default=1.0
        Time step between generations
        - Usually normalized to 1 (one generation)
        - Can represent actual time scale (years, days, etc.)
        - Purely for interpretation (doesn't affect dynamics)

    Equilibria:
    ----------
    The logistic map has two fixed points:

    **1. Extinction equilibrium: x* = 0**
       Stability: λ = r
       - Stable if r < 1 (low growth → extinction)
       - Unstable if r > 1 (population survives)

       Biological meaning: If population is too small (below critical
       threshold), it cannot sustain itself and dies out.

    **2. Nontrivial equilibrium: x* = 1 - 1/r (for r > 1)**
       Stability: λ = 2 - r
       - Stable if 1 < r < 3 (|λ| < 1)
       - Unstable if r > 3 (period-doubling begins)

       Biological meaning: Population reaches carrying capacity balance
       where births equal deaths.

    **Bifurcation Points:**
    - r = 1: Transcritical bifurcation (extinction ↔ survival)
    - r = 3: Period-doubling bifurcation (fixed point → period-2)
    - r ≈ 3.449: Period-4 bifurcation
    - r ≈ 3.544: Period-8 bifurcation
    - r ≈ 3.569: Onset of chaos (accumulation point)
    - r ≈ 3.83: Period-3 window opens

    **Feigenbaum Constants:**
    The period-doubling cascade follows universal scaling:
        δ = lim (rₙ - rₙ₋₁)/(rₙ₊₁ - rₙ) = 4.669...
        α = lim (dₙ/dₙ₊₁) = 2.502...

    These constants are universal across all period-doubling systems!

    Chaos Theory Concepts:
    ---------------------
    **Sensitive Dependence on Initial Conditions:**
    The hallmark of chaos. Two trajectories starting infinitesimally
    close diverge exponentially:

        |δx(k)| ≈ |δx(0)|·e^(λk)

    where λ > 0 is the Lyapunov exponent.

    For logistic map in chaotic regime:
        λ = lim (1/N)·Σ ln|r·(1 - 2x[k])|
        N→∞

    - λ < 0: Periodic or fixed point (convergence)
    - λ = 0: Neutral stability (period-doubling point)
    - λ > 0: Chaos (divergence)

    **Predictability Horizon:**
    If measurement accuracy is ε, prediction fails after time:
        T_predict ~ |λ|⁻¹·ln(ε)

    Example: For λ = 0.5 and ε = 10⁻⁶:
        T_predict ~ 2·ln(10⁶) ≈ 28 generations

    This is why weather prediction is limited despite deterministic physics!

    **Strange Attractor:**
    In chaotic regime, the system visits a fractal set:
    - Infinite complexity at all scales
    - Non-integer (fractal) dimension
    - Dense periodic orbits
    - Mixing property (ergodic)

    **Determinism vs Randomness:**
    The logistic map is COMPLETELY DETERMINISTIC (no randomness), yet
    produces output that appears random:
    - Passes statistical tests for randomness
    - Used in pseudo-random number generation
    - Cannot predict long-term behavior
    - "Deterministic chaos" is not an oxymoron!

    Control Objectives:
    ------------------
    Unlike typical control systems, controlling chaos involves different goals:

    1. **Chaos Control (OGY Method):**
       Goal: Stabilize unstable periodic orbits embedded in attractor
       Method: Small perturbations to system parameter
       Applications: Laser dynamics, cardiac arrhythmias

    2. **Chaos Synchronization:**
       Goal: Make two chaotic systems follow same trajectory
       Applications: Secure communications, neural networks

    3. **Anti-control (Chaotification):**
       Goal: Make periodic system chaotic
       Applications: Mixing, encryption, avoiding resonance

    4. **Harvesting Optimization:**
       Goal: Maximize sustainable yield while maintaining stability
       Control: u[k] = h·x[k] (proportional harvesting)
       Constraint: Keep r_effective in stable regime

    5. **Bifurcation Control:**
       Goal: Move bifurcation points by parameter variation
       Applications: Preventing unwanted oscillations

    Bifurcation Analysis:
    --------------------
    **Creating Bifurcation Diagrams:**
    1. Choose r values from 0 to 4
    2. For each r:
       a. Iterate from random initial condition
       b. Discard transient (e.g., first 1000 iterations)
       c. Plot next 100 iterations
    3. Result: shows all attractors vs parameter

    **Reading Bifurcation Diagrams:**
    - Single line: Fixed point
    - Two lines: Period-2 orbit
    - Four lines: Period-4 orbit
    - Dense region: Chaos
    - White gaps: Periodic windows

    **Universality:**
    All unimodal maps with quadratic maximum exhibit same qualitative
    bifurcation structure. The Feigenbaum constants are universal!

    Numerical Considerations:
    ------------------------
    **Floating-Point Precision:**
    Computer arithmetic introduces errors that can affect long-term behavior:
    - Chaotic systems amplify roundoff errors exponentially
    - Cannot compute trajectory accurately beyond predictability horizon
    - Use high-precision arithmetic for numerical studies
    - Machine epsilon (≈10⁻¹⁶) limits practical predictions

    **Initial Condition Sensitivity:**
    Even tiny errors in initial condition grow exponentially:
        |error[k]| ≈ |error[0]|·e^(λk)

    For λ = 0.5 and error[0] = 10⁻¹⁵:
        error[50] ≈ 10⁻¹⁵·e²⁵ ≈ 10⁻²

    **Stability of Fixed Points:**
    To find stability, compute derivative:
        λ = df/dx = r·(1 - 2x)

    At x* = 1 - 1/r:
        λ = r·(1 - 2(1 - 1/r)) = 2 - r

    Stable if |2 - r| < 1, i.e., 1 < r < 3.

    **Cobweb Diagrams:**
    Graphical method to visualize iterations:
    1. Plot y = f(x) and y = x
    2. Start at (x₀, 0)
    3. Go vertically to curve: (x₀, f(x₀))
    4. Go horizontally to diagonal: (f(x₀), f(x₀))
    5. Repeat

    Patterns:
    - Spiral inward → stable fixed point
    - Spiral outward → unstable fixed point
    - Rectangle → period-2 orbit
    - Complicated path → chaos

    Example Usage:
    -------------
    >>> # Create logistic map in chaotic regime
    >>> system = LogisticMap(r=3.9)
    >>> 
    >>> # Simulate from random initial condition
    >>> x0 = np.array([0.4])
    >>> result = system.simulate(
    ...     x0=x0,
    ...     u_sequence=None,  # Autonomous system
    ...     n_steps=100
    ... )
    >>> 
    >>> # Plot trajectory
    >>> import matplotlib.pyplot as plt
    >>> plt.figure(figsize=(10, 6))
    >>> plt.plot(result['time_steps'], result['states'][:, 0], 'b.-', markersize=3)
    >>> plt.xlabel('Generation k')
    >>> plt.ylabel('Population x[k]')
    >>> plt.title(f'Logistic Map (r = {system.r})')
    >>> plt.grid(alpha=0.3)
    >>> plt.show()
    >>> 
    >>> # Demonstrate sensitivity to initial conditions
    >>> x0_a = np.array([0.4])
    >>> x0_b = np.array([0.4 + 1e-10])  # Tiny difference
    >>> 
    >>> result_a = system.simulate(x0_a, None, n_steps=50)
    >>> result_b = system.simulate(x0_b, None, n_steps=50)
    >>> 
    >>> # Compute divergence
    >>> divergence = np.abs(result_a['states'][:, 0] - result_b['states'][:, 0])
    >>> 
    >>> plt.figure(figsize=(10, 6))
    >>> plt.semilogy(result_a['time_steps'], divergence, 'r-', linewidth=2)
    >>> plt.xlabel('Generation k')
    >>> plt.ylabel('|x_a[k] - x_b[k]|')
    >>> plt.title('Exponential Divergence (Sensitive Dependence)')
    >>> plt.grid(alpha=0.3)
    >>> plt.show()
    >>> 
    >>> # Estimate Lyapunov exponent
    >>> lyapunov = system.compute_lyapunov_exponent(x0=0.4, n_iterations=10000)
    >>> print(f"Lyapunov exponent: λ = {lyapunov:.4f}")
    >>> if lyapunov > 0:
    ...     print("System is CHAOTIC")
    ... else:
    ...     print("System is NOT chaotic (periodic or fixed point)")
    >>> 
    >>> # Generate bifurcation diagram
    >>> r_values = np.linspace(2.5, 4.0, 1000)
    >>> bifurcation_data = system.generate_bifurcation_diagram(
    ...     r_values=r_values,
    ...     n_transient=500,
    ...     n_samples=100
    ... )
    >>> 
    >>> plt.figure(figsize=(12, 8))
    >>> plt.plot(bifurcation_data['r'], bifurcation_data['x'], 
    ...          'k.', markersize=0.5, alpha=0.5)
    >>> plt.xlabel('Growth Rate r')
    >>> plt.ylabel('Population x*')
    >>> plt.title('Bifurcation Diagram: Period-Doubling Route to Chaos')
    >>> plt.xlim(2.5, 4.0)
    >>> plt.ylim(0, 1)
    >>> plt.grid(alpha=0.3)
    >>> 
    >>> # Annotate key bifurcation points
    >>> plt.axvline(3.0, color='r', linestyle='--', alpha=0.5, label='Period-2 bifurcation')
    >>> plt.axvline(3.449, color='g', linestyle='--', alpha=0.5, label='Period-4')
    >>> plt.axvline(3.569, color='b', linestyle='--', alpha=0.5, label='Chaos onset')
    >>> plt.legend()
    >>> plt.show()
    >>> 
    >>> # Find fixed points and check stability
    >>> fixed_points = system.find_fixed_points()
    >>> print("\\nFixed Points:")
    >>> for fp in fixed_points:
    ...     x_star = fp['x']
    ...     lambda_val = fp['eigenvalue']
    ...     stable = fp['stable']
    ...     print(f"  x* = {x_star:.4f}, λ = {lambda_val:.4f}, stable = {stable}")
    >>> 
    >>> # Cobweb diagram for visualization
    >>> fig, ax = plt.subplots(figsize=(8, 8))
    >>> system.plot_cobweb(ax, x0=0.1, n_iterations=50)
    >>> plt.show()
    >>> 
    >>> # Period-3 window (r ≈ 3.83)
    >>> system_p3 = LogisticMap(r=3.83)
    >>> result_p3 = system_p3.simulate(x0=np.array([0.5]), u_sequence=None, n_steps=100)
    >>> 
    >>> # Check for period-3 by looking at every 3rd point
    >>> period = system_p3.detect_period(result_p3['states'][:, 0])
    >>> print(f"\\nDetected period: {period}")
    >>> 
    >>> # Compare different r values
    >>> r_test = [2.8, 3.2, 3.5, 3.9]
    >>> fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    >>> 
    >>> for idx, r_val in enumerate(r_test):
    ...     ax = axes[idx // 2, idx % 2]
    ...     sys_temp = LogisticMap(r=r_val)
    ...     res_temp = sys_temp.simulate(x0=np.array([0.4]), u_sequence=None, n_steps=100)
    ...     
    ...     ax.plot(res_temp['time_steps'], res_temp['states'][:, 0], 'b.-', markersize=3)
    ...     ax.set_xlabel('Generation k')
    ...     ax.set_ylabel('Population x[k]')
    ...     ax.set_title(f'r = {r_val}')
    ...     ax.grid(alpha=0.3)
    ...     ax.set_ylim(0, 1)
    >>> 
    >>> plt.tight_layout()
    >>> plt.show()
    >>> 
    >>> # Analyze return map (x[k+1] vs x[k])
    >>> result_long = system.simulate(x0=np.array([0.4]), u_sequence=None, n_steps=1000)
    >>> x_k = result_long['states'][:-1, 0]
    >>> x_k1 = result_long['states'][1:, 0]
    >>> 
    >>> plt.figure(figsize=(8, 8))
    >>> plt.plot(x_k, x_k1, 'b.', markersize=1, alpha=0.5)
    >>> plt.plot([0, 1], [0, 1], 'r--', label='y = x (fixed points)')
    >>> x_plot = np.linspace(0, 1, 1000)
    >>> plt.plot(x_plot, system.r * x_plot * (1 - x_plot), 'g-', linewidth=2, 
    ...          label=f'y = {system.r}x(1-x)')
    >>> plt.xlabel('x[k]')
    >>> plt.ylabel('x[k+1]')
    >>> plt.title('Return Map (First-Return Plot)')
    >>> plt.legend()
    >>> plt.grid(alpha=0.3)
    >>> plt.axis('equal')
    >>> plt.xlim(0, 1)
    >>> plt.ylim(0, 1)
    >>> plt.show()
    >>> 
    >>> # Compute histogram (invariant measure)
    >>> result_hist = system.simulate(x0=np.array([0.4]), u_sequence=None, n_steps=50000)
    >>> x_hist = result_hist['states'][10000:, 0]  # Discard transient
    >>> 
    >>> plt.figure(figsize=(10, 6))
    >>> plt.hist(x_hist, bins=100, density=True, alpha=0.7, edgecolor='black')
    >>> plt.xlabel('Population x')
    >>> plt.ylabel('Probability Density')
    >>> plt.title(f'Invariant Measure (r = {system.r})')
    >>> plt.grid(alpha=0.3)
    >>> plt.show()

    Physical Insights:
    -----------------
    **Why Does Chaos Occur?**
    The logistic map has two competing effects:
    1. **Expansion**: For small x, f(x) ≈ rx (exponential growth)
    2. **Folding**: For x near 1, f(x) decreases (resource limitation)

    When r is large, expansion is strong, stretching the interval.
    The folding then brings points back, but in a complicated way.
    This "stretch and fold" mechanism creates sensitive dependence.

    **Biological Meaning of Chaos:**
    Chaotic population dynamics mean:
    - Population never settles to equilibrium
    - Year-to-year populations appear random
    - Long-term prediction impossible
    - Small environmental changes → large effects

    Examples in nature:
    - Measles epidemics (pre-vaccination)
    - Lynx-hare population cycles
    - Plankton blooms
    - Insect outbreaks

    **Period-3 and Li-Yorke Theorem:**
    "Period-3 implies chaos" (Li and Yorke, 1975)

    If a continuous map has a period-3 orbit, then:
    - It has periodic orbits of all periods
    - It has uncountably many aperiodic orbits
    - It exhibits sensitive dependence

    This is why the period-3 window at r ≈ 3.83 is remarkable!

    **Feigenbaum Universality:**
    Mitchell Feigenbaum (1975) discovered that ALL unimodal maps with
    quadratic maximum exhibit the same scaling:
        - δ = 4.669... (bifurcation spacing ratio)
        - α = 2.502... (parameter scaling)

    This means chaos theory has universal laws, like physics!

    **Tent Map and Topological Conjugacy:**
    The logistic map at r = 4 is topologically conjugate to the tent map:
        g(x) = { 2x       if x < 0.5
               { 2(1-x)   if x ≥ 0.5

    Transformation: x = sin²(πy/2)

    This allows exact analytical solutions for r = 4.

    **Connection to Cryptography:**
    Chaotic maps used in encryption because:
    - Deterministic (key = initial condition and parameter)
    - Sensitive dependence (good mixing)
    - Appears random (passes statistical tests)
    - Fast computation

    However, not cryptographically secure due to:
    - Finite precision issues
    - Reconstructability from time series

    **Control Paradox:**
    It's often EASIER to control chaotic systems than periodic ones!
    - Chaos has dense set of unstable periodic orbits
    - Can stabilize any orbit with small perturbations
    - Periodic systems may need large control effort to change behavior

    Common Pitfalls:
    ---------------
    1. **Escaping the unit interval:**
       For r > 4, trajectories can escape [0, 1] and diverge to -∞.
       Always check x[k] ∈ [0, 1].

    2. **Transient vs asymptotic behavior:**
       Must discard initial transient before analyzing attractor.
       Typical: skip first 500-1000 iterations.

    3. **Numerical precision limits:**
       Cannot compute chaotic trajectory accurately beyond ~50-100 iterations.
       Use double precision and be aware of limitations.

    4. **Mistaking chaos for randomness:**
       Chaos is deterministic! Same initial condition → same trajectory.
       But prediction is practically impossible for long times.

    5. **Period detection:**
       Short period easy to detect, but high-period orbits hard to distinguish
       from chaos. Use Lyapunov exponent, not just visual inspection.

    6. **Parameter precision:**
       Bifurcation points occur at specific r values.
       Need high precision to locate them accurately.

    Extensions and Variations:
    -------------------------
    1. **Generalized logistic map:**
       x[k+1] = r·x[k]^α·(1 - x[k])
       Different values of α change bifurcation structure

    2. **Coupled logistic maps:**
       Spatial extension: x_i[k+1] = f(x_i[k]) + ε·(x_{i-1} + x_{i+1} - 2x_i)
       Creates spatiotemporal chaos

    3. **Delayed logistic map:**
       x[k+1] = r·x[k]·(1 - x[k-τ])
       Time delay increases complexity

    4. **Stochastic logistic map:**
       x[k+1] = r·x[k]·(1 - x[k]) + σ·ξ[k]
       Noise + deterministic chaos = rich dynamics

    5. **Discrete-time Ricker model:**
       x[k+1] = x[k]·exp(r·(1 - x[k]))
       Similar dynamics, different form

    6. **Henon map (2D):**
       x[k+1] = 1 - a·x[k]² + y[k]
       y[k+1] = b·x[k]
       Classic 2D chaotic map

    See Also:
    --------
    HenonMap : 2D chaotic map
    StandardMap : Hamiltonian chaos
    """

    def define_system(
        self,
        r: float = 3.5,
        dt: float = 1.0,
        use_controlled_version: bool = False,
    ):
        """
        Define logistic map dynamics.

        Parameters
        ----------
        r : float
            Growth rate parameter (typically 0 < r ≤ 4)
        dt : float
            Time step between generations (typically 1.0)
        use_controlled_version : bool
            If True, adds control input u[k] to dynamics:
            x[k+1] = r·x[k]·(1 - x[k]) + u[k]
        """
        # Store parameters
        self._r_val = r
        self._use_controlled = use_controlled_version

        # Validate parameter
        if r <= 0:
            raise ValueError(f"Growth rate r must be positive, got r = {r}")
        if r > 4.0:
            import warnings
            warnings.warn(
                f"Growth rate r = {r} > 4 may cause trajectories to escape [0, 1]. "
                "Consider r ≤ 4 for bounded dynamics.",
                UserWarning
            )

        # State variable
        x = sp.symbols('x', real=True)
        
        # Symbolic parameter
        r_sym = sp.symbols('r', positive=True, real=True)

        self.state_vars = [x]
        self.parameters = {r_sym: r}
        self._dt = dt
        self.order = 1

        if use_controlled_version:
            # Controlled logistic map
            u = sp.symbols('u', real=True)
            self.control_vars = [u]
            x_next = r_sym * x * (1 - x) + u
        else:
            # Standard autonomous logistic map
            self.control_vars = []
            x_next = r_sym * x * (1 - x)

        self._f_sym = sp.Matrix([x_next])

    def setup_equilibria(self):
        """
        Set up fixed points of the logistic map.

        Adds:
        - Extinction equilibrium (x* = 0)
        - Nontrivial equilibrium (x* = 1 - 1/r) if r > 1
        """
        r = self._r_val

        # Extinction equilibrium (always exists)
        extinction_stable = r < 1
        extinction_eigenvalue = r

        self.add_equilibrium(
            "extinction",
            x_eq=np.array([0.0]),
            u_eq=np.array([0.0]) if self._use_controlled else np.array([]),
            verify=True,
            stability="stable" if extinction_stable else "unstable",
            eigenvalue=extinction_eigenvalue,
            notes=f"λ = {extinction_eigenvalue:.3f}. Stable if r < 1.",
        )

        # Nontrivial equilibrium (exists if r > 1)
        if r > 1:
            x_star = 1.0 - 1.0 / r
            eigenvalue = 2.0 - r
            nontrivial_stable = abs(eigenvalue) < 1  # True if 1 < r < 3

            if nontrivial_stable:
                stability_str = "stable"
                notes = f"λ = {eigenvalue:.3f}. Stable fixed point (1 < r < 3)."
            elif r < 3.449:
                stability_str = "unstable"
                notes = f"λ = {eigenvalue:.3f}. Unstable (period-2 orbit exists)."
            elif r < 3.569:
                stability_str = "unstable"
                notes = f"λ = {eigenvalue:.3f}. Unstable (period-doubling cascade)."
            else:
                stability_str = "unstable"
                notes = f"λ = {eigenvalue:.3f}. Unstable (chaotic regime)."

            self.add_equilibrium(
                "nontrivial",
                x_eq=np.array([x_star]),
                u_eq=np.array([0.0]) if self._use_controlled else np.array([]),
                verify=True,
                stability=stability_str,
                eigenvalue=eigenvalue,
                notes=notes,
            )

            self.set_default_equilibrium("nontrivial")
        else:
            self.set_default_equilibrium("extinction")

    @property
    def r(self) -> float:
        """Growth rate parameter."""
        return self._r_val

    def find_fixed_points(self) -> List[Dict]:
        """
        Find all fixed points and their stability.

        Returns
        -------
        list
            List of dictionaries containing:
            - 'x': Fixed point value
            - 'eigenvalue': Stability eigenvalue
            - 'stable': Boolean stability flag

        Examples
        --------
        >>> system = LogisticMap(r=2.5)
        >>> fixed_points = system.find_fixed_points()
        >>> for fp in fixed_points:
        ...     print(f"x* = {fp['x']:.3f}, λ = {fp['eigenvalue']:.3f}, stable = {fp['stable']}")
        """
        r = self._r_val
        fixed_points = []

        # Extinction
        fixed_points.append({
            'x': 0.0,
            'eigenvalue': r,
            'stable': r < 1,
            'type': 'extinction'
        })

        # Nontrivial (if exists)
        if r > 1:
            x_star = 1.0 - 1.0 / r
            eigenvalue = 2.0 - r
            fixed_points.append({
                'x': x_star,
                'eigenvalue': eigenvalue,
                'stable': abs(eigenvalue) < 1,
                'type': 'nontrivial'
            })

        return fixed_points

    def compute_lyapunov_exponent(
        self,
        x0: float = 0.4,
        n_iterations: int = 10000,
        n_transient: int = 1000,
    ) -> float:
        """
        Compute Lyapunov exponent to quantify chaos.

        The Lyapunov exponent λ measures average exponential divergence rate:
            λ = lim (1/N)·Σ ln|df/dx|
               N→∞

        Parameters
        ----------
        x0 : float
            Initial condition
        n_iterations : int
            Number of iterations for averaging
        n_transient : int
            Number of initial iterations to discard

        Returns
        -------
        float
            Lyapunov exponent
            - λ < 0: Stable fixed point or periodic orbit
            - λ = 0: Neutral (bifurcation point)
            - λ > 0: Chaos

        Examples
        --------
        >>> system = LogisticMap(r=3.9)
        >>> lyapunov = system.compute_lyapunov_exponent(n_iterations=50000)
        >>> print(f"Lyapunov exponent: {lyapunov:.4f}")
        >>> if lyapunov > 0.01:
        ...     print("System is CHAOTIC")
        """
        r = self._r_val
        x = x0

        # Discard transient
        for _ in range(n_transient):
            x = r * x * (1 - x)

        # Compute sum of log derivatives
        lyap_sum = 0.0
        for _ in range(n_iterations):
            # Derivative: df/dx = r·(1 - 2x)
            derivative = r * (1 - 2 * x)
            lyap_sum += np.log(abs(derivative))

            # Iterate
            x = r * x * (1 - x)

        return lyap_sum / n_iterations

    def generate_bifurcation_diagram(
        self,
        r_values: np.ndarray,
        n_transient: int = 500,
        n_samples: int = 100,
        x0: float = 0.5,
    ) -> Dict[str, np.ndarray]:
        """
        Generate bifurcation diagram data.

        Parameters
        ----------
        r_values : np.ndarray
            Array of r values to test
        n_transient : int
            Number of iterations to discard (transient)
        n_samples : int
            Number of points to sample after transient
        x0 : float
            Initial condition for each r

        Returns
        -------
        dict
            Dictionary with keys:
            - 'r': Array of r values (repeated for each sample)
            - 'x': Array of x values at attractor

        Examples
        --------
        >>> system = LogisticMap()
        >>> r_vals = np.linspace(2.5, 4.0, 1000)
        >>> bifurc_data = system.generate_bifurcation_diagram(r_vals)
        >>> 
        >>> import matplotlib.pyplot as plt
        >>> plt.figure(figsize=(12, 8))
        >>> plt.plot(bifurc_data['r'], bifurc_data['x'], 'k.', markersize=0.2)
        >>> plt.xlabel('r')
        >>> plt.ylabel('x*')
        >>> plt.title('Bifurcation Diagram')
        >>> plt.show()
        """
        r_plot = []
        x_plot = []

        for r_val in r_values:
            x = x0

            # Discard transient
            for _ in range(n_transient):
                x = r_val * x * (1 - x)

            # Sample attractor
            for _ in range(n_samples):
                x = r_val * x * (1 - x)
                r_plot.append(r_val)
                x_plot.append(x)

        return {
            'r': np.array(r_plot),
            'x': np.array(x_plot)
        }

    def detect_period(
        self,
        trajectory: np.ndarray,
        max_period: int = 20,
        tolerance: float = 1e-6,
    ) -> Optional[int]:
        """
        Detect period of orbit from trajectory.

        Parameters
        ----------
        trajectory : np.ndarray
            Time series data
        max_period : int
            Maximum period to check
        tolerance : float
            Tolerance for periodicity

        Returns
        -------
        int or None
            Detected period, or None if aperiodic

        Examples
        --------
        >>> system = LogisticMap(r=3.2)
        >>> result = system.simulate(x0=np.array([0.5]), u_sequence=None, n_steps=200)
        >>> period = system.detect_period(result['states'][:, 0])
        >>> print(f"Detected period: {period}")
        """
        n = len(trajectory)

        for p in range(1, max_period + 1):
            if n < 2 * p:
                continue

            # Check if x[k+p] ≈ x[k] for last half of trajectory
            is_periodic = True
            for i in range(n // 2, n - p):
                if abs(trajectory[i + p] - trajectory[i]) > tolerance:
                    is_periodic = False
                    break

            if is_periodic:
                return p

        return None  # Aperiodic (chaotic or transient)

    def plot_cobweb(
        self,
        ax,
        x0: float = 0.1,
        n_iterations: int = 50,
    ):
        """
        Create cobweb diagram visualization.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes to plot on
        x0 : float
            Initial condition
        n_iterations : int
            Number of iterations to plot

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots(figsize=(8, 8))
        >>> system = LogisticMap(r=3.5)
        >>> system.plot_cobweb(ax, x0=0.1, n_iterations=50)
        >>> plt.show()
        """
        r = self._r_val

        # Plot function and diagonal
        x_range = np.linspace(0, 1, 1000)
        ax.plot(x_range, r * x_range * (1 - x_range), 'b-', linewidth=2, label=f'f(x) = {r}x(1-x)')
        ax.plot(x_range, x_range, 'k--', linewidth=1, label='y = x')

        # Cobweb iteration
        x = x0
        x_points = [x0]
        y_points = [0]

        for _ in range(n_iterations):
            x_new = r * x * (1 - x)

            # Vertical line to function
            x_points.extend([x, x])
            y_points.extend([x, x_new])

            # Horizontal line to diagonal
            x_points.extend([x, x_new])
            y_points.extend([x_new, x_new])

            x = x_new

        ax.plot(x_points, y_points, 'r-', linewidth=0.5, alpha=0.7)
        ax.plot(x0, 0, 'go', markersize=8, label=f'Start: x₀ = {x0}')

        ax.set_xlabel('x[k]', fontsize=12)
        ax.set_ylabel('x[k+1]', fontsize=12)
        ax.set_title(f'Cobweb Diagram (r = {r})', fontsize=14)
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')

    def compute_bifurcation_points(self) -> Dict[str, float]:
        """
        Compute key bifurcation points.

        Returns
        -------
        dict
            Dictionary of bifurcation points:
            - 'transcritical': r = 1 (extinction ↔ nontrivial)
            - 'period_2': r = 3 (fixed point → period-2)
            - 'period_4': r ≈ 3.449 (period-2 → period-4)
            - 'period_8': r ≈ 3.544 (period-4 → period-8)
            - 'chaos_onset': r ≈ 3.569 (accumulation point)
            - 'period_3_window': r ≈ 3.83 (period-3 window opens)

        Examples
        --------
        >>> system = LogisticMap()
        >>> bifurc_points = system.compute_bifurcation_points()
        >>> for name, r_val in bifurc_points.items():
        ...     print(f"{name}: r = {r_val:.4f}")
        """
        return {
            'transcritical': 1.0,
            'period_2': 3.0,
            'period_4': 1 + np.sqrt(6),  # ≈ 3.449
            'period_8': 3.54409,
            'chaos_onset': 3.56995,  # Accumulation point (Feigenbaum)
            'period_3_window': 3.828,
        }

    def compute_feigenbaum_delta(
        self,
        n_bifurcations: int = 10,
    ) -> float:
        """
        Estimate Feigenbaum delta constant from period-doubling sequence.

        δ = lim (rₙ - rₙ₋₁)/(rₙ₊₁ - rₙ) ≈ 4.669...
           n→∞

        Parameters
        ----------
        n_bifurcations : int
            Number of bifurcation points to use

        Returns
        -------
        float
            Estimated δ value

        Notes
        -----
        This is a simplified estimation. Accurate computation requires
        finding bifurcation points numerically to high precision.

        Examples
        --------
        >>> system = LogisticMap()
        >>> delta = system.compute_feigenbaum_delta()
        >>> print(f"Feigenbaum δ ≈ {delta:.3f} (theoretical: 4.669)")
        """
        # Known bifurcation points (approximate)
        r_bifurc = [
            3.0,        # Period-2
            3.44949,    # Period-4
            3.54409,    # Period-8
            3.56441,    # Period-16
            3.56875,    # Period-32
            3.56969,    # Period-64
        ]

        # Compute ratios
        deltas = []
        for i in range(len(r_bifurc) - 2):
            delta_n = (r_bifurc[i+1] - r_bifurc[i]) / (r_bifurc[i+2] - r_bifurc[i+1])
            deltas.append(delta_n)

        # Return average of last few ratios (should converge to 4.669...)
        return np.mean(deltas[-3:])

    def classify_regime(self) -> str:
        """
        Classify dynamical regime based on r value.

        Returns
        -------
        str
            Regime classification

        Examples
        --------
        >>> system = LogisticMap(r=2.8)
        >>> print(system.classify_regime())
        'stable_fixed_point'
        >>>
        >>> system = LogisticMap(r=3.9)
        >>> print(system.classify_regime())
        'chaos'
        """
        r = self._r_val

        if r < 1:
            return "extinction"
        elif r < 3:
            return "stable_fixed_point"
        elif r < 1 + np.sqrt(6):  # ≈ 3.449
            return "period_2"
        elif r < 3.544:
            return "period_4_to_8"
        elif r < 3.569:
            return "period_doubling_cascade"
        elif r < 3.828:
            return "chaos"
        elif r < 3.858:
            return "period_3_window"
        elif r <= 4.0:
            return "chaos_with_windows"
        else:
            return "escape_regime"

    # def print_equations(self, simplify: bool = True):
    #     """
    #     Print symbolic equations.

    #     Parameters
    #     ----------
    #     simplify : bool
    #         If True, simplify expressions before printing
    #     """
    #     print("=" * 70)
    #     print(f"{self.__class__.__name__} (Discrete-Time, dt={self.dt})")
    #     print("=" * 70)
    #     print("The Logistic Map: Classic chaotic system")
    #     print(f"\nGrowth Rate: r = {self._r_val}")

    #     # Classify regime
    #     regime = self.classify_regime()
    #     regime_names = {
    #         "extinction": "Extinction (r < 1)",
    #         "stable_fixed_point": "Stable Fixed Point (1 < r < 3)",
    #         "period_2": "Period-2 Oscillation (3 < r < 3.449)",
    #         "period_4_to_8": "Period-4 to Period-8 (3.449 < r < 3.544)",
    #         "period_doubling_cascade": "Period-Doubling Cascade (3.544 < r < 3.569)",
    #         "chaos": "Chaos (3.569 < r < 3.828)",
    #         "period_3_window": "Period-3 Window (3.828 < r < 3.858)",
    #         "chaos_with_windows": "Chaos with Periodic Windows (3.858 < r ≤ 4)",
    #         "escape_regime": "Escape Regime (r > 4)",
    #     }
    #     print(f"Regime: {regime_names.get(regime, 'Unknown')}")

    #     print(f"\nState: x ∈ [0, 1] (population fraction)")
    #     if self._use_controlled:
    #         print(f"Control: u (external forcing/harvesting)")
    #     else:
    #         print("Autonomous system (no control)")

    #     print("\nDynamics: x[k+1] = f(x[k])")
    #     expr = self._f_sym[0]
    #     expr_sub = self.substitute_parameters(expr)
    #     if simplify:
    #         expr_sub = sp.simplify(expr_sub)
    #     print(f"  x[k+1] = {expr_sub}")

    #     # Fixed points
    #     print("\nFixed Points:")
    #     fixed_points = self.find_fixed_points()
    #     for fp in fixed_points:
    #         stability_str = "Stable" if fp['stable'] else "Unstable"
    #         print(f"  x* = {fp['x']:.4f}, λ = {fp['eigenvalue']:.4f} ({stability_str})")

    #     # Lyapunov exponent (if chaotic regime)
    #     if self._r_val > 3.4:
    #         try:
    #             lyapunov = self.compute_lyapunov_exponent(n_iterations=5000)
    #             print(f"\nLyapunov Exponent: λ = {lyapunov:.4f}")
    #             if lyapunov > 0.01:
    #                 print("  → CHAOTIC (sensitive dependence on initial conditions)")
    #             elif lyapunov > -0.01:
    #                 print("  → Neutral (bifurcation point)")
    #             else:
    #                 print("  → Periodic or Fixed Point")
    #         except:
    #             pass

    #     # Key bifurcation points
    #     print("\nKey Bifurcation Points:")
    #     bifurc = self.compute_bifurcation_points()
    #     print(f"  Period-2 bifurcation: r = {bifurc['period_2']:.3f}")
    #     print(f"  Period-4 bifurcation: r = {bifurc['period_4']:.3f}")
    #     print(f"  Chaos onset: r = {bifurc['chaos_onset']:.3f}")
    #     print(f"  Period-3 window: r = {bifurc['period_3_window']:.3f}")

    #     # Feigenbaum constant
    #     print(f"\nFeigenbaum δ (universal constant): 4.669...")

    #     print("\nPhysical Interpretation:")
    #     print("  - Discrete-time population model with limited resources")
    #     print("  - r·x: Reproduction (proportional to population)")
    #     print("  - -r·x²: Competition/resource limitation")
    #     print("  - Simple equation → complex behavior (chaos)")

    #     print("\nApplications:")
    #     print("  - Population dynamics (insects, fish stocks)")
    #     print("  - Chaos theory pedagogy")
    #     print("  - Pseudo-random number generation")
    #     print("  - Bifurcation theory")
    #     print("  - Cryptography (chaotic encryption)")

    #     print("=" * 70)


# Alias for backward compatibility
DiscreteLogisticMap = LogisticMap