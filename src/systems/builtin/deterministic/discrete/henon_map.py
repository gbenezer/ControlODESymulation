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
Discrete-time Hénon Map - Canonical 2D Dissipative Chaos.

This module provides the Hénon map, the prototypical two-dimensional
discrete dynamical system exhibiting strange attractors. It represents:
- The simplest 2D map with chaotic dynamics
- A model for dissipative systems (energy-losing)
- A paradigm for understanding strange attractors
- The transition from periodic to chaotic behavior
- Fractal basin boundaries and riddled basins

The Hénon map serves as:
- The 2D analog of the logistic map (1D chaos)
- A tractable example of the Poincaré-Bendixson theorem
- A model for turbulence onset and fluid mixing
- A testbed for numerical chaos detection algorithms
- An illustration of homoclinic tangles and bifurcations

Introduced by Michel Hénon in 1976 as a simplified model of the Poincaré
section of the Lorenz system, the Hénon map has become one of the most
studied discrete dynamical systems in chaos theory.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import sympy as sp
from scipy.optimize import fsolve
from scipy.spatial import ConvexHull

from src.systems.base.core.discrete_symbolic_system import DiscreteSymbolicSystem


class HenonMap(DiscreteSymbolicSystem):
    """
    The Hénon Map: Paradigm of 2D dissipative chaos and strange attractors.

    Physical System:
    ---------------
    The Hénon map was introduced by Michel Hénon (1976) as a simplified
    model for the Poincaré section of the Lorenz equations. It represents
    a generic quadratic area-contracting map that captures essential features
    of dissipative chaos.

    **Mathematical Form:**
    The defining equations are remarkably simple:

        x[n+1] = 1 - a·x[n]² + y[n]
        y[n+1] = b·x[n]

    Despite this simplicity, the map exhibits:
    - Chaotic attractors (strange attractors)
    - Period-doubling cascades
    - Fractal basin boundaries
    - Sensitive dependence on initial conditions
    - Homoclinic bifurcations

    **Physical Interpretation:**
    While originally a mathematical abstraction, the Hénon map can model:
    1. **Poincaré sections** of 3D continuous flows
    2. **Population dynamics** with two age classes
    3. **Chemical reactions** with two coupled species
    4. **Economic models** with production and capital
    5. **Fluid mixing** and turbulence onset

    The key feature is DISSIPATION: phase space volume contracts,
    leading to attractors (unlike conservative Hamiltonian systems).

    **Dissipation and Attractors:**
    The Jacobian determinant is:
        det(J) = -b

    For |b| < 1, the map is area-contracting:
    - Phase space volume shrinks by factor |b| each iteration
    - Asymptotically, dynamics confined to attractor
    - Attractor has dimension < 2 (fractal)

    This contrasts with:
    - Logistic map (1D): Dimension 0 or 1 attractors
    - Standard map (conservative): No attractors, volume-preserving
    - Hénon map (2D dissipative): Dimension ~1.26 attractor (fractal!)

    State Space:
    -----------
    State: x[n] = [x[n], y[n]]
        First coordinate:
        - x: "Position" or primary variable
          * Unbounded in principle: -∞ < x < ∞
          * Typically confined to bounded region by attractor
          * For standard parameters: -1.5 < x < 1.5

        Second coordinate:
        - y: "Velocity" or secondary variable
          * Coupled to previous x value: y[n] = b·x[n-1]
          * Acts as delayed feedback
          * For standard parameters: -0.4 < y < 0.4

    **Phase Space Structure:**
    The (x, y) plane contains:
    - Attractors (where orbits converge)
    - Basins of attraction (initial conditions leading to each attractor)
    - Basin boundaries (separating different basins)
    - Saddle points and unstable manifolds

    Control: u[n] (optional, for controlled Hénon map)
        - Can represent:
          * Perturbation to x dynamics: x[n+1] = 1 - a·x[n]² + y[n] + u[n]
          * External forcing or feedback control
          * Stabilization of unstable periodic orbits
        - Standard map: u = 0 (autonomous)

    Output: y[n] = [x[n], y[n]]
        - Full state measurement (both coordinates)
        - In practice, may observe only x (partial observation)
        - Time-delay embedding can reconstruct attractor from x alone

    Dynamics:
    --------
    The Hénon map is defined by:

        x[n+1] = 1 - a·x[n]² + y[n]
        y[n+1] = b·x[n]

    **Key Mathematical Properties:**

    1. **Quadratic nonlinearity:**
       - The x² term creates folding in phase space
       - Enables period-doubling and chaos
       - Simplest polynomial map with strange attractor

    2. **Area contraction (dissipation):**
       Jacobian: J = | -2ax   1 |
                     |   b    0 |
       
       det(J) = -b (constant, independent of state!)
       
       For |b| < 1:
       - Area contracts by factor |b| per iteration
       - Volume of phase space → 0 as n → ∞
       - Dynamics attracted to lower-dimensional set

    3. **Invertibility:**
       The map has a unique inverse (for b ≠ 0):
       
       x[n] = y[n+1]/b
       y[n] = x[n+1] - 1 + a·(y[n+1]/b)²

       This allows backward iteration (unlike logistic map).

    4. **Time-reversal symmetry (special case):**
       For b = -1, map is time-reversible (area-preserving)
       This is the limit connecting to Hamiltonian systems.

    Parameters:
    ----------
    a : float, default=1.4
        Nonlinearity parameter (controls folding strength)
        
        **Physical Meaning:**
        - Controls strength of quadratic nonlinearity
        - Larger |a| → stronger folding
        - Determines attractor complexity

        **Parameter Regimes:**

        **a = 0:**
        - Linear map: x[n+1] = 1 + y[n], y[n+1] = b·x[n]
        - Simple dynamics, no chaos
        - All orbits converge or diverge simply

        **0 < a < 0.368:**
        - Stable fixed point
        - All trajectories converge to single point
        - No oscillations or complexity

        **a ≈ 0.368:**
        - Flip bifurcation (period-doubling)
        - Fixed point becomes unstable
        - Period-2 orbit appears

        **0.368 < a < 1.06:**
        - Periodic attractors (period-2, 4, 8, ...)
        - Period-doubling cascade
        - Feigenbaum scaling (universal constants)
        - Resembles logistic map behavior

        **a ≈ 1.06:**
        - Onset of chaos
        - Accumulation point of period-doubling cascade
        - Transition from periodic to chaotic

        **1.06 < a < 1.4:**
        - Chaotic attractor (strange attractor)
        - Positive Lyapunov exponent
        - Fractal structure
        - Sensitive dependence on initial conditions

        **a = 1.4 (canonical value):**
        - Classic Hénon attractor
        - Well-studied strange attractor
        - Fractal dimension ≈ 1.26
        - Hausdorff dimension ≈ 1.261
        - Correlation dimension ≈ 1.25

        **a > 1.4:**
        - Attractor may fragment or disappear
        - Escape to infinity possible
        - Multiple attractors may coexist
        - Complex basin boundaries

        **a >> 1:**
        - Most orbits escape to infinity
        - Bounded attractors rare
        - Chaotic scattering regime

    b : float, default=0.3
        Dissipation parameter (controls area contraction)

        **Physical Meaning:**
        - Controls phase space volume contraction rate
        - |det(J)| = |b| = volume contraction factor
        - Larger |b| → slower contraction, less dissipation
        - Smaller |b| → faster contraction, more dissipation

        **Special Values:**

        **b = 0:**
        - Extreme dissipation (collapse to 1D)
        - y coordinate decouples: y[n+1] = 0
        - Effectively becomes 1D logistic-like map
        - Loses 2D structure

        **0 < b < 1:**
        - Standard dissipative regime
        - Area-contracting (det(J) < 1)
        - Attractors exist (typical case)
        - b = 0.3 is canonical value

        **b = 1:**
        - Borderline case (area-preserving)
        - Conservative limit
        - No attractors (like Standard Map)
        - Unlikely for physical dissipative systems

        **b = -1:**
        - Area-preserving AND time-reversible
        - Special mathematical interest
        - Connects to Hamiltonian dynamics
        - Not typical for dissipative systems

        **-1 < b < 0:**
        - Orientation-reversing
        - Still dissipative (|b| < 1)
        - Can have strange attractors
        - Less commonly studied

        **|b| > 1:**
        - Area-expanding
        - No bounded attractors
        - Orbits typically escape
        - Not physical for dissipative systems

    dt : float, default=1.0
        Time step between iterations
        - Usually normalized to 1
        - Represents sampling period
        - Only affects interpretation, not dynamics

    Equilibria and Fixed Points:
    ----------------------------
    Fixed points satisfy:
        x* = 1 - a·x*² + y*
        y* = b·x*

    Substituting second into first:
        x* = 1 - a·x*² + b·x*
        a·x*² - (b-1)·x* + 1 = 0

    Solutions (for standard b = 0.3, a varies):
        x* = [(b-1) ± √((b-1)² - 4a)] / (2a)

    **Fixed points exist if:**
        (b-1)² ≥ 4a
        
    For b = 0.3:
        0.49 ≥ 4a
        a ≤ 0.1225

    For a > 0.1225, fixed points are complex (no real equilibria).

    **Typical scenario (a = 1.4, b = 0.3):**
    - No real fixed points
    - System has no equilibria
    - All dynamics are transient or on attractor
    - Attractor is only long-term behavior

    **Stability of fixed points:**
    When fixed points exist, linearization gives:
        λ₁, λ₂ = eigenvalues of Jacobian

    Fixed point is stable if |λ₁|, |λ₂| < 1.

    Strange Attractor:
    -----------------
    The Hénon attractor (a = 1.4, b = 0.3) is a STRANGE ATTRACTOR:

    **Definition:** A strange attractor is an attracting set that:
    1. Is an attractor (nearby trajectories approach it)
    2. Is strange (exhibits sensitive dependence on IC)
    3. Has fractal structure (non-integer dimension)

    **Properties of Hénon attractor:**
    - **Fractal dimension:** D ≈ 1.26 (between 1D and 2D)
    - **Correlation dimension:** d_c ≈ 1.25
    - **Lyapunov exponents:** λ₁ ≈ 0.42, λ₂ ≈ -1.62
    - **Lyapunov dimension:** D_L = 1 + λ₁/|λ₂| ≈ 1.26
    - **Information dimension:** Similar to fractal dimension

    **Visual structure:**
    - Appears as collection of parallel curves (bands)
    - Actually infinitely many curves (fractal layering)
    - Self-similar structure at all scales
    - Gaps between bands contain saddle points

    **Unstable manifolds:**
    - Curves connecting unstable fixed points/periodic orbits
    - Create complicated tangles (homoclinic tangles)
    - Fold back on themselves infinitely
    - Define skeleton of attractor

    Chaos Characterization:
    ----------------------
    **Lyapunov Exponents:**
    For canonical Hénon map (a = 1.4, b = 0.3):
        λ₁ ≈ 0.42 (positive → chaos)
        λ₂ ≈ -1.62 (negative → dissipation)
        λ₁ + λ₂ = ln|b| ≈ -1.20 (contraction rate)

    The positive Lyapunov exponent confirms:
    - Exponential divergence of nearby trajectories
    - Sensitive dependence on initial conditions
    - Chaotic dynamics (deterministic but unpredictable)

    **Basin of Attraction:**
    The set of initial conditions leading to the attractor.
    For Hénon map:
    - Basin is typically simply connected (single piece)
    - But can have fractal boundary
    - Some parameters → multiple attractors with intertwined basins

    **Periodic Orbits:**
    Embedded in attractor are infinitely many unstable periodic orbits:
    - Period-1, 2, 3, 4, ... orbits
    - All are unstable (saddles)
    - Dense in the attractor
    - Can be used for control (OGY method)

    Bifurcations and Route to Chaos:
    --------------------------------
    As 'a' increases (with b = 0.3 fixed), the Hénon map undergoes:

    **1. Fixed point stability loss (a ≈ 0.37):**
       - Flip bifurcation (period-doubling)
       - Stable fixed point → period-2 orbit

    **2. Period-doubling cascade (0.37 < a < 1.06):**
       - Period-2 → period-4 → period-8 → ...
       - Geometric convergence (Feigenbaum constants)
       - δ ≈ 4.669... (universal)
       - α ≈ 2.502... (universal)

    **3. Onset of chaos (a ≈ 1.06):**
       - Accumulation point of cascade
       - Lyapunov exponent becomes positive
       - Attractor dimension becomes fractal

    **4. Fully developed chaos (1.06 < a < 1.4):**
       - Strange attractor forms
       - Sensitive dependence maximizes
       - Periodic windows may appear

    **5. Attractor crisis (a > 1.4):**
       - Attractor may collide with unstable manifold
       - Sudden destruction or enlargement
       - Possible escape to infinity

    This sequence is the **period-doubling route to chaos**, one of
    three universal routes (along with intermittency and quasi-periodicity).

    Control Objectives:
    ------------------
    **1. Chaos Control (OGY Method):**
       Goal: Stabilize unstable periodic orbits embedded in attractor
       Method: Apply small parameter perturbations
       - Locate periodic orbit (e.g., period-1)
       - Compute local stable/unstable directions
       - Apply control when trajectory near orbit
       - Perturbation: u[n] = -K·(x[n] - x_orbit)

    **2. Chaos Synchronization:**
       Goal: Make two Hénon maps follow same trajectory
       Applications: Secure communication, pattern recognition
       Method: Couple systems or drive-response configuration

    **3. Targeting:**
       Goal: Steer trajectory to specific location on attractor
       Useful for optimization and search

    **4. Anti-control (Chaotification):**
       Goal: Make periodic system chaotic
       Applications: Mixing, encryption, avoiding resonance

    **5. Basin of attraction enlargement:**
       Goal: Expand region attracting to desired attractor
       Important when multiple attractors coexist

    Numerical Considerations:
    ------------------------
    **Attractor Visualization:**
    Standard method:
    1. Choose initial condition (often random)
    2. Discard first ~1000 iterations (transient)
    3. Plot next 10,000+ points
    4. Result shows attractor structure

    **Transient Behavior:**
    - First iterations may not reflect attractor
    - Discard typically 100-1000 initial points
    - Longer transients near bifurcations

    **Numerical Precision:**
    - Double precision (64-bit) sufficient
    - Chaos amplifies roundoff errors exponentially
    - Long-term trajectories unreliable (>1000 iterations)
    - Statistical properties converge faster than trajectories

    **Basin of Attraction Computation:**
    - Grid initial conditions in (x, y) plane
    - Iterate each until convergence or escape
    - Color-code by attractor reached
    - Reveals basin structure and boundaries

    **Dimension Estimation:**
    Several algorithms available:
    - Box-counting dimension (covers attractor with boxes)
    - Correlation dimension (uses point correlations)
    - Lyapunov dimension (from Lyapunov exponents)
    - Information dimension (uses entropies)

    All give similar values (~1.26) for canonical Hénon attractor.

    Example Usage:
    -------------
    >>> # Create canonical Hénon map
    >>> system = HenonMap(a=1.4, b=0.3)
    >>> 
    >>> # Visualize strange attractor
    >>> x0 = np.array([0.1, 0.1])
    >>> result = system.simulate(
    ...     x0=x0,
    ...     u_sequence=None,
    ...     n_steps=10000
    ... )
    >>> 
    >>> # Plot attractor (discard transient)
    >>> import plotly.graph_objects as go
    >>> states_attractor = result['states'][1000:, :]  # Skip first 1000
    >>> 
    >>> fig = go.Figure()
    >>> fig.add_trace(go.Scatter(
    ...     x=states_attractor[:, 0],
    ...     y=states_attractor[:, 1],
    ...     mode='markers',
    ...     marker=dict(size=1, color='blue', opacity=0.5),
    ...     name='Hénon Attractor'
    ... ))
    >>> 
    >>> fig.update_layout(
    ...     title=f'Hénon Attractor (a={system.a}, b={system.b})',
    ...     xaxis_title='x',
    ...     yaxis_title='y',
    ...     width=800,
    ...     height=600,
    ...     plot_bgcolor='white'
    ... )
    >>> fig.show()
    >>> 
    >>> # Compute Lyapunov exponents
    >>> lyap_exp = system.compute_lyapunov_exponents(
    ...     x0=np.array([0.1, 0.1]),
    ...     n_iterations=50000
    ... )
    >>> print(f"Lyapunov exponents: λ₁ = {lyap_exp[0]:.4f}, λ₂ = {lyap_exp[1]:.4f}")
    >>> print(f"Sum: λ₁ + λ₂ = {sum(lyap_exp):.4f} (should ≈ ln|b| = {np.log(abs(system.b)):.4f})")
    >>> 
    >>> if lyap_exp[0] > 0:
    ...     print("System is CHAOTIC")
    ...     # Estimate attractor dimension
    ...     D_L = 1 + lyap_exp[0] / abs(lyap_exp[1])
    ...     print(f"Lyapunov dimension: D_L ≈ {D_L:.3f}")
    >>> 
    >>> # Sensitive dependence on initial conditions
    >>> x0_a = np.array([0.1, 0.1])
    >>> x0_b = np.array([0.1 + 1e-10, 0.1])  # Tiny difference
    >>> 
    >>> result_a = system.simulate(x0_a, None, n_steps=50)
    >>> result_b = system.simulate(x0_b, None, n_steps=50)
    >>> 
    >>> divergence = np.linalg.norm(
    ...     result_a['states'] - result_b['states'],
    ...     axis=1
    ... )
    >>> 
    >>> fig = go.Figure()
    >>> fig.add_trace(go.Scatter(
    ...     x=np.arange(51),
    ...     y=divergence,
    ...     mode='lines+markers',
    ...     name='Distance'
    ... ))
    >>> fig.update_yaxes(type='log')
    >>> fig.update_layout(
    ...     title='Exponential Divergence (Sensitive Dependence)',
    ...     xaxis_title='Iteration n',
    ...     yaxis_title='||x_a - x_b||',
    ... )
    >>> fig.show()
    >>> 
    >>> # Generate bifurcation diagram (varying 'a')
    >>> bifurcation_data = system.generate_bifurcation_diagram(
    ...     parameter='a',
    ...     param_range=(0.8, 1.5),
    ...     n_points=500,
    ...     n_transient=500,
    ...     n_samples=200
    ... )
    >>> 
    >>> fig = go.Figure()
    >>> fig.add_trace(go.Scatter(
    ...     x=bifurcation_data['param'],
    ...     y=bifurcation_data['x'],
    ...     mode='markers',
    ...     marker=dict(size=0.5, color='black'),
    ...     name='Bifurcation'
    ... ))
    >>> fig.update_layout(
    ...     title='Hénon Map Bifurcation Diagram',
    ...     xaxis_title='a (nonlinearity parameter)',
    ...     yaxis_title='x*',
    ...     width=1000,
    ...     height=600
    ... )
    >>> fig.show()
    >>> 
    >>> # Explore basin of attraction
    >>> basin_data = system.compute_basin_of_attraction(
    ...     x_range=(-2, 2),
    ...     y_range=(-2, 2),
    ...     resolution=400,
    ...     max_iterations=100
    ... )
    >>> 
    >>> fig = go.Figure(data=go.Heatmap(
    ...     z=basin_data['converged'],
    ...     x=basin_data['x_grid'][0, :],
    ...     y=basin_data['y_grid'][:, 0],
    ...     colorscale='RdBu',
    ...     showscale=True
    ... ))
    >>> fig.update_layout(
    ...     title='Basin of Attraction (blue=converges, red=escapes)',
    ...     xaxis_title='x₀',
    ...     yaxis_title='y₀',
    ...     width=800,
    ...     height=800
    ... )
    >>> fig.show()
    >>> 
    >>> # Compare multiple parameter values
    >>> from plotly.subplots import make_subplots
    >>> 
    >>> a_values = [0.9, 1.1, 1.3, 1.4]
    >>> fig = make_subplots(
    ...     rows=2, cols=2,
    ...     subplot_titles=[f'a = {a}' for a in a_values]
    ... )
    >>> 
    >>> for idx, a_val in enumerate(a_values):
    ...     row = idx // 2 + 1
    ...     col = idx % 2 + 1
    ...     
    ...     sys_temp = HenonMap(a=a_val, b=0.3)
    ...     res_temp = sys_temp.simulate(
    ...         x0=np.array([0.1, 0.1]),
    ...         u_sequence=None,
    ...         n_steps=5000
    ...     )
    ...     
    ...     states_plot = res_temp['states'][1000:, :]
    ...     
    ...     fig.add_trace(
    ...         go.Scatter(
    ...             x=states_plot[:, 0],
    ...             y=states_plot[:, 1],
    ...             mode='markers',
    ...             marker=dict(size=0.5),
    ...             showlegend=False
    ...         ),
    ...         row=row, col=col
    ...     )
    >>> 
    >>> fig.update_xaxes(title_text='x')
    >>> fig.update_yaxes(title_text='y')
    >>> fig.update_layout(
    ...     title_text='Hénon Map: Transition to Chaos',
    ...     height=800,
    ...     width=1000
    ... )
    >>> fig.show()
    >>> 
    >>> # Estimate correlation dimension
    >>> corr_dim = system.estimate_correlation_dimension(
    ...     x0=np.array([0.1, 0.1]),
    ...     n_points=5000,
    ...     n_transient=1000
    ... )
    >>> print(f"Correlation dimension: d_c ≈ {corr_dim:.3f}")
    >>> print(f"(Theoretical for Hénon attractor: ≈ 1.25)")

    Physical Insights:
    -----------------
    **Why 2D is Different from 1D:**
    - 1D maps (logistic): Attractors are points or periodic cycles
    - 2D maps (Hénon): Attractors can be STRANGE (fractal curves)
    - Dimension 1 < D < 2 possible (impossible in 1D!)
    - Richer dynamics: homoclinic tangles, cantori, etc.

    **Dissipation Creates Attractors:**
    - Area contraction (det(J) = -b with |b| < 1) crucial
    - Without dissipation: no attractors (like Standard Map)
    - Dissipation "squeezes" phase space onto lower-dimensional set
    - Competition: folding (nonlinearity) vs contraction (dissipation)

    **Folding Mechanism:**
    The x² term creates quadratic folding:
    - Phase space is stretched along one direction
    - Then folded back on itself
    - Repeated stretching + folding → fractal structure
    - This is the "baker's transformation" in disguise

    **Connection to Continuous Systems:**
    Hénon map approximates Poincaré section of:
    - Lorenz system (chaotic convection)
    - Duffing oscillator (forced nonlinear pendulum)
    - Other 3D continuous flows with strange attractors

    Discrete map captures essential features while being:
    - Easier to compute (no integration)
    - Easier to analyze (algebraic not differential)
    - Still exhibiting all chaos phenomena

    **Universality:**
    The period-doubling cascade in Hénon map obeys same scaling laws
    (Feigenbaum constants) as logistic map, forced pendulum, and
    countless other systems. This is UNIVERSAL - a profound discovery
    showing chaos theory has quantitative predictions!

    **Fractal Basin Boundaries:**
    When multiple attractors coexist, basin boundaries can be fractal:
    - Impossibly complicated boundary structure
    - Arbitrarily close ICs can lead to different attractors
    - "Riddled basins" - attractor's basin has holes everywhere
    - Final-state uncertainty for practical systems

    Common Pitfalls:
    ---------------
    1. **Not discarding transient:**
       First hundreds of iterations may not show attractor
       Always skip initial points before plotting

    2. **Wrong parameter values:**
       Using a > 1.5 or b > 1 can cause escape to infinity
       Canonical values (a=1.4, b=0.3) are well-behaved

    3. **Insufficient points:**
       Need thousands of points to see attractor structure
       Fractal detail requires high point density

    4. **Confusing with Hamiltonian systems:**
       Hénon map is DISSIPATIVE (area-contracting)
       Has attractors (unlike Standard Map)
       Lyapunov exponents sum to negative value

    5. **Expecting simple fixed points:**
       For typical parameters, no real fixed points exist
       All dynamics are transient or on attractor
       Cannot linearize around equilibrium (none exists!)

    6. **Ignoring basin of attraction:**
       Not all ICs converge to same attractor
       Some ICs escape to infinity
       Basin can have complex structure

    Extensions and Variations:
    -------------------------
    1. **Generalized Hénon map:**
       x[n+1] = c - a·x[n]^p + y[n]
       y[n+1] = b·x[n]
       Different exponents p change dynamics

    2. **Lozi map:**
       x[n+1] = 1 - a·|x[n]| + y[n]
       y[n+1] = b·x[n]
       Piecewise-linear version (easier analysis)

    3. **Coupled Hénon maps:**
       Network of interacting Hénon maps
       Models spatiotemporal chaos

    4. **3D Hénon map:**
       Extension to three dimensions
       Hyperchaos (multiple positive Lyapunov exponents)

    5. **Delayed Hénon map:**
       Include time delays in coupling
       Richer bifurcation structure

    6. **Noisy Hénon map:**
       Add stochastic perturbations
       Studies chaos-noise interaction

    See Also:
    --------
    LogisticMap : 1D chaotic map
    StandardMap : 2D conservative (Hamiltonian) chaos
    Lorenz : Continuous 3D chaos (Hénon as Poincaré section)
    """

    def define_system(
        self,
        a: float = 1.4,
        b: float = 0.3,
        dt: float = 1.0,
        use_controlled_version: bool = False,
    ):
        """
        Define Hénon map dynamics.

        Parameters
        ----------
        a : float
            Nonlinearity parameter (controls folding)
        b : float
            Dissipation parameter (controls area contraction)
        dt : float
            Time step between iterations (usually 1.0)
        use_controlled_version : bool
            If True, adds control input u[n] to x dynamics
        """
        # Store parameters
        self._a_val = a
        self._b_val = b
        self._use_controlled = use_controlled_version

        # Validate parameters
        if abs(b) > 1:
            import warnings
            warnings.warn(
                f"Dissipation parameter |b| = {abs(b)} > 1 causes area expansion. "
                "Orbits may escape to infinity. Typical range: |b| < 1.",
                UserWarning
            )

        if a > 1.5:
            import warnings
            warnings.warn(
                f"Nonlinearity parameter a = {a} > 1.5 may cause escape to infinity. "
                "Canonical value is a = 1.4.",
                UserWarning
            )

        # State variables
        x, y = sp.symbols('x y', real=True)
        
        # Symbolic parameters
        a_sym, b_sym = sp.symbols('a b', real=True)

        self.state_vars = [x, y]
        self.parameters = {a_sym: a, b_sym: b}
        self._dt = dt
        self.order = 1

        if use_controlled_version:
            # Controlled Hénon map
            u = sp.symbols('u', real=True)
            self.control_vars = [u]
            x_next = 1 - a_sym * x**2 + y + u
        else:
            # Standard autonomous map
            self.control_vars = []
            x_next = 1 - a_sym * x**2 + y

        y_next = b_sym * x

        self._f_sym = sp.Matrix([x_next, y_next])

    def setup_equilibria(self):
        """
        Set up fixed points if they exist.

        Fixed points satisfy:
            x* = 1 - a·x*² + y*
            y* = b·x*

        Substituting: a·x*² - (b-1)·x* + 1 = 0

        Real solutions exist only if (b-1)² ≥ 4a.
        """
        a = self._a_val
        b = self._b_val

        # Check if real fixed points exist
        discriminant = (b - 1)**2 - 4*a

        if discriminant >= 0:
            # Real fixed points exist
            x1 = ((b - 1) + np.sqrt(discriminant)) / (2*a)
            x2 = ((b - 1) - np.sqrt(discriminant)) / (2*a)

            y1 = b * x1
            y2 = b * x2

            # Add both fixed points
            for idx, (x_fp, y_fp) in enumerate([(x1, y1), (x2, y2)], 1):
                # Compute eigenvalues for stability
                J = self.compute_jacobian(np.array([x_fp, y_fp]))
                eigenvalues = np.linalg.eigvals(J)
                is_stable = np.all(np.abs(eigenvalues) < 1)

                self.add_equilibrium(
                    f"fixed_point_{idx}",
                    x_eq=np.array([x_fp, y_fp]),
                    u_eq=np.array([0.0]) if self._use_controlled else np.array([]),
                    verify=True,
                    stability="stable" if is_stable else "unstable",
                    eigenvalues=eigenvalues,
                    notes=f"Fixed point {idx}: λ = {eigenvalues}"
                )

            self.set_default_equilibrium("fixed_point_1")
        else:
            # No real fixed points - add reference for canonical attractor
            self.add_equilibrium(
                "attractor_reference",
                x_eq=np.array([0.0, 0.0]),
                u_eq=np.array([0.0]) if self._use_controlled else np.array([]),
                verify=False,
                notes=f"No real fixed points for a={a}, b={b}. "
                      "Dynamics converge to strange attractor."
            )
            self.set_default_equilibrium("attractor_reference")

    @property
    def a(self) -> float:
        """Nonlinearity parameter."""
        return self._a_val

    @property
    def b(self) -> float:
        """Dissipation parameter."""
        return self._b_val

    @property
    def area_contraction_rate(self) -> float:
        """Phase space area contraction rate (|det(J)| = |b|)."""
        return abs(self._b_val)

    def compute_jacobian(
        self,
        x: np.ndarray,
    ) -> np.ndarray:
        """
        Compute Jacobian matrix at state x.

        For Hénon map:
            J = | -2ax    1 |
                |   b     0 |

        Parameters
        ----------
        x : np.ndarray
            State [x, y]

        Returns
        -------
        np.ndarray
            Jacobian matrix (2×2)

        Notes
        -----
        Determinant is constant: det(J) = -b (independent of state!)
        This gives constant area contraction rate.
        """
        x_val = x[0]
        a = self._a_val
        b = self._b_val

        J = np.array([
            [-2*a*x_val, 1],
            [b, 0]
        ])

        return J

    def compute_lyapunov_exponents(
        self,
        x0: np.ndarray,
        n_iterations: int = 50000,
        n_transient: int = 1000,
    ) -> np.ndarray:
        """
        Compute both Lyapunov exponents using QR decomposition method.

        Parameters
        ----------
        x0 : np.ndarray
            Initial state [x₀, y₀]
        n_iterations : int
            Number of iterations for averaging
        n_transient : int
            Number of initial iterations to discard

        Returns
        -------
        np.ndarray
            Array of two Lyapunov exponents [λ₁, λ₂]
            λ₁: Largest (expansion rate)
            λ₂: Smallest (contraction rate)

        Notes
        -----
        For dissipative systems: λ₁ + λ₂ = ln|det(J)| = ln|b|
        
        For canonical Hénon (a=1.4, b=0.3):
            λ₁ ≈ 0.42 (positive → chaos)
            λ₂ ≈ -1.62 (negative → dissipation)
            λ₁ + λ₂ ≈ -1.20 ≈ ln(0.3)

        Examples
        --------
        >>> system = HenonMap(a=1.4, b=0.3)
        >>> lyap = system.compute_lyapunov_exponents(
        ...     x0=np.array([0.1, 0.1]),
        ...     n_iterations=100000
        ... )
        >>> print(f"λ₁ = {lyap[0]:.4f}, λ₂ = {lyap[1]:.4f}")
        >>> print(f"Sum = {sum(lyap):.4f} (should be ln|b| = {np.log(abs(system.b)):.4f})")
        """
        x = x0.copy()

        # Discard transient
        for _ in range(n_transient):
            x = self.step(x)

        # Initialize orthonormal basis
        Q = np.eye(2)
        lyap_sum = np.zeros(2)

        for _ in range(n_iterations):
            # Compute Jacobian at current point
            J = self.compute_jacobian(x)

            # Evolve tangent vectors
            Q = J @ Q

            # QR decomposition to orthonormalize
            Q, R = np.linalg.qr(Q)

            # Accumulate log of diagonal elements (growth rates)
            lyap_sum += np.log(np.abs(np.diag(R)))

            # Evolve state
            x = self.step(x)

        return lyap_sum / n_iterations

    def generate_bifurcation_diagram(
        self,
        parameter: str = 'a',
        param_range: Tuple[float, float] = (0.8, 1.5),
        n_points: int = 500,
        n_transient: int = 500,
        n_samples: int = 200,
        x0: np.ndarray = None,
    ) -> Dict:
        """
        Generate bifurcation diagram by varying a parameter.

        Parameters
        ----------
        parameter : str
            Parameter to vary ('a' or 'b')
        param_range : tuple
            Range of parameter values (min, max)
        n_points : int
            Number of parameter values to test
        n_transient : int
            Iterations to discard (transient)
        n_samples : int
            Points to sample after transient
        x0 : Optional[np.ndarray]
            Initial condition (None = use [0.1, 0.1])

        Returns
        -------
        dict
            Dictionary with:
            - 'param': Parameter values
            - 'x': x-coordinate values
            - 'y': y-coordinate values (optional)

        Examples
        --------
        >>> system = HenonMap()
        >>> bifurc_data = system.generate_bifurcation_diagram(
        ...     parameter='a',
        ...     param_range=(0.8, 1.5),
        ...     n_points=1000
        ... )
        >>> 
        >>> import plotly.graph_objects as go
        >>> fig = go.Figure()
        >>> fig.add_trace(go.Scatter(
        ...     x=bifurc_data['param'],
        ...     y=bifurc_data['x'],
        ...     mode='markers',
        ...     marker=dict(size=0.3, color='black')
        ... ))
        >>> fig.update_layout(title='Hénon Bifurcation Diagram')
        >>> fig.show()
        """
        if x0 is None:
            x0 = np.array([0.1, 0.1])

        param_values = np.linspace(param_range[0], param_range[1], n_points)
        param_plot = []
        x_plot = []
        y_plot = []

        for param_val in param_values:
            # Create temporary system with modified parameter
            if parameter == 'a':
                sys_temp = HenonMap(a=param_val, b=self._b_val)
            elif parameter == 'b':
                sys_temp = HenonMap(a=self._a_val, b=param_val)
            else:
                raise ValueError(f"Unknown parameter '{parameter}'. Use 'a' or 'b'.")

            x = x0.copy()

            # Discard transient
            for _ in range(n_transient):
                try:
                    x = sys_temp.step(x)
                    # Check for escape
                    if np.any(np.abs(x) > 1e6):
                        break
                except:
                    break

            # Sample attractor
            for _ in range(n_samples):
                try:
                    x = sys_temp.step(x)
                    if np.all(np.abs(x) < 1e6):  # Not escaped
                        param_plot.append(param_val)
                        x_plot.append(x[0])
                        y_plot.append(x[1])
                except:
                    break

        return {
            'param': np.array(param_plot),
            'x': np.array(x_plot),
            'y': np.array(y_plot),
        }

    def compute_basin_of_attraction(
        self,
        x_range: Tuple[float, float] = (-2, 2),
        y_range: Tuple[float, float] = (-2, 2),
        resolution: int = 400,
        max_iterations: int = 100,
        escape_radius: float = 1e3,
    ) -> Dict:
        """
        Compute basin of attraction.

        Grids initial conditions and determines which converge to attractor
        vs escape to infinity.

        Parameters
        ----------
        x_range : tuple
            Range of x initial conditions
        y_range : tuple
            Range of y initial conditions
        resolution : int
            Grid resolution (points per dimension)
        max_iterations : int
            Iterations before declaring convergence/escape
        escape_radius : float
            Radius beyond which orbit considered escaped

        Returns
        -------
        dict
            Dictionary with:
            - 'x_grid': Meshgrid of x values
            - 'y_grid': Meshgrid of y values
            - 'converged': Boolean array (True=converged, False=escaped)

        Examples
        --------
        >>> system = HenonMap(a=1.4, b=0.3)
        >>> basin = system.compute_basin_of_attraction(resolution=300)
        >>> 
        >>> import plotly.graph_objects as go
        >>> fig = go.Figure(data=go.Heatmap(
        ...     z=basin['converged'].astype(int),
        ...     x=basin['x_grid'][0, :],
        ...     y=basin['y_grid'][:, 0],
        ...     colorscale='RdBu'
        ... ))
        >>> fig.show()
        """
        x_vals = np.linspace(x_range[0], x_range[1], resolution)
        y_vals = np.linspace(y_range[0], y_range[1], resolution)
        x_grid, y_grid = np.meshgrid(x_vals, y_vals)

        converged = np.zeros((resolution, resolution), dtype=bool)

        for i in range(resolution):
            for j in range(resolution):
                x0 = np.array([x_grid[i, j], y_grid[i, j]])
                x = x0.copy()

                escaped = False
                for _ in range(max_iterations):
                    x = self.step(x)
                    if np.linalg.norm(x) > escape_radius:
                        escaped = True
                        break

                converged[i, j] = not escaped

        return {
            'x_grid': x_grid,
            'y_grid': y_grid,
            'converged': converged,
        }

    def estimate_correlation_dimension(
        self,
        x0: np.ndarray,
        n_points: int = 5000,
        n_transient: int = 1000,
        r_min: float = 0.001,
        r_max: float = 1.0,
        n_radii: int = 50,
    ) -> float:
        """
        Estimate correlation dimension using Grassberger-Procaccia algorithm.

        The correlation dimension d_c characterizes the fractal structure
        of the attractor.

        Parameters
        ----------
        x0 : np.ndarray
            Initial state
        n_points : int
            Number of points on attractor to use
        n_transient : int
            Points to discard as transient
        r_min : float
            Minimum radius for correlation sum
        r_max : float
            Maximum radius
        n_radii : int
            Number of radii to test

        Returns
        -------
        float
            Estimated correlation dimension

        Notes
        -----
        For canonical Hénon attractor: d_c ≈ 1.25

        Examples
        --------
        >>> system = HenonMap(a=1.4, b=0.3)
        >>> d_c = system.estimate_correlation_dimension(
        ...     x0=np.array([0.1, 0.1]),
        ...     n_points=10000
        ... )
        >>> print(f"Correlation dimension: {d_c:.3f}")
        """
        # Generate points on attractor
        result = self.simulate(
            x0=x0,
            u_sequence=None,
            n_steps=n_transient + n_points
        )
        points = result['states'][n_transient:, :]

        # Compute correlation sums
        radii = np.logspace(np.log10(r_min), np.log10(r_max), n_radii)
        C_r = np.zeros(n_radii)

        for i, r in enumerate(radii):
            count = 0
            for j in range(n_points):
                for k in range(j+1, n_points):
                    if np.linalg.norm(points[j] - points[k]) < r:
                        count += 1
            C_r[i] = 2 * count / (n_points * (n_points - 1))

        # Estimate dimension from slope of log(C_r) vs log(r)
        # In scaling region: C_r ~ r^d_c, so log(C_r) = d_c * log(r) + const
        
        # Use middle range for fit (avoid edge effects)
        idx_start = n_radii // 4
        idx_end = 3 * n_radii // 4
        
        log_r = np.log(radii[idx_start:idx_end])
        log_C = np.log(C_r[idx_start:idx_end] + 1e-10)  # Avoid log(0)

        # Linear fit
        d_c = np.polyfit(log_r, log_C, deg=1)[0]

        return d_c

    def classify_regime(self) -> str:
        """
        Classify dynamical regime based on parameters.

        Returns
        -------
        str
            Regime description

        Examples
        --------
        >>> system = HenonMap(a=1.4, b=0.3)
        >>> print(system.classify_regime())
        'strange_attractor'
        """
        a = self._a_val
        b = self._b_val

        # Check if fixed points exist
        discriminant = (b - 1)**2 - 4*a

        if discriminant >= 0:
            if a < 0.37:
                return "stable_fixed_point"
            elif a < 1.06:
                return "periodic_attractors"
            else:
                return "chaotic_with_fixed_points"
        else:
            if a < 1.06:
                return "periodic_no_fixed_points"
            elif 1.06 <= a <= 1.4:
                return "strange_attractor"
            else:
                return "complex_dynamics_or_escape"

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
    #     print("Hénon Map: Paradigm of 2D Dissipative Chaos")
    #     print(f"\nParameters: a = {self._a_val}, b = {self._b_val}")

    #     # Classify regime
    #     regime = self.classify_regime()
    #     regime_names = {
    #         "stable_fixed_point": "Stable Fixed Point",
    #         "periodic_attractors": "Periodic Attractors (Period-Doubling)",
    #         "chaotic_with_fixed_points": "Chaotic with Fixed Points",
    #         "periodic_no_fixed_points": "Periodic (No Fixed Points)",
    #         "strange_attractor": "Strange Attractor (Canonical)",
    #         "complex_dynamics_or_escape": "Complex Dynamics / Possible Escape",
    #     }
    #     print(f"Regime: {regime_names.get(regime, 'Unknown')}")

    #     print(f"\nState: x = [x, y]")
    #     if self._use_controlled:
    #         print(f"Control: u (additive control)")
    #     else:
    #         print("Autonomous system (no control)")

    #     print("\nDynamics: Hénon Map equations")
    #     print("  x[n+1] = 1 - a·x[n]² + y[n]")
    #     print("  y[n+1] = b·x[n]")

    #     # Show actual symbolic expressions
    #     print("\nSymbolic Dynamics:")
    #     for i, (var, expr) in enumerate(zip(self.state_vars, self._f_sym)):
    #         expr_sub = self.substitute_parameters(expr)
    #         if simplify:
    #             expr_sub = sp.simplify(expr_sub)
    #         label = ['x[n+1]', 'y[n+1]'][i]
    #         print(f"  {label} = {expr_sub}")

    #     # Jacobian
    #     print("\nJacobian Matrix:")
    #     print("  J = | -2ax    1 |")
    #     print("      |   b     0 |")
    #     print(f"  det(J) = -b = {-self._b_val} (constant!)")
    #     print(f"  Area contraction rate: |det(J)| = {abs(self._b_val)}")

    #     # Fixed points
    #     discriminant = (self._b_val - 1)**2 - 4*self._a_val
    #     print("\nFixed Points:")
    #     if discriminant >= 0:
    #         x1 = ((self._b_val - 1) + np.sqrt(discriminant)) / (2*self._a_val)
    #         x2 = ((self._b_val - 1) - np.sqrt(discriminant)) / (2*self._a_val)
    #         print(f"  x₁* = {x1:.4f}, y₁* = {self._b_val * x1:.4f}")
    #         print(f"  x₂* = {x2:.4f}, y₂* = {self._b_val * x2:.4f}")
    #     else:
    #         print(f"  No real fixed points (discriminant = {discriminant:.4f} < 0)")
    #         print(f"  Dynamics occur on strange attractor")

    #     # Lyapunov exponents (estimate for canonical case)
    #     if abs(self._a_val - 1.4) < 0.1 and abs(self._b_val - 0.3) < 0.1:
    #         print("\nLyapunov Exponents (canonical Hénon a=1.4, b=0.3):")
    #         print("  λ₁ ≈ 0.42 (positive → chaos)")
    #         print("  λ₂ ≈ -1.62 (negative → dissipation)")
    #         print(f"  λ₁ + λ₂ ≈ -1.20 ≈ ln|b| = {np.log(abs(self._b_val)):.2f}")
    #         print("\nFractal Dimension:")
    #         print("  Lyapunov dimension: D_L ≈ 1.26")
    #         print("  Correlation dimension: d_c ≈ 1.25")

    #     print("\nPhysical Interpretation:")
    #     print("  - x: Primary dynamical variable")
    #     print("  - y: Delayed feedback (y[n] = b·x[n-1])")
    #     print("  - Quadratic folding (x²) creates chaos")
    #     print("  - Area contraction (|b| < 1) creates attractor")

    #     print("\nKey Features:")
    #     print("  - Strange attractor (fractal dimension ~1.26)")
    #     print("  - Sensitive dependence on initial conditions")
    #     print("  - Period-doubling route to chaos")
    #     print("  - Dissipative dynamics (area-contracting)")
    #     print("  - Invertible map (can iterate backward)")

    #     print("\nApplications:")
    #     print("  - Model for Poincaré sections of chaotic flows")
    #     print("  - Population dynamics with age structure")
    #     print("  - Economic models (production-capital dynamics)")
    #     print("  - Turbulence and fluid mixing")
    #     print("  - Benchmark for chaos detection algorithms")

    #     print("=" * 70)


# Alias for backward compatibility
Henon = HenonMap
HenonAttractor = HenonMap