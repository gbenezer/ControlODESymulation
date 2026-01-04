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
Discrete-time Standard Map (Chirikov-Taylor Map) - Hamiltonian Chaos.

This module provides the Standard Map, also known as the Chirikov-Taylor map,
which is the paradigmatic example of Hamiltonian chaos. It describes:
- Kicked rotor dynamics (pendulum with periodic impulses)
- Area-preserving (symplectic) discrete dynamics
- Transition from regular to chaotic motion
- KAM (Kolmogorov-Arnold-Moser) theorem visualization
- Universal route to chaos in Hamiltonian systems

The Standard Map serves as:
- The "hydrogen atom" of chaos theory for Hamiltonian systems
- A model for particle accelerators and beam dynamics
- An example of non-integrable classical mechanics
- A testbed for KAM theory and resonance overlap
- A demonstration of mixed phase space (chaos + regularity coexisting)

Originally studied by Boris Chirikov (1969) and later by Joseph Ford,
the Standard Map has become fundamental in understanding the transition
from regular to chaotic motion in conservative (energy-preserving) systems.

Applications span from celestial mechanics to plasma physics to quantum chaos.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import sympy as sp
from scipy.optimize import fsolve

from src.systems.base.core.discrete_symbolic_system import DiscreteSymbolicSystem


class StandardMap(DiscreteSymbolicSystem):
    """
    The Standard Map (Chirikov-Taylor Map): Paradigm of Hamiltonian chaos.

    Physical System:
    ---------------
    The Standard Map models a kicked rotor: a pendulum that receives periodic
    impulses (kicks). Between kicks, the pendulum rotates freely; at each kick,
    it receives an angular momentum impulse proportional to sin(θ).

    **Physical Setup:**
    - Rotor with moment of inertia I (normalized to 1)
    - Free rotation between kicks (no gravity, no friction)
    - Periodic kicks at times t = n·τ (τ = kick period)
    - Kick strength K controls perturbation amplitude

    **Hamiltonian Description:**
    Between kicks (continuous time):
        H₀ = p²/2  (free rotation)
        θ̇ = ∂H₀/∂p = p
        ṗ = -∂H₀/∂θ = 0

    At kicks (instantaneous):
        H_kick = -K·cos(θ)·Σδ(t - nτ)
        Δp = K·sin(θ)

    **The Map:**
    Combining free rotation + kicks gives discrete-time evolution:
        p[n+1] = p[n] + K·sin(θ[n])
        θ[n+1] = θ[n] + p[n+1]  (mod 2π)

    **Key Feature - Area Preservation (Symplectic):**
    The Standard Map preserves phase space area (Liouville's theorem):
        |det(Jacobian)| = 1

    This is the discrete analog of energy conservation in Hamiltonian systems.
    Unlike dissipative systems (logistic map), there are NO attractors - 
    trajectories fill phase space densely.

    State Space:
    -----------
    State: x[n] = [θ[n], p[n]]
        Angular position:
        - θ: Angle [rad], periodic: θ ∈ [0, 2π) or (-π, π]
          * θ = 0: Reference position
          * θ + 2π ≡ θ (rotational symmetry)
          * Represents angular coordinate on circle

        Angular momentum:
        - p: Conjugate momentum [rad/time], unbounded: -∞ < p < ∞
          * p > 0: Counterclockwise rotation
          * p < 0: Clockwise rotation
          * p = 0: Stationary (at kick instant)
          * Between kicks: p is constant of motion
          * At kicks: p changes by K·sin(θ)

    **Phase Space Structure:**
    The (θ, p) plane has cylindrical topology:
    - θ is periodic (circle)
    - p is unbounded (line)
    - Phase space is a cylinder: S¹ × ℝ

    Control: u[n] (optional, for controlled standard map)
        - Can represent:
          * Variable kick strength: K → K + u[n]
          * Additional external torque
          * Feedback control to stabilize orbits
        - Standard map: u = 0 (autonomous)

    Output: y[n] = [θ[n], p[n]]
        - Full state measurement (both angle and momentum)
        - In practice:
          * θ measured via encoder/position sensor
          * p inferred from Δθ or measured via velocity sensor

    Dynamics:
    --------
    The Standard Map equations are:

        p[n+1] = p[n] + K·sin(θ[n])
        θ[n+1] = θ[n] + p[n+1]  (mod 2π)

    **Mathematical Properties:**

    1. **Area-preserving (symplectic):**
       Jacobian determinant = 1 (exactly)
       Phase space volume conserved
       No attractors or repellers

    2. **Time-reversal symmetry:**
       Map is reversible: can run backwards
       Inverse map exists

    3. **Two-dimensional:**
       Simplest non-integrable Hamiltonian system
       Higher dimensions allow Arnold diffusion

    4. **Poincaré map:**
       Samples continuous kicked rotor at kick times
       Reduces continuous flow to discrete map

    5. **Periodic in θ:**
       f(θ + 2π, p) = f(θ, p) + (2π, 0)
       Allows folding phase space onto cylinder

    Parameters:
    ----------
    K : float, default=1.0
        Kick strength (stochasticity parameter)
        
        **Physical Meaning:**
        - K measures perturbation strength relative to free motion
        - Dimensionless parameter (ratio of kick to rotation)
        - Controls degree of nonlinearity

        **Parameter Regimes:**

        **K = 0:**
        - Integrable limit (no kicks)
        - Pure rotation: θ[n+1] = θ[n] + p₀
        - All orbits are straight lines in (θ, p)
        - Every trajectory is regular (quasi-periodic)
        - Phase space foliated by invariant tori

        **0 < K < 0.5:**
        - Near-integrable regime
        - Most phase space covered by KAM tori (invariant curves)
        - Small chaotic regions near separatrices
        - Orbits mostly regular, quasi-periodic
        - KAM theorem applies: most tori survive

        **K ≈ 0.971635...**
        - Critical threshold for last KAM torus
        - Golden mean winding number (most robust)
        - Below: global stochastic layer blocked by KAM tori
        - Above: trajectories can diffuse to arbitrarily large |p|

        **0.5 < K < 2:**
        - Mixed phase space
        - Chaotic seas coexist with regular islands
        - Resonance overlap (Chirikov criterion)
        - Some KAM tori destroyed, others persist
        - Complicated fractal boundary structure

        **2 < K < 5:**
        - Mostly chaotic
        - Few surviving islands (high-order resonances)
        - Large-scale chaos, but bounded regions remain
        - Cantori (destroyed tori) create partial barriers

        **K > 5:**
        - Fully developed chaos
        - Almost all phase space is chaotic
        - Fast momentum diffusion: ⟨p²⟩ ~ K²·n (classical diffusion)
        - Only tiny islands remain at high-order resonances

        **K → ∞:**
        - Random walk limit
        - Momentum diffusion coefficient: D ~ K²/2
        - Approaches Brownian motion

    dt : float, default=1.0
        Time between kicks (kick period τ)
        - Usually normalized to 1
        - Physical time scale (seconds, milliseconds, etc.)
        - Doesn't affect map dynamics (only units)

    use_wrapped_theta : bool, default=True
        If True, θ ∈ [0, 2π); if False, θ unbounded
        - Wrapped: Natural for rotational dynamics
        - Unwrapped: Useful for tracking rotation number

    Equilibria and Periodic Orbits:
    -------------------------------
    Unlike dissipative systems, the Standard Map has NO fixed points in
    the usual sense (except at K = 0). Instead, it has:

    **Periodic Orbits:**
    - Period-1: (θ*, p*) such that map returns after 1 iteration
    - Period-n: Returns after n iterations
    - Infinitely many periodic orbits of each period
    - Form hierarchical island chain structure

    **Resonances:**
    For small K, periodic orbits occur near:
        p ≈ 2πm/n  (m, n integers)

    These are resonances where rotation number is rational.

    **Elliptic vs Hyperbolic Fixed Points:**
    - Elliptic: Stable, surrounded by invariant curves (islands)
    - Hyperbolic: Unstable, with stable/unstable manifolds (X-points)

    **Example Period-1 Orbits (K small):**
    - (0, 0): Elliptic, stable
    - (π, 0): Hyperbolic, unstable (separatrix)

    **KAM Tori (Invariant Curves):**
    For K < K_critical, some invariant curves survive:
    - Most robust: Golden mean winding number
        ω = (√5 - 1)/2 ≈ 0.618...
    - Form barriers to chaotic transport
    - Destroy via resonance overlap as K increases

    KAM Theory and Resonance Overlap:
    ---------------------------------
    **KAM Theorem (Kolmogorov-Arnold-Moser):**
    For sufficiently small perturbations, MOST invariant tori survive,
    though they are deformed. Tori are destroyed if:
    1. Winding number is rational (resonance)
    2. Winding number is "too well approximated" by rationals

    **Chirikov Resonance Overlap Criterion:**
    Chaos sets in when neighboring resonances overlap:
        ΔK ≈ K·(width of resonances)

    For Standard Map:
        K_critical ≈ 1 for onset of global stochasticity

    **Greene's Residue Criterion:**
    Precise threshold: K_c ≈ 0.971635...
    Last KAM torus has golden mean winding number.

    **Mixed Phase Space:**
    For K < K_c:
    - Chaotic sea confined by KAM tori
    - Regular islands embedded in chaos
    - Fractal boundaries (cantori)
    - No global diffusion

    For K > K_c:
    - Trajectories can diffuse to arbitrarily large |p|
    - Stochastic layer connects all regions
    - Cantori (partial barriers) slow diffusion

    Chaos Characterization:
    ----------------------
    **Lyapunov Exponents:**
    Area preservation implies:
        λ₁ + λ₂ = 0  (sum of Lyapunov exponents = 0)

    If λ₁ > 0 (chaos), then λ₂ = -λ₁ < 0.

    **Rotation Number:**
    For regular orbits, rotation number is defined:
        ω = lim (θ[n] - θ[0])/(2πn)
           n→∞

    - Rational ω: Periodic orbit
    - Irrational ω: Quasi-periodic (on KAM torus)
    - Undefined ω: Chaotic orbit

    **Poincaré-Birkhoff Theorem:**
    Each rational rotation number has at least two periodic orbits:
    one elliptic (stable) and one hyperbolic (unstable).

    **Winding Number Distribution:**
    For chaotic orbits, winding number has probabilistic description.

    Momentum Diffusion:
    ------------------
    In chaotic regime, momentum performs random walk:
        ⟨p²⟩ ~ D·n

    Diffusion coefficient:
    - K small: D ~ exp(-const/K) (exponentially suppressed)
    - K large: D ~ K²/2 (classical random walk)

    **Acceleration Modes:**
    At special K values, ballistic growth possible:
        ⟨p²⟩ ~ n²  (super-diffusive)

    This occurs when dynamics resonates with map periodicity.

    Control Objectives:
    ------------------
    **1. Chaos Control:**
       Goal: Stabilize unstable periodic orbits (UPO)
       Method: OGY (Ott-Grebogi-Yorke) control
       - Locate UPO in chaotic sea
       - Apply small perturbations to K
       - Stabilize orbit with minimal control

    **2. Chaos Suppression:**
       Goal: Restore regular motion by feedback
       Control: u[n] = -K_control·sin(θ[n])
       Reduces effective K below chaos threshold

    **3. Accelerator Mode Stabilization:**
       Goal: Maintain accelerator mode for rapid transport
       Applications: Particle beam control

    **4. Island Confinement:**
       Goal: Keep trajectory within regular island
       Useful for stability in beam dynamics

    **5. Transport Enhancement:**
       Goal: Maximize momentum diffusion
       Applications: Mixing, ergodic optimization

    Numerical Considerations:
    ------------------------
    **Angle Wrapping:**
    Always reduce θ modulo 2π to keep in [0, 2π):
        θ[n+1] = (θ[n] + p[n+1]) % (2π)

    Without wrapping, θ grows unbounded, complicating visualization.

    **Area Preservation Check:**
    Jacobian should always have determinant = 1:
        J = | ∂θ[n+1]/∂θ[n]  ∂θ[n+1]/∂p[n] |
            | ∂p[n+1]/∂θ[n]  ∂p[n+1]/∂p[n]  |

    det(J) = 1 + K·cos(θ)·1 - K·cos(θ)·1 = 1 ✓

    **Long-Time Integration:**
    - Standard Map is exactly area-preserving (no drift)
    - Can integrate indefinitely without accumulation of error
    - Unlike approximate integrators for continuous systems
    - Roundoff errors may accumulate but don't break symplecticity

    **Initial Conditions:**
    Choice of (θ₀, p₀) determines orbit character:
    - Regular regions: orbit stays on torus
    - Chaotic sea: orbit explores allowed region
    - Near separatrix: most sensitive to K

    **Visualization:**
    Phase space plots (Poincaré sections):
    - Plot (θ[n], p[n]) for many iterations
    - Regular orbits → closed curves (tori)
    - Chaotic orbits → scattered points (dense filling)
    - Mixed: islands + chaotic sea

    Example Usage:
    -------------
    >>> # Create Standard Map with moderate chaos
    >>> system = StandardMap(K=1.5)
    >>> 
    >>> # Single trajectory - chaotic sea
    >>> x0_chaotic = np.array([0.5, 0.5])
    >>> result_chaotic = system.simulate(
    ...     x0=x0_chaotic,
    ...     u_sequence=None,
    ...     n_steps=5000
    ... )
    >>> 
    >>> # Single trajectory - regular island
    >>> x0_regular = np.array([0.0, 0.1])
    >>> result_regular = system.simulate(
    ...     x0=x0_regular,
    ...     u_sequence=None,
    ...     n_steps=5000
    ... )
    >>> 
    >>> # Phase space portrait
    >>> import plotly.graph_objects as go
    >>> fig = go.Figure()
    >>> 
    >>> # Chaotic trajectory
    >>> fig.add_trace(go.Scatter(
    ...     x=result_chaotic['states'][:, 0],
    ...     y=result_chaotic['states'][:, 1],
    ...     mode='markers',
    ...     marker=dict(size=1, color='red', opacity=0.5),
    ...     name='Chaotic'
    ... ))
    >>> 
    >>> # Regular trajectory
    >>> fig.add_trace(go.Scatter(
    ...     x=result_regular['states'][:, 0],
    ...     y=result_regular['states'][:, 1],
    ...     mode='markers',
    ...     marker=dict(size=1, color='blue', opacity=0.5),
    ...     name='Regular'
    ... ))
    >>> 
    >>> fig.update_layout(
    ...     title=f'Standard Map Phase Space (K = {system.K})',
    ...     xaxis_title='θ [rad]',
    ...     yaxis_title='p [rad]',
    ...     width=800,
    ...     height=600
    ... )
    >>> fig.show()
    >>> 
    >>> # Generate phase space portrait with many initial conditions
    >>> phase_portrait = system.generate_phase_portrait(
    ...     n_trajectories=20,
    ...     n_steps=1000,
    ...     p_range=(-3, 3)
    ... )
    >>> fig_portrait = system.plot_phase_portrait(phase_portrait)
    >>> fig_portrait.show()
    >>> 
    >>> # Compute Lyapunov exponent
    >>> lyapunov = system.compute_lyapunov_exponent(
    ...     x0=np.array([1.0, 1.0]),
    ...     n_iterations=10000
    ... )
    >>> print(f"Lyapunov exponent: λ = {lyapunov:.4f}")
    >>> if lyapunov > 0.01:
    ...     print("Trajectory is CHAOTIC")
    >>> 
    >>> # Study K dependence (bifurcation-like diagram)
    >>> K_values = np.linspace(0.1, 5.0, 50)
    >>> lyapunov_vs_K = []
    >>> 
    >>> for K_val in K_values:
    ...     sys_temp = StandardMap(K=K_val)
    ...     lyap = sys_temp.compute_lyapunov_exponent(
    ...         x0=np.array([1.0, 1.0]),
    ...         n_iterations=5000
    ...     )
    ...     lyapunov_vs_K.append(lyap)
    >>> 
    >>> import plotly.graph_objects as go
    >>> fig = go.Figure()
    >>> fig.add_trace(go.Scatter(
    ...     x=K_values,
    ...     y=lyapunov_vs_K,
    ...     mode='lines+markers',
    ...     name='Lyapunov exponent'
    ... ))
    >>> fig.add_hline(y=0, line_dash='dash', line_color='red')
    >>> fig.update_layout(
    ...     title='Chaos Onset: Lyapunov Exponent vs Kick Strength',
    ...     xaxis_title='K (kick strength)',
    ...     yaxis_title='λ (Lyapunov exponent)',
    ...     width=900,
    ...     height=500
    ... )
    >>> fig.show()
    >>> 
    >>> # Compute rotation number
    >>> rotation_number = system.compute_rotation_number(
    ...     x0=np.array([0.0, 0.5]),
    ...     n_iterations=10000
    ... )
    >>> if rotation_number is not None:
    ...     print(f"Rotation number: ω = {rotation_number:.6f}")
    ...     # Check if rational
    ...     from fractions import Fraction
    ...     frac = Fraction(rotation_number).limit_denominator(1000)
    ...     print(f"Approximately: {frac}")
    ... else:
    ...     print("Orbit is chaotic (rotation number undefined)")
    >>> 
    >>> # Momentum diffusion analysis
    >>> diffusion_data = system.compute_momentum_diffusion(
    ...     x0=np.array([1.0, 1.0]),
    ...     n_steps=10000
    ... )
    >>> 
    >>> fig = go.Figure()
    >>> fig.add_trace(go.Scatter(
    ...     x=diffusion_data['n'],
    ...     y=diffusion_data['p_squared'],
    ...     mode='lines',
    ...     name='⟨p²⟩'
    ... ))
    >>> 
    >>> # Fit to p² ~ D·n
    >>> D_fit = np.polyfit(
    ...     diffusion_data['n'][1000:],
    ...     diffusion_data['p_squared'][1000:],
    ...     deg=1
    ... )[0]
    >>> fig.add_trace(go.Scatter(
    ...     x=diffusion_data['n'],
    ...     y=D_fit * diffusion_data['n'],
    ...     mode='lines',
    ...     line=dict(dash='dash'),
    ...     name=f'Fit: D = {D_fit:.3f}'
    ... ))
    >>> 
    >>> fig.update_layout(
    ...     title=f'Momentum Diffusion (K = {system.K})',
    ...     xaxis_title='Iteration n',
    ...     yaxis_title='⟨p²⟩',
    ...     width=800,
    ...     height=500
    ... )
    >>> fig.show()
    >>> 
    >>> # Compare different K values in phase space
    >>> from plotly.subplots import make_subplots
    >>> 
    >>> K_compare = [0.5, 1.0, 2.0, 4.0]
    >>> fig = make_subplots(
    ...     rows=2, cols=2,
    ...     subplot_titles=[f'K = {K}' for K in K_compare]
    ... )
    >>> 
    >>> for idx, K_val in enumerate(K_compare):
    ...     row = idx // 2 + 1
    ...     col = idx % 2 + 1
    ...     
    ...     sys_temp = StandardMap(K=K_val)
    ...     portrait = sys_temp.generate_phase_portrait(
    ...         n_trajectories=15,
    ...         n_steps=800,
    ...         p_range=(-np.pi, np.pi)
    ...     )
    ...     
    ...     for traj in portrait['trajectories']:
    ...         fig.add_trace(
    ...             go.Scatter(
    ...                 x=traj[:, 0],
    ...                 y=traj[:, 1],
    ...                 mode='markers',
    ...                 marker=dict(size=0.5),
    ...                 showlegend=False
    ...             ),
    ...             row=row, col=col
    ...         )
    >>> 
    >>> fig.update_xaxes(title_text='θ', range=[0, 2*np.pi])
    >>> fig.update_yaxes(title_text='p')
    >>> fig.update_layout(
    ...     title_text='Standard Map: Transition to Chaos',
    ...     height=800,
    ...     width=1000
    ... )
    >>> fig.show()

    Physical Insights:
    -----------------
    **Why Area Preservation Matters:**
    Conservation of phase space volume is the discrete analog of energy
    conservation. Unlike dissipative systems:
    - No attractors (everything keeps moving)
    - No basins of attraction
    - Volume in phase space is conserved
    - Ergodic properties possible

    **KAM Tori as Barriers:**
    Invariant curves (KAM tori) act as impenetrable barriers:
    - Prevent momentum diffusion
    - Confine chaotic regions
    - When destroyed → global transport possible
    - Golden mean torus is most robust

    **Resonance Overlap:**
    Chaos emerges when perturbation strong enough that:
    - Neighboring resonances overlap
    - Separatrices intersect chaotically
    - Homoclinic tangle forms
    - Regular motion destroyed

    **Connection to Quantum Mechanics:**
    Standard Map is prototypical for quantum chaos:
    - Classical chaos ↔ Quantum eigenfunctions
    - K plays role of effective ℏ (inverse)
    - "Quantum" Standard Map shows localization
    - Dynamical Anderson localization

    **Accelerator Physics:**
    Standard Map models:
    - Particles in storage rings
    - RF cavity kicks
    - Nonlinear resonances
    - Beam stability criteria

    **Celestial Mechanics:**
    Analogous to:
    - Asteroid belt gaps (resonances with Jupiter)
    - Satellite orbit perturbations
    - Three-body problem (restricted)
    - Kirkwood gaps explained by resonance overlap

    Common Pitfalls:
    ---------------
    1. **Forgetting angle wrapping:**
       Must reduce θ mod 2π for correct visualization
       Unwrapped θ grows linearly with rotation

    2. **Confusing with dissipative maps:**
       Standard Map has NO attractors
       Phase space is uniformly filled (ergodic)
       Cannot use attractor-finding methods

    3. **Insufficient iterations:**
       Need many iterations (>1000) to see structure
       Chaotic orbits fill region densely
       Regular orbits trace closed curves slowly

    4. **Wrong initial conditions:**
       Different regions show different behavior
       Need to sample many ICs to see full phase space

    5. **Ignoring symplectic structure:**
       Standard Map exactly preserves det(J) = 1
       Numerical methods must respect this
       Standard integration schemes may fail

    6. **Rotation number convergence:**
       Chaotic orbits have no well-defined rotation number
       Regular orbits require many iterations to converge
       Rational approximations can be misleading

    Extensions and Variations:
    -------------------------
    1. **Generalized Standard Map:**
       p[n+1] = p[n] + K·f(θ[n])
       θ[n+1] = θ[n] + p[n+1]
       Different f(θ) changes chaos threshold

    2. **Dissipative Standard Map:**
       p[n+1] = γ·p[n] + K·sin(θ[n])
       θ[n+1] = θ[n] + p[n+1]
       γ < 1 breaks area preservation, creates attractors

    3. **Standard Map with Noise:**
       Stochastic perturbations
       Studies interplay of chaos and noise

    4. **Four-Dimensional Standard Map:**
       Two coupled kicked rotors
       Shows Arnold diffusion

    5. **Quantum Standard Map:**
       Kicked rotor in quantum mechanics
       Dynamical localization phenomenon

    6. **Web Map (K < 0):**
       Accelerator-mode version
       Ballistic momentum growth

    See Also:
    --------
    HenonMap : 2D dissipative chaos
    LogisticMap : 1D dissipative chaos
    """

    def define_system(
        self,
        K: float = 1.0,
        dt: float = 1.0,
        use_wrapped_theta: bool = True,
        use_controlled_version: bool = False,
    ):
        """
        Define Standard Map dynamics.

        Parameters
        ----------
        K : float
            Kick strength (stochasticity parameter)
        dt : float
            Time between kicks (usually 1.0)
        use_wrapped_theta : bool
            If True, wrap θ to [0, 2π); if False, allow unbounded θ
        use_controlled_version : bool
            If True, adds control input u[n] to momentum equation
        """
        # Store parameters
        self._K_val = K
        self._use_wrapped = use_wrapped_theta
        self._use_controlled = use_controlled_version

        # Validate parameter
        if K < 0:
            import warnings
            warnings.warn(
                f"Kick strength K = {K} < 0 gives 'web map' with different dynamics.",
                UserWarning
            )

        # State variables
        theta, p = sp.symbols('theta p', real=True)
        
        # Symbolic parameter
        K_sym = sp.symbols('K', real=True)

        self.state_vars = [theta, p]
        self.parameters = {K_sym: K}
        self._dt = dt
        self.order = 1

        if use_controlled_version:
            # Controlled Standard Map
            u = sp.symbols('u', real=True)
            self.control_vars = [u]
            p_next = p + K_sym * sp.sin(theta) + u
        else:
            # Standard autonomous map
            self.control_vars = []
            p_next = p + K_sym * sp.sin(theta)

        # Theta update (with optional wrapping)
        theta_next = theta + p_next

        if use_wrapped_theta:
            # Will be wrapped to [0, 2π) in evaluation
            # Symbolic expression stays as-is
            pass

        self._f_sym = sp.Matrix([theta_next, p_next])

    def setup_equilibria(self):
        """
        Set up periodic orbits (no true fixed points for K ≠ 0).

        For K ≠ 0, there are no fixed points, only periodic orbits.
        We add the origin as a "reference point" for small K.
        """
        K = self._K_val

        if abs(K) < 1e-10:
            # Integrable case: straight lines are "equilibria"
            self.add_equilibrium(
                "integrable_reference",
                x_eq=np.array([0.0, 0.0]),
                u_eq=np.array([0.0]) if self._use_controlled else np.array([]),
                verify=False,
                notes="Integrable limit K=0: all points are neutral equilibria"
            )
        else:
            # Add period-1 elliptic point (approximate for small K)
            self.add_equilibrium(
                "elliptic_approximate",
                x_eq=np.array([0.0, 0.0]),
                u_eq=np.array([0.0]) if self._use_controlled else np.array([]),
                verify=False,
                notes=f"Approximate elliptic fixed point for K={K:.3f}. "
                      "Note: No true fixed points exist for K≠0 in Standard Map."
            )

    @property
    def K(self) -> float:
        """Kick strength parameter."""
        return self._K_val

    def step(
        self,
        x: np.ndarray,
        u: Optional[np.ndarray] = None,
        k: int = 0,
        backend: Optional[str] = None,
    ) -> np.ndarray:
        """
        Override step to add angle wrapping if enabled.

        Parameters
        ----------
        x : np.ndarray
            State [θ, p]
        u : Optional[np.ndarray]
            Control input (if enabled)
        k : int
            Time step
        backend : Optional[str]
            Backend to use

        Returns
        -------
        np.ndarray
            Next state [θ[k+1], p[k+1]] with θ wrapped if enabled
        """
        # Call parent step method
        x_next = super().step(x, u, k, backend)

        # Wrap theta if enabled
        if self._use_wrapped:
            x_next[0] = np.mod(x_next[0], 2 * np.pi)

        return x_next

    def compute_jacobian(
        self,
        x: np.ndarray,
    ) -> np.ndarray:
        """
        Compute Jacobian matrix at state x.

        For Standard Map:
            J = | 1 + K·cos(θ)    1 |
                | K·cos(θ)        1 |

        Parameters
        ----------
        x : np.ndarray
            State [θ, p]

        Returns
        -------
        np.ndarray
            Jacobian matrix (2×2)

        Notes
        -----
        Determinant is always 1 (area-preserving):
            det(J) = (1 + K·cos(θ))·1 - K·cos(θ)·1 = 1 ✓
        """
        theta = x[0]
        K = self._K_val

        J = np.array([
            [1 + K * np.cos(theta), 1],
            [K * np.cos(theta), 1]
        ])

        return J

    def verify_symplectic(
        self,
        x: np.ndarray,
        tolerance: float = 1e-10,
    ) -> bool:
        """
        Verify that Jacobian is symplectic (det = 1).

        Parameters
        ----------
        x : np.ndarray
            State to check
        tolerance : float
            Tolerance for determinant = 1

        Returns
        -------
        bool
            True if symplectic within tolerance

        Examples
        --------
        >>> system = StandardMap(K=2.0)
        >>> x_test = np.array([1.0, 0.5])
        >>> is_symplectic = system.verify_symplectic(x_test)
        >>> print(f"Symplectic: {is_symplectic}")
        """
        J = self.compute_jacobian(x)
        det_J = np.linalg.det(J)
        return abs(det_J - 1.0) < tolerance

    def compute_lyapunov_exponent(
        self,
        x0: np.ndarray,
        n_iterations: int = 10000,
        n_transient: int = 1000,
    ) -> float:
        """
        Compute largest Lyapunov exponent.

        For Standard Map, λ₁ + λ₂ = 0 (symplectic constraint).
        We compute λ₁ (largest exponent).

        Parameters
        ----------
        x0 : np.ndarray
            Initial state [θ₀, p₀]
        n_iterations : int
            Number of iterations for averaging
        n_transient : int
            Number of initial iterations to discard

        Returns
        -------
        float
            Largest Lyapunov exponent
            - λ > 0: Chaotic
            - λ ≈ 0: Neutral (near bifurcation or integrable)
            - λ < 0: Regular (on KAM torus)

        Examples
        --------
        >>> system = StandardMap(K=2.0)
        >>> lyap = system.compute_lyapunov_exponent(
        ...     x0=np.array([1.0, 1.0]),
        ...     n_iterations=20000
        ... )
        >>> print(f"Lyapunov exponent: {lyap:.4f}")
        """
        x = x0.copy()

        # Discard transient
        for _ in range(n_transient):
            x = self.step(x)

        # Initialize tangent vector
        v = np.array([1.0, 0.0])
        lyap_sum = 0.0

        for _ in range(n_iterations):
            # Compute Jacobian at current point
            J = self.compute_jacobian(x)

            # Evolve tangent vector
            v = J @ v

            # Renormalize and accumulate log(norm)
            norm_v = np.linalg.norm(v)
            lyap_sum += np.log(norm_v)
            v = v / norm_v

            # Evolve state
            x = self.step(x)

        return lyap_sum / n_iterations

    def compute_rotation_number(
        self,
        x0: np.ndarray,
        n_iterations: int = 10000,
        tolerance: float = 1e-6,
    ) -> Optional[float]:
        """
        Compute rotation number (winding number) for regular orbits.

        For regular (quasi-periodic) orbits on KAM tori:
            ω = lim (θ[n] - θ[0])/(2πn)
               n→∞

        Parameters
        ----------
        x0 : np.ndarray
            Initial state [θ₀, p₀]
        n_iterations : int
            Number of iterations
        tolerance : float
            Tolerance for convergence check

        Returns
        -------
        float or None
            Rotation number if orbit is regular, None if chaotic

        Examples
        --------
        >>> system = StandardMap(K=0.5)
        >>> # Regular orbit
        >>> omega = system.compute_rotation_number(np.array([0.5, 0.5]))
        >>> if omega is not None:
        ...     print(f"Rotation number: ω = {omega:.6f}")

        Notes
        -----
        For chaotic orbits, rotation number is undefined (returns None).
        For rational rotation numbers, orbit is periodic.
        For irrational rotation numbers, orbit is quasi-periodic (on torus).
        """
        # Track unwrapped angle
        theta_unwrapped = x0[0]
        x = x0.copy()

        # Iterate and accumulate angle changes
        for _ in range(n_iterations):
            x_next = self.step(x)
            
            # Compute unwrapped angle change
            dtheta = x_next[0] - x[0]
            
            # Handle wrapping discontinuities
            if self._use_wrapped:
                if dtheta > np.pi:
                    dtheta -= 2 * np.pi
                elif dtheta < -np.pi:
                    dtheta += 2 * np.pi
            
            theta_unwrapped += dtheta
            x = x_next

        # Compute rotation number
        omega = theta_unwrapped / (2 * np.pi * n_iterations)

        # Check for convergence (regular orbit)
        # Run again with more iterations and compare
        theta_unwrapped2 = x0[0]
        x = x0.copy()
        n_check = n_iterations * 2

        for _ in range(n_check):
            x_next = self.step(x)
            dtheta = x_next[0] - x[0]
            
            if self._use_wrapped:
                if dtheta > np.pi:
                    dtheta -= 2 * np.pi
                elif dtheta < -np.pi:
                    dtheta += 2 * np.pi
            
            theta_unwrapped2 += dtheta
            x = x_next

        omega2 = theta_unwrapped2 / (2 * np.pi * n_check)

        # Check convergence
        if abs(omega - omega2) < tolerance:
            return omega
        else:
            return None  # Chaotic (rotation number doesn't converge)

    def compute_momentum_diffusion(
        self,
        x0: np.ndarray,
        n_steps: int = 10000,
    ) -> Dict:
        """
        Compute momentum diffusion ⟨p²⟩ vs iteration number.

        Parameters
        ----------
        x0 : np.ndarray
            Initial state
        n_steps : int
            Number of iterations

        Returns
        -------
        dict
            Dictionary with:
            - 'n': Iteration numbers
            - 'p': Momentum values
            - 'p_squared': ⟨p²⟩ values

        Examples
        --------
        >>> system = StandardMap(K=3.0)
        >>> diff_data = system.compute_momentum_diffusion(
        ...     x0=np.array([1.0, 1.0]),
        ...     n_steps=20000
        ... )
        >>> 
        >>> import plotly.graph_objects as go
        >>> fig = go.Figure()
        >>> fig.add_trace(go.Scatter(
        ...     x=diff_data['n'],
        ...     y=diff_data['p_squared'],
        ...     mode='lines'
        ... ))
        >>> fig.update_layout(title='Momentum Diffusion')
        >>> fig.show()
        """
        result = self.simulate(
            x0=x0,
            u_sequence=None,
            n_steps=n_steps
        )

        p_values = result['states'][:, 1]
        p_squared = p_values ** 2

        return {
            'n': np.arange(n_steps + 1),
            'p': p_values,
            'p_squared': p_squared,
        }

    def classify_regime(self) -> str:
        """
        Classify dynamical regime based on K value.

        Returns
        -------
        str
            Regime description

        Examples
        --------
        >>> system = StandardMap(K=0.3)
        >>> print(system.classify_regime())
        'near_integrable'
        """
        K = self._K_val

        if abs(K) < 1e-10:
            return "integrable"
        elif K < 0.5:
            return "near_integrable"
        elif K < 1.0:
            return "mixed_phase_space"
        elif K < 2.0:
            return "mostly_chaotic"
        elif K <= 5.0:
            return "fully_chaotic"
        else:
            return "strong_chaos_diffusive"

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
    #     print("Standard Map (Chirikov-Taylor Map): Paradigm of Hamiltonian Chaos")
    #     print(f"\nKick Strength: K = {self._K_val}")

    #     # Classify regime
    #     regime = self.classify_regime()
    #     regime_names = {
    #         "integrable": "Integrable (K = 0)",
    #         "near_integrable": "Near-Integrable (K < 0.5, KAM tori dominate)",
    #         "mixed_phase_space": "Mixed Phase Space (0.5 < K < 1, chaos + islands)",
    #         "mostly_chaotic": "Mostly Chaotic (1 < K < 2, few islands remain)",
    #         "fully_chaotic": "Fully Chaotic (2 < K ≤ 5, almost all chaotic)",
    #         "strong_chaos_diffusive": "Strong Chaos (K > 5, rapid diffusion)",
    #     }
    #     print(f"Regime: {regime_names.get(regime, 'Unknown')}")

    #     print(f"\nState: x = [θ, p] (angle, momentum)")
    #     print(f"  - θ ∈ [0, 2π) (periodic)")
    #     print(f"  - p ∈ (-∞, ∞) (unbounded)")
    #     if self._use_controlled:
    #         print(f"Control: u (additional momentum kick)")
    #     else:
    #         print("Autonomous system (no control)")

    #     print("\nDynamics: Standard Map equations")
    #     print("  p[n+1] = p[n] + K·sin(θ[n])")
    #     print("  θ[n+1] = θ[n] + p[n+1]  (mod 2π)")

    #     # Show actual symbolic expressions
    #     print("\nSymbolic Dynamics:")
    #     for i, (_, expr) in enumerate(zip(self.state_vars, self._f_sym)):
    #         expr_sub = self.substitute_parameters(expr)
    #         if simplify:
    #             expr_sub = sp.simplify(expr_sub)
    #         label = ['θ[n+1]', 'p[n+1]'][i]
    #         print(f"  {label} = {expr_sub}")

    #     # Jacobian
    #     print("\nJacobian Matrix:")
    #     print("  J = | 1 + K·cos(θ)    1 |")
    #     print("      | K·cos(θ)        1 |")
    #     print(f"  det(J) = 1 (area-preserving / symplectic)")

    #     # Critical values
    #     print("\nCritical Parameters:")
    #     print("  K = 0: Integrable (straight-line flow)")
    #     print("  K ≈ 0.97: Last KAM torus destroyed (golden mean)")
    #     print("  K ≈ 1: Onset of global stochasticity")
    #     print("  K > 5: Classical diffusion regime (D ~ K²/2)")

    #     print("\nPhysical Interpretation:")
    #     print("  - Models kicked rotor (pendulum with periodic impulses)")
    #     print("  - θ: Angular position on cylinder S¹")
    #     print("  - p: Angular momentum (conserved between kicks)")
    #     print("  - K·sin(θ): Momentum kick at each iteration")
    #     print("  - Area-preserving: phase space volume conserved")

    #     print("\nKey Features:")
    #     print("  - Hamiltonian chaos (conservative dynamics)")
    #     print("  - KAM tori (invariant curves) for small K")
    #     print("  - Resonance overlap → chaos")
    #     print("  - Mixed phase space (regular islands + chaotic sea)")
    #     print("  - No attractors (trajectories explore allowed regions)")

    #     print("\nApplications:")
    #     print("  - Particle accelerator beam dynamics")
    #     print("  - Celestial mechanics (asteroid belt resonances)")
    #     print("  - Plasma confinement in fusion")
    #     print("  - Quantum chaos (quantum kicked rotor)")
    #     print("  - Fundamental test of KAM theory")

    #     print("=" * 70)


# Alias for backward compatibility
ChirikovTaylorMap = StandardMap
ChirikovMap = StandardMap
KickedRotor = StandardMap