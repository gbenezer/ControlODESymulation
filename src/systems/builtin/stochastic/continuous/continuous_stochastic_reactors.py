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
Stochastic Batch Reactor - Chemical Reactor with Process Noise
===============================================================

This module extends the deterministic batch reactor model to include stochastic
effects arising from process noise, measurement uncertainty, and inherent
randomness in chemical kinetics. The stochastic batch reactor serves as:

- A realistic model incorporating parameter uncertainty
- A benchmark for stochastic optimal control algorithms
- A test system for robust control design under uncertainty
- An illustration of noise propagation in nonlinear chemical systems
- A foundation for Bayesian state estimation and parameter identification

Stochastic effects are critical in chemical reactors for:
- Process control under uncertainty (fluctuating feed, catalyst aging)
- Safety analysis (rare events, tail risk assessment)
- Quality control (batch-to-batch variability)
- Optimal experimental design (parameter estimation with noise)

Physical Sources of Noise
--------------------------

**1. Kinetic Noise (Molecular Stochasticity):**
Chemical reactions are fundamentally stochastic at molecular level:
- Small molecule counts: Poisson noise in reaction events
- Thermal fluctuations: Random molecular collisions
- Catalyst heterogeneity: Variable active site distribution
- For macroscopic systems: Typically negligible (Law of Large Numbers)
- For microreactors: Can be significant

**2. Process Disturbances:**
External perturbations affecting reactor operation:
- Feed composition variations: Batch-to-batch variability in raw materials
- Temperature fluctuations: Imperfect temperature control, ambient changes
- Mixing imperfections: Local concentration gradients in "well-mixed" reactor
- Heat transfer variability: Fouling, jacket flow rate fluctuations
- Catalyst activity changes: Deactivation, regeneration cycles

**3. Parameter Uncertainty:**
Model parameters known only approximately:
- Reaction rate constants: k₁, k₂ (measurement error, batch variability)
- Activation energies: E₁, E₂ (empirical fit uncertainty)
- Heat transfer coefficient: α (fouling, flow conditions)
- Measurement noise: Sensor drift, calibration errors

**4. Model Mismatch:**
Unmodeled dynamics and approximations:
- Simplified kinetics: First-order approximation of complex mechanisms
- Perfect mixing assumption: Ignores spatial gradients
- Isothermal jacket: Ignores jacket dynamics
- Single-phase assumption: Ignores potential phase changes

Mathematical Formulation
-------------------------

**Stochastic Differential Equation (Itô form):**

The stochastic batch reactor dynamics:

    dC_A = (-r₁)·dt + σ_A·dW_A
    dC_B = (r₁ - r₂)·dt + σ_B·dW_B  
    dT = (Q - α·(T - T_amb))·dt + σ_T·dW_T

where:
- r₁, r₂: Reaction rates (deterministic part)
- σ_A, σ_B, σ_T: Diffusion coefficients (noise intensities)
- W_A, W_B, W_T: Independent Wiener processes (Brownian motions)
- dW ~ N(0, dt): Brownian motion increments

**Noise Structure:**

Three independent noise sources model different physical phenomena:

1. **Concentration Noise (σ_A, σ_B):**
   - Source: Feed variability, mixing imperfections, kinetic uncertainty
   - Units: [mol/(L·√s)]
   - Typical magnitude: σ_A ~ 0.001-0.01 mol/(L·√s)
   - Effect: Causes concentration to deviate from deterministic trajectory
   - Additive: Does not depend on current concentration

2. **Temperature Noise (σ_T):**
   - Source: Heat transfer fluctuations, ambient variations, sensor noise
   - Units: [K/√s]
   - Typical magnitude: σ_T ~ 0.1-1.0 K/√s
   - Effect: Creates temperature fluctuations affecting reaction rates
   - Additive: Constant noise intensity
   - Most critical: Temperature affects rates exponentially (Arrhenius)

**Why Additive Noise?**

This implementation uses **additive noise** (state-independent):
- Mathematically simpler: Easier analysis and simulation
- Physically reasonable: Many disturbances are external to state
- Computational efficiency: No state-dependent diffusion calculations
- Conservative: Doesn't vanish when state is small

Alternative: Multiplicative noise (state-dependent):
    dC_A = (-r₁)·dt + σ_A·C_A·dW_A

Would model: Relative errors scale with concentration.

**Noise Correlation:**

Independent noise sources: Cov[dW_A, dW_B] = 0

Physical justification:
- Different mechanisms (concentration vs temperature)
- Spatial separation (bulk vs jacket)
- Measurement independence

Extension: Correlated noise via covariance matrix Σ.

Key Properties
--------------

**1. Stochasticity:**
Sample paths are random, continuous but nowhere differentiable.

**2. Mean Behavior:**
Expected value: E[X(t)] evolves according to deterministic dynamics
(for additive noise).

**3. Variance Growth:**
Variance: Var[X(t)] grows with time due to noise accumulation.

**4. Non-Stationary:**
Unlike OU process, no equilibrium distribution (batch operation).

**5. Path-Wise Uniqueness:**
Each simulation produces different trajectory (Monte Carlo ensemble).

**6. Markov Property:**
Future evolution depends only on current state (memoryless).

Comparison with Deterministic Model
------------------------------------

**Deterministic:** dx = f(x,u)·dt
- Single trajectory
- Perfectly predictable
- Sufficient for nominal design

**Stochastic:** dx = f(x,u)·dt + g(x,u)·dW
- Ensemble of trajectories
- Probabilistic predictions
- Necessary for robust design

**When Stochastic Model is Essential:**
- Robust control design (worst-case analysis)
- Risk assessment (probability of constraint violation)
- State estimation (Kalman filter, particle filter)
- Parameter identification (Bayesian inference)
- Quality control (process capability analysis)

Applications
------------

**1. Robust Optimal Control:**
Maximize expected yield while managing risk:
    max E[C_B(t_f)] - λ·Var[C_B(t_f)]

Tradeoff between performance and reliability.

**2. Stochastic MPC:**
Model Predictive Control with chance constraints:
    P(T(t) < T_max) ≥ 0.95

Ensures safety with high probability.

**3. State Estimation:**
Kalman filter for noisy measurements:
    y_measured = h(x) + v

Optimal state estimate from noisy data.

**4. Parameter Identification:**
Bayesian inference of unknown parameters:
    P(θ | data) ∝ P(data | θ)·P(θ)

Quantify parameter uncertainty.

**5. Risk Analysis:**
Monte Carlo simulation for rare events:
    P(runaway) = P(T > T_critical)

Assess tail risk via ensemble statistics.

**6. Process Capability:**
Six Sigma analysis:
    C_pk = min(USL - μ, μ - LSL) / (3σ)

Quantify process variability.

Numerical Integration
---------------------

**Recommended Settings:**
- Standard accuracy: Euler-Maruyama, dt = 0.01-0.1 s
- High accuracy: Milstein or SRK, dt = 0.001-0.01 s
- Stiff systems: Use implicit methods (framework support)

**Convergence:**
Check convergence by halving dt:
- Weak convergence: Mean, variance should stabilize
- Strong convergence: Sample paths should converge

Monte Carlo Simulation
----------------------

**Ensemble Analysis:**

To characterize stochastic behavior:
1. Run N independent simulations (N = 100-10,000)
2. Compute statistics at each time:
   - Mean: μ(t) = (1/N)·Σ X_i(t)
   - Variance: σ²(t) = (1/N)·Σ (X_i(t) - μ(t))²
   - Percentiles: 5th, 50th, 95th for confidence bands
3. Visualize:
   - Spaghetti plot: All trajectories
   - Mean ± 2σ bands: 95% confidence region
   - Histogram at final time: Distribution shape

**Rare Event Simulation:**

For low-probability events (P < 0.01):
- Importance sampling: Bias toward rare region
- Multilevel Monte Carlo: Exploit multiple resolutions
- Splitting methods: Sequential sampling toward rare event

**Variance Reduction:**

Techniques to reduce Monte Carlo error:
- Antithetic variates: Use ±Z pairs
- Control variates: Use known expectation
- Common random numbers: Compare scenarios consistently

Parameter Selection Guidelines
-------------------------------

**Noise Intensity Selection:**

Rules of thumb for noise magnitudes:

1. **Concentration Noise (σ_A, σ_B):**
   - Small: σ ~ 0.001 mol/(L·√s) (precise control, large batch)
   - Medium: σ ~ 0.01 mol/(L·√s) (typical industrial)
   - Large: σ ~ 0.1 mol/(L·√s) (poor control, small batch)
   
   Guideline: σ_A ~ 0.01·C_A0 for 1% relative noise

2. **Temperature Noise (σ_T):**
   - Small: σ_T ~ 0.1 K/√s (good temperature control)
   - Medium: σ_T ~ 1.0 K/√s (typical industrial)
   - Large: σ_T ~ 5.0 K/√s (poor control, external disturbances)
   
   Guideline: σ_T ~ 1 K/√s for typical batch reactor

**Physical Validation:**

Check if noise levels are reasonable:
- Simulate deterministic + stochastic
- Compare trajectories: Should be close but not identical
- Variance should grow but not dominate
- Final concentrations: Coefficient of variation < 10-20%

Limitations and Extensions
---------------------------

**Current Model Limitations:**
- Additive noise only (no multiplicative)
- Independent noise sources (no correlation)
- Constant noise intensity (no state-dependence)
- No jumps (only continuous paths)
- No model parameter uncertainty (fixed parameters)

**Possible Extensions:**

1. **Multiplicative Noise:**
   ```
   dC_A = (-r₁)·dt + σ_A·C_A·dW_A
   ```
   Noise scales with concentration (relative errors).

2. **Correlated Noise:**
   ```
   dW = [dW_A, dW_B, dW_T]ᵀ with Cov[dW] = Σ·dt
   ```
   Model simultaneous disturbances.

3. **Jump Diffusion:**
   ```
   dX = f·dt + g·dW + h·dN
   ```
   where dN is Poisson process (discrete events).

4. **State-Dependent Diffusion:**
   ```
   g = g(x,t) - varies with state
   ```
   More realistic but more complex.

5. **Parameter Uncertainty:**
   ```
   dθ = σ_θ·dW_θ (random walk parameters)
   ```
   Model slowly varying parameters.

Common Pitfalls
---------------

1. **Wrong Time Scaling:**
   - Noise intensity has units [state]/√[time]
   - Don't forget √dt factor in discretization

2. **Too Large dt:**
   - SDE integration less stable than ODE
   - Use smaller dt than deterministic case

3. **Single Trajectory:**
   - One simulation is not representative
   - Always run ensemble (N ≥ 100)

4. **Ignoring Initial Transient:**
   - Initial conditions affect early dynamics
   - Report statistics after transient if analyzing steady-state

5. **Over-Interpreting Noise:**
   - Not all fluctuations are noise
   - Could be model mismatch or deterministic chaos

"""

from typing import List, Optional, Tuple

import numpy as np
import sympy as sp

from src.systems.base.core.continuous_stochastic_system import ContinuousStochasticSystem


class ContinuousStochasticBatchReactor(ContinuousStochasticSystem):
    """
    Continuous-time stochastic batch reactor with process noise.

    Extends the deterministic batch reactor to include stochastic effects
    from process disturbances, parameter uncertainty, and inherent randomness.
    This model is essential for robust control design, risk analysis, and
    state estimation under uncertainty.

    Stochastic Differential Equations
    ----------------------------------
    The reactor dynamics with additive noise:

        dC_A = (-r₁)·dt + σ_A·dW_A
        dC_B = (r₁ - r₂)·dt + σ_B·dW_B
        dT = (Q - α·(T - T_amb))·dt + σ_T·dW_T

    where:
        - r₁ = k₁·C_A·exp(-E₁/T): Reaction 1 rate (A → B)
        - r₂ = k₂·C_B·exp(-E₂/T): Reaction 2 rate (B → C)
        - σ_A, σ_B: Concentration noise intensities [mol/(L·√s)]
        - σ_T: Temperature noise intensity [K/√s]
        - W_A, W_B, W_T: Independent Wiener processes

    Physical Interpretation
    -----------------------
    **Noise Sources:**

    1. **Concentration Noise (σ_A, σ_B):**
       - Feed composition variability
       - Mixing imperfections
       - Sampling uncertainty
       - Kinetic parameter fluctuations

    2. **Temperature Noise (σ_T):**
       - Heat transfer variations
       - Ambient temperature changes
       - Control system imperfections
       - Measurement noise

    **Why Three Independent Sources?**
    - Different physical mechanisms
    - Spatial separation (bulk vs jacket)
    - Independent control loops

    State Space
    -----------
    State: x = [C_A, C_B, T]
        Same as deterministic model but now stochastic:
        - C_A(t): Random process, not deterministic function
        - C_B(t): Random process
        - T(t): Random process
        
        Each trajectory is one realization from probability distribution.

    Control: u = [Q]
        - Q: Heating/cooling rate [K/s]
        - Same as deterministic (no noise in control)

    Noise: w = [w_A, w_B, w_T]
        - Three independent Wiener processes
        - Dimension: nw = 3
        - Structure: Diagonal (uncorrelated)

    Key Properties
    --------------
    **Stochastic Nature:**
    - Multiple runs give different trajectories
    - Statistics (mean, variance) evolve over time
    - Need ensemble analysis (Monte Carlo)

    **Mean Behavior:**
    For additive noise: E[X(t)] follows deterministic dynamics approximately.

    **Variance Growth:**
    Variance increases with time due to noise accumulation:
        Var[C_A(t)] ≈ Var[C_A(0)] + σ_A²·t

    **Non-Gaussian:**
    Even with Gaussian noise, nonlinear dynamics create non-Gaussian
    distributions (except in linear approximation).

    Parameters
    ----------
    k1 : float
        Pre-exponential factor for A→B reaction [1/s]
    k2 : float
        Pre-exponential factor for B→C reaction [1/s]
    E1 : float
        Activation energy for reaction 1 [K]
    E2 : float
        Activation energy for reaction 2 [K]
    alpha : float
        Heat transfer coefficient [1/s]
    T_amb : float
        Ambient temperature [K]
    C_A0 : Optional[float]
        Initial concentration of A for equilibrium setup [mol/L]
    T0 : Optional[float]
        Initial temperature for equilibrium setup [K]

    sigma_A : float, default=0.01
        Concentration noise for A [mol/(L·√s)]
        - Controls C_A fluctuations
        - Typical: 0.001-0.1
        - Should be << C_A0 for realistic noise

    sigma_B : float, default=0.01
        Concentration noise for B [mol/(L·√s)]
        - Controls C_B fluctuations
        - Typical: 0.001-0.1

    sigma_T : float, default=1.0
        Temperature noise [K/√s]
        - Controls T fluctuations
        - Typical: 0.1-5.0
        - More critical than concentration noise
          (exponential effect via Arrhenius)

    Applications
    ------------
    **1. Robust Optimal Control:**
    Design controllers accounting for uncertainty:
        - Stochastic MPC with chance constraints
        - Risk-sensitive control
        - Robust trajectory optimization

    **2. State Estimation:**
    Kalman filter for noisy measurements:
        - Extended Kalman Filter (EKF)
        - Unscented Kalman Filter (UKF)
        - Particle filter

    **3. Risk Analysis:**
    Assess probability of constraint violation:
        - Monte Carlo simulation
        - Rare event estimation
        - Safety verification

    **4. Parameter Identification:**
    Estimate parameters from noisy data:
        - Maximum likelihood
        - Bayesian inference
        - Sequential Monte Carlo

    **5. Process Design:**
    Design for robustness:
        - Worst-case analysis
        - Six Sigma methodology
        - Process capability studies

    Numerical Simulation
    --------------------
    **Recommended Methods:**
    - Euler-Maruyama: Simple, robust, dt ~ 0.01-0.1 s
    - Milstein: Higher order, dt ~ 0.001-0.01 s
    - SRK: Even higher order, more cost

    **Monte Carlo Ensemble:**
    Run N = 100-10,000 simulations to characterize:
        - Mean trajectory: E[X(t)]
        - Variance: Var[X(t)]
        - Confidence bands: μ ± 2σ
        - Probability distributions

    **Noise Structure:**
    - Type: ADDITIVE (state-independent)
    - Dimension: nw = 3 (three Wiener processes)
    - Correlation: DIAGONAL (independent noise sources)

    Comparison with Deterministic
    ------------------------------
    **Deterministic:**
    - Single trajectory
    - Perfect prediction
    - Nominal design

    **Stochastic:**
    - Ensemble of trajectories
    - Probabilistic prediction
    - Robust design

    **When Stochastic is Necessary:**
    - Real process has significant noise
    - Robust control needed
    - Risk assessment required
    - Parameter uncertainty significant

    Limitations
    -----------
    - Additive noise only (not multiplicative)
    - Constant noise intensity (not state-dependent)
    - Independent noise sources (no correlation)
    - No jumps (only continuous Brownian paths)

    Extensions
    ----------
    - Multiplicative noise: g(x) = diag(σ_A·C_A, σ_B·C_B, σ_T·T)
    - Correlated noise: Full covariance matrix
    - Jump diffusion: Add Poisson jumps
    - Parameter uncertainty: Random walk parameters

    See Also
    --------
    ContinuousBatchReactor : Deterministic version
    OrnsteinUhlenbeck : Mean-reverting stochastic process
    GeometricBrownianMotion : Multiplicative noise example
    """

    def define_system(
        self,
        k1_val: float = 0.5,
        k2_val: float = 0.3,
        E1_val: float = 1000.0,
        E2_val: float = 1500.0,
        alpha_val: float = 0.1,
        T_amb_val: float = 300.0,
        sigma_A: float = 0.01,
        sigma_B: float = 0.01,
        sigma_T: float = 1.0,
        C_A0: Optional[float] = None,
        T0: Optional[float] = None,
    ):
        """
        Define stochastic batch reactor dynamics.

        Parameters
        ----------
        k1_val : float
            Pre-exponential factor for A→B reaction [1/s]
        k2_val : float
            Pre-exponential factor for B→C reaction [1/s]
        E1_val : float
            Activation energy for reaction 1 [K]
        E2_val : float
            Activation energy for reaction 2 [K]
        alpha_val : float
            Heat transfer coefficient [1/s]
        T_amb_val : float
            Ambient temperature [K]

        sigma_A : float, default=0.01
            Concentration noise intensity for A [mol/(L·√s)]
            - Typical: 0.001-0.1 mol/(L·√s)
            - Should be << C_A0 for realistic noise
            - Rule of thumb: σ_A ~ 0.01·C_A0

        sigma_B : float, default=0.01
            Concentration noise intensity for B [mol/(L·√s)]
            - Same magnitude as σ_A typically
            - Represents mixing and kinetic uncertainty

        sigma_T : float, default=1.0
            Temperature noise intensity [K/√s]
            - Typical: 0.1-5.0 K/√s
            - More critical than concentration noise
            - Affects reaction rates exponentially
            - Rule of thumb: σ_T ~ 1 K/√s

        C_A0, T0 : Optional[float]
            Initial conditions for equilibrium setup

        Notes
        -----
        **Drift (Deterministic Part):**
        Identical to deterministic reactor:
            f(x, u) = [-r₁, r₁ - r₂, Q - α·(T - T_amb)]ᵀ

        **Diffusion (Stochastic Part):**
        Diagonal matrix (additive, independent noise):
            g(x, u) = diag(σ_A, σ_B, σ_T)

        This gives three independent Wiener processes driving
        concentration and temperature fluctuations.

        **Noise Intensity Guidelines:**

        For 1% relative noise at C_A0 = 1.0 mol/L over 1 second:
            σ_A ~ 0.01 mol/(L·√s)

        For 1 K standard deviation over 1 second:
            σ_T ~ 1.0 K/√s

        **Physical Justification:**

        Additive noise models:
        - External disturbances (feed, ambient)
        - Measurement errors (sensors)
        - Control system imperfections
        - Unmodeled dynamics (lumped effects)

        Alternative: Multiplicative noise would be:
            g(x) = diag(σ_A·C_A, σ_B·C_B, σ_T·T)

        This models relative errors scaling with state magnitude.

        **Validation:**

        Check noise levels are reasonable:
        1. Run deterministic + stochastic simulations
        2. Compare final states
        3. Coefficient of variation should be 5-20%:
               CV = std(C_B_final) / mean(C_B_final)
        4. If CV > 30%: Noise too large
        5. If CV < 1%: Noise negligible
        """
        # Store initial conditions
        self.C_A0 = C_A0
        self.T0 = T0

        # State and control variables
        C_A, C_B, T = sp.symbols("C_A C_B T", real=True, positive=True)
        Q = sp.symbols("Q", real=True)

        # Parameters (kinetics and heat transfer)
        k1, k2, E1, E2, alpha, T_amb = sp.symbols(
            "k1 k2 E1 E2 alpha T_amb", real=True, positive=True
        )

        # Noise intensities
        sigma_A_sym = sp.symbols("sigma_A", real=True, positive=True)
        sigma_B_sym = sp.symbols("sigma_B", real=True, positive=True)
        sigma_T_sym = sp.symbols("sigma_T", real=True, positive=True)

        self.parameters = {
            k1: k1_val,
            k2: k2_val,
            E1: E1_val,
            E2: E2_val,
            alpha: alpha_val,
            T_amb: T_amb_val,
            sigma_A_sym: sigma_A,
            sigma_B_sym: sigma_B,
            sigma_T_sym: sigma_T,
        }

        self.state_vars = [C_A, C_B, T]
        self.control_vars = [Q]
        self.output_vars = []
        self.order = 1

        # Reaction rates (Arrhenius kinetics)
        r1 = k1 * C_A * sp.exp(-E1 / T)
        r2 = k2 * C_B * sp.exp(-E2 / T)

        # DRIFT (Deterministic part - same as deterministic reactor)
        dC_A_dt = -r1
        dC_B_dt = r1 - r2
        dT_dt = Q - alpha * (T - T_amb)

        self._f_sym = sp.Matrix([dC_A_dt, dC_B_dt, dT_dt])

        # DIFFUSION (Stochastic part - additive noise)
        # Diagonal matrix: three independent Wiener processes
        self.diffusion_expr = sp.Matrix([
            [sigma_A_sym, 0, 0],
            [0, sigma_B_sym, 0],
            [0, 0, sigma_T_sym]
        ])

        # Itô SDE interpretation
        self.sde_type = "ito"

        # Output: Full state measurement (with potential noise in practice)
        self._h_sym = sp.Matrix([C_A, C_B, T])

    def setup_equilibria(self):
        """
        Set up equilibrium points.

        Note: For stochastic systems, "equilibrium" refers to deterministic
        part only. Actual trajectories fluctuate around this point.
        """
        # Get parameters
        T_amb = self.parameters[sp.symbols("T_amb")]

        # Complete conversion equilibrium (deterministic part)
        self.add_equilibrium(
            "complete",
            x_eq=np.array([0.0, 0.0, T_amb]),
            u_eq=np.array([0.0]),
            verify=True,
            stability="stable",
            notes="Equilibrium of deterministic part (drift). Stochastic trajectories "
                  "fluctuate around this point with variance growing over time."
        )

        # Initial condition (if provided)
        if self.C_A0 is not None and self.T0 is not None:
            alpha = self.parameters[sp.symbols("alpha")]
            Q_init = alpha * (self.T0 - T_amb)

            self.add_equilibrium(
                "initial",
                x_eq=np.array([self.C_A0, 0.0, self.T0]),
                u_eq=np.array([Q_init]),
                verify=False,
                stability="unstable",
                notes=f"Initial state setpoint (deterministic part). Stochastic trajectories "
                      f"will fluctuate with std ~ [σ_A·√t, σ_B·√t, σ_T·√t]."
            )
            self.set_default_equilibrium("initial")
        else:
            self.set_default_equilibrium("complete")

    def get_noise_intensities(self) -> dict:
        """
        Get current noise intensity parameters.

        Returns
        -------
        dict
            Dictionary with keys 'sigma_A', 'sigma_B', 'sigma_T'

        Examples
        --------
        >>> reactor = ContinuousStochasticBatchReactor(
        ...     sigma_A=0.01, sigma_B=0.01, sigma_T=1.0
        ... )
        >>> noise = reactor.get_noise_intensities()
        >>> print(f"Temperature noise: {noise['sigma_T']} K/√s")
        """
        return {
            'sigma_A': self.parameters[sp.symbols('sigma_A')],
            'sigma_B': self.parameters[sp.symbols('sigma_B')],
            'sigma_T': self.parameters[sp.symbols('sigma_T')],
        }

    def estimate_variance_growth(self, t: float) -> np.ndarray:
        """
        Estimate variance growth for additive noise (approximate).

        For additive noise with independent sources:
            Var[X(t)] ≈ Var[X(0)] + diag(σ²)·t

        This is exact for linear systems, approximate for nonlinear.

        Parameters
        ----------
        t : float
            Time [s]

        Returns
        -------
        np.ndarray
            Estimated variance [Var(C_A), Var(C_B), Var(T)]

        Notes
        -----
        This is a rough estimate. For accurate statistics, run Monte Carlo.

        Examples
        --------
        >>> reactor = ContinuousStochasticBatchReactor(
        ...     sigma_A=0.01, sigma_B=0.01, sigma_T=1.0
        ... )
        >>> var_100s = reactor.estimate_variance_growth(t=100)
        >>> std_100s = np.sqrt(var_100s)
        >>> print(f"Estimated std after 100s: C_A={std_100s[0]:.3f}, "
        ...       f"C_B={std_100s[1]:.3f}, T={std_100s[2]:.3f}")
        """
        noise = self.get_noise_intensities()
        sigma_vec = np.array([noise['sigma_A'], noise['sigma_B'], noise['sigma_T']])
        return sigma_vec**2 * t

    def compute_signal_to_noise_ratio(self, x: np.ndarray, t: float) -> np.ndarray:
        """
        Compute signal-to-noise ratio for each state.

        SNR = |x| / (σ·√t)

        Higher SNR → noise less significant.

        Parameters
        ----------
        x : np.ndarray
            Current state [C_A, C_B, T]
        t : float
            Time since start [s]

        Returns
        -------
        np.ndarray
            SNR for [C_A, C_B, T]

        Notes
        -----
        SNR > 10: Noise negligible
        SNR ~ 1: Noise significant
        SNR < 0.1: Noise dominates

        Examples
        --------
        >>> reactor = ContinuousStochasticBatchReactor()
        >>> x = np.array([0.5, 0.3, 360.0])
        >>> snr = reactor.compute_signal_to_noise_ratio(x, t=10)
        >>> print(f"SNR: C_A={snr[0]:.1f}, C_B={snr[1]:.1f}, T={snr[2]:.1f}")
        """
        noise = self.get_noise_intensities()
        sigma_vec = np.array([noise['sigma_A'], noise['sigma_B'], noise['sigma_T']])
        
        if t <= 0:
            return np.inf * np.ones(3)
        
        return np.abs(x) / (sigma_vec * np.sqrt(t))


# Convenience function
def create_batch_reactor_with_noise(
    noise_level: str = 'medium',
    **kwargs
) -> ContinuousStochasticBatchReactor:
    """
    Create stochastic batch reactor with predefined noise levels.

    Parameters
    ----------
    noise_level : str, default='medium'
        Noise level: 'low', 'medium', or 'high'
    **kwargs
        Additional parameters passed to constructor

    Returns
    -------
    ContinuousStochasticBatchReactor

    Examples
    --------
    >>> # Low noise (precise control)
    >>> reactor_precise = create_batch_reactor_with_noise('low')
    >>> 
    >>> # High noise (pilot scale, poor control)
    >>> reactor_noisy = create_batch_reactor_with_noise('high')
    """
    noise_presets = {
        'low': {'sigma_A': 0.001, 'sigma_B': 0.001, 'sigma_T': 0.1},
        'medium': {'sigma_A': 0.01, 'sigma_B': 0.01, 'sigma_T': 1.0},
        'high': {'sigma_A': 0.1, 'sigma_B': 0.1, 'sigma_T': 5.0},
    }
    
    if noise_level not in noise_presets:
        raise ValueError(f"noise_level must be 'low', 'medium', or 'high', got {noise_level}")
    
    # Merge noise preset with user kwargs
    params = {**noise_presets[noise_level], **kwargs}
    
    return ContinuousStochasticBatchReactor(**params)
            

"""
Continuous Stochastic CSTR - Multiple Equilibria with Continuous-Time Noise
============================================================================

This module provides a continuous-time stochastic differential equation model
of a Continuous Stirred-Tank Reactor (CSTR), combining nonlinear dynamics with
continuous Brownian noise. This represents the theoretical foundation for:

- Stochastic bifurcation theory and noise-induced phenomena
- Continuous-time optimal control under uncertainty (stochastic HJB equations)
- Fokker-Planck analysis of stationary distributions
- Large deviations theory and rare event analysis
- Exit time problems and first passage theory

The continuous stochastic CSTR is the "ground truth" model from which
discrete versions are derived, providing the mathematical foundation for
understanding noise effects on multi-stable chemical reactors.

Physical Motivation
-------------------

**Why Continuous-Time Stochastic Model?**

Real chemical reactors experience continuous-time disturbances:
- Turbulent mixing: Continuous fluctuations in local concentration
- Heat transfer variations: Continuous ambient temperature changes
- Feed composition: Continuous upstream process noise
- Catalyst activity: Continuous aging and regeneration
- Molecular-level stochasticity: Continuous thermal fluctuations

These disturbances don't occur at discrete sampling instants but
continuously in time, making continuous SDE the natural model.

**Advantages of Continuous SDE:**
1. **Theoretical rigor:** Exact representation of continuous noise
2. **Fokker-Planck equation:** Analytical stationary distributions
3. **Large deviations:** Analytical rare event probabilities
4. **Stochastic control theory:** HJB equations, optimal stopping
5. **Foundation for discretization:** Provides ground truth

**When to Use Continuous vs Discrete:**

Continuous Stochastic:
- Theoretical analysis (Fokker-Planck, large deviations)
- Controller design (continuous-time HJB)
- Fundamental understanding (noise-induced phenomena)
- High sampling rates (Δt → 0 limit)

Discrete Stochastic:
- Digital implementation (PLC, DCS)
- Discrete Kalman filtering
- Discrete-time MPC
- Reinforcement learning

Mathematical Formulation
-------------------------

**Stochastic Differential Equations (Itô form):**

    dC_A = [(F/V)·(C_A_feed - C_A) - r]·dt + σ_C·dW_C
    dT = [(F/V)·(T_feed - T) + q_gen + q_removal]·dt + σ_T·dW_T

where:
    - r = k₀·C_A·exp(-E/T): Reaction rate (Arrhenius)
    - q_gen = (-ΔH/ρC_p)·r: Heat generation (exothermic)
    - q_removal = (UA/VρC_p)·(T_jacket - T): Jacket cooling
    - σ_C: Concentration noise intensity [mol/(L·√s)]
    - σ_T: Temperature noise intensity [K/√s]
    - W_C, W_T: Independent Wiener processes (Brownian motions)
    - dW ~ N(0, dt): Brownian motion increments

**Drift Vector:**
    f(X, u) = [(F/V)·(C_A_feed - C_A) - r,
               (F/V)·(T_feed - T) + q_gen + q_removal]ᵀ

Identical to deterministic CSTR.

**Diffusion Matrix:**
    g(X, u) = diag(σ_C, σ_T)

Additive noise (state-independent), diagonal (uncorrelated).

**Itô Interpretation:**
Standard interpretation in finance and engineering. Alternative
Stratonovich would modify drift by (1/2)·g·(∂g/∂X).

Fokker-Planck Equation
-----------------------

The probability density p(C_A, T, t) evolves according to the
**Fokker-Planck equation** (forward Kolmogorov equation):

    ∂p/∂t = -∇·(f·p) + (1/2)·∇·∇·(g·gᵀ·p)

Expanded:
    ∂p/∂t = -∂(f_C·p)/∂C_A - ∂(f_T·p)/∂T
            + (σ_C²/2)·∂²p/∂C_A² + (σ_T²/2)·∂²p/∂T²

**Stationary Distribution:**

At steady state (∂p/∂t = 0), the Fokker-Planck gives:
    p_∞(C_A, T) ∝ exp(-2·Φ(C_A, T)/σ²)

where Φ is the "potential function" (quasi-potential).

**For CSTR with Multiple Equilibria:**

Stationary distribution is **bimodal** (or multimodal):
- Peaks at each stable equilibrium
- Valleys at unstable equilibria (saddle points)
- Relative peak heights depend on basin sizes and noise intensity

**Key Insight:**
Even with noise, system spends most time near deterministic
stable equilibria, but occasionally transitions between them.

Large Deviations and Rare Events
---------------------------------

**Large Deviations Principle:**

For small noise σ → 0:
    P(rare event) ≈ exp(-S/σ²)

where S is the **action functional** (minimum "cost" path to event).

**For CSTR Transitions:**

Probability of escaping from basin of attraction:
    P(escape) ≈ exp(-ΔV/σ_T²)

where ΔV is "potential barrier" (related to saddle point height).

**Mean First Passage Time:**

Expected time to first reach boundary ∂D from x₀ ∈ D:
    E[τ_escape] ≈ (2π/λ_1)·exp(ΔV/σ²)

where λ_1 is eigenvalue of linearization at saddle.

**Implications:**
- Small noise: Exponentially long residence time (very stable)
- Large noise: Frequent transitions (unreliable operation)
- Temperature noise most critical (in exponent via Arrhenius coupling)

**Kramers' Rate:**

Classical result from chemical physics:
    Rate = (ω_well·ω_barrier/2π)·exp(-ΔV/k_B·T)

Applied to CSTR: Noise-induced transition rate.

Stochastic Bifurcation Theory
------------------------------

Unlike deterministic bifurcations (equilibrium count changes),
stochastic bifurcations involve qualitative changes in probability
distributions.

**P-Bifurcation (Phenomenological):**

Changes in stationary distribution shape as parameters vary:
- Unimodal → Bimodal: Two peaks emerge
- Bimodal → Unimodal: Peaks merge
- Occurs at different parameter values than deterministic bifurcations

**D-Bifurcation (Dynamic):**

Changes in sign of top Lyapunov exponent:
- From negative (stable) to positive (unstable)
- Noise can stabilize or destabilize

**For CSTR:**

As σ_T increases:
- Small σ_T: Bimodal distribution (two basins well-separated)
- Medium σ_T: Peaks broaden, overlap increases
- Large σ_T: Peaks merge, essentially unimodal
- Critical σ_T: Transition in qualitative behavior

**Noise-Induced Oscillations:**

Even if deterministic system has stable steady state, noise
can create oscillations (stochastic resonance).

For CSTR: Near Hopf bifurcation, small noise can induce
coherent oscillations.

Stochastic Optimal Control
---------------------------

**Continuous-Time HJB Equation:**

Value function V(x,t) satisfies Hamilton-Jacobi-Bellman:
    -∂V/∂t = min_u [L(x,u) + (∂V/∂x)ᵀ·f + (1/2)·tr(gᵀ·∂²V/∂x²·g)]

where:
- L: Running cost (quadratic typically)
- f: Drift
- g: Diffusion
- Second-order term: Effect of noise on value function

**Optimal Control:**
Feedback form: u*(x,t) = argmin[...]

In general, must solve PDE (HJB). For special cases (LQG),
explicit solution exists.

**Risk-Sensitive Control:**

Exponential cost functional:
    J = -ln E[exp(-θ·∫L dt)]

Parameter θ controls risk aversion:
- θ = 0: Risk-neutral (standard LQG)
- θ > 0: Risk-averse (penalize variance)
- θ < 0: Risk-seeking

**Chance-Constrained Control:**

Probabilistic constraints:
    P(g(X(t)) ≤ 0) ≥ 1 - ε for all t

Example: P(T < T_max) ≥ 0.99

Difficult to solve analytically; use scenario-based or sampling MPC.

Numerical Integration
---------------------

**SDE Integration Methods:**

1. **Euler-Maruyama (Most Common):**
   X[k+1] = X[k] + f(X[k], u)·Δt + g·√Δt·Z[k]
   
   - First-order convergence
   - Simple, robust
   - Recommended for CSTR

2. **Milstein (Higher Order):**
   Adds correction: +(1/2)·g·(∂g/∂X)·(Z² - 1)·Δt
   
   - For additive noise: Same as Euler-Maruyama
   - For multiplicative: Improves accuracy

3. **Stochastic Runge-Kutta:**
   - Higher-order schemes
   - More computational cost
   - Better for very smooth problems

**For CSTR:**
- Use Euler-Maruyama with dt = 0.01-0.1 s
- Check convergence: Halve dt, verify statistics unchanged
- Monitor positivity: C_A ≥ 0, T > 0

**Stiffness:**
CSTR can be moderately stiff. For stiff SDEs:
- Use implicit methods if available
- Or smaller dt with explicit methods
- Framework supports stiff SDE solvers

Monte Carlo Analysis
--------------------

**Ensemble Simulation:**

Run N = 100-10,000 independent trajectories to characterize:

1. **Stationary Distribution:**
   - Histogram at large t (after transient)
   - Should match Fokker-Planck prediction
   - Check for bimodality

2. **Transition Statistics:**
   - Count escapes from basin
   - Estimate transition rate
   - Validate large deviations prediction

3. **Mean and Variance:**
   - Temporal evolution of moments
   - Variance grows from noise accumulation
   - Eventually reaches stationary values

4. **Probability of Safety:**
   - Fraction satisfying constraints
   - Tail probabilities (rare events)

**Importance Sampling:**

For rare events (P < 0.01):
- Bias toward rare region via measure change
- Reweight samples for unbiased estimate
- Vastly more efficient than naive Monte Carlo

Applications
------------

**1. Fundamental Understanding:**
- Noise-induced transitions mechanisms
- Stochastic bifurcations
- Stationary distributions (Fokker-Planck)
- Optimal exit paths (instanton theory)

**2. Continuous-Time Controller Design:**
- Stochastic LQG (linear-quadratic-Gaussian)
- HJB-based optimal control
- Risk-sensitive control
- Barrier certificates

**3. Filter Design:**
- Continuous-time Kalman-Bucy filter
- Zakai equation (nonlinear filtering)
- Path integral formulation

**4. Reliability Engineering:**
- Mean first passage time calculations
- Failure probability estimation
- Preventive maintenance scheduling

**5. Process Design:**
- Noise tolerance analysis
- Robust parameter selection
- Safety margin determination

**6. Validation:**
- Ground truth for discrete models
- Verify discretization accuracy
- Benchmark for approximate methods

Comparison with Discrete Stochastic
------------------------------------

**Continuous Stochastic CSTR:**
- dX = f·dt + g·dW (SDE)
- Noise intensity: [state]/√[time]
- Fokker-Planck equation
- Theoretical foundation

**Discrete Stochastic CSTR:**
- X[k+1] = f + w[k]
- Noise variance: [state]²
- Master equation (discrete)
- Implementation ready

**Conversion:**
Discrete from continuous: σ_d = σ_c·√Δt

**Use Cases:**
- Continuous: Theory, design, high-rate sampling
- Discrete: Implementation, Kalman filter, MPC

Limitations
-----------
- Additive noise only (not multiplicative)
- Constant noise intensity (not state/time-dependent)
- Independent noise sources (no correlation)
- Gaussian noise (not heavy-tailed)
- No jumps (only continuous Brownian paths)

Extensions
----------
- Multiplicative noise: g(X) = diag(σ_C·C_A, σ_T·T)
- Correlated noise: Full 2×2 diffusion matrix
- Colored noise: Ornstein-Uhlenbeck driving process
- Jump diffusion: Poisson jumps for faults
- Parameter uncertainty: θ(t) follows SDE
- Time-varying noise: σ(t) deterministic or stochastic

"""


class ContinuousStochasticCSTR(ContinuousStochasticSystem):
    """
    Continuous-time stochastic CSTR with multiple equilibria and Brownian noise.

    Provides the continuous-time SDE formulation of the CSTR, combining
    nonlinear dynamics (multiple steady states, bifurcations) with continuous
    Brownian noise. This is the theoretical foundation for stochastic analysis,
    optimal control, and rare event estimation.

    Stochastic Differential Equations
    ----------------------------------
    Itô SDE form:

        dC_A = [(F/V)·(C_A_feed - C_A) - r]·dt + σ_C·dW_C
        dT = [(F/V)·(T_feed - T) + q_gen + q_removal]·dt + σ_T·dW_T

    where:
        - r = k₀·C_A·exp(-E/T): Reaction rate (Arrhenius)
        - q_gen = (-ΔH/ρC_p)·r: Heat generation
        - q_removal = (UA/VρC_p)·(T_jacket - T): Heat removal
        - σ_C: Concentration noise intensity [mol/(L·√s)]
        - σ_T: Temperature noise intensity [K/√s]
        - W_C(t), W_T(t): Independent Wiener processes

    Physical Interpretation
    -----------------------
    **Continuous-Time Disturbances:**

    Unlike discrete models where noise occurs at sampling instants,
    continuous noise represents:
    - Turbulent fluctuations (continuous)
    - Ambient variations (continuous)
    - Molecular stochasticity (continuous at microscale)
    - Unmodeled fast dynamics (continuous effective noise)

    **Noise Intensities:**

    1. **σ_C [mol/(L·√s)]:**
       - Feed composition fluctuations
       - Mixing imperfections (macro-mixing time scale)
       - Sampling variability
       - Typical: 0.0001-0.01 mol/(L·√s)

    2. **σ_T [K/√s]:**
       - Heat transfer coefficient variations
       - Ambient temperature changes
       - Jacket flow rate fluctuations
       - Most critical (exponential coupling)
       - Typical: 0.1-5.0 K/√s

    **Why Temperature Noise Dominates:**

    Arrhenius exponential sensitivity:
        ∂r/∂T = r·(E/T²)

    At T = 390 K with E = 8750 K:
        ∂r/∂T ≈ 0.058·r per K

    Temperature noise amplified exponentially through reaction rate,
    creating strong coupling to concentration dynamics.

    Multiple Steady States with Noise
    ----------------------------------
    **Deterministic Equilibria:**
    CSTR can have 1, 2, or 3 steady states (saddle-node bifurcation).

    **Stochastic Equilibria:**
    With noise, "equilibria" become probability distributions:
    - Stationary distribution p_∞(C_A, T) from Fokker-Planck
    - May be bimodal (two peaks at stable equilibria)
    - Transitions between basins via noise

    **Noise-Induced Transitions:**

    Even from stable equilibrium, noise can cause escape:
    - Fluctuations occasionally reach saddle point
    - Once over barrier, fall into other basin
    - Rare but catastrophic for operation

    **Mean First Passage Time:**

    Expected time to escape from basin:
        E[τ_escape] ≈ (2π/ω)·exp(ΔV/(σ_T²))

    where:
    - ΔV: Potential barrier (related to saddle height)
    - ω: Frequency at bottom of well (linearization eigenvalue)

    **Critical Noise Level:**

    σ_crit where transitions become frequent (τ_escape ~ operation time).

    For typical CSTR:
        σ_T_crit ~ 1-5 K/√s

    Above this, operation at high-conversion becomes unreliable.

    Fokker-Planck Analysis
    -----------------------
    **Stationary Distribution:**

    For Itô SDE: dX = f·dt + g·dW

    Stationary density satisfies:
        0 = -∇·(f·p_∞) + (1/2)·∇·∇·(D·p_∞)

    where D = g·gᵀ is diffusion matrix.

    **For CSTR:**
        D = diag(σ_C², σ_T²)

    In 2D, this is a PDE for p_∞(C_A, T).

    **Quasi-Potential:**

    For small noise:
        p_∞ ∝ exp(-2·Φ/σ²)

    where Φ satisfies:
        f·∇Φ - (1/2)·tr(D·∇∇Φ) = 0

    **Interpretation:**
    - Φ is like "energy" or "potential"
    - Minima at stable equilibria
    - Maxima at unstable equilibria (saddle points)
    - System prefers low-Φ regions

    **Computing Stationary Distribution:**

    Methods:
    1. **Long-time simulation:** Histogram after transient
    2. **Fokker-Planck solver:** Finite difference/element on PDE
    3. **Path integral:** Monte Carlo on action functional

    Stochastic Stability
    ---------------------

    **Different from Deterministic Stability:**

    Deterministic: Eigenvalues of Jacobian at equilibrium
    - All Re(λ) < 0 → stable
    - Any Re(λ) > 0 → unstable

    Stochastic: Lyapunov exponent of SDE
    - λ_L = lim_{t→∞} (1/t)·E[ln||δX(t)||]
    - λ_L < 0 → stable (perturbations decay)
    - λ_L > 0 → unstable (perturbations grow)

    **Noise Can Stabilize or Destabilize:**
    - Usually: Noise destabilizes (makes λ_L less negative)
    - Rarely: Noise stabilizes (noise-induced stability)

    **For CSTR:**
    - High-conversion state: Moderately stable deterministically
    - With noise: Stability margin reduced
    - Large σ_T can make effectively unstable (frequent escapes)

    Optimal Control Under Uncertainty
    ----------------------------------

    **Stochastic HJB Equation:**

    For infinite-horizon problem:
        0 = min_u [L(x,u) + (∂V/∂x)ᵀ·f + (1/2)·tr(gᵀ·∂²V/∂x²·g)]

    Optimal control: u*(x) from minimization.

    **For CSTR:**
    - Maintain high-conversion despite noise
    - Tradeoff: Performance vs robustness
    - May require backing away from optimal deterministic point

    **Risk-Sensitive Control:**

        J_θ = -ln E[exp(-θ·∫₀^∞ L dt)]

    Adjusts conservativeness:
    - Small θ: Nearly risk-neutral
    - Large θ: Very risk-averse (stay away from transitions)

    **Exit Time Control:**

    Minimize: E[∫₀^τ L dt]

    where τ is first exit time from safe region.

    Maximizes time until failure/transition.

    State Space
    -----------
    State: x = [C_A, T] ∈ ℝ₊ × ℝ₊
        - Stochastic processes (not deterministic functions)
        - Multiple modes possible (bimodal distribution)

    Control: u = T_jacket ∈ ℝ₊
        - Deterministic control (no noise in actuation)

    Noise: w = [w_C, w_T]
        - Independent Wiener processes
        - Continuous-time (Brownian motion)

    Parameters
    ----------
    F, V, C_A_feed, T_feed, k0, E, delta_H, rho, Cp, UA : float
        Same as deterministic CSTR (see ContinuousCSTR)

    sigma_C : float, default=0.001
        Concentration noise intensity [mol/(L·√s)]
        - Continuous-time units: per √s
        - Typical: 0.0001-0.01 mol/(L·√s)
        - Conversion to discrete: σ_d = σ_c·√Δt

    sigma_T : float, default=1.0
        Temperature noise intensity [K/√s]
        - Continuous-time units: per √s
        - Typical: 0.1-5.0 K/√s
        - Most critical parameter
        - Determines transition rates

    Stochastic Properties
    ---------------------
    - Noise Type: ADDITIVE (state-independent)
    - SDE Type: Itô (standard interpretation)
    - Noise Dimension: nw = 2
    - Correlation: DIAGONAL (independent)
    - Stationary: Yes (Fokker-Planck stationary distribution)
    - Ergodic: Yes (time averages = ensemble averages)

    Applications
    ------------
    **1. Theoretical Analysis:**
    - Fokker-Planck equation (stationary distribution)
    - Large deviations (rare events)
    - Stochastic bifurcations
    - Exit time problems

    **2. Continuous-Time Control:**
    - Stochastic HJB equation
    - Risk-sensitive control
    - Optimal stopping
    - Barrier certificates

    **3. Nonlinear Filtering:**
    - Zakai equation (unnormalized density)
    - Duncan-Mortensen-Zakai equation
    - Path integral formulation

    **4. Reliability Analysis:**
    - Mean first passage time
    - Transition rate estimation
    - Safety verification

    **5. Validation:**
    - Ground truth for discrete models
    - Benchmark for approximate methods

    Numerical Integration
    ---------------------
    **Recommended Methods:**
    - Euler-Maruyama: dt = 0.01-0.1 s
    - Milstein: Same as Euler for additive noise
    - Framework stiff solvers: For stiff CSTR

    **Convergence Check:**
    - Halve dt, verify moments unchanged
    - Weak convergence: E[X], Var[X]
    - Strong convergence: Sample paths

    Monte Carlo Guidelines
    -----------------------
    **Sample Size:**
    - Mean/variance: N = 100-1,000
    - Rare events (P ~ 0.01): N = 10,000-100,000
    - Use importance sampling for efficiency

    **Statistics:**
    - Mean trajectory: μ(t) = (1/N)·Σ X_i(t)
    - Variance: σ²(t) = (1/N)·Σ (X_i(t) - μ(t))²
    - Percentiles: 5th, 50th, 95th

    Comparison with Other Models
    -----------------------------
    **vs. Deterministic CSTR:**
    - Adds process noise
    - Enables reliability analysis
    - Captures transitions

    **vs. Discrete Stochastic CSTR:**
    - Continuous time (theoretical)
    - Noise in [state]/√[time]
    - Fokker-Planck equation

    **vs. Stochastic Batch Reactor:**
    - CSTR: Multiple equilibria, continuous operation
    - Batch: Transient, finite time

    Limitations
    -----------
    - Additive noise (not multiplicative)
    - Constant noise (not state-dependent)
    - No jumps (only continuous paths)
    - Computational cost (Monte Carlo expensive)

    See Also
    --------
    DiscreteStochasticCSTR : Discrete-time version
    ContinuousCSTR : Deterministic version
    OrnsteinUhlenbeck : Simple mean-reverting SDE
    """

    def define_system(
        self,
        F_val: float = 100.0,
        V_val: float = 100.0,
        C_A_feed_val: float = 1.0,
        T_feed_val: float = 350.0,
        k0_val: float = 7.2e10,
        E_val: float = 8750.0,
        delta_H_val: float = -5e4,
        rho_val: float = 1000.0,
        Cp_val: float = 0.239,
        UA_val: float = 5e4,
        sigma_C: float = 0.001,
        sigma_T: float = 1.0,
        x_ss: Optional[np.ndarray] = None,
        u_ss: Optional[np.ndarray] = None,
    ):
        """
        Define continuous-time stochastic CSTR dynamics.

        Parameters
        ----------
        F_val : float
            Volumetric flow rate [L/s]
        V_val : float
            Reactor volume [L]
        C_A_feed_val : float
            Feed concentration [mol/L]
        T_feed_val : float
            Feed temperature [K]
        k0_val : float
            Pre-exponential factor [1/s]
        E_val : float
            Activation energy [K] (dimensionless Eₐ/R)
        delta_H_val : float
            Heat of reaction [J/mol] (negative = exothermic)
        rho_val : float
            Density [kg/L]
        Cp_val : float
            Specific heat capacity [J/(kg·K)]
        UA_val : float
            Overall heat transfer coefficient × area [J/(s·K)]
        x_ss : Optional[np.ndarray]
            Steady-state [Cₐ, T] for equilibrium setup
        u_ss : Optional[np.ndarray]
            Steady-state [T_jacket] for equilibrium setup

        sigma_C : float, default=0.001
            Concentration noise intensity [mol/(L·√s)]
            - Continuous-time units: per √s
            - Typical: 0.0001-0.01 mol/(L·√s)
            - Smaller than batch reactor (continuous operation)

        sigma_T : float, default=1.0
            Temperature noise intensity [K/√s]
            - Continuous-time units: per √s
            - Typical: 0.1-5.0 K/√s
            - Determines transition rates (exponentially)
            - Critical parameter for reliability

        x_ss, u_ss : Optional[np.ndarray]
            Known steady state (if available)

        Notes
        -----
        **Noise Intensity Selection:**

        Physical reasoning:
        - σ_C ~ 0.001: Precise control, large reactor
        - σ_C ~ 0.01: Typical industrial
        - σ_T ~ 0.5: Good temperature control
        - σ_T ~ 2.0: Poor control, high variability

        **Temperature Noise Impact:**

        At high-conversion (T ≈ 390 K):
        - σ_T = 0.5 K/√s: Very stable, rare transitions
        - σ_T = 1.0 K/√s: Occasional transitions (hours)
        - σ_T = 2.0 K/√s: Frequent transitions (minutes)
        - σ_T = 5.0 K/√s: Very unstable, constant switching

        **Design Criterion:**

        Choose σ_T such that mean first passage time:
            τ_escape > 100·τ_operation

        Ensures reliable operation (99% success).

        **Conversion to Discrete:**

        For discrete model with sampling Δt:
            σ_discrete = σ_continuous·√Δt

        Example: σ_T = 1.0 K/√s, Δt = 5 s
            → σ_T_discrete = 1.0·√5 ≈ 2.24 K per step

        **Additive vs Multiplicative:**

        This uses additive (state-independent) noise.

        Alternative: Multiplicative noise
            g(X) = diag(σ_C·C_A, σ_T·T)

        Would represent:
        - Relative errors (percentage fluctuations)
        - State-dependent uncertainty
        - More complex analysis
        """
        # Store steady state
        self.x_ss = x_ss
        self.u_ss = u_ss

        # State and control variables
        C_A, T = sp.symbols("C_A T", real=True, positive=True)
        T_jacket = sp.symbols("T_jacket", real=True, positive=True)

        # Parameters
        F, V, C_A_feed, T_feed = sp.symbols("F V C_A_feed T_feed", real=True, positive=True)
        k0, E, delta_H, rho, Cp, UA = sp.symbols(
            "k0 E delta_H rho Cp UA", real=True, positive=True
        )

        # Noise intensities (continuous-time)
        sigma_C_sym = sp.symbols("sigma_C", real=True, positive=True)
        sigma_T_sym = sp.symbols("sigma_T", real=True, positive=True)

        self.parameters = {
            F: F_val,
            V: V_val,
            C_A_feed: C_A_feed_val,
            T_feed: T_feed_val,
            k0: k0_val,
            E: E_val,
            delta_H: delta_H_val,
            rho: rho_val,
            Cp: Cp_val,
            UA: UA_val,
            sigma_C_sym: sigma_C,
            sigma_T_sym: sigma_T,
        }

        self.state_vars = [C_A, T]
        self.control_vars = [T_jacket]
        self.output_vars = []
        self.order = 1

        # Reaction rate (Arrhenius)
        r = k0 * C_A * sp.exp(-E / T)

        # DRIFT (Deterministic part - same as deterministic CSTR)
        # Material balance
        dC_A_dt = (F / V) * (C_A_feed - C_A) - r

        # Energy balance
        dT_dt = (
            (F / V) * (T_feed - T)
            + ((-delta_H) / (rho * Cp)) * r
            + (UA / (V * rho * Cp)) * (T_jacket - T)
        )

        self._f_sym = sp.Matrix([dC_A_dt, dT_dt])

        # DIFFUSION (Stochastic part)
        # Diagonal: independent concentration and temperature noise
        self.diffusion_expr = sp.Matrix([
            [sigma_C_sym, 0],
            [0, sigma_T_sym]
        ])

        # Itô SDE
        self.sde_type = "ito"

        # Output
        self._h_sym = sp.Matrix([C_A, T])

    def setup_equilibria(self):
        """
        Set up equilibrium points (deterministic part).

        Note: These are centers of stationary distributions.
        Multiple equilibria may exist. Use find_steady_states().
        """
        if self.x_ss is not None and self.u_ss is not None:
            self.add_equilibrium(
                "steady_state",
                x_eq=self.x_ss,
                u_eq=self.u_ss,
                verify=True,
                stability="unknown",
                notes="Deterministic equilibrium. Fokker-Planck stationary distribution "
                      "peaks here. Multiple equilibria possible. Noise causes transitions."
            )
            self.set_default_equilibrium("steady_state")

    def get_noise_intensities(self) -> dict:
        """
        Get continuous-time noise intensities.

        Returns
        -------
        dict
            {'sigma_C': ..., 'sigma_T': ...}

        Notes
        -----
        Units: [state]/√[time]
        To convert to discrete: σ_d = σ_c·√Δt

        Examples
        --------
        >>> cstr = ContinuousStochasticCSTR(sigma_C=0.001, sigma_T=1.0)
        >>> noise = cstr.get_noise_intensities()
        >>> print(f"Temperature noise: {noise['sigma_T']} K/√s")
        """
        return {
            'sigma_C': self.parameters[sp.symbols('sigma_C')],
            'sigma_T': self.parameters[sp.symbols('sigma_T')],
        }

    def compute_residence_time(self) -> float:
        """
        Compute residence time τ = V/F [s].

        Returns
        -------
        float
            Residence time
        """
        F = self.parameters[sp.symbols("F")]
        V = self.parameters[sp.symbols("V")]
        return V / F

    def find_steady_states(
        self,
        T_jacket: float,
        T_range: tuple = (300.0, 500.0),
        n_points: int = 100,
    ) -> List[Tuple[float, float]]:
        """
        Find all steady states of deterministic part.

        These are centers of Fokker-Planck stationary distribution.

        Parameters
        ----------
        T_jacket : float
            Jacket temperature [K]
        T_range : tuple
            Search range
        n_points : int
            Initial guesses

        Returns
        -------
        List[Tuple[float, float]]
            [(C_A, T), ...] steady states

        Notes
        -----
        With noise, stationary distribution has peaks at these
        points (if stable) or valleys (if unstable).

        Examples
        --------
        >>> cstr = ContinuousStochasticCSTR()
        >>> states = cstr.find_steady_states(T_jacket=350.0)
        >>> print(f"Found {len(states)} steady states")
        """
        from scipy.optimize import fsolve

        F = self.parameters[sp.symbols("F")]
        V = self.parameters[sp.symbols("V")]
        C_A_feed = self.parameters[sp.symbols("C_A_feed")]
        T_feed = self.parameters[sp.symbols("T_feed")]
        k0 = self.parameters[sp.symbols("k0")]
        E = self.parameters[sp.symbols("E")]
        delta_H = self.parameters[sp.symbols("delta_H")]
        rho = self.parameters[sp.symbols("rho")]
        Cp = self.parameters[sp.symbols("Cp")]
        UA = self.parameters[sp.symbols("UA")]

        def equations(state):
            C_A, T = state
            if C_A < 0 or T < 250:
                return [1e10, 1e10]

            r = k0 * C_A * np.exp(-E / T)
            dC_A = (F / V) * (C_A_feed - C_A) - r
            dT = (
                (F / V) * (T_feed - T)
                + ((-delta_H) / (rho * Cp)) * r
                + (UA / (V * rho * Cp)) * (T_jacket - T)
            )
            return [dC_A, dT]

        steady_states = []
        for T_guess in np.linspace(T_range[0], T_range[1], n_points):
            r_g = k0 * C_A_feed * np.exp(-E / T_guess)
            C_A_g = C_A_feed / (1 + (V / F) * r_g / C_A_feed)
            C_A_g = np.clip(C_A_g, 0.0, C_A_feed)

            try:
                sol, info, ier, msg = fsolve(equations, [C_A_g, T_guess], full_output=True)
                if ier == 1:
                    C_A_s, T_s = sol
                    if (
                        0 <= C_A_s <= C_A_feed
                        and T_range[0] <= T_s <= T_range[1]
                        and not any(np.allclose([C_A_s, T_s], ss, rtol=1e-3) for ss in steady_states)
                    ):
                        steady_states.append((C_A_s, T_s))
            except:
                continue

        steady_states.sort(key=lambda x: x[1])
        return steady_states

    def estimate_escape_rate(
        self,
        x_basin: np.ndarray,
        barrier_height: float,
    ) -> float:
        """
        Estimate escape rate from basin using large deviations theory.

        Approximate formula:
            Rate ≈ (ω/2π)·exp(-ΔV/σ_T²)

        Parameters
        ----------
        x_basin : np.ndarray
            State in basin (stable equilibrium)
        barrier_height : float
            Potential barrier height (energy to saddle)

        Returns
        -------
        float
            Escape rate [1/s]

        Notes
        -----
        This is an approximation valid for small noise.
        For accurate rates, use Monte Carlo simulation.

        Examples
        --------
        >>> cstr = ContinuousStochasticCSTR(sigma_T=1.0)
        >>> # Approximate barrier height: 50 K² equivalent
        >>> rate = cstr.estimate_escape_rate(
        ...     x_basin=np.array([0.1, 390.0]),
        ...     barrier_height=50.0
        ... )
        >>> mean_time = 1.0 / rate
        >>> print(f"Mean escape time: {mean_time:.1f} s")
        """
        # Linearize at basin to get frequency
        A, B = self.linearize(x_basin, np.array([350.0]))
        eigenvalues = np.linalg.eigvals(A)
        omega = np.abs(np.min(np.real(eigenvalues)))  # Smallest (slowest) mode

        # Get temperature noise
        sigma_T = self.parameters[sp.symbols('sigma_T')]

        # Kramers-like formula
        rate = (omega / (2 * np.pi)) * np.exp(-barrier_height / sigma_T**2)

        return rate


# Convenience function
def create_continuous_stochastic_cstr_with_noise(
    noise_level: str = 'medium',
    **kwargs
) -> ContinuousStochasticCSTR:
    """
    Create continuous stochastic CSTR with predefined noise levels.

    Parameters
    ----------
    noise_level : str, default='medium'
        'low', 'medium', or 'high'
    **kwargs
        Additional parameters

    Returns
    -------
    ContinuousStochasticCSTR

    Notes
    -----
    Noise intensities in continuous-time units [state]/√[time].

    Examples
    --------
    >>> # Precise control (low noise)
    >>> cstr_precise = create_continuous_stochastic_cstr_with_noise('low')
    >>> 
    >>> # Typical industrial (medium noise)
    >>> cstr_typical = create_continuous_stochastic_cstr_with_noise('medium')
    >>> 
    >>> # Poor control (high noise)
    >>> cstr_noisy = create_continuous_stochastic_cstr_with_noise('high')
    """
    # Continuous-time noise intensities
    noise_presets = {
        'low': {'sigma_C': 0.0001, 'sigma_T': 0.1},
        'medium': {'sigma_C': 0.001, 'sigma_T': 1.0},
        'high': {'sigma_C': 0.01, 'sigma_T': 5.0},
    }
    
    if noise_level not in noise_presets:
        raise ValueError(f"noise_level must be 'low', 'medium', or 'high'")
    
    params = {**noise_presets[noise_level], **kwargs}
    
    return ContinuousStochasticCSTR(**params)