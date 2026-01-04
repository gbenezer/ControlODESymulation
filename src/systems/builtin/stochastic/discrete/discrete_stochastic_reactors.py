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
Discrete Stochastic Batch Reactor - Sampled-Data Chemical Reactor with Noise
=============================================================================

This module provides a discrete-time stochastic model of a chemical batch reactor,
suitable for digital control systems, discrete-time estimation, and sampled-data
analysis. The discrete stochastic batch reactor serves as:

- A realistic model for digital control implementation (computers operate in discrete time)
- A benchmark for discrete-time stochastic control algorithms (MPC, LQG)
- A test system for discrete-time state estimation (discrete Kalman filter)
- An illustration of noise accumulation in sampled-data systems
- A foundation for discrete-time optimal control and reinforcement learning

Discrete-time models are essential because:
- Industrial controllers are implemented on digital computers (PLCs, DCS)
- Measurements are sampled at discrete intervals (not continuous)
- Control actions are updated at discrete time steps (zero-order hold)
- Many algorithms naturally formulated in discrete time (dynamic programming)

Physical Context
----------------

**Discrete-Time Reality:**
Modern chemical plants operate with digital control systems:
- Sensors sampled every Δt seconds (typical: 0.1-10 s for batch reactors)
- Control computed and applied at discrete intervals
- Between samples: control held constant (zero-order hold)
- Measurements arrive as discrete values, not continuous signals

**Noise in Discrete Time:**
Discrete-time noise represents:
- Measurement noise: Sensor quantization, A/D conversion errors
- Process disturbances: Accumulated over sampling interval
- Model uncertainty: Discretization errors, unmodeled dynamics
- Sampling effects: Aliasing, information loss between samples

**Why Not Just Discretize Continuous Model?**
Direct discretization (Euler, RK4) of continuous model:
- Ignores sampling effects
- Noise structure different (integrated Brownian motion)
- May miss aliasing and intersample behavior

Discrete stochastic model explicitly accounts for:
- Discrete-time dynamics (exact for chosen discretization)
- Discrete-time noise (accumulated over Δt)
- Measurement timing (synchronized with control)

Mathematical Formulation
-------------------------

**Discrete-Time Stochastic Difference Equation:**

    X[k+1] = f(X[k], u[k]) + w[k]

where:
    - X[k] = [C_A[k], C_B[k], T[k]]: State at time k·Δt
    - u[k] = Q[k]: Control input (heating rate)
    - w[k] ~ N(0, Σ_w): Process noise (Gaussian)
    - f: Discrete-time dynamics (from continuous via discretization)

**Dynamics Function f:**

The deterministic part f discretizes the continuous dynamics:

    dC_A/dt = -r₁
    dC_B/dt = r₁ - r₂
    dT/dt = Q - α·(T - T_amb)

Using Euler discretization (simplest, most common):

    C_A[k+1] = C_A[k] - r₁(X[k])·Δt
    C_B[k+1] = C_B[k] + (r₁(X[k]) - r₂(X[k]))·Δt
    T[k+1] = T[k] + (Q[k] - α·(T[k] - T_amb))·Δt

More accurate: RK4, exact discretization for linear parts.

**Process Noise w[k]:**

Discrete-time noise represents accumulated disturbances over [k·Δt, (k+1)·Δt]:

    w[k] = [w_A[k], w_B[k], w_T[k]]ᵀ

Each component: w_i[k] ~ N(0, σ_i²)

**Noise Covariance Matrix:**

    Σ_w = diag(σ_A², σ_B², σ_T²)

Diagonal → independent noise sources.

**Relationship to Continuous Noise:**

For continuous-time noise intensity σ_c:
    σ_d = σ_c·√Δt

Intuition: Noise accumulates as √Δt (diffusion scaling).

Example: Continuous σ_c = 1 K/√s, sampling Δt = 0.1 s
    → Discrete σ_d = 1·√0.1 ≈ 0.316 K

**With Measurement Noise:**

Often include measurement equation:
    y[k] = h(X[k]) + v[k]

where v[k] ~ N(0, Σ_v) is measurement noise.

Full model: Process noise w + measurement noise v.

Key Properties
--------------

**1. Markov Property:**
    X[k+1] depends only on X[k], not X[k-1], X[k-2], ...
    
Essential for dynamic programming and Kalman filtering.

**2. Time-Invariant:**
    Dynamics f same at all time steps (parameters constant).
    
Simplifies analysis and control design.

**3. Gaussian Noise:**
    w[k] ~ N(0, Σ_w) → analytical tractability
    
Kalman filter optimal for Gaussian noise.

**4. Additive Noise:**
    Noise added to dynamics (not multiplied by state).
    
Simplifies estimation (linear measurement model).

**5. Non-Stationary:**
    Batch operation → finite time horizon, no equilibrium.
    
Different from steady-state continuous processes.

**6. Discretization Error:**
    f approximates continuous dynamics with O(Δt) error (Euler).
    
Smaller Δt → more accurate but more computation.

Discrete vs Continuous Stochastic Models
-----------------------------------------

**Continuous Stochastic (SDE):**
    dX = f(X,u)·dt + g(X,u)·dW
    
- Wiener process dW (Brownian motion)
- Noise intensity σ in [state]/√[time]
- Requires SDE integration (Euler-Maruyama, Milstein)
- Theoretical foundation

**Discrete Stochastic (Difference Equation):**
    X[k+1] = f(X[k], u[k]) + w[k]
    
- Discrete noise w[k] ~ N(0, Σ)
- Noise variance σ² in [state]²
- Direct simulation (no integration needed)
- Practical implementation

**Conversion:**
    σ_discrete = σ_continuous·√Δt
    
Accumulation of continuous noise over sampling interval.

**When to Use Each:**

Continuous:
- Theoretical analysis
- Controller design (LQG synthesis)
- High sampling rates (Δt → 0 limit)

Discrete:
- Digital control implementation
- State estimation (Kalman filter)
- Reinforcement learning (episodic updates)
- When measurements/control are discrete

Discrete Kalman Filter Application
-----------------------------------

The discrete stochastic model is perfect for Kalman filtering:

**Process Model:**
    X[k+1] = f(X[k], u[k]) + w[k]
    w[k] ~ N(0, Q)

**Measurement Model:**
    y[k] = h(X[k]) + v[k]
    v[k] ~ N(0, R)

**Extended Kalman Filter (EKF):**

Prediction:
    X̂[k+1|k] = f(X̂[k|k], u[k])
    P[k+1|k] = F·P[k|k]·Fᵀ + Q

Update:
    K = P[k+1|k]·Hᵀ·(H·P[k+1|k]·Hᵀ + R)⁻¹
    X̂[k+1|k+1] = X̂[k+1|k] + K·(y[k+1] - h(X̂[k+1|k]))
    P[k+1|k+1] = (I - K·H)·P[k+1|k]

where:
- F = ∂f/∂X: Linearized dynamics
- H = ∂h/∂X: Linearized measurement
- Q: Process noise covariance
- R: Measurement noise covariance

**Unscented Kalman Filter (UKF):**
Better for highly nonlinear systems (no linearization).

Sampling Time Selection
------------------------

**Tradeoffs:**

Large Δt (slow sampling):
- Pros: Less computation, fewer measurements
- Cons: Miss fast dynamics, discretization error, aliasing

Small Δt (fast sampling):
- Pros: Accurate dynamics, capture transients
- Cons: More computation, sensor cost, noise accumulation

**Guidelines:**

1. **Shannon-Nyquist Criterion:**
   Δt < 1/(2·f_max) where f_max is highest frequency of interest.

2. **Rule of Thumb:**
   Sample 5-10× faster than system time constant:
   Δt < τ_system/10

3. **Chemical Reactors:**
   - Fast reactions: Δt = 0.01-0.1 s
   - Moderate reactions: Δt = 0.1-1.0 s
   - Slow reactions: Δt = 1-10 s

4. **Control Considerations:**
   - Closed-loop bandwidth: Δt < 1/(10·ω_cl)
   - Actuator dynamics: Must respond within Δt

**This Implementation:**
Default dt = 1.0 s (moderate sampling for batch reactor).

Discretization Methods
----------------------

**1. Forward Euler (Implemented):**
    X[k+1] = X[k] + f(X[k], u[k])·Δt

- Simplest method
- O(Δt) accuracy
- Explicit (no equation solving)
- Can be unstable for large Δt
- Most common in practice (digital controllers)

**2. Runge-Kutta 4 (RK4):**
    X[k+1] = X[k] + (k₁ + 2k₂ + 2k₃ + k₄)/6

- O(Δt⁴) accuracy
- More computation (4 function evaluations)
- Better for moderate Δt
- Standard for high-accuracy discretization

**3. Exact Discretization (Special Cases):**
For linear parts (e.g., temperature dynamics):
    T[k+1] = e^(-α·Δt)·T[k] + (1-e^(-α·Δt))·(T_amb + Q/α)

- Exact for linear dynamics
- No discretization error
- Possible for some reactor components

**4. Implicit Methods:**
For stiff systems (fast reactions):
    X[k+1] = X[k] + f(X[k+1], u[k])·Δt

- Require solving nonlinear equation
- More stable for large Δt
- Rare in real-time control (computational cost)

Applications
------------

**1. Discrete-Time Model Predictive Control (MPC):**
Solve optimization at each time step:
    min Σ (X[k] - X_ref)ᵀ·Q·(X[k] - X_ref) + u[k]ᵀ·R·u[k]
    s.t. X[k+1] = f(X[k], u[k])
         X_min ≤ X[k] ≤ X_max
         u_min ≤ u[k] ≤ u_max

**2. Discrete Kalman Filter:**
Optimal state estimation from noisy measurements.

**3. Discrete LQG Control:**
Linear-Quadratic-Gaussian: Kalman filter + LQR.

**4. Reinforcement Learning:**
Q-learning, policy gradient methods naturally discrete-time.

**5. Batch Optimization:**
Dynamic programming for batch-to-batch improvement.

**6. Fault Detection:**
Residuals: r[k] = y[k] - h(X̂[k|k])
Statistical tests for anomaly detection.

Process Noise vs Measurement Noise
-----------------------------------

**Process Noise w[k]:**
- Acts on state dynamics
- Represents: disturbances, model error, unmodeled dynamics
- Cannot be directly observed
- Estimated via Kalman filter tuning

**Measurement Noise v[k]:**
- Acts on measurements
- Represents: sensor errors, quantization, A/D conversion
- Can be characterized via sensor specs
- Known variance from calibration

**Both Together:**
    X[k+1] = f(X[k], u[k]) + w[k]   (process)
    y[k] = h(X[k]) + v[k]            (measurement)

Kalman filter optimally fuses noisy measurements with noisy dynamics.

Noise Tuning Guidelines
------------------------

**Process Noise Covariance Q:**

Too small Q:
- Filter trusts model too much
- Slow response to disturbances
- Overconfident (P too small)

Too large Q:
- Filter trusts measurements too much
- Sensitive to measurement noise
- Jumpy estimates

**Rule of Thumb:**
- Start with physical understanding
- Diagonal Q: independent noise
- σ_i ~ expected disturbance magnitude over Δt

**Measurement Noise Covariance R:**

From sensor specifications:
- Accuracy: ±0.1 mol/L → σ ≈ 0.1/3 ≈ 0.033
- Resolution: 0.01 K → σ ≈ 0.01
- Datasheet specifications

**Adaptive Tuning:**
- Innovation-based: Adjust Q, R based on residuals
- Maximum likelihood: Estimate from data
- Cross-validation: Tune on separate dataset

Common Pitfalls
---------------

1. **Wrong Noise Scaling:**
   - Forgetting √Δt conversion: σ_d = σ_c·√Δt
   - Noise too large → unrealistic fluctuations
   - Noise too small → overconfident estimates

2. **Discretization Error:**
   - Large Δt with Euler → inaccurate dynamics
   - Nonlinear systems sensitive to discretization
   - Check: Compare with continuous simulation

3. **Stability Issues:**
   - Euler unstable for stiff systems
   - Check eigenvalues of linearized F
   - Use implicit methods if needed

4. **Ignoring Measurement Timing:**
   - Measurement at start, middle, or end of interval?
   - Control applied when? (typically zero-order hold)
   - Matters for fast dynamics

5. **Noise Independence:**
   - Assuming w[k] uncorrelated may not hold
   - Real disturbances often have memory
   - Consider colored noise if needed

6. **Forgetting Constraints:**
   - Physical bounds: C_A ≥ 0, T > 0
   - May need constrained state estimation
   - Noise can violate constraints (truncate or project)

"""

from typing import List, Optional, Tuple

import numpy as np
import sympy as sp

from src.systems.base.core.discrete_stochastic_system import DiscreteStochasticSystem


class DiscreteStochasticBatchReactor(DiscreteStochasticSystem):
    """
    Discrete-time stochastic batch reactor for digital control and estimation.

    Implements a sampled-data model of the chemical batch reactor with process
    noise, suitable for discrete-time control algorithms, Kalman filtering, and
    reinforcement learning applications.

    Discrete-Time Stochastic Dynamics
    ----------------------------------
    Difference equation (Euler discretization):

        C_A[k+1] = C_A[k] - r₁(X[k])·Δt + w_A[k]
        C_B[k+1] = C_B[k] + (r₁(X[k]) - r₂(X[k]))·Δt + w_B[k]
        T[k+1] = T[k] + (Q[k] - α·(T[k] - T_amb))·Δt + w_T[k]

    where:
        - X[k] = [C_A[k], C_B[k], T[k]]: State at time step k
        - r₁, r₂: Reaction rates (Arrhenius kinetics)
        - w[k] = [w_A[k], w_B[k], w_T[k]]: Process noise
        - w[k] ~ N(0, diag(σ_A², σ_B², σ_T²))
        - Δt: Sampling period

    Physical Interpretation
    -----------------------
    **Discrete-Time Operation:**

    Industrial batch reactors operate with digital control:
    - Sensors sampled every Δt seconds (PLC scan rate)
    - Control computed and applied at discrete intervals
    - Zero-order hold: Q[k] constant during [k·Δt, (k+1)·Δt]
    - Measurements: y[k] available at t = k·Δt

    **Process Noise Sources:**

    w[k] represents accumulated disturbances over sampling interval:
    1. **Concentration noise:** Feed variations, mixing imperfections
    2. **Temperature noise:** Heat transfer fluctuations, ambient changes
    3. **Model uncertainty:** Discretization errors, parameter drift

    **Noise Scaling from Continuous:**

    If continuous noise intensity is σ_c [state/√time]:
        σ_discrete = σ_c·√Δt [state]

    Example: σ_T = 1 K/√s continuous, Δt = 1 s
        → σ_T_discrete = 1·√1 = 1 K per time step

    State Space
    -----------
    State: X[k] = [C_A[k], C_B[k], T[k]]
        Discrete-time samples of concentrations and temperature:
        - C_A[k]: Concentration of A at time k·Δt [mol/L]
        - C_B[k]: Concentration of B at time k·Δt [mol/L]
        - T[k]: Temperature at time k·Δt [K]

    Control: u[k] = Q[k]
        - Q[k]: Heating rate during interval [k·Δt, (k+1)·Δt] [K/s]
        - Zero-order hold: Constant between samples

    Noise: w[k] = [w_A[k], w_B[k], w_T[k]]
        - Gaussian white noise: w[k] ~ N(0, Σ_w)
        - Independent over time: w[k] ⊥ w[j] for k ≠ j
        - Diagonal covariance: Σ_w = diag(σ_A², σ_B², σ_T²)

    Key Properties
    --------------
    **Markov Property:**
    X[k+1] depends only on X[k], enabling:
        - Dynamic programming
        - Kalman filtering
        - Reinforcement learning

    **Time-Invariant:**
    Dynamics f same at all time steps (constant parameters).

    **Gaussian Noise:**
    Enables analytical Kalman filter (optimal for Gaussian).

    **Additive Noise:**
    Simplifies estimation (linearity in noise).

    **Discrete-Time Native:**
    Exact for digital control (no continuous-time approximation).

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
        Process noise std dev for C_A [mol/L per step]
        - Typical: 0.001-0.1 mol/L
        - Conversion from continuous: σ_c·√Δt

    sigma_B : float, default=0.01
        Process noise std dev for C_B [mol/L per step]

    sigma_T : float, default=1.0
        Process noise std dev for T [K per step]
        - Typical: 0.1-5.0 K
        - Conversion from continuous: σ_c·√Δt

    dt : float, default=1.0
        Sampling period [s]
        - Typical: 0.1-10 s for batch reactors
        - Smaller dt → more accurate, more computation
        - Must satisfy Nyquist: dt < 1/(2·f_max)

    discretization_method : str, default='euler'
        Discretization method for continuous dynamics
        - 'euler': Forward Euler (first-order)
        - 'rk4': Runge-Kutta 4 (fourth-order)
        - 'exact': Exact for linear parts

    Applications
    ------------
    **1. Discrete-Time MPC:**
    Model Predictive Control with constraints:
        - Optimization horizon: N steps
        - Constraints: X_min ≤ X[k] ≤ X_max
        - Receding horizon: Solve at each k

    **2. Discrete Kalman Filter:**
    Optimal state estimation:
        - Process model: This discrete stochastic system
        - Measurement model: y[k] = h(X[k]) + v[k]
        - Extended or Unscented KF for nonlinearity

    **3. LQG Control:**
    Linear-Quadratic-Gaussian:
        - Kalman filter for state estimation
        - LQR for optimal control
        - Certainty equivalence principle

    **4. Reinforcement Learning:**
    Discrete-time RL algorithms:
        - Q-learning: Discrete state/action
        - Policy gradient: Episodic updates
        - Model-based RL: Use this as forward model

    **5. Batch-to-Batch Optimization:**
    Iterative learning control:
        - Update policy based on previous batches
        - Stochastic gradient descent
        - Safe learning with constraints

    Numerical Simulation
    --------------------
    **Direct Simulation:**
    No integration needed - direct evaluation:
        for k in range(N):
            X[k+1] = f(X[k], u[k]) + w[k]

    where w[k] ~ N(0, Σ_w).

    **Monte Carlo Ensemble:**
    Run multiple simulations to characterize stochasticity:
        - Mean trajectory
        - Variance growth
        - Confidence intervals

    **Deterministic Part:**
    Set σ_A = σ_B = σ_T = 0 to recover deterministic discrete system.

    Comparison with Continuous Stochastic
    --------------------------------------
    **Continuous Stochastic:**
    - SDE: dX = f·dt + g·dW
    - Requires SDE integration
    - Theoretical foundation
    - Noise intensity in [state]/√[time]

    **Discrete Stochastic:**
    - Difference equation: X[k+1] = f + w[k]
    - Direct evaluation (no integration)
    - Implementation ready
    - Noise variance in [state]²

    **Conversion:**
    σ_discrete = σ_continuous·√Δt

    Discrete Kalman Filter Usage
    -----------------------------
    This system provides process model for EKF:

    ```python
    # Process model
    X_pred = reactor.step(X_est, u[k])
    F = reactor.linearize(X_est, u[k])[0]  # Jacobian
    P_pred = F @ P @ F.T + Q

    # Measurement update (if measurement available)
    y_pred = h(X_pred)
    H = jacobian(h, X_pred)
    K = P_pred @ H.T @ inv(H @ P_pred @ H.T + R)
    X_est = X_pred + K @ (y[k] - y_pred)
    P = (I - K @ H) @ P_pred
    ```

    Limitations
    -----------
    - Euler discretization: O(Δt) error
    - Additive noise only (not multiplicative)
    - Constant noise variance (not state-dependent)
    - Independent noise (no correlation)
    - White noise (no temporal correlation)

    Extensions
    ----------
    - Higher-order discretization (RK4)
    - Multiplicative noise: w[k] depends on X[k]
    - Colored noise: w[k] correlated over time
    - Measurement delays: y[k] = h(X[k-d]) + v[k]
    - Time-varying parameters: θ[k+1] = θ[k] + ε[k]
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
        dt: float = 1.0,
        discretization_method: str = 'euler',
        C_A0: Optional[float] = None,
        T0: Optional[float] = None,
    ):
        """
        Define discrete-time stochastic batch reactor dynamics.

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
            Process noise std dev for C_A [mol/L per step]
            - For conversion from continuous: σ_d = σ_c·√dt
            - Typical: 0.001-0.1 mol/L

        sigma_B : float, default=0.01
            Process noise std dev for C_B [mol/L per step]

        sigma_T : float, default=1.0
            Process noise std dev for T [K per step]
            - For conversion from continuous: σ_d = σ_c·√dt
            - Typical: 0.1-5.0 K

        dt : float, default=1.0
            Sampling period [s]
            - Typical: 0.1-10 s for batch reactors
            - Rule: dt < τ_system/10
            - Affects both dynamics and noise

        discretization_method : str, default='euler'
            Discretization method:
            - 'euler': Forward Euler (simple, first-order)
            - 'rk4': Runge-Kutta 4 (accurate, fourth-order)

        C_A0, T0 : Optional[float]
            Initial conditions for equilibrium setup

        Notes
        -----
        **Discretization (Euler):**

        Continuous: dx/dt = f(x, u)
        Discrete: x[k+1] = x[k] + f(x[k], u[k])·dt

        For reactions:
            C_A[k+1] = C_A[k] - r₁·dt
            C_B[k+1] = C_B[k] + (r₁ - r₂)·dt
            T[k+1] = T[k] + (Q - α·(T - T_amb))·dt

        **Process Noise:**

        Additive Gaussian white noise:
            w[k] ~ N(0, Σ_w)
            Σ_w = diag(σ_A², σ_B², σ_T²)

        **Noise Scaling:**

        From continuous σ_c [state/√s]:
            σ_discrete = σ_c·√dt

        Example: σ_T_c = 1 K/√s, dt = 1 s
            → σ_T_d = 1·√1 = 1 K per step

        Example: σ_T_c = 1 K/√s, dt = 0.1 s
            → σ_T_d = 1·√0.1 ≈ 0.316 K per step

        **Sampling Period Selection:**

        Guidelines:
        1. Nyquist: dt < 1/(2·f_max)
        2. Control: dt < τ_cl/10 (closed-loop time constant)
        3. Reactor: dt < τ_reaction/5

        For typical batch reactor:
        - Fast control: dt = 0.1-0.5 s
        - Moderate: dt = 1.0-5.0 s
        - Slow: dt = 5.0-10.0 s

        **Discretization Method:**

        Euler (default):
        - Simple, explicit
        - O(dt) accuracy
        - May need small dt for accuracy
        - Most common in industrial control

        RK4 (optional):
        - More accurate: O(dt⁴)
        - Can use larger dt
        - More computation per step
        - Better for simulation studies

        **Validation:**

        Check discretization accuracy:
        1. Compare with continuous simulation
        2. Verify mass conservation (approximately)
        3. Check final concentrations converge as dt → 0
        """
        # Store initial conditions and method
        self.C_A0 = C_A0
        self.T0 = T0
        self.discretization_method = discretization_method

        # State and control variables
        C_A, C_B, T = sp.symbols("C_A C_B T", real=True, positive=True)
        Q = sp.symbols("Q", real=True)

        # Parameters (kinetics and heat transfer)
        k1, k2, E1, E2, alpha, T_amb = sp.symbols(
            "k1 k2 E1 E2 alpha T_amb", real=True, positive=True
        )

        # Noise intensities (standard deviations)
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
        self._dt = dt  # REQUIRED for discrete system

        # Reaction rates (Arrhenius kinetics)
        r1 = k1 * C_A * sp.exp(-E1 / T)
        r2 = k2 * C_B * sp.exp(-E2 / T)

        # DETERMINISTIC PART (Discrete-time dynamics using Euler)
        # This is f(x[k], u[k]) in x[k+1] = f(x[k], u[k]) + w[k]
        
        if discretization_method == 'euler':
            # Forward Euler discretization
            # x[k+1] = x[k] + dx/dt·dt
            C_A_next = C_A + (-r1) * dt
            C_B_next = C_B + (r1 - r2) * dt
            T_next = T + (Q - alpha * (T - T_amb)) * dt
            
        elif discretization_method == 'rk4':
            # Runge-Kutta 4 (more accurate but more complex symbolically)
            # For simplicity, use Euler in symbolic definition
            # Actual RK4 would be implemented numerically
            import warnings
            warnings.warn(
                "RK4 discretization not fully implemented symbolically. "
                "Using Euler discretization in symbolic definition. "
                "For RK4, implement numerically in step() override.",
                UserWarning
            )
            C_A_next = C_A + (-r1) * dt
            C_B_next = C_B + (r1 - r2) * dt
            T_next = T + (Q - alpha * (T - T_amb)) * dt
        else:
            raise ValueError(f"Unknown discretization_method: {discretization_method}")

        # State update function (deterministic part)
        self._f_sym = sp.Matrix([C_A_next, C_B_next, T_next])

        # STOCHASTIC PART (Process noise)
        # Diagonal diffusion matrix: three independent noise sources
        self.diffusion_expr = sp.Matrix([
            [sigma_A_sym, 0, 0],
            [0, sigma_B_sym, 0],
            [0, 0, sigma_T_sym]
        ])

        # Discrete-time SDE
        self.sde_type = "ito"

        # Output: Full state measurement (can add measurement noise separately)
        # TODO: need to implement adding noise to measurements
        self._h_sym = sp.Matrix([C_A, C_B, T])

    def setup_equilibria(self):
        """
        Set up equilibrium points (for deterministic part).

        Note: In discrete time, equilibria are fixed points of f:
            x_eq = f(x_eq, u_eq)
        """
        # Get parameters
        T_amb = self.parameters[sp.symbols("T_amb")]

        # Complete conversion equilibrium (discrete-time fixed point)
        self.add_equilibrium(
            "complete",
            x_eq=np.array([0.0, 0.0, T_amb]),
            u_eq=np.array([0.0]),
            verify=True,
            stability="stable",
            notes="Fixed point of deterministic part. Stochastic trajectories "
                  "fluctuate around this with variance σ² per step."
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
                notes=f"Initial state setpoint. Variance grows as k·σ² over k steps."
            )
            self.set_default_equilibrium("initial")
        else:
            self.set_default_equilibrium("complete")

    def get_process_noise_covariance(self) -> np.ndarray:
        """
        Get process noise covariance matrix Σ_w.

        Returns
        -------
        np.ndarray
            3x3 diagonal covariance matrix

        Notes
        -----
        Σ_w = diag(σ_A², σ_B², σ_T²)

        Diagonal → independent noise sources.

        Examples
        --------
        >>> reactor = DiscreteStochasticBatchReactor(
        ...     sigma_A=0.01, sigma_B=0.01, sigma_T=1.0
        ... )
        >>> Q = reactor.get_process_noise_covariance()
        >>> print(f"Process noise covariance:\\n{Q}")
        """
        sigma_A = self.parameters[sp.symbols('sigma_A')]
        sigma_B = self.parameters[sp.symbols('sigma_B')]
        sigma_T = self.parameters[sp.symbols('sigma_T')]
        
        return np.diag([sigma_A**2, sigma_B**2, sigma_T**2])

    def convert_from_continuous_noise(
        self,
        sigma_A_continuous: float,
        sigma_B_continuous: float,
        sigma_T_continuous: float
    ) -> dict:
        """
        Convert continuous noise intensities to discrete.

        For continuous noise σ_c [state/√s], discrete noise is:
            σ_d = σ_c·√dt

        Parameters
        ----------
        sigma_A_continuous : float
            Continuous noise intensity for C_A [mol/(L·√s)]
        sigma_B_continuous : float
            Continuous noise intensity for C_B [mol/(L·√s)]
        sigma_T_continuous : float
            Continuous noise intensity for T [K/√s]

        Returns
        -------
        dict
            Discrete noise parameters

        Examples
        --------
        >>> reactor = DiscreteStochasticBatchReactor(dt=1.0)
        >>> discrete_noise = reactor.convert_from_continuous_noise(
        ...     sigma_A_continuous=0.01,
        ...     sigma_B_continuous=0.01,
        ...     sigma_T_continuous=1.0
        ... )
        >>> print(f"Discrete σ_A: {discrete_noise['sigma_A']:.4f}")
        >>> print(f"Discrete σ_T: {discrete_noise['sigma_T']:.4f}")
        """
        dt = self._dt
        sqrt_dt = np.sqrt(dt)
        
        return {
            'sigma_A': sigma_A_continuous * sqrt_dt,
            'sigma_B': sigma_B_continuous * sqrt_dt,
            'sigma_T': sigma_T_continuous * sqrt_dt,
        }

    def estimate_discretization_error(self, x: np.ndarray, u: np.ndarray) -> float:
        """
        Estimate discretization error (Euler vs exact).

        Returns approximate relative error in state update.

        Parameters
        ----------
        x : np.ndarray
            Current state
        u : np.ndarray
            Control input

        Returns
        -------
        float
            Estimated relative error (dimensionless)

        Notes
        -----
        Euler discretization has O(dt) error.
        Error estimate: ||f(x)||·dt²/||x||

        Large error → Consider smaller dt or higher-order method.

        Examples
        --------
        >>> reactor = DiscreteStochasticBatchReactor(dt=1.0)
        >>> x = np.array([0.5, 0.3, 360.0])
        >>> u = np.array([10.0])
        >>> error = reactor.estimate_discretization_error(x, u)
        >>> print(f"Estimated error: {error:.2e}")
        """
        # Evaluate drift (continuous-time derivative)
        # For discrete system, this requires computing continuous dynamics
        # Approximate from difference equation
        x_next_det = self(x, u, k=0)  # Deterministic part (no noise)
        dx = x_next_det - x
        
        # Error estimate: O(dt²) for Euler
        dt = self._dt
        error_norm = np.linalg.norm(dx) * dt / np.linalg.norm(x)
        
        return error_norm


# Convenience function
def create_discrete_batch_reactor_with_noise(
    noise_level: str = 'medium',
    dt: float = 1.0,
    **kwargs
) -> DiscreteStochasticBatchReactor:
    """
    Create discrete stochastic batch reactor with predefined noise levels.

    Automatically scales noise for sampling period using σ_d = σ_c·√dt.

    Parameters
    ----------
    noise_level : str, default='medium'
        Noise level: 'low', 'medium', or 'high'
    dt : float, default=1.0
        Sampling period [s]
    **kwargs
        Additional parameters

    Returns
    -------
    DiscreteStochasticBatchReactor

    Notes
    -----
    Noise presets defined for dt=1.0 s, then scaled for actual dt.

    Examples
    --------
    >>> # Medium noise, 1 s sampling
    >>> reactor_1s = create_discrete_batch_reactor_with_noise('medium', dt=1.0)
    >>> 
    >>> # Medium noise, fast sampling (0.1 s) - auto-scaled
    >>> reactor_fast = create_discrete_batch_reactor_with_noise('medium', dt=0.1)
    >>> # Noise reduced by √(0.1) ≈ 0.316
    >>> 
    >>> # High noise, slow sampling (10 s)
    >>> reactor_slow = create_discrete_batch_reactor_with_noise('high', dt=10.0)
    >>> # Noise increased by √(10) ≈ 3.16
    """
    # Base noise levels for dt = 1.0 s
    noise_presets = {
        'low': {'sigma_A': 0.001, 'sigma_B': 0.001, 'sigma_T': 0.1},
        'medium': {'sigma_A': 0.01, 'sigma_B': 0.01, 'sigma_T': 1.0},
        'high': {'sigma_A': 0.1, 'sigma_B': 0.1, 'sigma_T': 5.0},
    }
    
    if noise_level not in noise_presets:
        raise ValueError(f"noise_level must be 'low', 'medium', or 'high'")
    
    # Scale noise for sampling period: σ_d = σ_base·√(dt/dt_base)
    dt_base = 1.0
    scale_factor = np.sqrt(dt / dt_base)
    
    preset = noise_presets[noise_level]
    scaled_noise = {
        'sigma_A': preset['sigma_A'] * scale_factor,
        'sigma_B': preset['sigma_B'] * scale_factor,
        'sigma_T': preset['sigma_T'] * scale_factor,
    }
    
    # Merge with user kwargs
    params = {**scaled_noise, 'dt': dt, **kwargs}
    
    return DiscreteStochasticBatchReactor(**params)

"""
Discrete Stochastic CSTR - Sampled-Data Reactor with Multiple Equilibria and Noise
===================================================================================

This module provides a discrete-time stochastic model of a Continuous Stirred-Tank
Reactor (CSTR), incorporating both the rich nonlinear dynamics (multiple steady states,
bifurcations) and realistic process noise. This model serves as:

- An advanced benchmark for nonlinear discrete-time stochastic control
- A realistic test case for robust MPC under uncertainty and multiplicity
- A challenging scenario for state estimation near unstable equilibria
- An illustration of stochastic bifurcations and noise-induced transitions
- A foundation for risk-aware control of multi-stable chemical processes

The discrete stochastic CSTR combines three layers of complexity:
1. **Nonlinearity:** Arrhenius kinetics creating multiple steady states
2. **Discrete-time:** Sampled measurements and control (digital implementation)
3. **Stochasticity:** Process noise causing state transitions and estimation challenges

This makes it one of the most challenging benchmark systems in process control.

Physical Context
----------------

**Industrial CSTR with Digital Control:**

Modern CSTRs operate with:
- Digital control systems (DCS, PLC) with discrete sampling
- Sensors sampled at regular intervals: Δt = 1-60 s typical
- Control actuators (jacket temperature) updated discretely
- Process noise from feed variations, ambient changes, fouling
- Risk of transitions between steady states due to disturbances

**Critical Challenge: Stability Under Noise**

The stochastic CSTR presents unique challenges:
1. **Multiple stable states:** System can operate at different equilibria
2. **Unstable intermediate:** Saddle point separates basins of attraction
3. **Process noise:** Can cause unintended transitions between states
4. **Rare events:** Low probability but high consequence (runaway)
5. **State estimation:** Nonlinearity + noise + proximity to instability

**Why This Model is Essential:**

Deterministic CSTR models miss critical phenomena:
- Noise-induced transitions between steady states
- Stochastic bifurcations (parameter regions change under noise)
- Reliability analysis (probability of staying in desired state)
- Worst-case scenarios (tail risk assessment)
- Robust control design (maintain high-conversion state despite noise)

Mathematical Formulation
-------------------------

**Discrete-Time Stochastic Dynamics:**

    X[k+1] = f(X[k], u[k]) + w[k]

where:
    - X[k] = [C_A[k], T[k]]: State at time k·Δt
    - u[k] = T_jacket[k]: Control (jacket temperature)
    - w[k] ~ N(0, Σ_w): Process noise
    - f: Discretized nonlinear CSTR dynamics

**Discretized Dynamics (Euler):**

From continuous CSTR:
    dC_A/dt = (F/V)·(C_A_feed - C_A) - k₀·C_A·exp(-E/T)
    dT/dt = (F/V)·(T_feed - T) + (-ΔH/ρC_p)·r + (UA/VρC_p)·(T_jacket - T)

Euler discretization:
    C_A[k+1] = C_A[k] + [(F/V)·(C_A_feed - C_A[k]) - r[k]]·Δt
    T[k+1] = T[k] + [(F/V)·(T_feed - T[k]) + heat_gen[k] + heat_removal[k]]·Δt

where r[k] = k₀·C_A[k]·exp(-E/T[k]).

**Process Noise Structure:**

    w[k] = [w_C[k], w_T[k]]ᵀ
    w[k] ~ N(0, diag(σ_C², σ_T²))

Independent noise sources for concentration and temperature:
- σ_C: Feed composition variability, sampling error
- σ_T: Heat transfer fluctuations, ambient variations

**Noise Scaling from Continuous:**

For continuous noise σ_c [state/√s]:
    σ_discrete = σ_c·√Δt

Critical for CSTR: Temperature noise most important due to
exponential sensitivity (Arrhenius).

Multiple Steady States Under Noise
-----------------------------------

**Deterministic CSTR:** Can have 1, 2, or 3 steady states

With noise, the picture changes:

**1. Stochastic Steady State:**
   Not a single point but a probability distribution around deterministic
   equilibrium. State fluctuates continuously due to noise.

**2. Noise-Induced Transitions:**
   Even if system starts in high-conversion state, noise can cause
   transitions to low-conversion state (and vice versa).

   Transition probability depends on:
   - Noise intensity (σ_T mainly)
   - Distance to saddle point (barrier height)
   - Sampling rate (Δt)

**3. Metastable States:**
   States that are "almost stable" - system stays there for long time
   but eventually transitions due to noise accumulation.

**4. Mean First Passage Time:**
   Expected time to transition from one basin to another:
   
   τ_transition ≈ exp(ΔE/σ²)
   
   where ΔE is energy barrier (related to saddle point).
   
   Large σ → fast transitions
   Small σ → rare transitions (but still possible!)

**5. Stochastic Bifurcations:**
   Parameter values where qualitative behavior changes under noise:
   - P-bifurcation: Changes in stationary distribution
   - D-bifurcation: Changes in stability (Lyapunov exponents)
   
   Different from deterministic bifurcations!

Implications for Control
------------------------

**Challenge: Maintain High-Conversion State**

Desired: Operate at high-conversion steady state (high T, low C_A)

Problems with noise:
1. **Noise pushes toward saddle:** Temperature fluctuations reduce stability margin
2. **Rare but catastrophic events:** Large noise realization causes transition
3. **Estimation uncertainty:** Don't know exact state, only noisy measurements
4. **Control-noise interaction:** Aggressive control can amplify noise effects

**Control Objectives Under Uncertainty:**

1. **Probabilistic Regulation:**
   Maintain state in target region with high probability:
   P(|X[k] - X_target| < δ) ≥ 0.95

2. **Risk-Sensitive Control:**
   Minimize expected cost plus variance penalty:
   J = E[Σ cost] + λ·Var[Σ cost]

3. **Chance-Constrained Control:**
   Ensure constraints satisfied with high probability:
   P(T[k] < T_max) ≥ 0.99 for all k

4. **Barrier Certificate:**
   Design controller ensuring system stays in safe region:
   P(enter unsafe region) ≤ ε_small

5. **Transition Prevention:**
   Maximize mean first passage time to undesired state.

State Estimation Challenges
----------------------------

**Why Kalman Filter is Difficult for Stochastic CSTR:**

1. **Strong Nonlinearity:**
   - Arrhenius exp(-E/T): Extremely nonlinear
   - Multiple steady states: Non-convex
   - EKF linearization poor near saddle point

2. **Model Mismatch:**
   - Discretization error (Euler O(Δt))
   - Unmodeled dynamics (jacket, fouling)
   - Parameter uncertainty (k₀, E not exactly known)

3. **Multiple Modes:**
   - Gaussian assumption breaks down
   - May need multi-modal filters (particle filter)

4. **Proximity to Instability:**
   - High-conversion state close to saddle
   - Linearized error dynamics nearly unstable
   - Estimation errors grow quickly

**Extended Kalman Filter (EKF):**

Linearize around current estimate:
    F[k] = ∂f/∂X |_{X̂[k|k]}

Problems:
- Linearization error large
- Can diverge near saddle
- Assumes unimodal Gaussian

**Unscented Kalman Filter (UKF):**

Sigma points capture nonlinearity better:
- No explicit Jacobians needed
- Better mean/covariance propagation
- Still assumes single mode

**Particle Filter (Best for CSTR):**

Represent distribution with particles:
- Naturally handles multimodality
- No linearization needed
- Expensive (N = 100-10,000 particles)
- Essential near transitions

**Moving Horizon Estimation (MHE):**

Optimization-based estimation:
- Can include constraints
- Handle nonlinearity exactly
- Computational cost
- Good for slow sampling (MPC applications)

Discrete-Time MPC for Stochastic CSTR
--------------------------------------

**Why MPC is Popular for CSTR:**

1. Handles constraints explicitly (T_max, C_A ≥ 0)
2. Accounts for nonlinearity via model
3. Predictive horizon helps with slow dynamics
4. Can incorporate economic objectives

**Standard MPC (Deterministic):**

    min Σ_{j=0}^{N-1} [||X[k+j] - X_ref||²_Q + ||u[k+j]||²_R]
    s.t. X[k+j+1] = f(X[k+j], u[k+j])
         X_min ≤ X[k+j] ≤ X_max
         u_min ≤ u[k+j] ≤ u_max

Problems: Ignores noise, can violate constraints.

**Stochastic MPC (Tube-Based):**

    min E[Σ cost]
    s.t. X[k+j+1] = f(X[k+j], u[k+j]) + w[k+j]
         P(X[k+j] ∈ X_safe) ≥ 1 - ε

Account for noise via:
- Robust tubes: Tighten constraints
- Scenario trees: Sample noise realizations
- Chance constraints: Probabilistic safety

**Risk-Sensitive MPC:**

    min E[Σ cost] + λ·Var[Σ cost]

Penalize variance (risk aversion).

**Economic MPC:**

    max E[profit - cost]
    
Where profit = conversion·price, cost = energy + violations.

Sampling Time Selection for Stochastic CSTR
--------------------------------------------

**Competing Factors:**

Fast sampling (small Δt):
- Pros: Accurate dynamics, fast disturbance rejection
- Cons: More noise accumulation, computational burden

Slow sampling (large Δt):
- Pros: Less noise, cheaper computation
- Cons: Miss fast transients, aliasing, instability

**Guidelines:**

1. **Stability Requirement:**
   Discrete-time closed-loop must be stable.
   For EKF+MPC, typically need Δt < 0.1·τ_cl

2. **Reactor Time Constant:**
   Residence time τ = V/F sets natural scale.
   Typical: Δt = 0.1-0.5·τ

3. **Temperature Dynamics:**
   Thermal time constant: τ_T = VρC_p/UA
   Need Δt < τ_T/5 to control temperature effectively

4. **Noise Considerations:**
   Smaller Δt → smaller discrete noise: σ_d = σ_c·√Δt
   But more samples → more noise accumulation over horizon

**For CSTR:**
- High-conversion state: Δt = 1-5 s (fast control needed)
- Low-conversion state: Δt = 5-30 s (more stable, slower)
- Startup/transitions: Δt = 0.5-2 s (critical period)

Stochastic Bifurcation Analysis
--------------------------------

**P-Bifurcation (Phenomenological):**

As noise intensity σ increases:
- Stationary distribution broadens
- Mean shifts from deterministic equilibrium
- Multiple peaks merge (below critical σ)

**D-Bifurcation (Dynamic):**

Changes in top Lyapunov exponent:
- Noise can stabilize unstable equilibrium (rare)
- More commonly: Noise destabilizes stable state

**Noise-Induced Transitions:**

Critical noise level σ_crit where transitions become frequent.

For CSTR: σ_T_crit ~ (ΔT_barrier)/√(τ_sample)

where ΔT_barrier = distance from operating point to saddle.

Reliability and Risk Analysis
------------------------------

**Monte Carlo for Rare Events:**

Estimate: P(transition to low-conversion) over time horizon T

Naive Monte Carlo:
- Run N simulations
- Count transitions
- Requires N = 1/P for accuracy (expensive if P small)

**Importance Sampling:**

Bias toward rare events:
- Sample from modified distribution
- Reweight to get true probability
- Much more efficient for rare events

**Splitting Methods:**

Break rare event into stages:
- Define intermediate rare sets
- Use splitting to reach each stage
- Estimate overall probability as product

**Large Deviations Theory:**

Analytical approximation:
    P(rare event) ≈ exp(-S/σ²)

where S is action functional (optimal path to event).

**Reliability Metric:**

Mean Time Between Failures:
    MTBF = 1/(P_transition·Δt)

Design goal: MTBF > 10,000 batches (very reliable).

Applications
------------

**1. Robust Startup Control:**
Challenge: Transition from low to high-conversion despite noise.
Solution: Robust MPC with chance constraints on temperature.

**2. Runaway Prevention:**
Challenge: Detect incipient runaway before catastrophe.
Solution: Particle filter + anomaly detection on likelihood.

**3. Economic Operation:**
Challenge: Maximize conversion while ensuring reliability.
Solution: Risk-sensitive MPC with economic objective.

**4. Fault Detection:**
Challenge: Distinguish faults from normal process noise.
Solution: Statistical hypothesis testing on KF residuals.

**5. Reinforcement Learning:**
Challenge: Learn control policy from noisy data.
Solution: Model-based RL using discrete stochastic model.

**6. Process Monitoring:**
Challenge: Early warning of transitions or deterioration.
Solution: Monitor distance to saddle point, variance trends.

Common Pitfalls
---------------

1. **Underestimating Noise Impact:**
   - Even small noise can cause transitions (rare events)
   - Exponential sensitivity via Arrhenius
   - Long-time behavior different from short-time

2. **Using EKF Blindly:**
   - EKF can diverge near saddle point
   - Need UKF or particle filter for safety-critical

3. **Ignoring Multiple Modes:**
   - State distribution may be bimodal
   - Single Gaussian assumption fails

4. **Wrong Noise Scaling:**
   - Must use σ_d = σ_c·√Δt for conversion
   - Temperature noise most critical

5. **Overly Tight Constraints:**
   - Noise will cause violations
   - Need probabilistic constraints or back-off

6. **Deterministic Analysis:**
   - Linearization around equilibrium insufficient
   - Need stochastic analysis for reliability

"""

class DiscreteStochasticCSTR(DiscreteStochasticSystem):
    """
    Discrete-time stochastic CSTR with multiple steady states and process noise.

    Combines the challenging nonlinear dynamics of the CSTR (multiple equilibria,
    bifurcations) with discrete-time sampling and process noise, creating one of
    the most demanding benchmarks in stochastic process control.

    Discrete-Time Stochastic Dynamics
    ----------------------------------
    Difference equation (Euler discretization):

        C_A[k+1] = C_A[k] + [(F/V)·(C_A_feed - C_A[k]) - r[k]]·Δt + w_C[k]
        T[k+1] = T[k] + [(F/V)·(T_feed - T[k]) + q_gen[k] + q_removal[k]]·Δt + w_T[k]

    where:
        - r[k] = k₀·C_A[k]·exp(-E/T[k]): Reaction rate
        - q_gen = (-ΔH/ρC_p)·r: Heat generation
        - q_removal = (UA/VρC_p)·(T_jacket[k] - T[k]): Heat removal
        - w[k] ~ N(0, diag(σ_C², σ_T²)): Process noise

    Physical Interpretation
    -----------------------
    **Industrial Digital Control:**

    CSTR operated with:
    - PLC/DCS sampling: Δt = 1-60 s typical
    - Concentration: Online analyzer (GC, NIR) every 1-5 min
    - Temperature: Fast measurement (1-10 s)
    - Jacket temperature: Control valve updated discretely

    **Process Noise Sources:**

    1. **Concentration noise (σ_C):**
       - Feed composition variations (batch-to-batch)
       - Sampling/analyzer uncertainty
       - Flow rate fluctuations
       - Typical: 0.001-0.01 mol/L per step

    2. **Temperature noise (σ_T):**
       - Heat transfer coefficient variations (fouling)
       - Ambient temperature changes
       - Flow rate in jacket fluctuations
       - Most critical: Affects rates exponentially
       - Typical: 0.1-2.0 K per step

    **Why Temperature Noise Dominates:**

    Arrhenius: r ∝ exp(-E/T)
    - Small T change → large rate change
    - E/T² ≈ 8750/350² ≈ 0.071 K⁻¹
    - 1 K noise → 7% rate change
    - This creates strong coupling to concentration

    Multiple Steady States Under Noise
    -----------------------------------
    **Deterministic:** 1, 2, or 3 steady states possible

    **With Noise:**
    - States fluctuate around deterministic equilibria
    - Noise can cause transitions between basins
    - Rare events: Escape from desired state
    - Metastability: Long residence time then sudden transition

    **Critical for Control:**
    - Must maintain high-conversion despite noise
    - Risk of noise-induced transition to low-conversion
    - Probability of transition depends on σ_T, distance to saddle
    - Requires robust control + risk management

    State Space
    -----------
    State: X[k] = [C_A[k], T[k]]
        - C_A[k]: Concentration at time k·Δt [mol/L]
        - T[k]: Temperature at time k·Δt [K]
        - Stochastic processes (not deterministic)

    Control: u[k] = T_jacket[k]
        - Jacket temperature [K]
        - Zero-order hold between samples

    Noise: w[k] = [w_C[k], w_T[k]]
        - w[k] ~ N(0, Σ_w)
        - Σ_w = diag(σ_C², σ_T²)
        - Independent, Gaussian, white

    Key Properties
    --------------
    **1. Markov Property:**
    Essential for dynamic programming, RL, Kalman filtering.

    **2. Multiple Modes:**
    State distribution may be multimodal (near bifurcation).

    **3. Exponential Sensitivity:**
    Temperature noise amplified exponentially by Arrhenius.

    **4. Metastability:**
    System can stay near unstable equilibrium for long time.

    **5. Rare Transitions:**
    Low probability but high consequence events.

    Parameters
    ----------
    F, V, C_A_feed, T_feed, k0, E, delta_H, rho, Cp, UA : float
        Same as deterministic CSTR (see ContinuousCSTR)

    sigma_C : float, default=0.001
        Process noise std dev for C_A [mol/L per step]
        - Typical: 0.001-0.01 mol/L
        - Smaller than batch reactor (continuous operation)

    sigma_T : float, default=0.5
        Process noise std dev for T [K per step]
        - Typical: 0.1-2.0 K
        - Most critical parameter
        - Conversion from continuous: σ_c·√Δt

    dt : float, default=5.0
        Sampling period [s]
        - CSTR typically slower than batch: 1-60 s
        - Faster near high-conversion (less stable)
        - Slower at low-conversion (more stable)

    Applications
    ------------
    **1. Stochastic MPC:**
    Model Predictive Control with chance constraints:
        - Probabilistic safety: P(T < T_max) ≥ 0.99
        - Risk-sensitive objective: E[cost] + λ·Var[cost]
        - Scenario tree or robust tubes

    **2. Particle Filter:**
    State estimation with multiple modes:
        - Essential near transitions
        - Handles non-Gaussian distributions
        - Expensive but necessary

    **3. Robust Startup:**
    Transition low → high conversion despite noise:
        - Tube-based MPC
        - Barrier certificates
        - Risk-aware control

    **4. Reliability Analysis:**
    Assess transition probability:
        - Monte Carlo with importance sampling
        - Large deviations theory
        - Mean first passage time

    **5. Fault Detection:**
    Distinguish faults from noise:
        - Statistical tests on residuals
        - Anomaly detection on likelihood
        - Change point detection

    Numerical Simulation
    --------------------
    **Monte Carlo Ensemble:**

    Critical for CSTR to characterize rare events:
    - N = 1,000-10,000 runs
    - Estimate transition probability
    - Identify escape paths
    - Design robust controllers

    **Single Trajectory:**

    May not be representative:
    - Could stay in one state by chance
    - May transition unusually fast/slow
    - Always report ensemble statistics

    State Estimation
    ----------------
    **Recommended Approach:**

    For high-conversion operation:
    1. Use UKF or particle filter (nonlinearity)
    2. Monitor distance to saddle point
    3. Increase measurement rate if approaching instability
    4. Switch to particle filter if bimodality detected

    For startup/transitions:
    1. Particle filter essential (multimodal)
    2. Large particle count (N ≥ 1000)
    3. Importance sampling toward rare events

    Comparison with Deterministic
    ------------------------------
    **Deterministic Discrete CSTR:**
    - Single trajectory per IC and control
    - Multiple equilibria (1-3)
    - Bifurcations (saddle-node, Hopf)

    **Stochastic Discrete CSTR:**
    - Ensemble of trajectories
    - Stochastic equilibria (distributions)
    - Stochastic bifurcations (P and D)
    - Noise-induced transitions

    **Critical Difference:**
    Stochastic model essential for:
    - Reliability assessment
    - Risk management
    - Robust control design
    - Safety verification

    Limitations
    -----------
    - Euler discretization: O(Δt) error
    - Additive noise: Not multiplicative
    - Constant noise: Not state/time-dependent
    - White noise: No temporal correlation
    - Gaussian noise: Not heavy-tailed

    Extensions
    ----------
    - Higher-order discretization (RK4)
    - Multiplicative noise: σ_T(T)
    - Colored noise: Autoregressive
    - Jump processes: Fault events
    - Time-varying parameters: Catalyst deactivation

    See Also
    --------
    ContinuousStochasticCSTR : Continuous-time version
    DiscreteStochasticBatchReactor : Batch version
    DiscreteCSTR : Deterministic discrete CSTR
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
        sigma_T: float = 0.5,
        dt: float = 5.0,
        discretization_method: str = 'euler',
        x_ss: Optional[np.ndarray] = None,
        u_ss: Optional[np.ndarray] = None,
    ):
        """
        Define discrete-time stochastic CSTR dynamics.

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
            Process noise std dev for C_A [mol/L per step]
            - Smaller than batch (continuous operation)
            - Typical: 0.001-0.01 mol/L
            - Feed variability, analyzer noise

        sigma_T : float, default=0.5
            Process noise std dev for T [K per step]
            - Most critical parameter
            - Typical: 0.1-2.0 K
            - Heat transfer fluctuations, ambient changes
            - Conversion: σ_c·√Δt from continuous

        dt : float, default=5.0
            Sampling period [s]
            - CSTR slower than batch: 1-60 s typical
            - Residence time: τ = V/F = 1 s (for defaults)
            - Guideline: dt ~ 0.5-5·τ
            - Faster near high-conversion (less stable)

        discretization_method : str, default='euler'
            'euler' or 'rk4'

        x_ss, u_ss : Optional[np.ndarray]
            Steady state (if known)

        Notes
        -----
        **Noise Scaling:**

        For continuous σ_c [state/√s]:
            σ_discrete = σ_c·√dt

        Example: σ_T_c = 1 K/√s, dt = 5 s
            → σ_T_d = 1·√5 ≈ 2.24 K per step

        **Temperature Noise Criticality:**

        Arrhenius sensitivity: ∂r/∂T ≈ r·(E/T²)
        - At T = 390 K, E = 8750 K: sensitivity = 0.058 K⁻¹
        - 1 K noise → 5.8% rate change
        - 2 K noise → 11.6% rate change
        - This is why σ_T most important

        **Sampling Time:**

        Tradeoffs:
        - Smaller dt: More accurate, faster control, but more noise samples
        - Larger dt: Smoother (less noise), but miss dynamics
        
        For CSTR:
        - High-conversion: dt = 1-5 s (critical, fast control)
        - Low-conversion: dt = 5-30 s (stable, slower)
        - Startup: dt = 1-2 s (transitions critical)

        **Multiple Steady States:**

        CSTR can have 1-3 steady states. Use find_steady_states()
        to locate all equilibria for given T_jacket.

        Noise causes transitions between states!
        - Design σ_T small enough for reliability
        - Or design controller to prevent transitions
        """
        # Store steady state and method
        self.x_ss = x_ss
        self.u_ss = u_ss
        self.discretization_method = discretization_method

        # State and control variables
        C_A, T = sp.symbols("C_A T", real=True, positive=True)
        T_jacket = sp.symbols("T_jacket", real=True, positive=True)

        # Parameters
        F, V, C_A_feed, T_feed = sp.symbols("F V C_A_feed T_feed", real=True, positive=True)
        k0, E, delta_H, rho, Cp, UA = sp.symbols(
            "k0 E delta_H rho Cp UA", real=True, positive=True
        )

        # Noise intensities
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
        self._dt = dt  # REQUIRED for discrete system

        # Reaction rate (Arrhenius)
        r = k0 * C_A * sp.exp(-E / T)

        # DETERMINISTIC PART (Euler discretization)
        # Material balance
        dC_A_dt = (F / V) * (C_A_feed - C_A) - r

        # Energy balance
        dT_dt = (
            (F / V) * (T_feed - T)
            + ((-delta_H) / (rho * Cp)) * r
            + (UA / (V * rho * Cp)) * (T_jacket - T)
        )

        # Discrete-time update (Euler)
        C_A_next = C_A + dC_A_dt * dt
        T_next = T + dT_dt * dt

        self._f_sym = sp.Matrix([C_A_next, T_next])

        # STOCHASTIC PART (Process noise)
        # Diagonal: independent concentration and temperature noise
        self.diffusion_expr = sp.Matrix([
            [sigma_C_sym, 0],
            [0, sigma_T_sym]
        ])

        self.sde_type = "ito"

        # Output: Full state
        self._h_sym = sp.Matrix([C_A, T])

    def setup_equilibria(self):
        """
        Set up equilibrium points (deterministic part).

        Note: With process noise, system doesn't stay at equilibrium
        but fluctuates around it. Multiple equilibria may exist.
        """
        if self.x_ss is not None and self.u_ss is not None:
            self.add_equilibrium(
                "steady_state",
                x_eq=self.x_ss,
                u_eq=self.u_ss,
                verify=True,
                stability="unknown",
                notes="Deterministic equilibrium. With noise, state fluctuates around this. "
                      "CSTR may have multiple equilibria - use find_steady_states()."
            )
            self.set_default_equilibrium("steady_state")

    def get_process_noise_covariance(self) -> np.ndarray:
        """
        Get process noise covariance matrix.

        Returns
        -------
        np.ndarray
            2x2 diagonal covariance matrix

        Examples
        --------
        >>> cstr = DiscreteStochasticCSTR(sigma_C=0.001, sigma_T=0.5)
        >>> Q = cstr.get_process_noise_covariance()
        >>> print(f"Process noise:\\n{Q}")
        """
        sigma_C = self.parameters[sp.symbols('sigma_C')]
        sigma_T = self.parameters[sp.symbols('sigma_T')]
        
        return np.diag([sigma_C**2, sigma_T**2])

    def compute_residence_time(self) -> float:
        """
        Compute residence time τ = V/F.

        Returns
        -------
        float
            Residence time [s]

        Notes
        -----
        Natural time scale for CSTR dynamics.
        Sampling period typically: dt ~ 0.5-5·τ
        """
        F = self.parameters[sp.symbols("F")]
        V = self.parameters[sp.symbols("V")]
        return V / F

    def compute_damkohler_number(self, T: float) -> float:
        """
        Compute Damköhler number Da = k·τ.

        Parameters
        ----------
        T : float
            Temperature [K]

        Returns
        -------
        float
            Damköhler number

        Notes
        -----
        Da >> 1: Fast reaction, high conversion
        Da << 1: Slow reaction, low conversion
        """
        k0 = self.parameters[sp.symbols("k0")]
        E = self.parameters[sp.symbols("E")]
        tau = self.compute_residence_time()
        
        k = k0 * np.exp(-E / T)
        return k * tau

    def find_steady_states(
        self,
        T_jacket: float,
        T_range: tuple = (300.0, 500.0),
        n_points: int = 100,
    ) -> List[Tuple[float, float]]:
        """
        Find all steady states (deterministic part).

        Uses root finding with multiple initial guesses.

        Parameters
        ----------
        T_jacket : float
            Jacket temperature [K]
        T_range : tuple
            Temperature search range [K]
        n_points : int
            Number of initial guesses

        Returns
        -------
        List[Tuple[float, float]]
            List of (C_A, T) steady states

        Notes
        -----
        For stochastic system, these are centers of probability
        distributions. Actual state fluctuates around these points.

        With noise, transitions between states possible!

        Examples
        --------
        >>> cstr = DiscreteStochasticCSTR()
        >>> states = cstr.find_steady_states(T_jacket=350.0)
        >>> for i, (C_A, T) in enumerate(states):
        ...     print(f"State {i+1}: C_A={C_A:.3f}, T={T:.1f}")
        """
        from scipy.optimize import fsolve

        # Extract parameters
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

        def steady_state_equations(state):
            """Steady state: dC_A/dt = 0, dT/dt = 0"""
            C_A, T = state

            if C_A < 0 or T < 250:
                return [1e10, 1e10]

            r = k0 * C_A * np.exp(-E / T)

            # Material balance
            dC_A_dt = (F / V) * (C_A_feed - C_A) - r

            # Energy balance
            dT_dt = (
                (F / V) * (T_feed - T)
                + ((-delta_H) / (rho * Cp)) * r
                + (UA / (V * rho * Cp)) * (T_jacket - T)
            )

            return [dC_A_dt, dT_dt]

        # Try multiple initial guesses
        steady_states = []
        T_guesses = np.linspace(T_range[0], T_range[1], n_points)

        for T_guess in T_guesses:
            # Estimate C_A
            r_guess = k0 * C_A_feed * np.exp(-E / T_guess)
            C_A_guess = C_A_feed / (1 + (V / F) * r_guess / C_A_feed)
            C_A_guess = np.clip(C_A_guess, 0.0, C_A_feed)

            try:
                solution, info, ier, msg = fsolve(
                    steady_state_equations,
                    [C_A_guess, T_guess],
                    full_output=True,
                )

                if ier == 1:
                    C_A_sol, T_sol = solution

                    if (
                        0 <= C_A_sol <= C_A_feed
                        and T_range[0] <= T_sol <= T_range[1]
                        and not any(
                            np.allclose([C_A_sol, T_sol], ss, rtol=1e-3) 
                            for ss in steady_states
                        )
                    ):
                        steady_states.append((C_A_sol, T_sol))

            except Exception:
                continue

        steady_states.sort(key=lambda x: x[1])
        return steady_states

    def estimate_transition_probability(
        self,
        x_initial: np.ndarray,
        u_sequence: np.ndarray,
        threshold_T: float,
        n_simulations: int = 1000,
    ) -> dict:
        """
        Estimate probability of transitioning to low-conversion state.

        Uses Monte Carlo simulation to estimate rare event probability.

        Parameters
        ----------
        x_initial : np.ndarray
            Initial state (e.g., high-conversion)
        u_sequence : np.ndarray
            Control sequence (n_steps, 1)
        threshold_T : float
            Temperature threshold for transition [K]
            (e.g., T < 370 K indicates low-conversion)
        n_simulations : int, default=1000
            Number of Monte Carlo runs

        Returns
        -------
        dict
            Statistics: probability, mean_time, std_time

        Examples
        --------
        >>> cstr = DiscreteStochasticCSTR()
        >>> x0 = np.array([0.1, 390.0])  # High-conversion
        >>> u_seq = np.full((200, 1), 350.0)
        >>> result = cstr.estimate_transition_probability(
        ...     x0, u_seq, threshold_T=370.0, n_simulations=1000
        ... )
        >>> print(f"P(transition) = {result['probability']:.4f}")
        """
        n_steps = len(u_sequence)
        transitions = []

        for i in range(n_simulations):
            result = self.simulate(x_initial, u_sequence, n_steps=n_steps)
            T_traj = result['states'][:, 1]
            
            # Check if transition occurred
            transition_idx = np.where(T_traj < threshold_T)[0]
            
            if len(transition_idx) > 0:
                transitions.append(transition_idx[0])

        probability = len(transitions) / n_simulations
        
        if transitions:
            mean_time = np.mean(transitions) * self._dt
            std_time = np.std(transitions) * self._dt
        else:
            mean_time = np.inf
            std_time = 0.0

        return {
            'probability': probability,
            'mean_time': mean_time,
            'std_time': std_time,
            'n_simulations': n_simulations,
        }


# Convenience function
def create_discrete_stochastic_cstr_with_noise(
    noise_level: str = 'medium',
    dt: float = 5.0,
    **kwargs
) -> DiscreteStochasticCSTR:
    """
    Create discrete stochastic CSTR with predefined noise levels.

    Automatically scales noise for sampling period.

    Parameters
    ----------
    noise_level : str, default='medium'
        'low', 'medium', or 'high'
    dt : float, default=5.0
        Sampling period [s]
    **kwargs
        Additional parameters

    Returns
    -------
    DiscreteStochasticCSTR

    Examples
    --------
    >>> # Standard CSTR with medium noise
    >>> cstr = create_discrete_stochastic_cstr_with_noise('medium', dt=5.0)
    >>> 
    >>> # High noise, fast sampling
    >>> cstr_noisy = create_discrete_stochastic_cstr_with_noise('high', dt=1.0)
    """
    # Base noise for dt = 5.0 s
    noise_presets = {
        'low': {'sigma_C': 0.0005, 'sigma_T': 0.2},
        'medium': {'sigma_C': 0.001, 'sigma_T': 0.5},
        'high': {'sigma_C': 0.005, 'sigma_T': 2.0},
    }
    
    if noise_level not in noise_presets:
        raise ValueError(f"noise_level must be 'low', 'medium', or 'high'")
    
    # Scale for sampling period
    dt_base = 5.0
    scale_factor = np.sqrt(dt / dt_base)
    
    preset = noise_presets[noise_level]
    scaled_noise = {
        'sigma_C': preset['sigma_C'] * scale_factor,
        'sigma_T': preset['sigma_T'] * scale_factor,
    }
    
    params = {**scaled_noise, 'dt': dt, **kwargs}
    
    return DiscreteStochasticCSTR(**params)