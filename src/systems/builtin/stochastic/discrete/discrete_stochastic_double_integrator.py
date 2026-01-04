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
Discrete Stochastic Double Integrator - Digital LQG Benchmark System
=====================================================================

This module provides the discrete-time stochastic double integrator, the
canonical benchmark for digital control, discrete Kalman filtering, and
sampled-data LQG design. The discrete stochastic double integrator serves as:

- The fundamental example for discrete-time LQG (Linear-Quadratic-Gaussian) control
- The standard benchmark for discrete Kalman filter implementation and testing
- A model for digital position control under uncertainty (real-time systems)
- An illustration of exact discretization vs approximate discretization
- The foundation for understanding sampled-data control with process noise

Unlike the continuous version (theoretical foundation), the discrete version
represents what actually gets implemented in digital controllers - every
real control system operates in discrete time with sampled measurements and
zero-order hold control.

Physical Context
----------------

**Digital Control Reality:**

All modern control systems are digital:
- Sensors sampled at discrete intervals (ADC conversion)
- Controller computed on microprocessor/FPGA
- Control output via DAC with zero-order hold
- Time steps: 1 ms to 1 s typical (depending on application)

**Examples of Digital Double Integrators:**

1. **CNC Machine Tool:**
   - Position: x (table position)
   - Velocity: v (feed rate)
   - Control: u (motor command)
   - Sampling: Δt = 1 ms (1 kHz servo loop)
   - Noise: Encoder quantization, motor ripple

2. **Satellite Station-Keeping:**
   - Position: x (orbital position error)
   - Velocity: v (velocity error)
   - Control: u (thruster pulse)
   - Sampling: Δt = 1-10 s (digital controller update rate)
   - Noise: Measurement errors, disturbance forces

3. **Hard Disk Drive:**
   - Position: x (read head position)
   - Velocity: v (head velocity)
   - Control: u (voice coil current)
   - Sampling: Δt = 10-100 μs (very fast servo)
   - Noise: Vibration, bearing runout

4. **Mobile Robot:**
   - Position: x (along path)
   - Velocity: v
   - Control: u (wheel motor command)
   - Sampling: Δt = 10-100 ms
   - Noise: Wheel slip, encoder errors

**Why Discrete Model is Essential:**

Continuous model: Theoretical (infinite sampling rate, continuous actuation)
Discrete model: Practical (actual implementation, digital computer)

Key differences:
- **Sampling effects:** Information loss between samples (aliasing)
- **Quantization:** Finite precision arithmetic, sensor resolution
- **Zero-order hold:** Control constant between samples (not smooth)
- **Computational delay:** Processing time affects stability

Mathematical Formulation
------------------------

**Discrete-Time State-Space:**

Exact discretization of continuous double integrator:
    x[k+1] = x[k] + v[k]·Δt + (1/2)·u[k]·Δt²
    v[k+1] = v[k] + u[k]·Δt

With process noise on velocity:
    x[k+1] = x[k] + v[k]·Δt + (1/2)·u[k]·Δt² + w_x[k]
    v[k+1] = v[k] + u[k]·Δt + w_v[k]

**Matrix Form:**
    X[k+1] = Φ·X[k] + Γ·u[k] + w[k]

where:
    X = [x, v]ᵀ
    
    Φ = [1  Δt]  (state transition matrix)
        [0   1]
    
    Γ = [Δt²/2]  (control input matrix)
        [Δt   ]
    
    w ~ N(0, Q)  (process noise)
    
    Q = [q_x    q_xv ]  (process noise covariance)
        [q_xv   q_v  ]

**Noise Structure:**

**Option 1: Velocity Noise Only (Common):**
Noise on velocity equation only (physical: force disturbances):
    Q = [0      0   ]
        [0   σ_v²·Δt]

**Option 2: Both Position and Velocity Noise:**
More general (includes model uncertainty):
    Q = [σ_x²·Δt³/3   σ_x²·Δt²/2]
        [σ_x²·Δt²/2   σ_x²·Δt   ]

This comes from integrating continuous noise σ·dW.

**This Implementation:** Option 2 (general case).

Exact vs Approximate Discretization
------------------------------------

**Exact Discretization:**

From continuous SDE:
    dx = v·dt
    dv = u·dt + σ·dW

Exact discrete-time equivalent (zero-order hold on u):
    x[k+1] = x[k] + v[k]·Δt + (1/2)·u[k]·Δt²
    v[k+1] = v[k] + u[k]·Δt + σ·√Δt·w[k]

Process noise covariance (from integration):
    Q_exact = σ²·[Δt³/3   Δt²/2]
                  [Δt²/2   Δt   ]

This is **exact** - no discretization error!

**Approximate (Euler):**
    x[k+1] = x[k] + v[k]·Δt
    v[k+1] = v[k] + u[k]·Δt + σ·√Δt·w[k]

Missing (1/2)·u·Δt² term in position.
- Error: O(Δt²) in position
- For small Δt: Negligible
- For large Δt: Can accumulate

**This Implementation:** Exact discretization.

Discrete LQG Control
--------------------

**The Canonical Digital Control Problem:**

**Infinite-Horizon Discrete LQR:**

Minimize:
    J = E[Σ_{k=0}^∞ (X[k]ᵀ·Q·X[k] + u[k]·R·u[k])]

**Solution:**
Discrete Algebraic Riccati Equation (DARE):
    P = Φᵀ·P·Φ - Φᵀ·P·Γ·(R + Γᵀ·P·Γ)⁻¹·Γᵀ·P·Φ + Q_cost

Optimal gain:
    K = (R + Γᵀ·P·Γ)⁻¹·Γᵀ·P·Φ

Optimal control:
    u[k] = -K·X[k]

**Discrete Kalman Filter:**

**Prediction Step:**
    X̂[k+1|k] = Φ·X̂[k|k] + Γ·u[k]
    P[k+1|k] = Φ·P[k|k]·Φᵀ + Q_noise

**Update Step (when measurement arrives):**
    K_kf = P[k+1|k]·Cᵀ·(C·P[k+1|k]·Cᵀ + R_meas)⁻¹
    X̂[k+1|k+1] = X̂[k+1|k] + K_kf·(y[k+1] - C·X̂[k+1|k])
    P[k+1|k+1] = (I - K_kf·C)·P[k+1|k]

**Separation Principle:**
Design LQR and Kalman filter independently, then combine:
    u[k] = -K·X̂[k|k]

**Certainty Equivalence:**
Use state estimate as if it were true state (optimal for Gaussian).

Sampling Time Selection
-----------------------

**Critical for Digital Control:**

**Too Slow (Large Δt):**
- Misses fast dynamics
- Aliasing (Nyquist violation)
- Instability (discrete eigenvalues outside unit circle)
- Poor disturbance rejection

**Too Fast (Small Δt):**
- Computational burden
- Quantization effects dominate
- Sensor noise amplified
- Numerical precision issues

**Guidelines:**

1. **Nyquist Criterion:**
   Δt < 1/(2·f_max) where f_max is highest frequency of interest

2. **Control Bandwidth:**
   Sample 5-10× faster than desired closed-loop bandwidth:
   Δt < 1/(10·ω_cl)

3. **Shannon's Rule:**
   For Δt sampling of continuous system with pole at ω:
   Discrete pole: z = e^(ω·Δt)
   Stability requires: |z| < 1

4. **Practical:**
   - Fast servo (hard disk): Δt = 10-100 μs
   - Medium (robotics): Δt = 1-100 ms
   - Slow (chemical): Δt = 0.1-10 s

**For Double Integrator:**
- Stability: Any Δt (marginally stable)
- Performance: Δt < 0.1/ω_d where ω_d is desired bandwidth

Noise Covariance Structure
---------------------------

**Process Noise Q:**

From continuous noise σ integrated over [k·Δt, (k+1)·Δt]:

    Q = σ²·[Δt³/3   Δt²/2]
            [Δt²/2   Δt   ]

**Structure:**
- Q₁₁ = σ²·Δt³/3: Position noise variance (cubic in Δt!)
- Q₂₂ = σ²·Δt: Velocity noise variance (linear in Δt)
- Q₁₂ = σ²·Δt²/2: Position-velocity covariance

**Why Non-Diagonal?**
Velocity noise integrates to position:
- Creates correlation between position and velocity errors
- Cannot have position noise without velocity noise
- Physical: Forces create accelerations (velocity changes)

**Scaling with Δt:**
- Smaller Δt: All elements shrink
- Q₁₁ shrinks fastest (Δt³)
- This is why fast sampling reduces noise accumulation

**Measurement Noise R:**

Sensor noise covariance:
- Position only: R = r_pos (scalar)
- Both states: R = diag(r_pos, r_vel)

Independent of Δt (sensor noise per sample, not per time).

Applications
------------

**1. Digital Control Implementation:**

Every real controller:
- Embedded systems (Arduino, Raspberry Pi)
- PLCs (industrial automation)
- DSP chips (real-time control)
- FPGAs (ultra-fast control)

**2. Discrete Kalman Filter:**

Standard state estimation:
- GPS/INS integration (navigation)
- Radar tracking
- Robot localization
- Sensor fusion

**3. Model Predictive Control (MPC):**

Native discrete formulation:
- Optimization over discrete horizon
- Constraints on states/inputs
- Used in process control, robotics

**4. Reinforcement Learning:**

Discrete-time RL:
- Q-learning, SARSA (discrete updates)
- Policy gradient (episodic)
- Model-based RL (forward dynamics)

**5. Digital Signal Processing:**

Discrete filters:
- Design in discrete domain
- Implementation on DSP
- Real-time processing

Comparison with Continuous
---------------------------

**Continuous Stochastic Double Integrator:**
- Theoretical foundation
- Continuous-time Riccati equations
- SDE integration required
- Noise intensity: σ [m/(s²·√s)]

**Discrete Stochastic Double Integrator:**
- Practical implementation
- Discrete-time Riccati equations (DARE)
- Direct evaluation (no integration)
- Noise covariance: Q [m²] matrix

**Key Differences:**

1. **Eigenvalues:**
   - Continuous: λ = {0, 0} (imaginary axis)
   - Discrete: z = {1, 1} (unit circle boundary)

2. **Stability:**
   - Continuous: Marginally stable (Re(λ) = 0)
   - Discrete: Marginally stable (|z| = 1)

3. **Noise:**
   - Continuous: Wiener process dW
   - Discrete: Gaussian w[k] with covariance Q

4. **Control:**
   - Continuous: u(t) continuous function
   - Discrete: u[k] zero-order hold

**Conversion:**
    Φ = exp(A·Δt) = [1  Δt]
                     [0   1]
    Q from integration of continuous noise

Common Pitfalls
---------------

1. **Wrong Noise Covariance:**
   - Using diagonal Q when should be coupled
   - Forgetting Δt scaling (Q ∝ Δt for velocity noise)
   - Using continuous σ instead of discrete Q

2. **Euler Instead of Exact:**
   - Missing (1/2)·u·Δt² term in position
   - Creates O(Δt²) bias
   - Use exact discretization

3. **Ignoring Sampling:**
   - Treating as continuous with small Δt
   - Missing quantization effects
   - Aliasing from undersampling

4. **Kalman Tuning:**
   - Q too small: Filter overconfident, slow tracking
   - Q too large: Filter jumpy, trusts measurements too much
   - Need physical understanding to set Q, R

5. **Marginal Stability:**
   - Open-loop eigenvalues at z=1 (unit circle)
   - Small numerical errors can cause drift
   - Need feedback for regulation

6. **Covariance Matrix:**
   - Q must be positive semi-definite
   - Check: All eigenvalues ≥ 0
   - Non-diagonal structure from integration

**Impact:**
Discrete double integrator demonstrated:
- Digital implementation matches theory
- Kalman filter optimal (no approximation for linear Gaussian)
- LQG can be implemented on modest hardware
- Foundation of modern digital control

"""

import numpy as np
import sympy as sp
from typing import Optional

from src.systems.base.core.discrete_stochastic_system import DiscreteStochasticSystem


class DiscreteStochasticDoubleIntegrator(DiscreteStochasticSystem):
    """
    Discrete-time stochastic double integrator - canonical digital LQG benchmark.

    The fundamental discrete-time linear system for testing digital control
    algorithms, discrete Kalman filtering, and sampled-data design. This
    represents what actually gets implemented in real digital controllers.

    Discrete-Time Stochastic Dynamics
    ----------------------------------
    Exact discretization with zero-order hold:

        x[k+1] = x[k] + v[k]·Δt + (1/2)·u[k]·Δt²
        v[k+1] = v[k] + u[k]·Δt

    With process noise:
        x[k+1] = x[k] + v[k]·Δt + (1/2)·u[k]·Δt² + w_x[k]
        v[k+1] = v[k] + u[k]·Δt + w_v[k]

    **Matrix Form:**
        X[k+1] = Φ·X[k] + Γ·u[k] + w[k]

    where:
        X = [x, v]ᵀ
        
        Φ = [1  Δt]  (exact discrete-time dynamics)
            [0   1]
        
        Γ = [Δt²/2]  (exact zero-order hold)
            [Δt   ]
        
        w ~ N(0, Q)  (correlated process noise)

    Physical Interpretation
    -----------------------
    **Discrete-Time Dynamics:**

    Position update:
        x[k+1] = x[k] + v[k]·Δt + (1/2)·u[k]·Δt²

    Terms:
    1. x[k]: Current position (persistence)
    2. v[k]·Δt: Displacement from velocity (integration)
    3. (1/2)·u[k]·Δt²: Displacement from acceleration (double integration)

    This is exact kinematics for constant acceleration over [k·Δt, (k+1)·Δt].

    Velocity update:
        v[k+1] = v[k] + u[k]·Δt

    Exact integration of constant acceleration.

    **Process Noise:**

    Represents disturbances accumulated over sampling interval:
    - w_x[k]: Position error (from integrated velocity noise)
    - w_v[k]: Velocity error (from force disturbances)

    **Covariance Structure:**

    From continuous noise σ [m/(s²·√s)]:
        Q = σ²·[Δt³/3   Δt²/2]
                [Δt²/2   Δt   ]

    Non-diagonal! Position and velocity errors correlated.

    Key Features
    ------------
    **Exact Discretization:**
    No discretization error in deterministic part (kinematic equations exact).

    **Linear Dynamics:**
    Enables analytical Riccati solutions, optimal Kalman filter.

    **Controllability:**
    Completely controllable: rank[Γ, Φ·Γ] = 2

    **Observability:**
    Depends on measurement:
    - C = [1, 0]: Position only (observable)
    - C = [0, 1]: Velocity only (NOT observable)
    - C = I: Both (trivially observable)

    **Marginal Stability:**
    Eigenvalues: z = {1, 1} (on unit circle)
    - Not asymptotically stable (need feedback)
    - Not unstable (doesn't diverge)

    **Non-Stationary (Open-Loop):**
    Variance grows with k (no equilibrium without control).

    Mathematical Properties
    -----------------------
    **State Transition Matrix:**
        Φ = [1  Δt]
            [0   1]

    Eigenvalues: λ = 1 (double)
    Eigenvector: [1, 0]ᵀ (position mode)

    **Controllability Matrix:**
        C_ctrl = [Γ, Φ·Γ] = [Δt²/2   Δt³/2]
                              [Δt     Δt²  ]

    rank = 2 (fully controllable)

    **Observability Matrix (C = [1,0]):**
        O = [C    ] = [1   0 ]
            [C·Φ  ]   [1   Δt]

    rank = 2 (observable from position)

    **Process Noise Covariance:**

    Exact from continuous:
        Q = σ²·[Δt³/3   Δt²/2]
                [Δt²/2   Δt   ]

    Properties:
    - Positive definite (if σ > 0)
    - Symmetric (by construction)
    - Scales with Δt (smaller Δt → smaller Q)

    **Discrete Riccati Equation:**

    For LQR, DARE:
        P = Q_cost + Φᵀ·P·Φ - Φᵀ·P·Γ·(R + Γᵀ·P·Γ)⁻¹·Γᵀ·P·Φ

    For Kalman, dual DARE:
        Σ = Q_noise + Φ·Σ·Φᵀ - Φ·Σ·Cᵀ·(R_meas + C·Σ·Cᵀ)⁻¹·C·Σ·Φᵀ

    Physical Interpretation
    -----------------------
    **Sampling Period Δt:**
    - Sets digital control rate
    - Affects noise accumulation (Q ∝ Δt)
    - Determines controllability region
    - Trade-off: Fast sampling vs computation

    **Process Noise Intensity σ:**
    - From continuous: σ [m/(s²·√s)]
    - Creates discrete Q via integration
    - Physical: Force disturbances, model uncertainty

    **Why Position Variance Larger:**

    Q₁₁/Q₂₂ = Δt²/3

    For Δt = 0.1 s: Q₁₁/Q₂₂ = 0.01/3 ≈ 0.0033
    
    Velocity noise integrates twice to position → amplification.

    State Space
    -----------
    State: X[k] = [x[k], v[k]] ∈ ℝ²
        - x: Position [m] (unbounded)
        - v: Velocity [m/s] (unbounded)

    Control: u[k] ∈ ℝ
        - Acceleration command [m/s²]
        - Zero-order hold between samples

    Noise: w[k] = [w_x[k], w_v[k]] ~ N(0, Q)
        - Correlated Gaussian white noise
        - Accumulated over sampling interval

    Parameters
    ----------
    sigma : float, default=0.1
        Continuous noise intensity [m/(s²·√s)]
        - Determines discrete Q via integration
        - Typical: 0.01-1.0

    dt : float, default=0.1
        Sampling period [s]
        - Critical design parameter
        - Typical: 0.001-1.0 depending on application
        - Affects both dynamics and noise

    m : float, default=1.0
        Mass [kg] (optional, typically normalized to 1)

    Stochastic Properties
    ---------------------
    - System Type: LINEAR
    - Noise Type: ADDITIVE (constant)
    - Discrete: Yes (native discrete-time)
    - Noise Dimension: nw = 2 (position and velocity)
    - Stationary: No (open-loop)
    - Gaussian: Yes (linear preserves Gaussianity)
    - Exact Discretization: Yes (no error)

    Applications
    ------------
    **1. Digital LQG Control:**
    - Discrete DARE for optimal gain
    - Discrete Kalman filter for estimation
    - Real-time implementation

    **2. Embedded Control:**
    - Microcontroller implementation
    - Fixed-point arithmetic
    - Real-time constraints

    **3. Discrete Kalman Filter:**
    - Standard benchmark for testing
    - Numerical stability analysis
    - Joseph form, square-root filters

    **4. Model Predictive Control:**
    - Discrete-time optimization
    - Constraint handling
    - Receding horizon

    **5. Reinforcement Learning:**
    - Model-based RL (forward dynamics)
    - Discrete state-action updates
    - LQR as baseline

    **6. Motion Control:**
    - CNC machines
    - Robotics
    - Hard disk drives
    - Satellite control

    Numerical Simulation
    --------------------
    **Direct Evaluation:**
    No integration needed - direct update:
        X[k+1] = Φ·X[k] + Γ·u[k] + w[k]

    where w[k] ~ N(0, Q) sampled directly.

    **Efficiency:**
    - Matrix multiplication (fast)
    - No time-stepping error
    - No stability issues
    - Vectorizable for batch

    Discrete Kalman Filter Example
    -------------------------------
    Standard implementation:

    ```python
    # Initialize
    X_hat = np.zeros(2)
    P = np.eye(2)
    
    for k in range(N):
        # Prediction
        X_pred = Phi @ X_hat + Gamma * u[k]
        P_pred = Phi @ P @ Phi.T + Q
        
        # Update (if measurement available)
        if y[k] is not None:
            C = np.array([[1, 0]])  # Position measurement
            S = C @ P_pred @ C.T + R_meas
            K = P_pred @ C.T / S
            
            X_hat = X_pred + K * (y[k] - C @ X_pred)
            P = (np.eye(2) - K @ C) @ P_pred
        else:
            X_hat = X_pred
            P = P_pred
    ```

    Comparison with Continuous
    ---------------------------
    **Continuous:**
    - Continuous-time Riccati (differential)
    - Kalman-Bucy filter (continuous)
    - Theoretical foundation

    **Discrete:**
    - Discrete-time Riccati (algebraic)
    - Discrete Kalman filter (recursive)
    - Practical implementation

    **Conversion:**
    - Φ = exp(A·Δt) exactly
    - Q from integration exactly
    - No approximation needed

    Limitations
    -----------
    - Linear only (no nonlinearity)
    - Constant parameters (no time-variation)
    - Gaussian noise (no heavy tails)
    - No constraints (in base formulation)
    - No delays (zero computation time assumed)

    Extensions
    ----------
    - Add damping: Φ₂₂ = (1 - b·Δt)
    - Nonlinear: u·v coupling
    - Constraints: |x| ≤ x_max
    - Delays: Measurement delay d steps
    - Multi-rate: Different sampling for sensors/actuators
    """

    def define_system(
        self,
        sigma: float = 0.1,
        dt: float = 0.1,
        m: float = 1.0,
        noise_on_position: bool = False,
    ):
        """
        Define discrete stochastic double integrator dynamics.

        Parameters
        ----------
        sigma : float, default=0.1
            Continuous noise intensity [m/(s²·√s)]
            - Determines discrete Q matrix
            - Typical: 0.01-1.0

        dt : float, default=0.1
            Sampling period [s]
            - Critical parameter (affects dynamics and noise)
            - Typical: 0.001-1.0
            - Choose based on bandwidth requirements

        m : float, default=1.0
            Mass [kg] (typically normalized to 1)

        noise_on_position : bool, default=False
            If True, add independent position noise (less physical)
            If False, only velocity noise (physical)

        Notes
        -----
        **Exact Discretization:**

        State transition (exact):
            Φ = [1  Δt]  (from exp(A·Δt))
                [0   1]

        Control input (exact for zero-order hold):
            Γ = [Δt²/2]
                [Δt   ]

        **Process Noise Covariance:**

        From continuous noise σ integrated over Δt:
            Q = σ²·[Δt³/3   Δt²/2]
                    [Δt²/2   Δt   ]

        This is **exact** integration of Wiener process.

        **Structure:**
        - Q₁₁ ∝ Δt³: Position noise (from double integration)
        - Q₂₂ ∝ Δt: Velocity noise (from single integration)
        - Q₁₂ ∝ Δt²: Correlation (from integration)

        **Scaling:**
        Smaller Δt:
        - Φ closer to identity (less change per step)
        - Γ smaller (less control authority per step)
        - Q smaller (less noise per step)
        - Need more steps for same motion

        **Physical Validity:**

        Standard: Noise on velocity only (force disturbances)
        - Physically motivated (Newton's law)
        - Q from integration exact

        Optional: Add position noise (model uncertainty)
        - Less physical but accounts for unmodeled effects
        - Increases Q₁₁ independently

        **Sampling Rate Selection:**

        For desired closed-loop bandwidth ω_cl:
            Δt < 1/(10·ω_cl)

        Example: ω_cl = 10 rad/s → Δt < 0.01 s (100 Hz minimum)

        **Eigenvalues:**

        Open-loop: z = {1, 1} (unit circle)
        Closed-loop (with LQR): Inside unit circle (stable)

        Examples
        --------
        >>> # Fast servo (100 Hz)
        >>> fast = DiscreteStochasticDoubleIntegrator(
        ...     sigma=0.1,
        ...     dt=0.01
        ... )
        >>> 
        >>> # Medium rate (20 Hz)
        >>> medium = DiscreteStochasticDoubleIntegrator(
        ...     sigma=0.1,
        ...     dt=0.05
        ... )
        >>> 
        >>> # Check noise covariance scaling
        >>> Q_fast = fast.get_process_noise_covariance()
        >>> Q_medium = medium.get_process_noise_covariance()
        >>> print(f"Ratio Q₂₂: {Q_medium[1,1] / Q_fast[1,1]:.1f}")  # = dt_ratio
        """
        if sigma < 0:
            raise ValueError(f"sigma must be non-negative, got {sigma}")
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt}")
        if m <= 0:
            raise ValueError(f"m must be positive, got {m}")

        # Store parameters
        self.sigma_continuous = sigma
        self.m = m

        # State variables
        x, v = sp.symbols("x v", real=True)
        u = sp.symbols("u", real=True)

        # Parameters
        sigma_sym = sp.symbols("sigma", real=True, nonnegative=True)
        dt_sym = sp.symbols("dt", positive=True)
        m_sym = sp.symbols("m", positive=True)

        self.state_vars = [x, v]
        self.control_vars = [u]
        self.order = 1
        self._dt = dt  # REQUIRED for discrete system

        # DETERMINISTIC PART (Exact discretization)
        # x[k+1] = x[k] + v[k]·Δt + (1/2)·u[k]·Δt²
        # v[k+1] = v[k] + u[k]·Δt/m
        x_next = x + v * dt_sym + (sp.Rational(1, 2)) * u * dt_sym**2 / m_sym
        v_next = v + u * dt_sym / m_sym

        self._f_sym = sp.Matrix([x_next, v_next])

        # STOCHASTIC PART (Process noise)
        # Exact covariance from integrating continuous noise
        # Q = σ²·[Δt³/3   Δt²/2]
        #         [Δt²/2   Δt   ]
        
        # Compute Q elements symbolically
        Q_11 = sigma_sym**2 * dt_sym**3 / 3  # Position variance
        Q_22 = sigma_sym**2 * dt_sym         # Velocity variance  
        Q_12 = sigma_sym**2 * dt_sym**2 / 2  # Covariance
        
        if noise_on_position:
            # Add independent position noise (optional, less physical)
            Q_11 = Q_11 + sigma_sym**2 * dt_sym

        # Cholesky-like factorization: Q = L·Lᵀ where L is diffusion matrix
        # For exact structure, use symbolic sqrt
        L_11 = sp.sqrt(Q_11)
        L_21 = Q_12 / L_11
        L_22 = sp.sqrt(Q_22 - L_21**2)

        self.diffusion_expr = sp.Matrix([
            [L_11, 0],
            [L_21, L_22]
        ])

        self.parameters = {
            sigma_sym: sigma,
            dt_sym: dt,
            m_sym: m,
        }

        # Discrete-time SDE
        self.sde_type = "ito"

        # Output: Typically position only
        self._h_sym = sp.Matrix([x])

    def setup_equilibria(self):
        """
        Set up equilibrium points.

        Origin is only equilibrium (marginally stable without control).
        """
        self.add_equilibrium(
            "origin",
            x_eq=np.array([0.0, 0.0]),
            u_eq=np.array([0.0]),
            verify=True,
            stability="marginally_stable",
            notes="Marginally stable at z=1 (unit circle). Without feedback, "
                  "variance grows unboundedly. Requires LQR for asymptotic stability."
        )
        self.set_default_equilibrium("origin")

    def get_state_transition_matrix(self) -> np.ndarray:
        """
        Get exact state transition matrix Φ.

        Returns
        -------
        np.ndarray
            Φ = [1  Δt]
                [0   1]

        Examples
        --------
        >>> system = DiscreteStochasticDoubleIntegrator(dt=0.1)
        >>> Phi = system.get_state_transition_matrix()
        >>> print(f"Φ:\n{Phi}")
        """
        dt = self._dt
        return np.array([[1.0, dt],
                        [0.0, 1.0]])

    def get_control_matrix(self) -> np.ndarray:
        """
        Get exact control input matrix Γ (zero-order hold).

        Returns
        -------
        np.ndarray
            Γ = [Δt²/2]
                [Δt   ]

        Examples
        --------
        >>> system = DiscreteStochasticDoubleIntegrator(dt=0.1)
        >>> Gamma = system.get_control_matrix()
        >>> print(f"Γ:\n{Gamma}")
        """
        dt = self._dt
        m = self.m
        return np.array([[0.5 * dt**2 / m],
                        [dt / m]])

    def get_process_noise_covariance(self) -> np.ndarray:
        """
        Get exact process noise covariance Q.

        From continuous noise σ integrated over Δt:
            Q = σ²·[Δt³/3   Δt²/2]
                    [Δt²/2   Δt   ]

        Returns
        -------
        np.ndarray
            Process noise covariance Q (2×2)

        Examples
        --------
        >>> system = DiscreteStochasticDoubleIntegrator(sigma=0.1, dt=0.1)
        >>> Q = system.get_process_noise_covariance()
        >>> print(f"Q:\n{Q}")
        >>> print(f"Q₁₁/Q₂₂ = {Q[0,0]/Q[1,1]:.4f} (should be Δt²/3)")
        """
        sigma = self.sigma_continuous
        dt = self._dt
        
        Q_11 = sigma**2 * dt**3 / 3
        Q_22 = sigma**2 * dt
        Q_12 = sigma**2 * dt**2 / 2
        
        return np.array([[Q_11, Q_12],
                        [Q_12, Q_22]])

    def get_discrete_eigenvalues(self) -> np.ndarray:
        """
        Get eigenvalues of discrete-time system (open-loop).

        Returns
        -------
        np.ndarray
            Eigenvalues (both equal to 1)

        Notes
        -----
        Marginally stable: On unit circle boundary.

        Examples
        --------
        >>> system = DiscreteStochasticDoubleIntegrator(dt=0.1)
        >>> eigs = system.get_discrete_eigenvalues()
        >>> print(f"Eigenvalues: {eigs}")  # [1, 1]
        >>> print(f"On unit circle: {np.allclose(np.abs(eigs), 1)}")
        """
        Phi = self.get_state_transition_matrix()
        return np.linalg.eigvals(Phi)


# Convenience functions
def create_digital_servo(
    sampling_rate_hz: float = 100,
    noise_level: float = 0.1,
) -> DiscreteStochasticDoubleIntegrator:
    """
    Create discrete double integrator for digital servo system.

    Parameters
    ----------
    sampling_rate_hz : float, default=100
        Sampling rate [Hz] (samples per second)
    noise_level : float, default=0.1
        Process noise intensity

    Returns
    -------
    DiscreteStochasticDoubleIntegrator

    Examples
    --------
    >>> # 1 kHz servo (fast)
    >>> fast_servo = create_digital_servo(sampling_rate_hz=1000)
    >>> 
    >>> # 10 Hz control (slow)
    >>> slow_servo = create_digital_servo(sampling_rate_hz=10)
    """
    dt = 1.0 / sampling_rate_hz
    return DiscreteStochasticDoubleIntegrator(sigma=noise_level, dt=dt, m=1.0)


def create_lqg_benchmark_discrete(
    dt: float = 0.1,
    sigma: float = 0.1,
) -> DiscreteStochasticDoubleIntegrator:
    """
    Create standard discrete LQG benchmark configuration.

    Parameters
    ----------
    dt : float, default=0.1
        Sampling period [s]
    sigma : float, default=0.1
        Process noise intensity

    Returns
    -------
    DiscreteStochasticDoubleIntegrator

    Notes
    -----
    Standard test problem:
    - Unit mass (m=1)
    - 10 Hz sampling (dt=0.1)
    - Moderate noise (σ=0.1)
    
    Typical LQR cost: Q = diag(100, 10), R = 1

    Examples
    --------
    >>> # Create benchmark
    >>> benchmark = create_lqg_benchmark_discrete(dt=0.1, sigma=0.1)
    >>> 
    >>> # Design discrete LQR
    >>> Phi = benchmark.get_state_transition_matrix()
    >>> Gamma = benchmark.get_control_matrix()
    >>> 
    >>> Q_cost = np.diag([100.0, 10.0])
    >>> R_cost = np.array([[1.0]])
    >>> 
    >>> # Solve discrete Riccati (DARE)
    >>> from scipy.linalg import solve_discrete_are
    >>> P = solve_discrete_are(Phi, Gamma, Q_cost, R_cost)
    >>> K = np.linalg.inv(R_cost + Gamma.T @ P @ Gamma) @ Gamma.T @ P @ Phi
    >>> print(f"Discrete LQR gain: K = {K}")
    """
    return DiscreteStochasticDoubleIntegrator(sigma=sigma, dt=dt, m=1.0)


def create_spacecraft_discrete(
    orbit_period_seconds: float = 5400,
    sampling_rate_hz: float = 1.0,
    noise_level: float = 0.01,
) -> DiscreteStochasticDoubleIntegrator:
    """
    Create discrete model for spacecraft station-keeping.

    Parameters
    ----------
    orbit_period_seconds : float, default=5400
        Orbital period [s] (90 min for LEO)
    sampling_rate_hz : float, default=1.0
        Control update rate [Hz]
    noise_level : float, default=0.01
        Disturbance noise level

    Returns
    -------
    DiscreteStochasticDoubleIntegrator

    Notes
    -----
    Spacecraft position control with:
    - Slow sampling (1 Hz typical for station-keeping)
    - Low noise (space is quiet)
    - Digital controller on board computer

    Examples
    --------
    >>> # LEO satellite with 1 Hz control
    >>> sat = create_spacecraft_discrete(
    ...     orbit_period_seconds=5400,
    ...     sampling_rate_hz=1.0,
    ...     noise_level=0.01
    ... )
    """
    dt = 1.0 / sampling_rate_hz
    return DiscreteStochasticDoubleIntegrator(sigma=noise_level, dt=dt, m=1000.0)