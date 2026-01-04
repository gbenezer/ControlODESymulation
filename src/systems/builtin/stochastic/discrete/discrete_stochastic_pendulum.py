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
Discrete Stochastic Pendulum - Digital Control of Nonlinear Oscillator with Noise
==================================================================================

This module provides the discrete-time stochastic pendulum, combining nonlinear
dynamics with digital control implementation and process noise. The discrete
stochastic pendulum serves as:

- A benchmark for nonlinear discrete-time stochastic control (swing-up and stabilization)
- A test case for discrete extended/unscented Kalman filtering on unstable systems
- An illustration of Kramers' escape in discrete time (noise-induced transitions)
- A model for digital implementation of balance control (real embedded systems)
- A challenge problem for robust reinforcement learning with discrete actions

The discrete stochastic pendulum represents what actually gets implemented in
real robotic systems - sampled measurements, zero-order hold control, and
process disturbances between samples.

Physical Context
----------------

**Digital Pendulum Control Reality:**

Every real pendulum control system operates digitally:
- Encoder sampled at discrete intervals (typical: 1-100 ms)
- Controller computed on microcontroller/FPGA
- Torque command via DAC with zero-order hold
- Disturbances between samples (wind gusts, vibrations)

**Examples of Digital Pendulum Systems:**

1. **Rotary Inverted Pendulum (Furuta):**
   - Educational lab system (Quanser, Feedback)
   - Sampling: Δt = 1-10 ms (100-1000 Hz)
   - Encoder: 2048-4096 counts/rev (quantization)
   - Motor: PWM control (discrete switching)
   - Noise: Encoder noise, motor ripple, friction variations

2. **Reaction Wheel Pendulum:**
   - Inertial stabilization (spacecraft analog)
   - Sampling: Δt = 10-50 ms
   - IMU + encoder measurements
   - Brushless motor control
   - Noise: Gyro drift, wheel imbalance

3. **Acrobot/Pendubot:**
   - Underactuated robotics research
   - Sampling: Δt = 5-20 ms
   - Joint encoders
   - DC motor actuation
   - Noise: Backlash, cable stretch, bearing friction

4. **Segway/Balance Robots:**
   - Real-time balance (human-machine)
   - Sampling: Δt = 1-5 ms (critical for stability)
   - IMU + wheel encoders
   - Wheel motors with friction
   - Noise: Ground vibration, rider movement

**Why Discrete Model Essential:**

Continuous model: Theoretical (infinite sampling, smooth control)
Discrete model: Practical (actual hardware constraints)

Critical differences:
- **Sampling delay:** Information Δt old when control applied
- **Zero-order hold:** Torque constant between samples (not smooth)
- **Quantization:** Finite encoder resolution (angle discretization)
- **Intersample disturbances:** Noise affects system between observations

Mathematical Formulation
------------------------

**Discrete-Time Pendulum Dynamics:**

Exact discretization (Euler for simplicity):
    θ[k+1] = θ[k] + ω[k]·Δt
    ω[k+1] = ω[k] + (-(g/L)·sin(θ[k]) - b·ω[k] + u[k])·Δt

With process noise:
    θ[k+1] = θ[k] + ω[k]·Δt + w_θ[k]
    ω[k+1] = ω[k] + (-(g/L)·sin(θ[k]) - b·ω[k] + u[k])·Δt + w_ω[k]

**Matrix Form:**
    X[k+1] = f(X[k], u[k]) + w[k]

where:
    X = [θ, ω]ᵀ
    f: Nonlinear discrete-time dynamics (Euler discretization)
    w ~ N(0, Q): Process noise

**Noise Covariance:**

From continuous noise σ integrated over Δt:
    Q ≈ σ²·[Δt   0  ]  (simplified, diagonal)
              [0    Δt]

More accurate: Account for integration structure.

**Zero-Order Hold Control:**

Control u[k] held constant during [k·Δt, (k+1)·Δt]:
- Step-wise constant (not smooth)
- Creates sampling artifacts
- Requires careful stability analysis

Discrete Kramers Escape
------------------------

**Continuous Kramers:**
Mean escape time: τ ~ exp(ΔE/σ²)

**Discrete Analog:**

For discrete-time with sampling Δt:
- Noise per step: σ_d = σ_c·√Δt
- Effective barrier: ΔE (same)
- Escape time: τ_discrete ~ exp(ΔE/(σ_d²))

**Sampling Rate Effect:**

Faster sampling (smaller Δt):
- Smaller noise per step: σ_d ↓
- More steps needed for escape
- Longer mean escape time (more stable)

Slower sampling (larger Δt):
- Larger noise per step: σ_d ↑
- Fewer steps to escape
- Shorter mean escape time (less stable)

**Practical Implication:**
Fast sampling improves robustness to noise!

Discrete Extended Kalman Filter
--------------------------------

**Why EKF Needed:**

Measurements noisy:
    y[k] = h(X[k]) + v[k]
    y[k] = θ[k] + v[k] (angle measurement)

Must estimate state from noisy observations.

**EKF for Pendulum:**

**Prediction:**
    X̂[k+1|k] = f(X̂[k|k], u[k])
    F[k] = ∂f/∂X |_{X̂[k|k]} (Jacobian)
    P[k+1|k] = F[k]·P[k|k]·F[k]ᵀ + Q

**Update:**
    H[k] = ∂h/∂X |_{X̂[k+1|k]} (measurement Jacobian)
    K[k] = P[k+1|k]·H[k]ᵀ·(H[k]·P[k+1|k]·H[k]ᵀ + R)⁻¹
    X̂[k+1|k+1] = X̂[k+1|k] + K[k]·(y[k+1] - h(X̂[k+1|k]))
    P[k+1|k+1] = (I - K[k]·H[k])·P[k+1|k]

**Challenges:**
- Nonlinear: sin(θ) in dynamics
- Unstable (upright): Errors grow exponentially
- Noise on unstable system: EKF can diverge

**UKF Alternative:**
Unscented Kalman Filter better for highly nonlinear.

Reinforcement Learning
-----------------------

**Discrete Pendulum RL Benchmark:**

Unlike continuous, discrete is what RL actually uses:
- State: (θ, ω) discretized or continuous
- Action: u[k] discrete or continuous
- Reward: -cost(θ, ω, u) per step
- Episode: Terminates if fallen

**Robustness:**

Training with noise (domain randomization):
- Random g, L, b (parameter uncertainty)
- Random σ_θ, σ_ω (process noise)
- Sim-to-real transfer improved

**Algorithms:**
- DQN: Discrete actions (left, right, none)
- DDPG, SAC: Continuous actions
- Model-based: Use this discrete model as forward predictor

Applications
------------

**1. Robotics:**

**Digital Balance Control:**
- Segway: 100-1000 Hz control loop
- Humanoid: 200-500 Hz
- Inverted pendulum lab: 50-200 Hz

**2. Reinforcement Learning:**

**OpenAI Gym:**
Pendulum-v1 environment (discrete-time):
- State: [cos(θ), sin(θ), ω]
- Action: Torque [-2, 2]
- Reward: -(θ² + 0.1·ω² + 0.001·u²)

**3. Embedded Systems:**

**Microcontroller Implementation:**
- Fixed-point arithmetic
- Deterministic timing (RTOS)
- Limited computation (simple controller)

**4. Control Education:**

**Laboratory Experiments:**
- Students implement discrete control
- Measure actual noise statistics
- Compare theory (continuous) vs practice (discrete)

**5. Fault Detection:**

**Anomaly Detection:**
- EKF residuals: r[k] = y[k] - ŷ[k|k-1]
- Statistical tests on r[k]
- Detect bearing failure, sensor faults

Sampling Time Selection
-----------------------

**Critical for Unstable System:**

Upright pendulum has unstable eigenvalue:
    λ_unstable = √(g/L)

For stability under discrete control, need:
    Δt < Δt_critical

**Guidelines:**

1. **Nyquist:**
   Δt < π/λ_unstable ≈ π/√(g/L)

2. **Control Bandwidth:**
   Sample 10× faster than desired closed-loop bandwidth:
   Δt < 1/(10·ω_cl)

3. **Practical:**
   - Fast (research): Δt = 1-5 ms
   - Moderate (education): Δt = 10-20 ms
   - Slow (demonstration): Δt = 50-100 ms

**For Pendulum (L=0.5m):**
- ω₀ ≈ 4.4 rad/s
- Nyquist: Δt < 0.71 s
- Control (ω_cl = 10): Δt < 0.01 s = 10 ms
- Practical: Δt = 5-10 ms typical

Noise Scaling
-------------

**Continuous Noise σ_c [rad/(s²·√s)]:**

Discrete noise σ_d [rad/s per step]:
    σ_d = σ_c·√Δt

**Example:**
- σ_c = 1.0 rad/(s²·√s)
- Δt = 0.01 s (10 ms)
- σ_d = 1.0·√0.01 = 0.1 rad/s per step

**Sampling Rate Effect:**

Faster sampling (smaller Δt):
- Smaller noise per step
- More stable to disturbances
- Better state estimation

Slower sampling (larger Δt):
- Larger noise per step
- Less stable
- Poorer estimation

Common Pitfalls
---------------

1. **Euler Instability:**
   - Standard Euler unstable for large Δt
   - Especially with nonlinear sin(θ)
   - Use RK4 or ensure small Δt

2. **Angle Wrapping:**
   - θ can exceed ±π (full rotations)
   - Must handle wrapping for control
   - Unwrapped for swing-up, wrapped for balance

3. **Quantization:**
   - Real encoders: Finite resolution
   - Creates limit cycles
   - Can destabilize marginal stability

4. **EKF Divergence:**
   - Upright is unstable
   - Linearization errors grow
   - Need small Δt, good initialization

5. **Zero-Order Hold:**
   - Control discontinuous (steps)
   - Creates high-frequency content
   - Can excite unmodeled modes

6. **Noise Underestimation:**
   - Small noise still causes falling (exponential instability)
   - Need Monte Carlo for reliability

**Impact:**
Discrete pendulum demonstrated:
- Digital control theory practical
- Nonlinear control implementable on modest hardware
- RL can solve continuous control problems
- Sim-to-real gap manageable with robustness

"""

import numpy as np
import sympy as sp
from typing import Optional

from src.systems.base.core.discrete_stochastic_system import DiscreteStochasticSystem


class DiscreteStochasticPendulum(DiscreteStochasticSystem):
    """
    Discrete-time stochastic pendulum for digital control and RL.

    Represents a pendulum sampled at discrete intervals with process noise,
    suitable for digital control implementation, discrete Kalman filtering,
    and reinforcement learning applications with realistic disturbances.

    Discrete-Time Stochastic Dynamics
    ----------------------------------
    Euler discretization with noise:

        θ[k+1] = θ[k] + ω[k]·Δt + w_θ[k]
        ω[k+1] = ω[k] + (-(g/L)·sin(θ[k]) - b·ω[k] + u[k])·Δt + w_ω[k]

    where:
        - θ[k]: Angle at time k·Δt [rad]
        - ω[k]: Angular velocity at time k·Δt [rad/s]
        - u[k]: Applied torque (zero-order hold) [rad/s²]
        - w_θ[k]: Angle noise [rad per step]
        - w_ω[k]: Angular velocity noise [rad/s per step]
        - Δt: Sampling period [s]

    **Deterministic Part:**
    Same nonlinear pendulum dynamics as continuous.

    **Stochastic Part:**
    Noise accumulated over sampling interval [k·Δt, (k+1)·Δt].

    **Zero-Order Hold:**
    Control u[k] constant during interval (digital implementation).

    Physical Interpretation
    -----------------------
    **Digital Control Loop:**

    1. **Sample:** Read encoder at t = k·Δt
       - Measure θ[k] (with quantization)
       - Estimate ω[k] (from differences or gyro)

    2. **Compute:** Calculate control u[k]
       - LQR, swing-up, learned policy
       - Computation time: τ_comp (typically << Δt)

    3. **Actuate:** Output u[k] via DAC/PWM
       - Held constant until t = (k+1)·Δt
       - Zero-order hold

    4. **Disturbances:** Between samples
       - Wind gusts: w_ω
       - Ground vibration: w_θ
       - Friction variations
       - Unmodeled dynamics

    **Noise Sources:**

    Angle noise w_θ[k]:
    - Direct angle disturbances (rare physically)
    - Encoder quantization (measurement as process noise)
    - Model uncertainty
    - Typical: 0.001-0.01 rad per step

    Angular velocity noise w_ω[k]:
    - Torque disturbances (wind, vibration)
    - Motor ripple, cogging
    - Bearing friction variations
    - Most critical for stability
    - Typical: 0.01-0.1 rad/s per step

    **Conversion from Continuous:**

    For continuous σ_c [rad/(s²·√s)]:
        σ_d = σ_c·√Δt [rad/s per step]

    Example: σ_c = 1.0, Δt = 0.01 s
        → σ_d = 0.1 rad/s per step

    Key Features
    ------------
    **Nonlinearity:**
    sin(θ) creates bistability and complex dynamics.

    **Unstable Equilibrium:**
    Upright (θ=0) exponentially unstable without control.

    **Bistability:**
    Two potential wells (downward stable, upward unstable).

    **Discrete Sampling:**
    Finite Δt creates sampling effects (aliasing, delay).

    **Zero-Order Hold:**
    Control discontinuous (step-wise).

    **Process Noise:**
    Disturbances between samples.

    Mathematical Properties
    -----------------------
    **Equilibria (Deterministic Part):**

    Downward: θ = 0, ω = 0 (stable)
    Upward: θ = π, ω = 0 (unstable)

    **Linearization (Small Angle):**

    Near downward (θ ≈ 0):
        [θ[k+1]] ≈ [1      Δt    ]·[θ[k]] + [0 ]·u[k]
        [ω[k+1]]   [-g/L·Δt  1-b·Δt] [ω[k]]   [Δt]

    Eigenvalues determine discrete stability.

    **Stability Condition (Linear):**

    For stable downward equilibrium:
    Eigenvalues of Φ must satisfy |λ| < 1.

    Requires: Δt < 2/(b + √(b² + 4g/L))

    **Upright (θ ≈ π):**

    Unstable - one eigenvalue |λ| > 1.
    Requires fast sampling and feedback.

    Physical Interpretation
    -----------------------
    **Sampling Period Δt:**
    - Digital control rate
    - Critical for unstable upright
    - Affects noise accumulation
    - Trade-off: Fast vs computation

    **Damping b:**
    - Energy dissipation [1/s]
    - Higher b: Easier to control
    - Typical: 0.1-1.0

    **Noise Intensity σ:**
    - Angular disturbance magnitude
    - Higher σ: More falls, harder balance
    - Typical: 0.01-0.1 rad/s per step

    State Space
    -----------
    State: X[k] = [θ[k], ω[k]]
        - θ: Angle (periodic, or unwrapped)
        - ω: Angular velocity

    Control: u[k] ∈ ℝ
        - Torque (zero-order hold)
        - Bounded: |u| ≤ u_max typically

    Noise: w[k] = [w_θ[k], w_ω[k]]
        - Gaussian per step
        - Accumulated over Δt

    Parameters
    ----------
    g : float, default=9.81
        Gravity [m/s²]

    L : float, default=1.0
        Pendulum length [m]

    b : float, default=0.5
        Damping [1/s]

    sigma_theta : float, default=0.01
        Angle noise [rad per step]
        - Encoder noise, model uncertainty
        - Typical: 0.001-0.01

    sigma_omega : float, default=0.05
        Angular velocity noise [rad/s per step]
        - Torque disturbances (most critical)
        - Convert from continuous: σ_c·√Δt
        - Typical: 0.01-0.1

    dt : float, default=0.01
        Sampling period [s]
        - Critical parameter
        - Typical: 0.001-0.1 s
        - Faster for upright stabilization

    m : float, default=1.0
        Mass [kg] (optional)

    Stochastic Properties
    ---------------------
    - System Type: NONLINEAR
    - Noise Type: ADDITIVE
    - Discrete: Yes (native discrete-time)
    - Stationary: No (open-loop)
    - Bistable: Yes (two equilibria)
    - Unstable: Yes (upright)

    Applications
    ------------
    **1. Digital Control:**
    - Microcontroller implementation
    - Real-time OS (RTOS)
    - Fixed-point arithmetic

    **2. State Estimation:**
    - Discrete EKF/UKF
    - Particle filter
    - Complementary filter (IMU+encoder)

    **3. Reinforcement Learning:**
    - OpenAI Gym Pendulum-v1
    - Discrete action spaces
    - Robust RL with noise

    **4. Embedded Systems:**
    - Arduino, STM32 implementation
    - FPGA for ultra-fast control
    - Education/research platforms

    **5. Robustness Analysis:**
    - Monte Carlo falling probability
    - Mean time to failure
    - Sensitivity to noise

    Numerical Simulation
    --------------------
    **Direct Evaluation:**
        X[k+1] = f(X[k], u[k]) + w[k]

    No integration needed (discrete-time native).

    **Euler Discretization:**
    Standard for discrete pendulum (matches RL benchmarks).

    **Higher-Order (RK4):**
    More accurate but less common in practice.

    Discrete EKF Implementation
    ----------------------------
    **Jacobian (Linearization):**

    F = ∂f/∂X = [1           Δt                    ]
                 [-g/L·cos(θ)·Δt  1 - b·Δt]

    At θ=0 (downward): cos(0) = 1
    At θ=π (upright): cos(π) = -1 (sign flip!)

    Comparison with Continuous
    ---------------------------
    **Continuous Stochastic:**
    - dW Brownian motion
    - σ in [rad/(s²·√s)]
    - SDE integration (Euler-Maruyama)
    - Theoretical foundation

    **Discrete Stochastic:**
    - w[k] Gaussian per step
    - σ in [rad/s per step]
    - Direct evaluation
    - Practical implementation

    **Conversion:**
        σ_discrete = σ_continuous·√Δt

    Limitations
    -----------
    - Euler discretization: O(Δt) error
    - Additive noise (not multiplicative)
    - 1D pendulum (no 3D motion)
    - Rigid body (no flexibility)

    Extensions
    ----------
    - RK4 discretization (higher accuracy)
    - Double pendulum (chaos + noise)
    - Flexible pendulum
    - 3D pendulum (spherical)
    - Multiplicative noise
    
    See Also
    --------
    StochasticPendulum : Continuous-time version
    Pendulum : Deterministic discrete version
    DiscreteStochasticDoubleIntegrator : Simpler linear system
    """

    def define_system(
        self,
        g: float = 9.81,
        L: float = 1.0,
        b: float = 0.5,
        sigma_theta: float = 0.01,
        sigma_omega: float = 0.05,
        dt: float = 0.01,
        m: float = 1.0,
    ):
        """
        Define discrete stochastic pendulum dynamics.

        Parameters
        ----------
        g : float, default=9.81
            Gravity [m/s²]

        L : float, default=1.0
            Pendulum length [m]

        b : float, default=0.5
            Damping [1/s]

        sigma_theta : float, default=0.01
            Angle noise per step [rad]
            - Encoder quantization, model error
            - Typical: 0.001-0.01

        sigma_omega : float, default=0.05
            Angular velocity noise per step [rad/s]
            - Torque disturbances (critical for upright)
            - Convert: σ_c·√Δt from continuous
            - Typical: 0.01-0.1

        dt : float, default=0.01
            Sampling period [s]
            - Typical: 0.001-0.1
            - Faster for upright control
            - Balance: Fast enough vs computation

        m : float, default=1.0
            Mass [kg]

        Notes
        -----
        **Sampling Rate Selection:**

        For upright stabilization:
        - Minimum: Δt < π/√(g/L) (Nyquist-like)
        - Recommended: Δt < 0.1·√(L/g) (10× rule)
        - For L=1m, g=9.81: Δt < 0.03 s = 30 ms

        Practical:
        - Fast (research): 1-5 ms
        - Moderate (education): 10-20 ms
        - Slow (demo): 50-100 ms

        **Noise Scaling:**

        From continuous σ_c [rad/(s²·√s)]:
            σ_discrete = σ_c·√Δt

        Example: σ_c = 1.0, Δt = 0.01
            → σ_d = 0.1 rad/s per step

        **Stability (Discrete Linearized):**

        At downward (θ=0):
        Eigenvalues of:
            Φ = [1           Δt        ]
                [-g/L·Δt   1-b·Δt]

        Stable if |λ| < 1 for both eigenvalues.

        Requires: Δt small enough.

        **Discretization Method:**

        This uses Euler (most common for RL/embedded):
        - Simple, explicit
        - O(Δt) error
        - Matches OpenAI Gym

        Alternative: RK4 (more accurate, more computation).

        **Noise Placement:**

        On both θ and ω equations (general).

        Physical: Primarily on ω (torque disturbances).
        Can set σ_θ = 0 for purely physical model.
        """
        if g <= 0:
            raise ValueError(f"g must be positive, got {g}")
        if L <= 0:
            raise ValueError(f"L must be positive, got {L}")
        if b < 0:
            raise ValueError(f"b must be non-negative, got {b}")
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt}")
        if m <= 0:
            raise ValueError(f"m must be positive, got {m}")

        # Store parameters
        self.g = g
        self.L = L
        self.b = b
        self.m = m

        # State variables
        theta, omega = sp.symbols("theta omega", real=True)
        u = sp.symbols("u", real=True)

        # Parameters
        g_sym, L_sym, b_sym, m_sym = sp.symbols("g L b m", positive=True)
        dt_sym = sp.symbols("dt", positive=True)
        sigma_theta_sym = sp.symbols("sigma_theta", nonnegative=True)
        sigma_omega_sym = sp.symbols("sigma_omega", nonnegative=True)

        self.state_vars = [theta, omega]
        self.control_vars = [u]
        self.order = 1
        self._dt = dt  # REQUIRED

        # DETERMINISTIC PART (Euler discretization)
        # θ[k+1] = θ[k] + ω[k]·Δt
        # ω[k+1] = ω[k] + (-(g/L)·sin(θ) - b·ω + u)·Δt

        theta_next = theta + omega * dt_sym
        omega_next = omega + (-(g_sym/L_sym) * sp.sin(theta) - b_sym * omega + u) * dt_sym

        self._f_sym = sp.Matrix([theta_next, omega_next])

        self.parameters = {
            g_sym: g,
            L_sym: L,
            b_sym: b,
            m_sym: m,
            dt_sym: dt,
            sigma_theta_sym: sigma_theta,
            sigma_omega_sym: sigma_omega,
        }

        # STOCHASTIC PART (process noise)
        self.diffusion_expr = sp.Matrix([
            [sigma_theta_sym, 0],
            [0, sigma_omega_sym]
        ])

        self.sde_type = "ito"

        # Output: Angle (typical measurement)
        self._h_sym = sp.Matrix([theta])

    def setup_equilibria(self):
        """Set up equilibrium points."""
        # Downward (stable)
        self.add_equilibrium(
            "downward",
            x_eq=np.array([0.0, 0.0]),
            u_eq=np.array([0.0]),
            verify=True,
            stability="stable",
            notes="Stable equilibrium (hanging down). With noise, oscillates around this."
        )

        # Upward (unstable)
        self.add_equilibrium(
            "upward",
            x_eq=np.array([np.pi, 0.0]),
            u_eq=np.array([0.0]),
            verify=True,
            stability="unstable",
            notes="Unstable equilibrium (inverted). Noise causes falling. "
                  "Mean time to fall depends exponentially on σ_ω."
        )

        self.set_default_equilibrium("downward")

    def get_natural_frequency(self) -> float:
        """Get natural frequency ω₀ = √(g/L) [rad/s]."""
        return np.sqrt(self.g / self.L)

    def get_nyquist_limit(self) -> float:
        """
        Get approximate Nyquist-like sampling limit.

        Returns
        -------
        float
            Maximum recommended Δt [s]

        Notes
        -----
        Rule: Δt < π/ω₀ for capturing dynamics.

        Examples
        --------
        >>> pend = DiscreteStochasticPendulum(g=9.81, L=1.0)
        >>> dt_max = pend.get_nyquist_limit()
        >>> print(f"Max Δt ≈ {dt_max:.3f} s")
        """
        omega_0 = self.get_natural_frequency()
        return np.pi / omega_0

    def compute_energy(self, x: np.ndarray) -> float:
        """
        Compute mechanical energy.

        Parameters
        ----------
        x : np.ndarray
            State [θ, ω]

        Returns
        -------
        float
            Energy [J/kg]

        Examples
        --------
        >>> pend = DiscreteStochasticPendulum()
        >>> x = np.array([0.5, 1.0])
        >>> E = pend.compute_energy(x)
        >>> print(f"Energy: {E:.3f}")
        """
        theta, omega = x
        KE = 0.5 * omega**2
        PE = (self.g / self.L) * (1 - np.cos(theta))
        return KE + PE

    def get_noise_intensities(self) -> dict:
        """Get noise parameters."""
        return {
            'sigma_theta': self.parameters[sp.symbols('sigma_theta')],
            'sigma_omega': self.parameters[sp.symbols('sigma_omega')],
        }


# Convenience functions
def create_furuta_pendulum(
    sampling_rate_hz: float = 200,
    noise_level: str = 'low',
) -> DiscreteStochasticPendulum:
    """
    Create discrete model for Furuta (rotary inverted) pendulum.

    Parameters
    ----------
    sampling_rate_hz : float, default=200
        Control loop rate [Hz]
    noise_level : str, default='low'
        'low', 'medium', or 'high'

    Returns
    -------
    DiscreteStochasticPendulum

    Examples
    --------
    >>> # Quanser-like system (200 Hz)
    >>> furuta = create_furuta_pendulum(sampling_rate_hz=200, noise_level='low')
    """
    dt = 1.0 / sampling_rate_hz

    noise_presets = {
        'low': {'sigma_theta': 0.001, 'sigma_omega': 0.01},
        'medium': {'sigma_theta': 0.005, 'sigma_omega': 0.03},
        'high': {'sigma_theta': 0.01, 'sigma_omega': 0.1},
    }

    noise = noise_presets.get(noise_level, noise_presets['low'])

    # Scale for sampling period
    noise_scaled = {
        'sigma_theta': noise['sigma_theta'] * np.sqrt(dt / 0.01),
        'sigma_omega': noise['sigma_omega'] * np.sqrt(dt / 0.01),
    }

    return DiscreteStochasticPendulum(
        g=9.81,
        L=0.3,  # Typical Furuta
        b=0.1,  # Low damping
        dt=dt,
        **noise_scaled
    )


def create_rl_pendulum(
    dt: float = 0.05,
) -> DiscreteStochasticPendulum:
    """
    Create pendulum matching OpenAI Gym Pendulum-v1 (with noise).

    Parameters
    ----------
    dt : float, default=0.05
        Time step [s] (Gym default: 0.05)

    Returns
    -------
    DiscreteStochasticPendulum

    Notes
    -----
    Matches Gym parameters with added process noise for robustness.

    Examples
    --------
    >>> # RL training environment
    >>> rl_pend = create_rl_pendulum(dt=0.05)
    """
    return DiscreteStochasticPendulum(
        g=10.0,  # Gym uses g=10
        L=1.0,
        b=0.0,   # Gym has no damping
        sigma_theta=0.005,
        sigma_omega=0.05,
        dt=dt,
        m=1.0
    )