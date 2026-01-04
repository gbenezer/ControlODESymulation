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
Discrete-time harmonic oscillator system.

This module provides discrete-time implementations of the damped harmonic
oscillator, a fundamental second-order system that appears throughout physics
and engineering. It serves as:
- A canonical example of oscillatory dynamics
- A model for vibration analysis and control
- A testbed for studying resonance and damping
- The linearization of many nonlinear systems (pendulum, mass-spring, etc.)

The harmonic oscillator represents discretized spring-mass-damper dynamics,
making it applicable to:
- Mechanical vibrations (springs, suspensions, structures)
- Electrical circuits (RLC circuits - inductor, resistor, capacitor)
- Acoustic systems (sound waves, resonators)
- Seismic analysis (building response to earthquakes)
- Control systems (actuator dynamics, sensor dynamics)
- Molecular dynamics (bond vibrations)
"""

from typing import Optional, Tuple

import numpy as np
import sympy as sp
from scipy import signal

from src.systems.base.core.discrete_symbolic_system import DiscreteSymbolicSystem


class DiscreteOscillator(DiscreteSymbolicSystem):
    """
    Discrete-time damped harmonic oscillator with external forcing.

    Physical System:
    ---------------
    The harmonic oscillator models a mass attached to a spring and damper,
    subject to external force. The continuous-time dynamics are:

        m·ẍ + c·ẋ + k·x = F(t)

    Dividing by mass and defining natural frequency ωₙ and damping ratio ζ:

        ẍ + 2ζωₙ·ẋ + ωₙ²·x = u(t)

    where:
        ωₙ = √(k/m)  [rad/s] : Natural (undamped) frequency
        ζ = c/(2√(km)) [-]   : Damping ratio

    The discrete-time version depends on the discretization method. This
    implementation provides several options:

    **Zero-Order Hold (ZOH) - Exact Discretization:**
    The exact solution assuming piecewise-constant control. For underdamped
    systems (0 < ζ < 1), this gives:

        x[k+1] = e^(-ζωₙdt)·[x[k]·cos(ωd·dt) + v[k]·sin(ωd·dt)/ωd]
        v[k+1] = e^(-ζωₙdt)·[-x[k]·ωd·sin(ωd·dt) + v[k]·cos(ωd·dt)] + dt·u[k]

    where ωd = ωₙ√(1 - ζ²) is the damped natural frequency.

    **Tustin/Bilinear Transform:**
    Preserves frequency response characteristics:

        (I - 0.5·dt·A)·x[k+1] = (I + 0.5·dt·A)·x[k] + 0.5·dt·B·(u[k] + u[k+1])

    This is the standard method for designing discrete filters from continuous
    prototypes, as it maps the jω axis in s-plane to the unit circle in z-plane.

    **Forward Euler:**
    Simple first-order approximation:

        x[k+1] = x[k] + dt·v[k]
        v[k+1] = v[k] + dt·(-ωₙ²·x[k] - 2ζωₙ·v[k] + u[k])

    **Backward Euler:**
    Implicit method with better stability for stiff systems.

    State Space:
    -----------
    State: x[k] = [x[k], v[k]]
        Displacement state:
        - x: Position/displacement from equilibrium [m]
          * Can be positive or negative
          * Zero at equilibrium (spring relaxed)
          * Bounded for stable systems: |x| < ∞
          * Oscillates around zero for underdamped systems

        Velocity state:
        - v: Velocity [m/s]
          * Rate of change of position
          * Zero at equilibrium
          * Phase-shifted 90° from position in steady-state oscillation
          * Maximum at equilibrium crossing for underdamped

    Control: u[k] = [F[k]]
        - F: External force/acceleration [m/s²] or [N/kg]
          * Directly enters as acceleration term
          * Can be used to:
            - Excite oscillations (resonance testing)
            - Damp oscillations (active damping)
            - Track reference trajectories
            - Reject disturbances

    Output: y[k] = [x[k]] or [x[k], v[k]]
        - Displacement-only measurement (most common):
          y[k] = x[k]
          Examples: LVDT, laser displacement, strain gauge
        - Full state measurement:
          y[k] = [x[k], v[k]]
          Examples: accelerometer integration, optical tracking

    Dynamics (General Form):
    ------------------------
    The discrete dynamics can be written in state-space form:

        x[k+1] = Ad·x[k] + Bd·u[k]

    where the matrices Ad and Bd depend on the discretization method and
    system parameters (ωₙ, ζ, dt).

    **Physical Interpretation of Parameters:**

    Natural Frequency ωₙ [rad/s]:
    - Frequency of oscillation without damping
    - Higher ωₙ → faster oscillations, stiffer spring
    - Related to spring constant: ωₙ = √(k/m)
    - Typical values:
      * Building structures: 0.1-10 rad/s
      * Vehicle suspensions: 5-50 rad/s
      * Machine tools: 100-1000 rad/s
      * MEMS devices: 10³-10⁶ rad/s

    Damping Ratio ζ [-]:
    - Dimensionless measure of energy dissipation
    - Controls how quickly oscillations decay
    - ζ = 0: Undamped (oscillates forever)
    - 0 < ζ < 1: Underdamped (oscillates with decay)
    - ζ = 1: Critically damped (fastest non-oscillatory)
    - ζ > 1: Overdamped (slow return, no oscillation)
    - Typical values:
      * Air damping: ζ ≈ 0.01-0.1
      * Automotive shocks: ζ ≈ 0.3-0.7
      * Seismic isolators: ζ ≈ 0.1-0.3
      * Critically damped instruments: ζ = 1.0

    **Regime Classification:**

    1. **Underdamped (0 < ζ < 1):**
       - Oscillates with exponentially decaying amplitude
       - Damped frequency: ωd = ωₙ√(1 - ζ²) < ωₙ
       - Decay rate: σ = ζωₙ
       - Period: T = 2π/ωd
       - Envelope: A(t) = A₀·e^(-ζωₙt)
       - Most common in practice

    2. **Critically Damped (ζ = 1):**
       - Fastest return to equilibrium without overshoot
       - No oscillation
       - Optimal for instruments, door closers
       - Two equal real eigenvalues: λ = -ωₙ

    3. **Overdamped (ζ > 1):**
       - Slow return to equilibrium
       - No oscillation
       - Two distinct real eigenvalues
       - Example: heavily damped pendulum in oil

    4. **Undamped (ζ = 0):**
       - Pure sinusoidal oscillation forever
       - No energy dissipation
       - Idealization (never exact in reality)
       - Eigenvalues on imaginary axis: λ = ±jωₙ

    Parameters:
    ----------
    omega_n : float, default=1.0
        Natural frequency [rad/s]
        Must be positive: ωₙ > 0
        Controls oscillation speed
        Related to spring stiffness and mass

    zeta : float, default=0.1
        Damping ratio [-]
        Must be non-negative: ζ ≥ 0
        Controls energy dissipation
        ζ = 0: undamped, ζ < 1: underdamped, ζ = 1: critical, ζ > 1: overdamped

    dt : float, default=0.1
        Sampling/discretization time step [s]
        Critical parameter affecting:
        - Accuracy of discrete approximation
        - Aliasing (must satisfy Nyquist: dt < π/ωₙ)
        - Numerical stability (depends on method)
        - Control bandwidth

        Guidelines:
        - Nyquist: dt < π/ωₙ (sample at least 2× per cycle)
        - Shannon: dt < 1/(10·fₙ) for good reconstruction
        - Rule of thumb: 10-20 samples per period T = 2π/ωₙ
        - For ωₙ = 1 rad/s: dt ≈ 0.1-0.3 s
        - For ωₙ = 10 rad/s: dt ≈ 0.01-0.03 s

    method : str, default='zoh'
        Discretization method:
        - 'zoh': Zero-order hold (exact, recommended for control)
        - 'tustin': Bilinear/trapezoidal (preserves frequency response)
        - 'euler': Forward Euler (simple, less accurate)
        - 'backward_euler': Backward Euler (more stable)
        - 'matched': Matched pole-zero (preserves poles)

    Equilibria:
    ----------
    **Stable equilibrium at origin:**
        x_eq = [0, 0]  (rest position, spring relaxed)
        u_eq = 0       (no external force)

    Stability depends on damping:
    - ζ > 0: Asymptotically stable (returns to origin)
    - ζ = 0: Marginally stable (oscillates forever)

    **Eigenvalue Analysis:**

    Continuous-time poles:
        λ = -ζωₙ ± jωₙ√(1 - ζ²)  (for ζ < 1)

    Discrete-time poles (ZOH):
        z = e^(λdt) = e^(-ζωₙdt)·e^(±jωd·dt)

    Stability requires |z| < 1:
        |z| = e^(-ζωₙdt) < 1  ⟹  ζωₙ > 0

    This is always satisfied for ζ > 0 (damped systems).

    For undamped (ζ = 0):
        |z| = 1  (marginally stable, poles on unit circle)

    **Forced Oscillation Equilibrium:**
    For constant forcing u = u_ss:
        x_ss = u_ss/ωₙ²  (static deflection)
        v_ss = 0

    This represents a new equilibrium position where spring force balances
    the constant applied force.

    Controllability:
    ---------------
    The discrete harmonic oscillator is COMPLETELY CONTROLLABLE for all
    discretization methods and parameter values (ωₙ > 0).

    **Controllability Matrix:**
        C = [Bd, Ad·Bd]

    **Physical Meaning:**
    - Can move to any position with any velocity
    - Requires at most 2 time steps (for linear systems)
    - Energy required depends on desired state and time
    - Practical limits: actuator force, position/velocity bounds

    **Minimum Energy Control:**
    To reach target state x_target from origin in time T:
        u*(t) = B'·e^(A'(T-t))·[∫₀ᵀ e^(At)·B·B'·e^(A't) dt]⁻¹·x_target

    This gives the control that minimizes ∫ u² dt.

    Observability:
    -------------
    **Displacement-only measurement: y[k] = x[k]**
        C = [1  0]

        Observability matrix:
        O = [C     ] = [1         0        ]
            [C·Ad  ]   [Ad[0,0]   Ad[0,1]  ]

        Fully observable for all parameter values.
        Can reconstruct velocity from position measurements over time.

    **Physical Interpretation:**
    Velocity is estimated from position differences:
        v[k] ≈ (x[k+1] - x[k])/dt

    Better estimates use multiple samples (Kalman filter, least squares).

    **Full state measurement: y[k] = [x[k], v[k]]**
        Trivially observable - direct measurement of all states.

    Frequency Response:
    ------------------
    The oscillator exhibits characteristic frequency-dependent behavior.

    **Continuous-time transfer function:**
        H(s) = 1/(s² + 2ζωₙs + ωₙ²)

    **Key frequencies:**

    1. **Natural frequency ωₙ:**
       - Resonance frequency for undamped system
       - Determines overall response speed

    2. **Damped natural frequency ωd:**
       ωd = ωₙ√(1 - ζ²)  [rad/s]
       - Actual oscillation frequency for underdamped
       - Lower than ωₙ due to damping

    3. **Resonant peak frequency ωᵣ:**
       ωᵣ = ωₙ√(1 - 2ζ²)  [rad/s] (for ζ < 1/√2)
       - Frequency of maximum gain
       - Only exists for light damping

    **Magnitude response:**
    At resonance (ω = ωᵣ):
        |H(jωᵣ)| = 1/(2ζ√(1 - ζ²))

    For small damping (ζ << 1):
        |H(jωₙ)| ≈ 1/(2ζ)  (quality factor Q = 1/(2ζ))

    **Phase response:**
    - Low frequency (ω << ωₙ): φ ≈ 0° (in-phase)
    - Resonance (ω = ωₙ): φ = -90° (quadrature)
    - High frequency (ω >> ωₙ): φ → -180° (out-of-phase)

    Control Objectives:
    ------------------
    Common control goals for harmonic oscillators:

    1. **Vibration Damping:**
       Goal: Increase effective damping to reduce oscillations
       Methods:
       - Velocity feedback: u = -k_d·v (adds damping)
       - LQR with velocity penalty
       - Active damping control
       Applications: building isolation, machine tools, aerospace

    2. **Resonance Tracking:**
       Goal: Maintain oscillation at specific frequency and amplitude
       Methods:
       - Sinusoidal forcing at ωd
       - Phase-locked loop
       Applications: clocks, oscillators, sensors

    3. **Position Regulation:**
       Goal: Drive to desired position and hold
       Control: u = -K·[x - x_ref, v]
       Design: LQR, pole placement

    4. **Setpoint Tracking:**
       Goal: Follow time-varying reference position
       Control: u = -K·[x - x_ref(t), v - v_ref(t)] + u_ff(t)
       Feedforward: u_ff = ẍ_ref + 2ζωₙv_ref + ωₙ²x_ref

    5. **Disturbance Rejection:**
       Goal: Maintain position despite external forces
       Methods:
       - High-gain feedback
       - Integral action
       - Disturbance observer
       Applications: precision positioning, isolation

    State Constraints:
    -----------------
    Physical constraints that must be enforced:

    1. **Position limits: x_min ≤ x[k] ≤ x_max**
       - Physical stops, workspace boundaries
       - Spring compression/extension limits
       - Typical: |x| ≤ 0.1 m for suspensions

    2. **Velocity limits: |v[k]| ≤ v_max**
       - Material stress limits (fatigue)
       - Safety considerations
       - Typical: |v| ≤ 1 m/s for mechanical systems

    3. **Control limits: |u[k]| ≤ u_max**
       - Actuator force saturation
       - Power constraints
       - Most critical practical constraint
       - Typical: |u| ≤ 10 m/s² for active systems

    4. **Frequency limits (for tracking):**
       - Cannot track above Nyquist frequency: f < 1/(2dt)
       - Anti-aliasing required for higher frequencies
       - Practical limit: f < 0.1/dt for good control

    Numerical Considerations:
    ------------------------
    **Stability:**

    For explicit methods (Euler), stability requires:
        dt < 2/(ζωₙ + √(ζ²ωₙ² + ωₙ²))

    Approximately: dt < 1/ωₙ for ζ small

    For ZOH and Tustin, the discretization is always stable if the
    continuous system is stable (ζ > 0).

    **Accuracy:**
    - ZOH: Exact for underdamped systems with constant control
    - Tustin: O(dt²) error, preserves frequency response
    - Euler: O(dt) error, can become unstable
    - Matched: Exact pole locations, approximate zeros

    **Aliasing:**
    If excited at frequencies above Nyquist (π/dt), aliasing occurs.
    The response appears at lower frequencies:
        f_apparent = |f_actual - n·f_sample|

    Prevention: Use anti-aliasing filters before sampling.

    **Resonance Amplification:**
    At resonance with small damping:
        Gain ≈ Q = 1/(2ζ)

    For ζ = 0.01: Q = 50 (34 dB amplification!)
    This can cause:
    - Large displacements from small inputs
    - Control saturation
    - Numerical overflow if not handled

    Control Design Examples:
    -----------------------
    **1. Damping Augmentation (Velocity Feedback):**
        u[k] = -k_d·v[k]

        Effective damping: ζ_eff = ζ + k_d/(2ωₙ)

        Choose k_d to achieve desired damping:
            k_d = 2ωₙ(ζ_desired - ζ)

        Example: ωₙ = 10 rad/s, ζ = 0.1, ζ_desired = 0.7
            k_d = 2(10)(0.7 - 0.1) = 12

    **2. LQR (Optimal State Feedback):**
        Minimize: J = Σ (x'·Q·x + u'·R·u)

        Tuning guidelines:
        - Large Q[0,0]: Penalize position error (stiff response)
        - Large Q[1,1]: Penalize velocity (add damping)
        - Large R: Penalize control effort (smooth, slow)

        Example:
            Q = diag([100, 10])  # Care about position
            R = 1
            Result: Fast settling, moderate control

    **3. Pole Placement:**
        Desired continuous-time poles:
            s = -ζ_d·ωₙ_d ± jωₙ_d√(1 - ζ_d²)

        Map to discrete:
            z = e^(s·dt)

        Example: ωₙ_d = 10, ζ_d = 0.7, dt = 0.01
            s = -7 ± j7.14
            z = 0.932 ± j0.069

    **4. Notch Filter (Resonance Suppression):**
        For systems with known resonance, add notch filter:
            H_notch(z) = (z² - 2cos(ω_notch·dt)z + 1)/(z² - 2r·cos(ω_notch·dt)z + r²)

        where r < 1 controls notch width.

    Example Usage:
    -------------
    >>> # Create underdamped oscillator with natural frequency 2π rad/s (1 Hz)
    >>> system = DiscreteOscillator(omega_n=2*np.pi, zeta=0.1, dt=0.01)
    >>> 
    >>> # Initial condition: displaced 1m, at rest
    >>> x0 = np.array([1.0, 0.0])  # [position, velocity]
    >>> 
    >>> # Free oscillation (no control)
    >>> result_free = system.simulate(
    ...     x0=x0,
    ...     u_sequence=None,
    ...     n_steps=500
    ... )
    >>> 
    >>> # Plot free oscillation
    >>> import matplotlib.pyplot as plt
    >>> t = result_free['time_steps'] * system.dt
    >>> fig, axes = plt.subplots(3, 1, figsize=(10, 8))
    >>> 
    >>> # Position
    >>> axes[0].plot(t, result_free['states'][:, 0])
    >>> axes[0].set_ylabel('Position [m]')
    >>> axes[0].grid()
    >>> axes[0].set_title('Free Oscillation (Underdamped)')
    >>> 
    >>> # Velocity
    >>> axes[1].plot(t, result_free['states'][:, 1])
    >>> axes[1].set_ylabel('Velocity [m/s]')
    >>> axes[1].grid()
    >>> 
    >>> # Phase portrait
    >>> axes[2].plot(result_free['states'][:, 0], result_free['states'][:, 1])
    >>> axes[2].set_xlabel('Position [m]')
    >>> axes[2].set_ylabel('Velocity [m/s]')
    >>> axes[2].grid()
    >>> axes[2].set_title('Phase Portrait (Spiral to Origin)')
    >>> 
    >>> plt.tight_layout()
    >>> plt.show()
    >>> 
    >>> # Compute theoretical decay envelope
    >>> A0 = 1.0  # Initial amplitude
    >>> envelope = A0 * np.exp(-system.zeta * system.omega_n * t)
    >>> 
    >>> # Verify decay rate
    >>> peaks = result_free['states'][::int(np.pi/(system.omega_d * system.dt)), 0]
    >>> print(f"Theoretical decay: {envelope[-1]:.4f}")
    >>> print(f"Simulated decay: {np.abs(peaks[-1]):.4f}")
    >>> 
    >>> # Design damping augmentation controller
    >>> zeta_desired = 0.7  # Critical damping
    >>> k_d = system.design_damping_controller(zeta_desired)
    >>> print(f"Damping gain: k_d = {k_d:.2f}")
    >>> 
    >>> def damping_controller(x, k):
    ...     return -k_d * x[1]  # Velocity feedback only
    >>> 
    >>> result_damped = system.rollout(x0, damping_controller, n_steps=500)
    >>> 
    >>> # Compare settling times
    >>> settling_threshold = 0.02 * A0  # 2% criterion
    >>> free_settling = np.where(np.abs(result_free['states'][:, 0]) < settling_threshold)[0]
    >>> damped_settling = np.where(np.abs(result_damped['states'][:, 0]) < settling_threshold)[0]
    >>> 
    >>> if len(free_settling) > 0 and len(damped_settling) > 0:
    ...     print(f"Free settling time: {free_settling[0] * system.dt:.2f} s")
    ...     print(f"Damped settling time: {damped_settling[0] * system.dt:.2f} s")
    >>> 
    >>> # Resonance testing - sweep frequency
    >>> frequencies = np.logspace(-1, 1, 50)  # 0.1 to 10 rad/s
    >>> gains = []
    >>> phases = []
    >>> 
    >>> for omega in frequencies:
    ...     # Apply sinusoidal input
    ...     t_sim = np.arange(0, 20, system.dt)
    ...     u_sin = np.sin(omega * t_sim)
    ...     
    ...     result = system.simulate(
    ...         x0=np.zeros(2),
    ...         u_sequence=u_sin,
    ...         n_steps=len(t_sim)
    ...     )
    ...     
    ...     # Measure steady-state amplitude and phase
    ...     x_ss = result['states'][-100:, 0]
    ...     u_ss = u_sin[-100:]
    ...     
    ...     # Compute gain (amplitude ratio)
    ...     gain = np.max(np.abs(x_ss)) / np.max(np.abs(u_ss))
    ...     gains.append(gain)
    ...     
    ...     # Compute phase (via FFT)
    ...     X_fft = np.fft.fft(x_ss)
    ...     U_fft = np.fft.fft(u_ss)
    ...     phase = np.angle(X_fft[1]) - np.angle(U_fft[1])
    ...     phases.append(np.rad2deg(phase))
    >>> 
    >>> # Plot Bode diagram
    >>> fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    >>> 
    >>> # Magnitude
    >>> axes[0].semilogx(frequencies, 20*np.log10(gains), 'b-', label='Simulated')
    >>> axes[0].axvline(system.omega_n, color='r', linestyle='--', label=f'ωₙ = {system.omega_n:.2f}')
    >>> axes[0].set_ylabel('Magnitude [dB]')
    >>> axes[0].grid(which='both')
    >>> axes[0].legend()
    >>> axes[0].set_title('Frequency Response (Bode Plot)')
    >>> 
    >>> # Phase
    >>> axes[1].semilogx(frequencies, phases, 'b-')
    >>> axes[1].axvline(system.omega_n, color='r', linestyle='--')
    >>> axes[1].axhline(-90, color='g', linestyle=':', label='-90° at resonance')
    >>> axes[1].set_xlabel('Frequency [rad/s]')
    >>> axes[1].set_ylabel('Phase [deg]')
    >>> axes[1].grid(which='both')
    >>> axes[1].legend()
    >>> 
    >>> plt.tight_layout()
    >>> plt.show()
    >>> 
    >>> # Design LQR controller
    >>> Ad, Bd = system.linearize(np.zeros(2), np.zeros(1))
    >>> Q = np.diag([100.0, 1.0])  # Care more about position
    >>> R = np.array([[1.0]])
    >>> lqr_result = system.control.design_lqr(
    ...     Ad, Bd, Q, R, system_type='discrete'
    ... )
    >>> K = lqr_result['gain']
    >>> print(f"LQR gain: K = {K}")
    >>> 
    >>> # Check closed-loop damping
    >>> zeta_cl, omega_cl = system.compute_closed_loop_damping(K)
    >>> print(f"Closed-loop damping: ζ = {zeta_cl:.3f}")
    >>> print(f"Closed-loop frequency: ω = {omega_cl:.2f} rad/s")
    >>> 
    >>> # Simulate with LQR
    >>> def lqr_controller(x, k):
    ...     return -K @ x
    >>> 
    >>> result_lqr = system.rollout(x0, lqr_controller, n_steps=500)
    >>> 
    >>> # Tracking example: follow sinusoidal reference
    >>> omega_ref = 0.5  # Below resonance
    >>> A_ref = 0.5      # Amplitude
    >>> t_track = np.arange(0, 10, system.dt)
    >>> x_ref = A_ref * np.sin(omega_ref * t_track)
    >>> v_ref = A_ref * omega_ref * np.cos(omega_ref * t_track)
    >>> a_ref = -A_ref * omega_ref**2 * np.sin(omega_ref * t_track)
    >>> 
    >>> def tracking_controller(x, k):
    ...     if k >= len(x_ref):
    ...         return np.array([0.0])
    ...     
    ...     # Feedforward + feedback
    ...     u_ff = a_ref[k] + 2*system.zeta*system.omega_n*v_ref[k] + system.omega_n**2*x_ref[k]
    ...     u_fb = -K @ (x - np.array([x_ref[k], v_ref[k]]))
    ...     return u_fb + u_ff
    >>> 
    >>> result_track = system.rollout(np.zeros(2), tracking_controller, n_steps=len(t_track))
    >>> 
    >>> # Plot tracking performance
    >>> plt.figure(figsize=(10, 6))
    >>> plt.plot(t_track, x_ref, 'r--', label='Reference', linewidth=2)
    >>> plt.plot(t_track, result_track['states'][:, 0], 'b-', label='Actual')
    >>> plt.xlabel('Time [s]')
    >>> plt.ylabel('Position [m]')
    >>> plt.legend()
    >>> plt.grid()
    >>> plt.title('Tracking Performance')
    >>> plt.show()
    >>> 
    >>> # Compute tracking error
    >>> error = result_track['states'][:, 0] - x_ref
    >>> rms_error = np.sqrt(np.mean(error**2))
    >>> max_error = np.max(np.abs(error))
    >>> print(f"RMS tracking error: {rms_error:.4f} m")
    >>> print(f"Max tracking error: {max_error:.4f} m")

    Physical Insights:
    -----------------
    **Energy Considerations:**
    The harmonic oscillator continuously exchanges energy between kinetic
    and potential forms:

        E_kinetic = 0.5·m·v²
        E_potential = 0.5·k·x²
        E_total = E_kinetic + E_potential

    For undamped (ζ = 0): Total energy is conserved
    For damped (ζ > 0): Energy decreases exponentially
        E(t) = E₀·e^(-2ζωₙt)

    Power dissipated by damping:
        P_damped = c·v² = 2mζωₙ·v²

    **Quality Factor:**
    The quality factor Q measures how underdamped the system is:
        Q = 1/(2ζ)

    High Q (low damping):
    - Sharp resonance peak
    - Long ringing time
    - Narrow bandwidth
    - Examples: tuning forks (Q ~ 1000), quartz crystals (Q ~ 10⁶)

    Low Q (high damping):
    - Broad response
    - Fast settling
    - Wide bandwidth
    - Examples: shock absorbers (Q ~ 1), damped doors (Q ~ 0.5)

    **Decay Rate vs Oscillation Frequency:**
    For underdamped systems, there's a tradeoff:
    - Decay rate: σ = ζωₙ (controls settling time)
    - Oscillation frequency: ωd = ωₙ√(1 - ζ²)

    As damping increases:
    - Decay rate increases (faster settling)
    - Oscillation frequency decreases
    - At critical damping: ωd = 0 (no oscillation)

    **Resonance Phenomenon:**
    At resonance, even small periodic forces can produce large displacements.
    Examples:
    - Tacoma Narrows Bridge collapse (1940)
    - Wine glass shattering from sound
    - Building damage in earthquakes
    - Mechanical vibration failures

    Prevention strategies:
    - Avoid excitation near ωₙ
    - Increase damping (ζ > 0.1 typically safe)
    - Use vibration isolators
    - Active control

    **Time Scales:**
    The system has characteristic time scales:

    1. Natural period: T = 2π/ωₙ
       - Time for one complete oscillation (undamped)

    2. Damped period: Td = 2π/ωd
       - Actual oscillation period (underdamped)
       - Td > T (damping slows oscillation)

    3. Time constant: τ = 1/(ζωₙ)
       - Time for amplitude to decay by factor e
       - Envelope decay: e^(-t/τ)

    4. Settling time (2% criterion): ts ≈ 4τ = 4/(ζωₙ)
       - Time to reach 2% of steady-state
       - Often used as design specification

    **Relationship to Electrical Circuits:**
    The RLC circuit is mathematically identical:

        L·d²q/dt² + R·dq/dt + (1/C)·q = V(t)

    Analogies:
        Mechanical    ↔  Electrical
        Position x    ↔  Charge q
        Velocity v    ↔  Current i
        Mass m        ↔  Inductance L
        Damping c     ↔  Resistance R
        Spring k      ↔  1/Capacitance

    This enables unified analysis and design across domains.

    Common Pitfalls:
    ---------------
    1. **Aliasing in high-frequency oscillators:**
       If ωₙ > π/dt (Nyquist), the discrete system appears slower.
       Fix: Increase sampling rate or add anti-aliasing filter.

    2. **Resonance amplification:**
       Small disturbances at ωₙ cause large response for small ζ.
       Fix: Avoid excitation near ωₙ or increase damping.

    3. **Incorrect damping ratio:**
       Using damping coefficient c instead of ratio ζ.
       Remember: ζ = c/(2√(km)) is dimensionless.

    4. **Forgetting frequency shift:**
       Damped frequency ωd ≠ ωₙ for ζ > 0.
       Use: ωd = ωₙ√(1 - ζ²)

    5. **Numerical instability with Euler:**
       For stiff systems (large ωₙ or ζ), Euler requires tiny dt.
       Fix: Use ZOH, Tustin, or implicit methods.

    6. **Control saturation near resonance:**
       At resonance, control effort can be excessive.
       Fix: Use anti-windup, gain scheduling, or notch filters.

    Extensions:
    ----------
    This basic oscillator can be extended to:

    1. **Nonlinear oscillator:**
       Add nonlinear spring: k → k₁x + k₃x³ (Duffing oscillator)
       Creates amplitude-dependent frequency

    2. **Coupled oscillators:**
       Multiple masses connected by springs
       Normal modes and mode shapes

    3. **Parametric oscillator:**
       Time-varying parameters: ωₙ(t)
       Can cause parametric resonance

    4. **Forced oscillator with multiple frequencies:**
       Beat phenomena, combination tones

    5. **With Coulomb friction:**
       Add dry friction: F_friction = μ·N·sign(v)
       Creates limit cycles

    6. **Nonlinear damping:**
       Quadratic drag: F_drag = c·v²
       Common in aerodynamics

    See Also:
    --------
    DiscreteDoubleIntegrator : Undamped limit (ζ = 0, ωₙ = 0)
    """

    def define_system(
        self,
        omega_n: float = 1.0,
        zeta: float = 0.1,
        dt: float = 0.1,
        method: str = 'zoh',
        mass: float = 1.0,
        spring_constant: Optional[float] = None,
        damping_coefficient: Optional[float] = None,
    ):
        """
        Define discrete-time harmonic oscillator dynamics.

        Parameters
        ----------
        omega_n : float
            Natural frequency [rad/s], must be positive
        zeta : float
            Damping ratio [-], must be non-negative
        dt : float
            Sampling time step [s]
        method : str
            Discretization method:
            - 'zoh': Zero-order hold (exact, recommended)
            - 'tustin': Bilinear/trapezoidal transform
            - 'euler': Forward Euler (simple)
            - 'backward_euler': Backward Euler (implicit)
            - 'matched': Matched pole-zero method
        mass : float
            System mass [kg] (only used with spring_constant/damping_coefficient)
        spring_constant : Optional[float]
            Spring constant k [N/m] (alternative to omega_n)
            If provided: omega_n = sqrt(k/m)
        damping_coefficient : Optional[float]
            Damping coefficient c [N·s/m] (alternative to zeta)
            If provided: zeta = c/(2*sqrt(k*m))
        """
        # Store configuration
        self._discretization_method = method
        self._mass = mass

        # Compute omega_n and zeta from physical parameters if provided
        if spring_constant is not None:
            omega_n = np.sqrt(spring_constant / mass)
            self._spring_constant = spring_constant
        else:
            self._spring_constant = mass * omega_n**2

        if damping_coefficient is not None:
            zeta = damping_coefficient / (2 * np.sqrt(self._spring_constant * mass))
            self._damping_coefficient = damping_coefficient
        else:
            self._damping_coefficient = 2 * zeta * np.sqrt(self._spring_constant * mass)

        # Store parameters
        self._omega_n_val = omega_n
        self._zeta_val = zeta

        # Validate parameters
        if omega_n <= 0:
            raise ValueError(f"Natural frequency must be positive, got omega_n = {omega_n}")
        if zeta < 0:
            raise ValueError(f"Damping ratio must be non-negative, got zeta = {zeta}")

        # Compute damped natural frequency (for underdamped case)
        if zeta < 1:
            self._omega_d_val = omega_n * np.sqrt(1 - zeta**2)
        else:
            self._omega_d_val = 0.0  # No oscillation for critically/overdamped

        # State variables
        x, v = sp.symbols('x v', real=True)
        u = sp.symbols('u', real=True)

        # Symbolic parameters
        omega_n_sym = sp.symbols('omega_n', positive=True, real=True)
        zeta_sym = sp.symbols('zeta', nonnegative=True, real=True)

        self.state_vars = [x, v]
        self.control_vars = [u]
        self.output_vars = []  # Position measurement only
        self._dt = dt
        self.order = 1

        self.parameters = {
            omega_n_sym: omega_n,
            zeta_sym: zeta,
        }

        # Define dynamics based on discretization method
        if method == 'zoh':
            # Zero-order hold (exact discretization)
            if zeta < 1:
                # Underdamped case
                omega_d = omega_n * sp.sqrt(1 - zeta_sym**2)
                exp_decay = sp.exp(-zeta_sym * omega_n_sym * dt)

                x_next = exp_decay * (
                    x * sp.cos(omega_d * dt) + v * sp.sin(omega_d * dt) / omega_d
                )
                v_next = exp_decay * (
                    -x * omega_d * sp.sin(omega_d * dt) + v * sp.cos(omega_d * dt)
                ) + dt * u

            elif abs(zeta - 1.0) < 1e-10:
                # Critically damped case
                exp_decay = sp.exp(-omega_n_sym * dt)
                x_next = exp_decay * (x + v * dt)
                v_next = exp_decay * (v - omega_n_sym * x * dt) + dt * u

            else:
                # Overdamped case
                lambda1 = -zeta_sym * omega_n_sym + omega_n_sym * sp.sqrt(zeta_sym**2 - 1)
                lambda2 = -zeta_sym * omega_n_sym - omega_n_sym * sp.sqrt(zeta_sym**2 - 1)

                exp1 = sp.exp(lambda1 * dt)
                exp2 = sp.exp(lambda2 * dt)

                A11 = (lambda1 * exp1 - lambda2 * exp2) / (lambda1 - lambda2)
                A12 = (exp1 - exp2) / (lambda1 - lambda2)
                A21 = lambda1 * lambda2 * (exp2 - exp1) / (lambda1 - lambda2)
                A22 = (lambda1 * exp2 - lambda2 * exp1) / (lambda1 - lambda2)

                x_next = A11 * x + A12 * v
                v_next = A21 * x + A22 * v + dt * u

        elif method == 'tustin':
            
            # TODO: verify this is mathematically valid; seems suspect
            # Bilinear/trapezoidal transform
            # (I - 0.5*dt*A)*x[k+1] = (I + 0.5*dt*A)*x[k] + 0.5*dt*B*(u[k] + u[k+1])
            # For simplicity, use u[k+1] ≈ u[k] (zero-order hold on control)
            
            # (1 + alpha)*v[k+1] + beta*x[k+1] = (1 - alpha)*v[k] - beta*x[k] + dt*u[k]
            # v[k+1] - dt*v[k+1] = v[k] + dt*v[k]

            # Simplified version: use forward differences for derivative approximation
            x_next = x + 0.5 * dt * (v + (v - 2 * zeta_sym * omega_n_sym * v - omega_n_sym**2 * x + u))
            v_next = v + 0.5 * dt * (
                (-2 * zeta_sym * omega_n_sym * v - omega_n_sym**2 * x + u)
                + (-2 * zeta_sym * omega_n_sym * (v + dt * (-2 * zeta_sym * omega_n_sym * v - omega_n_sym**2 * x + u))
                   - omega_n_sym**2 * (x + dt * v) + u)
            )

        elif method == 'euler':
            # Forward Euler (simple first-order approximation)
            x_next = x + dt * v
            v_next = v + dt * (-omega_n_sym**2 * x - 2 * zeta_sym * omega_n_sym * v + u)

        elif method == 'backward_euler':
            # Backward Euler (implicit method)
            # v[k+1] = v[k] + dt*(-ωₙ²*x[k+1] - 2ζωₙ*v[k+1] + u[k])
            # Solve for v[k+1] and x[k+1]

            denom = 1 + dt * 2 * zeta_sym * omega_n_sym + dt**2 * omega_n_sym**2
            v_next = (v + dt * (-omega_n_sym**2 * x + u)) / denom
            x_next = (x + dt * v + dt**2 * u) / denom

        elif method == 'matched':
            # Matched pole-zero method
            # Convert continuous poles to discrete, preserve zeros
            if zeta < 1:
                # Underdamped
                sigma = -zeta_sym * omega_n_sym
                omega_d = omega_n_sym * sp.sqrt(1 - zeta_sym**2)

                # Discrete poles: z = e^((sigma ± jωd)*dt)
                exp_sigma_dt = sp.exp(sigma * dt)
                cos_wd_dt = sp.cos(omega_d * dt)
                sin_wd_dt = sp.sin(omega_d * dt)

                x_next = exp_sigma_dt * (x * cos_wd_dt + v * sin_wd_dt / omega_d)
                v_next = exp_sigma_dt * (-x * omega_d * sin_wd_dt + v * cos_wd_dt) + dt * u

            else:
                # Use critically damped formula as approximation
                exp_decay = sp.exp(-omega_n_sym * dt)
                x_next = exp_decay * (x + v * dt)
                v_next = exp_decay * (v - omega_n_sym * x * dt) + dt * u

        else:
            raise ValueError(
                f"Unknown discretization method '{method}'. "
                f"Choose from: 'zoh', 'tustin', 'euler', 'backward_euler', 'matched'"
            )

        self._f_sym = sp.Matrix([x_next, v_next])
        self._h_sym = sp.Matrix([x])  # Measure position only

    def setup_equilibria(self):
        """
        Set up equilibrium points.

        Adds the origin (rest position) as the stable equilibrium for damped systems.
        """
        stability_type = "asymptotically_stable" if self._zeta_val > 0 else "marginally_stable"

        notes = (
            f"Natural frequency: ωₙ = {self._omega_n_val:.3f} rad/s\n"
            f"Damping ratio: ζ = {self._zeta_val:.3f}\n"
        )

        if self._zeta_val < 1:
            notes += f"Damped frequency: ωd = {self._omega_d_val:.3f} rad/s\n"
            notes += "Underdamped: oscillates with exponential decay"
        elif abs(self._zeta_val - 1.0) < 1e-6:
            notes += "Critically damped: fastest return without overshoot"
        elif self._zeta_val > 1:
            notes += "Overdamped: slow return without oscillation"
        else:
            notes += "Undamped: pure sinusoidal oscillation"

        self.add_equilibrium(
            "rest",
            x_eq=np.array([0.0, 0.0]),
            u_eq=np.array([0.0]),
            verify=True,
            stability=stability_type,
            notes=notes,
        )

        self.set_default_equilibrium("rest")

    @property
    def omega_n(self) -> float:
        """Natural frequency [rad/s]."""
        return self._omega_n_val

    @property
    def zeta(self) -> float:
        """Damping ratio [-]."""
        return self._zeta_val

    @property
    def omega_d(self) -> float:
        """Damped natural frequency [rad/s]."""
        return self._omega_d_val

    @property
    def spring_constant(self) -> float:
        """Spring constant k = m·ωₙ² [N/m]."""
        return self._spring_constant

    @property
    def damping_coefficient(self) -> float:
        """Damping coefficient c = 2ζ√(km) [N·s/m]."""
        return self._damping_coefficient

    @property
    def quality_factor(self) -> float:
        """Quality factor Q = 1/(2ζ) [-]."""
        if self._zeta_val > 0:
            return 1.0 / (2.0 * self._zeta_val)
        return np.inf

    @property
    def natural_period(self) -> float:
        """Natural period T = 2π/ωₙ [s]."""
        return 2.0 * np.pi / self._omega_n_val

    @property
    def damped_period(self) -> float:
        """Damped period Td = 2π/ωd [s] (for underdamped only)."""
        if self._omega_d_val > 0:
            return 2.0 * np.pi / self._omega_d_val
        return np.inf

    @property
    def time_constant(self) -> float:
        """Time constant τ = 1/(ζωₙ) [s]."""
        if self._zeta_val > 0:
            return 1.0 / (self._zeta_val * self._omega_n_val)
        return np.inf

    @property
    def settling_time(self) -> float:
        """Settling time (2% criterion) ts ≈ 4τ [s]."""
        if self._zeta_val > 0:
            return 4.0 * self.time_constant
        return np.inf

    def compute_system_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute discrete-time state-space matrices Ad, Bd.

        Returns
        -------
        tuple
            (Ad, Bd) where:
            - Ad: State transition matrix (2×2)
            - Bd: Control input matrix (2×1)

        Examples
        --------
        >>> system = DiscreteOscillator(omega_n=2*np.pi, zeta=0.1, dt=0.01)
        >>> Ad, Bd = system.compute_system_matrices()
        >>> print(f"Ad =\\n{Ad}")
        >>> print(f"Bd =\\n{Bd}")
        """
        Ad, Bd = self.linearize(np.zeros(2), np.zeros(1))
        return Ad, Bd

    def compute_eigenvalues(self) -> np.ndarray:
        """
        Compute discrete-time eigenvalues.

        Returns
        -------
        np.ndarray
            Complex eigenvalues (2,)

        Examples
        --------
        >>> system = DiscreteOscillator(omega_n=10, zeta=0.1, dt=0.01)
        >>> eigs = system.compute_eigenvalues()
        >>> print(f"Eigenvalues: {eigs}")
        >>> print(f"Magnitude: {np.abs(eigs)}")
        >>> print(f"Stable: {np.all(np.abs(eigs) < 1)}")

        Notes
        -----
        For continuous-time poles s = -ζωₙ ± jωₙ√(1-ζ²),
        discrete poles are z = e^(s·dt).

        Magnitude |z| = e^(-ζωₙdt) determines stability:
        - |z| < 1: Stable (decays)
        - |z| = 1: Marginally stable (ζ = 0)
        - |z| > 1: Unstable (grows)
        """
        Ad, _ = self.compute_system_matrices()
        return np.linalg.eigvals(Ad)

    def compute_frequency_response(
        self,
        frequencies: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute frequency response (Bode plot data).

        Parameters
        ----------
        frequencies : np.ndarray
            Frequencies to evaluate [rad/s]

        Returns
        -------
        tuple
            (magnitude_dB, phase_deg) where:
            - magnitude_dB: Magnitude in decibels
            - phase_deg: Phase in degrees

        Examples
        --------
        >>> system = DiscreteOscillator(omega_n=10, zeta=0.1, dt=0.01)
        >>> freqs = np.logspace(-1, 2, 100)
        >>> mag_dB, phase_deg = system.compute_frequency_response(freqs)
        >>>
        >>> import matplotlib.pyplot as plt
        >>> fig, axes = plt.subplots(2, 1)
        >>> axes[0].semilogx(freqs, mag_dB)
        >>> axes[0].set_ylabel('Magnitude [dB]')
        >>> axes[1].semilogx(freqs, phase_deg)
        >>> axes[1].set_ylabel('Phase [deg]')
        >>> axes[1].set_xlabel('Frequency [rad/s]')
        >>> plt.show()
        """
        Ad, Bd = self.compute_system_matrices()
        C = np.array([[1.0, 0.0]])  # Position output
        D = np.array([[0.0]])

        # Create discrete-time system
        sys_discrete = signal.StateSpace(Ad, Bd, C, D, dt=self._dt)

        # Compute frequency response
        w, H = signal.dfreqresp(sys_discrete, w=frequencies)

        magnitude_dB = 20 * np.log10(np.abs(H).flatten())
        phase_deg = np.angle(H, deg=True).flatten()

        return magnitude_dB, phase_deg

    def compute_resonant_frequency(self) -> float:
        """
        Compute resonant peak frequency ωᵣ [rad/s].

        Returns
        -------
        float
            Resonant frequency (0 if ζ ≥ 1/√2)

        Notes
        -----
        For ζ < 1/√2:
            ωᵣ = ωₙ√(1 - 2ζ²)

        The resonant frequency is where the magnitude response peaks.
        It's lower than the natural frequency and only exists for
        lightly damped systems.

        Examples
        --------
        >>> system = DiscreteOscillator(omega_n=10, zeta=0.1, dt=0.01)
        >>> omega_r = system.compute_resonant_frequency()
        >>> print(f"Resonant frequency: {omega_r:.2f} rad/s")
        >>> print(f"Natural frequency: {system.omega_n:.2f} rad/s")
        """
        if self._zeta_val < 1.0 / np.sqrt(2.0):
            return self._omega_n_val * np.sqrt(1.0 - 2.0 * self._zeta_val**2)
        return 0.0

    def compute_resonant_peak(self) -> float:
        """
        Compute resonant peak magnitude (gain at resonance).

        Returns
        -------
        float
            Peak magnitude (infinity if ζ = 0)

        Notes
        -----
        At resonance (ω = ωᵣ):
            |H(jωᵣ)| = 1/(2ζ√(1 - ζ²))

        For small damping (ζ << 1):
            |H(jωₙ)| ≈ Q = 1/(2ζ)

        Examples
        --------
        >>> system = DiscreteOscillator(omega_n=10, zeta=0.05, dt=0.01)
        >>> peak = system.compute_resonant_peak()
        >>> print(f"Resonant peak: {peak:.1f} ({20*np.log10(peak):.1f} dB)")
        >>> print(f"Quality factor: {system.quality_factor:.1f}")
        """
        if self._zeta_val > 0 and self._zeta_val < 1:
            return 1.0 / (2.0 * self._zeta_val * np.sqrt(1.0 - self._zeta_val**2))
        elif self._zeta_val == 0:
            return np.inf
        else:
            return 1.0

    def design_damping_controller(
        self,
        zeta_desired: float,
    ) -> float:
        """
        Design velocity feedback gain to achieve desired damping ratio.

        Control law: u[k] = -k_d·v[k]

        Parameters
        ----------
        zeta_desired : float
            Desired closed-loop damping ratio

        Returns
        -------
        float
            Velocity feedback gain k_d

        Examples
        --------
        >>> system = DiscreteOscillator(omega_n=10, zeta=0.1, dt=0.01)
        >>> k_d = system.design_damping_controller(zeta_desired=0.7)
        >>> print(f"Damping gain: k_d = {k_d:.2f}")

        Notes
        -----
        This adds damping without changing natural frequency (approximately).
        The effective damping becomes:
            ζ_eff ≈ ζ + k_d/(2ωₙ)

        Solving for k_d:
            k_d = 2ωₙ(ζ_desired - ζ)
        """
        k_d = 2.0 * self._omega_n_val * (zeta_desired - self._zeta_val)
        return max(0.0, k_d)  # Ensure non-negative

    def design_stiffness_controller(
        self,
        omega_n_desired: float,
    ) -> float:
        """
        Design position feedback gain to achieve desired natural frequency.

        Control law: u[k] = -k_p·x[k]

        Parameters
        ----------
        omega_n_desired : float
            Desired closed-loop natural frequency [rad/s]

        Returns
        -------
        float
            Position feedback gain k_p

        Examples
        --------
        >>> system = DiscreteOscillator(omega_n=5, zeta=0.1, dt=0.01)
        >>> k_p = system.design_stiffness_controller(omega_n_desired=10)
        >>> print(f"Stiffness gain: k_p = {k_p:.2f}")

        Notes
        -----
        This changes the effective natural frequency:
            ωₙ_eff = √(ωₙ² + k_p)

        Solving for k_p:
            k_p = ωₙ_desired² - ωₙ²
        """
        k_p = omega_n_desired**2 - self._omega_n_val**2
        return max(0.0, k_p)  # Ensure non-negative

    def compute_closed_loop_damping(
        self,
        K: np.ndarray,
    ) -> Tuple[float, float]:
        """
        Compute closed-loop damping ratio and natural frequency.

        Parameters
        ----------
        K : np.ndarray
            State feedback gain [k_p, k_d]

        Returns
        -------
        tuple
            (zeta_cl, omega_cl) closed-loop damping and frequency

        Examples
        --------
        >>> system = DiscreteOscillator(omega_n=10, zeta=0.1, dt=0.01)
        >>> K = np.array([[50, 10]])
        >>> zeta_cl, omega_cl = system.compute_closed_loop_damping(K)
        >>> print(f"Closed-loop: ζ = {zeta_cl:.3f}, ω = {omega_cl:.2f}")
        """
        k_p, k_d = K.flatten()

        # Closed-loop natural frequency
        omega_cl = np.sqrt(self._omega_n_val**2 + k_p)

        # Closed-loop damping ratio
        zeta_cl = (2.0 * self._zeta_val * self._omega_n_val + k_d) / (2.0 * omega_cl)

        return zeta_cl, omega_cl

    def compute_step_response_characteristics(
        self,
    ) -> dict:
        """
        Compute step response characteristics.

        Returns
        -------
        dict
            Dictionary containing:
            - 'rise_time': 10%-90% rise time [s]
            - 'peak_time': Time to first peak [s]
            - 'overshoot': Peak overshoot [%]
            - 'settling_time': 2% settling time [s]

        Examples
        --------
        >>> system = DiscreteOscillator(omega_n=10, zeta=0.3, dt=0.01)
        >>> chars = system.compute_step_response_characteristics()
        >>> for key, val in chars.items():
        ...     print(f"{key}: {val:.3f}")

        Notes
        -----
        These formulas are for underdamped systems (0 < ζ < 1).
        For critically/overdamped systems, some metrics don't apply.
        """
        characteristics = {}

        if 0 < self._zeta_val < 1:
            # Rise time (10% to 90%)
            beta = np.arctan(self._omega_d_val / (self._zeta_val * self._omega_n_val))
            t_r = (np.pi - beta) / self._omega_d_val
            characteristics['rise_time'] = t_r

            # Peak time
            t_p = np.pi / self._omega_d_val
            characteristics['peak_time'] = t_p

            # Percent overshoot
            overshoot = 100 * np.exp(-self._zeta_val
                                     # Settling time (2% criterion)
                                     np.pi / np.sqrt(1 - self._zeta_val**2))
            characteristics['overshoot'] = overshoot
            t_s = 4.0 / (self._zeta_val * self._omega_n_val)
            characteristics['settling_time'] = t_s

        elif abs(self._zeta_val - 1.0) < 1e-6:
            # Critically damped
            characteristics['rise_time'] = 2.2 / self._omega_n_val
            characteristics['peak_time'] = np.inf  # No overshoot
            characteristics['overshoot'] = 0.0
            characteristics['settling_time'] = 4.0 / self._omega_n_val

        else:
            # Overdamped or undamped
            characteristics['rise_time'] = np.inf
            characteristics['peak_time'] = np.inf
            characteristics['overshoot'] = 0.0
            characteristics['settling_time'] = np.inf

        return characteristics
    
    def generate_chirp_signal(
        self,
        f_start: float,
        f_end: float,
        duration: float,
        amplitude: float = 1.0,
        ) -> np.ndarray:
        """
        Generate chirp (frequency sweep) signal for system identification.
        Parameters
        ----------
        f_start : float
            Starting frequency [Hz]
        f_end : float
            Ending frequency [Hz]
        duration : float
            Signal duration [s]
        amplitude : float
            Signal amplitude

        Returns
        -------
        np.ndarray
            Chirp signal

        Examples
        --------
        >>> system = DiscreteOscillator(omega_n=10, zeta=0.1, dt=0.01)
        >>> chirp = system.generate_chirp_signal(
        ...     f_start=0.1,
        ...     f_end=5.0,
        ...     duration=10.0
        ... )
        >>> result = system.simulate(
        ...     x0=np.zeros(2),
        ...     u_sequence=chirp,
        ...     n_steps=len(chirp)
        ... )
        """
        t = np.arange(0, duration, self._dt)
        f_instantaneous = f_start + (f_end - f_start) * t / duration
        phase = 2 * np.pi * np.cumsum(f_instantaneous) * self._dt
        return amplitude * np.sin(phase)
    
    # def print_equations(self, simplify: bool = True):
    #     """
    #     Print symbolic equations using discrete-time notation.
    #     Parameters
    #     ----------
    #     simplify : bool
    #         If True, simplify expressions before printing
    #     """
    #     print("=" * 70)
    #     print(f"{self.__class__.__name__} (Discrete-Time, dt={self.dt} s)")
    #     print("=" * 70)
    #     print(f"Discretization Method: {self._discretization_method.upper()}")

    #     print("\nPhysical Parameters:")
    #     print(f"  Mass: m = {self._mass} kg")
    #     print(f"  Spring constant: k = {self._spring_constant:.2f} N/m")
    #     print(f"  Damping coefficient: c = {self._damping_coefficient:.3f} N·s/m")

    #     print("\nSystem Characteristics:")
    #     print(f"  Natural frequency: ωₙ = {self._omega_n_val:.3f} rad/s ({self._omega_n_val/(2*np.pi):.3f} Hz)")
    #     print(f"  Damping ratio: ζ = {self._zeta_val:.3f}")

    #     if self._zeta_val < 1:
    #         print(f"  Damped frequency: ωd = {self._omega_d_val:.3f} rad/s ({self._omega_d_val/(2*np.pi):.3f} Hz)")
    #         print(f"  Regime: UNDERDAMPED (oscillatory decay)")
    #     elif abs(self._zeta_val - 1.0) < 1e-6:
    #         print(f"  Regime: CRITICALLY DAMPED (fastest non-oscillatory)")
    #     else:
    #         print(f"  Regime: OVERDAMPED (slow non-oscillatory)")

    #     print(f"  Quality factor: Q = {self.quality_factor:.2f}")
    #     print(f"  Natural period: T = {self.natural_period:.3f} s")

    #     if self._zeta_val > 0:
    #         print(f"  Time constant: τ = {self.time_constant:.3f} s")
    #         print(f"  Settling time: ts = {self.settling_time:.3f} s")

    #     print(f"\nState: x = [x, v] (position, velocity)")
    #     print(f"Control: u = [F] (force/acceleration)")
    #     print(f"Dimensions: nx={self.nx}, nu={self.nu}")
    #     print(f"Sampling Period: {self.dt} s")

    #     print("\nDynamics: x[k+1] = f(x[k], u[k])")
    #     state_labels = ['x[k+1]', 'v[k+1]']
    #     for label, expr in zip(state_labels, self._f_sym):
    #         expr_sub = self.substitute_parameters(expr)
    #         if simplify:
    #             expr_sub = sp.simplify(expr_sub)
    #         print(f"  {label} = {expr_sub}")

    #     print("\nOutput: y[k] = h(x[k])")
    #     print(f"  y[k] = {self._h_sym[0]}")

    #     # Eigenvalues
    #     eigenvalues = self.compute_eigenvalues()
    #     print(f"\nDiscrete-Time Eigenvalues: {eigenvalues}")
    #     print(f"Magnitude: {np.abs(eigenvalues)}")
    #     print(f"Stability: {'Stable' if np.all(np.abs(eigenvalues) < 1) else 'Unstable'}")

    #     # Step response characteristics
    #     if 0 < self._zeta_val < 1:
    #         chars = self.compute_step_response_characteristics()
    #         print("\nStep Response Characteristics:")
    #         print(f"  Rise time (10%-90%): {chars['rise_time']:.3f} s")
    #         print(f"  Peak time: {chars['peak_time']:.3f} s")
    #         print(f"  Overshoot: {chars['overshoot']:.1f}%")
    #         print(f"  Settling time (2%): {chars['settling_time']:.3f} s")

    #     # Frequency response
    #     omega_r = self.compute_resonant_frequency()
    #     if omega_r > 0:
    #         peak = self.compute_resonant_peak()
    #         print("\nFrequency Response:")
    #         print(f"  Resonant frequency: ωᵣ = {omega_r:.3f} rad/s ({omega_r/(2*np.pi):.3f} Hz)")
    #         print(f"  Resonant peak: {peak:.2f} ({20*np.log10(peak):.1f} dB)")

    #     print("\nPhysical Interpretation:")
    #     print("  - x[k]: Displacement from equilibrium [m]")
    #     print("  - v[k]: Velocity [m/s]")
    #     print("  - u[k]: External force/acceleration [m/s²]")

    #     print("\nTypical Applications:")
    #     print("  - Mechanical vibration isolation")
    #     print("  - Suspension systems (automotive, seismic)")
    #     print("  - Sensor/actuator dynamics")
    #     print("  - RLC electrical circuits (analogous)")
    #     print("  - Building structural dynamics")

    #     print("=" * 70)