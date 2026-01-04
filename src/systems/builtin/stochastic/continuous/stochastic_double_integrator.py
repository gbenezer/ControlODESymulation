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
Stochastic Double Integrator - Canonical LQG Benchmark System
==============================================================

This module provides the stochastic double integrator, the fundamental benchmark
system for testing stochastic optimal control and estimation algorithms. The
stochastic double integrator serves as:

- The canonical example for Linear-Quadratic-Gaussian (LQG) control theory
- The simplest system exhibiting position-velocity dynamics with uncertainty
- A benchmark for Kalman filtering and state estimation algorithms
- The foundation for understanding stochastic controllability and observability
- A model for point-mass dynamics under random disturbances

The stochastic double integrator represents the most basic model of motion
under uncertainty, making it the "hello world" of stochastic control theory.
It captures the essential physics of moving objects (position, velocity, force)
while remaining analytically tractable.

Physical Motivation
-------------------

**Newton's Second Law with Random Forcing:**

Consider a unit mass moving in 1D under applied force u(t) and random
disturbances w(t):
    
    m·ẍ = u + F_random

For m = 1 (unit mass):
    ẍ = u + w(t)

In state-space form (position x, velocity v):
    ẋ = v
    v̇ = u + w(t)

This is the stochastic double integrator

**Physical Systems:**

1. **Spacecraft Position Control:**
   - No gravity (deep space)
   - Thruster force u (control)
   - Solar radiation pressure, micrometeorites (random)
   - Position x, velocity v states

2. **Ground Vehicle (1D):**
   - Horizontal motion (flat surface)
   - Motor force u (control)
   - Wind gusts, road roughness (random)
   - Frictionless assumption (or friction in u)

3. **Inventory Control:**
   - Stock level x (position analog)
   - Order rate v (velocity analog)  
   - Order quantity u (control)
   - Demand fluctuations (random)

4. **Economic Systems:**
   - Capital stock x
   - Investment flow v
   - Policy intervention u
   - Economic shocks (random)

5. **Robot End-Effector:**
   - Cartesian position x
   - Velocity v
   - Motor command u
   - Model uncertainty, external forces (random)

**Random Disturbance Sources:**

The stochastic forcing w(t) models:
- External disturbances: Wind, vibrations, impacts
- Model uncertainty: Unmodeled dynamics, parameter errors
- Measurement noise: Sensor errors (if added to measurement)
- Actuator noise: Imperfect force application

Mathematical Formulation
------------------------

**Stochastic Differential Equations:**

State-space form:
    dx = v·dt
    dv = u·dt + σ·dW

where:
    - x(t): Position (unbounded, can be positive or negative)
    - v(t): Velocity (unbounded)
    - u: Control force (deterministic input)
    - σ: Noise intensity [acceleration/√time]
    - W(t): Standard Wiener process
    - dW ~ N(0, dt): Brownian increment

**Matrix Form:**
    dX = (A·X + B·u)·dt + G·dW

where:
    X = [x, v]ᵀ
    A = [0  1]
        [0  0]
    B = [0]
        [1]
    G = [0]
        [σ]

**Linear Dynamics:**
This is a **linear SDE** - the simplest non-trivial case.

**Key Properties:**
- Drift: Linear in state (A·X) and control (B·u)
- Diffusion: Constant (additive noise)
- Double integrator: Two integrations from force to position

**Why the name "Double Integrator"?**

From control input u to position x requires two integrations:
1. First integration: u → v (force to velocity)
2. Second integration: v → x (velocity to position)

Frequency domain: x/u = 1/s² (two poles at origin).

Analytical Solution
-------------------

**Exact Solution:**

For initial condition X(0) = [x₀, v₀]ᵀ and constant control u:

Position:
    x(t) = x₀ + v₀·t + (1/2)·u·t² + ∫₀ᵗ σ·(t-s)·dW(s)

Velocity:
    v(t) = v₀ + u·t + ∫₀ᵗ σ·dW(s)

**Moments:**

Mean:
    E[x(t)] = x₀ + v₀·t + (1/2)·u·t²
    E[v(t)] = v₀ + u·t

Variance:
    Var[x(t)] = σ²·t³/3
    Var[v(t)] = σ²·t

Covariance:
    Cov[x(t), v(t)] = σ²·t²/2

**Distribution:**

    [x(t)]     [x₀ + v₀·t + u·t²/2]     σ²·[t³/3   t²/2]
    [v(t)] ~ N([v₀ + u·t           ],       [t²/2   t  ])

Bivariate normal with covariance growing with time.

**Asymptotic Behavior:**

Mean grows quadratically: E[x(t)] ~ u·t²/2
Variance grows cubically: Var[x(t)] ~ σ²·t³/3

System is **non-stationary** - variance unbounded.

Key Properties
--------------

**1. Linearity:**
Superposition principle applies:
- Solution is sum of homogeneous + particular + stochastic parts
- Enables exact analytical solution
- Kalman filter optimal (no approximation)

**2. Controllability:**
Completely controllable (rank[B, A·B] = 2).
- Can reach any state in finite time
- With noise: Reachable set is probability distribution

**3. Observability:**
Depends on measurement:
- Measure x only: Observable (can reconstruct v from dx/dt)
- Measure v only: NOT observable (x decoupled)
- Measure both: Trivially observable

**4. Marginally Stable:**
Eigenvalues of A: {0, 0} (double pole at origin).
- Not asymptotically stable (perturbations don't decay)
- Not unstable (perturbations don't grow exponentially)
- Critically stable (perturbations persist)

**5. Non-Stationary:**
No equilibrium distribution - variance grows indefinitely.

**6. Additive Noise:**
Noise enters through velocity equation only (not position).
- Physical: Force disturbances, not position disturbances
- G = [0, σ]ᵀ: Noise on velocity dynamics

**7. Gaussian Process:**
X(t) is Gaussian for all t (linear dynamics preserve Gaussianity).

Linear-Quadratic-Gaussian (LQG) Control
----------------------------------------

**Why Double Integrator is THE LQG Benchmark:**

The stochastic double integrator is the canonical example for LQG because:
1. **Simplest non-trivial system:** Two states, one control
2. **Analytically tractable:** Closed-form Riccati solutions
3. **Separation principle applies:** Design LQR and Kalman filter independently
4. **Practical relevance:** Models real physical systems
5. **Pedagogical value:** Understand before complex systems

**LQG Problem Formulation:**

**Cost Functional:**
    J = E[∫₀^∞ (xᵀ·Q·x + u·R·u)·dt]

where:
- Q: State penalty (positive semi-definite)
  * Q = diag(q_x, q_v): Penalize position and velocity errors
  * Typical: q_x >> q_v (position more important)
  
- R: Control penalty (positive definite)
  * R = r > 0: Penalize control effort
  * Larger r → less aggressive control

**Separation Principle:**

LQG separates into two independent problems:

1. **LQR (Optimal Control):**
   Assume perfect state knowledge.
   Optimal control: u = -K·X̂
   
   Gain K from algebraic Riccati equation (ARE):
       A^T·P + P·A - P·B·R^(-1)·B^T·P + Q = 0

2. **Kalman Filter (Optimal Estimation):**
   Estimate state from noisy measurements.
   Estimate: dX̂ = (A·X̂ + B·u)·dt + L·(dy - C·X̂·dt)
   
   Gain L from dual Riccati equation.

3. **Combine:**
   Use estimated state: u = -K·X̂

**Certainty Equivalence:**
Optimal controller uses state estimate as if it were true state.
No cautious behavior needed (Gaussian assumption).

**LQG Solution for Double Integrator:**

For typical Q = diag(1, 0.1), R = 0.1:
- LQR gain: K ≈ [3.16, 4.43] (computed from ARE)
- Kalman gain: L depends on measurement noise
- Closed-loop stable (with probability 1)

Kalman Filter Application
--------------------------

**Measurement Models:**

**Case 1: Position Measurement Only**
    y = x + v_meas
    
where v_meas ~ N(0, R_meas).

Kalman filter:
- Estimates both x and v from x measurements
- Uses dynamics (integration) to infer velocity
- Estimation error bounded

**Case 2: Both Position and Velocity**
    y = [x, v]ᵀ + noise

Trivial estimation (direct measurement).

**Case 3: Velocity Only**
    y = v + v_meas

Cannot estimate x (unobservable) - x drifts arbitrarily.

**Optimal Filtering:**

For linear Gaussian systems, Kalman filter is:
- Optimal (minimum mean-square error)
- Exact (no approximation)
- Recursive (efficient online computation)

**Continuous-Time Kalman-Bucy Filter:**

    dX̂ = (A·X̂ + B·u)·dt + P·Cᵀ·R^(-1)·(dy - C·X̂·dt)

where P satisfies Riccati equation:
    Ṗ = A·P + P·Aᵀ - P·Cᵀ·R^(-1)·C·P + G·Qₙ·Gᵀ

At steady state (t → ∞): Ṗ = 0 gives algebraic equation.

Physical Interpretation of Noise
---------------------------------

**Velocity Noise (σ):**

The noise enters the velocity equation:
    dv = u·dt + σ·dW

Physical sources:
- Random forces: Wind gusts, vibrations
- Actuator noise: Imperfect force application
- Model uncertainty: Unmodeled dynamics
- Process disturbances: External impacts

**Units:** [acceleration/√time]
- Example: σ = 0.1 m/(s²·√s) = 0.1 m/s^(3/2)

**Effect on Position:**

Noise on velocity integrates to position:
    x(t) = ... + ∫₀ᵗ ∫₀ˢ σ·dW(τ)·ds
         = ... + ∫₀ᵗ σ·(t-s)·dW(s)

Double integration amplifies noise:
- Velocity variance: σ²·t (linear growth)
- Position variance: σ²·t³/3 (cubic growth)

Position more uncertain than velocity over time!

**Why Noise on Velocity, Not Position?**

Physical: Forces cause accelerations (Newton's law).
- Direct position noise would violate Newton's law
- Velocity noise is physically meaningful
- Position noise would imply instantaneous displacement

Applications
------------

**1. Control Theory:**

**LQG Benchmark:**
- Test bed for LQG algorithms
- Validate implementation
- Compare controllers (LQR, H∞, MPC)

**Robust Control:**
- Model uncertainty via σ
- Design robust controllers (H∞, μ-synthesis)

**Adaptive Control:**
- Learn dynamics online
- Parameter estimation

**2. Robotics:**

**Point Mass Robot:**
- 1D motion (extend to 2D/3D)
- Position control with disturbances
- Path tracking under uncertainty

**UAV Position Control:**
- Each axis (x, y, z) is double integrator
- Wind disturbances modeled as σ·dW

**Manipulator (Simplified):**
- Each joint (after feedback linearization)
- Joint position/velocity with noise

**3. Aerospace:**

**Spacecraft:**
- Position control (no gravity, no friction)
- Thruster force u + random forces
- Optimal fuel consumption (LQR)

**Missile Guidance:**
- Lateral position control
- Aerodynamic disturbances

**4. Economics:**

**Capital Accumulation:**
- Capital stock x
- Investment flow v  
- Policy u
- Economic shocks σ·dW

**5. Signal Processing:**

**Tracking Filter:**
- Track object position from noisy measurements
- Constant velocity model + acceleration noise
- Benchmark for Kalman filter

**6. Machine Learning:**

**Stochastic Optimization:**
- Parameter x (position)
- Gradient v (velocity)
- Stochastic gradient descent
- Noise from mini-batching

Stochastic Controllability
---------------------------

**Controllability with Noise:**

Unlike deterministic controllability (reach exact point), stochastic
controllability asks:
- Can we steer probability distribution to target?
- What is minimum uncertainty achievable?

**Reachable Set:**

At time T, reachable states form probability distribution:
    X(T) ~ N(μ(T), Σ(T))

where:
- μ(T): Controllable (choose via u)
- Σ(T): NOT fully controllable (noise accumulation)

**Minimum Variance:**

Even with perfect control, cannot eliminate noise:
    Var[x(T)] ≥ σ²·T³/3

This is fundamental limit from stochastic nature.

**Covariance Steering:**

Can we reach specific covariance Σ_target?

For double integrator: Limited control over covariance.
- Diagonal variance: Controlled by T
- Correlation: Fixed by dynamics

**Information Control:**

Can steer information matrix (inverse covariance).
Applications: Sensor placement, experiment design.

Numerical Integration
---------------------

**Exact Discretization (Preferred):**

For linear SDE with constant coefficients, exact discrete-time model exists:

    X[k+1] = Φ·X[k] + Γ·u[k] + w_d[k]

where:
    Φ = exp(A·Δt) = [1  Δt]
                     [0   1]
    
    Γ = ∫₀^Δt exp(A·s)·ds·B = [Δt²/2]
                                [Δt   ]
    
    w_d ~ N(0, Q_d)
    
    Q_d = ∫₀^Δt exp(A·s)·G·Qₙ·Gᵀ·exp(Aᵀ·s)·ds
        = σ²·[Δt³/3   Δt²/2]
              [Δt²/2   Δt   ]

This is **exact** - no discretization error!

**Euler-Maruyama (Simple but Approximate):**

    x[k+1] = x[k] + v[k]·Δt
    v[k+1] = v[k] + u[k]·Δt + σ·√Δt·Z[k]

Simpler but O(Δt) error.

**Recommended:**
Use exact discretization for double integrator (known closed form).

LQG Design Example
------------------

**Problem:** Regulate to origin with minimum cost.

**Cost:**
    J = E[∫₀^∞ (q_x·x² + q_v·v² + r·u²)·dt]

**Solution:**

1. **Solve LQR (Control):**
   ARE: Aᵀ·P + P·A - P·B·R^(-1)·Bᵀ·P + Q = 0
   Gain: K = R^(-1)·Bᵀ·P

2. **Solve Kalman Filter (Estimation):**
   Dual ARE: Σ̇ = A·Σ + Σ·Aᵀ - Σ·Cᵀ·R^(-1)·C·Σ + G·Qₙ·Gᵀ
   Gain: L = Σ·Cᵀ·R^(-1)

3. **Implement LQG:**
   u = -K·X̂

**Typical Parameters:**

    Q = [100   0  ]  (penalize position heavily)
        [ 0   10  ]  (penalize velocity moderately)
    R = 1            (penalize control)
    
    Process noise: σ = 0.1
    Measurement noise: R_meas = 0.01 (position only)

**Result:**
- Closed-loop stable
- Position error ~ cm (depending on σ, R_meas)
- Control smooth (not bang-bang)

Applications to Practical Problems
-----------------------------------

**1. Satellite Attitude Control:**

Each axis (roll, pitch, yaw) modeled as double integrator:
- Position: Angle
- Velocity: Angular rate
- Control: Thruster torque
- Noise: Disturbance torques (solar, magnetic)

**2. Automated Guided Vehicle (AGV):**

1D motion along track:
- Position: x along track
- Velocity: v
- Control: Motor force
- Noise: Floor irregularities, wheel slip

**3. Drone Altitude Control:**

Vertical position:
- Position: Height z
- Velocity: Vertical velocity
- Control: Thrust
- Noise: Wind, air density variations

**4. Economic Policy:**

Macroeconomic stabilization:
- Output gap x (deviation from potential)
- Growth rate v
- Policy intervention u (fiscal/monetary)
- Shocks: Technology, preference, supply

**5. Portfolio Rebalancing:**

Asset position management:
- Position: Holdings
- Flow: Rebalancing rate
- Control: Trade execution
- Noise: Market impact, price movements

Extensions and Variations
--------------------------

**1. With Damping:**
    dv = (-b·v + u)·dt + σ·dW

Adds friction/drag. Changes to stable system (eigenvalues in LHP).

**2. Multiplicative Noise:**
    dv = u·dt + σ·v·dW

Noise proportional to velocity (more realistic for some systems).

**3. Position and Velocity Noise:**
    dx = v·dt + σ_x·dW_x
    dv = u·dt + σ_v·dW_v

Two independent noise sources.

**4. Colored Noise:**
    dv = u·dt + η(t)·dt
    dη = -α·η·dt + σ·dW

Noise has memory (OU process driving).

**5. Constrained:**
    |x| ≤ x_max (position limits)
    |v| ≤ v_max (velocity limits)
    |u| ≤ u_max (actuator saturation)

Requires constrained LQG or MPC.

**6. Higher Dimensions:**

2D double integrator:
    dx = v_x·dt
    dv_x = u_x·dt + σ_x·dW_x
    dy = v_y·dt  
    dv_y = u_y·dt + σ_y·dW_y

Independent or coupled via control/noise.

**7. Triple Integrator:**
    dx = v·dt
    dv = a·dt
    da = u·dt + σ·dW

Jerk-controlled system (smoother trajectories).

Common Pitfalls
---------------

1. **Confusing Deterministic and Stochastic Controllability:**
   - Deterministic: Can reach exact point
   - Stochastic: Can reach probability distribution
   - Minimum variance achievable (cannot eliminate noise)

2. **Ignoring Variance Growth:**
   - Position variance grows as t³ (very fast!)
   - Long-time control challenging
   - Need feedback to bound variance

3. **Wrong Noise Placement:**
   - Physically: Noise on velocity (forces)
   - Not on position (violates Newton's law)
   - G = [0, σ]ᵀ correct, G = [σ, 0]ᵀ wrong

4. **Assuming Stationarity:**
   - Open-loop non-stationary
   - Only stationary in closed-loop
   - Infinite-horizon cost requires stabilization

5. **Discretization Errors:**
   - Euler-Maruyama has O(Δt) error
   - Use exact discretization (known for linear systems)

6. **Measurement Model Confusion:**
   - Position only: Observable (can estimate v)
   - Velocity only: NOT observable (x drifts)
   - Both: Trivial

**Impact:**
Double integrator demonstrated that:
- Stochastic control can be tractable
- Optimal controllers are linear (for quadratic cost, Gaussian noise)
- Estimation and control separate (certainty equivalence)

Testing and Validation
-----------------------

**Analytical Validation:**

1. **Moment Check:**
   - Compare simulated mean/variance to analytical
   - Should match theoretical formulas

2. **Normality:**
   - State should be Gaussian
   - Q-Q plots should be linear

3. **LQR Verification:**
   - Closed-loop eigenvalues in LHP
   - Cost matches theoretical prediction

4. **Kalman Filter:**
   - Innovation sequence should be white noise
   - Covariance P matches theoretical

**Numerical Validation:**

1. **Exact vs Euler:**
   - Compare exact discretization with Euler-Maruyama
   - Verify O(Δt) convergence for Euler

2. **Variance Growth:**
   - Position variance ~ t³
   - Velocity variance ~ t
   - Linear regression: log(Var) vs log(t)

"""

import numpy as np
import sympy as sp

from src.systems.base.core.continuous_stochastic_system import ContinuousStochasticSystem


class StochasticDoubleIntegrator(ContinuousStochasticSystem):
    """
    Stochastic double integrator - canonical LQG benchmark system.

    The fundamental linear stochastic system combining position-velocity
    dynamics with random forcing. This is the "hello world" of stochastic
    control theory and the canonical example for LQG control and Kalman filtering.

    Stochastic Differential Equations
    ----------------------------------
    State-space form:
        dx = v·dt
        dv = u·dt + σ·dW

    Matrix form:
        dX = (A·X + B·u)·dt + G·dW

    where:
        X = [x, v]ᵀ: Position and velocity
        A = [0  1]: Drift matrix (double integrator)
            [0  0]
        B = [0]: Control input matrix
            [1]
        G = [0]: Noise input matrix (velocity noise only)
            [σ]
        u: Control force
        W(t): Standard Wiener process

    Physical Interpretation
    -----------------------
    **Newton's Second Law with Random Forces:**

    For unit mass: F = ma
        a = u + w(t)

    where u is applied force and w(t) is random disturbance.

    **State Variables:**
    - x: Position [m] (unbounded)
    - v: Velocity [m/s] (unbounded)

    **Control:**
    - u: Applied force [m/s²] (acceleration)

    **Noise:**
    - σ: Force noise intensity [m/(s²·√s)]
    - Enters velocity equation (Newton's law)

    Key Features
    ------------
    **Linearity:**
    Linear drift (A·X + B·u) enables exact solution.

    **Additive Noise:**
    Constant diffusion G (state-independent).

    **Double Integration:**
    Control u → velocity v → position x (two integrations).

    **Marginally Stable:**
    Eigenvalues: {0, 0} (poles at origin).
    - Not asymptotically stable
    - Requires feedback for regulation

    **Non-Stationary:**
    Variance grows unboundedly without control.

    **Gaussian:**
    Linear dynamics preserve Gaussianity.

    Mathematical Properties
    -----------------------
    **Exact Solution (u constant):**

    Position:
        x(t) = x₀ + v₀·t + (1/2)·u·t² + ∫₀ᵗ σ·(t-s)·dW(s)

    Velocity:
        v(t) = v₀ + u·t + σ·W(t)

    **Moments:**
    Mean:
        E[x(t)] = x₀ + v₀·t + (1/2)·u·t²
        E[v(t)] = v₀ + u·t

    Variance:
        Var[x(t)] = σ²·t³/3
        Var[v(t)] = σ²·t

    Covariance:
        Cov[x(t), v(t)] = σ²·t²/2

    **Distribution:**
        [x(t)]     [E[x]]     σ²·[t³/3   t²/2]
        [v(t)] ~ N([E[v]],        [t²/2   t  ])

    **Variance Growth:**
    - Position: O(t³) - very fast!
    - Velocity: O(t) - moderate
    - Correlation: O(t²)

    Physical Interpretation
    -----------------------
    **Noise Intensity σ:**
    - Units: [m/(s²·√s)] = [m/s^(3/2)]
    - Controls disturbance force magnitude
    - Typical: 0.01-1.0 depending on application

    **RMS Values:**
    After time t:
    - Position RMS: σ·t^(3/2)/√3
    - Velocity RMS: σ·√t

    **Why Position Variance Grows Faster:**
    Noise on velocity integrates to position:
    - Small velocity errors accumulate
    - Integration amplifies uncertainty
    - Requires active control to bound

    State Space
    -----------
    State: X = [x, v] ∈ ℝ²
        - Unbounded (can take any real values)
        - No equilibrium without control
        - Non-stationary (variance grows)

    Control: u ∈ ℝ
        - Force/acceleration input
        - Can be state feedback: u = u(x, v)

    Noise: w ∈ ℝ
        - Single Wiener process
        - Enters velocity equation only
        - G = [0, σ]ᵀ

    Parameters
    ----------
    sigma : float, default=0.1
        Noise intensity [m/(s²·√s)]
        - Controls disturbance magnitude
        - Typical: 0.01-1.0
        - Larger σ → more uncertainty

    m : float, default=1.0
        Mass [kg] (optional, typically normalized to 1)
        - Can include for dimensional analysis
        - Usually set to 1 (unit mass)

    Stochastic Properties
    ---------------------
    - System Type: LINEAR
    - Noise Type: ADDITIVE (constant)
    - SDE Type: Itô
    - Noise Dimension: nw = 1
    - Stationary: No (open-loop)
    - Gaussian: Yes (linear dynamics)
    - Controllable: Yes (completely)
    - Observable: Depends on measurement

    Applications
    ------------
    **1. LQG Control:**
    Canonical benchmark for Linear-Quadratic-Gaussian:
    - Separation principle demonstration
    - Kalman filter + LQR combination
    - Optimal for quadratic cost

    **2. Kalman Filtering:**
    Test state estimation algorithms:
    - Position measurement → estimate velocity
    - Optimal for Gaussian noise
    - Benchmark filter performance

    **3. Robotics:**
    - UAV position control (per axis)
    - Robot manipulator (per joint)
    - Mobile robot navigation

    **4. Aerospace:**
    - Spacecraft position control
    - Missile guidance
    - Aircraft lateral control

    **5. Economics:**
    - Capital accumulation models
    - Inventory control
    - Policy design under uncertainty

    Numerical Integration
    ---------------------
    **Exact Discretization (Recommended):**
    Use closed-form discrete-time model:
        X[k+1] = Φ·X[k] + Γ·u + w_d
    
    No discretization error, preserves all properties.

    **SDE Integration:**
    - Euler-Maruyama: Simple, O(√Δt) strong convergence
    - Milstein: Same as Euler for additive noise
    - Framework solvers: All work well (linear, non-stiff)

    LQG Control Design
    ------------------
    **Cost Function:**
        J = E[∫ (Xᵀ·Q·X + u·R·u)·dt]

    **Optimal Control:**
        u* = -K·X̂
    
    where K from LQR, X̂ from Kalman filter.

    **Separation Principle:**
    Design independently:
    - LQR gain K (assume perfect state)
    - Kalman gain L (estimate state)
    - Combine: u = -K·X̂

    Comparison with Deterministic
    ------------------------------
    **Deterministic:**
    - ẍ = u (no noise)
    - Perfect control possible
    - Exact regulation

    **Stochastic:**
    - ẍ = u + w (noise)
    - Minimum variance limit
    - Probabilistic regulation

    **Critical Difference:**
    Cannot eliminate noise effect - fundamental limit
    on achievable performance.

    Limitations
    -----------
    - No damping (purely integrating)
    - Linear (no nonlinear effects)
    - Additive noise only
    - Single control input
    - No constraints

    Extensions
    ----------
    - Add damping: -b·v term
    - Nonlinear: u·v coupling
    - Multiplicative noise
    - Constraints: |x| ≤ x_max
    - Higher dimensions: 2D, 3D

    Examples
    --------
    Basic double integrator:
    
    >>> # Standard configuration
    >>> system = StochasticDoubleIntegrator(sigma=0.1)
    >>> 
    >>> # Check properties
    >>> print(f"Linear system: {system.is_linear()}")  # True
    >>> print(f"Additive noise: {system.is_additive_noise()}")  # True
    >>> print(f"State dimension: {system.nx}")  # 2

    Evaluate dynamics:
    
    >>> # At rest with unit force
    >>> x = np.array([0.0, 0.0])
    >>> u = np.array([1.0])
    >>> drift = system.drift(x, u)  # [0, 1]
    >>> diffusion = system.diffusion(x, u)  # [[0], [0.1]]

    Different noise levels:
    
    >>> # Low noise (precise environment)
    >>> low_noise = StochasticDoubleIntegrator(sigma=0.01)
    >>> 
    >>> # High noise (harsh environment)
    >>> high_noise = StochasticDoubleIntegrator(sigma=1.0)

    Simulation:
    
    >>> # Free motion with noise (u=0)
    >>> x0 = np.array([1.0, 0.0])  # Start at x=1, v=0
    >>> result = system.integrate(
    ...     x0=x0,
    ...     u=None,  # Zero control
    ...     t_span=(0, 10),
    ...     method='euler-maruyama',
    ...     dt=0.01
    ... )
    >>> 
    >>> # Variance should grow
    >>> positions = result['y'][0, :]
    >>> velocities = result['y'][1, :]
    >>> print(f"Position variance: {np.var(positions):.3f}")

    See Also
    --------
    DoubleIntegrator : Deterministic version
    OrnsteinUhlenbeck : With damping (stable)
    """

    def define_system(
        self,
        sigma: float = 0.1,
        m: float = 1.0,
    ):
        """
        Define stochastic double integrator dynamics.

        Sets up linear SDE:
            dx = v·dt
            dv = u·dt + σ·dW

        Parameters
        ----------
        sigma : float, default=0.1
            Noise intensity [m/(s²·√s)]
            - Controls disturbance force magnitude
            - Typical: 0.01-1.0
            - Larger σ → more uncertainty

        m : float, default=1.0
            Mass [kg] (optional, typically 1)
            - Usually normalized to unit mass
            - Can include for dimensional analysis

        Notes
        -----
        **System Structure:**
        
        Linear drift:
            A = [0  1]  (standard double integrator)
                [0  0]
        
        Control matrix:
            B = [0]  (force affects acceleration)
                [1/m]
        
        Noise matrix:
            G = [0]  (noise on velocity only)
                [σ]

        **Eigenvalues:**
        λ = {0, 0} - double pole at origin (marginally stable).

        **Controllability:**
        rank[B, A·B] = 2 - completely controllable.

        **Observability:**
        If measuring x: rank[C; C·A] = 2 - observable
        If measuring v: NOT observable (x decoupled)

        **Noise Placement:**
        Physically correct: Noise on velocity (forces cause accelerations).
        
        Alternative (wrong): G = [σ, 0]ᵀ would be position noise,
        violating Newton's law.

        **Variance Growth:**
        Without control:
        - Var[x(t)] = σ²·t³/3 (cubic!)
        - Var[v(t)] = σ²·t (linear)

        Requires feedback to stabilize.

        **Parameter Scaling:**
        
        For spacecraft (SI units):
        - Position: meters
        - Velocity: m/s
        - Force: Newtons (= kg·m/s² = m/s² for m=1)
        - σ: m/s^(3/2)

        For normalized units:
        - Set m = 1 (dimensionless)
        - Position, velocity in natural units
        - Force = acceleration

        Examples
        --------
        >>> # Standard unit mass system
        >>> system = StochasticDoubleIntegrator(sigma=0.1, m=1.0)
        >>> 
        >>> # Spacecraft (1000 kg)
        >>> spacecraft = StochasticDoubleIntegrator(
        ...     sigma=0.01,  # Small disturbances
        ...     m=1000.0
        ... )
        >>> 
        >>> # High noise environment
        >>> harsh = StochasticDoubleIntegrator(sigma=1.0, m=1.0)
        """
        if sigma < 0:
            raise ValueError(f"sigma must be non-negative, got {sigma}")
        if m <= 0:
            raise ValueError(f"mass must be positive, got {m}")

        # State variables
        x, v = sp.symbols("x v", real=True)
        u = sp.symbols("u", real=True)

        # Parameters
        sigma_sym = sp.symbols("sigma", real=True, nonnegative=True)
        m_sym = sp.symbols("m", positive=True)

        self.state_vars = [x, v]
        self.control_vars = [u]

        # DRIFT (Deterministic dynamics)
        # dx/dt = v
        # dv/dt = u/m
        self._f_sym = sp.Matrix([
            v,
            u / m_sym
        ])

        self.parameters = {
            sigma_sym: sigma,
            m_sym: m,
        }
        self.order = 1  # First-order state-space (second-order in physics)

        # DIFFUSION (Stochastic part)
        # Noise on velocity only (physical: random forces)
        self.diffusion_expr = sp.Matrix([
            [0],
            [sigma_sym]
        ])

        # Itô SDE
        self.sde_type = "ito"

        # Output: Typically measure position only
        self._h_sym = sp.Matrix([x])

    def setup_equilibria(self):
        """
        Set up equilibrium points.

        For open-loop double integrator, only equilibrium is origin
        with zero control. System is marginally stable (not asymptotically).
        """
        self.add_equilibrium(
            "origin",
            x_eq=np.array([0.0, 0.0]),
            u_eq=np.array([0.0]),
            verify=True,
            stability="marginally_stable",
            notes="Origin is marginally stable equilibrium (eigenvalues at 0). "
                  "Without control, variance grows unboundedly. "
                  "Requires feedback (LQR) for asymptotic stability."
        )
        self.set_default_equilibrium("origin")

    def get_variance_at_time(self, t: float) -> np.ndarray:
        """
        Get analytical variance matrix at time t (open-loop, u=0, X₀=0).

        Returns 2×2 covariance matrix.

        Parameters
        ----------
        t : float
            Time [s]

        Returns
        -------
        np.ndarray
            Covariance matrix [[Var(x), Cov(x,v)],
                               [Cov(x,v), Var(v)]]

        Notes
        -----
        For open-loop (u=0) starting from origin:
            Var[x] = σ²·t³/3
            Var[v] = σ²·t
            Cov[x,v] = σ²·t²/2

        Examples
        --------
        >>> system = StochasticDoubleIntegrator(sigma=0.1)
        >>> Sigma = system.get_variance_at_time(t=10.0)
        >>> print(f"Position variance: {Sigma[0,0]:.2f}")
        >>> print(f"Velocity variance: {Sigma[1,1]:.2f}")
        """
        sigma = self.parameters[sp.symbols('sigma')]
        
        var_x = sigma**2 * t**3 / 3
        var_v = sigma**2 * t
        cov_xv = sigma**2 * t**2 / 2
        
        return np.array([
            [var_x, cov_xv],
            [cov_xv, var_v]
        ])

    def get_position_std(self, t: float) -> float:
        """
        Get analytical position standard deviation at time t.

        σ_x(t) = σ·t^(3/2)/√3

        Parameters
        ----------
        t : float
            Time [s]

        Returns
        -------
        float
            Position standard deviation

        Examples
        --------
        >>> system = StochasticDoubleIntegrator(sigma=0.1)
        >>> std_10s = system.get_position_std(t=10.0)
        >>> print(f"Position std after 10s: {std_10s:.3f} m")
        """
        sigma = self.parameters[sp.symbols('sigma')]
        return sigma * t**(1.5) / np.sqrt(3)

    def get_velocity_std(self, t: float) -> float:
        """
        Get analytical velocity standard deviation at time t.

        σ_v(t) = σ·√t

        Parameters
        ----------
        t : float
            Time [s]

        Returns
        -------
        float
            Velocity standard deviation

        Examples
        --------
        >>> system = StochasticDoubleIntegrator(sigma=0.1)
        >>> std_10s = system.get_velocity_std(t=10.0)
        >>> print(f"Velocity std after 10s: {std_10s:.3f} m/s")
        """
        sigma = self.parameters[sp.symbols('sigma')]
        return sigma * np.sqrt(t)


# Convenience functions
def create_spacecraft_model(
    noise_level: float = 0.01,
    mass: float = 1000.0,
) -> StochasticDoubleIntegrator:
    """
    Create double integrator for spacecraft position control.

    Parameters
    ----------
    noise_level : float, default=0.01
        Disturbance force noise [m/(s²·√s)]
        - Typical: 0.001-0.1 for spacecraft
        
    mass : float, default=1000.0
        Spacecraft mass [kg]

    Returns
    -------
    StochasticDoubleIntegrator

    Notes
    -----
    Spacecraft in deep space (no gravity, no friction):
    - Perfect double integrator
    - Random forces: Solar pressure, micrometeorites
    - Thruster control with noise

    Examples
    --------
    >>> # Small spacecraft with low disturbances
    >>> satellite = create_spacecraft_model(
    ...     noise_level=0.001,
    ...     mass=100.0
    ... )
    """
    return StochasticDoubleIntegrator(sigma=noise_level, m=mass)


def create_uav_axis_model(
    axis: str = 'z',
    noise_level: float = 0.1,
) -> StochasticDoubleIntegrator:
    """
    Create double integrator for UAV single-axis control.

    Parameters
    ----------
    axis : str, default='z'
        Axis name ('x', 'y', or 'z')
    noise_level : float, default=0.1
        Wind/disturbance noise [m/(s²·√s)]

    Returns
    -------
    StochasticDoubleIntegrator

    Notes
    -----
    Each UAV axis (after feedback linearization) is approximately
    a double integrator with noise from wind and model uncertainty.

    Examples
    --------
    >>> # Altitude control (z-axis)
    >>> z_axis = create_uav_axis_model(axis='z', noise_level=0.1)
    >>> 
    >>> # Horizontal position (windy conditions)
    >>> x_axis = create_uav_axis_model(axis='x', noise_level=0.5)
    """
    return StochasticDoubleIntegrator(sigma=noise_level, m=1.0)


def create_lqg_benchmark(
    sigma: float = 0.1,
) -> StochasticDoubleIntegrator:
    """
    Create standard LQG benchmark problem.

    Parameters
    ----------
    sigma : float, default=0.1
        Process noise intensity

    Returns
    -------
    StochasticDoubleIntegrator

    Notes
    -----
    Standard configuration for testing LQG algorithms:
    - Unit mass (m=1)
    - Moderate noise (σ=0.1)
    - Typical cost: Q = diag(100, 10), R = 1

    Examples
    --------
    >>> # Create benchmark
    >>> benchmark = create_lqg_benchmark(sigma=0.1)
    >>> 
    >>> # Design LQR
    >>> x_eq = np.zeros(2)
    >>> u_eq = np.zeros(1)
    >>> A, B = benchmark.linearize(x_eq, u_eq)
    >>> 
    >>> # LQR weights
    >>> Q = np.diag([100.0, 10.0])
    >>> R = np.array([[1.0]])
    >>> 
    >>> # Compute LQR gain
    >>> from scipy.linalg import solve_continuous_are
    >>> P = solve_continuous_are(A, B, Q, R)
    >>> K = np.linalg.inv(R) @ B.T @ P
    >>> print(f"LQR gain: K = {K}")
    """
    return StochasticDoubleIntegrator(sigma=sigma, m=1.0)