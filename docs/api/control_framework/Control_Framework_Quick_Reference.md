# Control Framework Quick Reference

## Quick Start

### 1. LQR Controller Design

```python
from src.systems.examples import Pendulum
import numpy as np

# Create system
system = Pendulum()

# Linearize at equilibrium
x_eq = np.array([np.pi, 0])  # Upright position
u_eq = np.zeros(1)
A, B = system.linearize(x_eq, u_eq)

# Design LQR controller
Q = np.diag([10, 1])   # State cost
R = np.array([[0.1]])  # Control cost

result = system.control.design_lqr(
    A, B, Q, R,
    system_type='continuous'
)

# Extract gain
K = result['gain']
print(f"LQR gain: {K}")
print(f"Stable: {np.all(np.real(result['closed_loop_eigenvalues']) < 0)}")

# Apply controller
u = -K @ (x - x_eq)
```

### 2. Kalman Filter Design

```python
# Define measurement model
C = np.array([[1, 0]])  # Measure position only

# Define noise covariances
Q_process = 0.01 * np.eye(2)  # Process noise
R_meas = np.array([[0.1]])     # Measurement noise

# Design Kalman filter
kalman = system.control.design_kalman(
    A, C, Q_process, R_meas,
    system_type='continuous'
)

# Extract Kalman gain
L = kalman['gain']
print(f"Observer stable: {np.all(np.abs(kalman['observer_eigenvalues']) < 1)}")

# Use in estimation loop
x_hat = np.zeros(2)
for k in range(N):
    # Prediction
    x_hat_pred = A @ x_hat + B @ u[k]
    
    # Correction
    innovation = y[k] - C @ x_hat_pred
    x_hat = x_hat_pred + L @ innovation
```

### 3. LQG Controller Design

```python
# Design combined LQR + Kalman filter
lqg = system.control.design_lqg(
    A, B, C,
    Q_state=np.diag([10, 1]),
    R_control=np.array([[0.1]]),
    Q_process=0.01 * np.eye(2),
    R_measurement=np.array([[0.1]]),
    system_type='continuous'
)

# Extract both gains
K = lqg['control_gain']      # LQR gain
L = lqg['estimator_gain']    # Kalman gain

print(f"LQG stable: {lqg['closed_loop_stable']}")
print(f"Separation verified: {lqg['separation_verified']}")

# Implement LQG controller
x_hat = np.zeros(2)
for k in range(N):
    # Control (certainty equivalence)
    u[k] = -K @ (x_hat - x_eq)
    
    # Prediction
    x_hat = A @ x_hat + B @ u[k]
    
    # Measurement update
    innovation = y[k] - C @ x_hat
    x_hat = x_hat + L @ innovation
```

### 4. System Analysis

```python
# Check stability
stability = system.analysis.stability(A, system_type='continuous')
print(f"Stable: {stability['is_stable']}")
print(f"Eigenvalues: {stability['eigenvalues']}")

# Check controllability
ctrl = system.analysis.controllability(A, B)
print(f"Controllable: {ctrl['is_controllable']}")
print(f"Rank: {ctrl['rank']}")

# Check observability
obs = system.analysis.observability(A, C)
print(f"Observable: {obs['is_observable']}")
```

---

## Control Design Methods

### LQR Controller

```python
# Continuous-time LQR
result = system.control.design_lqr(
    A, B,
    Q=np.diag([10, 1]),
    R=np.array([[0.1]]),
    system_type='continuous'
)

# Discrete-time LQR
result = system.control.design_lqr(
    A, B,
    Q=np.diag([10, 1]),
    R=np.array([[0.1]]),
    system_type='discrete'
)

# With cross-coupling term
result = system.control.design_lqr(
    A, B,
    Q=np.diag([10, 1]),
    R=np.array([[0.1]]),
    N=np.array([[0.5], [0.1]]),  # Cross-coupling
    system_type='continuous'
)

# Access result fields
K = result['gain']                         # Feedback gain (nu, nx)
P = result['cost_to_go']                   # Riccati solution (nx, nx)
eigs = result['closed_loop_eigenvalues']   # eig(A - BK)
margin = result['stability_margin']        # Robustness measure

# Verify stability
is_stable_continuous = np.all(np.real(eigs) < 0)
is_stable_discrete = np.all(np.abs(eigs) < 1)
```

### Kalman Filter

```python
# Continuous-time Kalman filter
kalman = system.control.design_kalman(
    A, C,
    Q=0.01 * np.eye(2),
    R=np.array([[0.1]]),
    system_type='continuous'
)

# Discrete-time Kalman filter
kalman = system.control.design_kalman(
    A, C,
    Q=0.01 * np.eye(2),
    R=np.array([[0.1]]),
    system_type='discrete'
)

# Access result fields
L = kalman['gain']                      # Kalman gain (nx, ny)
P = kalman['error_covariance']          # Error covariance (nx, nx)
S = kalman['innovation_covariance']     # Innovation cov (ny, ny)
eigs = kalman['observer_eigenvalues']   # eig(A - LC)

# Verify observer stability
obs_stable_continuous = np.all(np.real(eigs) < 0)
obs_stable_discrete = np.all(np.abs(eigs) < 1)
```

### LQG Controller

```python
# Design LQG (LQR + Kalman via separation principle)
lqg = system.control.design_lqg(
    A, B, C,
    Q_state=np.diag([10, 1]),      # LQR state cost
    R_control=np.array([[0.1]]),   # LQR control cost
    Q_process=0.01 * np.eye(2),    # Process noise covariance
    R_measurement=np.array([[0.1]]),  # Measurement noise covariance
    system_type='continuous'
)

# Access result fields
K = lqg['control_gain']                # LQR gain (nu, nx)
L = lqg['estimator_gain']              # Kalman gain (nx, ny)
P_ctrl = lqg['control_cost_to_go']     # Controller Riccati
P_est = lqg['estimation_error_covariance']  # Estimator Riccati
stable = lqg['closed_loop_stable']     # Overall stability
separated = lqg['separation_verified']  # Separation principle
ctrl_eigs = lqg['controller_eigenvalues']  # eig(A - BK)
est_eigs = lqg['estimator_eigenvalues']    # eig(A - LC)

# Verify separation principle
assert separated, "Separation principle should hold"
print(f"Combined system stable: {stable}")
```

---

## System Analysis Methods

### Stability Analysis

```python
# Continuous-time stability
stability = system.analysis.stability(A, system_type='continuous')

# Discrete-time stability
stability = system.analysis.stability(A, system_type='discrete')

# Access result fields
eigs = stability['eigenvalues']          # Complex eigenvalues
mags = stability['magnitudes']           # |λ| values
rho = stability['spectral_radius']       # max |λ|
is_stable = stability['is_stable']       # Asymptotic stability
is_marginal = stability['is_marginally_stable']  # On boundary
is_unstable = stability['is_unstable']   # Unstable

# Stability criteria
if stability['is_stable']:
    print("System is asymptotically stable")
elif stability['is_marginally_stable']:
    print("System is marginally stable")
else:
    print("System is unstable")
    print(f"Unstable eigenvalues: {eigs[np.real(eigs) > 0]}")  # Continuous
    print(f"Unstable eigenvalues: {eigs[np.abs(eigs) > 1]}")   # Discrete
```

### Controllability Analysis

```python
# Check controllability
ctrl = system.analysis.controllability(A, B)

# Access result fields
C_matrix = ctrl['controllability_matrix']  # (nx, nx*nu)
rank = ctrl['rank']                        # Rank of C_matrix
controllable = ctrl['is_controllable']     # rank == nx

# Check for uncontrollable modes
if not controllable:
    uncontrol_modes = ctrl['uncontrollable_modes']
    print(f"Uncontrollable eigenvalues: {uncontrol_modes}")
    
# Verify for LQR design
if controllable:
    print("✓ System is controllable - can design LQR")
else:
    print("✗ System not controllable - LQR may fail")
    print(f"Controllability rank: {rank} / {A.shape[0]}")
```

### Observability Analysis

```python
# Check observability
obs = system.analysis.observability(A, C)

# Access result fields
O_matrix = obs['observability_matrix']  # (nx*ny, nx)
rank = obs['rank']                      # Rank of O_matrix
observable = obs['is_observable']       # rank == nx

# Check for unobservable modes
if not observable:
    unobs_modes = obs['unobservable_modes']
    print(f"Unobservable eigenvalues: {unobs_modes}")

# Verify for Kalman filter design
if observable:
    print("✓ System is observable - can design Kalman filter")
else:
    print("✗ System not observable - Kalman filter may fail")
    print(f"Observability rank: {rank} / {A.shape[0]}")
```

---

## Common Patterns

### Pattern 1: Full State Feedback LQR

```python
# 1. Linearize at equilibrium
A, B = system.linearize(x_eq, u_eq)

# 2. Design LQR
Q = np.diag([10, 1])   # Penalize states
R = np.array([[0.1]])  # Penalize control
result = system.control.design_lqr(A, B, Q, R, system_type='continuous')
K = result['gain']

# 3. Implement controller
def lqr_controller(x, t):
    return -K @ (x - x_eq)

# 4. Simulate
result_sim = system.simulate(
    x0=x0,
    controller=lqr_controller,
    t_span=(0, 10),
    dt=0.01
)
```

### Pattern 2: Observer-Based Control

```python
# 1. Design controller (assumes full state)
A, B = system.linearize(x_eq, u_eq)
C = np.array([[1, 0]])  # Partial measurement

lqr = system.control.design_lqr(A, B, Q, R, system_type='continuous')
K = lqr['gain']

# 2. Design observer
kalman = system.control.design_kalman(A, C, Q_proc, R_meas, system_type='continuous')
L = kalman['gain']

# 3. Implement observer-based controller
x_hat = np.zeros(system.nx)

def observer_controller(x, t):
    # Use estimated state
    return -K @ (x_hat - x_eq)

# 4. Update loop
for k in range(N):
    # Get measurement
    y_meas = C @ x_true[k] + noise[k]
    
    # Control based on estimate
    u[k] = -K @ (x_hat - x_eq)
    
    # Propagate estimate
    x_hat_dot = A @ x_hat + B @ u[k] + L @ (y_meas - C @ x_hat)
    x_hat += dt * x_hat_dot
```

### Pattern 3: LQG Controller

```python
# 1. Design combined LQG
lqg = system.control.design_lqg(
    A, B, C,
    Q_state, R_control,
    Q_process, R_measurement,
    system_type='continuous'
)

K = lqg['control_gain']
L = lqg['estimator_gain']

# 2. Initialize estimate
x_hat = np.zeros(system.nx)

# 3. Control loop
for k in range(N):
    # Certainty equivalence control
    u[k] = -K @ (x_hat - x_eq)
    
    # Get measurement
    y_meas = C @ x_true[k] + meas_noise[k]
    
    # Prediction step
    x_hat = A @ x_hat + B @ u[k]
    
    # Correction step
    innovation = y_meas - C @ x_hat
    x_hat = x_hat + L @ innovation
```

### Pattern 4: Discrete-Time Control

```python
# 1. Design discrete LQR
Ad, Bd = discrete_system.linearize(x_eq, u_eq)
result = discrete_system.control.design_lqr(
    Ad, Bd, Q, R,
    system_type='discrete'  # Important!
)
K = result['gain']

# 2. Discrete control law
for k in range(N):
    u[k] = -K @ (x[k] - x_eq)
    x[k+1] = discrete_system.step(x[k], u[k])
```

### Pattern 5: Tuning LQR Weights

```python
# Bryson's rule: Normalize by maximum acceptable values
x_max = np.array([0.5, 2.0])  # Max acceptable deviation
u_max = np.array([10.0])      # Max acceptable control

Q = np.diag(1 / x_max**2)
R = np.diag(1 / u_max**2)

result = system.control.design_lqr(A, B, Q, R, system_type='continuous')

# Iterate if needed
for q_scale in [1.0, 10.0, 100.0]:
    Q_scaled = q_scale * Q
    result = system.control.design_lqr(A, B, Q_scaled, R, system_type='continuous')
    
    # Check performance
    K = result['gain']
    closed_loop_poles = result['closed_loop_eigenvalues']
    print(f"q_scale={q_scale}: poles = {closed_loop_poles}")
```

### Pattern 6: Pre-Flight Checks

```python
# Before designing controllers, verify system properties
A, B = system.linearize(x_eq, u_eq)
C = np.array([[1, 0]])

# 1. Check stability of open-loop
stability = system.analysis.stability(A, system_type='continuous')
print(f"Open-loop stable: {stability['is_stable']}")

# 2. Check controllability (required for LQR)
ctrl = system.analysis.controllability(A, B)
if not ctrl['is_controllable']:
    print("WARNING: System not controllable - LQR may fail")
    print(f"Uncontrollable modes: {ctrl['uncontrollable_modes']}")

# 3. Check observability (required for Kalman)
obs = system.analysis.observability(A, C)
if not obs['is_observable']:
    print("WARNING: System not observable - Kalman filter may fail")
    print(f"Unobservable modes: {obs['unobservable_modes']}")

# 4. Proceed with design only if conditions met
if ctrl['is_controllable'] and obs['is_observable']:
    lqg = system.control.design_lqg(A, B, C, Q, R, Q_proc, R_meas)
    print("✓ LQG design successful")
```

### Pattern 7: Multi-Backend Support

```python
import torch
import jax.numpy as jnp

# NumPy backend (default)
system_np = Pendulum()
system_np.set_default_backend('numpy')
result_np = system_np.control.design_lqr(A_np, B_np, Q_np, R_np)

# PyTorch backend
system_torch = Pendulum()
system_torch.set_default_backend('torch')
system_torch.set_default_device('cuda:0')
A_torch = torch.tensor(A_np, device='cuda:0')
B_torch = torch.tensor(B_np, device='cuda:0')
result_torch = system_torch.control.design_lqr(A_torch, B_torch, Q_torch, R_torch)

# Result is on GPU
assert result_torch['gain'].device.type == 'cuda'

# JAX backend
system_jax = Pendulum()
system_jax.set_default_backend('jax')
A_jax = jnp.array(A_np)
B_jax = jnp.array(B_np)
result_jax = system_jax.control.design_lqr(A_jax, B_jax, Q_jax, R_jax)
```

---

## Troubleshooting

### Issue: LQR design fails with "No stabilizing solution"

**Possible Causes:**
- System not controllable
- (Q, A) not detectable
- Ill-conditioned matrices

**Solutions:**
```python
# 1. Check controllability
ctrl = system.analysis.controllability(A, B)
if not ctrl['is_controllable']:
    print(f"System not controllable. Rank: {ctrl['rank']}/{A.shape[0]}")
    print(f"Uncontrollable modes: {ctrl['uncontrollable_modes']}")

# 2. Add regularization to Q
Q_regularized = Q + 1e-6 * np.eye(nx)

# 3. Check (Q, A) detectability
# All uncontrollable modes should have Re(λ) < 0 (continuous)
# or |λ| < 1 (discrete)
```

### Issue: Kalman filter design fails

**Possible Causes:**
- System not observable
- (A, Q^(1/2)) not controllable
- Singular measurement noise R

**Solutions:**
```python
# 1. Check observability
obs = system.analysis.observability(A, C)
if not obs['is_observable']:
    print(f"System not observable. Rank: {obs['rank']}/{A.shape[0]}")
    
    # Try adding measurements
    C_full = np.eye(nx)
    obs_full = system.analysis.observability(A, C_full)
    print(f"With full state measurement: {obs_full['is_observable']}")

# 2. Ensure R is positive definite
R_regularized = R + 1e-6 * np.eye(ny)

# 3. Check process noise controllability
# All unobservable modes should be damped by process noise
```

### Issue: Closed-loop unstable despite LQR design

**Possible Causes:**
- Wrong system_type (continuous vs discrete)
- Numerical errors in linearization
- Equilibrium not actually an equilibrium

**Solutions:**
```python
# 1. Verify equilibrium
f_eq = system(x_eq, u_eq)
print(f"f(x_eq, u_eq) = {f_eq}")  # Should be near zero
assert np.allclose(f_eq, 0, atol=1e-6), "Not an equilibrium!"

# 2. Check linearization accuracy
dx_linear = A @ (x - x_eq) + B @ (u - u_eq)
dx_nonlinear = system(x, u)
error = np.linalg.norm(dx_linear - dx_nonlinear)
print(f"Linearization error: {error}")

# 3. Verify system_type matches system
if isinstance(system, ContinuousSystemBase):
    system_type = 'continuous'
elif isinstance(system, DiscreteSystemBase):
    system_type = 'discrete'

result = system.control.design_lqr(A, B, Q, R, system_type=system_type)
```

### Issue: LQG eigenvalues don't match individual designs

**This is expected!** 
- LQG closed-loop has 2*nx eigenvalues
- Should be: eig(A-BK) ∪ eig(A-LC)

```python
# Verify separation
lqr = system.control.design_lqr(A, B, Q, R)
kalman = system.control.design_kalman(A, C, Q_proc, R_meas)
lqg = system.control.design_lqg(A, B, C, Q, R, Q_proc, R_meas)

# Combined eigenvalues
combined = np.concatenate([
    lqr['closed_loop_eigenvalues'],
    kalman['observer_eigenvalues']
])

# Should match LQG eigenvalues (up to ordering)
lqg_combined = np.concatenate([
    lqg['controller_eigenvalues'],
    lqg['estimator_eigenvalues']
])

assert np.allclose(sorted(combined), sorted(lqg_combined))
```

---

## Direct Function Usage (Advanced)

For cases where you need to use control functions without a system object:

```python
from src.control.classical_control_functions import (
    design_lqr,
    design_kalman_filter,
    design_lqg,
    analyze_stability,
    analyze_controllability,
    analyze_observability
)
import numpy as np

# Define matrices directly
A = np.array([[0, 1], [-2, -3]])
B = np.array([[0], [1]])
C = np.array([[1, 0]])
Q = np.diag([10, 1])
R = np.array([[0.1]])

# Call functions directly
lqr_result = design_lqr(A, B, Q, R, system_type='continuous', backend='numpy')
K = lqr_result['gain']

kalman_result = design_kalman_filter(
    A, C,
    Q=0.01*np.eye(2),
    R=np.array([[0.1]]),
    system_type='continuous',
    backend='numpy'
)
L = kalman_result['gain']

# Analysis
stability = analyze_stability(A, system_type='continuous', backend='numpy')
ctrl = analyze_controllability(A, B, backend='numpy')
obs = analyze_observability(A, C, backend='numpy')
```

---

## Testing Your Controller

```python
def test_lqr_design():
    """Test LQR controller design and stability."""
    # Create system
    system = Pendulum()
    x_eq = np.array([np.pi, 0])
    u_eq = np.zeros(1)
    
    # Linearize
    A, B = system.linearize(x_eq, u_eq)
    
    # Design LQR
    Q = np.diag([10, 1])
    R = np.array([[0.1]])
    result = system.control.design_lqr(A, B, Q, R, system_type='continuous')
    
    # Test 1: Gain has correct shape
    K = result['gain']
    assert K.shape == (1, 2), f"Gain shape incorrect: {K.shape}"
    
    # Test 2: Closed-loop is stable
    eigs = result['closed_loop_eigenvalues']
    assert np.all(np.real(eigs) < 0), "Closed-loop unstable!"
    
    # Test 3: P is positive definite
    P = result['cost_to_go']
    assert np.all(np.linalg.eigvals(P) > 0), "P not positive definite!"
    
    # Test 4: P satisfies Riccati equation
    # A'P + PA - PBR^{-1}B'P + Q = 0
    riccati_error = A.T @ P + P @ A - P @ B @ np.linalg.solve(R, B.T @ P) + Q
    assert np.allclose(riccati_error, 0, atol=1e-6), "Riccati not satisfied!"
    
    print("✓ All LQR tests passed")

test_lqr_design()
```

---

## References

- **Architecture:** See `Control_Framework_Architecture.md` for complete details
- **Type Definitions:** See `Type_System_Architecture.md` for TypedDict documentation
- **Pure Functions:** See `src/control/classical_control_functions.py`
- **Wrappers:** See `src/control/control_synthesis.py` and `src/control/system_analysis.py`
