# ControlDESymulation

> **Control Theory, ODE/SDE Modeling, and Synthetic Data Generation Through Symbolic Dynamical System Specification**

A Python library for constructing, analyzing, and simulating nonlinear dynamical systems using symbolic mathematics with multi-backend numerical execution. I developed it because I couldn't find an easy-to-use, reproducible, modular library for simulating simple physical systems in terms of symbolic variables. The main goal of this library is to enable realistic synthetic data generation for nonlinear dynamical systems, enable nonlinear state-space control theory research, build a framework for defining physical reinforcement learning environments in terms of state-space models, and allowing for reproducible specification of physical model systems for use in verifiable machine learning and safe/constrained RL contexts.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Key Features

- **Symbolic System Definition**: Define dynamics using SymPy's symbolic mathematics
- **Multi-Backend Execution**: Seamlessly execute on NumPy, PyTorch, or JAX
- **Support for Nonlinear State-Space Control**: LQR, LQG, Kalman filtering, and observers
- **Flexible Discretization**: Multiple integration schemes (Euler, Midpoint, RK4)
- **Rich Visualization**: Interactive Plotly plots with automatic layout
- **High Performance**: JIT compilation (JAX), GPU acceleration (PyTorch), symbolic optimization

---

## Why ControlDESymulation?

Most control libraries force you to choose: symbolic elegance (MATLAB Symbolic Toolbox) *or* numerical efficiency (SciPy, CasADi). ControlDESymulation gives you **both**:

1. **Write once, run anywhere**: Define your system symbolically, execute on any backend
2. **No code rewriting**: Switch between NumPy, PyTorch, and JAX without changing system definitions  
3. **Gradient-aware by design**: Automatic differentiation support for learned controllers
4. **Research-to-production**: Prototype in NumPy, scale with JAX, integrate with PyTorch neural networks

### Example: Define Once, Execute Everywhere

```python
import sympy as sp
from src.systems.base import SymbolicDynamicalSystem

class Pendulum(SymbolicDynamicalSystem):
    def __init__(self, m=0.15, l=0.5, beta=0.1, g=9.81):
        super().__init__(m, l, beta, g)
    
    def define_system(self, m_val, l_val, beta_val, g_val):
        """Define pendulum dynamics symbolically"""

        # Create symbols
        m, l, beta, g = sp.symbols('m l beta g', real=True, positive=True)
        θ, θ_dot = sp.symbols('theta theta_dot', real=True)
        u = sp.symbols('u', real=True)
        
        # Dynamics
        θ_ddot = -(g/l)*sp.sin(θ) - (beta/m)*θ_dot + u/(m*l**2)
        
        # Assign
        self.state_vars = [θ, θ_dot]
        self.control_vars = [u]
        self._f_sym = sp.Matrix([θ_ddot])
        self.parameters = {m: m_val, l: l_val, beta: beta_val, g: g_val}
        self.order = 2

pendulum = Pendulum()

# Execute on NumPy (CPU, prototyping)
import numpy as np
x_np = np.array([0.1, 0.0])
u_np = np.array([0.0])
dx_np = pendulum(x_np, u_np)  # Returns NumPy array

# Execute on PyTorch (GPU, neural integration)
import torch
x_torch = torch.tensor([0.1, 0.0], device='cuda')
u_torch = torch.tensor([0.0], device='cuda')
dx_torch = pendulum(x_torch, u_torch)  # Returns PyTorch tensor on GPU

# Execute on JAX (TPU, JIT compilation)
import jax.numpy as jnp
x_jax = jnp.array([0.1, 0.0])
u_jax = jnp.array([0.0])
dx_jax = pendulum(x_jax, u_jax)  # Returns JAX array, JIT-compiled
```

The **same symbolic definition** works with all three backends automatically.

---

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/gbenezer/ControlDESymulation.git
cd ControlDESymulation

# Create conda environment
conda create --name control_ode python=3.11
conda activate control_de

# Install dependencies
pip install -r requirements.txt

# Optional: Install JAX with GPU support
pip install jax[cuda12]  # For CUDA 12.x
```

### Basic Usage

```python
import torch
import numpy as np
from src.systems.builtin import SymbolicPendulum
from src.systems.base import GenericDiscreteTimeSystem, IntegrationMethod

# 1. Create continuous-time system
pendulum = SymbolicPendulum(m=0.15, l=0.5, beta=0.1, g=9.81)

# 2. Discretize for simulation
dt = 0.01
system = GenericDiscreteTimeSystem(
    pendulum, 
    dt=dt,
    integration_method=IntegrationMethod.RK4
)

# 3. Design LQR controller
Q = np.diag([10.0, 1.0])  # State cost
R = np.array([[0.1]])      # Control cost
K, S = system.dlqr_control(Q, R)

# 4. Simulate closed-loop system
x0 = torch.tensor([0.5, 0.0])  # Initial state: 0.5 rad, 0 rad/s
horizon = 1000

controller = lambda x: torch.tensor(K @ (x - system.x_equilibrium).numpy() + system.u_equilibrium.numpy())
trajectory = system.simulate(x0, controller=controller, horizon=horizon)

# 5. Visualize results
system.plot_trajectory(
    trajectory,
    state_names=['θ (rad)', 'θ̇ (rad/s)'],
    title='Pendulum Stabilization with LQR'
)
```

---

## Core Concepts

### 1. Symbolic System Definition

Define your dynamical system using SymPy:

```python
class MySystem(SymbolicDynamicalSystem):

    def __init__(self, param1 = 1.0, param2 = 0.5):
        super().__init__(param1, param2)

    def define_system(self, param1_val, param2_val):
        # Define symbolic variables
        x1, x2 = sp.symbols('x1 x2')
        u = sp.symbols('u')
        param1, param2 = sp.symbols('param1 param2')
        
        # Define dynamics: dx/dt = f(x, u)
        dx1_dt = x2
        dx2_dt = -param1 * x1 + param2 * u
        
        # Set system properties
        self.state_vars = [x1, x2]
        self.control_vars = [u]
        self._f_sym = sp.Matrix([dx1_dt, dx2_dt])
        self.parameters = {param1: param1_val, param2: param2_val}
        self.order = 1  # First-order system
```

### 2. Multi-Backend Execution

The library automatically detects the backend from input types:

```python
# NumPy backend (CPU)
dx_numpy = system(np.array([1.0, 0.0]), np.array([0.5]))

# PyTorch backend (CPU/GPU)
dx_torch = system(torch.tensor([1.0, 0.0]), torch.tensor([0.5]))

# JAX backend (CPU/GPU/TPU, JIT-compiled)
dx_jax = system(jnp.array([1.0, 0.0]), jnp.array([0.5]))
```

### 3. Discretization

Convert continuous-time systems to discrete-time:

```python
from src.systems.base import GenericDiscreteTimeSystem, IntegrationMethod

discrete_system = GenericDiscreteTimeSystem(
    continuous_system,
    dt=0.01,
    integration_method=IntegrationMethod.RK4,      # For velocities
    position_integration=IntegrationMethod.MidPoint  # For positions (2nd order)
)

# Now system computes x[k+1] = f_discrete(x[k], u[k])
x_next = discrete_system(x_current, u_current)
```

Available integration methods:
- `ExplicitEuler`: Fast, first-order accuracy
- `MidPoint`: Moderate speed, second-order accuracy  
- `RK4`: Slower, fourth-order accuracy (recommended)

Planned to be extended with support for scipy.integrate.solve_ivp, Diffrax, and torchdiffeq

### 4. Control Design

```python
# LQR (state feedback)
Q = np.eye(nx) * 10.0  # State cost
R = np.eye(nu) * 0.1   # Control cost
K, S = system.dlqr_control(Q, R)

# Kalman Filter (state estimation)
Q_process = np.eye(nx) * 0.01      # Process noise
R_measurement = np.eye(ny) * 0.1   # Measurement noise
L = system.discrete_kalman_gain(Q_process, R_measurement)

# LQG (output feedback)
K, L = system.dlqg_control(Q, R, Q_process, R_measurement)
```

### 5. Simulation

```python
# Simulation with pre-computed controls
u_sequence = torch.zeros(horizon, nu)
trajectory = system.simulate(x0, controller=u_sequence)

# Simulation with feedback controller
controller = lambda x: -K @ x
trajectory = system.simulate(x0, controller=controller, horizon=1000)

# Simulation with observer (output feedback)
trajectory = system.simulate(
    x0, 
    controller=controller,
    observer=observer,
    horizon=1000
)
```

---

## Built-in Systems

The library includes several pre-defined mechanical and aerial systems:

### Mechanical Systems

```python
from src.systems.builtin import (
    SymbolicPendulum,        # Simple pendulum
    CartPole,        # Inverted pendulum on cart
    Manipulator2Link, # Two-link planar robotic manipulator model
    PathTracking, # Simplistic model of a car on a circular track
)

# Example: CartPole
cartpole = CartPole(
    m_cart: float = 1.0,
    m_pole: float = 0.1,
    length: float = 0.5,
    gravity: float = 9.81,
    friction: float = 0.1,
)
```

### Aerial Systems

```python
from src.systems.builtin import (
    SymbolicQuadrotor2D,     # Planar quadrotor
    SymbolicQuadrotor2DLidar, # Planar quadrotor where only output are 4 LIDAR readings
    PVTOL, # Planar quadrotor, velocities relative to body frame rather than world frame
)

# Example: 2D Quadrotor
quad = SymbolicQuadrotor2D(
    m=0.5,      # Mass
    Ixx=0.01,   # Moment of inertia
    g=9.81      # Gravity
)
```

All systems support:
- Automatic linearization
- LQR/LQG control design
- Multi-backend execution
- Rich visualization

---

## Visualization

### Trajectory Plots

```python
system.plot_trajectory(
    trajectory,
    state_names=['x', 'y', 'θ', 'ẋ', 'ẏ', 'θ̇'],
    control_sequence=controls,
    control_names=['F_left', 'F_right'],
    title='Quadrotor Trajectory',
    colorway='Plotly',  # or 'D3', 'Set1', etc.
    save_html='trajectory.html'
)
```

Features:
- **Adaptive layout**: Automatically arranges subplots based on number of states
- **Interactive**: Zoom, pan, hover for values
- **Batch support**: Compare multiple trajectories
- **Compact mode**: For systems with many state variables

### 3D Visualization

```python
# 3D trajectory with time coloring (single trajectory only)
# otherwise analogous to plot_phase_portrait_3d
# (for theoretical 3D Quadrotor system)
system.plot_trajectory_3d(
    trajectory,
    state_indices=(0, 1, 2),
    state_names=('x', 'y', 'z'),
    title='Quadrotor 3D Path'
)

# Phase portraits
system.plot_phase_portrait_2d(trajectory, state_indices=(0, 1))
system.plot_phase_portrait_3d(trajectory, state_indices=(0, 2, 4))
```

---

## Advanced Features

### Higher-Order Systems

The library automatically handles arbitrary-order systems:

```python
class SecondOrderSystem(SymbolicDynamicalSystem):

    def __init__(self, k=10.0, c=0.5):
        super().__init__(k, c)

    def define_system(self, k_val, c_val):
        q, q_dot = sp.symbols('q q_dot')
        u = sp.symbols('u')
        k, c = sp.symbols('k c', real=True, positive=True)
        
        # Define acceleration: q̈ = f(q, q̇, u)
        q_ddot = -k*q - c*q_dot + u
        
        self.state_vars = [q, q_dot]
        self.control_vars = [u]
        self.parameters = {k: k_val, c: c_val}
        self._f_sym = sp.Matrix([q_ddot])
        self.order = 2  # Second-order
```

### Linearization

```python
# Symbolic linearization
A_sym, B_sym = system.linearized_dynamics_symbolic(x_eq, u_eq)

# Numerical linearization (any backend)
A, B = system.linearized_dynamics(x_torch, u_torch)

# Verify against autodiff
results = system.verify_jacobians(x, u, tol=1e-6)
print(f"A matches: {results['A_match']}, error: {results['A_error']:.2e}")
print(f"B matches: {results['B_match']}, error: {results['B_error']:.2e}")
```

### Performance Diagnostics

```python
# Check backend installation and capabilities
from src.systems.base.backend_utils import print_installation_summary
print_installation_summary()

# Output:
# ======================================================================
# Backend Installation Summary
# ======================================================================
# 
# 📊 NumPy
#   ✓ Installed: 2.0.1
#   Device: cpu
# 
# 🔥 PyTorch
#   ✓ Installed: 2.3.0
#   CUDA available: True
#   CUDA version: 12.1
#   GPU count: 1
#     [0] NVIDIA RTX 4090
# 
# ⚡ JAX
#   ✓ Installed: 0.4.30
#   GPU available: True
#   GPU count: 1
# ======================================================================
```

### Custom Output Functions

```python
class CustomOutputSystem(SymbolicDynamicalSystem):
    def __init__(self):
        super().__init__()

    def define_system(self):
        x1, x2 = sp.symbols('x1 x2', real=True)
        u = sp.symbols('u', real=True)
        
        self.state_vars = [x1, x2]
        self.control_vars = [u]
        self._f_sym = sp.Matrix([x2, -x1])
        self.order = 1
        
        # Custom output: y = h(x)
        # y[0] = x1
        # y[1] = x1^2 + x2^2
        self._h_sym = sp.Matrix([x1, x1**2 + x2**2])
        
        # Name the outputs for clarity
        self.output_vars = [sp.Symbol('y1'), sp.Symbol('y2')]
        
        # No parameters in this system
        self.parameters = {}

# Evaluate output
y = system.h(x)  # y = h(x)

# Linearize output
C = system.linearized_observation(x)  # C = dh/dx
```

---

## Testing & Validation

Run the test suite:

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/symbolic_dynamical_system_test.py

# Run with coverage
pytest --cov=src tests/
```

---

## Documentation

### System Properties

Every `SymbolicDynamicalSystem` provides:

```python
system.nx          # Number of states
system.nu          # Number of controls
system.ny          # Number of outputs
system.nq          # Number of generalized coordinates
system.order       # System order (1, 2, 3, ...)

system.x_equilibrium  # Equilibrium state
system.u_equilibrium  # Equilibrium control

system.state_vars     # List of SymPy symbols
system.control_vars   # List of SymPy symbols
system.parameters     # Dict of parameters
```

### Printing System Information

```python
# Print symbolic equations
system.print_equations(simplify=True)

# Comprehensive system info
discrete_system.print_info(
    include_equations=True,
    include_linearization=True
)

# Quick summary
print(discrete_system.summary())
```

---

## Use Cases

### 1. Prototyping Control Algorithms

```python
# Quickly test different LQR weights
for q_theta in [1, 10, 100]:
    Q = np.diag([q_theta, 1.0])
    K, _ = system.dlqr_control(Q, R)
    traj = system.simulate(x0, lambda x: K @ x, horizon=1000)
    system.plot_trajectory(traj, title=f'Q_θ = {q_theta}')
```

### 2. Neural Network Integration

```python
import torch.nn as nn

class NeuralController(nn.Module):
    def __init__(self, nx, nu):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(nx, 64),
            nn.Tanh(),
            nn.Linear(64, nu)
        )
    
    def forward(self, x):
        return self.net(x)

# Train with gradient-based optimization
controller = NeuralController(system.nx, system.nu)
optimizer = torch.optim.Adam(controller.parameters())

for epoch in range(num_epochs):
    trajectory = system.simulate(x0, controller, horizon=100)
    loss = compute_cost(trajectory)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### 3. Sim-to-Real Transfer

```python
# High-fidelity simulation (JAX, RK4)
jax_system = GenericDiscreteTimeSystem(
    dynamics, 
    dt=0.001,
    integration_method=IntegrationMethod.RK4
)

# Real-time controller (NumPy, fast Euler)
real_time_system = GenericDiscreteTimeSystem(
    dynamics,
    dt=0.01,
    integration_method=IntegrationMethod.ExplicitEuler
)

# Same controller works on both!
K, L = jax_system.dlqg_control(Q, R, Q_proc, R_meas)
```

### 4. Research: Learning Lyapunov Functions

```python
class NeuralLyapunov(nn.Module):
    def __init__(self, nx):
        super().__init__()
        self.V_net = nn.Sequential(...)
    
    def forward(self, x):
        return self.V_net(x)

# Verify Lyapunov conditions using symbolic Jacobians
V_network = NeuralLyapunov(system.nx)

x = torch.randn(batch_size, system.nx, requires_grad=True)
u = controller(x)

V = V_network(x)
dV_dx = torch.autograd.grad(V.sum(), x, create_graph=True)[0]
dx = system(x, u)  # Get dynamics

V_dot = (dV_dx * dx).sum(dim=1)  # dV/dt = (dV/dx) · f(x,u)

# Train: V_dot < 0 for stability
loss = torch.relu(V_dot + margin).mean()
```

---

## Development

<!-- ### Current Project Structure

```
ControlODESymulation/
├── src/
│   ├── systems/
│   │   ├── base/              # Core abstractions
│   │   │   ├── symbolic_dynamical_system.py
│   │   │   ├── generic_discrete_time_system.py
│   │   │   ├── codegen_utils.py
│   │   │   ├── array_backend.py
│   │   │   └── backend_utils.py
│   │   └── builtin/           # Pre-defined systems
│   │       ├── mechanical_systems.py
│   │       ├── abstract_symbolic_systems.py
│   │       └── aerial_systems.py
│   ├── controllers/           # Control algorithms
│   ├── observers/             # State estimators
│   └── lyapunov_functions/    # Legacy functions (need to be refactored)
├── tests/                     # Test suite
├── example_notebooks/         # Jupyter tutorials
└── requirements.txt
``` -->

### Contributing and Future Work

Currently not open to contributions, though may change after Phase 4 and once I learn how open-source development works

#### Phase 2 (Current):
- Refactoring of DiscreteTimeSystem
    - Construct DiscreteSimulator that uses Discretizer to handle trajectory simulation
        - Make sure this can support both autonomous and controlled systems
    - Construct StochasticDiscreteSimulator
    - Construct DiscreteLinearization that caches numerical linearization
    - Construct StochasticDiscreteLinearization
    - Construct unified DiscreteTimeSystem class from the above sub-object classes
        - Should
            - Take in a SymbolicDynamicalSystem or subclass thereof
            - Use that system along with the IntegratorFactory class to instantiate and store an appropriate numerical integrator
            - Use that numerical integrator to instantiate Discretizer, DiscreteSimulator, and DiscreteLinearization classes
            - Facilitate numerical simulation of the system
    - Construct analogous StochasticDiscreteTimeSystem class

#### Phase 3:
- Re-implement plotting utilities
    - Construct TrajectoryPlotGenerator class
        - arrays of 2D Plotly plots of state variables
        - add options to plot control and output variables along with external state estimates from observers
        - evaluate possibility/utility of 
    - Construct PhasePortraitGenerator class for 2D or 3D Plotly phase portrait generation
- Re-implement classical control theory capabilities
    - Construct ControlDesigner class
        - Wrapper class to either SymbolicDynamicalSystem (continuous) or DiscreteTimeSystem
            - Interface/access to traditional/classic nonlinear state-space control utilities (LQR/Kalman/LQG matrices, etc.)
            - Auto-detect time type (continuous vs. discrete)
    - libraries under consideration for backend implementation include control, control-toolbox, pysyscontrol, OpenControl, Kontrol
- Look back and assess if any additional refactoring needs to occur
    - Mainly looking for
        - God objects
        - Other code smells/design flaws
        - Inconsistencies in user-facing API
    - Will be trying to integrate the following tools for assistance
        - Pylint
        - Flake8
        - Coverage.py and/or Pytest-cov
        - Mutmut
        - PyDeps
        - Radon
- Address the warnings being raised by StochasticDynamicalSystems when parameters are used in the diffusion term but not _f_sym or _h_sym

#### Phase 4:
- Integration testing
    - Construct integration tests at interfaces between classes in currently envisioned typical pipelines
        - SymbolicDynamicalSystem / DiscreteTimeSystem / IntegratorFactory
        - StochasticDynamicalSystem / DiscreteTimeSystem / SDEIntegratorFactory
        - SymbolicDynamicalSystem / ControlDesigner
        - SymbolicDynamicalSystem / DiscreteTimeSystem / ControlDesigner
        - SymbolicDynamicalSystem / DiscreteTimeSystem / IntegratorFactory / TrajectoryPlotGenerator
        - StochasticDynamicalSystem / DiscreteTimeSystem / SDEIntegratorFactory / TrajectoryPlotGenerator
        - SymbolicDynamicalSystem / TrajectoryPlotGenerator (maybe)
- Again re-assess need to refactor
- Re-assess module level and function/method level docstrings and adjust as needed
- Re-code built-in system classes and re-assess utility of each
- Add some new systems
    - Industrial control systems
    - Stochastic systems
- Re-code existing and novel Jupyter demo notebooks
- Re-do README file
- Release/Publish

#### Phase 5:
- Start to address some ultimate goals of the library
    - Automatic Gymnasium, PyBullet, and/or Brax environment construction
    - Synthetic data generation and export
    - Integration with neural network verification libraries (VNN-Lib, Auto-Lirpa/CROWN, NFL-Veripy)
    - Re-introduce capabilities for (generalized) Lyapunov function/controller/observer synthesis, design and verification based on [Lyapunov-stable Neural Control for State and Output Feedback: A Novel Formulation](https://proceedings.mlr.press/v235/yang24f.html) and [Certifying Stability of Reinforcement Learning Policies using Generalized Lyapunov Functions](https://arxiv.org/abs/2505.10947v3)
    - Look at do-mpc, CasADi, acados, nMPyC, and GEKKO for optimal/MP control; see what could/should be implemented
    - Add parameter sensitivity analysis or other methods to probe dependencies of dynamical systems
    - Check to see if there's any interest in robust, adaptive, and/or stochastic control theory technique implementation
    - Coupling systems together to make larger composite systems
    - Consider supporting stochastic observation dynamics (diffusion + drift for output variables) if not available in framework

---

## Citation

If you use this library in your research, please cite:

```bibtex
@software{benezer2025controlde,
  author = {Benezer, Gil},
  title = {ControlDESymulation: Symbolic Dynamical System Specification for Modern Scientific Computing},
  year = {2025},
  url = {https://github.com/gbenezer/ControlDESymulation}
}
```

---

## Acknowledgments

This library builds on:
- **SymPy**: Symbolic mathematics
- **NumPy/SciPy**: Numerical computing
    - **DiffEqPy** for advanced ODE/SDE integration with NumPy
- **PyTorch**: Automatic differentiation and GPU acceleration
    - **torchdiffeq** for PyTorch compatible ODE integration
    - **TorchSDE** for PyTorch compatible SDE integration
- **JAX**: JIT compilation and functional programming
    - **Diffrax** for Jax compatible ODE/SDE integration
- **Plotly**: Interactive visualization
- **python-control**: Control theory algorithms

This library was inspired by and built from a class project for CS 7268, Verifiable Machine Learning, taught by Professor [Michael Everett](https://mfe7.github.io/) at Northeastern University in Fall 2025 (found [here](https://github.com/gbenezer/Lyapunov_Stable_NN_Controllers_Custom_Dynamics))

---

## License

MIT License - see [LICENSE](LICENSE) file for details.

---

## Links

- **GitHub**: [github.com/gbenezer/ControlDESymulation](https://github.com/gbenezer/ControlDESymulation)
- **Issues**: [Report bugs or request features](https://github.com/gbenezer/ControlDESymulation/issues)
- **Examples**: See `example_notebooks/` for Jupyter tutorials (once re-done)

---

## Contact

**Gil Benezer**
- Email: gil (dot) benezer (at) gmail (dot) com
- GitHub: [@gbenezer](https://github.com/gbenezer)

---

**Built for control theorists, machine learning researchers, and anyone who needs to model physical time-varying systems**