# ControlDESymulation

> **Control Theory, ODE/SDE Modeling, and Synthetic Data Generation Through Symbolic Dynamical System Specification**

A Python library for constructing, analyzing, and simulating nonlinear dynamical systems using symbolic mathematics with multi-backend numerical execution. I developed it because I couldn't find an easy-to-use, reproducible, modular library for simulating simple physical systems in terms of symbolic variables. The main goal of this library is to enable realistic synthetic data generation for nonlinear dynamical systems, enable nonlinear state-space control theory research, build a framework for defining physical reinforcement learning environments in terms of state-space models, and allowing for reproducible specification of physical model systems for use in verifiable machine learning and safe/constrained RL contexts.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: AGPLv3](https://img.shields.io/badge/License-AGPLv3-yellow.svg)](https://opensource.org/license/agpl-v3)
![Experimental](https://img.shields.io/badge/status-experimental-orange?logo=beaker)

---

## Key Features

- TODO: Re-do section

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
    
    def define_system(self, m_val=0.15, l_val=0.5, beta_val=0.1, g_val=9.81):
        """Define pendulum dynamics symbolically"""

        # Create symbols
        m, l, beta, g = sp.symbols('m l beta g', real=True, positive=True)
        theta_sym, theta_dot_sym = sp.symbols('theta theta_dot', real=True)
        u = sp.symbols('u', real=True)
        
        # Dynamics
        theta_ddot = -(g/l)*sp.sin(theta_sym) - (beta/m)*theta_dot_sym + u/(m*l**2)
        
        # Assign
        self.state_vars = [theta_sym, theta_dot_sym]
        self.control_vars = [u]
        self._f_sym = sp.Matrix([theta_ddot])
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
from src.systems.base import DiscreteTimeSystem
from src.controllers.control_designer import ControlDesigner
from src.visualization.trajectory_plotter import TrajectoryPlotter

# 1. Create continuous-time system and add equilibria
continuous_pendulum = SymbolicPendulum(
        m_val=1.0,
        l_val=0.5,
        beta_val=0.1,
        g_val=9.81
    )
    
# Add equilibria after creation
continuous_pendulum.add_equilibrium(
    'downward',
    x_eq=np.array([0.0, 0.0]),
    u_eq=np.array([0.0]),
    verify=True
)

continuous_pendulum.add_equilibrium(
    'upright',
    x_eq=np.array([np.pi, 0.0]),
    u_eq=np.array([0.0]),
    verify=True
)

# TODO: method needs to be added
continuous_pendulum.set_default_equilibrium('downward')

# 2. Discretize for simulation
# TODO: update once API is fully refactored
dt = 0.01
discrete_pendulum = DiscreteTimeSystem(
    continuous_pendulum, 
    dt=dt,
    method='euler'
)

# 3. Design LQR controller
# TODO: update once fully refactored
# current vision is to hand continuous-time
# and discrete-time systems to the same interface
Q = np.diag([10.0, 1.0])  # State cost
R = np.array([[0.1]])      # Control cost
continuous_designer = ControlDesigner(continuous_pendulum) # continuous time
Kc, Sc = continuous_designer.lqr_control(Q, R) # continuous system gain
discrete_designer = ControlDesigner(discrete_pendulum) # discrete system
Kd, Sd = discrete_designer.lqr_control(Q, R) # discrete system gain

# 4. Simulate closed-loop system (simulation only for discrete systems)
x0 = torch.tensor([0.5, 0.0])  # Initial state: 0.5 rad, 0 rad/s
horizon = 1000

controller = lambda x: torch.tensor(Kd @ (x - discrete_pendulum.default_equilibrium_state).numpy() + discrete_pendulum.default_equilibrium_control.numpy())
trajectory = discrete_pendulum.simulate(x0, controller=controller, horizon=horizon)

# 5. Visualize results
# TODO: update after full refactoring
plotter = TrajectoryPlotter(discrete_pendulum)
closed_loop_plotly_trajectory = TrajectoryPlotter.plot_2d_trajectory(
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

    def define_system(self, param1_val= 1.0, param2_val = 0.5):
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

- TODO: elaborate on backend functionality
The library can automatically detect backend from the input type:

```python

system = MySystem()

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
discrete_system = DiscreteTimeSystem(
    system,
    dt=0.01,
    method='euler'
)

# Now system computes x[k+1] = f_discrete(x[k], u[k])
x_next = discrete_system(x_current, u_current)
```

- TODO: Elaborate on extremely extensive refactoring

### 4. Control Design

```python

discrete_control_designer = ControlDesigner(discrete_system)

# LQR (state feedback)
Q = np.eye(nx) * 10.0  # State cost
R = np.eye(nu) * 0.1   # Control cost
K, S = discrete_control_designer.lqr(Q, R)

# Kalman Filter (state estimation)
Q_process = np.eye(nx) * 0.01      # Process noise
R_measurement = np.eye(ny) * 0.1   # Measurement noise
L = discrete_control_designer.kalman_gain(Q_process, R_measurement)

# LQG (output feedback)
K, L = discrete_control_designer.lqg_control(Q, R, Q_process, R_measurement)
```

### 5. Simulation

```python
# Simulation with pre-computed controls
u_sequence = torch.zeros(horizon, nu)
trajectory = discrete_system.simulate(x0, controller=u_sequence)

# Simulation with feedback controller
controller = lambda x: -K @ x
trajectory = discrete_system.simulate(x0, controller=controller, horizon=1000)

# Simulation with observer (output feedback)
trajectory = discrete_system.simulate(
    x0, 
    controller=controller,
    observer=observer,
    horizon=1000
)
```

---

## Built-in Systems

The library includes several pre-defined mechanical and aerial systems:
- TODO: Elaborate and fix

### Mechanical Systems

### Aerial Systems

### Linear Systems

### Stochastic Systems

---

## Visualization

### Trajectory Plots


### 3D Visualization

---

## Advanced Features

### Higher-Order Systems

The library automatically handles arbitrary-order systems:

```python
class SecondOrderSystem(SymbolicDynamicalSystem):

    def define_system(self, k_val=10.0, c_val=0.5):
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

### Custom Output Functions

```python
class CustomOutputSystem(SymbolicDynamicalSystem):

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

### Printing System Information


---

## Use Cases

### 1. Prototyping Control Algorithms

### 2. Neural Network Integration

### 3. Sim-to-Real Transfer

### 4. Research: Learning Lyapunov Functions

---

## Development

### Contributing and Future Work

Currently not open to contributions, though may change after Phase 4 and once I learn how open-source development works

#### Phase 1 (Current):
- Extensive class hierarchy refactoring
    - DiscreteSystemBase (current)
    - ContinuousSystemBase
    - DiscreteSymbolicSystem(DiscreteSystemBase)
    - ContinuousSymbolicSystem(ContinuousSystemBase)
    - DiscreteStochasticSystem(DiscreteSymbolicSystem)
    - ContinuousStochasticSystem(ContinuousSymbolicSystem)
    - DiscretizationWrapper (wraps either type of continuous system)

#### Phase 2 (Current):
- Refactoring of DiscreteTimeSystem
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
This project is licensed under the GNU Affero General Public License v3.0 (or later). The licence file is in the repository here: [LICENSE](LICENSE)

If you run a modified version of this software on a network, you must provide the complete corresponding source code to users of that service, as required by the AGPL.

Using this software internally or as an unmodified dependency does not require you to release your own source code. Only modifications to this software that are made available over a network must be shared, as required by the AGPL.

### License Rationale
Why this project uses the AGPL

This project is licensed under the GNU Affero General Public License v3.0 (AGPL) by design.

My primary goal is wide adoption by researchers, academic institutions, and non-SaaS organizations, while ensuring that improvements made for public-facing use remain available to the community.

The AGPL helps achieve this balance by:

- Allowing anyone to use the software freely, including for internal, commercial, and academic purposes
- Encouraging collaboration and reproducibility, especially in research and applied settings
- Preventing the software from being used in public services without contributing improvements back

In practice, this means:
- If you use the software as-is, there are no additional obligations
- If you modify the software and run it as a public or networked service, those modifications must be shared under the same license

This ensures that advances made using this project, especially in publicly accessible systems, remain available for others to learn from, build upon, and verify.

### License FAQ

This section answers common questions from organizations and researchers who are not software specialists.

- Can I use this software inside my organization?
    - Yes, you can:
        - Use it internally
        - Run it on your own servers
        - Use it for research, analysis, or operations
    - You do not need to release your own code unless you modify this software itself and make those modifications available over a network.
    
- Can I use this software for commercial purposes?
    - Yes, commercial use is explicitly allowed. The AGPL does not restrict charging for services, consulting, or results produced using the software.
    
- Do I have to open-source my entire system?
    - No, you only need to share
        - Modifications you make to this software, and
        - Only if those modifications are used in a network-accessible service
    - Your own proprietary systems, data, models, and workflows remain yours.

- What counts as a “modification”?
    - Examples of modifications:
        - Changing how this software works internally
        - Adding features directly inside this codebase
        - Fixing bugs in this project’s source code
    - Examples that are not modifications:
        - Writing scripts that use the software
        - Importing it as a dependency
        - Running it unchanged
        - Wrapping it with external tools

- What if we don’t change the code at all?
    - Then you have no obligation to publish anything. Simply using the software, even in a service, does not trigger AGPL requirements if the software itself is unmodified.

- Why does this license mention network use?
    - Many modern tools are used as services rather than distributed as files.
    - The AGPL ensures that improvements made for publicly accessible services are shared just like improvements made to distributed software. This supports fairness, transparency, and reproducibility.

- Is this license common in academia?
    - Yes, AGPL is widely used and accepted in:
        - Academic research software
        - Public-sector tools
        - Infrastructure and data platforms
    - It aligns well with open science and reproducible research practices.

- We’re unsure if our use case triggers AGPL. What should we do?
    - If you are:
        - Using the software unchanged → you’re fine
        - Unsure whether a modification applies → contact us or consult your legal/compliance team
            - I aim to be reasonable and collaborative, not adversarial.

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