# ControlDESymulation

> **Symbolic Dynamical Systems for Control Theory, Machine Learning, and Scientific Computing**

A Python library for defining, analyzing, and simulating nonlinear dynamical systems using **symbolic mathematics** with **multi-backend numerical execution**. Write your system once in SymPy, then seamlessly execute on NumPy, PyTorch, or JAX without code changes.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![CI](https://img.shields.io/badge/CI-passing-brightgreen)](https://github.com/gbenezer/ControlDESymulation/actions)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

## Why ControlDESymulation?

Most control and dynamics libraries force you to choose between **symbolic elegance** *or* **numerical efficiency**. ControlDESymulation gives you **both**:

### Key Benefits

1. **Write Once, Run Anywhere**: Define systems symbolically, execute on any backend (NumPy/PyTorch/JAX)
2. **No Backend Lock-in**: Switch between CPU, GPU, or TPU without changing your code
3. **Gradient-Aware**: Automatic differentiation support for learned controllers and neural ODEs
4. **Type-Safe**: Comprehensive type hints for better IDE support and fewer bugs
5. **Research to Production**: Prototype in NumPy, scale with JAX, integrate with PyTorch models

### Built For

- **Control theorists**: LQR, MPC, nonlinear control design
- **ML researchers**: Synthetic data generation, RL environments, neural ODEs
- **Roboticists**: Physics-based simulation and controller design
- **Scientists**: Reproducible modeling of physical systems

---

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/gbenezer/ControlDESymulation.git
cd ControlDESymulation

# Install with pip (editable mode)
pip install -e .

# Or with specific backends
pip install -e ".[jax,viz]"     # JAX + visualization
pip install -e ".[all]"         # Everything
```

### 30-Second Example (TODO: update after full refactor)

```python
import sympy as sp
import numpy as np
from src.systems.base import SymbolicDynamicalSystem

class Pendulum(SymbolicDynamicalSystem):
    def define_system(self, m=0.15, l=0.5, beta=0.1, g=9.81):
        # Symbolic variables
        theta, theta_dot = sp.symbols('theta theta_dot', real=True)
        u = sp.symbols('u', real=True)
        
        # Equation of motion
        theta_ddot = -(g/l)*sp.sin(theta) - (beta/m)*theta_dot + u/(m*l**2)
        
        # Define system
        self.state_vars = [theta, theta_dot]
        self.control_vars = [u]
        self._f_sym = sp.Matrix([theta_dot, theta_ddot])

# Create pendulum
pendulum = Pendulum()

# Evaluate dynamics (automatically uses NumPy)
x = np.array([0.1, 0.0])  # [angle, angular_velocity]
u = np.array([0.0])       # [torque]
dx = pendulum(x, u)       # Returns dx/dt

print(f"State derivative: {dx}")  # [angular_vel, angular_accel]
```

### Multi-Backend Execution

**Same definition, different backends:**

```python
# NumPy (CPU, prototyping)
import numpy as np
x_np = np.array([0.1, 0.0])
dx_np = pendulum(x_np, np.array([0.0]))

# PyTorch (GPU, neural networks)
import torch
x_torch = torch.tensor([0.1, 0.0], device='cuda')
dx_torch = pendulum(x_torch, torch.tensor([0.0], device='cuda'))

# JAX (TPU, JIT compilation)
import jax.numpy as jnp
x_jax = jnp.array([0.1, 0.0])
dx_jax = pendulum(x_jax, jnp.array([0.0]))
```

**No code changes required!** The backend is detected automatically.

---

## Features

### Core Capabilities

- **Symbolic System Definition**: Define systems using SymPy's symbolic math
- **Multi-Backend Execution**: NumPy, PyTorch, JAX with automatic backend detection
- **ODE/SDE Support**: Both deterministic and stochastic differential equations
- **Discrete-Time Systems**: Native symbolic discrete systems or discretization of continuous systems
- **Multiple Integrators**: SciPy, torchdiffeq, torchsde, Diffrax, DifferentialEquations.jl
- **Built-in Systems**: Pendulum, cart-pole, quadrotor, and more

### Control & Analysis

- **Linearization**: Symbolic and numerical linearization around equilibria
- **LQR/LQG**: Linear-Quadratic Regulator and Gaussian control
- **State Estimation**: Kalman Filter, Extended Kalman Filter, observers (Under Construction)
- **Discretization**: Multiple methods (Euler, RK4, zero-order hold, etc.) (Under Construction)
- **Trajectory Simulation**: Forward simulation with visualization (Under Construction)

### Advanced Features

- **Type System**: Comprehensive type definitions for better IDE support (Upcoming Refactor)
- **Structured Results**: Named tuples instead of raw arrays (e.g., `LinearizationResult`, `SimulationResult`) (Upcoming Refactor)
- **GPU Acceleration**: Seamless GPU support via PyTorch and JAX
- **Stochastic Systems**: SDEs with Brownian motion and custom noise
- **Modular Design**: Easy to extend and customize

---

## Example: LQR Control of Pendulum (Under Construction)

```python
import numpy as np
from src.systems.builtin import SymbolicPendulum
from src.control import ControlDesigner
from src.visualization import TrajectoryPlotter

# 1. Create system
pendulum = SymbolicPendulum(m_val=1.0, l_val=0.5, beta_val=0.1, g_val=9.81)

# 2. Add equilibrium point
pendulum.add_equilibrium(
    name='upright',
    x_eq=np.array([np.pi, 0.0]),  # Top position, zero velocity
    u_eq=np.array([0.0]),          # Zero torque
    verify=True
)

# 3. Design LQR controller
designer = ControlDesigner(pendulum)
Q = np.diag([10.0, 1.0])  # State cost
R = np.array([[0.1]])      # Control cost
K, S = designer.lqr_control(Q, R, equilibrium='upright')

# 4. Discretize system
discrete_pendulum = pendulum.discretize(dt=0.01, method='rk4')

# 5. Simulate closed-loop
x0 = np.array([np.pi + 0.2, 0.0])  # Start near upright
controller = lambda x: -K @ (x - pendulum.equilibrium['upright'].x_eq)
result = discrete_pendulum.simulate(x0, controller=controller, horizon=500)

# 6. Visualize
plotter = TrajectoryPlotter(discrete_pendulum)
plotter.plot_trajectory(
    result,
    state_names=['θ (rad)', 'θ̇ (rad/s)'],
    title='Pendulum Stabilization with LQR'
)
```

---

## System Types

### Continuous-Time Systems

```python
# Deterministic ODE: dx/dt = f(x, u, t)
class MyContinuousSystem(ContinuousSymbolicSystem):
    def define_system(self):
        # Define symbolic ODE
        pass

# Stochastic SDE: dx = f(x,u)dt + g(x,u)dW
class MyStochasticSystem(ContinuousStochasticSystem):
    def define_system(self):
        # Define drift and diffusion
        pass
```

### Discrete-Time Systems

```python
# Discrete deterministic: x[k+1] = f(x[k], u[k])
class MyDiscreteSystem(DiscreteSymbolicSystem):
    def define_system(self):
        # Define difference equation
        pass

# Discrete stochastic x[k+1] = f(x[k], u[k]) + g(x[k], u[k]) * w[k]
class MyDiscreteSystem(DiscreteStochasticSystem):
    def define_system(self):
        # Define difference equation
        pass

# Or discretize continuous system
discrete_sys = continuous_sys.discretize(dt=0.01, method='rk4')
```

---

## Architecture

### Clean Separation of Concerns

```
src/
├── systems/
│   ├── base/               # Core system classes
│   │   ├── continuous_symbolic_system.py
│   │   ├── continuous_stochastic_system.py
│   │   ├── discrete_symbolic_system.py
│   │   ├── discrete_stochastic_system.py
│   │   ├── discretization/      # Discretization methods
│   │   ├── numerical_integration/  # ODE/SDE solvers
│   │   └── utils/              # Backend management, code generation
│   └── builtin/            # Pre-defined systems
│       ├── mechanical_systems.py  # Pendulum, cart-pole, etc.
│       ├── aerial_systems.py      # Quadrotor, fixed-wing
│       └── stochastic/            # Brownian motion, OU process
├── types/                  # Type definitions (17 modules!)
│   ├── core.py            # StateVector, ControlVector, etc.
│   ├── linearization.py   # LinearizationResult, etc.
│   ├── trajectories.py    # SimulationResult, etc.
│   └── ...
├── control/               # Control design utilities
├── observers/             # State estimation
└── visualization/         # Plotting tools
```

### Abstract Base Classes (Upcoming Refactor)

All systems inherit from clean base classes:

- `ContinuousSystemBase`: Abstract interface for continuous-time systems
- `DiscreteSystemBase`: Abstract interface for discrete-time systems
- Type-safe with comprehensive type hints

---

## Built-in Systems

### Mechanical Systems
- **Pendulum**: Simple pendulum with damping
- **Cart-Pole**: Inverted pendulum on cart
- **Mass-Spring-Damper**: Linear oscillator

### Aerial Systems
- **Quadrotor**: 6-DOF quadrotor dynamics
- **PVTOL**: 6-DOF quadrotor dynamics, body-centered velocity coordinates

### Stochastic Processes
- **Brownian Motion**: Standard Wiener process
- **Geometric Brownian Motion**: Stock price model
- **Ornstein-Uhlenbeck**: Mean-reverting process
- **Discrete Random Walk**: Discrete-time stochastic

---

## Backends & Integrators

### Supported Backends

| Backend | Use Case | GPU | JIT | Auto-Diff |
|---------|----------|-----|-----|-----------|
| **NumPy** | Prototyping, CPU | ❌ | ❌ | ❌ |
| **PyTorch** | Neural networks, GPU | ✅ | ✅ | ✅ |
| **JAX** | Research, TPU, functional | ✅ | ✅ | ✅ |

### ODE/SDE Integrators

**Continuous (ODE)**:
- SciPy (`solve_ivp`)
- torchdiffeq (PyTorch)
- Diffrax (JAX)
- DifferentialEquations.jl (via DiffEqPy)
- Fixed-step: Euler, Midpoint, RK4

**Stochastic (SDE)**:
- torchsde (PyTorch)
- Diffrax (JAX)
- DifferentialEquations.jl (via DiffEqPy)

---

## Installation Options

### Basic (Core Only)
```bash
pip install -e .
```

Includes: NumPy, SymPy, SciPy, PyTorch (for now)

### With JAX
```bash
pip install -e ".[jax]"
```

### With Visualization
```bash
pip install -e ".[viz]"
```

Includes: Plotly, Matplotlib

### Development
```bash
pip install -e ".[dev]"
```

Includes: pytest, black, ruff, mypy, pre-commit

### Everything
```bash
pip install -e ".[all]"
```

---

## Documentation

### Quick Links (Upcoming)
- [Installation Guide](INSTALLATION.md)
- [Migration Guide](MIGRATION_GUIDE.md)
- [Changelog](CHANGELOG.md)
- [Contributing](CONTRIBUTING.md)
- [Examples](example_notebooks/)

### API Reference
Full API documentation coming soon. For now, see docstrings and type hints in source code.

---

## Roadmap

### Current: v0.1.0 (In Development)
- ✅ Refactored architecture with abstract base classes
- ✅ Comprehensive type system
- ✅ Structured return types
- ✅ Backward compatibility with v0.1.0
- 🚧 Complete documentation
- 🚧 Extended examples

### Future: v0.2.0+
- 🔮 **RL Environment Integration**: Automatic Gymnasium/PyBullet wrappers
- 🔮 **Synthetic Data Generation**: Batch simulation and export utilities
- 🔮 **Neural Network Verification**: VNN-Lib, Auto-LiRPA integration
- 🔮 **Lyapunov Synthesis**: Neural Lyapunov controller design
- 🔮 **Advanced MPC**: Integration with do-mpc, CasADi, acados
- 🔮 **Parameter Sensitivity**: Sobol indices, Morris screening
- 🔮 **Composite Systems**: Connect multiple systems together
- 🔮 **Stochastic Observations**: Noisy measurement models

See [Phase 17 in refactoring plan](docs/refactoring_checklist.md#phase-17-future-extensions-post-refactoring-roadmap) for details.

---

## Project Status

**Current Version**: 0.1.0 (refactoring to 0.2.0 in progress)  
**Status**: 🟡 Alpha - Active development  
**Test Coverage**: ~85% (140+ test files)  
**Type Coverage**: Partial (improving to 100%)

### What Works
- ✅ Symbolic system definition
- ✅ Multi-backend execution (NumPy, PyTorch, JAX)
- ✅ ODE/SDE simulation
- ✅ Discretization
- ✅ Basic control (LQR, linearization)
- ✅ State estimation (Kalman filters)
- ✅ Visualization

### In Progress
- 🚧 Architecture refactoring (see [refactoring plan](docs/refactoring_checklist.md))
- 🚧 Complete type coverage
- 🚧 Extended documentation
- 🚧 More examples

---

## Contributing

Contributions are not welcome until API stabilizes

### Development Setup

```bash
# Clone and install
git clone https://github.com/gbenezer/ControlDESymulation.git
cd ControlDESymulation
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/

# Format code
black src/ tests/
isort src/ tests/

# Type check
mypy src/

# Lint
ruff check src/ tests/
```

---

## Citation

If you use this library in your research, please cite:

```bibtex
@software{benezer2025controlde,
  author = {Benezer, Gil},
  title = {ControlDESymulation: Symbolic Dynamical System Specification for Modern Scientific Computing},
  year = {2025},
  url = {https://github.com/gbenezer/ControlDESymulation},
  version = {0.1.0}
}
```

---

## License

This project is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**.

### What This Means

**✅ You CAN**:
- Use for research, academic, commercial, internal purposes
- Modify and distribute
- Use in private or internal services

**⚠️ You MUST**:
- Share modifications if you run modified code as a public/network service
- Keep the same AGPL-3.0 license for modifications
- Provide source code to users of your network service

**❌ You DON'T need to**:
- Share your own code that *uses* this library
- Release internal modifications (unless network-accessible)
- Open-source your proprietary systems

### Why AGPL?

The AGPL ensures that improvements made for public-facing services are shared with the community, supporting **open science**, **reproducibility**, and **fairness**.

For commercial use or private licensing, please contact: gil.benezer@gmail.com

See [LICENSE](LICENSE) for full terms.

---

## Acknowledgments

This library builds on outstanding open-source tools:

### Core Dependencies
- **SymPy**: Symbolic mathematics
- **NumPy/SciPy**: Numerical computing
- **PyTorch**: Automatic differentiation and GPU support
- **JAX**: JIT compilation and functional programming

### Integrators
- **torchdiffeq**: PyTorch ODE solver
- **torchsde**: PyTorch SDE solver
- **Diffrax**: JAX ODE/SDE solver
- **DiffEqPy**: Julia DifferentialEquations.jl wrapper

### Other Tools
- **python-control**: Control theory algorithms
- **Plotly**: Interactive visualization
- **pytest**: Testing framework

### Inspiration

This library was inspired by a class project for **CS 7268: Verifiable Machine Learning** taught by Professor [Michael Everett](https://mfe7.github.io/) at Northeastern University in Fall 2025.

Original project: [Lyapunov-Stable Neural Controllers](https://github.com/gbenezer/Lyapunov_Stable_NN_Controllers_Custom_Dynamics)

---

## Links

- **GitHub**: [github.com/gbenezer/ControlDESymulation](https://github.com/gbenezer/ControlDESymulation)
- **Issues**: [Report bugs or request features](https://github.com/gbenezer/ControlDESymulation/issues)
- **Examples**: [Jupyter notebooks](example_notebooks/)
- **Documentation**: Coming soon

---

## Contact

**Gil Benezer**
- Email: gil.benezer@gmail.com
- GitHub: [@gbenezer](https://github.com/gbenezer)

For bug reports or feature requests, please [open an issue](https://github.com/gbenezer/ControlDESymulation/issues).

For commercial licensing or consulting, contact via email.

---

**Built for control theorists, machine learning researchers, roboticists, and scientists who need powerful, reproducible, and flexible dynamical system modeling.** 🚀

---

<p align="center">
  <sub>Write once, run anywhere. Define symbolically, execute numerically.</sub>
</p>