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

## 🎯 Core Features

- **Symbolic Specification**: Define systems using SymPy with automatic code generation
- **Multi-Backend Support**: Seamlessly switch between NumPy, PyTorch, JAX, and Julia
- **Dual Time Domains**: Full support for both continuous-time (ODEs/SDEs) and discrete-time systems
- **Stochastic Systems**: First-class support for stochastic differential equations (SDEs)
- **40+ Integration Methods**: Adaptive and fixed-step solvers from scipy, torchdiffeq, diffrax, and DiffEqPy
- **Type Safety**: Comprehensive TypedDict definitions with IDE autocomplete support
- **GPU Acceleration**: Native PyTorch and JAX support for GPU-based simulations
- **Zero Code Duplication**: Clean 4-layer architecture with composition over inheritance
- **Production Ready**: Extensive test coverage, comprehensive documentation, CI/CD workflows

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/gbenezer/ControlDESymulation.git
cd ControlDESymulation

# Install with pip (editable mode)
pip install -e .

# Or with specific backends
pip install -e ".[torch]"      # PyTorch support
pip install -e ".[jax]"        # JAX support
pip install -e ".[jax,viz]"    # JAX + visualization
pip install -e ".[all]"        # Everything
```

### Your First System

```python
from src.systems.base import ContinuousSymbolicSystem
import sympy as sp
import numpy as np

class Pendulum(ContinuousSymbolicSystem):
    """Simple pendulum with damping and control."""
    
    def define_system(self, m=1.0, l=0.5, g=9.81, b=0.1):
        # Define symbolic variables
        theta, theta_dot = sp.symbols('theta theta_dot', real=True)
        u = sp.symbols('u', real=True)
        m_sym, l_sym, g_sym, b_sym = sp.symbols('m l g b', positive=True)
        
        # Set state and control variables
        self.state_vars = [theta, theta_dot]
        self.control_vars = [u]
        
        # Define dynamics: dx/dt = f(x, u)
        self._f_sym = sp.Matrix([
            theta_dot,
            -(g_sym/l_sym)*sp.sin(theta) - (b_sym/(m_sym*l_sym**2))*theta_dot 
            + u/(m_sym*l_sym**2)
        ])
        
        # Set parameters and order
        self.parameters = {m_sym: m, l_sym: l, g_sym: g, b_sym: b}
        self.order = 1

    def setup_equilibria(self):
        # method to add equilibria to the system automatically after initialization

        # add the stable equilibrium where the pendulum is hanging down
        self.add_equilibrium(
            'downward',
            x_eq=np.array([0.0, 0.0]),
            u_eq=np.array([0.0]),
            verify=True
            )

        # add the unstable equilibrium where the pendulum is inverted
        self.add_equilibrium(
            'inverted',
            x_eq=np.array([np.pi, 0.0]),
            u_eq=np.array([0.0]),
            stability='unstable',
            notes='Requires active control'
            )

# Create and simulate
system = Pendulum(m=0.5, l=0.3)
x0 = np.array([0.1, 0.0])  # Initial angle and velocity

# Integrate with automatic solver selection
result = system.integrate(x0, u=None, t_span=(0, 5))

print(f"Integration successful: {result['success']}")
print(f"Solver used: {result['solver']}")
print(f"Final state: {result['x'][-1]}")

# Linearize around unstable equilibrium
x, u = system.get_equilibrium(name='inverted')
A, B = system.linearize(x, u)
print(f"A matrix:\n{A}")
```

### Multi-Backend Execution

**Same definition, different backends - no code changes required!**

```python
# NumPy (CPU, prototyping)
import numpy as np
pendulum.set_default_backend('numpy')
x_np = np.array([0.1, 0.0])
dx_np = pendulum(x_np, np.array([0.0]))

# PyTorch (GPU, neural networks)
import torch
pendulum.set_default_backend('torch')
pendulum.set_default_device('cuda:0')
x_torch = torch.tensor([0.1, 0.0], device='cuda')
dx_torch = pendulum(x_torch, torch.tensor([0.0], device='cuda'))

# JAX (TPU, JIT compilation)
import jax.numpy as jnp
pendulum.set_default_backend('jax')
x_jax = jnp.array([0.1, 0.0])
dx_jax = pendulum(x_jax, jnp.array([0.0]))
```

**Backend detection is automatic** - the same system object works with all three!

## 📚 Documentation Structure

### Quick Reference Guides

- **[UI Framework Quick Reference](docs/api/ui_framework/UI_Framework_Quick_Reference.md)** - System definition and usage patterns
- **[Integration Framework Quick Reference](docs/api/integration_framework/Integration_Framework_Quick_Reference.md)** - Solver selection and configuration
- **[Type System Quick Reference](docs/api/type_system/Type_System_Quick_Reference.md)** - TypedDict definitions and type safety
- **[Visualization Framework Quick Reference](docs/api/visualization_framework/Visualization_Framework_Quick_Reference.md)** - Plotting and analysis
- **[Control Framework Quick Reference](docs/api/control_framework/Control_Framework_Quick_Reference.md)** - Controller design
- **[Delegation Layer Quick Reference](docs/api/delegation_layer/Delegation_Layer_Quick_Reference.md)** - Low-level numerical operations

### Architecture Documentation

- **[UI Framework Architecture](docs/api/ui_framework/UI_Framework_Architecture.md)** - System class hierarchy and design
- **[Integration Framework Architecture](docs/api/integration_framework/Integration_Framework_Architecture.md)** - Numerical integration infrastructure
- **[Type System Architecture](docs/api/type_system/Type_System_Architecture.md)** - Type definitions and validation
- **[Visualization Framework Architecture](docs/api/visualization_framework/Visualization_Framework_Architecture.md)** - Plotting and analysis
- **[Control Framework Architecture](docs/api/control_framework/Control_Framework_Architecture.md)** - Control system utilities
- **[Delegation Layer Architecture](docs/api/delegation_layer/Delegation_Layer_Architecture.md)** - Backend abstraction layer
- **[Design Philosophy](docs/api/ControlDESymulation_Design_Philosophy.md)** - Core design principles

## 🏗️ Architecture Overview

ControlDESymulation follows a clean 4-layer architecture:

```
┌─────────────────────────────────────────────────────────────┐
│                  Layer 3: User Interface                     │
│  ContinuousSymbolicSystem  |  DiscreteSymbolicSystem         │
│  ContinuousStochasticSystem | DiscreteStochasticSystem       │
│                                                              │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│              Layer 2: Numerical Integration                  │
│  IntegratorFactory | SDEIntegratorFactory                    │
│  40+ solver methods across NumPy/PyTorch/JAX/Julia           │
│                                                              │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│              Layer 1: Delegation Layer                       │
│  Backend-agnostic numerical operations                       │
│  Array manipulation | Linear algebra | Special functions     │
│                                                              │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│              Layer 0: Type System                            │
│  TypedDict definitions for type safety                       │
│  218 types across 19 focused modules                         │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Design Principles

1. **Composition Over Inheritance** - Delegate to specialized utilities
2. **Backend Agnosticism** - Write once, run on NumPy/PyTorch/JAX/Julia
3. **Type-Driven Design** - Comprehensive TypedDict definitions
4. **Zero Code Duplication** - Shared functionality in base classes
5. **Mathematical Rigor** - Proper handling of ODEs, SDEs, difference equations
6. **Performance First** - Multi-backend support enables GPU and XLA acceleration

## 🔬 System Types

- **Continuous Deterministic Ordinary Differential Equations (Continuous ODEs)**
- **Discrete Deterministic Ordinary Differential Equations (Discrete ODEs)**
- **Continuous Stochastic Differential Equations (Continuous SDEs)**
- **Discrete Stochastic Differential Equations (Discrete SDEs)**
- **Discretized Continuous Systems (either ODE or SDE)**

All systems inherit from clean base classes:
- **SymbolicSystemBase**: Time-domain agnostic symbolic machinery
- **ContinuousSystemBase**: Abstract interface for continuous-time systems
- **DiscreteSystemBase**: Abstract interface for discrete-time systems

---

## 🎮 Backend Flexibility

ControlDESymulation supports multiple numerical backends with automatic detection:

| Backend | Use Case | GPU | JIT | Auto-Diff |
|---------|----------|-----|-----|-----------|
| **NumPy** | Prototyping, CPU | ❌ | ❌ | ❌ |
| **PyTorch** | Neural networks, GPU | ✅ | ✅ | ✅ |
| **JAX** | Research, TPU, functional | ✅ | ✅ | ✅ |

## 🧮 Integration Methods

### Adaptive ODE Solvers

| Method | Backend | Order | Best For |
|--------|---------|-------|----------|
| `LSODA` | NumPy | Variable | Auto-stiffness detection |
| `RK45` | NumPy | 5(4) | General non-stiff |
| `BDF` | NumPy | Variable | Stiff systems |
| `Tsit5` | Julia | 5 | High-performance non-stiff |
| `Vern9` | Julia | 9 | High accuracy |
| `Rodas5` | Julia | 5 | Stiff systems |
| `dopri5` | PyTorch/JAX | 5(4) | Neural ODEs, optimization |

### SDE Solvers

| Method | Order | Noise Type |
|--------|-------|------------|
| `euler-maruyama` | 0.5 (strong) | All |
| `heun` | 1.0 (strong) | Additive |
| `milstein` | 1.0 (strong) | Diagonal |
| `srk` | Various | Specific structures |

## 🎨 Advanced Features

- **Automatic Linearization**
- **Stability, Controllability, and Observability Analysis**
- **Management of Multiple Equilibria**
- **Arbitrary-Order Systems**
- **Classical Closed-Loop Control Design Tools**
- **SDE Monte Carlo Analysis**
- **And more**

## 🧪 Testing

ControlDESymulation includes comprehensive test suites:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html
```

## 📦 Installation Options

### Basic (Core Only)
```bash
pip install -e .
```

**Includes**: NumPy, SymPy, SciPy, pydantic

### With PyTorch
```bash
pip install -e ".[torch]"
```

**Adds**: PyTorch, torchdiffeq, torchsde

### With JAX
```bash
pip install -e ".[jax]"
```

**Adds**: JAX, jaxlib, diffrax

### With Visualization
```bash
pip install -e ".[viz]"
```

**Adds**: Plotly, Matplotlib

### Development
```bash
pip install -e ".[dev]"
```

**Adds**: pytest, black, ruff, mypy, pre-commit, coverage

### Everything
```bash
pip install -e ".[all]"
```

**Includes all backends, visualization, and development tools**

---

## 🤝 Contributing

Repository is not currently ready for other contributors. Once all files are fully documented, initial example notebooks and/or more formal tutorials, all warnings addressed, and all mypy/ruff/pylint issues are resolved, then contributions are welcome. Some examples of future help would be:

1. **Additional Example Systems** - More application domains
2. **Documentation** - Tutorials, guides, and examples
3. **Performance Optimization** - Profiling and speedups
4. **Additional Backends** - TensorFlow, CuPy, etc.
5. **Visualization Tools** - Enhanced plotting capabilities
6. **Control Algorithms** - MPC, LQG, H-infinity, etc.

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

Please see `CONTRIBUTING.md` for detailed guidelines. *(not yet written given lack of preparation)*

---

## 📖 Citation

If you use this library in your research, please cite:

```bibtex
@software{benezer2025controlde,
  author = {Benezer, Gil},
  title = {ControlDESymulation: Symbolic Dynamical System Specification for Modern Scientific Computing},
  year = {2025},
  url = {https://github.com/gbenezer/ControlDESymulation},
  version = {1.0.0}
}
```

---

## 📄 License

ControlDESymulation is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**. The license file is in the repository here: [LICENSE](LICENSE)

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

The AGPL ensures that improvements made for public-facing services are shared with the community, supporting **open science**, **reproducibility**, and **fairness**. This keeps the ecosystem healthy and prevents one-way value extraction.

For commercial use or private licensing inquiries, please contact: **gil.benezer@gmail.com**

See [LICENSE](LICENSE) for full terms.

---

### What This Means

**✅ You CAN**:
- Use for research, academic, commercial, and internal purposes
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

---

## 🙏 Acknowledgments

ControlDESymulation builds on excellent open-source libraries:

### Core Dependencies
- **SymPy** - Symbolic mathematics
- **NumPy/SciPy** - Numerical computing
- **PyTorch** - Deep learning and automatic differentiation
- **JAX** - High-performance numerical computing with XLA

### Integrators
- **scipy.integrate** - ODE/SDE solvers for NumPy backend
- **torchdiffeq** - PyTorch ODE solver
- **torchsde** - PyTorch SDE solver
- **Diffrax** - JAX ODE/SDE solver
- **DiffEqPy** - Julia DifferentialEquations.jl wrapper (world-class solvers)

### Other Tools
- **python-control** - Control theory algorithms
- **Plotly** - Interactive visualization
- **Matplotlib** - Publication-quality plots
- **pytest** - Testing framework
- **pydantic** - Data validation and type safety

### Inspiration

This library was inspired by a class project for **CS 7268: Verifiable Machine Learning** taught by Professor [Michael Everett](https://mfe7.github.io/) at Northeastern University in Fall 2024.

Original project: [Lyapunov-Stable Neural Controllers](https://github.com/gbenezer/Lyapunov_Stable_NN_Controllers_Custom_Dynamics)

---

## 📮 Contact

**Gil Benezer**
- **Email**: gil.benezer@gmail.com
- **GitHub**: [@gbenezer](https://github.com/gbenezer)

### Links

- **GitHub Repository**: [github.com/gbenezer/ControlDESymulation](https://github.com/gbenezer/ControlDESymulation)
- **Issues**: [Report bugs or request features](https://github.com/gbenezer/ControlDESymulation/issues)
- **Documentation**: See reference guides above

For bug reports or feature requests, please [open an issue](https://github.com/gbenezer/ControlDESymulation/issues).

For commercial licensing or consulting inquiries, contact via email.

## 🗺️ Roadmap

### v1.0 (Current - Release Candidate)
- ✅ Core 4-layer architecture
- ✅ Multi-backend support (NumPy/PyTorch/JAX/Julia)
- ✅ ODE and SDE integration with 40+ methods
- ✅ Type system with 200+ TypedDict definitions
- ✅ Comprehensive documentation
- ✅ Classical control design methods
- ✅ Advanced plotting capabilities
- 🔄 Constructing integration test suites for debugging and regression testing
- 🔄 Addressing warnings and ruff/mypy issues
- 🔄 Polishing and reorganizing example systems
- 🔄 Constructing notebooks and tutorials
- 🔄 Verifying and adding additional documentation

### v1.1 (Planned)
- [ ] RL Environment Synthesis using Gymnasium, PyBullet, and/or Brax
- [ ] Generation and Standardized Export of Synthetic Data
- [ ] System Identification and Bayesian Inference
- [ ] Neural Controller and Certificate Function Synthesis
- [ ] Model Predictive Control
- [ ] Stochastic/Noisy Output Dynamics
- [ ] Framework for System Coupling and Composition

### v2.0 (Future)
- [ ] Partial Differential Equations (PDEs)
- [ ] Hybrid systems (continuous + discrete events)
- [ ] Distributed Systems
- [ ] Delay Systems
- [ ] Reachability Analysis
- [ ] Conformal Methods
- [ ] Model Reduction

---

**Built for control theorists, machine learning researchers, roboticists, and scientists who need powerful, reproducible, and flexible dynamical system modeling.**

---

<p align="center">
  <sub>Write once, run anywhere. Define symbolically, execute numerically.</sub>
</p>
