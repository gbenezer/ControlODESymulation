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
Discrete System Base - Abstract Interface for Discrete-Time Systems

Defines the common interface that ALL discrete-time systems must implement,
regardless of whether they are:
- Native discrete (difference equations)
- Discretized continuous (via wrapper)
- Deterministic or stochastic
- Symbolic or numerical

This enables polymorphic code that works with any discrete system type.

Use Cases
---------
- Controller design functions that work with any discrete system
- Simulation frameworks that handle both native and wrapped systems
- Type hints for discrete-time algorithms
- Testing with mock discrete systems

Architecture
-----------
This is a PURE interface (ABC) with:
- Required abstract methods (must implement)
- Optional methods with default implementations (can override)
- Properties for system dimensions

Implementations:
- DiscreteSymbolicSystem: Native discrete, symbolic
- DiscreteStochasticSystem: Native discrete, symbolic, stochastic  
- DiscretizationWrapper: Wraps continuous systems

Design Philosophy
----------------
**Minimal Interface**: Only methods that ALL discrete systems must have.
**Flexible Return Types**: linearize() can return 2-tuple or 3-tuple.
**No Assumptions**: Doesn't assume symbolic, numerical, or backend.
**Practical Defaults**: Common operations have default implementations.

Examples
--------
>>> # Polymorphic controller design
>>> def design_lqr(system: DiscreteSystemBase, Q, R):
...     '''Works with ANY discrete system.'''
...     result = system.linearize(x_eq, u_eq)
...     Ad, Bd = result[0], result[1]  # Works for both 2 and 3-tuple
...     K = solve_dare(Ad, Bd, Q, R)
...     return K
>>> 
>>> # Works with native discrete
>>> native = DiscreteCartPole()
>>> K1 = design_lqr(native, Q, R)  ✅
>>> 
>>> # Works with discretized continuous
>>> continuous = Pendulum()
>>> discretizer = Discretizer(continuous, dt=0.01)
>>> wrapped = DiscretizationWrapper(continuous, discretizer)
>>> K2 = design_lqr(wrapped, Q, R)  ✅
>>> 
>>> # Works with stochastic
>>> stochastic = DiscreteAR1()
>>> K3 = design_lqr(stochastic, Q, R)  ✅

Type Checking Example
--------------------
>>> from typing import Union
>>> 
>>> def simulate_discrete_system(
...     system: DiscreteSystemBase,
...     x0: np.ndarray,
...     u_seq: Optional[np.ndarray] = None,
...     steps: int = 100
... ) -> np.ndarray:
...     '''Type checker knows system has step() method.'''
...     return system.simulate(x0, u_seq, steps)
"""

from abc import ABC, abstractmethod
from typing import Union, Tuple, Optional, Dict, Any
import numpy as np

# Type aliases
ArrayLike = Union[np.ndarray, 'torch.Tensor', 'jax.numpy.ndarray']
LinearizationResult = Union[
    Tuple[ArrayLike, ArrayLike],           # Deterministic: (Ad, Bd)
    Tuple[ArrayLike, ArrayLike, ArrayLike] # Stochastic: (Ad, Bd, Gd)
]


class DiscreteSystemBase(ABC):
    """
    Abstract base class for discrete-time dynamical systems.
    
    Defines common interface for all discrete systems:
        x[k+1] = F(x[k], u[k])           # Deterministic
        x[k+1] = f(x[k], u[k]) + g*w[k]  # Stochastic
    
    All discrete systems must implement:
    - step(): Compute next state
    - linearize(): Compute discrete-time Jacobians
    - Properties: nx, nu, is_stochastic
    
    Optional methods with defaults:
    - simulate(): Multi-step simulation
    - check_stability(): Eigenvalue-based stability
    - batch_step(): Vectorized stepping (if supported)
    
    Subclasses
    ----------
    - DiscreteSymbolicSystem: Native discrete, symbolic (SymPy)
    - DiscreteStochasticSystem: Native discrete, symbolic, stochastic
    - DiscretizationWrapper: Wraps continuous systems
    - Future: DiscreteNumericalSystem (function-based, no SymPy)
    
    Examples
    --------
    >>> class MyDiscreteSystem(DiscreteSystemBase):
    ...     '''Custom discrete system implementation.'''
    ...     
    ...     def step(self, x, u=None):
    ...         return x + 0.1 * u if u is not None else x
    ...     
    ...     def linearize(self, x_eq, u_eq=None):
    ...         Ad = np.eye(self.nx)
    ...         Bd = 0.1 * np.ones((self.nx, self.nu))
    ...         return Ad, Bd
    ...     
    ...     @property
    ...     def nx(self): return 2
    ...     
    ...     @property
    ...     def nu(self): return 1
    ...     
    ...     @property
    ...     def is_stochastic(self): return False
    >>> 
    >>> system = MyDiscreteSystem()
    >>> x_next = system.step(np.array([1.0, 0.0]), np.array([0.5]))
    """
    
    # ========================================================================
    # Core Interface - REQUIRED (Must Implement)
    # ========================================================================
    
    @abstractmethod
    def step(
        self,
        x: ArrayLike,
        u: Optional[ArrayLike] = None,
        **kwargs
    ) -> ArrayLike:
        """
        Compute one discrete time step: x[k+1] = F(x[k], u[k]).
        
        This is the fundamental operation for discrete systems.
        
        Parameters
        ----------
        x : ArrayLike
            Current state x[k], shape (nx,) or (batch, nx)
        u : Optional[ArrayLike]
            Control input u[k], shape (nu,) or (batch, nu)
            None for autonomous systems (nu=0)
        **kwargs
            Additional arguments (e.g., w for stochastic systems)
        
        Returns
        -------
        ArrayLike
            Next state x[k+1], same shape as x
        
        Notes
        -----
        For deterministic systems:
            x[k+1] = f(x[k], u[k])
        
        For stochastic systems (mean dynamics):
            E[x[k+1]] = f(x[k], u[k])
        
        Stochastic systems may accept additional kwargs like:
            - w: Standard normal noise (nw,)
            - For full stochastic step: x[k+1] = f(x,u) + g(x,u)*w
        
        Examples
        --------
        >>> # Deterministic step
        >>> x_next = system.step(x, u)
        >>> 
        >>> # Autonomous system
        >>> x_next = system.step(x)
        >>> 
        >>> # Stochastic system (mean)
        >>> x_next = system.step(x, u)  # Returns E[x[k+1]]
        >>> 
        >>> # Stochastic system (full, with noise)
        >>> x_next = system.step(x, u, w=np.random.randn(nw))
        """
        pass
    
    @abstractmethod
    def linearize(
        self,
        x_eq: ArrayLike,
        u_eq: Optional[ArrayLike] = None,
        **kwargs
    ) -> LinearizationResult:
        """
        Compute discrete-time linearization at equilibrium.
        
        Linearized dynamics around (x_eq, u_eq):
            δx[k+1] ≈ Ad*δx[k] + Bd*δu[k]              # Deterministic
            δx[k+1] ≈ Ad*δx[k] + Bd*δu[k] + Gd*w[k]    # Stochastic
        
        Parameters
        ----------
        x_eq : ArrayLike
            Equilibrium state for linearization (nx,)
        u_eq : Optional[ArrayLike]
            Equilibrium control (nu,)
            None for autonomous systems (nu=0)
        **kwargs
            Additional arguments (e.g., method='exact')
        
        Returns
        -------
        Ad : ArrayLike
            Discrete-time state matrix (nx, nx)
        Bd : ArrayLike  
            Discrete-time control matrix (nx, nu)
        Gd : ArrayLike (optional, stochastic only)
            Discrete-time noise gain matrix (nx, nw)
        
        Returns either:
        - (Ad, Bd) for deterministic systems
        - (Ad, Bd, Gd) for stochastic systems
        
        Notes
        -----
        For native discrete systems:
            Ad = ∂f/∂x at (x_eq, u_eq)
            Bd = ∂f/∂u at (x_eq, u_eq)
            Gd = g(x_eq, u_eq) for stochastic
        
        For discretized continuous systems:
            Depends on discretization method (euler, exact, etc.)
        
        Examples
        --------
        >>> # Deterministic system
        >>> Ad, Bd = system.linearize(x_eq, u_eq)
        >>> print(Ad.shape)  # (nx, nx)
        >>> 
        >>> # Stochastic system
        >>> Ad, Bd, Gd = system.linearize(x_eq, u_eq)
        >>> print(Gd.shape)  # (nx, nw)
        >>> 
        >>> # Handle both polymorphically
        >>> result = system.linearize(x_eq, u_eq)
        >>> Ad = result[0]  # Always first
        >>> Bd = result[1]  # Always second
        >>> if len(result) == 3:
        ...     Gd = result[2]  # Third if stochastic
        """
        pass
    
    # ========================================================================
    # System Properties - REQUIRED
    # ========================================================================
    
    @property
    @abstractmethod
    def nx(self) -> int:
        """
        Number of state variables.
        
        Returns
        -------
        int
            State dimension
        
        Examples
        --------
        >>> system.nx
        2
        """
        pass
    
    @property
    @abstractmethod
    def nu(self) -> int:
        """
        Number of control inputs.
        
        Returns 0 for autonomous systems.
        
        Returns
        -------
        int
            Control dimension
        
        Examples
        --------
        >>> system.nu
        1
        >>> 
        >>> # Autonomous system
        >>> autonomous_system.nu
        0
        """
        pass
    
    @property
    @abstractmethod
    def is_stochastic(self) -> bool:
        """
        Whether system has stochastic components.
        
        Returns
        -------
        bool
            True if system includes noise (w[k] terms)
            False for deterministic systems
        
        Examples
        --------
        >>> deterministic_system.is_stochastic
        False
        >>> 
        >>> stochastic_system.is_stochastic
        True
        """
        pass
    
    # ========================================================================
    # Optional Properties (Can Override)
    # ========================================================================
    
    @property
    def is_autonomous(self) -> bool:
        """
        Whether system is autonomous (no control inputs).
        
        Default implementation checks if nu == 0.
        
        Returns
        -------
        bool
            True if nu = 0
        
        Examples
        --------
        >>> system.is_autonomous
        False
        >>> 
        >>> autonomous_system.is_autonomous
        True
        """
        return self.nu == 0
    
    @property
    def ny(self) -> int:
        """
        Number of outputs.
        
        Default: ny = nx (full state output).
        Override if system has custom output mapping.
        
        Returns
        -------
        int
            Output dimension
        """
        return self.nx
    
    # ========================================================================
    # Optional Interface - Default Implementations
    # ========================================================================
    
    def simulate(
        self,
        x0: ArrayLike,
        u_seq: Optional[ArrayLike] = None,
        steps: int = 100,
        return_inputs: bool = False,
        **step_kwargs
    ) -> Union[ArrayLike, Tuple[ArrayLike, ArrayLike]]:
        """
        Simulate discrete-time trajectory.
        
        Default implementation using step() method.
        Can be overridden for efficiency (e.g., batched simulation).
        
        Parameters
        ----------
        x0 : ArrayLike
            Initial state (nx,)
        u_seq : Optional[ArrayLike]
            Control sequence (steps, nu) or None for autonomous
        steps : int
            Number of time steps to simulate
        return_inputs : bool
            If True, return (states, controls)
        **step_kwargs
            Additional arguments passed to step()
            (e.g., w_seq for stochastic systems)
        
        Returns
        -------
        trajectory : ArrayLike
            State trajectory (steps+1, nx)
            Includes initial state x0 at index 0
        controls : ArrayLike (optional)
            Control sequence used (steps, nu)
            Only if return_inputs=True
        
        Examples
        --------
        >>> # Autonomous system
        >>> trajectory = system.simulate(x0, steps=100)
        >>> print(trajectory.shape)  # (101, nx)
        >>> 
        >>> # Controlled system
        >>> u_seq = np.random.randn(100, nu)
        >>> trajectory = system.simulate(x0, u_seq, steps=100)
        >>> 
        >>> # With inputs returned
        >>> states, controls = system.simulate(
        ...     x0, u_seq, steps=100, return_inputs=True
        ... )
        >>> 
        >>> # Stochastic system with custom noise
        >>> w_seq = np.random.randn(100, nw)
        >>> trajectory = system.simulate(x0, u_seq, steps=100, w_seq=w_seq)
        """
        # Convert to numpy for consistency
        x0 = np.asarray(x0)
        
        # Initialize trajectory storage
        trajectory = [x0]
        x = x0
        
        # Extract noise sequence if provided (for stochastic)
        w_seq = step_kwargs.pop('w_seq', None)
        
        # Simulate
        for k in range(steps):
            # Get control for this step
            u = u_seq[k] if u_seq is not None else None
            
            # Get noise for this step (stochastic only)
            if w_seq is not None:
                step_kwargs['w'] = w_seq[k]
            
            # Compute next state
            x = self.step(x, u, **step_kwargs)
            trajectory.append(x)
        
        # Convert to array
        trajectory = np.array(trajectory)
        
        if return_inputs:
            return trajectory, u_seq
        else:
            return trajectory
    
    def check_stability(
        self,
        x_eq: ArrayLike,
        u_eq: Optional[ArrayLike] = None,
        tolerance: float = 1e-6,
        **linearize_kwargs
    ) -> Dict[str, Any]:
        """
        Check stability at equilibrium point.
        
        Default implementation using eigenvalues of Ad.
        System is stable if all |λ| < 1.
        
        Can be overridden for:
        - Custom stability criteria
        - Lyapunov-based stability
        - Stochastic stability (mean-square)
        
        Parameters
        ----------
        x_eq : ArrayLike
            Equilibrium state (nx,)
        u_eq : Optional[ArrayLike]
            Equilibrium control (nu,)
        tolerance : float
            Tolerance for marginal stability (|λ| ≈ 1)
        **linearize_kwargs
            Passed to linearize() (e.g., method='exact')
        
        Returns
        -------
        dict
            Stability analysis:
            - 'eigenvalues': Eigenvalues of Ad
            - 'magnitudes': Absolute values of eigenvalues
            - 'max_magnitude': Maximum |λ|
            - 'spectral_radius': Same as max_magnitude
            - 'is_stable': True if all |λ| < 1
            - 'is_marginally_stable': True if max|λ| ≈ 1
            - 'is_unstable': True if any |λ| > 1
        
        Examples
        --------
        >>> stability = system.check_stability(x_eq, u_eq)
        >>> 
        >>> if stability['is_stable']:
        ...     print("Equilibrium is stable!")
        ...     print(f"Spectral radius: {stability['spectral_radius']:.3f}")
        ... else:
        ...     print(f"Unstable with max|λ|={stability['max_magnitude']:.3f}")
        >>> 
        >>> # Visualize eigenvalues
        >>> import matplotlib.pyplot as plt
        >>> eigs = stability['eigenvalues']
        >>> plt.scatter(eigs.real, eigs.imag)
        >>> circle = plt.Circle((0, 0), 1, fill=False, color='red')
        >>> plt.gca().add_patch(circle)
        >>> plt.title('Eigenvalues (inside circle = stable)')
        """
        # Get linearization
        result = self.linearize(x_eq, u_eq, **linearize_kwargs)
        Ad = result[0]  # First element is always Ad
        
        # Convert to numpy for eigenvalue computation
        Ad_np = np.asarray(Ad)
        
        # Compute eigenvalues
        eigenvalues = np.linalg.eigvals(Ad_np)
        magnitudes = np.abs(eigenvalues)
        max_magnitude = np.max(magnitudes)
        
        # Stability classification
        is_stable = bool(max_magnitude < 1.0)
        is_marginally_stable = bool(np.abs(max_magnitude - 1.0) < tolerance)
        is_unstable = bool(max_magnitude > 1.0)
        
        return {
            'eigenvalues': eigenvalues,
            'magnitudes': magnitudes,
            'max_magnitude': float(max_magnitude),
            'spectral_radius': float(max_magnitude),
            'is_stable': is_stable,
            'is_marginally_stable': is_marginally_stable,
            'is_unstable': is_unstable,
        }
    
    def batch_step(
        self,
        x_batch: ArrayLike,
        u_batch: Optional[ArrayLike] = None,
        **kwargs
    ) -> ArrayLike:
        """
        Compute batch of steps: x_batch[k+1] = F(x_batch[k], u_batch[k]).
        
        Default implementation loops over batch.
        Override for vectorized/parallel implementations (e.g., PyTorch, JAX).
        
        Parameters
        ----------
        x_batch : ArrayLike
            Batch of states (batch_size, nx)
        u_batch : Optional[ArrayLike]
            Batch of controls (batch_size, nu)
            None for autonomous systems
        **kwargs
            Additional arguments for step()
        
        Returns
        -------
        ArrayLike
            Next states (batch_size, nx)
        
        Examples
        --------
        >>> # Process batch of initial conditions
        >>> x_batch = np.random.randn(100, nx)
        >>> u_batch = np.zeros((100, nu))
        >>> x_next_batch = system.batch_step(x_batch, u_batch)
        >>> print(x_next_batch.shape)  # (100, nx)
        
        Notes
        -----
        Override this for backends that support vectorization:
        - PyTorch: Can use vmap or batch operations
        - JAX: Can use jax.vmap
        - NumPy: May still need loop (default)
        """
        batch_size = x_batch.shape[0]
        results = []
        
        for i in range(batch_size):
            x_i = x_batch[i]
            u_i = u_batch[i] if u_batch is not None else None
            x_next_i = self.step(x_i, u_i, **kwargs)
            results.append(x_next_i)
        
        return np.array(results)
    
    # ========================================================================
    # Convenience Methods
    # ========================================================================
    
    def __call__(
        self,
        x: ArrayLike,
        u: Optional[ArrayLike] = None
    ) -> ArrayLike:
        """
        Make system callable: system(x, u) → x[k+1].
        
        Provides convenient syntax matching continuous systems.
        
        Examples
        --------
        >>> x_next = system(x, u)
        >>> # Equivalent to:
        >>> x_next = system.step(x, u)
        """
        return self.step(x, u)
    
    def is_controllable(
        self,
        x_eq: ArrayLike,
        u_eq: Optional[ArrayLike] = None,
        **linearize_kwargs
    ) -> bool:
        """
        Check if system is controllable at equilibrium.
        
        Uses controllability matrix rank test.
        
        Parameters
        ----------
        x_eq : ArrayLike
            Equilibrium state
        u_eq : Optional[ArrayLike]
            Equilibrium control
        **linearize_kwargs
            Passed to linearize()
        
        Returns
        -------
        bool
            True if controllable (rank = nx)
        
        Examples
        --------
        >>> if system.is_controllable(x_eq, u_eq):
        ...     print("System is controllable - can design LQR")
        """
        result = self.linearize(x_eq, u_eq, **linearize_kwargs)
        Ad, Bd = result[0], result[1]
        
        # Convert to numpy
        Ad_np = np.asarray(Ad)
        Bd_np = np.asarray(Bd)
        
        nx = Ad_np.shape[0]
        nu = Bd_np.shape[1]
        
        # Build controllability matrix: C = [Bd, Ad*Bd, ..., Ad^(n-1)*Bd]
        C = np.zeros((nx, nx * nu))
        Ad_power = np.eye(nx)
        
        for i in range(nx):
            C[:, i*nu:(i+1)*nu] = Ad_power @ Bd_np
            Ad_power = Ad_power @ Ad_np
        
        # Check rank
        rank = np.linalg.matrix_rank(C)
        return rank == nx
    
    # ========================================================================
    # Information Methods
    # ========================================================================
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get system information.
        
        Default implementation returns basic info.
        Override to provide more details.
        
        Returns
        -------
        dict
            System information including:
            - 'class': Class name
            - 'nx': State dimension
            - 'nu': Control dimension  
            - 'ny': Output dimension
            - 'is_stochastic': Stochastic flag
            - 'is_autonomous': Autonomous flag
        
        Examples
        --------
        >>> info = system.get_info()
        >>> print(f"System: {info['class']}")
        >>> print(f"States: {info['nx']}, Controls: {info['nu']}")
        >>> print(f"Stochastic: {info['is_stochastic']}")
        """
        return {
            'class': self.__class__.__name__,
            'nx': self.nx,
            'nu': self.nu,
            'ny': self.ny,
            'is_stochastic': self.is_stochastic,
            'is_autonomous': self.is_autonomous,
        }
    
    def __repr__(self) -> str:
        """
        String representation for debugging.
        
        Default implementation, can be overridden.
        
        Examples
        --------
        >>> repr(system)
        'DiscreteCartPole(nx=4, nu=1, deterministic)'
        """
        stochastic_str = "stochastic" if self.is_stochastic else "deterministic"
        return (
            f"{self.__class__.__name__}("
            f"nx={self.nx}, nu={self.nu}, {stochastic_str})"
        )
    
    def __str__(self) -> str:
        """
        Human-readable string representation.
        
        Default implementation, can be overridden.
        
        Examples
        --------
        >>> str(system)
        'DiscreteCartPole: 4 states, 1 control (deterministic)'
        """
        stochastic_str = "stochastic" if self.is_stochastic else "deterministic"
        return (
            f"{self.__class__.__name__}: "
            f"{self.nx} state{'s' if self.nx != 1 else ''}, "
            f"{self.nu} control{'s' if self.nu != 1 else ''} "
            f"({stochastic_str})"
        )


# ============================================================================
# Type Checking Utilities
# ============================================================================

def is_discrete_system(obj: Any) -> bool:
    """
    Check if object is a discrete-time system.
    
    Parameters
    ----------
    obj : Any
        Object to check
    
    Returns
    -------
    bool
        True if obj implements DiscreteSystemBase
    
    Examples
    --------
    >>> system = DiscreteCartPole()
    >>> is_discrete_system(system)
    True
    >>> 
    >>> continuous = Pendulum()
    >>> is_discrete_system(continuous)
    False
    """
    return isinstance(obj, DiscreteSystemBase)


def require_discrete_system(obj: Any, name: str = "system"):
    """
    Validate that object is a discrete system.
    
    Parameters
    ----------
    obj : Any
        Object to validate
    name : str
        Parameter name for error message
    
    Raises
    ------
    TypeError
        If obj is not a DiscreteSystemBase
    
    Examples
    --------
    >>> def my_controller(system):
    ...     require_discrete_system(system, "system")
    ...     # Now guaranteed to have step(), linearize(), etc.
    ...     return system.step(x, u)
    """
    if not isinstance(obj, DiscreteSystemBase):
        raise TypeError(
            f"Parameter '{name}' must be a DiscreteSystemBase, "
            f"got {type(obj).__name__}. "
            f"For continuous systems, wrap with DiscretizationWrapper."
        )


# ============================================================================
# Helper Functions
# ============================================================================

def simulate_discrete(
    system: DiscreteSystemBase,
    x0: ArrayLike,
    u_seq: Optional[ArrayLike] = None,
    steps: int = 100
) -> ArrayLike:
    """
    Convenience function for simulation.
    
    Delegates to system.simulate() but provides consistent interface.
    
    Parameters
    ----------
    system : DiscreteSystemBase
        Discrete system to simulate
    x0 : ArrayLike
        Initial state
    u_seq : Optional[ArrayLike]
        Control sequence
    steps : int
        Number of steps
    
    Returns
    -------
    ArrayLike
        Trajectory (steps+1, nx)
    
    Examples
    --------
    >>> trajectory = simulate_discrete(system, x0, u_seq, steps=100)
    """
    return system.simulate(x0, u_seq, steps)


def check_discrete_stability(
    system: DiscreteSystemBase,
    x_eq: ArrayLike,
    u_eq: Optional[ArrayLike] = None
) -> bool:
    """
    Quick stability check (returns boolean only).
    
    Parameters
    ----------
    system : DiscreteSystemBase
        System to check
    x_eq : ArrayLike
        Equilibrium state
    u_eq : Optional[ArrayLike]
        Equilibrium control
    
    Returns
    -------
    bool
        True if stable
    
    Examples
    --------
    >>> if check_discrete_stability(system, x_eq, u_eq):
    ...     print("Equilibrium is stable!")
    """
    stability = system.check_stability(x_eq, u_eq)
    return stability['is_stable']