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
Discrete-time Stochastic Dynamical System

Represents stochastic difference equations: x[k+1] = f(x[k], u[k]) + g(x[k], u[k]) * w[k]

This class combines DiscreteSymbolicSystem and StochasticDynamicalSystem concepts
to handle systems that are inherently discrete-time AND stochastic.
"""

from typing import List, Optional, Dict, Any, Union
import sympy as sp
import numpy as np

from src.systems.base.stochastic_dynamical_system import StochasticDynamicalSystem
from src.systems.base.utils.stochastic.noise_analysis import SDEType, NoiseType


class DiscreteStochasticSystem(StochasticDynamicalSystem):
    """
    Symbolic discrete-time stochastic dynamical system.
    
    Represents stochastic difference equations of the form:
        x[k+1] = f(x[k], u[k]) + g(x[k], u[k]) * w[k]
        y[k] = h(x[k])
    
    where:
        - x[k] is the state at discrete time k
        - u[k] is the control input
        - w[k] ~ N(0, I) is standard normal IID noise (nw-dimensional)
        - y[k] is the output
    
    This class inherits from StochasticDynamicalSystem but interprets:
        - _f_sym as the deterministic next state f(x[k], u[k])
        - diffusion_expr as the discrete-time noise gain g(x[k], u[k])
    
    Key Differences from Continuous SDEs
    ------------------------------------
    - **Time Domain**:
      - Continuous: t ∈ ℝ (real-valued time)
      - Discrete: k ∈ ℤ (integer time steps)
    
    - **Deterministic Part**:
      - Continuous: _f_sym = dx/dt (drift, derivative)
      - Discrete: _f_sym = f(x[k], u[k]) (next state mean)
    
    - **Stochastic Part**:
      - Continuous: dW (Brownian motion, correlated increments)
      - Discrete: w[k] (IID standard normal, independent)
    
    - **Noise Scaling**:
      - Continuous: g(x,u) scales Brownian increments dW
      - Discrete: g(x,u) is the noise gain (no dt scaling)
    
    - **Ito vs Stratonovich**:
      - Continuous: Different interpretations matter
      - Discrete: No distinction (both equivalent)
    
    Implementation Strategy
    -----------------------
    Uses minimal inheritance from StochasticDynamicalSystem:
    - Reuses all drift/diffusion machinery
    - Reuses noise analysis (additive, multiplicative, etc.)
    - Reuses code generation and caching
    - Reuses backend management
    - Only adds discrete-time flag and overrides interpretation
    
    Future Refactoring Path
    -----------------------
    If needed, could refactor to:
    ```python
    class SymbolicStochasticSystem(ABC):
        # Common: drift, diffusion, noise analysis, code gen
        pass
    
    class ContinuousStochasticSystem(SymbolicStochasticSystem):
        # _f_sym = dx/dt, diffusion scales dW
        pass
    
    class DiscreteStochasticSystem(SymbolicStochasticSystem):
        # _f_sym = x[k+1], diffusion scales w[k]
        pass
    ```
    
    Parameters
    ----------
    *args : tuple
        Positional arguments passed to define_system()
    **kwargs : dict
        Keyword arguments passed to define_system()
    
    Attributes (Same as Parent)
    --------------------------
    state_vars : List[sp.Symbol]
        State variables x[k]
    control_vars : List[sp.Symbol]
        Control input variables u[k]
    _f_sym : sp.Matrix
        Deterministic part f(x[k], u[k])
    diffusion_expr : sp.Matrix
        Noise gain matrix g(x[k], u[k]), shape (nx, nw)
    parameters : Dict[sp.Symbol, float]
        System parameters
    
    Attributes (Discrete-Specific)
    ------------------------------
    _is_discrete : bool
        Flag indicating discrete-time system (always True)
    sde_type : SDEType
        Set to ITO (convention, Stratonovich equivalent in discrete time)
    
    Examples
    --------
    Discrete-time Ornstein-Uhlenbeck (AR(1) process):
    >>> class DiscreteOU(DiscreteStochasticSystem):
    ...     '''Discrete OU: x[k+1] = (1 - α*dt)*x[k] + u[k] + σ*w[k]'''
    ...     
    ...     def define_system(self, alpha=1.0, sigma=0.5, dt=0.1):
    ...         x = sp.symbols('x', real=True)
    ...         u = sp.symbols('u', real=True)
    ...         
    ...         alpha_sym = sp.symbols('alpha', positive=True)
    ...         sigma_sym = sp.symbols('sigma', positive=True)
    ...         dt_sym = sp.symbols('dt', positive=True)
    ...         
    ...         # Deterministic part (Euler discretization of OU)
    ...         self.state_vars = [x]
    ...         self.control_vars = [u]
    ...         self._f_sym = sp.Matrix([(1 - alpha_sym*dt_sym) * x + u])
    ...         self.parameters = {alpha_sym: alpha, sigma_sym: sigma, dt_sym: dt}
    ...         self.order = 1
    ...         
    ...         # Noise gain (additive)
    ...         self.diffusion_expr = sp.Matrix([[sigma_sym]])
    ...         self.sde_type = 'ito'
    
    >>> system = DiscreteOU(alpha=2.0, sigma=0.3, dt=0.1)
    >>> 
    >>> # Evaluate deterministic part
    >>> x_k = np.array([1.0])
    >>> u_k = np.array([0.0])
    >>> f = system(x_k, u_k)  # Mean of x[k+1]
    >>> 
    >>> # Evaluate noise gain
    >>> g = system.diffusion(x_k, u_k)  # Shape (1, 1)
    >>> 
    >>> # Simulate one step with noise
    >>> w_k = np.random.randn(1)
    >>> x_next = f + g @ w_k
    
    Discrete-time Kalman filter model:
    >>> class DiscreteKalmanModel(DiscreteStochasticSystem):
    ...     '''Linear discrete model with process noise'''
    ...     
    ...     def define_system(self, A, B, G):
    ...         '''
    ...         x[k+1] = A*x[k] + B*u[k] + G*w[k]
    ...         
    ...         A: State transition matrix
    ...         B: Control matrix
    ...         G: Process noise gain
    ...         '''
    ...         # This would need matrix-valued parameters
    ...         # Simplified example for 2D:
    ...         x1, x2 = sp.symbols('x1 x2', real=True)
    ...         u = sp.symbols('u', real=True)
    ...         
    ...         a11, a12, a21, a22 = sp.symbols('a11 a12 a21 a22', real=True)
    ...         b1, b2 = sp.symbols('b1 b2', real=True)
    ...         g11, g12, g21, g22 = sp.symbols('g11 g12 g21 g22', real=True)
    ...         
    ...         self.state_vars = [x1, x2]
    ...         self.control_vars = [u]
    ...         self._f_sym = sp.Matrix([
    ...             a11*x1 + a12*x2 + b1*u,
    ...             a21*x1 + a22*x2 + b2*u
    ...         ])
    ...         self.parameters = {
    ...             a11: A[0,0], a12: A[0,1],
    ...             a21: A[1,0], a22: A[1,1],
    ...             b1: B[0,0], b2: B[1,0],
    ...             g11: G[0,0], g12: G[0,1],
    ...             g21: G[1,0], g22: G[1,1]
    ...         }
    ...         self.order = 1
    ...         
    ...         self.diffusion_expr = sp.Matrix([
    ...             [g11, g12],
    ...             [g21, g22]
    ...         ])
    ...         self.sde_type = 'ito'
    
    ARMA(1,1) process:
    >>> class ARMA11(DiscreteStochasticSystem):
    ...     '''ARMA(1,1): x[k+1] = φ*x[k] + θ*ε[k-1] + ε[k]
    ...     
    ...     State: [x[k], ε[k-1]]
    ...     '''
    ...     
    ...     def define_system(self, phi=0.9, theta=0.5, sigma=0.1):
    ...         x, eps = sp.symbols('x eps', real=True)
    ...         phi_sym, theta_sym, sigma_sym = sp.symbols('phi theta sigma', real=True)
    ...         
    ...         # No control (autonomous time series)
    ...         self.state_vars = [x, eps]
    ...         self.control_vars = []
    ...         
    ...         # Deterministic part
    ...         # x[k+1] = φ*x[k] + θ*ε[k-1] + new noise
    ...         # ε[k] = new noise (will be updated by stochastic part)
    ...         self._f_sym = sp.Matrix([
    ...             phi_sym * x + theta_sym * eps,
    ...             0  # ε[k] comes from noise
    ...         ])
    ...         self.parameters = {phi_sym: phi, theta_sym: theta, sigma_sym: sigma}
    ...         self.order = 1
    ...         
    ...         # Noise affects both (x gets new noise, eps stores it)
    ...         self.diffusion_expr = sp.Matrix([
    ...             [sigma_sym],
    ...             [sigma_sym]
    ...         ])
    ...         self.sde_type = 'ito'
    
    >>> arma = ARMA11(phi=0.9, theta=0.5, sigma=0.1)
    >>> arma.is_additive_noise()
    True
    
    Notes
    -----
    - All parent methods work (drift, diffusion, linearization)
    - Noise analysis works (additive, multiplicative, etc.)
    - Can use same integrators/discretizers (just skip dt scaling)
    - sde_type distinction doesn't matter (set to 'ito' by convention)
    
    See Also
    --------
    DiscreteSymbolicSystem : Deterministic discrete systems
    StochasticDynamicalSystem : Continuous-time SDEs
    StochasticDiscreteSimulator : Simulator for discrete stochastic systems
    """
    
    def __init__(self, *args, **kwargs):
        """
        Initialize discrete-time stochastic system.
        
        Follows the same template method pattern as parent:
        1. Set discrete-time flag
        2. Call parent __init__ (which validates and initializes)
        
        The parent's validation and initialization work correctly
        because discrete stochastic systems use the same symbolic machinery.
        """
        # Mark as discrete-time system
        self._is_discrete = True
        
        # Call parent initialization (handles everything else)
        super().__init__(*args, **kwargs)
        
        # Override sde_type interpretation message
        # In discrete time, Ito and Stratonovich are equivalent
        if hasattr(self, 'sde_type'):
            self.sde_type = SDEType.ITO  # Convention for discrete
    
    def forward(self, x_k, u_k=None, backend=None):
        """
        Evaluate deterministic part: f(x[k], u[k]).
        
        For discrete stochastic systems:
            E[x[k+1] | x[k], u[k]] = f(x[k], u[k])
        
        The stochastic part is added separately via diffusion.
        
        Parameters
        ----------
        x_k : ArrayLike
            Current state x[k]
        u_k : Optional[ArrayLike]
            Current control u[k] (None for autonomous)
        backend : Optional[str]
            Backend selection
        
        Returns
        -------
        ArrayLike
            Deterministic next state f(x[k], u[k])
        
        Examples
        --------
        >>> # Mean dynamics (no noise)
        >>> f = system.forward(x_k, u_k)
        >>> 
        >>> # Full stochastic step
        >>> g = system.diffusion(x_k, u_k)
        >>> w_k = np.random.randn(nw)
        >>> x_next = f + g @ w_k
        
        Notes
        -----
        Unlike continuous SDEs where forward() returns dx/dt,
        this returns E[x[k+1]] - the expected next state.
        """
        return super().forward(x_k, u_k, backend)
    
    def step_stochastic(self, x_k, u_k=None, w_k=None, backend=None):
        """
        Compute full stochastic step: x[k+1] = f(x[k], u[k]) + g(x[k], u[k]) * w[k].
        
        Parameters
        ----------
        x_k : ArrayLike
            Current state (nx,) or (batch, nx)
        u_k : Optional[ArrayLike]
            Control (nu,) or (batch, nu), None for autonomous
        w_k : Optional[ArrayLike]
            Standard normal noise (nw,) or (batch, nw)
            If None, generated automatically
        backend : Optional[str]
            Backend selection
        
        Returns
        -------
        ArrayLike
            Next state x[k+1]
        
        Examples
        --------
        >>> # Automatic noise generation
        >>> x_next = system.step_stochastic(x_k, u_k)
        >>> 
        >>> # Custom noise (for reproducibility)
        >>> w_k = np.random.randn(nw)
        >>> x_next = system.step_stochastic(x_k, u_k, w=w_k)
        >>> 
        >>> # Deterministic (w=0)
        >>> x_next = system.step_stochastic(x_k, u_k, w=np.zeros(nw))
        
        Notes
        -----
        For batched inputs:
            x_k: (batch, nx)
            u_k: (batch, nu) or None
            w_k: (batch, nw) or None
            → x_next: (batch, nx)
        
        If w_k is None, noise is generated per sample in batch.
        """
        backend = backend or self._default_backend
        
        # Deterministic part: f(x[k], u[k])
        f = self.forward(x_k, u_k, backend)
        
        # Stochastic part: g(x[k], u[k])
        g = self.diffusion(x_k, u_k, backend)
        
        # Generate noise if not provided
        if w_k is None:
            # Detect if batched
            if self._is_batched(x_k):
                batch_size = x_k.shape[0]
                noise_shape = (batch_size, self.nw)
            else:
                noise_shape = (self.nw,)
            
            w_k = self._generate_noise(noise_shape, backend)
        
        # Full stochastic step: x[k+1] = f + g*w
        # Need to handle different g shapes:
        # - Single: g is (nx, nw)
        # - Batched additive: g is (nx, nw) (constant, not batched)
        # - Batched multiplicative: g is (batch, nx, nw)
        
        is_batched = self._is_batched(x_k)
        g_is_batched = (self._get_ndim(g) == 3)
        
        if is_batched and not g_is_batched:
            # Batched input but constant diffusion (additive noise)
            # g: (nx, nw), w: (batch, nw) → need to broadcast
            if backend == 'numpy':
                # Expand g to (1, nx, nw) then broadcast multiply
                # w: (batch, nw) → (batch, nw, 1)
                # Result: (batch, nx)
                stochastic_term = (g @ w_k.T).T  # (nx,nw)@(nw,batch) = (nx,batch) → T → (batch,nx)
            elif backend == 'torch':
                import torch
                # g: (nx, nw), w: (batch, nw)
                stochastic_term = (g @ w_k.T).T
            elif backend == 'jax':
                import jax.numpy as jnp
                stochastic_term = (g @ w_k.T).T
        
        elif is_batched and g_is_batched:
            # Both batched (multiplicative noise)
            # g: (batch, nx, nw), w: (batch, nw)
            if backend == 'numpy':
                # Use einsum for batched matmul: (batch,nx,nw) @ (batch,nw) → (batch,nx)
                stochastic_term = np.einsum('ijk,ik->ij', g, w_k)
            elif backend == 'torch':
                import torch
                # g: (batch, nx, nw), w: (batch, nw, 1) → (batch, nx, 1) → (batch, nx)
                stochastic_term = torch.bmm(g, w_k.unsqueeze(-1)).squeeze(-1)
            elif backend == 'jax':
                import jax.numpy as jnp
                stochastic_term = jnp.einsum('ijk,ik->ij', g, w_k)
        
        else:
            # Single trajectory (not batched)
            # f: (nx,), g: (nx, nw), w: (nw,)
            if backend == 'numpy':
                stochastic_term = g @ w_k
            elif backend == 'torch':
                stochastic_term = g @ w_k
            elif backend == 'jax':
                stochastic_term = g @ w_k
        
        return f + stochastic_term
    
    def _is_batched(self, x: Any) -> bool:
        """Check if input is batched (2D)."""
        if isinstance(x, np.ndarray):
            return x.ndim > 1
        elif hasattr(x, 'ndim'):
            return x.ndim > 1
        elif hasattr(x, 'shape'):
            return len(x.shape) > 1
        return False
    
    def _get_ndim(self, arr) -> int:
        """Get number of dimensions of array."""
        if isinstance(arr, np.ndarray):
            return arr.ndim
        elif hasattr(arr, 'ndim'):
            return arr.ndim
        elif hasattr(arr, 'shape'):
            return len(arr.shape)
        else:
            # Scalar
            return 0
    
    def _generate_noise(self, shape, backend):
        """Generate standard normal noise."""
        if backend == 'numpy':
            return np.random.randn(*shape)
        elif backend == 'torch':
            import torch
            return torch.randn(*shape)
        elif backend == 'jax':
            import jax
            # JAX needs explicit key management
            # For simplicity, use random key (users should set seed externally)
            key = jax.random.PRNGKey(np.random.randint(0, 2**31))
            return jax.random.normal(key, shape)
    
    def print_equations(self, simplify: bool = True):
        """
        Print symbolic equations in human-readable format.
        
        Overrides parent to use discrete-time notation.
        
        Parameters
        ----------
        simplify : bool
            If True, simplify expressions before printing
        
        Examples
        --------
        >>> system.print_equations()
        ======================================================================
        DiscreteOU (Discrete-Time Stochastic)
        ======================================================================
        State Variables: [x]
        Control Variables: [u]
        Dimensions: nx=1, nu=1, nw=1
        Noise Type: additive
        
        Deterministic Part: f(x[k], u[k])
          f_0 = 0.9*x + u
        
        Stochastic Part: g(x[k], u[k])
          g_0 = [0.5]
        
        Full Dynamics:
          x[k+1] = f(x[k], u[k]) + g(x[k], u[k]) * w[k]
          where w[k] ~ N(0, I)
        ======================================================================
        """
        print("=" * 70)
        print(f"{self.__class__.__name__} (Discrete-Time Stochastic)")
        print("=" * 70)
        print(f"State Variables: {self.state_vars}")
        print(f"Control Variables: {self.control_vars}")
        print(f"Dimensions: nx={self.nx}, nu={self.nu}, nw={self.nw}")
        print(f"Noise Type: {self.noise_characteristics.noise_type.value}")
        
        print("\nDeterministic Part: f(x[k], u[k])")
        for i, (var, expr) in enumerate(zip(self.state_vars, self._f_sym)):
            expr_sub = self.substitute_parameters(expr)
            if simplify:
                expr_sub = sp.simplify(expr_sub)
            print(f"  f_{i} = {expr_sub}")
        
        print("\nStochastic Part: g(x[k], u[k])")
        for i in range(self.nx):
            row_exprs = []
            for j in range(self.nw):
                expr = self.diffusion_expr[i, j]
                expr_sub = self.substitute_parameters(expr)
                if simplify:
                    expr_sub = sp.simplify(expr_sub)
                row_exprs.append(str(expr_sub))
            print(f"  g_{i} = [{', '.join(row_exprs)}]")
        
        print("\nFull Dynamics:")
        print("  x[k+1] = f(x[k], u[k]) + g(x[k], u[k]) * w[k]")
        print("  where w[k] ~ N(0, I)")
        
        if self._h_sym is not None:
            print("\nOutput: y[k] = h(x[k])")
            for i, expr in enumerate(self._h_sym):
                expr_sub = self.substitute_parameters(expr)
                if simplify:
                    expr_sub = sp.simplify(expr_sub)
                print(f"  y[{i}] = {expr_sub}")
        
        print("=" * 70)
    
    def get_config_dict(self) -> Dict:
        """
        Get system configuration including discrete and stochastic flags.
        
        Extends parent's config with discrete-time indicator.
        
        Returns
        -------
        dict
            Configuration dictionary
        
        Examples
        --------
        >>> config = system.get_config_dict()
        >>> config['is_discrete']
        True
        >>> config['is_stochastic']
        True
        """
        # Get parent's config (from StochasticDynamicalSystem)
        config = super().get_config_dict()
        
        # Add discrete flag
        config['is_discrete'] = True
        
        # Ensure is_stochastic is present (should be from parent)
        if 'is_stochastic' not in config:
            config['is_stochastic'] = True
        
        return config
    
    def __repr__(self) -> str:
        """
        Detailed string representation.
        
        Returns
        -------
        str
            Representation indicating discrete-time stochastic system
        
        Examples
        --------
        >>> repr(system)
        'DiscreteOU(nx=1, nu=1, nw=1, discrete=True, noise=additive, backend=numpy)'
        """
        return (
            f"{self.__class__.__name__}("
            f"nx={self.nx}, nu={self.nu}, nw={self.nw}, "
            f"discrete=True, "
            f"noise={self.noise_characteristics.noise_type.value}, "
            f"backend={self._default_backend})"
        )
    
    def __str__(self) -> str:
        """
        Human-readable string representation.
        
        Returns
        -------
        str
            Concise representation
        
        Examples
        --------
        >>> str(system)
        'DiscreteOU: 1 state, 1 control, 1 noise source, discrete-time (additive)'
        """
        return (
            f"{self.__class__.__name__}: "
            f"{self.nx} state{'s' if self.nx != 1 else ''}, "
            f"{self.nu} control{'s' if self.nu != 1 else ''}, "
            f"{self.nw} noise source{'s' if self.nw != 1 else ''}, "
            f"discrete-time ({self.noise_characteristics.noise_type.value})"
        )
