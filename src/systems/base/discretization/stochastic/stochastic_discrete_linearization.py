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
Stochastic Discrete Linearization - Cached Linearization for Stochastic Discrete Systems

Provides caching layer for numerical linearization of discrete-time stochastic systems,
including both drift and diffusion terms.

This class computes and caches discrete-time Jacobian matrices (Ad, Bd) for the drift
term and diffusion matrices (Gd) at equilibrium or operating points.

Use Cases
---------
- LQG (Linear Quadratic Gaussian) controller design
- Kalman filter design (need A, G, C at operating point)
- Stochastic stability analysis
- Certainty equivalence control with noise characterization
- Covariance propagation in stochastic systems

Architecture
-----------
Extends DiscreteLinearization to handle stochastic systems with diffusion terms.
Delegates to:
- DiscreteStochasticSystem.linearized_dynamics() for pure discrete stochastic
- Discretizer.linearize() for discretized continuous SDEs

The caching strategy uses equilibrium name or state/control hash as key,
storing both drift linearization (Ad, Bd) and diffusion linearization (Gd).

Output/Observation Linearization
--------------------------------
The observation/output linearization (Cd, Dd) is NOT cached by this class as it:
1. Depends only on state (not control), making separate caching less beneficial
2. Is typically much cheaper to compute than dynamics linearization
3. Is already available via system.linearized_observation(x, backend)

For complete Kalman filter design, retrieve Cd separately:

>>> Ad, Bd, Gd = lin.compute('origin')  # Cached
>>> Cd = system.linearized_observation(x_eq, backend='numpy')  # Direct call
>>> Dd = system.linearized_observation_control(x_eq, u_eq, backend='numpy')  # If needed
>>> 
>>> # Now have all matrices for Kalman filter
>>> Q = Gd @ Gd.T  # Process noise
>>> K = design_kalman_filter(Ad, Cd, Q, R)

This design keeps the linearization cache focused on the expensive dynamics
computations while observation matrices remain readily accessible.

Examples
--------
>>> # Pure discrete stochastic system
>>> system = DiscreteAR1(phi=0.9, sigma=0.1)
>>> lin = StochasticDiscreteLinearization(system)
>>> 
>>> # First call computes
>>> Ad, Bd, Gd = lin.compute(x_eq, u_eq)
>>> 
>>> # Second call uses cache
>>> Ad2, Bd2, Gd2 = lin.compute(x_eq, u_eq)
>>> # Returns same objects (cached)
>>> 
>>> # Discretized continuous SDE
>>> sde_system = OrnsteinUhlenbeck()
>>> discretizer = Discretizer(sde_system, dt=0.01, method='euler-maruyama')
>>> lin = StochasticDiscreteLinearization(sde_system, discretizer=discretizer)
>>> 
>>> # Linearize at equilibrium
>>> system.add_equilibrium('origin', x_eq, u_eq)
>>> Ad, Bd, Gd = lin.compute('origin')
>>> 
>>> # Design Kalman filter (get Cd separately)
>>> Cd = system.linearized_observation(x_eq, backend='numpy')
>>> Q = Gd @ Gd.T  # Process noise covariance
>>> K = design_kalman_filter(Ad, Cd, Q, R)
"""

from typing import Union, Tuple, Optional, Dict, Any
import numpy as np
import warnings

# Conditional imports
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

from src.systems.base.discretization.discrete_linearization import DiscreteLinearization

# Type alias
ArrayLike = Union[np.ndarray, "torch.Tensor", "jnp.ndarray"]


class StochasticDiscreteLinearization(DiscreteLinearization):
    """
    Caches numerical linearization for discrete-time stochastic systems.
    
    Extends DiscreteLinearization to handle stochastic systems with diffusion terms.
    Computes and caches both drift linearization (Ad, Bd) and diffusion linearization (Gd).
    
    For discrete stochastic systems:
        x[k+1] = f(x[k], u[k]) + G(x[k]) * w[k]
    
    Linearization gives:
        δx[k+1] ≈ Ad*δx[k] + Bd*δu[k] + Gd*w[k]
    
    Where:
        Ad = ∂f/∂x at (x_eq, u_eq)
        Bd = ∂f/∂u at (x_eq, u_eq)
        Gd = G(x_eq)
    
    Attributes
    ----------
    system : Union[DiscreteStochasticSystem, StochasticDynamicalSystem]
        Stochastic system to linearize
    discretizer : Optional[Discretizer]
        Discretizer for continuous SDEs
    backend : str
        Backend for numerical arrays
    
    Examples
    --------
    >>> system = DiscreteAR1(phi=0.9, sigma=0.1)
    >>> lin = StochasticDiscreteLinearization(system)
    >>> 
    >>> # Compute at equilibrium
    >>> Ad, Bd, Gd = lin.compute(x_eq, u_eq)
    >>> 
    >>> # Process noise covariance
    >>> Q = Gd @ Gd.T
    >>> 
    >>> # Second call returns cached
    >>> Ad2, Bd2, Gd2 = lin.compute(x_eq, u_eq)
    >>> assert Ad is Ad2  # Same object (cached)
    """
    
    def __init__(
        self,
        system: Union['DiscreteStochasticSystem', 'StochasticDynamicalSystem'],
        discretizer: Optional['Discretizer'] = None,
    ):
        """
        Initialize stochastic discrete linearization cache.
        
        Parameters
        ----------
        system : Union[DiscreteStochasticSystem, StochasticDynamicalSystem]
            Stochastic system to linearize
        discretizer : Optional[Discretizer]
            Required for continuous SDEs, None for discrete stochastic
        
        Raises
        ------
        TypeError
            If continuous system without discretizer
        ValueError
            If system is not stochastic
        """
        # Validate system is stochastic
        is_stochastic = False
        try:
            from src.systems.base.discrete_stochastic_system import DiscreteStochasticSystem
            if isinstance(system, DiscreteStochasticSystem):
                is_stochastic = True
        except ImportError:
            pass
        
        try:
            from src.systems.base.stochastic_dynamical_system import StochasticDynamicalSystem
            if isinstance(system, StochasticDynamicalSystem):
                is_stochastic = True
        except ImportError:
            pass
        
        if not is_stochastic:
            raise ValueError(
                f"StochasticDiscreteLinearization requires a stochastic system. "
                f"{system.__class__.__name__} is not a DiscreteStochasticSystem or StochasticDynamicalSystem. "
                f"Use DiscreteLinearization for deterministic systems."
            )
        
        # Initialize parent class (this handles all the drift linearization)
        super().__init__(system, discretizer)
        
        # Override cache to store (Ad, Bd, Gd) tuples
        self._cache: Dict[str, Tuple[ArrayLike, ArrayLike, ArrayLike]] = {}
    
    # ========================================================================
    # Main Linearization API
    # ========================================================================
    
    def compute(
        self,
        x_eq: Union[ArrayLike, str],
        u_eq: Optional[ArrayLike] = None,
        method: str = 'euler',
        use_cache: bool = True,
    ) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
        """
        Compute discrete-time stochastic linearization: (Ad, Bd, Gd).
        
        Parameters
        ----------
        x_eq : Union[ArrayLike, str]
            Equilibrium state (nx,) or equilibrium name
        u_eq : Optional[ArrayLike]
            Equilibrium control (nu,)
            Ignored if x_eq is string (equilibrium name)
        method : str
            Discretization method for continuous SDEs
            ('euler', 'euler-maruyama', 'exact')
            Ignored for pure discrete stochastic systems
        use_cache : bool
            If True, use cached result if available
        
        Returns
        -------
        Ad : ArrayLike
            Discrete-time state matrix (nx, nx)
        Bd : ArrayLike
            Discrete-time control matrix (nx, nu)
        Gd : ArrayLike
            Discrete-time diffusion matrix (nx, nw)
        
        Examples
        --------
        >>> # At specific point
        >>> Ad, Bd, Gd = lin.compute(
        ...     x_eq=np.array([0.0]),
        ...     u_eq=np.array([0.0])
        ... )
        >>> 
        >>> # Process noise covariance
        >>> Q = Gd @ Gd.T
        >>> 
        >>> # At named equilibrium
        >>> system.add_equilibrium('origin', x_eq, u_eq)
        >>> Ad, Bd, Gd = lin.compute('origin')
        >>> 
        >>> # Force recomputation
        >>> Ad, Bd, Gd = lin.compute(x_eq, u_eq, use_cache=False)
        
        Notes
        -----
        For pure discrete stochastic systems, method parameter is ignored.
        For discretized continuous SDEs, method determines how continuous-time
        drift and diffusion are converted to discrete-time.
        """
        # Handle equilibrium name
        if isinstance(x_eq, str):
            equilibrium_name = x_eq
            x_eq, u_eq = self.system.equilibria.get_both(equilibrium_name, backend=self.backend)
            cache_key = equilibrium_name
        else:
            # Generate cache key from state/control
            cache_key = self._generate_cache_key(x_eq, u_eq, method)
        
        # Check cache
        if use_cache and cache_key in self._cache:
            self._stats['cache_hits'] += 1
            return self._cache[cache_key]
        
        # Compute drift linearization (Ad, Bd)
        if self.discretizer is not None:
            # Discretized continuous SDE
            Ad, Bd = self.discretizer.linearize(x_eq, u_eq, method=method)
        else:
            # Pure discrete stochastic system
            Ad, Bd = self.system.linearized_dynamics(x_eq, u_eq, backend=self.backend)
        
        # Compute diffusion linearization (Gd)
        # Handle autonomous systems (nu=0) by passing u_eq=None
        if self.system.nu == 0:
            Gd = self._compute_diffusion_linearization(x_eq, None, method)
        else:
            Gd = self._compute_diffusion_linearization(x_eq, u_eq, method)
        
        # Cache it
        self._cache[cache_key] = (Ad, Bd, Gd)
        self._stats['computes'] += 1
        
        return Ad, Bd, Gd
    
    def _compute_diffusion_linearization(
        self,
        x_eq: ArrayLike,
        u_eq: Optional[ArrayLike],
        method: str = 'euler'
    ) -> ArrayLike:
        """
        Compute diffusion matrix linearization.
        
        For discrete systems: Gd = G(x_eq, u_eq) or G(x_eq) for autonomous
        For continuous SDEs: depends on discretization method
        
        Parameters
        ----------
        x_eq : ArrayLike
            Equilibrium state
        u_eq : Optional[ArrayLike]
            Equilibrium control (None for autonomous systems with nu=0)
        method : str
            Discretization method
        
        Returns
        -------
        Gd : ArrayLike
            Discrete-time diffusion matrix (nx, nw)
        
        Notes
        -----
        For autonomous systems (nu=0), u_eq should be None and will be handled
        correctly by the underlying system.diffusion() method.
        """
        if self.discretizer is not None:
            # Discretized continuous SDE
            # Get continuous-time diffusion matrix
            # Pass u_eq=None for autonomous systems (nu=0)
            G_continuous = self.system.diffusion(x_eq, u_eq, backend=self.backend)
            
            dt = self.discretizer.dt
            
            if method in ['euler', 'euler-maruyama']:
                # Euler-Maruyama: x[k+1] = x[k] + f*dt + G*sqrt(dt)*w
                if TORCH_AVAILABLE and isinstance(G_continuous, torch.Tensor):
                    Gd = G_continuous * torch.sqrt(torch.tensor(dt, dtype=G_continuous.dtype))
                elif JAX_AVAILABLE and isinstance(G_continuous, jnp.ndarray):
                    Gd = G_continuous * jnp.sqrt(dt)
                else:
                    Gd = G_continuous * np.sqrt(dt)
            else:
                # For other methods, use sqrt(dt) scaling as approximation
                warnings.warn(
                    f"Diffusion discretization for method '{method}' not fully implemented. "
                    f"Using Euler-Maruyama approximation.",
                    UserWarning
                )
                if TORCH_AVAILABLE and isinstance(G_continuous, torch.Tensor):
                    Gd = G_continuous * torch.sqrt(torch.tensor(dt, dtype=G_continuous.dtype))
                elif JAX_AVAILABLE and isinstance(G_continuous, jnp.ndarray):
                    Gd = G_continuous * jnp.sqrt(dt)
                else:
                    Gd = G_continuous * np.sqrt(dt)
        else:
            # Pure discrete stochastic system
            # Diffusion is already in discrete-time form
            # Pass u_eq=None for autonomous systems (nu=0)
            Gd = self.system.diffusion(x_eq, u_eq, backend=self.backend)
        
        return Gd
    
    def compute_at_equilibria(
        self,
        equilibrium_names: Optional[list] = None,
        method: str = 'euler',
    ) -> Dict[str, Tuple[ArrayLike, ArrayLike, ArrayLike]]:
        """
        Compute stochastic linearization at all (or specified) equilibria.
        
        Useful for:
        - Designing multiple Kalman filters at operating points
        - Stochastic stability analysis at multiple equilibria
        - Gain scheduling for stochastic control
        
        Parameters
        ----------
        equilibrium_names : Optional[list]
            List of equilibrium names to linearize at
            If None, uses all equilibria
        method : str
            Discretization method
        
        Returns
        -------
        dict
            Mapping equilibrium_name -> (Ad, Bd, Gd)
        
        Examples
        --------
        >>> # Add multiple equilibria
        >>> system.add_equilibrium('low', x_low, u_low)
        >>> system.add_equilibrium('high', x_high, u_high)
        >>> 
        >>> # Linearize at all
        >>> linearizations = lin.compute_at_equilibria()
        >>> Ad_low, Bd_low, Gd_low = linearizations['low']
        >>> Ad_high, Bd_high, Gd_high = linearizations['high']
        >>> 
        >>> # Design Kalman filters
        >>> Q_low = Gd_low @ Gd_low.T
        >>> Q_high = Gd_high @ Gd_high.T
        """
        if equilibrium_names is None:
            equilibrium_names = self.system.list_equilibria()
        
        linearizations = {}
        
        for name in equilibrium_names:
            Ad, Bd, Gd = self.compute(name, method=method)
            linearizations[name] = (Ad, Bd, Gd)
        
        return linearizations
    
    # ========================================================================
    # Stochastic-Specific Analysis
    # ========================================================================
    
    def compute_process_noise_covariance(
        self,
        x_eq: Union[ArrayLike, str],
        u_eq: Optional[ArrayLike] = None,
        method: str = 'euler',
        noise_covariance: Optional[ArrayLike] = None,
    ) -> ArrayLike:
        """
        Compute discrete-time process noise covariance matrix.
        
        Q = E[Gd*w*w'*Gd'] = Gd * Qw * Gd'
        
        Parameters
        ----------
        x_eq : Union[ArrayLike, str]
            Equilibrium state or name
        u_eq : Optional[ArrayLike]
            Equilibrium control
        method : str
            Discretization method
        noise_covariance : Optional[ArrayLike]
            Noise covariance Qw (nw, nw)
            If None, assumes identity (white noise)
        
        Returns
        -------
        Q : ArrayLike
            Process noise covariance (nx, nx)
        
        Examples
        --------
        >>> # White noise (identity covariance)
        >>> Q = lin.compute_process_noise_covariance('origin')
        >>> 
        >>> # Colored noise
        >>> Qw = np.diag([0.1, 0.05])
        >>> Q = lin.compute_process_noise_covariance('origin', noise_covariance=Qw)
        >>> 
        >>> # Use in Kalman filter design
        >>> K = design_kalman_filter(Ad, Cd, Q, R)
        """
        Ad, Bd, Gd = self.compute(x_eq, u_eq, method)
        
        # Convert to numpy for computation
        if TORCH_AVAILABLE and isinstance(Gd, torch.Tensor):
            Gd_np = Gd.detach().cpu().numpy()
        elif JAX_AVAILABLE and isinstance(Gd, jnp.ndarray):
            Gd_np = np.array(Gd)
        else:
            Gd_np = Gd
        
        # Compute Q = Gd * Qw * Gd'
        if noise_covariance is None:
            # White noise: Q = Gd * Gd'
            Q = Gd_np @ Gd_np.T
        else:
            # Colored noise: Q = Gd * Qw * Gd'
            if TORCH_AVAILABLE and isinstance(noise_covariance, torch.Tensor):
                Qw_np = noise_covariance.detach().cpu().numpy()
            elif JAX_AVAILABLE and isinstance(noise_covariance, jnp.ndarray):
                Qw_np = np.array(noise_covariance)
            else:
                Qw_np = noise_covariance
            
            Q = Gd_np @ Qw_np @ Gd_np.T
        
        # Convert back to original backend
        if TORCH_AVAILABLE and isinstance(Gd, torch.Tensor):
            return torch.tensor(Q, dtype=Gd.dtype, device=Gd.device)
        elif JAX_AVAILABLE and isinstance(Gd, jnp.ndarray):
            return jnp.array(Q)
        else:
            return Q
    
    def check_mean_square_stability(
        self,
        x_eq: Union[ArrayLike, str],
        u_eq: Optional[ArrayLike] = None,
        method: str = 'euler',
        noise_covariance: Optional[ArrayLike] = None,
    ) -> Dict[str, Any]:
        """
        Check mean-square stability of linearized stochastic system.
        
        For linear stochastic system:
            x[k+1] = Ad*x[k] + Gd*w[k]
        
        Mean-square stable if all eigenvalues of Ad satisfy |λ| < 1.
        
        Parameters
        ----------
        x_eq : Union[ArrayLike, str]
            Equilibrium
        u_eq : Optional[ArrayLike]
            Control
        method : str
            Method
        noise_covariance : Optional[ArrayLike]
            Not used for stability check, but included for consistency
        
        Returns
        -------
        dict
            Stability analysis including:
            - 'is_ms_stable': True if mean-square stable
            - 'eigenvalues': Eigenvalues of Ad
            - 'max_magnitude': Maximum eigenvalue magnitude
            - 'steady_state_covariance': Steady-state state covariance (if stable)
        
        Examples
        --------
        >>> stability = lin.check_mean_square_stability('origin')
        >>> 
        >>> if stability['is_ms_stable']:
        ...     P_ss = stability['steady_state_covariance']
        ...     print(f"Steady-state variance: {np.trace(P_ss):.3f}")
        """
        # Get drift linearization for stability check
        stability_info = super().check_stability(x_eq, u_eq, method)
        
        # Rename for clarity
        stability_info['is_ms_stable'] = stability_info['is_stable']
        
        # Compute steady-state covariance if stable
        if stability_info['is_ms_stable']:
            Ad, Bd, Gd = self.compute(x_eq, u_eq, method)
            Q = self.compute_process_noise_covariance(x_eq, u_eq, method, noise_covariance)
            
            # Convert to numpy
            if TORCH_AVAILABLE and isinstance(Ad, torch.Tensor):
                Ad_np = Ad.detach().cpu().numpy()
                Q_np = Q.detach().cpu().numpy()
            elif JAX_AVAILABLE and isinstance(Ad, jnp.ndarray):
                Ad_np = np.array(Ad)
                Q_np = np.array(Q)
            else:
                Ad_np = Ad
                Q_np = Q
            
            # Solve discrete-time Lyapunov equation: P = Ad*P*Ad' + Q
            try:
                from scipy.linalg import solve_discrete_lyapunov
                P_ss = solve_discrete_lyapunov(Ad_np, Q_np)
                stability_info['steady_state_covariance'] = P_ss
            except Exception as e:
                warnings.warn(f"Could not compute steady-state covariance: {e}")
                stability_info['steady_state_covariance'] = None
        else:
            stability_info['steady_state_covariance'] = None
        
        return stability_info
    
    # ========================================================================
    # Utility Methods
    # ========================================================================
    
    def precompute_at_grid(
        self,
        x_grid: ArrayLike,
        u_grid: ArrayLike,
        method: str = 'euler'
    ):
        """
        Precompute stochastic linearizations at grid of operating points.
        
        Useful for gain-scheduled Kalman filtering or stochastic MPC.
        
        Parameters
        ----------
        x_grid : ArrayLike
            Grid of state points, shape (n_points, nx)
        u_grid : ArrayLike
            Grid of control points, shape (n_points, nu)
        method : str
            Discretization method
        
        Examples
        --------
        >>> # Create operating point grid
        >>> x_grid = np.linspace(-1, 1, 10).reshape(-1, 1)
        >>> u_grid = np.zeros((10, 1))
        >>> 
        >>> # Precompute
        >>> lin.precompute_at_grid(x_grid, u_grid)
        >>> 
        >>> # Later lookups are cached
        >>> Ad, Bd, Gd = lin.compute(x_grid[0], u_grid[0])  # Cache hit!
        """
        n_points = x_grid.shape[0]
        
        for i in range(n_points):
            x = x_grid[i]
            u = u_grid[i]
            self.compute(x, u, method=method, use_cache=True)
        
        print(f"Precomputed {n_points} stochastic linearizations")
    
    def get_cached(
        self,
        x_eq: Union[ArrayLike, str],
        u_eq: Optional[ArrayLike] = None,
        method: str = 'euler'
    ) -> Optional[Tuple[ArrayLike, ArrayLike, ArrayLike]]:
        """
        Get cached stochastic linearization without computing.
        
        Parameters
        ----------
        x_eq : Union[ArrayLike, str]
            State or equilibrium name
        u_eq : Optional[ArrayLike]
            Control
        method : str
            Method
        
        Returns
        -------
        Optional[Tuple[ArrayLike, ArrayLike, ArrayLike]]
            (Ad, Bd, Gd) if cached, None otherwise
        
        Examples
        --------
        >>> cached = lin.get_cached('origin')
        >>> if cached is not None:
        ...     Ad, Bd, Gd = cached
        ... else:
        ...     Ad, Bd, Gd = lin.compute('origin')
        """
        if isinstance(x_eq, str):
            cache_key = x_eq
        else:
            cache_key = self._generate_cache_key(x_eq, u_eq, method)
        
        return self._cache.get(cache_key)
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get stochastic linearization cache information.
        
        Returns
        -------
        dict
            Information about cache state
        
        Examples
        --------
        >>> info = lin.get_info()
        >>> print(info)
        """
        info = super().get_info()
        info['linearization_type'] = 'stochastic'
        return info
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        discretizer_str = f", discretizer={self.discretizer.method}" if self.discretizer else ""
        return (
            f"StochasticDiscreteLinearization("
            f"system={self.system.__class__.__name__}"
            f"{discretizer_str}, "
            f"cache_size={len(self._cache)})"
        )
    
    def __str__(self) -> str:
        """Human-readable string."""
        sys_type = "discrete-stochastic" if self.discretizer is None else "discretized-SDE"
        return (
            f"StochasticDiscreteLinearization({self.system.__class__.__name__}, "
            f"{sys_type}, {len(self._cache)} cached)"
        )