"""
DiffEqPySDEIntegrator: Julia-based SDE integration using DifferentialEquations.jl

This module provides access to Julia's powerful SDE solver ecosystem through the
diffeqpy Python interface. Julia's DifferentialEquations.jl is considered the
gold standard for SDE integration with:
- Extensive algorithm selection (40+ SDE solvers)
- Superior performance for stiff SDEs
- Advanced noise processes and adaptivity
- Automatic stiffness detection

Supports both Ito and Stratonovich interpretations, controlled and autonomous systems.

Mathematical Form
-----------------
Stochastic differential equations:

    dx = f(x, u, t)dt + g(x, u, t)dW

Available Algorithms
-------------------
**Recommended General Purpose:**
- EM (Euler-Maruyama): Order 0.5 strong, 1.0 weak - fast and robust
- LambaEM: Euler-Maruyama with lambda step size control
- SRIW1: Order 1.5 strong, 2.0 weak - better accuracy

**High Accuracy (Stochastic Runge-Kutta):**
- SRA1: Order 2.0 weak for diagonal noise
- SRA3: Order 2.0 weak for additive noise
- SOSRA: Order 2.0 weak for scalar noise
- SRI: Roessler SRI algorithms (various orders)

**Adaptive Methods:**
- RKMil: Runge-Kutta Milstein with adaptivity
- RKMilCommute: For commutative noise
- AutoEM: Automatic switching between methods

**Specialized:**
- ImplicitEM: Implicit Euler-Maruyama for stiff drift
- ImplicitRKMil: Implicit Runge-Kutta Milstein
- SKenCarp: Stochastic Kennedy-Carpenter IMEX methods

Installation
-----------
Requires Julia and diffeqpy:

1. Install Julia from https://julialang.org/downloads/
2. Install DifferentialEquations.jl:
   ```julia
   using Pkg
   Pkg.add("DifferentialEquations")
   ```
3. Install Python package:
   ```bash
   pip install diffeqpy
   ```
4. Setup (one time):
   ```python
   from diffeqpy import install
   install()
   ```

Examples
--------
>>> # Ornstein-Uhlenbeck process (autonomous)
>>> integrator = DiffEqPySDEIntegrator(
...     sde_system,
...     dt=0.01,
...     algorithm='EM',
...     backend='numpy'
... )
>>> 
>>> result = integrator.integrate(
...     x0=np.array([1.0]),
...     u_func=lambda t, x: None,
...     t_span=(0.0, 10.0)
... )
>>> 
>>> # Controlled SDE with high accuracy
>>> integrator = DiffEqPySDEIntegrator(
...     controlled_sde,
...     dt=0.001,
...     algorithm='SRIW1',
...     rtol=1e-6,
...     atol=1e-8
... )
>>> 
>>> result = integrator.integrate(
...     x0=np.array([1.0, 0.0]),
...     u_func=lambda t, x: -K @ x,
...     t_span=(0.0, 10.0)
... )
"""

from typing import Optional, Tuple, Callable, Dict, Any, List
import numpy as np

from src.systems.base.numerical_integration.stochastic.sde_integrator_base import (
    SDEIntegratorBase,
    SDEType,
    ConvergenceType,
    SDEIntegrationResult,
    StepMode,
    ArrayLike
)


class DiffEqPySDEIntegrator(SDEIntegratorBase):
    """
    Julia-based SDE integrator using DifferentialEquations.jl via diffeqpy.
    
    Provides access to Julia's extensive SDE solver ecosystem with superior
    performance and accuracy compared to pure Python implementations.
    
    Parameters
    ----------
    sde_system : StochasticDynamicalSystem
        SDE system to integrate (controlled or autonomous)
    dt : Optional[float]
        Time step (initial guess for adaptive, fixed for non-adaptive)
    step_mode : StepMode
        FIXED or ADAPTIVE stepping (most Julia SDE solvers are adaptive)
    backend : str
        Must be 'numpy' (Julia returns NumPy arrays via diffeqpy)
    algorithm : str
        Julia SDE algorithm name (default: 'EM')
        See list_algorithms() for available options
    sde_type : Optional[SDEType]
        SDE interpretation (None = use system's type)
    convergence_type : ConvergenceType
        Strong or weak convergence
    seed : Optional[int]
        Random seed for reproducibility
    **options
        Additional options:
        - rtol : float (default: 1e-3) - Relative tolerance
        - atol : float (default: 1e-6) - Absolute tolerance
        - save_everystep : bool (default: True) - Save at every step
        - dense : bool (default: False) - Dense output interpolation
        - callback : Callable - Julia callback function
        - noise_rate_prototype : array - Noise matrix prototype
    
    Raises
    ------
    ImportError
        If diffeqpy is not installed
    RuntimeError
        If Julia DifferentialEquations.jl is not available
    
    Notes
    -----
    - Backend must be 'numpy' (Julia/Python bridge uses NumPy)
    - Julia's automatic stiffness detection works well for most problems
    - For very stiff drift, use implicit methods (ImplicitEM, SKenCarp)
    - Julia handles noise generation internally with high-quality RNG
    
    Examples
    --------
    >>> # Basic usage with Euler-Maruyama
    >>> integrator = DiffEqPySDEIntegrator(
    ...     sde_system,
    ...     dt=0.01,
    ...     algorithm='EM'
    ... )
    >>> 
    >>> # High accuracy adaptive solver
    >>> integrator = DiffEqPySDEIntegrator(
    ...     sde_system,
    ...     dt=0.001,
    ...     algorithm='SRIW1',
    ...     rtol=1e-6,
    ...     atol=1e-8
    ... )
    >>> 
    >>> # Stiff drift with implicit solver
    >>> integrator = DiffEqPySDEIntegrator(
    ...     stiff_sde,
    ...     algorithm='ImplicitEM',
    ...     dt=0.01
    ... )
    """
    
    def __init__(
        self,
        sde_system,
        dt: Optional[float] = 0.01,
        step_mode: StepMode = StepMode.ADAPTIVE,
        backend: str = 'numpy',
        algorithm: str = 'EM',
        sde_type: Optional[SDEType] = None,
        convergence_type: ConvergenceType = ConvergenceType.STRONG,
        seed: Optional[int] = None,
        **options
    ):
        """Initialize DiffEqPy SDE integrator."""
        
        # Validate backend
        if backend != 'numpy':
            raise ValueError(
                "DiffEqPySDEIntegrator requires backend='numpy'. "
                "Julia/Python bridge uses NumPy arrays."
            )
        
        # Initialize base class
        super().__init__(
            sde_system,
            dt=dt,
            step_mode=step_mode,
            backend=backend,
            sde_type=sde_type,
            convergence_type=convergence_type,
            seed=seed,
            **options
        )
        
        self.algorithm = algorithm
        self._integrator_name = f"Julia-{algorithm}"
        
        # Try to import diffeqpy
        try:
            from diffeqpy import de
            self.de = de
        except ImportError as e:
            raise ImportError(
                "DiffEqPySDEIntegrator requires diffeqpy.\n\n"
                "Installation steps:\n"
                "1. Install Julia from https://julialang.org/downloads/\n"
                "2. Install DifferentialEquations.jl:\n"
                "   julia> using Pkg\n"
                "   julia> Pkg.add('DifferentialEquations')\n"
                "3. Install Python package:\n"
                "   pip install diffeqpy\n"
                "4. Setup (one time):\n"
                "   from diffeqpy import install; install()\n"
            ) from e
        
        # Validate algorithm availability
        available = self._get_available_algorithms()
        if algorithm not in available:
            raise ValueError(
                f"Unknown Julia SDE algorithm '{algorithm}'.\n"
                f"Available: {available[:10]}...\n"
                f"Use list_algorithms() for complete list."
            )
        
        # Get algorithm constructor
        self._algorithm_constructor = self._get_algorithm(algorithm)
        
        # Store options
        self.rtol = options.get('rtol', 1e-3)
        self.atol = options.get('atol', 1e-6)
        self.save_everystep = options.get('save_everystep', True)
        self.dense = options.get('dense', False)
    
    def validate_julia_setup(self):
        """
        Validate that Julia and DifferentialEquations.jl are properly set up.
        
        Returns
        -------
        bool
            True if setup is valid
            
        Raises
        ------
        RuntimeError
            If validation fails with details
            
        Examples
        --------
        >>> integrator = DiffEqPySDEIntegrator(sde_system, algorithm='EM')
        >>> integrator.validate_julia_setup()  # Raises if problems
        True
        """
        try:
            # Test basic Julia call
            result = self.de.eval("1 + 1")
            assert result == 2, "Basic Julia eval failed"
            
            # Check that SDEProblem exists
            assert hasattr(self.de, 'SDEProblem'), "SDEProblem not found in Julia"
            
            # Check that algorithm exists
            assert hasattr(self.de, self.algorithm), \
                f"Algorithm {self.algorithm} not found in DifferentialEquations.jl"
            
            # Try creating and solving a simple SDE problem
            def simple_drift(u, p, t):
                return np.array([-u[0]], dtype=np.float64)
            
            def simple_diffusion(u, p, t):
                return np.array([[0.1]], dtype=np.float64)
            
            test_problem = self.de.SDEProblem(
                simple_drift,
                simple_diffusion,
                np.array([1.0], dtype=np.float64),
                (0.0, 0.1),
                noise_rate_prototype=np.array([[0.1]], dtype=np.float64)
            )
            
            # Try solving
            test_alg = getattr(self.de, self.algorithm)()
            test_sol = self.de.solve(test_problem, test_alg, dt=0.01)
            
            assert hasattr(test_sol, 't'), "Solution missing time points"
            assert hasattr(test_sol, 'u'), "Solution missing states"
            assert len(test_sol.t) > 1, f"Solution has only {len(test_sol.t)} time point(s)"
            assert len(test_sol.u) > 1, f"Solution has only {len(test_sol.u)} state(s)"
            
            return True
            
        except Exception as e:
            import traceback
            raise RuntimeError(
                f"Julia/DiffEqPy validation failed:\n{str(e)}\n"
                f"{traceback.format_exc()}\n\n"
                f"Possible solutions:\n"
                f"1. Ensure Julia is installed and in PATH\n"
                f"2. Reinstall DifferentialEquations.jl:\n"
                f"   julia> using Pkg; Pkg.add('DifferentialEquations')\n"
                f"3. Reinstall diffeqpy:\n"
                f"   pip uninstall diffeqpy && pip install diffeqpy\n"
                f"4. Run setup:\n"
                f"   from diffeqpy import install; install()\n"
                f"5. Test Julia directly:\n"
                f"   julia> using DifferentialEquations\n"
                f"   julia> prob = SDEProblem(...)\n"
                f"   julia> solve(prob, EM())"
            ) from e
    
    @property
    def name(self) -> str:
        """Return integrator name."""
        mode_str = "Adaptive" if self.step_mode == StepMode.ADAPTIVE else "Fixed"
        return f"{self._integrator_name} ({mode_str})"
    
    def _get_available_algorithms(self) -> List[str]:
        """Get list of available Julia SDE algorithms."""
        # Common SDE algorithms in DifferentialEquations.jl
        return [
            # Euler-Maruyama family
            'EM', 'LambaEM', 'EulerHeun',
            # Stochastic Runge-Kutta
            'SRIW1', 'SRIW2', 'SOSRI', 'SOSRI2',
            'SRA', 'SRA1', 'SRA2', 'SRA3',
            'SOSRA', 'SOSRA2',
            # Milstein family
            'RKMil', 'RKMilCommute', 'RKMilGeneral',
            # Implicit methods
            'ImplicitEM', 'ImplicitEulerHeun', 'ImplicitRKMil',
            # IMEX methods
            'SKenCarp',
            # Rossler SRI
            'SRI', 'SRIW1Optimized', 'SRIW2Optimized',
            # Adaptive
            'AutoEM',
        ]
    
    def _get_algorithm(self, algorithm: str):
        """Get Julia algorithm constructor."""
        # Return the algorithm constructor from Julia
        # The actual algorithm object is created when needed
        return getattr(self.de, algorithm, None)
    
    def _setup_julia_problem(
        self,
        x0: np.ndarray,
        u_func: Callable,
        t_span: Tuple[float, float]
    ):
        """
        Setup Julia SDE problem.
        
        Creates drift and diffusion functions compatible with Julia's
        DifferentialEquations.jl interface.
        
        Parameters
        ----------
        x0 : np.ndarray
            Initial state
        u_func : Callable
            Control policy (or None for autonomous)
        t_span : Tuple[float, float]
            Time span (t0, tf)
            
        Returns
        -------
        SDEProblem
            Julia SDE problem object
        """
        t0, tf = t_span
        
        # Define drift function for Julia: f(u, p, t)
        # Julia convention: u = state, p = parameters, t = time
        def drift_func_julia(u, p, t):
            """Drift function: dx/dt = f(x, u, t)"""
            # Convert Julia array to NumPy
            x = np.array(u)
            
            # Get control
            control = u_func(float(t), x)
            
            # Evaluate drift
            dx = self.sde_system.drift(x, control, backend='numpy')
            
            # Track statistics
            self._stats['total_fev'] += 1
            
            return dx
        
        # Define diffusion function for Julia: g(u, p, t)
        def diffusion_func_julia(u, p, t):
            """Diffusion function: dW = g(x, u, t) * dW"""
            # Convert Julia array to NumPy
            x = np.array(u)
            
            # Get control
            control = u_func(float(t), x)
            
            # Evaluate diffusion
            g = self.sde_system.diffusion(x, control, backend='numpy')
            
            # Track statistics
            self._stats['diffusion_evals'] += 1
            
            return g
        
        # Create noise rate prototype (tells Julia the diffusion matrix shape)
        # This is g(x0, u0, t0) evaluated at initial conditions
        control_0 = u_func(t0, x0)
        g0 = self.sde_system.diffusion(x0, control_0, backend='numpy')
        
        # Create SDE problem
        # Julia expects: SDEProblem(f, g, u0, tspan; noise_rate_prototype)
        problem = self.de.SDEProblem(
            drift_func_julia,
            diffusion_func_julia,
            x0,
            (t0, tf),
            noise_rate_prototype=g0
        )
        
        return problem
    
    def step(
        self,
        x: ArrayLike,
        u: Optional[ArrayLike] = None,
        dt: Optional[float] = None,
        dW: Optional[ArrayLike] = None
    ) -> ArrayLike:
        """
        Take one SDE integration step.
        
        Note: Julia's solvers are optimized for full trajectory integration.
        Single-step interface is less efficient due to problem setup overhead.
        
        Parameters
        ----------
        x : ArrayLike
            Current state
        u : Optional[ArrayLike]
            Control input (None for autonomous)
        dt : Optional[float]
            Step size
        dW : Optional[ArrayLike]
            Brownian increments (not used - Julia generates internally)
            
        Returns
        -------
        ArrayLike
            Next state
        """
        step_size = dt if dt is not None else self.dt
        
        # Define constant control function
        u_func = lambda t, x_state: u
        
        # Setup problem for single step
        problem = self._setup_julia_problem(
            x, u_func, (0.0, step_size)
        )
        
        # Create algorithm instance
        alg = self._algorithm_constructor()
        
        # Solve for single step
        sol = self.de.solve(
            problem,
            alg,
            dt=step_size,
            adaptive=False,
            save_everystep=False,
            save_end=True
        )
        
        # Extract final state
        x_next = np.array(sol.u[-1])
        
        self._stats['total_steps'] += 1
        
        return x_next
    
    def integrate(
        self,
        x0: ArrayLike,
        u_func: Callable[[float, ArrayLike], Optional[ArrayLike]],
        t_span: Tuple[float, float],
        t_eval: Optional[ArrayLike] = None,
        dense_output: bool = False
    ) -> SDEIntegrationResult:
        """
        Integrate SDE over time interval using Julia solver.
        
        Parameters
        ----------
        x0 : ArrayLike
            Initial state (nx,)
        u_func : Callable
            Control policy: (t, x) -> u (or None for autonomous)
        t_span : Tuple[float, float]
            Integration interval (t_start, t_end)
        t_eval : Optional[ArrayLike]
            Specific times at which to save solution
        dense_output : bool
            If True, enable dense output interpolation
            
        Returns
        -------
        SDEIntegrationResult
            Integration result with trajectory and statistics
            
        Examples
        --------
        >>> # Autonomous SDE
        >>> result = integrator.integrate(
        ...     x0=np.array([1.0]),
        ...     u_func=lambda t, x: None,
        ...     t_span=(0.0, 10.0)
        ... )
        >>> 
        >>> # Controlled SDE
        >>> result = integrator.integrate(
        ...     x0=np.array([1.0, 0.0]),
        ...     u_func=lambda t, x: -K @ x,
        ...     t_span=(0.0, 10.0)
        ... )
        >>> 
        >>> # Save at specific times
        >>> t_eval = np.linspace(0, 10, 1001)
        >>> result = integrator.integrate(x0, u_func, (0, 10), t_eval=t_eval)
        """
        x0 = np.asarray(x0)
        t0, tf = t_span
        
        # Setup Julia SDE problem
        problem = self._setup_julia_problem(x0, u_func, t_span)
        
        # Create algorithm instance
        alg = self._algorithm_constructor()
        
        # Setup solver options
        solve_kwargs = {
            'dt': self.dt,
            'adaptive': self.step_mode == StepMode.ADAPTIVE,
        }
        
        # Add tolerance for adaptive methods
        if self.step_mode == StepMode.ADAPTIVE:
            solve_kwargs['reltol'] = self.rtol
            solve_kwargs['abstol'] = self.atol
        
        # Add save points if specified
        if t_eval is not None:
            solve_kwargs['saveat'] = list(np.asarray(t_eval))
        elif not self.save_everystep:
            # If not saving every step and no t_eval, just save endpoints
            solve_kwargs['saveat'] = [float(t0), float(tf)]
        # else: save_everystep=True is default, no saveat needed
        
        # Dense output
        if dense_output:
            solve_kwargs['dense'] = True
        
        # IMPORTANT: Don't set seed through solve options
        # Julia random seed must be set globally before calling solve
        if self.seed is not None:
            # Set Julia's random seed before solving
            self.de.eval(f"using Random; Random.seed!({self.seed})")
        
        # Solve SDE
        try:
            sol = self.de.solve(problem, alg, **solve_kwargs)
            
            # Extract solution - Julia returns arrays
            t_out = np.array(sol.t)
            
            # Convert list of arrays to 2D array (T, nx)
            if hasattr(sol, 'u') and len(sol.u) > 0:
                x_out = np.array([np.array(u) for u in sol.u])
            else:
                # Integration failed - no solution
                return SDEIntegrationResult(
                    t=np.array([t0]),
                    x=x0.reshape(1, -1),
                    success=False,
                    message="Julia SDE integration produced no output",
                    nfev=0,
                    nsteps=0,
                    diffusion_evals=0,
                    n_paths=1,
                    convergence_type=self.convergence_type,
                    solver=self.algorithm,
                    sde_type=self.sde_type
                )
            
            # Check success
            success = len(t_out) > 1 and np.all(np.isfinite(x_out))
            
            # Estimate function evaluations
            nsteps = len(t_out) - 1
            nfev = self._stats['total_fev']
            
            return SDEIntegrationResult(
                t=t_out,
                x=x_out,
                success=success,
                message="Julia SDE integration successful" if success else "Integration failed (NaN/Inf detected)",
                nfev=nfev,
                nsteps=nsteps,
                diffusion_evals=self._stats['diffusion_evals'],
                noise_samples=None,  # Julia doesn't expose noise samples
                n_paths=1,
                convergence_type=self.convergence_type,
                solver=self.algorithm,
                sde_type=self.sde_type,
                dense_solution=sol if dense_output else None
            )
            
        except Exception as e:
            import traceback
            return SDEIntegrationResult(
                t=np.array([t0]),
                x=x0.reshape(1, -1),
                success=False,
                message=f"Julia SDE integration failed: {str(e)}\n{traceback.format_exc()}",
                nfev=0,
                nsteps=0,
                diffusion_evals=0,
                n_paths=1,
                convergence_type=self.convergence_type,
                solver=self.algorithm,
                sde_type=self.sde_type
            )
    
    # ========================================================================
    # Algorithm Information
    # ========================================================================
    
    @staticmethod
    def list_algorithms() -> Dict[str, List[str]]:
        """
        List available Julia SDE algorithms by category.
        
        Returns
        -------
        Dict[str, List[str]]
            Algorithms organized by category
            
        Examples
        --------
        >>> algorithms = DiffEqPySDEIntegrator.list_algorithms()
        >>> print(algorithms['euler_maruyama'])
        ['EM', 'LambaEM', 'EulerHeun']
        """
        return {
            'euler_maruyama': [
                'EM',           # Classic Euler-Maruyama
                'LambaEM',      # Lambda-adapted Euler-Maruyama
                'EulerHeun',    # Euler-Heun (predictor-corrector)
            ],
            'stochastic_rk': [
                'SRIW1',        # Roessler SRI for diagonal noise (strong 1.5)
                'SRIW2',        # Higher order variant
                'SOSRI',        # Second-order SRI for scalar noise
                'SOSRI2',       # Variant
                'SRA',          # Roessler SRA
                'SRA1',         # Order 2.0 weak for diagonal noise
                'SRA2',         # Variant
                'SRA3',         # For additive noise
                'SOSRA',        # Second-order SRA for scalar noise
                'SOSRA2',       # Variant
            ],
            'milstein': [
                'RKMil',        # Runge-Kutta Milstein
                'RKMilCommute', # For commutative noise
                'RKMilGeneral', # General Milstein
            ],
            'implicit': [
                'ImplicitEM',          # Implicit Euler-Maruyama
                'ImplicitEulerHeun',   # Implicit Euler-Heun
                'ImplicitRKMil',       # Implicit RK Milstein
            ],
            'imex': [
                'SKenCarp',     # Stochastic Kennedy-Carpenter IMEX
            ],
            'adaptive': [
                'AutoEM',       # Automatic method selection
            ],
            'optimized': [
                'SRI',              # Roessler SRI base
                'SRIW1Optimized',   # Optimized SRIW1
                'SRIW2Optimized',   # Optimized SRIW2
            ]
        }
    
    @staticmethod
    def get_algorithm_info(algorithm: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific algorithm.
        
        Parameters
        ----------
        algorithm : str
            Algorithm name
            
        Returns
        -------
        Dict[str, Any]
            Algorithm properties and recommendations
            
        Examples
        --------
        >>> info = DiffEqPySDEIntegrator.get_algorithm_info('SRIW1')
        >>> print(info['description'])
        'Roessler SRI, strong order 1.5, for diagonal noise'
        """
        algorithm_info = {
            'EM': {
                'name': 'Euler-Maruyama',
                'strong_order': 0.5,
                'weak_order': 1.0,
                'description': 'Classic explicit method, robust and fast',
                'best_for': 'General purpose, quick simulations',
                'noise_type': 'any',
            },
            'LambaEM': {
                'name': 'Lambda-adapted Euler-Maruyama',
                'strong_order': 0.5,
                'weak_order': 1.0,
                'description': 'EM with lambda step size control',
                'best_for': 'When dt needs automatic adjustment',
                'noise_type': 'any',
            },
            'SRIW1': {
                'name': 'Roessler SRI W1',
                'strong_order': 1.5,
                'weak_order': 2.0,
                'description': 'High accuracy for diagonal noise',
                'best_for': 'High accuracy diagonal noise problems',
                'noise_type': 'diagonal',
            },
            'SRA1': {
                'name': 'Roessler SRA1',
                'strong_order': 1.0,
                'weak_order': 2.0,
                'description': 'Second-order weak for diagonal noise',
                'best_for': 'Monte Carlo simulations',
                'noise_type': 'diagonal',
            },
            'SRA3': {
                'name': 'Roessler SRA3',
                'strong_order': 1.0,
                'weak_order': 2.0,
                'description': 'Second-order weak for additive noise',
                'best_for': 'Fast Monte Carlo with additive noise',
                'noise_type': 'additive',
            },
            'RKMil': {
                'name': 'Runge-Kutta Milstein',
                'strong_order': 1.0,
                'weak_order': 1.0,
                'description': 'Includes Levy area approximation',
                'best_for': 'When derivatives of diffusion available',
                'noise_type': 'any',
            },
            'ImplicitEM': {
                'name': 'Implicit Euler-Maruyama',
                'strong_order': 0.5,
                'weak_order': 1.0,
                'description': 'Implicit for stiff drift',
                'best_for': 'Stiff SDEs',
                'noise_type': 'any',
            },
            'SKenCarp': {
                'name': 'Stochastic Kennedy-Carpenter IMEX',
                'strong_order': 'variable',
                'weak_order': 'variable',
                'description': 'IMEX for semi-stiff problems',
                'best_for': 'Stiff drift, non-stiff diffusion',
                'noise_type': 'any',
            },
        }
        
        return algorithm_info.get(
            algorithm,
            {
                'name': algorithm,
                'description': 'Julia SDE algorithm (details not available)',
                'best_for': 'Check Julia documentation',
            }
        )
    
    @staticmethod
    def recommend_algorithm(
        noise_type: str,
        stiffness: str = 'none',
        accuracy: str = 'medium'
    ) -> str:
        """
        Recommend Julia SDE algorithm based on problem characteristics.
        
        Parameters
        ----------
        noise_type : str
            'additive', 'diagonal', 'scalar', or 'general'
        stiffness : str
            'none', 'moderate', or 'severe'
        accuracy : str
            'low', 'medium', or 'high'
            
        Returns
        -------
        str
            Recommended algorithm name
            
        Examples
        --------
        >>> alg = DiffEqPySDEIntegrator.recommend_algorithm(
        ...     noise_type='diagonal',
        ...     stiffness='none',
        ...     accuracy='high'
        ... )
        >>> print(alg)
        'SRIW1'
        """
        if stiffness == 'severe':
            return 'ImplicitEM'
        elif stiffness == 'moderate':
            return 'SKenCarp'
        
        # Non-stiff recommendations
        if accuracy == 'high':
            if noise_type == 'additive':
                return 'SRA3'
            elif noise_type in ['diagonal', 'scalar']:
                return 'SRIW1'
            else:
                return 'RKMil'
        elif accuracy == 'medium':
            if noise_type == 'diagonal':
                return 'SRA1'
            else:
                return 'LambaEM'
        else:  # low accuracy
            return 'EM'


# ============================================================================
# Utility Functions
# ============================================================================

def create_diffeqpy_sde_integrator(
    sde_system,
    algorithm: str = 'EM',
    dt: float = 0.01,
    **options
) -> DiffEqPySDEIntegrator:
    """
    Quick factory for Julia SDE integrators.
    
    Parameters
    ----------
    sde_system : StochasticDynamicalSystem
        SDE system to integrate
    algorithm : str
        Julia algorithm name
    dt : float
        Time step
    **options
        Additional options
        
    Returns
    -------
    DiffEqPySDEIntegrator
        Configured integrator
        
    Examples
    --------
    >>> integrator = create_diffeqpy_sde_integrator(
    ...     sde_system,
    ...     algorithm='SRIW1',
    ...     dt=0.001,
    ...     rtol=1e-6
    ... )
    """
    return DiffEqPySDEIntegrator(
        sde_system,
        dt=dt,
        algorithm=algorithm,
        backend='numpy',
        **options
    )


def list_julia_sde_algorithms() -> None:
    """
    Print all available Julia SDE algorithms with descriptions.
    
    Examples
    --------
    >>> list_julia_sde_algorithms()
    
    Euler-Maruyama Family:
      - EM: Euler-Maruyama (strong 0.5, weak 1.0)
      - LambaEM: Lambda-adapted Euler-Maruyama
    ...
    """
    algorithms = DiffEqPySDEIntegrator.list_algorithms()
    
    print("Julia SDE Algorithms (via DifferentialEquations.jl)")
    print("=" * 60)
    
    category_names = {
        'euler_maruyama': 'Euler-Maruyama Family',
        'stochastic_rk': 'Stochastic Runge-Kutta Methods',
        'milstein': 'Milstein Family',
        'implicit': 'Implicit Methods (for stiff drift)',
        'imex': 'IMEX Methods (semi-stiff)',
        'adaptive': 'Adaptive Methods',
        'optimized': 'Optimized Variants',
    }
    
    for category, algs in algorithms.items():
        print(f"\n{category_names.get(category, category)}:")
        for alg in algs:
            info = DiffEqPySDEIntegrator.get_algorithm_info(alg)
            if 'strong_order' in info:
                print(f"  - {alg}: {info['description']} "
                      f"(strong {info['strong_order']}, weak {info['weak_order']})")
            else:
                print(f"  - {alg}: {info['description']}")
    
    print("\n" + "=" * 60)
    print("Use get_algorithm_info(name) for detailed information")
    print("Use recommend_algorithm() for automatic selection")