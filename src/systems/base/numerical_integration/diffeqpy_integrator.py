"""
DiffEqPyIntegrator: Julia DifferentialEquations.jl ODE solver via diffeqpy.

Provides access to Julia's DifferentialEquations.jl ecosystem - the most
comprehensive and performant ODE solver suite available in any language.

Features:
- 100+ ODE solvers (explicit, implicit, IMEX, stabilized, geometric)
- Automatic stiffness detection
- Event handling and callbacks
- Dense output and interpolation
- Exceptional performance for difficult ODEs

Requirements:
    Julia must be installed with DifferentialEquations.jl:
        julia> using Pkg
        julia> Pkg.add("DifferentialEquations")
    
    Python package:
        $ pip install diffeqpy

Notes
-----
This integrator only supports NumPy backend because Julia arrays are 
converted to/from NumPy. For gradient-based workflows:
- Use TorchDiffEqIntegrator for PyTorch with autograd
- Use DiffraxIntegrator for JAX with JIT and autograd

DiffEqPyIntegrator is ideal for:
- Production simulations requiring highest reliability
- Very stiff or difficult ODEs where scipy struggles
- Problems where Julia's specialized solvers excel
- Non-gradient workflows where accuracy/performance matter most

Examples
--------
>>> # High-accuracy non-stiff solver
>>> integrator = DiffEqPyIntegrator(
...     system,
...     backend='numpy',
...     algorithm='Vern9',
...     reltol=1e-12,
...     abstol=1e-14
... )
>>> 
>>> # Stiff system solver
>>> integrator = DiffEqPyIntegrator(
...     system,
...     algorithm='Rosenbrock23'
... )
>>> 
>>> # Auto-switching between non-stiff and stiff
>>> integrator = DiffEqPyIntegrator(
...     system,
...     algorithm='AutoTsit5(Rosenbrock23())'
... )
"""

import time
import numpy as np
from typing import Optional, Callable, Tuple, Dict, Any, List, TYPE_CHECKING

from src.systems.base.numerical_integration.integrator_base import (
    IntegratorBase,
    IntegrationResult,
    StepMode,
    ArrayLike
)

if TYPE_CHECKING:
    from src.systems.base.symbolic_dynamical_system import SymbolicDynamicalSystem


class DiffEqPyIntegrator(IntegratorBase):
    """
    ODE integrator using Julia's DifferentialEquations.jl via diffeqpy.
    
    Provides access to 100+ ODE solvers from the Julia ecosystem, including:
    
    **Non-Stiff Solvers (Explicit Runge-Kutta):**
    - Tsit5: Tsitouras 5(4) - recommended default
    - Vern6, Vern7, Vern8, Vern9: Verner methods (high order)
    - DP5, DP8: Dormand-Prince
    - TanYam7, TsitPap8: High-order methods
    
    **Stiff Solvers (Implicit/Rosenbrock):**
    - Rosenbrock23, Rosenbrock32: Rosenbrock methods (recommended)
    - Rodas4, Rodas4P, Rodas5: High-accuracy Rosenbrock
    - TRBDF2, KenCarp3-5: ESDIRK methods
    - RadauIIA5: Implicit Runge-Kutta
    - QNDF, FBDF: BDF variants
    
    **Auto-Switching:**
    - AutoTsit5(Rosenbrock23()): Auto-detects stiffness
    - AutoVern7(Rodas5()): High-accuracy auto-switching
    
    **Stabilized (for moderately stiff):**
    - ROCK2, ROCK4: Stabilized explicit
    - ESERK4, ESERK5: Stabilized ERK
    
    **Geometric (structure-preserving):**
    - SymplecticEuler, VelocityVerlet: Symplectic integrators
    - McAte2-5: McClellan-Aitken methods
    
    Parameters
    ----------
    system : SymbolicDynamicalSystem
        Continuous-time system to integrate
    dt : Optional[float]
        Time step (initial guess for adaptive, fixed for ConstantStepSize)
    step_mode : StepMode
        FIXED or ADAPTIVE stepping mode
    backend : str
        Must be 'numpy' (Julia arrays convert to NumPy)
    algorithm : str, optional
        Julia solver algorithm. Default: 'Tsit5'
        See list_algorithms() for available options
    save_everystep : bool, optional
        Save every integration step (default: False for adaptive)
    dense : bool, optional
        Enable dense output for interpolation (default: False)
    callback : Optional[Callable]
        Julia callback for events, termination, etc.
    **options
        Additional solver options:
        - reltol: Relative tolerance (default: 1e-6)
        - abstol: Absolute tolerance (default: 1e-8)
        - maxiters: Maximum iterations (default: 1e7)
        - dtmin: Minimum step size (adaptive only)
        - dtmax: Maximum step size (adaptive only)
    
    Attributes
    ----------
    de : module
        Julia DifferentialEquations module
    algorithm : str
        Algorithm name
    save_everystep : bool
        Whether to save all steps
    dense : bool
        Whether dense output is enabled
    
    Examples
    --------
    >>> # Default high-quality solver
    >>> integrator = DiffEqPyIntegrator(system, backend='numpy')
    >>> result = integrator.integrate(x0, u_func, (0, 10))
    >>> 
    >>> # Very high accuracy
    >>> integrator = DiffEqPyIntegrator(
    ...     system,
    ...     algorithm='Vern9',
    ...     reltol=1e-12,
    ...     abstol=1e-14
    ... )
    >>> 
    >>> # Stiff system
    >>> integrator = DiffEqPyIntegrator(
    ...     system,
    ...     algorithm='Rodas5',
    ...     reltol=1e-8
    ... )
    >>> 
    >>> # Fixed-step integration
    >>> integrator = DiffEqPyIntegrator(
    ...     system,
    ...     dt=0.01,
    ...     step_mode=StepMode.FIXED,
    ...     algorithm='Tsit5'
    ... )
    
    Notes
    -----
    - Julia must be installed with DifferentialEquations.jl
    - First call may be slow due to Julia JIT compilation
    - Subsequent calls are very fast (Julia's strength)
    - For difficult ODEs, this is often the best choice
    """
    
    def __init__(
        self,
        system: 'SymbolicDynamicalSystem',
        dt: Optional[float] = None,
        step_mode: StepMode = StepMode.ADAPTIVE,
        backend: str = 'numpy',
        algorithm: str = 'Tsit5',
        save_everystep: bool = False,
        dense: bool = False,
        callback: Optional[Any] = None,
        **options
    ):
        """
        Initialize DiffEqPy integrator.
        
        Parameters
        ----------
        system : SymbolicDynamicalSystem
            System to integrate
        dt : Optional[float]
            Time step (initial for adaptive, fixed for FIXED mode)
        step_mode : StepMode
            FIXED or ADAPTIVE
        backend : str
            Must be 'numpy'
        algorithm : str
            Julia algorithm name (default: 'Tsit5')
        save_everystep : bool
            Save every step (useful for visualization)
        dense : bool
            Enable dense output for interpolation
        callback : Optional
            Julia callback function
        **options
            Solver options (reltol, abstol, etc.)
        """
        if backend != 'numpy':
            raise ValueError(
                f"DiffEqPyIntegrator requires backend='numpy', got '{backend}'. "
                f"Julia arrays are automatically converted to NumPy."
            )
        
        super().__init__(system, dt, step_mode, backend, **options)
        
        self.algorithm = algorithm
        self.save_everystep = save_everystep
        self.dense = dense
        self.callback = callback
        self._integrator_name = f"DiffEqPy-{algorithm}"
        
        # Try to import diffeqpy
        try:
            from diffeqpy import de
            self.de = de
        except ImportError:
            raise ImportError(
                "diffeqpy is required for DiffEqPyIntegrator.\n\n"
                "Installation steps:\n"
                "1. Install Julia from https://julialang.org/downloads/\n"
                "2. Install DifferentialEquations.jl:\n"
                "   julia> using Pkg\n"
                "   julia> Pkg.add('DifferentialEquations')\n"
                "3. Install Python package:\n"
                "   pip install diffeqpy\n"
                "4. Run Julia setup from Python:\n"
                "   python -c 'from diffeqpy import install; install()'"
            )
        
        # Validate algorithm exists (but don't fail for auto-switching algorithms)
        if 'Auto' not in algorithm and '(' not in algorithm:
            try:
                self._get_algorithm()
            except Exception as e:
                raise ValueError(
                    f"Invalid algorithm '{algorithm}'. "
                    f"Use list_algorithms() to see available options.\n"
                    f"Error: {e}"
                )
    
    @property
    def name(self) -> str:
        """Return integrator name."""
        mode_str = "Fixed" if self.step_mode == StepMode.FIXED else "Adaptive"
        
        # Identify solver type
        stiff_algorithms = [
            'Rosenbrock23', 'Rosenbrock32', 'Rodas4', 'Rodas4P', 'Rodas5',
            'TRBDF2', 'KenCarp3', 'KenCarp4', 'KenCarp5',
            'RadauIIA5', 'QNDF', 'FBDF'
        ]
        stabilized_algorithms = ['ROCK2', 'ROCK4', 'ESERK4', 'ESERK5']
        geometric_algorithms = [
            'SymplecticEuler', 'VelocityVerlet', 
            'McAte2', 'McAte4', 'McAte5'
        ]
        
        if 'Auto' in self.algorithm:
            type_str = " (Auto-Stiffness)"
        elif self.algorithm in stiff_algorithms:
            type_str = " (Stiff)"
        elif self.algorithm in stabilized_algorithms:
            type_str = " (Stabilized)"
        elif self.algorithm in geometric_algorithms:
            type_str = " (Geometric)"
        else:
            type_str = ""
        
        return f"{self._integrator_name} ({mode_str}){type_str}"
    
    def _get_algorithm(self):
        """
        Get Julia algorithm object.
        
        Handles both simple algorithms and complex ones like:
        - AutoTsit5(Rosenbrock23())
        - Composite algorithm specifications
        
        Returns
        -------
        Julia algorithm object
        """
        algo_str = self.algorithm
        
        # Handle auto-switching algorithms
        if 'Auto' in algo_str or '(' in algo_str:
            try:
                # Evaluate as Julia expression
                # e.g., "AutoTsit5(Rosenbrock23())"
                return eval(f"self.de.{algo_str}")
            except Exception as e:
                raise ValueError(
                    f"Failed to create algorithm '{algo_str}'. "
                    f"Check syntax. Error: {e}"
                )
        else:
            # Simple algorithm name
            try:
                algo_class = getattr(self.de, algo_str)
                return algo_class()
            except AttributeError:
                raise ValueError(
                    f"Algorithm '{algo_str}' not found in DifferentialEquations.jl. "
                    f"Use list_algorithms() to see available options."
                )
    
    def step(
        self,
        x: ArrayLike,
        u: ArrayLike,
        dt: Optional[float] = None
    ) -> ArrayLike:
        """
        Take one integration step.
        
        For efficiency, uses integrate() internally with single step.
        
        Parameters
        ----------
        x : ArrayLike
            Current state (nx,)
        u : ArrayLike
            Control input (nu,) - assumed constant over step
        dt : Optional[float]
            Step size (uses self.dt if None)
            
        Returns
        -------
        ArrayLike
            Next state x(t + dt)
        """
        step_size = dt if dt is not None else self.dt
        if step_size is None:
            raise ValueError("Step size dt must be specified")
        
        x = np.asarray(x)
        u = np.asarray(u)
        
        # Use integrate for single step
        result = self.integrate(
            x0=x,
            u_func=lambda t, x_cur: u,
            t_span=(0.0, step_size),
            t_eval=np.array([0.0, step_size])
        )
        
        return result.x[-1]
    
    def integrate(
        self,
        x0: ArrayLike,
        u_func: Callable[[float, ArrayLike], ArrayLike],
        t_span: Tuple[float, float],
        t_eval: Optional[ArrayLike] = None,
        dense_output: bool = False
    ) -> IntegrationResult:
        """
        Integrate ODE using Julia's DifferentialEquations.jl.
        
        Parameters
        ----------
        x0 : ArrayLike
            Initial state (nx,)
        u_func : Callable[[float, ArrayLike], ArrayLike]
            Control policy (t, x) → u
        t_span : Tuple[float, float]
            Time interval (t_start, t_end)
        t_eval : Optional[ArrayLike]
            Specific times at which to store solution
            If None:
            - FIXED mode: Uses uniform grid with dt
            - ADAPTIVE mode: Julia chooses points adaptively
        dense_output : bool
            Enable dense output for interpolation
            
        Returns
        -------
        IntegrationResult
            Integration result with:
            - t: Time points
            - x: State trajectory
            - success: Whether integration succeeded
            - sol: Julia solution object (if dense_output=True)
            
        Examples
        --------
        >>> # Adaptive integration
        >>> result = integrator.integrate(
        ...     x0=np.array([1.0, 0.0]),
        ...     u_func=lambda t, x: -K @ x,
        ...     t_span=(0.0, 10.0)
        ... )
        >>> 
        >>> # Fixed-step with specific evaluation times
        >>> t_eval = np.linspace(0, 10, 1001)
        >>> result = integrator.integrate(
        ...     x0, u_func, (0, 10),
        ...     t_eval=t_eval
        ... )
        >>> 
        >>> # Dense output for interpolation
        >>> result = integrator.integrate(
        ...     x0, u_func, (0, 10),
        ...     dense_output=True
        ... )
        >>> # Interpolate at arbitrary time
        >>> x_at_5_5 = result.sol(5.5)
        """
        start_time = time.time()
        
        t0, tf = t_span
        x0 = np.asarray(x0)
        
        # Handle edge case
        if t0 == tf:
            return IntegrationResult(
                t=np.array([t0]),
                x=x0[None, :] if x0.ndim == 1 else x0,
                success=True,
                message="Zero time span",
                nfev=0,
                nsteps=0,
            )
        
        # Track function evaluations for this integration
        fev_count = [0]
        
        # Define ODE function for Julia
        # Julia signature: du = f(u, p, t)
        def ode_func(u_val, p, t):
            """
            ODE function in Julia's signature.
            
            Parameters
            ----------
            u_val : array
                Current state (Julia array)
            p : any
                Parameters (unused, required by Julia)
            t : float
                Current time
                
            Returns
            -------
            array
                State derivative dx/dt
            """
            # Convert from Julia arrays to NumPy
            x_np = np.array(u_val)
            
            # Evaluate control policy
            u_control = u_func(t, x_np)
            u_np = np.asarray(u_control)
            
            # Evaluate system dynamics
            dx = self.system(x_np, u_np, backend='numpy')
            
            # Track function evaluations
            fev_count[0] += 1
            
            return dx
        
        # Prepare time span
        tspan = (float(t0), float(tf))
        
        # Prepare save points (saveat)
        if t_eval is not None:
            # User specified evaluation times
            saveat = list(np.asarray(t_eval))
        elif self.step_mode == StepMode.FIXED:
            # Fixed-step mode: create uniform grid
            if self.dt is None:
                raise ValueError("dt required for FIXED step mode")
            n_steps = int(np.ceil((tf - t0) / self.dt)) + 1
            saveat = list(np.linspace(t0, tf, n_steps))
        else:
            # Adaptive mode with no t_eval: let Julia choose
            saveat = []
        
        # Build solver options
        solve_kwargs = {
            'reltol': self.rtol,
            'abstol': self.atol,
            'maxiters': int(self.options.get('maxiters', 1e7)),
        }
        
        # Add saveat if specified
        if saveat:
            solve_kwargs['saveat'] = saveat
        
        # Control step size behavior
        if self.step_mode == StepMode.FIXED and self.dt is not None:
            # Fixed step size
            solve_kwargs['dt'] = self.dt
            solve_kwargs['adaptive'] = False
        else:
            # Adaptive stepping
            solve_kwargs['save_everystep'] = self.save_everystep
            if self.options.get('dtmin') is not None:
                solve_kwargs['dtmin'] = self.options['dtmin']
            if self.options.get('dtmax') is not None:
                solve_kwargs['dtmax'] = self.options['dtmax']
        
        # Dense output
        solve_kwargs['dense'] = dense_output or self.dense
        
        # Callback (if provided)
        if self.callback is not None:
            solve_kwargs['callback'] = self.callback
        
        # Set up ODE problem
        prob = self.de.ODEProblem(ode_func, x0, tspan)
        
        # Get algorithm
        algorithm = self._get_algorithm()
        
        # Solve ODE
        try:
            sol = self.de.solve(prob, algorithm, **solve_kwargs)
            
            # Extract solution
            # Julia returns solution object with attributes:
            # - sol.t: time points (Vector)
            # - sol.u: state vectors (Vector of Vectors)
            t_out = np.array(sol.t)
            
            # Convert list of state vectors to 2D array
            # sol.u is a list of vectors, need to stack them
            x_out = np.array([np.array(x_i) for x_i in sol.u])
            
            # Determine success
            # Julia retcode: :Success means successful integration
            success = str(sol.retcode) == ':Success'
            message = f"Integration {sol.retcode}"
            
            # Update statistics
            nsteps = len(t_out) - 1
            nfev = fev_count[0]
            
            self._stats['total_steps'] += nsteps
            self._stats['total_fev'] += nfev
            
            elapsed = time.time() - start_time
            self._stats['total_time'] += elapsed
            
            return IntegrationResult(
                t=t_out,
                x=x_out,
                success=success,
                message=message,
                nfev=nfev,
                nsteps=nsteps,
                integration_time=elapsed,
                algorithm=self.algorithm,
                sol=sol if (dense_output or self.dense) else None,
            )
            
        except Exception as e:
            elapsed = time.time() - start_time
            self._stats['total_time'] += elapsed
            
            return IntegrationResult(
                t=np.array([t0]),
                x=x0[None, :] if x0.ndim == 1 else x0,
                success=False,
                message=f"Integration failed: {str(e)}",
                nfev=fev_count[0],
                nsteps=0,
                integration_time=elapsed,
            )
    
    def set_callback(self, callback):
        """
        Set Julia callback for events, termination, etc.
        
        Parameters
        ----------
        callback : Julia callback object
            See DifferentialEquations.jl documentation for callback types
        """
        self.callback = callback
    
    def __repr__(self) -> str:
        return (
            f"DiffEqPyIntegrator(algorithm='{self.algorithm}', "
            f"mode={self.step_mode.value}, "
            f"rtol={self.rtol:.1e}, atol={self.atol:.1e})"
        )


# ============================================================================
# Utility Functions
# ============================================================================

def list_algorithms() -> Dict[str, List[str]]:
    """
    List available Julia DifferentialEquations.jl algorithms.
    
    Returns
    -------
    Dict[str, List[str]]
        Dictionary mapping algorithm categories to lists of algorithm names
    
    Examples
    --------
    >>> algos = list_algorithms()
    >>> print(algos['nonstiff'])
    ['Tsit5', 'Vern7', 'Vern9', 'DP5', 'DP8', ...]
    >>> 
    >>> # Print all categories
    >>> for category, methods in algos.items():
    ...     print(f"{category}: {', '.join(methods)}")
    """
    return {
        'nonstiff': [
            'Tsit5',       # Tsitouras 5(4) - RECOMMENDED DEFAULT
            'Vern6',       # Verner 6(5)
            'Vern7',       # Verner 7(6)
            'Vern8',       # Verner 8(7)
            'Vern9',       # Verner 9(8) - very high accuracy
            'DP5',         # Dormand-Prince 5(4)
            'DP8',         # Dormand-Prince 8
            'TanYam7',     # Tanaka-Yamashita 7
            'TsitPap8',    # Tsitouras-Papakostas 8
        ],
        'stiff_rosenbrock': [
            'Rosenbrock23',   # RECOMMENDED for moderately stiff
            'Rosenbrock32',   # Alternative Rosenbrock
            'Rodas4',         # High accuracy
            'Rodas4P',        # Rodas4 with Predictive step
            'Rodas5',         # Very high accuracy
        ],
        'stiff_esdirk': [
            'TRBDF2',      # Trapezoidal + BDF2
            'KenCarp3',    # Kennedy-Carpenter ESDIRK 3
            'KenCarp4',    # Kennedy-Carpenter ESDIRK 4
            'KenCarp5',    # Kennedy-Carpenter ESDIRK 5
        ],
        'stiff_implicit': [
            'RadauIIA5',   # Implicit Runge-Kutta (very stable)
            'QNDF',        # Variable-order BDF
            'FBDF',        # Fixed-order BDF
        ],
        'auto_switching': [
            'AutoTsit5(Rosenbrock23())',    # RECOMMENDED auto-switch
            'AutoVern7(Rodas5())',          # High-accuracy auto-switch
            'AutoVern8(Rodas5())',
            'AutoVern9(Rodas5())',
        ],
        'stabilized': [
            'ROCK2',       # Stabilized explicit (moderately stiff)
            'ROCK4',       # Higher-order ROCK
            'ESERK4',      # Stabilized ERK 4
            'ESERK5',      # Stabilized ERK 5
        ],
        'geometric': [
            'SymplecticEuler',   # 1st order symplectic
            'VelocityVerlet',    # 2nd order symplectic
            'VerletLeapfrog',    # Leapfrog method
            'McAte2',            # McClellan-Aitken 2
            'McAte4',            # McClellan-Aitken 4
            'McAte5',            # McClellan-Aitken 5
        ],
        'low_order': [
            'Euler',         # Forward Euler (1st order)
            'Midpoint',      # Explicit midpoint (2nd order)
            'Heun',          # Heun's method (2nd order)
        ],
    }


def print_algorithm_recommendations():
    """
    Print recommendations for algorithm selection.
    
    Examples
    --------
    >>> print_algorithm_recommendations()
    """
    print("=" * 70)
    print("Julia DifferentialEquations.jl Algorithm Recommendations")
    print("=" * 70)
    print()
    print("🎯 DEFAULT CHOICE:")
    print("  Tsit5")
    print("  - Excellent general-purpose solver")
    print("  - Good accuracy and performance")
    print("  - Works for most non-stiff problems")
    print()
    print("🔄 AUTO-SWITCHING (BEST FOR UNKNOWN STIFFNESS):")
    print("  AutoTsit5(Rosenbrock23())")
    print("  - Automatically detects stiffness")
    print("  - Switches between Tsit5 (non-stiff) and Rosenbrock23 (stiff)")
    print("  - Set-and-forget option")
    print()
    print("📈 HIGH ACCURACY (NON-STIFF):")
    print("  Vern9")
    print("  - 9th order accuracy")
    print("  - Best for very smooth problems requiring high precision")
    print("  - Orbital mechanics, celestial mechanics")
    print()
    print("🔥 STIFF SYSTEMS:")
    print("  Rosenbrock23 (RECOMMENDED)")
    print("  - Fast and reliable for moderately stiff problems")
    print("  Rodas5")
    print("  - High accuracy for very stiff problems")
    print("  - Chemical kinetics, circuit simulation")
    print()
    print("⚡ STABILIZED (MODERATELY STIFF):")
    print("  ROCK4")
    print("  - Good for problems between non-stiff and stiff")
    print("  - Larger stability region than explicit methods")
    print()
    print("🎨 GEOMETRIC (STRUCTURE-PRESERVING):")
    print("  VelocityVerlet")
    print("  - Symplectic integrator for Hamiltonian systems")
    print("  - Preserves energy in conservative systems")
    print("  - Molecular dynamics, celestial mechanics")
    print()
    print("=" * 70)


def create_diffeqpy_integrator(
    system: 'SymbolicDynamicalSystem',
    algorithm: str = 'Tsit5',
    dt: Optional[float] = None,
    step_mode: StepMode = StepMode.ADAPTIVE,
    **options
) -> DiffEqPyIntegrator:
    """
    Quick factory for DiffEqPy integrators.
    
    Parameters
    ----------
    system : SymbolicDynamicalSystem
        System to integrate
    algorithm : str
        Julia algorithm name (default: 'Tsit5')
    dt : Optional[float]
        Time step (initial for adaptive)
    step_mode : StepMode
        FIXED or ADAPTIVE
    **options
        Additional solver options (reltol, abstol, etc.)
        
    Returns
    -------
    DiffEqPyIntegrator
        Configured integrator
        
    Examples
    --------
    >>> # Default high-quality solver
    >>> integrator = create_diffeqpy_integrator(system)
    >>> 
    >>> # High-accuracy solver
    >>> integrator = create_diffeqpy_integrator(
    ...     system,
    ...     algorithm='Vern9',
    ...     reltol=1e-12,
    ...     abstol=1e-14
    ... )
    >>> 
    >>> # Auto-switching solver
    >>> integrator = create_diffeqpy_integrator(
    ...     system,
    ...     algorithm='AutoTsit5(Rosenbrock23())'
    ... )
    """
    return DiffEqPyIntegrator(
        system=system,
        dt=dt,
        step_mode=step_mode,
        backend='numpy',
        algorithm=algorithm,
        **options
    )


# ============================================================================
# Integration with Factory
# ============================================================================

def get_algorithm_info(algorithm: str) -> Dict[str, Any]:
    """
    Get information about a specific Julia algorithm.
    
    Parameters
    ----------
    algorithm : str
        Algorithm name
        
    Returns
    -------
    Dict[str, Any]
        Information about the algorithm
        
    Examples
    --------
    >>> info = get_algorithm_info('Tsit5')
    >>> print(info['description'])
    """
    algorithm_info = {
        'Tsit5': {
            'name': 'Tsitouras 5(4)',
            'order': 5,
            'type': 'Explicit Runge-Kutta',
            'description': 'Excellent general-purpose solver with good efficiency',
            'best_for': 'Most non-stiff problems',
            'stability': 'Conditionally stable',
        },
        'Vern9': {
            'name': 'Verner 9(8)',
            'order': 9,
            'type': 'Explicit Runge-Kutta',
            'description': 'Very high accuracy solver',
            'best_for': 'High-precision requirements, smooth problems',
            'stability': 'Conditionally stable',
        },
        'Rosenbrock23': {
            'name': 'Rosenbrock 2/3',
            'order': 2,
            'type': 'Rosenbrock (Stiff)',
            'description': 'Efficient stiff solver with automatic Jacobian',
            'best_for': 'Moderately stiff problems',
            'stability': 'L-stable',
        },
        'Rodas5': {
            'name': 'Rodas 5(4)',
            'order': 5,
            'type': 'Rosenbrock (Stiff)',
            'description': 'High-accuracy stiff solver',
            'best_for': 'Very stiff problems requiring high accuracy',
            'stability': 'L-stable',
        },
        'AutoTsit5(Rosenbrock23())': {
            'name': 'Auto-switching Tsit5/Rosenbrock23',
            'order': 'Variable (2-5)',
            'type': 'Auto-switching',
            'description': 'Automatically switches between non-stiff and stiff solvers',
            'best_for': 'Unknown stiffness, set-and-forget',
            'stability': 'Adaptive',
        },
        'ROCK4': {
            'name': 'ROCK4',
            'order': 4,
            'type': 'Stabilized Explicit',
            'description': 'Stabilized explicit method for moderately stiff problems',
            'best_for': 'Moderately stiff ODEs with large explicit part',
            'stability': 'Extended stability',
        },
        'VelocityVerlet': {
            'name': 'Velocity Verlet',
            'order': 2,
            'type': 'Symplectic',
            'description': 'Structure-preserving integrator for Hamiltonian systems',
            'best_for': 'Conservative systems, molecular dynamics',
            'stability': 'Symplectic (preserves energy)',
        },
    }
    
    return algorithm_info.get(
        algorithm,
        {
            'name': algorithm,
            'description': 'No information available. Check Julia DifferentialEquations.jl documentation.'
        }
    )