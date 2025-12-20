"""
Stochastic Dynamical System - Coordinator

Extends SymbolicDynamicalSystem with SDE support by composing modular handlers.

This class is THIN - all logic is delegated to:
    - Parent class (drift handling)
    - DiffusionHandler (diffusion code generation)
    - NoiseCharacterizer (noise analysis)
    - SDEValidator (validation)
    - BackendManager (type conversions) - inherited

Mathematical Form:
    dx = f(x, u, t)dt + g(x, u, t)dW

Reuses:
    - SymbolicDynamicalSystem (parent) - ALL drift logic
    - DiffusionHandler - diffusion code generation
    - NoiseCharacterizer - noise analysis (via DiffusionHandler)
    - SDEValidator - validation
    - BackendManager - type conversions (inherited from parent)
"""

from typing import List, Optional, Dict, Any
import sympy as sp

from src.systems.base.symbolic_dynamical_system import SymbolicDynamicalSystem
from src.systems.base.utils.stochastic.diffusion_handler import DiffusionHandler
from src.systems.base.utils.stochastic.noise_analysis import NoiseCharacteristics, SDEType
from src.systems.base.utils.stochastic.sde_validator import SDEValidator


class StochasticDynamicalSystem(SymbolicDynamicalSystem):
    """
    Stochastic dynamical system - thin coordinator over modular components.
    
    Extends SymbolicDynamicalSystem with SDE support via composition.
    All logic is delegated to specialized handlers.
    
    Components:
    - Parent class: Handles ALL drift logic ✅
    - DiffusionHandler: Generates diffusion functions ✅
    - NoiseCharacterizer: Analyzes noise (via handler) ✅
    - SDEValidator: Validates SDE constraints ✅
    
    This class is ~200 lines (was 400+) because logic is modularized!
    
    Mathematical Form
    -----------------
    dx = f(x, u, t)dt + g(x, u, t)dW
    
    where:
        f(x, u, t) = drift term (deterministic)
        g(x, u, t) = diffusion matrix (nx, nw)
        dW = Brownian motion (nw independent Wiener processes)
    
    Parameters
    ----------
    drift_expr : sp.Matrix
        Symbolic drift vector f(x, u), shape (nx, 1)
    diffusion_expr : sp.Matrix
        Symbolic diffusion matrix g(x, u), shape (nx, nw)
    state_vars : List[sp.Symbol]
        State variables [x1, x2, ..., xn]
    control_vars : List[sp.Symbol]
        Control variables [u1, u2, ..., um]
    sde_type : str, optional
        'ito' or 'stratonovich'. Default: 'ito'
    validate : bool, optional
        Run validation on construction. Default: True
    **kwargs
        Additional arguments passed to parent class
    
    Examples
    --------
    >>> from sympy import symbols, Matrix
    >>> 
    >>> # Ornstein-Uhlenbeck process
    >>> x, u = symbols('x u')
    >>> drift = Matrix([-x + u])
    >>> diffusion = Matrix([[0.5]])
    >>> 
    >>> system = StochasticDynamicalSystem(
    ...     drift_expr=drift,
    ...     diffusion_expr=diffusion,
    ...     state_vars=[x],
    ...     control_vars=[u],
    ...     sde_type='ito'
    ... )
    >>> 
    >>> # Automatic noise analysis
    >>> print(system.noise_characteristics.noise_type)
    NoiseType.ADDITIVE
    >>> 
    >>> # Evaluate drift and diffusion
    >>> f_val = system.drift(np.array([1.0]), np.array([0.0]))
    >>> g_val = system.diffusion(np.array([1.0]), np.array([0.0]))
    """
    
    def __init__(
        self,
        drift_expr: sp.Matrix,
        diffusion_expr: sp.Matrix,
        state_vars: List[sp.Symbol],
        control_vars: Optional[List[sp.Symbol]] = None,
        sde_type: str = 'ito',
        validate: bool = True,
        **kwargs
    ):
        """Initialize stochastic system with validation and handler composition."""
        
        # ✅ VALIDATE: Use SDEValidator before construction
        if validate:
            validator = SDEValidator(strict=True)
            validator.validate_sde_system(
                drift_expr=drift_expr,
                diffusion_expr=diffusion_expr,
                state_vars=state_vars,
                control_vars=control_vars if control_vars else [],
                sde_type=sde_type,
                claimed_noise_type=kwargs.get('noise_type'),  # If user claims
                parameters=kwargs.get('parameters')
            )
        
        # ✅ REUSE: Parent class handles ALL drift logic
        super().__init__(
            dynamics_expr=drift_expr,
            state_vars=state_vars,
            control_vars=control_vars,
            **kwargs
        )
        
        # Store drift expression explicitly (alias for clarity)
        self.drift_expr = drift_expr
        self.diffusion_expr = diffusion_expr
        
        # ✅ COMPOSE: Create DiffusionHandler
        self.diffusion_handler = DiffusionHandler(
            diffusion_expr=diffusion_expr,
            state_vars=state_vars,
            control_vars=control_vars if control_vars else [],
            time_var=None,  # TODO: Support time-varying diffusion
            parameters=self.parameters if hasattr(self, 'parameters') else None
        )
        
        # Extract characteristics (computed by handler's NoiseCharacterizer)
        self.noise_characteristics: NoiseCharacteristics = (
            self.diffusion_handler.characteristics
        )
        
        # SDE metadata
        self.nw = diffusion_expr.shape[1]
        self.sde_type = SDEType(sde_type)
        self.is_stochastic = True
    
    # ========================================================================
    # Primary Interface - Delegate to Handlers
    # ========================================================================
    
    def drift(self, x, u, backend: str = 'numpy'):
        """
        Evaluate drift term f(x, u).
        
        Delegation: ✅ Parent class (SymbolicDynamicalSystem)
        
        Returns
        -------
        ArrayLike
            Drift vector, shape (nx,)
        """
        # ✅ DELEGATE: Parent class handles drift evaluation
        return self(x, u, backend=backend)
    
    def diffusion(self, x, u, backend: str = 'numpy'):
        """
        Evaluate diffusion term g(x, u).
        
        Delegation: ✅ DiffusionHandler
        
        Returns
        -------
        ArrayLike
            Diffusion matrix, shape (nx, nw)
        """
        # ✅ DELEGATE: DiffusionHandler generates and evaluates
        func = self.diffusion_handler.generate_function(backend)
        
        # ✅ REUSE: BackendManager from parent for type conversion
        # (Parent class has self.backend_mgr if using newer version)
        # Convert inputs to lists for lambdified function
        if backend == 'numpy':
            import numpy as np
            x_arr = np.atleast_1d(np.asarray(x))
            u_arr = np.atleast_1d(np.asarray(u))
        elif backend == 'torch':
            import torch
            x_arr = torch.atleast_1d(torch.as_tensor(x))
            u_arr = torch.atleast_1d(torch.as_tensor(u))
        elif backend == 'jax':
            import jax.numpy as jnp
            x_arr = jnp.atleast_1d(jnp.asarray(x))
            u_arr = jnp.atleast_1d(jnp.asarray(u))
        
        # Prepare inputs for lambdified function (expects scalars/arrays unpacked)
        x_list = [x_arr[i] for i in range(self.nx)]
        u_list = [u_arr[i] for i in range(self.nu)]
        
        # Evaluate
        result = func(*(x_list + u_list))
        
        # Ensure proper array type and shape (nx, nw)
        if backend == 'numpy':
            import numpy as np
            result = np.atleast_2d(np.asarray(result))
            if result.shape[0] == 1 and self.nx > 1:
                result = result.T  # Transpose if needed
        elif backend == 'torch':
            import torch
            result = torch.atleast_2d(torch.as_tensor(result))
        elif backend == 'jax':
            import jax.numpy as jnp
            result = jnp.atleast_2d(jnp.asarray(result))
        
        return result
    
    # ========================================================================
    # Noise Query Methods - Delegate to NoiseCharacteristics
    # ========================================================================
    
    def is_additive_noise(self) -> bool:
        """
        Check if noise is additive (constant).
        
        Delegation: ✅ NoiseCharacteristics
        """
        return self.noise_characteristics.is_additive
    
    def is_diagonal_noise(self) -> bool:
        """
        Check if noise sources are independent.
        
        Delegation: ✅ NoiseCharacteristics
        """
        return self.noise_characteristics.is_diagonal
    
    def is_scalar_noise(self) -> bool:
        """
        Check if single noise source.
        
        Delegation: ✅ NoiseCharacteristics
        """
        return self.noise_characteristics.is_scalar
    
    def recommend_solvers(self, backend: str = 'jax') -> List[str]:
        """
        Recommend efficient solvers for this noise structure.
        
        Delegation: ✅ NoiseCharacteristics
        
        Examples
        --------
        >>> solvers = system.recommend_solvers('jax')
        >>> integrator = DiffraxSDEIntegrator(system, sde_solver=solvers[0])
        """
        return self.noise_characteristics.recommended_solvers(backend)
    
    def get_noise_intensity(self, backend: str = 'numpy'):
        """
        Get constant noise matrix (additive noise only).
        
        Delegation: ✅ DiffusionHandler
        
        Returns
        -------
        ArrayLike or None
            Constant diffusion matrix if additive, else None
        
        Raises
        ------
        ValueError
            If noise is not additive
        """
        return self.diffusion_handler.get_constant_noise(backend)
    
    # ========================================================================
    # Conversion & Utilities
    # ========================================================================
    
    def to_ode(self):
        """
        Convert to deterministic ODE (remove noise).
        
        Returns
        -------
        SymbolicDynamicalSystem
            Deterministic version with drift only
        """
        return SymbolicDynamicalSystem(
            dynamics_expr=self.drift_expr,
            state_vars=self.state_vars,
            control_vars=self.control_vars,
            parameters=self.parameters if hasattr(self, 'parameters') else None
        )
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get comprehensive system information.
        
        Combines:
        - Parent class info (drift, dimensions)
        - DiffusionHandler info (diffusion, noise)
        - NoiseCharacteristics (automatic analysis)
        
        Returns
        -------
        Dict[str, Any]
            Complete system description
        """
        # ✅ REUSE: Get base info from parent if available
        base_info = super().get_info() if hasattr(super(), 'get_info') else {}
        
        # ✅ DELEGATE: Get diffusion info from handler
        diffusion_info = self.diffusion_handler.get_info()
        
        # Merge and add SDE-specific info
        sde_info = {
            **base_info,
            'system_type': 'StochasticDynamicalSystem',
            'is_stochastic': True,
            'sde_type': self.sde_type.value,
            'num_wiener_processes': self.nw,
            'diffusion': diffusion_info,
            'noise': {
                'type': self.noise_characteristics.noise_type.value,
                'is_additive': self.noise_characteristics.is_additive,
                'is_diagonal': self.noise_characteristics.is_diagonal,
                'is_scalar': self.noise_characteristics.is_scalar,
                'depends_on': {
                    'state': self.noise_characteristics.depends_on_state,
                    'control': self.noise_characteristics.depends_on_control,
                    'time': self.noise_characteristics.depends_on_time,
                }
            },
            'recommended_solvers': {
                'jax': self.recommend_solvers('jax'),
                'torch': self.recommend_solvers('torch'),
                'julia': self.recommend_solvers('numpy'),
            }
        }
        
        return sde_info
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"StochasticDynamicalSystem(\n"
            f"  nx={self.nx}, nu={self.nu}, nw={self.nw},\n"
            f"  noise_type={self.noise_characteristics.noise_type.value},\n"
            f"  sde_type='{self.sde_type.value}',\n"
            f"  additive={self.noise_characteristics.is_additive}\n"
            f")"
        )
    
    def __str__(self) -> str:
        """Human-readable string."""
        return (
            f"SDE System: {self.nx} states, {self.nu} controls, {self.nw} noise sources "
            f"({self.noise_characteristics.noise_type.value} noise)"
        )