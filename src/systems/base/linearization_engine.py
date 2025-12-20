"""
Linearization Engine for SymbolicDynamicalSystem

Handles linearization of system dynamics across multiple backends.

Responsibilities:
- Linearized dynamics computation: A = ∂f/∂x, B = ∂f/∂u
- Symbolic linearization (state-space form for higher-order systems)
- Numerical linearization (all backends)
- Jacobian verification against autodiff
- Performance tracking

This class focuses ONLY on dynamics linearization (A, B matrices).
Output linearization (C matrix) is handled by ObservationEngine.
"""

from typing import Tuple, Optional, Dict, Union, TYPE_CHECKING
import time
import numpy as np
import sympy as sp

if TYPE_CHECKING:
    import torch
    import jax.numpy as jnp
    from src.systems.base.symbolic_dynamical_system import SymbolicDynamicalSystem
    from src.systems.base.code_generator import CodeGenerator
    from src.systems.base.backend_manager import BackendManager

# Type alias
ArrayLike = Union[np.ndarray, "torch.Tensor", "jnp.ndarray"]


class LinearizationEngine:
    """
    Computes linearized dynamics across backends.
    
    Handles computation of A = ∂f/∂x and B = ∂f/∂u matrices both symbolically
    and numerically, with support for higher-order systems.
    
    Example:
        >>> engine = LinearizationEngine(system, code_gen, backend_mgr)
        >>> A, B = engine.compute_dynamics(x, u, backend='numpy')
        >>> 
        >>> # Symbolic linearization
        >>> A_sym, B_sym = engine.compute_symbolic(x_eq, u_eq)
        >>> 
        >>> # Verify against autodiff
        >>> results = engine.verify_jacobians(x, u, backend='torch')
    """
    
    def __init__(
        self,
        system: 'SymbolicDynamicalSystem',
        code_gen: 'CodeGenerator',
        backend_mgr: 'BackendManager'
    ):
        """
        Initialize linearization engine.
        
        Args:
            system: The dynamical system
            code_gen: Code generator for accessing Jacobian functions
            backend_mgr: Backend manager for detection/conversion
        """
        self.system = system
        self.code_gen = code_gen
        self.backend_mgr = backend_mgr
        
        # Performance tracking
        self._stats = {
            'calls': 0,
            'time': 0.0,
        }
    
    # ========================================================================
    # Main Linearization API
    # ========================================================================
    
    def compute_dynamics(
        self, x: ArrayLike, u: ArrayLike, backend: Optional[str] = None
    ) -> Tuple[ArrayLike, ArrayLike]:
        """
        Compute linearized dynamics: A = ∂f/∂x, B = ∂f/∂u.
        
        Automatically detects backend from input types.
        
        Args:
            x: State at which to linearize
            u: Control at which to linearize
            backend: Backend selection (None = auto-detect)
            
        Returns:
            Tuple of (A, B) matrices (type matches backend)
            
        Example:
            >>> A, B = engine.compute_dynamics(x_torch, u_torch)  # Returns torch
            >>> A, B = engine.compute_dynamics(x_np, u_np, backend='torch')  # Converts to torch
        """
        # Determine target backend
        if backend == 'default':
            target_backend = self.backend_mgr.default_backend
        elif backend is None:
            target_backend = self.backend_mgr.detect(x)
        else:
            target_backend = backend
        
        # Convert inputs if needed
        input_backend = self.backend_mgr.detect(x)
        if input_backend != target_backend:
            x = self.backend_mgr.convert(x, target_backend)
            u = self.backend_mgr.convert(u, target_backend)
        
        # Dispatch to backend-specific implementation
        if target_backend == 'numpy':
            return self._compute_dynamics_numpy(x, u)
        elif target_backend == 'torch':
            return self._compute_dynamics_torch(x, u)
        elif target_backend == 'jax':
            return self._compute_dynamics_jax(x, u)
        else:
            raise ValueError(f"Unknown backend: {target_backend}")
    
    def compute_symbolic(
        self, x_eq: Optional[sp.Matrix] = None, u_eq: Optional[sp.Matrix] = None
    ) -> Tuple[sp.Matrix, sp.Matrix]:
        """
        Compute symbolic linearization A = ∂f/∂x, B = ∂f/∂u.
        
        For higher-order systems, constructs the full state-space linearization.
        
        Args:
            x_eq: Equilibrium state (zeros if None)
            u_eq: Equilibrium control (zeros if None)
            
        Returns:
            Tuple of (A, B) symbolic matrices
            
        Example:
            >>> A_sym, B_sym = engine.compute_symbolic(
            ...     x_eq=sp.Matrix([0, 0]),
            ...     u_eq=sp.Matrix([0])
            ... )
        """
        if x_eq is None:
            x_eq = sp.Matrix([0] * self.system.nx)
        if u_eq is None:
            u_eq = sp.Matrix([0] * self.system.nu)
        
        # Compute symbolic Jacobians (cached in CodeGenerator)
        self.code_gen._compute_symbolic_jacobians()
        A_sym_cached = self.code_gen._A_sym_cache
        B_sym_cached = self.code_gen._B_sym_cache
        
        if self.system.order == 1:
            # First-order system: straightforward Jacobian
            A_sym = A_sym_cached
            B_sym = B_sym_cached
        elif self.system.order == 2:
            # Second-order system: x = [q, q̇], q̈ = f(x, u)
            # Construct state-space form:
            # d/dt [q]   = [0   I] [q]   + [0]    u
            #      [q̇]     [A_q A_q̇] [q̇]    [B_accel]
            
            nq = self.system.nq
            
            # Compute Jacobians of acceleration w.r.t. q and q̇
            A_accel = self.system._f_sym.jacobian(self.system.state_vars)
            B_accel = B_sym_cached
            
            # Construct full state-space matrices
            A_sym = sp.zeros(self.system.nx, self.system.nx)
            A_sym[:nq, nq:] = sp.eye(nq)  # dq/dt = q̇
            A_sym[nq:, :] = A_accel        # dq̇/dt = f(q, q̇, u)
            
            B_sym = sp.zeros(self.system.nx, self.system.nu)
            B_sym[nq:, :] = B_accel  # Control affects acceleration
        else:
            # Higher-order systems: x = [q, q', q'', ..., q^(n-1)]
            nq = self.system.nq
            order = self.system.order
            
            A_highest = self.system._f_sym.jacobian(self.system.state_vars)
            B_highest = B_sym_cached
            
            A_sym = sp.zeros(self.system.nx, self.system.nx)
            # Each derivative becomes the next one
            for i in range(order - 1):
                A_sym[i * nq : (i + 1) * nq, (i + 1) * nq : (i + 2) * nq] = sp.eye(nq)
            # Highest derivative
            A_sym[(order - 1) * nq :, :] = A_highest
            
            B_sym = sp.zeros(self.system.nx, self.system.nu)
            B_sym[(order - 1) * nq :, :] = B_highest
        
        # Substitute equilibrium point
        subs_dict = dict(zip(
            self.system.state_vars + self.system.control_vars,
            list(x_eq) + list(u_eq)
        ))
        A = A_sym.subs(subs_dict)
        B = B_sym.subs(subs_dict)
        
        # Substitute parameters
        A = self.system.substitute_parameters(A)
        B = self.system.substitute_parameters(B)
        
        return A, B
    
    # ========================================================================
    # Backend-Specific Implementations
    # ========================================================================
    
    def _compute_dynamics_numpy(
        self, x: np.ndarray, u: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """NumPy implementation using cached functions or symbolic evaluation."""
        start_time = time.time()
        
        # Handle batched input
        if x.ndim == 1:
            x = np.expand_dims(x, 0)
            u = np.expand_dims(u, 0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        batch_size = x.shape[0]
        
        A_batch = np.zeros((batch_size, self.system.nx, self.system.nx))
        B_batch = np.zeros((batch_size, self.system.nx, self.system.nu))
        
        # Try to get cached Jacobian functions
        A_func, B_func, _ = self.code_gen.get_jacobians('numpy')
        
        if A_func is not None and B_func is not None:
            # Use cached functions
            for i in range(batch_size):
                x_i = x[i]
                u_i = u[i]
                
                x_list = [x_i[j] for j in range(self.system.nx)]
                u_list = [u_i[j] for j in range(self.system.nu)]
                all_args = x_list + u_list
                
                A_result = A_func(*all_args)
                B_result = B_func(*all_args)
                
                A_batch[i] = np.array(A_result, dtype=np.float64)
                B_batch[i] = np.array(B_result, dtype=np.float64)
        else:
            # Fall back to symbolic evaluation
            for i in range(batch_size):
                x_np = np.atleast_1d(x[i])
                u_np = np.atleast_1d(u[i])
                
                A_sym, B_sym = self.compute_symbolic(sp.Matrix(x_np), sp.Matrix(u_np))
                A_batch[i] = np.array(A_sym, dtype=np.float64)
                B_batch[i] = np.array(B_sym, dtype=np.float64)
        
        if squeeze_output:
            A_batch = np.squeeze(A_batch, 0)
            B_batch = np.squeeze(B_batch, 0)
        
        # Update performance stats
        self._stats['calls'] += 1
        self._stats['time'] += time.time() - start_time
        
        return A_batch, B_batch
    
    def _compute_dynamics_torch(
        self, x: "torch.Tensor", u: "torch.Tensor"
    ) -> Tuple["torch.Tensor", "torch.Tensor"]:
        """PyTorch implementation using cached functions or symbolic evaluation."""
        import torch
        
        start_time = time.time()
        
        # Handle batched input
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
            u = u.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        batch_size = x.shape[0]
        device = x.device
        dtype = x.dtype
        
        A_batch = torch.zeros(batch_size, self.system.nx, self.system.nx, dtype=dtype, device=device)
        B_batch = torch.zeros(batch_size, self.system.nx, self.system.nu, dtype=dtype, device=device)
        
        # Try to get cached Jacobian functions
        A_func, B_func, _ = self.code_gen.get_jacobians('torch')
        
        if A_func is not None and B_func is not None:
            # Use cached functions
            for i in range(batch_size):
                x_i = x[i]
                u_i = u[i]
                
                x_list = [x_i[j] for j in range(self.system.nx)]
                u_list = [u_i[j] for j in range(self.system.nu)]
                all_args = x_list + u_list
                
                A_batch[i] = A_func(*all_args)
                B_batch[i] = B_func(*all_args)
        else:
            # Fall back to symbolic evaluation
            for i in range(batch_size):
                x_i = x[i] if batch_size > 1 else x.squeeze(0)
                u_i = u[i] if batch_size > 1 else u.squeeze(0)
                
                x_np = x_i.detach().cpu().numpy()
                u_np = u_i.detach().cpu().numpy()
                
                x_np = np.atleast_1d(x_np)
                u_np = np.atleast_1d(u_np)
                
                A_sym, B_sym = self.compute_symbolic(sp.Matrix(x_np), sp.Matrix(u_np))
                A_batch[i] = torch.tensor(
                    np.array(A_sym, dtype=np.float64), dtype=dtype, device=device
                )
                B_batch[i] = torch.tensor(
                    np.array(B_sym, dtype=np.float64), dtype=dtype, device=device
                )
        
        if squeeze_output:
            A_batch = A_batch.squeeze(0)
            B_batch = B_batch.squeeze(0)
        
        # Update performance stats
        self._stats['calls'] += 1
        self._stats['time'] += time.time() - start_time
        
        return A_batch, B_batch
    
    def _compute_dynamics_jax(
        self, x: "jnp.ndarray", u: "jnp.ndarray"
    ) -> Tuple["jnp.ndarray", "jnp.ndarray"]:
        """JAX implementation using automatic differentiation."""
        import jax
        import jax.numpy as jnp
        
        start_time = time.time()
        
        # Ensure dynamics function is available
        f_jax = self.code_gen.generate_dynamics('jax', jit=True)
        
        # Handle batched input
        if x.ndim == 1:
            x = jnp.expand_dims(x, 0)
            u = jnp.expand_dims(u, 0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        # Define dynamics function for Jacobian computation
        def dynamics_fn(x_i, u_i):
            x_list = [x_i[j] for j in range(self.system.nx)]
            u_list = [u_i[j] for j in range(self.system.nu)]
            return f_jax(*(x_list + u_list))
        
        # Compute Jacobians using JAX autodiff (vmap for batching)
        @jax.vmap
        def compute_jacobians(x_i, u_i):
            A = jax.jacobian(lambda x: dynamics_fn(x, u_i))(x_i)
            B = jax.jacobian(lambda u: dynamics_fn(x_i, u))(u_i)
            return A, B
        
        A_batch, B_batch = compute_jacobians(x, u)
        
        if squeeze_output:
            A_batch = jnp.squeeze(A_batch, 0)
            B_batch = jnp.squeeze(B_batch, 0)
        
        # Update performance stats
        self._stats['calls'] += 1
        self._stats['time'] += time.time() - start_time
        
        return A_batch, B_batch
    
    # ========================================================================
    # Jacobian Verification
    # ========================================================================
    
    def verify_jacobians(
        self, x: ArrayLike, u: ArrayLike, backend: str = 'torch', tol: float = 1e-4
    ) -> Dict[str, Union[bool, float]]:
        """
        Verify symbolic Jacobians against automatic differentiation.
        
        Uses autodiff to numerically compute Jacobians and compares against
        symbolic derivation. Requires autodiff backend (torch or jax).
        
        Args:
            x: State at which to verify
            u: Control at which to verify
            backend: Backend for autodiff ('torch' or 'jax', not 'numpy')
            tol: Tolerance for considering Jacobians equal
            
        Returns:
            Dict with 'A_match', 'B_match' booleans and error magnitudes
            
        Raises:
            ValueError: If backend doesn't support autodiff
            
        Example:
            >>> results = engine.verify_jacobians(x, u, backend='torch', tol=1e-4)
            >>> assert results['A_match'] is True
            >>> assert results['B_match'] is True
        """
        if backend not in ['torch', 'jax']:
            raise ValueError(
                f"Jacobian verification requires autodiff backend ('torch' or 'jax'), "
                f"got '{backend}'. NumPy doesn't support automatic differentiation."
            )
        
        # Check backend availability
        self.backend_mgr.require_backend(backend)
        
        # Dispatch to backend-specific verification
        if backend == 'torch':
            return self._verify_jacobians_torch(x, u, tol)
        else:  # jax
            return self._verify_jacobians_jax(x, u, tol)
    
    def _verify_jacobians_torch(
        self, x: ArrayLike, u: ArrayLike, tol: float
    ) -> Dict[str, Union[bool, float]]:
        """PyTorch-based Jacobian verification."""
        import torch
        
        # Convert to torch if needed
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(np.asarray(x), dtype=torch.float32)
        if not isinstance(u, torch.Tensor):
            u = torch.tensor(np.asarray(u), dtype=torch.float32)
        
        # Ensure proper 2D shape
        x_2d = x.reshape(1, -1) if len(x.shape) <= 1 else x
        u_2d = u.reshape(1, -1) if len(u.shape) <= 1 else u
        
        # Clone for autograd
        x_grad = x_2d.clone().requires_grad_(True)
        u_grad = u_2d.clone().requires_grad_(True)
        
        # Compute symbolic Jacobians
        A_sym, B_sym = self.compute_dynamics(x_2d.detach(), u_2d.detach(), backend='torch')
        
        # Ensure 3D shape for batch processing
        if len(A_sym.shape) == 2:
            A_sym = A_sym.unsqueeze(0)
            B_sym = B_sym.unsqueeze(0)
        
        # Compute numerical Jacobians via autograd
        # Need to import dynamics evaluator to get forward function
        from src.systems.base.dynamics_evaluator import DynamicsEvaluator
        dynamics_eval = DynamicsEvaluator(self.system, self.code_gen, self.backend_mgr)
        fx = dynamics_eval.evaluate(x_grad, u_grad, backend='torch')
        
        # Ensure fx is 2D for consistent indexing
        if fx.ndim == 1:
            fx = fx.unsqueeze(1)
        elif fx.ndim == 0:
            fx = fx.reshape(1, 1)
        
        # Determine output dimension
        n_outputs = self.system.nq if self.system.order > 1 else self.system.nx
        
        # Compute gradients
        A_num = torch.zeros_like(A_sym)
        B_num = torch.zeros_like(B_sym)
        
        if self.system.order == 1:
            # First-order: verify full A and B
            for i in range(n_outputs):
                if fx[0, i].requires_grad:
                    grad_x = torch.autograd.grad(
                        fx[0, i], x_grad, retain_graph=True, create_graph=False
                    )[0]
                    grad_u = torch.autograd.grad(
                        fx[0, i], u_grad, retain_graph=True, create_graph=False
                    )[0]
                    A_num[0, i] = grad_x[0]
                    B_num[0, i] = grad_u[0]
        else:
            # Higher-order: verify acceleration part
            for i in range(n_outputs):
                if fx[0, i].requires_grad:
                    grad_x = torch.autograd.grad(
                        fx[0, i], x_grad, retain_graph=True, create_graph=False
                    )[0]
                    grad_u = torch.autograd.grad(
                        fx[0, i], u_grad, retain_graph=True, create_graph=False
                    )[0]
                    
                    row_idx = (self.system.order - 1) * self.system.nq + i
                    A_num[0, row_idx] = grad_x[0]
                    B_num[0, row_idx] = grad_u[0]
            
            # Copy derivative relationships
            for i in range((self.system.order - 1) * self.system.nq):
                A_num[0, i] = A_sym[0, i]
                B_num[0, i] = B_sym[0, i]
        
        # Compute errors
        A_error = (A_sym - A_num).abs().max().item()
        B_error = (B_sym - B_num).abs().max().item()
        
        return {
            'A_match': bool(A_error < tol),
            'B_match': bool(B_error < tol),
            'A_error': float(A_error),
            'B_error': float(B_error),
        }
    
    def _verify_jacobians_jax(
        self, x: ArrayLike, u: ArrayLike, tol: float
    ) -> Dict[str, Union[bool, float]]:
        """JAX-based Jacobian verification."""
        import jax.numpy as jnp
        
        # Convert to JAX if needed
        if not isinstance(x, jnp.ndarray):
            x = jnp.array(np.asarray(x))
        if not isinstance(u, jnp.ndarray):
            u = jnp.array(np.asarray(u))
        
        # Ensure proper shape
        x_2d = x.reshape(1, -1) if x.ndim <= 1 else x
        u_2d = u.reshape(1, -1) if u.ndim <= 1 else u
        
        # Compute Jacobians using JAX autodiff
        A_jax, B_jax = self.compute_dynamics(x_2d, u_2d, backend='jax')
        
        # Compute symbolic Jacobians as ground truth
        x_np = np.array(x_2d[0])
        u_np = np.array(u_2d[0])
        A_sym, B_sym = self.compute_symbolic(sp.Matrix(x_np), sp.Matrix(u_np))
        A_sym_np = np.array(A_sym, dtype=np.float64)
        B_sym_np = np.array(B_sym, dtype=np.float64)
        
        # Convert JAX results to NumPy for comparison
        A_jax_np = np.array(A_jax)
        B_jax_np = np.array(B_jax)
        
        # Compute errors
        A_error = np.abs(A_sym_np - A_jax_np).max()
        B_error = np.abs(B_sym_np - B_jax_np).max()
        
        return {
            'A_match': bool(A_error < tol),
            'B_match': bool(B_error < tol),
            'A_error': float(A_error),
            'B_error': float(B_error),
        }
    
    # ========================================================================
    # Performance Tracking
    # ========================================================================
    
    def get_stats(self) -> dict:
        """
        Get performance statistics.
        
        Returns:
            Dict with call count, total time, and average time
        """
        return {
            'calls': self._stats['calls'],
            'total_time': self._stats['time'],
            'avg_time': self._stats['time'] / max(1, self._stats['calls']),
        }
    
    def reset_stats(self):
        """Reset performance counters."""
        self._stats['calls'] = 0
        self._stats['time'] = 0.0
    
    # ========================================================================
    # String Representations
    # ========================================================================
    
    def __repr__(self) -> str:
        """String representation for debugging"""
        return (
            f"LinearizationEngine("
            f"nx={self.system.nx}, nu={self.system.nu}, "
            f"calls={self._stats['calls']})"
        )
    
    def __str__(self) -> str:
        """Human-readable string"""
        return f"LinearizationEngine(calls={self._stats['calls']})"