from typing import Callable, Literal, Union
import sympy as sp
import numpy as np

Backend = Literal["numpy", "torch", "jax"]

def _torch_min(*args):
    """Handle Min for both scalars and tensors"""
    import torch
    if len(args) == 2:
        a, b = args
        a_tensor = torch.as_tensor(a) if not isinstance(a, torch.Tensor) else a
        b_tensor = torch.as_tensor(b) if not isinstance(b, torch.Tensor) else b
        return torch.minimum(a_tensor, b_tensor)
    else:
        tensors = [
            torch.as_tensor(a) if not isinstance(a, torch.Tensor) else a for a in args
        ]
        return torch.min(torch.stack(tensors))

def _torch_max(*args):
    """Handle Max for both scalars and tensors"""
    import torch
    if len(args) == 2:
        a, b = args
        a_tensor = torch.as_tensor(a) if not isinstance(a, torch.Tensor) else a
        b_tensor = torch.as_tensor(b) if not isinstance(b, torch.Tensor) else b
        return torch.maximum(a_tensor, b_tensor)
    else:
        tensors = [
            torch.as_tensor(a) if not isinstance(a, torch.Tensor) else a for a in args
        ]
        return torch.max(torch.stack(tensors))

def _identity_matrix(*args):
    """Handle ImmutableDenseMatrix - just return the args as tuple"""
    return args if len(args) > 1 else args[0]

# Standard mapping from SymPy functions to PyTorch functions
SYMPY_TO_TORCH = {
    # Trigonometric
    "sin": "torch.sin",
    "cos": "torch.cos",
    "tan": "torch.tan",
    "asin": "torch.asin",
    "acos": "torch.acos",
    "atan": "torch.atan",
    "atan2": "torch.atan2",
    "sinh": "torch.sinh",
    "cosh": "torch.cosh",
    "tanh": "torch.tanh",
    # Exponential/Logarithmic
    "exp": "torch.exp",
    "log": "torch.log",
    "sqrt": "torch.sqrt",
    # Absolute value and sign
    "Abs": "torch.abs",
    "abs": "torch.abs",
    "sign": "torch.sign",
    # Min/Max - use helper functions
    "Min": "_torch_min",
    "Max": "_torch_max",
    # Power
    "Pow": "torch.pow",
    # Rounding
    "floor": "torch.floor",
    "ceil": "torch.ceil",
    "round": "torch.round",
    # Additional
    "clip": "torch.clamp",
    "minimum": "torch.minimum",
    "maximum": "torch.maximum",
    # Matrix handling
    "ImmutableDenseMatrix": "_identity_matrix",
    "MutableDenseMatrix": "_identity_matrix",
    "Matrix": "_identity_matrix",
}

SYMPY_TO_TORCH_LAMBDIFY = {
    # For lambdify, we need actual function objects
    "sin": lambda: __import__('torch').sin,
    "cos": lambda: __import__('torch').cos,
    "tan": lambda: __import__('torch').tan,
    "asin": lambda: __import__('torch').asin,
    "acos": lambda: __import__('torch').acos,
    "atan": lambda: __import__('torch').atan,
    "atan2": lambda: __import__('torch').atan2,
    "sinh": lambda: __import__('torch').sinh,
    "cosh": lambda: __import__('torch').cosh,
    "tanh": lambda: __import__('torch').tanh,
    "exp": lambda: __import__('torch').exp,
    "log": lambda: __import__('torch').log,
    "sqrt": lambda: __import__('torch').sqrt,
    "Abs": lambda: __import__('torch').abs,
    "abs": lambda: __import__('torch').abs,
    "sign": lambda: __import__('torch').sign,
    "Min": _torch_min,
    "Max": _torch_max,
    "Pow": lambda: __import__('torch').pow,
    "floor": lambda: __import__('torch').floor,
    "ceil": lambda: __import__('torch').ceil,
    "round": lambda: __import__('torch').round,
    "clip": lambda: __import__('torch').clamp,
    "minimum": lambda: __import__('torch').minimum,
    "maximum": lambda: __import__('torch').maximum,
    "ImmutableDenseMatrix": _identity_matrix,
    "MutableDenseMatrix": _identity_matrix,
    "Matrix": _identity_matrix,
}


def generate_numpy_function(
    expr: Union[sp.Expr, list[sp.Expr], sp.Matrix],
    symbols: list[sp.Symbol],
) -> Callable:
    """
    Generate a NumPy function from SymPy expression(s).
    
    Args:
        expr: SymPy expression, list of expressions, or Matrix
        symbols: Input symbols in order
    
    Returns:
        Compiled NumPy function
    """
    if isinstance(expr, list):
        expr = sp.Matrix(expr)
    
    func = sp.lambdify(symbols, expr, modules="numpy")
    return func


def generate_torch_function(
    expr: Union[sp.Expr, list[sp.Expr], sp.Matrix],
    symbols: list[sp.Symbol],
    method: Literal["lambdify", "codegen"] = "lambdify",
) -> Callable:
    """
    Generate a PyTorch function from SymPy expression(s).
    
    Args:
        expr: SymPy expression, list of expressions, or Matrix
        symbols: Input symbols in order
        method: 'lambdify' (faster) or 'codegen' (more compatible)
    
    Returns:
        Compiled PyTorch function
    """
    import torch
    
    if isinstance(expr, list):
        expr = sp.Matrix(expr)
    
    if method == "lambdify":
        # Prepare modules dict with actual function objects
        modules_dict = {k: v() if callable(v) else v 
                       for k, v in SYMPY_TO_TORCH_LAMBDIFY.items()}
        
        func = sp.lambdify(symbols, expr, modules=[modules_dict])
        
        # Wrap to ensure proper tensor handling
        def wrapped_func(*args):
            result = func(*args)
            
            if isinstance(result, (list, tuple)):
                return torch.stack(list(result), dim=-1)
            elif isinstance(result, torch.Tensor):
                if len(result.shape) == 1:
                    return result.unsqueeze(-1)
                return result
            else:
                return torch.tensor([result]).unsqueeze(0)
        
        return wrapped_func
    
    elif method == "codegen":
        # Use code generation approach (your current method)
        from sympy import pycode
        
        # Generate function signature
        func_code_lines = [
            "def dynamics_func(" + ", ".join([str(v) for v in symbols]) + "):",
            "    import torch",
        ]
        
        # Generate code for each output component
        if isinstance(expr, sp.Matrix):
            results = []
            for i, e in enumerate(expr):
                code = pycode(e)
                # Replace module prefixes
                code = code.replace("numpy.", "torch.")
                code = code.replace("math.", "torch.")
                
                var_name = f"result_{i}"
                func_code_lines.append(f"    {var_name} = {code}")
                results.append(var_name)
            
            func_code_lines.append(f"    return ({', '.join(results)},)")
        else:
            code = pycode(expr)
            code = code.replace("numpy.", "torch.")
            code = code.replace("math.", "torch.")
            func_code_lines.append(f"    return {code}")
        
        func_code = "\n".join(func_code_lines)
        
        # Execute generated code
        namespace = {"torch": torch}
        exec(func_code, namespace)
        base_func = namespace["dynamics_func"]
        
        # Wrap to ensure proper tensor handling
        def wrapped_func(*args):
            result = base_func(*args)
            
            if isinstance(result, (list, tuple)):
                return torch.stack(list(result), dim=-1)
            elif isinstance(result, torch.Tensor):
                if len(result.shape) == 1:
                    return result.unsqueeze(-1)
                return result
            else:
                return torch.tensor([result]).unsqueeze(0)
        
        return wrapped_func
    
    else:
        raise ValueError(f"Unknown method: {method}")


def generate_jax_function(
    expr: Union[sp.Expr, list[sp.Expr], sp.Matrix],
    symbols: list[sp.Symbol],
    jit: bool = True,
) -> Callable:
    """
    Generate a JAX function from SymPy expression(s).
    
    Args:
        expr: SymPy expression, list of expressions, or Matrix
        symbols: Input symbols in order
        jit: Whether to JIT-compile the function
    
    Returns:
        Compiled JAX function
    """
    import jax
    import jax.numpy as jnp
    
    if isinstance(expr, list):
        expr = sp.Matrix(expr)
    
    func = sp.lambdify(
        symbols,
        expr,
        modules=[{
            'ImmutableDenseMatrix': lambda x: jnp.stack(list(x)),
            'MutableDenseMatrix': lambda x: jnp.stack(list(x)),
            'Matrix': lambda x: jnp.stack(list(x)),
        }, 'jax']
    )
    
    if jit:
        func = jax.jit(func)
    
    return func


def generate_function(
    expr: Union[sp.Expr, list[sp.Expr], sp.Matrix],
    symbols: list[sp.Symbol],
    backend: Backend = "numpy",
    **kwargs
) -> Callable:
    """
    Generate a function from SymPy expression for specified backend.
    
    Args:
        expr: SymPy expression, list of expressions, or Matrix
        symbols: Input symbols in order
        backend: Target backend ('numpy', 'torch', 'jax')
        **kwargs: Backend-specific options
    
    Returns:
        Compiled function for the specified backend
    """
    if backend == "numpy":
        return generate_numpy_function(expr, symbols)
    elif backend == "torch":
        return generate_torch_function(expr, symbols, **kwargs)
    elif backend == "jax":
        return generate_jax_function(expr, symbols, **kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend}")


def generate_jacobian_function(
    expr: Union[sp.Expr, list[sp.Expr], sp.Matrix],
    symbols: list[sp.Symbol],
    wrt_symbols: list[sp.Symbol],
    backend: Backend = "numpy",
    use_symbolic: bool = True,
    **kwargs
) -> Callable:
    """
    Generate Jacobian function.
    
    Args:
        expr: SymPy expression, list of expressions, or Matrix
        symbols: All input symbols in order
        wrt_symbols: Symbols to differentiate with respect to
        backend: Target backend
        use_symbolic: If True, compute Jacobian symbolically then compile.
                     If False (JAX only), use automatic differentiation.
        **kwargs: Backend-specific options
    
    Returns:
        Compiled Jacobian function
    """
    if use_symbolic:
        # Compute Jacobian symbolically
        if isinstance(expr, list):
            expr = sp.Matrix(expr)
        
        jacobian = expr.jacobian(wrt_symbols)
        return generate_function(jacobian, symbols, backend, **kwargs)
    
    else:
        # Use automatic differentiation (JAX only)
        if backend != "jax":
            raise ValueError("Automatic differentiation only supported for JAX backend")
        
        import jax
        
        # First generate the base function
        base_func = generate_jax_function(expr, symbols, jit=False)
        
        # Determine which argument indices correspond to wrt_symbols
        wrt_indices = [symbols.index(s) for s in wrt_symbols]
        
        # Create wrapper that computes Jacobian
        def jac_func(*args):
            # Define function of just the wrt variables
            def f_wrt(*wrt_vals):
                # Reconstruct full argument list
                full_args = list(args)
                for i, idx in enumerate(wrt_indices):
                    full_args[idx] = wrt_vals[i]
                return base_func(*full_args)
            
            # Stack wrt arguments for jacobian
            wrt_args = tuple(args[i] for i in wrt_indices)
            return jax.jacobian(f_wrt)(*wrt_args)
        
        if kwargs.get('jit', True):
            jac_func = jax.jit(jac_func)
        
        return jac_func