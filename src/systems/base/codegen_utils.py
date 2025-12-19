"""
Backend-agnostic code generation utilities.

All backends (NumPy, PyTorch, JAX) have equal support for:
- Matrix handling (ImmutableDenseMatrix conversion)
- Min/Max functions (variable argument handling)
- Consistent shape conventions (always return 1D arrays)

Backend-specific features (not technical debt):
- PyTorch: Gradient preservation via torch.as_tensor()
- JAX: JIT compilation support
- NumPy: Simplest implementation (reference)

Design Philosophy:
All backends are equally capable. Differences reflect the backends'
intrinsic capabilities (gradients, JIT, etc.), not implementation gaps.
"""

from typing import Callable, Literal, Union
import sympy as sp
import numpy as np
import torch

Backend = Literal["numpy", "torch", "jax"]


# Helper functions


def _numpy_min(*args):
    """
    Handle SymPy Min for NumPy backend.

    SymPy's Min can take arbitrary number of arguments: Min(x, y, z)
    NumPy's np.minimum only takes 2 arguments.

    Args:
        *args: Variable number of arguments

    Returns:
        Minimum value (scalar or array)

    Examples:
        >>> _numpy_min(1, 2, 3)
        1
        >>> _numpy_min(np.array([1, 2]), np.array([3, 0]))
        array([1, 0])
    """
    import numpy as np

    if len(args) == 0:
        raise ValueError("Min requires at least one argument")
    elif len(args) == 1:
        return args[0]
    elif len(args) == 2:
        return np.minimum(args[0], args[1])
    else:
        # More than 2 args: reduce iteratively
        result = args[0]
        for arg in args[1:]:
            result = np.minimum(result, arg)
        return result


def _numpy_max(*args):
    """
    Handle SymPy Max for NumPy backend.

    SymPy's Max can take arbitrary number of arguments: Max(x, y, z)
    NumPy's np.maximum only takes 2 arguments.

    Args:
        *args: Variable number of arguments

    Returns:
        Maximum value (scalar or array)

    Examples:
        >>> _numpy_max(1, 2, 3)
        3
        >>> _numpy_max(np.array([1, 2]), np.array([3, 0]))
        array([3, 2])
    """
    import numpy as np

    if len(args) == 0:
        raise ValueError("Max requires at least one argument")
    elif len(args) == 1:
        return args[0]
    elif len(args) == 2:
        return np.maximum(args[0], args[1])
    else:
        # More than 2 args: reduce iteratively
        result = args[0]
        for arg in args[1:]:
            result = np.maximum(result, arg)
        return result


def _torch_min(*args):
    """
    Handle SymPy Min for PyTorch backend.

    SymPy's Min can take arbitrary number of arguments: Min(x, y, z)
    Handles both scalars and tensors, preserving gradients.

    Args:
        *args: Variable number of arguments (scalars or tensors)

    Returns:
        Minimum value (scalar or tensor)

    Examples:
        >>> _torch_min(torch.tensor(1.0), torch.tensor(2.0))
        tensor(1.0)
        >>> _torch_min(1.0, 2.0, 3.0)
        tensor(1.0)
    """
    import torch

    if len(args) == 0:
        raise ValueError("Min requires at least one argument")
    elif len(args) == 1:
        return torch.as_tensor(args[0]) if not isinstance(args[0], torch.Tensor) else args[0]
    elif len(args) == 2:
        a, b = args
        a_tensor = torch.as_tensor(a) if not isinstance(a, torch.Tensor) else a
        b_tensor = torch.as_tensor(b) if not isinstance(b, torch.Tensor) else b
        return torch.minimum(a_tensor, b_tensor)
    else:
        # More than 2 args: convert all to tensors and reduce
        tensors = [torch.as_tensor(a) if not isinstance(a, torch.Tensor) else a for a in args]
        result = tensors[0]
        for t in tensors[1:]:
            result = torch.minimum(result, t)
        return result


def _torch_max(*args):
    """
    Handle SymPy Max for PyTorch backend.

    SymPy's Max can take arbitrary number of arguments: Max(x, y, z)
    Handles both scalars and tensors, preserving gradients.

    Args:
        *args: Variable number of arguments (scalars or tensors)

    Returns:
        Maximum value (scalar or tensor)

    Examples:
        >>> _torch_max(torch.tensor(1.0), torch.tensor(2.0))
        tensor(2.0)
        >>> _torch_max(1.0, 2.0, 3.0)
        tensor(3.0)
    """
    import torch

    if len(args) == 0:
        raise ValueError("Max requires at least one argument")
    elif len(args) == 1:
        return torch.as_tensor(args[0]) if not isinstance(args[0], torch.Tensor) else args[0]
    elif len(args) == 2:
        a, b = args
        a_tensor = torch.as_tensor(a) if not isinstance(a, torch.Tensor) else a
        b_tensor = torch.as_tensor(b) if not isinstance(b, torch.Tensor) else b
        return torch.maximum(a_tensor, b_tensor)
    else:
        # More than 2 args: convert all to tensors and reduce
        tensors = [torch.as_tensor(a) if not isinstance(a, torch.Tensor) else a for a in args]
        result = tensors[0]
        for t in tensors[1:]:
            result = torch.maximum(result, t)
        return result


def _jax_min(*args):
    """
    Handle SymPy Min for JAX backend.

    SymPy's Min can take arbitrary number of arguments: Min(x, y, z)
    JAX's jnp.minimum only takes 2 arguments.

    Args:
        *args: Variable number of arguments

    Returns:
        Minimum value (scalar or array)

    Examples:
        >>> _jax_min(1.0, 2.0, 3.0)
        Array(1.0, dtype=float32)
        >>> _jax_min(jnp.array([1, 2]), jnp.array([3, 0]))
        Array([1, 0], dtype=int32)
    """
    import jax.numpy as jnp

    if len(args) == 0:
        raise ValueError("Min requires at least one argument")
    elif len(args) == 1:
        return jnp.asarray(args[0])
    elif len(args) == 2:
        return jnp.minimum(args[0], args[1])
    else:
        # More than 2 args: reduce iteratively
        result = args[0]
        for arg in args[1:]:
            result = jnp.minimum(result, arg)
        return result


def _jax_max(*args):
    """
    Handle SymPy Max for JAX backend.

    SymPy's Max can take arbitrary number of arguments: Max(x, y, z)
    JAX's jnp.maximum only takes 2 arguments.

    Args:
        *args: Variable number of arguments

    Returns:
        Maximum value (scalar or array)

    Examples:
        >>> _jax_max(1.0, 2.0, 3.0)
        Array(3.0, dtype=float32)
        >>> _jax_max(jnp.array([1, 2]), jnp.array([3, 0]))
        Array([3, 2], dtype=int32)
    """
    import jax.numpy as jnp

    if len(args) == 0:
        raise ValueError("Max requires at least one argument")
    elif len(args) == 1:
        return jnp.asarray(args[0])
    elif len(args) == 2:
        return jnp.maximum(args[0], args[1])
    else:
        # More than 2 args: reduce iteratively
        result = args[0]
        for arg in args[1:]:
            result = jnp.maximum(result, arg)
        return result


def _numpy_matrix_handler(data):
    """
    Handle SymPy Matrix objects for NumPy backend.

    SymPy's lambdify returns ImmutableDenseMatrix as nested lists:
    [[x], [y]] → Should become [x, y]
    """
    if isinstance(data, (list, tuple)):
        if len(data) > 0 and isinstance(data[0], (list, tuple)):
            # Flatten one level: [[x], [y]] → [x, y]
            return [
                item[0] if isinstance(item, (list, tuple)) and len(item) == 1 else item
                for item in data
            ]
        else:
            return data
    else:
        return data


def _torch_matrix_handler(data):
    """
    Handle SymPy Matrix objects for PyTorch backend.

    SymPy's lambdify returns ImmutableDenseMatrix as nested lists:
    [[x], [y]] → Should become [x, y] for torch.stack()
    """
    if isinstance(data, (list, tuple)):
        # Flatten one level: [[x], [y]] -> [x, y]
        if len(data) > 0 and isinstance(data[0], (list, tuple)):
            # Nested list - flatten
            return [
                item[0] if isinstance(item, (list, tuple)) and len(item) == 1 else item
                for item in data
            ]
        else:
            # Already flat
            return data
    else:
        # Single item
        return data


def _jax_matrix_handler(data):
    """
    Handle SymPy Matrix objects for JAX backend.

    Converts SymPy Matrix to JAX array format.
    """
    import jax.numpy as jnp

    if hasattr(data, "__len__") and hasattr(data, "__getitem__"):
        try:
            elements = [data[i] for i in range(len(data))]
            return jnp.stack([jnp.asarray(e) for e in elements])
        except:
            return data
    else:
        return data


# NumPy mappings (minimal - only matrix handling needed)

SYMPY_TO_NUMPY_LAMBDIFY = {
    # Min/Max handling
    "Min": _numpy_min,
    "Max": _numpy_max,
    # Matrix handling
    "ImmutableDenseMatrix": _numpy_matrix_handler,
    "MutableDenseMatrix": _numpy_matrix_handler,
    "Matrix": _numpy_matrix_handler,
}


# PyTorch mapping for lambdify!
SYMPY_TO_TORCH_LAMBDIFY = {
    # Trigonometric
    "sin": torch.sin,
    "cos": torch.cos,
    "tan": torch.tan,
    "asin": torch.asin,
    "acos": torch.acos,
    "atan": torch.atan,
    "atan2": torch.atan2,
    "sinh": torch.sinh,
    "cosh": torch.cosh,
    "tanh": torch.tanh,
    # Exponential/Logarithmic
    "exp": torch.exp,
    "log": torch.log,
    "sqrt": torch.sqrt,
    # Absolute value and sign
    "Abs": torch.abs,
    "abs": torch.abs,
    "sign": torch.sign,
    # Min/Max - use helper functions directly
    "Min": _torch_min,
    "Max": _torch_max,
    # Power
    "Pow": torch.pow,
    # Rounding
    "floor": torch.floor,
    "ceil": torch.ceil,
    "round": torch.round,
    # Additional
    "clip": torch.clamp,
    "minimum": torch.minimum,
    "maximum": torch.maximum,
    # # Matrix handling
    "ImmutableDenseMatrix": _torch_matrix_handler,
    "MutableDenseMatrix": _torch_matrix_handler,
    "Matrix": _torch_matrix_handler,
}

# JAX mappings (minimal - only matrix handling needed)
SYMPY_TO_JAX_LAMBDIFY = {
    # Min/Max handling
    "Min": _jax_min,
    "Max": _jax_max,
    # Matrix handling
    "ImmutableDenseMatrix": _jax_matrix_handler,
    "MutableDenseMatrix": _jax_matrix_handler,
    "Matrix": _jax_matrix_handler,
}


def generate_numpy_function(
    expr: Union[sp.Expr, list[sp.Expr], sp.Matrix],
    symbols: list[sp.Symbol],
) -> Callable:
    """
    Generate a NumPy function from SymPy expression(s).

    Converts symbolic expressions to executable NumPy functions.
    Handles scalars, vectors, and matrices.

    Args:
        expr: SymPy expression, list, or Matrix
        symbols: Input symbols in order

    Returns:
        Compiled Numpy function

    Return Type Convention:
        All functions return 1D arrays, even for scalar expressions:
        - Scalar expr: returns shape (1,)
        - Vector expr: returns shape (n,)

        Extract scalar: result[0] or result.item()
    """
    # Convert to matrix for consistent handling
    if isinstance(expr, list):
        expr = sp.Matrix(expr)
    elif not isinstance(expr, sp.Matrix):
        expr = sp.Matrix([expr])

    func = sp.lambdify(symbols, expr, modules=[SYMPY_TO_NUMPY_LAMBDIFY, "numpy"])

    # Wrap to ensure proper array handling
    def wrapped_func(*args):
        result = func(*args)

        # Handle ImmutableDenseMatrix from SymPy
        if hasattr(result, "__class__") and "Matrix" in str(result.__class__):
            result = np.array([result[i] for i in range(len(result))]).flatten()

        # Handle different return types
        if isinstance(result, np.ndarray):
            if result.ndim == 0:
                return np.array([result])
            elif result.ndim == 1:
                return result
            elif result.ndim == 2 and result.shape[1] == 1:
                return result.flatten()
            else:
                return result.flatten()
        elif isinstance(result, np.matrix):
            # Convert matrix to array and flatten
            return np.asarray(result).flatten()
        elif isinstance(result, (list, tuple)):
            return np.array(result).flatten()
        else:
            return np.array([result])

    return wrapped_func


def generate_torch_function(expr, symbols):
    """
    Generate a PyTorch function from SymPy expression(s).

    Converts symbolic expressions to executable PyTorch functions.
    Preserves gradient tracking for autodifferentiation.
    Handles scalars, vectors, and matrices.

    Args:
        expr: SymPy expression, list, or Matrix
        symbols: Input symbols in order

    Returns:
        Compiled PyTorch function

    Return Type Convention:
        All functions return 1D arrays, even for scalar expressions:
        - Scalar expr: returns shape (1,)
        - Vector expr: returns shape (n,)

        Extract scalar: result[0] or result.item()
    """
    if isinstance(expr, list):
        expr = sp.Matrix(expr)
    elif not isinstance(expr, sp.Matrix):
        expr = sp.Matrix([expr])

    func = sp.lambdify(symbols, expr, modules=[SYMPY_TO_TORCH_LAMBDIFY])

    def wrapped_func(*args):
        result = func(*args)

        # SIMPLIFIED - match NumPy's behavior
        if isinstance(result, torch.Tensor):
            # Return as-is if already a tensor
            return result.flatten() if result.ndim > 1 else result

        elif isinstance(result, (list, tuple)):
            # Convert list to tensor (like NumPy does to array)
            tensors = []
            for item in result:
                if isinstance(item, torch.Tensor):
                    t = item if item.ndim > 0 else item.reshape(1)
                else:
                    t = torch.as_tensor(item, dtype=torch.float32)
                    t = t if t.ndim > 0 else t.reshape(1)
                tensors.append(t)

            if len(tensors) == 1:
                return tensors[0]
            else:
                # Stack and flatten (like NumPy)
                return torch.stack(tensors).flatten()

        else:
            return torch.tensor([result], dtype=torch.float32)

    return wrapped_func


def generate_jax_function(
    expr: Union[sp.Expr, list[sp.Expr], sp.Matrix],
    symbols: list[sp.Symbol],
    jit: bool = True,
) -> Callable:
    """
    Generate a JAX function from SymPy expression(s).

    Converts symbolic expressions to executable JAX functions.
    Supports JIT compilation for improved performance.
    Handles scalars, vectors, and matrices.

    Args:
        expr: SymPy expression, list, or Matrix
        symbols: Input symbols in order
        jit: Whether to JIT-compile the function (default: True)

    Returns:
        Compiled JAX function

    Return Type Convention:
        All functions return 1D arrays, even for scalar expressions:
        - Scalar expr: returns shape (1,)
        - Vector expr: returns shape (n,)

        Extract scalar: result[0] or result.item()
    """
    import jax
    import jax.numpy as jnp

    # Convert to matrix for consistent handling
    if isinstance(expr, list):
        expr = sp.Matrix(expr)
    elif not isinstance(expr, sp.Matrix):
        expr = sp.Matrix([expr])

    func = sp.lambdify(
        symbols,
        expr,
        modules=[SYMPY_TO_JAX_LAMBDIFY, "jax"],
    )

    # Wrap to ensure proper array handling
    def wrapped_func(*args):
        result = func(*args)

        # Handle ImmutableDenseMatrix from SymPy
        if hasattr(result, "__class__") and "Matrix" in str(result.__class__):
            result = [result[i] for i in range(len(result))]

        # Handle different return types
        if isinstance(result, jnp.ndarray):
            if result.ndim == 0:
                return jnp.array([result])
            elif result.ndim == 1:
                return result
            elif result.ndim == 2 and result.shape[1] == 1:
                return result.flatten()
            else:
                return result.flatten()
        elif isinstance(result, (list, tuple)):
            arrays = []
            for item in result:
                if isinstance(item, jnp.ndarray):
                    arrays.append(item)
                else:
                    arrays.append(jnp.asarray(item, dtype=jnp.float32))

            if len(arrays) == 0:
                raise ValueError("Empty result from lambdify")
            elif len(arrays) == 1:
                return jnp.atleast_1d(arrays[0])
            else:
                return jnp.stack(arrays, axis=0)
        else:
            return jnp.array([result])

    if jit:
        wrapped_func = jax.jit(wrapped_func)

    return wrapped_func


def generate_function(
    expr: Union[sp.Expr, list[sp.Expr], sp.Matrix],
    symbols: list[sp.Symbol],
    backend: Backend = "numpy",
    **kwargs,
) -> Callable:
    """
    Generate a function from SymPy expression for specified backend.

    Supports scalar expressions, vector expressions, and matrix expressions.
    All backends handle matrix types consistently.

    Args:
        expr: SymPy expression, list of expressions, or Matrix
        symbols: Input symbols in order
        backend: 'numpy', 'torch', or 'jax'
        **kwargs: Backend-specific options (e.g., jit=True for JAX)

    Returns:
        Compiled function that evaluates the expression

    Return Type Convention:
        All functions return 1D arrays, even for scalar expressions:
        - Scalar expr: returns shape (1,)
        - Vector expr: returns shape (n,)

        Extract scalar: result[0] or result.item()

    Examples:
        >>> x, y = sp.symbols('x y')
        >>>
        >>> # Scalar expression (must be extracted)
        >>> expr = x**2 + y**2
        >>> f = generate_function(expr, [x, y], backend='numpy')
        >>> f(3.0, 4.0)  # Returns [25.0]
        >>> f(3.0, 4.0).item()  # Returns 25.0
        >>>
        >>> # Vector/Matrix expression
        >>> expr = sp.Matrix([x**2, y**2, x*y])
        >>> f = generate_function(expr, [x, y], backend='torch')
        >>> f(torch.tensor(2.0), torch.tensor(3.0))  # Returns tensor([4., 9., 6.])
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
    **kwargs,
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

        if kwargs.get("jit", True):
            jac_func = jax.jit(jac_func)

        return jac_func
