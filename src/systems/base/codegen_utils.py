"""
Backend-agnostic code generation utilities.

CURRENT IMPLEMENTATION NOTE:
PyTorch backend has additional complexity compared to NumPy/JAX due to:
- Gradient tracking requirements (requires_grad preservation)
- Complex batching and shape inference
- Device management across CPU/CUDA

This is technical debt that should be addressed by adding
equivalent complexity to NumPy/JAX backends

For now, PyTorch privilege is accepted for backward compatibility.
"""

from typing import Callable, Literal, Union
import sympy as sp
import numpy as np
import torch

Backend = Literal["numpy", "torch", "jax"]


# Helper functions for PyTorch
def _torch_min(*args):
    """Handle Min for both scalars and tensors"""
    import torch

    if len(args) == 2:
        a, b = args
        a_tensor = torch.as_tensor(a) if not isinstance(a, torch.Tensor) else a
        b_tensor = torch.as_tensor(b) if not isinstance(b, torch.Tensor) else b
        return torch.minimum(a_tensor, b_tensor)
    else:
        tensors = [torch.as_tensor(a) if not isinstance(a, torch.Tensor) else a for a in args]
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
        tensors = [torch.as_tensor(a) if not isinstance(a, torch.Tensor) else a for a in args]
        return torch.max(torch.stack(tensors))


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
SYMPY_TO_JAX = {
    "ImmutableDenseMatrix": "_jax_matrix_handler",
    "MutableDenseMatrix": "_jax_matrix_handler",
    "Matrix": "_jax_matrix_handler",
}

SYMPY_TO_JAX_LAMBDIFY = {
    "ImmutableDenseMatrix": _jax_matrix_handler,
    "MutableDenseMatrix": _jax_matrix_handler,
    "Matrix": _jax_matrix_handler,
}


def generate_numpy_function(
    expr: Union[sp.Expr, list[sp.Expr], sp.Matrix],
    symbols: list[sp.Symbol],
) -> Callable:
    """Generate a NumPy function from SymPy expression(s)."""
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


def generate_torch_function(
    expr: Union[sp.Expr, list[sp.Expr], sp.Matrix], symbols: list[sp.Symbol]
) -> Callable:
    """Generate a PyTorch function from SymPy expression(s)."""

    # Convert to matrix for consistent handling
    if isinstance(expr, list):
        expr = sp.Matrix(expr)
    elif not isinstance(expr, sp.Matrix):
        expr = sp.Matrix([expr])

    # Remember the original shape
    original_shape = expr.shape  # e.g., (2, 1) for Jacobian

    func = sp.lambdify(symbols, expr, modules=[SYMPY_TO_TORCH_LAMBDIFY])

    def wrapped_func(*args):
        result = func(*args)

        # Handle SymPy Matrix
        if hasattr(result, "__class__"):
            class_name = str(result.__class__)
            if "Matrix" in class_name or "ImmutableDenseMatrix" in class_name:
                try:
                    n_rows = len(result)
                    result = [result[i] for i in range(n_rows)]
                except:
                    result = list(result) if hasattr(result, "__iter__") else [result]

        if isinstance(result, torch.Tensor):
            if result.ndim == 0:
                result = result.reshape(1)
            return result

        elif isinstance(result, (list, tuple)):
            if len(result) == 0:
                raise ValueError("Empty result from lambdify")

            tensors = []
            for item in result:
                if isinstance(item, torch.Tensor):
                    # Preserve existing tensor (keeps gradients!)
                    t = item.reshape(1) if item.ndim == 0 else item
                    tensors.append(t)
                else:
                    # Convert to tensor - use as_tensor to potentially preserve gradients
                    t = torch.as_tensor(item, dtype=torch.float32)
                    t = t.reshape(1) if t.ndim == 0 else t
                    tensors.append(t)

            if len(tensors) == 1:
                return tensors[0]
            else:
                result = torch.stack(tensors, dim=0)

                # Preserve original matrix shape if it was 2D column vector
                if original_shape[1] == 1 and result.ndim == 1:
                    result = result.unsqueeze(1)  # (2,) -> (2, 1)

                # Transpose for batching
                elif result.ndim == 2 and len(args) > 0:
                    if isinstance(args[0], torch.Tensor) and args[0].ndim > 0:
                        batch_size = args[0].shape[0]
                        if (
                            result.shape[1] == batch_size
                            and result.shape[0] != batch_size
                            and result.shape[0] < 10
                        ):
                            result = result.T

                return result
        else:
            return torch.tensor([result], dtype=torch.float32)

    return wrapped_func


def generate_jax_function(
    expr: Union[sp.Expr, list[sp.Expr], sp.Matrix],
    symbols: list[sp.Symbol],
    jit: bool = True,
) -> Callable:
    """Generate a JAX function from SymPy expression(s)."""
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
    """Generate a function from SymPy expression for specified backend."""
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
