"""
Backend-agnostic diagnostic and helper utilities.

Provides uniform diagnostics and device inspection for ALL backends. No backend is privileged.
"""

import numpy as np
from typing import Tuple, Any, Optional, Dict, List


# ============================================================================
# Device Information Helpers (All Backends)
# ============================================================================


def get_device_info(arr: Any, backend_name: Optional[str] = None) -> Tuple[Any, str]:
    """
    Get device information from array (works for any backend).

    Args:
        arr: Array from any backend (torch.Tensor, np.ndarray, jax.Array)
        backend_name: Optional backend name hint ('pytorch', 'numpy', 'jax')

    Returns:
        (device_object, device_string)

    Examples:
        >>> # PyTorch
        >>> arr = torch.ones(3, device='cuda')
        >>> device, device_str = get_device_info(arr)
        >>> print(device_str)  # "cuda:0"
        >>>
        >>> # JAX
        >>> arr = jnp.ones(3)
        >>> device, device_str = get_device_info(arr)
        >>> print(device_str)  # "gpu:0" or "cpu:0"
        >>>
        >>> # NumPy (always CPU)
        >>> arr = np.ones(3)
        >>> device, device_str = get_device_info(arr)
        >>> print(device_str)  # "cpu"
    """
    # Auto-detect backend if not provided
    if backend_name is None:
        backend_name = detect_backend(arr)

    if backend_name == "pytorch":
        return _get_torch_device_info(arr)
    elif backend_name == "jax":
        return _get_jax_device_info(arr)
    elif backend_name == "numpy":
        return _get_numpy_device_info(arr)
    else:
        return None, "unknown"


def _get_torch_device_info(arr) -> Tuple[Any, str]:
    """Get PyTorch device info"""
    import torch

    if not isinstance(arr, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(arr)}")

    device = arr.device
    device_str = str(device)

    return device, device_str


def _get_jax_device_info(arr) -> Tuple[Any, str]:
    """Get JAX device info (handles version differences)"""
    import jax

    # Try different JAX versions
    if hasattr(arr, "devices"):
        # Newest JAX (>= 0.4.1) - returns a set of devices
        devices = arr.devices()
        device = list(devices)[0]
        device_str = str(device)
    elif hasattr(arr, "device"):
        # JAX 0.3.x - 0.4.0 - device() method returns Device object
        device = arr.device()
        device_str = str(device)
    elif hasattr(arr, "device_buffer"):
        # Very old JAX
        device = arr.device_buffer.device()
        device_str = str(device)
    else:
        device = None
        device_str = "unknown"

    return device, device_str


def _get_numpy_device_info(arr) -> Tuple[Any, str]:
    """Get NumPy device info (always CPU)"""
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"Expected np.ndarray, got {type(arr)}")

    # NumPy is always CPU
    return None, "cpu"


def detect_backend(arr: Any) -> str:
    """
    Auto-detect backend from array type.

    Args:
        arr: Array-like object

    Returns:
        Backend name: 'pytorch', 'numpy', 'jax', or 'unknown'
    """
    # Check PyTorch
    try:
        import torch

        if isinstance(arr, torch.Tensor):
            return "pytorch"
    except ImportError:
        pass

    # Check JAX
    try:
        import jax.numpy as jnp

        # JAX arrays can be various types depending on version
        if isinstance(arr, jnp.ndarray) or type(arr).__module__.startswith("jax"):
            return "jax"
    except ImportError:
        pass

    # Check NumPy
    if isinstance(arr, np.ndarray):
        return "numpy"

    return "unknown"


# ============================================================================
# Installation Diagnostics (All Backends)
# ============================================================================


def diagnose_installation(backends: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Diagnose installation and capabilities for all backends.

    Args:
        backends: List of backends to check, or None for all
                 ['pytorch', 'numpy', 'jax']

    Returns:
        Dict with installation status and device info for each backend

    Example:
        >>> info = diagnose_installation()
        >>> print(info['pytorch']['installed'])  # True/False
        >>> print(info['pytorch']['gpu_available'])  # True/False
        >>> print(info['jax']['devices'])  # List of JAX devices
    """
    if backends is None:
        backends = ["numpy", "pytorch", "jax"]

    results = {}

    for backend in backends:
        if backend == "numpy":
            results["numpy"] = _diagnose_numpy()
        elif backend == "pytorch":
            results["pytorch"] = _diagnose_pytorch()
        elif backend == "jax":
            results["jax"] = _diagnose_jax()

    return results


def _diagnose_numpy() -> Dict[str, Any]:
    """Diagnose NumPy installation"""
    try:
        import numpy as np

        return {
            "installed": True,
            "version": np.__version__,
            "device": "cpu",
            "gpu_available": False,
            "devices": ["cpu"],
            "blas_info": np.__config__.show() if hasattr(np.__config__, "show") else "unknown",
        }
    except ImportError:
        return {
            "installed": False,
            "error": "NumPy not installed",
        }


def _diagnose_pytorch() -> Dict[str, Any]:
    """Diagnose PyTorch installation"""
    try:
        import torch

        info = {
            "installed": True,
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "devices": ["cpu"],
        }

        # CUDA info
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda
            info["gpu_count"] = torch.cuda.device_count()
            info["devices"].extend([f"cuda:{i}" for i in range(torch.cuda.device_count())])
            info["current_device"] = torch.cuda.current_device()

            # GPU names
            info["gpu_names"] = [
                torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
            ]

        # MPS (Apple Silicon)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            info["mps_available"] = True
            info["devices"].append("mps")

        return info

    except ImportError:
        return {
            "installed": False,
            "error": "PyTorch not installed",
        }


def _diagnose_jax() -> Dict[str, Any]:
    """Diagnose JAX installation"""
    try:
        import jax

        info = {
            "installed": True,
            "version": jax.__version__,
            "devices": [],
        }

        # CPU devices
        try:
            cpu_devices = jax.devices("cpu")
            info["cpu_count"] = len(cpu_devices)
            info["devices"].extend([str(dev) for dev in cpu_devices])
        except Exception as e:
            info["cpu_error"] = str(e)

        # GPU devices
        try:
            gpu_devices = jax.devices("gpu")
            info["gpu_available"] = len(gpu_devices) > 0
            info["gpu_count"] = len(gpu_devices)
            info["devices"].extend([str(dev) for dev in gpu_devices])
        except Exception as e:
            info["gpu_available"] = False
            info["gpu_error"] = str(e)

        # TPU devices
        try:
            tpu_devices = jax.devices("tpu")
            if len(tpu_devices) > 0:
                info["tpu_available"] = True
                info["tpu_count"] = len(tpu_devices)
                info["devices"].extend([str(dev) for dev in tpu_devices])
        except Exception:
            info["tpu_available"] = False

        return info

    except ImportError:
        return {
            "installed": False,
            "error": "JAX not installed",
        }


def print_installation_summary():
    """
    Print a human-readable summary of all backend installations.

    Usage:
        >>> from src.systems.base.backend_utils import print_installation_summary
        >>> print_installation_summary()
    """
    print("=" * 70)
    print("Backend Installation Summary")
    print("=" * 70)

    info = diagnose_installation()

    # NumPy
    print("\n📊 NumPy")
    if info["numpy"]["installed"]:
        print(f"  ✓ Installed: {info['numpy']['version']}")
        print(f"  Device: {info['numpy']['device']}")
    else:
        print(f"  ✗ Not installed")

    # PyTorch
    print("\n🔥 PyTorch")
    if info["pytorch"]["installed"]:
        print(f"  ✓ Installed: {info['pytorch']['version']}")
        print(f"  CUDA available: {info['pytorch']['cuda_available']}")
        if info["pytorch"]["cuda_available"]:
            print(f"  CUDA version: {info['pytorch']['cuda_version']}")
            print(f"  GPU count: {info['pytorch']['gpu_count']}")
            for i, name in enumerate(info["pytorch"]["gpu_names"]):
                print(f"    [{i}] {name}")
        print(f"  Available devices: {', '.join(info['pytorch']['devices'])}")
    else:
        print(f"  ✗ Not installed")

    # JAX
    print("\n⚡ JAX")
    if info["jax"]["installed"]:
        print(f"  ✓ Installed: {info['jax']['version']}")
        print(f"  GPU available: {info['jax'].get('gpu_available', False)}")
        if info["jax"].get("gpu_available"):
            print(f"  GPU count: {info['jax']['gpu_count']}")
        if info["jax"].get("tpu_available"):
            print(f"  TPU available: True")
            print(f"  TPU count: {info['jax']['tpu_count']}")
        print(f"  Available devices: {len(info['jax']['devices'])}")
        for dev in info["jax"]["devices"]:
            print(f"    - {dev}")
    else:
        print(f"  ✗ Not installed")
        print(f"  Install with: pip install jax[cuda12]  # For GPU")

    print("=" * 70)


# ============================================================================
# Backend Comparison Utilities
# ============================================================================


def compare_backend_performance(
    operation: str = "matmul",
    size: int = 1000,
    backends: Optional[List[str]] = None,
    device_types: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Compare performance across all available backends.

    Args:
        operation: Operation to benchmark ('matmul', 'sum', 'fft')
        size: Array size for benchmark
        backends: List of backends to test (None = all available)
        device_types: List of device types ('cpu', 'gpu')

    Returns:
        Dict mapping backend_device to execution time

    Example:
        >>> times = compare_backend_performance('matmul', size=2000)
        >>> for backend_device, time in sorted(times.items(), key=lambda x: x[1]):
        ...     print(f"{backend_device:20s}: {time:.4f}s")
        pytorch_gpu:0        : 0.0021s
        jax_gpu:0            : 0.0024s
        pytorch_cpu          : 0.0156s
        numpy_cpu            : 0.0189s
    """
    import time

    if backends is None:
        backends = ["numpy", "pytorch", "jax"]

    if device_types is None:
        device_types = ["cpu", "gpu"]

    results = {}

    for backend in backends:
        for device_type in device_types:
            # Skip invalid combinations
            if backend == "numpy" and device_type == "gpu":
                continue

            try:
                backend_device = f"{backend}_{device_type}"
                time_taken = _benchmark_backend(backend, device_type, operation, size)
                results[backend_device] = time_taken
            except Exception as e:
                results[f"{backend}_{device_type}"] = f"Error: {e}"

    return results


def _benchmark_backend(backend: str, device: str, operation: str, size: int) -> float:
    """Benchmark specific backend/device combination"""
    import time

    if backend == "numpy":
        import numpy as np

        arr = np.random.randn(size, size).astype(np.float32)

        # Warmup
        if operation == "matmul":
            _ = np.dot(arr, arr)

        # Benchmark
        start = time.time()
        for _ in range(10):
            if operation == "matmul":
                result = np.dot(arr, arr)
        return (time.time() - start) / 10

    elif backend == "pytorch":
        import torch

        device_str = device if device == "cpu" else "cuda"
        if device_str == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")

        arr = torch.randn(size, size, device=device_str, dtype=torch.float32)

        # Warmup
        if operation == "matmul":
            _ = torch.mm(arr, arr)
        if device_str == "cuda":
            torch.cuda.synchronize()

        # Benchmark
        start = time.time()
        for _ in range(10):
            if operation == "matmul":
                result = torch.mm(arr, arr)
        if device_str == "cuda":
            torch.cuda.synchronize()

        return (time.time() - start) / 10

    elif backend == "jax":
        import jax
        import jax.numpy as jnp

        # Get device
        if device == "cpu":
            jax_device = jax.devices("cpu")[0]
        else:
            gpu_devices = jax.devices("gpu")
            if len(gpu_devices) == 0:
                raise RuntimeError("JAX GPU not available")
            jax_device = gpu_devices[0]

        arr = jax.device_put(jax.random.normal(jax.random.PRNGKey(0), (size, size)), jax_device)

        # Warmup and compile
        if operation == "matmul":

            @jax.jit
            def matmul_op(x):
                return jnp.dot(x, x)

            _ = matmul_op(arr)
            if hasattr(_, "block_until_ready"):
                _.block_until_ready()

        # Benchmark
        start = time.time()
        for _ in range(10):
            result = matmul_op(arr)
            if hasattr(result, "block_until_ready"):
                result.block_until_ready()

        return (time.time() - start) / 10

    raise ValueError(f"Unknown backend: {backend}")


# ============================================================================
# Comprehensive Backend Diagnostics
# ============================================================================


def diagnose_all_backends():
    """
    Run comprehensive diagnostics for ALL backends.

    No backend is privileged - all get equal treatment.

    Usage:
        >>> from src.systems.base.backend_utils import diagnose_all_backends
        >>> diagnose_all_backends()
    """
    print("=" * 70)
    print("Comprehensive Backend Diagnostics")
    print("=" * 70)

    info = diagnose_installation()

    # NumPy
    _print_numpy_diagnostic(info["numpy"])

    # PyTorch
    _print_pytorch_diagnostic(info["pytorch"])

    # JAX
    _print_jax_diagnostic(info["jax"])

    # Summary comparison
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    installed = []
    if info["numpy"]["installed"]:
        installed.append("NumPy")
    if info["pytorch"]["installed"]:
        installed.append("PyTorch")
    if info["jax"]["installed"]:
        installed.append("JAX")

    print(f"Installed backends: {', '.join(installed) if installed else 'None'}")

    # GPU summary
    gpu_backends = []
    if info["pytorch"].get("cuda_available"):
        gpu_backends.append(f"PyTorch ({info['pytorch']['gpu_count']} GPU)")
    if info["jax"].get("gpu_available"):
        gpu_backends.append(f"JAX ({info['jax']['gpu_count']} GPU)")

    print(f"GPU-capable backends: {', '.join(gpu_backends) if gpu_backends else 'None'}")

    print("=" * 70)


def _print_numpy_diagnostic(info: Dict):
    """Print NumPy diagnostic info"""
    print("\n📊 NumPy Backend")
    print("-" * 70)

    if info["installed"]:
        print(f"✓ Status: Installed")
        print(f"  Version: {info['version']}")
        print(f"  Device: {info['device']} (always CPU)")
        print(f"  GPU support: Not applicable (CPU-only library)")

        # Try to detect BLAS backend
        try:
            import numpy as np

            config = np.__config__
            if hasattr(config, "blas_opt_info"):
                print(f"  BLAS backend: {config.blas_opt_info.get('name', 'unknown')}")
        except:
            pass
    else:
        print(f"✗ Status: Not installed")
        print(f"  Install with: pip install numpy")


def _print_pytorch_diagnostic(info: Dict):
    """Print PyTorch diagnostic info"""
    print("\n🔥 PyTorch Backend")
    print("-" * 70)

    if info["installed"]:
        print(f"✓ Status: Installed")
        print(f"  Version: {info['version']}")
        print(f"  CUDA available: {info['cuda_available']}")

        if info["cuda_available"]:
            print(f"  CUDA version: {info['cuda_version']}")
            print(f"  GPU count: {info['gpu_count']}")
            print(f"  Current device: cuda:{info['current_device']}")
            print(f"  GPU devices:")
            for i, name in enumerate(info["gpu_names"]):
                print(f"    [{i}] {name}")

        # MPS (Apple Silicon)
        if info.get("mps_available"):
            print(f"  MPS (Apple Silicon) available: Yes")

        print(f"  Available devices: {', '.join(info['devices'])}")
    else:
        print(f"✗ Status: Not installed")
        print(f"  Install with:")
        print(f"    CPU: pip install torch")
        print(f"    GPU: pip install torch --index-url https://download.pytorch.org/whl/cu121")


def _print_jax_diagnostic(info: Dict):
    """Print JAX diagnostic info"""
    print("\n⚡ JAX Backend")
    print("-" * 70)

    if info["installed"]:
        print(f"✓ Status: Installed")
        print(f"  Version: {info['version']}")
        print(f"  GPU available: {info.get('gpu_available', False)}")

        if "cpu_count" in info:
            print(f"  CPU devices: {info['cpu_count']}")

        if info.get("gpu_available"):
            print(f"  GPU count: {info['gpu_count']}")

        if info.get("tpu_available"):
            print(f"  TPU available: Yes")
            print(f"  TPU count: {info['tpu_count']}")

        print(f"  All devices ({len(info['devices'])}):")
        for dev in info["devices"]:
            print(f"    - {dev}")
    else:
        print(f"✗ Status: Not installed")
        print(f"  Install with:")
        print(f"    CPU: pip install jax jaxlib")
        print(f"    GPU: pip install jax[cuda12]  # For CUDA 12.x")
        print(f"         pip install jax[cuda11]  # For CUDA 11.x")
        print(f"    TPU: pip install jax[tpu]")


# ============================================================================
# Quick Test Functions (All Backends)
# ============================================================================


def quick_test_backend(backend_name: str, device: str = "cpu") -> bool:
    """
    Quick smoke test for a specific backend.

    Args:
        backend_name: 'numpy', 'pytorch', or 'jax'
        device: 'cpu' or 'gpu'

    Returns:
        True if backend works on specified device, False otherwise

    Example:
        >>> assert quick_test_backend('numpy', 'cpu')
        >>> assert quick_test_backend('pytorch', 'gpu')  # If GPU available
        >>> assert not quick_test_backend('numpy', 'gpu')  # NumPy has no GPU
    """
    try:
        if backend_name == "numpy":

            if device != "cpu":
                return False  # NumPy doesn't support GPU

            import numpy as np

            arr = np.ones((3, 3))
            result = np.dot(arr, arr)
            return True

        elif backend_name == "pytorch":
            import torch

            device_str = device if device == "cpu" else "cuda"

            # Check if requested device is available
            if device_str == "cuda" and not torch.cuda.is_available():
                return False

            arr = torch.ones(3, 3, device=device_str)
            result = torch.mm(arr, arr)
            return True

        elif backend_name == "jax":
            import jax
            import jax.numpy as jnp

            # Explicitly place array on requested device
            if device == "gpu":
                gpu_devices = jax.devices("gpu")
                if len(gpu_devices) == 0:
                    return False
                jax_device = gpu_devices[0]
            else:
                # Explicitly use CPU device
                jax_device = jax.devices("cpu")[0]

            # FIX 3: Use device_put to ensure array is on correct device
            arr = jax.device_put(jnp.ones((3, 3)), jax_device)
            result = jnp.dot(arr, arr)
            return True

        else:
            return False

    except Exception as e:
        # Return False on any error (don't print in production)
        print(f"Error testing {backend_name} on {device}: {e}")
        return False


# ============================================================================
# Usage Example
# ============================================================================

if __name__ == "__main__":
    # Run full diagnostic
    print_installation_summary()

    print("\n" + "=" * 70)
    print("Quick Backend Tests")
    print("=" * 70)

    # Test each backend
    for backend in ["numpy", "pytorch", "jax"]:
        for device in ["cpu", "gpu"]:
            if backend == "numpy" and device == "gpu":
                continue  # Skip NumPy GPU

            result = quick_test_backend(backend, device)
            status = "✓" if result else "✗"
            print(
                f"{status} {backend:10s} on {device:5s}: {'Working' if result else 'Not available'}"
            )

    # Performance comparison
    print("\n" + "=" * 70)
    print("Performance Comparison (1000x1000 matrix multiply)")
    print("=" * 70)

    try:
        times = compare_backend_performance("matmul", size=1000)

        # Sort by speed
        sorted_times = sorted(
            [(k, v) for k, v in times.items() if isinstance(v, float)], key=lambda x: x[1]
        )

        if sorted_times:
            fastest_time = sorted_times[0][1]

            for backend_device, time in sorted_times:
                speedup = fastest_time / time
                print(f"{backend_device:20s}: {time:.4f}s  ({speedup:.2f}x vs fastest)")
    except Exception as e:
        print(f"Performance comparison failed: {e}")
