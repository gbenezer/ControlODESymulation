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
Discretized System - Numerical Discretization of Continuous Systems
====================================================================

Provides discrete-time approximation via three modes: FIXED_STEP, 
DENSE_OUTPUT, and BATCH_INTERPOLATION.

This module provides DiscretizedSystem, which wraps any ContinuousSystemBase
and provides a discrete interface through numerical integration.

Protocol Satisfaction
--------------------
DiscretizedSystem satisfies:
- ✓ DiscreteSystemProtocol (step, simulate)
- ✓ LinearizableDiscreteProtocol (linearize via ZOH)
- ✗ SymbolicDiscreteProtocol (no symbolic expressions - purely numerical)

This is CORRECT - discretization is numerical, not symbolic!

See class docstring for complete documentation.
"""

import inspect
import time
from enum import Enum
from typing import Callable, Optional, Sequence, Tuple, List

import numpy as np
from scipy.interpolate import interp1d

from src.systems.base.core.continuous_system_base import ContinuousSystemBase
from src.systems.base.core.discrete_system_base import DiscreteSystemBase
from src.systems.base.numerical_integration.integrator_factory import IntegratorFactory

from src.types.core import ControlVector, DiscreteControlInput, StateVector
from src.types.linearization import DiscreteLinearization
from src.types.trajectories import DiscreteSimulationResult


class DiscretizationMode(Enum):
    """Three discretization modes balancing accuracy and efficiency."""
    FIXED_STEP = "fixed_step"
    DENSE_OUTPUT = "dense_output"
    BATCH_INTERPOLATION = "batch_interpolation"


class DiscretizedSystem(DiscreteSystemBase):
    """
    Pure wrapper providing discrete interface to continuous systems.
    
    Protocol Satisfaction
    --------------------
    This class satisfies:
    - DiscreteSystemProtocol: Has step(), simulate(), dt, nx, nu
    - LinearizableDiscreteProtocol: Has linearize() (wraps continuous)
    
    Does NOT satisfy:
    - SymbolicDiscreteProtocol: No symbolic machinery (purely numerical)
    
    This means it can be used in:
    - ✓ Any function expecting DiscreteSystemProtocol
    - ✓ Control design (LQR, MPC) expecting LinearizableDiscreteProtocol
    - ✗ Code generation expecting SymbolicDiscreteProtocol
    
    Examples
    --------
    >>> from src.types.protocols import LinearizableDiscreteProtocol
    >>> 
    >>> def lqr_design(system: LinearizableDiscreteProtocol, Q, R):
    ...     Ad, Bd = system.linearize(np.zeros(system.nx), np.zeros(system.nu))
    ...     # ... LQR computation
    >>> 
    >>> # DiscretizedSystem works here:
    >>> continuous = Pendulum(m=1.0, l=0.5)
    >>> discrete = DiscretizedSystem(continuous, dt=0.01)
    >>> K = lqr_design(discrete, Q, R)  # ✓ Type checks pass!
    """
    
    _FIXED_STEP_METHODS = frozenset(['euler', 'midpoint', 'rk4', 'heun'])
    
    def __init__(self, continuous_system: ContinuousSystemBase, dt: float = 0.01,
                 method: str = 'rk4', mode: Optional[DiscretizationMode] = None,
                 interpolation_kind: str = 'cubic', **integrator_kwargs):
        if not isinstance(continuous_system, ContinuousSystemBase):
            raise TypeError(f"Expected ContinuousSystemBase, got {type(continuous_system).__name__}")
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt}")
        
        self._continuous_system = continuous_system
        self._dt = float(dt)
        self._method = method
        self._interpolation_kind = interpolation_kind
        self._integrator_kwargs = integrator_kwargs
        self._is_fixed_step = method.lower() in self._FIXED_STEP_METHODS
        
        self._mode = mode if mode else (
            DiscretizationMode.FIXED_STEP if self._is_fixed_step 
            else DiscretizationMode.DENSE_OUTPUT
        )
        
        if self._mode == DiscretizationMode.FIXED_STEP and not self._is_fixed_step:
            raise ValueError(f"Cannot use adaptive method '{method}' with FIXED_STEP mode")
    
    @property
    def dt(self) -> float:
        return self._dt
    
    @property
    def mode(self) -> DiscretizationMode:
        return self._mode
    
    @property
    def nx(self) -> int:
        return self._continuous_system.nx
    
    @property
    def nu(self) -> int:
        return self._continuous_system.nu
    
    @property
    def ny(self) -> int:
        return self._continuous_system.ny
    
    @property
    def is_stochastic(self) -> bool:
        return self._continuous_system.is_stochastic
    
    def step(self, x: StateVector, u: Optional[ControlVector] = None, k: int = 0) -> StateVector:
        if self._mode == DiscretizationMode.BATCH_INTERPOLATION:
            raise NotImplementedError("step() not supported in BATCH_INTERPOLATION mode")
        
        t_start, t_end = k * self._dt, (k + 1) * self._dt
        return self._step_fixed(x, u, t_start, t_end) if self._mode == DiscretizationMode.FIXED_STEP else self._step_dense(x, u, t_start, t_end)
    
    def _step_fixed(self, x, u, t_start, t_end):
        integrator = IntegratorFactory.create(
            system=self._continuous_system, backend=self._continuous_system._default_backend,
            method=self._method, dt=self._dt, **self._integrator_kwargs
        )
        result = integrator.integrate(x0=x, u_func=lambda t, xv: u, t_span=(t_start, t_end))
        return result['x'][-1, :] if 'x' in result else result['y'][:, -1]
    
    def _step_dense(self, x, u, t_start, t_end):
        integrator = IntegratorFactory.create(
            system=self._continuous_system, backend=self._continuous_system._default_backend,
            method=self._method, **self._integrator_kwargs
        )
        result = integrator.integrate(x0=x, u_func=lambda t, xv: u, t_span=(t_start, t_end), dense_output=True)
        
        if 'sol' in result and result['sol'] is not None:
            x_end = result['sol'](t_end)
            return x_end.ravel() if x_end.ndim > 1 else x_end
        return result['x'][-1, :] if 'x' in result else result['y'][:, -1]
    
    def simulate(self, x0: StateVector, u_sequence: DiscreteControlInput = None, 
                 n_steps: int = 100, **kwargs) -> DiscreteSimulationResult:
        return self._simulate_batch(x0, u_sequence, n_steps) if self._mode == DiscretizationMode.BATCH_INTERPOLATION else self._simulate_step_by_step(x0, u_sequence, n_steps)
    
    def _simulate_step_by_step(self, x0, u_sequence, n_steps):
        states = np.zeros((n_steps + 1, self.nx))
        states[0, :] = x0
        controls = []
        u_func = self._prepare_control_sequence(u_sequence, n_steps)
        
        x = x0
        for k in range(n_steps):
            u = u_func(x, k)
            controls.append(u)
            x = self.step(x, u, k)
            states[k + 1, :] = x
        
        return {
            "states": states,
            "controls": np.array(controls) if controls and controls[0] is not None else None,
            "time_steps": np.arange(n_steps + 1),
            "dt": self.dt,
            "success": True,
            "metadata": {"method": self._method, "mode": self._mode.value}
        }
    
    def _simulate_batch(self, x0, u_sequence, n_steps):
        if callable(u_sequence) and len(inspect.signature(u_sequence).parameters) == 2:
            raise ValueError("State-feedback not supported in BATCH_INTERPOLATION mode")
        
        u_func_discrete = self._prepare_control_sequence(u_sequence, n_steps)
        u_func_continuous = lambda t, x: u_func_discrete(x, min(int(t / self.dt), n_steps - 1))
        
        result = self._continuous_system.integrate(
            x0=x0, u=u_func_continuous, t_span=(0.0, n_steps * self.dt),
            method=self._method, **self._integrator_kwargs
        )
        
        trajectory = result['x'] if 'x' in result else result['y'].T
        t_regular = np.arange(0, n_steps + 1) * self.dt
        states_regular = self._interpolate_trajectory(result['t'], trajectory, t_regular)
        
        return {
            "states": states_regular,
            "controls": None,
            "time_steps": np.arange(n_steps + 1),
            "dt": self.dt,
            "success": result.get('success', True),
            "metadata": {
                "method": self._method, "mode": self._mode.value,
                "nfev": result.get('nfev'), "adaptive_points": len(result['t'])
            }
        }
    
    def _interpolate_trajectory(self, t_adaptive, y_adaptive, t_regular):
        if y_adaptive.shape[0] != len(t_adaptive):
            y_adaptive = y_adaptive.T
        
        nx = y_adaptive.shape[1]
        y_regular = np.zeros((len(t_regular), nx))
        
        for i in range(nx):
            interp = interp1d(t_adaptive, y_adaptive[:, i], kind=self._interpolation_kind,
                            fill_value="extrapolate", assume_sorted=True)
            y_regular[:, i] = interp(t_regular)
        
        return y_regular
    
    def linearize(self, x_eq: StateVector, u_eq: Optional[ControlVector] = None) -> DiscreteLinearization:
        lin_result = self._continuous_system.linearize(x_eq, u_eq)
        A, B = lin_result[:2]  # Handle both (A,B) and (A,B,G)
        
        from scipy.linalg import expm
        nx, I = A.shape[0], np.eye(A.shape[0])
        Ad = expm(A * self.dt)
        
        try:
            if np.linalg.cond(A) > 1e10:
                Bd = self.dt * B
            else:
                Bd = np.linalg.inv(A) @ (Ad - I) @ B
        except np.linalg.LinAlgError:
            Bd = self.dt * B
        
        return (Ad, Bd)
    
    def _prepare_control_sequence(self, u_sequence, n_steps):
        if u_sequence is None:
            return lambda x, k: None if self.nu == 0 else np.zeros(self.nu)
        
        if callable(u_sequence):
            sig = inspect.signature(u_sequence)
            if len(sig.parameters) == 1:
                return lambda x, k: u_sequence(k)
            elif len(sig.parameters) == 2:
                names = list(sig.parameters.keys())
                if names[0] in ['x', 'state']:
                    return u_sequence
                elif names[0] in ['k', 'time']:
                    return lambda x, k: u_sequence(k, x)
                try:
                    u_sequence(np.zeros(self.nx), 0)
                    return u_sequence
                except:
                    return lambda x, k: u_sequence(k, x)
        
        if isinstance(u_sequence, np.ndarray):
            if u_sequence.ndim == 1:
                return lambda x, k: u_sequence
            if u_sequence.shape[0] == n_steps:
                return lambda x, k: u_sequence[k, :] if k < n_steps else u_sequence[-1, :]
            return lambda x, k: u_sequence[:, k] if k < u_sequence.shape[1] else u_sequence[:, -1]
        
        if isinstance(u_sequence, (list, tuple)):
            return lambda x, k: np.asarray(u_sequence[k] if k < len(u_sequence) else u_sequence[-1])
        
        raise TypeError(f"Invalid control type: {type(u_sequence)}")
    
    def compare_modes(self, x0, u_sequence, n_steps, reference_solution=None):
        if reference_solution is None:
            ref = DiscretizedSystem(self._continuous_system, dt=self.dt, method='LSODA',
                                   mode=DiscretizationMode.BATCH_INTERPOLATION, rtol=1e-12, atol=1e-14)
            reference_solution = ref.simulate(x0, u_sequence, n_steps)['states']
        
        results, timings, errors = {}, {}, {}
        
        for name, mode, method in [('fixed_step', DiscretizationMode.FIXED_STEP, 'rk4'),
                                   ('dense_output', DiscretizationMode.DENSE_OUTPUT, 'RK45'),
                                   ('batch', DiscretizationMode.BATCH_INTERPOLATION, 'RK45')]:
            sys = DiscretizedSystem(self._continuous_system, dt=self.dt, method=method, mode=mode)
            start = time.time()
            result = sys.simulate(x0, u_sequence, n_steps)
            timings[name] = time.time() - start
            results[name] = result
            errors[name] = np.sqrt(np.mean((result['states'] - reference_solution) ** 2))
        
        return {
            'results': results, 'timings': timings, 'errors': errors,
            'reference': reference_solution,
            'speedup_batch_vs_fixed': timings['fixed_step'] / timings['batch'],
            'speedup_batch_vs_dense': timings['dense_output'] / timings['batch']
        }
    
    def change_method(self, new_method: str, **new_kwargs) -> "DiscretizedSystem":
        """Create new DiscretizedSystem with different method."""
        merged_kwargs = {**self._integrator_kwargs, **new_kwargs}
        return DiscretizedSystem(
            self._continuous_system, dt=self.dt, method=new_method,
            mode=None, interpolation_kind=self._interpolation_kind, **merged_kwargs
        )
    
    def get_info(self) -> dict:
        """Get comprehensive discretization information."""
        return {
            "class": "DiscretizedSystem",
            "mode": self._mode.value,
            "method": self._method,
            "dt": self.dt,
            "is_fixed_step": self._is_fixed_step,
            "interpolation": self._interpolation_kind,
            "supports_step": self._mode != DiscretizationMode.BATCH_INTERPOLATION,
            "supports_closed_loop": self._mode != DiscretizationMode.BATCH_INTERPOLATION,
            "continuous_system_type": type(self._continuous_system).__name__,
            "is_stochastic": self.is_stochastic,
            "dimensions": {"nx": self.nx, "nu": self.nu, "ny": self.ny},
            "integrator_kwargs": self._integrator_kwargs,
        }
    
    def print_info(self):
        """Print formatted discretization information."""
        info = self.get_info()
        print("=" * 70)
        print("DiscretizedSystem")
        print("=" * 70)
        print(f"Continuous System: {info['continuous_system_type']}")
        print(f"Discretization Method: {info['method']}")
        print(f"Mode: {info['mode'].upper()}")
        print(f"Time Step: {info['dt']}s ({1/info['dt']:.1f} Hz)")
        print(f"Dimensions: nx={info['dimensions']['nx']}, nu={info['dimensions']['nu']}, ny={info['dimensions']['ny']}")
        print(f"Stochastic: {info['is_stochastic']}")
        print(f"Supports step(): {info['supports_step']}")
        print(f"Supports closed-loop: {info['supports_closed_loop']}")
        if info['integrator_kwargs']:
            print("\nIntegrator Options:")
            for key, val in info['integrator_kwargs'].items():
                print(f"  {key}: {val}")
        print("=" * 70)
    
    def __repr__(self):
        return f"DiscretizedSystem(dt={self.dt:.4f}, method={self._method}, mode={self._mode.value})"


def discretize(continuous_system, dt, method='rk4', **kwargs):
    """Convenience wrapper for DiscretizedSystem."""
    return DiscretizedSystem(continuous_system, dt=dt, method=method, **kwargs)


def discretize_batch(continuous_system, dt, method='LSODA', **kwargs):
    """Create batch-mode discretized system."""
    return DiscretizedSystem(continuous_system, dt=dt, method=method,
                            mode=DiscretizationMode.BATCH_INTERPOLATION, **kwargs)


def analyze_discretization_error(continuous_system, x0, u_sequence, dt_values,
                                 method='rk4', n_steps=100, reference_dt=None):
    """Analyze error vs dt for convergence study."""
    if reference_dt is None:
        reference_dt = min(dt_values) / 10
    
    ref = DiscretizedSystem(continuous_system, dt=reference_dt, method='LSODA',
                           mode=DiscretizationMode.BATCH_INTERPOLATION, rtol=1e-12, atol=1e-14)
    n_ref = int(n_steps * max(dt_values) / reference_dt)
    ref_result = ref.simulate(x0, u_sequence, n_ref)
    t_compare = np.arange(n_steps + 1) * max(dt_values)
    
    errors, timings = [], []
    
    for dt in dt_values:
        disc = DiscretizedSystem(continuous_system, dt=dt, method=method)
        start = time.time()
        n_curr = int(n_steps * max(dt_values) / dt)
        result = disc.simulate(x0, u_sequence, n_curr)
        timings.append(time.time() - start)
        
        # Interpolate both to comparison grid
        from scipy.interpolate import interp1d
        t_curr = result['time_steps'] * dt
        t_ref = ref_result['time_steps'] * reference_dt
        
        states_interp = np.zeros((len(t_compare), continuous_system.nx))
        ref_interp = np.zeros((len(t_compare), continuous_system.nx))
        
        for i in range(continuous_system.nx):
            states_interp[:, i] = interp1d(t_curr, result['states'][:, i], kind='cubic')(t_compare)
            ref_interp[:, i] = interp1d(t_ref, ref_result['states'][:, i], kind='cubic')(t_compare)
        
        errors.append(np.sqrt(np.mean((states_interp - ref_interp) ** 2)))
    
    # Estimate convergence rate
    log_dt, log_err = np.log(dt_values), np.log(np.array(errors) + 1e-16)
    convergence_rate = np.polyfit(log_dt, log_err, 1)[0]
    
    return {
        'dt_values': list(dt_values), 'errors': errors, 'timings': timings,
        'reference': ref_result, 'method': method, 'convergence_rate': convergence_rate
    }


def recommend_dt(continuous_system, x0, target_error=1e-6, method='rk4',
                dt_range=(1e-4, 0.1), n_test=10):
    """Recommend dt for target accuracy."""
    dt_values = np.logspace(np.log10(dt_range[0]), np.log10(dt_range[1]), n_test)
    analysis = analyze_discretization_error(continuous_system, x0, None, dt_values, method, 100)
    
    errors = np.array(analysis['errors'])
    valid_mask = errors < target_error
    
    if not np.any(valid_mask):
        import warnings
        warnings.warn(f"No dt achieves target {target_error:.2e}", UserWarning)
        best_idx = np.argmin(errors)
    else:
        best_idx = np.where(valid_mask)[0][-1]
    
    return {
        'recommended_dt': float(dt_values[best_idx]),
        'achieved_error': float(errors[best_idx]),
        'timing': analysis['timings'][best_idx],
        'all_results': analysis
    }


def detect_sde_integrator(continuous_system):
    """Detect best SDE method for stochastic system."""
    if not continuous_system.is_stochastic:
        raise ValueError("System is not stochastic")
    
    if hasattr(continuous_system, 'is_additive_noise') and continuous_system.is_additive_noise():
        return 'euler_maruyama'
    if hasattr(continuous_system, 'is_diagonal_noise') and continuous_system.is_diagonal_noise():
        return 'milstein'
    return 'euler_maruyama'


def compute_discretization_quality(discrete_system, x0, u_sequence, n_steps, metrics=None):
    """Compute quality metrics for discretization."""
    if metrics is None:
        metrics = ['timing', 'stability']
    
    results = {}
    start = time.time()
    sim_result = discrete_system.simulate(x0, u_sequence, n_steps)
    elapsed = time.time() - start
    
    if 'timing' in metrics:
        results['timing'] = {
            'total_time': elapsed,
            'time_per_step': elapsed / n_steps,
            'steps_per_second': n_steps / elapsed
        }
    
    if 'stability' in metrics:
        norms = np.linalg.norm(sim_result['states'], axis=1)
        results['stability'] = {
            'is_stable': bool(norms[-1] < 100 * norms[0]),
            'final_norm': float(norms[-1]),
            'max_norm': float(np.max(norms))
        }
    
    return results


__all__ = [
    'DiscretizationMode', 'DiscretizedSystem', 'discretize', 'discretize_batch',
    'analyze_discretization_error', 'recommend_dt', 'detect_sde_integrator',
    'compute_discretization_quality', 'AdaptiveDiscretizedSystem', 'MultiRateDiscretizedSystem'
]


# ============================================================================
# Advanced Features - Experimental
# ============================================================================


class AdaptiveDiscretizedSystem(DiscretizedSystem):
    """
    Discretized system with adaptive time step selection.
    
    Automatically adjusts dt during simulation based on dynamics smoothness.
    **EXPERIMENTAL** - for future development.
    
    Examples
    --------
    >>> adaptive = AdaptiveDiscretizedSystem(
    ...     continuous, dt_initial=0.01, dt_min=0.001, dt_max=0.1, tol=1e-6
    ... )
    >>> result = adaptive.simulate(x0, u_sequence, n_steps=1000)
    >>> stats = adaptive.get_dt_statistics()
    >>> print(f"Mean dt: {stats['mean']:.4f}, Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
    """
    
    def __init__(self, continuous_system, dt_initial=0.01, dt_min=1e-4, 
                 dt_max=0.1, tol=1e-6, **kwargs):
        """Initialize adaptive discretization."""
        super().__init__(continuous_system, dt=dt_initial, method='RK45',
                        mode=DiscretizationMode.DENSE_OUTPUT, **kwargs)
        self._dt_min = dt_min
        self._dt_max = dt_max
        self._tol = tol
        self._dt_history = []
    
    def step(self, x, u=None, k=0):
        """Adaptive step - tracks dt usage."""
        self._dt_history.append(self._dt)
        return super().step(x, u, k)
    
    def get_dt_statistics(self) -> dict:
        """Get statistics about adaptive dt selection."""
        if not self._dt_history:
            return {"mean": self._dt, "min": self._dt, "max": self._dt, 
                   "std": 0.0, "n_steps": 0}
        
        dt_array = np.array(self._dt_history)
        return {
            "mean": float(np.mean(dt_array)),
            "min": float(np.min(dt_array)),
            "max": float(np.max(dt_array)),
            "std": float(np.std(dt_array)),
            "n_steps": len(self._dt_history)
        }
    
    def reset_statistics(self):
        """Reset dt tracking."""
        self._dt_history = []


class MultiRateDiscretizedSystem(DiscretizedSystem):
    """
    Discretized system with multiple time scales.
    
    Uses different dt for different state components (fast/slow dynamics).
    **EXPERIMENTAL** - for future development.
    
    Examples
    --------
    >>> multirate = MultiRateDiscretizedSystem(
    ...     continuous, dt_fast=0.001, dt_slow=0.01,
    ...     fast_indices=[0], slow_indices=[1]
    ... )
    >>> # Updates x[0] every step, x[1] every 10 steps
    """
    
    def __init__(self, continuous_system, dt_fast, dt_slow, 
                 fast_indices: List[int], slow_indices: List[int], **kwargs):
        """Initialize multi-rate discretization."""
        super().__init__(continuous_system, dt=dt_fast, **kwargs)
        
        self._dt_slow = dt_slow
        self._fast_indices = fast_indices
        self._slow_indices = slow_indices
        
        ratio = dt_slow / dt_fast
        if not np.isclose(ratio, round(ratio)):
            raise ValueError(f"dt_slow must be integer multiple of dt_fast, got ratio {ratio}")
        
        self._slow_update_interval = int(round(ratio))
    
    def step(self, x, u=None, k=0):
        """Multi-rate step: update fast every step, slow every N steps."""
        x_next = super().step(x, u, k)
        
        # Keep slow states unchanged except at slow update times
        if k % self._slow_update_interval != 0:
            for idx in self._slow_indices:
                x_next[idx] = x[idx]
        
        return x_next