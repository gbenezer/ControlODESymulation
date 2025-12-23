"""
Custom Brownian path wrapper for Diffrax to support user-provided noise.

This allows deterministic testing and custom noise patterns.
"""

import jax.numpy as jnp
from jax import Array
import diffrax as dfx
from typing import Optional, Tuple
import equinox as eqx


class CustomBrownianPath(dfx.AbstractPath):
    """
    Custom Brownian motion that uses provided dW increments.
    
    This implements Diffrax's AbstractPath interface to allow
    user-specified noise instead of generating random noise.
    
    Uses Equinox Module for proper JAX/Diffrax integration.
    
    Attributes
    ----------
    t0 : float
        Start time
    t1 : float
        End time
    dW : Array
        Brownian increment for interval (t0, t1)
        Shape: (nw,) for the noise dimensions
    
    Examples
    --------
    >>> # Zero noise for deterministic testing
    >>> dW = jnp.zeros(1)
    >>> brownian = CustomBrownianPath(0.0, 0.01, dW)
    >>> 
    >>> # Custom noise pattern
    >>> dW = jnp.array([0.5])
    >>> brownian = CustomBrownianPath(0.0, 0.01, dW)
    """
    
    t0: float
    t1: float
    dW: Array
    
    def __init__(self, t0: float, t1: float, dW: Array):
        # Use object.__setattr__ for frozen dataclass
        object.__setattr__(self, 't0', t0)
        object.__setattr__(self, 't1', t1)
        object.__setattr__(self, 'dW', dW)
    
    @property
    def dt(self) -> float:
        """Time interval length."""
        return self.t1 - self.t0
    
    @property
    def shape(self) -> Tuple[int, ...]:
        """Shape of the noise."""
        return self.dW.shape
    
    def evaluate(
        self, 
        t0: float, 
        t1: Optional[float] = None, 
        left: bool = True
    ) -> Array:
        """
        Evaluate Brownian increment between t0 and t1.
        
        For custom noise, we provide the exact increment for our interval.
        Diffrax will call this to get dW values.
        
        This method must be JIT-compatible, so we use jnp.where() instead
        of Python if statements to avoid tracer boolean conversion errors.
        
        Parameters
        ----------
        t0 : float
            Start time of query
        t1 : Optional[float]
            End time of query (if None, return value at t0)
        left : bool
            Whether to use left or right limit
            
        Returns
        -------
        Array
            Brownian increment or value
        """
        if t1 is None:
            # Query for B(t0) - return cumulative value
            # Use linear interpolation: B(t) = dW * (t - t0) / (t1 - t0)
            dt = self.t1 - self.t0
            
            # Avoid division by zero
            alpha = jnp.where(
                dt != 0,
                (t0 - self.t0) / dt,
                0.0
            )
            
            return self.dW * alpha
        else:
            # Query for B(t1) - B(t0) = increment
            # Scale by sqrt(dt_query / dt_total) for sub-intervals
            dt_total = self.t1 - self.t0
            dt_query = t1 - t0
            
            # Avoid division by zero
            scale = jnp.where(
                dt_total > 0,
                jnp.sqrt(dt_query / dt_total),
                0.0
            )
            
            return self.dW * scale


def create_custom_or_random_brownian(
    key, 
    t0: float, 
    t1: float, 
    shape: Tuple[int, ...],
    dW: Optional[Array] = None
):
    """
    Create either custom or random Brownian motion for Diffrax.
    
    Parameters
    ----------
    key : jax.random.PRNGKey
        Random key (used if dW is None)
    t0 : float
        Start time
    t1 : float
        End time  
    shape : tuple
        Noise shape (nw,)
    dW : Optional[Array]
        Custom Brownian increment. If None, generates random.
        
    Returns
    -------
    Brownian motion object for Diffrax
    
    Examples
    --------
    >>> # Random noise
    >>> key = jax.random.PRNGKey(42)
    >>> brownian = create_custom_or_random_brownian(key, 0, 0.01, (1,))
    >>> 
    >>> # Custom noise (deterministic)
    >>> dW = jnp.array([0.5])
    >>> brownian = create_custom_or_random_brownian(key, 0, 0.01, (1,), dW=dW)
    """
    if dW is not None:
        # Use custom noise
        return CustomBrownianPath(t0, t1, dW)
    else:
        # Use Diffrax's random noise generator
        return dfx.VirtualBrownianTree(
            t0, t1, tol=1e-3, shape=shape, key=key
        )