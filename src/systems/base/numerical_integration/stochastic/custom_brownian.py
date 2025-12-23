"""
Custom Brownian path wrapper for Diffrax to support user-provided noise.

This allows deterministic testing and custom noise patterns.
"""

import jax.numpy as jnp
from jax import Array
import diffrax as dfx
from typing import Optional


class CustomBrownianPath:
    """
    Custom Brownian motion that uses provided dW increments.
    
    This is a minimal implementation that allows Diffrax to use
    user-specified noise instead of generating random noise.
    
    Parameters
    ----------
    t0 : float
        Start time
    t1 : float
        End time
    dW : Array
        Brownian increment for interval (t0, t1)
        Shape: (nw,) for the noise dimensions
    shape : tuple
        Shape of noise (nw,)
    
    Examples
    --------
    >>> # Zero noise for deterministic testing
    >>> dW = jnp.zeros(1)
    >>> brownian = CustomBrownianPath(0.0, 0.01, dW, shape=(1,))
    >>> 
    >>> # Custom noise pattern
    >>> dW = jnp.array([0.5])
    >>> brownian = CustomBrownianPath(0.0, 0.01, dW, shape=(1,))
    """
    
    def __init__(self, t0: float, t1: float, dW: Array, shape: tuple):
        self.t0 = t0
        self.t1 = t1
        self.dW = dW
        self.shape = shape
        self.dt = t1 - t0
        
        # Verify shape matches
        if dW.shape != shape:
            raise ValueError(
                f"dW shape {dW.shape} doesn't match expected shape {shape}"
            )
    
    def evaluate(self, t0: float, t1: Optional[float] = None, 
                 left: bool = True) -> Array:
        """
        Evaluate Brownian increment between t0 and t1.
        
        For custom noise, we assume a single step from self.t0 to self.t1.
        Any query within this interval returns the scaled increment.
        
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
            # Query for B(t0) - return scaled position
            if jnp.abs(t0 - self.t0) < 1e-10:
                return jnp.zeros_like(self.dW)
            elif jnp.abs(t0 - self.t1) < 1e-10:
                return self.dW
            else:
                # Linear interpolation for intermediate times
                alpha = (t0 - self.t0) / (self.t1 - self.t0)
                return self.dW * alpha
        else:
            # Query for B(t1) - B(t0)
            # For our single-step case, just return the full increment
            # if querying the full interval
            if (jnp.abs(t0 - self.t0) < 1e-10 and 
                jnp.abs(t1 - self.t1) < 1e-10):
                return self.dW
            else:
                # For sub-intervals, scale proportionally
                dt_query = t1 - t0
                scale = jnp.sqrt(dt_query / self.dt)
                return self.dW * scale


def create_custom_or_random_brownian(
    key, 
    t0: float, 
    t1: float, 
    shape: tuple,
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
        return CustomBrownianPath(t0, t1, dW, shape)
    else:
        # Use Diffrax's random noise generator
        return dfx.VirtualBrownianTree(
            t0, t1, tol=1e-3, shape=shape, key=key
        )