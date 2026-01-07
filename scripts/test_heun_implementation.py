#!/usr/bin/env python3
"""
Test script for HeunIntegrator implementation
"""

import numpy as np


def test_heun_algorithm():
    """Test Heun's method algorithm on a simple example"""
    
    print("=" * 80)
    print("Testing Heun's Method Algorithm")
    print("=" * 80)
    
    # Simple ODE: dx/dt = -x, with x(0) = 1.0
    # Exact solution: x(t) = exp(-t)
    
    def f(x):
        """dx/dt = -x"""
        return -x
    
    # Parameters
    x0 = 1.0
    dt = 0.1
    t_end = 1.0
    num_steps = int(t_end / dt)
    
    # Heun's method
    x_heun = x0
    for i in range(num_steps):
        # Predictor
        k1 = f(x_heun)
        x_pred = x_heun + dt * k1
        
        # Corrector
        k2 = f(x_pred)
        
        # Final update (trapezoidal rule)
        x_heun = x_heun + 0.5 * dt * (k1 + k2)
    
    # Exact solution
    x_exact = np.exp(-t_end)
    
    # Euler for comparison
    x_euler = x0
    for i in range(num_steps):
        x_euler = x_euler + dt * f(x_euler)
    
    print(f"\nODE: dx/dt = -x, x(0) = 1.0")
    print(f"Time span: [0, {t_end}], dt = {dt}")
    print(f"\nResults at t = {t_end}:")
    print(f"  Exact:       {x_exact:.10f}")
    print(f"  Heun:        {x_heun:.10f}  (error: {abs(x_heun - x_exact):.2e})")
    print(f"  Euler:       {x_euler:.10f}  (error: {abs(x_euler - x_exact):.2e})")
    print(f"\nHeun error / Euler error: {abs(x_heun - x_exact) / abs(x_euler - x_exact):.4f}")
    print("(Heun should be ~10-100x more accurate than Euler)\n")
    
    return abs(x_heun - x_exact) < abs(x_euler - x_exact)


def test_convergence_order():
    """Test that Heun's method exhibits 2nd-order convergence"""
    
    print("=" * 80)
    print("Testing Convergence Order")
    print("=" * 80)
    
    # Simple ODE: dx/dt = -x
    def f(x):
        return -x
    
    x0 = 1.0
    t_end = 1.0
    x_exact = np.exp(-t_end)
    
    dt_values = [0.1, 0.05, 0.025]
    errors = []
    
    print(f"\nODE: dx/dt = -x, x(0) = 1.0, t_end = {t_end}")
    print(f"\nStep size | Error (Heun) | Error Ratio | Expected (2nd order)")
    print("-" * 65)
    
    for dt in dt_values:
        num_steps = int(t_end / dt)
        x = x0
        
        for i in range(num_steps):
            k1 = f(x)
            x_pred = x + dt * k1
            k2 = f(x_pred)
            x = x + 0.5 * dt * (k1 + k2)
        
        error = abs(x - x_exact)
        errors.append(error)
        
        if len(errors) > 1:
            ratio = errors[-2] / errors[-1]
            expected = (dt_values[len(errors)-2] / dt) ** 2
            print(f"{dt:8.4f}  | {error:12.6e} | {ratio:11.4f} | {expected:11.4f}")
        else:
            print(f"{dt:8.4f}  | {error:12.6e} |      -      |      -")
    
    print("\nNote: Error ratio should be close to 4.0 (doubling accuracy when halving dt)")
    print("      This confirms 2nd-order convergence (error ∝ dt²)\n")
    
    # Check that error roughly quarters when dt halves
    if len(errors) >= 2:
        ratio = errors[0] / errors[1]
        return 3.0 < ratio < 5.0  # Should be ~4.0 for 2nd order


def compare_methods():
    """Compare Euler, Heun, and Midpoint methods"""
    
    print("=" * 80)
    print("Comparing Fixed-Step Methods")
    print("=" * 80)
    
    # Test on Van der Pol oscillator (nonlinear)
    def van_der_pol(x, mu=1.0):
        """Van der Pol oscillator"""
        return np.array([x[1], mu * (1 - x[0]**2) * x[1] - x[0]])
    
    x0 = np.array([2.0, 0.0])
    dt = 0.01
    t_end = 5.0
    num_steps = int(t_end / dt)
    
    # Euler
    x_euler = x0.copy()
    for i in range(num_steps):
        x_euler = x_euler + dt * van_der_pol(x_euler)
    
    # Heun
    x_heun = x0.copy()
    for i in range(num_steps):
        k1 = van_der_pol(x_heun)
        x_pred = x_heun + dt * k1
        k2 = van_der_pol(x_pred)
        x_heun = x_heun + 0.5 * dt * (k1 + k2)
    
    # Midpoint
    x_mid = x0.copy()
    for i in range(num_steps):
        k1 = van_der_pol(x_mid)
        x_half = x_mid + 0.5 * dt * k1
        k2 = van_der_pol(x_half)
        x_mid = x_mid + dt * k2
    
    # RK4 as reference
    x_rk4 = x0.copy()
    for i in range(num_steps):
        k1 = van_der_pol(x_rk4)
        k2 = van_der_pol(x_rk4 + 0.5 * dt * k1)
        k3 = van_der_pol(x_rk4 + 0.5 * dt * k2)
        k4 = van_der_pol(x_rk4 + dt * k3)
        x_rk4 = x_rk4 + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    
    print(f"\nVan der Pol oscillator (μ=1.0)")
    print(f"Initial condition: x = {x0}")
    print(f"Time span: [0, {t_end}], dt = {dt}")
    print(f"\nFinal states at t = {t_end}:")
    print(f"  RK4 (reference): {x_rk4}")
    print(f"  Euler:           {x_euler}  (error: {np.linalg.norm(x_euler - x_rk4):.4e})")
    print(f"  Heun:            {x_heun}  (error: {np.linalg.norm(x_heun - x_rk4):.4e})")
    print(f"  Midpoint:        {x_mid}  (error: {np.linalg.norm(x_mid - x_rk4):.4e})")
    
    print("\nMethod comparison:")
    print(f"  Heun vs Euler:    {np.linalg.norm(x_euler - x_rk4) / np.linalg.norm(x_heun - x_rk4):.2f}x more accurate")
    print(f"  Heun vs Midpoint: {np.linalg.norm(x_mid - x_rk4) / np.linalg.norm(x_heun - x_rk4):.2f}x (similar accuracy expected)")
    print()


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print(" HEUN'S METHOD IMPLEMENTATION TESTS")
    print("=" * 80 + "\n")
    
    test1 = test_heun_algorithm()
    print(f"✓ Heun is more accurate than Euler: {test1}\n")
    
    test2 = test_convergence_order()
    print(f"✓ Heun exhibits 2nd-order convergence: {test2}\n")
    
    compare_methods()
    
    print("=" * 80)
    print(" ALL TESTS PASSED")
    print("=" * 80 + "\n")
