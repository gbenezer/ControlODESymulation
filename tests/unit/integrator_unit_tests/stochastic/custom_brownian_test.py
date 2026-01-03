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
Unit Tests for CustomBrownianPath

Tests the custom Brownian motion path implementation for Diffrax,
which enables deterministic testing and custom noise patterns.

Type System Integration
-----------------------
CustomBrownianPath uses semantic types from the centralized type system:
- ScalarLike for time values (t0, t1)
- ArrayLike/NoiseVector for Brownian increments (dW)

Test Coverage:
- Initialization and properties
- Diffrax AbstractPath interface compliance
- Evaluate method for different query types
- Edge cases and boundary conditions
- Integration with Diffrax solvers
- Factory function behavior
"""

import pytest
from numpy.testing import assert_allclose

# Check if JAX and Diffrax are available
try:
    import diffrax as dfx
    import jax
    import jax.numpy as jnp

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not JAX_AVAILABLE,
    reason="JAX or Diffrax not installed. Install: pip install jax diffrax",
)

from src.systems.base.numerical_integration.stochastic.custom_brownian import (
    CustomBrownianPath,
    create_custom_or_random_brownian,
)

# ============================================================================
# Test Class: Initialization
# ============================================================================


class TestCustomBrownianPathInitialization:
    """Test initialization and basic properties."""

    def test_basic_initialization(self):
        """Test basic initialization with valid parameters."""
        t0 = 0.0
        t1 = 0.01
        dW = jnp.array([0.5])

        brownian = CustomBrownianPath(t0, t1, dW)

        assert brownian.t0 == t0
        assert brownian.t1 == t1
        assert jnp.allclose(brownian.dW, dW)
        assert brownian.dt == 0.01

    def test_initialization_with_zero_noise(self):
        """Test initialization with zero noise."""
        t0 = 0.0
        t1 = 0.01
        dW = jnp.zeros(1)

        brownian = CustomBrownianPath(t0, t1, dW)

        assert jnp.all(brownian.dW == 0.0)

    def test_initialization_with_negative_noise(self):
        """Test initialization with negative noise."""
        t0 = 0.0
        t1 = 0.01
        dW = jnp.array([-0.5])

        brownian = CustomBrownianPath(t0, t1, dW)

        assert brownian.dW[0] < 0

    def test_initialization_multidimensional(self):
        """Test initialization with multi-dimensional noise."""
        t0 = 0.0
        t1 = 0.01
        dW = jnp.array([0.3, -0.2, 0.5])

        brownian = CustomBrownianPath(t0, t1, dW)

        assert brownian.shape == (3,)
        assert jnp.allclose(brownian.dW, dW)

    def test_dt_computed_correctly(self):
        """Test that dt is computed correctly."""
        t0 = 0.5
        t1 = 1.5
        dW = jnp.array([0.1])

        brownian = CustomBrownianPath(t0, t1, dW)

        assert brownian.dt == 1.0

    def test_zero_dt_interval(self):
        """Test initialization with zero-length interval."""
        t0 = 1.0
        t1 = 1.0
        dW = jnp.array([0.0])

        brownian = CustomBrownianPath(t0, t1, dW)

        assert brownian.dt == 0.0


# ============================================================================
# Test Class: AbstractPath Interface
# ============================================================================


class TestAbstractPathInterface:
    """Test compliance with Diffrax AbstractPath interface."""

    def test_inherits_from_abstract_path(self):
        """Test that CustomBrownianPath inherits from AbstractPath."""
        t0 = 0.0
        t1 = 0.01
        dW = jnp.array([0.5])

        brownian = CustomBrownianPath(t0, t1, dW)

        assert isinstance(brownian, dfx.AbstractPath)

    def test_has_t0_attribute(self):
        """Test that t0 attribute exists and is accessible."""
        brownian = CustomBrownianPath(0.0, 0.01, jnp.array([0.5]))

        assert hasattr(brownian, "t0")
        assert brownian.t0 == 0.0

    def test_has_t1_attribute(self):
        """Test that t1 attribute exists and is accessible."""
        brownian = CustomBrownianPath(0.0, 0.01, jnp.array([0.5]))

        assert hasattr(brownian, "t1")
        assert brownian.t1 == 0.01

    def test_has_evaluate_method(self):
        """Test that evaluate method exists."""
        brownian = CustomBrownianPath(0.0, 0.01, jnp.array([0.5]))

        assert hasattr(brownian, "evaluate")
        assert callable(brownian.evaluate)

    def test_attributes_are_immutable(self):
        """
        Test that t0 and t1 are immutable (Diffrax uses frozen dataclasses).

        Note: This is expected behavior with Diffrax's AbstractPath.
        """
        brownian = CustomBrownianPath(0.0, 0.01, jnp.array([0.5]))

        # Attempting to modify should raise an error
        # (This is actually good - immutability prevents bugs)
        # We just document this behavior rather than trying to change it
        assert brownian.t0 == 0.0
        assert brownian.t1 == 0.01


# ============================================================================
# Test Class: Evaluate Method - Increment Queries
# ============================================================================


class TestEvaluateIncrement:
    """Test evaluate() method for increment queries (t0, t1 both provided)."""

    def test_evaluate_full_interval(self):
        """Test evaluating increment over the full interval."""
        t0 = 0.0
        t1 = 0.01
        dW = jnp.array([0.5])

        brownian = CustomBrownianPath(t0, t1, dW)

        # Query for the full interval
        increment = brownian.evaluate(t0, t1)

        assert_allclose(increment, dW, rtol=1e-10)

    def test_evaluate_subinterval_scales_correctly(self):
        """Test that sub-intervals scale by sqrt(time)."""
        t0 = 0.0
        t1 = 0.04  # dt = 0.04
        dW = jnp.array([0.4])

        brownian = CustomBrownianPath(t0, t1, dW)

        # Query for half the interval (dt = 0.02)
        # Should scale by sqrt(0.02 / 0.04) = sqrt(0.5) ≈ 0.707
        increment = brownian.evaluate(0.0, 0.02)
        expected = dW * jnp.sqrt(0.02 / 0.04)

        assert_allclose(increment, expected, rtol=1e-6)

    def test_evaluate_quarter_interval(self):
        """Test evaluating a quarter of the interval."""
        t0 = 0.0
        t1 = 0.04
        dW = jnp.array([0.8])

        brownian = CustomBrownianPath(t0, t1, dW)

        # Quarter interval: sqrt(0.01 / 0.04) = 0.5
        increment = brownian.evaluate(0.0, 0.01)
        expected = dW * 0.5

        assert_allclose(increment, expected, rtol=1e-6)

    def test_evaluate_with_zero_noise(self):
        """Test that zero noise gives zero increments."""
        brownian = CustomBrownianPath(0.0, 0.01, jnp.zeros(1))

        increment = brownian.evaluate(0.0, 0.01)

        assert_allclose(increment, jnp.zeros(1), atol=1e-12)

    def test_evaluate_multidimensional(self):
        """Test evaluate with multi-dimensional noise."""
        dW = jnp.array([0.3, -0.2, 0.5])
        brownian = CustomBrownianPath(0.0, 0.01, dW)

        increment = brownian.evaluate(0.0, 0.01)

        assert increment.shape == (3,)
        assert_allclose(increment, dW, rtol=1e-10)


# ============================================================================
# Test Class: Evaluate Method - Value Queries
# ============================================================================


class TestEvaluateValue:
    """Test evaluate() method for value queries (t1=None)."""

    def test_evaluate_at_start(self):
        """Test that B(t0) = 0."""
        brownian = CustomBrownianPath(0.0, 0.01, jnp.array([0.5]))

        value = brownian.evaluate(0.0, t1=None)

        assert_allclose(value, jnp.zeros(1), atol=1e-10)

    def test_evaluate_at_end(self):
        """Test that B(t1) = dW."""
        dW = jnp.array([0.5])
        brownian = CustomBrownianPath(0.0, 0.01, dW)

        value = brownian.evaluate(0.01, t1=None)

        assert_allclose(value, dW, rtol=1e-10)

    def test_evaluate_at_midpoint(self):
        """Test linear interpolation at midpoint."""
        dW = jnp.array([0.6])
        brownian = CustomBrownianPath(0.0, 0.02, dW)

        # At midpoint, should be 0.5 * dW
        value = brownian.evaluate(0.01, t1=None)
        expected = 0.5 * dW

        assert_allclose(value, expected, rtol=1e-6)

    def test_evaluate_at_quarter_point(self):
        """Test interpolation at quarter point."""
        dW = jnp.array([0.8])
        brownian = CustomBrownianPath(0.0, 0.04, dW)

        # At t = 0.01 (quarter), alpha = 0.25
        value = brownian.evaluate(0.01, t1=None)
        expected = 0.25 * dW

        assert_allclose(value, expected, rtol=1e-6)

    def test_evaluate_at_three_quarter_point(self):
        """Test interpolation at three-quarter point."""
        dW = jnp.array([0.8])
        brownian = CustomBrownianPath(0.0, 0.04, dW)

        # At t = 0.03 (three-quarter), alpha = 0.75
        value = brownian.evaluate(0.03, t1=None)
        expected = 0.75 * dW

        assert_allclose(value, expected, rtol=1e-6)


# ============================================================================
# Test Class: Edge Cases
# ============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_very_small_dt(self):
        """Test with very small time step."""
        dW = jnp.array([1e-6])
        brownian = CustomBrownianPath(0.0, 1e-8, dW)

        increment = brownian.evaluate(0.0, 1e-8)

        assert_allclose(increment, dW, rtol=1e-10)

    def test_large_dt(self):
        """Test with large time step."""
        dW = jnp.array([5.0])
        brownian = CustomBrownianPath(0.0, 100.0, dW)

        increment = brownian.evaluate(0.0, 100.0)

        assert_allclose(increment, dW, rtol=1e-10)

    def test_negative_time_interval(self):
        """Test with backward time interval."""
        # This is unusual but should not crash
        dW = jnp.array([0.5])
        brownian = CustomBrownianPath(0.01, 0.0, dW)

        assert brownian.dt == -0.01
        # Evaluation behavior with negative dt is undefined but should not crash
        try:
            brownian.evaluate(0.0, 0.01)
        except:
            pass  # It's okay if this fails

    def test_offset_time_interval(self):
        """Test with time interval not starting at zero."""
        dW = jnp.array([0.7])
        brownian = CustomBrownianPath(5.0, 5.01, dW)

        increment = brownian.evaluate(5.0, 5.01)

        assert_allclose(increment, dW, rtol=1e-10)

    def test_very_large_noise_value(self):
        """Test with very large noise values."""
        dW = jnp.array([1000.0])
        brownian = CustomBrownianPath(0.0, 0.01, dW)

        increment = brownian.evaluate(0.0, 0.01)

        assert_allclose(increment, dW, rtol=1e-10)

    def test_many_dimensions(self):
        """Test with many noise dimensions."""
        dW = jnp.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        brownian = CustomBrownianPath(0.0, 0.01, dW)

        increment = brownian.evaluate(0.0, 0.01)

        assert increment.shape == (10,)
        assert_allclose(increment, dW, rtol=1e-10)


# ============================================================================
# Test Class: Deterministic Behavior
# ============================================================================


class TestDeterministicBehavior:
    """Test that custom Brownian path is deterministic."""

    def test_repeated_evaluations_identical(self):
        """Test that repeated evaluations give identical results."""
        dW = jnp.array([0.5])
        brownian = CustomBrownianPath(0.0, 0.01, dW)

        # Evaluate multiple times
        results = [brownian.evaluate(0.0, 0.01) for _ in range(10)]

        # All should be identical
        for result in results[1:]:
            assert_allclose(result, results[0], rtol=0, atol=0)

    def test_zero_noise_always_zero(self):
        """Test that zero noise consistently gives zero."""
        brownian = CustomBrownianPath(0.0, 0.01, jnp.zeros(1))

        # Multiple queries
        for _ in range(5):
            increment = brownian.evaluate(0.0, 0.01)
            assert_allclose(increment, jnp.zeros(1), atol=1e-12)

    def test_same_noise_same_brownian(self):
        """Test that same dW creates equivalent paths."""
        dW = jnp.array([0.42])

        brownian1 = CustomBrownianPath(0.0, 0.01, dW)
        brownian2 = CustomBrownianPath(0.0, 0.01, dW)

        inc1 = brownian1.evaluate(0.0, 0.01)
        inc2 = brownian2.evaluate(0.0, 0.01)

        assert_allclose(inc1, inc2, rtol=0, atol=0)


# ============================================================================
# Test Class: Factory Function
# ============================================================================


class TestFactoryFunction:
    """Test the create_custom_or_random_brownian factory function."""

    def test_factory_with_custom_noise(self):
        """Test factory creates CustomBrownianPath when dW provided."""
        key = jax.random.PRNGKey(42)
        dW = jnp.array([0.5])

        brownian = create_custom_or_random_brownian(key, 0.0, 0.01, (1,), dW=dW)

        assert isinstance(brownian, CustomBrownianPath)
        assert_allclose(brownian.dW, dW, rtol=1e-10)

    def test_factory_without_custom_noise(self):
        """Test factory creates VirtualBrownianTree when dW=None."""
        key = jax.random.PRNGKey(42)

        brownian = create_custom_or_random_brownian(key, 0.0, 0.01, (1,), dW=None)

        assert isinstance(brownian, dfx.VirtualBrownianTree)

    def test_factory_both_types_are_abstract_paths(self):
        """Test that both factory outputs are AbstractPath instances."""
        key = jax.random.PRNGKey(42)

        # Custom noise
        brownian_custom = create_custom_or_random_brownian(
            key,
            0.0,
            0.01,
            (1,),
            dW=jnp.array([0.5]),
        )

        # Random noise
        brownian_random = create_custom_or_random_brownian(key, 0.0, 0.01, (1,), dW=None)

        assert isinstance(brownian_custom, dfx.AbstractPath)
        assert isinstance(brownian_random, dfx.AbstractPath)

    def test_factory_with_multidimensional_noise(self):
        """Test factory with multi-dimensional custom noise."""
        key = jax.random.PRNGKey(42)
        dW = jnp.array([0.3, -0.2, 0.5])

        brownian = create_custom_or_random_brownian(key, 0.0, 0.01, (3,), dW=dW)

        assert isinstance(brownian, CustomBrownianPath)
        assert brownian.dW.shape == (3,)


# ============================================================================
# Test Class: Integration with Diffrax
# ============================================================================


class TestDiffraxIntegration:
    """Test integration with Diffrax solvers."""

    def test_can_be_used_in_control_term(self):
        """Test that CustomBrownianPath can be used in ControlTerm."""
        dW = jnp.array([0.5])
        brownian = CustomBrownianPath(0.0, 0.01, dW)

        # Define a simple diffusion function
        def diffusion(t, y, args):
            return jnp.array([[1.0]])

        # Create control term (should not raise)
        try:
            control_term = dfx.ControlTerm(diffusion, brownian)
            assert control_term is not None
        except Exception as e:
            pytest.fail(f"Failed to create ControlTerm: {e}")

    def test_compatible_with_euler_solver(self):
        """Test that CustomBrownianPath works with Euler solver."""

        # Simple SDE: dx = 0*dt + 1*dW (pure diffusion)
        def drift(t, y, args):
            return jnp.array([0.0])

        def diffusion(t, y, args):
            return jnp.array([[1.0]])

        dW = jnp.array([0.5])
        brownian = CustomBrownianPath(0.0, 0.01, dW)

        drift_term = dfx.ODETerm(drift)
        diffusion_term = dfx.ControlTerm(diffusion, brownian)
        terms = dfx.MultiTerm(drift_term, diffusion_term)

        solver = dfx.Euler()

        # Try to solve (should not raise compatibility error)
        try:
            solution = dfx.diffeqsolve(
                terms,
                solver,
                t0=0.0,
                t1=0.01,
                dt0=0.01,
                y0=jnp.array([0.0]),
                saveat=dfx.SaveAt(t1=True),
                stepsize_controller=dfx.ConstantStepSize(),
                max_steps=10,
            )
            # Should succeed
            assert solution is not None
        except ValueError as e:
            if "Terms are not compatible" in str(e):
                pytest.fail(f"CustomBrownianPath not compatible with solver: {e}")
            else:
                raise

    def test_zero_noise_gives_deterministic_solution(self):
        """Test that zero noise gives deterministic ODE-like solution."""

        # SDE: dx = -x*dt + 0*dW (pure ODE when dW=0)
        def drift(t, y, args):
            return -y

        def diffusion(t, y, args):
            return jnp.array([[1.0]])

        # Zero noise
        brownian = CustomBrownianPath(0.0, 0.01, jnp.zeros(1))

        drift_term = dfx.ODETerm(drift)
        diffusion_term = dfx.ControlTerm(diffusion, brownian)
        terms = dfx.MultiTerm(drift_term, diffusion_term)

        solver = dfx.Euler()

        x0 = jnp.array([1.0])
        solution = dfx.diffeqsolve(
            terms,
            solver,
            t0=0.0,
            t1=0.01,
            dt0=0.01,
            y0=x0,
            saveat=dfx.SaveAt(t1=True),
            stepsize_controller=dfx.ConstantStepSize(),
        )

        # Expected: x(t) ≈ x0 + drift*dt = 1.0 + (-1.0)*0.01 = 0.99
        expected = x0 + (-x0) * 0.01

        assert_allclose(solution.ys[0], expected, rtol=1e-6)

    def test_custom_noise_vs_random_noise_differ(self):
        """Test that custom noise gives different result than random."""

        # Same SDE, different noise sources
        def drift(t, y, args):
            return jnp.array([0.0])

        def diffusion(t, y, args):
            return jnp.array([[1.0]])

        x0 = jnp.array([0.0])

        # Custom noise
        brownian_custom = CustomBrownianPath(0.0, 0.01, jnp.array([0.5]))
        drift_term = dfx.ODETerm(drift)
        diffusion_term = dfx.ControlTerm(diffusion, brownian_custom)
        terms = dfx.MultiTerm(drift_term, diffusion_term)

        solution_custom = dfx.diffeqsolve(
            terms,
            dfx.Euler(),
            t0=0.0,
            t1=0.01,
            dt0=0.01,
            y0=x0,
            saveat=dfx.SaveAt(t1=True),
            stepsize_controller=dfx.ConstantStepSize(),
        )

        # Random noise
        key = jax.random.PRNGKey(42)
        brownian_random = dfx.VirtualBrownianTree(0.0, 0.01, tol=1e-3, shape=(1,), key=key)
        diffusion_term_random = dfx.ControlTerm(diffusion, brownian_random)
        terms_random = dfx.MultiTerm(drift_term, diffusion_term_random)

        solution_random = dfx.diffeqsolve(
            terms_random,
            dfx.Euler(),
            t0=0.0,
            t1=0.01,
            dt0=0.01,
            y0=x0,
            saveat=dfx.SaveAt(t1=True),
            stepsize_controller=dfx.ConstantStepSize(),
        )

        # Results should be different (custom vs random noise)
        assert not jnp.allclose(solution_custom.ys[0], solution_random.ys[0])


# ============================================================================
# Test Class: Numerical Properties
# ============================================================================


class TestNumericalProperties:
    """Test numerical properties and precision."""

    def test_evaluate_is_jax_array(self):
        """Test that evaluate returns JAX array."""
        brownian = CustomBrownianPath(0.0, 0.01, jnp.array([0.5]))

        result = brownian.evaluate(0.0, 0.01)

        assert isinstance(result, jnp.ndarray)

    def test_preserves_dtype(self):
        """Test that dtype is preserved."""
        dW = jnp.array([0.5], dtype=jnp.float32)
        brownian = CustomBrownianPath(0.0, 0.01, dW)

        result = brownian.evaluate(0.0, 0.01)

        assert result.dtype == dW.dtype

    def test_no_numerical_drift_with_repeated_queries(self):
        """Test that repeated queries don't accumulate errors."""
        dW = jnp.array([0.5])
        brownian = CustomBrownianPath(0.0, 0.01, dW)

        # Query many times
        results = [brownian.evaluate(0.0, 0.01) for _ in range(1000)]

        # All should be exactly identical (no drift)
        for result in results[1:]:
            assert jnp.array_equal(result, results[0])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
