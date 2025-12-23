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

import pytest
import numpy as np
import warnings
from src.systems.base.utils.equilibrium_handler import EquilibriumHandler


class TestEquilibriumHandlerInit:
    """Test initialization and origin equilibrium"""

    def test_init_creates_origin(self):
        """Origin equilibrium should be created on initialization"""
        handler = EquilibriumHandler(nx=3, nu=2)
        
        assert handler.nx == 3
        assert handler.nu == 2
        assert "origin" in handler.list_names()
        assert handler._default == "origin"

    def test_origin_values(self):
        """Origin should have zero state and control"""
        handler = EquilibriumHandler(nx=4, nu=1)
        
        x_eq = handler.get_x("origin")
        u_eq = handler.get_u("origin")
        
        assert np.allclose(x_eq, np.zeros(4))
        assert np.allclose(u_eq, np.zeros(1))

    def test_different_dimensions(self):
        """Test various state/control dimensions"""
        for nx, nu in [(1, 1), (5, 3), (10, 2)]:
            handler = EquilibriumHandler(nx=nx, nu=nu)
            assert handler.get_x("origin").shape == (nx,)
            assert handler.get_u("origin").shape == (nu,)


class TestAddEquilibrium:
    """Test adding equilibrium points"""

    def test_add_basic(self):
        """Add simple equilibrium without verification"""
        handler = EquilibriumHandler(nx=2, nu=1)
        
        x_eq = np.array([1.0, 2.0])
        u_eq = np.array([0.5])
        
        handler.add("test", x_eq, u_eq)
        
        assert "test" in handler.list_names()
        assert np.allclose(handler.get_x("test"), x_eq)
        assert np.allclose(handler.get_u("test"), u_eq)

    def test_add_with_metadata(self):
        """Add equilibrium with custom metadata"""
        handler = EquilibriumHandler(nx=2, nu=1)
        
        handler.add(
            "stable", 
            np.array([1.0, 0.0]), 
            np.array([0.0]),
            stability="stable",
            description="Test equilibrium"
        )
        
        metadata = handler.get_metadata("stable")
        assert metadata["stability"] == "stable"
        assert metadata["description"] == "Test equilibrium"

    def test_add_scalar_conversion(self):
        """Scalar inputs should be converted to 1D arrays"""
        handler = EquilibriumHandler(nx=1, nu=1)
        
        handler.add("scalar", 5.0, 3.0)
        
        x_eq = handler.get_x("scalar")
        u_eq = handler.get_u("scalar")
        
        assert x_eq.shape == (1,)
        assert u_eq.shape == (1,)
        assert x_eq[0] == 5.0
        assert u_eq[0] == 3.0

    def test_add_wrong_x_dimension(self):
        """Adding equilibrium with wrong state dimension should raise error"""
        handler = EquilibriumHandler(nx=3, nu=2)
        
        with pytest.raises(ValueError, match="x_eq must have shape"):
            handler.add("bad", np.array([1.0, 2.0]), np.zeros(2))

    def test_add_wrong_u_dimension(self):
        """Adding equilibrium with wrong control dimension should raise error"""
        handler = EquilibriumHandler(nx=3, nu=2)
        
        with pytest.raises(ValueError, match="u_eq must have shape"):
            handler.add("bad", np.zeros(3), np.array([1.0]))

    def test_add_with_verification_valid(self):
        """Valid equilibrium should pass verification"""
        handler = EquilibriumHandler(nx=2, nu=1)
        
        # Dynamics: dx = u - x (equilibrium when u = x)
        def dynamics(x, u):
            return u[0] - x
        
        x_eq = np.array([2.0, 2.0])
        u_eq = np.array([2.0])
        
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            handler.add("valid", x_eq, u_eq, verify_fn=dynamics)
        
        metadata = handler.get_metadata("valid")
        assert metadata["verified"] is True
        assert metadata["max_residual"] < 1e-6

    def test_add_with_verification_invalid(self):
        """Invalid equilibrium should warn and set verified=False"""
        handler = EquilibriumHandler(nx=2, nu=1)
        
        # Dynamics: dx = u - x (equilibrium when u = x)
        def dynamics(x, u):
            return u[0] - x
        
        x_eq = np.array([1.0, 1.0])
        u_eq = np.array([5.0])  # Not an equilibrium
        
        with pytest.warns(UserWarning, match="may not be valid"):
            handler.add("invalid", x_eq, u_eq, verify_fn=dynamics, tol=1e-6)
        
        metadata = handler.get_metadata("invalid")
        assert metadata["verified"] is False
        assert metadata["max_residual"] > 1e-6

    def test_add_with_custom_tolerance(self):
        """Custom tolerance should affect verification"""
        handler = EquilibriumHandler(nx=2, nu=1)
        
        def dynamics(x, u):
            return np.array([0.001, 0.001])  # Small residual
        
        x_eq = np.array([0.0, 0.0])
        u_eq = np.array([0.0])
        
        # Should fail with strict tolerance
        with pytest.warns(UserWarning):
            handler.add("strict", x_eq, u_eq, verify_fn=dynamics, tol=1e-6)
        
        # Should pass with relaxed tolerance
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            handler.add("relaxed", x_eq, u_eq, verify_fn=dynamics, tol=1e-2)


class TestGetEquilibrium:
    """Test retrieving equilibrium points"""

    def test_get_x_default(self):
        """get_x() without name should return default equilibrium"""
        handler = EquilibriumHandler(nx=2, nu=1)
        
        x_origin = handler.get_x()
        assert np.allclose(x_origin, np.zeros(2))

    def test_get_u_default(self):
        """get_u() without name should return default equilibrium"""
        handler = EquilibriumHandler(nx=2, nu=1)
        
        u_origin = handler.get_u()
        assert np.allclose(u_origin, np.zeros(1))

    def test_get_both(self):
        """get_both should return both state and control"""
        handler = EquilibriumHandler(nx=2, nu=1)
        handler.add("test", np.array([1.0, 2.0]), np.array([3.0]))
        
        x_eq, u_eq = handler.get_both("test")
        
        assert np.allclose(x_eq, np.array([1.0, 2.0]))
        assert np.allclose(u_eq, np.array([3.0]))

    def test_get_nonexistent_equilibrium(self):
        """Getting nonexistent equilibrium should raise error"""
        handler = EquilibriumHandler(nx=2, nu=1)
        
        with pytest.raises(ValueError, match="Unknown equilibrium"):
            handler.get_x("nonexistent")
        
        with pytest.raises(ValueError, match="Unknown equilibrium"):
            handler.get_u("nonexistent")


class TestBackendConversion:
    """Test conversion to different computational backends"""

    def test_numpy_backend(self):
        """NumPy backend should return NumPy arrays"""
        handler = EquilibriumHandler(nx=2, nu=1)
        handler.add("test", np.array([1.0, 2.0]), np.array([3.0]))
        
        x_eq = handler.get_x("test", backend="numpy")
        
        assert isinstance(x_eq, np.ndarray)
        assert np.allclose(x_eq, np.array([1.0, 2.0]))

    def test_torch_backend(self):
        """PyTorch backend should return torch tensors"""
        torch = pytest.importorskip("torch")
        
        handler = EquilibriumHandler(nx=2, nu=1)
        handler.add("test", np.array([1.0, 2.0]), np.array([3.0]))
        
        x_eq = handler.get_x("test", backend="torch")
        u_eq = handler.get_u("test", backend="torch")
        
        assert isinstance(x_eq, torch.Tensor)
        assert isinstance(u_eq, torch.Tensor)
        # Should preserve NumPy's default float64 dtype
        assert x_eq.dtype == torch.float64
        assert torch.allclose(x_eq, torch.tensor([1.0, 2.0], dtype=torch.float64))

    def test_jax_backend(self):
        """JAX backend should return JAX arrays"""
        jax = pytest.importorskip("jax")
        
        handler = EquilibriumHandler(nx=2, nu=1)
        handler.add("test", np.array([1.0, 2.0]), np.array([3.0]))
        
        x_eq = handler.get_x("test", backend="jax")
        u_eq = handler.get_u("test", backend="jax")
        
        assert isinstance(x_eq, jax.Array)
        assert isinstance(u_eq, jax.Array)
        assert np.allclose(x_eq, np.array([1.0, 2.0]))

    def test_get_both_backend_conversion(self):
        """get_both should support backend conversion"""
        torch = pytest.importorskip("torch")
        
        handler = EquilibriumHandler(nx=2, nu=1)
        handler.add("test", np.array([1.0, 2.0]), np.array([3.0]))
        
        x_eq, u_eq = handler.get_both("test", backend="torch")
        
        assert isinstance(x_eq, torch.Tensor)
        assert isinstance(u_eq, torch.Tensor)

    def test_invalid_backend(self):
        """Invalid backend should raise error"""
        handler = EquilibriumHandler(nx=2, nu=1)
        
        with pytest.raises(ValueError, match="Unknown backend"):
            handler.get_x("origin", backend="tensorflow")


class TestDefaultEquilibrium:
    """Test default equilibrium management"""

    def test_initial_default(self):
        """Initial default should be 'origin'"""
        handler = EquilibriumHandler(nx=2, nu=1)
        assert handler._default == "origin"

    def test_set_default(self):
        """Should be able to change default equilibrium"""
        handler = EquilibriumHandler(nx=2, nu=1)
        handler.add("new_default", np.array([1.0, 2.0]), np.array([3.0]))
        
        handler.set_default("new_default")
        
        assert handler._default == "new_default"
        x_eq = handler.get_x()  # No name provided
        assert np.allclose(x_eq, np.array([1.0, 2.0]))

    def test_set_nonexistent_default(self):
        """Setting nonexistent equilibrium as default should raise error"""
        handler = EquilibriumHandler(nx=2, nu=1)
        
        with pytest.raises(ValueError, match="Unknown equilibrium"):
            handler.set_default("nonexistent")

    def test_get_metadata_default(self):
        """get_metadata without name should use default"""
        handler = EquilibriumHandler(nx=2, nu=1)
        handler.add("test", np.array([1.0, 2.0]), np.array([3.0]), info="test")
        handler.set_default("test")
        
        metadata = handler.get_metadata()
        assert metadata["info"] == "test"


class TestListAndMetadata:
    """Test listing and metadata operations"""

    def test_list_names_initial(self):
        """Initial list should contain only 'origin'"""
        handler = EquilibriumHandler(nx=2, nu=1)
        names = handler.list_names()
        
        assert len(names) == 1
        assert "origin" in names

    def test_list_names_multiple(self):
        """List should contain all added equilibria"""
        handler = EquilibriumHandler(nx=2, nu=1)
        handler.add("eq1", np.zeros(2), np.zeros(1))
        handler.add("eq2", np.ones(2), np.ones(1))
        
        names = handler.list_names()
        
        assert len(names) == 3
        assert all(name in names for name in ["origin", "eq1", "eq2"])

    def test_get_metadata_origin(self):
        """Origin should have empty metadata"""
        handler = EquilibriumHandler(nx=2, nu=1)
        metadata = handler.get_metadata("origin")
        
        assert isinstance(metadata, dict)
        assert len(metadata) == 0

    def test_get_metadata_nonexistent(self):
        """Getting metadata for nonexistent equilibrium should raise error"""
        handler = EquilibriumHandler(nx=2, nu=1)
        
        with pytest.raises(ValueError, match="Unknown equilibrium"):
            handler.get_metadata("nonexistent")


class TestRepr:
    """Test string representation"""

    def test_repr_basic(self):
        """Repr should show number of equilibria and names"""
        handler = EquilibriumHandler(nx=2, nu=1)
        repr_str = repr(handler)
        
        assert "EquilibriumHandler" in repr_str
        assert "1 equilibria" in repr_str or "1 equilibrium" in repr_str
        assert "origin" in repr_str

    def test_repr_multiple(self):
        """Repr should update with added equilibria"""
        handler = EquilibriumHandler(nx=2, nu=1)
        handler.add("eq1", np.zeros(2), np.zeros(1))
        handler.add("eq2", np.ones(2), np.ones(1))
        
        repr_str = repr(handler)
        
        assert "3 equilibria" in repr_str
        assert "eq1" in repr_str
        assert "eq2" in repr_str


class TestDimensionUpdates:
    """Test dimension property setters"""

    def test_update_nx_from_zero(self):
        """Updating nx from 0 should recreate origin with correct dimensions"""
        handler = EquilibriumHandler(nx=0, nu=0)
        
        # Origin starts with shape (0,)
        assert handler.get_x("origin").shape == (0,)
        
        # Update dimensions
        handler.nx = 3
        
        # Origin should be recreated with correct shape
        x_eq = handler.get_x("origin")
        assert x_eq.shape == (3,)
        assert np.allclose(x_eq, np.zeros(3))

    def test_update_nu_from_zero(self):
        """Updating nu from 0 should recreate origin with correct dimensions"""
        handler = EquilibriumHandler(nx=0, nu=0)
        
        # Origin starts with shape (0,)
        assert handler.get_u("origin").shape == (0,)
        
        # Update dimensions
        handler.nu = 2
        
        # Origin should be recreated with correct shape
        u_eq = handler.get_u("origin")
        assert u_eq.shape == (2,)
        assert np.allclose(u_eq, np.zeros(2))

    def test_update_both_dimensions_from_zero(self):
        """Updating both dimensions should work correctly"""
        handler = EquilibriumHandler(nx=0, nu=0)
        
        handler.nx = 4
        handler.nu = 2
        
        x_eq, u_eq = handler.get_both("origin")
        assert x_eq.shape == (4,)
        assert u_eq.shape == (2,)
        assert np.allclose(x_eq, np.zeros(4))
        assert np.allclose(u_eq, np.zeros(2))

    def test_cannot_change_nx_with_wrong_equilibria(self):
        """Changing nx should fail if existing equilibria have wrong dimensions"""
        handler = EquilibriumHandler(nx=2, nu=1)
        handler.add("test", np.array([1.0, 2.0]), np.array([0.5]))
        
        with pytest.raises(ValueError, match="Cannot change nx"):
            handler.nx = 3

    def test_cannot_change_nu_with_wrong_equilibria(self):
        """Changing nu should fail if existing equilibria have wrong dimensions"""
        handler = EquilibriumHandler(nx=2, nu=1)
        handler.add("test", np.array([1.0, 2.0]), np.array([0.5]))
        
        with pytest.raises(ValueError, match="Cannot change nu"):
            handler.nu = 2

    def test_can_set_same_dimensions(self):
        """Setting same dimensions should not raise error"""
        handler = EquilibriumHandler(nx=2, nu=1)
        handler.add("test", np.array([1.0, 2.0]), np.array([0.5]))
        
        # Should not raise
        handler.nx = 2
        handler.nu = 1
        
        # Equilibria should still be accessible
        x_eq = handler.get_x("test")
        assert np.allclose(x_eq, np.array([1.0, 2.0]))

    def test_dimension_properties_readable(self):
        """Dimension properties should be readable"""
        handler = EquilibriumHandler(nx=5, nu=3)
        
        assert handler.nx == 5
        assert handler.nu == 3


class TestEdgeCases:
    """Test edge cases and boundary conditions"""

    def test_overwrite_origin(self):
        """Adding equilibrium named 'origin' should overwrite default"""
        handler = EquilibriumHandler(nx=2, nu=1)
        
        handler.add("origin", np.array([5.0, 5.0]), np.array([5.0]))
        
        x_eq = handler.get_x("origin")
        assert np.allclose(x_eq, np.array([5.0, 5.0]))

    def test_single_dimension_system(self):
        """Test with nx=1, nu=1"""
        handler = EquilibriumHandler(nx=1, nu=1)
        handler.add("test", np.array([2.0]), np.array([3.0]))
        
        x_eq, u_eq = handler.get_both("test")
        
        assert x_eq.shape == (1,)
        assert u_eq.shape == (1,)
        assert x_eq[0] == 2.0
        assert u_eq[0] == 3.0

    def test_large_dimension_system(self):
        """Test with large state/control dimensions"""
        nx, nu = 100, 50
        handler = EquilibriumHandler(nx=nx, nu=nu)
        
        x_eq = np.random.randn(nx)
        u_eq = np.random.randn(nu)
        
        handler.add("large", x_eq, u_eq)
        
        x_retrieved = handler.get_x("large")
        u_retrieved = handler.get_u("large")
        
        assert np.allclose(x_retrieved, x_eq)
        assert np.allclose(u_retrieved, u_eq)

    def test_multiple_backends_same_data(self):
        """Same equilibrium should be consistent across backends"""
        torch = pytest.importorskip("torch")
        jax = pytest.importorskip("jax")
        
        handler = EquilibriumHandler(nx=3, nu=2)
        x_ref = np.array([1.0, 2.0, 3.0])
        u_ref = np.array([4.0, 5.0])
        
        handler.add("multi", x_ref, u_ref)
        
        x_np = handler.get_x("multi", backend="numpy")
        x_torch = handler.get_x("multi", backend="torch")
        x_jax = handler.get_x("multi", backend="jax")
        
        assert np.allclose(x_np, x_ref)
        # torch should preserve float64 from numpy
        assert torch.allclose(x_torch, torch.tensor(x_ref, dtype=torch.float64))
        assert np.allclose(np.array(x_jax), x_ref)


if __name__ == "__main__":
    # Run tests with pytest when executed directly
    pytest.main([__file__, "-v", "--tb=short"])