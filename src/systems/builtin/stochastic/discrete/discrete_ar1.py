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

# TODO: make module docstring

import sympy as sp

from src.systems.base.discrete_stochastic_system import DiscreteStochasticSystem


class DiscreteAR1(DiscreteStochasticSystem):
    """
    AR(1) process: x[k+1] = φ*x[k] + u[k] + σ*w[k]

    Autoregressive process with optional control input.
    """

    def define_system(self, phi=0.9, sigma=0.1):
        x = sp.symbols("x", real=True)
        u = sp.symbols("u", real=True)
        phi_sym, sigma_sym = sp.symbols("phi sigma", real=True)

        self.state_vars = [x]
        self.control_vars = [u]

        # AR(1) dynamics
        self._f_sym = sp.Matrix([phi_sym * x + u])

        # Additive noise
        self.diffusion_expr = sp.Matrix([[sigma_sym]])
        self.parameters = {phi_sym: phi, sigma_sym: sigma}
        self.order = 1
        self.sde_type = "ito"
