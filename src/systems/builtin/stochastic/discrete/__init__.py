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

from .discrete_ar1 import DiscreteAR1
from .discrete_white_noise import (
    DiscreteWhiteNoise,
    create_standard_white_noise,
    create_measurement_noise,
    create_thermal_noise,
)
from .discrete_random_walk import (
    DiscreteRandomWalk,
    create_symmetric_random_walk,
    create_biased_random_walk,
)
from .discrete_arma11 import DiscreteARMA11, create_economic_arma, create_sensor_arma
from .discrete_var1 import DiscreteVAR1, create_bivariate_var, create_macro_var
from .discrete_garch11 import DiscreteGARCH11, create_equity_garch, create_fx_garch
from .discrete_stochastic_double_integrator import (
    DiscreteStochasticDoubleIntegrator,
    create_digital_servo,
    create_lqg_benchmark_discrete,
    create_spacecraft_discrete,
)
from .discrete_stochastic_pendulum import (
    DiscreteStochasticPendulum,
    create_furuta_pendulum,
    create_rl_pendulum,
)
from .discrete_stochastic_queue import (
    DiscreteStochasticQueue,
    create_call_center_queue,
    create_network_queue,
)
from .discrete_stochastic_reactors import (
    DiscreteStochasticBatchReactor,
    DiscreteStochasticCSTR,
    create_discrete_batch_reactor_with_noise,
    create_discrete_stochastic_cstr_with_noise,
)
from .discrete_stochastic_logistic import (
    DiscreteStochasticLogisticMap,
    create_fixed_point_regime,
    create_chaotic_regime,
    create_edge_of_chaos,
)

# Aliases for convenience
AR1 = DiscreteAR1
WhiteNoise = DiscreteWhiteNoise
RandomWalk = DiscreteRandomWalk
ARMA11 = DiscreteARMA11
VAR1 = DiscreteVAR1
GARCH11 = DiscreteGARCH11
StochasticDoubleIntegrator = DiscreteStochasticDoubleIntegrator
StochasticPendulum = DiscreteStochasticPendulum
StochasticQueue = DiscreteStochasticQueue
StochasticBatchReactor = DiscreteStochasticBatchReactor
StochasticCSTR = DiscreteStochasticCSTR
LogisticMap = DiscreteStochasticLogisticMap

__all__ = [
    # Time series models
    "DiscreteAR1",
    "AR1",
    "DiscreteWhiteNoise",
    "WhiteNoise",
    "create_standard_white_noise",
    "create_measurement_noise",
    "create_thermal_noise",
    "DiscreteRandomWalk",
    "RandomWalk",
    "create_symmetric_random_walk",
    "create_biased_random_walk",
    "DiscreteARMA11",
    "ARMA11",
    "create_economic_arma",
    "create_sensor_arma",
    # Multivariate models
    "DiscreteVAR1",
    "VAR1",
    "create_bivariate_var",
    "create_macro_var",
    # Financial models
    "DiscreteGARCH11",
    "GARCH11",
    "create_equity_garch",
    "create_fx_garch",
    # Control systems
    "DiscreteStochasticDoubleIntegrator",
    "StochasticDoubleIntegrator",
    "create_digital_servo",
    "create_lqg_benchmark_discrete",
    "create_spacecraft_discrete",
    "DiscreteStochasticPendulum",
    "StochasticPendulum",
    "create_furuta_pendulum",
    "create_rl_pendulum",
    # Queueing systems
    "DiscreteStochasticQueue",
    "StochasticQueue",
    "create_call_center_queue",
    "create_network_queue",
    # Reactor systems
    "DiscreteStochasticBatchReactor",
    "StochasticBatchReactor",
    "create_discrete_batch_reactor_with_noise",
    "DiscreteStochasticCSTR",
    "StochasticCSTR",
    "create_discrete_stochastic_cstr_with_noise",
    # Nonlinear dynamics
    "DiscreteStochasticLogisticMap",
    "LogisticMap",
    "create_fixed_point_regime",
    "create_chaotic_regime",
    "create_edge_of_chaos",
]
