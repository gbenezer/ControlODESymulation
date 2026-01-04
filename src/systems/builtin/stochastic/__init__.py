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

from . import continuous
from . import discrete

# Continuous stochastic systems
from .continuous import (
    BrownianMotion,
    BrownianMotion2D,
    GeometricBrownianMotion,
    BrownianMotionWithDrift,
    GBM,
    OrnsteinUhlenbeck,
    MultivariateOrnsteinUhlenbeck,
    OU,
    CoxIngersollRoss,
    CIR,
    LangevinDynamics,
    StochasticDoubleIntegrator,
    ContinuousStochasticPendulum,
    StochasticCartPole,
    StochasticLorenz,
    StochasticSIR,
    SIR,
    ContinuousStochasticBatchReactor,
    ContinuousStochasticCSTR,
)

# Discrete stochastic systems
from .discrete import (
    DiscreteAR1,
    AR1,
    DiscreteWhiteNoise,
    WhiteNoise,
    create_standard_white_noise,
    create_measurement_noise,
    create_thermal_noise,
    DiscreteRandomWalk,
    RandomWalk,
    create_symmetric_random_walk,
    create_biased_random_walk,
    DiscreteARMA11,
    ARMA11,
    create_economic_arma,
    create_sensor_arma,
    DiscreteVAR1,
    VAR1,
    create_bivariate_var,
    create_macro_var,
    DiscreteGARCH11,
    GARCH11,
    create_equity_garch,
    create_fx_garch,
    DiscreteStochasticDoubleIntegrator,
    create_digital_servo,
    create_lqg_benchmark_discrete,
    create_spacecraft_discrete,
    DiscreteStochasticPendulum,
    create_furuta_pendulum,
    create_rl_pendulum,
    DiscreteStochasticQueue,
    StochasticQueue,
    create_call_center_queue,
    create_network_queue,
    DiscreteStochasticBatchReactor,
    DiscreteStochasticCSTR,
    create_discrete_batch_reactor_with_noise,
    create_discrete_stochastic_cstr_with_noise,
    DiscreteStochasticLogisticMap,
    LogisticMap,
    create_fixed_point_regime,
    create_chaotic_regime,
    create_edge_of_chaos,
)

__all__ = [
    # Submodules
    "continuous",
    "discrete",
    # Continuous: Brownian motion
    "BrownianMotion",
    "BrownianMotion2D",
    "GeometricBrownianMotion",
    "BrownianMotionWithDrift",
    "GBM",
    # Continuous: Ornstein-Uhlenbeck process
    "OrnsteinUhlenbeck",
    "MultivariateOrnsteinUhlenbeck",
    "OU",
    # Continuous: Cox-Ingersoll-Ross process
    "CoxIngersollRoss",
    "CIR",
    # Continuous: Langevin dynamics
    "LangevinDynamics",
    # Continuous: Stochastic control systems
    "StochasticDoubleIntegrator",
    "ContinuousStochasticPendulum",
    "StochasticCartPole",
    "StochasticLorenz",
    # Continuous: Epidemiological models
    "StochasticSIR",
    "SIR",
    # Continuous: Reactor systems
    "ContinuousStochasticBatchReactor",
    "ContinuousStochasticCSTR",
    # Discrete: Time series models
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
    # Discrete: Multivariate models
    "DiscreteVAR1",
    "VAR1",
    "create_bivariate_var",
    "create_macro_var",
    # Discrete: Financial models
    "DiscreteGARCH11",
    "GARCH11",
    "create_equity_garch",
    "create_fx_garch",
    # Discrete: Control systems
    "DiscreteStochasticDoubleIntegrator",
    "create_digital_servo",
    "create_lqg_benchmark_discrete",
    "create_spacecraft_discrete",
    "DiscreteStochasticPendulum",
    "create_furuta_pendulum",
    "create_rl_pendulum",
    # Discrete: Queueing systems
    "DiscreteStochasticQueue",
    "StochasticQueue",
    "create_call_center_queue",
    "create_network_queue",
    # Discrete: Reactor systems
    "DiscreteStochasticBatchReactor",
    "DiscreteStochasticCSTR",
    "create_discrete_batch_reactor_with_noise",
    "create_discrete_stochastic_cstr_with_noise",
    # Discrete: Nonlinear dynamics
    "DiscreteStochasticLogisticMap",
    "LogisticMap",
    "create_fixed_point_regime",
    "create_chaotic_regime",
    "create_edge_of_chaos",
]
