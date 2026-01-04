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

from . import deterministic
from . import stochastic

# NOTE: LogisticMap name collision
# Both deterministic.discrete and stochastic.discrete define LogisticMap:
#   - deterministic.LogisticMap: Classic chaotic map x_{k+1} = r * x_k * (1 - x_k)
#   - stochastic.LogisticMap: Alias for DiscreteStochasticLogisticMap (with noise)
# Only the deterministic LogisticMap is exported at this level.
# Access the stochastic version explicitly:
#   - from src.systems.builtin.stochastic import LogisticMap
#   - from src.systems.builtin.stochastic.discrete import DiscreteStochasticLogisticMap

# Deterministic systems
from .deterministic import (
    # Continuous - Linear systems
    LinearSystem,
    AutonomousLinearSystem,
    LinearSystem2D,
    # Continuous - Pendulum
    SymbolicPendulum,
    SymbolicPendulum2ndOrder,
    PendulumSystem,
    # Continuous - Cart-pole
    CartPole,
    CartPoleSystem,
    # Continuous - Aerial systems
    SymbolicQuadrotor2D,
    SymbolicQuadrotor2DLidar,
    PVTOL,
    # Continuous - Vehicles
    DubinsVehicle,
    PathTracking,
    # Continuous - Oscillators
    CoupledOscillatorSystem,
    NonlinearChainSystem,
    VanDerPolOscillator,
    ControlledVanDerPolOscillator,
    DuffingOscillator,
    # Continuous - Chaotic systems
    Lorenz,
    ControlledLorenz,
    # Continuous - Mechanical systems
    FifthOrderMechanicalSystem,
    Manipulator2Link,
    # Continuous - Reactor systems
    ContinuousBatchReactor,
    ContinuousCSTR,
    # Discrete - Mechanical systems
    DiscreteOscillator,
    DiscretePendulum,
    DiscreteDoubleIntegrator,
    DiscreteDoubleIntegratorWithForce,
    DiscreteRobotArm,
    DiscreteCartPole,
    # Discrete - Mobile robots
    DifferentialDriveRobot,
    # Discrete - Reactor systems
    DiscreteBatchReactor,
    DiscreteCSTR,
    # Discrete - Economic models
    DiscreteSolowModel,
    # Discrete - Chaotic maps
    LogisticMap,
    HenonMap,
    StandardMap,
)

# Stochastic systems
from .stochastic import (
    # Continuous - Brownian motion
    BrownianMotion,
    BrownianMotion2D,
    GeometricBrownianMotion,
    BrownianMotionWithDrift,
    GBM,
    # Continuous - Ornstein-Uhlenbeck process
    OrnsteinUhlenbeck,
    MultivariateOrnsteinUhlenbeck,
    OU,
    # Continuous - Cox-Ingersoll-Ross process
    CoxIngersollRoss,
    CIR,
    # Continuous - Langevin dynamics
    LangevinDynamics,
    # Continuous - Stochastic control systems
    StochasticDoubleIntegrator,
    ContinuousStochasticPendulum,
    StochasticCartPole,
    StochasticLorenz,
    # Continuous - Epidemiological models
    StochasticSIR,
    SIR,
    # Continuous - Reactor systems
    ContinuousStochasticBatchReactor,
    ContinuousStochasticCSTR,
    # Discrete - Time series models
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
    # Discrete - Multivariate models
    DiscreteVAR1,
    VAR1,
    create_bivariate_var,
    create_macro_var,
    # Discrete - Financial models
    DiscreteGARCH11,
    GARCH11,
    create_equity_garch,
    create_fx_garch,
    # Discrete - Stochastic control systems
    DiscreteStochasticDoubleIntegrator,
    create_digital_servo,
    create_lqg_benchmark_discrete,
    create_spacecraft_discrete,
    DiscreteStochasticPendulum,
    create_furuta_pendulum,
    create_rl_pendulum,
    # Discrete - Queueing systems
    DiscreteStochasticQueue,
    StochasticQueue,
    create_call_center_queue,
    create_network_queue,
    # Discrete - Reactor systems
    DiscreteStochasticBatchReactor,
    DiscreteStochasticCSTR,
    create_discrete_batch_reactor_with_noise,
    create_discrete_stochastic_cstr_with_noise,
    # Discrete - Nonlinear dynamics
    DiscreteStochasticLogisticMap,
    create_fixed_point_regime,
    create_chaotic_regime,
    create_edge_of_chaos,
)

__all__ = [
    # Submodules
    "deterministic",
    "stochastic",
    # Deterministic Continuous - Linear systems
    "LinearSystem",
    "AutonomousLinearSystem",
    "LinearSystem2D",
    # Deterministic Continuous - Pendulum
    "SymbolicPendulum",
    "SymbolicPendulum2ndOrder",
    "PendulumSystem",
    # Deterministic Continuous - Cart-pole
    "CartPole",
    "CartPoleSystem",
    # Deterministic Continuous - Aerial systems
    "SymbolicQuadrotor2D",
    "SymbolicQuadrotor2DLidar",
    "PVTOL",
    # Deterministic Continuous - Vehicles
    "DubinsVehicle",
    "PathTracking",
    # Deterministic Continuous - Oscillators
    "CoupledOscillatorSystem",
    "NonlinearChainSystem",
    "VanDerPolOscillator",
    "ControlledVanDerPolOscillator",
    "DuffingOscillator",
    # Deterministic Continuous - Chaotic systems
    "Lorenz",
    "ControlledLorenz",
    # Deterministic Continuous - Mechanical systems
    "FifthOrderMechanicalSystem",
    "Manipulator2Link",
    # Deterministic Continuous - Reactor systems
    "ContinuousBatchReactor",
    "ContinuousCSTR",
    # Deterministic Discrete - Mechanical systems
    "DiscreteOscillator",
    "DiscretePendulum",
    "DiscreteDoubleIntegrator",
    "DiscreteDoubleIntegratorWithForce",
    "DiscreteRobotArm",
    "DiscreteCartPole",
    # Deterministic Discrete - Mobile robots
    "DifferentialDriveRobot",
    # Deterministic Discrete - Reactor systems
    "DiscreteBatchReactor",
    "DiscreteCSTR",
    # Deterministic Discrete - Economic models
    "DiscreteSolowModel",
    # Deterministic Discrete - Chaotic maps
    "LogisticMap",
    "HenonMap",
    "StandardMap",
    # Stochastic Continuous - Brownian motion
    "BrownianMotion",
    "BrownianMotion2D",
    "GeometricBrownianMotion",
    "BrownianMotionWithDrift",
    "GBM",
    # Stochastic Continuous - Ornstein-Uhlenbeck process
    "OrnsteinUhlenbeck",
    "MultivariateOrnsteinUhlenbeck",
    "OU",
    # Stochastic Continuous - Cox-Ingersoll-Ross process
    "CoxIngersollRoss",
    "CIR",
    # Stochastic Continuous - Langevin dynamics
    "LangevinDynamics",
    # Stochastic Continuous - Control systems
    "StochasticDoubleIntegrator",
    "ContinuousStochasticPendulum",
    "StochasticCartPole",
    "StochasticLorenz",
    # Stochastic Continuous - Epidemiological models
    "StochasticSIR",
    "SIR",
    # Stochastic Continuous - Reactor systems
    "ContinuousStochasticBatchReactor",
    "ContinuousStochasticCSTR",
    # Stochastic Discrete - Time series models
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
    # Stochastic Discrete - Multivariate models
    "DiscreteVAR1",
    "VAR1",
    "create_bivariate_var",
    "create_macro_var",
    # Stochastic Discrete - Financial models
    "DiscreteGARCH11",
    "GARCH11",
    "create_equity_garch",
    "create_fx_garch",
    # Stochastic Discrete - Control systems
    "DiscreteStochasticDoubleIntegrator",
    "create_digital_servo",
    "create_lqg_benchmark_discrete",
    "create_spacecraft_discrete",
    "DiscreteStochasticPendulum",
    "create_furuta_pendulum",
    "create_rl_pendulum",
    # Stochastic Discrete - Queueing systems
    "DiscreteStochasticQueue",
    "StochasticQueue",
    "create_call_center_queue",
    "create_network_queue",
    # Stochastic Discrete - Reactor systems
    "DiscreteStochasticBatchReactor",
    "DiscreteStochasticCSTR",
    "create_discrete_batch_reactor_with_noise",
    "create_discrete_stochastic_cstr_with_noise",
    # Stochastic Discrete - Nonlinear dynamics
    "DiscreteStochasticLogisticMap",
    "create_fixed_point_regime",
    "create_chaotic_regime",
    "create_edge_of_chaos",
]
