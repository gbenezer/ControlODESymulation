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
Type System for Control Differential Equation Simulation Library

This package provides comprehensive type definitions for:

1. Core Types (vectors, matrices, dimensions)
   - StateVector, ControlVector, OutputVector
   - StateMatrix, InputMatrix, OutputMatrix, etc.
   - SystemDimensions, ArrayLike

2. Backend Support (NumPy, PyTorch, JAX)
   - Backend type literal
   - BackendConfig TypedDict

3. Trajectories and Simulation
   - TimePoints, StateTrajectory, ControlSequence
   - SimulationResult

4. Linearization
   - LinearizationResult, DeterministicLinearization
   - StochasticLinearization

5. Control Design
   - Classical: LQRResult, LQGResult, KalmanFilterResult
   - Advanced: MPCResult, MHEResult, H2ControlResult, HInfControlResult

6. State Estimation
   - EKFResult, UKFResult, ParticleFilterResult

7. System Identification
   - SystemIDResult, SubspaceIDResult
   - ERAResult, DMDResult, SINDyResult

8. Symbolic Computation
   - SymbolicExpression, SymbolicMatrix
   - SymbolicStateEquations, SymbolicJacobian

9. Reachability and Safety
   - ReachabilityResult, ROAResult
   - CBFResult, CLFResult, BarrierCertificateResult

10. Robustness Analysis
    - RobustStabilityResult, TubeMPCResult
    - StochasticMPCResult, RiskSensitiveResult

11. Optimization
    - OptimizationResult, TrajectoryOptimizationResult
    - ConvexOptimizationResult

12. Machine Learning
    - TrainingResult, RLTrainingResult
    - ImitationLearningResult

13. Conformal Prediction
    - ConformalPredictionResult, AdaptiveConformalResult

14. Contraction Theory
    - ContractionAnalysisResult, CCMResult
    - FunnelingResult

15. Differential Flatness
    - DifferentialFlatnessResult, TrajectoryPlanningResult

16. Model Reduction
    - BalancedRealizationResult, ReducedOrderModelResult

17. Utilities
    - Type guards, converters, validators
    - Protocols for structural subtyping

Usage
-----
>>> from src.types import (
...     StateVector, ControlVector, StateMatrix,
...     LQRResult, MPCResult, SimulationResult,
... )
>>>
>>> # Type-annotated function
>>> def simulate(x0: StateVector, u: ControlVector) -> SimulationResult:
...     ...

Import Patterns
--------------
# Import specific types
from src.types import StateVector, LQRResult

# Import from specific modules
from src.types.core import StateVector, SystemDimensions
from src.types.control_classical import LQRResult

# Import entire module
from src.types import core, estimation
"""

# =============================================================================
# Core Types - Fundamental vectors and matrices
# =============================================================================
# =============================================================================
# Backend Support - Multi-backend array operations
# =============================================================================
from .backends import (
    Backend,
    BackendConfig,
    Device,
    DiscretizationMethod,
    IntegrationMethod,
    NoiseType,
    OptimizationMethod,
    SDEIntegrationMethod,
    SDEType,
)

# =============================================================================
# Conformal Prediction - Distribution-free uncertainty quantification
# =============================================================================
from .conformal import (
    AdaptiveConformalResult,
    ConformalCalibrationResult,
    ConformalPredictionResult,
    ConformalPredictionSet,
    NonconformityScore,
)

# =============================================================================
# Contraction Theory - Contraction analysis and CCM
# =============================================================================
from .contraction import (
    CCMResult,
    ContractionAnalysisResult,
    ContractionMetric,
    ContractionRate,
    FunnelingResult,
    IncrementalStabilityResult,
)

# =============================================================================
# Advanced Control - MPC, MHE, H2, H-infinity
# =============================================================================
from .control_advanced import (
    AdaptiveControlResult,
    H2ControlResult,
    HInfControlResult,
    LMIResult,
    MHEResult,
    MPCResult,
    SlidingModeResult,
)

# =============================================================================
# Classical Control - LQR, LQG, Kalman, pole placement
# =============================================================================
from .control_classical import (
    ControllabilityInfo,
    KalmanFilterResult,
    LQGResult,
    LQRResult,
    LuenbergerObserverResult,
    ObservabilityInfo,
    PolePlacementResult,
    StabilityInfo,
)
from .core import (
    ArrayLike,
    ControlVector,
    CovarianceMatrix,
    FeedthroughMatrix,
    GainMatrix,
    InputMatrix,
    OutputMatrix,
    OutputVector,
    ParameterVector,
    StateMatrix,
    StateVector,
    SystemDimensions,
)

# =============================================================================
# State Estimation - EKF, UKF, Particle filters
# =============================================================================
from .estimation import (
    EKFResult,
    ParticleFilterResult,
    UKFResult,
)

# =============================================================================
# Differential Flatness - Flatness analysis and trajectory planning
# =============================================================================
from .flatness import (
    DifferentialFlatnessResult,
    FlatnessOutput,
    TrajectoryPlanningResult,
)

# =============================================================================
# System Identification - Data-driven model estimation
# =============================================================================
from .identification import (
    DMDResult,
    ERAResult,
    HankelMatrix,
    KoopmanResult,
    MarkovParameters,
    SINDyResult,
    SubspaceIDResult,
    SystemIDResult,
    ToeplitzMatrix,
    TrajectoryMatrix,
)

# =============================================================================
# Learning - Neural networks and reinforcement learning
# =============================================================================
from .learning import (
    Dataset,
    ImitationLearningResult,
    LearningRate,
    LossValue,
    NeuralDynamicsResult,
    NeuralNetworkConfig,
    OnlineAdaptationResult,
    PolicyEvaluationResult,
    RLTrainingResult,
    TrainingBatch,
    TrainingResult,
)

# =============================================================================
# Linearization - Jacobian computation and linearization results
# =============================================================================
from .linearization import (
    ContinuousLinearization,
    ContinuousStochasticLinearization,
    ControlJacobian,
    DeterministicLinearization,
    DiffusionJacobian,
    DiscreteLinearization,
    DiscreteStochasticLinearization,
    FullLinearization,
    FullStochasticLinearization,
    LinearizationCacheKey,
    LinearizationResult,
    ObservationLinearization,
    OutputJacobian,
    StateJacobian,
    StochasticLinearization,
)

# =============================================================================
# Model Reduction - Balanced realization and order reduction
# =============================================================================
from .model_reduction import (
    BalancedRealizationResult,
    ReducedOrderModelResult,
)

# =============================================================================
# Optimization - General and trajectory optimization
# =============================================================================
from .optimization import (
    ConstrainedOptimizationResult,
    ConvexOptimizationResult,
    OptimizationBounds,
    OptimizationResult,
    ParameterOptimizationResult,
    TrajectoryOptimizationResult,
)

# =============================================================================
# Reachability and Safety - Set-based analysis and verification
# =============================================================================
from .reachability import (
    BarrierCertificateResult,
    CBFResult,
    CLFResult,
    ReachabilityResult,
    ReachableSet,
    ROAResult,
    SafeSet,
    VerificationResult,
)

# =============================================================================
# Robustness - Uncertainty handling and robust control
# =============================================================================
from .robustness import (
    RiskSensitiveResult,
    RobustStabilityResult,
    StochasticMPCResult,
    StructuredSingularValueResult,
    TubeDefinition,
    TubeMPCResult,
    UncertaintySet,
)

# =============================================================================
# Symbolic Computation - Symbolic expressions and systems
# =============================================================================
from .symbolic import (
    ParameterDict,
    SubstitutionDict,
    SymbolDict,
    SymbolicDiffusionMatrix,
    SymbolicExpression,
    SymbolicGradient,
    SymbolicHessian,
    SymbolicJacobian,
    SymbolicMatrix,
    SymbolicOutputEquations,
    SymbolicStateEquations,
    SymbolicSymbol,
)

# =============================================================================
# Trajectories - Time series and simulation results
# =============================================================================
from .trajectories import (
    ControlSequence,
    IntegrationResult,
    NoiseSequence,
    OutputSequence,
    SimulationResult,
    StateTrajectory,
    TimePoints,
    TimeSpan,
    TrajectorySegment,
    TrajectoryStatistics,
)

# =============================================================================
# Utilities - Type guards, converters, protocols
# =============================================================================
from .utilities import (
    ArrayConverter,
    CacheKey,
    CacheStatistics,
    LinearizableProtocol,
    Metadata,
    PerformanceMetrics,
    SimulatableProtocol,
    StochasticProtocol,
    ValidationResult,
    check_control_shape,
    check_state_shape,
    ensure_backend,
    ensure_numpy,
    extract_dimensions,
    get_array_shape,
    get_backend,
    get_batch_size,
    is_batched,
    is_jax,
    is_numpy,
    is_torch,
)

# =============================================================================
# Public API - All exported symbols
# =============================================================================
__all__ = [
    # -------------------------------------------------------------------------
    # Core Types
    # -------------------------------------------------------------------------
    "ArrayLike",
    "StateVector",
    "ControlVector",
    "OutputVector",
    "ParameterVector",
    "StateMatrix",
    "InputMatrix",
    "OutputMatrix",
    "FeedthroughMatrix",
    "GainMatrix",
    "CovarianceMatrix",
    "SystemDimensions",
    # -------------------------------------------------------------------------
    # Backend Support
    # -------------------------------------------------------------------------
    "Backend",
    "Device",
    "BackendConfig",
    "IntegrationMethod",
    "DiscretizationMethod",
    "SDEIntegrationMethod",
    "OptimizationMethod",
    "NoiseType",
    "SDEType",
    # -------------------------------------------------------------------------
    # Trajectories
    # -------------------------------------------------------------------------
    "TimePoints",
    "TimeSpan",
    "StateTrajectory",
    "ControlSequence",
    "OutputSequence",
    "NoiseSequence",
    "IntegrationResult",
    "SimulationResult",
    "TrajectoryStatistics",
    "TrajectorySegment",
    # -------------------------------------------------------------------------
    # Linearization
    # -------------------------------------------------------------------------
    "DeterministicLinearization",
    "StochasticLinearization",
    "LinearizationResult",
    "ObservationLinearization",
    "ContinuousLinearization",
    "DiscreteLinearization",
    "ContinuousStochasticLinearization",
    "DiscreteStochasticLinearization",
    "FullLinearization",
    "FullStochasticLinearization",
    "StateJacobian",
    "ControlJacobian",
    "OutputJacobian",
    "DiffusionJacobian",
    "LinearizationCacheKey",
    # -------------------------------------------------------------------------
    # Classical Control
    # -------------------------------------------------------------------------
    "StabilityInfo",
    "ControllabilityInfo",
    "ObservabilityInfo",
    "LQRResult",
    "KalmanFilterResult",
    "LQGResult",
    "PolePlacementResult",
    "LuenbergerObserverResult",
    # -------------------------------------------------------------------------
    # Advanced Control
    # -------------------------------------------------------------------------
    "MPCResult",
    "MHEResult",
    "H2ControlResult",
    "HInfControlResult",
    "LMIResult",
    "AdaptiveControlResult",
    "SlidingModeResult",
    # -------------------------------------------------------------------------
    # State Estimation
    # -------------------------------------------------------------------------
    "EKFResult",
    "UKFResult",
    "ParticleFilterResult",
    # -------------------------------------------------------------------------
    # System Identification
    # -------------------------------------------------------------------------
    "HankelMatrix",
    "ToeplitzMatrix",
    "TrajectoryMatrix",
    "MarkovParameters",
    "SystemIDResult",
    "SubspaceIDResult",
    "ERAResult",
    "DMDResult",
    "SINDyResult",
    "KoopmanResult",
    # -------------------------------------------------------------------------
    # Symbolic Computation
    # -------------------------------------------------------------------------
    "SymbolicExpression",
    "SymbolicMatrix",
    "SymbolicSymbol",
    "SymbolDict",
    "SymbolicStateEquations",
    "SymbolicOutputEquations",
    "SymbolicDiffusionMatrix",
    "ParameterDict",
    "SubstitutionDict",
    "SymbolicJacobian",
    "SymbolicGradient",
    "SymbolicHessian",
    # -------------------------------------------------------------------------
    # Reachability and Safety
    # -------------------------------------------------------------------------
    "ReachableSet",
    "SafeSet",
    "ReachabilityResult",
    "ROAResult",
    "VerificationResult",
    "BarrierCertificateResult",
    "CBFResult",
    "CLFResult",
    # -------------------------------------------------------------------------
    # Robustness
    # -------------------------------------------------------------------------
    "UncertaintySet",
    "RobustStabilityResult",
    "StructuredSingularValueResult",
    "TubeDefinition",
    "TubeMPCResult",
    "StochasticMPCResult",
    "RiskSensitiveResult",
    # -------------------------------------------------------------------------
    # Optimization
    # -------------------------------------------------------------------------
    "OptimizationBounds",
    "OptimizationResult",
    "ConstrainedOptimizationResult",
    "TrajectoryOptimizationResult",
    "ConvexOptimizationResult",
    "ParameterOptimizationResult",
    # -------------------------------------------------------------------------
    # Learning
    # -------------------------------------------------------------------------
    "Dataset",
    "TrainingBatch",
    "LearningRate",
    "LossValue",
    "NeuralNetworkConfig",
    "TrainingResult",
    "NeuralDynamicsResult",
    "RLTrainingResult",
    "PolicyEvaluationResult",
    "ImitationLearningResult",
    "OnlineAdaptationResult",
    # -------------------------------------------------------------------------
    # Conformal Prediction
    # -------------------------------------------------------------------------
    "ConformalPredictionSet",
    "NonconformityScore",
    "ConformalCalibrationResult",
    "ConformalPredictionResult",
    "AdaptiveConformalResult",
    # -------------------------------------------------------------------------
    # Contraction Theory
    # -------------------------------------------------------------------------
    "ContractionMetric",
    "ContractionRate",
    "ContractionAnalysisResult",
    "CCMResult",
    "FunnelingResult",
    "IncrementalStabilityResult",
    # -------------------------------------------------------------------------
    # Differential Flatness
    # -------------------------------------------------------------------------
    "FlatnessOutput",
    "DifferentialFlatnessResult",
    "TrajectoryPlanningResult",
    # -------------------------------------------------------------------------
    # Model Reduction
    # -------------------------------------------------------------------------
    "BalancedRealizationResult",
    "ReducedOrderModelResult",
    # -------------------------------------------------------------------------
    # Utilities - Protocols
    # -------------------------------------------------------------------------
    "LinearizableProtocol",
    "SimulatableProtocol",
    "StochasticProtocol",
    # -------------------------------------------------------------------------
    # Utilities - Type Guards
    # -------------------------------------------------------------------------
    "is_batched",
    "get_batch_size",
    "is_numpy",
    "is_torch",
    "is_jax",
    "get_backend",
    # -------------------------------------------------------------------------
    # Utilities - Converters
    # -------------------------------------------------------------------------
    "ensure_numpy",
    "ensure_backend",
    "ArrayConverter",
    # -------------------------------------------------------------------------
    # Utilities - Validators
    # -------------------------------------------------------------------------
    "check_state_shape",
    "check_control_shape",
    "get_array_shape",
    "extract_dimensions",
    # -------------------------------------------------------------------------
    # Utilities - Cache and Metadata
    # -------------------------------------------------------------------------
    "CacheKey",
    "CacheStatistics",
    "Metadata",
    # -------------------------------------------------------------------------
    # Utilities - Validation and Performance
    # -------------------------------------------------------------------------
    "ValidationResult",
    "PerformanceMetrics",
]
