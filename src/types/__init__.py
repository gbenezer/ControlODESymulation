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

from typing import Dict, List, Optional, Any

"""
Types Module - Comprehensive Type Definitions for ControlDESymulation

Central import point for all type definitions across the framework.
Organized into domain-specific modules but re-exported here for convenience.

Usage
-----
>>> from src.systems.base.types import (
...     StateVector,
...     ControlVector,
...     LQRResult,
...     HankelMatrix,
...     Backend,
... )

Module Organization
------------------
- core: Basic arrays, vectors, matrices
- linearization: Linearization results
- trajectories: Time series and sequences
- symbolic: SymPy type definitions
- backends: Backend and device types
- control_classical: LQR, LQG, Kalman
- control_advanced: MPC, MHE, H∞, Adaptive
- estimation: EKF, UKF, Particle, Conformal
- identification: Hankel, ERA, DMD, SINDy, N4SID
- reachability: HJI, Safety, Contraction, Flatness
- robustness: Uncertainty, μ-analysis, Tubes
- utilities: Type guards, converters

Statistics
----------
- 170+ type definitions
- 40+ TypedDict result types
- 15+ utility functions
- 8+ Protocol definitions
"""

# ============================================================================
# Core Types (Basic Building Blocks)
# ============================================================================

from .core import (
    # Basic arrays
    ArrayLike,
    NumpyArray,
    TorchTensor,
    JaxArray,
    ScalarLike,
    IntegerLike,
    
    # Vectors
    StateVector,
    ControlVector,
    OutputVector,
    NoiseVector,
    ParameterVector,
    ResidualVector,
    
    # Matrices
    StateMatrix,
    ControlMatrix,
    OutputMatrix,
    DiffusionMatrix,
    FeedthroughMatrix,
    CovarianceMatrix,
    GainMatrix,
    ControllabilityMatrix,
    ObservabilityMatrix,
    CostMatrix,
    
    # Dimensions
    SystemDimensions,
    DimensionTuple,
    
    # Equilibria
    EquilibriumState,
    EquilibriumControl,
    EquilibriumPoint,
    EquilibriumName,
    EquilibriumIdentifier,
)


# ============================================================================
# Linearization Types
# ============================================================================

from .linearization import (
    DeterministicLinearization,
    StochasticLinearization,
    LinearizationResult,
    ObservationLinearization,
)


# ============================================================================
# Trajectory and Sequence Types
# ============================================================================

from .trajectories import (
    StateTrajectory,
    ControlSequence,
    OutputSequence,
    NoiseSequence,
    TimePoints,
    TimeSpan,
)


# ============================================================================
# Symbolic Types
# ============================================================================

from .symbolic import (
    SymbolicExpression,
    SymbolicMatrix,
    SymbolicSymbol,
    SymbolDict,
    SymbolicStateEquations,
    SymbolicOutputEquations,
    SymbolicDiffusionMatrix,
)


# ============================================================================
# Backend Types
# ============================================================================

from .backends import (
    Backend,
    Device,
    BackendConfig,
    IntegrationMethod,
    DiscretizationMethod,
    SDEIntegrationMethod,
    OptimizationMethod,
    NoiseType,
    SDEType,
    ConvergenceType,
    VALID_BACKENDS,
    VALID_DEVICES,
    DEFAULT_BACKEND,
    DEFAULT_DEVICE,
    DEFAULT_DTYPE,
)


# ============================================================================
# Classical Control and Estimation
# ============================================================================

from .control_classical import (
    LQRResult,
    LQGResult,
    KalmanFilterResult,
    StabilityInfo,
    ControllabilityInfo,
    ObservabilityInfo,
)


# ============================================================================
# Advanced Control
# ============================================================================

from .control_advanced import (
    MPCResult,
    MHEResult,
    H2ControlResult,
    HInfControlResult,
    LMIResult,
    AdaptiveControlResult,
    SlidingModeResult,
    StochasticMPCResult,
    RiskSensitiveResult,
)


# ============================================================================
# State Estimation (Beyond Kalman)
# ============================================================================

from .estimation import (
    EKFResult,
    UKFResult,
    ParticleFilterResult,
    ConformalPredictionSet,
    NonconformityScore,
    ConformalCalibrationResult,
    ConformalPredictionResult,
    AdaptiveConformalResult,
)


# ============================================================================
# System Identification
# ============================================================================

from .identification import (
    HankelMatrix,
    ToeplitzMatrix,
    TrajectoryMatrix,
    MarkovParameters,
    CorrelationMatrix,
    SystemIDResult,
    SubspaceIDResult,
    ERAResult,
    DMDResult,
    SINDyResult,
    KoopmanResult,
)


# ============================================================================
# Reachability, Safety, and Contraction
# ============================================================================

from .reachability import (
    # Sets
    ReachableSet,
    SafeSet,
    ValueFunction,
    LevelSet,
    
    # Reachability results
    ReachabilityResult,
    HJIReachabilityResult,
    BackwardReachableResult,
    ForwardReachableResult,
    ViabilityKernelResult,
    DifferentialGameResult,
    ROAResult,
    
    # Safety
    VerificationResult,
    BarrierCertificateResult,
    CBFResult,
    CLFResult,
    
    # Contraction
    ContractionMetric,
    ContractionRate,
    ContractionAnalysisResult,
    CCMResult,
    FunnelingResult,
    IncrementalStabilityResult,
    
    # Flatness
    FlatnessOutput,
    DifferentialFlatnessResult,
    TrajectoryPlanningResult,
)


# ============================================================================
# Robustness and Uncertainty
# ============================================================================

from .robustness import (
    UncertaintySet,
    RobustStabilityResult,
    StructuredSingularValueResult,
    TubeDefinition,
    TubeMPCResult,
    BalancedRealizationResult,
    ReducedOrderModelResult,
)


# ============================================================================
# Utility Functions and Type Guards
# ============================================================================

from .utilities import (
    # Type guards
    is_batched,
    get_batch_size,
    is_numpy,
    is_torch,
    is_jax,
    get_backend,
    
    # Converters
    ensure_numpy,
    ensure_backend,
    ArrayConverter,
    
    # Validators
    check_state_shape,
    check_control_shape,
    get_array_shape,
    extract_dimensions,
    
    # Protocols
    LinearizableProtocol,
    SimulatableProtocol,
    StochasticProtocol,
)


# ============================================================================
# Simulation and Integration Results
# ============================================================================

from .trajectories import (
    IntegrationResult,
    SimulationResult,
)


# ============================================================================
# Learning Types
# ============================================================================

from .identification import (
    Dataset,
    TrainingBatch,
    LearningRate,
    LossValue,
    NeuralNetworkConfig,
    TrainingResult,
    NeuralNetworkDynamicsResult,
    GPDynamicsResult,
)


# ============================================================================
# Optimization Types
# ============================================================================

from .control_advanced import (
    OptimizationBounds,
    OptimizationResult,
)


# ============================================================================
# Cache and Metadata
# ============================================================================

from .utilities import (
    CacheKey,
    CacheStatistics,
    Metadata,
)


# ============================================================================
# Function Types
# ============================================================================

from .core import (
    DynamicsFunction,
    OutputFunction,
    DiffusionFunction,
    ControlPolicy,
    StateEstimator,
    CostFunction,
    Constraint,
    IntegrationCallback,
    SimulationCallback,
    OptimizationCallback,
)


# ============================================================================
# Configuration Types
# ============================================================================

from .backends import (
    SystemConfig,
    IntegratorConfig,
    DiscretizerConfig,
)


# ============================================================================
# Performance and Validation
# ============================================================================

from .utilities import (
    ValidationResult,
    PerformanceMetrics,
)


# ============================================================================
# Type Variables
# ============================================================================

from .core import (
    T,
    S,
    C,
    MatrixT,
)


# ============================================================================
# Comprehensive Export List
# ============================================================================

__all__ = [
    # ========================================================================
    # Core Types
    # ========================================================================
    
    # Basic arrays
    'ArrayLike',
    'NumpyArray',
    'TorchTensor',
    'JaxArray',
    'ScalarLike',
    'IntegerLike',
    
    # Vectors
    'StateVector',
    'ControlVector',
    'OutputVector',
    'NoiseVector',
    'ParameterVector',
    'ResidualVector',
    
    # Matrices
    'StateMatrix',
    'ControlMatrix',
    'OutputMatrix',
    'DiffusionMatrix',
    'FeedthroughMatrix',
    'CovarianceMatrix',
    'GainMatrix',
    'ControllabilityMatrix',
    'ObservabilityMatrix',
    'CostMatrix',
    
    # Dimensions
    'SystemDimensions',
    'DimensionTuple',
    
    # Equilibria
    'EquilibriumState',
    'EquilibriumControl',
    'EquilibriumPoint',
    'EquilibriumName',
    'EquilibriumIdentifier',
    
    # ========================================================================
    # Linearization
    # ========================================================================
    
    'DeterministicLinearization',
    'StochasticLinearization',
    'LinearizationResult',
    'ObservationLinearization',
    
    # ========================================================================
    # Trajectories and Sequences
    # ========================================================================
    
    'StateTrajectory',
    'ControlSequence',
    'OutputSequence',
    'NoiseSequence',
    'TimePoints',
    'TimeSpan',
    'IntegrationResult',
    'SimulationResult',
    
    # ========================================================================
    # Symbolic
    # ========================================================================
    
    'SymbolicExpression',
    'SymbolicMatrix',
    'SymbolicSymbol',
    'SymbolDict',
    'SymbolicStateEquations',
    'SymbolicOutputEquations',
    'SymbolicDiffusionMatrix',
    
    # ========================================================================
    # Backend and Methods
    # ========================================================================
    
    'Backend',
    'Device',
    'BackendConfig',
    'IntegrationMethod',
    'DiscretizationMethod',
    'SDEIntegrationMethod',
    'OptimizationMethod',
    'NoiseType',
    'SDEType',
    'ConvergenceType',
    
    # ========================================================================
    # Classical Control and Estimation
    # ========================================================================
    
    'LQRResult',
    'LQGResult',
    'KalmanFilterResult',
    'StabilityInfo',
    'ControllabilityInfo',
    'ObservabilityInfo',
    
    # ========================================================================
    # Advanced Control
    # ========================================================================
    
    'MPCResult',
    'MHEResult',
    'H2ControlResult',
    'HInfControlResult',
    'LMIResult',
    'AdaptiveControlResult',
    'SlidingModeResult',
    'StochasticMPCResult',
    'RiskSensitiveResult',
    'OptimizationBounds',
    'OptimizationResult',
    
    # ========================================================================
    # State Estimation
    # ========================================================================
    
    'EKFResult',
    'UKFResult',
    'ParticleFilterResult',
    
    # ========================================================================
    # Conformal Prediction
    # ========================================================================
    
    'ConformalPredictionSet',
    'NonconformityScore',
    'ConformalCalibrationResult',
    'ConformalPredictionResult',
    'AdaptiveConformalResult',
    
    # ========================================================================
    # System Identification
    # ========================================================================
    
    'HankelMatrix',
    'ToeplitzMatrix',
    'TrajectoryMatrix',
    'MarkovParameters',
    'CorrelationMatrix',
    'SystemIDResult',
    'SubspaceIDResult',
    'ERAResult',
    'DMDResult',
    'SINDyResult',
    'KoopmanResult',
    'Dataset',
    'TrainingBatch',
    'LearningRate',
    'LossValue',
    'NeuralNetworkConfig',
    'TrainingResult',
    'NeuralNetworkDynamicsResult',
    'GPDynamicsResult',
    
    # ========================================================================
    # Reachability and Safety
    # ========================================================================
    
    'ReachableSet',
    'SafeSet',
    'ValueFunction',
    'LevelSet',
    'ReachabilityResult',
    'HJIReachabilityResult',
    'BackwardReachableResult',
    'ForwardReachableResult',
    'ViabilityKernelResult',
    'DifferentialGameResult',
    'ROAResult',
    'VerificationResult',
    'BarrierCertificateResult',
    'CBFResult',
    'CLFResult',
    
    # ========================================================================
    # Contraction Analysis
    # ========================================================================
    
    'ContractionMetric',
    'ContractionRate',
    'ContractionAnalysisResult',
    'CCMResult',
    'FunnelingResult',
    'IncrementalStabilityResult',
    
    # ========================================================================
    # Differential Flatness
    # ========================================================================
    
    'FlatnessOutput',
    'DifferentialFlatnessResult',
    'TrajectoryPlanningResult',
    
    # ========================================================================
    # Robustness and Uncertainty
    # ========================================================================
    
    'UncertaintySet',
    'RobustStabilityResult',
    'StructuredSingularValueResult',
    'TubeDefinition',
    'TubeMPCResult',
    'BalancedRealizationResult',
    'ReducedOrderModelResult',
    
    # ========================================================================
    # Function Types
    # ========================================================================
    
    'DynamicsFunction',
    'OutputFunction',
    'DiffusionFunction',
    'ControlPolicy',
    'StateEstimator',
    'CostFunction',
    'Constraint',
    
    # ========================================================================
    # Callbacks
    # ========================================================================
    
    'IntegrationCallback',
    'SimulationCallback',
    'OptimizationCallback',
    
    # ========================================================================
    # Protocols
    # ========================================================================
    
    'LinearizableProtocol',
    'SimulatableProtocol',
    'StochasticProtocol',
    
    # ========================================================================
    # Utilities
    # ========================================================================
    
    'is_batched',
    'get_batch_size',
    'is_numpy',
    'is_torch',
    'is_jax',
    'get_backend',
    'ensure_numpy',
    'ensure_backend',
    'check_state_shape',
    'check_control_shape',
    'get_array_shape',
    'extract_dimensions',
    'ArrayConverter',
    
    # ========================================================================
    # Cache and Metadata
    # ========================================================================
    
    'CacheKey',
    'CacheStatistics',
    'Metadata',
    'ValidationResult',
    'PerformanceMetrics',
    
    # ========================================================================
    # Configuration
    # ========================================================================
    
    'SystemConfig',
    'IntegratorConfig',
    'DiscretizerConfig',
    
    # ========================================================================
    # Type Variables
    # ========================================================================
    
    'T',
    'S',
    'C',
    'MatrixT',
    
    # ========================================================================
    # Constants
    # ========================================================================
    
    'VALID_BACKENDS',
    'VALID_DEVICES',
    'DEFAULT_BACKEND',
    'DEFAULT_DEVICE',
    'DEFAULT_DTYPE',
]


# ============================================================================
# Version Information
# ============================================================================

__version__ = '1.0.0'
__author__ = 'Gil Benezer'


# ============================================================================
# Convenience Imports for Common Patterns
# ============================================================================

# Most commonly used types (for quick reference)
COMMON_TYPES = {
    'arrays': (ArrayLike, StateVector, ControlVector),
    'matrices': (StateMatrix, ControlMatrix, GainMatrix),
    'results': (LQRResult, MPCResult, KalmanFilterResult),
    'analysis': (StabilityInfo, LinearizationResult),
}


def list_all_types() -> List[str]:
    """
    List all available type names.
    
    Returns
    -------
    List[str]
        All exported type names
    
    Examples
    --------
    >>> from src.systems.base.types import list_all_types
    >>> types = list_all_types()
    >>> print(f"Total types: {len(types)}")
    >>> print(f"Control types: {[t for t in types if 'Result' in t]}")
    """
    return sorted(__all__)


def list_types_by_category() -> Dict[str, List[str]]:
    """
    List types organized by category.
    
    Returns
    -------
    Dict[str, List[str]]
        Types grouped by domain
    
    Examples
    --------
    >>> from src.systems.base.types import list_types_by_category
    >>> categories = list_types_by_category()
    >>> print(categories['Control'])
    ['LQRResult', 'LQGResult', 'MPCResult', ...]
    """
    categories = {
        'Core Arrays': [
            'ArrayLike', 'StateVector', 'ControlVector', 'OutputVector',
            'NoiseVector', 'ParameterVector',
        ],
        'Matrices': [
            'StateMatrix', 'ControlMatrix', 'GainMatrix', 'CovarianceMatrix',
            'DiffusionMatrix', 'OutputMatrix',
        ],
        'Control': [
            'LQRResult', 'LQGResult', 'MPCResult', 'MHEResult',
            'H2ControlResult', 'HInfControlResult',
        ],
        'Estimation': [
            'KalmanFilterResult', 'EKFResult', 'UKFResult', 'ParticleFilterResult',
        ],
        'System ID': [
            'HankelMatrix', 'ERAResult', 'DMDResult', 'SINDyResult',
            'SubspaceIDResult', 'KoopmanResult',
        ],
        'Reachability': [
            'HJIReachabilityResult', 'BackwardReachableResult',
            'ForwardReachableResult', 'ViabilityKernelResult',
        ],
        'Safety': [
            'CBFResult', 'CLFResult', 'BarrierCertificateResult',
            'VerificationResult',
        ],
        'Contraction': [
            'ContractionAnalysisResult', 'CCMResult', 'FunnelingResult',
            'IncrementalStabilityResult',
        ],
        'Conformal': [
            'ConformalPredictionResult', 'ConformalCalibrationResult',
            'AdaptiveConformalResult',
        ],
        'Robustness': [
            'RobustStabilityResult', 'TubeMPCResult',
            'StructuredSingularValueResult',
        ],
    }
    return categories


def get_type_info(type_name: str) -> Optional[str]:
    """
    Get documentation for a specific type.
    
    Parameters
    ----------
    type_name : str
        Name of the type
    
    Returns
    -------
    Optional[str]
        Docstring of the type, or None if not found
    
    Examples
    --------
    >>> from src.systems.base.types import get_type_info
    >>> info = get_type_info('StateVector')
    >>> print(info)
    State vector x ∈ ℝⁿˣ.
    Shapes:
    - Single: (nx,)
    - Batched: (batch, nx)
    ...
    """
    if type_name in globals():
        obj = globals()[type_name]
        return obj.__doc__ if hasattr(obj, '__doc__') else None
    return None


# ============================================================================
# Quick Reference Guide
# ============================================================================

QUICK_REFERENCE = """
ControlDESymulation Types Quick Reference
=========================================

MOST COMMON TYPES:
  Arrays:      ArrayLike, StateVector, ControlVector
  Matrices:    StateMatrix, ControlMatrix, GainMatrix
  Results:     LQRResult, MPCResult, LinearizationResult
  Backend:     Backend ('numpy'|'torch'|'jax')

CONTROL DESIGN:
  Classical:   LQRResult, LQGResult, KalmanFilterResult
  Advanced:    MPCResult, MHEResult, H2ControlResult, HInfControlResult
  Adaptive:    AdaptiveControlResult, SlidingModeResult
  Stochastic:  StochasticMPCResult, RiskSensitiveResult

STATE ESTIMATION:
  Linear:      KalmanFilterResult
  Nonlinear:   EKFResult, UKFResult, ParticleFilterResult
  Predictive:  ConformalPredictionResult, AdaptiveConformalResult

SYSTEM IDENTIFICATION:
  Matrices:    HankelMatrix, ToeplitzMatrix, TrajectoryMatrix
  Methods:     ERAResult, DMDResult, SINDyResult, SubspaceIDResult
  Data-Driven: KoopmanResult, NeuralNetworkDynamicsResult

SAFETY & REACHABILITY:
  Sets:        ReachableSet, SafeSet, ValueFunction
  Analysis:    HJIReachabilityResult, BackwardReachableResult
  Synthesis:   CBFResult, CLFResult, BarrierCertificateResult

CONTRACTION & FLATNESS:
  Contraction: ContractionMetric, CCMResult, FunnelingResult
  Flatness:    FlatnessOutput, DifferentialFlatnessResult

ROBUSTNESS:
  Analysis:    RobustStabilityResult, StructuredSingularValueResult
  Control:     TubeMPCResult, UncertaintySet

UTILITIES:
  Guards:      is_batched(), get_backend(), is_numpy()
  Convert:     ensure_numpy(), ensure_backend(), ArrayConverter
  Validate:    check_state_shape(), check_control_shape()

Import everything:
  from src.systems.base.types import StateVector, LQRResult, Backend
"""


def print_quick_reference():
    """Print quick reference guide."""
    print(QUICK_REFERENCE)


# ============================================================================
# Module Information
# ============================================================================

def get_module_info() -> Dict[str, Any]:
    """
    Get information about types module.
    
    Returns
    -------
    Dict[str, Any]
        Module metadata and statistics
    
    Examples
    --------
    >>> from src.systems.base.types import get_module_info
    >>> info = get_module_info()
    >>> print(f"Total types: {info['total_types']}")
    >>> print(f"Version: {info['version']}")
    """
    return {
        'version': __version__,
        'author': __author__,
        'total_types': len(__all__),
        'categories': list(list_types_by_category().keys()),
        'submodules': [
            'core', 'linearization', 'trajectories', 'symbolic',
            'backends', 'control_classical', 'control_advanced',
            'estimation', 'identification', 'reachability',
            'robustness', 'utilities',
        ],
    }