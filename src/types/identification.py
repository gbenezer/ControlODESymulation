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
System Identification Types

Result types for data-driven system identification algorithms:
- Subspace methods (N4SID, MOESP, CVA)
- Eigensystem Realization Algorithm (ERA)
- Dynamic Mode Decomposition (DMD)
- Sparse Identification (SINDy)
- Koopman operator approximation

These methods learn dynamical system models from input-output data,
enabling model-based control without first-principles modeling.

Mathematical Background
----------------------
System Identification Problem:
    Given: Input-output data (u[1:N], y[1:N])
    Find: Model x[k+1] = Ax[k] + Bu[k]
                 y[k] = Cx[k] + Du[k]

    Minimize: Prediction error ||y - ŷ||²

Subspace Identification (N4SID):
    1. Build Hankel matrices from data
    2. Compute oblique projection: Π = Y_future / (U_past, Y_past)
    3. SVD for order selection: Π = UΣV'
    4. Extract observability: O = U_r
    5. Solve for A, B, C, D

ERA (Eigensystem Realization Algorithm):
    From impulse response {H[k]}:
    1. Build Hankel: H(0) = [H[1] H[2] ... H[r]]
                           [H[2] H[3] ... H[r+1]]
    2. SVD: H(0) = UΣV'
    3. Realization: C = U'_r, A = Σ_r^(-1/2) U'_r H(1) V_r Σ_r^(-1/2), B = Σ_r^(1/2) V'_r e_1

DMD (Dynamic Mode Decomposition):
    Given: Snapshot matrices X, Y (Y = dynamics(X))
    Find: A ≈ Y @ X^+
    Eigendecomposition: A = Φ Λ Φ^(-1)
    Modes Φ, eigenvalues Λ describe dynamics

SINDy (Sparse Identification):
    Discover equations: dx/dt = Θ(x) ξ
    Library Θ = [1, x, x², sin(x), ...]
    Sparse regression: minimize ||dX - Θ ξ||² + λ||ξ||₁

Usage
-----
>>> from src.types.identification import (
...     SystemIDResult,
...     SubspaceIDResult,
...     ERAResult,
...     DMDResult,
... )
>>>
>>> # Subspace identification
>>> result: SubspaceIDResult = n4sid(u_data, y_data, order=5)
>>> A, B, C, D = result['A'], result['B'], result['C'], result['D']
>>> print(f"Model fit: {result['fit_quality']:.1f}%")
>>>
>>> # DMD from trajectory data
>>> dmd_result: DMDResult = dmd(X_snapshots, Y_snapshots, rank=10)
>>> eigenvalues = dmd_result['eigenvalues']
>>> modes = dmd_result['modes']
"""

from typing import Callable, List, Optional

import numpy as np
from typing_extensions import TypedDict

from src.types.core import InputMatrix  # B matrix (nx, nu)
from src.types.core import (
    ArrayLike,
    ControllabilityMatrix,
    CovarianceMatrix,
    DiffusionMatrix,
    FeedthroughMatrix,
    ObservabilityMatrix,
    OutputMatrix,
    StateMatrix,
)

# ============================================================================
# Type Aliases for Identification
# ============================================================================

HankelMatrix = ArrayLike
"""
Hankel matrix for system identification.

Block Hankel matrix constructed from time-series data with constant
diagonals. Essential for subspace ID and ERA.

Shape: (rows*ny, cols) where rows = observability index

Examples
--------
>>> # Build Hankel from output measurements
>>> y_data = np.random.randn(100, 2)  # 100 samples, 2 outputs
>>> H = build_hankel(y_data, rows=10, cols=20)
>>> print(H.shape)  # (20, 20) for ny=2, rows=10
"""

ToeplitzMatrix = ArrayLike
"""
Toeplitz matrix (constant diagonals).

Used in convolution, impulse response analysis.

Examples
--------
>>> from scipy.linalg import toeplitz
>>> impulse = [1.0, 0.8, 0.6, 0.4]
>>> T: ToeplitzMatrix = toeplitz(impulse)
"""

TrajectoryMatrix = ArrayLike
"""
Data matrix for trajectory-based methods (DMD, SINDy, Koopman).

Columns are state snapshots at different times.

Examples
--------
>>> # DMD snapshot matrices
>>> X: TrajectoryMatrix = states[:, :-1]  # x[0:N-1]
>>> Y: TrajectoryMatrix = states[:, 1:]   # x[1:N]
"""

MarkovParameters = ArrayLike
"""
Markov parameters (impulse response coefficients).

For discrete LTI: H[k] = C A^(k-1) B for k≥1, H[0] = D

Shape: (n_params, ny, nu)

Examples
--------
>>> markov: MarkovParameters = system.impulse_response(n_steps=50)
>>> print(markov.shape)  # (50, ny, nu)
"""


# ============================================================================
# General System Identification
# ============================================================================


class SystemIDResult(TypedDict, total=False):
    """
    General system identification result.

    Contains identified state-space model (A, B, C, D) and quality metrics.

    Fields
    ------
    A : StateMatrix
        Identified state matrix (nx, nx)
    B : InputMatrix
        Identified input matrix (nx, nu)
    C : OutputMatrix
        Identified output matrix (ny, nx)
    D : FeedthroughMatrix
        Identified feedthrough (ny, nu)
    G : Optional[DiffusionMatrix]
        Noise gain matrix (stochastic ID) (nx, nw)
    Q : Optional[CovarianceMatrix]
        Process noise covariance (nx, nx)
    R : Optional[CovarianceMatrix]
        Measurement noise covariance (ny, ny)
    order : int
        Model order (number of states nx)
    fit_percentage : float
        Model fit quality (0-100%), VAF = (1 - ||y - ŷ||²/||y||²) × 100
    residuals : ArrayLike
        Prediction residuals y - ŷ (N, ny)
    method : str
        ID method used ('n4sid', 'moesp', 'era', 'okid', etc.)
    hankel_matrix : Optional[HankelMatrix]
        Hankel matrix (if applicable)
    singular_values : Optional[ArrayLike]
        SVD singular values for order selection

    Examples
    --------
    >>> # Identify system from data
    >>> result: SystemIDResult = identify_system(
    ...     u_data, y_data, order=3, method='n4sid'
    ... )
    >>>
    >>> # Extract model
    >>> A, B, C, D = result['A'], result['B'], result['C'], result['D']
    >>> print(f"Order: {result['order']}")
    >>> print(f"Fit: {result['fit_percentage']:.1f}%")
    >>>
    >>> # Validate on test data
    >>> y_pred = simulate_identified(A, B, C, D, u_test, x0)
    >>> residuals = y_test - y_pred
    >>>
    >>> # Check singular values for order selection
    >>> if 'singular_values' in result:
    ...     import matplotlib.pyplot as plt
    ...     plt.semilogy(result['singular_values'], 'o-')
    ...     plt.xlabel('Index')
    ...     plt.ylabel('Singular Value')
    ...     plt.title('Hankel SVD - Order Selection')
    """

    A: StateMatrix
    B: InputMatrix
    C: OutputMatrix
    D: FeedthroughMatrix
    G: Optional[DiffusionMatrix]
    Q: Optional[CovarianceMatrix]
    R: Optional[CovarianceMatrix]
    order: int
    fit_percentage: float
    residuals: ArrayLike
    method: str
    hankel_matrix: Optional[HankelMatrix]
    singular_values: Optional[ArrayLike]


# ============================================================================
# Subspace Identification
# ============================================================================


class SubspaceIDResult(TypedDict, total=False):
    """
    Subspace identification result (N4SID, MOESP, CVA).

    Subspace methods compute state-space models via geometric projections
    and SVD, avoiding nonlinear optimization.

    Fields
    ------
    A : StateMatrix
        State matrix (nx, nx)
    B : InputMatrix
        Input matrix (nx, nu)
    C : OutputMatrix
        Output matrix (ny, nx)
    D : FeedthroughMatrix
        Feedthrough (ny, nu)
    observability_matrix : ObservabilityMatrix
        Extended observability O = [C; CA; CA²; ...] (i*ny, nx)
    controllability_matrix : ControllabilityMatrix
        Extended controllability C = [B AB A²B ...] (nx, i*nu)
    hankel_matrix : HankelMatrix
        Data Hankel matrix
    projection_matrix : ArrayLike
        Oblique/orthogonal projection
    singular_values : ArrayLike
        For order selection (i, )
    order : int
        Selected model order nx
    fit_quality : float
        VAF (Variance Accounted For) percentage

    Examples
    --------
    >>> # N4SID identification
    >>> result: SubspaceIDResult = n4sid(
    ...     u_data, y_data, order=5, horizon=20
    ... )
    >>>
    >>> A = result['A']
    >>> observability = result['observability_matrix']
    >>>
    >>> # Order selection via singular values
    >>> sv = result['singular_values']
    >>> import matplotlib.pyplot as plt
    >>> plt.semilogy(sv, 'o-')
    >>> plt.axhline(sv[5], color='r', linestyle='--', label=f'Order {result["order"]}')
    >>> plt.legend()
    >>>
    >>> # Validate fit
    >>> print(f"Variance Accounted For: {result['fit_quality']:.1f}%")
    """

    A: StateMatrix
    B: InputMatrix
    C: OutputMatrix
    D: FeedthroughMatrix
    observability_matrix: ObservabilityMatrix
    controllability_matrix: ControllabilityMatrix
    hankel_matrix: HankelMatrix
    projection_matrix: ArrayLike
    singular_values: ArrayLike
    order: int
    fit_quality: float


# ============================================================================
# ERA (Eigensystem Realization Algorithm)
# ============================================================================


class ERAResult(TypedDict):
    """
    Eigensystem Realization Algorithm result.

    ERA identifies minimal realization from impulse response (Markov parameters).

    Fields
    ------
    A : StateMatrix
        Identified A matrix (nx, nx)
    B : InputMatrix
        Identified B matrix (nx, nu)
    C : OutputMatrix
        Identified C matrix (ny, nx)
    D : FeedthroughMatrix
        Identified D matrix (ny, nu)
    hankel_matrix : HankelMatrix
        Hankel built from Markov parameters
    singular_values : ArrayLike
        Hankel SVD singular values
    system_modes : np.ndarray
        System eigenvalues (poles)
    order : int
        Model order
    observability_matrix : ObservabilityMatrix
        Observability O
    controllability_matrix : ControllabilityMatrix
        Controllability C

    Examples
    --------
    >>> # Get impulse response from experiment or simulation
    >>> markov_params = compute_impulse_response(system, n_steps=100)
    >>>
    >>> # ERA identification
    >>> result: ERAResult = era(markov_params, order=10)
    >>>
    >>> A = result['A']
    >>> modes = result['system_modes']
    >>>
    >>> # Check system stability
    >>> if np.all(np.abs(modes) < 1):
    ...     print("Identified system is stable")
    >>>
    >>> # Singular value plot for order selection
    >>> sv = result['singular_values']
    >>> plt.semilogy(sv, 'o-')
    >>> plt.title('ERA Singular Values')
    """

    A: StateMatrix
    B: InputMatrix
    C: OutputMatrix
    D: FeedthroughMatrix
    hankel_matrix: HankelMatrix
    singular_values: ArrayLike
    system_modes: np.ndarray
    order: int
    observability_matrix: ObservabilityMatrix
    controllability_matrix: ControllabilityMatrix


# ============================================================================
# Dynamic Mode Decomposition (DMD)
# ============================================================================


class DMDResult(TypedDict, total=False):
    """
    Dynamic Mode Decomposition result.

    DMD extracts spatio-temporal coherent structures from data.
    Linear approximation: x[k+1] ≈ A_dmd x[k]

    Fields
    ------
    dynamics_matrix : StateMatrix
        DMD dynamics matrix Ã (nx, nx)
    modes : ArrayLike
        DMD modes (spatial patterns) (nx, rank)
    eigenvalues : np.ndarray
        DMD eigenvalues (complex) (rank,)
    amplitudes : ArrayLike
        Mode amplitudes (rank,)
    frequencies : ArrayLike
        Mode frequencies in rad/s (rank,)
    growth_rates : ArrayLike
        Mode growth/decay rates (rank,)
    rank : int
        Truncation rank
    singular_values : ArrayLike
        SVD singular values

    Examples
    --------
    >>> # DMD from snapshot data
    >>> X = states[:, :-1]  # x[0:N-1]
    >>> Y = states[:, 1:]   # x[1:N]
    >>>
    >>> result: DMDResult = dmd(X, Y, rank=10, dt=0.01)
    >>>
    >>> # Extract modal information
    >>> modes = result['modes']
    >>> eigenvalues = result['eigenvalues']
    >>> frequencies = result['frequencies']
    >>>
    >>> # Identify dominant modes
    >>> amplitudes = result['amplitudes']
    >>> dominant_idx = np.argsort(np.abs(amplitudes))[::-1][:5]
    >>>
    >>> print("Top 5 modes:")
    >>> for i in dominant_idx:
    ...     print(f"  λ = {eigenvalues[i]:.3f}, f = {frequencies[i]:.2f} rad/s")
    >>>
    >>> # Reconstruct dynamics
    >>> A_dmd = result['dynamics_matrix']
    >>> x_pred = A_dmd @ X[:, 0]  # One-step prediction
    """

    dynamics_matrix: StateMatrix
    modes: ArrayLike
    eigenvalues: np.ndarray
    amplitudes: ArrayLike
    frequencies: ArrayLike
    growth_rates: ArrayLike
    rank: int
    singular_values: ArrayLike


# ============================================================================
# SINDy (Sparse Identification of Nonlinear Dynamics)
# ============================================================================


class SINDyResult(TypedDict, total=False):
    """
    SINDy (Sparse Identification of Nonlinear Dynamics) result.

    Discovers sparse governing equations from data via regression.

    Fields
    ------
    coefficients : ArrayLike
        Sparse coefficient matrix Ξ (n_features, nx)
    active_terms : List[str]
        Human-readable active terms (e.g., ['x', 'x^2', 'sin(x)'])
    library_functions : List[Callable]
        Basis functions Θ used
    sparsity_level : float
        Fraction of zero coefficients (0-1)
    reconstruction_error : float
        ||dX - Θ Ξ||_F
    condition_number : float
        Condition number of library matrix Θ
    selected_features : List[int]
        Indices of non-zero features

    Examples
    --------
    >>> # Define library functions
    >>> library = [
    ...     lambda x: x,
    ...     lambda x: x**2,
    ...     lambda x: x**3,
    ...     lambda x: np.sin(x),
    ...     lambda x: np.cos(x),
    ... ]
    >>>
    >>> # SINDy identification
    >>> result: SINDyResult = sindy(
    ...     X_data, dX_data, library, threshold=0.1
    ... )
    >>>
    >>> # Print discovered equations
    >>> print("Identified dynamics:")
    >>> for i, terms in enumerate(result['active_terms']):
    ...     print(f"  dx{i}/dt = {terms}")
    >>>
    >>> # Check sparsity
    >>> print(f"Sparsity: {result['sparsity_level']*100:.1f}% zeros")
    >>> print(f"Reconstruction error: {result['reconstruction_error']:.2e}")
    >>>
    >>> # Validate on test data
    >>> Theta_test = build_library(X_test, result['library_functions'])
    >>> dX_pred = Theta_test @ result['coefficients']
    """

    coefficients: ArrayLike
    active_terms: List[str]
    library_functions: List[Callable]
    sparsity_level: float
    reconstruction_error: float
    condition_number: float
    selected_features: List[int]


# ============================================================================
# Koopman Operator
# ============================================================================


class KoopmanResult(TypedDict, total=False):
    """
    Koopman operator approximation result.

    Represents nonlinear dynamics as linear in lifted observable space.
    φ[k+1] = K φ[k], where φ = [φ₁(x), φ₂(x), ...]

    Fields
    ------
    koopman_operator : StateMatrix
        Koopman matrix K (lifted_dim, lifted_dim)
    lifting_functions : List[Callable]
        Observable functions φ(x)
    lifted_dimension : int
        Dimension of lifted space
    eigenvalues : np.ndarray
        Koopman eigenvalues
    eigenfunctions : ArrayLike
        Koopman eigenfunctions
    reconstruction_error : float
        Approximation error

    Examples
    --------
    >>> # Define observables (polynomial basis)
    >>> observables = [
    ...     lambda x: x[0],
    ...     lambda x: x[1],
    ...     lambda x: x[0]**2,
    ...     lambda x: x[0]*x[1],
    ...     lambda x: x[1]**2,
    ... ]
    >>>
    >>> result: KoopmanResult = koopman_approximation(
    ...     trajectories, observables
    ... )
    >>>
    >>> K = result['koopman_operator']
    >>> eigenvalues = result['eigenvalues']
    >>>
    >>> # Prediction in lifted space
    >>> def predict(x0, n_steps):
    ...     phi0 = np.array([obs(x0) for obs in observables])
    ...     phi_traj = [phi0]
    ...     for _ in range(n_steps):
    ...         phi_traj.append(K @ phi_traj[-1])
    ...     return np.array(phi_traj)
    >>>
    >>> # Extract original states (first two observables)
    >>> phi_pred = predict(x0, n_steps=100)
    >>> x_pred = phi_pred[:, :2]
    """

    koopman_operator: StateMatrix
    lifting_functions: List[Callable]
    lifted_dimension: int
    eigenvalues: np.ndarray
    eigenfunctions: ArrayLike
    reconstruction_error: float


# ============================================================================
# Export All
# ============================================================================

__all__ = [
    # Type aliases
    "HankelMatrix",
    "ToeplitzMatrix",
    "TrajectoryMatrix",
    "MarkovParameters",
    # Identification results
    "SystemIDResult",
    "SubspaceIDResult",
    "ERAResult",
    "DMDResult",
    "SINDyResult",
    "KoopmanResult",
]
