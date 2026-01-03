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
System Analysis Wrapper

Thin wrapper around system analysis functions for system composition.

Provides backend consistency with parent system while delegating to
pure functions in classical.py. This is NOT a heavy utility - it simply
routes to stateless algorithms.

Design Philosophy
-----------------
- Composition not inheritance
- Thin wrapper (no state, no caching)
- Routes to pure functions
- Backend consistency with parent system
- Clean integration with system classes

Architecture
------------
SystemAnalysis is a lightweight utility that:
1. Stores only the backend setting from parent system
2. Delegates all work to pure functions in classical.py
3. Provides clean method names for composition

Usage
-----
>>> # Direct instantiation (rare - usually via system)
>>> from src.control.analysis import SystemAnalysis
>>> import numpy as np
>>>
>>> analyzer = SystemAnalysis(backend='numpy')
>>> A = np.array([[0, 1], [-2, -3]])
>>> stability = analyzer.stability(A, system_type='continuous')
>>> print(f"Stable: {stability['is_stable']}")
>>>
>>> # Typical usage - via system composition
>>> system = Pendulum()
>>> A, B = system.linearize(x_eq, u_eq)
>>> stability = system.analysis.stability(A, system_type='continuous')
>>> ctrl_info = system.analysis.controllability(A, B)
"""

from src.types.backends import Backend
from src.types.control_classical import (
    ControllabilityInfo,
    ObservabilityInfo,
    StabilityInfo,
)
from src.types.core import (
    InputMatrix,
    OutputMatrix,
    StateMatrix,
)


class SystemAnalysis:
    """
    System analysis wrapper for composition.

    Thin wrapper that routes to pure system analysis functions while
    maintaining backend consistency with parent system.

    This class holds minimal state (just backend setting) and delegates
    all computation to pure functions in classical.py.

    Attributes
    ----------
    backend : Backend
        Computational backend ('numpy', 'torch', 'jax')

    Examples
    --------
    >>> # Via system composition (typical usage)
    >>> system = Pendulum()
    >>> A, B = system.linearize(x_eq, u_eq)
    >>>
    >>> # Check stability
    >>> stability = system.analysis.stability(A, system_type='continuous')
    >>> if stability['is_stable']:
    ...     print(f"Stable with eigenvalues: {stability['eigenvalues']}")
    ...     print(f"Stability margin: {stability['stability_margin']:.3f}")
    >>>
    >>> # Check controllability
    >>> ctrl_info = system.analysis.controllability(A, B)
    >>> if not ctrl_info['is_controllable']:
    ...     print("Warning: System is not fully controllable")
    ...     print(f"Controllable subspace dimension: {ctrl_info['rank']}")
    >>>
    >>> # Check observability
    >>> C = np.array([[1, 0]])  # Measure first state only
    >>> obs_info = system.analysis.observability(A, C)
    >>> if obs_info['is_observable']:
    ...     print("Full state can be estimated from measurements")

    Notes
    -----
    This is a thin wrapper - all algorithms are in classical.py.
    The wrapper only provides:
    1. Backend consistency with parent system
    2. Clean composition interface
    3. Convenience for system integration
    """

    def __init__(self, backend: Backend = "numpy"):
        """
        Initialize system analysis wrapper.

        Args:
            backend: Computational backend from parent system
                     ('numpy', 'torch', 'jax')

        Examples
        --------
        >>> # Usually created by system, not directly
        >>> analyzer = SystemAnalysis(backend='torch')
        """
        self.backend = backend

    def stability(
        self,
        A: StateMatrix,
        system_type: str = "continuous",
        tolerance: float = 1e-10,
    ) -> StabilityInfo:
        """
        Analyze system stability via eigenvalue analysis.

        Routes to classical.analyze_stability().

        Stability criteria:
            Continuous: All Re(λ) < 0 (left half-plane)
            Discrete:   All |λ| < 1 (inside unit circle)

        Args:
            A: State matrix (nx, nx)
            system_type: 'continuous' or 'discrete'
            tolerance: Tolerance for marginal stability detection

        Returns:
            StabilityInfo with eigenvalues, magnitudes, and stability flags

        Examples
        --------
        >>> # Analyze linearized system
        >>> system = Pendulum()
        >>> x_eq = np.array([np.pi, 0])  # Upright equilibrium
        >>> u_eq = np.zeros(1)
        >>> A, B = system.linearize(x_eq, u_eq)
        >>>
        >>> # Check stability
        >>> stability = system.analysis.stability(A, system_type='continuous')
        >>> print(f"Stable: {stability['is_stable']}")
        >>> print(f"Eigenvalues: {stability['eigenvalues']}")
        >>> print(f"Max magnitude: {stability['max_magnitude']:.3f}")
        >>>
        >>> # Check if marginal (on boundary)
        >>> if stability['is_marginally_stable']:
        ...     print("System is critically stable (eigenvalues on boundary)")
        >>>
        >>> # Discrete system
        >>> discrete_system = DiscreteSystem()
        >>> Ad, Bd = discrete_system.linearize(x_eq, u_eq)
        >>> stability_d = discrete_system.analysis.stability(Ad, system_type='discrete')
        >>> print(f"Spectral radius: {stability_d['spectral_radius']:.3f}")
        >>> print(f"Stable (|λ| < 1): {stability_d['is_stable']}")

        Notes
        -----
        - Continuous systems: Stable if all eigenvalues in left half-plane
        - Discrete systems: Stable if all eigenvalues inside unit circle
        - Marginal stability: Eigenvalues on stability boundary
        - Asymptotic stability: All trajectories converge to equilibrium
        - For nonlinear systems, this only gives local stability around equilibrium

        See Also
        --------
        controllability : Check if system is controllable
        observability : Check if system is observable
        """
        from src.control.classical_control_functions import analyze_stability

        return analyze_stability(A, system_type, tolerance)

    def controllability(
        self,
        A: StateMatrix,
        B: InputMatrix,
        tolerance: float = 1e-10,
    ) -> ControllabilityInfo:
        """
        Test controllability of linear system (A, B).

        Routes to classical.analyze_controllability().

        A system is controllable if all states can be driven to any
        desired value in finite time using appropriate control inputs.

        Controllability test: rank([B AB A²B ... A^(n-1)B]) = n

        Args:
            A: State matrix (nx, nx)
            B: Input matrix (nx, nu)
            tolerance: Tolerance for rank computation

        Returns:
            ControllabilityInfo with controllability matrix, rank, and flag

        Examples
        --------
        >>> # Test controllability of linearized system
        >>> system = Pendulum()
        >>> A, B = system.linearize(x_eq, u_eq)
        >>>
        >>> ctrl_info = system.analysis.controllability(A, B)
        >>> print(f"Controllable: {ctrl_info['is_controllable']}")
        >>> print(f"Controllability matrix rank: {ctrl_info['rank']}/{A.shape[0]}")
        >>>
        >>> if ctrl_info['is_controllable']:
        ...     print("All states can be controlled")
        ...     # Can design pole placement, LQR, etc.
        ...     lqr_result = system.design_lqr(Q, R)
        ... else:
        ...     print(f"Only {ctrl_info['rank']} states are controllable")
        ...     print("Cannot use pole placement or LQR for full state")
        >>>
        >>> # Example: Uncontrollable system
        >>> A_diag = np.array([[1, 0], [0, 2]])
        >>> B_identical = np.array([[1], [1]])  # Same input to both states
        >>> ctrl_info = system.analysis.controllability(A_diag, B_identical)
        >>> print(f"Controllable: {ctrl_info['is_controllable']}")  # False

        Notes
        -----
        - Controllability required for arbitrary pole placement
        - Stabilizability (weaker): Unstable modes must be controllable
        - Single-input systems: Often fully controllable with proper structure
        - Multi-input systems: Provides more design freedom
        - Numerical issues: Use tolerance for near-singular systems

        See Also
        --------
        observability : Dual concept for state estimation
        stability : Check stability of linearized system
        """
        from src.control.classical_control_functions import analyze_controllability

        return analyze_controllability(A, B, tolerance)

    def observability(
        self,
        A: StateMatrix,
        C: OutputMatrix,
        tolerance: float = 1e-10,
    ) -> ObservabilityInfo:
        """
        Test observability of linear system (A, C).

        Routes to classical.analyze_observability().

        A system is observable if the initial state can be determined
        from output measurements over a finite time interval.

        Observability test: rank([C; CA; CA²; ...; CA^(n-1)]) = n

        Args:
            A: State matrix (nx, nx)
            C: Output matrix (ny, nx)
            tolerance: Tolerance for rank computation

        Returns:
            ObservabilityInfo with observability matrix, rank, and flag

        Examples
        --------
        >>> # Test observability with partial state measurement
        >>> system = Pendulum()
        >>> A, _ = system.linearize(x_eq, u_eq)
        >>> C = np.array([[1, 0]])  # Measure position only, not velocity
        >>>
        >>> obs_info = system.analysis.observability(A, C)
        >>> print(f"Observable: {obs_info['is_observable']}")
        >>> print(f"Observability matrix rank: {obs_info['rank']}/{A.shape[0]}")
        >>>
        >>> if obs_info['is_observable']:
        ...     print("Full state can be estimated from position measurement")
        ...     # Can design Kalman filter, observer, etc.
        ...     kalman = system.control.design_kalman(A, C, Q_proc, R_meas)
        ... else:
        ...     print(f"Only {obs_info['rank']} states are observable")
        ...     print("Need additional measurements for full state estimation")
        >>>
        >>> # Example: Full state measurement
        >>> C_full = np.eye(2)  # Measure both position and velocity
        >>> obs_full = system.analysis.observability(A, C_full)
        >>> print(f"Observable: {obs_full['is_observable']}")  # True
        >>>
        >>> # Example: Unobservable system
        >>> A_diag = np.array([[1, 0], [0, 2]])
        >>> C_sum = np.array([[1, 1]])  # Can't distinguish individual states
        >>> obs_info = system.analysis.observability(A_diag, C_sum)
        >>> print(f"Observable: {obs_info['is_observable']}")  # False

        Notes
        -----
        - Observability required for state estimation (Kalman filter)
        - Detectability (weaker): Unstable modes must be observable
        - Dual to controllability: (A,C) observable ⟺ (A',C') controllable
        - More outputs generally → better observability
        - Trade-off: More sensors vs. complexity/cost

        See Also
        --------
        controllability : Dual concept for control design
        stability : Check stability of observer dynamics
        """
        from src.control.classical_control_functions import analyze_observability

        return analyze_observability(A, C, tolerance)

    def analyze_linearization(
        self,
        A: StateMatrix,
        B: InputMatrix,
        C: OutputMatrix,
        system_type: str = "continuous",
    ) -> dict:
        """
        Comprehensive analysis of linearized system.

        Convenience method that runs stability, controllability, and
        observability analysis in one call.

        Args:
            A: State matrix (nx, nx)
            B: Input matrix (nx, nu)
            C: Output matrix (ny, nx)
            system_type: 'continuous' or 'discrete'

        Returns:
            Dictionary containing:
                - stability: StabilityInfo
                - controllability: ControllabilityInfo
                - observability: ObservabilityInfo
                - summary: Dict with combined assessment

        Examples
        --------
        >>> # Complete analysis of linearized system
        >>> system = Pendulum()
        >>> x_eq = np.array([np.pi, 0])  # Upright
        >>> u_eq = np.zeros(1)
        >>> A, B = system.linearize(x_eq, u_eq)
        >>> C = np.array([[1, 0]])  # Measure position
        >>>
        >>> analysis = system.analysis.analyze_linearization(
        ...     A, B, C,
        ...     system_type='continuous'
        ... )
        >>>
        >>> # Check results
        >>> print(f"Stable: {analysis['stability']['is_stable']}")
        >>> print(f"Controllable: {analysis['controllability']['is_controllable']}")
        >>> print(f"Observable: {analysis['observability']['is_observable']}")
        >>>
        >>> # Summary assessment
        >>> summary = analysis['summary']
        >>> if summary['ready_for_lqr']:
        ...     print("System is stabilizable via LQR")
        >>> if summary['ready_for_kalman']:
        ...     print("State estimation possible via Kalman filter")
        >>> if summary['ready_for_lqg']:
        ...     print("Full LQG controller can be designed")

        Notes
        -----
        This is a convenience method for comprehensive system analysis.
        Use individual methods (stability, controllability, observability)
        if you only need specific information.
        """
        # Run all analyses
        stability = self.stability(A, system_type)
        controllability = self.controllability(A, B)
        observability = self.observability(A, C)

        # Create summary assessment
        summary = {
            "is_stable": stability["is_stable"],
            "is_controllable": controllability["is_controllable"],
            "is_observable": observability["is_observable"],
            "ready_for_lqr": controllability["is_controllable"],
            "ready_for_kalman": observability["is_observable"],
            "ready_for_lqg": controllability["is_controllable"] and observability["is_observable"],
            "stabilizable": controllability["is_controllable"],  # TODO: proper stabilizability test
            "detectable": observability["is_observable"],  # TODO: proper detectability test
        }

        return {
            "stability": stability,
            "controllability": controllability,
            "observability": observability,
            "summary": summary,
        }


# ============================================================================
# Export
# ============================================================================

__all__ = [
    "SystemAnalysis",
]
