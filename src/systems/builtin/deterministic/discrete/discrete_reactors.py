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

from typing import Optional

import numpy as np
import sympy as sp

from src.systems.base.core.discrete_symbolic_system import DiscreteSymbolicSystem

class DiscreteBatchReactor(DiscreteSymbolicSystem):
    """
    Discrete-time chemical batch reactor with temperature control.

    Physical System:
    ---------------
    A well-mixed batch reactor where chemical species A converts to B,
    which then converts to product C. The reactor operates in discrete
    time steps (sampling intervals) typical of:
    - Digital control systems with periodic measurements
    - Batch processing with staged operations
    - Industrial reactors with discrete valve/heater actuation

    The reaction sequence is:
        A → B → C

    Both reactions are first-order and temperature-dependent following
    Arrhenius kinetics. Temperature affects reaction rates exponentially,
    creating strong nonlinear coupling between composition and thermal
    dynamics.

    State Space:
    -----------
    State: x[k] = [Cₐ[k], Cᵦ[k], T[k]]
        Concentration states:
        - Cₐ: Concentration of reactant A [mol/L]
          * Initial concentration typically Cₐ[0] = 1.0 mol/L
          * Decreases monotonically (consumed by first reaction)
          * Must remain non-negative: Cₐ ≥ 0

        - Cᵦ: Concentration of intermediate B [mol/L]
          * Produced from A, consumed to form C
          * Non-monotonic: rises then falls
          * Maximum occurs when r₁ = r₂ (production = consumption)
          * Must remain non-negative: Cᵦ ≥ 0

        Temperature state:
        - T: Reactor temperature [K]
          * Typical range: 300-400 K (27-127°C)
          * Affects reaction rates exponentially via Arrhenius
          * Subject to heat loss to ambient (cooling)
          * Controlled by external heating Q

    Control: u[k] = [Q[k]]
        - Q: Heating/cooling rate [K/s]
          * Q > 0: Heating applied
          * Q < 0: Active cooling
          * Q = 0: Natural heat loss only
          * Typical range: -50 to +50 K/s

    Output: y[k] = [Cₐ[k], Cᵦ[k], T[k]]
        - Full state measurement (all concentrations and temperature)
        - In practice, concentration may be measured via:
          * Spectroscopy (UV-Vis, IR)
          * Chromatography (GC, HPLC)
          * Online analyzers
        - Temperature measured via thermocouple or RTD

    Dynamics:
    --------
    The discrete-time dynamics use Euler discretization:

        Cₐ[k+1] = Cₐ[k] - dt·r₁[k]
        Cᵦ[k+1] = Cᵦ[k] + dt·(r₁[k] - r₂[k])
        T[k+1] = T[k] + dt·(Q[k] - α·(T[k] - Tₐₘᵦ))

    **Reaction Rates (Arrhenius kinetics)**:
        r₁[k] = k₁·Cₐ[k]·exp(-E₁/T[k])    [mol/(L·s)]
        r₂[k] = k₂·Cᵦ[k]·exp(-E₂/T[k])    [mol/(L·s)]

    where:
    - k₁, k₂: Pre-exponential factors (frequency factors)
    - E₁, E₂: Activation energies [K] (using Eₐ/R as temperature)
    - exp(-E/T): Arrhenius temperature dependence

    **Physical Interpretation**:

    Reaction 1 (A → B):
    - Rate r₁ proportional to Cₐ (first-order kinetics)
    - Exponentially increases with temperature
    - Higher E₁ → more temperature sensitive
    - Depletes reactant A, produces intermediate B

    Reaction 2 (B → C):
    - Rate r₂ proportional to Cᵦ (first-order kinetics)
    - Exponentially increases with temperature
    - Higher E₂ → more temperature sensitive
    - Consumes intermediate B, produces final product C

    Temperature dynamics:
    - Q[k]: External heating/cooling control
    - -α·(T - Tₐₘᵦ): Heat loss to ambient (Newton's cooling)
    - α: Heat transfer coefficient [1/s]
    - Tₐₘᵦ: Ambient temperature [K]

    **Nonlinear Coupling**:
    The system exhibits strong nonlinear coupling:
    1. Temperature affects reaction rates exponentially
    2. Reactions may be exothermic/endothermic (not modeled here)
    3. Competing reactions create non-monotonic Cᵦ profile

    Parameters:
    ----------
    k1 : float, default=0.5
        Pre-exponential factor for reaction 1 (A→B) [1/s]
        Higher k₁ → faster depletion of A
        Typical range: 0.1 - 10.0

    k2 : float, default=0.3
        Pre-exponential factor for reaction 2 (B→C) [1/s]
        Higher k₂ → faster conversion of B to C
        Typical range: 0.1 - 10.0

    E1 : float, default=1000.0
        Activation energy for reaction 1 [K] (actually Eₐ/R)
        Higher E₁ → more sensitive to temperature
        Physical Eₐ typically 8,000 - 30,000 K

    E2 : float, default=1500.0
        Activation energy for reaction 2 [K] (actually Eₐ/R)
        E₂ > E₁ means reaction 2 is more temperature-sensitive
        Creates selectivity control via temperature

    alpha : float, default=0.1
        Heat transfer coefficient [1/s]
        Characterizes cooling rate to ambient
        Higher α → faster heat loss, harder to maintain temperature

    T_amb : float, default=300.0
        Ambient temperature [K] (27°C)
        System equilibrium temperature with Q = 0

    dt : float, default=1.0
        Sampling/discretization time step [s]
        Critical parameter affecting stability:
        - Too large → numerical instability
        - Too small → slow simulation, control system bandwidth
        Typical: 0.1 - 10.0 seconds

    Equilibria:
    ----------
    **Steady-state (complete conversion)**:
        x_eq = [0, 0, Tₐₘᵦ]  (all reactants consumed, cooled to ambient)
        u_eq = 0  (no heating needed)

    This equilibrium is reached after sufficient batch time when:
    - All A has converted to B: Cₐ → 0
    - All B has converted to C: Cᵦ → 0
    - Temperature equilibrates with ambient: T → Tₐₘᵦ

    This is a **stable equilibrium** (globally attracting).

    **Optimal operating point** (maximum B yield):
        If goal is to maximize Cᵦ at a specific time, equilibrium
        concept doesn't apply. Instead, use optimal control to find
        temperature trajectory Q[k] that maximizes Cᵦ at final time.

    **Temperature setpoint equilibrium** (partial reaction):
        For constant T* > Tₐₘᵦ maintained by control:
        - Requires Q_eq = α·(T* - Tₐₘᵦ) to balance heat loss
        - Concentrations evolve according to reaction kinetics at T*
        - Not a true equilibrium (Cₐ, Cᵦ still changing)

    Control Objectives:
    ------------------
    Common control goals for batch reactors:

    1. **Temperature tracking**: Maintain T[k] ≈ T_ref[k]
       - Maximize reaction rate
       - Ensure safety (prevent runaway)
       - LQR/MPC controllers typical

    2. **Yield optimization**: Maximize Cᵦ at final time
       - Requires optimal temperature trajectory
       - May involve heating → cooling profile
       - Dynamic programming or direct optimization

    3. **Batch time minimization**: Reach Cₐ < ε in minimum time
       - Subject to temperature constraints (T_min ≤ T ≤ T_max)
       - Bang-bang control often optimal

    4. **Selectivity control**: Maximize ratio Cᵦ/Cᶜ
       - Exploit different activation energies (E₁ vs E₂)
       - Intermediate temperature maximizes B

    State Constraints:
    -----------------
    Physical constraints that must be enforced:

    1. **Non-negativity**: Cₐ[k] ≥ 0, Cᵦ[k] ≥ 0
       - Concentrations cannot be negative
       - Euler discretization may violate if dt too large

    2. **Conservation**: Cₐ[k] + Cᵦ[k] + Cᶜ[k] = Cₐ[0]
       - Total moles conserved (if C tracked)
       - Useful for validation

    3. **Temperature limits**: T_min ≤ T[k] ≤ T_max
       - Safety: prevent runaway or solidification
       - Typical: 280 K ≤ T ≤ 450 K

    4. **Actuation limits**: Q_min ≤ Q[k] ≤ Q_max
       - Physical heating/cooling capacity
       - Typical: -50 ≤ Q ≤ 50 K/s

    Numerical Considerations:
    ------------------------
    **Stability**: The explicit Euler discretization is stable if:
        dt < 1/λ_max

    where λ_max is the maximum eigenvalue of the Jacobian.

    For this system, linearizing around typical operating points:
        λ_max ≈ max(k₁·exp(-E₁/T), k₂·exp(-E₂/T), α)

    At high temperature, reaction rates can be very fast, requiring
    small dt for stability. Rule of thumb:
        dt < 0.1 / max(k₁·exp(-E₁/T), k₂·exp(-E₂/T))

    **Accuracy**: Higher-order methods (RK4, etc.) can be used:
        system_continuous = ContinuousBatchReactor(...)
        system_discrete = system_continuous.discretize(dt=1.0, method='rk4')

    **Stiffness**: If E₁ ≫ E₂ or vice versa, system may be stiff,
    requiring implicit methods or very small dt.

    Example Usage:
    -------------
    >>> # Create reactor with default parameters
    >>> reactor = DiscreteBatchReactor(dt=0.5)
    >>> 
    >>> # Initial condition: fresh batch
    >>> x0 = np.array([1.0, 0.0, 350.0])  # [Cₐ, Cᵦ, T]
    >>> 
    >>> # Simulate with constant heating
    >>> result = reactor.simulate(
    ...     x0=x0,
    ...     u_sequence=np.array([10.0]),  # Constant Q = 10 K/s
    ...     n_steps=100
    ... )
    >>> 
    >>> # Plot concentration profiles
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(result['time_steps'], result['states'][:, 0], label='Cₐ')
    >>> plt.plot(result['time_steps'], result['states'][:, 1], label='Cᵦ')
    >>> plt.xlabel('Time step')
    >>> plt.ylabel('Concentration [mol/L]')
    >>> plt.legend()
    >>> 
    >>> # Design LQR temperature controller
    >>> T_ref = 360.0  # Reference temperature
    >>> x_ref = np.array([0.5, 0.3, T_ref])
    >>> u_ref = reactor._calculate_steady_heating(T_ref)
    >>> 
    >>> Ad, Bd = reactor.linearize(x_ref, u_ref)
    >>> Q_lqr = np.diag([0, 0, 100])  # Only care about temperature
    >>> R_lqr = np.array([[1.0]])
    >>> lqr_result = reactor.control.design_lqr(Ad, Bd, Q_lqr, R_lqr, 
    ...                                          system_type='discrete')
    >>> K = lqr_result['gain']
    >>> 
    >>> # Simulate with LQR control
    >>> def lqr_controller(x, k):
    ...     return -K @ (x - x_ref) + u_ref
    >>> 
    >>> result_lqr = reactor.rollout(x0, lqr_controller, n_steps=100)

    Physical Insights:
    -----------------
    **Reaction Selectivity**:
    Since E₂ > E₁ (default), reaction 2 is more temperature-sensitive.
    This means:
    - Low T: Slow r₂, Cᵦ accumulates (favors intermediate)
    - High T: Fast r₂, Cᵦ depletes quickly (favors product)

    **Temperature Control Strategy**:
    To maximize Cᵦ yield:
    1. Heat initially to accelerate reaction 1 (produce B)
    2. Cool before reaction 2 becomes too fast (preserve B)
    3. Optimal trajectory: heating → plateau → cooling

    **Batch Time vs. Yield Tradeoff**:
    - High temperature: Fast reactions, short batch time, but may
      overshoot optimal Cᵦ (too much conversion to C)
    - Low temperature: Slow reactions, long batch time, but can
      maintain high Cᵦ for longer
    - Economic optimum balances these factors

    **Safety Considerations**:
    - Exothermic reactions (not modeled) can cause thermal runaway
    - High temperature reduces selectivity, may form byproducts
    - Emergency cooling (Q < 0) must be available
    - Temperature constraints critical for safe operation

    See Also:
    --------
    ContinuousBatchReactor : Continuous-time version of this system
    DiscreteCSTR : Continuous stirred-tank reactor (continuous flow)
    LogisticMap : Simpler discrete nonlinear dynamics
    DiscretePendulum : Another discrete nonlinear control problem
    """

    def define_system(
        self,
        k1_val: float = 0.5,
        k2_val: float = 0.3,
        E1_val: float = 1000.0,
        E2_val: float = 1500.0,
        alpha_val: float = 0.1,
        T_amb_val: float = 300.0,
        dt: float = 1.0,
        C_A0: Optional[float] = None,
        T0: Optional[float] = None,
    ):
        """
        Define symbolic discrete-time batch reactor dynamics.

        Parameters
        ----------
        k1_val : float
            Pre-exponential factor for A→B reaction [1/s]
        k2_val : float
            Pre-exponential factor for B→C reaction [1/s]
        E1_val : float
            Activation energy for reaction 1 [K]
        E2_val : float
            Activation energy for reaction 2 [K]
        alpha_val : float
            Heat transfer coefficient [1/s]
        T_amb_val : float
            Ambient temperature [K]
        dt : float
            Discretization time step [s]
        C_A0 : Optional[float]
            Initial concentration of A for equilibrium setup [mol/L]
        T0 : Optional[float]
            Initial temperature for equilibrium setup [K]
        """
        # Store initial conditions for equilibrium setup
        self.C_A0 = C_A0
        self.T0 = T0

        # State variables
        C_A, C_B, T = sp.symbols("C_A C_B T", real=True, positive=True)
        Q = sp.symbols("Q", real=True)

        # Parameters
        k1, k2, E1, E2, alpha, T_amb = sp.symbols(
            "k1 k2 E1 E2 alpha T_amb", real=True, positive=True
        )

        self.parameters = {
            k1: k1_val,
            k2: k2_val,
            E1: E1_val,
            E2: E2_val,
            alpha: alpha_val,
            T_amb: T_amb_val,
        }

        self.state_vars = [C_A, C_B, T]
        self.control_vars = [Q]
        self._dt = dt
        self.order = 1

        # Reaction rates (Arrhenius kinetics)
        # r1 = k1 * C_A * exp(-E1/T)  [A -> B]
        # r2 = k2 * C_B * exp(-E2/T)  [B -> C]
        r1 = k1 * C_A * sp.exp(-E1 / T)
        r2 = k2 * C_B * sp.exp(-E2 / T)

        # Discrete-time dynamics (Euler discretization)
        # Concentration of A decreases by reaction 1
        C_A_next = C_A - dt * r1

        # Concentration of B increases by reaction 1, decreases by reaction 2
        C_B_next = C_B + dt * (r1 - r2)

        # Temperature increases by heating Q, decreases by cooling to ambient
        T_next = T + dt * (Q - alpha * (T - T_amb))

        self._f_sym = sp.Matrix([C_A_next, C_B_next, T_next])

    def setup_equilibria(self):
        """
        Set up equilibrium points for the batch reactor.

        Adds two equilibria:
        1. 'complete': Complete conversion (Cₐ=0, Cᵦ=0, T=Tₐₘᵦ)
        2. 'initial': Optional initial state if C_A0 and T0 specified
        """
        # Get parameter values
        T_amb = self.parameters[sp.symbols("T_amb")]

        # Complete conversion equilibrium (stable, global attractor)
        self.add_equilibrium(
            "complete",
            x_eq=np.array([0.0, 0.0, T_amb]),
            u_eq=np.array([0.0]),
            verify=True,
            stability="stable",
            notes="Complete conversion: all reactants consumed, cooled to ambient",
        )

        # If initial conditions specified, add as reference point
        if self.C_A0 is not None and self.T0 is not None:
            # Calculate required heating to maintain initial temperature
            alpha = self.parameters[sp.symbols("alpha")]
            Q_init = alpha * (self.T0 - T_amb)

            self.add_equilibrium(
                "initial",
                x_eq=np.array([self.C_A0, 0.0, self.T0]),
                u_eq=np.array([Q_init]),
                verify=False,  # Not a true equilibrium (C_A still changes)
                stability="unstable",
                notes=f"Initial fresh batch state with T maintained at {self.T0} K",
            )

            self.set_default_equilibrium("initial")
        else:
            self.set_default_equilibrium("complete")

    def calculate_steady_heating(self, T_setpoint: float) -> float:
        """
        Calculate steady-state heating required to maintain temperature setpoint.

        Parameters
        ----------
        T_setpoint : float
            Desired reactor temperature [K]

        Returns
        -------
        float
            Required heating rate Q [K/s]

        Notes
        -----
        At steady state (constant T), heat input must balance heat loss:
            Q = α·(T - T_amb)
        """
        alpha = self.parameters[sp.symbols("alpha")]
        T_amb = self.parameters[sp.symbols("T_amb")]

        return alpha * (T_setpoint - T_amb)

    def compute_conversion(self, C_A: float, C_A0: float) -> float:
        """
        Compute fractional conversion of reactant A.

        Parameters
        ----------
        C_A : float
            Current concentration of A [mol/L]
        C_A0 : float
            Initial concentration of A [mol/L]

        Returns
        -------
        float
            Conversion fraction X_A (0 = no conversion, 1 = complete)

        Examples
        --------
        >>> reactor = DiscreteBatchReactor()
        >>> X = reactor.compute_conversion(C_A=0.3, C_A0=1.0)
        >>> print(f"Conversion: {X*100:.1f}%")
        Conversion: 70.0%
        """
        return (C_A0 - C_A) / C_A0

    def compute_selectivity(self, C_B: float, C_A: float, C_A0: float) -> float:
        """
        Compute selectivity to intermediate B.

        Parameters
        ----------
        C_B : float
            Current concentration of B [mol/L]
        C_A : float
            Current concentration of A [mol/L]
        C_A0 : float
            Initial concentration of A [mol/L]

        Returns
        -------
        float
            Selectivity S_B = C_B / (C_A0 - C_A) (moles B per mole A converted)

        Notes
        -----
        Selectivity measures how much intermediate B is produced per
        mole of A consumed. Values:
        - S_B = 1.0: Perfect selectivity (all A → B, no B → C yet)
        - S_B < 1.0: Some B has already converted to C
        - S_B → 0: Most B has converted to C (over-reacted)

        Examples
        --------
        >>> reactor = DiscreteBatchReactor()
        >>> S = reactor.compute_selectivity(C_B=0.5, C_A=0.3, C_A0=1.0)
        >>> print(f"Selectivity: {S:.2f} mol B / mol A converted")
        """
        A_consumed = C_A0 - C_A
        if A_consumed < 1e-10:
            return 0.0  # No conversion yet
        return C_B / A_consumed

    def compute_yield(self, C_B: float, C_A0: float) -> float:
        """
        Compute yield of intermediate B.

        Parameters
        ----------
        C_B : float
            Current concentration of B [mol/L]
        C_A0 : float
            Initial concentration of A [mol/L]

        Returns
        -------
        float
            Yield Y_B = C_B / C_A0 (moles B per initial mole A)

        Notes
        -----
        Yield is the most important metric for batch optimization.
        Combines both conversion and selectivity:
            Y_B = X_A · S_B

        Examples
        --------
        >>> reactor = DiscreteBatchReactor()
        >>> Y = reactor.compute_yield(C_B=0.4, C_A0=1.0)
        >>> print(f"Yield: {Y*100:.1f}%")
        Yield: 40.0%
        """
        return C_B / C_A0

    # def print_equations(self, simplify: bool = True):
    #     """
    #     Print symbolic equations using discrete-time notation.

    #     Parameters
    #     ----------
    #     simplify : bool
    #         If True, simplify expressions before printing
    #     """
    #     print("=" * 70)
    #     print(f"{self.__class__.__name__} (Discrete-Time, dt={self.dt} s)")
    #     print("=" * 70)
    #     print(f"State Variables: {self.state_vars}")
    #     print(f"Control Variables: {self.control_vars}")
    #     print(f"Dimensions: nx={self.nx}, nu={self.nu}, ny={self.ny}")
    #     print(f"Sampling Period: {self.dt} s")

    #     # Extract parameter values for display
    #     k1_val = self.parameters[sp.symbols("k1")]
    #     k2_val = self.parameters[sp.symbols("k2")]
    #     E1_val = self.parameters[sp.symbols("E1")]
    #     E2_val = self.parameters[sp.symbols("E2")]
    #     alpha_val = self.parameters[sp.symbols("alpha")]
    #     T_amb_val = self.parameters[sp.symbols("T_amb")]

    #     print("\nPhysical Parameters:")
    #     print(f"  k₁ = {k1_val} 1/s (pre-exponential, A→B)")
    #     print(f"  k₂ = {k2_val} 1/s (pre-exponential, B→C)")
    #     print(f"  E₁ = {E1_val} K (activation energy, A→B)")
    #     print(f"  E₂ = {E2_val} K (activation energy, B→C)")
    #     print(f"  α = {alpha_val} 1/s (heat transfer coefficient)")
    #     print(f"  T_amb = {T_amb_val} K (ambient temperature)")

    #     print("\nChemical Reactions:")
    #     print("  A → B  (first-order, rate r₁ = k₁·Cₐ·exp(-E₁/T))")
    #     print("  B → C  (first-order, rate r₂ = k₂·Cᵦ·exp(-E₂/T))")

    #     print("\nDynamics: x[k+1] = f(x[k], u[k])")
    #     for var, expr in zip(self.state_vars, self._f_sym):
    #         expr_sub = self.substitute_parameters(expr)
    #         if simplify:
    #             expr_sub = sp.simplify(expr_sub)
    #         print(f"  {var}[k+1] = {expr_sub}")

    #     if self._h_sym is not None:
    #         print("\nOutput: y[k] = h(x[k])")
    #         for i, expr in enumerate(self._h_sym):
    #             expr_sub = self.substitute_parameters(expr)
    #             if simplify:
    #                 expr_sub = sp.simplify(expr_sub)
    #             print(f"  y[{i}][k] = {expr_sub}")

    #     print("\nPhysical Interpretation:")
    #     print("  - Cₐ[k]: Concentration of reactant A [mol/L]")
    #     print("  - Cᵦ[k]: Concentration of intermediate B [mol/L]")
    #     print("  - T[k]: Reactor temperature [K]")
    #     print("  - Q[k]: Heating/cooling rate [K/s]")

    #     print("\nTypical Operating Range:")
    #     print("  - Cₐ: 0 - 1.0 mol/L")
    #     print("  - Cᵦ: 0 - 0.5 mol/L")
    #     print("  - T: 300 - 400 K")
    #     print("  - Q: -50 to +50 K/s")

    #     print("=" * 70)

class DiscreteCSTR(DiscreteSymbolicSystem):
    """
    Discrete-time Continuous Stirred-Tank Reactor (CSTR) with cooling jacket.

    Physical System:
    ---------------
    A continuous flow reactor where reactant A converts to product B in
    an exothermic reaction, modeled in discrete time with periodic sampling
    and control actuation.

    Unlike batch reactors, CSTRs operate at steady state with continuous
    feed and product removal. The discrete-time formulation is appropriate
    for digital control systems with:
    - Periodic concentration measurements (e.g., via online analyzers)
    - Discrete temperature sensor readings
    - Digital control actuation of cooling jacket

    Key features:
    - Continuous flow: Feed enters, product exits at same rate
    - Perfect mixing: Uniform concentration and temperature
    - Exothermic reaction: Heat generation from reaction
    - Jacket cooling: Heat removal to maintain temperature
    - Discrete measurements and control updates

    The reactor can exhibit:
    - Multiple steady states (low/high conversion)
    - Oscillatory behavior (limit cycles)
    - Thermal runaway (if cooling insufficient)
    - Complex bifurcations as parameters vary

    State Space:
    -----------
    State: x[k] = [Cₐ[k], T[k]]
        Concentration state:
        - Cₐ: Concentration of reactant A in reactor [mol/L]
          * Lower than feed concentration due to reaction
          * Cₐ,feed > Cₐ > 0
          * Steady-state value depends on temperature and residence time
          * High Cₐ → low conversion (inefficient)
          * Low Cₐ → high conversion (efficient but expensive cooling)

        Temperature state:
        - T: Reactor temperature [K]
          * Higher than feed due to exothermic reaction
          * T > T_feed (for exothermic reactions)
          * Critical state: affects reaction rate exponentially
          * Small T change → large rate change (Arrhenius)
          * Must be controlled to prevent runaway

    Control: u[k] = [T_jacket[k]]
        - T_jacket: Cooling jacket temperature [K]
          * Manipulated variable for temperature control
          * Typically T_jacket < T (removing heat)
          * Can be T_jacket > T for startup heating
          * Typical range: 280-340 K
          * Physical limits: chiller/heater capacity

    Output: y[k] = [Cₐ[k], T[k]]
        - Full state measurement
        - In practice:
          * Cₐ measured via online analyzer (GC, HPLC, spectroscopy)
          * T measured via thermocouple or RTD
          * Both have sampling delays and noise

    Dynamics:
    --------
    The discrete-time dynamics use Euler discretization:

        Cₐ[k+1] = Cₐ[k] + dt·[(F/V)·(Cₐ,feed - Cₐ[k]) - r[k]]
        T[k+1] = T[k] + dt·[(F/V)·(T_feed - T[k]) + (-ΔH/ρCₚ)·r[k] + UA/(VρCₚ)·(T_jacket[k] - T[k])]

    **Reaction Rate (Arrhenius kinetics)**:
        r[k] = k₀·Cₐ[k]·exp(-E/T[k])  [mol/(L·s)]

    where:
    - k₀: Pre-exponential factor [1/s]
    - E: Activation energy [K] (using Eₐ/R as temperature)
    - exp(-E/T): Arrhenius temperature dependence

    **Physical Interpretation**:

    Material Balance:
    - (F/V)·(Cₐ,feed - Cₐ): Convective in/out (dilution)
    - F/V = 1/τ: Inverse residence time [1/s]
    - τ = V/F: Average time molecule spends in reactor [s]
    - -r: Consumption by reaction
    - At steady state: inflow - outflow - reaction = 0

    Energy Balance:
    - (F/V)·(T_feed - T): Convective heat in/out
    - (-ΔH/ρCₚ)·r: Heat generation from reaction
      * ΔH < 0 (exothermic) → heat generation
      * |ΔH| large → strong thermal coupling
    - UA/(VρCₚ)·(T_jacket - T): Heat removal via jacket
      * UA: Overall heat transfer coefficient × area
      * Larger UA → better temperature control
      * T_jacket < T → cooling (typical)

    **Nonlinear Coupling**:
    1. Temperature affects reaction rate exponentially (Arrhenius)
    2. Reaction generates heat (thermal feedback)
    3. High T → fast reaction → more heat → higher T (runaway risk)
    4. Cooling must balance heat generation for stability

    Parameters:
    ----------
    F : float, default=100.0
        Volumetric flow rate [L/s]
        Higher F → shorter residence time → lower conversion
        Lower F → longer residence time → higher conversion

    V : float, default=100.0
        Reactor volume [L]
        Determines residence time τ = V/F
        Larger V → more conversion for given F

    C_A_feed : float, default=1.0
        Feed concentration [mol/L]
        Typical: 0.5-2.0 mol/L
        Higher feed → more product but more heat generation

    T_feed : float, default=350.0
        Feed temperature [K]
        Typical: 300-360 K
        Pre-heating can improve conversion but reduces stability margin

    k0 : float, default=7.2e10
        Pre-exponential factor [1/s]
        Collision frequency in Arrhenius equation
        Typical: 10⁶-10¹² for liquid phase reactions
        Determines reaction speed at given temperature

    E : float, default=8750.0
        Activation energy [K] (actually Eₐ/R)
        Energy barrier for reaction to occur
        Typical: 5000-15000 K for Eₐ/R
        Higher E → more temperature-sensitive
        Physical Eₐ typically 40-120 kJ/mol

    delta_H : float, default=-5e4
        Heat of reaction [J/mol]
        Negative = exothermic (releases heat)
        Positive = endothermic (absorbs heat)
        Typical for exothermic: -20,000 to -200,000 J/mol
        Larger |ΔH| → stronger thermal coupling, harder control

    rho : float, default=1000.0
        Density [kg/L]
        Typical for aqueous solutions: 900-1100 kg/L
        Affects thermal inertia (heat capacity)

    Cp : float, default=0.239
        Specific heat capacity [J/(kg·K)]
        Typical for aqueous: 0.2-0.5 J/(kg·K)
        Higher Cₚ → slower temperature changes (more stable)

    UA : float, default=5e4
        Overall heat transfer coefficient × area [J/(s·K)]
        Combines jacket film coefficient, wall conduction, reactor-side film
        Typical: 10³-10⁵ J/(s·K)
        Higher UA → better temperature control, faster cooling
        Limited by physical design (jacket size, flow rate)

    dt : float, default=0.1
        Sampling/discretization time step [s]
        Critical parameter for stability!
        - Too large → numerical instability, oscillations
        - Too small → slow simulation, high computational cost
        - Rule of thumb: dt < 0.1/max(λ) where λ are eigenvalues
        - Typical for CSTR: 0.01-1.0 seconds
        - Must be smaller than fastest system time scale

    Equilibria:
    ----------
    **Multiple Steady States** (hallmark of CSTRs!):

    CSTR can have 1, 2, or 3 steady states depending on parameters.
    For given feed conditions and jacket temperature:

    1. **Low conversion state** (stable):
       - Low T ≈ T_feed + small rise
       - Low reaction rate (slow kinetics)
       - High Cₐ ≈ Cₐ,feed (minimal conversion)
       - Heat generation < Heat removal
       - Easy to control but inefficient
       - Attractive for cold startup

    2. **High conversion state** (stable):
       - High T >> T_feed
       - High reaction rate (fast kinetics)
       - Low Cₐ << Cₐ,feed (high conversion)
       - Heat generation balanced by cooling
       - Desirable operating point (efficient)
       - Risk: close to instability/runaway

    3. **Intermediate state** (unstable):
       - Saddle point in phase space
       - Not physically realizable (unstable)
       - Forms separatrix between basins of attraction
       - System will move toward stable states

    **Stability depends on**:
    - Residence time τ = V/F (longer → more conversion, less stable)
    - Activation energy E (higher → more sensitive)
    - Heat of reaction ΔH (larger |ΔH| → more coupling)
    - Cooling capacity UA (higher → more stable)
    - Feed temperature T_feed (higher → less stable margin)

    **Bifurcation Behavior**:
    As cooling capacity (T_jacket) decreases:
    1. Unique stable high-conversion state
    2. Saddle-node bifurcation → 3 steady states appear
    3. Two stable states (low and high conversion)
    4. Another saddle-node → only low-conversion state
    5. Further decrease → thermal runaway (no steady state)

    Control Objectives:
    ------------------
    Common control goals for CSTR:

    1. **Setpoint tracking**: Maintain T[k] ≈ T_setpoint
       - Most common objective
       - Balances conversion and stability
       - PID/LQR/MPC controllers typical
       - Challenge: nonlinearity and multiple steady states

    2. **Startup control**: Transition low → high conversion state
       - Must cross unstable intermediate state
       - Requires large transient cooling capacity
       - Bang-bang or optimal trajectory control
       - Risk of overshoot → runaway

    3. **Disturbance rejection**: Handle feed variations
       - Feed concentration changes: Cₐ,feed(t)
       - Feed temperature disturbances: T_feed(t)
       - Flow rate fluctuations: F(t)
       - Jacket temperature limits

    4. **Optimal operation**: Maximize profit
       - Balance conversion (revenue) vs cooling cost
       - Economic objective: J = price·F·(Cₐ,feed - Cₐ) - cooling_cost
       - Constraint: T_max safety limit
       - May operate near instability for profit

    5. **Runaway prevention**: Safety constraint
       - Monitor temperature rate: dT/dt < threshold
       - Emergency cooling if T > T_max
       - May require batch shutdown

    State Constraints:
    -----------------
    Physical constraints that must be enforced:

    1. **Non-negativity**: Cₐ[k] ≥ 0
       - Concentration cannot be negative
       - Physical meaning: species present or absent
       - Euler discretization may violate if dt too large

    2. **Concentration bounds**: 0 ≤ Cₐ[k] ≤ Cₐ,feed
       - Cannot exceed feed concentration (dilution + reaction)
       - Upper bound: Cₐ ≤ Cₐ,feed (no reaction case)
       - Useful for validation

    3. **Temperature limits**: T_min ≤ T[k] ≤ T_max
       - Safety: prevent runaway (T_max ≈ 450-500 K)
       - Operability: prevent freezing/solidification (T_min ≈ 280 K)
       - Jacket temperature limits: T_jacket,min ≤ T_jacket ≤ T_jacket,max
       - Typical limits: 280 K ≤ T ≤ 450 K

    4. **Jacket temperature constraints**: T_jacket,min ≤ T_jacket[k] ≤ T_jacket,max
       - Physical cooling/heating capacity
       - Chiller: T_jacket,min ≈ 280 K
       - Heater: T_jacket,max ≈ 400 K
       - Rate limit: |T_jacket[k+1] - T_jacket[k]| ≤ ΔT_jacket,max

    Numerical Considerations:
    ------------------------
    **Stability**: The explicit Euler discretization is stable if:
        dt < 2/λ_max

    where λ_max is the maximum eigenvalue of the Jacobian.

    For CSTR, typical eigenvalues:
        λ₁ ≈ -(1/τ + k₀·exp(-E/T))  (concentration dynamics)
        λ₂ ≈ -(1/τ + UA/(VρCₚ))  (temperature dynamics)

    At high temperature, λ_max can be large (fast dynamics), requiring
    small dt for stability.

    **Rule of thumb**:
        dt < 0.1 · min(τ, VρCₚ/UA, 1/(k₀·exp(-E/T)))

    **Stiffness**: CSTR is moderately stiff due to:
    - Fast reaction at high temperature
    - Different time scales (concentration vs temperature)
    - Exponential temperature dependence

    For better accuracy, use higher-order discretization:
```python
        cstr_continuous = ContinuousCSTR(...)
        cstr_discrete = cstr_continuous.discretize(dt=0.1, method='rk4')
```

    **Multiple Steady States**: Discrete system inherits multiple equilibria
    from continuous system. Simulation starting point determines which
    equilibrium is reached (basin of attraction).

    Example Usage:
    -------------
    >>> # Create CSTR with default parameters
    >>> cstr = DiscreteCSTR(dt=0.1)
    >>> 
    >>> # High conversion steady state (typical operating point)
    >>> x_high = np.array([0.1, 390.0])  # [Low Cₐ, High T]
    >>> u_high = np.array([350.0])  # [Cool jacket temperature]
    >>> 
    >>> # Verify it's an equilibrium
    >>> x_next = cstr.step(x_high, u_high)
    >>> print(f"Change: {np.linalg.norm(x_next - x_high):.2e}")  # Should be small
    >>> 
    >>> # Simulate with constant cooling
    >>> result = cstr.simulate(
    ...     x0=x_high,
    ...     u_sequence=np.array([350.0]),  # Constant jacket temp
    ...     n_steps=100
    ... )
    >>> 
    >>> # Plot concentration and temperature
    >>> import matplotlib.pyplot as plt
    >>> fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    >>> axes[0].plot(result['time_steps'] * cstr.dt, result['states'][:, 0])
    >>> axes[0].set_ylabel('Cₐ [mol/L]')
    >>> axes[1].plot(result['time_steps'] * cstr.dt, result['states'][:, 1])
    >>> axes[1].set_ylabel('T [K]')
    >>> axes[1].set_xlabel('Time [s]')
    >>> 
    >>> # Design LQR controller for temperature regulation
    >>> T_setpoint = 390.0
    >>> C_A_setpoint = 0.1
    >>> x_ref = np.array([C_A_setpoint, T_setpoint])
    >>> 
    >>> # Calculate required jacket temperature for steady state
    >>> # This requires solving energy balance, simplified here
    >>> u_ref = np.array([350.0])
    >>> 
    >>> # Linearize at operating point
    >>> Ad, Bd = cstr.linearize(x_ref, u_ref)
    >>> 
    >>> # Check discrete stability
    >>> eigenvalues = np.linalg.eigvals(Ad)
    >>> print(f"Eigenvalues: {eigenvalues}")
    >>> print(f"Stable: {np.all(np.abs(eigenvalues) < 1)}")
    >>> 
    >>> # Design LQR (care more about temperature than concentration)
    >>> Q_lqr = np.diag([1.0, 100.0])  # Penalize temperature error heavily
    >>> R_lqr = np.array([[1.0]])
    >>> lqr_result = cstr.control.design_lqr(Ad, Bd, Q_lqr, R_lqr, 
    ...                                       system_type='discrete')
    >>> K = lqr_result['gain']
    >>> 
    >>> # Simulate with LQR control
    >>> def lqr_controller(x, k):
    ...     return -K @ (x - x_ref) + u_ref
    >>> 
    >>> result_lqr = cstr.rollout(x_high, lqr_controller, n_steps=200)
    >>> 
    >>> # Startup simulation: low conversion → high conversion
    >>> x_low = np.array([0.9, 355.0])  # Low conversion state
    >>> 
    >>> def startup_controller(x, k):
    ...     # Aggressive cooling to reach high-conversion state
    ...     if k < 50:
    ...         return np.array([340.0])  # Strong cooling
    ...     else:
    ...         return lqr_controller(x, k)  # Switch to regulator
    >>> 
    >>> result_startup = cstr.rollout(x_low, startup_controller, n_steps=200)
    >>> 
    >>> # Check if startup was successful
    >>> final_state = result_startup['states'][-1, :]
    >>> distance_to_target = np.linalg.norm(final_state - x_ref)
    >>> print(f"Final state: Cₐ={final_state[0]:.3f}, T={final_state[1]:.1f}")
    >>> print(f"Distance to target: {distance_to_target:.3f}")

    Physical Insights:
    -----------------
    **Thermal Runaway Risk**:
    If cooling is insufficient, positive feedback occurs:
    1. Temperature increases
    2. Reaction rate increases exponentially (Arrhenius)
    3. More heat generated (exothermic)
    4. Temperature increases further → runaway!

    Prevention:
    - Adequate cooling capacity (large UA)
    - Temperature limits and alarms
    - Emergency cooling/shutdown procedures
    - Conservative setpoint selection

    **Multiple Steady States**:
    Creates control challenges:
    - Which steady state to operate at?
    - How to transition between states?
    - Risk of unintended switching due to disturbances
    - Hysteresis in startup/shutdown procedures

    **Residence Time Effects**:
    - Short τ (high F/V): Low conversion, stable, safe
    - Long τ (low F/V): High conversion, less stable, runaway risk
    - Economic optimum: maximize profit subject to stability

    **Jacket Temperature Selection**:
    - Lower T_jacket: More cooling, enables higher conversion
    - But: smaller stability margin, closer to bifurcation
    - Higher T_jacket: More stable, but lower conversion
    - Must balance economics and safety

    **Startup Strategy**:
    Transitioning from low to high conversion:
    1. Begin at low-conversion state (safe, stable)
    2. Gradually decrease T_jacket (increase cooling)
    3. System may jump to high-conversion state (bifurcation)
    4. Or use aggressive transient cooling
    5. Once at high conversion, switch to regulatory control

    **Oscillatory Behavior**:
    Near instability boundaries, system may exhibit:
    - Sustained oscillations (Hopf bifurcation)
    - Limit cycles in Cₐ-T phase plane
    - Period-doubling route to chaos (rare but possible)
    - Quasiperiodic dynamics

    Comparison with Continuous Version:
    ----------------------------------
    This discrete-time CSTR approximates the continuous-time system:
    - Continuous system: dx/dt = f(x, u) (ground truth)
    - Discrete system: x[k+1] = x[k] + dt·f(x[k], u[k]) (Euler approximation)

    Advantages of discrete formulation:
    - Natural for digital control (computers, PLCs)
    - Fixed time step (predictable computation)
    - Easy to implement in simulation
    - Matches physical sampling of sensors

    Disadvantages:
    - Approximation error (depends on dt)
    - Stability limited by time step
    - May not capture fast transients accurately

    For better accuracy, create from continuous version:
```python
    cstr_continuous = ContinuousCSTR(F=100, V=100, ...)
    cstr_discrete = cstr_continuous.discretize(dt=0.1, method='rk4')
```

    See Also:
    --------
    ContinuousCSTR : Continuous-time version (more accurate)
    DiscreteBatchReactor : Batch operation instead of continuous
    ContinuousBatchReactor : Continuous batch reactor
    """

    def define_system(
        self,
        F_val: float = 100.0,  # Flow rate [L/s]
        V_val: float = 100.0,  # Volume [L]
        C_A_feed_val: float = 1.0,  # Feed concentration [mol/L]
        T_feed_val: float = 350.0,  # Feed temperature [K]
        k0_val: float = 7.2e10,  # Pre-exponential [1/s]
        E_val: float = 8750.0,  # Activation energy [K]
        delta_H_val: float = -5e4,  # Heat of reaction [J/mol]
        rho_val: float = 1000.0,  # Density [kg/L]
        Cp_val: float = 0.239,  # Heat capacity [J/(kg*K)]
        UA_val: float = 5e4,  # Heat transfer coef [J/(s*K)]
        dt: float = 0.1,
        x_ss: Optional[np.ndarray] = None,
        u_ss: Optional[np.ndarray] = None,
    ):
        """
        Define discrete-time CSTR dynamics.

        Parameters
        ----------
        F_val : float
            Volumetric flow rate [L/s]
        V_val : float
            Reactor volume [L]
        C_A_feed_val : float
            Feed concentration [mol/L]
        T_feed_val : float
            Feed temperature [K]
        k0_val : float
            Pre-exponential factor [1/s]
        E_val : float
            Activation energy [K]
        delta_H_val : float
            Heat of reaction [J/mol] (negative = exothermic)
        rho_val : float
            Density [kg/L]
        Cp_val : float
            Specific heat capacity [J/(kg·K)]
        UA_val : float
            Overall heat transfer coefficient × area [J/(s·K)]
        dt : float
            Sampling time step [s]
        x_ss : Optional[np.ndarray]
            Steady-state [Cₐ, T] for equilibrium setup
        u_ss : Optional[np.ndarray]
            Steady-state [T_jacket] for equilibrium setup
        """
        self.x_ss = x_ss
        self.u_ss = u_ss

        # State and control variables
        C_A, T = sp.symbols("C_A T", real=True, positive=True)
        T_jacket = sp.symbols("T_jacket", real=True, positive=True)

        # Parameters
        F, V, C_A_feed, T_feed = sp.symbols("F V C_A_feed T_feed", real=True, positive=True)
        k0, E, delta_H, rho, Cp, UA = sp.symbols(
            "k0 E delta_H rho Cp UA", real=True, positive=True
        )

        self.parameters = {
            F: F_val,
            V: V_val,
            C_A_feed: C_A_feed_val,
            T_feed: T_feed_val,
            k0: k0_val,
            E: E_val,
            delta_H: delta_H_val,
            rho: rho_val,
            Cp: Cp_val,
            UA: UA_val,
        }

        self.state_vars = [C_A, T]
        self.control_vars = [T_jacket]
        self._dt = dt
        self.order = 1

        # Reaction rate (Arrhenius kinetics)
        r = k0 * C_A * sp.exp(-E / T)

        # Discrete-time dynamics (Euler discretization)
        # Material balance
        C_A_next = C_A + dt * ((F / V) * (C_A_feed - C_A) - r)

        # Energy balance
        T_next = (
            T
            + dt * ((F / V) * (T_feed - T))
            + dt * ((-delta_H) / (rho * Cp)) * r
            + dt * (UA / (V * rho * Cp)) * (T_jacket - T)
        )

        self._f_sym = sp.Matrix([C_A_next, T_next])

    def setup_equilibria(self):
        """
        Set up steady-state equilibrium if provided.

        Notes
        -----
        CSTR can have multiple steady states! Only add user-provided
        equilibrium. Finding all equilibria requires solving nonlinear
        algebraic equations (see find_steady_states() method).
        """
        if self.x_ss is not None and self.u_ss is not None:
            self.add_equilibrium(
                "steady_state",
                x_eq=self.x_ss,
                u_eq=self.u_ss,
                verify=True,
                stability="unknown",
                notes="User-provided steady state - CSTR may have multiple equilibria",
            )
            self.set_default_equilibrium("steady_state")

    def compute_conversion(self, C_A: float, C_A_feed: float) -> float:
        """
        Compute fractional conversion of reactant A.

        Parameters
        ----------
        C_A : float
            Current reactor concentration [mol/L]
        C_A_feed : float
            Feed concentration [mol/L]

        Returns
        -------
        float
            Conversion fraction X_A = (C_A_feed - C_A) / C_A_feed

        Examples
        --------
        >>> cstr = DiscreteCSTR()
        >>> X = cstr.compute_conversion(C_A=0.1, C_A_feed=1.0)
        >>> print(f"Conversion: {X*100:.1f}%")
        Conversion: 90.0%

        Notes
        -----
        High conversion (X > 0.9) typically corresponds to high-temperature
        steady state with fast kinetics and strong exothermic heat generation.
        """
        return (C_A_feed - C_A) / C_A_feed

    def compute_residence_time(self) -> float:
        """
        Compute residence time τ = V/F.

        Returns
        -------
        float
            Residence time [s]

        Examples
        --------
        >>> cstr = DiscreteCSTR(F=100.0, V=100.0)
        >>> tau = cstr.compute_residence_time()
        >>> print(f"Residence time: {tau} s")
        Residence time: 1.0 s

        Notes
        -----
        Residence time is the average time a molecule spends in the reactor.
        - Longer τ (smaller F): More conversion, less stable
        - Shorter τ (larger F): Less conversion, more stable
        """
        F = self.parameters[sp.symbols("F")]
        V = self.parameters[sp.symbols("V")]
        return V / F

    def compute_damkohler_number(self, T: float) -> float:
        """
        Compute Damköhler number Da = k·τ (reaction rate × residence time).

        Parameters
        ----------
        T : float
            Temperature [K]

        Returns
        -------
        float
            Damköhler number [dimensionless]

        Notes
        -----
        Damköhler number measures reaction rate relative to flow rate:
        - Da << 1: Reaction slow, flow dominates, low conversion
        - Da >> 1: Reaction fast, kinetics dominate, high conversion
        - Da ≈ 1: Balanced, optimal efficiency

        Examples
        --------
        >>> cstr = DiscreteCSTR()
        >>> Da_low = cstr.compute_damkohler_number(T=350.0)
        >>> Da_high = cstr.compute_damkohler_number(T=400.0)
        >>> print(f"Da(350K) = {Da_low:.2f}")
        >>> print(f"Da(400K) = {Da_high:.2f}")
        """
        k0 = self.parameters[sp.symbols("k0")]
        E = self.parameters[sp.symbols("E")]
        tau = self.compute_residence_time()

        k = k0 * np.exp(-E / T)
        return k * tau

    def find_steady_states(
        self,
        T_jacket: float,
        T_range: tuple = (300.0, 500.0),
        n_points: int = 100,
    ) -> list:
        """
        Find all steady states for a given jacket temperature.

        Uses graphical method: plots dC_A/dt and dT/dt as functions of T,
        finds where both are zero simultaneously.

        Parameters
        ----------
        T_jacket : float
            Jacket temperature [K]
        T_range : tuple
            Temperature range to search (T_min, T_max) [K]
        n_points : int
            Number of points for graphical search

        Returns
        -------
        list
            List of (C_A, T) steady states

        Examples
        --------
        >>> cstr = DiscreteCSTR()
        >>> steady_states = cstr.find_steady_states(T_jacket=350.0)
        >>> print(f"Found {len(steady_states)} steady states")
        >>> for i, (C_A, T) in enumerate(steady_states):
        ...     print(f"  State {i+1}: C_A={C_A:.3f}, T={T:.1f}")

        Notes
        -----
        This is a simple implementation. For production code, use:
        - scipy.optimize.fsolve for more robust root finding
        - Continuation methods for bifurcation analysis
        - homotopy methods for finding all solutions
        """
        from scipy.optimize import fsolve

        # Extract parameters
        F = self.parameters[sp.symbols("F")]
        V = self.parameters[sp.symbols("V")]
        C_A_feed = self.parameters[sp.symbols("C_A_feed")]
        T_feed = self.parameters[sp.symbols("T_feed")]
        k0 = self.parameters[sp.symbols("k0")]
        E = self.parameters[sp.symbols("E")]
        delta_H = self.parameters[sp.symbols("delta_H")]
        rho = self.parameters[sp.symbols("rho")]
        Cp = self.parameters[sp.symbols("Cp")]
        UA = self.parameters[sp.symbols("UA")]

        def steady_state_equations(state):
            """Steady state: dC_A/dt = 0, dT/dt = 0"""
            C_A, T = state
            r = k0 * C_A * np.exp(-E / T)

            # Material balance
            dC_A_dt = (F / V) * (C_A_feed - C_A) - r

            # Energy balance
            dT_dt = (
                (F / V) * (T_feed - T)
                + ((-delta_H) / (rho * Cp)) * r
                + (UA / (V * rho * Cp)) * (T_jacket - T)
            )

            return [dC_A_dt, dT_dt]

        # Try multiple initial guesses across temperature range
        steady_states = []
        T_guesses = np.linspace(T_range[0], T_range[1], n_points)

        for T_guess in T_guesses:
            # Estimate C_A from material balance at this T
            r_guess = k0 * C_A_feed * np.exp(-E / T_guess)
            C_A_guess = C_A_feed / (1 + (V / F) * r_guess / C_A_feed)
            C_A_guess = np.clip(C_A_guess, 0.0, C_A_feed)

            try:
                solution, info, ier, msg = fsolve(
                    steady_state_equations,
                    [C_A_guess, T_guess],
                    full_output=True,
                )

                if ier == 1:  # Solution found
                    C_A_sol, T_sol = solution

                    # Check if solution is physical and unique
                    if (
                        0 <= C_A_sol <= C_A_feed
                        and T_range[0] <= T_sol <= T_range[1]
                        and not any(
                            np.allclose([C_A_sol, T_sol], ss, rtol=1e-3) for ss in steady_states
                        )
                    ):
                        steady_states.append((C_A_sol, T_sol))

            except Exception:
                continue

        return steady_states

    # def print_equations(self, simplify: bool = True):
    #     """
    #     Print symbolic equations using discrete-time notation.

    #     Parameters
    #     ----------
    #     simplify : bool
    #         If True, simplify expressions before printing
    #     """
    #     print("=" * 70)
    #     print(f"{self.__class__.__name__} (Discrete-Time, dt={self.dt} s)")
    #     print("=" * 70)
    #     print("Continuous Stirred-Tank Reactor with Cooling Jacket")
    #     print("\nReaction: A → B (exothermic)")
    #     print(f"\nState: x = [Cₐ, T]")
    #     print(f"Control: u = [T_jacket]")
    #     print(f"Dimensions: nx={self.nx}, nu={self.nu}")
    #     print(f"Sampling Period: {self.dt} s")

    #     # Calculate and display characteristic parameters
    #     tau = self.compute_residence_time()
    #     print(f"\nCharacteristic Parameters:")
    #     print(f"  Residence time τ = V/F = {tau:.2f} s")
    #     print(f"  Damköhler number Da(350K) = {self.compute_damkohler_number(350.0):.2f}")
    #     print(f"  Damköhler number Da(400K) = {self.compute_damkohler_number(400.0):.2f}")

    #     print("\nDynamics: x[k+1] = f(x[k], u[k])")
    #     for var, expr in zip(self.state_vars, self._f_sym):
    #         expr_sub = self.substitute_parameters(expr)
    #         if simplify:
    #             expr_sub = sp.simplify(expr_sub)
    #         print(f"  {var}[k+1] = {expr_sub}")

    #     print("\nPhysical Interpretation:")
    #     print("  - Cₐ: Reactor concentration [mol/L]")
    #     print("  - T: Reactor temperature [K]")
    #     print("  - T_jacket: Cooling jacket temperature [K]")

    #     print("\nTypical Operating Range:")
    #     C_A_feed = self.parameters[sp.symbols("C_A_feed")]
    #     print(f"  - Cₐ: 0 - {C_A_feed} mol/L")
    #     print("  - T: 350 - 450 K")
    #     print("  - T_jacket: 280 - 360 K")

    #     print("\nStability Note:")
    #     print("  ⚠ CSTR can have MULTIPLE steady states!")
    #     print("  Use find_steady_states() to locate all equilibria.")

    #     print("=" * 70)