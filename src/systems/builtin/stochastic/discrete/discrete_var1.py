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
Discrete VAR Process - Vector Autoregressive Models
====================================================

This module provides Vector Autoregressive (VAR) models, the fundamental
framework for analyzing multiple interrelated time series. VAR processes
serve as:

- The multivariate generalization of AR(p) models
- The discrete-time analog of multivariate Ornstein-Uhlenbeck processes
- The foundation for Granger causality testing (Nobel Prize 2003)
- A benchmark for macroeconomic forecasting and policy analysis
- The basis for cointegration analysis and error correction models

VAR models extend univariate AR by allowing:
1. **Multiple variables:** Model systems of time series jointly
2. **Cross-effects:** One variable influences another's future
3. **Dynamic interactions:** Capture feedback and spillovers
4. **Correlation:** Model contemporaneous relationships
5. **Impulse responses:** Trace effects of shocks through system

The VAR(1) model is the simplest multivariate autoregressive process,
using one lag to capture dynamics in systems with interdependent variables.

Mathematical Background
-----------------------

**Univariate AR(1):**
    x[k] = φ·x[k-1] + w[k]

Single variable, no interaction with others.

**Multivariate Extension - VAR(1):**
    X[k] = A·X[k-1] + w[k]

where:
- X ∈ ℝⁿ: Vector of n variables
- A ∈ ℝⁿˣⁿ: Coefficient matrix (captures all interactions)
- w ∈ ℝⁿ: Vector white noise with covariance Σ_w

**Key Insight:**
Off-diagonal elements of A capture cross-variable effects:
- A_ij: Effect of X_j[k-1] on X_i[k]
- Dynamic interdependence built into structure

Mathematical Formulation
------------------------

**VAR(1) Process:**

Standard form:
    X[k] = A·X[k-1] + w[k]

where:
    - X[k] ∈ ℝⁿ: State vector at time k
    - A ∈ ℝⁿˣⁿ: Coefficient matrix (autoregressive)
    - w[k] ~ N(0, Σ_w): Vector white noise
    - Σ_w ∈ ℝⁿˣⁿ: Innovation covariance (symmetric, positive definite)

**Component Form:**
    X_i[k] = Σⱼ A_ij·X_j[k-1] + w_i[k]

Each variable is linear combination of all lagged variables.

**With Control:**
    X[k] = A·X[k-1] + B·u[k] + w[k]

Adds control/exogenous inputs u ∈ ℝᵖ via matrix B ∈ ℝⁿˣᵖ.

**Non-Centered Form:**
    X[k] = c + A·X[k-1] + w[k]

where c ∈ ℝⁿ is constant vector (allows non-zero mean).

**Mean Deviation Form:**
    (X[k] - μ) = A·(X[k-1] - μ) + w[k]

where μ = (I - A)⁻¹·c is the mean vector.

Relationship to Multivariate OU
--------------------------------

**Continuous Multivariate OU:**
    dX = Ā·X·dt + Σ̄·dW

**Exact Discretization (Δt sampling):**
    X[k] = Φ·X[k-1] + w[k]

where:
    Φ = exp(Ā·Δt) (matrix exponential)
    Cov[w] = ∫₀^Δt exp(Ā·s)·Σ̄·Σ̄ᵀ·exp(Āᵀ·s)·ds

**Connection:**
VAR(1) is discrete-time equivalent of multivariate OU.

**Conversion:**
From VAR to continuous (approximate for small Δt):
    Ā ≈ (A - I)/Δt

Analytical Properties
---------------------

**Stationarity:**

VAR(1) is stationary if and only if all eigenvalues of A satisfy |λ_i| < 1.

Equivalently: All eigenvalues inside unit circle in complex plane.

**Stationary Mean:**
For centered VAR (c = 0): E[X[k]] = 0

For non-centered: E[X[k]] = (I - A)⁻¹·c

**Stationary Covariance:**

Satisfies discrete Lyapunov equation:
    Γ(0) = A·Γ(0)·Aᵀ + Σ_w

Solution:
    vec(Γ(0)) = (I - A ⊗ A)⁻¹·vec(Σ_w)

where ⊗ is Kronecker product, vec() vectorizes matrix.

**Autocovariance:**
    Γ(h) = A·Γ(h-1) = Aʰ·Γ(0) for h ≥ 1

Geometric decay controlled by eigenvalues of A.

**Cross-Correlation:**
    Corr[X_i[k], X_j[k+h]] = Γ_ij(h)/√(Γ_ii(0)·Γ_jj(0))

Can be asymmetric: Corr[X_i, X_j[+h]] ≠ Corr[X_j, X_i[+h]]

Granger Causality
-----------------

**Fundamental Concept (Nobel Prize 2003):**

**Definition:**
X_j "Granger-causes" X_i if past values of X_j improve prediction of X_i
beyond what X_i's own past provides.

**Mathematical Test:**

For VAR(1): X_j Granger-causes X_i if A_ij ≠ 0

More generally: Regress X_i on:
- Restricted: Only lags of X_i
- Unrestricted: Lags of X_i AND X_j

F-test: Does adding X_j lags significantly reduce forecast error?

**Example (2D VAR):**
    X₁[k] = a₁₁·X₁[k-1] + a₁₂·X₂[k-1] + w₁[k]
    X₂[k] = a₂₁·X₁[k-1] + a₂₂·X₂[k-1] + w₂[k]

Tests:
- X₂ → X₁: Is a₁₂ ≠ 0? (X₂ causes X₁)
- X₁ → X₂: Is a₂₁ ≠ 0? (X₁ causes X₂)

**Possibilities:**
- Unidirectional: X₁ → X₂ but not X₂ → X₁
- Bidirectional: X₁ ↔ X₂ (feedback)
- No causality: a₁₂ = a₂₁ = 0 (independent)

**Applications:**
- Monetary policy: Does money supply cause inflation?
- Markets: Do stock prices cause trading volume?
- Macroeconomics: Does consumption cause investment?

**Caution:**
Granger causality ≠ true causality!
- Statistical, not causal
- Can be spurious (omitted variables)
- Temporal precedence, not mechanism

Impulse Response Analysis
--------------------------

**Impulse Response Function (IRF):**

Effect of one-time shock to variable j on variable i over time.

**Mathematical Definition:**

Shock: w_j[0] = 1, all other w = 0

Response: X_i[h] for h = 0, 1, 2, ...

**Computation:**

From moving average representation:
    X[k] = Σ Ψ_h·w[k-h]

where Ψ_h = Aʰ (h-step impulse response matrix).

**Interpretation:**
IRF_ij(h) = (Ψ_h)_ij = effect of 1 unit shock to j on i after h periods.

**Applications:**

1. **Monetary Policy:**
   - Shock: Interest rate increase
   - Trace: Effect on GDP, inflation, employment

2. **Oil Price Shock:**
   - Shock: Oil price increase
   - Trace: Effect on inflation, output, stock market

3. **Technology Shock:**
   - Shock: Productivity innovation
   - Trace: Effect on investment, consumption, wages

**Structural VAR (SVAR):**

Identify contemporaneous relationships:
    A₀·X[k] = A₁·X[k-1] + w[k]

Impose restrictions (economic theory) to identify structural shocks.

Forecast Error Variance Decomposition
--------------------------------------

**Question:** What fraction of forecast error variance for X_i is due to
shocks in X_j?

**h-Step Forecast Error:**
    X[k+h] - X̂[k+h|k] = Σⱼ₌₀^(h-1) Ψ_j·w[k+h-j]

**Variance Decomposition:**

Fraction of h-step forecast error variance for X_i due to shocks in w_j:
    ω_ij(h) = Σₛ₌₀^(h-1) (Ψ_s)²_ij / Σⱼ Σₛ₌₀^(h-1) (Ψ_s)²_ij

**Interpretation:**
- ω_ij(h) = 0.7: 70% of forecast error for X_i due to X_j shocks
- Track over horizons: Short vs long term attribution

Applications
------------

**1. Macroeconomic Forecasting:**

**Central Banks:**
- GDP, inflation, interest rates jointly
- Policy scenario analysis
- Economic projections

**Business Cycle:**
- Output, employment, investment
- Leading indicators
- Recession prediction

**2. Financial Markets:**

**Multi-Asset Dynamics:**
- Stock indices (S&P 500, NASDAQ, international)
- Bond yields (short, medium, long)
- Exchange rates (EUR/USD, GBP/USD, etc.)

**Volatility Spillovers:**
- VIX, realized volatility across assets
- Contagion analysis
- Systemic risk

**3. International Economics:**

**Trade Dynamics:**
- Imports, exports across countries
- Exchange rate effects
- Global value chains

**Policy Spillovers:**
- US Fed → ECB → BoJ
- Transmission mechanisms
- Currency wars

**4. Energy Markets:**

**Commodity System:**
- Oil, gas, coal prices
- Renewable generation
- Demand elasticities

**5. Environmental:**

**Climate Variables:**
- Temperature, precipitation across regions
- Spatial dependence
- Extreme event propagation

**6. Neuroscience:**

**Multi-Electrode Recordings:**
- Neural activity across brain regions
- Effective connectivity
- Information flow

Numerical Simulation
--------------------

**Direct Evaluation:**
    X[k+1] = A·X[k] + w[k]

where w[k] ~ N(0, Σ_w).

**Algorithm:**
```python
X = np.zeros((N+1, n))
X[0] = X0

for k in range(N):
    w_k = np.random.multivariate_normal(np.zeros(n), Sigma_w)
    X[k+1] = A @ X[k] + w_k
```

**Efficient (Vectorized):**
Generate all innovations first, use matrix operations.

Parameter Estimation
--------------------

**Ordinary Least Squares (OLS):**

Equation-by-equation:
For each i, regress X_i[k] on X₁[k-1], ..., X_n[k-1]

Advantage: Simple, fast
Disadvantage: Ignores cross-equation restrictions

**Multivariate Least Squares:**
Estimate all equations jointly.

**Maximum Likelihood:**
Gaussian likelihood (if w ~ N(0, Σ_w)):
    ℓ(A, Σ_w) = -(Nn/2)·ln(2π) - (N/2)·ln|Σ_w| - (1/2)·ΣTrace(Σ_w⁻¹·ε·εᵀ)

**Bayesian:**
Prior on A, Σ_w (Minnesota prior common).

Common Pitfalls
---------------

1. **Overparameterization:**
   - VAR(1) with n variables: n² + n(n+1)/2 parameters!
   - n = 10: 155 parameters
   - Easy to overfit

2. **Spurious Regression:**
   - Non-stationary variables (unit roots)
   - Can find spurious relationships
   - Test for unit roots first (multivariate ADF)

3. **Lag Selection:**
   - Using VAR(1) when VAR(p) needed
   - Information criteria (AIC, BIC)
   - Or theory-driven

4. **Identification:**
   - Correlation ≠ causation
   - Granger causality ≠ true causality
   - Need theory for structural interpretation

5. **Stability:**
   - Must check eigenvalues of A
   - All |λ_i| < 1 for stationarity
   - Can have complex eigenvalues (oscillations)

6. **Small Sample Bias:**
   - OLS biased in small samples (especially with persistence)
   - Bias correction or bootstrap

**Impact:**
VAR demonstrated:
- Theory-free modeling can work (atheoretical approach)
- Multivariate essential (univariate misses interactions)
- Granger causality useful (even if not true causality)

"""

import numpy as np
import sympy as sp
from typing import Optional, Union, List, Tuple

from src.systems.base.core.discrete_stochastic_system import DiscreteStochasticSystem


class DiscreteVAR1(DiscreteStochasticSystem):
    """
    Vector Autoregressive process of order 1 - multivariate time series model.

    The fundamental model for systems of interrelated time series, combining
    multiple variables with dynamic cross-effects. This is the discrete-time
    analog of multivariate Ornstein-Uhlenbeck and the foundation for modern
    macroeconomic analysis.

    Vector Difference Equation
    ---------------------------
    Standard VAR(1) form:
        X[k] = A·X[k-1] + w[k]

    With control:
        X[k] = A·X[k-1] + B·u[k] + w[k]

    where:
        - X[k] ∈ ℝⁿ: State vector (n time series)
        - A ∈ ℝⁿˣⁿ: Coefficient matrix (VAR coefficients)
        - B ∈ ℝⁿˣᵖ: Control matrix (optional)
        - u[k] ∈ ℝᵖ: Exogenous inputs
        - w[k] ~ N(0, Σ_w): Vector white noise
        - Σ_w ∈ ℝⁿˣⁿ: Innovation covariance

    **Component Equations:**
        X_i[k] = Σⱼ A_ij·X_j[k-1] + w_i[k]

    Each variable depends on all lagged variables.

    Physical Interpretation
    -----------------------
    **Coefficient Matrix A:**

    Diagonal elements A_ii:
    - Own-lag effect (like univariate AR)
    - Persistence of variable i
    - Typical: 0.5-0.9 (positive persistence)

    Off-diagonal elements A_ij (i ≠ j):
    - Cross-lag effect (variable j → variable i)
    - Spillover, contagion, transmission
    - Can be positive (co-movement) or negative (offset)
    - Zero: No direct effect (X_j doesn't Granger-cause X_i)

    **Example (2D Macro Model):**
        [GDP[k]    ] = [a₁₁  a₁₂]·[GDP[k-1]    ] + [w₁[k]]
        [Inflation[k]]   [a₂₁  a₂₂] [Inflation[k-1]]   [w₂[k]]

    Interpretation:
    - a₁₁: GDP persistence (momentum)
    - a₁₂: Effect of past inflation on GDP (Phillips curve)
    - a₂₁: Effect of past GDP on inflation (demand-pull)
    - a₂₂: Inflation persistence (expectations)

    **Innovation Covariance Σ_w:**

    Diagonal elements (Σ_w)_ii:
    - Variance of idiosyncratic shock to variable i

    Off-diagonal elements (Σ_w)_ij:
    - Contemporaneous correlation between shocks
    - Common factors affecting multiple variables
    - Non-zero typical (shocks correlated)

    Key Features
    ------------
    **Multivariate:**
    n variables modeled jointly (not independently).

    **Cross-Effects:**
    A_ij captures dynamic interaction (j → i).

    **Stationarity:**
    All eigenvalues of A inside unit circle: |λ| < 1

    **Symmetry:**
    A can be asymmetric (feedback loops possible).

    **Correlated Shocks:**
    Σ_w typically non-diagonal (contemporaneous correlation).

    **Markov:**
    Future depends only on X[k], not X[k-2], X[k-3], ...

    **Gaussian:**
    If w ~ N(0, Σ_w), then X is multivariate Gaussian.

    Mathematical Properties
    -----------------------
    **Eigenvalue Decomposition:**
        A = V·Λ·V⁻¹

    Eigenvalues λ_i determine:
    - Stability: All |λ_i| < 1 required
    - Time scales: τ_i = -1/ln|λ_i| periods
    - Oscillations: Im(λ_i) ≠ 0 → cycles

    **Stationary Covariance:**

    Discrete Lyapunov:
        Γ₀ = A·Γ₀·Aᵀ + Σ_w

    Numerical solution via:
    - scipy.linalg.solve_discrete_lyapunov
    - Bartels-Stewart algorithm
    - Kronecker form: (I - A⊗A)·vec(Γ₀) = vec(Σ_w)

    **Impulse Response:**

    h-step response:
        Ψ_h = Aʰ

    Trace shock propagation through system.

    Physical Interpretation
    -----------------------
    **Matrix A Structure:**

    Full (dense):
    - All variables interact
    - n² parameters
    - Flexible but many parameters

    Diagonal:
    - No cross-effects (independent AR(1)s)
    - n parameters
    - Restrictive but parsimonious

    Block diagonal:
    - Groups of interacting variables
    - Within-group coupling, no between-group
    - Intermediate complexity

    Sparse:
    - Most A_ij = 0 (network structure)
    - Few interactions
    - Common in high-dimensional VARs

    State Space
    -----------
    State: X ∈ ℝⁿ
        - Vector of n time series
        - Unbounded (Gaussian)

    Control: u ∈ ℝᵖ (optional)
        - Exogenous inputs
        - Policy variables

    Noise: w ∈ ℝⁿ
        - Vector white noise
        - Covariance Σ_w (can be correlated)

    Parameters
    ----------
    A : np.ndarray or list, shape (n, n)
        VAR coefficient matrix
        - Diagonal: Own-lag effects
        - Off-diagonal: Cross-effects
        - Eigenvalues must satisfy |λ| < 1

    Sigma_w : np.ndarray or list, shape (n, n)
        Innovation covariance matrix
        - Must be symmetric positive definite
        - Diagonal: Idiosyncratic variances
        - Off-diagonal: Contemporaneous correlations

    B : Optional[np.ndarray], shape (n, p)
        Control/exogenous input matrix

    dt : float, default=1.0
        Sampling period (time units)

    Stochastic Properties
    ---------------------
    - System Type: LINEAR (multivariate AR)
    - Noise Type: ADDITIVE (vector white noise)
    - Markov: Yes (one lag)
    - Stationary: If all |λ(A)| < 1
    - Gaussian: If w ~ N(0, Σ_w)
    - Dimension: n (number of variables)

    Applications
    ------------
    **1. Macroeconomics:**
    - GDP, inflation, interest rates
    - Unemployment, investment, consumption
    - Fiscal and monetary policy analysis

    **2. Finance:**
    - Multi-asset portfolios
    - Volatility spillovers
    - Market contagion

    **3. International:**
    - Multi-country models
    - Trade dynamics
    - Exchange rates

    **4. Energy:**
    - Electricity, gas, oil prices
    - Demand forecasting
    - Renewable integration

    **5. Operations:**
    - Multi-product demand
    - Supply chain dynamics
    - Inventory systems

    Numerical Simulation
    --------------------
    **Direct Matrix Multiplication:**
        X[k+1] = A @ X[k] + w[k]

    **Efficient:** Use NumPy matrix operations.

    **Stability Check:**
    Before long simulation, verify eigenvalues inside unit circle.

    Granger Causality Testing
    --------------------------
    **Implementation:**

    Test if X_j → X_i (j Granger-causes i):
    1. Restricted model: X_i[k] = a_ii·X_i[k-1] + ε[k]
    2. Unrestricted: X_i[k] = a_ii·X_i[k-1] + a_ij·X_j[k-1] + ε[k]
    3. F-test: Is a_ij significantly non-zero?

    Comparison with Scalar AR
    --------------------------
    **Scalar AR(1):**
    - 1 variable, 1 lag
    - 1 parameter (φ)

    **VAR(1):**
    - n variables, 1 lag
    - n² parameters (A matrix)
    - Captures interactions

    **When VAR Essential:**
    - Variables economically related
    - Spillovers/contagion
    - Joint forecasting better

    Limitations
    -----------
    - Linear dynamics only
    - Constant parameters
    - Short memory (lag 1 only, extend to VAR(p))
    - Many parameters (n² for n variables)
    - Curse of dimensionality (high n)

    Extensions
    ----------
    - VAR(p): Multiple lags
    - VARMA: Add moving average
    - SVAR: Structural identification
    - BVAR: Bayesian shrinkage (Minnesota prior)
    - Factor-Augmented VAR: Handle high dimension
    
    See Also
    --------
    DiscreteAR1 : Univariate version (n=1)
    MultivariateOrnsteinUhlenbeck : Continuous-time analog
    DiscreteARMA11 : Univariate with MA component
    """

    def define_system(
        self,
        A: Union[np.ndarray, list],
        Sigma_w: Union[np.ndarray, list],
        B: Optional[Union[np.ndarray, list]] = None,
        dt: float = 1.0,
    ):
        """
        Define VAR(1) process dynamics.

        Parameters
        ----------
        A : np.ndarray or list, shape (n, n)
            VAR coefficient matrix
            - Diagonal: Own-lag effects (persistence)
            - Off-diagonal: Cross-lag effects (spillovers)
            - Must have eigenvalues |λ| < 1 for stationarity

        Sigma_w : np.ndarray or list, shape (n, n)
            Innovation covariance matrix
            - Must be symmetric positive definite
            - Diagonal: Shock variances
            - Off-diagonal: Shock correlations

        B : Optional[np.ndarray], shape (n, p)
            Control/exogenous input matrix

        dt : float, default=1.0
            Sampling period (e.g., daily, monthly, quarterly)

        Raises
        ------
        ValueError
            If dimensions incompatible or Σ_w not positive definite

        UserWarning
            If eigenvalues of A outside unit circle (non-stationary)

        Notes
        -----
        **Stationarity:**
        Check all eigenvalues of A: |λ_i| < 1

        **Parameter Count:**
        - VAR coefficients: n²
        - Covariance: n(n+1)/2
        - Total: n² + n(n+1)/2

        Example: n=3 → 9 + 6 = 15 parameters!

        **Granger Causality:**
        X_j Granger-causes X_i if A_ij ≠ 0.
        Test via F-test on individual coefficients.

        **Impulse Response:**
        h-step response: Ψ_h = Aʰ
        Trace shock effects over time.
        """
        # Convert to numpy
        A = np.array(A, dtype=float)
        Sigma_w = np.array(Sigma_w, dtype=float)

        # Validate dimensions
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError(f"A must be square, got shape {A.shape}")

        n = A.shape[0]

        if Sigma_w.shape != (n, n):
            raise ValueError(f"Sigma_w must be ({n},{n}), got {Sigma_w.shape}")

        # Check positive definite
        eigenvals_Sigma = np.linalg.eigvals(Sigma_w)
        if not np.all(eigenvals_Sigma > -1e-10):  # Small tolerance for numerical error
            raise ValueError(f"Sigma_w must be positive definite, got eigenvalues: {eigenvals_Sigma}")

        # Check stationarity
        eigenvals_A = np.linalg.eigvals(A)
        if not np.all(np.abs(eigenvals_A) < 1):
            import warnings
            warnings.warn(
                f"VAR(1) has eigenvalues outside unit circle: {eigenvals_A}. "
                f"System is non-stationary. For stationarity, need all |λ| < 1.",
                UserWarning
            )

        # Process control matrix
        if B is not None:
            B = np.array(B, dtype=float)
            if B.shape[0] != n:
                raise ValueError(f"B must have {n} rows, got {B.shape[0]}")
            p = B.shape[1]
        else:
            p = 0

        # Store matrices
        self._A_matrix = A
        self._Sigma_w_matrix = Sigma_w
        self._B_matrix = B
        self._n_vars = n
        self._n_controls = p

        # Create symbolic variables
        state_vars = [sp.symbols(f"X{i}", real=True) for i in range(n)]

        if p > 0:
            control_vars = [sp.symbols(f"u{i}", real=True) for i in range(p)]
        else:
            control_vars = []

        # Build dynamics symbolically
        X_vec = sp.Matrix(state_vars)
        A_sym = sp.Matrix(A)

        if p > 0:
            u_vec = sp.Matrix(control_vars)
            B_sym = sp.Matrix(B)
            next_state = A_sym * X_vec + B_sym * u_vec
        else:
            next_state = A_sym * X_vec

        # Build diffusion (Cholesky of Σ_w)
        L = np.linalg.cholesky(Sigma_w)
        L_sym = sp.Matrix(L)

        # System definition
        self.state_vars = state_vars
        self.control_vars = control_vars
        self._f_sym = next_state
        self.diffusion_expr = L_sym  # Cholesky factor

        self.parameters = {}  # Matrices in symbolic expressions
        self.order = 1
        self._dt = dt
        self.sde_type = "ito"

        # Output: Full state
        self._h_sym = X_vec

    def get_var_matrix(self) -> np.ndarray:
        """Get VAR coefficient matrix A."""
        return self._A_matrix

    def get_innovation_covariance(self) -> np.ndarray:
        """Get innovation covariance Σ_w."""
        return self._Sigma_w_matrix

    def get_eigenvalues(self) -> np.ndarray:
        """
        Get eigenvalues of A (determines stability and dynamics).

        Returns
        -------
        np.ndarray
            Eigenvalues (may be complex)

        Examples
        --------
        >>> var = DiscreteVAR1(A=[[0.8, 0.1], [0.1, 0.7]], 
        ...                    Sigma_w=np.eye(2)*0.01)
        >>> eigs = var.get_eigenvalues()
        >>> print(f"Eigenvalues: {eigs}")
        """
        return np.linalg.eigvals(self._A_matrix)

    def get_stationary_covariance(self) -> np.ndarray:
        """
        Compute stationary covariance Γ₀.

        Solves: Γ₀ = A·Γ₀·Aᵀ + Σ_w

        Returns
        -------
        np.ndarray
            Stationary covariance

        Examples
        --------
        >>> var = DiscreteVAR1(A=[[0.8, 0], [0, 0.7]], Sigma_w=np.eye(2)*0.01)
        >>> Gamma = var.get_stationary_covariance()
        >>> print(f"Stationary covariance:\n{Gamma}")
        """
        from scipy.linalg import solve_discrete_lyapunov
        return solve_discrete_lyapunov(self._A_matrix, self._Sigma_w_matrix)

    def compute_impulse_response(self, horizon: int = 20) -> np.ndarray:
        """
        Compute impulse response function.

        Parameters
        ----------
        horizon : int, default=20
            Number of periods ahead

        Returns
        -------
        np.ndarray
            IRF tensor (horizon+1, n, n)
            IRF[h, i, j] = response of variable i to shock in j after h periods

        Examples
        --------
        >>> var = DiscreteVAR1(A=[[0.8, 0.2], [0.1, 0.7]], Sigma_w=np.eye(2)*0.01)
        >>> irf = var.compute_impulse_response(horizon=10)
        >>> print(f"Impact of shock to X₁ on X₂ after 5 periods: {irf[5, 1, 0]:.3f}")
        """
        n = self._n_vars
        irf = np.zeros((horizon + 1, n, n))

        A_power = np.eye(n)
        for h in range(horizon + 1):
            irf[h] = A_power
            A_power = A_power @ self._A_matrix

        return irf

    def test_granger_causality(self, i: int, j: int) -> dict:
        """
        Test if variable j Granger-causes variable i.

        Parameters
        ----------
        i : int
            Target variable index
        j : int
            Source variable index

        Returns
        -------
        dict
            {'causes': bool, 'coefficient': float}

        Notes
        -----
        Simple test: Is A_ij ≠ 0?
        
        For rigorous test, need F-statistic from data.

        Examples
        --------
        >>> var = DiscreteVAR1(A=[[0.8, 0.2], [0.0, 0.7]], Sigma_w=np.eye(2)*0.01)
        >>> test = var.test_granger_causality(i=0, j=1)
        >>> print(f"X₁ causes X₀: {test}")  # True (A₀₁ = 0.2)
        """
        coefficient = self._A_matrix[i, j]
        causes = not np.isclose(coefficient, 0, atol=1e-10)

        return {
            'causes': causes,
            'coefficient': coefficient,
            'interpretation': f"X_{j} → X_{i}" if causes else f"X_{j} ↛ X_{i}"
        }


# Convenience functions
def create_bivariate_var(
    persistence1: float = 0.7,
    persistence2: float = 0.7,
    coupling12: float = 0.2,
    coupling21: float = 0.2,
    shock_std1: float = 0.1,
    shock_std2: float = 0.1,
    correlation: float = 0.0,
) -> DiscreteVAR1:
    """
    Create 2D VAR with specified structure.

    Parameters
    ----------
    persistence1, persistence2 : float
        Own-lag effects (diagonal of A)
    coupling12, coupling21 : float
        Cross-effects (off-diagonal of A)
    shock_std1, shock_std2 : float
        Innovation standard deviations
    correlation : float, default=0.0
        Innovation correlation (-1 to 1)

    Returns
    -------
    DiscreteVAR1

    Examples
    --------
    >>> # Symmetric coupling, uncorrelated shocks
    >>> var = create_bivariate_var(
    ...     persistence1=0.8,
    ...     persistence2=0.7,
    ...     coupling12=0.1,
    ...     coupling21=0.1,
    ...     correlation=0.0
    ... )
    """
    A = np.array([[persistence1, coupling12],
                  [coupling21, persistence2]])

    # Build covariance from marginal variances and correlation
    var1 = shock_std1**2
    var2 = shock_std2**2
    cov = correlation * shock_std1 * shock_std2

    Sigma_w = np.array([[var1, cov],
                        [cov, var2]])

    return DiscreteVAR1(A=A, Sigma_w=Sigma_w, dt=1.0)


def create_macro_var(
    n_variables: int = 3,
    dt: float = 1.0,
) -> DiscreteVAR1:
    """
    Create VAR for macroeconomic variables.

    Standard: GDP, inflation, interest rate (3 variables).

    Parameters
    ----------
    n_variables : int, default=3
        Number of macro variables
    dt : float, default=1.0
        Sampling period (1 = quarterly typical)

    Returns
    -------
    DiscreteVAR1

    Notes
    -----
    Default structure:
    - High persistence (0.7-0.9)
    - Moderate cross-effects (0.1-0.2)
    - Small correlations

    Examples
    --------
    >>> # Quarterly macro VAR
    >>> macro = create_macro_var(n_variables=3, dt=1.0)
    """
    # Simple default: Moderate persistence, weak coupling
    A = 0.7 * np.eye(n_variables) + 0.1 * (np.ones((n_variables, n_variables)) - np.eye(n_variables))

    # Uncorrelated shocks (simplified)
    Sigma_w = 0.01 * np.eye(n_variables)

    return DiscreteVAR1(A=A, Sigma_w=Sigma_w, dt=dt)