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
Control System Design and Analysis
===================================

This module provides tools for control system design and analysis, including
classical control synthesis (LQR, LQG, Kalman filtering) and system analysis
(stability, controllability, observability).

Control Synthesis
-----------------
>>> from src.control import ControlSynthesis, design_lqr, design_lqg
>>>
>>> # Object-oriented interface
>>> synth = ControlSynthesis(system)
>>> K = synth.lqr(Q, R)
>>>
>>> # Functional interface
>>> result = design_lqr(A, B, Q, R)
>>> K = result['K']
>>>
>>> # LQG design (LQR + Kalman filter)
>>> result = design_lqg(A, B, C, Q, R, Qn, Rn)

System Analysis
---------------
>>> from src.control import (
...     SystemAnalysis,
...     analyze_stability,
...     analyze_controllability,
...     analyze_observability,
... )
>>>
>>> # Object-oriented interface
>>> analysis = SystemAnalysis(system)
>>> info = analysis.stability()
>>> info = analysis.controllability()
>>>
>>> # Functional interface
>>> stability = analyze_stability(A)
>>> ctrl = analyze_controllability(A, B)
>>> obs = analyze_observability(A, C)

Authors
-------
Gil Benezer

License
-------
GNU Affero General Public License v3.0
"""

# Classes
from .control_synthesis import ControlSynthesis
from .system_analysis import SystemAnalysis

# Functional interface
from .classical_control_functions import (
    analyze_controllability,
    analyze_observability,
    analyze_stability,
    design_kalman_filter,
    design_lqg,
    design_lqr,
)

# Export public API
__all__ = [
    # Classes
    "ControlSynthesis",
    "SystemAnalysis",
    # Control design functions
    "design_lqr",
    "design_kalman_filter",
    "design_lqg",
    # Analysis functions
    "analyze_stability",
    "analyze_controllability",
    "analyze_observability",
]