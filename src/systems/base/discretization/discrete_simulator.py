"""
Simulates discrete-time trajectories using a discretizer.

Handles:
- Controller types (sequence, function, nn.Module, None)
- Output feedback with observers
- Batched simulation
- Return options (all states vs final, with/without controls)
"""