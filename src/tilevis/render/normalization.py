"""Color normalization policies (per-tensor, shared-step, global-pass).

Single responsibility: scale float arrays symmetrically to [-1, 1] using robust statistics.
Allowed dependencies: numpy, tilevis.core, stdlib.
"""
