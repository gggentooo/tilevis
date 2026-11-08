"""Step builder for multi-head self-attention.

Single responsibility: recompute attention weights, verify fidelity, and auto-detect n_heads.
Allowed dependencies: torch, tilevis.core, stdlib.
"""
