"""Capture buffer for storing hooked tensor inputs and outputs.

Single responsibility: isolate CPU tensor copies with batch dimension removed during a forward pass.
Allowed dependencies: torch, tilevis.core, stdlib.
"""
