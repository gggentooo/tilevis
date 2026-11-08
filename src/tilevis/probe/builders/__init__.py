"""Step builders converting captured tensors into Trace Step objects.

Single responsibility: package layer inputs, weights, outputs, and fidelity into Step records.
Allowed dependencies: torch, tilevis.core, stdlib.
"""
