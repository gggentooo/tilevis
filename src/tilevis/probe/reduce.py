"""Dimensionality reducers and consistent axis IndexPlan management.

Single responsibility: downsample large tensors to displayable tile sizes deterministically.
Allowed dependencies: tilevis.core, numpy, torch, stdlib.
"""
