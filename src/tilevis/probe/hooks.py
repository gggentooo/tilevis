"""Forward hook context manager for non-invasive PyTorch execution capture.

Single responsibility: register and cleanly remove PyTorch module hooks with execution counting.
Allowed dependencies: torch, tilevis.core, stdlib.
"""
