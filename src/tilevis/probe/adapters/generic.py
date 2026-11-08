"""Fallback generic architecture adapter for arbitrary PyTorch models.

Single responsibility: inspect leaf modules of unknown architectures in execution order.
Allowed dependencies: torch, tilevis.core, stdlib.
"""
