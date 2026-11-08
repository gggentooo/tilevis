"""Checkpoint loading, unwrapping, and prefix stripping.

Single responsibility: load raw .pth checkpoints and normalize state dicts.
Allowed dependencies: torch, stdlib, tilevis.core.
"""
