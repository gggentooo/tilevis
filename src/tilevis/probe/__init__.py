"""Probe module for attaching non-invasive forward hooks to PyTorch LLM models.

Single responsibility: hook execution, capture intermediate activations/weights, and build Traces.
Allowed dependencies: tilevis.core, torch, tiktoken, user model code, stdlib.
Forbidden dependencies: pygame, tilevis.render, tilevis.narration.
"""
