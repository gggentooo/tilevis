"""Step builder for final norm, LM-head projection, and next-token prediction.

Single responsibility: build Step for output logits and token probability ranking.
Allowed dependencies: torch, tilevis.core, stdlib.
"""
