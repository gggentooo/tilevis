"""Weight extractors for linear layers, embeddings, layer norms, and LoRA wrappers.

Single responsibility: extract transposed (in, out) weight matrices and normalization parameters.
Allowed dependencies: torch, tilevis.core, stdlib.
"""
