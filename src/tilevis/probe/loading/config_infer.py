"""Architecture configuration inference from checkpoint tensor shapes.

Single responsibility: infer n_layers, emb_dim, context_length, vocab_size, and qkv_bias.
Allowed dependencies: tilevis.core, stdlib.
"""
