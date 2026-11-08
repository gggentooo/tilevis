"""Multi-pass token generation loop matching previous_chapters semantics.

Single responsibility: drive greedy or sampling generation loop and crop context window.
Allowed dependencies: torch, tilevis.core, stdlib.
"""
