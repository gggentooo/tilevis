"""LoRA detection and weight folding.

Single responsibility: fold LinearWithLoRA weights into standard Linear weight matrices.
Allowed dependencies: torch, tilevis.core, stdlib.
"""
