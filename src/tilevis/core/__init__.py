"""Core data contract and registry utilities for tilevis.

Single responsibility: define the immutable Trace format, enums, errors, and registries.
Allowed dependencies: stdlib and numpy only.
"""

from tilevis.core.errors import (
    ConfigInferenceError,
    FidelityWarning,
    ModelLoadError,
    RenderError,
    TilevisError,
    TraceFormatError,
)
from tilevis.core.registry import Registry

__all__ = [
    "ConfigInferenceError",
    "FidelityWarning",
    "ModelLoadError",
    "Registry",
    "RenderError",
    "TilevisError",
    "TraceFormatError",
]
