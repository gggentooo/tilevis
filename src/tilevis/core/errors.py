"""Exception and warning hierarchy for tilevis.

Single responsibility: define all custom error and warning types.
Allowed dependencies: stdlib only.
"""

from __future__ import annotations


class TilevisError(Exception):
    """Base exception class for all tilevis errors."""


class ModelLoadError(TilevisError):
    """Raised when a model checkpoint, profile, or user code cannot be loaded."""


class ConfigInferenceError(TilevisError):
    """Raised when architecture config cannot be inferred or conflicts with checkpoint."""


class TraceFormatError(TilevisError):
    """Raised when a Trace archive is corrupted, unsupported, or fails schema validation."""


class RenderError(TilevisError):
    """Raised when an error occurs during renderer initialization, layout, or playback."""


class FidelityWarning(UserWarning):
    """Issued when captured/recomputed layer values diverge from true model activations."""
