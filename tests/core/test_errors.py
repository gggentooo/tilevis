"""Unit tests for error and warning hierarchy."""

from __future__ import annotations

import pytest

from tilevis.core.errors import (
    ConfigInferenceError,
    FidelityWarning,
    ModelLoadError,
    RenderError,
    TilevisError,
    TraceFormatError,
)


def test_error_hierarchy() -> None:
    assert issubclass(TilevisError, Exception)
    assert issubclass(ModelLoadError, TilevisError)
    assert issubclass(ConfigInferenceError, TilevisError)
    assert issubclass(TraceFormatError, TilevisError)
    assert issubclass(RenderError, TilevisError)

    assert issubclass(FidelityWarning, UserWarning)


def test_error_instantiation_and_messages() -> None:
    err = ModelLoadError("Failed to load checkpoint file foo.pth")
    assert str(err) == "Failed to load checkpoint file foo.pth"
    assert isinstance(err, TilevisError)

    with pytest.warns(FidelityWarning, match="divergence detected"):
        import warnings

        warnings.warn("divergence detected in attention heads", FidelityWarning, stacklevel=2)
