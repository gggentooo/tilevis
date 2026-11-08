"""Core data contract, serialization, and registry utilities for tilevis.

Single responsibility: define the immutable Trace format, enums, errors, and registries.
Allowed dependencies: stdlib and numpy only.
"""

from tilevis.core.config import PresenterSettings, RecordSettings, load_toml
from tilevis.core.errors import (
    ConfigInferenceError,
    FidelityWarning,
    ModelLoadError,
    RenderError,
    TilevisError,
    TraceFormatError,
)
from tilevis.core.kinds import (
    DetailLevel,
    Semantic,
    Stage,
    StepKind,
    TensorRole,
    parse_kind,
)
from tilevis.core.registry import Registry
from tilevis.core.serialization import MIGRATIONS, load_trace, save_trace
from tilevis.core.trace import (
    TRACE_SCHEMA_VERSION,
    AxisInfo,
    Fidelity,
    ModelSpecInfo,
    PassRecord,
    Prediction,
    Step,
    TensorSnapshot,
    TensorStats,
    Trace,
)
from tilevis.core.validation import validate_trace

__all__ = [
    "MIGRATIONS",
    "TRACE_SCHEMA_VERSION",
    "AxisInfo",
    "ConfigInferenceError",
    "DetailLevel",
    "Fidelity",
    "FidelityWarning",
    "ModelLoadError",
    "ModelSpecInfo",
    "PassRecord",
    "Prediction",
    "PresenterSettings",
    "RecordSettings",
    "Registry",
    "RenderError",
    "Semantic",
    "Stage",
    "Step",
    "StepKind",
    "TensorRole",
    "TensorSnapshot",
    "TensorStats",
    "TilevisError",
    "Trace",
    "TraceFormatError",
    "load_toml",
    "load_trace",
    "parse_kind",
    "save_trace",
    "validate_trace",
]
