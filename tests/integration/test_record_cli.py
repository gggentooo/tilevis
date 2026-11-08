"""Integration tests for the `tilevis record` CLI subcommand and sample traces.

Single responsibility: test CLI invocation of record, output file generation,
and verification of committed sample traces.
"""

from __future__ import annotations

from pathlib import Path

from tilevis.cli import main
from tilevis.core.serialization import load_trace
from tilevis.core.validation import validate_trace


def test_record_cli_execution(tmp_path: Path) -> None:
    """Test full CLI execution of tilevis record subcommand."""
    out_path = tmp_path / "cli_test.trace.npz"

    ret = main(
        [
            "record",
            "--profile",
            "profiles/tiny_random.toml",
            "--prompt",
            "Testing CLI recording",
            "--new-tokens",
            "2",
            "-o",
            str(out_path),
        ]
    )

    assert ret == 0
    assert out_path.exists()
    trace = load_trace(out_path)
    assert len(trace.passes) == 2
    validate_trace(trace, strict=True)


def test_committed_sample_traces_load_and_validate() -> None:
    """Verify that committed sample traces load, validate, and conform to standards."""
    root = Path(__file__).resolve().parent.parent.parent
    sample_dir = root / "traces" / "samples"

    demo_trace_path = sample_dir / "tiny_demo.trace.npz"
    ch07_trace_path = sample_dir / "tiny_ch07.trace.npz"

    assert demo_trace_path.exists(), f"Sample trace missing: {demo_trace_path}"
    assert ch07_trace_path.exists(), f"Sample trace missing: {ch07_trace_path}"

    t_demo = load_trace(demo_trace_path)
    validate_trace(t_demo, strict=True)
    assert len(t_demo.passes) == 3
    assert t_demo.model.n_layers == 3

    t_ch07 = load_trace(ch07_trace_path)
    validate_trace(t_ch07, strict=True)
    assert len(t_ch07.passes) == 3
    assert t_ch07.model.n_layers == 1
