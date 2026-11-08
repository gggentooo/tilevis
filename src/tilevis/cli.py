"""Command-line interface for tilevis.

Single responsibility: parse CLI arguments and dispatch to lazy-imported probe or render modules.
Allowed dependencies: stdlib (argparse, sys, logging, pathlib), core.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from typing import Any

import tilevis
from tilevis.logging_setup import setup_logging


def _record_command(args: argparse.Namespace) -> int:
    """Handle `tilevis record` subcommand."""
    from pathlib import Path

    from tilevis.core.config import RecordSettings
    from tilevis.core.errors import TilevisError
    from tilevis.core.serialization import save_trace
    from tilevis.probe.loading.factory import build_model
    from tilevis.probe.loading.profile import load_profile
    from tilevis.probe.recorder import TraceRecorder

    try:
        profile = load_profile(args.profile)
        overrides: dict[str, Any] = {}
        if getattr(args, "n_heads", None) is not None:
            overrides["n_heads"] = args.n_heads
        if getattr(args, "seed", None) is not None:
            overrides["seed"] = args.seed

        device_req = getattr(args, "device", "auto")
        built_model = build_model(profile, overrides=overrides, device=device_req)

        settings = RecordSettings(
            new_tokens=args.new_tokens,
            strategy=args.strategy,
            temperature=args.temperature,
            top_k=args.top_k,
            seed=args.seed,
            max_features=args.max_features,
            max_tokens=args.max_tokens,
            device=device_req,
        )

        recorder = TraceRecorder(built_model, settings)
        trace = recorder.record(args.prompt)

        out_path = Path(args.output).resolve()
        save_trace(trace, out_path)

        total_steps = sum(len(p.steps) for p in trace.passes)
        sys.stdout.write(
            f"Successfully recorded {len(trace.passes)} pass(es) "
            f"({total_steps} steps) to {out_path.name}\n"
        )
        return 0
    except TilevisError as exc:
        if getattr(args, "verbose", False):
            raise
        sys.stderr.write(f"Error: {exc}\n")
        return 1


def _play_command(args: argparse.Namespace) -> int:
    """Handle `tilevis play` subcommand."""
    # Lazy import of render module when implemented in Phase 5
    sys.stdout.write("Subcommand 'play' is not implemented yet (scheduled for Phase 5).\n")
    return 1


def _inspect_command(args: argparse.Namespace) -> int:
    """Handle `tilevis inspect` subcommand."""
    from tilevis.probe.inspect import inspect_model

    return inspect_model(
        profile_path=getattr(args, "profile", None),
        checkpoint_path=getattr(args, "checkpoint", None),
    )


def _check_command(args: argparse.Namespace) -> int:
    """Handle `tilevis check` subcommand."""
    # Preflight environment check when implemented in Phase 8
    sys.stdout.write("Subcommand 'check' is not implemented yet (scheduled for Phase 8).\n")
    return 1


def _demo_command(args: argparse.Namespace) -> int:
    """Handle `tilevis demo` subcommand."""
    # In-memory demo when implemented in Phase 7
    sys.stdout.write("Subcommand 'demo' is not implemented yet (scheduled for Phase 7).\n")
    return 1


def build_parser() -> argparse.ArgumentParser:
    """Construct and configure the argument parser."""
    parser = argparse.ArgumentParser(
        prog="tilevis",
        description=(
            "LLM Tile Visualizer: step-by-step colored-tile animation for GPT forward passes"
        ),
    )
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"tilevis {tilevis.__version__}",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable detailed debug logging",
    )

    subparsers = parser.add_subparsers(
        title="subcommands",
        dest="subcommand",
        metavar="<command>",
    )

    # Subcommand: record
    record_parser = subparsers.add_parser(
        "record",
        help="Record a model forward pass and token generation into a Trace file",
    )
    record_parser.add_argument(
        "--profile",
        required=True,
        help="Path to model profile TOML file",
    )
    record_parser.add_argument(
        "--prompt",
        required=True,
        help="Initial prompt text to feed the model",
    )
    record_parser.add_argument(
        "--new-tokens",
        type=int,
        default=3,
        help="Number of new tokens to generate (default: 3)",
    )
    record_parser.add_argument(
        "--strategy",
        choices=["greedy", "sample"],
        default="greedy",
        help="Generation strategy (default: greedy)",
    )
    record_parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default: 0.0)",
    )
    record_parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Top-k filtering threshold",
    )
    record_parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed for reproducibility",
    )
    record_parser.add_argument(
        "--n-heads",
        type=int,
        default=None,
        help="Explicit attention heads override (inferred if omitted)",
    )
    record_parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="Device to run inference on (default: auto)",
    )
    record_parser.add_argument(
        "--max-features",
        type=int,
        default=64,
        help="Maximum features displayed per dimension before reduction",
    )
    record_parser.add_argument(
        "--max-tokens",
        type=int,
        default=24,
        help="Maximum tokens displayed before reduction",
    )
    record_parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output Trace file path (*.trace.npz)",
    )
    record_parser.set_defaults(func=_record_command)

    # Subcommand: play
    play_parser = subparsers.add_parser(
        "play",
        help="Play back an interactive Trace file using pygame-ce",
    )
    play_parser.add_argument(
        "trace",
        help="Path to .trace.npz archive to visualize",
    )
    play_parser.add_argument(
        "--presenter",
        default="presenter/default.toml",
        help="Path to presenter settings TOML file",
    )
    play_parser.add_argument(
        "--window",
        choices=["borderless", "windowed", "exclusive"],
        default=None,
        help="Window display mode",
    )
    play_parser.add_argument(
        "--theme",
        choices=["dark", "light"],
        default=None,
        help="Visual theme override",
    )
    play_parser.add_argument(
        "--detail",
        type=int,
        choices=[1, 2, 3],
        default=None,
        help="Detail level: 1 (Story), 2 (Standard), 3 (Deep)",
    )
    play_parser.set_defaults(func=_play_command)

    # Subcommand: inspect
    inspect_parser = subparsers.add_parser(
        "inspect",
        help="Inspect a model profile or checkpoint without recording",
    )
    group = inspect_parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--profile",
        help="Path to model profile TOML file",
    )
    group.add_argument(
        "--checkpoint",
        help="Path to raw PyTorch .pth checkpoint file",
    )
    inspect_parser.set_defaults(func=_inspect_command)

    # Subcommand: check
    check_parser = subparsers.add_parser(
        "check",
        help="Run preflight environment checks (display, DPI, font, traces)",
    )
    check_parser.add_argument(
        "traces",
        nargs="*",
        help="Optional paths to .trace.npz files to validate",
    )
    check_parser.set_defaults(func=_check_command)

    # Subcommand: demo
    demo_parser = subparsers.add_parser(
        "demo",
        help="Run an instant demo with an in-memory random model",
    )
    demo_parser.add_argument(
        "--prompt",
        default="Every effort moves you forward",
        help="Demo prompt text",
    )
    demo_parser.set_defaults(func=_demo_command)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Main CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    setup_logging(verbose=getattr(args, "verbose", False))

    if not hasattr(args, "func"):
        parser.print_help()
        return 0

    try:
        return int(args.func(args))
    except Exception as exc:
        if getattr(args, "verbose", False):
            raise
        sys.stderr.write(f"Error: {exc}\n")
        return 1
