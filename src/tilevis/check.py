"""Preflight diagnostics and trace verification for lecture-hall deployment.

Single responsibility: inspect runtime environment, display, fonts, and validate trace archives.
Allowed dependencies: tilevis.core, tilevis.render, stdlib.
"""

from __future__ import annotations

import importlib.metadata
import importlib.resources
import logging
import platform
import sys
import time
from pathlib import Path
from typing import TextIO

from tilevis.core.errors import TraceFormatError
from tilevis.core.serialization import load_trace
from tilevis.core.validation import validate_trace

logger = logging.getLogger("tilevis.check")


def check_python_version() -> tuple[bool, str]:
    """Verify running on supported Python 3.11 - 3.13."""
    ver = sys.version_info
    ver_str = f"{ver.major}.{ver.minor}.{ver.micro}"
    if (3, 11) <= (ver.major, ver.minor) <= (3, 13):
        return True, f"Python {ver_str} ({platform.architecture()[0]})"
    return (
        False,
        f"Python {ver_str} is not officially supported; tilevis requires Python 3.11 - 3.13",
    )


def check_os_and_dpi() -> tuple[bool, str]:
    """Report operating system and High-DPI scaling status."""
    os_name = platform.system()
    os_ver = platform.release()
    arch = platform.machine()
    msg = f"{os_name} {os_ver} ({arch})"

    if os_name == "Windows":
        from tilevis.render.platform import enable_high_dpi

        ok = enable_high_dpi()
        dpi_status = "enabled (per-monitor v2)" if ok else "standard"
        msg += f" · Windows High-DPI: {dpi_status}"

    return True, msg


def check_pygame_packages() -> tuple[bool, str, list[str]]:
    """Verify pygame-ce installation and detect legacy pygame package conflicts."""
    warnings: list[str] = []

    # Query installed package metadata
    installed_dists = {dist.metadata["Name"].lower() for dist in importlib.metadata.distributions()}

    has_ce = "pygame-ce" in installed_dists
    has_legacy = "pygame" in installed_dists and not has_ce

    # Check for mutual collision
    if "pygame-ce" in installed_dists and "pygame" in installed_dists:
        return (
            False,
            "Package collision: both 'pygame' and 'pygame-ce' are installed",
            [
                "Both packages write to the same 'pygame' namespace and conflict.",
                "To resolve: run 'pip uninstall -y pygame pygame-ce' then 'pip install pygame-ce'.",
            ],
        )

    import pygame

    sdl_ver = ".".join(str(v) for v in pygame.get_sdl_version())
    ce_ver = getattr(pygame, "__version__", "unknown")

    if has_legacy:
        return (
            False,
            f"Legacy 'pygame' ({ce_ver}) installed instead of 'pygame-ce'",
            ["tilevis requires 'pygame-ce' for Windows High-DPI and stable font rendering."],
        )

    return True, f"pygame-ce {ce_ver} (SDL {sdl_ver})", warnings


def check_bundled_fonts() -> tuple[bool, str]:
    """Verify bundled OFL Noto Sans fonts are present and non-empty."""
    try:
        font_dir = importlib.resources.files("tilevis.assets.fonts")
        reg_file = font_dir.joinpath("NotoSans-Regular.ttf")
        bold_file = font_dir.joinpath("NotoSans-Bold.ttf")

        reg_bytes = reg_file.read_bytes()
        bold_bytes = bold_file.read_bytes()

        if len(reg_bytes) < 1000 or len(bold_bytes) < 1000:
            return False, "Bundled font files are corrupted or unexpectedly small"

        return True, f"Noto Sans Regular ({len(reg_bytes)} B) & Bold ({len(bold_bytes)} B)"
    except Exception as exc:
        return False, f"Failed to locate bundled fonts: {exc}"


def check_display_modes() -> tuple[bool, str]:
    """Detect available desktop resolutions without requiring a window."""
    try:
        import pygame

        if not pygame.display.get_init():
            pygame.display.init()

        sizes = pygame.display.get_desktop_sizes()
        if sizes:
            res_str = ", ".join(f"{w}x{h}" for w, h in sizes)
            return True, f"Detected display(s): {res_str}"
        info = pygame.display.Info()
        return True, f"Current display resolution: {info.current_w}x{info.current_h}"
    except Exception as exc:
        return True, f"Display query skipped (headless or offscreen): {exc}"


def check_trace_file(path: Path) -> tuple[bool, str, list[str]]:
    """Load, benchmark, and strictly validate a single Trace archive."""
    details: list[str] = []
    if not path.is_file():
        return False, f"Trace file not found: {path}", ["Check the provided file path."]

    t0 = time.perf_counter()
    try:
        trace = load_trace(path)
    except TraceFormatError as exc:
        return False, f"Failed to load trace {path.name}: {exc}", []
    except Exception as exc:
        return False, f"Corrupted or invalid trace archive {path.name}: {exc}", [str(exc)]

    load_time_ms = (time.perf_counter() - t0) * 1000.0

    # Structural & numerical validation
    issues = validate_trace(trace, strict=False)
    if issues:
        err_msgs = [f"Validation error: {iss}" for iss in issues]
        msg = f"Trace validation failed for {path.name} with {len(issues)} issue(s)"
        return False, msg, err_msgs

    # Telemetry summary
    total_steps = sum(len(p.steps) for p in trace.passes)
    model = trace.model
    spec_summary = (
        f"model='{model.name}' ({model.n_layers}L, {model.emb_dim}D, "
        f"{model.n_heads or '?'}H, {model.n_params:,} params)"
    )
    details.append(f"Model: {spec_summary}")
    struct_str = (
        f"Structure: {len(trace.passes)} pass(es), {total_steps} step(s) · "
        f"Load time: {load_time_ms:.1f} ms"
    )
    details.append(struct_str)

    # Fidelity inspect
    fidelity_warns = 0
    fidelity_fails = 0
    for p in trace.passes:
        for s in p.steps:
            if s.fidelity.status == "failed":
                fidelity_fails += 1
            elif s.fidelity.status in ("approximate", "unverified"):
                fidelity_warns += 1

    if fidelity_fails > 0:
        details.append(f"WARNING: {fidelity_fails} step(s) reported FAILED numerical fidelity")
    elif fidelity_warns > 0:
        details.append(f"Note: {fidelity_warns} step(s) reported unverified/approximate fidelity")

    return True, f"Valid trace ({path.name})", details


def run_checks(
    trace_paths: list[str] | None = None,
    out: TextIO | None = None,
) -> int:
    """Execute full preflight environment and trace diagnostic suite."""
    if out is None:
        out = sys.stdout
    out.write("=======================================================\n")
    out.write("  LLM Tile Visualizer (`tilevis`) — Preflight Check\n")
    out.write("=======================================================\n\n")

    overall_ok = True

    # 1. Environment & libraries
    env_checks = [
        ("Python runtime", check_python_version),
        ("Operating system", check_os_and_dpi),
        ("Bundled fonts", check_bundled_fonts),
        ("Display capabilities", check_display_modes),
    ]

    out.write("[Runtime Environment]\n")
    for name, fn in env_checks:
        ok, msg = fn()
        icon = "[PASS]" if ok else "[FAIL]"
        out.write(f"  {icon:6s} {name:22s} : {msg}\n")
        if not ok:
            overall_ok = False

    # pygame check (handles warnings list)
    pg_ok, pg_msg, pg_details = check_pygame_packages()
    pg_icon = "[PASS]" if pg_ok else "[FAIL]"
    out.write(f"  {pg_icon:6s} {'pygame library':22s} : {pg_msg}\n")
    for det in pg_details:
        out.write(f"         {det}\n")
    if not pg_ok:
        overall_ok = False

    out.write("\n[Trace Verification]\n")

    resolved_paths: list[Path] = []
    if trace_paths:
        for p_str in trace_paths:
            resolved_paths.append(Path(p_str))
    else:
        # Default sample traces
        samples_dir = Path("traces/samples")
        if not samples_dir.is_dir():
            pkg_root = Path(__file__).resolve().parent.parent.parent
            samples_dir = pkg_root / "traces" / "samples"
        if samples_dir.is_dir():
            resolved_paths = sorted(samples_dir.glob("*.trace.npz"))

    if not resolved_paths:
        out.write("  [WARN] No trace files specified or found in traces/samples/\n")
    else:
        for trace_p in resolved_paths:
            ok, msg, details = check_trace_file(trace_p)
            icon = "[PASS]" if ok else "[FAIL]"
            out.write(f"  {icon:6s} {trace_p.name}\n")
            out.write(f"         Status: {msg}\n")
            for det in details:
                out.write(f"         {det}\n")
            if not ok:
                overall_ok = False

    out.write("\n-------------------------------------------------------\n")
    if overall_ok:
        out.write("Result: All preflight checks PASSED. Ready for presentation.\n")
        return 0
    else:
        out.write("Result: Preflight checks FAILED. Review errors above.\n")
        return 1
