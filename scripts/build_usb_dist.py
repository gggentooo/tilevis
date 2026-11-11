#!/usr/bin/env python3
"""Build portable standalone executables and stage ready-to-copy USB thumb-drive distribution.

Supported platforms:
- macOS: builds native standalone binary and tilevis_player.app bundle
- Windows: builds native standalone tilevis_player.exe and dependency directory
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


def build_usb_distribution() -> int:
    """Run PyInstaller build and assemble portable dist/usb_drive/ directory."""
    repo_root = Path(__file__).resolve().parent.parent
    os.chdir(repo_root)

    print("==================================================================")
    print("  LLM Tile Visualizer — Building Portable USB Player Package")
    print("==================================================================")
    print(f"Platform : {sys.platform} ({sys.version})")
    print(f"Repo root: {repo_root}")

    # 1. Run PyInstaller
    spec_path = repo_root / "packaging" / "tilevis_player.spec"
    if not spec_path.is_file():
        sys.stderr.write(f"Error: Spec file not found at {spec_path}\n")
        return 1

    cache_dir = repo_root / "build" / "pyinstaller_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYINSTALLER_CONFIG_DIR"] = str(cache_dir)

    pyinstaller_cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        str(spec_path),
        "--noconfirm",
        "--clean",
        "--workpath",
        str(repo_root / "build"),
        "--distpath",
        str(repo_root / "dist"),
    ]
    print(f"\nExecuting: {' '.join(pyinstaller_cmd)}\n")
    res = subprocess.run(pyinstaller_cmd, env=env, check=False)
    if res.returncode != 0:
        sys.stderr.write(
            "\nPyInstaller build failed. Verify 'pip install -e \".[build]\"' is installed.\n"
        )
        return res.returncode

    # 2. Stage USB Thumb-Drive Folder
    dist_dir = repo_root / "dist"
    usb_dir = dist_dir / "usb_drive" / "tilevis_player"
    if usb_dir.exists():
        shutil.rmtree(usb_dir)
    usb_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nStaging USB thumb-drive folder at: {usb_dir}")

    # Copy binary / application bundle
    if sys.platform == "darwin":
        app_bundle = dist_dir / "tilevis_player.app"
        if app_bundle.is_dir():
            print("  Copying macOS Application Bundle (tilevis_player.app)...")
            shutil.copytree(app_bundle, usb_dir / "tilevis_player.app", symlinks=True)

        coll_dir = dist_dir / "tilevis_player"
        if coll_dir.is_dir():
            print("  Copying standalone Unix CLI binary folder (bin/)...")
            shutil.copytree(coll_dir, usb_dir / "bin", symlinks=True)
    else:
        coll_dir = dist_dir / "tilevis_player"
        if coll_dir.is_dir():
            print("  Copying Windows standalone executable and DLL dependencies...")
            for item in coll_dir.iterdir():
                dest = usb_dir / item.name
                if item.is_dir():
                    shutil.copytree(item, dest)
                else:
                    shutil.copy2(item, dest)

    # 3. Create External, Editable traces/ Folder on USB
    usb_traces = usb_dir / "traces"
    usb_traces.mkdir(parents=True, exist_ok=True)
    print("  Populating external editable traces/ directory...")

    trace_sources = [repo_root / "traces", repo_root / "traces" / "samples"]
    copied_count = 0
    for src in trace_sources:
        if not src.is_dir():
            continue
        for trace_file in sorted(src.glob("*.trace.npz")):
            dest_file = usb_traces / trace_file.name
            if not dest_file.exists():
                shutil.copy2(trace_file, dest_file)
                copied_count += 1
                print(f"    + {trace_file.name}")

    print(f"  Copied {copied_count} trace file(s) into USB traces/ folder.")

    # 4. Copy presenter configuration
    presenter_src = repo_root / "presenter" / "default.toml"
    if presenter_src.is_file():
        usb_presenter = usb_dir / "presenter"
        usb_presenter.mkdir(parents=True, exist_ok=True)
        shutil.copy2(presenter_src, usb_presenter / "default.toml")

    # 5. Write Quick-Start Guide for Lecture Hall
    readme_txt = usb_dir / "README.txt"
    readme_content = """LLM Tile Visualizer (tilevis) — Portable Player for Lecture Hall
==================================================================

How to Present:
1. Copy this entire 'tilevis_player' folder directly onto your USB thumb drive.
2. On the lecture-hall computer:
   - On macOS: Double-click 'tilevis_player.app' (or run './bin/tilevis_player')
   - On Windows: Double-click 'tilevis_player.exe'
3. The interactive Launcher GUI will open automatically!
   - Use the dropdown to select which trace to visualize.
   - Choose your preferred window mode (Borderless / Windowed), theme, and detail.
   - Click 'Launch Presentation ▶' (or press Enter).

Adding New Teammate Models:
- Simply drag and drop any new '*.trace.npz' files directly into the 'traces/' folder.
- In the Launcher GUI, click 'Refresh ↻' (or press R) to detect them instantly.

No Python, PyTorch, or CUDA installation is required on the lecture PC!
"""
    readme_txt.write_text(readme_content, encoding="utf-8")

    print("\n==================================================================")
    print("  Build Succeeded! Ready for USB Thumb-Drive Deployment.")
    print(f"  Distribution path: {usb_dir}")
    print("==================================================================\n")
    return 0


if __name__ == "__main__":
    sys.exit(build_usb_distribution())
