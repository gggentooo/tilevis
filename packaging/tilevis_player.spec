# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller specification for building standalone tilevis player for Windows lecture PCs.

Bundles only playback modules (core, narration, render, assets, presenter, sample traces)
and explicitly excludes heavy recording & dev dependencies (torch, tiktoken, matplotlib, etc.).
"""

from pathlib import Path

block_cipher = None

SPEC_ROOT = Path(SPECPATH).resolve()
REPO_ROOT = SPEC_ROOT.parent

datas = [
    (str(REPO_ROOT / "src" / "tilevis" / "assets" / "fonts" / "*.ttf"), "tilevis/assets/fonts"),
    (str(REPO_ROOT / "src" / "tilevis" / "assets" / "fonts" / "*.txt"), "tilevis/assets/fonts"),
    (str(REPO_ROOT / "src" / "tilevis" / "narration" / "locales" / "*.toml"), "tilevis/narration/locales"),
    (str(REPO_ROOT / "presenter" / "*.toml"), "presenter"),
    (str(REPO_ROOT / "traces" / "samples" / "*.trace.npz"), "traces/samples"),
]

excludes = [
    "torch",
    "torchvision",
    "torchaudio",
    "tiktoken",
    "matplotlib",
    "pytest",
    "pytest_cov",
    "pytest_timeout",
    "hypothesis",
    "mypy",
    "ruff",
    "scipy",
    "IPython",
    "jupyter",
]

a = Analysis(
    [str(REPO_ROOT / "src" / "tilevis" / "__main__.py")],
    pathex=[str(REPO_ROOT / "src")],
    binaries=[],
    datas=datas,
    hiddenimports=[
        "tilevis.core",
        "tilevis.narration",
        "tilevis.render",
        "tilevis.check",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="tilevis_player",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="tilevis_player",
)
