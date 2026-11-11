# Standalone Player Packaging (PyInstaller & USB Thumb Drive)

This directory contains the packaging specifications and tools to bundle `tilevis` as a standalone, zero-dependency executable distribution for the lecture-hall presentation computer.

- **macOS:** Produces `tilevis_player.app` application bundle and Unix CLI binary.
- **Windows:** Produces `tilevis_player.exe` and portable DLL runtime folder.
- **Portable USB Thumb Drive:** Staged automatically at `dist/usb_drive/tilevis_player/` with external editable `traces/` folder.

> **Note on Cross-Compilation:** PyInstaller does **not** cross-compile Windows executables from macOS or Linux, nor macOS executables from Windows.
> - To build the **macOS application bundle**, run `python scripts/build_usb_dist.py` on macOS.
> - To build the **Windows `.exe`**, run `python scripts/build_usb_dist.py` (or `scripts\build_windows.bat`) on Windows, or download the prebuilt multi-platform artifact from GitHub Actions (`.github/workflows/build_executables.yml`).

---

## 1. Prerequisites

Install `tilevis` with the `[build]` extra:

```bash
pip install -e ".[build]"
```

Verify `pyinstaller` is available:

```bash
pyinstaller --version
```

---

## 2. Automated USB Thumb-Drive Build (`scripts/build_usb_dist.py`)

Run the automated staging script from the repository root:

```bash
# macOS / Linux
python scripts/build_usb_dist.py

# Windows
python scripts\build_usb_dist.py
```

The script builds the native binary and stages a ready-to-copy folder:

```text
dist/usb_drive/tilevis_player/
├── tilevis_player.app/       # macOS only: double-clickable application bundle
├── bin/                      # macOS/Linux standalone binary folder
├── tilevis_player.exe        # Windows only: double-clickable executable
├── _internal/                # Standalone bundled Python runtime & DLLs
├── traces/                   # External editable trace archive folder
│   ├── *.trace.npz           # User traces and sample traces (drag & drop new ones here!)
├── presenter/
│   └── default.toml          # Presenter settings (theme, window mode, locale)
└── README.txt                # Lecture-hall quick start instructions
```

---

## 3. GitHub Actions Multi-Platform CI

For teams where members work on different OSs, the repository includes `.github/workflows/build_executables.yml`.
Every push to `main` or tag runs a build matrix across `windows-latest` and `macos-latest`, producing downloadable standalone zips:
- `tilevis-player-windows-x64.zip`
- `tilevis-player-macos-x64.zip`

---

## 4. Deployment & Presentation Instructions

1. Copy the entire `dist/usb_drive/tilevis_player/` folder directly onto a USB thumb drive.
2. Plug the USB thumb drive into the presentation PC in the lecture hall.
3. Launch the presentation:
   - **macOS:** Double-click `tilevis_player.app` (or `./bin/tilevis_player`).
   - **Windows:** Double-click `tilevis_player.exe`.
4. The **Interactive Launcher GUI** will open automatically:
   - Select your desired model trace from the dropdown.
   - Adjust theme, window mode, detail level, and language with one click.
   - Click **Launch Presentation ▶** (or press Enter) to start!

No Python runtime, PyTorch, CUDA, or internet connection is required on stage.
