# Standalone Player Packaging (PyInstaller)

This directory contains the PyInstaller specification to bundle `tilevis` as a standalone Windows folder for the lecture-hall PC.

> **Note on Cross-Compilation:** PyInstaller does **not** cross-compile Windows executables from macOS or Linux. Run this packaging build **on a Windows machine**.

---

## 1. Prerequisites (on Windows)

Install `tilevis` with the `[build]` extra:

```cmd
pip install -e ".[build]"
```

Verify `pyinstaller` is available:

```cmd
pyinstaller --version
```

---

## 2. Building the Standalone Player

Run `pyinstaller` using the spec file from the project root:

```cmd
pyinstaller packaging\tilevis_player.spec
```

The output folder will be generated at:

```text
dist\tilevis_player\
├── tilevis_player.exe
├── tilevis\
│   ├── assets\fonts\
│   └── narration\locales\
├── presenter\
└── traces\samples\
```

---

## 3. Deployment & Presentation Instructions

1. Copy the `dist\tilevis_player\` directory to a USB thumb drive or the presentation laptop.
2. Ensure teammate `*.trace.npz` files are placed inside `dist\tilevis_player\traces\` or passed by full path.
3. Open a command prompt inside `dist\tilevis_player\` and run:

   ```cmd
   tilevis_player.exe check traces\samples\tiny_demo.trace.npz
   tilevis_player.exe play traces\samples\tiny_demo.trace.npz --window borderless
   ```

No Python runtime, PyTorch, or CUDA installation is required on the lecture-hall computer.
