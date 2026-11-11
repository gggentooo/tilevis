# LLM Tile Visualizer (`tilevis`)

`tilevis` is a Python application that transforms a GPT-style Large Language Model's forward pass into an intuitive, step-by-step, animated visualization of colored tiles. Every node activation and weight is represented as a colorblind-safe tile, allowing presenters to step through internal transformer calculations one layer at a time.

Designed for live classroom presentations to students with little to no deep learning background, `tilevis` runs cleanly on top of `ch07/01_main-chapter-code/previous_chapters.py` without modifying any file in `ch07/`.

---

## 1. 60-Second Quick Start (Lecture-Hall Playback)

The presentation PC requires **only Python, NumPy, and pygame-ce** (no PyTorch, CUDA, or model checkpoints required on stage).

### Installation (Playback Only)

```bash
git clone https://github.com/your-team/llm_tile_visualizer.git
cd llm_tile_visualizer

# Create virtual environment and install lightweight base runtime
python -m venv .venv
# Windows: .venv\Scripts\activate   |   macOS/Linux: source .venv/bin/activate
pip install -e .
```

### Preflight Diagnostics, Interactive Launcher & Playback

```bash
# 1. Run preflight environment check
python -m tilevis check

# 2. Open the Interactive Trace Launcher GUI (Default!)
python -m tilevis
# or: tilevis launch

# 3. Or launch directly into a trace file (or double-click scripts/play_demo.bat on Windows)
python -m tilevis play traces/samples/tiny_demo.trace.npz --window borderless
```

---

## 2. Portable USB Thumb-Drive Player (Zero Python Needed on Lecture PC)

To present in the lecture hall without installing Python, PyTorch, or CUDA:
1. Build the standalone portable distribution: `python scripts/build_usb_dist.py`.
2. Copy the resulting `dist/usb_drive/tilevis_player/` folder directly onto a USB thumb drive.
3. On stage:
   - **macOS:** Double-click `tilevis_player.app`
   - **Windows:** Double-click `tilevis_player.exe`
4. The interactive launcher opens instantly, discovers all `.trace.npz` files in `traces/`, and lets you start the presentation with one click.

---

## 2. Guide for Teammates: Recording Your Model Checkpoint

Teammates with trained `.pth` models (including different layer counts, attention heads, context lengths, or LoRA wrappers) can record their models on their development laptops (requires the `[record]` extra with PyTorch):

### Step 1: Install Recording Dependencies

```bash
pip install -e ".[record]"
```

### Step 2: Create a Model Profile (`profiles/my_model.toml`)

Create a TOML profile pointing to your checkpoint (hyperparameters like `emb_dim`, `n_layers`, and `context_length` are inferred automatically):

```toml
[profile]
name = "Team Model — 6-Layer Finetuned GPT"

[model]
code = "ch07/01_main-chapter-code/previous_chapters.py"
class = "GPTModel"
checkpoint = "path/to/my_checkpoint.pth"
seed = 42

[text]
tokenizer = "gpt2"
template = "alpaca"   # "raw" | "alpaca"
```

Verify model loading and configuration inference:

```bash
tilevis inspect --profile profiles/my_model.toml
```

### Step 3: Record Forward Generation Run

```bash
tilevis record \
  --profile profiles/my_model.toml \
  --prompt "Every effort moves you forward" \
  --new-tokens 3 \
  --strategy greedy \
  -o traces/my_model.trace.npz
```

Copy the resulting `.trace.npz` file to a USB thumb drive. It contains everything needed for playback on any lecture computer.

---

## 3. Presenter Controls Cheat Sheet

The entire presentation can be driven using **either the keyboard or mouse alone**:

| Control | Keyboard Shortcut | Mouse Action | Purpose |
|---|---|---|---|
| **Next Step** | `Right` / `Space` / `Return` / `PageDown` | Left-Click (on stage or `Next >` button) | Advance one layer or calculation step |
| **Previous Step** | `Left` / `Backspace` / `PageUp` | Right-Click (or `< Prev` button) | Step backward one layer |
| **Skip Animation** | `Right` / `Space` (while animating) | Left-Click (while animating) | Complete layer animation immediately |
| **Next Stage** | `Shift + Right` | — | Jump to next architectural section (`Input` / `Block` / `Output`) |
| **Prev Stage** | `Shift + Left` | — | Jump to previous architectural section |
| **Next Token** | `Ctrl + Right` | — | Jump to the next generated word pass |
| **Jump to Block** | — | Click Minimap segment (`B1`, `B2`, etc.) | Jump directly to any transformer block |
| **Detail Level** | `D` | Click on-screen `Detail` button | Cycle detail: `Story` (collapsed) ↔ `Standard` ↔ `Deep` |
| **Nerd Mode** | `N` | Click on-screen `Nerd` button | Toggle floating technical parameters & tensor stats |
| **Language** | `L` | Click on-screen `Lang: EN` / `Lang: KO` button | Toggle caption language (English ↔ 한국어) |
| **Theme** | `T` | — | Toggle color theme (`dark` ↔ `light`) |
| **Autoplay** | `P` | — | Start/stop automated step progression |
| **Speed** | `+` / `-` | — | Increase / decrease animation speed |
| **Fullscreen** | `F11` | — | Toggle borderless fullscreen / windowed mode |
| **Help Overlay** | `H` / `F1` | Click on-screen `?` button | Open/close keyboard & mouse shortcut cheat sheet |
| **Exit** | `Escape` (press twice within 2.0s) | — | Safe exit confirmation |

---

## 4. Troubleshooting & Presentation Tips

| Issue | Cause | Solution |
|---|---|---|
| **Screenshare shows a black window** | Hardware exclusive fullscreen flickering in Zoom/Teams | Use borderless windowed mode: `tilevis play <trace> --window borderless` (default). |
| **Text or tiles appear blurry on lecture display** | Windows display scaling set to 125% or 150% without High-DPI awareness | Run `tilevis check` to verify DPI scaling. `tilevis` enables per-monitor v2 DPI awareness automatically. |
| **File not found or encoding crash on Korean Windows** | Windows defaults text encoding to `cp949` | `tilevis` enforces explicit `utf-8` on all file I/O. Use `scripts/play_demo.bat` which configures `chcp 65001`. |
| **Presentation PC has no Python installed** | Lecturer computer locked down | Build standalone player on Windows with `pyinstaller packaging/tilevis_player.spec` and copy the folder on a USB stick. |
| **Package import collision** | Both `pygame` and `pygame-ce` installed | Run `tilevis check`. If collision is detected: `pip uninstall -y pygame pygame-ce && pip install pygame-ce`. |

---

## 5. Development & Contribution

- Implementation Guide: [`agents.md`](agents.md)
- Technical Architecture: [`technical_document.md`](technical_document.md)
- Engineering Log: [`changelog.md`](changelog.md)
