# Changelog — LLM Tile Visualizer (`tilevis`)

All notable changes, phase completions, verification results, and architectural decisions for `tilevis` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), extended with per-phase requirement traceability and verification summaries as specified in [`agents.md`](agents.md) §10.

---

## Phase Roadmap & Status

| Version | Phase | Title | Status |
|---|---|---|---|
| `0.0.0` | Phase 0 | Architecture, Requirements & Specification | Completed (2026-11-08) |
| `0.1.0` | Phase 1 | Project scaffolding and tooling | Completed (2026-11-08) |
| `0.2.0` | Phase 2 | Core data contract and trace format | Planned |
| `0.3.0` | Phase 3 | Model loading, config inference, text and adapters | Planned |
| `0.4.0` | Phase 4 | Probe: hooks, builders, reduction, generation, recorder | Planned |
| `0.5.0` | Phase 5 | Renderer foundation, navigation, tiles and generic scene | Planned |
| `0.6.0` | Phase 6 | Dedicated scenes: matmul, attention, residual and prediction | Planned |
| `0.7.0` | Phase 7 | Presentation UX, widgets, nerd mode and polish | Planned |
| `1.0.0` | Phase 8 | Windows lecture-hall readiness, preflight, packaging and docs | Planned |

---

## [Unreleased]

### Added
- *(Phase 2 core contract pending.)*

---

## [0.1.0] - 2026-11-08 — Phase 1: Project Scaffolding and Tooling

**Requirements addressed:** R7, R8, R9, R10, R11

### Added
- `pyproject.toml`: Modern setuptools build specification (`requires-python = ">=3.11"`) with base runtime dependencies (`numpy>=1.26`, `pygame-ce>=2.5`), optional extras (`record`, `dev`, `build`), entry points (`tilevis = tilevis.cli:main`), and tool configurations for `ruff`, `mypy`, and `pytest`.
- Package skeleton per `agents.md` §4.2 under `src/tilevis/` (`core`, `probe`, `narration`, `render`, `assets`), with module docstrings defining responsibilities and allowed dependencies.
- `src/tilevis/core/errors.py`: Exception and warning hierarchy (`TilevisError`, `ModelLoadError`, `ConfigInferenceError`, `TraceFormatError`, `RenderError`, and `FidelityWarning`).
- `src/tilevis/core/registry.py`: Generic `Registry[K, V]` with decorator-based registration, duplicate protection, `override=True` support, iteration, and informative error messages.
- `src/tilevis/logging_setup.py`: Console logging setup with encoding safety (`errors="replace"`) for Windows consoles and `--verbose` debug support.
- `src/tilevis/cli.py`: Argparse command-line interface with lazy imports and 5 stubbed subcommands (`record`, `play`, `inspect`, `check`, `demo`), plus `--version` and `--verbose`.
- `tests/test_import_boundaries.py`: AST-based import parser and clean-subprocess verification enforcing strict module boundaries (no PyTorch in render/core/narration; no Pygame in probe; lazy imports in CLI).
- `tests/compat_matrix.py`: Compatibility matrix specification with 8 standard model configurations (`tiny_ch07`, `small_bias`, `heads_8_of_64`, `head_dim_1`, `deep_narrow`, `lora_small`, `wrapped_ckpt`, `gpt2_small_shape`).
- `tests/conftest.py`: Headless test environment (`SDL_VIDEODRIVER=dummy`, `SDL_AUDIODRIVER=dummy`, `MPLBACKEND=Agg`).
- Supporting configs: `.gitignore`, `.github/workflows/ci.yml`, `profiles/tiny_random.toml`, `profiles/example_team_model.toml`, `presenter/default.toml`, `scripts/play_demo.bat`, and `scripts/play_demo.sh`.

### Decisions & Rationale
- **Module import boundary enforcement:** Built AST visitor and isolated subprocess tests to permanently prevent accidental leakage of `torch` into the playback path or `pygame` into the probe module.
- **Console encoding protection:** Reconfigured stdout/stderr streams with `errors="replace"` to protect against `cp949` crashes on Korean Windows installations.
- **Lazy CLI subcommands:** Stubs in `cli.py` return exit code 1 with user-friendly notices without eagerly importing deep-learning or GUI dependencies.

### Verification & Test Summary
- **Automated tests:** 22 passed in 0.34s, 94% test coverage across active scaffolding modules.
- **Compatibility matrix:** All 8 matrix entries verified for structural integrity and valid parameter mapping.
- **Weakness probes:**
  - Verified import boundary checker detects deliberate violations in temporary modules.
  - Verified no conflict between `pygame` and `pygame-ce` packages.
  - Verified CLI executes in non-ASCII directory paths (`학생_모델_디렉토리`).
- **Static checks:** `ruff check` (clean), `ruff format --check` (clean), `mypy --strict` on `core`, `narration`, `render` (clean), and `mypy` on `probe` (clean).
- **Manual checks:** N/A (headless scaffolding phase).

### Known Issues & Deferred Items
- Subcommands print placeholder notices until implemented in their respective phases (Phases 3, 4, 5, 7, 8).

---

## [0.0.0] - 2026-11-08 — Phase 0: Architecture, Requirements & Specification

**Requirements established:** R1, R2, R3, R4, R5, R6, R7, R8, R9, R10, R11

### Added
- [`architecture_proposal.md`](architecture_proposal.md): initial high-level architectural proposal separating PyTorch hooking (`probe`) from tile visualization (`render`) via an immutable `Trace` data contract.
- [`agents.md`](agents.md): authoritative engineering guide, strict package-boundary rules, library selection, `Trace` schema v1 specification, model-agnostic adapter/inference design, 8-phase development plan with 6-part verification gates, and changelog maintenance rules.
- [`technical_document.md`](technical_document.md): human-oriented contributor reference covering installation modes, package dependency rules, `*.trace.npz` format, dynamic model onboarding (`profiles/*.toml`), renderer/playlist mechanics, CLI usage, and extension recipes.
- [`changelog.md`](changelog.md): phase tracking table and chronological engineering log.

### Decisions & Rationale
- **Renderer library (`pygame-ce`, R9, R10):** Confirmed `pygame` as the rendering backend, standardized on `pygame-ce >= 2.5` (`import pygame`) for active maintenance, fast Windows wheel availability on Python 3.11–3.13, and hardware-friendly 2D surface operations.
- **Record-then-play architecture (`*.trace.npz`, R2, R7, R10):** Decoupled PyTorch model recording (`tilevis.probe`) from interactive playback (`tilevis.render`) through a pickle-free NumPy/JSON archive (`allow_pickle=False`). Enables instant backward/forward stepping and allows the Windows lecture-hall PC to run playback with **only `numpy` and `pygame-ce`** (zero PyTorch dependency).
- **Dynamic multi-model compatibility (R6, R8):** Designed shape-based config inference (`vocab_size`, `emb_dim`, `context_length`, `n_layers`, `qkv_bias`), runtime `n_heads` verification, LoRA weight merging (`LinearWithLoRA` from `ch07/01_main-chapter-code/exercise_experiments.py`), and duck-typed `ArchitectureAdapter`s so any teammate's checkpoint works without hard-coded dimensions and without modifying `ch07/`.
- **Whole-layer step granularity + playlist filtering (R2, R3, R5):** One keypress advances and animates one full layer/step (skipping to the step's end state if pressed mid-animation). Playlist detail filtering (`STORY`, `STANDARD`, `DEEP` + collapsible blocks) keeps deep models concise for non-expert audiences while preserving full detail on demand.
- **Scope includes internal layers and next-word prediction (R3, R4):** Every recorded forward pass captures token/position embeddings, all transformer block transformations (LayerNorm, QKV, causal attention pattern & mix, output projection, residuals, feed-forward expand/GELU/contract), final LayerNorm, LM-head logits, and top-$k$ next-token probabilities.
- **Windows single-screen screenshare defaults (R10):** Defaulted presentation mode to High-DPI-aware `borderless` window (`NOFRAME` sized to desktop) rather than exclusive fullscreen to prevent black screenshare captures, blurry Windows DPI scaling, and `Alt+Tab` disruption, with full support for both keyboard and mouse-only navigation.

### Verification & Test Summary
- **Automated tests:** N/A (specification phase only; no application code written yet per user request).
- **Upstream integrity check:** Verified `ch07/` remains completely unmodified.

### Known Issues & Deferred Items
- Implementation begins in Phase 1 (`0.1.0`: Project scaffolding and tooling).
