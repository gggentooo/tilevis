# Changelog — LLM Tile Visualizer (`tilevis`)

All notable changes, phase completions, verification results, and architectural decisions for `tilevis` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), extended with per-phase requirement traceability and verification summaries as specified in [`agents.md`](agents.md) §10.

---

## Phase Roadmap & Status

| Version | Phase | Title | Status |
|---|---|---|---|
| `0.0.0` | Phase 0 | Architecture, Requirements & Specification | Completed (2026-11-08) |
| `0.1.0` | Phase 1 | Project scaffolding and tooling | Completed (2026-11-08) |
| `0.2.0` | Phase 2 | Core data contract and trace format | Completed (2026-11-08) |
| `0.3.0` | Phase 3 | Model loading, config inference, text and adapters | Planned |
| `0.4.0` | Phase 4 | Probe: hooks, builders, reduction, generation, recorder | Planned |
| `0.5.0` | Phase 5 | Renderer foundation, navigation, tiles and generic scene | Planned |
| `0.6.0` | Phase 6 | Dedicated scenes: matmul, attention, residual and prediction | Planned |
| `0.7.0` | Phase 7 | Presentation UX, widgets, nerd mode and polish | Planned |
| `1.0.0` | Phase 8 | Windows lecture-hall readiness, preflight, packaging and docs | Planned |

---

## [Unreleased]

### Added
- *(Phase 3 model loading, config inference, text and adapters pending.)*

---

## [0.2.0] - 2026-11-08 — Phase 2: Core Data Contract and Trace Format

**Requirements addressed:** R1, R2, R3, R4, R7, R8, R11

### Added
- `src/tilevis/core/kinds.py`: Vocabulary enumerations (`StepKind`, `Semantic`, `DetailLevel`, `Stage`, `TensorRole`) implemented as `StrEnum`s with tolerant `parse_kind()` fallback for forward compatibility.
- `src/tilevis/core/trace.py`: Core frozen slotted dataclasses (`Trace`, `PassRecord`, `Step`, `TensorSnapshot`, `AxisInfo`, `TensorStats`, `Prediction`, `Fidelity`, `ModelSpecInfo`) with read-only array protection (`arr.setflags(write=False)`) and query helpers (`Trace.step_by_id`, `PassRecord.steps_for_block`, `Step.by_role`).
- `src/tilevis/core/validation.py`: `validate_trace(trace, strict=False)` verifying schema versions, numeric finiteness, axis alignment, label correspondence, step uniqueness, single terminal `NEXT_TOKEN` positioning, and descending prediction probabilities.
- `src/tilevis/core/config.py`: `PresenterSettings` and `RecordSettings` dataclasses, safe TOML loader `load_toml()`, and `warn_unknown_keys()`.
- `src/tilevis/core/serialization.py`: Atomic, compressed NumPy/JSON archive serializer (`save_trace`) and loader (`load_trace`) with `allow_pickle=False`, UTF-8 JSON manifest extraction, and sequential `MIGRATIONS` execution.
- `tests/fixtures/fake_traces.py`: Torch-free synthetic trace generator `make_fake_trace()` producing valid, reproducible traces of any model architecture shape.
- Comprehensive test suites under `tests/core/`:
  - `test_trace_dataclasses.py`: Dataclass immutability, read-only NumPy array enforcement, docstring examples, query helpers.
  - `test_config.py`: Default configurations, TOML loading, and unknown key warnings.
  - `test_validation.py`: Invariant validation rules with positive and negative test cases.
  - `test_serialization.py`: Lossless roundtrip, Unicode/Korean path and tokens handling, corrupted zip rejection, missing array/manifest handling, pickle rejection, atomic write rollback, unknown step kind tolerance.
  - `test_property_serialization.py`: Hypothesis property-based testing across randomized valid traces.
  - `test_compat_roundtrip.py`: Lossless roundtrip across all 8 `compat_matrix` architecture shapes and large trace budget verification (<50MB, <2s load).

### Decisions & Rationale
- **Atomic File Writing:** Writes to a temporary `.trace.npz.tmp.<pid>.npz` archive before atomic rename (`Path.replace`), ensuring interrupted or failed writes never corrupt existing traces on disk.
- **Read-Only NumPy Enforcements:** All snapshot arrays have their flags set to `write=False` upon dataclass initialization, preventing accidental mutation during playback.
- **Strict Pickle Rejection:** Pre-verifies archive entries using `allow_pickle=False` to reject untrusted or malicious NumPy pickled objects when opening traces.
- **Forward-Compatible Kinds:** String-based enums with `parse_kind()` ensure newer traces with novel layer kinds load without crashing, falling back to generic handling.

### Verification & Test Summary
- **Automated tests:** 66 passed in 1.32s, 92% coverage across active modules (`core` modules achieve 90%–100% coverage).
- **Compatibility matrix:** All 8 architecture shapes (`tiny_ch07`, `small_bias`, `heads_8_of_64`, `head_dim_1`, `deep_narrow`, `lora_small`, `wrapped_ckpt`, `gpt2_small_shape`) round-trip losslessly.
- **Performance budgets:** Fake trace shaped like `gpt2_small_shape` (12 layers, 3 passes, 8 tokens, 64 features) achieves 0.28 MB archive size (well below the 50 MB budget) and loads/validates in 0.05s (well below the 2.0s budget).
- **Weakness probes:**
  - Hypothesis property fuzzing verifies random valid traces roundtrip identically.
  - Corrupted archives, missing manifests, and missing arrays raise `TraceFormatError`, never unhandled internal exceptions.
  - Untrusted pickle objects inside archives are rejected with `TraceFormatError`.
  - Mid-write IOErrors leave existing trace files intact.
  - Newer schema versions trigger clear update prompts.
  - Unknown step kinds load safely without errors.
  - Korean tokens, emoji, newline characters, and non-ASCII directory paths round-trip losslessly.
- **Static checks:** `ruff check` (clean), `ruff format --check` (clean), `mypy --strict` on `core`, `narration`, `render` (clean), and `mypy` on `probe` (clean).

### Known Issues & Deferred Items
- Probe hook implementation and live PyTorch model recording deferred to Phase 4.

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
